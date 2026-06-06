"""Hopper FP8xFP4 MegaMoE correctness and benchmark harness.

Mirrors :file:`test_mega_moe_hopper.py` but exercises the FP4-weight path:

* Activations: FP8 (E4M3) with per-128 K float SFA — same as the SM90 FP8 path.
* Weights:     packed FP4 (E2M1) with per-32 K UE8M0 SFB. Each storage byte
               holds 2 nibbles (low nibble = even K, high nibble = odd K).
* Kernel:      ``sm90_fp8_fp4_mega_moe_impl`` (decode-to-SMEM SS-mode WGMMA).
               The per-32 SFB is folded into the FP4 → E4M3 dequant via
               ``fp4x4_to_scaled_e4m3x4_humming`` (Plan C).

The reference runs in FP32 by dequantizing the same packed FP4 weights and
their UE8M0 SFs, so any numerical disagreement should be at WGMMA accumulator
precision plus the exact-FP32-vs-pow2-promote difference (small).

Default mode runs the layered correctness suite. ``--bench`` switches to the
FP4 fused-vs-FP8 low-latency benchmark that used to live in a separate script.
"""

import argparse
import json
import math
import os
import random
import sys
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major
from test_mega_moe_hopper import (
    BASELINE_L2_ACT_SF_GRAN,
    _bench_cuda_events,
    _import_deep_ep,
    _make_deep_ep_buffer,
    _make_deep_ep_low_latency_buffer,
    _quantize_grouped_fp8_block_128_128,
    swiglu_apply_weight_to_fp8_triton,
    swiglu_masked_post_quant_to_fp8,
)


SM90_FP4_KERNEL_NAME = "sm90_fp8_fp4_mega_moe_impl"
SM90_FP8_KERNEL_NAME = "sm90_fp8_mega_moe_impl"


# ----------------------------------------------------------------------------
# Quantization helpers
# ----------------------------------------------------------------------------

def _quantize_grouped_fp4_per32(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row, per-32 K FP4 quantization with UE8M0 SFB.

    Args
    ----
    w : (G, N, K) bf16, with K % 32 == 0 and K % 2 == 0.

    Returns
    -------
    fp4 : (G, N, K // 2) torch.int8 — packed E2M1; low nibble = even K.
    sf  : (G, N, K // 32) torch.float32 — UE8M0-rounded scale (= amax / 6
          rounded up to nearest power of two), the same dtype the SM100 FP4
          path expects pre-``transform_sf_into_required_layout``.
    """
    g, n, k = w.shape
    assert k % 32 == 0
    fp4 = torch.empty((g, n, k // 2), device=w.device, dtype=torch.int8)
    sf = torch.empty((g, n, k // 32), device=w.device, dtype=torch.float)
    for i in range(g):
        fp4[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
    return fp4, sf


def _dequant_fp4_per32(fp4: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_quantize_grouped_fp4_per32`. Returns (G, N, K) fp32."""
    g, n, half_k = fp4.shape
    k = half_k * 2
    pb = fp4.to(torch.uint8)
    lo = (pb & 0x0F).to(torch.int)
    hi = ((pb >> 4) & 0x0F).to(torch.int)
    codes = torch.stack([lo, hi], dim=-1).reshape(g, n, k)  # (G, N, K) int
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float, device=fp4.device,
    )
    sign = (codes & 0x08) != 0
    mag = (codes & 0x07).to(torch.long)
    val = table[mag]
    val = torch.where(sign & (mag != 0), -val, val)
    # Broadcast SF: each 32 K-cols share one SF.
    if os.getenv('DSV4_FP4_REFERENCE_CLAMP_SF_MIN_2M6', '0') == '1':
        sf = sf.clamp_min(2.0 ** -6)
    sf_broad = sf.unsqueeze(-1).expand(g, n, sf.size(-1), 32).reshape(g, n, k)
    return val * sf_broad


def _dequant_per_token_per_128_k(
    x_fp8: torch.Tensor, sf: torch.Tensor
) -> torch.Tensor:
    """For (M, K) fp8 with (M, K // 128) float SF (per-token, K-major)."""
    m, k = x_fp8.shape
    assert k % 128 == 0
    x_view = x_fp8.float().view(m, k // 128, 128)
    return (x_view * sf.unsqueeze(-1)).view(m, k)


def _load_dsv4_checkpoint_layer_weights(
    model_path: str,
    layer_idx: int,
    rank_idx: int,
    num_ranks: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    model_dir = Path(model_path)
    weight_map = json.loads((model_dir / 'model.safetensors.index.json').read_text())[
        'weight_map'
    ]
    num_experts = 256
    experts_per_rank = num_experts // num_ranks
    start = rank_idx * experts_per_rank
    end = start + experts_per_rank
    cache: Dict[str, Any] = {}

    def get_tensor(name: str) -> torch.Tensor:
        file_name = weight_map[name]
        if file_name not in cache:
            cache[file_name] = safe_open(str(model_dir / file_name),
                                         framework='pt', device='cpu')
        return cache[file_name].get_tensor(name)

    l1_w, l1_s, l2_w, l2_s = [], [], [], []
    for expert_id in range(start, end):
        prefix = f'layers.{layer_idx}.ffn.experts.{expert_id}'
        w1 = get_tensor(f'{prefix}.w1.weight')
        w3 = get_tensor(f'{prefix}.w3.weight')
        s1 = get_tensor(f'{prefix}.w1.scale')
        s3 = get_tensor(f'{prefix}.w3.scale')
        l1_w.append(torch.cat([w1, w3], dim=0))
        l1_s.append(torch.cat([s1.float(), s3.float()], dim=0))
        l2_w.append(get_tensor(f'{prefix}.w2.weight'))
        l2_s.append(get_tensor(f'{prefix}.w2.scale').float())

    return (
        torch.stack(l1_w, dim=0).cuda(),
        torch.stack(l1_s, dim=0).cuda(),
        torch.stack(l2_w, dim=0).cuda(),
        torch.stack(l2_s, dim=0).cuda(),
    )


def _m_grouped_fp8_gemm_nt_masked(*args, **kwargs):
    fn = (
        getattr(deep_gemm, "m_grouped_fp8_gemm_nt_masked", None)
        or getattr(deep_gemm, "fp8_m_grouped_gemm_nt_masked", None)
        or getattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked", None)
    )
    if fn is None:
        raise AttributeError("no masked grouped FP8 GEMM API is exported by deep_gemm")
    return fn(*args, **kwargs)


def _safe_div(a: float, b: float) -> float:
    return float("nan") if b == 0 else a / b


def _all_rank_metrics(values: Tuple[float, ...]) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64, device="cuda")
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered).cpu()


def _flush_l2_if_requested(l2_flush_gb: float):
    if l2_flush_gb <= 0:
        return
    free_bytes, _ = torch.cuda.mem_get_info()
    flush_bytes = min(int(l2_flush_gb * 1e9), int(free_bytes * 0.5))
    if flush_bytes >= 4:
        torch.empty(flush_bytes // 4, dtype=torch.int, device="cuda").zero_()


def _bench_cuda_event_sections(
    sections,
    num_warmup: int = 3,
    num_repeat: int = 10,
    l2_flush_gb: float = 0.0,
    barrier=None,
):
    for _ in range(num_warmup):
        for _, fn in sections:
            fn()
    torch.cuda.synchronize()

    section_times_ms = {name: [] for name, _ in sections}
    total_times_ms = []
    for _ in range(num_repeat):
        if barrier is not None:
            barrier()
        _flush_l2_if_requested(l2_flush_gb)
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        events = []
        total_start.record()
        for name, fn in sections:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((name, start, end))
        total_end.record()
        total_end.synchronize()
        total_times_ms.append(total_start.elapsed_time(total_end))
        for name, start, end in events:
            section_times_ms[name].append(start.elapsed_time(end))

    def median_sec(values_ms):
        values_ms = sorted(values_ms)
        return values_ms[len(values_ms) // 2] / 1e3

    section_times = {
        name: median_sec(values) for name, values in section_times_ms.items()
    }
    return section_times, median_sec(total_times_ms)


# ----------------------------------------------------------------------------
# PyTorch reference
# ----------------------------------------------------------------------------

def _swiglu_fp32(gate_up: torch.Tensor, clamp: float) -> torch.Tensor:
    """SwiGLU: silu(min(gate, c)) * clamp(up, -c, c)."""
    n2 = gate_up.size(-1)
    half = n2 // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    if math.isfinite(clamp):
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return torch.nn.functional.silu(gate) * up


def _reference_fused(
    x_fp8_local: torch.Tensor, x_sf_local: torch.Tensor,
    topk_idx_local: torch.Tensor, topk_weights_local: torch.Tensor,
    l1_w_fp4: torch.Tensor, l1_w_sf: torch.Tensor,
    l2_w_fp4: torch.Tensor, l2_w_sf: torch.Tensor,
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    num_experts: int, num_topk: int,
    hidden: int, intermediate_hidden: int,
    activation_clamp: float,
) -> torch.Tensor:
    """FP32 reference for the SM90 FP4 mega-MoE kernel.

    All-gathers tokens / topk decisions / per-rank weights, then for each
    global token routes through its topk experts, applies the L1+SwiGLU+L2
    path, and reduces over topk on the source rank.
    """
    num_experts_per_rank = num_experts // num_ranks

    x_fp8_g = uneven_all_gather(x_fp8_local, group=group)
    x_sf_g = uneven_all_gather(x_sf_local, group=group)
    topk_idx_g = uneven_all_gather(topk_idx_local, group=group)
    topk_w_g = uneven_all_gather(topk_weights_local, group=group)
    mg = x_fp8_g.size(0)

    local_size = torch.tensor([x_fp8_local.size(0)], device='cuda', dtype=torch.long)
    sizes_t = torch.empty(num_ranks, dtype=torch.long, device='cuda')
    dist.all_gather_into_tensor(sizes_t, local_size, group=group)
    sizes_list = sizes_t.tolist()

    l1_w_g = [torch.empty_like(l1_w_fp4) for _ in range(num_ranks)]
    l1_sf_g = [torch.empty_like(l1_w_sf) for _ in range(num_ranks)]
    l2_w_g = [torch.empty_like(l2_w_fp4) for _ in range(num_ranks)]
    l2_sf_g = [torch.empty_like(l2_w_sf) for _ in range(num_ranks)]
    dist.all_gather(l1_w_g, l1_w_fp4, group=group)
    dist.all_gather(l1_sf_g, l1_w_sf, group=group)
    dist.all_gather(l2_w_g, l2_w_fp4, group=group)
    dist.all_gather(l2_sf_g, l2_w_sf, group=group)
    l1_w_all = torch.stack(l1_w_g, dim=0)   # (R, E_pr, 2*IH, H/2)
    l1_sf_all = torch.stack(l1_sf_g, dim=0)
    l2_w_all = torch.stack(l2_w_g, dim=0)
    l2_sf_all = torch.stack(l2_sf_g, dim=0)

    combine_buf = torch.zeros(
        mg, num_topk, hidden, dtype=torch.float32, device='cuda')

    x_fp32 = _dequant_per_token_per_128_k(x_fp8_g, x_sf_g)  # (Mg, H)

    # Token-chunked dequant to bound peak memory of the per-token gather.
    _CHUNK = int(os.getenv('DSV4_FP4_REFERENCE_CHUNK', '64'))
    for k in range(num_topk):
        mask = topk_idx_g[:, k] >= 0
        if not mask.any():
            continue
        sel_idx_full = mask.nonzero(as_tuple=False).squeeze(-1)
        for c0 in range(0, sel_idx_full.numel(), _CHUNK):
            sel_idx = sel_idx_full[c0:c0 + _CHUNK]
            eids = topk_idx_g[sel_idx, k]
            weights = topk_w_g[sel_idx, k]
            x_sel = x_fp32[sel_idx]                              # (S, H)

            dst_rank = (eids // num_experts_per_rank).long()
            dst_local = (eids % num_experts_per_rank).long()

            # L1 GEMM
            l1_w_sel = _dequant_fp4_per32(
                l1_w_all[dst_rank, dst_local],                   # (S, 2*IH, H/2) int8
                l1_sf_all[dst_rank, dst_local],                  # (S, 2*IH, H/32)
            )                                                    # (S, 2*IH, H) fp32
            l1_y = torch.einsum('sk,snk->sn', x_sel, l1_w_sel)
            del l1_w_sel

            l1_y = _swiglu_fp32(l1_y, activation_clamp) * weights.unsqueeze(-1)

            s_, ih = l1_y.shape
            assert ih == intermediate_hidden and ih % 64 == 0
            l1_view = l1_y.view(s_, ih // 64, 64)
            amax = l1_view.abs().amax(dim=-1).clamp(1e-4)
            sf2 = amax / 448.0
            l1_q = (l1_view / sf2.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
            l2_in = (l1_q * sf2.unsqueeze(-1)).view(s_, ih)

            l2_w_sel = _dequant_fp4_per32(
                l2_w_all[dst_rank, dst_local],                   # (S, H, IH/2) int8
                l2_sf_all[dst_rank, dst_local],                  # (S, H, IH/32)
            )                                                    # (S, H, IH) fp32
            l2_y = torch.einsum('sn,smn->sm', l2_in, l2_w_sel)
            del l2_w_sel

            combine_buf[sel_idx, k] = l2_y.to(torch.bfloat16).float()

    y_full_bf16 = combine_buf.to(torch.bfloat16).sum(dim=1).to(torch.bfloat16)
    start = sum(sizes_list[:rank_idx])
    end = start + sizes_list[rank_idx]
    return y_full_bf16[start:end].contiguous()


# ----------------------------------------------------------------------------
# Single-scenario runner
# ----------------------------------------------------------------------------

def _run_scenario(
    name: str,
    cfg: Dict[str, Any],
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    diff_tol: float,
):
    num_max = cfg['num_max_tokens_per_rank']
    num_tokens = cfg.get('num_tokens', num_max)
    hidden = cfg['hidden']
    intermediate_hidden = cfg['intermediate_hidden']
    num_experts = cfg['num_experts']
    num_topk = cfg['num_topk']
    masked_ratio = cfg.get('masked_ratio', 0.0)
    activation_clamp = cfg.get('activation_clamp', 10.0)
    fast_math = cfg.get('fast_math', True)

    assert num_experts % num_ranks == 0
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max
    # Hard kernel constraints.
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64, (
        f'SM90 fused kernel requires intermediate_hidden <= 4096, '
        f'got {intermediate_hidden}'
    )

    torch.manual_seed(rank_idx * 1000 + abs(hash(name)) % 1000)
    random.seed(rank_idx * 1000 + abs(hash(name)) % 1000)

    # ---- Inputs (bf16) ------------------------------------------------------
    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < masked_ratio, -1)
        topk_w.masked_fill_(topk_idx < 0, 0)

    # FP8 activations with per-128 K float SF (SM90 format) — same as SM90 FP8 path.
    x_fp8, x_sf = per_token_cast_to_fp8(
        x_bf, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)

    if cfg.get('checkpoint_model_path'):
        l1_w_fp4, l1_w_sf, l2_w_fp4, l2_w_sf = _load_dsv4_checkpoint_layer_weights(
            cfg['checkpoint_model_path'],
            cfg.get('checkpoint_layer_idx', 0),
            rank_idx,
            num_ranks,
        )
    else:
        l1_bf = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden),
            dtype=torch.bfloat16, device='cuda') * 0.05
        l2_bf = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden),
            dtype=torch.bfloat16, device='cuda') * 0.05
        # FP4 weights with per-32 K UE8M0 SF — DSV4 standard.
        l1_w_fp4, l1_w_sf = _quantize_grouped_fp4_per32(l1_bf)
        l2_w_fp4, l2_w_sf = _quantize_grouped_fp4_per32(l2_bf)

    # SM90 FP4 weight transform: gate/up gran-8 N interleave + UE8M0 SFB k-major
    # packing into uint32. Both pieces are needed — see
    # ``transform_weights_for_mega_moe_sm90_fp4`` for layout details.
    transformed_l1, transformed_l2 = (
        deep_gemm.transform_weights_for_mega_moe_sm90_fp4(
            (l1_w_fp4, l1_w_sf), (l2_w_fp4, l2_w_sf)
        )
    )

    # ---- Allocate symm buffer ----------------------------------------------
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max, num_topk,
        hidden, intermediate_hidden,
    )
    cum_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')

    # ---- Run fused ----------------------------------------------------------
    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_w)

    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_fp4_mega_moe(
        y_fused, transformed_l1, transformed_l2, buffer,
        cumulative_local_expert_recv_stats=cum_stats,
        recipe=(1, 1, 32),
        activation='swiglu',
        activation_clamp=activation_clamp if math.isfinite(activation_clamp) else None,
        fast_math=fast_math,
    )
    torch.cuda.synchronize()

    # ---- Reference & check --------------------------------------------------
    y_ref = _reference_fused(
        x_fp8, x_sf, topk_idx, topk_w,
        l1_w_fp4, l1_w_sf, l2_w_fp4, l2_w_sf,
        rank_idx, num_ranks, group,
        num_experts, num_topk,
        hidden, intermediate_hidden,
        activation_clamp,
    )

    diff = calc_diff(y_fused, y_ref)
    ok = diff < diff_tol
    dist_print(f'  [{name:<32}] diff={diff:.4f} '
               f'(tol={diff_tol:.2f}) {"OK" if ok else "FAIL"}',
               once_in_node=True)
    if not ok:
        for label, tensor in (('fused', y_fused), ('ref', y_ref)):
            tensor_f = tensor.float()
            dist_print(
                f'    {label}: abs_max={tensor_f.abs().max().item():.6g} '
                f'mean={tensor_f.mean().item():.6g} '
                f'nonzero={(tensor_f != 0).sum().item()}/{tensor_f.numel()} '
                f'finite={torch.isfinite(tensor_f).all().item()}',
                once_in_node=True,
            )
    assert ok, f'{name}: diff={diff} >= tol={diff_tol}'

    buffer.destroy()
    dist.barrier()


# ----------------------------------------------------------------------------
# Scenario tables (smaller than the SM90 FP8 test — FP4 reference is heavier)
# ----------------------------------------------------------------------------

_SMOKE = dict(
    num_max_tokens_per_rank=64, num_tokens=64,
    hidden=512, intermediate_hidden=512,
    num_experts=8, num_topk=2,
)


def _layer1_smoke() -> List[Tuple[str, Dict[str, Any]]]:
    return [('L1.smoke', dict(_SMOKE))]


def _layer3_shape_sweep(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    base_experts = 8 * num_ranks
    out = []
    for hidden in (512, 1024):
        for ih in (512, 1024):
            for topk in (1, 2):
                cfg = dict(num_max_tokens_per_rank=128, num_tokens=128,
                           hidden=hidden, intermediate_hidden=ih,
                           num_experts=base_experts, num_topk=topk)
                out.append((f'L3.h{hidden}_ih{ih}_k{topk}', cfg))
    return out


def _layer4_edges(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    base = dict(num_max_tokens_per_rank=128,
                hidden=512, intermediate_hidden=512,
                num_experts=8 * num_ranks, num_topk=2)
    out = []
    for mr in (0.0, 0.5):
        cfg = dict(base); cfg.update(num_tokens=128, masked_ratio=mr)
        out.append((f'L4.mask{mr:.1f}', cfg))
    for c in (1.0, math.inf):
        cfg = dict(base); cfg.update(num_tokens=128, activation_clamp=c)
        out.append((f'L4.clamp{c}', cfg))
    cfg = dict(base); cfg.update(num_tokens=0)
    out.append(('L4.tokens0', cfg))
    return out


def _layer5_dsv4_shape(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    assert num_ranks == 8, 'DSV4 shape test expects 8 ranks'
    return [('L5.dsv4_h4096_ih2048_e256_k6', dict(
        num_max_tokens_per_rank=128, num_tokens=64,
        hidden=4096, intermediate_hidden=2048,
        num_experts=256, num_topk=6,
        activation_clamp=10.0,
    ))]


def _layer6_dsv4_checkpoint(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    assert num_ranks == 8, 'DSV4 checkpoint test expects 8 ranks'
    model_path = os.getenv('DSV4_FP4_MODEL_PATH', '/data00/models/DeepSeek-V4-Flash')
    return [('L6.dsv4_ckpt_layer0_h4096_ih2048_e256_k6', dict(
        num_max_tokens_per_rank=128, num_tokens=64,
        hidden=4096, intermediate_hidden=2048,
        num_experts=256, num_topk=6,
        activation_clamp=10.0,
        checkpoint_model_path=model_path,
        checkpoint_layer_idx=0,
    ))]


def _layer7_dsv4_2wg(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    # 2-WG split-MN accuracy guard. num_tokens=512 drives
    # expected_tokens_per_expert = 512 * num_topk / experts_per_rank
    #                            = 512 * 6 / (256 / 8) = 96 >= 64,
    # so get_block_config_for_mega_moe_sm90 takes the auto_split_mn branch:
    # block_m=128 with TWO epilogue warpgroups. This is the only accuracy
    # scenario that exercises the 2-WG path; L1/L3/L4/L5 are all 1-WG
    # (num_tokens<=128 -> expected<64). It guards the default that turns the
    # math warpgroup's FP4 decode OFF on the 2-WG path (decode is offloaded
    # to the assist warps and written to the shared decoded-B smem tile, so the
    # numerics must be identical to the math-on path).
    assert num_ranks == 8, 'DSV4 2-WG shape test expects 8 ranks'
    return [('L7.dsv4_2wg_nt512_h4096_ih2048_e256_k6', dict(
        num_max_tokens_per_rank=512, num_tokens=512,
        hidden=4096, intermediate_hidden=2048,
        num_experts=256, num_topk=6,
        activation_clamp=10.0,
    ))]


def _layer8_pro_smoke(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ('L8.pro_b128_1wg_h7168_ih3072_e384_k6', dict(
            num_max_tokens_per_rank=128, num_tokens=128,
            hidden=7168, intermediate_hidden=3072,
            num_experts=384, num_topk=6,
            activation_clamp=10.0,
        )),
        ('L8.pro_b512_2wg_h7168_ih3072_e384_k6', dict(
            num_max_tokens_per_rank=512, num_tokens=512,
            hidden=7168, intermediate_hidden=3072,
            num_experts=384, num_topk=6,
            activation_clamp=10.0,
        )),
    ]


# ----------------------------------------------------------------------------
# Benchmark mode
# ----------------------------------------------------------------------------

def _run_benchmark(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if get_arch_major() != 9:
        dist_print(
            f"[SKIP] test_mega_moe_fp4_hopper --bench requires SM90; "
            f"got SM{get_arch_major()}0",
            once_in_node=True,
        )
        dist.destroy_process_group()
        return

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens if args.num_tokens else num_max_tokens_per_rank
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    run_fp8_normal_baseline_enabled = (
        args.run_normal_baseline and not args.ncu_profile_only
    )
    run_fp8_ll_baseline_enabled = (
        not args.skip_fp8_ll_baseline and not args.ncu_profile_only
    )

    assert num_tokens <= num_max_tokens_per_rank
    assert num_experts % num_ranks == 0
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64
    if args.fp4_mode == "predecode-fp8-ll" and not run_fp8_ll_baseline_enabled:
        raise ValueError(
            "--fp4-mode predecode-fp8-ll requires the FP8 low-latency baseline"
        )

    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    l1_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * args.weight_scale
    l2_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * args.weight_scale

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    topk_idx_ll = topk_idx.to(torch.int64)

    x_fp8, x_sf = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )

    l1_fp4 = _quantize_grouped_fp4_per32(l1_bf16)
    l2_fp4 = _quantize_grouped_fp4_per32(l2_bf16)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90_fp4(
        l1_fp4, l2_fp4
    )

    predecoded_l1 = predecoded_l2 = None
    predecoded_ll_l1 = predecoded_ll_l2 = None
    if args.fp4_mode == "predecode-fp8-fused":
        predecoded_l1, predecoded_l2 = (
            deep_gemm.transform_weights_for_mega_moe_sm90_fp4_predecoded(
                l1_fp4, l2_fp4
            )
        )
    elif args.fp4_mode == "predecode-fp8-ll":
        l1_fp4_data, l1_fp4_sf = l1_fp4
        l2_fp4_data, l2_fp4_sf = l2_fp4
        l1_predecode_fp8 = _dequant_fp4_per32(
            l1_fp4_data, l1_fp4_sf
        ).to(torch.float8_e4m3fn)
        l2_predecode_fp8 = _dequant_fp4_per32(
            l2_fp4_data, l2_fp4_sf
        ).to(torch.float8_e4m3fn)
        l1_predecode_sf = torch.ones(
            (num_experts_per_rank, (intermediate_hidden * 2) // 128, hidden // 128),
            dtype=torch.float32,
            device="cuda",
        )
        l2_predecode_sf = torch.ones(
            (num_experts_per_rank, hidden // 128, intermediate_hidden // 128),
            dtype=torch.float32,
            device="cuda",
        )
        predecoded_ll_l1 = (l1_predecode_fp8.contiguous(), l1_predecode_sf)
        predecoded_ll_l2 = (l2_predecode_fp8.contiguous(), l2_predecode_sf)

    l1_fp8 = _quantize_grouped_fp8_block_128_128(l1_bf16)
    l2_fp8 = _quantize_grouped_fp8_block_128_128(l2_bf16)

    clamp_arg = args.activation_clamp if math.isfinite(args.activation_clamp) else None
    cum_stats = torch.zeros((num_experts_per_rank,), dtype=torch.int, device="cuda")
    sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def fp4_prepare_inputs():
        sym_buffer.x[:num_tokens].copy_(x_fp8)
        sym_buffer.x_sf[:num_tokens].copy_(x_sf)
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

    def fp4_fused_kernel():
        if args.fp4_mode == "predecode-fp8-fused":
            active_l1, active_l2 = predecoded_l1, predecoded_l2
        else:
            active_l1, active_l2 = transformed_l1, transformed_l2
        deep_gemm.fp8_fp4_mega_moe(
            y_fused,
            active_l1,
            active_l2,
            sym_buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=clamp_arg,
            fast_math=bool(args.fast_math),
        )
        return y_fused

    def run_fp4_fused():
        fp4_prepare_inputs()
        return fp4_fused_kernel()

    if args.ncu_profile_only:
        dist_print(
            f"[NCU] FP4 Hopper tokens={num_tokens} hidden={hidden} "
            f"ih={intermediate_hidden}",
            once_in_node=True,
        )
        run_fp4_fused()
        torch.cuda.synchronize()
        dist.barrier()
        sym_buffer.destroy()
        dist.destroy_process_group()
        return

    normal_buffer = None
    ll_buffer = None
    if run_fp8_normal_baseline_enabled or run_fp8_ll_baseline_enabled:
        deep_ep = _import_deep_ep()
        if deep_ep is None:
            raise RuntimeError("deep_ep is required for baseline comparisons")

    if run_fp8_normal_baseline_enabled:
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        normal_buffer = _make_deep_ep_buffer(
            deep_ep,
            group,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            sym_buffer.buffer.nbytes,
        )
        normal_cum_stats = torch.zeros(
            (num_experts_per_rank,), dtype=torch.int, device="cuda"
        )
        normal_state = {}

    def fp8_normal_dispatch():
        recv_x, _, recv_topk_weights, handle, _ = normal_buffer.dispatch(
            (x_fp8, x_sf),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=normal_cum_stats,
            num_experts=num_experts,
            expert_alignment=alignment,
            do_cpu_sync=False,
            do_handle_copy=False,
            do_expand=True,
            use_tma_aligned_col_major_sf=False,
        )
        normal_state["recv_x"] = recv_x
        normal_state["recv_topk_weights"] = recv_topk_weights
        normal_state["handle"] = handle

    def fp8_normal_l1_gemm():
        recv_x = normal_state["recv_x"]
        handle = normal_state["handle"]
        l1_y = torch.empty(
            (recv_x[0].size(0), intermediate_hidden * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            recv_x,
            l1_fp8,
            l1_y,
            handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            disable_ue8m0_cast=True,
        )
        normal_state["l1_y"] = l1_y

    def fp8_normal_swiglu_quant():
        normal_state["l1_act"] = swiglu_apply_weight_to_fp8_triton(
            x=normal_state["l1_y"],
            topk_weights=normal_state["recv_topk_weights"],
            clamp_value=clamp_arg,
            num_per_channels=BASELINE_L2_ACT_SF_GRAN,
            use_ue8m0_scale=True,
        )

    def fp8_normal_l2_gemm():
        handle = normal_state["handle"]
        l2_y = torch.empty(
            (normal_state["l1_act"][0].size(0), hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            normal_state["l1_act"],
            l2_fp8,
            l2_y,
            handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            disable_ue8m0_cast=True,
        )
        normal_state["l2_y"] = l2_y

    def fp8_normal_combine():
        combined = normal_buffer.combine(
            normal_state["l2_y"],
            handle=normal_state["handle"],
        )[0]
        normal_state["combined"] = combined
        return combined

    def run_fp8_normal_baseline():
        fp8_normal_dispatch()
        fp8_normal_l1_gemm()
        fp8_normal_swiglu_quant()
        fp8_normal_l2_gemm()
        return fp8_normal_combine()

    if run_fp8_ll_baseline_enabled:

        ll_buffer = _make_deep_ep_low_latency_buffer(
            deep_ep, group, num_max_tokens_per_rank, hidden, num_experts
        )
        m_max_ll = num_max_tokens_per_rank * num_ranks
        expected_m_ll = max(
            1,
            (num_max_tokens_per_rank * num_ranks * num_topk + num_experts - 1)
            // num_experts,
        )
        ll_l1_y = torch.empty(
            (num_experts_per_rank, m_max_ll, intermediate_hidden * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        ll_l2_y = torch.empty(
            (num_experts_per_rank, m_max_ll, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        ll_combined = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        ll_state = {}

    def fp8_ll_dispatch():
        (recv_x_data, recv_x_sf), masked_m, ll_handle, event, hook = (
            ll_buffer.low_latency_dispatch(
                x_bf16,
                topk_idx_ll,
                num_max_tokens_per_rank,
                num_experts,
                use_fp8=True,
                round_scale=False,
                use_ue8m0=False,
                async_finish=False,
                return_recv_hook=False,
            )
        )
        ll_state["recv"] = (recv_x_data, recv_x_sf)
        ll_state["masked_m"] = masked_m
        ll_state["ll_handle"] = ll_handle

    def fp8_ll_l1_gemm():
        _m_grouped_fp8_gemm_nt_masked(
            ll_state["recv"],
            l1_fp8,
            ll_l1_y,
            ll_state["masked_m"],
            expected_m_ll,
            disable_ue8m0_cast=True,
        )

    def fp4_predecode_ll_l1_gemm():
        _m_grouped_fp8_gemm_nt_masked(
            ll_state["recv"],
            predecoded_ll_l1,
            ll_l1_y,
            ll_state["masked_m"],
            expected_m_ll,
            disable_ue8m0_cast=True,
        )

    def fp8_ll_swiglu_quant():
        l1_act_fp8, l1_act_sf = swiglu_masked_post_quant_to_fp8(
            ll_l1_y,
            ll_state["masked_m"],
            quant_group_size=BASELINE_L2_ACT_SF_GRAN,
            clamp_value=clamp_arg,
            use_ue8m0_scale=False,
        )
        ll_state["l1_act"] = (l1_act_fp8, l1_act_sf)

    def fp8_ll_l2_gemm():
        _m_grouped_fp8_gemm_nt_masked(
            ll_state["l1_act"],
            l2_fp8,
            ll_l2_y,
            ll_state["masked_m"],
            expected_m_ll,
            disable_ue8m0_cast=True,
        )

    def fp4_predecode_ll_l2_gemm():
        _m_grouped_fp8_gemm_nt_masked(
            ll_state["l1_act"],
            predecoded_ll_l2,
            ll_l2_y,
            ll_state["masked_m"],
            expected_m_ll,
            disable_ue8m0_cast=True,
        )

    def fp8_ll_combine():
        combined, event, hook = ll_buffer.low_latency_combine(
            ll_l2_y,
            topk_idx_ll,
            topk_weights,
            ll_state["ll_handle"],
            use_logfmt=False,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
            out=ll_combined,
        )
        ll_state["combined"] = combined
        return combined

    def run_fp8_low_latency_baseline():
        fp8_ll_dispatch()
        fp8_ll_l1_gemm()
        fp8_ll_swiglu_quant()
        fp8_ll_l2_gemm()
        return fp8_ll_combine()

    def run_fp4_predecode_low_latency():
        fp8_ll_dispatch()
        fp4_predecode_ll_l1_gemm()
        fp8_ll_swiglu_quant()
        fp4_predecode_ll_l2_gemm()
        return fp8_ll_combine()

    fused_out = (
        run_fp4_predecode_low_latency()
        if args.fp4_mode == "predecode-fp8-ll"
        else run_fp4_fused()
    )
    assert fused_out.shape == (num_tokens, hidden)
    if args.check_predecode_runtime_match:
        if args.fp4_mode != "predecode-fp8-fused":
            raise ValueError(
                "--check-predecode-runtime-match is only valid with "
                "--fp4-mode predecode-fp8-fused"
            )
        y_runtime = torch.empty_like(y_fused)
        fp4_prepare_inputs()
        cum_stats.zero_()
        deep_gemm.fp8_fp4_mega_moe(
            y_runtime,
            transformed_l1,
            transformed_l2,
            sym_buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=clamp_arg,
            fast_math=bool(args.fast_math),
        )
        torch.cuda.synchronize()
        runtime_diff = calc_diff(fused_out, y_runtime)
        ok = runtime_diff < args.check_diff_tol
        dist_print(
            f"  [predecode_vs_runtime] diff={runtime_diff:.4f} "
            f"(tol={args.check_diff_tol:.2f}) {'OK' if ok else 'FAIL'}",
            once_in_node=True,
        )
        if not ok:
            raise AssertionError(
                f"predecoded FP4 path differs from runtime FP4 path: "
                f"diff={runtime_diff:.6f}"
            )
    if run_fp8_normal_baseline_enabled:
        normal_out = run_fp8_normal_baseline()
        assert normal_out.shape == (num_tokens, hidden)
    if run_fp8_ll_baseline_enabled:
        ll_out = run_fp8_low_latency_baseline()
        assert ll_out.shape == (num_tokens, hidden)
    torch.cuda.synchronize()

    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[
        (gathered_topk_idx < rank_idx * num_experts_per_rank)
        | (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)
    ] = -1
    local_expert_ids = gathered_topk_idx[gathered_topk_idx != -1]
    num_recv_tokens = int(local_expert_ids.numel())
    num_touched_experts = int(torch.unique(local_expert_ids).numel())

    if args.profile_breakdown:
        if args.fp4_mode == "predecode-fp8-ll":
            fp4_sections = [
                ("fp4_ll_dispatch", fp8_ll_dispatch),
                ("fp4_ll_l1_gemm", fp4_predecode_ll_l1_gemm),
                ("fp4_ll_swiglu_quant", fp8_ll_swiglu_quant),
                ("fp4_ll_l2_gemm", fp4_predecode_ll_l2_gemm),
                ("fp4_ll_combine", fp8_ll_combine),
            ]
        else:
            fp4_sections = [
                ("fp4_prepare_inputs", fp4_prepare_inputs),
                ("fp4_fused_kernel", fp4_fused_kernel),
            ]
        dist.barrier()
        fp4_profile, fp4_profile_total = _bench_cuda_event_sections(
            fp4_sections,
            num_warmup=args.profile_warmup,
            num_repeat=args.profile_repeat,
            l2_flush_gb=args.profile_l2_flush_gb,
            barrier=dist.barrier,
        )
        profile_names = ["fp4_total", *[name for name, _ in fp4_sections]]
        profile_values = [
            fp4_profile_total,
            *[fp4_profile[name] for name, _ in fp4_sections],
        ]
        if run_fp8_ll_baseline_enabled:
            fp8_ll_sections = [
                ("fp8_ll_dispatch", fp8_ll_dispatch),
                ("fp8_ll_l1_gemm", fp8_ll_l1_gemm),
                ("fp8_ll_swiglu_quant", fp8_ll_swiglu_quant),
                ("fp8_ll_l2_gemm", fp8_ll_l2_gemm),
                ("fp8_ll_combine", fp8_ll_combine),
            ]
            dist.barrier()
            fp8_profile, fp8_profile_total = _bench_cuda_event_sections(
                fp8_ll_sections,
                num_warmup=args.profile_warmup,
                num_repeat=args.profile_repeat,
                l2_flush_gb=args.profile_l2_flush_gb,
                barrier=dist.barrier,
            )
            profile_names += ["fp8_ll_total", *[name for name, _ in fp8_ll_sections]]
            profile_values += [
                fp8_profile_total,
                *[fp8_profile[name] for name, _ in fp8_ll_sections],
            ]
        profile_metrics = _all_rank_metrics(tuple(profile_values))
        profile_count_metrics = _all_rank_metrics(
            (float(num_recv_tokens), float(num_touched_experts))
        )
        if rank_idx == 0:
            profile_result = {
                "batch_per_rank": num_tokens,
                "num_ranks": num_ranks,
                "num_experts": num_experts,
                "num_topk": num_topk,
                "recv_tokens_total": int(profile_count_metrics[:, 0].sum().item()),
                "active_experts_max": int(profile_count_metrics[:, 1].max().item()),
                "profile_repeat": args.profile_repeat,
                "profile_warmup": args.profile_warmup,
                "profile_l2_flush_gb": args.profile_l2_flush_gb,
                "fp8_ll_baseline_enabled": run_fp8_ll_baseline_enabled,
            }
            for i, name in enumerate(profile_names):
                profile_result[f"{name}_us_max"] = round(
                    float(profile_metrics[:, i].max().item() * 1e6), 3
                )
                profile_result[f"{name}_us_mean"] = round(
                    float(profile_metrics[:, i].mean().item() * 1e6), 3
                )
            print("PROFILE_JSON " + json.dumps(profile_result, sort_keys=True), flush=True)
        dist.barrier()

    if args.fp4_mode == "predecode-fp8-ll":
        fused_timing_method = "cuda_events_low_latency_pipeline"
        t_fused = _bench_cuda_events(
            run_fp4_predecode_low_latency,
            num_warmup=args.num_warmup,
            num_repeat=args.num_repeat,
            l2_flush_gb=args.l2_flush_gb,
        )
    else:
        fp4_kernel_name = (
            SM90_FP8_KERNEL_NAME
            if args.fp4_mode == "predecode-fp8-fused"
            else SM90_FP4_KERNEL_NAME
        )
        t_fused = bench_kineto(
            run_fp4_fused,
            fp4_kernel_name,
            num_tests=args.num_bench_tests,
            barrier=dist.barrier,
            flush_l2=bool(args.kineto_flush_l2),
        )
        kineto_ok = torch.tensor([1 if t_fused > 0 else 0], dtype=torch.int, device="cuda")
        dist.all_reduce(kineto_ok, op=dist.ReduceOp.MIN)
        fused_timing_method = "kineto_kernel"
        if kineto_ok.item() == 0:
            fused_timing_method = "cuda_events_fallback"
            t_fused = _bench_cuda_events(
                run_fp4_fused,
                num_warmup=args.num_warmup,
                num_repeat=args.num_repeat,
                l2_flush_gb=args.l2_flush_gb,
            )

    t_ll = float("nan")
    t_normal = float("nan")
    if run_fp8_normal_baseline_enabled:
        dist.barrier()
        t_normal = _bench_cuda_events(
            run_fp8_normal_baseline,
            num_warmup=args.num_warmup,
            num_repeat=args.num_repeat,
            l2_flush_gb=args.l2_flush_gb,
        )
        dist.barrier()

    if run_fp8_ll_baseline_enabled:
        dist.barrier()
        t_ll = _bench_cuda_events(
            run_fp8_low_latency_baseline,
            num_warmup=args.num_warmup,
            num_repeat=args.num_repeat,
            l2_flush_gb=args.l2_flush_gb,
        )
        dist.barrier()

    metrics = _all_rank_metrics(
        (
            t_fused,
            t_normal,
            t_ll,
            float(num_recv_tokens),
            float(num_touched_experts),
        )
    )
    if rank_idx == 0:
        fused_us_max = float(metrics[:, 0].max().item() * 1e6)
        fused_us_mean = float(metrics[:, 0].mean().item() * 1e6)
        normal_us_max = None
        normal_us_mean = None
        speedup_vs_fp8_normal_max = None
        if run_fp8_normal_baseline_enabled:
            normal_us_max = float(metrics[:, 1].max().item() * 1e6)
            normal_us_mean = float(metrics[:, 1].mean().item() * 1e6)
            speedup_vs_fp8_normal_max = round(
                _safe_div(normal_us_max, fused_us_max), 4
            )
        ll_us_max = None
        ll_us_mean = None
        speedup_vs_fp8_ll_max = None
        if run_fp8_ll_baseline_enabled:
            ll_us_max = float(metrics[:, 2].max().item() * 1e6)
            ll_us_mean = float(metrics[:, 2].mean().item() * 1e6)
            speedup_vs_fp8_ll_max = round(_safe_div(ll_us_max, fused_us_max), 4)
        result = {
            "batch_per_rank": num_tokens,
            "num_ranks": num_ranks,
            "hidden": hidden,
            "intermediate_hidden": intermediate_hidden,
            "num_experts": num_experts,
            "num_topk": num_topk,
            "recv_tokens_total": int(metrics[:, 3].sum().item()),
            "active_experts_max": int(metrics[:, 4].max().item()),
            "fp4_megamoe_us_max": round(fused_us_max, 3),
            "fp4_megamoe_us_mean": round(fused_us_mean, 3),
            "fp4_mode": args.fp4_mode,
            "fp4_timing_method": fused_timing_method,
            "fp8_normal_baseline_enabled": run_fp8_normal_baseline_enabled,
            "fp8_normal_baseline_us_max": (
                None if normal_us_max is None else round(normal_us_max, 3)
            ),
            "fp8_normal_baseline_us_mean": (
                None if normal_us_mean is None else round(normal_us_mean, 3)
            ),
            "speedup_vs_fp8_normal_max": speedup_vs_fp8_normal_max,
            "fp8_ll_baseline_enabled": run_fp8_ll_baseline_enabled,
            "fp8_ll_baseline_us_max": None if ll_us_max is None else round(ll_us_max, 3),
            "fp8_ll_baseline_us_mean": None if ll_us_mean is None else round(ll_us_mean, 3),
            "speedup_vs_fp8_ll_max": speedup_vs_fp8_ll_max,
            "num_bench_tests": args.num_bench_tests,
            "num_warmup": args.num_warmup,
            "num_repeat": args.num_repeat,
            "l2_flush_gb": args.l2_flush_gb,
            "kineto_flush_l2": bool(args.kineto_flush_l2),
        }
        print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)

    dist.barrier()
    sym_buffer.destroy()
    if normal_buffer is not None:
        normal_buffer.destroy()
    if ll_buffer is not None:
        ll_buffer.destroy()
    dist.destroy_process_group()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    if args.bench:
        _run_benchmark(local_rank, num_local_ranks, args)
        return

    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    if get_arch_major() != 9:
        dist_print(
            f'[SKIP] test_mega_moe_fp4_hopper requires SM90; '
            f'got SM{get_arch_major()}0',
            once_in_node=True)
        dist.destroy_process_group()
        return

    diff_tol = args.diff_tol
    layers: List[Tuple[str, Dict[str, Any]]] = []
    if 1 in args.layers:
        layers += _layer1_smoke()
    if 3 in args.layers:
        layers += _layer3_shape_sweep(num_ranks)
    if 4 in args.layers:
        layers += _layer4_edges(num_ranks)
    if 5 in args.layers:
        layers += _layer5_dsv4_shape(num_ranks)
    if 6 in args.layers:
        layers += _layer6_dsv4_checkpoint(num_ranks)
    if 7 in args.layers:
        layers += _layer7_dsv4_2wg(num_ranks)
    if 8 in args.layers or args.pro_smoke:
        layers += _layer8_pro_smoke(num_ranks)

    if args.filter:
        layers = [(n, c) for n, c in layers if args.filter in n]

    dist_print(f'SM90 FP4 MegaMoE test plan: {len(layers)} scenarios across '
               f'layers {sorted(args.layers)} on {num_ranks} ranks',
               once_in_node=True)

    failures: List[str] = []
    for name, cfg in layers:
        try:
            _run_scenario(name, cfg, rank_idx, num_ranks, group, diff_tol)
        except AssertionError as ex:
            dist_print(f'  [{name}] FAIL: {ex}', once_in_node=True)
            failures.append(name)
            if args.fail_fast:
                break

    dist_print('', once_in_node=True)
    if failures:
        dist_print(f'FAILED {len(failures)}/{len(layers)} scenarios: {failures}',
                   once_in_node=True)
    else:
        dist_print(f'PASSED all {len(layers)} scenarios', once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hopper FP4 MegaMoE tests and benchmark')
    parser.add_argument('--bench', action='store_true',
                        help='Run FP4 fused vs FP8 low-latency benchmark mode')
    parser.add_argument('--ncu-profile-only', action='store_true',
                        help='With --bench, run one FP4 fused kernel launch for NCU')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--local-rank-idx', type=int, default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 3, 4],
                        help='Correctness layers to run (1, 3, 4, 5, 6, 7, 8). '
                             'Default: 1 3 4. Layer 8 is the Pro smoke shape.')
    parser.add_argument('--pro-smoke', action='store_true',
                        help='Also run DeepSeek-V4-Pro smoke scenarios')
    parser.add_argument('--filter', type=str, default='')
    parser.add_argument('--diff-tol', type=float, default=0.10,
                        help='calc_diff tolerance (default 0.10; FP4 weights '
                             'introduce more quantization noise than FP8).')
    parser.add_argument('--fail-fast', action='store_true')

    parser.add_argument('--num-max-tokens-per-rank', type=int, default=1)
    parser.add_argument('--num-tokens', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=4096)
    parser.add_argument('--intermediate-hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--weight-scale', type=float, default=0.05)
    parser.add_argument(
        '--fp4-mode',
        choices=('runtime', 'predecode-fp8-fused', 'predecode-fp8-ll'),
        default='runtime',
        help='Benchmark mode: runtime FP4 kernel, or predecoded FP8 controls',
    )
    parser.add_argument('--num-bench-tests', type=int, default=20)
    parser.add_argument('--num-warmup', type=int, default=5)
    parser.add_argument('--num-repeat', type=int, default=20)
    parser.add_argument('--l2-flush-gb', type=float, default=0.0)
    parser.add_argument('--profile-breakdown', action='store_true',
                        help='Emit PROFILE_JSON with CUDA-event stage timings')
    parser.add_argument('--skip-fp8-ll-baseline', action='store_true',
                        help='Only measure the FP4 fused path')
    parser.add_argument('--run-normal-baseline', action='store_true',
                        help='Also measure the normal DeepEP dispatch/combine FP8 baseline')
    parser.add_argument('--profile-warmup', type=int, default=3)
    parser.add_argument('--profile-repeat', type=int, default=10)
    parser.add_argument('--profile-l2-flush-gb', type=float, default=0.0)
    parser.add_argument('--kineto-flush-l2', type=int, default=0)
    parser.add_argument('--check-predecode-runtime-match', action='store_true',
                        help='With --bench --fp4-mode predecode-fp8-fused, '
                             'compare predecoded API output against runtime FP4')
    parser.add_argument('--check-diff-tol', type=float, default=0.02)
    args = parser.parse_args()

    np_ = args.num_processes
    if args.local_rank_idx is not None:
        test(args.local_rank_idx, np_, args)
    else:
        torch.multiprocessing.spawn(test, args=(np_, args), nprocs=np_)
