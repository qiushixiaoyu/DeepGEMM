"""Smoke + correctness tests for the SM90 (Hopper) FP8×FP4 MegaMoE kernel.

Mirrors :file:`test_mega_moe_sm90.py` but exercises the FP4-weight path:

* Activations: FP8 (E4M3) with per-128 K float SFA — same as the SM90 FP8 path.
* Weights:     packed FP4 (E2M1) with per-32 K UE8M0 SFB. Each storage byte
               holds 2 nibbles (low nibble = even K, high nibble = odd K).
* Kernel:      ``sm90_fp8_fp4_mega_moe_impl`` (decode-to-SMEM SS-mode WGMMA).
               The per-32 SFB is folded into the FP4 → E4M3 dequant via
               ``fp4x4_to_scaled_e4m3x4_humming`` (Plan C).

The reference runs in FP32 by dequantizing the same packed FP4 weights and
their UE8M0 SFs, so any numerical disagreement should be at WGMMA accumulator
precision plus the exact-FP32-vs-pow2-promote difference (small).
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
from deep_gemm.testing import calc_diff, get_arch_major


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


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    if get_arch_major() != 9:
        dist_print(
            f'[SKIP] test_mega_moe_sm90_fp4 requires SM90; '
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
    parser = argparse.ArgumentParser(description='SM90 FP4 MegaMoE tests')
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 3, 4],
                        help='Which layers to run (1, 3, 4, 5, 6, 7). Default: 1 3 4. '
                             'Layer 7 is the 2-WG (num_tokens=512) accuracy guard.')
    parser.add_argument('--filter', type=str, default='')
    parser.add_argument('--diff-tol', type=float, default=0.10,
                        help='calc_diff tolerance (default 0.10; FP4 weights '
                             'introduce more quantization noise than FP8).')
    parser.add_argument('--fail-fast', action='store_true')
    args = parser.parse_args()

    np_ = args.num_processes
    torch.multiprocessing.spawn(test, args=(np_, args), nprocs=np_)
