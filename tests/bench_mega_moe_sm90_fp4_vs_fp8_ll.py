"""Benchmark SM90 FP4 MegaMoE against the FP8 low-latency MoE baseline.

The fused side calls ``deep_gemm.fp8_fp4_mega_moe`` with FP4 expert weights.
The baseline side mirrors sglang's DeepEP low-latency decode pipeline:
``low_latency_dispatch(use_fp8=True) -> masked grouped FP8 GEMM -> masked
SwiGLU+FP8 quant -> masked grouped FP8 GEMM -> low_latency_combine``.
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Tuple

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.testing import bench_kineto, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist, uneven_all_gather
from test_mega_moe_hopper import (
    BASELINE_L2_ACT_SF_GRAN,
    _bench_cuda_events,
    _import_deep_ep,
    _make_deep_ep_low_latency_buffer,
    _quantize_grouped_fp8_block_128_128,
    swiglu_masked_post_quant_to_fp8,
)
from test_mega_moe_sm90_fp4 import _quantize_grouped_fp4_per32


SM90_FP4_KERNEL_NAME = "sm90_fp8_fp4_mega_moe_impl"


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


def benchmark(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(20260524 + rank_idx)
    random.seed(20260524 + rank_idx)

    if get_arch_major() != 9:
        if rank_idx == 0:
            print(f"[SKIP] requires SM90; got SM{get_arch_major()}0", flush=True)
        dist.destroy_process_group()
        return

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens if args.num_tokens else num_max_tokens_per_rank
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_experts_per_rank = num_experts // num_ranks

    assert num_tokens <= num_max_tokens_per_rank
    assert num_experts % num_ranks == 0
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64

    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    topk_idx_ll = topk_idx.to(torch.int64)

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

    x_fp8, x_sf = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )

    l1_fp4 = _quantize_grouped_fp4_per32(l1_bf16)
    l2_fp4 = _quantize_grouped_fp4_per32(l2_bf16)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90_fp4(
        l1_fp4, l2_fp4
    )

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

    def run_fp4_fused():
        sym_buffer.x[:num_tokens].copy_(x_fp8)
        sym_buffer.x_sf[:num_tokens].copy_(x_sf)
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)
        deep_gemm.fp8_fp4_mega_moe(
            y_fused,
            transformed_l1,
            transformed_l2,
            sym_buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=clamp_arg,
            fast_math=bool(args.fast_math),
        )
        return y_fused

    deep_ep = _import_deep_ep()
    if deep_ep is None:
        raise RuntimeError("deep_ep is required for the FP8 low-latency baseline")

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

    def run_fp8_low_latency_baseline():
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
        _m_grouped_fp8_gemm_nt_masked(
            (recv_x_data, recv_x_sf),
            l1_fp8,
            ll_l1_y,
            masked_m,
            expected_m_ll,
            disable_ue8m0_cast=True,
        )
        l1_act_fp8, l1_act_sf = swiglu_masked_post_quant_to_fp8(
            ll_l1_y,
            masked_m,
            quant_group_size=BASELINE_L2_ACT_SF_GRAN,
            clamp_value=clamp_arg,
            use_ue8m0_scale=False,
        )
        _m_grouped_fp8_gemm_nt_masked(
            (l1_act_fp8, l1_act_sf),
            l2_fp8,
            ll_l2_y,
            masked_m,
            expected_m_ll,
            disable_ue8m0_cast=True,
        )
        combined, event, hook = ll_buffer.low_latency_combine(
            ll_l2_y,
            topk_idx_ll,
            topk_weights,
            ll_handle,
            use_logfmt=False,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
            out=ll_combined,
        )
        return combined

    # Smoke both paths before measuring.
    fused_out = run_fp4_fused()
    ll_out = run_fp8_low_latency_baseline()
    assert fused_out.shape == (num_tokens, hidden)
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

    t_fused = bench_kineto(
        run_fp4_fused,
        SM90_FP4_KERNEL_NAME,
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

    dist.barrier()
    t_ll = _bench_cuda_events(
        run_fp8_low_latency_baseline,
        num_warmup=args.num_warmup,
        num_repeat=args.num_repeat,
        l2_flush_gb=args.l2_flush_gb,
    )
    dist.barrier()

    metrics = _all_rank_metrics(
        (t_fused, t_ll, float(num_recv_tokens), float(num_touched_experts))
    )
    if rank_idx == 0:
        fused_us_max = float(metrics[:, 0].max().item() * 1e6)
        fused_us_mean = float(metrics[:, 0].mean().item() * 1e6)
        ll_us_max = float(metrics[:, 1].max().item() * 1e6)
        ll_us_mean = float(metrics[:, 1].mean().item() * 1e6)
        result = {
            "batch_per_rank": num_tokens,
            "num_ranks": num_ranks,
            "hidden": hidden,
            "intermediate_hidden": intermediate_hidden,
            "num_experts": num_experts,
            "num_topk": num_topk,
            "recv_tokens_total": int(metrics[:, 2].sum().item()),
            "active_experts_max": int(metrics[:, 3].max().item()),
            "fp4_megamoe_us_max": round(fused_us_max, 3),
            "fp4_megamoe_us_mean": round(fused_us_mean, 3),
            "fp4_timing_method": fused_timing_method,
            "fp8_ll_baseline_us_max": round(ll_us_max, 3),
            "fp8_ll_baseline_us_mean": round(ll_us_mean, 3),
            "speedup_vs_fp8_ll_max": round(_safe_div(ll_us_max, fused_us_max), 4),
            "num_bench_tests": args.num_bench_tests,
            "num_warmup": args.num_warmup,
            "num_repeat": args.num_repeat,
            "l2_flush_gb": args.l2_flush_gb,
            "kineto_flush_l2": bool(args.kineto_flush_l2),
        }
        print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)

    dist.barrier()
    sym_buffer.destroy()
    ll_buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SM90 FP4 MegaMoE vs FP8 DeepEP low-latency baseline"
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--local-rank-idx", type=int, default=None)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--num-bench-tests", type=int, default=20)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-repeat", type=int, default=20)
    parser.add_argument("--l2-flush-gb", type=float, default=0.0)
    parser.add_argument(
        "--kineto-flush-l2",
        type=int,
        default=0,
        help="Whether bench_kineto flushes 8GB L2 before each fused-kernel run",
    )
    args = parser.parse_args()

    if args.local_rank_idx is not None:
        benchmark(args.local_rank_idx, args.num_processes, args)
    else:
        torch.multiprocessing.spawn(
            benchmark, args=(args.num_processes, args), nprocs=args.num_processes
        )
