"""Profile the normal FP8 DeepEP + grouped-GEMM MegaMoE baseline.

This diagnostic script measures the non-fused FP8 normal pipeline sections:
dispatch -> L1 grouped GEMM -> SwiGLU+FP8 quant -> L2 grouped GEMM -> combine.
It is intentionally separate from the FP4 megakernel benchmark so Phase 0 can
estimate how much serial communication the normal baseline exposes.
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Iterable, Tuple

import torch
import torch.distributed as dist


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
TESTS_ROOT = os.path.join(REPO_ROOT, "tests")
if TESTS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_ROOT)

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist, uneven_all_gather
from test_mega_moe_hopper import (
    BASELINE_L2_ACT_SF_GRAN,
    _import_deep_ep,
    _make_deep_ep_buffer,
    _quantize_grouped_fp8_block_128_128,
    swiglu_apply_weight_to_fp8_triton,
)


def _all_rank_metrics(values: Tuple[float, ...]) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64, device="cuda")
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered).cpu()


def _parse_batches(text: str) -> Iterable[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def _flush_l2_if_requested(l2_flush_gb: float) -> None:
    if l2_flush_gb <= 0:
        return
    free_bytes, _ = torch.cuda.mem_get_info()
    flush_bytes = min(int(l2_flush_gb * 1e9), int(free_bytes * 0.5))
    if flush_bytes >= 4:
        torch.empty(flush_bytes // 4, dtype=torch.int, device="cuda").zero_()


def _bench_cuda_event_sections(
    sections,
    num_warmup: int,
    num_repeat: int,
    l2_flush_gb: float,
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

    return (
        {name: median_sec(values) for name, values in section_times_ms.items()},
        median_sec(total_times_ms),
    )


def benchmark(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(23000 + rank_idx)
    random.seed(23000 + rank_idx)

    if get_arch_major() != 9:
        if rank_idx == 0:
            print(f"[SKIP] requires SM90; got SM{get_arch_major()}0", flush=True)
        dist.destroy_process_group()
        return

    deep_ep = _import_deep_ep()
    if deep_ep is None:
        raise RuntimeError("deep_ep is required for the normal FP8 baseline")

    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_experts % num_ranks == 0
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64

    batches = list(_parse_batches(args.batches))
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
    l1_fp8 = _quantize_grouped_fp8_block_128_128(l1_bf16)
    l2_fp8 = _quantize_grouped_fp8_block_128_128(l2_bf16)

    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    clamp_arg = args.activation_clamp if math.isfinite(args.activation_clamp) else None

    if rank_idx == 0:
        meta = {
            "batches": batches,
            "comparison": "fp8 normal pipeline section breakdown",
            "hidden": hidden,
            "intermediate_hidden": intermediate_hidden,
            "num_experts": num_experts,
            "num_ranks": num_ranks,
            "num_topk": num_topk,
            "profile_l2_flush_gb": args.profile_l2_flush_gb,
            "profile_repeat": args.profile_repeat,
            "profile_warmup": args.profile_warmup,
            "weight_scale": args.weight_scale,
        }
        print("PROFILE_META_JSON " + json.dumps(meta, sort_keys=True), flush=True)

    for batch in batches:
        sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            batch,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        ep_buffer = _make_deep_ep_buffer(
            deep_ep,
            group,
            batch,
            hidden,
            num_topk,
            sym_buffer.buffer.nbytes,
        )
        cum_stats = torch.zeros((num_experts_per_rank,), dtype=torch.int, device="cuda")

        torch.manual_seed(24000 + rank_idx * 100000 + batch)
        x_bf16 = torch.randn((batch, hidden), dtype=torch.bfloat16, device="cuda")
        scores = torch.randn((batch, num_experts), dtype=torch.float, device="cuda")
        topk_weights, topk_idx = torch.topk(
            scores, num_topk, dim=-1, largest=True, sorted=False
        )
        x_fp8 = per_token_cast_to_fp8(
            x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
        )
        state = {}

        def fp8_dispatch():
            recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                x_fp8,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                cumulative_local_expert_recv_stats=cum_stats,
                num_experts=num_experts,
                expert_alignment=alignment,
                do_cpu_sync=False,
                do_handle_copy=False,
                do_expand=True,
                use_tma_aligned_col_major_sf=False,
            )
            state["recv_x"] = recv_x
            state["recv_topk_weights"] = recv_topk_weights
            state["handle"] = handle
            return recv_x

        def fp8_l1_gemm():
            recv_x = state["recv_x"]
            handle = state["handle"]
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
            state["l1_y"] = l1_y
            return l1_y

        def fp8_swiglu_quant():
            l1_y_fp8 = swiglu_apply_weight_to_fp8_triton(
                x=state["l1_y"],
                topk_weights=state["recv_topk_weights"],
                clamp_value=clamp_arg,
                num_per_channels=BASELINE_L2_ACT_SF_GRAN,
                use_ue8m0_scale=True,
            )
            state["l1_y_fp8"] = l1_y_fp8
            return l1_y_fp8

        def fp8_l2_gemm():
            handle = state["handle"]
            l2_y = torch.empty(
                (state["l1_y_fp8"][0].size(0), hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                state["l1_y_fp8"],
                l2_fp8,
                l2_y,
                handle.psum_num_recv_tokens_per_expert,
                use_psum_layout=True,
                disable_ue8m0_cast=True,
            )
            state["l2_y"] = l2_y
            return l2_y

        def fp8_combine():
            combined = ep_buffer.combine(state["l2_y"], handle=state["handle"])[0]
            state["combined"] = combined
            return combined

        sections = [
            ("fp8_normal_dispatch", fp8_dispatch),
            ("fp8_normal_l1_gemm", fp8_l1_gemm),
            ("fp8_normal_swiglu_quant", fp8_swiglu_quant),
            ("fp8_normal_l2_gemm", fp8_l2_gemm),
            ("fp8_normal_combine", fp8_combine),
        ]
        for _, fn in sections:
            fn()
        assert state["combined"].shape == (batch, hidden)
        torch.cuda.synchronize()

        gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
        gathered_topk_idx[
            (gathered_topk_idx < rank_idx * num_experts_per_rank)
            | (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)
        ] = -1
        local_expert_ids = gathered_topk_idx[gathered_topk_idx != -1]
        num_recv_tokens = int(local_expert_ids.numel())
        num_touched_experts = int(torch.unique(local_expert_ids).numel())

        dist.barrier()
        profile, total = _bench_cuda_event_sections(
            sections,
            num_warmup=args.profile_warmup,
            num_repeat=args.profile_repeat,
            l2_flush_gb=args.profile_l2_flush_gb,
            barrier=dist.barrier,
        )
        names = ["fp8_normal_total", *[name for name, _ in sections]]
        values = [total, *[profile[name] for name, _ in sections]]
        metrics = _all_rank_metrics(tuple(values))
        count_metrics = _all_rank_metrics(
            (float(num_recv_tokens), float(num_touched_experts))
        )
        if rank_idx == 0:
            result = {
                "active_experts_max": int(count_metrics[:, 1].max().item()),
                "batch_per_rank": batch,
                "hidden": hidden,
                "intermediate_hidden": intermediate_hidden,
                "num_experts": num_experts,
                "num_ranks": num_ranks,
                "num_topk": num_topk,
                "profile_l2_flush_gb": args.profile_l2_flush_gb,
                "profile_repeat": args.profile_repeat,
                "profile_warmup": args.profile_warmup,
                "recv_tokens_total": int(count_metrics[:, 0].sum().item()),
            }
            for i, name in enumerate(names):
                result[f"{name}_us_max"] = round(float(metrics[:, i].max().item() * 1e6), 3)
                result[f"{name}_us_mean"] = round(float(metrics[:, i].mean().item() * 1e6), 3)
            print("PROFILE_JSON " + json.dumps(result, sort_keys=True), flush=True)

        dist.barrier()
        ep_buffer.destroy()
        sym_buffer.destroy()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP8 normal MegaMoE pipeline breakdown")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--local-rank-idx", type=int, default=None)
    parser.add_argument("--batches", type=str, default="512 2048")
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate-hidden", type=int, default=3072)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--profile-warmup", type=int, default=3)
    parser.add_argument("--profile-repeat", type=int, default=10)
    parser.add_argument("--profile-l2-flush-gb", type=float, default=0.0)
    args = parser.parse_args()

    if args.local_rank_idx is not None:
        benchmark(args.local_rank_idx, args.num_processes, args)
    else:
        torch.multiprocessing.spawn(
            benchmark, args=(args.num_processes, args), nprocs=args.num_processes
        )
