"""Benchmark SM90 FP4 MegaMoE against the FP8 low-latency MoE baseline.

The runtime fused side calls ``deep_gemm.fp8_fp4_mega_moe`` with FP4 expert
weights.  The baseline side mirrors sglang's DeepEP low-latency decode pipeline:
``low_latency_dispatch(use_fp8=True) -> masked grouped FP8 GEMM -> masked
SwiGLU+FP8 quant -> masked grouped FP8 GEMM -> low_latency_combine``.
``--fp4-mode predecode-fp8-ll`` keeps that low-latency pipeline but feeds it
FP4-derived weights that were predecoded to FP8 before the timed region.
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
from test_mega_moe_sm90_fp4 import _dequant_fp4_per32


SM90_FP4_KERNEL_NAME = "sm90_fp8_fp4_mega_moe_impl"
SM90_FP8_KERNEL_NAME = "sm90_fp8_mega_moe_impl"


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
    """Return median per-section and total timings in seconds.

    The sections are measured in one end-to-end pass so dependent pipeline
    stages, e.g. dispatch -> GEMM -> combine, see realistic dataflow.
    """
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

    section_times = {name: median_sec(values) for name, values in section_times_ms.items()}
    return section_times, median_sec(total_times_ms)


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
    run_fp8_ll_baseline_enabled = not args.skip_fp8_ll_baseline

    assert num_tokens <= num_max_tokens_per_rank
    assert num_experts % num_ranks == 0
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64
    if args.fp4_mode == "predecode-fp8-ll" and not run_fp8_ll_baseline_enabled:
        raise ValueError("--skip-fp8-ll-baseline is incompatible with --fp4-mode predecode-fp8-ll")

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

    predecoded_l1 = predecoded_l2 = None
    predecoded_ll_l1 = predecoded_ll_l2 = None
    if args.fp4_mode in ("predecode-fp8-fused", "predecode-fp8-ll"):
        l1_fp4_data, l1_fp4_sf = l1_fp4
        l2_fp4_data, l2_fp4_sf = l2_fp4
        l1_predecode_fp8 = _dequant_fp4_per32(l1_fp4_data, l1_fp4_sf).to(torch.float8_e4m3fn)
        l2_predecode_fp8 = _dequant_fp4_per32(l2_fp4_data, l2_fp4_sf).to(torch.float8_e4m3fn)
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
        if args.fp4_mode == "predecode-fp8-fused":
            predecoded_l1, predecoded_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
                predecoded_ll_l1,
                predecoded_ll_l2,
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

    def fp4_prepare_inputs():
        sym_buffer.x[:num_tokens].copy_(x_fp8)
        sym_buffer.x_sf[:num_tokens].copy_(x_sf)
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

    fp4_clock_profile = None
    if args.fp4_clock_profile:
        if args.fp4_mode != "runtime":
            raise ValueError("--fp4-clock-profile only applies to --fp4-mode runtime")
        fp4_clock_profile = torch.zeros((16,), dtype=torch.int64, device="cuda")

    def fp4_fused_kernel(clock_profile=None):
        if args.fp4_mode == "predecode-fp8-fused":
            deep_gemm.fp8_mega_moe(
                y_fused,
                predecoded_l1,
                predecoded_l2,
                sym_buffer,
                cumulative_local_expert_recv_stats=cum_stats,
                recipe=(128, 128, 128),
                activation="swiglu",
                activation_clamp=clamp_arg,
                fast_math=bool(args.fast_math),
            )
        else:
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
                fp4_clock_profile=clock_profile,
            )
        return y_fused

    def run_fp4_fused(clock_profile=None):
        fp4_prepare_inputs()
        return fp4_fused_kernel(clock_profile)

    ll_buffer = None
    if run_fp8_ll_baseline_enabled:
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

    # Smoke both paths before measuring.
    fused_out = (
        run_fp4_predecode_low_latency()
        if args.fp4_mode == "predecode-fp8-ll"
        else run_fp4_fused()
    )
    assert fused_out.shape == (num_tokens, hidden)
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

    if fp4_clock_profile is not None:
        fp4_clock_profile.zero_()
        dist.barrier()
        run_fp4_fused(fp4_clock_profile)
        torch.cuda.synchronize()
        dist.all_reduce(fp4_clock_profile, op=dist.ReduceOp.SUM)
        if rank_idx == 0:
            values = [int(v) for v in fp4_clock_profile.cpu().tolist()]
            clock_khz = float(torch.cuda.get_device_properties(0).clock_rate)

            def cycles_per_block(slot_base: int, rel_slot: int) -> float:
                count = values[slot_base]
                return float("nan") if count == 0 else values[slot_base + rel_slot] / count

            def us_per_block(slot_base: int, rel_slot: int) -> float:
                cycles = cycles_per_block(slot_base, rel_slot)
                return float("nan") if math.isnan(cycles) else cycles * 1000.0 / clock_khz

            clock_result = {
                "batch_per_rank": num_tokens,
                "num_ranks": num_ranks,
                "clock_rate_khz": int(clock_khz),
                "l1_profiled_k_blocks": values[0],
                "l1_full_wait_cycles_per_block": round(cycles_per_block(0, 1), 3),
                "l1_decode_sync_cycles_per_block": round(cycles_per_block(0, 2), 3),
                "l1_wait_decode_cycles_per_block": round(cycles_per_block(0, 1) + cycles_per_block(0, 2), 3),
                "l1_wgmma_cycles_per_block": round(cycles_per_block(0, 3), 3),
                "l1_promote_cycles_per_block": round(cycles_per_block(0, 4), 3),
                "l1_full_wait_us_per_block": round(us_per_block(0, 1), 6),
                "l1_decode_sync_us_per_block": round(us_per_block(0, 2), 6),
                "l1_wait_decode_us_per_block": round(us_per_block(0, 1) + us_per_block(0, 2), 6),
                "l1_wgmma_us_per_block": round(us_per_block(0, 3), 6),
                "l1_promote_us_per_block": round(us_per_block(0, 4), 6),
                "l2_profiled_k_blocks": values[8],
                "l2_full_wait_cycles_per_block": round(cycles_per_block(8, 1), 3),
                "l2_decode_sync_cycles_per_block": round(cycles_per_block(8, 2), 3),
                "l2_wait_decode_cycles_per_block": round(cycles_per_block(8, 1) + cycles_per_block(8, 2), 3),
                "l2_wgmma_cycles_per_block": round(cycles_per_block(8, 3), 3),
                "l2_promote_cycles_per_block": round(cycles_per_block(8, 4), 3),
                "l2_full_wait_us_per_block": round(us_per_block(8, 1), 6),
                "l2_decode_sync_us_per_block": round(us_per_block(8, 2), 6),
                "l2_wait_decode_us_per_block": round(us_per_block(8, 1) + us_per_block(8, 2), 6),
                "l2_wgmma_us_per_block": round(us_per_block(8, 3), 6),
                "l2_promote_us_per_block": round(us_per_block(8, 4), 6),
                "raw_slots": values,
            }
            print("CLOCK_PROFILE_JSON " + json.dumps(clock_result, sort_keys=True), flush=True)
        dist.barrier()

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
        profile_values = [fp4_profile_total, *[fp4_profile[name] for name, _ in fp4_sections]]
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
            profile_values += [fp8_profile_total, *[fp8_profile[name] for name, _ in fp8_ll_sections]]
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
                profile_result[f"{name}_us_max"] = round(float(profile_metrics[:, i].max().item() * 1e6), 3)
                profile_result[f"{name}_us_mean"] = round(float(profile_metrics[:, i].mean().item() * 1e6), 3)
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
        (t_fused, t_ll, float(num_recv_tokens), float(num_touched_experts))
    )
    if rank_idx == 0:
        fused_us_max = float(metrics[:, 0].max().item() * 1e6)
        fused_us_mean = float(metrics[:, 0].mean().item() * 1e6)
        ll_us_max = None
        ll_us_mean = None
        speedup_vs_fp8_ll_max = None
        if run_fp8_ll_baseline_enabled:
            ll_us_max = float(metrics[:, 1].max().item() * 1e6)
            ll_us_mean = float(metrics[:, 1].mean().item() * 1e6)
            speedup_vs_fp8_ll_max = round(_safe_div(ll_us_max, fused_us_max), 4)
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
            "fp4_mode": args.fp4_mode,
            "fp4_timing_method": fused_timing_method,
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
    if ll_buffer is not None:
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
    parser.add_argument(
        "--fp4-mode",
        choices=("runtime", "predecode-fp8-fused", "predecode-fp8-ll"),
        default="runtime",
        help=(
            "runtime runs the SM90 FP8xFP4 kernel; predecode-fp8-fused decodes "
            "FP4 weights to E4M3 outside the timed region and reuses fp8_mega_moe; "
            "predecode-fp8-ll feeds FP4-derived E4M3 weights into the DeepEP "
            "low-latency grouped-GEMM pipeline"
        ),
    )
    parser.add_argument("--num-bench-tests", type=int, default=20)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-repeat", type=int, default=20)
    parser.add_argument("--l2-flush-gb", type=float, default=0.0)
    parser.add_argument(
        "--profile-breakdown",
        action="store_true",
        help="Emit PROFILE_JSON with CUDA-event timings for FP4 and FP8 LL stages",
    )
    parser.add_argument(
        "--fp4-clock-profile",
        action="store_true",
        help="Emit CLOCK_PROFILE_JSON with in-kernel FP4 wait/decode, WGMMA, and promote cycle counters",
    )
    parser.add_argument(
        "--skip-fp8-ll-baseline",
        action="store_true",
        help="Only measure the FP4 fused path; useful when DeepEP low-latency transport is unavailable",
    )
    parser.add_argument("--profile-warmup", type=int, default=3)
    parser.add_argument("--profile-repeat", type=int, default=10)
    parser.add_argument(
        "--profile-l2-flush-gb",
        type=float,
        default=0.0,
        help="Optional L2 flush size for each profile-breakdown iteration",
    )
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
