"""Two-node SM90 MegaMoE vs DeepEP performance sweep.

This is a benchmark driver only.  It keeps model weights and communication
buffers alive while sweeping token counts, so large model shapes do not pay
weight construction/quantization and process-group startup at every point.
"""

import argparse
import math
import random

import torch
import torch.distributed as dist

import test_mega_moe_hopper as base


def _global_stats(seconds: float) -> tuple[float, float, float]:
    value = torch.tensor([seconds], dtype=torch.float64, device="cuda")
    max_value = value.clone()
    min_value = value.clone()
    sum_value = value.clone()
    dist.all_reduce(max_value, op=dist.ReduceOp.MAX)
    dist.all_reduce(min_value, op=dist.ReduceOp.MIN)
    dist.all_reduce(sum_value, op=dist.ReduceOp.SUM)
    return (
        float(max_value.item()),
        float(sum_value.item() / dist.get_world_size()),
        float(min_value.item()),
    )


def _print_rank0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, world_size, group = base.init_dist(local_rank, num_local_ranks)
    torch.manual_seed(20260721 + rank)
    random.seed(20260721 + rank)

    if base.get_arch_major() != 9:
        raise RuntimeError(f"SM90 benchmark requires Hopper, got SM{base.get_arch_major()}0")

    hidden = args.hidden
    intermediate = args.intermediate_hidden
    num_experts = args.num_experts
    topk = args.num_topk
    batches = args.batches
    max_tokens = max(batches)
    local_experts = num_experts // world_size
    need_fused = args.path in ("both", "fused")
    need_deep_ep = args.path in ("both", "deep_ep")

    if args.mode == "normal" and args.path == "both":
        raise RuntimeError(
            "DeepEP normal and the 16-rank MegaMOE NVSHMEM team must be "
            "benchmarked in isolated processes; use --path fused/deep_ep"
        )

    assert num_experts % world_size == 0
    assert hidden % 128 == 0
    assert intermediate % 128 == 0 and intermediate <= 4096

    _print_rank0(
        rank,
        f"[CONFIG] model={args.model_name} mode={args.mode} path={args.path} world_size={world_size} "
        f"hidden={hidden} intermediate={intermediate} experts={num_experts} "
        f"topk={topk} local_experts={local_experts} batches={','.join(map(str, batches))} "
        f"warmup={args.num_warmup} repeat={args.num_repeat} "
        f"fused_tests={args.num_bench_tests} l2_flush_gb={args.l2_flush_gb}",
    )

    # Construct and quantize model weights once for the full sweep.
    l1_bf16 = torch.randn(
        (local_experts, intermediate * 2, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05
    l1_weights = base._quantize_grouped_fp8_block_128_128(l1_bf16)
    del l1_bf16
    torch.cuda.empty_cache()

    l2_bf16 = torch.randn(
        (local_experts, hidden, intermediate),
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05
    l2_weights = base._quantize_grouped_fp8_block_128_128(l2_bf16)
    del l2_bf16
    torch.cuda.empty_cache()

    transformed_l1 = transformed_l2 = None
    if need_fused:
        transformed_l1, transformed_l2 = base.deep_gemm.transform_weights_for_mega_moe_sm90(
            l1_weights, l2_weights
        )
    alignment = base.deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    base.deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    deep_ep = base._import_deep_ep() if need_deep_ep else None
    if need_deep_ep and deep_ep is None:
        raise RuntimeError("deep_ep is required for this comparison")

    ep_buffer = None
    ll_buffer = None
    if args.mode == "normal" and need_deep_ep:
        ep_buffer = base._make_deep_ep_buffer(
            deep_ep,
            group,
            max_tokens,
            hidden,
            topk,
            0,
        )

    sym_buffer = None
    if need_fused:
        sym_buffer = base.deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts, max_tokens, topk, hidden, intermediate
        )

    if args.mode == "low_latency" and need_deep_ep:
        ll_buffer = base._make_deep_ep_low_latency_buffer(
            deep_ep, group, max_tokens, hidden, num_experts
        )

    clamp = args.activation_clamp if math.isfinite(args.activation_clamp) else None
    cumulative_fused = torch.zeros(local_experts, dtype=torch.int, device="cuda")
    cumulative_baseline = torch.zeros_like(cumulative_fused)

    # Low-latency masked GEMM storage depends only on the largest sweep point.
    if args.mode == "low_latency" and need_deep_ep:
        ll_m_max = max_tokens * world_size
        ll_expected_m = max(1, (max_tokens * world_size * topk + num_experts - 1) // num_experts)
        ll_l1_y = torch.empty(
            (local_experts, ll_m_max, intermediate * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        ll_l2_y = torch.empty(
            (local_experts, ll_m_max, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    _print_rank0(
        rank,
        f"[MEMORY] allocated_gib={(total_bytes-free_bytes)/2**30:.3f} "
        f"free_gib={free_bytes/2**30:.3f} "
        f"sym_buffer_gib={(sym_buffer.buffer.nbytes/2**30 if sym_buffer is not None else 0):.3f}",
    )

    for batch in batches:
        # Each point gets deterministic but independent routing and inputs.
        torch.manual_seed(20260721 + rank * 100000 + batch)
        x_bf16 = torch.randn((batch, hidden), dtype=torch.bfloat16, device="cuda")
        scores = torch.randn((batch, num_experts), dtype=torch.float, device="cuda")
        topk_weights, topk_idx = torch.topk(scores, topk, dim=-1, largest=True, sorted=False)
        x_fp8 = base.per_token_cast_to_fp8(
            x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
        )
        fused_out = (
            torch.empty((batch, hidden), dtype=torch.bfloat16, device="cuda")
            if need_fused
            else None
        )

        def run_fused():
            sym_buffer.x[:batch].copy_(x_fp8[0])
            sym_buffer.x_sf[:batch].copy_(x_fp8[1])
            sym_buffer.topk_idx[:batch].copy_(topk_idx)
            sym_buffer.topk_weights[:batch].copy_(topk_weights)
            base.deep_gemm.fp8_mega_moe(
                fused_out,
                transformed_l1,
                transformed_l2,
                sym_buffer,
                cumulative_local_expert_recv_stats=cumulative_fused,
                recipe=(128, 128, 128),
                activation="swiglu",
                activation_clamp=clamp,
                fast_math=bool(args.fast_math),
            )
            return fused_out

        if args.mode == "normal" and need_deep_ep:
            def run_deep_ep():
                recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                    x_fp8,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    cumulative_local_expert_recv_stats=cumulative_baseline,
                    num_experts=num_experts,
                    expert_alignment=alignment,
                    do_cpu_sync=False,
                    do_handle_copy=False,
                    do_expand=True,
                    use_tma_aligned_col_major_sf=False,
                )
                recv_tokens = recv_x[0].size(0)
                l1_y = torch.empty(
                    (recv_tokens, intermediate * 2), dtype=torch.bfloat16, device="cuda"
                )
                base.deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    recv_x,
                    l1_weights,
                    l1_y,
                    handle.psum_num_recv_tokens_per_expert,
                    use_psum_layout=True,
                    disable_ue8m0_cast=True,
                )
                l1_fp8 = base.swiglu_apply_weight_to_fp8_triton(
                    x=l1_y,
                    topk_weights=recv_topk_weights,
                    clamp_value=clamp,
                    num_per_channels=base.BASELINE_L2_ACT_SF_GRAN,
                    use_ue8m0_scale=True,
                )
                l2_y = torch.empty((recv_tokens, hidden), dtype=torch.bfloat16, device="cuda")
                base.deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    l1_fp8,
                    l2_weights,
                    l2_y,
                    handle.psum_num_recv_tokens_per_expert,
                    use_psum_layout=True,
                    disable_ue8m0_cast=True,
                )
                return ep_buffer.combine(l2_y, handle=handle)[0]
        elif args.mode == "low_latency" and need_deep_ep:
            topk_idx_ll = topk_idx.to(torch.int64)
            ll_combined = torch.empty((batch, hidden), dtype=torch.bfloat16, device="cuda")

            def run_deep_ep():
                (recv_x_data, recv_x_sf), masked_m, handle, _, _ = ll_buffer.low_latency_dispatch(
                    x_bf16,
                    topk_idx_ll,
                    max_tokens,
                    num_experts,
                    use_fp8=True,
                    round_scale=False,
                    use_ue8m0=False,
                    async_finish=False,
                    return_recv_hook=False,
                )
                base.deep_gemm.fp8_m_grouped_gemm_nt_masked(
                    (recv_x_data, recv_x_sf),
                    l1_weights,
                    ll_l1_y,
                    masked_m,
                    ll_expected_m,
                    disable_ue8m0_cast=True,
                )
                l1_fp8, l1_sf = base.swiglu_masked_post_quant_to_fp8(
                    ll_l1_y,
                    masked_m,
                    quant_group_size=base.BASELINE_L2_ACT_SF_GRAN,
                    clamp_value=clamp,
                    use_ue8m0_scale=False,
                )
                base.deep_gemm.fp8_m_grouped_gemm_nt_masked(
                    (l1_fp8, l1_sf),
                    l2_weights,
                    ll_l2_y,
                    masked_m,
                    ll_expected_m,
                    disable_ue8m0_cast=True,
                )
                combined, _, _ = ll_buffer.low_latency_combine(
                    ll_l2_y,
                    topk_idx_ll,
                    topk_weights,
                    handle,
                    use_logfmt=False,
                    zero_copy=False,
                    async_finish=False,
                    return_recv_hook=False,
                    out=ll_combined,
                )
                return combined

        # Compile/JIT enabled paths before timing this point.
        if need_fused:
            run_fused()
        if need_deep_ep:
            run_deep_ep()
        torch.cuda.synchronize()
        dist.barrier()

        fused_stats = None
        deep_ep_stats = None
        if need_fused:
            fused_seconds = base.bench_kineto(
                run_fused,
                base.SM90_KERNEL_NAME,
                barrier=lambda: dist.barrier(),
                num_tests=args.num_bench_tests,
                suppress_kineto_output=True,
            )
            fused_stats = _global_stats(fused_seconds)
        if need_deep_ep:
            deep_ep_seconds = base._bench_cuda_events(
                run_deep_ep,
                num_warmup=args.num_warmup,
                num_repeat=args.num_repeat,
                l2_flush_gb=args.l2_flush_gb,
            )
            deep_ep_stats = _global_stats(deep_ep_seconds)

        if args.path == "both":
            fused_max, fused_mean, fused_min = fused_stats
            deep_ep_max, deep_ep_mean, deep_ep_min = deep_ep_stats
            _print_rank0(
                rank,
                f"[RESULT] model={args.model_name} mode={args.mode} batch={batch} "
                f"fused_us_max={fused_max*1e6:.3f} fused_us_mean={fused_mean*1e6:.3f} "
                f"fused_us_min={fused_min*1e6:.3f} deep_ep_us_max={deep_ep_max*1e6:.3f} "
                f"deep_ep_us_mean={deep_ep_mean*1e6:.3f} deep_ep_us_min={deep_ep_min*1e6:.3f} "
                f"speedup={deep_ep_max/fused_max:.4f}",
            )
        else:
            time_max, time_mean, time_min = fused_stats or deep_ep_stats
            _print_rank0(
                rank,
                f"[RESULT] model={args.model_name} mode={args.mode} path={args.path} batch={batch} "
                f"time_us_max={time_max*1e6:.3f} time_us_mean={time_mean*1e6:.3f} "
                f"time_us_min={time_min*1e6:.3f}",
            )
        dist.barrier()

    _print_rank0(rank, f"[DONE] model={args.model_name} mode={args.mode}")
    dist.barrier()
    if sym_buffer is not None:
        sym_buffer.destroy()
    if ep_buffer is not None:
        ep_buffer.destroy()
    if ll_buffer is not None:
        ll_buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", choices=("low_latency", "normal"), required=True)
    parser.add_argument("--path", choices=("both", "fused", "deep_ep"), default="both")
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate-hidden", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--num-topk", type=int, required=True)
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-bench-tests", type=int, default=30)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-repeat", type=int, default=20)
    parser.add_argument("--l2-flush-gb", type=float, default=1.0)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    args = parser.parse_args()
    torch.multiprocessing.spawn(run, args=(args.num_processes, args), nprocs=args.num_processes)
