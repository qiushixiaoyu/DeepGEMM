"""Two-node SM90 MegaMoE vs DeepEP performance sweep.

This is a benchmark driver only.  It keeps model weights and communication
buffers alive while sweeping token counts, so large model shapes do not pay
weight construction/quantization and process-group startup at every point.
"""

import argparse
import math
import os
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


def _make_world_size_matched_route(
    rank: int,
    world_size: int,
    batch: int,
    topk: int,
    local_experts: int,
) -> torch.Tensor:
    """Build an exact per-rank workload match for no-RDMA/RDMA A/B.

    Every destination rank receives ``batch * topk`` routes, and the local
    expert index for each ``(destination, token, topk-slot)`` is independent
    of world size.  Consequently an 8-rank run with half as many global
    experts and a 16-rank run have identical local-expert counts, tile counts,
    and WGMMA work per corresponding rank.  On 16 ranks the interleaved rank
    offsets put half of top-k slots on the other 8-GPU node.
    """
    token = torch.arange(batch, dtype=torch.int64, device="cuda")[:, None]
    slot = torch.arange(topk, dtype=torch.int64, device="cuda")[None, :]
    half_world = world_size // 2
    rank_offset = slot // 2 + (slot % 2) * half_world
    dst_rank = (rank + rank_offset) % world_size
    local_expert = (dst_rank + token * topk + slot) % local_experts
    return dst_rank * local_experts + local_expert


def _make_rdma_compiled_local_route(
    rank: int,
    world_size: int,
    batch: int,
    topk: int,
    local_experts: int,
) -> torch.Tensor:
    """Keep the matched workload but make every data route node-local.

    This is the middle arm of the strict A/B/C test:

    A. 8 ranks, no inter-node specialization compiled;
    B. 16 ranks, inter-node specialization compiled, data routes node-local;
    C. 16 ranks, inter-node specialization compiled, half of routes remote.

    The local expert assignment is inherited from the world-size-matched route,
    while the destination rank is folded into the source rank's 8-GPU node.
    Thus B retains the RDMA/control code and the same per-rank GEMM work as C,
    but dispatch READ and scatter WRITE do not issue inter-node data WQEs.
    """
    if world_size % 16 != 0:
        raise ValueError(
            "DG_PERF_ROUTE_MATCH_LOCAL_NODE requires a multiple of 16 ranks"
        )
    token = torch.arange(batch, dtype=torch.int64, device="cuda")[:, None]
    slot = torch.arange(topk, dtype=torch.int64, device="cuda")[None, :]
    half_world = world_size // 2
    rank_offset = slot // 2 + (slot % 2) * half_world
    original_dst_rank = (rank + rank_offset) % world_size
    local_expert = (original_dst_rank + token * topk + slot) % local_experts
    node_base = (rank // 8) * 8
    local_dst_rank = node_base + original_dst_rank % 8
    return local_dst_rank * local_experts + local_expert


def _dump_fp8_phase_profile(sym_buffer, model_name: str, batch: int, rank: int) -> None:
    """Read the silent FP8 phase profiler from the SymmBuffer tail."""
    if os.environ.get("DG_MEGA_MOE_PHASE_PROFILE", "0") != "1":
        return
    profile_rows, profile_slots, num_sms = 256, 40, 78
    flat = sym_buffer.buffer.view(torch.uint8).view(-1)
    tail = flat[-(profile_rows * profile_slots * 8):]
    profile = tail.view(torch.int64).view(profile_rows, profile_slots).cpu()
    rows = profile[:num_sms]
    total = rows[:, 11]
    slowest_sm = int(total.argmax())
    slowest = profile[slowest_sm].tolist()
    entry = rows[:, 31]
    sent = rows[:, 32]
    ready = rows[:, 33]
    l1_blocks = int(rows[:, 13].sum())
    l2_blocks = int(rows[:, 14].sum())
    l1_mainloop = rows[:, 36]
    l2_mainloop = rows[:, 37]
    l1_a_wait = rows[:, 38]
    l2_a_wait = rows[:, 39]
    math_proxy = l1_mainloop + l2_mainloop - l1_a_wait - l2_a_wait
    math_proxy_sum = int(math_proxy.sum())
    num_math_blocks = l1_blocks + l2_blocks
    print(
        f"[HOSTPROF_SKEW] batch={batch} rank={rank} "
        f"cta_entry_skew_ns={int(entry.max() - entry.min())} "
        f"first_ready_ns={int((ready - entry.min()).min())} "
        f"last_ready_ns={int((ready - entry.min()).max())} "
        f"last_sent_ns={int((sent - entry.min()).max())}",
        flush=True,
    )
    print(
        f"[HOSTPROF_AGG] batch={batch} rank={rank} slowest_sm={slowest_sm} "
        f"total_min={int(total.min())} total_p50={int(total.median())} "
        f"total_max={int(total.max())} "
        f"meta_max={int(rows[:, 0].max())} barrier_max={int(rows[:, 1].max())} "
        f"pull_max={int(rows[:, 2].max())} l1_max={int(rows[:, 5].max())} "
        f"l2_max={int(rows[:, 6].max())} pub_max={int(rows[:, 8].max())} "
        f"combbar_max={int(rows[:, 9].max())} reduce_max={int(rows[:, 10].max())} "
        f"CLEANUPBAR_max={int(rows[:, 4].max())} "
        f"CLEANUPBAR_p50={int(rows[:, 4].median())} "
        f"scatter_max={int(rows[:, 7].max())} "
        f"a_wait_max={int(rows[:, 35].max())} "
        f"l1_mainloop_sum={int(l1_mainloop.sum())} "
        f"l2_mainloop_sum={int(l2_mainloop.sum())} "
        f"l1_a_wait_sum={int(l1_a_wait.sum())} "
        f"l2_a_wait_sum={int(l2_a_wait.sum())} "
        f"l1_blocks={l1_blocks} l2_blocks={l2_blocks} "
        f"math_proxy_sum={math_proxy_sum} "
        f"math_proxy_cyc_per_block="
        f"{math_proxy_sum / max(1, num_math_blocks):.3f}",
        flush=True,
    )
    print(
        f"[HOSTPROF] model={model_name} batch={batch} rank={rank} "
        f"meta_cyc={slowest[0]} barrier_cyc={slowest[1]} "
        f"pull_cyc={slowest[2]} total_cyc={slowest[11]} "
        f"entry_ns={slowest[31]} sent_ns={slowest[32]} ready_ns={slowest[33]}",
        flush=True,
    )


def _ep_call_kwargs(deep_ep, alignment):
    """DG_EP_REF_CALL=1 时改用 DeepEP 参考测试的显式 config 且不传 expert_alignment。"""
    if int(os.environ.get("DG_EP_REF_CALL", "0")):
        return {"config": deep_ep.Config(24, 8, 512, 16, 128)}
    return {"expert_alignment": alignment}


def _ep_combine_kwargs(deep_ep):
    if int(os.environ.get("DG_EP_REF_CALL", "0")):
        return {"config": deep_ep.Config(24, 8, 512, 16, 128)}
    return {}


def _make_legacy_normal_buffer(deep_ep, group, hidden: int):
    """Create the production-style DeepEP normal buffer.

    Legacy ``Buffer`` dispatches one token per destination rank. SGLang then
    uses ``ep_scatter``/``ep_gather`` around the grouped GEMMs; it does not
    assume that communication already returned an expert-expanded layout.
    """
    hidden_bytes = hidden * 2
    num_nvl_bytes = 0
    num_rdma_bytes = 0
    for config in (
        deep_ep.Buffer.get_dispatch_config(group.size()),
        deep_ep.Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            num_nvl_bytes,
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
        )
        num_rdma_bytes = max(
            num_rdma_bytes,
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
        )
    # DG_EP_BIG_BUFFER=1: 用 DeepEP 参考测试的宽裕尺寸 (NVL 2GB / RDMA 1GB, QP 24)
    # 而非 size hint 推荐值，用于定位四机 notify_dispatch 死锁。
    if int(os.environ.get("DG_EP_BIG_BUFFER", "0")):
        num_nvl_bytes, num_rdma_bytes = int(2e9), int(1e9)
        num_qps = 24
    else:
        num_qps = deep_ep.Buffer.num_sms
    return deep_ep.Buffer(
        group,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        num_qps_per_rank=num_qps,
        explicitly_destroy=True,
    )


def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    if args.row_combine:
        os.environ["DG_MEGA_MOE_ROW_COMBINE"] = "1"
    if args.phase_profile:
        if args.path != "fused":
            raise RuntimeError("--phase-profile requires --path fused")
        os.environ["DG_MEGA_MOE_PHASE_PROFILE"] = "1"
    if args.deep_ep_phase_profile:
        if args.path != "deep_ep" or args.mode != "low_latency":
            raise RuntimeError(
                "--deep-ep-phase-profile requires --path deep_ep "
                "--mode low_latency"
            )
        if not args.fuse_shared:
            raise RuntimeError(
                "--deep-ep-phase-profile currently profiles the full-layer "
                "baseline and requires --fuse-shared"
            )

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
    # DG_SWEEP_MAX_TOKENS: 把「对称缓冲定尺」与「实际 batch」解耦，用于定位
    # 递增序崩溃/大 pool 崩溃这两类问题（正常扫描不要设）。
    _mt_override = int(os.environ.get("DG_SWEEP_MAX_TOKENS", "0"))
    if _mt_override:
        assert _mt_override >= max_tokens, "override 必须 >= 最大 batch"
        max_tokens = _mt_override
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
        f"fuse_shared={int(args.fuse_shared)} "
        f"deep_ep_phase_profile={int(args.deep_ep_phase_profile)} "
        f"normal_impl={'legacy_scatter_gather' if args.mode == 'normal' else 'n/a'} "
        f"row_combine={int(args.row_combine)} "
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

    shared_l1_weights = shared_l2_weights = None
    if args.fuse_shared:
        shared_l1_bf16 = torch.randn(
            (1, intermediate * 2, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        ) * 0.05
        shared_l1_weights = base._quantize_grouped_fp8_block_128_128(shared_l1_bf16)
        del shared_l1_bf16
        torch.cuda.empty_cache()

        shared_l2_bf16 = torch.randn(
            (1, hidden, intermediate),
            dtype=torch.bfloat16,
            device="cuda",
        ) * 0.05
        shared_l2_weights = base._quantize_grouped_fp8_block_128_128(shared_l2_bf16)
        del shared_l2_bf16
        torch.cuda.empty_cache()

    transformed_l1 = transformed_l2 = None
    transformed_shared_l1 = transformed_shared_l2 = None
    if need_fused:
        transformed_l1, transformed_l2 = base.deep_gemm.transform_weights_for_mega_moe_sm90(
            l1_weights, l2_weights
        )
        if args.fuse_shared:
            transformed_shared_l1, transformed_shared_l2 = (
                base.deep_gemm.transform_weights_for_mega_moe_sm90(
                    shared_l1_weights, shared_l2_weights
                )
            )
    alignment = base.deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    base.deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    deep_ep = base._import_deep_ep() if need_deep_ep else None
    if need_deep_ep and deep_ep is None:
        raise RuntimeError("deep_ep is required for this comparison")

    ep_buffer = None
    ll_buffer = None
    if args.mode == "normal" and need_deep_ep:
        ep_buffer = _make_legacy_normal_buffer(deep_ep, group, hidden)
        from deepep_scatter_gather import ep_gather, ep_scatter

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
    shared_stream = torch.cuda.Stream() if args.fuse_shared and need_deep_ep else None
    if args.dump_output_dir:
        os.makedirs(args.dump_output_dir, exist_ok=True)

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
        if os.environ.get("DG_PERF_ROUTE_MATCH_LOCAL_NODE", "0") == "1":
            topk_idx = _make_rdma_compiled_local_route(
                rank, world_size, batch, topk, local_experts
            )
        elif os.environ.get("DG_PERF_ROUTE_MATCH_WORLD_SIZE", "0") == "1":
            topk_idx = _make_world_size_matched_route(
                rank, world_size, batch, topk, local_experts
            )
        elif os.environ.get("DG_PERF_ROUTE_LOCAL_NODE", "0") == "1":
            # Diagnostic: keep every routed token inside this rank's NVLink
            # domain.  The full multi-node control flow (16-rank barriers,
            # 16 source slots, gateway handshake) is unchanged; only the
            # inter-node data/entry traffic goes to zero.
            nvl_peers = 8
            node_id = rank // nvl_peers
            experts_per_rank_local = num_experts // world_size
            node_expert_base = node_id * nvl_peers * experts_per_rank_local
            node_expert_span = nvl_peers * experts_per_rank_local
            topk_idx = node_expert_base + (topk_idx % node_expert_span)
        x_fp8 = base.per_token_cast_to_fp8(
            x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
        )
        fused_out = (
            torch.empty((batch, hidden), dtype=torch.bfloat16, device="cuda")
            if need_fused
            else None
        )

        if args.fuse_shared and need_deep_ep:
            shared_l1_weights_single = (
                shared_l1_weights[0][0],
                shared_l1_weights[1][0],
            )
            shared_l2_weights_single = (
                shared_l2_weights[0][0],
                shared_l2_weights[1][0],
            )
            shared_l1_y = torch.empty(
                (batch, intermediate * 2), dtype=torch.bfloat16, device="cuda"
            )
            shared_y = torch.empty(
                (batch, hidden), dtype=torch.bfloat16, device="cuda"
            )

            def run_shared_expert():
                base.deep_gemm.fp8_gemm_nt(
                    x_fp8,
                    shared_l1_weights_single,
                    shared_l1_y,
                    recipe=(1, 128, 128),
                    disable_ue8m0_cast=True,
                )
                shared_l1_fp8 = base.swiglu_apply_weight_to_fp8_triton(
                    x=shared_l1_y,
                    topk_weights=None,
                    clamp_value=clamp,
                    num_per_channels=base.BASELINE_L2_ACT_SF_GRAN,
                    use_ue8m0_scale=True,
                )
                base.deep_gemm.fp8_gemm_nt(
                    shared_l1_fp8,
                    shared_l2_weights_single,
                    shared_y,
                    recipe=(1, 128, 128),
                    disable_ue8m0_cast=True,
                )
                return shared_y

        def run_fused():
            sym_buffer.x[:batch].copy_(x_fp8[0])
            sym_buffer.x_sf[:batch].copy_(x_fp8[1])
            sym_buffer.topk_idx[:batch].copy_(topk_idx)
            sym_buffer.topk_weights[:batch].copy_(topk_weights)
            if args.fuse_shared:
                base.deep_gemm.fp8_mega_moe_with_shared(
                    fused_out,
                    transformed_l1,
                    transformed_l2,
                    transformed_shared_l1,
                    transformed_shared_l2,
                    sym_buffer,
                    cumulative_local_expert_recv_stats=cumulative_fused,
                    recipe=(128, 128, 128),
                    activation="swiglu",
                    activation_clamp=clamp,
                    fast_math=bool(args.fast_math),
                )
            else:
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

        if args.phase_profile:
            # The first launch can include per-rank JIT and CUDA context skew,
            # which is observed by the kernel's global dispatch barrier.  Warm
            # every rank first, then profile a synchronized steady-state launch.
            run_fused()
            torch.cuda.synchronize()
            dist.barrier()
            _print_rank0(rank, f"[PROFILE_WARMUP_DONE] model={args.model_name} batch={batch}")
            run_fused()
            torch.cuda.synchronize()
            dist.barrier()
            _dump_fp8_phase_profile(sym_buffer, args.model_name, batch, rank)
            _print_rank0(rank, f"[PROFILE_POINT] model={args.model_name} batch={batch}")
            continue

        if args.mode == "normal" and need_deep_ep:
            def run_deep_ep_routed():
                (
                    num_tokens_per_rank,
                    num_tokens_per_rdma_rank,
                    num_tokens_per_expert,
                    is_token_in_rank,
                    _,
                ) = ep_buffer.get_dispatch_layout(topk_idx, num_experts)
                (
                    recv_x,
                    recv_topk_idx,
                    recv_topk_weights,
                    num_recv_tokens_per_expert,
                    handle,
                    _,
                ) = ep_buffer.dispatch(
                    x_fp8,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    num_tokens_per_rank=num_tokens_per_rank,
                    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                    is_token_in_rank=is_token_in_rank,
                    num_tokens_per_expert=num_tokens_per_expert,
                    **_ep_call_kwargs(deep_ep, alignment),
                )

                # Match SGLang normal mode: communication returns one row per
                # destination rank, then ep_scatter expands and groups rows by
                # local expert for the contiguous grouped GEMMs.
                expanded_tokens = sum(num_recv_tokens_per_expert)
                recv_token_rows = recv_x[0].size(0)
                expanded_x = torch.empty(
                    (expanded_tokens, hidden),
                    dtype=recv_x[0].dtype,
                    device="cuda",
                )
                expanded_sf = torch.empty(
                    (expanded_tokens, recv_x[1].shape[1]),
                    dtype=recv_x[1].dtype,
                    device="cuda",
                )
                m_indices = torch.empty(
                    expanded_tokens, dtype=torch.int32, device="cuda"
                )
                output_index = torch.empty_like(recv_topk_idx, dtype=torch.int32)
                counts_gpu = torch.tensor(
                    num_recv_tokens_per_expert,
                    dtype=torch.int32,
                    pin_memory=True,
                    device="cpu",
                ).cuda(non_blocking=True)
                expert_start_loc = torch.empty_like(counts_gpu)
                ep_scatter(
                    recv_x[0],
                    recv_x[1],
                    recv_topk_idx,
                    counts_gpu,
                    expert_start_loc,
                    expanded_x,
                    expanded_sf,
                    m_indices,
                    output_index,
                    scale_ue8m0=False,
                )
                l1_y = torch.empty(
                    (expanded_tokens, intermediate * 2),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                base.deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (expanded_x, expanded_sf),
                    l1_weights,
                    l1_y,
                    m_indices,
                    use_psum_layout=False,
                    disable_ue8m0_cast=True,
                )
                l1_fp8 = base.swiglu_apply_weight_to_fp8_triton(
                    x=l1_y,
                    topk_weights=None,
                    clamp_value=clamp,
                    num_per_channels=base.BASELINE_L2_ACT_SF_GRAN,
                    use_ue8m0_scale=True,
                )
                l2_y = torch.empty(
                    (expanded_tokens, hidden), dtype=torch.bfloat16, device="cuda"
                )
                base.deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    l1_fp8,
                    l2_weights,
                    l2_y,
                    m_indices,
                    use_psum_layout=False,
                    disable_ue8m0_cast=True,
                )
                gathered_y = torch.empty(
                    (recv_token_rows, hidden), dtype=torch.bfloat16, device="cuda"
                )
                ep_gather(
                    l2_y,
                    recv_topk_idx,
                    recv_topk_weights,
                    output_index,
                    gathered_y,
                )
                return ep_buffer.combine(gathered_y, handle=handle, **_ep_combine_kwargs(deep_ep))[0]
        elif args.mode == "low_latency" and need_deep_ep:
            topk_idx_ll = topk_idx.to(torch.int64)
            ll_combined = torch.empty((batch, hidden), dtype=torch.bfloat16, device="cuda")

            def run_deep_ep_routed(profile=None):
                def record_profile(phase: str, boundary: str) -> None:
                    if profile is not None and profile["target"] == phase:
                        profile[boundary].record()

                record_profile("dispatch", "start")
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
                record_profile("dispatch", "end")
                record_profile("l1", "start")
                base.deep_gemm.fp8_m_grouped_gemm_nt_masked(
                    (recv_x_data, recv_x_sf),
                    l1_weights,
                    ll_l1_y,
                    masked_m,
                    ll_expected_m,
                    disable_ue8m0_cast=True,
                )
                record_profile("l1", "end")
                record_profile("activation", "start")
                l1_fp8, l1_sf = base.swiglu_masked_post_quant_to_fp8(
                    ll_l1_y,
                    masked_m,
                    quant_group_size=base.BASELINE_L2_ACT_SF_GRAN,
                    clamp_value=clamp,
                    use_ue8m0_scale=False,
                )
                record_profile("activation", "end")
                record_profile("l2", "start")
                base.deep_gemm.fp8_m_grouped_gemm_nt_masked(
                    (l1_fp8, l1_sf),
                    l2_weights,
                    ll_l2_y,
                    masked_m,
                    ll_expected_m,
                    disable_ue8m0_cast=True,
                )
                record_profile("l2", "end")
                record_profile("combine", "start")
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
                record_profile("combine", "end")
                return combined

        if need_deep_ep:
            if args.fuse_shared:
                def run_deep_ep():
                    current_stream = torch.cuda.current_stream()
                    shared_stream.wait_stream(current_stream)
                    with torch.cuda.stream(shared_stream):
                        shared_output = run_shared_expert()
                    routed_output = run_deep_ep_routed()
                    current_stream.wait_stream(shared_stream)
                    routed_output.add_(shared_output)
                    return routed_output
            else:
                run_deep_ep = run_deep_ep_routed

        # Compile/JIT enabled paths before timing this point.
        if need_fused:
            run_fused()
        if need_deep_ep:
            run_deep_ep()
        torch.cuda.synchronize()
        dist.barrier()

        if args.deep_ep_phase_profile:
            phase_order = (
                "total",
                "routed",
                "dispatch",
                "l1",
                "activation",
                "l2",
                "combine",
                "shared",
                "shared_wait",
                "add",
            )
            phase_samples = {name: [] for name in phase_order}

            # The generic JIT launch above has already compiled every kernel.
            # Additional warmups settle communication epochs and the shared
            # stream before collecting event intervals.
            for _ in range(args.num_warmup):
                run_deep_ep()
            torch.cuda.synchronize()
            dist.barrier()

            # Profile one interval per full pipeline execution. Recording a
            # dense chain of events materially perturbs these sub-millisecond
            # kernels, while this isolated method has the same two-event
            # structure as the total-time benchmark.
            for target in phase_order:
                torch.cuda.synchronize()
                dist.barrier()
                for _ in range(args.num_repeat):
                    if args.l2_flush_gb > 0:
                        free_bytes, _ = torch.cuda.mem_get_info()
                        flush_bytes = min(
                            int(args.l2_flush_gb * 1e9), int(free_bytes * 0.5)
                        )
                        if flush_bytes >= 4:
                            torch.empty(
                                flush_bytes // 4, dtype=torch.int, device="cuda"
                            ).zero_()

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    profile = {"target": target, "start": start, "end": end}
                    current_stream = torch.cuda.current_stream()

                    if target in ("total", "routed"):
                        start.record(current_stream)
                    shared_stream.wait_stream(current_stream)
                    with torch.cuda.stream(shared_stream):
                        if target == "shared":
                            start.record(shared_stream)
                        shared_output = run_shared_expert()
                        if target == "shared":
                            end.record(shared_stream)

                    routed_output = run_deep_ep_routed(profile)
                    if target == "routed":
                        end.record(current_stream)
                    if target == "shared_wait":
                        start.record(current_stream)
                    current_stream.wait_stream(shared_stream)
                    if target == "shared_wait":
                        end.record(current_stream)
                    if target == "add":
                        start.record(current_stream)
                    routed_output.add_(shared_output)
                    if target in ("total", "add"):
                        end.record(current_stream)
                    torch.cuda.synchronize()
                    phase_samples[target].append(start.elapsed_time(end) * 1e3)

            local_medians = []
            for name in phase_order:
                values = sorted(phase_samples[name])
                local_medians.append(values[len(values) // 2])
            local_profile = torch.tensor(
                local_medians, dtype=torch.float64, device="cuda"
            )
            gathered_profiles = [
                torch.empty_like(local_profile) for _ in range(world_size)
            ]
            dist.all_gather(gathered_profiles, local_profile)
            if rank == 0:
                profiles = torch.stack(gathered_profiles).cpu()
                critical_rank = int(torch.argmax(profiles[:, 0]).item())
                critical = profiles[critical_rank]
                values = {
                    name: float(critical[idx].item())
                    for idx, name in enumerate(phase_order)
                }
                phase_sum = sum(
                    values[name]
                    for name in ("dispatch", "l1", "activation", "l2", "combine")
                )
                print(
                    f"[DEEP_EP_PHASE_PROFILE] model={args.model_name} batch={batch} "
                    f"critical_rank={critical_rank} repeats={args.num_repeat} "
                    f"total_us={values['total']:.3f} routed_us={values['routed']:.3f} "
                    f"dispatch_us={values['dispatch']:.3f} l1_us={values['l1']:.3f} "
                    f"activation_us={values['activation']:.3f} l2_us={values['l2']:.3f} "
                    f"combine_us={values['combine']:.3f} shared_us={values['shared']:.3f} "
                    f"shared_wait_us={values['shared_wait']:.3f} add_us={values['add']:.3f} "
                    f"routed_phase_sum_us={phase_sum:.3f}",
                    flush=True,
                )
            dist.barrier()
            continue

        if args.accuracy_only:
            output = run_fused() if need_fused else run_deep_ep()
            torch.cuda.synchronize()
            assert output.shape == (batch, hidden)
            assert output.dtype == torch.bfloat16
            assert torch.isfinite(output).all(), (
                f"non-finite output: path={args.path} mode={args.mode} batch={batch} rank={rank}"
            )
            if args.dump_output_dir:
                output_path = os.path.join(
                    args.dump_output_dir,
                    f"{args.model_name}_{args.path}_{args.mode}_b{batch}_r{rank}.pt",
                )
                torch.save(output.detach().cpu(), output_path)
            _print_rank0(
                rank,
                f"[ACCURACY] model={args.model_name} mode={args.mode} path={args.path} "
                f"batch={batch} finite=1 dump={int(bool(args.dump_output_dir))}",
            )
            dist.barrier()
            continue

        fused_stats = None
        deep_ep_stats = None
        if need_fused:
            # DG_FUSED_EVENT_TIMING=1 times the fused kernel with the same
            # free-running CUDA-event method as the DeepEP baseline, so the
            # two are not compared across different timing methodologies.
            if os.environ.get("DG_FUSED_EVENT_TIMING", "0") == "1":
                fused_seconds = base._bench_cuda_events(
                    run_fused,
                    num_warmup=args.num_warmup,
                    num_repeat=args.num_repeat,
                    l2_flush_gb=args.l2_flush_gb,
                )
            else:
              fused_seconds = base.bench_kineto(
                run_fused,
                base.SM90_KERNEL_NAME,
                barrier=lambda: dist.barrier(),
                num_tests=args.num_bench_tests,
                suppress_kineto_output=True,
                # bench_kineto hardcodes an 8 GB flush; honour --l2-flush-gb 0
                # so the fused path can be compared against DeepEP fairly.
                flush_l2=args.l2_flush_gb > 0,
            )
            fused_stats = _global_stats(fused_seconds)

            # Read the LAST launch's counters without device-side printf.
            _dump_fp8_phase_profile(sym_buffer, args.model_name, batch, rank)
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
    parser.add_argument(
        "--phase-profile",
        action="store_true",
        help="run each fused point once with device-side phase cycle profiling",
    )
    parser.add_argument(
        "--deep-ep-phase-profile",
        action="store_true",
        help=(
            "profile low-latency dispatch, GEMMs, activation, combine, the "
            "overlapped shared expert, wait, and final add with CUDA events"
        ),
    )
    parser.add_argument(
        "--row-combine",
        action="store_true",
        help="use symmetric row staging and warp-cooperative internode combine writes",
    )
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
    parser.add_argument(
        "--fuse-shared",
        action="store_true",
        help=(
            "include one replicated shared expert; MegaMOE fuses it into the "
            "kernel and DeepEP overlaps the standalone FP8 shared MLP"
        ),
    )
    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="run finite-output checks and optional dumps without timing",
    )
    parser.add_argument(
        "--dump-output-dir",
        help="optional per-rank output directory used for isolated-path accuracy comparison",
    )
    args = parser.parse_args()
    torch.multiprocessing.spawn(run, args=(args.num_processes, args), nprocs=args.num_processes)
