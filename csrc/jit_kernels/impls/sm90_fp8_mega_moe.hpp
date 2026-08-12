#pragma once

#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE host runtime
// ----------------------------------------------------------------------------
// This is the SM90 counterpart of `SM100FP8FP4MegaMoERuntime`. The kernel
// itself lives in `deep_gemm/impls/sm90_fp8_mega_moe.cuh`.
//
// Differences from SM100 path:
//   * Activations and weights are both FP8 (e4m3); no FP4.
//   * Activation/weight scale factors (SF) are float, not UE8M0 int + per-32
//     UTCCP layout. L1 activation SF and weight SF are per-128 K; the fused L1
//     epilogue writes L2 activation SF at per-64 K granularity.
//   * No tensor memory: WGMMA accumulators are register-resident.
//   * Cluster size is at most 2 (TMA multicast on A); no 2-CTA UMMA.
// ============================================================================

class SM90FP8MegaMoERuntime final : public LaunchRuntime<SM90FP8MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        int epilogue_registers;
        bool reuse_accum_as_final;
        bool l2_arrival_counter;
        bool l2_epilogue_requires_full_sync;
        bool split_phase_hot_path;
        bool dispatch_expert_ready;
        bool lazy_expert_count;
        bool combine_full_row;
        bool combine_expert_ready;
        bool use_swap_ab;
        bool fuse_shared_expert;
        MegaMoESM90Config config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormaps for activations and weights. Weight scale factors use
        // block (128, 128) quantization and are loaded by the math warpgroup
        // directly from global memory (no TMA descriptor required).
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        const float* l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const float* l2_weights_sf;
        CUtensorMap tensor_map_shared_l1_acts;
        const float* shared_l1_acts_sf;
        CUtensorMap tensor_map_shared_l1_weights;
        const float* shared_l1_weights_sf;
        CUtensorMap tensor_map_shared_l2_weights;
        const float* shared_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // Inter-node build: ranks span more than one NVLink domain (assume 8
        // NVLink peers per node). Inject macros so barrier.cuh (and later
        // dispatch/combine) take the NVSHMEM remote path. The `nvshmem` mention
        // in the comment also makes the JIT compiler device-link libnvshmem_device.
        constexpr int kNvlPeers = 8;
        std::string internode_prefix;
        if (args.num_ranks > kNvlPeers)
            internode_prefix = fmt::format(
                "// inter-node mega-moe: uses nvshmem device functions\n"
                "#define DG_MEGA_MOE_INTERNODE\n"
                "#define DG_MEGA_MOE_NVL_PEERS {}\n", kNvlPeers);
        if (get_env<int>("DG_MEGA_MOE_PHASE_PROFILE", 0) != 0)
            internode_prefix += "#define DG_MEGA_MOE_PHASE_PROFILE 1\n";
        // Keep the counters, drop the device printf: the host reads the
        // profile region (the tail of the symmetric buffer) after the run,
        // so a real benchmark loop stays unperturbed.
        // Removes the tag-3 cross-rank barrier: epoch-tagged control-plane
        // slots are never cleared, making cleanup rank-local (see the
        // "Barrier 与跨卡 Store 优化方案" doc, section 6).  Requires the
        // expert-ready dispatch protocol, which is what tags the slots.
        const bool no_tag3 =
            get_env<int>("DG_MEGA_MOE_NO_TAG3",
                         args.dispatch_expert_ready ? 1 : 0) != 0;
        DG_HOST_ASSERT(not no_tag3 or args.dispatch_expert_ready);
        if (no_tag3)
            internode_prefix += "#define DG_MEGA_MOE_NO_TAG3 1\n";
        if (get_env<int>("DG_MEGA_MOE_SKIP_NVSHMEM_QUIET", 0) != 0)
            internode_prefix +=
                "#define DG_MEGA_MOE_SKIP_NVSHMEM_QUIET 1\n";
        if (get_env<int>("DG_MEGA_MOE_PHASE_PROFILE_SILENT", 0) != 0)
            internode_prefix +=
                "#define DG_MEGA_MOE_PHASE_PROFILE_SILENT 1\n";
        const bool merge_ab_loader =
            get_env<int>("DG_MEGA_MOE_MERGE_AB_LOADER", 0) != 0;
        const bool async_publisher =
            get_env<int>("DG_MEGA_MOE_ASYNC_PUBLISHER", 0) != 0;
        const bool async_stage_local_rows =
            get_env<int>("DG_MEGA_MOE_ASYNC_STAGE_LOCAL_ROWS", 0) != 0;
        const bool async_publisher_row_doorbell =
            get_env<int>("DG_MEGA_MOE_ASYNC_PUBLISHER_ROW_DOORBELL", 0) != 0;
        DG_HOST_ASSERT(not async_publisher or merge_ab_loader);
        DG_HOST_ASSERT(not async_publisher or args.num_ranks > kNvlPeers);
        DG_HOST_ASSERT(not async_publisher or args.combine_full_row);
        DG_HOST_ASSERT(not async_publisher or args.combine_expert_ready);
        DG_HOST_ASSERT(not async_stage_local_rows or async_publisher);
        DG_HOST_ASSERT(not async_publisher_row_doorbell or async_publisher);
        const int publisher_max_active =
            get_env<int>("DG_MEGA_MOE_PUBLISHER_MAX_ACTIVE", 0);
        DG_HOST_ASSERT(
            publisher_max_active >= 0 and publisher_max_active <= 1024);
        DG_HOST_ASSERT(publisher_max_active == 0 or async_publisher);
        const int qp_inflight_wqes =
            get_env<int>("DG_MEGA_MOE_QP_INFLIGHT_WQES", 0);
        DG_HOST_ASSERT(qp_inflight_wqes >= 0 and qp_inflight_wqes <= 4096);
        DG_HOST_ASSERT(qp_inflight_wqes == 0 or async_publisher);
        const int publisher_rate_mbps =
            get_env<int>("DG_MEGA_MOE_PUBLISHER_RATE_MBPS", 0);
        DG_HOST_ASSERT(
            publisher_rate_mbps >= 0 and publisher_rate_mbps <= 60000);
        DG_HOST_ASSERT(publisher_rate_mbps == 0 or async_publisher);
        // Congestion controls (per-dst token bucket, dst grouping) exist for
        // the multi-MB bursts of large batches.  A decode-shaped instance
        // never approaches the ECN trip point, so there they are pure
        // overhead — measured -8%~-15% on b2..b64.  Split the default by
        // instance capacity, the same launch-time selection used for V3.
        const bool congestion_control_needed =
            args.num_max_tokens_per_rank > 512;
        const int publisher_link_mbps =
            get_env<int>("DG_MEGA_MOE_PUBLISHER_LINK_MBPS", 50000);
        DG_HOST_ASSERT(
            publisher_link_mbps > 0 and publisher_link_mbps <= 100000);
        // Defaults to auto (-1) with the async publisher: 56% of line rate
        // split across inter-node peers.  The 44% headroom covers dispatch
        // traffic sharing each receiver NIC — the publish fan-in budget
        // measures at 24~28 GB/s on 400G links, well below line rate, and
        // 56% lands on the per-dst rate validated on all three models
        // (3500 MB/s at 400G / 16 ranks).  Explicit 0 disables.  The
        // historical batch-4096 bistable slow mode under per-dst buckets
        // is fixed by chain polling below, which is also on by default.
        int publisher_dst_rate_mbps =
            get_env<int>("DG_MEGA_MOE_PUBLISHER_DST_RATE_MBPS", -1);
        if (publisher_dst_rate_mbps < 0)
            publisher_dst_rate_mbps =
                (async_publisher and congestion_control_needed)
                ? publisher_link_mbps * 56 / 100 /
                      static_cast<int>(args.num_ranks - kNvlPeers)
                : 0;
        DG_HOST_ASSERT(publisher_dst_rate_mbps <= 60000);
        DG_HOST_ASSERT(publisher_dst_rate_mbps == 0 or async_publisher);
        // Dst grouping and chain polling are the validated defaults with
        // the async publisher (all three models beat inline publish across
        // b1024-8192 with CNP at zero).  Explicit 0 opts out of either.
        const bool publisher_dst_grouped =
            get_env<int>("DG_MEGA_MOE_PUBLISHER_DST_GROUPED",
                         (async_publisher and congestion_control_needed)
                             ? 1 : 0) != 0;
        DG_HOST_ASSERT(not publisher_dst_grouped or async_publisher);
        // Chain polling advances whichever chained pair has data ready
        // instead of serially owning one pair until its long-tail block
        // lands.  It assumes the dst-grouped chain layout and replaces the
        // per-pair loop body, so the other publisher experiments (and the
        // pair-granular profiler) are mutually exclusive with it: when any
        // of those is explicitly enabled the DEFAULT gracefully degrades
        // to the serial grouped path, while an EXPLICIT poll request still
        // asserts on the illegal combination.
        // Two-level dispatch handshake (same-rail gateway aggregation):
        // inter-node route entries and count rows are staged on the local
        // same-rail gateway over NVLink, which forwards them as a few bulk
        // WRITEs plus one trailing 1KB manifest.  First increment supports
        // exactly two nodes and the expert-ready dispatch protocol.
        // 0 = off, 1 = alpha (aggregation only, keeps grid_sync),
        // 2 = beta (eager send-when-full triggers, removes grid_sync),
        // 3 = beta + V3 (whole collect box shipped as one WRITE into the
        //     remote landing zone; pull reads inter entries from there).
        // V3's bulk WRITE size scales with the instance's max tokens per
        // rank, so it is a small-instance path by construction.  Default:
        // V3 on decode-shaped instances (capacity <= 512 tokens/rank,
        // measured net win across b1-256 on two nodes), classic path
        // otherwise — this IS the design doc's launch-time path selection.
        // Explicit env always overrides.
        const bool gateway_default_capable =
            args.num_ranks == 2 * static_cast<uint32_t>(kNvlPeers) and
            args.dispatch_expert_ready and
            args.num_max_tokens_per_rank <= 512;
        const int dispatch_gateway = get_env<int>(
            "DG_MEGA_MOE_DISPATCH_GATEWAY",
            gateway_default_capable ? 3 : 0);
        DG_HOST_ASSERT(dispatch_gateway >= 0 and dispatch_gateway <= 3);
        DG_HOST_ASSERT(dispatch_gateway == 0 or
                       args.num_ranks == 2 * static_cast<uint32_t>(kNvlPeers));
        DG_HOST_ASSERT(dispatch_gateway == 0 or args.dispatch_expert_ready);
        // Correctness never depends on this bound (cell capacity equals the
        // instance's max tokens); it only guards against pointlessly large
        // bulk WRITEs.  Production-shape V3 instances should be <= 512.
        DG_HOST_ASSERT(dispatch_gateway < 3 or
                       args.num_max_tokens_per_rank <= 4096);
        const bool chain_poll_incompatible =
            publisher_max_active != 0 or
            async_publisher_row_doorbell or
            async_stage_local_rows or
            get_env<int>("DG_MEGA_MOE_PHASE_PROFILE", 0) != 0;
        const bool publisher_chain_poll =
            get_env<int>("DG_MEGA_MOE_PUBLISHER_CHAIN_POLL",
                         (async_publisher and publisher_dst_grouped and
                          not chain_poll_incompatible) ? 1 : 0) != 0;
        DG_HOST_ASSERT(not publisher_chain_poll or publisher_dst_grouped);
        DG_HOST_ASSERT(
            not publisher_chain_poll or not chain_poll_incompatible);
        if (merge_ab_loader)
            internode_prefix += "#define DG_MEGA_MOE_MERGE_AB_LOADER 1\n";
        if (async_publisher)
            internode_prefix += "#define DG_MEGA_MOE_ASYNC_PUBLISHER 1\n";
        if (async_stage_local_rows)
            internode_prefix +=
                "#define DG_MEGA_MOE_ASYNC_STAGE_LOCAL_ROWS 1\n";
        if (async_publisher_row_doorbell)
            internode_prefix +=
                "#define DG_MEGA_MOE_ASYNC_PUBLISHER_ROW_DOORBELL 1\n";
        if (publisher_max_active != 0)
            internode_prefix += fmt::format(
                "#define DG_MEGA_MOE_PUBLISHER_MAX_ACTIVE {}\n",
                publisher_max_active);
        if (qp_inflight_wqes != 0)
            internode_prefix += fmt::format(
                "#define DG_MEGA_MOE_QP_INFLIGHT_WQES {}\n",
                qp_inflight_wqes);
        if (publisher_rate_mbps != 0)
            internode_prefix += fmt::format(
                "#define DG_MEGA_MOE_PUBLISHER_RATE_MBPS {}\n",
                publisher_rate_mbps);
        if (publisher_dst_rate_mbps != 0)
            internode_prefix += fmt::format(
                "#define DG_MEGA_MOE_PUBLISHER_DST_RATE_MBPS {}\n",
                publisher_dst_rate_mbps);
        if (publisher_dst_grouped)
            internode_prefix +=
                "#define DG_MEGA_MOE_PUBLISHER_DST_GROUPED 1\n";
        if (publisher_chain_poll)
            internode_prefix +=
                "#define DG_MEGA_MOE_PUBLISHER_CHAIN_POLL 1\n";
        if (dispatch_gateway >= 1)
            internode_prefix +=
                "#define DG_MEGA_MOE_DISPATCH_GATEWAY 1\n";
        if (dispatch_gateway >= 2)
            internode_prefix +=
                "#define DG_MEGA_MOE_DISPATCH_GATEWAY_EAGER 1\n";
        if (dispatch_gateway >= 3)
            internode_prefix +=
                "#define DG_MEGA_MOE_DISPATCH_GATEWAY_V3 1\n";
        const bool combine_batch_doorbell =
            get_env<int>("DG_MEGA_MOE_COMBINE_BATCH_DOORBELL", 0) != 0;
        if (combine_batch_doorbell)
            internode_prefix += "#define DG_MEGA_MOE_COMBINE_BATCH_DOORBELL 1\n";
        const int combine_compact_batch_rows =
            get_env<int>("DG_MEGA_MOE_COMBINE_COMPACT_BATCH_ROWS", 0);
        DG_HOST_ASSERT(
            combine_compact_batch_rows == 0 or
            combine_compact_batch_rows == 8 or
            combine_compact_batch_rows == 16);
        DG_HOST_ASSERT(
            combine_compact_batch_rows == 0 or combine_batch_doorbell);
        if (combine_compact_batch_rows != 0)
            internode_prefix += fmt::format(
                "#define DG_MEGA_MOE_COMBINE_COMPACT_BATCH_ROWS {}\n",
                combine_compact_batch_rows);
        if (get_env<int>("DG_MEGA_MOE_RANK_MAJOR_POOL", 0) != 0)
            internode_prefix += "#define DG_MEGA_MOE_RANK_MAJOR_POOL 1\n";
        const auto expert_schedule =
            get_env<std::string>("DG_MEGA_MOE_EXPERT_SCHEDULE", "id");
        bool expert_schedule_enabled = false;
        if (expert_schedule == "load") {
            expert_schedule_enabled = true;
            internode_prefix +=
                "#define DG_MEGA_MOE_EXPERT_SCHEDULE_PRIORITY 1\n"
                "#define DG_MEGA_MOE_EXPERT_SCHEDULE_LOAD 1\n";
        } else if (expert_schedule == "remote_load") {
            expert_schedule_enabled = true;
            internode_prefix +=
                "#define DG_MEGA_MOE_EXPERT_SCHEDULE_PRIORITY 1\n"
                "#define DG_MEGA_MOE_EXPERT_SCHEDULE_REMOTE_LOAD 1\n";
        } else if (expert_schedule == "wave_load") {
            expert_schedule_enabled = true;
            internode_prefix +=
                "#define DG_MEGA_MOE_EXPERT_SCHEDULE_PRIORITY 1\n"
                "#define DG_MEGA_MOE_EXPERT_SCHEDULE_LOAD 1\n"
                "#define DG_MEGA_MOE_EXPERT_SCHEDULE_WITHIN_WAVE 1\n";
        } else if (expert_schedule == "wave_remote_load") {
            expert_schedule_enabled = true;
            internode_prefix +=
                "#define DG_MEGA_MOE_EXPERT_SCHEDULE_PRIORITY 1\n"
                "#define DG_MEGA_MOE_EXPERT_SCHEDULE_REMOTE_LOAD 1\n"
                "#define DG_MEGA_MOE_EXPERT_SCHEDULE_WITHIN_WAVE 1\n";
        } else if (expert_schedule == "staged_remote") {
            const int prefix_experts =
                get_env<int>("DG_MEGA_MOE_EXPERT_PREFIX", 2);
            const int remote_min_tokens =
                get_env<int>("DG_MEGA_MOE_EXPERT_REMOTE_MIN_TOKENS", 2);
            const int schedule_max_tokens =
                get_env<int>("DG_MEGA_MOE_EXPERT_SCHEDULE_MAX_TOKENS", 16);
            DG_HOST_ASSERT(prefix_experts >= 0 and prefix_experts <= 32);
            DG_HOST_ASSERT(remote_min_tokens > 0);
            DG_HOST_ASSERT(schedule_max_tokens > 0);
            if (args.num_tokens <= schedule_max_tokens) {
                expert_schedule_enabled = true;
                internode_prefix += fmt::format(
                    "#define DG_MEGA_MOE_EXPERT_SCHEDULE_PRIORITY 1\n"
                    "#define DG_MEGA_MOE_EXPERT_SCHEDULE_STAGED_REMOTE 1\n"
                    "#define DG_MEGA_MOE_EXPERT_PREFIX {}\n"
                    "#define DG_MEGA_MOE_EXPERT_REMOTE_MIN_TOKENS {}\n",
                    prefix_experts, remote_min_tokens);
            }
        } else {
            DG_HOST_ASSERT(expert_schedule == "id" and
                           "Invalid DG_MEGA_MOE_EXPERT_SCHEDULE");
        }
        const int decode_active_max_tokens =
            get_env<int>("DG_MEGA_MOE_DECODE_ACTIVE_MAX_TOKENS", 0);
        DG_HOST_ASSERT(decode_active_max_tokens >= 0);
        const bool decode_active_experts =
            decode_active_max_tokens > 0 and
            args.num_tokens <= decode_active_max_tokens;
        if (decode_active_experts) {
            // The first decode-specialized decomposition deliberately keeps
            // the full persistent grid.  It reuses the existing eager count
            // path and only compacts empty experts out of the deterministic
            // dispatch/GEMM schedule.
            DG_HOST_ASSERT(args.num_ranks > kNvlPeers);
            DG_HOST_ASSERT(args.dispatch_expert_ready);
            DG_HOST_ASSERT(not args.lazy_expert_count);
            DG_HOST_ASSERT(not expert_schedule_enabled);
            DG_HOST_ASSERT(args.num_experts / args.num_ranks <= 32);
            internode_prefix +=
                "#define DG_MEGA_MOE_DECODE_ACTIVE_EXPERTS 1\n"
                "#define DG_MEGA_MOE_DISPATCH_FOLLOW_EXPERT_SCHEDULE 1\n";
        }
        if (expert_schedule_enabled and
            get_env<int>("DG_MEGA_MOE_DISPATCH_FOLLOW_EXPERT_SCHEDULE", 0) != 0)
            internode_prefix +=
                "#define DG_MEGA_MOE_DISPATCH_FOLLOW_EXPERT_SCHEDULE 1\n";
        if (get_env<int>("DG_MEGA_MOE_COMBINE_PARALLEL_READY", 1) != 0)
            internode_prefix += "#define DG_MEGA_MOE_COMBINE_PARALLEL_READY 1\n";
        return internode_prefix + fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)",
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.epilogue_registers,
    args.reuse_accum_as_final ? "true" : "false",
    args.l2_arrival_counter ? "true" : "false",
    args.l2_epilogue_requires_full_sync ? "true" : "false",
    args.split_phase_hot_path ? "true" : "false",
    args.dispatch_expert_ready ? "true" : "false",
    args.lazy_expert_count ? "true" : "false",
    args.combine_full_row ? "true" : "false",
    args.combine_expert_ready ? "true" : "false",
    args.use_swap_ab ? "true" : "false",
    args.fuse_shared_expert ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l2_weights_sf,
            args.tensor_map_shared_l1_acts,
            args.shared_l1_acts_sf,
            args.tensor_map_shared_l1_weights,
            args.shared_l1_weights_sf,
            args.tensor_map_shared_l2_weights,
            args.shared_l2_weights_sf
        ));
    }
};

static void sm90_fp8_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& shared_l1_acts,
    const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l1_weights,
    const torch::Tensor& shared_l1_weights_sf,
    const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l2_weights_sf,
    const bool& fuse_shared_expert,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& dispatch_expert_ready,
    const bool& lazy_expert_count,
    const bool& combine_full_row,
    const bool& combine_expert_ready
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Heuristics
    const auto config = get_mega_moe_config_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens);
    DG_HOST_ASSERT(not fuse_shared_expert or num_tokens <= config.block_m);
    const int default_epilogue_registers =
        config.num_epilogue_threads == 512 ? 112 : 0;
    const int epilogue_registers = default_epilogue_registers;
    if (epilogue_registers > 0) {
        const int dispatch_registers =
            config.num_epilogue_threads == 512 ? 32 : 48;
        const int non_epilogue_registers =
            config.num_epilogue_threads == 512 ? 24 : 40;
        DG_HOST_ASSERT(dispatch_registers * config.num_dispatch_threads +
                       non_epilogue_registers * config.num_non_epilogue_threads +
                       epilogue_registers * config.num_epilogue_threads <= 64512);
    }
    const bool reuse_accum_as_final = config.block_m == 128;
    const bool default_split_mn_barrier_opt =
        config.block_m == 128 and config.block_n == 256 and
        config.num_epilogue_threads == 512;
    const bool split_phase_hot_path =
        config.block_m == 128 and config.block_n == 256 and hidden >= 7168;
    const bool decode_split_n_path =
        config.block_m == 64 and config.num_epilogue_threads == 256;
    const bool decode_split_n_bn256 =
        decode_split_n_path and config.block_n == 256;
    const bool decode_l2_counter =
        decode_split_n_bn256 and num_tokens >= 4 and num_tokens <= 128;
    const bool l2_arrival_counter =
        default_split_mn_barrier_opt or decode_l2_counter;
    const bool l2_epilogue_requires_full_sync =
        not l2_arrival_counter;
    const bool use_swap_ab = should_use_swap_ab_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        config.block_m, config.num_epilogue_threads);

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-64 for L2 (MN-major, no swizzle).
    // Weight SF: block (128, 128) raw float pointer (no TMA descriptor).
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 64;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.block_m, kGranK,
                                                        1, 0);
    const int weight_tma_block_n = config.block_n > 256 ? 256 : config.block_n;
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, weight_tma_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    // L1 output (post-SwiGLU FP8): N is halved. The correctness path stages
    // this tile in plain row-major SMEM before the TMA store. Later L2 TMA
    // loads may still swizzle from this row-major global buffer into their own
    // SMEM tile.
    // The usual TMA store is issued per warpgroup, each writing a `WG_BLOCK_M`
    // row tile from its own SMEM offset. The m64n128 2-WG split-N decode path is
    // different: both warpgroups stage one joint 64-column L1-output tile and a
    // single warpgroup issues the combined store, so the descriptor must cover
    // the full block_m x (block_n / 2) tile.
    const int num_epilogue_warpgroups_h = config.num_epilogue_threads / 128;
    const bool split_n_warpgroups =
        config.block_m == 64 and num_epilogue_warpgroups_h > 1 and
        config.block_n % num_epilogue_warpgroups_h == 0 and
        (config.block_n / num_epilogue_warpgroups_h == 64 or
         config.block_n / num_epilogue_warpgroups_h == 128);
    const bool split_mn_warpgroups =
        config.block_m == 128 and config.block_n == 256 and num_epilogue_warpgroups_h == 4;
    const int wg_split_m = split_n_warpgroups ? 1 :
        (split_mn_warpgroups ? 2 : num_epilogue_warpgroups_h);
    const int wg_split_n = split_n_warpgroups ? num_epilogue_warpgroups_h :
        (split_mn_warpgroups ? 2 : 1);
    DG_HOST_ASSERT(wg_split_m * wg_split_n == num_epilogue_warpgroups_h);
    const int wg_block_m = config.block_m / wg_split_m;
    const int wg_block_n = config.block_n / wg_split_n;
    const int wg_l1_out_block_n = wg_block_n / 2;
    const bool split_n_shares_sf =
        split_n_warpgroups and wg_l1_out_block_n < kL2ActsSFGranK;
    const int l1_output_swizzle_mode = 0;
    const int l1_output_box_n =
        split_n_shares_sf ? config.block_n / 2 : wg_l1_out_block_n;
    const int l1_output_box_m =
        split_n_shares_sf ? config.block_m : wg_block_m;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       l1_output_box_n, l1_output_box_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       l1_output_swizzle_mode);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, weight_tma_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_shared_l1_acts = make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        config.block_k, config.block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode);
    const auto tensor_map_shared_l1_weights = make_tma_2d_desc(
        shared_l1_weights,
        hidden, intermediate_hidden * 2,
        config.block_k, weight_tma_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode);
    const auto tensor_map_shared_l2_weights = make_tma_2d_desc(
        shared_l2_weights,
        intermediate_hidden, hidden,
        config.block_k, weight_tma_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM90FP8MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .epilogue_registers = epilogue_registers,
        .reuse_accum_as_final = reuse_accum_as_final,
        .l2_arrival_counter = l2_arrival_counter,
        .l2_epilogue_requires_full_sync = l2_epilogue_requires_full_sync,
        .split_phase_hot_path = split_phase_hot_path,
        .dispatch_expert_ready = dispatch_expert_ready,
        .lazy_expert_count = lazy_expert_count,
        .combine_full_row = combine_full_row,
        .combine_expert_ready = combine_expert_ready,
        .use_swap_ab = use_swap_ab,
        .fuse_shared_expert = fuse_shared_expert,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .l1_weights_sf = l1_weights_sf.data_ptr<float>(),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_weights_sf = l2_weights_sf.data_ptr<float>(),
        .tensor_map_shared_l1_acts = tensor_map_shared_l1_acts,
        .shared_l1_acts_sf = shared_l1_acts_sf.data_ptr<float>(),
        .tensor_map_shared_l1_weights = tensor_map_shared_l1_weights,
        .shared_l1_weights_sf = shared_l1_weights_sf.data_ptr<float>(),
        .tensor_map_shared_l2_weights = tensor_map_shared_l2_weights,
        .shared_l2_weights_sf = shared_l2_weights_sf.data_ptr<float>(),
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_mega_moe", code);
    SM90FP8MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
