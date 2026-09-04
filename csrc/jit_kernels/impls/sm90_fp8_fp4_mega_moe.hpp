#pragma once

#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../apis/sm90_mega_cpu_proxy.hpp"
#include "runtime_utils.hpp"

#include <array>
#include <mutex>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 x FP4 MegaMoE host runtime
// ----------------------------------------------------------------------------
// Counterpart of `SM90FP8MegaMoERuntime` with these differences:
//   * L1/L2 weights are packed E2M1 (FP4): each storage byte holds 2 nibbles
//     (low nibble = even K, high nibble = odd K). Host code builds the TMA
//     descriptors from a byte view of the packed tensors, so TensorMap sees
//     dense bytes while the kernel interprets each byte as two FP4 elements
//     in the `b_packed_dtype_t = int8_t` SMEM tile.
//   * Weight scale factors switch from per-128 K float to per-32 K UE8M0,
//     packed as int32 along K (4 bytes = 4 K-groups = BLOCK_K=128 K-cols
//     for one N-row). They are passed as `const uint32_t*` instead of
//     `const float*`. The kernel reads them via `__ldg` from global, so
//     no TMA descriptor is required.
//   * Activation side is identical to the FP8 path: FP8 e4m3 K-major with
//     128B swizzle, per-128 K float SFA for L1 and per-64 K float SFA for
//     L2 (filled by the L1 epilogue's per-token SwiGLU+quant).
//   * The kernel applies the per-32 SFB on the fly during dequant through a
//     constant-memory UE8M0->E4M3 LUT, so the only SF that the promote loop
//     still applies is SFA. There is no `weight_sf` ldg in the math warpgroup
//     beyond the SFB UE8M0 word.
// ============================================================================

class SM90FP8FP4MegaMoERuntime final : public LaunchRuntime<SM90FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int requested_num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        int num_l1_ring_tokens, num_l1_sf_storage_tokens;
        int num_l2_ring_tokens, num_l2_sf_storage_tokens;
        float activation_clamp;
        bool fast_math;
        // Read the four packed FP4 words for one K/32 group with a
        // single wide shared load while keeping the default work partition.
        bool use_wide_load_decode;
        // Overlap FP4 decode with WGMMA. When false, the math
        // warpgroup only waits on the decode barrier; non-epilogue warps do the
        // decode work and can run ahead through pipeline stages.
        bool math_wg_participates_in_fp4_decode;
        // Limit how many warps inside the math warpgroup help decode.
        // This keeps CTA size fixed while reducing math-side non-tensor-core
        // work that can interfere with WGMMA issue.
        int num_math_wg_decode_warps;
        // Skip early non-epilogue warps as FP4 decode helpers.
        // 0 keeps the existing 4 assist warps; 2 skips the two TMA loader
        // warps; 4 leaves all decode work to the math warpgroup.
        int first_fp4_decode_assist_warp;
        // Split packed-B readiness from the A+B full barrier so the
        // assist warps can start FP4 decode while A/SFA TMA is still in flight.
        bool use_early_b_decode;
        // Replace the FP4 decode rendezvous sync with a per-stage
        // mbarrier so assist warps can run ahead after publishing a decoded tile.
        bool use_decode_done_mbarrier;
        // Mirror the FP8 split-MN arrival-counter path for FP4 L1->L2
        // readiness, avoiding the bitmask update's CTA-wide epilogue sync.
        bool use_l2_arrival_counter;
        // Split each SS N=128 WGMMA into two N=64 WGMMAs so the
        // per-K-block accumulator is 32 floats instead of 64. Large-token SS
        // shapes enable this to reduce accumulator pressure while keeping SS
        // scheduling.
        bool use_ss_nsplit;
        // swapAB path: use decoded weight as WGMMA-M and tokens as WGMMA-N.
        bool use_swap_ab;
        bool use_swap_ab_fast_amax;
        MegaMoESM90Config config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;
        mega::SM90MegaMoECPUProxyLaunch cpu_proxy;
        int sidecar_num_blocks;

        // Tensormaps for activations (FP8) and packed FP4 weights.
        // Weight UE8M0 SFB are passed as raw uint32* (no TMA descriptor).
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        const uint32_t* l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const uint32_t* l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // Inter-node build: ranks span more than one NVLink domain.  Inject
        // the same macros as the FP8 runtime so barrier.cuh (and the ported
        // dispatch/combine branches) take the NVSHMEM remote path.  The
        // `nvshmem` mention in the comment also makes the JIT compiler
        // device-link libnvshmem_device.
        constexpr int kNvlPeers = 8;
        // The shared implementation header now always parses the combine-ring
        // type, whose IBGDA helper includes NVSHMEM device headers even when a
        // <=8-rank specialization compiles every remote branch away.  Keep the
        // marker unconditional so the JIT compiler supplies those include and
        // device-link flags for single-node FP4 as well.
        std::string internode_prefix =
            "// sm90 fp4 mega-moe protocol revision 3\n"
            "// sm90 fp4 mega-moe support uses nvshmem device helpers\n";
        if (args.num_ranks > kNvlPeers) {
            internode_prefix += fmt::format(
                "// inter-node mega-moe: uses nvshmem device functions\n"
                "#define DG_MEGA_MOE_INTERNODE\n"
                "#define DG_MEGA_MOE_NVL_PEERS {}\n", kNvlPeers);
            const bool use_cpu_proxy = get_env<int>(
                "DG_MEGA_MOE_FP4_CPU_PROXY", 0) != 0;
            const bool use_sidecar_publisher = get_env<int>(
                "DG_MEGA_MOE_FP4_SIDECAR_PUBLISHER", 0) != 0;
            DG_HOST_ASSERT(not (use_cpu_proxy and use_sidecar_publisher));
            if (use_cpu_proxy) {
                DG_HOST_ASSERT(
                    mega::sm90_mega_moe_cpu_proxy_is_initialized());
                internode_prefix +=
                    "#define DG_MEGA_MOE_FP4_CPU_PROXY 1\n";
                if (get_env<int>(
                        "DG_MEGA_MOE_FP4_CPU_PROXY_ASYNC_CREDIT", 0) != 0)
                    internode_prefix +=
                        "#define "
                        "DG_MEGA_MOE_FP4_CPU_PROXY_ASYNC_CREDIT 1\n";
            }
            if (use_sidecar_publisher) {
                const int combine_stage_tokens =
                    layout::get_num_sm90_combine_ring_tokens(
                        args.num_ranks, args.num_max_tokens_per_rank,
                        args.num_topk,
                        args.num_experts / args.num_ranks);
                DG_HOST_ASSERT(args.num_ranks == 16);
                DG_HOST_ASSERT(args.num_experts / args.num_ranks <= 32);
                DG_HOST_ASSERT(
                    combine_stage_tokens == args.config.num_max_pool_tokens);
                internode_prefix +=
                    "#define DG_MEGA_MOE_FP4_SIDECAR_PUBLISHER 1\n";
                const bool sidecar_dispatch_rdma = get_env<int>(
                    "DG_MEGA_MOE_FP4_SIDECAR_DISPATCH_RDMA", 0) != 0;
                if (sidecar_dispatch_rdma)
                    internode_prefix +=
                        "#define "
                        "DG_MEGA_MOE_FP4_SIDECAR_DISPATCH_RDMA 1\n";
                const bool aggregate_local = get_env<int>(
                    "DG_MEGA_MOE_FP4_SIDECAR_AGGREGATE_LOCAL", 0) != 0;
                const bool expert_centric = get_env<int>(
                    "DG_MEGA_MOE_FP4_SIDECAR_EXPERT_CENTRIC", 0) != 0;
                DG_HOST_ASSERT(
                    not sidecar_dispatch_rdma or
                    (not aggregate_local and not expert_centric));
                const int shared_metadata_mode = get_env<int>(
                    "DG_MEGA_MOE_FP4_SIDECAR_SHARED_METADATA", -1);
                DG_HOST_ASSERT(
                    shared_metadata_mode >= -1 and
                    shared_metadata_mode <= 1);
                const bool shared_metadata =
                    shared_metadata_mode == 1 or
                    (shared_metadata_mode == -1 and
                     not aggregate_local and not expert_centric and
                     args.num_tokens >= 8);
                DG_HOST_ASSERT(
                    static_cast<int>(aggregate_local) +
                    static_cast<int>(expert_centric) +
                    static_cast<int>(shared_metadata) <= 1);
                if (aggregate_local)
                    internode_prefix +=
                        "#define "
                        "DG_MEGA_MOE_FP4_SIDECAR_AGGREGATE_LOCAL 1\n";
                if (shared_metadata)
                    internode_prefix +=
                        "#define "
                        "DG_MEGA_MOE_FP4_SIDECAR_SHARED_METADATA 1\n";
                if (expert_centric) {
                    const int expert_peer_groups = get_env<int>(
                        "DG_MEGA_MOE_FP4_SIDECAR_EXPERT_PEER_GROUPS", 2);
                    DG_HOST_ASSERT(
                        expert_peer_groups == 1 or
                        expert_peer_groups == 2 or
                        expert_peer_groups == 4);
                    internode_prefix +=
                        "#define "
                        "DG_MEGA_MOE_FP4_SIDECAR_EXPERT_CENTRIC 1\n";
                    internode_prefix += fmt::format(
                        "#define "
                        "DG_MEGA_MOE_FP4_SIDECAR_EXPERT_PEER_GROUPS {}\n",
                        expert_peer_groups);
                }
            }
            const int dispatch_gateway =
                args.requested_num_max_tokens_per_rank <=
                    layout::kGatewayDenseMaxRequestedTokens ? 4 : 3;
            if (dispatch_gateway == 3)
                internode_prefix +=
                    "#define DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED 1\n";
            else {
                internode_prefix +=
                    "#define DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3 1\n";
                // Only the prefix that can own input tokens participates in
                // dense metadata production.  The rest of the grid remains
                // available to pull/GEMM/scatter without paying empty
                // expert stake-out, system-fence and direction-counter work.
                const int tokens_per_metadata_cta =
                    (args.config.num_dispatch_threads / 32) *
                    (32 / args.num_topk);
                const int num_metadata_sms = std::min(
                    args.launch_args.grid_dim.first,
                    std::max(1,
                             (args.num_tokens + tokens_per_metadata_cta - 1) /
                                 tokens_per_metadata_cta));
                internode_prefix += fmt::format(
                    "#define DG_MEGA_MOE_NUM_METADATA_SMS {}\n",
                    num_metadata_sms);
            }

            // A hot async-publisher spin competes with FP4 decode/GEMM for
            // issue/cache resources when a compact expert set waits for the
            // next output tile. A wide expert set already gets a natural
            // backoff from its longer scan, while very sparse or dense work
            // is response-latency sensitive. Select the A-B-A validated
            // density window at JIT time so inactive shapes compile the
            // original eager polling loop without runtime branching.
            const int num_experts_per_rank =
                args.num_experts / args.num_ranks;
            const float expected_rows_per_local_expert =
                static_cast<float>(args.num_tokens) * args.num_topk /
                num_experts_per_rank;
            constexpr int64_t kPublisherBackoffMaxWeightElems =
                16ll * 1024 * 1024;
            const bool weight_light =
                static_cast<int64_t>(args.hidden) *
                    args.intermediate_hidden <
                kPublisherBackoffMaxWeightElems;
            const float max_backoff_rows_per_expert =
                weight_light ? 16.0f : 8.0f;
            const bool auto_publisher_idle_backoff =
                num_experts_per_rank <= 32 and
                expected_rows_per_local_expert >= 1.0f and
                expected_rows_per_local_expert <=
                    max_backoff_rows_per_expert;
            // Diagnostic override for isolating the outer ready-polling loop:
            // 0 keeps the density heuristic, 1 forces eager polling, and 2
            // forces adaptive idle backoff for every inter-node shape.
            // A sidecar runs on reserved SMs, so sleeping no longer protects
            // math/decode issue bandwidth and only adds publish latency.
            // Default it to eager polling while preserving the diagnostic
            // override and the fused-kernel density heuristic.
            const int publisher_backoff_mode = get_env<int>(
                "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MODE",
                use_sidecar_publisher ? 1 : 0);
            DG_HOST_ASSERT(
                publisher_backoff_mode >= 0 and
                publisher_backoff_mode <= 2);
            const bool use_publisher_idle_backoff =
                publisher_backoff_mode == 2 or
                (publisher_backoff_mode == 0 and
                 auto_publisher_idle_backoff);
            const int use_publish_row_mask = get_env<int>(
                "DG_MEGA_MOE_FP4_PUBLISH_ROW_MASK", 0);
            DG_HOST_ASSERT(
                use_publish_row_mask == 0 or use_publish_row_mask == 1);
            if (use_publish_row_mask != 0)
                internode_prefix +=
                    "#define DG_MEGA_MOE_FP4_PUBLISH_ROW_MASK 1\n";
            if (use_publisher_idle_backoff) {
                const int publisher_backoff_initial_ns = get_env<int>(
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS", 64);
                const int publisher_backoff_max_ns = get_env<int>(
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MAX_NS", 512);
                const int publisher_few_pending_chains = get_env<int>(
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_FEW_PENDING_CHAINS", 0);
                const int publisher_few_pending_max_ns = get_env<int>(
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_FEW_PENDING_MAX_NS",
                    publisher_backoff_max_ns);
                const int publisher_long_idle_threshold = get_env<int>(
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD", 0);
                const int publisher_long_idle_max_ns = get_env<int>(
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_MAX_NS",
                    publisher_backoff_max_ns);
                const int publisher_progress_block_budget = get_env<int>(
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_PROGRESS_BLOCK_BUDGET", 0);
                const int publisher_progress_yield_ns = get_env<int>(
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_PROGRESS_YIELD_NS", 64);
                DG_HOST_ASSERT(publisher_backoff_initial_ns > 0 and
                               publisher_backoff_max_ns >=
                                   publisher_backoff_initial_ns and
                               publisher_backoff_max_ns <= 1000000 and
                               publisher_few_pending_chains >= 0 and
                               publisher_few_pending_chains <= 32 and
                               publisher_few_pending_max_ns >=
                                   publisher_backoff_initial_ns and
                               publisher_few_pending_max_ns <= 1000000 and
                               publisher_long_idle_threshold >= 0 and
                               publisher_long_idle_threshold <= 1000000 and
                               publisher_long_idle_max_ns >=
                                   publisher_backoff_max_ns and
                               publisher_long_idle_max_ns <= 1000000 and
                               publisher_progress_block_budget >= 0 and
                               publisher_progress_block_budget <= 1024 and
                               publisher_progress_yield_ns > 0 and
                               publisher_progress_yield_ns <= 1000000);
                internode_prefix += fmt::format(
                    "#define "
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS {}\n"
                    "#define "
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MAX_NS {}\n"
                    "#define "
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_FEW_PENDING_CHAINS {}\n"
                    "#define "
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_FEW_PENDING_MAX_NS {}\n"
                    "#define "
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD {}\n"
                    "#define "
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_MAX_NS {}\n",
                    publisher_backoff_initial_ns,
                    publisher_backoff_max_ns,
                    publisher_few_pending_chains,
                    publisher_few_pending_max_ns,
                    publisher_long_idle_threshold,
                    publisher_long_idle_max_ns);
                if (publisher_progress_block_budget > 0)
                    internode_prefix += fmt::format(
                        "#define "
                        "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_PROGRESS_BLOCK_BUDGET {}\n"
                        "#define "
                        "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_PROGRESS_YIELD_NS {}\n",
                        publisher_progress_block_budget,
                        publisher_progress_yield_ns);
            }
            // Four neighboring decode lanes consume the four K groups of
            // one N row and share its packed SFB word.  At middle density the
            // work per block amortizes one subgroup shuffle and benefits from
            // replacing four shared loads with one.  Very sparse work remains
            // shuffle-latency sensitive, while the shared M128 crossover
            // switches execution topology and showed no benefit.  Use the
            // same weight-footprint boundary as the other FP4 heuristics.
            const float min_sfb_broadcast_rows = weight_light ? 24.0f : 16.0f;
            const bool use_sfb_subgroup_broadcast =
                expected_rows_per_local_expert >= min_sfb_broadcast_rows and
                expected_rows_per_local_expert <= 32.0f;
            if (use_sfb_subgroup_broadcast)
                internode_prefix +=
                    "#define DG_MEGA_MOE_FP4_SFB_SUBGROUP_BROADCAST 1\n";
            // The generic swapAB path normally promotes 17--24 valid expert
            // rows to N=32.  Use the native N=24 WGMMA bucket only in the
            // A-B-A validated middle-density window: below it, the extra
            // specialization does not amortize its instruction footprint;
            // above it, routing tails made N24 slower for the tested shapes.
            const bool use_fp4_swap_ab_n24 =
                expected_rows_per_local_expert >= 16.0f and
                expected_rows_per_local_expert <= 24.0f;
            if (use_fp4_swap_ab_n24)
                internode_prefix +=
                    "#define DG_MEGA_MOE_FP4_SWAP_AB_N24 1\n";
            // Preserve Hopper's native 8-column WGMMA granularity above
            // N=32 instead of promoting every 33--64 row expert tile directly
            // to N=64.  Select how many extra buckets are compiled from the
            // A-B-A validated Flash/Pro/Kimi density and weight-footprint
            // bands, keeping unnecessary template bodies out of the hot loop.
            int fp4_swap_ab_fine_bucket_level = 0;
            if (args.intermediate_hidden <= 2048 and
                expected_rows_per_local_expert >= 24.0f and
                expected_rows_per_local_expert <= 48.0f) {
                fp4_swap_ab_fine_bucket_level =
                    expected_rows_per_local_expert <= 32.0f ? 3 : 2;
            }
            if (args.intermediate_hidden >= 3072 and
                expected_rows_per_local_expert >= 24.0f and
                expected_rows_per_local_expert <= 48.0f) {
                fp4_swap_ab_fine_bucket_level =
                    weight_light ? 2 :
                    expected_rows_per_local_expert < 28.0f ? 1 : 2;
            }
            if (fp4_swap_ab_fine_bucket_level > 0)
                internode_prefix += fmt::format(
                    "#define DG_MEGA_MOE_FP4_SWAP_AB_FINE_BUCKETS {}\n",
                    fp4_swap_ab_fine_bucket_level);
            // The swapAB promotion loop consumes adjacent token scales in
            // pairs.  A single aligned 64-bit shared load reduces the L1/L2
            // promotion instruction stream without changing predicates or
            // resources.  A-B-A showed a stable win in the 24--48-row band;
            // sparse shapes do not amortize the wider load.
            const bool use_swap_scale_float2 =
                expected_rows_per_local_expert >= 24.0f and
                expected_rows_per_local_expert <= 48.0f;
            if (use_swap_scale_float2)
                internode_prefix +=
                    "#define DG_MEGA_MOE_FP4_SWAP_SCALE_FLOAT2_LEVEL 3\n";
        }
        if (get_env<int>("DG_MEGA_MOE_PHASE_PROFILE", 0) != 0)
            internode_prefix += "#define DG_MEGA_MOE_PHASE_PROFILE 1\n";
        // Match the FP8 production path: textual device diagnostics pull
        // vprintf and a call stack into every specialization, even though
        // they are only reachable after a protocol failure. Keep all traps
        // in production and make the text opt-in for debugging.
        const bool device_diagnostics =
            get_env<int>("DG_MEGA_MOE_DEVICE_DIAGNOSTICS", 0) != 0;
        if (device_diagnostics)
            internode_prefix +=
                "#define DG_MEGA_MOE_DEVICE_DIAGNOSTICS 1\n";
        else if (args.num_ranks > kNvlPeers)
            internode_prefix +=
                "#define DG_DEVICE_ASSERT_TRAP_ONLY 1\n";
        if (get_env<int>("DG_MEGA_MOE_PHASE_PROFILE_SILENT", 0) != 0)
            internode_prefix +=
                "#define DG_MEGA_MOE_PHASE_PROFILE_SILENT 1\n";
        const int actual_pool_tokens = layout::get_num_max_pool_tokens(
            args.num_ranks, args.num_tokens, args.num_topk,
            args.num_experts / args.num_ranks);
        const bool l1_ring_active =
            args.num_l1_ring_tokens < args.config.num_max_pool_tokens and
            actual_pool_tokens > args.num_l1_ring_tokens;
        const bool l2_ring_active =
            args.num_l2_ring_tokens < args.config.num_max_pool_tokens and
            actual_pool_tokens > args.num_l2_ring_tokens;
        if (l1_ring_active)
            internode_prefix += "#define DG_MEGA_MOE_L1_RING_ACTIVE 1\n";
        if (l2_ring_active)
            internode_prefix += "#define DG_MEGA_MOE_L2_RING_ACTIVE 1\n";
        return internode_prefix + fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
#ifdef DG_MEGA_MOE_FP4_SIDECAR_ONLY
    auto ptr = reinterpret_cast<void*>(
        &sm90_fp8_fp4_mega_moe_sidecar_publisher<
            {},
            {}, {},
            {}, {},
            {}, {},
            {},
            {}, {}, {}, {}, {},
            {}
        >);
#else
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {}, {}, {}, {}, {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}
    >);
#endif
}};
)",
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.block_m, args.config.block_n,
    args.config.num_max_pool_tokens,
    args.num_l1_ring_tokens,
    args.num_l1_sf_storage_tokens,
    args.num_l2_ring_tokens,
    args.num_l2_sf_storage_tokens,
    args.num_ranks > kNvlPeers ?
        layout::get_num_sm90_combine_ring_tokens(
            args.num_ranks, args.num_max_tokens_per_rank, args.num_topk,
            args.num_experts / args.num_ranks) :
        args.config.num_max_pool_tokens,
    args.num_ranks,
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.num_l1_ring_tokens,
    args.num_l1_sf_storage_tokens,
    args.num_l2_ring_tokens,
    args.num_l2_sf_storage_tokens,
    args.num_ranks > kNvlPeers ?
        layout::get_num_sm90_combine_ring_tokens(
            args.num_ranks, args.num_max_tokens_per_rank, args.num_topk,
            args.num_experts / args.num_ranks) :
        args.config.num_max_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.use_wide_load_decode ? "true" : "false",
    args.math_wg_participates_in_fp4_decode ? "true" : "false",
    args.num_math_wg_decode_warps,
    args.first_fp4_decode_assist_warp,
    args.use_early_b_decode ? "true" : "false",
    args.use_decode_done_mbarrier ? "true" : "false",
    args.use_l2_arrival_counter ? "true" : "false",
    args.use_ss_nsplit ? "true" : "false",
    args.use_swap_ab ? "true" : "false",
    args.use_swap_ab_fast_amax ? "true" : "false");
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
            args.cpu_proxy.buffer,
            args.cpu_proxy.dev_comms,
            args.cpu_proxy.windows,
            args.cpu_proxy.combine_offset,
            args.cpu_proxy.staging_offset,
            args.cpu_proxy.signal_epoch_offset,
            args.cpu_proxy.completion_request_offset,
            args.cpu_proxy.combine_slot_bytes,
            args.cpu_proxy.staging_slot_bytes,
            args.cpu_proxy.num_slots
        ));
    }

    static cudaStream_t get_sidecar_stream() {
        constexpr int kMaxCUDADevices = 64;
        static std::array<cudaStream_t, kMaxCUDADevices> streams{};
        static std::mutex streams_mutex;
        int device_idx = 0;
        DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));
        DG_HOST_ASSERT(device_idx >= 0 and device_idx < kMaxCUDADevices);
        std::lock_guard<std::mutex> lock(streams_mutex);
        auto& stream = streams[device_idx];
        if (stream == nullptr) {
            int least_priority = 0;
            int greatest_priority = 0;
            DG_CUDA_RUNTIME_CHECK(cudaDeviceGetStreamPriorityRange(
                &least_priority, &greatest_priority));
            DG_CUDA_RUNTIME_CHECK(cudaStreamCreateWithPriority(
                &stream, cudaStreamNonBlocking, greatest_priority));
        }
        return stream;
    }

    static void launch_sidecar(
            const std::shared_ptr<KernelRuntime>& kernel_runtime,
            const Args& args) {
        const bool aggregate_local = get_env<int>(
            "DG_MEGA_MOE_FP4_SIDECAR_AGGREGATE_LOCAL", 0) != 0;
        const bool expert_centric = get_env<int>(
            "DG_MEGA_MOE_FP4_SIDECAR_EXPERT_CENTRIC", 0) != 0;
        const int shared_metadata_mode = get_env<int>(
            "DG_MEGA_MOE_FP4_SIDECAR_SHARED_METADATA", -1);
        DG_HOST_ASSERT(
            shared_metadata_mode >= -1 and shared_metadata_mode <= 1);
        const bool shared_metadata =
            shared_metadata_mode == 1 or
            (shared_metadata_mode == -1 and
             not aggregate_local and not expert_centric and
             args.num_tokens >= 8);
        DG_HOST_ASSERT(
            static_cast<int>(aggregate_local) +
            static_cast<int>(expert_centric) +
            static_cast<int>(shared_metadata) <= 1);
        const int expert_peer_groups = expert_centric ? get_env<int>(
            "DG_MEGA_MOE_FP4_SIDECAR_EXPERT_PEER_GROUPS", 2) : 1;
        DG_HOST_ASSERT(
            expert_peer_groups == 1 or
            expert_peer_groups == 2 or
            expert_peer_groups == 4);
        const int expert_warps_per_cta = math::ceil_div(
            args.num_experts / args.num_ranks,
            args.sidecar_num_blocks);
        const int sidecar_threads = expert_centric ?
            expert_warps_per_cta * expert_peer_groups * 32 :
            (aggregate_local ? 288 : 512);
        DG_HOST_ASSERT(sidecar_threads > 0 and sidecar_threads <= 512);
        constexpr int kSidecarDynamicSmemBytes = 128 * 1024;
        const auto kernel = kernel_runtime->kernel;
        const auto stream = get_sidecar_stream();
        const auto config = construct_launch_config(
            kernel, stream, kSidecarDynamicSmemBytes,
            dim3(args.sidecar_num_blocks), dim3(sidecar_threads), 1, false);
        auto sym_buffer = args.sym_buffer_ptrs;
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, sym_buffer));
    }
};

static void sm90_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& requested_num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& math_wg_participates_in_fp4_decode = false,
    const int& num_math_wg_decode_warps = 0,
    const int& first_fp4_decode_assist_warp = 0,
    const bool& use_wide_load_decode = false,
    const bool& use_early_b_decode = false,
    const bool& use_decode_done_mbarrier = false,
    const bool& use_l2_arrival_counter = false,
    const bool& use_ss_nsplit = false,
    const bool& use_swap_ab = false,
    const bool& use_swap_ab_fast_amax = false
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_l1_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_l1_sf_storage_tokens =
        static_cast<int>(l1_acts_sf.size(0));
    const auto num_l2_storage_tokens = static_cast<int>(l2_acts.size(0));
    const auto num_l2_sf_storage_tokens =
        static_cast<int>(l2_acts_sf.size(0));
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk,
        num_experts_per_rank);
    const bool l2_ring_enabled =
        num_l2_storage_tokens < num_max_pool_tokens;
    constexpr int kSharedScratchRows = 128;
    const int num_l2_ring_tokens = l2_ring_enabled ?
        num_l2_storage_tokens - kSharedScratchRows :
        num_max_pool_tokens;
    const int num_compute_ring_tokens = std::min(
        num_l1_ring_tokens, num_l2_ring_tokens);
    const bool use_cpu_proxy = num_ranks > 8 and
        get_env<int>("DG_MEGA_MOE_FP4_CPU_PROXY", 0) != 0;
    const bool use_sidecar_publisher = num_ranks > 8 and
        get_env<int>("DG_MEGA_MOE_FP4_SIDECAR_PUBLISHER", 0) != 0;
    DG_HOST_ASSERT(not (use_cpu_proxy and use_sidecar_publisher));
    const int sidecar_num_blocks = use_sidecar_publisher ?
        get_env<int>("DG_MEGA_MOE_FP4_SIDECAR_BLOCKS", 4) : 0;
    DG_HOST_ASSERT(
        not use_sidecar_publisher or
        (sidecar_num_blocks > 0 and
         sidecar_num_blocks <= num_experts_per_rank));
    const auto cpu_proxy = use_cpu_proxy ?
        mega::sm90_mega_moe_cpu_proxy_get_launch(
            num_ranks, num_experts, num_topk,
            num_max_tokens_per_rank, hidden) :
        mega::SM90MegaMoECPUProxyLaunch{};

    // Sanity: SFB tensors must be uint32 (UE8M0 packed) and weight tensors
    // must use byte-addressable packed FP4 storage (1 byte = 2 nibbles).
    DG_HOST_ASSERT(l1_weights_sf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(l2_weights_sf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(num_math_wg_decode_warps >= 0 and num_math_wg_decode_warps <= 4);
    DG_HOST_ASSERT(math_wg_participates_in_fp4_decode or num_math_wg_decode_warps == 0);
    DG_HOST_ASSERT(first_fp4_decode_assist_warp >= 0 and first_fp4_decode_assist_warp <= 4);

    // Heuristics
    const auto config = get_mega_moe_config_sm90_fp4(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_l1_sf_storage_tokens,
        use_early_b_decode, use_decode_done_mbarrier,
        use_swap_ab, use_swap_ab_fast_amax,
        num_compute_ring_tokens);

    // Tensormap construction
    constexpr int kGranK         = 128;  // L1 acts SF granularity (per-128 K)
    const int kL2ActsSFGranK = config.block_n == 64 ? 32 : 64;

    // Acts: FP8 e4m3, identical to FP8 path
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, num_l1_ring_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        num_l1_sf_storage_tokens, hidden,
                                                        config.block_m, kGranK,
                                                        1, 0);

    // Packed FP4 weight tile: each byte = 2 nibbles. SM90 loads these as raw
    // bytes and software-decodes them before WGMMA, so the TensorMap uses a
    // UINT8 view with a packed K axis.
    const auto l1_weights_bytes = l1_weights.scalar_type() == torch::kByte
        ? l1_weights : l1_weights.view(torch::kByte);
    const auto l2_weights_bytes = l2_weights.scalar_type() == torch::kByte
        ? l2_weights : l2_weights.view(torch::kByte);
    const auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights_bytes,
        hidden / 2, num_experts_per_rank * intermediate_hidden * 2,
        config.block_k / 2, config.block_n,
        static_cast<int>(l1_weights_bytes.stride(-2)),
        config.swizzle_weights_mode, /*swizzle_base=*/0,
        /*allow_tf32=*/false);

    // L1 output (post-SwiGLU FP8): N is halved.
    // Mirror the FP8 split-N infrastructure: when BLOCK_M=64 and the host
    // heuristics asked for multiple math warpgroups (`num_epilogue_warpgroups
    // > 1`), each warpgroup shares the same BLOCK_M rows but only owns
    // `WG_BLOCK_N = BLOCK_N / num_wg` columns. Each warpgroup issues its own
    // TMA store with that column tile, so the descriptor outer-box must be
    // `(WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2, l1_output_box_m)`. The split-N
    // gating must match the kernel-side `kSplitNWarpgroups` predicate, which
    // requires WG_BLOCK_N >= 64 (so the FP8MMASelector remains valid).
    const int num_epilogue_warpgroups_h = config.num_epilogue_threads / 128;
    const bool split_n_warpgroups =
        config.block_m == 64 and num_epilogue_warpgroups_h > 1 and
        config.block_n % num_epilogue_warpgroups_h == 0 and
        (config.block_n / num_epilogue_warpgroups_h) >= 64;
    const int wg_split_m = split_n_warpgroups ? 1 : num_epilogue_warpgroups_h;
    const int wg_split_n = split_n_warpgroups ? num_epilogue_warpgroups_h : 1;
    DG_HOST_ASSERT(wg_split_m * wg_split_n == num_epilogue_warpgroups_h);
    const int wg_block_m = config.block_m / wg_split_m;
    const int wg_block_n = config.block_n / wg_split_n;
    const int wg_l1_out_block_n = wg_block_n / 2;
    const int l1_output_box_m = wg_block_m;
    // Split-N with 32 post-SwiGLU cols per WG uses one combined 64-col TMA
    // store from WG0, matching the 64-col L2 activation-scale group.
    const bool split_n_combines_l1_store = split_n_warpgroups and wg_l1_out_block_n < 64;
    const int tma_l1_out_box_n = split_n_combines_l1_store ? (config.block_n / 2) : wg_l1_out_block_n;
    const int tma_l1_out_box_m = split_n_combines_l1_store ? config.block_m : l1_output_box_m;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, num_l2_storage_tokens,
                                                       tma_l1_out_box_n, tma_l1_out_box_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);

    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, num_l2_storage_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        num_l2_sf_storage_tokens, intermediate_hidden,
                                                        config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(
        l2_weights_bytes,
        intermediate_hidden / 2, num_experts_per_rank * hidden,
        config.block_k / 2, config.block_n,
        static_cast<int>(l2_weights_bytes.stride(-2)),
        config.swizzle_weights_mode, /*swizzle_base=*/0,
        /*allow_tf32=*/false);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const int compute_num_sms =
        num_sms - sidecar_num_blocks;
    DG_HOST_ASSERT(
        not use_sidecar_publisher or compute_num_sms >= num_ranks);
    DG_HOST_ASSERT(
        not use_sidecar_publisher or compute_num_sms % 2 == 0);
    const SM90FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .requested_num_max_tokens_per_rank =
            requested_num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .num_l1_ring_tokens = num_l1_ring_tokens,
        .num_l1_sf_storage_tokens = num_l1_sf_storage_tokens,
        .num_l2_ring_tokens = num_l2_ring_tokens,
        .num_l2_sf_storage_tokens = num_l2_sf_storage_tokens,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .use_wide_load_decode = use_wide_load_decode,
        .math_wg_participates_in_fp4_decode = math_wg_participates_in_fp4_decode,
        .num_math_wg_decode_warps = num_math_wg_decode_warps,
        .first_fp4_decode_assist_warp = first_fp4_decode_assist_warp,
        .use_early_b_decode = use_early_b_decode,
        .use_decode_done_mbarrier = use_decode_done_mbarrier,
        .use_l2_arrival_counter = use_l2_arrival_counter,
        .use_ss_nsplit = use_ss_nsplit,
        .use_swap_ab = use_swap_ab,
        .use_swap_ab_fast_amax = use_swap_ab_fast_amax,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .cpu_proxy = cpu_proxy,
        .sidecar_num_blocks = sidecar_num_blocks,
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .l1_weights_sf = reinterpret_cast<const uint32_t*>(l1_weights_sf.data_ptr()),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_weights_sf = reinterpret_cast<const uint32_t*>(l2_weights_sf.data_ptr()),
        .launch_args = LaunchArgs(compute_num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8FP4MegaMoERuntime::generate(args);
    const auto runtime_name = "sm90_fp8_fp4_mega_moe";
    const auto runtime = compiler->build(runtime_name, code);
    if (use_sidecar_publisher) {
        const auto sidecar_code =
            "#define DG_MEGA_MOE_FP4_SIDECAR_ONLY 1\n" + code;
        const auto sidecar_runtime = compiler->build(
            "sm90_fp8_fp4_mega_moe_sidecar", sidecar_code);
        SM90FP8FP4MegaMoERuntime::launch_sidecar(sidecar_runtime, args);
    }
    SM90FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
