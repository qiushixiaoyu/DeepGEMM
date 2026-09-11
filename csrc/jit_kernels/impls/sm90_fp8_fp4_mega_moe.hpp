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

// Small requested capacities use dense metadata; larger capacities retain
// the packed protocol. Storage alignment must not change this boundary.
static int get_sm90_fp4_dispatch_gateway(const int& requested_capacity) {
    DG_HOST_ASSERT(requested_capacity > 0);
    return requested_capacity > layout::kGatewayDenseMaxRequestedTokens ? 3 : 4;
}

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
        // Split each SS N=128 WGMMA into two N=64 WGMMAs so the
        // per-K-block accumulator is 32 floats instead of 64. Large-token SS
        // shapes enable this to reduce accumulator pressure while keeping SS
        // scheduling.
        bool use_ss_nsplit;
        // swapAB path: use decoded weight as WGMMA-M and tokens as WGMMA-N.
        bool use_swap_ab;
        bool use_swap_ab_fast_amax;
        bool sfb_n_contiguous;
        bool use_activation_row_buckets;
        MegaMoESM90Config config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

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
        // Optional M64/swap-AB input boxes: 8, 16, and 32 token rows.
        // Append these to the kernel ABI only for the enabled specialization.
        CUtensorMap tensor_map_l1_act_rows[3];
        CUtensorMap tensor_map_l2_act_rows[3];

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
        if (args.sfb_n_contiguous)
            internode_prefix += "#define DG_MEGA_MOE_FP4_SFB_N_CONTIGUOUS 1\n";
        if (get_env<int>("DG_MEGA_MOE_FP4_SFB_TMA", 0) != 0) {
            DG_HOST_ASSERT(args.sfb_n_contiguous);
            internode_prefix += "#define DG_MEGA_MOE_FP4_SFB_TMA 1\n";
        }
        if (get_env<int>("DG_MEGA_MOE_FP4_PAIRED_PRMT", 0) != 0)
            internode_prefix += "#define DG_MEGA_MOE_FP4_PAIRED_PRMT 1\n";
        if (get_env<int>("DG_MEGA_MOE_FP4_PACKED_GMMA_DESC", 0) != 0)
            internode_prefix += "#define DG_MEGA_MOE_FP4_PACKED_GMMA_DESC 1\n";
        if (get_env<int>("DG_MEGA_MOE_FP4_ROW_PARALLEL_QUANT", 0) != 0)
            internode_prefix += "#define DG_MEGA_MOE_FP4_ROW_PARALLEL_QUANT 1\n";
        if (use_sm90_fp4_n64_decode_ready(
                args.config.block_m, args.config.block_n,
                args.config.num_epilogue_threads / 128,
                args.use_swap_ab, args.use_decode_done_mbarrier)) {
            DG_HOST_ASSERT(args.num_math_wg_decode_warps == 0 and
                           not args.math_wg_participates_in_fp4_decode);
            const int helper_warps = args.config.num_non_epilogue_threads / 32 -
                                     args.first_fp4_decode_assist_warp;
            DG_HOST_ASSERT(helper_warps > 0 and helper_warps % 2 == 0);
            internode_prefix += "#define DG_MEGA_MOE_FP4_N64_DECODE_READY 1\n";
            const int skip_math_b_input_wait = get_env<int>(
                "DG_MEGA_MOE_FP4_SKIP_MATH_B_INPUT_WAIT", 0);
            DG_HOST_ASSERT(skip_math_b_input_wait == 0 or skip_math_b_input_wait == 1);
            if (skip_math_b_input_wait != 0 and args.use_early_b_decode)
                internode_prefix +=
                    "#define DG_MEGA_MOE_FP4_SKIP_MATH_B_INPUT_WAIT 1\n";
            const int full_unroll = get_env<int>(
                "DG_MEGA_MOE_FP4_DECODE_FULL_UNROLL", 0);
            DG_HOST_ASSERT(full_unroll == 0 or full_unroll == 1);
            // Four helper warps split into two 64-thread groups. Each
            // thread decodes four K32 groups in the N64 half. The 512-thread
            // CTA was validated for this unroll; eight-helper CTAs already
            // expand their two iterations and should keep the original codegen.
            const int cta_threads = args.config.num_dispatch_threads +
                                    args.config.num_non_epilogue_threads +
                                    args.config.num_epilogue_threads;
            const int balanced_registers = get_env<int>(
                "DG_MEGA_MOE_FP4_BALANCED_WG_REGISTERS", 0);
            DG_HOST_ASSERT(balanced_registers == 0 or balanced_registers == 1);
            if (balanced_registers != 0 and args.num_ranks > kNvlPeers and
                args.config.block_m == 64 and args.config.block_n == 128 and
                args.config.block_k == 128 and args.config.num_dispatch_threads == 64 and
                args.config.num_epilogue_threads == 256 and
                args.first_fp4_decode_assist_warp == 2 and
                ((helper_warps == 4 and cta_threads == 512) or
                 (helper_warps == 8 and cta_threads == 640))) {
                internode_prefix +=
                    "#define DG_MEGA_MOE_FP4_BALANCED_WG_REGISTERS 1\n";
                const int decode_register_boost = get_env<int>(
                    "DG_MEGA_MOE_FP4_DECODE_REGISTER_BOOST", 0);
                DG_HOST_ASSERT(decode_register_boost == 0 or decode_register_boost == 1);
                const int decode_registers = decode_register_boost == 0 ? 40 :
                    (cta_threads == 640 ? 48 : 64);
                internode_prefix += fmt::format(
                    "#define DG_MEGA_MOE_FP4_BALANCED_WG_DECODE_REGISTERS {}\n",
                    decode_registers);
                const int eight_math_registers = get_env<int>(
                    "DG_MEGA_MOE_FP4_EIGHT_MATH_REGISTERS", 0);
                DG_HOST_ASSERT(eight_math_registers >= 0 and eight_math_registers <= 2);
                const float math_expected_rows_per_expert =
                    static_cast<float>(args.num_tokens) * args.num_topk /
                    (args.num_experts / args.num_ranks);
                // Mode 1 is the original forced A/B. Mode 2 pairs with the
                // density helper policy and changes only its higher-density
                // M64 profile; low density keeps the original 128 math quota.
                const bool use_eight_math_registers = eight_math_registers == 1 or
                    (eight_math_registers == 2 and math_expected_rows_per_expert >= 32.0f);
                if (use_eight_math_registers) {
                    DG_HOST_ASSERT(cta_threads == 640 and helper_warps == 8 and
                                   decode_register_boost == 1);
                    internode_prefix +=
                        "#define DG_MEGA_MOE_FP4_EIGHT_MATH_REGISTERS 1\n";
                }
            }
            const bool unroll_four_groups = helper_warps == 4 and cta_threads == 512;
            if (full_unroll != 0 and unroll_four_groups)
                internode_prefix += "#define DG_MEGA_MOE_FP4_DECODE_FULL_UNROLL 1\n";
            const int sfb_lookahead = get_env<int>(
                "DG_MEGA_MOE_FP4_SFB_SHARED_LOOKAHEAD", 0);
            DG_HOST_ASSERT(sfb_lookahead >= 0 and sfb_lookahead <= 2);
            // Current ready-stage reads only: four K32 groups per helper.
            // Keep every other topology on the original decode implementation.
            if (sfb_lookahead != 0 and full_unroll != 0 and
                unroll_four_groups and args.use_wide_load_decode and
                args.config.block_k == 128) {
                DG_HOST_ASSERT(get_env<int>(
                    "DG_MEGA_MOE_FP4_SFB_BROADCAST_OVERRIDE", -1) == 0);
                internode_prefix +=
                    "#define DG_MEGA_MOE_FP4_SFB_SHARED_LOOKAHEAD 1\n";
            } else if (sfb_lookahead == 2 and helper_warps == 8 and
                       cta_threads == 640 and args.use_wide_load_decode and
                       args.config.block_k == 128) {
                // Mode 2 additionally covers two K32 groups per helper.
                // Do not emit FULL_UNROLL for this topology: its original
                // two-iteration loop is already expanded by the compiler.
                DG_HOST_ASSERT(get_env<int>(
                    "DG_MEGA_MOE_FP4_SFB_BROADCAST_OVERRIDE", -1) == 0);
                internode_prefix +=
                    "#define DG_MEGA_MOE_FP4_SFB_SHARED_LOOKAHEAD_8W 1\n";
            }
        }
        if (args.num_ranks > kNvlPeers) {
            internode_prefix += fmt::format(
                "// inter-node mega-moe: uses nvshmem device functions\n"
                "#define DG_MEGA_MOE_INTERNODE\n"
                "#define DG_MEGA_MOE_NVL_PEERS {}\n", kNvlPeers);
            const int dispatch_gateway = get_sm90_fp4_dispatch_gateway(
                args.requested_num_max_tokens_per_rank);
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
            const int publisher_backoff_mode = get_env<int>(
                "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MODE", 0);
            DG_HOST_ASSERT(
                publisher_backoff_mode >= 0 and
                publisher_backoff_mode <= 2);
            const bool use_publisher_idle_backoff =
                publisher_backoff_mode == 2 or
                (publisher_backoff_mode == 0 and
                 auto_publisher_idle_backoff);
            if (use_publisher_idle_backoff) {
                const int publisher_backoff_initial_ns = get_env<int>(
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS", 64);
                const int publisher_backoff_max_ns = get_env<int>(
                    "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MAX_NS", 512);
                DG_HOST_ASSERT(publisher_backoff_initial_ns > 0 and
                               publisher_backoff_max_ns >=
                                   publisher_backoff_initial_ns and
                               publisher_backoff_max_ns <= 1000000);
                internode_prefix += fmt::format(
                    "#define DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS {}\n"
                    "#define DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MAX_NS {}\n",
                    publisher_backoff_initial_ns,
                    publisher_backoff_max_ns);
            }
            // Four neighboring decode lanes consume the four K groups of
            // one N row and share its packed SFB word.  At middle density the
            // work per block amortizes one subgroup shuffle and benefits from
            // replacing four shared loads with one.  Very sparse work remains
            // shuffle-latency sensitive, while the shared M128 crossover
            // switches execution topology and showed no benefit.  Use the
            // same weight-footprint boundary as the other FP4 heuristics.
            const float min_sfb_broadcast_rows = weight_light ? 24.0f : 16.0f;
            const bool auto_sfb_subgroup_broadcast =
                expected_rows_per_local_expert >= min_sfb_broadcast_rows and
                expected_rows_per_local_expert <= 32.0f;
            // Diagnostic overrides preserve the density policy at -1.
            // Explicit 0 emits no macro: the decode implementation uses ifdef.
            const int sfb_broadcast_override = get_env<int>(
                "DG_MEGA_MOE_FP4_SFB_BROADCAST_OVERRIDE", -1);
            DG_HOST_ASSERT(sfb_broadcast_override >= -1 and
                           sfb_broadcast_override <= 1);
            const bool use_sfb_subgroup_broadcast =
                sfb_broadcast_override < 0 ? auto_sfb_subgroup_broadcast :
                sfb_broadcast_override != 0;
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
            const int fine_bucket_cap = get_env<int>(
                "DG_MEGA_MOE_FP4_FINE_BUCKET_CAP", -1);
            DG_HOST_ASSERT(fine_bucket_cap >= -1 and
                           fine_bucket_cap <= 3);
            if (fine_bucket_cap >= 0)
                fp4_swap_ab_fine_bucket_level =
                    std::min(fp4_swap_ab_fine_bucket_level, fine_bucket_cap);
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
            const int scale_float2_override = get_env<int>(
                "DG_MEGA_MOE_FP4_SCALE_FLOAT2_OVERRIDE", -1);
            DG_HOST_ASSERT(scale_float2_override >= -1 and
                           scale_float2_override <= 3);
            const int scale_float2_level = scale_float2_override < 0 ?
                (use_swap_scale_float2 ? 3 : 0) : scale_float2_override;
            if (scale_float2_level > 0)
                internode_prefix += fmt::format(
                    "#define DG_MEGA_MOE_FP4_SWAP_SCALE_FLOAT2_LEVEL {}\n",
                    scale_float2_level);
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
        // SILENT suppresses profile printf, not clock/counter/store overhead.
        // Disable PHASE_PROFILE for uninstrumented performance measurements.
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
        if (args.use_activation_row_buckets)
            internode_prefix +=
                "#define DG_MEGA_MOE_FP4_ACTIVATION_ROW_TMA 1\n";
        return internode_prefix + fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
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
        {}, {}, {}, {}, {}
    >);
}};
)",
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
    args.use_ss_nsplit ? "true" : "false",
    args.use_swap_ab ? "true" : "false",
    args.use_swap_ab_fast_amax ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if (args.use_activation_row_buckets) {
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
                args.tensor_map_l1_act_rows[0],
                args.tensor_map_l1_act_rows[1],
                args.tensor_map_l1_act_rows[2],
                args.tensor_map_l2_act_rows[0],
                args.tensor_map_l2_act_rows[1],
                args.tensor_map_l2_act_rows[2]
            ));
            return;
        }
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
            args.l2_weights_sf
        ));
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

    // Only swap-AB gives the math instruction a runtime token-row extent.
    // The conventional M128 path must retain its full activation tile.
    const int activation_row_tma = get_env<int>(
        "DG_MEGA_MOE_FP4_ACTIVATION_ROW_TMA", 0);
    DG_HOST_ASSERT(activation_row_tma == 0 or activation_row_tma == 1);
    const bool use_activation_row_buckets = activation_row_tma != 0 and
        use_swap_ab and not use_ss_nsplit and config.block_m == 64 and
        (config.block_n == 128 or config.block_n == 256) and
        config.num_epilogue_threads == 256 and config.block_k == 128 and
        config.swizzle_acts_mode == 128 and config.cluster_size == 1;
    CUtensorMap tensor_map_l1_act_rows[3] = {};
    CUtensorMap tensor_map_l2_act_rows[3] = {};
    if (use_activation_row_buckets) {
        for (int i = 0; i < 3; ++i) {
            const int rows = 8 << i;
            tensor_map_l1_act_rows[i] = make_tma_2d_desc(
                l1_acts, hidden, num_l1_ring_tokens,
                config.block_k, rows,
                static_cast<int>(l1_acts.stride(-2)),
                config.swizzle_acts_mode);
            tensor_map_l2_act_rows[i] = make_tma_2d_desc(
                l2_acts, intermediate_hidden, num_l2_storage_tokens,
                config.block_k, rows,
                static_cast<int>(l2_acts.stride(-2)),
                config.swizzle_acts_mode);
        }
    }

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
    // Launch
    const auto num_sms = device_runtime->get_num_sms();
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
        .use_ss_nsplit = use_ss_nsplit,
        .use_swap_ab = use_swap_ab,
        .use_swap_ab_fast_amax = use_swap_ab_fast_amax,
        .sfb_n_contiguous = not l1_weights_sf.is_contiguous() or not l2_weights_sf.is_contiguous(),
        .use_activation_row_buckets = use_activation_row_buckets,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .l1_weights_sf = reinterpret_cast<const uint32_t*>(l1_weights_sf.data_ptr()),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_weights_sf = reinterpret_cast<const uint32_t*>(l2_weights_sf.data_ptr()),
        .tensor_map_l1_act_rows = {tensor_map_l1_act_rows[0],
                                   tensor_map_l1_act_rows[1],
                                   tensor_map_l1_act_rows[2]},
        .tensor_map_l2_act_rows = {tensor_map_l2_act_rows[0],
                                   tensor_map_l2_act_rows[1],
                                   tensor_map_l2_act_rows[2]},
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8FP4MegaMoERuntime::generate(args);
    const auto runtime_name = "sm90_fp8_fp4_mega_moe";
    const auto runtime = compiler->build(runtime_name, code);
    SM90FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
