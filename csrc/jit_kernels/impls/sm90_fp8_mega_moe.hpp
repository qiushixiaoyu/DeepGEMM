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
        int requested_num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        int num_l1_ring_tokens, num_l1_sf_storage_tokens;
        int num_l2_ring_tokens, num_l2_sf_storage_tokens;
        float activation_clamp;
        bool fast_math;
        int epilogue_registers;
        bool reuse_accum_as_final;
        bool l2_arrival_counter;
        bool l2_epilogue_requires_full_sync;
        bool split_phase_hot_path;
        bool use_swap_ab;
        bool use_activation_row_buckets;
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

        CUtensorMap tensor_map_l1_act_rows[3];
        CUtensorMap tensor_map_l2_act_rows[3];

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // Inter-node build: ranks span more than one NVLink domain (assume 8
        // NVLink peers per node). Inject macros so barrier.cuh (and later
        // dispatch/combine) take the NVSHMEM remote path. The `nvshmem` mention
        // in the comment also makes the JIT compiler device-link libnvshmem_device.
        constexpr int kNvlPeers = 8;
        const bool internode = args.num_ranks > kNvlPeers;
        std::string internode_prefix;
        // Validated opt-in optimizations; keep FP8's own density policy.
        const int packed_desc = get_env<int>("DG_MEGA_MOE_FP8_PACKED_GMMA_DESC", 0);
        DG_HOST_ASSERT(packed_desc == 0 or packed_desc == 1);
        if (internode)
            internode_prefix = fmt::format(
                "// inter-node mega-moe: uses nvshmem device functions\n"
                "#define DG_MEGA_MOE_INTERNODE\n"
                "#define DG_MEGA_MOE_NVL_PEERS {}\n",
                kNvlPeers);
        if (packed_desc)
            internode_prefix += "#define DG_MEGA_MOE_FP8_PACKED_GMMA_DESC 1\n";
        if (args.use_activation_row_buckets)
            internode_prefix += "#define DG_MEGA_MOE_FP8_ACTIVATION_ROW_TMA 1\n";
        const int skip_inactive_m_wg = get_env<int>("DG_MEGA_MOE_FP8_SKIP_INACTIVE_M_WG", 0);
        DG_HOST_ASSERT(skip_inactive_m_wg == 0 or skip_inactive_m_wg == 1);
        // The validated split-M layout is disjoint from the M64 hybrid path.
        // Do not silently enable this on experimental two-WG M128 shapes.
        if (skip_inactive_m_wg and args.config.block_m == 128 and
            args.config.block_n == 256 and args.config.num_epilogue_threads == 512 and
            not args.use_swap_ab)
            internode_prefix += "#define DG_MEGA_MOE_FP8_SKIP_INACTIVE_M_WG 1\n";
        const int row_parallel_quant = get_env<int>("DG_MEGA_MOE_FP8_ROW_PARALLEL_QUANT", 0);
        DG_HOST_ASSERT(row_parallel_quant >= 0 and row_parallel_quant <= 2);
        const float expected_rows = static_cast<float>(args.num_tokens) * args.num_ranks *
            args.num_topk / args.num_experts;
        const int hybrid_max_rows = get_env<int>("DG_MEGA_MOE_FP8_HYBRID_MAX_ROWS", 0);
        DG_HOST_ASSERT(hybrid_max_rows == 0 or hybrid_max_rows == 32);
        const bool hybrid_blocks = hybrid_max_rows and args.use_swap_ab and
            args.config.block_m == 64 and args.config.block_n == 256 and
            args.config.num_epilogue_threads == 256 and
            static_cast<int64_t>(args.hidden) * args.intermediate_hidden < 16ll * 1024 * 1024 and
            expected_rows > 16.0f and expected_rows <= 64.0f;
        if (hybrid_blocks)
            internode_prefix += fmt::format("#define DG_MEGA_MOE_FP8_HYBRID_MAX_ROWS {}\n", hybrid_max_rows);
        const bool expanded_streaming_band =
            static_cast<int64_t>(args.hidden) * args.intermediate_hidden >= 16ll * 1024 * 1024 and
            expected_rows > 24.0f and expected_rows <= 32.0f;
        if (args.use_swap_ab and (hybrid_blocks or row_parallel_quant == 1 or
                                 (row_parallel_quant == 2 and expanded_streaming_band)))
            internode_prefix += "#define DG_MEGA_MOE_FP8_ROW_PARALLEL_QUANT 1\n";
        if (get_env<int>("DG_MEGA_MOE_PHASE_PROFILE", 0) != 0)
            internode_prefix += "#define DG_MEGA_MOE_PHASE_PROFILE 1\n";
        // SILENT only removes profile printf. Clock reads, counters and
        // profile stores still perturb execution when PHASE_PROFILE is on.
        // Disable PHASE_PROFILE for uninstrumented performance measurements.
        if (get_env<int>("DG_MEGA_MOE_PHASE_PROFILE_SILENT", 0) != 0)
            internode_prefix +=
                "#define DG_MEGA_MOE_PHASE_PROFILE_SILENT 1\n";
        if (get_env<int>("DG_MEGA_MOE_DEBUG_NVSHMEM_STATE", 0) != 0)
            internode_prefix +=
                "#define DG_MEGA_MOE_DEBUG_NVSHMEM_STATE 1\n";
        // Device printf forces ptxas to emit vprintf calls and a call stack for
        // the entire fused kernel.  On the small swapAB shapes this also makes
        // ptxas serialize every QGMMA with a DEPBAR, even though the print is
        // only reachable on a protocol failure.  Production keeps the trap and
        // assertions but compiles textual diagnostics only when requested.
        const bool device_diagnostics =
            get_env<int>("DG_MEGA_MOE_DEVICE_DIAGNOSTICS", 0) != 0;
        if (device_diagnostics)
            internode_prefix +=
                "#define DG_MEGA_MOE_DEVICE_DIAGNOSTICS 1\n";
        else if (internode)
            internode_prefix +=
                "#define DG_DEVICE_ASSERT_TRAP_ONLY 1\n";
        // Unified two-level dispatch handshake. Packed mode compacts live route
        // cells into per-remote-node, per-epoch packed slots. All packed
        // shapes send live header/offset/payload bytes followed by a separate
        // fixed-address manifest WRITE on the same QP. Mode 4 is the normal
        // small-capacity dense-V3 format: one full collect box plus its
        // manifest, without compaction/decode. Both gateway modes use the current
        // double-buffered epoch lifetime. Inter-node builds always use one of
        // these two gateway formats; the per-entry direct path is not built.
        DG_HOST_ASSERT(not internode or
                       args.num_ranks % static_cast<uint32_t>(kNvlPeers) == 0);
        // Storage is aligned to 384 tokens, but protocol selection follows
        // the exact capacity requested by the caller: small-capacity buffers
        // use the dense V3 wire format and larger buffers use packed metadata.
        const int dispatch_gateway = internode
            ? (args.requested_num_max_tokens_per_rank <=
                   layout::kGatewayDenseMaxRequestedTokens ? 4 : 3)
            : 0;
        if (dispatch_gateway == 3)
            internode_prefix +=
                "#define DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED 1\n";
        if (dispatch_gateway == 4)
            internode_prefix +=
                "#define DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3 1\n";
        const int num_experts_per_rank =
            static_cast<int>(args.num_experts / args.num_ranks);
        // The route-count pass assigns one contiguous token tranche to each
        // CTA.  Decode-sized launches therefore have a compile-time-known
        // prefix of CTAs that can possibly own metadata; waiting for all 78
        // CTAs only delays manifest publication after the last real producer
        // has already finished.  Keep the full grid for pull/GEMM/scatter,
        // but let the count/manifest protocol wait for exactly that producer
        // prefix.  This is derived from the actual JIT shape, not an A/B
        // switch, and applies to every dense-decode expert topology.
        if (dispatch_gateway == 4) {
            const int tokens_per_metadata_cta =
                (args.config.num_dispatch_threads / 32) *
                (32 / args.num_topk);
            const int num_metadata_sms = std::min(
                args.launch_args.grid_dim.first,
                std::max(1, (args.num_tokens + tokens_per_metadata_cta - 1) /
                                tokens_per_metadata_cta));
            internode_prefix += fmt::format(
                "#define DG_MEGA_MOE_NUM_METADATA_SMS {}\n",
                num_metadata_sms);
        }
        // Active-pair completion removes zero-token (expert, dst) system
        // atomics.  Its count snapshot pays off when the many-local-expert
        // publisher matrix is sparse, but becomes overhead as rows per expert
        // increase.  Select by expected work density instead of a model- or
        // batch-specific upper bound.
        const float expected_rows_per_local_expert =
            static_cast<float>(args.num_tokens * args.num_topk) /
            num_experts_per_rank;
        const bool active_pair_completion =
            dispatch_gateway == 4 and args.num_tokens >= 16 and
            num_experts_per_rank >= 48 and
            expected_rows_per_local_expert <= 6.0f;
        if (active_pair_completion)
            internode_prefix +=
                "#define DG_MEGA_MOE_ACTIVE_PAIR_COMPLETION 1\n";
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
        {}, {}, {}, {},
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
    args.num_l1_ring_tokens,
    args.num_l1_sf_storage_tokens,
    args.num_l2_ring_tokens,
    args.num_l2_sf_storage_tokens,
    internode ?
        layout::get_num_sm90_combine_ring_tokens(
            args.num_ranks, args.num_max_tokens_per_rank, args.num_topk,
            args.num_experts / args.num_ranks) :
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
    args.use_swap_ab ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // Preserve the original kernel ABI for every non-row-TMA specialization.
        if (args.use_activation_row_buckets) {
            DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
                args.y, args.cumulative_local_expert_recv_stats, args.num_tokens,
                args.sym_buffer_ptrs, args.tensor_map_l1_acts,
                args.tensor_map_l1_acts_sf, args.tensor_map_l1_weights,
                args.l1_weights_sf, args.tensor_map_l1_output,
                args.tensor_map_l2_acts, args.tensor_map_l2_acts_sf,
                args.tensor_map_l2_weights, args.l2_weights_sf,
                args.tensor_map_l1_act_rows[0], args.tensor_map_l1_act_rows[1],
                args.tensor_map_l1_act_rows[2], args.tensor_map_l2_act_rows[0],
                args.tensor_map_l2_act_rows[1], args.tensor_map_l2_act_rows[2]
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

static void sm90_fp8_mega_moe(
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
    const bool& fast_math
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
    // L2 exposes one extra 128-row scratch/guard block in its tensor extent;
    // L1 has no such suffix and therefore carries the unambiguous compact-ring
    // decision.  Pass only the reusable L2 rows to the JIT template: the
    // device layout adds the same guard block back when slicing storage.
    const bool l2_ring_enabled =
        num_l1_ring_tokens < num_max_pool_tokens;
    constexpr int kL2RingScratchRows = 128;
    const int num_l2_ring_tokens = l2_ring_enabled ?
        num_l2_storage_tokens - kL2RingScratchRows :
        num_max_pool_tokens;
    DG_HOST_ASSERT(not l2_ring_enabled or
                   num_l2_ring_tokens == num_l1_ring_tokens);
    const int num_compute_ring_tokens = std::min(
        num_l1_ring_tokens, num_l2_ring_tokens);

    // Heuristics.  The physical compute capacity participates in expert-wave
    // selection so one complete L1 wave always fits before its L2 phase can
    // retire and recycle the slots.
    const auto config = get_mega_moe_config_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_l1_sf_storage_tokens,
        num_compute_ring_tokens);
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
    // L1/L2 K extents are JIT constants for every generated shape.  Keep a
    // single statically unrolled implementation; the former Pro b1/b2 runtime
    // loop exception no longer helps after the N256 internal-N64 path.
    const bool split_phase_hot_path = true;
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
        config.block_m, config.num_epilogue_threads,
        hidden, intermediate_hidden);

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-64 for L2 (MN-major, no swizzle).
    // Weight SF: block (128, 128) raw float pointer (no TMA descriptor).
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 64;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, num_l1_ring_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        num_l1_sf_storage_tokens, hidden,
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
                                                       intermediate_hidden, num_l2_storage_tokens,
                                                       l1_output_box_n, l1_output_box_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       l1_output_swizzle_mode);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, num_l2_storage_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        num_l2_sf_storage_tokens, intermediate_hidden,
                                                        config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, weight_tma_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    // Reduce only activation payload, not the shared allocation/stride or SFA.
    // All padded rows consumed by a swap WGMMA remain inside the selected box.
    const int activation_row_tma = get_env<int>("DG_MEGA_MOE_FP8_ACTIVATION_ROW_TMA", 0);
    DG_HOST_ASSERT(activation_row_tma == 0 or activation_row_tma == 1);
    const bool use_activation_row_buckets = activation_row_tma != 0 and
        use_swap_ab and config.block_m == 64 and config.block_n == 256 and
        config.block_k == 128 and config.num_epilogue_threads == 256 and
        config.swizzle_acts_mode == 128 and config.cluster_size == 1;
    CUtensorMap tensor_map_l1_act_rows[3] = {};
    CUtensorMap tensor_map_l2_act_rows[3] = {};
    if (use_activation_row_buckets) {
        for (int i = 0; i < 3; ++i) {
            const int rows = 8 << i;
            tensor_map_l1_act_rows[i] = make_tma_2d_desc(
                l1_acts, hidden, num_l1_ring_tokens, config.block_k, rows,
                static_cast<int>(l1_acts.stride(-2)), config.swizzle_acts_mode);
            tensor_map_l2_act_rows[i] = make_tma_2d_desc(
                l2_acts, intermediate_hidden, num_l2_storage_tokens, config.block_k, rows,
                static_cast<int>(l2_acts.stride(-2)), config.swizzle_acts_mode);
        }
    }

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM90FP8MegaMoERuntime::Args args = {
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
        .epilogue_registers = epilogue_registers,
        .reuse_accum_as_final = reuse_accum_as_final,
        .l2_arrival_counter = l2_arrival_counter,
        .l2_epilogue_requires_full_sync = l2_epilogue_requires_full_sync,
        .split_phase_hot_path = split_phase_hot_path,
        .use_swap_ab = use_swap_ab,
        .use_activation_row_buckets = use_activation_row_buckets,
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
        .tensor_map_l1_act_rows = {tensor_map_l1_act_rows[0], tensor_map_l1_act_rows[1], tensor_map_l1_act_rows[2]},
        .tensor_map_l2_act_rows = {tensor_map_l2_act_rows[0], tensor_map_l2_act_rows[1], tensor_map_l2_act_rows[2]},
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_mega_moe", code);
    SM90FP8MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
