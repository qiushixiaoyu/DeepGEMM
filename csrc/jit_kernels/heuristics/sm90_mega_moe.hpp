#pragma once

#include "mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) MegaMoE configuration
// ----------------------------------------------------------------------------
// SM90 differs from SM100 in:
//   - No tensor memory (TMEM): WGMMA accumulators live in registers.
//   - No FP4: weights are FP8 e4m3 with per-128 channel float scales.
//   - No 2-CTA cluster MMA: TMA multicast cluster=2 may still be used.
//   - Activation SF is float, not UE8M0 int: L1 input uses per-128 K and the
//     fused L1 epilogue writes L2 activation SF at per-64 K granularity.
// The kernel implementation is in `deep_gemm/impls/sm90_fp8_mega_moe.cuh`.
// ============================================================================

struct MegaMoESM90Config {
    int block_m, block_n, block_k;
    int cluster_size;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int swizzle_acts_mode, swizzle_weights_mode;
    int num_experts_per_wave;
    int num_stages, smem_size;
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

// Keep the FP4 M64/swapAB and M128/non-swap regions contiguous.  Boundary
// sweeps place the crossover between 54.9 and 59.4 expected rows/expert; 59
// selects the lower-padding M64 path below that point and the lower-task-count
// M128 path at and above it without leaving an M64/non-swap gap.
constexpr float kFP4SM90M128CrossoverRows = 59.0f;

static std::tuple<int, int> get_block_config_for_mega_moe_sm90(
    const int& num_ranks, const int& num_experts,
    const int& num_topk, const int& num_tokens) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn = expected_tokens_per_expert > 64.0f;
    if (auto_split_mn)
        return {128, 512};

    const int block_m = 64;
    const int num_epilogue_warpgroups = 2;

    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );
    return {block_m, num_epilogue_warpgroups * 128};
}

static int get_num_experts_per_wave_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms,
    const int& num_ring_tokens, const int& num_max_tokens_per_rank, const int& num_ranks) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk,
        num_experts_per_rank);
    // A compact physical pool may be reused only after the current expert
    // wave finishes L2.  Let the common ring-aware heuristic cap the wave so
    // its worst-case routed rows fit; otherwise a later L1 block in the same
    // wave could wait for L2 work that the scheduler cannot start yet.
    //
    // Size this bound from the current launch's token count, not the buffer's
    // configured maximum.  The physical ring is allocated from the maximum,
    // but a small request whose entire routed working set fits must retain the
    // original full-expert wave instead of being throttled as if every slot in
    // the SymmBuffer were active.
    if (num_ring_tokens < num_max_pool_tokens)
        return get_num_experts_per_wave_for_mega_moe(
            num_experts_per_rank, num_tokens, num_topk,
            intermediate_hidden, block_m, block_n, num_sms,
            num_ring_tokens, align(num_tokens, block_m), num_ranks);
    if (expected_tokens_per_expert < 1.0f or expected_tokens_per_expert > 4.0f)
        return num_experts_per_rank;

    if (block_m == 64 and intermediate_hidden >= 3072) {
        const int num_n_blocks_per_expert = (2 * intermediate_hidden) / block_n;
        const int single_wave_blocks =
            num_experts_per_rank * num_n_blocks_per_expert;
        if (single_wave_blocks >= 4 * num_sms)
            return num_experts_per_rank;
    }
    return get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms,
        num_ring_tokens, num_max_tokens_per_rank, num_ranks);
}

// swapAB keeps a BLOCK_N=256 scheduler/loader tile and executes two internal
// N64 swapped-operand WGMMA subtiles per warpgroup. This preserves the wide
// launch/protocol granularity while avoiding wasted M work in latency-bound
// decode shapes. Two conditions are calibrated on Flash / GLM5.2 / Pro
// (`test_logs/20260813_swapab_calib` and the 2026-08-21 N256 revalidation):
//
//   per-expert weights = 3 * hidden * intermediate bytes (FP8: L1 2HI + L2 HI)
//     Flash 25.2 MB, GLM5.2 37.7 MB -> swapAB wins at small M; both now use
//                                      the N256/internal-N64 variant
//     Pro   66.1 MB                 -> use the N256/internal-N64 swap path;
//                                      it retains one scheduler/loader block
//                                      while running two N64 swap subtiles/WG.
//   tokens per expert
//     Flash: swapAB wins at 1.5/3/6, ties at 12/24
//     GLM:   wins at 0.5/4, ties at 2/8, and wins again at 16 with the
//            N256/internal-N64 path (2026-08-22); the old loss at 16 came
//            from the removed N128 scheduler variant.
//
// Hence: all swapAB shapes use N256/internal-N64.  The profitable density
// band grows when each expert must stream a large weight footprint: light
// weights switch back above 16 expected rows/expert, while weight-streaming
// shapes keep swap through 24.  This uses workload properties rather than a
// model-name or exact H/I whitelist.
static bool should_use_swap_ab_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& block_m, const int& num_epilogue_threads,
    const int& hidden, const int& intermediate_hidden) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    // N256/internal-N64 already keeps the wide scheduler/loader tile, so
    // weight-streaming-bound shapes can use swap for a wider M range without
    // doubling the outer task count.
    constexpr int64_t kSwapAbMaxWeightElems = 16ll * 1024 * 1024;
    const int64_t weight_elems =
        static_cast<int64_t>(hidden) * intermediate_hidden;
    const bool weight_light = weight_elems < kSwapAbMaxWeightElems;
    const int hybrid_max_rows = get_env<int>("DG_MEGA_MOE_FP8_HYBRID_MAX_ROWS", 0);
    DG_HOST_ASSERT(hybrid_max_rows == 0 or hybrid_max_rows == 32);
    // Allocate the existing swap staging for both block-level alternatives.
    // Confine this opt-in to the light-weight non-swap M64 density band.
    if (hybrid_max_rows and weight_light and decode_split_n_path and
        expected_tokens_per_expert > 16.0f and expected_tokens_per_expert <= 64.0f and
        hidden % 256 == 0 and (2 * intermediate_hidden) % 256 == 0)
        return true;
    constexpr float kSwapAbMaxTokensPerExpert = 16.0f;
    const bool light_weight_decode =
        weight_light and expected_tokens_per_expert > 0.0f and
        expected_tokens_per_expert <= kSwapAbMaxTokensPerExpert;
    const int streaming_density32 = get_env<int>("DG_MEGA_MOE_FP8_STREAMING_DENSITY32", 0);
    DG_HOST_ASSERT(streaming_density32 == 0 or streaming_density32 == 1);
    const float kStreamingSwapAbMaxTokensPerExpert = streaming_density32 ? 32.0f : 24.0f;
    const bool weight_streaming_decode =
        not weight_light and
        expected_tokens_per_expert > 0.0f and
        expected_tokens_per_expert <= kStreamingSwapAbMaxTokensPerExpert;
    return decode_split_n_path and
           (light_weight_decode or weight_streaming_decode);
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& use_swap_ab = false) {
    constexpr int kSmemAlignment = 1024;

    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    const int smem_cd_l1 = block_m * (block_n / 2);
    const int smem_cd_l2 = block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd_swap_l1 = use_swap_ab
        ? block_m * (block_n / 2) *
              (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(uint8_t)))
        : 0;
    const int smem_cd = align(
        std::max(std::max(smem_cd_l1, smem_cd_l2), smem_cd_swap_l1),
        kSmemAlignment);

    const int smem_sfa_per_stage = align(2 * block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfb_per_stage = 0;
    const int smem_per_stage = block_m * block_k + block_n * block_k +
                               smem_sfa_per_stage + smem_sfb_per_stage;

    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_barriers_per_stage = 2 * 8;
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_barriers_fixed;

    const int num_stages = (smem_capacity - smem_fixed) /
                           (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(num_stages >= 2);
    const int smem_size = smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(smem_size <= smem_capacity);
    return {num_stages, smem_size};
}

static std::tuple<int, int> get_block_config_for_mega_moe_sm90_fp4(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens) {
    (void)num_max_tokens_per_rank;

    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn =
        expected_tokens_per_expert >= kFP4SM90M128CrossoverRows;
    const bool ultra_small_split_n =
        expected_tokens_per_expert > 0.0f and
        expected_tokens_per_expert < 0.375f;
    int block_m = auto_split_mn ? 128 : 64;
    int num_epilogue_warpgroups = (auto_split_mn or ultra_small_split_n) ? 2 : block_m / 64;
    DG_HOST_ASSERT(block_m >= 64 and block_m % 64 == 0);
    DG_HOST_ASSERT(num_epilogue_warpgroups >= 1 and
                   ((block_m / num_epilogue_warpgroups == 64) or
                    (block_m == 64 and num_epilogue_warpgroups > 1)));

    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );
    return {block_m, num_epilogue_warpgroups * 128};
}

static int get_num_experts_per_wave_for_mega_moe_sm90_fp4(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms,
    const int& num_ring_tokens, const int& num_max_tokens_per_rank, const int& num_ranks) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk,
        num_experts_per_rank);
    // A wave must fit in the physical compute ring even when the SymmBuffer was
    // allocated for a larger request.  Use this launch's aligned token count
    // for the lifetime bound so an inactive max-token tail does not throttle
    // the scheduler.
    int max_ring_safe_experts = num_experts_per_rank;
    if (num_ring_tokens < num_max_pool_tokens) {
        const int active_max_tokens = align(num_tokens, block_m);
        while (max_ring_safe_experts > 0 and
               get_num_wave_pool_tokens(
                   num_ranks, num_topk, active_max_tokens,
                   max_ring_safe_experts, block_m) > num_ring_tokens)
            -- max_ring_safe_experts;
        DG_HOST_ASSERT(max_ring_safe_experts > 0 and
                       "FP4 compute ring is too small for one expert wave");
    }

    // JIT cannot see the realized route histogram, but uniform routing gives
    // a stable estimate of the number of non-empty local experts:
    //   P(expert active) = 1 - exp(-expected rows/expert).
    // Multiply the CTA count of an active expert by this probability to size
    // the contiguous expert-ID span needed to expose four CTA waves.  This keeps
    // sparse launches wide enough to find their scattered active experts while
    // avoiding full-expert waves once a smaller set already saturates the GPU.
    const float active_fraction = std::max(
        1.0f / static_cast<float>(num_experts_per_rank),
        1.0f - std::exp(-expected_tokens_per_expert));
    const float expected_rows_per_active_expert =
        expected_tokens_per_expert / active_fraction;
    const int expected_m_blocks_per_active_expert = std::max(
        ceil_div(static_cast<int>(std::ceil(expected_rows_per_active_expert)),
                 block_m),
        1);
    const int l1_n_blocks_per_expert = (2 * intermediate_hidden) / block_n;
    const float expected_l1_ctas_per_scheduled_expert =
        active_fraction * expected_m_blocks_per_active_expert *
        l1_n_blocks_per_expert;
    constexpr int kTargetCTAsPerSM = 4;
    int min_experts_to_fill_sms = static_cast<int>(std::ceil(
        kTargetCTAsPerSM * static_cast<float>(num_sms) /
        expected_l1_ctas_per_scheduled_expert));

    // Four-expert granularity keeps CTA distribution stable without encoding
    // model-local expert divisors such as 8/16/24/32 in a rule table.
    constexpr int kExpertWaveGranularity = 4;
    min_experts_to_fill_sms = std::max(1, min_experts_to_fill_sms);
    if (min_experts_to_fill_sms >= kExpertWaveGranularity)
        min_experts_to_fill_sms = align(
            min_experts_to_fill_sms, kExpertWaveGranularity);
    if (min_experts_to_fill_sms >= max_ring_safe_experts)
        return max_ring_safe_experts;

    // Among resource-sufficient waves, prefer a balanced final wave.  Search
    // only up to 2x the minimum so tail balancing cannot silently turn the
    // resource formula back into a full-expert model table.
    const int sweep_max = std::min(
        max_ring_safe_experts, 2 * min_experts_to_fill_sms);
    int best_wave = min_experts_to_fill_sms;
    float best_tail_ratio = -1.0f;
    for (int wave = min_experts_to_fill_sms;
         wave <= sweep_max; wave += kExpertWaveGranularity) {
        const int remainder = num_experts_per_rank % wave;
        const float tail_ratio = remainder == 0
            ? 1.0f : static_cast<float>(remainder) / wave;
        if (tail_ratio > best_tail_ratio) {
            best_tail_ratio = tail_ratio;
            best_wave = wave;
        }
    }
    return std::min(best_wave, max_ring_safe_experts);
}

// Experimental per-WG weight readiness. Shared-memory accounting and JIT
// emission must use this exact predicate; split-M consumes the whole N tile.
static bool use_sm90_fp4_n64_decode_ready(
    int block_m, int block_n, int num_epilogue_warpgroups,
    bool use_swap_ab, bool use_decode_done_mbarrier) {
    const int enabled = get_env<int>("DG_MEGA_MOE_FP4_N64_DECODE_READY", 0);
    DG_HOST_ASSERT(enabled == 0 or enabled == 1);
    return enabled != 0 and block_m == 64 and block_n == 128 and
           num_epilogue_warpgroups == 2 and use_swap_ab and
           use_decode_done_mbarrier;
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90_fp4(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& use_early_b_decode = false,
    const bool& use_decode_done_mbarrier = false,
    const bool& use_swap_ab = false,
    const bool& use_swap_ab_fast_amax = false) {
    constexpr int kSmemAlignment = 1024;

    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = block_m * (block_n / 2);
    const int smem_cd_l2 = block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd_swap_l1 = use_swap_ab
        ? block_m * (block_n / 2) *
              (use_swap_ab_fast_amax
                   ? static_cast<int>(sizeof(uint8_t))
                   : static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(uint8_t)))
        : 0;
    const int smem_cd_base = std::max(smem_cd_l1, smem_cd_l2);
    const int smem_cd = align(std::max(smem_cd_base, smem_cd_swap_l1), kSmemAlignment);

    const bool fp4_split_n_eligible =
        block_m == 64 and num_epilogue_warpgroups > 1 and
        block_n % num_epilogue_warpgroups == 0 and
        (block_n / num_epilogue_warpgroups) >= 64;
    const int kL2ActsSFGranK = block_n == 64 ? 32 : 64;
    const int wg_l1_out_block_n = fp4_split_n_eligible
        ? (block_n / num_epilogue_warpgroups) / 2
        : 0;
    const bool split_n_shares_sf =
        fp4_split_n_eligible and wg_l1_out_block_n < kL2ActsSFGranK;
    const int fp4_split_n_amax_scratch_slots = 32 * 2 * 2;
    const int smem_amax_scratch = split_n_shares_sf
        ? align(fp4_split_n_amax_scratch_slots * static_cast<int>(sizeof(uint32_t)),
                kSmemAlignment)
        : 0;
    const int l2_sfa_groups_per_block_k = block_k / kL2ActsSFGranK;
    const int smem_sfa_per_stage =
        align(l2_sfa_groups_per_block_k * block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfb_per_stage =
        align(block_n * static_cast<int>(sizeof(uint32_t)), 128);

    const int smem_b_decoded_per_stage = block_n * block_k;
    const int smem_b_packed_per_stage = block_n * (block_k / 2);
    const int smem_per_stage = block_m * block_k +
                               smem_b_decoded_per_stage +
                               smem_b_packed_per_stage +
                               smem_sfa_per_stage +
                               smem_sfb_per_stage;

    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_decode_full_per_stage = use_early_b_decode ? 8 : 0;
    const int decode_ready_groups = use_sm90_fp4_n64_decode_ready(
        block_m, block_n, num_epilogue_warpgroups,
        use_swap_ab, use_decode_done_mbarrier) ? 2 : 1;
    const int smem_decode_done_per_stage =
        use_decode_done_mbarrier ? 8 * decode_ready_groups : 0;
    const int smem_barriers_per_stage =
        2 * 8 + smem_decode_full_per_stage + smem_decode_done_per_stage;
    const int smem_fixed =
        smem_dispatch_size + smem_cd + smem_amax_scratch + smem_barriers_fixed;

    // Fill the available shared-memory budget; no model/density stage cap.
    const int num_stages = (smem_capacity - smem_fixed) /
                           (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(num_stages >= 2);
    const int smem_size =
        smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(smem_size <= smem_capacity);
    return {num_stages, smem_size};
}

static MegaMoESM90Config get_mega_moe_config_sm90_fp4(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens,
    const bool& use_early_b_decode = false,
    const bool& use_decode_done_mbarrier = false,
    const bool& use_swap_ab = false,
    const bool& use_swap_ab_fast_amax = false,
    const int& num_compute_ring_tokens = -1) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90_fp4(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const int block_k = 128;
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const int block_n = 128;
    int fp4_num_epilogue_warpgroups = num_epilogue_threads / 128;
    const bool fp4_flash_shape = intermediate_hidden <= 2048;
    const bool fp4_pro_shape = intermediate_hidden >= 3072;
    const bool fp4_flash_or_pro_shape = fp4_flash_shape or fp4_pro_shape;
    // Shape bands depend only on model shape and routing density; kernel bands add tile/thread constraints.
    const bool fp4_split_n_eligible =
        block_m == 64 and block_n % 128 == 0;
    const bool fp4_split_n_shape_band =
        (fp4_flash_or_pro_shape and
         expected_tokens_per_expert > 0.0f and
         expected_tokens_per_expert < kFP4SM90M128CrossoverRows);
    if (fp4_split_n_eligible and fp4_split_n_shape_band) {
        fp4_num_epilogue_warpgroups = 2;
    }
    DG_HOST_ASSERT(fp4_num_epilogue_warpgroups >= 1);
    DG_HOST_ASSERT((block_m / fp4_num_epilogue_warpgroups == 64) or
                   (block_m == 64 and fp4_num_epilogue_warpgroups > 1 and
                    block_n % fp4_num_epilogue_warpgroups == 0 and
                    (block_n / fp4_num_epilogue_warpgroups) >= 64));
    const int fp4_num_epilogue_threads = fp4_num_epilogue_warpgroups * 128;
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 0;

    const int num_sms = device_runtime->get_num_sms();
    int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90_fp4(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms,
        num_compute_ring_tokens < 0 ? num_max_pool_tokens :
            std::min(num_max_pool_tokens, num_compute_ring_tokens),
        num_max_tokens_per_rank, num_ranks);
    const int experts_per_wave_override =
        get_env<int>("DG_MEGA_MOE_FP4_EXPERTS_PER_WAVE", 0);
    DG_HOST_ASSERT(experts_per_wave_override >= 0);
    if (experts_per_wave_override > 0) {
        // Isolate wave scheduling from CTA budget during small-batch A/B.
        // Full-pool storage avoids bypassing the compute-ring lifetime bound.
        DG_HOST_ASSERT(num_compute_ring_tokens < 0 or
                       num_compute_ring_tokens >= num_max_pool_tokens);
        DG_HOST_ASSERT(experts_per_wave_override <= num_experts_per_rank);
        num_experts_per_wave = experts_per_wave_override;
    }
    const bool fp4_small_block_n_kernel =
        block_m == 64 and block_n == 128;
    const bool fp4_split_n_decode_thread_kernel_band =
        fp4_small_block_n_kernel and fp4_split_n_shape_band;
    const bool fp4_2wg_decode_offload_kernel_band =
        block_m == 128 and block_n == 128 and
        fp4_num_epilogue_threads == 256 and
        expected_tokens_per_expert >= kFP4SM90M128CrossoverRows;
    const bool fp4_decode_assist_thread_kernel_band =
        fp4_2wg_decode_offload_kernel_band or
        (fp4_small_block_n_kernel and
         expected_tokens_per_expert > 0.0f and expected_tokens_per_expert <= 24.0f);
    const int default_num_dispatch_threads =
        (fp4_split_n_decode_thread_kernel_band or
         fp4_decode_assist_thread_kernel_band) ? 64 : 128;
    const int num_dispatch_threads = default_num_dispatch_threads;
    DG_HOST_ASSERT(num_dispatch_threads == 64 or num_dispatch_threads == 128);
    // Once a split-N expert has enough rows, four decode-assist warps are
    // sufficient to cover packed-FP4 conversion.  Keeping eight helpers for
    // this middle-density band over-subscribes the SM warp schedulers and
    // delays WGMMA/TMA issue.  Very sparse experts remain latency-sensitive:
    // retain the original eight helpers below six expected rows/expert.
    const bool fp4_middle_density_decode_assist_kernel_band =
        fp4_split_n_decode_thread_kernel_band and
        expected_tokens_per_expert >= 6.0f;
    int num_non_epilogue_threads =
        fp4_split_n_decode_thread_kernel_band ?
            (fp4_middle_density_decode_assist_kernel_band ? 192 : 320) :
        (fp4_decode_assist_thread_kernel_band ? 192 : 128);
    // Experimental topology A/B under the WG-uniform full-LTO register
    // profile. Reuse the existing eight-helper implementation, without
    // changing tile/stage, protocol, or the M128 path.
    const int eight_helpers = get_env<int>("DG_MEGA_MOE_FP4_EIGHT_HELPERS", 0);
    DG_HOST_ASSERT(eight_helpers >= 0 and eight_helpers <= 2);
    const int eight_math_registers = get_env<int>(
        "DG_MEGA_MOE_FP4_EIGHT_MATH_REGISTERS", 0);
    DG_HOST_ASSERT(eight_math_registers >= 0 and eight_math_registers <= 2);
    // Helper mode 2 keeps four helpers at higher density unless the paired
    // density-register policy is selected. That policy retains the original
    // low-density quotas and gives higher-density eight-helper math WGs 144
    // registers. It is opt-in pending broader EP/shape/skew validation.
    // Neither the threshold nor selection depends on a model or exact batch.
    if (eight_math_registers == 2)
        DG_HOST_ASSERT(eight_helpers == 2);
    const bool extend_eight_helpers = eight_helpers == 1 or
        (eight_helpers == 2 and
         (expected_tokens_per_expert < 32.0f or eight_math_registers == 2));
    if (extend_eight_helpers and num_ranks > 8 and
        block_m == 64 and block_n == 128 and block_k == 128 and
        num_dispatch_threads == 64 and num_non_epilogue_threads == 192 and
        fp4_num_epilogue_threads == 256) {
        DG_HOST_ASSERT(use_sm90_fp4_n64_decode_ready(
            block_m, block_n, fp4_num_epilogue_warpgroups,
            use_swap_ab, use_decode_done_mbarrier));
        DG_HOST_ASSERT(get_env<int>("DG_MEGA_MOE_FP4_BALANCED_WG_REGISTERS", 0) == 1);
        DG_HOST_ASSERT(get_env<int>("DG_MEGA_MOE_FP4_DECODE_REGISTER_BOOST", 0) == 1);
        num_non_epilogue_threads = 320;
    }
    DG_HOST_ASSERT(num_non_epilogue_threads >= 128 and
                   num_non_epilogue_threads % 64 == 0);
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90_fp4(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, fp4_num_epilogue_threads / 32,
        use_early_b_decode, use_decode_done_mbarrier,
        use_swap_ab, use_swap_ab_fast_amax);

    const auto config = MegaMoESM90Config {
        block_m, block_n, block_k,
        cluster_size,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, fp4_num_epilogue_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoESM90FP4Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, early_b_decode={}, decode_done_mbarrier={}, swap_ab={}, swap_ab_fast_amax={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk,
            use_early_b_decode, use_decode_done_mbarrier,
            use_swap_ab, use_swap_ab_fast_amax);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

static MegaMoESM90Config get_mega_moe_config_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens,
    const int& num_compute_ring_tokens) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_topk, num_tokens);
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn =
        block_m == 128 and num_epilogue_threads == 512;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    const bool decode_use_block_n_256 =
        decode_split_n_path and intermediate_hidden >= 2048 and
        expected_tokens_per_expert >= 0.25f and
        (2 * intermediate_hidden) % 256 == 0 and hidden % 256 == 0;
    const bool use_swap_ab = should_use_swap_ab_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        block_m, num_epilogue_threads, hidden, intermediate_hidden);
    int block_n = use_swap_ab ? 256
                              : (auto_split_mn ? 256 :
                                 (decode_use_block_n_256 ? 256 : 128));
    const int block_k = 128;
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    const int num_sms = device_runtime->get_num_sms();
    int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms,
        num_compute_ring_tokens, num_max_tokens_per_rank, num_ranks);
    const bool reduce_decode_threads = num_epilogue_threads == 128;
    const bool decode_split_n =
        block_m == 64 and num_epilogue_threads == 256;
    const bool shrink_non_epilogue = reduce_decode_threads or decode_split_n;
    const int num_dispatch_threads =
        (num_epilogue_threads == 512 or shrink_non_epilogue) ? 64 : 128;
    const bool split_sfa_loader_warp = false;
    const int num_non_epilogue_threads =
        split_sfa_loader_warp ? 128 :
            ((num_epilogue_threads == 512 or shrink_non_epilogue) ? 64 : 128);
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        use_swap_ab);

    const auto config = MegaMoESM90Config {
        block_m, block_n, block_k,
        cluster_size,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoESM90Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, swap_ab={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk,
            use_swap_ab);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
