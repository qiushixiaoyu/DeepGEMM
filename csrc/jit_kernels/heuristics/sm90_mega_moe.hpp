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
    constexpr float kSwapAbMaxTokensPerExpert = 16.0f;
    const bool light_weight_decode =
        weight_light and expected_tokens_per_expert > 0.0f and
        expected_tokens_per_expert <= kSwapAbMaxTokensPerExpert;
    constexpr float kStreamingSwapAbMaxTokensPerExpert = 24.0f;
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
        expected_tokens_per_expert >= 64.0f;
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

struct FP4SM90WaveRule {
    float min_tokens_per_expert;
    float max_tokens_per_expert;
    bool include_min;
    int required_expert_divisor;
    int num_experts_per_wave;
};

enum class FP4SM90StageShape {
    Any,
    Flash,
    Pro,
    NotPro,
};

struct FP4SM90StageCapRule {
    float min_tokens_per_expert;
    float max_tokens_per_expert;
    bool include_min;
    bool include_max;
    FP4SM90StageShape shape;
    int num_stages_cap;
};

static bool try_get_num_experts_per_wave_for_sm90_fp4(
    const FP4SM90WaveRule* rules, const int& num_rules,
    const float& expected_tokens_per_expert, const int& num_experts_per_rank,
    int& num_experts_per_wave) {
    for (int i = 0; i < num_rules; ++ i) {
        const auto& rule = rules[i];
        const bool in_lower_bound = rule.include_min
            ? expected_tokens_per_expert >= rule.min_tokens_per_expert
            : expected_tokens_per_expert > rule.min_tokens_per_expert;
        if (!in_lower_bound or expected_tokens_per_expert >= rule.max_tokens_per_expert)
            continue;

        if (rule.num_experts_per_wave == 0) {
            if (num_experts_per_rank <= 0)
                continue;
            num_experts_per_wave = num_experts_per_rank;
            return true;
        }
        DG_HOST_ASSERT(rule.required_expert_divisor > 0);
        if (num_experts_per_rank % rule.required_expert_divisor == 0) {
            num_experts_per_wave = rule.num_experts_per_wave;
            return true;
        }
    }
    return false;
}

static bool fp4_sm90_stage_shape_matches(
    const FP4SM90StageShape& shape, const bool& fp4_flash_shape, const bool& fp4_pro_shape) {
    switch (shape) {
        case FP4SM90StageShape::Any:
            return true;
        case FP4SM90StageShape::Flash:
            return fp4_flash_shape;
        case FP4SM90StageShape::Pro:
            return fp4_pro_shape;
        case FP4SM90StageShape::NotPro:
            return !fp4_pro_shape;
    }
    DG_HOST_ASSERT(false);
    return false;
}

static int get_default_num_stages_cap_for_mega_moe_sm90_fp4(
    const int& intermediate_hidden, const int& block_m, const int& block_n,
    const float& expected_tokens_per_expert) {
    if (!(block_m == 64 and block_n == 128)) {
        return 0;
    }

    const bool fp4_flash_shape = intermediate_hidden <= 2048;
    const bool fp4_pro_shape = intermediate_hidden >= 3072;
    // Ordered first-match rules preserve the historical stage-cap priority.
    static constexpr FP4SM90StageCapRule stage_cap_rules[] = {
        {6.0f, 12.0f, true, false, FP4SM90StageShape::Flash, 5},
        {3.0f, 6.0f, false, false, FP4SM90StageShape::Flash, 4},
        {0.0f, 0.25f, false, false, FP4SM90StageShape::Pro, 5},
        {0.375f, 0.75f, true, false, FP4SM90StageShape::Pro, 5},
        {1.5f, 3.0f, true, false, FP4SM90StageShape::Pro, 5},
        {1.0f, 1.5f, true, false, FP4SM90StageShape::Pro, 5},
        {24.0f, 64.0f, true, false, FP4SM90StageShape::Pro, 5},
        {0.375f, 0.75f, true, false, FP4SM90StageShape::Any, 6},
        {3.0f, 6.0f, false, false, FP4SM90StageShape::Flash, 6},
        {1.5f, 3.0f, true, false, FP4SM90StageShape::NotPro, 6},
        {1.5f, 24.0f, true, true, FP4SM90StageShape::Any, 5},
    };
    for (const auto& rule: stage_cap_rules) {
        const bool in_lower_bound = rule.include_min
            ? expected_tokens_per_expert >= rule.min_tokens_per_expert
            : expected_tokens_per_expert > rule.min_tokens_per_expert;
        const bool in_upper_bound = rule.include_max
            ? expected_tokens_per_expert <= rule.max_tokens_per_expert
            : expected_tokens_per_expert < rule.max_tokens_per_expert;
        if (in_lower_bound and in_upper_bound and
            fp4_sm90_stage_shape_matches(rule.shape, fp4_flash_shape, fp4_pro_shape)) {
            return rule.num_stages_cap;
        }
    }
    return 0;
}

static int get_num_experts_per_wave_for_mega_moe_sm90_fp4(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms,
    const int& num_ring_tokens, const int& num_max_tokens_per_rank, const int& num_ranks) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const bool fp4_compact_n256 =
        get_env<int>("DG_MEGA_MOE_FP4_COMPACT_N256", 0) != 0;
    const bool fp4_small_block_n_kernel =
        block_m == 64 and
        (block_n == 128 or (fp4_compact_n256 and block_n == 256));
    const bool fp4_flash_shape = intermediate_hidden <= 2048;
    const bool fp4_pro_shape = intermediate_hidden >= 3072;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk,
        num_experts_per_rank);
    // The tuned FP4 wave table predates compact compute pools.  Cap every
    // table result with the common ring-aware heuristic so a complete L1 wave
    // fits before L2 must retire and recycle any physical slot.  Use the
    // current launch size for this bound; the ring is sized from the buffer
    // maximum, but inactive SymmBuffer capacity must not throttle decode.
    const int ring_safe_num_experts_per_wave =
        num_ring_tokens < num_max_pool_tokens ?
            get_num_experts_per_wave_for_mega_moe(
                num_experts_per_rank, num_tokens, num_topk,
                intermediate_hidden, block_m, block_n, num_sms,
                num_ring_tokens, align(num_tokens, block_m), num_ranks) :
            num_experts_per_rank;
    int fp4_num_experts_per_wave = 0;
    if (fp4_small_block_n_kernel and fp4_flash_shape) {
        static constexpr FP4SM90WaveRule flash_wave_rules[] = {
            {0.75f, 1.0f, true, 16, 16},
            {1.5f, 2.0f, true, 16, 16},
            {3.0f, 6.0f, true, 16, 16},
            {6.0f, 12.0f, true, 32, 32},
            {6.0f, 12.0f, true, 8, 8},
            {24.0f, 32.0f, true, 16, 16},
            {12.0f, 24.0f, true, 32, 32},
            {12.0f, 24.0f, true, 8, 8},
            {32.0f, 64.0f, true, 16, 16},
        };
        if (try_get_num_experts_per_wave_for_sm90_fp4(
                flash_wave_rules,
                static_cast<int>(sizeof(flash_wave_rules) / sizeof(flash_wave_rules[0])),
                expected_tokens_per_expert, num_experts_per_rank,
                fp4_num_experts_per_wave)) {
            return std::min(fp4_num_experts_per_wave,
                            ring_safe_num_experts_per_wave);
        }
    }
    if (fp4_small_block_n_kernel and fp4_pro_shape) {
        static constexpr FP4SM90WaveRule pro_wave_rules[] = {
            {0.0f, 0.25f, false, 16, 16},
            {0.25f, 0.375f, true, 16, 16},
            {0.375f, 0.75f, true, 16, 16},
            {0.25f, 1.0f, true, 24, 24},
            {1.0f, 1.5f, true, 1, 0},
            {1.5f, 3.0f, true, 16, 16},
            {3.0f, 6.0f, true, 8, 8},
            {6.0f, 12.0f, true, 16, 16},
            {6.0f, 12.0f, true, 8, 8},
            {12.0f, 24.0f, true, 24, 24},
            {12.0f, 24.0f, true, 8, 8},
            {24.0f, 64.0f, true, 8, 8},
        };
        if (try_get_num_experts_per_wave_for_sm90_fp4(
                pro_wave_rules,
                static_cast<int>(sizeof(pro_wave_rules) / sizeof(pro_wave_rules[0])),
                expected_tokens_per_expert, num_experts_per_rank,
                fp4_num_experts_per_wave)) {
            return std::min(fp4_num_experts_per_wave,
                            ring_safe_num_experts_per_wave);
        }
    }
    if (expected_tokens_per_expert < 1.0f or expected_tokens_per_expert > 4.0f) {
        return ring_safe_num_experts_per_wave;
    }
    return get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms,
        num_ring_tokens, num_max_tokens_per_rank, num_ranks);
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90_fp4(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& use_early_b_decode = false,
    const bool& use_decode_done_mbarrier = false,
    const int& default_num_stages_cap = 0,
    const bool& use_swap_ab = false,
    const bool& use_swap_ab_fast_amax = false,
    const bool& use_compact_n256 = false) {
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

    // Compact N256 keeps one decoded N64 slot per split-N math WG. Each WG
    // consumes its first internal-N64 subtile, releases the slot, then reuses
    // it for the second. Packed-B remains a full coalesced TMA tile.
    const int smem_b_decoded_per_stage = use_compact_n256
        ? 2 * 64 * block_k
        : block_n * block_k;
    const int smem_b_packed_per_stage = block_n * (block_k / 2);
    const int smem_per_stage = block_m * block_k +
                               smem_b_decoded_per_stage +
                               smem_b_packed_per_stage +
                               smem_sfa_per_stage +
                               smem_sfb_per_stage;

    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_decode_full_per_stage = use_early_b_decode ? 8 : 0;
    const int smem_decode_done_per_stage =
        use_decode_done_mbarrier and not use_compact_n256 ? 8 : 0;
    const int smem_compact_n256_per_stage = use_compact_n256
        ? (2 * 2 + 2) * 8  // ready[group][subtile] + release[group]
        : 0;
    const int smem_barriers_per_stage =
        2 * 8 + smem_decode_full_per_stage + smem_decode_done_per_stage +
        smem_compact_n256_per_stage;
    const int smem_fixed =
        smem_dispatch_size + smem_cd + smem_amax_scratch + smem_barriers_fixed;

    const int max_num_stages = (smem_capacity - smem_fixed) /
                               (smem_per_stage + smem_barriers_per_stage);
    int num_stages = max_num_stages;
    if (default_num_stages_cap > 0) {
        num_stages = std::min(num_stages, default_num_stages_cap);
    }
    DG_HOST_ASSERT(num_stages >= 2);
    return {num_stages,
            smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage)};
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
    const bool fp4_flash_shape = intermediate_hidden <= 2048;
    const bool fp4_pro_shape = intermediate_hidden >= 3072;
    const bool fp4_flash_or_pro_shape = fp4_flash_shape or fp4_pro_shape;
    const bool use_compact_n256 =
        get_env<int>("DG_MEGA_MOE_FP4_COMPACT_N256", 0) != 0 and
        use_swap_ab and block_m == 64 and fp4_flash_or_pro_shape and
        expected_tokens_per_expert > 0.0f and
        expected_tokens_per_expert < 64.0f;
    const int block_n = use_compact_n256 ? 256 : 128;
    int fp4_num_epilogue_warpgroups = num_epilogue_threads / 128;
    // Shape bands depend only on model shape and routing density; kernel bands add tile/thread constraints.
    const bool fp4_split_n_eligible =
        block_m == 64 and block_n % 128 == 0;
    const bool fp4_split_n_shape_band =
        (fp4_flash_or_pro_shape and
         expected_tokens_per_expert > 0.0f and
         expected_tokens_per_expert < 64.0f);
    if (fp4_split_n_eligible and fp4_split_n_shape_band) {
        fp4_num_epilogue_warpgroups = 2;
    }
    DG_HOST_ASSERT(not use_compact_n256 or
                   (block_n == 256 and fp4_num_epilogue_warpgroups == 2));
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
    const bool fp4_small_block_n_kernel =
        block_m == 64 and
        (block_n == 128 or use_compact_n256);
    const bool fp4_split_n_decode_thread_kernel_band =
        fp4_small_block_n_kernel and fp4_split_n_shape_band;
    const bool fp4_2wg_decode_offload_kernel_band =
        block_m == 128 and block_n == 128 and
        fp4_num_epilogue_threads == 256 and expected_tokens_per_expert >= 64.0f;
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
    const int num_non_epilogue_threads =
        fp4_split_n_decode_thread_kernel_band ?
            (fp4_middle_density_decode_assist_kernel_band ? 192 : 320) :
        (fp4_decode_assist_thread_kernel_band ? 192 : 128);
    DG_HOST_ASSERT(num_non_epilogue_threads >= 128 and
                   num_non_epilogue_threads % 64 == 0);
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    const int default_num_stages_cap = get_default_num_stages_cap_for_mega_moe_sm90_fp4(
        intermediate_hidden, block_m, block_n, expected_tokens_per_expert);
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90_fp4(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, fp4_num_epilogue_threads / 32,
        use_early_b_decode, use_decode_done_mbarrier, default_num_stages_cap,
        use_swap_ab, use_swap_ab_fast_amax, use_compact_n256);

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
            "MegaMoESM90FP4Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, early_b_decode={}, decode_done_mbarrier={}, swap_ab={}, swap_ab_fast_amax={}, compact_n256={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk,
            use_early_b_decode, use_decode_done_mbarrier,
            use_swap_ab, use_swap_ab_fast_amax, use_compact_n256);
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
