#pragma once

#include "mega_moe.hpp"
#include "sm90.hpp"

namespace deep_gemm {

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
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn = expected_tokens_per_expert >= 64.0f;
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
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
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
        intermediate_hidden, block_m, block_n, num_sms);
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps) {
    constexpr int kSmemAlignment = 1024;

    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    const int smem_cd_l1 = block_m * (block_n / 2);
    const int smem_cd_l2 = block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

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
    const bool auto_split_mn = expected_tokens_per_expert >= 64.0f;
    int block_m = auto_split_mn ? 128 : 64;
    int num_epilogue_warpgroups = auto_split_mn ? 2 : block_m / 64;
    block_m = get_env<int>("DG_SM90_FP4_BLOCK_M", get_env<int>("DG_SM90_BLOCK_M", block_m));
    num_epilogue_warpgroups =
        get_env<int>("DG_SM90_NUM_EPILOGUE_WARPGROUPS", num_epilogue_warpgroups);
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
    const bool& use_rs_mode = false) {
    if (const int override_value = get_env<int>("DG_SM90_NUM_EXPERTS_PER_WAVE", 0);
        override_value > 0) {
        DG_HOST_ASSERT(override_value <= num_experts_per_rank and
                       num_experts_per_rank % override_value == 0);
        return override_value;
    }

    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    if (use_rs_mode and block_m == 64 and block_n == 64 and
        expected_tokens_per_expert >= 1.0f and expected_tokens_per_expert < 2.0f) {
        return num_experts_per_rank;
    }
    if (!use_rs_mode and block_m == 64 and block_n == 128 and
        intermediate_hidden >= 3072 and
        expected_tokens_per_expert >= 1.0f and expected_tokens_per_expert < 1.5f and
        num_experts_per_rank % 8 == 0) {
        return 8;
    }
    if (expected_tokens_per_expert < 1.0f or expected_tokens_per_expert > 4.0f) {
        return num_experts_per_rank;
    }
    return get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90_fp4(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& use_rs_mode = false,
    const bool& use_rs_stage_sfb = false,
    const bool& use_early_b_decode = false,
    const bool& use_decode_done_mbarrier = false,
    const int& default_num_stages_cap = 0) {
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
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    const int kL2ActsSFGranK = block_n == 64 ? 32 : 64;
    const bool fp4_split_n_eligible =
        not use_rs_mode and block_m == 64 and num_epilogue_warpgroups > 1 and
        block_n % num_epilogue_warpgroups == 0 and
        (block_n / num_epilogue_warpgroups) >= 64;
    const int wg_l1_out_block_n = fp4_split_n_eligible
        ? (block_n / num_epilogue_warpgroups) / 2
        : 0;
    const bool split_n_shares_sf =
        fp4_split_n_eligible and wg_l1_out_block_n < kL2ActsSFGranK;
    const int smem_amax_scratch = split_n_shares_sf
        ? align(64 * static_cast<int>(sizeof(uint32_t)), kSmemAlignment)
        : 0;
    const int smem_rs_accum = use_rs_mode
        ? align(block_m * 64 * static_cast<int>(sizeof(float)), kSmemAlignment)
        : 0;

    const int l2_sfa_groups_per_block_k = block_n == 64 ? 4 : 2;
    const int smem_sfa_per_stage =
        align(l2_sfa_groups_per_block_k * block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfb_per_stage = (use_rs_mode and not use_rs_stage_sfb)
        ? 0
        : align(block_n * static_cast<int>(sizeof(uint32_t)), 128);

    const int smem_b_decoded_per_stage = use_rs_mode ? 0 : block_n * block_k;
    const int smem_b_packed_per_stage = block_n * (block_k / 2);
    const int smem_per_stage = block_m * block_k +
                               smem_b_decoded_per_stage +
                               smem_b_packed_per_stage +
                               smem_sfa_per_stage + smem_sfb_per_stage;

    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_decode_full_per_stage = use_early_b_decode ? 8 : 0;
    const int smem_decode_done_per_stage =
        (!use_rs_mode and use_decode_done_mbarrier) ? 8 : 0;
    const int smem_barriers_per_stage =
        2 * 8 + smem_decode_full_per_stage + smem_decode_done_per_stage;
    const int smem_fixed =
        smem_dispatch_size + smem_cd + smem_amax_scratch + smem_rs_accum + smem_barriers_fixed;

    const int max_num_stages = (smem_capacity - smem_fixed) /
                               (smem_per_stage + smem_barriers_per_stage);
    int num_stages = max_num_stages;
    if (default_num_stages_cap > 0) {
        num_stages = std::min(num_stages, default_num_stages_cap);
    }
    if (const int override_num_stages = get_env<int>("DG_SM90_FP4_NUM_STAGES", 0);
        override_num_stages > 0) {
        num_stages = std::min(max_num_stages, override_num_stages);
    } else if (const int override_num_stages = get_env<int>("DG_SM90_NUM_STAGES", 0);
               override_num_stages > 0) {
        num_stages = std::min(max_num_stages, override_num_stages);
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
    const bool& use_rs_mode = false,
    const bool& use_rs_stage_sfb = false,
    const bool& use_early_b_decode = false,
    const bool& use_decode_done_mbarrier = false) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90_fp4(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const int default_block_n = use_rs_mode ? 64 : 128;
    const int block_n = get_env<int>("DG_SM90_FP4_BLOCK_N", default_block_n);
    DG_HOST_ASSERT(block_n == 64 or block_n == 128);
    const int block_k = 128;

    int fp4_num_epilogue_warpgroups = num_epilogue_threads / 128;
    const bool fp4_split_n_eligible =
        not use_rs_mode and block_m == 64 and block_n == 128;
    if (fp4_split_n_eligible and get_env<int>("DG_SM90_FP4_SPLIT_N", 0) != 0) {
        fp4_num_epilogue_warpgroups = 2;
    }
    fp4_num_epilogue_warpgroups = get_env<int>(
        "DG_SM90_FP4_NUM_EPILOGUE_WARPGROUPS", fp4_num_epilogue_warpgroups);
    DG_HOST_ASSERT(fp4_num_epilogue_warpgroups >= 1);
    DG_HOST_ASSERT((block_m / fp4_num_epilogue_warpgroups == 64) or
                   (block_m == 64 and fp4_num_epilogue_warpgroups > 1 and
                    block_n % fp4_num_epilogue_warpgroups == 0 and
                    (block_n / fp4_num_epilogue_warpgroups) >= 64));
    DG_HOST_ASSERT(not use_rs_mode or fp4_num_epilogue_warpgroups == 1);
    const int fp4_num_epilogue_threads = fp4_num_epilogue_warpgroups * 128;
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = use_rs_mode ? 64 : 0;

    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90_fp4(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms, use_rs_mode);

    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const bool fp4_decode_heavy_small_batch =
        !use_rs_mode and block_m == 64 and block_n == 128 and
        expected_tokens_per_expert > 0.0f and expected_tokens_per_expert <= 24.0f;
    const bool fp4_2wg_decode_offload_band =
        !use_rs_mode and block_m == 128 and block_n == 128 and
        fp4_num_epilogue_threads == 256 and expected_tokens_per_expert >= 64.0f;
    const int default_num_dispatch_threads =
        (fp4_decode_heavy_small_batch or fp4_2wg_decode_offload_band) ? 64 : 128;
    const int num_dispatch_threads =
        get_env<int>("DG_SM90_FP4_NUM_DISPATCH_THREADS", default_num_dispatch_threads);
    DG_HOST_ASSERT(num_dispatch_threads == 64 or num_dispatch_threads == 128);
    const int default_num_non_epilogue_threads =
        (fp4_decode_heavy_small_batch or fp4_2wg_decode_offload_band) ? 192 : 128;
    const int num_non_epilogue_threads =
        get_env<int>("DG_SM90_FP4_NUM_NON_EPILOGUE_THREADS", default_num_non_epilogue_threads);
    DG_HOST_ASSERT(num_non_epilogue_threads >= 128 and
                   num_non_epilogue_threads % 64 == 0);
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    const bool fp4_stage4_decode_band =
        !use_rs_mode and block_m == 64 and block_n == 128 and
        expected_tokens_per_expert >= 6.0f and expected_tokens_per_expert < 12.0f;
    const bool fp4_stage6_decode_band =
        !use_rs_mode and block_m == 64 and block_n == 128 and
        ((expected_tokens_per_expert >= 0.375f and expected_tokens_per_expert < 0.75f) or
         (expected_tokens_per_expert >= 1.5f and expected_tokens_per_expert < 3.0f));
    const bool fp4_stage5_decode_heavy_batch =
        !use_rs_mode and block_m == 64 and block_n == 128 and
        expected_tokens_per_expert >= 1.5f and expected_tokens_per_expert <= 24.0f;
    const int default_num_stages_cap =
        fp4_stage4_decode_band ? 4 :
        (fp4_stage6_decode_band ? 6 : (fp4_stage5_decode_heavy_batch ? 5 : 0));

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90_fp4(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, fp4_num_epilogue_threads / 32,
        use_rs_mode, use_rs_stage_sfb, use_early_b_decode, use_decode_done_mbarrier,
        default_num_stages_cap);

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
            "MegaMoESM90FP4Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, rs_mode={}, early_b_decode={}, decode_done_mbarrier={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk,
            use_rs_mode, use_early_b_decode, use_decode_done_mbarrier);
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
    const int& num_padded_sf_pool_tokens) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn = expected_tokens_per_expert >= 64.0f;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    const bool decode_use_block_n_256 =
        decode_split_n_path and intermediate_hidden >= 3072 and
        expected_tokens_per_expert >= 0.25f and
        (2 * intermediate_hidden) % 256 == 0;
    const int block_n = auto_split_mn ? 256
                                      : (decode_use_block_n_256 ? 256 : 128);
    const int block_k = 128;
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);

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
        num_dispatch_threads / 32, num_epilogue_threads / 32);

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
            "MegaMoESM90Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
