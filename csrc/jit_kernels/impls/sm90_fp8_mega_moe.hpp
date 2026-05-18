#pragma once

#include <torch/python.h>
#include <algorithm>
#include <cstdlib>
#include <string>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

static int get_active_sms_for_sm90_mega_moe(
    const int& physical_num_sms,
    const int& num_tokens,
    const int& num_topk,
    const MegaMoESM90Config& config) {
    DG_HOST_ASSERT(physical_num_sms >= 1);

    const int forced_num_sms = get_env<int>("DG_SM90_MEGA_MOE_ACTIVE_SMS", 0);
    if (forced_num_sms > 0)
        return std::clamp(forced_num_sms, 1, physical_num_sms);

    if (not get_env<int>("DG_SM90_MEGA_MOE_ENABLE_ACTIVE_SMS_HEURISTIC", 0))
        return physical_num_sms;

    // Small-batch decode runs spend a large fraction of time in the persistent
    // kernel's grid-wide dispatch / count / barrier work.  Keep the full-grid
    // path for larger tiles, but reduce the number of participating CTAs when
    // the expected per-rank routed slots are tiny.  The scheduler still covers
    // all L1/L2 blocks by striding over the smaller logical grid.
    if (config.block_m != 64)
        return physical_num_sms;

    const int num_slots_per_rank = num_tokens * num_topk;
    int active_sms = physical_num_sms;
    if (num_slots_per_rank <= 12) {
        active_sms = 16;
    } else if (num_slots_per_rank <= 24) {
        active_sms = 24;
    } else if (num_slots_per_rank <= 48) {
        active_sms = 32;
    } else if (num_slots_per_rank <= 96) {
        active_sms = 48;
    }

    // At least two CTAs are required by workspace cleanup paths that split
    // work between sm_idx == 0 and sm_idx > 0.
    return std::clamp(active_sms, std::min(2, physical_num_sms), physical_num_sms);
}

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE host runtime
// ----------------------------------------------------------------------------
// This is the SM90 counterpart of `SM100FP8FP4MegaMoERuntime`. The kernel
// itself lives in `deep_gemm/impls/sm90_fp8_mega_moe.cuh` and is currently a
// skeleton: dispatch/combine paths are intended to be portable from the SM100
// version, while the GEMM (TMA load + WGMMA + epilogue) is being implemented
// in a follow-up step.
//
// Differences from SM100 path:
//   * Activations and weights are both FP8 (e4m3); no FP4.
//   * Activation/weight scale factors (SF) are per-128-channel float (not UE8M0
//     int + per-32 UTCCP layout).
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
        uint64_t* stage_profile;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const char* dbg_env = std::getenv("DG_DEBUG_SCHED_TRACE");
        const bool dbg_on = dbg_env != nullptr && std::string(dbg_env) != "0";
        const bool l2_dual_accum_on = get_env<int>("DG_SM90_MEGA_MOE_L2_DUAL_ACCUM", 0) != 0;
        const bool sfb_smem_on = get_env<int>("DG_SM90_MEGA_MOE_SFB_SMEM", 0) != 0;
        const bool l2_sfa_pair_tma_on = get_env<int>("DG_SM90_MEGA_MOE_L2_SFA_PAIR_TMA", 0) != 0;
        const bool l1_direct_store_on = get_env<int>("DG_SM90_MEGA_MOE_L1_DIRECT_STORE", 0) != 0;
        const bool l2_act_sf_per128_on = get_env<int>("DG_SM90_MEGA_MOE_L2_ACT_SF_PER128", 0) != 0;
        const bool stage_profile_on = get_env<int>("DG_SM90_MEGA_MOE_STAGE_PROFILE", 0) != 0;
        const bool split_a_sfa_producer_on = get_env<int>("DG_SM90_MEGA_MOE_SPLIT_A_SFA_PRODUCER", 0) != 0;
        const int force_combine_chunks = get_env<int>("DG_SM90_MEGA_MOE_FORCE_COMBINE_CHUNKS", 0);
        DG_HOST_ASSERT(force_combine_chunks == 0 or force_combine_chunks == 1 or force_combine_chunks == 2);
        DG_HOST_ASSERT(not (l2_act_sf_per128_on and l2_dual_accum_on));
        DG_HOST_ASSERT(not (l2_act_sf_per128_on and l2_sfa_pair_tma_on));
        return fmt::format(R"(
    // wave_v2_active_sms_v1_dbg_trace_v21_split_a_sfa_producer (bump to invalidate JIT cache when sm90_fp8_mega_moe.cuh changes)
    {}{}{}{}{}{}{}{}{}#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

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
        {}, {}, {},
        {}, {},
        {},
        {}
    >);
}};
    )", dbg_on ? "#define DG_DEBUG_SCHED_TRACE\n" : "",
    l2_dual_accum_on ? "#define DG_SM90_MEGA_MOE_L2_DUAL_ACCUM 1\n" : "",
    sfb_smem_on ? "#define DG_SM90_MEGA_MOE_SFB_SMEM 1\n" : "",
    l2_sfa_pair_tma_on ? "#define DG_SM90_MEGA_MOE_L2_SFA_PAIR_TMA 1\n" : "",
    l1_direct_store_on ? "#define DG_SM90_MEGA_MOE_L1_DIRECT_STORE 1\n" : "",
    l2_act_sf_per128_on ? "#define DG_SM90_MEGA_MOE_L2_ACT_SF_PER128 1\n" : "",
    stage_profile_on ? "#define DG_SM90_MEGA_MOE_STAGE_PROFILE 1\n" : "",
    split_a_sfa_producer_on ? "#define DG_SM90_MEGA_MOE_SPLIT_A_SFA_PRODUCER 1\n" : "",
    force_combine_chunks > 0 ? fmt::format("#define DG_SM90_MEGA_MOE_FORCE_COMBINE_CHUNKS {}\n", force_combine_chunks) : "",
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
    args.fast_math ? "true" : "false");
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
            args.stage_profile
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
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    uint64_t* stage_profile = nullptr
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Heuristics
    const auto config = get_mega_moe_config_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens);

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-64 for L2 (MN-major, no swizzle).
    // Weight SF: block (128, 128) raw float pointer (no TMA descriptor).
    constexpr int kGranK = 128;
    const bool l2_act_sf_per128_on = get_env<int>("DG_SM90_MEGA_MOE_L2_ACT_SF_PER128", 0) != 0;
    constexpr int kL2ActsSFGranKDefault = 64;
    constexpr int kL2ActsSFGranKPer128 = 128;
    const int kL2ActsSFGranK = l2_act_sf_per128_on ? kL2ActsSFGranKPer128 : kL2ActsSFGranKDefault;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    // L1 output (post-SwiGLU FP8): N is halved. The SM90 epilogue writes this
    // staging tile to SMEM as plain row-major bytes, so the TMA store descriptor
    // must use no shared-memory swizzle. Later L2 TMA loads may still swizzle
    // from this row-major global buffer into their own SMEM tile.
    // The TMA store is issued *per warpgroup*, each writing a `WG_BLOCK_M`
    // (= block_m / num_epilogue_warpgroups) row tile from its own SMEM offset.
    // The descriptor outer-box dim therefore must be `WG_BLOCK_M`, not block_m.
    const int num_epilogue_warpgroups_h = config.num_epilogue_threads / 128;
    const int wg_block_m = config.block_m / num_epilogue_warpgroups_h;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       config.block_n / 2, wg_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const bool l2_sfa_pair_tma_on = get_env<int>("DG_SM90_MEGA_MOE_L2_SFA_PAIR_TMA", 0) != 0;
    const auto tensor_map_l2_acts_sf = [&]() {
        if (l2_act_sf_per128_on) {
            const int raw_cols = intermediate_hidden / kL2ActsSFGranKDefault;
            const auto l2_acts_sf_final = l2_acts_sf.slice(1, raw_cols, raw_cols + intermediate_hidden / kL2ActsSFGranKPer128);
            return make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf_final,
                                    config.num_padded_sf_pool_tokens, intermediate_hidden,
                                    config.block_m, kL2ActsSFGranKPer128,
                                    1, 0);
        }
        if (l2_sfa_pair_tma_on) {
            const int shape_mn = get_tma_aligned_size(
                config.num_padded_sf_pool_tokens,
                static_cast<int>(l2_acts_sf.element_size()));
            return make_tma_2d_desc(
                l2_acts_sf,
                shape_mn, ceil_div(intermediate_hidden, kL2ActsSFGranK),
                config.block_m, 2,
                shape_mn,
                0);
        }
        return make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                config.num_padded_sf_pool_tokens, intermediate_hidden,
                                config.block_m, kL2ActsSFGranK,
                                1, 0);
    }();
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const auto active_sms = get_active_sms_for_sm90_mega_moe(num_sms, num_tokens, num_topk, config);
    const int cta_multiplier = get_env<int>("DG_SM90_MEGA_MOE_CTA_MULTIPLIER", 1);
    DG_HOST_ASSERT(cta_multiplier == 1 or cta_multiplier == 2);
    const int launch_ctas = active_sms * cta_multiplier;
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        std::cout << fmt::format(
            "SM90FP8MegaMoE active_sms={} launch_ctas={} cta_multiplier={} physical_sms={} num_tokens={} num_topk={} block_m={}",
            active_sms, launch_ctas, cta_multiplier, num_sms, num_tokens, num_topk, config.block_m) << std::endl;
    }
    const SM90FP8MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
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
        .stage_profile = stage_profile,
        .launch_args = LaunchArgs(launch_ctas, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_mega_moe", code);
    SM90FP8MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
