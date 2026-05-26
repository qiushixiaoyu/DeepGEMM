#pragma once

#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 x FP4 MegaMoE host runtime
// ----------------------------------------------------------------------------
// Counterpart of `SM90FP8MegaMoERuntime` with these differences:
//   * L1/L2 weights are packed E2M1 (FP4): each storage byte holds 2 nibbles
//     (low nibble = even K, high nibble = odd K). The TMA descriptors
//     therefore use `kPackedFP4` view of the weight tensors so that
//     `aten_dtype_to_tensor_map_dtype` selects the FP4 descriptor type
//     (`CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B` — dense, 2 nibbles/byte).
//     We pass `fp4_unpacked_smem=false` so the smem layout is also dense
//     (1 byte = 2 elements), matching the `b_packed_dtype_t = int8_t` SMEM
//     tile that the kernel expects.
//   * Weight scale factors switch from per-128 K float to per-32 K UE8M0,
//     packed as int32 along K (4 bytes = 4 K-groups = BLOCK_K=128 K-cols
//     for one N-row). They are passed as `const uint32_t*` instead of
//     `const float*`. The kernel reads them via `__ldg` from global, so
//     no TMA descriptor is required.
//   * Activation side is identical to the FP8 path: FP8 e4m3 K-major with
//     128B swizzle, per-128 K float SFA for L1 and per-64 K float SFA for
//     L2 (filled by the L1 epilogue's per-token SwiGLU+quant).
//   * The kernel applies the per-32 SFB on the fly during dequant via
//     `fp4x4_to_scaled_e4m3x4_humming` (Plan C), so the only SF that the
//     promote loop still applies is SFA. There is no `weight_sf` ldg in
//     the math warpgroup beyond the SFB UE8M0 word.
// ============================================================================

class SM90FP8FP4MegaMoERuntime final : public LaunchRuntime<SM90FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        // A/B knob: decode two consecutive K/32 groups in one work item. Keep
        // this as a JIT-time specialization so the default helper does not
        // carry both code paths and pollute small-batch codegen.
        bool use_kg_pair_decode;
        // A/B knob: reduce decoded-tile shared-store instruction count by
        // writing two adjacent u64 outputs with one st.shared.v2.u64.
        bool use_vector_store_decode;
        // A/B knob: build the scaled E4M3 LUT in registers from UE8M0 instead
        // of loading the 64-bit LUT from constant memory.
        bool use_dynamic_lut_decode;
        // A/B knob: bypass the constant LUT for the common UE8M0 scale values
        // seen in small-batch FP4 expert weights.
        bool use_common_lut_fast_path;
        // A/B knob: decode one K/32 group at a time and let the math warpgroup
        // consume each group immediately, overlapping the next group's decode
        // with the current WGMMA batch.
        bool use_kg_pipeline_decode;
        // Plan-C / humming decode: fold SFB exponent into the FP4 → E4M3 LUT.
        // Keep as a JIT-time knob so we can A/B against Plan B (post-MMA
        // promote) without recompiling the host runtime.
        bool fuse_scale_b_humming_decode;
        // UE8M0 SFB uses pure power-of-two promote (exp_offset = e8m0 - 121).
        // Currently always true (DSV4 standard); exposed for parity with
        // future FP32 SFB experiments.
        bool scale_b_pow2_promote;
        // Experimental RS-mode plumbing. When enabled, the JIT instantiates a
        // distinct kernel/config intended to keep decoded FP4 weight fragments
        // in registers instead of writing an E4M3 B tile back to shared memory.
        bool use_rs_mode;
        // A/B knob for overlapping FP4 decode with WGMMA. When false, the math
        // warpgroup only waits on the decode barrier; non-epilogue warps do the
        // decode work and can run ahead through pipeline stages.
        bool math_wg_participates_in_fp4_decode;
        // A/B knob: limit how many warps inside the math warpgroup help decode.
        // This keeps CTA size fixed while testing whether reducing math-side
        // non-tensor-core work improves WGMMA feed.
        int num_math_wg_decode_warps;
        // A/B knob: skip early non-epilogue warps as FP4 decode helpers.
        // 0 keeps the existing 4 assist warps; 2 skips the two TMA loader
        // warps; 4 leaves all decode work to the math warpgroup.
        int first_fp4_decode_assist_warp;
        // A/B knob: split packed-B readiness from the A+B full barrier so the
        // assist warps can start FP4 decode while A/SFA TMA is still in flight.
        bool use_early_b_decode;
        // A/B knob: replace the FP4 decode rendezvous sync with a per-stage
        // mbarrier so assist warps can run ahead after publishing a decoded tile.
        bool use_decode_done_mbarrier;
        // Debug-only instrumentation. This is a JIT-time knob so normal
        // performance builds do not carry clock64 instructions or hot-path
        // branches.
        bool use_clock_profile;
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
        uint64_t* fp4_clock_profile;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

// JIT cache version: sm90_fp8_fp4_mega_moe_decode_assist_warp_v1
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_fp4_mega_moe_impl<
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
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}, {},
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
    args.use_kg_pair_decode ? "true" : "false",
    args.use_vector_store_decode ? "true" : "false",
    args.use_dynamic_lut_decode ? "true" : "false",
    args.use_common_lut_fast_path ? "true" : "false",
    args.use_kg_pipeline_decode ? "true" : "false",
    args.fuse_scale_b_humming_decode ? "true" : "false",
    args.scale_b_pow2_promote        ? "true" : "false",
    args.use_rs_mode                 ? "true" : "false",
    args.math_wg_participates_in_fp4_decode ? "true" : "false",
    args.num_math_wg_decode_warps,
    args.first_fp4_decode_assist_warp,
    args.use_early_b_decode ? "true" : "false",
    args.use_decode_done_mbarrier ? "true" : "false",
    args.use_clock_profile ? "true" : "false");
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
            args.fp4_clock_profile
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
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& fuse_scale_b_humming_decode = true,
    const bool& scale_b_pow2_promote        = true,
    const bool& use_rs_mode                 = false,
    const bool& math_wg_participates_in_fp4_decode = true,
    const int& num_math_wg_decode_warps = 4,
    const int& first_fp4_decode_assist_warp = 0,
    const bool& use_kg_pair_decode = false,
    const bool& use_vector_store_decode = false,
    const bool& use_dynamic_lut_decode = false,
    const bool& use_common_lut_fast_path = false,
    const bool& use_kg_pipeline_decode = false,
    const bool& use_early_b_decode = false,
    const bool& use_decode_done_mbarrier = false,
    const std::optional<torch::Tensor>& fp4_clock_profile = std::nullopt
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Sanity: SFB tensors must be uint32 (UE8M0 packed) and weight tensors
    // must be `kPackedFP4`-viewable storage (1 byte = 2 nibbles).
    DG_HOST_ASSERT(l1_weights_sf.scalar_type() == torch::kInt32 or
                   l1_weights_sf.scalar_type() == torch::kUInt32);
    DG_HOST_ASSERT(l2_weights_sf.scalar_type() == torch::kInt32 or
                   l2_weights_sf.scalar_type() == torch::kUInt32);
    DG_HOST_ASSERT(num_math_wg_decode_warps >= 0 and num_math_wg_decode_warps <= 4);
    DG_HOST_ASSERT(math_wg_participates_in_fp4_decode or num_math_wg_decode_warps == 0);
    DG_HOST_ASSERT(first_fp4_decode_assist_warp >= 0 and first_fp4_decode_assist_warp <= 4);

    // Heuristics
    const auto config = get_mega_moe_config_sm90_fp4(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens,
        use_rs_mode, use_early_b_decode, use_decode_done_mbarrier);

    // Tensormap construction
    constexpr int kGranK         = 128;  // L1 acts SF granularity (per-128 K)
    const int kL2ActsSFGranK = config.block_n == 64 ? 32 : 64;

    // Acts: FP8 e4m3, identical to FP8 path
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.block_m, kGranK,
                                                        1, 0);

    // Packed FP4 weight tile: each byte = 2 nibbles. SM90 loads these as raw
    // bytes and software-decodes them before WGMMA, so the TensorMap must be a
    // UINT8 view with a packed K axis rather than a native FP4 TensorMap.
    const auto l1_weights_bytes = l1_weights.scalar_type() == torch::kByte
        ? l1_weights : l1_weights.view(torch::kByte);
    const auto l2_weights_bytes = l2_weights.scalar_type() == torch::kByte
        ? l2_weights : l2_weights.view(torch::kByte);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights_bytes,
                                                        hidden / 2, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k / 2, config.block_n,
                                                        static_cast<int>(l1_weights_bytes.stride(-2)),
                                                        config.swizzle_weights_mode, /*swizzle_base=*/0,
                                                        /*allow_tf32=*/false);

    // L1 output (post-SwiGLU FP8): N is halved.
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
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights_bytes,
                                                        intermediate_hidden / 2, num_experts_per_rank * hidden,
                                                        config.block_k / 2, config.block_n,
                                                        static_cast<int>(l2_weights_bytes.stride(-2)),
                                                        config.swizzle_weights_mode, /*swizzle_base=*/0,
                                                        /*allow_tf32=*/false);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
    uint64_t* fp4_clock_profile_ptr = nullptr;
    if (fp4_clock_profile.has_value())
        fp4_clock_profile_ptr = reinterpret_cast<uint64_t*>(fp4_clock_profile->data_ptr<int64_t>());

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM90FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .use_kg_pair_decode = use_kg_pair_decode,
        .use_vector_store_decode = use_vector_store_decode,
        .use_dynamic_lut_decode = use_dynamic_lut_decode,
        .use_common_lut_fast_path = use_common_lut_fast_path,
        .use_kg_pipeline_decode = use_kg_pipeline_decode,
        .fuse_scale_b_humming_decode = fuse_scale_b_humming_decode,
        .scale_b_pow2_promote        = scale_b_pow2_promote,
        .use_rs_mode                 = use_rs_mode,
        .math_wg_participates_in_fp4_decode = math_wg_participates_in_fp4_decode,
        .num_math_wg_decode_warps = num_math_wg_decode_warps,
        .first_fp4_decode_assist_warp = first_fp4_decode_assist_warp,
        .use_early_b_decode = use_early_b_decode,
        .use_decode_done_mbarrier = use_decode_done_mbarrier,
        .use_clock_profile = fp4_clock_profile_ptr != nullptr,
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
        .fp4_clock_profile = fp4_clock_profile_ptr,
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8FP4MegaMoERuntime::generate(args);
    const auto runtime_name = use_rs_mode
        ? (args.use_clock_profile ? "sm90_fp8_fp4_mega_moe_rs_clock_profile" : "sm90_fp8_fp4_mega_moe_rs")
        : (args.use_clock_profile ? "sm90_fp8_fp4_mega_moe_clock_profile" : "sm90_fp8_fp4_mega_moe");
    const auto runtime = compiler->build(runtime_name, code);
    SM90FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
