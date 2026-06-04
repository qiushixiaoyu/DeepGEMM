#pragma once

#include <functional>
#include <pybind11/functional.h>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_mega_moe_pre_dispatch.hpp"
#include "../jit_kernels/impls/sm90_fp8_mega_moe.hpp"
#include "../jit_kernels/impls/sm90_fp8_fp4_mega_moe.hpp"
#include "../utils/math.hpp"
#include "../utils/system.hpp"

namespace deep_gemm::mega {

static int get_token_alignment_for_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static bool is_packed_fp4_storage(const torch::Tensor& t) {
    return t.scalar_type() == kPackedFP4 or t.scalar_type() == torch::kByte;
}

static bool is_packed_ue8m0_storage(const torch::Tensor& t) {
    return t.scalar_type() == torch::kInt or t.scalar_type() == torch::kUInt32;
}

static std::tuple<int, int, int> check_grouped_ab_sm90_fp4_mega_moe(const torch::Tensor& ab) {
    const auto [num_groups, mn, packed_k] = get_shape<3>(ab);
    DG_HOST_ASSERT(is_packed_fp4_storage(ab));
    DG_HOST_ASSERT(get_major_type_ab(ab) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(packed_k > 0 and packed_k % 64 == 0);
    return {num_groups, mn, packed_k * 2};
}

static void check_sm90_fp4_sfb_layout(const torch::Tensor& sf,
                                      const int& mn, const int& k,
                                      const int& num_groups) {
    // SM90 FP4 MegaMoE reads SFB directly as:
    //   base + expert * (MN * K/128) + n * (K/128) + k_block
    // so unlike SM100's UTCCP/TMA layout, the required layout is ordinary
    // contiguous [E, MN, K/128] int32/uint32.
    DG_HOST_ASSERT(is_packed_ue8m0_storage(sf));
    DG_HOST_ASSERT(sf.dim() == 3);
    DG_HOST_ASSERT(sf.size(0) == num_groups);
    DG_HOST_ASSERT(sf.size(1) == mn);
    DG_HOST_ASSERT(sf.size(2) == ceil_div(k, 128));
    DG_HOST_ASSERT(sf.is_contiguous());
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);

    // Architecture-dependent SF dtype for the user-facing tensor view:
    //   * SM100: per-32 UE8M0 packed 4-into-int (`torch::kInt`).
    //   * SM90 : per-128 channel float (`torch::kFloat32`).
    // Both use the same number of bytes per token (hidden / 32), so the symmetric
    // buffer layout is shared; only the slice view dtype changes.
    const auto arch_major = device_runtime->get_arch_major();
    const bool is_sm90 = arch_major == 9;
    const auto sf_dtype = is_sm90 ? torch::kFloat32 : torch::kInt;

    // Workspace bytes
    const auto workspace = layout::Workspace(nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);

    // When `DG_USE_FP4_ACTS=1`, the symmetric `x` slot and the
    // L1 token pool both hold packed E2M1 (FP4) instead of dense E4M3 (FP8).
    // The per-token byte footprint halves; the SF slot is unchanged
    // (`hidden/32` UE8M0 bytes — same `gran_k=32` for FP4 and FP8 acts under
    // `kind::mxf8f6f4`). The host-side flag is read from the env so the
    // existing `use_fp8_dispatch` API surface (which is hardcoded `true`
    // throughout) doesn't need to change to opt in.
    const bool host_use_fp4_acts = get_env<int>("DG_USE_FP4_ACTS") != 0;
    const int input_token_bytes = host_use_fp4_acts ? (hidden / 2) : hidden;

    // When `DG_USE_FP8_COMBINE=1`, the combine slot
    // holds FP8 E4M3 (kHidden bytes/token) + a separate combine_sf slot
    // holding UE8M0 SF bytes (kHidden/128 bytes/token, gran_k=128). When off,
    // the combine slot holds BF16 (kHidden*2 bytes/token) and combine_sf is
    // unused (zero-sized).
    const bool host_use_fp8_combine = get_env<int>("DG_USE_FP8_COMBINE") != 0;
    constexpr int kCombineGranK = 128;
    const int combine_token_bytes = host_use_fp8_combine ? hidden : (hidden * 2);
    const int combine_sf_bytes_per_token = host_use_fp8_combine ? (hidden / kCombineGranK) : 0;

    // Layouts
    const auto fp8_token_layout = layout::Data(input_token_bytes);
    const auto combine_token_layout = layout::Data(combine_token_bytes);
    // SF layout: bytes/token may not be a multiple of 16 (e.g. hidden=7168 →
    // 7168/128=56 bytes), so disable TMA alignment requirement (the writes
    // are 1-byte stores via `sym_buffer.map`, not TMA).
    const auto combine_sf_layout = layout::Data(combine_sf_bytes_per_token, false);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
    const auto fp8_sf_layout = layout::Data(hidden / 32);
    const bool host_use_sm90_fp4_rs_mode =
        is_sm90 and get_env<int>("DG_SM90_FP4_RS_MODE") != 0;
    const int default_sm90_fp4_block_n = host_use_sm90_fp4_rs_mode ? 64 : 128;
    const int sm90_fp4_block_n =
        get_env<int>("DG_SM90_FP4_BLOCK_N", default_sm90_fp4_block_n);
    if (is_sm90)
        DG_HOST_ASSERT(sm90_fp4_block_n == 64 or sm90_fp4_block_n == 128);
    const int sm90_l2_act_sf_gran_k = sm90_fp4_block_n == 64 ? 32 : 64;

    // L2 acts SF granularity differs by arch:
    //   * SM100 packs 4 UE8M0 bytes per int along K, so each token uses
    //     `intermediate_hidden / 32` bytes (per-32 K).
    //   * SM90 stores per-64 K floats so that each L1 epilogue block (which
    //     produces 64 post-SwiGLU columns) can write its own SF independently
    //     without cross-CTA amax synchronisation. SM90 FP4 BLOCK_N=64 mode
    //     produces 32 post-SwiGLU columns per L1 block, so its
    //     L2 SF view also switches to per-32.
    const int fp8_intermediate_sf_bytes_per_token =
        is_sm90 ? (intermediate_hidden * static_cast<int>(sizeof(float)) / sm90_l2_act_sf_gran_k)
                : (intermediate_hidden / 32);
    const auto fp8_intermediate_sf_layout = layout::Data(fp8_intermediate_sf_bytes_per_token);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Input buffers
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_tokens_per_rank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, num_max_tokens_per_rank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    // Buffer configs
    const auto num_max_pool_tokens = static_cast<int>(workspace.num_max_pool_tokens);
    int num_max_padded_sf_pool_tokens = 0;
    for (int block_m: layout::kCandidateBlockM) {
        num_max_padded_sf_pool_tokens = std::max(
            num_max_padded_sf_pool_tokens,
            layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m)
        );
    }

    // L1 input buffer
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_pool_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_max_pool_tokens,
        l1_sf_buffer.get_end_ptr());

    // L2 input buffer
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, num_max_pool_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l2_token_buffer.get_end_ptr());

    // Combine input buffer: BF16 tokens (default) OR FP8 (when host_use_fp8_combine)
    // for cross-rank combine.
    const auto combine_token_buffer = layout::Buffer(
        combine_token_layout, num_topk, num_max_tokens_per_rank,
        l2_sf_buffer.get_end_ptr());
    // Combine SF buffer: only sized when host_use_fp8_combine (otherwise zero).
    // Layout matches combine_token_buffer's [num_topk][num_max_tokens_per_rank]
    // outer shape, with kHidden/128 SF bytes per token.
    const auto combine_sf_buffer = layout::Buffer(
        combine_sf_layout, num_topk, num_max_tokens_per_rank,
        combine_token_buffer.get_end_ptr());

    // Check SF buffer requirements
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    // SM100 packs 4 UE8M0 bytes per int along K, so the padded SF token count
    // must be divisible by 4. SM90 stores per-128 floats and has no such constraint.
    if (not is_sm90)
        DG_HOST_ASSERT(num_max_padded_sf_pool_tokens % 4 == 0);

    // Slice function: creates `(x, x_sf, topk_weights, topk_idx, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf)` tensor views from the raw buffer
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    //        Dtype is per-arch (see `sf_dtype` above): float on SM90, int (packed UE8M0) on SM100.
    // Under `host_use_fp4_acts`, the `x` and `l1_acts` views
    // expose packed E2M1 (`kPackedFP4` = `torch::kInt8`, 2 elements/byte) of
    // shape `[..., hidden / 2]`. Underlying buffer bytes are the same as the
    // sized `fp8_token_layout` slot, just half the row width.
    const auto x_dtype = host_use_fp4_acts ? kPackedFP4 : torch::kFloat8_e4m3fn;
    const int x_inner_cols = host_use_fp4_acts ? (hidden / 2) : hidden;
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, x_inner_cols},
            torch::TensorOptions().dtype(x_dtype).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(sf_dtype).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_token_buffer.base)),
            {num_max_pool_tokens, x_inner_cols},
            torch::TensorOptions().dtype(x_dtype).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, hidden / 128},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(sf_dtype).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_max_pool_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, is_sm90 ? intermediate_hidden / sm90_l2_act_sf_gran_k : intermediate_hidden / 128},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(sf_dtype).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {reinterpret_cast<int64_t>(combine_sf_buffer.get_end_ptr()), slice_input_buffers};
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        arch_major == 9 ? check_grouped_ab_sm90_fp4_mega_moe(l1_weights)
                        : check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        arch_major == 9 ? check_grouped_ab_sm90_fp4_mega_moe(l2_weights)
                        : check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check weight SF layout for UE8M0 packing.
    constexpr int kGranMN = 1, kGranK = 32;
    if (arch_major == 9) {
        check_sm90_fp4_sfb_layout(l1_weights_sf, intermediate_hidden * 2, hidden,
                                  num_experts_per_rank);
        check_sm90_fp4_sfb_layout(l2_weights_sf, hidden, intermediate_hidden,
                                  num_experts_per_rank);
    } else {
        check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                        num_experts_per_rank, true, false, torch::kInt);
        check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                        num_experts_per_rank, true, false, torch::kInt);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }
    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    // Pick up FP4-acts flag from `DG_USE_FP4_ACTS` env var.
    // Default off — preserves byte-identical FP8-acts behavior. Setting
    // `DG_USE_FP4_ACTS=1` flips L1's epilogue quant to E2M1 + UE8M0 SF.
    const bool use_fp4_acts = get_env<int>("DG_USE_FP4_ACTS") != 0;
    // When also `DG_USE_MXF4_KIND=1`, the L1 and L2 mainloops
    // run `tcgen05.mma.kind::mxf4.block_scale.block32` instead of
    // `kind::mxf8f6f4` — K=64 dense per call (vs K=32 with-padding), dense
    // FP4 smem (`_ALIGN8B`, half the byte footprint), scale_vec::2X SF
    // protocol with HALF-WORD address bits. Only honored when
    // `DG_USE_FP4_ACTS=1` (kind::mxf4 is FP4-only).
    const bool use_mxf4_kind = use_fp4_acts and get_env<int>("DG_USE_MXF4_KIND") != 0;
    // When `DG_USE_FP8_COMBINE=1`, the L2 epilogue
    // ships FP8 E4M3 + per-(token, N=128) UE8M0 SF over NVLink instead of
    // BF16. The combine reduce dequantizes on the fly. NVLink bytes/token
    // halve (from kHidden*2 → kHidden + kHidden/128). Independent of the
    // FP4-acts / MXF4-kind flags above (those control the dispatch a2a +
    // mainloops; this controls the combine a2a only).
    const bool use_fp8_combine = get_env<int>("DG_USE_FP8_COMBINE") != 0;

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(y,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               cumulative_local_expert_recv_stats,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation_clamp, fast_math,
                               use_fp4_acts, use_mxf4_kind, use_fp8_combine);
    } else if (arch_major == 9) {
        // SM90 (Hopper): FP8 acts × FP4 weights. Default is decode-to-SMEM
        // SS-mode WGMMA; `DG_SM90_FP4_RS_MODE=1` routes through the
        // RS-mode JIT/config plumbing. The activation a2a stays FP8 (no
        // `use_fp4_acts` on SM90 yet — there is no native FP4 WGMMA on Hopper,
        // and FP4 acts would still need a per-tile dequant pass that bottlenecks
        // on smem write bandwidth). The combine path likewise stays BF16.
        DG_HOST_ASSERT(not use_fp4_acts and not use_mxf4_kind and not use_fp8_combine);
        const bool fuse_scale_b_humming_decode =
            get_env<int>("DG_SM90_FP4_FUSE_SCALE_B_HUMMING_DECODE") != 0;
        const bool use_rs_mode = get_env<int>("DG_SM90_FP4_RS_MODE") != 0;
        const float expected_tokens_per_expert =
            static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
        // Decode lookahead uses a one-way decode_done mbarrier, keeps the math
        // warpgroup as a pure WGMMA consumer, and excludes loader warps from
        // decode-assist so they stay on the TMA/SFB producer path.
        const bool fp4_decode_lookahead_band =
            !use_rs_mode and
            expected_tokens_per_expert >= 3.0f and expected_tokens_per_expert <= 6.0f;
        // At higher one-warpgroup occupancy, lookahead is useful only while
        // assist warps can decode a full stage before WGMMA consumes it.
        const bool fp4_bigband_lookahead_band =
            !use_rs_mode and
            expected_tokens_per_expert >= 12.0f and expected_tokens_per_expert <= 24.0f;
        // Low-occupancy decode offload uses assist warps for FP4 decode while
        // keeping loader warps on the producer path.
        const bool fp4_b4_skip_decode_band =
            !use_rs_mode and
            expected_tokens_per_expert >= 0.5f and expected_tokens_per_expert < 1.0f;
        // The 2-WG split-MN path is register-pressured, so decode is fully
        // offloaded from the math warpgroup once that path is selected.
        const bool fp4_2wg_decode_offload_band =
            !use_rs_mode and expected_tokens_per_expert >= 64.0f;
        const bool default_math_wg_decode =
            // Turn math-WG decode off only in occupancy bands where assist
            // warps can hide FP4 decode behind WGMMA or where split-MN
            // accumulator pressure benefits from full decode offload.
            !use_rs_mode and
            ((expected_tokens_per_expert > 0.0f and expected_tokens_per_expert < 0.375f) or
             (expected_tokens_per_expert >= 1.0f and expected_tokens_per_expert < 2.0f) or
             fp4_decode_lookahead_band or fp4_b4_skip_decode_band or
             fp4_bigband_lookahead_band or fp4_2wg_decode_offload_band);
        const bool math_wg_participates_in_fp4_decode =
            get_env<int>("DG_SM90_FP4_MATH_WG_DECODE",
                         default_math_wg_decode ? 0 : 1) != 0;
        const int num_math_wg_decode_warps =
            get_env<int>("DG_SM90_FP4_MATH_WG_DECODE_WARPS",
                         math_wg_participates_in_fp4_decode ? 4 : 0);
        const bool default_skip_loader_decode_assist =
            // The two loader-side non-epilogue warps are on the TMA/SFB
            // producer path. Skip them in bands where decode can hide behind
            // WGMMA with the remaining assist warps.
            !use_rs_mode and
            ((expected_tokens_per_expert > 0.0f and expected_tokens_per_expert < 0.375f) or
             (expected_tokens_per_expert >= 1.5f and expected_tokens_per_expert < 3.0f) or
             fp4_decode_lookahead_band or fp4_b4_skip_decode_band or
             fp4_bigband_lookahead_band);
        const int first_fp4_decode_assist_warp =
            get_env<int>("DG_SM90_FP4_FIRST_DECODE_ASSIST_WARP",
                         default_skip_loader_decode_assist ? 2 : 0);
        const bool use_kg_pair_decode =
            get_env<int>("DG_SM90_FP4_KG_PAIR_DECODE", 0) != 0;
        const bool use_vector_store_decode =
            get_env<int>("DG_SM90_FP4_VECTOR_STORE_DECODE", 1) != 0;
        const bool use_skip_zero_sfb_decode =
            get_env<int>("DG_SM90_FP4_SKIP_ZERO_DECODE", 0) != 0;
        const bool use_dynamic_lut_decode =
            get_env<int>("DG_SM90_FP4_DYNAMIC_LUT_DECODE", 0) != 0;
        const bool use_common_lut_fast_path =
            get_env<int>("DG_SM90_FP4_COMMON_LUT_FASTPATH", 0) != 0;
        const bool use_kg_pipeline_decode =
            get_env<int>("DG_SM90_FP4_KG_PIPELINE_DECODE", 0) != 0;
        const bool default_rs_group_k_promote =
            // Batch the four Linear1 RS K/32 WGMMAs into one commit/wait for
            // small batches. This is a measured win for batch/rank <= 16.
            use_rs_mode and num_tokens <= 16;
        const bool use_rs_group_k_promote =
            get_env<int>("DG_SM90_FP4_RS_GROUP_K_PROMOTE",
                         default_rs_group_k_promote ? 1 : 0) != 0;
        const bool use_rs_l2_group_k2_promote =
            get_env<int>("DG_SM90_FP4_RS_L2_GROUP_K2_PROMOTE", 0) != 0;
        const bool default_rs_transpose_vec_load =
            // Tiny-batch RS mode benefits from vectorized transpose readback.
            use_rs_mode and num_tokens <= 2;
        const bool use_rs_transpose_vec_load =
            get_env<int>("DG_SM90_FP4_RS_TRANSPOSE_VEC_LOAD",
                         default_rs_transpose_vec_load ? 1 : 0) != 0;
        const bool default_rs_guard_transpose_valid =
            // Extra valid-row predicates are left off by default.
            false;
        const bool use_rs_guard_transpose_valid =
            get_env<int>("DG_SM90_FP4_RS_GUARD_TRANSPOSE_VALID",
                         default_rs_guard_transpose_valid ? 1 : 0) != 0;
        const bool default_rs_sfa_vec_load =
            // Adjacent SFA entries are naturally 8-byte aligned in the RS
            // promote loop. Vectorized shared loads are a measured win for
            // small-batch RS mode and remain neutral in register/spill usage.
            use_rs_mode and num_tokens <= 16;
        const bool use_rs_sfa_vec_load =
            get_env<int>("DG_SM90_FP4_RS_SFA_VEC_LOAD",
                         default_rs_sfa_vec_load ? 1 : 0) != 0;
        const bool use_rs_sfa_bcast_load =
            get_env<int>("DG_SM90_FP4_RS_SFA_BCAST_LOAD", 0) != 0;
        const bool default_rs_sfb_word_reuse =
            // One packed UE8M0 word feeds four RS K/32 slices. Reusing it
            // cuts repeated SFB global loads in small-batch RS mode.
            use_rs_mode and num_tokens <= 16;
        const bool use_rs_sfb_word_reuse =
            get_env<int>("DG_SM90_FP4_RS_SFB_WORD_REUSE",
                         default_rs_sfb_word_reuse ? 1 : 0) != 0;
        const bool use_rs_sfb_bcast_load =
            get_env<int>("DG_SM90_FP4_RS_SFB_BCAST_LOAD", 0) != 0;
        const bool default_rs_stage_sfb =
            // Staging packed SFB in shared memory is a stable win once the
            // small batch has enough expert work (batch/rank 4..16). Batch 1
            // and 2 are noisy or regressive, so keep the env knob for those.
            use_rs_mode and num_tokens >= 4 and num_tokens <= 16;
        const bool use_rs_stage_sfb =
            get_env<int>("DG_SM90_FP4_RS_STAGE_SFB",
                         default_rs_stage_sfb ? 1 : 0) != 0;
        const bool default_rs_decode_pair_shfl =
            // Pair-shuffle currently increases register pressure in RS mode.
            false;
        const bool use_rs_decode_pair_shfl =
            get_env<int>("DG_SM90_FP4_RS_DECODE_PAIR_SHFL",
                         default_rs_decode_pair_shfl ? 1 : 0) != 0;
        const bool use_rs_direct_l2_scatter =
            get_env<int>("DG_SM90_FP4_RS_DIRECT_L2_SCATTER", 0) != 0;
        const bool default_ss_early_b_decode =
            // Early packed-B decode helps when there is enough FP4 work to
            // overlap with A/SFA TMA, but stays off for very small batches.
            !use_rs_mode and
            ((expected_tokens_per_expert >= 1.5f and expected_tokens_per_expert <= 3.0f) or
             (expected_tokens_per_expert >= 6.0f and expected_tokens_per_expert <= 24.0f));
        const bool default_early_b_decode =
            // RS mode uses early packed-B decode for small batches where it can
            // overlap with activation readiness.
            (use_rs_mode and num_tokens > 0 and num_tokens <= 16) or
            default_ss_early_b_decode;
        const bool use_early_b_decode =
            get_env<int>("DG_SM90_FP4_EARLY_B_DECODE", default_early_b_decode ? 1 : 0) != 0;
        const bool default_decode_done_mbarrier =
            // Enable the one-way decode mbarrier only in lookahead bands where
            // assist warps can publish decoded stages ahead of the math WG.
            fp4_decode_lookahead_band or fp4_bigband_lookahead_band;
        const bool use_decode_done_mbarrier =
            get_env<int>("DG_SM90_FP4_DECODE_MBARRIER",
                         default_decode_done_mbarrier ? 1 : 0) != 0;
        const bool default_l2_arrival_counter =
            // Use a lightweight arrival counter only in the small band where it
            // pairs with the skipped post-scatter sync.
            !use_rs_mode and
            expected_tokens_per_expert >= 0.375f and expected_tokens_per_expert < 0.75f;
        const bool use_l2_arrival_counter =
            get_env<int>("DG_SM90_FP4_L2_ARRIVAL_COUNTER",
                         default_l2_arrival_counter ? 1 : 0) != 0;
        const bool default_skip_l2_epilogue_sync =
            // This is only enabled for the small-token band where the following
            // grid/NVLink synchronization already orders the L2 scatter.
            !use_rs_mode and
            expected_tokens_per_expert >= 0.375f and expected_tokens_per_expert < 0.75f;
        const bool skip_l2_epilogue_sync =
            get_env<int>("DG_SM90_FP4_SKIP_L2_EPILOGUE_SYNC",
                         default_skip_l2_epilogue_sync ? 1 : 0) != 0;
        // SS N-split replaces each N=128 WGMMA with two N=64 WGMMAs, reducing
        // per-K-block accumulator pressure on the 2-WG split-M path. The
        // kernel-side `kSSNSplitActive` gate keeps single-WG shapes inert.
        const bool default_use_ss_nsplit =
            !use_rs_mode and expected_tokens_per_expert >= 64.0f;
        const bool use_ss_nsplit =
            get_env<int>("DG_SM90_FP4_SS_NSPLIT",
                         default_use_ss_nsplit ? 1 : 0) != 0;
        sm90_fp8_fp4_mega_moe(y,
                              l1_acts, l1_acts_sf,
                              l2_acts, l2_acts_sf,
                              l1_weights, l2_weights,
                              l1_weights_sf, l2_weights_sf,
                              cumulative_local_expert_recv_stats,
                              sym_buffer_ptrs,
                              rank_idx, num_max_tokens_per_rank,
                              num_experts_per_rank,
                              num_tokens, num_topk,
                              hidden, intermediate_hidden,
                              activation_clamp, fast_math,
                              fuse_scale_b_humming_decode,
                              /*scale_b_pow2_promote=*/true,
                              use_rs_mode,
                              math_wg_participates_in_fp4_decode,
                              num_math_wg_decode_warps,
                              first_fp4_decode_assist_warp,
                              use_kg_pair_decode,
                              use_vector_store_decode,
                              use_skip_zero_sfb_decode,
                              use_dynamic_lut_decode,
                              use_common_lut_fast_path,
                              use_kg_pipeline_decode,
                              use_rs_group_k_promote,
                              use_rs_l2_group_k2_promote,
                              use_rs_transpose_vec_load,
                              use_rs_guard_transpose_valid,
                              use_rs_sfa_vec_load,
                              use_rs_sfa_bcast_load,
                              use_rs_sfb_word_reuse,
                              use_rs_sfb_bcast_load,
                              use_rs_stage_sfb,
                              use_rs_decode_pair_shfl,
                              use_rs_direct_l2_scatter,
                              use_early_b_decode,
                              use_decode_done_mbarrier,
                              use_l2_arrival_counter,
                              skip_l2_epilogue_sync,
                              use_ss_nsplit);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

}

// SM90 (Hopper) FP8 MegaMoE entry point.
//
// Mirrors `fp8_fp4_mega_moe` but expects FP8 (e4m3) weights with per-128 channel
// float scale factors. Top-level routing (which entry to call) is the caller's
// responsibility (see `deep_gemm/mega/__init__.py`).
static void fp8_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Architecture check
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 9);

    // Config checks: SM90 uses block (128, 128) float SF for weights.
    // Activation SF is float as well: L1 input is per-128 K and the L2
    // intermediate activation produced by the fused L1 epilogue is per-64 K.
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 128 and rn == 128 and rk == 128);
    DG_HOST_ASSERT(activation == "swiglu");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks: SM90 weights must be FP8 e4m3, K-major
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kFloat8_e4m3fn);
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Shape constraints required by the SM90 kernel:
    //   * Hidden dims must be multiples of 128 (per-128 SF + scheduler integer-tiling).
    //   * `l2_arrival_mask` is uint64, with one bit per L1-output N-block of size 64 in the
    //     intermediate dim, so `kNumL1BlockNs = intermediate_hidden / 64` must be ≤ 64.
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(intermediate_hidden / 64 <= 64);

    // Check weight SF layout (block (128, 128) float, MN-major; not TMA-loaded
    // so no TMA-stride alignment is required, but we do require contiguity in
    // the K-direction within each expert).
    constexpr int kGranMN = 128, kGranK = 128;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    sm90_fp8_mega_moe(y,
                     l1_acts, l1_acts_sf,
                     l2_acts, l2_acts_sf,
                     l1_weights, l2_weights,
                     l1_weights_sf, l2_weights_sf,
                     cumulative_local_expert_recv_stats,
                     sym_buffer_ptrs,
                     rank_idx, num_max_tokens_per_rank,
                     num_experts_per_rank,
                     num_tokens, num_topk,
                     hidden, intermediate_hidden,
                     activation_clamp, fast_math);

}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
    m.def("fp8_fp4_mega_moe",
          [](const torch::Tensor& y,
             const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
             const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const torch::Tensor& sym_buffer,
             const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
             const int& num_max_tokens_per_rank, const int& num_experts, const int& num_topk,
             const std::tuple<int, int, int>& recipe,
             const std::string& activation,
             const std::optional<float>& activation_clamp_opt,
             const bool& fast_math) {
              fp8_fp4_mega_moe(
                  y,
                  std::make_tuple(l1_weights, l1_weights_sf),
                  std::make_tuple(l2_weights, l2_weights_sf),
                  cumulative_local_expert_recv_stats,
                  sym_buffer, sym_buffer_ptrs, rank_idx,
                  num_max_tokens_per_rank, num_experts, num_topk,
                  recipe, activation, activation_clamp_opt, fast_math);
          });
    m.def("mega_moe_pre_dispatch", &mega_moe_pre_dispatch,
          pybind11::arg("x"),
          pybind11::arg("topk_idx"),
          pybind11::arg("topk_weights"),
          pybind11::arg("buf_x"),
          pybind11::arg("buf_x_sf"),
          pybind11::arg("buf_topk_idx"),
          pybind11::arg("buf_topk_weights"),
          pybind11::arg("num_tokens"),
          pybind11::arg("group_size") = 32,
          pybind11::arg("use_fp4_acts") = false);
    m.def("fp8_mega_moe",
          [](const torch::Tensor& y,
             const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
             const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const torch::Tensor& sym_buffer,
             const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
             const int& num_max_tokens_per_rank, const int& num_experts, const int& num_topk,
             const std::tuple<int, int, int>& recipe,
             const std::string& activation,
             const std::optional<float>& activation_clamp_opt,
             const bool& fast_math) {
              fp8_mega_moe(
                  y,
                  std::make_tuple(l1_weights, l1_weights_sf),
                  std::make_tuple(l2_weights, l2_weights_sf),
                  cumulative_local_expert_recv_stats,
                  sym_buffer, sym_buffer_ptrs, rank_idx,
                  num_max_tokens_per_rank, num_experts, num_topk,
                  recipe, activation, activation_clamp_opt, fast_math);
          });
#endif
}

} // namespace deep_gemm::mega
