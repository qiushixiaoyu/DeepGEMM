#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "mega.hpp"
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm90_fp8_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm90_fp8_mega_moe.hpp"
#include "../jit_kernels/impls/sm90_mega_moe_pre_dispatch.hpp"
#include "../utils/layout.hpp"
#include "../utils/system.hpp"
#include "../utils/sm90_mega_moe_rdma.hpp"

namespace deep_gemm::mega {

static int get_token_alignment_for_sm90_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static void mega_moe_pre_dispatch_sm90(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& buf_x,
    const torch::Tensor& buf_x_sf,
    const torch::Tensor& buf_topk_idx,
    const torch::Tensor& buf_topk_weights,
    const int& num_tokens,
    const int& group_size,
    const float& routed_scaling_factor) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    sm90_mega_moe_pre_dispatch(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens, group_size, routed_scaling_factor);
}

static bool is_packed_fp4_storage_sm90(const torch::Tensor& t) {
    return t.scalar_type() == kPackedFP4 or t.scalar_type() == torch::kByte;
}

static std::tuple<int, int, int> check_grouped_ab_sm90_fp4_mega_moe(const torch::Tensor& ab) {
    const auto [num_groups, mn, packed_k] = get_shape<3>(ab);
    DG_HOST_ASSERT(is_packed_fp4_storage_sm90(ab));
    DG_HOST_ASSERT(get_major_type_ab(ab) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(packed_k > 0 and packed_k % 64 == 0);
    return {num_groups, mn, packed_k * 2};
}

static void check_sm90_fp4_sfb_layout(const torch::Tensor& sf,
                                      const int& mn, const int& k,
                                      const int& num_groups) {
    DG_HOST_ASSERT(sf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(sf.dim() == 3);
    DG_HOST_ASSERT(sf.size(0) == num_groups);
    DG_HOST_ASSERT(sf.size(1) == mn);
    DG_HOST_ASSERT(sf.size(2) == ceil_div(k, 128));
    const bool n_contiguous = sf.stride(0) == mn * ceil_div(k, 128) and
                              sf.stride(1) == 1 and sf.stride(2) == mn;
    DG_HOST_ASSERT(sf.is_contiguous() or n_contiguous);
}

struct FP4SM90APIDefaults {
    bool wide_load_decode;
    bool early_b_decode;
    bool decode_done_mbarrier;
    bool ss_nsplit;
    bool swap_ab;
    bool swap_ab_fast_amax;
};

static FP4SM90APIDefaults get_fp4_sm90_api_defaults(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const float rows = expected_tokens_per_expert;
    const bool has_rows = rows > 0.0f;
    // Shape bands exclude kernel tile/thread constraints; JIT heuristics add
    // those as kernel bands.  They remain broad workload classes rather than
    // exact model identifiers.
    const bool fp4_flash_shape = intermediate_hidden <= 2048;
    const bool fp4_pro_shape = intermediate_hidden >= 3072;
    const bool fp4_middle_shape = !fp4_flash_shape and !fp4_pro_shape;

    const bool below_m128_crossover =
        has_rows and rows < kFP4SM90M128CrossoverRows;
    const bool m128_or_larger =
        rows >= kFP4SM90M128CrossoverRows;

    // All supported Flash/Pro-like M64 shapes use swapAB until the shared
    // crossover. Reuse this topology decision for packed-weight load width
    // rather than maintaining separate sparse-density exceptions.
    const bool default_swap_ab =
        (fp4_flash_shape or fp4_pro_shape) and below_m128_crossover;

    // Wide loads are validated for this M64 swapAB topology. Do not expand
    // them into M128 or the middle-shape non-swap kernels.
    const bool default_wide_load_decode = default_swap_ab;

    // Early-B remains useful only in the middle-shape 1.5--3 band, the
    // Flash-like 2--3 band, and dense M128 shapes other than the first
    // Pro-like M128 region.  Expressing those regions directly removes the
    // previous chain of overlapping exclusion predicates.
    const bool default_ss_early_b_decode =
        (fp4_middle_shape and rows >= 1.5f and rows <= 3.0f) or
        (fp4_flash_shape and rows >= 2.0f and rows < 3.0f) or
        (m128_or_larger and
         !(fp4_pro_shape and rows < 128.0f));

    // Dedicated decode assistants are independent producers. A one-way
    // stage mbarrier lets them advance without a rendezvous with math
    // consumers; no model or density window is needed for this topology.
    const bool default_decode_done_mbarrier =
        has_rows;

    // Warp-cooperative amax avoids the FP32 full-tile staging and serial
    // per-token row scan in the swapAB L1 epilogue.  The 16--32 row window
    // amortizes its two CTA reductions for both light and streaming-weight
    // Pro shapes; sparser work did not show a repeatable gain.
    const bool default_swap_ab_fast_amax =
        fp4_pro_shape and
        rows >= 16.0f and rows <= 32.0f;
    return {
        default_wide_load_decode,
        default_ss_early_b_decode,
        default_decode_done_mbarrier,
        m128_or_larger,
        default_swap_ab,
        default_swap_ab_fast_amax
    };
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_sm90_mega_moe_impl(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    check_sm90_mega_moe_rdma_topology(num_ranks);
    // This custom IBGDA protocol uses ordered WRITEs, not atomic signals.
    // Require the setting at process startup, before any NVSHMEM allocator
    // initializes/registers memory; changing it after initialization is unsafe.
    if (get_env<std::string>("NVSHMEM_IB_ENABLE_RELAXED_ORDERING", "") != "0")
        DG_HOST_UNREACHABLE(
            "SM90 MegaMoE RDMA requires NVSHMEM_IB_ENABLE_RELAXED_ORDERING=0 "
            "at process startup, before NVSHMEM initialization; restart all ranks.");
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(use_fp8_dispatch);
    DG_HOST_ASSERT(activation == "swiglu");

    const auto workspace = layout::SM90Workspace(
        nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);

    const auto fp8_token_layout = layout::Data(hidden);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
    const auto fp8_sf_layout = layout::Data(hidden / 32);
    const int sm90_l2_act_sf_gran_k = 64;
    const auto fp8_intermediate_sf_layout =
        layout::Data(intermediate_hidden * static_cast<int>(sizeof(float)) / sm90_l2_act_sf_gran_k);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);
    // Inter-node dispatch pulls SF and routing weight into one cache-line-isolated
    // row per pool token.  Keeping this storage separate from the combine buffer
    // prevents RNIC writes from aliasing cache lines later consumed by combine.
    const auto dispatch_staging_layout = layout::Data(
        math::align(static_cast<uint32_t>(hidden / 32 + sizeof(float)), 128u));

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

    const auto num_max_pool_tokens = static_cast<int>(workspace.num_max_pool_tokens);
    const auto num_compute_ring_tokens = static_cast<int>(
        layout::get_num_sm90_compute_ring_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk,
            num_experts / num_ranks));
    // Below the transition threshold the smooth-capacity policy intentionally
    // returns the full pool.  Treat that as "ring disabled".
    const bool effective_l1_ring = num_compute_ring_tokens < num_max_pool_tokens;
    const bool effective_l2_ring = effective_l1_ring;
    const auto num_l1_ring_tokens = effective_l1_ring ?
        num_compute_ring_tokens : num_max_pool_tokens;
    const auto num_l2_ring_tokens = effective_l2_ring ?
        num_compute_ring_tokens : num_max_pool_tokens;
    const auto num_combine_staging_tokens =
        static_cast<int>(layout::get_num_sm90_combine_ring_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk,
            num_experts / num_ranks));
    // Both SM90 FP8 and FP4 MegaMoE recipes use BLOCK_M={64,128}.  Sizing
    // against the common BLOCK_M=8 candidate inflated each SF pool to 16x
    // payload rows; two SF rows per payload row is the exact worst case.
    const auto num_max_padded_sf_pool_tokens = static_cast<int>(
        layout::get_num_sm90_compute_sf_ring_tokens(
            num_max_pool_tokens));
    const auto num_l1_padded_sf_ring_tokens = effective_l1_ring ?
        static_cast<int>(layout::get_num_sm90_compute_sf_ring_tokens(
            num_l1_ring_tokens)) : num_max_padded_sf_pool_tokens;
    // A compact L2 ring reserves one 128-row scratch/guard block after the
    // reusable payload region.  The JIT kernel slices and indexes this block
    // explicitly, so host sizing must include it for both token and FP8 SF
    // storage.  It is a constant-size addition, not a return to full-pool SF
    // sizing.
    constexpr int kSM90L2RingScratchTokens = 128;
    const auto num_l2_storage_tokens = num_l2_ring_tokens +
        (effective_l2_ring ? kSM90L2RingScratchTokens : 0);
    const auto num_l2_padded_sf_ring_tokens = effective_l2_ring ?
        static_cast<int>(layout::get_num_sm90_compute_sf_ring_tokens(
            num_l2_ring_tokens)) + kSM90L2RingScratchTokens :
        num_max_padded_sf_pool_tokens;

    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_l1_ring_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_l1_padded_sf_ring_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_l1_ring_tokens,
        l1_sf_buffer.get_end_ptr());

    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, num_l2_storage_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, num_l2_padded_sf_ring_tokens,
        l2_token_buffer.get_end_ptr());

    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        l2_sf_buffer.get_end_ptr());
    const auto dispatch_staging_buffer = layout::Buffer(
        dispatch_staging_layout, 1, num_max_pool_tokens,
        combine_token_buffer.get_end_ptr());

    void* symm_buffer_end = dispatch_staging_buffer.get_end_ptr();
    {
        const auto combine_full_row_arrival_buffer = layout::Buffer(
            layout::Data(sizeof(uint32_t), false), 1,
            workspace.num_max_pool_blocks, symm_buffer_end);
        // Phase-profile builds record the local ready publication timestamp so
        // the publisher can report ready-to-observe latency.  Reserve it in
        // every build to keep the public size calculation JIT-independent.
        const auto combine_full_row_ready_timestamp_buffer = layout::Buffer(
            layout::Data(sizeof(uint64_t), false), 1,
            workspace.num_max_pool_blocks,
            combine_full_row_arrival_buffer.get_end_ptr());
        // Preserve legacy publisher padding so existing symmetric-buffer
        // offsets stay unchanged. The retired row mask is never accessed.
        constexpr int kPublishReservedWordsPerRank = 4;
        const auto combine_publish_reserved_buffer = layout::Buffer(
            layout::Data(
                num_ranks * kPublishReservedWordsPerRank * sizeof(uint32_t),
                false),
            1, workspace.num_max_pool_blocks,
            combine_full_row_ready_timestamp_buffer.get_end_ptr());
        const auto combine_full_row_staging_base = reinterpret_cast<void*>(math::align(
            reinterpret_cast<uint64_t>(combine_publish_reserved_buffer.get_end_ptr()),
            static_cast<uint64_t>(128)));
        const auto combine_full_row_staging_buffer = layout::Buffer(
            bf16_token_layout, 1, num_combine_staging_tokens,
            combine_full_row_staging_base);
        symm_buffer_end = combine_full_row_staging_buffer.get_end_ptr();
    }
    const auto phase_profile_buffer = layout::Buffer(
        layout::Data(layout::kSM90MegaMoEProfileSlots * sizeof(uint64_t), false),
        1, layout::kSM90MegaMoEProfileMaxSMs, symm_buffer_end);
    symm_buffer_end = phase_profile_buffer.get_end_ptr();

    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);

    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
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
            {num_l1_ring_tokens, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_l1_padded_sf_ring_tokens, hidden / 128},
            {1, num_l1_padded_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_l2_storage_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_l2_padded_sf_ring_tokens, intermediate_hidden / sm90_l2_act_sf_gran_k},
            {1, num_l2_padded_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {reinterpret_cast<int64_t>(symm_buffer_end), slice_input_buffers};
}

// RDMA FP8 and FP4 share the full-row combine/ring policy.
// The shared layout reserves storage for both dense-V3 and packed metadata.
static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_sm90_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    return get_symm_buffer_size_for_sm90_mega_moe_impl(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, use_fp8_dispatch, activation);
}

static void fp8_mega_moe_impl(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& requested_num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    check_sm90_mega_moe_rdma_topology(num_ranks);
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 9);

    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(requested_num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(
        requested_num_max_tokens_per_rank <= num_max_tokens_per_rank);
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 128 and rn == 128 and rk == 128);
    DG_HOST_ASSERT(activation == "swiglu");

    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

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
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(intermediate_hidden / 64 <= 64);

    constexpr int kGranMN = 128, kGranK = 128;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto num_rc_per_pe = get_env<int>("NVSHMEM_IBGDA_NUM_RC_PER_PE", 0);
    // Expert QPs occupy [0, E); gateway metadata uses the spare QP E.
    DG_HOST_ASSERT(num_rc_per_pe >= num_experts_per_rank + 1);
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_sm90_mega_moe_impl(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);
    // Inputs are populated by the Python pre-dispatch helper.  Keep them in
    // the common slice for API/layout compatibility, but the fused kernel only
    // needs the compute pools here.
    (void)x;
    (void)x_sf;
    (void)topk_idx;
    (void)topk_weights;

    sm90_fp8_mega_moe(y,
                     l1_acts, l1_acts_sf,
                     l2_acts, l2_acts_sf,
                     l1_weights, l2_weights,
                     l1_weights_sf, l2_weights_sf,
                     cumulative_local_expert_recv_stats,
                     sym_buffer_ptrs,
                     rank_idx, num_max_tokens_per_rank,
                     requested_num_max_tokens_per_rank,
                     num_experts_per_rank,
                     num_tokens, num_topk,
                     hidden, intermediate_hidden,
                     activation_clamp, fast_math);

    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void fp8_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& requested_num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    fp8_mega_moe_impl(
        y, l1_weights_tuple, l2_weights_tuple,
        cumulative_local_expert_recv_stats,
        sym_buffer, sym_buffer_ptrs, rank_idx,
        num_max_tokens_per_rank, requested_num_max_tokens_per_rank,
        num_experts, num_topk,
        recipe, activation, activation_clamp_opt, fast_math);
}

static void fp8_fp4_mega_moe_sm90(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& requested_num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    check_sm90_mega_moe_rdma_topology(num_ranks);
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 9);

    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(requested_num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(
        requested_num_max_tokens_per_rank <= num_max_tokens_per_rank);
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");

    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_sm90_fp4_mega_moe(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_sm90_fp4_mega_moe(l2_weights);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(intermediate_hidden / 64 <= 64);

    check_sm90_fp4_sfb_layout(l1_weights_sf, intermediate_hidden * 2, hidden,
                              num_experts_per_rank);
    check_sm90_fp4_sfb_layout(l2_weights_sf, hidden, intermediate_hidden,
                              num_experts_per_rank);
    DG_HOST_ASSERT(l1_weights_sf.is_contiguous() == l2_weights_sf.is_contiguous() or
                   l1_weights_sf.size(2) == 1 or l2_weights_sf.size(2) == 1);

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_sm90_mega_moe_impl(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);
    (void)x;
    (void)x_sf;
    (void)topk_idx;
    (void)topk_weights;

    DG_HOST_ASSERT(get_env<int>("DG_USE_FP4_ACTS") == 0);
    DG_HOST_ASSERT(get_env<int>("DG_USE_FP8_COMBINE") == 0);

    auto fp4_defaults = get_fp4_sm90_api_defaults(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden);
    // Expert-ready dispatch and full-row async combine use one extra gateway QP.
    const auto num_rc_per_pe = get_env<int>("NVSHMEM_IBGDA_NUM_RC_PER_PE", 0);
    DG_HOST_ASSERT(num_rc_per_pe >= num_experts_per_rank + 1);
    sm90_fp8_fp4_mega_moe(y,
                          l1_acts, l1_acts_sf,
                          l2_acts, l2_acts_sf,
                          l1_weights, l2_weights,
                          l1_weights_sf, l2_weights_sf,
                          cumulative_local_expert_recv_stats,
                          sym_buffer_ptrs,
                          rank_idx, num_max_tokens_per_rank,
                          requested_num_max_tokens_per_rank,
                          num_experts_per_rank,
                          num_tokens, num_topk,
                          hidden, intermediate_hidden,
                          activation_clamp, fast_math,
                          fp4_defaults.wide_load_decode,
                          fp4_defaults.early_b_decode,
                          fp4_defaults.decode_done_mbarrier,
                          fp4_defaults.ss_nsplit,
                          fp4_defaults.swap_ab,
                          fp4_defaults.swap_ab_fast_amax);

    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void register_sm90_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_sm90_mega_moe", &get_token_alignment_for_sm90_mega_moe);
    m.def("get_symm_buffer_size_for_sm90_mega_moe", &get_symm_buffer_size_for_sm90_mega_moe);
    m.def("fp8_fp4_mega_moe_sm90", &fp8_fp4_mega_moe_sm90);
    m.def("fp8_mega_moe", &fp8_mega_moe);
#endif
}

} // namespace deep_gemm::mega
