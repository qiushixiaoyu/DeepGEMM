#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/fp4_decode_detail.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/impls/sm90_mega_moe_combine_ring.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
#include <nccl_device.h>
#endif

namespace deep_gemm {

__forceinline__ __device__ void sm90_fp8_fp4_mega_moe_get_e4m3_sf_and_sf_inv(
    const float2& amax, float2& sf, float2& sf_inv) {
    constexpr float kScale = 1.0f / 448.0f;
    const auto scaled = make_float2(__fmul_rn(amax.x, kScale), __fmul_rn(amax.y, kScale));
    const auto exp_x = math::fast_log2_ceil(scaled.x);
    const auto exp_y = math::fast_log2_ceil(scaled.y);
    sf.x = math::fast_pow2(exp_x), sf_inv.x = math::fast_pow2(-exp_x);
    sf.y = math::fast_pow2(exp_y), sf_inv.y = math::fast_pow2(-exp_y);
}

struct SM90FP8FP4MegaMoEData {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    CUTLASS_HOST_DEVICE
    constexpr explicit SM90FP8FP4MegaMoEData(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
#if defined(__CUDA_ARCH__)
        DG_TRAP_ONLY_DEVICE_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
#else
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
#endif
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct SM90FP8FP4MegaMoEBuffer {
    SM90FP8FP4MegaMoEData data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;
    void* base;

    CUTLASS_HOST_DEVICE
    SM90FP8FP4MegaMoEBuffer(const SM90FP8FP4MegaMoEData& data_layout,
                            const uint32_t& num_ranks,
                            const uint32_t& max_num_tokens_per_rank,
                            void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    SM90FP8FP4MegaMoEBuffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    SM90FP8FP4MegaMoEData get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
#if defined(__CUDA_ARCH__)
        DG_TRAP_ONLY_DEVICE_ASSERT(num_ranks == 1 or global);
#else
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
#endif
        return SM90FP8FP4MegaMoEData(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

template <uint32_t kNumExpertsPerRank, uint32_t kNumExpertsPerLane, typename Scheduler>
CUTLASS_DEVICE void sm90_fp8_fp4_mega_moe_fetch_cached_expert_recv_count(
    Scheduler& scheduler,
    const uint32_t* cached_recv_counts) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
        const auto expert_idx = i * 32 + ptx::get_lane_idx();
        uint32_t value = 0;
        if (expert_idx < kNumExpertsPerRank)
            value = cached_recv_counts[expert_idx];
        scheduler.stored_num_tokens_per_expert[i] = value;
    }
    __syncwarp();
}

template <
    uint32_t kNumExpertsPerRank,
    uint32_t kNumExpertsPerLane,
    uint32_t kNumL1BlockKs,
    uint32_t kNumL2BlockKs,
    typename Scheduler,
    typename Func>
CUTLASS_DEVICE void sm90_fp8_fp4_mega_moe_for_each_cached_block(
    Scheduler& scheduler,
    Func&& func,
    const uint32_t* cached_recv_counts) {
    sm90_fp8_fp4_mega_moe_fetch_cached_expert_recv_count<
        kNumExpertsPerRank, kNumExpertsPerLane>(scheduler, cached_recv_counts);
    scheduler.set_expert_idx(0);

    while (true) {
        CUTE_TIE_DECL(scheduler.get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
        if (block_phase == sched::BlockPhase::None)
            break;

        // The scheduler selects L1/L2 at runtime, while both K extents are
        // fixed by the JIT shape.  Dispatch through a templated callback so
        // every participating role sees an exact phase and K-loop bound.
        if (block_phase == sched::BlockPhase::Linear2) {
            func.template operator()<
                sched::BlockPhase::Linear2, kNumL2BlockKs>(
                    current_local_expert_idx, m_block_idx, n_block_idx);
        } else {
            func.template operator()<
                sched::BlockPhase::Linear1, kNumL1BlockKs>(
                    current_local_expert_idx, m_block_idx, n_block_idx);
        }
    }
}

template <
    uint32_t LOAD_BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t kScaleBGranK,
    uint32_t kNumSFBPerBlockK,
    typename PackedT,
    typename DecodedT>
__device__ __forceinline__ void dequant_fp4_b_tile_to_e4m3_smem_wide_load(
    const uint32_t decode_thread_idx,
    const uint32_t num_decode_threads,
    const PackedT* __restrict__ smem_b_packed_stage,
    DecodedT* __restrict__ smem_b_stage,
    const uint32_t* __restrict__ smem_sfb_stage) {
    constexpr uint32_t kPackedWordsPerKG = kScaleBGranK / 8;  // 4
    constexpr uint32_t kGroupsPerTile = LOAD_BLOCK_N * kNumSFBPerBlockK;
    DG_STATIC_ASSERT(kPackedWordsPerKG == 4, "Wide-load decode assumes per-32K groups");

    for (uint32_t group = decode_thread_idx; group < kGroupsPerTile; group += num_decode_threads) {
        const uint32_t n_row = group / kNumSFBPerBlockK;
        const uint32_t kg = group - n_row * kNumSFBPerBlockK;
#ifdef DG_MEGA_MOE_FP4_SFB_SUBGROUP_BROADCAST
        uint32_t sfb_word = 0;
        const uint32_t warp_lane = threadIdx.x & 31u;
        if ((warp_lane & 3u) == 0)
            sfb_word = smem_sfb_stage[n_row];
        sfb_word = __shfl_sync(0xffffffffu, sfb_word, warp_lane & ~3u);
#else
        const uint32_t sfb_word = smem_sfb_stage[n_row];
#endif
        const uint32_t e8m0 = (sfb_word >> (kg * 8)) & 0xffu;

        const auto* packed_row = reinterpret_cast<const uint32_t*>(
            smem_b_packed_stage + n_row * (BLOCK_K / 2));
        auto* decoded_row_u64 = reinterpret_cast<uint64_t*>(
            smem_b_stage + n_row * BLOCK_K);
        const uint32_t row_swizzle = n_row & 7u;

        const uint32_t seg_base = kg * 2u;
        const uint32_t swz_seg_0 = seg_base ^ row_swizzle;
        const uint32_t swz_seg_1 = (seg_base + 1u) ^ row_swizzle;
        const uint64_t scaled_lut =
            fp4_decode_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0);
        const uint32_t scaled_lut_lo = static_cast<uint32_t>(scaled_lut);
        const uint32_t scaled_lut_hi = static_cast<uint32_t>(scaled_lut >> 32);

        const uint4 packed = reinterpret_cast<const uint4*>(packed_row)[kg];
        const uint32_t lo_0 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.x & 0xffffu, scaled_lut_lo, scaled_lut_hi);
        const uint32_t hi_0 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.x >> 16, scaled_lut_lo, scaled_lut_hi);
        const uint32_t lo_1 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.y & 0xffffu, scaled_lut_lo, scaled_lut_hi);
        const uint32_t hi_1 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.y >> 16, scaled_lut_lo, scaled_lut_hi);
        const uint32_t lo_2 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.z & 0xffffu, scaled_lut_lo, scaled_lut_hi);
        const uint32_t hi_2 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.z >> 16, scaled_lut_lo, scaled_lut_hi);
        const uint32_t lo_3 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.w & 0xffffu, scaled_lut_lo, scaled_lut_hi);
        const uint32_t hi_3 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.w >> 16, scaled_lut_lo, scaled_lut_hi);
        ptx::st_shared(
            decoded_row_u64 + swz_seg_0 * 2u,
            lo_0, hi_0, lo_1, hi_1);
        ptx::st_shared(
            decoded_row_u64 + swz_seg_1 * 2u,
            lo_2, hi_2, lo_3, hi_3);
    }
}

template <
    uint32_t LOAD_BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t kScaleBGranK,
    uint32_t kNumSFBPerBlockK,
    typename PackedT,
    typename DecodedT>
__device__ __forceinline__ void dequant_fp4_b_tile_to_e4m3_smem_vec_store(
    const uint32_t decode_thread_idx,
    const uint32_t num_decode_threads,
    const PackedT* __restrict__ smem_b_packed_stage,
    DecodedT* __restrict__ smem_b_stage,
    const uint32_t* __restrict__ smem_sfb_stage) {
    constexpr uint32_t kPackedWordsPerKG = kScaleBGranK / 8;  // 4
    constexpr uint32_t kPackedWordPairsPerKG = kPackedWordsPerKG / 2;
    constexpr uint32_t kGroupsPerTile = LOAD_BLOCK_N * kNumSFBPerBlockK;
    DG_STATIC_ASSERT(kPackedWordsPerKG == 4, "Vector-store decode assumes per-32K groups");

    for (uint32_t group = decode_thread_idx; group < kGroupsPerTile; group += num_decode_threads) {
        const uint32_t n_row = group / kNumSFBPerBlockK;
        const uint32_t kg = group - n_row * kNumSFBPerBlockK;
#ifdef DG_MEGA_MOE_FP4_SFB_SUBGROUP_BROADCAST
        uint32_t sfb_word = 0;
        const uint32_t warp_lane = threadIdx.x & 31u;
        if ((warp_lane & 3u) == 0)
            sfb_word = smem_sfb_stage[n_row];
        sfb_word = __shfl_sync(0xffffffffu, sfb_word, warp_lane & ~3u);
#else
        const uint32_t sfb_word = smem_sfb_stage[n_row];
#endif
        const uint32_t e8m0 = (sfb_word >> (kg * 8)) & 0xffu;

        const auto* packed_row = reinterpret_cast<const uint32_t*>(
            smem_b_packed_stage + n_row * (BLOCK_K / 2));
        auto* decoded_row_u64 = reinterpret_cast<uint64_t*>(
            smem_b_stage + n_row * BLOCK_K);
        const uint32_t row_swizzle = n_row & 7u;
        const uint64_t scaled_lut =
            fp4_decode_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0);
        const uint32_t scaled_lut_lo = static_cast<uint32_t>(scaled_lut);
        const uint32_t scaled_lut_hi = static_cast<uint32_t>(scaled_lut >> 32);

        #pragma unroll
        for (uint32_t pair = 0; pair < kPackedWordPairsPerKG; ++ pair) {
            const uint32_t pw_global_0 = kg * kPackedWordsPerKG + pair * 2u;
            const uint32_t packed_0 = packed_row[pw_global_0];
            const uint32_t packed_1 = packed_row[pw_global_0 + 1u];
            const uint32_t lo_0 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
                packed_0 & 0xffffu, scaled_lut_lo, scaled_lut_hi);
            const uint32_t hi_0 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
                packed_0 >> 16, scaled_lut_lo, scaled_lut_hi);
            const uint32_t lo_1 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
                packed_1 & 0xffffu, scaled_lut_lo, scaled_lut_hi);
            const uint32_t hi_1 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
                packed_1 >> 16, scaled_lut_lo, scaled_lut_hi);
            const uint32_t seg_id = pw_global_0 >> 1;
            const uint32_t swz_seg = seg_id ^ row_swizzle;
            ptx::st_shared(
                decoded_row_u64 + swz_seg * 2u,
                lo_0, hi_0, lo_1, hi_1);
        }
    }
}

template <
    uint32_t LOAD_BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t kScaleBGranK,
    uint32_t kNumSFBPerBlockK,
    bool kUseWideLoadDecode,
    typename PackedT,
    typename DecodedT>
__device__ __forceinline__ void dequant_fp4_b_tile_to_e4m3_smem_dispatch(
    const uint32_t decode_thread_idx,
    const uint32_t num_decode_threads,
    const PackedT* __restrict__ smem_b_packed_stage,
    DecodedT* __restrict__ smem_b_stage,
    const uint32_t* __restrict__ smem_sfb_stage) {
    if constexpr (kUseWideLoadDecode) {
        dequant_fp4_b_tile_to_e4m3_smem_wide_load<
            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK>(
            decode_thread_idx, num_decode_threads,
            smem_b_packed_stage, smem_b_stage, smem_sfb_stage);
    } else {
        dequant_fp4_b_tile_to_e4m3_smem_vec_store<
            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK>(
            decode_thread_idx, num_decode_threads,
            smem_b_packed_stage, smem_b_stage, smem_sfb_stage);
    }
}

// ============================================================================
// Optional SM90 FP4 GPU sidecar publisher.
// ----------------------------------------------------------------------------
// This kernel owns only combine publication.  Each 16-warp CTA covers all
// destinations for one strided subset of local experts.  The host gives every
// CTA enough dynamic shared memory to prevent co-residency with a compute CTA
// and reserves the same number of SMs from the fused grid.
//
// The first experiment deliberately keeps the established full-pool metadata
// traversal and IBGDA protocol.  It therefore isolates code/register/scheduler
// interference from descriptor-queue and row-coalescing changes.  Compact
// combine-ring support is left to a later change after this causal A/B.
// ============================================================================
template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumL1RingTokens,
    uint32_t kNumL1SFStorageTokens,
    uint32_t kNumL2RingTokens,
    uint32_t kNumL2SFStorageTokens,
    uint32_t kNumCombineStageTokens,
    uint32_t kNumRanks,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks>
CUTLASS_GLOBAL __launch_bounds__(512, 1) void
sm90_fp8_fp4_mega_moe_sidecar_publisher(
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
#ifdef DG_MEGA_MOE_INTERNODE
#ifdef DG_MEGA_MOE_FP4_SIDECAR_AGGREGATE_LOCAL
    constexpr uint32_t kLocalRanks = DG_MEGA_MOE_NVL_PEERS;
    constexpr uint32_t kRemotePublisherWarps = kLocalRanks;
    constexpr uint32_t kLocalAggregatorWarp = kRemotePublisherWarps;
    constexpr uint32_t kPublisherWarpsPerCTA =
        kRemotePublisherWarps + 1;
#else
    constexpr uint32_t kPublisherWarpsPerCTA = 16;
#endif
    constexpr uint32_t kPublishRowsPerBatch = 32;
    constexpr uint32_t kCombineFullRowPublishReadyBit = 1u << 31;
    constexpr uint32_t kNumL1RingBlocks = kNumL1RingTokens / BLOCK_M;
    constexpr uint32_t kNumL2RingBlocks = kNumL2RingTokens / BLOCK_M;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, 128u);
    constexpr bool kL1RingStorageCompact =
        kNumL1RingTokens < kNumMaxPoolTokens;
    constexpr bool kL2RingStorageCompact =
        kNumL2RingTokens < kNumMaxPoolTokens;
#ifdef DG_MEGA_MOE_L1_RING_ACTIVE
    constexpr bool kL1RingEnabled = kL1RingStorageCompact;
#else
    constexpr bool kL1RingEnabled = false;
#endif
    constexpr uint32_t kNumL2StorageTokens =
        kNumL2RingTokens + (kL2RingStorageCompact ? 128u : 0u);
    constexpr uint32_t kL2ActsSFGranK = BLOCK_N == 64 ? 32 : 64;

    DG_STATIC_ASSERT(kNumRanks == 16,
                     "The first sidecar A/B targets the two-node 16-rank topology");
    DG_STATIC_ASSERT(kNumExpertsPerRank <= 32,
                     "Sidecar pending mask supports at most 32 local experts");
    DG_STATIC_ASSERT(kNumCombineStageTokens == kNumMaxPoolTokens,
                     "Sidecar A/B currently requires full-pool combine staging");
    DG_STATIC_ASSERT(kNumL1RingTokens % BLOCK_M == 0 and
                         kNumL2RingTokens % BLOCK_M == 0,
                     "Invalid SM90 FP4 ring capacity");
    DG_STATIC_ASSERT(kNumL1SFStorageTokens >=
                         kNumL1RingBlocks * math::constexpr_align(BLOCK_M, 128u),
                     "Invalid L1 scale storage");
    DG_STATIC_ASSERT(kNumL2SFStorageTokens >=
                         kNumL2RingBlocks * math::constexpr_align(BLOCK_M, 128u),
                     "Invalid L2 scale storage");

    const auto get_l1_ring_block_idx = [](const uint32_t& pool_block_idx) {
        if constexpr (kL1RingEnabled)
            return pool_block_idx % kNumL1RingBlocks;
        return pool_block_idx;
    };
    const auto get_l1_ring_token_idx = [&](const uint32_t& pool_token_idx) {
        const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;
        return get_l1_ring_block_idx(pool_block_idx) * BLOCK_M +
            pool_token_idx % BLOCK_M;
    };
    const auto get_l1_ring_wave_idx = [](const uint32_t& pool_block_idx) {
        if constexpr (kL1RingEnabled)
            return pool_block_idx / kNumL1RingBlocks;
        return 0u;
    };

    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
#ifdef DG_MEGA_MOE_FP4_SIDECAR_AGGREGATE_LOCAL
    DG_STATIC_ASSERT(kNumRanks == 2 * kLocalRanks,
                     "Local aggregation requires exactly two NVLink domains");
    const uint32_t local_rank_base =
        sym_buffer.rank_idx / kLocalRanks * kLocalRanks;
    const uint32_t remote_rank_base = local_rank_base ^ kLocalRanks;
    const bool is_local_aggregator = warp_idx == kLocalAggregatorWarp;
    const uint32_t dst_rank_idx = is_local_aggregator ?
        local_rank_base : remote_rank_base + warp_idx;
#else
    const uint32_t dst_rank_idx = warp_idx;
#endif

    // The dynamic region is intentionally touched: launch-time allocation is
    // what pins each sidecar CTA to its own SM instead of allowing both CTAs
    // to share one SM and leave a compute CTA unschedulable.
    extern __shared__ __align__(16) uint8_t sidecar_sm_reservation[];
    if (threadIdx.x == 0)
        sidecar_sm_reservation[0] = 0;
    __syncthreads();

    const auto workspace = layout::SM90Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk);

    // Recreate the host-agreed symmetric-buffer layout only as far as the
    // combine destination and full-row staging regions used by publication.
    constexpr auto fp8_token_layout = SM90FP8FP4MegaMoEData(kHidden);
    constexpr auto bf16_token_layout =
        SM90FP8FP4MegaMoEData(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout =
        SM90FP8FP4MegaMoEData(kIntermediateHidden);
    constexpr auto fp8_sf_layout = SM90FP8FP4MegaMoEData(kHidden / 32);
    constexpr auto fp8_intermediate_sf_layout = SM90FP8FP4MegaMoEData(
        kIntermediateHidden * sizeof(float) / kL2ActsSFGranK);
    constexpr auto input_topk_idx_layout =
        SM90FP8FP4MegaMoEData(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout =
        SM90FP8FP4MegaMoEData(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout =
        SM90FP8FP4MegaMoEData(sizeof(float), false);

    const auto input_token_buffer = SM90FP8FP4MegaMoEBuffer(
        fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer = SM90FP8FP4MegaMoEBuffer(
        fp8_sf_layout, 1, kNumMaxTokensPerRank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = SM90FP8FP4MegaMoEBuffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = SM90FP8FP4MegaMoEBuffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank,
        input_topk_idx_buffer.get_end_ptr());
    const auto l1_token_buffer = SM90FP8FP4MegaMoEBuffer(
        fp8_token_layout, 1, kNumL1RingTokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = SM90FP8FP4MegaMoEBuffer(
        fp8_sf_layout, 1, kNumL1SFStorageTokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = SM90FP8FP4MegaMoEBuffer(
        l1_topk_weights_layout, 1, kNumL1RingTokens,
        l1_sf_buffer.get_end_ptr());
    const auto l2_token_buffer = SM90FP8FP4MegaMoEBuffer(
        fp8_intermediate_token_layout, 1, kNumL2StorageTokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = SM90FP8FP4MegaMoEBuffer(
        fp8_intermediate_sf_layout, 1, kNumL2SFStorageTokens,
        l2_token_buffer.get_end_ptr());
    const auto combine_token_buffer = SM90FP8FP4MegaMoEBuffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank,
        l2_sf_buffer.get_end_ptr());
    constexpr auto dispatch_staging_layout = SM90FP8FP4MegaMoEData(
        math::constexpr_align<uint32_t>(
            kHidden / 32 + sizeof(float), 128u), false);
    const auto dispatch_staging_buffer = SM90FP8FP4MegaMoEBuffer(
        dispatch_staging_layout, 1, kNumMaxPoolTokens,
        combine_token_buffer.get_end_ptr());
    const auto combine_full_row_arrival_buffer = SM90FP8FP4MegaMoEBuffer(
        SM90FP8FP4MegaMoEData(sizeof(uint32_t), false), 1,
        kNumMaxPoolTokens / layout::kMinCandidateBlockM,
        dispatch_staging_buffer.get_end_ptr());
    const auto combine_full_row_ready_timestamp_buffer =
        SM90FP8FP4MegaMoEBuffer(
            SM90FP8FP4MegaMoEData(sizeof(uint64_t), false), 1,
            kNumMaxPoolTokens / layout::kMinCandidateBlockM,
            combine_full_row_arrival_buffer.get_end_ptr());
    constexpr uint32_t kPublishRowMaskStorageWords = 4;
    const auto combine_publish_row_mask_buffer = SM90FP8FP4MegaMoEBuffer(
        SM90FP8FP4MegaMoEData(
            kNumRanks * kPublishRowMaskStorageWords * sizeof(uint32_t),
            false),
        1, kNumMaxPoolTokens / layout::kMinCandidateBlockM,
        combine_full_row_ready_timestamp_buffer.get_end_ptr());
    const auto get_combine_publish_row_mask_ptr =
        [&](const uint32_t& pool_block_idx, const uint32_t& dst_rank_idx,
            const uint32_t& mask_word_idx = 0) {
            return combine_publish_row_mask_buffer
                       .get_data_buffer(pool_block_idx)
                       .get_base_ptr<uint32_t>() +
                dst_rank_idx * kPublishRowMaskStorageWords + mask_word_idx;
        };
    const auto combine_full_row_staging_base = reinterpret_cast<void*>(
        math::align(
            reinterpret_cast<uint64_t>(
                combine_publish_row_mask_buffer.get_end_ptr()),
            static_cast<uint64_t>(128)));
    const auto combine_full_row_staging_buffer = SM90FP8FP4MegaMoEBuffer(
        SM90FP8FP4MegaMoEData(kHidden * sizeof(nv_bfloat16)), 1,
        kNumCombineStageTokens, combine_full_row_staging_base);

    // Arm exactly the next epoch only after every publisher CTA is resident
    // and has observed the same current epoch.  The fused kernel waits for
    // this release before incrementing launch_epoch, so no late sidecar block
    // can accidentally attach itself to the following invocation.
    __shared__ uint64_t sidecar_previous_epoch;
    if (threadIdx.x == 0) {
        sidecar_previous_epoch =
            ptx::ld_acq_sys(workspace.get_launch_epoch_ptr());
        const auto arrival_ptr =
            workspace.get_sidecar_publisher_arrival_count_ptr();
        const uint32_t old_arrivals =
            ptx::atomic_add_acq_rel_sys(arrival_ptr, 1);
        DG_TRAP_ONLY_DEVICE_ASSERT(old_arrivals < gridDim.x);
        if (old_arrivals + 1 == gridDim.x) {
            ptx::st_release_sys(arrival_ptr, 0u);
            ptx::st_release_sys(
                workspace.get_sidecar_publisher_armed_epoch_ptr(),
                sidecar_previous_epoch + 1);
        }
    }
    __syncthreads();
    const uint64_t target_epoch = sidecar_previous_epoch + 1;
    constexpr uint64_t kTimeoutCycles = 60ull * 2000000000ull;
    const uint64_t epoch_wait_start = clock64();
    uint64_t armed_epoch = ptx::ld_acq_sys(
        workspace.get_sidecar_publisher_armed_epoch_ptr());
    while (armed_epoch != target_epoch) {
        DG_TRAP_ONLY_DEVICE_ASSERT(
            clock64() - epoch_wait_start < kTimeoutCycles);
        armed_epoch = ptx::ld_acq_sys(
            workspace.get_sidecar_publisher_armed_epoch_ptr());
    }
    uint64_t launch_epoch = ptx::ld_acq_sys(workspace.get_launch_epoch_ptr());
    while (launch_epoch != target_epoch) {
        DG_TRAP_ONLY_DEVICE_ASSERT(
            clock64() - epoch_wait_start < kTimeoutCycles);
        launch_epoch = ptx::ld_acq_sys(workspace.get_launch_epoch_ptr());
    }
    __syncwarp();

    if (warp_idx >= kPublisherWarpsPerCTA or dst_rank_idx >= kNumRanks)
        return;

    __shared__ uint32_t expert_pair_tokens
        [kNumExpertsPerRank][kNumRanks];
    __shared__ uint32_t expert_total_tokens[kNumExpertsPerRank];
    __shared__ uint32_t expert_blocks[kNumExpertsPerRank];
    __shared__ uint32_t expert_pool[kNumExpertsPerRank];
#ifdef DG_MEGA_MOE_FP4_SIDECAR_EXPERT_CENTRIC
    __shared__ uint32_t expert_publish_group_done[kNumExpertsPerRank];
#endif
    __shared__ uint32_t chain_cursor
        [kPublisherWarpsPerCTA][kNumExpertsPerRank];
#ifdef DG_MEGA_MOE_FP4_SIDECAR_SHARED_METADATA
    __shared__ layout::TokenSrcMetadata
        shared_publish_metadata[BLOCK_M];
    __shared__ uint32_t shared_publish_cursor[kNumExpertsPerRank];
    __shared__ uint32_t shared_publish_pending;
    __shared__ uint32_t shared_publish_selected_expert;
    __shared__ uint32_t shared_publish_selected_cursor;
    __shared__ uint32_t shared_publish_selected_has_block;
#endif

    const auto wait_dispatch_count =
        [&](const uint32_t& src_rank_idx,
            const uint32_t& expert_idx) -> uint32_t {
        constexpr uint64_t kCountTimeoutCycles = 60ull * 2000000000ull;
        const uint32_t expected_epoch = static_cast<uint32_t>(launch_epoch);
        const uint32_t epoch_slot = expected_epoch & 1u;
        const uint64_t wait_start = clock64();
        const uint64_t* slot_ptr = nullptr;
#ifdef DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3
        slot_ptr = workspace.get_dispatch_epoch_count_ptr(
            epoch_slot, src_rank_idx, expert_idx);
#else
#ifdef DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED
        const uint32_t local_node_idx =
            sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
        const uint32_t src_node_idx =
            src_rank_idx / DG_MEGA_MOE_NVL_PEERS;
        if (src_node_idx != local_node_idx) {
            const uint32_t rel_src_node = layout::gateway_rel_node(
                src_node_idx, local_node_idx);
            const uint32_t cell_idx =
                src_rank_idx % DG_MEGA_MOE_NVL_PEERS *
                    kNumExpertsPerRank + expert_idx;
            if constexpr (kNumTopk >= 8) {
                const auto header = workspace.get_gateway_packed_header_ptr(
                    true, rel_src_node, epoch_slot);
                uint64_t header_epoch = ptx::ld_acq_sys(&header->epoch);
                while (static_cast<uint32_t>(header_epoch) !=
                       expected_epoch) {
                    DG_TRAP_ONLY_DEVICE_ASSERT(
                        clock64() - wait_start < kCountTimeoutCycles);
                    header_epoch = ptx::ld_acq_sys(&header->epoch);
                }
                const uint32_t total_entries = ptx::ld_acq_sys(
                    &header->total_entries);
                DG_TRAP_ONLY_DEVICE_ASSERT(
                    total_entries <=
                    workspace.get_gateway_max_packed_entries());
                slot_ptr =
                    workspace.get_gateway_packed_compact_manifest_ptr(
                        true, rel_src_node, epoch_slot,
                        total_entries, cell_idx);
            } else {
                slot_ptr = workspace.get_gateway_packed_manifest_ptr(
                    true, rel_src_node, epoch_slot, cell_idx);
            }
        } else {
            slot_ptr = workspace.get_dispatch_epoch_count_ptr(
                epoch_slot, src_rank_idx, expert_idx);
        }
#endif
#endif
        DG_TRAP_ONLY_DEVICE_ASSERT(slot_ptr != nullptr);
        uint64_t value = ptx::ld_acq_sys(slot_ptr);
        while (static_cast<uint32_t>(value >> 32) != expected_epoch) {
            DG_TRAP_ONLY_DEVICE_ASSERT(
                clock64() - wait_start < kCountTimeoutCycles);
            value = ptx::ld_acq_sys(slot_ptr);
        }
        return static_cast<uint32_t>(value);
    };

    // Counts are destination-independent.  Spread experts across all resident
    // publisher warps so the 16 source-rank epoch loads for different experts
    // can overlap; the old single-warp loop serialized all 256 system loads.
    const uint32_t active_publisher_warps = blockDim.x / 32;
    for (uint32_t expert = warp_idx; expert < kNumExpertsPerRank;
         expert += active_publisher_warps) {
        uint32_t source_count = 0;
        if (lane_idx < kNumRanks) {
            source_count = wait_dispatch_count(lane_idx, expert);
            expert_pair_tokens[expert][lane_idx] = source_count;
        }
        const uint32_t expert_total =
            __reduce_add_sync(0xffffffffu, source_count);
        if (lane_idx == 0) {
            expert_total_tokens[expert] = expert_total;
            expert_blocks[expert] = math::ceil_div(expert_total, BLOCK_M);
        }
    }
    __syncthreads();

    // Only the short 16-element prefix sum remains serial.
    if (threadIdx.x == 0) {
        uint32_t pool_block_offset = 0;
        #pragma unroll
        for (uint32_t expert = 0; expert < kNumExpertsPerRank; ++ expert) {
            expert_pool[expert] = pool_block_offset;
            pool_block_offset += expert_blocks[expert];
#ifdef DG_MEGA_MOE_FP4_SIDECAR_EXPERT_CENTRIC
            expert_publish_group_done[expert] = 0;
#endif
        }
    }
#if !defined(DG_MEGA_MOE_FP4_SIDECAR_EXPERT_CENTRIC) && \
    !defined(DG_MEGA_MOE_FP4_SIDECAR_SHARED_METADATA)
    for (uint32_t expert = lane_idx; expert < kNumExpertsPerRank;
         expert += 32)
        chain_cursor[warp_idx][expert] = 0;
#endif
    __syncthreads();

#ifdef DG_MEGA_MOE_FP4_SIDECAR_DISPATCH_RDMA
    // Move only the inter-node token/SF/top-k-weight pulls out of the fused
    // kernel.  The fused dispatch warps retain route/count production and the
    // NVLink-local pull path.  Sidecar warps reproduce the scheduler's
    // round-robin source ordering so both producers address exactly the same
    // logical pool without a descriptor queue.
    const uint32_t global_dispatch_warp =
        blockIdx.x * kPublisherWarpsPerCTA + warp_idx;
    const uint32_t num_global_dispatch_warps =
        gridDim.x * kPublisherWarpsPerCTA;
    const uint32_t local_node_idx =
        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
    const uint32_t dispatch_epoch_slot =
        static_cast<uint32_t>(launch_epoch) & 1u;

    for (uint32_t expert = 0; expert < kNumExpertsPerRank; ++expert) {
        const uint32_t expert_total = expert_total_tokens[expert];
        for (uint32_t token_idx_in_expert = global_dispatch_warp;
             token_idx_in_expert < expert_total;
             token_idx_in_expert += num_global_dispatch_warps) {
            uint32_t remaining[kNumRanks];
            #pragma unroll
            for (uint32_t rank = 0; rank < kNumRanks; ++rank)
                remaining[rank] = expert_pair_tokens[expert][rank];

            uint32_t slot_idx = token_idx_in_expert;
            uint32_t round_offset = 0;
            uint32_t src_rank_idx = 0;
            uint32_t token_idx_in_rank = 0;
            while (true) {
                uint32_t num_active_ranks = 0;
                uint32_t round_length = 0xffffffffu;
                #pragma unroll
                for (uint32_t rank = 0; rank < kNumRanks; ++rank) {
                    if (remaining[rank] != 0) {
                        ++num_active_ranks;
                        round_length = cute::min(
                            round_length, remaining[rank]);
                    }
                }
                DG_TRAP_ONLY_DEVICE_ASSERT(num_active_ranks != 0);
                const uint32_t round_tokens =
                    round_length * num_active_ranks;
                if (slot_idx < round_tokens) {
                    uint32_t selected = slot_idx % num_active_ranks;
                    #pragma unroll
                    for (uint32_t rank = 0; rank < kNumRanks; ++rank) {
                        if (remaining[rank] != 0) {
                            if (selected == 0) {
                                src_rank_idx = rank;
                                break;
                            }
                            --selected;
                        }
                    }
                    token_idx_in_rank =
                        round_offset + slot_idx / num_active_ranks;
                    break;
                }
                slot_idx -= round_tokens;
                round_offset += round_length;
                #pragma unroll
                for (uint32_t rank = 0; rank < kNumRanks; ++rank)
                    remaining[rank] -=
                        cute::min(remaining[rank], round_length);
            }

            if (src_rank_idx / DG_MEGA_MOE_NVL_PEERS == local_node_idx)
                continue;

            uint32_t src_token_topk_idx = 0;
            if (lane_idx == 0) {
#ifdef DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED
                src_token_topk_idx =
                    *workspace.get_gateway_packed_landing_entry_ptr(
                        layout::gateway_rel_node(
                            src_rank_idx / DG_MEGA_MOE_NVL_PEERS,
                            local_node_idx),
                        dispatch_epoch_slot,
                        src_rank_idx % DG_MEGA_MOE_NVL_PEERS,
                        expert, token_idx_in_rank);
#else
                src_token_topk_idx =
                    *workspace.get_gateway_dense_landing_entry_ptr(
                        layout::gateway_rel_node(
                            src_rank_idx / DG_MEGA_MOE_NVL_PEERS,
                            local_node_idx),
                        dispatch_epoch_slot,
                        src_rank_idx % DG_MEGA_MOE_NVL_PEERS,
                        expert, token_idx_in_rank);
#endif
            }
            src_token_topk_idx = __shfl_sync(
                0xffffffffu, src_token_topk_idx, 0);
            const uint32_t src_token_idx =
                src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx =
                src_token_topk_idx % kNumTopk;
            const uint32_t pool_token_idx =
                expert_pool[expert] * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;
            const uint32_t token_idx_in_block = pool_token_idx % BLOCK_M;
            if constexpr (kL1RingEnabled) {
                const uint32_t empty_target =
                    get_l1_ring_wave_idx(pool_block_idx) *
                    (kHidden / BLOCK_N);
                const auto empty_ptr = workspace.get_l1_ring_empty_count_ptr(
                    get_l1_ring_block_idx(pool_block_idx));
                while (ptx::ld_acq(empty_ptr) < empty_target);
            }
            const uint32_t l1_ring_token_idx =
                get_l1_ring_token_idx(pool_token_idx);
            const auto inter_staging = dispatch_staging_buffer
                .get_data_buffer(pool_token_idx).get_base_ptr<float>();

            if (cute::elect_one_sync()) {
                const comm::ibgda::GetRequest read_requests[3] = {
                    {
                        reinterpret_cast<uint64_t>(
                            l1_token_buffer
                                .get_data_buffer(l1_ring_token_idx)
                                .get_base_ptr()),
                        reinterpret_cast<uint64_t>(
                            input_token_buffer
                                .get_data_buffer(src_token_idx)
                                .get_base_ptr()),
                        kHidden
                    },
                    {
                        reinterpret_cast<uint64_t>(inter_staging),
                        reinterpret_cast<uint64_t>(
                            input_sf_buffer
                                .get_data_buffer(src_token_idx)
                                .get_base_ptr<float>()),
                        (kHidden / 128) * sizeof(float)
                    },
                    {
                        reinterpret_cast<uint64_t>(
                            inter_staging + kHidden / 128),
                        reinterpret_cast<uint64_t>(
                            input_topk_weights_buffer
                                .get_base_ptr<float>() +
                            src_token_topk_idx),
                        sizeof(float)
                    }
                };
                const int dispatch_qp_id = static_cast<int>(expert);
                const auto completion_idx =
                    comm::ibgda::get_batch_thread(
                        read_requests, static_cast<int>(src_rank_idx),
                        dispatch_qp_id);
                comm::ibgda::wait_until(
                    static_cast<int>(src_rank_idx), dispatch_qp_id,
                    completion_idx);
            }
            __syncwarp();

            constexpr uint32_t kNumSFFloats = kHidden / 128;
            #pragma unroll
            for (uint32_t sf_idx = lane_idx; sf_idx < kNumSFFloats;
                 sf_idx += 32) {
                const uint32_t sf_pool_token_idx =
                    get_l1_ring_block_idx(pool_block_idx) * SF_BLOCK_M +
                    token_idx_in_block;
                l1_sf_buffer.get_base_ptr<float>()[
                    sf_idx * kNumL1SFStorageTokens + sf_pool_token_idx] =
                    __ldcv(inter_staging + sf_idx);
            }
            __syncwarp();

            if (cute::elect_one_sync()) {
                *l1_topk_weights_buffer
                     .get_data_buffer(l1_ring_token_idx)
                     .get_base_ptr<float>() =
                    __ldcv(inter_staging + kHidden / 128);
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {src_rank_idx, src_token_idx, src_topk_idx};
#ifdef DG_MEGA_MOE_FP4_PUBLISH_ROW_MASK
                atomicOr(
                    get_combine_publish_row_mask_ptr(
                        pool_block_idx, src_rank_idx,
                        token_idx_in_block / 32),
                    1u << (token_idx_in_block % 32));
#endif
                ptx::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(pool_block_idx), 1);
            }
            __syncwarp();
        }
    }
    __syncthreads();
#endif

#ifdef DG_MEGA_MOE_FP4_SIDECAR_SHARED_METADATA
    // Cooperatively load one ready expert block's metadata once, then let the
    // eight remote-destination warps issue their PUTs in parallel from shared
    // memory.  The other eight warps publish same-node ready epochs, so every
    // warp has exactly one destination and the original peer parallelism is
    // retained without dispatch-side row-mask atomics.
    constexpr uint32_t kLocalRanks = DG_MEGA_MOE_NVL_PEERS;
    DG_STATIC_ASSERT(kNumRanks == 2 * kLocalRanks,
                     "Shared metadata publication requires two NVLink domains");
    DG_STATIC_ASSERT(kPublisherWarpsPerCTA == kNumRanks,
                     "Shared metadata publication requires one warp per rank");
    const uint32_t local_rank_base =
        sym_buffer.rank_idx / kLocalRanks * kLocalRanks;
    const uint32_t remote_rank_base = local_rank_base ^ kLocalRanks;
    const bool shared_publish_is_inter = warp_idx < kLocalRanks;
    const uint32_t shared_publish_dst_rank = shared_publish_is_inter ?
        remote_rank_base + warp_idx :
        local_rank_base + warp_idx - kLocalRanks;
    constexpr uint32_t kNoSelectedExpert = 0xffffffffu;

    if (threadIdx.x == 0) {
        uint32_t pending = 0;
        #pragma unroll
        for (uint32_t expert = 0; expert < kNumExpertsPerRank; ++ expert) {
            shared_publish_cursor[expert] = 0;
            if (expert % gridDim.x == blockIdx.x)
                pending |= 1u << expert;
        }
        shared_publish_pending = pending;
    }
    __syncthreads();

    uint32_t scan_start = blockIdx.x;
    const uint64_t shared_publish_wait_start = clock64();
    while (true) {
        if (threadIdx.x == 0) {
            uint32_t selected_expert = kNoSelectedExpert;
            uint32_t selected_cursor = 0;
            uint32_t selected_has_block = 0;
            while (shared_publish_pending != 0 and
                   selected_expert == kNoSelectedExpert) {
                DG_TRAP_ONLY_DEVICE_ASSERT(
                    clock64() - shared_publish_wait_start < kTimeoutCycles);
                #pragma unroll
                for (uint32_t offset = 0;
                     offset < kNumExpertsPerRank; ++ offset) {
                    const uint32_t expert =
                        (scan_start + offset) % kNumExpertsPerRank;
                    if ((shared_publish_pending & (1u << expert)) == 0)
                        continue;
                    const uint32_t cursor =
                        shared_publish_cursor[expert];
                    if (cursor == expert_blocks[expert]) {
                        selected_expert = expert;
                        selected_cursor = cursor;
                        break;
                    }
                    const uint32_t pool_block_idx =
                        expert_pool[expert] + cursor;
                    const auto arrival_ptr = combine_full_row_arrival_buffer
                        .get_data_buffer(pool_block_idx)
                        .get_base_ptr<uint32_t>();
                    if ((ptx::ld_acq(arrival_ptr) &
                         kCombineFullRowPublishReadyBit) != 0) {
                        selected_expert = expert;
                        selected_cursor = cursor;
                        selected_has_block = 1;
                        scan_start = (expert + 1) % kNumExpertsPerRank;
                        break;
                    }
                }
            }
            shared_publish_selected_expert = selected_expert;
            shared_publish_selected_cursor = selected_cursor;
            shared_publish_selected_has_block = selected_has_block;
        }
        __syncthreads();

        const uint32_t expert = shared_publish_selected_expert;
        if (expert == kNoSelectedExpert)
            return;
        const uint32_t cursor = shared_publish_selected_cursor;
        const bool has_block = shared_publish_selected_has_block != 0;
        const uint32_t expert_total = expert_total_tokens[expert];
        uint32_t valid_m = 0;
        uint32_t m_idx = 0;
        if (has_block) {
            const uint32_t pool_block_idx =
                expert_pool[expert] + cursor;
            m_idx = pool_block_idx * BLOCK_M;
            valid_m = cute::min(
                expert_total - cursor * BLOCK_M, BLOCK_M);
            for (uint32_t row = threadIdx.x; row < valid_m;
                 row += blockDim.x)
                shared_publish_metadata[row] =
                    *workspace.get_token_src_metadata_ptr(m_idx + row);
        }
        __syncthreads();

        if (has_block and shared_publish_is_inter) {
            const int scatter_qp_id = static_cast<int>(expert);
            for (uint32_t row_base = 0; row_base < valid_m;
                 row_base += kPublishRowsPerBatch) {
                const uint32_t row = row_base + lane_idx;
                bool row_active = false;
                uint64_t req_rptr = 0;
                uint64_t req_lptr = 0;
                if (row < valid_m) {
                    const auto src_metadata = shared_publish_metadata[row];
                    row_active =
                        src_metadata.rank_idx == shared_publish_dst_rank;
                    if (row_active) {
                        const auto staging_row =
                            combine_full_row_staging_buffer
                                .get_data_buffer(m_idx + row);
                        const auto dst_row = combine_token_buffer
                            .get_rank_buffer(src_metadata.topk_idx)
                            .get_data_buffer(src_metadata.token_idx);
                        req_rptr = reinterpret_cast<uint64_t>(
                            dst_row.get_base_ptr());
                        req_lptr = reinterpret_cast<uint64_t>(
                            staging_row.get_base_ptr());
                    }
                }
                comm::ibgda::put_nbi_warp_batch_rows(
                    req_rptr, req_lptr,
                    kHidden * sizeof(nv_bfloat16), row_active,
                    static_cast<int>(shared_publish_dst_rank),
                    scatter_qp_id, static_cast<int>(lane_idx));
            }
        }

        const bool expert_complete =
            not has_block or cursor + 1 == expert_blocks[expert];
        if (expert_complete) {
            __threadfence_system();
            __syncwarp();
            if (lane_idx == 0 and
                expert_pair_tokens[expert][shared_publish_dst_rank] != 0) {
                const uint32_t global_expert_idx =
                    sym_buffer.rank_idx * kNumExpertsPerRank + expert;
                const auto ready_ptr =
                    workspace.get_combine_ready_epoch_ptr(global_expert_idx);
                if (shared_publish_is_inter) {
                    comm::ibgda::put_inline_with_credit<uint64_t>(
                        ready_ptr, launch_epoch,
                        static_cast<int>(shared_publish_dst_rank),
                        static_cast<int>(expert));
                } else {
                    ptx::st_relaxed_sys(
                        sym_buffer.map(
                            ready_ptr, shared_publish_dst_rank),
                        launch_epoch);
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            if (has_block)
                shared_publish_cursor[expert] = cursor + 1;
            if (expert_complete) {
                ptx::st_release_sys(
                    workspace.get_combine_publish_done_epoch_ptr(expert),
                    launch_epoch);
                shared_publish_pending &= ~(1u << expert);
            }
        }
        __syncthreads();
    }
#endif

#ifdef DG_MEGA_MOE_FP4_SIDECAR_EXPERT_CENTRIC
    // A small, tunable group of warps owns one expert.  Every group warp reads
    // each row's metadata once and serves a disjoint subset of remote peers.
    // Two groups reduce metadata traffic 4x while retaining four-way peer
    // submission parallelism; four groups trade another metadata copy for a
    // shorter two-peer serial chain.
    constexpr uint32_t kPeerGroups =
        DG_MEGA_MOE_FP4_SIDECAR_EXPERT_PEER_GROUPS;
    const uint32_t expert_warp_idx = warp_idx / kPeerGroups;
    const uint32_t peer_group_idx = warp_idx % kPeerGroups;
    const uint32_t expert = blockIdx.x + expert_warp_idx * gridDim.x;
    if (expert >= kNumExpertsPerRank)
        return;

    constexpr uint32_t kLocalRanks = DG_MEGA_MOE_NVL_PEERS;
    DG_STATIC_ASSERT(kNumRanks == 2 * kLocalRanks,
                     "Expert-centric publication requires two NVLink domains");
    DG_STATIC_ASSERT(
        kLocalRanks % kPeerGroups == 0,
        "Peer groups must evenly divide the remote NVLink domain");
    constexpr uint32_t kPeersPerGroup = kLocalRanks / kPeerGroups;
    const uint32_t local_rank_base =
        sym_buffer.rank_idx / kLocalRanks * kLocalRanks;
    const uint32_t remote_rank_base = local_rank_base ^ kLocalRanks;
    const uint32_t group_remote_begin =
        remote_rank_base + peer_group_idx * kPeersPerGroup;
    const uint32_t expert_total = expert_total_tokens[expert];
    const uint32_t num_blocks = expert_blocks[expert];
    const int scatter_qp_id = static_cast<int>(expert);

    for (uint32_t cursor = 0; cursor < num_blocks;) {
        const uint32_t pool_block_idx = expert_pool[expert] + cursor;
        const auto arrival_ptr = combine_full_row_arrival_buffer
            .get_data_buffer(pool_block_idx)
            .get_base_ptr<uint32_t>();
        uint32_t arrival = 0;
        if (lane_idx == 0)
            arrival = ptx::ld_acq(arrival_ptr);
        arrival = __shfl_sync(0xffffffffu, arrival, 0);
        if ((arrival & kCombineFullRowPublishReadyBit) == 0)
            continue;

        const uint32_t m_idx = pool_block_idx * BLOCK_M;
        const uint32_t valid_m = cute::min(
            expert_total - cursor * BLOCK_M, BLOCK_M);
        for (uint32_t row_base = 0; row_base < valid_m;
             row_base += kPublishRowsPerBatch) {
            const uint32_t row = row_base + lane_idx;
            bool valid_row = row < valid_m;
            layout::TokenSrcMetadata src_metadata{};
            uint64_t req_rptr = 0;
            uint64_t req_lptr = 0;
            if (valid_row) {
                src_metadata =
                    *workspace.get_token_src_metadata_ptr(m_idx + row);
                const bool row_is_remote =
                    src_metadata.rank_idx >= remote_rank_base and
                    src_metadata.rank_idx < remote_rank_base + kLocalRanks;
                if (row_is_remote) {
                    const auto staging_row = combine_full_row_staging_buffer
                        .get_data_buffer(m_idx + row);
                    const auto dst_row = combine_token_buffer
                        .get_rank_buffer(src_metadata.topk_idx)
                        .get_data_buffer(src_metadata.token_idx);
                    req_rptr = reinterpret_cast<uint64_t>(
                        dst_row.get_base_ptr());
                    req_lptr = reinterpret_cast<uint64_t>(
                        staging_row.get_base_ptr());
                }
            }

            #pragma unroll
            for (uint32_t remote = 0; remote < kPeersPerGroup; ++ remote) {
                const uint32_t dst_rank_idx = group_remote_begin + remote;
                const bool row_active = valid_row and
                    src_metadata.rank_idx == dst_rank_idx;
                comm::ibgda::put_nbi_warp_batch_rows(
                    req_rptr, req_lptr,
                    kHidden * sizeof(nv_bfloat16), row_active,
                    static_cast<int>(dst_rank_idx), scatter_qp_id,
                    static_cast<int>(lane_idx));
            }
        }
        ++ cursor;
        __syncwarp();
    }

    __threadfence_system();
    __syncwarp();
    if (lane_idx == 0) {
        const uint32_t old_groups = atomicAdd(
            &expert_publish_group_done[expert], 1u);
        DG_TRAP_ONLY_DEVICE_ASSERT(old_groups < kPeerGroups);
        if (old_groups + 1 == kPeerGroups) {
            const uint32_t global_expert_idx =
                sym_buffer.rank_idx * kNumExpertsPerRank + expert;
            const auto ready_ptr = workspace.get_combine_ready_epoch_ptr(
                global_expert_idx);
            #pragma unroll
            for (uint32_t dst_rank_idx = 0;
                 dst_rank_idx < kNumRanks; ++ dst_rank_idx) {
                if (expert_pair_tokens[expert][dst_rank_idx] == 0)
                    continue;
                const bool is_inter =
                    dst_rank_idx / kLocalRanks !=
                    sym_buffer.rank_idx / kLocalRanks;
                if (is_inter) {
                    comm::ibgda::put_inline_with_credit<uint64_t>(
                        ready_ptr, launch_epoch,
                        static_cast<int>(dst_rank_idx), scatter_qp_id);
                } else {
                    ptx::st_relaxed_sys(
                        sym_buffer.map(ready_ptr, dst_rank_idx),
                        launch_epoch);
                }
            }
            ptx::st_release_sys(
                workspace.get_combine_publish_done_epoch_ptr(expert),
                launch_epoch);
        }
    }
    return;
#endif

    const bool chain_is_inter =
        dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
    uint32_t chain_pending = 0;
    #pragma unroll
    for (uint32_t expert = 0; expert < kNumExpertsPerRank; ++ expert) {
        if (expert % gridDim.x == blockIdx.x)
            chain_pending |= 1u << expert;
    }
#ifdef DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS
    uint32_t publisher_backoff_ns =
        DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS;
    uint32_t publisher_idle_passes = 0;
#endif
    const uint64_t publish_wait_start = clock64();
    while (chain_pending != 0) {
        bool made_progress = false;
        DG_TRAP_ONLY_DEVICE_ASSERT(
            clock64() - publish_wait_start < kTimeoutCycles);

        #pragma unroll
        for (uint32_t expert = 0; expert < kNumExpertsPerRank; ++ expert) {
            if ((chain_pending & (1u << expert)) == 0)
                continue;
            const uint32_t expert_total = expert_total_tokens[expert];
            const uint32_t num_blocks = expert_blocks[expert];
            uint32_t cursor = chain_cursor[warp_idx][expert];
            const int scatter_qp_id = static_cast<int>(expert);

#ifdef DG_MEGA_MOE_FP4_SIDECAR_AGGREGATE_LOCAL
            if (is_local_aggregator) {
                const bool local_lane = lane_idx < kLocalRanks;
                const uint32_t local_dst_rank =
                    local_rank_base + lane_idx;
                const uint32_t local_pair_tokens = local_lane ?
                    expert_pair_tokens[expert][local_dst_rank] : 0;
                const uint32_t local_active_mask = __ballot_sync(
                    0xffffffffu, local_pair_tokens != 0);

                if (local_active_mask != 0) {
                    while (cursor < num_blocks) {
                        const uint32_t pool_block_idx =
                            expert_pool[expert] + cursor;
                        const auto arrival_ptr =
                            combine_full_row_arrival_buffer
                                .get_data_buffer(pool_block_idx)
                                .get_base_ptr<uint32_t>();
                        uint32_t arrival = 0;
                        if (lane_idx == 0)
                            arrival = ptx::ld_acq(arrival_ptr);
                        arrival = __shfl_sync(
                            0xffffffffu, arrival, 0);
                        if ((arrival &
                             kCombineFullRowPublishReadyBit) == 0)
                            break;

                        made_progress = true;
                        ++ cursor;
                        if (lane_idx == 0)
                            chain_cursor[warp_idx][expert] = cursor;
                        __syncwarp();
                    }
                    if (cursor != num_blocks)
                        continue;

                    __threadfence_system();
                    __syncwarp();
                    if (local_lane and local_pair_tokens != 0) {
                        const uint32_t global_expert_idx =
                            sym_buffer.rank_idx *
                                kNumExpertsPerRank + expert;
                        const auto ready_ptr =
                            workspace.get_combine_ready_epoch_ptr(
                                global_expert_idx);
                        ptx::st_relaxed_sys(
                            sym_buffer.map(ready_ptr, local_dst_rank),
                            launch_epoch);
                    }
                    __syncwarp();
                }

                if (local_lane) {
                    const auto old_pair_done =
                        ptx::atomic_add_acq_rel_sys(
                            workspace
                                .get_combine_publish_pair_done_count_ptr(
                                    expert),
                            1);
                    DG_TRAP_ONLY_DEVICE_ASSERT(
                        old_pair_done < kNumRanks);
                    if (old_pair_done + 1 == kNumRanks) {
                        ptx::st_release_sys(
                            workspace
                                .get_combine_publish_done_epoch_ptr(
                                    expert),
                            launch_epoch);
                    }
                }
                __syncwarp();
                chain_pending &= ~(1u << expert);
                made_progress = true;
                continue;
            }
#endif

            const uint32_t pair_tokens =
                expert_pair_tokens[expert][dst_rank_idx];
            if (pair_tokens != 0) {
                while (cursor < num_blocks) {
                    const uint32_t pool_block_idx =
                        expert_pool[expert] + cursor;
                    const auto arrival_ptr = combine_full_row_arrival_buffer
                        .get_data_buffer(pool_block_idx)
                        .get_base_ptr<uint32_t>();
                    uint32_t arrival = 0;
                    if (lane_idx == 0)
                        arrival = ptx::ld_acq(arrival_ptr);
                    arrival = __shfl_sync(0xffffffffu, arrival, 0);
                    if ((arrival & kCombineFullRowPublishReadyBit) == 0)
                        break;

                    made_progress = true;
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t valid_m = cute::min(
                        expert_total - cursor * BLOCK_M, BLOCK_M);
                    // Same-node combine rows are written directly by the
                    // fused kernel.  Their publisher warp only needs to wait
                    // until every expert block is complete before releasing
                    // the destination-ready epoch; it must not reload and
                    // classify every row's metadata.
                    if (chain_is_inter) {
                        for (uint32_t row_base = 0; row_base < valid_m;
                             row_base += kPublishRowsPerBatch) {
                            const uint32_t row = row_base + lane_idx;
                            bool row_active = false;
                            uint64_t req_rptr = 0;
                            uint64_t req_lptr = 0;
                            if (row < valid_m) {
                                const auto src_metadata =
                                    *workspace.get_token_src_metadata_ptr(
                                        m_idx + row);
                                row_active =
                                    src_metadata.rank_idx == dst_rank_idx;
                                if (row_active) {
                                    const auto staging_row =
                                        combine_full_row_staging_buffer
                                            .get_data_buffer(m_idx + row);
                                    const auto dst_row = combine_token_buffer
                                        .get_rank_buffer(src_metadata.topk_idx)
                                        .get_data_buffer(src_metadata.token_idx);
                                    req_rptr = reinterpret_cast<uint64_t>(
                                        dst_row.get_base_ptr());
                                    req_lptr = reinterpret_cast<uint64_t>(
                                        staging_row.get_base_ptr());
                                }
                            }
                            comm::ibgda::put_nbi_warp_batch_rows(
                                req_rptr, req_lptr,
                                kHidden * sizeof(nv_bfloat16), row_active,
                                static_cast<int>(dst_rank_idx),
                                scatter_qp_id,
                                static_cast<int>(lane_idx));
                        }
                    }
                    ++ cursor;
                    if (lane_idx == 0)
                        chain_cursor[warp_idx][expert] = cursor;
                    __syncwarp();
                }
                if (cursor != num_blocks)
                    continue;

                __threadfence_system();
                __syncwarp();
                if (lane_idx == 0) {
                    const uint32_t global_expert_idx =
                        sym_buffer.rank_idx * kNumExpertsPerRank + expert;
                    const auto ready_ptr =
                        workspace.get_combine_ready_epoch_ptr(
                            global_expert_idx);
                    if (chain_is_inter) {
                        comm::ibgda::put_inline_with_credit<uint64_t>(
                            ready_ptr, launch_epoch,
                            static_cast<int>(dst_rank_idx), scatter_qp_id);
                    } else {
                        ptx::st_relaxed_sys(
                            sym_buffer.map(ready_ptr, dst_rank_idx),
                            launch_epoch);
                    }
                }
                __syncwarp();
            }

            if (lane_idx == 0) {
                const auto old_pair_done = ptx::atomic_add_acq_rel_sys(
                    workspace.get_combine_publish_pair_done_count_ptr(
                        expert),
                    1);
                DG_TRAP_ONLY_DEVICE_ASSERT(old_pair_done < kNumRanks);
                if (old_pair_done + 1 == kNumRanks) {
                    ptx::st_release_sys(
                        workspace.get_combine_publish_done_epoch_ptr(expert),
                        launch_epoch);
                }
            }
            __syncwarp();
            chain_pending &= ~(1u << expert);
            made_progress = true;
        }

#ifdef DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS
        if (made_progress) {
            publisher_backoff_ns =
                DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS;
            publisher_idle_passes = 0;
        } else if (chain_pending != 0) {
            publisher_idle_passes +=
                publisher_idle_passes != 0xffffffffu;
            uint32_t max_backoff_ns =
                DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MAX_NS;
            if constexpr (
                DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_FEW_PENDING_CHAINS > 0) {
                if (__popc(chain_pending) <=
                    DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_FEW_PENDING_CHAINS)
                    max_backoff_ns =
                        DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_FEW_PENDING_MAX_NS;
                else if constexpr (
                    DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD > 0) {
                    if (publisher_idle_passes >=
                        DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD)
                        max_backoff_ns =
                            DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_MAX_NS;
                }
            } else if constexpr (
                DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD > 0) {
                if (publisher_idle_passes >=
                    DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD)
                    max_backoff_ns =
                        DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_MAX_NS;
            }
            publisher_backoff_ns = cute::min(
                publisher_backoff_ns, max_backoff_ns);
            __nanosleep(publisher_backoff_ns);
            publisher_backoff_ns = cute::min(
                publisher_backoff_ns * 2u, max_backoff_ns);
        }
#endif
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_TRAP_ONLY_DEVICE_ASSERT(false and
            "Sidecar publisher requires the inter-node specialization");
#endif
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_TRAP_ONLY_DEVICE_ASSERT(false and
            "This kernel only supports sm_90");
#endif
}

// ============================================================================
// SM90 (Hopper) FP8 x FP4 MegaMoE - software-dequant path.
// ----------------------------------------------------------------------------
// Variant of `sm90_fp8_mega_moe_impl` for DSV4-style packed FP4 expert weights.
// The dispatch / scheduler / SwiGLU / combine machinery is identical to the
// FP8 implementation; the only differences are confined to:
//
//   1. Weight TMA load:  shape changes from (LOAD_BLOCK_N, BLOCK_K) of e4m3
//      to (LOAD_BLOCK_N, BLOCK_K/2) of packed int8 (each byte = 2 nibbles).
//   2. SFB:              loaded as UE8M0 packed int32 (per-32 K granularity)
//      via `cp.async`, since TMA does not natively stride FP4 layouts.
//   3. Mainloop decode:  the host path uses the UE8M0 LUT decoder to dequant
//      the packed FP4 weight tile into an E4M3 shared-memory tile. SS-mode WGMMA
//      then consumes that tile exactly like the FP8 path, preserving the existing
//      per-token SwiGLU amax / quantize epilogue.
// ============================================================================

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumL1RingTokens,
    uint32_t kNumL1SFStorageTokens,
    uint32_t kNumL2RingTokens,
    uint32_t kNumL2SFStorageTokens,
    uint32_t kNumCombineStageTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kUseWideLoadDecode        = false,  // Read one K-group's packed FP4 words as uint4
    bool kMathWGParticipatesInFP4Decode = false,
    uint32_t kNumMathWGDecodeWarps = 0,
    uint32_t kFirstFP4DecodeAssistWarp = 0,  // Skip early non-epilogue warps as decode helpers
    bool kEarlyBDecode            = false,  // Overlap assist decode with A/SFA TMA
    bool kDecodeDoneMBarrier      = false,  // One-way decode-done mbarrier instead of rendezvous sync
    bool kL2ArrivalCounter        = false,  // Count ready L1 output slices instead of bitmask + CTA sync
    bool kFP4SSNSplit             = false,  // Split SS N=128 WGMMA into 2x N=64 to reduce accum pressure
    bool kFP4SwapAB               = false,  // weight@M, token@N for small-batch padding relief
    bool kFP4SwapABFastAmax       = false,  // Reuse L1 swapAB store lanes to publish per-token partial amax
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_fp4_mega_moe_impl(void* y,
                           int* cumulative_local_expert_recv_stats,
                           const uint32_t num_tokens,
                           const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                           const uint32_t* __restrict__ l1_weights_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                           const uint32_t* __restrict__ l2_weights_sf,
                           void* cpu_proxy_buffer,
                           const void* cpu_proxy_dev_comms_ptr,
                           const void* cpu_proxy_windows_ptr,
                           const uint64_t cpu_proxy_combine_offset,
                           const uint64_t cpu_proxy_staging_offset,
                           const uint64_t cpu_proxy_signal_epoch_offset,
                           const uint64_t cpu_proxy_completion_request_offset,
                           const uint64_t cpu_proxy_combine_slot_bytes,
                           const uint64_t cpu_proxy_staging_slot_bytes,
                           const uint32_t cpu_proxy_num_slots) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

#ifdef DG_MEGA_MOE_INTERNODE
    constexpr bool kDispatchExpertReady = true;
    constexpr bool kCombineFullRow = true;
    constexpr bool kCombineExpertReady = true;
#ifdef DG_MEGA_MOE_NUM_METADATA_SMS
    constexpr uint32_t kNumMetadataSMs =
        DG_MEGA_MOE_NUM_METADATA_SMS;
#else
    constexpr uint32_t kNumMetadataSMs = kNumSMs;
#endif

#else
    constexpr bool kDispatchExpertReady = false;
    constexpr bool kCombineFullRow = false;
    constexpr bool kCombineExpertReady = false;
    constexpr uint32_t kNumMetadataSMs = kNumSMs;
#endif

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(kNumDispatchThreads == 64 or kNumDispatchThreads == 128,
                     "Dispatch supports 2 or 4 warps");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads >= 128 and kNumNonEpilogueThreads % 64 == 0,
                     "Invalid number of GEMM TMA/decode-assist warps");
    DG_STATIC_ASSERT((kNumDispatchThreads + kNumNonEpilogueThreads) % 128 == 0,
                     "Math warps must start on a warpgroup boundary");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of math/epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(not kCombineExpertReady or kCombineFullRow,
                     "Per-expert ready requires full-row combine staging");
    DG_STATIC_ASSERT(not kCombineExpertReady or kNumRanks <= 64,
                     "Per-expert destination mask supports at most 64 ranks");
    DG_STATIC_ASSERT(kNumMetadataSMs >= 1 and kNumMetadataSMs <= kNumSMs,
                     "Invalid number of metadata producer CTAs");
    DG_STATIC_ASSERT(kNumCombineStageTokens <= kNumMaxPoolTokens,
                     "Combine staging cannot exceed the logical full pool");
    DG_STATIC_ASSERT(not kCombineExpertReady or
                         kNumCombineStageTokens >=
                             kNumRanks * kNumMaxTokensPerRank,
                     "Combine ring must fit one maximum-fan-in expert");
#ifdef DG_MEGA_MOE_INTERNODE
    DG_STATIC_ASSERT(kDispatchExpertReady,
                     "FP4 inter-node requires expert-ready dispatch");
#if !defined(DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED) && \
    !defined(DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3)
#error "Inter-node SM90 FP4 MegaMoE requires packed or dense-V3 metadata"
#endif
#endif
#if defined(DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED) && \
    defined(DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3)
#error "Packed and dense-V3 metadata modes are mutually exclusive"
#endif
    // Inter-node publication uses the loader warp freed by the merged A/B
    // loader; there is no runtime protocol or publisher-owner A/B selector.
    DG_STATIC_ASSERT(BLOCK_M % 64 == 0, "BLOCK_M must be a multiple of WGMMA::M (64)");
    DG_STATIC_ASSERT(BLOCK_N % 8 == 0, "BLOCK_N must be compatible with SM90 FP8 WGMMA shapes");
    DG_STATIC_ASSERT(BLOCK_K == 128, "BLOCK_K is fixed to 128 (per-128 SF)");
    DG_STATIC_ASSERT(kNumMathWGDecodeWarps <= kNumEpilogueWarps,
                     "Math decode warps cannot exceed epilogue warps");
    DG_STATIC_ASSERT(kMathWGParticipatesInFP4Decode or kNumMathWGDecodeWarps == 0,
                     "Math decode warp count requires math WG decode participation");
    DG_STATIC_ASSERT(kFirstFP4DecodeAssistWarp <= kNumMMANonEpilogueWarps,
                     "First FP4 decode assist warp is out of range");
    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
    DG_TRAP_ONLY_DEVICE_ASSERT(cpu_proxy_buffer != nullptr);
    DG_TRAP_ONLY_DEVICE_ASSERT(cpu_proxy_dev_comms_ptr != nullptr);
    DG_TRAP_ONLY_DEVICE_ASSERT(cpu_proxy_windows_ptr != nullptr);
    const auto cpu_proxy_dev_comms =
        reinterpret_cast<const ncclDevComm*>(cpu_proxy_dev_comms_ptr);
    const auto cpu_proxy_windows =
        reinterpret_cast<const ncclWindow_t*>(cpu_proxy_windows_ptr);
    constexpr uint32_t kCPUProxyPeersPerNode = 8;
    const int cpu_proxy_peer =
        sym_buffer.rank_idx / kCPUProxyPeersPerNode == 0 ? 1 : 0;
#else
    (void)cpu_proxy_buffer;
    (void)cpu_proxy_dev_comms_ptr;
    (void)cpu_proxy_windows_ptr;
    (void)cpu_proxy_combine_offset;
    (void)cpu_proxy_staging_offset;
    (void)cpu_proxy_signal_epoch_offset;
    (void)cpu_proxy_completion_request_offset;
    (void)cpu_proxy_combine_slot_bytes;
    (void)cpu_proxy_staging_slot_bytes;
    (void)cpu_proxy_num_slots;
#endif

    // Prefetch all TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing (mirror SM100 layout, except SF
    // for L2 activations uses per-64 K granularity)
    // =====================================================================
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, 128u);
    constexpr uint32_t kNumL1RingBlocks = kNumL1RingTokens / BLOCK_M;
    constexpr uint32_t kNumL2RingBlocks = kNumL2RingTokens / BLOCK_M;
    constexpr bool kL1RingStorageCompact =
        kNumL1RingTokens < kNumMaxPoolTokens;
    constexpr bool kL2RingStorageCompact =
        kNumL2RingTokens < kNumMaxPoolTokens;
#ifdef DG_MEGA_MOE_L1_RING_ACTIVE
    constexpr bool kL1RingEnabled = kL1RingStorageCompact;
#else
    constexpr bool kL1RingEnabled = false;
#endif
#ifdef DG_MEGA_MOE_L2_RING_ACTIVE
    constexpr bool kL2RingEnabled = kL2RingStorageCompact;
#else
    constexpr bool kL2RingEnabled = false;
#endif
    constexpr uint32_t kNumL2StorageTokens =
        kNumL2RingTokens + (kL2RingStorageCompact ? 128u : 0u);
    DG_STATIC_ASSERT(kNumMaxPoolTokens % BLOCK_M == 0,
                     "Invalid SM90 FP4 MegaMoE pool size");
    DG_STATIC_ASSERT(kNumL1RingTokens % BLOCK_M == 0 and
                         kNumL2RingTokens % BLOCK_M == 0,
                     "Invalid SM90 FP4 compute ring capacity");
    DG_STATIC_ASSERT(kNumL1SFStorageTokens >=
                         kNumL1RingBlocks * SF_BLOCK_M,
                     "Invalid SM90 FP4 L1 SF ring capacity");
    DG_STATIC_ASSERT(kNumL2SFStorageTokens >=
                         kNumL2RingBlocks * SF_BLOCK_M,
                     "Invalid SM90 FP4 L2 SF ring capacity");

    const auto get_l1_ring_block_idx = [](const uint32_t& pool_block_idx) {
        if constexpr (kL1RingEnabled)
            return pool_block_idx % kNumL1RingBlocks;
        return pool_block_idx;
    };
    const auto get_l2_ring_block_idx = [](const uint32_t& pool_block_idx) {
        if constexpr (kL2RingEnabled)
            return pool_block_idx % kNumL2RingBlocks;
        return pool_block_idx;
    };
    const auto get_l1_ring_token_idx = [&](const uint32_t& pool_token_idx) {
        const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;
        return get_l1_ring_block_idx(pool_block_idx) * BLOCK_M +
            pool_token_idx % BLOCK_M;
    };
    const auto get_l1_ring_wave_idx = [](const uint32_t& pool_block_idx) {
        if constexpr (kL1RingEnabled)
            return pool_block_idx / kNumL1RingBlocks;
        return 0u;
    };
    const auto get_l2_ring_wave_idx = [](const uint32_t& pool_block_idx) {
        if constexpr (kL2RingEnabled)
            return pool_block_idx / kNumL2RingBlocks;
        return 0u;
    };

    const auto workspace = layout::SM90Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = SM90FP8FP4MegaMoEData(kHidden);
    constexpr auto bf16_token_layout             = SM90FP8FP4MegaMoEData(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = SM90FP8FP4MegaMoEData(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = SM90FP8FP4MegaMoEData(kHidden / 32);
    // L2 activation SF is per-64 for BLOCK_N=128 and per-32 for BLOCK_N=64.
    constexpr uint32_t kL2ActsSFGranK = BLOCK_N == 64 ? 32 : 64;
    constexpr auto fp8_intermediate_sf_layout =
        SM90FP8FP4MegaMoEData(kIntermediateHidden * sizeof(float) / kL2ActsSFGranK);
    constexpr auto input_topk_idx_layout         = SM90FP8FP4MegaMoEData(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = SM90FP8FP4MegaMoEData(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = SM90FP8FP4MegaMoEData(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = SM90FP8FP4MegaMoEBuffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = SM90FP8FP4MegaMoEBuffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = SM90FP8FP4MegaMoEBuffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = SM90FP8FP4MegaMoEBuffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = SM90FP8FP4MegaMoEBuffer(fp8_token_layout, 1, kNumL1RingTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = SM90FP8FP4MegaMoEBuffer(fp8_sf_layout, 1, kNumL1SFStorageTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = SM90FP8FP4MegaMoEBuffer(l1_topk_weights_layout, 1, kNumL1RingTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = SM90FP8FP4MegaMoEBuffer(fp8_intermediate_token_layout, 1, kNumL2StorageTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = SM90FP8FP4MegaMoEBuffer(fp8_intermediate_sf_layout, 1, kNumL2SFStorageTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = SM90FP8FP4MegaMoEBuffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // Inter-node SF/weight staging area (allocated by the shared host sizing
    // whether or not this build uses it).  Each pool token owns a separate,
    // cache-line-aligned row; it never aliases the later combine destination.
    constexpr auto dispatch_staging_layout = SM90FP8FP4MegaMoEData(
        math::constexpr_align<uint32_t>(kHidden / 32 + sizeof(float), 128u), false);
    const auto dispatch_staging_buffer = SM90FP8FP4MegaMoEBuffer(
        dispatch_staging_layout, 1, kNumMaxPoolTokens,
        combine_token_buffer.get_end_ptr());

    // Full-row combine keeps one registered BF16 row per pool token.  A
    // separate per-pool-block arrival counter lets the last L2 N-block CTA
    // hand the completed rows to the publisher warp.
    constexpr uint32_t kNumMaxPoolBlocks =
        kNumMaxPoolTokens / layout::kMinCandidateBlockM;
    constexpr uint32_t kCombineFullRowPublishReadyBit = 1u << 31;
    constexpr uint32_t kPublishRowsPerMaskWord = 32;
    constexpr uint32_t kPublishRowMaskWords =
        BLOCK_M / kPublishRowsPerMaskWord;
    // Keep the per-destination stride independent of the selected BLOCK_M so
    // all FP4 specializations agree with the shared host-side buffer layout.
    constexpr uint32_t kPublishRowMaskStorageWords = 4;
    DG_STATIC_ASSERT(kPublishRowMaskWords <=
                         kPublishRowMaskStorageWords,
                     "Publish row mask does not cover BLOCK_M");
    constexpr auto combine_full_row_arrival_layout = SM90FP8FP4MegaMoEData(
        kCombineFullRow ? sizeof(uint32_t) : 0u, false);
    const auto combine_full_row_arrival_buffer = SM90FP8FP4MegaMoEBuffer(
        combine_full_row_arrival_layout, 1, kNumMaxPoolBlocks,
        dispatch_staging_buffer.get_end_ptr());
    constexpr auto combine_full_row_ready_timestamp_layout =
        SM90FP8FP4MegaMoEData(
            kCombineFullRow ? sizeof(uint64_t) : 0u, false);
    const auto combine_full_row_ready_timestamp_buffer =
        SM90FP8FP4MegaMoEBuffer(
            combine_full_row_ready_timestamp_layout, 1,
            kNumMaxPoolBlocks,
            combine_full_row_arrival_buffer.get_end_ptr());
    constexpr auto combine_publish_row_mask_layout =
        SM90FP8FP4MegaMoEData(
            kCombineFullRow ?
                kNumRanks * kPublishRowMaskStorageWords *
                    sizeof(uint32_t) :
                0u,
            false);
    const auto combine_publish_row_mask_buffer = SM90FP8FP4MegaMoEBuffer(
        combine_publish_row_mask_layout, 1, kNumMaxPoolBlocks,
        combine_full_row_ready_timestamp_buffer.get_end_ptr());
    const auto get_combine_publish_row_mask_ptr =
        [&](const uint32_t& pool_block_idx, const uint32_t& dst_rank_idx,
            const uint32_t& mask_word_idx = 0) {
            return combine_publish_row_mask_buffer
                       .get_data_buffer(pool_block_idx)
                       .get_base_ptr<uint32_t>() +
                dst_rank_idx * kPublishRowMaskStorageWords + mask_word_idx;
        };
    const auto combine_full_row_staging_base = reinterpret_cast<void*>(
        kCombineFullRow ? math::align(
            reinterpret_cast<uint64_t>(combine_publish_row_mask_buffer.get_end_ptr()),
            static_cast<uint64_t>(128)) :
            reinterpret_cast<uint64_t>(dispatch_staging_buffer.get_end_ptr()));
    constexpr auto combine_full_row_staging_layout = SM90FP8FP4MegaMoEData(
        kCombineFullRow ? kHidden * sizeof(nv_bfloat16) : 0u);
    const auto combine_full_row_staging_buffer = SM90FP8FP4MegaMoEBuffer(
        combine_full_row_staging_layout, 1, kNumCombineStageTokens,
        combine_full_row_staging_base);

#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
    constexpr uint64_t kCPUProxyRowBytes =
        kHidden * sizeof(nv_bfloat16);
    DG_TRAP_ONLY_DEVICE_ASSERT(cpu_proxy_num_slots > 0);
    uint32_t cpu_proxy_slot_idx = 0;
    const auto get_cpu_proxy_combine_row_offset =
        [&](const uint32_t& topk_slot_idx, const uint32_t& token_idx) {
            return cpu_proxy_combine_offset +
                static_cast<uint64_t>(cpu_proxy_slot_idx) *
                    cpu_proxy_combine_slot_bytes +
                (static_cast<uint64_t>(topk_slot_idx) *
                     kNumMaxTokensPerRank + token_idx) *
                    kCPUProxyRowBytes;
        };
    const auto get_cpu_proxy_staging_row_offset =
        [&](const uint32_t& staging_row_idx) {
            return cpu_proxy_staging_offset +
                static_cast<uint64_t>(cpu_proxy_slot_idx) *
                    cpu_proxy_staging_slot_bytes +
                static_cast<uint64_t>(staging_row_idx) *
                    kCPUProxyRowBytes;
        };
    const auto get_cpu_proxy_combine_row_ptr =
        [&](const uint32_t& topk_slot_idx, const uint32_t& token_idx) {
            return math::advance_ptr<uint8_t>(
                cpu_proxy_buffer,
                get_cpu_proxy_combine_row_offset(
                    topk_slot_idx, token_idx));
        };
    const auto get_cpu_proxy_staging_row_ptr =
        [&](const uint32_t& staging_row_idx) {
            return math::advance_ptr<uint8_t>(
                cpu_proxy_buffer,
                get_cpu_proxy_staging_row_offset(staging_row_idx));
        };
    const auto get_cpu_proxy_signal_epoch_ptr =
        [&](const uint32_t& local_expert_idx,
            const uint32_t& dst_rank_idx) {
            return math::advance_ptr<uint64_t>(
                cpu_proxy_buffer,
                cpu_proxy_signal_epoch_offset +
                (static_cast<uint64_t>(local_expert_idx) * kNumRanks +
                 dst_rank_idx) * sizeof(uint64_t));
        };
    const auto get_cpu_proxy_completion_request_ptr =
        [&](const uint32_t& slot_idx,
            const uint32_t& remote_local_rank) {
            return math::advance_ptr<ncclGinRequest_t>(
                cpu_proxy_buffer,
                cpu_proxy_completion_request_offset) +
                static_cast<uint64_t>(slot_idx) *
                    kCPUProxyPeersPerNode + remote_local_rank;
        };
#endif

    constexpr bool kCombineStageRing =
        kCombineFullRow and kCombineExpertReady and
        kNumCombineStageTokens < kNumMaxPoolTokens;
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY_ASYNC_CREDIT
    DG_STATIC_ASSERT(
        not kCombineStageRing,
        "Async CPU proxy credit requires full-pool combine staging");
#endif
    const auto combine_stage_ring = SM90CombineStageRing<
        kCombineStageRing, kNumCombineStageTokens,
        kNumExpertsPerRank, kNumRanks>(workspace);

    constexpr uint32_t kPhaseProfileMaxSMs =
        layout::kSM90MegaMoEProfileMaxSMs;
    constexpr uint32_t kPhaseProfileSlots =
        layout::kSM90MegaMoEProfileSlots;
    const auto phase_profile_buffer = SM90FP8FP4MegaMoEBuffer(
        SM90FP8FP4MegaMoEData(
            kPhaseProfileSlots * sizeof(uint64_t), false),
        1, kPhaseProfileMaxSMs,
        combine_full_row_staging_buffer.get_end_ptr());

#ifdef DG_MEGA_MOE_PHASE_PROFILE
    DG_STATIC_ASSERT(kNumSMs <= kPhaseProfileMaxSMs,
                     "Too many SMs for FP4 phase profiler");
    enum ProfileSlot : uint32_t {
        kProfileMetadata = 0,
        kProfileDispatchBarrier = 1,
        kProfileDispatchPull = 2,
        kProfileRemoteRead = 3,
        kProfileCleanupBarrier = 4,
        kProfileL1 = 5,
        kProfileL2 = 6,
        kProfileScatter = 7,
        kProfileCombineBarrier = 9,
        kProfileCombineReduce = 10,
        kProfileTotal = 11,
        kProfileRemoteReadCount = 12,
        kProfileL1BlockCount = 13,
        kProfileL2BlockCount = 14,
        kProfileStartClock = 16,
        kProfileCombineReadyWait = 17,
        kProfileDecodeWait = 19,
        kProfileAWait = 20,
        kProfileLoaderPoolWait = 21,
        kProfilePublishTotalBlocks = 22,
        kProfilePublishEmptyDstBlocks = 23,
        kProfilePublishMetadataLoads = 24,
        kProfilePublishRDMABatches = 25,
        kProfilePublishOuterLoops = 26,
        kProfilePublishReadyChecks = 27,
        kProfilePublishEmptyPasses = 28,
        kProfilePublishSleepStats = 29,
        kProfilePublishReadyObserveStats = 30,
        kProfileEntryGlobaltimer = 31,
        kProfileCountsSentGlobaltimer = 32,
        kProfileCountsReadyGlobaltimer = 33,
        kProfilePublishMaxRequestedSleepNs = 34,
    };
    auto phase_profile = phase_profile_buffer.get_data_buffer(sm_idx)
        .get_base_ptr<unsigned long long>();
#endif

#ifdef DG_MEGA_MOE_INTERNODE
    if (sm_idx == 0 and thread_idx == 0)
        DG_DEVICE_ASSERT(
            comm::ibgda::ibgda_get_state()->num_rc_per_pe >=
            kNumExpertsPerRank + 1);
#endif

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    // The WGMMA still consumes E4M3 on both operands. We reuse the SS-mode
    // selector and dequant the packed FP4 weight into a second SMEM tile of
    // E4M3 right before issuing each k-block's WGMMA group; see decode_b_tile.
    using b_dtype_t        = cutlass::float_e4m3_t;
    // Storage type for the packed FP4 weight tile in SMEM/global. Each byte
    // packs 2 nibbles (low nibble = lower-K element, high nibble = upper-K),
    // matching DSV4's TMA-friendly layout.
    using b_packed_dtype_t = int8_t;
    // FP4 SS split-N infrastructure: when BLOCK_M=64 and BLOCK_N is a
    // multiple of 128, the host heuristics may request
    // `kNumEpilogueWarpgroups == BLOCK_N / 128 > 1` math warpgroups. In that
    // mode every WG shares the same BLOCK_M rows and partitions the N
    // columns, so each WG owns WG_BLOCK_N = BLOCK_N / num_wg columns. The
    // packed-B / SFB / decoded-B SMEM tiles still cover the full LOAD_BLOCK_N
    // because FP4 decode is shared across WGs (see comment on smem_b below);
    // split-N only manifests in WGMMA descriptors and the L1/L2 epilogue.
    constexpr bool kSplitNWarpgroups =
        BLOCK_M == 64 and
        kNumEpilogueWarpgroups > 1 and
        BLOCK_N % kNumEpilogueWarpgroups == 0 and
        (BLOCK_N / kNumEpilogueWarpgroups) >= 64;
    constexpr uint32_t kWarpgroupSplitM = kSplitNWarpgroups ? 1u : kNumEpilogueWarpgroups;
    constexpr uint32_t kWarpgroupSplitN = kSplitNWarpgroups ? kNumEpilogueWarpgroups : 1u;
    constexpr uint32_t WG_BLOCK_M = BLOCK_M / kWarpgroupSplitM;
    constexpr uint32_t WG_BLOCK_N = BLOCK_N / kWarpgroupSplitN;
    constexpr bool kSwapABEligible =
        kFP4SwapAB and kSplitNWarpgroups and (BLOCK_M == 64)
        and (BLOCK_N == 128 or BLOCK_N == 256)
        and (kWarpgroupSplitN == 2) and (not kFP4SSNSplit);
    constexpr bool kSwapABL1Active = kSwapABEligible;
    constexpr bool kSwapABL2Active = kSwapABEligible;
    constexpr bool kSwapABFastAmaxActive =
        kSwapABL1Active and kFP4SwapABFastAmax;
    constexpr uint32_t kSwapABNSubtiles = WG_BLOCK_N / 64;
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    DG_STATIC_ASSERT(not kSwapABEligible or (BLOCK_M % 8 == 0),
                     "swapAB epilogue token chunks assume BLOCK_M is a multiple of 8");
    DG_STATIC_ASSERT(not kSwapABEligible or
                         (kSwapABNSubtiles == 1 or kSwapABNSubtiles == 2),
                     "swapAB supports one or two internal N64 subtiles");
    using L1WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;  // M=64, N=WG_BLOCK_N, K=32
    using L2WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT(kWarpgroupSplitM * kWarpgroupSplitN == kNumEpilogueWarpgroups,
                     "Invalid warpgroup split");
    DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M,
                     "Each warpgroup must run exactly one WGMMA-M tile");
    DG_STATIC_ASSERT(BLOCK_M % kWarpgroupSplitM == 0 and BLOCK_N % kWarpgroupSplitN == 0,
                     "Invalid warpgroup tile shape");

    // Cluster=1 -> no multicast, A/B are loaded full-sized
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t L1_OUT_BLOCK_N  = BLOCK_N / 2;  // post-SwiGLU
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2;
    // In the split-N=2, BLOCK_N=128 path each WG produces only
    // WG_L1_OUT_BLOCK_N post-SwiGLU columns. WG0 issues one combined 64-column
    // TMA store after both WGs reduce amax, matching the 64-column SF block.
    constexpr bool kSplitNCombinesL1Store = kSplitNWarpgroups and (WG_L1_OUT_BLOCK_N < 64);
    constexpr bool kSplitNSharesSF = kSplitNWarpgroups and (WG_L1_OUT_BLOCK_N < kL2ActsSFGranK);
    DG_STATIC_ASSERT(not kSplitNSharesSF or kSplitNWarpgroups,
                     "share-SF only meaningful under split-N");
    DG_STATIC_ASSERT(not kSplitNSharesSF or (kWarpgroupSplitN == 2),
                     "share-SF currently only supports split-N=2");
    DG_STATIC_ASSERT(not kSwapABFastAmaxActive or kSplitNSharesSF,
                     "swapAB fast-amax currently assumes the split-N shared-SF shape");
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    // The decoded E4M3 B tile uses 128B swizzle to match the SS WGMMA
    // descriptor. The packed FP4 source tile is linear in the default
    // decode-to-SMEM path because only the dequant code reads it by (row, col).
    constexpr uint32_t kSwizzleBMode        = BLOCK_K * sizeof(b_dtype_t);  // 128
    constexpr uint32_t kSwizzleBPackedMode  = 0;
    constexpr uint32_t kSwizzleCDMode  = 128;
    constexpr uint32_t kGranK          = 128;          // L1 acts SF base granularity
    constexpr uint32_t kNumL2SFAPerBlockK = BLOCK_K / kL2ActsSFGranK;
    // SFB granularity for FP4 weights: per-32 K (DSV4 standard, UE8M0).
    // BLOCK_K=128 has 4 SFB groups along K, exactly one per WGMMA::K tile.
    constexpr uint32_t kScaleBGranK     = 32;
    constexpr uint32_t kNumSFBPerBlockK = BLOCK_K / kScaleBGranK;  // 4
    static_assert(L1WGMMA::K == kScaleBGranK,
                  "WGMMA::K must equal kScaleBGranK so that 1 wgmma == 1 SFB block");

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    // Decoded e4m3 B tile (consumed by WGMMA via SS descriptor)
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
        LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // Packed FP4 source tile (TMA-loaded raw nibbles)
    constexpr uint32_t SMEM_B_PACKED_SIZE_PER_STAGE =
        LOAD_BLOCK_N * (BLOCK_K / 2) * sizeof(b_packed_dtype_t);
    // SFA per-stage must be sized for the larger of L1 (BLOCK_M floats) and
    // L2 (2*BLOCK_M floats per-64, or 4*BLOCK_M floats per-32 with BLOCK_N=64).
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>(kNumL2SFAPerBlockK * BLOCK_M * sizeof(float), 128u);
    // SFB UE8M0 per-32: the decode-to-SMEM path stages one packed uint32 per
    // N row per BLOCK_K in SMEM. Each word contains the 4 K/32 scale bytes, so
    // dequant avoids reloading the same word once per K group.
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>(LOAD_BLOCK_N * sizeof(uint32_t), 128u);

    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes). With split-M each math WG
    // writes a disjoint WG_BLOCK_M slice (rows are partitioned), and with
    // split-N each WG writes a disjoint column slice of the same row range
    // (rows are shared); in both cases the total rows x cols footprint is
    // exactly BLOCK_M x BLOCK_N (resp. BLOCK_M x L1_OUT_BLOCK_N for the L1
    // FP8 staging tile), so the total size does NOT scale with
    // kNumEpilogueWarpgroups.
    constexpr uint32_t SMEM_CD_L1_SIZE = BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SWAP_L1_FP32_SIZE =
        (kSwapABL1Active and not kSwapABFastAmaxActive)
            ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(float)
            : 0;
    constexpr uint32_t SMEM_CD_SWAP_L1_FP8_SIZE =
        kSwapABL1Active ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0;
    constexpr uint32_t SMEM_CD_SWAP_L1_AMAX_SIZE =
        kSwapABL1Active ? BLOCK_M * kNumEpilogueWarps * sizeof(float) : 0;
    constexpr uint32_t SMEM_CD_SWAP_L1_SIZE =
        kSwapABL1Active ? (SMEM_CD_SWAP_L1_FP32_SIZE + SMEM_CD_SWAP_L1_FP8_SIZE) : 0;
    constexpr uint32_t SMEM_CD_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_SIZE    = math::constexpr_align(
        SMEM_CD_BASE_SIZE > SMEM_CD_SWAP_L1_SIZE ? SMEM_CD_BASE_SIZE : SMEM_CD_SWAP_L1_SIZE,
        kSharedMemoryAlignment);
    DG_STATIC_ASSERT(not kSwapABL1Active or SMEM_CD_SWAP_L1_AMAX_SIZE <= SMEM_CD_SWAP_L1_FP8_SIZE,
                     "swapAB fast-amax partials must fit in the FP8 staging tile");

    // When SF is shared by two split-N WGs, reduce the per-row amax in SMEM.
    // Only col_idx==0 lanes write SF, so those lanes publish each WG's amax,
    // synchronize once, and read both halves back to avoid atomicMax and a
    // second epilogue-wide barrier.
    //   row_slot = warp_idx_in_wg * 8 + row_idx  (row_idx = lane_idx / 4)
    //   scratch[row_slot][wg_n_idx][r0/r1]
    // 32 rows x 2 WGs x 2 row values (r0, r1) = 128 float slots.
    constexpr uint32_t kAmaxScratchSlots = 32 * 2 * 2;
    constexpr uint32_t SMEM_AMAX_SCRATCH_SIZE = kSplitNSharesSF ?
        math::constexpr_align<uint32_t>(kAmaxScratchSlots * sizeof(uint32_t),
                                        kSharedMemoryAlignment) : 0;
    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        SMEM_AMAX_SCRATCH_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                      SMEM_B_PACKED_SIZE_PER_STAGE +
                      SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = SM90FP8FP4MegaMoEBuffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_gemm_base);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_gemm_base);
    auto smem_cd_swap_l1_fp32 = reinterpret_cast<float*>(smem_gemm_base);
    auto smem_cd_swap_l1_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(
        math::advance_ptr(smem_gemm_base, SMEM_CD_SWAP_L1_FP32_SIZE));
    auto smem_cd_swap_l1_amax = reinterpret_cast<float*>(smem_cd_swap_l1_fp8);
    // share-SF amax scratch lives in its own region after SMEM_CD.
    auto smem_amax_scratch = reinterpret_cast<uint32_t*>(
        math::advance_ptr(smem_gemm_base, SMEM_CD_SIZE));
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(
            smem_gemm_base,
            SMEM_CD_SIZE + SMEM_AMAX_SCRATCH_SIZE +
                i * SMEM_A_SIZE_PER_STAGE);
    });
    // Decoded e4m3 B tile (the operand actually consumed by WGMMA).
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base,
            SMEM_CD_SIZE + SMEM_AMAX_SCRATCH_SIZE +
            kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    // Packed FP4 source tile (TMA-loaded; consumed only by the math warpgroup
    // during the FP4-to-E4M3 dequant pass).
    auto smem_b_packed = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_packed_dtype_t>(smem_gemm_base,
            SMEM_CD_SIZE + SMEM_AMAX_SCRATCH_SIZE
            + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE)
            + i * SMEM_B_PACKED_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + SMEM_AMAX_SCRATCH_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE
                      + SMEM_B_SIZE_PER_STAGE
                      + SMEM_B_PACKED_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });

    auto sfb_start_ptr = sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE;
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sfb_start_ptr + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Barriers live after SFA and staged SFB.
    constexpr bool kUseEarlyBDecode = kEarlyBDecode;
    constexpr uint32_t kNumDecodeFullBarriers = kUseEarlyBDecode ? kNumStages : 0;
    constexpr bool kUseDecodeDoneMBarrier = kDecodeDoneMBarrier;
    constexpr uint32_t kNumDecodeDoneBarriers = kUseDecodeDoneMBarrier ? kNumStages : 0;
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sfb_start_ptr + kNumStages * SMEM_SFB_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto decode_full_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + i;
    });
    auto decode_done_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + kNumDecodeFullBarriers + i;
    });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + kNumDecodeFullBarriers + kNumDecodeDoneBarriers + i;
    });
    auto combine_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + kNumDecodeFullBarriers + kNumDecodeDoneBarriers + kNumStages + i;
    });

    // =====================================================================
    // Initialization
    // =====================================================================
#ifdef DG_MEGA_MOE_PHASE_PROFILE
    if (thread_idx < kPhaseProfileSlots)
        phase_profile[thread_idx] = 0;
    if (thread_idx == 0) {
        phase_profile[kProfileStartClock] = clock64();
        phase_profile[kProfileEntryGlobaltimer] = ptx::get_globaltimer();
    }
#endif
    if constexpr (kDispatchExpertReady or kCombineExpertReady) {
        // One launch epoch governs dispatch metadata, combine ready, and ring
        // lifetime.  Only SM 0 advances it, exactly once per kernel launch.
        if (sm_idx == 0 and thread_idx == 0) {
#ifdef DG_MEGA_MOE_FP4_SIDECAR_PUBLISHER
            const uint64_t next_epoch =
                ptx::ld_acq_sys(workspace.get_launch_epoch_ptr()) + 1;
            constexpr uint64_t kSidecarArmTimeoutCycles =
                60ull * 2000000000ull;
            const uint64_t arm_wait_start = clock64();
            while (ptx::ld_acq_sys(
                       workspace.get_sidecar_publisher_armed_epoch_ptr()) !=
                   next_epoch) {
                DG_TRAP_ONLY_DEVICE_ASSERT(
                    clock64() - arm_wait_start <
                        kSidecarArmTimeoutCycles);
            }
#endif
            ptx::atomic_add_sys(workspace.get_launch_epoch_ptr(), 1ull);
        }
    }
    if (warp_idx == 0) {
        // Clean expert-count shared memory
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            ptx::st_shared(smem_expert_count + i, 0u);
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Default path uses one A+B full barrier. The early-B path
                // splits packed-B readiness so assist warps can decode while
                // A/SFA TMA is still in flight; the main full barrier then only
                // tracks the A/SFA producer.
                // One warp issues A/SFA and B and commits their total byte
                // count with a single producer arrival.
                full_barriers[i]->init(1);
                if constexpr (kUseEarlyBDecode)
                    decode_full_barriers[i]->init(1);
                if constexpr (kUseDecodeDoneMBarrier) {
                    // decode_done is a one-way producer->consumer mbarrier:
                    // only the warps that actually run `decode_fp4_b_stage`
                    // arrive on it (via `arrive_or_sync_fp4_decode_done`).
                    // Those are the decode-assist warps -- i.e.
                    // `kNumMMANonEpilogueWarps` minus the leading loader warps
                    // that skip decode-assist when `kFirstFP4DecodeAssistWarp`
                    // > 0 -- plus the optional math-WG decode warps. Counting
                    // all `kNumMMANonEpilogueWarps` here over-counts arrivals
                    // by `kFirstFP4DecodeAssistWarp`, so the consumer `wait()`
                    // would never complete.
                    constexpr uint32_t kDecodeDoneArrivers =
                        (kNumMMANonEpilogueWarps - kFirstFP4DecodeAssistWarp) +
                        kNumMathWGDecodeWarps;
                    decode_done_barriers[i]->init(kDecodeDoneArrivers);
                }
                // Each math warp arrives once per stage release.
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if constexpr (kDispatchExpertReady or kCombineExpertReady)
        comm::grid_sync<kNumSMs, 2>(
            workspace, sm_idx, thread_idx, [&]() { __syncthreads(); });
    uint64_t kernel_launch_epoch = 0;
    if constexpr (kDispatchExpertReady or kCombineExpertReady) {
        kernel_launch_epoch = ptx::ld_acq_sys(
            workspace.get_launch_epoch_ptr());
        DG_TRAP_ONLY_DEVICE_ASSERT(kernel_launch_epoch != 0);
    }

#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
    cpu_proxy_slot_idx = static_cast<uint32_t>(
        (kernel_launch_epoch - 1) % cpu_proxy_num_slots);
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY_ASYNC_CREDIT
    // Every publisher has stopped posting when the previous kernel returns.
    // Before a physical slot wraps, consume the completion snapshots recorded
    // after that slot's prior launch.  Other CTAs stay behind a local grid
    // barrier so no producer can overwrite staging while the proxy still owns
    // it.  Launches that use a fresh slot pay neither a CQ wait nor this sync.
    if (kernel_launch_epoch > cpu_proxy_num_slots) {
        if (sm_idx == 0 and warp_idx == 0 and
            lane_idx < kCPUProxyPeersPerNode) {
            const auto& pair_dev_comm = cpu_proxy_dev_comms[lane_idx];
            ncclGin net(
                pair_dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU);
            net.wait(
                *get_cpu_proxy_completion_request_ptr(
                    cpu_proxy_slot_idx, lane_idx),
                ncclCoopThread());
        }
        comm::grid_sync<kNumSMs, 3>(
            workspace, sm_idx, thread_idx, [&]() { __syncthreads(); });
    }
#endif
#endif

    if constexpr (kCombineStageRing) {
        if (sm_idx == 0 and warp_idx == 0) {
            combine_stage_ring.begin_launch(kernel_launch_epoch);
        }
        __syncthreads();
        combine_stage_ring.wait_for_launch_init();
        __syncthreads();
    }

    // =====================================================================
    // Scheduler
    // =====================================================================
    constexpr uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
    constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    constexpr uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N;
    constexpr uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K;
    constexpr uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K;
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks,
        kNumExpertsPerLane, kNumL1BlockNs, kNumL2BlockNs,
        kNumL1BlockKs, kNumL2BlockKs,
        layout::SM90Workspace, (kNumTopk >= 8)>(
            workspace,
            kDispatchExpertReady ? kernel_launch_epoch : 0,
            sym_buffer.rank_idx);
    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;
    constexpr uint32_t kSchedulerCountCacheBarrierIdx   = 14;
    constexpr uint32_t kFP4DecodeBarrierIdx             = 15;
#ifdef DG_MEGA_MOE_INTERNODE
    // A dedicated non-epilogue warp publishes concurrently with dispatch.
    constexpr uint32_t kNumAsyncPublisherThreads = 32;
#else
    constexpr uint32_t kNumAsyncPublisherThreads = 0;
#endif
    constexpr uint32_t kNumDispatchEpilogueSyncThreads =
        kNumDispatchThreads + kNumEpilogueThreads +
        kNumAsyncPublisherThreads;
    DG_STATIC_ASSERT(kEpilogueWGBarrierStartIdx + kNumEpilogueWarpgroups <= kSchedulerCountCacheBarrierIdx,
                     "Epilogue WG barriers overlap scheduler-count cache barrier");
    const uint32_t* cached_recv_counts = smem_expert_count;
    auto cache_expert_recv_counts = [&]() {
        // The inter-node scheduler reads count/manifest from the same
        // epoch-parity landing slot as route payload.  The double-buffered
        // lifetime makes the old FP4-only global snapshot unnecessary; each
        // CTA can cache the launch's finalized expert totals directly.
        if (thread_idx < kNumExpertsPerRank)
            smem_expert_count[thread_idx] =
                scheduler.wait_expert_recv_count(thread_idx);
        ptx::sync_unaligned(kNumThreads, kSchedulerCountCacheBarrierIdx);
    };

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    // Split-N halves the live accumulator footprint per math warpgroup, so it
    // does not need the full 208-register epilogue allocation used by the
    // regular N=128 path.
    constexpr uint32_t kNumDispatchRegisters    = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters =
        kSplitNWarpgroups ? 160 : 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // SFB UE8M0 layouts (one uint32 per (n_row, 4 K-groups = BLOCK_K=128)):
    //   L1: shape [E, 2*IH, H/128] uint32, gran_mn=1 along N.
    //   L2: shape [E, H, IH/128] uint32.
    constexpr uint32_t kL1SFBKWords     = kHidden / 128;
    constexpr uint32_t kL2SFBKWords     = kIntermediateHidden / 128;
    constexpr uint32_t kL1SFBPerExpert  = (kIntermediateHidden * 2) * kL1SFBKWords;
    constexpr uint32_t kL2SFBPerExpert  = kHidden * kL2SFBKWords;
    constexpr uint32_t kNumFP4DecodeAssistWarps =
        kNumMMANonEpilogueWarps - kFirstFP4DecodeAssistWarp;
    constexpr uint32_t kNumFP4DecodeAssistThreads = kNumFP4DecodeAssistWarps * 32;
    constexpr uint32_t kNumFP4DecodeWorkerThreads = kNumFP4DecodeAssistThreads +
        kNumMathWGDecodeWarps * 32;
    constexpr uint32_t kNumFP4DecodeBarrierThreads =
        kNumFP4DecodeAssistThreads + kNumEpilogueThreads;
    auto arrive_or_sync_fp4_decode_done = [&](const uint32_t& cur_stage_idx) {
        if constexpr (kUseDecodeDoneMBarrier) {
            __syncwarp();
            if (lane_idx == 0)
                decode_done_barriers[cur_stage_idx]->arrive();
        } else {
            ptx::sync_aligned(kNumFP4DecodeBarrierThreads, kFP4DecodeBarrierIdx);
        }
    };
    auto wait_fp4_decode_done = [&](const uint32_t& cur_stage_idx,
                                    const uint32_t& cur_phase) {
        if constexpr (kUseDecodeDoneMBarrier) {
            decode_done_barriers[cur_stage_idx]->wait(cur_phase);
        } else {
            ptx::sync_aligned(kNumFP4DecodeBarrierThreads, kFP4DecodeBarrierIdx);
        }
    };
    auto wait_fp4_decode_input_ready = [&](const uint32_t& cur_stage_idx,
                                           const uint32_t& cur_phase) {
        if constexpr (kUseEarlyBDecode) {
            decode_full_barriers[cur_stage_idx]->wait(cur_phase);
        } else {
            full_barriers[cur_stage_idx]->wait(cur_phase);
        }
    };
    auto decode_fp4_b_stage = [&](const uint32_t& cur_stage_idx,
                                  const uint32_t& decode_thread_idx) {
        dequant_fp4_b_tile_to_e4m3_smem_dispatch<
            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK,
            kUseWideLoadDecode>(
            decode_thread_idx, kNumFP4DecodeWorkerThreads,
            smem_b_packed[cur_stage_idx], smem_b[cur_stage_idx],
            smem_sfb[cur_stage_idx]);
        arrive_or_sync_fp4_decode_done(cur_stage_idx);
    };

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" to SF token index is now the simple
    //       per-block linear mapping (no 4x32 transpose).
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        const bool profile_dispatch_leader = thread_idx == 0;
        const uint64_t profile_metadata_start =
            profile_dispatch_leader ? clock64() : 0;
#endif

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumMetadataSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count tokens per expert
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Stake out per-expert SM offsets via global atomic
        if (sm_idx < kNumMetadataSMs) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts;
                 i += kNumDispatchThreads) {
                const uint64_t send_value =
                    (1ull << 32) |
                    static_cast<uint64_t>(smem_expert_count[i]);
                smem_expert_count[i] = static_cast<uint32_t>(
                    ptx::atomic_add(
                        workspace.get_expert_send_count_ptr(i), send_value));
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

#if (defined(DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED) || \
     defined(DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3)) && \
    defined(DG_MEGA_MOE_INTERNODE)
        const uint32_t gateway_epoch_slot =
            static_cast<uint32_t>(kernel_launch_epoch) & 1u;
#endif

        // Write source token-topk indices to remote ranks.  Only a thread
        // that actually publishes an entry needs a system fence before the
        // CTA completion handoff below.
        bool wrote_route_entry = false;
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_local_expert_idx = expert_idx % kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                dst_local_expert_idx, sym_buffer.rank_idx, dst_slot_idx);
#ifdef DG_MEGA_MOE_INTERNODE
            if (dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS) {
                // Stage inter-node route entries in the local same-rail
                // gateway's epoch-parity collect box.  The gateway later
                // forwards either one dense box or one compact live payload.
                const auto gateway_rank_idx =
                    sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS *
                        DG_MEGA_MOE_NVL_PEERS +
                    dst_rank_idx % DG_MEGA_MOE_NVL_PEERS;
                *sym_buffer.map(
                    workspace.get_gateway_entry_ptr(
                        sym_buffer.rank_idx % DG_MEGA_MOE_NVL_PEERS,
                        dst_local_expert_idx, dst_slot_idx,
                        layout::gateway_rel_node(
                            dst_rank_idx / DG_MEGA_MOE_NVL_PEERS,
                            sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS),
                        gateway_epoch_slot),
                    gateway_rank_idx) = token_topk_idx;
            } else
#endif
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
            wrote_route_entry = true;
        });

        // Every writer makes its same-node route stores visible before the
        // grid-wide handoff to SM 0.  Inter-node WQE ordering is provided by
        // the per-expert RC QP itself.
        if constexpr (kDispatchExpertReady) {
            if (wrote_route_entry)
                __threadfence_system();
        }

#if (defined(DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED) || \
     defined(DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3)) && \
    defined(DG_MEGA_MOE_INTERNODE)
        // Eager send-when-full handshake.  Every CTA contributes once to each
        // destination counter after fencing its route stores.  The last CTA
        // owns publication of that direction's final manifest/count row.
        if constexpr (kDispatchExpertReady) {
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            if (sm_idx < kNumMetadataSMs and thread_idx < kNumRanks) {
                const uint32_t dst_rank_idx = thread_idx;
                const auto old_done = ptx::atomic_add_acq_rel_sys(
                    workspace.get_gateway_direction_done_ptr(dst_rank_idx), 1);
                DG_TRAP_ONLY_DEVICE_ASSERT(old_done < kNumMetadataSMs);
                if (old_done + 1 == kNumMetadataSMs) {
                    const auto dispatch_epoch_full = kernel_launch_epoch;
                    const auto dispatch_epoch =
                        static_cast<uint32_t>(dispatch_epoch_full);
                    const bool dst_is_inter =
                        dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
                    const auto gateway_rank_idx =
                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS *
                            DG_MEGA_MOE_NVL_PEERS +
                        dst_rank_idx % DG_MEGA_MOE_NVL_PEERS;
                    const auto src_nvl_idx =
                        sym_buffer.rank_idx % DG_MEGA_MOE_NVL_PEERS;
                    for (uint32_t e = 0; e < kNumExpertsPerRank; ++ e) {
                        const auto expert_status =
                            *workspace.get_expert_send_count_ptr(
                                dst_rank_idx * kNumExpertsPerRank + e);
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            (expert_status >> 32) == kNumMetadataSMs);
                        const uint64_t ready_status =
                            (static_cast<uint64_t>(dispatch_epoch) << 32) |
                            static_cast<uint32_t>(expert_status);
                        if (dst_is_inter) {
                            const uint32_t rel_dst = layout::gateway_rel_node(
                                dst_rank_idx / DG_MEGA_MOE_NVL_PEERS,
                                sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS);
                            *sym_buffer.map(
                                workspace.get_gateway_manifest_ptr(
                                    src_nvl_idx, e, rel_dst,
                                    dispatch_epoch & 1u),
                                gateway_rank_idx) = ready_status;
                        } else {
                            ptx::st_relaxed_sys(
                                sym_buffer.map(
                                    workspace.get_dispatch_epoch_count_ptr(
                                        dispatch_epoch & 1u,
                                        sym_buffer.rank_idx, e),
                                    dst_rank_idx),
                                ready_status);
                        }
                    }
                    if (dst_is_inter) {
                        __threadfence_system();
                        const uint32_t rel_dst = layout::gateway_rel_node(
                            dst_rank_idx / DG_MEGA_MOE_NVL_PEERS,
                            sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS);
                        ptx::st_release_sys(
                            sym_buffer.map(
                                workspace.get_gateway_flag_ptr(
                                    src_nvl_idx, rel_dst,
                                    dispatch_epoch & 1u),
                                gateway_rank_idx),
                            dispatch_epoch_full);
                    }
                }
            }
        }
#else
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );
#endif

#if (defined(DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED) || \
     defined(DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3)) && \
    defined(DG_MEGA_MOE_INTERNODE)
        if constexpr (false) {
#else
        if (sm_idx == 0) {
#endif
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                const auto recv_count_ptr =
#if (defined(DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED) || \
     defined(DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3)) && \
    defined(DG_MEGA_MOE_INTERNODE)
                    workspace.get_dispatch_epoch_count_ptr(
                        static_cast<uint32_t>(kernel_launch_epoch) & 1u,
                        sym_buffer.rank_idx, dst_local_expert_idx);
#else
                    workspace.get_expert_recv_count_ptr(
                        sym_buffer.rank_idx, dst_local_expert_idx);
#endif
#ifdef DG_MEGA_MOE_INTERNODE
                uint64_t ready_status = expert_status;
                if constexpr (kDispatchExpertReady) {
                    const auto dispatch_epoch =
                        static_cast<uint32_t>(kernel_launch_epoch);
                    ready_status =
                        (static_cast<uint64_t>(dispatch_epoch) << 32) |
                        static_cast<uint32_t>(expert_status);
                }
                if (dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                    sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS) {
                    const auto gateway_rank_idx =
                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS *
                            DG_MEGA_MOE_NVL_PEERS +
                        dst_rank_idx % DG_MEGA_MOE_NVL_PEERS;
                    *sym_buffer.map(
                        workspace.get_gateway_manifest_ptr(
                            sym_buffer.rank_idx % DG_MEGA_MOE_NVL_PEERS,
                            dst_local_expert_idx,
                            layout::gateway_rel_node(
                                dst_rank_idx / DG_MEGA_MOE_NVL_PEERS,
                                sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS),
                            gateway_epoch_slot),
                        gateway_rank_idx) = ready_status;
                } else if constexpr (kDispatchExpertReady) {
                    ptx::st_relaxed_sys(
                        sym_buffer.map(recv_count_ptr, dst_rank_idx),
                        ready_status);
                } else {
                    *sym_buffer.map(recv_count_ptr, dst_rank_idx) = ready_status;
                }
#else
                *sym_buffer.map(recv_count_ptr, dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
#endif
            }
#if (defined(DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED) || \
     defined(DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3)) && \
    defined(DG_MEGA_MOE_INTERNODE)
            if constexpr (kDispatchExpertReady) {
                ptx::sync_aligned(
                    kNumDispatchThreads, kDispatchBarrierIdx);
                __threadfence_system();
                constexpr uint32_t kNumNodes =
                    kNumRanks / DG_MEGA_MOE_NVL_PEERS;
                constexpr uint32_t kNumRemoteNodes = kNumNodes - 1;
                for (uint32_t task = thread_idx;
                     task < kNumRemoteNodes * DG_MEGA_MOE_NVL_PEERS;
                     task += kNumDispatchThreads) {
                    const auto node_base = sym_buffer.rank_idx /
                        DG_MEGA_MOE_NVL_PEERS * DG_MEGA_MOE_NVL_PEERS;
                    const auto flag_epoch = kernel_launch_epoch;
                    const uint32_t rel_dst =
                        task / DG_MEGA_MOE_NVL_PEERS;
                    const uint32_t gateway_nvl =
                        task % DG_MEGA_MOE_NVL_PEERS;
                    ptx::st_release_sys(
                        sym_buffer.map(
                            workspace.get_gateway_flag_ptr(
                                sym_buffer.rank_idx % DG_MEGA_MOE_NVL_PEERS,
                                rel_dst,
                                static_cast<uint32_t>(flag_epoch) & 1u),
                            node_base + gateway_nvl),
                        flag_epoch);
                }
            }
#endif
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

#if defined(DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3) and \
    defined(DG_MEGA_MOE_INTERNODE)
        // Dense V3 forwards the complete epoch-parity collect box and its
        // trailing manifest.  One CTA owns each remote node; the receiver uses
        // the matching double-buffered landing slot without any decode step.
        if constexpr (kDispatchExpertReady) {
            constexpr uint32_t kNumNodes =
                kNumRanks / DG_MEGA_MOE_NVL_PEERS;
            constexpr uint32_t kNumRemoteNodes = kNumNodes - 1;
            constexpr uint32_t kNumGatewayCells =
                DG_MEGA_MOE_NVL_PEERS * kNumExpertsPerRank;
            DG_STATIC_ASSERT(
                kNumRemoteNodes < kNumSMs,
                "Dense V3 gateway needs CTA 0 plus one CTA per remote node");
            if (sm_idx >= 1 and sm_idx <= kNumRemoteNodes) {
                const uint32_t rel_dst = sm_idx - 1;
                const uint64_t my_epoch = kernel_launch_epoch;
                const uint32_t epoch_slot =
                    static_cast<uint32_t>(my_epoch) & 1u;
                const uint32_t my_node_idx = sym_buffer.rank_idx /
                    DG_MEGA_MOE_NVL_PEERS;
                const uint32_t my_nvl_idx = sym_buffer.rank_idx %
                    DG_MEGA_MOE_NVL_PEERS;
                const uint32_t dst_node = layout::gateway_abs_node(
                    rel_dst, my_node_idx);
                const int peer_rank_idx = static_cast<int>(
                    dst_node * DG_MEGA_MOE_NVL_PEERS + my_nvl_idx);
                const uint32_t rel_self_at_dst = layout::gateway_rel_node(
                    my_node_idx, dst_node);
                const uint32_t node_src_base =
                    my_node_idx * DG_MEGA_MOE_NVL_PEERS;
                constexpr int kGatewayQpId = kNumExpertsPerRank;
                constexpr int64_t kGatewayTimeoutCycles =
                    60ll * 2000000000ll;
                const uint64_t gateway_wait_start = clock64();

                // Each local source rank raises this exact destination/epoch
                // flag only after its dense entries and manifest are visible.
                if (warp_idx == 0) {
                    for (uint32_t i = lane_idx;
                         i < DG_MEGA_MOE_NVL_PEERS; i += 32) {
                        while (ptx::ld_acq_sys(
                                   workspace.get_gateway_flag_ptr(
                                       i, rel_dst, epoch_slot)) != my_epoch)
                            DG_TRAP_ONLY_DEVICE_ASSERT(
                                clock64() - gateway_wait_start <
                                kGatewayTimeoutCycles);
                    }
                    __syncwarp();

                    // Reserving the complete pair preserves RC ordering:
                    // observing a manifest entry implies that every fragment
                    // of the preceding dense box WRITE has landed.
                    __threadfence_system();
                    const comm::ibgda::PutRequest requests[2] = {
                        {
                            reinterpret_cast<uint64_t>(
                                workspace.get_gateway_dense_landing_slot_ptr(
                                    rel_self_at_dst, epoch_slot)),
                            reinterpret_cast<uint64_t>(
                                workspace.get_gateway_entry_ptr(
                                    0, 0, 0, rel_dst, epoch_slot)),
                            static_cast<size_t>(
                                workspace.get_gateway_dense_slot_bytes())
                        },
                        {
                            reinterpret_cast<uint64_t>(
                                workspace.get_dispatch_epoch_count_ptr(
                                    epoch_slot, node_src_base, 0)),
                            reinterpret_cast<uint64_t>(
                                workspace.get_gateway_manifest_ptr(
                                    0, 0, rel_dst, epoch_slot)),
                            static_cast<size_t>(
                                kNumGatewayCells * sizeof(uint64_t))
                        }
                    };
                    comm::ibgda::put_nbi_warp_group(
                        requests, peer_rank_idx, kGatewayQpId,
                        static_cast<int>(lane_idx));
                }
            }
        }
#endif

#if defined(DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED) and \
    defined(DG_MEGA_MOE_INTERNODE)
        // One CTA per remote node compacts the live cell prefixes into a
        // double-buffered send slot.  Header, offsets, payload and manifest
        // are forwarded with a constant one/two-WQE batch independent of the
        // configured maximum token capacity.
        if constexpr (kDispatchExpertReady) {
            constexpr uint32_t kNumNodes =
                kNumRanks / DG_MEGA_MOE_NVL_PEERS;
            constexpr uint32_t kNumRemoteNodes = kNumNodes - 1;
            constexpr uint32_t kNumGatewayCells =
                DG_MEGA_MOE_NVL_PEERS * kNumExpertsPerRank;
            DG_STATIC_ASSERT(
                kNumRemoteNodes < kNumSMs,
                "Packed gateway needs CTA 0 plus one CTA per remote node");
            if (sm_idx >= 1 and sm_idx <= kNumRemoteNodes) {
                const uint32_t rel_dst = sm_idx - 1;
                const uint64_t my_epoch = kernel_launch_epoch;
                const uint32_t epoch_slot =
                    static_cast<uint32_t>(my_epoch) & 1u;
                const uint32_t my_node_idx = sym_buffer.rank_idx /
                    DG_MEGA_MOE_NVL_PEERS;
                const uint32_t my_nvl_idx = sym_buffer.rank_idx %
                    DG_MEGA_MOE_NVL_PEERS;
                const uint32_t dst_node = layout::gateway_abs_node(
                    rel_dst, my_node_idx);
                const int peer_rank_idx = static_cast<int>(
                    dst_node * DG_MEGA_MOE_NVL_PEERS + my_nvl_idx);
                const uint32_t rel_self_at_dst = layout::gateway_rel_node(
                    my_node_idx, dst_node);
                constexpr int kGatewayQpId = kNumExpertsPerRank;
                constexpr int64_t kGatewayTimeoutCycles =
                    60ll * 2000000000ll;
                const uint64_t gateway_wait_start = clock64();

                // Protect the double-buffered registered send slot until the
                // RNIC has consumed the previous same-parity transfer.
                if (warp_idx == 0 and lane_idx == 0) {
                    const uint64_t previous_completion = ptx::ld_acq_sys(
                        workspace.get_gateway_send_completion_ptr(
                            rel_dst, epoch_slot));
                    if (previous_completion != 0)
                        comm::ibgda::wait_until(
                            peer_rank_idx, kGatewayQpId,
                            previous_completion);
                }
                if (warp_idx == 0)
                    __syncwarp();

                if (warp_idx == 0) {
                    for (uint32_t i = lane_idx;
                         i < DG_MEGA_MOE_NVL_PEERS; i += 32) {
                        while (ptx::ld_acq_sys(
                                   workspace.get_gateway_flag_ptr(
                                       i, rel_dst, epoch_slot)) != my_epoch)
                            DG_TRAP_ONLY_DEVICE_ASSERT(
                                clock64() - gateway_wait_start <
                                kGatewayTimeoutCycles);
                    }
                    __syncwarp();
                }

                uint32_t packed_total_entries = 0;
                if (warp_idx == 0) {
                    uint32_t running_total = 0;
                    for (uint32_t cell_base = 0;
                         cell_base < kNumGatewayCells; cell_base += 32) {
                        const uint32_t cell_idx = cell_base + lane_idx;
                        uint64_t ready_status = 0;
                        if (cell_idx < kNumGatewayCells) {
                            const uint32_t src_nvl =
                                cell_idx / kNumExpertsPerRank;
                            const uint32_t expert =
                                cell_idx % kNumExpertsPerRank;
                            ready_status = ptx::ld_acq_sys(
                                workspace.get_gateway_manifest_ptr(
                                    src_nvl, expert, rel_dst, epoch_slot));
                            DG_TRAP_ONLY_DEVICE_ASSERT(
                                static_cast<uint32_t>(ready_status >> 32) ==
                                static_cast<uint32_t>(my_epoch));
                        }
                        const uint32_t cell_count =
                            static_cast<uint32_t>(ready_status);
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            cell_count <= workspace.num_max_tokens_per_rank);
                        uint32_t inclusive = cell_count;
                        #pragma unroll
                        for (uint32_t delta = 1; delta < 32; delta <<= 1) {
                            const uint32_t preceding = __shfl_up_sync(
                                0xffffffff, inclusive, delta);
                            if (lane_idx >= delta)
                                inclusive += preceding;
                        }
                        const uint32_t batch_base = __shfl_sync(
                            0xffffffff, running_total, 0);
                        if (cell_idx < kNumGatewayCells) {
                            *workspace.get_gateway_packed_offset_ptr(
                                false, rel_dst, epoch_slot, cell_idx) =
                                batch_base + inclusive - cell_count;
                            if constexpr (kNumTopk < 8)
                                *workspace.get_gateway_packed_manifest_ptr(
                                    false, rel_dst, epoch_slot, cell_idx) =
                                    ready_status;
                        }
                        const uint32_t batch_total = __shfl_sync(
                            0xffffffff, inclusive, 31);
                        if (lane_idx == 0)
                            running_total = batch_base + batch_total;
                        __syncwarp();
                    }
                    packed_total_entries = __shfl_sync(
                        0xffffffff, running_total, 0);
                    if (lane_idx == 0) {
                        *workspace.get_gateway_packed_offset_ptr(
                            false, rel_dst, epoch_slot,
                            kNumGatewayCells) = packed_total_entries;
                        *workspace.get_gateway_packed_header_ptr(
                            false, rel_dst, epoch_slot) = {
                                my_epoch, packed_total_entries,
                                kNumGatewayCells};
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            packed_total_entries <=
                            workspace.get_gateway_max_packed_entries());
                    }
                    __syncwarp();
                }

                ptx::sync_aligned(
                    kNumDispatchThreads, kDispatchBarrierIdx);
                packed_total_entries =
                    workspace.get_gateway_packed_header_ptr(
                        false, rel_dst, epoch_slot)->total_entries;

                for (uint32_t cell_idx = warp_idx;
                     cell_idx < kNumGatewayCells;
                     cell_idx += kNumDispatchWarps) {
                    const uint32_t src_nvl =
                        cell_idx / kNumExpertsPerRank;
                    const uint32_t expert =
                        cell_idx % kNumExpertsPerRank;
                    uint64_t ready_status = 0;
                    if (lane_idx == 0)
                        ready_status = ptx::ld_acq_sys(
                            workspace.get_gateway_manifest_ptr(
                                src_nvl, expert, rel_dst, epoch_slot));
                    ready_status = __shfl_sync(
                        0xffffffff, ready_status, 0);
                    const uint32_t cell_count =
                        static_cast<uint32_t>(ready_status);
                    const uint32_t packed_offset =
                        *workspace.get_gateway_packed_offset_ptr(
                            false, rel_dst, epoch_slot, cell_idx);
                    for (uint32_t i = lane_idx; i < cell_count; i += 32)
                        *workspace.get_gateway_packed_payload_ptr(
                            false, rel_dst, epoch_slot,
                            packed_offset + i) =
                            *workspace.get_gateway_entry_ptr(
                                src_nvl, expert, i, rel_dst, epoch_slot);
                    if constexpr (kNumTopk >= 8) {
                        if (lane_idx == 0)
                            *workspace.get_gateway_packed_compact_manifest_ptr(
                                false, rel_dst, epoch_slot,
                                packed_total_entries, cell_idx) = ready_status;
                    }
                }
                ptx::sync_aligned(
                    kNumDispatchThreads, kDispatchBarrierIdx);

                if (warp_idx == 0) {
                    __threadfence_system();
                    __syncwarp();

                    uint64_t completion = 0;
                    if constexpr (kNumTopk >= 8) {
                        const comm::ibgda::PutRequest requests[1] = {{
                            reinterpret_cast<uint64_t>(
                                workspace.get_gateway_packed_landing_slot_ptr(
                                    rel_self_at_dst, epoch_slot)),
                            reinterpret_cast<uint64_t>(
                                workspace.get_gateway_packed_send_slot_ptr(
                                    rel_dst, epoch_slot)),
                            static_cast<size_t>(
                                workspace
                                    .get_gateway_packed_compact_manifest_offset_bytes(
                                        packed_total_entries) +
                                static_cast<uint64_t>(kNumGatewayCells) *
                                    sizeof(uint64_t))
                        }};
                        completion = comm::ibgda::put_nbi_warp_group(
                            requests, peer_rank_idx, kGatewayQpId,
                            static_cast<int>(lane_idx));
                    } else {
                        const comm::ibgda::PutRequest requests[2] = {
                            {
                                reinterpret_cast<uint64_t>(
                                    workspace.get_gateway_packed_landing_slot_ptr(
                                        rel_self_at_dst, epoch_slot)),
                                reinterpret_cast<uint64_t>(
                                    workspace.get_gateway_packed_send_slot_ptr(
                                        rel_dst, epoch_slot)),
                                static_cast<size_t>(
                                    workspace.get_gateway_packed_payload_offset_bytes() +
                                    static_cast<uint64_t>(packed_total_entries) *
                                        sizeof(uint32_t))
                            },
                            {
                                reinterpret_cast<uint64_t>(
                                    workspace.get_gateway_packed_manifest_ptr(
                                        true, rel_self_at_dst, epoch_slot)),
                                reinterpret_cast<uint64_t>(
                                    workspace.get_gateway_packed_manifest_ptr(
                                        false, rel_dst, epoch_slot)),
                                static_cast<size_t>(
                                    static_cast<uint64_t>(kNumGatewayCells) *
                                    sizeof(uint64_t))
                            }
                        };
                        completion = comm::ibgda::put_nbi_warp_group(
                            requests, peer_rank_idx, kGatewayQpId,
                            static_cast<int>(lane_idx));
                    }
                    if (lane_idx == 0)
                        ptx::st_release_sys(
                            workspace.get_gateway_send_completion_ptr(
                                rel_dst, epoch_slot),
                            completion);
                }
            }
        }
#endif

        // Header/manifest/payload share a QP completion boundary.  No tag-1
        // barrier or snapshot is needed before the epoch-matched count waits.
        __threadfence_system();

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (profile_dispatch_leader) {
            phase_profile[kProfileMetadata] =
                clock64() - profile_metadata_start;
            phase_profile[kProfileCountsSentGlobaltimer] =
                ptx::get_globaltimer();
        }
        const uint64_t profile_dispatch_barrier_start =
            profile_dispatch_leader ? clock64() : 0;
#endif

        if constexpr (not kDispatchExpertReady) {
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                false, true);
        }
        // Expert-ready: no cross-rank barrier -- the epoch-matched slot wait
        // inside `cache_expert_recv_counts` below is the arrival signal.

        // Cache finalized expert counts before the dispatch/epilogue rendezvous
        // so loader warps can leave the all-CTA count barrier and start waiting
        // on L1 arrivals while dispatch and epilogue complete their handshake.
        cache_expert_recv_counts();

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (profile_dispatch_leader) {
            phase_profile[kProfileDispatchBarrier] =
                clock64() - profile_dispatch_barrier_start;
            phase_profile[kProfileCountsReadyGlobaltimer] =
                ptx::get_globaltimer();
        }
#endif

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
#ifdef DG_MEGA_MOE_INTERNODE
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        const uint64_t profile_pull_start = lane_idx == 0 ? clock64() : 0;
        uint64_t profile_remote_read_cycles = 0;
        uint32_t profile_remote_read_count = 0;
#endif

        sm90_fp8_fp4_mega_moe_fetch_cached_expert_recv_count<
            kNumExpertsPerRank, kNumExpertsPerLane>(scheduler, cached_recv_counts);

        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int      current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
#ifdef DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED
        const uint32_t dispatch_epoch_slot =
            static_cast<uint32_t>(kernel_launch_epoch) & 1u;
#endif
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumPullWarps = kNumDispatchWarps;
        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumPullWarps;
        for (uint32_t token_idx = sm_idx * kNumPullWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks ?
                        scheduler.wait_source_expert_recv_count(
                            j, current_expert_idx) : 0;
                }
            }

            // Round-robin rank selection (identical to SM100)
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

#if defined(DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED) || \
    defined(DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3)
            const bool entry_is_inter =
                current_rank_in_expert_idx / DG_MEGA_MOE_NVL_PEERS !=
                sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
            uint32_t src_token_topk_idx = 0;
            if (lane_idx == 0) {
                src_token_topk_idx = entry_is_inter
#ifdef DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED
                    ? *workspace.get_gateway_packed_landing_entry_ptr(
                          layout::gateway_rel_node(
                              current_rank_in_expert_idx /
                                  DG_MEGA_MOE_NVL_PEERS,
                              sym_buffer.rank_idx /
                                  DG_MEGA_MOE_NVL_PEERS),
                          dispatch_epoch_slot,
                          current_rank_in_expert_idx %
                              DG_MEGA_MOE_NVL_PEERS,
                          current_expert_idx, token_idx_in_rank)
#else
                    ? *workspace.get_gateway_dense_landing_entry_ptr(
                          layout::gateway_rel_node(
                              current_rank_in_expert_idx /
                                  DG_MEGA_MOE_NVL_PEERS,
                              sym_buffer.rank_idx /
                                  DG_MEGA_MOE_NVL_PEERS),
                          static_cast<uint32_t>(kernel_launch_epoch) & 1u,
                          current_rank_in_expert_idx %
                              DG_MEGA_MOE_NVL_PEERS,
                          current_expert_idx, token_idx_in_rank)
#endif
                    : *workspace.get_src_token_topk_idx_ptr(
                          current_expert_idx,
                          current_rank_in_expert_idx,
                          token_idx_in_rank);
            }
            src_token_topk_idx = __shfl_sync(
                0xffffffff, src_token_topk_idx, 0);
#else
            const uint32_t src_token_topk_idx =
                *workspace.get_src_token_topk_idx_ptr(
                    current_expert_idx, current_rank_in_expert_idx,
                    token_idx_in_rank);
#endif
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;
            const uint32_t token_idx_in_block = pool_token_idx % BLOCK_M;
            if constexpr (kL1RingEnabled) {
                const uint32_t empty_target =
                    get_l1_ring_wave_idx(pool_block_idx) *
                    (L2_SHAPE_N / BLOCK_N);
                const auto empty_ptr = workspace.get_l1_ring_empty_count_ptr(
                    get_l1_ring_block_idx(pool_block_idx));
                while (ptx::ld_acq(empty_ptr) < empty_target);
            }
            const uint32_t l1_ring_token_idx =
                get_l1_ring_token_idx(pool_token_idx);

#ifdef DG_MEGA_MOE_INTERNODE
            // Source rank on another node: token/SF/weight travel over IBGDA
            // verbs (RDMA READ) instead of NVLink P2P.  Decided per token.
            const bool tok_is_inter =
                current_rank_in_expert_idx / DG_MEGA_MOE_NVL_PEERS !=
                sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
            // QP(src_rank, local_expert) is shared across dispatch warps;
            // completion waits target only this caller's reserved batch.
            const int inter_qp_id = static_cast<int>(current_expert_idx);
            const auto inter_staging = dispatch_staging_buffer
                .get_data_buffer(pool_token_idx).get_base_ptr<float>();
#endif

#ifdef DG_MEGA_MOE_FP4_SIDECAR_DISPATCH_RDMA
            // The sidecar owns every inter-node token in this mode.  This is
            // warp-uniform because source-rank selection is broadcast above.
            if (tok_is_inter)
                continue;
#endif

            // Pull token data into SMEM (NVLink TMA), or RDMA READ the
            // token/SF/weight triplet straight into pool + staging rows.
            if (cute::elect_one_sync()) {
#ifdef DG_MEGA_MOE_INTERNODE
                if (tok_is_inter) {
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    const uint64_t profile_remote_read_start =
                        lane_idx == 0 ? clock64() : 0;
#endif
                    const comm::ibgda::GetRequest read_requests[3] = {
                        {
                            reinterpret_cast<uint64_t>(l1_token_buffer.get_data_buffer(l1_ring_token_idx).get_base_ptr()),
                            reinterpret_cast<uint64_t>(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr()),
                            pull_buffer.get_num_bytes()
                        },
                        {
                            reinterpret_cast<uint64_t>(inter_staging),
                            reinterpret_cast<uint64_t>(input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>()),
                            (kHidden / 128) * sizeof(float)
                        },
                        {
                            reinterpret_cast<uint64_t>(inter_staging + kHidden / 128),
                            reinterpret_cast<uint64_t>(input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx),
                            sizeof(float)
                        }
                    };
                    const auto completion_idx = comm::ibgda::get_batch_thread(
                        read_requests, static_cast<int>(current_rank_in_expert_idx), inter_qp_id);
                    comm::ibgda::wait_until(
                        static_cast<int>(current_rank_in_expert_idx), inter_qp_id, completion_idx);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    if (lane_idx == 0) {
                        profile_remote_read_cycles +=
                            clock64() - profile_remote_read_start;
                        ++ profile_remote_read_count;
                    }
#endif
                } else
#endif
                ptx::tma_load_1d(
                    pull_buffer.get_base_ptr(),
                    sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                   current_rank_in_expert_idx),
                    pull_mbarrier, kHidden);
            }
            __syncwarp();

            // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
            constexpr uint32_t kNumSFFloats = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
            const uint32_t sf_pool_token_idx =
                get_l1_ring_block_idx(pool_block_idx) * SF_BLOCK_M +
                token_idx_in_block;
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFFloats) {
#ifdef DG_MEGA_MOE_INTERNODE
                    // Inter-node: the RDMA READ landed SF in this pool token's
                    // staging row.  The buffer was zeroed at allocation, so the
                    // line may sit stale in L2; `ld.global.cv` refetches the
                    // system-memory value the RNIC wrote.
                    const float sf_val = tok_is_inter
                        ? __ldcv(inter_staging + j)
                        : remote_sf_ptr[j];
                    local_sf_ptr[j * kNumL1SFStorageTokens + sf_pool_token_idx] = sf_val;
#else
                    local_sf_ptr[j * kNumL1SFStorageTokens + sf_pool_token_idx] = remote_sf_ptr[j];
#endif
                }
            }
            __syncwarp();

            if (cute::elect_one_sync()) {
#ifdef DG_MEGA_MOE_INTERNODE
                // Inter-node: the routing weight rode along with SF in the
                // staging row (right after the SF floats).
                const float weight = tok_is_inter
                    ? __ldcv(inter_staging + kHidden / 128)
                    : *sym_buffer.map(
                          input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                          current_rank_in_expert_idx);
#else
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
#endif
                *l1_topk_weights_buffer.get_data_buffer(l1_ring_token_idx).get_base_ptr<float>() = weight;

#ifdef DG_MEGA_MOE_INTERNODE
                if (tok_is_inter) {
                    // Token already delivered into the L1 pool by the blocking
                    // batch READ; no smem bounce to flush, just publish
                    // metadata and the arrival.
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
#ifdef DG_MEGA_MOE_FP4_PUBLISH_ROW_MASK
                    atomicOr(
                        get_combine_publish_row_mask_ptr(
                            pool_block_idx, current_rank_in_expert_idx,
                            token_idx_in_block / kPublishRowsPerMaskWord),
                        1u << (token_idx_in_block %
                               kPublishRowsPerMaskWord));
#endif
                    if constexpr (kCombineStageRing)
                        combine_stage_ring.mark_internode_rows(
                            current_expert_idx);
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                } else
#endif
                {
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(l1_ring_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
#ifdef DG_MEGA_MOE_FP4_PUBLISH_ROW_MASK
                    atomicOr(
                        get_combine_publish_row_mask_ptr(
                            pool_block_idx, current_rank_in_expert_idx,
                            token_idx_in_block / kPublishRowsPerMaskWord),
                        1u << (token_idx_in_block %
                               kPublishRowsPerMaskWord));
#endif

                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                }
            }
            __syncwarp();
        }

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (lane_idx == 0) {
            atomicMax(
                phase_profile + kProfileDispatchPull,
                static_cast<unsigned long long>(
                    clock64() - profile_pull_start));
            atomicMax(
                phase_profile + kProfileRemoteRead,
                static_cast<unsigned long long>(profile_remote_read_cycles));
            atomicAdd(
                phase_profile + kProfileRemoteReadCount,
                static_cast<unsigned long long>(profile_remote_read_count));
        }
#endif

#endif

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        const uint64_t profile_cleanup_barrier_start =
            profile_dispatch_leader ? clock64() : 0;
#endif
        // Cleanup workspace, overlapping with combine.
        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (profile_dispatch_leader)
            phase_profile[kProfileCleanupBarrier] =
                clock64() - profile_cleanup_barrier_start;
#endif

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
#if (defined(DG_MEGA_MOE_DISPATCH_GATEWAY_PACKED) || \
     defined(DG_MEGA_MOE_DISPATCH_GATEWAY_DENSE_V3)) && \
    defined(DG_MEGA_MOE_INTERNODE)
            if (thread_idx < kNumRanks)
                *workspace.get_gateway_direction_done_ptr(thread_idx) = 0;
#endif
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
#ifdef DG_MEGA_MOE_INTERNODE
                const uint32_t num_recv_tokens = scheduler.get_num_tokens(i);
#else
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
#endif
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                const uint32_t expert_pool_block_offset =
                    scheduler.get_pool_block_offset(i);

                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

#ifdef DG_MEGA_MOE_INTERNODE
                // Pair publication is distributed independently of cleanup
                // ownership.  Wait only for this expert, preserving overlap
                // with combine and with publication of all other experts.
                if (thread_idx == 0) {
                    constexpr int64_t kCleanupTimeoutCycles =
                        60ll * 2000000000ll;
                    const uint64_t wait_start = clock64();
                    const uint64_t expected_launch_epoch =
                        kernel_launch_epoch;
                    while (ptx::ld_acq_sys(
                               workspace.get_combine_publish_done_epoch_ptr(i)) !=
                           expected_launch_epoch) {
                        if (clock64() - wait_start >= kCleanupTimeoutCycles) {
#ifdef DG_MEGA_MOE_DEVICE_DIAGNOSTICS
                            printf("FP4_ASYNC_CLEANUP_TIMEOUT rank=%u sm=%u "
                                   "expert=%u done_epoch=%llu expected=%llu "
                                   "pair_done=%u\n",
                                   sym_buffer.rank_idx, sm_idx, i,
                                   static_cast<unsigned long long>(ptx::ld_acq_sys(
                                       workspace.get_combine_publish_done_epoch_ptr(i))),
                                   static_cast<unsigned long long>(expected_launch_epoch),
                                   *workspace.get_combine_publish_pair_done_count_ptr(i));
#endif
                            DG_TRAP_ONLY_DEVICE_ASSERT(false);
                        }
                    }
                }
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
#endif
                if constexpr (kCombineExpertReady) {
                    // These alias the publisher's pair-done count and done
                    // epoch; clearing rearms them for the next launch.
                    if (thread_idx == 0) {
                        *workspace.get_combine_posted_block_count_ptr(i) = 0;
                        *workspace.get_combine_dst_rank_mask_ptr(i) = 0;
                    }
                }

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                if constexpr (not kDispatchExpertReady) {
                    for (uint32_t j = thread_idx; j < kNumRanks;
                         j += kNumDispatchThreads)
                        *workspace.get_expert_recv_count_ptr(j, i) = 0;
                    __syncwarp();
                }

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                    if constexpr (kCombineFullRow)
                        *combine_full_row_arrival_buffer
                             .get_data_buffer(expert_pool_block_offset + j)
                             .get_base_ptr<uint32_t>() = 0;
                }
#ifdef DG_MEGA_MOE_FP4_PUBLISH_ROW_MASK
                const uint32_t num_publish_row_mask_words =
                    num_recv_m_blocks * kNumRanks *
                    kPublishRowMaskStorageWords;
                auto publish_row_mask = combine_publish_row_mask_buffer
                    .get_data_buffer(expert_pool_block_offset)
                    .get_base_ptr<uint32_t>();
                for (uint32_t j = thread_idx;
                     j < num_publish_row_mask_words;
                     j += kNumDispatchThreads)
                    publish_row_mask[j] = 0;
#endif
                __syncwarp();
            }
        }

        // Rearm physical-slot generations for the next launch.  Logical
        // arrival/count arrays above keep their original full-pool indexing;
        // these counters belong to the compact physical rings only.
        if (sm_idx == 0) {
            if constexpr (kL1RingEnabled) {
                for (uint32_t i = thread_idx; i < kNumL1RingBlocks;
                     i += kNumDispatchThreads)
                    *workspace.get_l1_ring_empty_count_ptr(i) = 0;
            }
            if constexpr (kL2RingEnabled) {
                for (uint32_t i = thread_idx; i < kNumL2RingBlocks;
                     i += kNumDispatchThreads)
                    *workspace.get_l2_ring_empty_count_ptr(i) = 0;
            }
        }

        if constexpr (kDispatchExpertReady) {
            comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);
                });
        } else {
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kNumDispatchThreads,
                kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                    workspace, sym_buffer, sm_idx, thread_idx,
                    [=]() {
                        ptx::sync_aligned(
                            kNumDispatchThreads, kDispatchBarrierIdx);
                    },
                    true, false);
        }
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY_ASYNC_CREDIT
        // All expert/destination publishers have submitted their final strong
        // signal before the cleanup rendezvous above can complete.  Capture
        // one queue completion target per remote pair without waiting for it;
        // the target is consumed only when this launch slot is reused.
        if (sm_idx == 0 and warp_idx == 0 and
            lane_idx < kCPUProxyPeersPerNode) {
            const auto& pair_dev_comm = cpu_proxy_dev_comms[lane_idx];
            ncclGin net(
                pair_dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU);
            net.flushAsync(
                ncclTeamWorld(pair_dev_comm), cpu_proxy_peer,
                get_cpu_proxy_completion_request_ptr(
                    cpu_proxy_slot_idx, lane_idx),
                ncclCoopThread());
        }
#endif
        if constexpr (kCombineStageRing) {
            if (sm_idx == 0 and warp_idx == 0) {
                combine_stage_ring.finish_launch(kernel_launch_epoch);
                if (lane_idx == 0)
                    combine_stage_ring.close_launch_gate();
            }
        }

    // =====================================================================
    // ROLE 2: Merged GEMM TMA loader (A+SFA, B+SFB)
    //   Non-epilogue warp 0 loads both operands, warp 1 publishes async
    //   inter-node output, and the remaining warps assist FP4 decode.
    // =====================================================================
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        cache_expert_recv_counts();

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        uint64_t profile_loader_pool_wait_cycles = 0;
#endif
        sm90_fp8_fp4_mega_moe_for_each_cached_block<
            kNumExpertsPerRank, kNumExpertsPerLane, L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
            scheduler, [&]<sched::BlockPhase kBlockPhase, uint32_t kNumBlockKs>(
                           const uint32_t& local_expert_idx,
                           const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            constexpr auto block_phase = kBlockPhase;
            constexpr uint32_t num_k_blocks = kNumBlockKs;
            const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;
            const auto tensor_map_b_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_weights : &tensor_map_l1_weights;
            const uint32_t shape_n =
                block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // Wait for the pool to be ready.  For L1 blocks this is the
            // direct gate on dispatch-pull landings: rows of this block that
            // originate on remote ranks must have been pulled over RDMA
            // before the loader may issue the A-side TMA.
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            const uint64_t profile_pool_wait_start = clock64();
#endif
            if (block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const auto expected = scheduler.template get_valid_m<false>();
                while (ptx::ld_acq(ptr) != expected);
            } else {
                constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                if constexpr (kL2ArrivalCounter) {
                    const auto ptr = reinterpret_cast<const uint32_t*>(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx));
                    const uint32_t expected = kNumL1BlockNs * kNumEpilogueWarpgroups;
                    while (ptx::ld_acq(ptr) != expected);
                } else {
                    const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                    // Each L1 N block sets one bit; total bits = L1_SHAPE_N / BLOCK_N.
                    const uint64_t expected = (kNumL1BlockNs >= 64)
                        ? ~0ull : ((1ull << kNumL1BlockNs) - 1ull);
                    while (ptx::ld_acq_gpu(ptr) != expected);
                }
            }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            profile_loader_pool_wait_cycles +=
                clock64() - profile_pool_wait_start;
#endif
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t m_idx =
                        block_phase == sched::BlockPhase::Linear1 ?
                            get_l1_ring_block_idx(pool_block_idx) * BLOCK_M :
                            get_l2_ring_block_idx(pool_block_idx) * BLOCK_M;
                    const uint32_t sfa_m_idx =
                        block_phase == sched::BlockPhase::Linear1 ?
                            get_l1_ring_block_idx(pool_block_idx) * SF_BLOCK_M :
                            get_l2_ring_block_idx(pool_block_idx) * SF_BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load A
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, 1);

                    // Merged B loader: issue the packed-FP4 weight TMA from
                    // this warp too, freeing loader warp 1 for the publisher.
                    // Producer arrivals for the merged path are deferred until
                    // after the SFB shared-memory writes below: SFB does not
                    // ride the TMA tx count, so an early arrival would let a
                    // consumer read stale scale factors.
                    const uint32_t n_idx =
                        local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                    auto b_full_barrier = kUseEarlyBDecode
                        ? decode_full_barriers[stage_idx]
                        : full_barriers[stage_idx];
                    tma::copy<
                        BLOCK_K / 2, LOAD_BLOCK_N,
                        kSwizzleBPackedMode, b_packed_dtype_t>(
                        tensor_map_b_ptr, b_full_barrier,
                        smem_b_packed[stage_idx],
                        k_block_idx * (BLOCK_K / 2), n_idx, 1);

                    // TMA load SFA
                    if (block_phase == sched::BlockPhase::Linear1) {
                        // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            sfa_m_idx, k_block_idx, 1);
                    } else {
                        // L2 SFA descriptor box is (block_mn, 1).  Default
                        // BLOCK_N=128 loads two per-64 groups; BLOCK_N=64
                        // loads four per-32 groups so each 32-column L1
                        // output block keeps its own quant scale.
                        #pragma unroll
                        for (uint32_t sf_group = 0; sf_group < kNumL2SFAPerBlockK; ++ sf_group) {
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] + sf_group * BLOCK_M,
                                sfa_m_idx,
                                k_block_idx * kNumL2SFAPerBlockK + sf_group, 1);
                        }
                    }
                }
                __syncwarp();

                // Merged SFB load: the whole warp cooperates (one UE8M0 word
                // per N row), exactly as loader warp 1 used to.
                const bool is_l1 = block_phase == sched::BlockPhase::Linear1;
                const uint32_t* sfb_base = is_l1 ? l1_weights_sf : l2_weights_sf;
                const uint32_t sfb_per_expert = is_l1 ? kL1SFBPerExpert : kL2SFBPerExpert;
                const uint32_t sfb_k_words = is_l1 ? kL1SFBKWords : kL2SFBKWords;
                #pragma unroll
                for (uint32_t row = lane_idx; row < LOAD_BLOCK_N; row += 32) {
                    const uint32_t n_global = n_block_idx * BLOCK_N + row;
                    smem_sfb[stage_idx][row] = __ldg(sfb_base
                        + local_expert_idx * sfb_per_expert
                        + n_global * sfb_k_words
                        + k_block_idx);
                }
                __syncwarp();

                // Deferred producer arrivals (see the note at the B TMA):
                // program order after the SFB stores plus the __syncwarp above
                // orders the stores before the release-semantics arrivals.
                if (cute::elect_one_sync()) {
                    const uint32_t sfa_bytes =
                        block_phase == sched::BlockPhase::Linear1
                            ? BLOCK_M * static_cast<uint32_t>(sizeof(float))
                            : kNumL2SFAPerBlockK * BLOCK_M *
                                  static_cast<uint32_t>(sizeof(float));
                    if constexpr (kUseEarlyBDecode) {
                        decode_full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_B_PACKED_SIZE_PER_STAGE);
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + sfa_bytes);
                    } else {
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + sfa_bytes +
                            SMEM_B_PACKED_SIZE_PER_STAGE);
                    }
                }
                __syncwarp();

                if constexpr (kFirstFP4DecodeAssistWarp == 0) {
                    const uint32_t decode_thread_idx =
                        (warp_idx - kNumDispatchWarps) * 32 + lane_idx;
                    wait_fp4_decode_input_ready(stage_idx, phase);
                    decode_fp4_b_stage(stage_idx, decode_thread_idx);
                }
            }
        }, cached_recv_counts);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (lane_idx == 0)
            phase_profile[kProfileLoaderPoolWait] =
                profile_loader_pool_wait_cycles;
#endif

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        cache_expert_recv_counts();

        // B/SFB moved into loader warp 0, leaving warp 1 as the persistent
        // async publisher.  It participates in the two communication-role
        // rendezvous but stays outside FP4 decode-done barriers.
        DG_STATIC_ASSERT(kFirstFP4DecodeAssistWarp >= 2 or kNumMMANonEpilogueWarps < 2,
                         "Merged loader frees warp 1, so decode assist must skip it");
#ifdef DG_MEGA_MOE_INTERNODE
        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);

#ifndef DG_MEGA_MOE_FP4_SIDECAR_PUBLISHER
        sm90_fp8_fp4_mega_moe_fetch_cached_expert_recv_count<
            kNumExpertsPerRank, kNumExpertsPerLane>(
                scheduler, cached_recv_counts);

        // Assign one destination rank to each CTA modulo kNumRanks, then chain
        // that CTA's expert subset.  A warp polls all chained expert cursors and
        // publishes every ready block before moving on, avoiding head-of-line
        // stalls while preserving exclusive QP(dst, expert) ownership.
        const uint64_t launch_epoch = kernel_launch_epoch;
        constexpr uint32_t kPublishRowsPerBatch = 32;
        DG_STATIC_ASSERT(kNumSMs >= kNumRanks,
                         "Dst-grouped publisher needs one warp per dst");
        const uint32_t dst_rank_idx = sm_idx % kNumRanks;
        const uint32_t grouped_slot_idx = sm_idx / kNumRanks;
        const uint32_t grouped_num_slots =
            (kNumSMs - 1 - dst_rank_idx) / kNumRanks + 1;
        constexpr uint32_t kMaxChainPairs = math::constexpr_ceil_div(
            kNumExpertsPerRank, kNumSMs / kNumRanks);
        const bool chain_is_inter =
            dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
            sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;

        uint32_t chain_pending = 0;
        uint32_t chain_expert[kMaxChainPairs];
        uint32_t chain_tokens[kMaxChainPairs];
        uint32_t chain_blocks[kMaxChainPairs];
        uint32_t chain_pool[kMaxChainPairs];
        uint32_t chain_stage_base[kMaxChainPairs];
        uint32_t chain_stage_ready[kMaxChainPairs];
        uint32_t chain_cursor[kMaxChainPairs];
        uint32_t chain_lane_rows[kMaxChainPairs];
#ifdef DG_MEGA_MOE_PHASE_PROFILE
        uint64_t publisher_total_blocks = 0;
        uint64_t publisher_empty_dst_blocks = 0;
        uint64_t publisher_metadata_loads = 0;
        uint64_t publisher_rdma_batches = 0;
        uint64_t publisher_outer_loops = 0;
        uint64_t publisher_ready_checks = 0;
        uint64_t publisher_empty_passes = 0;
        uint64_t publisher_sleep_calls = 0;
        uint64_t publisher_requested_sleep_ns = 0;
        uint64_t publisher_max_requested_sleep_ns = 0;
        uint64_t publisher_ready_observe_ns = 0;
        uint64_t publisher_ready_observe_samples = 0;
#endif
        #pragma unroll
        for (uint32_t idx = 0; idx < kMaxChainPairs; ++ idx) {
            const uint32_t e = grouped_slot_idx + idx * grouped_num_slots;
            chain_expert[idx] = e;
            chain_tokens[idx] = 0;
            chain_blocks[idx] = 0;
            chain_pool[idx] = 0;
            chain_stage_base[idx] = 0;
            chain_stage_ready[idx] = 0;
            chain_cursor[idx] = 0;
            chain_lane_rows[idx] = 0;
            if (e >= kNumExpertsPerRank)
                continue;
            uint32_t pair_tokens = 0;
            if (lane_idx == 0)
                pair_tokens = scheduler.wait_source_expert_recv_count(
                    dst_rank_idx, e);
            chain_tokens[idx] = __shfl_sync(
                0xffffffff, pair_tokens, 0);
            chain_blocks[idx] = math::ceil_div(
                scheduler.get_num_tokens(e), BLOCK_M);
            chain_pool[idx] = scheduler.get_pool_block_offset(e);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            if (lane_idx == 0) {
                publisher_total_blocks += chain_blocks[idx];
                if (chain_tokens[idx] == 0)
                    publisher_empty_dst_blocks += chain_blocks[idx];
            }
#endif
            chain_pending |= 1u << idx;
        }

        const uint64_t chain_wait_start = clock64();
        constexpr int64_t kPublishTimeoutCycles = 60ll * 2000000000ll;
#ifdef DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS
        DG_STATIC_ASSERT(
            DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS > 0,
            "Publisher backoff must start above zero");
        DG_STATIC_ASSERT(
            DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MAX_NS >=
                DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS,
            "Publisher backoff maximum must not be below its initial value");
        uint32_t publisher_backoff_ns =
            DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS;
        uint32_t publisher_idle_passes = 0;
#ifdef DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_PROGRESS_BLOCK_BUDGET
        DG_STATIC_ASSERT(
            DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_PROGRESS_BLOCK_BUDGET > 0,
            "Publisher progress block budget must be positive when enabled");
        DG_STATIC_ASSERT(
            DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_PROGRESS_YIELD_NS > 0,
            "Publisher progress yield must be positive when enabled");
        uint32_t publisher_progress_blocks = 0;
#endif
#endif
        while (chain_pending != 0) {
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            if (lane_idx == 0)
                ++ publisher_outer_loops;
#endif
#if defined(DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS) || \
    defined(DG_MEGA_MOE_PHASE_PROFILE)
            bool chain_made_progress = false;
#endif
            if (clock64() - chain_wait_start >= kPublishTimeoutCycles) {
#ifdef DG_MEGA_MOE_DEVICE_DIAGNOSTICS
                if (lane_idx == 0)
                    printf(
                        "FP4_ASYNC_CHAIN_TIMEOUT "
                        "rank=%u sm=%u epoch=%llu dst=%u pending=0x%x\n",
                        sym_buffer.rank_idx, sm_idx,
                        static_cast<unsigned long long>(launch_epoch),
                        dst_rank_idx, chain_pending);
#endif
                __syncwarp();
                DG_TRAP_ONLY_DEVICE_ASSERT(false);
            }
            #pragma unroll
            for (uint32_t idx = 0; idx < kMaxChainPairs; ++ idx) {
                if ((chain_pending & (1u << idx)) == 0)
                    continue;
                const uint32_t local_expert_idx = chain_expert[idx];
                const int scatter_qp_id =
                    decltype(combine_stage_ring)::scatter_qp_id(
                        local_expert_idx);
                if (chain_tokens[idx] != 0) {
                    const uint32_t expert_num_tokens =
                        scheduler.get_num_tokens(local_expert_idx);
                    while (chain_cursor[idx] < chain_blocks[idx]) {
                        const uint32_t pool_block_idx =
                            chain_pool[idx] + chain_cursor[idx];
                        const auto arrival_ptr =
                            combine_full_row_arrival_buffer
                                .get_data_buffer(pool_block_idx)
                                .get_base_ptr<uint32_t>();
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                        if (lane_idx == 0)
                            ++ publisher_ready_checks;
#endif
                        const uint32_t arrival = ptx::ld_acq(arrival_ptr);
                        if ((arrival & kCombineFullRowPublishReadyBit) == 0)
                            break;
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                        if (lane_idx == 0) {
                            const uint64_t ready_timestamp =
                                *combine_full_row_ready_timestamp_buffer
                                     .get_data_buffer(pool_block_idx)
                                     .get_base_ptr<uint64_t>();
                            publisher_ready_observe_ns +=
                                ptx::get_globaltimer() - ready_timestamp;
                            ++ publisher_ready_observe_samples;
                        }
#endif
#if defined(DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS) || \
    defined(DG_MEGA_MOE_PHASE_PROFILE)
                        chain_made_progress = true;
#endif
                        if constexpr (kCombineStageRing) {
                            if (chain_is_inter and
                                chain_stage_ready[idx] == 0) {
                                uint32_t stage_base = 0;
                                if (lane_idx == 0)
                                    stage_base =
                                        combine_stage_ring.wait_segment_base(
                                            local_expert_idx);
                                chain_stage_base[idx] = __shfl_sync(
                                    0xffffffff, stage_base, 0);
                                chain_stage_ready[idx] = 1;
                            }
                        }
                        const uint32_t m_idx = pool_block_idx * BLOCK_M;
                        const uint32_t valid_m = cute::min(
                            expert_num_tokens -
                                chain_cursor[idx] * BLOCK_M,
                            BLOCK_M);
                        uint32_t block_active_rows = 0;
                        for (uint32_t row_base = 0; row_base < valid_m;
                             row_base += kPublishRowsPerBatch) {
                            const uint32_t row = row_base + lane_idx;
                            bool row_active = false;
                            uint64_t req_rptr = 0;
                            uint64_t req_lptr = 0;
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                            uint64_t req_roffset = 0;
                            uint64_t req_loffset = 0;
#endif
#ifdef DG_MEGA_MOE_FP4_PUBLISH_ROW_MASK
                            uint32_t row_mask = 0;
                            if (lane_idx == 0)
                                row_mask = ptx::ld_acq(
                                    get_combine_publish_row_mask_ptr(
                                        pool_block_idx, dst_rank_idx,
                                        row_base /
                                            kPublishRowsPerMaskWord));
                            row_mask = __shfl_sync(
                                0xffffffff, row_mask, 0);
                            const uint32_t valid_batch_rows = cute::min(
                                valid_m - row_base,
                                kPublishRowsPerBatch);
                            const uint32_t valid_row_mask =
                                valid_batch_rows == kPublishRowsPerBatch ?
                                    0xffffffffu :
                                    (1u << valid_batch_rows) - 1u;
#ifdef DG_MEGA_MOE_DEVICE_DIAGNOSTICS
                            if (lane_idx == 0 and
                                (row_mask & ~valid_row_mask) != 0) {
                                printf(
                                    "FP4_PUBLISH_ROW_MASK_OOB "
                                    "rank=%u sm=%u expert=%u dst=%u "
                                    "block=%u half=%u mask=0x%x "
                                    "valid_mask=0x%x\n",
                                    sym_buffer.rank_idx, sm_idx,
                                    local_expert_idx, dst_rank_idx,
                                    pool_block_idx,
                                    row_base / kPublishRowsPerBatch,
                                    row_mask, valid_row_mask);
                                DG_TRAP_ONLY_DEVICE_ASSERT(false);
                            }
#endif
                            row_mask &= valid_row_mask;
                            block_active_rows |= row_mask;
                            row_active =
                                (row_mask & (1u << lane_idx)) != 0;
                            if (row_active) {
                                ++ chain_lane_rows[idx];
                                if (chain_is_inter) {
                                    const auto src_metadata =
                                        *workspace
                                             .get_token_src_metadata_ptr(
                                                 m_idx + row);
#ifdef DG_MEGA_MOE_DEVICE_DIAGNOSTICS
                                    if (src_metadata.rank_idx !=
                                        dst_rank_idx) {
                                        printf(
                                            "FP4_PUBLISH_ROW_MASK_MISMATCH "
                                            "rank=%u sm=%u expert=%u "
                                            "dst=%u row=%u metadata_dst=%u\n",
                                            sym_buffer.rank_idx, sm_idx,
                                            local_expert_idx, dst_rank_idx,
                                            m_idx + row,
                                            src_metadata.rank_idx);
                                        DG_TRAP_ONLY_DEVICE_ASSERT(false);
                                    }
#endif
                                    const uint32_t staging_row_idx =
                                        kCombineStageRing ?
                                            chain_stage_base[idx] +
                                                chain_cursor[idx] * BLOCK_M +
                                                row :
                                            m_idx + row;
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                                    req_roffset =
                                        get_cpu_proxy_combine_row_offset(
                                            src_metadata.topk_idx,
                                            src_metadata.token_idx);
                                    req_loffset =
                                        get_cpu_proxy_staging_row_offset(
                                            staging_row_idx);
#else
                                    const auto staging_row =
                                        combine_full_row_staging_buffer
                                            .get_data_buffer(
                                                staging_row_idx);
                                    const auto dst_row =
                                        combine_token_buffer
                                            .get_rank_buffer(
                                                src_metadata.topk_idx)
                                            .get_data_buffer(
                                                src_metadata.token_idx);
                                    req_rptr = reinterpret_cast<uint64_t>(
                                        dst_row.get_base_ptr());
                                    req_lptr = reinterpret_cast<uint64_t>(
                                        staging_row.get_base_ptr());
#endif
                                }
                            }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                            if (lane_idx == 0) {
                                if (chain_is_inter)
                                    publisher_metadata_loads +=
                                        __popc(row_mask);
                                if (chain_is_inter and row_mask != 0)
                                    ++ publisher_rdma_batches;
                            }
#endif
                            if (chain_is_inter and row_mask != 0) {
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                                if (row_active) {
                                    const uint32_t remote_local_rank =
                                        dst_rank_idx %
                                        kCPUProxyPeersPerNode;
                                    const auto& pair_dev_comm =
                                        cpu_proxy_dev_comms[
                                            remote_local_rank];
                                    ncclGin net(
                                        pair_dev_comm, 0,
                                        NCCL_GIN_RESOURCE_SHARING_GPU);
                                    net.put(
                                        ncclTeamWorld(pair_dev_comm),
                                        cpu_proxy_peer,
                                        cpu_proxy_windows[remote_local_rank],
                                        req_roffset,
                                        cpu_proxy_windows[remote_local_rank],
                                        req_loffset,
                                        kCPUProxyRowBytes,
                                        ncclGin_None{}, ncclGin_None{},
                                        ncclCoopThread());
                                }
#else
                                comm::ibgda::put_nbi_warp_batch_rows(
                                    req_rptr, req_lptr,
                                    kHidden * sizeof(nv_bfloat16),
                                    row_active,
                                    static_cast<int>(dst_rank_idx),
                                    scatter_qp_id,
                                    static_cast<int>(lane_idx));
#endif
                            }
#else
                            if (row < valid_m) {
                                const auto src_metadata =
                                    *workspace.get_token_src_metadata_ptr(
                                        m_idx + row);
                                row_active =
                                    src_metadata.rank_idx == dst_rank_idx;
                                if (row_active) {
                                    ++ chain_lane_rows[idx];
                                    if (chain_is_inter) {
                                        const uint32_t staging_row_idx =
                                            kCombineStageRing ?
                                                chain_stage_base[idx] +
                                                    chain_cursor[idx] *
                                                        BLOCK_M + row :
                                                m_idx + row;
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                                        req_roffset =
                                            get_cpu_proxy_combine_row_offset(
                                                src_metadata.topk_idx,
                                                src_metadata.token_idx);
                                        req_loffset =
                                            get_cpu_proxy_staging_row_offset(
                                                staging_row_idx);
#else
                                        const auto staging_row =
                                            combine_full_row_staging_buffer
                                                .get_data_buffer(
                                                    staging_row_idx);
                                        const auto dst_row =
                                            combine_token_buffer
                                                .get_rank_buffer(
                                                    src_metadata.topk_idx)
                                                .get_data_buffer(
                                                    src_metadata.token_idx);
                                        req_rptr =
                                            reinterpret_cast<uint64_t>(
                                                dst_row.get_base_ptr());
                                        req_lptr =
                                            reinterpret_cast<uint64_t>(
                                                staging_row.get_base_ptr());
#endif
                                    }
                                }
                            }
                            const uint32_t active_rows = __ballot_sync(
                                0xffffffff, row_active);
                            block_active_rows |= active_rows;
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                            if (lane_idx == 0) {
                                publisher_metadata_loads += cute::min(
                                    valid_m - row_base,
                                    kPublishRowsPerBatch);
                                if (chain_is_inter)
                                    ++ publisher_rdma_batches;
                            }
#endif
                            if (chain_is_inter) {
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                                if (row_active) {
                                    const uint32_t remote_local_rank =
                                        dst_rank_idx %
                                        kCPUProxyPeersPerNode;
                                    const auto& pair_dev_comm =
                                        cpu_proxy_dev_comms[
                                            remote_local_rank];
                                    ncclGin net(
                                        pair_dev_comm, 0,
                                        NCCL_GIN_RESOURCE_SHARING_GPU);
                                    net.put(
                                        ncclTeamWorld(pair_dev_comm),
                                        cpu_proxy_peer,
                                        cpu_proxy_windows[remote_local_rank],
                                        req_roffset,
                                        cpu_proxy_windows[remote_local_rank],
                                        req_loffset,
                                        kCPUProxyRowBytes,
                                        ncclGin_None{}, ncclGin_None{},
                                        ncclCoopThread());
                                }
#else
                                comm::ibgda::put_nbi_warp_batch_rows(
                                    req_rptr, req_lptr,
                                    kHidden * sizeof(nv_bfloat16),
                                    row_active,
                                    static_cast<int>(dst_rank_idx),
                                    scatter_qp_id,
                                    static_cast<int>(lane_idx));
#endif
                            }
#endif
                        }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                        if (lane_idx == 0 and block_active_rows == 0)
                            ++ publisher_empty_dst_blocks;
#endif
                        ++ chain_cursor[idx];
#ifdef DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_PROGRESS_BLOCK_BUDGET
                        // Yield at block granularity so a single productive
                        // scan cannot monopolize issue slots needed by the
                        // FP4 decode and WGMMA warps.
                        ++ publisher_progress_blocks;
                        if (publisher_progress_blocks >=
                            DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_PROGRESS_BLOCK_BUDGET) {
                            __nanosleep(
                                DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_PROGRESS_YIELD_NS);
                            publisher_progress_blocks = 0;
                        }
#endif
                    }
                    if (chain_cursor[idx] != chain_blocks[idx])
                        continue;
                    uint32_t pair_rows = chain_lane_rows[idx];
                    #pragma unroll
                    for (uint32_t offset = 16; offset != 0; offset >>= 1)
                        pair_rows += __shfl_down_sync(
                            0xffffffff, pair_rows, offset);
#ifdef DG_MEGA_MOE_DEVICE_DIAGNOSTICS
                    if (lane_idx == 0 and pair_rows != chain_tokens[idx])
                        printf(
                            "FP4_ASYNC_PAIR_COUNT_MISMATCH "
                            "rank=%u sm=%u epoch=%llu expert=%u dst=%u "
                            "observed_rows=%u expected_rows=%u "
                            "expert_tokens=%u blocks=%u\n",
                            sym_buffer.rank_idx, sm_idx,
                            static_cast<unsigned long long>(launch_epoch),
                            local_expert_idx, dst_rank_idx, pair_rows,
                            chain_tokens[idx], expert_num_tokens,
                            chain_blocks[idx]);
#endif
                    if (lane_idx == 0)
                        DG_DEVICE_ASSERT(pair_rows == chain_tokens[idx]);

                    __threadfence_system();
                    __syncwarp();
                    const uint32_t global_expert_idx =
                        sym_buffer.rank_idx * kNumExpertsPerRank +
                        local_expert_idx;
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                    if (chain_is_inter) {
                        if (lane_idx == 0) {
                            const auto signal_epoch_ptr =
                                get_cpu_proxy_signal_epoch_ptr(
                                    local_expert_idx, dst_rank_idx);
                            const uint64_t previous_signal_epoch =
                                ptx::ld_acq_sys(signal_epoch_ptr);
                            DG_DEVICE_ASSERT(
                                launch_epoch > previous_signal_epoch);
                            const uint64_t signal_delta =
                                launch_epoch - previous_signal_epoch;
                            const uint32_t remote_local_rank =
                                dst_rank_idx % kCPUProxyPeersPerNode;
                            const auto& pair_dev_comm =
                                cpu_proxy_dev_comms[remote_local_rank];
                            ncclGin net(
                                pair_dev_comm, 0,
                                NCCL_GIN_RESOURCE_SHARING_GPU);
                            net.put(
                                ncclTeamWorld(pair_dev_comm),
                                cpu_proxy_peer,
                                cpu_proxy_windows[remote_local_rank], 0,
                                cpu_proxy_windows[remote_local_rank], 0, 0,
                                ncclGin_StrongSignalAdd{
                                    static_cast<ncclGinSignal_t>(
                                        local_expert_idx),
                                    signal_delta},
                                ncclGin_None{}, ncclCoopThread());
                            // The legacy path drains every expert/destination
                            // chain here.  The async-credit path snapshots the
                            // shared proxy queue once after all publishers are
                            // done and waits only when its physical slot wraps.
#ifndef DG_MEGA_MOE_FP4_CPU_PROXY_ASYNC_CREDIT
                            net.flush(ncclCoopThread());
#endif
                            ptx::st_release_sys(
                                signal_epoch_ptr, launch_epoch);
                        }
                    } else
#endif
                    if (lane_idx == 0) {
                        const auto ready_ptr =
                            workspace.get_combine_ready_epoch_ptr(
                                global_expert_idx);
                        if (chain_is_inter) {
                            const uint64_t completion =
                                comm::ibgda::put_inline_with_credit<uint64_t>(
                                    ready_ptr, launch_epoch,
                                    static_cast<int>(dst_rank_idx),
                                    scatter_qp_id);
                            if constexpr (kCombineStageRing)
                                combine_stage_ring.record_completion(
                                    local_expert_idx, dst_rank_idx,
                                    completion);
                        } else {
                            ptx::st_relaxed_sys(
                                sym_buffer.map(ready_ptr, dst_rank_idx),
                                launch_epoch);
                        }
                    }
                    __syncwarp();
                }

                if (lane_idx == 0) {
                    const auto old_pair_done = ptx::atomic_add_acq_rel_sys(
                        workspace.get_combine_publish_pair_done_count_ptr(
                            local_expert_idx), 1);
#ifdef DG_MEGA_MOE_DEVICE_DIAGNOSTICS
                    if (old_pair_done >= kNumRanks)
                        printf(
                            "FP4_ASYNC_PAIR_DONE_OVERFLOW "
                            "rank=%u sm=%u epoch=%llu expert=%u dst=%u "
                            "old_done=%u\n",
                            sym_buffer.rank_idx, sm_idx,
                            static_cast<unsigned long long>(launch_epoch),
                            local_expert_idx, dst_rank_idx, old_pair_done);
#endif
                    DG_DEVICE_ASSERT(old_pair_done < kNumRanks);
                    if (old_pair_done + 1 == kNumRanks) {
                        ptx::st_release_sys(
                            workspace.get_combine_publish_done_epoch_ptr(
                                local_expert_idx),
                            launch_epoch);
                        if constexpr (kCombineStageRing)
                            combine_stage_ring.mark_published(
                                local_expert_idx, launch_epoch);
                    }
                }
                __syncwarp();
                chain_pending &= ~(1u << idx);
#if defined(DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS) || \
    defined(DG_MEGA_MOE_PHASE_PROFILE)
                chain_made_progress = true;
#endif
            }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            if (lane_idx == 0 and not chain_made_progress and
                chain_pending != 0)
                ++ publisher_empty_passes;
#endif
#ifdef DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS
            // Reset immediately after useful work. Consecutive idle scans use
            // bounded exponential backoff to reduce scheduler pressure while
            // retaining low wake-up latency when the producer is nearly ready.
            if (chain_made_progress) {
                publisher_backoff_ns =
                    DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS;
                publisher_idle_passes = 0;
            } else if (chain_pending != 0) {
                publisher_idle_passes +=
                    publisher_idle_passes != 0xffffffffu;
                uint32_t publisher_current_max_backoff_ns =
                    DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MAX_NS;
                if constexpr (
                    DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_FEW_PENDING_CHAINS > 0) {
                    if (__popc(chain_pending) <=
                        DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_FEW_PENDING_CHAINS)
                        publisher_current_max_backoff_ns =
                            DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_FEW_PENDING_MAX_NS;
                    else if constexpr (
                        DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD > 0) {
                        if (publisher_idle_passes >=
                            DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD)
                            publisher_current_max_backoff_ns =
                                DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_MAX_NS;
                    }
                } else if constexpr (
                    DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD > 0) {
                    if (publisher_idle_passes >=
                        DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD)
                        publisher_current_max_backoff_ns =
                            DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_LONG_IDLE_MAX_NS;
                }
                // Clip before sleeping as well as after doubling.  This makes
                // the few-pending cap effective immediately when the chain
                // count drops near the tail.
                publisher_backoff_ns = cute::min(
                    publisher_backoff_ns,
                    publisher_current_max_backoff_ns);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                if (lane_idx == 0) {
                    ++ publisher_sleep_calls;
                    publisher_requested_sleep_ns += publisher_backoff_ns;
                    publisher_max_requested_sleep_ns = cute::max(
                        publisher_max_requested_sleep_ns,
                        static_cast<uint64_t>(publisher_backoff_ns));
                }
#endif
                __nanosleep(publisher_backoff_ns);
                publisher_backoff_ns = cute::min(
                    publisher_backoff_ns * 2u,
                    publisher_current_max_backoff_ns);
            }
#endif
        }

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (lane_idx == 0) {
            phase_profile[kProfilePublishTotalBlocks] =
                publisher_total_blocks;
            phase_profile[kProfilePublishEmptyDstBlocks] =
                publisher_empty_dst_blocks;
            phase_profile[kProfilePublishMetadataLoads] =
                publisher_metadata_loads;
            phase_profile[kProfilePublishRDMABatches] =
                publisher_rdma_batches;
            phase_profile[kProfilePublishOuterLoops] =
                publisher_outer_loops;
            phase_profile[kProfilePublishReadyChecks] =
                publisher_ready_checks;
            phase_profile[kProfilePublishEmptyPasses] =
                publisher_empty_passes;
            const uint64_t sleep_calls =
                publisher_sleep_calls > 0xffffffffull ?
                    0xffffffffull : publisher_sleep_calls;
            const uint64_t requested_sleep_ns =
                publisher_requested_sleep_ns > 0xffffffffull ?
                    0xffffffffull : publisher_requested_sleep_ns;
            phase_profile[kProfilePublishSleepStats] =
                (sleep_calls << 32) | requested_sleep_ns;
            constexpr uint64_t kReadyObserveNsMask =
                (1ull << 48) - 1ull;
            const uint64_t ready_observe_ns =
                publisher_ready_observe_ns > kReadyObserveNsMask ?
                    kReadyObserveNsMask :
                    publisher_ready_observe_ns;
            const uint64_t ready_observe_samples =
                publisher_ready_observe_samples > 0xffffull ?
                    0xffffull : publisher_ready_observe_samples;
            phase_profile[kProfilePublishReadyObserveStats] =
                (ready_observe_samples << 48) |
                ready_observe_ns;
            phase_profile[kProfilePublishMaxRequestedSleepNs] =
                publisher_max_requested_sleep_ns;
        }
#endif
#endif  // DG_MEGA_MOE_FP4_SIDECAR_PUBLISHER

        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);
#endif

    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Remaining non-epilogue warps keep the non-epilogue register allocation
        // and, when selected by kFirstFP4DecodeAssistWarp, assist FP4 decode.
        // They still participate in the warpgroup-collective
        // `setmaxnreg.dec.sync.aligned` so the math warpgroup's
        // `warpgroup_reg_alloc` can succeed.
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        cache_expert_recv_counts();

        {
            const uint32_t non_epilogue_warp_idx = warp_idx - kNumDispatchWarps;
            if (non_epilogue_warp_idx >= kFirstFP4DecodeAssistWarp) {
                const uint32_t decode_thread_idx =
                    (non_epilogue_warp_idx - kFirstFP4DecodeAssistWarp) * 32 + lane_idx;

                sm90_fp8_fp4_mega_moe_for_each_cached_block<
                    kNumExpertsPerRank, kNumExpertsPerLane, L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
                    scheduler, [&]<sched::BlockPhase kBlockPhase, uint32_t kNumBlockKs>(
                                   const uint32_t& local_expert_idx,
                                   const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    constexpr auto block_phase = kBlockPhase;
                    constexpr uint32_t num_k_blocks = kNumBlockKs;
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        wait_fp4_decode_input_ready(stage_idx, phase);
                        decode_fp4_b_stage(stage_idx, decode_thread_idx);
                    }
                }, cached_recv_counts);
            }
        }

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
    // =====================================================================
    // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
    // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        const bool profile_math_leader =
            epilogue_wg_idx == 0 and warp_idx_in_wg == 0 and lane_idx == 0;
        uint64_t profile_l1_cycles = 0;
        uint64_t profile_l2_cycles = 0;
        uint64_t profile_scatter_cycles = 0;
        uint64_t profile_combine_ready_wait_cycles = 0;
        uint64_t profile_decode_wait_cycles = 0;
        uint64_t profile_a_wait_cycles = 0;
        uint32_t profile_l1_block_count = 0;
        uint32_t profile_l2_block_count = 0;
#endif

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        constexpr uint32_t WG_SMEM_CD_L1_STRIDE_N =
            kSplitNCombinesL1Store ? L1_OUT_BLOCK_N : WG_L1_OUT_BLOCK_N;
        constexpr uint32_t WG_SMEM_CD_L2_STRIDE_N = BLOCK_N;

        // WG_BLOCK_M / WG_BLOCK_N are now defined at the outer template scope
        // (see the GEMM data-types block). They are aware of split-N mode, so
        // we must NOT redefine them locally as `BLOCK_M / kNumEpilogueWarpgroups`
        // -- in split-N that would be 32 instead of 64.
        DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M, "Each warpgroup must run exactly one WGMMA per K-block");

        // Decompose `epilogue_wg_idx` into (m,n) coordinates over the
        // (kWarpgroupSplitM, kWarpgroupSplitN) grid:
        //   - split-M path: kWarpgroupSplitN == 1, n_idx == 0
        //   - split-N path: kWarpgroupSplitM == 1, m_idx == 0
        // Both factors collapse cleanly so the same expressions cover both.
        const uint32_t epilogue_wg_m_idx = epilogue_wg_idx / kWarpgroupSplitN;
        const uint32_t epilogue_wg_n_idx = epilogue_wg_idx - epilogue_wg_m_idx * kWarpgroupSplitN;
        const uint32_t wg_m_offset       = epilogue_wg_m_idx * WG_BLOCK_M;
        const uint32_t wg_n_offset       = epilogue_wg_n_idx * WG_BLOCK_N;
        const uint32_t wg_l1_out_n_offset = epilogue_wg_n_idx * WG_L1_OUT_BLOCK_N;
        const uint32_t smem_a_wg_offset   = wg_m_offset * BLOCK_K;
        // smem_b in FP4 SS path is the *decoded* E4M3 tile and stays full
        // LOAD_BLOCK_N rows because FP4 decode is shared across WGs; split-N
        // only shifts the WGMMA-B descriptor base by `wg_n_offset * BLOCK_K`
        // bytes, picking up the WG's own column slice.
        const uint32_t smem_b_wg_offset   = wg_n_offset * BLOCK_K;
        // When two split-N WGs share one SF block (32 output cols/WG), they stage
        // into one joint L1 tile so WG0 can issue a combined TMA store. Otherwise
        // each WG owns a compact contiguous staging tile, matching the TMA box.
        const uint32_t smem_cd_l1_wg_offset =
            kSplitNCombinesL1Store ? wg_l1_out_n_offset
                                   : epilogue_wg_idx * WG_BLOCK_M * WG_L1_OUT_BLOCK_N;
        // L2 BF16 staging keeps the full BLOCK_N row stride. In split-N the
        // WG offset selects the column half while preserving the original tile
        // layout used by the scatter path.
        const uint32_t smem_cd_l2_wg_offset = wg_m_offset * BLOCK_N + wg_n_offset;

        cache_expert_recv_counts();

        // Sync with dispatch
        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);

        sm90_fp8_fp4_mega_moe_for_each_cached_block<
            kNumExpertsPerRank, kNumExpertsPerLane, L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
            scheduler, [&]<sched::BlockPhase kBlockPhase, uint32_t kNumBlockKs>(
                           const uint32_t& local_expert_idx,
                           const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            constexpr auto block_phase = kBlockPhase;
            constexpr uint32_t num_k_blocks = kNumBlockKs;
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            const uint64_t profile_math_block_start =
                profile_math_leader ? clock64() : 0;
            uint64_t profile_scatter_block_start = 0;
#endif
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t l1_ring_m_idx =
                get_l1_ring_block_idx(pool_block_idx) * BLOCK_M;
            const uint32_t l2_ring_m_idx =
                get_l2_ring_block_idx(pool_block_idx) * BLOCK_M;
            const uint32_t l2_ring_sf_m_idx =
                get_l2_ring_block_idx(pool_block_idx) * SF_BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            constexpr bool kSSNSplitActive =
                kFP4SSNSplit and (WG_BLOCK_N == 128)
                and (kL2ActsSFGranK == 64)
                and (kNumEpilogueWarpgroups > 1);
            using SSHalfWGMMA =
                typename mma::sm90::FP8MMASelector<(WG_BLOCK_N >= 64 ? WG_BLOCK_N / 2 : WG_BLOCK_N)>::type;
            constexpr uint32_t kSSHalfAccum = SSHalfWGMMA::kNumAccum;
            constexpr uint32_t kSSAccum = kSSNSplitActive ? kSSHalfAccum : kAccumPerThread;
            // A split-M warpgroup can be outside the valid rows of a short or
            // tail block.  It must keep participating in decode and pipeline
            // barriers, but issuing WGMMA for it is pure waste: the epilogue
            // already drops the corresponding fragment below.  Split-N
            // warpgroups all cover the same M rows and therefore stay active.
            const bool wg_has_valid_rows =
                kSplitNWarpgroups or wg_m_offset < valid_m;
            constexpr bool kDirectAccumulator =
                not kSwapABEligible and BLOCK_M == 128 and BLOCK_N == 128 and
                WG_BLOCK_N == 128;
            float final_accum[kAccumPerThread] = {};
            float direct_prev_scale_0 = 1.0f;
            float direct_prev_scale_1 = 1.0f;
            const auto direct_rescale_final = [&] (
                    const float& prev_scale_0,
                    const float& prev_scale_1,
                    const float& scale_0,
                    const float& scale_1) {
                const float ratio_0 = prev_scale_0 *
                    (kFastMath ? math::fast_rcp(scale_0) : 1.0f / scale_0);
                const float ratio_1 = prev_scale_1 *
                    (kFastMath ? math::fast_rcp(scale_1) : 1.0f / scale_1);
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                    final_accum[i * 4 + 0] *= ratio_0;
                    final_accum[i * 4 + 1] *= ratio_0;
                    final_accum[i * 4 + 2] *= ratio_1;
                    final_accum[i * 4 + 3] *= ratio_1;
                }
            };
            const auto direct_postscale_final = [&] (
                    const float& scale_0, const float& scale_1) {
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                    final_accum[i * 4 + 0] *= scale_0;
                    final_accum[i * 4 + 1] *= scale_0;
                    final_accum[i * 4 + 2] *= scale_1;
                    final_accum[i * 4 + 3] *= scale_1;
                }
            };
            // `valid_m` is tile-invariant, hence so is the swapAB WGMMA-N
            // bucket.  Dispatch the bucket once per tile instead of walking
            // the runtime branch chain in every K stage.  Besides removing
            // the repeated compares, this gives the compiler one fixed
            // WGMMA shape for the complete K loop.
            const auto run_k_stages = [&]<uint32_t kNSwap>() {
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                const uint64_t profile_a_wait_start =
                    profile_math_leader ? clock64() : 0;
#endif
                full_barriers[stage_idx]->wait(phase);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                if (profile_math_leader)
                    profile_a_wait_cycles += clock64() - profile_a_wait_start;
#endif

                // Read SF (must precede warpgroup_arrive)
                float scale_a_0_lo, scale_a_1_lo;
                float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
                if (wg_has_valid_rows and
                    block_phase == sched::BlockPhase::Linear1) {
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + wg_m_offset + r_0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + wg_m_offset + r_1);
                } else if constexpr (kL2ActsSFGranK == 64) {
                    if (wg_has_valid_rows) {
                    // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + wg_m_offset + r_0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + wg_m_offset + r_1);
                    scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + wg_m_offset + r_0);
                    scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + wg_m_offset + r_1);
                    }
                }

                // ----- FP4-to-E4M3 dequant of the packed weight tile -----
                // The packed FP4 tile in `smem_b_packed[stage_idx]` is decoded
                // into the E4M3 tile in `smem_b[stage_idx]` with the per-32
                // UE8M0 SFB baked in via the constant FP4-to-E4M3 LUT.
                // After this call, `smem_b[stage]` is byte-equivalent to a
                // pre-scaled FP8 weight tile: the subsequent SS-mode WGMMA
                // accumulator already includes SFB, and only SFA needs to be
                // applied in the promote loop below.
                //
                // Non-epilogue warps assist the math warpgroup. Decode work
                // is partitioned over the assist threads plus the
                // epilogue/math threads, then all participants rendezvous
                // before WGMMA reads the decoded shared tile.
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                const uint64_t profile_decode_wait_start =
                    profile_math_leader ? clock64() : 0;
#endif
                if constexpr (kUseEarlyBDecode)
                    wait_fp4_decode_input_ready(stage_idx, phase);
                const bool math_warp_decodes =
                    epilogue_warp_idx < kNumMathWGDecodeWarps;
                if constexpr (kNumMathWGDecodeWarps > 0) {
                    if (math_warp_decodes) {
                        const uint32_t decode_thread_idx =
                            kNumFP4DecodeAssistThreads + epilogue_thread_idx;
                        dequant_fp4_b_tile_to_e4m3_smem_dispatch<
                            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK,
                            kUseWideLoadDecode>(
                            decode_thread_idx, kNumFP4DecodeWorkerThreads,
                            smem_b_packed[stage_idx], smem_b[stage_idx],
                            smem_sfb[stage_idx]);
                    }
                }
                if constexpr (kNumMathWGDecodeWarps > 0) {
                    if (math_warp_decodes)
                        arrive_or_sync_fp4_decode_done(stage_idx);
                    if constexpr (kUseDecodeDoneMBarrier) {
                        wait_fp4_decode_done(stage_idx, phase);
                    } else {
                        if (!math_warp_decodes)
                            wait_fp4_decode_done(stage_idx, phase);
                    }
                } else {
                    wait_fp4_decode_done(stage_idx, phase);
                }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                if (profile_math_leader)
                    profile_decode_wait_cycles +=
                        clock64() - profile_decode_wait_start;
#endif

                if (not wg_has_valid_rows) {
                    // Return this warp's stage credit exactly as the WGMMA
                    // path does.  The active split-M warpgroup remains the
                    // final consumer, so the loader cannot recycle the stage
                    // before its reads complete.
                    if (lane_idx == 0)
                        empty_barriers[stage_idx]->arrive();
                    continue;
                }

                if (block_phase == sched::BlockPhase::Linear1) {
                    if constexpr (kSwapABL1Active) {
                        // L1 swapAB: WGMMA-M is the 64-row weight slice owned by
                        // this split-N WG; WGMMA-N is the valid token count,
                        // bucketed to Hopper's 8-column granularity.
                        auto run_swap_ab_l1 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            constexpr uint32_t kSwapAccumStride =
                                kAccumPerThread / kSwapABNSubtiles;
                            DG_STATIC_ASSERT(
                                kSwapAccum <= kSwapAccumStride,
                                "swap accumulator does not fit N64 subtile slot");

                            #pragma unroll
                            for (uint32_t sub = 0;
                                 sub < kSwapABNSubtiles; ++ sub) {
                                float swap_accum[kSwapAccum];
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + smem_b_wg_offset +
                                            sub * 64 * BLOCK_K +
                                            k * SwapWGMMA::K,
                                        1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k * SwapWGMMA::K,
                                        1);
                                    SwapWGMMA::wgmma(
                                        desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();

                                const uint32_t accum_base =
                                    sub * kSwapAccumStride;
                                #pragma unroll
                                for (uint32_t i = 0;
                                     i < kSwapAccum / 4; ++ i) {
                                    const uint32_t token_0 =
                                        i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
#if defined(DG_MEGA_MOE_FP4_SWAP_SCALE_FLOAT2_LEVEL) && \
    (DG_MEGA_MOE_FP4_SWAP_SCALE_FLOAT2_LEVEL & 1)
                                    const float2 scale = ptx::ld_shared(
                                        reinterpret_cast<const float2*>(
                                            smem_sfa[stage_idx] + token_0));
                                    if (token_0 < valid_m) {
                                        final_accum[accum_base + i * 4 + 0] +=
                                            scale.x * swap_accum[i * 4 + 0];
                                        final_accum[accum_base + i * 4 + 2] +=
                                            scale.x * swap_accum[i * 4 + 2];
                                    }
                                    if (token_1 < valid_m) {
                                        final_accum[accum_base + i * 4 + 1] +=
                                            scale.y * swap_accum[i * 4 + 1];
                                        final_accum[accum_base + i * 4 + 3] +=
                                            scale.y * swap_accum[i * 4 + 3];
                                    }
#else
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + token_0);
                                        final_accum[accum_base + i * 4 + 0] +=
                                            scale_0 * swap_accum[i * 4 + 0];
                                        final_accum[accum_base + i * 4 + 2] +=
                                            scale_0 * swap_accum[i * 4 + 2];
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + token_1);
                                        final_accum[accum_base + i * 4 + 1] +=
                                            scale_1 * swap_accum[i * 4 + 1];
                                        final_accum[accum_base + i * 4 + 3] +=
                                            scale_1 * swap_accum[i * 4 + 3];
                                    }
#endif
                                }
                            }

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();
                        };

                        run_swap_ab_l1.template operator()<kNSwap>();
                    } else if constexpr (kDirectAccumulator) {
                        if (k_block_idx != 0)
                            direct_rescale_final(
                                direct_prev_scale_0, direct_prev_scale_1,
                                scale_a_0_lo, scale_a_1_lo);

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(final_accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0;
                             k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + smem_a_wg_offset +
                                    k * WGMMA::K,
                                1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + smem_b_wg_offset +
                                    k * WGMMA::K,
                                1);
                            WGMMA::wgmma(
                                desc_a, desc_b, final_accum, true);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(final_accum[i]);
                        ptx::warpgroup_wait<0>();

                        if (lane_idx == 0)
                            empty_barriers[stage_idx]->arrive();
                        direct_prev_scale_0 = scale_a_0_lo;
                        direct_prev_scale_1 = scale_a_1_lo;
                    } else {
                    // Single per-128 K-block WGMMA group
                    if constexpr (kSSNSplitActive) {
                        float accum[kSSAccum];
                        #pragma unroll
                        for (uint32_t nh = 0; nh < 2; ++ nh) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSSHalfAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / SSHalfWGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * SSHalfWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset
                                        + nh * (WG_BLOCK_N / 2) * BLOCK_K
                                        + k * SSHalfWGMMA::K, 1);
                                SSHalfWGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kSSHalfAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kSSHalfAccum / 4; ++ i) {
                                const uint32_t f = nh * kSSHalfAccum + i * 4;
                                final_accum[f+0] += scale_a_0_lo * accum[i*4+0];
                                final_accum[f+1] += scale_a_0_lo * accum[i*4+1];
                                final_accum[f+2] += scale_a_1_lo * accum[i*4+2];
                                final_accum[f+3] += scale_a_1_lo * accum[i*4+3];
                            }
                        }
                        if (lane_idx == 0)
                            empty_barriers[stage_idx]->arrive();
                    } else {
                        float accum[kAccumPerThread];
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();

                        if (lane_idx == 0)
                            empty_barriers[stage_idx]->arrive();

                        // L1: SFB is already baked into the decoded E4M3 tile,
                        // so only SFA remains.
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                        }
                    }
                    }
                } else {
                    if constexpr (kSwapABL2Active) {
                        DG_STATIC_ASSERT(kL2ActsSFGranK == 64,
                                         "L2 swapAB assumes per-64 activation scales");
                        auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            constexpr uint32_t kSwapAccumStride =
                                kAccumPerThread / kSwapABNSubtiles;
                            DG_STATIC_ASSERT(
                                kSwapAccum <= kSwapAccumStride,
                                "swap accumulator does not fit N64 subtile slot");

                            #pragma unroll
                            for (uint32_t sub = 0;
                                 sub < kSwapABNSubtiles; ++ sub) {
                                float swap_accum[kSwapAccum];
                                const uint32_t accum_base =
                                    sub * kSwapAccumStride;

                                auto promote_swap_accum =
                                    [&](const uint32_t& sf_group) {
                                    #pragma unroll
                                    for (uint32_t i = 0;
                                         i < kSwapAccum / 4; ++ i) {
                                        const uint32_t token_0 =
                                            i * 8 + col_idx * 2;
                                        const uint32_t token_1 = token_0 + 1;
#if defined(DG_MEGA_MOE_FP4_SWAP_SCALE_FLOAT2_LEVEL) && \
    (DG_MEGA_MOE_FP4_SWAP_SCALE_FLOAT2_LEVEL & 2)
                                        const float2 scale = ptx::ld_shared(
                                            reinterpret_cast<const float2*>(
                                                smem_sfa[stage_idx] +
                                                sf_group * BLOCK_M + token_0));
                                        if (token_0 < valid_m) {
                                            final_accum[
                                                accum_base + i * 4 + 0] +=
                                                scale.x * swap_accum[i * 4 + 0];
                                            final_accum[
                                                accum_base + i * 4 + 2] +=
                                                scale.x * swap_accum[i * 4 + 2];
                                        }
                                        if (token_1 < valid_m) {
                                            final_accum[
                                                accum_base + i * 4 + 1] +=
                                                scale.y * swap_accum[i * 4 + 1];
                                            final_accum[
                                                accum_base + i * 4 + 3] +=
                                                scale.y * swap_accum[i * 4 + 3];
                                        }
#else
                                        if (token_0 < valid_m) {
                                            const float scale_0 = ptx::ld_shared(
                                                smem_sfa[stage_idx] +
                                                sf_group * BLOCK_M + token_0);
                                            final_accum[
                                                accum_base + i * 4 + 0] +=
                                                scale_0 * swap_accum[i * 4 + 0];
                                            final_accum[
                                                accum_base + i * 4 + 2] +=
                                                scale_0 * swap_accum[i * 4 + 2];
                                        }
                                        if (token_1 < valid_m) {
                                            const float scale_1 = ptx::ld_shared(
                                                smem_sfa[stage_idx] +
                                                sf_group * BLOCK_M + token_1);
                                            final_accum[
                                                accum_base + i * 4 + 1] +=
                                                scale_1 * swap_accum[i * 4 + 1];
                                            final_accum[
                                                accum_base + i * 4 + 3] +=
                                                scale_1 * swap_accum[i * 4 + 3];
                                        }
#endif
                                    }
                                };

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + smem_b_wg_offset +
                                            sub * 64 * BLOCK_K +
                                            k * SwapWGMMA::K,
                                        1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k * SwapWGMMA::K,
                                        1);
                                    SwapWGMMA::wgmma(
                                        desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                promote_swap_accum(0);

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
                                    const uint32_t k_off =
                                        (BLOCK_K / 2) + k * SwapWGMMA::K;
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + smem_b_wg_offset +
                                            sub * 64 * BLOCK_K + k_off,
                                        1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k_off, 1);
                                    SwapWGMMA::wgmma(
                                        desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                promote_swap_accum(1);
                            }

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();
                        };

                        run_swap_ab_l2.template operator()<kNSwap>();
                    } else if constexpr (kDirectAccumulator) {
                        DG_STATIC_ASSERT(
                            kL2ActsSFGranK == 64,
                            "Direct M128/N128 L2 expects per-64 activation scales");
                        if (k_block_idx != 0)
                            direct_rescale_final(
                                direct_prev_scale_0, direct_prev_scale_1,
                                scale_a_0_lo, scale_a_1_lo);

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(final_accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0;
                             k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + smem_a_wg_offset +
                                    k * WGMMA::K,
                                1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + smem_b_wg_offset +
                                    k * WGMMA::K,
                                1);
                            WGMMA::wgmma(
                                desc_a, desc_b, final_accum, true);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(final_accum[i]);
                        ptx::warpgroup_wait<0>();

                        direct_rescale_final(
                            scale_a_0_lo, scale_a_1_lo,
                            scale_a_0_hi, scale_a_1_hi);
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(final_accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0;
                             k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                            const uint32_t k_off =
                                BLOCK_K / 2 + k * WGMMA::K;
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + smem_a_wg_offset + k_off,
                                1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + smem_b_wg_offset + k_off,
                                1);
                            WGMMA::wgmma(
                                desc_a, desc_b, final_accum, true);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(final_accum[i]);
                        ptx::warpgroup_wait<0>();

                        if (lane_idx == 0)
                            empty_barriers[stage_idx]->arrive();
                        direct_prev_scale_0 = scale_a_0_hi;
                        direct_prev_scale_1 = scale_a_1_hi;
                    } else if constexpr (kL2ActsSFGranK == 32) {
                        // L2 BLOCK_N=64: L1 produced 32-column FP8 chunks with
                        // independent SF, so promote each WGMMA::K=32 slice with
                        // its own activation scale.
                        float accum[kAccumPerThread];
                        #pragma unroll
                        for (uint32_t sf_group = 0; sf_group < kNumL2SFAPerBlockK; ++ sf_group) {
                            const float scale_a_0 = ptx::ld_shared(
                                smem_sfa[stage_idx] + sf_group * BLOCK_M + wg_m_offset + r_0);
                            const float scale_a_1 = ptx::ld_shared(
                                smem_sfa[stage_idx] + sf_group * BLOCK_M + wg_m_offset + r_1);
                            const uint32_t k_off = sf_group * WGMMA::K;
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, false);
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0 * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0 * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1 * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1 * accum[i*4+3];
                            }
                        }

                        if (lane_idx == 0)
                            empty_barriers[stage_idx]->arrive();
                    } else {
                        if constexpr (kSSNSplitActive) {
                            // L2 per-64 SFA with split-N WGMMA: each N half owns a
                            // 32-float accumulator, then promotes into its slice of
                            // final_accum before the next half reuses the accumulator.
                            float accum[kSSAccum];
                            #pragma unroll
                            for (uint32_t nh = 0; nh < 2; ++ nh) {
                                const uint32_t n_off = nh * (WG_BLOCK_N / 2) * BLOCK_K;
                                const uint32_t fbase = nh * kSSHalfAccum;

                                // First K half: K=0..63, SFA = scale_a_*_lo
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < (BLOCK_K / 2) / SSHalfWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + smem_a_wg_offset + k * SSHalfWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + smem_b_wg_offset + n_off + k * SSHalfWGMMA::K, 1);
                                    SSHalfWGMMA::wgmma(desc_a, desc_b, accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();

                                // L2 first half: SFB baked into decoded E4M3 tile.
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSHalfAccum / 4; ++ i) {
                                    final_accum[fbase+i*4+0] += scale_a_0_lo * accum[i*4+0];
                                    final_accum[fbase+i*4+1] += scale_a_0_lo * accum[i*4+1];
                                    final_accum[fbase+i*4+2] += scale_a_1_lo * accum[i*4+2];
                                    final_accum[fbase+i*4+3] += scale_a_1_lo * accum[i*4+3];
                                }

                                // Second K half: K=64..127, SFA = scale_a_*_hi
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < (BLOCK_K / 2) / SSHalfWGMMA::K; ++ k) {
                                    const uint32_t k_off = (BLOCK_K / 2) + k * SSHalfWGMMA::K;
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + smem_b_wg_offset + n_off + k_off, 1);
                                    SSHalfWGMMA::wgmma(desc_a, desc_b, accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();

                                // L2 second half: SFB baked into decoded E4M3 tile.
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSHalfAccum / 4; ++ i) {
                                    final_accum[fbase+i*4+0] += scale_a_0_hi * accum[i*4+0];
                                    final_accum[fbase+i*4+1] += scale_a_0_hi * accum[i*4+1];
                                    final_accum[fbase+i*4+2] += scale_a_1_hi * accum[i*4+2];
                                    final_accum[fbase+i*4+3] += scale_a_1_hi * accum[i*4+3];
                                }
                            }

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();
                        } else {
                            float accum[kAccumPerThread];
                            // L2: split BLOCK_K=128 into two halves (per-64 SFA), each 2 WGMMAs.
                            // First half: K=0..63, SFA = scale_a_*_lo
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            // L2 first half: SFB baked into decoded E4M3 tile.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                            }

                            // Second half: K=64..127, SFA = scale_a_*_hi
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            // L2 second half: SFB baked into decoded E4M3 tile.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0_hi * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_hi * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_hi * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_hi * accum[i*4+3];
                            }
                        }
                    }
                }
            }
            };

            if constexpr (kSwapABEligible) {
                const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                // Bucket availability is a GEMM policy, independent of the
                // number of experts scheduled in a wave.
                if (n_swap <= 8) {
                    run_k_stages.template operator()<8>();
                } else if (n_swap <= 16) {
                    run_k_stages.template operator()<16>();
#ifdef DG_MEGA_MOE_FP4_SWAP_AB_N24
                } else if (n_swap <= 24) {
                    run_k_stages.template operator()<24>();
#endif
                } else if (n_swap <= 32) {
                    run_k_stages.template operator()<32>();
#if defined(DG_MEGA_MOE_FP4_SWAP_AB_FINE_BUCKETS) && DG_MEGA_MOE_FP4_SWAP_AB_FINE_BUCKETS >= 1
                } else if (n_swap <= 40) {
                    run_k_stages.template operator()<40>();
#endif
#if defined(DG_MEGA_MOE_FP4_SWAP_AB_FINE_BUCKETS) && DG_MEGA_MOE_FP4_SWAP_AB_FINE_BUCKETS >= 2
                } else if (n_swap <= 48) {
                    run_k_stages.template operator()<48>();
#endif
#if defined(DG_MEGA_MOE_FP4_SWAP_AB_FINE_BUCKETS) && DG_MEGA_MOE_FP4_SWAP_AB_FINE_BUCKETS >= 3
                } else if (n_swap <= 56) {
                    run_k_stages.template operator()<56>();
#endif
                } else {
                    run_k_stages.template operator()<64>();
                }
            } else {
                run_k_stages.template operator()<0>();
            }
            if constexpr (kDirectAccumulator) {
                if (num_k_blocks != 0 and wg_has_valid_rows)
                    direct_postscale_final(
                        direct_prev_scale_0, direct_prev_scale_1);
            }

            // L2 is the last consumer of both physical compute slots.  One
            // retirement is published per L2 N-block after every math
            // warpgroup has finished reading the tile.  The generation count
            // lets a later logical pool block safely reuse the same address.
            if constexpr (kL1RingEnabled or kL2RingEnabled) {
                if (block_phase == sched::BlockPhase::Linear2) {
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        if constexpr (kL1RingEnabled)
                            ptx::red_add_rel(
                                workspace.get_l1_ring_empty_count_ptr(
                                    get_l1_ring_block_idx(pool_block_idx)),
                                1);
                        if constexpr (kL2RingEnabled)
                            ptx::red_add_rel(
                                workspace.get_l2_ring_empty_count_ptr(
                                    get_l2_ring_block_idx(pool_block_idx)),
                                1);
                    }
                    __syncwarp();
                }
            }

            // Skip epilogue when block is past valid M (still must release via empty).
            // In split-N mode, `wg_m_offset` is 0 for all WGs (they share the same M
            // rows), so this skip is effectively per-block, not per-WG.
            if (wg_m_offset >= valid_m) {
                // Trigger any combine/sync logic minimally
                if (block_phase == sched::BlockPhase::Linear1)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                else
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            const uint32_t row_offset_r0 = wg_m_offset + r_0;
            const uint32_t row_offset_r1 = wg_m_offset + r_1;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;

            if (block_phase == sched::BlockPhase::Linear1) {
                if constexpr (kL2RingEnabled) {
                    const uint32_t empty_target =
                        get_l2_ring_wave_idx(pool_block_idx) *
                        (L2_SHAPE_N / BLOCK_N);
                    const auto empty_ptr = workspace.get_l2_ring_empty_count_ptr(
                        get_l2_ring_block_idx(pool_block_idx));
                    while (ptx::ld_acq(empty_ptr) < empty_target);
                }
                if constexpr (kSwapABL1Active) {
                    if constexpr (kSwapABNSubtiles == 2) {
                    // N256/internal-N64: each WG owns one N128 weight slice,
                    // represented as two adjacent N64 swap subtiles.  Unlike
                    // the N128 outer tile, each WG now produces a complete
                    // 64-column activation-SF group, so quantization and the
                    // TMA store can remain WG-local.
                    DG_STATIC_ASSERT(not kSwapABFastAmaxActive,
                                     "N256 A/B starts with the proven FP32 staging epilogue");
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath
                            ? math::fast_rcp(1.0f + e)
                            : 1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp !=
                                      cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp !=
                                      cute::numeric_limits<float>::infinity())
                            x = cute::min(
                                cute::max(x, -kActivationClamp),
                                kActivationClamp);
                    };

                    constexpr uint32_t kSwapSmemRowStride =
                        WG_L1_OUT_BLOCK_N;
                    constexpr uint32_t kSwapAccumStride =
                        kAccumPerThread / kSwapABNSubtiles;
                    const uint32_t swap_smem_wg_base =
                        epilogue_wg_idx * BLOCK_M * WG_L1_OUT_BLOCK_N;

                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        #pragma unroll
                        for (uint32_t sub = 0;
                             sub < kSwapABNSubtiles; ++ sub) {
                            const uint32_t accum_base =
                                sub * kSwapAccumStride;
                            const uint32_t local_out_col =
                                sub * 32 + warp_idx_in_wg * 8 + row_idx;
                            if (token_0 < valid_m) {
                                float gate =
                                    final_accum[accum_base + i * 4 + 0];
                                float up =
                                    final_accum[accum_base + i * 4 + 2];
                                clamp_gate(gate);
                                clamp_up(up);
                                smem_cd_swap_l1_fp32[
                                    swap_smem_wg_base +
                                    token_0 * kSwapSmemRowStride +
                                    local_out_col] = silu(gate) * up;
                            }
                            if (token_1 < valid_m) {
                                float gate =
                                    final_accum[accum_base + i * 4 + 1];
                                float up =
                                    final_accum[accum_base + i * 4 + 3];
                                clamp_gate(gate);
                                clamp_up(up);
                                smem_cd_swap_l1_fp32[
                                    swap_smem_wg_base +
                                    token_1 * kSwapSmemRowStride +
                                    local_out_col] = silu(gate) * up;
                            }
                        }
                    };

                    const uint32_t num_swap_token_chunks =
                        (valid_m + 7u) / 8u;
                    store_l1_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1;
                             i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l1_swap_chunk(i);
                        }
                    }

                    ptx::sync_aligned(
                        128,
                        kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    const uint32_t swap_quant_thread_idx =
                        warp_idx_in_wg * 32 + lane_idx;
                    constexpr uint32_t kSwapQuantThreads = 128;
                    for (uint32_t token = swap_quant_thread_idx;
                         token < valid_m;
                         token += kSwapQuantThreads) {
                        float amax = 0.0f;
                        #pragma unroll
                        for (uint32_t col = 0;
                             col < kSwapSmemRowStride; ++ col) {
                            const float v = smem_cd_swap_l1_fp32[
                                swap_smem_wg_base +
                                token * kSwapSmemRowStride + col];
                            amax = cute::max(amax, cute::abs(v));
                        }
                        const float topk_weight =
                            *l1_topk_weights_buffer
                                .get_data_buffer(l1_ring_m_idx + token)
                                .get_base_ptr<float>();
                        amax *= cute::abs(topk_weight);
                        float2 amax_pair = {amax, amax};
                        float2 sf_pair, sf_inv_pair;
                        sm90_fp8_fp4_mega_moe_get_e4m3_sf_and_sf_inv(
                            amax_pair, sf_pair, sf_inv_pair);
                        const float sf_inv =
                            topk_weight * sf_inv_pair.x;

                        const uint32_t sf_n_block_idx_local =
                            n_block_idx * kWarpgroupSplitN +
                            epilogue_wg_n_idx;
                        l2_sf_buffer.get_base_ptr<float>()[
                            sf_n_block_idx_local *
                                kNumL2SFStorageTokens +
                            l2_ring_sf_m_idx + token] = sf_pair.x;

                        #pragma unroll
                        for (uint32_t col = 0;
                             col < kSwapSmemRowStride; col += 2) {
                            const float v0 = smem_cd_swap_l1_fp32[
                                swap_smem_wg_base +
                                token * kSwapSmemRowStride + col] * sf_inv;
                            const float v1 = smem_cd_swap_l1_fp32[
                                swap_smem_wg_base +
                                token * kSwapSmemRowStride + col + 1] * sf_inv;
                            const __nv_fp8x2_e4m3 pair(
                                make_float2(v0, v1));
                            *reinterpret_cast<uint16_t*>(
                                smem_cd_swap_l1_fp8 +
                                swap_smem_wg_base +
                                token * kSwapSmemRowStride + col) = pair.__x;
                        }
                    }

                    ptx::sync_aligned(
                        128,
                        kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_swap_l1_fp8 + swap_smem_wg_base,
                            n_block_idx * L1_OUT_BLOCK_N +
                                wg_l1_out_n_offset,
                            l2_ring_m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();

                    if constexpr (kL2ArrivalCounter) {
                        if (warp_idx_in_wg == 0 and
                            cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(
                                    workspace.get_l2_arrival_mask_ptr(
                                        pool_block_idx)),
                                1);
                        }
                    } else {
                        ptx::sync_aligned(
                            kNumEpilogueThreads,
                            kEpilogueFullBarrierIdx);
                        if (epilogue_warp_idx == 0 and
                            cute::elect_one_sync()) {
                            ptx::red_or_rel_gpu(
                                workspace.get_l2_arrival_mask_ptr(
                                    pool_block_idx),
                                1ull << n_block_idx);
                        }
                    }
                    __syncwarp();
                    if constexpr (kL2ArrivalCounter)
                        ptx::sync_aligned(
                            kNumEpilogueThreads,
                            kEpilogueFullBarrierIdx);
                    } else {
                    // Each split-N WG writes its disjoint 32-column range into
                    // a shared token-major tile. The default path stages FP32,
                    // then scans complete token rows for amax before quantizing.
                    // The fast-amax path keeps each thread's small token slice in
                    // registers, publishes per-token partial amax, and writes FP8
                    // directly after the cross-warp reduction.
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };

                    const uint32_t out_col_base =
                        wg_l1_out_n_offset + warp_idx_in_wg * 8 + row_idx;
                    float swap_v0[kSwapABTokenChunks] = {};
                    float swap_v1[kSwapABTokenChunks] = {};
                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        const bool valid_token_0 = token_0 < valid_m;
                        const bool valid_token_1 = token_1 < valid_m;

                        float v0 = 0.0f;
                        if (valid_token_0) {
                            float g0 = final_accum[i * 4 + 0];
                            float u0 = final_accum[i * 4 + 2];
                            clamp_gate(g0);
                            clamp_up(u0);
                            const float weight_0 = *l1_topk_weights_buffer
                                .get_data_buffer(l1_ring_m_idx + token_0)
                                .get_base_ptr<float>();
                            if constexpr (kSwapABFastAmaxActive) {
                                v0 = silu(g0) * u0 * weight_0;
                                swap_v0[i] = v0;
                            } else {
                                v0 = silu(g0) * u0;
                                smem_cd_swap_l1_fp32[token_0 * L1_OUT_BLOCK_N + out_col_base] = v0;
                            }
                        }

                        float v1 = 0.0f;
                        if (valid_token_1) {
                            float g1 = final_accum[i * 4 + 1];
                            float u1 = final_accum[i * 4 + 3];
                            clamp_gate(g1);
                            clamp_up(u1);
                            const float weight_1 = *l1_topk_weights_buffer
                                .get_data_buffer(l1_ring_m_idx + token_1)
                                .get_base_ptr<float>();
                            if constexpr (kSwapABFastAmaxActive) {
                                v1 = silu(g1) * u1 * weight_1;
                                swap_v1[i] = v1;
                            } else {
                                v1 = silu(g1) * u1;
                                smem_cd_swap_l1_fp32[token_1 * L1_OUT_BLOCK_N + out_col_base] = v1;
                            }
                        }

                        if constexpr (kSwapABFastAmaxActive) {
                            const float amax0 = math::warp_reduce<4, true>(
                                cute::abs(v0), math::ReduceMax<float>());
                            const float amax1 = math::warp_reduce<4, true>(
                                cute::abs(v1), math::ReduceMax<float>());
                            if (row_idx == 0) {
                                if (valid_token_0)
                                    smem_cd_swap_l1_amax[token_0 * kNumEpilogueWarps + epilogue_warp_idx] = amax0;
                                if (valid_token_1)
                                    smem_cd_swap_l1_amax[token_1 * kNumEpilogueWarps + epilogue_warp_idx] = amax1;
                            }
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    store_l1_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l1_swap_chunk(i);
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if constexpr (kSwapABFastAmaxActive) {
                        for (uint32_t token = epilogue_thread_idx;
                             token < valid_m;
                             token += kNumEpilogueThreads) {
                            float amax = 0.0f;
                            #pragma unroll
                            for (uint32_t w = 0; w < kNumEpilogueWarps; ++ w)
                                amax = cute::max(
                                    amax, smem_cd_swap_l1_amax[token * kNumEpilogueWarps + w]);
                            float2 amax_pair = {amax, amax};
                            float2 sf_pair, sf_inv_pair;
                            sm90_fp8_fp4_mega_moe_get_e4m3_sf_and_sf_inv(
                                amax_pair, sf_pair, sf_inv_pair);

                            auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                            const uint32_t token_idx =
                                l2_ring_sf_m_idx + token;
                            sf_base_ptr[
                                n_block_idx * kNumL2SFStorageTokens +
                                token_idx] = sf_pair.x;
                            ptx::st_shared(smem_amax_scratch + token, __float_as_uint(sf_inv_pair.x));
                        }

                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            const uint32_t token_0 = i * 8 + col_idx * 2;
                            const uint32_t token_1 = token_0 + 1;
                            if (token_0 < valid_m) {
                                const float sf_inv = __uint_as_float(
                                    ptx::ld_shared(smem_amax_scratch + token_0));
                                const __nv_fp8_e4m3 q(swap_v0[i] * sf_inv);
                                reinterpret_cast<uint8_t*>(
                                    smem_cd_swap_l1_fp8)[token_0 * L1_OUT_BLOCK_N + out_col_base] =
                                        *reinterpret_cast<const uint8_t*>(&q);
                            }
                            if (token_1 < valid_m) {
                                const float sf_inv = __uint_as_float(
                                    ptx::ld_shared(smem_amax_scratch + token_1));
                                const __nv_fp8_e4m3 q(swap_v1[i] * sf_inv);
                                reinterpret_cast<uint8_t*>(
                                    smem_cd_swap_l1_fp8)[token_1 * L1_OUT_BLOCK_N + out_col_base] =
                                        *reinterpret_cast<const uint8_t*>(&q);
                            }
                        }
                    } else {
                        for (uint32_t token = epilogue_thread_idx; token < valid_m; token += kNumEpilogueThreads) {
                            float amax = 0.0f;
                            #pragma unroll
                            for (uint32_t col = 0; col < L1_OUT_BLOCK_N; ++ col) {
                                const float v = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col];
                                amax = cute::max(amax, cute::abs(v));
                            }
                            const float wtok = *l1_topk_weights_buffer
                                .get_data_buffer(l1_ring_m_idx + token)
                                .get_base_ptr<float>();
                            amax *= cute::abs(wtok);
                            float2 amax_pair = {amax, amax};
                            float2 sf_pair, sf_inv_pair;
                            sm90_fp8_fp4_mega_moe_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                            const float sf = sf_pair.x;
                            const float sf_inv = wtok * sf_inv_pair.x;

                            auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                            const uint32_t token_idx =
                                l2_ring_sf_m_idx + token;
                            sf_base_ptr[
                                n_block_idx * kNumL2SFStorageTokens +
                                token_idx] = sf;

                            #pragma unroll
                            for (uint32_t col = 0; col < L1_OUT_BLOCK_N; col += 2) {
                                const float v0 = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 0] * sf_inv;
                                const float v1 = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 1] * sf_inv;
                                const __nv_fp8x2_e4m3 pair(make_float2(v0, v1));
                                auto* ptr = reinterpret_cast<uint16_t*>(
                                    smem_cd_swap_l1_fp8 + token * L1_OUT_BLOCK_N + col);
                                *ptr = pair.__x;
                            }
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_swap_l1_fp8,
                            n_block_idx * L1_OUT_BLOCK_N,
                            l2_ring_m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();

                    if constexpr (kL2ArrivalCounter) {
                        if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)),
                                kWarpgroupSplitN);
                        }
                    } else {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            ptx::red_or_rel_gpu(
                                workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                                1ull << n_block_idx);
                        }
                    }
                    __syncwarp();
                    if constexpr (kL2ArrivalCounter)
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
                } else {
                // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                // Layout in `final_accum`:
                //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                //   Pair `p` in [0, 8): gate chunk = 2p, up chunk = 2p+1.
                //
                // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                constexpr uint32_t kNumPairs = kAccumPerThread / 8;  // 8 for BLOCK_N=128
                float swiglu_r0[kNumPairs][2];
                float swiglu_r1[kNumPairs][2];

                // Per-row amax across all 8 pairs
                float amax_r0 = 0.0f, amax_r1 = 0.0f;

                // Compute SwiGLU + per-pair amax
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const uint32_t gate = 2 * p, up = 2 * p + 1;

                    // Apply optional clamp on gate / up before SwiGLU
                    // Match SM100 reference: gate is clamped only on the upper
                    // side (very-negative gate is fine because SiLU(-inf) -> 0),
                    // while up is clamped both sides.
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };
                    float g_r0_c0 = final_accum[gate*4 + 0]; clamp_gate(g_r0_c0);
                    float g_r0_c1 = final_accum[gate*4 + 1]; clamp_gate(g_r0_c1);
                    float g_r1_c0 = final_accum[gate*4 + 2]; clamp_gate(g_r1_c0);
                    float g_r1_c1 = final_accum[gate*4 + 3]; clamp_gate(g_r1_c1);
                    float u_r0_c0 = final_accum[up*4   + 0]; clamp_up(u_r0_c0);
                    float u_r0_c1 = final_accum[up*4   + 1]; clamp_up(u_r0_c1);
                    float u_r1_c0 = final_accum[up*4   + 2]; clamp_up(u_r1_c0);
                    float u_r1_c1 = final_accum[up*4   + 3]; clamp_up(u_r1_c1);

                    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };

                    if (valid_r0) {
                        swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                        swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                        amax_r0 = cute::max(amax_r0, cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                    } else {
                        swiglu_r0[p][0] = 0.0f;
                        swiglu_r0[p][1] = 0.0f;
                    }
                    if (valid_r1) {
                        swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                        swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                        amax_r1 = cute::max(amax_r1, cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                    } else {
                        swiglu_r1[p][0] = 0.0f;
                        swiglu_r1[p][1] = 0.0f;
                    }
                }

                // Fold topk weight into the row amax and final quantization scale.
                float weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                    .get_data_buffer(l1_ring_m_idx + row_offset_r0)
                    .get_base_ptr<float>() : 0.0f;
                float weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                    .get_data_buffer(l1_ring_m_idx + row_offset_r1)
                    .get_base_ptr<float>() : 0.0f;
                amax_r0 *= cute::abs(weight_r0);
                amax_r1 *= cute::abs(weight_r1);

                // Reduce amax across the 4 col-lanes that share the same row.
                // In WGMMA m64n128k32 output, the 4 lanes (`lane_idx & 3` differs,
                // `lane_idx >> 2` same) hold all N positions for the same r_0/r_1,
                // so we need an INTRA-group reduction (`xor 1, xor 2`), which is
                // `warp_reduce<4, false>`. Using `<4, true>` would instead merge
                // amax across 8 different rows -- giving wrong per-row SF.
                amax_r0 = math::warp_reduce<4, false>(amax_r0, math::ReduceMax<float>());
                amax_r1 = math::warp_reduce<4, false>(amax_r1, math::ReduceMax<float>());

                // Shared-SF split-N reduction: both N-half WGs publish their
                // per-row amax, synchronize once, then read both values and
                // take the max in registers.
                if constexpr (kSplitNSharesSF) {
                    if (col_idx == 0) {
                        const uint32_t row_slot = warp_idx_in_wg * 8 + row_idx;
                        const uint32_t slot = row_slot * 4 + epilogue_wg_n_idx * 2;
                        ptx::st_shared(smem_amax_scratch + slot + 0,
                                       __float_as_uint(amax_r0));
                        ptx::st_shared(smem_amax_scratch + slot + 1,
                                       __float_as_uint(amax_r1));
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    // Both WGs read the merged amax.
                    if (col_idx == 0) {
                        const uint32_t row_slot = warp_idx_in_wg * 8 + row_idx;
                        const uint32_t slot = row_slot * 4;
                        const float wg0_r0 = __uint_as_float(
                            ptx::ld_shared(smem_amax_scratch + slot + 0));
                        const float wg0_r1 = __uint_as_float(
                            ptx::ld_shared(smem_amax_scratch + slot + 1));
                        const float wg1_r0 = __uint_as_float(
                            ptx::ld_shared(smem_amax_scratch + slot + 2));
                        const float wg1_r1 = __uint_as_float(
                            ptx::ld_shared(smem_amax_scratch + slot + 3));
                        amax_r0 = cute::max(wg0_r0, wg1_r0);
                        amax_r1 = cute::max(wg0_r1, wg1_r1);
                    }
                    // Broadcast the merged col_idx==0 amax across the row's
                    // four column lanes; this is no longer a reduction after
                    // both split-N WGs have published their row max.
                    const int row_group_leader = static_cast<int>(lane_idx - col_idx);
                    amax_r0 = __shfl_sync(0xffffffff, amax_r0, row_group_leader);
                    amax_r1 = __shfl_sync(0xffffffff, amax_r1, row_group_leader);
                }

                // Compute SF and inverse SF for each row
                float sf_r0, sf_inv_r0;
                float sf_r1, sf_inv_r1;
                {
                    float2 amax_pair = {amax_r0, amax_r1};
                    float2 sf_pair, sf_inv_pair;
                    sm90_fp8_fp4_mega_moe_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                    sf_r0 = sf_pair.x; sf_inv_r0 = sf_inv_pair.x;
                    sf_r1 = sf_pair.y; sf_inv_r1 = sf_inv_pair.y;
                }
                const float weighted_sf_inv_r0 = weight_r0 * sf_inv_r0;
                const float weighted_sf_inv_r1 = weight_r1 * sf_inv_r1;

                // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                // The L1-output TMA store descriptor is built with swizzle_mode = 0
                // to match this plain row-major SMEM staging tile.
                //
                // Per pair `p`, each thread holds 4 FP8 values to write at:
                //   (row r_0, cols p*8 + col_idx*2 + {0,1})  -> packed as fp8x2 (2 bytes)
                //   (row r_1, cols p*8 + col_idx*2 + {0,1})  -> packed as fp8x2 (2 bytes)
                // The shared tile is either the joint SF-sharing tile or a compact
                // per-WG tile; WG_SMEM_CD_L1_STRIDE_N selects the matching row stride.
                auto* smem_cd_l1_wg = smem_cd_l1 + smem_cd_l1_wg_offset;
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const float v00 = swiglu_r0[p][0] * weighted_sf_inv_r0;
                    const float v01 = swiglu_r0[p][1] * weighted_sf_inv_r0;
                    const float v10 = swiglu_r1[p][0] * weighted_sf_inv_r1;
                    const float v11 = swiglu_r1[p][1] * weighted_sf_inv_r1;

                    const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                    const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

                    const uint32_t col = p * 8 + col_idx * 2;
                    auto* p0 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_0 * WG_SMEM_CD_L1_STRIDE_N + col);
                    auto* p1 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_1 * WG_SMEM_CD_L1_STRIDE_N + col);
                    if (valid_r0)
                        *p0 = r0_pair.__x;
                    if (valid_r1)
                        *p1 = r1_pair.__x;
                }

                // Write SF as float at `[token, k_sf_idx]` in the L2 acts SF buffer.
                // Each row is contributed by lanes col_idx in [0, 3]; only col_idx == 0 writes.
                // Shared-SF writes once from the first N-half WG.
                const bool sf_writer = (col_idx == 0) and
                    (not kSplitNSharesSF or epilogue_wg_n_idx == 0);
                if (sf_writer) {
                    auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                    // SF buffer is (kNumL2SFStorageTokens x
                    // kIntermediateHidden/granularity), MN-major.
                    const uint32_t token_r0 =
                        l2_ring_sf_m_idx + row_offset_r0;
                    const uint32_t token_r1 =
                        l2_ring_sf_m_idx + row_offset_r1;
                    // Shared-SF spans both split-N WGs; otherwise each WG owns
                    // a distinct local SF N block.
                    const uint32_t sf_n_block_idx_local = n_block_idx * kWarpgroupSplitN + epilogue_wg_n_idx;
                    const uint32_t k_sf_idx = kSplitNSharesSF ? n_block_idx : sf_n_block_idx_local;
                    if (valid_r0)
                        sf_base_ptr[k_sf_idx * kNumL2SFStorageTokens + token_r0] = sf_r0;
                    if (valid_r1)
                        sf_base_ptr[k_sf_idx * kNumL2SFStorageTokens + token_r1] = sf_r1;
                }

                // Combined split-N store needs an epilogue-wide sync so WG0 can
                // store the full L1 staging tile after WG1 writes its slice.
                if constexpr (kSplitNCombinesL1Store) {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                }

                // Issue TMA store of the entire tile. Padding rows beyond
                // `valid_m` are written with stale/garbage FP8 to the L1-output
                // pool buffer, but they are never consumed downstream: the L2
                // GEMM tile loads them, but its NVLink-scatter epilogue is
                // gated by `m_idx_in_block >= valid_m`, and stale SF in the
                // padding rows can produce NaN accumulators that simply stay
                // in registers (only valid rows are converted to BF16 and
                // STSM'd into smem). Using TMA for partial tiles is a large
                // win for low-batch / decode where every tile is partial.
                if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                    // In the 32-col/WG split-N path, WG0 stores the combined
                    // BLOCK_M x L1_OUT_BLOCK_N tile. Other paths store per-WG
                    // slices independently.
                    if constexpr (kSplitNCombinesL1Store) {
                        if (epilogue_wg_n_idx == 0) {
                            const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd_l1,
                                out_n_idx,
                                l2_ring_m_idx);
                            cute::tma_store_arrive();
                        }
                        // WG1 is covered by WG0's combined store.
                    } else {
                        // The TMA descriptor was built with box
                        // (WG_L1_OUT_BLOCK_N, l1_output_box_m=WG_BLOCK_M), so each
                        // WG advances column by `wg_l1_out_n_offset` (split-N) and
                        // row by `wg_m_offset` (split-M). In default single-WG
                        // mode both offsets are zero and this reduces to the
                        // historical `(n_block_idx * L1_OUT_BLOCK_N, m_idx)`.
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_offset;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1 + smem_cd_l1_wg_offset,
                            out_n_idx,
                            l2_ring_m_idx + wg_m_offset);
                        cute::tma_store_arrive();
                    }
                }
                __syncwarp();
                ptx::tma_store_wait<0>();

                // Notify L2 that this N block's L1 output (and SF) is ready
                if constexpr (kL2ArrivalCounter) {
                    if constexpr (kSplitNCombinesL1Store) {
                        // Only WG0 issues the combined TMA store. Once it
                        // drains, both N slices are visible, so a single
                        // +kWarpgroupSplitN keeps the counter expectation valid.
                        if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)),
                                kWarpgroupSplitN);
                        }
                    } else if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        ptx::red_add_rel(
                            reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)), 1);
                    }
                } else {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_or_rel_gpu(
                            workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                            1ull << n_block_idx);
                    }
                }
                __syncwarp();
                // Counter mode skips the bitmask path's epilogue-wide sync.
                // Add a tail sync so WG1 cannot overwrite the shared L1 staging
                // tile for the next N block while WG0's combined store drains.
                if constexpr (kSplitNCombinesL1Store and kL2ArrivalCounter)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                if constexpr (kSwapABL2Active) {
                    auto store_bf16 = [&](const uint32_t& token, const uint32_t& col, float value) {
                        smem_cd_l2[smem_cd_l2_wg_offset + token * BLOCK_N + col] =
                            __float2bfloat16_rn(value);
                    };

                    auto store_l2_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        constexpr uint32_t kSwapAccumStride =
                            kAccumPerThread / kSwapABNSubtiles;
                        #pragma unroll
                        for (uint32_t sub = 0;
                             sub < kSwapABNSubtiles; ++ sub) {
                            const uint32_t accum_base =
                                sub * kSwapAccumStride;
                            const uint32_t col_base = sub * 64;
                            if (token_0 < valid_m) {
                                store_bf16(
                                    token_0, col_base + r_0,
                                    final_accum[
                                        accum_base + i * 4 + 0]);
                                store_bf16(
                                    token_0, col_base + r_1,
                                    final_accum[
                                        accum_base + i * 4 + 2]);
                            }
                            if (token_1 < valid_m) {
                                store_bf16(
                                    token_1, col_base + r_0,
                                    final_accum[
                                        accum_base + i * 4 + 1]);
                                store_bf16(
                                    token_1, col_base + r_1,
                                    final_accum[
                                        accum_base + i * 4 + 3]);
                            }
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    store_l2_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l2_swap_chunk(i);
                        }
                    }
                } else if constexpr (not kSwapABL2Active) {
                    // STSM into smem_cd_l2 (BF16). Reuse SM100 column-swizzle layout.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                        // Each i consumes 8 floats (one 16x256b chunk in SM100 terms).
                        // For SM90 WGMMA layout, 8 floats per i correspond to 2 chunks of 4 floats:
                        //   final_accum[i*8 + (0..3)] = chunk 2i: (r0c0, r0c1, r1c0, r1c1)
                        //   final_accum[i*8 + (4..7)] = chunk 2i+1: same shape
                        const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;

                        // Write to SMEM at appropriate position
                        // Row r_0 cols [chunk_lo*8 + col_idx*2, chunk_lo*8 + col_idx*2 + 1] = r0_lo
                        // Row r_0 cols [chunk_hi*8 + col_idx*2, chunk_hi*8 + col_idx*2 + 1] = r0_hi
                        // Row r_1 cols [chunk_lo*8 + col_idx*2, chunk_lo*8 + col_idx*2 + 1] = r1_lo
                        // Row r_1 cols [chunk_hi*8 + col_idx*2, chunk_hi*8 + col_idx*2 + 1] = r1_hi
                        auto write_pair = [&](uint32_t row, uint32_t col, uint32_t packed) {
                            auto smem_ptr = smem_cd_l2
                                + smem_cd_l2_wg_offset
                                + row * WG_SMEM_CD_L2_STRIDE_N
                                + col;
                            // BF16 STS: 2 bf16 elements
                            *reinterpret_cast<uint32_t*>(smem_ptr) = packed;
                        };
                        if (valid_r0) {
                            const uint32_t r0_lo = math::cast_into_bf16_and_pack(
                                final_accum[chunk_lo*4 + 0], final_accum[chunk_lo*4 + 1]);
                            const uint32_t r0_hi = math::cast_into_bf16_and_pack(
                                final_accum[chunk_hi*4 + 0], final_accum[chunk_hi*4 + 1]);
                            write_pair(r_0, chunk_lo * 8 + col_idx * 2, r0_lo);
                            write_pair(r_0, chunk_hi * 8 + col_idx * 2, r0_hi);
                        }
                        if (valid_r1) {
                            const uint32_t r1_lo = math::cast_into_bf16_and_pack(
                                final_accum[chunk_lo*4 + 2], final_accum[chunk_lo*4 + 3]);
                            const uint32_t r1_hi = math::cast_into_bf16_and_pack(
                                final_accum[chunk_hi*4 + 2], final_accum[chunk_hi*4 + 3]);
                            write_pair(r_1, chunk_lo * 8 + col_idx * 2, r1_lo);
                            write_pair(r_1, chunk_hi * 8 + col_idx * 2, r1_hi);
                        }
                    }
                }

                {
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    if (profile_math_leader)
                        profile_scatter_block_start = clock64();
#endif
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Scatter to remote ranks via NVLink (one row per warp-pair)
                    // Each warpgroup-warp covers 8 unique rows x 2 (r_0 + r_1 doubled by warps)
                    // Lane group of 16 within a warp maps to one row.
                    const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                    const uint32_t lane_in_row = lane_idx % 16;
                    // In split-N each WG owns WG_BLOCK_N (= BLOCK_N/num_wg)
                    // columns of every row; in split-M each WG owns the full
                    // BLOCK_N columns of its WG_BLOCK_M rows. Either way the
                    // per-WG column footprint is WG_BLOCK_N.
                    constexpr uint32_t kColsPerScatterLane = WG_BLOCK_N / 16;
                    static_assert(WG_BLOCK_N % 16 == 0, "Scatter layout expects an even lane partition");
                    static_assert(kColsPerScatterLane == 4 or kColsPerScatterLane == 8,
                                  "L2 scatter currently supports WG_BLOCK_N=64 or 128");

                    uint32_t combine_stage_expert_base = 0;
                    if constexpr (kCombineStageRing) {
                        const bool has_internode_rows =
                            combine_stage_ring.has_internode_rows(
                                local_expert_idx);
                        if (lane_idx == 0 and has_internode_rows) {
                            combine_stage_expert_base =
                                combine_stage_ring.reserve_segment(
                                    local_expert_idx,
                                    scheduler.get_current_num_m_blocks() *
                                        BLOCK_M,
                                    kernel_launch_epoch);
                        }
                        combine_stage_expert_base = __shfl_sync(
                            0xffffffff, combine_stage_expert_base, 0);
                    }

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = wg_m_offset + row_in_wg;
                        if (m_idx_in_block >= valid_m) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;

                        // WG_BLOCK_N=128 scatters 8 BF16s/lane (=16B, uint4).
                        // For WG_BLOCK_N=64 each lane owns 4 BF16s (=8B), so
                        // use uint2; a uint4 load would be misaligned for
                        // odd lanes.
                        auto smem_ptr = smem_cd_l2
                            + smem_cd_l2_wg_offset
                            + row_in_wg * WG_SMEM_CD_L2_STRIDE_N
                            + lane_in_row * kColsPerScatterLane;
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
#ifdef DG_MEGA_MOE_INTERNODE
                        // Inter-node target rank: mirror the intra-node vector
                        // store with one inline RDMA WRITE, then wait before
                        // reusing the QP.  Correctness-first inline publish;
                        // the async publisher replaces this in the perf step.
                        const bool row_is_inter =
                            dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                            sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
                        const int scatter_qp_id =
                            static_cast<int>(dst_token_idx + lane_in_row);
#endif
                        if constexpr (kColsPerScatterLane == 8) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
#ifdef DG_MEGA_MOE_INTERNODE
                            if constexpr (kCombineFullRow) {
                                if (row_is_inter) {
                                    const uint32_t staging_row_idx =
                                        kCombineStageRing ?
                                            combine_stage_expert_base +
                                                m_block_idx * BLOCK_M +
                                                m_idx_in_block :
                                            m_idx + m_idx_in_block;
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                                    const auto staging_base_ptr =
                                        get_cpu_proxy_staging_row_ptr(
                                            staging_row_idx);
#else
                                    const auto staging_base_ptr =
                                        combine_full_row_staging_buffer
                                            .get_data_buffer(staging_row_idx)
                                            .get_base_ptr();
#endif
                                    auto staging_ptr = math::advance_ptr<uint4>(
                                        staging_base_ptr,
                                        (n_idx + wg_n_offset) * sizeof(nv_bfloat16) +
                                            lane_in_row * sizeof(uint4));
                                    *staging_ptr = packed;
                                    if (lane_in_row == 0 and
                                        not kCombineExpertReady)
                                        atomicExch(smem_expert_count, 1u);
                                } else {
                                    *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                                }
                            } else if (row_is_inter) {
                                comm::ibgda::put_inline<uint4>(
                                    dst_ptr, packed,
                                    static_cast<int>(dst_rank_idx), scatter_qp_id);
                                comm::ibgda::quiet(
                                    static_cast<int>(dst_rank_idx), scatter_qp_id);
                            } else
#endif
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        } else {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint2>(
                                dst_token.get_base_ptr(),
                                (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint2));
#ifdef DG_MEGA_MOE_INTERNODE
                            if constexpr (kCombineFullRow) {
                                if (row_is_inter) {
                                    const uint32_t staging_row_idx =
                                        kCombineStageRing ?
                                            combine_stage_expert_base +
                                                m_block_idx * BLOCK_M +
                                                m_idx_in_block :
                                            m_idx + m_idx_in_block;
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                                    const auto staging_base_ptr =
                                        get_cpu_proxy_staging_row_ptr(
                                            staging_row_idx);
#else
                                    const auto staging_base_ptr =
                                        combine_full_row_staging_buffer
                                            .get_data_buffer(staging_row_idx)
                                            .get_base_ptr();
#endif
                                    auto staging_ptr = math::advance_ptr<uint2>(
                                        staging_base_ptr,
                                        (n_idx + wg_n_offset) * sizeof(nv_bfloat16) +
                                            lane_in_row * sizeof(uint2));
                                    *staging_ptr = packed;
                                    if (lane_in_row == 0 and
                                        not kCombineExpertReady)
                                        atomicExch(smem_expert_count, 1u);
                                } else {
                                    *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                                }
                            } else if (row_is_inter) {
                                comm::ibgda::put_inline<uint2>(
                                    dst_ptr, packed,
                                    static_cast<int>(dst_rank_idx), scatter_qp_id);
                                comm::ibgda::quiet(
                                    static_cast<int>(dst_rank_idx), scatter_qp_id);
                            } else
#endif
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                }

#if defined(DG_MEGA_MOE_INTERNODE)
                if constexpr (kCombineFullRow) {
                    // Publish this N-block only after every warpgroup completed
                    // its fragments. GPU-scope acq_rel RMWs are sufficient for
                    // the local producer/publisher hand-off. The last producer
                    // retains a system-scope release so the staged payload is
                    // visible to the RNIC before the publisher rings a doorbell.
                    // The scatter section executes per warpgroup (split-N: two
                    // WGs own halves of this block's columns; split-M: rows),
                    // so a CTA-wide sync here can deadlock against a WG still
                    // in another block.  Each WG leader instead contributes
                    // one arrival for its fragment: the WG-local barrier makes
                    // the fragment's stores visible to the leader, and the
                    // cumulative acq_rel RMW releases them to the dedicated
                    // async publisher warp.
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    if (lane_idx == 0 and warp_idx_in_wg == 0 and
                        (smem_expert_count[0] != 0 or kCombineExpertReady)) {
                        const auto arrival_ptr = combine_full_row_arrival_buffer
                            .get_data_buffer(m_idx / BLOCK_M)
                            .get_base_ptr<uint32_t>();
                        const auto old = ptx::atomic_add_acq_rel_gpu(arrival_ptr, 1);
                        // Split-N always needs every column warpgroup.
                        // Split-M only runs the M warpgroups covered by this
                        // block's valid rows: the early-return path above
                        // deliberately skips a WG whose M offset is outside a
                        // short/tail block.
                        const uint32_t num_active_m_wgs =
                            math::ceil_div(valid_m, WG_BLOCK_M);
                        const uint32_t num_l2_fragments =
                            kNumL2BlockNs * num_active_m_wgs *
                            (BLOCK_N / WG_BLOCK_N);
                        if (old + 1 == num_l2_fragments) {
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                            *combine_full_row_ready_timestamp_buffer
                                 .get_data_buffer(m_idx / BLOCK_M)
                                 .get_base_ptr<uint64_t>() =
                                     ptx::get_globaltimer();
#endif
                            ptx::st_release_sys(
                                arrival_ptr,
                                kCombineFullRowPublishReadyBit |
                                    num_l2_fragments);
                        }
                    }
                }

#endif

#ifdef DG_MEGA_MOE_PHASE_PROFILE
                if (profile_math_leader)
                    profile_scatter_cycles +=
                        clock64() - profile_scatter_block_start;
#endif

                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            if (profile_math_leader) {
                const uint64_t block_cycles =
                    clock64() - profile_math_block_start;
                if (block_phase == sched::BlockPhase::Linear1) {
                    profile_l1_cycles += block_cycles;
                    ++ profile_l1_block_count;
                } else {
                    profile_l2_cycles += block_cycles;
                    ++ profile_l2_block_count;
                }
            }
#endif
        }, cached_recv_counts);

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (profile_math_leader) {
            phase_profile[kProfileL1] = profile_l1_cycles;
            phase_profile[kProfileL2] = profile_l2_cycles;
            phase_profile[kProfileScatter] = profile_scatter_cycles;
            phase_profile[kProfileL1BlockCount] = profile_l1_block_count;
            phase_profile[kProfileL2BlockCount] = profile_l2_block_count;
            phase_profile[kProfileDecodeWait] = profile_decode_wait_cycles;
            phase_profile[kProfileAWait] = profile_a_wait_cycles;
        }
        const uint64_t profile_combine_barrier_start =
            profile_math_leader ? clock64() : 0;
#endif

        // ---------------- COMBINE ----------------
        if constexpr (kCombineExpertReady) {
            // All local L2 producers and ready notifications must be submitted
            // before dispatch cleanup reuses their counters.  Remote completion
            // is observed below per dependency, so no all-QP quiet or
            // cross-rank collective is needed on the combine hot path.
            comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
                workspace, sm_idx, epilogue_thread_idx,
                [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); });
        } else {
            // Legacy: NVLink barrier signals remote ranks that this rank's
            // GEMM outputs (NVLink scatter targets) are fully written.
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                 kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
            );
        }

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state.
        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (profile_math_leader)
            phase_profile[kProfileCombineBarrier] =
                clock64() - profile_combine_barrier_start;
        const uint64_t profile_combine_reduce_start =
            lane_idx == 0 ? clock64() : 0;
#endif

        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr bool kOneCombineChunkFits =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE;
        constexpr bool kTwoCombineChunksFit =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / 2 <= SMEM_BEFORE_BARRIER_SIZE;
        constexpr uint32_t kNumChunks =
            (kOneCombineChunkFits and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 :
            (kTwoCombineChunksFit ? 2 : 4);
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        DG_TRAP_ONLY_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;

            if constexpr (kCombineExpertReady) {
                // The ready WRITE follows all output rows for this expert on
                // the same RC QP.  An acquire-system load that observes this
                // launch's epoch can therefore safely precede the gather.
                if (stored_topk_slot_idx >= 0) {
                    DG_TRAP_ONLY_DEVICE_ASSERT(
                        stored_topk_slot_idx < static_cast<int>(kNumExperts));
                    const uint64_t launch_epoch = kernel_launch_epoch;
                    constexpr uint64_t kReadyTimeoutCycles =
                        60ull * 2000000000ull;
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    const uint64_t profile_ready_wait_start = clock64();
#endif
                    const auto ready_wait_start = clock64();
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                    const uint32_t global_expert_idx =
                        static_cast<uint32_t>(stored_topk_slot_idx);
                    const uint32_t expert_src_rank =
                        global_expert_idx / kNumExpertsPerRank;
                    const bool expert_is_inter =
                        expert_src_rank / DG_MEGA_MOE_NVL_PEERS !=
                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
                    if (expert_is_inter) {
                        const uint32_t source_local_rank =
                            expert_src_rank % kCPUProxyPeersPerNode;
                        const auto& pair_dev_comm =
                            cpu_proxy_dev_comms[source_local_rank];
                        ncclGin net(
                            pair_dev_comm, 0,
                            NCCL_GIN_RESOURCE_SHARING_GPU);
                        while (net.readSignal(
                                   static_cast<ncclGinSignal_t>(
                                       global_expert_idx %
                                       kNumExpertsPerRank)) < launch_epoch)
                            DG_TRAP_ONLY_DEVICE_ASSERT(
                                clock64() - ready_wait_start <
                                kReadyTimeoutCycles);
                    } else {
#endif
                        const auto ready_ptr =
                            workspace.get_combine_ready_epoch_ptr(
                                static_cast<uint32_t>(stored_topk_slot_idx));
                        while (ptx::ld_acq_sys(ready_ptr) != launch_epoch)
                            DG_TRAP_ONLY_DEVICE_ASSERT(
                                clock64() - ready_wait_start <
                                kReadyTimeoutCycles);
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                    }
#endif
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    profile_combine_ready_wait_cycles +=
                        clock64() - profile_ready_wait_start;
#endif
                }
                __syncwarp();
            }
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        const int selected_global_expert_idx =
                            __shfl_sync(
                                0xffffffff, stored_topk_slot_idx, slot_idx);
                        if (cute::elect_one_sync()) {
#ifdef DG_MEGA_MOE_FP4_CPU_PROXY
                            const uint32_t selected_src_rank =
                                static_cast<uint32_t>(
                                    selected_global_expert_idx) /
                                kNumExpertsPerRank;
                            const bool selected_is_inter =
                                selected_src_rank / DG_MEGA_MOE_NVL_PEERS !=
                                sym_buffer.rank_idx /
                                    DG_MEGA_MOE_NVL_PEERS;
                            const auto src_base_ptr = selected_is_inter ?
                                get_cpu_proxy_combine_row_ptr(
                                    slot_idx, token_idx) :
                                combine_token_buffer
                                    .get_rank_buffer(slot_idx)
                                    .get_data_buffer(token_idx)
                                    .get_base_ptr();
#else
                            const auto src_base_ptr =
                                combine_token_buffer
                                    .get_rank_buffer(slot_idx)
                                    .get_data_buffer(token_idx)
                                    .get_base_ptr();
#endif
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                src_base_ptr,
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (lane_idx == 0) {
            atomicMax(
                phase_profile + kProfileCombineReduce,
                static_cast<unsigned long long>(
                    clock64() - profile_combine_reduce_start));
            atomicMax(
                phase_profile + kProfileCombineReadyWait,
                static_cast<unsigned long long>(
                    profile_combine_ready_wait_cycles));
        }
        if (profile_math_leader)
            phase_profile[kProfileTotal] =
                clock64() - phase_profile[kProfileStartClock];

        comm::grid_sync<kNumSMs, 2>(
            workspace, sm_idx, epilogue_thread_idx,
            [&]() {
                ptx::sync_aligned(
                    kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            });

#ifndef DG_MEGA_MOE_PHASE_PROFILE_SILENT
        if (sm_idx == 0 and epilogue_thread_idx == 0) {
            unsigned long long max_cycles[kPhaseProfileSlots] = {};
            unsigned long long remote_read_count = 0;
            unsigned long long l1_block_count = 0;
            unsigned long long l2_block_count = 0;
            unsigned long long sum_l1_cycles = 0;
            unsigned long long sum_l2_cycles = 0;
            unsigned long long sum_decode_wait_cycles = 0;
            unsigned long long sum_a_wait_cycles = 0;
            unsigned long long sum_scatter_cycles = 0;
            unsigned long long publish_total_blocks = 0;
            unsigned long long publish_empty_dst_blocks = 0;
            unsigned long long publish_metadata_loads = 0;
            unsigned long long publish_rdma_batches = 0;
            unsigned long long publish_outer_loops = 0;
            unsigned long long publish_ready_checks = 0;
            unsigned long long publish_empty_passes = 0;
            unsigned long long publish_sleep_calls = 0;
            unsigned long long publish_requested_sleep_ns = 0;
            unsigned long long publish_max_requested_sleep_ns = 0;
            unsigned long long publish_ready_observe_ns = 0;
            unsigned long long publish_ready_observe_samples = 0;
            for (uint32_t sm = 0; sm < kNumSMs; ++ sm) {
                const auto row = phase_profile_buffer.get_data_buffer(sm)
                    .get_base_ptr<unsigned long long>();
                #pragma unroll
                for (uint32_t slot = 0; slot < kPhaseProfileSlots; ++ slot)
                    max_cycles[slot] = max_cycles[slot] > row[slot] ?
                        max_cycles[slot] : row[slot];
                remote_read_count += row[kProfileRemoteReadCount];
                l1_block_count += row[kProfileL1BlockCount];
                l2_block_count += row[kProfileL2BlockCount];
                sum_l1_cycles += row[kProfileL1];
                sum_l2_cycles += row[kProfileL2];
                sum_decode_wait_cycles += row[kProfileDecodeWait];
                sum_a_wait_cycles += row[kProfileAWait];
                sum_scatter_cycles += row[kProfileScatter];
                publish_total_blocks +=
                    row[kProfilePublishTotalBlocks];
                publish_empty_dst_blocks +=
                    row[kProfilePublishEmptyDstBlocks];
                publish_metadata_loads +=
                    row[kProfilePublishMetadataLoads];
                publish_rdma_batches +=
                    row[kProfilePublishRDMABatches];
                publish_outer_loops +=
                    row[kProfilePublishOuterLoops];
                publish_ready_checks +=
                    row[kProfilePublishReadyChecks];
                publish_empty_passes +=
                    row[kProfilePublishEmptyPasses];
                const uint64_t sleep_stats =
                    row[kProfilePublishSleepStats];
                publish_sleep_calls += sleep_stats >> 32;
                publish_requested_sleep_ns +=
                    sleep_stats & 0xffffffffull;
                publish_max_requested_sleep_ns = cute::max(
                    publish_max_requested_sleep_ns,
                    row[kProfilePublishMaxRequestedSleepNs]);
                const uint64_t ready_observe_stats =
                    row[kProfilePublishReadyObserveStats];
                publish_ready_observe_samples +=
                    ready_observe_stats >> 48;
                publish_ready_observe_ns +=
                    ready_observe_stats & ((1ull << 48) - 1ull);
            }
            printf(
                "FP4_MEGA_MOE_PHASE_PROFILE rank=%u tokens=%u block_m=%u "
                "block_n=%u metadata_cycles=%llu "
                "dispatch_barrier_cycles=%llu dispatch_pull_cycles=%llu "
                "remote_read_cycles=%llu cleanup_barrier_cycles=%llu "
                "l1_cycles=%llu l2_cycles=%llu decode_wait_cycles=%llu a_wait_cycles=%llu "
                "loader_pool_wait_cycles=%llu "
                "scatter_cycles=%llu "
                "combine_barrier_cycles=%llu combine_ready_wait_cycles=%llu "
                "combine_reduce_cycles=%llu total_cycles=%llu "
                "remote_reads=%llu l1_blocks=%llu l2_blocks=%llu\n",
                sym_buffer.rank_idx, num_tokens, BLOCK_M, BLOCK_N,
                max_cycles[kProfileMetadata],
                max_cycles[kProfileDispatchBarrier],
                max_cycles[kProfileDispatchPull],
                max_cycles[kProfileRemoteRead],
                max_cycles[kProfileCleanupBarrier],
                max_cycles[kProfileL1], max_cycles[kProfileL2],
                max_cycles[kProfileDecodeWait],
                max_cycles[kProfileAWait],
                max_cycles[kProfileLoaderPoolWait],
                max_cycles[kProfileScatter],
                max_cycles[kProfileCombineBarrier],
                max_cycles[kProfileCombineReadyWait],
                max_cycles[kProfileCombineReduce],
                max_cycles[kProfileTotal], remote_read_count,
                l1_block_count, l2_block_count);
            // CUDA device printf accepts at most 32 arguments including the
            // format string. Keep publisher diagnostics in a separate record
            // so the final values are not read from unrelated vararg state.
            printf(
                "FP4_MEGA_MOE_PUBLISH_PROFILE rank=%u tokens=%u "
                "publish_total_blocks=%llu "
                "publish_empty_dst_blocks=%llu "
                "publish_metadata_loads=%llu "
                "publish_rdma_batches=%llu "
                "publish_outer_loops=%llu "
                "publish_ready_checks=%llu "
                "publish_empty_passes=%llu "
                "publish_sleep_calls=%llu "
                "publish_requested_sleep_ns=%llu "
                "publish_max_requested_sleep_ns=%llu "
                "publish_ready_observe_ns=%llu "
                "publish_ready_observe_samples=%llu\n",
                sym_buffer.rank_idx, num_tokens,
                publish_total_blocks, publish_empty_dst_blocks,
                publish_metadata_loads, publish_rdma_batches,
                publish_outer_loops, publish_ready_checks,
                publish_empty_passes, publish_sleep_calls,
                publish_requested_sleep_ns,
                publish_max_requested_sleep_ns,
                publish_ready_observe_ns,
                publish_ready_observe_samples);
            printf(
                "FP4_MEGA_MOE_SYNC_PROFILE rank=%u tokens=%u entry_ns=%llu "
                "counts_sent_ns=%llu counts_ready_ns=%llu\n",
                sym_buffer.rank_idx, num_tokens,
                phase_profile[kProfileEntryGlobaltimer],
                phase_profile[kProfileCountsSentGlobaltimer],
                phase_profile[kProfileCountsReadyGlobaltimer]);
        }
#endif
#endif
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_TRAP_ONLY_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
