#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/fp4_rs_detail.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#define __CLION_IDE__

namespace deep_gemm {

template <
    uint32_t LOAD_BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t kScaleBGranK,
    uint32_t kNumSFBPerBlockK,
    bool kUseDynamicLutDecode,
    bool kUseCommonLutFastPath,
    bool kFuseScaleBHummingDecode,
    typename PackedT,
    typename DecodedT>
__device__ __forceinline__ void dequant_fp4_b_tile_to_e4m3_smem(
    const uint32_t decode_thread_idx,
    const uint32_t num_decode_threads,
    const PackedT* __restrict__ smem_b_packed_stage,
    DecodedT* __restrict__ smem_b_stage,
    const uint32_t* __restrict__ smem_sfb_stage) {
    constexpr uint32_t kPackedWordsPerKG = kScaleBGranK / 8;  // 4
    constexpr uint32_t kGroupsPerTile = LOAD_BLOCK_N * kNumSFBPerBlockK;

    for (uint32_t group = decode_thread_idx; group < kGroupsPerTile; group += num_decode_threads) {
            const uint32_t n_row = group / kNumSFBPerBlockK;
            const uint32_t kg = group - n_row * kNumSFBPerBlockK;
            const uint32_t sfb_word = smem_sfb_stage[n_row];
            const uint32_t e8m0 = (sfb_word >> (kg * 8)) & 0xffu;

            const auto* packed_row = reinterpret_cast<const uint32_t*>(
                smem_b_packed_stage + n_row * (BLOCK_K / 2));
            auto* decoded_row_u64 = reinterpret_cast<uint64_t*>(
                smem_b_stage + n_row * BLOCK_K);
            const uint32_t row_swizzle = n_row & 7u;

            uint32_t scaled_lut_lo = 0;
            uint32_t scaled_lut_hi = 0;
            if constexpr (!kFuseScaleBHummingDecode) {
                if constexpr (kUseDynamicLutDecode) {
                    const auto scaled_lut =
                        fp4_rs_detail::make_scaled_e4m3_lut_from_e8m0_fast(e8m0);
                    scaled_lut_lo = scaled_lut.lo;
                    scaled_lut_hi = scaled_lut.hi;
                } else {
                    const uint64_t scaled_lut =
                        kUseCommonLutFastPath ?
                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0) :
                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0);
                    scaled_lut_lo = static_cast<uint32_t>(scaled_lut);
                    scaled_lut_hi = static_cast<uint32_t>(scaled_lut >> 32);
                }
            }

            #pragma unroll
            for (uint32_t pw = 0; pw < kPackedWordsPerKG; ++ pw) {
                const uint32_t pw_global = kg * kPackedWordsPerKG + pw;
                const uint32_t packed = packed_row[pw_global];
                uint32_t lo, hi;
                if constexpr (kFuseScaleBHummingDecode) {
                    if (e8m0 >= 121u and e8m0 <= 149u) {
                        const uint32_t exp_offset = e8m0 - 121u;
                        lo = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed & 0xffffu, exp_offset);
                        hi = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed >> 16, exp_offset);
                    } else {
                        lo = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed & 0xffffu, e8m0);
                        hi = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed >> 16, e8m0);
                    }
                } else {
                    lo = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                        packed & 0xffffu, scaled_lut_lo, scaled_lut_hi);
                    hi = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                        packed >> 16, scaled_lut_lo, scaled_lut_hi);
                }
                const uint64_t out8 =
                    static_cast<uint64_t>(lo) |
                    (static_cast<uint64_t>(hi) << 32);
                const uint32_t seg_id     = pw_global >> 1;
                const uint32_t off_in_seg = pw_global & 1u;
                const uint32_t swz_seg    = seg_id ^ row_swizzle;
                decoded_row_u64[swz_seg * 2u + off_in_seg] = out8;
            }
    }
}

template <
    uint32_t LOAD_BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t kScaleBGranK,
    uint32_t kNumSFBPerBlockK,
    bool kUseDynamicLutDecode,
    bool kUseCommonLutFastPath,
    bool kSkipZeroSFBDecode,
    bool kFuseScaleBHummingDecode,
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
        const uint32_t sfb_word = smem_sfb_stage[n_row];
        const uint32_t e8m0 = (sfb_word >> (kg * 8)) & 0xffu;

        const auto* packed_row = reinterpret_cast<const uint32_t*>(
            smem_b_packed_stage + n_row * (BLOCK_K / 2));
        auto* decoded_row_u64 = reinterpret_cast<uint64_t*>(
            smem_b_stage + n_row * BLOCK_K);
        const uint32_t row_swizzle = n_row & 7u;

        if constexpr (kSkipZeroSFBDecode) {
            if (e8m0 == 0u) {
                #pragma unroll
                for (uint32_t pair = 0; pair < kPackedWordPairsPerKG; ++ pair) {
                    const uint32_t pw_global_0 = kg * kPackedWordsPerKG + pair * 2u;
                    const uint32_t seg_id = pw_global_0 >> 1;
                    const uint32_t swz_seg = seg_id ^ row_swizzle;
                    ptx::st_shared_v2_u64(decoded_row_u64 + swz_seg * 2u, 0ull, 0ull);
                }
                continue;
            }
        }

        uint32_t scaled_lut_lo = 0;
        uint32_t scaled_lut_hi = 0;
        if constexpr (!kFuseScaleBHummingDecode) {
            if constexpr (kUseDynamicLutDecode) {
                const auto scaled_lut =
                    fp4_rs_detail::make_scaled_e4m3_lut_from_e8m0_fast(e8m0);
                scaled_lut_lo = scaled_lut.lo;
                scaled_lut_hi = scaled_lut.hi;
            } else {
                const uint64_t scaled_lut =
                    kUseCommonLutFastPath ?
                    fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0) :
                    fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0);
                scaled_lut_lo = static_cast<uint32_t>(scaled_lut);
                scaled_lut_hi = static_cast<uint32_t>(scaled_lut >> 32);
            }
        }

        #pragma unroll
        for (uint32_t pair = 0; pair < kPackedWordPairsPerKG; ++ pair) {
            const uint32_t pw_global_0 = kg * kPackedWordsPerKG + pair * 2u;
            const uint32_t packed_0 = packed_row[pw_global_0];
            const uint32_t packed_1 = packed_row[pw_global_0 + 1u];
            uint32_t lo_0, hi_0, lo_1, hi_1;
            if constexpr (kFuseScaleBHummingDecode) {
                if (e8m0 >= 121u and e8m0 <= 149u) {
                    const uint32_t exp_offset = e8m0 - 121u;
                    lo_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_0 & 0xffffu, exp_offset);
                    hi_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_0 >> 16, exp_offset);
                    lo_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_1 & 0xffffu, exp_offset);
                    hi_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_1 >> 16, exp_offset);
                } else {
                    lo_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_0 & 0xffffu, e8m0);
                    hi_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_0 >> 16, e8m0);
                    lo_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_1 & 0xffffu, e8m0);
                    hi_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_1 >> 16, e8m0);
                }
            } else {
                lo_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                    packed_0 & 0xffffu, scaled_lut_lo, scaled_lut_hi);
                hi_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                    packed_0 >> 16, scaled_lut_lo, scaled_lut_hi);
                lo_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                    packed_1 & 0xffffu, scaled_lut_lo, scaled_lut_hi);
                hi_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                    packed_1 >> 16, scaled_lut_lo, scaled_lut_hi);
            }
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
    bool kUseDynamicLutDecode,
    bool kUseCommonLutFastPath,
    bool kFuseScaleBHummingDecode,
    typename PackedT,
    typename DecodedT>
__device__ __forceinline__ void dequant_fp4_b_kg_to_e4m3_smem_vec_store(
    const uint32_t decode_thread_idx,
    const uint32_t num_decode_threads,
    const uint32_t kg,
    const PackedT* __restrict__ smem_b_packed_stage,
    DecodedT* __restrict__ smem_b_stage,
    const uint32_t* __restrict__ smem_sfb_stage) {
    constexpr uint32_t kPackedWordsPerKG = kScaleBGranK / 8;  // 4
    constexpr uint32_t kPackedWordPairsPerKG = kPackedWordsPerKG / 2;
    DG_STATIC_ASSERT(kPackedWordsPerKG == 4, "Per-KG vector-store decode assumes per-32K groups");

    for (uint32_t n_row = decode_thread_idx; n_row < LOAD_BLOCK_N; n_row += num_decode_threads) {
        const uint32_t sfb_word = smem_sfb_stage[n_row];
        const uint32_t e8m0 = (sfb_word >> (kg * 8)) & 0xffu;

        const auto* packed_row = reinterpret_cast<const uint32_t*>(
            smem_b_packed_stage + n_row * (BLOCK_K / 2));
        auto* decoded_row_u64 = reinterpret_cast<uint64_t*>(
            smem_b_stage + n_row * BLOCK_K);
        const uint32_t row_swizzle = n_row & 7u;

        uint32_t scaled_lut_lo = 0;
        uint32_t scaled_lut_hi = 0;
        if constexpr (!kFuseScaleBHummingDecode) {
            if constexpr (kUseDynamicLutDecode) {
                const auto scaled_lut =
                    fp4_rs_detail::make_scaled_e4m3_lut_from_e8m0_fast(e8m0);
                scaled_lut_lo = scaled_lut.lo;
                scaled_lut_hi = scaled_lut.hi;
            } else {
                const uint64_t scaled_lut =
                    kUseCommonLutFastPath ?
                    fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0) :
                    fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0);
                scaled_lut_lo = static_cast<uint32_t>(scaled_lut);
                scaled_lut_hi = static_cast<uint32_t>(scaled_lut >> 32);
            }
        }

        #pragma unroll
        for (uint32_t pair = 0; pair < kPackedWordPairsPerKG; ++ pair) {
            const uint32_t pw_global_0 = kg * kPackedWordsPerKG + pair * 2u;
            const uint32_t packed_0 = packed_row[pw_global_0];
            const uint32_t packed_1 = packed_row[pw_global_0 + 1u];
            uint32_t lo_0, hi_0, lo_1, hi_1;
            if constexpr (kFuseScaleBHummingDecode) {
                if (e8m0 >= 121u and e8m0 <= 149u) {
                    const uint32_t exp_offset = e8m0 - 121u;
                    lo_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_0 & 0xffffu, exp_offset);
                    hi_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_0 >> 16, exp_offset);
                    lo_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_1 & 0xffffu, exp_offset);
                    hi_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_1 >> 16, exp_offset);
                } else {
                    lo_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_0 & 0xffffu, e8m0);
                    hi_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_0 >> 16, e8m0);
                    lo_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_1 & 0xffffu, e8m0);
                    hi_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_1 >> 16, e8m0);
                }
            } else {
                lo_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                    packed_0 & 0xffffu, scaled_lut_lo, scaled_lut_hi);
                hi_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                    packed_0 >> 16, scaled_lut_lo, scaled_lut_hi);
                lo_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                    packed_1 & 0xffffu, scaled_lut_lo, scaled_lut_hi);
                hi_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                    packed_1 >> 16, scaled_lut_lo, scaled_lut_hi);
            }
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
    bool kUseDynamicLutDecode,
    bool kUseCommonLutFastPath,
    bool kFuseScaleBHummingDecode,
    typename PackedT,
    typename DecodedT>
__device__ __forceinline__ void dequant_fp4_b_tile_to_e4m3_smem_kg_pair(
    const uint32_t decode_thread_idx,
    const uint32_t num_decode_threads,
    const PackedT* __restrict__ smem_b_packed_stage,
    DecodedT* __restrict__ smem_b_stage,
    const uint32_t* __restrict__ smem_sfb_stage) {
    constexpr uint32_t kPackedWordsPerKG = kScaleBGranK / 8;  // 4
    constexpr uint32_t kPackedWordPairsPerKG = kPackedWordsPerKG / 2;
    static_assert(kNumSFBPerBlockK % 2 == 0, "KG-pair decode expects an even K-group count");
    static_assert(kPackedWordsPerKG == 4, "KG-pair vector store assumes per-32K groups");
    constexpr uint32_t kPairsPerRow = kNumSFBPerBlockK / 2;
    constexpr uint32_t kPairsPerTile = LOAD_BLOCK_N * kPairsPerRow;

    for (uint32_t pair = decode_thread_idx; pair < kPairsPerTile; pair += num_decode_threads) {
        const uint32_t n_row = pair / kPairsPerRow;
        const uint32_t kg_base = (pair - n_row * kPairsPerRow) * 2u;
        const uint32_t sfb_word = smem_sfb_stage[n_row];

        const auto* packed_row = reinterpret_cast<const uint32_t*>(
            smem_b_packed_stage + n_row * (BLOCK_K / 2));
        auto* decoded_row_u64 = reinterpret_cast<uint64_t*>(
            smem_b_stage + n_row * BLOCK_K);
        const uint32_t row_swizzle = n_row & 7u;

        #pragma unroll
        for (uint32_t sub = 0; sub < 2; ++ sub) {
            const uint32_t kg = kg_base + sub;
            const uint32_t e8m0 = (sfb_word >> (kg * 8)) & 0xffu;

            if (e8m0 == 0u) {
                #pragma unroll
                for (uint32_t pair_idx = 0; pair_idx < kPackedWordPairsPerKG; ++ pair_idx) {
                    const uint32_t pw_global = kg * kPackedWordsPerKG + pair_idx * 2u;
                    const uint32_t seg_id = pw_global >> 1;
                    const uint32_t swz_seg = seg_id ^ row_swizzle;
                    ptx::st_shared_v2_u64(decoded_row_u64 + swz_seg * 2u, 0ull, 0ull);
                }
                continue;
            }

            uint32_t scaled_lut_lo = 0;
            uint32_t scaled_lut_hi = 0;
            if constexpr (!kFuseScaleBHummingDecode) {
                if constexpr (kUseDynamicLutDecode) {
                    const auto scaled_lut =
                        fp4_rs_detail::make_scaled_e4m3_lut_from_e8m0_fast(e8m0);
                    scaled_lut_lo = scaled_lut.lo;
                    scaled_lut_hi = scaled_lut.hi;
                } else {
                    const uint64_t scaled_lut =
                        kUseCommonLutFastPath ?
                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0) :
                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0);
                    scaled_lut_lo = static_cast<uint32_t>(scaled_lut);
                    scaled_lut_hi = static_cast<uint32_t>(scaled_lut >> 32);
                }
            }

            #pragma unroll
            for (uint32_t pair_idx = 0; pair_idx < kPackedWordPairsPerKG; ++ pair_idx) {
                const uint32_t pw_global_0 = kg * kPackedWordsPerKG + pair_idx * 2u;
                const uint32_t packed_0 = packed_row[pw_global_0];
                const uint32_t packed_1 = packed_row[pw_global_0 + 1u];
                uint32_t lo_0, hi_0, lo_1, hi_1;
                if constexpr (kFuseScaleBHummingDecode) {
                    if (e8m0 >= 121u and e8m0 <= 149u) {
                        const uint32_t exp_offset = e8m0 - 121u;
                        lo_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_0 & 0xffffu, exp_offset);
                        hi_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_0 >> 16, exp_offset);
                        lo_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_1 & 0xffffu, exp_offset);
                        hi_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(packed_1 >> 16, exp_offset);
                    } else {
                        lo_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_0 & 0xffffu, e8m0);
                        hi_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_0 >> 16, e8m0);
                        lo_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_1 & 0xffffu, e8m0);
                        hi_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(packed_1 >> 16, e8m0);
                    }
                } else {
                    lo_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                        packed_0 & 0xffffu, scaled_lut_lo, scaled_lut_hi);
                    hi_0 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                        packed_0 >> 16, scaled_lut_lo, scaled_lut_hi);
                    lo_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                        packed_1 & 0xffffu, scaled_lut_lo, scaled_lut_hi);
                    hi_1 = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                        packed_1 >> 16, scaled_lut_lo, scaled_lut_hi);
                }
                const uint32_t seg_id = pw_global_0 >> 1;
                const uint32_t swz_seg = seg_id ^ row_swizzle;
                ptx::st_shared(
                    decoded_row_u64 + swz_seg * 2u,
                    lo_0, hi_0, lo_1, hi_1);
            }
        }
    }
}

template <
    uint32_t LOAD_BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t kScaleBGranK,
    uint32_t kNumSFBPerBlockK,
    bool kUseKGPairDecode,
    bool kUseVectorStoreDecode,
    bool kSkipZeroSFBDecode,
    bool kUseDynamicLutDecode,
    bool kUseCommonLutFastPath,
    bool kFuseScaleBHummingDecode,
    typename PackedT,
    typename DecodedT>
__device__ __forceinline__ void dequant_fp4_b_tile_to_e4m3_smem_dispatch(
    const uint32_t decode_thread_idx,
    const uint32_t num_decode_threads,
    const PackedT* __restrict__ smem_b_packed_stage,
    DecodedT* __restrict__ smem_b_stage,
    const uint32_t* __restrict__ smem_sfb_stage) {
    if constexpr (kUseKGPairDecode) {
        dequant_fp4_b_tile_to_e4m3_smem_kg_pair<
            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK,
            kUseDynamicLutDecode, kUseCommonLutFastPath, kFuseScaleBHummingDecode>(
            decode_thread_idx, num_decode_threads,
            smem_b_packed_stage, smem_b_stage, smem_sfb_stage);
    } else if constexpr (kUseVectorStoreDecode) {
        dequant_fp4_b_tile_to_e4m3_smem_vec_store<
            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK,
            kUseDynamicLutDecode, kUseCommonLutFastPath,
            kSkipZeroSFBDecode, kFuseScaleBHummingDecode>(
            decode_thread_idx, num_decode_threads,
            smem_b_packed_stage, smem_b_stage, smem_sfb_stage);
    } else {
        dequant_fp4_b_tile_to_e4m3_smem<
            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK,
            kUseDynamicLutDecode, kUseCommonLutFastPath, kFuseScaleBHummingDecode>(
            decode_thread_idx, num_decode_threads,
            smem_b_packed_stage, smem_b_stage, smem_sfb_stage);
    }
}

// ============================================================================
// SM90 (Hopper) FP8 x FP4 MegaMoE — software-dequant path.
// ----------------------------------------------------------------------------
// Variant of `sm90_fp8_mega_moe_impl` for DSV4-style packed FP4 expert weights.
// The dispatch / scheduler / SwiGLU / combine machinery is identical to the
// FP8 implementation; the only differences are confined to:
//
//   1. Weight TMA load:  shape changes from (LOAD_BLOCK_N, BLOCK_K) of e4m3
//      to (LOAD_BLOCK_N, BLOCK_K/2) of packed int8 (each byte = 2 nibbles).
//   2. SFB:              loaded as UE8M0 packed int32 (per-32 K granularity)
//      via `cp.async`, since TMA does not natively stride FP4 layouts.
//   3. Mainloop decode:  the default host path uses the UE8M0 LUT decoder
//      (`fp4x4_to_scaled_e4m3x4_e8m0`) to dequant the packed FP4 weight tile.
//      `DG_SM90_FP4_FUSE_SCALE_B_HUMMING_DECODE=1` switches the JIT to the
//      humming decoder A/B path. The default decode-to-SMEM path writes an E4M3
//      tile (`smem_b_decoded`) that SS-mode WGMMA consumes exactly like the FP8
//      path, preserving the existing per-token SwiGLU amax / quantize epilogue.
//   4. RS mode:          enabled behind `DG_SM90_FP4_RS_MODE`, with a separate
//      JIT key/config and packed-B swizzle. The current path runs the RS WGMMA
//      mainloop. L1 uses a per-RS-slice shared-memory accumulator transpose bridge
//      to reuse the existing SwiGLU/amax/quantize epilogue; L2 writes BF16
//      directly from the native RS accumulator layout before NVLink scatter.
//
// Implementation note:
//   The DESIGN_fp4_sm90_mega_moe.md document specifies "scheme C / RS-mode"
//   as the long-term optimal path. The default path remains the conservative
//   decode-to-shared implementation. `kUseRSMode` switches the hot loop to
//   direct register decode + RS WGMMA. Linear1 still uses a temporary
//   accumulator scratch tile to convert each RS N-slice back to the existing
//   SwiGLU epilogue layout; Linear2 already stores BF16 directly from the
//   native RS layout. The native Linear1 epilogue was tested but regressed due
//   to spill pressure, so the slice transpose bridge is currently the faster
//   correctness-preserving path.
//
// See PR332 for the reference standalone RS-mode kernel from which the FP4
// dequant primitives in `fp4_rs_detail.cuh` were extracted.
// ============================================================================

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    // FP4-specific knobs (all defaults match the design doc's "Plan C" path).
    bool kUseKGPairDecode          = false,  // A/B: decode two consecutive K-groups per work item
    bool kUseVectorStoreDecode     = false,  // A/B: write adjacent decoded u64 values with one v2.u64 store
    bool kSkipZeroSFBDecode        = false,  // A/B: skip LUT decode when UE8M0 scale byte is zero
    bool kUseDynamicLutDecode      = false,  // A/B: build per-SFB LUT in registers instead of constant-cache lookup
    bool kUseCommonLutFastPath     = false,  // A/B: immediate LUTs for common UE8M0 values
    bool kUseKGPipelineDecode      = false,  // A/B: overlap per-K-group decode with each WGMMA issue
    bool kRSGroupKPromote          = false,  // A/B: RS Linear1 batches four K/32 WGMMA slices, then promotes once
    bool kRSL2GroupK2Promote       = false,  // A/B: RS Linear2 batches two K/32 WGMMA slices per wait
    bool kRSUseTransposeVecLoad    = false,  // A/B: vectorize RS L1 scratch readback into SS epilogue layout
    bool kRSGuardTransposeValid    = false,  // A/B: skip RS L1 transpose rows outside valid_m
    bool kRSUseSFAVecLoad          = false,  // A/B: vectorize adjacent RS SFA loads in promote loops
    bool kRSBroadcastSFALoad       = false,  // A/B: one row lane loads SFA, then shuffles to same-col lanes
    bool kRSReuseSFBWord           = false,  // A/B: reuse one packed SFB word across four K/32 slices
    bool kRSBroadcastSFBLoad       = false,  // A/B: one col-pair lane loads SFB, then shuffles to same-row lanes
    bool kRSStageSFB               = false,  // A/B: stage packed SFB words in SMEM for RS decode
    bool kRSDecodePairShfl         = false,  // A/B: share one packed FP4 load across adjacent low/high lanes
    bool kRSDirectL2Scatter        = false,  // A/B: skip RS L2 BF16 SMEM staging and scatter from registers
    bool kFuseScaleBHummingDecode = true,   // Plan C: bake SFB into decode LUT
    bool kScaleBPow2Promote       = true,   // UE8M0 SFB: exponent-shift promote
    bool kUseRSMode               = false,  // correctness-first RS mainloop
    bool kMathWGParticipatesInFP4Decode = true,
    uint32_t kNumMathWGDecodeWarps = kMathWGParticipatesInFP4Decode ? (kNumEpilogueThreads / 32) : 0,
    uint32_t kFirstFP4DecodeAssistWarp = 0,  // A/B: skip early non-epilogue warps as decode helpers
    bool kEarlyBDecode            = false,  // A/B: overlap assist decode with A/SFA TMA
    bool kDecodeDoneMBarrier      = false,  // A/B: one-way decode-done mbarrier instead of rendezvous sync
    bool kL2ArrivalCounter        = false,  // A/B: count ready L1 output slices instead of bitmask + CTA sync
    bool kSkipL2EpilogueSync      = false,  // A/B: rely on following grid/NVLink sync after L2 scatter
    bool kClockProfile            = false,  // debug-only clock64 phase counters
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
                           uint64_t* fp4_clock_profile) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

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
    DG_STATIC_ASSERT(BLOCK_M % 64 == 0, "BLOCK_M must be a multiple of WGMMA::M (64)");
    DG_STATIC_ASSERT(BLOCK_N % 8 == 0, "BLOCK_N must be compatible with SM90 FP8 WGMMA shapes");
    DG_STATIC_ASSERT(BLOCK_K == 128, "BLOCK_K is fixed to 128 (per-128 SF)");
    DG_STATIC_ASSERT(kNumMathWGDecodeWarps <= kNumEpilogueWarps,
                     "Math decode warps cannot exceed epilogue warps");
    DG_STATIC_ASSERT(kMathWGParticipatesInFP4Decode or kNumMathWGDecodeWarps == 0,
                     "Math decode warp count requires math WG decode participation");
    DG_STATIC_ASSERT(kFirstFP4DecodeAssistWarp <= kNumMMANonEpilogueWarps,
                     "First FP4 decode assist warp is out of range");
    DG_STATIC_ASSERT(!kUseKGPipelineDecode or !kUseRSMode,
                     "Per-KG pipeline is only implemented for decode-to-SMEM mode");
    DG_STATIC_ASSERT(!kUseRSMode or (BLOCK_M == 64 and kNumEpilogueWarpgroups == 1),
                     "The initial RS path supports BLOCK_M=64 with one epilogue warpgroup");

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

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
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32);
    // L2 activation SF is per-64 for the default BLOCK_N=128 path.  When
    // BLOCK_N=64, each L1 block emits only 32 post-SwiGLU columns, so the
    // intermediate SF buffer switches to per-32 to keep quant/dequant exact.
    constexpr uint32_t kL2ActsSFGranK  = BLOCK_N == 64 ? 32 : 64;
    constexpr auto fp8_intermediate_sf_layout =
        layout::Data(kIntermediateHidden * sizeof(float) / kL2ActsSFGranK);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

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
    using L1WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;  // M=64, N=BLOCK_N, K=32
    using L2WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using L1RSWGMMA = typename mma::sm90::FP8MMASelectorRS<BLOCK_M>::type;  // weight rows as M, tokens as N
    using L2RSWGMMA = typename mma::sm90::FP8MMASelectorRS<BLOCK_M>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    static_assert(L1RSWGMMA::M == 64 and L1RSWGMMA::N == BLOCK_M and L1RSWGMMA::K == 32,
                  "Unexpected RS WGMMA shape");

    // Cluster=1 -> no multicast, A/B are loaded full-sized
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t L1_OUT_BLOCK_N  = BLOCK_N / 2;  // post-SwiGLU
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    // 解码后的 E4M3 B-tile 使用 128B swizzle，以匹配 WGMMA SS 描述符
    // （layout_type = B128，与 FP8 baseline 一致）。dequant lambda 在写入
    // smem_b 时按 128B swizzle 计算字节偏移：col_byte ^= (n_row & 7) * 16。
    // TMA 加载的原始 FP4 packed 源 tile 在默认 decode-to-SMEM 路径走
    // swizzle=0（线性行优先），因为它只被 dequant lambda 通过普通
    // (row, col) 地址读取。RS mode 会切到 64B swizzle，匹配 PR332 的
    // register-fragment 地址映射。
    constexpr uint32_t kSwizzleBMode        = BLOCK_K * sizeof(b_dtype_t);  // 128
    constexpr uint32_t kSwizzleBPackedMode  = kUseRSMode ? 64 : 0;
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
        kUseRSMode ? 0 : LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // Packed FP4 source tile (TMA-loaded raw nibbles)
    constexpr uint32_t SMEM_B_PACKED_SIZE_PER_STAGE = LOAD_BLOCK_N * (BLOCK_K / 2) * sizeof(b_packed_dtype_t);
    // RS computes a transposed accumulator tile (weight rows x token cols).
    // L1 materialises one 64-column RS slice at a time back to token rows x
    // weight cols so the existing SwiGLU epilogue can remain byte-for-byte
    // equivalent. L2 skips this scratch path and writes BF16 directly from
    // native RS layout.
    constexpr uint32_t SMEM_RS_ACCUM_SIZE =
        kUseRSMode ? math::constexpr_align<uint32_t>(BLOCK_M * L1RSWGMMA::M * sizeof(float),
                                                     kSharedMemoryAlignment) : 0;
    // SFA per-stage must be sized for the larger of L1 (BLOCK_M floats) and
    // L2 (2*BLOCK_M floats per-64, or 4*BLOCK_M floats per-32 with BLOCK_N=64).
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>(kNumL2SFAPerBlockK * BLOCK_M * sizeof(float), 128u);
    // SFB UE8M0 per-32: the decode-to-SMEM path stages one packed uint32 per
    // N row per BLOCK_K in SMEM. Each word contains the 4 K/32 scale bytes, so
    // dequant avoids reloading the same word once per K group.
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = (kUseRSMode and !kRSStageSFB) ? 0 :
        math::constexpr_align<uint32_t>(LOAD_BLOCK_N * sizeof(uint32_t), 128u);

    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
    constexpr uint32_t SMEM_CD_L1_SIZE = kNumEpilogueWarpgroups * BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = kNumEpilogueWarpgroups * BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SIZE    = math::constexpr_align(
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE, kSharedMemoryAlignment);

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE + SMEM_RS_ACCUM_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                      SMEM_B_PACKED_SIZE_PER_STAGE +
                      SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_gemm_base);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_gemm_base);
    auto smem_rs_accum = reinterpret_cast<float*>(
        math::advance_ptr(smem_gemm_base, SMEM_CD_SIZE));

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(
            smem_gemm_base, SMEM_CD_SIZE + SMEM_RS_ACCUM_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    // Decoded e4m3 B tile (the operand actually consumed by WGMMA).
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base,
            SMEM_CD_SIZE + SMEM_RS_ACCUM_SIZE +
            kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    // Packed FP4 source tile (TMA-loaded; consumed only by the math warpgroup
    // during the FP4 → E4M3 dequant pass).
    auto smem_b_packed = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_packed_dtype_t>(smem_gemm_base,
            SMEM_CD_SIZE + SMEM_RS_ACCUM_SIZE
            + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE)
            + i * SMEM_B_PACKED_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + SMEM_RS_ACCUM_SIZE +
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
    constexpr bool kUseDecodeDoneMBarrier = (!kUseRSMode and kDecodeDoneMBarrier);
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
                // Default path uses one A+B full barrier. The early-B A/B path
                // splits packed-B readiness so assist warps can decode while
                // A/SFA TMA is still in flight; the main full barrier then only
                // tracks the A/SFA producer.
                full_barriers[i]->init(kUseEarlyBDecode ? 1 : 2);
                if constexpr (kUseEarlyBDecode)
                    decode_full_barriers[i]->init(1);
                if constexpr (kUseDecodeDoneMBarrier) {
                    constexpr uint32_t kDecodeWorkerWarps =
                        kNumMMANonEpilogueWarps + kNumMathWGDecodeWarps;
                    decode_done_barriers[i]->init(kDecodeWorkerWarps);
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

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks, /*kClusterSize=*/1u>(workspace);

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
    constexpr uint32_t kFP4DecodeBarrierIdx             = 15;

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    // For the 256-epilogue-thread case (block_m=128, 2 math WGs):
    //   128*48 + 128*40 + 256*208 = 64512 exactly.
    constexpr uint32_t kNumDispatchRegisters    = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters    = 208;
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
        if constexpr (kUseKGPipelineDecode) {
            #pragma unroll
            for (uint32_t kg = 0; kg < kNumSFBPerBlockK; ++ kg) {
                dequant_fp4_b_kg_to_e4m3_smem_vec_store<
                    LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK,
                    kUseDynamicLutDecode, kUseCommonLutFastPath, kFuseScaleBHummingDecode>(
                    decode_thread_idx, kNumFP4DecodeAssistThreads, kg,
                    smem_b_packed[cur_stage_idx], smem_b[cur_stage_idx], smem_sfb[cur_stage_idx]);
                ptx::sync_aligned(kNumFP4DecodeBarrierThreads, kFP4DecodeBarrierIdx);
            }
        } else {
            dequant_fp4_b_tile_to_e4m3_smem_dispatch<
                LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK,
                kUseKGPairDecode, kUseVectorStoreDecode,
                kSkipZeroSFBDecode, kUseDynamicLutDecode,
                kUseCommonLutFastPath, kFuseScaleBHummingDecode>(
                decode_thread_idx, kNumFP4DecodeWorkerThreads,
                smem_b_packed[cur_stage_idx], smem_b[cur_stage_idx], smem_sfb[cur_stage_idx]);
            arrive_or_sync_fp4_decode_done(cur_stage_idx);
        }
    };

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" → SF token index is now the simple
    //       per-block linear mapping (no 4×32 transpose).
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
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
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source token-topk indices to remote ranks
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            false, true);

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int      current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
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
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
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

            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

            // TMA pull token data into SMEM
            if (cute::elect_one_sync()) {
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
            const uint32_t sf_pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFFloats)
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            if (cute::elect_one_sync()) {
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                ptx::tma_store_1d(
                    l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                    pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
                ptx::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
            }
            __syncwarp();
        }

        // Cleanup workspace, overlapping with combine
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                }
                __syncwarp();
            }
        }

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, false);

    // =====================================================================
    // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
    //   Warps inside `kNumNonEpilogueThreads`: warp 0 loads A + SFA,
    //   warp 1 loads B + SFB, remaining warps are decode-assist only.
    // =====================================================================
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // Wait for the pool to be ready
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
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load A
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, 1);

                    // TMA load SFA
                    if (block_phase == sched::BlockPhase::Linear1) {
                        // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            m_idx, k_block_idx, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
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
                                m_idx, k_block_idx * kNumL2SFAPerBlockK + sf_group, 1);
                        }
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + kNumL2SFAPerBlockK * BLOCK_M * sizeof(float));
                    }
                }
                __syncwarp();

                if constexpr (!kUseRSMode and kFirstFP4DecodeAssistWarp == 0) {
                    const uint32_t decode_thread_idx =
                        (warp_idx - kNumDispatchWarps) * 32 + lane_idx;
                    wait_fp4_decode_input_ready(stage_idx, phase);
                    decode_fp4_b_stage(stage_idx, decode_thread_idx);
                }
            }
        });

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_b_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;

            const uint32_t shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                    // Packed FP4: each byte covers two K elements, so the
                    // logical K index advances by BLOCK_K (matching the FP8
                    // path), but the TMA box is BLOCK_K/2 bytes along the K
                    // dimension and the underlying tensor descriptor was built
                    // with `(N, K_packed = K/2)` shape on the host side. We
                    // pass `k_idx_packed = k_block_idx * (BLOCK_K / 2)` so the
                    // descriptor coordinate matches the packed K axis.
                    const uint32_t k_idx_packed = k_block_idx * (BLOCK_K / 2);

                    // TMA load packed FP4 weight tile into smem_b_packed.
                    auto b_full_barrier = kUseEarlyBDecode
                        ? decode_full_barriers[stage_idx]
                        : full_barriers[stage_idx];
                    tma::copy<BLOCK_K / 2, LOAD_BLOCK_N, kSwizzleBPackedMode, b_packed_dtype_t>(
                        tensor_map_b_ptr, b_full_barrier, smem_b_packed[stage_idx],
                        k_idx_packed, n_idx, 1);
                }
                __syncwarp();

                if constexpr (!kUseRSMode or kRSStageSFB) {
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
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    if constexpr (kUseEarlyBDecode) {
                        decode_full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_PACKED_SIZE_PER_STAGE);
                    } else {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_PACKED_SIZE_PER_STAGE);
                    }
                }
                __syncwarp();

                if constexpr (!kUseRSMode and kFirstFP4DecodeAssistWarp <= 1) {
                    const uint32_t decode_thread_idx =
                        (1u - kFirstFP4DecodeAssistWarp) * 32 + lane_idx;
                    wait_fp4_decode_input_ready(stage_idx, phase);
                    decode_fp4_b_stage(stage_idx, decode_thread_idx);
                }
            }
        });

    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Idle non-epilogue warps (kNumDispatchWarps+2, +3). They must still
        // participate in the warpgroup-collective `setmaxnreg.dec.sync.aligned`
        // so that the math warpgroup's `warpgroup_reg_alloc` can succeed.
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        if constexpr (!kUseRSMode) {
            const uint32_t non_epilogue_warp_idx = warp_idx - kNumDispatchWarps;
            if (non_epilogue_warp_idx >= kFirstFP4DecodeAssistWarp) {
                const uint32_t decode_thread_idx =
                    (non_epilogue_warp_idx - kFirstFP4DecodeAssistWarp) * 32 + lane_idx;

                scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                             const uint32_t& local_expert_idx,
                                             const uint32_t& num_k_blocks,
                                             const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        wait_fp4_decode_input_ready(stage_idx, phase);
                        decode_fp4_b_stage(stage_idx, decode_thread_idx);
                    }
                });
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
        auto fp4_profile_add = [&](uint32_t slot, uint64_t value) {
            if constexpr (kClockProfile) {
                if (epilogue_thread_idx == 0 and fp4_clock_profile != nullptr) {
                    auto* profile = reinterpret_cast<unsigned long long*>(fp4_clock_profile);
                    atomicAdd(profile + slot, static_cast<unsigned long long>(value));
                }
            }
        };

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M, "Each warpgroup must run exactly one WGMMA per K-block");
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");

        // Sync with dispatch
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const uint32_t valid_m = scheduler.template get_valid_m<false>();

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            float final_accum[kAccumPerThread] = {};

            if constexpr (kUseRSMode) {
                using RSWGMMA = L1RSWGMMA;
                constexpr uint32_t kRSAccumPerThread = RSWGMMA::kNumAccum;  // M=64(weight), N=64(tokens)
                constexpr uint32_t kNumRSNSlices = BLOCK_N / RSWGMMA::M;
                DG_STATIC_ASSERT(kNumRSNSlices == 1 or kNumRSNSlices == 2,
                                 "RS path expects BLOCK_N=64 or BLOCK_N=128");
                DG_STATIC_ASSERT(RSWGMMA::N == BLOCK_M, "RS WGMMA N dimension must cover the token block");
                DG_STATIC_ASSERT(kAccumPerThread == kRSAccumPerThread * kNumRSNSlices,
                                 "RS slices must cover the SS epilogue accumulator layout");
                constexpr uint32_t BLOCK_K_PACKED = BLOCK_K / 2;
                constexpr uint32_t kLdmatrixVecBytes = 16 / sizeof(a_dtype_t);
                const uint32_t rs_lane_row = lane_idx / 4;
                const uint32_t rs_lane_col_pair = lane_idx % 4;
                const uint32_t packed_shift = (rs_lane_col_pair & 1u) * 16u;
                const uint32_t rs_lane_pair_col = (rs_lane_col_pair & 2u) * 2u;
                uint64_t rs_profile_count = 0;
                uint64_t rs_profile_full_wait = 0;
                uint64_t rs_profile_decode = 0;
                uint64_t rs_profile_wgmma = 0;
                uint64_t rs_profile_promote = 0;
                uint64_t rs_profile_t0 = 0;
                uint64_t rs_profile_t1 = 0;

                auto load_rs_packed_nibbles = [&](uint32_t mat,
                                                  uint32_t cur_stage,
                                                  uint32_t k_inner,
                                                  uint32_t rs_n_slice) -> uint32_t {
                    DG_STATIC_ASSERT(BLOCK_K_PACKED == 64 and RSWGMMA::K == 32,
                                     "RS packed-B address path assumes 64-byte K tiles and K=32 WGMMA");
                    const uint32_t addr_lane = mat * 8 + rs_lane_row;
                    const uint32_t addr_tid_g = warp_idx_in_wg * 32 + addr_lane;
                    const uint32_t addr_t_row = (addr_tid_g & 15u) | ((addr_tid_g >> 5) << 4);
                    const uint32_t addr_t_col = ((addr_tid_g >> 4) & 1u) * kLdmatrixVecBytes;
                    const uint32_t n_row = rs_n_slice * RSWGMMA::M + addr_t_row;
                    const uint32_t row_offset = n_row * BLOCK_K_PACKED;
                    const uint32_t raw_col = addr_t_col / 2 + rs_lane_pair_col;
                    const uint32_t swizzle_xor = (addr_t_row & 7u) >> 1;
                    const uint32_t swizzled_col = ((k_inner ^ swizzle_xor) << 4) + raw_col;
                    uint32_t packed_word = 0;
                    if constexpr (kRSDecodePairShfl) {
                        if ((rs_lane_col_pair & 1u) == 0)
                            packed_word = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                smem_b_packed[cur_stage] + row_offset + swizzled_col));
                        packed_word = __shfl_sync(0xffffffff, packed_word, lane_idx & ~1u);
                    } else {
                        packed_word = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                            smem_b_packed[cur_stage] + row_offset + swizzled_col));
                    }
                    return (packed_word >> packed_shift) & 0xffffu;
                };

                auto load_rs_sfb_word = [&](uint32_t mat,
                                            uint32_t cur_k_block_idx,
                                            uint32_t rs_n_slice,
                                            const uint32_t* sfb_base_for_block_phase,
                                            uint32_t sfb_per_expert,
                                            uint32_t sfb_k_words) -> uint32_t {
                    const uint32_t addr_lane = mat * 8 + rs_lane_row;
                    const uint32_t addr_tid_g = warp_idx_in_wg * 32 + addr_lane;
                    const uint32_t addr_t_row = (addr_tid_g & 15u) | ((addr_tid_g >> 5) << 4);
                    const uint32_t n_row = rs_n_slice * RSWGMMA::M + addr_t_row;
                    const uint32_t n_global = n_block_idx * BLOCK_N + n_row;
                    uint32_t sfb_word = 0;
                    if constexpr (kRSBroadcastSFBLoad) {
                        if (rs_lane_col_pair == 0) {
                            if constexpr (kRSStageSFB) {
                                sfb_word = ptx::ld_shared(smem_sfb[stage_idx] + n_row);
                            } else {
                                sfb_word = __ldg(sfb_base_for_block_phase
                                    + local_expert_idx * sfb_per_expert
                                    + n_global * sfb_k_words
                                    + cur_k_block_idx);
                            }
                        }
                        return __shfl_sync(0xffffffff, sfb_word, lane_idx & ~3u);
                    } else {
                        if constexpr (kRSStageSFB)
                            return ptx::ld_shared(smem_sfb[stage_idx] + n_row);
                        return __ldg(sfb_base_for_block_phase
                            + local_expert_idx * sfb_per_expert
                            + n_global * sfb_k_words
                            + cur_k_block_idx);
                    }
                };
                auto unpack_rs_e8m0 = [](uint32_t sfb_word, uint32_t k_inner) -> uint32_t {
                    return (sfb_word >> (k_inner * 8)) & 0xffu;
                };
                auto load_rs_e8m0 = [&](uint32_t mat,
                                        uint32_t cur_k_block_idx,
                                        uint32_t k_inner,
                                        uint32_t rs_n_slice,
                                        const uint32_t* sfb_base_for_block_phase,
                                        uint32_t sfb_per_expert,
                                        uint32_t sfb_k_words) -> uint32_t {
                    return unpack_rs_e8m0(
                        load_rs_sfb_word(mat, cur_k_block_idx, rs_n_slice,
                                         sfb_base_for_block_phase, sfb_per_expert, sfb_k_words),
                        k_inner);
                };

                auto decode_rs_a_reg = [&](uint32_t packed_shifted, uint32_t e8m0) -> uint32_t {
                    if (e8m0 == 0u) {
                        return 0u;
                    } else if constexpr (kFuseScaleBHummingDecode) {
                        if (e8m0 >= 121u and e8m0 <= 149u) {
                            return fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(
                                packed_shifted, e8m0 - 121u);
                        } else {
                            return fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(
                                packed_shifted, e8m0);
                        }
                    } else {
                        return fp4_rs_detail::fp4x4_to_scaled_e4m3x4_e8m0_fast(
                            packed_shifted, e8m0);
                    }
                };
                auto decode_rs_a_reg_lut = [&](uint32_t packed_shifted,
                                               uint32_t scaled_lut_lo,
                                               uint32_t scaled_lut_hi) -> uint32_t {
                    return fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                        packed_shifted, scaled_lut_lo, scaled_lut_hi);
                };
                auto load_rs_sfa_pair = [&](const float* ptr) -> float2 {
                    if constexpr (kRSBroadcastSFALoad) {
                        float2 scale = make_float2(0.0f, 0.0f);
                        if (row_idx == 0) {
                            if constexpr (kRSUseSFAVecLoad) {
                                scale = ptx::ld_shared(reinterpret_cast<const float2*>(ptr));
                            } else {
                                scale = make_float2(ptx::ld_shared(ptr), ptx::ld_shared(ptr + 1));
                            }
                        }
                        scale.x = __shfl_sync(0xffffffff, scale.x, col_idx);
                        scale.y = __shfl_sync(0xffffffff, scale.y, col_idx);
                        return scale;
                    } else if constexpr (kRSUseSFAVecLoad) {
                        return ptx::ld_shared(reinterpret_cast<const float2*>(ptr));
                    } else {
                        return make_float2(ptx::ld_shared(ptr), ptx::ld_shared(ptr + 1));
                    }
                };

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    uint64_t rs_stage_decode = 0;
                    uint64_t rs_stage_wgmma = 0;
                    uint64_t rs_stage_promote = 0;
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0)
                            rs_profile_t0 = clock64();
                    }
                    wait_fp4_decode_input_ready(stage_idx, phase);
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0) {
                            rs_profile_t1 = clock64();
                            rs_profile_full_wait += rs_profile_t1 - rs_profile_t0;
                        }
                    }
                    const auto desc_b_base = mma::sm90::make_smem_desc(smem_a[stage_idx], 1);
                    const uint32_t desc_b_base_lo = __shfl_sync(0xffffffff, desc_b_base.reg32_[0], 0);
                    bool rs_acts_ready = !kUseEarlyBDecode;
                    auto wait_rs_acts_ready = [&]() {
                        if constexpr (kUseEarlyBDecode) {
                            if (!rs_acts_ready) {
                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0)
                                        rs_profile_t0 = clock64();
                                }
                                full_barriers[stage_idx]->wait(phase);
                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0) {
                                        rs_profile_t1 = clock64();
                                        rs_profile_full_wait += rs_profile_t1 - rs_profile_t0;
                                    }
                                }
                                rs_acts_ready = true;
                            }
                        }
                    };

                    #pragma unroll
                    for (uint32_t rs_n_slice = 0; rs_n_slice < kNumRSNSlices; ++ rs_n_slice) {
                        const bool group_k_promote =
                            kRSGroupKPromote and block_phase == sched::BlockPhase::Linear1;
                        if (group_k_promote) {
                            // Linear1 has one SFA value per 128-K stage. Batch the
                            // four RS K/32 WGMMA slices into one warpgroup commit/wait,
                            // then apply SFA once.
                            float rs_accum[kRSAccumPerThread];
                            const uint32_t rs_accum_base = rs_n_slice * kRSAccumPerThread;
                            #pragma unroll
                            for (uint32_t i = 0; i < kRSAccumPerThread; ++ i)
                                rs_accum[i] = 0.0f;

                            #pragma unroll
                            for (uint32_t i = 0; i < kRSAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(rs_accum[i]);

                            uint32_t sfb_word_02 = 0;
                            uint32_t sfb_word_13 = 0;
                            if constexpr (kRSReuseSFBWord) {
                                sfb_word_02 = load_rs_sfb_word(
                                    0, k_block_idx, rs_n_slice,
                                    l1_weights_sf, kL1SFBPerExpert, kL1SFBKWords);
                                sfb_word_13 = load_rs_sfb_word(
                                    1, k_block_idx, rs_n_slice,
                                    l1_weights_sf, kL1SFBPerExpert, kL1SFBKWords);
                            }

                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / RSWGMMA::K; ++ k) {
                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0)
                                        rs_profile_t0 = clock64();
                                }
                                uint32_t e8m0_02, e8m0_13;
                                if constexpr (kRSReuseSFBWord) {
                                    e8m0_02 = unpack_rs_e8m0(sfb_word_02, k);
                                    e8m0_13 = unpack_rs_e8m0(sfb_word_13, k);
                                } else {
                                    e8m0_02 = load_rs_e8m0(0, k_block_idx, k, rs_n_slice,
                                                          l1_weights_sf, kL1SFBPerExpert, kL1SFBKWords);
                                    e8m0_13 = load_rs_e8m0(1, k_block_idx, k, rs_n_slice,
                                                          l1_weights_sf, kL1SFBPerExpert, kL1SFBKWords);
                                }
                                uint32_t a0, a1, a2, a3;
                                if constexpr (kFuseScaleBHummingDecode or kUseDynamicLutDecode) {
                                    a0 = decode_rs_a_reg(load_rs_packed_nibbles(0, stage_idx, k, rs_n_slice), e8m0_02);
                                    a1 = decode_rs_a_reg(load_rs_packed_nibbles(1, stage_idx, k, rs_n_slice), e8m0_13);
                                    a2 = decode_rs_a_reg(load_rs_packed_nibbles(2, stage_idx, k, rs_n_slice), e8m0_02);
                                    a3 = decode_rs_a_reg(load_rs_packed_nibbles(3, stage_idx, k, rs_n_slice), e8m0_13);
                                } else {
                                    const uint64_t scaled_lut_02 =
                                        kUseCommonLutFastPath ?
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0_02) :
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0_02);
                                    const uint64_t scaled_lut_13 =
                                        kUseCommonLutFastPath ?
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0_13) :
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0_13);
                                    const uint32_t lut_02_lo = static_cast<uint32_t>(scaled_lut_02);
                                    const uint32_t lut_02_hi = static_cast<uint32_t>(scaled_lut_02 >> 32);
                                    const uint32_t lut_13_lo = static_cast<uint32_t>(scaled_lut_13);
                                    const uint32_t lut_13_hi = static_cast<uint32_t>(scaled_lut_13 >> 32);
                                    a0 = decode_rs_a_reg_lut(load_rs_packed_nibbles(0, stage_idx, k, rs_n_slice),
                                                             lut_02_lo, lut_02_hi);
                                    a1 = decode_rs_a_reg_lut(load_rs_packed_nibbles(1, stage_idx, k, rs_n_slice),
                                                             lut_13_lo, lut_13_hi);
                                    a2 = decode_rs_a_reg_lut(load_rs_packed_nibbles(2, stage_idx, k, rs_n_slice),
                                                             lut_02_lo, lut_02_hi);
                                    a3 = decode_rs_a_reg_lut(load_rs_packed_nibbles(3, stage_idx, k, rs_n_slice),
                                                             lut_13_lo, lut_13_hi);
                                }
                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0) {
                                        rs_profile_t1 = clock64();
                                        rs_stage_decode += rs_profile_t1 - rs_profile_t0;
                                    }
                                }

                                wait_rs_acts_ready();
                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0)
                                        rs_profile_t0 = clock64();
                                }
                                auto desc_b = desc_b_base;
                                desc_b.reg32_[0] = desc_b_base_lo + k * RSWGMMA::K / 16;
                                // Each RS slice rewrites the register-backed A operand.
                                // Fence it explicitly while still keeping one commit/wait
                                // for the four K slices.
                                ptx::warpgroup_arrive();
                                RSWGMMA::wgmma(a0, a1, a2, a3, desc_b, rs_accum, k != 0);
                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0) {
                                        rs_profile_t1 = clock64();
                                        rs_stage_wgmma += rs_profile_t1 - rs_profile_t0;
                                    }
                                }
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kRSAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(rs_accum[i]);
                            ptx::warpgroup_wait<0>();

                            if constexpr (kClockProfile) {
                                if (epilogue_thread_idx == 0)
                                    rs_profile_t0 = clock64();
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < kRSAccumPerThread / 4; ++ i) {
                                const uint32_t token_0 = i * 8 + col_idx * 2;
                                const float2 scale_a = load_rs_sfa_pair(smem_sfa[stage_idx] + token_0);
                                final_accum[rs_accum_base + i * 4 + 0] += scale_a.x * rs_accum[i * 4 + 0];
                                final_accum[rs_accum_base + i * 4 + 1] += scale_a.y * rs_accum[i * 4 + 1];
                                final_accum[rs_accum_base + i * 4 + 2] += scale_a.x * rs_accum[i * 4 + 2];
                                final_accum[rs_accum_base + i * 4 + 3] += scale_a.y * rs_accum[i * 4 + 3];
                            }
                            if constexpr (kClockProfile) {
                                if (epilogue_thread_idx == 0) {
                                    rs_profile_t1 = clock64();
                                    rs_stage_promote += rs_profile_t1 - rs_profile_t0;
                                }
                            }
                        } else if (kRSL2GroupK2Promote and kL2ActsSFGranK == 32 and
                                   block_phase == sched::BlockPhase::Linear2) {
                            // Diagnostic path: keep two independent K/32
                            // accumulators live so two L2 WGMMAs can share one
                            // commit/wait while preserving per-K activation
                            // scale application.
                            const uint32_t rs_accum_base = rs_n_slice * kRSAccumPerThread;
                            #pragma unroll
                            for (uint32_t k_pair = 0; k_pair < BLOCK_K / RSWGMMA::K; k_pair += 2) {
                                float rs_accum_0[kRSAccumPerThread];
                                float rs_accum_1[kRSAccumPerThread];
                                #pragma unroll
                                for (uint32_t i = 0; i < kRSAccumPerThread; ++ i) {
                                    rs_accum_0[i] = 0.0f;
                                    rs_accum_1[i] = 0.0f;
                                    ptx::warpgroup_fence_operand(rs_accum_0[i]);
                                    ptx::warpgroup_fence_operand(rs_accum_1[i]);
                                }

                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0)
                                        rs_profile_t0 = clock64();
                                }
                                const uint32_t e8m0_02_0 = load_rs_e8m0(
                                    0, k_block_idx, k_pair, rs_n_slice,
                                    l2_weights_sf, kL2SFBPerExpert, kL2SFBKWords);
                                const uint32_t e8m0_13_0 = load_rs_e8m0(
                                    1, k_block_idx, k_pair, rs_n_slice,
                                    l2_weights_sf, kL2SFBPerExpert, kL2SFBKWords);
                                uint32_t a0_0, a1_0, a2_0, a3_0;
                                if constexpr (kFuseScaleBHummingDecode or kUseDynamicLutDecode) {
                                    a0_0 = decode_rs_a_reg(load_rs_packed_nibbles(0, stage_idx, k_pair, rs_n_slice), e8m0_02_0);
                                    a1_0 = decode_rs_a_reg(load_rs_packed_nibbles(1, stage_idx, k_pair, rs_n_slice), e8m0_13_0);
                                    a2_0 = decode_rs_a_reg(load_rs_packed_nibbles(2, stage_idx, k_pair, rs_n_slice), e8m0_02_0);
                                    a3_0 = decode_rs_a_reg(load_rs_packed_nibbles(3, stage_idx, k_pair, rs_n_slice), e8m0_13_0);
                                } else {
                                    const uint64_t scaled_lut_02 =
                                        kUseCommonLutFastPath ?
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0_02_0) :
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0_02_0);
                                    const uint64_t scaled_lut_13 =
                                        kUseCommonLutFastPath ?
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0_13_0) :
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0_13_0);
                                    const uint32_t lut_02_lo = static_cast<uint32_t>(scaled_lut_02);
                                    const uint32_t lut_02_hi = static_cast<uint32_t>(scaled_lut_02 >> 32);
                                    const uint32_t lut_13_lo = static_cast<uint32_t>(scaled_lut_13);
                                    const uint32_t lut_13_hi = static_cast<uint32_t>(scaled_lut_13 >> 32);
                                    a0_0 = decode_rs_a_reg_lut(load_rs_packed_nibbles(0, stage_idx, k_pair, rs_n_slice),
                                                               lut_02_lo, lut_02_hi);
                                    a1_0 = decode_rs_a_reg_lut(load_rs_packed_nibbles(1, stage_idx, k_pair, rs_n_slice),
                                                               lut_13_lo, lut_13_hi);
                                    a2_0 = decode_rs_a_reg_lut(load_rs_packed_nibbles(2, stage_idx, k_pair, rs_n_slice),
                                                               lut_02_lo, lut_02_hi);
                                    a3_0 = decode_rs_a_reg_lut(load_rs_packed_nibbles(3, stage_idx, k_pair, rs_n_slice),
                                                               lut_13_lo, lut_13_hi);
                                }
                                const uint32_t k_next = k_pair + 1;
                                const uint32_t e8m0_02_1 = load_rs_e8m0(
                                    0, k_block_idx, k_next, rs_n_slice,
                                    l2_weights_sf, kL2SFBPerExpert, kL2SFBKWords);
                                const uint32_t e8m0_13_1 = load_rs_e8m0(
                                    1, k_block_idx, k_next, rs_n_slice,
                                    l2_weights_sf, kL2SFBPerExpert, kL2SFBKWords);
                                uint32_t a0_1, a1_1, a2_1, a3_1;
                                if constexpr (kFuseScaleBHummingDecode or kUseDynamicLutDecode) {
                                    a0_1 = decode_rs_a_reg(load_rs_packed_nibbles(0, stage_idx, k_next, rs_n_slice), e8m0_02_1);
                                    a1_1 = decode_rs_a_reg(load_rs_packed_nibbles(1, stage_idx, k_next, rs_n_slice), e8m0_13_1);
                                    a2_1 = decode_rs_a_reg(load_rs_packed_nibbles(2, stage_idx, k_next, rs_n_slice), e8m0_02_1);
                                    a3_1 = decode_rs_a_reg(load_rs_packed_nibbles(3, stage_idx, k_next, rs_n_slice), e8m0_13_1);
                                } else {
                                    const uint64_t scaled_lut_02 =
                                        kUseCommonLutFastPath ?
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0_02_1) :
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0_02_1);
                                    const uint64_t scaled_lut_13 =
                                        kUseCommonLutFastPath ?
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0_13_1) :
                                        fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0_13_1);
                                    const uint32_t lut_02_lo = static_cast<uint32_t>(scaled_lut_02);
                                    const uint32_t lut_02_hi = static_cast<uint32_t>(scaled_lut_02 >> 32);
                                    const uint32_t lut_13_lo = static_cast<uint32_t>(scaled_lut_13);
                                    const uint32_t lut_13_hi = static_cast<uint32_t>(scaled_lut_13 >> 32);
                                    a0_1 = decode_rs_a_reg_lut(load_rs_packed_nibbles(0, stage_idx, k_next, rs_n_slice),
                                                               lut_02_lo, lut_02_hi);
                                    a1_1 = decode_rs_a_reg_lut(load_rs_packed_nibbles(1, stage_idx, k_next, rs_n_slice),
                                                               lut_13_lo, lut_13_hi);
                                    a2_1 = decode_rs_a_reg_lut(load_rs_packed_nibbles(2, stage_idx, k_next, rs_n_slice),
                                                               lut_02_lo, lut_02_hi);
                                    a3_1 = decode_rs_a_reg_lut(load_rs_packed_nibbles(3, stage_idx, k_next, rs_n_slice),
                                                               lut_13_lo, lut_13_hi);
                                }
                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0) {
                                        rs_profile_t1 = clock64();
                                        rs_stage_decode += rs_profile_t1 - rs_profile_t0;
                                    }
                                }

                                wait_rs_acts_ready();
                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0)
                                        rs_profile_t0 = clock64();
                                }
                                auto desc_b_0 = desc_b_base;
                                desc_b_0.reg32_[0] = desc_b_base_lo + k_pair * RSWGMMA::K / 16;
                                ptx::warpgroup_arrive();
                                RSWGMMA::wgmma(a0_0, a1_0, a2_0, a3_0, desc_b_0, rs_accum_0, false);
                                auto desc_b_1 = desc_b_base;
                                desc_b_1.reg32_[0] = desc_b_base_lo + k_next * RSWGMMA::K / 16;
                                ptx::warpgroup_arrive();
                                RSWGMMA::wgmma(a0_1, a1_1, a2_1, a3_1, desc_b_1, rs_accum_1, false);
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kRSAccumPerThread; ++ i) {
                                    ptx::warpgroup_fence_operand(rs_accum_0[i]);
                                    ptx::warpgroup_fence_operand(rs_accum_1[i]);
                                }
                                ptx::warpgroup_wait<0>();
                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0) {
                                        rs_profile_t1 = clock64();
                                        rs_stage_wgmma += rs_profile_t1 - rs_profile_t0;
                                    }
                                }

                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0)
                                        rs_profile_t0 = clock64();
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < kRSAccumPerThread / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const float2 scale_a = load_rs_sfa_pair(
                                        smem_sfa[stage_idx] + k_pair * BLOCK_M + token_0);
                                    final_accum[rs_accum_base + i * 4 + 0] += scale_a.x * rs_accum_0[i * 4 + 0];
                                    final_accum[rs_accum_base + i * 4 + 1] += scale_a.y * rs_accum_0[i * 4 + 1];
                                    final_accum[rs_accum_base + i * 4 + 2] += scale_a.x * rs_accum_0[i * 4 + 2];
                                    final_accum[rs_accum_base + i * 4 + 3] += scale_a.y * rs_accum_0[i * 4 + 3];
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < kRSAccumPerThread / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const float2 scale_a = load_rs_sfa_pair(
                                        smem_sfa[stage_idx] + k_next * BLOCK_M + token_0);
                                    final_accum[rs_accum_base + i * 4 + 0] += scale_a.x * rs_accum_1[i * 4 + 0];
                                    final_accum[rs_accum_base + i * 4 + 1] += scale_a.y * rs_accum_1[i * 4 + 1];
                                    final_accum[rs_accum_base + i * 4 + 2] += scale_a.x * rs_accum_1[i * 4 + 2];
                                    final_accum[rs_accum_base + i * 4 + 3] += scale_a.y * rs_accum_1[i * 4 + 3];
                                }
                                if constexpr (kClockProfile) {
                                    if (epilogue_thread_idx == 0) {
                                        rs_profile_t1 = clock64();
                                        rs_stage_promote += rs_profile_t1 - rs_profile_t0;
                                    }
                                }
                            }
                        } else {
                        const bool is_l1 = block_phase == sched::BlockPhase::Linear1;
                        const uint32_t* sfb_base = is_l1 ? l1_weights_sf : l2_weights_sf;
                        const uint32_t sfb_per_expert = is_l1 ? kL1SFBPerExpert : kL2SFBPerExpert;
                        const uint32_t sfb_k_words = is_l1 ? kL1SFBKWords : kL2SFBKWords;
                        uint32_t sfb_word_02 = 0;
                        uint32_t sfb_word_13 = 0;
                        if constexpr (kRSReuseSFBWord) {
                            sfb_word_02 = load_rs_sfb_word(
                                0, k_block_idx, rs_n_slice,
                                sfb_base, sfb_per_expert, sfb_k_words);
                            sfb_word_13 = load_rs_sfb_word(
                                1, k_block_idx, rs_n_slice,
                                sfb_base, sfb_per_expert, sfb_k_words);
                        }

                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / RSWGMMA::K; ++ k) {
                            float rs_accum[kRSAccumPerThread];
                            if constexpr (kClockProfile) {
                                if (epilogue_thread_idx == 0)
                                    rs_profile_t0 = clock64();
                            }
                            uint32_t e8m0_02, e8m0_13;
                            if constexpr (kRSReuseSFBWord) {
                                e8m0_02 = unpack_rs_e8m0(sfb_word_02, k);
                                e8m0_13 = unpack_rs_e8m0(sfb_word_13, k);
                            } else {
                                e8m0_02 = load_rs_e8m0(0, k_block_idx, k, rs_n_slice,
                                                      sfb_base, sfb_per_expert, sfb_k_words);
                                e8m0_13 = load_rs_e8m0(1, k_block_idx, k, rs_n_slice,
                                                      sfb_base, sfb_per_expert, sfb_k_words);
                            }
                            uint32_t a0, a1, a2, a3;
                            if constexpr (kFuseScaleBHummingDecode or kUseDynamicLutDecode) {
                                a0 = decode_rs_a_reg(load_rs_packed_nibbles(0, stage_idx, k, rs_n_slice), e8m0_02);
                                a1 = decode_rs_a_reg(load_rs_packed_nibbles(1, stage_idx, k, rs_n_slice), e8m0_13);
                                a2 = decode_rs_a_reg(load_rs_packed_nibbles(2, stage_idx, k, rs_n_slice), e8m0_02);
                                a3 = decode_rs_a_reg(load_rs_packed_nibbles(3, stage_idx, k, rs_n_slice), e8m0_13);
                            } else {
                                const uint64_t scaled_lut_02 =
                                    kUseCommonLutFastPath ?
                                    fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0_02) :
                                    fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0_02);
                                const uint64_t scaled_lut_13 =
                                    kUseCommonLutFastPath ?
                                    fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_common_const(e8m0_13) :
                                    fp4_rs_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0_13);
                                const uint32_t lut_02_lo = static_cast<uint32_t>(scaled_lut_02);
                                const uint32_t lut_02_hi = static_cast<uint32_t>(scaled_lut_02 >> 32);
                                const uint32_t lut_13_lo = static_cast<uint32_t>(scaled_lut_13);
                                const uint32_t lut_13_hi = static_cast<uint32_t>(scaled_lut_13 >> 32);
                                a0 = decode_rs_a_reg_lut(load_rs_packed_nibbles(0, stage_idx, k, rs_n_slice),
                                                         lut_02_lo, lut_02_hi);
                                a1 = decode_rs_a_reg_lut(load_rs_packed_nibbles(1, stage_idx, k, rs_n_slice),
                                                         lut_13_lo, lut_13_hi);
                                a2 = decode_rs_a_reg_lut(load_rs_packed_nibbles(2, stage_idx, k, rs_n_slice),
                                                         lut_02_lo, lut_02_hi);
                                a3 = decode_rs_a_reg_lut(load_rs_packed_nibbles(3, stage_idx, k, rs_n_slice),
                                                         lut_13_lo, lut_13_hi);
                            }
                            if constexpr (kClockProfile) {
                                if (epilogue_thread_idx == 0) {
                                    rs_profile_t1 = clock64();
                                    rs_stage_decode += rs_profile_t1 - rs_profile_t0;
                                }
                            }

                            wait_rs_acts_ready();
                            if constexpr (kClockProfile) {
                                if (epilogue_thread_idx == 0)
                                    rs_profile_t0 = clock64();
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < kRSAccumPerThread; ++ i) {
                                rs_accum[i] = 0.0f;
                                ptx::warpgroup_fence_operand(rs_accum[i]);
                            }
                            ptx::warpgroup_arrive();
                            auto desc_b = desc_b_base;
                            desc_b.reg32_[0] = desc_b_base_lo + k * RSWGMMA::K / 16;
                            RSWGMMA::wgmma(a0, a1, a2, a3, desc_b, rs_accum, false);
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kRSAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(rs_accum[i]);
                            ptx::warpgroup_wait<0>();
                            if constexpr (kClockProfile) {
                                if (epilogue_thread_idx == 0) {
                                    rs_profile_t1 = clock64();
                                    rs_stage_wgmma += rs_profile_t1 - rs_profile_t0;
                                }
                            }

                            const uint32_t sfa_group =
                                kL2ActsSFGranK == 32 ? k : (k < 2 ? 0u : 1u);
                            const uint32_t rs_accum_base = rs_n_slice * kRSAccumPerThread;
                            if constexpr (kClockProfile) {
                                if (epilogue_thread_idx == 0)
                                    rs_profile_t0 = clock64();
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < kRSAccumPerThread / 4; ++ i) {
                                const uint32_t token_0 = i * 8 + col_idx * 2;
                                float2 scale_a;
                                if (block_phase == sched::BlockPhase::Linear1) {
                                    scale_a = load_rs_sfa_pair(smem_sfa[stage_idx] + token_0);
                                } else {
                                    scale_a = load_rs_sfa_pair(smem_sfa[stage_idx] + sfa_group * BLOCK_M + token_0);
                                }
                                final_accum[rs_accum_base + i * 4 + 0] += scale_a.x * rs_accum[i * 4 + 0];
                                final_accum[rs_accum_base + i * 4 + 1] += scale_a.y * rs_accum[i * 4 + 1];
                                final_accum[rs_accum_base + i * 4 + 2] += scale_a.x * rs_accum[i * 4 + 2];
                                final_accum[rs_accum_base + i * 4 + 3] += scale_a.y * rs_accum[i * 4 + 3];
                            }
                            if constexpr (kClockProfile) {
                                if (epilogue_thread_idx == 0) {
                                    rs_profile_t1 = clock64();
                                    rs_stage_promote += rs_profile_t1 - rs_profile_t0;
                                }
                            }
                        }
                        }
                    }

                    if (lane_idx == 0)
                        empty_barriers[stage_idx]->arrive();
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0) {
                            ++ rs_profile_count;
                            rs_profile_decode += rs_stage_decode;
                            rs_profile_wgmma += rs_stage_wgmma;
                            rs_profile_promote += rs_stage_promote;
                        }
                    }
                }

                if (block_phase == sched::BlockPhase::Linear1) {
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0)
                            rs_profile_t0 = clock64();
                    }
                    #pragma unroll
                    for (uint32_t rs_n_slice = 0; rs_n_slice < kNumRSNSlices; ++ rs_n_slice) {
                        const uint32_t rs_accum_base = rs_n_slice * kRSAccumPerThread;
                        #pragma unroll
                        for (uint32_t i = 0; i < kRSAccumPerThread / 4; ++ i) {
                            const uint32_t token_0 = i * 8 + col_idx * 2;
                            const uint32_t token_1 = token_0 + 1;
                            if constexpr (kRSGuardTransposeValid) {
                                if (token_0 < valid_m) {
                                    smem_rs_accum[token_0 * RSWGMMA::M + r_0] = final_accum[rs_accum_base + i * 4 + 0];
                                    smem_rs_accum[token_0 * RSWGMMA::M + r_1] = final_accum[rs_accum_base + i * 4 + 2];
                                }
                                if (token_1 < valid_m) {
                                    smem_rs_accum[token_1 * RSWGMMA::M + r_0] = final_accum[rs_accum_base + i * 4 + 1];
                                    smem_rs_accum[token_1 * RSWGMMA::M + r_1] = final_accum[rs_accum_base + i * 4 + 3];
                                }
                            } else {
                                smem_rs_accum[token_0 * RSWGMMA::M + r_0] = final_accum[rs_accum_base + i * 4 + 0];
                                smem_rs_accum[token_1 * RSWGMMA::M + r_0] = final_accum[rs_accum_base + i * 4 + 1];
                                smem_rs_accum[token_0 * RSWGMMA::M + r_1] = final_accum[rs_accum_base + i * 4 + 2];
                                smem_rs_accum[token_1 * RSWGMMA::M + r_1] = final_accum[rs_accum_base + i * 4 + 3];
                            }
                        }
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        #pragma unroll
                        for (uint32_t i = 0; i < kRSAccumPerThread / 4; ++ i) {
                            const uint32_t out_col = i * 8 + col_idx * 2;
                            const uint32_t out_chunk = rs_n_slice * (kRSAccumPerThread / 4) + i;
                            if constexpr (kRSGuardTransposeValid) {
                                const bool valid_out_r0 = r_0 < valid_m;
                                const bool valid_out_r1 = r_1 < valid_m;
                                if constexpr (kRSUseTransposeVecLoad) {
                                    const float2 row_0 = valid_out_r0
                                        ? ptx::ld_shared(reinterpret_cast<const float2*>(
                                            smem_rs_accum + r_0 * RSWGMMA::M + out_col))
                                        : make_float2(0.0f, 0.0f);
                                    const float2 row_1 = valid_out_r1
                                        ? ptx::ld_shared(reinterpret_cast<const float2*>(
                                            smem_rs_accum + r_1 * RSWGMMA::M + out_col))
                                        : make_float2(0.0f, 0.0f);
                                    final_accum[out_chunk * 4 + 0] = row_0.x;
                                    final_accum[out_chunk * 4 + 1] = row_0.y;
                                    final_accum[out_chunk * 4 + 2] = row_1.x;
                                    final_accum[out_chunk * 4 + 3] = row_1.y;
                                } else {
                                    final_accum[out_chunk * 4 + 0] = valid_out_r0
                                        ? smem_rs_accum[r_0 * RSWGMMA::M + out_col + 0] : 0.0f;
                                    final_accum[out_chunk * 4 + 1] = valid_out_r0
                                        ? smem_rs_accum[r_0 * RSWGMMA::M + out_col + 1] : 0.0f;
                                    final_accum[out_chunk * 4 + 2] = valid_out_r1
                                        ? smem_rs_accum[r_1 * RSWGMMA::M + out_col + 0] : 0.0f;
                                    final_accum[out_chunk * 4 + 3] = valid_out_r1
                                        ? smem_rs_accum[r_1 * RSWGMMA::M + out_col + 1] : 0.0f;
                                }
                            } else if constexpr (kRSUseTransposeVecLoad) {
                                const auto row_0 = ptx::ld_shared(reinterpret_cast<const float2*>(
                                    smem_rs_accum + r_0 * RSWGMMA::M + out_col));
                                const auto row_1 = ptx::ld_shared(reinterpret_cast<const float2*>(
                                    smem_rs_accum + r_1 * RSWGMMA::M + out_col));
                                final_accum[out_chunk * 4 + 0] = row_0.x;
                                final_accum[out_chunk * 4 + 1] = row_0.y;
                                final_accum[out_chunk * 4 + 2] = row_1.x;
                                final_accum[out_chunk * 4 + 3] = row_1.y;
                            } else {
                                final_accum[out_chunk * 4 + 0] = smem_rs_accum[r_0 * RSWGMMA::M + out_col + 0];
                                final_accum[out_chunk * 4 + 1] = smem_rs_accum[r_0 * RSWGMMA::M + out_col + 1];
                                final_accum[out_chunk * 4 + 2] = smem_rs_accum[r_1 * RSWGMMA::M + out_col + 0];
                                final_accum[out_chunk * 4 + 3] = smem_rs_accum[r_1 * RSWGMMA::M + out_col + 1];
                            }
                        }
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0) {
                            rs_profile_t1 = clock64();
                            rs_profile_promote += rs_profile_t1 - rs_profile_t0;
                        }
                    }
                }

                if constexpr (kClockProfile) {
                    if (epilogue_thread_idx == 0) {
                        const uint32_t slot_base = block_phase == sched::BlockPhase::Linear1 ? 0 : 8;
                        fp4_profile_add(slot_base, rs_profile_count);
                        fp4_profile_add(slot_base + 1, rs_profile_full_wait);
                        fp4_profile_add(slot_base + 2, rs_profile_decode);
                        fp4_profile_add(slot_base + 3, rs_profile_wgmma);
                        fp4_profile_add(slot_base + 4, rs_profile_promote);
                    }
                }

            } else {
                float accum[kAccumPerThread];
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                uint64_t fp4_profile_t0 = 0;
                uint64_t fp4_profile_t_full_wait_done = 0;
                uint64_t fp4_profile_t_decode_done = 0;
                uint64_t fp4_profile_t_wgmma_lo_done = 0;
                uint64_t fp4_profile_t_promote_lo_done = 0;
                uint64_t fp4_profile_t_wgmma_hi_done = 0;
                uint64_t fp4_profile_t_promote_done = 0;
                if constexpr (kClockProfile) {
                    if (epilogue_thread_idx == 0)
                        fp4_profile_t0 = clock64();
                }
                full_barriers[stage_idx]->wait(phase);
                if constexpr (kClockProfile) {
                    if (epilogue_thread_idx == 0)
                        fp4_profile_t_full_wait_done = clock64();
                }

                // Read SF (must precede warpgroup_arrive)
                float scale_a_0_lo, scale_a_1_lo;
                float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
                if (block_phase == sched::BlockPhase::Linear1) {
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + epilogue_wg_idx * WGMMA::M + r_0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + epilogue_wg_idx * WGMMA::M + r_1);
                } else if constexpr (kL2ActsSFGranK == 64) {
                    // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + epilogue_wg_idx * WGMMA::M + r_0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + epilogue_wg_idx * WGMMA::M + r_1);
                    scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + epilogue_wg_idx * WGMMA::M + r_0);
                    scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + epilogue_wg_idx * WGMMA::M + r_1);
                }

                // ----- FP4 → E4M3 dequant of the packed weight tile -----
                // The packed FP4 tile in `smem_b_packed[stage_idx]` is decoded
                // into the E4M3 tile in `smem_b[stage_idx]` with the per-32
                // UE8M0 SFB baked in via `fp4x4_to_scaled_e4m3x4_humming`.
                // After this call, `smem_b[stage]` is byte-equivalent to a
                // pre-scaled FP8 weight tile: the subsequent SS-mode WGMMA
                // accumulator already includes SFB, and only SFA needs to be
                // applied in the promote loop below.
                //
                // Non-epilogue warps assist the math warpgroup. Decode work
                // is partitioned over the assist threads plus the
                // epilogue/math threads, then all participants rendezvous
                // before WGMMA reads the decoded shared tile.
                if constexpr (kUseKGPipelineDecode) {
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0)
                            fp4_profile_t_decode_done = fp4_profile_t_full_wait_done;
                    }
                } else {
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
                                kUseKGPairDecode, kUseVectorStoreDecode,
                                kSkipZeroSFBDecode, kUseDynamicLutDecode,
                                kUseCommonLutFastPath, kFuseScaleBHummingDecode>(
                                decode_thread_idx, kNumFP4DecodeWorkerThreads,
                                smem_b_packed[stage_idx], smem_b[stage_idx], smem_sfb[stage_idx]);
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
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0)
                            fp4_profile_t_decode_done = clock64();
                    }
                }

                if (block_phase == sched::BlockPhase::Linear1) {
                    // Single per-128 K-block WGMMA group
                    if constexpr (kUseKGPipelineDecode) {
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            ptx::sync_aligned(kNumFP4DecodeBarrierThreads, kFP4DecodeBarrierIdx);
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + epilogue_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + epilogue_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();
                    }
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0)
                            fp4_profile_t_wgmma_lo_done = clock64();
                    }

                    if (lane_idx == 0)
                        empty_barriers[stage_idx]->arrive();

                    // L1: SFB is already baked into the decoded E4M3 tile by
                    // `fp4x4_to_scaled_e4m3x4_humming`, so the gate/up scales
                    // do NOT need to be re-applied here (in contrast to the
                    // pure-FP8 path, which still needs gate_sf/up_sf since FP8
                    // weights are stored unscaled). Only SFA remains.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                    }
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0) {
                            fp4_profile_t_promote_done = clock64();
                            fp4_profile_add(0, 1);
                            fp4_profile_add(1, fp4_profile_t_full_wait_done - fp4_profile_t0);
                            fp4_profile_add(2, fp4_profile_t_decode_done - fp4_profile_t_full_wait_done);
                            fp4_profile_add(3, fp4_profile_t_wgmma_lo_done - fp4_profile_t_decode_done);
                            fp4_profile_add(4, fp4_profile_t_promote_done - fp4_profile_t_wgmma_lo_done);
                        }
                    }
                } else {
                    if constexpr (kL2ActsSFGranK == 32) {
                    // L2 BLOCK_N=64: L1 produced 32-column FP8 chunks with
                    // independent SF, so promote each WGMMA::K=32 slice with
                    // its own activation scale.
                    #pragma unroll
                    for (uint32_t sf_group = 0; sf_group < kNumL2SFAPerBlockK; ++ sf_group) {
                        const float scale_a_0 = ptx::ld_shared(
                            smem_sfa[stage_idx] + sf_group * BLOCK_M + epilogue_wg_idx * WGMMA::M + r_0);
                        const float scale_a_1 = ptx::ld_shared(
                            smem_sfa[stage_idx] + sf_group * BLOCK_M + epilogue_wg_idx * WGMMA::M + r_1);
                        const uint32_t k_off = sf_group * WGMMA::K;
                        if constexpr (kUseKGPipelineDecode)
                            ptx::sync_aligned(kNumFP4DecodeBarrierThreads, kFP4DecodeBarrierIdx);
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + epilogue_wg_idx * WGMMA::M * BLOCK_K + k_off, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + k_off, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, false);
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();
                        if constexpr (kClockProfile) {
                            if (epilogue_thread_idx == 0) {
                                if (sf_group == 0)
                                    fp4_profile_t_wgmma_lo_done = clock64();
                                if (sf_group + 1 == kNumL2SFAPerBlockK)
                                    fp4_profile_t_wgmma_hi_done = clock64();
                            }
                        }

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += scale_a_0 * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0 * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1 * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1 * accum[i*4+3];
                        }
                        if constexpr (kClockProfile) {
                            if (epilogue_thread_idx == 0 and sf_group == 0)
                                fp4_profile_t_promote_lo_done = clock64();
                        }
                    }

                    if (lane_idx == 0)
                        empty_barriers[stage_idx]->arrive();

                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0) {
                            fp4_profile_t_promote_done = clock64();
                            fp4_profile_add(8, 1);
                            fp4_profile_add(9, fp4_profile_t_full_wait_done - fp4_profile_t0);
                            fp4_profile_add(10, fp4_profile_t_decode_done - fp4_profile_t_full_wait_done);
                            fp4_profile_add(11, (fp4_profile_t_wgmma_lo_done - fp4_profile_t_decode_done) +
                                                (fp4_profile_t_wgmma_hi_done - fp4_profile_t_promote_lo_done));
                            fp4_profile_add(12, (fp4_profile_t_promote_lo_done - fp4_profile_t_wgmma_lo_done) +
                                                (fp4_profile_t_promote_done - fp4_profile_t_wgmma_hi_done));
                        }
                    }
                    } else {
                    // L2: split BLOCK_K=128 into two halves (per-64 SFA), each 2 WGMMAs.
                    // First half: K=0..63, SFA = scale_a_*_lo
                    if constexpr (kUseKGPipelineDecode) {
                        #pragma unroll
                        for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                            ptx::sync_aligned(kNumFP4DecodeBarrierThreads, kFP4DecodeBarrierIdx);
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + epilogue_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + epilogue_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();
                    }
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0)
                            fp4_profile_t_wgmma_lo_done = clock64();
                    }

                    // L2 first half: SFB baked into decoded E4M3 tile by humming.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                    }
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0)
                            fp4_profile_t_promote_lo_done = clock64();
                    }

                    // Second half: K=64..127, SFA = scale_a_*_hi
                    if constexpr (kUseKGPipelineDecode) {
                        #pragma unroll
                        for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                            ptx::sync_aligned(kNumFP4DecodeBarrierThreads, kFP4DecodeBarrierIdx);
                            const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + epilogue_wg_idx * WGMMA::M * BLOCK_K + k_off, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + k_off, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                            const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + epilogue_wg_idx * WGMMA::M * BLOCK_K + k_off, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + k_off, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();
                    }
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0)
                            fp4_profile_t_wgmma_hi_done = clock64();
                    }

                    if (lane_idx == 0)
                        empty_barriers[stage_idx]->arrive();

                    // L2 second half: SFB baked into decoded E4M3 tile by humming.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] += scale_a_0_hi * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0_hi * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1_hi * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1_hi * accum[i*4+3];
                    }
                    if constexpr (kClockProfile) {
                        if (epilogue_thread_idx == 0) {
                            fp4_profile_t_promote_done = clock64();
                            fp4_profile_add(8, 1);
                            fp4_profile_add(9, fp4_profile_t_full_wait_done - fp4_profile_t0);
                            fp4_profile_add(10, fp4_profile_t_decode_done - fp4_profile_t_full_wait_done);
                            fp4_profile_add(11, (fp4_profile_t_wgmma_lo_done - fp4_profile_t_decode_done) +
                                                (fp4_profile_t_wgmma_hi_done - fp4_profile_t_promote_lo_done));
                            fp4_profile_add(12, (fp4_profile_t_promote_lo_done - fp4_profile_t_wgmma_lo_done) +
                                                (fp4_profile_t_promote_done - fp4_profile_t_wgmma_hi_done));
                        }
                    }
                    }
                }
            }
            }

            // Skip epilogue when block is past valid M (still must release via empty)
            if (epilogue_wg_idx * WG_BLOCK_M >= valid_m) {
                // Trigger any combine/sync logic minimally
                if (block_phase == sched::BlockPhase::Linear1)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                else
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const uint32_t row_offset_r0 = epilogue_wg_idx * WG_BLOCK_M + r_0;
            const uint32_t row_offset_r1 = epilogue_wg_idx * WG_BLOCK_M + r_1;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;

            if (block_phase == sched::BlockPhase::Linear1) {
                // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                // Layout in `final_accum`:
                //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
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

                // Apply token weight: SwiGLU * topk_weight (single load per row)
                float weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                    .get_data_buffer(m_idx + row_offset_r0)
                    .get_base_ptr<float>() : 0.0f;
                float weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                    .get_data_buffer(m_idx + row_offset_r1)
                    .get_base_ptr<float>() : 0.0f;
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    swiglu_r0[p][0] *= weight_r0;
                    swiglu_r0[p][1] *= weight_r0;
                    swiglu_r1[p][0] *= weight_r1;
                    swiglu_r1[p][1] *= weight_r1;
                }
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

                // Compute SF and inverse SF for each row
                float sf_r0, sf_inv_r0;
                float sf_r1, sf_inv_r1;
                {
                    float2 amax_pair = {amax_r0, amax_r1};
                    float2 sf_pair, sf_inv_pair;
                    math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                    sf_r0 = sf_pair.x; sf_inv_r0 = sf_inv_pair.x;
                    sf_r1 = sf_pair.y; sf_inv_r1 = sf_inv_pair.y;
                }

                // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                // The L1-output TMA store descriptor is built with swizzle_mode = 0
                // to match this plain row-major SMEM staging tile.
                //
                // Per pair `p`, each thread holds 4 FP8 values to write at:
                //   (row r_0, cols p*8 + col_idx*2 + {0,1})  -> packed as fp8x2 (2 bytes)
                //   (row r_1, cols p*8 + col_idx*2 + {0,1})  -> packed as fp8x2 (2 bytes)
                auto* smem_cd_l1_wg = smem_cd_l1 + epilogue_wg_idx * WG_BLOCK_M * L1_OUT_BLOCK_N;
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const float v00 = swiglu_r0[p][0] * sf_inv_r0;
                    const float v01 = swiglu_r0[p][1] * sf_inv_r0;
                    const float v10 = swiglu_r1[p][0] * sf_inv_r1;
                    const float v11 = swiglu_r1[p][1] * sf_inv_r1;

                    const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                    const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

                    const uint32_t col = p * 8 + col_idx * 2;
                    auto* p0 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_0 * L1_OUT_BLOCK_N + col);
                    auto* p1 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_1 * L1_OUT_BLOCK_N + col);
                    if (valid_r0)
                        *p0 = r0_pair.__x;
                    if (valid_r1)
                        *p1 = r1_pair.__x;
                }

                // Write SF as float at `[token, n_block_idx]` in L2 acts SF buffer (per-64 layout).
                // Each row is contributed by lanes col_idx ∈ {0..3}; only col_idx == 0 writes.
                if (col_idx == 0) {
                    auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                    // SF buffer is (kNumPaddedSFPoolTokens × kIntermediateHidden/64), MN-major:
                    //   addr[k_idx * num_padded_sf_pool_tokens + token_idx]
                    const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                    const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                    const uint32_t k_sf_idx = n_block_idx;  // one per-64 SF per L1 block
                    if (valid_r0)
                        sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0;
                    if (valid_r1)
                        sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1;
                }

                // Sync the warpgroup before TMA store
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

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
                    const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                    cute::tma_store_fence();
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_l1_output,
                        smem_cd_l1 + epilogue_wg_idx * WG_BLOCK_M * L1_OUT_BLOCK_N,
                        out_n_idx,
                        m_idx + epilogue_wg_idx * WG_BLOCK_M);
                    cute::tma_store_arrive();
                }
                __syncwarp();
                ptx::tma_store_wait<0>();

                // Notify L2 that this N block's L1 output (and SF) is ready
                if constexpr (kL2ArrivalCounter) {
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
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
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                if constexpr (kUseRSMode and kRSDirectL2Scatter) {
                    using RSWGMMA = L2RSWGMMA;
                    constexpr uint32_t kRSAccumPerThread = RSWGMMA::kNumAccum;
                    constexpr uint32_t kNumRSNSlices = BLOCK_N / RSWGMMA::M;

                    auto write_remote_bf16_scalar = [&](uint32_t row, uint32_t col, float value) {
                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + row);
                        const auto dst_token = combine_token_buffer.get_rank_buffer(src_metadata.topk_idx)
                                               .get_data_buffer(src_metadata.token_idx);
                        auto dst_ptr = math::advance_ptr<uint16_t>(
                            dst_token.get_base_ptr(),
                            n_idx * sizeof(nv_bfloat16) + col * sizeof(nv_bfloat16));
                        float value_copy = value;
                        float value_dup = value;
                        const uint32_t packed = math::cast_into_bf16_and_pack(value_copy, value_dup);
                        *sym_buffer.map(dst_ptr, src_metadata.rank_idx) = static_cast<uint16_t>(packed);
                    };

                    #pragma unroll
                    for (uint32_t rs_n_slice = 0; rs_n_slice < kNumRSNSlices; ++ rs_n_slice) {
                        const uint32_t n_row_0 = rs_n_slice * RSWGMMA::M + r_0;
                        const uint32_t n_row_1 = n_row_0 + 8;
                        const uint32_t rs_accum_base = rs_n_slice * kRSAccumPerThread;
                        #pragma unroll
                        for (uint32_t i = 0; i < kRSAccumPerThread / 4; ++ i) {
                            const uint32_t token_0 = i * 8 + col_idx * 2;
                            const uint32_t token_1 = token_0 + 1;
                            if (token_0 < valid_m) {
                                write_remote_bf16_scalar(token_0, n_row_0, final_accum[rs_accum_base + i * 4 + 0]);
                                write_remote_bf16_scalar(token_0, n_row_1, final_accum[rs_accum_base + i * 4 + 2]);
                            }
                            if (token_1 < valid_m) {
                                write_remote_bf16_scalar(token_1, n_row_0, final_accum[rs_accum_base + i * 4 + 1]);
                                write_remote_bf16_scalar(token_1, n_row_1, final_accum[rs_accum_base + i * 4 + 3]);
                            }
                        }
                    }
                } else if constexpr (kUseRSMode) {
                    using RSWGMMA = L2RSWGMMA;
                    constexpr uint32_t kRSAccumPerThread = RSWGMMA::kNumAccum;
                    constexpr uint32_t kNumRSNSlices = BLOCK_N / RSWGMMA::M;
                    auto write_bf16_scalar = [&](uint32_t row, uint32_t col, float value) {
                        auto smem_ptr = smem_cd_l2
                            + epilogue_wg_idx * WG_BLOCK_M * BLOCK_N
                            + row * BLOCK_N
                            + col;
                        float value_copy = value;
                        float value_dup = value;
                        const uint32_t packed = math::cast_into_bf16_and_pack(value_copy, value_dup);
                        *reinterpret_cast<uint16_t*>(smem_ptr) = static_cast<uint16_t>(packed);
                    };

                    #pragma unroll
                    for (uint32_t rs_n_slice = 0; rs_n_slice < kNumRSNSlices; ++ rs_n_slice) {
                        const uint32_t n_row_0 = rs_n_slice * RSWGMMA::M + r_0;
                        const uint32_t n_row_1 = n_row_0 + 8;
                        const uint32_t rs_accum_base = rs_n_slice * kRSAccumPerThread;
                        #pragma unroll
                        for (uint32_t i = 0; i < kRSAccumPerThread / 4; ++ i) {
                            const uint32_t token_0 = i * 8 + col_idx * 2;
                            const uint32_t token_1 = token_0 + 1;
                            if (token_0 < valid_m) {
                                write_bf16_scalar(token_0, n_row_0, final_accum[rs_accum_base + i * 4 + 0]);
                                write_bf16_scalar(token_0, n_row_1, final_accum[rs_accum_base + i * 4 + 2]);
                            }
                            if (token_1 < valid_m) {
                                write_bf16_scalar(token_1, n_row_0, final_accum[rs_accum_base + i * 4 + 1]);
                                write_bf16_scalar(token_1, n_row_1, final_accum[rs_accum_base + i * 4 + 3]);
                            }
                        }
                    }
                } else {
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
                                + epilogue_wg_idx * WG_BLOCK_M * BLOCK_N
                                + row * BLOCK_N
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

                if constexpr (!(kUseRSMode and kRSDirectL2Scatter)) {
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Scatter to remote ranks via NVLink (one row per warp-pair)
                    // Each warpgroup-warp covers 8 unique rows × 2 (r_0 + r_1 doubled by warps)
                    // Lane group of 16 within a warp → 1 row.
                    const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                    const uint32_t lane_in_row = lane_idx % 16;
                    constexpr uint32_t kColsPerScatterLane = BLOCK_N / 16;
                    static_assert(BLOCK_N % 16 == 0, "Scatter layout expects an even lane partition");
                    static_assert(kColsPerScatterLane == 4 or kColsPerScatterLane == 8,
                                  "L2 scatter currently supports BLOCK_N=64 or 128");

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + row_in_wg;
                        if (m_idx_in_block >= valid_m) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;

                        // BLOCK_N=128 scatters 8 BF16s/lane (=16B, uint4).  For
                        // BLOCK_N=64 each lane owns 4 BF16s (=8B), so use uint2;
                        // a uint4 load would be misaligned for odd lanes.
                        auto smem_ptr = smem_cd_l2
                            + epilogue_wg_idx * WG_BLOCK_M * BLOCK_N
                            + row_in_wg * BLOCK_N
                            + lane_in_row * kColsPerScatterLane;
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        if constexpr (kColsPerScatterLane == 8) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        } else {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint2>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint2));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                }

                if constexpr (not kSkipL2EpilogueSync)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        });

        // ---------------- COMBINE ----------------
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr uint32_t kNumChunks =
            (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE
             and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 : 2;
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
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
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
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_TRAP_ONLY_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
