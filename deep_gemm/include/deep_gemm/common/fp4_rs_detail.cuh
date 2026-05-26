#pragma once

// FP4 RS-mode helpers extracted from PR332 (deepgemm_pr332_fp4_sm90.patch).
//
// This header collects the device-side primitives required by the SM90
// register-source (RS) WGMMA path that consumes packed FP4 (E2M1) weights:
//
//   1. `final_accum_*` helpers       — scaled FP32 / BF16 accumulator promotion
//   2. `keep_float_live`             — barrier intrinsic to defeat scheduler
//      reordering between scale-multiply and FMA promotion
//   3. `FP8MMAF16AccumM64N8K32RS`    — MMA_64x8x32_F16E4M3E4M3_RS variant (used
//      by the F16-accum fast path)
//   4. `unpack_f16_accum_m64n8`      — half2 → float2 unpacker
//   5. `scale_float_by_pow2`         — IEEE exponent-shift fast multiplier
//   6. `SM90_U32x4_LDSM_T` /
//      `SM90_U32x2_STSM_T`           — transposed ldmatrix / stmatrix needed
//      to feed RS-mode A operand layout
//   7. `fp4_to_e4m3_byte` etc.       — scalar / vectorised E2M1 → E4M3 decoders
//   8. `pow2_scale_to_exp_shift`,
//      `make_scaled_e4m3_lut`,
//      `fp4x4_to_scaled_e4m3x4_*`    — UE8M0 SF "scaled-fused decode" path
//      (encodes the per-32 SFB into the dequant LUT so the mainloop avoids
//      a separate FP32 promote multiply).
//
// Naming convention reminder for PR332 RS-mode kernels:
//   In RS-mode WGMMA the *A* operand is in registers; the FP4 path uses A for
//   the **weight** (FP4, dequantised into a_regs) and B for the **activation**
//   (FP8, kept in shared memory). This is the opposite of conventional GEMM
//   naming and matters when reading scale_a / scale_b in the SS reference.
//
// Source: deepgemm_pr332_fp4_sm90.patch L4661-L5226.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cute/config.hpp>

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {
namespace fp4_rs_detail {

template <bool kBF16FinalAccum>
struct FinalAccumStorage {
    using type = float;
};

template <>
struct FinalAccumStorage<true> {
    using type = nv_bfloat162;
};

template <bool kBF16FinalAccum, typename dtype_t>
CUTLASS_DEVICE void final_accum_init(dtype_t* base, uint32_t idx) {
    if constexpr (kBF16FinalAccum) {
        base[idx] = __float22bfloat162_rn({0.0f, 0.0f});
    } else {
        base[idx] = 0.0f;
    }
}

template <bool kBF16FinalAccum, bool kFmaPromote, bool kBF16PromoteMath, typename dtype_t>
CUTLASS_DEVICE void final_accum_promote_pair(dtype_t* base, uint32_t pair_idx,
                                             float scale_0, float value_0,
                                             float scale_1, float value_1) {
    if constexpr (kBF16PromoteMath) {
        const nv_bfloat162 scale = __float22bfloat162_rn({scale_0, scale_1});
        const nv_bfloat162 value = __float22bfloat162_rn({value_0, value_1});
        nv_bfloat162 dst;
        if constexpr (kBF16FinalAccum) {
            dst = base[pair_idx];
        } else {
            const uint32_t idx = pair_idx * 2;
            dst = __float22bfloat162_rn({base[idx + 0], base[idx + 1]});
        }
        const nv_bfloat162 out = __hfma2(scale, value, dst);
        if constexpr (kBF16FinalAccum) {
            base[pair_idx] = out;
        } else {
            const uint32_t idx = pair_idx * 2;
            const float2 out_f = __bfloat1622float2(out);
            base[idx + 0] = out_f.x;
            base[idx + 1] = out_f.y;
        }
    } else if constexpr (kBF16FinalAccum) {
        float2 dst = __bfloat1622float2(base[pair_idx]);
        if constexpr (kFmaPromote) {
            dst.x = __fmaf_rn(scale_0, value_0, dst.x);
            dst.y = __fmaf_rn(scale_1, value_1, dst.y);
        } else {
            dst.x += scale_0 * value_0;
            dst.y += scale_1 * value_1;
        }
        base[pair_idx] = __float22bfloat162_rn(dst);
    } else {
        const uint32_t idx = pair_idx * 2;
        if constexpr (kFmaPromote) {
            base[idx + 0] = __fmaf_rn(scale_0, value_0, base[idx + 0]);
            base[idx + 1] = __fmaf_rn(scale_1, value_1, base[idx + 1]);
        } else {
            base[idx + 0] += scale_0 * value_0;
            base[idx + 1] += scale_1 * value_1;
        }
    }
}

template <bool kBF16FinalAccum, bool kFmaPromote, bool kBF16PromoteMath, typename dtype_t>
CUTLASS_DEVICE void final_accum_promote_pair2(dtype_t* base, uint32_t pair_idx,
                                              float scale_0_a, float value_0_a,
                                              float scale_1_a, float value_1_a,
                                              float scale_0_b, float value_0_b,
                                              float scale_1_b, float value_1_b) {
    if constexpr (kBF16PromoteMath) {
        const nv_bfloat162 scale_a = __float22bfloat162_rn({scale_0_a, scale_1_a});
        const nv_bfloat162 value_a = __float22bfloat162_rn({value_0_a, value_1_a});
        const nv_bfloat162 scale_b = __float22bfloat162_rn({scale_0_b, scale_1_b});
        const nv_bfloat162 value_b = __float22bfloat162_rn({value_0_b, value_1_b});
        nv_bfloat162 dst;
        if constexpr (kBF16FinalAccum) {
            dst = base[pair_idx];
        } else {
            const uint32_t idx = pair_idx * 2;
            dst = __float22bfloat162_rn({base[idx + 0], base[idx + 1]});
        }
        dst = __hfma2(scale_a, value_a, dst);
        dst = __hfma2(scale_b, value_b, dst);
        if constexpr (kBF16FinalAccum) {
            base[pair_idx] = dst;
        } else {
            const uint32_t idx = pair_idx * 2;
            const float2 out_f = __bfloat1622float2(dst);
            base[idx + 0] = out_f.x;
            base[idx + 1] = out_f.y;
        }
    } else if constexpr (kBF16FinalAccum) {
        float2 dst = __bfloat1622float2(base[pair_idx]);
        if constexpr (kFmaPromote) {
            dst.x = __fmaf_rn(scale_0_a, value_0_a, dst.x);
            dst.y = __fmaf_rn(scale_1_a, value_1_a, dst.y);
            dst.x = __fmaf_rn(scale_0_b, value_0_b, dst.x);
            dst.y = __fmaf_rn(scale_1_b, value_1_b, dst.y);
        } else {
            dst.x += scale_0_a * value_0_a;
            dst.y += scale_1_a * value_1_a;
            dst.x += scale_0_b * value_0_b;
            dst.y += scale_1_b * value_1_b;
        }
        base[pair_idx] = __float22bfloat162_rn(dst);
    } else {
        const uint32_t idx = pair_idx * 2;
        float dst_0 = base[idx + 0];
        float dst_1 = base[idx + 1];
        if constexpr (kFmaPromote) {
            dst_0 = __fmaf_rn(scale_0_a, value_0_a, dst_0);
            dst_1 = __fmaf_rn(scale_1_a, value_1_a, dst_1);
            dst_0 = __fmaf_rn(scale_0_b, value_0_b, dst_0);
            dst_1 = __fmaf_rn(scale_1_b, value_1_b, dst_1);
        } else {
            dst_0 += scale_0_a * value_0_a;
            dst_1 += scale_1_a * value_1_a;
            dst_0 += scale_0_b * value_0_b;
            dst_1 += scale_1_b * value_1_b;
        }
        base[idx + 0] = dst_0;
        base[idx + 1] = dst_1;
    }
}

template <bool kBF16FinalAccum, bool kFmaPromote, bool kBF16PromoteMath, typename dtype_t>
CUTLASS_DEVICE void final_accum_promote_pair4(dtype_t* base, uint32_t pair_idx,
                                              float scale_0_a, float value_0_a,
                                              float scale_1_a, float value_1_a,
                                              float scale_0_b, float value_0_b,
                                              float scale_1_b, float value_1_b,
                                              float scale_0_c, float value_0_c,
                                              float scale_1_c, float value_1_c,
                                              float scale_0_d, float value_0_d,
                                              float scale_1_d, float value_1_d) {
    if constexpr (kBF16PromoteMath) {
        nv_bfloat162 dst;
        if constexpr (kBF16FinalAccum) {
            dst = base[pair_idx];
        } else {
            const uint32_t idx = pair_idx * 2;
            dst = __float22bfloat162_rn({base[idx + 0], base[idx + 1]});
        }
        dst = __hfma2(__float22bfloat162_rn({scale_0_a, scale_1_a}),
                      __float22bfloat162_rn({value_0_a, value_1_a}), dst);
        dst = __hfma2(__float22bfloat162_rn({scale_0_b, scale_1_b}),
                      __float22bfloat162_rn({value_0_b, value_1_b}), dst);
        dst = __hfma2(__float22bfloat162_rn({scale_0_c, scale_1_c}),
                      __float22bfloat162_rn({value_0_c, value_1_c}), dst);
        dst = __hfma2(__float22bfloat162_rn({scale_0_d, scale_1_d}),
                      __float22bfloat162_rn({value_0_d, value_1_d}), dst);
        if constexpr (kBF16FinalAccum) {
            base[pair_idx] = dst;
        } else {
            const uint32_t idx = pair_idx * 2;
            const float2 out_f = __bfloat1622float2(dst);
            base[idx + 0] = out_f.x;
            base[idx + 1] = out_f.y;
        }
    } else if constexpr (kBF16FinalAccum) {
        float2 dst = __bfloat1622float2(base[pair_idx]);
        if constexpr (kFmaPromote) {
            dst.x = __fmaf_rn(scale_0_a, value_0_a, dst.x);
            dst.y = __fmaf_rn(scale_1_a, value_1_a, dst.y);
            dst.x = __fmaf_rn(scale_0_b, value_0_b, dst.x);
            dst.y = __fmaf_rn(scale_1_b, value_1_b, dst.y);
            dst.x = __fmaf_rn(scale_0_c, value_0_c, dst.x);
            dst.y = __fmaf_rn(scale_1_c, value_1_c, dst.y);
            dst.x = __fmaf_rn(scale_0_d, value_0_d, dst.x);
            dst.y = __fmaf_rn(scale_1_d, value_1_d, dst.y);
        } else {
            dst.x += scale_0_a * value_0_a;
            dst.y += scale_1_a * value_1_a;
            dst.x += scale_0_b * value_0_b;
            dst.y += scale_1_b * value_1_b;
            dst.x += scale_0_c * value_0_c;
            dst.y += scale_1_c * value_1_c;
            dst.x += scale_0_d * value_0_d;
            dst.y += scale_1_d * value_1_d;
        }
        base[pair_idx] = __float22bfloat162_rn(dst);
    } else {
        const uint32_t idx = pair_idx * 2;
        float dst_0 = base[idx + 0];
        float dst_1 = base[idx + 1];
        if constexpr (kFmaPromote) {
            dst_0 = __fmaf_rn(scale_0_a, value_0_a, dst_0);
            dst_1 = __fmaf_rn(scale_1_a, value_1_a, dst_1);
            dst_0 = __fmaf_rn(scale_0_b, value_0_b, dst_0);
            dst_1 = __fmaf_rn(scale_1_b, value_1_b, dst_1);
            dst_0 = __fmaf_rn(scale_0_c, value_0_c, dst_0);
            dst_1 = __fmaf_rn(scale_1_c, value_1_c, dst_1);
            dst_0 = __fmaf_rn(scale_0_d, value_0_d, dst_0);
            dst_1 = __fmaf_rn(scale_1_d, value_1_d, dst_1);
        } else {
            dst_0 += scale_0_a * value_0_a;
            dst_1 += scale_1_a * value_1_a;
            dst_0 += scale_0_b * value_0_b;
            dst_1 += scale_1_b * value_1_b;
            dst_0 += scale_0_c * value_0_c;
            dst_1 += scale_1_c * value_1_c;
            dst_0 += scale_0_d * value_0_d;
            dst_1 += scale_1_d * value_1_d;
        }
        base[idx + 0] = dst_0;
        base[idx + 1] = dst_1;
    }
}

template <bool kBF16FinalAccum, bool kFmaPromote, bool kBF16PromoteMath, typename dtype_t>
CUTLASS_DEVICE void final_accum_promote_pair4x2(dtype_t* base, uint32_t pair_idx,
                                                float scale_0_a, float value_0_a,
                                                float scale_1_a, float value_1_a,
                                                float scale_0_b, float value_0_b,
                                                float scale_1_b, float value_1_b,
                                                float scale_0_c, float value_0_c,
                                                float scale_1_c, float value_1_c,
                                                float scale_0_d, float value_0_d,
                                                float scale_1_d, float value_1_d,
                                                float scale_2_a, float value_2_a,
                                                float scale_3_a, float value_3_a,
                                                float scale_2_b, float value_2_b,
                                                float scale_3_b, float value_3_b,
                                                float scale_2_c, float value_2_c,
                                                float scale_3_c, float value_3_c,
                                                float scale_2_d, float value_2_d,
                                                float scale_3_d, float value_3_d) {
    final_accum_promote_pair4<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
        base, pair_idx,
        scale_0_a, value_0_a, scale_1_a, value_1_a,
        scale_0_b, value_0_b, scale_1_b, value_1_b,
        scale_0_c, value_0_c, scale_1_c, value_1_c,
        scale_0_d, value_0_d, scale_1_d, value_1_d);
    final_accum_promote_pair4<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
        base, pair_idx + 1,
        scale_2_a, value_2_a, scale_3_a, value_3_a,
        scale_2_b, value_2_b, scale_3_b, value_3_b,
        scale_2_c, value_2_c, scale_3_c, value_3_c,
        scale_2_d, value_2_d, scale_3_d, value_3_d);
}

CUTLASS_DEVICE void keep_float_live(float value) {
    asm volatile("" :: "f"(value) : "memory");
}

// MMA_64x8x32_F16E4M3E4M3_RS variant — F16 accumulator fast path. Used by the
// kBF16FinalAccum + N==8 micro-kernel that promotes a half2 lane back into
// float for the BF16 epilogue.
struct FP8MMAF16AccumM64N8K32RS {
    static constexpr uint32_t M = 64;
    static constexpr uint32_t N = 8;
    static constexpr uint32_t K = 32;
    static constexpr uint32_t kNumAccum = 2;

    template <typename GmmaDescriptor>
    __forceinline__ __device__ static void wgmma(uint32_t const* a, GmmaDescriptor const& desc,
                                                 uint32_t* d, bool scale_d) {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %7, 0;\n"
            "  wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e4m3 "
            "  {%0, %1}, {%2, %3, %4, %5}, %6, p, 1, 1;\n"
            "}\n"
            : "+r"(d[0]), "+r"(d[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "l"(desc.desc_), "r"(static_cast<uint32_t>(scale_d)));
    }
};

CUTLASS_DEVICE void unpack_f16_accum_m64n8(uint32_t const* src, float* dst) {
    #pragma unroll
    for (uint32_t i = 0; i < FP8MMAF16AccumM64N8K32RS::kNumAccum; ++ i) {
        const half2 value_h = *reinterpret_cast<const half2*>(src + i);
        const float2 value_f = __half22float2(value_h);
        dst[i * 2 + 0] = value_f.x;
        dst[i * 2 + 1] = value_f.y;
    }
}

// Multiply `value` by `pow2_scale` assuming the latter is a power of two.
// Implementation: shift the IEEE-754 exponent field by (exp(scale) - 127).
// This is single-instruction on the SASS side and avoids a full FP32 multiply
// in the mainloop's UE8M0 promote path.
CUTLASS_DEVICE float scale_float_by_pow2(float value, float pow2_scale) {
    uint32_t value_bits = *reinterpret_cast<uint32_t*>(&value);
    const uint32_t scale_bits = *reinterpret_cast<uint32_t*>(&pow2_scale);
    const int32_t exp_shift = static_cast<int32_t>((scale_bits >> 23) & 0xffu) - 127;
    value_bits = static_cast<uint32_t>(static_cast<int32_t>(value_bits) + (exp_shift << 23));
    return *reinterpret_cast<float*>(&value_bits);
}

template <bool kBF16FinalAccum, bool kFmaPromote, bool kBF16PromoteMath, bool kScaleBPow2Promote,
          typename dtype_t>
CUTLASS_DEVICE void final_accum_promote_pair4_split_scale(dtype_t* base, uint32_t pair_idx,
                                                          float scale_a_0, float scale_a_1,
                                                          float scale_b_a, float scale_b_b,
                                                          float scale_b_c, float scale_b_d,
                                                          float value_0_a, float value_1_a,
                                                          float value_0_b, float value_1_b,
                                                          float value_0_c, float value_1_c,
                                                          float value_0_d, float value_1_d) {
    auto prod_0 = [&](float scale_b) {
        return kScaleBPow2Promote ? scale_float_by_pow2(scale_a_0, scale_b) : scale_a_0 * scale_b;
    };
    auto prod_1 = [&](float scale_b) {
        return kScaleBPow2Promote ? scale_float_by_pow2(scale_a_1, scale_b) : scale_a_1 * scale_b;
    };
    if constexpr (kBF16PromoteMath) {
        nv_bfloat162 dst;
        if constexpr (kBF16FinalAccum) {
            dst = base[pair_idx];
        } else {
            const uint32_t idx = pair_idx * 2;
            dst = __float22bfloat162_rn({base[idx + 0], base[idx + 1]});
        }
        dst = __hfma2(__float22bfloat162_rn({prod_0(scale_b_a), prod_1(scale_b_a)}),
                      __float22bfloat162_rn({value_0_a, value_1_a}), dst);
        dst = __hfma2(__float22bfloat162_rn({prod_0(scale_b_b), prod_1(scale_b_b)}),
                      __float22bfloat162_rn({value_0_b, value_1_b}), dst);
        dst = __hfma2(__float22bfloat162_rn({prod_0(scale_b_c), prod_1(scale_b_c)}),
                      __float22bfloat162_rn({value_0_c, value_1_c}), dst);
        dst = __hfma2(__float22bfloat162_rn({prod_0(scale_b_d), prod_1(scale_b_d)}),
                      __float22bfloat162_rn({value_0_d, value_1_d}), dst);
        if constexpr (kBF16FinalAccum) {
            base[pair_idx] = dst;
        } else {
            const uint32_t idx = pair_idx * 2;
            const float2 out_f = __bfloat1622float2(dst);
            base[idx + 0] = out_f.x;
            base[idx + 1] = out_f.y;
        }
    } else if constexpr (kBF16FinalAccum) {
        float2 dst = __bfloat1622float2(base[pair_idx]);
        if constexpr (kFmaPromote) {
            dst.x = __fmaf_rn(prod_0(scale_b_a), value_0_a, dst.x);
            dst.y = __fmaf_rn(prod_1(scale_b_a), value_1_a, dst.y);
            dst.x = __fmaf_rn(prod_0(scale_b_b), value_0_b, dst.x);
            dst.y = __fmaf_rn(prod_1(scale_b_b), value_1_b, dst.y);
            dst.x = __fmaf_rn(prod_0(scale_b_c), value_0_c, dst.x);
            dst.y = __fmaf_rn(prod_1(scale_b_c), value_1_c, dst.y);
            dst.x = __fmaf_rn(prod_0(scale_b_d), value_0_d, dst.x);
            dst.y = __fmaf_rn(prod_1(scale_b_d), value_1_d, dst.y);
        } else {
            dst.x += prod_0(scale_b_a) * value_0_a;
            dst.y += prod_1(scale_b_a) * value_1_a;
            dst.x += prod_0(scale_b_b) * value_0_b;
            dst.y += prod_1(scale_b_b) * value_1_b;
            dst.x += prod_0(scale_b_c) * value_0_c;
            dst.y += prod_1(scale_b_c) * value_1_c;
            dst.x += prod_0(scale_b_d) * value_0_d;
            dst.y += prod_1(scale_b_d) * value_1_d;
        }
        base[pair_idx] = __float22bfloat162_rn(dst);
    } else {
        const uint32_t idx = pair_idx * 2;
        float dst_0 = base[idx + 0];
        float dst_1 = base[idx + 1];
        if constexpr (kFmaPromote) {
            dst_0 = __fmaf_rn(prod_0(scale_b_a), value_0_a, dst_0);
            dst_1 = __fmaf_rn(prod_1(scale_b_a), value_1_a, dst_1);
            dst_0 = __fmaf_rn(prod_0(scale_b_b), value_0_b, dst_0);
            dst_1 = __fmaf_rn(prod_1(scale_b_b), value_1_b, dst_1);
            dst_0 = __fmaf_rn(prod_0(scale_b_c), value_0_c, dst_0);
            dst_1 = __fmaf_rn(prod_1(scale_b_c), value_1_c, dst_1);
            dst_0 = __fmaf_rn(prod_0(scale_b_d), value_0_d, dst_0);
            dst_1 = __fmaf_rn(prod_1(scale_b_d), value_1_d, dst_1);
        } else {
            dst_0 += prod_0(scale_b_a) * value_0_a;
            dst_1 += prod_1(scale_b_a) * value_1_a;
            dst_0 += prod_0(scale_b_b) * value_0_b;
            dst_1 += prod_1(scale_b_b) * value_1_b;
            dst_0 += prod_0(scale_b_c) * value_0_c;
            dst_1 += prod_1(scale_b_c) * value_1_c;
            dst_0 += prod_0(scale_b_d) * value_0_d;
            dst_1 += prod_1(scale_b_d) * value_1_d;
        }
        base[idx + 0] = dst_0;
        base[idx + 1] = dst_1;
    }
}

template <bool kBF16FinalAccum, typename dtype_t>
CUTLASS_DEVICE float final_accum_load_scalar(const dtype_t* base, uint32_t idx) {
    if constexpr (kBF16FinalAccum) {
        const float2 value = __bfloat1622float2(base[idx / 2]);
        return (idx % 2 == 0) ? value.x : value.y;
    } else {
        return base[idx];
    }
}

template <bool kBF16FinalAccum, typename dtype_t>
CUTLASS_DEVICE nv_bfloat162 final_accum_load_pair_bf16(const dtype_t* base, uint32_t pair_idx) {
    if constexpr (kBF16FinalAccum) {
        return base[pair_idx];
    } else {
        return __float22bfloat162_rn({base[pair_idx * 2 + 0], base[pair_idx * 2 + 1]});
    }
}

// `ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16`: load 4 8x8 b16 matrices
// from swizzled smem into 4 b32 registers per lane, transposed. The transpose
// option is the key to feeding RS-mode WGMMA's A operand layout when the
// source tile is laid out K-major in smem (which is the only layout the
// dequant step can produce cheaply from packed FP4 input).
//
// Note: WGMMA RS-mode A still consumes 8-bit (E4M3) pairs, so we phrase the
// granularity as b16 ldmatrix and let the upper layer treat each b32 register
// as 4 packed E4M3 lanes.
struct SM90_U32x4_LDSM_T {
    CUTLASS_DEVICE static void
    copy(uint32_t& dst_0, uint32_t& dst_1, uint32_t& dst_2, uint32_t& dst_3, void* smem_src) {
        asm volatile(
            "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(dst_0), "=r"(dst_1), "=r"(dst_2), "=r"(dst_3)
            : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

template <uint32_t COL_BYTES, uint32_t NV>
CUTLASS_DEVICE uint32_t permute_col(const uint32_t row, const uint32_t col) {
    constexpr uint32_t strd = 128 / (COL_BYTES < 128u ? COL_BYTES : 128u);
    return ((col / NV) ^ (row % 8 / strd)) * NV;
}

template <typename dtype_t>
struct SM90_U32x2_STSM_T {
    CUTLASS_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16.trans [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

// Decode a single FP4 (E2M1) code (low 4 bits) to its E4M3 byte
// representation. Same byte-LUT formulation as the scalar path inside the SS
// kernel, hoisted to namespace scope so it can also be called from
// register-level dequant routines.
//   mag {0..7} -> {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c}
//   sign bit copied from FP4 bit 3 to E4M3 bit 7 (FP4 -0 maps to E4M3 -0;
//   WGMMA treats -0 as 0, so this is numerically safe and saves a branch).
CUTLASS_DEVICE uint32_t fp4_to_e4m3_byte(uint32_t code) {
    constexpr uint32_t LUT_LO = 0x3c383000u;
    constexpr uint32_t LUT_HI = 0x4c484440u;
    const uint32_t mag      = code & 0x07u;
    const uint32_t mag_byte = __byte_perm(LUT_LO, LUT_HI, mag) & 0xffu;
    const uint32_t sign     = (code & 0x08u) << 4;  // 0 or 0x80
    return mag_byte | sign;
}

// Decode 8 packed FP4 codes (one 32-bit word holding 8 nibbles) into 8 E4M3
// bytes packed into a single 64-bit value (low byte = code 0).
//
// This is the register-resident analogue of the SS variant's
// `fp4_pair_to_e4m3_pair` chained over 4 bytes. It is the building block the
// second-round kernel will use after `ldmatrix.x4.trans` to materialise A
// operand registers for RS-mode WGMMA without round-tripping through smem.
CUTLASS_DEVICE uint64_t fp4x8_to_e4m3x8(uint32_t packed) {
    uint64_t out = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t nib = (packed >> (i * 4)) & 0x0fu;
        out |= static_cast<uint64_t>(fp4_to_e4m3_byte(nib)) << (i * 8);
    }
    return out;
}

CUTLASS_DEVICE void fast_fp4_to_e4m3_convert(uint32_t outputs[2], uint32_t input) {
    const uint64_t decoded = fp4x8_to_e4m3x8(input);
    outputs[0] = static_cast<uint32_t>(decoded);
    outputs[1] = static_cast<uint32_t>(decoded >> 32);
}

// Vectorised FP4 → E4M3 decode for 4 nibbles at a time. Issues 2 × `prmt` and
// 1 × `lop3.b32 0xf8` (which evaluates `(a & ~c) | (b & c)` here applying
// 0x80808080 sign mask from `sign_bytes` onto `mag_bytes`). Throughput on H100
// is approximately 1.33 nibble/cycle/lane.
CUTLASS_DEVICE uint32_t fp4x4_to_e4m3x4(uint32_t packed) {
    constexpr uint32_t pos0 = 0x3c383000u;
    constexpr uint32_t pos1 = 0x4c484440u;
    const uint32_t lut_idx = packed & 0x7777u;
    const uint32_t sign_shifted = packed << 4;
    uint32_t mag_bytes, sign_bytes;
    asm volatile(
        "{\n"
        "  prmt .b32 %0, %3, %4, %2;\n"
        "  prmt .b32 %1, %5, %6, 0xd9c8;\n"
        "}\n"
        : "=r"(mag_bytes), "=r"(sign_bytes)
        : "r"(lut_idx), "r"(pos0), "r"(pos1), "r"(sign_shifted), "r"(packed));
    uint32_t out;
    asm volatile(
        "{\n"
        "  lop3.b32 %0, %1, %2, 0x80808080, 0xf8;\n"
        "}\n"
        : "=r"(out)
        : "r"(mag_bytes), "r"(sign_bytes));
    return out;
}

// Returns `floor(log2(scale))` for a positive power-of-two `scale`, derived
// from the IEEE-754 exponent field directly (no `__log2f`).
CUTLASS_DEVICE int32_t pow2_scale_to_exp_shift(float scale) {
    const uint32_t scale_bits = *reinterpret_cast<uint32_t*>(&scale);
    return static_cast<int32_t>((scale_bits >> 23) & 0xffu) - 127;
}

// Dual byte LUT used by `fp4x4_to_scaled_e4m3x4_lut`. `lo` covers magnitudes
// 0..3, `hi` covers magnitudes 4..7. The mantissa LUT for unscaled FP4 is
// `(0x3c383000u, 0x4c484440u)`; calling `make_scaled_e4m3_lut(exp_offset)`
// produces a LUT pre-baked with a per-32 SF such that the dequant output is
// `value * 2^(exp_offset - 6)`.
struct ScaledE4M3Lut {
    uint32_t lo;
    uint32_t hi;
};

// Runtime FP4 MegaMoE decodes one (N row, K/32 group) at a time, so building
// the same scaled E4M3 LUT from UE8M0 in every decode group sits directly on
// the hot path. Keep the 256-entry mapping in constant memory and make the
// decode loop pay one cached load instead of the branch/IMAD chain below.
static __device__ __constant__ __align__(8) uint64_t kScaledE4M3LutFromE8M0[256] = {
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0100000000000000ull,
    0x0201010000000000ull, 0x0302020101000000ull, 0x0604030202010000ull, 0x0c08060403020100ull,
    0x14100c0806040200ull, 0x1c1814100c080400ull, 0x24201c1814100800ull, 0x2c2824201c181000ull,
    0x34302c2824201800ull, 0x3c3834302c282000ull, 0x44403c3834302800ull, 0x4c4844403c383000ull,
    0x54504c4844403800ull, 0x5c5854504c484000ull, 0x64605c5854504800ull, 0x6c6864605c585000ull,
    0x74706c6864605800ull, 0x7c7874706c686000ull, 0x84807c7874706800ull, 0x8c8884807c787000ull,
    0x94908c8884807800ull, 0x9c9894908c888000ull, 0xa4a09c9894908800ull, 0xaca8a4a09c989000ull,
    0xb4b0aca8a4a09800ull, 0xbcb8b4b0aca8a000ull, 0xc4c0bcb8b4b0a800ull, 0xccc8c4c0bcb8b000ull,
    0xd4d0ccc8c4c0b800ull, 0xdcd8d4d0ccc8c000ull, 0xe4e0dcd8d4d0c800ull, 0xece8e4e0dcd8d000ull,
    0xf4f0ece8e4e0d800ull, 0xfcf8f4f0ece8e000ull, 0x0500fcf8f4f0e800ull, 0x0d090500fcf8f000ull,
    0x15110d080500f800ull, 0x1d1915100d090000ull, 0x25211d1815110800ull, 0x2d2925201d191000ull,
    0x35312d2825211800ull, 0x3d3935302d292000ull, 0x45413d3835312800ull, 0x4d4945403d393000ull,
    0x55514d4845413800ull, 0x5d5955504d494000ull, 0x65615d5855514800ull, 0x6d6965605d595000ull,
    0x75716d6865615800ull, 0x7d7975706d696000ull, 0x85817d7875716800ull, 0x8d8985807d797000ull,
    0x95918d8885817800ull, 0x9d9995908d898000ull, 0xa5a19d9895918800ull, 0xada9a5a09d999000ull,
    0xb5b1ada8a5a19800ull, 0xbdb9b5b0ada9a000ull, 0xc5c1bdb8b5b1a800ull, 0xcdc9c5c0bdb9b000ull,
    0xd5d1cdc8c5c1b800ull, 0xddd9d5d0cdc9c000ull, 0xe5e1ddd8d5d1c800ull, 0xede9e5e0ddd9d000ull,
    0xf5f1ede8e5e1d800ull, 0xfdf9f5f0ede9e000ull, 0x0601fdf8f5f1e800ull, 0x0e0a0600fdf9f000ull,
    0x16120e080601f800ull, 0x1e1a16100e0a0000ull, 0x26221e1816120800ull, 0x2e2a26201e1a1000ull,
    0x36322e2826221800ull, 0x3e3a36302e2a2000ull, 0x46423e3836322800ull, 0x4e4a46403e3a3000ull,
    0x56524e4846423800ull, 0x5e5a56504e4a4000ull, 0x66625e5856524800ull, 0x6e6a66605e5a5000ull,
    0x76726e6866625800ull, 0x7e7a76706e6a6000ull, 0x86827e7876726800ull, 0x8e8a86807e7a7000ull,
    0x96928e8886827800ull, 0x9e9a96908e8a8000ull, 0xa6a29e9896928800ull, 0xaeaaa6a09e9a9000ull,
    0xb6b2aea8a6a29800ull, 0xbebab6b0aeaaa000ull, 0xc6c2beb8b6b2a800ull, 0xcecac6c0bebab000ull,
    0xd6d2cec8c6c2b800ull, 0xdedad6d0cecac000ull, 0xe6e2ded8d6d2c800ull, 0xeeeae6e0dedad000ull,
    0xf6f2eee8e6e2d800ull, 0xfefaf6f0eeeae000ull, 0x0702fef8f6f2e800ull, 0x0f0b0700fefaf000ull,
    0x17130f080702f800ull, 0x1f1b17100f0b0000ull, 0x27231f1817130800ull, 0x2f2b27201f1b1000ull,
    0x37332f2827231800ull, 0x3f3b37302f2b2000ull, 0x47433f3837332800ull, 0x4f4b47403f3b3000ull,
    0x57534f4847433800ull, 0x5f5b57504f4b4000ull, 0x67635f5857534800ull, 0x6f6b67605f5b5000ull,
    0x77736f6867635800ull, 0x7f7b77706f6b6000ull, 0x87837f7877736800ull, 0x8f8b87807f7b7000ull,
    0x97938f8887837800ull, 0x9f9b97908f8b8000ull, 0xa7a39f9897938800ull, 0xafaba7a09f9b9000ull,
    0xb7b3afa8a7a39800ull, 0xbfbbb7b0afaba000ull, 0xc7c3bfb8b7b3a800ull, 0xcfcbc7c0bfbbb000ull,
    0xd7d3cfc8c7c3b800ull, 0xdfdbd7d0cfcbc000ull, 0xe7e3dfd8d7d3c800ull, 0xefebe7e0dfdbd000ull,
    0xf7f3efe8e7e3d800ull, 0xfffbf7f0efebe000ull, 0x0803fff8f7f3e800ull, 0x100c0800fffbf000ull,
    0x181410080803f800ull, 0x201c1810100c0000ull, 0x2824201818140800ull, 0x302c2820201c1000ull,
    0x3834302828241800ull, 0x403c3830302c2000ull, 0x4844403838342800ull, 0x504c4840403c3000ull
};

CUTLASS_DEVICE ScaledE4M3Lut make_scaled_e4m3_lut(uint32_t exp_offset) {
    // Random FP4 per-32 scales are overwhelmingly ceil-pow2(max(abs(x))/6)
    // in {2^-1, 1}, which maps to exp_offset {5, 6}. Fast-path those to
    // avoid the dynamic IMAD chain in the WGMMA issue loop.
    if (exp_offset == 5u)
        return {0x34302800u, 0x44403c38u};
    if (exp_offset == 6u)
        return {0x3c383000u, 0x4c484440u};
    const uint32_t exp_offset_buffer1 =
        exp_offset * 0x08080800u + (exp_offset ? 0xfffffc00u : 0u);
    const uint32_t exp_offset_buffer2 = exp_offset * 0x08080808u;
    constexpr uint32_t mantissa_lo = 0x0c080400u;
    constexpr uint32_t mantissa_hi = 0x1c181410u;
    return {mantissa_lo + exp_offset_buffer1, mantissa_hi + exp_offset_buffer2};
}

CUTLASS_DEVICE ScaledE4M3Lut make_scaled_e4m3_lut_from_e8m0(uint32_t e8m0) {
    // For DSV4 MXFP4 checkpoints the expert scales frequently land below
    // 2^-6 (UE8M0 <= 120). These values must be represented with E4M3
    // subnormals after FP4 decode. The generic positive-offset path below
    // intentionally starts at 2^-6, so use exact byte LUTs for the subnormal
    // range instead of saturating all small scales to 2^-6.
    if (e8m0 <= 114u)
        return {0x00000000u, 0x00000000u};
    if (e8m0 == 115u)
        return {0x00000000u, 0x01000000u};
    if (e8m0 == 116u)
        return {0x00000000u, 0x02010100u};
    if (e8m0 == 117u)
        return {0x01000000u, 0x03020201u};
    if (e8m0 == 118u)
        return {0x02010000u, 0x06040302u};
    if (e8m0 == 119u)
        return {0x03020100u, 0x0c080604u};
    if (e8m0 == 120u)
        return {0x06040200u, 0x14100c08u};
    return make_scaled_e4m3_lut(e8m0 - 121u);
}

CUTLASS_DEVICE ScaledE4M3Lut make_scaled_e4m3_lut_from_e8m0_fast(uint32_t e8m0) {
    if (e8m0 >= 121u)
        return make_scaled_e4m3_lut(e8m0 - 121u);
    return make_scaled_e4m3_lut_from_e8m0(e8m0);
}

CUTLASS_DEVICE uint64_t pack_scaled_e4m3_lut_from_e8m0_fast(uint32_t e8m0) {
    const auto lut = make_scaled_e4m3_lut_from_e8m0_fast(e8m0);
    return static_cast<uint64_t>(lut.lo) | (static_cast<uint64_t>(lut.hi) << 32);
}

CUTLASS_DEVICE uint64_t pack_scaled_e4m3_lut_from_e8m0_const(uint32_t e8m0) {
    return kScaledE4M3LutFromE8M0[e8m0 & 0xffu];
}

CUTLASS_DEVICE uint64_t pack_scaled_e4m3_lut_from_e8m0_common_const(uint32_t e8m0) {
    // DSV4-style random FP4 expert weights overwhelmingly quantize to a tiny
    // UE8M0 set around 2^-6..2^-4. Fast-path those values with immediates so
    // the decode hot path can avoid a divergent constant-cache LUT load.
    if (e8m0 == 122u)
        return 0x24201c1814100800ull;
    if (e8m0 == 121u)
        return 0x1c1814100c080400ull;
    if (e8m0 == 123u)
        return 0x2c2824201c181000ull;
    if (e8m0 == 120u)
        return 0x14100c0806040200ull;
    return pack_scaled_e4m3_lut_from_e8m0_const(e8m0);
}

CUTLASS_DEVICE uint32_t fp4x4_to_scaled_e4m3x4_lut(
        uint32_t packed, uint32_t lut_lo, uint32_t lut_hi) {
    const uint32_t lut_idx = packed & 0x7777u;
    const uint32_t sign_shifted = packed << 4;
    uint32_t mantissa_bytes, sign_bytes;
    asm volatile(
        "{\n"
        "  prmt .b32 %0, %3, %4, %2;\n"
        "  prmt .b32 %1, %5, %6, 0xd9c8;\n"
        "}\n"
        : "=r"(mantissa_bytes), "=r"(sign_bytes)
        : "r"(lut_idx), "r"(lut_lo), "r"(lut_hi), "r"(sign_shifted), "r"(packed));
    return (sign_bytes & 0x80808080u) | mantissa_bytes;
}

CUTLASS_DEVICE uint32_t fp4x4_to_scaled_e4m3x4_lut(uint32_t packed, ScaledE4M3Lut lut) {
    return fp4x4_to_scaled_e4m3x4_lut(packed, lut.lo, lut.hi);
}

CUTLASS_DEVICE uint32_t fp4x4_to_scaled_e4m3x4_offset(uint32_t packed, uint32_t exp_offset) {
    return fp4x4_to_scaled_e4m3x4_lut(packed, make_scaled_e4m3_lut(exp_offset));
}

CUTLASS_DEVICE uint32_t fp4x4_to_scaled_e4m3x4_e8m0(uint32_t packed, uint32_t e8m0) {
    return fp4x4_to_scaled_e4m3x4_lut(packed, make_scaled_e4m3_lut_from_e8m0(e8m0));
}

CUTLASS_DEVICE uint32_t fp4x4_to_scaled_e4m3x4_e8m0_fast(uint32_t packed, uint32_t e8m0) {
    if (e8m0 >= 121u)
        return fp4x4_to_scaled_e4m3x4_offset(packed, e8m0 - 121u);
    return fp4x4_to_scaled_e4m3x4_e8m0(packed, e8m0);
}

// Hummingbird variant: rather than building a LUT and using `prmt`, this
// computes the scaled E4M3 mantissa with one `lop3.b32 0xf8` and adjusts the
// exponent with a per-byte `__byte_perm`. Useful when each lane sees a
// different `exp_offset` and the LUT-build amortisation argument breaks down.
CUTLASS_DEVICE uint32_t fp4x4_to_scaled_e4m3x4_humming(uint32_t packed_nibbles, uint32_t exp_offset) {
    const uint32_t packed =
        (packed_nibbles & 0x000fu) |
        ((packed_nibbles & 0x00f0u) << 4) |
        ((packed_nibbles & 0x0f00u) << 8) |
        ((packed_nibbles & 0xf000u) << 12);

    const uint32_t exp_offset_buffer1 = exp_offset * 0x08080800u + (exp_offset ? 0xfffffc00u : 0u);
    const uint32_t exp_offset_buffer2 = exp_offset * 0x08080808u;
    const uint32_t exp_offsets0 = __byte_perm(exp_offset_buffer1, exp_offset_buffer2, packed);
    const uint32_t exp_offsets1 = __byte_perm(exp_offset_buffer1, exp_offset_buffer2, packed >> 16);

    const uint32_t mag_bytes = (packed & 0x07070707u) << 2;
    const uint32_t sign_bytes = (packed << 4) & 0x80808080u;
    const uint32_t scaled_mag =
        __vadd4(mag_bytes, __byte_perm(exp_offsets0, exp_offsets1, 0x6420));
    return sign_bytes | scaled_mag;
}

}  // namespace fp4_rs_detail

// Dispatch helper used by the FP4 RS kernel to specialise the mainloop on
// `num_former_iters` (number of K-iterations consumed by the head warpgroup).
template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
CUTLASS_DEVICE void dispatch_num_former_iters_rs(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters_rs<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

}  // namespace deep_gemm
