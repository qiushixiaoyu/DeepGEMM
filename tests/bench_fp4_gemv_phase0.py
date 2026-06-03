"""Phase-0 standalone FP4 GEMV microbench for SM90 MegaMoE small batches.

This intentionally does not call or modify the fused MegaMoE kernel.  It
measures a simple CUDA-core FP4 thin-GEMM prototype so we can decide whether a
future GEMV mainloop is worth integrating into the big kernel.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dtype) TORCH_CHECK(x.scalar_type() == dtype, #x " has wrong dtype")

constexpr int kThreads = 128;
constexpr int kBlockT = 8;

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t code) {
    const uint8_t mag = code & 0x7u;
    float v;
    switch (mag) {
        case 0: v = 0.0f; break;
        case 1: v = 0.5f; break;
        case 2: v = 1.0f; break;
        case 3: v = 1.5f; break;
        case 4: v = 2.0f; break;
        case 5: v = 3.0f; break;
        case 6: v = 4.0f; break;
        default: v = 6.0f; break;
    }
    return (code & 0x8u) ? -v : v;
}

__device__ __forceinline__ float ue8m0_to_scale(uint8_t exp) {
    return __uint_as_float(static_cast<uint32_t>(exp) << 23u);
}

__global__ void fp4_decode_only_kernel(const uint8_t* __restrict__ w_packed,
                                       const uint8_t* __restrict__ sf,
                                       float* __restrict__ scratch,
                                       unsigned long long* __restrict__ cycles,
                                       int t, int n, int k) {
    const int col = blockIdx.x;
    const int tb = blockIdx.y;
    const int tid = threadIdx.x;
    const int packed_k = k / 2;
    const int groups_k = k / 128;
    const int token_base = tb * kBlockT;
    const int valid_t = max(0, min(kBlockT, t - token_base));
    const unsigned long long start = clock64();

    float accum = 0.0f;
    for (int pk = tid; pk < packed_k; pk += blockDim.x) {
        const uint8_t packed = w_packed[col * packed_k + pk];
        const int k0 = pk * 2;
        const float scale = ue8m0_to_scale(sf[col * groups_k + k0 / 128]);
        const float w0 = fp4_e2m1_to_float(packed & 0x0fu) * scale;
        const float w1 = fp4_e2m1_to_float(packed >> 4) * scale;
        accum += w0 + w1;
    }

    __shared__ float smem[kThreads];
    smem[tid] = accum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int block_idx = tb * n + col;
        scratch[block_idx] = smem[0] * static_cast<float>(valid_t);
        cycles[block_idx] = clock64() - start;
    }
}

__global__ void fp4_gemv_kernel(const half* __restrict__ x,
                                const uint8_t* __restrict__ w_packed,
                                const uint8_t* __restrict__ sf,
                                float* __restrict__ out,
                                unsigned long long* __restrict__ cycles,
                                int t, int n, int k) {
    const int col = blockIdx.x;
    const int tb = blockIdx.y;
    const int tid = threadIdx.x;
    const int packed_k = k / 2;
    const int groups_k = k / 128;
    const int token_base = tb * kBlockT;
    const unsigned long long start = clock64();

    float accum[kBlockT];
    #pragma unroll
    for (int i = 0; i < kBlockT; ++i) {
        accum[i] = 0.0f;
    }

    for (int pk = tid; pk < packed_k; pk += blockDim.x) {
        const uint8_t packed = w_packed[col * packed_k + pk];
        const int k0 = pk * 2;
        const int k1 = k0 + 1;
        const float scale = ue8m0_to_scale(sf[col * groups_k + k0 / 128]);
        const float w0 = fp4_e2m1_to_float(packed & 0x0fu) * scale;
        const float w1 = fp4_e2m1_to_float(packed >> 4) * scale;

        #pragma unroll
        for (int tt = 0; tt < kBlockT; ++tt) {
            const int token = token_base + tt;
            if (token < t) {
                const half* row = x + token * k;
                accum[tt] = __fmaf_rn(__half2float(row[k0]), w0, accum[tt]);
                accum[tt] = __fmaf_rn(__half2float(row[k1]), w1, accum[tt]);
            }
        }
    }

    __shared__ float smem[kBlockT][kThreads];
    #pragma unroll
    for (int tt = 0; tt < kBlockT; ++tt) {
        smem[tt][tid] = accum[tt];
    }
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int tt = 0; tt < kBlockT; ++tt) {
                smem[tt][tid] += smem[tt][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int tt = 0; tt < kBlockT; ++tt) {
            const int token = token_base + tt;
            if (token < t) {
                out[token * n + col] = smem[tt][0];
            }
        }
        cycles[tb * n + col] = clock64() - start;
    }
}

void launch_decode_only(torch::Tensor w_packed,
                        torch::Tensor sf,
                        torch::Tensor scratch,
                        torch::Tensor cycles,
                        int t) {
    CHECK_CUDA(w_packed);
    CHECK_CUDA(sf);
    CHECK_CUDA(scratch);
    CHECK_CUDA(cycles);
    CHECK_CONTIGUOUS(w_packed);
    CHECK_CONTIGUOUS(sf);
    CHECK_CONTIGUOUS(scratch);
    CHECK_CONTIGUOUS(cycles);
    CHECK_DTYPE(w_packed, torch::kUInt8);
    CHECK_DTYPE(sf, torch::kUInt8);
    CHECK_DTYPE(scratch, torch::kFloat32);
    CHECK_DTYPE(cycles, torch::kInt64);
    const int n = static_cast<int>(w_packed.size(0));
    const int k = static_cast<int>(w_packed.size(1)) * 2;
    TORCH_CHECK(k % 128 == 0, "k must be divisible by 128");
    TORCH_CHECK(sf.size(0) == n && sf.size(1) == k / 128, "bad sf shape");
    const dim3 grid(n, (t + kBlockT - 1) / kBlockT);
    fp4_decode_only_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        w_packed.data_ptr<uint8_t>(),
        sf.data_ptr<uint8_t>(),
        scratch.data_ptr<float>(),
        reinterpret_cast<unsigned long long*>(cycles.data_ptr<int64_t>()),
        t, n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gemv(torch::Tensor x,
                 torch::Tensor w_packed,
                 torch::Tensor sf,
                 torch::Tensor out,
                 torch::Tensor cycles) {
    CHECK_CUDA(x);
    CHECK_CUDA(w_packed);
    CHECK_CUDA(sf);
    CHECK_CUDA(out);
    CHECK_CUDA(cycles);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(w_packed);
    CHECK_CONTIGUOUS(sf);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(cycles);
    CHECK_DTYPE(x, torch::kFloat16);
    CHECK_DTYPE(w_packed, torch::kUInt8);
    CHECK_DTYPE(sf, torch::kUInt8);
    CHECK_DTYPE(out, torch::kFloat32);
    CHECK_DTYPE(cycles, torch::kInt64);
    const int t = static_cast<int>(x.size(0));
    const int k = static_cast<int>(x.size(1));
    const int n = static_cast<int>(w_packed.size(0));
    TORCH_CHECK(w_packed.size(1) * 2 == k, "bad packed weight shape");
    TORCH_CHECK(k % 128 == 0, "k must be divisible by 128");
    TORCH_CHECK(sf.size(0) == n && sf.size(1) == k / 128, "bad sf shape");
    TORCH_CHECK(out.size(0) == t && out.size(1) == n, "bad out shape");
    const dim3 grid(n, (t + kBlockT - 1) / kBlockT);
    fp4_gemv_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        w_packed.data_ptr<uint8_t>(),
        sf.data_ptr<uint8_t>(),
        out.data_ptr<float>(),
        reinterpret_cast<unsigned long long*>(cycles.data_ptr<int64_t>()),
        t, n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_only", &launch_decode_only, "FP4 decode-only microbench");
    m.def("gemv", &launch_gemv, "FP4 GEMV microbench");
}
"""


def build_extension(verbose: bool):
    build_root = Path(os.environ.get("DG_FP4_GEMV_MICROBENCH_BUILD_DIR", "/tmp/dg_fp4_gemv_phase0_build"))
    build_root.mkdir(parents=True, exist_ok=True)
    src = build_root / "fp4_gemv_phase0_ext.cu"
    src.write_text(CUDA_SRC)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    return load(
        name="fp4_gemv_phase0_ext",
        sources=[str(src)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        extra_cflags=["-O3"],
        build_directory=str(build_root),
        verbose=verbose,
    )


def fp4_table(device):
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=device,
    )


def unpack_fp4(w_packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    n, packed_k = w_packed.shape
    k = packed_k * 2
    lo = w_packed & 0x0F
    hi = w_packed >> 4
    codes = torch.empty((n, k), dtype=torch.long, device=w_packed.device)
    codes[:, 0::2] = lo.long()
    codes[:, 1::2] = hi.long()
    values = fp4_table(w_packed.device)[codes]
    scales = torch.ldexp(
        torch.ones_like(sf, dtype=torch.float32),
        sf.to(torch.int32) - 127,
    )
    return values * scales.repeat_interleave(128, dim=1)


def make_inputs(t: int, n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed + t * 1000003 + n * 9176 + k)
    x = torch.randn((t, k), device="cuda", dtype=torch.float16, generator=gen)
    codes_lo = torch.randint(0, 16, (n, k // 2), device="cuda", dtype=torch.uint8, generator=gen)
    codes_hi = torch.randint(0, 16, (n, k // 2), device="cuda", dtype=torch.uint8, generator=gen)
    w_packed = (codes_lo | (codes_hi << 4)).contiguous()
    sf = torch.randint(123, 130, (n, k // 128), device="cuda", dtype=torch.uint8, generator=gen)
    return x, w_packed, sf


def bench(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return sorted(times)[len(times) // 2]


def cycle_stats(cycles: torch.Tensor):
    c = cycles.to(torch.float64).flatten().cpu()
    return {
        "cta_cycles_mean": round(float(c.mean().item()), 3),
        "cta_cycles_p50": round(float(c.median().item()), 3),
        "cta_cycles_p90": round(float(torch.quantile(c, 0.9).item()), 3),
        "cta_cycles_max": round(float(c.max().item()), 3),
    }


def run_correctness(ext):
    t, n, k = 3, 64, 128
    x, w_packed, sf = make_inputs(t, n, k, seed=17)
    out = torch.empty((t, n), device="cuda", dtype=torch.float32)
    cycles = torch.empty((n * ((t + 7) // 8),), device="cuda", dtype=torch.int64)
    ext.gemv(x, w_packed, sf, out, cycles)
    ref = x.float() @ unpack_fp4(w_packed, sf).t()
    torch.cuda.synchronize()
    max_diff = float((out - ref).abs().max().item())
    result = {"kind": "correctness", "t": t, "n": n, "k": k, "max_diff": max_diff, "passed": max_diff < 2e-2}
    print("MICRO_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    if not result["passed"]:
        raise RuntimeError(f"correctness failed: max_diff={max_diff}")


def run_case(ext, stage: str, t: int, n: int, k: int, warmup: int, repeat: int, seed: int):
    x, w_packed, sf = make_inputs(t, n, k, seed=seed)
    token_blocks = (t + 7) // 8
    out = torch.empty((t, n), device="cuda", dtype=torch.float32)
    scratch = torch.empty((token_blocks, n), device="cuda", dtype=torch.float32)
    cycles = torch.empty((token_blocks, n), device="cuda", dtype=torch.int64)

    def decode_fn():
        ext.decode_only(w_packed, sf, scratch, cycles, t)

    decode_us = bench(decode_fn, warmup, repeat)
    decode_stats = cycle_stats(cycles)
    decode_result = {
        "kind": "decode_only",
        "stage": stage,
        "t": t,
        "n": n,
        "k": k,
        "block_t": 8,
        "kernel_us_median": round(decode_us, 3),
        "num_ctas": int(token_blocks * n),
        **decode_stats,
    }
    print("MICRO_RESULT_JSON " + json.dumps(decode_result, sort_keys=True), flush=True)

    def gemv_fn():
        ext.gemv(x, w_packed, sf, out, cycles)

    gemv_us = bench(gemv_fn, warmup, repeat)
    gemv_stats = cycle_stats(cycles)
    gemv_result = {
        "kind": "gemv_decode_fma",
        "stage": stage,
        "t": t,
        "n": n,
        "k": k,
        "block_t": 8,
        "kernel_us_median": round(gemv_us, 3),
        "decode_only_us_median": round(decode_us, 3),
        "decode_only_fraction": round(decode_us / gemv_us, 4) if gemv_us else float("nan"),
        "num_ctas": int(token_blocks * n),
        **gemv_stats,
    }
    print("MICRO_RESULT_JSON " + json.dumps(gemv_result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--stages", nargs="+", default=["l1", "l2"], choices=["l1", "l2"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    print("MICRO_CONTEXT_JSON " + json.dumps({
        "device": props.name,
        "sm": f"{props.major}{props.minor}",
        "clock_rate_khz": getattr(props, "clock_rate", None),
        "tokens": args.tokens,
        "stages": args.stages,
        "note": "Standalone CUDA-core prototype; one CTA computes one output column for up to 8 tokens.",
    }, sort_keys=True), flush=True)

    ext = build_extension(verbose=args.verbose_build)
    run_correctness(ext)
    shapes = {
        "l1": {"n": 4096, "k": 4096},
        "l2": {"n": 4096, "k": 2048},
    }
    for stage in args.stages:
        shape = shapes[stage]
        for t in args.tokens:
            run_case(ext, stage, t, shape["n"], shape["k"], args.warmup, args.repeat, args.seed)


if __name__ == "__main__":
    main()
