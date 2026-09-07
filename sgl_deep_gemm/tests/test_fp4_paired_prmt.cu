#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <deep_gemm/common/fp4_decode_detail.cuh>

constexpr uint64_t kCasesPerMode = 256ull * 65536ull;
constexpr uint64_t kNumCases = 3ull * kCasesPerMode;

__global__ void check_paired_prmt(unsigned long long* mismatches) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < kNumCases; i += gridDim.x * blockDim.x) {
        const uint32_t half = static_cast<uint32_t>(i) & 0xffffu;
        const uint32_t scale = (i >> 16) & 0xffu;
        const uint32_t mode = i / kCasesPerMode;
        uint32_t packed = half | ((mode == 0 ? half ^ 0xffffu : half) << 16);
        if (mode == 2) {
            packed = static_cast<uint32_t>(i) * 0x9e3779b9u + 0x85ebca6bu;
            packed ^= packed >> 16;
            packed *= 0x7feb352du;
            packed ^= packed >> 15;
        }
        const uint64_t lut = deep_gemm::fp4_decode_detail::pack_scaled_e4m3_lut_from_e8m0_const(scale);
        const uint32_t lut_lo = static_cast<uint32_t>(lut);
        const uint32_t lut_hi = static_cast<uint32_t>(lut >> 32);
        uint32_t lo, hi;
        deep_gemm::fp4_decode_detail::fp4x8_to_scaled_e4m3x8_lut(
            packed, lut_lo, lut_hi, lo, hi);
        const uint32_t old_lo = deep_gemm::fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed & 0xffffu, lut_lo, lut_hi);
        const uint32_t old_hi = deep_gemm::fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed >> 16, lut_lo, lut_hi);
        uint64_t reference = 0;
        #pragma unroll
        for (uint32_t nibble_idx = 0; nibble_idx < 8; ++nibble_idx) {
            const uint32_t nibble = (packed >> (4 * nibble_idx)) & 0xfu;
            const uint64_t value = ((lut >> (8 * (nibble & 7u))) & 0xffu) |
                                   ((nibble & 8u) << 4);
            reference |= value << (8 * nibble_idx);
        }
        const uint64_t actual = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
        if (actual != reference || lo != old_lo || hi != old_hi)
            atomicAdd(mismatches, 1ull);
    }
}

#define CUDA_CHECK(call) do { \
    const cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
        return 2; \
    } \
} while (0)

int main() {
    unsigned long long* device_mismatches = nullptr;
    CUDA_CHECK(cudaMalloc(&device_mismatches, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(device_mismatches, 0, sizeof(unsigned long long)));
    check_paired_prmt<<<1024, 256>>>(device_mismatches);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long mismatches = 0;
    CUDA_CHECK(cudaMemcpy(&mismatches, device_mismatches, sizeof(mismatches), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_mismatches));
    std::printf("cases=%llu ue8m0_values=256 mismatches=%llu\n",
                static_cast<unsigned long long>(kNumCases), mismatches);
    return mismatches == 0 ? 0 : 1;
}
