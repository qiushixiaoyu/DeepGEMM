#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <cuda_runtime.h>
#define DG_MEGA_MOE_FP4_PACKED_GMMA_DESC 1
#include <cute/numeric/integer_sequence.hpp>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/mma/sm90.cuh>

#define CHECK_CUDA(expr) do { const auto rc = (expr); if (rc != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(rc)); \
    std::exit(2); } } while (0)

CUTE_HOST_DEVICE uint64_t reference(uint32_t address, int layout,
                                    uint32_t leading, uint32_t stride) {
    cute::GmmaDescriptor d;
    d.bitfield.start_address_ = address >> 4;
    d.bitfield.layout_type_ = layout;
    d.bitfield.leading_byte_offset_ = leading >> 4;
    d.bitfield.stride_byte_offset_ = stride >> 4;
    d.bitfield.base_offset_ = 0;
    return d.desc_;
}

__global__ void test_fields(unsigned long long* mismatches) {
    const uint32_t address = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t offsets[] = {0, 15, 16, 1024, 262128, 262143,
                                262144, 0xffffffffu};
    const uint32_t advances[] = {0, 32, 8192, 0xfffffff0u};
    unsigned long long errors = 0;
    for (int layout = -1; layout <= 4; ++layout)
        for (int i = 0; i < 8; ++i) {
            const uint32_t leading = offsets[i], stride = offsets[7 - i];
            errors += reference(address, layout, leading, stride) !=
                deep_gemm::mma::sm90::pack_smem_desc_bits(
                    address, layout, leading, stride);
            const uint64_t base = reference(address, layout, leading, stride);
            for (int j = 0; j < 4; ++j) {
                const uint32_t offset = advances[j];
                errors += deep_gemm::mma::sm90::advance_smem_desc_bits(base, offset) !=
                    reference(address + offset, layout, leading, stride);
            }
        }
    if (errors) atomicAdd(mismatches, errors);
}

__global__ void test_pointer(unsigned long long* mismatches) {
    __shared__ __align__(16) char storage[32768];
    unsigned long long errors = 0;
    for (int i = threadIdx.x * 16; i < sizeof(storage); i += blockDim.x * 16) {
        auto* ptr = storage + i;
        const auto addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
        errors += deep_gemm::mma::sm90::make_smem_desc(ptr, 1).desc_ !=
                  reference(addr, 1, 0, 1024);
        const auto base = deep_gemm::mma::sm90::make_smem_desc(storage, 1);
        errors += deep_gemm::mma::sm90::advance_smem_desc_bits(base.desc_, i) !=
                  deep_gemm::mma::sm90::make_smem_desc(ptr, 1).desc_;
    }
    if (errors) atomicAdd(mismatches, errors);
}

int main() {
    unsigned long long host_errors = 0;
    for (uint32_t address : {0u, 15u, 16u, 262128u, 262144u, 0xffffffffu})
        for (int layout = -1; layout <= 4; ++layout)
            host_errors += reference(address, layout, 0xffffffffu, 1024) !=
                deep_gemm::mma::sm90::pack_smem_desc_bits(
                    address, layout, 0xffffffffu, 1024);
    unsigned long long* device_errors = nullptr;
    CHECK_CUDA(cudaMalloc(&device_errors, sizeof(*device_errors)));
    CHECK_CUDA(cudaMemset(device_errors, 0, sizeof(*device_errors)));
    test_fields<<<1024, 256>>>(device_errors);
    CHECK_CUDA(cudaGetLastError());
    test_pointer<<<1, 256>>>(device_errors);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    unsigned long long errors = 0;
    CHECK_CUDA(cudaMemcpy(&errors, device_errors, sizeof(errors), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_errors));
    std::printf("[GMMA_DESC_UNIT] field_cases=12582912 advance_cases=50331648 pointer_cases=4096 host_cases=36 mismatches=%llu\n",
                errors + host_errors);
    return errors + host_errors ? 1 : 0;
}
