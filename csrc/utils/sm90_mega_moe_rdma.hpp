#pragma once

#include "exception.hpp"

namespace deep_gemm {

// This operator uses eight-GPU NVLink domains connected by RDMA. Keep this
// guard shared by the public sizing/launch APIs and direct JIT entry points.
inline void check_sm90_mega_moe_rdma_topology(const int num_ranks) {
    if (num_ranks <= 8 or num_ranks > 64 or num_ranks % 8 != 0)
        DG_HOST_UNREACHABLE(
            "SM90 FP4/FP8 MegaMoE is RDMA-only: requires 16..64 ranks in "
            "complete eight-GPU NVLink domains with node-contiguous ranks; "
            "single-node execution is not supported.");
}

} // namespace deep_gemm
