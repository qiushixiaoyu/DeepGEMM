#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../utils/exception.hpp"
#include "../utils/system.hpp"

#ifdef DG_HAS_NCCL_GIN
#include <nccl.h>
#include <nccl_device.h>
#endif

namespace deep_gemm::mega {

struct SM90MegaMoECPUProxyLaunch {
    void* buffer = nullptr;
    const void* dev_comms = nullptr;
    const void* windows = nullptr;
    uint64_t combine_offset = 0;
    uint64_t staging_offset = 0;
    uint64_t signal_epoch_offset = 0;
    uint64_t completion_request_offset = 0;
    uint64_t combine_slot_bytes = 0;
    uint64_t staging_slot_bytes = 0;
    uint64_t total_bytes = 0;
    uint32_t num_stage_rows = 0;
    uint32_t num_slots = 1;
};

#ifdef DG_HAS_NCCL_GIN

struct SM90MegaMoECPUProxyState {
    static constexpr int kNumRemotePeers = 8;
    std::array<ncclComm_t, kNumRemotePeers> comms{};
    std::array<ncclWindow_t, kNumRemotePeers> windows{};
    std::array<ncclDevComm, kNumRemotePeers> dev_comms{};
    ncclDevComm* dev_comms_device = nullptr;
    ncclWindow_t* windows_device = nullptr;
    void* buffer = nullptr;
    size_t total_bytes = 0;
    size_t combine_offset = 0;
    size_t staging_offset = 0;
    size_t signal_epoch_offset = 0;
    size_t completion_request_offset = 0;
    size_t combine_slot_bytes = 0;
    size_t staging_slot_bytes = 0;
    int device = -1;
    int rank = -1;
    int num_ranks = 0;
    int num_experts = 0;
    int num_topk = 0;
    int num_max_tokens_per_rank = 0;
    int hidden = 0;
    int num_stage_rows = 0;
    int num_slots = 1;
};

// This header is consumed by both the TVM-FFI registration unit and the
// MegaMoE host-runtime path.  The proxy lifetime must therefore be process
// global rather than translation-unit local: a namespace-scope `static`
// gives init() and the launch path distinct state when they are emitted from
// different units/DSOs.
inline std::unique_ptr<SM90MegaMoECPUProxyState>
    sm90_mega_moe_cpu_proxy_state;

static void sm90_mega_moe_check_nccl(ncclResult_t result, const char* expression) {
    if (result == ncclSuccess)
        return;
    fprintf(stderr, "SM90 MegaMoE CPU proxy: %s failed: %s\n",
            expression, ncclGetErrorString(result));
    DG_HOST_ASSERT(false and "NCCL GIN CPU proxy call failed");
}

#define DG_SM90_MEGA_MOE_NCCL_CHECK(expr) \
    sm90_mega_moe_check_nccl((expr), #expr)

static size_t sm90_mega_moe_cpu_proxy_align(const size_t value) {
    constexpr size_t kAlignment = NCCL_WIN_REQUIRED_ALIGNMENT;
    return (value + kAlignment - 1) / kAlignment * kAlignment;
}

static torch::Tensor sm90_mega_moe_cpu_proxy_get_unique_id() {
    constexpr int kNumPairs =
        SM90MegaMoECPUProxyState::kNumRemotePeers *
        SM90MegaMoECPUProxyState::kNumRemotePeers;
    auto result = torch::empty(
        {kNumPairs, static_cast<int64_t>(sizeof(ncclUniqueId))},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    auto bytes = result.data_ptr<uint8_t>();
    for (int pair = 0; pair < kNumPairs; ++ pair) {
        ncclUniqueId id{};
        DG_SM90_MEGA_MOE_NCCL_CHECK(ncclGetUniqueId(&id));
        std::memcpy(bytes + pair * sizeof(id), &id, sizeof(id));
    }
    return result;
}

static void sm90_mega_moe_cpu_proxy_init(
    const torch::Tensor& unique_id,
    const int rank,
    const int num_ranks,
    const int num_experts,
    const int num_topk,
    const int num_max_tokens_per_rank,
    const int hidden) {
    DG_HOST_ASSERT(sm90_mega_moe_cpu_proxy_state == nullptr);
    DG_HOST_ASSERT(unique_id.scalar_type() == torch::kUInt8);
    constexpr int kNumRemotePeers =
        SM90MegaMoECPUProxyState::kNumRemotePeers;
    DG_HOST_ASSERT(
        unique_id.numel() ==
        kNumRemotePeers * kNumRemotePeers * sizeof(ncclUniqueId));
    DG_HOST_ASSERT(rank >= 0 and rank < num_ranks);
    DG_HOST_ASSERT(num_ranks > 8 and num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_topk > 0 and num_max_tokens_per_rank > 0 and hidden > 0);

    auto uid_cpu = unique_id.to(torch::kCPU).contiguous();
    auto state = std::make_unique<SM90MegaMoECPUProxyState>();
    state->rank = rank;
    state->num_ranks = num_ranks;
    state->num_experts = num_experts;
    state->num_topk = num_topk;
    state->num_max_tokens_per_rank = num_max_tokens_per_rank;
    state->hidden = hidden;
    state->num_stage_rows = static_cast<int>(
        layout::get_num_sm90_combine_ring_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk,
            num_experts / num_ranks));
    state->num_slots = std::max(
        1, get_env<int>("DG_MEGA_MOE_FP4_CPU_PROXY_SLOTS", 1));
    DG_HOST_ASSERT(state->num_slots <= 64);
    if (get_env<int>("DG_MEGA_MOE_FP4_CPU_PROXY_ASYNC_CREDIT", 0) != 0) {
        const int full_pool_rows = static_cast<int>(
            layout::get_num_max_pool_tokens(
                num_ranks, num_max_tokens_per_rank, num_topk,
                num_experts / num_ranks));
        DG_HOST_ASSERT(
            state->num_stage_rows == full_pool_rows and
            "async CPU proxy credit requires full-pool staging");
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&state->device));

    const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(nv_bfloat16);
    state->combine_slot_bytes = sm90_mega_moe_cpu_proxy_align(
        static_cast<size_t>(num_topk) * num_max_tokens_per_rank * row_bytes);
    state->staging_slot_bytes = sm90_mega_moe_cpu_proxy_align(
        static_cast<size_t>(state->num_stage_rows) * row_bytes);
    const size_t signal_epoch_bytes = sm90_mega_moe_cpu_proxy_align(
        static_cast<size_t>(num_experts / num_ranks) * num_ranks *
        sizeof(uint64_t));
    const size_t completion_request_bytes = sm90_mega_moe_cpu_proxy_align(
        static_cast<size_t>(state->num_slots) * kNumRemotePeers *
        sizeof(ncclGinRequest_t));
    state->combine_offset = 0;
    state->staging_offset =
        state->combine_slot_bytes * state->num_slots;
    state->signal_epoch_offset = state->staging_offset +
        state->staging_slot_bytes * state->num_slots;
    state->completion_request_offset =
        state->signal_epoch_offset + signal_epoch_bytes;
    state->total_bytes = state->completion_request_offset +
        completion_request_bytes;

    DG_HOST_ASSERT(num_ranks == 2 * kNumRemotePeers);
    const int node_idx = rank / kNumRemotePeers;
    const int local_rank = rank % kNumRemotePeers;
    const auto uid_bytes = uid_cpu.data_ptr<uint8_t>();
    // Build 64 independent cross-node rank pairs. Each process participates in
    // eight two-rank communicators; the wavefront call order below avoids a
    // global full-connection provider setup, which is unsupported on COMM5/6.
    for (int remote_local_rank = 0;
         remote_local_rank < kNumRemotePeers; ++ remote_local_rank) {
        const int node0_local_rank = node_idx == 0 ?
            local_rank : remote_local_rank;
        const int node1_local_rank = node_idx == 0 ?
            remote_local_rank : local_rank;
        const int pair_idx =
            node0_local_rank * kNumRemotePeers + node1_local_rank;
        ncclUniqueId id{};
        std::memcpy(
            &id, uid_bytes + pair_idx * sizeof(ncclUniqueId), sizeof(id));
        DG_SM90_MEGA_MOE_NCCL_CHECK(ncclCommInitRank(
            &state->comms[remote_local_rank], 2, id, node_idx));
    }

    DG_SM90_MEGA_MOE_NCCL_CHECK(
        ncclMemAlloc(&state->buffer, state->total_bytes));
    DG_CUDA_RUNTIME_CHECK(cudaMemset(state->buffer, 0, state->total_bytes));

    for (int remote_local_rank = 0;
         remote_local_rank < kNumRemotePeers; ++ remote_local_rank) {
        auto comm = state->comms[remote_local_rank];
        ncclCommProperties properties = NCCL_COMM_PROPERTIES_INITIALIZER;
        DG_SM90_MEGA_MOE_NCCL_CHECK(
            ncclCommQueryProperties(comm, &properties));
        DG_HOST_ASSERT(properties.deviceApiSupport);
        DG_HOST_ASSERT(properties.ginType == NCCL_GIN_TYPE_PROXY);
        DG_SM90_MEGA_MOE_NCCL_CHECK(ncclCommWindowRegister(
            comm, state->buffer, state->total_bytes,
            &state->windows[remote_local_rank], NCCL_WIN_COLL_SYMMETRIC));

        ncclDevCommRequirements requirements =
            NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        requirements.ginForceEnable = true;
        requirements.ginContextCount = 1;
        requirements.ginSignalCount = num_experts / num_ranks;
        requirements.ginCounterCount = 0;
        requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
        requirements.ginQueueDepth = 512;
        requirements.ginStrongSignalsRequired = true;
        requirements.ginVaSignalsRequired = false;
        DG_SM90_MEGA_MOE_NCCL_CHECK(ncclDevCommCreate(
            comm, &requirements, &state->dev_comms[remote_local_rank]));
    }

    DG_CUDA_RUNTIME_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state->dev_comms_device),
        sizeof(state->dev_comms)));
    DG_CUDA_RUNTIME_CHECK(cudaMemcpy(
        state->dev_comms_device, state->dev_comms.data(),
        sizeof(state->dev_comms),
        cudaMemcpyHostToDevice));
    DG_CUDA_RUNTIME_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state->windows_device),
        sizeof(state->windows)));
    DG_CUDA_RUNTIME_CHECK(cudaMemcpy(
        state->windows_device, state->windows.data(), sizeof(state->windows),
        cudaMemcpyHostToDevice));

    sm90_mega_moe_cpu_proxy_state = std::move(state);
}

static bool sm90_mega_moe_cpu_proxy_is_initialized() {
    return sm90_mega_moe_cpu_proxy_state != nullptr;
}

static SM90MegaMoECPUProxyLaunch sm90_mega_moe_cpu_proxy_get_launch(
    const int num_ranks,
    const int num_experts,
    const int num_topk,
    const int num_max_tokens_per_rank,
    const int hidden) {
    DG_HOST_ASSERT(sm90_mega_moe_cpu_proxy_state != nullptr);
    const auto& state = *sm90_mega_moe_cpu_proxy_state;
    int current_device = -1;
    DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&current_device));
    DG_HOST_ASSERT(state.device == current_device);
    DG_HOST_ASSERT(state.num_ranks == num_ranks);
    DG_HOST_ASSERT(state.num_experts == num_experts);
    DG_HOST_ASSERT(state.num_topk == num_topk);
    DG_HOST_ASSERT(state.num_max_tokens_per_rank == num_max_tokens_per_rank);
    DG_HOST_ASSERT(state.hidden == hidden);
    return {
        .buffer = state.buffer,
        .dev_comms = state.dev_comms_device,
        .windows = state.windows_device,
        .combine_offset = state.combine_offset,
        .staging_offset = state.staging_offset,
        .signal_epoch_offset = state.signal_epoch_offset,
        .completion_request_offset = state.completion_request_offset,
        .combine_slot_bytes = state.combine_slot_bytes,
        .staging_slot_bytes = state.staging_slot_bytes,
        .total_bytes = state.total_bytes,
        .num_stage_rows = static_cast<uint32_t>(state.num_stage_rows),
        .num_slots = static_cast<uint32_t>(state.num_slots),
    };
}

static void sm90_mega_moe_cpu_proxy_destroy() {
    if (sm90_mega_moe_cpu_proxy_state == nullptr)
        return;
    auto state = std::move(sm90_mega_moe_cpu_proxy_state);
    DG_CUDA_RUNTIME_CHECK(cudaSetDevice(state->device));
    auto stream = at::cuda::getCurrentCUDAStream(state->device).stream();
    uint32_t* barrier = nullptr;
    DG_CUDA_RUNTIME_CHECK(cudaMalloc(&barrier, sizeof(uint32_t)));
    for (int remote_local_rank = 0;
         remote_local_rank < SM90MegaMoECPUProxyState::kNumRemotePeers;
         ++ remote_local_rank) {
        auto comm = state->comms[remote_local_rank];
        DG_CUDA_RUNTIME_CHECK(
            cudaMemsetAsync(barrier, 0, sizeof(uint32_t), stream));
        DG_SM90_MEGA_MOE_NCCL_CHECK(ncclAllReduce(
            barrier, barrier, 1, ncclUint32, ncclSum, comm, stream));
        DG_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
        DG_SM90_MEGA_MOE_NCCL_CHECK(ncclDevCommDestroy(
            comm, &state->dev_comms[remote_local_rank]));
        DG_SM90_MEGA_MOE_NCCL_CHECK(ncclCommWindowDeregister(
            comm, state->windows[remote_local_rank]));
        DG_SM90_MEGA_MOE_NCCL_CHECK(ncclCommFinalize(comm));
        DG_SM90_MEGA_MOE_NCCL_CHECK(ncclCommDestroy(comm));
    }
    DG_CUDA_RUNTIME_CHECK(cudaFree(barrier));
    DG_CUDA_RUNTIME_CHECK(cudaFree(state->dev_comms_device));
    DG_CUDA_RUNTIME_CHECK(cudaFree(state->windows_device));
    DG_SM90_MEGA_MOE_NCCL_CHECK(ncclMemFree(state->buffer));
}

#undef DG_SM90_MEGA_MOE_NCCL_CHECK

#else

static torch::Tensor sm90_mega_moe_cpu_proxy_get_unique_id() {
    DG_HOST_ASSERT(false and "DeepGEMM was built without NCCL GIN support");
    return {};
}

static void sm90_mega_moe_cpu_proxy_init(
    const torch::Tensor&, const int, const int, const int, const int,
    const int, const int) {
    DG_HOST_ASSERT(false and "DeepGEMM was built without NCCL GIN support");
}

static bool sm90_mega_moe_cpu_proxy_is_initialized() {
    return false;
}

static SM90MegaMoECPUProxyLaunch sm90_mega_moe_cpu_proxy_get_launch(
    const int, const int, const int, const int, const int) {
    DG_HOST_ASSERT(false and "DeepGEMM was built without NCCL GIN support");
    return {};
}

static void sm90_mega_moe_cpu_proxy_destroy() {}

#endif

} // namespace deep_gemm::mega
