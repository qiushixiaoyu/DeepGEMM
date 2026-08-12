#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

// Inter-node cross-rank barrier uses NVSHMEM for remote ranks. Only pulled in
// when the host codegen defines DG_MEGA_MOE_INTERNODE (i.e. num_ranks spans
// multiple nodes); single-node kernels never touch NVSHMEM.
#ifdef DG_MEGA_MOE_INTERNODE
#include <nvshmem.h>
#include <nvshmemx.h>
#include <deep_gemm/comm/ibgda.cuh>
#endif

namespace deep_gemm::comm {

// 60s timeout, at 2 GHz
constexpr int64_t kNumTimeoutCycles = 60ll * 2000000000ll;

CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename WorkspaceT, typename sync_scope_t>
CUTLASS_DEVICE void grid_sync(const WorkspaceT& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.template get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1000) && \
        !(defined(DG_NVLINK_BARRIER_VERBOSE_TIMEOUT) && DG_NVLINK_BARRIER_VERBOSE_TIMEOUT)
        do {
            new_value = ptx::ld_acq(count_ptr);
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
#else
        const auto start_clock = clock64();
        do {
            new_value = ptx::ld_acq(count_ptr);
            if (clock64() - start_clock >= kNumTimeoutCycles) {
                printf("DeepGEMM grid sync timeout: sm=%u, thread=%u, grid_sync_idx=%u, old=%u, current=%u, expected_tag=%u\n",
                       sm_idx, thread_idx, kGridSyncIndex, old_value, new_value, old_value ^ kFinishSumTag);
                DG_DEVICE_ASSERT(false and "Grid sync timeout");
            }
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
#endif
    }
    sync_scope();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag, typename WorkspaceT, typename sync_scope_t>
CUTLASS_DEVICE void nvlink_barrier(const WorkspaceT& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true,
                                   unsigned long long* profile_quiet_ns = nullptr,
                                   unsigned long long* profile_sync_ns = nullptr,
                                   unsigned long long* profile_ibgda_ns = nullptr) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    // Grid sync before NVLink signaling
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    // NVLink cross-rank barrier, only SM 0 participates
    if (sm_idx == 0) {
#ifdef DG_MEGA_MOE_INTERNODE
        // Inter-node: 先逐 (远端 pe, qp) 等待本 rank 经 IBGDA verbs 发出的所有 WQE 完成
        // (与 DeepEP 同源的 per-QP quiet；每个 (pe,qp) 恰好由一个线程 poll，满足 poll_cq
        // 的并发约束——调用侧保证此刻没有其他 warp 在用这些 QP)。公开 nvshmem_quiet()
        // 只兜底非 verbs 路径(如 sync_all 内部)。到达语义仍由 NVSHMEM collective
        // sync 承担(单线程 PE 级 barrier)。
        {
            const auto n_qps = ibgda::num_qps();
            for (uint32_t i = thread_idx; i < kNumRanks * n_qps; i += kNumThreads) {
                const auto pe = i / n_qps;
                if (pe / DG_MEGA_MOE_NVL_PEERS != sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS)
                    ibgda::quiet(static_cast<int>(pe), static_cast<int>(i % n_qps));
            }
        }
        if (profile_ibgda_ns != nullptr and thread_idx == 0)
            *profile_ibgda_ns = ptx::get_globaltimer();
        // Diagnostic A/B: the explicit per-(pe, qp) ibgda::quiet loop above
        // already drains every verbs-path QP; this global quiet measured
        // ~450 us/launch (63% of the b1 kernel).  Gated for a correctness
        // review before it can become the default.
#if !defined(DG_MEGA_MOE_SKIP_NVSHMEM_QUIET) and !defined(DG_MEGA_MOE_NO_TAG3)
        nvshmem_quiet();
#endif
        if (profile_quiet_ns != nullptr and thread_idx == 0)
            *profile_quiet_ns = ptx::get_globaltimer();
#ifndef DG_MEGA_MOE_NO_TAG3
        // With epoch-tagged control-plane slots (and no clearing) the cleanup
        // is purely rank-local, so this cross-rank rendezvous is unnecessary.
        // Kernels are stream-ordered, which already separates a rank's own
        // launches; the per-QP quiet above still protects the staging buffer.
        if (thread_idx == 0)
            nvshmem_sync_all();
#endif
        if (profile_sync_ns != nullptr and thread_idx == 0)
            *profile_sync_ns = ptx::get_globaltimer();
        sync_scope();
#else
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // Send signals to remote ranks
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // Update status and wait arrival
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            const auto start_clock = clock64();
            while (ptx::ld_acq_sys(signal_ptr) != target) {
                if (clock64() - start_clock >= kNumTimeoutCycles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1000) && \
        !(defined(DG_NVLINK_BARRIER_VERBOSE_TIMEOUT) && DG_NVLINK_BARRIER_VERBOSE_TIMEOUT)
                    DG_TRAP_ONLY_DEVICE_ASSERT(false);
#else
                    printf("DeepGEMM NVLink barrier timeout: rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                           sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
                    DG_DEVICE_ASSERT(false and "NVLink barrier timeout");
#endif
                }
            }
        }
#endif
    }

    // Grid sync after NVLink completion
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}

} // namespace deep_gemm::comm
