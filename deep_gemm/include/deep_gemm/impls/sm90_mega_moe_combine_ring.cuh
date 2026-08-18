#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/comm/ibgda.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm {

// A launch-local, variable-length expert segment ring for registered BF16
// combine staging.  Logical pool metadata and arrival counters remain in the
// full-pool address space; only the RDMA source payload uses these physical
// segment rows.
template <
    bool kEnabled,
    uint32_t kCapacityRows,
    uint32_t kNumExpertsPerRank,
    uint32_t kNumRanks>
struct SM90CombineStageRing {
    layout::SM90Workspace workspace;

    // Dispatch READ and scatter WRITE share QP(peer, expert).  The packed
    // metadata gateway alone reserves QP kNumExpertsPerRank.
    CUTLASS_DEVICE static constexpr int scatter_qp_id(
        const uint32_t& local_expert_idx) {
        return static_cast<int>(local_expert_idx);
    }

    CUTLASS_DEVICE explicit SM90CombineStageRing(
        const layout::SM90Workspace& workspace): workspace(workspace) {}

    CUTLASS_DEVICE void wait_for_launch_init() const {
        if constexpr (kEnabled) {
            constexpr uint64_t kTimeoutCycles = 120ull * 2000000000ull;
            const auto start = clock64();
            while (ptx::ld_acq_sys(
                       &workspace.get_combine_ring_control_ptr()->init_epoch) ==
                   0) {
                DG_TRAP_ONLY_DEVICE_ASSERT(
                    clock64() - start < kTimeoutCycles);
            }
        }
    }

    // Called by rank-local SM 0 warp 0 before this launch starts touching
    // route metadata.  Kernel launch ordering guarantees the previous launch
    // has stopped producing new WQEs; exact ready-WQE completions below
    // guarantee the RNIC has also stopped reading every still-live source
    // segment.  Experts/QPs are independent, so the warp retires them in
    // parallel instead of serializing every CQ read on one lane.
    CUTLASS_DEVICE void begin_launch(const uint64_t& launch_epoch) const {
        if constexpr (kEnabled) {
            const uint32_t lane_idx = ptx::get_lane_idx();
            auto control = workspace.get_combine_ring_control_ptr();
            const uint64_t previous_epoch = launch_epoch - 1;
            const uint32_t queue_head = control->queue_head;
            const uint32_t queue_tail = control->queue_tail;
            const bool already_reclaimed =
                ptx::ld_acq_sys(&control->reclaimed_epoch) == previous_epoch;

            for (uint32_t queue_idx = queue_tail + lane_idx;
                 queue_idx < queue_head; queue_idx += 32) {
                const uint32_t expert_idx =
                    *workspace.get_combine_ring_queue_ptr(queue_idx);
                DG_DEVICE_ASSERT(expert_idx < kNumExpertsPerRank);
                auto segment =
                    workspace.get_combine_ring_segment_ptr(expert_idx);

                if (not already_reclaimed and
                    segment->state == static_cast<uint32_t>(
                        layout::SM90CombineRingSegmentState::Ready)) {
                    constexpr uint64_t kTimeoutCycles =
                        120ull * 2000000000ull;
                    const auto start = clock64();
                    while (ptx::ld_acq_sys(&segment->publish_epoch) !=
                           previous_epoch) {
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            clock64() - start < kTimeoutCycles);
                    }
                    for (uint32_t dst = 0; dst < kNumRanks; ++ dst) {
                        const uint64_t completion = ptx::ld_acq_sys(
                            workspace.get_combine_ring_completion_target_ptr(
                                expert_idx, dst));
                        if (completion != 0)
                            comm::ibgda::wait_until(
                                static_cast<int>(dst),
                                scatter_qp_id(expert_idx), completion);
                    }
                }
            }
            __syncwarp();

            if (lane_idx == 0) {
                control->alloc_head_ticket = 0;
                control->reclaim_tail_ticket = 0;
                control->allocator_lock = 0;
                control->queue_head = 0;
                control->queue_tail = 0;
                control->prealloc_epoch = 0;
                control->prealloc_success = 0;
            }
            for (uint32_t expert = lane_idx; expert < kNumExpertsPerRank;
                 expert += 32) {
                auto segment = workspace.get_combine_ring_segment_ptr(expert);
                segment->reclaim_begin_ticket = 0;
                segment->end_ticket = 0;
                segment->publish_epoch = 0;
                segment->physical_base_row = 0;
                segment->num_rows = 0;
                segment->state = static_cast<uint32_t>(
                    layout::SM90CombineRingSegmentState::Free);
                segment->has_internode_rows = 0;
                *workspace.get_combine_ring_queue_ptr(expert) = 0;
            }
            for (uint32_t idx = lane_idx;
                 idx < kNumExpertsPerRank * kNumRanks; idx += 32) {
                const uint32_t expert = idx / kNumRanks;
                const uint32_t dst = idx % kNumRanks;
                *workspace.get_combine_ring_completion_target_ptr(
                    expert, dst) = 0;
            }
            __syncwarp();
            if (lane_idx == 0) {
                __threadfence_system();
                ptx::st_release_sys(&control->init_epoch, launch_epoch);
            }
            __syncwarp();
        }
    }

    // Consume the exact local CQ targets as soon as the publisher rendezvous
    // has completed, while the receiver-side combine/reduce is still running.
    // The next launch can then reset descriptors without putting these CQ
    // polls on its critical path.
    CUTLASS_DEVICE void finish_launch(const uint64_t& launch_epoch) const {
        if constexpr (kEnabled) {
            const uint32_t lane_idx = ptx::get_lane_idx();
            auto control = workspace.get_combine_ring_control_ptr();
            const uint32_t queue_head = control->queue_head;
            const uint32_t queue_tail = control->queue_tail;
            // A successful bulk preallocation proves that no physical row was
            // recycled during this launch.  Preserve the original full-pool
            // cross-launch lifetime semantics in that case: the publisher
            // rendezvous has posted every ready WQE, and no local CQ wait is
            // added to the operator critical path.  Exact CQ completion is
            // required only by the incremental allocator, which may actually
            // release and reuse a segment within the same launch.
            const bool reused_segments =
                ptx::ld_acq_sys(&control->prealloc_epoch) != launch_epoch or
                ptx::ld_acq_sys(&control->prealloc_success) == 0;
            for (uint32_t queue_idx = queue_tail + lane_idx;
                 queue_idx < queue_head; queue_idx += 32) {
                const uint32_t expert_idx =
                    *workspace.get_combine_ring_queue_ptr(queue_idx);
                DG_DEVICE_ASSERT(expert_idx < kNumExpertsPerRank);
                auto segment =
                    workspace.get_combine_ring_segment_ptr(expert_idx);
                if (reused_segments) {
                    constexpr uint64_t kTimeoutCycles =
                        120ull * 2000000000ull;
                    const auto start = clock64();
                    while (ptx::ld_acq_sys(&segment->publish_epoch) !=
                           launch_epoch) {
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            clock64() - start < kTimeoutCycles);
                    }
                    for (uint32_t dst = 0; dst < kNumRanks; ++ dst) {
                        const uint64_t completion = ptx::ld_acq_sys(
                            workspace.get_combine_ring_completion_target_ptr(
                                expert_idx, dst));
                        if (completion != 0)
                            comm::ibgda::wait_until(
                                static_cast<int>(dst),
                                scatter_qp_id(expert_idx), completion);
                    }
                }
                segment->state = static_cast<uint32_t>(
                    layout::SM90CombineRingSegmentState::Retired);
            }
            __syncwarp();
            if (lane_idx == 0) {
                __threadfence_system();
                ptx::st_release_sys(
                    &control->reclaimed_epoch, launch_epoch);
            }
            __syncwarp();
        }
    }

    // Eager-count no-wrap fast path.  One controller warp reserves every
    // active expert as a compact prefix before routed GEMM is released.  This
    // removes per-L2-CTA allocator contention for the common case where the
    // actual batch is much smaller than the configured maximum.  If the
    // active rows do not fit, publish a failed decision and retain the normal
    // completion-protected incremental allocator.
    template <uint32_t kBlockM, typename SchedulerT>
    CUTLASS_DEVICE void preallocate_active_experts(
        SchedulerT& scheduler, const uint64_t& launch_epoch) const {
        if constexpr (kEnabled) {
            const uint32_t lane_idx = ptx::get_lane_idx();
            uint64_t total_rows = 0;
            #pragma unroll
            for (uint32_t expert = 0; expert < kNumExpertsPerRank; ++ expert) {
                const uint32_t tokens = scheduler.get_num_tokens(expert);
                if (tokens != 0)
                    total_rows += math::align<uint64_t>(
                        static_cast<uint64_t>(tokens),
                        static_cast<uint64_t>(kBlockM));
            }

            auto control = workspace.get_combine_ring_control_ptr();
            if (lane_idx == 0) {
                DG_DEVICE_ASSERT(control->queue_head == 0);
                DG_DEVICE_ASSERT(control->queue_tail == 0);
                DG_DEVICE_ASSERT(control->alloc_head_ticket == 0);
                control->prealloc_success = total_rows <= kCapacityRows;
            }
            __syncwarp();

            if (total_rows <= kCapacityRows) {
                uint64_t head = 0;
                #pragma unroll
                for (uint32_t expert = 0; expert < kNumExpertsPerRank;
                     ++ expert) {
                    const uint32_t tokens = scheduler.get_num_tokens(expert);
                    if (tokens == 0)
                        continue;
                    const uint32_t rows = math::align<uint32_t>(tokens, kBlockM);
                    if (lane_idx == 0) {
                        auto segment =
                            workspace.get_combine_ring_segment_ptr(expert);
                        *workspace.get_combine_ring_queue_ptr(
                            control->queue_head) = expert;
                        ++ control->queue_head;
                        segment->reclaim_begin_ticket = head;
                        segment->end_ticket = head + rows;
                        segment->publish_epoch = 0;
                        segment->physical_base_row =
                            static_cast<uint32_t>(head);
                        segment->num_rows = rows;
                        segment->state = static_cast<uint32_t>(
                            layout::SM90CombineRingSegmentState::Ready);
                        head += rows;
                    }
                }
                if (lane_idx == 0)
                    control->alloc_head_ticket = head;
            }

            if (lane_idx == 0) {
                __threadfence_system();
                ptx::st_release_sys(&control->prealloc_epoch, launch_epoch);
            }
            __syncwarp();
        }
    }

    CUTLASS_DEVICE bool wait_for_preallocation(
        const uint64_t& launch_epoch) const {
        if constexpr (kEnabled) {
            auto control = workspace.get_combine_ring_control_ptr();
            constexpr uint64_t kTimeoutCycles = 120ull * 2000000000ull;
            const auto start = clock64();
            while (ptx::ld_acq_sys(&control->prealloc_epoch) != launch_epoch)
                DG_TRAP_ONLY_DEVICE_ASSERT(
                    clock64() - start < kTimeoutCycles);
            return ptx::ld_acq_sys(&control->prealloc_success) != 0;
        }
        return false;
    }

    // Cleared by the dispatch controller before it exits.  The next launch
    // cannot begin until this kernel has completed, so current-launch users do
    // not need to observe the gate again.
    CUTLASS_DEVICE void close_launch_gate() const {
        if constexpr (kEnabled)
            ptx::st_release_sys(
                &workspace.get_combine_ring_control_ptr()->init_epoch, 0ull);
    }

    CUTLASS_DEVICE void mark_internode_rows(
        const uint32_t& local_expert_idx) const {
        if constexpr (kEnabled) {
            DG_DEVICE_ASSERT(local_expert_idx < kNumExpertsPerRank);
            atomicExch(
                &workspace.get_combine_ring_segment_ptr(local_expert_idx)
                     ->has_internode_rows,
                1u);
        }
    }

    CUTLASS_DEVICE bool has_internode_rows(
        const uint32_t& local_expert_idx) const {
        if constexpr (kEnabled) {
            return ptx::ld_acq_sys(
                       &workspace.get_combine_ring_segment_ptr(local_expert_idx)
                            ->has_internode_rows) != 0;
        }
        return false;
    }

    CUTLASS_DEVICE void lock_allocator() const {
        if constexpr (kEnabled) {
            auto lock = &workspace.get_combine_ring_control_ptr()->allocator_lock;
            constexpr uint64_t kTimeoutCycles = 120ull * 2000000000ull;
            const auto start = clock64();
            while (atomicCAS(lock, 0u, 1u) != 0u)
                DG_TRAP_ONLY_DEVICE_ASSERT(
                    clock64() - start < kTimeoutCycles);
        }
    }

    CUTLASS_DEVICE void unlock_allocator() const {
        if constexpr (kEnabled) {
            __threadfence_system();
            atomicExch(
                &workspace.get_combine_ring_control_ptr()->allocator_lock, 0u);
        }
    }

    CUTLASS_DEVICE void reclaim_oldest(const uint64_t& launch_epoch) const {
        if constexpr (kEnabled) {
            auto control = workspace.get_combine_ring_control_ptr();
            DG_DEVICE_ASSERT(control->queue_tail < control->queue_head);
            const uint32_t expert_idx =
                *workspace.get_combine_ring_queue_ptr(control->queue_tail);
            DG_DEVICE_ASSERT(expert_idx < kNumExpertsPerRank);
            auto segment = workspace.get_combine_ring_segment_ptr(expert_idx);

            constexpr uint64_t kTimeoutCycles = 120ull * 2000000000ull;
            const auto start = clock64();
            while (ptx::ld_acq_sys(&segment->publish_epoch) != launch_epoch)
                DG_TRAP_ONLY_DEVICE_ASSERT(
                    clock64() - start < kTimeoutCycles);

            for (uint32_t dst = 0; dst < kNumRanks; ++ dst) {
                const uint64_t completion = ptx::ld_acq_sys(
                    workspace.get_combine_ring_completion_target_ptr(
                        expert_idx, dst));
                if (completion != 0)
                    comm::ibgda::wait_until(
                        static_cast<int>(dst), scatter_qp_id(expert_idx),
                        completion);
            }
            __threadfence_system();
            segment->state = static_cast<uint32_t>(
                layout::SM90CombineRingSegmentState::Retired);
            control->reclaim_tail_ticket = segment->end_ticket;
            ++ control->queue_tail;
        }
    }

    CUTLASS_DEVICE uint32_t reserve_segment(
        const uint32_t& local_expert_idx,
        const uint32_t& num_rows,
        const uint64_t& launch_epoch) const {
        if constexpr (kEnabled) {
            DG_DEVICE_ASSERT(local_expert_idx < kNumExpertsPerRank);
            DG_DEVICE_ASSERT(num_rows > 0 and num_rows <= kCapacityRows);
            wait_for_launch_init();

            auto segment =
                workspace.get_combine_ring_segment_ptr(local_expert_idx);
            uint32_t state = ptx::ld_acq_sys(&segment->state);
            if (state == static_cast<uint32_t>(
                             layout::SM90CombineRingSegmentState::Ready))
                return segment->physical_base_row;

            const uint32_t old_state = atomicCAS(
                &segment->state,
                static_cast<uint32_t>(
                    layout::SM90CombineRingSegmentState::Free),
                static_cast<uint32_t>(
                    layout::SM90CombineRingSegmentState::Allocating));
            if (old_state == static_cast<uint32_t>(
                                 layout::SM90CombineRingSegmentState::Free)) {
                lock_allocator();
                auto control = workspace.get_combine_ring_control_ptr();
                uint64_t reclaim_begin = 0;
                uint64_t payload_begin = 0;
                uint64_t end = 0;
                while (true) {
                    uint64_t head = control->alloc_head_ticket;
                    uint64_t tail = control->reclaim_tail_ticket;
                    if (control->queue_head == control->queue_tail) {
                        // With no live segment, discard tail fragmentation and
                        // begin at the next physical row zero.
                        head = math::align<uint64_t>(
                            head, static_cast<uint64_t>(kCapacityRows));
                        tail = head;
                        control->alloc_head_ticket = head;
                        control->reclaim_tail_ticket = tail;
                    }

                    reclaim_begin = head;
                    payload_begin = head;
                    const uint64_t physical =
                        payload_begin % static_cast<uint64_t>(kCapacityRows);
                    if (physical + num_rows > kCapacityRows)
                        payload_begin += kCapacityRows - physical;
                    end = payload_begin + num_rows;
                    if (end - tail <= kCapacityRows)
                        break;
                    reclaim_oldest(launch_epoch);
                }

                DG_DEVICE_ASSERT(control->queue_head < kNumExpertsPerRank);
                *workspace.get_combine_ring_queue_ptr(control->queue_head) =
                    local_expert_idx;
                ++ control->queue_head;
                control->alloc_head_ticket = end;

                segment->reclaim_begin_ticket = reclaim_begin;
                segment->end_ticket = end;
                segment->publish_epoch = 0;
                segment->physical_base_row = static_cast<uint32_t>(
                    payload_begin % static_cast<uint64_t>(kCapacityRows));
                segment->num_rows = num_rows;
                __threadfence_system();
                atomicExch(
                    &segment->state,
                    static_cast<uint32_t>(
                        layout::SM90CombineRingSegmentState::Ready));
                unlock_allocator();
                return segment->physical_base_row;
            }

            DG_DEVICE_ASSERT(
                old_state == static_cast<uint32_t>(
                                 layout::SM90CombineRingSegmentState::Allocating) or
                old_state == static_cast<uint32_t>(
                                 layout::SM90CombineRingSegmentState::Ready));
            constexpr uint64_t kTimeoutCycles = 120ull * 2000000000ull;
            const auto start = clock64();
            while (ptx::ld_acq_sys(&segment->state) !=
                   static_cast<uint32_t>(
                       layout::SM90CombineRingSegmentState::Ready)) {
                DG_TRAP_ONLY_DEVICE_ASSERT(
                    clock64() - start < kTimeoutCycles);
            }
            return segment->physical_base_row;
        }
        return 0;
    }

    CUTLASS_DEVICE uint32_t wait_segment_base(
        const uint32_t& local_expert_idx) const {
        if constexpr (kEnabled) {
            auto segment =
                workspace.get_combine_ring_segment_ptr(local_expert_idx);
            constexpr uint64_t kTimeoutCycles = 120ull * 2000000000ull;
            const auto start = clock64();
            while (ptx::ld_acq_sys(&segment->state) !=
                   static_cast<uint32_t>(
                       layout::SM90CombineRingSegmentState::Ready)) {
                DG_TRAP_ONLY_DEVICE_ASSERT(
                    clock64() - start < kTimeoutCycles);
            }
            return segment->physical_base_row;
        }
        return 0;
    }

    CUTLASS_DEVICE void record_completion(
        const uint32_t& local_expert_idx,
        const uint32_t& dst_rank_idx,
        const uint64_t& completion_target) const {
        if constexpr (kEnabled) {
            DG_DEVICE_ASSERT(local_expert_idx < kNumExpertsPerRank);
            DG_DEVICE_ASSERT(dst_rank_idx < kNumRanks);
            DG_DEVICE_ASSERT(completion_target != 0);
            ptx::st_release_sys(
                workspace.get_combine_ring_completion_target_ptr(
                    local_expert_idx, dst_rank_idx),
                completion_target);
        }
    }

    CUTLASS_DEVICE void mark_published(
        const uint32_t& local_expert_idx,
        const uint64_t& launch_epoch) const {
        if constexpr (kEnabled) {
            auto segment =
                workspace.get_combine_ring_segment_ptr(local_expert_idx);
            const uint32_t state = ptx::ld_acq_sys(&segment->state);
            if (state == static_cast<uint32_t>(
                             layout::SM90CombineRingSegmentState::Ready))
                ptx::st_release_sys(&segment->publish_epoch, launch_epoch);
            else
                DG_DEVICE_ASSERT(
                    state == static_cast<uint32_t>(
                                 layout::SM90CombineRingSegmentState::Free));
        }
    }
};

}  // namespace deep_gemm
