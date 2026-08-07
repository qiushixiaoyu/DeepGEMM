#pragma once

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

// Computation phase for the current block
enum class BlockPhase {
    None = 0,
    Linear1 = 1,
    Linear2 = 2
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K,
          typename WorkspaceT = layout::Workspace,
          bool kLazyExpertCount = false>
struct MegaMoEScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerWave > 0 and kNumExpertsPerWave <= kNumExpertsPerRank, "Invalid wave config");

    // NOTES: N block counts must be even so that 2 adjacent CTAs in a cluster
    // always land on the same m_block_idx with n_block_idx differing by 1
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL1BlockNs % 2 == 0, "L1 N block count must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL2BlockNs % 2 == 0, "L2 N block count must be even for 2-CTA cluster");

    // Arrival counts
    const WorkspaceT& workspace;
    const uint64_t* dispatch_launch_epoch_ptr;

    // Scheduler state
    BlockPhase next_phase = BlockPhase::Linear1;

    // Current expert and block indices
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    // Per-expert token counts.  The eager scheduler fills the complete array
    // before scheduling starts; the lazy scheduler fills entries on demand.
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t stored_num_tokens_per_expert_valid[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit MegaMoEScheduler(
        const WorkspaceT& workspace, const uint64_t* dispatch_launch_epoch_ptr = nullptr):
        workspace(workspace), dispatch_launch_epoch_ptr(dispatch_launch_epoch_ptr) {
        block_idx = blockIdx.x;
    }

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        // Align up to wave boundary, clamped for the last partial wave
        const auto aligned = math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
        return cute::min(aligned, kNumExpertsPerRank);
    }

    // Wait until all source ranks have published the count for one local
    // expert, then return its aggregate token count.  In expert-ready mode,
    // each source's route entries and final epoch/count marker share one RC
    // QP, so observing the marker also orders the corresponding route slots.
    CUTLASS_DEVICE uint32_t wait_expert_recv_count(const uint32_t& expert_idx) const {
        uint64_t value = 0;
#ifdef DG_MEGA_MOE_INTERNODE
        // Keep a bounded wait so a missing RDMA write fails instead of hanging forever.
        constexpr int64_t kSlotTimeoutCycles = 60ll * 2000000000ll;
        const auto start_clock = clock64();
        uint32_t expected_marker = kNumSMs;
        if (dispatch_launch_epoch_ptr != nullptr) {
            expected_marker = static_cast<uint32_t>(
                ptx::ld_acq_sys(dispatch_launch_epoch_ptr));
            while (expected_marker == 0) {
                DG_TRAP_ONLY_DEVICE_ASSERT(clock64() - start_clock < kSlotTimeoutCycles);
                expected_marker = static_cast<uint32_t>(
                    ptx::ld_acq_sys(dispatch_launch_epoch_ptr));
            }
        }
        for (uint32_t src = 0; src < kNumRanks; ++ src) {
            const auto slot_ptr = workspace.get_expert_recv_count_ptr(src, expert_idx);
            uint64_t slot_value = ptx::ld_acq_sys(slot_ptr);
            while (static_cast<uint32_t>(slot_value >> 32) != expected_marker) {
                DG_TRAP_ONLY_DEVICE_ASSERT(clock64() - start_clock < kSlotTimeoutCycles);
                if (dispatch_launch_epoch_ptr != nullptr)
                    expected_marker = static_cast<uint32_t>(
                        ptx::ld_acq_sys(dispatch_launch_epoch_ptr));
                slot_value = ptx::ld_acq_sys(slot_ptr);
            }
            value += slot_value & 0xffffffffull;
        }
#else
        do {
            value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
        } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
#endif
        return static_cast<uint32_t>(value);
    }

    // A single producer warp aggregates one expert at a time and publishes it
    // into the otherwise-unused inter-node recv-count-sum slots.  Lanes split
    // source ranks, so an expert's source slots are polled in parallel instead
    // of being serialized by one owner lane.  The epoch is the cache lifetime:
    // consumers ignore stale entries from previous launches.
    CUTLASS_DEVICE void publish_expert_recv_counts() const {
#ifdef DG_MEGA_MOE_INTERNODE
        constexpr int64_t kSlotTimeoutCycles = 60ll * 2000000000ll;
        const auto start_clock = clock64();
        uint32_t expected_marker = static_cast<uint32_t>(
            ptx::ld_acq_sys(dispatch_launch_epoch_ptr));
        while (expected_marker == 0) {
            DG_TRAP_ONLY_DEVICE_ASSERT(clock64() - start_clock < kSlotTimeoutCycles);
            expected_marker = static_cast<uint32_t>(
                ptx::ld_acq_sys(dispatch_launch_epoch_ptr));
        }

        const auto lane_idx = ptx::get_lane_idx();
        for (uint32_t expert_idx = 0; expert_idx < kNumExpertsPerRank; ++ expert_idx) {
            uint32_t lane_count = 0;
            for (uint32_t src = lane_idx; src < kNumRanks; src += 32) {
                const auto slot_ptr = workspace.get_expert_recv_count_ptr(src, expert_idx);
                uint64_t slot_value = ptx::ld_acq_sys(slot_ptr);
                while (static_cast<uint32_t>(slot_value >> 32) != expected_marker) {
                    DG_TRAP_ONLY_DEVICE_ASSERT(clock64() - start_clock < kSlotTimeoutCycles);
                    slot_value = ptx::ld_acq_sys(slot_ptr);
                }
                lane_count += static_cast<uint32_t>(slot_value);
            }
            const auto expert_count = __reduce_add_sync(0xffffffff, lane_count);
            if (lane_idx == 0) {
                const auto published_value =
                    (static_cast<uint64_t>(expected_marker) << 32) | expert_count;
                ptx::st_release_sys(
                    workspace.get_expert_recv_count_sum_ptr(expert_idx), published_value);
            }
            __syncwarp();
        }
#endif
    }

    CUTLASS_DEVICE uint32_t wait_published_expert_recv_count(
        const uint32_t& expert_idx) const {
#ifdef DG_MEGA_MOE_INTERNODE
        constexpr int64_t kSlotTimeoutCycles = 60ll * 2000000000ll;
        const auto start_clock = clock64();
        uint32_t expected_marker = static_cast<uint32_t>(
            ptx::ld_acq_sys(dispatch_launch_epoch_ptr));
        while (expected_marker == 0) {
            DG_TRAP_ONLY_DEVICE_ASSERT(clock64() - start_clock < kSlotTimeoutCycles);
            expected_marker = static_cast<uint32_t>(
                ptx::ld_acq_sys(dispatch_launch_epoch_ptr));
        }
        const auto published_ptr = workspace.get_expert_recv_count_sum_ptr(expert_idx);
        uint64_t value = ptx::ld_acq_sys(published_ptr);
        while (static_cast<uint32_t>(value >> 32) != expected_marker) {
            DG_TRAP_ONLY_DEVICE_ASSERT(clock64() - start_clock < kSlotTimeoutCycles);
            value = ptx::ld_acq_sys(published_ptr);
        }
        return static_cast<uint32_t>(value);
#else
        return wait_expert_recv_count(expert_idx);
#endif
    }

    // Warp-collective lazy fill.  Only the lane owning `expert_idx` waits for
    // the producer's published cache entry; the other lanes wait at the warp
    // boundary and consume the value through a shuffle.
    CUTLASS_DEVICE void ensure_expert_recv_count(const uint32_t& expert_idx) {
        DG_STATIC_ASSERT(kNumExpertsPerLane > 0, "Invalid number of experts per lane");
        const auto owner_lane = expert_idx % 32;
        const auto owner_slot = expert_idx / 32;
        uint32_t valid = 0;
        if (ptx::get_lane_idx() == owner_lane)
            valid = stored_num_tokens_per_expert_valid[owner_slot];
        valid = ptx::exchange(valid, owner_lane);
        if (valid == 0) {
            if (ptx::get_lane_idx() == owner_lane) {
                stored_num_tokens_per_expert[owner_slot] =
                    wait_published_expert_recv_count(expert_idx);
                stored_num_tokens_per_expert_valid[owner_slot] = 1;
            }
            __syncwarp();
        }
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) {
        if constexpr (kLazyExpertCount)
            ensure_expert_recv_count(expert_idx);
        uint32_t valid_value;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
        if constexpr (kLazyExpertCount) {
            // The pool layout remains the fixed expert-prefix layout.  Expert
            // e can start once counts [0, e] are known; later experts are not
            // part of its placement dependency.
            for (uint32_t i = 0; i < expert_idx; ++ i)
                ensure_expert_recv_count(i);
        }
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        if (current_local_expert_idx < kNumExpertsPerRank)
            current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kNumL1BlockNs;
            if (m_block_idx < num_m_blocks)
                return true;

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                m_block_idx = block_idx / kNumL2BlockNs;
                return true;
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    // Core state machine: assigns the next block
    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (true) {
            if (current_local_expert_idx >= kNumExpertsPerRank)
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    // Found a new L1 block
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    // Jump to next block
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // L1 for the current wave is complete, transition to L2
                    next_phase = BlockPhase::Linear2;
                    set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
                }
            } else {
                if (fetch_next_l2_block()) {
                    // Found a new L2 block
                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                    // Jump to next block
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // Move to L1 of the next wave
                    next_phase = BlockPhase::Linear1;
                }
            }
        }

        // All waves and experts are fully processed
        return {BlockPhase::None, 0, 0, 0};
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            if (expert_idx < kNumExpertsPerRank) {
                stored_num_tokens_per_expert[i] = wait_expert_recv_count(expert_idx);
                stored_num_tokens_per_expert_valid[i] = 1;
            }
        }
        __syncwarp();
    }

    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        // Wait for all expert counters to be finalized
        if constexpr (not kLazyExpertCount)
            fetch_expert_recv_count();

        // Initialize current expert with 0
        set_expert_idx(0);

        // Iterate over all blocks
        // TODO: add swizzle within expert waves for better L2 cache utilization
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;

            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
    }
};

} // namespace deep_gemm::sched
