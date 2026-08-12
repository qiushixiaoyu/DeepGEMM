#pragma once

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

static constexpr int kNumCandidateBlockMs = 7;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
static constexpr int kMaxCandidateBlockM = 192;
static constexpr int kMinCandidateBlockM = 8;

// Matches DG_MEGA_MOE_NVL_PEERS (8 GPUs per NVLink domain); namespace scope
// so device code can bind it to const-reference parameters.
static constexpr uint32_t kGatewayNvlPeers = 8;
static constexpr int kLCMCandidateBlockM = 384;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sf_ring_tokens(T num_ring_tokens, T block_m) {
    return (num_ring_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Ring-buffer capacity used by reusable token/data buffers
    uint32_t num_ring_tokens;
    uint32_t num_ring_blocks;

    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;

    // For both grid barrier and NVLink barrier
    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_ring_tokens(num_ring_tokens) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_ring_blocks = num_ring_tokens / kMinCandidateBlockM;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 full token count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L1 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 full block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices (full)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
    // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l2_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        return get_l1_full_count_ptr(pool_block_idx);
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        return reinterpret_cast<uint64_t*>(get_l2_full_count_ptr()) + pool_block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages (full)
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

// SM90 MegaMoE predates the reusable ring workspace used by the SM100 kernels.
// Keep its compact layout explicit so Hopper codegen and buffer slicing stay
// identical to the tuned SM90 implementation while SM100 can use Workspace.
struct SM90Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    SM90Workspace(void* base,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;

        // Dispatch completion protocol.  Per-source expert slots pack the
        // low 32-bit token count with the low 32 bits of this launch epoch.
        num_bytes += sizeof(uint64_t);

        // Per-expert combine completion protocol.  The launch epoch is local
        // to each rank and advances once per collective invocation.  Ready
        // epochs are indexed by global expert, while publication counters and
        // destination masks are owned by this rank's local experts.
        num_bytes += sizeof(uint64_t);
        num_bytes += num_experts * sizeof(uint64_t);
        num_bytes += math::align(num_experts_per_rank, 2u) * sizeof(uint32_t);
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        num_bytes += num_experts * sizeof(uint64_t) * 2;
        num_bytes += num_experts_per_rank * sizeof(uint64_t);
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Gateway collect box for the two-level dispatch handshake: local
        // NVLink peers stage their inter-node route entries and manifest
        // rows here, and this rank forwards them over one same-rail QP.
        //   entries:  [src nvl peer][local expert][token slot] u32
        //   manifest: [src nvl peer][local expert] u64 (epoch<<32 | count)
        //   flags:    [src nvl peer] u64 launch-epoch flags
        num_bytes += kGatewayNvlPeers * num_experts_per_rank *
            num_max_tokens_per_rank * sizeof(uint32_t);
        num_bytes += kGatewayNvlPeers * num_experts_per_rank * sizeof(uint64_t);
        num_bytes += kGatewayNvlPeers * sizeof(uint64_t);
        // Eager handshake: per-destination-rank CTA completion counters
        // (local, zeroed by dispatch cleanup each launch).
        num_bytes += num_ranks * sizeof(uint32_t);

        // V3 landing zone: a byte-for-byte mirror of the remote gateway's
        // collect-box entry area, so the gateway ships all entries as one
        // contiguous WRITE.  Pull reads inter-node entries from here.
        num_bytes += kGatewayNvlPeers * num_experts_per_rank *
            num_max_tokens_per_rank * sizeof(uint32_t);
        return math::align<uint64_t>(num_bytes, 16);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        return math::advance_ptr<int>(
            base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint64_t* get_dispatch_launch_epoch_ptr() const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes);
    }

    CUTLASS_DEVICE
    uint64_t* get_combine_launch_epoch_ptr() const {
        return get_dispatch_launch_epoch_ptr() + 1;
    }

    CUTLASS_DEVICE
    uint64_t* get_combine_ready_epoch_ptr(const uint32_t& global_expert_idx = 0) const {
        return get_combine_launch_epoch_ptr() + 1 + global_expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_combine_posted_block_count_ptr(const uint32_t& local_expert_idx = 0) const {
        const auto base = get_combine_ready_epoch_ptr(num_experts);
        return reinterpret_cast<uint32_t*>(base) + local_expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_combine_dst_rank_mask_ptr(const uint32_t& local_expert_idx = 0) const {
        const auto base = get_combine_posted_block_count_ptr(
            math::align(num_experts_per_rank, 2u));
        return reinterpret_cast<uint64_t*>(base) + local_expert_idx;
    }

    // Async pair-publisher aliases.  The inline scatter path uses these rows
    // as posted-M-block counters and destination masks; the async path does
    // not, so it reuses the same storage for the number of completed
    // (expert, destination-rank) tasks and the launch epoch that releases the
    // expert to overlapping cleanup.  Keeping aliases here makes the two
    // compile-time protocols explicit without growing SymmBuffer.
    CUTLASS_DEVICE
    uint32_t* get_combine_publish_pair_done_count_ptr(
        const uint32_t& local_expert_idx = 0) const {
        return get_combine_posted_block_count_ptr(local_expert_idx);
    }

    CUTLASS_DEVICE
    uint64_t* get_combine_publish_done_epoch_ptr(
        const uint32_t& local_expert_idx = 0) const {
        return get_combine_dst_rank_mask_ptr(local_expert_idx);
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return get_combine_dst_rank_mask_ptr(num_experts_per_rank) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }

    // Gateway collect box accessors (see get_num_bytes for the layout).
    // Cell capacity is num_max_tokens_per_rank: one source rank can route
    // at most its own token count to a single expert (top-k never repeats
    // an expert within a token).
    CUTLASS_DEVICE
    uint32_t* get_gateway_entry_ptr(const uint32_t& src_nvl_idx = 0,
                                    const uint32_t& expert_idx = 0,
                                    const uint32_t& token_idx = 0) const {
        const auto base = reinterpret_cast<uint32_t*>(
            get_token_src_metadata_ptr(num_max_pool_tokens));
        return base +
            (src_nvl_idx * num_experts_per_rank + expert_idx) *
                num_max_tokens_per_rank + token_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_gateway_manifest_ptr(const uint32_t& src_nvl_idx = 0,
                                       const uint32_t& expert_idx = 0) const {
        // Pass prvalue copies: binding the namespace constexpr directly to
        // a const reference would ODR-use it in device code.
        const auto base = reinterpret_cast<uint64_t*>(
            get_gateway_entry_ptr(uint32_t(kGatewayNvlPeers)));
        return base + src_nvl_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_gateway_flag_ptr(const uint32_t& src_nvl_idx = 0) const {
        return get_gateway_manifest_ptr(uint32_t(kGatewayNvlPeers)) +
            src_nvl_idx;
    }

    // Eager handshake: counts CTAs that finished writing route entries for
    // one destination rank.  Reaching kNumSMs implies every CTA passed
    // stake-out and fenced its stores, so the counter's final adder is the
    // direction's trigger.  Local memory, reset by dispatch cleanup.
    CUTLASS_DEVICE
    uint32_t* get_gateway_direction_done_ptr(
        const uint32_t& dst_rank_idx = 0) const {
        return reinterpret_cast<uint32_t*>(
            get_gateway_flag_ptr(uint32_t(kGatewayNvlPeers))) +
            dst_rank_idx;
    }

    // V3 landing zone (same [src nvl][expert][slot] shape as the collect
    // box).  Written by the remote same-rail gateway as one bulk WRITE;
    // pull reads inter-node route entries from here instead of the inbox.
    CUTLASS_DEVICE
    uint32_t* get_gateway_landing_ptr(const uint32_t& src_nvl_idx = 0,
                                      const uint32_t& expert_idx = 0,
                                      const uint32_t& token_idx = 0) const {
        const auto base = get_gateway_direction_done_ptr(num_ranks);
        return base +
            (src_nvl_idx * num_experts_per_rank + expert_idx) *
                num_max_tokens_per_rank + token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& num_max_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

// Optional SM90 phase-profiler storage. The allocation is always present on
// the diagnostic branch, so toggling the JIT-only profiler does not change the
// public API or any tensor slice.
static constexpr uint32_t kSM90MegaMoEProfileMaxSMs = 256;
// 31 phase slots + 3 absolute globaltimer stamps (kernel entry, counts sent,
// count barrier released) used to separate launch skew from protocol cost.
static constexpr uint32_t kSM90MegaMoEProfileSlots = 37;

} // namespace deep_gemm::layout
