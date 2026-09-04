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

// Compact index of a REMOTE node as seen from `my_node`: nodes other than
// my_node map to [0, num_nodes - 1).  Gateway collect boxes and landing
// zones are dimensioned per remote node with this index.
CUTLASS_HOST_DEVICE constexpr uint32_t gateway_rel_node(
    const uint32_t& node_idx, const uint32_t& my_node_idx) {
    return node_idx - (node_idx > my_node_idx ? 1u : 0u);
}

// Inverse of gateway_rel_node(): turn a compact remote-node index back into
// the absolute node index used by ranks/QPs.
CUTLASS_HOST_DEVICE constexpr uint32_t gateway_abs_node(
    const uint32_t& rel_node_idx, const uint32_t& my_node_idx) {
    return rel_node_idx + (rel_node_idx >= my_node_idx ? 1u : 0u);
}
static constexpr int kLCMCandidateBlockM = 384;
// Protocol policy uses the caller's pre-alignment capacity.  The physical
// workspace remains aligned to kLCMCandidateBlockM.
static constexpr int kGatewayDenseMaxRequestedTokens = 256;

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

// SM90 full-row combine staging uses full-pool capacity for decode-shaped
// instances, then transitions monotonically to a quarter-pool target at an
// 8192-token configuration.  The interpolation is performed in pool-row space
// (not ratio space), so increasing the configured token capacity can never
// reduce the number of allocated staging rows.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sm90_combine_ring_tokens(
    T num_ranks, T num_max_tokens_per_rank, T num_topk,
    T num_experts_per_rank) {
    const T alignment = static_cast<T>(kLCMCandidateBlockM);
    const T low_tokens = math::constexpr_align(static_cast<T>(1024), alignment);
    const T high_tokens = math::constexpr_align(static_cast<T>(8192), alignment);
    const T full = get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk,
        num_experts_per_rank);
    if (num_max_tokens_per_rank <= low_tokens)
        return full;

    const auto target_capacity = [=](const T& tokens, const T& pool) {
        const T quarter = math::constexpr_ceil_div(pool, static_cast<T>(4));
        const T single_expert = num_ranks * tokens;
        return math::constexpr_align(
            quarter > single_expert ? quarter : single_expert, alignment);
    };

    const T low_full = get_num_max_pool_tokens(
        num_ranks, low_tokens, num_topk, num_experts_per_rank);
    const T high_full = get_num_max_pool_tokens(
        num_ranks, high_tokens, num_topk, num_experts_per_rank);
    const T high_target_raw = target_capacity(high_tokens, high_full);
    const T high_target = high_target_raw > low_full ?
        high_target_raw : low_full;

    T capacity;
    if (num_max_tokens_per_rank < high_tokens) {
        const uint64_t numerator =
            static_cast<uint64_t>(full - low_full) *
            static_cast<uint64_t>(high_target - low_full);
        const uint64_t denominator =
            static_cast<uint64_t>(high_full - low_full);
        const T interpolated = low_full + static_cast<T>(
            math::constexpr_ceil_div(numerator, denominator));
        const T minimum = target_capacity(num_max_tokens_per_rank, full);
        capacity = interpolated > minimum ? interpolated : minimum;
        capacity = capacity > low_full ? capacity : low_full;
    } else {
        const T minimum = target_capacity(num_max_tokens_per_rank, full);
        capacity = minimum > high_target ? minimum : high_target;
    }

    capacity = math::constexpr_align(capacity, alignment);
    return capacity < full ? capacity : full;
}

// L1/L2 payload pools use the same monotonic capacity curve as combine.  The
// data-lifetime protocol differs (GPU full/empty instead of NIC CQ), but the
// worst-case row demand and the one-expert safety floor are identical.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sm90_compute_ring_tokens(
    T num_ranks, T num_max_tokens_per_rank, T num_topk,
    T num_experts_per_rank) {
    return get_num_sm90_combine_ring_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk,
        num_experts_per_rank);
}

// SM90 FP8 and FP4 select BLOCK_M={64,128}.  SF rows are padded to 128 rows per
// M block, so sizing for BLOCK_M=64 (two SF rows per payload row) covers both
// recipes.  This intentionally does not inherit the common candidate list,
// whose BLOCK_M=8 entry made the old full-pool SF storage 16x too large.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sm90_compute_sf_ring_tokens(
    T num_ring_tokens) {
    return num_ring_tokens * static_cast<T>(2);
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

enum class SM90CombineRingSegmentState : uint32_t {
    Free = 0,
    Allocating = 1,
    Ready = 2,
    Retired = 3,
};

struct alignas(16) SM90CombineRingControl {
    uint64_t init_epoch;
    uint64_t prealloc_epoch;
    uint64_t reclaimed_epoch;
    uint64_t alloc_head_ticket;
    uint64_t reclaim_tail_ticket;
    uint32_t allocator_lock;
    uint32_t queue_head;
    uint32_t queue_tail;
    uint32_t prealloc_success;
};

struct alignas(16) SM90CombineRingSegment {
    uint64_t reclaim_begin_ticket;
    uint64_t end_ticket;
    uint64_t publish_epoch;
    uint32_t physical_base_row;
    uint32_t num_rows;
    uint32_t state;
    uint32_t has_internode_rows;
};

// Fixed prefix of one packed dispatch-metadata slot.  The offset table
// immediately follows this header; the payload starts at a fixed aligned
// offset and the publish-manifest follows the live payload.  Manifest rows
// retain the existing (epoch << 32 | count) representation.
struct alignas(16) SM90GatewayPackedHeader {
    uint64_t epoch;
    uint32_t total_entries;
    uint32_t num_cells;
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
    uint32_t num_topk;
    // NVLink domains spanned by the ranks (1 on single node).  Gateway
    // collect boxes / landing zones hold one bank per REMOTE node.
    uint32_t num_nodes;

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
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_topk(num_topk) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
        num_nodes = num_ranks > kGatewayNvlPeers
            ? num_ranks / kGatewayNvlPeers : 1u;
    }


    CUTLASS_HOST_DEVICE
    uint64_t get_gateway_num_cells() const {
        return static_cast<uint64_t>(kGatewayNvlPeers) *
            num_experts_per_rank;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_gateway_max_packed_entries() const {
        return static_cast<uint64_t>(kGatewayNvlPeers) *
            num_max_tokens_per_rank *
            math::constexpr_min(num_topk, num_experts_per_rank);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_gateway_packed_payload_offset_bytes() const {
        return math::align<uint64_t>(
            sizeof(SM90GatewayPackedHeader) +
                (get_gateway_num_cells() + 1) * sizeof(uint32_t),
            16);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_gateway_packed_manifest_offset_bytes() const {
        return math::align<uint64_t>(
            get_gateway_packed_payload_offset_bytes() +
                get_gateway_max_packed_entries() * sizeof(uint32_t),
            16);
    }

    // Dense wire-format manifest follows the LIVE payload rather than the
    // maximum-capacity payload.  It therefore remains the publish marker
    // while fitting in one contiguous RDMA WRITE.  Sparse shapes retain the
    // fixed-address accessor above to avoid a receiver-side header dependency.
    CUTLASS_HOST_DEVICE
    uint64_t get_gateway_packed_compact_manifest_offset_bytes(
        const uint64_t& total_entries) const {
        return math::align<uint64_t>(
            get_gateway_packed_payload_offset_bytes() +
                total_entries * sizeof(uint32_t),
            16);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_gateway_packed_slot_bytes() const {
        return math::align<uint64_t>(
            get_gateway_packed_manifest_offset_bytes() +
                get_gateway_num_cells() * sizeof(uint64_t),
            16);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_gateway_dense_slot_bytes() const {
        return get_gateway_num_cells() * num_max_tokens_per_rank *
            sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;

        // One launch epoch is shared by dispatch metadata, combine readiness,
        // and ring lifetime.  SM 0 advances it once and publishes the latched
        // value to the whole local grid before any protocol state is touched.
        num_bytes += sizeof(uint64_t);
        // Sidecar launch control: armed epoch plus a reusable CTA-arrival
        // counter.  Keeping both in the common layout makes sidecar and fused
        // specializations agree on every following workspace offset.
        num_bytes += 2 * sizeof(uint64_t);
        // Per-expert combine completion protocol.  Ready epochs are indexed
        // by global expert, while publication counters and destination masks
        // are owned by this rank's local experts.
        num_bytes += num_experts * sizeof(uint64_t);
        num_bytes += math::align(num_experts_per_rank, 2u) * sizeof(uint32_t);
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        num_bytes += num_experts * sizeof(uint64_t) * 2;
        num_bytes += num_experts_per_rank * sizeof(uint64_t);
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);

        // Reusable L1/L2 payload slots keep readiness in the logical
        // full-pool counters above.  Only reuse safety is physical: one empty
        // counter per possible BLOCK_M=64 slot for each pool.  Larger recipes
        // use the prefix of these arrays.
        const uint64_t num_sm90_ring_blocks =
            num_max_pool_tokens / static_cast<uint32_t>(64);
        num_bytes += num_sm90_ring_blocks * sizeof(uint32_t);
        num_bytes += num_sm90_ring_blocks * sizeof(uint32_t);

        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Lifecycle-protected expert-segment ring for full-row combine
        // staging.  Payload rows live outside the workspace; this compact
        // control plane is always present so host sizing and FP8/FP4 kernel
        // slicing remain identical across protocol A/B variants.
        num_bytes += sizeof(SM90CombineRingControl);
        num_bytes += num_experts_per_rank * sizeof(SM90CombineRingSegment);
        num_bytes += math::align<uint64_t>(
            num_experts_per_rank * sizeof(uint32_t), 8);
        num_bytes += static_cast<uint64_t>(num_experts_per_rank) *
            num_ranks * sizeof(uint64_t);
        num_bytes = math::align<uint64_t>(num_bytes, 16);

        // Gateway control/data plane.  Every producer and receiver buffer is
        // double-buffered by dispatch epoch so payload AND count/manifest have
        // the same lifetime.  Both local collect-slot and remote packed-
        // landing reuse are protected by the all-source metadata epoch
        // dependency.  Completion tickets separately protect the local HBM
        // send source until the NIC has consumed it.
        const uint64_t num_remote_nodes = num_nodes - 1;
        constexpr uint64_t kNumEpochSlots = 2;
        num_bytes += num_remote_nodes * kNumEpochSlots * kGatewayNvlPeers *
            num_experts_per_rank * num_max_tokens_per_rank * sizeof(uint32_t);
        num_bytes += num_remote_nodes * kNumEpochSlots * kGatewayNvlPeers *
            num_experts_per_rank * sizeof(uint64_t);
        num_bytes += num_remote_nodes * kNumEpochSlots *
            kGatewayNvlPeers * sizeof(uint64_t);
        // Eager handshake: per-destination-rank CTA completion counters
        // (local, zeroed by dispatch cleanup each launch).
        num_bytes += num_ranks * sizeof(uint32_t);

        // Packed send/landing slots: [relative node][epoch slot].
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        num_bytes += num_remote_nodes * kNumEpochSlots *
            get_gateway_packed_slot_bytes();
        // Exact producer indices returned by the grouped WRITE helper.
        num_bytes += num_remote_nodes * kNumEpochSlots * sizeof(uint64_t);
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        num_bytes += num_remote_nodes * kNumEpochSlots *
            get_gateway_packed_slot_bytes();
        // Independent dense-V3 landing mirror.  It is intentionally separate
        // from both the outgoing collect box and packed landing slots: the
        // former can be populated concurrently for the opposite direction,
        // while packed capacity only covers live top-k entries rather than
        // every dense (source, expert, token-slot) cell.
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        num_bytes += num_remote_nodes * kNumEpochSlots *
            get_gateway_dense_slot_bytes();
        // Same-node per-source counts need the same two-epoch lifetime as the
        // remote packed manifest.  A faster local rank may enter e+1 while a
        // peer is still consuming e; the legacy single slot cannot represent
        // both markers simultaneously.
        num_bytes += kNumEpochSlots * num_ranks *
            num_experts_per_rank * sizeof(uint64_t);
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
    uint64_t* get_launch_epoch_ptr() const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes);
    }

    // The second 64-bit launch-control word is reserved for the optional
    // SM90 FP4 sidecar publisher.  The sidecar publishes the epoch it is
    // armed for before the fused kernel advances get_launch_epoch_ptr().
    // This closes the cross-stream race where the fused kernel could produce
    // and clean a short launch before the sidecar had become resident.
    CUTLASS_DEVICE
    uint64_t* get_sidecar_publisher_armed_epoch_ptr() const {
        return get_launch_epoch_ptr() + 1;
    }

    CUTLASS_DEVICE
    uint32_t* get_sidecar_publisher_arrival_count_ptr() const {
        return reinterpret_cast<uint32_t*>(get_launch_epoch_ptr() + 2);
    }

    CUTLASS_DEVICE
    uint64_t* get_combine_ready_epoch_ptr(const uint32_t& global_expert_idx = 0) const {
        return get_launch_epoch_ptr() + 3 + global_expert_idx;
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
    uint32_t* get_l1_ring_empty_count_ptr(
        const uint32_t& ring_block_idx = 0) const {
        return reinterpret_cast<uint32_t*>(
            get_l2_arrival_mask_ptr(num_max_pool_blocks)) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_ring_empty_count_ptr(
        const uint32_t& ring_block_idx = 0) const {
        const uint32_t num_sm90_ring_blocks = num_max_pool_tokens / 64u;
        return get_l1_ring_empty_count_ptr(num_sm90_ring_blocks) +
            ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const uint32_t num_sm90_ring_blocks = num_max_pool_tokens / 64u;
        const auto base = get_l2_ring_empty_count_ptr(
            num_sm90_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }

    CUTLASS_DEVICE
    SM90CombineRingControl* get_combine_ring_control_ptr() const {
        return reinterpret_cast<SM90CombineRingControl*>(
            get_token_src_metadata_ptr(num_max_pool_tokens));
    }

    CUTLASS_DEVICE
    SM90CombineRingSegment* get_combine_ring_segment_ptr(
        const uint32_t& local_expert_idx = 0) const {
        return reinterpret_cast<SM90CombineRingSegment*>(
            get_combine_ring_control_ptr() + 1) + local_expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_combine_ring_queue_ptr(
        const uint32_t& queue_idx = 0) const {
        return reinterpret_cast<uint32_t*>(
            get_combine_ring_segment_ptr(num_experts_per_rank)) + queue_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_combine_ring_completion_target_ptr(
        const uint32_t& local_expert_idx = 0,
        const uint32_t& dst_rank_idx = 0) const {
        const auto target_base = reinterpret_cast<uint64_t*>(math::align(
            reinterpret_cast<uint64_t>(
                get_combine_ring_queue_ptr(num_experts_per_rank)),
            static_cast<uint64_t>(8)));
        return target_base +
            static_cast<uint64_t>(local_expert_idx) * num_ranks +
            dst_rank_idx;
    }

    // Gateway collect box accessors (see get_num_bytes for the layout).
    // Cell capacity is num_max_tokens_per_rank: one source rank can route
    // at most its own token count to a single expert (top-k never repeats
    // an expert within a token).
    CUTLASS_DEVICE
    uint32_t* get_gateway_entry_base_ptr() const {
        return reinterpret_cast<uint32_t*>(math::align(
            reinterpret_cast<uint64_t>(
                get_combine_ring_completion_target_ptr(
                    num_experts_per_rank, 0)),
            static_cast<uint64_t>(16)));
    }

    // Collect-box entries: one bank per REMOTE destination node and epoch
    // slot, then [src nvl][expert][token].
    CUTLASS_DEVICE
    uint32_t* get_gateway_entry_ptr(const uint32_t& src_nvl_idx = 0,
                                    const uint32_t& expert_idx = 0,
                                    const uint32_t& token_idx = 0,
                                    const uint32_t& rel_dst_node = 0,
                                    const uint32_t& epoch_slot = 0) const {
        return get_gateway_entry_base_ptr() +
            ((((static_cast<uint64_t>(rel_dst_node) * 2 + epoch_slot) *
                    uint32_t(kGatewayNvlPeers) + src_nvl_idx) *
                num_experts_per_rank + expert_idx) *
             num_max_tokens_per_rank + token_idx);
    }

    CUTLASS_DEVICE
    uint64_t* get_gateway_manifest_base_ptr() const {
        // Past every entry bank.
        return reinterpret_cast<uint64_t*>(
            get_gateway_entry_base_ptr() +
            static_cast<uint64_t>(num_nodes - 1) * 2 *
                uint32_t(kGatewayNvlPeers) * num_experts_per_rank *
                num_max_tokens_per_rank);
    }

    // Manifest rows: one bank per REMOTE destination node, then
    // [src nvl][expert] of (epoch << 32 | count).
    CUTLASS_DEVICE
    uint64_t* get_gateway_manifest_ptr(const uint32_t& src_nvl_idx = 0,
                                       const uint32_t& expert_idx = 0,
                                       const uint32_t& rel_dst_node = 0,
                                       const uint32_t& epoch_slot = 0) const {
        return get_gateway_manifest_base_ptr() +
            (((static_cast<uint64_t>(rel_dst_node) * 2 + epoch_slot) *
                  uint32_t(kGatewayNvlPeers) + src_nvl_idx) *
             num_experts_per_rank + expert_idx);
    }

    // One flag per destination-node/epoch-slot/local-source tuple.
    CUTLASS_DEVICE
    uint64_t* get_gateway_flag_ptr(const uint32_t& src_nvl_idx = 0,
                                   const uint32_t& rel_dst_node = 0,
                                   const uint32_t& epoch_slot = 0) const {
        return get_gateway_manifest_base_ptr() +
            static_cast<uint64_t>(num_nodes - 1) * 2 *
                uint32_t(kGatewayNvlPeers) * num_experts_per_rank +
            (static_cast<uint64_t>(rel_dst_node) * 2 + epoch_slot) *
                uint32_t(kGatewayNvlPeers) + src_nvl_idx;
    }

    // Eager handshake: counts metadata CTAs that finished writing route
    // entries for one destination rank.  Reaching the producer count implies
    // every relevant CTA passed stake-out and fenced its stores, so the
    // counter's final adder is the direction's trigger.  Local memory, reset
    // by dispatch cleanup.
    CUTLASS_DEVICE
    uint32_t* get_gateway_direction_done_ptr(
        const uint32_t& dst_rank_idx = 0) const {
        return reinterpret_cast<uint32_t*>(
            get_gateway_flag_ptr(
                0, num_nodes - 1, 0)) +
            dst_rank_idx;
    }

    CUTLASS_DEVICE
    uint8_t* get_gateway_packed_send_base_ptr() const {
        return reinterpret_cast<uint8_t*>(math::align(
            reinterpret_cast<uint64_t>(
                get_gateway_direction_done_ptr(num_ranks)),
            static_cast<uint64_t>(16)));
    }

    CUTLASS_DEVICE
    uint8_t* get_gateway_packed_send_slot_ptr(
        const uint32_t& rel_dst_node = 0,
        const uint32_t& epoch_slot = 0) const {
        return get_gateway_packed_send_base_ptr() +
            (static_cast<uint64_t>(rel_dst_node) * 2 + epoch_slot) *
                get_gateway_packed_slot_bytes();
    }

    CUTLASS_DEVICE
    uint64_t* get_gateway_send_completion_ptr(
        const uint32_t& rel_dst_node = 0,
        const uint32_t& epoch_slot = 0) const {
        return reinterpret_cast<uint64_t*>(
            get_gateway_packed_send_base_ptr() +
            static_cast<uint64_t>(num_nodes - 1) * 2 *
                get_gateway_packed_slot_bytes()) +
            static_cast<uint64_t>(rel_dst_node) * 2 + epoch_slot;
    }

    CUTLASS_DEVICE
    uint8_t* get_gateway_packed_landing_base_ptr() const {
        return reinterpret_cast<uint8_t*>(math::align(
            reinterpret_cast<uint64_t>(
                get_gateway_send_completion_ptr(num_nodes - 1, 0)),
            static_cast<uint64_t>(16)));
    }

    CUTLASS_DEVICE
    uint8_t* get_gateway_packed_landing_slot_ptr(
        const uint32_t& rel_src_node = 0,
        const uint32_t& epoch_slot = 0) const {
        return get_gateway_packed_landing_base_ptr() +
            (static_cast<uint64_t>(rel_src_node) * 2 + epoch_slot) *
                get_gateway_packed_slot_bytes();
    }

    CUTLASS_DEVICE
    SM90GatewayPackedHeader* get_gateway_packed_header_ptr(
        const bool& landing, const uint32_t& rel_node,
        const uint32_t& epoch_slot) const {
        return reinterpret_cast<SM90GatewayPackedHeader*>(
            landing ? get_gateway_packed_landing_slot_ptr(rel_node, epoch_slot)
                    : get_gateway_packed_send_slot_ptr(rel_node, epoch_slot));
    }

    CUTLASS_DEVICE
    uint32_t* get_gateway_packed_offset_ptr(
        const bool& landing, const uint32_t& rel_node,
        const uint32_t& epoch_slot, const uint32_t& cell_idx = 0) const {
        return reinterpret_cast<uint32_t*>(
            get_gateway_packed_header_ptr(landing, rel_node, epoch_slot) + 1) +
            cell_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_gateway_packed_payload_ptr(
        const bool& landing, const uint32_t& rel_node,
        const uint32_t& epoch_slot, const uint32_t& entry_idx = 0) const {
        auto slot = landing
            ? get_gateway_packed_landing_slot_ptr(rel_node, epoch_slot)
            : get_gateway_packed_send_slot_ptr(rel_node, epoch_slot);
        return reinterpret_cast<uint32_t*>(
            slot + get_gateway_packed_payload_offset_bytes()) + entry_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_gateway_packed_manifest_ptr(
        const bool& landing, const uint32_t& rel_node,
        const uint32_t& epoch_slot, const uint32_t& cell_idx = 0) const {
        auto slot = landing
            ? get_gateway_packed_landing_slot_ptr(rel_node, epoch_slot)
            : get_gateway_packed_send_slot_ptr(rel_node, epoch_slot);
        return reinterpret_cast<uint64_t*>(
            slot + get_gateway_packed_manifest_offset_bytes()) + cell_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_gateway_packed_compact_manifest_ptr(
        const bool& landing, const uint32_t& rel_node,
        const uint32_t& epoch_slot, const uint64_t& total_entries,
        const uint32_t& cell_idx = 0) const {
        auto slot = landing
            ? get_gateway_packed_landing_slot_ptr(rel_node, epoch_slot)
            : get_gateway_packed_send_slot_ptr(rel_node, epoch_slot);
        return reinterpret_cast<uint64_t*>(
            slot + get_gateway_packed_compact_manifest_offset_bytes(
                       total_entries)) + cell_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_gateway_packed_landing_entry_base_ptr(
        const uint32_t& rel_src_node, const uint32_t& epoch_slot,
        const uint32_t& src_nvl_idx, const uint32_t& expert_idx) const {
        const uint32_t cell_idx =
            src_nvl_idx * num_experts_per_rank + expert_idx;
        const uint32_t offset = *get_gateway_packed_offset_ptr(
            true, rel_src_node, epoch_slot, cell_idx);
        return get_gateway_packed_payload_ptr(
            true, rel_src_node, epoch_slot, offset);
    }

    CUTLASS_DEVICE
    uint32_t* get_gateway_packed_landing_entry_ptr(
        const uint32_t& rel_src_node, const uint32_t& epoch_slot,
        const uint32_t& src_nvl_idx, const uint32_t& expert_idx,
        const uint32_t& token_idx) const {
        return get_gateway_packed_landing_entry_base_ptr(
            rel_src_node, epoch_slot, src_nvl_idx, expert_idx) + token_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_gateway_dense_landing_base_ptr() const {
        return reinterpret_cast<uint32_t*>(math::align(
            reinterpret_cast<uint64_t>(
                get_gateway_packed_landing_base_ptr() +
                static_cast<uint64_t>(num_nodes - 1) * 2 *
                    get_gateway_packed_slot_bytes()),
            static_cast<uint64_t>(16)));
    }

    CUTLASS_DEVICE
    uint32_t* get_gateway_dense_landing_slot_ptr(
        const uint32_t& rel_src_node = 0,
        const uint32_t& epoch_slot = 0) const {
        return reinterpret_cast<uint32_t*>(
            reinterpret_cast<uint8_t*>(get_gateway_dense_landing_base_ptr()) +
            (static_cast<uint64_t>(rel_src_node) * 2 + epoch_slot) *
                get_gateway_dense_slot_bytes());
    }

    CUTLASS_DEVICE
    uint32_t* get_gateway_dense_landing_entry_ptr(
        const uint32_t& rel_src_node, const uint32_t& epoch_slot,
        const uint32_t& src_nvl_idx, const uint32_t& expert_idx,
        const uint32_t& token_idx) const {
        return get_gateway_dense_landing_slot_ptr(rel_src_node, epoch_slot) +
            (static_cast<uint64_t>(src_nvl_idx) * num_experts_per_rank +
             expert_idx) * num_max_tokens_per_rank + token_idx;
    }

    // Double-buffered same-node count table: [epoch slot][source rank][local
    // expert].  Remote-node sources continue to use the packed landing
    // manifest; this table closes the equivalent lifetime hole on NVLink.
    CUTLASS_DEVICE
    uint64_t* get_dispatch_epoch_count_ptr(
        const uint32_t& epoch_slot = 0, const uint32_t& rank_idx = 0,
        const uint32_t& expert_idx = 0) const {
        return reinterpret_cast<uint64_t*>(
            reinterpret_cast<uint8_t*>(get_gateway_dense_landing_base_ptr()) +
            static_cast<uint64_t>(num_nodes - 1) * 2 *
                get_gateway_dense_slot_bytes()) +
            (static_cast<uint64_t>(epoch_slot) * num_ranks + rank_idx) *
                num_experts_per_rank + expert_idx;
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
// Phase slots include absolute globaltimer stamps plus an L1/L2 split of the
// math mainloop and its input-arrival wait.  The latter is diagnostic-only and
// lets a matched no-RDMA/RDMA A/B distinguish WGMMA/codegen regression from
// delayed TMA input arrival.
static constexpr uint32_t kSM90MegaMoEProfileSlots = 40;

} // namespace deep_gemm::layout
