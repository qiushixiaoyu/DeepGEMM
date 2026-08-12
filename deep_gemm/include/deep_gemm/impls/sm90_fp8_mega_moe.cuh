#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#define __CLION_IDE__

namespace deep_gemm {

template <float kActivationClamp>
__forceinline__ __device__ float sm90_fp8_mega_moe_clamp_gate(float x) {
    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
        x = cute::min(x, kActivationClamp);
    return x;
}

template <float kActivationClamp>
__forceinline__ __device__ float sm90_fp8_mega_moe_clamp_up(float x) {
    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
        x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
    return x;
}

template <bool kFastMath>
__forceinline__ __device__ float sm90_fp8_mega_moe_silu(float x) {
    const float e = kFastMath ? __expf(-x) : expf(-x);
    const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
    return x * sig;
}

template <bool kFastMath, float kActivationClamp>
__forceinline__ __device__ float sm90_fp8_mega_moe_swiglu(float g, float u) {
    g = sm90_fp8_mega_moe_clamp_gate<kActivationClamp>(g);
    u = sm90_fp8_mega_moe_clamp_up<kActivationClamp>(u);
    return sm90_fp8_mega_moe_silu<kFastMath>(g) * u;
}

__forceinline__ __device__ void sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(
    const float2& amax, float2& sf, float2& sf_inv) {
    constexpr float kScale = 1.0f / 448.0f;
    const auto scaled = make_float2(__fmul_rn(amax.x, kScale), __fmul_rn(amax.y, kScale));
    const auto exp_x = math::fast_log2_ceil(scaled.x);
    const auto exp_y = math::fast_log2_ceil(scaled.y);
    sf.x = math::fast_pow2(exp_x), sf_inv.x = math::fast_pow2(-exp_x);
    sf.y = math::fast_pow2(exp_y), sf_inv.y = math::fast_pow2(-exp_y);
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane,
          uint32_t kNumL1BlockNs, uint32_t kNumL2BlockNs,
          uint32_t kNumL1BlockKs, uint32_t kNumL2BlockKs,
          typename WorkspaceT, bool kLazyExpertCount,
          typename L1Func, typename L2Func>
CUTLASS_DEVICE void sm90_fp8_mega_moe_for_each_block_split(
    sched::MegaMoEScheduler<BLOCK_M, BLOCK_N, BLOCK_K,
                            L1_SHAPE_N, L1_SHAPE_K,
                            L2_SHAPE_N, L2_SHAPE_K,
                            kNumExpertsPerRank,
                            kNumExpertsPerWave,
                            kNumSMs, kNumRanks,
                            kNumExpertsPerLane,
                            kNumL1BlockNs, kNumL2BlockNs,
                            kNumL1BlockKs, kNumL2BlockKs,
                            WorkspaceT, kLazyExpertCount>& scheduler,
    L1Func&& l1_func, L2Func&& l2_func) {
    if constexpr (not kLazyExpertCount)
        scheduler.fetch_expert_recv_count();
    scheduler.prepare_expert_schedule();
    scheduler.set_expert_idx(0);

    while (true) {
        CUTE_TIE_DECL(scheduler.get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
        if (block_phase == sched::BlockPhase::None)
            break;

        if (block_phase == sched::BlockPhase::Linear1) {
            l1_func(current_local_expert_idx, kNumL1BlockKs, m_block_idx, n_block_idx);
        } else {
            l2_func(current_local_expert_idx, kNumL2BlockKs, m_block_idx, n_block_idx);
        }
    }
}

// Schedule one always-local shared expert before the routed-expert scheduler.
// The shared expert consumes this rank's original input rows directly, so it
// neither waits for dispatch counts nor participates in route metadata/RDMA.
// The final pool block is reserved as its L1-output/L2-input scratch.  Fusion
// is currently enabled only when num_tokens <= BLOCK_M, hence one M block is
// sufficient and the routed pool's worst-case padding leaves this tail block
// outside the live routed allocation.
template <uint32_t BLOCK_M, uint32_t kNumL1BlockNs,
          uint32_t kNumL2BlockNs, uint32_t kNumL1BlockKs,
          uint32_t kNumL2BlockKs, uint32_t kNumSMs,
          uint32_t kNumPoolBlocks, uint32_t kSharedExpertSentinel,
          typename SchedulerT, typename Func>
CUTLASS_DEVICE void sm90_fp8_mega_moe_for_each_shared_block(
    SchedulerT& scheduler, const uint32_t num_tokens, Func&& func) {
    DG_STATIC_ASSERT(kNumPoolBlocks > 0, "Shared expert requires a pool block");
    scheduler.current_num_tokens = num_tokens;
    scheduler.current_pool_block_offset = kNumPoolBlocks - 1;

    for (uint32_t task = blockIdx.x; task < kNumL1BlockNs;
         task += kNumSMs) {
        func(sched::BlockPhase::Linear1, kSharedExpertSentinel,
             kNumL1BlockKs, 0u, task);
    }
    for (uint32_t task = blockIdx.x; task < kNumL2BlockNs;
         task += kNumSMs) {
        func(sched::BlockPhase::Linear2, kSharedExpertSentinel,
             kNumL2BlockKs, 0u, task);
    }
}

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE — full implementation
// ----------------------------------------------------------------------------
// Pipeline (cluster=1, no TMA multicast):
//   * Dispatch warps: pull tokens (FP8) and SF (per-128 channel float) from
//     remote ranks via NVLink into the local L1 pool.
//   * GEMM TMA-load warps (1 for A+SFA, 1 for B+SFB) feed the pipeline stages.
//   * Math warpgroups (totalling kNumEpilogueThreads) consume each
//     stage with WGMMA, accumulate into registers, then run the epilogue:
//       - L1 (Linear1): SwiGLU with gate/up granularity-8 interleaved layout,
//         per-row amax over each output-SF group, FP8 e4m3 quantize, stage
//         through SMEM, then TMA store to local L1 output buffer.
//         The per-row SF is written as a *float* into the L2-acts SF buffer at
//         per-64 K granularity (one SF per L1 N block), so each block is fully
//         self-contained and no cross-CTA amax synchronisation is needed.
//       - L2 (Linear2): BF16 cast of the GEMM output, stage through SMEM,
//         then NVLink scatter to remote combine buffers.
//   * After all GEMM blocks, the math warps run the COMBINE step (top-k
//     reduction in BF16) — ported verbatim from the SM100 kernel.
// ============================================================================

#ifdef DG_MEGA_MOE_PUBLISHER_MAX_ACTIVE
// FIFO ticket semaphore bounding the number of concurrently publishing
// inter-node pairs, so the burst injection rate stays under the RoCE
// DCQCN/ECN trip point.  Both counters only grow across launches; module
// load zero-initializes them once, so no per-launch reset or cleanup epoch
// is involved.
__device__ static uint64_t g_publisher_limiter_ticket_head = 0;
__device__ static uint64_t g_publisher_limiter_released = 0;
#endif

#if defined(DG_MEGA_MOE_PUBLISHER_RATE_MBPS) or \
    defined(DG_MEGA_MOE_PUBLISHER_DST_RATE_MBPS)
// Token bucket pinning a publisher injection rate below the RoCE ECN trip
// point.  Virtual-time form: each submission pre-charges its bytes and waits
// until wall time catches up with bytes/rate.  A rate in MB/s numerically
// equals bytes/us, so integer division against a us clock is exact enough.
// Counters only grow across launches; the floor clamp refills at most
// kPublisherBurstBytes of credit after an idle gap, so a launch cannot open
// with an unbounded burst.  The global variant uses one bucket per GPU; the
// per-destination variant shades one bucket per dst rank so a skewed fan-in
// only throttles senders of the hot receiver instead of every link.
__device__ __forceinline__ void publisher_token_bucket_wait(
    uint64_t* epoch_ptr, uint64_t* bytes_ptr,
    const uint64_t& rate_bytes_per_us, const uint64_t& burst_bytes,
    const uint32_t& batch_bytes) {
    // burst_bytes must be scaled so that the RECEIVER-side burst stays
    // invariant: with per-destination buckets each receiver absorbs one
    // bucket's burst from every remote sender, so the per-bucket burst is
    // the per-sender budget divided by the number of inter-node peers.
    uint64_t t0 = ptx::ld_acq_gpu(epoch_ptr);
    if (t0 == 0) {
        const uint64_t now = ptx::get_globaltimer();
        atomicCAS(reinterpret_cast<unsigned long long*>(epoch_ptr),
                  0ull, static_cast<unsigned long long>(now));
        t0 = ptx::ld_acq_gpu(epoch_ptr);
    }
    const uint64_t elapsed_us = (ptx::get_globaltimer() - t0) / 1000;
    const uint64_t quota_bytes = elapsed_us * rate_bytes_per_us;
    if (quota_bytes > burst_bytes) {
        const uint64_t floor_bytes = quota_bytes - burst_bytes;
        if (ptx::ld_acq_gpu(bytes_ptr) < floor_bytes)
            atomicMax(reinterpret_cast<unsigned long long*>(bytes_ptr),
                      static_cast<unsigned long long>(floor_bytes));
    }
    const uint64_t my_end_bytes = atomicAdd(
        reinterpret_cast<unsigned long long*>(bytes_ptr),
        static_cast<unsigned long long>(batch_bytes)) + batch_bytes;
    if (my_end_bytes <= burst_bytes)
        return;
    const uint64_t deadline_us =
        (my_end_bytes - burst_bytes) / rate_bytes_per_us;
    constexpr int64_t kRateTimeoutCycles = 60ll * 2000000000ll;
    const uint64_t wait_start = clock64();
    while ((ptx::get_globaltimer() - t0) / 1000 < deadline_us)
        DG_TRAP_ONLY_DEVICE_ASSERT(
            clock64() - wait_start < kRateTimeoutCycles);
}
#endif

#ifdef DG_MEGA_MOE_PUBLISHER_RATE_MBPS
__device__ static uint64_t g_publisher_rate_epoch_ns = 0;
__device__ static uint64_t g_publisher_rate_bytes = 0;
#endif

#ifdef DG_MEGA_MOE_PUBLISHER_DST_RATE_MBPS
// One bucket per destination rank (64 = kCombineExpertReady rank limit).
// Only inter-node destinations ever index in; buckets share one epoch.
__device__ static uint64_t g_publisher_dst_rate_epoch_ns = 0;
__device__ static uint64_t g_publisher_dst_rate_bytes[64] = {};
#endif

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    uint32_t kEpilogueRegisterBudget,
    bool kReuseAccumAsFinal,
    bool kL2ArrivalCounter,
    bool kL2EpilogueRequiresFullSync,
    bool kSplitPhaseHotPath,
    bool kDispatchExpertReady,
    bool kLazyExpertCount,
    bool kCombineFullRow,
    bool kCombineExpertReady,
    bool kFP8SwapAB = false,
    bool kFuseSharedExpert = false,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_mega_moe_impl(void* y,
                       int* cumulative_local_expert_recv_stats,
                       const uint32_t num_tokens,
                       const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                       const float* __restrict__ l1_weights_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                       const float* __restrict__ l2_weights_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                       const float* __restrict__ shared_l1_acts_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                       const float* __restrict__ shared_l1_weights_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                       const float* __restrict__ shared_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(kNumDispatchThreads >= 64 and kNumDispatchThreads % 64 == 0,
                     "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 64 or kNumNonEpilogueThreads == 128,
                     "Invalid number of GEMM TMA warps");
    DG_STATIC_ASSERT((kNumDispatchThreads + kNumNonEpilogueThreads) % 128 == 0,
                     "Math warpgroup start must be 128-thread aligned");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of math/epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
#ifdef DG_MEGA_MOE_ASYNC_PUBLISHER
    DG_STATIC_ASSERT(kCombineFullRow and kCombineExpertReady,
                     "Async publisher requires full-row expert-ready combine");
    DG_STATIC_ASSERT(kNumMMANonEpilogueWarps >= 2,
                     "Async publisher reuses the second GEMM loader warp");
#endif
    DG_STATIC_ASSERT(BLOCK_M % 64 == 0, "BLOCK_M must be a multiple of WGMMA::M (64)");
    DG_STATIC_ASSERT(BLOCK_N == 128 or BLOCK_N == 256 or BLOCK_N == 512,
                     "SM90 MegaMoE supports CTA BLOCK_N=128/256/512");
    DG_STATIC_ASSERT(BLOCK_K == 128, "BLOCK_K is fixed to 128 (per-128 SF)");
    DG_STATIC_ASSERT(not kCombineExpertReady or kCombineFullRow,
                     "Per-expert ready requires full-row combine staging");
    DG_STATIC_ASSERT(not kLazyExpertCount or kDispatchExpertReady,
                     "Lazy expert counts require expert-ready dispatch");
    DG_STATIC_ASSERT(not kCombineExpertReady or kNumRanks <= 64,
                     "Per-expert destination mask supports at most 64 ranks");
    DG_STATIC_ASSERT(not kFuseSharedExpert or BLOCK_M <= kNumMaxTokensPerRank,
                     "Shared expert scratch requires one valid input block");

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    // Prefetch all TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        if constexpr (kFuseSharedExpert) {
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
        }
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing (mirror SM100 layout, except SF
    // for L2 activations uses per-64 K granularity)
    // =====================================================================
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, 128u);
    constexpr uint32_t kNumPoolBlocks = kNumMaxPoolTokens / BLOCK_M;
    DG_STATIC_ASSERT(kNumMaxPoolTokens % BLOCK_M == 0, "Invalid SM90 MegaMoE pool size");
    DG_STATIC_ASSERT(kNumPaddedSFPoolTokens >= kNumPoolBlocks * SF_BLOCK_M,
                     "Invalid SM90 MegaMoE SF pool capacity");

    const auto workspace = layout::SM90Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32);
    // Per-64 K float SF (SM90 only): 4 bytes per per-64 group => `kIntermediateHidden / 16` bytes/token
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 16);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);
    constexpr auto dispatch_staging_layout       = layout::Data(
        math::constexpr_align<uint32_t>(kHidden / 32 + sizeof(float), 128u));

    // Registered input area
    const auto input_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // Inter-node SF/weight staging area. Each pool token owns a separate,
    // cache-line-aligned row; it never aliases the later combine destination.
    const auto dispatch_staging_buffer = layout::Buffer(
        dispatch_staging_layout, 1, kNumMaxPoolTokens,
        combine_token_buffer.get_end_ptr());

    // Full-row combine keeps one registered BF16 row per pool token.  A
    // separate per-pool-block arrival counter lets the last L2 N-block CTA
    // publish the complete row with one large RDMA WRITE.
    constexpr uint32_t kNumMaxPoolBlocks =
        kNumMaxPoolTokens / layout::kMinCandidateBlockM;
    constexpr uint32_t kCombineFullRowPublishReadyBit = 1u << 31;
    constexpr auto combine_full_row_arrival_layout = layout::Data(
        kCombineFullRow ? sizeof(uint32_t) : 0u, false);
    const auto combine_full_row_arrival_buffer = layout::Buffer(
        combine_full_row_arrival_layout, 1, kNumMaxPoolBlocks,
        dispatch_staging_buffer.get_end_ptr());
    const auto combine_full_row_staging_base = reinterpret_cast<void*>(
        kCombineFullRow ? math::align(
            reinterpret_cast<uint64_t>(combine_full_row_arrival_buffer.get_end_ptr()),
            static_cast<uint64_t>(128)) :
            reinterpret_cast<uint64_t>(dispatch_staging_buffer.get_end_ptr()));
    constexpr auto combine_full_row_staging_layout = layout::Data(
        kCombineFullRow ? kHidden * sizeof(nv_bfloat16) : 0u);
    const auto combine_full_row_staging_buffer = layout::Buffer(
        combine_full_row_staging_layout, 1, kNumMaxPoolTokens,
        combine_full_row_staging_base);

    constexpr uint32_t kPhaseProfileMaxSMs =
        layout::kSM90MegaMoEProfileMaxSMs;
    constexpr uint32_t kPhaseProfileSlots =
        layout::kSM90MegaMoEProfileSlots;
    const auto phase_profile_buffer = layout::Buffer(
        layout::Data(kPhaseProfileSlots * sizeof(uint64_t), false),
        1, kPhaseProfileMaxSMs,
        combine_full_row_staging_buffer.get_end_ptr());

#ifdef DG_MEGA_MOE_PHASE_PROFILE
    DG_STATIC_ASSERT(kNumSMs <= kPhaseProfileMaxSMs,
                     "Too many SMs for phase profiler");
    enum ProfileSlot : uint32_t {
        kProfileMetadata = 0,
        kProfileDispatchBarrier = 1,
        kProfileDispatchPull = 2,
        kProfileRemoteRead = 3,
        kProfileCleanupBarrier = 4,
        kProfileL1 = 5,
        kProfileL2 = 6,
        kProfileScatter = 7,
        kProfileScatterPublish = 8,
        kProfileCombineBarrier = 9,
        kProfileCombineReduce = 10,
        kProfileTotal = 11,
        kProfileRemoteReadCount = 12,
        kProfileL1BlockCount = 13,
        kProfileL2BlockCount = 14,
        kProfileScatterWriteCount = 15,
        kProfileStartClock = 16,
        kProfileCombineReadyWait = 17,
        kProfileScatterStaging = 18,
        kProfileScatterArrival = 19,
        kProfileScatterWQE = 20,
        kProfileScatterReady = 21,
        kProfileScatterReadyMask = 22,
        kProfileScatterReadyAtomic = 23,
        kProfileScatterReadyFence = 24,
        kProfileScatterReadyNotify = 25,
        kProfileScatterReadySync = 26,
        kProfileCombineWork = 27,
        kProfileCombineTokenMax = 28,
        kProfileCombineTokenCount = 29,
        // Packed as [wait cycles:32 | global expert:16 | token:16].
        kProfileLongestReadyDependency = 30,
        // Absolute globaltimer (ns) stamps written only by SM 0, so they are
        // race-free and comparable across ranks on the same clock domain.
        // Their spread separates launch skew from protocol serialization.
        kProfileEntryGlobaltimer = 31,
        kProfileCountsSentGlobaltimer = 32,
        kProfileCountsReadyGlobaltimer = 33,
        // Splits the inter-node cleanup barrier into its IBGDA-quiet part
        // and its nvshmem_sync_all() part.
        kProfileCleanupQuietGlobaltimer = 34,
        kProfileCleanupSyncGlobaltimer = 35,
        kProfileCleanupIbgdaGlobaltimer = 36,
    };
    enum ExpertReadyProfileSlot : uint32_t {
        kExpertProfileEpoch = 0,
        kExpertProfileNumTokens = 1,
        kExpertProfileNumMBlocks = 2,
        kExpertProfileDstRankMask = 3,
        kExpertProfileFinalScatter = 4,
        kExpertProfileFinalWQE = 5,
        kExpertProfileReadyNotify = 6,
        kExpertProfilePublishGlobaltimer = 7,
        kExpertProfileSM = 8,
        kExpertProfilePoolBlockOffset = 9,
    };
    // Each (local expert, destination rank) has one exclusive publisher, so
    // its diagnostic row needs no initialization or atomic aggregation.  The
    // epoch is written last and lets the reader reject stale rows.  Reuse the
    // otherwise-unused middle of the fixed profiler allocation, between the
    // active per-SM rows and the expert-ready rows at its tail.
    enum PairPublisherProfileSlot : uint32_t {
        kPairProfileEpoch = 0,
        kPairProfileNumRows = 1,
        kPairProfileSubmitCount = 2,
        kPairProfileMaxBatchRows = 3,
        kPairProfileWQECycles = 4,
        kPairProfileStartGlobaltimer = 5,
        kPairProfileDataDoneGlobaltimer = 6,
        kPairProfileDoneGlobaltimer = 7,
        kPairProfileNumSlots = 8,
    };
    constexpr uint32_t kExpertProfileRowBase =
        kPhaseProfileMaxSMs - kNumExpertsPerRank;
    DG_STATIC_ASSERT(
        kNumSMs <= kExpertProfileRowBase,
        "Expert-ready profiler rows overlap per-SM profiler rows");
    constexpr uint32_t kNumPairProfileRows =
        kNumExpertsPerRank * kNumRanks;
    constexpr uint32_t kPairProfileFlatBase =
        kNumSMs * kPhaseProfileSlots;
    DG_STATIC_ASSERT(
        kPairProfileFlatBase +
                kNumPairProfileRows * kPairProfileNumSlots <=
            kExpertProfileRowBase * kPhaseProfileSlots,
        "Pair-publisher profiler overlaps expert-ready profiler rows");
    auto phase_profile = phase_profile_buffer.get_data_buffer(sm_idx)
        .get_base_ptr<unsigned long long>();
    auto pair_profile_base = phase_profile_buffer.get_data_buffer(0)
        .get_base_ptr<unsigned long long>() + kPairProfileFlatBase;
#endif

#ifdef DG_MEGA_MOE_INTERNODE
    if constexpr (kCombineFullRow) {
        if (sm_idx == 0 and thread_idx == 0)
            DG_DEVICE_ASSERT(
                comm::ibgda::ibgda_get_state()->num_rc_per_pe >= kNumExpertsPerRank);
    }
#endif

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    constexpr bool kSplitNWarpgroups =
        BLOCK_M == 64 and kNumEpilogueWarpgroups > 1 and
        BLOCK_N % kNumEpilogueWarpgroups == 0 and
        ((BLOCK_N / kNumEpilogueWarpgroups == 64) or (BLOCK_N / kNumEpilogueWarpgroups == 128));
    constexpr bool kSplitMNWarpgroups =
        BLOCK_M == 128 and BLOCK_N == 256 and kNumEpilogueWarpgroups == 4;
    constexpr uint32_t kWarpgroupSplitM = kSplitNWarpgroups ? 1 :
        (kSplitMNWarpgroups ? 2 : kNumEpilogueWarpgroups);
    constexpr uint32_t kWarpgroupSplitN = kSplitNWarpgroups ? kNumEpilogueWarpgroups :
        (kSplitMNWarpgroups ? 2 : 1);
    constexpr uint32_t WG_BLOCK_M = BLOCK_M / kWarpgroupSplitM;
    constexpr uint32_t WG_BLOCK_N = BLOCK_N / kWarpgroupSplitN;
    constexpr uint32_t kNumCombineWarps = kNumEpilogueWarps;
    using L1WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;  // M=64, N=WG_BLOCK_N, K=32
    using L2WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    constexpr uint32_t kL1OutputArrivalParts = 1;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT(kWarpgroupSplitM * kWarpgroupSplitN == kNumEpilogueWarpgroups,
                     "Invalid warpgroup split");
    DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M,
                     "Each warpgroup must run exactly one WGMMA-M tile");
    DG_STATIC_ASSERT(kNumCombineWarps <= kNumEpilogueWarps,
                     "Combine warp count must fit in epilogue warps");

    // Cluster=1 -> no multicast, A/B are loaded full-sized
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t L1_OUT_BLOCK_N  = BLOCK_N / 2;  // post-SwiGLU
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2;
    // When WG_L1_OUT_BLOCK_N < 64 the two N-split warpgroups jointly own a
    // single per-64 L2-acts SF group, so they must publish ONE shared SF slot
    // (k_sf_idx == n_block_idx) instead of one per warpgroup. The amax that
    // feeds that shared SF must be reduced across both warpgroups.
    constexpr bool kSplitNSharesSF = kSplitNWarpgroups and (WG_L1_OUT_BLOCK_N < 64);
    constexpr bool kSwapABEligible =
        kFP8SwapAB and kSplitNWarpgroups and (BLOCK_M == 64) and (BLOCK_N == 128) and
        (kWarpgroupSplitN == 2);
    constexpr bool kSwapABActive = kSwapABEligible;
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    DG_STATIC_ASSERT(not kSwapABEligible or (BLOCK_M % 8 == 0),
                     "swapAB epilogue token chunks assume BLOCK_M is a multiple of 8");
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    constexpr uint32_t kSwizzleBMode   = BLOCK_K * sizeof(b_dtype_t);   // 128
    constexpr uint32_t kSwizzleCDMode  = 128;
    constexpr uint32_t kGranK          = 128;          // L1 acts SF, weights SF
    constexpr uint32_t kL2ActsSFGranK  = 64;           // L2 acts SF (per-64 K, SM90 only)

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // SFA per-stage must be sized for the larger of L1 (BLOCK_M floats) and L2 (2*BLOCK_M floats per-64).
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>(2 * BLOCK_M * sizeof(float), 128u);
    // Block (128, 128) weight SF is loaded directly from global by the math
    // warpgroup, so no SMEM is needed.
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = 0;

    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte), L2 BF16
    // (BLOCK_M * BLOCK_N * 2 bytes), and the swapAB L1 FP32+FP8 staging
    // buffers. Split-M warpgroups own disjoint row slices; shared-SF split-N
    // warpgroups stage disjoint column slices into one CTA tile.
    constexpr uint32_t SMEM_CD_L1_SIZE = BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SWAP_L1_FP32_SIZE =
        kSwapABActive ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(float) : 0;
    constexpr uint32_t SMEM_CD_SWAP_L1_FP8_SIZE =
        kSwapABActive ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0;
    constexpr uint32_t SMEM_CD_SWAP_L1_SIZE =
        kSwapABActive ? (SMEM_CD_SWAP_L1_FP32_SIZE + SMEM_CD_SWAP_L1_FP8_SIZE) : 0;
    constexpr uint32_t SMEM_CD_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_SIZE    = math::constexpr_align(
        SMEM_CD_BASE_SIZE > SMEM_CD_SWAP_L1_SIZE ? SMEM_CD_BASE_SIZE : SMEM_CD_SWAP_L1_SIZE,
        kSharedMemoryAlignment);

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_gemm_base);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_gemm_base);
    auto smem_cd_swap_l1_fp32 = reinterpret_cast<float*>(smem_gemm_base);
    auto smem_cd_swap_l1_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(
        math::advance_ptr(smem_gemm_base, SMEM_CD_SWAP_L1_FP32_SIZE));

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });

    // Barriers live after SF (SFB is loaded directly from global, no SMEM)
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });

    // =====================================================================
    // Initialization
    // =====================================================================
#ifdef DG_MEGA_MOE_PHASE_PROFILE
    if (thread_idx < kPhaseProfileSlots)
        phase_profile[thread_idx] = 0;
    if (thread_idx == 0)
        phase_profile[kProfileStartClock] = clock64();
    // Per-SM row, so every CTA stamps its own entry: the spread across
    // CTAs is exactly the intra-rank CTA start skew.
    if (thread_idx == 0)
        phase_profile[kProfileEntryGlobaltimer] = ptx::get_globaltimer();
#endif
    if constexpr (kDispatchExpertReady) {
        // The ready slot packs this epoch in its high 32 bits and the token
        // count in its low 32 bits.  Tag 3 still clears slots after each call.
        if (sm_idx == 0 and thread_idx == 0)
            ptx::atomic_add_sys(workspace.get_dispatch_launch_epoch_ptr(), 1ull);
    }
    if constexpr (kCombineExpertReady) {
        // Collective calls advance in lock-step across ranks.  The persistent
        // 64-bit epoch lets receivers distinguish this launch from stale ready
        // notifications without clearing the ready array between invocations.
        if (sm_idx == 0 and thread_idx == 0)
            ptx::atomic_add_sys(workspace.get_combine_launch_epoch_ptr(), 1ull);
    }
    if (warp_idx == 0) {
        // Clean expert-count shared memory
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            ptx::st_shared(smem_expert_count + i, 0u);
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
#ifdef DG_MEGA_MOE_MERGE_AB_LOADER
                // One warp issues A/SFA and B and commits their total byte
                // count with a single producer arrival.
                full_barriers[i]->init(1);
#else
                // Two producer warps (A+SFA loader, B+SFB loader) each call
                // `arrive_and_expect_tx` per stage, so init count must be 2.
                full_barriers[i]->init(2);
#endif
                // Each math warp arrives once per stage release.
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumCombineWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    constexpr uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
    constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    constexpr uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N;
    constexpr uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K;
    constexpr uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K;
    constexpr uint32_t kSharedExpertSentinel = kNumExpertsPerRank;
    constexpr uint32_t kSharedPoolBlockIdx = kNumPoolBlocks - 1;
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks,
        kNumExpertsPerLane, kNumL1BlockNs, kNumL2BlockNs,
        kNumL1BlockKs, kNumL2BlockKs,
        layout::SM90Workspace, kLazyExpertCount>(
            workspace,
            kDispatchExpertReady ? workspace.get_dispatch_launch_epoch_ptr() : nullptr,
            sym_buffer.rank_idx);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;
#ifdef DG_MEGA_MOE_ASYNC_PUBLISHER
    constexpr uint32_t kNumAsyncPublisherThreads = 32;
#else
    constexpr uint32_t kNumAsyncPublisherThreads = 0;
#endif
    constexpr uint32_t kNumDispatchEpilogueSyncThreads =
        kNumDispatchThreads + kNumEpilogueThreads + kNumAsyncPublisherThreads;

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    // For the 256-epilogue-thread split-N decode path:
    //   64*48 + 64*40 + 256*168 = 48640 <= 64512.
    // For the 512-epilogue-thread split-MN path, trim dispatch and loader roles
    // so launch bounds still leave enough WGMMA registers.
    // Reduced-thread decode (kNumThreads<=256) raises the launch-bounds
    // register ceiling to 65536/256=256; grant the epilogue warpgroup the full
    // 256 so the accumulator double-buffer fits without spilling.
    //   64*48 + 64*40 + 128*256 = 38400 <= 64512.
    constexpr uint32_t kNumEpilogueRegisters    =
        kEpilogueRegisterBudget == 0 ?
            (kNumEpilogueThreads == 512 ? 112 :
                (kNumEpilogueThreads == 256 ? 168 :
                    (kNumThreads <= 256u ? 256 : 208))) :
            kEpilogueRegisterBudget;
    // The 512-epilogue-thread path has only 3584 registers of headroom at
    // epilogue=112.  Raising epilogue to 120 is not viable without changing
    // the role topology: dispatch=24 stalls the split-MN path, while
    // non-epilogue=16 is below ptxas' setmaxnreg.dec legal minimum.
    constexpr uint32_t kNumDispatchRegisters =
        kNumEpilogueThreads == 512 ? 32 : 48;
    constexpr uint32_t kNumNonEpilogueRegisters =
        kNumEpilogueThreads == 512 ? 24 : 40;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" → SF token index is now the simple
    //       per-block linear mapping (no 4×32 transpose).
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        const bool profile_dispatch_leader = warp_idx == 0 and lane_idx == 0;
        const uint64_t profile_metadata_start =
            profile_dispatch_leader ? clock64() : 0;
#endif

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count tokens per expert
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Stake out per-expert SM offsets via global atomic
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source token-topk indices to remote ranks
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_local_expert_idx = expert_idx % kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                dst_local_expert_idx, sym_buffer.rank_idx, dst_slot_idx);
#ifdef DG_MEGA_MOE_INTERNODE
            if (dst_rank_idx / DG_MEGA_MOE_NVL_PEERS != sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS) {
#ifdef DG_MEGA_MOE_DISPATCH_GATEWAY
                // Two-level handshake: stage the inter-node entry in the
                // local same-rail gateway's collect box over NVLink; the
                // gateway forwards it in bulk.  Slot indices are the same
                // ones the remote inbox uses, so the gateway can copy cell
                // prefixes verbatim.
                const auto gateway_rank_idx =
                    sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS *
                        DG_MEGA_MOE_NVL_PEERS +
                    dst_rank_idx % DG_MEGA_MOE_NVL_PEERS;
                *sym_buffer.map(
                    workspace.get_gateway_entry_ptr(
                        sym_buffer.rank_idx % DG_MEGA_MOE_NVL_PEERS,
                        dst_local_expert_idx, dst_slot_idx),
                    gateway_rank_idx) = token_topk_idx;
#else
                if constexpr (kDispatchExpertReady)
                    comm::ibgda::put_inline_with_credit<uint32_t>(
                        dst_ptr, token_topk_idx, static_cast<int>(dst_rank_idx),
                        static_cast<int>(dst_local_expert_idx));
                else
                    comm::ibgda::put_inline<uint32_t>(
                        dst_ptr, token_topk_idx, static_cast<int>(dst_rank_idx),
                        static_cast<int>(dst_slot_idx));
#endif
            } else
#endif
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // Every writer makes its same-node route stores visible before the
        // grid-wide handoff to SM 0.  Inter-node WQE ordering is provided by
        // the per-expert RC QP itself.
        if constexpr (kDispatchExpertReady)
            __threadfence_system();

#if defined(DG_MEGA_MOE_DISPATCH_GATEWAY_EAGER) and \
    defined(DG_MEGA_MOE_INTERNODE)
        // Eager (send-when-full) handshake: no grid_sync.  Each CTA bumps
        // one completion counter per destination rank after fencing its
        // route stores; the CTA that brings a counter to kNumSMs is that
        // direction's trigger.  Reaching kNumSMs also proves every CTA
        // passed stake-out, so expert_send_count is final — the trigger
        // writes the manifest row (inter-node: to the local gateway, plus
        // the flag) or the recv-count row (same-node: directly) without
        // any further waiting.  The acq_rel counter chain orders every
        // CTA's fenced stores before the trigger's row writes.
        if constexpr (kDispatchExpertReady) {
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            if (thread_idx < kNumRanks) {
                const uint32_t dst_rank_idx = thread_idx;
                const auto old_done = ptx::atomic_add_acq_rel_sys(
                    workspace.get_gateway_direction_done_ptr(dst_rank_idx),
                    1);
                DG_TRAP_ONLY_DEVICE_ASSERT(old_done < kNumSMs);
                if (old_done + 1 == kNumSMs) {
                    const auto dispatch_epoch_full = ptx::ld_acq_sys(
                        workspace.get_dispatch_launch_epoch_ptr());
                    const auto dispatch_epoch =
                        static_cast<uint32_t>(dispatch_epoch_full);
                    const bool dst_is_inter =
                        dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
                    const auto gateway_rank_idx =
                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS *
                            DG_MEGA_MOE_NVL_PEERS +
                        dst_rank_idx % DG_MEGA_MOE_NVL_PEERS;
                    const auto src_nvl_idx =
                        sym_buffer.rank_idx % DG_MEGA_MOE_NVL_PEERS;
                    for (uint32_t e = 0; e < kNumExpertsPerRank; ++ e) {
                        const auto expert_status =
                            *workspace.get_expert_send_count_ptr(
                                dst_rank_idx * kNumExpertsPerRank + e);
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            (expert_status >> 32) == kNumSMs);
                        const uint64_t ready_status =
                            (static_cast<uint64_t>(dispatch_epoch) << 32) |
                            static_cast<uint32_t>(expert_status);
                        if (dst_is_inter) {
                            *sym_buffer.map(
                                workspace.get_gateway_manifest_ptr(
                                    src_nvl_idx, e),
                                gateway_rank_idx) = ready_status;
                        } else {
                            ptx::st_relaxed_sys(
                                sym_buffer.map(
                                    workspace.get_expert_recv_count_ptr(
                                        sym_buffer.rank_idx, e),
                                    dst_rank_idx),
                                ready_status);
                        }
                    }
                    if (dst_is_inter) {
                        __threadfence_system();
                        ptx::st_release_sys(
                            sym_buffer.map(
                                workspace.get_gateway_flag_ptr(
                                    src_nvl_idx),
                                gateway_rank_idx),
                            dispatch_epoch_full);
                    }
                }
            }
        }
#else
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );
#endif

#if defined(DG_MEGA_MOE_DISPATCH_GATEWAY_EAGER) and \
    defined(DG_MEGA_MOE_INTERNODE)
        if constexpr (false) {
#else
        if (sm_idx == 0) {
#endif
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                const auto recv_count_ptr = workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx);
                const auto recv_count_sum_ptr = workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx);
#ifdef DG_MEGA_MOE_INTERNODE
                uint64_t ready_status = expert_status;
                if constexpr (kDispatchExpertReady) {
                    const auto dispatch_epoch = static_cast<uint32_t>(
                        ptx::ld_acq_sys(workspace.get_dispatch_launch_epoch_ptr()));
                    ready_status =
                        (static_cast<uint64_t>(dispatch_epoch) << 32) |
                        static_cast<uint32_t>(expert_status);
                }
                if (dst_rank_idx / DG_MEGA_MOE_NVL_PEERS != sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS) {
#ifdef DG_MEGA_MOE_DISPATCH_GATEWAY
                    // Stage this direction's manifest row on the local
                    // gateway; the gateway's trailing 1KB WRITE lands all
                    // eight rows in the remote recv-count table at once.
                    const auto gateway_rank_idx =
                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS *
                            DG_MEGA_MOE_NVL_PEERS +
                        dst_rank_idx % DG_MEGA_MOE_NVL_PEERS;
                    *sym_buffer.map(
                        workspace.get_gateway_manifest_ptr(
                            sym_buffer.rank_idx % DG_MEGA_MOE_NVL_PEERS,
                            dst_local_expert_idx),
                        gateway_rank_idx) = ready_status;
#else
                    if constexpr (kDispatchExpertReady)
                        comm::ibgda::put_inline_with_credit<uint64_t>(
                            recv_count_ptr, ready_status, static_cast<int>(dst_rank_idx),
                            static_cast<int>(dst_local_expert_idx));
                    else
                        comm::ibgda::put_inline<uint64_t>(
                            recv_count_ptr, ready_status, static_cast<int>(dst_rank_idx),
                            static_cast<int>(dst_local_expert_idx));
#endif
                } else if constexpr (kDispatchExpertReady) {
                    ptx::st_relaxed_sys(
                        sym_buffer.map(recv_count_ptr, dst_rank_idx), ready_status);
                } else {
                    *sym_buffer.map(recv_count_ptr, dst_rank_idx) = ready_status;
                }
#else
                *sym_buffer.map(recv_count_ptr, dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(sym_buffer.map(recv_count_sum_ptr, dst_rank_idx), expert_status);
#endif
            }
#if defined(DG_MEGA_MOE_DISPATCH_GATEWAY) and \
    defined(DG_MEGA_MOE_INTERNODE)
            // All of this rank's entries and manifest rows are staged on
            // the gateways.  Raise our per-source flag on every local
            // gateway so each direction's forwarder can depart.
            if constexpr (kDispatchExpertReady) {
                ptx::sync_aligned(
                    kNumDispatchThreads, kDispatchBarrierIdx);
                __threadfence_system();
                if (thread_idx < DG_MEGA_MOE_NVL_PEERS) {
                    const auto node_base = sym_buffer.rank_idx /
                        DG_MEGA_MOE_NVL_PEERS * DG_MEGA_MOE_NVL_PEERS;
                    const auto flag_epoch = ptx::ld_acq_sys(
                        workspace.get_dispatch_launch_epoch_ptr());
                    ptx::st_release_sys(
                        sym_buffer.map(
                            workspace.get_gateway_flag_ptr(
                                sym_buffer.rank_idx %
                                DG_MEGA_MOE_NVL_PEERS),
                            node_base + thread_idx),
                        flag_epoch);
                }
            }
#endif
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

#if defined(DG_MEGA_MOE_DISPATCH_GATEWAY) and \
    defined(DG_MEGA_MOE_INTERNODE)
        // Second level of the handshake: this rank is the gateway for its
        // rail direction.  Wait for all local peers' flags, forward every
        // non-empty (src, expert) cell as one WRITE, then land the 1KB
        // manifest into the remote recv-count rows on the same dedicated
        // QP — RC ordering makes the manifest imply the entries, the same
        // trick the expert-ready protocol already relies on.
        if constexpr (kDispatchExpertReady) {
            if (sm_idx == 0 and warp_idx == 0) {
                constexpr int kGatewayQpId = 16;
                const auto my_epoch = ptx::ld_acq_sys(
                    workspace.get_dispatch_launch_epoch_ptr());
                const auto peer_rank_idx = static_cast<int>(
                    (sym_buffer.rank_idx + DG_MEGA_MOE_NVL_PEERS) %
                    kNumRanks);
                const auto node_src_base = sym_buffer.rank_idx /
                    DG_MEGA_MOE_NVL_PEERS * DG_MEGA_MOE_NVL_PEERS;
                constexpr int64_t kGatewayTimeoutCycles =
                    60ll * 2000000000ll;
                const uint64_t gateway_wait_start = clock64();
                for (uint32_t i = lane_idx; i < DG_MEGA_MOE_NVL_PEERS;
                     i += 32) {
                    while (ptx::ld_acq_sys(
                               workspace.get_gateway_flag_ptr(i)) !=
                           my_epoch)
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            clock64() - gateway_wait_start <
                            kGatewayTimeoutCycles);
                }
                __syncwarp();
#ifdef DG_MEGA_MOE_DISPATCH_GATEWAY_V3
                // V3: ship the whole collect box as ONE contiguous WRITE
                // into the remote landing zone (byte-for-byte mirror), no
                // matter how routing scattered the cells.  Feasible only
                // because the per-slot capacity is bounded small by the
                // host-side path selection (max tokens per rank <= 2048).
                comm::ibgda::put_nbi_warp(
                    reinterpret_cast<uint64_t>(
                        workspace.get_gateway_landing_ptr(0, 0, 0)),
                    reinterpret_cast<uint64_t>(
                        workspace.get_gateway_entry_ptr(0, 0, 0)),
                    static_cast<size_t>(DG_MEGA_MOE_NVL_PEERS) *
                        kNumExpertsPerRank *
                        workspace.num_max_tokens_per_rank *
                        sizeof(uint32_t),
                    peer_rank_idx, kGatewayQpId,
                    static_cast<int>(lane_idx));
#else
                for (uint32_t src_nvl = 0;
                     src_nvl < DG_MEGA_MOE_NVL_PEERS; ++ src_nvl) {
                    for (uint32_t e = 0; e < kNumExpertsPerRank; ++ e) {
                        uint64_t row = 0;
                        if (lane_idx == 0)
                            row = ptx::ld_acq_sys(
                                workspace.get_gateway_manifest_ptr(
                                    src_nvl, e));
                        row = __shfl_sync(0xffffffff, row, 0);
                        const auto cell_count =
                            static_cast<uint32_t>(row);
                        if (cell_count == 0)
                            continue;
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            cell_count <=
                            workspace.num_max_tokens_per_rank);
                        comm::ibgda::put_nbi_warp(
                            reinterpret_cast<uint64_t>(
                                workspace.get_src_token_topk_idx_ptr(
                                    e, node_src_base + src_nvl, 0)),
                            reinterpret_cast<uint64_t>(
                                workspace.get_gateway_entry_ptr(
                                    src_nvl, e, 0)),
                            cell_count * sizeof(uint32_t),
                            peer_rank_idx, kGatewayQpId,
                            static_cast<int>(lane_idx));
                    }
                }
#endif
                comm::ibgda::put_nbi_warp(
                    reinterpret_cast<uint64_t>(
                        workspace.get_expert_recv_count_ptr(
                            node_src_base, 0)),
                    reinterpret_cast<uint64_t>(
                        workspace.get_gateway_manifest_ptr(0, 0)),
                    DG_MEGA_MOE_NVL_PEERS * kNumExpertsPerRank *
                        sizeof(uint64_t),
                    peer_rank_idx, kGatewayQpId,
                    static_cast<int>(lane_idx));
            }
        }
#endif

        // Shared math reuses the beginning of the CTA scratch area that held
        // metadata counts.  Release it as soon as all route metadata has been
        // constructed, before eager count waiting and RDMA pull, so shared
        // GEMM can cover both routed front-end phases.
        if constexpr (kFuseSharedExpert)
            ptx::sync_unaligned(
                kNumDispatchEpilogueSyncThreads,
                kDispatchWithEpilogueBarrierIdx);

        // In expert-ready mode, route entries and the final epoch/count marker
        // use the same per-expert RC QP.  The QP's in-order execution makes the
        // marker the completion boundary, so no tag-1 barrier/quiet is needed.
        // The legacy path still relies on the global barrier below.

        // Make all dispatch metadata pushes (intra-node NVLink stores + inter-node NVSHMEM
        // puts/atomics) visible system-wide before the cross-rank barrier, so the pull phase
        // observes up-to-date counts/indices. Without this, a plain NVLink store to a peer's
        // mapped address may not be visible to the peer at barrier release, causing rank-to-rank
        // progress skew (some ranks stall in the pull loop while others reach the next barrier).
        __threadfence_system();
#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (profile_dispatch_leader)
            phase_profile[kProfileMetadata] = clock64() - profile_metadata_start;
        if (profile_dispatch_leader)
            phase_profile[kProfileCountsSentGlobaltimer] =
                ptx::get_globaltimer();
        const uint64_t profile_dispatch_barrier_start =
            profile_dispatch_leader ? clock64() : 0;
#endif
        if constexpr (kDispatchExpertReady) {
            if constexpr (not kLazyExpertCount) {
                scheduler.fetch_expert_recv_count();
            }
        } else {
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                false, true);
        }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (profile_dispatch_leader)
            phase_profile[kProfileDispatchBarrier] =
                clock64() - profile_dispatch_barrier_start;
        if (profile_dispatch_leader)
            phase_profile[kProfileCountsReadyGlobaltimer] =
                ptx::get_globaltimer();
#endif

        // The non-fused path starts routed math and dispatch pull together.
        // Shared fusion has already rendezvoused immediately after metadata,
        // allowing shared GEMM to overlap eager-count wait and pull.
        if constexpr (not kFuseSharedExpert)
            ptx::sync_unaligned(
                kNumDispatchEpilogueSyncThreads,
                kDispatchWithEpilogueBarrierIdx);

        // In lazy mode, one global producer warp turns the per-source ready
        // slots into an epoch-tagged per-expert cache.  Other warps proceed
        // independently and wait only for the expert prefix they need.
        if constexpr (kLazyExpertCount) {
            if (sm_idx == 0 and warp_idx == 0)
                scheduler.publish_expert_recv_counts();
        }

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        const uint64_t profile_pull_start = lane_idx == 0 ? clock64() : 0;
        uint64_t profile_remote_read_cycles = 0;
        uint32_t profile_remote_read_count = 0;
#endif

        if constexpr (not kDispatchExpertReady)
            scheduler.fetch_expert_recv_count();

#ifdef DG_MEGA_MOE_DISPATCH_FOLLOW_EXPERT_SCHEDULE
        // Use exactly the same deterministic order as the L1/L2 scheduler.
        // Counts are complete here, so every dispatch and compute warp builds
        // identical keys without publishing another shared order buffer.
        scheduler.prepare_expert_schedule();
#endif

#ifdef DG_MEGA_MOE_RANK_MAJOR_PAIR_TASKS
        DG_STATIC_ASSERT(
            kDispatchExpertReady and not kLazyExpertCount,
            "Rank-major pool requires eager expert-ready dispatch");
        constexpr uint32_t kNumGlobalWarps =
            kNumSMs * kNumDispatchWarps;
        constexpr uint32_t kRankBatchTokens = 32;
        const uint32_t global_warp_idx =
            sm_idx * kNumDispatchWarps + warp_idx;
        const uint32_t num_dispatch_experts =
            scheduler.get_num_scheduled_experts();

        // Keep every expert contiguous for grouped GEMM, but place each
        // source rank in one contiguous sub-range inside that expert.  Work
        // assignment is by (expert, source-rank), independently of pool
        // order, so all ranks/QPs can still make progress concurrently.
        for (uint32_t task_idx = global_warp_idx;
             task_idx < num_dispatch_experts * kNumRanks;
             task_idx += kNumGlobalWarps) {
            const uint32_t schedule_pos = task_idx / kNumRanks;
#ifdef DG_MEGA_MOE_DISPATCH_FOLLOW_EXPERT_SCHEDULE
            const uint32_t current_expert_idx =
                scheduler.get_scheduled_expert_idx(schedule_pos);
#else
            const uint32_t current_expert_idx = schedule_pos;
#endif
            const uint32_t current_rank_in_expert_idx = task_idx % kNumRanks;
            const uint32_t expert_pool_block_offset =
                scheduler.get_pool_block_offset(current_expert_idx);

            uint32_t rank_count = 0;
            uint32_t rank_prefix = 0;
            if (lane_idx == 0) {
                #pragma unroll
                for (uint32_t src_rank_idx = 0;
                     src_rank_idx < kNumRanks; ++ src_rank_idx) {
                    const uint32_t count = static_cast<uint32_t>(
                        *workspace.get_expert_recv_count_ptr(
                            src_rank_idx, current_expert_idx));
                    if (src_rank_idx < current_rank_in_expert_idx)
                        rank_prefix += count;
                    else if (src_rank_idx == current_rank_in_expert_idx)
                        rank_count = count;
                }
            }
            rank_count = __shfl_sync(0xffffffff, rank_count, 0);
            rank_prefix = __shfl_sync(0xffffffff, rank_prefix, 0);
            if (rank_count == 0)
                continue;

#ifdef DG_MEGA_MOE_INTERNODE
            const bool tok_is_inter =
                current_rank_in_expert_idx / DG_MEGA_MOE_NVL_PEERS !=
                sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
            if (tok_is_inter) {
                const int inter_qp_id =
                    static_cast<int>(current_expert_idx);
                for (uint32_t chunk_base = 0; chunk_base < rank_count;
                     chunk_base += kRankBatchTokens) {
                    const uint32_t batch_count = cute::min(
                        rank_count - chunk_base, kRankBatchTokens);
                    const bool active = lane_idx < batch_count;
                    const uint32_t token_idx_in_rank = chunk_base + lane_idx;
                    uint32_t src_token_topk_idx = 0;
                    uint32_t src_token_idx = 0;
                    uint32_t src_topk_idx = 0;
                    uint32_t pool_token_idx = 0;
                    uint64_t inter_staging_ptr = 0;
                    if (active) {
                        src_token_topk_idx =
#ifdef DG_MEGA_MOE_DISPATCH_GATEWAY_V3
                            // Inter-node entries landed in the V3 mirror
                            // zone, not the classic inbox.
                            *workspace.get_gateway_landing_ptr(
                                current_rank_in_expert_idx %
                                    DG_MEGA_MOE_NVL_PEERS,
                                current_expert_idx,
                                token_idx_in_rank);
#else
                            *workspace.get_src_token_topk_idx_ptr(
                                current_expert_idx,
                                current_rank_in_expert_idx,
                                token_idx_in_rank);
#endif
                        src_token_idx = src_token_topk_idx / kNumTopk;
                        src_topk_idx = src_token_topk_idx % kNumTopk;
                        const uint32_t token_idx_in_expert =
                            rank_prefix + token_idx_in_rank;
                        pool_token_idx =
                            expert_pool_block_offset * BLOCK_M +
                            token_idx_in_expert;
                        inter_staging_ptr = reinterpret_cast<uint64_t>(
                            dispatch_staging_buffer
                                .get_data_buffer(pool_token_idx)
                                .get_base_ptr<float>());
                    }

                    const comm::ibgda::GetRequest read_requests[3] = {
                        {
                            active ? reinterpret_cast<uint64_t>(
                                l1_token_buffer
                                    .get_data_buffer(pool_token_idx)
                                    .get_base_ptr()) : 0,
                            active ? reinterpret_cast<uint64_t>(
                                input_token_buffer
                                    .get_data_buffer(src_token_idx)
                                    .get_base_ptr()) : 0,
                            pull_buffer.get_num_bytes()
                        },
                        {
                            inter_staging_ptr,
                            active ? reinterpret_cast<uint64_t>(
                                input_sf_buffer
                                    .get_data_buffer(src_token_idx)
                                    .get_base_ptr<float>()) : 0,
                            (kHidden / 128) * sizeof(float)
                        },
                        {
                            inter_staging_ptr +
                                (active ? (kHidden / 128) * sizeof(float) : 0),
                            active ? reinterpret_cast<uint64_t>(
                                input_topk_weights_buffer
                                    .get_base_ptr<float>() +
                                src_token_topk_idx) : 0,
                            sizeof(float)
                        }
                    };
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    const uint64_t profile_remote_read_start =
                        lane_idx == 0 ? clock64() : 0;
#endif
#ifdef DG_MEGA_MOE_DISPATCH_BATCH_READ
                    const auto completion_idx = comm::ibgda::get_batch_warp(
                        read_requests, active,
                        static_cast<int>(current_rank_in_expert_idx),
                        inter_qp_id, static_cast<int>(lane_idx));
                    if (lane_idx == 0)
                        comm::ibgda::wait_until(
                            static_cast<int>(current_rank_in_expert_idx),
                            inter_qp_id, completion_idx);
#else
                    // A/B fallback: preserve rank-major placement and the
                    // (expert, source-rank) task mapping, but let each active
                    // lane post and wait for its own three-request token group.
                    // This isolates layout/scheduling from warp-wide WQE
                    // reservation and ring-credit behavior.
                    for (uint32_t token_owner = 0;
                         token_owner < batch_count; ++ token_owner) {
                        if (lane_idx == token_owner) {
                            const auto completion_idx =
                                comm::ibgda::get_batch_thread(
                                    read_requests,
                                    static_cast<int>(current_rank_in_expert_idx),
                                    inter_qp_id);
                            comm::ibgda::wait_until(
                                static_cast<int>(current_rank_in_expert_idx),
                                inter_qp_id, completion_idx);
                        }
                        __syncwarp();
                    }
#endif
                    __syncwarp();
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    if (lane_idx == 0) {
                        profile_remote_read_cycles +=
                            clock64() - profile_remote_read_start;
                        profile_remote_read_count += batch_count;
                    }
#endif

                    // Consume one completed row at a time with warp-coalesced
                    // SF stores.  The expensive RNIC posting/completion was
                    // already shared by the whole rank segment above.
                    for (uint32_t token_in_batch = 0;
                         token_in_batch < batch_count; ++ token_in_batch) {
                        const uint32_t token_src_token_idx = __shfl_sync(
                            0xffffffff, src_token_idx,
                            static_cast<int>(token_in_batch));
                        const uint32_t token_src_topk_idx = __shfl_sync(
                            0xffffffff, src_topk_idx,
                            static_cast<int>(token_in_batch));
                        const uint32_t token_pool_idx = __shfl_sync(
                            0xffffffff, pool_token_idx,
                            static_cast<int>(token_in_batch));
                        const uint64_t token_staging_ptr = __shfl_sync(
                            0xffffffff, inter_staging_ptr,
                            static_cast<int>(token_in_batch));
                        const auto token_staging =
                            reinterpret_cast<const float*>(token_staging_ptr);
                        const uint32_t token_idx_in_expert =
                            rank_prefix + chunk_base + token_in_batch;
                        const uint32_t pool_block_idx =
                            expert_pool_block_offset +
                            token_idx_in_expert / BLOCK_M;
                        const uint32_t token_idx_in_block =
                            token_idx_in_expert % BLOCK_M;
                        const uint32_t sf_pool_token_idx =
                            pool_block_idx * SF_BLOCK_M + token_idx_in_block;

                        constexpr uint32_t kNumSFFloats = kHidden / 128;
                        const auto local_sf_ptr =
                            l1_sf_buffer.get_base_ptr<float>();
                        #pragma unroll
                        for (uint32_t sf_iter = 0;
                             sf_iter < math::constexpr_ceil_div(
                                 kNumSFFloats, 32u); ++ sf_iter) {
                            const uint32_t sf_idx = sf_iter * 32 + lane_idx;
                            if (sf_idx < kNumSFFloats)
                                local_sf_ptr[
                                    sf_idx * kNumPaddedSFPoolTokens +
                                    sf_pool_token_idx] =
                                    __ldcv(token_staging + sf_idx);
                        }
                        __syncwarp();

                        if (lane_idx == 0) {
                            *l1_topk_weights_buffer
                                .get_data_buffer(token_pool_idx)
                                .get_base_ptr<float>() =
                                __ldcv(token_staging + kHidden / 128);
                            *workspace.get_token_src_metadata_ptr(
                                token_pool_idx) = {
                                    current_rank_in_expert_idx,
                                    token_src_token_idx,
                                    token_src_topk_idx};
                            ptx::red_add_rel(
                                workspace.get_l1_arrival_count_ptr(
                                    pool_block_idx), 1);
                        }
                        __syncwarp();
                    }
                }
                continue;
            }
#endif

            // Same-node source segment.  Keep the existing warp-cooperative
            // TMA path, but walk this rank's contiguous pool sub-range.
            for (uint32_t token_idx_in_rank = 0;
                 token_idx_in_rank < rank_count; ++ token_idx_in_rank) {
                const uint32_t src_token_topk_idx =
                    *workspace.get_src_token_topk_idx_ptr(
                        current_expert_idx, current_rank_in_expert_idx,
                        token_idx_in_rank);
                const uint32_t src_token_idx =
                    src_token_topk_idx / kNumTopk;
                const uint32_t src_topk_idx =
                    src_token_topk_idx % kNumTopk;
                const uint32_t token_idx_in_expert =
                    rank_prefix + token_idx_in_rank;
                const uint32_t pool_token_idx =
                    expert_pool_block_offset * BLOCK_M +
                    token_idx_in_expert;
                const uint32_t pool_block_idx =
                    expert_pool_block_offset +
                    token_idx_in_expert / BLOCK_M;
                const uint32_t token_idx_in_block =
                    token_idx_in_expert % BLOCK_M;

                if (cute::elect_one_sync())
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(
                            input_token_buffer
                                .get_data_buffer(src_token_idx)
                                .get_base_ptr(),
                            current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                __syncwarp();

                constexpr uint32_t kNumSFFloats = kHidden / 128;
                const auto remote_sf_ptr = sym_buffer.map(
                    input_sf_buffer.get_data_buffer(src_token_idx)
                        .get_base_ptr<float>(),
                    current_rank_in_expert_idx);
                const auto local_sf_ptr =
                    l1_sf_buffer.get_base_ptr<float>();
                const uint32_t sf_pool_token_idx =
                    pool_block_idx * SF_BLOCK_M + token_idx_in_block;
                #pragma unroll
                for (uint32_t sf_iter = 0;
                     sf_iter < math::constexpr_ceil_div(
                         kNumSFFloats, 32u); ++ sf_iter) {
                    const uint32_t sf_idx = sf_iter * 32 + lane_idx;
                    if (sf_idx < kNumSFFloats)
                        local_sf_ptr[
                            sf_idx * kNumPaddedSFPoolTokens +
                            sf_pool_token_idx] = remote_sf_ptr[sf_idx];
                }
                __syncwarp();

                if (lane_idx == 0) {
                    const auto weight_local_ptr =
                        input_topk_weights_buffer.get_base_ptr<float>() +
                        src_token_topk_idx;
                    *l1_topk_weights_buffer
                        .get_data_buffer(pool_token_idx)
                        .get_base_ptr<float>() =
                        *sym_buffer.map(
                            weight_local_ptr,
                            current_rank_in_expert_idx);
                }
                __syncwarp();

                if (lane_idx == 0) {
                    ptx::mbarrier_arrive_and_set_tx(
                        pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(
                        pull_mbarrier, pull_mbarrier_phase);
                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx)
                            .get_base_ptr(),
                        pull_buffer.get_base_ptr(),
                        pull_buffer.get_num_bytes());
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) = {
                        current_rank_in_expert_idx,
                        src_token_idx, src_topk_idx};
                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(pool_block_idx), 1);
                }
                __syncwarp();
            }
        }
#else
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int      current_expert_idx = -1;
#ifdef DG_MEGA_MOE_DISPATCH_FOLLOW_EXPERT_SCHEDULE
        int      current_schedule_pos = -1;
#endif
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
                int old_expert_idx = current_expert_idx;
                while (token_idx >= expert_end_idx) {
#ifdef DG_MEGA_MOE_DISPATCH_FOLLOW_EXPERT_SCHEDULE
                    if (++ current_schedule_pos >=
                        scheduler.get_num_scheduled_experts())
                        break;
                    current_expert_idx = static_cast<int>(
                        scheduler.get_scheduled_expert_idx(
                            static_cast<uint32_t>(current_schedule_pos)));
                    expert_pool_block_offset = scheduler.get_pool_block_offset(
                        static_cast<uint32_t>(current_expert_idx));
#else
                    if (++ current_expert_idx >= kNumExpertsPerRank)
                        break;
                    expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
#endif
                    expert_start_idx = expert_end_idx;
                    expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
                }
#ifdef DG_MEGA_MOE_DISPATCH_FOLLOW_EXPERT_SCHEDULE
                if (current_schedule_pos >=
                    scheduler.get_num_scheduled_experts())
                    break;
#else
                if (current_expert_idx >= kNumExpertsPerRank)
                    break;
#endif

                if (old_expert_idx != current_expert_idx) {
                    old_expert_idx = current_expert_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        stored_rank_count[i] = j < kNumRanks ?
                            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                    }
                }

                uint32_t current_rank_in_expert_idx;
                uint32_t token_idx_in_rank;
                const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
                uint32_t pool_token_idx_in_expert = token_idx_in_expert;
#ifdef DG_MEGA_MOE_RANK_MAJOR_POOL
                // Layout-only rank-major mapping.  Keep the original global
                // token-stride warp scheduler and per-token WQE path intact;
                // only map the expert-local linear slot into a contiguous
                // source-rank segment.
                DG_STATIC_ASSERT(
                    kNumRanks <= 32,
                    "Layout-only rank-major mapping supports at most one warp of ranks");
                uint32_t physical_inclusive_rank_count = stored_rank_count[0];
                #pragma unroll
                for (uint32_t offset = 1; offset < 32; offset <<= 1) {
                    const uint32_t other = __shfl_up_sync(
                        0xffffffff, physical_inclusive_rank_count, offset);
                    if (lane_idx >= offset)
                        physical_inclusive_rank_count += other;
                }

                const uint32_t owner_mask = __ballot_sync(
                    0xffffffff, token_idx_in_expert < physical_inclusive_rank_count);
                DG_DEVICE_ASSERT(owner_mask != 0);
                current_rank_in_expert_idx = __ffs(owner_mask) - 1;
                token_idx_in_rank = token_idx_in_expert;
                if (current_rank_in_expert_idx != 0)
                    token_idx_in_rank -= __shfl_sync(
                        0xffffffff, physical_inclusive_rank_count,
                        current_rank_in_expert_idx - 1);

                uint32_t physical_rank_prefix = 0;
                if (current_rank_in_expert_idx != 0)
                    physical_rank_prefix = __shfl_sync(
                        0xffffffff, physical_inclusive_rank_count,
                        current_rank_in_expert_idx - 1);
                pool_token_idx_in_expert =
                    physical_rank_prefix + token_idx_in_rank;
#else
                // Round-robin rank selection (identical to SM100)
                uint32_t remaining[kNumRanksPerLane];
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] = stored_rank_count[i];
                uint32_t offset = 0;
                uint32_t slot_idx = token_idx_in_expert;
                while (true) {
                    uint32_t num_actives_in_lane = 0;
                    uint32_t min_in_lane = 0xffffffff;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        num_actives_in_lane += remaining[i] > 0;
                        if (remaining[i] > 0)
                            min_in_lane = cute::min(min_in_lane, remaining[i]);
                    }
                    const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                    const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                    const uint32_t num_round_tokens = length * num_active_ranks;
                    if (slot_idx < num_round_tokens) {
                        const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                        uint32_t num_seen_ranks = 0;
                        current_rank_in_expert_idx = 0;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                            const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                            const uint32_t num_active_lanes = __popc(mask);
                            if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                                current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                            num_seen_ranks += num_active_lanes;
                        }
                        token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                        break;
                    }
                    slot_idx -= num_round_tokens;
                    offset += length;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                        remaining[i] -= cute::min(remaining[i], length);
                }
#endif

#ifdef DG_MEGA_MOE_DISPATCH_GATEWAY_V3
                const bool entry_is_inter =
                    current_rank_in_expert_idx / DG_MEGA_MOE_NVL_PEERS !=
                    sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
                const uint32_t src_token_topk_idx = entry_is_inter
                    ? *workspace.get_gateway_landing_ptr(
                          current_rank_in_expert_idx %
                              DG_MEGA_MOE_NVL_PEERS,
                          current_expert_idx, token_idx_in_rank)
                    : *workspace.get_src_token_topk_idx_ptr(
                          current_expert_idx, current_rank_in_expert_idx,
                          token_idx_in_rank);
#else
                const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
#endif
                const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
                const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

                const uint32_t pool_token_idx =
                    expert_pool_block_offset * BLOCK_M + pool_token_idx_in_expert;

#ifdef DG_MEGA_MOE_INTERNODE
                // Source rank on another node: token/SF/weight travel over IBGDA verbs
                // (RDMA READ) instead of NVLink P2P. Decided once per pulled token.
                const bool tok_is_inter =
                    current_rank_in_expert_idx / DG_MEGA_MOE_NVL_PEERS != sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
                // The optimized path shares QP(src_rank, local_expert) across
                // dispatch warps; completion waits target only the caller's
                // reserved batch.  Legacy keeps its per-warp QP for A/B.
                const int inter_qp_id = kDispatchExpertReady ?
                    current_expert_idx : static_cast<int>(sm_idx * kNumDispatchWarps + warp_idx);
                const auto inter_staging = dispatch_staging_buffer
                    .get_data_buffer(pool_token_idx).get_base_ptr<float>();
#endif

                // Pull token data. Overlap a remote TMA load with SF copy and
                // then use TMA store to materialize the local L1 input.
                if (cute::elect_one_sync()) {
#ifdef DG_MEGA_MOE_INTERNODE
                    if (tok_is_inter) {
                        // Inter-node: RDMA READ token straight into the local L1 pool
                        // (skipping smem bounce and both TMAs), plus SF & routing weight
                        // into the per-warp staging row.  The optimized path reserves the
                        // three READs together, rings one doorbell, and waits for that
                        // batch's completion index; legacy uses three posts plus quiet.
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                        const uint64_t profile_remote_read_start = clock64();
#endif
                        if constexpr (kDispatchExpertReady) {
                            const comm::ibgda::GetRequest read_requests[3] = {
                                {
                                    reinterpret_cast<uint64_t>(l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr()),
                                    reinterpret_cast<uint64_t>(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr()),
                                    pull_buffer.get_num_bytes()
                                },
                                {
                                    reinterpret_cast<uint64_t>(inter_staging),
                                    reinterpret_cast<uint64_t>(input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>()),
                                    (kHidden / 128) * sizeof(float)
                                },
                                {
                                    reinterpret_cast<uint64_t>(inter_staging + kHidden / 128),
                                    reinterpret_cast<uint64_t>(input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx),
                                    sizeof(float)
                                }
                            };
                            const auto completion_idx = comm::ibgda::get_batch_thread(
                                read_requests, static_cast<int>(current_rank_in_expert_idx), inter_qp_id);
                            comm::ibgda::wait_until(
                                static_cast<int>(current_rank_in_expert_idx), inter_qp_id, completion_idx);
                        } else {
                            comm::ibgda::get_thread(
                                reinterpret_cast<uint64_t>(l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr()),
                                reinterpret_cast<uint64_t>(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr()),
                                pull_buffer.get_num_bytes(),
                                static_cast<int>(current_rank_in_expert_idx), inter_qp_id);
                            comm::ibgda::get_thread(
                                reinterpret_cast<uint64_t>(inter_staging),
                                reinterpret_cast<uint64_t>(input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>()),
                                (kHidden / 128) * sizeof(float),
                                static_cast<int>(current_rank_in_expert_idx), inter_qp_id);
                            comm::ibgda::get_thread(
                                reinterpret_cast<uint64_t>(inter_staging + kHidden / 128),
                                reinterpret_cast<uint64_t>(input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx),
                                sizeof(float),
                                static_cast<int>(current_rank_in_expert_idx), inter_qp_id);
                            comm::ibgda::quiet(
                                static_cast<int>(current_rank_in_expert_idx), inter_qp_id);
                        }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                        profile_remote_read_cycles +=
                            clock64() - profile_remote_read_start;
                        ++ profile_remote_read_count;
#endif
                    } else
#endif
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                }
                __syncwarp();

                // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
                constexpr uint32_t kNumSFFloats = kHidden / 128;
                DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
                const auto input_sf_local = input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>();
                const auto remote_sf_ptr = sym_buffer.map(input_sf_local, current_rank_in_expert_idx);
                const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
                const uint32_t pool_block_idx =
                    expert_pool_block_offset + pool_token_idx_in_expert / BLOCK_M;
                const uint32_t token_idx_in_block =
                    pool_token_idx_in_expert % BLOCK_M;
                const uint32_t sf_pool_token_idx = pool_block_idx * SF_BLOCK_M + token_idx_in_block;
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats) {
#ifdef DG_MEGA_MOE_INTERNODE
                        // Inter-node: SF 已由 RDMA READ 落进 pool-token 专属暂存行。
                        // SymmBuffer allocation zeroes the entire buffer before launch, so
                        // the cache line may already reside in L2 when the RNIC overwrites
                        // it. `ld.global.cv` invalidates the matching L2 line and refetches
                        // the system-memory value written before the per-QP quiet completed.
                        const float sf_val = tok_is_inter
                            ? __ldcv(inter_staging + j)
                            : remote_sf_ptr[j];
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = sf_val;
#else
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
#endif
                    }
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    const auto weight_local_ptr =
                        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx;
#ifdef DG_MEGA_MOE_INTERNODE
                    // Inter-node: 路由权重已随 SF 一起 READ 进暂存行(紧跟 SF 之后)。
                    const float weight = tok_is_inter
                        ? __ldcv(inter_staging + kHidden / 128)
                        : *sym_buffer.map(weight_local_ptr, current_rank_in_expert_idx);
#else
                    const float weight = *sym_buffer.map(weight_local_ptr, current_rank_in_expert_idx);
#endif
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
#ifdef DG_MEGA_MOE_INTERNODE
                    if (tok_is_inter) {
                        // Token already delivered into the L1 pool by the blocking get;
                        // no smem bounce to flush, just publish metadata and arrival.
                        *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                            {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
                        ptx::red_add_rel(
                            workspace.get_l1_arrival_count_ptr(pool_block_idx), 1);
                    } else
#endif
                    {
                        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                        ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                        ptx::tma_store_1d(
                            l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                            pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                        *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                            {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                        cute::tma_store_arrive();
                        ptx::tma_store_wait<0>();
                        ptx::red_add_rel(
                            workspace.get_l1_arrival_count_ptr(pool_block_idx), 1);
                    }
                }
                __syncwarp();
            }
#endif

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (lane_idx == 0) {
            atomicMax(phase_profile + kProfileDispatchPull,
                      static_cast<unsigned long long>(clock64() - profile_pull_start));
            atomicMax(phase_profile + kProfileRemoteRead,
                      static_cast<unsigned long long>(profile_remote_read_cycles));
            atomicAdd(phase_profile + kProfileRemoteReadCount,
                      static_cast<unsigned long long>(profile_remote_read_count));
        }
#endif

        // Cleanup workspace, overlapping with combine.
        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
#if defined(DG_MEGA_MOE_DISPATCH_GATEWAY_EAGER) and \
    defined(DG_MEGA_MOE_INTERNODE)
            // Reset the eager-handshake CTA completion counters for the
            // next launch (same protected window as send-count zeroing).
            if (thread_idx < kNumRanks)
                *workspace.get_gateway_direction_done_ptr(thread_idx) = 0;
#endif
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
#ifdef DG_MEGA_MOE_INTERNODE
                // 方案c：recv_count_sum 跨节点不再写入，token 总数改为对 16 个 per-source 槽本地求和
                // (低32位=count)。此读在下方 sync_aligned 之前完成，槽的零化在其之后，顺序有 barrier 保护。
                uint32_t num_recv_tokens = 0;
                for (uint32_t j = 0; j < kNumRanks; ++ j)
                    num_recv_tokens += static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, i));
#else
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
#endif
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                const uint32_t cleanup_expert_pool_block_offset =
                    scheduler.get_pool_block_offset(i);

                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

#ifdef DG_MEGA_MOE_ASYNC_PUBLISHER
                // Pair publication is distributed independently of cleanup
                // ownership.  Wait only for this expert, preserving overlap
                // with combine and with publication of all other experts.
                if (thread_idx == 0) {
                    constexpr int64_t kCleanupTimeoutCycles =
                        60ll * 2000000000ll;
                    const uint64_t wait_start = clock64();
                    const uint64_t expected_launch_epoch = ptx::ld_acq_sys(
                        workspace.get_combine_launch_epoch_ptr());
                    uint64_t done_epoch = ptx::ld_acq_sys(
                        workspace.get_combine_publish_done_epoch_ptr(i));
                    while (done_epoch != expected_launch_epoch) {
                        if (clock64() - wait_start >=
                            kCleanupTimeoutCycles) {
                            printf(
                                "MEGA_MOE_ASYNC_CLEANUP_TIMEOUT "
                                "rank=%u sm=%u expert=%u done_epoch=%llu "
                                "expected_epoch=%llu pair_done=%u "
                                "expected_pairs=%u\\n",
                                sym_buffer.rank_idx, sm_idx, i,
                                static_cast<unsigned long long>(done_epoch),
                                static_cast<unsigned long long>(
                                    expected_launch_epoch),
                                ptx::ld_acq_sys(
                                    workspace
                                        .get_combine_publish_pair_done_count_ptr(
                                            i)),
                                kNumRanks);
                            DG_TRAP_ONLY_DEVICE_ASSERT(false);
                        }
                        done_epoch = ptx::ld_acq_sys(
                            workspace.get_combine_publish_done_epoch_ptr(i));
                    }
                }
                ptx::sync_aligned(
                    kNumDispatchThreads, kDispatchBarrierIdx);
#endif

                if constexpr (kCombineExpertReady) {
                    if (thread_idx == 0) {
                        *workspace.get_combine_posted_block_count_ptr(i) = 0;
                        *workspace.get_combine_dst_rank_mask_ptr(i) = 0;
                    }
                }

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    // Lazy cache entries carry a launch epoch and remain valid
                    // until the producer overwrites them on the next launch.
                    // Do not clear them while a slower SM may still consume
                    // this launch's count.
                    if constexpr (not kLazyExpertCount)
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

#ifndef DG_MEGA_MOE_NO_TAG3
                // Slots carry (epoch << 32 | count) and consumers match on the
                // epoch, so clearing is only needed when the cross-rank tag-3
                // barrier still orders this against the next launch.  Without
                // that barrier we must NOT clear: a faster peer may already
                // have published its next-epoch count into this slot.
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();
#endif

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(
                        cleanup_expert_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(
                        cleanup_expert_pool_block_offset + j) = 0;
                    if constexpr (kCombineFullRow)
                        *combine_full_row_arrival_buffer
                             .get_data_buffer(
                                 cleanup_expert_pool_block_offset + j)
                             .get_base_ptr<uint32_t>() = 0;
                }
                __syncwarp();
            }
        }

        if constexpr (kFuseSharedExpert) {
            if (sm_idx == 0 and thread_idx == 0) {
                *workspace.get_l2_arrival_mask_ptr(kSharedPoolBlockIdx) = 0;
                if constexpr (kCombineFullRow)
                    *combine_full_row_arrival_buffer
                         .get_data_buffer(kSharedPoolBlockIdx)
                         .get_base_ptr<uint32_t>() = 0;
            }
        }

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        const uint64_t profile_cleanup_barrier_start =
            profile_dispatch_leader ? clock64() : 0;
#endif
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, false
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            , phase_profile + kProfileCleanupQuietGlobaltimer,
            phase_profile + kProfileCleanupSyncGlobaltimer,
            phase_profile + kProfileCleanupIbgdaGlobaltimer
#endif
        );
#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (profile_dispatch_leader)
            phase_profile[kProfileCleanupBarrier] =
                clock64() - profile_cleanup_barrier_start;
#endif

    // =====================================================================
    // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
    //   Warps inside `kNumNonEpilogueThreads`: warp 0 loads A + SFA,
    //   warp 1 loads B.
    // =====================================================================
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        auto process_a_sfa_block = [&](const auto& block_phase,
                                       const uint32_t& local_expert_idx,
                                       const uint32_t& num_k_blocks,
                                       const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const bool is_shared_expert =
                local_expert_idx == kSharedExpertSentinel;
            const bool is_shared_l1 =
                is_shared_expert and
                block_phase == sched::BlockPhase::Linear1;
            const auto tensor_map_a_ptr = is_shared_l1
                ? &tensor_map_shared_l1_acts
                : (block_phase == sched::BlockPhase::Linear2
                    ? &tensor_map_l2_acts : &tensor_map_l1_acts);
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;
#ifdef DG_MEGA_MOE_MERGE_AB_LOADER
            const auto tensor_map_b_ptr = is_shared_expert
                ? (block_phase == sched::BlockPhase::Linear2
                    ? &tensor_map_shared_l2_weights
                    : &tensor_map_shared_l1_weights)
                : (block_phase == sched::BlockPhase::Linear2
                    ? &tensor_map_l2_weights : &tensor_map_l1_weights);
            const uint32_t shape_n =
                block_phase == sched::BlockPhase::Linear2
                    ? L2_SHAPE_N : L1_SHAPE_N;
#endif

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // Wait for the pool to be ready
            if (block_phase == sched::BlockPhase::Linear1) {
                if (not is_shared_expert) {
                    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    const auto expected = scheduler.template get_valid_m<false>();
                    while (ptx::ld_acq(ptr) != expected);
                }
            } else {
                constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                if constexpr (kL2ArrivalCounter) {
                    const auto ptr = reinterpret_cast<const uint32_t*>(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx));
                    const uint32_t active_m_wgs = math::ceil_div(
                        scheduler.template get_valid_m<false>(), WG_BLOCK_M);
                    const uint32_t expected =
                        kNumL1BlockNs * active_m_wgs * kWarpgroupSplitN * kL1OutputArrivalParts;
                    while (ptx::ld_acq(ptr) != expected);
                } else {
                    const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                    const uint64_t expected = (kNumL1BlockNs >= 64)
                        ? ~0ull : ((1ull << kNumL1BlockNs) - 1ull);
                    while (ptx::ld_acq_gpu(ptr) != expected);
                }
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (is_shared_l1) {
                    // Pre-dispatch stores input SF row-major.  Load it directly
                    // while the activation tile itself is brought in by TMA.
                    const uint32_t valid_m =
                        scheduler.template get_valid_m<false>();
                    for (uint32_t row = lane_idx; row < BLOCK_M; row += 32) {
                        const uint32_t token_idx = m_block_idx * BLOCK_M + row;
                        smem_sfa[stage_idx][row] = token_idx < valid_m
                            ? __ldg(shared_l1_acts_sf +
                                    token_idx * kNumL1BlockKs + k_block_idx)
                            : 0.0f;
                    }
                    __syncwarp();
                }

                if (cute::elect_one_sync()) {
                    const uint32_t m_idx = is_shared_l1
                        ? m_block_idx * BLOCK_M
                        : pool_block_idx * BLOCK_M;
                    const uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load A
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, 1);

                    // TMA load SFA
                    uint32_t expected_tx_bytes = SMEM_A_SIZE_PER_STAGE;
                    if (is_shared_l1) {
                    } else if (block_phase == sched::BlockPhase::Linear1) {
                        // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            sfa_m_idx, k_block_idx, 1);
                        expected_tx_bytes += BLOCK_M * sizeof(float);
                    } else {
                        // L2 SFA per-64: descriptor box is (block_mn, 1) (see make_tma_sf_desc),
                        // so we must issue two single-group TMAs and place them at smem offsets
                        // 0 and BLOCK_M to match math's load offsets (`+ 0 * BLOCK_M` / `+ 1 * BLOCK_M`).
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            sfa_m_idx, k_block_idx * 2, 1);
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx],
                            smem_sfa[stage_idx] + BLOCK_M,
                            sfa_m_idx, k_block_idx * 2 + 1, 1);
                        expected_tx_bytes += 2 * BLOCK_M * sizeof(float);
                    }

#ifdef DG_MEGA_MOE_MERGE_AB_LOADER
                    const uint32_t n_idx =
                        (is_shared_expert ? 0u : local_expert_idx * shape_n) +
                        n_block_idx * BLOCK_N;
                    if constexpr (LOAD_BLOCK_N <= 256) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx],
                            smem_b[stage_idx], k_idx, n_idx, 1);
                    } else {
                        DG_STATIC_ASSERT(
                            LOAD_BLOCK_N % 256 == 0,
                            "Large B tiles are loaded as 256-column TMA slices");
                        #pragma unroll
                        for (uint32_t b_slice_idx = 0;
                             b_slice_idx < LOAD_BLOCK_N / 256;
                             ++ b_slice_idx) {
                            tma::copy<BLOCK_K, 256, kSwizzleBMode, b_dtype_t>(
                                tensor_map_b_ptr, full_barriers[stage_idx],
                                smem_b[stage_idx] +
                                    b_slice_idx * 256 * BLOCK_K,
                                k_idx, n_idx + b_slice_idx * 256, 1);
                        }
                    }
                    expected_tx_bytes += SMEM_B_SIZE_PER_STAGE;
#endif
                    full_barriers[stage_idx]->arrive_and_expect_tx(
                        expected_tx_bytes);
                }
                __syncwarp();
            }
        };

        if constexpr (kFuseSharedExpert) {
            sm90_fp8_mega_moe_for_each_shared_block<
                BLOCK_M, kNumL1BlockNs, kNumL2BlockNs,
                kNumL1BlockKs, kNumL2BlockKs, kNumSMs,
                kNumPoolBlocks, kSharedExpertSentinel>(
                    scheduler, num_tokens, process_a_sfa_block);
        }

        if constexpr (kSplitPhaseHotPath) {
            sm90_fp8_mega_moe_for_each_block_split(
                scheduler,
                [&](const uint32_t& local_expert_idx,
                    const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    process_a_sfa_block(
                        std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
                        local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
                },
                [&](const uint32_t& local_expert_idx,
                    const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    process_a_sfa_block(
                        std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{},
                        local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
                });
        } else {
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                process_a_sfa_block(block_phase, local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
            });
        }

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

#ifdef DG_MEGA_MOE_ASYNC_PUBLISHER
        // The merged loader leaves this warp free.  Keep one persistent
        // publisher warp per CTA.  Completed L2 M blocks can be posted while
        // other CTAs are still computing later blocks.  Per-expert done
        // epochs, rather than CTA ownership, protect arrival/count cleanup.
        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);

        if constexpr (not kLazyExpertCount)
            scheduler.fetch_expert_recv_count();
        scheduler.prepare_expert_schedule();

        // Spread (local expert, destination rank) pairs across all CTAs.  One
        // warp owns one pair at a time, and therefore exclusively owns
        // QP(dst_rank, local_expert) while publishing that pair.  This keeps
        // all publisher warps useful even when the number of local experts is
        // much smaller than the number of SMs.
        const uint64_t launch_epoch = ptx::ld_acq_sys(
            workspace.get_combine_launch_epoch_ptr());
        DG_TRAP_ONLY_DEVICE_ASSERT(launch_epoch != 0);
        constexpr uint32_t kNumPublishPairTasks =
            kNumExpertsPerRank * kNumRanks;
        constexpr uint32_t kPublishRowsPerBatch = 32;
#if defined(DG_MEGA_MOE_PUBLISHER_DST_GROUPED) and \
    defined(DG_MEGA_MOE_PUBLISHER_CHAIN_POLL)
        // Chain polling on top of dst grouping.  The warp keeps one cursor
        // per chained pair and round-robins, publishing whichever pair has
        // its next M block already arrived instead of spinning on one
        // pair's long-tail block while ready data of later pairs sits
        // idle (the batch-4096 authorization gap).  Per-pair semantics are
        // unchanged: blocks post in order within a pair, ready posts last
        // on the same QP, and each pair/QP is still owned by exactly one
        // warp from start to finish.
        DG_STATIC_ASSERT(kNumSMs >= kNumRanks,
                         "Dst-grouped publisher needs one warp per dst");
        const uint32_t dst_rank_idx = sm_idx % kNumRanks;
        const uint32_t grouped_slot_idx = sm_idx / kNumRanks;
        const uint32_t grouped_num_slots =
            (kNumSMs - 1 - dst_rank_idx) / kNumRanks + 1;
        constexpr uint32_t kMaxChainPairs = math::constexpr_ceil_div(
            kNumExpertsPerRank, kNumSMs / kNumRanks);
        const bool chain_is_inter =
            dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
            sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;

        // Constant-indexed register arrays (fully unrolled accesses).
        uint32_t chain_pending = 0;
        uint32_t chain_expert[kMaxChainPairs];
        uint32_t chain_tokens[kMaxChainPairs];
        uint32_t chain_blocks[kMaxChainPairs];
        uint32_t chain_pool[kMaxChainPairs];
        uint32_t chain_cursor[kMaxChainPairs];
        uint32_t chain_lane_rows[kMaxChainPairs];
        #pragma unroll
        for (uint32_t idx = 0; idx < kMaxChainPairs; ++ idx) {
            const uint32_t e = grouped_slot_idx + idx * grouped_num_slots;
            chain_expert[idx] = e;
            chain_tokens[idx] = 0;
            chain_blocks[idx] = 0;
            chain_pool[idx] = 0;
            chain_cursor[idx] = 0;
            chain_lane_rows[idx] = 0;
            if (e >= kNumExpertsPerRank)
                continue;
            uint32_t pair_tokens = 0;
            if (lane_idx == 0) {
                pair_tokens = static_cast<uint32_t>(ptx::ld_acq_sys(
                    workspace.get_expert_recv_count_ptr(dst_rank_idx, e)));
            }
            chain_tokens[idx] = __shfl_sync(0xffffffff, pair_tokens, 0);
            chain_blocks[idx] = math::ceil_div(
                scheduler.get_num_tokens(e), BLOCK_M);
            chain_pool[idx] = scheduler.get_pool_block_offset(e);
            chain_pending |= 1u << idx;
        }

        const uint64_t chain_wait_start = clock64();
        constexpr int64_t kPublishTimeoutCycles = 60ll * 2000000000ll;
        while (chain_pending != 0) {
            DG_TRAP_ONLY_DEVICE_ASSERT(
                clock64() - chain_wait_start < kPublishTimeoutCycles);
            #pragma unroll
            for (uint32_t idx = 0; idx < kMaxChainPairs; ++ idx) {
                if ((chain_pending & (1u << idx)) == 0)
                    continue;
                const uint32_t local_expert_idx = chain_expert[idx];
                if (chain_tokens[idx] != 0) {
                    const uint32_t expert_num_tokens =
                        scheduler.get_num_tokens(local_expert_idx);
                    // Publish every block that has already arrived, then
                    // yield to the next chained pair instead of spinning.
                    while (chain_cursor[idx] < chain_blocks[idx]) {
                        const uint32_t pool_block_idx =
                            chain_pool[idx] + chain_cursor[idx];
                        const auto arrival_ptr =
                            combine_full_row_arrival_buffer
                                .get_data_buffer(pool_block_idx)
                                .get_base_ptr<uint32_t>();
                        if ((ptx::ld_acq_sys(arrival_ptr) &
                             kCombineFullRowPublishReadyBit) == 0)
                            break;
                        const uint32_t m_idx = pool_block_idx * BLOCK_M;
                        const uint32_t valid_m = cute::min(
                            expert_num_tokens -
                                chain_cursor[idx] * BLOCK_M,
                            BLOCK_M);
                        for (uint32_t row_base = 0; row_base < valid_m;
                             row_base += kPublishRowsPerBatch) {
                            const uint32_t row = row_base + lane_idx;
                            bool row_active = false;
                            uint64_t req_rptr = 0;
                            uint64_t req_lptr = 0;
                            if (row < valid_m) {
                                const auto src_metadata =
                                    *workspace.get_token_src_metadata_ptr(
                                        m_idx + row);
                                row_active =
                                    src_metadata.rank_idx == dst_rank_idx;
                                if (row_active) {
                                    ++ chain_lane_rows[idx];
                                    if (chain_is_inter) {
                                        const auto staging_row =
                                            combine_full_row_staging_buffer
                                                .get_data_buffer(
                                                    m_idx + row);
                                        const auto dst_row =
                                            combine_token_buffer
                                                .get_rank_buffer(
                                                    src_metadata.topk_idx)
                                                .get_data_buffer(
                                                    src_metadata.token_idx);
                                        req_rptr =
                                            reinterpret_cast<uint64_t>(
                                                dst_row.get_base_ptr());
                                        req_lptr =
                                            reinterpret_cast<uint64_t>(
                                                staging_row.get_base_ptr());
                                    }
                                }
                            }
                            if (chain_is_inter) {
#if defined(DG_MEGA_MOE_PUBLISHER_RATE_MBPS) or \
    defined(DG_MEGA_MOE_PUBLISHER_DST_RATE_MBPS)
                                const uint32_t rate_rows = __popc(
                                    __ballot_sync(0xffffffff, row_active));
                                const uint32_t rate_bytes =
                                    rate_rows * kHidden *
                                    static_cast<uint32_t>(
                                        sizeof(nv_bfloat16));
#ifdef DG_MEGA_MOE_PUBLISHER_DST_RATE_MBPS
                                if (lane_idx == 0 and rate_rows != 0)
                                    publisher_token_bucket_wait(
                                        &g_publisher_dst_rate_epoch_ns,
                                        g_publisher_dst_rate_bytes +
                                            dst_rank_idx,
                                        DG_MEGA_MOE_PUBLISHER_DST_RATE_MBPS,
                                        (4ull << 20) /
                                            (kNumRanks -
                                             DG_MEGA_MOE_NVL_PEERS),
                                        rate_bytes);
#endif
#ifdef DG_MEGA_MOE_PUBLISHER_RATE_MBPS
                                if (lane_idx == 0 and rate_rows != 0)
                                    publisher_token_bucket_wait(
                                        &g_publisher_rate_epoch_ns,
                                        &g_publisher_rate_bytes,
                                        DG_MEGA_MOE_PUBLISHER_RATE_MBPS,
                                        4ull << 20,
                                        rate_bytes);
#endif
                                __syncwarp();
#endif
                                comm::ibgda::put_nbi_warp_batch_rows(
                                    req_rptr, req_lptr,
                                    kHidden * sizeof(nv_bfloat16),
                                    row_active,
                                    static_cast<int>(dst_rank_idx),
                                    static_cast<int>(local_expert_idx),
                                    static_cast<int>(lane_idx));
                            }
                        }
                        ++ chain_cursor[idx];
                    }
                    if (chain_cursor[idx] != chain_blocks[idx])
                        continue;
                    uint32_t pair_rows = chain_lane_rows[idx];
                    #pragma unroll
                    for (uint32_t offset = 16; offset != 0; offset >>= 1)
                        pair_rows += __shfl_down_sync(
                            0xffffffff, pair_rows, offset);
                    if (lane_idx == 0)
                        DG_DEVICE_ASSERT(pair_rows == chain_tokens[idx]);
                    // Same-node HBM stores become visible before their
                    // ready store.  Inter-node data batches and ready use
                    // the same QP(dst_rank, local_expert), so RC ordering
                    // replaces quiet.
                    __threadfence_system();
                    __syncwarp();
                    const uint32_t global_expert_idx =
                        sym_buffer.rank_idx * kNumExpertsPerRank +
                        local_expert_idx;
                    if (lane_idx == 0) {
                        const auto ready_ptr =
                            workspace.get_combine_ready_epoch_ptr(
                                global_expert_idx);
                        if (chain_is_inter) {
                            comm::ibgda::put_inline_with_credit<uint64_t>(
                                ready_ptr, launch_epoch,
                                static_cast<int>(dst_rank_idx),
                                static_cast<int>(local_expert_idx));
                        } else {
                            ptx::st_relaxed_sys(
                                sym_buffer.map(ready_ptr, dst_rank_idx),
                                launch_epoch);
                        }
                    }
                    __syncwarp();
                }
                // Zero-token pairs complete immediately; publishing pairs
                // reach here right after their ready is posted.
                if (lane_idx == 0) {
                    const auto old_pair_done = ptx::atomic_add_acq_rel_sys(
                        workspace.get_combine_publish_pair_done_count_ptr(
                            local_expert_idx),
                        1);
                    DG_DEVICE_ASSERT(old_pair_done < kNumRanks);
                    if (old_pair_done + 1 == kNumRanks) {
                        ptx::st_release_sys(
                            workspace.get_combine_publish_done_epoch_ptr(
                                local_expert_idx),
                            launch_epoch);
                    }
                }
                __syncwarp();
                chain_pending &= ~(1u << idx);
            }
        }

        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);
#else
#ifdef DG_MEGA_MOE_PUBLISHER_DST_GROUPED
        // Destination-grouped assignment: one warp serves exactly one dst
        // rank (dst d -> warps {d, d + kNumRanks, ...}), experts strided
        // across that group.  A rate-limit wait on one destination bucket
        // can then only delay pairs already queued on the same bucket,
        // removing the cross-destination head-of-line blocking suspected
        // behind the batch-4096 bistable slow mode.
        DG_STATIC_ASSERT(kNumSMs >= kNumRanks,
                         "Dst-grouped publisher needs one warp per dst");
        const uint32_t dst_rank_idx = sm_idx % kNumRanks;
        const uint32_t grouped_slot_idx = sm_idx / kNumRanks;
        const uint32_t grouped_num_slots =
            (kNumSMs - 1 - dst_rank_idx) / kNumRanks + 1;
        for (uint32_t local_expert_idx = grouped_slot_idx;
             local_expert_idx < kNumExpertsPerRank;
             local_expert_idx += grouped_num_slots) {
            const uint32_t pair_task_idx =
                local_expert_idx * kNumRanks + dst_rank_idx;
#else
        for (uint32_t pair_task_idx = sm_idx;
             pair_task_idx < kNumPublishPairTasks;
             pair_task_idx += kNumSMs) {
            const uint32_t local_expert_idx = pair_task_idx / kNumRanks;
            const uint32_t dst_rank_idx = pair_task_idx % kNumRanks;
#endif
            const uint32_t expert_num_tokens =
                scheduler.get_num_tokens(local_expert_idx);
            const uint32_t expert_pool_block_offset =
                scheduler.get_pool_block_offset(local_expert_idx);
            const uint32_t expert_num_m_blocks =
                math::ceil_div(expert_num_tokens, BLOCK_M);
            uint32_t pair_num_tokens = 0;
            if (lane_idx == 0) {
                pair_num_tokens = static_cast<uint32_t>(ptx::ld_acq_sys(
                    workspace.get_expert_recv_count_ptr(
                        dst_rank_idx, local_expert_idx)));
            }
            pair_num_tokens = __shfl_sync(
                0xffffffff, pair_num_tokens, 0);
            const bool pair_is_inter =
                dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
            uint32_t lane_pair_rows = 0;
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            const uint64_t profile_publisher_start =
                lane_idx == 0 ? clock64() : 0;
            const uint64_t profile_pair_start_globaltimer =
                lane_idx == 0 ? ptx::get_globaltimer() : 0;
            uint64_t profile_publisher_wqe_cycles = 0;
            uint32_t profile_submit_count = 0;
            uint32_t profile_max_batch_rows = 0;
            uint64_t profile_data_done_globaltimer =
                profile_pair_start_globaltimer;
#endif

            if (pair_num_tokens != 0) {
                for (uint32_t m_block_idx = 0;
                     m_block_idx < expert_num_m_blocks;
                     ++ m_block_idx) {
                    const uint32_t pool_block_idx =
                        expert_pool_block_offset + m_block_idx;
                    const auto arrival_ptr = combine_full_row_arrival_buffer
                        .get_data_buffer(pool_block_idx)
                        .get_base_ptr<uint32_t>();
                    constexpr int64_t kPublishTimeoutCycles =
                        60ll * 2000000000ll;
                    const uint64_t wait_start = clock64();
                    while ((ptx::ld_acq_sys(arrival_ptr) &
                            kCombineFullRowPublishReadyBit) == 0) {
                        if (clock64() - wait_start >=
                            kPublishTimeoutCycles) {
                            if (lane_idx == 0) {
                                printf(
                                    "MEGA_MOE_ASYNC_PUBLISH_TIMEOUT "
                                    "rank=%u sm=%u expert=%u dst=%u "
                                    "mblock=%u pool_block=%u "
                                    "arrival=0x%x tokens=%u "
                                    "pair_tokens=%u num_mblocks=%u\\n",
                                    sym_buffer.rank_idx, sm_idx,
                                    local_expert_idx, dst_rank_idx,
                                    m_block_idx, pool_block_idx,
                                    ptx::ld_acq_sys(arrival_ptr),
                                    expert_num_tokens, pair_num_tokens,
                                    expert_num_m_blocks);
                            }
                            __syncwarp();
                            DG_TRAP_ONLY_DEVICE_ASSERT(false);
                        }
                    }

#ifdef DG_MEGA_MOE_PUBLISHER_MAX_ACTIVE
                    // Acquire only after the first M block is ready: a pair
                    // waiting on upstream compute must not hold a slot.  The
                    // slot is held until this pair's ready is submitted.
                    if (m_block_idx == 0 and pair_is_inter) {
                        if (lane_idx == 0) {
                            const uint64_t ticket = atomicAdd(
                                reinterpret_cast<unsigned long long*>(
                                    &g_publisher_limiter_ticket_head), 1ull);
                            const uint64_t limiter_wait_start = clock64();
                            while (ptx::ld_acq_gpu(
                                       &g_publisher_limiter_released) +
                                   DG_MEGA_MOE_PUBLISHER_MAX_ACTIVE <=
                                   ticket) {
                                DG_TRAP_ONLY_DEVICE_ASSERT(
                                    clock64() - limiter_wait_start <
                                    kPublishTimeoutCycles);
                            }
                        }
                        __syncwarp();
                    }
#endif

                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t valid_m = cute::min(
                        expert_num_tokens - m_block_idx * BLOCK_M,
                        BLOCK_M);
                    for (uint32_t row_base = 0; row_base < valid_m;
                         row_base += kPublishRowsPerBatch) {
                        const uint32_t row = row_base + lane_idx;
                        bool row_active = false;
                        uint64_t req_rptr = 0;
                        uint64_t req_lptr = 0;
                        if (row < valid_m) {
                            const auto src_metadata =
                                *workspace.get_token_src_metadata_ptr(
                                    m_idx + row);
                            row_active =
                                src_metadata.rank_idx == dst_rank_idx;
                            if (row_active) {
                                ++ lane_pair_rows;
#ifdef DG_MEGA_MOE_ASYNC_STAGE_LOCAL_ROWS
                                const auto staging_row =
                                    combine_full_row_staging_buffer
                                        .get_data_buffer(m_idx + row);
                                const auto dst_row = combine_token_buffer
                                    .get_rank_buffer(src_metadata.topk_idx)
                                    .get_data_buffer(src_metadata.token_idx);
                                req_lptr = reinterpret_cast<uint64_t>(
                                    staging_row.get_base_ptr());
                                req_rptr = reinterpret_cast<uint64_t>(
                                    pair_is_inter
                                        ? dst_row.get_base_ptr()
                                        : sym_buffer.map(
                                              dst_row.get_base_ptr(),
                                              dst_rank_idx));
#else
                                if (pair_is_inter) {
                                    const auto staging_row =
                                        combine_full_row_staging_buffer
                                            .get_data_buffer(m_idx + row);
                                    const auto dst_row = combine_token_buffer
                                        .get_rank_buffer(
                                            src_metadata.topk_idx)
                                        .get_data_buffer(
                                            src_metadata.token_idx);
                                    req_rptr = reinterpret_cast<uint64_t>(
                                        dst_row.get_base_ptr());
                                    req_lptr = reinterpret_cast<uint64_t>(
                                        staging_row.get_base_ptr());
                                }
#endif
                            }
                        }

                        if (pair_is_inter) {
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                            const uint32_t batch_active_mask =
                                __ballot_sync(0xffffffff, row_active);
                            const uint32_t batch_active_rows =
                                __popc(batch_active_mask);
                            const uint64_t profile_wqe_start =
                                lane_idx == 0 ? clock64() : 0;
#endif
#if defined(DG_MEGA_MOE_PUBLISHER_RATE_MBPS) or \
    defined(DG_MEGA_MOE_PUBLISHER_DST_RATE_MBPS)
                            const uint32_t rate_rows = __popc(
                                __ballot_sync(0xffffffff, row_active));
                            const uint32_t rate_bytes =
                                rate_rows * kHidden *
                                static_cast<uint32_t>(sizeof(nv_bfloat16));
#ifdef DG_MEGA_MOE_PUBLISHER_DST_RATE_MBPS
                            if (lane_idx == 0 and rate_rows != 0)
                                publisher_token_bucket_wait(
                                    &g_publisher_dst_rate_epoch_ns,
                                    g_publisher_dst_rate_bytes +
                                        dst_rank_idx,
                                    DG_MEGA_MOE_PUBLISHER_DST_RATE_MBPS,
                                    (4ull << 20) /
                                        (kNumRanks -
                                         DG_MEGA_MOE_NVL_PEERS),
                                    rate_bytes);
#endif
#ifdef DG_MEGA_MOE_PUBLISHER_RATE_MBPS
                            if (lane_idx == 0 and rate_rows != 0)
                                publisher_token_bucket_wait(
                                    &g_publisher_rate_epoch_ns,
                                    &g_publisher_rate_bytes,
                                    DG_MEGA_MOE_PUBLISHER_RATE_MBPS,
                                    4ull << 20,
                                    rate_bytes);
#endif
                            __syncwarp();
#endif
#ifdef DG_MEGA_MOE_ASYNC_PUBLISHER_ROW_DOORBELL
                            // Diagnostic A/B: preserve pair ownership but
                            // submit one row per reserve/doorbell, matching
                            // the old publisher cadence.  All lanes still
                            // cooperate to materialize one registered row.
                            uint32_t pending_rows =
                                __ballot_sync(0xffffffff, row_active);
                            while (pending_rows != 0) {
                                const int owner_lane = __ffs(pending_rows) - 1;
                                const auto row_rptr = __shfl_sync(
                                    0xffffffff, req_rptr, owner_lane);
                                const auto row_lptr = __shfl_sync(
                                    0xffffffff, req_lptr, owner_lane);
                                comm::ibgda::put_nbi_warp(
                                    row_rptr, row_lptr,
                                    kHidden * sizeof(nv_bfloat16),
                                    static_cast<int>(dst_rank_idx),
                                    static_cast<int>(local_expert_idx),
                                    static_cast<int>(lane_idx));
                                pending_rows &= pending_rows - 1;
                            }
#else
                            comm::ibgda::put_nbi_warp_batch_rows(
                                req_rptr, req_lptr,
                                kHidden * sizeof(nv_bfloat16),
                                row_active,
                                static_cast<int>(dst_rank_idx),
                                static_cast<int>(local_expert_idx),
                                static_cast<int>(lane_idx));
#endif
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                            if (lane_idx == 0) {
                                profile_publisher_wqe_cycles +=
                                    clock64() - profile_wqe_start;
                                if (batch_active_rows != 0) {
#ifdef DG_MEGA_MOE_ASYNC_PUBLISHER_ROW_DOORBELL
                                    profile_submit_count += batch_active_rows;
                                    profile_max_batch_rows = 1;
#else
                                    ++ profile_submit_count;
                                    profile_max_batch_rows = cute::max(
                                        profile_max_batch_rows,
                                        batch_active_rows);
#endif
                                }
                                atomicAdd(
                                    phase_profile +
                                        kProfileScatterWriteCount,
                                    static_cast<unsigned long long>(
                                        batch_active_rows));
                            }
#endif
                        }
#ifdef DG_MEGA_MOE_ASYNC_STAGE_LOCAL_ROWS
                        else {
                            // Keep the experimental local path structurally
                            // identical to the RDMA path: math warps only
                            // publish a complete row into staging, and the
                            // pair owner copies it to the destination HBM.
                            // One active lane describes one row; the full warp
                            // then performs a coalesced NVLink copy.
                            uint32_t pending_rows =
                                __ballot_sync(0xffffffff, row_active);
                            constexpr uint32_t kNumRowVecs =
                                kHidden * sizeof(nv_bfloat16) / sizeof(uint4);
                            while (pending_rows != 0) {
                                const int owner_lane = __ffs(pending_rows) - 1;
                                const auto row_src_addr = __shfl_sync(
                                    0xffffffff, req_lptr, owner_lane);
                                const auto row_dst_addr = __shfl_sync(
                                    0xffffffff, req_rptr, owner_lane);
                                const auto row_src = reinterpret_cast<const uint4*>(
                                    row_src_addr);
                                auto row_dst = reinterpret_cast<uint4*>(
                                    row_dst_addr);
                                for (uint32_t vec_idx = lane_idx;
                                     vec_idx < kNumRowVecs;
                                     vec_idx += 32)
                                    row_dst[vec_idx] = row_src[vec_idx];
                                __syncwarp();
                                pending_rows &= pending_rows - 1;
                            }
                        }
#endif
                    }
                }

                #pragma unroll
                for (uint32_t offset = 16; offset != 0; offset >>= 1)
                    lane_pair_rows += __shfl_down_sync(
                        0xffffffff, lane_pair_rows, offset);
                if (lane_idx == 0)
                    DG_DEVICE_ASSERT(lane_pair_rows == pair_num_tokens);

#ifdef DG_MEGA_MOE_PHASE_PROFILE
                const uint64_t profile_ready_start =
                    lane_idx == 0 ? clock64() : 0;
#endif
                // Same-node HBM stores become visible before their ready
                // store.  Inter-node data batches and ready use the same
                // QP(dst_rank, local_expert), so RC ordering replaces quiet.
                __threadfence_system();
                __syncwarp();
                const uint32_t global_expert_idx =
                    sym_buffer.rank_idx * kNumExpertsPerRank +
                    local_expert_idx;
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                if (lane_idx == 0)
                    profile_data_done_globaltimer = ptx::get_globaltimer();
#endif
                if (lane_idx == 0) {
                    const auto ready_ptr =
                        workspace.get_combine_ready_epoch_ptr(
                            global_expert_idx);
                    if (pair_is_inter) {
                        comm::ibgda::put_inline_with_credit<uint64_t>(
                            ready_ptr, launch_epoch,
                            static_cast<int>(dst_rank_idx),
                            static_cast<int>(local_expert_idx));
                    } else {
                        ptx::st_relaxed_sys(
                            sym_buffer.map(ready_ptr, dst_rank_idx),
                            launch_epoch);
                    }
                }
                __syncwarp();
#ifdef DG_MEGA_MOE_PUBLISHER_MAX_ACTIVE
                // Matches the m_block_idx == 0 acquire above exactly once
                // per non-empty inter-node pair.
                if (pair_is_inter and lane_idx == 0)
                    atomicAdd(
                        reinterpret_cast<unsigned long long*>(
                            &g_publisher_limiter_released), 1ull);
#endif
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                if (lane_idx == 0) {
                    atomicMax(
                        phase_profile + kProfileScatterWQE,
                        static_cast<unsigned long long>(
                            profile_publisher_wqe_cycles));
                    atomicMax(
                        phase_profile + kProfileScatterReady,
                        static_cast<unsigned long long>(
                            clock64() - profile_ready_start));
                    atomicMax(
                        phase_profile + kProfileScatterPublish,
                        static_cast<unsigned long long>(
                            clock64() - profile_publisher_start));
                }
#endif
            }

            // Zero-token pairs also complete this task, so cleanup always
            // waits for the fixed kNumRanks count.  The system-scope acq_rel
            // chain gathers all pair publishers; the last pair releases the
            // expert epoch consumed by overlapping dispatch cleanup.
            if (lane_idx == 0) {
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                const auto pair_profile = pair_profile_base +
                    pair_task_idx * kPairProfileNumSlots;
                pair_profile[kPairProfileNumRows] = pair_num_tokens;
                pair_profile[kPairProfileSubmitCount] =
                    profile_submit_count;
                pair_profile[kPairProfileMaxBatchRows] =
                    profile_max_batch_rows;
                pair_profile[kPairProfileWQECycles] =
                    profile_publisher_wqe_cycles;
                pair_profile[kPairProfileStartGlobaltimer] =
                    profile_pair_start_globaltimer;
                pair_profile[kPairProfileDataDoneGlobaltimer] =
                    profile_data_done_globaltimer;
                pair_profile[kPairProfileDoneGlobaltimer] =
                    ptx::get_globaltimer();
                ptx::st_release_sys(
                    reinterpret_cast<uint64_t*>(
                        pair_profile + kPairProfileEpoch),
                    launch_epoch);
#endif
                const auto old_pair_done = ptx::atomic_add_acq_rel_sys(
                    workspace.get_combine_publish_pair_done_count_ptr(
                        local_expert_idx),
                    1);
                DG_DEVICE_ASSERT(old_pair_done < kNumRanks);
                if (old_pair_done + 1 == kNumRanks) {
                    ptx::st_release_sys(
                        workspace.get_combine_publish_done_epoch_ptr(
                            local_expert_idx),
                        launch_epoch);
                }
            }
            __syncwarp();
        }

        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);
#endif  // DG_MEGA_MOE_PUBLISHER_CHAIN_POLL
#elif !defined(DG_MEGA_MOE_MERGE_AB_LOADER)
        auto process_b_block = [&](const auto& block_phase,
                                   const uint32_t& local_expert_idx,
                                   const uint32_t& num_k_blocks,
                                   const uint32_t& m_block_idx,
                                   const uint32_t& n_block_idx) {
            const bool is_shared_expert =
                local_expert_idx == kSharedExpertSentinel;
            const auto tensor_map_b_ptr = is_shared_expert
                ? (block_phase == sched::BlockPhase::Linear2
                    ? &tensor_map_shared_l2_weights
                    : &tensor_map_shared_l1_weights)
                : (block_phase == sched::BlockPhase::Linear2
                    ? &tensor_map_l2_weights : &tensor_map_l1_weights);

            const uint32_t shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t n_idx =
                        (is_shared_expert ? 0u : local_expert_idx * shape_n) +
                        n_block_idx * BLOCK_N;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load B (weight SF is now loaded directly by math warps from global)
                    if constexpr (LOAD_BLOCK_N <= 256) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx],
                            k_idx, n_idx, 1);
                    } else {
                        DG_STATIC_ASSERT(LOAD_BLOCK_N % 256 == 0,
                                         "Large B tiles are loaded as 256-column TMA slices");
                        #pragma unroll
                        for (uint32_t b_slice_idx = 0; b_slice_idx < LOAD_BLOCK_N / 256; ++ b_slice_idx) {
                            tma::copy<BLOCK_K, 256, kSwizzleBMode, b_dtype_t>(
                                tensor_map_b_ptr, full_barriers[stage_idx],
                                smem_b[stage_idx] + b_slice_idx * 256 * BLOCK_K,
                                k_idx, n_idx + b_slice_idx * 256, 1);
                        }
                    }

                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        };

        if constexpr (kFuseSharedExpert) {
            sm90_fp8_mega_moe_for_each_shared_block<
                BLOCK_M, kNumL1BlockNs, kNumL2BlockNs,
                kNumL1BlockKs, kNumL2BlockKs, kNumSMs,
                kNumPoolBlocks, kSharedExpertSentinel>(
                    scheduler, num_tokens, process_b_block);
        }

        scheduler.for_each_block(process_b_block);
#endif

    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Idle non-epilogue warps (kNumDispatchWarps+2, +3). They must still
        // participate in the warpgroup-collective `setmaxnreg.dec.sync.aligned`
        // so that the math warpgroup's `warpgroup_reg_alloc` can succeed.
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
    // =====================================================================
    // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
    // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        const bool profile_math_leader =
            epilogue_wg_idx == 0 and warp_idx_in_wg == 0 and lane_idx == 0;
        uint64_t profile_l1_cycles = 0;
        uint64_t profile_l2_cycles = 0;
        uint64_t profile_scatter_cycles = 0;
        uint64_t profile_scatter_publish_cycles = 0;
        uint64_t profile_scatter_staging_cycles = 0;
        uint64_t profile_scatter_arrival_cycles = 0;
        uint64_t profile_scatter_wqe_cycles = 0;
        uint64_t profile_scatter_ready_cycles = 0;
        uint64_t profile_scatter_ready_mask_cycles = 0;
        uint64_t profile_scatter_ready_atomic_cycles = 0;
        uint64_t profile_scatter_ready_fence_cycles = 0;
        uint64_t profile_scatter_ready_notify_cycles = 0;
        uint64_t profile_scatter_ready_sync_cycles = 0;
        uint32_t profile_l1_block_count = 0;
        uint32_t profile_l2_block_count = 0;
#endif

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        // When the two N-split warpgroups share a single per-64 SF group they
        // also stage into ONE shared row-major L1-output tile (stride
        // L1_OUT_BLOCK_N), each writing its own WG_L1_OUT_BLOCK_N-column half,
        // so a single combined TMA store matches the host descriptor box.
        constexpr uint32_t WG_SMEM_CD_L1_STRIDE_N =
            kSplitNSharesSF ? L1_OUT_BLOCK_N : WG_L1_OUT_BLOCK_N;
        constexpr uint32_t WG_SMEM_CD_L2_STRIDE_N = WG_BLOCK_N;

        // Sync with dispatch in the full communication path.
        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);

        auto process_math_block = [&](const auto& block_phase,
                                      const uint32_t& local_expert_idx,
                                      const uint32_t& num_k_blocks,
                                      const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const bool is_shared_expert =
                local_expert_idx == kSharedExpertSentinel;
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            const uint64_t profile_math_block_start =
                profile_math_leader ? clock64() : 0;
            uint64_t profile_scatter_block_start = 0;
            uint64_t profile_scatter_block_cycles = 0;
            uint64_t profile_scatter_publish_block_cycles = 0;
            uint64_t profile_scatter_staging_block_cycles = 0;
            uint64_t profile_scatter_arrival_block_cycles = 0;
            uint64_t profile_scatter_wqe_block_cycles = 0;
            uint64_t profile_scatter_ready_block_cycles = 0;
            uint64_t profile_scatter_ready_mask_block_cycles = 0;
            uint64_t profile_scatter_ready_atomic_block_cycles = 0;
            uint64_t profile_scatter_ready_fence_block_cycles = 0;
            uint64_t profile_scatter_ready_notify_block_cycles = 0;
            uint64_t profile_scatter_ready_sync_block_cycles = 0;
            bool profile_expert_ready_published = false;
            uint64_t profile_expert_ready_epoch = 0;
            uint64_t profile_expert_dst_rank_mask = 0;
            uint64_t profile_expert_publish_globaltimer = 0;
#endif
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const uint32_t epilogue_wg_m_idx = epilogue_wg_idx / kWarpgroupSplitN;
            const uint32_t epilogue_wg_n_idx = epilogue_wg_idx - epilogue_wg_m_idx * kWarpgroupSplitN;
            const uint32_t wg_n_offset = epilogue_wg_n_idx * WG_BLOCK_N;
            const uint32_t wg_l1_out_n_offset = epilogue_wg_n_idx * WG_L1_OUT_BLOCK_N;
            const uint32_t row_base = epilogue_wg_m_idx * WG_BLOCK_M;
            const uint32_t row_offset_r0 = row_base + r_0;
            const uint32_t row_offset_r1 = row_base + r_1;
            const uint32_t sf_n_block_idx = kSplitNSharesSF ? n_block_idx
                : (n_block_idx * kWarpgroupSplitN + epilogue_wg_n_idx);
            const uint32_t smem_a_wg_offset = epilogue_wg_m_idx * WG_BLOCK_M * BLOCK_K;
            const uint32_t smem_b_wg_offset = epilogue_wg_n_idx * WG_BLOCK_N * BLOCK_K;
            // In the shared-tile case the WG stages into the joint L1-output tile
            // at its own column offset (row stride L1_OUT_BLOCK_N); otherwise each
            // WG owns a disjoint contiguous WG_BLOCK_M x WG_L1_OUT_BLOCK_N slice.
            const uint32_t smem_cd_l1_wg_offset = kSplitNSharesSF ? wg_l1_out_n_offset
                : (epilogue_wg_idx * WG_BLOCK_M * WG_L1_OUT_BLOCK_N);
            const uint32_t smem_cd_l2_wg_offset = epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;
            float final_accum[kAccumPerThread] = {};

            if constexpr (kReuseAccumAsFinal) {
                auto prescale_l1_final = [&](const float& scale_a_0, const float& scale_a_1,
                                             const float& gate_sf, const float& up_sf) {
                    const float inv_s0_gate = kFastMath ? math::fast_rcp(scale_a_0 * gate_sf) : 1.0f / (scale_a_0 * gate_sf);
                    const float inv_s1_gate = kFastMath ? math::fast_rcp(scale_a_1 * gate_sf) : 1.0f / (scale_a_1 * gate_sf);
                    const float inv_s0_up = kFastMath ? math::fast_rcp(scale_a_0 * up_sf) : 1.0f / (scale_a_0 * up_sf);
                    const float inv_s1_up = kFastMath ? math::fast_rcp(scale_a_1 * up_sf) : 1.0f / (scale_a_1 * up_sf);
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float inv_s0 = (i & 1u) ? inv_s0_up : inv_s0_gate;
                        const float inv_s1 = (i & 1u) ? inv_s1_up : inv_s1_gate;
                        final_accum[i*4+0] *= inv_s0;
                        final_accum[i*4+1] *= inv_s0;
                        final_accum[i*4+2] *= inv_s1;
                        final_accum[i*4+3] *= inv_s1;
                    }
                };
                auto postscale_l1_final = [&](const float& scale_a_0, const float& scale_a_1,
                                              const float& gate_sf, const float& up_sf) {
                    const float s0_gate = scale_a_0 * gate_sf;
                    const float s1_gate = scale_a_1 * gate_sf;
                    const float s0_up = scale_a_0 * up_sf;
                    const float s1_up = scale_a_1 * up_sf;
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float s0 = (i & 1u) ? s0_up : s0_gate;
                        const float s1 = (i & 1u) ? s1_up : s1_gate;
                        final_accum[i*4+0] *= s0;
                        final_accum[i*4+1] *= s0;
                        final_accum[i*4+2] *= s1;
                        final_accum[i*4+3] *= s1;
                    }
                };
                auto prescale_l2_final = [&](const float& scale_a_0, const float& scale_a_1,
                                             const float& l2_sf) {
                    const float inv_s0 = kFastMath ? math::fast_rcp(scale_a_0 * l2_sf) : 1.0f / (scale_a_0 * l2_sf);
                    const float inv_s1 = kFastMath ? math::fast_rcp(scale_a_1 * l2_sf) : 1.0f / (scale_a_1 * l2_sf);
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] *= inv_s0;
                        final_accum[i*4+1] *= inv_s0;
                        final_accum[i*4+2] *= inv_s1;
                        final_accum[i*4+3] *= inv_s1;
                    }
                };
                auto postscale_l2_final = [&](const float& scale_a_0, const float& scale_a_1,
                                              const float& l2_sf) {
                    const float s0 = scale_a_0 * l2_sf;
                    const float s1 = scale_a_1 * l2_sf;
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] *= s0;
                        final_accum[i*4+1] *= s0;
                        final_accum[i*4+2] *= s1;
                        final_accum[i*4+3] *= s1;
                    }
                };
                auto rescale_l1_final = [&](const float& prev_scale_a_0, const float& prev_scale_a_1,
                                            const float& prev_gate_sf, const float& prev_up_sf,
                                            const float& scale_a_0, const float& scale_a_1,
                                            const float& gate_sf, const float& up_sf) {
                    const float r0_gate = (prev_scale_a_0 * prev_gate_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_0 * gate_sf) : 1.0f / (scale_a_0 * gate_sf));
                    const float r1_gate = (prev_scale_a_1 * prev_gate_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_1 * gate_sf) : 1.0f / (scale_a_1 * gate_sf));
                    const float r0_up = (prev_scale_a_0 * prev_up_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_0 * up_sf) : 1.0f / (scale_a_0 * up_sf));
                    const float r1_up = (prev_scale_a_1 * prev_up_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_1 * up_sf) : 1.0f / (scale_a_1 * up_sf));
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float r0 = (i & 1u) ? r0_up : r0_gate;
                        const float r1 = (i & 1u) ? r1_up : r1_gate;
                        final_accum[i*4+0] *= r0;
                        final_accum[i*4+1] *= r0;
                        final_accum[i*4+2] *= r1;
                        final_accum[i*4+3] *= r1;
                    }
                };
                auto rescale_l2_final = [&](const float& prev_scale_a_0, const float& prev_scale_a_1,
                                            const float& prev_l2_sf,
                                            const float& scale_a_0, const float& scale_a_1,
                                            const float& l2_sf) {
                    const float r0 = (prev_scale_a_0 * prev_l2_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_0 * l2_sf) : 1.0f / (scale_a_0 * l2_sf));
                    const float r1 = (prev_scale_a_1 * prev_l2_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_1 * l2_sf) : 1.0f / (scale_a_1 * l2_sf));
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] *= r0;
                        final_accum[i*4+1] *= r0;
                        final_accum[i*4+2] *= r1;
                        final_accum[i*4+3] *= r1;
                    }
                };
                auto rescale_l2_act_final = [&](const float& prev_scale_a_0, const float& prev_scale_a_1,
                                                const float& scale_a_0, const float& scale_a_1) {
                    const float r0 = prev_scale_a_0 * (kFastMath ? math::fast_rcp(scale_a_0) : 1.0f / scale_a_0);
                    const float r1 = prev_scale_a_1 * (kFastMath ? math::fast_rcp(scale_a_1) : 1.0f / scale_a_1);
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] *= r0;
                        final_accum[i*4+1] *= r0;
                        final_accum[i*4+2] *= r1;
                        final_accum[i*4+3] *= r1;
                    }
                };

                if constexpr (kHidden >= 7168) {
                    float prev_scale_a_0 = 1.0f, prev_scale_a_1 = 1.0f;
                    float prev_gate_sf = 1.0f, prev_up_sf = 1.0f, prev_l2_sf = 1.0f;
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        full_barriers[stage_idx]->wait(phase);

                        float scale_a_0_lo, scale_a_1_lo;
                        float scale_a_0_hi, scale_a_1_hi;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                            scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                        } else {
                            scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
                            scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
                            scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
                            scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
                        }

                        constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                        constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                        constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                        constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                        constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;
                        float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            const uint32_t gate_n = sf_n_block_idx / 2u;
                            const uint32_t up_n   = kL1SFGateBlks + gate_n;
                            const float* base = (is_shared_expert
                                ? shared_l1_weights_sf
                                : l1_weights_sf + local_expert_idx * kL1SFPerExpert) +
                                k_block_idx;
                            gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                            up_sf   = __ldg(base + up_n   * kL1SFKBlocks);
                        } else {
                            const float* base = is_shared_expert
                                ? shared_l2_weights_sf
                                : l2_weights_sf + local_expert_idx * kL2SFPerExpert;
                            l2_sf = __ldg(base + sf_n_block_idx * kL2SFKBlocks +
                                               k_block_idx);
                        }

                        if (block_phase == sched::BlockPhase::Linear1) {
                            if (k_block_idx != 0)
                                rescale_l1_final(prev_scale_a_0, prev_scale_a_1,
                                                 prev_gate_sf, prev_up_sf,
                                                 scale_a_0_lo, scale_a_1_lo,
                                                 gate_sf, up_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            prev_scale_a_0 = scale_a_0_lo;
                            prev_scale_a_1 = scale_a_1_lo;
                            prev_gate_sf = gate_sf;
                            prev_up_sf = up_sf;
                        } else {
                            if (k_block_idx != 0)
                                rescale_l2_final(prev_scale_a_0, prev_scale_a_1, prev_l2_sf,
                                                 scale_a_0_lo, scale_a_1_lo, l2_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            rescale_l2_act_final(scale_a_0_lo, scale_a_1_lo,
                                                 scale_a_0_hi, scale_a_1_hi);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            prev_scale_a_0 = scale_a_0_hi;
                            prev_scale_a_1 = scale_a_1_hi;
                            prev_l2_sf = l2_sf;
                        }
                    }

                    if (num_k_blocks != 0) {
                        if (block_phase == sched::BlockPhase::Linear1) {
                            postscale_l1_final(prev_scale_a_0, prev_scale_a_1,
                                               prev_gate_sf, prev_up_sf);
                        } else {
                            postscale_l2_final(prev_scale_a_0, prev_scale_a_1, prev_l2_sf);
                        }
                    }
                } else {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        full_barriers[stage_idx]->wait(phase);

                        float scale_a_0_lo, scale_a_1_lo;
                        float scale_a_0_hi, scale_a_1_hi;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                            scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                        } else {
                            scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
                            scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
                            scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
                            scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
                        }

                        constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                        constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                        constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                        constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                        constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;
                        float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            const uint32_t gate_n = sf_n_block_idx / 2u;
                            const uint32_t up_n   = kL1SFGateBlks + gate_n;
                            const float* base = (is_shared_expert
                                ? shared_l1_weights_sf
                                : l1_weights_sf + local_expert_idx * kL1SFPerExpert) +
                                k_block_idx;
                            gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                            up_sf   = __ldg(base + up_n   * kL1SFKBlocks);
                        } else {
                            const float* base = is_shared_expert
                                ? shared_l2_weights_sf
                                : l2_weights_sf + local_expert_idx * kL2SFPerExpert;
                            l2_sf = __ldg(base + sf_n_block_idx * kL2SFKBlocks +
                                               k_block_idx);
                        }

                        if (block_phase == sched::BlockPhase::Linear1) {
                            if (k_block_idx != 0)
                                prescale_l1_final(scale_a_0_lo, scale_a_1_lo, gate_sf, up_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            postscale_l1_final(scale_a_0_lo, scale_a_1_lo, gate_sf, up_sf);
                        } else {
                            if (k_block_idx != 0)
                                prescale_l2_final(scale_a_0_lo, scale_a_1_lo, l2_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            postscale_l2_final(scale_a_0_lo, scale_a_1_lo, l2_sf);
                            prescale_l2_final(scale_a_0_hi, scale_a_1_hi, l2_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            postscale_l2_final(scale_a_0_hi, scale_a_1_hi, l2_sf);
                        }
                    }
                }
            } else {
                float accum[kAccumPerThread];

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);

                    // Read SF (must precede warpgroup_arrive)
                    float scale_a_0_lo, scale_a_1_lo;
                    float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
                    if (block_phase == sched::BlockPhase::Linear1) {
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    } else {
                        // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
                        scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
                        scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
                    }

                    // ----- Block (128, 128) weight SF (loaded directly from global) -----
                    // L1 weight SF shape: (E, 2*IH/128, H/128) MN-major. The N axis is
                    // [gate(IH/128), up(IH/128)]; with the gate/up gran-8 interleave on
                    // the FP8 weight, each logical 128-wide N tile covers 64 rows of gate
                    // plus 64 rows of up taken from the same original 128-row block, so:
                    //     gate_sf_n = sf_n_block_idx / 2
                    //     up_sf_n   = (IH/128) + sf_n_block_idx / 2
                    //
                    // L2 weight SF shape: (E, H/128, IH/128) MN-major. One scalar per
                    // logical 128x128 weight-SF tile, broadcast across the matching
                    // WGMMA accumulators.
                    //
                    // Load the weight scale after the barrier from all WG threads.
                    // This keeps scale loads close to their WGMMA use and lets the
                    // read-only cache coalesce the same-address accesses.
                    constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                    constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                    constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                    constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                    constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;
                    float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
                    if (block_phase == sched::BlockPhase::Linear1) {
                        const uint32_t gate_n = sf_n_block_idx / 2u;
                        const uint32_t up_n   = kL1SFGateBlks + gate_n;
                        const float* base = (is_shared_expert
                            ? shared_l1_weights_sf
                            : l1_weights_sf + local_expert_idx * kL1SFPerExpert) +
                            k_block_idx;
                        gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                        up_sf   = __ldg(base + up_n   * kL1SFKBlocks);
                    } else {
                        const float* base = is_shared_expert
                            ? shared_l2_weights_sf
                            : l2_weights_sf + local_expert_idx * kL2SFPerExpert;
                        l2_sf = __ldg(base + sf_n_block_idx * kL2SFKBlocks +
                                           k_block_idx);
                    }

                    if (block_phase == sched::BlockPhase::Linear1) {
                        if constexpr (kSwapABActive) {
                            auto run_swap_ab_l1 = [&]<uint32_t N_SWAP>() {
                                using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                                constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                                float swap_accum[kSwapAccum];

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + smem_b_wg_offset + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    const float scale_0 = token_0 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + token_0) : 0.0f;
                                    const float scale_1 = token_1 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + token_1) : 0.0f;
                                    final_accum[i * 4 + 0] += scale_0 * gate_sf * swap_accum[i * 4 + 0];
                                    final_accum[i * 4 + 2] += scale_0 * up_sf * swap_accum[i * 4 + 2];
                                    final_accum[i * 4 + 1] += scale_1 * gate_sf * swap_accum[i * 4 + 1];
                                    final_accum[i * 4 + 3] += scale_1 * up_sf * swap_accum[i * 4 + 3];
                                }

                                if (lane_idx == 0)
                                    empty_barriers[stage_idx]->arrive();
                            };

                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if constexpr (kIntermediateHidden <= 2048) {
                                if (n_swap <= 8) {
                                    run_swap_ab_l1.template operator()<8>();
                                } else if (n_swap <= 16) {
                                    run_swap_ab_l1.template operator()<16>();
                                } else if (n_swap <= 32) {
                                    run_swap_ab_l1.template operator()<32>();
                                } else {
                                    run_swap_ab_l1.template operator()<64>();
                                }
                            } else {
                                switch (n_swap) {
                                    case 8:  run_swap_ab_l1.template operator()<8>();  break;
                                    case 16: run_swap_ab_l1.template operator()<16>(); break;
                                    case 24: run_swap_ab_l1.template operator()<24>(); break;
                                    case 32: run_swap_ab_l1.template operator()<32>(); break;
                                    case 40: run_swap_ab_l1.template operator()<40>(); break;
                                    case 48: run_swap_ab_l1.template operator()<48>(); break;
                                    case 56: run_swap_ab_l1.template operator()<56>(); break;
                                    default: run_swap_ab_l1.template operator()<64>(); break;
                                }
                            }
                        } else {
                            // Single per-128 K-block WGMMA group
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            // L1: gate/up alternate at gran=8 along N; each `i` block of 8
                            // cols belongs entirely to one of {gate, up}, so .x and .y
                            // share the same scalar.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float sb = (i & 1u) ? up_sf : gate_sf;
                                final_accum[i*4+0] += scale_a_0_lo * sb * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_lo * sb * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_lo * sb * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_lo * sb * accum[i*4+3];
                            }
                        }
                    } else {
                        if constexpr (kSwapABActive) {
                            DG_STATIC_ASSERT(kL2ActsSFGranK == 64,
                                             "L2 swapAB assumes per-64 activation scales");
                            auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
                                using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                                constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                                float swap_accum[kSwapAccum];

                                auto promote_swap_accum = [&](const uint32_t& sf_group) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                        const uint32_t token_0 = i * 8 + col_idx * 2;
                                        const uint32_t token_1 = token_0 + 1;
                                        const float scale_0 = token_0 < valid_m ?
                                            ptx::ld_shared(smem_sfa[stage_idx] + sf_group * BLOCK_M + token_0) : 0.0f;
                                        const float scale_1 = token_1 < valid_m ?
                                            ptx::ld_shared(smem_sfa[stage_idx] + sf_group * BLOCK_M + token_1) : 0.0f;
                                        final_accum[i * 4 + 0] += scale_0 * l2_sf * swap_accum[i * 4 + 0];
                                        final_accum[i * 4 + 2] += scale_0 * l2_sf * swap_accum[i * 4 + 2];
                                        final_accum[i * 4 + 1] += scale_1 * l2_sf * swap_accum[i * 4 + 1];
                                        final_accum[i * 4 + 3] += scale_1 * l2_sf * swap_accum[i * 4 + 3];
                                    }
                                };

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + smem_b_wg_offset + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                promote_swap_accum(0);

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
                                    const uint32_t k_off = (BLOCK_K / 2) + k * SwapWGMMA::K;
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k_off, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                promote_swap_accum(1);

                                if (lane_idx == 0)
                                    empty_barriers[stage_idx]->arrive();
                            };

                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if constexpr (kIntermediateHidden <= 2048) {
                                if (n_swap <= 8) {
                                    run_swap_ab_l2.template operator()<8>();
                                } else if (n_swap <= 16) {
                                    run_swap_ab_l2.template operator()<16>();
                                } else if (n_swap <= 32) {
                                    run_swap_ab_l2.template operator()<32>();
                                } else {
                                    run_swap_ab_l2.template operator()<64>();
                                }
                            } else {
                                switch (n_swap) {
                                    case 8:  run_swap_ab_l2.template operator()<8>();  break;
                                    case 16: run_swap_ab_l2.template operator()<16>(); break;
                                    case 24: run_swap_ab_l2.template operator()<24>(); break;
                                    case 32: run_swap_ab_l2.template operator()<32>(); break;
                                    case 40: run_swap_ab_l2.template operator()<40>(); break;
                                    case 48: run_swap_ab_l2.template operator()<48>(); break;
                                    case 56: run_swap_ab_l2.template operator()<56>(); break;
                                    default: run_swap_ab_l2.template operator()<64>(); break;
                                }
                            }
                        } else {
                            // L2: split BLOCK_K=128 into two halves (per-64 SFA), each 2 WGMMAs.
                            // First half: K=0..63, SFA = scale_a_*_lo
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            // L2 first half: single scalar `l2_sf` broadcast across N.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0_lo * l2_sf * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_lo * l2_sf * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_lo * l2_sf * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_lo * l2_sf * accum[i*4+3];
                            }

                            // Second half: K=64..127, SFA = scale_a_*_hi
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            // L2 second half: same broadcast scalar `l2_sf`.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0_hi * l2_sf * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_hi * l2_sf * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_hi * l2_sf * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_hi * l2_sf * accum[i*4+3];
                            }
                        }
                    }
                }
            }

            // Skip epilogue when block is past valid M (still must release via empty)
            if (row_base >= valid_m and
                (block_phase == sched::BlockPhase::Linear1 or not kCombineFullRow)) {
                if (block_phase == sched::BlockPhase::Linear1) {
                    if constexpr (not kL2ArrivalCounter)
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {
                    if constexpr (kL2EpilogueRequiresFullSync)
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                return;
            }

            if (block_phase == sched::BlockPhase::Linear1) {
                if constexpr (kSwapABActive) {
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };

                    const uint32_t out_col_base =
                        wg_l1_out_n_offset + warp_idx_in_wg * 8 + row_idx;
                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        if (token_0 < valid_m) {
                            float g0 = final_accum[i * 4 + 0];
                            float u0 = final_accum[i * 4 + 2];
                            clamp_gate(g0);
                            clamp_up(u0);
                            const float weight_0 = is_shared_expert ? 1.0f
                                : *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token_0)
                                    .get_base_ptr<float>();
                            smem_cd_swap_l1_fp32[token_0 * L1_OUT_BLOCK_N + out_col_base] =
                                silu(g0) * u0 * weight_0;
                        }
                        if (token_1 < valid_m) {
                            float g1 = final_accum[i * 4 + 1];
                            float u1 = final_accum[i * 4 + 3];
                            clamp_gate(g1);
                            clamp_up(u1);
                            const float weight_1 = is_shared_expert ? 1.0f
                                : *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token_1)
                                    .get_base_ptr<float>();
                            smem_cd_swap_l1_fp32[token_1 * L1_OUT_BLOCK_N + out_col_base] =
                                silu(g1) * u1 * weight_1;
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    store_l1_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l1_swap_chunk(i);
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    for (uint32_t token = epilogue_thread_idx; token < valid_m; token += kNumEpilogueThreads) {
                        float amax = 0.0f;
                        #pragma unroll
                        for (uint32_t col = 0; col < L1_OUT_BLOCK_N; ++ col) {
                            const float v = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col];
                            amax = cute::max(amax, cute::abs(v));
                        }
                        float2 amax_pair = {amax, amax};
                        float2 sf_pair, sf_inv_pair;
                        sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                        const float sf = sf_pair.x;
                        const float sf_inv = sf_inv_pair.x;

                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        // ROOT-CAUSE FIX: the L2-activation SF pool is strided by SF_BLOCK_M
                        // (=align(BLOCK_M,128)=128), which is how the L2 producer reads it
                        // (sfa_m_idx = pool_block_idx * SF_BLOCK_M) and how the non-swap L1
                        // writes it. This swapAB path used BLOCK_M (64), so for pool_block_idx>=1
                        // the SF landed in the wrong rows -> L2 read stale SF -> every pool block
                        // after the first was corrupted (block 0 was correct because 0*64==0*128).
                        const uint32_t token_idx = pool_block_idx * SF_BLOCK_M + token;
                        sf_base_ptr[n_block_idx * kNumPaddedSFPoolTokens + token_idx] = sf;

                        #pragma unroll
                        for (uint32_t col = 0; col < L1_OUT_BLOCK_N; col += 2) {
                            const float v0 = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 0] * sf_inv;
                            const float v1 = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 1] * sf_inv;
                            const __nv_fp8x2_e4m3 pair(make_float2(v0, v1));
                            auto* ptr = reinterpret_cast<uint16_t*>(
                                smem_cd_swap_l1_fp8 + token * L1_OUT_BLOCK_N + col);
                            *ptr = pair.__x;
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_swap_l1_fp8,
                            n_block_idx * L1_OUT_BLOCK_N,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();

                    if constexpr (kL2ArrivalCounter) {
                        if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)),
                                kWarpgroupSplitN);
                        }
                    } else {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            ptx::red_or_rel_gpu(
                                workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                                1ull << n_block_idx);
                        }
                    }
                    __syncwarp();
                    if constexpr (kL2ArrivalCounter)
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {

                // ---------------- L1 EPILOGUE: activation + FP8 quantize + TMA store ----------------
                // Layout in `final_accum`:
                //   kAccumPerThread/4 chunks, each chunk = 4 floats per thread =
                //   (r0c0, r0c1, r1c0, r1c1).
                //   Gate and up chunks alternate; pair `p` uses chunks 2p and 2p+1.
                //
                // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                float sf_r0, sf_inv_r0;
                float sf_r1, sf_inv_r1;

                float swiglu_r0[kNumPairs][2];
                float swiglu_r1[kNumPairs][2];
                float amax_r0 = 0.0f, amax_r1 = 0.0f;

                auto clamp_gate = [](float& x) {
                    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                        x = cute::min(x, kActivationClamp);
                };
                auto clamp_up = [](float& x) {
                    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                        x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                };
                auto silu = [](float x) -> float {
                    const float e = kFastMath ? __expf(-x) : expf(-x);
                    const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                    return x * sig;
                };

                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const uint32_t gate = 2 * p, up = 2 * p + 1;

                    float g_r0_c0 = final_accum[gate*4 + 0];
                    float g_r0_c1 = final_accum[gate*4 + 1];
                    float g_r1_c0 = final_accum[gate*4 + 2];
                    float g_r1_c1 = final_accum[gate*4 + 3];
                    float u_r0_c0 = final_accum[up*4   + 0];
                    float u_r0_c1 = final_accum[up*4   + 1];
                    float u_r1_c0 = final_accum[up*4   + 2];
                    float u_r1_c1 = final_accum[up*4   + 3];
                    clamp_gate(g_r0_c0);
                    clamp_gate(g_r0_c1);
                    clamp_gate(g_r1_c0);
                    clamp_gate(g_r1_c1);
                    clamp_up(u_r0_c0);
                    clamp_up(u_r0_c1);
                    clamp_up(u_r1_c0);
                    clamp_up(u_r1_c1);

                    if (valid_r0) {
                        swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                        swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                        amax_r0 = cute::max(amax_r0, cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                    } else {
                        swiglu_r0[p][0] = 0.0f;
                        swiglu_r0[p][1] = 0.0f;
                    }
                    if (valid_r1) {
                        swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                        swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                        amax_r1 = cute::max(amax_r1, cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                    } else {
                        swiglu_r1[p][0] = 0.0f;
                        swiglu_r1[p][1] = 0.0f;
                    }
                }

                // Apply token weight: SwiGLU * topk_weight (single load per row)
                const float weight_r0 = valid_r0
                    ? (is_shared_expert ? 1.0f
                        : *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r0)
                            .get_base_ptr<float>())
                    : 0.0f;
                const float weight_r1 = valid_r1
                    ? (is_shared_expert ? 1.0f
                        : *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r1)
                            .get_base_ptr<float>())
                    : 0.0f;
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    swiglu_r0[p][0] *= weight_r0;
                    swiglu_r0[p][1] *= weight_r0;
                    swiglu_r1[p][0] *= weight_r1;
                    swiglu_r1[p][1] *= weight_r1;
                }

                amax_r0 *= cute::abs(weight_r0);
                amax_r1 *= cute::abs(weight_r1);

                // Reduce amax across the 4 col-lanes that share the same row. In the
                // SM90 WGMMA output layout, lanes with the same `lane_idx >> 2` and
                // different `lane_idx & 3` partition the WG-owned output columns for
                // the same r_0/r_1, so this is an INTRA-group reduction
                // (`warp_reduce<4, false>`). Using `<4, true>` would instead merge
                // amax across 8 different rows -- giving wrong per-row SF.
                amax_r0 = math::warp_reduce<4, false>(amax_r0, math::ReduceMax<float>());
                amax_r1 = math::warp_reduce<4, false>(amax_r1, math::ReduceMax<float>());

                // Phase 2: cross-WG amax. When two N-split warpgroups share one
                // per-64 SF group, each WG so far only saw its own
                // WG_L1_OUT_BLOCK_N columns; the true per-row amax spans both
                // halves. Reduce across both warpgroups through a small smem
                // scratch (carved from the upper, currently-unused half of the
                // CD staging region) so BOTH WGs quantize with the SAME SF.
                if constexpr (kSplitNSharesSF) {
                    float* amax_scratch = reinterpret_cast<float*>(
                        reinterpret_cast<uint8_t*>(smem_cd_l1) + SMEM_CD_SIZE / 2);
                    #pragma unroll
                    for (uint32_t i = epilogue_thread_idx; i < BLOCK_M; i += kNumEpilogueThreads)
                        amax_scratch[i] = 0.0f;
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (col_idx == 0) {
                        atomicMax(reinterpret_cast<unsigned int*>(&amax_scratch[r_0]), __float_as_uint(amax_r0));
                        atomicMax(reinterpret_cast<unsigned int*>(&amax_scratch[r_1]), __float_as_uint(amax_r1));
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    amax_r0 = amax_scratch[r_0];
                    amax_r1 = amax_scratch[r_1];
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }

                // Compute SF and inverse SF for each row
                float2 amax_pair = {amax_r0, amax_r1};
                float2 sf_pair, sf_inv_pair;
                sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                sf_r0 = sf_pair.x; sf_inv_r0 = sf_inv_pair.x;
                sf_r1 = sf_pair.y; sf_inv_r1 = sf_inv_pair.y;

                // Quantize and write to the shared-memory staging tile.
                auto* smem_cd_l1_wg = smem_cd_l1 + smem_cd_l1_wg_offset;
                DG_STATIC_ASSERT(kNumPairs % 2 == 0, "L1 staging stores two 8-byte chunks at once");
                #pragma unroll
                for (uint32_t p_base = 0; p_base < kNumPairs; p_base += 2) {
                    uint16_t r0_bits[2], r1_bits[2];
                    #pragma unroll
                    for (uint32_t q = 0; q < 2; ++ q) {
                        const uint32_t p = p_base + q;
                        const float v00 = swiglu_r0[p][0] * sf_inv_r0;
                        const float v01 = swiglu_r0[p][1] * sf_inv_r0;
                        const float v10 = swiglu_r1[p][0] * sf_inv_r1;
                        const float v11 = swiglu_r1[p][1] * sf_inv_r1;

                        const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                        const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
                        r0_bits[q] = valid_r0 ? r0_pair.__x : 0u;
                        r1_bits[q] = valid_r1 ? r1_pair.__x : 0u;
                    }

                    #pragma unroll
                    for (uint32_t q = 0; q < 2; ++ q) {
                        const uint32_t p = p_base + q;
                        const uint32_t col = p * 8 + col_idx * 2;
                        auto* p0 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1_wg + r_0 * WG_SMEM_CD_L1_STRIDE_N + col);
                        auto* p1 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1_wg + r_1 * WG_SMEM_CD_L1_STRIDE_N + col);
                        if (valid_r0)
                            *p0 = r0_bits[q];
                        if (valid_r1)
                            *p1 = r1_bits[q];
                    }
                }

                // Write SF as float at `[token, n_block_idx]` in L2 acts SF buffer (per-64 layout).
                // Each row is contributed by lanes col_idx in {0..3}; only col_idx == 0 writes.
                // In the shared-SF split both warpgroups own the same per-64 group and rows, so
                // only the first N-split warpgroup publishes the SF slot to avoid a write race.
                if (col_idx == 0 and (not kSplitNSharesSF or epilogue_wg_n_idx == 0)) {
                    auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                    // SF buffer is (kNumPaddedSFPoolTokens x kIntermediateHidden/64), MN-major:
                    //   addr[k_idx * num_padded_sf_pool_tokens + token_idx]
                    const uint32_t token_r0 = pool_block_idx * SF_BLOCK_M + row_offset_r0;
                    const uint32_t token_r1 = pool_block_idx * SF_BLOCK_M + row_offset_r1;
                    const uint32_t k_sf_idx = sf_n_block_idx;  // one per-64 post-SwiGLU group
                    if (valid_r0)
                        sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0;
                    if (valid_r1)
                        sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1;
                }

                // Sync the warpgroup before TMA store. In the shared-tile split
                // both N-split warpgroups must finish writing their halves of the
                // joint L1-output tile, so sync across all epilogue threads.
                if constexpr (kSplitNSharesSF)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                else
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                // Issue TMA store of the entire tile. Padding rows beyond
                // `valid_m` are written with stale/garbage FP8 to the L1-output
                // pool buffer, but they are never consumed downstream: the L2
                // GEMM tile loads them, but its NVLink-scatter epilogue is
                // gated by `m_idx_in_block >= valid_m`, and stale SF in the
                // padding rows can produce NaN accumulators that simply stay
                // in registers (only valid rows are converted to BF16 and
                // STSM'd into smem). Using TMA for partial tiles is a large
                // win for low-batch / decode where every tile is partial.
                if constexpr (kSplitNSharesSF) {
                    // One combined store of the joint L1_OUT_BLOCK_N tile, issued
                    // by the first N-split warpgroup once both halves are staged.
                    if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1,
                            out_n_idx,
                            m_idx + row_base);
                        cute::tma_store_arrive();
                    }
                } else {
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_offset;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1 + smem_cd_l1_wg_offset,
                            out_n_idx,
                            m_idx + row_base);
                        cute::tma_store_arrive();
                    }
                }
                __syncwarp();
                ptx::tma_store_wait<0>();

                // Notify L2 that this L1 output (and SF) is ready. Counter mode lets
                // independent WG tiles publish arrivals without the CTA-wide barrier
                // needed before the single bit-mask update.
                if constexpr (kL2ArrivalCounter) {
                    if constexpr (kSplitNSharesSF) {
                        // The combined tile counts for both N-split warpgroups; the
                        // storing warpgroup publishes all kWarpgroupSplitN arrivals
                        // after its TMA store has drained.
                        if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)),
                                kWarpgroupSplitN);
                        }
                    } else if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        ptx::red_add_rel(
                            reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)), 1);
                    }
                } else {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_or_rel_gpu(
                            workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                            1ull << n_block_idx);
                    }
                }
                __syncwarp();
                // In the shared-tile split only the first warpgroup issues and
                // drains the combined TMA store; gate the other warpgroup so it
                // cannot overwrite the joint smem tile in the next block until
                // that store has drained.
                if constexpr (kSplitNSharesSF)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                const uint32_t lane_in_row = lane_idx % 16;
                const uint32_t cols_per_lane = WG_BLOCK_N / 16;

                if constexpr (kCombineFullRow) {
                    // Dispatch no longer uses these shared words after its
                    // pre-pull rendezvous.  Reuse them as CTA-local flags so
                    // legacy full-row mode can bypass the arrival/send path
                    // for local-only pool blocks.
                    if (epilogue_thread_idx == 0) {
                        smem_expert_count[0] = 0;  // contains a remote row
                        smem_expert_count[1] = 0;  // this CTA is last producer
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }

                if constexpr (kSwapABActive) {
                    auto store_bf16 = [&](const uint32_t& token, const uint32_t& col, float value) {
                        smem_cd_l2[smem_cd_l2_wg_offset + token * WG_BLOCK_N + col] =
                            __float2bfloat16_rn(value);
                    };

                    auto store_l2_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        if (token_0 < valid_m) {
                            store_bf16(token_0, r_0, final_accum[i * 4 + 0]);
                            store_bf16(token_0, r_1, final_accum[i * 4 + 2]);
                        }
                        if (token_1 < valid_m) {
                            store_bf16(token_1, r_0, final_accum[i * 4 + 1]);
                            store_bf16(token_1, r_1, final_accum[i * 4 + 3]);
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    store_l2_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l2_swap_chunk(i);
                        }
                    }
                } else {
                    // STSM into smem_cd_l2 (BF16). Reuse SM100 column-swizzle layout.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                        // Each i consumes 8 floats (one 16x256b chunk in SM100 terms).
                        // For SM90 WGMMA layout, 8 floats per i correspond to 2 chunks of 4 floats:
                        //   final_accum[i*8 + (0..3)] = chunk 2i: (r0c0, r0c1, r1c0, r1c1)
                        //   final_accum[i*8 + (4..7)] = chunk 2i+1: same shape
                        const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;

                        auto write_pair = [&](uint32_t row, uint32_t col, uint32_t packed) {
                            auto smem_ptr = smem_cd_l2
                                + smem_cd_l2_wg_offset
                                + row * WG_BLOCK_N
                                + col;
                            // BF16 STS: 2 bf16 elements
                            *reinterpret_cast<uint32_t*>(smem_ptr) = packed;
                        };
                        if (valid_r0) {
                            const uint32_t r0_lo = math::cast_into_bf16_and_pack(
                                final_accum[chunk_lo*4 + 0], final_accum[chunk_lo*4 + 1]);
                            const uint32_t r0_hi = math::cast_into_bf16_and_pack(
                                final_accum[chunk_hi*4 + 0], final_accum[chunk_hi*4 + 1]);
                            write_pair(r_0, chunk_lo * 8 + col_idx * 2, r0_lo);
                            write_pair(r_0, chunk_hi * 8 + col_idx * 2, r0_hi);
                        }
                        if (valid_r1) {
                            const uint32_t r1_lo = math::cast_into_bf16_and_pack(
                                final_accum[chunk_lo*4 + 2], final_accum[chunk_lo*4 + 3]);
                            const uint32_t r1_hi = math::cast_into_bf16_and_pack(
                                final_accum[chunk_hi*4 + 2], final_accum[chunk_hi*4 + 3]);
                            write_pair(r_1, chunk_lo * 8 + col_idx * 2, r1_lo);
                            write_pair(r_1, chunk_hi * 8 + col_idx * 2, r1_hi);
                        }
                    }
                }

#ifdef DG_MEGA_MOE_PHASE_PROFILE
                if (profile_math_leader)
                    profile_scatter_block_start = clock64();
#endif
                // In full-row mode all warpgroups must converge: inactive
                // split-M warpgroups also participate so the CTA-wide staging
                // and last-producer protocol cannot deadlock.
                if constexpr (kCombineFullRow)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                else if constexpr (kSwapABActive)
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                else
                    __syncwarp();

                // Scatter to remote ranks via NVLink (one row per warp-pair)
                // Each warpgroup-warp covers 8 unique rows x 2 (r_0 + r_1 doubled by warps)
                // Lane group of 16 within a warp -> 1 row.
                // Each lane copies `cols_per_lane` BF16 (= cols_per_lane*2 bytes) as one
                // vector. WG_BLOCK_N=128 -> 8 BF16 = uint4; WG_BLOCK_N=64 -> 4 BF16 = uint2.
                using ScatterVec = std::conditional_t<(WG_BLOCK_N <= 64), uint2, uint4>;
                DG_STATIC_ASSERT(cols_per_lane * sizeof(nv_bfloat16) == sizeof(ScatterVec),
                                 "Scatter vector width must match cols_per_lane");

                if (is_shared_expert) {
                    // Shared-expert rows never leave this rank.  Publish them
                    // directly in the final output buffer; the existing
                    // pre-combine grid rendezvous provides the consumer-side
                    // ordering before routed results are accumulated.
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg =
                            warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t token_idx = row_base + row_in_wg;
                        if (token_idx >= valid_m)
                            break;

                        const auto smem_ptr = smem_cd_l2
                            + smem_cd_l2_wg_offset
                            + row_in_wg * WG_BLOCK_N
                            + lane_in_row * cols_per_lane;
                        const auto packed =
                            *reinterpret_cast<const ScatterVec*>(smem_ptr);
                        auto dst_ptr = math::advance_ptr<ScatterVec>(
                            y,
                            (token_idx * kHidden + n_idx + wg_n_offset) *
                                sizeof(nv_bfloat16) +
                                lane_in_row * sizeof(ScatterVec));
                        *dst_ptr = packed;
                    }
                    __threadfence();
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    if (profile_math_leader) {
                        profile_l2_cycles +=
                            clock64() - profile_math_block_start;
                        ++ profile_l2_block_count;
                    }
#endif
                    return;
                }
#if defined(DG_MEGA_MOE_INTERNODE)
                if constexpr (kCombineFullRow) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg =
                            warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = row_base + row_in_wg;
                        if (m_idx_in_block >= valid_m)
                            break;

                        auto smem_ptr = smem_cd_l2
                            + smem_cd_l2_wg_offset
                            + row_in_wg * WG_BLOCK_N
                            + lane_in_row * cols_per_lane;
                        const auto packed = *reinterpret_cast<ScatterVec*>(smem_ptr);

                        const auto src_metadata =
                            *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;
                        const auto dst_token = combine_token_buffer
                            .get_rank_buffer(dst_topk_idx)
                            .get_data_buffer(dst_token_idx);
                        auto dst_ptr = math::advance_ptr<ScatterVec>(
                            dst_token.get_base_ptr(),
                            (n_idx + wg_n_offset) * sizeof(nv_bfloat16) +
                                lane_in_row * sizeof(ScatterVec));
                        const bool row_is_inter =
                            dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                            sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
                        bool row_uses_staging = row_is_inter;
#ifdef DG_MEGA_MOE_ASYNC_STAGE_LOCAL_ROWS
                        row_uses_staging = true;
#endif

                        if (row_uses_staging) {
                            const auto staging_row = combine_full_row_staging_buffer
                                .get_data_buffer(m_idx + m_idx_in_block);
                            auto staging_ptr = math::advance_ptr<ScatterVec>(
                                staging_row.get_base_ptr(),
                                (n_idx + wg_n_offset) * sizeof(nv_bfloat16) +
                                    lane_in_row * sizeof(ScatterVec));
                            *staging_ptr = packed;
                            if (row_is_inter and lane_in_row == 0)
                                atomicExch(smem_expert_count, 1u);
                        } else {
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                    __syncwarp();

#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    uint64_t profile_arrival_start = 0;
                    if (profile_math_leader) {
                        // Includes the initial CTA rendezvous, shared-memory
                        // reads, metadata lookup, and HBM/NVLink row stores.
                        profile_scatter_staging_block_cycles =
                            clock64() - profile_scatter_block_start;
                        profile_arrival_start = clock64();
                    }
#endif

                    // Publish this N-block only after every warpgroup has
                    // completed its HBM fragments.  The system-scope acq_rel
                    // RMW makes the final CTA acquire all earlier producers.
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_thread_idx == 0 and
                        (smem_expert_count[0] != 0 or kCombineExpertReady)) {
                        const auto arrival_ptr = combine_full_row_arrival_buffer
                            .get_data_buffer(pool_block_idx)
                            .get_base_ptr<uint32_t>();
                        const auto old = ptx::atomic_add_acq_rel_sys(arrival_ptr, 1);
                        smem_expert_count[1] = old + 1 == kNumL2BlockNs;
#ifdef DG_MEGA_MOE_ASYNC_PUBLISHER
                        if (smem_expert_count[1] != 0) {
                            // The last N-block publishes a release marker; the
                            // dedicated warp acquires it before reading the
                            // completed full-row staging data.  Reusing the
                            // arrival word avoids another symmetric buffer.
                            ptx::st_release_sys(
                                arrival_ptr,
                                kCombineFullRowPublishReadyBit |
                                    kNumL2BlockNs);
                        }
#endif
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

#ifndef DG_MEGA_MOE_ASYNC_PUBLISHER
                    // The last N-block CTA owns publication.  Its epilogue
                    // warps divide the rows; all lanes in a warp cooperate on
                    // the registered WRITE.  qp_id is the sender local expert.
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    uint64_t profile_publish_start = 0;
                    if (profile_math_leader) {
                        // Includes the CTA rendezvous and system-scope arrival
                        // RMW that elect the final N-block producer.
                        profile_scatter_arrival_block_cycles =
                            clock64() - profile_arrival_start;
                        profile_publish_start = clock64();
                    }
#endif
                    if (smem_expert_count[1] != 0) {
#ifdef DG_MEGA_MOE_COMBINE_BATCH_DOORBELL
                        // Assign each destination rank to exactly one
                        // epilogue warp.  Active lanes then contribute one row
                        // each to a shared QP reservation and one doorbell.
                        for (uint32_t dst_rank_idx = epilogue_warp_idx;
                             dst_rank_idx < kNumRanks;
                             dst_rank_idx += kNumEpilogueWarps) {
                            if (dst_rank_idx / DG_MEGA_MOE_NVL_PEERS ==
                                sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS)
                                continue;

#ifdef DG_MEGA_MOE_COMBINE_COMPACT_BATCH_ROWS
                            // Round-robin pool order normally leaves only a
                            // few rows for this (dst_rank, expert) in each
                            // 32-row lane window.  Compact matches across the
                            // full M block, then submit bounded 8/16-row
                            // batches.  This preserves the physical pool
                            // layout and active-QP parallelism while reducing
                            // reservation/doorbell traffic; the hard cap also
                            // prevents the known 32-WQE same-QP long tail.
                            constexpr uint32_t kCompactBatchRows =
                                DG_MEGA_MOE_COMBINE_COMPACT_BATCH_ROWS;
                            constexpr uint32_t kNumRowGroups =
                                math::constexpr_ceil_div(BLOCK_M, 32u);
                            uint32_t row_masks[kNumRowGroups] = {};
                            uint32_t num_dst_rows = 0;
                            #pragma unroll
                            for (uint32_t group = 0;
                                 group < kNumRowGroups; ++ group) {
                                const uint32_t row = group * 32 + lane_idx;
                                uint32_t row_dst_rank_idx = 0;
                                if (row < valid_m) {
                                    const auto src_metadata =
                                        *workspace.get_token_src_metadata_ptr(
                                            m_idx + row);
                                    row_dst_rank_idx = src_metadata.rank_idx;
                                }
                                const uint32_t mask = __ballot_sync(
                                    0xffffffff,
                                    row < valid_m and
                                        row_dst_rank_idx == dst_rank_idx);
                                row_masks[group] = mask;
                                num_dst_rows += __popc(mask);
                            }

                            for (uint32_t batch_base = 0;
                                 batch_base < num_dst_rows;
                                 batch_base += kCompactBatchRows) {
                                const uint32_t batch_count = cute::min(
                                    num_dst_rows - batch_base,
                                    kCompactBatchRows);
                                const bool active = lane_idx < batch_count;
                                uint32_t selected_row = 0;
                                if (active) {
                                    uint32_t ordinal = batch_base + lane_idx;
                                    #pragma unroll
                                    for (uint32_t group = 0;
                                         group < kNumRowGroups; ++ group) {
                                        const uint32_t group_count =
                                            __popc(row_masks[group]);
                                        if (ordinal < group_count) {
                                            selected_row = group * 32 +
                                                __fns(
                                                    row_masks[group], 0,
                                                    ordinal + 1);
                                            break;
                                        }
                                        ordinal -= group_count;
                                    }
                                }

                                uint64_t src_ptr = 0;
                                uint64_t dst_ptr = 0;
                                if (active) {
                                    const auto src_metadata =
                                        *workspace.get_token_src_metadata_ptr(
                                            m_idx + selected_row);
                                    DG_DEVICE_ASSERT(
                                        src_metadata.rank_idx == dst_rank_idx);
                                    const auto staging_row =
                                        combine_full_row_staging_buffer
                                            .get_data_buffer(
                                                m_idx + selected_row);
                                    const auto dst_row = combine_token_buffer
                                        .get_rank_buffer(
                                            src_metadata.topk_idx)
                                        .get_data_buffer(
                                            src_metadata.token_idx);
                                    src_ptr = reinterpret_cast<uint64_t>(
                                        staging_row.get_base_ptr());
                                    dst_ptr = reinterpret_cast<uint64_t>(
                                        dst_row.get_base_ptr());
                                }
                                comm::ibgda::put_nbi_warp_batch_rows(
                                    dst_ptr, src_ptr,
                                    kHidden * sizeof(nv_bfloat16), active,
                                    static_cast<int>(dst_rank_idx),
                                    static_cast<int>(local_expert_idx),
                                    static_cast<int>(lane_idx));
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                                if (lane_idx == 0)
                                    atomicAdd(
                                        phase_profile +
                                            kProfileScatterWriteCount,
                                        static_cast<unsigned long long>(
                                            batch_count));
#endif
                            }
#else
                            for (uint32_t row_base = 0; row_base < valid_m;
                                 row_base += 32) {
                                const uint32_t row = row_base + lane_idx;
                                uint32_t row_dst_rank_idx = 0;
                                uint32_t row_dst_token_idx = 0;
                                uint32_t row_dst_topk_idx = 0;
                                if (row < valid_m) {
                                    const auto src_metadata =
                                        *workspace.get_token_src_metadata_ptr(m_idx + row);
                                    row_dst_rank_idx = src_metadata.rank_idx;
                                    row_dst_token_idx = src_metadata.token_idx;
                                    row_dst_topk_idx = src_metadata.topk_idx;
                                }

                                const bool active = row < valid_m and
                                    row_dst_rank_idx == dst_rank_idx;
                                uint64_t src_ptr = 0;
                                uint64_t dst_ptr = 0;
                                if (active) {
                                    const auto staging_row =
                                        combine_full_row_staging_buffer
                                            .get_data_buffer(m_idx + row);
                                    const auto dst_row = combine_token_buffer
                                        .get_rank_buffer(row_dst_topk_idx)
                                        .get_data_buffer(row_dst_token_idx);
                                    src_ptr = reinterpret_cast<uint64_t>(
                                        staging_row.get_base_ptr());
                                    dst_ptr = reinterpret_cast<uint64_t>(
                                        dst_row.get_base_ptr());
                                }
                                const uint32_t active_mask = __ballot_sync(
                                    0xffffffff, active);
                                comm::ibgda::put_nbi_warp_batch_rows(
                                    dst_ptr, src_ptr,
                                    kHidden * sizeof(nv_bfloat16), active,
                                    static_cast<int>(dst_rank_idx),
                                    static_cast<int>(local_expert_idx),
                                    static_cast<int>(lane_idx));
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                                if (lane_idx == 0)
                                    atomicAdd(
                                        phase_profile + kProfileScatterWriteCount,
                                        static_cast<unsigned long long>(
                                            __popc(active_mask)));
#endif
                            }
#endif
                        }
#else
                        for (uint32_t row = epilogue_warp_idx;
                             row < valid_m; row += kNumEpilogueWarps) {
                            const auto src_metadata =
                                *workspace.get_token_src_metadata_ptr(m_idx + row);
                            const uint32_t dst_rank_idx = src_metadata.rank_idx;
                            if (dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                                sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS) {
                                const auto staging_row = combine_full_row_staging_buffer
                                    .get_data_buffer(m_idx + row);
                                const auto dst_row = combine_token_buffer
                                    .get_rank_buffer(src_metadata.topk_idx)
                                    .get_data_buffer(src_metadata.token_idx);
                                comm::ibgda::put_nbi_warp(
                                    reinterpret_cast<uint64_t>(dst_row.get_base_ptr()),
                                    reinterpret_cast<uint64_t>(staging_row.get_base_ptr()),
                                    kHidden * sizeof(nv_bfloat16),
                                    static_cast<int>(dst_rank_idx),
                                    static_cast<int>(local_expert_idx),
                                    static_cast<int>(lane_idx));
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                                if (lane_idx == 0)
                                    atomicAdd(phase_profile + kProfileScatterWriteCount, 1ull);
#endif
                            }
                        }
#endif
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    uint64_t profile_ready_start = 0;
                    if (profile_math_leader) {
                        // The final producer builds/posts full-row WQEs; the
                        // following CTA rendezvous is part of this interval.
                        profile_scatter_wqe_block_cycles =
                            clock64() - profile_publish_start;
                        profile_ready_start = clock64();
                    }
#endif
                    if constexpr (kCombineExpertReady) {
                        // Every M block, including a local-only block, reaches
                        // this point after all of its N blocks have published.
                        // Aggregate the source ranks touched by the expert and
                        // let the last M block emit one ready notification per
                        // destination rank.
#ifdef DG_MEGA_MOE_COMBINE_PARALLEL_READY
                        if (epilogue_warp_idx == 0 and smem_expert_count[1] != 0) {
#else
                        if (epilogue_thread_idx == 0 and smem_expert_count[1] != 0) {
#endif
#ifdef DG_MEGA_MOE_COMBINE_PARALLEL_READY
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                            uint64_t profile_ready_subphase_start =
                                profile_math_leader ? clock64() : 0;
#endif
                            uint64_t block_dst_rank_mask = 0;
                            for (uint32_t row = lane_idx; row < valid_m; row += 32) {
                                const auto src_metadata =
                                    *workspace.get_token_src_metadata_ptr(m_idx + row);
                                block_dst_rank_mask |= 1ull << src_metadata.rank_idx;
                            }
                            #pragma unroll
                            for (uint32_t offset = 16; offset != 0; offset >>= 1)
                                block_dst_rank_mask |= __shfl_down_sync(
                                    0xffffffff, block_dst_rank_mask, offset);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                            if (profile_math_leader) {
                                profile_scatter_ready_mask_block_cycles =
                                    clock64() - profile_ready_subphase_start;
                                profile_ready_subphase_start = clock64();
                            }
#endif

                            uint32_t publish_ready = 0;
                            uint64_t dst_rank_mask = 0;
                            uint64_t launch_epoch = 0;
                            if (lane_idx == 0) {
                                const auto dst_rank_mask_ptr =
                                    workspace.get_combine_dst_rank_mask_ptr(
                                        local_expert_idx);
                                ptx::red_or_rel_gpu(
                                    dst_rank_mask_ptr, block_dst_rank_mask);
                                __threadfence();

                                const auto posted_block_count_ptr =
                                    workspace.get_combine_posted_block_count_ptr(
                                        local_expert_idx);
                                const auto old_posted_block_count =
                                    ptx::atomic_add_acq_rel_sys(
                                        posted_block_count_ptr, 1);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                                profile_scatter_ready_atomic_block_cycles =
                                    clock64() - profile_ready_subphase_start;
                                profile_ready_subphase_start = clock64();
#endif
                                if (old_posted_block_count + 1 ==
                                    scheduler.get_current_num_m_blocks()) {
                                    dst_rank_mask = ptx::ld_acq_gpu(
                                        dst_rank_mask_ptr);
                                    launch_epoch = ptx::ld_acq_sys(
                                        workspace.get_combine_launch_epoch_ptr());
                                    DG_TRAP_ONLY_DEVICE_ASSERT(launch_epoch != 0);

                                    // Acquire chains through the per-block and
                                    // per-expert counters.  Publish all same-node
                                    // stores before the local ready store; RC-QP
                                    // ordering provides the corresponding order
                                    // for inter-node data WQEs and the ready WQE.
                                    __threadfence_system();
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                                    profile_scatter_ready_fence_block_cycles =
                                        clock64() - profile_ready_subphase_start;
                                    profile_ready_subphase_start = clock64();
#endif
                                    publish_ready = 1;
                                }
                            }

                            publish_ready = __shfl_sync(
                                0xffffffff, publish_ready, 0);
                            dst_rank_mask = __shfl_sync(
                                0xffffffff, dst_rank_mask, 0);
                            launch_epoch = __shfl_sync(
                                0xffffffff, launch_epoch, 0);
                            if (publish_ready != 0) {
                                // Lane 0's system fence must precede all lanes'
                                // notification WQEs.  Each lane owns distinct
                                // destination-rank QPs, so notifications can be
                                // posted in parallel without QP contention.
                                __syncwarp();
                                const uint32_t global_expert_idx =
                                    sym_buffer.rank_idx * kNumExpertsPerRank +
                                    local_expert_idx;
                                for (uint32_t dst_rank_idx = lane_idx;
                                     dst_rank_idx < kNumRanks;
                                     dst_rank_idx += 32) {
                                    if ((dst_rank_mask & (1ull << dst_rank_idx)) == 0)
                                        continue;
                                    const auto ready_ptr =
                                        workspace.get_combine_ready_epoch_ptr(
                                            global_expert_idx);
                                    if (dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS) {
                                        comm::ibgda::put_inline_with_credit<uint64_t>(
                                            ready_ptr, launch_epoch,
                                            static_cast<int>(dst_rank_idx),
                                            static_cast<int>(local_expert_idx));
                                    } else {
                                        ptx::st_relaxed_sys(
                                            sym_buffer.map(ready_ptr, dst_rank_idx),
                                            launch_epoch);
                                    }
                                }
                                __syncwarp();
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                                if (profile_math_leader) {
                                    profile_scatter_ready_notify_block_cycles =
                                        clock64() - profile_ready_subphase_start;
                                    profile_expert_ready_published = true;
                                    profile_expert_ready_epoch = launch_epoch;
                                    profile_expert_dst_rank_mask = dst_rank_mask;
                                    profile_expert_publish_globaltimer =
                                        ptx::get_globaltimer();
                                }
#endif
                            }
#else
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                            uint64_t profile_ready_subphase_start = clock64();
#endif
                            uint64_t block_dst_rank_mask = 0;
                            for (uint32_t row = 0; row < valid_m; ++ row) {
                                const auto src_metadata =
                                    *workspace.get_token_src_metadata_ptr(m_idx + row);
                                block_dst_rank_mask |= 1ull << src_metadata.rank_idx;
                            }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                            profile_scatter_ready_mask_block_cycles =
                                clock64() - profile_ready_subphase_start;
                            profile_ready_subphase_start = clock64();
#endif

                            const auto dst_rank_mask_ptr =
                                workspace.get_combine_dst_rank_mask_ptr(local_expert_idx);
                            ptx::red_or_rel_gpu(dst_rank_mask_ptr, block_dst_rank_mask);
                            __threadfence();

                            const auto posted_block_count_ptr =
                                workspace.get_combine_posted_block_count_ptr(local_expert_idx);
                            const auto old_posted_block_count =
                                ptx::atomic_add_acq_rel_sys(posted_block_count_ptr, 1);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                            profile_scatter_ready_atomic_block_cycles =
                                clock64() - profile_ready_subphase_start;
                            profile_ready_subphase_start = clock64();
#endif
                            if (old_posted_block_count + 1 ==
                                scheduler.get_current_num_m_blocks()) {
                                const uint64_t dst_rank_mask =
                                    ptx::ld_acq_gpu(dst_rank_mask_ptr);
                                const uint64_t launch_epoch = ptx::ld_acq_sys(
                                    workspace.get_combine_launch_epoch_ptr());
                                const uint32_t global_expert_idx =
                                    sym_buffer.rank_idx * kNumExpertsPerRank +
                                    local_expert_idx;
                                DG_TRAP_ONLY_DEVICE_ASSERT(launch_epoch != 0);

                                __threadfence_system();
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                                profile_scatter_ready_fence_block_cycles =
                                    clock64() - profile_ready_subphase_start;
                                profile_ready_subphase_start = clock64();
#endif
                                for (uint32_t dst_rank_idx = 0;
                                     dst_rank_idx < kNumRanks; ++ dst_rank_idx) {
                                    if ((dst_rank_mask & (1ull << dst_rank_idx)) == 0)
                                        continue;
                                    const auto ready_ptr =
                                        workspace.get_combine_ready_epoch_ptr(
                                            global_expert_idx);
                                    if (dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS) {
                                        comm::ibgda::put_inline_with_credit<uint64_t>(
                                            ready_ptr, launch_epoch,
                                            static_cast<int>(dst_rank_idx),
                                            static_cast<int>(local_expert_idx));
                                    } else {
                                        ptx::st_relaxed_sys(
                                            sym_buffer.map(ready_ptr, dst_rank_idx),
                                            launch_epoch);
                                    }
                                }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                                profile_scatter_ready_notify_block_cycles =
                                    clock64() - profile_ready_subphase_start;
                                profile_expert_ready_published = true;
                                profile_expert_ready_epoch = launch_epoch;
                                profile_expert_dst_rank_mask = dst_rank_mask;
                                profile_expert_publish_globaltimer =
                                    ptx::get_globaltimer();
#endif
                            }
#endif
                        }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                        const uint64_t profile_ready_sync_start =
                            profile_math_leader ? clock64() : 0;
#endif
                        ptx::sync_aligned(
                            kNumEpilogueThreads, kEpilogueFullBarrierIdx);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                        if (profile_math_leader)
                            profile_scatter_ready_sync_block_cycles =
                                clock64() - profile_ready_sync_start;
#endif
                    }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    if (profile_math_leader) {
                        // Destination-mask aggregation, expert completion
                        // accounting, system fence, and ordered ready WQEs.
                        profile_scatter_ready_block_cycles =
                            clock64() - profile_ready_start;
                        profile_scatter_publish_block_cycles =
                            clock64() - profile_publish_start;
                    }
#endif
#endif  // DG_MEGA_MOE_ASYNC_PUBLISHER
                } else
#endif
                {
                #pragma unroll
                for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                    const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                    const uint32_t m_idx_in_block = row_base + row_in_wg;
                    if (m_idx_in_block >= valid_m) break;

                    // Read cols_per_lane BF16 (= one ScatterVec) from smem
                    auto smem_ptr = smem_cd_l2
                        + smem_cd_l2_wg_offset
                        + row_in_wg * WG_BLOCK_N
                        + lane_in_row * cols_per_lane;
                    const auto packed = *reinterpret_cast<ScatterVec*>(smem_ptr);

                    const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                    const uint32_t dst_rank_idx = src_metadata.rank_idx;
                    const uint32_t dst_token_idx = src_metadata.token_idx;
                    const uint32_t dst_topk_idx = src_metadata.topk_idx;
                    const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                           .get_data_buffer(dst_token_idx);
                    auto dst_ptr = math::advance_ptr<ScatterVec>(
                        dst_token.get_base_ptr(),
                        (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(ScatterVec));
#ifdef DG_MEGA_MOE_INTERNODE
                    // Inter-node target rank: mirror the intra-node vector store with
                    // one inline RDMA WRITE, then wait before reusing the QP.
                    if (dst_rank_idx / DG_MEGA_MOE_NVL_PEERS != sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS) {
                        const int scatter_qp_id = static_cast<int>(dst_token_idx + lane_in_row);
                        comm::ibgda::put_inline<ScatterVec>(
                            dst_ptr, packed, static_cast<int>(dst_rank_idx), scatter_qp_id);
                        comm::ibgda::quiet(static_cast<int>(dst_rank_idx), scatter_qp_id);
                    } else
#endif
                    *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                }
                }

#ifdef DG_MEGA_MOE_INTERNODE
                if constexpr (not kCombineFullRow and kL2EpilogueRequiresFullSync)
#else
                if constexpr (kL2EpilogueRequiresFullSync)
#endif
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                if (profile_math_leader) {
                    profile_scatter_block_cycles =
                        clock64() - profile_scatter_block_start;
                    profile_scatter_cycles += profile_scatter_block_cycles;
                    profile_scatter_publish_cycles +=
                        profile_scatter_publish_block_cycles;
                    profile_scatter_staging_cycles +=
                        profile_scatter_staging_block_cycles;
                    profile_scatter_arrival_cycles +=
                        profile_scatter_arrival_block_cycles;
                    profile_scatter_wqe_cycles +=
                        profile_scatter_wqe_block_cycles;
                    profile_scatter_ready_cycles +=
                        profile_scatter_ready_block_cycles;
                    profile_scatter_ready_mask_cycles +=
                        profile_scatter_ready_mask_block_cycles;
                    profile_scatter_ready_atomic_cycles +=
                        profile_scatter_ready_atomic_block_cycles;
                    profile_scatter_ready_fence_cycles +=
                        profile_scatter_ready_fence_block_cycles;
                    profile_scatter_ready_notify_cycles +=
                        profile_scatter_ready_notify_block_cycles;
                    profile_scatter_ready_sync_cycles +=
                        profile_scatter_ready_sync_block_cycles;
                    if (profile_expert_ready_published) {
                        // Reuse rows beyond the active SM range for one
                        // launch-epoch-tagged record per local expert.  This
                        // keeps the diagnostic path API/workspace neutral.
                        const auto expert_profile = phase_profile_buffer
                            .get_data_buffer(
                                kExpertProfileRowBase + local_expert_idx)
                            .get_base_ptr<uint64_t>();
                        expert_profile[kExpertProfileNumTokens] =
                            scheduler.current_num_tokens;
                        expert_profile[kExpertProfileNumMBlocks] =
                            scheduler.get_current_num_m_blocks();
                        expert_profile[kExpertProfileDstRankMask] =
                            profile_expert_dst_rank_mask;
                        expert_profile[kExpertProfileFinalScatter] =
                            profile_scatter_block_cycles;
                        expert_profile[kExpertProfileFinalWQE] =
                            profile_scatter_wqe_block_cycles;
                        expert_profile[kExpertProfileReadyNotify] =
                            profile_scatter_ready_notify_block_cycles;
                        expert_profile[kExpertProfilePublishGlobaltimer] =
                            profile_expert_publish_globaltimer;
                        expert_profile[kExpertProfileSM] = sm_idx;
                        expert_profile[kExpertProfilePoolBlockOffset] =
                            scheduler.get_current_pool_block_offset();
                        ptx::st_release_sys(
                            expert_profile + kExpertProfileEpoch,
                            profile_expert_ready_epoch);
                    }
                }
#endif
            }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            if (profile_math_leader) {
                const uint64_t elapsed = clock64() - profile_math_block_start;
                if (block_phase == sched::BlockPhase::Linear2) {
                    profile_l2_cycles += elapsed;
                    ++ profile_l2_block_count;
                } else {
                    profile_l1_cycles += elapsed;
                    ++ profile_l1_block_count;
                }
            }
#endif
        };

        if constexpr (kFuseSharedExpert) {
            sm90_fp8_mega_moe_for_each_shared_block<
                BLOCK_M, kNumL1BlockNs, kNumL2BlockNs,
                kNumL1BlockKs, kNumL2BlockKs, kNumSMs,
                kNumPoolBlocks, kSharedExpertSentinel>(
                    scheduler, num_tokens, process_math_block);
        }

        if constexpr (kSplitPhaseHotPath) {
            sm90_fp8_mega_moe_for_each_block_split(
                scheduler,
                [&](const uint32_t& local_expert_idx,
                    const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    process_math_block(
                        std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
                        local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
                },
                [&](const uint32_t& local_expert_idx,
                    const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    process_math_block(
                        std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{},
                        local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
                });
        } else {
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                process_math_block(block_phase, local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
            });
        }

#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (profile_math_leader) {
            phase_profile[kProfileL1] = profile_l1_cycles;
            phase_profile[kProfileL2] = profile_l2_cycles;
            phase_profile[kProfileScatter] = profile_scatter_cycles;
            phase_profile[kProfileScatterPublish] =
                profile_scatter_publish_cycles;
            phase_profile[kProfileScatterStaging] =
                profile_scatter_staging_cycles;
            phase_profile[kProfileScatterArrival] =
                profile_scatter_arrival_cycles;
            phase_profile[kProfileScatterWQE] =
                profile_scatter_wqe_cycles;
            phase_profile[kProfileScatterReady] =
                profile_scatter_ready_cycles;
            phase_profile[kProfileScatterReadyMask] =
                profile_scatter_ready_mask_cycles;
            phase_profile[kProfileScatterReadyAtomic] =
                profile_scatter_ready_atomic_cycles;
            phase_profile[kProfileScatterReadyFence] =
                profile_scatter_ready_fence_cycles;
            phase_profile[kProfileScatterReadyNotify] =
                profile_scatter_ready_notify_cycles;
            phase_profile[kProfileScatterReadySync] =
                profile_scatter_ready_sync_cycles;
            phase_profile[kProfileL1BlockCount] = profile_l1_block_count;
            phase_profile[kProfileL2BlockCount] = profile_l2_block_count;
        }
#endif

        // Publish same-node NVLink stores at system scope.  In expert-ready
        // mode, final per-expert producers also issue an ordered ready store;
        // legacy mode still relies on the global barrier below.
        __threadfence_system();

        // ---------------- COMBINE ----------------
#ifdef DG_MEGA_MOE_PHASE_PROFILE
        const uint64_t profile_combine_barrier_start =
            profile_math_leader ? clock64() : 0;
#endif
        if constexpr (kCombineExpertReady) {
            // All local L2 producers and ready notifications must be submitted
            // before dispatch cleanup reuses their counters.  Remote completion
            // is observed below per dependency, so no all-QP quiet or cross-rank
            // collective is needed on the combine hot path.
            comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
                workspace, sm_idx, epilogue_thread_idx,
                [&]() {
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                });
        } else {
            // Legacy protocol: all ranks wait until every scatter is complete.
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                 kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() {
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                });
        }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (profile_math_leader)
            phase_profile[kProfileCombineBarrier] =
                clock64() - profile_combine_barrier_start;
#endif

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state.
        ptx::sync_unaligned(
            kNumDispatchEpilogueSyncThreads,
            kDispatchWithEpilogueBarrierIdx);

        if (epilogue_warp_idx >= kNumCombineWarps)
            return;

        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr uint32_t kDefaultNumChunks =
            (kNumChunkSlots * kNumCombineWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE
             and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 : 2;
        // Flash-style hidden=7168 is 7 * 1024. Splitting combine into 7 chunks
        // keeps each lane's BF16 reduce accumulator much smaller without
        // violating the 32-lane uint4 mapping.
        constexpr uint32_t kSplitMNNumChunks = (kHidden % 7 == 0) ? 7 : (kHidden >= 1024 ? 4 : 1);
        constexpr uint32_t kNumChunks = kSplitMNWarpgroups ? kSplitMNNumChunks : kDefaultNumChunks;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumCombineWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        DG_TRAP_ONLY_DEVICE_ASSERT(kNumChunkSlots * kNumCombineWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumCombineWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumCombineWarps * 2) * kNumChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
#ifdef DG_MEGA_MOE_PHASE_PROFILE
        const uint64_t profile_combine_reduce_start =
            lane_idx == 0 ? clock64() : 0;
        uint64_t profile_combine_ready_wait_cycles = 0;
        uint64_t profile_combine_work_cycles = 0;
        uint64_t profile_combine_token_max_cycles = 0;
        uint64_t profile_combine_token_count = 0;
#endif
        for (uint32_t token_idx = sm_idx * kNumCombineWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumCombineWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;

            if constexpr (kCombineExpertReady) {
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                const uint64_t profile_ready_wait_start =
                    lane_idx == 0 ? clock64() : 0;
                uint64_t profile_dependency_wait_cycles = 0;
#endif
                // The ready WRITE follows all output rows for this expert on
                // the same RC QP.  An acquire-system load that observes this
                // launch's epoch can therefore safely precede the TMA gather.
                if (stored_topk_slot_idx >= 0) {
                    DG_TRAP_ONLY_DEVICE_ASSERT(
                        stored_topk_slot_idx < static_cast<int>(kNumExperts));
                    const auto ready_ptr = workspace.get_combine_ready_epoch_ptr(
                        static_cast<uint32_t>(stored_topk_slot_idx));
                    const uint64_t launch_epoch = ptx::ld_acq_sys(
                        workspace.get_combine_launch_epoch_ptr());
                    constexpr uint64_t kReadyTimeoutCycles =
                        60ull * 2000000000ull;
                    const auto ready_wait_start = clock64();
                    while (ptx::ld_acq_sys(ready_ptr) != launch_epoch)
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            clock64() - ready_wait_start < kReadyTimeoutCycles);
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                    profile_dependency_wait_cycles =
                        clock64() - ready_wait_start;
#endif
                }
                __syncwarp();
#ifdef DG_MEGA_MOE_PHASE_PROFILE
                if (lane_idx == 0)
                    profile_combine_ready_wait_cycles +=
                        clock64() - profile_ready_wait_start;
                if (stored_topk_slot_idx >= 0) {
                    const uint64_t capped_wait =
                        profile_dependency_wait_cycles <= 0xffffffffull ?
                        profile_dependency_wait_cycles : 0xffffffffull;
                    const uint64_t packed_dependency =
                        (capped_wait << 32) |
                        (static_cast<uint64_t>(stored_topk_slot_idx) << 16) |
                        static_cast<uint64_t>(token_idx & 0xffffu);
                    atomicMax(
                        phase_profile + kProfileLongestReadyDependency,
                        static_cast<unsigned long long>(packed_dependency));
                }
#endif
            }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            const uint64_t profile_combine_work_start =
                lane_idx == 0 ? clock64() : 0;
#endif
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                if constexpr (kFuseSharedExpert) {
                    // Seed the routed top-k reduction with the already
                    // published local shared-expert output.
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto shared_values = *math::advance_ptr<const uint4>(
                            y,
                            static_cast<uint64_t>(token_idx) * kNumHiddenBytes +
                                chunk_byte_offset +
                                (j * 32 + lane_idx) * sizeof(uint4));
                        const auto bf16_values =
                            reinterpret_cast<const nv_bfloat162*>(&shared_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(
                                reduced[j * kNumElemsPerUint4 + l],
                                bf16_values[l]);
                    }
                }
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
            if (lane_idx == 0) {
                const uint64_t token_work_cycles =
                    clock64() - profile_combine_work_start;
                profile_combine_work_cycles += token_work_cycles;
                profile_combine_token_max_cycles =
                    profile_combine_token_max_cycles > token_work_cycles ?
                    profile_combine_token_max_cycles : token_work_cycles;
                ++profile_combine_token_count;
            }
#endif
        }
#ifdef DG_MEGA_MOE_PHASE_PROFILE
        if (lane_idx == 0) {
            atomicMax(phase_profile + kProfileCombineReduce,
                      static_cast<unsigned long long>(
                          clock64() - profile_combine_reduce_start));
            atomicMax(phase_profile + kProfileCombineReadyWait,
                      static_cast<unsigned long long>(
                          profile_combine_ready_wait_cycles));
            atomicMax(phase_profile + kProfileCombineWork,
                      static_cast<unsigned long long>(
                          profile_combine_work_cycles));
            atomicMax(phase_profile + kProfileCombineTokenMax,
                      static_cast<unsigned long long>(
                          profile_combine_token_max_cycles));
            atomicMax(phase_profile + kProfileCombineTokenCount,
                      static_cast<unsigned long long>(
                          profile_combine_token_count));
        }
        if (profile_math_leader)
            phase_profile[kProfileTotal] =
                clock64() - phase_profile[kProfileStartClock];

        // Only math/epilogue threads converge here. Dispatch and TMA roles use
        // role-specific register reallocations and must not join this barrier.
        comm::grid_sync<kNumSMs, 2>(
            workspace, sm_idx, epilogue_thread_idx,
            [&]() {
                ptx::sync_aligned(
                    kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            });

        // Silent mode keeps every profile counter in the symmetric buffer but
        // skips the device-side printf aggregation, which costs ~1.5 ms per
        // launch and drops most lines under a real benchmark loop.  The host
        // reads the buffer's tail after the run instead.
#ifndef DG_MEGA_MOE_PHASE_PROFILE_SILENT
        if (sm_idx == 0 and epilogue_thread_idx == 0) {
            unsigned long long max_cycles[kPhaseProfileSlots] = {};
            unsigned long long remote_read_count = 0;
            unsigned long long l1_block_count = 0;
            unsigned long long l2_block_count = 0;
            unsigned long long scatter_write_count = 0;
            unsigned long long combine_ready_wait_cycles = 0;
            uint32_t scatter_critical_sm = 0;
            unsigned long long scatter_critical_cycles = 0;
            unsigned long long scatter_critical_publish_cycles = 0;
            unsigned long long scatter_critical_staging_cycles = 0;
            unsigned long long scatter_critical_arrival_cycles = 0;
            unsigned long long scatter_critical_wqe_cycles = 0;
            unsigned long long scatter_critical_ready_cycles = 0;
            unsigned long long scatter_critical_ready_mask_cycles = 0;
            unsigned long long scatter_critical_ready_atomic_cycles = 0;
            unsigned long long scatter_critical_ready_fence_cycles = 0;
            unsigned long long scatter_critical_ready_notify_cycles = 0;
            unsigned long long scatter_critical_ready_sync_cycles = 0;
            for (uint32_t sm = 0; sm < kNumSMs; ++ sm) {
                const auto row = phase_profile_buffer.get_data_buffer(sm)
                    .get_base_ptr<unsigned long long>();
                #pragma unroll
                for (uint32_t slot = 0; slot < kPhaseProfileSlots; ++ slot)
                    max_cycles[slot] = max_cycles[slot] > row[slot]
                        ? max_cycles[slot] : row[slot];
                remote_read_count += row[kProfileRemoteReadCount];
                l1_block_count += row[kProfileL1BlockCount];
                l2_block_count += row[kProfileL2BlockCount];
                scatter_write_count += row[kProfileScatterWriteCount];
                combine_ready_wait_cycles =
                    combine_ready_wait_cycles > row[kProfileCombineReadyWait] ?
                    combine_ready_wait_cycles : row[kProfileCombineReadyWait];
                if (row[kProfileScatter] > scatter_critical_cycles) {
                    scatter_critical_sm = sm;
                    scatter_critical_cycles = row[kProfileScatter];
                    scatter_critical_publish_cycles =
                        row[kProfileScatterPublish];
                    scatter_critical_staging_cycles =
                        row[kProfileScatterStaging];
                    scatter_critical_arrival_cycles =
                        row[kProfileScatterArrival];
                    scatter_critical_wqe_cycles = row[kProfileScatterWQE];
                    scatter_critical_ready_cycles = row[kProfileScatterReady];
                    scatter_critical_ready_mask_cycles =
                        row[kProfileScatterReadyMask];
                    scatter_critical_ready_atomic_cycles =
                        row[kProfileScatterReadyAtomic];
                    scatter_critical_ready_fence_cycles =
                        row[kProfileScatterReadyFence];
                    scatter_critical_ready_notify_cycles =
                        row[kProfileScatterReadyNotify];
                    scatter_critical_ready_sync_cycles =
                        row[kProfileScatterReadySync];
                }
            }
            printf(
                "MEGA_MOE_PHASE_PROFILE rank=%u tokens=%u block_m=%u block_n=%u wg_n=%u "
                "metadata_cycles=%llu dispatch_barrier_cycles=%llu dispatch_pull_cycles=%llu "
                "remote_read_cycles=%llu cleanup_barrier_cycles=%llu l1_cycles=%llu "
                "l2_cycles=%llu scatter_cycles=%llu scatter_publish_cycles=%llu "
                "scatter_critical_publish_cycles=%llu "
                "scatter_staging_cycles=%llu scatter_arrival_cycles=%llu "
                "scatter_wqe_cycles=%llu scatter_ready_cycles=%llu "
                "scatter_ready_mask_cycles=%llu scatter_ready_atomic_cycles=%llu "
                "scatter_ready_fence_cycles=%llu scatter_ready_notify_cycles=%llu "
                "scatter_ready_sync_cycles=%llu scatter_critical_sm=%u "
                "combine_barrier_cycles=%llu combine_ready_wait_cycles=%llu "
                "combine_reduce_cycles=%llu total_cycles=%llu "
                "remote_reads=%llu l1_blocks=%llu l2_blocks=%llu\n",
                sym_buffer.rank_idx, num_tokens, BLOCK_M, BLOCK_N, WG_BLOCK_N,
                max_cycles[kProfileMetadata], max_cycles[kProfileDispatchBarrier],
                max_cycles[kProfileDispatchPull], max_cycles[kProfileRemoteRead],
                max_cycles[kProfileCleanupBarrier], max_cycles[kProfileL1],
                max_cycles[kProfileL2], max_cycles[kProfileScatter],
                max_cycles[kProfileScatterPublish],
                scatter_critical_publish_cycles,
                scatter_critical_staging_cycles,
                scatter_critical_arrival_cycles,
                scatter_critical_wqe_cycles,
                scatter_critical_ready_cycles,
                scatter_critical_ready_mask_cycles,
                scatter_critical_ready_atomic_cycles,
                scatter_critical_ready_fence_cycles,
                scatter_critical_ready_notify_cycles,
                scatter_critical_ready_sync_cycles,
                scatter_critical_sm,
                max_cycles[kProfileCombineBarrier],
                combine_ready_wait_cycles, max_cycles[kProfileCombineReduce],
                max_cycles[kProfileTotal],
                remote_read_count, l1_block_count, l2_block_count);
            // Keep the main device printf at CUDA's 32-argument limit.
            printf(
                "MEGA_MOE_SCATTER_COUNT_PROFILE rank=%u tokens=%u "
                "scatter_writes=%llu\n",
                sym_buffer.rank_idx, num_tokens, scatter_write_count);
            // Absolute SM-0 stamps.  Comparing the entry stamps across ranks
            // separates launch skew from protocol serialization; comparing
            // counts_sent to counts_ready gives the negotiation's own cost.
            printf(
                "MEGA_MOE_SYNC_PROFILE rank=%u tokens=%u entry_ns=%llu "
                "counts_sent_ns=%llu counts_ready_ns=%llu\n",
                sym_buffer.rank_idx, num_tokens,
                static_cast<unsigned long long>(
                    phase_profile[kProfileEntryGlobaltimer]),
                static_cast<unsigned long long>(
                    phase_profile[kProfileCountsSentGlobaltimer]),
                static_cast<unsigned long long>(
                    phase_profile[kProfileCountsReadyGlobaltimer]));
            printf(
                "MEGA_MOE_COMBINE_PROFILE rank=%u tokens=%u "
                "combine_work_cycles=%llu combine_token_max_cycles=%llu "
                "combine_token_count=%llu\n",
                sym_buffer.rank_idx, num_tokens,
                max_cycles[kProfileCombineWork],
                max_cycles[kProfileCombineTokenMax],
                max_cycles[kProfileCombineTokenCount]);
            const unsigned long long launch_epoch = ptx::ld_acq_sys(
                workspace.get_combine_launch_epoch_ptr());
#ifdef DG_MEGA_MOE_ASYNC_PUBLISHER
            // Pair profiles are published by a different role on every CTA.
            // Dispatch cleanup is allowed to clear the aliased per-expert
            // done epoch immediately after its block rendezvous, so wait on
            // the persistent epoch of each diagnostic pair row itself.
            for (uint32_t pair_task_idx = 0;
                 pair_task_idx < kNumPairProfileRows;
                 ++ pair_task_idx) {
                constexpr int64_t kProfilePairTimeoutCycles =
                    60ll * 2000000000ll;
                const uint64_t wait_start = clock64();
                const auto pair_profile = pair_profile_base +
                    pair_task_idx * kPairProfileNumSlots;
                while (ptx::ld_acq_sys(reinterpret_cast<uint64_t*>(
                           pair_profile + kPairProfileEpoch)) !=
                       launch_epoch) {
                    if (clock64() - wait_start >=
                        kProfilePairTimeoutCycles) {
                        printf(
                            "MEGA_MOE_PAIR_PROFILE_TIMEOUT rank=%u "
                            "expert=%u dst=%u launch_epoch=%llu "
                            "pair_epoch=%llu\n",
                            sym_buffer.rank_idx,
                            pair_task_idx / kNumRanks,
                            pair_task_idx % kNumRanks,
                            launch_epoch,
                            static_cast<unsigned long long>(ptx::ld_acq_sys(
                                reinterpret_cast<uint64_t*>(
                                    pair_profile + kPairProfileEpoch))));
                        DG_TRAP_ONLY_DEVICE_ASSERT(false);
                    }
                }
            }
            for (uint32_t local_expert_idx = 0;
                 local_expert_idx < kNumExpertsPerRank;
                 ++ local_expert_idx) {
                uint32_t total_rows = 0;
                uint32_t rdma_rows = 0;
                uint32_t nonzero_pairs = 0;
                uint32_t rdma_pairs = 0;
                uint32_t submit_count = 0;
                uint32_t max_batch_rows = 0;
                uint32_t hottest_pair_rows = 0;
                uint32_t hottest_pair_dst = 0;
                uint32_t slowest_pair_dst = 0;
                uint32_t last_pair_dst = 0;
                uint32_t last_pair_rows = 0;
                unsigned long long wqe_cycles_sum = 0;
                unsigned long long slowest_pair_ns = 0;
                unsigned long long max_ready_post_ns = 0;
                unsigned long long first_nonzero_start = ~0ull;
                unsigned long long first_nonzero_done = ~0ull;
                unsigned long long last_nonzero_done = 0;
                unsigned long long last_pair_done = 0;
                for (uint32_t dst_rank_idx = 0;
                     dst_rank_idx < kNumRanks;
                     ++ dst_rank_idx) {
                    const uint32_t pair_task_idx =
                        local_expert_idx * kNumRanks + dst_rank_idx;
                    const auto pair_profile = pair_profile_base +
                        pair_task_idx * kPairProfileNumSlots;
                    DG_DEVICE_ASSERT(
                        ptx::ld_acq_sys(
                            reinterpret_cast<uint64_t*>(
                                pair_profile + kPairProfileEpoch)) ==
                        launch_epoch);
                    const uint32_t pair_rows = static_cast<uint32_t>(
                        pair_profile[kPairProfileNumRows]);
                    const auto pair_start =
                        pair_profile[kPairProfileStartGlobaltimer];
                    const auto pair_done =
                        pair_profile[kPairProfileDoneGlobaltimer];
                    const auto data_done =
                        pair_profile[kPairProfileDataDoneGlobaltimer];
                    total_rows += pair_rows;
                    if (pair_rows != 0) {
                        ++ nonzero_pairs;
                        first_nonzero_start =
                            first_nonzero_start < pair_start ?
                            first_nonzero_start : pair_start;
                        first_nonzero_done =
                            first_nonzero_done < pair_done ?
                            first_nonzero_done : pair_done;
                        last_nonzero_done =
                            last_nonzero_done > pair_done ?
                            last_nonzero_done : pair_done;
                        const auto ready_post_ns = pair_done - data_done;
                        max_ready_post_ns =
                            max_ready_post_ns > ready_post_ns ?
                            max_ready_post_ns : ready_post_ns;
                        const auto pair_elapsed = pair_done - pair_start;
                        if (pair_elapsed > slowest_pair_ns) {
                            slowest_pair_ns = pair_elapsed;
                            slowest_pair_dst = dst_rank_idx;
                        }
                    }
                    if (pair_rows > hottest_pair_rows) {
                        hottest_pair_rows = pair_rows;
                        hottest_pair_dst = dst_rank_idx;
                    }
                    if (pair_done > last_pair_done) {
                        last_pair_done = pair_done;
                        last_pair_dst = dst_rank_idx;
                        last_pair_rows = pair_rows;
                    }
                    const bool pair_is_inter =
                        dst_rank_idx / DG_MEGA_MOE_NVL_PEERS !=
                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS;
                    if (pair_is_inter) {
                        rdma_rows += pair_rows;
                        rdma_pairs += pair_rows != 0;
                        submit_count += static_cast<uint32_t>(
                            pair_profile[kPairProfileSubmitCount]);
                        max_batch_rows = cute::max(
                            max_batch_rows,
                            static_cast<uint32_t>(
                                pair_profile[kPairProfileMaxBatchRows]));
                        wqe_cycles_sum +=
                            pair_profile[kPairProfileWQECycles];
                    }
                }
                const unsigned long long done_tail_ns =
                    nonzero_pairs == 0 ? 0 :
                    last_pair_done - first_nonzero_start;
                const unsigned long long nonzero_done_skew_ns =
                    nonzero_pairs == 0 ? 0 :
                    last_nonzero_done - first_nonzero_done;
                printf(
                    "MEGA_MOE_PAIR_PUBLISH_PROFILE rank=%u tokens=%u "
                    "launch_epoch=%llu expert=%u total_rows=%u "
                    "rdma_rows=%u nonzero_pairs=%u rdma_pairs=%u "
                    "hottest_dst=%u hottest_rows=%u submits=%u "
                    "max_batch_rows=%u rows_per_submit_x100=%u "
                    "wqe_cycles_sum=%llu slowest_dst=%u "
                    "slowest_pair_ns=%llu last_dst=%u last_rows=%u "
                    "max_ready_post_ns=%llu nonzero_done_skew_ns=%llu "
                    "done_tail_ns=%llu\n",
                    sym_buffer.rank_idx, num_tokens, launch_epoch,
                    local_expert_idx, total_rows, rdma_rows,
                    nonzero_pairs, rdma_pairs, hottest_pair_dst,
                    hottest_pair_rows, submit_count, max_batch_rows,
                    submit_count == 0 ? 0 :
                        rdma_rows * 100 / submit_count,
                    wqe_cycles_sum, slowest_pair_dst, slowest_pair_ns,
                    last_pair_dst, last_pair_rows,
                    max_ready_post_ns, nonzero_done_skew_ns, done_tail_ns);
            }
#endif
            const unsigned long long longest_ready_dependency =
                max_cycles[kProfileLongestReadyDependency];
            const uint32_t longest_ready_expert = static_cast<uint32_t>(
                (longest_ready_dependency >> 16) & 0xffffull);
            printf(
                "MEGA_MOE_LAST_READY_PROFILE receiver_rank=%u tokens=%u "
                "launch_epoch=%llu wait_cycles=%llu global_expert=%u "
                "src_rank=%u local_expert=%u token_idx=%u\n",
                sym_buffer.rank_idx, num_tokens, launch_epoch,
                longest_ready_dependency >> 32, longest_ready_expert,
                longest_ready_expert / kNumExpertsPerRank,
                longest_ready_expert % kNumExpertsPerRank,
                static_cast<uint32_t>(longest_ready_dependency & 0xffffull));

            // Globaltimer is comparable only within one GPU.  Its delta from
            // the first local expert publication exposes producer-side skew;
            // receiver wait cycles above identify the dependency that saw the
            // largest ready delay on each destination rank.
            unsigned long long first_expert_publish = ~0ull;
            for (uint32_t local_expert_idx = 0;
                 local_expert_idx < kNumExpertsPerRank; ++ local_expert_idx) {
                const auto expert_profile = phase_profile_buffer
                    .get_data_buffer(kExpertProfileRowBase + local_expert_idx)
                    .get_base_ptr<uint64_t>();
                if (ptx::ld_acq_sys(expert_profile + kExpertProfileEpoch) !=
                    launch_epoch)
                    continue;
                const auto publish_globaltimer =
                    expert_profile[kExpertProfilePublishGlobaltimer];
                first_expert_publish = first_expert_publish < publish_globaltimer ?
                    first_expert_publish : publish_globaltimer;
            }
            for (uint32_t local_expert_idx = 0;
                 local_expert_idx < kNumExpertsPerRank; ++ local_expert_idx) {
                const auto expert_profile = phase_profile_buffer
                    .get_data_buffer(kExpertProfileRowBase + local_expert_idx)
                    .get_base_ptr<uint64_t>();
                if (ptx::ld_acq_sys(expert_profile + kExpertProfileEpoch) !=
                    launch_epoch)
                    continue;
                const auto publish_globaltimer =
                    expert_profile[kExpertProfilePublishGlobaltimer];
                const uint32_t expert_tokens = static_cast<uint32_t>(
                    expert_profile[kExpertProfileNumTokens]);
                const uint32_t first_pool_token = static_cast<uint32_t>(
                    expert_profile[kExpertProfilePoolBlockOffset]) * BLOCK_M;
                uint32_t same_rank_rows = 0;
                uint32_t nvl_rows = 0;
                uint32_t rdma_rows = 0;
                for (uint32_t row = 0; row < expert_tokens; ++ row) {
                    const auto src_metadata =
                        *workspace.get_token_src_metadata_ptr(
                            first_pool_token + row);
                    if (src_metadata.rank_idx == sym_buffer.rank_idx) {
                        ++ same_rank_rows;
#ifdef DG_MEGA_MOE_INTERNODE
                    } else if (
                        src_metadata.rank_idx / DG_MEGA_MOE_NVL_PEERS ==
                        sym_buffer.rank_idx / DG_MEGA_MOE_NVL_PEERS) {
                        ++ nvl_rows;
                    } else {
                        ++ rdma_rows;
#else
                    } else {
                        ++ nvl_rows;
#endif
                    }
                }
                printf(
                    "MEGA_MOE_EXPERT_READY_PROFILE producer_rank=%u tokens=%u "
                    "launch_epoch=%llu global_expert=%u local_expert=%u "
                    "expert_tokens=%llu same_rank_rows=%u nvl_rows=%u "
                    "rdma_rows=%u m_blocks=%llu dst_mask=0x%llx "
                    "final_scatter_cycles=%llu final_wqe_cycles=%llu "
                    "ready_notify_cycles=%llu publish_globaltimer=%llu "
                    "publish_delta_ns=%llu producer_sm=%llu\n",
                    sym_buffer.rank_idx, num_tokens, launch_epoch,
                    sym_buffer.rank_idx * kNumExpertsPerRank + local_expert_idx,
                    local_expert_idx,
                    expert_profile[kExpertProfileNumTokens],
                    same_rank_rows, nvl_rows, rdma_rows,
                    expert_profile[kExpertProfileNumMBlocks],
                    expert_profile[kExpertProfileDstRankMask],
                    expert_profile[kExpertProfileFinalScatter],
                    expert_profile[kExpertProfileFinalWQE],
                    expert_profile[kExpertProfileReadyNotify],
                    publish_globaltimer,
                    publish_globaltimer - first_expert_publish,
                    expert_profile[kExpertProfileSM]);
            }
        }
#endif
#endif  // DG_MEGA_MOE_PHASE_PROFILE_SILENT
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_TRAP_ONLY_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
