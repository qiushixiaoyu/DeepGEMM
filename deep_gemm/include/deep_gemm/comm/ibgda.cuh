// IBGDA device verbs for mega-moe internode paths.
//
// Portions derived from DeepEP (csrc/kernels/legacy/ibgda_device.cuh), itself
// derived from NVSHMEM (non_abi/device/pt-to-pt/ibgda_device.cuh).
// Copyright (c) NVIDIA Corporation. Licensed under the NVSHMEM SLA.
//
// 为什么存在这个文件：NVSHMEM 3.4.5 的公开 device API(nvshmem_putmem/uint64_p/
// getmem + nvshmem_quiet)在本集群(16-PE 混合 P2P+IBGDA、RoCE 多网卡)上对一半的
// 目标 PE 静默丢写(实测奇数 PE 全丢，零误投递、零报错)；而 DeepEP 用同一套 IBGDA
// RC QP 基础设施、但以显式 QP 选择 + 显式 per-QP quiet 的内部 verbs 驱动则全通。
// 故 mega 的跨节点数据面改用本文件的 verbs，不再走公开 device API。
//
// 相对 DeepEP 原文件的扩展：
//   1. put_inline<T> 支持 4/8/16 字节 inline RDMA WRITE(源在寄存器即可，无需注册)；
//   2. get_thread：RDMA READ(DeepEP 纯 push 无此需求；mega 的 dispatch 是 pull 语义)；
//   3. poll_cq 的 cons_idx 更新改为 atomicMax，允许多 warp 并发 quiet 同一 QP
//      (mega 的 pull 由多 warp 并发发起，DeepEP 原实现假设单线程独占 QP)。
//   4. put_nbi_warp：从 registered symmetric HBM 发出普通 RDMA WRITE，供
//      full-row combine 使用；每条消息完成全部 WQE 后只敲一次 doorbell。
//
// 使用约束(调用方负责)：
//   - 仅用于跨节点 PE(同节点走 NVLink P2P store，不经此文件)；
//   - qp_id 任意 int，内部对 num_rc_per_pe*num_devices 取模；调用方应让并发流量
//     分摊到不同 qp_id 上，保证两次 quiet 之间单 QP 在途 WQE < NVSHMEM_QP_DEPTH
//     (压力场景使用 NVSHMEM_QP_DEPTH=4096；full-row WRITE 在 reserve 时检查 credit)；
//   - 阻塞式读 = get_thread + quiet(同一 pe/qp)。

#pragma once

#include <nvshmem.h>
#include <device_host_transport/nvshmem_common_ibgda.h>
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>

namespace deep_gemm::comm::ibgda {

static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64, "Invalid QP minimum depth");

// ---------------------------------------------------------------------------
// Minimal PTX helpers (vendored from DeepEP utils.cuh)
// ---------------------------------------------------------------------------

__device__ static __forceinline__ uint64_t HtoBE64(uint64_t x) {
    uint64_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 lo;\n\t"
        ".reg .b32 hi;\n\t"
        ".reg .b32 new_lo;\n\t"
        ".reg .b32 new_hi;\n\t"
        "mov.b64 {lo,hi}, %1;\n\t"
        "prmt.b32 new_hi, lo, ign, 0x0123;\n\t"
        "prmt.b32 new_lo, hi, ign, 0x0123;\n\t"
        "mov.b64 %0, {new_lo,new_hi};\n\t"
        "}"
        : "=l"(ret)
        : "l"(x));
    return ret;
}

__device__ static __forceinline__ uint32_t HtoBE32(uint32_t x) {
    uint32_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        "prmt.b32 %0, %1, ign, 0x0123;\n\t"
        "}"
        : "=r"(ret)
        : "r"(x));
    return ret;
}

__device__ static __forceinline__ uint16_t HtoBE16(uint16_t x) {
    auto a = static_cast<uint32_t>(x);
    uint32_t d;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        "mov.b32 mask, 0x4401;\n\t"
        "mov.b32 ign, 0x0;\n\t"
        "prmt.b32 %0, %1, ign, mask;\n\t"
        "}"
        : "=r"(d)
        : "r"(a));
    return static_cast<uint16_t>(d);
}

__device__ static __forceinline__ void memory_fence_cta() {
    asm volatile("fence.acq_rel.cta;" ::: "memory");
}

__device__ static __forceinline__ uint16_t ld_na_relaxed(const uint16_t* ptr) {
    uint16_t val;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(val) : "l"(ptr));
    return val;
}

__device__ static __forceinline__ uint64_t ld_na_relaxed(const uint64_t* ptr) {
    uint64_t val;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(val) : "l"(ptr));
    return val;
}

__device__ static __forceinline__ void st_na_relaxed(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ static __forceinline__ void st_na_relaxed(const int4* ptr, int4 val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.b32 [%0], {%1, %2, %3, %4};"
                 : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__device__ static __forceinline__ void st_na_release(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ static __forceinline__ void st_na_release(const uint64_t* ptr, uint64_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}

// ---------------------------------------------------------------------------
// QP / state access (DeepEP 原样)
// ---------------------------------------------------------------------------

typedef struct mlx5_wqe_ctrl_seg __attribute__((__aligned__(8))) ibgda_ctrl_seg_t;

__device__ static __forceinline__ nvshmemi_ibgda_device_state_t* ibgda_get_state() {
    return &nvshmemi_ibgda_device_state_d;
}

// 每对 (本 PE, 远端 PE) 的可用 QP 总数；qp_id 对其取模
__device__ static __forceinline__ uint32_t num_qps() {
    auto state = ibgda_get_state();
    return state->num_rc_per_pe * state->num_devices_initialized;
}

__device__ static __forceinline__ nvshmemi_ibgda_device_qp_t* ibgda_get_rc(int pe, int id) {
    auto state = ibgda_get_state();
    const auto num_rc_per_pe = state->num_rc_per_pe;
    return &state->globalmem
                .rcs[pe * num_rc_per_pe * state->num_devices_initialized + id % (num_rc_per_pe * state->num_devices_initialized)];
}

__device__ static __forceinline__ void ibgda_lock_acquire(int* lock) {
    while (atomicCAS(lock, 0, 1) == 1)
        ;
    memory_fence_cta();
}

__device__ static __forceinline__ void ibgda_lock_release(int* lock) {
    memory_fence_cta();
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(lock), "r"(0));
}

// ---------------------------------------------------------------------------
// Doorbell / submit (DeepEP 原样)
// ---------------------------------------------------------------------------

__device__ static __forceinline__ void ibgda_update_dbr(nvshmemi_ibgda_device_qp_t* qp, uint32_t dbrec_head) {
    __be32 dbrec_val;
    __be32* dbrec_ptr = qp->tx_wq.dbrec;
    asm("{\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"
        ".reg .b32 ign;\n\t"
        "and.b32 dbrec_head_16b, %1, 0xffff;\n\t"
        "prmt.b32 %0, dbrec_head_16b, ign, 0x123;\n\t"
        "}"
        : "=r"(dbrec_val)
        : "r"(dbrec_head));
    st_na_release(reinterpret_cast<uint32_t*>(dbrec_ptr), dbrec_val);
}

__device__ static __forceinline__ void ibgda_ring_db(nvshmemi_ibgda_device_qp_t* qp, uint16_t prod_idx) {
    auto bf_ptr = reinterpret_cast<uint64_t*>(qp->tx_wq.bf);
    ibgda_ctrl_seg_t ctrl_seg = {.opmod_idx_opcode = HtoBE32(prod_idx << 8), .qpn_ds = HtoBE32(qp->qpn << 8)};
    st_na_release(bf_ptr, *(reinterpret_cast<uint64_t*>(&ctrl_seg)));
}

__device__ static __forceinline__ void ibgda_post_send(nvshmemi_ibgda_device_qp_t* qp, uint64_t new_prod_idx) {
    nvshmemi_ibgda_device_qp_management_t* mvars = &qp->mvars;
    uint64_t old_prod_idx;

    ibgda_lock_acquire(&mvars->post_send_lock);
    old_prod_idx = atomicMax(reinterpret_cast<unsigned long long int*>(&mvars->tx_wq.prod_idx), new_prod_idx);
    if (new_prod_idx > old_prod_idx) {
        ibgda_update_dbr(qp, new_prod_idx);
        ibgda_ring_db(qp, new_prod_idx);
    }
    ibgda_lock_release(&mvars->post_send_lock);
}

__device__ static __forceinline__ void ibgda_submit_requests(nvshmemi_ibgda_device_qp_t* qp,
                                                             uint64_t base_wqe_idx,
                                                             uint32_t num_wqes) {
    auto state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_management_t* mvars = &qp->mvars;
    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

    // WQE writes must be finished first
    __threadfence();

    unsigned long long int* ready_idx =
        (unsigned long long int*)(state->use_async_postsend ? qp->tx_wq.prod_idx : &mvars->tx_wq.ready_head);

    // Wait for prior WQE slots to be filled first
    while (atomicCAS(ready_idx, base_wqe_idx, new_wqe_idx) != base_wqe_idx)
        ;

    if (!state->use_async_postsend)
        ibgda_post_send(qp, new_wqe_idx);
}

// ---------------------------------------------------------------------------
// lkey / rkey 查表 (DeepEP 原样)
// ---------------------------------------------------------------------------

__device__ static __forceinline__ uint64_t
ibgda_get_lkey_and_rkey(uint64_t laddr, __be32* lkey, uint64_t raddr, int dst_pe, uint64_t* out_raddr, __be32* out_rkey, uint32_t dev_idx) {
    auto state = ibgda_get_state();
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);
    auto log2_cumem_granularity = state->log2_cumem_granularity;

    uint64_t idx = ((laddr - heap_start) >> log2_cumem_granularity) * state->num_devices_initialized + dev_idx;
    auto device_key = state->constmem.lkeys[idx];
    auto lchunk_size = device_key.next_addr - laddr;
    *lkey = device_key.key;

    uint64_t roffset = raddr - heap_start;
    idx = ((roffset >> log2_cumem_granularity) * nvshmemi_device_state_d.npes) * state->num_devices_initialized +
        dst_pe * state->num_devices_initialized + dev_idx;
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS) {
        device_key = state->constmem.rkeys[idx];
    } else {
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
    }
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
    *out_rkey = device_key.key;

    auto rchunk_size = device_key.next_addr - roffset;
    return min(lchunk_size, rchunk_size);
}

__device__ static __forceinline__ void ibgda_get_rkey(uint64_t addr, int dst_pe, uint64_t* out_raddr, __be32* out_rkey, uint32_t dev_idx) {
    auto state = ibgda_get_state();
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);

    uint64_t roffset = addr - heap_start;
    uint64_t idx = ((roffset >> state->log2_cumem_granularity) * nvshmemi_device_state_d.npes * state->num_devices_initialized) +
        dst_pe * state->num_devices_initialized + dev_idx;
    nvshmemi_ibgda_device_key_t device_key;
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS)
        device_key = state->constmem.rkeys[idx];
    else
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
    *out_rkey = device_key.key;
}

__device__ static __forceinline__ uint64_t ibgda_reserve_wqe_slots(nvshmemi_ibgda_device_qp_t* qp, uint32_t num_wqes) {
    auto mvars = &qp->mvars;
    return atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.resv_head), static_cast<unsigned long long>(num_wqes));
}

__device__ static __forceinline__ void* ibgda_get_wqe_ptr(nvshmemi_ibgda_device_qp_t* qp, uint16_t wqe_idx) {
    uint16_t cnt = qp->tx_wq.nwqes;
    uint16_t idx = wqe_idx & (cnt - 1);
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(qp->tx_wq.wqe) + (idx << MLX5_SEND_WQE_SHIFT));
}

// Defined below the public request helpers.  The declaration is needed by the
// credit-aware full-row reservation path.
__device__ static __forceinline__ void ibgda_poll_cq(
    nvshmemi_ibgda_device_cq_t* cq, uint64_t idx);

// Serialize only reservation/credit maintenance, not WQE construction.  If a
// QP is about to wrap, wait until all earlier reservations become ready, post
// their contiguous prefix and reclaim it before handing out new slots.
__device__ static __forceinline__ uint64_t ibgda_reserve_wqe_slots_with_credit(
    nvshmemi_ibgda_device_qp_t* qp, uint32_t num_wqes) {
    auto state = ibgda_get_state();
    auto mvars = &qp->mvars;
    DG_DEVICE_ASSERT(num_wqes > 0 and num_wqes < qp->tx_wq.nwqes);

    ibgda_lock_acquire(&mvars->post_send_lock);
    uint64_t base_wqe_idx = ld_na_relaxed(&mvars->tx_wq.resv_head);
    uint64_t cq_cons_idx = ld_na_relaxed(reinterpret_cast<uint64_t*>(qp->tx_wq.cq->cons_idx));
    const uint32_t inflight_cap = qp->tx_wq.nwqes;
    if (base_wqe_idx + num_wqes - cq_cons_idx >= inflight_cap) {
        uint64_t* ready_idx = state->use_async_postsend ?
            qp->tx_wq.prod_idx : &mvars->tx_wq.ready_head;
        constexpr uint64_t kCreditTimeoutCycles = 120ull * 2000000000ull;
        const auto start_clock = clock64();
        while (ld_na_relaxed(ready_idx) < base_wqe_idx)
            DG_TRAP_ONLY_DEVICE_ASSERT(clock64() - start_clock < kCreditTimeoutCycles);

        const uint64_t ready_head = ld_na_relaxed(ready_idx);
        if (not state->use_async_postsend) {
            const auto old_prod_idx = atomicMax(
                reinterpret_cast<unsigned long long int*>(qp->tx_wq.prod_idx),
                static_cast<unsigned long long int>(ready_head));
            if (ready_head > old_prod_idx) {
                ibgda_update_dbr(qp, ready_head);
                ibgda_ring_db(qp, ready_head);
            }
        }
        ibgda_poll_cq(qp->tx_wq.cq, ready_head);
    }

    base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes);
    ibgda_lock_release(&mvars->post_send_lock);
    return base_wqe_idx;
}

// ---------------------------------------------------------------------------
// WQE 构造
// ---------------------------------------------------------------------------

// 扩展版 inline RDMA WRITE WQE：支持 4/8/16 字节负载(DeepEP 原版仅 4B)。
// WQE 布局: ctrl(16B) + raddr(16B) + inl_seg(4B) + data(kBytes)，总长 ≤ 64B(单 WQEBB)。
template <uint32_t kBytes>
__device__ static __forceinline__ void ibgda_write_rdma_write_inl_wqe(
    nvshmemi_ibgda_device_qp_t* qp, const uint32_t* val, uint64_t raddr, __be32 rkey, uint16_t wqe_idx, void** out_wqes) {
    static_assert(kBytes == 4 or kBytes == 8 or kBytes == 16, "Unsupported inline size");
    // ds = ceil((16 + 16 + 4 + kBytes) / 16)
    constexpr uint32_t kNumDS = (16 + 16 + 4 + kBytes + 15) / 16;

    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;

    auto* ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    auto* raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    auto* inl_seg_ptr = reinterpret_cast<mlx5_wqe_inl_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
    auto* wqe_data_ptr = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(inl_seg_ptr) + sizeof(*inl_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    inl_seg.byte_count = HtoBE32(kBytes | MLX5_INLINE_SEG);

    ctrl_seg = {0};
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | kNumDS);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    static_assert(sizeof(*ctrl_seg_ptr) == 16 and sizeof(*raddr_seg_ptr) == 16 and sizeof(*inl_seg_ptr) == 4, "Invalid seg sizes");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<uint32_t*>(inl_seg_ptr), *reinterpret_cast<const uint32_t*>(&inl_seg));
    #pragma unroll
    for (uint32_t i = 0; i < kBytes / 4; ++ i)
        st_na_relaxed(wqe_data_ptr + i, val[i]);
}

// Registered-buffer RDMA WRITE WQE.  Unlike put_inline, the source payload
// remains in symmetric HBM and is described by an lkey-bearing data segment.
__device__ static __forceinline__ void ibgda_write_rdma_write_wqe(
    nvshmemi_ibgda_device_qp_t* qp,
    uint64_t laddr, __be32 lkey,
    uint64_t raddr, __be32 rkey,
    uint32_t bytes, uint16_t wqe_idx, void** out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_data_seg data_seg;

    auto* ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    auto* raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(
        reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    auto* data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(
        reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    data_seg.byte_count = HtoBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HtoBE64(laddr);

    ctrl_seg = {0};
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr),
                  *reinterpret_cast<const int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr),
                  *reinterpret_cast<const int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr),
                  *reinterpret_cast<const int4*>(&data_seg));
}

// RDMA READ WQE：与 WRITE 同构，opcode 换 RDMA_READ；raddr = 远端【源】，data seg = 本地【目的】。
__device__ static __forceinline__ void ibgda_write_rdma_read_wqe(nvshmemi_ibgda_device_qp_t* qp,
                                                                 uint64_t laddr,
                                                                 __be32 lkey,
                                                                 uint64_t raddr,
                                                                 __be32 rkey,
                                                                 uint32_t bytes,
                                                                 uint16_t wqe_idx,
                                                                 void** out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_data_seg data_seg;

    auto* ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    auto* raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    auto* data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    data_seg.byte_count = HtoBE32(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = HtoBE64(laddr);

    ctrl_seg = {0};
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_READ);

    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
}

// ---------------------------------------------------------------------------
// 对外 API
// ---------------------------------------------------------------------------

// inline 单值 RDMA WRITE：rptr 为本地对称地址(自动翻译到 dst_pe)，value 在寄存器即可。
// T 大小限 4/8/16 字节。不跨 cumem chunk(单值不可能跨 512MB 边界)。
template <typename T>
__device__ static __forceinline__ void put_inline(T* rptr, const T& value, int dst_pe, int qp_id) {
    static_assert(sizeof(T) == 4 or sizeof(T) == 8 or sizeof(T) == 16, "Unsupported inline size");
    __be32 rkey;
    uint64_t raddr;
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), dst_pe, &raddr, &rkey, qp->dev_idx);

    uint64_t base_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
    void* wqe_ptrs = ibgda_get_wqe_ptr(qp, base_wqe_idx);
    ibgda_write_rdma_write_inl_wqe<sizeof(T)>(
        qp, reinterpret_cast<const uint32_t*>(&value), raddr, rkey, static_cast<uint16_t>(base_wqe_idx), &wqe_ptrs);

    ibgda_submit_requests(qp, base_wqe_idx, 1);
}

// Credit-aware inline WRITE for completion notifications sharing a busy data
// QP.  The notification must be reserved after all earlier data WQEs and must
// not overwrite an in-flight ring slot when a long expert fan-in wraps the QP.
template <typename T>
__device__ static __forceinline__ void put_inline_with_credit(
    T* rptr, const T& value, int dst_pe, int qp_id) {
    static_assert(sizeof(T) == 4 or sizeof(T) == 8 or sizeof(T) == 16,
                  "Unsupported inline size");
    __be32 rkey;
    uint64_t raddr;
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    ibgda_get_rkey(
        reinterpret_cast<uint64_t>(rptr), dst_pe, &raddr, &rkey, qp->dev_idx);

    const uint64_t base_wqe_idx = ibgda_reserve_wqe_slots_with_credit(qp, 1);
    void* wqe_ptr = ibgda_get_wqe_ptr(qp, base_wqe_idx);
    ibgda_write_rdma_write_inl_wqe<sizeof(T)>(
        qp, reinterpret_cast<const uint32_t*>(&value), raddr, rkey,
        static_cast<uint16_t>(base_wqe_idx), &wqe_ptr);

    ibgda_submit_requests(qp, base_wqe_idx, 1);
}

// Warp-cooperative registered HBM WRITE.  A row that crosses registration
// chunks may use several WQEs, but they are reserved contiguously and posted by
// one doorbell only after every lane-owned WQE is ready.
__device__ static __forceinline__ void put_nbi_warp(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes,
    int dst_pe, int qp_id, int lane_id) {
    uint32_t num_wqes = 0;
    __be32 my_lkey = 0;
    uint64_t my_laddr = 0;
    __be32 my_rkey = 0;
    uint64_t my_raddr = 0;
    uint64_t my_chunk_size = 0;

    auto qp = ibgda_get_rc(dst_pe, qp_id);
    auto remaining_bytes = bytes;
    while (remaining_bytes > 0) {
        DG_DEVICE_ASSERT(num_wqes < 32);
        if (lane_id == static_cast<int>(num_wqes)) {
            my_chunk_size = min(
                remaining_bytes,
                ibgda_get_lkey_and_rkey(
                    my_laddr = req_lptr, &my_lkey,
                    req_rptr, dst_pe, &my_raddr, &my_rkey, qp->dev_idx));
        }
        const auto chunk_size = __shfl_sync(
            0xffffffff, my_chunk_size, static_cast<int>(num_wqes));
        DG_DEVICE_ASSERT(chunk_size > 0);
        remaining_bytes -= chunk_size;
        req_lptr += chunk_size;
        req_rptr += chunk_size;
        ++ num_wqes;
    }

    uint64_t base_wqe_idx = 0;
    if (lane_id == 0)
        base_wqe_idx = ibgda_reserve_wqe_slots_with_credit(qp, num_wqes);
    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);

    if (lane_id < static_cast<int>(num_wqes)) {
        const auto wqe_idx = base_wqe_idx + lane_id;
        auto wqe_ptr = ibgda_get_wqe_ptr(qp, wqe_idx);
        ibgda_write_rdma_write_wqe(
            qp, my_laddr, my_lkey, my_raddr, my_rkey,
            static_cast<uint32_t>(my_chunk_size),
            static_cast<uint16_t>(wqe_idx), &wqe_ptr);
    }
    __syncwarp();

    if (lane_id == 0)
        ibgda_submit_requests(qp, base_wqe_idx, num_wqes);
    __syncwarp();
}

// Batch one row request per active lane onto a shared (dst_pe, qp_id).  Each
// lane constructs all registration-boundary fragments for its row, while lane
// 0 reserves the combined WQE range and rings one doorbell for the full batch.
__device__ static __forceinline__ void put_nbi_warp_batch_rows(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes, bool active,
    int dst_pe, int qp_id, int lane_id) {
    auto qp = ibgda_get_rc(dst_pe, qp_id);

    uint32_t lane_num_wqes = 0;
    uint64_t remaining_bytes = active ? bytes : 0;
    uint64_t laddr = req_lptr;
    uint64_t raddr = req_rptr;
    while (remaining_bytes > 0) {
        __be32 lkey, rkey;
        uint64_t real_raddr;
        const uint64_t chunk = min(
            remaining_bytes,
            ibgda_get_lkey_and_rkey(
                laddr, &lkey, raddr, dst_pe, &real_raddr, &rkey,
                qp->dev_idx));
        DG_DEVICE_ASSERT(chunk > 0);
        ++ lane_num_wqes;
        laddr += chunk;
        raddr += chunk;
        remaining_bytes -= chunk;
    }

    uint32_t inclusive_wqes = lane_num_wqes;
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(
            0xffffffff, inclusive_wqes, offset);
        if (lane_id >= static_cast<int>(offset))
            inclusive_wqes += other;
    }
    const uint32_t lane_wqe_offset = inclusive_wqes - lane_num_wqes;
    const uint32_t total_wqes = __shfl_sync(
        0xffffffff, inclusive_wqes, 31);

    uint64_t base_wqe_idx = 0;
    if (lane_id == 0 and total_wqes != 0)
        base_wqe_idx = ibgda_reserve_wqe_slots_with_credit(qp, total_wqes);
    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);

    remaining_bytes = active ? bytes : 0;
    laddr = req_lptr;
    raddr = req_rptr;
    uint32_t local_wqe_idx = 0;
    while (remaining_bytes > 0) {
        __be32 lkey, rkey;
        uint64_t real_raddr;
        const uint64_t chunk = min(
            remaining_bytes,
            ibgda_get_lkey_and_rkey(
                laddr, &lkey, raddr, dst_pe, &real_raddr, &rkey,
                qp->dev_idx));
        const uint64_t wqe_idx =
            base_wqe_idx + lane_wqe_offset + local_wqe_idx;
        auto wqe_ptr = ibgda_get_wqe_ptr(qp, wqe_idx);
        ibgda_write_rdma_write_wqe(
            qp, laddr, lkey, real_raddr, rkey,
            static_cast<uint32_t>(chunk), static_cast<uint16_t>(wqe_idx),
            &wqe_ptr);
        ++ local_wqe_idx;
        laddr += chunk;
        raddr += chunk;
        remaining_bytes -= chunk;
    }
    DG_DEVICE_ASSERT(local_wqe_idx == lane_num_wqes);
    __syncwarp();

    if (lane_id == 0 and total_wqes != 0)
        ibgda_submit_requests(qp, base_wqe_idx, total_wqes);
    __syncwarp();
}

// 单线程 RDMA READ(非阻塞发起)：laddr 本地目的(须在对称堆内)，raddr 本地对称地址
// (翻译到 src_pe 的远端源)。阻塞语义 = get_thread(...) 后跟 quiet(src_pe, qp_id)。
__device__ static __forceinline__ void get_thread(uint64_t laddr, uint64_t raddr, size_t bytes, int src_pe, int qp_id) {
    auto qp = ibgda_get_rc(src_pe, qp_id);

    // 逐 chunk(512MB 粒度下几乎恒为单 chunk)
    while (bytes > 0) {
        __be32 lkey, rkey;
        uint64_t real_raddr;
        auto chunk = min(static_cast<uint64_t>(bytes),
                         ibgda_get_lkey_and_rkey(laddr, &lkey, raddr, src_pe, &real_raddr, &rkey, qp->dev_idx));

        uint64_t wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
        void* wqe_ptr = ibgda_get_wqe_ptr(qp, wqe_idx);
        ibgda_write_rdma_read_wqe(qp, laddr, lkey, real_raddr, rkey, static_cast<uint32_t>(chunk),
                                  static_cast<uint16_t>(wqe_idx), &wqe_ptr);
        ibgda_submit_requests(qp, wqe_idx, 1);

        laddr += chunk;
        raddr += chunk;
        bytes -= chunk;
    }
}

struct GetRequest {
    uint64_t laddr;
    uint64_t raddr;
    size_t bytes;
};

// Batch the same small request group for one token per active lane.  All
// tokens target one (src_pe, qp_id), so the warp reserves one contiguous WQE
// range and rings one doorbell after every lane has materialized its WQEs.
// Remote and local addresses may remain fully scattered; only the QP posting
// operation is coalesced.
template <uint32_t kNumRequests>
__device__ static __forceinline__ uint64_t get_batch_warp(
    const GetRequest (&requests)[kNumRequests], bool active,
    int src_pe, int qp_id, int lane_id) {
    auto qp = ibgda_get_rc(src_pe, qp_id);

    uint32_t lane_num_wqes = 0;
    #pragma unroll
    for (uint32_t request_idx = 0; request_idx < kNumRequests; ++ request_idx) {
        uint64_t laddr = requests[request_idx].laddr;
        uint64_t raddr = requests[request_idx].raddr;
        size_t bytes = active ? requests[request_idx].bytes : 0;
        while (bytes > 0) {
            __be32 lkey, rkey;
            uint64_t real_raddr;
            const auto chunk = min(
                static_cast<uint64_t>(bytes),
                ibgda_get_lkey_and_rkey(
                    laddr, &lkey, raddr, src_pe, &real_raddr, &rkey,
                    qp->dev_idx));
            DG_DEVICE_ASSERT(chunk > 0);
            ++ lane_num_wqes;
            laddr += chunk;
            raddr += chunk;
            bytes -= chunk;
        }
    }

    uint32_t inclusive_wqes = lane_num_wqes;
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(
            0xffffffff, inclusive_wqes, offset);
        if (lane_id >= static_cast<int>(offset))
            inclusive_wqes += other;
    }
    const uint32_t lane_wqe_offset = inclusive_wqes - lane_num_wqes;
    const uint32_t total_wqes = __shfl_sync(
        0xffffffff, inclusive_wqes, 31);

    uint64_t base_wqe_idx = 0;
    if (lane_id == 0 and total_wqes != 0)
        base_wqe_idx = ibgda_reserve_wqe_slots_with_credit(qp, total_wqes);
    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);

    uint32_t local_wqe_idx = 0;
    #pragma unroll
    for (uint32_t request_idx = 0; request_idx < kNumRequests; ++ request_idx) {
        uint64_t laddr = requests[request_idx].laddr;
        uint64_t raddr = requests[request_idx].raddr;
        size_t bytes = active ? requests[request_idx].bytes : 0;
        while (bytes > 0) {
            __be32 lkey, rkey;
            uint64_t real_raddr;
            const auto chunk = min(
                static_cast<uint64_t>(bytes),
                ibgda_get_lkey_and_rkey(
                    laddr, &lkey, raddr, src_pe, &real_raddr, &rkey,
                    qp->dev_idx));
            const uint64_t wqe_idx =
                base_wqe_idx + lane_wqe_offset + local_wqe_idx;
            auto wqe_ptr = ibgda_get_wqe_ptr(qp, wqe_idx);
            ibgda_write_rdma_read_wqe(
                qp, laddr, lkey, real_raddr, rkey,
                static_cast<uint32_t>(chunk), static_cast<uint16_t>(wqe_idx),
                &wqe_ptr);
            ++ local_wqe_idx;
            laddr += chunk;
            raddr += chunk;
            bytes -= chunk;
        }
    }
    DG_DEVICE_ASSERT(local_wqe_idx == lane_num_wqes);
    __syncwarp();

    if (lane_id == 0 and total_wqes != 0)
        ibgda_submit_requests(qp, base_wqe_idx, total_wqes);
    __syncwarp();
    return base_wqe_idx + total_wqes;
}

// Reserve all READ WQEs for a small request group at once and publish them
// with one doorbell.  The returned producer index identifies exactly this
// group's last WQE, so a caller sharing the QP need not quiet later requests.
// Registration-boundary splits are included in the same reservation.
template <uint32_t kNumRequests>
__device__ static __forceinline__ uint64_t get_batch_thread(
    const GetRequest (&requests)[kNumRequests], int src_pe, int qp_id) {
    auto qp = ibgda_get_rc(src_pe, qp_id);

    uint32_t num_wqes = 0;
    #pragma unroll
    for (uint32_t request_idx = 0; request_idx < kNumRequests; ++ request_idx) {
        uint64_t laddr = requests[request_idx].laddr;
        uint64_t raddr = requests[request_idx].raddr;
        size_t bytes = requests[request_idx].bytes;
        while (bytes > 0) {
            __be32 lkey, rkey;
            uint64_t real_raddr;
            const auto chunk = min(
                static_cast<uint64_t>(bytes),
                ibgda_get_lkey_and_rkey(
                    laddr, &lkey, raddr, src_pe, &real_raddr, &rkey, qp->dev_idx));
            DG_DEVICE_ASSERT(chunk > 0);
            ++ num_wqes;
            laddr += chunk;
            raddr += chunk;
            bytes -= chunk;
        }
    }

    const uint64_t base_wqe_idx =
        ibgda_reserve_wqe_slots_with_credit(qp, num_wqes);
    uint32_t wqe_offset = 0;
    #pragma unroll
    for (uint32_t request_idx = 0; request_idx < kNumRequests; ++ request_idx) {
        uint64_t laddr = requests[request_idx].laddr;
        uint64_t raddr = requests[request_idx].raddr;
        size_t bytes = requests[request_idx].bytes;
        while (bytes > 0) {
            __be32 lkey, rkey;
            uint64_t real_raddr;
            const auto chunk = min(
                static_cast<uint64_t>(bytes),
                ibgda_get_lkey_and_rkey(
                    laddr, &lkey, raddr, src_pe, &real_raddr, &rkey, qp->dev_idx));
            const uint64_t wqe_idx = base_wqe_idx + wqe_offset;
            void* wqe_ptr = ibgda_get_wqe_ptr(qp, wqe_idx);
            ibgda_write_rdma_read_wqe(
                qp, laddr, lkey, real_raddr, rkey,
                static_cast<uint32_t>(chunk), static_cast<uint16_t>(wqe_idx), &wqe_ptr);
            ++ wqe_offset;
            laddr += chunk;
            raddr += chunk;
            bytes -= chunk;
        }
    }
    DG_DEVICE_ASSERT(wqe_offset == num_wqes);
    ibgda_submit_requests(qp, base_wqe_idx, num_wqes);
    return base_wqe_idx + num_wqes;
}

// CQ poll (相对 DeepEP：cons_idx 用 atomicMax 更新，允许多 warp 并发 quiet 同一 QP。
// 并发时每个等待者都等到自己快照的 prod_idx 完成为止，可能多等别人的 WQE，正确性不受影响)
__device__ static __forceinline__ void ibgda_poll_cq(nvshmemi_ibgda_device_cq_t* cq, uint64_t idx) {
    const auto cqe64 = static_cast<mlx5_cqe64*>(cq->cqe);
    const uint32_t ncqes = cq->ncqes;
    memory_fence_cta();
    if (ld_na_relaxed(reinterpret_cast<uint64_t*>(cq->cons_idx)) >= idx)
        return;
    uint16_t wqe_counter;
    do {
        wqe_counter = HtoBE16(ld_na_relaxed(&cqe64->wqe_counter));
    } while ((static_cast<uint16_t>(static_cast<uint16_t>(idx) - wqe_counter - static_cast<uint16_t>(2)) < ncqes));
    atomicMax(reinterpret_cast<unsigned long long*>(cq->cons_idx), static_cast<unsigned long long>(idx));
    memory_fence_cta();
}

// Wait for a previously returned producer index, rather than snapshotting the
// QP head and accidentally waiting for unrelated later work on a shared QP.
__device__ static __forceinline__ void wait_until(int peer, int qp_id, uint64_t completion_idx) {
    auto qp = ibgda_get_rc(peer, qp_id);
    ibgda_poll_cq(qp->tx_wq.cq, completion_idx);
}

// 等待 (dst_pe, qp_id) 上当前已提交的所有 WQE 完成
__device__ static __forceinline__ void quiet(int dst_pe, int qp_id) {
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    auto state = ibgda_get_state();
    uint64_t prod_idx = state->use_async_postsend ? ld_na_relaxed(qp->tx_wq.prod_idx) : ld_na_relaxed(&qp->mvars.tx_wq.ready_head);
    ibgda_poll_cq(qp->tx_wq.cq, prod_idx);
}

} // namespace deep_gemm::comm::ibgda
