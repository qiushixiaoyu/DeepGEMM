# SM90 FP8 MegaMOE RDMA 分支与优化记录

更新时间：2026-08-07

## 2026-08-07 当前状态补充

本节覆盖下文 2026-08-05 的“当前”描述；下文继续保留，作为简化基线和历史优化分支的查阅记录。

- 当前开发分支：`experiment/mega-moe-combine-expert-ready`。
- 当前 HEAD：`ab4804e0c`。运行时功能仍以 `2a127b975` 为基础；后续 `b04d9299e`、`ab4804e0c` 只扩展 phase profiler，将 scatter 拆成四段并保证四段来自同一个 critical SM。
- dispatch 与 combine 均已采用 per-expert ready 协议；dispatch QP/metadata、三条 READ 的 reserve/doorbell，以及 combine full-pool staging 等后续改造已经落在本分支历史中。
- scheduler 新增 `DG_MEGA_MOE_SCHEDULER_COUNT_IMPL=eager|lazy`。公开 Python/C++ 接口不变，默认值仍为 `eager`，只有显式设置 `lazy` 才启用本轮实验路径。
- lazy 路径由一个全局 producer warp 按 expert 聚合各 source rank 的 count，并以 `(launch_epoch << 32) | count` 发布到 internode 路径下闲置的 `recv_count_sum`；consumer scheduler 按 expert 等待共享结果。没有新增 SymmBuffer 空间。

本轮提交演进：

| 提交 | 内容 | 决策 |
|---|---|---|
| `225c87404` | scheduler-local lazy cache，每个 scheduler 重复轮询 source count | 精度通过；性能回退 6%～19% |
| `278fc30c7` | single global producer，lane 并行聚合 source，epoch-tagged shared cache | 精度通过；最终保留的 lazy 实验实现 |
| `4d06101ba` | 多个 dispatch warp 按 expert 分片发布 | 精度通过；低 batch 退化，拒绝 |
| `2a127b975` | 回退 sharded producer，恢复 single producer | 当前代码状态 |

最终 single-producer tree 已在 COMM5、COMM6 的 `mega_moe_rdma` 容器中完成双机 16 卡验证：full-remote t64、同一 SymmBuffer t256 A/B 交替 100 次、7 种极端路由，以及 `NVSHMEM_QP_DEPTH=4096` 下 full-remote t8192 全部通过；最大归一化 diff 约 `0.0006`，无 NaN/nonfinite。

性能方面，single-producer lazy 相对 eager 在 DeepSeekV4Flash、DeepSeekV4Pro、GLM5.2 的 batch 1/64/256 共 9 个点中，仅 Flash/b64 快 3.4%，其余 8 点慢 1.4%～4.4%。phase profile 表明全量 count wait 已从 dispatch barrier 中消失，但逐 expert cache wait、固定前缀扫描和 producer/consumer 调度成本抵消了提前 pull 的收益。因此当前只保留 lazy 实验入口，不切换默认路径。

详细命令、精度、性能及 phase 数据见 `../test_logs/20260807_lazy_shared_count/RESULTS.md` 和 `../test_logs/20260807_lazy_shared_count/COMMANDS.md`。

### Scatter publish 四段 profiling

`b04d9299e` 新增 `staging`、`arrival`、`WQE+doorbell`、`expert ready` 四段；`ab4804e0c` 先选 scatter 最慢的 SM，再读取该 SM 的四段，避免各字段独立取 max 后不能相加。profile buffer 每 rank 增加 8 KiB，公开接口不变。

COMM5/COMM6 上开启 profiler 的 full-remote t64 精度通过，最大归一化 diff 约 `0.0006`。三模型 b64/b256 的五次 critical-SM 中位样本显示：staging 仅占 scatter 4.2%～6.0%，arrival 占 2.9%～4.0%；b64 的 expert-ready 占 66.2%～73.1%，而 Flash/GLM b256 的逐行 WQE+doorbell 增长到 43%～44%。下一步优先拆解/优化 ready 协议，并在高 batch A/B 验证同一 `(dst_rank, expert)` 的多行 WQE 批量 doorbell。

详细表格、命令和原始日志见 `../test_logs/20260807_scatter_split_profile/RESULTS.md` 和 `../test_logs/20260807_scatter_split_profile/COMMANDS.md`。

## 当前决策

当前 `mega_moe_rdma` 分支作为“最小正确性基线”：保留双节点 RDMA 所需的功能和已经验证过的精度修复，移除第一轮性能优化及其观测代码。该版本用于重新设计通信方案前的正确性基准，不代表最终性能版本。

不回退到最早的 RDMA 提交 `d29074db1`。该版本将 dispatch 的 SF/weight staging 复用到 combine buffer，已经确认会触发 RNIC 写与 GPU L2 cache alias，full-remote 场景存在精度问题。最小可用语义必须包含 `0c7b8e452` 引入的独立、对称、128-byte 对齐 dispatch staging。

## 分支快照

| 仓库 | 分支 | 提交 | 用途 |
|---|---|---|---|
| DeepGEMM | `mega_moe_rdma` | `bec6186a3`（简化实现提交） | 当前最小正确性基线 |
| DeepGEMM | `backup/mega-moe-rdma-optimized-20260805` | `7acd19940` | 完整优化快照；本地与 `origin` 均已保存 |
| sglang | `mega_moe_rdma` | `5c2006914` | 当前必要的 NVSHMEM SymmetricMemory 接入 |
| sglang | `backup/mega-moe-rdma-adapter-20260805` | `5c2006914` | sglang 配套适配快照；本地与 `origin` 均已保存 |

后续性能实验建议从当前最小基线新建 `experiment/*` 分支，逐项加入优化并分别做精度、性能 A/B；不要直接在备份分支上继续开发。

## 提交与功能分层

| 提交 | 分层 | 内容 | 当前基线 |
|---|---|---|---|
| `d29074db1` | Core RDMA | NVSHMEM device-link/JIT 初始化、direct IBGDA verbs、跨节点 metadata、dispatch READ、combine WRITE、跨节点 barrier | 保留必要部分 |
| `0c7b8e452` | Correctness | 独立 dispatch staging 与 `ld.global.cv`，解决 RNIC/L2 cache alias | 必须保留 |
| `cd7c36653` | Performance/observability | phase-level profiling、row-combine、行级 registered WRITE、批量 quiet | 已移除，保存在优化备份分支 |
| `7acd19940` | Performance/memory | full-pool staging 改 bounded ring、metadata QP-prefix quiet、跳过冗余 QP/public quiet | 已移除，保存在优化备份分支 |
| `bec6186a3` | Simplification | 恢复逐 vector combine WRITE 和完整 barrier；删除 profiling、row staging、优化专用 verbs/测试工具 | 当前实现 |

## 当前保留的必要修改

当前相对机内 SM90 FP8 MegaMOE 基线只保留 9 个运行时文件的修改：

1. `build_sgl_deep_gemm.sh`：链接 NVSHMEM host library。
2. `csrc/apis/sm90_mega.hpp`：为 dispatch SF/weight 分配独立对称 staging。
3. `csrc/jit/compiler.hpp`：对含 NVSHMEM 的 JIT kernel 执行 relocatable compile 和 device-link。
4. `csrc/jit/handle.hpp`：对 JIT module 执行 `nvshmemx_cumodule_init`。
5. `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`：16 rank 时注入 internode 宏；当前拓扑假设每节点连续 8 rank。
6. `deep_gemm/include/deep_gemm/comm/barrier.cuh`：跨节点先完成 direct-IBGDA QP，再执行 `nvshmem_sync_all()`。
7. `deep_gemm/include/deep_gemm/comm/ibgda.cuh`：仅保留 inline WRITE、READ、quiet 及其必要 WQE/CQ/rkey 支撑。
8. `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`：将机内远端访存语义映射到 RDMA，并保留两处 system fence。
9. `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`：逐 source 等待 metadata marker 并在本地汇总 expert token count。

sglang 的 `5c2006914` 仍然需要保留：它在分配 SymmetricMemory 前选择 NVSHMEM backend，使跨节点 PE 使用同一对称堆语义。

## 当前 dispatch/combine 语义

当前路径刻意保持接近机内 MegaMOE：

```text
机内 map() + vector store
    -> 跨节点 put_inline + per-QP completion

机内 remote load/TMA
    -> 跨节点 token/SF/weight 三次 RDMA READ + 同 QP quiet

机内 NVLink barrier
    -> direct-IBGDA 全 QP completion + nvshmem_sync_all + 原 grid sync

机内 combine vector scatter
    -> 跨节点每个 ScatterVec 一次 inline WRITE + 逐消息 quiet
```

同节点访问仍走原 NVLink 路径，L1/L2 GEMM、scheduler 主流程和本地 combine-reduce 不做性能结构调整。

## 本轮移除项

- `DG_MEGA_MOE_PHASE_PROFILE` 及 kernel 内 phase cycle 统计/打印。
- `DG_MEGA_MOE_ROW_COMBINE` 及 row-combine JIT 分支。
- full-pool row staging 和固定 16 MiB bounded staging ring。
- 16 个 lane fragment 合并为一个 128/256-byte registered RDMA WRITE。
- 每个 epilogue warp 按目标 PE 批量 quiet。
- metadata touched-QP 前缀 quiet、cleanup/combine 跳过 QP scan，以及跳过 public `nvshmem_quiet()`。
- 只服务上述优化的 `put_nbi_warp`、普通 registered RDMA WRITE WQE、批量 post 参数。
- 未使用的 NIC atomic、P2P helper 和 barrier signal layout 预留。
- phase/profile 参数和性能 sweep；需要时从优化备份分支取回。

因此当前分支不再支持 `DG_MEGA_MOE_ROW_COMBINE=1` 或 `DG_MEGA_MOE_PHASE_PROFILE=1`。设置这些变量不会启用对应路径。

## 接口与显存

- 公开 Python/C++ MegaMOE 调用接口没有变化。
- 保留的额外显存仅是精度必需的 dispatch staging，大小随 pool token 数增长。
- 已移除 combine row staging，因此不存在 full-pool 6–12 GiB 或 bounded 16 MiB 的 combine staging 额外占用。
- 压力场景继续使用 `NVSHMEM_QP_DEPTH=4096`；该值是每条 QP 的 WQE ring 深度，不是 QP 数量。

## 验证状态

历史 `0c7b8e452` 基线已经通过 random t256 和 full-remote t2048，归一化 diff 约 `0.0006`。当前 `bec6186a3` 在该语义上进一步删除未使用 helper、诊断和测试资产；Mac 已完成 diff、shell/Python 语法、残留符号及 host/device staging 布局静态检查。

当前尚未执行新的双节点 JIT/精度验证。后续必须在当时有效的两个 Pod 中重新同步源码、构建 wheel 并 `--force-reinstall`，使用全新的 `DG_JIT_CACHE_DIR`，先跑 full-remote t64，再跑 t2048 短复跑和完整精度矩阵。历史 wrapper 含旧 Pod/IP，并且部分命令强制 row-combine，不能直接复用。

## 恢复和参考

查看完整优化版：

```bash
git switch backup/mega-moe-rdma-optimized-20260805
```

只查看某个优化文件而不切换当前分支：

```bash
git show backup/mega-moe-rdma-optimized-20260805:<path>
```

重要历史资料：

- `../test_logs/20260721_153000/accuracy_root_cause.md`
- `../test_logs/20260721_phase_profile/PROFILE_AB_SUMMARY.md`
- `../test_logs/20260721_bounded_barrier/TEST_SUMMARY.md`
- `/Users/Shared/yd/work_context/SM90_MegaMOE_RDMA_优化进展与后续方案.md`
