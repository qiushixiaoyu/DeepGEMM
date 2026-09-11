# SM90 FP8 MegaMoE RDMA 分支与优化记录（历史归档）

历史记录截至：2026-08-18；归档说明更新：2026-09-11。

> 本文不是当前分支的运行说明。各节中的“当前”“保留”“已删除”和性能数据，
> 均指该节记录时的代码快照；旧环境变量、命令和分支名仅供历史追溯，不能作为
> 当前配置使用。当前支持范围、推荐配置和双机命令统一见
> [SM90 MegaMoE RDMA](docs/SM90_MEGAMOE_RDMA.md)，后续优化取舍见
> [清理记录](docs/SM90_MEGAMOE_RDMA_CLEANUP.md)。

特别注意当前实现与下文历史快照的差别：

- SM90 FP4/FP8 仅支持跨机 RDMA，单机 fallback 已删除；双机内部的 NVLink
  通信仍保留。
- dense V3 根据实际 token 数和 dispatch 分工推导 metadata producer CTA
  前缀；packed 保留完整 producer grid。不能把旧的“全部 SM”结论套到两者。
- FP8 的 split-phase 分派已用于全部生成形状，不再只限于宽 M128 shape；
  FP4 固定由 non-epilogue warp 2 开始的 helpers 做在线 decode，math 不参与。
- 当前仍有必要的 phase profiling 和诊断开关；历史“已移除”的描述不代表
  当前不可用。主性能测量应关闭这些打点，并使用 Kineto 的目标 kernel 时间。

## 2026-08-18 固定生产路径（历史快照）

本节记录 2026-08-18 的状态，覆盖更早历史快照，但不覆盖上述当前发布说明。

- scheduler 只保留 eager count，lazy producer/cache/prefix-wait 实现及
  `DG_MEGA_MOE_SCHEDULER_COUNT_IMPL` 已删除。专家按本地 expert ID 固定顺序
  调度；load/remote-load/wave/staged-remote、decode active-expert 和 dispatch
  follow-schedule 等实验实现及其全部环境变量/编译宏均已删除。
- 跨节点 dispatch/combine 固定使用 per-expert ready。workspace cleanup
  结束后只做本 rank 的 grid sync，不再执行 tag3 跨 rank barrier。
  `DG_MEGA_MOE_DISPATCH_IMPL`、`DG_MEGA_MOE_COMBINE_IMPL` 及其 host/JIT/kernel
  选择参数均已删除；跨机由拓扑固定为 expert-ready，单机固定为原生 NVLink
  协议，不再将后者暴露为 legacy 模式。
- dispatch metadata 不再支持直发 A/B：调用方请求的 `max_token <= 256`
  时固定使用 dense V3 整发，否则固定使用 packed 压缩；
  `DG_MEGA_MOE_DISPATCH_GATEWAY` 已删除。
- merged A/B loader 固定开启；跨节点 publisher 固定为 async、按目的 rank
  分组并 chain-poll CQ。inline RDMA publisher、串行 publisher fallback、
  token-bucket 限速及其相关环境变量均已删除。
- combine ready 的 mask/notify 固定使用 warp 并行实现，每个 CTA 的 publisher
  warp 都参与其 `(dst rank, expert)` 工作分片；
  `DG_MEGA_MOE_COMBINE_PARALLEL_READY` 已删除。该宏在删除前已经没有 kernel
  引用，仅是 JIT 侧的失效残留。
- L1、L2、combine staging 固定使用 bounded ring 容量策略，不再接受三个
  `*_STAGE_RING` 环境变量。小容量下 ring 容量可能等于 full-pool，这是容量
  策略的自然结果，不是关闭 ring。
- rank-major pool 和 rank-major pair task 实现已删除。
- gateway metadata 的 producer 不再支持限定活跃 SM；
  `DG_MEGA_MOE_GATEWAY_ACTIVE_METADATA_SMS` 及对应分支已删除，固定由全部 SM
  分摊 expert metadata，并以全部 CTA 到达作为 direction 完成条件。
- FP8 swap-A/B 不再提供 kill switch；`DG_SM90_FP8_SWAP_AB` 已删除，只保留
  `should_use_swap_ab_for_mega_moe_sm90()` 按 shape 自动选择。
- SM90 FP8 MegaMOE 内融合共享专家的实验功能已删除，包括
  `SGLANG_MEGA_MOE_FUSE_SHARED_EXPERT`、专用 Python/C++/TVM FFI API、kernel
  模板分支及为共享专家保留的 L2 scratch。共享专家恢复走 SGLang 原有的
  独立计算/stream overlap 路径；其他 MoE 后端的通用共享专家功能不受影响。
- FP8 merged A/B loader 固定为唯一实现，旧双 loader 路径和
  `DG_MEGA_MOE_MERGE_AB_LOADER` 编译宏均已删除。FP4 的同名实验分支本轮保留。

保留下来的 `DG_MEGA_MOE_DISPATCH_GATEWAY_{PACKED,DENSE_V3}` 只是在 JIT
生成源码时写入的内部编译标记，不再读取同名环境变量。async publisher
直接绑定 inter-node 拓扑，不再有独立编译开关或运行时 A/B。

### 两个按 shape 自动选择的内部优化

- `split_phase_hot_path` 不改变任务和同步协议；scheduler 仍按原顺序取 block，
  但把 L1/L2 分派给两个 phase 在编译期已知的 callback。这样 phase 判断只留在
  外层分派点，编译器可消掉各自重计算体里的分支并专门化地址计算。当前仅对
  `BLOCK_M=128`、`BLOCK_N=256` 且 `hidden>=7168` 的宽 shape 开启。
- L2 arrival counter 用“完成一个 L1 producer 就加一，达到期望数量后启动 L2”
  替代通用路径的 64-bit 到达 bitmask 和部分 full-CTA 同步。它只在 producer
  数量和拓扑已经验证的 `128x256/512-epilogue-thread` 路径，以及 batch
  4～128 的 `64x256/256-epilogue-thread` decode 路径开启；其他 shape
  继续使用通用同步。

这两项都是由已选出的 kernel shape 自动派生的内部编译特化，不读取环境变量，
也不是保留给用户的 A/B 分支。

## 2026-08-07 状态补充（历史快照）

本节覆盖下文 2026-08-05 的“当前”描述；下文继续保留，作为简化基线和历史优化分支的查阅记录。

- 当前开发分支：`experiment/mega-moe-combine-expert-ready`。
- 当前 HEAD：`cf99dde67`。`7bae60048` 增加 expert-ready 子阶段打点和 combine 多行批量 doorbell 实验；`cf99dde67` 将 ready mask/notify 改为 warp 并行，并保留串行回退路径。
- dispatch 与 combine 均已采用 per-expert ready 协议；dispatch QP/metadata、三条 READ 的 reserve/doorbell，以及 combine full-pool staging 等后续改造已经落在本分支历史中。
- scheduler 新增 `DG_MEGA_MOE_SCHEDULER_COUNT_IMPL=eager|lazy`。公开 Python/C++ 接口不变，默认值仍为 `eager`，只有显式设置 `lazy` 才启用本轮实验路径。
- lazy 路径由一个全局 producer warp 按 expert 聚合各 source rank 的 count，并以 `(launch_epoch << 32) | count` 发布到 internode 路径下闲置的 `recv_count_sum`；consumer scheduler 按 expert 等待共享结果。没有新增 SymmBuffer 空间。

本轮提交演进：

| 提交 | 内容 | 决策 |
|---|---|---|
| `225c87404` | scheduler-local lazy cache，每个 scheduler 重复轮询 source count | 精度通过；性能回退 6%～19% |
| `278fc30c7` | single global producer，lane 并行聚合 source，epoch-tagged shared cache | 精度通过；最终保留的 lazy 实验实现 |
| `4d06101ba` | 多个 dispatch warp 按 expert 分片发布 | 精度通过；低 batch 退化，拒绝 |
| `2a127b975` | 回退 sharded producer，恢复 single producer | 当前 eager/lazy scheduler 功能基线 |
| `7bae60048` | ready 五段 profile；同 `(dst_rank, expert)` 多行一次 reserve/doorbell | 精度通过；batch 路径保留为显式实验开关 |
| `cf99dde67` | warp 并行 destination mask 与 per-rank ready notify | 精度通过；b64/b256 三模型均明显改善，默认开启 |

最终 single-producer tree 已在 COMM5、COMM6 的 `mega_moe_rdma` 容器中完成双机 16 卡验证：full-remote t64、同一 SymmBuffer t256 A/B 交替 100 次、7 种极端路由，以及 `NVSHMEM_QP_DEPTH=4096` 下 full-remote t8192 全部通过；最大归一化 diff 约 `0.0006`，无 NaN/nonfinite。

性能方面，single-producer lazy 相对 eager 在 DeepSeekV4Flash、DeepSeekV4Pro、GLM5.2 的 batch 1/64/256 共 9 个点中，仅 Flash/b64 快 3.4%，其余 8 点慢 1.4%～4.4%。phase profile 表明全量 count wait 已从 dispatch barrier 中消失，但逐 expert cache wait、固定前缀扫描和 producer/consumer 调度成本抵消了提前 pull 的收益。因此当前只保留 lazy 实验入口，不切换默认路径。

详细命令、精度、性能及 phase 数据见 `../test_logs/20260807_lazy_shared_count/RESULTS.md` 和 `../test_logs/20260807_lazy_shared_count/COMMANDS.md`。

### Scatter publish 四段 profiling

`b04d9299e` 新增 `staging`、`arrival`、`WQE+doorbell`、`expert ready` 四段；`ab4804e0c` 先选 scatter 最慢的 SM，再读取该 SM 的四段，避免各字段独立取 max 后不能相加。profile buffer 每 rank 增加 8 KiB，公开接口不变。

COMM5/COMM6 上开启 profiler 的 full-remote t64 精度通过，最大归一化 diff 约 `0.0006`。三模型 b64/b256 的五次 critical-SM 中位样本显示：staging 仅占 scatter 4.2%～6.0%，arrival 占 2.9%～4.0%；b64 的 expert-ready 占 66.2%～73.1%，而 Flash/GLM b256 的逐行 WQE+doorbell 增长到 43%～44%。下一步优先拆解/优化 ready 协议，并在高 batch A/B 验证同一 `(dst_rank, expert)` 的多行 WQE 批量 doorbell。

详细表格、命令和原始日志见 `../test_logs/20260807_scatter_split_profile/RESULTS.md` 和 `../test_logs/20260807_scatter_split_profile/COMMANDS.md`。

### Expert-ready 与 batch doorbell 优化

`7bae60048` 将 ready 进一步拆成 mask、atomic、system fence、notify、CTA sync 五段，并增加 `put_nbi_warp_batch_rows`：同一 warp 的 active lane 各负责一行，经 WQE-count prefix sum 后由 lane 0 一次 reserve、一次 doorbell。combine 继续使用 `QP(dst_rank, local_expert)`，不增加 `quiet`，公开接口不变。profile buffer从 22 增至 27 个 slot，每 rank 增加 10 KiB。

五段基线显示串行 ready notify 占 ready 的约 76%～80%。`cf99dde67` 因此将 mask 形成与 ready notify 改为一个 warp 并行；lane 0 仍独占 per-expert atomic、最终 producer 判定和 system fence，ready WQE 与先前 data WQE 仍在同一 RC QP 上保持顺序。当时曾通过 `DG_MEGA_MOE_COMBINE_PARALLEL_READY` 提供串行 A/B；该开关和串行路径现已删除，详见文首固定生产路径。

最终组合“parallel ready + batch doorbell”在 COMM5/COMM6 上通过 full-remote t64、同一 SymmBuffer t256 A/B 交替 100 次和 QP depth 4096 下 full-remote t8192，最大归一化 diff `0.0006`，无 NaN/nonfinite。parallel ready 使三模型 b64/b256 的 ready 段下降 56.7%～73.6%，notify 下降 77.7%～86.4%。batch doorbell 在 parallel-ready 基线上使 Flash/GLM b256 的 WQE 段下降 39.7%/42.8%；最终端到端相对原始逐行+串行 ready 分别改善 9.8%/13.1%。

当前 `DG_MEGA_MOE_COMBINE_BATCH_DOORBELL` 默认仍为 0：本轮只按计划验证 Flash/GLM b256，待补低 batch、Pro 和路由矩阵后再决定是否默认启用。详细命令、样本、phase 表和原始日志见 `../test_logs/20260807_ready_batch_doorbell/RESULTS.md`。

## 2026-08-05 决策（历史快照）

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
