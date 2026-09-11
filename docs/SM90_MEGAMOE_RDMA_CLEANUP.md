# SM90 MegaMoE RDMA：有效优化保留与实验清理

2026-09-09。基于 experiment/mega-moe-combine-expert-ready / c12738ad8。
本次是代码归并和清理，不是新性能优化，也不自动改变生产默认开关。

本文保留 2026-09-09 的清理记录。当前发布使用说明、运行条件和双机命令统一见
[SM90 MegaMoE RDMA](SM90_MEGAMOE_RDMA.md)，不要把下方历史证据目录当作随仓库发布的依赖。

## 2026-09-11：固定路径与历史说明整理

- FP8 删除恒为 true 的 split-phase 选择参数、未实例化的动态 phase 分支及
  仅服务该分支的遍历 helper；保留静态 phase/K 分派和原 scheduler 顺序。
- FP4 删除公开调用链恒为 false/0 的 math-decode 选择参数及不可达实现；
  helper 起始位置由参数改为固定的 non-epilogue warp 2，保留现有 helper
  数量、decode-ready、math 等待和 stage credit。
- FP8 dispatch QP 固定使用 local expert，删除不可达的 per-warp QP 表达式。
- 旧分支说明明确标为历史归档，修正当前 warp 分工和 profiling 注释。
- 本轮不修改 LTO、公开接口校验、资源生命周期、workspace 布局、通信协议、
  环境变量默认值或有效优化的启用范围。内部模板参数减少不改变 Python 调用
  签名；修改后仍需重建 wheel，静态检查不代表已完成 GPU 性能回归。

## 2026-09-11：发布前冗余与诊断修正

- FP8 NVSHMEM 状态打印按实际 `npes` 遍历远端 heap 表，检查空表；仅首个
  epoch 的各 NVLink domain leader 打印，避免 EP16 的固定下标越界。
- 删除 FP8 scatter 的无效共享 remote-row/last-producer 标记，将最后生产者
  判断留在同一个 leader 的局部变量中；原有 CTA/WG 同步、arrival、release
  和 ring credit 保留。FP4 同步删除恒 false 的 remote-row 标记分支。
- 删除强制定义的 IDE 宏、固定协议 bool 和固定关闭的 SFA loader 选择项；
  固定的 full-row/arrival/reserved 区域仍保持相同大小和偏移。
- 删除仓库内无调用的 IBGDA `put_inline`、`put_nbi_warp`、`get_thread`、
  `get_batch_warp`。带 credit 的通知、批量 READ/WRITE、精确完成等待以及
  SM100 仍使用的共享 barrier 保留，不改变底层 CQ/WQE 实现。
- 修正旧 metadata/global-barrier、NVSHMEM 历史故障范围和已验证 opt-in
  配置的注释。README 明确 `DG_COMM_KERNEL_DEBUG` 是 kernel 后清空整个
  symmetric buffer 的破坏性诊断，不是仅打印日志。

本轮不调整有效优化开关、形状/密度阈值、LTO、公开 API 或内存布局。
新增 CPU 源码契约和边界测试不代替重建 wheel 后的双机精度验证。

## 2026-09-11：失效 profiling 与剩余固定路径清理

- FP8 删除没有有效赋值的 publisher 子阶段计时、scatter-write 计数，以及
  恒 false 的生产者 expert-ready 记录及其读取/输出代码。保留有效指标和
  接收端 longest-ready 诊断；profiling 默认值与 SILENT 开关语义不变。
- 保留 profiling 缓冲区大小及有效 slot 编号，旧 slot 8、15、19–26 只作
  reserved。删除仅服务于失效 expert 记录的行重叠断言，保留有效的 SM 数量
  边界检查。日志消费者不得把删除的字段补零当作真实测量值。
- 两种精度删除 packed/dense-V3 配置下不可达的旧 metadata 发布块，保留
  其外部 dispatch rendezvous 和当前 gateway 发布/转发顺序。
- FP8 删除无正常 JIT 入口的 sparse-stakeout、idle-nanosleep 分支；保留
  原 dense stake-out 原子操作。FP4 自适应退避及其所有有效控制不变。
- FP8 固定为当前两条公开路径共有的 64 dispatch / 64 non-epilogue 线程，
  删除无调用的额外 idle-warp 分支；静态检查同步约束为当前受支持拓扑。
  保留 math 的 256/512 线程选择及现有寄存器请求，未修改 setmaxnreg/LTO。
- 删除未引用的局部常量、FP4 combine 的无消费者 shuffle、两个内部 helper
  的未使用形参；不改变公开 API、容量计算或 ring sizing。
- 统一在线 FP4 decode、线程配额、已验证 opt-in 配置和 profiling 的说明。

本轮不修改有效优化、环境默认值、形状阈值、workspace 布局或 RDMA 生命周期。
本地源码契约/等价性检查不能替代 Linux wheel 重建和双机数值/性能验证。

以下为 2026-09-09 的历史清理记录。

## 保留的实现

- FP4：V38 完整有效路径，包括在线 FP4 decode、paired PRMT、packed GMMA
  descriptor、行内并行量化、N64 decode-ready、decode 展开、SFB shared
  lookahead、4/8 helper 拓扑与配套寄存器分配、跳过冗余 math-B 等待、
  activation row-TMA。保留 SFB TMA、密度分桶和基础 publisher idle backoff。
- FP8：packed GMMA descriptor、swap M64 activation row-TMA、
  重权重密度 (24,32] 的 streaming swap/行内并行量化、
  轻权重 M64 的逐块 hybrid，以及 M128/N256/四个 math WG 的 inactive-WG 跳过。
- hybrid 的完整 math-WG 入口同步、ring credit/退役、发布顺序、精度公式、
  形状/密度检查均保留。FP4 没有预解码缓存。
- dense/packed 两种 RDMA 协议均保留：按调用者请求容量选择，≤256 为
  dense V3，更大为 packed；不是根据内部对齐后的容量选择。
- 通用 fallback、必要的诊断/计数开关仍保留，不把“默认关闭”当作无效依据。

有效配置仍为 opt-in，保持历史启用范围。FP8 B128 改进并非所有 batch 都变快：
最近 GLM 两轮均值 B128 降低 8.75%，B256 降低 5.66%，B96 增加 0.55%。
B64 的微小变化不算稳定优化收益。

## 删除/不再引入的实验

| 范围 | 实验 | 决策依据 |
|---|---|---|
| FP4 | L2 arrival counter | 泛化未显示稳定收益；保留原 bitmask，同步未删除 |
| FP4 | publish row mask | dispatch atomicOr、mask 维护抵消 metadata traversal 收益 |
| FP4 | GMMA descriptor reuse | 不保留复用分支；descriptor 打包本身保留 |
| FP4 | 强制 packed 小容量 | B16/64 两轮回退约 5.8%/6.8%；仅删除强制入口 |
| FP4 | progress budget/yield、few-pending/long-idle sleep | 全范围不稳定或回退；基础 64→512 ns 退避保留 |
| FP8 | SFA float2 | 无收益且有严格精度检查失败 |
| FP8 | device LTO/寄存器调整/publisher backoff | 无整体收益；不能据此删除有效的 FP4 LTO |
| FP8 | 强制 swap、M128 两 WG、M64 narrow-N、expert wave cap | 回退、精度风险或无稳定收益 |
| FP8 | early weight-SF load | B64 两轮方向不一致；实际 LDG 仍位于等待之后 |

sidecar / CPU proxy 已在 c12738ad8 删除，本次不重新引入。
whole-K SFA cache、normal-M64 row-TMA 扩展和其他被否决的实验只保留历史副本。
不删除 FP8 原有 L2 counter，也不删除 RDMA/combine 的 arrival counters。

## 配置复现与限制

FP4 的 V38 配置：

```bash
export DG_MEGA_MOE_FP4_SFB_N_CONTIGUOUS=1 DG_MEGA_MOE_FP4_SFB_TMA=1
export DG_MEGA_MOE_FP4_PAIRED_PRMT=1 DG_MEGA_MOE_FP4_PACKED_GMMA_DESC=1
export DG_MEGA_MOE_FP4_SFB_BROADCAST_OVERRIDE=0 DG_MEGA_MOE_FP4_FINE_BUCKET_CAP=2
export DG_MEGA_MOE_FP4_SCALE_FLOAT2_OVERRIDE=-1 DG_MEGA_MOE_FP4_EXPERTS_PER_WAVE=0
export DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MODE=0
export DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS=64
export DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MAX_NS=512
export DG_MEGA_MOE_FP4_ROW_PARALLEL_QUANT=1 DG_MEGA_MOE_FP4_N64_DECODE_READY=1
export DG_MEGA_MOE_FP4_DECODE_FULL_UNROLL=1 DG_MEGA_MOE_FP4_SFB_SHARED_LOOKAHEAD=2
export DG_MEGA_MOE_FP4_BALANCED_WG_REGISTERS=1 DG_MEGA_MOE_FP4_DECODE_REGISTER_BOOST=1
export DG_MEGA_MOE_FP4_EIGHT_HELPERS=2 DG_MEGA_MOE_FP4_EIGHT_MATH_REGISTERS=2
export DG_MEGA_MOE_FP4_SKIP_MATH_B_INPUT_WAIT=1 DG_MEGA_MOE_FP4_ACTIVATION_ROW_TMA=1
```

该 FP4 recipe **还必须**设置配套的 DG_JIT_NVSHMEM_LTO_LINKER 和
DG_JIT_NVSHMEM_LTO_ARCHIVE：使用已验证的 NVSHMEM LTO archive、linker 和资源门禁。
源码未附带第三方二进制。仅设置上述开关而缺少 LTO 工具会明确报错，不能回退为
未经验证的寄存器配置。工具和 archive 的版本/哈希见下方 V38/三模型实验日志。
H20/CUDA 13.0/NVSHMEM 3.4.5/EP16 已验证；其他架构、编译器或拓扑需要重新验证。

FP8 保留配置：

```bash
export DG_MEGA_MOE_FP8_PACKED_GMMA_DESC=1 DG_MEGA_MOE_FP8_ACTIVATION_ROW_TMA=1
export DG_MEGA_MOE_FP8_STREAMING_DENSITY32=1 DG_MEGA_MOE_FP8_ROW_PARALLEL_QUANT=2
export DG_MEGA_MOE_FP8_HYBRID_MAX_ROWS=32 DG_MEGA_MOE_FP8_SKIP_INACTIVE_M_WG=1
```

由形状/密度规则决定实际启用，不按模型名字或 batch 列表打补丁。
API/workspace 的旧 row-mask 区域改为明确的 reserved padding，保留字节数和后续
偏移，避免把缓冲 ABI 改动混入此轮清理；无 mask 构建、清零或读取。
旧实验环境变量不再被读取，请从启动配置移除。

## 证据与可恢复性

以下路径相对于上层 mega_moe_rdma 工作区：

- backups/mega_moe_cleanup_20260909/：main.bundle、main_before.patch、
  fp4_v38.patch、fp8_v5.patch、fp8_v6.patch 和原工作区状态。
- test_logs/20260907_fp4_margin20_quick/activation_row_tma_full_v38/
- test_logs/20260907_fp4_l2_counter_removal/
- test_logs/20260907_fp4_gpu_publisher_packed/
- test_logs/20260908_multimodel_2n/
- test_logs/20260908_fp8_crossover_v3/
- test_logs/20260909_fp8_glm_v6/
- test_logs/20260909_mega_moe_cleanup/：本轮检查、命令与回归结果。

旧实验目录和测试日志不删除。保留已有未提交修改和 test_logs_local/。
本轮新增源码结构测试不代替 CUDA 编译、数值精度或性能验证。
性能口径始终为目标 MegaMoE kernel-only（Kineto），若引用 DeepEP 则是其
low-latency end-to-end，二者不表示相同端到端边界。
