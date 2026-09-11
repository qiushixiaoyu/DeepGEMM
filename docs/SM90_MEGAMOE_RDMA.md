# SM90 MegaMoE RDMA：支持范围与双机复现

适用于本分支的 `sgl-deep-gemm` wheel（Python 导入名为 `deep_gemm`）。
SM90 FP4/FP8 MegaMoE **仅支持跨机 RDMA**，已删除单机 fallback；单机及不完整
八卡域拓扑会在对称内存分配前被拒绝。双机内部的 NVLink 通信仍然保留。
其他 GEMM 和 SM100 算子的单机支持不受影响。
本说明整理已有实现和测试方法，不代表重新完成了一轮 GPU 验证。
历史优化取舍见 [清理记录](SM90_MEGAMOE_RDMA_CLEANUP.md)，packed manifest
的顺序和生命周期约束见 [协议说明](SM90_PACKED_MANIFEST_PROTOCOL.md)。

## 支持矩阵

“源码允许”不等于“所有组合都已上板验证”。发布前应在实际环境重新验精度。

| 项目 | FP4 MegaMoE | FP8 MegaMoE |
|---|---|---|
| 计算 | FP8 activation × packed E2M1 FP4 weight；kernel 内在线 decode 后执行 WGMMA | FP8 E4M3 activation × FP8 E4M3 weight；无 FP4 decode |
| 权重 scale / recipe | packed UE8M0，`(1, 1, 32)` | FP32 block scale，`(128, 128, 128)` |
| 输出 / 激活 | BF16 / SwiGLU | BF16 / SwiGLU |
| 公开调用 | `fp8_fp4_mega_moe` | `fp8_mega_moe` |
| 权重转换 | `transform_weights_for_mega_moe_sm90_fp4` | `transform_weights_for_mega_moe_sm90` |
| GPU | SM90 Hopper；双机测试环境为每节点 8×H20 | 相同 |
| 节点内 / 跨节点 | 节点内 NVLink；跨节点 NVSHMEM IBGDA，融合 GPU publisher | 相同 |
| 跨节点拓扑 | 每个 NVLink 域固定 8 ranks；完整域、连续 rank 编号；总数 16～64 且为 8 的倍数；不支持单机 | 相同 |
| 已验证主拓扑 | 两节点、16 ranks、每 GPU 一个进程 | 相同；不把源码 rank 上限当作 EP64 实测结论 |
| 形状必要条件 | `hidden`、`intermediate` 为 128 的倍数，`intermediate <= 4096`；experts 可被 EP size 整除 | 相同；仍须满足实际 JIT tile/thread/layout 检查 |
| 当前推荐设备链接 | 满足 balanced-register 形状条件且启用时使用 full device LTO；其余走普通 RDC | 普通 NVCC RDC；不默认引入 FP8 LTO 实验 |
| 不包含的路径 | 单机 fallback、预解码权重缓存、sidecar publisher、CPU proxy、强制单机模拟 RDMA | 单机 fallback、sidecar / CPU proxy |

输入应使用 buffer 暴露的视图：`x` 为 FP8 E4M3，`x_sf` 为 FP32，
`topk_idx` 为 int64，`topk_weights` 为 FP32；路由必须是合法 expert ID。
不要自行改变 tensor stride 或手工拼装 workspace。API 的必要条件不是完整的
输入合法性检查清单；本轮文档清理不补充接口校验逻辑。

当前三模型的合成 MoE 层参数如下，不表示完整模型推理性能：

| 模型 | Hidden | Intermediate | Experts | Top-k | EP16 每 rank 的最小 RC QP 数 |
|---|---:|---:|---:|---:|---:|
| GLM5.2 / GLM5.3 | 6144 | 2048 | 256 | 8 | 17 |
| DeepSeekV4Pro | 7168 | 3072 | 384 | 6 | 25 |
| KimiK3 | 3584 | 3072 | 896 | 16 | 57 |

GLM5.3 在本项目使用用户确认的同一组 MoE 维度。小 batch 常用九点为
`1 2 4 8 16 32 64 128 256`，不是逐个整数穷举。大 batch 常用扩展点为
`512 1024 2048 4096 8192`；新路由、拓扑和边界组合需要额外验证。

## 协议、warp 分工与默认行为

- 两种精度都按**调用者请求的 buffer 容量**选择 dispatch metadata 协议：
  `<=256` 使用 dense V3，`>256` 使用 packed。内部容量向 384 对齐不改变选择。
  一次申请容量 8192 后测 B1，仍是 packed，不能与容量 256 的 B1 混为同一配置。
- dense V3 是正式的小容量路径，不是仅供诊断的旧实现。packed 发送 live payload
  和固定地址 manifest；两者都保留 epoch、发送完成 credit 和 ring 复用保护。
- warp 编号要区分 CTA 编号与 non-epilogue 相对编号。dispatch warps 位于 CTA
  开头；之后 non-epilogue warp 0 负责合并的 operand loader，warp 1 负责跨机
  publisher。FP4 decode helpers 从该区域的
  warp 2 开始；math warpgroups 处理 WGMMA 和 epilogue。FP8 无 decode helpers，
  weight scale 由 math 路径加载。线程数和配额由形状决定，不能把示例拓扑当作常量。
- 基础 FP4 publisher backoff 默认 `MODE=0`，由密度策略决定是否退避；启用时
  初始 64 ns、上限 512 ns。不是所有 batch 都无条件 sleep。
- `DG_MEGA_MOE_PHASE_PROFILE=1` 会插入时钟读取、计数和写回。
  `DG_MEGA_MOE_PHASE_PROFILE_SILENT=1` 仅关闭 profile 打印，不能消除这些开销。
  主性能测试必须关闭 phase profiling 和 device diagnostics；Kineto 本身也并非
  零扰动，比较时应使用相同采样和跨 rank 对齐方法。

## 构建与运行前提

已验证环境系列为 Linux、H20、CUDA 13.0、NVSHMEM 3.4.5，以及具备 NVSHMEM
对称内存后端的 PyTorch。wheel 元数据要求 Python >=3.10；测试脚本还需要
PyTorch、Triton 等依赖。主 README 的通用最低版本不能替代这里的 RDMA 要求。
尤其不能假设任意 PyTorch 安装都具有已配置的跨节点 symmetric-memory 后端。

1. 使用 `build_sgl_deep_gemm.sh` 构建并 `--force-reinstall` 同一 wheel 到两个节点。
   该构建脚本在 Python 环境的 `nvidia/nvshmem/lib` 查找 host library；运行时
   `DG_NVSHMEM_HOME` 则用于 JIT 查找 headers 和 device archive。
2. 跨机 symmetric buffer 必须来自已初始化的 NVSHMEM 后端，不能用仅节点内的
   allocator 替代。本文命令针对已配置此能力的 `mega_moe_rdma` 容器；其中
   `MEGA_MOE_FORCE_NVSHMEM_SYMM_MEM` / `TORCH_SYMMMEM` 是环境集成约定，
   并非本仓库负责实现的后端选择 API。迁移环境时必须核验实际 allocator。
3. 所有 ranks 在 NVSHMEM 初始化／注册内存**之前**设置
   `NVSHMEM_IB_ENABLE_RELAXED_ORDERING=0`。已有进程必须重启，运行中改环境不能
   修复已注册的 MR。
4. `NVSHMEM_IBGDA_NUM_RC_PER_PE >= experts / EP_size + 1`：expert 数据占用前
   E 个 QP，metadata gateway 使用额外的 QP E。还需可用的 GPUDirect RDMA、
   IBGDA、足够的注册内存，以及经过核对的 GPU/NIC rail 映射。
5. 本文使用 NVCC 后端：`DG_JIT_USE_NVRTC=0`。FP4 LTO 还需另行提供可执行
   `DG_JIT_NVSHMEM_LTO_LINKER` 和 `DG_JIT_NVSHMEM_LTO_ARCHIVE`。后者必须是含
   LTO 中间表示、与当前 NVSHMEM headers/host runtime 和 CUDA 工具链配套的
   device archive，不是把普通 archive 改名。仓库不附带这两个外部产物。

当前 LTO linker 接口为 `linker kernel.o archive_path archive output.cubin`；
它需以 `sm_90a` 做完整设备端 LTO 链接。归档其实现、构建参数和 archive hash。
缺少依赖时不要宣称使用推荐 FP4 LTO 配置；可以明确设置
`DG_MEGA_MOE_FP4_BALANCED_WG_REGISTERS=0` 测非 LTO 基线，但结果须单独标注。
FP8 普通 RDC 不需要这些 LTO 产物，仍需要普通 NVSHMEM device archive。

两条编译路径可以在同一 wheel 中共存。缓存键包含源码标记，但目前不包含外部
LTO linker/archive 的内容：更换依赖时使用新的 JIT 缓存目录并重新验证，避免
误用旧 cubin。需要验证动态寄存器时应检查实际 SASS，而不是只看环境变量。

## 推荐配置（opt-in，不修改源码默认值）

以下是已保留的有效配置组合，形状／密度门禁仍决定实际启用范围；不是承诺
每个开关都改善任意模型或 batch。设置环境后再启动进程、转换权重和触发 JIT。

FP4：

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
export DG_JIT_NVSHMEM_LTO_LINKER=/absolute/path/to/validated/linker
export DG_JIT_NVSHMEM_LTO_ARCHIVE=/absolute/path/to/validated/libnvshmem_device_lto.a
test -x "$DG_JIT_NVSHMEM_LTO_LINKER"
test -f "$DG_JIT_NVSHMEM_LTO_ARCHIVE"
```

FP8（继续使用普通 RDC，未合入最近的 FP8 LTO/动态寄存器实验）：

```bash
export DG_MEGA_MOE_FP8_PACKED_GMMA_DESC=1 DG_MEGA_MOE_FP8_ACTIVATION_ROW_TMA=1
export DG_MEGA_MOE_FP8_STREAMING_DENSITY32=1 DG_MEGA_MOE_FP8_ROW_PARALLEL_QUANT=2
export DG_MEGA_MOE_FP8_HYBRID_MAX_ROWS=32 DG_MEGA_MOE_FP8_SKIP_INACTIVE_M_WG=1
```

## 双机复现：先精度，再性能

下列为 Bash 命令模板，使用仓库内的测试入口，不依赖私有实验脚本。需替换
路径、提交、网络映射和端口；本轮只做本地静态检查，未重新在 GPU 执行此模板。
从干净启动环境开始，不继承旧实验的 `DG_*`、NVSHMEM 或 profiling 覆盖项。

### 1. 同步并安装

分别进入 COMM5、COMM6 的 `mega_moe_rdma` 容器。两边使用相同的新目录和提交；
不要覆盖已有实验。优先从本分支远端 clone；网络不可用时应报告，并传输包含
`.git` 的完整副本或 Git bundle。尚未提交的本地变更必须另外保存并应用 patch，
仅 clone 当前 HEAD 不会包含这些变更。

```bash
# 在两个节点的容器内执行；填写新的目录和待验证提交。
set -euo pipefail
export RDMA_WORKDIR=/data00/mega_moe_release_check_YYYYMMDD_run1
export RDMA_SOURCE_REV=REPLACE_WITH_TESTED_COMMIT
test ! -e "$RDMA_WORKDIR"
mkdir -p "$RDMA_WORKDIR"
git clone --recursive --branch experiment/mega-moe-combine-expert-ready \
  https://github.com/qiushixiaoyu/DeepGEMM.git "$RDMA_WORKDIR/DeepGEMM"
export RDMA_REPO="$RDMA_WORKDIR/DeepGEMM"
cd "$RDMA_REPO"
git checkout --detach "$RDMA_SOURCE_REV"
git submodule update --init --recursive
# 若验证未提交变更：先 git apply --check /path/to/source.patch，再 git apply。
git diff --check
git rev-parse HEAD | tee "$RDMA_WORKDIR/source_commit.txt"
git diff --binary > "$RDMA_WORKDIR/source.patch"
git submodule status | tee "$RDMA_WORKDIR/submodules.txt"
bash build_sgl_deep_gemm.sh 2>&1 | tee "$RDMA_WORKDIR/build.log"
# 选取刚构建的精确文件，不能用可能匹配多个版本的通配符安装。
export RDMA_WHEEL=/absolute/path/to/new/sgl_deep_gemm-version-tags.whl
python3 -m pip install --force-reinstall --no-deps "$RDMA_WHEEL" \
  2>&1 | tee "$RDMA_WORKDIR/install.log"
sha256sum "$RDMA_WHEEL" | tee "$RDMA_WORKDIR/wheel.sha256"
```

可以只在一个节点构建，再把同一 wheel 传到另一节点强制安装。两节点的
source patch、submodule revisions 和安装 wheel hash 必须相同。检查导入路径
时从源码目录外运行，避免导入到仓库的另一套 Python 包。

### 2. 设置运行环境与模型

两边执行相同设置，仅 `RANK` 不同。本测试脚本自行 spawn 8 个本地进程：
**`WORLD_SIZE=2` 表示节点数，`RANK=0/1` 表示节点编号**，最终 EP size 是 16。
不要再套一层 `torchrun --nproc_per_node=8`。

```bash
export MASTER_ADDR=192.168.24.5 MASTER_PORT=29651 WORLD_SIZE=2
export RANK=0  # COMM5 为 0；COMM6 必须改为 1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=8 NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
export NVSHMEM_IB_GID_INDEX=3 NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_PE_MAPPING=mlx5_1:1:1,mlx5_2:1:1,mlx5_3:1:1,mlx5_4:1:1,mlx5_5:1:1,mlx5_6:1:1,mlx5_7:1:1,mlx5_8:1:1
export NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_DISABLE_P2P=0
export NVSHMEM_IB_ENABLE_RELAXED_ORDERING=0 NVSHMEM_QP_DEPTH=4096
export NVSHMEM_MAX_TEAMS=32 NVSHMEM_DISABLE_NVLS=1 NVSHMEM_DISABLE_MNNVL=1
export NVSHMEM_CUMEM_GRANULARITY=536870912 NVSHMEM_DEBUG=WARN
export DG_NVSHMEM_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$DG_NVSHMEM_HOME/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export MEGA_MOE_FORCE_NVSHMEM_SYMM_MEM=1 TORCH_SYMMMEM=NVSHMEM
export DG_JIT_USE_NVRTC=0 DG_TEST_USE_SOURCE_TREE=0
export DG_MEGA_MOE_PHASE_PROFILE=0 DG_MEGA_MOE_PHASE_PROFILE_SILENT=0
export DG_MEGA_MOE_DEVICE_DIAGNOSTICS=0 DG_MEGA_MOE_DEBUG_NVSHMEM_STATE=0
export DG_COMM_KERNEL_DEBUG=0 DG_USE_NVIDIA_TOOLS=0
export FP8_TRANSFER_STREAM_ALIGNMENT=1 PYTHONHASHSEED=0
export DG_FP8_REFERENCE_CHUNK=16 DSV4_FP4_REFERENCE_CHUNK=16
export PYTHONPATH="$RDMA_REPO/sgl_deep_gemm/tests"
export RDMA_LOG_DIR
RDMA_LOG_DIR=$(mktemp -d "$RDMA_WORKDIR/run.XXXXXX")
export DG_JIT_CACHE_DIR="$RDMA_LOG_DIR/jit"
cd "$RDMA_WORKDIR"
python3 -c 'import torch, deep_gemm; print(torch.__version__, torch.version.cuda, deep_gemm.__file__)'

MODEL=GLM5.2; HIDDEN=6144; INTERMEDIATE=2048; EXPERTS=256; TOPK=8
# 其他模型：按上表一起替换这五项。
export NVSHMEM_IBGDA_NUM_RC_PER_PE=$((EXPERTS / 16 + 1))
export DG_SWEEP_MAX_TOKENS=256
BATCHES=(1 2 4 8 16 32 64 128 256)
```

上面的 IP、`eth0`、GID、HCA 和 Python 安装路径来自 COMM5/COMM6 历史环境，
不是可移植默认值。先在两个节点核对网络拓扑和 NIC 状态，再使用。运行各精度
前设置对应的推荐配置；FP8 不要求 FP4 的 LTO 工具存在。

### 3. 精度和性能命令

在两个节点定义以下函数，每次同时启动同名测试；上一项在**两节点均成功**后
再启动下一项。所有测试串行占用同一组 GPU，不同时跑 FP4、FP8 或 DeepEP。

```bash
set -euo pipefail
run_sweep() {
  local tag=$1
  shift
  local command=(python3 "$RDMA_REPO/sgl_deep_gemm/tests/test_mega_moe_perf_sweep.py"
    --model-name "$MODEL" --hidden "$HIDDEN" --intermediate-hidden "$INTERMEDIATE"
    --num-experts "$EXPERTS" --num-topk "$TOPK" --num-processes 8
    --batches "${BATCHES[@]}" --num-bench-tests 50 --num-warmup 5 --num-repeat 50
    --mega-timing kineto --deep-ep-timing cuda-events --deep-ep-expected-m batch
    --l2-flush-gb 0 --activation-clamp 10 --fast-math 1 --diff-tol 0.005
    --kineto-trace-dir "$RDMA_LOG_DIR/traces_$tag" "$@")
  printf '%q ' "${command[@]}" > "$RDMA_LOG_DIR/$tag.command.txt"
  printf '\n' >> "$RDMA_LOG_DIR/$tag.command.txt"
  (env | sort | sed -n '/^DG_/p;/^NVSHMEM_/p;/^NCCL_/p;/^GLOO_/p;/^MASTER_/p;/^WORLD_SIZE=/p;/^RANK=/p;/^CUDA_VISIBLE_DEVICES=/p;/^TORCH_SYMMMEM=/p;/^MEGA_MOE_/p;/^FP8_TRANSFER_/p;/^PYTHONHASHSEED=/p;/^PYTHONPATH=/p;/^LD_LIBRARY_PATH=/p;/^DSV4_/p') \
    > "$RDMA_LOG_DIR/$tag.env.txt"
  "${command[@]}" 2>&1 | tee "$RDMA_LOG_DIR/${tag}_node${RANK}.log"
}

# FP8：全输出 reference 和 12 次复用检查；每次新建 reference 目录。
run_sweep fp8_accuracy --mode low_latency --path fused --mega-num-sms 78 \
  --accuracy-only --check-fp8-reference --align-accuracy --accuracy-repeats 12 \
  --equivalence-tol 0.0000001 --reference-cache-dir "$RDMA_LOG_DIR/ref_fp8" \
  --write-reference-cache
# 两节点精度成功后执行。
run_sweep fp8_perf --mode low_latency --path fused --mega-num-sms 78

# FP4：在线 decode；小 batch 每个输出行均与 FP4 reference 比较。
run_sweep fp4_accuracy --mode low_latency --path fused --mega-num-sms 78 \
  --fp4-runtime --accuracy-only --check-fp4-reference --reference-max-local-tokens 256
# 两节点精度成功后执行。
run_sweep fp4_perf --mode low_latency --path fused --mega-num-sms 78 --fp4-runtime
```

FP8 的 12 次复用检查不能理解成 FP4 也执行了 12 次完整 reference 检查。
大 batch 下上述 FP4 命令只抽样最多 256 行；若需要全部行，增大
`--reference-max-local-tokens` 并预留 reference 的显存和时间。
合成测试可能受参考计算峰值内存限制，OOM 不等于 fused kernel 本身不支持该形状。

这里统一使用 sweep 的 `--mega-timing kineto`。若改用独立
`test_mega_moe_fp4_hopper.py`，则应使用它自己的
`--fp4-runtime-timing kineto`；不要把两个脚本的参数混用。

### 4. DeepEP 对比和大 batch

DeepEP 是可选基线，需要安装兼容版本；normal 路径还需要测试目录中的
`deepep_scatter_gather.py` 及其依赖。normal 的 DeepEP 与 MegaMoE 必须在
独立进程分别运行，不能使用 `--mode normal --path both`。

```bash
# DeepEP low-latency 的独立输出检查（此入口只检查 finite，不是 reference 验证）。
run_sweep deepep_ll_check --mode low_latency --path deep_ep --accuracy-only \
  --dump-output-dir "$RDMA_LOG_DIR/deepep_ll_outputs"
# 还需通过该 DeepEP 版本自身的精度测试／输出对照后，才采集基线性能。
run_sweep deepep_ll_perf --mode low_latency --path deep_ep
```

KimiK3 不测 low-latency 基线。切换到 normal 的大容量 sweep 时，两边设置
`DG_SWEEP_MAX_TOKENS=8192`，在 `BATCHES` 中加入
`512 1024 2048 4096 8192`，将命令的 mode 改为 `normal`，并更换日志 tag、
reference 和 trace 目录。对三条路径分别重新完成精度门禁后测性能。DeepEP normal
也先单独导出并核对输出；仅有 finite 检查不能宣称精度通过。
有意测试回切时可用反向 batch 顺序，但不要把重复写 reference cache 的报错
当作通信故障，应使用新的 reference 目录或显式复用已有 cache。

## 结果审计与归档

- 主表必须标注 **MegaMoE kernel-only（Kineto）对比 DeepEP end-to-end**。
  MegaMoE 只统计目标 `sm90_fp8_fp4_mega_moe_impl` 或 `sm90_fp8_mega_moe_impl`；
  不计输入准备／复制、wrapper、host launch、辅助 kernel 和 kernel 外同步等待。
- 检查 `mega_timing_method=kineto_kernel`，FP4 还应有
  `fp4_timing_method=kineto_kernel`；DeepEP 使用 `cuda_events_end_to_end`。
  DeepEP 基线计算使用 FP8 权重，不是 FP4 pipeline。
- 此模板使用 warm steady-state（`--l2-flush-gb 0`）和每 rank 50 个目标 kernel
  样本；保留每个 trace 和 `.samples.json`，不要只保存四舍五入后的汇总行。
  对照时固定容量、SM 数、路由、输入和配置，跑至少两个相反顺序的轮次。
- 保存两个节点的提交、patch、wheel/依赖 hash、启动环境、命令、精度日志和
  原始性能数据。换编译／链接依赖时使用新缓存；跨节点检查实际导入的安装位置。
- 在容器宿主机用 `docker cp mega_moe_rdma:/absolute/run/path /absolute/host/path`
  导出，再从 Mac 拉回两个节点的记录，分别归档。不要只保留服务器上的日志。
- 本地源码结构检查不是 CUDA 编译、GPU 数值精度、RDMA 正确性或性能验证。

## 本地维护检查（无 GPU）

```bash
python3 sgl_deep_gemm/tests/test_sm90_rdma_cleanup.py
python3 sgl_deep_gemm/tests/test_sm90_rdma_only.py
python3 sgl_deep_gemm/tests/test_sm90_packed_manifest.py
python3 sgl_deep_gemm/tests/test_sm90_combine_ring_alignment.py
git diff --check
```
