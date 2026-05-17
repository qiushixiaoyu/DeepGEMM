# SM90 MegaMoE 优化实验记录

除特别说明外，性能 sweep 使用如下配置：

- 脚本：`tests/test_mega_moe_hopper.py`
- GPU/rank 数：`8`
- 模型配置：DeepSeek-V4-Flash 风格
- `hidden=4096`
- `intermediate_hidden=2048`
- `num_experts=256`
- `num_topk=6`
- 只跑 fused 路径，baseline 关闭
- `num_bench_tests=5`
- 用户要求覆盖的 batch 列表：`1 2 4 8 16 32 64 512 1024 4096 9192`

## 2026-05-15：active-SM heuristic v1

### 假设

NCU 显示小 batch 下 eligible warps 很低，同时固定开销较高。因此尝试在 `block_m=64` 的小 batch decode 场景中减少参与 persistent kernel 的 CTA/SM 数量，同时在更大配置下保留 full-SM 参与。

### 改动

文件：`csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`

新增 `get_active_sms_for_sm90_mega_moe()`，并将 launch 参数接到 `LaunchArgs(active_sms, ...)`。

实测之后，该 heuristic **默认关闭**，只通过以下环境变量作为实验开关使用：

- `DG_SM90_MEGA_MOE_ENABLE_ACTIVE_SMS_HEURISTIC=1`
- `DG_SM90_MEGA_MOE_ACTIVE_SMS=<N>` 用于强制指定 active SM 数

### 精度验证

命令：

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8731 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

### 性能

原始日志：

- 关闭 active-SM/full-SM baseline：`benchmark_outputs/sm90_active_sms_v1_perf_disabled.log`
- 开启 active-SM 实验：`benchmark_outputs/sm90_active_sms_v1_perf_enabled.log`

8 个 rank 的 fused time 中位数：

| batch | full-SM 中位数 us | active-SM v1 中位数 us | active/full | 结论 |
|---:|---:|---:|---:|---|
| 1 | 190.0 | 748.0 | 3.94x | 拒绝 |
| 2 | 303.5 | 1146.5 | 3.78x | 拒绝 |
| 4 | 423.5 | 1202.5 | 2.84x | 拒绝 |
| 8 | 511.5 | 1269.0 | 2.48x | 拒绝 |
| 16 | 617.5 | 916.0 | 1.48x | 拒绝 |
| 32 | 583.5 | 607.5 | 1.04x | 拒绝 |
| 64 | 600.0 | 未运行 | n/a | 看到明显劣化后停止 |
| 512 | 1472.5 | 未运行 | n/a | 看到明显劣化后停止 |
| 1024 | 2815.0 | 未运行 | n/a | 看到明显劣化后停止 |
| 4096 | 8646.5 | 未运行 | n/a | 看到明显劣化后停止 |
| 9192 | 18202.0 | 未运行 | n/a | 看到明显劣化后停止 |

### 结论

不接受为默认优化。profiling 假设不完整：虽然小 batch 确实有较高的固定 dispatch/sync 开销，但当前 fused kernel 仍然需要 full-SM 并行度来维持 L1/L2 block stream 的推进。减少 persistent CTA 带来的损失大于节省的开销。

该实现仅保留为 env-gated 实验 hook。下一步优化应该聚焦于降低 all-expert dispatch/count/scheduler 的固定开销，而不是减少可执行 GEMM work 的 CTA 数量。

## 2026-05-15：block_m=64 wave heuristic v2

### 假设

原始 SM90 `block_m=64` heuristic 会强制所有本地 expert 放在同一个 wave 中。更宽泛的 generic-wave 实验显示，额外的 wave 边界并不总是有收益，但在中等稀疏的 decode 区间，较窄的 expert wave 可能带来收益。

对于 DeepSeek-V4-Flash 风格配置（`hidden=4096`、`intermediate_hidden=2048`、`num_experts=256`、`topk=6`、EP8），只有 batch 8 和 16 会真正改变行为：

- batch 1/2/4：每 expert 的期望 token 数 < 1，generic heuristic 也会返回 all-expert-per-wave
- batch 8/16：每 expert 的期望 token 数为 1.5/3.0，generic heuristic 会返回更窄的 wave
- batch 32/64：每 expert 的期望 token 数为 6.0/12.0，保持 single-wave，避免额外 wave 边界开销
- batch >= 512：`block_m > 64`，在这次改动前就已经使用 generic heuristic

### 改动

文件：`csrc/jit_kernels/heuristics/mega_moe.hpp`

对于 SM90 `block_m=64`，仅在以下条件下默认使用 generic wave scheduling：

```text
1.0 <= expected_tokens_per_expert <= 4.0
```

更宽泛的实验模式仍然保留在环境变量后面：

```text
DG_SM90_MEGA_MOE_BLOCK64_GENERIC_WAVES=1
```

### 精度验证

命令：

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8751 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

### 性能

原始日志：

- 更宽泛的 generic-wave 探索：`benchmark_outputs/sm90_block64_generic_waves_perf_enabled.log`
- 接受的 v2 完整 sweep：`benchmark_outputs/sm90_block64_wave_v2_perf.log`
- v2 小 batch 复跑：`benchmark_outputs/sm90_block64_wave_v2_small_rerun.log`

完整 sweep 中 8 个 rank 的 fused time 中位数：

| batch | recv token 中位数 | active expert 中位数 | full-SM us | generic all-block64 us | wave v2 us | v2/full | v2 变化 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.5 | 5.0 | 190.0 | 193.0 | 197.5 | 1.039x | +3.9% |
| 2 | 11.5 | 9.5 | 303.5 | 274.0 | 363.5 | 1.198x | +19.8% |
| 4 | 22.0 | 15.5 | 423.5 | 423.0 | 410.5 | 0.969x | -3.1% |
| 8 | 46.0 | 25.0 | 511.5 | 509.0 | 506.5 | 0.990x | -1.0% |
| 16 | 94.0 | 31.0 | 617.5 | 572.5 | 573.5 | 0.929x | -7.1% |
| 32 | 190.0 | 32.0 | 583.5 | 585.5 | 597.0 | 1.023x | +2.3% |
| 64 | 380.5 | 32.0 | 600.0 | 605.0 | 604.0 | 1.007x | +0.7% |
| 512 | 3061.0 | 32.0 | 1472.5 | 1446.0 | 1451.5 | 0.986x | -1.4% |
| 1024 | 6151.0 | 32.0 | 2815.0 | 2811.0 | 2829.5 | 1.005x | +0.5% |
| 4096 | 24545.5 | 32.0 | 8646.5 | 8548.5 | 8603.0 | 0.995x | -0.5% |
| 9192 | 55194.0 | 32.0 | 18202.0 | 18232.5 | 18229.5 | 1.002x | +0.2% |

使用 `num_bench_tests=7`、`num_warmup=3`、`num_repeat=7` 的小 batch 复跑结果：

| batch | full-SM us | v2 完整 sweep us | v2 复跑 us | 复跑/full | 复跑变化 |
|---:|---:|---:|---:|---:|---:|
| 1 | 190.0 | 197.5 | 195.0 | 1.026x | +2.6% |
| 2 | 303.5 | 363.5 | 281.5 | 0.928x | -7.2% |
| 4 | 423.5 | 410.5 | 416.0 | 0.982x | -1.8% |
| 8 | 511.5 | 506.5 | 508.5 | 0.994x | -0.6% |
| 16 | 617.5 | 573.5 | 578.0 | 0.936x | -6.4% |
| 32 | 583.5 | 597.0 | 586.0 | 1.004x | +0.4% |
| 64 | 600.0 | 604.0 | 612.5 | 1.021x | +2.1% |

### 结论

目前接受为保守默认策略。本次 sweep 中，DeepSeek-V4-Flash 配置下真正改变行为的 batch 只有 8 和 16；batch 16 稳定提升约 6-7%，batch 8 基本持平到小幅变快。batch 1/2/4 和 32/64 要么 wave 数与旧 single-wave 路径相同，要么被刻意保留在 single-wave 路径上，因此这些 batch 的小幅波动应视为 run-to-run noise，除非用同一份 build 做 A/B toggle 复现。

后续如需进一步确认，可以增加同一份 build 下的 `DG_SM90_MEGA_MOE_BLOCK64_FORCE_SINGLE_WAVE` override，只对 `block_m=64` batch 做成对 A/B sweep，从而去掉跨 build/跨 run 的噪声。

## 2026-05-15：zero-token cleanup skip v1

### 假设

小 batch 只会激活一部分本地 expert。cleanup 阶段中，SM90 仍然会为每个本地 expert 清理 per-rank recv-count slots 和 arrival state。尝试对 `expert_recv_count_sum` 低 32 位为 0 的 expert 跳过 per-rank recv-count 和 arrival cleanup，同时仍然清理 sum counter，避免下一轮 scheduler wait 因高 32 位 stale value 而提前通过。

### 测试改动

文件：`deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`

在 dispatch cleanup 内：

- 始终清理 `workspace.get_expert_recv_count_sum_ptr(i)`
- 如果 `num_recv_tokens == 0`，跳过：
  - 清理 `workspace.get_expert_recv_count_ptr(j, i)`
  - 清理 `workspace.get_l1_arrival_count_ptr(...)`
  - 清理 `workspace.get_l2_arrival_mask_ptr(...)`

测试时同步 bump 了 `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp` 中的 JIT cache marker。

### 精度验证

命令：

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8761 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

### 性能

原始日志：

- `benchmark_outputs/sm90_zero_token_cleanup_v1_perf.log`

8 个 rank 的 fused time 中位数，对比已接受的 wave v2 完整 sweep：

| batch | recv token 中位数 | active expert 中位数 | wave v2 us | zero cleanup us | zero/v2 | 变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.5 | 5.0 | 197.5 | 204.0 | 1.033x | +3.3% |
| 2 | 11.5 | 9.5 | 363.5 | 272.5 | 0.750x | -25.0% |
| 4 | 22.0 | 15.5 | 410.5 | 412.0 | 1.004x | +0.4% |
| 8 | 46.0 | 25.0 | 506.5 | 655.0 | 1.293x | +29.3% |
| 16 | 94.0 | 31.0 | 573.5 | 574.5 | 1.002x | +0.2% |
| 32 | 190.0 | 32.0 | 597.0 | 587.0 | 0.983x | -1.7% |
| 64 | 380.5 | 32.0 | 604.0 | 601.5 | 0.996x | -0.4% |
| 512 | 3061.0 | 32.0 | 1451.5 | 1540.0 | 1.061x | +6.1% |
| 1024 | 6151.0 | 32.0 | 2829.5 | 2805.0 | 0.991x | -0.9% |
| 4096 | 24545.5 | 32.0 | 8603.0 | 8622.5 | 1.002x | +0.2% |
| 9192 | 55194.0 | 32.0 | 18229.5 | 18153.0 | 0.996x | -0.4% |

### 结论

拒绝并已回退。虽然精度验证通过，但性能表现不稳定，并且在 batch 8 和 batch 512 出现明确回退。可能原因是这些 cleanup stores 并不是主要固定开销；移除它们反而改变了 post-combine cleanup 路径附近的 ordering/cache 行为，无法稳定缩短关键路径。

源码改动已在测量后回退，原始日志保留用于后续对比。

## 2026-05-16：large-batch wave granularity factor=4 v1

### 假设

大 batch 下 `block_m > 64` 路径使用 generic wave scheduling。原始 heuristic 使用 `imbalance_factor=2`，会为每个 wave 准备大约 `2 * num_sms` 个 L1 block。尝试把大 batch 的默认 `imbalance_factor` 提高到 4，减少 wave 边界数量，从而降低 `set_expert_idx` rewind、workspace re-read、barrier reset 等跨 wave 固定开销。

对 DeepSeek-V4-Flash 风格配置，理论影响主要在大 batch：

- batch 512：`num_experts_per_wave` 约从 16 增大到 32
- batch 4096：`num_experts_per_wave` 约从 2 增大到 4
- batch 9192：`num_experts_per_wave` 约从 1 增大到 2
- batch 1/16/64：仍在 `block_m=64` 路径，默认行为理论上不应改变

### 测试改动

文件：`csrc/jit_kernels/heuristics/mega_moe.hpp`

为 `get_num_experts_per_wave_for_mega_moe()` 增加可配置的 `imbalance_factor`，并在 SM90 `block_m > 64` 路径中临时使用：

```text
DG_SM90_MEGA_MOE_LARGE_WAVE_IMBALANCE_FACTOR=4
```

实验结束后，该环境变量入口保留，但默认值已回退为 2，因此当前默认路径与已接受的 wave v2 策略一致。

### 精度验证

实验改动后的验证命令：

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8781 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

回退默认值并重建后的验证命令：

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8782 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

### 性能

原始日志：

- `benchmark_outputs/sm90_large_wave_factor4_v1_perf.log`

8 个 rank 的 fused time 中位数，对比已接受的 wave v2 完整 sweep：

| batch | recv token 中位数 | active expert 中位数 | wave v2 us | factor=4 us | factor4/v2 | 变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.5 | 5.0 | 197.5 | 195.5 | 0.990x | -1.0% |
| 16 | 94.0 | 31.0 | 573.5 | 682.5 | 1.190x | +19.0% |
| 64 | 380.5 | 32.0 | 604.0 | 618.0 | 1.023x | +2.3% |
| 512 | 3061.0 | 32.0 | 1451.5 | 1452.5 | 1.001x | +0.1% |
| 4096 | 24545.5 | 32.0 | 8603.0 | 9085.0 | 1.056x | +5.6% |
| 9192 | 55194.0 | 32.0 | 18229.5 | 18311.5 | 1.004x | +0.4% |

### 结论

拒绝作为默认优化。大 batch 没有收益，batch 4096 明确回退约 5.6%，batch 512/9192 基本持平但略慢。batch 16/64 理论上不应受该改动影响，观察到的波动更像 run-to-run noise 或系统干扰，但这不影响结论：减少大 batch wave 数量并没有降低关键路径时间，反而可能降低了跨 expert 的调度粒度和负载均衡能力。

源码默认值已回退为 `imbalance_factor=2`，仅保留 `DG_SM90_MEGA_MOE_LARGE_WAVE_IMBALANCE_FACTOR` 作为后续 profiling 的实验开关。

## 2026-05-16：wave v2 默认路径对 legacy baseline 的大 batch 对照

### 目的

用户明确希望大 batch 的 fused SM90 MegaMoE 性能也要优于 `test_mega_moe_hopper.py` 中的 legacy baseline。因此在回退 `factor=4` 默认策略之后，额外启用 `--run-baseline` 做一次当前默认路径和 legacy baseline 的直接对照。

### 命令

```bash
BATCHES="1 16 64 512 4096 9192"
COMMON_ARGS="--num-processes 8 --hidden 4096 --intermediate-hidden 2048 --num-experts 256 --num-topk 6 --num-bench-tests 5 --num-warmup 2 --num-repeat 5 --l2-flush-gb 0 --run-baseline"
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH

for b in $BATCHES; do
  MASTER_PORT=$((9600 + b % 1000)) \
  python tests/test_mega_moe_hopper.py $COMMON_ARGS \
    --num-max-tokens-per-rank "$b" --num-tokens "$b"
done
```

### 性能

原始日志：

- `benchmark_outputs/sm90_wave_v2_vs_legacy_baseline_batches_1_9192.log`

8 个 rank 的 time 中位数：

| batch | recv token 中位数 | active expert 中位数 | fused us | baseline us | baseline/fused | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 5.5 | 5.0 | 184.0 | 342.5 | 1.861x | fused 更快 |
| 16 | 94.0 | 31.0 | 568.0 | 638.5 | 1.124x | fused 更快 |
| 64 | 380.5 | 32.0 | 591.0 | 657.0 | 1.112x | fused 更快 |
| 512 | 3061.0 | 32.0 | 1433.0 | 1226.0 | 0.856x | baseline 更快 |
| 4096 | 24545.5 | 32.0 | 8572.0 | 6876.0 | 0.802x | baseline 更快 |
| 9192 | 55194.0 | 32.0 | 18130.0 | 15001.0 | 0.827x | baseline 更快 |

### 结论

当前默认 fused 路径在 batch 1/16/64 仍然快于 legacy baseline，但从 batch 512 开始落后，batch 4096/9192 的差距约为 17-20%。后续优化优先级需要调整：继续优化小 batch 固定开销的收益有限，下一轮应优先围绕大 batch 的 fused GEMM/epilogue/combine 吞吐缺口做实验。

## 2026-05-16：large-batch wave granularity factor=1 v1

### 假设

`factor=4` 变宽 wave 后没有收益，说明减少 wave 边界不是正确方向。反过来尝试把大 batch `block_m>64` 的 `imbalance_factor` 从默认 2 降到 1，让每个 wave 更窄，提升跨 expert 的负载均衡粒度，观察是否能改善 batch 512/4096/9192。

该实验只通过环境变量覆盖，不改变源码默认：

```text
DG_SM90_MEGA_MOE_LARGE_WAVE_IMBALANCE_FACTOR=1
```

### 精度验证

命令：

```bash
DG_SM90_MEGA_MOE_LARGE_WAVE_IMBALANCE_FACTOR=1 \
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8783 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

### 性能

原始日志：

- `benchmark_outputs/sm90_large_wave_factor1_v1_vs_baseline.log`

8 个 rank 的 time 中位数：

| batch | recv token 中位数 | active expert 中位数 | default fused us | factor=1 fused us | factor1/default | baseline us | baseline/factor1 | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5.5 | 5.0 | 184.0 | 184.0 | 1.000x | 329.0 | 1.788x | fused 更快 |
| 16 | 94.0 | 31.0 | 568.0 | 566.5 | 0.997x | 633.0 | 1.117x | fused 更快 |
| 64 | 380.5 | 32.0 | 591.0 | 591.0 | 1.000x | 663.0 | 1.122x | fused 更快 |
| 512 | 3061.0 | 32.0 | 1433.0 | 1460.0 | 1.019x | 1229.0 | 0.842x | baseline 更快 |
| 4096 | 24545.5 | 32.0 | 8572.0 | 8567.0 | 0.999x | 6890.5 | 0.804x | baseline 更快 |
| 9192 | 55194.0 | 32.0 | 18130.0 | 18112.0 | 0.999x | 15003.5 | 0.828x | baseline 更快 |

### 结论

拒绝作为默认优化。factor=1 在 batch 4096/9192 上只有噪声级变化，没有缩小相对 baseline 的核心差距；batch 512 还明确慢了约 1.9%。这说明单纯调整 large-wave 粒度无法解决大 batch 落后 legacy baseline 的问题。

下一步应转向更结构性的瓶颈：SM90 fused 路径的大 batch L2 GEMM/epilogue/combine 吞吐。尤其需要关注 fused 使用 per-64 L2 activation SF，而 legacy baseline 使用 per-128 L2 activation SF；此外 fused 的 combine reduction 在同一个 persistent kernel 尾部完成，可能不如 DeepEP 独立 combine 路径高效。

## 2026-05-16：SM90 默认 `block_m=64` v1

### 假设

旧 SM90 heuristic 在大 batch 下会切到 `block_m=128`，也就是一个 CTA 内使用 2 个 math/epilogue warpgroups。理论上这能提高每个 CTA 的 M 方向工作量，但在当前 fused SM90 实现里也带来两个代价：

- 共享内存占用更大，pipeline stage 更少；
- fused L2/epilogue/combine 路径更长，`block_m=128` 的 2-WG CTA 没有超过 legacy grouped-GEMM baseline。

尝试让 SM90 默认始终使用 `block_m=64`、1 个 math/epilogue warpgroup，以更细粒度的 CTA/tile 调度换取更高 pipeline 吞吐。

### 改动

文件：`csrc/jit_kernels/heuristics/mega_moe.hpp`

- SM90 默认 `block_m=64`、`num_epilogue_warpgroups=1`。
- 保留环境变量 `DG_SM90_MEGA_MOE_FORCE_BLOCK_M=64|128`，便于后续 A/B 回溯旧策略。

### 精度验证

实验环境变量覆盖路径：

```bash
DG_SM90_MEGA_MOE_FORCE_BLOCK_M=64 \
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8784 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

改成默认策略并重建后的无 env 验证：

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8786 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

### 性能

原始日志：

- env 覆盖实验：`benchmark_outputs/sm90_force_block_m64_v1_vs_baseline.log`
- 最终默认策略：`benchmark_outputs/sm90_block_m64_default_v1_vs_baseline.log`

最终默认策略中 8 个 rank 的 time 中位数：

| batch | recv token 中位数 | active expert 中位数 | 旧默认 fused us | `block_m=64` 默认 us | 新/旧 | baseline us | baseline/新默认 | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5.5 | 5.0 | 184.0 | 184.0 | 1.000x | 330.5 | 1.796x | fused 更快 |
| 16 | 94.0 | 31.0 | 568.0 | 566.0 | 0.996x | 631.5 | 1.116x | fused 更快 |
| 64 | 380.5 | 32.0 | 591.0 | 593.5 | 1.004x | 658.5 | 1.110x | fused 更快 |
| 512 | 3061.0 | 32.0 | 1433.0 | 1139.0 | 0.795x | 1224.0 | 1.075x | fused 更快 |
| 4096 | 24545.5 | 32.0 | 8572.0 | 6945.0 | 0.810x | 6892.0 | 0.992x | baseline 更快 |
| 9192 | 55194.0 | 32.0 | 18130.0 | 15151.5 | 0.836x | 15013.5 | 0.991x | baseline 更快 |

### 结论

接受为默认优化。`block_m=64` 对大 batch 的收益很明确：

- batch 512：fused 从 1433 us 降到 1139 us，反超 baseline，约 1.075x；
- batch 4096：fused 从 8572 us 降到 6945 us，接近 baseline，但仍慢约 0.8%；
- batch 9192：fused 从 18130 us 降到 15151.5 us，接近 baseline，但仍慢约 0.9%。

这个优化显著缩小了大 batch 差距，但尚未完全满足“所有大 batch 都比 baseline 快”的目标。后续应继续聚焦 4096/9192 上剩余约 1% 的差距，优先检查 fused L2 per-64 activation SF 和 kernel 内 combine reduction 的额外成本。

## 2026-05-16：`block_m=64` + generic block64 waves v1

### 假设

`block_m=64` 大幅改善大 batch 后，尝试进一步打开 `DG_SM90_MEGA_MOE_BLOCK64_GENERIC_WAVES=1`，让大 batch block64 路径也使用 generic wave scheduling，以更窄的 expert wave 改善跨 expert 负载均衡，争取补齐 4096/9192 相对 baseline 的最后约 1% 差距。

### 精度验证

命令：

```bash
DG_SM90_MEGA_MOE_FORCE_BLOCK_M=64 \
DG_SM90_MEGA_MOE_BLOCK64_GENERIC_WAVES=1 \
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8785 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

### 性能

原始日志：

- `benchmark_outputs/sm90_force_block_m64_generic_waves_v1_vs_baseline.log`

该 sweep 在 batch 1 和 batch 16 完成后，于 batch 64 启动阶段长时间无输出，疑似 hang。已终止本轮 benchmark 进程，未继续跑 512/4096/9192。

已完成批次：

| batch | fused us | baseline us | baseline/fused | 结论 |
|---:|---:|---:|---:|---|
| 1 | 183.0 | 351.5 | 1.921x | fused 更快 |
| 16 | 566.0 | 636.5 | 1.125x | fused 更快 |

### 结论

拒绝作为默认优化。虽然精度验证通过，但 batch 64 sweep 疑似 hang，稳定性不满足要求。当前默认继续保留 block64 single-wave 规则：除 `1.0 <= expected_tokens_per_expert <= 4.0` 的中等稀疏区间外，`block_m=64` 默认使用 single-wave。

## 2026-05-16：L2 per-64 双 accumulator 合并 WGMMA wait 实验

### 背景

SM90 fused L2 GEMM 的 activation SF 是 per-64 K。每个 `BLOCK_K=128` 因此被拆成两个 half：

- `K=0..63` 使用 `scale_a_*_lo`；
- `K=64..127` 使用 `scale_a_*_hi`。

原实现每个 half 各自执行一组 `warpgroup_arrive` / `wgmma` / `commit` / `wait`，然后分别乘 scale 累加到 `final_accum`。这个路径保持 per-64 量化语义，但比 SM100 native block-scaled UMMA 多了软件 scale 和一次额外 WGMMA 同步。

直接改成真正 per-128 L2 activation SF 需要跨 L1 N-block 做 amax，并且两个 64-col half 必须用同一个 scale 重新量化；当前 `cluster_size=1` 的 L1 epilogue 无法在不引入跨 CTA 同步/延迟量化的情况下正确做到。因此本轮先做更保守的原型：保留 per-64 量化语义，只把两个 half 的 WGMMA 放进同一个 commit/wait 组。

### 改动

文件：

- `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`
- `tests/test_mega_moe_hopper.py`

实现：

- 新增环境变量 `DG_SM90_MEGA_MOE_L2_DUAL_ACCUM=1`。
- env 打开时，L2 path 使用 `accum` 和 `accum_hi` 两个临时 accumulator：
  - first half 写入 `accum`；
  - second half 写入 `accum_hi`；
  - 两个 half 共用一次 `warpgroup_arrive` / `commit` / `wait`；
  - wait 后按原 per-64 scale 语义累加到 `final_accum`。
- env 打开时将 `kNumEpilogueRegisters` 从 208 提高到 240，并限制 `BLOCK_M == 64`。
- 修复 `tests/test_mega_moe_hopper.py` 的本地导入路径：
  - 将 `/workspace/DeepGEMM` 插入 `sys.path`，避免导入环境中的旧 `deep_gemm`；
  - 将 `/workspace/DeepEP` 插入 `sys.path`，确保 baseline 使用当前工作区里的 `deep_ep.ElasticBuffer`。

### 精度验证

命令：

```bash
DG_SM90_MEGA_MOE_L2_DUAL_ACCUM=1 \
python tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 --fail-fast
```

结果：

```text
SM90 MegaMoE test plan: 1 scenarios across layers [1] on 8 ranks
  [L1.smoke                        ] diff=0.0006 (tol=0.07) OK
PASSED all 1 scenarios
```

扩展分层精度：

```bash
DG_SM90_MEGA_MOE_L2_DUAL_ACCUM=1 \
python tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

原始日志：

- `benchmark_outputs/sm90_l2_dual_accum_correctness_l1.log`
- `benchmark_outputs/sm90_l2_dual_accum_correctness_layers_1_4.log`

### 性能

主对比口径：`num-max-tokens-per-rank == batch`，与既有 `sm90_block_m64_default_v1_vs_baseline.log` 对齐。

命令模板：

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:${LD_LIBRARY_PATH}
for b in 1 16 64 512 4096 9192; do
  DG_SM90_MEGA_MOE_L2_DUAL_ACCUM=1 python tests/test_mega_moe_hopper.py \
    --num-processes 8 \
    --num-max-tokens-per-rank ${b} \
    --num-tokens ${b} \
    --hidden 4096 \
    --intermediate-hidden 2048 \
    --num-experts 256 \
    --num-topk 6 \
    --run-baseline \
    --num-bench-tests 30 \
    --num-warmup 5 \
    --num-repeat 20 \
    --l2-flush-gb 8
done
```

原始日志：

- `benchmark_outputs/sm90_l2_dual_accum_v4flash_batches_1_9192_matched_capacity.log`

8 个 rank 的 time 中位数：

| batch | dual-accum fused us | baseline us | baseline/fused | 默认 block64 fused us | dual/default | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 184.0 | 250.0 | 1.359x | 184.0 | 1.000x | fused 更快 |
| 16 | 566.0 | 580.5 | 1.026x | 566.0 | 1.000x | fused 略快 |
| 64 | 590.0 | 600.0 | 1.017x | 593.5 | 0.994x | fused 略快 |
| 512 | 1140.0 | 1212.5 | 1.064x | 1139.0 | 1.001x | fused 更快 |
| 4096 | 6948.0 | 6894.0 | 0.992x | 6945.0 | 1.000x | baseline 略快 |
| 9192 | 15149.5 | 15000.0 | 0.990x | 15151.5 | 1.000x | baseline 略快 |

补充口径：固定 `num-max-tokens-per-rank=9192` 的 capacity sweep。

原始日志：

- `benchmark_outputs/sm90_l2_dual_accum_v4flash_batches_1_9192_final.log`

该口径下 batch 4096 因 baseline 也承担固定大 capacity 的开销，表现为 fused 约 1.14x 更快；但 batch 9192 仍约 0.99x，不能证明大 batch 已经稳定超过 baseline。因此主结论仍以 matched-capacity 表为准。

### 结论

拒绝作为默认优化，保留为 env-gated 实验路径。

精度上没有问题，但性能上没有解决大 batch 收益不足：

- batch 4096：dual-accum fused 约 6948 us，baseline 约 6894 us，仍慢约 0.8%；
- batch 9192：dual-accum fused 约 15149.5 us，baseline 约 15000 us，仍慢约 1.0%；
- 与默认 `block_m=64` 相比，dual-accum 基本持平，说明额外 accumulator/register pressure 抵消了少一次 WGMMA wait 的收益，或者 WGMMA wait 不是当前大 batch 的主瓶颈。

后续继续推进更结构性的方向：真正 per-128 L2 activation SF / `cluster_size=2` A multicast。这需要解决 L1 epilogue 的跨 CTA amax 与统一 scale 量化问题，才可能从根上减少 SM90 L2 per-64 scale 带来的额外成本。

## 2026-05-16：SFB/weight scale SMEM staging 实验

### 背景

SM90 当前默认路径中，block-(128,128) weight scale 由 math warpgroup 在 `full_barriers[stage_idx]->wait(phase)` 之后直接从 global memory 读取：

- L1：每个 `(expert, n_block, k_block)` 读取 `gate_sf` 和 `up_sf` 两个 scalar；
- L2：每个 `(expert, n_block, k_block)` 读取一个 `l2_sf` scalar。

这些 scalar 很小，但每个 math warpgroup 的 128 个线程都会执行相同地址的 `__ldg`。前序 profiling 显示大 batch 下 MIO / memory 相关开销仍然值得排查，因此做一个保守实验：让 B-loader producer warp 在加载 B tile 的同时，把 1-2 个 SFB scalar 写入 shared memory，math warpgroup 改为从 SMEM 读取。

### 改动

新增环境变量开关：

- `DG_SM90_MEGA_MOE_SFB_SMEM=1`

默认关闭，不影响默认实现。

代码改动：

- `csrc/jit_kernels/heuristics/mega_moe.hpp`
  - 开关开启时，每个 pipeline stage 额外预留 `align(2 * sizeof(float), 128)` 字节的 SFB SMEM；
- `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`
  - JIT 代码生成时根据环境变量注入 `#define DG_SM90_MEGA_MOE_SFB_SMEM 1`；
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`
  - B-loader warp 在 `arrive_and_expect_tx` 前写入 `smem_sfb[stage_idx]`；
  - math warpgroup 在 GEMM 前通过 `ptx::ld_shared` 读取 `gate_sf/up_sf/l2_sf`；
  - 默认路径仍保留原来的 direct global `__ldg`。

### 精度

命令：

```bash
DG_SM90_MEGA_MOE_SFB_SMEM=1 \
python tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

原始日志：

- `benchmark_outputs/sm90_sfb_smem_correctness_layers_1_4.log`

### 性能

命令模板：

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:${LD_LIBRARY_PATH}
for b in 1 16 64 512 4096 9192; do
  DG_SM90_MEGA_MOE_SFB_SMEM=1 python tests/test_mega_moe_hopper.py \
    --num-processes 8 \
    --num-max-tokens-per-rank ${b} \
    --num-tokens ${b} \
    --hidden 4096 \
    --intermediate-hidden 2048 \
    --num-experts 256 \
    --num-topk 6 \
    --run-baseline \
    --num-bench-tests 30 \
    --num-warmup 5 \
    --num-repeat 20 \
    --l2-flush-gb 8
done
```

原始日志：

- `benchmark_outputs/sm90_sfb_smem_v4flash_batches_1_9192_matched_capacity.log`

8 个 rank 的 time 中位数：

| batch | SFB-SMEM fused us | baseline us | baseline/fused | 结论 |
|---:|---:|---:|---:|---|
| 1 | 184.0 | 255.5 | 1.389x | fused 更快 |
| 16 | 566.0 | 574.0 | 1.014x | fused 略快 |
| 64 | 590.0 | 599.5 | 1.016x | fused 略快 |
| 512 | 1140.0 | 1206.5 | 1.058x | fused 更快 |
| 4096 | 6957.0 | 6896.0 | 0.991x | baseline 略快 |
| 9192 | 15137.0 | 15053.0 | 0.994x | baseline 略快 |

### 结论

拒绝作为默认优化，保留为 env-gated 实验路径。

该实验精度没有问题，但性能上没有解决大 batch 收益不足：

- batch 1/512 仍保持 fused 优势，说明 staging 没有明显破坏小/中 batch；
- batch 4096：fused 约 6957 us，baseline 约 6896 us，baseline 快约 0.9%；
- batch 9192：fused 约 15137 us，baseline 约 15053 us，baseline 快约 0.6%；
- 与默认 direct `__ldg` 路径相比，整体差异接近噪声级，说明 weight scale direct global load 不是当前大 batch 的主瓶颈。Hopper read-only cache 很可能已经把同地址 scalar load 合并得足够好，而把 SFB 放到 SMEM 还会轻微增加 pipeline stage 的 SMEM 占用。

后续继续测试 tile/scheduling 方向，优先验证 `BLOCK_M=128` 在大 batch 下是否能降低调度和 combine 开销；如果仍不能超过 baseline，再进入更重的 per-128 L2 activation SF / `cluster_size=2` 结构性改造。

## 2026-05-16：强制 `BLOCK_M=128` tile/scheduling 实验

### 背景

默认 SM90 路径已经改为 `BLOCK_M=64`，原因是它在 DeepSeek-V4-Flash 的 batch sweep 中整体更稳：小 batch 有更细的并行粒度，大 batch 也没有明显劣化。不过从大 batch 角度看，`BLOCK_M=128` 理论上可能减少 CTA 数量、barrier 次数和 combine 调度次数，因此需要单独验证它是否能解决 batch 4096/9192 下 fused 稍慢于 baseline 的问题。

本实验不改 kernel 逻辑，只使用已有环境变量强制 tile：

- `DG_SM90_MEGA_MOE_FORCE_BLOCK_M=128`

### 精度

命令：

```bash
DG_SM90_MEGA_MOE_FORCE_BLOCK_M=128 \
python tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

原始日志：

- `benchmark_outputs/sm90_block_m128_correctness_layers_1_4.log`

### 性能

命令模板：

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:${LD_LIBRARY_PATH}
for b in 1 16 64 512 4096 9192; do
  DG_SM90_MEGA_MOE_FORCE_BLOCK_M=128 python tests/test_mega_moe_hopper.py \
    --num-processes 8 \
    --num-max-tokens-per-rank ${b} \
    --num-tokens ${b} \
    --hidden 4096 \
    --intermediate-hidden 2048 \
    --num-experts 256 \
    --num-topk 6 \
    --run-baseline \
    --num-bench-tests 30 \
    --num-warmup 5 \
    --num-repeat 20 \
    --l2-flush-gb 8
done
```

原始日志：

- `benchmark_outputs/sm90_block_m128_v4flash_batches_1_9192_matched_capacity.log`

8 个 rank 的 time 中位数：

| batch | block_m128 fused us | baseline us | baseline/fused | 结论 |
|---:|---:|---:|---:|---|
| 1 | 419.0 | 247.0 | 0.589x | baseline 明显更快 |
| 16 | 1386.0 | 573.0 | 0.413x | baseline 明显更快 |
| 64 | 1414.0 | 593.5 | 0.420x | baseline 明显更快 |
| 512 | 1424.0 | 1212.5 | 0.852x | baseline 更快 |
| 4096 | 8544.0 | 6898.5 | 0.807x | baseline 明显更快 |
| 9192 | 18084.5 | 14992.5 | 0.829x | baseline 明显更快 |

### 结论

拒绝该方向作为默认或动态大 batch 切换策略。

`BLOCK_M=128` 虽然精度正确，但性能全面退化：

- 小 batch 退化最严重，batch 16/64 只有 baseline 的约 41-42%；
- 大 batch 也没有翻盘，batch 4096/9192 分别慢约 24%/21%；
- HBM 指标也明显下降，例如 batch 9192 fused 约 96 GB/s，而 baseline 约 115-116 GB/s。

这说明当前 SM90 fused kernel 的大 batch 问题不是 CTA 数太多造成的简单调度开销。相反，`BLOCK_M=128` 让每个 CTA 绑定两个 math warpgroups、降低 pipeline depth/调度灵活性，反而压低了吞吐。继续保留 `BLOCK_M=64` 默认策略。

后续应把优先级转向结构性差异：L2 per-64 activation SF、L1 epilogue quantization/store、combine/reduction，以及 SM100 FP4 路径中的 TMA SF/TMEM/cluster 机制与 SM90 FP8 路径的差距。

## 2026-05-16：Combine `kNumChunks=2` 实验

### 背景

大 batch 下 fused 与 baseline 的差距非常接近输出里估算的 `reduction(us)`：

- batch 4096：`reduction(us)` 约 36 us；
- batch 9192：`reduction(us)` 约 81 us。

combine/reduction 阶段每个 warp 负责一个 token，把 top-k 个 expert 输出按 hidden 维累加再写回。默认逻辑会在 SMEM/register 条件允许时使用 `kNumChunks=1`，这样每个 warp 一次处理完整 hidden，但 `reduced[]` 寄存器数组较大。实验思路是强制拆成 2 个 hidden chunk，降低 combine 阶段的寄存器压力，观察是否能改善大 batch。

新增环境变量开关：

- `DG_SM90_MEGA_MOE_FORCE_COMBINE_CHUNKS=2`

默认关闭；也支持强制 `1` 用于 A/B，但本轮只测试 `2`。

### 改动

- `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`
  - JIT 生成时读取 `DG_SM90_MEGA_MOE_FORCE_COMBINE_CHUNKS`；
  - 合法值限定为 `0/1/2`；
  - 非 0 时注入 `#define DG_SM90_MEGA_MOE_FORCE_COMBINE_CHUNKS N`。
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`
  - combine 阶段的 `kNumChunks` 支持被宏覆盖；
  - 默认 heuristic 不变。

### 精度

命令：

```bash
DG_SM90_MEGA_MOE_FORCE_COMBINE_CHUNKS=2 \
python tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

原始日志：

- `benchmark_outputs/sm90_combine_chunks2_correctness_layers_1_4.log`

### 性能

命令模板：

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:${LD_LIBRARY_PATH}
for b in 1 16 64 512 4096 9192; do
  DG_SM90_MEGA_MOE_FORCE_COMBINE_CHUNKS=2 python tests/test_mega_moe_hopper.py \
    --num-processes 8 \
    --num-max-tokens-per-rank ${b} \
    --num-tokens ${b} \
    --hidden 4096 \
    --intermediate-hidden 2048 \
    --num-experts 256 \
    --num-topk 6 \
    --run-baseline \
    --num-bench-tests 30 \
    --num-warmup 5 \
    --num-repeat 20 \
    --l2-flush-gb 8
done
```

原始日志：

- `benchmark_outputs/sm90_combine_chunks2_v4flash_batches_1_9192_matched_capacity.log`

8 个 rank 的 time 中位数：

| batch | chunks2 fused us | baseline us | baseline/fused | 结论 |
|---:|---:|---:|---:|---|
| 1 | 184.0 | 250.5 | 1.361x | fused 更快 |
| 16 | 567.0 | 575.0 | 1.014x | fused 略快 |
| 64 | 590.0 | 611.5 | 1.036x | fused 更快 |
| 512 | 1140.0 | 1205.5 | 1.057x | fused 更快 |
| 4096 | 6955.0 | 6891.5 | 0.991x | baseline 略快 |
| 9192 | 15132.0 | 14954.0 | 0.988x | baseline 略快 |

### 结论

拒绝作为默认优化，保留为 env-gated 实验路径。

`kNumChunks=2` 精度正确，但性能几乎与默认 fused 持平：

- batch 1/512 维持 fused 优势；
- batch 4096/9192 仍然 baseline 略快；
- 拆 chunk 没有明显降低大 batch 的 fused time，说明 combine 阶段不是由单个 chunk 的寄存器数组单点限制；TMA load/store 次数增加也可能抵消了寄存器压力下降。

当前三类轻量实验的结论趋同：小改动无法让 SM90 FP8 fused 在大 batch 稳定超过 SM90 grouped-GEMM baseline。后续真正可能改变大 batch 结果的方向应是结构性改造：

- L1 epilogue 生成真正 per-128 L2 activation SF，减少 L2 per-64 带来的额外 scale/TMA/WGMMA 分裂成本；
- 借鉴 SM100 的跨 warpgroup amax reduction / cluster 协作机制，解决 per-128 SF 需要两个 64-col L1 block 共同决定 scale 的问题；
- 进一步拆解 fused kernel 中 L2 GEMM 和 combine/reduction 的时间占比，确认 baseline 在 DeepEP combine 上是否有额外 overlap 优势。

## 2026-05-16：L2 per-64 SFA pair TMA 实验

### 背景

SM90 fused 当前保持 L2 activation SF per-64 语义。每个 L2 `BLOCK_K=128` 对应两个 64-col scale group，因此 producer warp 在 L2 阶段会对同一个 `(m_block, k_block)` 发两次 SFA TMA：

- 第一次加载 `k_block_idx * 2`，写到 `smem_sfa[stage] + 0`；
- 第二次加载 `k_block_idx * 2 + 1`，写到 `smem_sfa[stage] + BLOCK_M`。

本实验不改变数值语义，只把 L2 SFA descriptor 的 outer box 改成 2，让一次 TMA transaction 同时加载两个 per-64 scale column，SMEM 布局仍为 `[lo BLOCK_M][hi BLOCK_M]`。目标是降低 L2 阶段的 TMA transaction 数和 producer warp 指令开销。

新增环境变量开关：

- `DG_SM90_MEGA_MOE_L2_SFA_PAIR_TMA=1`

默认关闭。

### 改动

- `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`
  - JIT 生成时读取 `DG_SM90_MEGA_MOE_L2_SFA_PAIR_TMA`；
  - 开关开启时注入 `#define DG_SM90_MEGA_MOE_L2_SFA_PAIR_TMA 1`；
  - host 侧为 L2 activation SF 构造 `block_outer=2` 的 TMA descriptor。
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`
  - L2 producer warp 从两次 `tma::copy<BLOCK_M, 1>` 改为一次 `tma::copy<BLOCK_M, 2>`；
  - `arrive_and_expect_tx` 字节数仍为 `SMEM_A_SIZE_PER_STAGE + 2 * BLOCK_M * sizeof(float)`；
  - 默认路径不变。

### 精度

命令：

```bash
DG_SM90_MEGA_MOE_L2_SFA_PAIR_TMA=1 \
python tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

原始日志：

- `benchmark_outputs/sm90_l2_sfa_pair_tma_correctness_layers_1_4.log`

### 性能

命令模板：

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:${LD_LIBRARY_PATH}
for b in 1 16 64 512 4096 9192; do
  DG_SM90_MEGA_MOE_L2_SFA_PAIR_TMA=1 python tests/test_mega_moe_hopper.py \
    --num-processes 8 \
    --num-max-tokens-per-rank ${b} \
    --num-tokens ${b} \
    --hidden 4096 \
    --intermediate-hidden 2048 \
    --num-experts 256 \
    --num-topk 6 \
    --run-baseline \
    --num-bench-tests 30 \
    --num-warmup 5 \
    --num-repeat 20 \
    --l2-flush-gb 8
done
```

原始日志：

- `benchmark_outputs/sm90_l2_sfa_pair_tma_v4flash_batches_1_9192_matched_capacity.log`

8 个 rank 的 time 中位数：

| batch | pair-TMA fused us | baseline us | baseline/fused | 结论 |
|---:|---:|---:|---:|---|
| 1 | 183.5 | 250.5 | 1.365x | fused 更快 |
| 16 | 565.5 | 575.5 | 1.018x | fused 略快 |
| 64 | 591.0 | 595.5 | 1.008x | fused 略快 |
| 512 | 1142.0 | 1209.0 | 1.059x | fused 更快 |
| 4096 | 6956.0 | 6899.5 | 0.992x | baseline 略快 |
| 9192 | 15134.5 | 14975.5 | 0.989x | baseline 略快 |

### 结论

拒绝作为默认优化，保留为 env-gated 实验路径。

精度正确，但性能基本没有改善：

- batch 1/16 与默认 fused 持平；
- batch 512 仍保持 fused 优势，但没有比默认更好；
- batch 4096/9192 仍然 baseline 略快，pair TMA 无法解决大 batch 差距；
- 说明 L2 per-64 SFA 的两次 TMA transaction 不是当前主瓶颈，或者 transaction 数减少被 descriptor/2D tile 开销抵消。

下一步优先测试 L1 epilogue 输出写回路径：当前 SM90 先把 L1 FP8 tile 写入 SMEM，再用 TMA store 写到 `l2_token_buffer`，随后等待 store 完成并通知 L2。这个路径与 SM100 的 TMEM/双 store stage 机制差异较大，可能是 fused 大 batch 中比 grouped-GEMM baseline 多出来的固定开销之一。

## 2026-05-16：L1 epilogue direct global store 实验

### 背景

SM90 L1 epilogue 当前写回路径：

1. math warpgroup 将 SwiGLU 后的 FP8 输出先写到 `smem_cd_l1`；
2. 单个 elected 线程发起 `SM90_TMA_STORE_2D`，把整个 L1 output tile 写入 `l2_token_buffer`；
3. 等待 TMA store 完成；
4. 设置 `l2_arrival_mask`，通知 L2 阶段可以读取该 N block。

这个路径语义干净，但每个 L1 N block 都有一次 TMA store 和一次 `tma_store_wait`。本实验绕过 SMEM/TMA store，直接由 epilogue threads 把 FP8 pair 写到 `l2_token_buffer`，然后通过 `__threadfence()` + epilogue sync + release `red_or` 保证 L2 读取可见性。

新增环境变量开关：

- `DG_SM90_MEGA_MOE_L1_DIRECT_STORE=1`

默认关闭。

### 改动

- `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`
  - JIT 生成时读取 `DG_SM90_MEGA_MOE_L1_DIRECT_STORE`；
  - 开关开启时注入 `#define DG_SM90_MEGA_MOE_L1_DIRECT_STORE 1`。
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`
  - L1 epilogue quantize 后直接写 `l2_token_buffer`；
  - 跳过 `smem_cd_l1` 写入、`SM90_TMA_STORE_2D` 和 `tma_store_wait`；
  - 为保证跨 CTA 可见性，在通知 L2 前执行 `__threadfence()`，然后走原有 epilogue sync 和 release `red_or`。

### 精度

命令：

```bash
DG_SM90_MEGA_MOE_L1_DIRECT_STORE=1 \
python tests/test_mega_moe_sm90.py --num-processes 8 --layers 1 2 3 4 --fail-fast
```

结果：

```text
PASSED all 28 scenarios
```

原始日志：

- `benchmark_outputs/sm90_l1_direct_store_correctness_layers_1_4.log`

### 性能

命令模板：

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:${LD_LIBRARY_PATH}
for b in 1 16 64 512 4096 9192; do
  DG_SM90_MEGA_MOE_L1_DIRECT_STORE=1 python tests/test_mega_moe_hopper.py \
    --num-processes 8 \
    --num-max-tokens-per-rank ${b} \
    --num-tokens ${b} \
    --hidden 4096 \
    --intermediate-hidden 2048 \
    --num-experts 256 \
    --num-topk 6 \
    --run-baseline \
    --num-bench-tests 30 \
    --num-warmup 5 \
    --num-repeat 20 \
    --l2-flush-gb 8
done
```

原始日志：

- `benchmark_outputs/sm90_l1_direct_store_v4flash_batches_1_9192_matched_capacity.log`

8 个 rank 的 time 中位数：

| batch | direct-store fused us | baseline us | baseline/fused | 结论 |
|---:|---:|---:|---:|---|
| 1 | 184.0 | 255.5 | 1.389x | fused 更快 |
| 16 | 566.5 | 582.5 | 1.028x | fused 略快 |
| 64 | 590.0 | 587.5 | 0.996x | 基本持平 |
| 512 | 1141.0 | 1203.5 | 1.055x | fused 更快 |
| 4096 | 6957.0 | 6901.5 | 0.992x | baseline 略快 |
| 9192 | 15133.5 | 14958.0 | 0.988x | baseline 略快 |

### 结论

拒绝作为默认优化，保留为 env-gated 实验路径。

direct global store 精度正确，但性能没有优于默认 TMA store 路径：

- 小 batch 和默认 fused 基本持平；
- batch 4096/9192 仍然 baseline 略快；
- 说明 L1 output 的 TMA store/wait 不是单独主瓶颈，或者 direct global store 的非 TMA 写入 + `__threadfence()` 成本抵消了省掉的 TMA store；
- 这个结果也说明当前默认的 SMEM + TMA store 路径虽然复杂，但仍是更合理的跨 CTA producer/consumer 发布方式。

到目前为止，围绕 L2 scale TMA、L1 output store、combine chunk、SFB staging、block tile 的轻量实验都没有改变大 batch 结论。下一步需要进入更重的结构性实验：真正将 SM90 L2 activation SF 从 per-64 改到 per-128，或者做 profiler-guided 的阶段拆分计时，量化 L1/L2/combine 各自的真实时间占比后再定点改。
