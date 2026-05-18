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

## 2026-05-17：SM90 stage-level profiling / baseline 分段计时

### 目的

前序轻量实验已经基本排除：

- L2 per-64 SFA 的两次 TMA transaction；
- L1 output 的 TMA store/wait；
- combine chunk 数；
- SFB scalar global load；
- `BLOCK_M=128` / large-wave 粒度。

本轮改为直接拆阶段时间，确认 `512/4096/9192` 上 fused 与 legacy baseline 的差距主要来自哪里。

### 改动

新增 env-gated in-kernel `clock64` profile：

- `DG_SM90_MEGA_MOE_STAGE_PROFILE=1`
- `tests/test_mega_moe_hopper.py --stage-profile 1`

实现方式：

- `deep_gemm.fp8_mega_moe(..., stage_profile=profile_tensor)` 额外传入 CUDA int64 tensor；
- SM90 kernel 在 env 打开时写每个 CTA 的 dispatch、producer wait、L1 GEMM、L1 epilogue、L2 GEMM、L2 epilogue、combine barrier/reduce counters；
- benchmark 脚本额外打印 rank 级摘要；
- 同时给 legacy baseline 加一轮单次 CUDA event 分段计时：DeepEP dispatch、L1 grouped GEMM、Triton SwiGLU/quant、L2 grouped GEMM、DeepEP combine。

默认不传 profile tensor、不开 env 时，行为不变。

### 日志

- 主 profile：`benchmark_outputs/sm90_stage_profile_v4flash_batches_512_4096_9192.log`
- baseline 分段补充：`benchmark_outputs/sm90_stage_profile_with_baseline_stages_v4flash_512_4096_9192.log`

### 性能主口径

8 rank 中位数，matched capacity，`num_bench_tests=30`、baseline `num_repeat=20`：

| batch | fused us | baseline us | baseline/fused | 结论 |
|---:|---:|---:|---:|---|
| 512 | 1153 | 1215 | 1.054x | fused 更快 |
| 4096 | 6999 | 6881 | 0.983x | baseline 快约 118 us |
| 9192 | 15287 | 15006 | 0.982x | baseline 快约 281 us |

### fused stage profile 摘要

下表为 rank 中位数。`L1/L2` 是 max-CTA 子段估计，主要用于相对判断；绝对值会因 dispatch/GEMM overlap、SM clock 转换和 barrier 结构低于完整 kernel wall time。

| batch | fused L1 total us | L1 GEMM | L1 epi | fused L2 total us | L2 GEMM | L2 epi | combine reduce us | combine barrier us | agg math L1/L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 676.7 | 616.6 | 59.5 | 354.8 | 297.3 | 57.9 | 9.9 | 35.3 | 65.6% / 34.4% |
| 4096 | 4084.6 | 3716.6 | 368.7 | 2144.8 | 1793.6 | 351.4 | 74.2 | 79.9 | 65.3% / 34.7% |
| 9192 | 8943.3 | 8103.8 | 839.4 | 4693.6 | 3923.7 | 770.8 | 165.7 | 105.8 | 65.3% / 34.7% |

### baseline 分段摘要

单次 CUDA event 分段计时，rank 中位数，单位 ms：

| batch | dispatch | L1 GEMM | SwiGLU+quant | L2 GEMM | combine | stage sum |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.110 | 0.585 | 0.124 | 0.307 | 0.107 | 1.232 |
| 4096 | 0.400 | 3.290 | 0.822 | 1.687 | 0.757 | 6.956 |
| 9192 | 0.713 | 7.165 | 1.809 | 3.661 | 1.652 | 15.001 |

### 结论

大 batch 剩余差距主要对齐到 fused L2 GEMM 子段，而不是 L1 或 combine：

- L1 路径：fused `L1 GEMM + L1 epi` 与 baseline `L1 GEMM + SwiGLU/quant` 基本持平，甚至略快；
  - batch 4096：fused L1 约 4.085 ms，baseline L1+SwiGLU 约 4.112 ms；
  - batch 9192：fused L1 约 8.943 ms，baseline L1+SwiGLU 约 8.974 ms。
- L2 GEMM 子段：大 batch 下 fused 开始慢于 baseline L2 grouped GEMM，慢出来的量与整体差距量级高度一致；
  - batch 4096：fused L2 GEMM 约 1.794 ms，baseline L2 GEMM 约 1.687 ms，差约 107 us；整体 baseline 快约 118 us；
  - batch 9192：fused L2 GEMM 约 3.924 ms，baseline L2 GEMM 约 3.661 ms，差约 263 us；整体 baseline 快约 281 us。
- combine 不是当前主因：baseline combine 本身明显更长，但 fused 的 L2 epilogue 已经承担了 remote scatter，所以应看 `L2 epi + combine` 的组合；该组合没有解释 baseline 的领先。
- L2 epilogue 占 fused L2 比例约 16%，高于 L1 epilogue 的约 8%，但前序 direct-store / combine-chunk 实验说明单独改写回和 reduction 不能打开局面。

下一步应优先做真正结构性 L2 方向，而不是继续轻量开关：

1. 真正 per-128 L2 activation SF，减少 SM90 当前 per-64 L2 GEMM 的 split-WGMMA/software-scale 成本；
2. 或先做一个更细 L2-only micro profile/variant，把 L2 GEMM 中 `per-64 scale split` 与 `L2 epilogue scatter` 再拆开，验证 per-128 改造的理论收益上限。

## 2026-05-17：SM90 L2 activation SF per-128 v1（事后重标定）

### 目的

验证上一轮 profiling 指向的结构性方向：把 L2 activation SF 从 per-64 改为 per-128，让 L2 GEMM 避免当前每个 `BLOCK_K=128` 被拆成两个 per-64 scale 半块的 WGMMA/software-scale 路径。

### 实现

新增 env-gated 实验路径：

- `DG_SM90_MEGA_MOE_L2_ACT_SF_PER128=1`

实现方式是保守的 v1，不改 L1 GEMM tile：

- L1 epilogue 仍按当前每个 CTA 产出 64 列 SwiGLU/FP8，并写 raw per-64 scale；
- workspace 新增 `l2_sf_pair_arrival_mask`，用于标记相邻两个 64-col half 的 raw output/SF 已经 ready；
- odd half CTA 等待 even half ready 后，在 L2 activation buffer 上做 in-place 重标定：
  - `pair_sf = max(sf_even, sf_odd)`；
  - 读取已写出的 FP8，按 `old_sf / pair_sf` 重量化回 FP8；
  - 额外写 final per-128 SF 列；
  - 一次性发布两个 half 的 final L2 ready bit；
- L2 侧用 final per-128 SF TMA descriptor，每个 `BLOCK_K=128` 只加载 1 组 activation SF，并走单组 full-K WGMMA。

同时更新 SM90 correctness reference，使该 env 下 reference 使用 per-128 activation quant。

正确性 smoke：

```bash
DG_SM90_MEGA_MOE_L2_ACT_SF_PER128=1 MASTER_PORT=9991 \
python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 --fail-fast
```

结果：`[L1.smoke] diff=0.0007 (tol=0.07) OK`。

### 日志

- 主性能：`benchmark_outputs/sm90_l2_act_sf_per128_v1_v4flash_512_4096_9192.log`
- stage profile：`benchmark_outputs/sm90_l2_act_sf_per128_stage_profile_v1_v4flash_4096_9192.log`

### 性能结果

8 rank 中位数，matched capacity，`num_bench_tests=30`、baseline `num_repeat=20`：

| batch | per-128 v1 fused us | baseline us | baseline/fused | 相比默认 fused |
|---:|---:|---:|---:|---:|
| 512 | 1543.0 | 1201.5 | 0.779x | 默认 fused 约 1153 us，v1 慢约 390 us |
| 4096 | 10029.5 | 6896.0 | 0.688x | 默认 fused 约 6999 us，v1 慢约 3031 us |
| 9192 | 22109.0 | 14997.5 | 0.678x | 默认 fused 约 15287 us，v1 慢约 6822 us |

### stage profile 摘要

rank 中位数，单位 us。对照上一轮默认 fused profile：

| batch | v1 L1 total | L1 GEMM | L1 epi | v1 L2 total | L2 GEMM | L2 epi | producer L2 wait | 默认 L1 epi | 默认 L2 GEMM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 6807.8 | 3923.5 | 2893.7 | 2553.8 | 2174.7 | 384.6 | 341.8 | 368.7 | 1793.6 |
| 9192 | 15140.4 | 8564.0 | 6591.5 | 5768.8 | 4951.8 | 815.2 | 986.9 | 839.4 | 3923.7 |

### 结论

per-128 方向的这个 v1 形态失败，不能继续沿用“事后重标定”作为优化实现：

- 主要新增开销落在 L1 epilogue：
  - batch 4096：L1 epi 从约 369 us 增到约 2894 us，新增约 2.5 ms；
  - batch 9192：L1 epi 从约 839 us 增到约 6591 us，新增约 5.75 ms。
- L2 并没有得到预期收益：
  - batch 4096：L2 GEMM 从约 1794 us 增到约 2175 us；
  - batch 9192：L2 GEMM 从约 3924 us 增到约 4952 us；
  - 同时 producer L2 wait 也明显上升，说明 L2 被 L1 pair finalize/发布节奏拖住。
- 根因是 v1 需要额外从 global 读已量化 FP8、重新量化写回 128 列，并引入 pair half 等待与发布同步；这笔内存读写/同步成本远大于省掉 per-64 split-WGMMA 的潜在收益。

后续方向应改为真正结构性 per-128，而不是 post-hoc rescale：

1. L1 epilogue 内部跨两个 64-col half 聚合 amax，在写出前一次性使用 per-128 scale 量化，避免读回/写回；
2. 或改变 L1 output tile/scheduler，让同一 CTA 或确定配对 CTA 在寄存器/SMEM 阶段完成 pair-scale finalize；
3. 在继续 per-128 前，先做一个 L2-only synthetic variant：直接喂 per-128-ready activation/SF 给 L2，估算单独去掉 split-WGMMA 的收益上限，避免再被 L1 finalize 成本污染判断。

## 2026-05-17：SM90 L2 activation SF per-128 v2（写出前 finalize）

### 目的

修正 v1 的主要问题：不再把已经写到 global 的 FP8 activation 读回重标定，而是在 L1 epilogue 写出前等待 pair raw SF，直接用 per-128 final SF 量化本 CTA 负责的 64 列。

### 实现

仍使用同一个 env：

- `DG_SM90_MEGA_MOE_L2_ACT_SF_PER128=1`

与 v1 的差异：

- 每个 L1 half CTA 先计算本 half 的 raw per-64 SF；
- 写 raw SF 后发布 `l2_sf_pair_arrival_mask`；
- 等待相邻 half 的 raw SF ready；
- `pair_sf = max(sf_self, sf_peer)` 后，在寄存器中的 SwiGLU 值直接按 `pair_sf` 量化写出；
- 每个 half 只发布自己的 final `l2_arrival_mask` bit；
- 删除 v1 的 global FP8 readback / in-place rescale / odd-half finalize loop。

### 正确性

先跑 correctness，再跑性能。

Smoke：

```bash
DG_SM90_MEGA_MOE_L2_ACT_SF_PER128=1 MASTER_PORT=9992 \
python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 --fail-fast
```

结果：`[L1.smoke] diff=0.0006 (tol=0.07) OK`。

Heuristic sweep：

```bash
DG_SM90_MEGA_MOE_L2_ACT_SF_PER128=1 MASTER_PORT=9994 \
python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 --fail-fast
```

结果：5/5 通过，`t64/t256/t512/t2048` 均 `diff=0.0006`。

### 日志

- 主性能：`benchmark_outputs/sm90_l2_act_sf_per128_v2_prewrite_v4flash_512_4096_9192.log`
- stage profile：`benchmark_outputs/sm90_l2_act_sf_per128_v2_prewrite_stage_profile_v4flash_4096_9192.log`

### 性能结果

8 rank 中位数，matched capacity，`num_bench_tests=30`、baseline `num_repeat=20`：

| batch | per-128 v2 fused us | baseline us | baseline/fused | 相比默认 fused | 相比 v1 |
|---:|---:|---:|---:|---:|---:|
| 512 | 1234.0 | 1211.0 | 0.981x | 默认 fused 约 1153 us，v2 慢约 81 us | v1 1543 us，v2 快约 309 us |
| 4096 | 7536.0 | 6910.5 | 0.917x | 默认 fused 约 6999 us，v2 慢约 537 us | v1 10029.5 us，v2 快约 2493 us |
| 9192 | 16474.0 | 15000.0 | 0.911x | 默认 fused 约 15287 us，v2 慢约 1187 us | v1 22109 us，v2 快约 5635 us |

### stage profile 摘要

rank 中位数，单位 us。对照上一轮默认 fused profile：

| batch | v2 L1 total | L1 GEMM | L1 epi | v2 L2 total | L2 GEMM | L2 epi | producer L2 wait | 默认 L1 epi | 默认 L2 GEMM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 4534.8 | 3941.3 | 595.7 | 2290.3 | 1931.8 | 361.6 | 33.6 | 368.7 | 1793.6 |
| 9192 | 9956.1 | 8611.5 | 1339.9 | 5034.7 | 4248.9 | 783.6 | 73.2 | 839.4 | 3923.7 |

### 结论

v2 证明了 v1 的失败主要来自 global readback/rescale：去掉这部分后，三组 batch 都大幅恢复。但 per-128 prewrite 仍不该作为默认优化：

- v2 仍慢于默认 fused：
  - batch 4096 慢约 0.54 ms；
  - batch 9192 慢约 1.19 ms。
- L1 epilogue 仍有额外成本：
  - batch 4096：L1 epi 从默认约 369 us 增到约 596 us；
  - batch 9192：L1 epi 从默认约 839 us 增到约 1340 us；
  - 这部分主要是 pair raw-SF publish/wait、额外 full-epilogue sync、final SF 写入。
- L2 GEMM 也没有变快，反而略慢：
  - batch 4096：L2 GEMM 从默认约 1794 us 增到约 1932 us；
  - batch 9192：L2 GEMM 从默认约 3924 us 增到约 4249 us。

这说明“把 L2 activation SF 改成 per-128”本身并没有自动带来预期收益；当前 SM90 per-64 split-WGMMA 路径虽然多一次 scale/split，但可能更匹配现有 A/B TMA、pipeline 和 WGMMA issue 结构。下一步不应继续在 per-128 activation SF 上硬推，而应回到 L2-only 的具体微结构拆解：

1. 做 L2-only synthetic/profile：固定同一份 L2 input，比较 per-64 split、per-128 single-group、dual-accum 的纯 L2 kernel 时间；
2. 如果 L2-only 仍显示 per-128 不快，转向 L2 epilogue/scatter 或 scheduler overlap；
3. 如果 L2-only 显示 per-128 快，但 full fused 慢，则说明问题在 L1/L2 handoff 同步，需要重新设计 pair handoff，而不是增加 epilogue wait。

## 2026-05-17：NCU 对比补充（default fused vs per-128 v2，batch 4096）

### 目的

在继续改代码前，用 NCU 验证 stage profile 的判断：当前大 batch 差距是否真的应该继续押注 per-128 L2 activation SF，还是更像 fused persistent kernel 内部同步/调度问题。

### 日志

- default fused full sections：`benchmark_outputs/ncu_sm90_default_b4096_full/`
- per-128 v2 full sections：`benchmark_outputs/ncu_sm90_per128v2_b4096_full/`
- default fused SourceCounters：`benchmark_outputs/ncu_sm90_default_b4096_source/`

说明：NCU 使用 application replay，且本轮 `--clock-control none`，所以 report 里的 `gpu__time_duration` 只用于 sanity check，不作为性能结论。性能结论仍以前面的 benchmark/stage profile 为准。

### 关键指标

8 rank NCU report 的中位数：

| metric | default | per-128 v2 | 变化 |
|---|---:|---:|---:|
| `launch__registers_per_thread` | 168 | 168 | 持平 |
| `launch__shared_mem_per_block_dynamic` | 209.6 KB | 209.6 KB | 持平 |
| `launch__waves_per_multiprocessor` | 1 | 1 | 持平 |
| `inst_executed` | 6.88e8 | 8.15e8 | +18.4% |
| `smsp__issue_active.avg.per_cycle_active` | 0.075 | 0.110 | +46.7% |
| `smsp__warps_active.avg.per_cycle_active` | 2.49 | 2.49 | 持平 |
| `smsp__warps_eligible.avg.per_cycle_active` | 0.067 | 0.114 | +71.5% |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active` | 22.9 | 20.5 | -10.7% |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active` | 6.32 | 5.93 | -6.1% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | 4.17% | 6.17% | +47.8% |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | 1.63% | 2.37% | +45.2% |

### SourceCounters 观察

default fused 的 SourceCounters 显示 stall 主体集中在同步/等待，而不是 DRAM/TMA 饱和：

- top stall 主要落在 `BAR.SYNC.DEFER_BLOCKING`、role/pipeline 分支后的 barrier stall，以及多处 `WARPGROUP.DEPBAR.LE gsb0`；
- 聚合 8 rank 后，最大的几个 source/SASS 点分别约为：
  - `@P0 BRA ...`：约 4.50M barrier samples；
  - `BAR.SYNC.DEFER_BLOCKING 0x1, 0x100`：约 3.55M barrier samples；
  - `@!P3 BRA ...`：约 1.47M barrier samples；
  - 多个 `WARPGROUP.DEPBAR.LE gsb0` 合计约 0.8M barrier samples 量级。

这些点和代码结构对应的是 persistent CTA 内的角色分工、producer/consumer barrier、WGMMA wait 以及 L1/L2 handoff 同步。结合 full sections 中的低 DRAM 利用率、低 eligible warps 和 1 CTA/SM（209 KB dynamic SMEM），当前更像同步/调度受限，而不是带宽受限。

### 结论

NCU 没有支持继续把 per-128 v2 往默认路径推进：

- per-128 v2 没有降低寄存器/SMEM，也没有改变 1 CTA/SM 的占用结构；
- 它增加了约 18% 的指令，并引入额外 L1 handoff 工作；这与 stage profile 中 L1 epi、L2 GEMM 都变慢一致；
- default fused 的主症状是低 eligible warp + barrier/source wait，而不是 DRAM、L2 或 TMA pipe 饱和。

下一步应该先做更精确的 L2-only / micro profile，而不是继续直接改 per-128 handoff：

1. 在同一份 L2 input 上隔离比较 default per-64 split、dual-accum、synthetic per-128 single-group；
2. 如果 L2-only 仍不快，优化重点转向 L2 epilogue/scatter 或 persistent pipeline 的 barrier/occupancy；
3. 如果 L2-only 显示 per-128 快，再回头重设计 L1/L2 handoff，目标是避免 v2 的 pair publish/wait 和额外 full-epilogue sync。

## 2026-05-18：SM90 同步细项 profile 验证（v17）

### 目的

验证“同步受限”具体落在哪里，尤其区分：

- GEMM producer/consumer `full_barriers.wait`；
- WGMMA `warpgroup_wait`；
- L1 epilogue WG sync / TMA store wait / publish 前 full sync；
- L2 epilogue WG sync / scatter 后 full sync。

### 改动

- 文件：`deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`
- 将 `DG_SM90_MEGA_MOE_STAGE_PROFILE` 的 CTA stride 从 24 扩到 32；
- 新增 profile 槽位：
  - `20`: GEMM `full_barriers.wait`；
  - `21`: WGMMA `warpgroup_wait<0>`；
  - `22`: L1 WG sync；
  - `23`: L1 TMA store wait；
  - `24`: L1 full-epilogue sync；
  - `25`: L2 WG sync；
  - `26`: L2 full-epilogue sync。
- 同步更新 `tests/test_mega_moe_hopper.py` 的 summary/format，以及 `csrc/apis/mega.hpp` 的 profile tensor size check。

### 验证

容器内重编 `_C` 后确认 JIT marker 已更新为 `v17_sync_profile_detail`。

Correctness：

```bash
python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：28/28 通过，所有 diff 均在 `tol=0.07` 内。

日志：

- `benchmark_outputs/sm90_sync_profile_correctness_full.log`
- `benchmark_outputs/sm90_sync_profile_correctness_l1.log`
- `benchmark_outputs/sm90_sync_detail_stage_profile_v17_512_4096_9192.log`
- `benchmark_outputs/sm90_sync_detail_perf_noprofile_v17_512_4096_9192.log`

### 正常性能（不打开 stage profile define）

matched capacity，8 rank，`num_bench_tests=5`、baseline `num_repeat=5`：

| batch | fused us | baseline us | 结论 |
|---:|---:|---:|---|
| 512 | 1133-1135 | 1228-1230 | fused 快约 8% |
| 4096 | 6913-6916 | 6899-6905 | 基本持平，baseline 略快约 0.2% |
| 9192 | 15107-15111 | 14999-15002 | baseline 快约 0.7% |

说明：打开 `DG_SM90_MEGA_MOE_STAGE_PROFILE=1` 后，fused 计时会明显变慢（例如 batch 4096 约 7.48 ms、9192 约 16.33 ms），因此该模式只用于拆解占比，不作为真实性能。

### 同步细项观察

stage-profile 模式下，关键 rank 的量级如下：

| batch | fused(profile) | math max | `full_barriers.wait` | WGMMA wait | L1 WG/TMA/full | L2 WG/full |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1225 us | 1090-1097 us | 80-85 us | 1.7 us | 0.2 / 1.8-1.9 / 0.0 us | 0.2 / 21-23 us |
| 4096 | 7473-7477 us | 6640-6749 us | 454-462 us | 10.6-10.7 us | 1.4 / 11.3-11.5 / 0.2 us | 0.7 / 29-34 us |
| 9192 | 16333-16336 us | 14515-14742 us | 969-984 us | 23.1-23.5 us | 3.1 / 24.5-25.1 / 0.4 us | 1.4-1.5 / 50-62 us |

### 结论

本轮验证不支持“L1 TMA store wait 或 epilogue full sync 是大 batch 主因”：

- L1 TMA wait 很小：4096 约 11 us，9192 约 25 us；
- L1 publish 前 full sync 几乎为 0.2-0.4 us；
- L2 scatter 后 full sync 也只有几十 us；
- WGMMA wait 很小，说明 default per-64 split 多出的 WGMMA wait 不是主要瓶颈。

更大的同步项是 GEMM `full_barriers.wait`，随 batch 放大到 4096 约 0.46 ms、9192 约 0.98 ms。它更像 producer/consumer pipeline 等待 A/B/SFA TMA 到达或 persistent CTA 内角色调度造成的空转，而不是 L1 output store/wait 本身。

下一步优化优先级应调整为：

1. 继续拆 `full_barriers.wait` 的来源：A+SFA producer、B producer、SFA load、pipeline stage depth；
2. 优先做 L2-only / producer micro profile，而不是先动 L1 publish sync；
3. 如果要做结构优化，方向更像 SM100 的 producer/consumer 解耦、提高可隐藏等待的并行度，或减少 1 CTA/SM 下的 pipeline 空窗。

## 2026-05-18：SM90 producer 细项 profile（v18）

### 目的

继续拆上一轮发现的 `full_barriers.wait`。本轮重点区分：

- A+SFA producer 是否比 B producer 更重；
- producer 是否自己卡在 `empty_barriers.wait`；
- L2 per-64 SFA 的两次 TMA 是否确实造成 A producer 压力。

### 改动

- 文件：`deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`
- 将 stage profile stride 从 32 扩到 40；
- 新增 profile 槽位：
  - `27/28`: A+SFA producer 在 L1/L2 的 `empty_barriers.wait`；
  - `29/30`: A+SFA producer 在 L1/L2 的 issue 区间；
  - `31/32`: B producer 在 L1/L2 的 `empty_barriers.wait`；
  - `33/34`: B producer 在 L1/L2 的 issue 区间。
- JIT marker 更新为 `v18_producer_profile_detail`。

### 验证

```bash
python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：28/28 通过。

日志：

- `benchmark_outputs/sm90_producer_profile_correctness_full.log`
- `benchmark_outputs/sm90_producer_profile_correctness_l1.log`
- `benchmark_outputs/sm90_producer_detail_stage_profile_v18_512_4096_9192.log`
- `benchmark_outputs/sm90_producer_detail_pair_tma_stage_profile_v18_4096_9192.log`

### default producer profile 摘要

8 rank 中位数，单位 us。注意这些是 profile 模式下的 per-CTA 累计计时，只用于结构占比判断。

| batch | math full wait | A issue L1 | A issue L2 | B issue L1 | B issue L2 | A empty L1/L2 | B empty L1/L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 84.7 | 98.1 | 47.7 | 49.0 | 25.5 | 522.1 / 292.3 | 605.7 / 335.0 |
| 4096 | 462.4 | 589.1 | 292.9 | 305.0 | 158.1 | 3218.5 / 1793.5 | 3646.6 / 2047.2 |
| 9192 | 999.2 | 1315.7 | 635.8 | 661.6 | 340.8 | 7068.4 / 3935.1 | 8000.9 / 4489.7 |

### pair-TMA 对照

只打开 `DG_SM90_MEGA_MOE_L2_SFA_PAIR_TMA=1`，其它相同：

| batch | default A issue L2 | pair-TMA A issue L2 | default full wait | pair-TMA full wait | 结论 |
|---:|---:|---:|---:|---:|---|
| 4096 | 292.9 | 285.1 | 462.4 | 465.9 | A issue 略降，但 full wait 不降 |
| 9192 | 635.8 | 618.6 | 999.2 | 984.1 | A issue 略降，full wait 小幅波动 |

### 结论

本轮更清楚地把瓶颈定位到了 producer/consumer pipeline 节奏，而不是某个单点 transaction 数：

- A+SFA issue 明显比 B issue 重，约为 B 的 1.9-2.0x；
- 但 producer 的 `empty_barriers.wait` 远大于 issue 时间，说明 producer 很多时候也在等 math 释放 stage；
- pair-TMA 能降低一点 L2 A issue，但无法稳定降低 `full_barriers.wait`，因此“L2 SFA 两次 TMA”不是主因；
- `full_barriers.wait` 更像 TMA transaction 完成时间、math 消费节奏、stage 环形队列深度在 1 CTA/SM 下共同形成的空窗。

下一步不建议继续做 pair-TMA 或 L1 store/wait 小改。更值得做的方向：

1. 做 L2-only microbenchmark，隔离 L2 GEMM+producer，不带 L1/dispatch/combine；
2. 做 stage-depth/SMEM 结构实验，验证是否能通过减少 per-CTA SMEM 或改变 stage 数换 occupancy/隐藏等待；
3. 如果继续融合路径，考虑更接近 SM100 的 accumulator/producer 解耦，而不是继续减少单个 TMA transaction。

## 2026-05-18：SM90 forced stage-depth sweep（v19）

### 目的

验证上一轮提出的结构假设：在 1 CTA/SM 下，stage ring 深度、TMA transaction 完成和 math 消费节奏互相卡住。先不改 kernel 主体，只把 SM90 pipeline stage 数从默认 max stage 改成可控变量，观察：

- 更短 stage ring 是否减少 producer/consumer 空窗；
- 降低 per-CTA shared memory 后，潜在更高 CTA residency 是否能改善大 batch；
- small batch 优势是否被破坏。

### 改动

- 文件：`csrc/jit_kernels/heuristics/mega_moe.hpp`
- 新增 `DG_SM90_MEGA_MOE_FORCE_NUM_STAGES`：
  - 默认 `0`，保持原来的 max-stage 策略；
  - 非 0 时 assert 在 `[2, max_num_stages]` 内，然后强制使用该 stage 数。
- JIT marker 更新为 `v19_force_stage_sweep`。

### 正确性验证

先 smoke `stage=3..7`，再对最激进的两个低 stage 跑完整 correctness：

```bash
DG_SM90_MEGA_MOE_FORCE_NUM_STAGES=2 python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
DG_SM90_MEGA_MOE_FORCE_NUM_STAGES=3 python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

- `stage=2`: 28/28 通过；
- `stage=3`: 28/28 通过；
- `stage=3..7` smoke 均通过；
- `DG_JIT_DEBUG=1` 确认 JIT marker 为 `v19_force_stage_sweep`，且 forced stage 生效。

日志：

- `benchmark_outputs/sm90_force_stages_v19_correctness_smoke_s3.log`
- `benchmark_outputs/sm90_force_stages_v19_correctness_smoke_s3_s7.log`
- `benchmark_outputs/sm90_force_stages_v19_correctness_full_s2.log`
- `benchmark_outputs/sm90_force_stages_v19_correctness_full_s3.log`

### V4 shape 下的 SMEM

配置：8 rank，`hidden=4096`，`intermediate_hidden=2048`，`num_topk=6`，`num_tokens=4096`。

| forced stages | smem_size | smem KiB |
|---:|---:|---:|
| 2 | 84096 | 82.1 |
| 3 | 109200 | 106.6 |
| 4 | 134304 | 131.2 |
| 5 | 159408 | 155.7 |
| 6 | 184512 | 180.2 |
| 7 | 209616 | 204.7 |

`stage=2/3` 已经明显低于默认 7-stage 的 204.7 KiB，理论上是最可能验证“降低 SMEM / 提高 CTA residency 是否能隐藏等待”的两个点。

### 正常性能（不打开 stage profile define）

matched capacity，8 rank，`num_bench_tests=5`、baseline `num_repeat=5`。表内为 8 rank median，单位 us；`gap` 为 fused 相对 baseline，负数表示 fused 快。

| forced stages | batch 512 fused/base/gap | batch 4096 fused/base/gap | batch 9192 fused/base/gap |
|---:|---:|---:|---:|
| 2 | 1143.5 / 1227.0 / -6.81% | 6957.0 / 6894.0 / +0.91% | 15150.5 / 14984.0 / +1.11% |
| 3 | 1134.0 / 1337.0 / -15.18% | 6921.5 / 6898.0 / +0.34% | 15141.0 / 15038.0 / +0.68% |
| 4 | 1135.0 / 1226.0 / -7.42% | 6940.0 / 6903.0 / +0.54% | 15175.0 / 14975.0 / +1.34% |
| 5 | 1134.0 / 1236.0 / -8.25% | 6930.0 / 6886.0 / +0.64% | 15130.0 / 15007.0 / +0.82% |
| 6 | 1134.0 / 1229.0 / -7.73% | 6928.0 / 6920.0 / +0.12% | 15139.0 / 15007.0 / +0.88% |
| 7 | 1134.0 / 1227.0 / -7.58% | 6914.5 / 6918.0 / -0.05% | 15091.0 / 14975.5 / +0.77% |

日志：

- `benchmark_outputs/sm90_force_stages_v19_perf_512_4096_9192.log`
- `benchmark_outputs/sm90_force_stages_v19_config_4096.log`

### 结论

本轮基本排除“单纯 forced stage-depth / 降 SMEM 就能解决大 batch gap”：

- small batch 512 基本稳定，fused 仍保持优势，说明低 stage 不破坏 correctness 和基本 pipeline；
- 大 batch 4096/9192 没有随 stage 变小而改善，`stage=2/3` 反而不如默认 7-stage 稳；
- `stage=2/3` 已把 SMEM 降到 82.1/106.6 KiB，但没有转化为明显速度收益，说明问题不是简单的 CTA residency 开关；
- 默认 7-stage 仍是本轮最稳的点：4096 基本持平，9192 仍比 baseline 慢约 0.8%。

下一步不要把 forced stage 作为优化方向本身。更合理的推进是：继续保留该 env 开关作为诊断工具，然后做更结构性的拆分实验，优先隔离 L2-only producer/math cadence，或者尝试重分配 producer 工作，让 A/SFA 和 B producer 不再在同一个固定节奏里互相拖住。

## 2026-05-18：SM90 naive 2 CTA/SM launch 验证（v20）

### 目的

上一轮 stage sweep 虽然把 `stage=2/3` 的 SMEM 降到了 82.1/106.6 KiB，但 launch grid 仍然只有 `active_sms` 个 CTA，因此并没有真正验证 2 CTA/SM。为了直接验证“多 CTA residency 是否能隐藏 producer/math 互等”，本轮加一个诊断性 CTA multiplier。

### 改动

- 文件：`csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`
- 新增 `DG_SM90_MEGA_MOE_CTA_MULTIPLIER`：
  - 默认 `1`，保持原行为；
  - 允许值 `1/2`；
  - `2` 时 launch grid 从 `active_sms` 改为 `active_sms * 2`，scheduler 的 `kNumSMs` 也随之变为 2x CTA 数。
- JIT marker 更新为 `v20_cta_multiplier`。

### 验证

先跑最小 smoke：

```bash
DG_SM90_MEGA_MOE_FORCE_NUM_STAGES=3 \
DG_SM90_MEGA_MOE_CTA_MULTIPLIER=2 \
DG_JIT_DEBUG=1 \
timeout 90s python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 --fail-fast
```

JIT/launch 确认：

- `active_sms=78`
- `launch_ctas=156`
- `cta_multiplier=2`
- `num_stages=3`
- smoke 90s timeout，退出状态 `124`，没有到达 correctness PASS。

日志：

- `benchmark_outputs/sm90_cta2_v20_correctness_smoke_s3.log`
- `benchmark_outputs/sm90_cta2_v20_correctness_smoke_s3_pipefail.log`

### 结论

naive 2 CTA/SM 不能直接作为当前 fused kernel 的优化方向：

- 当前 kernel 使用 software grid sync / NVLink barrier，要求参与 grid sync 的 CTA 都能同时常驻；
- 虽然 `stage=3` 的 SMEM 已降到约 95-107 KiB，但 SM90 这版每 CTA 的寄存器预算接近 64K（代码中 register reconfiguration budget 为 64512），实际仍是 1 CTA/SM；
- 因此 launch 2x CTA 后，部分 CTA 无法常驻，已常驻 CTA 在 grid sync 等待未常驻 CTA，造成 timeout；
- 这解释了为什么上一轮降低 SMEM 没有自动得到 2 CTA/SM 收益：真正的限制不只是 SMEM，还有 register/role budget 和 cooperative-style grid sync 语义。

后续如果要走 2 CTA/SM，必须先做更重的结构改造：降低每 CTA register footprint、减少线程/角色数，或拆掉当前要求全 CTA 同驻的 grid sync 语义。短期更现实的优化方向仍是 producer/math cadence 内部重排，或者拆 L2-only 进行隔离分析。

## 2026-05-18：SM90 split A/SFA producer 验证（v21）

### 目的

v18 producer profile 显示 A+SFA issue 明显重于 B issue，约为 B 的 1.9-2.0x。本轮尝试使用原本 idle 的 non-epilogue warp，把 SFA 从 A producer 中拆出去：

- warp `kNumDispatchWarps + 0`: 只 TMA load A；
- warp `kNumDispatchWarps + 1`: 继续 TMA load B/SFB；
- warp `kNumDispatchWarps + 2`: 新增 SFA producer；
- full barrier arrival count 从 2 改为 3。

### 改动

- 文件：
  - `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`
  - `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`
- 新增 `DG_SM90_MEGA_MOE_SPLIT_A_SFA_PRODUCER`，默认关闭；
- JIT marker 更新为 `v21_split_a_sfa_producer`。

### 正确性验证

默认路径 smoke：

```bash
DG_JIT_DEBUG=1 python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 --fail-fast
```

split producer smoke：

```bash
DG_SM90_MEGA_MOE_SPLIT_A_SFA_PRODUCER=1 DG_JIT_DEBUG=1 \
python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 --fail-fast
```

完整 correctness：

```bash
DG_SM90_MEGA_MOE_SPLIT_A_SFA_PRODUCER=1 \
python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

结果：

- 默认 smoke 通过；
- split producer smoke 通过；
- split producer 28/28 通过；
- ptxas 仍为 `Used 168 registers`，没有新增 spill。

日志：

- `benchmark_outputs/sm90_split_a_sfa_v21_correctness_smoke_default.log`
- `benchmark_outputs/sm90_split_a_sfa_v21_correctness_smoke_split.log`
- `benchmark_outputs/sm90_split_a_sfa_v21_correctness_full.log`
- `benchmark_outputs/sm90_split_a_sfa_v21_perf_512_4096_9192.log`

### 正常性能

matched capacity，8 rank，`num_bench_tests=5`、baseline `num_repeat=5`。表内为 8 rank median，单位 us；`gap` 为 fused 相对 baseline，负数表示 fused 快。

| batch | split fused | baseline | gap |
|---:|---:|---:|---:|
| 512 | 1139.0 | 1221.5 | -6.75% |
| 4096 | 6954.5 | 6897.0 | +0.83% |
| 9192 | 15206.5 | 15010.0 | +1.31% |

对照默认 v17/v18 正常性能：512 默认约 1133-1135 us，4096 默认约 6913-6916 us，9192 默认约 15107-15111 us。split A/SFA 在三个 batch 上均没有收益，大 batch 退化更明显。

### 结论

“A+SFA issue 比 B 重”是真现象，但把 SFA 拆到 idle warp 不是有效优化：

- 新增 producer 后，full barrier arrival 从 2 变 3，math 需要等第三个 arrival；
- SFA 自身 transaction 很小，拆分后节省不了 A producer 的主路径，反而增加了一条同步参与者；
- producer profile 里更大的项是 `empty_barriers.wait`，不是 issue 本身；拆 issue 不能解决 producer 等 math 释放 stage 的问题。

下一步不要继续沿“把 SFA 单独拆 warp”深挖。更值得做的是 L2-only 隔离，或者更激进地改 math/epilogue 持有 accumulator 的节奏，而不是只拆 producer transaction。

## 2026-05-18：baseline L2-only 隔离下界（v1）

### 目的

前几轮已经排除了 stage-depth、naive 2 CTA/SM、split A/SFA producer。最后补一个 L2-only 隔离参照：不改 fused kernel 正路径，在 `tests/test_mega_moe_hopper.py` 中新增 benchmark-only 模式，先用 baseline 跑 dispatch → L1 GEMM → SwiGLU/quant，固定好 L2 输入和 handle，然后只反复测 L2 grouped GEMM。

这个结果不是 fused L2 的逐项计时，而是一个“单独 L2 grouped GEMM 下界”，用于判断大 batch gap 是否可能由 L2 GEMM 计算本身单独造成。

### 改动

- 文件：`tests/test_mega_moe_hopper.py`
- 新增参数 `--baseline-l2-only-profile`：
  - 自动启用 baseline buffer；
  - 准备 L2 输入后，只测 `m_grouped_fp8_gemm_nt_contiguous(l2_input, l2_weights, ...)`；
  - 打印每个 rank 的 L2 GEMM event 时间。

### 验证和结果

命令：

```bash
python3 tests/test_mega_moe_hopper.py \
  --num-processes 8 \
  --num-max-tokens-per-rank <batch> --num-tokens <batch> \
  --hidden 4096 --intermediate-hidden 2048 \
  --num-experts 256 --num-topk 6 \
  --num-warmup 5 --num-repeat 20 \
  --l2-flush-gb 0 \
  --baseline-l2-only-profile
```

结果（8 rank，单位 us）：

| batch | L2-only median | min | max |
|---:|---:|---:|---:|
| 4096 | 1704.4 | 1681.3 | 1723.8 |
| 9192 | 3683.8 | 3666.6 | 3719.8 |

日志：

- `benchmark_outputs/sm90_baseline_l2_only_v2_4096_9192.log`

### 结论

单独 L2 grouped GEMM 的量级远低于 fused 端到端：

- batch 4096：L2-only 约 1.70 ms，fused 约 6.91-6.96 ms；
- batch 9192：L2-only 约 3.68 ms，fused 约 15.09-15.21 ms。

因此大 batch gap 不像是“L2 GEMM 计算本身太慢”单独造成的。更合理的解释仍是 fused 内部 L1→L2 的生产/消费依赖、stage ring、full barrier wait，以及 epilogue/accumulator 持有时间共同形成的 cadence 问题。

到目前为止，四个结构方向的结论是：

1. forced stage-depth：可诊断，但不是优化旋钮；
2. naive 2 CTA/SM：会因 register residency + software grid sync 语义 timeout，不能直接做；
3. split A/SFA producer：正确但性能退化，不继续；
4. L2-only 下界：L2 GEMM 本身不是单独主因，下一刀应落在 fused pipeline cadence，而不是 L2 GEMM 算子本体。
