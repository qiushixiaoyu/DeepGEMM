# SM90 MegaMoE optimization experiments

Config used for performance sweeps unless noted:

- Script: `tests/test_mega_moe_hopper.py`
- GPUs/ranks: `8`
- Model: DeepSeek-V4-Flash style
- `hidden=4096`
- `intermediate_hidden=2048`
- `num_experts=256`
- `num_topk=6`
- Fused-only, baseline disabled
- `num_bench_tests=5`
- Batch list requested: `1 2 4 8 16 32 64 512 1024 4096 9192`

## 2026-05-15: active-SM heuristic v1

### Hypothesis

NCU showed low eligible warps and high fixed overhead for small batches. Try reducing the number of participating persistent CTAs for `block_m=64` small-batch decode, while keeping full SM participation for larger configs.

### Change

File: `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`

Added `get_active_sms_for_sm90_mega_moe()` and wired `LaunchArgs(active_sms, ...)`.

After measurement, the heuristic is **disabled by default** and only available behind:

- `DG_SM90_MEGA_MOE_ENABLE_ACTIVE_SMS_HEURISTIC=1`
- `DG_SM90_MEGA_MOE_ACTIVE_SMS=<N>` for forced override

### Correctness

Command:

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8731 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

Result:

```text
PASSED all 28 scenarios
```

### Performance

Raw logs:

- Disabled/full-SM baseline: `benchmark_outputs/sm90_active_sms_v1_perf_disabled.log`
- Enabled active-SM experiment: `benchmark_outputs/sm90_active_sms_v1_perf_enabled.log`

Median fused time across 8 ranks:

| batch | full-SM median us | active-SM v1 median us | active/full | verdict |
|---:|---:|---:|---:|---|
| 1 | 190.0 | 748.0 | 3.94x | rejected |
| 2 | 303.5 | 1146.5 | 3.78x | rejected |
| 4 | 423.5 | 1202.5 | 2.84x | rejected |
| 8 | 511.5 | 1269.0 | 2.48x | rejected |
| 16 | 617.5 | 916.0 | 1.48x | rejected |
| 32 | 583.5 | 607.5 | 1.04x | rejected |
| 64 | 600.0 | not run | n/a | stopped after clear degradation |
| 512 | 1472.5 | not run | n/a | stopped after clear degradation |
| 1024 | 2815.0 | not run | n/a | stopped after clear degradation |
| 4096 | 8646.5 | not run | n/a | stopped after clear degradation |
| 9192 | 18202.0 | not run | n/a | stopped after clear degradation |

### Conclusion

Rejected as a default optimization. The profiling hypothesis was incomplete: although small batches pay high fixed dispatch/sync cost, the current fused kernel still needs full-SM parallelism to keep the L1/L2 block stream moving. Reducing persistent CTAs hurts more than it saves.

Keep the env-gated implementation only as an experiment hook. Next optimization should target reducing all-expert dispatch/count/scheduler overhead without reducing the number of CTAs available to execute GEMM work.

## 2026-05-15: block_m=64 wave heuristic v2

### Hypothesis

The original SM90 `block_m=64` heuristic forced all local experts into one wave. The broader generic-wave experiment showed that extra wave boundaries are not universally good, but the mid-sparse decode band can benefit from narrower expert waves.

For the DeepSeek-V4-Flash style shape (`hidden=4096`, `intermediate_hidden=2048`, `num_experts=256`, `topk=6`, EP8), only batches 8 and 16 materially change behavior:

- batch 1/2/4: expected tokens per expert < 1, generic heuristic also returns all experts per wave
- batch 8/16: expected tokens per expert is 1.5/3.0, generic heuristic returns narrower waves
- batch 32/64: expected tokens per expert is 6.0/12.0, keep single-wave to avoid extra wave-boundary overhead
- batch >= 512: `block_m > 64`, already uses the generic heuristic before this change

### Change

File: `csrc/jit_kernels/heuristics/mega_moe.hpp`

For SM90 `block_m=64`, default to generic wave scheduling only when:

```text
1.0 <= expected_tokens_per_expert <= 4.0
```

The broader experiment remains available behind:

```text
DG_SM90_MEGA_MOE_BLOCK64_GENERIC_WAVES=1
```

### Correctness

Command:

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8751 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

Result:

```text
PASSED all 28 scenarios
```

### Performance

Raw logs:

- Broader generic-wave exploration: `benchmark_outputs/sm90_block64_generic_waves_perf_enabled.log`
- Accepted v2 full sweep: `benchmark_outputs/sm90_block64_wave_v2_perf.log`
- v2 small-batch rerun: `benchmark_outputs/sm90_block64_wave_v2_small_rerun.log`

Median fused time across 8 ranks, full sweep:

| batch | recv tok med | active exp med | full-SM us | generic all-block64 us | wave v2 us | v2/full | v2 delta |
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

Small-batch rerun with `num_bench_tests=7`, `num_warmup=3`, `num_repeat=7`:

| batch | full-SM us | v2 full sweep us | v2 rerun us | rerun/full | rerun delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 190.0 | 197.5 | 195.0 | 1.026x | +2.6% |
| 2 | 303.5 | 363.5 | 281.5 | 0.928x | -7.2% |
| 4 | 423.5 | 410.5 | 416.0 | 0.982x | -1.8% |
| 8 | 511.5 | 506.5 | 508.5 | 0.994x | -0.6% |
| 16 | 617.5 | 573.5 | 578.0 | 0.936x | -6.4% |
| 32 | 583.5 | 597.0 | 586.0 | 1.004x | +0.4% |
| 64 | 600.0 | 604.0 | 612.5 | 1.021x | +2.1% |

### Conclusion

Accepted as a conservative default for now. The only behavior-changing DeepSeek-V4-Flash batches in this sweep are batch 8 and 16; batch 16 improves consistently by about 6-7%, and batch 8 is roughly neutral to slightly faster. Batch 1/2/4 and 32/64 either use the same wave count as the old single-wave path or are intentionally kept on that path, so their small swings should be treated as run-to-run noise unless reproduced with a same-build A/B toggle.

Follow-up if needed: add a same-build `DG_SM90_MEGA_MOE_BLOCK64_FORCE_SINGLE_WAVE` override and run paired A/B sweeps for only `block_m=64` batches to remove cross-run noise.

## 2026-05-15: zero-token cleanup skip v1

### Hypothesis

Small batches activate only a subset of local experts. During cleanup, SM90 still clears per-rank recv-count slots and arrival state for every local expert. Try skipping the per-rank recv-count and arrival cleanup for experts whose `expert_recv_count_sum` low 32 bits are zero, while still clearing the sum counter so the next iteration cannot pass the scheduler wait on stale high 32 bits.

### Change Tested

File: `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`

Inside dispatch cleanup:

- always clear `workspace.get_expert_recv_count_sum_ptr(i)`
- if `num_recv_tokens == 0`, skip:
  - `workspace.get_expert_recv_count_ptr(j, i)` clears
  - `workspace.get_l1_arrival_count_ptr(...)` clears
  - `workspace.get_l2_arrival_mask_ptr(...)` clears

The JIT cache marker in `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp` was bumped for the test.

### Correctness

Command:

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/workspace/DeepEP:$PYTHONPATH \
MASTER_PORT=8761 \
python tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 3 4 --fail-fast
```

Result:

```text
PASSED all 28 scenarios
```

### Performance

Raw log:

- `benchmark_outputs/sm90_zero_token_cleanup_v1_perf.log`

Median fused time across 8 ranks, compared with accepted wave v2 full sweep:

| batch | recv tok med | active exp med | wave v2 us | zero cleanup us | zero/v2 | delta |
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

### Conclusion

Rejected and reverted. Even though correctness passed, the performance profile was unstable and had clear regressions at batch 8 and batch 512. The likely issue is that these cleanup stores are not the dominant fixed cost; removing them changes ordering/cache behavior around the post-combine cleanup path without reliably reducing the critical path.

The source change was reverted after measurement. The raw log is kept for future comparison.
