# SM90 MegaMoE NCU profiling: DeepSeek-V4-Flash config

Config:

- `num_processes=8`
- `hidden=4096`
- `intermediate_hidden=2048`
- `num_experts=256`
- `num_topk=6`
- batches: `1 2 4 8 16 32`
- kernel filter: `sm90_fp8_mega_moe_impl`

Report files:

- batch 1/2/4/8/16: `benchmark_outputs/ncu_sm90_v4flash_valid_b*/mega-moe-sm90.{0..7}.ncu-rep`
- batch 32: `benchmark_outputs/ncu_sm90_v4flash_light_b32/mega-moe-sm90.{0..7}.ncu-rep`
- per-rank CSV: `benchmark_outputs/ncu_sm90_v4flash_batches_1_32_metrics.csv`
- aggregate CSV: `benchmark_outputs/ncu_sm90_v4flash_batches_1_32_summary.csv`

Note: full-section application replay for batch 32 hung in replay, so batch 32 was rerun with `SpeedOfLight + LaunchStats`. The table below uses median across 8 ranks to reduce replay/routing outliers. NCU replay time is not the production kernel time; use the benchmark table for real timing.

## Benchmark result

From `bench_mega_moe_sm90.py` rank0 output:

| batch | recv tokens | active experts | time (us) | TFLOPS | HBM (GB/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 7 | 193.6 | 1.8 | 911 |
| 2 | 12 | 11 | 303.6 | 2.0 | 912 |
| 4 | 27 | 15 | 415.8 | 3.3 | 909 |
| 8 | 34 | 21 | 498.5 | 3.4 | 1061 |
| 16 | 98 | 30 | 573.8 | 8.6 | 1319 |
| 32 | 191 | 32 | 587.4 | 16.4 | 1376 |

## NCU median metrics

| batch | NCU time med (ms) | SOL comp/mem % | SM SOL % | DRAM SOL % | L2 SOL % | DRAM GB/s | active warps/sched | eligible warps/sched | tensor pipe % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.350 | 3.021 | 2.451 | 0.164 | 0.858 | 6.6 | 2.498 | 0.014 | 0.064 |
| 2 | 13.502 | 3.456 | 2.414 | 0.612 | 1.821 | 24.6 | 2.496 | 0.015 | 0.242 |
| 4 | 36.762 | 1.511 | 2.276 | 0.352 | 1.432 | 12.5 | 2.495 | 0.017 | 0.139 |
| 8 | 15.533 | 7.870 | 5.514 | 2.807 | 4.106 | 45.6 | 2.498 | 0.019 | 1.119 |
| 16 | 19.811 | 6.435 | 4.144 | 2.010 | 4.053 | 44.1 | 2.499 | 0.017 | 0.795 |
| 32 | 20.897 | 3.218 | 3.276 | 1.139 | 1.501 | n/a | n/a | n/a | 0.451 |

Launch shape is identical for all sampled batches:

| grid CTAs | block threads | regs/thread | dynamic shared memory | waves/SM | replay passes |
|---:|---:|---:|---:|---:|---:|
| 78 | 384 | 168 | 209.616 KiB | 1.0 | 10 |

## Readout

- The kernel is fixed at one CTA per SM (`78` CTAs on this H20), with high dynamic shared memory (`~209.6 KiB`) and `168` registers/thread. Occupancy is therefore structurally low.
- For batch 1-16, NCU reports only about `2.5` active warps per scheduler and near-zero eligible warps per scheduler (`~0.014-0.019` median). That points to scheduler starvation / waiting behavior rather than raw tensor-core saturation.
- DRAM and L2 SOL are low in NCU median data, so the sampled kernel is not obviously HBM-bandwidth-bound in these small batches. The benchmark HBM GB/s estimate rises with batch because fixed overhead is amortized over more useful work.
- The real benchmark time grows from `193.6 us` to `587.4 us`, but useful work grows much faster; TFLOPS improves from `1.8` to `16.4`. That matches the NCU picture: small-batch performance is dominated by fixed launch/synchronization/routing overhead and low available work per rank.
- Batch 16 to 32 has almost flat latency (`573.8 us -> 587.4 us`) while rank0 recv tokens nearly double (`98 -> 191`), so the kernel is starting to amortize fixed overhead well by batch 32.
