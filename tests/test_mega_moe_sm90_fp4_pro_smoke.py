"""DeepSeek-V4-Pro shape smoke tests for SM90 FP8xFP4 MegaMoE."""

import argparse
import sys
import torch

import test_mega_moe_sm90_fp4 as fp4_test


PRO_SCENARIOS = [
    (
        "Pro.b128_1wg_h7168_ih3072_e384_k6",
        dict(
            num_max_tokens_per_rank=128,
            num_tokens=128,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=384,
            num_topk=6,
            activation_clamp=10.0,
        ),
    ),
    (
        "Pro.b512_2wg_h7168_ih3072_e384_k6",
        dict(
            num_max_tokens_per_rank=512,
            num_tokens=512,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=384,
            num_topk=6,
            activation_clamp=10.0,
        ),
    ),
]


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = fp4_test.init_dist(local_rank, num_local_ranks)

    if fp4_test.get_arch_major() != 9:
        fp4_test.dist_print(
            f"[SKIP] requires SM90; got SM{fp4_test.get_arch_major()}0",
            once_in_node=True,
        )
        fp4_test.dist.destroy_process_group()
        return

    fp4_test.dist_print(
        f"SM90 FP4 Pro smoke plan: {len(PRO_SCENARIOS)} scenarios on {num_ranks} ranks",
        once_in_node=True,
    )

    failures = []
    for name, cfg in PRO_SCENARIOS:
        try:
            fp4_test._run_scenario(name, cfg, rank_idx, num_ranks, group, args.diff_tol)
        except AssertionError as ex:
            fp4_test.dist_print(f"  [{name}] FAIL: {ex}", once_in_node=True)
            failures.append(name)
            if args.fail_fast:
                break

    fp4_test.dist_print("", once_in_node=True)
    if failures:
        fp4_test.dist_print(
            f"FAILED {len(failures)}/{len(PRO_SCENARIOS)} scenarios: {failures}",
            once_in_node=True,
        )
    else:
        fp4_test.dist_print(
            f"PASSED all {len(PRO_SCENARIOS)} scenarios", once_in_node=True
        )

    fp4_test.dist.barrier()
    fp4_test.dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SM90 FP4 MegaMoE Pro smoke")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--diff-tol", type=float, default=0.07)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(
        test, args=(args.num_processes, args), nprocs=args.num_processes
    )
