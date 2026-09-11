"""CPU source-contract tests for fixed SM90 paths; no CUDA validation implied."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def split_arguments(text):
    """Split this JIT emitter's arguments without splitting nested calls."""
    arguments, start, depth = [], 0, 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            arguments.append(text[start:i].strip())
            start = i + 1
    arguments.append(text[start:].strip())
    return arguments


class FixedPathsTest(unittest.TestCase):
    def test_jit_template_arguments_match_device_parameters(self):
        # Check every explicitly emitted parameter, not just the placeholder
        # count: deleting a boolean in the middle must not shift later flags.
        expected = {
            "kNumMaxTokensPerRank": "args.num_max_tokens_per_rank",
            "kHidden": "args.hidden", "kIntermediateHidden": "args.intermediate_hidden",
            "kNumExperts": "args.num_experts", "kNumTopk": "args.num_topk",
            "kNumExpertsPerWave": "args.config.num_experts_per_wave",
            "BLOCK_M": "args.config.block_m", "BLOCK_N": "args.config.block_n",
            "BLOCK_K": "args.config.block_k",
            "kNumMaxPoolTokens": "args.config.num_max_pool_tokens",
            "kNumL1RingTokens": "args.num_l1_ring_tokens",
            "kNumL1SFStorageTokens": "args.num_l1_sf_storage_tokens",
            "kNumL2RingTokens": "args.num_l2_ring_tokens",
            "kNumL2SFStorageTokens": "args.num_l2_sf_storage_tokens",
            "kNumCombineStageTokens": "layout::get_num_sm90_combine_ring_tokens("
                "args.num_ranks, args.num_max_tokens_per_rank, args.num_topk, "
                "args.num_experts / args.num_ranks)",
            "kNumStages": "args.config.num_stages",
            "kNumDispatchThreads": "args.config.num_dispatch_threads",
            "kNumNonEpilogueThreads": "args.config.num_non_epilogue_threads",
            "kNumEpilogueThreads": "args.config.num_epilogue_threads",
            "kNumSMs": "args.launch_args.grid_dim.first", "kNumRanks": "args.num_ranks",
            "kActivationClamp": "to_string(args.activation_clamp)",
            "kFastMath": 'args.fast_math ? "true" : "false"',
        }
        flags = {
            "sm90_fp8_mega_moe": {
                "kReuseAccumAsFinal": "reuse_accum_as_final",
                "kL2ArrivalCounter": "l2_arrival_counter",
                "kL2EpilogueRequiresFullSync": "l2_epilogue_requires_full_sync",
                "kFP8SwapAB": "use_swap_ab",
            },
            "sm90_fp8_fp4_mega_moe": {
                "kUseWideLoadDecode": "use_wide_load_decode",
                "kEarlyBDecode": "use_early_b_decode",
                "kDecodeDoneMBarrier": "use_decode_done_mbarrier",
                "kFP4SSNSplit": "use_ss_nsplit",
                "kFP4SwapAB": "use_swap_ab",
                "kFP4SwapABFastAmax": "use_swap_ab_fast_amax",
            },
        }
        for name, bools in flags.items():
            with self.subTest(kernel=name):
                host = (ROOT / f"csrc/jit_kernels/impls/{name}.hpp").read_text()
                device = (ROOT / f"deep_gemm/include/deep_gemm/impls/{name}.cuh").read_text()
                declaration = re.search(
                    r"template <\s*uint32_t kNumMaxTokensPerRank,.*?>\s*CUTLASS_GLOBAL",
                    device, re.S).group()
                declaration = re.sub(r"//[^\n]*", "", declaration)
                params = re.findall(r"(?:uint32_t|float|bool)\s+(\w+)", declaration)
                emitted = re.search(
                    r'return internode_prefix \+ fmt::format\(R"\((.*?)\)",(.*?)\);\s*}',
                    host, re.S)
                arguments = split_arguments(emitted[2])
                placeholders = re.search(r"_impl<(.*?)>\);", emitted[1], re.S)[1]
                self.assertEqual(placeholders.count("{}"), len(arguments))
                bindings = dict(zip(params, arguments))
                wanted = dict(expected)
                wanted.update({param: f'args.{field} ? "true" : "false"'
                               for param, field in bools.items()})
                if name == "sm90_fp8_mega_moe":
                    wanted.update(kNumPaddedSFPoolTokens="args.config.num_padded_sf_pool_tokens")
                compact = lambda values: {key: re.sub(r"\s+", "", value)
                                          for key, value in values.items()}
                self.assertEqual(compact(bindings), compact(wanted))
                self.assertEqual(len(arguments), len(wanted))

    def test_fp8_has_only_static_phase_traversal(self):
        host = (ROOT / "csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp").read_text()
        device = (ROOT / "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh").read_text()
        self.assertNotIn("split_phase_hot_path", host)
        self.assertNotIn("kSplitPhaseHotPath", device)
        self.assertNotIn("sm90_fp8_mega_moe_for_each_cached_block", device)
        self.assertEqual(device.count("sm90_fp8_mega_moe_for_each_static_block"), 3)
        self.assertIn("const int inter_qp_id = current_expert_idx;", device)

    def test_fp4_preserves_helper_only_decode_synchronization(self):
        host = (ROOT / "csrc/jit_kernels/impls/sm90_fp8_fp4_mega_moe.hpp").read_text()
        api = (ROOT / "csrc/apis/sm90_mega.hpp").read_text()
        device = (ROOT / "deep_gemm/include/deep_gemm/impls/sm90_fp8_fp4_mega_moe.cuh").read_text()
        for retired in ("math_wg_participates", "num_math_wg_decode_warps",
                        "first_fp4_decode_assist_warp", "first_decode_assist_warp",
                        "kMathWGParticipatesInFP4Decode", "kNumMathWGDecodeWarps"):
            self.assertNotIn(retired, host + api + device)
        for source in (host, device):
            self.assertIn("kFirstFP4DecodeAssistWarp = 2;", source)
        self.assertRegex(device, r"kDecodeDoneArrivers =\s*"
                         r"kNumMMANonEpilogueWarps - kFirstFP4DecodeAssistWarp;")
        self.assertIn("kDecodeDoneArrivers / kDecodeReadyGroups", device)
        self.assertIn("kNumFP4DecodeWorkerThreads = kNumFP4DecodeAssistThreads;", device)
        self.assertIn("kNumFP4DecodeAssistThreads + kNumEpilogueThreads;", device)
        self.assertIn("ptx::sync_aligned(kNumFP4DecodeBarrierThreads, kFP4DecodeBarrierIdx);", device)
        self.assertIn("empty_barriers[i]->init(kNumEpilogueWarps);", device)
        self.assertNotIn("math_warp_decodes", device)
        self.assertNotIn("if (non_epilogue_warp_idx >= kFirstFP4DecodeAssistWarp)", device)
        math_wait = device[device.index("// Only non-epilogue helper warps decode."):]
        math_wait = math_wait[:math_wait.index("if (not wg_has_valid_rows)")]
        self.assertEqual(math_wait.count("wait_fp4_decode_done(stage_idx, phase);"), 1)
        self.assertIn("wait_fp4_decode_input_ready(stage_idx, phase);", math_wait)
        self.assertNotIn("dequant_fp4_b_tile", math_wait)

    def test_history_points_to_current_release_guide(self):
        history = (ROOT / "SM90_FP8_MEGAMOE_RDMA_BRANCHES.md").read_text()
        preamble = history.split("## 2026-08-18")[0]
        self.assertIn("历史归档", preamble)
        self.assertIn("docs/SM90_MEGAMOE_RDMA.md", preamble)
        self.assertIn("单机 fallback 已删除", preamble)
        self.assertIn("仅供历史追溯", preamble)

    def test_nvshmem_debug_print_respects_actual_table_size(self):
        device = (ROOT / "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh").read_text()
        helper = re.search(r"template <typename State>\n.*?\n}\n", device, re.S).group()
        self.assertIn("prev_epoch == 0", device)
        self.assertIn("rank_idx % DG_MEGA_MOE_NVL_PEERS == 0", device)
        self.assertNotRegex(device, r"peer_heap_base_remote\[\d+\]")
        # Compile the real diagnostic helper against a bounds-checking table.
        # printf arguments still evaluate, including every table access.
        code = r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#define __device__
#define __forceinline__ inline
template <typename... Args> void discard_print(const char*, Args...) {}
#define printf discard_print
struct Table {
    std::vector<void*> values;
    bool absent;
    mutable unsigned reads = 0;
    bool operator==(std::nullptr_t) const { return absent; }
    void* operator[](int peer) const {
        assert(!absent);
        ++reads;
        return values.at(peer);
    }
};
struct State {
    int npes, node_npes = 8, node_mype = 0;
    bool enable_rail_opt = false;
    void* heap_base = nullptr;
    uint64_t heap_size = 0;
    Table peer_heap_base_remote;
};
''' + helper + r'''
int main() {
    for (int npes : {0, 1, 8, 16, 24, 32, 40, 48, 56, 64})
    for (bool absent : {false, true}) {
        State st{npes, 8, 0, false, nullptr, 0,
                 Table{std::vector<void*>(npes), absent}};
        sm90_fp8_mega_moe_debug_nvshmem_state(st, 0);
        assert(st.peer_heap_base_remote.reads == (absent ? 0u : unsigned(npes)));
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="sm90-debug-table-") as directory:
            binary = str(Path(directory) / "probe")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-x", "c++", "-", "-o", binary],
                           input=code, text=True, check=True)
            subprocess.run([binary], check=True)

    def test_fixed_scatter_path_retains_synchronization(self):
        for name in ("sm90_fp8_mega_moe", "sm90_fp8_fp4_mega_moe"):
            device = (ROOT / f"deep_gemm/include/deep_gemm/impls/{name}.cuh").read_text()
            self.assertNotRegex(device, r"#define\s+__CLION_IDE__")
            for flag in ("kDispatchExpertReady", "kCombineExpertReady", "kCombineFullRow"):
                self.assertNotRegex(device, rf"\b{flag}\b")
            self.assertNotIn("atomicExch(smem_expert_count", device)
            self.assertNotRegex(device, r"smem_expert_count\[[01]\]\s*=")
            scope = "gpu" if name.endswith("fp4_mega_moe") else "sys"
            self.assertTrue(f"ptx::atomic_add_acq_rel_{scope}(arrival_ptr, 1)" in device)
            self.assertIn("ptx::st_release_sys(", device)
            self.assertIn("ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx)", device)
            if name.endswith("fp4_mega_moe"):
                self.assertIn("ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx)", device)

    def test_only_unused_ibgda_entrypoints_removed(self):
        source = (ROOT / "deep_gemm/include/deep_gemm/comm/ibgda.cuh").read_text()
        for name in ("put_inline", "put_nbi_warp", "get_thread", "get_batch_warp"):
            self.assertNotRegex(source, rf"\b{name}\s*(?:<|\()")
        for name in ("put_inline_with_credit", "put_nbi_warp_group", "put_nbi_warp_batch_rows",
                     "get_batch_thread", "wait_until", "quiet",
                     "ibgda_reserve_wqe_slots_with_credit", "ibgda_poll_cq"):
            self.assertRegex(source, rf"\b{name}\s*\(")
        self.assertIn("Copyright (c) NVIDIA Corporation", source)
        self.assertIn("Licensed under the NVSHMEM SLA", source)
        heuristics = (ROOT / "csrc/jit_kernels/heuristics/sm90_mega_moe.hpp").read_text()
        self.assertNotIn("split_sfa_loader_warp", heuristics)

    def test_destructive_debug_switch_is_documented(self):
        readme = (ROOT / "README.md").read_text()
        line = next(line for line in readme.splitlines() if "`DG_COMM_KERNEL_DEBUG`" in line)
        self.assertIn("after each Mega MoE kernel", line)
        self.assertIn("destroys inputs and protocol state", line)
        barrier = (ROOT / "deep_gemm/include/deep_gemm/comm/barrier.cuh").read_text()
        self.assertIn("CUTLASS_DEVICE void nvlink_barrier(", barrier)
        self.assertNotIn("450 us/launch", barrier)


if __name__ == "__main__":
    unittest.main()
