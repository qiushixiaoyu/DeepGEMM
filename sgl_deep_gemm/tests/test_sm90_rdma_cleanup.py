"""CPU source-contract checks; not a substitute for CUDA/accuracy testing."""
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "deep_gemm/include/deep_gemm/impls"
HOSTS = ROOT / "csrc/jit_kernels/impls"


class RetainedOptimizationsTest(unittest.TestCase):
    def test_retired_switches_have_no_runtime_reader(self):
        retired = {
            "FP4": ["FORCE_INTERNODE", "PUBLISH_ROW_MASK", "REUSE_GMMA_DESC", "FORCE_PACKED_DISPATCH",
                    "ASYNC_PUBLISHER_PROGRESS_BLOCK_BUDGET", "ASYNC_PUBLISHER_PROGRESS_YIELD_NS",
                    "ASYNC_PUBLISHER_FEW_PENDING_CHAINS", "ASYNC_PUBLISHER_FEW_PENDING_MAX_NS",
                    "ASYNC_PUBLISHER_LONG_IDLE_THRESHOLD", "ASYNC_PUBLISHER_LONG_IDLE_MAX_NS"],
            "FP8": ["SCALE_FLOAT2", "DEVICE_LTO", "BALANCED_MATH_REGS",
                    "PUBLISH_BACKOFF_MAX_NS", "PUBLISH_BACKOFF_INITIAL_NS",
                    "SWAP_AB_OVERRIDE", "DENSE_TWO_WG", "M64_NARROW_N", "EXPERT_WAVE_CAP",
                    "WEIGHT_SF_EARLY_LOAD"],
        }
        sources = [p for directory in (ROOT / "csrc", ROOT / "deep_gemm/include")
                   for p in directory.rglob("*") if p.suffix in (".hpp", ".cuh", ".cpp")]
        # Historical cleanup records may name retired experiments; current
        # user-facing recipes must not advertise them as usable controls.
        sources += [ROOT / "README.md", ROOT / "sgl_deep_gemm/README.md",
                    ROOT / "docs/SM90_MEGAMOE_RDMA.md"]
        contents = {source: source.read_text() for source in sources}
        for dtype, names in retired.items():
            for name in names:
                key = f"DG_MEGA_MOE_{dtype}_{name}"
                for source, content in contents.items():
                    self.assertNotIn(key, content, f"{key}: {source}")

    def test_buffer_and_fp4_launch_use_real_topology(self):
        source = (ROOT / "csrc/apis/sm90_mega.hpp").read_text()
        start = source.index("get_symm_buffer_size_for_sm90_mega_moe(")
        sizing = source[start:source.index("static void fp8_mega_moe_impl(", start)]
        self.assertIn("const bool internode = num_ranks > 8;", sizing)
        self.assertIn("internode, internode,", sizing)
        self.assertNotIn("get_env", sizing)
        self.assertIn("const bool fp4_internode = num_ranks > 8;", source)

    def test_documented_mega_switches_exist_in_implementation(self):
        implementation = "\n".join(
            p.read_text() for directory in (ROOT / "csrc", ROOT / "deep_gemm", ROOT / "sgl_deep_gemm")
            for p in directory.rglob("*")
            if p.suffix in (".hpp", ".cuh", ".cpp", ".py") and "tests" not in p.parts
        )
        for path in (ROOT / "README.md", ROOT / "docs/SM90_MEGAMOE_RDMA.md"):
            switches = set(re.findall(r"\bDG_MEGA_MOE_[A-Z0-9_]+\b", path.read_text()))
            self.assertTrue(switches)
            for switch in switches:
                self.assertIn(switch, implementation, f"{switch}: {path}")

    def test_reproduction_bash_syntax(self):
        guide = (ROOT / "docs/SM90_MEGAMOE_RDMA.md").read_text()
        blocks = re.findall(r"```bash\n(.*?)\n```", guide, flags=re.S)
        self.assertGreaterEqual(len(blocks), 6)
        for index, block in enumerate(blocks):
            # Parse only: never build, launch a process group, or contact a node.
            result = subprocess.run(["bash", "-n"], input=block, text=True,
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, f"block {index}: {result.stderr}")

    def test_effective_switches_are_retained(self):
        kept = {
            "sm90_fp8_fp4_mega_moe.hpp": [
                "FP4_PAIRED_PRMT", "FP4_PACKED_GMMA_DESC", "FP4_ROW_PARALLEL_QUANT",
                "FP4_N64_DECODE_READY", "FP4_DECODE_FULL_UNROLL", "FP4_SFB_SHARED_LOOKAHEAD",
                "FP4_BALANCED_WG_REGISTERS", "FP4_DECODE_REGISTER_BOOST",
                "FP4_SKIP_MATH_B_INPUT_WAIT", "FP4_EIGHT_MATH_REGISTERS",
                "FP4_ACTIVATION_ROW_TMA", "FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS",
                "FP4_ASYNC_PUBLISHER_BACKOFF_MAX_NS"],
            "sm90_fp8_mega_moe.hpp": [
                "FP8_PACKED_GMMA_DESC", "FP8_ACTIVATION_ROW_TMA", "FP8_ROW_PARALLEL_QUANT",
                "FP8_HYBRID_MAX_ROWS", "FP8_SKIP_INACTIVE_M_WG"],
        }
        for file, names in kept.items():
            source = (HOSTS / file).read_text()
            for name in names:
                self.assertIn("DG_MEGA_MOE_" + name, source)
        heuristics = (ROOT / "csrc/jit_kernels/heuristics/sm90_mega_moe.hpp").read_text()
        self.assertIn("DG_MEGA_MOE_FP8_STREAMING_DENSITY32", heuristics)
        self.assertIn("DG_MEGA_MOE_FP4_EIGHT_HELPERS", heuristics)

    def test_fp4_lto_does_not_enable_fp8_lto(self):
        compiler = (ROOT / "csrc/jit/compiler.hpp").read_text()
        self.assertIn("#define DG_MEGA_MOE_FP4_BALANCED_WG_REGISTERS 1", compiler)
        self.assertNotIn("DG_MEGA_MOE_FP8_DEVICE_LTO", compiler)
        self.assertIn("DG_JIT_NVSHMEM_LTO_ARCHIVE", compiler)

    def test_hybrid_keeps_full_math_entry_rendezvous(self):
        source = (KERNELS / "sm90_fp8_mega_moe.cuh").read_text()
        start = source.index("auto process_math_block =")
        block = source[start:source.index("if constexpr (kSplitPhaseHotPath)", start)]
        barrier = "ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);"
        self.assertLess(block.index(barrier), block.index("scheduler.template get_valid_m<false>()"))
        self.assertIn("not kReuseAccumAsFinal", block)
        self.assertIn("BLOCK_M == 64 and BLOCK_N == 256", block)

    def test_reserved_layout_agrees(self):
        host = (ROOT / "csrc/apis/sm90_mega.hpp").read_text()
        kernel = (KERNELS / "sm90_fp8_fp4_mega_moe.cuh").read_text()
        for source in (host, kernel):
            self.assertRegex(source, r"kPublishReservedWordsPerRank = 4;")
            self.assertIn("combine_publish_reserved_buffer.get_end_ptr()", source)
            self.assertNotIn("get_combine_publish_row_mask_ptr", source)
        self.assertNotIn("kL2ArrivalCounter", kernel)
        # FP8's independent pre-existing protocol is not the removed FP4 experiment.
        self.assertIn("kL2ArrivalCounter", (KERNELS / "sm90_fp8_mega_moe.cuh").read_text())

    def test_both_precisions_keep_packed_descriptor_support(self):
        source = (ROOT / "deep_gemm/include/deep_gemm/mma/sm90.cuh").read_text()
        self.assertIn("defined(DG_MEGA_MOE_FP4_PACKED_GMMA_DESC)", source)
        self.assertIn("defined(DG_MEGA_MOE_FP8_PACKED_GMMA_DESC)", source)


if __name__ == "__main__":
    unittest.main()
