"""CPU-only RDMA contract checks; not a substitute for a rebuilt GPU wheel."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RDMAOnlyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ast.parse((ROOT / "sgl_deep_gemm/__init__.py").read_text())
        names = {"_check_sm90_mega_moe_rdma_group", "SM90SymmBuffer",
                 "get_symm_buffer_for_sm90_mega_moe", "fp8_fp4_mega_moe", "fp8_mega_moe"}
        nodes = [node for node in source.body if getattr(node, "name", None) in names]
        assert len(nodes) == len(names)
        # Execute only these real API definitions, without importing torch or
        # the native module. Invalid topology must fail before accessing either.
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        cls.api = {"torch": SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True, get_device_capability=lambda: (9, 0)))}
        exec(compile(module, str(ROOT / "sgl_deep_gemm/__init__.py"), "exec"), cls.api)

    def test_python_topology_boundaries(self):
        for ranks in range(-2, 74):
            group = SimpleNamespace(size=lambda: ranks)
            if 8 < ranks <= 64 and ranks % 8 == 0:
                self.api["_check_sm90_mega_moe_rdma_group"](group)
            else:
                with self.subTest(ranks=ranks), self.assertRaisesRegex(ValueError, "RDMA-only"):
                    self.api["_check_sm90_mega_moe_rdma_group"](group)

    def test_reject_before_allocator_or_native_launch(self):
        for ranks in (0, 1, 2, 4, 8, 9, 15, 17, 65):
            group = SimpleNamespace(size=lambda: ranks)
            buffer = SimpleNamespace(group=group)
            calls = (
                lambda: self.api["SM90SymmBuffer"](group, 256, 384, 8, 6144, 2048),
                lambda: self.api["get_symm_buffer_for_sm90_mega_moe"](group, 256, 256, 8, 6144, 2048),
                lambda: self.api["fp8_fp4_mega_moe"](None, None, None, buffer),
                lambda: self.api["fp8_mega_moe"](None, None, None, buffer),
            )
            for index, call in enumerate(calls):
                with self.subTest(ranks=ranks, entry=index), self.assertRaisesRegex(ValueError, "RDMA-only"):
                    call()

    def test_non_sm90_delegation_unchanged(self):
        cuda = self.api["torch"].cuda
        original = cuda.get_device_capability
        cuda.get_device_capability = lambda: (10, 0)
        sentinel = object()
        self.api["mega"] = SimpleNamespace(fp8_fp4_mega_moe=lambda *a, **k: sentinel)
        try:
            self.assertIs(self.api["fp8_fp4_mega_moe"](None, None, None, None), sentinel)
        finally:
            cuda.get_device_capability = original

    def test_cpp_topology_boundaries(self):
        # Compile the actual guard with only its CUDA-dependent exception
        # include replaced; condition and message are taken from the source.
        source = (ROOT / "csrc/utils/sm90_mega_moe_rdma.hpp").read_text()
        source = source.replace('#pragma once', '').replace('#include "exception.hpp"', '')
        code = '#include <stdexcept>\n#include <cassert>\n#include <string>\n'
        code += '#define DG_HOST_UNREACHABLE(reason) throw std::runtime_error(reason)\n'
        code += source + r'''
int main() {
    for (int ranks = -2; ranks < 74; ++ranks) {
        bool accepted = true;
        try { deep_gemm::check_sm90_mega_moe_rdma_topology(ranks); }
        catch (const std::runtime_error& error) {
            assert(std::string(error.what()).find("RDMA-only") != std::string::npos);
            accepted = false;
        }
        assert(accepted == (ranks > 8 && ranks <= 64 && ranks % 8 == 0));
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="sm90-rdma-only-") as directory:
            binary = str(Path(directory) / "guard")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-x", "c++", "-", "-o", binary],
                           input=code, text=True, check=True)
            subprocess.run([binary], check=True)

    def test_jit_and_device_no_single_node_specialization(self):
        for name in ("sm90_fp8_mega_moe", "sm90_fp8_fp4_mega_moe"):
            host = (ROOT / f"csrc/jit_kernels/impls/{name}.hpp").read_text()
            self.assertIn("check_sm90_mega_moe_rdma_topology(args.num_ranks)", host)
            self.assertIn("check_sm90_mega_moe_rdma_topology(num_ranks)", host)
            self.assertIn("RDMA-only protocol revision 4", host)
            self.assertNotIn("if (args.num_ranks >", host)
            kernel = (ROOT / f"deep_gemm/include/deep_gemm/impls/{name}.cuh").read_text()
            self.assertIn('a single-node kernel is not supported', kernel)
            self.assertIn("kNumRanks > 8", kernel)
            self.assertIn("kNumRanks <= 64 and kNumRanks % 8 == 0", kernel)
            for flag in ("kDispatchExpertReady", "kCombineFullRow", "kCombineExpertReady"):
                self.assertNotIn(f"bool {flag} =", kernel)
            self.assertNotIn("comm::nvlink_barrier<", kernel)
            # Same-node transport is part of the RDMA operator, not fallback.
            self.assertIn("sym_buffer.map(", kernel)
            self.assertIn("tok_is_inter", kernel)
            self.assertIn("comm::ibgda::", kernel)


if __name__ == "__main__":
    unittest.main()
