"""CPU checks of the actual FP4 dispatch-policy function, without CUDA imports."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DispatchGatewayPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            raise unittest.SkipTest("A C++ compiler is required")
        header = (ROOT / "csrc/jit_kernels/impls/sm90_fp8_fp4_mega_moe.hpp").read_text()
        match = re.search(
            r"static int get_sm90_fp4_dispatch_gateway\([^\n]+\) \{.*?^\}",
            header, re.MULTILINE | re.DOTALL,
        )
        assert match is not None, "policy function not found"
        layout = (ROOT / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh").read_text()
        threshold = re.search(r"kGatewayDenseMaxRequestedTokens = (\d+);", layout)
        assert threshold is not None
        cls.threshold = int(threshold.group(1))
        cls.tmp = tempfile.TemporaryDirectory(prefix="fp4-dispatch-policy-")
        cls.addClassCleanup(cls.tmp.cleanup)
        source = Path(cls.tmp.name) / "policy.cpp"
        cls.binary = Path(cls.tmp.name) / "policy"
        source.write_text(
            "#include <cstdlib>\n#include <iostream>\n#include <stdexcept>\n"
            "#include <string>\n"
            "#define DG_HOST_ASSERT(x) do { if (!(x)) throw std::runtime_error(\"invalid\"); } while (0)\n"
            "template<class T> T get_env(const char* name, T value) {\n"
            "  const char* env = std::getenv(name); return env ? static_cast<T>(std::stoi(env)) : value;\n}\n"
            f"namespace layout {{ constexpr int kGatewayDenseMaxRequestedTokens = {cls.threshold}; }}\n"
            + match.group(0)
            + "\nint main(int argc, char** argv) {\n"
            "  if (argc != 3) return 3;\n"
            "  if (std::string(argv[2]) == \"unset\") unsetenv(\"DG_MEGA_MOE_FP4_FORCE_PACKED_DISPATCH\");\n"
            "  else setenv(\"DG_MEGA_MOE_FP4_FORCE_PACKED_DISPATCH\", argv[2], 1);\n"
            "  try { std::cout << get_sm90_fp4_dispatch_gateway(std::stoi(argv[1])); }\n"
            "  catch (const std::exception&) { return 2; }\n"
            "}\n"
        )
        subprocess.run([compiler, "-std=c++17", str(source), "-o", str(cls.binary)], check=True)

    def run_policy(self, capacity, force):
        return subprocess.run(
            [str(self.binary), str(capacity), str(force)],
            check=False, capture_output=True, text=True,
        )

    def test_default_preserves_capacity_boundary(self):
        for force in ("unset", 0):
            for capacity in (1, 16, 64, self.threshold, self.threshold + 1, 384, 1024, 8192):
                with self.subTest(force=force, capacity=capacity):
                    result = self.run_policy(capacity, force)
                    self.assertEqual(result.returncode, 0)
                    self.assertEqual(int(result.stdout), 4 if capacity <= self.threshold else 3)

    def test_override_always_uses_packed(self):
        for capacity in (1, 16, 64, self.threshold, self.threshold + 1, 384, 1024, 8192):
            with self.subTest(capacity=capacity):
                result = self.run_policy(capacity, 1)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "3")

    def test_invalid_overrides_and_capacities_are_rejected(self):
        for capacity, force in ((16, -1), (16, 2), (0, 0), (0, 1), (-1, 0)):
            with self.subTest(capacity=capacity, force=force):
                self.assertEqual(self.run_policy(capacity, force).returncode, 2)

    def test_fp8_jit_is_not_opted_in(self):
        fp8 = (ROOT / "csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp").read_text()
        self.assertNotIn("DG_MEGA_MOE_FP4_FORCE_PACKED_DISPATCH", fp8)
        self.assertNotIn("get_sm90_fp4_dispatch_gateway", fp8)


if __name__ == "__main__":
    unittest.main()
