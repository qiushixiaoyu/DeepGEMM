import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
FP8_PATH = REPO_ROOT / "python/sglang/srt/layers/quantization/fp8.py"
DEEPSEEK_V4_PATH = REPO_ROOT / "python/sglang/srt/models/deepseek_v4.py"


class TestMegaMoeInvariants(unittest.TestCase):
    def _parse(self, path: pathlib.Path):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        return source, tree

    def test_sm90_all_fp8_trigger_condition_is_guarded(self):
        """Ensure non-FP4 mega build path is SM90-only and excludes SM100."""
        source, tree = self._parse(FP8_PATH)

        condition_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_src = ast.get_source_segment(source, node.test) or ""
                if (
                    "not self.is_fp4_expert" in test_src
                    and "envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get()" in test_src
                    and "is_sm90_supported()" in test_src
                    and "not is_sm100_supported()" in test_src
                ):
                    condition_found = True
                    break

        self.assertTrue(
            condition_found,
            "Expected non-FP4 MegaMoE trigger to be guarded by SM90 and not SM100.",
        )

    def test_k_dimension_rule_differs_for_fp8_vs_fp4(self):
        """Ensure K derivation uses K=last_dim for SM90 FP8, K=last_dim*2 otherwise."""
        source, tree = self._parse(DEEPSEEK_V4_PATH)

        target_if = None
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_src = ast.get_source_segment(source, node.test) or ""
                if test_src.strip() == "use_sm90_fp8_mega":
                    target_if = node
                    break

        self.assertIsNotNone(
            target_if,
            "Missing `if use_sm90_fp8_mega` branch for K dimension derivation.",
        )

        def collect_assign_exprs(nodes):
            result = {}
            for n in nodes:
                if isinstance(n, ast.Assign) and len(n.targets) == 1:
                    target = n.targets[0]
                    if isinstance(target, ast.Name):
                        result[target.id] = ast.get_source_segment(source, n.value)
            return result

        body_assigns = collect_assign_exprs(target_if.body)
        orelse_assigns = collect_assign_exprs(target_if.orelse)

        self.assertEqual(
            body_assigns.get("k1"),
            "half_k1",
            "SM90 FP8 path must set k1 = half_k1.",
        )
        self.assertEqual(
            body_assigns.get("k2"),
            "half_k2",
            "SM90 FP8 path must set k2 = half_k2.",
        )
        self.assertEqual(
            orelse_assigns.get("k1"),
            "half_k1 * 2",
            "Non-SM90-FP8 path must set k1 = half_k1 * 2.",
        )
        self.assertEqual(
            orelse_assigns.get("k2"),
            "half_k2 * 2",
            "Non-SM90-FP8 path must set k2 = half_k2 * 2.",
        )


if __name__ == "__main__":
    unittest.main()
