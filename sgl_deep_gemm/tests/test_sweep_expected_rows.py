"""CPU-only checks of benchmark scheduling hints, without importing CUDA."""
import ast
import pathlib
import unittest


source = pathlib.Path(__file__).with_name("test_mega_moe_perf_sweep.py")
module = ast.parse(source.read_text())
function = next(node for node in module.body
                if isinstance(node, ast.FunctionDef) and node.name == "_deep_ep_expected_rows")
scope = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
expected_rows = scope["_deep_ep_expected_rows"]


class ExpectedRowsTests(unittest.TestCase):
    def test_all_glm52_batches(self):
        for batch in range(1, 257):
            self.assertEqual(expected_rows(batch, 256, 16, 8, 256, "batch"), (batch + 1) // 2)
            self.assertEqual(expected_rows(batch, 256, 16, 8, 256, "capacity"), 128)

    def test_batch_hint_independent_of_capacity(self):
        for capacity in (64, 128, 256, 512):
            self.assertEqual(expected_rows(16, capacity, 16, 8, 256, "batch"), 8)

    def test_rounding_and_zero_work(self):
        self.assertEqual(expected_rows(17, 256, 16, 6, 384, "batch"), 5)
        self.assertEqual(expected_rows(0, 256, 16, 8, 256, "batch"), 1)

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            expected_rows(16, 256, 16, 8, 256, "invalid")


if __name__ == "__main__":
    unittest.main()
