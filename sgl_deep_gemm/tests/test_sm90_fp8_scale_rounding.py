"""Source-contract guard; GPU skew/repeat and performance gates are also needed."""
from pathlib import Path
import re
import unittest


HEADER = (Path(__file__).resolve().parents[2] /
          'deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh')


class FP8ScaleRoundingTest(unittest.TestCase):
    def test_swap_l1_has_uniform_explicit_rounding(self):
        source = HEADER.read_text()
        begin = source.index('const uint32_t accum_base =\n')
        end = source.index('const uint32_t n_swap =', begin)
        promotion = source[begin:end]
        self.assertEqual(promotion.count('__fmaf_rn('), 4)
        self.assertEqual(promotion.count('__fmul_rn('), 4)
        for index, scale, weight in ((0, 0, 'gate'), (2, 0, 'up'),
                                      (1, 1, 'gate'), (3, 1, 'up')):
            accumulator = f'final_accum[accum_base + i * 4 + {index}]'
            expected = (f'{accumulator} = __fmaf_rn('
                        f'__fmul_rn(scale_{scale}, {weight}_sf), '
                        f'swap_accum[i * 4 + {index}], {accumulator});')
            self.assertIn(re.sub(r'\s+', '', expected), re.sub(r'\s+', '', promotion))
        self.assertNotRegex(promotion, r'final_accum\[[^\]]+\]\s*\+=')


if __name__ == '__main__':
    unittest.main()
