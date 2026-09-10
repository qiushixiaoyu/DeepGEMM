"""Source-contract regression; run the actual-layout CPU test and GPU gates too."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / 'deep_gemm/include/deep_gemm'


class PackedManifestTest(unittest.TestCase):
    def test_scheduler_shuffle_fallback_is_defined(self):
        text = (ROOT / 'scheduler/mega_moe.cuh').read_text()
        start = text.index('uint32_t get_num_tokens(')
        end = text.index('// Get pool block offset', start)
        self.assertIn('uint32_t valid_value = 0;', text[start:end])

    def test_ordering_is_an_explicit_startup_contract(self):
        api = ROOT.parents[2] / 'csrc/apis/sm90_mega.hpp'
        text = api.read_text()
        self.assertIn('get_env<std::string>("NVSHMEM_IB_ENABLE_RELAXED_ORDERING", "") != "0"', text)
        self.assertIn('before NVSHMEM initialization; restart all ranks.', text)

    def test_no_live_tail_readiness(self):
        for path in (ROOT / 'layout/mega_moe.cuh', ROOT / 'scheduler/mega_moe.cuh',
                     *(ROOT / 'impls').glob('sm90*mega_moe.cuh')):
            text = path.read_text()
            self.assertNotIn('kCompactGatewayManifest', text)
            self.assertNotIn('get_gateway_packed_compact_manifest', text)

    def test_both_precisions_publish_fixed_manifest_after_live_payload(self):
        for name in ('sm90_fp8_mega_moe.cuh', 'sm90_fp8_fp4_mega_moe.cuh'):
            text = (ROOT / 'impls' / name).read_text()
            start = text.index('// Two ordered WRITEs on the same RC QP:')
            end = text.index('if (lane_idx == 0)', start)
            send = text[start:end]
            self.assertIn('PutRequest requests[2]', send)
            self.assertLess(send.index('get_gateway_packed_payload_offset_bytes'),
                            send.index('get_gateway_packed_manifest_ptr'))
            self.assertIn('requests, peer_rank_idx, kGatewayQpId', send)
            self.assertNotIn('get_gateway_max_packed_entries', send)
            self.assertNotIn('kNumTopk', send)
            # Retain source-buffer lifetime and system visibility protection.
            self.assertIn('__threadfence_system();', text[start-100:start])
            self.assertIn('get_gateway_send_completion_ptr', text[end:end+320])


if __name__ == '__main__':
    unittest.main()
