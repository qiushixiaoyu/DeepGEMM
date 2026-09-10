"""CPU regression test compiled from the actual SM90 workspace layout/getters.

No CUDA runtime or GPU is needed. An optional original layout additionally
checks that the three benchmark shapes retain their downstream buffer offsets.
"""
import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-layout', type=Path)
    args = parser.parse_args()
    include = Path(__file__).resolve().parents[2] / 'deep_gemm/include/deep_gemm'
    layout = (include / 'layout/mega_moe.cuh').read_text().split('\nstruct Data {')[0]
    math = (include / 'common/math.cuh').read_text().split('#ifdef DG_IN_CUDA_COMPILATION')[0]
    assert 'get_combine_ring_control_ptr' in layout and 'struct Data {' not in layout

    def strip_includes(source):
        return re.sub(r'^\s*#(?:include|pragma)[^\n]*', '', source, flags=re.M)

    code = r'''
#include <cassert>
#include <cstdint>
#include <iostream>
#include <new>
#define CUTLASS_HOST_DEVICE
#define CUTLASS_DEVICE
#define DG_UNIFIED_ASSERT(x) assert(x)
#define DG_STATIC_ASSERT(x, ...) static_assert(x, __VA_ARGS__)
'''
    code += strip_includes(math) + '\n}\n'
    code += strip_includes(layout) + '\n}\n'
    if args.baseline_layout:
        old = args.baseline_layout.read_text().split('\nstruct Data {')[0]
        old = old.replace('namespace deep_gemm::layout {', 'namespace deep_gemm::old_layout {')
        code += '#define CHECK_BASELINE\n' + strip_includes(old) + '\n}\n'
    code += r'''
int main() {
  using namespace deep_gemm;
  static_assert(alignof(layout::SM90CombineRingControl)==16);
  static_assert(alignof(layout::SM90CombineRingSegment)==16);
  static_assert(sizeof(layout::SM90CombineRingControl)%16==0);
  static_assert(sizeof(layout::SM90CombineRingSegment)%16==0);
  unsigned cases=0, old_bad=0, unchanged_benchmarks=0;
  for (uint32_t ranks : {8u,16u,32u})
  for (uint32_t local : {1u,2u,3u,4u,8u,16u,24u,56u,64u})
  for (uint32_t topk : {1u,6u,8u,16u})
  for (uint32_t requested : {1u,256u,257u,1024u,1025u,4096u,8192u}) {
    if (topk>ranks*local) continue;
    const auto capacity=math::align(requested,384u);
    layout::SM90Workspace w(nullptr,ranks,ranks*local,capacity,topk);
    const uint64_t bytes=w.get_num_bytes();
    void* storage=::operator new(bytes,std::align_val_t(16));
    w.base=storage;
    const auto start=reinterpret_cast<uintptr_t>(storage);
    const auto tail=reinterpret_cast<uintptr_t>(w.get_token_src_metadata_ptr(w.num_max_pool_tokens));
    const auto control=reinterpret_cast<uintptr_t>(w.get_combine_ring_control_ptr());
    const auto segments=reinterpret_cast<uintptr_t>(w.get_combine_ring_segment_ptr());
    const auto queue=reinterpret_cast<uintptr_t>(w.get_combine_ring_queue_ptr());
    const auto completion=reinterpret_cast<uintptr_t>(w.get_combine_ring_completion_target_ptr());
    const auto gateway=reinterpret_cast<uintptr_t>(w.get_gateway_entry_base_ptr());
    const auto count_end=reinterpret_cast<uintptr_t>(w.get_dispatch_epoch_count_ptr(2));
    assert(control%16==0 && segments%16==0);
    assert(control>=tail && control-tail<16);
    assert(segments==control+sizeof(layout::SM90CombineRingControl));
    assert(queue==segments+local*sizeof(layout::SM90CombineRingSegment));
    assert(completion%8==0 && completion>=queue+local*sizeof(uint32_t));
    assert(gateway%16==0 && gateway>=completion+local*ranks*sizeof(uint64_t));
    assert(math::align<uintptr_t>(count_end-start,16)==bytes);
    assert(reinterpret_cast<uintptr_t>(w.get_end_ptr())==start+bytes);
    for (uint32_t e=0;e<local;++e)
      assert(reinterpret_cast<uintptr_t>(w.get_combine_ring_segment_ptr(e))%16==0);
    if (ranks>8) for (uint32_t n=0;n<ranks/8-1;++n) for (uint32_t epoch=0;epoch<2;++epoch) {
      assert(reinterpret_cast<uintptr_t>(w.get_gateway_packed_send_slot_ptr(n,epoch))%16==0);
      assert(reinterpret_cast<uintptr_t>(w.get_gateway_packed_landing_slot_ptr(n,epoch))%16==0);
      for (bool landing : {false,true}) {
        const auto slot=reinterpret_cast<uintptr_t>(landing
            ? w.get_gateway_packed_landing_slot_ptr(n,epoch)
            : w.get_gateway_packed_send_slot_ptr(n,epoch));
        const auto payload_end=reinterpret_cast<uintptr_t>(
            w.get_gateway_packed_payload_ptr(landing,n,epoch))
            + w.get_gateway_max_packed_entries()*sizeof(uint32_t);
        const auto manifest=reinterpret_cast<uintptr_t>(
            w.get_gateway_packed_manifest_ptr(landing,n,epoch));
        const auto manifest_end=reinterpret_cast<uintptr_t>(
            w.get_gateway_packed_manifest_ptr(landing,n,epoch,w.get_gateway_num_cells()));
        assert(manifest%16==0 && manifest>=payload_end && manifest-payload_end<16);
        assert(manifest_end<=slot+w.get_gateway_packed_slot_bytes());
        assert(slot+w.get_gateway_packed_slot_bytes()<=start+bytes);
        // Decreasing live payload size must not move the readiness location.
        for (uint64_t live : {0ull,1ull,static_cast<unsigned long long>(w.get_gateway_max_packed_entries())})
          assert(slot+w.get_gateway_packed_payload_offset_bytes()+live*4<=manifest);
      }
    }
    if (ranks==16 && local==56 && topk==16 && requested==8192) {
      // Reproduce the observed Kimi 2048->1024 alias without a GPU/RNIC.
      // An early new header leaves the old payload at the compact address.
      auto old_words=w.get_gateway_packed_payload_ptr(true,0,1,8106);
      old_words[0]=12537; old_words[1]=313;
      const uint64_t legacy_offset=math::align<uint64_t>(
          w.get_gateway_packed_payload_offset_bytes()+8079*4,16)+13*8;
      assert(reinterpret_cast<uintptr_t>(old_words)==
          reinterpret_cast<uintptr_t>(w.get_gateway_packed_landing_slot_ptr(0,1))+legacy_offset);
      auto fixed=w.get_gateway_packed_manifest_ptr(true,0,1,13);
      *fixed=(311ull<<32)|25;
      w.get_gateway_packed_header_ptr(true,0,1)->epoch=313;
      assert(old_words[1]==313 && old_words[0]!=11); // False legacy readiness.
      assert((*fixed>>32)!=313);                  // Fixed slot still waits.
      *fixed=(313ull<<32)|11;                    // Trailing manifest arrives.
      assert((*fixed>>32)==313 && static_cast<uint32_t>(*fixed)==11);
    }
#ifdef CHECK_BASELINE
    old_layout::SM90Workspace old(storage,ranks,ranks*local,capacity,topk);
    old_bad += reinterpret_cast<uintptr_t>(old.get_combine_ring_control_ptr())%16!=0;
    assert(bytes>=old.get_num_bytes() && bytes-old.get_num_bytes()<=16);
    const bool benchmark=ranks==16 && ((local==24 && topk==6) || (local==16 && topk==8) || (local==56 && topk==16));
    if (benchmark) {
      assert(bytes==old.get_num_bytes());
      assert(w.get_gateway_entry_base_ptr()==old.get_gateway_entry_base_ptr());
      assert(w.get_end_ptr()==old.get_end_ptr());
      ++unchanged_benchmarks;
    }
#endif
    ::operator delete(storage,std::align_val_t(16));
    ++cases;
  }
  std::cout << "RING_LAYOUT_PASS cases=" << cases << " old_misaligned=" << old_bad
            << " unchanged_benchmark_layouts=" << unchanged_benchmarks << '\n';
}
'''
    with tempfile.TemporaryDirectory(prefix='sm90-ring-layout-') as temp:
        binary = str(Path(temp) / 'probe')
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O0', '-x', 'c++', '-', '-o', binary],
                       input=code, text=True, check=True)
        subprocess.run([binary], check=True)


if __name__ == '__main__':
    main()
