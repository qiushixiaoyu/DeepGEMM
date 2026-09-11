"""CPU contracts for cleanup; preprocessing/mock readers do not validate CUDA."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "deep_gemm/include/deep_gemm/impls"
FP8 = KERNELS / "sm90_fp8_mega_moe.cuh"
FP4 = KERNELS / "sm90_fp8_fp4_mega_moe.cuh"
CXX = os.environ.get("CXX", "c++")


def preprocess(source, protocol, profile):
    # Include-free preprocessing checks this header's conditional structure,
    # not CUDA types or dependency definitions.
    source = re.sub(r"^\s*#include[^\n]*", "", source, flags=re.M)
    flags = ["-D__CUDA_ARCH__=900", "-DDG_MEGA_MOE_INTERNODE=1",
             "-DDG_MEGA_MOE_NVL_PEERS=8",
             f"-DDG_MEGA_MOE_DISPATCH_GATEWAY_{protocol}=1"]
    if profile != "off":
        flags.append("-DDG_MEGA_MOE_PHASE_PROFILE=1")
    if profile == "silent":
        flags.append("-DDG_MEGA_MOE_PHASE_PROFILE_SILENT=1")
    result = subprocess.run([CXX, "-E", "-P", "-x", "c++", *flags, "-"],
                            input=source, text=True, capture_output=True, check=True)
    return result.stdout


class ProfileCleanupTest(unittest.TestCase):
    def test_retired_fp8_metrics_are_absent_but_live_metrics_remain(self):
        source = FP8.read_text()
        for name in ("profile_expert_", "kExpertProfile", "profile_arrival_start",
                     "kProfileScatterWriteCount", "MEGA_MOE_SCATTER_COUNT_PROFILE",
                     "MEGA_MOE_EXPERT_READY_PROFILE"):
            self.assertNotIn(name, source)
        for suffix in ("publish", "arrival", "wqe", "ready", "ready_mask",
                       "ready_atomic", "ready_fence", "ready_notify", "ready_sync"):
            self.assertNotIn(f"profile_scatter_{suffix}_", source)
            self.assertNotIn(f"scatter_{suffix}_cycles=", source)
        for slot in ("L1", "L2", "AWait", "L1AWait", "L2AWait", "L1Mainloop",
                     "L2Mainloop", "Scatter", "ScatterStaging", "LoaderPoolWait"):
            self.assertRegex(source, rf"phase_profile\[kProfile{slot}\]\s*=")
        self.assertIn("MEGA_MOE_LAST_READY_PROFILE", source)
        self.assertRegex(source, r"atomicMax\(\s*phase_profile \+ kProfileLongestReadyDependency")

    def test_profile_slot_numbers_and_reserved_storage(self):
        source = FP8.read_text()
        enum = re.search(r"enum ProfileSlot : uint32_t \{(.*?)\n    };", source, re.S)[1]
        slots = {name: int(number) for name, number in
                 re.findall(r"kProfile(\w+) = (\d+)", enum)}
        expected = dict(zip(
            ("Metadata", "DispatchBarrier", "DispatchPull", "RemoteRead", "CleanupBarrier",
             "L1", "L2", "Scatter", "CombineBarrier", "CombineReduce", "Total",
             "RemoteReadCount", "L1BlockCount", "L2BlockCount", "StartClock",
             "CombineReadyWait", "ScatterStaging", "CombineWork", "CombineTokenMax",
             "CombineTokenCount", "LongestReadyDependency", "EntryGlobaltimer",
             "CountsSentGlobaltimer", "CountsReadyGlobaltimer", "LoaderPoolWait",
             "AWait", "L1Mainloop", "L2Mainloop", "L1AWait", "L2AWait"),
            (*range(8), *range(9, 15), *range(16, 19), *range(27, 40))))
        self.assertEqual(slots, expected)
        layout = (ROOT / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh").read_text()
        self.assertIn("kSM90MegaMoEProfileMaxSMs = 256;", layout)
        self.assertIn("kSM90MegaMoEProfileSlots = 40;", layout)
        self.assertIn("DG_STATIC_ASSERT(kNumSMs <= kPhaseProfileMaxSMs", source)
        self.assertIn("layout::Data(kPhaseProfileSlots * sizeof(uint64_t), false)", source)

    def test_conditional_paths_and_forwarding_rendezvous(self):
        for path in (FP8, FP4):
            source = path.read_text()
            self.assertNotIn("if constexpr (false)", source)
            self.assertIn("// All dispatch threads rendezvous before gateway forwarding.\n"
                          "        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);",
                          source)
            for protocol in ("PACKED", "DENSE_V3"):
                for profile in ("off", "on", "silent"):
                    with self.subTest(kernel=path.name, protocol=protocol, profile=profile):
                        result = preprocess(source, protocol, profile)
                        self.assertIn("get_gateway_flag_ptr", result)
                        self.assertIn("ptx::st_release_sys", result)
                        self.assertEqual("profile_math_leader" in result, profile != "off")
                        if path == FP8:
                            self.assertEqual('"MEGA_MOE_PHASE_PROFILE rank=' in result,
                                             profile == "on")

    def test_only_fp8_orphan_backoff_is_removed(self):
        fp8, fp4 = FP8.read_text(), FP4.read_text()
        for name in ("DG_MEGA_MOE_SPARSE_STAKEOUT",
                     "DG_MEGA_MOE_ASYNC_PUBLISHER_IDLE_NANOSLEEP", "chain_made_progress"):
            self.assertNotIn(name, fp8)
        self.assertIn("(1ull << 32) |", fp8)
        self.assertIn("workspace.get_expert_send_count_ptr(i), send_value", fp8)
        for name in ("chain_made_progress", "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_INITIAL_NS",
                     "DG_MEGA_MOE_FP4_ASYNC_PUBLISHER_BACKOFF_MAX_NS", "__nanosleep"):
            self.assertIn(name, fp4)

    def test_fp8_fixed_role_topology_and_quotas(self):
        heuristics = (ROOT / "csrc/jit_kernels/heuristics/sm90_mega_moe.hpp").read_text()
        self.assertIn("const int num_dispatch_threads = 64;", heuristics)
        self.assertIn("const int num_non_epilogue_threads = 64;", heuristics)
        source = FP8.read_text()
        self.assertIn("DG_STATIC_ASSERT(kNumDispatchThreads == 64,", source)
        self.assertIn("DG_STATIC_ASSERT(kNumNonEpilogueThreads == 64,", source)
        self.assertIn("kNumEpilogueThreads == 256 or kNumEpilogueThreads == 512", source)
        self.assertNotIn("Idle non-epilogue warps", source)
        self.assertIn("kNumEpilogueThreads == 512 ? 112 : 168", source)
        self.assertIn("kNumEpilogueThreads == 512 ? 32 : 48", source)
        self.assertIn("kNumEpilogueThreads == 512 ? 24 : 40", source)

    def test_fp4_counts_only_nonempty_rdma_batches(self):
        source = FP4.read_text()
        marker = source.index("publisher_metadata_loads +=")
        start = source.rfind("#ifdef DG_MEGA_MOE_PHASE_PROFILE\n", 0, marker)
        start += len("#ifdef DG_MEGA_MOE_PHASE_PROFILE\n")
        body = source[start:source.index("#endif", marker)]
        # Compile the real counter update, not a separately reimplemented
        # predicate. CPU masks model empty, sparse, full and partial row groups.
        code = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
namespace cute { using std::min; }
struct Counts { uint64_t metadata = 0, batches = 0; };
void record(Counts& counts, uint32_t valid_m, uint32_t row_base,
            uint32_t active_rows, bool chain_is_inter, uint32_t lane_idx) {
    constexpr uint32_t kPublishRowsPerBatch = 32;
    auto& publisher_metadata_loads = counts.metadata;
    auto& publisher_rdma_batches = counts.batches;
''' + body + r'''
}
int main() {
    for (uint32_t valid_m : {0u,1u,31u,32u,33u,63u,64u,65u,127u,128u})
    for (bool inter : {false,true}) for (uint32_t lane = 0; lane < 32; ++lane)
    for (uint32_t pattern : {0u,1u,0x80000000u,0xaaaaaaaau,0xffffffffu}) {
        Counts counts;
        uint64_t expected_batches = 0;
        for (uint32_t row_base = 0; row_base < valid_m; row_base += 32) {
            const uint32_t rows = std::min(32u, valid_m - row_base);
            const uint32_t valid_mask = rows == 32 ? 0xffffffffu : ((1u << rows) - 1);
            const uint32_t mask = pattern & valid_mask;
            record(counts, valid_m, row_base, mask, inter, lane);
            expected_batches += inter && lane == 0 && mask != 0;
        }
        assert(counts.batches == expected_batches);
        assert(counts.metadata == (lane == 0 ? valid_m : 0));
    }
    Counts regression;
    record(regression, 64, 0, 0, true, 0);
    record(regression, 64, 32, 0xffffffffu, true, 0);
    assert(regression.metadata == 64 && regression.batches == 1);
}
'''
        with tempfile.TemporaryDirectory(prefix="sm90-batch-counter-") as directory:
            binary = str(Path(directory) / "counter")
            subprocess.run([CXX, "-std=c++17", "-x", "c++", "-", "-o", binary],
                           input=code, text=True, capture_output=True, check=True)
            subprocess.run([binary], check=True)
        # The metric counts non-empty submissions, not calls or fragment WQEs.
        ibgda = (ROOT / "deep_gemm/include/deep_gemm/comm/ibgda.cuh").read_text()
        self.assertIn("if (lane_id == 0 and total_wqes != 0)\n"
                      "        ibgda_submit_requests(qp, base_wqe_idx, total_wqes);", ibgda)

    def test_fp4_unused_profile_sums_removed(self):
        source = FP4.read_text()
        for name in ("sum_l1_cycles", "sum_l2_cycles", "sum_decode_wait_cycles",
                     "sum_a_wait_cycles", "sum_scatter_cycles"):
            self.assertNotIn(name, source)
        for name in ("max_cycles[kProfileL1]", "max_cycles[kProfileL2]",
                     "max_cycles[kProfileDecodeWait]", "max_cycles[kProfileAWait]",
                     "max_cycles[kProfileScatter]", "FP4_MEGA_MOE_PUBLISH_PROFILE"):
            self.assertIn(name, source)

    def test_fp8_fixed_register_requests_compile_without_override(self):
        host = (ROOT / "csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp").read_text()
        device = FP8.read_text()
        self.assertNotIn("kEpilogueRegisterBudget", host + device)
        self.assertNotIn("epilogue_registers", host)
        begin = device.index("    constexpr uint32_t kNumEpilogueRegisters =")
        end = device.index("    constexpr uint32_t kDispatchGridSyncIndex", begin)
        body = device[begin:end]
        code = r'''
#include <cstdint>
#define DG_STATIC_ASSERT static_assert
template<uint32_t kNumEpilogueThreads> struct Quotas {
    static constexpr uint32_t kNumDispatchThreads = 64, kNumNonEpilogueThreads = 64;
''' + body.replace("constexpr uint32_t", "static constexpr uint32_t") + r'''
};
static_assert(Quotas<256>::kNumEpilogueRegisters == 168);
static_assert(Quotas<256>::kNumDispatchRegisters == 48);
static_assert(Quotas<256>::kNumNonEpilogueRegisters == 40);
static_assert(Quotas<512>::kNumEpilogueRegisters == 112);
static_assert(Quotas<512>::kNumDispatchRegisters == 32);
static_assert(Quotas<512>::kNumNonEpilogueRegisters == 24);
'''
        subprocess.run([CXX, "-std=c++17", "-x", "c++", "-fsyntax-only", "-"],
                       input=code, text=True, capture_output=True, check=True)
        self.assertIn("if (config.num_epilogue_threads == 512)", host)
        self.assertIn("DG_HOST_ASSERT(32 * config.num_dispatch_threads +", host)
        self.assertIn("112 * config.num_epilogue_threads <= 64512", host)

    def test_real_profile_reader_compiles_and_reports_live_fields(self):
        source = FP8.read_text()
        enum = re.search(r"enum ProfileSlot : uint32_t \{.*?\n    };", source, re.S)[0]
        body = source.split("#ifndef DG_MEGA_MOE_PHASE_PROFILE_SILENT\n", 1)[1]
        body = body.split("#endif  // DG_MEGA_MOE_PHASE_PROFILE_SILENT", 1)[0]
        # Compile the actual aggregation and printf expressions against a
        # two-row CPU buffer. This catches field/format/argument drift.
        code = r'''
#include <cstdint>
#include <cstdio>
constexpr uint32_t kPhaseProfileSlots = 40, kNumSMs = 2;
unsigned long long rows[kNumSMs][kPhaseProfileSlots];
struct Row {
    unsigned long long* data;
    template<typename T> T* get_base_ptr() { return reinterpret_cast<T*>(data); }
};
struct Buffer { Row get_data_buffer(uint32_t sm) { return {rows[sm]}; } };
int main() {
    for (uint32_t sm = 0; sm < kNumSMs; ++sm)
        for (uint32_t slot = 0; slot < kPhaseProfileSlots; ++slot)
            rows[sm][slot] = sm * 100 + slot;
    // Critical scatter is SM1, but the largest staging value is on SM0.
    // Preserve the existing critical-SM staging metric, not a separate max.
    rows[0][18] = 9999;
    Buffer phase_profile_buffer;
    auto phase_profile = rows[0];
    struct { uint32_t rank_idx; } sym_buffer{0};
    constexpr uint32_t sm_idx = 0, epilogue_thread_idx = 0;
    constexpr uint32_t num_tokens = 16, BLOCK_M = 64, BLOCK_N = 256, WG_BLOCK_N = 128;
    constexpr uint32_t kNumExpertsPerRank = 16;
    constexpr unsigned long long kernel_launch_epoch = 7;
''' + enum + body + "\n}\n"
        with tempfile.TemporaryDirectory(prefix="sm90-profile-reader-") as directory:
            binary = str(Path(directory) / "reader")
            subprocess.run([CXX, "-std=c++17", "-Wformat", "-Werror=format",
                            "-Wno-unknown-pragmas", "-x", "c++", "-", "-o", binary],
                           input=code, text=True, capture_output=True, check=True)
            output = subprocess.check_output([binary], text=True)
        self.assertEqual(len(output.splitlines()), 4)
        main = output.splitlines()[0]
        values = dict(re.findall(r"(\w+)=(\d+)", main))
        self.assertEqual(values["scatter_cycles"], "107")
        self.assertEqual(values["scatter_staging_cycles"], "118")
        self.assertEqual(values["scatter_critical_sm"], "1")
        self.assertEqual(values["remote_reads"], "124")
        self.assertNotIn("scatter_publish_cycles", output)
        self.assertNotIn("scatter_writes", output)
        self.assertIn("MEGA_MOE_LAST_READY_PROFILE", output)


if __name__ == "__main__":
    unittest.main()
