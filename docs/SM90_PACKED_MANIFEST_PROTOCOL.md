# SM90 MegaMoE RDMA: fixed manifest publication

Both FP4 and FP8 use a fixed-address packed gateway manifest for every top-k.
The token payload remains compact: only its live bytes are transferred. The
unused capacity between payload and manifest is allocated but never sent.

## Why a live-tail manifest is unsafe

The landing slot is reused by epoch parity. If a smaller batch follows a larger
batch, the new live-tail manifest can fall inside old token metadata. The header
may arrive before the rest of its RDMA WRITE. A pair of old 32-bit token entries
can then look like a valid 64-bit `{epoch, count}` value, even when the epoch
matches. Neither waiting on the header nor checking only the count bound makes
this protocol safe.

Captured Kimi evidence: epoch313/local expert13 returned count12537 and later11;
the cached total was wrong by exactly12526. Another run returned21238 then27.
These values were traced to an older B2048 token payload while entering B1024.
This is a dispatch readiness bug, not an FP4 decode or L2 arrival-counter bug.

## Producer / consumer contract

1. Producer prepares offsets, compact live payload and fixed manifest in the
   registered send slot. All dispatch writers rendezvous before publication.
2. A system fence makes these stores visible before RNIC submission.
3. The same RC QP receives two ordered requests: live header/offset/payload,
   followed by the fixed manifest. Request fragmentation retains their order.
4. Consumers poll the aligned 64-bit epoch/count in the fixed manifest directly;
   they do not use a header to find readiness. That region is beyond the maximum
   payload extent, so no previous legal payload can contain a false marker there.
5. Existing parity lifetime and per-slot QP send-completion credits remain in
   force. Do not overwrite registered source memory while RNIC reads are pending.

There is no new ready queue, global quiet, sidecar, CPU proxy, predecode, or
model/batch-specific workaround. The existing workspace size and capacity-based
dense-V3/packed policy are unchanged.

## Required startup environment

Set `NVSHMEM_IB_ENABLE_RELAXED_ORDERING=0` in every rank's launch environment
**before any NVSHMEM initialization or memory registration**. Restart all ranks
if NVSHMEM has already initialized with a different setting. The SM90 host API
rejects an unset/nonzero value for inter-node execution. This guard checks the
declared environment; it cannot inspect or repair an already-registered MR.

The custom WRITE-based readiness protocol requires write-after-write visibility
ordering. Same-QP submission order is not sufficient on a relaxed-order MR.
NVSHMEM3.4.5 defaults relaxed ordering on; this requirement makes the previously
implicit transport assumption explicit. It is independent of, and not proven to
be the cause of, the captured stale-token alias.

References: [RDMA-core registration contract](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3)
and [NVSHMEM transport defaults](https://github.com/NVIDIA/nvshmem/blob/devel/src/modules/transport/common/env_defs.h).

## Regression scope

CPU tests: `test_sm90_packed_manifest.py` checks source contracts;
`test_sm90_combine_ring_alignment.py` compiles the actual layout/getters across
749 shapes and reproduces the observed old-payload alias. These are not RDMA
ordering proofs or replacements for distributed GPU tests.

The artifact `test_logs/20260909_packed_manifest_fix` records the first wheel,
exact source patch, launch commands, numerical gates and the additional FP4 B16
failure found before performance testing. Its `RESULTS.md` distinguishes those
diagnostic runs from valid performance measurements.

Follow-up validation is in `test_logs/20260909_packed_manifest_fix_v2`. Read its
actual completion/results, not this protocol description, to determine which
tests have finished. Changing the relaxed-ordering setting, manifest, and the
scheduler correction below together is not a single-variable performance A/B.

## Defined scheduler shuffle inputs

`MegaMoEScheduler::get_num_tokens` now initializes its lane-local fallback to
zero. Previously the conditional assignment read an indeterminate value in
nonselected lanes. For experts >=32, even the eventually selected lane read the
fallback in an earlier loop iteration. A later shuffle discarding other lanes
does not make those preceding C++ reads defined.

This is a general scheduler correctness correction, not a Kimi-specific branch.
It changes neither valid expert counts nor block assignments by design. The
same-input, uninstrumented Kimi B16 validation failed with the old fallback and
passed with the defined fallback. See `B16_VALIDATION.md` in the revision2
artifact for numerical checks and actual generated-code comparison. Broader
batch/model and performance conclusions require the full regression results.
