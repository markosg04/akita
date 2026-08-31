# Spec: Metal Compute Backend Track

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-08-19 |
| Status | active |
| Supersedes | The historical CPU cutover record in `archive/2026-Q3/akita-compute-backend-metal-cutover.md` |
| Book-chapter | book/src/roadmap/compute-backends.md |

## Summary

The CPU compute-backend cutover is complete. The remaining work is the Metal
backend track. This specification records only that current work. The detailed
CPU migration history remains in the archived cutover record.

Metal is an optional prover implementation. It must not change the PCS
protocol, verifier behavior, transcript order, schedule selection, proof
serialization, or security sizing. The host and protocol layers remain the
owners of those decisions.

## Scope

The track covers:

- a `crates/akita-metal` implementation with explicit capability reporting;
- safe device, buffer, and pipeline ownership;
- typed preparation from the canonical expanded setup and selected schedule;
- one deterministic dispatch smoke test before production kernels;
- dense ring and NTT kernels followed by field, MLE, and sum-check kernels;
- deterministic CPU and Metal differential tests for each migrated operation;
- a documented Jolt opening adapter after the core backend boundary is stable.

The CPU backend remains the reference implementation. Unsupported hardware must
continue to use the CPU path without compiling or loading Metal-only code.

## Current port campaign

The initial implementation ports the existing Metal backend from Akita
`0e52ebf17e36a2c303756d882ca9eae7faf47b42` onto
`2d1ab310c8edcf1bef8218fe38e8d6acd5977fe7`. The source branch remains an
immutable correctness and performance reference. This is a semantic port onto
the current compute boundary, not a merge of the source branch's prover,
protocol, planner, field, or verifier code.

### Exact boundary

Akita continues to own schedule selection, transcript operations, proof
assembly, validation, and the canonical `jolt-field` representation. Metal owns
only prepared device state and implementations of the existing operation traits,
starting with `RootCommitKernel` and then the opening, tensor, coefficient-packing,
ring-switch, and row-operation boundaries already exposed by `akita-prover`.

The caller must observe the same commitments, evaluation claims, proof bytes,
errors, and verifier result for identical inputs. A qualified Metal operation
must report an error rather than silently execute on the CPU. Unqualified shapes
may select the existing CPU backend before execution.

The packed D512/K256 root input is a row-major byte matrix plus an optional
compact active-row bitset and column mask. A nonzero byte selects that one-hot
row. Byte zero is absent unless both the row bit and column bit are set, in
which case it selects row zero. This preserves Jolt's committed-zero semantics
without widening every selector or scanning a per-row mask buffer on device.
The differential oracle must exercise absent zeroes, committed zeroes, nonzero
selectors, live columns, and padded columns.

### Lower bounds

The port introduces no new kernel algorithm, so its first gate is parity and
reconstruction of the source measurements rather than a new performance claim.
The source campaign measured 412.5 GiB/s of achievable unified-memory copy
bandwidth on the target M4 Max. For the retained packed-root successor, compulsory
traffic was approximately 12.6 GB, giving a favorable traffic floor of
`12.6 GB / 442.9 GB/s = 28.5 ms`. Its retained arithmetic performs approximately
60 billion signed vector scale-adds, so its compute floor is
`60e9 / measured_scale_adds_per_second`; the target-specific issue rate must be
measured before that kernel may be changed. The bottomed-out latency is the larger
of those two values.

The retained first fused opening fold had an approximately 1.58-second modeled
floor and was already classified as near-floor. It is ported unchanged and may
not be retuned until the current stack reproduces its operation boundary and a
new traffic/compute model supersedes that floor. The Jolt PIOP kernels retain
their pre-registered per-member ceilings, but their denominators must be refreshed
after the shared-field and packed-witness cutovers.

### Adjustment candidates

The first implementation uses the current source-typed operation traits and
current packed witness layouts. Adapter work may change borrowed views, prepared
state ownership, capability reporting, and buffer lifetimes. It may not restore
old protocol stages, generated schedules, `akita-field`, or superseded CPU data
layouts. Fusion, layout changes, or arithmetic changes require a separate
target-specific lower bound and falsification bar before implementation.

### Falsification bar

The first slice is rejected unless a deterministic device dispatch passes on
macOS, unsupported targets compile without Metal dependencies, and setup identity
is checked through `ComputeBackendSetup`. Each subsequent operation must pass an
exact differential test against `CpuBackend` before integration. Complete Jolt
sentinels must verify, qualified routes must report `fallback=false`, and the CPU
path must remain byte-identical with Metal disabled.

Performance promotion is separate from structural port acceptance. Once the
current CPU and Metal paths share the same revision, fixtures, schedules, and
timing boundaries, the campaign targets at least 5x complete-prover speedup for
Fibonacci, SHA-2 chain, and BTreeMap at `T = 2^28`, at most 90 GiB peak RSS, no
swap growth, and no more than 3% regression at the retained lower-scale guards.

## Invariants

1. The backend does not sample transcript challenges or choose protocol order.
2. The backend receives typed prepared state and does not expose device storage
   through protocol-facing setup or proof types.
3. The verifier has no dependency on the Metal or prover backend crates.
4. A Metal result is keyed by an existing protocol operation, not a backend
   invented semantic identifier.
5. Unsupported devices return a typed error or use the CPU backend. They do
   not panic or silently change the schedule.
6. Every migrated operation has one backend boundary. Compatibility shims and
   parallel old and new APIs are not introduced.

## Acceptance criteria

- The workspace builds without Metal dependencies on unsupported targets.
- Device discovery and one deterministic dispatch have focused tests.
- Migrated kernels have CPU differential tests for supported field and ring
  profiles.
- Backend setup rejects mismatched setup metadata or schedule artifacts.
- The verifier and serialized proof remain unchanged for a CPU reference run.
- Any performance claim includes the command, target, hardware, and baseline.

## References

- `book/src/roadmap/compute-backends.md`
- `crates/akita-prover/src/compute/`
- `crates/akita-algebra/src/ntt/`
- `crates/akita-prover/src/kernels/`
