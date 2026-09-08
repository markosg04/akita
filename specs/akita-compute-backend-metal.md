# Spec: Metal compute backend

| Field | Value |
|---|---|
| Author(s) | Quang Dao, Markos Georghiades |
| Created | 2026-08-19 |
| Status | active |
| Supersedes | The historical CPU cutover record in `archive/2026-Q3/akita-compute-backend-metal-cutover.md` |
| Book-chapter | book/src/roadmap/compute-backends.md |

## Decision and boundary

`akita-metal` supplies an optional macOS implementation of existing prover
compute operations. Akita's protocol layer continues to own schedules,
security sizing, transcript rules, proof assembly, and verification. Selecting
Metal for an admitted operation must preserve the CPU result; it is not a
protocol mode or a reason to change parameters.

The user-facing operation coverage and fallback rules live in the book chapter.
The CPU backend remains the independent differential oracle. Protocol-facing
setup and proof types must not contain device buffers, queues, or pipelines.

## Input and lifetime invariants

Prepared state is bound to the canonical expanded setup. Matrix prefixes are
cached by their validated shape and retained through owned Metal buffers.
A larger resident prefix may serve a smaller request only when it refers to
the same canonical flattened matrix prefix.

Packed one-hot inputs use row-major bytes plus an active-zero-row bitset and
column mask. Byte zero is absent unless both the row and column bits are set;
then it commits lane zero. Validation must cover absent zeroes, committed
zeroes, nonzero selectors, live columns, and padded columns.

The D128/rank-3 path maps a K256 selector to position
`2 * row + lane / 128` and shift `lane % 128` within its block.
Its signed-radix accumulation must remain exact under the normalization bound
documented in the shader. Changing this arithmetic requires a fresh bound and
CPU differential tests, not only a successful proof.

## Failure and compatibility

Unsupported targets must compile without loading Metal. Device, shape,
allocation, and submission failures return typed errors. Policy-controlled
CPU fallback is selected before execution; a dispatched operation's failure
must not silently restart the protocol on another backend. Explicit CPU
delegations are part of the backend's documented operation coverage.

The backend does not choose a new schedule, reorder transcript absorption,
weaken a verifier check, or change serialization. A consumer that changes its
public-parameter catalog must validate that transition separately, regenerate
its artifacts, and document compatibility. Backend parity is not evidence
for the security of a different catalog.

## Verification obligations

- CPU differential tests cover each accelerated operation and supported shape.
- Prepared setup rejects mismatched setup identity.
- Consumer end-to-end proofs verify with the canonical verifier.
- Unsupported-target builds keep Metal out of the verifier dependency closure.
- Performance claims identify the hardware, source revision, input shape,
  timing boundary, memory use, and whether results are component or full-proof
  measurements. Copy-bandwidth estimates alone do not establish kernel
  optimality.

The implementation's upstream integration must be revalidated when operation
traits or trusted schedule catalogs change. Passing tests against the pinned
consumer revision does not establish compatibility with a newer protocol API.

## References

- `book/src/roadmap/compute-backends.md`
- `docs/compute-backends.md`
- `crates/akita-metal/benches/README.md`
- `crates/akita-prover/src/compute/`
