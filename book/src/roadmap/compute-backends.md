# Compute backends (Metal/GPU)

The optional `akita-metal` crate implements prover compute operations on macOS.
It uses the same field representation, setup, and operation traits as
`CpuBackend`. The verifier does not depend on Metal.

Metal accelerates one-hot root commitment, packed coefficient extraction and
opening folds, supported ring-switch rows, and direct digit/relation range
proof work. The host backend drives the transcript in protocol order. CPU code
handles operations outside the accelerated shapes, including recursive
commitment and unsupported opening representations.

## Selecting a backend

Construct `MetalBackend` with `MetalExecutionPolicy::RequireMetal` to reject
missing devices and unsupported shapes at policy-controlled boundaries.
`PreferMetal` permits CPU selection before those operations execute.
Explicit CPU delegations for operations not implemented on Metal remain part
of either policy; `RequireMetal` does not mean the whole proof runs on the GPU.
On non-macOS targets construction returns `UnsupportedPlatform`.

Prepare the backend from the canonical setup through `ComputeBackendSetup`.
`MetalPreparedSetup` retains CPU state and lazily allocated device matrix
prefixes. Setup identity is checked before use. Device buffers do not become
part of the protocol's setup or proof types.

The packed one-hot view is row-major bytes with an active-zero-row bitset and
a column mask. A nonzero byte selects its lane; zero contributes only when
both mask bits select it. The specialized root paths include D512/rank 1
and D128/rank 3. The caller's admitted schedule determines which shape is used;
the backend does not relax SIS bounds or select a different catalog.

## Validation and measurement

Run the Metal crate's CPU differential tests serially on an idle Mac:

```bash
cargo nextest run -p akita-metal --lib --test-threads 1
```

The component benchmarks are documented in
`crates/akita-metal/benches/README.md`. Their synthetic workloads and GPU
timestamps are not complete-prover rates. End-to-end throughput requires
a consumer's verified proof harness, matching schedules, and explicit timing
boundaries. Repeated output equality alone does not establish CPU parity.

The ownership contract is in `docs/compute-backends.md`; the implementation
constraints and remaining validation obligations are in
`specs/akita-compute-backend-metal.md`.
