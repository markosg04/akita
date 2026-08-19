# Metal packed one-hot commitment

`akita-metal` accelerates the inner root commitment for an exact fp128 packed
one-hot source on Apple GPUs. The backend implements Akita's existing compute
traits, returns the canonical `CommitInnerWitness`, and delegates later commit
stages to `CpuBackend`. It does not change proof bytes, transcript behavior, or
verifier logic.

## Supported schedule

The specialized path accepts the Jolt-oriented schedule below. Other packed
shapes either return an error (`RequireMetal`) or select CPU before dispatch
(`PreferMetal`). A submitted Metal command is never silently retried on CPU.

- field: `Prime128OffsetA7F7`
- one-hot width: `K = 256`, with byte zero denoting an absent entry
- inner ring: `D = 512`; A rank: 1; positions per block: `2^19`
- four equal position partials per output block
- logical column capacity 32, with 25, 28, or 32 live columns
- trace lengths `2^25` through `2^28`

The benchmark schedule uses `D = 64` for the outer commitment, `D = 128` for
the evaluation-trace opening commitment, eight outer slices, and three fold
digits. The outer and compression stages remain on CPU and are included in
full-commit timing; the opening stage is outside `commit` and is not timed.

## Data and execution

`PackedOneHotPoly` owns aligned cycle-major bytes. Its physical layout is
`lanes[trace_row][live_column]`; omitted columns up to the logical capacity are
implicit zeroes. The aligned owner lets Metal wrap the lane allocation without
an input copy.

The setup-bound backend caches the exact A-matrix prefix. Each 1,024-thread
threadgroup contains 32 SIMDgroups, and each SIMDgroup owns one
`(column, block)` task. A threadgroup loads four matrix positions (eight trace
rows) into 32 KiB of transposed threadgroup memory. SIMD ballots compact the
nonzero lanes. Two coefficient-band dispatches keep eight fp128 coefficients
live per lane and cover all 512 output coefficients. Four independently written
position partials are reduced by a second device kernel.

Field accumulation is exact modulo `2^128 - 0xa809`. The kernel tracks whole
128-bit wraps while accumulating transposed limbs, then applies the Solinas
correction before writing a partial. Focused tests compare the complete output
against Akita's CPU packed-source implementation at zero, one, lane 255, mixed,
and block-boundary patterns.

## Cost model

For `W = live_columns * blocks_per_column` tasks and a 4 GiB A prefix, the two
coefficient bands read approximately

```text
2 * ceil(W / 32) * 4 GiB
```

from the matrix. Packed lanes are scanned twice. Output size is
`32 * blocks_per_column * 512 * 16` bytes, and device scratch is four times the
output size. This path is therefore dominated by matrix traffic; sparsity
mainly changes selected-lane arithmetic rather than the fixed matrix scan.

## Benchmark

`packed_onehot_commit` is a full-commit evaluator, not an inner-kernel
microbenchmark. It checks exact commitment and hint equality, requires Metal,
asserts zero-copy input and zero CPU inner work, alternates CPU/Metal order, and
reports the point ratio plus a paired-bootstrap 95% lower bound.

```bash
AKITA_PACKED_WORKLOAD=fp128_d512_k256_t25_c25_d25 \
AKITA_PACKED_SAMPLES=15 \
cargo bench -p akita-metal --bench packed_onehot_commit
```

Schedule selection and downstream trace interop are intentionally outside this
crate. A caller may supply these commitment parameters explicitly while those
integration choices are evaluated separately.
