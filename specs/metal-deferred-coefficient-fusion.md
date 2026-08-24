# Fuse the deferred D512 coefficient index into its consumer

## Decision and boundary

When a packed fp128 D512/K256 opening carries
`DeferredCoefficientPackingIndex`, Metal will partition one 32-tile stream
window in threadgroup memory and consume it in the existing cache-local
coefficient-packing schedule. A retained coefficient index continues to use the
current indexed kernel. The accepted fused fold route is unchanged.

The input is the immutable row-major lane table and the 32 MiB combined-weight
table. The output remains 64 canonical fp128 coordinates for every committed
block, in the same order. Claimed evaluation, proof bytes, transcript,
commitment, and verifier behavior do not change. This is a prover-local
schedule change; it adds no proof field or protocol parameter.

The accepted integrated BTreeMap T28 parent measures 49.29 s and 49.55 s at
81.93--82.03 GiB RSS. Its only remaining deferred index is exactly
18,182,307,840 bytes and costs 0.991--0.994 s wall and 0.095--0.101 s GPU. The
fixed opening harness measures 0.784 s for root coefficient packing, including
0.525 s of index preparation, and 4.255 s command wall against 3.212 s
GPU-active for the whole opening. An earlier retained-index T28 run measured
0.447 s for the same 32-tile packing schedule. These observations bound the
component being replaced; they are not additive.

## T28 floor

At T28 there are 512 trace blocks, 30 live columns, two row parities, 262,144
positions, and 1,024 position tiles. The index covers 30,720 streams and
31,457,280 stream/tile layouts:

- 8,053,063,680 `u16` record slots use 16,106,127,360 bytes;
- 1,038,090,240 `u16` offsets use 2,076,180,480 bytes;
- the accepted consumer launches 983,040 stream/window groups and writes
  503,316,480 bytes of partial roots before its flat reduction; and
- final roots use 16,777,216 bytes.

The fused kernel reads the 7.5 GiB lane table once. At full occupancy it issues
128.85 GB of logical 16-byte weight loads, while adjacent stream groups reuse
the same 1 MiB weight window. Counting every logical weight load, the lane
pass, the 32 MiB table, partial write/read, and final output gives 128.48 GiB,
or 0.312 s at the measured 412.5 GiB/s. Perfect inter-group cache reuse lowers
the compulsory DRAM term to 8.48 GiB, or 0.021 s, so execution and cache
throughput rather than raw DRAM capacity set the useful floor.

Partitioning is serial before weight consumption. The current builder performs
the same lane classification and two threadgroup atomic updates for each live
selector in 0.095--0.107 s GPU. The fixed harness leaves 0.259 s after its
index interval for weight construction, indexed consumption, reduction, and
readback; the older retained run gives a conservative 0.447 s bound for that
work. Removing global record traffic but adding local window storage yields a
calibrated fused floor of about 0.40 s. The promotion bar is 0.50 s for the
complete root-packing span, corresponding to 80% floor efficiency.

The selected kernel reserves 30,720 of the device's 32,768 threadgroup bytes:
8,192 lane bytes, 8,192 `u16` records, 1,024 atomic cursors, and 1,024 `u16`
starts. This admits one resident group per core. Each group still supplies
eight SIMDgroups; a loss from reduced latency hiding is the main structural
falsifier and is not hidden inside the traffic estimate.

## Kernel and ownership

One 256-thread group owns one `(trace block, column, parity, 32-tile window)`.
It clears the 32-by-32 bucket cursors, reads and stores up to 8,192 selectors,
and counts each nonzero selector by `(tile, hot >> 3)`. Thirty-two threads then
prefix one tile each and reset the cursors to bucket starts. A second local pass
scatters `(hot & 7, position_in_tile)` into the window record array.

The consumer mapping stays unchanged: eight adjacent lanes own each high
bucket, walk its records for all 32 tiles, and gather
`combined_weight[position][low]`. Four independent `u64` limb sums remain below
`2^45` for one 8,192-position partial. Three SIMD shuffle steps reduce each
eight-lane bucket without the current 8 KiB reduction array or its three
threadgroup barriers. The existing flat kernel combines the window partials
and performs canonical fp128 addition.

`MetalCommitBackend` consumes the deferred marker once and selects a fused
runtime source containing the existing index geometry. A retained
`PackedFp128D512CoefficientPackingIndex` selects the unchanged source. Both use
one canonical dispatch for validation, partial allocation, reduction, readback,
and metrics. Unsupported geometry returns an error under `RequireMetal`; it
does not use CPU silently. The fused path records zero opening-index time and
bytes.

## Alternatives and invariants

A tile-at-a-time local queue needs less threadgroup memory but introduces three
to five barriers for each of 31,457,280 tile instances. The selected whole-window
queue pays four barriers per group instead. A second global index chunk would
retain allocation, record traffic, and another command boundary. Direct output
atomics and output-major gathers were already rejected by exact T28 treatments.

The retained route remains available for indices produced during commitment.
Root-buffer reuse and command batching are separate candidates if the command
wall remains above GPU-active time after this index disappears. No fold,
challenge, opening geometry, or protocol change is part of O2.

## Falsification and verification

Change the forced-deferred coefficient test first and observe it fail while the
old path allocates the index. Exact CPU/Metal checks must cover retained and
fused routes, zero selectors, padded columns, maximal bucket skew, at least two
full 32-tile windows, a partial window, marker consumption, and canonical root
coordinates. The deferred route must report zero index time and bytes.

Then run one verified Fibonacci T25 sentinel. Its retained route must remain at
or below 6.16 s against the accepted 5.98 s observation. One BTreeMap T28
treatment is admitted after those checks and must satisfy all of:

- complete proving at most 49.05 s, a 0.50 s improvement over the slower parent;
- root coefficient packing at most 0.50 s and opening command wall at most 4.20 s;
- exactly zero deferred opening-index bytes and time;
- successful proof verification with no unexplained fallback; and
- peak RSS at most 90 GiB.

The predicted complete-prover saving is 0.5--0.9 s. Repeat once only for an
ambiguous result or parent promotion. Reject the candidate if exactness fails,
the local latency bar fails, or complete proving saves less than 0.5 s. A miss
must identify occupancy, partition work, or allocation latency as the broken
assumption before reranking; do not tune the 32-tile window, fold kernel,
protocol, or evaluator inside this candidate.
