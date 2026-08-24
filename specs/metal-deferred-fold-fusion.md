# Fuse the deferred D512 fold index into its consumer

## Decision and boundary

When a packed fp128 D512/K256 opening carries `DeferredFoldIndex`, Metal will
partition each 256-task tile in threadgroup memory and consume those residue runs in
the same kernel. Retained indices keep the current indexed kernel. Coefficient
packing remains a separate candidate.

The input is the immutable packed lane table plus the transcript-derived D64
challenge table. The device returns the same ordered D512 centered coefficients and
balanced successor digits in eight position chunks. The caller must observe the same
claimed evaluation, proof bytes, transcript, commitment, and verifier result. This is
a prover-local schedule change; no proof field, protocol parameter, or verifier code
changes.

The measured integrated BTreeMap T28 denominator is 50.46 s at 80.10 GiB RSS. Its
opening reports 50,897,879,040 bytes of deferred indices, 2.997 s of index-construction
wall time, 1.104 s of index GPU time, and a 5.539 s command interval with 2.617 s
GPU-active. These aggregate counters include both indices. The fold share is exactly
32,715,571,200 bytes (30.469 GiB); coefficient packing accounts for the remaining
18,182,307,840 bytes. An earlier isolated T28 measurement of this same `u32` fold
index took 1.804 s wall and 1.004 s GPU. The accepted nibble-packed indexed consumer
took 0.722--0.726 s GPU in the standalone T28 opening harness.

## T28 floor

For 30 live columns, 512 blocks per column, and 262,144 positions, each position has
30,720 tasks in 120 tiles. The full grid visits 8,053,063,680 packed lanes. The
current fold index reserves one `u32` record per task slot and eight `u16` counts per
tile. Only live records are written, so the 30.469 GiB allocation is a residency fact,
not 60.938 GiB of guaranteed write/read traffic.

The fused kernel must read the 7.5 GiB lane table and write 0.5 GiB of centered output
plus 0.375 GiB of three-plane digits. The packed challenge table occupies only
3.75 MiB, but table capacity is not its traffic cost: four records are consumed in
parallel by four eight-lane groups, so each live record causes 32 logical challenge
bytes. The nearly all-live T28 control therefore reads about 240 GiB of challenge
words. Compulsory traffic is about 248.4 GiB, or 0.602 s at the measured 412.5 GiB/s.

Partitioning is a serial phase before those challenge reads:

- 31,457,280 tiles execute eight residue ballots in each of eight SIMDgroups;
- 62,914,560 full-threadgroup barriers separate producer and consumer phases; and
- each live selector drives the accepted nibble lookup, packed addition, and final
  SIMD reduction.

The existing index builder processes the same task grid in 1.004 s GPU. Subtracting
its 0.074 s ideal record-and-count write floor gives an observed
partition/synchronization floor near 0.93 s, or 8.7 billion task visits/s. The
accepted nibble consumer takes 0.722--0.726 s. Removing its global record reads saves
at most another 0.074 s, leaving about 0.65 s for challenge traffic and integer work.
The barriers make the two phases serial within each tile, so the calibrated fused
floor is about 1.58 s, not the maximum of the two phase floors. Partition work remains
dominant until its rate exceeds roughly 13 billion task visits/s. Metal does not
expose the register count here, so the 2.00 s promotion bar allows 20% above the
calibrated floor for register and barrier effects.

## Kernel and ownership

One 256-thread group owns one output position. For each tile, its eight producer
SIMDgroups use ballots to write `(challenge_index, source_high)` into the existing
64-by-32 `u32` threadgroup queue, partitioned by producer and low residue. After the
barrier, consumer SIMDgroup `r` walks the eight producer runs for residue `r`.
Four eight-lane source groups reuse the accepted biased-nibble challenge mapping and
accumulate even and odd destination quads in `int4` registers. Each producer run has
at most 32 entries, so debiasing after every run preserves the current no-byte-carry
bound. A second barrier protects the queue before the next tile.

This keeps the proven queue geometry but replaces the old 64-byte dense update with
the accepted 32-byte nibble update and emits the accepted balanced digit layout. It
also avoids the global cross-producer prefix used only to build a persistent index.
Threadgroup storage is 8,320 bytes for records and counts. The two live `int4`
accumulators add six scalar registers relative to the older dense fused kernel;
occupancy loss is therefore an explicit falsifier.

`MetalCommitBackend` consumes the deferred marker once and selects the fused runtime
source. A retained `PackedFp128D512FoldIndex` selects the unchanged indexed source.
Both routes share one canonical streaming dispatch and its validation, chunking,
readback, and metric accounting. Unsupported geometry returns an error under
`RequireMetal`; it does not silently use CPU.

## Adjustment candidates

The selected treatment combines the existing local residue queue with the accepted
nibble consumer. It removes the persistent fold index without changing tile geometry
or challenge representation. Three alternatives remain separate:

- the retained index path remains unchanged through T27 in this experiment; whether
  fusion should replace it there is a later route-selection question;
- chunking a global index lowers peak residency but preserves record traffic, two
  kernels, and command boundaries, so it loses to local fusion for a one-shot T28
  index;
- fusing coefficient packing could remove the other 16.934 GiB index, but combining
  it now would obscure attribution and increase register and command-lifetime risk.

The older local dense consumer is not reconsidered: it performs 64 signed-byte
challenge loads for each live record and does not emit successor digits. No protocol
or challenge-distribution change is needed for this candidate.

## Claim-to-code and ambiguity register

| Requirement | Code and evidence |
|---|---|
| Tile partition matches the current index | fused shader reuses `akita_fp128_d512_build_fold_index` task decoding and the existing direct queue layout |
| Fold and digit bytes are exact | reuse the nibble consumer and `akita_store_indexed_fold_value`; compare with `CpuBackend` |
| Deferred storage disappears | deferred-opening test requires zero fold-index bytes and one accelerated call |
| Retained behavior does not change | existing retained indexed tests plus the Fibonacci T25 sentinel |
| Protocol and soundness do not change | identical proof/transcript/evaluation and verifier acceptance |

The current telemetry does not split the fresh 1.104 s index GPU counter between fold
and coefficient packing. Metal also does not expose register count through this
runtime. The first is handled by exact byte accounting and the historical isolated
fold measurement; the second is decided by the predeclared GPU-time bar. Neither
uncertainty permits changing the evaluator or combining another optimization.

## Falsification and verification

Change the forced-deferred fold test first and observe it fail while the old path
allocates the index. Then require exact centered coefficients, balanced digits, chunk
order, and marker consumption. Run the focused Akita Metal test only, followed by one
verified Fibonacci T25 Jolt sentinel. T25 retains its index and is therefore a
regression guard, not evidence for fused performance; it must remain at or below
6.30 s against the frozen 6.11 s observation.

One BTreeMap T28 treatment is admitted only after those checks. It must report exactly
18,182,307,840 deferred index bytes, at most 2.00 s packed-decompose GPU time, at most
5.00 s aggregate opening command wall, at most 49.96 s complete proving, successful
proof verification, no unexplained fallback, and at most 90 GiB RSS. The prediction
is a 0.5--1.1 s complete-prover saving. Deleting the 1.804 s fold-index preparation
is partly offset by making the consumer about 0.85--1.25 s slower than its current
0.726 s indexed kernel; allocation and command-lifecycle savings provide the upside.

Reject the candidate if exactness fails, the fused GPU interval exceeds 2.00 s, the
index byte count is not exact, or complete proving saves less than 0.5 s. A miss names
either register/barrier cost or host allocation as the broken assumption before the
queue is reranked. Do not tune tile size, chunk count, coefficient packing, or the
protocol inside this candidate.

## Result

Akita `a454c7575` passed exact CPU/Metal parity for centered coefficients, balanced
digits, retained and fused routes, multiple full tiles, a partial tail tile, and
stream order. The integrated Fibonacci T25 guard verified in 5.98 s against its
6.30 s limit.

Two verified BTreeMap T28 runs took 49.29 s and 49.55 s, compared with the 50.46 s
parent. Their opening command intervals were 4.574 s and 4.532 s, peak RSS was
82.03 GiB and 81.93 GiB, and both reported exactly 18,182,307,840 remaining index
bytes. The unchanged T28 opening harness measured 1.813 s of fused packed-fold GPU
time, 87% of the calibrated 1.58 s floor and below the 2.00 s bar. Its transcript
and verifier checks passed.

Promote the candidate. The worst integrated observation saves 0.91 s and raises the
BTreeMap speedup from 3.301x to 3.361x. The remaining 16.934 GiB coefficient index
takes about 0.99 s wall but only 0.10 s GPU to build, so its allocation and command
lifecycle is the next opening candidate; no further fold-queue tuning is justified.
