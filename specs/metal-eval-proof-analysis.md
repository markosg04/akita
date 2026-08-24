# Metal evaluation-proof analysis

## Decision boundary

The optimization target is the complete Jolt adapter call to
`AkitaNativeBatching::prove_batch`. Validation, statement/transcript binding,
Akita proving, every command submission and synchronization, host reconstruction,
proof serialization, and transcript absorption are timed. Setup and the existing
commit are outside the timer. Any new preparation shifted before the call is
reported and charged in the integrated Jolt result.

The accepted representation is Jolt's existing row-major byte source with one
K256 selector per live semantic column, packed by Akita into 32 consecutive
`K*T` column segments. It is not transposed or materialized as fp128 values.
The root proof reverses the Jolt point once at the adapter boundary, as today.

## Fixed cases and oracle

The evaluator has three cases: `T=2^25` with 29 and 30 live columns, and
`T=2^28` with 30 live columns and 253,779,321 populated rows. The physical row
count remains `T`; padding rows and the two unused column slots are zero.

For live row `r` and semantic column `c`, the fixture selects

```text
1 + (((r mod 128) + 17*c) mod 128).
```

The selector is never the zero sentinel and varies across rows and columns. Its
MLE is computed independently from the column, row, and K256 equality factors.
For a partial trace, a binary-prefix recurrence computes the equality mass of
the populated high-row prefix separately for each low-seven-bit residue. Oracle
cost is `O(columns * 128 * log(T))`, independent of physical trace size.

## Retained CPU decomposition

The retained `T=2^28` profile is a provisional anchor, not a frozen evaluator
control:

| Phase | CPU wall time |
|---|---:|
| Root trace decompose/fold | 5.15 s |
| Root stage-2 sumcheck | 4.75 s |
| Root stage-1 sumcheck | 3.06 s |
| Ring-switch witness build | 1.76 s |
| Next-witness commitment | 1.45 s |
| Root coefficient packing | 1.29 s |
| Root NTT preparation | 1.32 s |
| Remaining work | 3.12 s |
| Complete call | 21.9 s |

The 5x target is at most 4.38 s against that provisional 21.9-second call. The
older 20.8-second CPU measurement implies a stricter 4.16-second target; the
frozen evaluator control will decide the actual threshold.

Optimizing only the 5.15-second root fold has an Amdahl ceiling of 1.30x. Even
eliminating root fold, coefficient packing, and root NTT entirely caps speedup at
2.15x. Root fold plus both root sumchecks must move, and the ring-switch/commit
pair must either move or overlap substantially.

## Traffic floors

This machine is a 16-core, 40-GPU-core M4 Max with 128 GiB unified memory and a
546 GB/s advertised memory-bandwidth ceiling. The model below uses that ceiling
only as a hard lower bound; candidate predictions use measured command counters.

At max scale the packed selector source contains

```text
2^28 rows * 30 live columns * 1 byte = 8.05 GB (7.50 GiB).
```

A compulsory source pass therefore has a 14.7 ms ideal DRAM floor. A design
that rereads the source once for root evaluation/fold, once for decompose/fold,
and once for coefficient packing has a 24.2 GB input floor, or 44.2 ms at peak,
before fp128 arithmetic, challenge tables, output traffic, and command costs.
The source is small enough relative to the 90 GiB campaign limit to retain as a
shared Metal buffer; expanding it to fp128 is not (128.8 GB before intermediates).

The current root decompose schedule reports 8,192 challenge blocks, 524,288
positions, D=512, and a rejected 4.03 GB dense rotation table. Materializing that
table is neither required nor attractive. The first Metal kernel should generate
or tile rotations in threadgroup/local state and consume each lane byte directly.
Its minimum useful output is the folded witness, not a dense root polynomial.

For a halving sumcheck table with `N` fp128 elements, a separate read, message
scan, and fold/write per round moves at least approximately `4N*16` bytes over
all rounds. Fusing message production with the in-place fold lowers this toward
`3N*16`; keeping the shrinking table resident avoids host traffic entirely.
Exact `N`, lane count, and arithmetic intensity will be emitted by the evaluator
before a sumcheck kernel is accepted.

## Initial complete-call budget

The first credible target budget for `T=2^28` is:

| Component | Target |
|---|---:|
| Root sparse folds and coefficient packing | 0.65 s |
| Root stage-1 and stage-2 sumchecks | 1.15 s |
| Ring switch and next-witness commitment | 0.70 s |
| NTT preparation | 0.35 s |
| Later recursive folds | 0.55 s |
| Host serial, transcript, proof assembly, sync | 0.30 s |
| Contingency | 0.45 s |
| Complete call | 4.15 s |

This is a coverage budget, not a claim of achieved performance. It requires a
device-resident root pipeline and large-round state. Isolated Metal kernels that
return full intermediates to the host after each phase do not satisfy the model.

## Candidate order and falsifiers

1. Route one required Metal backend through the existing proof stack. Keep CPU
   behavior byte-identical and make every qualified fallback observable. Reject
   if routing or setup changes proof bytes.
2. Fuse root evaluate/fold and decompose/fold over the packed byte source. Reject
   if the source is expanded, scanned more than the declared count, or the full
   call projection remains above the 5x threshold.
3. Move root coefficient packing into the same resident-source pipeline. Reject
   if output/readback traffic dominates or root source residency exceeds budget.
4. Move large stage-1 and stage-2 tables to Metal with fused message-and-fold
   rounds and a fixed CPU tail. Reject if per-round synchronization leaves the
   projected pair above 1.15 seconds.
5. Move or fuse ring-switch witness construction, its next commitment, and NTT
   preparation. Reject if host materialization preserves the current 4.5-second
   combined floor.

Each iteration states one predicted complete-call result and one falsifying
counter observation before it runs. Routine evidence is one focused parity test
and one `T=2^25`, 30-column single shot. Max scale is reserved for architectural
milestones and final validation.

## Stage-2 resident candidate

The current accepted `T=2^25`, 30-column treatment spends 3.044 s in Stage 2
and 8.619 s in the complete opening. Stage 2 is the next denominator. Metal
will own the compact signed-byte witness, two alternating fp128 suffix tables,
and the per-round partial reductions. Akita retains the exact input claim,
structured coefficient/lane factors, sparse additional terms, transcript
absorption and challenge sampling, and the final oracle check. Each round returns
only `[c0, c2, c3]` for the canonical compressed cubic and, at the end, one
folded witness value. No protocol message or variable order changes.

For one pair, write `w(t)=w0+t*dw`, `p(t)=p0+t*dp`, and the current equality
factor as `e*(l0+t*dl)`. The dense contribution returned by the kernel is

```text
c0 = e*l0*w0*(w0+1) + w0*p0
c2 = e*(l0*dw^2 + dl*dw*(2*w0+1)) + dw*dp
c3 = e*dl*dw^2
```

where `p` is the sum of the ordinary rank-one relation weight and Akita's
structured linear trace weight. A sparse kernel adds the existing compression
and restricted-binary terms with the same coefficient formulas used by the CPU.

Let `N` be the padded byte-witness domain. Exact schedule replay gives `2^30`
at T25 and `2^31` at T28; the latter is not `2^33` because the T28 catalog row
uses a different root fold geometry.
Three compact prefix rounds reread at most `3N` bytes. The materializing pass
reads another `N` bytes and writes `2N` bytes measured in original-byte units;
the geometric fp128 suffix contributes less than `6N` further bytes. Thus the
witness traffic ceiling is about `12N` bytes: 12.9 GB at T25 and 25.8 GB at T28,
plus equality/relation streams. At the advertised 546 GB/s this is a
23.6 ms / 47.2 ms traffic floor. That is only a hard floor until a dedicated
copy result is retained; the observed 0.609 s Stage-1 GPU interval is the useful
machine-calibrated comparison because it scans the same domain with a similar
count of fp128 products. Stage 2 is expected to be compute-bound by roughly
10--14 fp128 products per pair across a geometric total of fewer than `N` pairs,
with structured-support density deciding the upper end.

The session allocates the live compact witness plus fp128 tables of `N/8` and
`N/16` elements. Exact root live lengths are 839,195,008 bytes at T25 and
1,344,379,328 bytes at T28. Resident storage is therefore about 4.06 GB and
7.79 GB respectively, before relation tables and partials. This fits the 90 GiB
campaign guard with substantially more margin than the original `4N` estimate.

Pre-registered prediction: Stage 2 should fall below 0.95 s and the complete
T25 opening below 6.6 s with exact proof bytes. Reject or redesign if the focused
parity test differs, any Stage-2 fallback occurs, resident allocation departs
materially from the `4N` model, or the one-shot Stage-2 time is at least 1.5 s.
The candidate is an intermediate bottleneck milestone; even its predicted result
does not yet satisfy the 2.209 s complete-opening target.

The first treatment produced exact proof bytes, a matching transcript, and a
passing verifier, but measured 1.651 s for Stage 2 and 7.238 s for the complete
T25 opening. It therefore failed the pre-registered 1.5-second Stage-2
falsifier. It remains the architectural parent because it removed 1.39 seconds
from Stage 2 without changing the protocol. A follow-up that retained factored
linear sources measured 1.615 s and did not materially change upload time or
allocation, so that mechanism was rejected and removed.

## Stage-2 resident lane-weight candidate

Exact schedule replay identifies the dominant repeated relation stream. At
T25 the root uses 64 coefficient coordinates (`6` low variables), a padded
lane capacity of `2^24`, and 13,112,422 live lanes. The same `2^24`-element
fp128 lane-weight table is sent for rounds 0 through 6; subsequent lane rounds
send its geometric halves. This is exactly

```text
(7 * 2^24 + (2^24 - 1)) * 16 = 2,147,483,632 bytes.
```

At T28 the lane capacity is `2^25`, so the corresponding traffic is just under
4 GiB. In contrast, the negative-binary support contains only 28,672 T25 and
53,248 T28 coordinates. Sparse additional terms are therefore not the first
remaining bandwidth target.

The next candidate uploads the initial lane weights once, retains two Metal
buffers of `C` and `C/2` fp128 elements, reuses the first table during the six
coefficient folds, and folds it on the GPU when lane challenges begin. This
requires 384 MiB at T25 and 768 MiB at T28. It removes repeated host field-to-
limb conversion and at least 1.625 GiB / 3.25 GiB of requested buffer storage,
while adding one geometric GPU fold totaling fewer than `C` field elements.
The transcript and sumcheck messages remain unchanged.

Pre-registered prediction: exact parity remains, T25 cumulative allocation
falls from 12.10 GB to at most 10.50 GB, Stage 2 reaches at most 1.40 s, and the
complete opening reaches at most 7.0 s. Reject the mechanism if allocation does
not fall by at least 1.5 GiB or Stage 2 remains at least 1.50 s in the single
T25 treatment.

Measured result: allocation fell to 10.295 GB, upload time fell from 231.0 ms
to 129.4 ms, Stage 2 fell to 1.452 s, and the complete opening fell to 7.035 s.
Proof bytes, transcript, evaluation, and verifier all match the frozen CPU
control. The mechanism clears both falsifiers and is retained. It misses the
1.40-second Stage-2 prediction, so the remaining Stage-2 GPU arithmetic is not
treated as solved.

## Centered-only root fold witness

Schedule replay exposed a backend-neutral representation error after the root
fold. The Metal kernel returns 268,435,456 centered `i32` coefficients at T25
(1 GiB), but `DecomposeFoldWitness` immediately expanded and retained the same
values as 268,435,456 `Fp128` elements (4 GiB). Production ring switching reads
only the centered coefficients and extrema; the field-valued copy was dead
outside test and aggregation helpers. The treatment removes that redundant
field storage throughout Akita and constructs the large Metal output directly
in its destination `Vec<i32>` when the allocator provides page-aligned storage,
avoiding a second 1 GiB readback copy. The protocol, transcript, and witness
ordering are unchanged.

Pre-registered prediction: focused parity and verifier checks remain exact,
peak RSS falls by at least 3 GiB from the retained 25.60 GB treatment, aggregate
readback falls materially, and complete T25 opening reaches at most 6.7 s.
Reject the wall-time mechanism if the complete call improves by less than
0.20 s; reject the memory mechanism if peak RSS falls by less than 3 GiB. The
centered-only representation may remain even if either performance prediction
fails because it removes semantically dead storage, but a failed mechanism will
not count as progress toward the 5x target.

Measured result: proof bytes, transcript, claimed evaluation, and verifier all
match. Metal readback fell from 95.45 ms to 0.061 ms and ring-switch witness
construction fell by 97.71 ms, but the complete opening improved by only
158.51 ms, from 7.035 s to 6.877 s. Peak RSS was effectively unchanged at
25.58 GB. Both pre-registered performance mechanisms are rejected: the direct
destination write is real but too small, while the removed field allocation was
apparently not resident at the measured peak or was masked by NTT caches. The
centered-only representation remains as a simpler API with exact parity, but it
does not count as a target milestone.

## Preweighted coefficient packing

The retained T25 result spends 441 ms in root coefficient packing and 552 ms
in recursive coefficient packing. Both compute

```text
sum_position source[position, subring, low]
             * position_weight[position]
             * packing_weight[low].
```

The public weights repeat for every block and source. The treatment computes
`position_weight * packing_weight` once for each block-local `(position, low)`
coordinate, then uses a Metal reduction specialized to either signed `i8`
suffix coefficients or packed row-major one-hot indices. The dense suffix
kernel reads each live byte once. The initial one-hot kernel may reread the
roughly 1 GB packed lane table once per subring coordinate in its first simple
form, but performs only one field addition per hot entry; its 64 GB worst-case
read volume has a roughly 117 ms advertised-bandwidth floor and substantial
cache reuse between adjacent output groups. The generic kernels and scheduling
live in Akita Metal; Jolt supplies only its row-layout adapter. No challenge,
message, or output-coordinate order changes.

Pre-registered prediction: root and recursive coefficient-packing phases each
fall below 180 ms, the complete T25 opening reaches at most 6.25 s, exact proof
and transcript parity remain, and every qualified suffix/packed-row packing
call takes the Metal route. Reject the treatment if either phase remains at or
above 300 ms, the complete improvement is below 350 ms, or a qualified packing
call falls back to CPU. A root-only or suffix-only win may be retained as a
separately evidenced submechanism, but neither alone clears this treatment's
combined prediction.

Measured result: proof, transcript, evaluation, and verifier remain exact, and
the qualified coefficient-packing calls report no CPU-tail work. The initial
packed-root kernel nevertheless took 31.490 s and drove total GPU time to
32.739 s. Its separate threadgroup per `(block, subring)` reread is therefore
not a bandwidth stream: the strided duplicate scans defeat cache locality and
leave nearly all lanes inactive during each bucket update. The combined
treatment is rejected. The signed-suffix kernel and shared preweighting API are
retained pending an isolated phase measurement; no suffix speedup is claimed
from this run.

The root replacement uses one 32-lane SIMDgroup per bounded row tile. Each lane
owns two of the 64 subring buckets. A tile loads 32 rows once, sorts `(bucket,
weight)` pairs inside the SIMDgroup, performs a segmented field reduction, and
lets each bucket owner gather its tile sum into a register accumulator. The
kernel emits 64 partials per row tile; a second kernel reduces those partials
to the canonical block coordinates. At an 8192-row tile, T25 uses about 128 MiB
of partial storage, reads the roughly 1 GiB lane table once, and performs the
unavoidable hot-entry field additions with useful SIMD occupancy.

Pre-registered prediction for the tiled root replacement: exact parity, root
coefficient packing below 300 ms, complete T25 opening at most 6.45 s, and no
qualified coefficient-packing fallback. Reject it if root packing is at least
500 ms, complete improvement versus the 6.877 s centered-witness parent is
below 250 ms, or partial storage projects above the 90 GiB T28 guard.

## Root schedule domain cliff

Static planner replay exposed a valid schedule that is useful for absolute
prover time but not, by itself, for the matched-schedule Metal/CPU ratio. Jolt's
current K256 catalog does not use Akita's unconstrained choice for variables
38--41: it forces D512/B64/D128, `2^19` root positions, and inner rank one to
match the accepted commit backend. The root witness is dominated by

```text
Z coefficients = positions * num_digits_fold * D_A.
```

The following planner-valid constraints retain the current ring roles and
root ranks while crossing the next-witness power-of-two boundary:

| Shape | Geometry | Live handoff | Sumcheck domain | Setup fields |
|---|---|---:|---:|---:|
| T25 current | `2^19` positions, 1024 blocks | 839,195,008 | `2^30` | 268,435,456 |
| T25 candidate | `2^17` positions, 4096 blocks | 336,697,792 | `2^29` | 67,108,864 |
| T28 current | `2^19` positions, 8192 blocks | 1,344,379,328 | `2^31` | 268,435,456 |
| T28 candidate | `2^18` positions, 16,384 blocks | 987,901,760 | `2^30` | 180,355,072 |

T25 keeps the current D512/B64/D128 rank-1 layout, three fold digits, and D256
coefficient-packing challenge. Its first recursive handoff also falls from
17,267,136 to 13,766,592 coefficients. T28 keeps the current 1/2/1 role ranks
and four fold digits; its coefficient-packing challenge subring changes from
D256 to D128. The proof payloads become slightly smaller. Every row was
produced by the ordinary security-checked planner; no security bound was
relaxed.

For the initial commit, dominant source work is invariant. Hot-entry additions
are unchanged. Matrix width falls by 4x/2x while block streams rise by 4x/2x,
so the modeled matrix-byte product is approximately unchanged and the smaller
matrix should cache better. Inner output rises from 8 to 32 MiB at T25 and 64
to 128 MiB at T28. The T28 backend would need to admit 512 blocks per column;
that is a shape extension, not a new algorithm.

This is not retained in the present eval-speedup search. A fair 5x claim runs
CPU and Metal on the same public schedule. The candidate halves both CPU and
Metal Stage-1/Stage-2 work and shrinks most relation work on both backends. A
phase-scaled matched CPU projects to roughly 5.1--6.7 seconds and Metal to
roughly 2.8--3.6 seconds, still near the observed 2x ratio. There is no stated
differential mechanism capable of turning this work reduction into 5x. The
catalog and frozen CPU anchor therefore remain unchanged. Reconsider this
schedule in the later absolute end-to-end goal, after the generic eval backend
has cleared its matched-schedule gate.

## Sparse-histogram Stage-1 two-round prefix

The retained Metal Stage 1 takes 611.95 ms and scans the compact range-image
source four times before it becomes an fp128 table at round three: the initial
round, the folds after challenges zero and one, and the materializing fold after
challenge two. Akita's CPU prover already has a prover-internal two-round prefix
construction that reconstructs the same ordinary first two proof messages from
a 5- or 21-value bivariate grid. Using it on Metal reduces compact-source scans
from `4N` to `3N` without changing a proof or transcript byte.

The Metal prefix must preserve the sparse-data advantage. Computing all 21 grid
values independently would perform 21 field-scaled accumulations per four input
digits. Instead, map each four-digit tuple to one of 16 (`b=4`) or 256 (`b=8`)
classes and accumulate one equality weight into a class histogram. Within each
SIMDgroup, equal classes are coalesced before a threadgroup atomic update. Exact
fp128 sums reuse the backend's bounded eight-`u16` digit accumulator. Coalescing
bounds a bucket to at most one update per SIMD batch, so every signed 32-bit
atomic digit remains below its checked limit. The threadgroup result is reduced
modulo the field before being written. A small device reduction combines class
histograms, then applies the existing integer lookup table to produce the prefix
grid. The ordinary Metal round path resumes after challenge one and materializes
the fp128 table after challenge two.

Pre-registered prediction: exact proof and transcript parity, no Stage-1
fallback, Stage 1 at most 450 ms, and complete T25 opening at most 5.45 s. Reject
the treatment if Stage 1 is at least 540 ms, complete improvement versus the
5.567 s zero-copy-ring parent is below 75 ms, the exact limb accumulation fails
parity, or the bounded T28 memory projection exceeds 90 GiB. A miss should lead
to a histogram redesign rather than independent 21-value full-table scans.

Measured result: proof digest, transcript, claimed evaluation, and verifier all
match the frozen CPU anchor. Stage 1 reached 573.23 ms and the complete opening
reached 5.514 s, only 38.72 ms and 52.74 ms better than the retained parent.
Aggregate GPU-active time fell by about 101 ms, but the prefix's histogram,
factor-table setup, and extra orchestration returned much of that gain. Both the
540 ms Stage-1 bound and the 75 ms complete-improvement bound were crossed. The
treatment is rejected and its backend/API surface is removed.

## Native-width Montgomery reduction

The retained six-prime D512 relation computes Montgomery multiplication in MSL
by materializing two signed 64-bit products. This is unnecessary for the fixed
30-bit primes. If `c = a*b`, `m = low32(c)*p_inv (mod 2^32)`, and
`p*p_inv = 1 (mod 2^32)`, then `low32(m*p) = low32(c)`. The low words of
`c - m*p` cancel exactly, with no borrow, so

```text
(c - m*p) >> 32 = mulhi(a, b) - mulhi(m, p).
```

The replacement uses signed 32-bit low and high multiplies and is bit-identical
to Akita's scalar/NEON Montgomery convention. It changes no CRT prime, transform,
or reconstruction rule. This primitive is also the arithmetic base for the next
D64/D128 recursive-witness commitment candidate.

Pre-registered prediction: the focused D512 relation parity test remains exact,
the T25 ring-switch phase falls from 877 ms to at most 700 ms, and the complete
opening reaches at most 5.40 s. Reject the performance mechanism if ring switch
remains at or above 800 ms or the complete improvement versus the 5.567 s parent
is below 100 ms. Any proof mismatch rejects the implementation outright.

Measured result: focused and integrated proof parity pass, but ring switch reached
840.82 ms and the complete opening reached 5.469 s. The 97.73 ms complete-call
delta is just below the registered minimum and is not corroborated by the target
phase, which improved by only 36.19 ms. The native-width identity is correct but
does not change this kernel's limiting throughput enough; the treatment is
rejected and reverted.

## Fused exact recursive-witness commitment

The retained stack still sends every recursive `commit_w` inner product to the
CPU. At T25 the dominant first call commits 401 blocks of 16,384 D128 signed-i8
rings against three public A rows and accounts for most of the measured 833 ms
next-witness-commit phase. The source is about 841 MB, the canonical A prefix is
about 101 MB, and the final inner rows are only 2.5 MB.

The treatment implements a generic fp128 D64/D128 Metal kernel in Akita. It uses
six 30-bit CRT primes for the exact negacyclic product, transforms the public
matrix once per charged dispatch, and fuses source conversion, forward NTT,
pointwise row products, inverse NTT, and CRT reconstruction. One 1024-thread
group handles 16 source blocks for one prime. It streams each matrix frequency
once across those blocks, keeping per-block/per-row accumulators in registers;
there are no block-by-column partial buffers. T25 therefore reads about 3.9 GB
of transformed matrix data and 5.0 GB of source-prime data, with about 155 MB of
device scratch. T28's scheduled D64 first suffix projects to about 14.5 GB of
streamed traffic and under 310 MB of scratch. Both remain far below the 90 GiB
guard. Later small D64 levels use the same exact kernel rather than a hidden CPU
fallback. The proof, transcript, schedule, and public commitment are unchanged.

Pre-registered prediction: focused D64 and D128 CPU/Metal commitment parity pass;
the T25 next-witness-commit phase reaches at most 550 ms; the complete opening
reaches at most 5.30 s; and no recursive commitment takes the CPU route. Reject
the performance mechanism if the next-witness phase remains at or above 700 ms,
the complete improvement versus the 5.567 s parent is below 150 ms, parity fails,
or the T28 memory projection exceeds 90 GiB.

The first integration launch failed closed before timing because the second T25
D64 level has five A rows, while the initial kernel admission cap was three from
the dominant first-level shape. Raising the static cap to eight adds at most 2.5
KiB of threadgroup storage and does not change the first-level traffic or timing
prediction. The focused rank-five case is added to the credibility gate before
relaunching the same single treatment.

Measured result: D64/D128 focused parity, proof bytes, transcript, evaluation,
and verifier all pass, but the next-witness phase regressed to 936.37 ms and the
complete opening reached 5.540 s. GPU-active time increased by about 795 ms over
the retained parent. The implementation performed about 36 million full
threadgroup barriers while staging D64/D128 transforms, so its nominal 9 GB
traffic bound did not describe the actual synchronization floor. It crosses both
performance falsifiers and is rejected; the barrier-heavy kernel body is removed.

## SIMDgroup-register recursive NTT

The revised exact commit keeps the already-validated operation boundary and CRT
layout but replaces the rejected transform kernel. One SIMDgroup owns one source
block and each lane keeps four D128 coefficients (or two D64 coefficients) in
registers. The first two D128 DIF stages pair registers locally; the remaining
five stages use `simd_shuffle_xor`. The inverse applies the exact reverse DIT
sequence. No source NTT or inverse NTT uses threadgroup memory or a threadgroup
barrier.

Sixteen SIMDgroups share a 512-thread dispatch group. They read the same matrix
frequencies independently, trading a conservative 60--65 GB T25 matrix-read
upper bound (about 160 ms at advertised bandwidth, with likely cache reuse) for
removing the measured synchronization bottleneck. Source traffic remains about
5 GB and scratch remains under 160 MB. Arithmetic and CRT reconstruction are
unchanged.

Pre-registered prediction: focused D64/D128 parity remains exact, T25 recursive
commit reaches at most 500 ms, and complete opening reaches at most 5.20 s.
Reject if recursive commit is at least 650 ms, complete improvement versus the
5.567 s parent is below 200 ms, proof parity fails, or any recursive commit uses
the CPU route.

Measured result: exact D64/D128 parity and integrated proof parity pass, but the
recursive-commit phase reached 886.71 ms and the complete opening reached 5.566
s, effectively identical to the retained 5.567 s parent. Removing the barrier
floor improved only about 50 ms over the first GPU version; aggregate GPU-active
time still rose by about 733 ms. The six-prime transforms, pointwise products,
and repeated matrix reads are the actual floor. The treatment is rejected and
Jolt routes recursive commit back to its faster CPU backend. Further work here
requires a protocol/representation change, not another NTT micro-optimization.

## Factored Stage-2 two-round prefix

The retained Stage-2 kernel spends roughly 13 generic fp128 multiplications per
pair. Its first two rounds cover `N/2 + N/4 = 3N/4` pairs, about 805 million at
T25, even though the witness is still a signed-byte table. This is the wrong
arithmetic representation, not a bandwidth problem: those two rounds alone
perform about 10.5 billion generic field multiplications.

Akita's CPU prover already computes the same two ordinary round messages from a
transient `{0,1,Infinity}^2` bivariate grid. The Metal treatment ports that exact
prover-internal construction. One logical thread owns a live lane and its 16
four-digit coefficient quads. Eight norm and eight relation grid coordinates
are dispatched independently to keep register pressure low. For the norm grid,
512-lane-aligned threadgroups accumulate `E_first * small_integer` first and
apply their common `E_second` factor once after reduction. For the ordinary
relation grid, a lane accumulates `alpha_point * small_integer` over its 16
quads and applies the resident lane weight once. Structured-linear segments use
the same factor-outside-the-quad-sum rule. A one-word signed-small multiplier
handles the bounded bilinear digit values without invoking the generic 4-by-4
limb product.

The T25 operation model is about 105 million generic relation multiplications,
fewer than one million norm outer multiplications, and about 3.35 billion
one-word small-scalar products, plus actual structured support. The compact
witness is read 16 times, or about 13.4 GB; that is only a 24.6 ms ideal traffic
floor on the target machine. Bounded prefix partials use about 6.6 MB at T25 and
about 10.5 MB for the T28 populated shape. The grid reconstructs the canonical
compressed round-zero and round-one polynomials on the host, after which the
existing compact round-two path resumes. No verifier, transcript, variable
order, or serialized proof changes.

Pre-registered prediction: focused CPU/Metal parity remains exact, Stage 2 falls
from 1.49 s to at most 0.85 s, and the complete T25 opening reaches at most
4.95 s. Reject or redesign if Stage 2 remains at or above 1.10 s, complete-call
improvement versus the 5.567 s parent is below 350 ms, any qualified prefix uses
the old full scans, proof parity fails, or the T28 projection exceeds 90 GiB.

Measured result: the focused parity test and the integrated proof are exact.
Stage 2 reached 976.71 ms and the complete opening reached 5.018 s, improvements
of 512.82 ms and 548.74 ms respectively. GPU-active time fell by 494.20 ms;
allocation increased by only 13.42 MB, matching the bounded prefix workspace.
The treatment clears both falsifiers and is retained. It misses the 0.85 s phase
prediction, so the remaining one-word scalar products and later generic fp128
rounds remain an explicit arithmetic target.

## Native-u32 fp128 arithmetic

Every current fp128 add, subtract, small-scalar product, and generic product is
written with MSL `ulong` temporaries. The operands are 32-bit limbs, but a full
product still issues sixteen 64-bit multiplies and the new Stage-2 prefix issues
four more 64-bit multiplies for each bounded scalar product. The accepted prefix
performs about 3.35 billion such small-scalar products, so its nominally cheaper
arithmetic still requests roughly 13.4 billion 64-bit limb multiplies.

Metal exposes the exact low word with ordinary `uint` multiplication and the
exact high word with `mulhi(uint,uint)`. Grade-school multiplication can carry
these two words with overflow comparisons alone. Reduction modulo Akita's
`2^128 - 0xffffa7f7` prime uses the same identity: multiply each high limb by
the fixed 32-bit offset as a low/high pair, add the low word, and propagate a
32-bit carry. Addition and subtraction likewise need only 32-bit operations and
carry/borrow comparisons. This changes no field representation or reduction
rule; it removes accidental 64-bit arithmetic from the common Metal primitive.

Pre-registered prediction: the direct Stage-2 parity test and integrated proof
remain exact, Stage 2 reaches at most 0.70 s, Stage 1 reaches at most 0.48 s, and
the complete T25 opening reaches at most 4.45 s. Reject the performance mechanism
if Stage 2 remains at or above 0.90 s, complete improvement versus 5.018 s is
below 250 ms, or any field/ring/proof parity check fails.

Measured result: exact parity holds, but Stage 2 regressed to 1.013 s, Stage 1
regressed to 673.46 ms, and the complete opening regressed to 5.081 s. GPU-active
time increased by 88.84 ms. The compiler/architecture handles the widened limb
expressions better than the explicit low/high/carry sequence; the mechanism is
rejected and fully reverted.

## Factored Stage-2 three-round prefix

The retained two-round prefix still resumes the generic Stage-2 scan for round
two over `N/8` pairs. At T25 that round alone performs roughly 1.74 billion
generic fp128 multiplications. Extending the same canonical construction to a
transient `{0,1,Infinity}^3` grid removes that scan without changing the proof,
transcript, verifier, or variable order. The Metal kernel evaluates 27 norm and
27 ordinary-relation grid points over eight signed-byte coefficients at a time;
the host reconstructs the first three ordinary round messages, while the
existing sparse additional relation remains on its exact compact path.

For the populated T25 shape, about 13.1 million logical lanes each process eight
octets. The treatment performs about 5.67 billion one-word small products and
27 full lane-weight products per lane, plus structured-linear support, instead
of the eliminated round-two generic products. Its 54 compact-witness passes
move about 45 GB, an ideal target-machine traffic floor near 83 ms. Prefix
partials occupy about 22 MB at T25 and remain negligible relative to the 90 GiB
T28 guard. A four-round grid would triple the point count again while removing
only `N/16` generic pairs, so it is not implied by this treatment.

Pre-registered prediction: focused CPU/Metal parity remains exact, Stage 2
reaches at most 0.78 s, and the complete T25 opening reaches at most 4.85 s.
Reject or redesign if Stage 2 remains at or above 0.90 s, complete improvement
versus the retained 5.018 s parent is below 150 ms, proof/transcript parity
fails, the qualified round-two generic scan remains, or the T28 projection
exceeds 90 GiB.

Measured result: proof, transcript, and verifier parity remained exact, but
Stage 2 regressed to 1.053 s and the complete opening regressed to 5.165 s.
GPU-active time rose by 99.84 ms and allocation rose by 31.13 MB. The extra
38 grid passes cost more than the eliminated round-two generic scan on this
architecture, crossing both registered timing falsifiers. The treatment is
rejected and removed; the two-round prefix remains the retained boundary.

## Pipelined ring switch and recursive commitment

The retained root fold serializes an 875.6 ms ring-switch witness build and an
832.3 ms commitment of that witness. This serialization is not inherent to the
protocol. The witness is laid out as a large `[Z | E | T]` body followed by a
small relation/compression suffix, while the inner Ajtai commitment applies the
same matrix independently to each source block.

For the frozen T25 schedule, the exact root body is

```text
Z = 524288 * 1 * 3 * 512       = 805306368 bytes
E = 1 * 1024 * 43 * 256       =  11272192 bytes
T = 1 * 1024 * 1 * 43 * 512   =  22544384 bytes
body                            = 839122944 bytes
complete next-level blocks      = 400 / 401
complete-block prefix           = 838860800 bytes
live witness                    = 839195008 bytes
relation/compression suffix     =     72064 bytes
```

Thus 99.96% of the live witness and 400 of 401 commitment blocks are independent
of the Metal relation quotient. Akita will launch quotient construction on the
ring-switch backend while a host branch emits the canonical body and commits
the 400 complete blocks. Once both finish, it emits the exact suffix, commits
the one boundary block, concatenates the inner rows in canonical block order,
and runs the unchanged outer commitment/compression path. No proof message,
transcript order, schedule parameter, or witness byte changes.

This is a backend-neutral protocol scheduler seam: any thread-safe commitment
and ring-switch backends may overlap, and ineligible encodings or small prefixes
keep the existing serial path. It also avoids coupling Metal implementation
details to the protocol crate.

Pre-registered prediction: focused commitment composition and complete proof
parity remain exact; the combined root `ring_switch_build_w` plus
`next_witness_commit` critical path falls from 1.708 s to at most 1.30 s; and
the complete T25 opening reaches at most 4.62 s. Reject the performance
mechanism if the combined phases remain at or above 1.50 s, complete improvement
versus the retained 5.018 s parent is below 250 ms, proof/transcript parity
fails, or the implementation copies the 800 MiB prefix.

Measured result: proof digest, transcript, evaluation, and verifier all match
the frozen CPU control. The combined phases fell to 1.132 s, the residual
post-join commitment is 51.1 ms, and the complete opening fell to 4.549 s, a
469.1 ms improvement over the retained parent. Peak RSS was 15.98 GB and no
prefix copy was introduced. The mechanism clears every registered falsifier
and is retained. Complete speedup is now 2.43x versus the 11.043 s CPU anchor;
the 5x threshold remains 2.209 s.

## Root response-basis audit

The honest one-hot bound, rather than an arbitrary byte layout, determines the
three current response planes. At the T25 root the exact policy threshold is 88
for the D256 challenge subring and 60 for a full D512 challenge. Replaying the
security-checked planner at opening bases 3 through 6 gives live handoffs of
839,195,008, 562,107,904, 557,382,912, and 554,232,896 coefficients for the
ordinary D256 challenge. All bases through 6 therefore remain above `2^29` and
retain the same `2^30` Stage-2 domain. A D512 challenge likewise retains two
base-64 response digits and a 560,005,696-coefficient handoff.

Base 128 is the first existing signed-byte representation that fits the D512
threshold in one digit, but Akita's range-proof topology is intentionally
defined only through base 64. Extending it would add two product substages and
change the protocol. It may be useful for the later absolute-time campaign,
but it has no demonstrated matched-CPU differential and is not the next
benchmark candidate. The current catalog and frozen CPU anchor remain intact.

## SIMDgroup-partitioned packed root fold

The packed root decompose kernel currently assigns all eight SIMDgroups in a
256-thread threadgroup to one output position. They concurrently scatter about
44,000 signed challenge terms into the same 512 threadgroup atomics. T25 launches
524,288 such threadgroups. The arithmetic count is inherent to the sparse fold,
but neither the eight-way contention nor the per-position threadgroup lifecycle
is inherent.

The treatment assigns one output position to each 32-lane SIMDgroup. One
256-thread threadgroup therefore owns eight positions and eight disjoint 512-bin
accumulators. Total sparse additions are unchanged, while competing writers per
accumulator fall from 256 to 32 and the number of threadgroups and full-group
barriers falls by eight. The output remains the same position-major centered
`i32` table. The Jolt adapter adds a phase span so the first treatment measures
this previously hidden part of `other_ns`; this instrumentation changes no
protocol or timing boundary.

Pre-registered prediction: focused packed-fold parity and complete proof,
transcript, evaluation, and verifier parity remain exact; the T25 complete call
falls to at most 4.30 s. Reject the performance mechanism if the complete call
is at least 4.40 s, improvement versus the retained 4.549 s parent is below
150 ms, the packed fold falls back, or the T28 projection exceeds 90 GiB.

Measured result: focused parity and the complete proof, transcript, evaluation,
and verifier all remained exact, but the complete call was 4.483 s. This is only
66.1 ms below the retained parent, crosses the 4.40 s absolute falsifier, and is
consistent with run noise because GPU-active time increased by 3.0 ms. The newly
visible packed root fold took 476.4 ms. Eight times more threadgroup storage
reduced resident groups while the kernel retained every sparse atomic addition,
canceling the expected scheduling gain. The kernel mapping is removed; the
phase span is retained because it revealed a real 476 ms optimization target.

## Four-point fused Stage-2 prefix

The retained two-round Stage-2 prefix computes eight norm points and eight
relation points over the same compact digit quads. Its Metal grid assigns one
point to each grid slice, so the identical four source digits are loaded in 16
separate slices and each slice pays a separate threadgroup lifecycle. The point
evaluations are independent, but the source loads are not.

The treatment assigns four point slots to each thread. Four grid slices cover
the same 16 outputs. Each digit quad is loaded once and feeds two norm and two
relation evaluations; structured linear source quads are likewise loaded once
for two relation points. The proof messages, point set, reduction order within
each point, transcript, and resident tables are unchanged. Register state grows
by four fp128 accumulators, while threadgroup storage grows from eight to 32
fp128 values and remains 512 bytes.

Pre-registered prediction: focused and complete parity remain exact, Stage 2 is
at most 0.80 s, and the complete T25 opening is at most 4.30 s. Reject the
mechanism if Stage 2 is at least 0.90 s, complete improvement versus the retained
4.549 s parent is below 150 ms, proof/transcript parity fails, or the qualified
route falls back.

Measured result: the focused proof and complete proof/transcript/verifier stayed
exact, but Stage 2 was 985.3 ms, only 4.9 ms below the retained parent. Complete
time was 4.464 s, an 85.5 ms difference below the registered acceptance floor;
GPU-active time fell by only 22.4 ms. Compact loads were therefore not the
prefix bottleneck: cache coalescing already hid most repeated traffic, while the
larger register set reduced occupancy. The fused-point mapping is removed.

## Cached root-response handoff

The retained D512 packed fold produces 524,288 centered response rings. Metal
currently writes 1,073,741,824 bytes of `i32` coefficients, the runtime copies
that buffer back to ordinary `Vec` storage, and Akita later scans the same table
again to emit three balanced base-8 response planes. Those 805,306,368 bytes are
the first canonical range of the next witness. The ring-switch path then commits
400 complete blocks covering almost that entire range. None of the second
decomposition, witness-prefix allocation, or copy is protocol work.

The treatment adds an optional, backend-produced balanced-response cache to the
generic decompose-fold result. The response basis/depth and exact next-witness
capacity come from Akita's checked level layout. The Metal fold writes the three
`i8` planes alongside each centered coefficient. For the eligible single-group,
single-chunk layout, ring switch promotes that allocation in place to the
canonical witness, skips Z decomposition/emission, and retains the existing
quotient/CPU-prefix-commit join. CPU and unsupported backends ignore the hint;
the centered `i32` response remains authoritative for grinding and quotient
construction. Proof bytes, transcript order, matrix schedule, and verifier are
unchanged.

The extra device writes have an approximately 0.81 GB traffic floor, while the
treatment removes a roughly 0.10 s centered-buffer copy plus at least 2.4 GB of
host read/decompose/write/copy traffic. It should also move the body branch of
the existing ring-switch/commit join below its commitment branch. Pre-registered
prediction: focused cached-digit parity and complete proof/transcript/verifier
parity remain exact; root decompose-fold is at most 0.42 s, ring-switch build is
at most 0.90 s, and complete T25 opening is at most 4.20 s. Reject the mechanism
if complete improvement versus the retained 4.549 s parent is below 200 ms,
root fold plus ring switch is not reduced by at least 200 ms, the cache is
copied while promoting it to the witness, or peak RSS exceeds the 90 GiB T28
projection.

Measured result: the proof digest, transcript, evaluation, and verifier stayed
exact, and ring-switch build fell to 945.1 ms. The extra response writes raised
root fold to 515.8 ms, however, so the combined phases improved by only about
0.10--0.11 s. Complete opening was 4.396 s, a 153.8 ms gain versus the retained
parent but below the registered 200 ms materiality gate; peak RSS was 15.17 GB.
Canonical Z decomposition/emission is therefore only a roughly 0.15 s cost, and
the cache surface is rejected and removed rather than retained as generic API.

## Setup-resident D512 relation transform

The retained root A relation multiplies 524,288 public D512 setup rings by the
centered response under six exact 30-bit CRT primes. Its timed Metal kernel
converts each 128-bit setup coefficient to every prime and performs the same
forward length-1024 transform on every proof. The matrix and all of those
transforms depend only on the reusable Akita setup, not on the witness,
transcript, opening point, or proof.

The treatment adds a prepared Akita Metal cache containing the six transformed
prime images in `(column, prime, frequency)` order. The exact cache size is

```text
524288 columns * 6 primes * 1024 frequencies * 4 bytes
    = 12,884,901,888 bytes = 12 GiB.
```

It is constructed explicitly by backend setup preparation and reported outside
the timed opening; lazy construction remains charged if a generic caller omits
prewarming. The timed relation kernel then transforms only the private `i32`
response, reads 12 GiB of prepared residues instead of rereading 24 GiB of raw
fp128 matrix limbs across six primes, and removes one of its two forward NTTs.
Reduction, CRT reconstruction, quotient semantics, proof bytes, transcript,
schedule, and verifier remain unchanged. The cache raises the T25 projection
from roughly 16 GB to roughly 29 GB and has the same 12 GiB root width at T28,
remaining well below 90 GiB.

Pre-registered prediction: focused relation and complete proof parity remain
exact; prepared setup reports a 12 GiB cache hit in the timed call; ring-switch
build reaches at most 0.70 s; and complete T25 opening reaches at most 4.20 s.
Reject the mechanism if ring switch remains at or above 0.85 s, complete gain
versus the retained 4.549 s parent is below 250 ms, cache construction occurs
inside the timed opening, parity fails, or the T28 projection exceeds 90 GiB.

Measured result: the focused cache-miss/cache-hit test and the complete proof,
transcript, evaluation, and verifier all remained exact. Explicit preparation
took 1.652 s outside the opening, and the timed allocation count confirms that
the prepared matrix was not rebuilt. GPU-active time fell by 165.8 ms, but
`ring_switch_build_w` fell by only 13.6 ms to 1.068 s and complete opening fell
by only 65.2 ms to 4.484 s. Both timing falsifiers trigger. The existing
relation/host-emission join already hides this device work behind the host
branch, so shortening the noncritical relation branch cannot materially shorten
the proof. The 12 GiB cache and its setup/adapter surface are removed.

## Bucket-owner packed root coefficient packing

The retained root coefficient-packing phase is still a deliberate CPU fallback
and costs 0.42--0.50 s at T25. It cannot simply overlap the packed root fold:
its scalar opening and D-role payload are absorbed before Fiat--Shamir samples
the sparse fold challenges. Moving this work off the critical path therefore
requires accelerating the exact operation rather than reordering it.

The existing Metal kernel is also analytically unsuitable at this scale. The
T25 source contains about 1.006 billion live `(row, column)` selectors, and the
kernel performs eight contended threadgroup integer atomics for each selected
fp128 weight. The treatment instead assigns each output subring bucket to one
thread in each of four row shards. Threads in a shard read the same selector
address, which Metal can broadcast across the two SIMDgroups covering the 64
buckets; only the owning bucket gathers the weight and performs an exact fp128
addition. Four shard results are reduced from 4 KiB of threadgroup storage.
The partial layout, final reduction, scalar opening, transcript, and proof are
unchanged. The Jolt adapter routes its singleton packed root batch to Akita's
generic packed-one-hot coefficient-packing operation and removes this fallback.

Pre-registered prediction: focused and complete parity remain exact; root
coefficient packing reaches at most 0.25 s; and complete T25 opening reaches at
most 4.32 s. Reject the mechanism if the phase is at least 0.35 s, complete
improvement versus the retained 4.549 s parent is below 150 ms, the operation
falls back, or proof/transcript/verifier parity fails.

Measured result: focused and complete proof/transcript/verifier parity remained
exact, and CPU-tail work fell by exactly 1,006,632,960 units, confirming that
the intended root operation ran on Metal. The phase took 30.524 s, however,
with 31.799 s total GPU-active time, and complete opening regressed to 34.762 s.
Broadcast selector loads did not make the replicated loop free: 64 bucket
owners replayed every row and forced roughly one masked fp128 SIMD path per
selector. The phase and complete-time falsifiers trigger by orders of
magnitude. Remove the kernel and adapter route. A future packing design must
partition each selector once and then perform a segmented reduction or sort;
it cannot gather by rescanning the rows for every destination bucket.

The original atomic scatter was then checked with one bounded exact scaling
probe rather than a full treatment. A saturated 2^24-row, one-column instance
matching the T25 block geometry covers 16,777,216 selectors and measured
12.234 ms GPU-active and 16.069 ms command wall. T25 contains exactly 60 times
that selector work, projecting to about 0.734 s GPU-active and 0.964 s command
wall under linear saturated scaling. Both already exceed the 0.35 s gate and
the retained CPU phase, so the atomic route is rejected analytically without a
full T25 relink. The temporary probe is removed.

## Bounded-scalar fp128 multiplication

The retained Stage-1 and Stage-2 kernels use two exact scalar-multiply paths.
The `long` path expands every scalar into two 32-bit words and issues eight
limb products, even though every caller is bounded by a signed 30-bit CRT
digit, a base-8 witness value, or a degree-four range-polynomial constant.
The `int` path needs only four independent limb products, but expresses them
as a carry-dependent loop. The Stage-2 two-round prefix alone applies the
`int` path to roughly 3.36 billion compact point contributions at T25, while
Stage 1 uses the unnecessarily wide path for its compact folds and fixed
constants.

The treatment makes the signed-32-bit bound explicit in the Metal primitive,
computes the four raw limb products independently before carry propagation,
and routes the wider-named helper through that exact implementation. It does
not change a field value, proof message, transcript, backend seam, or protocol
configuration. Its useful mechanism is lower scalar-multiply latency and
instruction-level parallelism, not reduced source traffic.

Pre-registered prediction: focused direct Stage-1 and Stage-2 proof parity
passes; Stage 1 reaches at most 0.52 s, Stage 2 reaches at most 0.80 s, and the
complete T25 opening reaches at most 4.25 s. Reject the treatment if Stage 2 is
at least 0.90 s, complete improvement versus the retained 4.549 s parent is
below 150 ms, aggregate GPU-active improvement is below 120 ms, or any proof,
transcript, evaluation, or verifier result differs.

Measured result: the interrupted command retained the Cargo artifact lock, so
the retry yielded two exact observations rather than the intended one. Both are
decisive regressions: complete opening was 6.983 s and 7.103 s, Stage 2 was
1.328 s and 1.287 s, ring-switch construction was 2.207 s and 2.310 s, and GPU
active time was 2.858 s and 2.682 s. Both proofs retained the frozen digest and
passed evaluation, transcript, CPU-parity, and verifier checks. The broken
assumption is that expressing four products as a `ulong4` would expose useful
instruction-level parallelism without changing resource pressure. The result is
consistent instead with a materially worse compiled schedule or live-state
footprint, and routing the generic `long` helper through it also penalized the
CRT reconstruction paths. Revert both helper edits. Any future scalar-arithmetic
candidate must isolate one caller family and clear a focused codegen/timing probe
before another complete treatment.

## Cached Stage-2 relation-prefix multiples

The retained two-round Stage-2 prefix evaluates eight ordinary-relation grid
points over 13,112,422 live lanes and 16 digit quads per lane. The compact
witness point is always in `[-14, 14]`, while the field-valued alpha point is
one of only `8 * 16 = 128` values shared by every lane. The current kernel still
computes

```text
8 * 13,112,422 * 16 = 1,678,390,016
```

signed-small field products, or 6.714 billion 32-by-64-bit limb products, for
this ordinary-relation half alone.

The treatment precomputes all 29 exact signed multiples of each shared alpha
point on the host and uploads a 128-by-29 fp128 table (59,392 bytes). The
relation-prefix kernel uses its bounded witness point as a table index. The norm
half and structured-linear sources retain their existing arithmetic because
their field operands vary by lane. Inputs, eight compressed-grid outputs,
challenge order, proof bytes, and verifier are unchanged; out-of-range indices
fail the host-side shape/range invariant rather than wrapping.

The compulsory relation-half digit traffic is eight scans, about 6.71 GB at
T25. Logical lookup traffic is 26.85 GB, but its entire 59 KiB source repeats
across every threadgroup and should be cache-resident. A warm 512 MiB Metal blit
on this M4 Max measured 2.642 ms, or 406.4 GB/s counting its read plus write
traffic. Even charging every logical lookup to DRAM gives an 82.6 ms traffic
floor for the 33.56 GB combined stream; a cache-resident table lowers it toward
16.5 ms for the compulsory digits. The retained prefix requests roughly 13.43
billion limb products across its norm and ordinary halves. Against the measured
0.990 s complete Stage-2 phase, 13.6 billion limb products/s is a conservative
phase-wide issue calibration; the unchanged norm half alone prices near 0.49 s
at that rate before later rounds. Compute, not compulsory DRAM, therefore remains
the predicted bound. Eliminating half of the dominant compact-prefix products
has a useful ceiling of roughly 0.2--0.4 s after unchanged later rounds and norm
work.

Pre-registered prediction: focused direct Stage-2 and complete proof parity
remain exact; Stage 2 reaches at most 0.78 s; GPU-active time falls by at least
120 ms; and complete T25 opening reaches at most 4.30 s. Reject the treatment if
Stage 2 is at least 0.88 s, complete improvement versus 4.549 s is below 150 ms,
GPU-active improvement is below 120 ms, the table exceeds 64 KiB, or any proof,
transcript, evaluation, or verifier result differs.

Measured result: proof, transcript, evaluation, CPU parity, and verifier all
remained exact. Stage 2 reached 0.975 s, only 15.0 ms below the retained parent;
GPU-active time improved by 23.9 ms, and complete opening reached 4.475 s, a
73.9 ms gain. All three timing falsifiers trigger. The broken assumption is that
the ordinary small products were both phase-dominant and more expensive than
divergent reads from the 59 KiB table. The lookup stream replaced most of the
nominal arithmetic saving with cache/address work, while norm and later-round
arithmetic remained. Remove the table and ABI changes. A further Stage-2 design
must reduce the number of grid contributions or sumcheck passes, not substitute
one operation for every existing contribution.

## Deferred Stage-2 prefix accumulation

The same two-round prefix currently reduces every one of its 16 local
`field * signed-small` contributions to canonical fp128 before adding it to the
thread accumulator. Across the eight norm and eight relation grid points this
requests about 3.357 billion small-field products and the same number of
per-product carry/fold sequences at T25.

Akita Metal already has an exact redundant accumulator that represents a field
value as eight signed radix-`2^16` digits and canonicalizes only after a bounded
batch. Extend it with a signed-small update and use one accumulator for each
16-quad inner sum. A relation witness point is in `[-14,14]`; a norm value is at
most `14 * 15 = 210` in magnitude. Therefore every signed digit accumulator is
bounded by

```text
16 * 65535 * 210 = 220,197,600 < i32::MAX.
```

The structured-linear inner sums have the smaller relation bound. The existing
reducer handles the final signed carry modulo `2^128 - 0xffffa7f7`. This replaces
four 64-bit limb products plus carry/fold work per contribution with eight
32-bit digit products/adds and one canonical reduction per 16 contributions.
It adds four 32-bit live accumulator words relative to the canonical fp128 sum,
while eliminating the five-word product and folded temporaries inside the loop.
The kernel boundary, traffic, grid values, transcript, proof, and verifier are
unchanged.

The compact witness still makes 16 scans, about 13.43 GB. At the measured
406.4 GB/s read-plus-write rate that compulsory stream has a 33.0 ms traffic
floor; the operation remains arithmetic-bound. Pre-registered prediction:
focused and complete parity remain exact; Stage 2 reaches at most 0.72 s;
GPU-active time improves by at least 150 ms; and complete T25 opening reaches at
most 4.25 s. Reject the treatment if Stage 2 is at least 0.85 s, complete gain
versus 4.549 s is below 150 ms, GPU-active gain is below 150 ms, the bound is
exceeded for any admitted basis/point, or any proof, transcript, evaluation, or
verifier result differs.

Measured result: proof, transcript, evaluation, CPU parity, and verifier all
remained exact. Stage 2 reached 0.947 s, a 43.0 ms improvement; GPU-active time
improved by 58.9 ms; and complete opening reached 4.443 s, a 105.9 ms gain. All
registered timing gates miss. The broken assumption is that per-contribution
canonicalization dominated the prefix. Eight scaled 16-bit digit updates and
the larger live accumulator recovered only a small fraction of that work on
this GPU. Remove the helper and kernel change. Together with the cached-multiple
result, this closes the local small-field arithmetic family: reaching 5x now
requires eliminating contributions, scans, or serial phases rather than another
representation of the same prefix products.

## Compact Stage-2 residency through the coefficient boundary

The retained Metal session materializes fp128 witness tables after three compact
rounds even though the T25 root has six coefficient-address rounds. This cutoff
is a backend constant, not a protocol requirement. The existing compact fold
kernel accepts the complete prefix weight table and can defer materialization
until all six coefficient challenges have been bound. The first resident tables
therefore shrink from `N/8 + N/16` fp128 elements to `N/64 + N/128`, a reduction
of 189,582,540 fp128 elements, or 2.825 GiB at `N = 2^30`.

Rounds three through five each recompute one compact MLE pass from the signed-byte
witness using prefix tables of width 8, 16, and 32. The materializing pass uses
width 64 and directly emits one fp128 value per relation lane while producing the
first lane-round message. Compared with the retained parent, this adds three
compact signed-small passes and removes three large generic fp128 fold/message
passes. Challenges, round polynomials, variable order, final witness evaluation,
proof bytes, and verifier code are unchanged.

Pre-registered prediction: the focused Stage-2 proof remains exact, resident
allocation falls by at least 2.7 GiB, Stage 2 reaches at most 0.78 s, complete
T25 opening reaches at most 4.30 s, and aggregate GPU-active time improves by at
least 120 ms. Reject the performance mechanism if Stage 2 is at least 0.88 s,
complete improvement versus the 4.549 s parent is below 150 ms, allocation falls
by less than 2.7 GiB, GPU-active improvement is below 120 ms, or any proof,
transcript, evaluation, or verifier value differs.

Measured result: exact proof, transcript, evaluation, CPU parity, and verifier
checks all passed. Requested allocation fell from 11.585 GB to 8.670 GB, clearing
the memory prediction, but Stage 2 reached 0.975 s, complete opening reached
4.535 s, and GPU-active time was unchanged at 1.593 s. The wider compact prefix
therefore exchanges generic field folds for almost exactly equivalent recompute
work. Every timing gate fails; restore the three-round cutoff. The run also
captured the previously unbucketed root decompose-fold at 0.474 s and reduced
`other_ns` to 0.335 s, closing the phase accounting without another diagnostic
run.

## Segmented bucket-major root coefficient packing

The retained Jolt route sends 1,006,632,960 non-padding T25 selectors through a
CPU scatter/reduction and spends about 0.49 s in root coefficient packing. The
existing generic Metal kernel instead performs eight contended signed-digit
atomics for every selector. A saturated 16,777,216-selector probe projected that
kernel to 0.73 s GPU-active and 0.96 s command time, so merely routing it cannot
win.

Partition each row tile by its subring bucket before doing any fp128 arithmetic.
A 46 x 256 = 11,776-row tile fits in the 32 KiB Metal threadgroup limit with
23,552 bytes of local row indices, four 1,024-byte bucket-counter banks, 512
bytes of bucket offsets, and a 4,096-byte fp128 reduction array (32,256 bytes
total). The first pass builds four banked 32-bit histograms. A 256-entry prefix
scan assigns disjoint bucket-major ranges. The second pass scatters 16-bit local
row indices using the same banked counters. Each bucket then reads only its own
rows and performs ordinary fp128 additions. Buckets above a fixed skew threshold
use all 256 threads and the reserved reduction array, so a trace dominated by
one selector does not serialize on one bucket owner.

This changes eight contended digit atomics into two banked 32-bit index atomics
per valid selector. At T25 it rereads roughly 1.0 GB of selectors twice, reads
about 16.1 GB of combined weights, and writes about 0.38 GB of partials before a
small second reduction. Against the measured 203.2 GB/s payload bandwidth, the
unavoidable payload floor is about 91 ms. The 90 row partials per block require
0.35 GiB at T25 and 2.81 GiB at T28; projected peak memory remains well below the
90 GiB gate. Protocol order, challenges, proof bytes, and verifier code are
unchanged. Jolt's singleton trace adapter can expose its already-aligned
`PackedOneHotView` directly to the generic Akita operation.

Pre-registered prediction: focused coefficient-packing parity and the complete
proof remain exact; the root coefficient-packing phase reaches at most 0.30 s;
complete T25 opening reaches at most 4.35 s; and the operation removes exactly
the root packing CPU fallback. Reject the treatment if root packing exceeds
0.35 s, complete improvement versus the 4.549 s parent is below 150 ms, the
operation falls back, T28 projected peak memory exceeds 90 GiB, or any proof,
transcript, evaluation, CPU parity, or verifier result differs.

Measured result: all exactness checks passed and the adapter removed precisely
1,006,632,960 root fallback work units, leaving 51,949,056 unrelated CPU-tail
units. Root coefficient packing nevertheless took 0.724 s, complete opening
took 4.744 s, and GPU-active time rose from 1.593 s to 2.002 s. The root phase
and complete-proof falsifiers both trigger. Two banked index atomics, three lane
reads, bucket scatter, canonical fp128 additions, and 0.35 GiB of partial traffic
collectively cost approximately the same as the rejected eight-digit-atomic
projection. Remove the kernel and adapter route. Together with the original
atomic projection and the 30.5 s bucket-owner result, this closes standalone
GPU root coefficient packing: a winning design must fuse it with another root
pass or avoid constructing the coefficient vector.

## Deferred CPU root coefficient scatter for the hybrid route

The retained hybrid route spends about 0.49 s in a generic CPU scatter over
1,006,632,960 live trace selectors. Each accepted selector currently performs
two divisions, two remainders, checked slice access, an fp128 multiplication,
extension-coordinate extraction, and a canonical fp128 addition. For the exact
production root geometry (`D = 512`, `K = 256`, base-field opening, stride two),
the map instead has the closed form

```text
position = local_row / 2
bucket   = ((local_row & 1) << 7) | (hot / 2)
low      = hot & 1.
```

Precompute the two products of every position weight with the two packing
weights once, reducing roughly one billion fp128 multiplies to 1,048,576.
For each of the 32 independent row blocks, accumulate the selected canonical
weight into a 30 x 256 table represented by two wrapping `u64` limbs and an
`i32` count of `2^128` wraps. A bucket receives at most 524,288 additions even
under maximal selector skew, so the wrap count is strictly below `2^20`.
Canonicalize only the 245,760 live output coordinates after the scan, using
`2^128 = MODULUS_OFFSET`; padded columns retain their existing zero blocks.

The scan reads about 1.0 GB of selector lanes and 16.1 GB of weight values. It
does no per-selector field multiplication, allocation, or fallible dispatch.
The per-task accumulator is about 160 KiB, or about 5 MiB over 32 tasks. The
protocol, transcript, proof bytes, backend schedule, CPU baseline, and verifier
are unchanged: this is explicitly the CPU half of the retained hybrid Metal
route, not a CPU-baseline change.

Pre-registered prediction: focused generic-versus-specialized parity and the
complete proof remain exact; root coefficient packing reaches at most 0.30 s;
complete T25 opening improves by at least 150 ms versus the 4.549 s retained
parent; and no Metal operation route or fallback count changes. Reject the
treatment if root packing exceeds 0.35 s, complete gain is below 150 ms, the
closed-form geometry is used outside its checked preconditions, maximal-skew
wraps exceed the bound, or any proof, transcript, evaluation, CPU parity, or
verifier result differs.

Measured result: the focused generic parity and maximal-skew tests passed.
Root coefficient packing fell from 0.497 s to 0.132 s, while complete T25
opening fell from 4.549 s to 4.193 s, a 356.8 ms improvement. Proof digest,
transcript, claimed evaluation, CPU parity, and verifier result remained exact.
GPU-active time, allocation, and CPU-tail counters stayed effectively unchanged,
as predicted for a host-only hybrid improvement. Retain this path.

## Eval-oriented large-root schedule

The generated Jolt K256 catalog deliberately constrains one-polynomial roots at
38--41 variables to the commit-oriented `(A, B, D) = (512, 64, 128)` geometry,
`2^19` positions per block, and rank one. At T25 this creates an 839,195,008-byte
first recursive witness, padded to a `2^30` Stage-1/2 domain, and requires
268,435,456 setup fields.

Constrained enumeration with the current planner finds a Pareto-dominating
eval schedule that preserves `A = 512`, `B = 64`, rank one, and the exact
76,420-byte T25 planner payload estimate, while using `D = 64`. The smallest
admitted position domains are shape-derived: `2^16` for 38--39 variables,
`2^17` for 40, and `2^18` for 41. At T25 its first witness is 337,246,656
bytes, its Stage-1/2 domain is `2^29`, and its setup capacity is 33,554,432
fields. At T28 the first witness contracts from 1,344,379,328 to 942,810,240
bytes and its padded sumcheck domain halves from `2^31` to `2^30`. Thus the existing D512
packed-root kernels remain applicable; root selector traffic is unchanged and
root partial output grows from 4 MiB to 32 MiB, but both dominant sumcheck
domains halve and the generated witness/first recursive opening contracts by
2.49x.

Using the retained 4.193 s parent, halving Stage 1 and Stage 2 alone predicts a
0.80 s saving. Scaling only the host-emitted portion of the 1.09 s ring-switch
build and the 0.188 s first recursive coefficient packing predicts a further
0.45--0.65 s saving. Allowing 0.15 s for the eightfold block count gives a
2.65--2.95 s Metal prediction. This is an architectural milestone, not the
final 5x result; the next kernel work starts from its smaller exact domain.

Because the public schedule changes, the old CPU anchor is invalid. Run one
fresh CPU-then-Metal pair and require byte-identical proofs within that pair.
Accept the schedule if the first witness and domains match the model, Metal is
at most 3.10 s with at least 1.0 s improvement over the retained parent, setup
contracts eightfold, memory stays below 90 GiB, and both proofs verify with
matching transcripts. Reject it if any structural prediction fails or if the
extra block count consumes the modeled gain.

Measured result: the fresh CPU/Metal pair produced byte-identical 270,508-byte
proofs, matching evaluations and transcripts, and successful verification.
Metal complete opening fell from 4.193 s to 2.938 s, a 1.254 s improvement.
Allocation fell from 11.58 GB to 5.16 GB and peak RSS to 10.7 GB. The new CPU
anchor is 6.951 s. Stage 1 and Stage 2 fell to 0.325 s and 0.454 s, ring-switch
build to 0.594 s, and recursive coefficient packing to 0.048 s. Root coefficient
packing regressed from 0.132 s to 0.439 s because the retained closed-form host
scatter admitted only stride two while the new root uses stride eight. Retain
the schedule and generalize that specialization next.

## Arbitrary-stride deferred root scatter

The accepted eval schedule changes coefficient-packing geometry from stride two
and width 256 to stride eight and width 64. The retained host specialization
rejects that shape, returning to the generic per-selector fp128 multiply and
raising root packing from 0.132 s to 0.439 s.

For any base-field D512/K256 packing geometry the same closed form applies. Let
`stride = 512 / partial_width`, `coefficient = 256 * row_parity + hot`, then

```text
position = local_row / 2
bucket   = coefficient / stride
low      = coefficient % stride.
```

Precompute `position_weight * packing_weight[low]` for every position and low
index, then reuse the existing exact limb-plus-wrap accumulator over the runtime
partial width. The new schedule needs only 524,288 combined weights (8 MiB),
and each of its 256 row-block tasks holds 30 x 64 deferred coordinates. Total
selector traffic and arithmetic are otherwise the same as the measured 0.132 s
stride-two route; the greater task count adds only final-coordinate overhead.

Pre-registered prediction: focused stride-two and stride-eight parity remain
exact; root coefficient packing reaches at most 0.20 s; complete T25 opening is
at most 2.74 s and improves by at least 180 ms versus the 2.938 s schedule
parent; route/fallback counters do not change. Reject on a checked-geometry
escape, maximal-skew wrap failure, root packing above 0.24 s, complete gain below
180 ms, or any proof, transcript, evaluation, CPU parity, or verifier mismatch.

Measured result: both stride geometries and the maximal-skew accumulator matched
the generic reference exactly. Root packing reached 0.141 s and complete opening
reached 2.726 s, a 212.1 ms improvement. Proof digest, transcript, evaluation,
CPU parity, verifier, allocation, and route counters remained valid. Retain the
runtime-stride specialization.

## Ring-switch critical-path diagnostic

The retained 2.726 s treatment spends 0.583 s in `ring_switch_build_w`, but that
span joins relation-quotient construction with host body emission followed by the
recursive-commit prefix. Optimizing either branch without knowing which controls
the join can produce no wall-time gain. The compact witness is also uploaded once
per direct sumcheck, but all recorded Metal uploads total only 0.110 s, so even
perfect Stage-1/2 buffer reuse cannot close the remaining 0.517 s to the original
CPU 5x threshold.

Add diagnostic-only nested span buckets for output allocation, group emission,
Z decomposition/emission, E/T emission, relation quotient, relation-row Metal,
recursive inner commitment, and trailing relation-row emission. Run exactly one
T25 Metal shot with otherwise identical code. This does not select or promote a
candidate. It falsifies witness/commit fusion as the next family if the joined
body-plus-prefix branch is not the critical branch, and it falsifies relation
fusion if relation quotient is not the critical branch. Remove or keep the
diagnostic buckets only as harness telemetry after interpreting the result.

Measured result: the exact diagnostic shot completed in 2.759 s with the retained
proof digest, evaluation, transcript, CPU parity, and verifier result. Relation
quotient construction took 0.554 s of the 0.592 s ring-switch span. Group
emission took only 0.060 s, including 0.022 s of Z decomposition, 0.026 s of T
emission, and 0.008 s of Z emission; allocation and trailing R emission were
below 1 ms each. Therefore direct witness-emission fusion cannot materially
improve the active join. The relation branch is the next target, but the body
branch still contains the pipelined D256 recursive prefix commit and will become
visible after roughly 0.15 s of relation improvement.

## Fully Metal ring-switch join

The current relation kernel recomputes six length-1024 transforms of the public
D512 setup matrix on every proof. A previously measured setup-resident transform
removed 165.8 ms of GPU-active work, but it was correctly rejected under the old
schedule because the much larger host-emission/CPU-prefix branch hid the saving.
The new schedule reverses that join: relation quotient is 0.554 s while plain
emission is 0.060 s. The remaining body branch commits 160 complete D256 blocks
through the CPU because the generic Metal recursive kernel currently admits only
D64/D128.

Treat the join as one architectural candidate. Retain the six exact public-matrix
transforms in prepared Metal setup, and extend the existing SIMD recursive
commitment to D256 with its public matrix transform resident as well. No protocol,
schedule, transcript, or proof changes. The timed relation then transforms only
the private centered response; the body branch commits the already-produced i8
prefix on Metal. Both setup transforms are reusable and remain outside the proof
boundary; their bytes count toward RSS. The T25 transformed relation matrix is
1.5 GiB, and the D256 recursive matrix is below 0.5 GiB. The corresponding T28
projection remains well below the 90 GiB cap.

Pre-registered prediction: focused D512 relation and D256 recursive-commit parity
remain exact with second-prewarm cache hits; no implicit CPU call remains for the
D256 prefix; relation GPU-active time falls by at least 130 ms; complete T25 falls
by at least 120 ms and ring-switch build is at most 0.47 s. Reject on any proof,
transcript, evaluation, verifier, or route mismatch, a timed cache construction,
ring-switch above 0.52 s, complete gain below 100 ms, or projected max-scale RSS
above 90 GiB. This candidate is necessary join cleanup, not by itself the 5x bar.

The first treatment was a valid rejection of the implementation, not of the
mechanism. It preserved exact proof parity but regressed to 2.796 s, with a
0.624 s ring switch and 6.766 GB of timed allocation. The 1,610,612,736-byte
allocation increase exactly equals the D512 transform, revealing that Jolt's
public `prepare_opening_backend` entry remained a no-op while the new prewarm
had accidentally been placed only in the streaming-commit preparation helper.
The Jolt proof stack also still assigned its commit cluster to CPU, making the
new D256 route unreachable. The corrected treatment moves transform residency
to the public opening-preparation boundary, assigns recursive commit to the same
prepared Metal backend, and keeps large-witness prefix pipelining enabled for
an accelerator commit cluster. Its tightened hot-route falsifiers are 5.3 GB
timed allocation, 0.47 s relation, 0.52 s ring switch, or 2.626 s complete.

The corrected treatment was exact but also failed. Timed allocation confirmed
that the D512 transform was hot, yet relation quotient improved only from about
0.554 s to 0.525 s. The cached representation reads six 1024-point residue
vectors per column (three times the canonical matrix bytes), while every private
RHS still executes the same ten threadgroup barriers. Removing only the matrix
butterfly arithmetic therefore leaves the controlling synchronization cost.
Routing recursive commit to Metal was worse: its exact D256 SIMD path took
0.516 s and raised complete opening to 2.859 s. Revert both routes.

The next relation design targets the synchronization floor. For degree-511
polynomials `a` and `z`, compute both the cyclic and negacyclic length-512
products. Their coefficient vectors are `low + high` and `low - high`, so the
desired high half is `(cyclic - negacyclic) / 2`. Two specialized 512-point
SIMDgroup transforms have the same transform-point count as one padded
1024-point NTT, but keep sixteen coefficients per lane in registers and replace
threadgroup barriers with SIMD shuffles. Pre-transform both public setup views;
the timed kernel transforms only the small signed RHS, multiplies, inverses, and
writes tiled partials. Pre-register 0.34 s relation, 0.39 s ring switch, and
2.56 s complete; reject above 0.42 s relation or below 120 ms complete gain.

The two-mode treatment was exact and reached 2.627 s, a 99 ms improvement, but
relation remained 0.502 s and therefore missed both promotion gates. Splitting
the product preserved transform-point arithmetic while doubling RHS/setup reads
and threadgroups. The next refinement retains one padded 1024-point transform.
Eight SIMDgroups in a 256-thread group each keep 32 coefficients per lane,
process eight columns, and reduce their 1024 transformed accumulators through
one exact 32 KiB threadgroup tile. This restores single-mode traffic while
removing every per-column threadgroup barrier. Keep the same 0.42 s relation and
120 ms complete-gain falsifiers.

The padded SIMD refinement failed its focused gate before correctness returned:
the 67-column test was still executing after 93 seconds and was interrupted.
Thirty-two transformed values plus thirty-two accumulators per lane force
catastrophic thread-local spills on this GPU. Restore the accepted relation path.

The next candidate attacks the padded proof domains. The retained root uses
outer/open basis three and four fold digits. With the same `(A, B, D) =
(512, 64, 64)` and rank-one geometry, basis four should reduce the response to
three digits. At T25 the modeled root witness falls from 337,246,656 bytes to
about 251.7 MB, crossing below `2^28`; Stage 1 and Stage 2 therefore both halve
their domains. Their retained 0.764 s sum predicts roughly 0.38 s directly,
with additional ring-switch, finalization, and NTT savings. Extend the explicit
large-root constraint to basis fields, regenerate the catalog, and require a
fresh matched CPU anchor because this is a public schedule change. Pre-register
2.15 s complete, with 2.30 s as the rejection ceiling.

The basis-four candidate failed the planner gate before measurement. The
accepted D512/D64 small-block geometry produced no complete row. Independently
relaxing the rank, ring dimensions, and block size also produced no row. The
failure was not only the planner's nondecreasing response-basis search rule:
temporarily admitting a basis drop after the root still produced no complete
schedule, even with every other root constraint removed. Restore the accepted
basis-three/four-digit catalog and the original planner search. This candidate
therefore consumes no matched CPU/Metal benchmark and does not invalidate the
frozen control.

## Heterogeneous streaming route to five times

The retained T25 pair is now the decision baseline: CPU completes in 6.951 s,
Metal in 2.726 s, and the five-times threshold is 1.390 s. Eliminating all
reported GPU-active time would reach only 3.59x; eliminating complete Metal
command-wall time would reach only 3.86x. The remaining campaign therefore uses
an operation-specific pipeline rather than trying to place every operation on
Metal. Backend-only changes preserve the accepted schedule, proof bytes,
transcript order, and verifier.

The target budget is:

| Critical-path component | Retained T25 | Target |
|---|---:|---:|
| Root coefficient packing | 0.141 s | 0.08--0.10 s |
| Root decompose plus A relation | 1.134 s | 0.35--0.45 s |
| Witness emission and ring finalization | 0.197 s | 0.12--0.15 s |
| Stage 1 plus Stage 2 | 0.759 s | 0.40--0.45 s |
| Recursive tail | 0.247 s | 0.16--0.20 s |
| Transcript, allocation, and unbucketed work | 0.249 s | 0.08--0.12 s |

The midpoint is about 1.33 s, or 5.2x. The pessimistic endpoint is about
1.47 s, so no single kernel establishes the goal. The root pipeline and
resident sumchecks are both required.

### Exact boundary and data flow

Jolt retains its existing row-major packed K256 source. Akita owns a typed
`ResidentFoldWitness` whose storage is an aligned shared Metal buffer, whose
logical view is position-major D512 centered coefficients, and whose state
records completed position ranges, checked infinity norm, and the canonical
recursive-witness destination. No full fp128 root polynomial or transposed
selector table is admitted.

The intended execution is:

```text
Metal decompose position chunk
    -> signal disjoint shared range
CPU cached D512 relation consumes that range
    -> reduce one exact partial relation row
Metal or CPU emits balanced Z digits into resident W
    -> recursive commitment consumes complete canonical blocks
join relation row and body
    -> finalize, then reuse resident W in Stage 1 and Stage 2
```

The A relation is additive over position chunks, so chunk partials can be
reduced in canonical field arithmetic without changing the resulting row. A
range becomes CPU-readable only after its Metal completion event; Metal never
rewrites a signaled range. Unsupported shapes fail closed. Deliberate CPU
routes are reported separately from unsupported-operation fallbacks.

### Parallelism and floors

The T25 root has 38 MLE variables. D512 consumes nine coefficient bits; the
accepted root geometry splits the remaining 29 bits into 16 position bits and
13 block bits. The shortest root axis is therefore 8,192, not the total GPU
workload. The decompose kernel launches 65,536 position groups and consumes
1,006,632,960 live selectors. Each selector applies 19 signed challenge terms,
for 19.126 billion 32-bit histogram additions, and writes 128 MiB of centered
output. Occupancy is sufficient; atomic serialization and the one-shot API are
the relevant limits.

The measured M4 Max shared-buffer payload rate is 203.2 GB/s, while a warm
read-plus-write blit reaches 406.4 GB/s. Input plus output traffic alone is only
about 1.13 GB, so the decompose phase's 0.551 s is not a bulk-bandwidth floor.
At the observed update rate it sustains about 34.7 billion histogram additions
per second. A useful redesign must reduce contention or overlap its latency; a
layout that merely adds work to increase occupancy is rejected.

Eight fully independent 512-bin histograms would use 16 KiB of threadgroup
memory. That mapping is nearly isomorphic to the rejected eight-position
SIMDgroup treatment: it has the same total SIMDgroup count, local-memory
footprint, and per-lane atomic work, plus a final reduction. It is no longer the
first candidate. A four-shard 8 KiB variant remains a bounded later question,
but only a short atomic calibration can justify implementing it.

For Stage 1 and Stage 2, the retained schedule begins from a padded `2^29`
domain and enters fp128 tables after the compact prefix. Work below `2^14` is a
negligible fraction of the geometric data pass. Small-round occupancy is handled
by a fixed CPU cutoff or fused tail; the large rounds keep W, equality tables,
and ping-pong storage resident and transfer only each 16-byte challenge.

### Ranked implementation slices

The first two slices below have now been falsified: global D512 passes lost the
threadgroup-local transform advantage, and four-way decompose sharding did not
improve atomic throughput. The current order is therefore:

1. Factor the coefficient-prefix rounds over Akita's native
   `lane x coefficient` domain. A lane owner should apply the lane equality and
   relation weights once after reducing its remaining coefficient pairs rather
   than once per pair.
2. Emit Z digits and recursive-commit blocks from the resident centered buffer
   without another full allocation or canonical copy.
3. Retain the compact witness allocation across Stage 1 and Stage 2, but do not
   claim more than the measured 0.110 s whole-proof upload ceiling for this
   change. Residency is supporting work, not the main sumcheck mechanism.
4. Cache challenge-independent relation-weight structure, retain per-level
   D64/D128 Metal and D256 CPU routing, and remove transient allocation from the
   remaining serial tail.

Only if these unchanged-protocol slices stabilize above 1.390 s may a radix-four
sumcheck round be considered. It must be isolated because it changes proof and
verifier behavior and requires a fresh matched CPU anchor. Activity masks may
improve naturally sparse Jolt traces, but the dense T25 fixture remains the
five-times stress case.

### First resumed candidate: explicit CPU D512 relation

The first candidate changes only the eligible rank-one D512 A-relation route.
It records a planned CPU operation rather than a fallback and calls the already
prepared CPU relation kernel. Focused CPU/Metal relation parity is the exactness
gate. Reuse the frozen T25 control and run one Metal treatment only after that
test passes.

Prediction: relation quotient is at most 0.46 s, ring-switch build at most
0.50 s, and complete opening at most 2.62 s with unchanged proof, transcript,
evaluation, verifier result, allocation envelope, and schedule digest. Reject
if the route is reported as a fallback, relation exceeds 0.50 s, complete gain
versus 2.726 s is below 80 ms, or any correctness guard changes. A rejection
does not invalidate streaming; it means overlap needs an isolated CPU resource
policy rather than the default Rayon pool.

Measured result: the deliberate CPU route was exact but decisively slower. The
270,508-byte proof, digest, claimed evaluation, transcript, CPU parity, and
verifier all matched, and telemetry reported one planned CPU call covering
33,554,432 scalar work units. Complete opening nevertheless rose from 2.726 s
to 2.938 s. Relation quotient rose to 0.699 s and the joined ring-switch span to
0.738 s, triggering every performance falsifier. GPU-active time fell from
0.792 s to 0.701 s, so the regression is host work on the critical path rather
than a slower Metal kernel. The full route also retained pre-existing CPU
fallbacks outside this planned call.

The CPU-only anchor's 0.427 s ring-switch span was not a transferable estimate
for a cold mixed route. The CPU relation lazily requests cyclic and negacyclic
setup transforms through `with_shared_ntt`, while the Metal parent owns a
different prepared transform path. Its first use therefore combines public
cache preparation, the private response transform, and the quotient matvec.
Do not retain the serial planned route. A streaming candidate must start the
public CPU cache preparation concurrently with earlier Metal root work, expose
that preparation in the timed boundary, and consume exact completed position
chunks without competing with the prefix branch for the default Rayon pool.

### Second resumed candidate: exact decompose/relation pipeline

The serial measurement fixes a tighter bound for the next candidate. Its root
decompose and CPU A-relation spans total 0.576 + 0.699 = 1.275 s. The relation
contains 0.241 s of cold public NTT preparation, leaving about 0.458 s of
private-response transform and matvec work. Starting cache preparation and
Metal decomposition together, then evaluating exact relation windows as their
position ranges complete, has an ideal joined time of about 0.699 s. Moving
the public preparation start ahead of root decomposition lowers the ideal join
to the 0.576 s Metal span. Relative to the rejected 2.938 s treatment, these
are complete-opening floors of 2.362 s and 2.239 s respectively; 2.30 s is not
a credible ceiling unless public preparation overlaps earlier root work.

The candidate uses four position ranges. Each Metal command writes a disjoint
position-major D512 output, after which a host worker may read but no longer
mutate that range. The worker builds the full row-major cyclic and negacyclic
public prefixes once, evaluates each matrix window at its original column
offset, converts the partial quotient to the canonical field, and adds the
four partials. The accepted folded witness carries this derived quotient to
the later relation builder. Its canonical centered coefficients remain the
source of truth; CPU and unsupported shapes carry no derived value. The later
consumer checks ring dimension, matrix width, row count, and coefficient count
before bypassing the ordinary relation kernel. No transcript or proof field is
added.

Pre-register unchanged proof bytes, proof digest, evaluation, transcript,
verifier result, schedule digest, and commitment digest. A focused test must
show that window partials sum to the ordinary full CPU relation for uneven
ranges and that cached and uncached witnesses compare canonically. The single
T25 treatment must report one planned CPU A-relation and no additional fallback.
Without earlier prewarm, expect 2.36--2.48 s complete and reject above 2.55 s,
above 0.82 s for the joined decompose/relation path, or below 0.30 s overlap
savings relative to the rejected serial treatment. If that boundary passes,
start the same public preparation from the schedule-derived prewarm phase and
target 2.20--2.35 s; do not hide preparation outside the timed `prove_batch`
boundary.

Measured result: the implementation is exact but the mechanism is rejected.
The proof, transcript, evaluation, commitment and schedule digests, CPU parity,
and verifier all match. Complete opening was 2.835 s, above both the retained
2.726 s parent and the 2.55 s rejection ceiling. Root decompose rose from
0.576 s in the serial hybrid treatment to 0.777 s, while the later relation
span fell from 0.699 s to 0.471 s. The approximately 0.20-second root increase
therefore replaced, rather than overlapped, the approximately 0.23-second
relation decrease.

Aggregate GPU-active time fell to 0.693 s and upload time to 0.089 s, so input
residency and repeated upload are not the observed limiter. Source inspection
also rules out lost B/A fusion: the protocol invokes A and sliced B relations
separately. The actual fixed cost is that every A window invokes a complete
quotient kernel and reconstructs its CRT/NTT accumulators into canonical field
rings. In particular, the last completed Metal range still pays a full
finalization on the critical path.

Two invalid executions produced no timing artifacts but localized a
candidate-specific failure. D512 arithmetic placed roughly 80 KiB of fixed
scratch into recursive Rayon bridge frames, while cold cache preparation nests
approximately 0.56 MiB and 0.72 MiB fixed-array frames. A bounded private pool
with 8 MiB stacks passed full-width parity, including the exact i16-tail route.
That workaround is not retained: removing the rejected windowed pipeline also
removes its private pool and leaves the parent capacity-parallel quotient path
unchanged.

### Third resumed candidate: incremental NTT-domain A relation

Keep the same four readiness ranges, but replace field-valued window results
with one opaque CPU relation accumulator. Each range transforms its centered
response and adds products into persistent cyclic and negacyclic CRT/NTT rows
at the original matrix offsets. The exact i16 tail, when required, is retained
in parallel accumulators. Only after the final range does the accumulator run
one inverse/reconstruction and form the canonical quotient row. The proof-level
derived cache and all validation remain unchanged.

This removes three redundant window finalizations and, more importantly, makes
the final-range work proportional to its coefficients rather than a complete
quotient call. Pre-register root decompose at most 0.64 s, the later relation
span at most 0.50 s, and complete opening at most 2.66 s, with unchanged proof,
transcript, evaluation, verifier, schedule, and no added fallback. Reject the
mechanism above 2.70 s or if telemetry shows more than one final
reconstruction. If it passes, start the same public NTT preparation at the
timed opening boundary so its 0.20 s cold cost overlaps coefficient packing;
that is a separate mechanism.

Measured result: reject and revert. Correctness remained exact, but complete
opening rose to 7.719 s and root decompose/fold alone rose to 5.678 s. The
model's implicit one-reconstruction assumption was false for this f128/D512
envelope. Exactness capacity divides the 65,536-column relation into many CRT
chunks. The retained kernel schedules those chunks in parallel; the persistent
implementation consumed them sequentially while preserving only the readiness
boundary state. GPU-active, command-wall, and upload counters remained near the
retained range, confirming a host scheduling regression. Do not retry this
mechanism or tune its partition count. Isolated CPU A-window streaming is no
longer the next route to five times; a useful root change must preserve
capacity-chunk parallelism or compute more of the relation on Metal.

Both rejected A-window implementations and their derived-witness, chunked
dispatch, Rayon-pool, and large-stack support have been removed. The active
implementation is again the exact 2.726 s retained parent. Benchmark route
telemetry and the rejection artifacts remain; no candidate-only protocol or
backend surface remains in the production path.

### Fourth resumed candidate: globally batched D512 transforms

The rollback initially stopped one layer too early and left the first rejected
planned-CPU route active. The journaled six-prime Metal partial, reduction, and
CRT-reconstruction kernels have now been restored. The 67-column focused test
passes exact CPU parity under RequireMetal; no new large benchmark was needed
to establish this source-state correction.

The retained partial kernel launches 6,144 threadgroups at T25. Each group owns
one 64-column tile and one CRT prime. For every column it converts 512 fp128
setup coefficients and 512 centered coefficients, performs two padded
length-1024 forward NTTs, and crosses eleven threadgroup barriers. Across the
phase this is approximately:

    canonical matrix reads      65,536 * 6 * 512 * 16 B = 3.00 GiB
    centered RHS reads           65,536 * 6 * 512 *  4 B = 0.75 GiB
    Montgomery products         about 6.2 billion
    threadgroup barriers         1,024 * 6 * 64 * 11 = 4.33 million

The earlier setup-resident treatment removed the public transform but retained
the same column-local RHS barriers, reaching only about 0.525 s. The two-mode
SIMD treatment reached about 0.502 s, while the 32-value-per-lane padded
treatment spilled catastrophically. The next design therefore changes the
execution geometry rather than adding another per-thread register transform.

Precompute the public six-prime frequency matrix at setup preparation. During
the proof, convert the centered RHS once into six padded residue batches, run
ten global DIF NTT stages through ping-pong buffers, and launch a
frequency-major tile reduction:

    centered D512 RHS
        -> six-prime padded conversion
        -> 10 globally batched NTT stages
        -> frequency-major (matrix * RHS) reduction over 64-column tiles
        -> retained six-prime tile reduction and one inverse NTT
        -> retained Garner reconstruction of the high 512 coefficients

This preserves the existing tile partials and one final reconstruction, so it
does not serialize the capacity-safe exactness partitions that defeated the
incremental CPU candidate. At T25 one full transformed view is 1.50 GiB. Ten
out-of-place stages move about 30 GiB, and conversion plus pointwise reduction
adds roughly 6 GiB. At 406 GB/s the traffic floor is about 90 ms; allowing
roughly half of peak bandwidth and the observed integer-Montgomery rate gives a
credible 0.24--0.36 s relation range. A passing implementation will shard the
batch buffers before T28 so temporary storage is bounded independently of
trace size; the frequency matrix remains setup state and the T28 projection
must stay below 90 GiB.

Pre-register exact focused parity across both a 64-column tile boundary and a
batch-shard boundary, no fallback on the qualified route, one inverse/CRT
finalization, and no setup transform inside timed prove_batch. Run one T25
treatment only after that gate. Promote only if relation quotient is at most
0.38 s, ring switch is at most 0.46 s, complete opening improves by at least
150 ms from 2.726 s, proof/transcript/evaluation/verifier and schedule digests
match, and projected T28 peak memory is at most 90 GiB. A miss rejects the
global-pass mechanism; it does not justify another CPU-window overlap.

Measured result: exact but rejected. The proof, transcript, evaluation,
verifier, commitment, fixture, and schedule digests all match. Complete opening
was 2.768 s, a 42.1 ms regression from the 2.726 s parent. Relation quotient
rose from about 0.554 s to 0.620 s, ring switch reached 0.656 s, and GPU-active
time rose by about 237.6 ms. Backend preparation built the bounded public cache
in 0.445 s outside prove_batch, peak RSS was 7.10 GB, and no planned CPU route
was used. The 0.38 s relation and 150 ms complete-gain gates both failed.

The analytical traffic floor was too optimistic because it treated the warm
blit rate as attainable for ten dependent integer-NTT passes. In practice the
global schedule writes and rereads roughly 30 GiB while executing the same
two-billion private butterflies; the retained threadgroup kernel keeps each
1024-point transform in 8 KiB of local memory. Removing its barriers did not
repay the lost locality. Remove the global kernels, bounded frequency cache,
and Jolt prewarm hook, and restore the local six-prime parent without another
large measurement.

### Fifth resumed candidate: local radix-four D512 NTT

Retain the parent kernel's 8 KiB matrix/RHS threadgroup arrays and 64-column
tiles. Fuse consecutive DIF radix-two stages. One 256-thread group owns 256
radix-four butterflies; each thread loads four local values, computes the two
dependent radix-two stages in registers, and writes four outputs before one
barrier. Five fused stages reproduce the current ten-stage transform exactly.
The load plus transform barrier count falls from eleven to six per column,
thread count falls from 512 to 256, matrix/RHS traffic is unchanged, and no
large register array or global ping-pong buffer is introduced.

This is the locality-preserving counterpart to the rejected global design.
Arithmetic count and CRT/reconstruction are unchanged, so the only claimed
mechanisms are fewer barriers and more resident threadgroups. Pre-register
focused exact CPU parity across 67 columns, no qualified-route fallback, and
one T25 treatment. Promote only if relation quotient is at most 0.44 s,
ring switch is at most 0.50 s, complete opening improves by at least 120 ms
from 2.726 s, proof/transcript/evaluation/verifier and schedule digests match,
and peak memory remains within the retained envelope. Otherwise restore the
radix-two kernel without another large run.

Measured result: exact but rejected. Complete opening reached 2.622 s, a
103.8 ms improvement, but missed the 120 ms promotion gate. Relation quotient
was 0.524 s and ring switch was 0.560 s, both above their ceilings. GPU-active
time fell by only about 18 ms from the retained parent. The fused transform
therefore confirms that barriers are secondary to the six-prime arithmetic and
matrix/RHS traffic. Restore radix two without another large run and stop
iterating on D512 transform geometry in this campaign.

### Sixth resumed candidate: four-shard decompose histogram

The retained root decompose launches one 256-thread group per position and
sends all eight SIMDgroups into the same 512 atomic bins. At T25 that is
19.126 billion signed additions into only 2 KiB of threadgroup state. Give each
pair of SIMDgroups its own 512-bin shard, using 8 KiB total, then sum four
shards once when writing the 512 centered coefficients. Input traversal,
challenge terms, output order, and arithmetic remain unchanged.

The extra initialization and final reduction are negligible beside the atomic
stream. Four shards should reduce the dominant collision domain without the
16 KiB footprint and cross-position remapping of the rejected eight-position
treatment. Pre-register focused exact CPU parity, zero decompose fallback, and
one T25 treatment. Promote only if root decompose is at most 0.42 s, complete
opening improves by at least 120 ms from 2.726 s, all proof/transcript/evaluation
and verifier guards match, and the retained memory envelope holds. Otherwise
restore the one-histogram kernel without another large run.

Measured result: exact but rejected. Root decompose was 0.561 s, about 10 ms
slower than the retained one-histogram kernel, and complete opening was
2.640 s, only 86.6 ms below the parent. Both promotion gates failed. Splitting
SIMDgroups across four collision domains did not change aggregate atomic
throughput enough to repay the extra 8 KiB initialization and final shard
reduction. Restore the single histogram and stop treating atomic collision
width as the main decompose limiter.

### Stage-1/Stage-2 first-principles reassessment

The retained T25 parent spends 0.322 s in Stage 1 and 0.429 s in Stage 2, or
0.751 s together. The matched CPU phases total 2.157 s, so this isolated Metal
path is already 2.87x faster. Eliminating both phases entirely would still leave
a 1.975 s opening, above the 1.390 s five-times ceiling. Sumcheck work must move,
but it cannot replace the root relation and serial-tail work.

Cross-stage buffer reuse is not the missing mechanism. Every Metal upload in
the complete retained proof, including root and ring-switch operations, totals
only 0.110 s. Stage 1 and Stage 2 do upload the same compact digit allocation,
but even a zero-cost handoff cannot account for the required 0.30--0.35 s
reduction in their combined budget.

The large rounds are also not occupancy-limited. Both stages start at `2^29`,
cap their launches at 4,096 groups of 256 threads, and give each thread many
pairs in the dominant rounds. The late small tables contain a negligible share
of the geometric work. The useful remaining structure is algebraic:

* the direct kernels always carry four coefficient lanes. A basis-four route
  could drop two, but the measured Jolt schedule in fact exercises basis eight;
* every partial kernel reduces those lanes through four 256-entry threadgroup
  arrays and nine full barriers, although the accepted prefix kernel already
  demonstrates an exact one-barrier SIMDgroup reduction;
* the first six variables are coefficient variables over a 64-coordinate
  block. During those rounds, equality weights factor into a small coefficient
  table and a lane factor, while Stage-2 relation weights factor into alpha and
  a lane weight. The generic flat-pair kernels recompute those outer products
  for every coefficient pair.

The coefficient-domain design assigns one logical owner to a relation lane.
For each round it reduces the remaining coefficient pairs locally, then applies
the lane equality and ordinary-relation weights once. For the Stage-2 suffix
after the existing two-round prefix, the remaining pair counts per lane are
`8, 4, 2, 1`. Ordinary relation weighting falls from four full products per pair
to two per pair plus two per lane: 60 products become 38 across those rounds.
The norm term obtains a similar outer-factor saving. This keeps the canonical
round order and transcript; it is preferable to a three-round grid, whose 54
compact passes already regressed in measurement.

Before adding that new address schedule, run one smaller code-shape treatment.
Specialize the existing basis-four path to two live coefficient accumulators and
use SIMDgroup reductions for direct range and relation partials. The dominant
pair arithmetic remains, so this candidate is not projected to close the phase
budget by itself. It tests whether unused output work and reduction barriers are
material on the compiled M4 path without introducing a new protocol or buffer.

Pre-register exact focused Stage-1 and Stage-2 parity, unchanged proof,
transcript, evaluation, verifier, schedule, fallback count, and memory envelope.
Predict Stage 1 at 0.23--0.28 s, no Stage-2 regression, and complete opening at
2.61--2.67 s. Promote only if Stage 1 is at most 0.28 s, aggregate GPU-active
time falls by at least 30 ms, and the complete gain from 2.726 s is at least
60 ms. Otherwise restore the reduction code without another large run and move
directly to the product-domain coefficient-prefix kernel.

Measured result: reject and restore the parent. Focused Stage-1 and Stage-2
parity and every integrated correctness guard passed. Complete opening was
2.638 s, an 87.7 ms gain, and Stage 2 fell by 19.6 ms to 0.409 s. Stage 1,
however, rose slightly to 0.326 s rather than reaching 0.28 s, and aggregate
GPU-active time improved by only 16.6 ms rather than 30 ms. The apparent
complete gain is therefore mostly unrelated host variance. More importantly,
the Stage-1 result falsifies the assumed basis-four narrowing for this schedule:
the four-lane basis-eight arithmetic remains live. Reduction barriers are a
small Stage-2 effect, not the missing phase mechanism. Remove the SIMDgroup
reduction helper and proceed to coefficient/lane factorization without another
large run.

### Coefficient-lane Stage-2 suffix

The retained Stage-2 two-round prefix already assigns a logical thread to a
relation lane and moves the ordinary lane weight outside its coefficient loop.
It then resumes the flat-pair kernel for round two and the remaining coefficient
rounds. At that boundary the per-lane coefficient-pair counts are `8, 4, 2, 1`.
The flat kernel recomputes the lane weight and the high equality factor for every
pair even though both are constant for the lane.

The treatment extends lane ownership from the accepted prefix through the
coefficient suffix. One thread processes the contiguous coefficient block for a
lane. It accumulates three equality-weighted norm terms, two alpha-weighted
ordinary relation terms, and the structured-linear terms in registers. It then
applies `E_second`, the Gruen linear scalars, and the resident lane weight once.
The compact resume and compact-to-field materialization use the same address
schedule; field rounds fold each lane's adjacent coefficients in place. The
ordinary lane round resumes the existing flat kernel when the coefficient count
reaches one.

For eight pairs, the virtual term falls from roughly eleven full fp128 products
per pair to six per pair plus seven per lane. The ordinary relation falls from
four products per pair to two per pair plus two per lane. Later rounds have less
reuse, so the candidate is admitted only while at least two coefficient values
remain. Input/output traffic and resident allocation are unchanged; contiguous
per-thread coefficient blocks trade SIMD-coalesced scalar loads for cache-line
reuse without adding a transpose or full table.

Pre-register exact direct Stage-2 parity and unchanged integrated proof,
transcript, evaluation, verifier, schedule, routes, fallback count, and memory.
Predict Stage 2 at 0.34--0.38 s and complete opening at 2.61--2.67 s. Promote
only if Stage 2 is at most 0.38 s, aggregate GPU-active time falls by at least
40 ms, and complete opening improves by at least 60 ms from 2.726 s. Reject on
register spilling, a Stage-2 time above 0.40 s, or any correctness/route change;
restore without another large run and reassess a coefficient-subgroup mapping
rather than tuning workgroup counts.

The focused proof was exact, but the `T=2^25` treatment rejected the mapping.
Stage 2 measured 0.427760 s versus the 0.429051 s parent, and aggregate
GPU-active time fell by only 16.5 ms. The favorable 2.643188 s complete time is
therefore host-side variation, not evidence for the candidate mechanism. The
saved common products were offset by seven live fp128 accumulators and strided
coefficient access. The code was removed and retained parity rechecked. Any
successor must keep pair-coalesced reads and factor lane constants within
aligned SIMD subgroups; thread-per-lane ownership is closed.

### Register-tiled D512 relation NTT

The two rejected SIMD relation variants do not close register-tiled transforms.
The padded variant kept 32 transformed values and 32 accumulators per lane and
spilled catastrophically. The locality-preserving radix-four variant retained
threadgroup exchange at six stages and improved the exact relation by about
59 ms. A 256-thread radix-2 mapping instead keeps only four values per thread.
The first two forward stages (`512`, `256`) are register butterflies; stages
`128`, `64`, and `32` alternate two 1,024-word threadgroup buffers; the last five
stages use SIMD shuffle butterflies. The inverse reverses that schedule. Matrix
and response transforms carry separate ping-pong buffers so they share the same
three exchange barriers plus one end-of-column reuse barrier. The footprint is
16 KiB and permits two resident groups under
the 32 KiB limit; live state is four matrix values, four response values, and
four accumulators per thread.

This preserves the six-prime CRT, length-1,024 DIF/DIT ordering, 64-column tile,
canonical matrix, quotient, proof, transcript, and verifier. It changes neither
the transform count nor global partial traffic. It targets the observed
synchronization floor without the spill mechanism of the earlier SIMD attempt.
The partial pass falls from ten full-group barriers per column to four, while
the small inverse reduction likewise falls from ten to three.

Pre-register focused exact CPU parity across 67 columns and one unchanged T25
treatment. Predict relation quotient at 0.40--0.47 s, ring-switch build at most
0.50 s, aggregate GPU-active improvement of at least 60 ms, and complete opening
at most 2.606 s (at least 120 ms below the 2.726 s parent). Reject on any
correctness, schedule, route, allocation, or memory change; relation above
0.49 s; GPU-active improvement below 45 ms; or complete gain below 100 ms.
Restore without another large run if rejected.

The synchronized treatment was exact but rejected. Relation quotient measured
0.530737 s and ring-switch build 0.567322 s, missing the 0.49 s and 0.50 s
ceilings. Aggregate GPU-active time fell by only 36.0 ms, below the 45 ms
falsifier. Complete opening reached 2.610897 s, 115.2 ms below the parent, but
the phase and device counters show that most of that difference is unrelated
host variation. Four barriers instead of ten do not control the transform; the
unchanged modular butterfly and canonical-matrix work dominates. Remove the
register-tiled kernels and restore the exact 512-thread radix-two parent.

### Tile-local five-prime D512 reconstruction

The six-prime route currently adds every 64-column NTT tile in the residue
domain and reconstructs once. Its exactness bound therefore prices all 65,536
T25 columns at once:

```text
2 * width * 512 * floor(q / 2) * ||z||_inf < product(primes).
```

Five of the existing approximately 30-bit primes provide just under 150 bits
and cannot cover that global sum. They do cover one 64-column tile for every
realized response with `||z||_inf <= 127`. The fixed sparse one-hot response is
expected to remain inside this bound. Reconstructing each tile modulo fp128 and
then adding the tile results in fp128 is algebraically identical to the global
integer reconstruction, while making the exactness width 64 instead of 65,536.
No protocol value or transcript event changes.

The treatment runs five forward transforms per column instead of six. It also
runs an inverse transform and CRT reconstruction per tile, followed by one
field reduction across tiles. At T25 this changes six global inverse transforms
to 5,120 tile inverses, only 1.56% as many transforms as the 327,680 forward
matrix/response pairs. It adds about 20 MiB of tile residues and 8 MiB of fp128
tile output. T28 scratch scales to about 160 MiB of residues plus 64 MiB of tile
output, well inside the 90 GiB envelope. The hard route check rejects responses
above the exact five-prime tile capacity rather than silently weakening CRT
correctness.

Pre-register focused exact parity across a tile boundary and one unchanged T25
treatment. Predict relation quotient at 0.43--0.50 s, ring-switch build at most
0.54 s, aggregate GPU-active improvement of at least 45 ms, and complete
opening at most 2.646 s (at least 80 ms below the 2.726 s parent). Promote only
if every proof, transcript, evaluation, verifier, schedule, route, and memory
guard is unchanged, the response takes the five-prime route, relation is at
most 0.50 s, GPU-active improves by at least 40 ms, and complete improves by at
least 60 ms. Otherwise restore the global six-prime reconstruction without a
second large run.

Measured result: reject the fixed 64-column geometry before assessing kernel
performance. Focused exact parity passed, and the integrated proof, transcript,
evaluation, and verifier remained exact, but the fixed T25 response exceeded
the five-prime tile bound of 127. The runtime correctly selected CPU for the A
relation: fallback calls rose from 42 to 43 and CPU-tail work rose by exactly
33,554,432, equal to `65,536 * 512`. Relation quotient consequently regressed to
0.648186 s and complete opening to 2.718797 s. The lower GPU-active counter does
not measure the candidate because its relation kernels did not run. Do not
weaken the bound or count this as a Metal result; any successor must derive its
tile width from the exact CRT capacity and realized response norm.

### Capacity-derived five-prime D512 tiles

The fixed-width failure does not falsify tile-local reconstruction. For the
first five primes and fp128 D512, the exact maximum tile width is
`floor(8190 / ||z||_inf)`, capped at 64 and at the live column count. Select that
width at the existing runtime shape boundary and require the dispatch params to
match it. This admits every response through norm 8,190, including the public
terminal-response envelope used by the fixed route, while preserving the same
strict CRT inequality. It also exposes the actual selected geometry through
the deterministic transient allocation count; no benchmark-only protocol or
wire field is needed.

Forward work remains five transforms per column. Per-tile inverse work is a
fraction `1 / width` of that forward count per prime. Width 16 therefore adds
6.25% inverse work while removing 16.7% of forward residue channels; width 8 is
roughly the analytical break-even region after CRT and field reduction. The
worst capacity-safe width one route is correct but not expected to be fast and
must fail the phase gate rather than silently falling back. At T28, even the
width-one scratch upper bound remains below the 90 GiB process envelope, while
the selected fixed-workload width should be identical to T25 because the
response distribution and public bound are schedule-derived rather than trace
length-derived.

Pre-register one focused parity case with norm 2,570, forcing width three, then
one unchanged T25 treatment. Require no additional fallback or tail work,
exact proof/transcript/evaluation/verifier/schedule parity, and memory within
the envelope. Predict selected width at least eight, relation quotient at most
0.52 s, ring switch at most 0.56 s, GPU-active improvement at least 30 ms, and
complete opening at most 2.676 s. Promote only if relation is at most 0.52 s,
GPU-active improves by at least 25 ms, and complete improves by at least 50 ms
from the 2.726 s parent. A narrower or slower exact route is rejected without a
repeat treatment.

Measured result: the qualified Metal route was exact and selected 49-column
tiles, inferred exactly from its 1,338 tile allocations. This places the fixed
response norm between 164 and 167. Proof, transcript, evaluation, verifier,
schedule, fallback count, tail work, and memory all matched the parent.
Relation quotient improved by about 28 ms to 0.526129 s and complete opening by
82.4 ms to 2.643772 s. The mechanism still rejects: relation missed 0.52 s,
ring switch was 0.563364 s, and GPU-active time improved only 17.9 ms rather
than 25 ms. The complete delta is not supported by the device counter. Restore
global six-prime reconstruction. Removing one sixth of the forward residue
channels is worth only about 18 ms of aggregate device time on this path, so
additional CRT scheduling refinements cannot close the remaining phase budget.

### Bit-sliced root decompose histogram

The retained root kernel expands each of 1,006,632,960 live selectors into 19
threadgroup atomic updates. That 19.126-billion-update stream is not required by
the protocol. The runtime root batch uses coefficients in `{+1,-1,+2,-2}`.
Store each challenge as four 512-bit support masks. For a selector
value `s`, a word of the negacyclic rotation is obtained from two adjacent mask
words; the wrapped destination prefix swaps the positive and negative masks.
One 32-bit operation then represents contributions to 32 histogram bins.

Use 16 logical shards per position and 16 threads per shard, one thread per
32-bin word. A shard scans every sixteenth selector. It accumulates the rotated
positive and negative masks into one 16-plane two's-complement bit-sliced
counter: magnitude-one addition/subtraction starts at plane zero and magnitude
two starts at plane one. Each thread decodes its 32 signed counts into a 16 KiB
threadgroup `short` table. After one barrier, 256 threads reduce the 16 shards
into the canonical 512 signed coefficients. There are no atomics and no output
layout, proof, transcript, or verifier changes.

At T25 each shard sees 960 selectors, so ten counter bits are sufficient; the
16-plane representation also covers the projected T28 task count. The challenge
masks occupy about 1.88 MiB versus 0.42 MiB for sparse positions and signs and
are reused by all 65,536 position groups. Lane reads remain one per selector.
The kernel replaces 291,840 scalar atomics per position with 245,760 rotated
word updates whose average carry/borrow depth is small, plus 8,192 local count
decodes. It trades cache-resident mask traffic and integer logic for the entire
atomic expansion.

Pre-register the existing independent CPU/Metal packed-decompose parity test
and one unchanged T25 treatment. Require exact proof, transcript, evaluation,
verifier, schedule, fallback count, and memory parity. Predict root decompose at
0.18--0.32 s, aggregate GPU-active improvement at least 180 ms, and complete
opening at most 2.48 s. Promote only if root decompose is at most 0.34 s,
GPU-active improves by at least 140 ms, and complete improves by at least 180 ms
from 2.726 s. Reject any coefficient outside `+/-1,+/-2`; restore the
sparse atomic parent without a repeat treatment if the counter spills or any
performance gate fails.

The first integrated attempt was invalid before measurement: the independent
`+/-1` parity case passed, but the fixed runtime batch contains a magnitude-two
coefficient and the encoder rejected it before producing a proof or JSON
record. Extend the same representation to four masks (`+1`, `-1`, `+2`,
`-2`). Magnitude-two carry or borrow begins at counter plane one, so this does
not duplicate an update or change the counter width. Strengthen focused parity
to mixed magnitudes and retry the original treatment and falsifiers; this is a
shape correction, not a second measurement of a completed candidate.

Measured result: reject and restore the atomic parent. Mixed-magnitude focused
parity and the full proof were exact, with unchanged transcript, evaluation,
verifier, schedule, fallback count, and memory class. Root decompose instead
rose from 0.551 s to 5.315516 s; aggregate GPU-active time rose to 5.541354 s
and complete opening to 7.402580 s. Sixteen live counter planes, four rotated
mask classes, divergent carry/borrow propagation, and shard decoding create
far more integer/register-private work than the M4's fast threadgroup atomic
path. This misses every falsifier by an order of magnitude. Do not tune shard
count or mask layout: the representation expansion itself is wrong for this
device.

### SIMD-owned i16 root histograms

The bit-sliced failure does not establish that atomics are required. Preserve
the original sparse 19-term traversal and give each of 32 SIMD lanes a private
512-bin signed-i16 histogram in threadgroup memory. The 32 histograms occupy
exactly 32 KiB. One SIMDgroup scans the position's selectors in 32 shards; each
lane performs ordinary indexed load/add/store operations only in its own 1 KiB
slice. After one barrier, 256 threads sum the 32 shards into the canonical i32
output. This removes every atomic while avoiding mask rotation, bit-plane
counters, divergent carry propagation, and per-thread private arrays.

At T25 each owner sees 480 selectors and at most `480 * challenge_l1` magnitude
in one bin. Compute the maximum challenge L1 norm on the host and cap each task
chunk so `ceil(tasks_in_chunk / 32) * challenge_l1 <= i16::MAX`. T25 takes one
chunk. Larger shapes dispatch multiple independent position/chunk groups into
i32 partials and run one field-free reduction kernel, so T28 correctness does
not depend on an empirical response norm. One T25 partial plane is 128 MiB; the
T28 projection remains inside 90 GiB.

The first occupancy calculation above omitted residency and therefore
overstated the candidate. A 32 KiB static allocation admits only one such
threadgroup per M4 GPU core. Seven of its eight SIMDgroups are idle during the
update stream, whereas the retained 2 KiB kernel can keep roughly four
256-thread groups, or 32 updating SIMDgroups, resident. The candidate therefore
needs an ordinary threadgroup load/add/store to beat the current low-contention
atomic by about 32x, before paying initialization, shard reduction, and any T28
chunk reduction. The four-shard result already showed that collisions are not
the controlling atomic cost. A 32x local-memory instruction advantage is not a
credible precondition, so reject this candidate analytically without compiling
or spending a T25 treatment. The single atomic histogram remains the root
decompose floor on this M4 path.

### Uncovered-route and phase-boundary audit

The retained complete route still reports 42 CPU fallbacks and 1,222,052,736
scalar tail work units. That count is not a usable optimization signal: digit
row calls add both work and call counts, while recursive/suffix opening views
add call counts without work or elapsed time. Before another kernel candidate,
separate digit-row CPU time from recursive/suffix fold and batch time, and
record each operation's `(D, rows, columns, batch, log_basis)` shape in bounded
aggregate telemetry. This is route instrumentation, not an optimization and
does not justify a repeated CPU control.

The audit has two falsifiers. If all delegated CPU work is below 0.10 s, porting
it cannot materially move the 2.726 s opening and exists only as final
fail-closed cleanup. If it is material, the next candidate must cover the
dominant shape through an operation-level Akita route and predict complete-call
time from its measured CPU duration; a raw work-unit count is insufficient.
In parallel, classify whether the root relation, response emission, and first
recursive commitment serialize or overlap at their transcript boundaries. The
unchanged-protocol measured floor is already at least 1.85 s from root
decompose, D512 relation, and the two direct sumchecks alone, so a later
protocol/schedule candidate is now analytically permitted by the goal contract.

Measured result: the attributed recursive/suffix view fallbacks total only
43.739 ms. The dominant item is one D256 decompose at 30.559 ms; all D64
evaluate/decompose work totals about 13.18 ms, and the batch adapters return
their per-source route almost immediately. Porting this family cannot move the
complete proof materially and is deferred to final fail-closed cleanup.

The 25 unattributed calls carry 1,222,052,736 work units. Source inspection
identifies 1,006,632,960 of them as the already-specialized packed-root
coefficient-packing route, whose independent span is 145.843 ms. The remaining
215,419,776 units are small recursive ring-relation and adapter routes; the
entire ring-switch span beyond the 559.026 ms root relation is only about
39.3 ms in this shot. Therefore the large work-unit number is not hidden
seconds of CPU work. Reclassify deliberate hybrid routes as planned CPU during
production cleanup, but do not port them as the next speed mechanism.

The audit treatment was exact (proof, transcript, evaluation, CPU parity, and
verifier), completed in 2.764247 s, and retained the schedule and memory class.
Its timing is diagnostic, not a new parent. Remove the temporary record vector
after preserving these aggregates. The decisive lower bound is now measured:
root decompose 0.565363 s + root relation 0.559026 s + Stage 1 0.317381 s +
Stage 2 0.427185 s = 1.868955 s. Those phases alone exceed the 1.390292 s goal
by 478.7 ms, so unchanged-protocol microkernels cannot establish five times.

### Quotient-free committed A relations

The measured phase floor admits a protocol change, and the 559 ms A-quotient
is the first protocol-created cost. Recursive folds currently commit
`[Z | E | T | R]`, where the A row of `R` is the high half of the ordinary
polynomial product. After that commitment is absorbed, the transcript samples
`alpha` and Stage 2 checks

```text
A(alpha) * z(alpha) - c(alpha) * t(alpha)
    - (alpha^D + 1) * r(alpha) = 0.
```

The quotient only converts a reduced-ring equation into an ordinary-polynomial
identity. Akita's terminal verifier already checks the corresponding A equation
directly in `F[X]/(X^D+1)` without quotient rows. At a recursive level, retain
the existing load-bearing order--commit the next witness before sampling
`alpha`--and apply the random linear functional

```text
L_alpha(sum_k p_k X^k) = sum_k p_k alpha^k
```

to the reduced A-row residual. A nonzero residual of degree below `D` passes
with probability at most `(D-1)/|F|`, no worse than the current degree-below-
`2D` quotient identity. The verifier derives the same weights from public
setup, challenges, and `alpha`; no prover-supplied quotient evaluation is
admitted.

For a public ring `a`, all coefficient weights needed for a private ring `z`
are obtained in linear work. If

```text
s_j = L_alpha(a * X^j mod (X^D + 1)),
```

then

```text
s_0 = a(alpha)
s_(j+1) = alpha * s_j - (alpha^D + 1) * a_(D-1-j).
```

The same recurrence applies to each sparse fold challenge on the T side. For
the T25 rank-one root this replaces 65,536 six-prime padded D512 quotient
products with about 67 million exact fp128 multiply-adds and 512 MiB of
coefficient weights. Those weights are per-proof because `alpha` is sampled
after the witness binding. Generate them into resident Metal storage and
consume them in Stage 2; a host `Vec<F>` materialization is outside the design.

The production representation should encode reduced A rows in the checked
relation layout and omit their R rows. A temporary zero R slot is acceptable
only for a focused prototype and cannot be promoted: it leaves dead committed
coordinates and obscures the protocol. Consistency, B, D, and compression rows
remain on their current quotient semantics in the first cut. CPU and Metal
must produce identical proofs under the new public schedule/protocol digest,
so the old CPU anchor becomes invalid only when this cutover is enabled.

Pre-register the protocol mechanism at 0.12--0.24 s for reduced-weight
construction and at most 0.30 s for the complete relation replacement. Reject
it if direct weights require a dense host allocation, are sampled before the
next-witness binding, add a free prover-selected scalar, raise Stage 2 by more
than 0.12 s, or leave the complete-call projection above 1.50 s after the
factored-sumcheck and serial-tail budgets below are applied. The independent
correctness gate compares the recurrence against direct cyclotomic products
for every supported A dimension before any full treatment.

### Pair-coalesced factored equality rounds

The direct Stage-1 Metal kernels currently form

```text
weight(low, high) = E_first[low] * E_second[high]
```

for every pair and then multiply that weight by four range-polynomial
coefficients. Akita's CPU prover uses the equivalent but cheaper order: for
each fixed `high`, reduce `E_first[low] * q_k(low, high)` over `low`, then
multiply each of the four totals by `E_second[high]` once. The T25 direct domain
has 29 variables, so the initial split is `2^14 x 2^14`. The initial Metal
round therefore performs 268,435,456 avoidable full fp128 equality
multiplications; the factored form performs 65,536 full outer multiplications
and retains the same four signed-small products per pair.

Map one 256-thread group to one high bucket. Threads walk the contiguous low
range, retain four accumulators, reduce exactly as the parent, and let the
group leader apply the high factor. This preserves pair-coalesced digit and
field reads, unlike the rejected thread-per-lane Stage-2 suffix. Use the
factored path while `num_first >= 512`; the first six geometric rounds then
take it, while small late rounds retain the flat kernel. T25 needs 16,384
partial groups and T28 at most 32,768, only 1--2 MiB of partial storage.

This kernel is an implementation slice of the quotient-free architecture, not
a claim that unchanged protocol can reach five times. Pre-register one
factored-shape CPU/Metal parity test and one unchanged T25 treatment. Predict
Stage 1 at 0.20--0.25 s, aggregate GPU-active improvement of at least 55 ms,
and a complete improvement of at least 70 ms from the 2.726 s parent. Promote
the kernel only if Stage 1 is at most 0.26 s, GPU-active improves by at least
45 ms, the complete call improves by at least 60 ms, and proof, transcript,
evaluation, verifier, schedule, routes, and memory remain exact. Restore the
flat kernels without a repeat treatment on any miss.

The single T25 treatment measured 2.601689 s complete, 0.701325 s GPU-active,
and 0.268577 s for Stage 1. It preserved evaluation, proof, transcript,
verifier, schedule, routes, and memory. The 124.453 ms complete improvement and
90.337 ms GPU-active improvement cleared their gates, but Stage 1 missed its
260 ms absolute ceiling by 8.577 ms. The candidate was therefore rejected and
the flat kernels restored without remeasurement. The result still validates
equality factorization as a useful operation count reduction; it does not
justify retaining this one-threadgroup-per-high mapping.

The combined post-cutover target budget is 0.565 s root decompose, at most
0.30 s direct A relation, 0.20--0.25 s Stage 1, 0.27--0.32 s Stage 2, and at
most 0.25 s for the remaining serialized tail. Its midpoint is about 1.43 s,
so quotient removal and equality factoring are necessary but not sufficient;
the direct-weight Stage-2 path and the 0.857 s residual tail retain independent
falsifiers rather than being hidden behind an optimistic aggregate.

### Resident reduced-source folding

The first quotient-free treatment accidentally rebuilt, converted, and
uploaded the complete reduced sources before every direct Stage-2 proof. At
T25 these were a 512 MiB setup source and a 64 MiB sparse-challenge source.
Keeping the two source tables resident and folding them in the same command
stream as the witness removed that repeated host path. Exact CPU parity,
transcript agreement, and verification all held. Stage 2 fell from 1.131 s to
0.621 s and the complete Metal opening fell from 3.377 s to 2.950 s. The
matching post-cutover CPU opening is 7.599 s, for 2.58x. GPU-active time was
0.713 s and the complete proof remained 265,189 bytes.

The retained implementation still constructs those 576 MiB in a host
`Vec<F>` and then blits them once. This costs 185 ms in the separately measured
Stage-2 preparation span and 155 ms of reported upload. Replace the values with
a typed source plan: the setup source carries `(D, rows, columns, row weights,
alpha)`, and the sparse source carries checked term offsets, positions, small
coefficients, and `alpha`. The Metal backend reads its already-prepared setup,
builds each negacyclic shift sequence directly into the resident table, and
reports that command as GPU work. The CPU backend materializes the same plan
through the canonical recurrence.

Pre-register exact proof, transcript, evaluation, and verifier parity. Predict
Stage-2 preparation below 20 ms, total upload below 80 ms, and complete T25 at
most 2.82 s. Reject the first kernel mapping if Stage 2 is at least 0.62 s or
the complete opening does not improve by 100 ms from 2.950 s. Independently of
that mapping, retain the typed plan if parity holds: it removes a backend-
neutral 576 MiB intermediate and is required for a max-buffer-safe T28 design.
Do not run T28 until the initial two-round prefix can consume the semantic
sources without allocating the full pre-fold table.

The first device generator preserved exact parity and reduced peak RSS from
8.62 GiB to 6.84 GiB, but failed its wall-time gates. Stage 2 reached 0.580 s,
upload reached 113 ms, and complete opening reached 2.922 s, only 28 ms below
the resident host-generated parent. GPU-active time increased from 0.713 s to
0.791 s because one thread in each source-column group executed all 512
dependent recurrence steps. Stage-2 preparation remained 192 ms; this exposed
a separate dense lane-to-segment layout cost rather than source generation.

Treat the recurrence as an affine prefix scan. Each of 32 SIMD lanes owns 16
consecutive shifts, computes its local affine transform, scans the 32 block
transforms with SIMD shuffles, then replays its 16 outputs from the resulting
start state. The setup dot product and exact recurrence are unchanged. A
focused setup-plus-sparse generator test must match the CPU recurrence before
measurement. Predict generator GPU depth to fall by at least eightfold, Stage 2
at most 0.55 s, aggregate GPU-active time at most 0.75 s, and complete opening
at most 2.87 s. Reject the SIMD mapping if Stage 2 is at least 0.57 s,
GPU-active improvement is below 30 ms, or complete improvement from 2.922 s is
below 40 ms. Keep the semantic source plan independently of this mapping.

The SIMD treatment remained exact. Stage 2 fell from 0.580 s to 0.532 s and
the complete opening fell from 2.922 s to 2.850 s, but aggregate GPU-active
time improved by only 10 ms, from 0.791 s to 0.781 s. It therefore missed the
pre-registered 30 ms GPU gate. Root decomposition and Stage 1 moved by 22 ms
and 17 ms in the opposite direction during the treatment, so the aggregate
counter cannot isolate the generator. Do not promote this mapping from that
run. Add dedicated source-construction command and GPU counters, and restore
the sequential generator while testing the independent host-layout change.

The remaining 0.19 s Stage-2 preparation path represented lane support three
times: first as a per-lane `Option`, then as overlap entries in a `BTreeMap`,
and finally as Metal CSR offsets. Replace these with one canonical CSR map
(`segments`, `lane_offsets`, `lane_segments`) at preparation time. Packing
merge concatenates segments and rebuilds the canonical map; Metal layout
clones it without another support expansion. This preserves the checked
strided segment representation and changes neither source values nor proof
arithmetic.

Pre-register the five focused evaluation-trace tests and direct-relation
CPU/Metal parity before one sequential-generator T25 treatment. The treatment
must preserve evaluation, proof, transcript, verifier, schedule, routes, and
memory shape. Predict Stage-2 preparation at most 80 ms and complete opening
at most 2.82 s. Reject the CSR implementation if preparation is at least
120 ms or complete improvement is below 60 ms from the 2.922 s sequential
device-generator parent. Report dedicated source command/GPU time so a later
SIMD treatment can be judged without unrelated-kernel noise.

The isolated CSR treatment passed. All five evaluation-trace tests and the
focused CPU/Metal direct-relation proof passed first. At T25 the proof digest,
transcript, evaluation, verifier, schedule, routes, and proof size remained
exact. Stage-2 preparation fell from 192 ms to 93 ms, Stage 2 from 580 ms to
503 ms, and the complete opening from 2.922 s to 2.723 s. The preparation
result missed the 80 ms prediction but cleared the 120 ms rejection boundary;
the 199 ms complete improvement cleared the 60 ms promotion gate. Peak RSS was
7.21 GiB. Dedicated sequential source construction measured 86 ms command
wall and 70 ms GPU-active, establishing the isolated ceiling for any successor
generator mapping.

### Register-distributed root histogram

The retained packed D512 root fold launches one 256-thread group per output
position. Its eight SIMDgroups issue 19.126 billion signed additions into one
shared 512-bin atomic histogram. An earlier eight-position mapping retained
eight separate shared atomic histograms, consumed 16 KiB of threadgroup memory,
and lost occupancy; that result does not test register ownership.

Assign one output position to each SIMDgroup and distribute its 512 bins as 16
private `i32` accumulators per lane. For each batch of 32 contributions, route
the packed `(destination, signed value)` from each source lane with
`simd_shuffle`; only the lane owning `destination mod 32` updates its register.
Each lane finally writes its 16 disjoint coefficients. This keeps selector and
challenge reads, signed negacyclic rotation, arithmetic count, output order,
and transcript unchanged, while removing all histogram atomics, threadgroup
storage, and full-threadgroup barriers. Eight positions share a threadgroup
only as eight independent SIMDgroups.

Pre-register independent packed-fold CPU/Metal parity before one T25 treatment.
Predict root decompose-fold at 0.22--0.32 s and complete opening at most 2.50 s.
Promote if exact proof, transcript, evaluation, verifier, schedule, and routes
hold, root decompose-fold is below 0.40 s, and complete gain from the 2.723 s
CSR parent is at least 120 ms. Reject immediately on a Metal compile failure,
focused mismatch, root time at least 0.40 s, or complete gain below 120 ms.

The focused independent oracle and full proof remained exact, but the treatment
was decisively rejected. Root decompose-fold took 10.937 s and complete opening
took 13.146 s. Routing a batch of 32 contributions requires 32 SIMD broadcasts
and comparisons; across 19.126 billion contributions this was about 20 times
slower than the shared atomic parent. Restore the atomic kernel without a repeat
treatment and close register-distributed histogramming on M4. The focused root
oracle remains useful coverage for the specialized kernel.

### Isolated affine source scan

With canonical CSR retained, the sequential reduced-source generator now has a
clean baseline of 86 ms command wall and 70 ms GPU-active. Reapply the exact
32-lane affine prefix scan, without changing the accepted layout or any other
kernel. The existing setup-plus-sparse recurrence oracle is the correctness
gate. Run one T25 treatment and promote only if source GPU time improves by at
least 20 ms, Stage 2 improves by at least 25 ms from 503 ms, complete opening
does not regress from 2.723 s, and all proof/transcript/verifier/schedule guards
remain exact. Reject without repeat measurement on any miss.

The isolated affine scan passed every gate. Source construction fell from
86 ms to 34 ms command wall and from 70 ms to 21 ms GPU-active. Stage 2 fell
from 503 ms to 456 ms and complete opening fell from 2.723 s to 2.690 s. The
proof digest, transcript, evaluation, verifier, schedule, route counters,
265,189-byte proof size, and memory envelope remained exact. Retain the SIMD
scan; its remaining 21 ms device cost is no longer a material target.

### Live-prefix direct sumchecks

The T38 root emits 337,224,640 live coefficients into a `2^29` sumcheck
domain. Removing every remaining quotient and compression witness saves only
109,520 coefficients in total, so neither change can cross the `2^28`
boundary. The useful backend invariant is instead that the remaining
199,646,272 entries form one zero suffix. After binding a low variable, a
live prefix of length `L` remains a live prefix of length `ceil(L / 2)`.

Both direct sumchecks may omit pairs outside that prefix without changing a
round polynomial. Stage 1's range image is zero on a zero digit. In Stage 2,
the virtual range term and the ordinary or reduced linear relation terms all
contain the witness value or its pair delta, so a zero/zero pair contributes
zero even when its equality, alpha, or lane weights are nonzero. Sparse
additional pairs remain explicit; field-table reads outside the live prefix
must return zero. The transcript domain, round count, round order, proof, and
verifier are unchanged.

Across all rounds, active pair work falls asymptotically from `2^29` to
337,224,640, a ratio of 0.628. Allocate the two resident fp128 tables from the
live length after the three compact-prefix rounds, track the live length
separately from the protocol domain, and dispatch only
`ceil(live_length / 2)` pairs. Existing non-power-of-two CPU/Metal parity tests
are the focused gate. Run one T25 treatment after they pass. Predict Stage 1
at 0.20--0.24 s, Stage 2 at 0.29--0.36 s, and complete opening at 2.35--2.48 s.
Promote only with exact proof, transcript, evaluation, verifier, schedule, and
routes, Stage 1 below 0.27 s, Stage 2 below 0.40 s, at least 100 ms complete
gain from 2.690 s, and no memory regression.

The first integrated attempt failed the Stage-2 final-claim check at a D64
suffix. A new nonzero-relation focused case reproduced it. Skipping witness
zeroes is sound during coefficient rounds, but the nonzero relation-lane weight
table must still fold over its complete padded domain during later lane rounds;
otherwise a future boundary pair reads a weight that was never produced.
Restoring only those late full-domain weight folds fixed the regression. The
ordinary virtual-only case, the nonzero relation case, and Stage 1 then all
matched CPU.

The corrected T25 treatment preserved the proof digest, transcript,
evaluation, verifier, schedule, routes, and 265,189-byte proof. Complete
opening fell from 2.690 s to 2.561 s, Stage 1 from 328 ms to 250 ms, Stage 2
from 456 ms to 414 ms, GPU-active time from 771 ms to 661 ms, command wall
from 914 ms to 787 ms, and reported allocation from 6.35 GB to 5.14 GB. Stage
1 and the 100 ms complete-gain gate passed; Stage 2 missed its 400 ms ceiling
by 14 ms. The discrepancy is analytical: the existing two-round prefix had
already restricted the two dominant Stage-2 coefficient rounds to live lanes,
so this change could only trim the remaining quarter of its geometric work.
Retain the exact implementation and its memory reduction as the working
parent, but record the Stage-2 prediction as missed rather than claiming a
full gate pass.

## Accepted route after the live-prefix treatment

Minor protocol changes are in scope, but the next slice remains a scheduling
change.  The root fold challenge is selected by Fiat--Shamir grinding, so a
consumer cannot defer the fold until the winning nonce is known without running
the expensive decomposition twice.  Instead, each nonce owns a speculative
prefix accumulator.  Completed Metal position chunks are converted to canonical
Z digits and committed while later chunks execute.  A rejected nonce drops that
accumulator; only the accepted nonce's prefix reaches `PreparedFold`, and the
commitment is still absorbed at the existing transcript position.

The streamed boundary is exact:

* input remains the packed row-major K256 source plus the already-sampled sparse
  challenges;
* Metal owns ordered, disjoint D512 position chunks and never rewrites a chunk
  after signaling it;
* a chunk contains complete `(position, inner digit, fold digit, coefficient)`
  Z records in canonical witness order;
* the CPU consumer accepts only a contiguous prefix ending on a successor
  commitment-block boundary, commits each block once, and concatenates inner
  rows in increasing block order;
* E, T, R, compression, outer commitment, proof bytes, and transcript order stay
  on their current paths;
* unqualified geometry uses the ordinary synchronous fold and cannot be reported
  as a streamed route.

For T25, use eight position chunks unless commitment-block alignment requires a
larger divisor.  The root currently costs 0.558 s and the overlapping host
emission/prefix branch is approximately 0.40--0.47 s.  The registered complete
root-chain target is 0.62--0.68 s including the 0.055 s trailing commitment.
Reject the mapping if exact chunk parity fails, if the streamed inner rows differ
from a one-shot commitment, if a rejected grind attempt leaks state into the
winner, or if the root chain remains above 0.75 s.  One focused chunk/commitment
parity test precedes one T25 treatment; no T28 run belongs to this slice.

The corrected T25 treatment preserved the proof digest, transcript, evaluation,
verifier, schedule, routes, and 265,189-byte proof. Complete opening fell from
2.561 s to 2.522 s. The streamed root plus trailing commitment took 0.709 s,
clearing the 0.75 s falsifier but missing the 0.62--0.68 s target. Root
decompose/fold rose from 0.558 s to 0.652 s because the span now includes the
overlapped host prefix work; `ring_switch_build_w` fell from 0.473 s to 0.401 s
and the trailing commitment remained 0.057 s. An initial implementation that
replaced rather than extended the existing prefix pipeline serialized the
non-Z suffix and regressed to 2.648 s; composing the early Z prefix with the
ordinary tail prefix removed that serialization. Retain the corrected schedule
as a small exact win, not as the main route to 5x.

The subsequent committed-witness range/relation treatment was rejected. It
replaced the staged auxiliary range image with

```text
rho * eq(tau0, x) * product_{d=-4}^{3}(W(x) - d) + W(x) * R(x).
```

The focused CPU and Metal sumcheck implementations agreed exactly, but the T25
pair exposed both a protocol integration defect and a decisive cost-model miss:
both proofs had the same bytes yet failed verifier replay, while the degree-nine
Metal sumcheck alone took 1.658 s against the registered 0.55 s rejection bound.
The complete Metal opening was 3.485 s. The matched CPU opening rose to 13.234 s,
so the apparent 3.80x ratio does not establish the five-times goal and the
candidate is not retained.

The analytical error was composing the degree-four range polynomial with
`W(W+1)` after multilinear folding. That creates a degree-nine round polynomial
and many generic fp128 products. The existing protocol instead treats the
pointwise range image as its own multilinear table, keeping Stage 1 at degree
five and using Stage 2 to prove its product link to `W`. A direct final-point
check cannot replace that link because, away from Boolean points,
`MLE(W(W+1))(r) != W(r)(W(r)+1)`. Any successor must retain a genuine product
argument or pay for a committed auxiliary representation; it cannot claim a
low-degree one-pass proof by conflating those evaluations. The experimental
code and wire change were removed, and the accepted staged route remains the
parent.

### Retained outer-relation quotients

The dominant child of `ring_switch_build_w` recomputes the B-role cyclic
product from `T` solely to recover the quotient of a product whose negacyclic
image was already computed during commitment. Folding `T` across commitment
blocks cannot remove this work: B has independent columns for every
block/A-row/digit position, so a challenge fold does not commute through the
current SIS map. Increasing the outer basis is also not a local win. The root
planner deliberately uses basis eight; basis 64 introduces three range-product
sumchecks and falls outside the specialized Metal Stage-1 routes.

Retain the B quotient in the prover-only commitment hint instead. The exact
producer boundary is the outer commitment mat-vec. For output coefficient
`k`, split convolution terms into non-wrapping and wrapping sets. Negacyclic B
is `low - high`, cyclic B is `low + high`, and the retained quotient is
`(cyclic - negacyclic) / 2 = high`. A Metal outer-commit kernel can therefore
emit the commitment and quotient from the same matrix and digit reads. The
hint carries one native-D64 quotient row per logical B row; proof bytes,
transcript, verifier, commitment, and security parameters do not change.
CPU and unsupported backends may use the existing cyclic computation, but the
comparison must not count work merely moved outside the measured interval.

At T25, the current complete opening is 2.522 s, `ring_switch_build_w` is
0.401 s, and its measured relation-quotient child is 0.363 s. The retained
rows are tiny (logical B rows times 64 field elements), but consistency and
other relation families remain. Thus 0.363 s is an upper bound, not a predicted
gain. Promote the slice only if independent commitment/product parity passes,
the same hint survives serialization, proof/transcript/verifier outputs remain
exact, eval proof improves by at least 120 ms, and integrated commit plus eval
improves by at least 80 ms. Reject it if the fused commit overhead consumes
more than half of the removed eval time. One focused parity run and one T25
treatment are sufficient; no T28 run belongs to this slice.

The treatment was exact but missed its promotion gate. Commitment parity,
proof bytes, transcript replay, and verification all passed. The Metal opening
fell from 2.522 s to 2.420 s, a 102 ms gain against the registered 120 ms
minimum. The nested quotient span fell from 363 ms to 0.74 ms, but
`ring_switch_build_w` only fell from 401 ms to 287 ms; the old nested span was
therefore not a removable-wall estimate. The matched CPU commit/open pair was
27.131 s + 6.917 s, while Metal was 4.557 s + 2.420 s, for 4.88x combined.
This is useful evidence and a working-candidate baseline, not a promoted
production design: the serialized hint surface is too large for a 102 ms eval
gain unless a later candidate amortizes or supersedes it.

### Basis-32 root schedule candidate

The next candidate changes protocol configuration, not the Jolt PIOP or source
layout. The current root uses opening/outer basis 8, hence 43 digits for each
128-bit field value. At T25, its T segment is exactly
`43 * 2^22 = 180,355,072` coefficients. Basis 32 uses 26 digits, reducing that
segment to `26 * 2^22 = 109,051,904`. Holding every other segment fixed gives
an upper bound of 265,921,536 live witness coefficients, below `2^28` by
2,513,920. Thus the padded Stage-1/Stage-2 domain falls from `2^29` to `2^28`;
any additional E/D shrink is upside, not required for the cliff.

The security price is explicit. For the root D64 B matrix with eight slices,
8,192 live blocks, D512 input rows, and one A output row, basis 8 has physical
width `1024 * 8 * 43 = 352,256`. Its collision bound is 7, rounded to the
audited bucket 7, and rank one admits 503,495 columns. Basis 32 has physical
width `1024 * 8 * 26 = 212,992`; its collision bound is 31, whose D64 rank-one
cutoff is only 18,455, so B must rise to rank two (rank-two cutoff
57,253,878). Across all slices, the B mat-vec work proxy rises about 21% even
though T shrinks 40%. This rank change must be included in commit and
recursive-source measurements; it cannot be reported as a free digit-depth
win.

Akita's canonical basis-32 range proof has one arity-four product sumcheck and
one quartic leaf sumcheck. The existing Metal direct route only implements the
single low-basis leaf. Falling back to CPU or silently timing only one stage is
disallowed. The implementation candidate is therefore:

1. expose the root opening basis as a config-owned schedule choice and
   regenerate the Jolt K256 catalog, leaving the PIOP and packed trace layout
   unchanged;
2. add D64 signed-small multiplication for basis-32 commitment digits instead
   of repeated addition;
3. implement the existing arity-four product plus quartic leaf transcript on
   Metal, retaining exact child claims and proof shape; and
4. keep the `2^28` Boolean domain in the evaluator even though its live prefix
   is slightly smaller.

Pre-registered falsifiers: reject before a large run if the generated T25 row
does not cross to `2^28`, if any root B/D SIS row lacks an audited rank, if a
qualified Stage-1 operation falls back to CPU, or if focused CPU/Metal proof
and commitment parity fail. After those gates, run one T25 CPU/Metal pair.
Against the 2.420 s working candidate, require at least 250 ms complete-opening
gain and Stage 1 no slower than 300 ms; against the 2.522 s accepted parent,
require at least 350 ms. Report fresh matched CPU, Metal commit, and integrated
commit-plus-open numbers because the security-driven B rank changes. Do not run
T28 for this slice.

The regenerated catalog passes exact expansion and SIS validation. The selected
T25 root is more favorable than the conservative digit-only estimate:

```text
root B/D log basis             3 -> 5
root opening challenge ring   64 -> 128
root fold digits               4 -> 2
outer B rank                   1 -> 2
outer slices                   8 -> 4
recursive folds                6 -> 5
live W               337,224,640 -> 203,531,008
padded W domain              2^29 -> 2^28
```

The planner halves the slice count when B rank doubles, so the number of outer
commitment rows remains eight. Across all slices, B mat-vec work is about 21%
higher, as predicted. The exact witness is only 75.8% live within the new
domain and is 39.6% smaller than the old live witness; the extra reduction
comes from the two-digit fold and changed D/opening geometry. This makes the
schedule a root-kernel candidate as well as a sumcheck-domain candidate.

The focused basis-32 implementation matched the canonical CPU proof exactly on
a partially populated Boolean domain, including both tree stages, child claims,
transcript challenges, final evaluation, and implicit padding. D64 commitment
rows also matched for signed basis-32 digits, and the generated T25 row passed
the rank-two SIS audit. The single valid T25 pair nevertheless rejected the
schedule:

```text
                                      working b8       b32 treatment
CPU opening                              6.917 s           6.818 s
Metal opening                            2.420 s           2.648 s
Metal Stage 1                            0.243 s           0.501 s
Metal Stage 2                            0.418 s           0.383 s
Metal NTT preparation                    0.080 s           0.178 s
CPU commit                              27.131 s          28.425 s
Metal commit                             4.557 s           4.967 s
combined CPU / combined Metal              4.88x             4.63x
```

The domain cliff was real, but the cost model treated Stage 1 like one halved
scan. Canonical basis 32 instead performs an arity-four product sumcheck and a
quartic leaf sumcheck. The product stage materializes four full-field lanes, so
the two half-domain proofs move and multiply more data than the old single
basis-eight proof. The security-driven schedule also increased NTT preparation.
Exactness, commitment parity, verifier replay, and the 90 GiB memory guard all
passed, but the treatment missed both the 250 ms opening-gain bar and the 300 ms
Stage-1 ceiling. Remove the forced basis-32 schedule and its production routing;
do not use the smaller witness domain as a performance proxy.

### Retained outer digit planes

The restored basis-eight root spends 286.6 ms in `ring_switch_build_w` even
after the retained B quotient makes relation-quotient construction negligible.
The dominant unbucketed operation is avoidable: commitment already constructs
the complete `t_hat` decomposition before the outer mat-vec, but its prover hint
retains only the fp128 inner rows. Opening then decomposes those rows a second
time. At T25 the discarded/rebuilt value is exactly

```text
8192 blocks * 1 A row * 43 digits * 512 coefficients
    = 180,355,072 signed bytes.
```

Retain the existing `DigitBlocks` values in `AkitaCommitmentHint` by moving their
allocation after the outer commitment consumes it. Ring switch validates the
retained stride, block count, and per-block plane count against the public
schedule before using it; old or unsupported hints retain the canonical
decomposition fallback. The hint remains prover-only and serializable. This
changes no commitment, proof message, transcript event, verifier equation, or
timed-boundary ownership, and it adds no commitment arithmetic. The cost is an
extra 172 MiB of retained T25 hint memory when callers also keep the fp128 inner
rows.

Pre-register exact retained-versus-recomputed digit parity, hint serialization
round-trip, and complete proof/transcript/evaluation/verifier parity. Run one
T25 CPU/Metal pair only after the focused gate. Predict ring-switch build at
most 0.16 s and Metal opening at most 2.30 s. Promote if opening improves by at
least 80 ms from the 2.420 s working parent, Metal commit regresses by at most
25 ms, combined commit-plus-open improves, and peak memory remains below the
90 GiB T28 projection. Reject the retained representation on any shape escape,
copy of the 180 MiB digit body, or correctness mismatch.

The focused parity and serialization gates passed, but the single T25 pair
rejected the representation:

```text
                                      working parent      retained T digits
Metal opening                              2.420 s             2.415 s
Metal ring-switch build                    0.287 s             0.280 s
Metal commit                               4.557 s             4.741 s
fresh CPU commit + opening                       -            33.933 s
Metal commit + opening                     6.977 s             7.156 s
fresh combined speedup                           -              4.742x
```

The retained 180 MiB removed only 6.6 ms from ring-switch construction and
5.3 ms from the complete Metal opening. Metal commitment was 184.0 ms slower,
so combined Metal time regressed by 178.8 ms. Proof bytes, transcript replay,
claimed evaluation, commitment parity, verifier acceptance, and the memory
guard all passed. The measurement falsifies the assumption that reconstructing
`t_hat` is a material part of the current ring-switch cost. Remove the retained
digits from the hint and keep the canonical reconstruction path.

### Non-invasive boundary and SIMD-bucket equality reduction

Further work on the current campaign keeps the proof protocol fixed. In
particular, quotient-free A relations, mixed-basis range choreography, altered
fold-challenge distributions, extra witness commitments, and transcript or
verifier changes are deferred. Permitted changes are kernels, internal prover
representations, device residency, overlap, and prover-only hint data that do
not affect a commitment or proof byte.

A focused ring-build audit also closes the apparent unaccounted host pass. In
two exact T25 Metal openings, reconstructing all T digit planes cost 17.5--22.1
ms. Moving the retained inner rows was 0.005 ms, layout preparation was 0.059
ms, wrapping the witness was 18.3 ms, and releasing the large temporary
materialization was 0.012 ms. The rest of the enclosing ring-build span is
primarily the prefix commitment already reported in the coefficient-packing and
NTT buckets. Therefore neither retained T digits nor another host-copy cleanup
has a material ceiling.

The tested backend-only candidate revisited equality factorization with a mapping
that was not tested by the rejected one-threadgroup-per-high-bucket kernel. One
SIMDgroup owns one `E_second` bucket. Its 32 lanes walk consecutive `E_first`
pairs, retain the four range-polynomial accumulators, and reduce with SIMD
shuffles. A 256-thread group processes eight independent high buckets, and one
lane per SIMDgroup applies `E_second` once before writing the bucket partial.
This preserves coalesced witness and `E_first` reads, keeps all eight SIMDgroups
useful, and removes the full-threadgroup reduction barriers.

For a field-valued range round, the flat kernel performs one multiplication to
form `E_first * E_second` and four more to weight the round coefficients per
pair. The SIMD-bucket form performs the four `E_first` products per pair and
only four `E_second` products per high bucket. The compact integer round removes
the per-pair full equality multiplication entirely. Partial storage is four
field elements per high bucket: at most 1 MiB for the T25 split and 2 MiB for
T28. The table, round order, transcript, proof, and verifier are unchanged.

The pre-registered focused gate was exact CPU/Metal round parity at a partial
live prefix. After that, one T25 treatment had to keep proof, transcript, evaluation, schedule,
routes, and verifier exact; reduce Stage 1 from the 0.243 s working-parent phase
to at most 0.21 s; reduce aggregate GPU-active time by at least 30 ms; and
improve complete opening by at least 50 ms. A miss rejects the mapping without
a repeat treatment. If retained, the same SIMD-bucket decomposition may be
applied separately to only the virtual equality term in the generic Stage-2
suffix; that is a later candidate with its own gate.

The partial-prefix parity gate passed for all three kernel forms. The single
T25 treatment then rejected the mapping: Stage 1 fell from 242.75 ms to
226.22 ms, GPU-active time fell from 645.82 ms to 626.18 ms, and complete
opening rose from 2.420 s to 2.562 s. Proof bytes, transcript replay, claimed
evaluation, commitment parity, and verifier acceptance remained exact. The
16.53 ms Stage-1 gain confirms the factorization but misses the 32.75 ms phase
bar, while the 19.64 ms GPU reduction shows that the removed equality multiply
and barriers are not a large enough share of the fp128 round. No kernel or
dispatch code from this candidate is retained, and its Stage-2 analogue is not
worth pursuing independently.

### Position-partitioned CPU/Metal root fold

The max-scale root is not occupancy-limited. T25 already launches 65,536
256-thread groups. The T28 evaluator launches 262,144 groups, doubles the work
inside each group, and has 253,779,321 populated rows. Its exact nonzero update
count is therefore

```text
253,779,321 rows * 30 columns * 19 terms = 144,654,212,970 additions,
```

or 7.563 times T25. Scaling the best visible 476.4 ms packed fold at its current
update rate gives a favorable 3.60-second T28 Metal projection. More groups do
not improve resident occupancy; they add scheduling waves.

Output positions are independent, however. Partition the canonical position
axis once: Metal writes the prefix while the existing Rayon implementation
writes the suffix. The ranges read disjoint rows inside every trace block, write
disjoint D512 outputs, and concatenate without a reduction. Total selector and
output traffic is unchanged. Sparse-challenge preparation is shared; proof
bytes, transcript order, schedule, commitment, and verifier equations do not
change. The streamed commitment sink consumes the Metal prefix first and the
completed CPU suffix second, preserving canonical order.

For full-route times `G` and `C`, an ideal linear split assigns `C/(C+G)` of the
positions to Metal and `G/(C+G)` to CPU, with latency

```text
H = C*G/(C+G).
```

Using the provisional T28 CPU root time of 5.15 seconds and the favorable
3.60-second Metal projection gives `H = 2.12` seconds, a 1.48-second root
reduction. This does not establish five times by itself: the favorable complete
T28 projection moves from roughly 6.2--6.7 seconds to 4.7--5.2 seconds before
any contention. It does restore a credible path when combined with a genuine
large-round pass reduction.

The adverse term is CPU contention. The streamed root already uses host workers
to emit the successor commitment prefix, so CPU suffix work can delay that
consumer even when the root arithmetic overlaps perfectly. Unified-memory
traffic should not be the first limit (the partition preserves one aggregate
source pass), but the complete-call gate prices both CPU scheduling and memory
controller contention rather than assuming ideal overlap.

Before implementation, one current T25 CPU-only evaluator record must recover
the isolated packed-root span. With `G = 0.476` seconds, reject without code if
`C > 2.35` seconds: even ideal partitioning would then save less than 80 ms.
Otherwise add a range form of the existing CPU and Metal operations and use the
model-derived split. Focused parity must compare the concatenated result with
the ordinary one-shot fold, including an uneven boundary. One T25 Metal
treatment follows. Retain only if root command wall falls by at least 70 ms,
the complete opening improves by at least 50 ms from 2.420 seconds, proof and
transcript artifacts remain exact, and no qualified route is reported as a
fallback. Do not run T28 until the measured split and the remaining phase model
project at most 4.5 seconds.

The diagnostic CPU record completed the opening in 6.511 seconds and measured
the isolated root fold at 2.240852 seconds. It therefore clears the 2.35-second
code gate narrowly. Against the 476.4 ms Metal command, the ideal model assigns
82.47% of positions to Metal and 17.53% to CPU, predicts 392.9 ms, and exposes
only 83.5 ms of T25 root improvement. This candidate is marginal at T25 and is
retained for implementation only because the measured T28 CPU/root projection
has a much larger overlap term. Any material contention with the streamed
commitment consumer will reject it at the complete-call gate.

The implementation used the fixed nearby split of 53/64 positions on Metal
and 11/64 on CPU. The focused Metal range test and concatenated CPU range test
both passed, including an uneven boundary. The single T25 treatment then
rejected the mechanism:

```text
                                      working parent       position split
root decompose/fold                         0.476 s              1.558 s
complete Metal opening                     2.420 s              2.955 s
aggregate GPU active                       0.646 s              0.581 s
```

The root span regressed by 1.082 seconds (3.27 times the parent latency), and
the complete opening regressed by 534.5 ms. The 64.4 ms reduction in aggregate
GPU-active time shows that reducing the Metal prefix did take effect; the
critical path instead waited on the planned CPU suffix and its ordered handoff.
The ideal model's independence assumption is therefore false on this shared
SoC execution path. Proof size and digest, claimed evaluation, transcript
replay, commitment parity, verifier acceptance, and the memory guard all
passed. Remove the range adapters and hybrid schedule. Do not extrapolate CPU
and Metal root rates independently at T28.

### Max-scale control refresh

The retained architecture is now a max-scale milestone: packed Metal root
folding, live-prefix direct sumchecks, streamed successor commitment, semantic
device source generation, and retained outer quotients all compose in the
canonical evaluator. The remaining T28 model is nevertheless internally
inconsistent. The current T25 CPU root processed 19,126,026,240 sparse updates
in 2.240852 seconds. The populated T28 case requires 144,654,212,970 updates,
or 7.5632 times as many. Equal throughput predicts a 16.95-second CPU root,
whereas the old provisional max-scale table assigns it only 5.15 seconds. T25
already has ample position parallelism, so a hidden 3.29-times CPU throughput
increase at T28 is not a credible planning assumption.

Refresh the max-scale control once with one CPU-then-Metal pair in the existing
single-shot harness. This is a diagnostic of the retained revision, not a
candidate treatment, and it reuses one fixture and commitment preparation for
both openings. Require proof-byte parity, matching claimed evaluation and
transcript, verifier acceptance, exact schedule/fixture digests, commitment
parity, and peak RSS below 90 GiB. Do not repeat the pair. If Metal already
clears five times, move to cleanup and held-out validation. Otherwise use the
measured disjoint phases and ratio to set the next T28 mechanism's minimum
required gain; reject any mechanism whose analytical ceiling cannot close that
measured gap.

The first attempt stopped before either opening because the eval-oriented T28
root uses 262,144 positions per commitment block and therefore 512 blocks per
column. The packed D512 commit route registered only 32, 64, 128, and 256 even
though its kernel, command streaming, and output indexing are parameterized by
the block count. This is an obsolete dispatch bound rather than a new commit
algorithm. Relative to the previously measured T28 commit geometry, halving
positions per block doubles block tasks: total lane probes and modeled matrix
streaming remain constant. The 512-block shape needs a 128 MiB output and a
2 GiB partial buffer, both checked against the Metal device buffer limit, and
its largest output index is only 8,388,607.

Register 512 at both validation layers and require exact CPU/Metal inner-row
parity from a real 512-block panel dispatch before resuming the same control.
That focused regression passed. The failed attempt produced no opening result,
so the resumed pair remains the one max-scale measurement authorized above.

The next ranked mechanisms are deliberately bounded. A D512 challenge family
with 15 coefficients in `+/-1` and two in `+/-2` has 128.36 bits of raw support,
the same L1 mass 19, and weight 17; it can remove only 10.5% of root additions
and still needs a complete schedule-security audit. Larger root rings reduce
the production challenge weight to 16 at D1024 and 14 at D2048, but require new
commitment schedules and kernels. Repeated bivariate sumcheck rounds can reduce
table traffic, but their 25-point Stage-1 grid raises arithmetic and register
pressure. None is promoted ahead of the unchanged-protocol position split.

### Lazy Stage-2 packing-lane index

Stage-2 preparation rebuilt the same lane-to-segment CSR three times: once for
coefficient-packing terms, once for negacyclic setup terms, and once after
merging them. The direct Metal path only needs the merged index. Retain the
compact checked segment geometry during construction and merge, then build the
CSR once on first direct-layout access. This changes no proof data, arithmetic,
or backend route.

The focused coefficient-packing suite passed 22 tests. The single T28
treatment preserved the proof digest, claimed evaluation, transcript replay,
commitment parity, verifier acceptance, and memory guard. Stage-2 preparation
fell from 321.108 ms to 18.252 ms, and complete opening fell from 8.143 s to
7.979 s despite a noisy 122 ms increase in the enclosing Stage-2 span. Against
the frozen 26.566 s CPU control, the retained stack is now 3.330x. The exact
five-times ceiling is 5.313 s, leaving 2.665 s to remove from the non-root
critical path.

### Accelerator-owned Stage-2 lane-weight folding

A command-level diagnostic at T28 measured about 444 ms of Stage-1 command
waits inside its 580 ms span. Stage 2 was different: about 660 ms of command
waits inside a 1.241 s span. The host still folds the full relation-lane weight
table after every lane challenge even though the resident Metal session folds
the same table for the next round. At T28 the first lane round starts from
`2^25` fp128 entries; the geometric CPU fold performs about `2^25` field
multiplications and reads or writes roughly 1 GiB in addition to the already
measured Metal work.

Keep the canonical host state for the equality factors, small coefficient
factor, and sparse additional relation. Stop folding only the accelerator-owned
lane-weight table. At the final round, read the two resident lane weights and
fold them with the final transcript challenge before reconstructing the
canonical final claim. This is a backend-state change only: round messages,
challenges, proof bytes, and verifier equations are unchanged.

The fast gate is one focused direct-proof parity test followed by one T25
treatment. Retain only if all proof/transcript/evaluation/route guards pass,
Stage 2 improves by at least 40 ms, and the complete opening improves by at
least 30 ms. A pass authorizes one T28 treatment; require at least 0.30 s from
Stage 2 there. Otherwise restore the host fold and do not tune it.

The focused parity tests passed. The first integrated attempt exposed a private
Metal-buffer read in the candidate's final scalar handoff; a 32-byte final blit
and an odd-lane regression localized and fixed that implementation error before
remeasurement. The corrected T25 treatment preserved every correctness and
route guard, but Stage 2 improved only 17.4 ms, from 471.8 to 454.4 ms. Complete
opening improved 39.9 ms, from 2.061 to 2.021 s, which is not credible evidence
for a larger mechanism after the phase gate missed. Reject the candidate,
remove its final handoff, and do not run it at T28.

### Borrowed resident Stage-2 lane weights

The previous treatment isolated only the geometric CPU fold. It left the larger
boundary cost intact: the T28 `2^25`-element field table is converted into a
second 512 MiB limb vector and then copied again into a Metal buffer. Akita's
fp128 field is already stored as the same four canonical 32-bit limbs consumed
by the kernel. The conversion therefore adds traffic and allocation but no
representation change.

Move the original lane-weight vector out of the canonical state for the Metal
session, borrow its bytes as the initial shared Metal table, and keep it alive
and immutable while the GPU owns subsequent folds. The no-copy route is used
only when the allocation satisfies Metal's 16 KiB alignment and size rules;
record its borrowed byte count explicitly. A 32-byte final blit reconstructs
the canonical final lane scalar. CPU proving and unqualified Metal fallback keep
the ordinary owned-vector path. No proof or transcript value changes.

This combines two costs that cannot be removed independently: retaining the
source is what makes the no-copy buffer safe, while accelerator-owned folding
prevents the host from reallocating that source. The preregistered gate assumed
the T25 main table was 256 MiB and required at least that much reported no-copy
input, Stage 2 at most 0.40 s, upload at least 40 ms below the 76 ms parent, and
complete opening at least 60 ms below 2.061 s.

The focused parity cases passed, including odd lane counts and nonzero lane
weights. The T25 treatment then reported 139,231,232 borrowed bytes, revealing
that the main T25 table is 128 MiB, not the 256 MiB assumed by the gate (T28 is
512 MiB). More importantly, zero-copy did not improve the critical path:
Stage 2 was 454.8 ms versus 454.4 ms for accelerator-owned folding alone,
upload was 80.2 ms versus the 76.3 ms parent, and complete opening was 2.060 s
versus 2.061 s. The initial table copy is therefore not the missing Stage-2
mechanism. Reject and remove this candidate without a T28 run.

### Root-chain timeline and static E/T prefix

One diagnostic T25 run recorded first-entry and last-exit times for the retained
pipeline, then removed the probe. NTT preparation occupied the first 71 ms,
root coefficient packing the next 97 ms, and ring-relation opening preparation
ended at 242 ms. The streamed root fold then ran from 244 to 597 ms. Its Metal
work was 269 ms and its host prefix consumer accumulated 301 ms, so those two
branches already have essentially no spare T25 overlap.

The next serialized interval is structurally different. Ring-switch group
emission ended near 716 ms, but the build did not reach its tiny R emission
until 904 ms. The intervening 188 ms is the CPU commitment of the complete
E/T blocks following the already-committed Z prefix. E and T are available
before fold grinding: E is fixed by opening-row preparation, and T is fixed by
the retained commitment hint. Neither depends on the fold nonce or challenges.
Committing their complete successor blocks early changes neither witness bytes
nor transcript order.

At T25 the root Metal time is shorter than the existing Z-prefix consumer, so
early E/T work can save at most noise and may contend. At T28 the measured root
Metal time is 2.345 s while the Z consumer accumulates 1.427 s, leaving about
0.918 s of host work under the GPU critical path. The serialized E/T prefix is
therefore a max-scale scheduling candidate, not a small-scale speedup claim.
The implementation must validate that Z ends on a successor commitment-block
boundary, commit only a contiguous complete-block E/T range, concatenate inner
rows in canonical order, and leave the boundary block plus R/compression suffix
on the existing path. Unsupported layouts retain the current schedule.

The credibility gate is exact focused prefix composition followed by one T25
treatment for contention telemetry. T25 may be neutral, but must not regress by
more than 25 ms and must report the static-prefix byte count and CPU duration.
Only a measured T28 overlap model of at least 0.40 s authorizes one T28 run; that
run must improve complete opening by at least 0.35 s with exact proof,
transcript, evaluation, commitment, verifier, route, and memory guards.

The T25 treatment was exact and improved rather than merely holding steady.
The static worker took 259 ms; root streaming rose from 353 to 431 ms under CPU
contention, but ring-switch build fell from 283 to 93 ms and complete opening
fell from 2.056 to 1.967 s. This exposed only 78 ms of the overlapped work and
removed 190 ms from the serialized tail, authorizing the max-scale run.

The single T28 milestone preserved every correctness, route, and memory guard.
The static worker took 611 ms and remained entirely under the root critical
path: root streaming was 2.578 s versus 2.600 s in the parent. Ring-switch build
fell from 823 to 298 ms, and complete opening fell from 7.979 to 7.375 s, a
604 ms gain. Against the frozen 26.566 s CPU control this is 3.602x. The exact
five-times ceiling remains 5.313 s, leaving 2.062 s on the Metal critical path.
Retain the generic pre-fold hook, complete-block range checks, and canonical
prefix composition.

### Subring-owned root fold

The max-scale refresh also corrects an earlier operation count.  The root uses
an ambient D512 ring, but its coefficient-packing challenges are sampled in
the D64 subring and embedded with stride eight.  Each challenge therefore has
the production D64 shell (31 coefficients of magnitude one and 10 of
magnitude two), not the native D512 weight 19 assumed in the earlier roofline.
The populated T28 fold performs

```text
253,779,321 rows * 30 columns * 41 terms = 312,148,564,830 atomics.
```

That embedding exposes a stronger factorization.  Write a source coefficient
as `s = low + 8 * high`, with `low < 8`, and write the embedded challenge as
`c(X^8)`.  Then

```text
X^s c(X^8) = X^low Y^high c(Y),  where Y = X^8 and Y^64 = -1.
```

Thus the D512 output is exactly eight independent D64 negacyclic histograms.
Give one SIMDgroup to each `low` residue and two D64 destinations to each of
its 32 lanes.  Source SIMDgroups partition each 256-task tile into the eight
residues with ballots and a 4 KiB threadgroup queue.  After one barrier, the
destination SIMDgroup reads a dense 64-byte challenge row and accumulates its
two bins in `i32` registers.  A second barrier makes the queue reusable.  This
removes all response-histogram atomics, preserves all eight active SIMDgroups,
and uses no large threadgroup histogram.  It performs 64 ordinary signed-byte
lookups/adds per selected source instead of 41 contended atomics; challenge
storage is under 1 MiB at T28 and every accumulator is bounded by the existing
response-digit check.

This is distinct from the rejected register-distributed D512 kernel: that
kernel routed each sparse contribution through 32 SIMD broadcasts.  Here the
subring factorization makes ownership direct, so no contribution is shuffled.
It is also distinct from the rejected 32 KiB private-histogram design: this
kernel uses two scalar accumulators per lane and roughly 4.25 KiB of shared
queue state.

Pre-register the existing independent packed-fold oracle plus an embedded-D64
focused case.  Promote to one T28 credibility run only after exact focused
parity and a release build.  On T28 require exact proof bytes, transcript,
evaluation, commitment, verifier, schedule, routes, and memory guards.  Retain
only if packed-decompose GPU time is at most 1.35 s and complete opening is at
most 6.45 s (at least 0.925 s better than the 7.374884 s parent).  A kernel
compile failure, any parity failure, or either timing miss restores the atomic
parent without tuning tile sizes.

The focused oracle and the single T28 proof were exact, but both performance
falsifiers triggered.  Packed-decompose GPU time fell only from 2.307 s to
1.802 s, and complete opening fell from 7.374884 s to 7.006756 s.  This is a
real 368 ms whole-proof gain, but only a 1.28x root-kernel gain and less than
40% of the required treatment effect.  The result implies that the ordinary
dense add/load path is only about twice as cheap as the shared atomic path;
performing 64 dense operations for every 41 sparse operations consumes most of
that advantage.  Restore the atomic parent and do not tune queue tiles or
barrier placement.  A viable root replacement must also reduce arithmetic,
not merely exchange sparse atomics for dense lane-owned additions.

### Factored Stage-2 relation prefix

For each of the eight relation grid points, the retained two-round prefix forms
16 challenge-weighted digit quads per live lane and then multiplies the result
by that lane's equality weight.  Exact distributivity permits the opposite
order: reduce `lane_weight * witness_quad` over lanes first, then perform only
16 full challenge products per workgroup.  A candidate mapped one SIMDgroup to
each quad pair, cached at most 512 lane weights in threadgroup memory, and left
the structured linear terms on their canonical lane-owned path.  It changed no
message, challenge, proof field, or dispatch count and removed roughly 114
million full fp128 products at T28.

The existing 16-quad CPU/Metal parity test passed.  The single T25 gate retained
the proof digest, transcript, claimed evaluation, verifier result, commitment,
route, and memory guards.  Stage 2, however, was 470.466 ms versus 474.889 ms
for the retained parent, and complete opening was 1.942101 s versus 1.966950 s.
Both improvements are noise-scale and miss the preregistered 420 ms Stage-2 and
1.920 s complete ceilings.  The mandatory digit scans, reductions, and
structured terms control this kernel, not the displaced full products.  Revert
the candidate and do not spend a T28 run on this algebraic ordering.

### Block-batched root coefficient packing

The most attractive apparent pass elimination would derive the sparse root-fold
challenge before constructing the coefficient-packing opening payload, allowing
the 1.098 s host packing scan to overlap the 2.578 s Metal fold.  Reject that
reorder analytically.  The protocol deliberately absorbs the complete opening
digit payload before drawing the fold challenge; moving the draw earlier would
allow a prover to choose a high-dimensional opening witness after seeing the
challenge.  A later scalar evaluation check does not by itself prove that the
lost independence is harmless.  This needs a separate soundness argument or an
additional binding commitment and is outside the minor protocol-change budget.

The unchanged-protocol candidate instead attacks a host traversal defect.  At
T28 the specialized root packing route launches 512 independent row-block
tasks.  Every task walks the same `2^18 x 8` table of combined position and
packing weights while accumulating its own `30 x 64` deferred fp128 histogram.
Batch four blocks per Rayon task, visit one position at a time, load its eight
weights once, and update four independent histograms.  This retains 128 T28
tasks and 16 T25 tasks, keeps about 150 KiB of accumulator state per task, and
does not change the source, output order, field reduction, transcript, or proof.

The focused generic-versus-specialized oracle covers both stride-two and
stride-eight geometries and must remain exact.  One T25 treatment is only a
contention/non-regression gate: preserve every correctness and route guard,
keep root coefficient packing at most 110 ms, and keep complete opening at most
2.00 s.  A pass authorizes one T28 treatment.  Retain there only if root
coefficient packing is at most 650 ms and complete opening is at most 6.95 s,
with the frozen proof, transcript, evaluation, commitment, verifier, route, and
memory guards unchanged.  Either T28 timing miss restores the one-block
traversal without tuning the batch size.

The exact stride-two/stride-eight oracle passed.  The T25 treatment preserved
the proof digest, transcript, evaluation, commitment, verifier, route, and
memory guards, but rejected the traversal before T28: root coefficient packing
rose from 91.564 ms to 125.279 ms and complete opening rose from 1.966950 s to
2.017813 s.  Interleaving four distant row-block streams costs more than the
shared weight-row reuse saves.  Restore the one-block traversal and do not tune
the batch size.

### Fixed-shape raw-limb root packing loop

The next candidate preserves the accepted one-block sequential traversal.  For
the exact Jolt root shape (`30` live columns, embedding stride `8`, partial
width `64`), predecode the combined weights to their two canonical limbs, use
fixed loop bounds and the closed-form even/odd bucket map, and update the same
deferred histograms.  The crate forbids unsafe code, so this remains a checked
specialization; the benefit must come from removing field decoding and giving
the optimizer fixed index ranges, not from bypassing the project invariant.
All other shapes retain the generic loop.

Require the existing stride-two/stride-eight generic oracle first.  One T25
treatment must preserve every exactness and route guard, put root coefficient
packing at or below 85 ms, and keep complete opening at or below 1.96 s.  Only
then run T28.  Retain at T28 only if root coefficient packing is at most 800 ms
and complete opening is at most 7.10 s, with every frozen correctness, route,
and memory guard intact.  A miss restores the generic inner loop without
tuning unroll factors or representation details.

The focused oracle and T25 gate passed: packing fell from 91.564 ms to 76.907
ms and complete opening reached 1.937607 s.  The one authorized T28 treatment
also preserved every exactness, route, verifier, and memory guard, but rejected
the mechanism.  Packing fell only from 1.097750 s to 979.611 ms, while complete
opening fell from 7.374884 s to 7.334839 s.  These miss the 800 ms and 7.10 s
ceilings; neighboring phase variation absorbs most of the 118 ms local gain.
Restore the generic loop.  Together with the atomic, segmented, block-batched,
and fixed-shape results, this closes local host root coefficient-packing loops
as a route to five times.

### Run-coalesced Metal root coefficient packing

The packed Metal kernel originally emitted eight threadgroup atomics for every
selected row.  In Jolt's T28 layout, a Metal thread visits rows 256 apart and
those contributions remain in the same coefficient bucket for the full
32,768-row partial.  An exact candidate accumulated that run canonically in an
fp128 register and emitted only one eight-digit atomic flush.  An adversarial
alternating-bucket test and a maximal-run test both matched the CPU definition;
the Jolt stride-two and stride-eight oracle also passed.

The T25 gate was exact and passed its loose credibility ceilings at 135.116 ms
for root packing and 1.980265 s complete.  It nevertheless trailed the retained
1.966950 s parent.  The one authorized T28 treatment was also exact, used zero
planned CPU packing calls, and preserved every verifier, transcript,
commitment, route, and memory guard.  Root packing was 1.034708 s versus the
1.097750 s CPU route, while complete opening regressed from 7.374884 s to
7.549810 s.  This rejects canonical run accumulation: it exchanges contended
atomics for roughly eight billion modular fp128 additions and then scales almost
exactly with the row count.

The next representation candidate keeps the proven run ownership but stores
eight independent radix-2^16 digit sums.  A run has at most 128 contributions,
so each local digit fits in `uint`; the aggregate bound remains
`32,768 * 65,535 < i32::MAX`, exactly the invariant already used to size row
partials.  This removes carry propagation and modular reduction from every row;
only the final existing wide reduction interprets the digit sums as a field
element.  Require the same three parity oracles.  At T25 require root packing at
most 80 ms and complete opening at most 1.94 s.  Only then authorize T28, where
retention requires root packing at most 450 ms and complete opening at most
6.70 s.  A miss restores the CPU route and the original generic Metal kernel.

The three parity oracles passed, but the T25 treatment rejected the carry-free
representation before T28.  Root packing was 132.215 ms, statistically
indistinguishable from the 135.116 ms canonical-run candidate and still slower
than the 91.564 ms CPU parent; complete opening was 2.060936 s.  Canonical carry
propagation was therefore not material.  The remaining first-order defect is
the memory traversal: one threadgroup owns one column, so adjacent SIMD lanes
load bytes 30 apart and each row is revisited by all 30 column groups.

A column-tiled candidate uses the same 8 KiB threadgroup digit budget to own
four columns when the root subring dimension is 64 (two at dimension 128 and
one at dimension 256).  Each thread consumes the columns for one row together,
keeps one carry-free run per local column, and writes the unchanged
column-major partial layout.  The qualified D512/K256/stride-eight index map is
`position = row / 2`, `coefficient = 256 * (row & 1) + hot`, avoiding dynamic
division in the dominant shape.  This reduces root threadgroups and lane-cache
transactions by up to four without increasing shared memory.  Require the
stride-two/stride-eight Jolt oracle plus long-run and alternating-run Metal
oracles.  At T25 require packing at most 80 ms and complete opening at most
1.94 s; at T28 require at most 450 ms and 6.70 s respectively.  Do not tune the
tile width after either miss.

All parity oracles passed, but the T25 treatment rejected before T28.  Root
packing took 141.627 ms and complete opening took 2.072766 s, slower than both
run-accumulation candidates and the retained CPU-packing parent.  Coalescing
four source columns did not offset register-array indexing and unchanged
weight traffic.  Restore the CPU route and generic Metal kernel.  The atomic,
segmented, block-batched, canonical-run, carry-free-run, and column-tiled
results close standalone Metal coefficient packing for this root shape.

### Pair-packed subring-owned root fold

Changing the production D64 challenge to D128--D512 is not a fair-ratio
candidate.  The challenge weights would fall from 41 to 31, 23, or 19, but the
same reduction applies to the 15.604 s CPU root decomposition.  A linear T28
projection for D512 changes the CPU control from 26.566 s to about 18.193 s and
the Metal treatment from 7.375 s to about 6.138 s: roughly 2.96x, farther from
the matched-protocol five-times objective.  Keep the schedule fixed and attack
only accelerator overhead.

The atomic root kernel performs 312.149 billion threadgroup atomics.  The exact
subring-owned treatment proved that dense D64 ownership works, but used 64
ordinary challenge loads and additions per selected source and reached only
1.802 s.  The next candidate removes its queue and barriers and packs two
destination coefficients into one 32-bit accumulator.  Two SIMDgroups own four
of the eight embedded D64 residue classes each.  A lane owns two adjacent D64
coefficients.  For every selected source, only the owning SIMDgroup loads one
prepacked challenge pair and performs one ordinary add per lane.  The pair uses
two 16-bit biased fields; at most 30,720 sources contribute to a position, so
`4 * 30,720 < 2^17` and each field is accumulated in a 32-bit lane without
cross-field carry.  Subtracting twice the source count recovers the two exact
signed sums.

Store even- and odd-start challenge pairs once per production D64 challenge.
The table is about 3.75 MiB at T28, versus roughly 1 TiB of logical sparse-term
traffic in the atomic kernel, and is shared by the many simultaneously active
position groups.  Each source now causes 32 coalesced ordinary pair loads/adds
instead of 41 atomics or 64 dense scalar loads/adds.  The input scan is repeated
by two SIMDgroups (about 16 GiB at T28), a small fraction of the challenge-term
traffic.  The conservative extrapolation from the exact 1.802 s dense kernel is
0.90--1.15 s of GPU time.

This kernel alone cannot remove all of that interval from the complete proof.
The streamed host Z-prefix consumer currently takes 1.586 s and becomes the
critical branch once GPU time falls below it.  Therefore treat this as the first
half of one architecture: first make the root fold sub-second, then keep its
balanced Z digits resident and feed the successor commitment without the host
stream.  The latter is authorized only by a successful kernel result.

Require the generic arbitrary-position fallback oracle and a new embedded-D64
oracle before timing.  At T25 require exact proof/transcript/evaluation/
commitment/verifier parity, no fallback, root-fold GPU time at most 210 ms, and
complete opening at most 1.97 s.  One passing gate authorizes T28.  At T28 retain
only if root-fold GPU time is at most 1.15 s and complete opening is at most
6.90 s.  A miss restores the atomic kernel without tile or SIMDgroup tuning.  A
pass makes resident Z-to-successor commitment, rather than another root
microkernel, the next candidate.

Both exactness oracles passed, but the T25 credibility run rejected the kernel
before T28.  Packed-decompose GPU time rose from 273.624 ms to 753.433 ms,
streamed root wall time rose from 429.910 ms to 813.420 ms, and complete opening
rose from 1.966950 s to 2.389943 s.  Replacing atomics did not compensate for
serial task/address generation in two SIMDgroups and the much larger rotated
pair lookup surface.  Restore the atomic kernel and sparse challenge buffers;
do not tune the pair table or SIMDgroup count.

### Column-major sidecar for root coefficient packing

The standalone Metal packing kernels have so far consumed Jolt's row-major
lane table directly.  A threadgroup owns one `(column, block, row-partial)`, so
adjacent SIMD lanes read bytes 30 apart.  At T28 this turns an 8 GiB source pass
into poorly coalesced cache-line traffic.  Tiling several columns inside the
packing kernel increased register pressure and did not fix that input contract.

Keep the row-major table used by the PIOP and commitment paths.  At the root
opening boundary, transpose it once on device into a temporary column-major
sidecar with a 32-by-32 byte tile.  The existing single-column packing kernel
then reads consecutive rows from consecutive bytes, retaining its checked
wide-digit accumulator, output order, and reduction.  The sidecar adds exactly
one source-sized allocation and 16 GiB of linear T28 transpose traffic.  At the
measured machine bandwidth that traffic has a tens-of-milliseconds floor; even
a conservative 0.10--0.20 s transpose leaves room to reduce the 1.098 s host
packing phase by 0.5--0.8 s.  Peak memory projects near 35 GiB, below the 90 GiB
guard.  No Jolt layout, proof field, challenge, transcript, or verifier changes.

Require stride-two and stride-eight CPU/Metal parity.  One T25 treatment must
report zero planned CPU packing calls, root packing at most 110 ms, complete
opening at most 2.00 s, and every correctness/route/memory guard exact.  Only a
pass authorizes T28.  At T28 retain only if root packing is at most 550 ms and
complete opening is at most 6.85 s.  Either miss restores the optimized CPU
route and removes the sidecar without tuning the transpose tile.

Exact stride-eight parity passed and the route reported zero planned CPU work,
but the T25 treatment rejected before T28.  Root packing took 159.844 ms and
complete opening took 2.073550 s, versus 91.564 ms and 1.966950 s for the
retained CPU-packing parent.  The extra device transpose and allocation cost
more than coalesced reads recovered.  Restore the optimized CPU route and
remove the sidecar; the row-major source contract is not the limiting seam at
this scale.

### Microtiled eight-residue root fold

The rejected pair-packed root kernel assigned four D64 residue classes to each
of only two SIMDgroups.  It therefore serialized four queue drains per active
group and left six SIMDgroups idle.  That result does not test packed arithmetic
with the eight-residue occupancy of the exact 1.802 s dense-subring kernel.

Retain one SIMDgroup per embedded D64 residue.  For each 256-source microtile,
the group lanes own the 32 adjacent coefficient pairs, and each queued source
adds one prepacked biased pair.  A coefficient in `[-2, 2]` is stored as
`coefficient + 2`; at most 256 sources enter one residue queue, so each 16-bit
half accumulates at most `4 * 256 = 1024` and cannot carry into its neighbor.
After every microtile, subtract twice the queue count and add the two exact
signed values to persistent `i32` accumulators.  Even- and odd-start D64 pair
tables remain about 3.75 MiB at T28.  The generic sparse-atomic kernel remains
the fail-closed route for non-embedded or wider challenges.

This keeps the proven 8 KiB queue/barrier geometry, restores all eight active
SIMDgroups, and performs 32 packed additions per selected source instead of 64
dense additions or 41 atomics.  The exact dense result calibrates a 0.9--1.3 s
T28 GPU interval; the 1.586 s host Z-prefix consumer will then control the root
wall time.  This candidate is useful only if it makes resident Z commitment or
later sumcheck work the next bottleneck.

Require the arbitrary-position atomic fallback oracle and an embedded-D64
pair oracle before timing.  One T25 treatment must keep every proof,
transcript, evaluation, commitment, verifier, route, and memory guard exact,
put packed-decompose GPU time at most 240 ms, and complete opening at most
1.95 s.  Only a pass authorizes T28.  At T28 retain only if GPU time is at most
1.30 s and complete opening is at most 6.60 s.  A miss restores the atomic
kernel without changing the microtile or queue geometry.

Both exactness oracles passed, but the T25 treatment rejected before T28.
Packed-decompose GPU time was 258.893 ms, only 14.731 ms below the 273.624 ms
atomic parent and above the 240 ms gate.  Complete opening regressed to
2.055697 s.  Eight active residue owners removed the earlier serialization,
but 256-source queue construction and its 240 threadgroup barriers now control
the kernel; halving the dense add count is nearly immaterial.  Restore the
sparse atomic kernel and remove the pair-table route.

### Concurrent public NTT prewarm

The retained T28 proof serializes 0.332 s of public NTT-cache construction
before 1.098 s of root coefficient packing and the 2.578 s streamed root fold.
The Metal facade builds these retained NTT slots through its CPU backend. A
top-level overlap can therefore remove at most the full 0.332 s and may recover
less if cache construction competes with CPU coefficient packing. It cannot by
itself close the 2.062 s five-times gap; its purpose is to test whether this
independent schedule/setup work belongs off the critical path before larger
resident-root changes.

For accelerator-root stacks only, build the already validated, schedule-derived
NTT plan in a scoped worker while the ordinary prover proceeds. Lazy cache
cells preserve exact ownership: a proof operation that reaches a slot early
waits for the same construction rather than building a duplicate. CPU-only
proving retains the serial order. No schedule, transcript event, proof field,
opening value, or verifier operation changes, and the worker is joined before
the proving call returns. Use the already validated 8 MiB worker-stack bound:
the CRT preparation path contains large fixed arrays and overflows macOS's
default spawned-thread stack before any measurement can begin.

Require focused exact proof parity before timing. One T25 treatment must keep
the proof, transcript, evaluation, commitment, verifier, route, and memory
guards exact; complete opening must be at most 1.927 s, a 40 ms gain over the
1.966950 s retained parent. A miss restores serial prewarm without tuning. Only
a pass authorizes T28, where retention requires complete opening at most 7.175
s, a 0.20 s gain over the 7.374884 s parent. The analytical T28 floor for this
candidate alone is 7.043 s; any larger claim is impossible because all
remaining work is unchanged.

The single T25 treatment was exact but missed the gain gate. Complete opening
was 1.951969 s, only 14.981 ms below the 1.966950 s parent and above the 1.927 s
ceiling. Proof digest, transcript, evaluation, commitment, verifier, route, and
memory guards all matched. Contention consumed nearly all of the nominal
overlap: NTT preparation rose from 88.100 to 182.438 ms and root coefficient
packing rose from 96.744 to 138.833 ms. Reject without T28, restore serial
prewarm, and do not overlap two CPU-heavy setup/opening phases again. A useful
prewarm overlap would need to start after coefficient packing and hide under
the Metal root fold; that requires a deliberate phase-boundary API and remains
bounded to 0.332 s at T28.

### Four-shard root histogram

For a fixed root position, adjacent tasks vary over `(block, column, row-half)`.
The Jolt fixture's row residue is independent of block because the 65,536 root
positions are divisible by 128. The embedded D64 challenge preserves the
source residue, so 256 workers distribute as about 32 simultaneous writers to
each 64-bucket residue class. Under an independent-bucket model those 32 writes
occupy about 25.3 distinct buckets: roughly 21% serialize on a same-step
collision before accounting for repeated structured values.

Give each fixed set of 64 workers its own 512-entry threadgroup histogram and
reduce four exact shard values only after the source scan. Each shard then has
about eight writers per residue and an expected 7.58 distinct buckets, about
5% collision loss. The kernel still performs the same 312.149 billion exact
T28 atomics and reads the same source/challenges; it grows threadgroup storage
from 2 to 8 KiB. Four 256-thread groups still fit a 32 KiB threadgroup-memory
budget, so the mapping should not reduce the thread-count occupancy ceiling.
This is a contention treatment, not a claim to have removed the atomic floor.

Require the existing D512 packed-fold CPU/Metal oracle. One T25 treatment must
keep every proof, transcript, evaluation, commitment, verifier, route, and
memory guard exact, put packed-decompose GPU time at most 250 ms, and complete
opening at most 1.94 s. Only a pass authorizes T28. At T28 retain only if GPU
time is at most 2.05 s and complete opening is at most 7.15 s, gains of at least
0.257 s locally and 0.225 s end to end. A miss restores the single histogram
without changing shard count or layout.

The focused oracle passed, but the single T25 treatment rejected before T28.
Packed-decompose GPU time was 274.600 ms versus 273.624 ms for the single-
histogram parent, and complete opening was 1.976056 s versus 1.966950 s. Every
proof, transcript, evaluation, commitment, verifier, route, and memory guard
remained exact. The collision estimate did not translate into throughput:
threadgroup atomic issue rate, not same-bucket serialization, controls this
kernel. Restore the single histogram and do not tune shard count or layout.

### Barrier-free SIMDgroup-local subring histograms

The dense subring-owned treatment removed atomics but retained a shared queue
and 240 producer/consumer barriers per T28 position. Pair packing reduced its
arithmetic but not that synchronization floor. Instead, let each of the eight
SIMDgroups consume its own strided eighth of the source tasks and retain a
complete D512 partial as sixteen `i32` accumulators per lane. For each 32-task
batch, a ballot identifies the source residue; the 32 destination owners read
the corresponding dense D64 challenge coefficients directly and apply the
negacyclic sign. The eight SIMDgroups write their partials once, cross one
threadgroup barrier, and reduce to the canonical D512 output.

At T28 this performs 7.613 billion selected-source visits and 487.3 billion
ordinary signed-byte load/add updates, versus 312.149 billion threadgroup
atomics. The dense challenge table is 3.75 MiB and shared across all 65,536
position groups. Per-group scratch is 16 KiB; one final store/read reduction is
32 KiB and negligible beside the source loop. The scratch may halve resident
threadgroups relative to the atomic kernel, while the sixteen lane accumulators
may add register pressure. Those are explicit falsifiers, not omitted costs.
Arbitrary D512 challenges retain the atomic path; only challenges whose support
is exactly embedded at stride eight use the new kernel.

Require both arbitrary-position fallback parity and embedded-D64 CPU/Metal
parity. One T25 treatment must preserve every proof, transcript, evaluation,
commitment, verifier, route, and memory guard, put packed-decompose GPU time at
most 235 ms, and complete opening at most 1.94 s. Only a pass authorizes T28.
At T28 retain only if GPU time is at most 1.45 s and complete opening is at most
6.65 s. A miss removes the dense route without tuning batch size, scratch
layout, or accumulator packing.

Both focused routes passed, but the single T25 treatment rejected before T28.
Packed-decompose GPU time rose to 322.490 ms from the 273.624 ms atomic parent,
and complete opening was 1.971287 s, above the 1.94 s gate. Proof, transcript,
evaluation, commitment, verifier, route, and memory guards remained exact. The
extra 1.56x dense coefficient work plus the 16-accumulator and 16 KiB occupancy
cost exceeded the saved atomics and barriers. Remove the dense route. Together
with the queue-owned, pair-packed, microtiled, and sharded results, this closes
dense/register histogram reformulations of the current root equation.

### Large-tile residue-owned root gather

The rejected residue kernels each retained one of two controlling costs. The
two-SIMDgroup pair kernel serialized four residues per group. The eight-group
microtile kernel routed each 256-task tile through a shared queue and crossed
240 threadgroup barriers at T28. The barrier-free kernel instead gave every
group a full D512 partial, requiring sixteen live `i32` accumulators per lane
and 64 dense scalar updates per selected source.

The next mapping combines none of those mechanisms. Stage 8,192 task selectors
once into an 8 KiB threadgroup tile. Each of the eight SIMDgroups owns one
embedded-D64 residue, scans the cheap staged bytes, and selects its tasks with
a SIMD ballot. Its 32 lanes own two D64 coefficients each and accumulate only
an `int2`. A selected source reads the canonical 64-byte dense challenge row,
using the subring source shift to derive two negacyclic coefficients per lane.
The tile needs two barriers, so T28's 30,720 tasks use eight barriers total
rather than 240. The source table is read once; only the 8 KiB threadgroup tile
is scanned eight times. The qualified dense challenge table is about 0.94 MiB
at T28 and arbitrary D512 challenges retain the atomic kernel.

This replaces 41 scalar threadgroup atomics per selected source with 32
coalesced pairs of byte loads and ordinary `int2` accumulation. Its logical
T28 challenge traffic is about 515 GB. Charging that entirely to the machine's
advertised 546 GB/s gives a deliberately conservative 0.94 s traffic floor;
the table is cache-reused across 262,144 positions, but address generation and
SIMD selection add work. The calibrated prediction is 1.0--1.4 s of GPU time,
versus the 2.307 s atomic parent. No source layout, challenge distribution,
proof message, transcript event, or verifier equation changes.

Require arbitrary-D512 fallback parity and embedded-D64 residue-gather parity.
One T25 treatment must keep every proof, transcript, evaluation, commitment,
verifier, route, and memory guard exact and put packed-decompose GPU time at
most 220 ms; complete opening need only remain at most 2.00 s because its
0.375 s host consumer already controls the T25 root branch. Only a pass
authorizes T28. Retain there only if packed-decompose GPU time is at most
1.45 s and complete opening is at most 6.65 s. A miss removes the pipeline and
dense table without changing tile size or adding another treatment.

Both exactness routes passed, but the T25 credibility gate rejected the
candidate before T28. Packed-decompose GPU time rose from 273.624 ms to
371.942 ms, missing the 220 ms gate, while complete opening was 1.991994 s.
The eight repeated scans of each staged selector tile cost more than the
atomics they replaced at this scale. Remove the residue-gather pipeline and
dense table without tuning the tile size.

### Audited D256 root packing challenge

The sparse atomic root kernel is the only measured formulation that exploits
the challenge support efficiently. Change the work count rather than its
mapping: for the qualified Jolt K256 rows, constrain the already-supported
subring coefficient-packing method from D64 to D256. This selects Akita's
production D256 challenge family, 23 random signed unit coefficients with a
131-bit support floor, instead of D64's 31 signed unit and 10 signed double
coefficients. No sampler, verifier equation, transcript mechanism, or security
floor changes.

The exact planner comparison keeps the T28 root at D512/rank one with 262,144
positions. Sparse fold work falls to `23 / 41 = 56.1%`. The wider challenge
also distributes root histogram updates over four times as many buckets and
removes the magnitude-two case. The offsetting costs are explicit. At T28 the
root successor grows from 942,788,224 to 1,078,019,008 coefficients (14.3%),
the first recursive successor grows from 18,329,408 to 25,446,016
coefficients, proof payload grows from 76,138 to 76,760 bytes (0.82%), and
setup capacity grows from 180,355,072 to 360,710,144 field elements. Root
fold and opening digit depths remain unchanged at 4 and 43. At T25 the root
successor grows 20.0%; proof payload is unchanged at 75,210 bytes.

The local T28 prediction is 1.2--1.4 s of packed-decompose GPU time, down from
2.307 s. The 1.586 s host Z-prefix consumer should then control that branch,
while the wider relation and recursive successor add an estimated 0.25--0.45
s elsewhere. The predicted complete interval is 6.5--6.9 s. This cannot reach
five-times alone; it is useful only if it removes at least 0.575 s and leaves a
smaller, measured Stage-1/Stage-2 target.

Use one T25 credibility run under the regenerated schedule. Require every
proof, transcript, evaluation, commitment, verifier, route, and memory guard,
packed-decompose GPU time at most 190 ms, and complete opening at most 2.20 s.
This gate permits the expected small-shape relation expansion because its job
is to falsify the local work model. A pass authorizes one T28 run. Retain only
if packed-decompose GPU time is at most 1.40 s, complete opening is at most
6.80 s against the frozen 7.374884 s Metal parent, and peak RSS remains below
90 GiB. Compare both the original frozen CPU control and a same-schedule CPU
control; report both ratios. Any T28 miss restores the D64 catalog rows.

The local work gate passed, but the end-to-end T25 gate rejected the treatment.
Packed-decompose GPU time fell from 273.624 ms to 184.851 ms, while the host
prefix consumer rose from 374.807 ms to 781.139 ms and ring-relation
preparation rose from 639.872 ms to 1,441.997 ms. Complete opening therefore
rose from 1.966950 s to 2.953407 s. The same-schedule CPU control took
6.662883 s, so the treatment reached only 2.256x. Every proof, transcript,
evaluation, commitment, verifier, route, and memory guard passed. Reject
without T28 and restore D64. A wider root challenge is viable only after the
prefix consumer and relation construction can consume the wider output without
material host work.

### On-demand large-root NTT preparation

The retained T28 proof pays 331.552 ms to build every retained NTT slot before
root coefficient packing. The earlier concurrent-prewarm treatment started at
that same boundary and failed because construction competed with the 1.098 s
CPU packing scan. The existing lazy cache provides a later boundary without a
new worker: if eager prewarm is disabled, the static E/T prefix worker first
requests the slots when fold grinding starts. Cache construction then runs
beside the 2.307 s root GPU fold, and concurrent users wait on the same slot
rather than building duplicates.

This is safe only as a scheduling policy. Every operation retains its checked
cache key and first-use construction path; the schedule, matrices, proof,
transcript, and verifier are unchanged. The retained root's streamed host
consumer takes 1.586 s and the static E/T worker takes 0.611 s. Adding the full
0.332 s construction cost to either branch remains below the GPU interval, so
the analytical best case is 7.043 s. Apply on-demand construction only to the
41-variable Metal trace route; smaller routes keep eager preparation because
their host consumer already controls root wall time.

Use one T28 treatment against the frozen 7.374884 s parent. Require identical
proof digest, transcript, evaluation, commitment, verifier, schedule, routes,
and memory guard. Retain only if complete opening is at most 7.150 s, root
streaming is at most 2.70 s, and no formerly retained cache is built after the
root overlap. Any miss restores eager preparation and removes the policy from
the Jolt route without tuning the variable threshold.

The single T28 treatment was exact but rejected. Complete opening rose to
7.538420 s, while packed-decompose GPU time remained 2.297 s and root streaming
remained 2.557 s. NTT construction still took 333.840 ms; its first serialized
demand was relation-opening preparation, which rose from 177.201 ms to
506.535 ms. The static E/T worker therefore never owned the first cache demand,
invalidating the overlap model. Restore eager construction and remove the stack
policy rather than moving the threshold.

### Stage-2 product-count reassessment

An environment-gated T25 trace of the retained parent identified the dominant
Stage-2 instance as a 2^29 domain with 5,269,135 live lanes, a 2^23 lane
capacity, six coefficient variables, four structured-linear sources, and
614,776 sparse lane segments. Its two-round compact prefix took 63.292 ms of
GPU time. The remaining large coefficient dispatches took 42.997, 24.825,
10.218, 5.208, and 2.294 ms; the geometric lane tail was negligible. Thus T28
already has enough occupancy. The scalable target is the product count in the
first six dispatches, not dispatch width or the late-round tail.

For each post-prefix pair let `d = right - left`, `e` be the remaining equality
factor, `l0` the Gruen linear factor at zero, `ld = l1 - l0`, and `p0,p1` the
ordinary plus structured-linear relation factors. The retained kernel computes

```text
c0 += e*l0*left*(left + 1) + left*p0
c2 += e*(l0*d^2 + ld*d*(2*left + 1)) + d*(p1 - p0)
c3 += e*ld*d^2.
```

Factor the shared outer witness values instead:

```text
e0 = e*l0; ed = e*ld
c0 += left*(e0*(left + 1) + p0)
c2 += d*(e0*d + ed*(2*left + 1) + p1 - p0)
c3 += ed*d^2.
```

This is the same polynomial and preserves every address, load, store, reduction,
round, and transcript value. It removes three of the fifteen full field
multiplications per pair while leaving structured-linear source products
unchanged. The 20% arithmetic ceiling applies to the 85.542 ms T25 post-prefix
root work, for a 17.108 ms local ceiling. At T28 the corresponding analytical
ceiling is about 0.137 s under linear scaling; this candidate is incremental and
cannot by itself close the 2.062 s end-to-end gap.

Require focused direct Stage-2 CPU/Metal proof parity. The single T25 treatment
must preserve the proof, transcript, evaluation, commitment, verifier, route,
and memory guards. Retain only if the five traced root coefficient dispatches
total at most 72.0 ms, a 15.8% local gain, and Stage-2 sumcheck is at most
0.465 s. A miss restores the original coefficient equations without tuning.
Only a pass authorizes T28, where retention requires Stage 2 at most 1.10 s and
complete opening at most 7.30 s.

Both focused parity tests passed, and the T25 treatment kept every end-to-end
exactness and route guard, but the local gate rejected it. The five root
coefficient dispatches totaled 78.453 ms, down 7.089 ms or 8.3% rather than the
required 15.8%. Stage 2 fell from 474.889 to 464.456 ms and complete opening
from 1.966950 to 1.956050 s. The smaller-than-product-count gain shows these
rounds are only partly multiplier-bound. Do not spend a T28 run on the miss;
restore the original equations and target compact-prefix work or field traffic.

### Post-packing NTT prewarm at T28

Deferring retained NTT construction until the root-fold worker owns the first
request was exact and passed its T28 gate. Complete opening fell from
7.006756 s to 6.746238 s, or 3.938x against the fixed 26.566117 s CPU anchor.
Root coefficient packing remained on the specialized CPU route; NTT work was
hidden beside the root fold rather than competing with that scan. The proof,
transcript, evaluation, commitment route, verifier, and 90 GiB memory guard all
passed. Retain this scheduling policy for the 41-variable Metal route.

### Disjoint CPU/Metal root-packing columns

The root coefficient-packing outputs are independent by column, so a 15/15
CPU/Metal split appeared to have a 0.53--0.75 s local bound from the measured
1.061 s CPU and 1.035 s full-Metal routes. Focused range-composition, maximal
atomic-load, alternating-bucket, and Jolt routing oracles all passed.

The single T28 treatment was exact but rejected. Root packing reached 0.732 s,
meeting its 0.75 s local gate, while complete opening reached only 6.717749 s,
missing the 6.45 s end-to-end gate. Metal GPU-active time rose by 0.749 s and
the following packed root fold rose by 0.261 s; static E/T work rose another
0.063 s. The 0.329 s packing gain therefore produced only a 0.028 s complete
gain, or 3.955x overall. This is a unified-memory/power-domain coupling, not a
column-independence failure. Remove the hybrid route, column-range API, and
run-coalesced kernel rather than tuning split ratios.

### Live-prefix factored Stage-1 equality rounds

The earlier pair-coalesced treatment saved 124 ms complete and 90 ms GPU-active
at T25 but was rejected because its 268.6 ms Stage 1 missed an absolute 260 ms
gate. That mapping predated live-prefix dispatch and used four 256-entry fp128
threadgroup reduction arrays. Reintroduce the algebra with two corrections:

1. launch only `ceil(live_pairs / num_first)` high buckets, with a bounded final
   bucket, so the retained zero suffix stays skipped; and
2. reduce four accumulators within SIMDgroups, cross one threadgroup barrier,
   and reduce eight SIMDgroup totals, cutting scratch from 16 KiB to 512 B.

For every live pair the flat kernel forms `e_first * e_second` and then applies
four coefficient products. The factored kernel applies the four products to
`e_first`, reduces by high bucket, and multiplies four totals by `e_second`.
It removes one full fp128 multiply per live pair and changes no table fold,
round polynomial, transcript event, proof field, or verifier equation. Use it
while `num_first >= 512`; smaller rounds keep the flat mapping.

The prior exact T25 treatment is sufficient scale evidence. Require one
live-prefix focused CPU/Metal proof-parity test, then one T28 treatment against
the retained 6.746238 s parent. Retain only with exact proof, transcript,
evaluation, commitment route, verifier, and memory; Stage 1 at most 0.50 s,
GPU-active at most 2.72 s, and complete opening at most 6.64 s. Any miss removes
the three factored pipelines without threshold or workgroup tuning.

The focused sparse-live-prefix proof matched CPU exactly, and the T28 treatment
preserved the proof, transcript, evaluation, commitment route, verifier, and
memory guard. Stage 1 fell to 498.735 ms and cleared its 500 ms gate. Aggregate
GPU-active time was unchanged at 2.786 s, however, and complete opening rose to
7.017111 s after the earlier CPU root-packing scan varied from 1.061 s to
1.287 s. The candidate therefore missed both remaining gates. Remove the three
factored pipelines and focused test without a repeat or threshold change. The
74.5 ms local reduction is evidence for equality factorization, but is too
small to survive the complete-call noise floor or close the 1.433 s gap.

### Aligned coefficient-subgroup Stage 2

The rejected coefficient-lane suffix assigned a whole relation lane to one
thread. It saved common products but changed adjacent coefficient reads into
strided lane reads and kept seven fp128 accumulators live across a loop. The
tested mapping preserved the pair-major access pattern. During the remaining
coefficient rounds, an aligned SIMD subgroup owns one relation lane: eight,
four, or two adjacent lanes each process one adjacent coefficient pair, reduce
their factored terms with SIMD shuffles, and let the subgroup leader apply the
lane equality and ordinary-relation weights once.

For eight pairs, the flat kernel performs about fifteen full fp128 products per
pair. The subgroup form performs ten per pair plus nine per relation lane,
reducing that part from about 120 to 89 products. The four-pair round falls from
about 60 to 49. Unlike the prior lane-owned treatment, every thread holds one
pair rather than a seven-accumulator loop, and adjacent SIMD lanes retain
coalesced witness, alpha, equality, and structured-linear reads. The final
partial reduction also uses SIMDgroup sums and one threadgroup barrier instead
of four 256-entry arrays and nine barriers. Ordinary proof messages,
challenges, variable order, resident tables, transcript events, and verifier
equations are unchanged.

The exact T28 parent spends 1.160862 s in Stage 2. The two eligible coefficient
dispatches account for most of the post-prefix geometric work. The calibrated
prediction is 0.90--1.00 s for Stage 2, a 0.16--0.26 s complete-call ceiling;
this cannot close the 1.433 s gap alone. Because prior small-scale gates
misclassified mechanisms whose work scales with the coefficient-domain table,
the campaign is now explicitly T28-only. Require the existing virtual-only and
nonzero structured-relation CPU/Metal proof parity tests before one T28 run.
Retain only if proof bytes, transcript, evaluation, commitment route, verifier,
and memory remain exact, Stage 2 is at most 1.00 s, and complete opening does
not exceed 6.75 s. Reject without subgroup-width or workgroup tuning if Stage 2
is at least 1.05 s or either exactness or end-to-end guard fails.

The exact virtual-only and nonzero structured-relation parity tests passed.
The one T28 treatment was nevertheless a clear rejection: Stage 2 increased
from 1.160862 s to 1.237700 s and complete opening increased from 6.746238 s
to 7.097380 s. GPU-active time also rose by 103.5 ms. The fp128 SIMD shuffle
and segmented-reduction cost did not repay the saved products. Remove the
three pipelines and do not tune subgroup widths or workgroup counts.

### Barrier-free pair-packed subring root fold

The retained D512 root-fold kernel spends 1.805 s of GPU time at T28. Each
output position has 30,720 source tasks. Its 256-task tiles use eight ballots
per SIMDgroup to partition tasks by the low three source bits, write and drain
a 2,048-entry threadgroup queue, and cross two barriers. That is 120 tiles and
240 barriers per output position. The queue makes all eight SIMDgroups useful,
but each selected source still performs 64 scalar challenge loads and adds.

Use one SIMDgroup for each low residue and let every SIMDgroup scan the source
tasks directly. One ballot selects its residue from each 32-task batch; set-bit
lanes broadcast the source high bits while the 32 destination lanes own
adjacent D64 coefficient pairs. Two 32-entry packed tables per challenge cover
even- and odd-source shifts. A lane rotation plus a pair-wide wrap negation
implements the negacyclic sign; the odd boundary negates only the low member of
one pair. This is 64 packed `u32` entries per challenge, about 3.75 MiB at T28,
instead of a source-by-destination table.

Bias each signed byte by 128 in a 16-bit half. Drain after at most three
capacity-32 trace blocks: at most 192 selected terms times the maximum encoded
value 256 is 49,152, so neither half can carry into the other. Subtract
`128 * selected_count` from each half on drain. This supports the full i8
coefficient range, including negating -128, and accumulates into the same i32
centered output. Compared with the retained kernel, the ballot count is
unchanged, source-byte traffic rises eightfold, and cached challenge traffic
doubles in bytes; in exchange it removes every queue access and barrier and
replaces 64 scalar additions per selected source with 32 packed additions.
The extra source scan is under 8 GiB for the measured T28 root source and is
not the roofline constraint on this machine.

Require the existing arbitrary-source CPU oracle, extended to extreme i8
coefficients, before one T28 treatment. Preserve proof bytes, transcript,
evaluation, commitment route, verifier acceptance, and the 90 GiB memory guard.
Retain only if packed root-fold GPU time is at most 1.25 s and complete opening
is at most 6.55 s, versus the fixed 1.805 s and 6.746238 s parent. A successful
kernel makes the 1.633 s CPU streamed consumer the root critical path, so it
then authorizes a matrix-stationary CPU recursive-prefix commit candidate. A
miss removes the packed tables and kernel without tuning tile sizes.

The full-column extreme-i8 CPU oracle passed, and the T28 proof, transcript,
evaluation, commitment, verifier, and memory checks remained exact. The timing
gate rejected the architecture decisively. Packed root-fold GPU time rose from
1.804525 s to 2.412536 s, its wall time rose from 2.058016 s to 2.715247 s, and
complete opening rose from 6.746238 s to 7.546650 s. Total GPU-active time rose
by 0.668153 s, while unified-memory and power contention also raised the CPU
consumer from 1.632966 s to 1.833557 s and root coefficient packing by 0.126020
s. The retained shared task scan and queue amortize source decoding better than
eight independent SIMDgroup scans; pair packing does not repay the duplicated
source and wider challenge traffic. Restore the retained kernel and do not tune
the scan tile or packed bias.

### Matrix-stationary streamed recursive prefixes

The retained root path overlaps Metal folding with CPU commitment of completed
Z chunks. The successor uses a large D64 inner matrix, while each of the eight
streamed calls contains only tens of complete witness blocks and at most seven
matrix rows. The generic CPU mat-vec therefore selects block parallelism: every
worker block walks the entire matrix. This supplies abundant Rayon tasks but
logically rereads a matrix with at least eight 4 MiB L2 tiles for every block.
The one-shot CPU baseline has hundreds of blocks and remains correctly served
by that mapping.

Use Akita's existing exact column-tiled mat-vec only when all of the following
hold: the normal small-row block route is otherwise eligible, there are 16--64
blocks, and the matrix spans at least eight cache tiles. A tile then retains all
block accumulators, reuses its matrix working set across the streamed batch,
and still exposes at least eight Rayon tasks. The arithmetic, CRT parameters,
reconstruction order within each accumulator, inner rows, commitment, proof,
and transcript are unchanged. Larger one-shot commits and smaller matrices keep
block parallelism, so this is an accelerator-pipeline host optimization rather
than a new CPU control.

The expected local effect is to reduce the 1.632966 s streamed consumer to
1.25--1.40 s and reduce shared-memory/power contention enough to lower the
concurrent 1.804525 s root GPU span to 1.65--1.75 s. Require the existing
block-parallel/column-tiled fp128 parity test, then one T28 treatment. Retain
only if the consumer is at most 1.40 s, root GPU time at most 1.75 s, root wall
time at most 1.90 s, and complete opening at most 6.50 s, with exact proof,
transcript, evaluation, commitment, verifier, and memory guards. A miss restores
the original routing predicate without threshold tuning.

The fp128 parity test passed, and the single T28 treatment preserved every
correctness and memory guard, but the route failed by a wide margin. The
matrix-stationary mat-vecs themselves took 2.270384 s, the streamed consumer
rose from 1.632966 s to 3.449563 s, root wall time rose from 2.058016 s to
3.767709 s, and complete opening rose from 6.746238 s to 8.751451 s. Root GPU
time also increased to 1.895570 s under the extra host contention. At this
shape, cache reuse cannot repay the loss of block-level Rayon fanout. Restore
block parallelism and remove the diagnostic bucket without trying intermediate
block or tile thresholds.

### T28 resident Stage-2 source handoff

The max-scale-only diagnostic separates occupancy from orchestration.  In the
exact T28 treatment, Stage 2 took 1.254484 s while its Metal timestamp intervals
totaled 0.575855 s.  The host spent 0.125444 s preparing the direct session,
0.119075 s constructing dispatch buffers, 0.035852 s folding canonical state,
and 0.011136 s constructing per-round data.  Session construction occupied
0.234597 s, overlapping the reported buffer and source-command intervals.  The
first two terms include an unnecessary 512 MiB boundary traversal: the
`2^25` canonical fp128 lane weights are converted to an equal limb vector and
then copied into a new shared Metal buffer, although the canonical fp128 storage
is already the kernel ABI.

Revisit the earlier borrowed-lane treatment only at its actual target scale and
combine the inseparable pieces.  Move the canonical lane-weight vector out of
the direct prover state, expose its bytes through the existing alignment-checked
no-copy Metal buffer helper, and keep the owner alive for the session.  The GPU
owns all lane folds after the coefficient boundary; two private geometric
scratch buffers preserve the borrowed source, and a 32-byte final readback
restores the one canonical lane value needed by the final-claim check.  The host
continues to fold equality, alpha, and sparse additional terms.  This changes no
round polynomial, transcript event, proof field, or verifier equation.

The measured removable ceiling is 0.28--0.36 s: about 0.125 s of conversion,
most of 0.119 s of initial buffer setup, and 0.036 s of duplicate host folding,
with allocator traffic as upside.  It cannot close the 1.433015 s five-times gap
alone; it is retained only as a prerequisite resident boundary for a later
static/dynamic Stage-2 split.  Require focused direct-proof parity, then one
exact T28 treatment.  Preserve proof bytes, transcript, evaluation, commitment,
verifier, schedule, route, and memory guards.  Retain only if Stage 2 is at most
1.02 s, Stage-2 host preparation is at most 30 ms, Stage-2 buffer setup is at
most 55 ms, and complete opening is at most 6.75 s when root coefficient packing
is within 100 ms of its 1.060554 s parent.  If root packing is outside that
noise band, the local gates decide retention but the run is not evidence toward
the end-to-end five-times claim.  Do not tune alignment thresholds after a miss.

Both focused direct-proof parity cases passed, including odd live lanes and
nonzero structured relation weights.  The exact T28 proof, transcript,
evaluation, commitment, verifier, schedule, route, and memory guards also
passed.  The performance gates rejected the mechanism.  Stage 2 fell only from
1.254484 s to 1.151672 s, host preparation fell from 125.444 ms to 102.360 ms,
and buffer setup fell from 119.075 ms to 95.163 ms.  Duplicate host binding did
fall from 35.852 ms to 8.989 ms, but the combined treatment removed only
102.812 ms from Stage 2.  Complete opening was 7.216811 s with a noisy
1.364550 s root packing scan, so it is not end-to-end evidence.  Remove the
borrowed source, extra ping-pong table, and final scalar handoff without tuning
alignment.  The diagnostic also prices the next boundary: sparse additional
terms take 13.872 ms and direct-state construction takes 18.102 ms; neither is
the missing mechanism.  The 102 ms host-preparation residual is the
structured-linear layout conversion, and its corresponding buffers dominate
the remaining 95 ms setup interval.

### Direct D-role product dispatch at T28

The exact diagnostic treatment localizes 416.908 ms of the 477.081 ms
relation-opening-row phase in `compute_relation_v`. This is not an inherent
D64 floor. The generic ring-switch entry admits Metal only for the root D512 A
relation, so the D64 opening product silently takes its CPU fallback even though
Akita Metal already has an exact D64 digit-row operation for this shape,
including retained quotient rows. The T28 admission oracle covers one row, two
vectors, and 1,409,024 D64 digit columns.

Route D-only opening products through `DigitRowsComputeBackend` and consume its
negacyclic rows plus retained quotients directly. CPU and other backends keep
their existing exact implementation through the same generic operation. This
changes no matrix, digit order, arithmetic result, proof field, transcript
event, schedule, or verifier equation; it removes an accidental operation-trait
detour and one reported CPU fallback.

The removable ceiling is about 0.30--0.38 s, not the full five-times gap. Run
the existing D64 product CPU/Metal parity test, then one exact T28 treatment.
Require the frozen proof digest, transcript, evaluation, commitment, verifier,
schedule, route, and memory guards. Retain only if `compute_relation_v` is at
most 180 ms and the opening-row phase is at most 240 ms. When root coefficient
packing is within 100 ms of its 1.060554 s retained value, also require complete
opening at most 6.55 s; otherwise the local gates decide retention and the run
does not update the end-to-end record. A miss restores the fused ring-switch
entry without tuning column-partial width.

The focused D64 product parity test passed. The exact T28 treatment preserved
the proof digest, transcript, evaluation, commitment, verifier, schedule,
route, and memory guards. `compute_relation_v` fell from 416.908 ms to
102.155 ms and its enclosing opening-row phase fell from 477.081 ms to
163.439 ms, clearing both local gates. Reported CPU fallbacks fell from 28 to
26 and CPU-tail work fell by 92,654,336 units. Retain the generic product
dispatch. Complete opening was 7.357703 s, but root coefficient packing drifted
to 1.469749 s, 409 ms above the retained value and outside the declared noise
band; this run therefore does not replace the 6.746238 s end-to-end record.
The supported 314.8 ms local gain projects that record to roughly 6.43 s and
leaves about 1.12 s to remove for five times.

### Barrier-free four-byte root fold with direct successor digits

The retained subring-owned root fold is fully occupied at T28: it launches one
256-thread group for each of 262,144 positions. Its 1.804525 s GPU interval is
therefore not a small-grid problem. For each position it scans 30,720 source
tasks in 120 tiles, compacts them into eight residue queues, and crosses two
threadgroup barriers per tile. Each selected source is then handled as 64
separate signed-byte loads and additions. The root's concurrent host consumer
takes 1.632966 s, of which 927.356 ms is the balanced decomposition of the
accepted centered output and 688.970 ms is the successor commitment prefix.
A faster fold alone would consequently expose the host branch rather than
reduce root wall time by its full local gain.

Use one SIMDgroup for each of the eight embedded D64 residues, but give every
SIMDgroup its own read-only scan of the packed selectors. A ballot filters that
scan to the group's residue without a threadgroup queue or barrier. Sixteen
lanes own four consecutive D64 destinations each; the other sixteen lanes take
alternate selected sources for the same destinations. Before the fold, a
small Metal expansion kernel stores one positive and one negative biased
four-byte word for each of the 64 cyclic starts of every challenge. A lane
usually loads one word; the unique four-destination group that straddles the
negacyclic boundary combines the corresponding positive and negative words
with a byte mask. Accumulate at most 63 sources per lane half, unpack the four
bytes, subtract the exact bias, and combine the two halves with one SIMD
shuffle. No byte can carry because every partial byte is at most
`63 * 4 = 252`.

At T28 the packed table is 15,360 challenges times 512 bytes, or exactly
7.5 MiB. The mapping repeats the 8.05 GB selector scan eight times, but retains
the existing approximately 487 GB of logical challenge bytes: sixteen
coalesced four-byte loads replace 64 scalar loads for each selected source.
It removes 62.9 million queue barriers across the grid and reduces accumulator
add instructions by four. The extra selector traffic has a roughly 0.12 s
bandwidth floor at 546 GB/s; table construction and storage are small beside
the retained challenge traffic.

Pair this kernel with a generic optional streaming hint. The root sink requests
its existing balanced base and digit count; the kernel writes those exact
digits beside each centered coefficient after the final accumulation. The sink
then appends the device-produced bytes directly and skips its 927 ms CPU
decomposition. Backends that cannot produce the hint, roots with more than one
inner digit, coefficients outside `[-2, 2]`, and non-D64 embeddings retain the
current exact path. This changes no sampled challenge, arithmetic value,
message, transcript event, proof field, schedule, or verifier equation.

Require an embedded-D64 oracle that checks both centered coefficients and every
balanced digit, plus the arbitrary-position fallback oracle. Then run one exact
T28 treatment. Preserve the frozen proof digest, transcript, evaluation,
commitment, verifier, schedule, route, and 90 GiB memory guard. Retain locally
only if packed-decompose GPU time is at most 1.30 s, its streamed consumer is at
most 0.85 s, and root wall time is at most 1.50 s, versus 1.804525 s,
1.632966 s, and 2.063250 s. If root coefficient packing is within 100 ms of its
1.060554 s retained value, also require complete opening at most 6.00 s;
otherwise local gates decide retention and the run is not an end-to-end record.
A miss restores the queue kernel and host decomposition without tuning batch
size, table layout, or admission thresholds.

Both focused oracles passed, including enough selected sources to cross the
126-source packed-byte flush, and the exact T28 proof preserved the frozen
proof digest, transcript, evaluation, commitment, verifier, schedule, route,
and memory guards. Direct successor digits behaved as modeled: the streamed
consumer fell from 1.632966 s to 668.642 ms, with digit append taking 37.233 ms
and recursive prefix commitment taking 592.671 ms. The fold kernel failed its
local gate catastrophically, however. Packed-decompose GPU time rose from
1.804525 s to 10.011945 s, root wall time rose from 2.063250 s to 10.193080 s,
and complete opening rose to 15.080456 s.

The compact lookup preserved logical byte count but not transaction locality.
For one source, the sixteen destination owners address four-byte words at
different cyclic starts, spreading 64 useful bytes across roughly 512 table
bytes instead of the retained kernel's two contiguous 32-byte destination
ranges. The source-dependent rotations also defeat reuse across neighboring
tasks, and each SIMDgroup repeats the selector address generation. Reject the
kernel and optional digit handoff together as preregistered; do not tune the
table layout. The measured consumer result remains evidence that direct digits
are worth pairing with a future root kernel whose primary output is already
coalesced.

### Source-contiguous global residue queues

The rejected barrier-free treatment localized its loss to transaction layout,
not packed arithmetic or direct digits. Preserve those two successful pieces
but change both sides of the source handoff. One 256-thread group scans a
512-task tile once, with each thread decoding two selectors. SIMD ballots count
the eight residues per producer group; one short group-wide prefix places all
live tasks into a single 512-entry array in eight contiguous residue ranges.
The eight consumer SIMDgroups then drain one range each. This uses about 2.5 KiB
of threadgroup state, versus the retained partitioned queue's roughly 8 KiB,
and crosses three barriers per 512 tasks: 180 barriers per T28 position instead
of 240.

Transpose the packed challenge expansion to `[challenge][source_high][quad]`.
Each `(challenge, source_high)` row contains the sixteen already-signed,
four-byte-biased destination quads consecutively. The row is 64 bytes, so the
sixteen active lanes issue one coalesced cache-line access per queued source;
the rejected treatment spread those same 64 useful bytes over about 512 bytes.
The table is 15,360 times 64 times 64 bytes, or 60 MiB at T28. It is larger
than the rejected 7.5 MiB lookup but restores the physical traffic model:
approximately 487 GB of challenge rows plus one 8.05 GB selector scan. Four
coefficient additions remain packed into one carry-safe `uint`, and the final
owner writes both the centered result and its requested balanced successor
digits.

This is a new mapping rather than a table-size retry: the producer scan changes
from eight independent scans to one global compaction, the queue changes from
64 fixed producer partitions to eight globally contiguous ranges, and the
consumer access changes from strided words to a single coalesced row. Require
the centered/digit CPU oracle across a 512-task tile plus the arbitrary-D512
fallback oracle. Then run one T25 credibility treatment. Preserve every frozen
correctness, route, and memory guard; require packed-decompose GPU time at most
220 ms and its consumer at most 120 ms. Only a pass authorizes T28.

At T28 retain only if packed-decompose GPU time is at most 1.35 s, consumer time
is at most 0.85 s, and root wall time is at most 1.55 s. If root coefficient
packing is within 100 ms of 1.060554 s, also require complete opening at most
6.05 s. Otherwise the local gates decide retention and the run is not an
end-to-end record. A miss restores the retained dense queue and host digits;
do not tune the tile, queue, or table dimensions.

The centered/digit and arbitrary-position oracles passed, but the single T25
credibility treatment rejected the architecture before T28. Packed-decompose
GPU time was 452.978 ms, versus 273.624 ms for the retained parent and more than
twice the 220 ms gate. Direct digits again helped the host branch, reducing its
consumer from 374.807 ms to 231.607 ms, but root wall time still rose from
429.910 ms to 500.384 ms and complete opening rose from 1.966950 s to
2.054479 s. Every proof, transcript, evaluation, commitment, verifier, route,
and memory guard remained exact.

Restoring one source scan and coalesced 64-byte rows removed the catastrophic
transaction amplification, but the 60 MiB source-specific table turns the
logical 64-byte update into real streaming traffic and the global prefix adds a
third barrier. Those costs exceed the packed-add reduction. Remove the table,
queue, and digit handoff without a T28 run. Together with the partitioned,
pair-packed, microtiled, barrier-free, large-tile, and source-contiguous results,
this closes shared-queue and dense-row remappings of the current root equation.

### Commit-retained residue index and direct successor digits

The closed root mappings all rebuild selector locality after the fold challenge
is known.  That restriction is unnecessary: selector positions and their D64
residue classes are witness data, not Fiat--Shamir data.  The Metal commitment
already consumes the same immutable packed source before the timed opening and
may retain a prover-only acceleration hint without changing the commitment,
proof, transcript, or verifier.

For each root position and 256-task tile, build eight contiguous residue runs.
Each live run entry is the existing 20-bit `(challenge_index, source_high)`
record stored in a `u32`; eight `u16` counts describe the runs.  Slots remain
fixed at 256 per tile, so no global prefix or variable-size allocation is
needed.  The T28 source has about 8.05 billion task slots: records require about
32.2 GB and counts about 0.50 GB.  Combined with the measured 25.7 GB proof
peak, the conservative resident projection is below 60 GB and the existing
90 GiB guard.  Index construction reads the source once and writes the records
once.  Its traffic floor is about 0.08 seconds at the measured 406 GB/s copy
rate, but its per-tile ballot and barrier work must be charged to the Metal
commit phase and reported separately.

The opening kernel assigns one SIMDgroup to each embedded D64 residue.  A group
walks only its prebuilt run, while its 32 lanes retain the two coalesced
destination accumulators of the current exact subring kernel.  There is no
producer queue and no threadgroup barrier.  Challenge traffic and arithmetic
remain the same 64 dense signed-byte loads/adds per selected source; this
treatment removes only the measured partition/synchronization machinery.  At
the final store, each lane also performs the existing balanced decomposition
and writes the exact successor digit planes.  The generic chunk sink advertises
the requested basis and digit count; CPU and unsupported Metal routes retain
host decomposition.

Against the retained T28 root, the calibrated target is 1.05--1.25 seconds of
indexed GPU work and 0.60--0.75 seconds in the remaining recursive-prefix
consumer, for a 1.25--1.45 second root wall interval instead of 2.063 seconds.
The supported D64 dispatch projects the complete parent to roughly 6.43
seconds, so this mechanism is necessary but not sufficient: its expected
0.61--0.81 second complete gain leaves 0.31--0.51 seconds for the already
localized resident Stage-2, Stage-1 factor, and command/session boundaries.

First require focused index-order, centered-fold, and emitted-digit parity.
Then run one T25 credibility treatment.  It must preserve proof bytes,
transcript, evaluation, commitment, verifier, schedule, required route, and
fallback counts; indexed GPU time must be at most 0.22 seconds, the streamed
consumer at most 0.15 seconds, root wall time at most 0.30 seconds, and complete
opening at most 1.84 seconds.  Only a pass authorizes one T28 treatment.  At
T28 retain only if indexed GPU time is at most 1.30 seconds, the consumer at
most 0.80 seconds, root wall time at most 1.50 seconds, complete opening at most
5.90 seconds when root packing is within its declared noise band, peak RSS is
at most 90 GiB, and index construction adds at most 1.0 second to Metal commit.
Any exactness, route, memory, or local timing miss removes the index and digit
hint without tuning tile dimensions.

The focused parity test passed for index order, centered coefficients, chunk
offsets, and balanced-digit reconstruction.  Two identical T25 treatments also
preserved the commitment, proof digest, transcript, evaluation, verifier, route,
and memory guards.  They reported one indexed call and 134,217,728 direct digit
bytes.  Root GPU time was stable at 155.42 ms and root wall time was 287.97--
302.29 ms, clearing the two mechanism gates.  The streamed consumer was
259.58--269.10 ms because that counter includes the exact successor commitment,
not only the eliminated decomposition.  Complete opening was 1.927--1.948 s;
Stage 2 was 0.519--0.527 s versus 0.408 s in the older T25 parent.  The candidate
therefore did not pass the preregistered T25 promotion gate.

Do not silently promote it.  One T28 run is nevertheless authorized as a
diagnostic, not a retained-result claim: the campaign target is now explicitly
T28-only, the root mechanism itself met its scale-predictive GPU and wall gates,
and the diagnostic directly measures whether the 8x index, root consumer, and
Stage-2 terms leave a tractable residual.  A miss still requires a new mechanism
before any confirmation run; a 5x diagnostic result still requires an unchanged
confirmation before the claim is accepted.

The first T28 diagnostic localized a lifetime defect and a needless index-width
cost.  Root arithmetic itself reached 1.197926 s, but the 32.716 GB `u32` index
remained cached after its only use.  Stage 1 rose to 0.817918 s and Stage 2 to
1.995420 s, versus 0.573203 s and 1.160862 s in the retained T28 parent.  Complete
opening was 7.574916 s.  Root wall was 1.856189 s, with a 0.771274 s streamed
consumer, and index construction took 1.804118 s wall / 1.004396 s GPU.  Exactness,
route, and the host RSS guard passed, but every promotion timing gate failed.

The next bounded treatment changes the index representation and lifetime, not
the protocol or tile geometry.  A record needs only its 8-bit task offset within
the fixed 256-task tile and its 6-bit source high part, so it fits in `u16`.
Lane zero reconstructs `(column, trace_block)` from the tile-local task once and
broadcasts the challenge index across the SIMDgroup.  The opening takes, rather
than clones, the one-shot cache entry, releasing it before Stage 1.  Exact index
storage is then 2,076,180,480 bytes at T25 and 16,609,443,840 bytes at T28.

Require the same focused parity test, then one T25 treatment.  It must report
the exact compact byte count, one indexed call, exact proof/transcript/verifier,
root GPU at most 0.20 s, root wall at most 0.34 s, and index construction at most
0.20 s.  A pass authorizes one T28 treatment.  At T28 require exact compact bytes,
index construction at most 1.20 s, root GPU at most 1.30 s, root wall at most
1.60 s, Stage 1 at most 0.68 s, Stage 2 at most 1.40 s, and complete opening at
most 6.20 s.  These are mechanism gates, not a 5x claim.

The compact record family did not clear its GPU gate.  Direct task-to-challenge
division produced 0.286315 s of T25 root GPU time.  Replacing division with a
30 KiB invariant lookup reduced that to 0.220552 s, and lane-zero broadcast was
0.222433 s.  All forms were exact; the latter two used 2,076,211,200 index bytes,
and the broadcast form reached 1.806573 s complete.  They still miss the 0.20 s
root-GPU bar and project materially above the accepted `u32` kernel at T28.
Close compact records without a T28 run.  Restore direct `u32` challenge/source
records while retaining the one-shot cache eviction.

### Cached recursive-commit matrix transforms

Each streamed root Z chunk invokes the same successor inner commitment, but the
current Metal call allocates and recomputes the setup-matrix NTT every time.
The transform depends only on `(setup, D, n_a, active_a_cols)`, not on witness
digits, fold challenges, or the chunk's block count.  Cache the exact private
transform on `MetalPreparedSetup`; static E/T, all Z chunks, and a matching tail
commit can then reuse it.  The cache key is setup-scoped, so a larger unrelated
matrix cannot alias an exact recursive geometry.  Proof values and transcript
order are unchanged.

Require D64 and D128 repeated-commit parity, with one cache miss followed by one
hit, then one T25 treatment.  Preserve every existing exactness and route guard,
report at least one transform hit, keep root GPU at most 0.18 s, reduce the root
Z commitment span below 0.20 s and root wall below 0.28 s, and keep complete
opening below 1.82 s.  A pass authorizes one T28 treatment with the restored
one-shot index lifetime.  There require root GPU at most 1.30 s, root wall at
most 1.60 s, Stage 1 at most 0.68 s, Stage 2 at most 1.40 s, and complete opening
at most 6.10 s.  A local miss removes the transform cache without cache-shape or
chunk-count tuning.

The repeated D64/D128 oracle passed, but the T25 treatment reported zero cache
hits and zero misses.  It was exact and reached 1.788661 s complete, 0.155321 s
of root GPU time, and 0.283944 s of root wall time, but its 0.231385 s root Z
commitment span remained unchanged.  The cause is schedule geometry, not cache
lookup failure: after root folding, T25 reinterprets the digit byte stream in
the first recursive level's D256 ring, which is outside this Metal kernel's
D64/D128 domain.  Thus the preregistered T25 cache gate rejects the candidate
but does not test the T28 mechanism.

The T28 diagnostic was exact and reached 6.198161 s complete, 1.188603 s of
root GPU time, 1.793973 s of root wall time, 0.577206 s in Stage 1, and
1.154537 s in Stage 2.  It also reported zero cache hits and zero misses.  The
schedule does reinterpret each root chunk as a D128 witness with 16,384
positions per block, but those streamed prefix commitments execute on the CPU
commit cluster.  The opening backend is Metal; the commit and tensor clusters
remain CPU so that prefix work can overlap the root fold.  The recursive Metal
cache is therefore unreachable on the actual proof path.

An exact-shape standalone probe closes the question of routing that cluster to
Metal.  For D128, one row, 16,384 columns, and 32 blocks, the first Metal call
took 161.242 ms wall / 153.615 ms GPU, including 9.474 ms for the matrix
transform.  A cached call still took 143.334 ms wall / 137.368 ms GPU.  Eight
chunks project to about 1.15 s of additional device work, compared with the
measured 0.702 s CPU prefix span, before accounting for contention with the
1.189 s root fold.  Do not integrate the recursive Metal route into the root
stream.  Keep the generic cache candidate isolated until cleanup; it is not a
5x mechanism.

The T28 diagnostic leaves 0.884938 s above the 5x threshold of 5.313223358 s.
Its root GPU and CPU consumer account for 1.980041 s of work but overlap by only
0.186068 s inside a 1.793973 s wall interval.  The next bounded experiment uses
two disjoint output/digit buffer pairs and submits chunk `i + 2` only after the
CPU has consumed chunk `i`.  With GPU chunks near 149 ms and consumers near
99 ms, the device remains critical and the ideal root wall is about 1.29 s.
Require exact centered coefficients, balanced digits, callback order, proof,
transcript, evaluation, commitment, verifier, and required routes.  Promote a
T28 treatment only if root wall is at most 1.45 s without increasing root GPU
above 1.30 s; otherwise remove the pipeline candidate.

The treatment was exact but failed the mechanism gate.  Complete opening rose
to 6.576235 s, root wall rose to 1.987200 s, and measured root GPU work rose to
1.801904 s.  The consumer fell to 0.687141 s, but repeated fresh shared-buffer
writes and a 46.573 ms readback replaced the full output's efficient no-copy
backing.  This is not an overlap limitation that slot-count tuning can fix.
Remove the double-buffer implementation and retain the original full-buffer
dispatch.

### Nibble-packed indexed root gather

The retained indexed kernel remains bandwidth-bound after queue and lifetime
changes.  For every selected source it loads 64 signed challenge bytes: two
scalar bytes in each of 32 SIMD lanes.  At T28 this is about 515 GB of logical
challenge traffic.  The measured 1.188603 s GPU interval therefore corresponds
to about 433 GB/s before record, count, output, and instruction traffic.  Queue
tuning cannot remove this floor.

Production D64 challenges have coefficients in `[-2, 2]`.  Store biased
coefficients in four-bit nibbles and precompute eight source-phase rows.  Each
row has eight 32-bit words, where one word supplies the eight consecutive
destination coefficients owned by a lane.  Split each SIMDgroup into four
eight-lane groups so that it consumes four independent index records at once.
The eight loads for one record are a contiguous 32-byte permutation; this is
32 bytes per selected source instead of 64.  Accumulate even and odd nibbles as
packed bytes for at most 63 records per batch, then debias into signed `int4`
accumulators.  The 252-record batch bound prevents cross-byte carry.  Preserve
the current index, full output buffer, direct balanced digits, callback order,
and protocol.

This differs from the rejected preexpanded challenge table: that layout spread
64 useful bytes for a source across roughly 512 bytes and required a 60 MB
table.  The phase table is 256 bytes per challenge (about 3.75 MB at T28), and
all 32 useful bytes for one source are contiguous.  At the measured device
bandwidth its challenge-traffic floor is about 0.47 s.  Allowing record traffic,
nibble expansion, shuffles, and occupancy gives a preregistered root-GPU range
of 0.70--0.90 s.

First require focused centered-coefficient and balanced-digit parity, including
an out-of-range challenge fallback.  Then run one exact T25 treatment.  Promote
to T28 only if root GPU is at most 0.13 s (from 0.155321 s) with exact proof,
transcript, evaluation, commitment, verifier, and indexed-route evidence.  At
T28 require root GPU at most 0.90 s, root wall at most 1.50 s, and complete
opening at most 5.90 s.  This is a milestone mechanism gate, not a 5x claim;
the remaining gap must then come from the measured Stage 1/2 overhead without
an invasive protocol change.

The focused test passed with a full 256-record single-residue tile, exercising
the 252/4 batch boundary while matching CPU centered coefficients and every
balanced digit.  The exact T25 treatment also passed: root GPU fell from
0.155321 s to 0.094822 s, root wall was 0.283992 s, and complete opening was
1.781748 s.  Proof bytes, transcript, claimed evaluation, commitment, verifier,
indexed-route count, and direct-digit count all matched.  The root wall is now
limited by the 0.267079 s streamed CPU consumer at this scale.  This clears the
preregistered local gate and authorizes one T28 treatment.

The exact T28 treatment preserved the frozen proof digest, transcript, claimed
evaluation, commitment, verifier, indexed route, direct-digit count, and memory
guard.  Root GPU time fell from 1.188603 s to 0.725943 s, a 1.64x local speedup,
and cleared the 0.90 s gate.  Root wall fell from 1.793973 s to 1.549965 s but
missed the 1.50 s gate because the CPU prefix consumer rose from 0.791438 s to
0.918106 s.  Complete opening was 6.230682 s, versus 6.198161 s in the parent,
so this is not end-to-end promotion evidence.  The unrelated root coefficient
packing span also rose by 0.157594 s while relation preparation fell by
0.188158 s, confirming that one treatment cannot resolve those concurrent host
spans below roughly 0.2 s.

Retain the packed kernel mechanism: its GPU result reproduced the T25 ratio at
T28 and removed 0.462660 s of measured device work without changing any proof
artifact.  Do not claim a complete-proof gain from this run.  Against the exact
5x threshold, the measured residual is 0.917459 s.  The hard lower bounds from
this record are 0.918106 s for the streamed CPU prefix work, 0.573146 s for
Stage 1 (0.391323 s of Metal commands), and 1.201278 s for Stage 2 (0.561451 s
of Metal commands).  Further progress must reduce those work terms or their
boundaries; additional root queue or command-buffer tuning cannot supply the
remaining gain.

### Ternary indexed root gather

The root's embedded challenges come from the production D512 family, whose 19
nonzero coefficients are all `+/-1`; the nibble kernel conservatively supports
`+/-2`.  Bias the actual ternary table by one and store sixteen two-bit values
per word.  Sixteen source phases and four destination quads still require 64
words per challenge, so preparation storage remains about 3.75 MB at T28, but
only four contiguous words (16 bytes) are read per selected source.

Split each SIMDgroup into eight four-lane source groups.  A lane owns sixteen
destinations.  Masks at bit offsets 0, 2, 4, and 6 turn one word directly into
four packed-byte accumulators; each byte holds four destinations separated by
four coefficients.  At most 32 records reach one source group in a 256-record
tile, so the biased sum is at most 64 and cannot carry across bytes.  Negacyclic
signs replace a selected biased byte `b` by `2-b`, followed by one exact bias
subtraction per group.  Custom challenges containing `+/-2` use the existing
generic dense Metal route.

The logical challenge traffic falls from about 257 GB to 129 GB.  The measured
nibble kernel sustains an effective 354 GB/s including its other traffic and
integer work, giving a traffic-only floor near 0.36 s.  Allow 0.45--0.58 s at
T28 for extra record decoding, four packed accumulators, and final SIMD
reductions.  Require the existing full-residue 256-record centered/digit oracle,
then one exact T25 treatment.  Promote only if root GPU is at most 0.075 s,
with all proof, transcript, evaluation, commitment, verifier, route, and memory
guards intact.  At T28 require root GPU at most 0.58 s, root wall at most
1.40 s, and complete opening at most 6.05 s.  A local miss restores the nibble
kernel without subgroup or packing-width tuning.

The focused synthetic ternary oracle passed, but the exact T25 treatment did
not select the route: the production root includes magnitude-two coefficients.
The qualification guard correctly used the generic dense Metal kernel, reported
zero indexed calls and zero direct-digit bytes, and took 0.224967 s of root GPU
time.  This both misses the 0.075 s gate and falsifies the production-family
premise.  Do not change the sampled challenge distribution to fit the kernel.
Restore the nibble route and do not run T28.

### Root stream contention panel

The nibble treatment changes the root balance: 0.725943 s of GPU work now runs
beside 0.918106 s of CPU prefix work, but their 1.549965 s wall interval hides
only 0.094084 s. The eight queued chunks therefore do not provide useful
pipeline overlap at T28. They instead make the all-core D128 prefix commits
compete with the indexed kernel for the unified memory and power domains.

Measure one exact T28 panel at 1, 2, 4, and 8 equal position chunks from the
same release binary. The diagnostic switch changes only execution boundaries;
the centered coefficients, balanced digits, prefix merge order, transcript,
proof, and verifier remain fixed. Retain the fastest fixed chunk count only if
all exactness and route guards pass and root wall improves by at least 100 ms
against the same-panel eight-chunk control. The one-chunk endpoint deliberately
removes overlap: it is useful if lower GPU/CPU contention and one larger CPU
commit call repay serialization. Remove the diagnostic environment switch
after selecting or rejecting the fixed policy.

All four T28 proofs were exact. The same-binary results were:

| chunks | root wall | root GPU | consumer | complete |
|---:|---:|---:|---:|---:|
| 8 | 1.520660 s | 0.722430 s | 0.864705 s | 6.035833 s |
| 4 | 1.611182 s | 0.734020 s | 0.847616 s | 6.347841 s |
| 2 | 1.778337 s | 0.732743 s | 0.878090 s | 6.606341 s |
| 1 | 2.141562 s | 0.761734 s | 0.762775 s | 7.352025 s |

Larger chunks do not lower either kernel work or CPU commit work enough to
repay their delayed first handoff. The one-chunk endpoint also exposes about
0.62 s outside the reported GPU and consumer intervals, so serializing the two
workers is especially harmful. Retain eight chunks and remove the diagnostic
switch. The fresh eight-chunk control is the current end-to-end record, 4.401x
against the frozen CPU time, and leaves 0.722610 s to the 5x threshold.

### Paired Stage-2 prefix grid

The canonical two-round Stage-2 prefix asks for eight norm and eight relation
grid values. With the default omitted norm corner, both sets use the identical
eight nonzero grid points, but the retained kernel dispatches separate
threadgroups and repeats digit interpolation for each set. Pair the norm and
relation accumulators in one threadgroup per point. This halves point-group
count and shares digit loads and interpolation; it does not remove either
field-weighted accumulation, change the compressed grid, or change any proof
message.

The 63.292 ms T25 prefix is the local target. Because field arithmetic remains,
the expected improvement is 15--30%, not 2x. Require both direct Stage-2
CPU/Metal parity cases, including structured linear terms. Admit one T25 exact
treatment only if the focused tests pass. Retain for T28 only if Stage-2 GPU
time falls by at least 8% and Stage-2 wall does not regress; otherwise restore
the separate point groups without workgroup tuning.

Both focused parity cases passed and the T25 proof, transcript, evaluation,
commitment, verifier, route, and memory guards were exact. The local mechanism
missed: Stage-2 GPU time fell only from 213.195 ms to 206.520 ms (3.1%), and
Stage-2 wall fell from 478.578 ms to 470.237 ms. Field-weighted accumulation,
not repeated digit interpolation or point-group launch count, controls the
prefix. Reject without T28 and restore the separate point groups.

### Native-u32 fp128 multiplication

The Stage-2 prefix remained field-accumulation-bound after point pairing. The
shared field primitive currently implements a 128-by-128 product with sixteen
MSL `ulong` multiplies, then reduces modulo the fp128 pseudo-Mersenne prime.
Replace the product only with a seven-diagonal Comba accumulator using `uint`
low products, `mulhi` high products, and explicit carries. Apply the same
32-bit carry form to field-by-signed-i32 multiplication, which dominates the
compact Stage-1 and Stage-2 prefix terms. Keep the existing reduction and all
field representations unchanged.

The operation count does not predict the Apple GPU mapping: this wins only if
native 32-bit multiply-high and carry chains beat compiler lowering of 64-bit
integer multiplication. Require all Akita Metal tests that exercise field
arithmetic and the focused Stage-1/Stage-2 CPU parity cases. Then run one exact
T25 treatment. Retain for T28 only if aggregate Stage-1 plus Stage-2 GPU time
falls by at least 15% without increasing any exact Metal commitment or relation
failure; otherwise restore the original product immediately.

All 33 Akita Metal tests passed, including full and recursive commitments,
D64/D512 relations, and both sumchecks. The exact T25 treatment also preserved
every proof and route guard, but the performance gate rejected the primitive:
direct-range plus direct-relation GPU time rose from 373.044 ms to 497.424 ms,
or 33.3%. The explicit `mulhi` carry dependency chain is worse than the Metal
compiler's `ulong` lowering on this Apple GPU. Restore the original primitive
and do not tune Comba unrolling or limb width in this campaign.

### Dense known-balanced recursive-prefix commit

The current T28 root interval is 1.520660 s: 0.722430 s in the indexed Metal
producer and 0.864705 s in the CPU successor-commit consumer, with only
0.066475 s of overlap. Routing the consumer to the existing exact Metal
D128 commit is already falsified: a cached 32-block call costs 0.143334 s, or
about 1.15 s for the eight stream chunks, before contending with the producer.
Keep the CPU route.

The CPU consumer nevertheless uses the generic sparse digit mat-vec after the
root producer has supplied an authenticated `known_balanced_log_basis`. That
path first validates every byte and then tests every D128 digit plane for zero
inside the NTT mat-vec. The random fold response is dense; those scans do not
change its arithmetic and almost never skip a transform. Dispatch a
known-balanced recursive source through the existing dense digit mat-vec,
which removes both scans while preserving the same cached setup transform and
exact CRT arithmetic. Unknown sources retain validation and sparse skipping.
This is an execution-only CPU-consumer change, not a proof or schedule change.

First require the recursive witness and root-fold tests that exercise known
balanced prefixes. Then run one exact T25 treatment. Promote to T28 only if the
root Z-prefix span is at most 0.225 s (from 0.246202 s), the streamed consumer
is at most 0.245 s (from 0.267079 s), and every proof, transcript, evaluation,
commitment, verifier, route, and memory guard passes. At T28 retain only if the
root Z-prefix span and consumer each improve by at least 80 ms and complete
opening improves by at least 50 ms against the 6.035833 s same-campaign record.
The candidate is complementary to Stage 1/2 work; it is not by itself a 5x
claim.

The Akita prover suite exercised the affected recursive-commit path before two
unrelated schedule-generation tests failed on the dirty campaign parent. The
exact T25 treatment preserved all proof and route guards. Its root Z-prefix
span fell only from 0.246202 s to 0.236235 s, and its streamed consumer fell
from 0.267079 s to 0.258596 s. Both 3--4% changes miss the local gates, while
complete opening rose from 1.781748 s to 1.805370 s. The consumer is controlled
by the exact CRT mat-vec, not its validation and zero-plane scans. Restore the
sparse path and do not run T28.

### P-core-sized streamed D128 commit tasks

The fresh T28 root interval contains 0.722430 s of Metal work and 0.864705 s of
CPU prefix commitment but takes 1.520660 s wall, so only 0.066475 s overlaps.
The existing producer already submits all eight disjoint commands before it
waits for the first callback; another queue or buffer pipeline cannot create
the missing overlap. The 16-worker CPU mat-vec instead saturates all 12
performance and four efficiency cores while the GPU is active, forcing the two
branches to share the package and memory controller nearly serially.

For only the measured large streamed shape (D128, one A row, at least 16,384
columns, and 16--64 blocks), group blocks into 12 ordered Rayon tasks. Each
task processes two or three blocks sequentially. This keeps the performance
cores occupied, avoids scheduling the four efficiency cores as independent
matrix walkers, and gives each worker short-term reuse of the transformed A
row. Every block still runs the same exact CRT/NTT operations and results are
flattened in canonical block order. Other commit shapes, including the frozen
CPU control's one-shot path, retain ordinary block parallelism.

Require the affected Akita commit composition tests, then one exact T28 Metal
treatment. Preserve proof bytes, transcript, evaluation, commitment, verifier,
schedule, route counters, and the memory guard. Retain only if the streamed
consumer is at most 1.05 s, root GPU is at most 0.80 s, root wall is at most
1.30 s, and complete opening is at most 5.85 s against the 6.035833 s record.
Any miss restores per-block parallelism without trying intermediate task
counts; the fixed 12 has a hardware rationale, not a search rationale.

The exact T28 treatment preserved the proof digest, transcript, evaluation,
commitment, verifier result, required routes, and memory guard, but failed both
end-to-end timing gates. The streamed consumer was 0.886344 s and root GPU work
was unchanged at 0.724338 s, yet root wall rose from 1.520660 s to 1.562907 s
and complete opening rose from 6.035833 s to 6.207277 s. Grouping work onto 12
tasks neither created useful CPU/GPU overlap nor exposed meaningful transform
reuse. Restore ordinary block parallelism and do not search intermediate task
counts.

### Cross-stage compact-witness residency

Stage 1 and Stage 2 consume the same `Arc<[i8]>` compact witness. The retained
Metal path nevertheless allocates and copies that 1.344 GB source into a new
shared buffer for each session. At T28, Stage-2 session setup takes 180.888 ms;
the aggregate Metal upload/setup counter is 256.775 ms. Stage 1 has completed
all commands before Stage 2 starts, so its immutable source buffer can be moved
between sessions without synchronization or protocol changes.

Retain the Stage-1 buffer behind the exact `Arc` allocation identity and take it
when Stage 2 presents the same allocation. A weak source reference prevents an
address-reuse match and lets abandoned entries be pruned. A miss performs the
existing copy. Proof messages, transcript operations, field tables, fold order,
and command boundaries are unchanged. The saved-work ceiling is one 1.344 GB
allocation-and-copy, not the whole Stage-2 setup span.

Require direct Stage-1 and Stage-2 parity tests, then one exact T25 treatment.
Retain for T28 only if the T25 Stage-2 session-setup span falls by at least
15 ms without increasing Stage-1 wall time, and every proof, transcript,
evaluation, commitment, verifier, route, and memory guard passes. At T28 require
the Stage-2 session-setup span to fall by at least 40 ms and complete opening to
improve by at least 30 ms against the 6.035833 s record. A miss restores the
copy; do not add pointer-only or digest-based cache matching.

All three focused direct-proof parity tests passed. The exact T25 treatment
also preserved the proof digest, transcript, evaluation, commitment, verifier,
routes, and memory guard, and reported three exact-allocation reuse hits across
the recursive levels. Stage-2 session setup fell from 68.292 ms to 57.221 ms,
only 11.071 ms, while complete opening rose from 1.781748 s to 1.798787 s. The
local gate rejects the mechanism, and its measured T28 projection is below the
40 ms promotion threshold. Restore independent session buffers and do not run
T28.

### SIMDgroup sumcheck partial reductions

Every ordinary direct-range and direct-relation workgroup currently stores
three or four fp128 accumulators for all 256 threads, then crosses nine full
threadgroup barriers to reduce them. The two-round Stage-2 prefix already uses
the appropriate Apple-GPU hierarchy: reduce within each 32-lane SIMDgroup,
store eight sums per accumulator, cross one barrier, and let the first
SIMDgroup finish. Apply that exact pattern to the ordinary Stage-1 and Stage-2
partial kernels and their final reducer. Threadgroup storage falls from 16 KiB
to 512 bytes for Stage 1 and from 16 KiB to 384 bytes for Stage 2. Contributions,
canonical field additions, partial layout, messages, and command boundaries are
unchanged.

Require all three direct-proof parity tests, then one exact T25 treatment. The
retained T25 parent spends 373.044 ms of GPU time in the two direct sessions.
Promote to T28 only if their aggregate is at most 345 ms, complete opening is at
most 1.75 s, and all proof, transcript, evaluation, commitment, verifier, route,
and memory guards pass. At T28 retain only if aggregate direct GPU time improves
by at least 70 ms and complete opening improves by at least 50 ms against the
6.035833 s record. A T25 miss restores the shared-memory tree without tuning
thread count or SIMD width.

The three focused parity tests and the exact T25 proof passed every correctness,
route, and memory guard. Performance moved in the wrong direction: direct-range
plus direct-relation GPU time rose from 373.044 ms to 388.917 ms, and complete
opening rose from 1.781748 s to 1.813275 s. SIMD shuffle plus canonical field-add
work costs more than the shared-memory tree on this kernel, while the smaller
threadgroup allocation does not expose useful occupancy. Restore the original
tree reduction and do not run T28.

## Claim-to-code map

| Claim | Current code seam | Intended change |
|---|---|---|
| Backends are selected without changing the protocol | `akita-pcs/src/scheme/mod.rs`, `akita-prover/src/compute/stack.rs` | Generalize the public batched-prove entry to heterogeneous cluster stacks |
| Packed trace remains the root source | `jolt-akita/src/trace_onehot.rs` | Add Metal implementations for its existing root views; no new logical representation |
| Setup and matrices are reused | `akita-metal/src/prepared.rs` | Extend prepared Metal state to proof kernels and persistent workspaces |
| D512 relation remains an exact qualified Metal route | `akita-metal/src/ring_switch.rs`, opening metrics | Retain the exact threadgroup-local six-prime route; global and radix-four transform variants are closed |
| Root centered output becomes reusable resident state | root opening/ring-switch orchestration | Add a typed resident witness only after the relation mechanism passes |
| Accepted root Z can feed commitment before ring-switch assembly | `OpeningBatchKernel`, root fold grind, recursive commit prefix | Stream ordered position chunks into a nonce-local, block-aligned inner-commit accumulator; retain only the accepted nonce |
| Unsupported qualified work fails closed | Jolt's runtime backend selector and Akita Metal policy | Required route plus per-operation route metrics; no silent CPU fallback |
| Proof and verifier stay unchanged | `akita-pcs` prover orchestration and verifier | Backend-only arithmetic and product-domain scheduling; byte parity in the fixed evaluator |

## Ambiguity register

| Question | Resolution rule |
|---|---|
| Whether one uniform Metal facade or four heterogeneous types yields simpler bounds | Start with heterogeneous public routing; use one facade only if it removes monomorphization or lifetime friction without hiding fallbacks |
| Whether root operations need a fused trait | Add one only if measured intermediate traffic makes the 4.15-second budget impossible through existing operation traits |
| Exact CPU-tail cutoff | Derive from one warm treatment using table-size counters; keep it fixed and public-shape-derived afterward |
| Whether the canonical schedule can clear 5x | Preserve it until a measured uncovered floor exceeds the target; isolate any later schedule change from kernels |
| Whether CPU A-relation overlap is still a useful route | Resolved negatively for this campaign: both serial and windowed treatments lost; preserve the Metal relation and capacity-parallel exactness |
## Resident sumcheck transcript checkpoint

### Question and bound

Can the existing direct F128 Stage-1 and Stage-2 proofs execute all dependent
sumcheck rounds in one Metal command buffer while preserving the exact proof
and verifier?  At T28 the retained implementation spends 1.725 s of wall time
in these stages (0.571 s Stage 1 and 1.154 s Stage 2), but only about 0.952 s in
GPU timestamp intervals.  Eliminating only command submission cannot close the
0.723 s complete-proof gap to 5x; a useful resident path must also remove the
per-round host transcript, equality/source reconstruction, uploads, and binds.

The first falsifiable prerequisite is exact transcript continuation.  Akita's
Blake2b sponge state at a sumcheck boundary must be exportable in bounded space,
and an independent replay must produce every subsequent 32-byte challenge
exactly.  This is execution metadata only: transcript messages, proof bytes,
challenge reduction, and verifier behavior remain unchanged.

### Candidate

Replace the opaque default Blake2b bridge with a byte-identical local bridge
whose state has an explicit checkpoint.  Validate it against Spongefish's
current Blake2b bridge over transitions through start, streaming absorb,
ratchet, partial squeeze, and resumed absorb, as well as Akita's existing golden
challenge.  Expose the checkpoint through an optional generic transcript hook;
non-Blake2b and custom transcripts return no checkpoint and retain the current
backend path.

### Credibility gate

Do not write an all-round Metal sumcheck kernel unless all of the following hold:

- the differential sponge test is byte-exact for every squeeze;
- the existing Akita transcript golden challenge is unchanged;
- a checkpoint taken after a partial squeeze resumes byte-exactly across an
  absorb and at least two further challenges;
- direct Stage-1 and Stage-2 proof parity tests remain exact after the substrate
  change.

Reject the route if exact continuation requires changing a transcript message
or verifier schedule.  Passing this gate authorizes only a Stage-1 resident
prototype at T25; T28 remains gated on exact proof parity and a measured
Stage-1 wall-time saving of at least 60 ms at T25 or a complete-proof saving of
at least 40 ms.

The checkpoint substrate passed differential Spongefish transitions, the
existing golden challenge, partial-squeeze continuation, and all three direct
proof parity tests. A one-command Stage-1 prototype was therefore measured at
T25. It preserved the proof digest and every verifier/transcript guard, but
Stage 1 rose from 247.297 ms to 291.058 ms and complete opening rose from
1.781748 s to 1.790656 s. Direct-range command wall rose from 192.584 ms to
237.212 ms and GPU time from 159.849 ms to 217.574 ms. Batching removed only
about 13 ms of command-submission gap while the serial one-thread Blake2b work
added about 58 ms of GPU time; post-command host replay left the roughly 54 ms
non-command portion unchanged. Reject and remove the resident Stage-1 route;
do not run it at T28. Retain only the byte-exact checkpoint/challenge
prerequisite while evaluating Stage 2, whose larger challenge-dependent host
state transition can plausibly amortize it.

### Resident Stage-2 suffix after the bivariate prefix

The retained T25 Stage 2 takes 478.578 ms, of which 280.738 ms is Metal command
wall and 213.195 ms is GPU time. At T28 it takes 1.153783 s, while the direct
relation command accounts for 722.567 ms wall and 561.119 ms GPU. The T28
stage therefore contains about 593 ms outside GPU timestamp intervals. That is
the useful ceiling for a resident design; command batching alone can recover
only the 161 ms command-wall/GPU difference.

Keep the canonical two-round bivariate prefix. After its second challenge the
host and device are at the same ordinary-round boundary: the transcript has an
exact checkpoint, the device retains the compact witness and structured-linear
tables, and the next round polynomial is already in the resident output
buffers. Execute rounds 2 through the final round in one command buffer. Each
round must:

- absorb the exact trimmed compressed polynomial and derive the ordinary
  Blake2b challenge;
- update the split-equality scalar while selecting a precomputed,
  challenge-independent equality-table layer;
- fold the coefficient-alpha or lane-weight table and the structured-linear
  source;
- fold sparse additional weights using a fixed parent topology, retaining
  algebraic zeros rather than changing support according to a challenge; and
- fold the witness and produce the next ordinary and additional coefficients.

The proof format, transcript labels and bytes, round count, polynomial degree,
two-round prefix, verifier, and protocol configuration are unchanged. The host
replays returned messages and challenges after the command to construct the
canonical prover state and independently checks the final claim. Transcripts
without an executable Blake2b checkpoint retain the existing round-at-a-time
path.

The added device work is one serial Blake2b transition per remaining round plus
small geometric folds. The rejected Stage-1 prototype measured about 58 ms for
its complete challenge chain, so a T25 Stage-2 suffix is expected to add roughly
60--80 ms of GPU work while removing most later command waits, equality/alpha
uploads, and host round construction. It deliberately does not claim the full
593 ms T28 ceiling: host replay remains, and the first two rounds and source
construction remain separate.

Before T28, require both direct Stage-2 CPU-parity tests, including nonzero
additional terms, and one exact T25 treatment. Preserve proof bytes, transcript,
evaluation, verifier, required routes, and memory guard. Promote only if Stage 2
is at most 390 ms (an 88 ms saving) and complete opening is at most 1.710 s (a
72 ms saving) against the 1.781748 s retained parent. A miss removes the
resident route without tuning Blake2b or changing the protocol. At T28 retain
only if complete opening improves by at least 180 ms against 6.035833 s; then
remeasure the uncovered floor before selecting the next mechanism toward the
5.313223 s target.

Both focused Stage-2 parity tests passed, including the nonzero compression and
restricted-binary addends. The exact T25 treatment also preserved the proof
digest, transcript, evaluation, verifier, routes, and memory guard, but missed
both performance gates. Stage 2 rose from 478.578 ms to 499.275 ms and complete
opening rose from 1.781748 s to 1.799908 s. Direct-relation GPU time increased
27.080 ms, while reducing the command-wall/GPU gap recovered only about 9 ms;
host prepare, bind, round construction, buffer setup, and session setup improved
by roughly 11 ms in aggregate. Applying the measured fixed GPU cost and the
larger T28 command gap projects less than 100 ms of T28 benefit, below the
180 ms promotion gate and far below the 723 ms complete-proof deficit. Remove
the resident Stage-2 route and do not run it at T28. Exact transcript execution
remains a validated primitive, but neither sumcheck has enough avoidable host
work to amortize serial device-side Blake2b on this GPU.

## Commit-retained destination index for root coefficient packing

The current exact T28 record is 6.035833292 seconds against a frozen CPU time
of 26.566116792 seconds. Five times therefore requires at most 5.313223358
seconds, a 0.722609934-second reduction. Root coefficient packing is a serial
1.200439958-second host phase. Prior local treatments do not have that ceiling:
four-block weight reuse regressed, fixed-shape raw limbs saved only 0.118
seconds, and a 15/15 CPU/Metal column split saved 0.329 seconds locally but
almost none end to end because it contended with the following root kernel.

Build a second, one-shot opening index while Metal commitment already owns the
packed D512/K256 source. For each `(trace block, column, row parity, 256-position
tile)`, partition the 256 selectors into the 32 `hot >> 3` buckets. Store a
`u16` record containing the eight-bit position within the tile and the three
low selector bits, plus 33 `u16` bucket offsets. T28 contains 8,053,317,376
selectors, so records require 16.107 GB and offsets about 2.08 GB. This index
is independent of every Fiat--Shamir value and changes neither commitment nor
protocol. It is consumed and released by coefficient packing before the
position-major root-fold index is consumed.

At opening time, assign one 256-thread group to each output
`(column, trace block, parity bucket)`. A thread owns one or four position tiles,
reads only the prepartitioned records for its bucket, and gathers the exact
precomputed `position_weight * packing_weight[low]`. Instead of a canonical
fp128 addition per selector, it accumulates the four 32-bit limbs independently
in four 64-bit counters. At most `2^18` records contribute to an output, so
every counter is below `2^50`. One group reduction and one final carry/fold by
`2^128 = MODULUS_OFFSET (mod p)` produce the canonical field value. Output
ownership removes atomics; maximal selector skew changes load balance but not
the bound or arithmetic.

The conservative T28 opening traffic is 32.2 GB of record/offset reads plus
128.9 GB of logical weight reads. The 32 MB weight table is reused across
neighboring bucket groups, so 161.1 GB is an upper traffic model rather than a
cache-adjusted claim. Charging all of it to the measured 400 GB/s class device
bandwidth gives a 0.40-second floor. The candidate has enough isolated ceiling
only if root packing reaches at most 0.45 seconds; that saves at least 0.75
seconds and can cross the fixed 5x threshold without relying on another phase.

First require focused generic-versus-indexed parity at stride eight, including
all-one-bucket skew, padded columns, partial final rows, and exact canonical
limb reduction. One T25 credibility treatment must preserve commitment, proof
bytes, transcript, evaluation, verifier, route, and memory guards; use exactly
one indexed packing call; keep root packing at most 60 ms; and keep complete
opening at most 1.76 seconds. Only a pass authorizes T28. At T28 retain only if
root packing is at most 0.45 seconds, complete opening is at most 5.313223358
seconds, peak live memory remains below 90 GiB, and total Metal commitment
remains below 39.183 seconds (five times the frozen 195.915989-second CPU
commit). Any miss removes the destination index and gather kernel without
tuning tile size or weakening an exactness gate.

The index and gather passed the focused exactness suite and the T25 gate. T25
root packing fell from 100.175 ms to 54.522 ms and complete opening fell from
1.781748 s to 1.696212 s. The exact T28 treatment did not preserve that
scaling: root packing took 866.719 ms and complete opening took 5.839570 s, or
4.549x against the frozen CPU anchor. The proof, transcript, evaluation,
commitment, verifier, route, 26.87 GB peak RSS, and 37.723-second Metal commit
all passed. Retain the challenge-independent index as a substrate, but reject
the output-major gather.

The miss is a cache-boundary error in the original traffic model. The combined
weight table is 8 MiB at T25 and 32 MiB at T28. T25 moves about 18.6 GB of
records, offsets, and logical weights in 54.5 ms, approximately 342 GB/s. T28
moves about 139 GB in 866.7 ms, approximately 161 GB/s. The output-major kernel
visits one bucket across the entire position domain, so each group gathers
sparsely and effectively randomly from the full weight table. Treating those
loads as streaming device bandwidth was therefore invalid once the table
crossed the effective cache size.

## Chunk-local all-bucket root packing

Keep the exact retained index, but transpose only its opening traversal. One
threadgroup handles all 32 buckets for one `(trace block, column, parity)`
stream over a fixed 32-tile window. Eight threads own each bucket. They read its
prepartitioned records, gather weights only from the window, accumulate limbs
in `u64`, and reduce within the eight-thread bucket group. Thirty-two tiles
cover 8,192 positions, so the combined-weight working set is exactly 1 MiB.
Dispatch windows outermost so consecutive groups reuse that window across
streams. Do not change the retained record layout or protocol.

At T28 this produces 32 partials for each of 983,040 live outputs over 32
windows: 503.3 MB of partial storage and about 1.0 GB of added write/read
traffic. Record and offset traffic remains one pass and every logical weight is
still loaded once, but the full-table random working set is removed. The first
pass uses three threadgroup barriers rather than the output-major tree's eight;
a second flat kernel adds the 32 canonical partials per output. A twofold
recovery from the observed 161 GB/s would put packing near 0.43 seconds and
complete opening near the fixed 5.313223-second threshold.

Use 32 tiles because its 1 MiB working set is safely below both measured table
regimes; do not sweep it. Require the existing varied, zero, padding,
multi-tile, partial-tile, maximal-skew, and one-shot cache tests. At T25 require
root packing at most 75 ms and complete opening at most 1.76 seconds. Only then
run T28. Retain at max scale only if root packing is at most 0.43 seconds,
complete opening is at most 5.313223358 seconds, peak RSS remains below 90 GiB,
and Metal commit remains below 39.183 seconds.

The chunk-local implementation passed focused exactness and T25: root packing
was 34.883 ms and complete opening was 1.755302 s. At T28 it reduced root
packing to 446.700 ms, but missed the 430 ms local gate; complete opening was
5.573686 s (4.766x), and the otherwise unchanged commit measured 39.620
seconds. Exactness and the 27.26 GB RSS guard passed. Do not tune the 32-tile
window. The locality premise is validated by the 420 ms improvement over the
output-major gather, but 983,040 short groups, three barriers per group, a
503.3 MB allocation, and the second pass are now exposed overhead.

## Monotone all-bucket stream traversal

Assign one 256-threadgroup to each complete `(trace block, column, parity)`
stream, with eight threads per bucket as in the chunk-local kernel. Traverse
all position tiles monotonically, keep the 32 bucket sums live in registers and
threadgroup state, reduce once, and write final coefficients directly. Dispatch
trace block outermost so the 60 live streams for a block advance through the
same weight region together. This preserves cache-line locality without a
window buffer and reduces the group count from 983,040 to 32,768, barriers from
about 2.95 million to 98,304, and intermediate traffic from 1.0 GB to zero.
The maximum-skew sum remains below `2^50` at the T28 schedule, so deferred limb
arithmetic is unchanged.

This mechanism can recover at most the 446.7 ms packing phase and therefore is
not by itself an analytical guarantee of 5x; it must expose a packing floor
before choosing the final non-packing target. Require the same exactness suite,
then one T25 treatment with root packing at most 55 ms and complete opening at
most 1.76 seconds. At T28 retain only if packing is at most 0.32 seconds and
complete opening is at most 5.45 seconds. The hard 5.313223-second proof and
39.183-second commit gates remain unchanged for final acceptance.

The monotone traversal passed exactness and its ambiguity-band T25 repeat at
34.413 ms packing and 1.752686 s complete. It failed at T28: groups drifted
across the full 32 MiB weight table, packing regressed to 615.251 ms, and
complete opening regressed to 5.825263 s. Reject it and restore bounded windows.

## Four-stream SIMD-pair window packing

The bounded-window kernel still launches 983,040 groups, assigns eight threads
to each bucket, crosses three barriers per group, and writes 503 MB of partials.
Instead, one group handles four streams in the same 32-tile window. Two SIMD
lanes own each bucket; each 32-lane SIMDgroup therefore covers sixteen adjacent
buckets of one stream. This matches the observed Jolt half-range selector
distribution without forcing inactive and active buckets through the same
divergent loop. Normalize the two deferred limb sums, exchange them with a SIMD
shuffle, and let the even lane write the exact bucket partial. No threadgroup
memory or barrier remains.

At T28 the group count falls fourfold to 262,144. Logical record and weight
traffic is unchanged, the live weight window remains 1 MiB, and padded columns
add only zero partials. Partial storage is 536.9 MB because it now uses the
canonical output layout. Require the existing exactness suite, then one T25
treatment with packing at most 35 ms and complete opening at most 1.76 seconds.
At T28 retain only if packing is at most 0.30 seconds and complete opening is at
most 5.40 seconds. This is the last packing traversal in this campaign; a miss
restores the 32-tile eight-thread winner and moves to a non-packing mechanism.

The four-stream kernel passed exactness and T25 at 30.110 ms packing and
1.744464 s complete opening. It failed both T28 gates: packing was 406.503 ms
and complete opening was 5.567019 s. The 40.196 ms local packing improvement
over the chunk-local parent produced only a 6.667 ms complete-opening gain,
while the unrelated root consumer grew by 45.365 ms. Reject the kernel and
restore the 32-tile eight-thread implementation. The result also closes further
packing-layout tuning as the path to 5x: even removing the remaining 406.503 ms
entirely would leave too little robust margin once cross-phase contention is
included.

## Wider opening-basis audit

Changing the root opening basis from log basis 3 to log basis 4 would reduce
the balanced root digit count from four to three, but it is not a small
admissible schedule change. The Metal direct Stage 1 implementation supports
log bases 2 and 3; log basis 4 selects the tree protocol. More importantly, an
exhaustive schedule-generator query found no T28 row with log basis 4 under the
current audited SIS policy: not at the existing D512/B64/D64 geometry, not at
any admitted rank, not with B or D widened to 128, and not across any admitted
dimension or block geometry. The original generated schedule was restored
byte-for-byte. Reopening this path would require changing the challenge or
security policy as well as implementing a new Metal protocol path, so it is
outside the approved minor-change scope.

## Disjoint-resource root pipeline

The retained T28 root decompose/recursive-commit pipeline takes 1.604709 s.
Inside that wall interval, eight Metal chunks account for 755.303 ms of GPU
span and their ordered CPU consumers account for 918.453 ms. With average
chunk service times of about 94.4 ms and 114.8 ms, an unconstrained two-stage
pipeline has the lower-bound schedule

```text
G + 8 C = 94.4 ms + 8 * 114.8 ms = 1.013 s,
```

which exposes roughly 592 ms of analytical headroom. Only 260.463 ms is needed
to move the frozen 5.573686-second parent below the 5.313223-second 5x limit.
The present implementation commits all eight commands before consuming them,
but every command writes a disjoint offset of the same two Metal buffer
resources. CPU reads from those resources while later commands remain GPU
writers. The observed wall time is only 69.048 ms below GPU-plus-consumer time,
consistent with resource/coherence serialization rather than the intended
chunk pipeline.

Give every chunk its own shared output and digit buffers. Commands continue to
share the immutable index and challenge buffer, are committed in order, and are
consumed in order. Copy each completed centered-output chunk into the existing
global result vector; consume its digit buffer directly. This changes no
arithmetic, proof data, transcript, protocol, chunk count, or total live byte
count. Distinct writable Metal resources are the only experimental variable.

T25 is a correctness and regression gate, not a speedup oracle: its 97.210 ms
GPU span is already almost fully hidden behind its 255.256 ms consumer, and its
273.639 ms wall is near the 267 ms two-stage floor. Require exact commitment,
proof bytes, transcript, evaluation, verifier, route, and one indexed dispatch;
require complete opening at most 1.80 seconds and packed-decompose wall at most
0.30 seconds. Then run one exact T28 treatment. Retain only if complete opening
is at most 5.313223358 seconds, packed-decompose wall improves by at least
0.30 seconds to at most 1.304709 seconds, peak RSS stays below 90 GiB, and all
exactness and routing gates pass. A miss closes buffer identity as the cause
without changing chunk count or adding another sweep.

The focused indexed fold/digit oracle and both restored packing oracles passed.
T25 also passed its regression gate at 1.748185 s complete and 280.570 ms packed
decompose wall, with exact commitment, proof, transcript, evaluation, verifier,
and route behavior. The single T28 treatment falsified the mechanism. It was
exact, but packed-decompose wall regressed to 1.632787 s and complete opening to
5.637833 s. GPU span was 797.325 ms and consumer time was 840.713 ms; their sum
is only 5.251 ms above the measured wall. Distinct output and digit buffer
identities therefore did not create material T28 overlap. Restore the shared
buffers and do not sweep chunk count. The remaining target must reduce one of
the two services rather than trying to overlap them through Metal resource
identity.

## Q128 CRT-prime audit

The proposed CPU recursive-commit reduction from six CRT primes to five is
already the retained implementation. `Q128_NUM_PRIMES` is five, the selected
capacity profile is `Q128/5xi32`, and five roughly 30-bit primes are the exact
Q128 reconstruction set. The six-prime constants elsewhere in the Metal code
belong to D512 relation and recursive kernels with different accumulation
bounds; they are not the streamed D128 CPU commitment denominator. Reducing the
D128 path further would no longer reconstruct every Q128 value. Close this
candidate without code or a benchmark.

## Stage-2 static-session overlap

The retained T28 opening is 5.573686375 s, 260.463017 ms above the fixed
5.313223358-second five-times threshold. Its direct Stage-2 call is 1.218878250
s. The following measured intervals are independent of the Stage-1 challenges
and transcript messages:

| Work | T28 wall |
|---|---:|
| opening-term preparation | 25.359084 ms |
| direct layout and field-to-limb preparation | 121.535209 ms |
| resident Metal session construction | 219.110750 ms |
| **overlap ceiling** | **365.850084 ms** |

The 0.154959 ms equality-prefix preparation inside the host interval is
challenge-dependent; subtracting it does not change the displayed precision.
The static inputs already exist after ring-switch finalization: the compact
witness, relation-weight factorization, opening semantics, setup-linear terms,
and relation geometry. Stage 1 reads the compact witness, tau0, geometry, and
basis but does not consume the relation factorization or structured-linear
terms. Stage 2 alone needs the equality prefix derived from the Stage-1 point,
the batching challenges sampled afterward, and sparse additional-relation
terms derived from those values.

Add a backend preparation token to the direct relation-proof operation. For an
accelerator backend, build the structured-linear terms and the complete static
Metal session on one scoped host worker while the caller proves Stage 1. The
ordinary CPU backend returns an empty token and keeps the current sequential
path. The Metal token owns its buffers, converted layout, and setup timings;
the later proof call supplies only the equality-prefix and transcript-dependent
round data. The worker is joined before Stage 2, so errors cannot escape the
proof call and no detached work survives it. This changes neither a sumcheck
message nor its order.

The static session writes the same buffers and runs the same 55.241834 ms GPU
linear-source kernel. Its compulsory device work cannot disappear. Charging
that entire GPU interval as serialized Stage-1 interference gives the
conservative critical-path estimate

```text
5.573686375 - 0.365850084 + 0.055241834 = 5.263078125 s
26.566116792 / 5.263078125 = 5.048x.
```

This estimate gives only 50.145 ms of margin. The independently exact
live-prefix Stage-1 factorization reduced T28 Stage 1 by 74.5 ms; applying it
after this candidate would move the estimate to 5.188578125 s, or 5.120x.
It remains a separate candidate and is not used to excuse a static-session
regression.

The scheduling change adds no total buffer traffic. It overlaps the Stage-2
session with the approximately 5.5 GiB Stage-1 resident session, raising peak
residency but remaining far below the 90 GiB limit from the current 27.26 GiB
peak. The source kernel's observed 55.24 ms is a stronger floor than pricing
its traffic at the measured 406.4 GB/s copy rate. Host allocation and copy work
may contend with Stage 1; that contention is the principal falsifier.

Implement only this static/dynamic split. First require focused direct Stage-2
CPU/Metal proof parity and a T25 exact regression run no slower than 1.85 s.
Then run one T28 treatment. The mechanism is locally retained only if Stage-2
sumcheck falls to at most 0.94 s, the post-Stage-1 join wait is at most 25 ms,
all proof, transcript, evaluation, commitment, verifier, route, and 90 GiB
guards pass, and complete opening is at most 5.36 s. Final success still
requires at most 5.313223358 s. If the local gates pass but the hard gate misses,
restore the exact Stage-1 factorization and run one combined T28 treatment. If
either local gate fails, remove the preparation token and do not tune worker or
queue counts.

Both focused direct-proof parity cases passed. The T25 treatment was exact and
improved complete opening from 1.755302 s to 1.598440 s, below its 1.85-second
regression limit. The first T28 treatment also passed every proof, transcript,
evaluation, commitment, verifier, route, and memory guard:

| Metric | Retained parent | Static session |
|---|---:|---:|
| complete opening | 5.573686 s | **5.032946 s** |
| speedup over 26.566117 s CPU | 4.766x | **5.278x** |
| Stage 1 | 0.582471 s | 0.643429 s |
| Stage 2 sumcheck body | 1.218878 s | 0.744129 s |
| root coefficient packing | 0.446700 s | 0.451829 s |
| root streamed decompose/fold | 1.629001 s | 1.554023 s |
| aggregate GPU-active time | 2.003677 s | 2.015088 s |
| peak RSS | 27.256 GB | 27.115 GB |

The unchanged source kernel and buffer work increased Stage 1 by 60.958 ms
through queue and memory contention, as the conservative model allowed. The
Stage-2 body nevertheless fell by 474.749 ms, and the complete call improved by
540.740 ms. Root streaming happened to improve by 74.977 ms in this run, but
removing that entire favorable movement still leaves 5.107923 s, safely below
the 5.313223-second threshold. The candidate therefore does not rely on the
root-phase fluctuation to clear five times. The exact proof digest remains
`b9139d872029400fe920feebea66e643eac957f8f4f8b445efee9f6c203f6dea`.
Do not restore the Stage-1 factorization; it is unnecessary complexity now that
the single architectural change clears the target with margin. Revalidate the
cleaned head once at T28 before promotion.

The cleaned-head T28 validation measured 5.175134666 s, or 5.133x against the
same 26.566116792-second CPU anchor. This is 138.089 ms below the exact
five-times limit. The two candidate observations are therefore 5.278x and
5.133x; the worse observation clears the target. The final run again matched
the proof digest above, CPU proof, transcript, claimed evaluation, commitment,
and verifier, with 27.104 GB peak RSS and no fallback on qualified operations.
Its Metal commitment was 37.991499208 s versus the frozen 195.915989208-second
CPU commitment, or 5.157x, so the concurrent preparation did not sacrifice the
T28 commitment target.

This accepts the T28-only eval-proof claim at the revalidated evidence stage.
The T25 regression sentinel remains exact and improved to 1.598440458 s, but it
is 3.963x against its 6.335011958-second CPU anchor and is not part of the
re-scoped five-times claim.
