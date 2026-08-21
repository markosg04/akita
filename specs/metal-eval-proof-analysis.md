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

## Claim-to-code map

| Claim | Current code seam | Intended change |
|---|---|---|
| Backends are selected without changing the protocol | `akita-pcs/src/scheme/mod.rs`, `akita-prover/src/compute/stack.rs` | Generalize the public batched-prove entry to heterogeneous cluster stacks |
| Packed trace remains the root source | `jolt-akita/src/trace_onehot.rs` | Add Metal implementations for its existing root views; no new logical representation |
| Setup and matrices are reused | `akita-metal/src/prepared.rs` | Extend prepared Metal state to proof kernels and persistent workspaces |
| D512 relation remains an exact qualified Metal route | `akita-metal/src/ring_switch.rs`, opening metrics | Retain the exact threadgroup-local six-prime route; global and radix-four transform variants are closed |
| Root centered output becomes reusable resident state | root opening/ring-switch orchestration | Add a typed resident witness only after the relation mechanism passes |
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
