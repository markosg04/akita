# Spec: Exact five-prime RNS packed root

| Field         | Value |
|---------------|-------|
| Author(s)     | Akita Metal performance work |
| Created       | 2026-08-24 |
| Status        | historical |
| PR            | |
| Supersedes    | |
| Superseded-by | |
| Book-chapter  | |

## Summary

The packed fp128 D512 root spends about 7.77 of its 12.51 GPU seconds in
carry-dependent fp128 coefficient addition. Represent the public matrix and root
partials modulo five existing 30-bit auxiliary primes, accumulate five independent
residues, and reconstruct the exact fp128 output once after the 16 position partials
are reduced. This keeps the current 40-u32 per-thread accumulator state while
replacing four serial carry stages with independent modular additions. It changes
only prover-backend arithmetic; the canonical commitment and proof are unchanged.

## Intent

### Goal

Implement one exact five-prime RNS execution path for the qualified K256/D512/rank-one
packed root in `akita-metal`, with reusable public matrix preparation and canonical
fp128 output at the existing `RootCommitKernel` boundary.

### Invariants

- The root output is bit-for-bit equal to the CPU packed commitment for every valid
  lane value, sparse or dense activity, positive and negacyclic contributions,
  partial final streams, and all registered 32--512 block geometries.
- Matrix coefficients use their centered lift before residue conversion. After all
  16 position partials are summed, centered Garner reconstruction recovers the exact
  signed integer sum and then maps it into `Prime128OffsetA7F7`.
- Capacity is checked from the same prime list used by the device resources. Four
  primes are insufficient; five must satisfy the strict whole-block bound.
- Public matrix conversion is setup-bound and reusable. It is excluded from the
  proof timer only through the existing explicit prewarm boundary.
- The lane source remains zero-copy and cycle-major. Task ownership, the hybrid CPU
  tail, output order, commitment, opening hints, transcript, proof bytes, verifier,
  and soundness assumptions do not change.
- A qualified RNS path fails closed on missing resources, capacity failure, invalid
  geometry, command failure, or reconstruction failure. It may not silently use the
  old Metal kernel or CPU beyond the planned hybrid tail.

### Non-Goals

- This does not change the Akita protocol, setup serialization, Jolt geometry, or
  verifier.
- This does not use an NTT. A monomial shift is still a residue-array rotation and
  sign change; adding pointwise NTT multiplication would increase dominant work.
- This does not sweep prime counts, tile sizes, hybrid splits, or command batches.
- This does not retain both full fp128 and RNS root matrix caches. Later smaller
  fp128 prefixes may be prepared lazily, but the 2-GiB fp128 root cache must not be
  created merely to support this path.

## Evaluation

### Acceptance Criteria

- [ ] A red route test distinguishes the RNS root before implementation and passes
      only when the new device path is selected.
- [ ] An exact capacity test proves five primes support 524,288 centered terms and
      the first four do not.
- [ ] Focused CPU/Metal parity covers both signs, shifts 0/1/255/256/511, 31/32/33
      task boundaries, a final residue/tile tail, resident and streaming owners, and
      the 512-block geometry.
- [ ] One verified BTreeMap T25 sentinel reports zero-copy input, the planned CPU
      tail only, no swaps, and root GPU time at most 1.25 seconds. A clear miss ends
      the candidate without T28 or parameter tuning.
- [ ] If T25 passes, one verified BTreeMap T28 treatment reports five residues,
      exactly 896 matrix streams, 2,405,181,685,760 modeled matrix-read bytes, root
      GPU time at most 10.8 seconds, complete traced proving at most 46.3 seconds,
      peak RSS at most 90 GiB, and no swap growth.

### Testing Strategy

Watch the route/capacity test fail before adding the path. Use independent CPU packed
commitment output as the arithmetic oracle; do not retain a copy of the superseded
Metal kernel in tests. Extend the existing packed D512 parity cases rather than add a
second fixture framework. Run the focused `cargo nextest` suite, formatting, diff
checks, and the Akita documentation guardrail before either runtime sentinel. The
integrated Jolt proof must verify before any timing is valid.

### Performance

The accepted BTreeMap T28 root performs about 1.019 trillion fp128 coefficient
additions and measures 12.510 seconds. Its 1,810.10 GiB modeled traffic has a
4.388-second floor at the measured 412.5 GiB/s; activity calibration attributes
about 7.774 seconds to the current carried-add mapping.

Five residues expand matrix and partial storage by 25%. The RNS root has:

```text
matrix bytes                   2,684,354,560
coefficient-band streams                 896
matrix-read bytes           2,405,181,685,760
lane-read bytes                15,005,122,560
partial write + read bytes      5,368,709,120
canonical output bytes            134,217,728
total modeled traffic        2,425,689,735,168  (2,259.10 GiB)
traffic floor                            5.477 s
```

The state remains five `uint4` vectors per accumulator. A positive current update
executes four serial carry-producing word updates plus a wrap update. An RNS update
executes five independent add/conditional-subtract updates; negative updates use
independent subtract/conditional-add. The expected arithmetic-term reduction is
25--45%, not the five-to-four residue ratio alone, because matrix gathers, ballots,
barriers, and stores remain. Three positions occupy 30 KiB of threadgroup memory,
so the position loop grows from 4,096 to 5,462 tiles and must handle one final
position. The prediction, including this barrier increase and final Garner work, is
10.0--10.8 seconds at T28. The 10.8-second root and 46.3-second proof limits are the
pre-registered falsifiers, not aspirational post-hoc thresholds.

The RNS root cache is 0.5 GiB larger than the current fp128 root cache. Preparing it
must replace, rather than accompany, that cache; otherwise the memory model is
invalid. Conversion time is charged to reusable public preprocessing and reported,
but is not part of `jolt_prover::prove` under the existing evaluator contract.

## Design

### Architecture

For output block `b`, live column `c`, coefficient `k`, and trace row `r`, let
`a(r,k)` be the selected coefficient of the public matrix row after the existing
negacyclic rotation, and let `epsilon(r,k)` be its `+1` or `-1` sign. The canonical
root computes

```text
S[b,c,k] = sum_r epsilon(r,k) * center(a(r,k))  (mod p).
```

There are at most 524,288 contributing rows per block. For the first five existing
auxiliary primes

```text
q = [1073692673, 1073668097, 1073707009, 1073738753, 1073732609]
Q = product(q)
  = 1427021764075536559521940416710679405150431233.
```

With `p = 2^128 - 2^32 + 22537`, centered inputs give the strict exactness
requirement

```text
524288 * p < Q.
```

The ratio `Q / (524288 * p)` is 7.9987. The four-prime product has only 120
bits and fails. Thus the residues uniquely determine every whole-block signed sum,
not merely each 1/16 partial. This is an implementation representation, not a new
commitment assumption.

The prepared setup owns one RNS root matrix buffer keyed by the existing D512/rank
one/active-column geometry. It reuses the first five prime, limb-weight, field-modulus,
Garner, and field-partial-product constants already built for exact D512 linear
relations. Matrix conversion centers fp128 coefficients and emits Montgomery residues;
addition and subtraction preserve that representation.

The root kernel keeps one `(column, block)` task per SIMDgroup and the current two
coefficient bands and 16 position partials. A 1,024-thread group loads three matrix
positions as five residue planes (30 KiB), ballots six trace rows, and updates two
five-vector accumulators. It writes five residue planes per partial. The reduction
kernel sums all 16 partials modulo each prime, exits Montgomery form, performs
five-prime centered Garner reconstruction, and writes the existing `AkitaFp128`
output. Host reconstruction and hybrid-tail merge then remain unchanged.

### Alternatives Considered

- Five canonical 32-bit limbs without residues need roughly 45-bit per-digit sums or
  periodic carry propagation; prior same-state carry-save C6 regressed 8.1%.
- Four residues plus a small tail cannot meet the 147-bit centered whole-block bound;
  the missing residue needs about 27 more bits.
- Six existing residues raise traffic and state by 50% when five already have an
  eightfold exactness margin.
- NTT-domain accumulation replaces shifts with six modular multiplications per
  coefficient and increases the dominant arithmetic.
- Two-task matrix reuse halves traffic but doubled live accumulator state and
  regressed the T25 root from 1.473 to 1.729 seconds.
- Five-command batching improved T25 fixed cost but saved only 0.228 seconds at T28,
  showing that dispatch occupancy is not the saturated-scale limiter.

## Result

The exact implementation passed the capacity ratchet, all four focused CPU/Metal
parity cases, the full 40-test `akita-metal` suite, and an integrated verified Jolt
proof. The reusable RNS prewarm populated no fp128 root cache. At the fixed T25
sentinel it reported a 1.580892-second root GPU interval, versus 1.473321 seconds for
the accepted one-stream fp128 root. Matrix bytes, modeled matrix reads, and scratch
grew by exactly 25%; input remained zero-copy, the planned eight-block CPU tail was
unchanged, peak RSS was 16.91 GiB, and neither process nor system swap grew.

The 1.580892-second result misses the pre-registered 1.25-second gate and regresses
the accepted root by 7.3%. The extra residue traffic, additional tile barriers, and
Garner reconstruction outweighed the reduction in carry dependencies at this scale.
The candidate was rejected without a repeat, T28 run, or parameter sweep, and its
runtime source was restored. Reopen only with a representation that avoids the 25%
traffic expansion or with independent evidence that materially changes the fast
gate.

## Documentation

This historical design record and its append-only Jolt campaign ledger are sufficient
during experimentation because the proof protocol is unchanged. If retained, fold
the reusable backend representation and capacity rationale into
`book/src/how/optimizations.md`; otherwise mark this record historical with its
negative result.

## Execution

1. Register and observe the route/capacity test failure.
2. Add one RNS matrix cache and conversion path using the existing exact CRT resource
   constants; do not prepare the fp128 root matrix on this route.
3. Add the three-position residue root and fused partial-reduction/reconstruction
   kernels, retaining the existing task and hybrid ownership.
4. Pass focused exact parity and repository formatting/documentation guards.
5. Rebuild the fixed Jolt evaluator, run one T25 sentinel, and promote once to T28
   only if the registered gate passes.
6. Keep or exactly restore runtime sources and record the model update.

## References

- `specs/metal-packed-root-matrix-reuse.md`
- `specs/large-digit-ntt-infrastructure.md`
- `crates/akita-algebra/src/ntt/crt.rs`
- `crates/akita-metal/src/kernels/onehot.metal`
- Jolt `benchmark-runs/akita-metal-e2e-polish/events.jsonl`
