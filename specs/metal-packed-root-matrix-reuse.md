# Packed fp128 D512 root matrix reuse

## Decision

Test one change to the packed fp128 D512 root kernel: while a public matrix tile is
resident, give each SIMDgroup two `(column, block)` tasks instead of one. This halves
the number of complete matrix streams at the integrated BTreeMap T28 geometry.

This is a prover-backend scheduling change. It does not change the packed source,
canonical commitment, opening hint, proof schedule, transcript, proof bytes, verifier,
or soundness assumptions.

## Exact boundary

The input is the existing cycle-major packed byte owner and the cached D512 setup
matrix. The output remains the same 16 position partials for every output coefficient,
followed by the existing reduction kernel and host reconstruction. The following are
unchanged:

- K256, D512, rank one, 32-column capacity, and the hybrid CPU tail;
- two coefficient bands and 16 disjoint position partials;
- lane row order, absent-entry encoding, negacyclic rotation, and fp128 reduction;
- output and scratch allocation sizes, command completion, readback, and metrics;
- the CPU implementation and all verifier-facing data.

Only the mapping within `akita_packed_onehot_commit_fp128_d512_panels` changes. A
1,024-thread group still loads four matrix positions into exactly 32 KiB of
threadgroup memory. Its 32 SIMDgroups each retain two task accumulator pairs across
the tile loop and store both tasks at the end. A partial final stream masks tasks
beyond `dispatch_tasks`.

## Current cost and lower bounds

The accepted integrated BTreeMap T28 observation reports:

```text
Metal tasks                    14,310
coefficient bands                   2
tasks per stream                   32
matrix streams                      896
matrix bytes              2,147,483,648
modeled matrix reads    1,924,145,348,608  (1,792 GiB)
modeled lane reads         15,005,122,560  (13.975 GiB)
scratch capacity            2,147,483,648  (2 GiB)
root GPU time                     12.510 s
backend time                      12.611 s
overlapped CPU leg                 5.288 s
```

The root writes the 2 GiB partial plane once, the reducer reads it once, and writes
0.125 GiB of output. Counting those transfers gives 1,810.10 GiB of modeled device
traffic. At the measured 412.5 GiB/s read/write rate, its traffic floor is 4.388 s.

The arithmetic term is independent. Each selected entry adds one rotated D512 row.
A fixed-geometry T25 calibration used the unchanged Metal-only evaluator at 25% and
75% activity:

| Activity | Metal tasks | Matrix reads | Hot entries | GPU time |
|---|---:|---:|---:|---:|
| 25% | 750 | 192 GiB | 209,700,358 | 1.150650 s |
| 75% | 775 | 200 GiB | 629,149,348 | 2.783455 s |

Correcting hot entries for the two-block and one-block CPU tails gives 211.402
billion additional Metal coefficient additions. The extra matrix and lane traffic
has a 0.0195 s floor. The residual delta is 1.6133 s, or about 131.0 billion fp128
coefficient additions/s for the current transposed-accumulator mapping. BTreeMap's
477 of 485 populated Metal blocks therefore imply about 1.019 trillion GPU
coefficient additions and a calibrated 7.77 s compute term. This is a floor for the
unchanged arithmetic mapping, not a hardware-wide fp128 limit.

The candidate uses 64 tasks per stream:

```text
matrix streams       2 * ceil(14,310 / 64) = 448
matrix reads                       896 GiB
total modeled traffic          914.10 GiB
traffic floor                    2.216 s
compute term                     7.774 s
```

The arithmetic term should bind after matrix reuse. A useful implementation should
therefore land near 7.8--9.5 s GPU time, saving roughly 3.0--4.7 s from the complete
prover if the root remains on the critical path. The entire root can save at most
about 4.74 s under this arithmetic mapping.

## Register risk and alternatives

One transposed accumulator contains five four-lane vectors, or twenty 32-bit scalar
components. The accepted kernel keeps two accumulators per thread; this candidate
keeps four. The 32 KiB matrix tile already consumes the full threadgroup-memory
budget, so the group count cannot fall below one resident group per core due to this
change. Register spills or reduced instruction issue can still erase the traffic
gain. Metal does not expose a reliable register count here, so GPU time is the direct
falsifier.

Increasing the matrix tile is ineligible because the current tile already uses 32
KiB. Launching more groups changes occupancy but not matrix streams. Processing more
than two tasks per SIMDgroup is not part of this treatment; it may be priced only
from the two-task result rather than explored as a reuse-factor sweep.

## Test and promotion gate

Change an existing task-grid assertion to require 64 tasks per stream and observe it
fail before the implementation. Exact CPU/Metal parity must then cover task counts
31, 32, 33, 63, 64, and 65, both coefficient bands, a partial final stream, sparse
and dense rows, and resident and streaming packed owners.

After focused parity, run one warm verified T25 sentinel. Admit one integrated
BTreeMap T28 treatment only if it uses the new route and reports half the expected
matrix streams. At T28 require:

- `matrix_block_streams = 448` and modeled matrix reads 962,072,674,304 bytes;
- exact commitment/proof verification and no CPU fallback beyond the planned tail;
- root GPU time at most 10.5 s; above this value falsifies the useful-ceiling model;
- complete proving at least 0.5 s faster than the accepted 48.08 s parent, or at
  least 5% faster in the affected root span;
- peak RSS at most 90 GiB.

The 7.8--9.5 s range is the mechanism prediction, while 10.5 s is the rejection
boundary. Repeat once only for a surprising result, threshold ambiguity, or parent
promotion. On failure, revert the task mapping and record the register/issue result;
do not tune larger reuse factors under the same candidate.

## Result

The red route assertion first observed 32 tasks per stream. After the implementation,
exact CPU/Metal parity passed at 31, 32, 33, 63, 64, and 65 tasks, along with the
resident, streaming, sparse-rotation, and 512-block cases.

The single BTreeMap T25 sentinel verified and reported the intended route. Modeled
matrix reads fell exactly from 207,232,172,032 to 104,152,956,928 bytes. Root GPU
time nevertheless rose from 1.473321 s to 1.729072 s, command wall rose from
1.497966 s to 1.751841 s, and complete proving rose from 6.45 s to 6.71 s. Peak RSS
fell from 16.83 to 15.35 GiB and no swap or fallback occurred.

This is a device-work regression despite removing half the logical matrix traffic.
Four live accumulators per thread reduce issue throughput or spill enough to cost at
least 0.49 s relative to the ideal traffic saving at this scale. C1 is rejected
without a T28 run and reverted in `be8c706a9`. Do not try a larger task-reuse factor;
a future root candidate must reduce arithmetic with one task per SIMDgroup or use a
different representation that does not double live accumulator state.
