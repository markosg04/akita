# Batch packed D512 root streams per Metal command

## Decision

Test one scheduling change before another accumulator representation: encode five
existing packed fp128 D512 matrix streams in each command instead of one. The shader,
task mapping, arithmetic, buffers, hybrid split, output, commitment, opening hint,
transcript, proof, verifier, and soundness are unchanged.

The candidate is admitted because the current dispatch is structurally smaller than
the target GPU. It submits 32 threadgroups per command: 16 position partials times two
coefficient bands. Each group uses the full 32-KiB threadgroup-memory allocation, so
at most one such group can reside on a GPU core. The target M4 Max has 40 GPU cores.
Commands on the one queue cannot expose more than their encoded grid, leaving at least
eight cores without a group whenever command boundaries prevent cross-command
scheduling.

## Exact boundary

Only `FP128_D512_STREAMS_PER_COMMAND` and the runtime command partition change. One
stream remains 32 `(column, block)` tasks, and the kernel continues to derive its
local stream, coefficient band, position partial, and task from the unchanged
parameters. A regular five-stream command contains 160 threadgroups and at most 160
tasks. The final command masks its existing partial task and stream tails.

The command's zero-copy lane slice expands from roughly one or two trace blocks to at
most six contiguous blocks at the integrated 30-column geometry. No lane bytes are
copied, retained beyond command completion, or reordered. Matrix and partial buffers
are shared exactly as before. Total matrix streams, matrix reads, lane probes,
coefficient additions, partial writes, reduction work, scratch bytes, and readback are
identical. The public API and protocol have no new configuration.

## Lower bounds and prediction

The accepted BTreeMap T28 root has 14,310 Metal tasks and therefore 448 base streams,
896 coefficient-band matrix streams, and 14,336 root threadgroups. It measures
12.510 seconds on the GPU and 12.611 seconds through the backend. The existing
one-stream commands expose 32 groups each, or 448 separate 32-group grids.

Under the conservative one-group-per-core residency bound, a 40-core GPU needs one
group-duration wave for each current command: 448 waves. Five-stream commands change
the regular grid to 160 groups, exactly four 40-group waves. There are 89 regular
commands and one three-stream tail, whose 96 groups require three waves, for 359 waves
in total. This is a 19.9% reduction in the grid-wave ceiling. It also removes 358
command buffers and their encoder, queue-transition, and lane-buffer setup costs.

This is not credited as a guaranteed 19.9% latency win. A driver may overlap adjacent
command buffers, group duration varies with activity, and bandwidth can bind before
core count. The registered prediction is a T28 root of 10.0--11.8 seconds, saving
0.7--2.5 seconds in the affected span and at least 0.5 seconds in the complete proof.
The traffic and arithmetic floors remain the previously measured 4.388 and 7.774
seconds; batching moves neither bound.

Five is selected analytically rather than swept: `5 * 32 = 160` is divisible by 40.
On a 32-core device it exposes five full waves instead of five separate full grids; on
smaller devices, combining grids cannot increase the ideal wave count. Larger batches
receive no credit until this treatment establishes that command boundaries are real.

## Correctness and falsification gate

First change the existing task-grid test to require five streams in a regular command
and observe it fail. Exact CPU/Metal parity must still cover 31, 32, 33, 159, 160, and
161 tasks, both coefficient bands, a partial final stream, resident and streaming
owners, and the 512-block geometry. Add no alternate shader or protocol path.

After focused parity and formatting, build once and run one warm verified BTreeMap T25
sentinel. It must report unchanged task, matrix-stream, lane-read, scratch, and hybrid
counters; remain zero-copy; use no fallback or swaps; keep RSS below 90 GiB; and put
the root GPU span at or below 1.40 seconds. A clear miss rejects the candidate without
T28 or a batch-size sweep.

If T25 passes, run one BTreeMap T28 treatment. Require:

- exactly 90 root commands for 448 base streams and five streams per regular command;
- unchanged 896 coefficient-band matrix streams and 1,924,145,348,608 modeled matrix
  bytes;
- root GPU time at most 11.8 seconds;
- complete traced proving at most 46.89 seconds, at least 0.5 seconds below the
  accepted 47.388661-second trace;
- exact proof verification, no fallback, no swap growth, and peak RSS at most 90 GiB.

Repeat once only for a surprising result, threshold ambiguity, or parent promotion.
On failure, restore one stream per command and record whether the root span, command
wall, or zero-copy lane owner falsified the scheduling premise.

## Result

Rejected and restored to one stream per command. The red schedule ratchet failed
before the edit, and four focused resident/streaming CPU-parity tests passed after it.
At T25, five-stream batching reduced root GPU time from 1.473 to 1.209 seconds and the
verified complete proof measured 6.11 seconds with zero swaps. This confirms a fixed
command-boundary cost at the smaller geometry.

The promoted T28 treatment did not support the occupancy model. It used the expected
90 commands, 896 coefficient-band matrix streams, and 1,924,145,348,608 modeled
matrix bytes, but root GPU time was 12.282 seconds: only 0.228 seconds below the
accepted 12.510 seconds and above the 11.8-second gate. The verified traced proof was
49.275 seconds, peak RSS was 80.08 GiB, and system swap usage did not grow. Adjacent
command buffers therefore overlap enough, or arithmetic and matrix traffic saturate
enough, that the one-command/one-wave bound does not describe T28. Do not sweep batch
sizes for the T28 campaign. The T25 result may be reconsidered later as a public
geometry-based lower-scale crossover after the dominant T28 path changes.
