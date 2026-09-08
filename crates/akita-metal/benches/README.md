# Metal component benchmarks

Run one process at a time on an idle Mac. These are synthetic component
benchmarks, not full-proof throughput measurements. Retain cold and warm
results separately, use a process timeout, and leave 120 seconds between
processes when comparing implementations.

```bash
cargo bench -p akita-metal --bench packed_onehot_commit
cargo bench -p akita-metal --bench packed_onehot_fold
cargo bench -p akita-metal --bench ring_switch_rows
```

## Packed commitment

`packed_onehot_commit` measures the production packed commit path.
`AKITA_METAL_RING_D=128` selects rank 3. Shape controls are
`AKITA_METAL_LOG_ROWS`, `AKITA_METAL_COLUMNS`,
`AKITA_METAL_POSITIONS_PER_BLOCK`, `AKITA_METAL_DENSITY_PERCENT`, and
`AKITA_METAL_SAMPLES`.

Repeated-output equality is a consistency check. Independent CPU parity is
covered by the kernel tests, not established by repeated GPU execution.

## Packed opening fold

`packed_onehot_fold` measures the D128 fold using the canonical subring64
challenge sampler and embedding. The same shape variables select its input;
positions `2^(log_rows - 9)` select the T18/T22/T28 control geometries.

The CPU comparison checks every position at width <= 1024 and 1024 spread
positions otherwise. The output states the checked extent and records a
full-output checksum; sampled parity is not a full-output CPU comparison.

## Ring-switch rows

`ring_switch_rows` measures D64 D-role relation rows.
`AKITA_METAL_D_ROLE_COLUMNS` and `AKITA_METAL_D_ROLE_ROWS` select the
matrix. Both negacyclic and cyclic outputs are compared in full against the
CPU path, including the cold call. Jolt's T28 D128/rank-3 root uses 1,409,024
columns and 2 rows; the benchmark's default is 3 rows.

For full kernel parity, run:

```bash
cargo nextest run -p akita-metal --lib --test-threads 1
```
