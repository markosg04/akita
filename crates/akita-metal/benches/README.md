# Packed commitment and opening controls

Run one process at a time on an idle Mac. These are manual component
benchmarks, not full-proof throughput scores. Use a process timeout and
retain cold and warm results separately. Campaign comparisons use120s gaps.

`cargo bench -p akita-metal --bench packed_onehot_commit` measures the
actual packed commit path. `AKITA_METAL_RING_D=128` selects rank3;
`AKITA_METAL_LOG_ROWS`, `AKITA_METAL_COLUMNS`,
`AKITA_METAL_POSITIONS_PER_BLOCK`, `AKITA_METAL_DENSITY_PERCENT` and
`AKITA_METAL_SAMPLES` select its shape. Repeated-output equality is a
consistency check; independent CPU parity is covered by the kernel tests
and production commitment checks, not established by repetition.

`cargo bench -p akita-metal --bench packed_onehot_fold` measures the D128
fold using the canonical subring64 production challenge sampler/embedding.
The same shape variables select its input; use positions2^(log_rows-9)
for T18/T22/T28 controls. CPU comparison covers every position at width<=1024
and1024 spread positions otherwise; the output states this distinction and
records the full-output checksum. The source remains synthetic.

`cargo bench -p akita-metal --bench ring_switch_rows` measures D64 D-role
relation rows. `AKITA_METAL_D_ROLE_COLUMNS` and `AKITA_METAL_D_ROLE_ROWS`
select the matrix. Both negacyclic and cyclic outputs are compared fully
against the CPU path, including the first/cold call. Jolt's C2 root shape
is1409024 columns and2 rows; it is not the benchmark's default row count3.

The earlier E6 on-device synthetic-matrix prototype is not a production
kernel and is excluded from the landing revision. Its source remains in
the preserved experimental history.
