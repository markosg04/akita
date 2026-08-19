use std::mem::size_of;
use std::time::Instant;

use akita_field::AkitaError;
use akita_prover::compute::CommitInnerPlan;
use akita_prover::{CommitInnerWitness, PackedOneHotView, StreamingPackedOneHotView};
use akita_types::RingVec;
use metal::Buffer;

use crate::backend::{MetalCommitBackend, MetalCommitMetrics};
use crate::field::{Fp128Limbs, MetalField, F};
use crate::prepared::MetalPreparedSetup;
use crate::runtime::{DispatchOutcome, MetalRuntime, PackedOneHotCommitParams};
use crate::MetalCommitError;

const RING_D: usize = 512;
const ONEHOT_K: usize = 256;
const COLUMN_CAPACITY: usize = 32;
const POSITION_PARTIALS: usize = 4;
const INNER_RANK: usize = 1;
#[cfg(test)]
const TASKS_PER_STREAM: usize = 32;

pub(crate) struct ValidatedShape {
    active_a_cols: usize,
    output_coefficients: usize,
    positions_per_partial: usize,
    blocks_per_column: usize,
    live_columns: usize,
}

pub(crate) trait PackedCommitInput {
    fn num_rows(&self) -> usize;
    fn num_columns(&self) -> usize;
    fn column_capacity(&self) -> usize;
    fn onehot_k(&self) -> usize;
    fn lane_count(&self) -> usize;
    fn hot_entries(&self) -> Result<usize, AkitaError>;
    fn dispatch(
        &self,
        runtime: &MetalRuntime,
        matrix: &Buffer,
        params: PackedOneHotCommitParams,
    ) -> Result<DispatchOutcome, MetalCommitError>;
}

impl<const D: usize> PackedCommitInput for PackedOneHotView<'_, F, D> {
    fn num_rows(&self) -> usize {
        (*self).num_rows()
    }

    fn num_columns(&self) -> usize {
        (*self).num_columns()
    }

    fn column_capacity(&self) -> usize {
        (*self).column_capacity()
    }

    fn onehot_k(&self) -> usize {
        (*self).onehot_k()
    }

    fn lane_count(&self) -> usize {
        (*self).lanes().len()
    }

    fn hot_entries(&self) -> Result<usize, AkitaError> {
        Ok((*self).hot_entries())
    }

    fn dispatch(
        &self,
        runtime: &MetalRuntime,
        matrix: &Buffer,
        params: PackedOneHotCommitParams,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        runtime.dispatch_packed_onehot(matrix, (*self).lanes(), params)
    }
}

impl<const D: usize> PackedCommitInput for StreamingPackedOneHotView<F, D> {
    fn num_rows(&self) -> usize {
        StreamingPackedOneHotView::num_rows(self)
    }

    fn num_columns(&self) -> usize {
        StreamingPackedOneHotView::num_columns(self)
    }

    fn column_capacity(&self) -> usize {
        StreamingPackedOneHotView::column_capacity(self)
    }

    fn onehot_k(&self) -> usize {
        StreamingPackedOneHotView::onehot_k(self)
    }

    fn lane_count(&self) -> usize {
        StreamingPackedOneHotView::lane_count(self)
    }

    fn hot_entries(&self) -> Result<usize, AkitaError> {
        StreamingPackedOneHotView::wait_hot_entries(self)
    }

    fn dispatch(
        &self,
        runtime: &MetalRuntime,
        matrix: &Buffer,
        params: PackedOneHotCommitParams,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        runtime.dispatch_streaming_packed_onehot(matrix, self, params)
    }
}

pub(crate) fn validate_shape<const D: usize>(
    source: &impl PackedCommitInput,
    plan: CommitInnerPlan,
) -> Result<ValidatedShape, AkitaError> {
    if D != RING_D
        || source.onehot_k() != ONEHOT_K
        || source.column_capacity() != COLUMN_CAPACITY
        || !(1..=COLUMN_CAPACITY).contains(&source.num_columns())
        || plan.n_a != INNER_RANK
        || plan.num_digits_inner != 1
        || plan.num_positions_per_block == 0
        || !plan
            .num_positions_per_block
            .is_multiple_of(POSITION_PARTIALS)
    {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 Metal D512 commit requires K256/nonempty-C<=32-cap32/rank1 and four integral partials"
                .into(),
        )
        .into_akita());
    }
    let positions_per_partial = plan.num_positions_per_block / POSITION_PARTIALS;
    if !positions_per_partial.is_multiple_of(4) {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 partials must contain whole four-position matrix tiles".into(),
        )
        .into_akita());
    }
    let rows_per_block = plan
        .num_positions_per_block
        .checked_mul(2)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 rows per block").into_akita())?;
    if !source.num_rows().is_multiple_of(rows_per_block) {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 panels require an integral number of trace blocks".into(),
        )
        .into_akita());
    }
    let blocks_per_column = source.num_rows() / rows_per_block;
    if !matches!(blocks_per_column, 32 | 64 | 128 | 256) {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 panels require 32, 64, 128, or 256 blocks per column".into(),
        )
        .into_akita());
    }
    let output_coefficients = COLUMN_CAPACITY
        .checked_mul(blocks_per_column)
        .and_then(|count| count.checked_mul(INNER_RANK))
        .and_then(|count| count.checked_mul(D))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 output").into_akita())?;
    Ok(ValidatedShape {
        active_a_cols: plan.num_positions_per_block,
        output_coefficients,
        positions_per_partial,
        blocks_per_column,
        live_columns: source.num_columns(),
    })
}

pub(crate) fn commit_validated<const D: usize>(
    backend: &MetalCommitBackend<F>,
    prepared: &MetalPreparedSetup<F>,
    runtime: &MetalRuntime,
    source: &impl PackedCommitInput,
    plan: CommitInnerPlan,
    shape: ValidatedShape,
) -> Result<CommitInnerWitness<F>, AkitaError> {
    let total_start = Instant::now();
    let matrix = prepared.matrix(runtime, D, plan.n_a, shape.active_a_cols)?;
    let work_units = shape
        .live_columns
        .checked_mul(shape.blocks_per_column)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 work units").into_akita())?;
    let params = PackedOneHotCommitParams {
        num_rows: to_u64(source.num_rows(), "fp128 D512 row count")?,
        num_columns: to_u64(shape.live_columns, "fp128 D512 column count")?,
        lane_stride: to_u64(shape.live_columns, "fp128 D512 lane stride")?,
        column_capacity: COLUMN_CAPACITY as u64,
        onehot_k: ONEHOT_K as u64,
        ring_d: D as u64,
        n_a: INNER_RANK as u64,
        positions_per_block: to_u64(plan.num_positions_per_block, "fp128 D512 block positions")?,
        num_digits_inner: 1,
        blocks_per_column: to_u64(shape.blocks_per_column, "fp128 D512 block count")?,
        full_blocks_per_column: to_u64(shape.blocks_per_column, "fp128 D512 full blocks")?,
        boundary_columns: 0,
        num_blocks: to_u64(work_units, "fp128 D512 work units")?,
        task_offset: 0,
        dispatch_tasks: to_u64(work_units, "fp128 D512 dispatch work units")?,
        lane_row_offset: 0,
        output_coefficients: to_u64(shape.output_coefficients, "fp128 D512 output")?,
        columns_per_threadgroup: 1,
        position_partials_per_block: POSITION_PARTIALS as u64,
        positions_per_partial: to_u64(
            shape.positions_per_partial,
            "fp128 D512 positions per partial",
        )?,
        log_ring_d: 9,
    };
    let outcome = source
        .dispatch(runtime, matrix.buffer.as_ref(), params)
        .map_err(MetalCommitError::into_akita)?;
    let output_reconstruction_start = Instant::now();
    let coefficients = outcome
        .coefficients
        .into_iter()
        .enumerate()
        .map(|(index, value)| F::from_device(value, index))
        .collect::<Result<Vec<_>, _>>()
        .map_err(MetalCommitError::into_akita)?;
    let witness = CommitInnerWitness {
        inner_rows: RingVec::from_coeffs_with_ring_dim(coefficients, D)?,
    };
    let output_reconstruction_time = output_reconstruction_start.elapsed();

    let hot_entries = source.hot_entries()?;
    let field_additions = hot_entries
        .checked_mul(INNER_RANK)
        .and_then(|count| count.checked_mul(D))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 additions").into_akita())?;
    let reduction_field_additions = shape
        .output_coefficients
        .checked_mul(POSITION_PARTIALS - 1)
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 reduction additions").into_akita()
        })?;
    let gathered_matrix_bytes = field_additions
        .checked_mul(size_of::<Fp128Limbs>())
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 gathered bytes").into_akita())?;
    let modeled_matrix_read_bytes = matrix
        .bytes
        .checked_mul(outcome.matrix_block_streams)
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 modeled matrix bytes").into_akita()
        })?;
    let task_row_probes = source
        .lane_count()
        .checked_mul(INNER_RANK)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 probes").into_akita())?;
    let selected_lane_broadcasts = hot_entries
        .checked_mul(INNER_RANK)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 broadcasts").into_akita())?;
    backend
        .record_metrics(MetalCommitMetrics {
            kernel: outcome.kernel,
            blocks_per_threadgroup: outcome.blocks_per_threadgroup,
            columns_per_threadgroup: outcome.columns_per_threadgroup,
            cpu_blocks: 0,
            metal_blocks: shape.blocks_per_column,
            metal_full_blocks: shape.blocks_per_column,
            metal_boundary_columns: 0,
            cpu_columns: 0,
            metal_columns: shape.live_columns,
            cpu_work_units: 0,
            metal_work_units: work_units,
            cpu_rank_rows: 0,
            metal_rank_rows: INNER_RANK,
            num_sources: 1,
            hot_entries,
            lane_scan_ballots: to_u64(task_row_probes / 8, "fp128 D512 ballots")?,
            selected_lane_broadcasts: to_u64(selected_lane_broadcasts, "fp128 D512 broadcasts")?,
            field_additions: to_u64(field_additions, "fp128 D512 additions")?,
            reduction_field_additions: to_u64(
                reduction_field_additions,
                "fp128 D512 reduction additions",
            )?,
            gathered_matrix_bytes: to_u64(gathered_matrix_bytes, "fp128 D512 gathered bytes")?,
            modeled_matrix_read_bytes: to_u64(
                modeled_matrix_read_bytes,
                "fp128 D512 modeled matrix bytes",
            )?,
            modeled_lane_read_bytes: to_u64(task_row_probes, "fp128 D512 modeled lane bytes")?,
            index_bytes: source.lane_count(),
            input_zero_copy: outcome.input_zero_copy,
            output_bytes: shape.output_coefficients * size_of::<Fp128Limbs>(),
            scratch_bytes: outcome.scratch_bytes,
            matrix_bytes: matrix.bytes,
            matrix_cache_hit: matrix.cache_hit,
            matrix_prepare_time: matrix.prepare_time,
            index_pack_time: std::time::Duration::ZERO,
            buffer_setup_time: outcome.timings.buffer_setup,
            command_wall_time: outcome.timings.command_wall,
            gpu_time: outcome.timings.gpu,
            cpu_time: std::time::Duration::ZERO,
            readback_copy_time: outcome.timings.readback_copy,
            output_reconstruction_time,
            merge_time: std::time::Duration::ZERO,
            total_time: total_start.elapsed(),
            digit_rows_calls: 0,
            digit_rows_time: std::time::Duration::ZERO,
            compression_calls: 0,
            compression_time: std::time::Duration::ZERO,
        })
        .map_err(MetalCommitError::into_akita)?;
    Ok(witness)
}

fn to_u64(value: usize, name: &'static str) -> Result<u64, AkitaError> {
    u64::try_from(value).map_err(|_| MetalCommitError::ShapeOverflow(name).into_akita())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
    use akita_prover::{
        AkitaProverSetup, ComputeBackendSetup, CpuBackend, PackedOneHotPoly,
        PackedOneHotStreamBuffer, RootCommitSource, RootPolyMeta, StreamingPackedOneHotPoly,
    };
    use akita_types::SetupMatrixCapacity;

    use crate::{MetalCommitBackend, MetalExecutionPolicy, MetalOneHotKernel};

    use super::F;

    #[test]
    fn task_and_partial_grids_are_bijective() {
        for (columns, blocks) in [(25usize, 32usize), (28, 64), (32, 128), (32, 256)] {
            let tasks = columns * blocks;
            let streams = tasks.div_ceil(super::TASKS_PER_STREAM);
            let mut task_map = BTreeSet::new();
            for stream in 0..streams {
                for simdgroup in 0..super::TASKS_PER_STREAM {
                    let task = stream * super::TASKS_PER_STREAM + simdgroup;
                    if task < tasks {
                        assert!(task_map.insert((task / columns, task % columns)));
                    }
                }
            }
            assert_eq!(task_map.len(), tasks);

            let mut partial_map = BTreeSet::new();
            for group in 0..streams * super::POSITION_PARTIALS {
                assert!(partial_map.insert((group / streams, group % streams,)));
            }
            assert_eq!(partial_map.len(), streams * super::POSITION_PARTIALS);
        }
    }

    #[test]
    fn parity_specialization_matches_negacyclic_rotation() {
        for row_parity in 0..2 {
            for hot in 1..256 {
                let shift = row_parity * 256 + hot;
                for coefficient in 0..512 {
                    let generic = ((coefficient + 512 - shift) & 511, coefficient >= shift);
                    let group = coefficient / 128;
                    let specialized = if row_parity == 0 {
                        if group < 2 {
                            ((coefficient + 512 - hot) & 511, coefficient >= hot)
                        } else {
                            (coefficient - hot, true)
                        }
                    } else if group < 2 {
                        (coefficient + 256 - hot, false)
                    } else {
                        (
                            (coefficient + 512 - (256 + hot)) & 511,
                            coefficient >= 256 + hot,
                        )
                    };
                    assert_eq!(specialized, generic);
                }
            }
        }
    }

    #[test]
    fn exact_fp128_d512_panels_match_cpu_on_sparse_boundaries() {
        const ROWS: usize = 1_024;
        const CAPACITY: usize = 32;
        const POSITIONS_PER_BLOCK: usize = 16;
        let plan = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            23,
            1,
            SetupMatrixCapacity {
                num_field_elements: POSITIONS_PER_BLOCK * super::RING_D,
            },
        )
        .unwrap();
        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();

        for columns in 25..=CAPACITY {
            let lanes = (0..ROWS * columns)
                .map(|index| match (index / columns, index % columns) {
                    (0 | 31 | 32 | 1_023, 0) => 255,
                    (row, column) if column + 1 == columns => (row % 256) as u8,
                    (row, column) if (row + column).is_multiple_of(4) => 0,
                    (row, column) => ((row * 73 + column * 19) % 255 + 1) as u8,
                })
                .collect();
            let poly =
                PackedOneHotPoly::<F>::new(super::ONEHOT_K, CAPACITY, columns, lanes).unwrap();
            assert_eq!(RootPolyMeta::<F>::num_vars(&poly), 23);
            let cpu_output = cpu
                .commit_inner_group(
                    &cpu_prepared,
                    vec![RootCommitSource::<F, 512>::commit_view(&poly).unwrap()],
                    plan,
                )
                .unwrap();
            let metal_output = metal
                .commit_inner_group(
                    &metal_prepared,
                    vec![RootCommitSource::<F, 512>::commit_view(&poly).unwrap()],
                    plan,
                )
                .unwrap();
            assert_eq!(cpu_output[0].inner_rows, metal_output[0].inner_rows);
            let metrics = metal.last_commit_metrics().unwrap().unwrap();
            assert_eq!(metrics.kernel, MetalOneHotKernel::PackedFp128D512Panels);
            assert_eq!(metrics.cpu_work_units, 0);
            assert_eq!(metrics.metal_work_units, columns * 32);
        }
    }

    #[test]
    fn streaming_fp128_d512_panels_match_resident_input() {
        const ROWS: usize = 16_384;
        const COLUMNS: usize = 29;
        const CAPACITY: usize = 32;
        const POSITIONS_PER_BLOCK: usize = 32;
        let lane = |row: usize, column: usize| {
            if (row + column).is_multiple_of(5) {
                0
            } else {
                ((row * 73 + column * 19) % 255 + 1) as u8
            }
        };
        let resident = PackedOneHotPoly::<F>::new(
            super::ONEHOT_K,
            CAPACITY,
            COLUMNS,
            (0..ROWS)
                .flat_map(|row| (0..COLUMNS).map(move |column| lane(row, column)))
                .collect(),
        )
        .unwrap();
        let buffer =
            PackedOneHotStreamBuffer::zeroed(super::ONEHOT_K, CAPACITY, COLUMNS, ROWS).unwrap();
        let (stream, mut writer) = StreamingPackedOneHotPoly::<F>::from_buffer(buffer);
        let plan = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            23,
            1,
            SetupMatrixCapacity {
                num_field_elements: POSITIONS_PER_BLOCK * super::RING_D,
            },
        )
        .unwrap();
        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let prepared = metal.prepare_setup(&setup).unwrap();
        let resident_output = metal
            .commit_inner_group(
                &prepared,
                vec![RootCommitSource::<F, 512>::commit_view(&resident).unwrap()],
                plan,
            )
            .unwrap();
        let streaming_output = std::thread::scope(|scope| {
            let producer = scope.spawn(move || {
                for _ in 0..8 {
                    writer
                        .fill_next_rows_in_place(ROWS / 8, |row, output| {
                            output
                                .iter_mut()
                                .enumerate()
                                .for_each(|(column, value)| *value = lane(row, column));
                            Ok(())
                        })
                        .unwrap();
                }
                writer.finish().unwrap();
            });
            let output = metal
                .commit_inner_group(
                    &prepared,
                    vec![RootCommitSource::<F, 512>::commit_view(&stream).unwrap()],
                    plan,
                )
                .unwrap();
            producer.join().unwrap();
            output
        });
        assert_eq!(
            resident_output[0].inner_rows,
            streaming_output[0].inner_rows
        );
        assert!(
            metal
                .last_commit_metrics()
                .unwrap()
                .unwrap()
                .input_zero_copy
        );
        assert_eq!(stream.finalize().unwrap().lanes(), resident.lanes());
    }
}
