use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use akita_field::AkitaError;
use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
use akita_prover::{
    CommitInnerWitness, CpuBackend, CpuPreparedSetup, PackedOneHotView, StreamingPackedOneHotView,
};
use akita_types::RingVec;
use metal::Buffer;

use crate::backend::{MetalCommitBackend, MetalCommitMetrics};
use crate::field::{Fp128Limbs, MetalField, F};
use crate::prepared::MetalPreparedSetup;
use crate::runtime::{
    DispatchOutcome, MetalRuntime, PackedCoefficientPackingIndexParams, PackedFoldIndexParams,
    PackedOneHotCommitParams, FP128_D512_FOLD_INDEX_COUNT_BUCKETS,
    FP128_D512_FOLD_INDEX_TILE_TASKS, FP128_D512_MAX_ROWS_PER_PARTIAL,
    FP128_D512_PACKING_INDEX_BUCKET_OFFSETS, FP128_D512_PACKING_INDEX_TILE_POSITIONS,
    FP128_D512_POSITION_PARTIALS,
};
use crate::MetalCommitError;

const RING_D: usize = 512;
const ONEHOT_K: usize = 256;
const COLUMN_CAPACITY: usize = 32;
const POSITION_PARTIALS: usize = FP128_D512_POSITION_PARTIALS;
const INNER_RANK: usize = 1;
const HYBRID_CPU_SPARSE_RATE_DIVISOR: usize = 12;
const HYBRID_CPU_DENSE_RATE_DIVISOR: usize = 24;
const HYBRID_CPU_MAX_BLOCKS: usize = 8;
const HYBRID_CPU_MIN_WORKLOAD_BLOCKS: usize = 12;
const HYBRID_SPARSE_PERCENT: usize = 30;
#[cfg(test)]
const TASKS_PER_STREAM: usize = 32;

pub(crate) struct ValidatedShape {
    active_a_cols: usize,
    output_coefficients: usize,
    positions_per_partial: usize,
    blocks_per_column: usize,
    populated_blocks_per_column: usize,
    live_columns: usize,
}

pub(crate) struct OpeningIndexPlan {
    pub(crate) fold: PackedFoldIndexParams,
    pub(crate) packing: PackedCoefficientPackingIndexParams,
    pub(crate) fold_allocation_bytes: usize,
    pub(crate) packing_allocation_bytes: usize,
    pub(crate) total_allocation_bytes: usize,
}

pub(crate) struct DeferredFoldIndex;
pub(crate) struct DeferredCoefficientPackingIndex;

pub(crate) fn opening_index_plan(
    num_rows: usize,
    live_columns: usize,
    num_positions: usize,
    blocks_per_column: usize,
) -> Result<OpeningIndexPlan, AkitaError> {
    let tasks_per_position = blocks_per_column
        .checked_mul(live_columns)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 fold-index tasks").into_akita()
        })?;
    let tiles_per_position = tasks_per_position.div_ceil(FP128_D512_FOLD_INDEX_TILE_TASKS);
    let fold_record_slots = num_positions
        .checked_mul(tiles_per_position)
        .and_then(|count| count.checked_mul(FP128_D512_FOLD_INDEX_TILE_TASKS))
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 fold-index records").into_akita()
        })?;
    let fold_count_entries = num_positions
        .checked_mul(tiles_per_position)
        .and_then(|count| count.checked_mul(FP128_D512_FOLD_INDEX_COUNT_BUCKETS))
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 fold-index counts").into_akita()
        })?;
    let fold_allocation_bytes = fold_record_slots
        .checked_mul(size_of::<u32>())
        .and_then(|bytes| {
            fold_count_entries
                .checked_mul(size_of::<u16>())
                .and_then(|counts| bytes.checked_add(counts))
        })
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 fold-index allocation").into_akita()
        })?;

    let position_tiles = num_positions.div_ceil(FP128_D512_PACKING_INDEX_TILE_POSITIONS);
    let packing_layouts = blocks_per_column
        .checked_mul(live_columns)
        .and_then(|count| count.checked_mul(2))
        .and_then(|count| count.checked_mul(position_tiles))
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 packing-index layouts").into_akita()
        })?;
    let packing_record_slots = packing_layouts
        .checked_mul(FP128_D512_PACKING_INDEX_TILE_POSITIONS)
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 packing-index records").into_akita()
        })?;
    let packing_offset_entries = packing_layouts
        .checked_mul(FP128_D512_PACKING_INDEX_BUCKET_OFFSETS)
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 packing-index offsets").into_akita()
        })?;
    let packing_allocation_bytes = packing_record_slots
        .checked_mul(size_of::<u16>())
        .and_then(|bytes| {
            packing_offset_entries
                .checked_mul(size_of::<u16>())
                .and_then(|offsets| bytes.checked_add(offsets))
        })
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 packing-index allocation").into_akita()
        })?;
    let total_allocation_bytes = fold_allocation_bytes
        .checked_add(packing_allocation_bytes)
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("fp128 D512 opening-index allocation").into_akita()
        })?;
    let output_coefficients = num_positions.checked_mul(RING_D).ok_or_else(|| {
        MetalCommitError::ShapeOverflow("fp128 D512 fold-index output").into_akita()
    })?;

    Ok(OpeningIndexPlan {
        fold: PackedFoldIndexParams {
            num_rows: to_u64(num_rows, "fp128 D512 fold-index rows")?,
            num_columns: to_u64(live_columns, "fp128 D512 fold-index columns")?,
            lane_stride: to_u64(live_columns, "fp128 D512 fold-index lane stride")?,
            num_positions: to_u64(num_positions, "fp128 D512 fold-index positions")?,
            position_start: 0,
            blocks_per_column: to_u64(blocks_per_column, "fp128 D512 fold-index blocks")?,
            tasks_per_position: to_u64(tasks_per_position, "fp128 D512 fold-index tasks")?,
            tiles_per_position: to_u64(tiles_per_position, "fp128 D512 fold-index tiles")?,
            record_slots: to_u64(fold_record_slots, "fp128 D512 fold-index records")?,
            count_entries: to_u64(fold_count_entries, "fp128 D512 fold-index counts")?,
            output_coefficients: to_u64(output_coefficients, "fp128 D512 fold-index output")?,
            fold_digits: 0,
            fold_log_basis: 0,
        },
        packing: PackedCoefficientPackingIndexParams {
            num_rows: to_u64(num_rows, "fp128 D512 packing-index rows")?,
            num_columns: to_u64(live_columns, "fp128 D512 packing-index columns")?,
            lane_stride: to_u64(live_columns, "fp128 D512 packing-index lane stride")?,
            num_positions: to_u64(num_positions, "fp128 D512 packing-index positions")?,
            blocks_per_column: to_u64(blocks_per_column, "fp128 D512 packing-index blocks")?,
            position_tiles: to_u64(position_tiles, "fp128 D512 packing-index position tiles")?,
            record_slots: to_u64(packing_record_slots, "fp128 D512 packing-index records")?,
            offset_entries: to_u64(packing_offset_entries, "fp128 D512 packing-index offsets")?,
        },
        fold_allocation_bytes,
        packing_allocation_bytes,
        total_allocation_bytes,
    })
}

pub(crate) trait PackedCommitInput: Sync {
    fn num_rows(&self) -> usize;
    fn populated_rows(&self) -> usize;
    fn num_columns(&self) -> usize;
    fn column_capacity(&self) -> usize;
    fn onehot_k(&self) -> usize;
    fn lane_count(&self) -> usize;
    fn hot_entries(&self) -> Result<usize, AkitaError>;
    fn wait_lanes(&self, rows: Range<usize>) -> Result<&[u8], AkitaError>;
    fn retains_opening_acceleration(&self) -> bool;
    fn retain_opening_acceleration<T: Send + Sync + 'static>(&self, value: Arc<T>);
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

    fn populated_rows(&self) -> usize {
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

    fn wait_lanes(&self, rows: Range<usize>) -> Result<&[u8], AkitaError> {
        let first = rows
            .start
            .checked_mul(self.num_columns())
            .ok_or_else(|| AkitaError::InvalidInput("packed first lane overflow".into()))?;
        let end = rows
            .end
            .checked_mul(self.num_columns())
            .ok_or_else(|| AkitaError::InvalidInput("packed final lane overflow".into()))?;
        self.lanes().get(first..end).ok_or_else(|| {
            AkitaError::InvalidInput("packed lane range exceeds the commit source".into())
        })
    }

    fn retains_opening_acceleration(&self) -> bool {
        PackedOneHotView::retains_opening_acceleration(*self)
    }

    fn retain_opening_acceleration<T: Send + Sync + 'static>(&self, value: Arc<T>) {
        let _ = PackedOneHotView::retain_opening_acceleration(*self, value);
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

    fn populated_rows(&self) -> usize {
        StreamingPackedOneHotView::populated_rows(self)
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

    fn wait_lanes(&self, rows: Range<usize>) -> Result<&[u8], AkitaError> {
        StreamingPackedOneHotView::wait_lanes(self, rows)
    }

    fn retains_opening_acceleration(&self) -> bool {
        true
    }

    fn retain_opening_acceleration<T: Send + Sync + 'static>(&self, value: Arc<T>) {
        StreamingPackedOneHotView::retain_opening_acceleration(self, value);
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

fn hybrid_cpu_tail_blocks(
    populated_blocks: usize,
    sample_hot_entries: usize,
    sample_lanes: usize,
) -> usize {
    if populated_blocks < HYBRID_CPU_MIN_WORKLOAD_BLOCKS || sample_lanes == 0 {
        return 0;
    }
    let sparse = (sample_hot_entries as u128) * 100
        <= (sample_lanes as u128) * (HYBRID_SPARSE_PERCENT as u128);
    let divisor = if sparse {
        HYBRID_CPU_SPARSE_RATE_DIVISOR
    } else {
        HYBRID_CPU_DENSE_RATE_DIVISOR
    };
    let target = (populated_blocks / divisor).max(1);
    (1usize << target.ilog2()).min(HYBRID_CPU_MAX_BLOCKS)
}

fn hybrid_sample(
    source: &impl PackedCommitInput,
    rows_per_block: usize,
) -> Result<(usize, usize), AkitaError> {
    let sample_rows = rows_per_block.min(source.populated_rows());
    let lanes = source.wait_lanes(0..sample_rows)?;
    Ok((lanes.iter().filter(|&&lane| lane != 0).count(), lanes.len()))
}

struct CpuTailCommit {
    witness: CommitInnerWitness<F>,
    hot_entries: usize,
    elapsed: std::time::Duration,
}

fn commit_cpu_tail<const D: usize>(
    backend: CpuBackend,
    prepared: &CpuPreparedSetup<F>,
    source: &impl PackedCommitInput,
    plan: CommitInnerPlan,
    first_block: usize,
    block_count: usize,
) -> Result<CpuTailCommit, AkitaError> {
    let start = Instant::now();
    let rows_per_block = plan
        .num_positions_per_block
        .checked_mul(2)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 rows per block").into_akita())?;
    let first_row = first_block
        .checked_mul(rows_per_block)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("hybrid CPU first row").into_akita())?;
    let final_row = first_block
        .checked_add(block_count)
        .and_then(|block| block.checked_mul(rows_per_block))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("hybrid CPU final row").into_akita())?;
    let lanes = source.wait_lanes(first_row..final_row)?;
    let view = PackedOneHotView::<F, D>::new(
        source.onehot_k(),
        source.column_capacity(),
        source.num_columns(),
        lanes,
    )?;
    let hot_entries = view.hot_entries();
    let mut witnesses = backend.commit_inner_group(prepared, vec![view], plan)?;
    if witnesses.len() != 1 {
        return Err(AkitaError::InvalidSetup(
            "hybrid CPU root commit returned an invalid witness count".into(),
        ));
    }
    Ok(CpuTailCommit {
        witness: witnesses.remove(0),
        hot_entries,
        elapsed: start.elapsed(),
    })
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
            "fp128 Metal D512 commit requires K256/nonempty-C<=32-cap32/rank1 and integral position partials"
                .into(),
        )
        .into_akita());
    }
    let positions_per_partial = plan.num_positions_per_block / POSITION_PARTIALS;
    if !positions_per_partial.is_multiple_of(4)
        || positions_per_partial
            .checked_mul(2)
            .is_none_or(|rows| rows > FP128_D512_MAX_ROWS_PER_PARTIAL)
    {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 partials must contain whole four-position matrix tiles within the signed-digit bound"
                .into(),
        )
        .into_akita());
    }
    let rows_per_block = plan
        .num_positions_per_block
        .checked_mul(2)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 rows per block").into_akita())?;
    if source.populated_rows() > source.num_rows()
        || !source.num_rows().is_multiple_of(rows_per_block)
    {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 panels require a bounded populated prefix and integral trace blocks".into(),
        )
        .into_akita());
    }
    let blocks_per_column = source.num_rows() / rows_per_block;
    if !matches!(blocks_per_column, 32 | 64 | 128 | 256 | 512) {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 panels require 32, 64, 128, 256, or 512 blocks per column".into(),
        )
        .into_akita());
    }
    let output_coefficients = COLUMN_CAPACITY
        .checked_mul(blocks_per_column)
        .and_then(|count| count.checked_mul(INNER_RANK))
        .and_then(|count| count.checked_mul(D))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 output").into_akita())?;
    let populated_blocks_per_column = source.populated_rows().div_ceil(rows_per_block);
    Ok(ValidatedShape {
        active_a_cols: plan.num_positions_per_block,
        output_coefficients,
        positions_per_partial,
        blocks_per_column,
        populated_blocks_per_column,
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
    let rows_per_block = plan
        .num_positions_per_block
        .checked_mul(2)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 rows per block").into_akita())?;
    let (sample_hot_entries, sample_lanes) = hybrid_sample(source, rows_per_block)?;
    let cpu_blocks = hybrid_cpu_tail_blocks(
        shape.populated_blocks_per_column,
        sample_hot_entries,
        sample_lanes,
    );
    let metal_blocks = shape
        .populated_blocks_per_column
        .checked_sub(cpu_blocks)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("hybrid Metal blocks").into_akita())?;
    let metal_work_units = shape
        .live_columns
        .checked_mul(metal_blocks)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 work units").into_akita())?;
    let cpu_work_units = shape
        .live_columns
        .checked_mul(cpu_blocks)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("hybrid CPU work units").into_akita())?;
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
        full_blocks_per_column: to_u64(metal_blocks, "fp128 D512 Metal blocks")?,
        boundary_columns: 0,
        num_blocks: to_u64(metal_work_units, "fp128 D512 work units")?,
        task_offset: 0,
        dispatch_tasks: to_u64(metal_work_units, "fp128 D512 dispatch work units")?,
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
    let cpu_backend = backend.cpu_backend();
    let (metal_result, cpu_result) = std::thread::scope(|scope| {
        let cpu_worker = (cpu_blocks != 0).then(|| {
            scope.spawn(|| {
                commit_cpu_tail::<D>(
                    cpu_backend,
                    &prepared.cpu,
                    source,
                    plan,
                    metal_blocks,
                    cpu_blocks,
                )
            })
        });
        let metal_result = source.dispatch(runtime, matrix.buffer.as_ref(), params);
        let cpu_result = cpu_worker.map(|worker| {
            worker.join().map_err(|_| {
                AkitaError::InvalidSetup("hybrid CPU root commit worker panicked".into())
            })?
        });
        Ok::<_, AkitaError>((metal_result, cpu_result))
    })?;
    let outcome = metal_result.map_err(MetalCommitError::into_akita)?;
    let cpu_commit = cpu_result.transpose()?;
    let opening_plan = opening_index_plan(
        source.num_rows(),
        shape.live_columns,
        plan.num_positions_per_block,
        shape.blocks_per_column,
    )?;
    let retains_opening_acceleration = source.retains_opening_acceleration();
    let (opening_index_time, opening_index_gpu_time, opening_index_bytes) =
        if retains_opening_acceleration
            && backend
                .opening_acceleration_policy()
                .allows_retention(opening_plan.total_allocation_bytes)
        {
            let index_start = Instant::now();
            let lanes = source.wait_lanes(0..source.num_rows())?;
            let index = runtime
                .prepare_packed_fp128_d512_fold_index(lanes, opening_plan.fold)
                .map_err(MetalCommitError::into_akita)?;
            let gpu_time = index.timings.gpu.unwrap_or_default();
            let allocation_bytes = index.allocation_bytes;
            debug_assert_eq!(allocation_bytes, opening_plan.fold_allocation_bytes);
            source.retain_opening_acceleration(Arc::new(index));

            let packing_index = runtime
                .prepare_packed_fp128_d512_coefficient_packing_index(lanes, opening_plan.packing)
                .map_err(MetalCommitError::into_akita)?;
            let gpu_time = gpu_time + packing_index.timings.gpu.unwrap_or_default();
            debug_assert_eq!(
                packing_index.allocation_bytes,
                opening_plan.packing_allocation_bytes
            );
            let allocation_bytes = allocation_bytes
                .checked_add(packing_index.allocation_bytes)
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("fp128 D512 opening-index allocation")
                        .into_akita()
                })?;
            debug_assert_eq!(allocation_bytes, opening_plan.total_allocation_bytes);
            source.retain_opening_acceleration(Arc::new(packing_index));
            (index_start.elapsed(), gpu_time, allocation_bytes)
        } else {
            if retains_opening_acceleration {
                source.retain_opening_acceleration(Arc::new(DeferredFoldIndex));
                source.retain_opening_acceleration(Arc::new(DeferredCoefficientPackingIndex));
            }
            (std::time::Duration::ZERO, std::time::Duration::ZERO, 0)
        };
    let output_reconstruction_start = Instant::now();
    let mut coefficients = outcome
        .coefficients
        .into_iter()
        .enumerate()
        .map(|(index, value)| F::from_device(value, index))
        .collect::<Result<Vec<_>, _>>()
        .map_err(MetalCommitError::into_akita)?;
    let merge_start = Instant::now();
    if let Some(cpu_commit) = &cpu_commit {
        let cpu_coefficients = cpu_commit.witness.inner_rows.coeffs();
        let cpu_column_width = cpu_blocks
            .checked_mul(D)
            .ok_or_else(|| MetalCommitError::ShapeOverflow("hybrid CPU column").into_akita())?;
        let output_column_width = shape
            .blocks_per_column
            .checked_mul(D)
            .ok_or_else(|| MetalCommitError::ShapeOverflow("hybrid output column").into_akita())?;
        let expected_cpu_coefficients = COLUMN_CAPACITY
            .checked_mul(cpu_column_width)
            .ok_or_else(|| MetalCommitError::ShapeOverflow("hybrid CPU output").into_akita())?;
        if cpu_coefficients.len() != expected_cpu_coefficients {
            return Err(AkitaError::InvalidSetup(
                "hybrid CPU root commit returned an invalid output shape".into(),
            ));
        }
        for column in 0..COLUMN_CAPACITY {
            let source_start = column * cpu_column_width;
            let output_start = column * output_column_width + metal_blocks * D;
            coefficients[output_start..output_start + cpu_column_width]
                .copy_from_slice(&cpu_coefficients[source_start..source_start + cpu_column_width]);
        }
    }
    let merge_time = merge_start.elapsed();
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
        .live_columns
        .checked_mul(metal_blocks)
        .and_then(|count| count.checked_mul(INNER_RANK))
        .and_then(|count| count.checked_mul(D))
        .and_then(|count| count.checked_mul(POSITION_PARTIALS - 1))
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
    let task_row_probes = metal_blocks
        .checked_mul(plan.num_positions_per_block)
        .and_then(|count| count.checked_mul(2))
        .and_then(|count| count.checked_mul(shape.live_columns))
        .and_then(|count| count.checked_mul(INNER_RANK))
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 probes").into_akita())?;
    let cpu_hot_entries = cpu_commit.as_ref().map_or(0, |commit| commit.hot_entries);
    let metal_hot_entries = hot_entries.checked_sub(cpu_hot_entries).ok_or_else(|| {
        AkitaError::InvalidSetup("hybrid CPU hot-entry count exceeds the source total".into())
    })?;
    let selected_lane_broadcasts = metal_hot_entries
        .checked_mul(INNER_RANK)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 broadcasts").into_akita())?;
    backend
        .record_metrics(MetalCommitMetrics {
            kernel: outcome.kernel,
            blocks_per_threadgroup: outcome.blocks_per_threadgroup,
            columns_per_threadgroup: outcome.columns_per_threadgroup,
            cpu_blocks,
            metal_blocks,
            metal_full_blocks: metal_blocks,
            metal_boundary_columns: 0,
            cpu_columns: shape.live_columns * usize::from(cpu_blocks != 0),
            metal_columns: shape.live_columns,
            cpu_work_units,
            metal_work_units,
            cpu_rank_rows: INNER_RANK * usize::from(cpu_blocks != 0),
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
            opening_index_time,
            opening_index_gpu_time,
            opening_index_bytes,
            buffer_setup_time: outcome.timings.buffer_setup,
            command_wall_time: outcome.timings.command_wall,
            gpu_time: outcome.timings.gpu,
            cpu_time: cpu_commit
                .as_ref()
                .map_or(std::time::Duration::ZERO, |commit| commit.elapsed),
            readback_copy_time: outcome.timings.readback_copy,
            output_reconstruction_time,
            merge_time,
            total_time: total_start.elapsed(),
            digit_rows_calls: 0,
            digit_rows_metal_calls: 0,
            digit_rows_max_rows: 0,
            digit_rows_max_columns: 0,
            digit_rows_max_batch: 0,
            digit_rows_time: std::time::Duration::ZERO,
            digit_rows_gpu_time: std::time::Duration::ZERO,
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

    use akita_challenges::SparseChallenge;
    use akita_prover::compute::{
        BalancedDigitRequest, CommitInnerPlan, DecomposeFoldChunk, DecomposeFoldChunkSink,
        DecomposeFoldPlan, OpeningFoldKernel, RootCommitKernel,
    };
    use akita_prover::{
        AkitaProverSetup, ComputeBackendSetup, CpuBackend, OneHotPoly, PackedOneHotPoly,
        PackedOneHotStreamBuffer, RootCommitSource, RootOpeningSource, RootPolyMeta,
        StreamingPackedOneHotPoly,
    };
    use akita_types::SetupMatrixCapacity;

    use crate::{
        MetalCommitBackend, MetalExecutionPolicy, MetalOneHotKernel, OpeningAccelerationPolicy,
    };

    use super::F;

    #[derive(Default)]
    struct DigitSink {
        chunks: usize,
    }

    impl DecomposeFoldChunkSink for DigitSink {
        fn balanced_digit_request(&self) -> Option<BalancedDigitRequest> {
            Some(BalancedDigitRequest {
                num_digits: 1,
                log_basis: 3,
            })
        }

        fn consume(
            &mut self,
            chunk: DecomposeFoldChunk<'_>,
        ) -> Result<(), akita_field::AkitaError> {
            assert!(chunk.balanced_digits().is_some());
            self.chunks += 1;
            Ok(())
        }
    }

    #[test]
    fn opening_index_plan_bounds_t28_retention() {
        let t28 = super::opening_index_plan(1 << 28, 30, 1 << 18, 512).unwrap();
        assert_eq!(t28.fold_allocation_bytes, 32_715_571_200);
        assert_eq!(t28.packing_allocation_bytes, 18_182_307_840);
        assert_eq!(t28.total_allocation_bytes, 50_897_879_040);

        let budget = OpeningAccelerationPolicy::RetainUpToBytes(32 << 30);
        assert!(!budget.allows_retention(t28.total_allocation_bytes));

        let t27 = super::opening_index_plan(1 << 27, 30, 1 << 17, 512).unwrap();
        assert!(budget.allows_retention(t27.total_allocation_bytes));
    }

    #[test]
    fn deferred_fold_index_is_fused_at_opening() {
        const ROWS: usize = 4_096;
        const COLUMNS: usize = 3;
        const CAPACITY: usize = 32;
        const POSITIONS_PER_BLOCK: usize = 64;
        let commit_plan = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            25,
            1,
            SetupMatrixCapacity {
                num_field_elements: POSITIONS_PER_BLOCK * super::RING_D,
            },
        )
        .unwrap();
        let lanes = (0..ROWS * COLUMNS)
            .map(|index| ((index * 73 + 19) % 256) as u8)
            .collect::<Vec<_>>();
        let poly =
            PackedOneHotPoly::<F>::new(super::ONEHOT_K, CAPACITY, COLUMNS, lanes.clone()).unwrap();
        let reference = OneHotPoly::<F, u8>::new(
            super::ONEHOT_K,
            (0..CAPACITY)
                .flat_map(|column| {
                    let lanes = &lanes;
                    (0..ROWS).map(move |row| {
                        (column < COLUMNS)
                            .then(|| lanes[row * COLUMNS + column])
                            .filter(|&lane| lane != 0)
                    })
                })
                .collect(),
        )
        .unwrap();
        let blocks_per_column = ROWS / 2 / POSITIONS_PER_BLOCK;
        let challenges = (0..CAPACITY * blocks_per_column)
            .map(|block| SparseChallenge {
                positions: (0..19)
                    .map(|term| ((term * 8 + block * 16) % super::RING_D) as u32)
                    .collect::<Vec<_>>()
                    .into(),
                coeffs: (0..19)
                    .map(|term| {
                        if (term + block).is_multiple_of(2) {
                            1
                        } else {
                            -1
                        }
                    })
                    .collect::<Vec<_>>()
                    .into(),
            })
            .collect::<Vec<_>>();
        let opening_plan = DecomposeFoldPlan {
            challenges: &challenges,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits: 1,
            log_basis: 3,
        };

        let cpu = CpuBackend::DEFAULT;
        let expected = cpu
            .decompose_fold(
                None,
                RootOpeningSource::<F, 512>::opening_view(&reference).unwrap(),
                opening_plan,
            )
            .unwrap();
        let metal = MetalCommitBackend::new_with_opening_acceleration_policy(
            MetalExecutionPolicy::RequireMetal,
            OpeningAccelerationPolicy::RetainUpToBytes(0),
        )
        .unwrap();
        let prepared = metal.prepare_setup(&setup).unwrap();
        metal
            .commit_inner_group(
                &prepared,
                vec![RootCommitSource::<F, 512>::commit_view(&poly).unwrap()],
                commit_plan,
            )
            .unwrap();
        assert_eq!(
            metal
                .last_commit_metrics()
                .unwrap()
                .unwrap()
                .opening_index_bytes,
            0
        );

        metal.begin_opening_metrics().unwrap();
        let mut sink = DigitSink::default();
        let actual = metal
            .decompose_fold_packed_fp128_d512_streaming(
                poly.view::<512>().unwrap(),
                opening_plan,
                &mut sink,
            )
            .unwrap();

        assert_eq!(actual, expected);
        assert!(sink.chunks > 0);
        let metrics = metal.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.opening_index_bytes, 0);
        assert_eq!(metrics.packed_decompose_indexed_calls, 1);
        assert!(poly
            .view::<512>()
            .unwrap()
            .opening_acceleration::<super::DeferredFoldIndex>()
            .is_none());
    }

    #[test]
    fn hybrid_tail_tracks_density_and_caps_unified_memory_contention() {
        let sparse = |blocks| super::hybrid_cpu_tail_blocks(blocks, 3, 10);
        let dense = |blocks| super::hybrid_cpu_tail_blocks(blocks, 4, 10);
        assert_eq!(sparse(1), 0);
        assert_eq!(sparse(11), 0);
        assert_eq!(sparse(12), 1);
        assert_eq!(sparse(27), 2);
        assert_eq!(dense(27), 1);
        assert_eq!(sparse(216), 8);
        assert_eq!(dense(216), 8);
        assert_eq!(sparse(256), 8);
    }

    #[test]
    fn task_and_partial_grids_are_bijective() {
        for (columns, blocks) in [
            (25usize, 32usize),
            (28, 64),
            (32, 128),
            (32, 256),
            (30, 512),
        ] {
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
    fn wide_digit_bound_fits_largest_position_partial() {
        let max_rows = (1usize << 18) * 2 / super::POSITION_PARTIALS;
        let max_digit_sum = max_rows * u16::MAX as usize;
        assert_eq!(max_rows, super::FP128_D512_MAX_ROWS_PER_PARTIAL);
        assert!(max_digit_sum < i32::MAX as usize);
    }

    #[test]
    fn exact_fp128_d512_panels_match_cpu_on_sparse_boundaries() {
        const ROWS: usize = 4_096;
        const CAPACITY: usize = 32;
        const POSITIONS_PER_BLOCK: usize = 64;
        let plan = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            25,
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
            assert_eq!(RootPolyMeta::<F>::num_vars(&poly), 25);
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
            assert_eq!(metrics.kernel, MetalOneHotKernel::PackedFp128D512WidePanels);
            assert_eq!(metrics.cpu_work_units, columns);
            assert_eq!(metrics.metal_work_units, columns * 31);
        }
    }

    #[test]
    fn exact_fp128_d512_panels_match_cpu_at_512_blocks() {
        const ROWS: usize = 65_536;
        const COLUMNS: usize = 1;
        const CAPACITY: usize = 32;
        const POSITIONS_PER_BLOCK: usize = 64;
        let plan = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            29,
            1,
            SetupMatrixCapacity {
                num_field_elements: POSITIONS_PER_BLOCK * super::RING_D,
            },
        )
        .unwrap();
        let lanes = (0..ROWS)
            .map(|row| match row {
                0 | 31 | 32 | 65_535 => 255,
                row if row.is_multiple_of(4) => 0,
                row => ((row * 73) % 255 + 1) as u8,
            })
            .collect();
        let poly = PackedOneHotPoly::<F>::new(super::ONEHOT_K, CAPACITY, COLUMNS, lanes).unwrap();
        assert_eq!(RootPolyMeta::<F>::num_vars(&poly), 29);

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let cpu_output = cpu
            .commit_inner_group(
                &cpu_prepared,
                vec![RootCommitSource::<F, 512>::commit_view(&poly).unwrap()],
                plan,
            )
            .unwrap();
        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        let metal_output = metal
            .commit_inner_group(
                &metal_prepared,
                vec![RootCommitSource::<F, 512>::commit_view(&poly).unwrap()],
                plan,
            )
            .unwrap();

        assert_eq!(cpu_output[0].inner_rows, metal_output[0].inner_rows);
        let metrics = metal.last_commit_metrics().unwrap().unwrap();
        assert_eq!(metrics.kernel, MetalOneHotKernel::PackedFp128D512WidePanels);
        assert_eq!(metrics.cpu_work_units, 8);
        assert_eq!(metrics.metal_work_units, 504);
    }

    #[test]
    fn streaming_fp128_d512_panels_match_resident_input() {
        const ROWS: usize = 4_096;
        const POPULATED_ROWS: usize = 2_345;
        const COLUMNS: usize = 29;
        const CAPACITY: usize = 32;
        const POSITIONS_PER_BLOCK: usize = 64;
        let lane = |row: usize, column: usize| {
            if row >= POPULATED_ROWS || (row + column).is_multiple_of(5) {
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
        let (stream, mut writer) =
            StreamingPackedOneHotPoly::<F>::from_buffer_with_zero_suffix(buffer, POPULATED_ROWS)
                .unwrap();
        let plan = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            25,
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
                let mut remaining = POPULATED_ROWS;
                while remaining != 0 {
                    let rows = remaining.min(ROWS / 8);
                    writer
                        .fill_next_rows_in_place(rows, |row, output| {
                            output
                                .iter_mut()
                                .enumerate()
                                .for_each(|(column, value)| *value = lane(row, column));
                            Ok(())
                        })
                        .unwrap();
                    remaining -= rows;
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
        assert_eq!(stream.finalize().unwrap().lanes(), resident.lanes());
        let metrics = metal.last_commit_metrics().unwrap().unwrap();
        let populated_blocks = POPULATED_ROWS.div_ceil(POSITIONS_PER_BLOCK * 2);
        let sample_rows = POSITIONS_PER_BLOCK * 2;
        let sample_lanes = &resident.lanes()[..sample_rows * COLUMNS];
        let cpu_blocks = super::hybrid_cpu_tail_blocks(
            populated_blocks,
            sample_lanes.iter().filter(|&&lane| lane != 0).count(),
            sample_lanes.len(),
        );
        assert_eq!(metrics.cpu_work_units, COLUMNS * cpu_blocks);
        assert_eq!(
            metrics.metal_work_units,
            COLUMNS * (populated_blocks - cpu_blocks)
        );
    }
}
