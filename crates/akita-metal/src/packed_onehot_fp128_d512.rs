use std::mem::size_of;
use std::time::{Duration, Instant};

use akita_error::AkitaError;
use akita_prover::compute::CommitInnerPlan;
use akita_prover::CommitInnerWitness;
use akita_types::RingVec;

use crate::backend::{MetalBackend, MetalCommitMetrics};
use crate::field::{Fp128Limbs, MetalField, F};
use crate::onehot::to_u64;
use crate::packed_onehot::PackedOneHotCommitView;
use crate::prepared::MetalPreparedSetup;
use crate::runtime::{
    MetalOneHotKernel, MetalRuntime, PackedOneHotCommitParams, FP128_D512_POSITION_PARTIALS,
    FP128_D512_TASKS_PER_STREAM,
};
use crate::MetalCommitError;

const RING_D: usize = 512;
const K16_COLUMN_CAPACITY: usize = 64;
const K256_COLUMN_CAPACITY: usize = 32;
const INNER_RANK: usize = 1;
const FIVE_STREAM_MAX_POSITIONS: usize = 1 << 16;
const FP128_D512_POSITIONS_PER_TILE: usize = 4;

#[derive(Clone, Debug, Eq, PartialEq)]
struct RootTaskDensityCensus {
    sampled_stream_tiles: u64,
    empty_stream_tiles: u64,
    sampled_hot_entries: u64,
    sampled_lane_slots: u64,
    sampled_zero_mask_probes: u64,
    sampled_selected_zero_entries: u64,
    sampled_even_row_entries: u64,
    sampled_zero_tasks: usize,
    task_density_ppm: [u64; 5],
    shift_quartiles: [u64; 4],
    active_tasks_per_tile: [u64; 6],
    current_barrier_iterations: u64,
    column_major_barrier_iterations: u64,
    ideal_barrier_iterations: u64,
}

pub(crate) struct ValidatedShape {
    active_a_cols: usize,
    output_coefficients: usize,
    positions_per_partial: usize,
    blocks_per_column: usize,
    full_blocks_per_column: usize,
    live_columns: usize,
}

fn packed_streams_per_command(num_positions: usize) -> usize {
    if num_positions <= FIVE_STREAM_MAX_POSITIONS {
        5
    } else {
        1
    }
}

fn root_census_tile_stride() -> Result<Option<usize>, AkitaError> {
    let Some(value) = std::env::var_os("AKITA_METAL_ROOT_CENSUS_TILE_STRIDE") else {
        return Ok(None);
    };
    let value = value.into_string().map_err(|_| {
        AkitaError::InvalidInput("root census tile stride is not valid UTF-8".into())
    })?;
    let stride = value.parse::<usize>().map_err(|_| {
        AkitaError::InvalidInput(format!("invalid root census tile stride {value:?}"))
    })?;
    if stride == 0 {
        return Err(AkitaError::InvalidInput(
            "root census tile stride must be nonzero".into(),
        ));
    }
    Ok(Some(stride))
}

fn density_ppm(hot_entries: u64, lane_slots: u64) -> u64 {
    if lane_slots == 0 {
        return 0;
    }
    ((u128::from(hot_entries) * 1_000_000) / u128::from(lane_slots)) as u64
}

fn density_quantile(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
    let index = (sorted.len() - 1) * numerator / denominator;
    sorted[index]
}

fn sampled_tiles(tiles_per_block: usize, tile_stride: usize) -> Vec<usize> {
    (0..tiles_per_block)
        .step_by(tile_stride)
        .enumerate()
        .map(|(stratum, first_tile)| {
            let stratum_len = (tiles_per_block - first_tile).min(tile_stride);
            let offset = stratum
                .wrapping_mul(2_654_435_761)
                .wrapping_add(1_013_904_223)
                % stratum_len;
            first_tile + offset
        })
        .collect()
}

fn grouped_barrier_iterations(
    selected_rows: &[u16],
    sample_count: usize,
    ordered_task: impl Fn(usize) -> usize,
) -> u64 {
    let task_count = selected_rows.len() / sample_count;
    let mut iterations = 0u64;
    for first_task in (0..task_count).step_by(FP128_D512_TASKS_PER_STREAM) {
        let final_task = (first_task + FP128_D512_TASKS_PER_STREAM).min(task_count);
        for sample in 0..sample_count {
            let maximum = (first_task..final_task)
                .map(|task| selected_rows[ordered_task(task) * sample_count + sample])
                .max()
                .unwrap_or_default();
            iterations += u64::from(maximum);
        }
    }
    iterations
}

fn ideal_barrier_iterations(
    selected_rows: &[u16],
    sample_count: usize,
    rows_per_tile: usize,
) -> u64 {
    let task_count = selected_rows.len() / sample_count;
    let mut iterations = 0u64;
    let mut histogram = vec![0usize; rows_per_tile + 1];
    for sample in 0..sample_count {
        histogram.fill(0);
        for task in 0..task_count {
            histogram[usize::from(selected_rows[task * sample_count + sample])] += 1;
        }
        let mut slots_remaining = 0usize;
        for selected_count in (0..=rows_per_tile).rev() {
            let mut tasks_at_count = histogram[selected_count];
            while tasks_at_count != 0 {
                if slots_remaining == 0 {
                    iterations += selected_count as u64;
                    slots_remaining = FP128_D512_TASKS_PER_STREAM;
                }
                let grouped = tasks_at_count.min(slots_remaining);
                tasks_at_count -= grouped;
                slots_remaining -= grouped;
            }
        }
    }
    iterations
}

fn sample_root_task_density<const D: usize>(
    source: PackedOneHotCommitView<'_>,
    plan: CommitInnerPlan,
    full_blocks_per_column: usize,
    tile_stride: usize,
) -> RootTaskDensityCensus {
    let live_columns = source.num_columns();
    let rows_per_position = D / source.onehot_k();
    let rows_per_tile = FP128_D512_POSITIONS_PER_TILE * rows_per_position;
    let rows_per_block = plan.num_positions_per_block * rows_per_position;
    let tiles_per_block = plan.num_positions_per_block / FP128_D512_POSITIONS_PER_TILE;
    let task_count = live_columns * full_blocks_per_column;
    let stream_count = task_count.div_ceil(FP128_D512_TASKS_PER_STREAM);
    let sampled_tiles = sampled_tiles(tiles_per_block, tile_stride);
    let sample_count = sampled_tiles.len();
    let mut selected_rows = vec![0u16; task_count * sample_count];
    let mut task_hot_entries = vec![0u64; task_count];
    let mut task_lane_slots = vec![0u64; task_count];
    let mut sampled_hot_entries = 0u64;
    let mut sampled_lane_slots = 0u64;
    let mut sampled_zero_mask_probes = 0u64;
    let mut sampled_selected_zero_entries = 0u64;
    let mut sampled_even_row_entries = 0u64;
    let mut shift_quartiles = [0u64; 4];
    for task in 0..task_count {
        let block = task / live_columns;
        let column = task % live_columns;
        for (sample, &tile) in sampled_tiles.iter().enumerate() {
            let first_row = block * rows_per_block + tile * rows_per_tile;
            let mut task_selected_rows = 0u16;
            for row in first_row..first_row + rows_per_tile {
                let lane = source.lanes()[row * live_columns + column];
                sampled_zero_mask_probes +=
                    u64::from(lane == 0 && source.zero_column_mask() & (1u64 << column) != 0);
                let selected = lane != 0 || source.commits_zero_at(row, column);
                task_hot_entries[task] += u64::from(selected);
                sampled_hot_entries += u64::from(selected);
                task_selected_rows += u16::from(selected);
                if selected {
                    sampled_selected_zero_entries += u64::from(lane == 0);
                    sampled_even_row_entries +=
                        u64::from((row - block * rows_per_block).is_multiple_of(2));
                    shift_quartiles[usize::from(lane) / 64] += 1;
                }
            }
            selected_rows[task * sample_count + sample] = task_selected_rows;
            task_lane_slots[task] += rows_per_tile as u64;
            sampled_lane_slots += rows_per_tile as u64;
        }
    }

    let mut empty_stream_tiles = 0u64;
    let mut active_tasks_per_tile = [0u64; 6];
    for first_task in (0..task_count).step_by(FP128_D512_TASKS_PER_STREAM) {
        let final_task = (first_task + FP128_D512_TASKS_PER_STREAM).min(task_count);
        for sample in 0..sample_count {
            let active_tasks = (first_task..final_task)
                .filter(|&task| selected_rows[task * sample_count + sample] != 0)
                .count();
            let bin = match active_tasks {
                0 => {
                    empty_stream_tiles += 1;
                    0
                }
                1..=4 => 1,
                5..=8 => 2,
                9..=16 => 3,
                17..=24 => 4,
                _ => 5,
            };
            active_tasks_per_tile[bin] += 1;
        }
    }
    let sampled_stream_tiles = (stream_count * sample_count) as u64;
    let current_barrier_iterations =
        grouped_barrier_iterations(&selected_rows, sample_count, |task| task);
    let column_major_barrier_iterations =
        grouped_barrier_iterations(&selected_rows, sample_count, |task| {
            let column = task / full_blocks_per_column;
            let block = task % full_blocks_per_column;
            block * live_columns + column
        });
    let ideal_barrier_iterations =
        ideal_barrier_iterations(&selected_rows, sample_count, rows_per_tile);

    let mut task_densities = task_hot_entries
        .iter()
        .zip(&task_lane_slots)
        .map(|(&hot_entries, &lane_slots)| density_ppm(hot_entries, lane_slots))
        .collect::<Vec<_>>();
    task_densities.sort_unstable();
    RootTaskDensityCensus {
        sampled_stream_tiles,
        empty_stream_tiles,
        sampled_hot_entries,
        sampled_lane_slots,
        sampled_zero_mask_probes,
        sampled_selected_zero_entries,
        sampled_even_row_entries,
        sampled_zero_tasks: task_hot_entries
            .iter()
            .filter(|&&hot_entries| hot_entries == 0)
            .count(),
        task_density_ppm: [
            density_quantile(&task_densities, 10, 100),
            density_quantile(&task_densities, 25, 100),
            density_quantile(&task_densities, 50, 100),
            density_quantile(&task_densities, 75, 100),
            density_quantile(&task_densities, 90, 100),
        ],
        shift_quartiles,
        active_tasks_per_tile,
        current_barrier_iterations,
        column_major_barrier_iterations,
        ideal_barrier_iterations,
    }
}

pub(crate) fn validate_shape<const D: usize>(
    source: PackedOneHotCommitView<'_>,
    plan: CommitInnerPlan,
) -> Result<ValidatedShape, AkitaError> {
    let supported_source = matches!(
        (source.onehot_k(), source.column_capacity()),
        (16, K16_COLUMN_CAPACITY) | (256, K256_COLUMN_CAPACITY)
    );
    if D != RING_D
        || !supported_source
        || !(1..=source.column_capacity()).contains(&source.num_columns())
        || plan.n_a != INNER_RANK
        || plan.num_digits_inner != 1
        || plan.num_positions_per_block == 0
        || !plan
            .num_positions_per_block
            .is_multiple_of(FP128_D512_POSITION_PARTIALS)
    {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 Metal D512 commit requires K16/capacity-64 or K256/capacity-32, rank 1, and integral position partials"
                .into(),
        )
        .into_akita());
    }
    let positions_per_partial = plan.num_positions_per_block / FP128_D512_POSITION_PARTIALS;
    if !positions_per_partial.is_multiple_of(4) {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 partials must contain whole four-position matrix tiles".into(),
        )
        .into_akita());
    }
    let rows_per_position = D / source.onehot_k();
    let rows_per_block = plan
        .num_positions_per_block
        .checked_mul(rows_per_position)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 rows per block").into_akita())?;
    if !source.num_rows().is_multiple_of(rows_per_block) {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 panels require integral trace blocks".into(),
        )
        .into_akita());
    }
    let blocks_per_column = source.num_rows() / rows_per_block;
    if !blocks_per_column.is_power_of_two() || blocks_per_column > 512 {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 panels require at most 512 power-of-two blocks per column".into(),
        )
        .into_akita());
    }
    let full_blocks_per_column = source
        .zero_suffix_start()
        .div_ceil(rows_per_block)
        .max(1)
        .min(blocks_per_column);
    let output_coefficients = source
        .column_capacity()
        .checked_mul(blocks_per_column)
        .and_then(|count| count.checked_mul(INNER_RANK))
        .and_then(|count| count.checked_mul(D))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 output").into_akita())?;
    Ok(ValidatedShape {
        active_a_cols: plan.num_positions_per_block,
        output_coefficients,
        positions_per_partial,
        blocks_per_column,
        full_blocks_per_column,
        live_columns: source.num_columns(),
    })
}

pub(crate) fn commit_validated<const D: usize>(
    backend: &MetalBackend,
    prepared: &MetalPreparedSetup,
    runtime: &MetalRuntime,
    source: PackedOneHotCommitView<'_>,
    plan: CommitInnerPlan,
    shape: ValidatedShape,
) -> Result<CommitInnerWitness<F>, AkitaError> {
    if let Some(tile_stride) = root_census_tile_stride()? {
        let census_start = Instant::now();
        let census =
            sample_root_task_density::<D>(source, plan, shape.full_blocks_per_column, tile_stride);
        tracing::debug!(
            census_s = census_start.elapsed().as_secs_f64(),
            tile_stride,
            sampled_stream_tiles = census.sampled_stream_tiles,
            empty_stream_tiles = census.empty_stream_tiles,
            empty_stream_tile_fraction =
                census.empty_stream_tiles as f64 / census.sampled_stream_tiles as f64,
            sampled_hot_entries = census.sampled_hot_entries,
            sampled_lane_slots = census.sampled_lane_slots,
            zero_column_mask = source.zero_column_mask(),
            zero_mask_columns = source.zero_column_mask().count_ones(),
            sampled_zero_mask_probes = census.sampled_zero_mask_probes,
            sampled_zero_mask_probe_fraction =
                census.sampled_zero_mask_probes as f64 / census.sampled_lane_slots as f64,
            sampled_zero_mask_success_fraction = census.sampled_selected_zero_entries as f64
                / census.sampled_zero_mask_probes as f64,
            sampled_selected_zero_entries = census.sampled_selected_zero_entries,
            sampled_selected_zero_fraction =
                census.sampled_selected_zero_entries as f64 / census.sampled_hot_entries as f64,
            sampled_even_row_entries = census.sampled_even_row_entries,
            sampled_even_row_fraction =
                census.sampled_even_row_entries as f64 / census.sampled_hot_entries as f64,
            shift_0_63 = census.shift_quartiles[0],
            shift_64_127 = census.shift_quartiles[1],
            shift_128_191 = census.shift_quartiles[2],
            shift_192_255 = census.shift_quartiles[3],
            sampled_density = census.sampled_hot_entries as f64 / census.sampled_lane_slots as f64,
            exact_hot_entries = source.hot_entries(),
            sampled_zero_tasks = census.sampled_zero_tasks,
            task_density_p10 = census.task_density_ppm[0] as f64 / 1_000_000.0,
            task_density_p25 = census.task_density_ppm[1] as f64 / 1_000_000.0,
            task_density_p50 = census.task_density_ppm[2] as f64 / 1_000_000.0,
            task_density_p75 = census.task_density_ppm[3] as f64 / 1_000_000.0,
            task_density_p90 = census.task_density_ppm[4] as f64 / 1_000_000.0,
            active_tasks_0 = census.active_tasks_per_tile[0],
            active_tasks_1_4 = census.active_tasks_per_tile[1],
            active_tasks_5_8 = census.active_tasks_per_tile[2],
            active_tasks_9_16 = census.active_tasks_per_tile[3],
            active_tasks_17_24 = census.active_tasks_per_tile[4],
            active_tasks_25_32 = census.active_tasks_per_tile[5],
            current_barrier_iterations = census.current_barrier_iterations,
            column_major_barrier_iterations = census.column_major_barrier_iterations,
            column_major_barrier_reduction = 1.0
                - census.column_major_barrier_iterations as f64
                    / census.current_barrier_iterations as f64,
            ideal_barrier_iterations = census.ideal_barrier_iterations,
            ideal_barrier_reduction = 1.0
                - census.ideal_barrier_iterations as f64 / census.current_barrier_iterations as f64,
            "sampled packed Metal root task density"
        );
    }
    let total_start = Instant::now();
    let matrix = tracing::info_span!("packed_metal_matrix_prepare")
        .in_scope(|| prepared.matrix(runtime, D, plan.n_a, shape.active_a_cols))?;
    let work_units = shape
        .live_columns
        .checked_mul(shape.full_blocks_per_column)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 work units").into_akita())?;
    let params = PackedOneHotCommitParams {
        num_rows: to_u64(source.num_rows(), "fp128 D512 row count")?,
        num_columns: to_u64(shape.live_columns, "fp128 D512 column count")?,
        lane_stride: to_u64(shape.live_columns, "fp128 D512 lane stride")?,
        column_capacity: to_u64(source.column_capacity(), "fp128 D512 column capacity")?,
        onehot_k: to_u64(source.onehot_k(), "fp128 D512 one-hot K")?,
        ring_d: D as u64,
        n_a: INNER_RANK as u64,
        positions_per_block: to_u64(plan.num_positions_per_block, "fp128 D512 block positions")?,
        num_digits_inner: 1,
        blocks_per_column: to_u64(shape.blocks_per_column, "fp128 D512 block count")?,
        full_blocks_per_column: to_u64(
            shape.full_blocks_per_column,
            "fp128 D512 full block count",
        )?,
        boundary_columns: 0,
        num_blocks: to_u64(work_units, "fp128 D512 work units")?,
        task_offset: 0,
        dispatch_tasks: to_u64(work_units, "fp128 D512 dispatch work units")?,
        lane_row_offset: 0,
        output_coefficients: to_u64(shape.output_coefficients, "fp128 D512 output")?,
        columns_per_threadgroup: 1,
        position_partials_per_block: FP128_D512_POSITION_PARTIALS as u64,
        positions_per_partial: to_u64(
            shape.positions_per_partial,
            "fp128 D512 positions per partial",
        )?,
        log_ring_d: 9,
        zero_column_mask: source.zero_column_mask(),
    };
    let outcome = tracing::info_span!("packed_metal_dispatch")
        .in_scope(|| {
            runtime.dispatch_packed_onehot(
                matrix.buffer.as_ref(),
                source.lanes(),
                source.active_zero_rows(),
                params,
                packed_streams_per_command(shape.active_a_cols),
            )
        })
        .map_err(MetalCommitError::into_akita)?;

    let _reconstruction_span = tracing::info_span!("packed_metal_reconstruction").entered();
    let reconstruction_start = Instant::now();
    let coefficients = outcome
        .coefficients
        .into_iter()
        .enumerate()
        .map(|(index, coefficient)| F::from_device(coefficient, index))
        .collect::<Result<Vec<_>, _>>()
        .map_err(MetalCommitError::into_akita)?;
    let witness = CommitInnerWitness {
        inner_rows: RingVec::from_coeffs_with_ring_dim(coefficients, D)?,
    };
    let output_reconstruction_time = reconstruction_start.elapsed();

    let field_additions = source
        .hot_entries()
        .checked_mul(INNER_RANK)
        .and_then(|count| count.checked_mul(D))
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 additions").into_akita())?;
    let gathered_matrix_bytes = field_additions
        .checked_mul(size_of::<Fp128Limbs>())
        .ok_or_else(|| MetalCommitError::ShapeOverflow("fp128 D512 gathered bytes").into_akita())?;
    tracing::debug!(
        total_s = total_start.elapsed().as_secs_f64(),
        matrix_prepare_s = matrix.prepare_time.as_secs_f64(),
        matrix_cache_hit = matrix.cache_hit,
        buffer_setup_s = outcome.timings.buffer_setup.as_secs_f64(),
        command_wall_s = outcome.timings.command_wall.as_secs_f64(),
        gpu_s = outcome.timings.gpu.map(|duration| duration.as_secs_f64()),
        panel_gpu_active_s = outcome
            .panel_gpu_active
            .map(|duration| duration.as_secs_f64()),
        panel_gpu_span_s = outcome
            .panel_gpu_span
            .map(|duration| duration.as_secs_f64()),
        reduction_gpu_s = outcome.reduction_gpu.map(|duration| duration.as_secs_f64()),
        command_buffers = outcome.command_buffers,
        matrix_block_streams = outcome.matrix_block_streams,
        input_zero_copy = outcome.input_zero_copy,
        readback_copy_s = outcome.timings.readback_copy.as_secs_f64(),
        reconstruction_s = output_reconstruction_time.as_secs_f64(),
        lane_bytes = source.lanes().len(),
        hot_entries = source.hot_entries(),
        zero_suffix_start = source.zero_suffix_start(),
        blocks_per_column = shape.blocks_per_column,
        full_blocks_per_column = shape.full_blocks_per_column,
        output_coefficients = shape.output_coefficients,
        "completed packed Metal trace commitment"
    );
    backend
        .record_commit_metrics(MetalCommitMetrics {
            kernel: MetalOneHotKernel::PackedFp128D512Panels,
            blocks_per_threadgroup: outcome.blocks_per_threadgroup,
            num_sources: 1,
            hot_entries: source.hot_entries(),
            field_additions: to_u64(field_additions, "fp128 D512 additions")?,
            gathered_matrix_bytes: to_u64(gathered_matrix_bytes, "fp128 D512 gathered bytes")?,
            output_bytes: shape
                .output_coefficients
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("fp128 D512 output bytes").into_akita()
                })?,
            scratch_bytes: outcome.scratch_bytes,
            matrix_bytes: matrix.bytes,
            matrix_cache_hit: matrix.cache_hit,
            matrix_prepare_time: matrix.prepare_time,
            index_pack_time: Duration::ZERO,
            buffer_setup_time: outcome.timings.buffer_setup,
            command_wall_time: outcome.timings.command_wall,
            gpu_time: outcome.timings.gpu,
            panel_gpu_active_time: outcome.panel_gpu_active,
            panel_gpu_span: outcome.panel_gpu_span,
            reduction_gpu_time: outcome.reduction_gpu,
            command_buffers: outcome.command_buffers,
            matrix_block_streams: outcome.matrix_block_streams,
            readback_copy_time: outcome.timings.readback_copy,
            output_reconstruction_time,
            total_time: total_start.elapsed(),
        })
        .map_err(MetalCommitError::into_akita)?;
    Ok(witness)
}

#[cfg(test)]
mod tests {
    use akita_prover::compute::RootCommitKernel;
    use akita_prover::{
        AkitaProverSetup, ComputeBackendSetup, CpuBackend, OneHotPoly, RootCommitSource,
    };
    use akita_types::SetupMatrixCapacity;

    use super::*;
    use crate::{MetalExecutionPolicy, PackedOneHotCommitView};

    fn assert_panel_parity(
        onehot_k: usize,
        rows: usize,
        capacity: usize,
        zero_suffix_start: Option<usize>,
    ) {
        const LIVE_COLUMNS: usize = 5;
        const POSITIONS_PER_BLOCK: usize = 64;
        const ZERO_COLUMN_MASK: u64 = 0b0_1010;

        let lanes = (0..rows * LIVE_COLUMNS)
            .map(|index| {
                let row = index / LIVE_COLUMNS;
                if row < zero_suffix_start.unwrap_or(rows) && index.is_multiple_of(97) {
                    ((index.wrapping_mul(73) % (onehot_k - 1)) + 1) as u8
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        let mut active_zero_rows = vec![0u64; rows / u64::BITS as usize];
        for row in (0..zero_suffix_start.unwrap_or(rows)).step_by(11) {
            active_zero_rows[row / u64::BITS as usize] |= 1u64 << (row % u64::BITS as usize);
        }
        let indices: Vec<Option<u8>> = (0..capacity)
            .flat_map(|column| {
                let lanes = &lanes;
                let active_zero_rows = &active_zero_rows;
                (0..rows).map(move |row| {
                    if column >= LIVE_COLUMNS {
                        None
                    } else {
                        let lane = lanes[row * LIVE_COLUMNS + column];
                        let committed_zero = ZERO_COLUMN_MASK & (1u64 << column) != 0
                            && active_zero_rows[row / u64::BITS as usize]
                                & (1u64 << (row % u64::BITS as usize))
                                != 0;
                        (lane != 0 || committed_zero).then_some(lane)
                    }
                })
            })
            .collect();
        let hot_entries = indices.iter().filter(|entry| entry.is_some()).count();
        let generic = OneHotPoly::<F, u8>::new(onehot_k, indices).unwrap();
        let plan = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: POSITIONS_PER_BLOCK,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };
        let num_vars = (rows * capacity * onehot_k).trailing_zeros() as usize;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            num_vars,
            1,
            SetupMatrixCapacity {
                num_field_elements: POSITIONS_PER_BLOCK * RING_D,
            },
        )
        .unwrap();

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let cpu_view =
            <OneHotPoly<F, u8> as RootCommitSource<F, RING_D>>::commit_view(&generic).unwrap();
        let cpu_witness = cpu
            .commit_inner_group(&cpu_prepared, vec![cpu_view], plan)
            .unwrap()
            .remove(0);

        let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        let source = if let (256, Some(zero_suffix_start)) = (onehot_k, zero_suffix_start) {
            PackedOneHotCommitView::new_k256_with_precomputed_metrics(
                capacity,
                LIVE_COLUMNS,
                &lanes,
                &active_zero_rows,
                ZERO_COLUMN_MASK,
                hot_entries,
                zero_suffix_start,
            )
            .unwrap()
        } else {
            PackedOneHotCommitView::new_with_active_zero_rows(
                onehot_k,
                capacity,
                LIVE_COLUMNS,
                &lanes,
                &active_zero_rows,
                ZERO_COLUMN_MASK,
            )
            .unwrap()
        };
        let metal_witness = metal
            .commit_packed_onehot::<RING_D>(&metal_prepared, source, plan)
            .unwrap();
        assert_eq!(cpu_witness.inner_rows, metal_witness.inner_rows);

        let second = metal
            .commit_packed_onehot::<RING_D>(&metal_prepared, source, plan)
            .unwrap();
        assert_eq!(cpu_witness.inner_rows, second.inner_rows);
        let metrics = metal.last_commit_metrics().unwrap().unwrap();
        assert_eq!(metrics.kernel, MetalOneHotKernel::PackedFp128D512Panels);
        assert!(metrics.matrix_cache_hit);
    }

    #[test]
    fn parity_d512_k256_panels() {
        assert_panel_parity(256, 4096, 32, None);
    }

    #[test]
    fn parity_d512_k256_panels_skip_zero_suffix() {
        assert_panel_parity(256, 4096, 32, Some(2048));
    }

    #[test]
    fn parity_d512_k16_panels() {
        assert_panel_parity(16, 2048, 64, None);
    }

    #[test]
    fn root_task_density_census_counts_selected_zero_and_empty_tiles() {
        const ROWS: usize = 4096;
        const LIVE_COLUMNS: usize = 5;
        const ROWS_PER_BLOCK: usize = 128;
        let mut lanes = vec![0u8; ROWS * LIVE_COLUMNS];
        let mut active_zero_rows = vec![0u64; ROWS.div_ceil(u64::BITS as usize)];
        for block in 0..ROWS / ROWS_PER_BLOCK {
            let row = block * ROWS_PER_BLOCK;
            active_zero_rows[row / u64::BITS as usize] |= 1u64 << (row % u64::BITS as usize);
            for column in 1..LIVE_COLUMNS {
                lanes[row * LIVE_COLUMNS + column] = column as u8;
            }
        }
        let source = PackedOneHotCommitView::new_with_active_zero_rows(
            256,
            32,
            LIVE_COLUMNS,
            &lanes,
            &active_zero_rows,
            1,
        )
        .unwrap();
        let plan = CommitInnerPlan {
            n_a: 1,
            num_positions_per_block: 64,
            num_digits_inner: 1,
            log_basis_inner: 8,
        };

        let census = sample_root_task_density::<RING_D>(source, plan, ROWS / ROWS_PER_BLOCK, 1);

        assert_eq!(source.hot_entries(), 160);
        assert_eq!(census.sampled_stream_tiles, 80);
        assert_eq!(census.empty_stream_tiles, 75);
        assert_eq!(census.sampled_hot_entries, 160);
        assert_eq!(census.sampled_lane_slots, 20_480);
        assert_eq!(census.sampled_zero_mask_probes, 4096);
        assert_eq!(census.sampled_selected_zero_entries, 32);
        assert_eq!(census.sampled_even_row_entries, 160);
        assert_eq!(census.sampled_zero_tasks, 0);
        assert_eq!(census.task_density_ppm, [7812; 5]);
        assert_eq!(census.shift_quartiles, [160, 0, 0, 0]);
        assert_eq!(census.active_tasks_per_tile, [75, 0, 0, 0, 0, 5]);
        assert_eq!(census.current_barrier_iterations, 5);
        assert_eq!(census.column_major_barrier_iterations, 5);
        assert_eq!(census.ideal_barrier_iterations, 5);
    }
}
