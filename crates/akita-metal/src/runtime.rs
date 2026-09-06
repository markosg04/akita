use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Range;
use std::time::{Duration, Instant};

use akita_algebra::{CrtCapacity, GarnerData, NttPrime};
#[cfg(test)]
use jolt_field::Zero;
use jolt_field::{One, Ring};
use metal::objc::rc::autoreleasepool;
use metal::objc::{runtime::Sel, Message};
use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use crate::error::metal_status::CommandStatus;
use crate::field::{Fp128Limbs, F};
use crate::MetalCommitError;

const DIRECT_KERNEL_NAME: &str = "akita_onehot_commit_gather";
const BLOCK_BATCHED_KERNEL_NAME: &str = "akita_onehot_commit_block_batched";
const PACKED_FP128_D512_PANELS_KERNEL_NAME: &str = "akita_packed_onehot_commit_fp128_d512_panels";
const PACKED_PARTIAL_REDUCTION_KERNEL_NAME: &str = "akita_packed_onehot_reduce_partials";
const PACKED_FP128_D128_RANK3_KERNEL_NAME: &str = "akita_packed_onehot_commit_fp128_d128_rank3";
const FP128_D128_DECOMPOSE_FOLD_KERNEL_NAME: &str = "akita_fp128_d128_decompose_fold";
const FP128_D128_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME: &str =
    "akita_fp128_d128_subring64_decompose_fold";
const FP128_D64_DIGIT_ROWS_PARTIALS_KERNEL_NAME: &str = "akita_fp128_d64_digit_rows_partials";
const FP128_D64_DIGIT_ROWS_REDUCE_KERNEL_NAME: &str = "akita_fp128_d64_digit_rows_reduce";
const FP128_I8_COEFFICIENT_PACKING_KERNEL_NAME: &str = "akita_fp128_i8_coefficient_packing";
const FP128_PACKED_ONEHOT_COEFFICIENT_PACKING_PARTIALS_KERNEL_NAME: &str =
    "akita_fp128_packed_onehot_coefficient_packing_partials";
const FP128_PACKED_ONEHOT_COEFFICIENT_PACKING_REDUCE_KERNEL_NAME: &str =
    "akita_fp128_packed_onehot_coefficient_packing_reduce";
const FP128_D512_DECOMPOSE_FOLD_KERNEL_NAME: &str = "akita_fp128_d512_decompose_fold";
const FP128_D512_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME: &str =
    "akita_fp128_d512_subring64_decompose_fold";
const FP128_D512_BUILD_FOLD_INDEX_KERNEL_NAME: &str = "akita_fp128_d512_build_fold_index";
const FP128_D512_BUILD_COEFFICIENT_PACKING_INDEX_KERNEL_NAME: &str =
    "akita_fp128_d512_build_coefficient_packing_index";
const FP128_D512_INDEXED_COEFFICIENT_PACKING_KERNEL_NAME: &str =
    "akita_fp128_d512_indexed_coefficient_packing_partials";
const FP128_D512_INDEXED_COEFFICIENT_PACKING_REDUCE_KERNEL_NAME: &str =
    "akita_fp128_d512_indexed_coefficient_packing_reduce";
const FP128_D512_INDEXED_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME: &str =
    "akita_fp128_d512_indexed_subring64_decompose_fold";
const FP128_D512_FUSED_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME: &str =
    "akita_fp128_d512_fused_subring64_decompose_fold";
const FP128_D512_LINEAR_RELATION_PARTIALS_KERNEL_NAME: &str =
    "akita_fp128_d512_linear_relation_partials";
const FP128_D512_LINEAR_RELATION_REDUCE_KERNEL_NAME: &str =
    "akita_fp128_d512_linear_relation_reduce";
const FP128_D512_LINEAR_RELATION_RECONSTRUCT_KERNEL_NAME: &str =
    "akita_fp128_d512_linear_relation_reconstruct";
const FP128_RECURSIVE_COMMIT_MATRIX_NTT_KERNEL_NAME: &str =
    "akita_fp128_recursive_commit_matrix_ntt";
const FP128_RECURSIVE_COMMIT_MATVEC_KERNEL_NAME: &str = "akita_fp128_recursive_commit_matvec";
const FP128_RECURSIVE_COMMIT_RECONSTRUCT_KERNEL_NAME: &str =
    "akita_fp128_recursive_commit_reconstruct";
const FP128_DIRECT_RANGE_INITIAL_KERNEL_NAME: &str = "akita_fp128_direct_range_initial_partials";
const FP128_DIRECT_RANGE_COMPACT_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_range_compact_fold_partials";
const FP128_DIRECT_RANGE_FIELD_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_range_field_fold_partials";
const FP128_DIRECT_RANGE_REDUCE_KERNEL_NAME: &str = "akita_fp128_direct_range_reduce";
const FP128_DIRECT_RANGE_FINALIZE_KERNEL_NAME: &str = "akita_fp128_direct_range_finalize";
const FP128_BLAKE2B_SUMCHECK_CHALLENGE_KERNEL_NAME: &str = "akita_fp128_blake2b_sumcheck_challenge";
const FP128_BLAKE2B_RELATION_SUMCHECK_ROUND_KERNEL_NAME: &str =
    "akita_fp128_blake2b_relation_sumcheck_round";
const FP128_DIRECT_RELATION_INITIAL_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_initial_partials";
const FP128_DIRECT_RELATION_COMPACT_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_compact_fold_partials";
const FP128_DIRECT_RELATION_FIELD_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_field_fold_partials";
const FP128_DIRECT_RELATION_ADDITIONAL_COMPACT_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_additional_compact_partials";
const FP128_DIRECT_RELATION_ADDITIONAL_FIELD_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_additional_field_partials";
const FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_two_round_prefix_partials";
const FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_REDUCE_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_two_round_prefix_reduce";
const FP128_DIRECT_RELATION_LINEAR_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_linear_fold";
const FP128_DIRECT_RELATION_ALPHA_FOLD_KERNEL_NAME: &str = "akita_fp128_direct_relation_alpha_fold";
const FP128_DIRECT_RELATION_SCALAR_ADVANCE_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_scalar_advance";
const FP128_DIRECT_RELATION_ADDITIONAL_FOLD_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_additional_fold";
const FP128_DIRECT_RELATION_SETUP_SOURCE_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_setup_source";
const FP128_DIRECT_RELATION_SPARSE_SOURCE_KERNEL_NAME: &str =
    "akita_fp128_direct_relation_sparse_source";
const KERNEL_SOURCE: &str = include_str!("kernels/onehot.metal");
const FP128_D512_THREADS: usize = 1_024;
const PACKED_ONEHOT_BUFFER_ALIGNMENT: usize = 16 * 1024;
pub(crate) const FP128_D512_TASKS_PER_STREAM: usize = 32;
pub(crate) const FP128_D512_POSITION_PARTIALS: usize = 16;
pub(crate) const FP128_D128_RANK3_TASKS_PER_STREAM: usize = 64;
pub(crate) const FP128_D128_RANK3_POSITION_PARTIAL_ALIGNMENT: usize = 16;
const FP128_D128_RANK3_THREADGROUP_BYTES: usize = 5 * 1_024 * size_of::<u32>();
const FP128_D128_RANK3_RING_D: u64 = 128;
const FP128_D128_RANK3_INNER_RANK: u64 = 3;
const FP128_D512_TILE_FIELD_ELEMENTS: usize = 2_048;
const FP128_D512_THREADGROUP_BYTES: usize =
    FP128_D512_TILE_FIELD_ELEMENTS * size_of::<Fp128Limbs>();
const FP128_D512_COEFFICIENT_BANDS: usize = 2;
const FP128_D64_DIGIT_ROWS_THREADS: usize = 256;
const FP128_D64_DIGIT_ROWS_PARTIAL_THREADS: usize = 64;
pub(crate) const FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL: usize = 128;
const FP128_COEFFICIENT_PACKING_THREADS: usize = 256;
pub(crate) const FP128_D512_PACKING_INDEX_TILE_POSITIONS: usize = 256;
pub(crate) const FP128_D512_PACKING_INDEX_BUCKET_OFFSETS: usize = 33;
const FP128_D512_PACKING_TILES_PER_CHUNK: usize = 32;
const FP128_PACKED_COEFFICIENT_PACKING_PARTIAL_THREADS: usize = 256;
const FP128_PACKED_COEFFICIENT_PACKING_REDUCE_THREADS: usize = 256;
pub(crate) const FP128_D512_FOLD_INDEX_TILE_TASKS: usize = 256;
pub(crate) const FP128_D512_FOLD_INDEX_COUNT_BUCKETS: usize = 8;
const FP128_D512_SUBRING_DIMENSION: usize = 64;
pub(crate) const FP128_PACKED_COEFFICIENT_PACKING_ROWS_PER_PARTIAL: usize =
    i32::MAX as usize / u16::MAX as usize;
const FP128_DIRECT_RANGE_THREADS: usize = 256;
const FP128_DIRECT_RANGE_MAX_WORKGROUPS: usize = 4_096;
const FP128_DIRECT_RANGE_STORED_COEFFICIENTS: usize = 4;
const FP128_DIRECT_RELATION_STORED_COEFFICIENTS: usize = 4;
const FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS: usize = 16;
const FP128_DIRECT_RELATION_PREFIX_LANES_PER_THREAD: usize = 2;
const FP128_D512_LINEAR_RELATION_THREADS: usize = 512;
const FP128_D512_LINEAR_RELATION_NTT_SIZE: usize = 1_024;
const FP128_D512_LINEAR_RELATION_COLUMNS_PER_TILE: usize = 64;
pub(crate) const FP128_D512_LINEAR_RELATION_NUM_PRIMES: usize = 6;
const FP128_D512_LINEAR_RELATION_RAW_PRIMES: [i32; FP128_D512_LINEAR_RELATION_NUM_PRIMES] = [
    1_073_692_673,
    1_073_668_097,
    1_073_707_009,
    1_073_738_753,
    1_073_732_609,
    1_073_698_817,
];
const FP128_RECURSIVE_COMMIT_THREADS: usize = 512;
const FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS: usize = 256;
const FP128_RECURSIVE_COMMIT_BLOCKS_PER_GROUP: usize = 16;
const FP128_RECURSIVE_COMMIT_MAX_ROWS: usize = 8;

fn pack_biased_subring64_challenges(dense_challenges: &[i8]) -> Result<Vec<u32>, MetalCommitError> {
    if !dense_challenges
        .len()
        .is_multiple_of(FP128_D512_SUBRING_DIMENSION)
        || dense_challenges
            .iter()
            .any(|&coefficient| !(-2..=2).contains(&coefficient))
    {
        return Err(MetalCommitError::UnsupportedShape(
            "indexed D512 fold requires D64 challenge coefficients in [-2, 2]".into(),
        ));
    }

    let mut packed = Vec::with_capacity(dense_challenges.len());
    for challenge in dense_challenges.chunks_exact(FP128_D512_SUBRING_DIMENSION) {
        for source_phase in 0..8 {
            for destination_quad in 0..8 {
                let start = (8 * destination_quad + FP128_D512_SUBRING_DIMENSION - source_phase)
                    % FP128_D512_SUBRING_DIMENSION;
                let word = (0..8).fold(0u32, |word, offset| {
                    let position = (start + offset) % FP128_D512_SUBRING_DIMENSION;
                    let biased = u32::from((challenge[position] + 2) as u8);
                    word | (biased << (4 * offset))
                });
                packed.push(word);
            }
        }
    }
    Ok(packed)
}

fn validate_packed_fold_index_geometry(
    params: PackedFoldIndexParams,
    lane_count: usize,
) -> Result<(), MetalCommitError> {
    let expected_lanes = params
        .num_rows
        .checked_mul(params.lane_stride)
        .ok_or(MetalCommitError::ShapeOverflow("fold-index lanes"))?;
    let expected_tasks = params
        .blocks_per_column
        .checked_mul(params.num_columns)
        .and_then(|count| count.checked_mul(2))
        .ok_or(MetalCommitError::ShapeOverflow(
            "fold-index tasks per position",
        ))?;
    let expected_tiles = expected_tasks.div_ceil(FP128_D512_FOLD_INDEX_TILE_TASKS as u64);
    let expected_records = params
        .num_positions
        .checked_mul(expected_tiles)
        .and_then(|count| count.checked_mul(FP128_D512_FOLD_INDEX_TILE_TASKS as u64))
        .ok_or(MetalCommitError::ShapeOverflow("fold-index records"))?;
    let expected_counts = params
        .num_positions
        .checked_mul(expected_tiles)
        .and_then(|count| count.checked_mul(FP128_D512_FOLD_INDEX_COUNT_BUCKETS as u64))
        .ok_or(MetalCommitError::ShapeOverflow("fold-index counts"))?;
    if params.num_rows == 0
        || params.num_columns == 0
        || params.num_columns > params.lane_stride
        || params.num_positions == 0
        || params.position_start != 0
        || params.tasks_per_position != expected_tasks
        || params.tiles_per_position != expected_tiles
        || params.record_slots != expected_records
        || params.count_entries != expected_counts
        || u64::try_from(lane_count).ok() != Some(expected_lanes)
    {
        return Err(MetalCommitError::UnsupportedShape(
            "fp128 D512 packed fold-index geometry is unsupported".into(),
        ));
    }
    Ok(())
}

/// Metal implementation selected for a one-hot inner commitment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MetalOneHotKernel {
    /// One output coefficient per thread with direct matrix gathers.
    #[default]
    DirectGather,
    /// Several source blocks per threadgroup with cache-local global reads.
    BlockBatched,
    /// Exact fp128 D512 panels with one block-column task per SIMDgroup.
    PackedFp128D512Panels,
    /// Exact fp128 D128 rank-3 per-element tiles with two tasks per SIMDgroup.
    PackedFp128D128Rank3,
}

/// Stable Metal device properties relevant to commitment scheduling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceCapabilities {
    /// Registry-reported device name.
    pub name: String,
    /// Maximum allocation length.
    pub max_buffer_length: u64,
    /// Recommended resident working set.
    pub recommended_max_working_set_size: u64,
    /// Maximum threadgroup-local memory.
    pub max_threadgroup_memory_length: u64,
    /// SIMD execution width of the baseline one-hot pipeline.
    pub thread_execution_width: usize,
    /// Maximum threads in one baseline one-hot threadgroup.
    pub max_total_threads_per_threadgroup: usize,
    /// Static threadgroup memory reserved by the baseline pipeline.
    pub static_threadgroup_memory_length: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct OneHotCommitParams {
    pub(crate) num_sources: u64,
    pub(crate) chunks_per_source: u64,
    pub(crate) onehot_k: u64,
    pub(crate) ring_d: u64,
    pub(crate) n_a: u64,
    pub(crate) positions_per_block: u64,
    pub(crate) num_digits_inner: u64,
    pub(crate) num_blocks: u64,
    pub(crate) total_field_elements: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) blocks_per_threadgroup: u64,
    pub(crate) log_onehot_k: u64,
    pub(crate) log_ring_d: u64,
}

const _: [(); 104] = [(); size_of::<OneHotCommitParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedOneHotCommitParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) lane_stride: u64,
    pub(crate) column_capacity: u64,
    pub(crate) onehot_k: u64,
    pub(crate) ring_d: u64,
    pub(crate) n_a: u64,
    pub(crate) positions_per_block: u64,
    pub(crate) num_digits_inner: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) full_blocks_per_column: u64,
    pub(crate) boundary_columns: u64,
    pub(crate) num_blocks: u64,
    pub(crate) task_offset: u64,
    pub(crate) dispatch_tasks: u64,
    pub(crate) lane_row_offset: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) columns_per_threadgroup: u64,
    pub(crate) position_partials_per_block: u64,
    pub(crate) positions_per_partial: u64,
    pub(crate) log_ring_d: u64,
    pub(crate) zero_column_mask: u64,
}

const _: [(); 176] = [(); size_of::<PackedOneHotCommitParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DigitRowsParams {
    pub(crate) num_vectors: u64,
    pub(crate) num_rows: u64,
    pub(crate) num_cols: u64,
    pub(crate) ring_d: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) columns_per_partial: u64,
    pub(crate) column_partials: u64,
    pub(crate) retain_quotients: u64,
}

const _: [(); 64] = [(); size_of::<DigitRowsParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct I8CoefficientPackingParams {
    pub(crate) num_sources: u64,
    pub(crate) source_coefficients: u64,
    pub(crate) live_coefficients: u64,
    pub(crate) num_live_positions: u64,
    pub(crate) positions_per_block: u64,
    pub(crate) num_blocks: u64,
    pub(crate) ring_d: u64,
    pub(crate) stride: u64,
    pub(crate) subring_dimension: u64,
    pub(crate) output_coefficients: u64,
}

const _: [(); 80] = [(); size_of::<I8CoefficientPackingParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedOneHotCoefficientPackingParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) column_capacity: u64,
    pub(crate) onehot_k: u64,
    pub(crate) ring_d: u64,
    pub(crate) positions_per_block: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) rows_per_block: u64,
    pub(crate) rows_per_partial: u64,
    pub(crate) row_partials_per_block: u64,
    pub(crate) num_blocks: u64,
    pub(crate) stride: u64,
    pub(crate) subring_dimension: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) partial_coefficients: u64,
    pub(crate) zero_column_mask: u64,
}

const _: [(); 128] = [(); size_of::<PackedOneHotCoefficientPackingParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedDecomposeFoldParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) lane_stride: u64,
    pub(crate) num_positions: u64,
    pub(crate) position_start: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) challenge_weight: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) zero_column_mask: u64,
}

const _: [(); 72] = [(); size_of::<PackedDecomposeFoldParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedFoldIndexParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) lane_stride: u64,
    pub(crate) num_positions: u64,
    pub(crate) position_start: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) tasks_per_position: u64,
    pub(crate) tiles_per_position: u64,
    pub(crate) record_slots: u64,
    pub(crate) count_entries: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) fold_digits: u64,
    pub(crate) fold_log_basis: u64,
}

const _: [(); 104] = [(); size_of::<PackedFoldIndexParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedCoefficientPackingIndexParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) lane_stride: u64,
    pub(crate) num_positions: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) position_tiles: u64,
    pub(crate) record_slots: u64,
    pub(crate) offset_entries: u64,
}

const _: [(); 64] = [(); size_of::<PackedCoefficientPackingIndexParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct D512LinearRelationParams {
    pub(crate) num_columns: u64,
    pub(crate) columns_per_tile: u64,
    pub(crate) num_tiles: u64,
    pub(crate) num_primes: u64,
    pub(crate) ntt_size: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) rhs_abs_bound: u64,
}

const _: [(); 56] = [(); size_of::<D512LinearRelationParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecursiveCommitParams {
    pub(crate) num_blocks: u64,
    pub(crate) blocks_per_group: u64,
    pub(crate) num_block_groups: u64,
    pub(crate) num_rows: u64,
    pub(crate) num_cols: u64,
    pub(crate) ring_d: u64,
    pub(crate) num_primes: u64,
    pub(crate) matrix_rings: u64,
    pub(crate) output_coefficients: u64,
    pub(crate) rhs_abs_bound: u64,
}

const _: [(); 80] = [(); size_of::<RecursiveCommitParams>()];

#[repr(C)]
#[derive(Clone, Copy)]
struct D512LinearNttPrime {
    p: i32,
    pinv: i32,
    mont: i32,
    montsq: i32,
}

const _: [(); 16] = [(); size_of::<D512LinearNttPrime>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRangeParams {
    pub(crate) live_len: u64,
    pub(crate) current_len: u64,
    pub(crate) current_live_len: u64,
    pub(crate) input_live_len: u64,
    pub(crate) pair_count: u64,
    pub(crate) num_first: u64,
    pub(crate) num_second: u64,
    pub(crate) workgroups: u64,
    pub(crate) basis: u64,
    pub(crate) prefix_size: u64,
    pub(crate) materialize_prefix: u64,
    pub(crate) resident_challenges: u64,
}

const _: [(); 96] = [(); size_of::<DirectRangeParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Blake2bSumcheckChallengeParams {
    include_claim: u64,
    coefficient_count: u64,
    prior_squeezed_bytes: u64,
    reserved: u64,
}

const _: [(); 32] = [(); size_of::<Blake2bSumcheckChallengeParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRelationParams {
    pub(crate) live_len: u64,
    pub(crate) current_len: u64,
    pub(crate) current_live_len: u64,
    pub(crate) input_live_len: u64,
    pub(crate) pair_count: u64,
    pub(crate) num_first: u64,
    pub(crate) num_second: u64,
    pub(crate) workgroups: u64,
    pub(crate) current_coeff_count: u64,
    pub(crate) live_lane_count: u64,
    pub(crate) prefix_size: u64,
    pub(crate) materialize_prefix: u64,
    pub(crate) linear_mode: u64,
    pub(crate) additional_pair_count: u64,
    pub(crate) additional_workgroups: u64,
    pub(crate) fold_lane_weights: u64,
    pub(crate) resident_challenges: u64,
}

const _: [(); 136] = [(); size_of::<DirectRelationParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DirectRelationTranscriptParams {
    prior_squeezed_bytes: u64,
    has_additional: u64,
}

const _: [(); 16] = [(); size_of::<DirectRelationTranscriptParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DirectRelationTwoRoundPrefixParams {
    live_lane_count: u64,
    coefficient_count: u64,
    y_quads: u64,
    equality_first_len: u64,
    workgroups: u64,
    lanes_per_thread: u64,
    norm_omitted_corner: u64,
    linear_mode: u64,
}

const _: [(); 64] = [(); size_of::<DirectRelationTwoRoundPrefixParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRelationScalars {
    pub(crate) l_at_0: Fp128Limbs,
    pub(crate) l_at_1: Fp128Limbs,
    pub(crate) binary_batching: Fp128Limbs,
}

const _: [(); 48] = [(); size_of::<DirectRelationScalars>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRelationLinearSegment {
    pub(crate) factor: Fp128Limbs,
    pub(crate) source_index: u32,
    pub(crate) target_lane_start: u32,
    pub(crate) target_lane_stride: u32,
    pub(crate) source_lane_start: u32,
    pub(crate) source_lane_stride: u32,
    pub(crate) lane_count: u32,
}

const _: [(); 48] = [(); size_of::<DirectRelationLinearSegment>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DirectRelationLinearFoldParams {
    current_coeff_count: u64,
    source_lane_count: u64,
    current_live_lane_count: u64,
    output_len: u64,
    mode: u64,
}

const _: [(); 40] = [(); size_of::<DirectRelationLinearFoldParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DirectRelationReducedSourceParams {
    ring_dimension: u64,
    row_count: u64,
    item_count: u64,
    reserved: u64,
    alpha: Fp128Limbs,
    wrap_correction: Fp128Limbs,
}

const _: [(); 64] = [(); size_of::<DirectRelationReducedSourceParams>()];

pub(crate) enum DirectRelationLinearSourceInput {
    Values(Vec<Fp128Limbs>),
    ReducedSetup {
        matrix: Buffer,
        ring_dimension: usize,
        row_count: usize,
        column_count: usize,
        row_weights: Vec<Fp128Limbs>,
        alpha_powers: Vec<Fp128Limbs>,
        alpha: Fp128Limbs,
        wrap_correction: Fp128Limbs,
    },
    ReducedSparse {
        ring_dimension: usize,
        challenge_count: usize,
        term_offsets: Vec<u32>,
        positions: Vec<u32>,
        coefficients: Vec<i8>,
        alpha_powers: Vec<Fp128Limbs>,
        alpha: Fp128Limbs,
        wrap_correction: Fp128Limbs,
    },
}

impl DirectRelationLinearSourceInput {
    fn element_len(&self) -> Option<usize> {
        match self {
            Self::Values(values) => Some(values.len()),
            Self::ReducedSetup {
                ring_dimension,
                column_count,
                ..
            } => ring_dimension.checked_mul(*column_count),
            Self::ReducedSparse {
                ring_dimension,
                challenge_count,
                ..
            } => ring_dimension.checked_mul(*challenge_count),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRelationAdditionalPair {
    pub(crate) parent: u64,
    pub(crate) reserved: u64,
    pub(crate) linear: [Fp128Limbs; 2],
    pub(crate) binary: [Fp128Limbs; 2],
}

const _: [(); 80] = [(); size_of::<DirectRelationAdditionalPair>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DirectRelationAdditionalFoldMapping {
    parent: u64,
    left: u32,
    right: u32,
}

const _: [(); 16] = [(); size_of::<DirectRelationAdditionalFoldMapping>()];

trait PackedLaneSource {
    fn lane_count(&self) -> usize;

    fn wait_lanes(&self, rows: Range<usize>, lane_stride: usize)
        -> Result<&[u8], MetalCommitError>;
}

struct ResidentPackedLanes<'a> {
    lanes: &'a [u8],
}

impl PackedLaneSource for ResidentPackedLanes<'_> {
    fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    fn wait_lanes(
        &self,
        rows: Range<usize>,
        lane_stride: usize,
    ) -> Result<&[u8], MetalCommitError> {
        let first = rows
            .start
            .checked_mul(lane_stride)
            .ok_or(MetalCommitError::ShapeOverflow("packed first lane"))?;
        let final_lane = rows
            .end
            .checked_mul(lane_stride)
            .ok_or(MetalCommitError::ShapeOverflow("packed final lane"))?;
        self.lanes
            .get(first..final_lane)
            .ok_or(MetalCommitError::ShapeOverflow("packed command lane range"))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DispatchTimings {
    pub(crate) buffer_setup: Duration,
    pub(crate) command_wall: Duration,
    pub(crate) gpu: Option<Duration>,
    pub(crate) readback_copy: Duration,
}

pub(crate) struct DispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) panel_gpu_active: Option<Duration>,
    pub(crate) panel_gpu_span: Option<Duration>,
    pub(crate) reduction_gpu: Option<Duration>,
    pub(crate) command_buffers: usize,
    pub(crate) kernel: MetalOneHotKernel,
    pub(crate) blocks_per_threadgroup: usize,
    pub(crate) columns_per_threadgroup: usize,
    pub(crate) matrix_block_streams: usize,
    pub(crate) scratch_bytes: usize,
    pub(crate) input_zero_copy: bool,
}

pub(crate) struct DigitRowsDispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct CoefficientPackingDispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct PackedDecomposeFoldDispatchOutcome {
    pub(crate) centered_coefficients: Vec<i32>,
    pub(crate) timings: DispatchTimings,
    pub(crate) consumer_time: Duration,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct PackedFp128D512FoldIndex {
    records: Buffer,
    counts: Buffer,
    params: PackedFoldIndexParams,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum PackedFp128D512FoldSource<'a> {
    Retained(&'a PackedFp128D512FoldIndex),
    Fused(PackedFoldIndexParams),
}

pub(crate) struct PackedFp128D512CoefficientPackingIndex {
    records: Buffer,
    offsets: Buffer,
    params: PackedCoefficientPackingIndexParams,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct D512LinearRelationDispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct RecursiveCommitDispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct RecursiveCommitMatrixNttOutcome {
    pub(crate) buffer: Buffer,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRangeRoundOutcome {
    pub(crate) coefficients: [Fp128Limbs; FP128_DIRECT_RANGE_STORED_COEFFICIENTS],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

#[cfg(test)]
pub(crate) struct Blake2bSumcheckChallengeOutcome {
    pub(crate) challenge: Fp128Limbs,
    pub(crate) chaining_value: [u8; 64],
}

pub(crate) struct DirectRangeAdvanceOutcome {
    pub(crate) next_coefficients: Option<[Fp128Limbs; FP128_DIRECT_RANGE_STORED_COEFFICIENTS]>,
    pub(crate) final_evaluation: Option<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

#[cfg(test)]
pub(crate) struct DirectRangeResidentOutcome {
    pub(crate) round_coefficients: Vec<[Fp128Limbs; FP128_DIRECT_RANGE_STORED_COEFFICIENTS]>,
    pub(crate) challenges: Vec<Fp128Limbs>,
    pub(crate) final_evaluation: Fp128Limbs,
    pub(crate) chaining_value: [u8; 64],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRangeSession {
    compact_digits: Buffer,
    tables: [Buffer; 2],
    partials: Buffer,
    round_output: Buffer,
    final_output: Buffer,
    live_len: usize,
    current_len: usize,
    current_live_len: usize,
    current_table: Option<usize>,
    compact_prefix_rounds: usize,
    rounds_folded: usize,
    allocation_bytes: usize,
}

pub(crate) struct DirectRelationRoundData<'a> {
    pub(crate) e_first: &'a [Fp128Limbs],
    pub(crate) e_second: &'a [Fp128Limbs],
    pub(crate) alpha: &'a [Fp128Limbs],
    pub(crate) additional_pairs: &'a [DirectRelationAdditionalPair],
    pub(crate) scalars: DirectRelationScalars,
    pub(crate) live_lane_count: usize,
}

pub(crate) struct DirectRelationRoundOutcome {
    pub(crate) coefficients: [Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS],
    pub(crate) additional_coefficients: [Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRelationTwoRoundPrefixOutcome {
    pub(crate) norm_evals_except_corner: [Fp128Limbs; 8],
    pub(crate) relation_evals_except_corner: [Fp128Limbs; 8],
    pub(crate) additional_coefficients: [Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRelationAdditionalOutcome {
    pub(crate) coefficients: [Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRelationAdvanceOutcome {
    pub(crate) next_coefficients: Option<[Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS]>,
    pub(crate) next_additional_coefficients:
        Option<[Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS]>,
    pub(crate) final_evaluation: Option<Fp128Limbs>,
    pub(crate) final_linear_evaluation: Option<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRelationResidentEqRound {
    pub(crate) e_first: Vec<Fp128Limbs>,
    pub(crate) e_second: Vec<Fp128Limbs>,
    pub(crate) tau: Fp128Limbs,
}

pub(crate) struct DirectRelationResidentOutcome {
    pub(crate) round_coefficients: Vec<[Fp128Limbs; 3]>,
    pub(crate) coefficient_counts: Vec<usize>,
    pub(crate) challenges: Vec<Fp128Limbs>,
    pub(crate) final_evaluation: Fp128Limbs,
    pub(crate) final_linear_evaluation: Fp128Limbs,
    pub(crate) chaining_value: [u8; 64],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRelationSession {
    compact_digits: Buffer,
    tables: [Buffer; 2],
    partials: Buffer,
    round_output: Buffer,
    additional_output: Buffer,
    final_output: Buffer,
    linear_final_output: Buffer,
    linear_segments: Buffer,
    lane_offsets: Buffer,
    lane_segments: Buffer,
    linear_source_lane_offsets: Buffer,
    linear_tables: [Buffer; 2],
    current_linear_table: usize,
    linear_mode: usize,
    linear_source_lane_count: usize,
    linear_current_coeff_count: usize,
    linear_current_live_lane_count: usize,
    lane_weight_tables: [Buffer; 2],
    two_round_prefix_partials: Buffer,
    two_round_prefix_output: Buffer,
    two_round_prefix_max_workgroups: usize,
    live_len: usize,
    current_len: usize,
    current_live_len: usize,
    current_table: Option<usize>,
    current_lane_weight_table: usize,
    current_lane_count: usize,
    coefficient_rounds: usize,
    compact_prefix_rounds: usize,
    rounds_folded: usize,
    allocation_bytes: usize,
}

struct DirectRelationRoundBuffers {
    e_first: Buffer,
    e_second: Buffer,
    alpha: Buffer,
    allocation_bytes: usize,
}

#[derive(Clone, Copy)]
enum Fp128KernelBinding<'a> {
    Inline(Fp128Limbs),
    Buffer(&'a Buffer, u64),
}

struct PackedLaneBuffer<'a> {
    buffer: Buffer,
    zero_copy: bool,
    marker: PhantomData<&'a [u8]>,
}

struct SharedByteBuffer<'a> {
    buffer: Buffer,
    zero_copy: bool,
    marker: PhantomData<&'a [u8]>,
}

pub(crate) struct SharedSliceBuffer<'a, T> {
    pub(crate) buffer: Buffer,
    pub(crate) zero_copy: bool,
    marker: PhantomData<&'a [T]>,
}

struct D512LinearRelationResources {
    primes: Buffer,
    fwd_twiddles: Buffer,
    inv_twiddles: Buffer,
    d_inv: Buffer,
    limb_weights: Buffer,
    field_moduli: Buffer,
    garner_gamma: Buffer,
    field_partial_products: Buffer,
}

struct RecursiveCommitResources {
    ring_d: usize,
    primes: Buffer,
    fwd_twiddles: Buffer,
    inv_twiddles: Buffer,
    psi_pows: Buffer,
    inverse_scale: Buffer,
    limb_weights: Buffer,
    field_moduli: Buffer,
    garner_gamma: Buffer,
    field_partial_products: Buffer,
}

pub(crate) struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    direct_pipeline: ComputePipelineState,
    block_batched_pipeline: ComputePipelineState,
    packed_fp128_d512_pipeline: ComputePipelineState,
    packed_fp128_d128_rank3_pipeline: ComputePipelineState,
    packed_partial_reduction_pipeline: ComputePipelineState,
    fp128_d64_digit_rows_partials_pipeline: ComputePipelineState,
    fp128_d64_digit_rows_reduce_pipeline: ComputePipelineState,
    fp128_i8_coefficient_packing_pipeline: ComputePipelineState,
    fp128_packed_onehot_coefficient_packing_partials_pipeline: ComputePipelineState,
    fp128_packed_onehot_coefficient_packing_reduce_pipeline: ComputePipelineState,
    fp128_d512_decompose_fold_pipeline: ComputePipelineState,
    fp128_d128_decompose_fold_pipeline: ComputePipelineState,
    fp128_d128_subring64_decompose_fold_pipeline: ComputePipelineState,
    fp128_d512_subring64_decompose_fold_pipeline: ComputePipelineState,
    fp128_d512_build_fold_index_pipeline: ComputePipelineState,
    fp128_d512_build_coefficient_packing_index_pipeline: ComputePipelineState,
    fp128_d512_indexed_coefficient_packing_pipeline: ComputePipelineState,
    fp128_d512_indexed_coefficient_packing_reduce_pipeline: ComputePipelineState,
    fp128_d512_indexed_subring64_decompose_fold_pipeline: ComputePipelineState,
    fp128_d512_fused_subring64_decompose_fold_pipeline: ComputePipelineState,
    fp128_d512_linear_relation_partials_pipeline: ComputePipelineState,
    fp128_d512_linear_relation_reduce_pipeline: ComputePipelineState,
    fp128_d512_linear_relation_reconstruct_pipeline: ComputePipelineState,
    fp128_d512_linear_relation_resources: D512LinearRelationResources,
    fp128_recursive_commit_matrix_ntt_pipeline: ComputePipelineState,
    fp128_recursive_commit_matvec_pipeline: ComputePipelineState,
    fp128_recursive_commit_reconstruct_pipeline: ComputePipelineState,
    fp128_recursive_commit_resources: [RecursiveCommitResources; 2],
    fp128_direct_range_initial_pipeline: ComputePipelineState,
    fp128_direct_range_compact_fold_pipeline: ComputePipelineState,
    fp128_direct_range_field_fold_pipeline: ComputePipelineState,
    fp128_direct_range_reduce_pipeline: ComputePipelineState,
    fp128_direct_range_finalize_pipeline: ComputePipelineState,
    fp128_blake2b_sumcheck_challenge_pipeline: ComputePipelineState,
    fp128_blake2b_relation_sumcheck_round_pipeline: ComputePipelineState,
    fp128_direct_relation_initial_pipeline: ComputePipelineState,
    fp128_direct_relation_compact_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_field_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_additional_compact_pipeline: ComputePipelineState,
    fp128_direct_relation_additional_field_pipeline: ComputePipelineState,
    fp128_direct_relation_two_round_prefix_pipeline: ComputePipelineState,
    fp128_direct_relation_two_round_prefix_reduce_pipeline: ComputePipelineState,
    fp128_direct_relation_linear_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_alpha_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_scalar_advance_pipeline: ComputePipelineState,
    fp128_direct_relation_additional_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_setup_source_pipeline: ComputePipelineState,
    fp128_direct_relation_sparse_source_pipeline: ComputePipelineState,
}

fn pow_mod(mut base: i64, mut exponent: i64, modulus: i64) -> i64 {
    let mut result = 1i64;
    base %= modulus;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    result
}

fn primitive_root_of_unity(modulus: i64, order: usize) -> i64 {
    let half = (modulus - 1) / 2;
    let exponent = (modulus - 1) / order as i64;
    for candidate in 2..modulus {
        if pow_mod(candidate, half, modulus) == modulus - 1 {
            let root = pow_mod(candidate, exponent, modulus);
            if pow_mod(root, order as i64, modulus) == 1
                && pow_mod(root, order as i64 / 2, modulus) == modulus - 1
            {
                return root;
            }
        }
    }
    unreachable!("the fixed NTT prime has no root of the required order")
}

fn constant_buffer<T>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast::<c_void>(),
        size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn d512_linear_relation_resources(device: &Device) -> D512LinearRelationResources {
    let primes = FP128_D512_LINEAR_RELATION_RAW_PRIMES.map(NttPrime::<i32>::compute);
    let mut device_primes = Vec::with_capacity(primes.len());
    let mut fwd_twiddles = Vec::with_capacity(primes.len() * FP128_D512_LINEAR_RELATION_NTT_SIZE);
    let mut inv_twiddles = Vec::with_capacity(primes.len() * FP128_D512_LINEAR_RELATION_NTT_SIZE);
    let mut d_inv = Vec::with_capacity(primes.len());
    let mut limb_weights = Vec::with_capacity(primes.len() * 4);
    let mut field_moduli = Vec::with_capacity(primes.len());
    let field_modulus = (-F::one()).to_canonical_u128() + 1;

    for prime in primes {
        device_primes.push(D512LinearNttPrime {
            p: prime.p,
            pinv: prime.pinv,
            mont: prime.mont,
            montsq: prime.montsq,
        });
        let modulus = i64::from(prime.p);
        let root = primitive_root_of_unity(modulus, FP128_D512_LINEAR_RELATION_NTT_SIZE);
        let root_inverse = pow_mod(root, modulus - 2, modulus);
        let one = prime.from_canonical(1);
        let mut prime_fwd = vec![0i32; FP128_D512_LINEAR_RELATION_NTT_SIZE];
        let mut prime_inv = vec![0i32; FP128_D512_LINEAR_RELATION_NTT_SIZE];
        for stage in 0..FP128_D512_LINEAR_RELATION_NTT_SIZE.ilog2() as usize {
            let len = 1usize << stage;
            let exponent = (FP128_D512_LINEAR_RELATION_NTT_SIZE / (2 * len)) as i64;
            let fwd_step = prime.from_canonical(pow_mod(root, exponent, modulus) as i32);
            let inv_step = prime.from_canonical(pow_mod(root_inverse, exponent, modulus) as i32);
            let mut fwd = one;
            let mut inv = one;
            for offset in 0..len {
                prime_fwd[len - 1 + offset] = fwd.raw();
                prime_inv[len - 1 + offset] = inv.raw();
                fwd = prime.mul(fwd, fwd_step);
                inv = prime.mul(inv, inv_step);
            }
        }
        fwd_twiddles.extend(prime_fwd);
        inv_twiddles.extend(prime_inv);
        let inverse = prime.from_canonical(pow_mod(
            FP128_D512_LINEAR_RELATION_NTT_SIZE as i64,
            modulus - 2,
            modulus,
        ) as i32);
        d_inv.push(inverse.raw());

        let radix = (1u128 << 32) % prime.p as u128;
        let mut weight = 1u128;
        for _ in 0..4 {
            limb_weights.push(prime.from_canonical(weight as i32).raw());
            weight = weight * radix % prime.p as u128;
        }
        field_moduli.push(
            prime
                .from_canonical((field_modulus % prime.p as u128) as i32)
                .raw(),
        );
    }

    let garner = GarnerData::compute(&primes);
    let garner_gamma = garner
        .gamma
        .into_iter()
        .flatten()
        .map(|value| value as u32)
        .collect::<Vec<_>>();
    let mut partial_product = F::one();
    let mut field_partial_products = Vec::with_capacity(primes.len());
    for prime in primes {
        field_partial_products.push(Fp128Limbs::from_field(partial_product));
        partial_product *= F::from_u64(prime.p as u64);
    }

    D512LinearRelationResources {
        primes: constant_buffer(device, &device_primes),
        fwd_twiddles: constant_buffer(device, &fwd_twiddles),
        inv_twiddles: constant_buffer(device, &inv_twiddles),
        d_inv: constant_buffer(device, &d_inv),
        limb_weights: constant_buffer(device, &limb_weights),
        field_moduli: constant_buffer(device, &field_moduli),
        garner_gamma: constant_buffer(device, &garner_gamma),
        field_partial_products: constant_buffer(device, &field_partial_products),
    }
}

fn recursive_commit_resources(device: &Device, ring_d: usize) -> RecursiveCommitResources {
    let primes = FP128_D512_LINEAR_RELATION_RAW_PRIMES.map(NttPrime::<i32>::compute);
    let mut device_primes = Vec::with_capacity(primes.len());
    let mut fwd_twiddles = Vec::with_capacity(primes.len() * ring_d);
    let mut inv_twiddles = Vec::with_capacity(primes.len() * ring_d);
    let mut psi_pows = Vec::with_capacity(primes.len() * ring_d);
    let mut inverse_scale = Vec::with_capacity(primes.len() * ring_d);
    let mut limb_weights = Vec::with_capacity(primes.len() * 4);
    let mut field_moduli = Vec::with_capacity(primes.len());
    let field_modulus = (-F::one()).to_canonical_u128() + 1;

    for prime in primes {
        device_primes.push(D512LinearNttPrime {
            p: prime.p,
            pinv: prime.pinv,
            mont: prime.mont,
            montsq: prime.montsq,
        });
        let modulus = i64::from(prime.p);
        let psi = primitive_root_of_unity(modulus, 2 * ring_d);
        let psi_inverse = pow_mod(psi, modulus - 2, modulus);
        let omega = psi * psi % modulus;
        let omega_inverse = pow_mod(omega, modulus - 2, modulus);
        let one = prime.from_canonical(1);
        let mut prime_fwd = vec![0i32; ring_d];
        let mut prime_inv = vec![0i32; ring_d];
        for stage in 0..ring_d.ilog2() as usize {
            let len = 1usize << stage;
            let exponent = (ring_d / (2 * len)) as i64;
            let fwd_step = prime.from_canonical(pow_mod(omega, exponent, modulus) as i32);
            let inv_step = prime.from_canonical(pow_mod(omega_inverse, exponent, modulus) as i32);
            let mut fwd = one;
            let mut inv = one;
            for offset in 0..len {
                prime_fwd[len - 1 + offset] = fwd.raw();
                prime_inv[len - 1 + offset] = inv.raw();
                fwd = prime.mul(fwd, fwd_step);
                inv = prime.mul(inv, inv_step);
            }
        }
        fwd_twiddles.extend(prime_fwd);
        inv_twiddles.extend(prime_inv);

        let psi_mont = prime.from_canonical(psi as i32);
        let psi_inverse_mont = prime.from_canonical(psi_inverse as i32);
        let d_inverse = prime.from_canonical(pow_mod(ring_d as i64, modulus - 2, modulus) as i32);
        let mut psi_power = one;
        let mut psi_inverse_power = one;
        for _ in 0..ring_d {
            psi_pows.push(psi_power.raw());
            inverse_scale.push(prime.mul(d_inverse, psi_inverse_power).raw());
            psi_power = prime.mul(psi_power, psi_mont);
            psi_inverse_power = prime.mul(psi_inverse_power, psi_inverse_mont);
        }

        let radix = (1u128 << 32) % prime.p as u128;
        let mut weight = 1u128;
        for _ in 0..4 {
            limb_weights.push(prime.from_canonical(weight as i32).raw());
            weight = weight * radix % prime.p as u128;
        }
        field_moduli.push(
            prime
                .from_canonical((field_modulus % prime.p as u128) as i32)
                .raw(),
        );
    }

    let garner = GarnerData::compute(&primes);
    let garner_gamma = garner
        .gamma
        .into_iter()
        .flatten()
        .map(|value| value as u32)
        .collect::<Vec<_>>();
    let mut partial_product = F::one();
    let mut field_partial_products = Vec::with_capacity(primes.len());
    for prime in primes {
        field_partial_products.push(Fp128Limbs::from_field(partial_product));
        partial_product *= F::from_u64(prime.p as u64);
    }

    RecursiveCommitResources {
        ring_d,
        primes: constant_buffer(device, &device_primes),
        fwd_twiddles: constant_buffer(device, &fwd_twiddles),
        inv_twiddles: constant_buffer(device, &inv_twiddles),
        psi_pows: constant_buffer(device, &psi_pows),
        inverse_scale: constant_buffer(device, &inverse_scale),
        limb_weights: constant_buffer(device, &limb_weights),
        field_moduli: constant_buffer(device, &field_moduli),
        garner_gamma: constant_buffer(device, &garner_gamma),
        field_partial_products: constant_buffer(device, &field_partial_products),
    }
}

impl MetalRuntime {
    pub(crate) fn new() -> Result<Self, MetalCommitError> {
        let device = Device::system_default().ok_or(MetalCommitError::DeviceUnavailable)?;
        Self::from_device(device)
    }

    pub(crate) fn from_device(device: Device) -> Result<Self, MetalCommitError> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(KERNEL_SOURCE, &options)
            .map_err(MetalCommitError::LibraryCompilation)?;
        let pipeline = |name| {
            let function = library
                .get_function(name, None)
                .map_err(|message| MetalCommitError::FunctionLookup { name, message })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|message| MetalCommitError::PipelineCompilation { name, message })
        };
        let fp128_d512_linear_relation_resources = d512_linear_relation_resources(&device);
        let fp128_recursive_commit_resources = [
            recursive_commit_resources(&device, 64),
            recursive_commit_resources(&device, 128),
        ];
        Ok(Self {
            queue: device.new_command_queue(),
            direct_pipeline: pipeline(DIRECT_KERNEL_NAME)?,
            block_batched_pipeline: pipeline(BLOCK_BATCHED_KERNEL_NAME)?,
            packed_fp128_d512_pipeline: pipeline(PACKED_FP128_D512_PANELS_KERNEL_NAME)?,
            packed_fp128_d128_rank3_pipeline: pipeline(PACKED_FP128_D128_RANK3_KERNEL_NAME)?,
            packed_partial_reduction_pipeline: pipeline(PACKED_PARTIAL_REDUCTION_KERNEL_NAME)?,
            fp128_d64_digit_rows_partials_pipeline: pipeline(
                FP128_D64_DIGIT_ROWS_PARTIALS_KERNEL_NAME,
            )?,
            fp128_d64_digit_rows_reduce_pipeline: pipeline(
                FP128_D64_DIGIT_ROWS_REDUCE_KERNEL_NAME,
            )?,
            fp128_i8_coefficient_packing_pipeline: pipeline(
                FP128_I8_COEFFICIENT_PACKING_KERNEL_NAME,
            )?,
            fp128_packed_onehot_coefficient_packing_partials_pipeline: pipeline(
                FP128_PACKED_ONEHOT_COEFFICIENT_PACKING_PARTIALS_KERNEL_NAME,
            )?,
            fp128_packed_onehot_coefficient_packing_reduce_pipeline: pipeline(
                FP128_PACKED_ONEHOT_COEFFICIENT_PACKING_REDUCE_KERNEL_NAME,
            )?,
            fp128_d512_decompose_fold_pipeline: pipeline(FP128_D512_DECOMPOSE_FOLD_KERNEL_NAME)?,
            fp128_d128_decompose_fold_pipeline: pipeline(FP128_D128_DECOMPOSE_FOLD_KERNEL_NAME)?,
            fp128_d128_subring64_decompose_fold_pipeline: pipeline(
                FP128_D128_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME,
            )?,
            fp128_d512_subring64_decompose_fold_pipeline: pipeline(
                FP128_D512_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME,
            )?,
            fp128_d512_build_fold_index_pipeline: pipeline(
                FP128_D512_BUILD_FOLD_INDEX_KERNEL_NAME,
            )?,
            fp128_d512_build_coefficient_packing_index_pipeline: pipeline(
                FP128_D512_BUILD_COEFFICIENT_PACKING_INDEX_KERNEL_NAME,
            )?,
            fp128_d512_indexed_coefficient_packing_pipeline: pipeline(
                FP128_D512_INDEXED_COEFFICIENT_PACKING_KERNEL_NAME,
            )?,
            fp128_d512_indexed_coefficient_packing_reduce_pipeline: pipeline(
                FP128_D512_INDEXED_COEFFICIENT_PACKING_REDUCE_KERNEL_NAME,
            )?,
            fp128_d512_indexed_subring64_decompose_fold_pipeline: pipeline(
                FP128_D512_INDEXED_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME,
            )?,
            fp128_d512_fused_subring64_decompose_fold_pipeline: pipeline(
                FP128_D512_FUSED_SUBRING64_DECOMPOSE_FOLD_KERNEL_NAME,
            )?,
            fp128_d512_linear_relation_partials_pipeline: pipeline(
                FP128_D512_LINEAR_RELATION_PARTIALS_KERNEL_NAME,
            )?,
            fp128_d512_linear_relation_reduce_pipeline: pipeline(
                FP128_D512_LINEAR_RELATION_REDUCE_KERNEL_NAME,
            )?,
            fp128_d512_linear_relation_reconstruct_pipeline: pipeline(
                FP128_D512_LINEAR_RELATION_RECONSTRUCT_KERNEL_NAME,
            )?,
            fp128_d512_linear_relation_resources,
            fp128_recursive_commit_matrix_ntt_pipeline: pipeline(
                FP128_RECURSIVE_COMMIT_MATRIX_NTT_KERNEL_NAME,
            )?,
            fp128_recursive_commit_matvec_pipeline: pipeline(
                FP128_RECURSIVE_COMMIT_MATVEC_KERNEL_NAME,
            )?,
            fp128_recursive_commit_reconstruct_pipeline: pipeline(
                FP128_RECURSIVE_COMMIT_RECONSTRUCT_KERNEL_NAME,
            )?,
            fp128_recursive_commit_resources,
            fp128_direct_range_initial_pipeline: pipeline(FP128_DIRECT_RANGE_INITIAL_KERNEL_NAME)?,
            fp128_direct_range_compact_fold_pipeline: pipeline(
                FP128_DIRECT_RANGE_COMPACT_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_range_field_fold_pipeline: pipeline(
                FP128_DIRECT_RANGE_FIELD_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_range_reduce_pipeline: pipeline(FP128_DIRECT_RANGE_REDUCE_KERNEL_NAME)?,
            fp128_direct_range_finalize_pipeline: pipeline(
                FP128_DIRECT_RANGE_FINALIZE_KERNEL_NAME,
            )?,
            fp128_blake2b_sumcheck_challenge_pipeline: pipeline(
                FP128_BLAKE2B_SUMCHECK_CHALLENGE_KERNEL_NAME,
            )?,
            fp128_blake2b_relation_sumcheck_round_pipeline: pipeline(
                FP128_BLAKE2B_RELATION_SUMCHECK_ROUND_KERNEL_NAME,
            )?,
            fp128_direct_relation_initial_pipeline: pipeline(
                FP128_DIRECT_RELATION_INITIAL_KERNEL_NAME,
            )?,
            fp128_direct_relation_compact_fold_pipeline: pipeline(
                FP128_DIRECT_RELATION_COMPACT_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_relation_field_fold_pipeline: pipeline(
                FP128_DIRECT_RELATION_FIELD_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_relation_additional_compact_pipeline: pipeline(
                FP128_DIRECT_RELATION_ADDITIONAL_COMPACT_KERNEL_NAME,
            )?,
            fp128_direct_relation_additional_field_pipeline: pipeline(
                FP128_DIRECT_RELATION_ADDITIONAL_FIELD_KERNEL_NAME,
            )?,
            fp128_direct_relation_two_round_prefix_pipeline: pipeline(
                FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_KERNEL_NAME,
            )?,
            fp128_direct_relation_two_round_prefix_reduce_pipeline: pipeline(
                FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_REDUCE_KERNEL_NAME,
            )?,
            fp128_direct_relation_linear_fold_pipeline: pipeline(
                FP128_DIRECT_RELATION_LINEAR_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_relation_alpha_fold_pipeline: pipeline(
                FP128_DIRECT_RELATION_ALPHA_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_relation_scalar_advance_pipeline: pipeline(
                FP128_DIRECT_RELATION_SCALAR_ADVANCE_KERNEL_NAME,
            )?,
            fp128_direct_relation_additional_fold_pipeline: pipeline(
                FP128_DIRECT_RELATION_ADDITIONAL_FOLD_KERNEL_NAME,
            )?,
            fp128_direct_relation_setup_source_pipeline: pipeline(
                FP128_DIRECT_RELATION_SETUP_SOURCE_KERNEL_NAME,
            )?,
            fp128_direct_relation_sparse_source_pipeline: pipeline(
                FP128_DIRECT_RELATION_SPARSE_SOURCE_KERNEL_NAME,
            )?,
            device,
        })
    }

    pub(crate) fn capabilities(&self) -> MetalDeviceCapabilities {
        MetalDeviceCapabilities {
            name: self.device.name().to_owned(),
            max_buffer_length: self.device.max_buffer_length(),
            recommended_max_working_set_size: self.device.recommended_max_working_set_size(),
            max_threadgroup_memory_length: self.device.max_threadgroup_memory_length(),
            thread_execution_width: self.block_batched_pipeline.thread_execution_width() as usize,
            max_total_threads_per_threadgroup: self
                .block_batched_pipeline
                .max_total_threads_per_threadgroup()
                as usize,
            static_threadgroup_memory_length: self
                .block_batched_pipeline
                .static_threadgroup_memory_length(),
        }
    }

    pub(crate) fn supports_packed_fp128_d512_panels(&self) -> bool {
        self.packed_fp128_d512_pipeline
            .max_total_threads_per_threadgroup()
            >= FP128_D512_THREADS as u64
            && self
                .packed_fp128_d512_pipeline
                .static_threadgroup_memory_length()
                == FP128_D512_THREADGROUP_BYTES as u64
    }

    pub(crate) fn supports_packed_fp128_d128_rank3(&self) -> bool {
        self.packed_fp128_d128_rank3_pipeline
            .max_total_threads_per_threadgroup()
            >= FP128_D512_THREADS as u64
            && self
                .packed_fp128_d128_rank3_pipeline
                .static_threadgroup_memory_length()
                == FP128_D128_RANK3_THREADGROUP_BYTES as u64
    }

    pub(crate) fn supports_fp128_d64_digit_rows<const D: usize>(
        &self,
        num_vectors: usize,
        num_rows: usize,
        num_cols: usize,
        retain_quotients: bool,
    ) -> bool {
        let Ok(num_vectors) = u64::try_from(num_vectors) else {
            return false;
        };
        let Ok(num_rows) = u64::try_from(num_rows) else {
            return false;
        };
        let Ok(num_cols) = u64::try_from(num_cols) else {
            return false;
        };
        if D != 64
            || num_vectors == 0
            || num_rows == 0
            || num_cols == 0
            || num_cols > u64::from(u32::MAX)
        {
            return false;
        }
        let column_partials = num_cols.div_ceil(FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64);
        let output_coefficients = num_vectors
            .checked_mul(num_rows)
            .and_then(|count| count.checked_mul(D as u64));
        let threadgroups = num_vectors
            .checked_mul(num_rows)
            .and_then(|count| count.checked_mul(column_partials));
        let matrix_bytes = num_rows
            .checked_mul(num_cols)
            .and_then(|count| count.checked_mul(D as u64))
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64));
        let digit_bytes = num_vectors
            .checked_mul(num_cols)
            .and_then(|count| count.checked_mul(D as u64));
        let product_count = 1 + u64::from(retain_quotients);
        let partial_bytes = threadgroups
            .and_then(|count| count.checked_mul(D as u64))
            .and_then(|count| count.checked_mul(product_count))
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64));
        let output_bytes = output_coefficients
            .and_then(|count| count.checked_mul(product_count))
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64));
        let maximum = self.device.max_buffer_length();
        output_coefficients
            .and_then(|count| count.checked_mul(product_count))
            .is_some_and(|count| count <= u64::from(u32::MAX))
            && threadgroups.is_some_and(|count| count <= u64::from(u32::MAX))
            && [matrix_bytes, digit_bytes, partial_bytes, output_bytes]
                .into_iter()
                .all(|bytes| bytes.is_some_and(|bytes| bytes != 0 && bytes <= maximum))
            && self
                .fp128_d64_digit_rows_partials_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_D64_DIGIT_ROWS_PARTIAL_THREADS as u64
            && self
                .fp128_d64_digit_rows_reduce_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_D64_DIGIT_ROWS_THREADS as u64
    }

    pub(crate) fn supports_fp128_d512_linear_relation(
        &self,
        num_columns: usize,
        rhs_abs_bound: u64,
    ) -> bool {
        if num_columns == 0 || rhs_abs_bound >= FP128_D512_LINEAR_RELATION_RAW_PRIMES[0] as u64 {
            return false;
        }
        let capacity = CrtCapacity::from_prime_moduli(
            FP128_D512_LINEAR_RELATION_RAW_PRIMES.map(|prime| prime as u128),
        );
        let field_modulus = (-F::one()).to_canonical_u128() + 1;
        if !capacity.supports_modulus(num_columns, 512, field_modulus, rhs_abs_bound) {
            return false;
        }
        let num_tiles = num_columns.div_ceil(FP128_D512_LINEAR_RELATION_COLUMNS_PER_TILE);
        let matrix_bytes = num_columns
            .checked_mul(512)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()));
        let rhs_bytes = num_columns
            .checked_mul(512)
            .and_then(|count| count.checked_mul(size_of::<i32>()));
        let partial_bytes = num_tiles
            .checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES)
            .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NTT_SIZE))
            .and_then(|count| count.checked_mul(size_of::<i32>()));
        [matrix_bytes, rhs_bytes, partial_bytes]
            .into_iter()
            .all(|bytes| {
                bytes.is_some_and(|bytes| {
                    bytes != 0 && bytes as u64 <= self.device.max_buffer_length()
                })
            })
            && num_tiles
                .checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES)
                .is_some_and(|groups| groups <= u32::MAX as usize)
            && self
                .fp128_d512_linear_relation_partials_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_D512_LINEAR_RELATION_THREADS as u64
            && self
                .fp128_d512_linear_relation_reduce_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_D512_LINEAR_RELATION_THREADS as u64
            && self
                .fp128_d512_linear_relation_reconstruct_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_D512_LINEAR_RELATION_THREADS as u64
    }

    fn recursive_commit_resources(&self, ring_d: usize) -> Option<&RecursiveCommitResources> {
        self.fp128_recursive_commit_resources
            .iter()
            .find(|resources| resources.ring_d == ring_d)
    }

    pub(crate) fn supports_fp128_recursive_commit<const D: usize>(
        &self,
        num_blocks: usize,
        num_rows: usize,
        num_cols: usize,
        rhs_abs_bound: u64,
    ) -> bool {
        if !matches!(D, 64 | 128)
            || num_blocks == 0
            || num_rows == 0
            || num_rows > FP128_RECURSIVE_COMMIT_MAX_ROWS
            || num_cols == 0
            || rhs_abs_bound >= FP128_D512_LINEAR_RELATION_RAW_PRIMES[0] as u64
            || self.recursive_commit_resources(D).is_none()
        {
            return false;
        }
        let capacity = CrtCapacity::from_prime_moduli(
            FP128_D512_LINEAR_RELATION_RAW_PRIMES.map(|prime| prime as u128),
        );
        let field_modulus = (-F::one()).to_canonical_u128() + 1;
        if !capacity.supports_modulus(num_cols, D, field_modulus, rhs_abs_bound) {
            return false;
        }
        let matrix_rings = num_rows.checked_mul(num_cols);
        let source_bytes = num_blocks
            .checked_mul(num_cols)
            .and_then(|count| count.checked_mul(D));
        let matrix_ntt_bytes = matrix_rings
            .and_then(|count| count.checked_mul(D))
            .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
            .and_then(|count| count.checked_mul(size_of::<i32>()));
        let residue_bytes = num_blocks
            .checked_mul(num_rows)
            .and_then(|count| count.checked_mul(D))
            .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
            .and_then(|count| count.checked_mul(size_of::<u32>()));
        let output_bytes = num_blocks
            .checked_mul(num_rows)
            .and_then(|count| count.checked_mul(D))
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()));
        let maximum = self.device.max_buffer_length();
        [source_bytes, matrix_ntt_bytes, residue_bytes, output_bytes]
            .into_iter()
            .all(|bytes| bytes.is_some_and(|bytes| bytes != 0 && bytes as u64 <= maximum))
            && matrix_rings
                .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
                .is_some_and(|groups| groups <= u32::MAX as usize)
            && num_blocks
                .div_ceil(FP128_RECURSIVE_COMMIT_BLOCKS_PER_GROUP)
                .checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES)
                .is_some_and(|groups| groups <= u32::MAX as usize)
            && output_bytes
                .map(|bytes| bytes / size_of::<Fp128Limbs>())
                .is_some_and(|count| {
                    count.div_ceil(FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS) <= u32::MAX as usize
                })
            && self
                .fp128_recursive_commit_matrix_ntt_pipeline
                .max_total_threads_per_threadgroup()
                >= D as u64
            && self
                .fp128_recursive_commit_matvec_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_RECURSIVE_COMMIT_THREADS as u64
            && self
                .fp128_recursive_commit_reconstruct_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS as u64
    }

    pub(crate) fn shared_buffer_from_slice<T>(
        &self,
        values: &[T],
    ) -> Result<Buffer, MetalCommitError> {
        let bytes = size_of_val(values);
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal input buffer".into(),
            ));
        }
        Ok(self.device.new_buffer_with_data(
            values.as_ptr().cast::<c_void>(),
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    fn shared_buffer_from_digit_rows<const D: usize>(
        &self,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Buffer, MetalCommitError> {
        let bytes = digit_vectors.iter().try_fold(0usize, |total, digits| {
            total
                .checked_add(size_of_val(*digits))
                .ok_or(MetalCommitError::ShapeOverflow("digit-row input bytes"))
        })?;
        let buffer = self.shared_buffer(bytes)?;
        let mut destination = buffer.contents().cast::<u8>();
        for digits in digit_vectors {
            let len = size_of_val(*digits);
            // SAFETY: `buffer` owns `bytes`, the checked sum of the disjoint
            // source lengths, and both pointers are valid for `len` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(digits.as_ptr().cast::<u8>(), destination, len);
                destination = destination.add(len);
            }
        }
        Ok(buffer)
    }

    pub(crate) fn shared_slice_buffer<'a, T>(
        &self,
        values: &'a [T],
    ) -> Result<SharedSliceBuffer<'a, T>, MetalCommitError> {
        let bytes = size_of_val(values);
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal input buffer".into(),
            ));
        }
        let zero_copy = values
            .as_ptr()
            .addr()
            .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
            && bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
        let buffer = if zero_copy {
            self.device.new_buffer_with_bytes_no_copy(
                values.as_ptr().cast_mut().cast::<c_void>(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            self.shared_buffer_from_slice(values)?
        };
        Ok(SharedSliceBuffer {
            buffer,
            zero_copy,
            marker: PhantomData,
        })
    }

    fn shared_byte_buffer_from_slices<'a>(
        &self,
        values: &[&'a [i8]],
    ) -> Result<SharedByteBuffer<'a>, MetalCommitError> {
        let bytes = values.iter().try_fold(0usize, |total, value| {
            total
                .checked_add(size_of_val(*value))
                .ok_or(MetalCommitError::ShapeOverflow("byte input"))
        })?;
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal byte input".into(),
            ));
        }
        let zero_copy = values.len() == 1
            && values[0]
                .as_ptr()
                .addr()
                .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
            && bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
        let buffer = if zero_copy {
            self.device.new_buffer_with_bytes_no_copy(
                values[0].as_ptr().cast::<c_void>(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            let buffer = self.shared_buffer(bytes)?;
            let mut destination = buffer.contents().cast::<u8>();
            for value in values {
                let len = size_of_val(*value);
                // SAFETY: `buffer` owns the checked sum of every source length;
                // the destination advances over disjoint initialized ranges.
                unsafe {
                    std::ptr::copy_nonoverlapping(value.as_ptr().cast::<u8>(), destination, len);
                    destination = destination.add(len);
                }
            }
            buffer
        };
        Ok(SharedByteBuffer {
            buffer,
            zero_copy,
            marker: PhantomData,
        })
    }

    pub(crate) fn private_buffer_from_slice<T>(
        &self,
        values: &[T],
    ) -> Result<Buffer, MetalCommitError> {
        let bytes = size_of_val(values);
        let staging = self.shared_buffer_from_slice(values)?;
        let buffer = self.private_buffer(bytes)?;
        let command = self.queue.new_command_buffer();
        command.set_label("Akita immutable setup upload");
        let encoder = command.new_blit_command_encoder();
        encoder.copy_from_buffer(&staging, 0, &buffer, 0, bytes as u64);
        encoder.end_encoding();
        let _ = complete_command(command)?;
        Ok(buffer)
    }

    fn packed_lane_buffer<'a>(
        &self,
        lanes: &'a [u8],
    ) -> Result<PackedLaneBuffer<'a>, MetalCommitError> {
        let bytes = lanes.len();
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal input buffer".into(),
            ));
        }
        let zero_copy = lanes
            .as_ptr()
            .addr()
            .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
            && bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
        let buffer = if zero_copy {
            self.device.new_buffer_with_bytes_no_copy(
                lanes.as_ptr().cast::<c_void>(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            self.shared_buffer_from_slice(lanes)?
        };
        Ok(PackedLaneBuffer {
            buffer,
            zero_copy,
            marker: PhantomData,
        })
    }

    fn shared_buffer(&self, bytes: usize) -> Result<Buffer, MetalCommitError> {
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal output buffer".into(),
            ));
        }
        Ok(self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared))
    }

    fn private_buffer(&self, bytes: usize) -> Result<Buffer, MetalCommitError> {
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal scratch buffer".into(),
            ));
        }
        Ok(self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModePrivate))
    }

    fn validate_buffer_length(&self, bytes: usize) -> Result<(), MetalCommitError> {
        let requested =
            u64::try_from(bytes).map_err(|_| MetalCommitError::ShapeOverflow("buffer bytes"))?;
        let maximum = self.device.max_buffer_length();
        if requested > maximum {
            return Err(MetalCommitError::BufferTooLong { requested, maximum });
        }
        Ok(())
    }

    pub(crate) fn dispatch_onehot(
        &self,
        matrix: &Buffer,
        hot_indices: &[u16],
        mut params: OneHotCommitParams,
        kernel: MetalOneHotKernel,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let blocks_per_threadgroup = self.dispatch_geometry(params, kernel)?;
            params.blocks_per_threadgroup = blocks_per_threadgroup as u64;
            let buffer_start = Instant::now();
            let indices = self.shared_buffer_from_slice(hot_indices)?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("output coefficients"))?;
            if kernel == MetalOneHotKernel::DirectGather
                && params.output_coefficients > u64::from(u32::MAX)
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "one-hot output grid exceeds u32::MAX threads".into(),
                ));
            }
            let output_bytes = output_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("output bytes"))?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita one-hot root commitment");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label(match kernel {
                MetalOneHotKernel::DirectGather => "Akita one-hot gather",
                MetalOneHotKernel::BlockBatched => "Akita one-hot block batch",
                MetalOneHotKernel::PackedFp128D512Panels
                | MetalOneHotKernel::PackedFp128D128Rank3 => {
                    return Err(MetalCommitError::UnsupportedShape(
                        "packed kernel requires packed parameters".into(),
                    ));
                }
            });
            self.encode_onehot(encoder, matrix, &indices, &output, &params, kernel)?;
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: `output` is live shared storage for exactly `output_count`
            // aligned `Fp128Limbs` values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(DispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                panel_gpu_active: gpu,
                panel_gpu_span: gpu,
                reduction_gpu: None,
                command_buffers: 1,
                kernel,
                blocks_per_threadgroup,
                columns_per_threadgroup: 0,
                matrix_block_streams: 0,
                scratch_bytes: 0,
                input_zero_copy: false,
            })
        })
    }

    pub(crate) fn dispatch_fp128_d64_digit_rows<const D: usize>(
        &self,
        matrix: &Buffer,
        digit_vectors: &[&[[i8; D]]],
        retain_quotients: bool,
        params: DigitRowsParams,
    ) -> Result<DigitRowsDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let expected_output = params
                .num_vectors
                .checked_mul(params.num_rows)
                .and_then(|count| count.checked_mul(params.ring_d))
                .ok_or(MetalCommitError::ShapeOverflow("digit-row output"))?;
            let product_count = 1u64 + u64::from(retain_quotients);
            let total_output = params
                .output_coefficients
                .checked_mul(product_count)
                .ok_or(MetalCommitError::ShapeOverflow("digit-row product output"))?;
            let expected_vector_width = usize::try_from(params.num_cols)
                .map_err(|_| MetalCommitError::ShapeOverflow("digit-row column count"))?;
            let expected_vector_count = usize::try_from(params.num_vectors)
                .map_err(|_| MetalCommitError::ShapeOverflow("digit-row vector count"))?;
            let expected_row_count = usize::try_from(params.num_rows)
                .map_err(|_| MetalCommitError::ShapeOverflow("digit-row row count"))?;
            let expected_column_partials = params
                .num_cols
                .div_ceil(FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64);
            let partial_count = params
                .num_vectors
                .checked_mul(params.num_rows)
                .and_then(|count| count.checked_mul(expected_column_partials))
                .and_then(|count| count.checked_mul(params.ring_d))
                .and_then(|count| count.checked_mul(product_count))
                .ok_or(MetalCommitError::ShapeOverflow("digit-row partials"))?;
            let expected_matrix_bytes = params
                .num_rows
                .checked_mul(params.num_cols)
                .and_then(|count| count.checked_mul(params.ring_d))
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64))
                .ok_or(MetalCommitError::ShapeOverflow("digit-row matrix bytes"))?;
            if D != 64
                || params.ring_d != 64
                || params.num_vectors == 0
                || params.num_rows == 0
                || digit_vectors.len() != expected_vector_count
                || digit_vectors
                    .iter()
                    .any(|digits| digits.len() != expected_vector_width)
                || params.output_coefficients != expected_output
                || params.retain_quotients != u64::from(retain_quotients)
                || params.columns_per_partial != FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64
                || params.column_partials != expected_column_partials
                || total_output > u64::from(u32::MAX)
                || params
                    .num_vectors
                    .checked_mul(params.num_rows)
                    .and_then(|count| count.checked_mul(params.column_partials))
                    .is_none_or(|count| count > u64::from(u32::MAX))
                || matrix.length() < expected_matrix_bytes
                || !self.supports_fp128_d64_digit_rows::<D>(
                    expected_vector_count,
                    expected_row_count,
                    expected_vector_width,
                    retain_quotients,
                )
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D64 digit rows exceed the kernel's index or device-buffer limits".into(),
                ));
            }

            let buffer_start = Instant::now();
            let (digit_buffer, output, partials, output_count) = {
                let input_bytes = digit_vectors.iter().try_fold(0usize, |total, digits| {
                    total
                        .checked_add(size_of_val(*digits))
                        .ok_or(MetalCommitError::ShapeOverflow("digit-row input bytes"))
                })?;
                let span = tracing::info_span!(
                    "MetalDigitRows::buffer_setup",
                    input_bytes,
                    num_vectors = expected_vector_count,
                    num_cols = expected_vector_width,
                    num_rows = expected_row_count,
                    ring_dimension = D,
                );
                let _entered = span.enter();
                let digit_buffer = self.shared_buffer_from_digit_rows(digit_vectors)?;
                let output_count = usize::try_from(total_output).map_err(|_| {
                    MetalCommitError::ShapeOverflow("digit-row output coefficients")
                })?;
                let output_bytes = output_count
                    .checked_mul(size_of::<Fp128Limbs>())
                    .ok_or(MetalCommitError::ShapeOverflow("digit-row output bytes"))?;
                let output = self.shared_buffer(output_bytes)?;
                let partial_count = usize::try_from(partial_count)
                    .map_err(|_| MetalCommitError::ShapeOverflow("digit-row partial count"))?;
                let partial_bytes = partial_count
                    .checked_mul(size_of::<Fp128Limbs>())
                    .ok_or(MetalCommitError::ShapeOverflow("digit-row partial bytes"))?;
                let partials = self.private_buffer(partial_bytes)?;
                (digit_buffer, output, partials, output_count)
            };
            let buffer_setup = buffer_start.elapsed();

            let (command_wall, gpu) = {
                let span = tracing::info_span!(
                    "MetalDigitRows::command",
                    num_vectors = expected_vector_count,
                    num_cols = expected_vector_width,
                    num_rows = expected_row_count,
                    ring_dimension = D,
                );
                let _entered = span.enter();
                let command = self.queue.new_command_buffer();
                command.set_label("Akita fp128 D64 digit rows");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 D64 digit-row partials");
                encoder.set_compute_pipeline_state(&self.fp128_d64_digit_rows_partials_pipeline);
                encoder.set_buffer(0, Some(matrix), 0);
                encoder.set_buffer(1, Some(&digit_buffer), 0);
                encoder.set_buffer(2, Some(&partials), 0);
                set_inline_bytes(encoder, 3, &params);
                encoder.dispatch_thread_groups(
                    MTLSize::new(
                        params.num_vectors * params.num_rows * params.column_partials,
                        1,
                        1,
                    ),
                    MTLSize::new(FP128_D64_DIGIT_ROWS_PARTIAL_THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 D64 digit-row reduction");
                encoder.set_compute_pipeline_state(&self.fp128_d64_digit_rows_reduce_pipeline);
                encoder.set_buffer(0, Some(&partials), 0);
                encoder.set_buffer(1, Some(&output), 0);
                set_inline_bytes(encoder, 2, &params);
                encoder.dispatch_thread_groups(
                    MTLSize::new(total_output, 1, 1),
                    MTLSize::new(FP128_D64_DIGIT_ROWS_THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                complete_command(command)?
            };

            let readback_start = Instant::now();
            let coefficients = {
                let span = tracing::info_span!(
                    "MetalDigitRows::readback",
                    output_count,
                    output_bytes = output_count * size_of::<Fp128Limbs>(),
                );
                let _entered = span.enter();
                // SAFETY: `output` is live shared storage for exactly `output_count`
                // aligned `Fp128Limbs` values.
                unsafe {
                    std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                        .to_vec()
                }
            };
            Ok(DigitRowsDispatchOutcome {
                coefficients,
                allocation_bytes: akita_error::checked::sum([
                    digit_buffer.length() as usize,
                    output.length() as usize,
                    partials.length() as usize,
                ])
                .ok_or(MetalCommitError::ShapeOverflow(
                    "digit-row allocation bytes",
                ))?,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
            })
        })
    }

    pub(crate) fn dispatch_fp128_i8_coefficient_packing(
        &self,
        sources: &[&[i8]],
        combined_weights: &[Fp128Limbs],
        params: I8CoefficientPackingParams,
    ) -> Result<CoefficientPackingDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let source_count = usize::try_from(params.num_sources)
                .map_err(|_| MetalCommitError::ShapeOverflow("coefficient-packing sources"))?;
            let source_coefficients = usize::try_from(params.source_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("coefficient-packing source width"))?;
            let weight_count = params
                .positions_per_block
                .checked_mul(params.stride)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "coefficient-packing weights",
                ))?;
            let expected_output = params
                .num_sources
                .checked_mul(params.num_blocks)
                .and_then(|count| count.checked_mul(params.subring_dimension))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "coefficient-packing output",
                ))?;
            if source_count == 0
                || source_coefficients == 0
                || sources.len() != source_count
                || sources
                    .iter()
                    .any(|source| source.len() != source_coefficients)
                || params.live_coefficients > params.source_coefficients
                || params.ring_d == 0
                || params.stride == 0
                || params.subring_dimension == 0
                || params
                    .stride
                    .checked_mul(params.subring_dimension)
                    .is_none_or(|dimension| dimension != params.ring_d)
                || params.num_live_positions == 0
                || params.positions_per_block == 0
                || params.num_blocks == 0
                || params.output_coefficients != expected_output
                || params.output_coefficients > u64::from(u32::MAX)
                || u64::try_from(combined_weights.len()).ok() != Some(weight_count)
                || self
                    .fp128_i8_coefficient_packing_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_COEFFICIENT_PACKING_THREADS as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 i8 coefficient-packing shape is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let source = self.shared_byte_buffer_from_slices(sources)?;
            let weights = self.shared_buffer_from_slice(combined_weights)?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("coefficient-packing output"))?;
            let output_bytes = output_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("coefficient-packing output bytes"),
            )?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 i8 coefficient packing");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_i8_coefficient_packing_pipeline);
            encoder.set_buffer(0, Some(&source.buffer), 0);
            encoder.set_buffer(1, Some(&weights), 0);
            encoder.set_buffer(2, Some(&output), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.output_coefficients, 1, 1),
                MTLSize::new(FP128_COEFFICIENT_PACKING_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: `output` is live shared storage for `output_count`
            // aligned canonical limb values written by the completed command.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            let source_bytes = source_count.checked_mul(source_coefficients).ok_or(
                MetalCommitError::ShapeOverflow("coefficient-packing input bytes"),
            )?;
            let allocation_bytes = output_bytes
                .checked_add(size_of_val(combined_weights))
                .and_then(|bytes| {
                    bytes.checked_add(if source.zero_copy { 0 } else { source_bytes })
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "coefficient-packing allocation bytes",
                ))?;
            Ok(CoefficientPackingDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                allocation_bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_packed_onehot_coefficient_packing(
        &self,
        lanes: &[u8],
        active_zero_rows: &[u64],
        combined_weights: &[Fp128Limbs],
        params: PackedOneHotCoefficientPackingParams,
    ) -> Result<CoefficientPackingDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let expected_lanes = params.num_rows.checked_mul(params.num_columns).ok_or(
                MetalCommitError::ShapeOverflow("packed coefficient-packing lanes"),
            )?;
            let expected_blocks = params
                .column_capacity
                .checked_mul(params.blocks_per_column)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "packed coefficient-packing blocks",
                ))?;
            let expected_output = params
                .num_blocks
                .checked_mul(params.subring_dimension)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "packed coefficient-packing output",
                ))?;
            let expected_weights = params
                .positions_per_block
                .checked_mul(params.stride)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "packed coefficient-packing weights",
                ))?;
            let expected_row_partials = params
                .rows_per_block
                .div_ceil(params.rows_per_partial.max(1));
            let expected_partials = params
                .num_blocks
                .checked_mul(params.row_partials_per_block)
                .and_then(|count| count.checked_mul(params.subring_dimension))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "packed coefficient-packing partials",
                ))?;
            let live_column_mask = if params.num_columns >= u64::BITS as u64 {
                u64::MAX
            } else {
                (1u64 << params.num_columns) - 1
            };
            let expected_active_zero_words = params.num_rows.div_ceil(u64::BITS as u64);
            if params.num_rows == 0
                || params.num_columns == 0
                || params.num_columns > params.column_capacity
                || params.onehot_k == 0
                || params.ring_d == 0
                || params.positions_per_block == 0
                || params.blocks_per_column == 0
                || params.rows_per_block == 0
                || params.rows_per_partial
                    != FP128_PACKED_COEFFICIENT_PACKING_ROWS_PER_PARTIAL as u64
                || params.row_partials_per_block != expected_row_partials
                || params.stride == 0
                || params.subring_dimension == 0
                || params.subring_dimension > FP128_PACKED_COEFFICIENT_PACKING_REDUCE_THREADS as u64
                || params
                    .stride
                    .checked_mul(params.subring_dimension)
                    .is_none_or(|dimension| dimension != params.ring_d)
                || params.num_blocks != expected_blocks
                || params.output_coefficients != expected_output
                || params.partial_coefficients != expected_partials
                || params.output_coefficients > u64::from(u32::MAX)
                || params
                    .num_blocks
                    .checked_mul(params.row_partials_per_block)
                    .is_none_or(|groups| groups > u64::from(u32::MAX))
                || u64::try_from(lanes.len()).ok() != Some(expected_lanes)
                || params.zero_column_mask & !live_column_mask != 0
                || (params.zero_column_mask == 0 && !active_zero_rows.is_empty())
                || (params.zero_column_mask != 0
                    && u64::try_from(active_zero_rows.len()).ok()
                        != Some(expected_active_zero_words))
                || u64::try_from(combined_weights.len()).ok() != Some(expected_weights)
                || self
                    .fp128_packed_onehot_coefficient_packing_partials_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_PACKED_COEFFICIENT_PACKING_PARTIAL_THREADS as u64
                || self
                    .fp128_packed_onehot_coefficient_packing_reduce_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_PACKED_COEFFICIENT_PACKING_REDUCE_THREADS as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 packed one-hot coefficient-packing shape is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_bytes = lanes.len();
            let lanes = self.packed_lane_buffer(lanes)?;
            let no_active_zero_rows = [0u64];
            let active_zero_rows = if active_zero_rows.is_empty() {
                no_active_zero_rows.as_slice()
            } else {
                active_zero_rows
            };
            let active_zero_bytes = size_of_val(active_zero_rows);
            let active_zero_rows = self.shared_buffer_from_slice(active_zero_rows)?;
            let weights = self.shared_buffer_from_slice(combined_weights)?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("coefficient-packing output"))?;
            let output_bytes = output_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("coefficient-packing output bytes"),
            )?;
            let output = self.shared_buffer(output_bytes)?;
            let partial_count = usize::try_from(params.partial_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("coefficient-packing partials"))?;
            let partial_bytes = partial_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("coefficient-packing partial bytes"),
            )?;
            let partials = self.private_buffer(partial_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 packed one-hot coefficient packing");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.fp128_packed_onehot_coefficient_packing_partials_pipeline,
            );
            encoder.set_buffer(0, Some(&lanes.buffer), 0);
            encoder.set_buffer(1, Some(&weights), 0);
            encoder.set_buffer(2, Some(&partials), 0);
            encoder.set_buffer(3, Some(&active_zero_rows), 0);
            set_inline_bytes(encoder, 4, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_blocks * params.row_partials_per_block, 1, 1),
                MTLSize::new(
                    FP128_PACKED_COEFFICIENT_PACKING_PARTIAL_THREADS as u64,
                    1,
                    1,
                ),
            );
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.fp128_packed_onehot_coefficient_packing_reduce_pipeline,
            );
            encoder.set_buffer(0, Some(&partials), 0);
            encoder.set_buffer(1, Some(&output), 0);
            set_inline_bytes(encoder, 2, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_blocks, 1, 1),
                MTLSize::new(FP128_PACKED_COEFFICIENT_PACKING_REDUCE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: the completed command initialized the full shared output.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            let allocation_bytes = output_bytes
                .checked_add(partial_bytes)
                .and_then(|bytes| bytes.checked_add(size_of_val(combined_weights)))
                .and_then(|bytes| bytes.checked_add(active_zero_bytes))
                .and_then(|bytes| bytes.checked_add(if lanes.zero_copy { 0 } else { lane_bytes }))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "coefficient-packing allocation bytes",
                ))?;
            Ok(CoefficientPackingDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                allocation_bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_indexed_packed_onehot_coefficient_packing(
        &self,
        index: &PackedFp128D512CoefficientPackingIndex,
        combined_weights: &[Fp128Limbs],
        params: PackedOneHotCoefficientPackingParams,
    ) -> Result<CoefficientPackingDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let index_params = index.params;
            let expected_blocks = params
                .column_capacity
                .checked_mul(params.blocks_per_column)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed coefficient-packing blocks",
                ))?;
            let expected_output = expected_blocks
                .checked_mul(params.subring_dimension)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed coefficient-packing output",
                ))?;
            let expected_weights = params
                .positions_per_block
                .checked_mul(params.stride)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed coefficient-packing weights",
                ))?;
            let tile_chunks = index_params
                .position_tiles
                .div_ceil(FP128_D512_PACKING_TILES_PER_CHUNK as u64);
            let live_streams = params
                .blocks_per_column
                .checked_mul(params.num_columns)
                .and_then(|count| count.checked_mul(2))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed coefficient-packing live streams",
                ))?;
            let partial_groups =
                live_streams
                    .checked_mul(tile_chunks)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "indexed coefficient-packing partial groups",
                    ))?;
            let partial_count = live_streams
                .checked_mul(32)
                .and_then(|count| count.checked_mul(tile_chunks))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed coefficient-packing partials",
                ))?;
            if params.num_rows == 0
                || params.num_columns == 0
                || params.num_columns > params.column_capacity
                || params.onehot_k != 256
                || params.ring_d != 512
                || params.stride != 8
                || params.subring_dimension != 64
                || params.positions_per_block == 0
                || params.positions_per_block.checked_mul(2) != Some(params.rows_per_block)
                || params.blocks_per_column == 0
                || params.num_blocks != expected_blocks
                || params.output_coefficients != expected_output
                || params.output_coefficients > u64::from(u32::MAX)
                || partial_groups > u64::from(u32::MAX)
                || u64::try_from(combined_weights.len()).ok() != Some(expected_weights)
                || index_params.num_rows != params.num_rows
                || index_params.num_columns != params.num_columns
                || index_params.lane_stride != params.num_columns
                || index_params.num_positions != params.positions_per_block
                || index_params.blocks_per_column != params.blocks_per_column
                || index_params.position_tiles
                    != params
                        .positions_per_block
                        .div_ceil(FP128_D512_PACKING_INDEX_TILE_POSITIONS as u64)
                || self
                    .fp128_d512_indexed_coefficient_packing_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_COEFFICIENT_PACKING_THREADS as u64
                || self
                    .fp128_d512_indexed_coefficient_packing_reduce_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_COEFFICIENT_PACKING_THREADS as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "indexed fp128 packed coefficient-packing shape is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let weights = self.shared_buffer_from_slice(combined_weights)?;
            let output_count = usize::try_from(params.output_coefficients).map_err(|_| {
                MetalCommitError::ShapeOverflow("indexed coefficient-packing output")
            })?;
            let output_bytes = output_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("indexed coefficient-packing output bytes"),
            )?;
            let output = self.shared_buffer(output_bytes)?;
            let partial_count = usize::try_from(partial_count).map_err(|_| {
                MetalCommitError::ShapeOverflow("indexed coefficient-packing partial count")
            })?;
            let partial_bytes = partial_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("indexed coefficient-packing partial bytes"),
            )?;
            let partials = self.private_buffer(partial_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 indexed coefficient packing");
            let encoder = command.new_compute_command_encoder();
            encoder
                .set_compute_pipeline_state(&self.fp128_d512_indexed_coefficient_packing_pipeline);
            encoder.set_buffer(0, Some(&index.records), 0);
            encoder.set_buffer(1, Some(&index.offsets), 0);
            encoder.set_buffer(2, Some(&weights), 0);
            encoder.set_buffer(3, Some(&partials), 0);
            set_inline_bytes(encoder, 4, &index_params);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(partial_groups, 1, 1),
                MTLSize::new(FP128_COEFFICIENT_PACKING_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.fp128_d512_indexed_coefficient_packing_reduce_pipeline,
            );
            encoder.set_buffer(0, Some(&partials), 0);
            encoder.set_buffer(1, Some(&output), 0);
            set_inline_bytes(encoder, 2, &index_params);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_threads(
                MTLSize::new(params.output_coefficients, 1, 1),
                MTLSize::new(FP128_COEFFICIENT_PACKING_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: the completed command initialized exactly `output_count`
            // aligned canonical limb values in shared storage.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(CoefficientPackingDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                allocation_bytes: output_bytes
                    .checked_add(size_of_val(combined_weights))
                    .and_then(|bytes| bytes.checked_add(partial_bytes))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "indexed coefficient-packing allocation bytes",
                    ))?,
            })
        })
    }

    pub(crate) fn dispatch_packed_fp128_d512_decompose_fold_streaming(
        &self,
        lanes: &[u8],
        active_zero_rows: &[u64],
        challenge_positions: &[u16],
        challenge_coefficients: &[i8],
        dense_subring64_challenges: Option<&[i8]>,
        params: PackedDecomposeFoldParams,
        position_chunk_len: usize,
        consume: impl FnMut(usize, &[i32]),
    ) -> Result<PackedDecomposeFoldDispatchOutcome, MetalCommitError> {
        self.dispatch_packed_fp128_decompose_fold_streaming(
            512,
            lanes,
            active_zero_rows,
            challenge_positions,
            challenge_coefficients,
            dense_subring64_challenges,
            params,
            position_chunk_len,
            consume,
        )
    }

    /// Packed decompose-fold at ring dimension 512 (rank-1 row) or 128
    /// (rank-3 row), with sparse or embedded subring-64 challenges.
    pub(crate) fn dispatch_packed_fp128_decompose_fold_streaming(
        &self,
        ring_d: usize,
        lanes: &[u8],
        active_zero_rows: &[u64],
        challenge_positions: &[u16],
        challenge_coefficients: &[i8],
        dense_subring64_challenges: Option<&[i8]>,
        params: PackedDecomposeFoldParams,
        position_chunk_len: usize,
        mut consume: impl FnMut(usize, &[i32]),
    ) -> Result<PackedDecomposeFoldDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let dense = dense_subring64_challenges.is_some();
            let (fold_pipeline, fold_threads) = match (ring_d, dense) {
                (512, false) => (&self.fp128_d512_decompose_fold_pipeline, 256u64),
                (128, false) => (&self.fp128_d128_decompose_fold_pipeline, 128u64),
                (512, true) => (&self.fp128_d512_subring64_decompose_fold_pipeline, 256u64),
                (128, true) => (&self.fp128_d128_subring64_decompose_fold_pipeline, 256u64),
                _ => {
                    return Err(MetalCommitError::UnsupportedShape(
                        "packed decompose-fold supports ring dimensions 512 and 128".into(),
                    ));
                }
            };
            let ring_d_u64 = ring_d as u64;
            let expected_challenge_terms = params
                .num_columns
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(params.challenge_weight))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "decompose-fold challenge terms",
                ))?;
            let expected_lanes = params
                .num_rows
                .checked_mul(params.lane_stride)
                .ok_or(MetalCommitError::ShapeOverflow("decompose-fold lanes"))?;
            let expected_output = params
                .num_positions
                .checked_mul(ring_d_u64)
                .ok_or(MetalCommitError::ShapeOverflow("decompose-fold output"))?;
            let expected_dense_challenges = params
                .num_columns
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(64))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "dense subring challenge coefficients",
                ))?;
            let expected_active_zero_words = params.num_rows.div_ceil(u64::BITS.into());
            let live_column_mask = if params.num_columns >= u64::BITS.into() {
                u64::MAX
            } else {
                (1u64 << params.num_columns) - 1
            };
            if params.num_rows == 0
                || params.num_columns == 0
                || params.num_columns > params.lane_stride
                || (ring_d == 128 && dense && params.num_columns > 32)
                || params.zero_column_mask & !live_column_mask != 0
                || (params.zero_column_mask == 0 && !active_zero_rows.is_empty())
                || (params.zero_column_mask != 0
                    && u64::try_from(active_zero_rows.len()).ok()
                        != Some(expected_active_zero_words))
                || params.challenge_weight == 0
                || params.position_start != 0
                || position_chunk_len == 0
                || params.output_coefficients != expected_output
                || u64::try_from(lanes.len()).ok() != Some(expected_lanes)
                || u64::try_from(challenge_positions.len()).ok() != Some(expected_challenge_terms)
                || challenge_positions.len() != challenge_coefficients.len()
                || dense_subring64_challenges.is_some_and(|dense| {
                    u64::try_from(dense.len()).ok() != Some(expected_dense_challenges)
                        || expected_dense_challenges > u64::from(u32::MAX) + 1
                })
                || params.num_positions > u64::from(u32::MAX)
                || fold_pipeline.max_total_threads_per_threadgroup() < fold_threads
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 packed decompose-fold geometry is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_buffer = self.packed_lane_buffer(lanes)?;
            let no_active_zero_rows = [0u64];
            let active_zero_rows = if active_zero_rows.is_empty() {
                no_active_zero_rows.as_slice()
            } else {
                active_zero_rows
            };
            let active_zero_buffer = self.shared_buffer_from_slice(active_zero_rows)?;
            let positions = self.shared_buffer_from_slice(challenge_positions)?;
            let coefficients = self.shared_buffer_from_slice(challenge_coefficients)?;
            let dense_challenges = dense_subring64_challenges
                .map(|dense| self.shared_buffer_from_slice(dense))
                .transpose()?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("decompose-fold output count"))?;
            let output_bytes = output_count.checked_mul(size_of::<i32>()).ok_or(
                MetalCommitError::ShapeOverflow("decompose-fold output bytes"),
            )?;
            let mut centered_coefficients = vec![0i32; output_count];
            let output_zero_copy = centered_coefficients
                .as_ptr()
                .addr()
                .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
                && output_bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
            let output = if output_zero_copy {
                self.device.new_buffer_with_bytes_no_copy(
                    centered_coefficients.as_mut_ptr().cast::<c_void>(),
                    output_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            } else {
                self.shared_buffer(output_bytes)?
            };
            let buffer_setup = buffer_start.elapsed();

            let total_positions = usize::try_from(params.num_positions)
                .map_err(|_| MetalCommitError::ShapeOverflow("decompose-fold position count"))?;
            let chunk_len = position_chunk_len.min(total_positions);
            let command_start = Instant::now();
            let mut commands = Vec::with_capacity(total_positions.div_ceil(chunk_len));
            for position_start in (0..total_positions).step_by(chunk_len) {
                let position_end = position_start
                    .saturating_add(chunk_len)
                    .min(total_positions);
                let position_count = position_end - position_start;
                let mut command_params = params;
                command_params.position_start = u64::try_from(position_start).map_err(|_| {
                    MetalCommitError::ShapeOverflow("decompose-fold position offset")
                })?;
                let output_offset = position_start
                    .checked_mul(ring_d)
                    .and_then(|count| count.checked_mul(size_of::<i32>()))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "decompose-fold output offset",
                    ))?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita fp128 packed decompose-fold");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 packed decompose-fold");
                encoder.set_compute_pipeline_state(fold_pipeline);
                if let Some(dense_challenges) = dense_challenges.as_ref() {
                    encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
                    encoder.set_buffer(1, Some(dense_challenges), 0);
                    encoder.set_buffer(2, Some(&output), output_offset as u64);
                    set_inline_bytes(encoder, 3, &command_params);
                    encoder.set_buffer(4, Some(&active_zero_buffer), 0);
                } else {
                    encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
                    encoder.set_buffer(1, Some(&positions), 0);
                    encoder.set_buffer(2, Some(&coefficients), 0);
                    encoder.set_buffer(3, Some(&output), output_offset as u64);
                    set_inline_bytes(encoder, 4, &command_params);
                    encoder.set_buffer(5, Some(&active_zero_buffer), 0);
                }
                encoder.dispatch_thread_groups(
                    MTLSize::new(position_count as u64, 1, 1),
                    MTLSize::new(fold_threads, 1, 1),
                );
                encoder.end_encoding();
                command.commit();
                commands.push((command, position_start, position_end));
            }

            let mut readback_copy = Duration::ZERO;
            let mut consumer_time = Duration::ZERO;
            for (command, position_start, position_end) in &commands {
                command.wait_until_completed();
                validate_completed_command(command)?;
                let coefficient_start = position_start * ring_d;
                let coefficient_end = position_end * ring_d;
                if !output_zero_copy {
                    let readback_start = Instant::now();
                    // SAFETY: this completed command initialized the disjoint
                    // coefficient range copied into equally sized host storage.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            output.contents().cast::<i32>().add(coefficient_start),
                            centered_coefficients.as_mut_ptr().add(coefficient_start),
                            coefficient_end - coefficient_start,
                        );
                    }
                    readback_copy += readback_start.elapsed();
                }
                let consumer_start = Instant::now();
                consume(
                    *position_start,
                    &centered_coefficients[coefficient_start..coefficient_end],
                );
                consumer_time += consumer_start.elapsed();
            }
            let command_wall = command_start.elapsed();
            let gpu = commands.first().and_then(|(first, _, _)| {
                commands
                    .last()
                    .and_then(|(last, _, _)| completed_commands_gpu_span(first, last))
            });
            let allocation_bytes = output_bytes
                .checked_add(size_of_val(challenge_positions))
                .and_then(|bytes| bytes.checked_add(size_of_val(challenge_coefficients)))
                .and_then(|bytes| bytes.checked_add(size_of_val(active_zero_rows)))
                .and_then(|bytes| {
                    bytes.checked_add(dense_subring64_challenges.map_or(0, size_of_val))
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "decompose-fold allocation bytes",
                ))?;
            Ok(PackedDecomposeFoldDispatchOutcome {
                centered_coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                consumer_time,
                allocation_bytes,
            })
        })
    }

    pub(crate) fn prepare_packed_fp128_d512_fold_index(
        &self,
        lanes: &[u8],
        params: PackedFoldIndexParams,
    ) -> Result<PackedFp128D512FoldIndex, MetalCommitError> {
        autoreleasepool(|| {
            validate_packed_fold_index_geometry(params, lanes.len())?;
            if params.fold_digits != 0
                || params.fold_log_basis != 0
                || self
                    .fp128_d512_build_fold_index_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_D512_FOLD_INDEX_TILE_TASKS as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 packed fold-index geometry is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_buffer = self.packed_lane_buffer(lanes)?;
            let record_count = usize::try_from(params.record_slots)
                .map_err(|_| MetalCommitError::ShapeOverflow("fold-index record count"))?;
            let record_bytes = record_count
                .checked_mul(size_of::<u32>())
                .ok_or(MetalCommitError::ShapeOverflow("fold-index record bytes"))?;
            let count_count = usize::try_from(params.count_entries)
                .map_err(|_| MetalCommitError::ShapeOverflow("fold-index count count"))?;
            let count_bytes = count_count
                .checked_mul(size_of::<u16>())
                .ok_or(MetalCommitError::ShapeOverflow("fold-index count bytes"))?;
            let records = self.private_buffer(record_bytes)?;
            let counts = self.private_buffer(count_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 D512 packed fold index");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 D512 packed fold index");
            encoder.set_compute_pipeline_state(&self.fp128_d512_build_fold_index_pipeline);
            encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&records), 0);
            encoder.set_buffer(2, Some(&counts), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_positions, 1, 1),
                MTLSize::new(FP128_D512_FOLD_INDEX_TILE_TASKS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let allocation_bytes =
                record_bytes
                    .checked_add(count_bytes)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "fold-index allocation bytes",
                    ))?;
            Ok(PackedFp128D512FoldIndex {
                records,
                counts,
                params,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: Duration::ZERO,
                },
                allocation_bytes,
            })
        })
    }

    pub(crate) fn prepare_packed_fp128_d512_coefficient_packing_index(
        &self,
        lanes: &[u8],
        params: PackedCoefficientPackingIndexParams,
    ) -> Result<PackedFp128D512CoefficientPackingIndex, MetalCommitError> {
        autoreleasepool(|| {
            let expected_lanes = params
                .num_rows
                .checked_mul(params.lane_stride)
                .ok_or(MetalCommitError::ShapeOverflow("packing-index lanes"))?;
            let expected_tiles = params
                .num_positions
                .div_ceil(FP128_D512_PACKING_INDEX_TILE_POSITIONS as u64);
            let expected_streams = params
                .blocks_per_column
                .checked_mul(params.num_columns)
                .and_then(|count| count.checked_mul(2))
                .ok_or(MetalCommitError::ShapeOverflow("packing-index streams"))?;
            let expected_layouts = expected_streams
                .checked_mul(expected_tiles)
                .ok_or(MetalCommitError::ShapeOverflow("packing-index layouts"))?;
            let expected_records = expected_layouts
                .checked_mul(FP128_D512_PACKING_INDEX_TILE_POSITIONS as u64)
                .ok_or(MetalCommitError::ShapeOverflow("packing-index records"))?;
            let expected_offsets = expected_layouts
                .checked_mul(FP128_D512_PACKING_INDEX_BUCKET_OFFSETS as u64)
                .ok_or(MetalCommitError::ShapeOverflow("packing-index offsets"))?;
            let expected_groups = params
                .blocks_per_column
                .checked_mul(expected_tiles)
                .ok_or(MetalCommitError::ShapeOverflow("packing-index groups"))?;
            if params.num_rows == 0
                || params.num_columns == 0
                || params.num_columns > 32
                || params.num_columns > params.lane_stride
                || params.num_positions == 0
                || params.blocks_per_column == 0
                || params.position_tiles != expected_tiles
                || params.record_slots != expected_records
                || params.offset_entries != expected_offsets
                || u64::try_from(lanes.len()).ok() != Some(expected_lanes)
                || expected_groups > u64::from(u32::MAX)
                || self
                    .fp128_d512_build_coefficient_packing_index_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_D512_PACKING_INDEX_TILE_POSITIONS as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 coefficient-packing index geometry is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_buffer = self.packed_lane_buffer(lanes)?;
            let record_count = usize::try_from(params.record_slots)
                .map_err(|_| MetalCommitError::ShapeOverflow("packing-index record count"))?;
            let record_bytes = record_count.checked_mul(size_of::<u16>()).ok_or(
                MetalCommitError::ShapeOverflow("packing-index record bytes"),
            )?;
            let offset_count = usize::try_from(params.offset_entries)
                .map_err(|_| MetalCommitError::ShapeOverflow("packing-index offset count"))?;
            let offset_bytes = offset_count.checked_mul(size_of::<u16>()).ok_or(
                MetalCommitError::ShapeOverflow("packing-index offset bytes"),
            )?;
            let records = self.private_buffer(record_bytes)?;
            let offsets = self.private_buffer(offset_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 D512 coefficient-packing index");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.fp128_d512_build_coefficient_packing_index_pipeline,
            );
            encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&records), 0);
            encoder.set_buffer(2, Some(&offsets), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(expected_groups, 1, 1),
                MTLSize::new(FP128_D512_PACKING_INDEX_TILE_POSITIONS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            Ok(PackedFp128D512CoefficientPackingIndex {
                records,
                offsets,
                params,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: Duration::ZERO,
                },
                allocation_bytes: record_bytes.checked_add(offset_bytes).ok_or(
                    MetalCommitError::ShapeOverflow("packing-index allocation bytes"),
                )?,
            })
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the packed root dispatch keeps source, challenge, digit, and streaming boundaries explicit"
    )]
    pub(crate) fn dispatch_packed_fp128_d512_subring64_decompose_fold_streaming(
        &self,
        lanes: &[u8],
        dense_subring64_challenges: &[i8],
        source: PackedFp128D512FoldSource<'_>,
        params: PackedDecomposeFoldParams,
        fold_digits: usize,
        fold_log_basis: u32,
        position_chunk_len: usize,
        mut consume: impl FnMut(usize, &[i32], &[i8]),
    ) -> Result<PackedDecomposeFoldDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let expected_dense = params
                .num_columns
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(64))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed fold dense challenges",
                ))?;
            let expected_output = params
                .num_positions
                .checked_mul(512)
                .ok_or(MetalCommitError::ShapeOverflow("indexed fold output"))?;
            let index_params = match source {
                PackedFp128D512FoldSource::Retained(index) => index.params,
                PackedFp128D512FoldSource::Fused(params) => params,
            };
            validate_packed_fold_index_geometry(index_params, lanes.len())?;
            let max_threads = match source {
                PackedFp128D512FoldSource::Retained(_) => self
                    .fp128_d512_indexed_subring64_decompose_fold_pipeline
                    .max_total_threads_per_threadgroup(),
                PackedFp128D512FoldSource::Fused(_) => self
                    .fp128_d512_fused_subring64_decompose_fold_pipeline
                    .max_total_threads_per_threadgroup(),
            };
            if params.num_rows == 0
                || params.num_columns == 0
                || params.num_columns > params.lane_stride
                || params.position_start != 0
                || params.output_coefficients != expected_output
                || fold_digits == 0
                || !(1..=8).contains(&fold_log_basis)
                || position_chunk_len == 0
                || u64::try_from(dense_subring64_challenges.len()).ok() != Some(expected_dense)
                || index_params.num_rows != params.num_rows
                || index_params.num_columns != params.num_columns
                || index_params.lane_stride != params.lane_stride
                || index_params.num_positions != params.num_positions
                || index_params.blocks_per_column != params.blocks_per_column
                || max_threads < 256
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 indexed fold geometry is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_buffer = match source {
                PackedFp128D512FoldSource::Retained(_) => None,
                PackedFp128D512FoldSource::Fused(_) => Some(self.packed_lane_buffer(lanes)?),
            };
            let packed_challenges = pack_biased_subring64_challenges(dense_subring64_challenges)?;
            let packed_challenges_buffer = self.shared_buffer_from_slice(&packed_challenges)?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("indexed fold output count"))?;
            let output_bytes = output_count
                .checked_mul(size_of::<i32>())
                .ok_or(MetalCommitError::ShapeOverflow("indexed fold output bytes"))?;
            let digit_bytes = output_count
                .checked_mul(fold_digits)
                .ok_or(MetalCommitError::ShapeOverflow("indexed fold digit bytes"))?;
            let mut centered_coefficients = vec![0i32; output_count];
            let output_zero_copy = centered_coefficients
                .as_ptr()
                .addr()
                .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
                && output_bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
            let output = if output_zero_copy {
                self.device.new_buffer_with_bytes_no_copy(
                    centered_coefficients.as_mut_ptr().cast::<c_void>(),
                    output_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            } else {
                self.shared_buffer(output_bytes)?
            };
            let digits = self.shared_buffer(digit_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let total_positions = usize::try_from(params.num_positions)
                .map_err(|_| MetalCommitError::ShapeOverflow("indexed fold positions"))?;
            let chunk_len = position_chunk_len.min(total_positions);
            let command_start = Instant::now();
            let mut commands = Vec::with_capacity(total_positions.div_ceil(chunk_len));
            for position_start in (0..total_positions).step_by(chunk_len) {
                let position_end = position_start
                    .saturating_add(chunk_len)
                    .min(total_positions);
                let position_count = position_end - position_start;
                let mut command_params = index_params;
                command_params.position_start = u64::try_from(position_start)
                    .map_err(|_| MetalCommitError::ShapeOverflow("indexed fold position offset"))?;
                command_params.output_coefficients = params.output_coefficients;
                command_params.fold_digits = u64::try_from(fold_digits)
                    .map_err(|_| MetalCommitError::ShapeOverflow("indexed fold digit count"))?;
                command_params.fold_log_basis = u64::from(fold_log_basis);
                let output_offset = position_start
                    .checked_mul(512)
                    .and_then(|count| count.checked_mul(size_of::<i32>()))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "indexed fold output offset",
                    ))?;
                let digit_offset = position_start
                    .checked_mul(512)
                    .and_then(|count| count.checked_mul(fold_digits))
                    .ok_or(MetalCommitError::ShapeOverflow("indexed fold digit offset"))?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita fp128 D512 packed subring64 decompose-fold");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 D512 packed subring64 decompose-fold");
                match source {
                    PackedFp128D512FoldSource::Retained(index) => {
                        encoder.set_compute_pipeline_state(
                            &self.fp128_d512_indexed_subring64_decompose_fold_pipeline,
                        );
                        encoder.set_buffer(0, Some(&index.records), 0);
                        encoder.set_buffer(1, Some(&index.counts), 0);
                        encoder.set_buffer(2, Some(&packed_challenges_buffer), 0);
                        encoder.set_buffer(3, Some(&output), output_offset as u64);
                        encoder.set_buffer(4, Some(&digits), digit_offset as u64);
                        set_inline_bytes(encoder, 5, &command_params);
                    }
                    PackedFp128D512FoldSource::Fused(_) => {
                        let lane_buffer = lane_buffer.as_ref().ok_or_else(|| {
                            MetalCommitError::UnsupportedShape(
                                "fused packed fold is missing its lane buffer".into(),
                            )
                        })?;
                        encoder.set_compute_pipeline_state(
                            &self.fp128_d512_fused_subring64_decompose_fold_pipeline,
                        );
                        encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
                        encoder.set_buffer(1, Some(&packed_challenges_buffer), 0);
                        encoder.set_buffer(2, Some(&output), output_offset as u64);
                        encoder.set_buffer(3, Some(&digits), digit_offset as u64);
                        set_inline_bytes(encoder, 4, &command_params);
                    }
                }
                encoder.dispatch_thread_groups(
                    MTLSize::new(position_count as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                encoder.end_encoding();
                command.commit();
                commands.push((command, position_start, position_end));
            }

            let mut readback_copy = Duration::ZERO;
            let mut consumer_time = Duration::ZERO;
            for (command, position_start, position_end) in &commands {
                command.wait_until_completed();
                validate_completed_command(command)?;
                let coefficient_start = position_start * 512;
                let coefficient_end = position_end * 512;
                if !output_zero_copy {
                    let readback_start = Instant::now();
                    // SAFETY: this completed command initialized the disjoint output range.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            output.contents().cast::<i32>().add(coefficient_start),
                            centered_coefficients.as_mut_ptr().add(coefficient_start),
                            coefficient_end - coefficient_start,
                        );
                    }
                    readback_copy += readback_start.elapsed();
                }
                let digit_start = coefficient_start * fold_digits;
                let digit_end = coefficient_end * fold_digits;
                // SAFETY: this completed command initialized the matching digit range.
                let digit_slice = unsafe {
                    std::slice::from_raw_parts(
                        digits.contents().cast::<i8>().add(digit_start),
                        digit_end - digit_start,
                    )
                };
                let consumer_start = Instant::now();
                consume(
                    *position_start,
                    &centered_coefficients[coefficient_start..coefficient_end],
                    digit_slice,
                );
                consumer_time += consumer_start.elapsed();
            }
            let command_wall = command_start.elapsed();
            let gpu = commands.first().and_then(|(first, _, _)| {
                commands
                    .last()
                    .and_then(|(last, _, _)| completed_commands_gpu_span(first, last))
            });
            let allocation_bytes = output_bytes
                .checked_add(digit_bytes)
                .and_then(|bytes| bytes.checked_add(size_of_val(packed_challenges.as_slice())))
                .and_then(|bytes| {
                    bytes.checked_add(
                        lane_buffer
                            .as_ref()
                            .map_or(0, |buffer| usize::from(!buffer.zero_copy) * lanes.len()),
                    )
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed fold allocation bytes",
                ))?;
            Ok(PackedDecomposeFoldDispatchOutcome {
                centered_coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                consumer_time,
                allocation_bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_d512_linear_relation<const D: usize>(
        &self,
        matrix: &Buffer,
        rhs: &[[i32; D]],
        params: D512LinearRelationParams,
    ) -> Result<D512LinearRelationDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let num_columns = usize::try_from(params.num_columns)
                .map_err(|_| MetalCommitError::ShapeOverflow("D512 relation columns"))?;
            let num_tiles = usize::try_from(params.num_tiles)
                .map_err(|_| MetalCommitError::ShapeOverflow("D512 relation tiles"))?;
            let expected_tiles = num_columns.div_ceil(FP128_D512_LINEAR_RELATION_COLUMNS_PER_TILE);
            let expected_matrix_bytes = num_columns
                .checked_mul(512)
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow("D512 relation matrix"))?;
            if D != 512
                || rhs.len() != num_columns
                || params.columns_per_tile != FP128_D512_LINEAR_RELATION_COLUMNS_PER_TILE as u64
                || num_tiles != expected_tiles
                || params.num_primes != FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64
                || params.ntt_size != FP128_D512_LINEAR_RELATION_NTT_SIZE as u64
                || params.output_coefficients != 512
                || matrix.length() < expected_matrix_bytes as u64
                || !self.supports_fp128_d512_linear_relation(num_columns, params.rhs_abs_bound)
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 linear relation exceeds the exact CRT or device limits".into(),
                ));
            }

            let buffer_start = Instant::now();
            let rhs_buffer = self.shared_slice_buffer(rhs)?;
            let partial_count = num_tiles
                .checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES)
                .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NTT_SIZE))
                .ok_or(MetalCommitError::ShapeOverflow("D512 relation partials"))?;
            let partial_bytes = partial_count.checked_mul(size_of::<i32>()).ok_or(
                MetalCommitError::ShapeOverflow("D512 relation partial bytes"),
            )?;
            let partials = self.private_buffer(partial_bytes)?;
            let residue_count = FP128_D512_LINEAR_RELATION_NUM_PRIMES
                .checked_mul(FP128_D512_LINEAR_RELATION_NTT_SIZE)
                .ok_or(MetalCommitError::ShapeOverflow("D512 relation residues"))?;
            let residue_bytes = residue_count.checked_mul(size_of::<u32>()).ok_or(
                MetalCommitError::ShapeOverflow("D512 relation residue bytes"),
            )?;
            let residues = self.private_buffer(residue_bytes)?;
            let output_bytes = 512usize
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("D512 relation output"))?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let resources = &self.fp128_d512_linear_relation_resources;
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 D512 linear relation");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita D512 linear relation tiled NTT");
            encoder.set_compute_pipeline_state(&self.fp128_d512_linear_relation_partials_pipeline);
            encoder.set_buffer(0, Some(matrix), 0);
            encoder.set_buffer(1, Some(&rhs_buffer.buffer), 0);
            encoder.set_buffer(2, Some(&partials), 0);
            encoder.set_buffer(3, Some(&resources.primes), 0);
            encoder.set_buffer(4, Some(&resources.limb_weights), 0);
            encoder.set_buffer(5, Some(&resources.field_moduli), 0);
            encoder.set_buffer(6, Some(&resources.fwd_twiddles), 0);
            set_inline_bytes(encoder, 7, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_tiles * params.num_primes, 1, 1),
                MTLSize::new(FP128_D512_LINEAR_RELATION_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita D512 linear relation reduction");
            encoder.set_compute_pipeline_state(&self.fp128_d512_linear_relation_reduce_pipeline);
            encoder.set_buffer(0, Some(&partials), 0);
            encoder.set_buffer(1, Some(&residues), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.inv_twiddles), 0);
            encoder.set_buffer(4, Some(&resources.d_inv), 0);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_primes, 1, 1),
                MTLSize::new(FP128_D512_LINEAR_RELATION_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita D512 linear relation CRT reconstruction");
            encoder
                .set_compute_pipeline_state(&self.fp128_d512_linear_relation_reconstruct_pipeline);
            encoder.set_buffer(0, Some(&residues), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.garner_gamma), 0);
            encoder.set_buffer(4, Some(&resources.field_partial_products), 0);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(FP128_D512_LINEAR_RELATION_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: the shared output contains exactly 512 initialized fp128 limbs.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), 512).to_vec()
            };
            let readback_copy = readback_start.elapsed();
            let rhs_bytes = size_of_val(rhs);
            let allocation_bytes = partial_bytes
                .checked_add(residue_bytes)
                .and_then(|bytes| bytes.checked_add(output_bytes))
                .and_then(|bytes| {
                    bytes.checked_add(if rhs_buffer.zero_copy { 0 } else { rhs_bytes })
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "D512 relation allocation bytes",
                ))?;
            Ok(D512LinearRelationDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes,
            })
        })
    }
    pub(crate) fn prepare_fp128_recursive_commit_matrix<const D: usize>(
        &self,
        matrix: &Buffer,
        params: RecursiveCommitParams,
    ) -> Result<RecursiveCommitMatrixNttOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let num_rows = usize::try_from(params.num_rows)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive commit rows"))?;
            let num_cols = usize::try_from(params.num_cols)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive commit columns"))?;
            let expected_matrix_rings =
                num_rows
                    .checked_mul(num_cols)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "recursive commit matrix rings",
                    ))?;
            let expected_matrix_bytes = expected_matrix_rings
                .checked_mul(D)
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit matrix bytes",
                ))?;
            if params.ring_d != D as u64
                || params.num_primes != FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64
                || params.matrix_rings != expected_matrix_rings as u64
                || matrix.length() < expected_matrix_bytes as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 recursive commitment matrix geometry is unsupported".into(),
                ));
            }
            let resources = self.recursive_commit_resources(D).ok_or_else(|| {
                MetalCommitError::UnsupportedShape(format!(
                    "no recursive commitment resources for D={D}"
                ))
            })?;

            let buffer_start = Instant::now();
            let matrix_ntt_count = expected_matrix_rings
                .checked_mul(D)
                .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit transformed matrix",
                ))?;
            let matrix_ntt_bytes = matrix_ntt_count.checked_mul(size_of::<i32>()).ok_or(
                MetalCommitError::ShapeOverflow("recursive commit transformed matrix bytes"),
            )?;
            let matrix_ntt = self.private_buffer(matrix_ntt_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 recursive commitment matrix NTT");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita recursive commitment matrix NTT");
            encoder.set_compute_pipeline_state(&self.fp128_recursive_commit_matrix_ntt_pipeline);
            encoder.set_buffer(0, Some(matrix), 0);
            encoder.set_buffer(1, Some(&matrix_ntt), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.limb_weights), 0);
            encoder.set_buffer(4, Some(&resources.field_moduli), 0);
            encoder.set_buffer(5, Some(&resources.fwd_twiddles), 0);
            encoder.set_buffer(6, Some(&resources.psi_pows), 0);
            set_inline_bytes(encoder, 7, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.matrix_rings * params.num_primes, 1, 1),
                MTLSize::new(params.ring_d, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            Ok(RecursiveCommitMatrixNttOutcome {
                buffer: matrix_ntt,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: Duration::ZERO,
                },
                allocation_bytes: matrix_ntt_bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_recursive_commit<const D: usize>(
        &self,
        matrix_ntt: &Buffer,
        digits: &[i8],
        params: RecursiveCommitParams,
    ) -> Result<RecursiveCommitDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let num_blocks = usize::try_from(params.num_blocks)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive commit blocks"))?;
            let num_rows = usize::try_from(params.num_rows)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive commit rows"))?;
            let num_cols = usize::try_from(params.num_cols)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive commit columns"))?;
            let expected_matrix_rings =
                num_rows
                    .checked_mul(num_cols)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "recursive commit matrix rings",
                    ))?;
            let expected_source_bytes = num_blocks
                .checked_mul(num_cols)
                .and_then(|count| count.checked_mul(D))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit source bytes",
                ))?;
            let expected_output = num_blocks
                .checked_mul(num_rows)
                .and_then(|count| count.checked_mul(D))
                .ok_or(MetalCommitError::ShapeOverflow("recursive commit output"))?;
            let expected_matrix_ntt_bytes = expected_matrix_rings
                .checked_mul(D)
                .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
                .and_then(|count| count.checked_mul(size_of::<i32>()))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit transformed matrix bytes",
                ))?;
            let expected_block_groups =
                num_blocks.div_ceil(FP128_RECURSIVE_COMMIT_BLOCKS_PER_GROUP);
            if params.blocks_per_group != FP128_RECURSIVE_COMMIT_BLOCKS_PER_GROUP as u64
                || params.num_block_groups != expected_block_groups as u64
                || params.ring_d != D as u64
                || params.num_primes != FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64
                || params.matrix_rings != expected_matrix_rings as u64
                || params.output_coefficients != expected_output as u64
                || digits.len() < expected_source_bytes
                || matrix_ntt.length() < expected_matrix_ntt_bytes as u64
                || !self.supports_fp128_recursive_commit::<D>(
                    num_blocks,
                    num_rows,
                    num_cols,
                    params.rhs_abs_bound,
                )
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 recursive commitment exceeds the exact CRT or device limits".into(),
                ));
            }
            let resources = self.recursive_commit_resources(D).ok_or_else(|| {
                MetalCommitError::UnsupportedShape(format!(
                    "no recursive commitment resources for D={D}"
                ))
            })?;

            let buffer_start = Instant::now();
            let digit_buffer = self.shared_slice_buffer(digits)?;
            let residue_count = expected_output
                .checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES)
                .ok_or(MetalCommitError::ShapeOverflow("recursive commit residues"))?;
            let residue_bytes = residue_count.checked_mul(size_of::<u32>()).ok_or(
                MetalCommitError::ShapeOverflow("recursive commit residue bytes"),
            )?;
            let residues = self.private_buffer(residue_bytes)?;
            let output_bytes = expected_output.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("recursive commit output bytes"),
            )?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 recursive witness commitment");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita recursive commitment exact matvec");
            encoder.set_compute_pipeline_state(&self.fp128_recursive_commit_matvec_pipeline);
            encoder.set_buffer(0, Some(&digit_buffer.buffer), 0);
            encoder.set_buffer(1, Some(matrix_ntt), 0);
            encoder.set_buffer(2, Some(&residues), 0);
            encoder.set_buffer(3, Some(&resources.primes), 0);
            encoder.set_buffer(4, Some(&resources.fwd_twiddles), 0);
            encoder.set_buffer(5, Some(&resources.inv_twiddles), 0);
            encoder.set_buffer(6, Some(&resources.psi_pows), 0);
            encoder.set_buffer(7, Some(&resources.inverse_scale), 0);
            set_inline_bytes(encoder, 8, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_block_groups * params.num_primes, 1, 1),
                MTLSize::new(FP128_RECURSIVE_COMMIT_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita recursive commitment CRT reconstruction");
            encoder.set_compute_pipeline_state(&self.fp128_recursive_commit_reconstruct_pipeline);
            encoder.set_buffer(0, Some(&residues), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.garner_gamma), 0);
            encoder.set_buffer(4, Some(&resources.field_partial_products), 0);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    params
                        .output_coefficients
                        .div_ceil(FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS as u64),
                    1,
                    1,
                ),
                MTLSize::new(FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: the shared output contains exactly `expected_output`
            // initialized, aligned fp128 limb values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), expected_output)
                    .to_vec()
            };
            let readback_copy = readback_start.elapsed();
            let allocation_bytes = residue_bytes
                .checked_add(output_bytes)
                .and_then(|bytes| {
                    bytes.checked_add(if digit_buffer.zero_copy {
                        0
                    } else {
                        size_of_val(digits)
                    })
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit allocation bytes",
                ))?;
            Ok(RecursiveCommitDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes,
            })
        })
    }

    #[cfg(test)]
    pub(crate) fn dispatch_fp128_blake2b_sumcheck_challenge(
        &self,
        chaining_value: &[u8; 64],
        prior_squeezed_bytes: usize,
        claim: Option<Fp128Limbs>,
        coefficients: &[Fp128Limbs],
    ) -> Result<Blake2bSumcheckChallengeOutcome, MetalCommitError> {
        if coefficients.is_empty() || coefficients.len() > 4 {
            return Err(MetalCommitError::UnsupportedShape(
                "Blake2b sumcheck challenge requires one to four coefficients".into(),
            ));
        }
        autoreleasepool(|| {
            let include_claim = claim.is_some();
            let state = self.shared_buffer_from_slice(chaining_value)?;
            let claim = self.shared_buffer_from_slice(&[claim.unwrap_or_default()])?;
            let coefficients = self.shared_buffer_from_slice(coefficients)?;
            let challenge = self.shared_buffer(size_of::<Fp128Limbs>())?;
            let params = Blake2bSumcheckChallengeParams {
                include_claim: u64::from(include_claim),
                coefficient_count: coefficients.length() / 16,
                prior_squeezed_bytes: u64::try_from(prior_squeezed_bytes)
                    .map_err(|_| MetalCommitError::ShapeOverflow("Blake2b prior squeeze length"))?,
                reserved: 0,
            };
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 Blake2b sumcheck challenge");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_blake2b_sumcheck_challenge_pipeline);
            encoder.set_buffer(0, Some(&state), 0);
            encoder.set_buffer(1, Some(&claim), 0);
            encoder.set_buffer(2, Some(&coefficients), 0);
            encoder.set_buffer(3, Some(&challenge), 0);
            set_inline_bytes(encoder, 4, &params);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            let _ = complete_command(command)?;

            // SAFETY: both shared output buffers were initialized by the completed kernel.
            let challenge = unsafe { *challenge.contents().cast::<Fp128Limbs>() };
            let mut next_chaining_value = [0u8; 64];
            // SAFETY: `state` is a 64-byte shared buffer valid for the destination length.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    state.contents().cast::<u8>(),
                    next_chaining_value.as_mut_ptr(),
                    next_chaining_value.len(),
                );
            }
            Ok(Blake2bSumcheckChallengeOutcome {
                challenge,
                chaining_value: next_chaining_value,
            })
        })
    }

    #[cfg(test)]
    pub(crate) fn dispatch_fp128_direct_range_resident(
        &self,
        session: &mut DirectRangeSession,
        equality_schedule: &[(Vec<Fp128Limbs>, Vec<Fp128Limbs>)],
        basis: usize,
        chaining_value: &[u8; 64],
        prior_squeezed_bytes: usize,
    ) -> Result<DirectRangeResidentOutcome, MetalCommitError> {
        let num_rounds = equality_schedule.len();
        if num_rounds == 0
            || num_rounds != session.current_len.trailing_zeros() as usize
            || session.current_table.is_some()
            || session.rounds_folded != 0
        {
            return Err(MetalCommitError::UnsupportedShape(
                "resident direct range session has malformed initial state".into(),
            ));
        }
        let coefficient_count = match basis {
            4 => 2usize,
            8 => 4usize,
            _ => {
                return Err(MetalCommitError::UnsupportedShape(
                    "resident direct range proof supports basis four or eight".into(),
                ))
            }
        };

        autoreleasepool(|| {
            let setup_start = Instant::now();
            let equality_buffers = equality_schedule
                .iter()
                .map(|(first, second)| {
                    Ok((
                        self.shared_buffer_from_slice(first)?,
                        self.shared_buffer_from_slice(second)?,
                    ))
                })
                .collect::<Result<Vec<_>, MetalCommitError>>()?;
            let round_output_bytes = num_rounds
                .checked_mul(FP128_DIRECT_RANGE_STORED_COEFFICIENTS)
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct range round outputs",
                ))?;
            let challenge_bytes = num_rounds.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("resident direct range challenges"),
            )?;
            let round_outputs = self.shared_buffer(round_output_bytes)?;
            let challenges = self.shared_buffer(challenge_bytes)?;
            let state = self.shared_buffer_from_slice(chaining_value)?;
            let claim = self.shared_buffer_from_slice(&[Fp128Limbs::default()])?;
            let buffer_setup = setup_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 resident direct range proof");
            let (first, second) = &equality_schedule[0];
            let mut params = direct_range_params(
                session.live_len,
                session.current_len,
                session.current_live_len,
                session.current_live_len,
                first,
                second,
                basis,
            )?;
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 resident direct range initial partials");
            encoder.set_compute_pipeline_state(&self.fp128_direct_range_initial_pipeline);
            encoder.set_buffer(0, Some(&session.compact_digits), 0);
            encoder.set_buffer(1, Some(&equality_buffers[0].0), 0);
            encoder.set_buffer(2, Some(&equality_buffers[0].1), 0);
            encoder.set_buffer(3, Some(&session.partials), 0);
            set_inline_bytes(encoder, 4, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.workgroups, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            encode_direct_range_reduction_at_offset(
                command,
                &self.fp128_direct_range_reduce_pipeline,
                &session.partials,
                &round_outputs,
                0,
                &params,
            );

            let mut current_len = session.current_len;
            let mut current_live_len = session.current_live_len;
            let mut current_table = session.current_table;
            let mut rounds_folded = session.rounds_folded;
            for round in 0..num_rounds {
                let transcript_params = Blake2bSumcheckChallengeParams {
                    include_claim: u64::from(round == 0),
                    coefficient_count: coefficient_count as u64,
                    prior_squeezed_bytes: if round == 0 {
                        u64::try_from(prior_squeezed_bytes).map_err(|_| {
                            MetalCommitError::ShapeOverflow(
                                "resident direct range prior squeeze length",
                            )
                        })?
                    } else {
                        32
                    },
                    reserved: 0,
                };
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 resident direct range challenge");
                encoder.set_compute_pipeline_state(&self.fp128_blake2b_sumcheck_challenge_pipeline);
                encoder.set_buffer(0, Some(&state), 0);
                encoder.set_buffer(1, Some(&claim), 0);
                encoder.set_buffer(
                    2,
                    Some(&round_outputs),
                    (round * FP128_DIRECT_RANGE_STORED_COEFFICIENTS * size_of::<Fp128Limbs>())
                        as u64,
                );
                encoder.set_buffer(
                    3,
                    Some(&challenges),
                    (round * size_of::<Fp128Limbs>()) as u64,
                );
                set_inline_bytes(encoder, 4, &transcript_params);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();

                let next_len = current_len / 2;
                let next_live_len = current_live_len.div_ceil(2);
                if next_len == 1 {
                    let table = current_table.ok_or_else(|| {
                        MetalCommitError::UnsupportedShape(
                            "resident direct range compact prefix reaches final fold".into(),
                        )
                    })?;
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 resident direct range final fold");
                    encoder.set_compute_pipeline_state(&self.fp128_direct_range_finalize_pipeline);
                    encoder.set_buffer(0, Some(&session.tables[table]), 0);
                    encoder.set_buffer(1, Some(&session.final_output), 0);
                    encoder.set_buffer(
                        2,
                        Some(&challenges),
                        (round * size_of::<Fp128Limbs>()) as u64,
                    );
                    set_inline_bytes(encoder, 3, &(current_live_len as u64));
                    encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                    encoder.end_encoding();
                    current_len = 1;
                    current_live_len = next_live_len;
                    break;
                }

                let (first, second) = &equality_schedule[round + 1];
                params = direct_range_params(
                    session.live_len,
                    next_len,
                    next_live_len,
                    current_live_len,
                    first,
                    second,
                    basis,
                )?;
                let output_table = current_table.map_or(0, |current| 1 - current);
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 resident direct range fold and partials");
                if let Some(table) = current_table {
                    encoder
                        .set_compute_pipeline_state(&self.fp128_direct_range_field_fold_pipeline);
                    encoder.set_buffer(0, Some(&session.tables[table]), 0);
                    encoder.set_buffer(
                        5,
                        Some(&challenges),
                        (round * size_of::<Fp128Limbs>()) as u64,
                    );
                } else {
                    let prefix_size = 1usize
                        .checked_shl(u32::try_from(rounds_folded + 1).map_err(|_| {
                            MetalCommitError::ShapeOverflow(
                                "resident direct range compact prefix width",
                            )
                        })?)
                        .ok_or(MetalCommitError::ShapeOverflow(
                            "resident direct range compact prefix size",
                        ))?;
                    params.prefix_size = prefix_size as u64;
                    params.materialize_prefix =
                        u64::from(rounds_folded + 1 >= session.compact_prefix_rounds);
                    params.resident_challenges = 1;
                    encoder
                        .set_compute_pipeline_state(&self.fp128_direct_range_compact_fold_pipeline);
                    encoder.set_buffer(0, Some(&session.compact_digits), 0);
                    encoder.set_buffer(5, Some(&challenges), 0);
                }
                encoder.set_buffer(1, Some(&session.tables[output_table]), 0);
                encoder.set_buffer(2, Some(&equality_buffers[round + 1].0), 0);
                encoder.set_buffer(3, Some(&equality_buffers[round + 1].1), 0);
                encoder.set_buffer(4, Some(&session.partials), 0);
                set_inline_bytes(encoder, 6, &params);
                encoder.dispatch_thread_groups(
                    MTLSize::new(params.workgroups, 1, 1),
                    MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                encode_direct_range_reduction_at_offset(
                    command,
                    &self.fp128_direct_range_reduce_pipeline,
                    &session.partials,
                    &round_outputs,
                    ((round + 1) * FP128_DIRECT_RANGE_STORED_COEFFICIENTS * size_of::<Fp128Limbs>())
                        as u64,
                    &params,
                );

                if current_table.is_some() || rounds_folded + 1 >= session.compact_prefix_rounds {
                    current_table = Some(output_table);
                }
                current_len = next_len;
                current_live_len = next_live_len;
                rounds_folded += 1;
            }

            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: the completed command initialized all round and challenge outputs.
            let round_values = unsafe {
                std::slice::from_raw_parts(
                    round_outputs.contents().cast::<Fp128Limbs>(),
                    num_rounds * FP128_DIRECT_RANGE_STORED_COEFFICIENTS,
                )
            };
            let round_coefficients = round_values
                .chunks_exact(FP128_DIRECT_RANGE_STORED_COEFFICIENTS)
                .map(|values| std::array::from_fn(|index| values[index]))
                .collect::<Vec<_>>();
            // SAFETY: the completed command initialized exactly `num_rounds` challenges.
            let challenge_values = unsafe {
                std::slice::from_raw_parts(challenges.contents().cast::<Fp128Limbs>(), num_rounds)
            }
            .to_vec();
            // SAFETY: the completed command initialized the final scalar output.
            let final_evaluation = unsafe { *session.final_output.contents().cast::<Fp128Limbs>() };
            let mut next_chaining_value = [0u8; 64];
            // SAFETY: `state` remains a 64-byte shared buffer after command completion.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    state.contents().cast::<u8>(),
                    next_chaining_value.as_mut_ptr(),
                    next_chaining_value.len(),
                );
            }
            let readback_copy = readback_start.elapsed();
            session.current_len = current_len;
            session.current_live_len = current_live_len;
            session.current_table = current_table;
            session.rounds_folded = rounds_folded;
            let equality_bytes = equality_schedule
                .iter()
                .try_fold(0usize, |bytes, (first, second)| {
                    bytes
                        .checked_add(size_of_val(first.as_slice()))
                        .and_then(|sum| sum.checked_add(size_of_val(second.as_slice())))
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct range equality buffers",
                ))?;
            let allocation_bytes = equality_bytes
                .checked_add(round_output_bytes)
                .and_then(|bytes| bytes.checked_add(challenge_bytes))
                .and_then(|bytes| bytes.checked_add(64 + size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct range dispatch allocation bytes",
                ))?;
            Ok(DirectRangeResidentOutcome {
                round_coefficients,
                challenges: challenge_values,
                final_evaluation,
                chaining_value: next_chaining_value,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes,
            })
        })
    }

    pub(crate) fn begin_fp128_direct_range(
        &self,
        digits: &[i8],
        domain_len: usize,
        compact_prefix_rounds: usize,
    ) -> Result<(DirectRangeSession, Duration), MetalCommitError> {
        if domain_len < 4
            || !domain_len.is_power_of_two()
            || digits.len() > domain_len
            || compact_prefix_rounds == 0
            || compact_prefix_rounds >= domain_len.trailing_zeros() as usize
        {
            return Err(MetalCommitError::UnsupportedShape(
                "direct range proof requires a power-of-two domain of at least four entries".into(),
            ));
        }
        let setup_start = Instant::now();
        let compact_digits = self.shared_buffer_from_slice(digits)?;
        let compact_prefix_size = 1usize
            .checked_shl(u32::try_from(compact_prefix_rounds).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct range compact prefix width")
            })?)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range compact prefix size",
            ))?;
        let first_table_len = digits.len().div_ceil(compact_prefix_size).max(1);
        let second_table_len = first_table_len.div_ceil(2).max(1);
        let first_table_bytes = first_table_len.checked_mul(size_of::<Fp128Limbs>()).ok_or(
            MetalCommitError::ShapeOverflow("direct range first table bytes"),
        )?;
        let second_table_bytes = second_table_len
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range second table bytes",
            ))?;
        let tables = [
            self.private_buffer(first_table_bytes)?,
            self.private_buffer(second_table_bytes)?,
        ];
        let maximum_pairs = digits.len().div_ceil(2);
        let maximum_workgroups = direct_range_workgroups(maximum_pairs);
        let partial_bytes = maximum_workgroups
            .checked_mul(FP128_DIRECT_RANGE_STORED_COEFFICIENTS)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range partial bytes",
            ))?;
        let round_output_bytes = FP128_DIRECT_RANGE_STORED_COEFFICIENTS
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range message output bytes",
            ))?;
        let final_output_bytes = size_of::<Fp128Limbs>();
        let partials = self.private_buffer(partial_bytes)?;
        let round_output = self.shared_buffer(round_output_bytes)?;
        let final_output = self.shared_buffer(final_output_bytes)?;
        let allocation_bytes = digits
            .len()
            .checked_add(first_table_bytes)
            .and_then(|bytes| bytes.checked_add(second_table_bytes))
            .and_then(|bytes| bytes.checked_add(partial_bytes))
            .and_then(|bytes| bytes.checked_add(round_output_bytes))
            .and_then(|bytes| bytes.checked_add(final_output_bytes))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range resident allocation bytes",
            ))?;
        Ok((
            DirectRangeSession {
                compact_digits,
                tables,
                partials,
                round_output,
                final_output,
                live_len: digits.len(),
                current_len: domain_len,
                current_live_len: digits.len(),
                current_table: None,
                compact_prefix_rounds,
                rounds_folded: 0,
                allocation_bytes,
            },
            setup_start.elapsed(),
        ))
    }

    pub(crate) fn dispatch_fp128_direct_range_initial(
        &self,
        session: &DirectRangeSession,
        e_first: &[Fp128Limbs],
        e_second: &[Fp128Limbs],
        basis: usize,
    ) -> Result<DirectRangeRoundOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let params = direct_range_params(
                session.live_len,
                session.current_len,
                session.current_live_len,
                session.current_live_len,
                e_first,
                e_second,
                basis,
            )?;
            let buffer_start = Instant::now();
            let first = self.shared_buffer_from_slice(e_first)?;
            let second = self.shared_buffer_from_slice(e_second)?;
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct range initial round");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 direct range initial partials");
            encoder.set_compute_pipeline_state(&self.fp128_direct_range_initial_pipeline);
            encoder.set_buffer(0, Some(&session.compact_digits), 0);
            encoder.set_buffer(1, Some(&first), 0);
            encoder.set_buffer(2, Some(&second), 0);
            encoder.set_buffer(3, Some(&session.partials), 0);
            set_inline_bytes(encoder, 4, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.workgroups, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            encode_direct_range_reduction(
                command,
                &self.fp128_direct_range_reduce_pipeline,
                &session.partials,
                &session.round_output,
                &params,
            );
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = read_direct_range_coefficients(&session.round_output);
            let readback_copy = readback_start.elapsed();
            Ok(DirectRangeRoundOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: size_of_val(e_first) + size_of_val(e_second),
            })
        })
    }

    pub(crate) fn dispatch_fp128_direct_range_advance(
        &self,
        session: &mut DirectRangeSession,
        challenge: Fp128Limbs,
        next_eq: Option<(&[Fp128Limbs], &[Fp128Limbs])>,
        prefix_weights: &[Fp128Limbs],
        basis: usize,
    ) -> Result<DirectRangeAdvanceOutcome, MetalCommitError> {
        autoreleasepool(|| {
            if session.current_len < 2 || !session.current_len.is_power_of_two() {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct range session has no foldable table".into(),
                ));
            }
            let next_len = session.current_len / 2;
            let next_live_len = session.current_live_len.div_ceil(2);
            if next_len == 1 {
                if next_eq.is_some() {
                    return Err(MetalCommitError::UnsupportedShape(
                        "final direct range fold received a next-round equality table".into(),
                    ));
                }
                let current_table = session.current_table.ok_or_else(|| {
                    MetalCommitError::UnsupportedShape(
                        "direct range compact domain is too small for resident execution".into(),
                    )
                })?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita fp128 direct range final fold");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 direct range final fold");
                encoder.set_compute_pipeline_state(&self.fp128_direct_range_finalize_pipeline);
                encoder.set_buffer(0, Some(&session.tables[current_table]), 0);
                encoder.set_buffer(1, Some(&session.final_output), 0);
                set_inline_bytes(encoder, 2, &challenge);
                set_inline_bytes(encoder, 3, &(session.current_live_len as u64));
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();
                let (command_wall, gpu) = complete_command(command)?;
                let readback_start = Instant::now();
                // SAFETY: `final_output` is a shared buffer containing one fp128 value.
                let final_evaluation =
                    unsafe { *session.final_output.contents().cast::<Fp128Limbs>() };
                let readback_copy = readback_start.elapsed();
                session.current_len = 1;
                session.current_live_len = next_live_len;
                return Ok(DirectRangeAdvanceOutcome {
                    next_coefficients: None,
                    final_evaluation: Some(final_evaluation),
                    timings: DispatchTimings {
                        buffer_setup: Duration::ZERO,
                        command_wall,
                        gpu,
                        readback_copy,
                    },
                    allocation_bytes: 0,
                });
            }

            let (e_first, e_second) = next_eq.ok_or_else(|| {
                MetalCommitError::UnsupportedShape(
                    "non-final direct range fold is missing equality factors".into(),
                )
            })?;
            let mut params = direct_range_params(
                session.live_len,
                next_len,
                next_live_len,
                session.current_live_len,
                e_first,
                e_second,
                basis,
            )?;
            let output_table = session.current_table.map_or(0, |current| 1 - current);
            let buffer_start = Instant::now();
            let first = self.shared_buffer_from_slice(e_first)?;
            let second = self.shared_buffer_from_slice(e_second)?;
            let prefix = if session.current_table.is_none() {
                let expected_prefix_size = 1usize
                    .checked_shl(u32::try_from(session.rounds_folded + 1).map_err(|_| {
                        MetalCommitError::ShapeOverflow("direct range compact prefix width")
                    })?)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "direct range compact prefix size",
                    ))?;
                if prefix_weights.len() != expected_prefix_size {
                    return Err(MetalCommitError::UnsupportedShape(
                        "direct range compact prefix weights have the wrong length".into(),
                    ));
                }
                params.prefix_size = expected_prefix_size as u64;
                params.materialize_prefix =
                    u64::from(session.rounds_folded + 1 >= session.compact_prefix_rounds);
                Some(self.shared_buffer_from_slice(prefix_weights)?)
            } else {
                None
            };
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct range fold and next round");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 direct range fold and next-round partials");
            if let Some(current_table) = session.current_table {
                encoder.set_compute_pipeline_state(&self.fp128_direct_range_field_fold_pipeline);
                encoder.set_buffer(0, Some(&session.tables[current_table]), 0);
            } else {
                encoder.set_compute_pipeline_state(&self.fp128_direct_range_compact_fold_pipeline);
                encoder.set_buffer(0, Some(&session.compact_digits), 0);
            }
            encoder.set_buffer(1, Some(&session.tables[output_table]), 0);
            encoder.set_buffer(2, Some(&first), 0);
            encoder.set_buffer(3, Some(&second), 0);
            encoder.set_buffer(4, Some(&session.partials), 0);
            if let Some(prefix) = &prefix {
                encoder.set_buffer(5, Some(prefix), 0);
            } else {
                set_inline_bytes(encoder, 5, &challenge);
            }
            set_inline_bytes(encoder, 6, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.workgroups, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            encode_direct_range_reduction(
                command,
                &self.fp128_direct_range_reduce_pipeline,
                &session.partials,
                &session.round_output,
                &params,
            );
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = read_direct_range_coefficients(&session.round_output);
            let readback_copy = readback_start.elapsed();
            if session.current_table.is_some()
                || session.rounds_folded + 1 >= session.compact_prefix_rounds
            {
                session.current_table = Some(output_table);
            }
            session.current_len = next_len;
            session.current_live_len = next_live_len;
            session.rounds_folded += 1;
            Ok(DirectRangeAdvanceOutcome {
                next_coefficients: Some(coefficients),
                final_evaluation: None,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: size_of_val(e_first)
                    + size_of_val(e_second)
                    + size_of_val(prefix_weights),
            })
        })
    }

    pub(crate) fn direct_range_session_allocation_bytes(
        &self,
        session: &DirectRangeSession,
    ) -> usize {
        session.allocation_bytes
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the session boundary keeps independent resident relation tables explicit"
    )]
    pub(crate) fn begin_fp128_direct_relation(
        &self,
        digits: &[i8],
        domain_len: usize,
        compact_prefix_rounds: usize,
        coefficient_rounds: usize,
        lane_weights: &[Fp128Limbs],
        linear_segments: &[DirectRelationLinearSegment],
        lane_offsets: &[u32],
        lane_segments: &[u32],
        linear_sources: &[DirectRelationLinearSourceInput],
        linear_dense_values: &[Fp128Limbs],
    ) -> Result<(DirectRelationSession, DispatchTimings), MetalCommitError> {
        let coefficient_count = 1usize
            .checked_shl(u32::try_from(coefficient_rounds).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation coefficient rounds")
            })?)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation coefficient count",
            ))?;
        let live_lane_count = digits.len() / coefficient_count;
        let linear_source_elements = linear_sources.iter().try_fold(0usize, |total, source| {
            total
                .checked_add(source.element_len().ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation linear source length",
                ))?)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation linear source length",
                ))
        })?;
        let linear_source_lane_count = linear_source_elements
            .checked_div(coefficient_count)
            .filter(|_| linear_source_elements.is_multiple_of(coefficient_count))
            .ok_or(MetalCommitError::UnsupportedShape(
                "factored linear source does not match the coefficient geometry".into(),
            ))?;
        let mut linear_source_lane_offsets = Vec::with_capacity(linear_sources.len() + 1);
        linear_source_lane_offsets.push(0u32);
        let mut source_lane_cursor = 0usize;
        for source in linear_sources {
            let source_elements = source.element_len().ok_or(MetalCommitError::ShapeOverflow(
                "direct relation linear source length",
            ))?;
            let source_lanes = source_elements
                .checked_div(coefficient_count)
                .filter(|_| source_elements.is_multiple_of(coefficient_count))
                .ok_or(MetalCommitError::UnsupportedShape(
                    "direct relation source is not coefficient aligned".into(),
                ))?;
            source_lane_cursor = source_lane_cursor.checked_add(source_lanes).ok_or(
                MetalCommitError::ShapeOverflow("direct relation linear source lanes"),
            )?;
            linear_source_lane_offsets.push(u32::try_from(source_lane_cursor).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation linear source lanes")
            })?);
        }
        let linear_mode = if !linear_dense_values.is_empty() {
            2
        } else if !linear_sources.is_empty() {
            1
        } else {
            0
        };
        let linear_shape_is_valid = match linear_mode {
            0 => linear_dense_values.is_empty(),
            1 => linear_source_elements != 0 && linear_dense_values.is_empty(),
            2 => {
                linear_sources.is_empty()
                    && coefficient_count == 1
                    && linear_dense_values.len() == live_lane_count
            }
            _ => false,
        };
        if domain_len < 16
            || !domain_len.is_power_of_two()
            || digits.len() > domain_len
            || compact_prefix_rounds == 0
            || compact_prefix_rounds > coefficient_rounds
            || coefficient_rounds > domain_len.trailing_zeros() as usize
            || !digits.len().is_multiple_of(coefficient_count)
            || lane_weights.is_empty()
            || !lane_weights.len().is_power_of_two()
            || coefficient_count.checked_mul(lane_weights.len()) != Some(domain_len)
            || lane_offsets.is_empty()
            || !linear_shape_is_valid
        {
            return Err(MetalCommitError::UnsupportedShape(
                "direct relation proof has malformed resident geometry".into(),
            ));
        }
        let setup_start = Instant::now();
        let compact_digits = self.shared_buffer_from_slice(digits)?;
        let compact_prefix_size = 1usize
            .checked_shl(u32::try_from(compact_prefix_rounds).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation compact prefix width")
            })?)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation compact prefix size",
            ))?;
        let first_table_len = digits.len().div_ceil(compact_prefix_size).max(1);
        let second_table_len = first_table_len.div_ceil(2).max(1);
        let first_table_bytes = first_table_len.checked_mul(size_of::<Fp128Limbs>()).ok_or(
            MetalCommitError::ShapeOverflow("direct relation first table bytes"),
        )?;
        let second_table_bytes = second_table_len
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation second table bytes",
            ))?;
        let tables = [
            self.private_buffer(first_table_bytes)?,
            self.private_buffer(second_table_bytes)?,
        ];
        let second_lane_count = (lane_weights.len() / 2).max(1);
        let second_lane_bytes = second_lane_count
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation second lane-weight table",
            ))?;
        let lane_weight_tables = [
            self.shared_buffer_from_slice(lane_weights)?,
            self.private_buffer(second_lane_bytes)?,
        ];
        let maximum_workgroups = direct_range_workgroups(digits.len().div_ceil(2));
        let partial_bytes = maximum_workgroups
            .checked_mul(FP128_DIRECT_RELATION_STORED_COEFFICIENTS)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation partial bytes",
            ))?;
        let output_bytes = FP128_DIRECT_RELATION_STORED_COEFFICIENTS
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation output bytes",
            ))?;
        let partials = self.private_buffer(partial_bytes)?;
        let round_output = self.shared_buffer(output_bytes)?;
        let additional_output = self.shared_buffer(output_bytes)?;
        let final_output = self.shared_buffer(size_of::<Fp128Limbs>())?;
        let linear_final_output = self.shared_buffer(size_of::<Fp128Limbs>())?;
        let two_round_prefix_max_workgroups =
            live_lane_count.div_ceil(FP128_DIRECT_RANGE_THREADS).max(1);
        let two_round_prefix_partial_bytes = two_round_prefix_max_workgroups
            .checked_mul(FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation two-round prefix partial bytes",
            ))?;
        let two_round_prefix_output_bytes = FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation two-round prefix output bytes",
            ))?;
        let two_round_prefix_partials = self.private_buffer(two_round_prefix_partial_bytes)?;
        let two_round_prefix_output = self.shared_buffer(two_round_prefix_output_bytes)?;

        let zero_segment = DirectRelationLinearSegment {
            factor: Fp128Limbs::default(),
            source_index: 0,
            target_lane_start: 0,
            target_lane_stride: 1,
            source_lane_start: 0,
            source_lane_stride: 1,
            lane_count: 0,
        };
        let zero_u32 = 0u32;
        let segment_bytes = if linear_segments.is_empty() {
            size_of::<DirectRelationLinearSegment>()
        } else {
            size_of_val(linear_segments)
        };
        let linear_segments = self.shared_buffer_from_slice(if linear_segments.is_empty() {
            std::slice::from_ref(&zero_segment)
        } else {
            linear_segments
        })?;
        let lane_offsets_buffer = self.shared_buffer_from_slice(lane_offsets)?;
        let lane_segments_buffer = self.shared_buffer_from_slice(if lane_segments.is_empty() {
            std::slice::from_ref(&zero_u32)
        } else {
            lane_segments
        })?;
        let linear_source_lane_offsets_buffer =
            self.shared_buffer_from_slice(if linear_source_lane_offsets.is_empty() {
                std::slice::from_ref(&zero_u32)
            } else {
                &linear_source_lane_offsets
            })?;
        let linear_capacity = linear_source_elements
            .max(linear_dense_values.len())
            .max(live_lane_count)
            .max(1);
        let linear_table_bytes = linear_capacity.checked_mul(size_of::<Fp128Limbs>()).ok_or(
            MetalCommitError::ShapeOverflow("direct relation linear table bytes"),
        )?;
        let linear_tables = [
            self.private_buffer(linear_table_bytes)?,
            self.private_buffer(linear_table_bytes)?,
        ];
        let command = self.queue.new_command_buffer();
        command.set_label("Akita resident direct relation linear source construction");
        let mut command_needed = false;
        let mut source_input_bytes = 0usize;
        let mut source_element_offset = 0usize;
        if !linear_dense_values.is_empty() {
            let staging = self.shared_buffer_from_slice(linear_dense_values)?;
            let encoder = command.new_blit_command_encoder();
            encoder.copy_from_buffer(
                &staging,
                0,
                &linear_tables[0],
                0,
                size_of_val(linear_dense_values) as u64,
            );
            encoder.end_encoding();
            command_needed = true;
            source_input_bytes = size_of_val(linear_dense_values);
        }
        for source in linear_sources {
            let destination_offset = source_element_offset
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation linear source offset",
                ))?;
            match source {
                DirectRelationLinearSourceInput::Values(values) => {
                    let staging = self.shared_buffer_from_slice(values)?;
                    let encoder = command.new_blit_command_encoder();
                    encoder.copy_from_buffer(
                        &staging,
                        0,
                        &linear_tables[0],
                        destination_offset as u64,
                        size_of_val(values.as_slice()) as u64,
                    );
                    encoder.end_encoding();
                    source_input_bytes = source_input_bytes
                        .checked_add(size_of_val(values.as_slice()))
                        .ok_or(MetalCommitError::ShapeOverflow(
                            "direct relation source input bytes",
                        ))?;
                }
                DirectRelationLinearSourceInput::ReducedSetup {
                    matrix,
                    ring_dimension,
                    row_count,
                    column_count,
                    row_weights,
                    alpha_powers,
                    alpha,
                    wrap_correction,
                } => {
                    if *ring_dimension < 32
                        || *ring_dimension > 512
                        || !ring_dimension.is_multiple_of(32)
                        || row_weights.len() != *row_count
                        || alpha_powers.len() != *ring_dimension
                    {
                        return Err(MetalCommitError::UnsupportedShape(
                            "reduced setup source geometry is unsupported".into(),
                        ));
                    }
                    let weights = self.shared_buffer_from_slice(row_weights)?;
                    let powers = self.shared_buffer_from_slice(alpha_powers)?;
                    let params = DirectRelationReducedSourceParams {
                        ring_dimension: *ring_dimension as u64,
                        row_count: *row_count as u64,
                        item_count: *column_count as u64,
                        reserved: 0,
                        alpha: *alpha,
                        wrap_correction: *wrap_correction,
                    };
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 reduced setup source");
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_setup_source_pipeline,
                    );
                    encoder.set_buffer(0, Some(matrix), 0);
                    encoder.set_buffer(1, Some(&weights), 0);
                    encoder.set_buffer(2, Some(&powers), 0);
                    encoder.set_buffer(3, Some(&linear_tables[0]), destination_offset as u64);
                    set_inline_bytes(encoder, 4, &params);
                    encoder.dispatch_thread_groups(
                        MTLSize::new(*column_count as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    encoder.end_encoding();
                    source_input_bytes = source_input_bytes
                        .checked_add(size_of_val(row_weights.as_slice()))
                        .and_then(|bytes| bytes.checked_add(size_of_val(alpha_powers.as_slice())))
                        .ok_or(MetalCommitError::ShapeOverflow(
                            "direct relation source input bytes",
                        ))?;
                }
                DirectRelationLinearSourceInput::ReducedSparse {
                    ring_dimension,
                    challenge_count,
                    term_offsets,
                    positions,
                    coefficients,
                    alpha_powers,
                    alpha,
                    wrap_correction,
                } => {
                    if *ring_dimension < 32
                        || *ring_dimension > 512
                        || !ring_dimension.is_multiple_of(32)
                        || term_offsets.len() != challenge_count + 1
                        || positions.len() != coefficients.len()
                        || term_offsets.last().copied().map(|offset| offset as usize)
                            != Some(positions.len())
                        || positions
                            .iter()
                            .any(|position| *position as usize >= *ring_dimension)
                        || alpha_powers.len() != *ring_dimension
                    {
                        return Err(MetalCommitError::UnsupportedShape(
                            "reduced sparse source geometry is unsupported".into(),
                        ));
                    }
                    let offsets = self.shared_buffer_from_slice(term_offsets)?;
                    let positions_buffer = self.shared_buffer_from_slice(positions)?;
                    let coefficients_buffer = self.shared_buffer_from_slice(coefficients)?;
                    let powers = self.shared_buffer_from_slice(alpha_powers)?;
                    let params = DirectRelationReducedSourceParams {
                        ring_dimension: *ring_dimension as u64,
                        row_count: 0,
                        item_count: *challenge_count as u64,
                        reserved: 0,
                        alpha: *alpha,
                        wrap_correction: *wrap_correction,
                    };
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 reduced sparse source");
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_sparse_source_pipeline,
                    );
                    encoder.set_buffer(0, Some(&offsets), 0);
                    encoder.set_buffer(1, Some(&positions_buffer), 0);
                    encoder.set_buffer(2, Some(&coefficients_buffer), 0);
                    encoder.set_buffer(3, Some(&powers), 0);
                    encoder.set_buffer(4, Some(&linear_tables[0]), destination_offset as u64);
                    set_inline_bytes(encoder, 5, &params);
                    encoder.dispatch_thread_groups(
                        MTLSize::new(*challenge_count as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    encoder.end_encoding();
                    source_input_bytes = source_input_bytes
                        .checked_add(size_of_val(term_offsets.as_slice()))
                        .and_then(|bytes| bytes.checked_add(size_of_val(positions.as_slice())))
                        .and_then(|bytes| bytes.checked_add(size_of_val(coefficients.as_slice())))
                        .and_then(|bytes| bytes.checked_add(size_of_val(alpha_powers.as_slice())))
                        .ok_or(MetalCommitError::ShapeOverflow(
                            "direct relation source input bytes",
                        ))?;
                }
            }
            source_element_offset = source_element_offset
                .checked_add(source.element_len().ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation linear source length",
                ))?)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation linear source length",
                ))?;
            command_needed = true;
        }
        let buffer_setup = setup_start.elapsed();
        let (command_wall, gpu) = if command_needed {
            complete_command(command)?
        } else {
            (Duration::ZERO, None)
        };
        let allocation_bytes = digits
            .len()
            .checked_add(first_table_bytes)
            .and_then(|bytes| bytes.checked_add(second_table_bytes))
            .and_then(|bytes| bytes.checked_add(partial_bytes))
            .and_then(|bytes| bytes.checked_add(2 * output_bytes))
            .and_then(|bytes| bytes.checked_add(size_of::<Fp128Limbs>()))
            .and_then(|bytes| bytes.checked_add(size_of::<Fp128Limbs>()))
            .and_then(|bytes| bytes.checked_add(segment_bytes))
            .and_then(|bytes| bytes.checked_add(size_of_val(lane_offsets)))
            .and_then(|bytes| bytes.checked_add(size_of_val(lane_segments)))
            .and_then(|bytes| bytes.checked_add(size_of_val(linear_source_lane_offsets.as_slice())))
            .and_then(|bytes| bytes.checked_add(2 * linear_table_bytes))
            .and_then(|bytes| bytes.checked_add(size_of_val(lane_weights)))
            .and_then(|bytes| bytes.checked_add(second_lane_bytes))
            .and_then(|bytes| bytes.checked_add(two_round_prefix_partial_bytes))
            .and_then(|bytes| bytes.checked_add(two_round_prefix_output_bytes))
            .and_then(|bytes| bytes.checked_add(source_input_bytes))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation resident allocation bytes",
            ))?;
        Ok((
            DirectRelationSession {
                compact_digits,
                tables,
                partials,
                round_output,
                additional_output,
                final_output,
                linear_final_output,
                linear_segments,
                lane_offsets: lane_offsets_buffer,
                lane_segments: lane_segments_buffer,
                linear_source_lane_offsets: linear_source_lane_offsets_buffer,
                linear_tables,
                current_linear_table: 0,
                linear_mode,
                linear_source_lane_count,
                linear_current_coeff_count: coefficient_count,
                linear_current_live_lane_count: live_lane_count,
                lane_weight_tables,
                two_round_prefix_partials,
                two_round_prefix_output,
                two_round_prefix_max_workgroups,
                live_len: digits.len(),
                current_len: domain_len,
                current_live_len: digits.len(),
                current_table: None,
                current_lane_weight_table: 0,
                current_lane_count: lane_weights.len(),
                coefficient_rounds,
                compact_prefix_rounds,
                rounds_folded: 0,
                allocation_bytes,
            },
            DispatchTimings {
                buffer_setup,
                command_wall,
                gpu,
                readback_copy: Duration::ZERO,
            },
        ))
    }

    pub(crate) fn dispatch_fp128_direct_relation_two_round_prefix(
        &self,
        session: &DirectRelationSession,
        equality_first: &[Fp128Limbs],
        equality_second: &[Fp128Limbs],
        alpha_points: &[Fp128Limbs],
        norm_omitted_corner: usize,
        round: DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationTwoRoundPrefixOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let coefficient_count = round.alpha.len();
            if session.rounds_folded != 0
                || session.current_table.is_some()
                || coefficient_count < 4
                || !coefficient_count.is_power_of_two()
                || norm_omitted_corner >= 4
                || !matches!(session.linear_mode, 0 | 1)
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation two-round prefix state is malformed".into(),
                ));
            }
            let y_quads = coefficient_count / 4;
            let equality_entries = equality_first
                .len()
                .checked_mul(equality_second.len())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation prefix equality entries",
                ))?;
            let expected_equality_entries = session.current_lane_count.checked_mul(y_quads).ok_or(
                MetalCommitError::ShapeOverflow("direct relation prefix domain"),
            )?;
            if equality_first.is_empty()
                || equality_second.is_empty()
                || !equality_first.len().is_power_of_two()
                || !equality_second.len().is_power_of_two()
                || equality_entries != expected_equality_entries
                || alpha_points.len()
                    != 8usize
                        .checked_mul(y_quads)
                        .ok_or(MetalCommitError::ShapeOverflow(
                            "direct relation prefix alpha points",
                        ))?
                || round.live_lane_count
                    != session.live_len.checked_div(coefficient_count).ok_or(
                        MetalCommitError::ShapeOverflow("direct relation prefix live lane count"),
                    )?
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation two-round prefix geometry does not match the session".into(),
                ));
            }
            let equality_block_lanes = equality_first.len() / y_quads;
            let lanes_per_thread = if equality_block_lanes.is_multiple_of(
                FP128_DIRECT_RANGE_THREADS * FP128_DIRECT_RELATION_PREFIX_LANES_PER_THREAD,
            ) {
                FP128_DIRECT_RELATION_PREFIX_LANES_PER_THREAD
            } else {
                1
            };
            let workgroups = round
                .live_lane_count
                .div_ceil(FP128_DIRECT_RANGE_THREADS * lanes_per_thread)
                .max(1);
            if workgroups > session.two_round_prefix_max_workgroups {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation two-round prefix exceeds its workspace".into(),
                ));
            }
            let params = DirectRelationTwoRoundPrefixParams {
                live_lane_count: u64::try_from(round.live_lane_count).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation prefix live lanes")
                })?,
                coefficient_count: u64::try_from(coefficient_count).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation prefix coefficients")
                })?,
                y_quads: u64::try_from(y_quads)
                    .map_err(|_| MetalCommitError::ShapeOverflow("direct relation prefix quads"))?,
                equality_first_len: u64::try_from(equality_first.len()).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation prefix equality split")
                })?,
                workgroups: u64::try_from(workgroups).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation prefix workgroups")
                })?,
                lanes_per_thread: lanes_per_thread as u64,
                norm_omitted_corner: norm_omitted_corner as u64,
                linear_mode: session.linear_mode as u64,
            };
            let mut additional_params = direct_relation_params(
                session,
                session.current_len,
                session.current_lane_count,
                false,
                &round,
            )?;
            additional_params.prefix_size = 1;

            let buffer_start = Instant::now();
            let buffers = self.direct_relation_round_buffers(&round)?;
            let equality_first_buffer = self.shared_buffer_from_slice(equality_first)?;
            let equality_second_buffer = self.shared_buffer_from_slice(equality_second)?;
            let alpha_points_buffer = self.shared_buffer_from_slice(alpha_points)?;
            let prefix_weight = [Fp128Limbs::from_u128(1)];
            let prefix = self.shared_buffer_from_slice(&prefix_weight)?;
            let additional_pairs = if round.additional_pairs.is_empty() {
                None
            } else {
                Some(self.shared_buffer_from_slice(round.additional_pairs)?)
            };
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct relation two-round prefix");
            let encoder = command.new_compute_command_encoder();
            encoder
                .set_compute_pipeline_state(&self.fp128_direct_relation_two_round_prefix_pipeline);
            encoder.set_buffer(0, Some(&session.compact_digits), 0);
            encoder.set_buffer(1, Some(&equality_first_buffer), 0);
            encoder.set_buffer(2, Some(&equality_second_buffer), 0);
            encoder.set_buffer(3, Some(&alpha_points_buffer), 0);
            encoder.set_buffer(
                4,
                Some(&session.lane_weight_tables[session.current_lane_weight_table]),
                0,
            );
            encoder.set_buffer(
                5,
                Some(&session.linear_tables[session.current_linear_table]),
                0,
            );
            encoder.set_buffer(6, Some(&session.linear_source_lane_offsets), 0);
            encoder.set_buffer(7, Some(&session.lane_offsets), 0);
            encoder.set_buffer(8, Some(&session.lane_segments), 0);
            encoder.set_buffer(9, Some(&session.linear_segments), 0);
            encoder.set_buffer(10, Some(&session.two_round_prefix_partials), 0);
            set_inline_bytes(encoder, 11, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    params.workgroups,
                    FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS as u64,
                    1,
                ),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.fp128_direct_relation_two_round_prefix_reduce_pipeline,
            );
            encoder.set_buffer(0, Some(&session.two_round_prefix_partials), 0);
            encoder.set_buffer(1, Some(&session.two_round_prefix_output), 0);
            set_inline_bytes(encoder, 2, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS as u64, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            if let Some(additional_pairs) = &additional_pairs {
                self.encode_direct_relation_additional_compact(
                    command,
                    session,
                    &prefix,
                    additional_pairs,
                    &round.scalars,
                    &additional_params,
                );
            }
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let (norm_evals_except_corner, relation_evals_except_corner) =
                read_direct_relation_prefix_evals(&session.two_round_prefix_output);
            let additional_coefficients = if additional_pairs.is_some() {
                read_direct_relation_coefficients(&session.additional_output)
            } else {
                [Fp128Limbs::default(); FP128_DIRECT_RELATION_STORED_COEFFICIENTS]
            };
            let readback_copy = readback_start.elapsed();
            let allocation_bytes = buffers
                .allocation_bytes
                .checked_add(size_of_val(equality_first))
                .and_then(|bytes| bytes.checked_add(size_of_val(equality_second)))
                .and_then(|bytes| bytes.checked_add(size_of_val(alpha_points)))
                .and_then(|bytes| bytes.checked_add(size_of_val(&prefix_weight)))
                .and_then(|bytes| bytes.checked_add(size_of_val(round.additional_pairs)))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation prefix dispatch allocation bytes",
                ))?;
            Ok(DirectRelationTwoRoundPrefixOutcome {
                norm_evals_except_corner,
                relation_evals_except_corner,
                additional_coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_direct_relation_additional_compact_only(
        &self,
        session: &mut DirectRelationSession,
        challenge: Fp128Limbs,
        current_len: usize,
        prefix_weights: &[Fp128Limbs],
        round: DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationAdditionalOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let buffer_start = Instant::now();
            let prefix = self.shared_buffer_from_slice(prefix_weights)?;
            let pairs = (!round.additional_pairs.is_empty())
                .then(|| self.shared_buffer_from_slice(round.additional_pairs))
                .transpose()?;
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            self.encode_direct_relation_linear_fold(command, session, challenge)?;
            if let Some(pairs) = &pairs {
                let mut params = direct_relation_params(
                    session,
                    current_len,
                    session.current_lane_count,
                    false,
                    &round,
                )?;
                params.prefix_size = u64::try_from(prefix_weights.len()).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation additional prefix")
                })?;
                self.encode_direct_relation_additional_compact(
                    command,
                    session,
                    &prefix,
                    pairs,
                    &round.scalars,
                    &params,
                );
            }
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = if pairs.is_some() {
                read_direct_relation_coefficients(&session.additional_output)
            } else {
                [Fp128Limbs::default(); FP128_DIRECT_RELATION_STORED_COEFFICIENTS]
            };
            let readback_copy = readback_start.elapsed();
            Ok(DirectRelationAdditionalOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: size_of_val(prefix_weights)
                    .checked_add(size_of_val(round.additional_pairs))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "direct relation additional dispatch allocation bytes",
                    ))?,
            })
        })
    }

    pub(crate) fn dispatch_fp128_direct_relation_resume_after_two_round_prefix(
        &self,
        session: &mut DirectRelationSession,
        challenge: Fp128Limbs,
        prefix_weights: &[Fp128Limbs],
        round: DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationRoundOutcome, MetalCommitError> {
        autoreleasepool(|| {
            if session.rounds_folded != 0
                || session.current_table.is_some()
                || prefix_weights.len() != 4
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation two-round resume state is malformed".into(),
                ));
            }
            let current_len = session.current_len / 4;
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct relation resume after two-round prefix");
            self.encode_direct_relation_linear_fold(command, session, challenge)?;
            let mut params = direct_relation_params(
                session,
                current_len,
                session.current_lane_count,
                false,
                &round,
            )?;
            params.prefix_size = 4;
            let buffer_start = Instant::now();
            let buffers = self.direct_relation_round_buffers(&round)?;
            let prefix = self.shared_buffer_from_slice(prefix_weights)?;
            let additional_pairs = if round.additional_pairs.is_empty() {
                None
            } else {
                Some(self.shared_buffer_from_slice(round.additional_pairs)?)
            };
            let buffer_setup = buffer_start.elapsed();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_direct_relation_initial_pipeline);
            encoder.set_buffer(0, Some(&session.compact_digits), 0);
            self.encode_direct_relation_round_buffers(
                encoder,
                1,
                session,
                &buffers,
                &session.lane_weight_tables[session.current_lane_weight_table],
            );
            encoder.set_buffer(10, Some(&session.partials), 0);
            encoder.set_buffer(11, Some(&prefix), 0);
            set_inline_bytes(encoder, 12, &round.scalars);
            set_inline_bytes(encoder, 13, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.workgroups, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            encode_direct_relation_reduction(
                command,
                &self.fp128_direct_range_reduce_pipeline,
                &session.partials,
                &session.round_output,
                &params,
            );
            if let Some(additional_pairs) = &additional_pairs {
                self.encode_direct_relation_additional_compact(
                    command,
                    session,
                    &prefix,
                    additional_pairs,
                    &round.scalars,
                    &params,
                );
            }
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = read_direct_relation_coefficients(&session.round_output);
            let additional_coefficients = if additional_pairs.is_some() {
                read_direct_relation_coefficients(&session.additional_output)
            } else {
                [Fp128Limbs::default(); FP128_DIRECT_RELATION_STORED_COEFFICIENTS]
            };
            let readback_copy = readback_start.elapsed();
            session.current_len = current_len;
            session.current_live_len = params.current_live_len as usize;
            session.rounds_folded = 2;
            Ok(DirectRelationRoundOutcome {
                coefficients,
                additional_coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: buffers
                    .allocation_bytes
                    .checked_add(size_of_val(prefix_weights))
                    .and_then(|bytes| bytes.checked_add(size_of_val(round.additional_pairs)))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "direct relation resume allocation bytes",
                    ))?,
            })
        })
    }

    pub(crate) fn dispatch_fp128_direct_relation_resident_suffix(
        &self,
        session: &mut DirectRelationSession,
        prefix_challenges: &[Fp128Limbs],
        equality_schedule: &[DirectRelationResidentEqRound],
        initial_round: DirectRelationRoundData<'_>,
        chaining_value: &[u8; 64],
        prior_squeezed_bytes: usize,
    ) -> Result<DirectRelationResidentOutcome, MetalCommitError> {
        let suffix_rounds = equality_schedule.len();
        if suffix_rounds == 0
            || prefix_challenges.len() != session.rounds_folded
            || session.rounds_folded != 2
            || suffix_rounds != session.current_len.trailing_zeros() as usize
            || session.current_table.is_some()
            || session.compact_prefix_rounds != 3
            || initial_round.alpha.is_empty()
            || initial_round.e_first != equality_schedule[0].e_first
            || initial_round.e_second != equality_schedule[0].e_second
        {
            return Err(MetalCommitError::UnsupportedShape(
                "resident direct relation suffix has malformed initial state".into(),
            ));
        }
        let _ = direct_relation_params(
            session,
            session.current_len,
            session.current_lane_count,
            false,
            &initial_round,
        )?;
        let additional_schedule = direct_relation_additional_fold_schedule(
            initial_round.additional_pairs,
            suffix_rounds.saturating_sub(1),
        )?;

        autoreleasepool(|| {
            let setup_start = Instant::now();
            let equality_buffers = equality_schedule
                .iter()
                .map(|round| {
                    Ok((
                        self.shared_buffer_from_slice(&round.e_first)?,
                        self.shared_buffer_from_slice(&round.e_second)?,
                    ))
                })
                .collect::<Result<Vec<_>, MetalCommitError>>()?;
            let taus = equality_schedule
                .iter()
                .map(|round| round.tau)
                .collect::<Vec<_>>();
            let tau_buffer = self.shared_buffer_from_slice(&taus)?;

            let alpha_bytes = size_of_val(initial_round.alpha);
            let alpha_tables = [
                self.shared_buffer_from_slice(initial_round.alpha)?,
                self.private_buffer(alpha_bytes.max(size_of::<Fp128Limbs>()))?,
            ];
            let scalar_values = vec![initial_round.scalars; suffix_rounds];
            let scalars = self.shared_buffer_from_slice(&scalar_values)?;

            let zero_pair = DirectRelationAdditionalPair {
                parent: 0,
                reserved: 0,
                linear: [Fp128Limbs::default(); 2],
                binary: [Fp128Limbs::default(); 2],
            };
            let initial_pair_storage = if initial_round.additional_pairs.is_empty() {
                std::slice::from_ref(&zero_pair)
            } else {
                initial_round.additional_pairs
            };
            let pair_capacity = initial_pair_storage.len();
            let pair_bytes = pair_capacity
                .checked_mul(size_of::<DirectRelationAdditionalPair>())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct relation additional pairs",
                ))?;
            let additional_pairs = [
                self.shared_buffer_from_slice(initial_pair_storage)?,
                self.private_buffer(pair_bytes)?,
            ];
            let additional_mappings = additional_schedule
                .iter()
                .map(|mappings| {
                    if mappings.is_empty() {
                        Ok(None)
                    } else {
                        self.shared_buffer_from_slice(mappings).map(Some)
                    }
                })
                .collect::<Result<Vec<_>, MetalCommitError>>()?;

            let total_rounds = prefix_challenges.len().checked_add(suffix_rounds).ok_or(
                MetalCommitError::ShapeOverflow("resident direct relation challenge count"),
            )?;
            let mut challenge_values = vec![Fp128Limbs::default(); total_rounds];
            challenge_values[..prefix_challenges.len()].copy_from_slice(prefix_challenges);
            let challenges = self.shared_buffer_from_slice(&challenge_values)?;
            let proof_coefficient_count =
                suffix_rounds
                    .checked_mul(3)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "resident direct relation proof coefficients",
                    ))?;
            let proof_bytes = proof_coefficient_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct relation proof bytes",
                ))?;
            let proof_coefficients = self.shared_buffer(proof_bytes)?;
            let count_bytes = suffix_rounds.checked_mul(size_of::<u32>()).ok_or(
                MetalCommitError::ShapeOverflow("resident direct relation coefficient counts"),
            )?;
            let coefficient_counts = self.shared_buffer(count_bytes)?;
            let state = self.shared_buffer_from_slice(chaining_value)?;
            let buffer_setup = setup_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 resident direct relation suffix");
            let mut current_len = session.current_len;
            let mut current_live_len = session.current_live_len;
            let mut current_live_lane_count = initial_round.live_lane_count;
            let mut current_alpha_len = initial_round.alpha.len();
            let mut current_alpha_table = 0usize;
            let mut current_additional_table = 0usize;
            let mut current_additional_count = initial_round.additional_pairs.len();
            let mut current_table = session.current_table;
            let mut current_lane_weight_table = session.current_lane_weight_table;
            let mut current_lane_count = session.current_lane_count;
            let mut rounds_folded = session.rounds_folded;
            let had_linear_terms = session.linear_mode != 0;

            for suffix_round in 0..suffix_rounds {
                session.rounds_folded = rounds_folded;
                session.current_live_len = current_live_len;
                let round_index = prefix_challenges.len() + suffix_round;
                let challenge_offset = round_index.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                    MetalCommitError::ShapeOverflow("resident direct relation challenge offset"),
                )? as u64;
                let proof_offset = suffix_round
                    .checked_mul(3 * size_of::<Fp128Limbs>())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "resident direct relation proof offset",
                    ))? as u64;
                let count_offset = suffix_round.checked_mul(size_of::<u32>()).ok_or(
                    MetalCommitError::ShapeOverflow("resident direct relation count offset"),
                )? as u64;
                let transcript_params = DirectRelationTranscriptParams {
                    prior_squeezed_bytes: if suffix_round == 0 {
                        u64::try_from(prior_squeezed_bytes).map_err(|_| {
                            MetalCommitError::ShapeOverflow(
                                "resident direct relation prior squeeze length",
                            )
                        })?
                    } else {
                        32
                    },
                    has_additional: u64::from(current_additional_count != 0),
                };
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 resident direct relation challenge");
                encoder.set_compute_pipeline_state(
                    &self.fp128_blake2b_relation_sumcheck_round_pipeline,
                );
                encoder.set_buffer(0, Some(&state), 0);
                encoder.set_buffer(1, Some(&session.round_output), 0);
                encoder.set_buffer(2, Some(&session.additional_output), 0);
                encoder.set_buffer(3, Some(&proof_coefficients), proof_offset);
                encoder.set_buffer(4, Some(&coefficient_counts), count_offset);
                encoder.set_buffer(5, Some(&challenges), challenge_offset);
                set_inline_bytes(encoder, 6, &transcript_params);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();

                let next_len = current_len / 2;
                let next_live_len = current_live_len.div_ceil(2);
                self.encode_direct_relation_linear_fold_from_buffer(
                    command,
                    session,
                    &challenges,
                    challenge_offset,
                )?;
                if next_len == 1 {
                    let table = current_table.ok_or_else(|| {
                        MetalCommitError::UnsupportedShape(
                            "resident direct relation compact suffix reaches final fold".into(),
                        )
                    })?;
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 resident direct relation final fold");
                    encoder.set_compute_pipeline_state(&self.fp128_direct_range_finalize_pipeline);
                    encoder.set_buffer(0, Some(&session.tables[table]), 0);
                    encoder.set_buffer(1, Some(&session.final_output), 0);
                    encoder.set_buffer(2, Some(&challenges), challenge_offset);
                    set_inline_bytes(encoder, 3, &(current_live_len as u64));
                    encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                    encoder.end_encoding();
                    if had_linear_terms {
                        let encoder = command.new_blit_command_encoder();
                        encoder.copy_from_buffer(
                            &session.linear_tables[session.current_linear_table],
                            0,
                            &session.linear_final_output,
                            0,
                            size_of::<Fp128Limbs>() as u64,
                        );
                        encoder.end_encoding();
                    }
                    current_len = 1;
                    current_live_len = next_live_len;
                    rounds_folded += 1;
                    break;
                }

                let fold_lane_weights = rounds_folded >= session.coefficient_rounds;
                let next_alpha_len = if fold_lane_weights {
                    current_alpha_len
                } else {
                    current_alpha_len / 2
                };
                let next_live_lane_count = if fold_lane_weights {
                    current_live_lane_count.div_ceil(2)
                } else {
                    current_live_lane_count
                };
                let next_lane_count = if fold_lane_weights {
                    current_lane_count / 2
                } else {
                    current_lane_count
                };
                let next_lane_weight_table = if fold_lane_weights {
                    1 - current_lane_weight_table
                } else {
                    current_lane_weight_table
                };

                if !fold_lane_weights {
                    let output_alpha_table = 1 - current_alpha_table;
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 resident direct relation alpha fold");
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_alpha_fold_pipeline,
                    );
                    encoder.set_buffer(0, Some(&alpha_tables[current_alpha_table]), 0);
                    encoder.set_buffer(1, Some(&alpha_tables[output_alpha_table]), 0);
                    encoder.set_buffer(2, Some(&challenges), challenge_offset);
                    set_inline_bytes(encoder, 3, &(next_alpha_len as u64));
                    encoder.dispatch_threads(
                        MTLSize::new(next_alpha_len as u64, 1, 1),
                        MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
                    );
                    encoder.end_encoding();
                    current_alpha_table = output_alpha_table;
                }

                let scalar_offset = suffix_round
                    .checked_mul(size_of::<DirectRelationScalars>())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "resident direct relation scalar offset",
                    ))? as u64;
                let next_scalar_offset = (suffix_round + 1)
                    .checked_mul(size_of::<DirectRelationScalars>())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "resident direct relation next scalar offset",
                    ))? as u64;
                let tau_offset = suffix_round.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                    MetalCommitError::ShapeOverflow("resident direct relation tau offset"),
                )? as u64;
                let next_tau_offset = (suffix_round + 1)
                    .checked_mul(size_of::<Fp128Limbs>())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "resident direct relation next tau offset",
                    ))? as u64;
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 resident direct relation scalar advance");
                encoder.set_compute_pipeline_state(
                    &self.fp128_direct_relation_scalar_advance_pipeline,
                );
                encoder.set_buffer(0, Some(&scalars), scalar_offset);
                encoder.set_buffer(1, Some(&challenges), challenge_offset);
                encoder.set_buffer(2, Some(&tau_buffer), tau_offset);
                encoder.set_buffer(3, Some(&tau_buffer), next_tau_offset);
                encoder.set_buffer(4, Some(&scalars), next_scalar_offset);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();

                let next_mappings = &additional_schedule[suffix_round];
                let next_additional_count = next_mappings.len();
                if current_additional_count != 0 {
                    let mapping_buffer =
                        additional_mappings[suffix_round].as_ref().ok_or_else(|| {
                            MetalCommitError::UnsupportedShape(
                                "resident direct relation additional topology vanished".into(),
                            )
                        })?;
                    let output_additional_table = 1 - current_additional_table;
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 resident direct relation additional fold");
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_additional_fold_pipeline,
                    );
                    encoder.set_buffer(0, Some(&additional_pairs[current_additional_table]), 0);
                    encoder.set_buffer(1, Some(&additional_pairs[output_additional_table]), 0);
                    encoder.set_buffer(2, Some(mapping_buffer), 0);
                    encoder.set_buffer(3, Some(&challenges), challenge_offset);
                    set_inline_bytes(encoder, 4, &(next_additional_count as u64));
                    encoder.dispatch_threads(
                        MTLSize::new(next_additional_count as u64, 1, 1),
                        MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
                    );
                    encoder.end_encoding();
                    current_additional_table = output_additional_table;
                }

                let next_eq = &equality_schedule[suffix_round + 1];
                let additional_parents_in_range = next_mappings
                    .last()
                    .is_none_or(|mapping| mapping.parent < (next_len / 2) as u64);
                let mut params = direct_relation_params_shape(
                    session,
                    next_len,
                    next_lane_count,
                    fold_lane_weights,
                    next_eq.e_first.len(),
                    next_eq.e_second.len(),
                    next_alpha_len,
                    next_live_lane_count,
                    next_additional_count,
                    additional_parents_in_range,
                )?;
                let output_table = current_table.map_or(0, |current| 1 - current);
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 resident direct relation fold and partials");
                if let Some(table) = current_table {
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_field_fold_pipeline,
                    );
                    encoder.set_buffer(0, Some(&session.tables[table]), 0);
                    encoder.set_buffer(12, Some(&challenges), challenge_offset);
                    encoder.set_buffer(
                        15,
                        Some(&session.lane_weight_tables[current_lane_weight_table]),
                        0,
                    );
                } else {
                    params.prefix_size = 1u64 << (rounds_folded + 1);
                    params.materialize_prefix =
                        u64::from(rounds_folded + 1 >= session.compact_prefix_rounds);
                    params.resident_challenges = 1;
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_compact_fold_pipeline,
                    );
                    encoder.set_buffer(0, Some(&session.compact_digits), 0);
                    encoder.set_buffer(12, Some(&challenges), 0);
                }
                encoder.set_buffer(1, Some(&session.tables[output_table]), 0);
                encoder.set_buffer(2, Some(&equality_buffers[suffix_round + 1].0), 0);
                encoder.set_buffer(3, Some(&equality_buffers[suffix_round + 1].1), 0);
                encoder.set_buffer(4, Some(&alpha_tables[current_alpha_table]), 0);
                encoder.set_buffer(
                    5,
                    Some(&session.lane_weight_tables[next_lane_weight_table]),
                    0,
                );
                encoder.set_buffer(
                    6,
                    Some(&session.linear_tables[session.current_linear_table]),
                    0,
                );
                encoder.set_buffer(7, Some(&session.linear_source_lane_offsets), 0);
                encoder.set_buffer(8, Some(&session.lane_offsets), 0);
                encoder.set_buffer(9, Some(&session.lane_segments), 0);
                encoder.set_buffer(10, Some(&session.linear_segments), 0);
                encoder.set_buffer(11, Some(&session.partials), 0);
                encoder.set_buffer(13, Some(&scalars), next_scalar_offset);
                set_inline_bytes(encoder, 14, &params);
                encoder.dispatch_thread_groups(
                    MTLSize::new(params.workgroups, 1, 1),
                    MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                encode_direct_relation_reduction(
                    command,
                    &self.fp128_direct_range_reduce_pipeline,
                    &session.partials,
                    &session.round_output,
                    &params,
                );

                if next_additional_count != 0 {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 resident direct relation additional round");
                    if current_table.is_none() {
                        encoder.set_compute_pipeline_state(
                            &self.fp128_direct_relation_additional_compact_pipeline,
                        );
                        encoder.set_buffer(0, Some(&session.compact_digits), 0);
                        encoder.set_buffer(1, Some(&challenges), 0);
                        encoder.set_buffer(2, Some(&additional_pairs[current_additional_table]), 0);
                        encoder.set_buffer(3, Some(&session.partials), 0);
                        encoder.set_buffer(4, Some(&scalars), next_scalar_offset);
                        set_inline_bytes(encoder, 5, &params);
                    } else {
                        encoder.set_compute_pipeline_state(
                            &self.fp128_direct_relation_additional_field_pipeline,
                        );
                        encoder.set_buffer(0, Some(&session.tables[output_table]), 0);
                        encoder.set_buffer(1, Some(&additional_pairs[current_additional_table]), 0);
                        encoder.set_buffer(2, Some(&session.partials), 0);
                        encoder.set_buffer(3, Some(&scalars), next_scalar_offset);
                        set_inline_bytes(encoder, 4, &params);
                    }
                    encoder.dispatch_thread_groups(
                        MTLSize::new(params.additional_workgroups, 1, 1),
                        MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
                    );
                    encoder.end_encoding();
                    let mut reduction_params = params;
                    reduction_params.workgroups = params.additional_workgroups;
                    encode_direct_relation_reduction(
                        command,
                        &self.fp128_direct_range_reduce_pipeline,
                        &session.partials,
                        &session.additional_output,
                        &reduction_params,
                    );
                }

                if current_table.is_some() || rounds_folded + 1 >= session.compact_prefix_rounds {
                    current_table = Some(output_table);
                }
                current_len = next_len;
                current_live_len = next_live_len;
                current_live_lane_count = next_live_lane_count;
                current_alpha_len = next_alpha_len;
                current_additional_count = next_additional_count;
                current_lane_weight_table = next_lane_weight_table;
                current_lane_count = next_lane_count;
                rounds_folded += 1;
            }

            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: the completed command initialized all suffix proof slots.
            let proof_values = unsafe {
                std::slice::from_raw_parts(
                    proof_coefficients.contents().cast::<Fp128Limbs>(),
                    proof_coefficient_count,
                )
            };
            let round_coefficients = proof_values
                .chunks_exact(3)
                .map(|values| std::array::from_fn(|index| values[index]))
                .collect::<Vec<_>>();
            // SAFETY: the completed command initialized one count per suffix round.
            let coefficient_counts = unsafe {
                std::slice::from_raw_parts(
                    coefficient_counts.contents().cast::<u32>(),
                    suffix_rounds,
                )
            }
            .iter()
            .map(|&count| count as usize)
            .collect::<Vec<_>>();
            // SAFETY: the completed command initialized every suffix challenge.
            let challenge_values = unsafe {
                std::slice::from_raw_parts(challenges.contents().cast::<Fp128Limbs>(), total_rounds)
            }[prefix_challenges.len()..]
                .to_vec();
            // SAFETY: the final fold initialized both scalar outputs when present.
            let final_evaluation = unsafe { *session.final_output.contents().cast() };
            let final_linear_evaluation = if had_linear_terms {
                unsafe { *session.linear_final_output.contents().cast() }
            } else {
                Fp128Limbs::default()
            };
            let mut next_chaining_value = [0u8; 64];
            // SAFETY: `state` remains a 64-byte shared buffer after completion.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    state.contents().cast::<u8>(),
                    next_chaining_value.as_mut_ptr(),
                    next_chaining_value.len(),
                );
            }
            let readback_copy = readback_start.elapsed();

            session.current_len = current_len;
            session.current_live_len = current_live_len;
            session.current_table = current_table;
            session.current_lane_weight_table = current_lane_weight_table;
            session.current_lane_count = current_lane_count;
            session.rounds_folded = rounds_folded;
            let equality_bytes = equality_schedule
                .iter()
                .try_fold(0usize, |bytes, round| {
                    bytes
                        .checked_add(size_of_val(round.e_first.as_slice()))
                        .and_then(|sum| sum.checked_add(size_of_val(round.e_second.as_slice())))
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct relation equality buffers",
                ))?;
            let mapping_bytes = additional_schedule
                .iter()
                .try_fold(0usize, |bytes, mappings| {
                    bytes.checked_add(size_of_val(mappings.as_slice()))
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct relation additional mappings",
                ))?;
            let allocation_bytes = equality_bytes
                .checked_add(size_of_val(taus.as_slice()))
                .and_then(|bytes| bytes.checked_add(2 * alpha_bytes))
                .and_then(|bytes| bytes.checked_add(size_of_val(scalar_values.as_slice())))
                .and_then(|bytes| bytes.checked_add(2 * pair_bytes))
                .and_then(|bytes| bytes.checked_add(mapping_bytes))
                .and_then(|bytes| bytes.checked_add(size_of_val(challenge_values.as_slice())))
                .and_then(|bytes| bytes.checked_add(proof_bytes))
                .and_then(|bytes| bytes.checked_add(count_bytes + 64))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct relation allocation bytes",
                ))?;
            Ok(DirectRelationResidentOutcome {
                round_coefficients,
                coefficient_counts,
                challenges: challenge_values,
                final_evaluation,
                final_linear_evaluation,
                chaining_value: next_chaining_value,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_direct_relation_initial(
        &self,
        session: &DirectRelationSession,
        round: DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationRoundOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let mut params = direct_relation_params(
                session,
                session.current_len,
                session.current_lane_count,
                false,
                &round,
            )?;
            let prefix_weight = [Fp128Limbs::from_u128(1)];
            params.prefix_size = 1;
            let buffer_start = Instant::now();
            let buffers = self.direct_relation_round_buffers(&round)?;
            let prefix = self.shared_buffer_from_slice(&prefix_weight)?;
            let additional_pairs = if round.additional_pairs.is_empty() {
                None
            } else {
                Some(self.shared_buffer_from_slice(round.additional_pairs)?)
            };
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct relation initial round");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 direct relation initial partials");
            encoder.set_compute_pipeline_state(&self.fp128_direct_relation_initial_pipeline);
            encoder.set_buffer(0, Some(&session.compact_digits), 0);
            self.encode_direct_relation_round_buffers(
                encoder,
                1,
                session,
                &buffers,
                &session.lane_weight_tables[session.current_lane_weight_table],
            );
            encoder.set_buffer(10, Some(&session.partials), 0);
            encoder.set_buffer(11, Some(&prefix), 0);
            set_inline_bytes(encoder, 12, &round.scalars);
            set_inline_bytes(encoder, 13, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.workgroups, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            encode_direct_relation_reduction(
                command,
                &self.fp128_direct_range_reduce_pipeline,
                &session.partials,
                &session.round_output,
                &params,
            );
            if let Some(additional_pairs) = &additional_pairs {
                self.encode_direct_relation_additional_compact(
                    command,
                    session,
                    &prefix,
                    additional_pairs,
                    &round.scalars,
                    &params,
                );
            }
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = read_direct_relation_coefficients(&session.round_output);
            let additional_coefficients = if additional_pairs.is_some() {
                read_direct_relation_coefficients(&session.additional_output)
            } else {
                [Fp128Limbs::default(); FP128_DIRECT_RELATION_STORED_COEFFICIENTS]
            };
            let readback_copy = readback_start.elapsed();
            Ok(DirectRelationRoundOutcome {
                coefficients,
                additional_coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: buffers.allocation_bytes
                    + size_of_val(&prefix_weight)
                    + size_of_val(round.additional_pairs),
            })
        })
    }

    pub(crate) fn dispatch_fp128_direct_relation_advance(
        &self,
        session: &mut DirectRelationSession,
        challenge: Fp128Limbs,
        prefix_weights: &[Fp128Limbs],
        next_round: Option<DirectRelationRoundData<'_>>,
    ) -> Result<DirectRelationAdvanceOutcome, MetalCommitError> {
        autoreleasepool(|| {
            if session.current_len < 2 || !session.current_len.is_power_of_two() {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation session has no foldable table".into(),
                ));
            }
            let next_len = session.current_len / 2;
            if next_len == 1 {
                if next_round.is_some() {
                    return Err(MetalCommitError::UnsupportedShape(
                        "final direct relation fold received another round".into(),
                    ));
                }
                let current_table = session.current_table.ok_or_else(|| {
                    MetalCommitError::UnsupportedShape(
                        "direct relation compact domain is too small".into(),
                    )
                })?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita fp128 direct relation final fold");
                self.encode_direct_relation_linear_fold(command, session, challenge)?;
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.fp128_direct_range_finalize_pipeline);
                encoder.set_buffer(0, Some(&session.tables[current_table]), 0);
                encoder.set_buffer(1, Some(&session.final_output), 0);
                set_inline_bytes(encoder, 2, &challenge);
                set_inline_bytes(encoder, 3, &(session.current_live_len as u64));
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();
                if session.linear_mode != 0 {
                    let encoder = command.new_blit_command_encoder();
                    encoder.copy_from_buffer(
                        &session.linear_tables[session.current_linear_table],
                        0,
                        &session.linear_final_output,
                        0,
                        size_of::<Fp128Limbs>() as u64,
                    );
                    encoder.end_encoding();
                }
                let (command_wall, gpu) = complete_command(command)?;
                let readback_start = Instant::now();
                // SAFETY: `final_output` contains one initialized fp128 value.
                let final_evaluation =
                    unsafe { *session.final_output.contents().cast::<Fp128Limbs>() };
                let final_linear_evaluation = if session.linear_mode == 0 {
                    Fp128Limbs::default()
                } else {
                    // SAFETY: the final command copied one initialized fp128 value.
                    unsafe { *session.linear_final_output.contents().cast::<Fp128Limbs>() }
                };
                let readback_copy = readback_start.elapsed();
                session.current_len = 1;
                session.current_live_len = session.current_live_len.div_ceil(2);
                return Ok(DirectRelationAdvanceOutcome {
                    next_coefficients: None,
                    next_additional_coefficients: None,
                    final_evaluation: Some(final_evaluation),
                    final_linear_evaluation: Some(final_linear_evaluation),
                    timings: DispatchTimings {
                        buffer_setup: Duration::ZERO,
                        command_wall,
                        gpu,
                        readback_copy,
                    },
                    allocation_bytes: 0,
                });
            }

            let round = next_round.ok_or_else(|| {
                MetalCommitError::UnsupportedShape(
                    "non-final direct relation fold is missing round data".into(),
                )
            })?;
            let fold_lane_weights = session.rounds_folded >= session.coefficient_rounds;
            let next_lane_count = if fold_lane_weights {
                if session.current_lane_count < 2 || !session.current_lane_count.is_power_of_two() {
                    return Err(MetalCommitError::UnsupportedShape(
                        "direct relation lane-weight table is not foldable".into(),
                    ));
                }
                session.current_lane_count / 2
            } else {
                session.current_lane_count
            };
            let next_lane_weight_table = if fold_lane_weights {
                1 - session.current_lane_weight_table
            } else {
                session.current_lane_weight_table
            };
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct relation fold and next round");
            self.encode_direct_relation_linear_fold(command, session, challenge)?;
            let mut params = direct_relation_params(
                session,
                next_len,
                next_lane_count,
                fold_lane_weights,
                &round,
            )?;
            if params.current_len != next_len as u64 {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation round data does not match the folded table".into(),
                ));
            }
            let output_table = session.current_table.map_or(0, |current| 1 - current);
            let buffer_start = Instant::now();
            let buffers = self.direct_relation_round_buffers(&round)?;
            let prefix = if session.current_table.is_none() {
                let expected_prefix_size = 1usize
                    .checked_shl(u32::try_from(session.rounds_folded + 1).map_err(|_| {
                        MetalCommitError::ShapeOverflow("direct relation compact prefix width")
                    })?)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "direct relation compact prefix size",
                    ))?;
                if prefix_weights.len() != expected_prefix_size {
                    return Err(MetalCommitError::UnsupportedShape(
                        "direct relation compact prefix weights have the wrong length".into(),
                    ));
                }
                params.prefix_size = expected_prefix_size as u64;
                params.materialize_prefix =
                    u64::from(session.rounds_folded + 1 >= session.compact_prefix_rounds);
                Some(self.shared_buffer_from_slice(prefix_weights)?)
            } else {
                None
            };
            let additional_pairs = if round.additional_pairs.is_empty() {
                None
            } else {
                Some(self.shared_buffer_from_slice(round.additional_pairs)?)
            };
            let buffer_setup = buffer_start.elapsed();
            let encoder = command.new_compute_command_encoder();
            if let Some(current_table) = session.current_table {
                encoder.set_compute_pipeline_state(&self.fp128_direct_relation_field_fold_pipeline);
                encoder.set_buffer(0, Some(&session.tables[current_table]), 0);
                encoder.set_buffer(
                    15,
                    Some(&session.lane_weight_tables[session.current_lane_weight_table]),
                    0,
                );
            } else {
                if fold_lane_weights {
                    return Err(MetalCommitError::UnsupportedShape(
                        "direct relation cannot fold lane weights from compact witness state"
                            .into(),
                    ));
                }
                encoder
                    .set_compute_pipeline_state(&self.fp128_direct_relation_compact_fold_pipeline);
                encoder.set_buffer(0, Some(&session.compact_digits), 0);
            }
            encoder.set_buffer(1, Some(&session.tables[output_table]), 0);
            self.encode_direct_relation_round_buffers(
                encoder,
                2,
                session,
                &buffers,
                &session.lane_weight_tables[next_lane_weight_table],
            );
            encoder.set_buffer(11, Some(&session.partials), 0);
            if let Some(prefix) = &prefix {
                encoder.set_buffer(12, Some(prefix), 0);
            } else {
                set_inline_bytes(encoder, 12, &challenge);
            }
            set_inline_bytes(encoder, 13, &round.scalars);
            set_inline_bytes(encoder, 14, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.workgroups, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            encode_direct_relation_reduction(
                command,
                &self.fp128_direct_range_reduce_pipeline,
                &session.partials,
                &session.round_output,
                &params,
            );
            if let Some(additional_pairs) = &additional_pairs {
                if let Some(prefix) = &prefix {
                    self.encode_direct_relation_additional_compact(
                        command,
                        session,
                        prefix,
                        additional_pairs,
                        &round.scalars,
                        &params,
                    );
                } else {
                    self.encode_direct_relation_additional_field(
                        command,
                        &session.tables[output_table],
                        session,
                        additional_pairs,
                        &round.scalars,
                        &params,
                    );
                }
            }
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = read_direct_relation_coefficients(&session.round_output);
            let additional_coefficients = if additional_pairs.is_some() {
                read_direct_relation_coefficients(&session.additional_output)
            } else {
                [Fp128Limbs::default(); FP128_DIRECT_RELATION_STORED_COEFFICIENTS]
            };
            let readback_copy = readback_start.elapsed();
            if session.current_table.is_some()
                || session.rounds_folded + 1 >= session.compact_prefix_rounds
            {
                session.current_table = Some(output_table);
            }
            session.current_len = next_len;
            session.current_live_len = params.current_live_len as usize;
            session.rounds_folded += 1;
            if fold_lane_weights {
                session.current_lane_weight_table = next_lane_weight_table;
                session.current_lane_count = next_lane_count;
            }
            Ok(DirectRelationAdvanceOutcome {
                next_coefficients: Some(coefficients),
                next_additional_coefficients: Some(additional_coefficients),
                final_evaluation: None,
                final_linear_evaluation: None,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: buffers.allocation_bytes
                    + size_of_val(prefix_weights)
                    + size_of_val(round.additional_pairs),
            })
        })
    }

    pub(crate) fn direct_relation_session_allocation_bytes(
        &self,
        session: &DirectRelationSession,
    ) -> usize {
        session.allocation_bytes
    }

    fn encode_direct_relation_linear_fold(
        &self,
        command: &CommandBufferRef,
        session: &mut DirectRelationSession,
        challenge: Fp128Limbs,
    ) -> Result<(), MetalCommitError> {
        self.encode_direct_relation_linear_fold_with_binding(
            command,
            session,
            Fp128KernelBinding::Inline(challenge),
        )
    }

    fn encode_direct_relation_linear_fold_from_buffer(
        &self,
        command: &CommandBufferRef,
        session: &mut DirectRelationSession,
        challenges: &Buffer,
        challenge_offset: u64,
    ) -> Result<(), MetalCommitError> {
        self.encode_direct_relation_linear_fold_with_binding(
            command,
            session,
            Fp128KernelBinding::Buffer(challenges, challenge_offset),
        )
    }

    fn encode_direct_relation_linear_fold_with_binding(
        &self,
        command: &CommandBufferRef,
        session: &mut DirectRelationSession,
        challenge: Fp128KernelBinding<'_>,
    ) -> Result<(), MetalCommitError> {
        if session.linear_mode == 0 {
            return Ok(());
        }
        let folding_coefficients = session.rounds_folded < session.coefficient_rounds;
        let (mode, output_len, next_coeff_count, next_live_lane_count) = if folding_coefficients {
            if session.linear_mode != 1 || session.linear_current_coeff_count < 2 {
                return Err(MetalCommitError::UnsupportedShape(
                    "factored direct relation source is not coefficient-foldable".into(),
                ));
            }
            let next_coeff_count = session.linear_current_coeff_count / 2;
            let output_len = session
                .linear_source_lane_count
                .checked_mul(next_coeff_count)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation folded linear source",
                ))?;
            (
                1usize,
                output_len,
                next_coeff_count,
                session.linear_current_live_lane_count,
            )
        } else {
            if session.linear_current_coeff_count != 1
                || session.linear_current_live_lane_count == 0
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation linear lane source is not foldable".into(),
                ));
            }
            let next_live_lane_count = session.linear_current_live_lane_count.div_ceil(2);
            let mode = if session.linear_mode == 1 { 2 } else { 3 };
            (mode, next_live_lane_count, 1, next_live_lane_count)
        };
        let params = DirectRelationLinearFoldParams {
            current_coeff_count: u64::try_from(session.linear_current_coeff_count)
                .map_err(|_| MetalCommitError::ShapeOverflow("direct relation linear width"))?,
            source_lane_count: u64::try_from(session.linear_source_lane_count).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation linear source lanes")
            })?,
            current_live_lane_count: u64::try_from(session.linear_current_live_lane_count)
                .map_err(|_| MetalCommitError::ShapeOverflow("direct relation linear lanes"))?,
            output_len: u64::try_from(output_len).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation folded linear output")
            })?,
            mode: mode as u64,
        };
        let output_table = 1 - session.current_linear_table;
        let encoder = command.new_compute_command_encoder();
        encoder.set_label("Akita fp128 resident direct relation linear fold");
        encoder.set_compute_pipeline_state(&self.fp128_direct_relation_linear_fold_pipeline);
        encoder.set_buffer(
            0,
            Some(&session.linear_tables[session.current_linear_table]),
            0,
        );
        encoder.set_buffer(1, Some(&session.linear_tables[output_table]), 0);
        encoder.set_buffer(2, Some(&session.linear_source_lane_offsets), 0);
        encoder.set_buffer(3, Some(&session.lane_offsets), 0);
        encoder.set_buffer(4, Some(&session.lane_segments), 0);
        encoder.set_buffer(5, Some(&session.linear_segments), 0);
        set_fp128_binding(encoder, 6, challenge);
        set_inline_bytes(encoder, 7, &params);
        encoder.dispatch_threads(
            MTLSize::new(params.output_len, 1, 1),
            MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
        );
        encoder.end_encoding();
        session.current_linear_table = output_table;
        session.linear_current_coeff_count = next_coeff_count;
        session.linear_current_live_lane_count = next_live_lane_count;
        if !folding_coefficients {
            session.linear_mode = 2;
        }
        Ok(())
    }

    fn direct_relation_round_buffers(
        &self,
        round: &DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationRoundBuffers, MetalCommitError> {
        let allocation_bytes = size_of_val(round.e_first)
            .checked_add(size_of_val(round.e_second))
            .and_then(|bytes| bytes.checked_add(size_of_val(round.alpha)))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation round allocation bytes",
            ))?;
        Ok(DirectRelationRoundBuffers {
            e_first: self.shared_buffer_from_slice(round.e_first)?,
            e_second: self.shared_buffer_from_slice(round.e_second)?,
            alpha: self.shared_buffer_from_slice(round.alpha)?,
            allocation_bytes,
        })
    }

    fn encode_direct_relation_round_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        start: u64,
        session: &DirectRelationSession,
        buffers: &DirectRelationRoundBuffers,
        lane_weights: &Buffer,
    ) {
        encoder.set_buffer(start, Some(&buffers.e_first), 0);
        encoder.set_buffer(start + 1, Some(&buffers.e_second), 0);
        encoder.set_buffer(start + 2, Some(&buffers.alpha), 0);
        encoder.set_buffer(start + 3, Some(lane_weights), 0);
        encoder.set_buffer(
            start + 4,
            Some(&session.linear_tables[session.current_linear_table]),
            0,
        );
        encoder.set_buffer(start + 5, Some(&session.linear_source_lane_offsets), 0);
        encoder.set_buffer(start + 6, Some(&session.lane_offsets), 0);
        encoder.set_buffer(start + 7, Some(&session.lane_segments), 0);
        encoder.set_buffer(start + 8, Some(&session.linear_segments), 0);
    }

    fn encode_direct_relation_additional_compact(
        &self,
        command: &CommandBufferRef,
        session: &DirectRelationSession,
        prefix: &Buffer,
        pairs: &Buffer,
        scalars: &DirectRelationScalars,
        params: &DirectRelationParams,
    ) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.fp128_direct_relation_additional_compact_pipeline);
        encoder.set_buffer(0, Some(&session.compact_digits), 0);
        encoder.set_buffer(1, Some(prefix), 0);
        encoder.set_buffer(2, Some(pairs), 0);
        encoder.set_buffer(3, Some(&session.partials), 0);
        set_inline_bytes(encoder, 4, scalars);
        set_inline_bytes(encoder, 5, params);
        encoder.dispatch_thread_groups(
            MTLSize::new(params.additional_workgroups, 1, 1),
            MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
        );
        encoder.end_encoding();
        let mut reduction_params = *params;
        reduction_params.workgroups = params.additional_workgroups;
        encode_direct_relation_reduction(
            command,
            &self.fp128_direct_range_reduce_pipeline,
            &session.partials,
            &session.additional_output,
            &reduction_params,
        );
    }

    fn encode_direct_relation_additional_field(
        &self,
        command: &CommandBufferRef,
        witness: &Buffer,
        session: &DirectRelationSession,
        pairs: &Buffer,
        scalars: &DirectRelationScalars,
        params: &DirectRelationParams,
    ) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.fp128_direct_relation_additional_field_pipeline);
        encoder.set_buffer(0, Some(witness), 0);
        encoder.set_buffer(1, Some(pairs), 0);
        encoder.set_buffer(2, Some(&session.partials), 0);
        set_inline_bytes(encoder, 3, scalars);
        set_inline_bytes(encoder, 4, params);
        encoder.dispatch_thread_groups(
            MTLSize::new(params.additional_workgroups, 1, 1),
            MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
        );
        encoder.end_encoding();
        let mut reduction_params = *params;
        reduction_params.workgroups = params.additional_workgroups;
        encode_direct_relation_reduction(
            command,
            &self.fp128_direct_range_reduce_pipeline,
            &session.partials,
            &session.additional_output,
            &reduction_params,
        );
    }

    pub(crate) fn dispatch_packed_onehot(
        &self,
        matrix: &Buffer,
        lanes: &[u8],
        active_zero_rows: &[u64],
        params: PackedOneHotCommitParams,
        streams_per_command: usize,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        let expected_active_words = params.num_rows.div_ceil(u64::BITS as u64);
        let zero_mask_exceeds_columns = params.num_columns < u64::BITS as u64
            && params.zero_column_mask >> params.num_columns != 0;
        if zero_mask_exceeds_columns
            || (params.zero_column_mask == 0 && !active_zero_rows.is_empty())
            || (params.zero_column_mask != 0
                && u64::try_from(active_zero_rows.len()).ok() != Some(expected_active_words))
        {
            return Err(MetalCommitError::UnsupportedShape(
                "packed committed-zero selector geometry is invalid".into(),
            ));
        }
        self.dispatch_packed_onehot_source(
            matrix,
            &ResidentPackedLanes { lanes },
            active_zero_rows,
            params,
            streams_per_command,
        )
    }

    fn dispatch_packed_onehot_source<S: PackedLaneSource>(
        &self,
        matrix: &Buffer,
        source: &S,
        active_zero_rows: &[u64],
        mut params: PackedOneHotCommitParams,
        streams_per_command: usize,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let lane_count = params
                .num_rows
                .checked_mul(params.lane_stride)
                .ok_or(MetalCommitError::ShapeOverflow("packed lane count"))?;
            let expected_tasks = params
                .num_columns
                .checked_mul(params.full_blocks_per_column)
                .ok_or(MetalCommitError::ShapeOverflow("packed task count"))?;
            let expected_output = params
                .column_capacity
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(params.n_a))
                .and_then(|count| count.checked_mul(params.ring_d))
                .ok_or(MetalCommitError::ShapeOverflow("packed output"))?;
            let supported_source = matches!(
                (params.onehot_k, params.column_capacity),
                (16, 64) | (256, 32)
            );
            if lane_count != source.lane_count() as u64
                || params.ring_d != 512
                || !supported_source
                || params.num_columns == 0
                || params.num_columns > params.column_capacity
                || params.lane_stride != params.num_columns
                || params.n_a != 1
                || params.num_digits_inner != 1
                || params.position_partials_per_block != FP128_D512_POSITION_PARTIALS as u64
                || !params.positions_per_partial.is_multiple_of(4)
                || !params.blocks_per_column.is_power_of_two()
                || params.blocks_per_column > 512
                || params.full_blocks_per_column > params.blocks_per_column
                || params.boundary_columns != 0
                || params.num_blocks != expected_tasks
                || params.task_offset != 0
                || params.dispatch_tasks != params.num_blocks
                || params.lane_row_offset != 0
                || params.output_coefficients != expected_output
                || !matches!(streams_per_command, 1 | 5)
                || (streams_per_command == 5 && params.positions_per_block > (1 << 16))
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D512 dispatch geometry is outside the registered schedule".into(),
                ));
            }
            let expected_matrix_bytes = params
                .n_a
                .checked_mul(params.positions_per_block)
                .and_then(|count| count.checked_mul(params.ring_d))
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64))
                .ok_or(MetalCommitError::ShapeOverflow("packed matrix bytes"))?;
            if matrix.length() < expected_matrix_bytes {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D512 matrix prefix is shorter than the plan".into(),
                ));
            }
            if !self.supports_packed_fp128_d512_panels() {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D512 pipeline does not expose the registered resources".into(),
                ));
            }
            params.columns_per_threadgroup = 1;
            let base_streams = params
                .num_blocks
                .div_ceil(FP128_D512_TASKS_PER_STREAM as u64);
            let matrix_block_streams = base_streams
                .checked_mul(FP128_D512_COEFFICIENT_BANDS as u64)
                .ok_or(MetalCommitError::ShapeOverflow("coefficient-band streams"))?;
            let threadgroups = params
                .n_a
                .checked_mul(params.position_partials_per_block)
                .and_then(|count| count.checked_mul(matrix_block_streams))
                .ok_or(MetalCommitError::ShapeOverflow("packed threadgroups"))?;
            if threadgroups > u64::from(u32::MAX)
                || params.output_coefficients > u64::from(u32::MAX)
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D512 grid exceeds u32 indexing".into(),
                ));
            }

            let buffer_start = Instant::now();
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("output coefficients"))?;
            let output_bytes = output_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("output bytes"))?;
            let output = self.shared_buffer(output_bytes)?;
            let partial_count = output_count
                .checked_mul(params.position_partials_per_block as usize)
                .ok_or(MetalCommitError::ShapeOverflow("partial coefficients"))?;
            let scratch_bytes = partial_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("partial bytes"))?;
            let partials = self.private_buffer(scratch_bytes)?;
            let no_active_zero_rows = [0u64];
            let active_zero_rows = if active_zero_rows.is_empty() {
                no_active_zero_rows.as_slice()
            } else {
                active_zero_rows
            };
            let active_zero_rows = self.shared_buffer_from_slice(active_zero_rows)?;
            let mut buffer_setup = buffer_start.elapsed();

            let command_start = Instant::now();
            let command_count = base_streams.div_ceil(streams_per_command as u64) as usize;
            let mut commands = Vec::with_capacity(command_count);
            let mut lane_buffers = Vec::with_capacity(command_count);
            let mut input_zero_copy = true;
            for first_stream in (0..base_streams).step_by(streams_per_command) {
                let stream_count = (base_streams - first_stream).min(streams_per_command as u64);
                let task_offset = first_stream * FP128_D512_TASKS_PER_STREAM as u64;
                let mut dispatch_params = params;
                dispatch_params.task_offset = task_offset;
                dispatch_params.dispatch_tasks = (stream_count
                    * FP128_D512_TASKS_PER_STREAM as u64)
                    .min(params.num_blocks - task_offset);
                let final_task = task_offset + dispatch_params.dispatch_tasks - 1;
                let first_block = task_offset / params.num_columns;
                let final_block = final_task / params.num_columns;
                let rows_per_position = params.ring_d / params.onehot_k;
                let rows_per_block = params.positions_per_block * rows_per_position;
                let first_row = first_block * rows_per_block;
                let final_row = (final_block + 1) * rows_per_block;
                let first_row = usize::try_from(first_row)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed first row"))?;
                let final_row = usize::try_from(final_row)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed final row"))?;
                let lane_stride = usize::try_from(params.lane_stride)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed lane stride"))?;
                let command_lanes = source.wait_lanes(first_row..final_row, lane_stride)?;
                let lane_buffer_start = Instant::now();
                let lane_buffer = self.packed_lane_buffer(command_lanes)?;
                buffer_setup += lane_buffer_start.elapsed();
                input_zero_copy &= lane_buffer.zero_copy;
                dispatch_params.lane_row_offset = first_row as u64;
                let command_threadgroups = dispatch_params
                    .n_a
                    .checked_mul(dispatch_params.position_partials_per_block)
                    .and_then(|count| {
                        count.checked_mul(stream_count * FP128_D512_COEFFICIENT_BANDS as u64)
                    })
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "packed command threadgroups",
                    ))?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita packed fp128 D512 root commitment");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita packed fp128 D512 coefficient bands");
                encoder.set_compute_pipeline_state(&self.packed_fp128_d512_pipeline);
                encoder.set_buffer(0, Some(matrix), 0);
                encoder.set_buffer(1, Some(&lane_buffer.buffer), 0);
                encoder.set_buffer(2, Some(&partials), 0);
                set_inline_bytes(encoder, 3, &dispatch_params);
                encoder.set_buffer(4, Some(&active_zero_rows), 0);
                encoder.dispatch_thread_groups(
                    MTLSize::new(command_threadgroups, 1, 1),
                    MTLSize::new(FP128_D512_THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                command.commit();
                commands.push(command);
                lane_buffers.push(lane_buffer);
            }

            let reduction_command = self.queue.new_command_buffer();
            reduction_command.set_label("Akita packed fp128 D512 partial reduction");
            let reduction = reduction_command.new_compute_command_encoder();
            reduction.set_label("Akita packed fp128 D512 partial reduction");
            reduction.set_compute_pipeline_state(&self.packed_partial_reduction_pipeline);
            reduction.set_buffer(0, Some(&partials), 0);
            reduction.set_buffer(1, Some(&output), 0);
            set_inline_bytes(reduction, 2, &params);
            let reduction_width = (self
                .packed_partial_reduction_pipeline
                .thread_execution_width()
                * 4)
            .min(
                self.packed_partial_reduction_pipeline
                    .max_total_threads_per_threadgroup(),
            );
            reduction.dispatch_threads(
                MTLSize::new(params.output_coefficients, 1, 1),
                MTLSize::new(reduction_width, 1, 1),
            );
            reduction.end_encoding();
            reduction_command.commit();
            reduction_command.wait_until_completed();
            let command_wall = command_start.elapsed();
            for command in &commands {
                validate_completed_command(command)?;
            }
            validate_completed_command(reduction_command)?;
            let panel_gpu_active = commands.iter().try_fold(Duration::ZERO, |total, command| {
                total.checked_add(completed_command_gpu_time(command)?)
            });
            let panel_gpu_span = commands
                .first()
                .zip(commands.last())
                .and_then(|(first, last)| completed_commands_gpu_span(first, last));
            let reduction_gpu = completed_command_gpu_time(reduction_command);
            let gpu = commands
                .first()
                .and_then(|first| completed_commands_gpu_span(first, reduction_command));

            let readback_start = Instant::now();
            // SAFETY: `output` is live shared storage for exactly `output_count`
            // aligned `Fp128Limbs` values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(DispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                panel_gpu_active,
                panel_gpu_span,
                reduction_gpu,
                command_buffers: commands.len() + 1,
                kernel: MetalOneHotKernel::PackedFp128D512Panels,
                blocks_per_threadgroup: FP128_D512_TASKS_PER_STREAM,
                columns_per_threadgroup: 1,
                matrix_block_streams: matrix_block_streams as usize,
                scratch_bytes,
                input_zero_copy,
            })
        })
    }

    /// Dispatch the packed fp128 D128 rank-3 commit: threadgroups own one
    /// matrix element and one position partial for 64 (column, block) tasks.
    pub(crate) fn dispatch_packed_onehot_d128_rank3(
        &self,
        matrix: &Buffer,
        lanes: &[u8],
        active_zero_rows: &[u64],
        mut params: PackedOneHotCommitParams,
        streams_per_command: usize,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let expected_active_words = params.num_rows.div_ceil(u64::BITS as u64);
            let zero_mask_exceeds_columns = params.num_columns < u64::BITS as u64
                && params.zero_column_mask >> params.num_columns != 0;
            if zero_mask_exceeds_columns
                || (params.zero_column_mask == 0 && !active_zero_rows.is_empty())
                || (params.zero_column_mask != 0
                    && u64::try_from(active_zero_rows.len()).ok() != Some(expected_active_words))
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed committed-zero selector geometry is invalid".into(),
                ));
            }
            let lane_count = params
                .num_rows
                .checked_mul(params.lane_stride)
                .ok_or(MetalCommitError::ShapeOverflow("packed lane count"))?;
            let expected_tasks = params
                .num_columns
                .checked_mul(params.full_blocks_per_column)
                .ok_or(MetalCommitError::ShapeOverflow("packed task count"))?;
            let expected_output = params
                .column_capacity
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(params.n_a))
                .and_then(|count| count.checked_mul(params.ring_d))
                .ok_or(MetalCommitError::ShapeOverflow("packed output"))?;
            let partial_alignment = FP128_D128_RANK3_POSITION_PARTIAL_ALIGNMENT as u64;
            if lane_count != lanes.len() as u64
                || params.ring_d != FP128_D128_RANK3_RING_D
                || params.onehot_k != 256
                || params.column_capacity != 32
                || params.num_columns == 0
                || params.num_columns > params.column_capacity
                || params.lane_stride != params.num_columns
                || params.n_a != FP128_D128_RANK3_INNER_RANK
                || params.num_digits_inner != 1
                || params.position_partials_per_block != FP128_D512_POSITION_PARTIALS as u64
                || !params
                    .positions_per_partial
                    .is_multiple_of(partial_alignment)
                || !params.positions_per_block.is_multiple_of(2)
                || !params.blocks_per_column.is_power_of_two()
                || params.blocks_per_column > (1 << 12)
                || params.full_blocks_per_column > params.blocks_per_column
                || params.boundary_columns != 0
                || params.num_blocks != expected_tasks
                || params.task_offset != 0
                || params.dispatch_tasks != params.num_blocks
                || params.lane_row_offset != 0
                || params.output_coefficients != expected_output
                || streams_per_command == 0
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D128 rank-3 dispatch geometry is outside the registered schedule"
                        .into(),
                ));
            }
            let expected_matrix_bytes = params
                .n_a
                .checked_mul(params.positions_per_block)
                .and_then(|count| count.checked_mul(params.ring_d))
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64))
                .ok_or(MetalCommitError::ShapeOverflow("packed matrix bytes"))?;
            if matrix.length() < expected_matrix_bytes {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D128 rank-3 matrix prefix is shorter than the plan".into(),
                ));
            }
            if !self.supports_packed_fp128_d128_rank3() {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D128 rank-3 pipeline does not expose the registered resources"
                        .into(),
                ));
            }
            params.columns_per_threadgroup = 1;
            let tasks_per_stream = FP128_D128_RANK3_TASKS_PER_STREAM as u64;
            let base_streams = params.num_blocks.div_ceil(tasks_per_stream);
            let groups_per_stream = params
                .n_a
                .checked_mul(params.position_partials_per_block)
                .ok_or(MetalCommitError::ShapeOverflow("packed partial groups"))?;
            let threadgroups = base_streams
                .checked_mul(groups_per_stream)
                .ok_or(MetalCommitError::ShapeOverflow("packed threadgroups"))?;
            if threadgroups > u64::from(u32::MAX)
                || params.output_coefficients > u64::from(u32::MAX)
                || params
                    .output_coefficients
                    .checked_mul(params.position_partials_per_block)
                    .is_none_or(|count| count > u64::from(u32::MAX))
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D128 rank-3 grid exceeds u32 indexing".into(),
                ));
            }

            let buffer_start = Instant::now();
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("output coefficients"))?;
            let output_bytes = output_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("output bytes"))?;
            let output = self.shared_buffer(output_bytes)?;
            let partial_count = output_count
                .checked_mul(params.position_partials_per_block as usize)
                .ok_or(MetalCommitError::ShapeOverflow("partial coefficients"))?;
            let scratch_bytes = partial_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("partial bytes"))?;
            let partials = self.private_buffer(scratch_bytes)?;
            let no_active_zero_rows = [0u64];
            let active_zero_rows = if active_zero_rows.is_empty() {
                no_active_zero_rows.as_slice()
            } else {
                active_zero_rows
            };
            let active_zero_rows = self.shared_buffer_from_slice(active_zero_rows)?;
            let mut buffer_setup = buffer_start.elapsed();

            let source = ResidentPackedLanes { lanes };
            let command_start = Instant::now();
            let command_count = base_streams.div_ceil(streams_per_command as u64) as usize;
            let mut commands = Vec::with_capacity(command_count);
            let mut lane_buffers = Vec::with_capacity(command_count);
            let mut input_zero_copy = true;
            // Two ring positions per trace row at K = 256, D = 128.
            let rows_per_block = params.positions_per_block / 2;
            for first_stream in (0..base_streams).step_by(streams_per_command) {
                let stream_count = (base_streams - first_stream).min(streams_per_command as u64);
                let task_offset = first_stream * tasks_per_stream;
                let mut dispatch_params = params;
                dispatch_params.task_offset = task_offset;
                dispatch_params.dispatch_tasks =
                    (stream_count * tasks_per_stream).min(params.num_blocks - task_offset);
                let final_task = task_offset + dispatch_params.dispatch_tasks - 1;
                let first_block = task_offset / params.num_columns;
                let final_block = final_task / params.num_columns;
                let first_row = usize::try_from(first_block * rows_per_block)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed first row"))?;
                let final_row = usize::try_from((final_block + 1) * rows_per_block)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed final row"))?;
                let lane_stride = usize::try_from(params.lane_stride)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed lane stride"))?;
                let command_lanes = source.wait_lanes(first_row..final_row, lane_stride)?;
                let lane_buffer_start = Instant::now();
                let lane_buffer = self.packed_lane_buffer(command_lanes)?;
                buffer_setup += lane_buffer_start.elapsed();
                input_zero_copy &= lane_buffer.zero_copy;
                dispatch_params.lane_row_offset = first_row as u64;
                let command_threadgroups = stream_count.checked_mul(groups_per_stream).ok_or(
                    MetalCommitError::ShapeOverflow("packed command threadgroups"),
                )?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita packed fp128 D128 rank-3 root commitment");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita packed fp128 D128 rank-3 element tiles");
                encoder.set_compute_pipeline_state(&self.packed_fp128_d128_rank3_pipeline);
                encoder.set_buffer(0, Some(matrix), 0);
                encoder.set_buffer(1, Some(&lane_buffer.buffer), 0);
                encoder.set_buffer(2, Some(&partials), 0);
                set_inline_bytes(encoder, 3, &dispatch_params);
                encoder.set_buffer(4, Some(&active_zero_rows), 0);
                encoder.dispatch_thread_groups(
                    MTLSize::new(command_threadgroups, 1, 1),
                    MTLSize::new(FP128_D512_THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                command.commit();
                commands.push(command);
                lane_buffers.push(lane_buffer);
            }

            let reduction_command = self.queue.new_command_buffer();
            reduction_command.set_label("Akita packed fp128 D128 rank-3 partial reduction");
            let reduction = reduction_command.new_compute_command_encoder();
            reduction.set_label("Akita packed fp128 D128 rank-3 partial reduction");
            reduction.set_compute_pipeline_state(&self.packed_partial_reduction_pipeline);
            reduction.set_buffer(0, Some(&partials), 0);
            reduction.set_buffer(1, Some(&output), 0);
            set_inline_bytes(reduction, 2, &params);
            let reduction_width = (self
                .packed_partial_reduction_pipeline
                .thread_execution_width()
                * 4)
            .min(
                self.packed_partial_reduction_pipeline
                    .max_total_threads_per_threadgroup(),
            );
            reduction.dispatch_threads(
                MTLSize::new(params.output_coefficients, 1, 1),
                MTLSize::new(reduction_width, 1, 1),
            );
            reduction.end_encoding();
            reduction_command.commit();
            reduction_command.wait_until_completed();
            let command_wall = command_start.elapsed();
            for command in &commands {
                validate_completed_command(command)?;
            }
            validate_completed_command(reduction_command)?;
            let panel_gpu_active = commands.iter().try_fold(Duration::ZERO, |total, command| {
                total.checked_add(completed_command_gpu_time(command)?)
            });
            let panel_gpu_span = commands
                .first()
                .zip(commands.last())
                .and_then(|(first, last)| completed_commands_gpu_span(first, last));
            let reduction_gpu = completed_command_gpu_time(reduction_command);
            let gpu = commands
                .first()
                .and_then(|first| completed_commands_gpu_span(first, reduction_command));

            let readback_start = Instant::now();
            // SAFETY: `output` is live shared storage for exactly `output_count`
            // aligned `Fp128Limbs` values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(DispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                panel_gpu_active,
                panel_gpu_span,
                reduction_gpu,
                command_buffers: commands.len() + 1,
                kernel: MetalOneHotKernel::PackedFp128D128Rank3,
                blocks_per_threadgroup: FP128_D128_RANK3_TASKS_PER_STREAM,
                columns_per_threadgroup: 1,
                matrix_block_streams: threadgroups as usize,
                scratch_bytes,
                input_zero_copy,
            })
        })
    }

    fn dispatch_geometry(
        &self,
        params: OneHotCommitParams,
        kernel: MetalOneHotKernel,
    ) -> Result<usize, MetalCommitError> {
        match kernel {
            MetalOneHotKernel::DirectGather => Ok(1),
            MetalOneHotKernel::BlockBatched => {
                let ring_d = usize::try_from(params.ring_d)
                    .map_err(|_| MetalCommitError::ShapeOverflow("ring dimension"))?;
                let max_threads = self
                    .block_batched_pipeline
                    .max_total_threads_per_threadgroup() as usize;
                let blocks = max_threads / ring_d;
                if blocks == 0 {
                    Err(MetalCommitError::UnsupportedShape(format!(
                        "D={} exceeds the block-batched pipeline's {max_threads}-thread limit",
                        params.ring_d
                    )))
                } else {
                    Ok(blocks)
                }
            }
            MetalOneHotKernel::PackedFp128D512Panels | MetalOneHotKernel::PackedFp128D128Rank3 => {
                Err(MetalCommitError::UnsupportedShape(
                    "packed kernel requires packed parameters".into(),
                ))
            }
        }
    }

    fn encode_onehot(
        &self,
        encoder: &ComputeCommandEncoderRef,
        matrix: &Buffer,
        indices: &Buffer,
        output: &Buffer,
        params: &OneHotCommitParams,
        kernel: MetalOneHotKernel,
    ) -> Result<(), MetalCommitError> {
        let pipeline = match kernel {
            MetalOneHotKernel::DirectGather => &self.direct_pipeline,
            MetalOneHotKernel::BlockBatched => &self.block_batched_pipeline,
            MetalOneHotKernel::PackedFp128D512Panels | MetalOneHotKernel::PackedFp128D128Rank3 => {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed kernel requires packed parameters".into(),
                ));
            }
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(matrix), 0);
        encoder.set_buffer(1, Some(indices), 0);
        encoder.set_buffer(2, Some(output), 0);
        set_inline_bytes(encoder, 3, params);
        match kernel {
            MetalOneHotKernel::DirectGather => {
                let width = (pipeline.thread_execution_width() * 4)
                    .min(pipeline.max_total_threads_per_threadgroup());
                encoder.dispatch_threads(
                    MTLSize::new(params.output_coefficients, 1, 1),
                    MTLSize::new(width, 1, 1),
                );
            }
            MetalOneHotKernel::BlockBatched => {
                let block_groups = params.num_blocks.div_ceil(params.blocks_per_threadgroup);
                let threadgroups = params
                    .num_sources
                    .checked_mul(params.n_a)
                    .and_then(|count| count.checked_mul(block_groups))
                    .ok_or(MetalCommitError::ShapeOverflow("block threadgroups"))?;
                encoder.dispatch_thread_groups(
                    MTLSize::new(threadgroups, 1, 1),
                    MTLSize::new(params.blocks_per_threadgroup * params.ring_d, 1, 1),
                );
            }
            MetalOneHotKernel::PackedFp128D512Panels | MetalOneHotKernel::PackedFp128D128Rank3 => {
                unreachable!()
            }
        }
        Ok(())
    }
}

fn direct_range_workgroups(pair_count: usize) -> usize {
    pair_count
        .div_ceil(FP128_DIRECT_RANGE_THREADS)
        .clamp(1, FP128_DIRECT_RANGE_MAX_WORKGROUPS)
}

fn direct_range_params(
    live_len: usize,
    current_len: usize,
    current_live_len: usize,
    input_live_len: usize,
    e_first: &[Fp128Limbs],
    e_second: &[Fp128Limbs],
    basis: usize,
) -> Result<DirectRangeParams, MetalCommitError> {
    let domain_pair_count = current_len / 2;
    let pair_count = current_live_len.div_ceil(2);
    let equality_entries =
        e_first
            .len()
            .checked_mul(e_second.len())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range equality entries",
            ))?;
    if !matches!(basis, 4 | 8)
        || e_first.is_empty()
        || e_second.is_empty()
        || !e_first.len().is_power_of_two()
        || !e_second.len().is_power_of_two()
        || equality_entries != domain_pair_count
        || current_live_len > current_len
        || pair_count > domain_pair_count
    {
        return Err(MetalCommitError::UnsupportedShape(
            "direct range equality factors do not match the round geometry".into(),
        ));
    }
    Ok(DirectRangeParams {
        live_len: u64::try_from(live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range live length"))?,
        current_len: u64::try_from(current_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range current length"))?,
        current_live_len: u64::try_from(current_live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range current live length"))?,
        input_live_len: u64::try_from(input_live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range input live length"))?,
        pair_count: u64::try_from(pair_count)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range pair count"))?,
        num_first: u64::try_from(e_first.len())
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range first equality table"))?,
        num_second: u64::try_from(e_second.len())
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range second equality table"))?,
        workgroups: u64::try_from(direct_range_workgroups(pair_count))
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range workgroups"))?,
        basis: basis as u64,
        prefix_size: 1,
        materialize_prefix: 0,
        resident_challenges: 0,
    })
}

fn direct_relation_params(
    session: &DirectRelationSession,
    current_len: usize,
    lane_count: usize,
    fold_lane_weights: bool,
    round: &DirectRelationRoundData<'_>,
) -> Result<DirectRelationParams, MetalCommitError> {
    direct_relation_params_shape(
        session,
        current_len,
        lane_count,
        fold_lane_weights,
        round.e_first.len(),
        round.e_second.len(),
        round.alpha.len(),
        round.live_lane_count,
        round.additional_pairs.len(),
        !round
            .additional_pairs
            .iter()
            .any(|pair| pair.parent >= (current_len / 2) as u64),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "resident relation rounds provide table geometry without host field values"
)]
fn direct_relation_params_shape(
    session: &DirectRelationSession,
    current_len: usize,
    lane_count: usize,
    fold_lane_weights: bool,
    e_first_len: usize,
    e_second_len: usize,
    alpha_len: usize,
    live_lane_count: usize,
    additional_pair_count: usize,
    additional_parents_in_range: bool,
) -> Result<DirectRelationParams, MetalCommitError> {
    if current_len < 2 || !current_len.is_power_of_two() {
        return Err(MetalCommitError::UnsupportedShape(
            "direct relation round length is malformed".into(),
        ));
    }
    let domain_pair_count = current_len / 2;
    let current_live_len =
        alpha_len
            .checked_mul(live_lane_count)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation current live length",
            ))?;
    let active_pair_count = current_live_len.div_ceil(2);
    let pair_count = if fold_lane_weights {
        domain_pair_count
    } else {
        active_pair_count
    };
    let equality_entries =
        e_first_len
            .checked_mul(e_second_len)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation equality entries",
            ))?;
    let relation_entries =
        alpha_len
            .checked_mul(lane_count)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation rank-one entries",
            ))?;
    let linear_is_valid = match session.linear_mode {
        0 => true,
        1 => {
            session.linear_source_lane_count != 0 && session.linear_current_coeff_count == alpha_len
        }
        2 => alpha_len == 1 && session.linear_current_live_lane_count == live_lane_count,
        _ => false,
    };
    if e_first_len == 0
        || e_second_len == 0
        || !e_first_len.is_power_of_two()
        || !e_second_len.is_power_of_two()
        || equality_entries != domain_pair_count
        || alpha_len == 0
        || !alpha_len.is_power_of_two()
        || lane_count == 0
        || !lane_count.is_power_of_two()
        || relation_entries != current_len
        || current_live_len > current_len
        || active_pair_count > domain_pair_count
        || live_lane_count > lane_count
        || (fold_lane_weights && alpha_len != 1)
        || !linear_is_valid
        || !additional_parents_in_range
    {
        return Err(MetalCommitError::UnsupportedShape(
            "direct relation factors do not match the round geometry".into(),
        ));
    }
    let additional_workgroups = if additional_pair_count == 0 {
        1
    } else {
        direct_range_workgroups(additional_pair_count)
    };
    Ok(DirectRelationParams {
        live_len: u64::try_from(session.live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation live length"))?,
        current_len: u64::try_from(current_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation current length"))?,
        current_live_len: u64::try_from(current_live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation current live length"))?,
        input_live_len: u64::try_from(session.current_live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation input live length"))?,
        pair_count: u64::try_from(pair_count)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation pair count"))?,
        num_first: u64::try_from(e_first_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation first equality table"))?,
        num_second: u64::try_from(e_second_len).map_err(|_| {
            MetalCommitError::ShapeOverflow("direct relation second equality table")
        })?,
        workgroups: u64::try_from(direct_range_workgroups(pair_count))
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation workgroups"))?,
        current_coeff_count: u64::try_from(alpha_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation coefficient count"))?,
        live_lane_count: u64::try_from(live_lane_count)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation live lanes"))?,
        prefix_size: 1,
        materialize_prefix: 0,
        linear_mode: session.linear_mode as u64,
        additional_pair_count: u64::try_from(additional_pair_count).map_err(|_| {
            MetalCommitError::ShapeOverflow("direct relation additional pair count")
        })?,
        additional_workgroups: u64::try_from(additional_workgroups).map_err(|_| {
            MetalCommitError::ShapeOverflow("direct relation additional workgroups")
        })?,
        fold_lane_weights: u64::from(fold_lane_weights),
        resident_challenges: 0,
    })
}

fn direct_relation_additional_fold_schedule(
    initial_pairs: &[DirectRelationAdditionalPair],
    transitions: usize,
) -> Result<Vec<Vec<DirectRelationAdditionalFoldMapping>>, MetalCommitError> {
    let mut parents = initial_pairs
        .iter()
        .map(|pair| pair.parent)
        .collect::<Vec<_>>();
    if parents.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(MetalCommitError::UnsupportedShape(
            "direct relation additional parents are not strictly ordered".into(),
        ));
    }
    let mut schedule = Vec::with_capacity(transitions);
    for _ in 0..transitions {
        let mut mappings = Vec::with_capacity(parents.len());
        let mut cursor = 0usize;
        while cursor < parents.len() {
            let parent = parents[cursor] >> 1;
            let mut left = u32::MAX;
            let mut right = u32::MAX;
            while cursor < parents.len() && parents[cursor] >> 1 == parent {
                let index = u32::try_from(cursor).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation additional topology")
                })?;
                if parents[cursor] & 1 == 0 {
                    left = index;
                } else {
                    right = index;
                }
                cursor += 1;
            }
            mappings.push(DirectRelationAdditionalFoldMapping {
                parent,
                left,
                right,
            });
        }
        parents = mappings.iter().map(|mapping| mapping.parent).collect();
        schedule.push(mappings);
    }
    Ok(schedule)
}

fn encode_direct_range_reduction(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    partials: &Buffer,
    output: &Buffer,
    params: &DirectRangeParams,
) {
    encode_direct_range_reduction_at_offset(command, pipeline, partials, output, 0, params);
}

fn encode_direct_range_reduction_at_offset(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    partials: &Buffer,
    output: &Buffer,
    output_offset: u64,
    params: &DirectRangeParams,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_label("Akita fp128 direct range partial reduction");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(partials), 0);
    encoder.set_buffer(1, Some(output), output_offset);
    set_inline_bytes(encoder, 2, params);
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
    );
    encoder.end_encoding();
}

fn encode_direct_relation_reduction(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    partials: &Buffer,
    output: &Buffer,
    params: &DirectRelationParams,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(partials), 0);
    encoder.set_buffer(1, Some(output), 0);
    set_inline_bytes(encoder, 2, params);
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
    );
    encoder.end_encoding();
}

fn read_direct_range_coefficients(
    output: &Buffer,
) -> [Fp128Limbs; FP128_DIRECT_RANGE_STORED_COEFFICIENTS] {
    // SAFETY: `output` is shared storage for four initialized fp128 values.
    let values = unsafe {
        std::slice::from_raw_parts(
            output.contents().cast::<Fp128Limbs>(),
            FP128_DIRECT_RANGE_STORED_COEFFICIENTS,
        )
    };
    std::array::from_fn(|index| values[index])
}

fn read_direct_relation_coefficients(
    output: &Buffer,
) -> [Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS] {
    // SAFETY: `output` is shared storage for four initialized fp128 values.
    let values = unsafe {
        std::slice::from_raw_parts(
            output.contents().cast::<Fp128Limbs>(),
            FP128_DIRECT_RELATION_STORED_COEFFICIENTS,
        )
    };
    std::array::from_fn(|index| values[index])
}

fn read_direct_relation_prefix_evals(output: &Buffer) -> ([Fp128Limbs; 8], [Fp128Limbs; 8]) {
    // SAFETY: `output` contains sixteen initialized fp128 values.
    let values = unsafe {
        std::slice::from_raw_parts(
            output.contents().cast::<Fp128Limbs>(),
            FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS,
        )
    };
    (
        std::array::from_fn(|index| values[index]),
        std::array::from_fn(|index| values[8 + index]),
    )
}

fn set_inline_bytes<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<c_void>(),
    );
}

fn set_fp128_binding(
    encoder: &ComputeCommandEncoderRef,
    index: u64,
    binding: Fp128KernelBinding<'_>,
) {
    match binding {
        Fp128KernelBinding::Inline(value) => set_inline_bytes(encoder, index, &value),
        Fp128KernelBinding::Buffer(buffer, offset) => {
            encoder.set_buffer(index, Some(buffer), offset);
        }
    }
}

fn complete_command(
    command: &CommandBufferRef,
) -> Result<(Duration, Option<Duration>), MetalCommitError> {
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    let wall = start.elapsed();
    validate_completed_command(command)?;
    Ok((wall, completed_command_gpu_time(command)))
}

fn validate_completed_command(command: &CommandBufferRef) -> Result<(), MetalCommitError> {
    let status = command.status();
    if status != MTLCommandBufferStatus::Completed {
        return Err(MetalCommitError::CommandFailed(map_command_status(status)));
    }
    Ok(())
}

fn map_command_status(status: MTLCommandBufferStatus) -> CommandStatus {
    match status {
        MTLCommandBufferStatus::NotEnqueued => CommandStatus::NotEnqueued,
        MTLCommandBufferStatus::Enqueued => CommandStatus::Enqueued,
        MTLCommandBufferStatus::Committed => CommandStatus::Committed,
        MTLCommandBufferStatus::Scheduled => CommandStatus::Scheduled,
        MTLCommandBufferStatus::Completed => CommandStatus::Completed,
        MTLCommandBufferStatus::Error => CommandStatus::Error,
    }
}

fn command_buffer_timestamp(command: &CommandBufferRef, name: &'static str) -> Option<f64> {
    // SAFETY: `command` is a live `MTLCommandBuffer`, and both selected
    // properties have the Objective-C signature `NSTimeInterval -> f64`.
    unsafe { command.send_message::<(), f64>(Sel::register(name), ()) }.ok()
}

fn completed_command_gpu_time(command: &CommandBufferRef) -> Option<Duration> {
    let start = command_buffer_timestamp(command, "GPUStartTime")?;
    let end = command_buffer_timestamp(command, "GPUEndTime")?;
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return None;
    }
    Some(Duration::from_secs_f64(end - start))
}

fn completed_commands_gpu_span(
    first: &CommandBufferRef,
    last: &CommandBufferRef,
) -> Option<Duration> {
    let start = command_buffer_timestamp(first, "GPUStartTime")?;
    let end = command_buffer_timestamp(last, "GPUEndTime")?;
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return None;
    }
    Some(Duration::from_secs_f64(end - start))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_flat_negacyclic_shift_sequence_into(
        coefficients: &[F],
        alpha: F,
        evaluations: &mut [F],
    ) {
        assert_eq!(evaluations.len(), coefficients.len());
        let mut evaluation = F::zero();
        let mut power = F::one();
        for &coefficient in coefficients {
            evaluation += power * coefficient;
            power *= alpha;
        }
        let wrap_correction = power + F::one();
        for (output, &coefficient) in evaluations.iter_mut().zip(coefficients.iter().rev()) {
            *output = evaluation;
            evaluation = alpha * evaluation - wrap_correction * coefficient;
        }
    }

    #[test]
    fn packed_d128_decompose_fold_matches_model() {
        // At K = 256, D = 128 one trace row spans two ring positions.
        const POSITIONS: usize = 10;
        const BLOCKS_PER_COLUMN: usize = 129;
        const COLUMNS: usize = 3;
        const CHALLENGE_WEIGHT: usize = 4;
        let runtime = MetalRuntime::new().unwrap();
        let num_rows = POSITIONS * BLOCKS_PER_COLUMN / 2;
        let lanes = (0..num_rows * COLUMNS)
            .map(|index| match index % 7 {
                0 => 0,
                1 => 255,
                2 => 128,
                _ => 1 + ((17 * index + 29) % 254) as u8,
            })
            .collect::<Vec<_>>();
        for embedding_stride in [1, 2] {
            let mut challenge_positions = Vec::new();
            let mut challenge_coefficients = Vec::new();
            let mut dense_challenges = vec![0i8; COLUMNS * BLOCKS_PER_COLUMN * 64];
            for challenge in 0..COLUMNS * BLOCKS_PER_COLUMN {
                for term in 0..CHALLENGE_WEIGHT {
                    let position = ((97 * challenge + 131 * term) % (128 / embedding_stride))
                        * embedding_stride;
                    let coefficient = [1, -2, 3, -4][term];
                    challenge_positions.push(position as u16);
                    challenge_coefficients.push(coefficient);
                    if embedding_stride == 2 {
                        dense_challenges[challenge * 64 + position / 2] = coefficient;
                    }
                }
            }
            let params = PackedDecomposeFoldParams {
                num_rows: num_rows as u64,
                num_columns: COLUMNS as u64,
                lane_stride: COLUMNS as u64,
                num_positions: POSITIONS as u64,
                position_start: 0,
                blocks_per_column: BLOCKS_PER_COLUMN as u64,
                challenge_weight: CHALLENGE_WEIGHT as u64,
                output_coefficients: (POSITIONS * 128) as u64,
                zero_column_mask: 1,
            };
            let mut active_zero_rows = vec![0u64; num_rows.div_ceil(u64::BITS as usize)];
            active_zero_rows[0] = 0b1011;
            let mut streamed = Vec::new();
            let actual = runtime
                .dispatch_packed_fp128_decompose_fold_streaming(
                    128,
                    &lanes,
                    &active_zero_rows,
                    &challenge_positions,
                    &challenge_coefficients,
                    (embedding_stride == 2).then_some(dense_challenges.as_slice()),
                    params,
                    4,
                    |_, coefficients| streamed.extend_from_slice(coefficients),
                )
                .unwrap()
                .centered_coefficients;
            let mut expected = vec![0i32; POSITIONS * 128];
            for position in 0..POSITIONS {
                for trace_block in 0..BLOCKS_PER_COLUMN {
                    for column in 0..COLUMNS {
                        let ring = trace_block * POSITIONS + position;
                        let row = ring / 2;
                        let half = ring % 2;
                        let hot = usize::from(lanes[row * COLUMNS + column]);
                        let committed_zero = hot == 0
                            && column == 0
                            && active_zero_rows[row / u64::BITS as usize]
                                & (1u64 << (row % u64::BITS as usize))
                                != 0;
                        if (hot == 0 && !committed_zero) || hot / 128 != half {
                            continue;
                        }
                        let source_coefficient = hot % 128;
                        let challenge = column * BLOCKS_PER_COLUMN + trace_block;
                        let challenge_start = challenge * CHALLENGE_WEIGHT;
                        for term in 0..CHALLENGE_WEIGHT {
                            let mut destination = source_coefficient
                                + usize::from(challenge_positions[challenge_start + term]);
                            let mut value =
                                i32::from(challenge_coefficients[challenge_start + term]);
                            if destination >= 128 {
                                destination -= 128;
                                value = -value;
                            }
                            expected[position * 128 + destination] += value;
                        }
                    }
                }
            }
            assert_eq!(actual, expected);
            assert_eq!(streamed, expected);
        }
    }

    #[test]
    fn packed_d512_decompose_fold_matches_cpu() {
        const POSITIONS: usize = 9;
        const BLOCKS_PER_COLUMN: usize = 2;
        const COLUMNS: usize = 3;
        const CHALLENGE_WEIGHT: usize = 4;
        let runtime = MetalRuntime::new().unwrap();
        let num_rows = 2 * POSITIONS * BLOCKS_PER_COLUMN;
        let lanes = (0..num_rows * COLUMNS)
            .map(|index| match index % 11 {
                0 => 0,
                1 => 255,
                _ => 1 + ((17 * index + 29) % 254) as u8,
            })
            .collect::<Vec<_>>();
        let mut challenge_positions = Vec::new();
        let mut challenge_coefficients = Vec::new();
        for challenge in 0..COLUMNS * BLOCKS_PER_COLUMN {
            for term in 0..CHALLENGE_WEIGHT {
                challenge_positions.push(((97 * challenge + 131 * term) % 512) as u16);
                challenge_coefficients.push([1, -2, 3, -4][term]);
            }
        }
        let params = PackedDecomposeFoldParams {
            num_rows: num_rows as u64,
            num_columns: COLUMNS as u64,
            lane_stride: COLUMNS as u64,
            num_positions: POSITIONS as u64,
            position_start: 0,
            blocks_per_column: BLOCKS_PER_COLUMN as u64,
            challenge_weight: CHALLENGE_WEIGHT as u64,
            output_coefficients: (POSITIONS * 512) as u64,
            zero_column_mask: 1,
        };
        let active_zero_rows = [1u64];
        let mut streamed = Vec::new();
        let mut chunk_starts = Vec::new();
        let actual = runtime
            .dispatch_packed_fp128_d512_decompose_fold_streaming(
                &lanes,
                &active_zero_rows,
                &challenge_positions,
                &challenge_coefficients,
                None,
                params,
                3,
                |position_start, coefficients| {
                    chunk_starts.push(position_start);
                    streamed.extend_from_slice(coefficients);
                },
            )
            .unwrap()
            .centered_coefficients;
        let mut expected = vec![0i32; POSITIONS * 512];
        for position in 0..POSITIONS {
            for trace_block in 0..BLOCKS_PER_COLUMN {
                for column in 0..COLUMNS {
                    for row_in_ring in 0..2 {
                        let ring = trace_block * POSITIONS + position;
                        let row = 2 * ring + row_in_ring;
                        let hot = usize::from(lanes[row * COLUMNS + column]);
                        let committed_zero = hot == 0
                            && column == 0
                            && active_zero_rows[row / u64::BITS as usize]
                                & (1u64 << (row % u64::BITS as usize))
                                != 0;
                        if hot == 0 && !committed_zero {
                            continue;
                        }
                        let source_coefficient = row_in_ring * 256 + hot;
                        let challenge = column * BLOCKS_PER_COLUMN + trace_block;
                        let challenge_start = challenge * CHALLENGE_WEIGHT;
                        for term in 0..CHALLENGE_WEIGHT {
                            let mut destination = source_coefficient
                                + usize::from(challenge_positions[challenge_start + term]);
                            let mut value =
                                i32::from(challenge_coefficients[challenge_start + term]);
                            if destination >= 512 {
                                destination -= 512;
                                value = -value;
                            }
                            expected[position * 512 + destination] += value;
                        }
                    }
                }
            }
        }
        assert_eq!(actual, expected);
        assert_eq!(streamed, expected);
        assert_eq!(chunk_starts, [0, 3, 6]);
    }

    #[test]
    fn packed_d512_subring64_decompose_fold_routes_match_cpu() {
        const POSITIONS: usize = 9;
        const BLOCKS_PER_COLUMN: usize = 2;
        const COLUMNS: usize = 129;
        const CHALLENGE_WEIGHT: usize = 4;
        let runtime = MetalRuntime::new().unwrap();
        let num_rows = 2 * POSITIONS * BLOCKS_PER_COLUMN;
        let lanes = (0..num_rows * COLUMNS)
            .map(|index| 1 + (8 * ((17 * index + 29) % 32)) as u8)
            .collect::<Vec<_>>();
        let mut challenge_positions = Vec::new();
        let mut challenge_coefficients = Vec::new();
        let mut dense_challenges = vec![0i8; COLUMNS * BLOCKS_PER_COLUMN * 64];
        for challenge in 0..COLUMNS * BLOCKS_PER_COLUMN {
            for term in 0..CHALLENGE_WEIGHT {
                let subring_position = (13 * challenge + 17 * term) % 64;
                let coefficient = [1, -2, 2, -1][term];
                challenge_positions.push((subring_position * 8) as u16);
                challenge_coefficients.push(coefficient);
                dense_challenges[challenge * 64 + subring_position] = coefficient;
            }
        }
        let params = PackedDecomposeFoldParams {
            num_rows: num_rows as u64,
            num_columns: COLUMNS as u64,
            lane_stride: COLUMNS as u64,
            num_positions: POSITIONS as u64,
            position_start: 0,
            blocks_per_column: BLOCKS_PER_COLUMN as u64,
            challenge_weight: CHALLENGE_WEIGHT as u64,
            output_coefficients: (POSITIONS * 512) as u64,
            zero_column_mask: 0,
        };
        let actual = runtime
            .dispatch_packed_fp128_d512_decompose_fold_streaming(
                &lanes,
                &[],
                &challenge_positions,
                &challenge_coefficients,
                Some(&dense_challenges),
                params,
                POSITIONS,
                |_, _| {},
            )
            .unwrap()
            .centered_coefficients;
        let tasks_per_position = BLOCKS_PER_COLUMN * COLUMNS * 2;
        let tiles_per_position = tasks_per_position.div_ceil(256);
        let fold_params = PackedFoldIndexParams {
            num_rows: num_rows as u64,
            num_columns: COLUMNS as u64,
            lane_stride: COLUMNS as u64,
            num_positions: POSITIONS as u64,
            position_start: 0,
            blocks_per_column: BLOCKS_PER_COLUMN as u64,
            tasks_per_position: tasks_per_position as u64,
            tiles_per_position: tiles_per_position as u64,
            record_slots: (POSITIONS * tiles_per_position * 256) as u64,
            count_entries: (POSITIONS * tiles_per_position * 8) as u64,
            output_coefficients: (POSITIONS * 512) as u64,
            fold_digits: 0,
            fold_log_basis: 0,
        };
        let index = runtime
            .prepare_packed_fp128_d512_fold_index(&lanes, fold_params)
            .unwrap();
        let mut indexed_streamed = Vec::new();
        let mut indexed_digits = Vec::new();
        let indexed = runtime
            .dispatch_packed_fp128_d512_subring64_decompose_fold_streaming(
                &lanes,
                &dense_challenges,
                PackedFp128D512FoldSource::Retained(&index),
                params,
                4,
                4,
                4,
                |_, coefficients, digits| {
                    indexed_streamed.extend_from_slice(coefficients);
                    indexed_digits.extend_from_slice(digits);
                },
            )
            .unwrap()
            .centered_coefficients;
        let mut fused_streamed = Vec::new();
        let mut fused_digits = Vec::new();
        let mut fused_chunk_starts = Vec::new();
        let fused = runtime
            .dispatch_packed_fp128_d512_subring64_decompose_fold_streaming(
                &lanes,
                &dense_challenges,
                PackedFp128D512FoldSource::Fused(fold_params),
                params,
                4,
                4,
                4,
                |position_start, coefficients, digits| {
                    fused_chunk_starts.push(position_start);
                    fused_streamed.extend_from_slice(coefficients);
                    fused_digits.extend_from_slice(digits);
                },
            )
            .unwrap()
            .centered_coefficients;

        let mut expected = vec![0i32; POSITIONS * 512];
        for position in 0..POSITIONS {
            for trace_block in 0..BLOCKS_PER_COLUMN {
                for column in 0..COLUMNS {
                    for row_in_ring in 0..2 {
                        let ring = trace_block * POSITIONS + position;
                        let row = 2 * ring + row_in_ring;
                        let hot = usize::from(lanes[row * COLUMNS + column]);
                        if hot == 0 {
                            continue;
                        }
                        let source_coefficient = row_in_ring * 256 + hot;
                        let challenge = column * BLOCKS_PER_COLUMN + trace_block;
                        let challenge_start = challenge * CHALLENGE_WEIGHT;
                        for term in 0..CHALLENGE_WEIGHT {
                            let mut destination = source_coefficient
                                + usize::from(challenge_positions[challenge_start + term]);
                            let mut value =
                                i32::from(challenge_coefficients[challenge_start + term]);
                            if destination >= 512 {
                                destination -= 512;
                                value = -value;
                            }
                            expected[position * 512 + destination] += value;
                        }
                    }
                }
            }
        }
        assert_eq!(actual, expected);
        assert_eq!(indexed, expected);
        assert_eq!(indexed_streamed, expected);
        assert_eq!(fused, expected);
        assert_eq!(fused_streamed, expected);
        assert_eq!(fused_chunk_starts, [0, 4, 8]);
        for digits in [&indexed_digits, &fused_digits] {
            for position in 0..POSITIONS {
                for coefficient in 0..512 {
                    let reconstructed = (0..4).rev().fold(0i32, |value, digit| {
                        let value_index = position * 4 * 512 + digit * 512 + coefficient;
                        let value_digit = digits[value_index];
                        assert!((-8..8).contains(&value_digit));
                        value * 16 + i32::from(value_digit)
                    });
                    assert_eq!(reconstructed, expected[position * 512 + coefficient]);
                }
            }
        }
    }

    #[test]
    fn reduced_linear_sources_match_cpu_recurrence() {
        const D: usize = 64;
        let runtime = MetalRuntime::new().unwrap();
        let alpha = F::from_i64(7);
        let row_weights = [F::from_i64(3), F::from_i64(-5)];
        let matrix = (0..2 * D)
            .map(|index| F::from_i64((index as i64 % 19) - 9))
            .collect::<Vec<_>>();
        let matrix_limbs = matrix
            .iter()
            .copied()
            .map(Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let matrix_buffer = runtime.shared_buffer_from_slice(&matrix_limbs).unwrap();
        let mut power = F::one();
        let mut alpha_powers = Vec::with_capacity(D);
        for _ in 0..D {
            alpha_powers.push(Fp128Limbs::from_field(power));
            power *= alpha;
        }
        let alpha_limbs = Fp128Limbs::from_field(alpha);
        let wrap_correction = Fp128Limbs::from_field(power + F::one());
        let sources = vec![
            DirectRelationLinearSourceInput::ReducedSetup {
                matrix: matrix_buffer,
                ring_dimension: D,
                row_count: 2,
                column_count: 1,
                row_weights: row_weights
                    .into_iter()
                    .map(Fp128Limbs::from_field)
                    .collect(),
                alpha_powers: alpha_powers.clone(),
                alpha: alpha_limbs,
                wrap_correction,
            },
            DirectRelationLinearSourceInput::ReducedSparse {
                ring_dimension: D,
                challenge_count: 1,
                term_offsets: vec![0, 3],
                positions: vec![0, 11, 63],
                coefficients: vec![1, -2, 3],
                alpha_powers,
                alpha: alpha_limbs,
                wrap_correction,
            },
        ];
        let segments = [
            DirectRelationLinearSegment {
                factor: Fp128Limbs::from_field(F::one()),
                source_index: 0,
                target_lane_start: 0,
                target_lane_stride: 1,
                source_lane_start: 0,
                source_lane_stride: 1,
                lane_count: 1,
            },
            DirectRelationLinearSegment {
                factor: Fp128Limbs::from_field(F::one()),
                source_index: 1,
                target_lane_start: 0,
                target_lane_stride: 1,
                source_lane_start: 0,
                source_lane_stride: 1,
                lane_count: 1,
            },
        ];
        let (session, _) = runtime
            .begin_fp128_direct_relation(
                &[0i8; D],
                D,
                3,
                D.trailing_zeros() as usize,
                &[Fp128Limbs::from_field(F::zero())],
                &segments,
                &[0, 2],
                &[0, 1],
                &sources,
                &[],
            )
            .unwrap();
        let output = runtime
            .shared_buffer(2 * D * size_of::<Fp128Limbs>())
            .unwrap();
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_blit_command_encoder();
        encoder.copy_from_buffer(
            &session.linear_tables[0],
            0,
            &output,
            0,
            (2 * D * size_of::<Fp128Limbs>()) as u64,
        );
        encoder.end_encoding();
        complete_command(command).unwrap();
        let actual =
            unsafe { std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), 2 * D) }
                .iter()
                .enumerate()
                .map(|(index, value)| value.into_field(index).unwrap())
                .collect::<Vec<_>>();

        let combined = (0..D)
            .map(|coefficient| {
                row_weights[0] * matrix[coefficient] + row_weights[1] * matrix[D + coefficient]
            })
            .collect::<Vec<_>>();
        let mut expected_setup = vec![F::zero(); D];
        eval_flat_negacyclic_shift_sequence_into(&combined, alpha, &mut expected_setup);
        let mut sparse = vec![F::zero(); D];
        sparse[0] = F::from_i64(1);
        sparse[11] = F::from_i64(-2);
        sparse[63] = F::from_i64(3);
        let mut expected_sparse = vec![F::zero(); D];
        eval_flat_negacyclic_shift_sequence_into(&sparse, alpha, &mut expected_sparse);
        assert_eq!(actual[..D], expected_setup);
        assert_eq!(actual[D..], expected_sparse);
    }
}
