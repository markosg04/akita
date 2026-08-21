use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Range;
use std::time::{Duration, Instant};

use akita_algebra::{CrtCapacity, GarnerData, NttPrime};
use akita_prover::{StreamingPackedOneHotView, PACKED_ONEHOT_BUFFER_ALIGNMENT};
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
const FP128_D64_DIGIT_ROWS_PARTIALS_KERNEL_NAME: &str = "akita_fp128_d64_digit_rows_partials";
const FP128_D64_DIGIT_ROWS_REDUCE_KERNEL_NAME: &str = "akita_fp128_d64_digit_rows_reduce";
const FP128_I8_COEFFICIENT_PACKING_KERNEL_NAME: &str = "akita_fp128_i8_coefficient_packing";
const FP128_PACKED_ONEHOT_COEFFICIENT_PACKING_PARTIALS_KERNEL_NAME: &str =
    "akita_fp128_packed_onehot_coefficient_packing_partials";
const FP128_PACKED_ONEHOT_COEFFICIENT_PACKING_REDUCE_KERNEL_NAME: &str =
    "akita_fp128_packed_onehot_coefficient_packing_reduce";
const FP128_D512_DECOMPOSE_FOLD_KERNEL_NAME: &str = "akita_fp128_d512_decompose_fold";
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
const KERNEL_SOURCE: &str = include_str!("kernels/onehot.metal");
const FP128_D512_THREADS: usize = 1_024;
const FP128_D512_TASKS_PER_STREAM: usize = 32;
const FP128_D512_STREAMS_PER_COMMAND: usize = 1;
pub(crate) const FP128_D512_POSITION_PARTIALS: usize = 16;
const FP128_D512_TILE_FIELD_ELEMENTS: usize = 2_048;
const FP128_D512_THREADGROUP_BYTES: usize =
    FP128_D512_TILE_FIELD_ELEMENTS * size_of::<Fp128Limbs>();
const FP128_D512_COEFFICIENT_BANDS: usize = 2;
const FP128_D64_DIGIT_ROWS_THREADS: usize = 256;
const FP128_D64_DIGIT_ROWS_PARTIAL_THREADS: usize = 64;
pub(crate) const FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL: usize = 64;
const FP128_COEFFICIENT_PACKING_THREADS: usize = 256;
const FP128_PACKED_COEFFICIENT_PACKING_PARTIAL_THREADS: usize = 256;
const FP128_PACKED_COEFFICIENT_PACKING_REDUCE_THREADS: usize = 256;
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
}

const _: [(); 168] = [(); size_of::<PackedOneHotCommitParams>()];

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
}

const _: [(); 56] = [(); size_of::<DigitRowsParams>()];

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
}

const _: [(); 120] = [(); size_of::<PackedOneHotCoefficientPackingParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedDecomposeFoldParams {
    pub(crate) num_rows: u64,
    pub(crate) num_columns: u64,
    pub(crate) lane_stride: u64,
    pub(crate) num_positions: u64,
    pub(crate) blocks_per_column: u64,
    pub(crate) challenge_weight: u64,
    pub(crate) output_coefficients: u64,
}

const _: [(); 56] = [(); size_of::<PackedDecomposeFoldParams>()];

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
    pub(crate) pair_count: u64,
    pub(crate) num_first: u64,
    pub(crate) num_second: u64,
    pub(crate) workgroups: u64,
    pub(crate) basis: u64,
    pub(crate) prefix_size: u64,
    pub(crate) materialize_prefix: u64,
}

const _: [(); 72] = [(); size_of::<DirectRangeParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRelationParams {
    pub(crate) live_len: u64,
    pub(crate) current_len: u64,
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
}

const _: [(); 112] = [(); size_of::<DirectRelationParams>()];

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
    pub(crate) source_lane_start: u32,
    pub(crate) lane_count: u32,
}

const _: [(); 32] = [(); size_of::<DirectRelationLinearSegment>()];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectRelationAdditionalPair {
    pub(crate) parent: u64,
    pub(crate) reserved: u64,
    pub(crate) linear: [Fp128Limbs; 2],
    pub(crate) binary: [Fp128Limbs; 2],
}

const _: [(); 80] = [(); size_of::<DirectRelationAdditionalPair>()];

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

impl<const D: usize> PackedLaneSource for StreamingPackedOneHotView<F, D> {
    fn lane_count(&self) -> usize {
        StreamingPackedOneHotView::lane_count(self)
    }

    fn wait_lanes(
        &self,
        rows: Range<usize>,
        _lane_stride: usize,
    ) -> Result<&[u8], MetalCommitError> {
        StreamingPackedOneHotView::wait_lanes(self, rows)
            .map_err(|error| MetalCommitError::InputStream(error.to_string()))
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
}

pub(crate) struct CoefficientPackingDispatchOutcome {
    pub(crate) coefficients: Vec<Fp128Limbs>,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct PackedDecomposeFoldDispatchOutcome {
    pub(crate) centered_coefficients: Vec<i32>,
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

pub(crate) struct DirectRangeRoundOutcome {
    pub(crate) coefficients: [Fp128Limbs; FP128_DIRECT_RANGE_STORED_COEFFICIENTS],
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

pub(crate) struct DirectRangeAdvanceOutcome {
    pub(crate) next_coefficients: Option<[Fp128Limbs; FP128_DIRECT_RANGE_STORED_COEFFICIENTS]>,
    pub(crate) final_evaluation: Option<Fp128Limbs>,
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
    current_table: Option<usize>,
    compact_prefix_rounds: usize,
    rounds_folded: usize,
    allocation_bytes: usize,
}

pub(crate) struct DirectRelationRoundData<'a> {
    pub(crate) e_first: &'a [Fp128Limbs],
    pub(crate) e_second: &'a [Fp128Limbs],
    pub(crate) alpha: &'a [Fp128Limbs],
    pub(crate) linear_values: &'a [Fp128Limbs],
    pub(crate) source_offsets: &'a [u32],
    pub(crate) linear_mode: usize,
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
    linear_segments: Buffer,
    lane_offsets: Buffer,
    lane_segments: Buffer,
    lane_weight_tables: [Buffer; 2],
    two_round_prefix_partials: Buffer,
    two_round_prefix_output: Buffer,
    two_round_prefix_max_workgroups: usize,
    live_len: usize,
    current_len: usize,
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
    linear_values: Buffer,
    source_offsets: Buffer,
    allocation_bytes: usize,
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
    packed_partial_reduction_pipeline: ComputePipelineState,
    fp128_d64_digit_rows_partials_pipeline: ComputePipelineState,
    fp128_d64_digit_rows_reduce_pipeline: ComputePipelineState,
    fp128_i8_coefficient_packing_pipeline: ComputePipelineState,
    fp128_packed_onehot_coefficient_packing_partials_pipeline: ComputePipelineState,
    fp128_packed_onehot_coefficient_packing_reduce_pipeline: ComputePipelineState,
    fp128_d512_decompose_fold_pipeline: ComputePipelineState,
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
    fp128_direct_relation_initial_pipeline: ComputePipelineState,
    fp128_direct_relation_compact_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_field_fold_pipeline: ComputePipelineState,
    fp128_direct_relation_additional_compact_pipeline: ComputePipelineState,
    fp128_direct_relation_additional_field_pipeline: ComputePipelineState,
    fp128_direct_relation_two_round_prefix_pipeline: ComputePipelineState,
    fp128_direct_relation_two_round_prefix_reduce_pipeline: ComputePipelineState,
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

    pub(crate) fn supports_fp128_d64_digit_rows<const D: usize>(
        &self,
        num_vectors: usize,
        num_rows: usize,
        num_cols: usize,
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
        let partial_bytes = threadgroups
            .and_then(|count| count.checked_mul(D as u64))
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64));
        let output_bytes =
            output_coefficients.and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64));
        let maximum = self.device.max_buffer_length();
        output_coefficients.is_some_and(|count| count <= u64::from(u32::MAX))
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
                MetalOneHotKernel::PackedFp128D512Panels => {
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
        params: DigitRowsParams,
    ) -> Result<DigitRowsDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let expected_output = params
                .num_vectors
                .checked_mul(params.num_rows)
                .and_then(|count| count.checked_mul(params.ring_d))
                .ok_or(MetalCommitError::ShapeOverflow("digit-row output"))?;
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
                || params.columns_per_partial != FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64
                || params.column_partials != expected_column_partials
                || params.output_coefficients > u64::from(u32::MAX)
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
                )
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D64 digit rows exceed the kernel's index or device-buffer limits".into(),
                ));
            }

            let buffer_start = Instant::now();
            let digit_buffer = self.shared_buffer_from_digit_rows(digit_vectors)?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("digit-row output coefficients"))?;
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
            let buffer_setup = buffer_start.elapsed();

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
                MTLSize::new(params.output_coefficients, 1, 1),
                MTLSize::new(FP128_D64_DIGIT_ROWS_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: `output` is live shared storage for exactly `output_count`
            // aligned `Fp128Limbs` values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(DigitRowsDispatchOutcome {
                coefficients,
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
            set_inline_bytes(encoder, 3, &params);
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

    pub(crate) fn dispatch_packed_fp128_d512_decompose_fold(
        &self,
        lanes: &[u8],
        challenge_positions: &[u16],
        challenge_coefficients: &[i8],
        params: PackedDecomposeFoldParams,
    ) -> Result<PackedDecomposeFoldDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
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
                .checked_mul(512)
                .ok_or(MetalCommitError::ShapeOverflow("decompose-fold output"))?;
            if params.num_rows == 0
                || params.num_columns == 0
                || params.num_columns > params.lane_stride
                || params.challenge_weight == 0
                || params.output_coefficients != expected_output
                || u64::try_from(lanes.len()).ok() != Some(expected_lanes)
                || u64::try_from(challenge_positions.len()).ok() != Some(expected_challenge_terms)
                || challenge_positions.len() != challenge_coefficients.len()
                || params.num_positions > u64::from(u32::MAX)
                || self
                    .fp128_d512_decompose_fold_pipeline
                    .max_total_threads_per_threadgroup()
                    < 256
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 packed decompose-fold geometry is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_buffer = self.packed_lane_buffer(lanes)?;
            let positions = self.shared_buffer_from_slice(challenge_positions)?;
            let coefficients = self.shared_buffer_from_slice(challenge_coefficients)?;
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

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 D512 packed decompose-fold");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 D512 packed decompose-fold");
            encoder.set_compute_pipeline_state(&self.fp128_d512_decompose_fold_pipeline);
            encoder.set_buffer(0, Some(&lane_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&positions), 0);
            encoder.set_buffer(2, Some(&coefficients), 0);
            encoder.set_buffer(3, Some(&output), 0);
            set_inline_bytes(encoder, 4, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_positions, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            if !output_zero_copy {
                // SAFETY: both buffers contain exactly `output_count`
                // initialized i32 values and do not overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output.contents().cast::<i32>(),
                        centered_coefficients.as_mut_ptr(),
                        output_count,
                    );
                }
            }
            let readback_copy = readback_start.elapsed();
            let allocation_bytes = output_bytes
                .checked_add(size_of_val(challenge_positions))
                .and_then(|bytes| bytes.checked_add(size_of_val(challenge_coefficients)))
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
    pub(crate) fn dispatch_fp128_recursive_commit<const D: usize>(
        &self,
        matrix: &Buffer,
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
            let expected_matrix_bytes = expected_matrix_rings
                .checked_mul(D)
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit matrix bytes",
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
                || matrix.length() < expected_matrix_bytes as u64
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

            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita recursive commitment exact matvec");
            encoder.set_compute_pipeline_state(&self.fp128_recursive_commit_matvec_pipeline);
            encoder.set_buffer(0, Some(&digit_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&matrix_ntt), 0);
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
            let allocation_bytes = matrix_ntt_bytes
                .checked_add(residue_bytes)
                .and_then(|bytes| bytes.checked_add(output_bytes))
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
        let first_table_len = domain_len >> compact_prefix_rounds;
        let second_table_len = first_table_len / 2;
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
        let maximum_pairs = domain_len / 2;
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
            let pair_count = session.current_len / 2;
            let params = direct_range_params(
                session.live_len,
                session.current_len,
                pair_count,
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
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();
                let (command_wall, gpu) = complete_command(command)?;
                let readback_start = Instant::now();
                // SAFETY: `final_output` is a shared buffer containing one fp128 value.
                let final_evaluation =
                    unsafe { *session.final_output.contents().cast::<Fp128Limbs>() };
                let readback_copy = readback_start.elapsed();
                session.current_len = 1;
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
            let pair_count = next_len / 2;
            let mut params = direct_range_params(
                session.live_len,
                next_len,
                pair_count,
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
    ) -> Result<(DirectRelationSession, Duration), MetalCommitError> {
        let coefficient_count = 1usize
            .checked_shl(u32::try_from(coefficient_rounds).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation coefficient rounds")
            })?)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation coefficient count",
            ))?;
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
        {
            return Err(MetalCommitError::UnsupportedShape(
                "direct relation proof has malformed resident geometry".into(),
            ));
        }
        let setup_start = Instant::now();
        let compact_digits = self.shared_buffer_from_slice(digits)?;
        let first_table_len = domain_len >> compact_prefix_rounds;
        let second_table_len = first_table_len / 2;
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
        let maximum_workgroups = direct_range_workgroups(domain_len / 2);
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
        let live_lane_count = digits.len() / coefficient_count;
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
            source_lane_start: 0,
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
        let allocation_bytes = digits
            .len()
            .checked_add(first_table_bytes)
            .and_then(|bytes| bytes.checked_add(second_table_bytes))
            .and_then(|bytes| bytes.checked_add(partial_bytes))
            .and_then(|bytes| bytes.checked_add(2 * output_bytes))
            .and_then(|bytes| bytes.checked_add(size_of::<Fp128Limbs>()))
            .and_then(|bytes| bytes.checked_add(segment_bytes))
            .and_then(|bytes| bytes.checked_add(size_of_val(lane_offsets)))
            .and_then(|bytes| bytes.checked_add(size_of_val(lane_segments)))
            .and_then(|bytes| bytes.checked_add(size_of_val(lane_weights)))
            .and_then(|bytes| bytes.checked_add(second_lane_bytes))
            .and_then(|bytes| bytes.checked_add(two_round_prefix_partial_bytes))
            .and_then(|bytes| bytes.checked_add(two_round_prefix_output_bytes))
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
                linear_segments,
                lane_offsets: lane_offsets_buffer,
                lane_segments: lane_segments_buffer,
                lane_weight_tables,
                two_round_prefix_partials,
                two_round_prefix_output,
                two_round_prefix_max_workgroups,
                live_len: digits.len(),
                current_len: domain_len,
                current_table: None,
                current_lane_weight_table: 0,
                current_lane_count: lane_weights.len(),
                coefficient_rounds,
                compact_prefix_rounds,
                rounds_folded: 0,
                allocation_bytes,
            },
            setup_start.elapsed(),
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
                || !matches!(round.linear_mode, 0 | 1)
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
                linear_mode: round.linear_mode as u64,
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
            encoder.set_buffer(5, Some(&buffers.linear_values), 0);
            encoder.set_buffer(6, Some(&buffers.source_offsets), 0);
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
        session: &DirectRelationSession,
        current_len: usize,
        prefix_weights: &[Fp128Limbs],
        round: DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationAdditionalOutcome, MetalCommitError> {
        autoreleasepool(|| {
            if round.additional_pairs.is_empty() {
                return Ok(DirectRelationAdditionalOutcome {
                    coefficients: [Fp128Limbs::default();
                        FP128_DIRECT_RELATION_STORED_COEFFICIENTS],
                    timings: DispatchTimings::default(),
                    allocation_bytes: 0,
                });
            }
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
            let buffer_start = Instant::now();
            let prefix = self.shared_buffer_from_slice(prefix_weights)?;
            let pairs = self.shared_buffer_from_slice(round.additional_pairs)?;
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            self.encode_direct_relation_additional_compact(
                command,
                session,
                &prefix,
                &pairs,
                &round.scalars,
                &params,
            );
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = read_direct_relation_coefficients(&session.additional_output);
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
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct relation resume after two-round prefix");
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
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.fp128_direct_range_finalize_pipeline);
                encoder.set_buffer(0, Some(&session.tables[current_table]), 0);
                encoder.set_buffer(1, Some(&session.final_output), 0);
                set_inline_bytes(encoder, 2, &challenge);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();
                let (command_wall, gpu) = complete_command(command)?;
                let readback_start = Instant::now();
                // SAFETY: `final_output` contains one initialized fp128 value.
                let final_evaluation =
                    unsafe { *session.final_output.contents().cast::<Fp128Limbs>() };
                let readback_copy = readback_start.elapsed();
                session.current_len = 1;
                return Ok(DirectRelationAdvanceOutcome {
                    next_coefficients: None,
                    next_additional_coefficients: None,
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
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct relation fold and next round");
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
            session.rounds_folded += 1;
            if fold_lane_weights {
                session.current_lane_weight_table = next_lane_weight_table;
                session.current_lane_count = next_lane_count;
            }
            Ok(DirectRelationAdvanceOutcome {
                next_coefficients: Some(coefficients),
                next_additional_coefficients: Some(additional_coefficients),
                final_evaluation: None,
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

    fn direct_relation_round_buffers(
        &self,
        round: &DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationRoundBuffers, MetalCommitError> {
        let zero_field = Fp128Limbs::default();
        let zero_u32 = 0u32;
        let linear_values = if round.linear_values.is_empty() {
            std::slice::from_ref(&zero_field)
        } else {
            round.linear_values
        };
        let source_offsets = if round.source_offsets.is_empty() {
            std::slice::from_ref(&zero_u32)
        } else {
            round.source_offsets
        };
        let allocation_bytes = size_of_val(round.e_first)
            .checked_add(size_of_val(round.e_second))
            .and_then(|bytes| bytes.checked_add(size_of_val(round.alpha)))
            .and_then(|bytes| bytes.checked_add(size_of_val(linear_values)))
            .and_then(|bytes| bytes.checked_add(size_of_val(source_offsets)))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation round allocation bytes",
            ))?;
        Ok(DirectRelationRoundBuffers {
            e_first: self.shared_buffer_from_slice(round.e_first)?,
            e_second: self.shared_buffer_from_slice(round.e_second)?,
            alpha: self.shared_buffer_from_slice(round.alpha)?,
            linear_values: self.shared_buffer_from_slice(linear_values)?,
            source_offsets: self.shared_buffer_from_slice(source_offsets)?,
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
        encoder.set_buffer(start + 4, Some(&buffers.linear_values), 0);
        encoder.set_buffer(start + 5, Some(&buffers.source_offsets), 0);
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
        params: PackedOneHotCommitParams,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        self.dispatch_packed_onehot_source(matrix, &ResidentPackedLanes { lanes }, params)
    }

    pub(crate) fn dispatch_streaming_packed_onehot<const D: usize>(
        &self,
        matrix: &Buffer,
        source: &StreamingPackedOneHotView<F, D>,
        params: PackedOneHotCommitParams,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        self.dispatch_packed_onehot_source(matrix, source, params)
    }

    fn dispatch_packed_onehot_source<S: PackedLaneSource>(
        &self,
        matrix: &Buffer,
        source: &S,
        mut params: PackedOneHotCommitParams,
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
            if lane_count != source.lane_count() as u64
                || params.ring_d != 512
                || params.onehot_k != 256
                || params.column_capacity != 32
                || params.num_columns == 0
                || params.num_columns > params.column_capacity
                || params.lane_stride != params.num_columns
                || params.n_a != 1
                || params.num_digits_inner != 1
                || params.position_partials_per_block != FP128_D512_POSITION_PARTIALS as u64
                || !params.positions_per_partial.is_multiple_of(4)
                || !matches!(params.blocks_per_column, 32 | 64 | 128 | 256)
                || params.full_blocks_per_column > params.blocks_per_column
                || params.boundary_columns != 0
                || params.num_blocks != expected_tasks
                || params.task_offset != 0
                || params.dispatch_tasks != params.num_blocks
                || params.lane_row_offset != 0
                || params.output_coefficients != expected_output
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
            let mut buffer_setup = buffer_start.elapsed();

            let command_start = Instant::now();
            let command_count =
                base_streams.div_ceil(FP128_D512_STREAMS_PER_COMMAND as u64) as usize;
            let mut commands = Vec::with_capacity(command_count);
            let mut lane_buffers = Vec::with_capacity(command_count);
            let mut input_zero_copy = true;
            for first_stream in (0..base_streams).step_by(FP128_D512_STREAMS_PER_COMMAND) {
                let stream_count =
                    (base_streams - first_stream).min(FP128_D512_STREAMS_PER_COMMAND as u64);
                let task_offset = first_stream * FP128_D512_TASKS_PER_STREAM as u64;
                let mut dispatch_params = params;
                dispatch_params.task_offset = task_offset;
                dispatch_params.dispatch_tasks = (stream_count
                    * FP128_D512_TASKS_PER_STREAM as u64)
                    .min(params.num_blocks - task_offset);
                let final_task = task_offset + dispatch_params.dispatch_tasks - 1;
                let first_block = task_offset / params.num_columns;
                let final_block = final_task / params.num_columns;
                let rows_per_block = params.positions_per_block * 2;
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
                kernel: MetalOneHotKernel::PackedFp128D512Panels,
                blocks_per_threadgroup: FP128_D512_TASKS_PER_STREAM,
                columns_per_threadgroup: 1,
                matrix_block_streams: matrix_block_streams as usize,
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
            MetalOneHotKernel::PackedFp128D512Panels => Err(MetalCommitError::UnsupportedShape(
                "packed kernel requires packed parameters".into(),
            )),
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
            MetalOneHotKernel::PackedFp128D512Panels => {
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
            MetalOneHotKernel::PackedFp128D512Panels => unreachable!(),
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
    pair_count: usize,
    e_first: &[Fp128Limbs],
    e_second: &[Fp128Limbs],
    basis: usize,
) -> Result<DirectRangeParams, MetalCommitError> {
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
        || equality_entries != pair_count
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
    })
}

fn direct_relation_params(
    session: &DirectRelationSession,
    current_len: usize,
    lane_count: usize,
    fold_lane_weights: bool,
    round: &DirectRelationRoundData<'_>,
) -> Result<DirectRelationParams, MetalCommitError> {
    if current_len < 2 || !current_len.is_power_of_two() {
        return Err(MetalCommitError::UnsupportedShape(
            "direct relation round length is malformed".into(),
        ));
    }
    let pair_count = current_len / 2;
    let equality_entries = round
        .e_first
        .len()
        .checked_mul(round.e_second.len())
        .ok_or(MetalCommitError::ShapeOverflow(
            "direct relation equality entries",
        ))?;
    let relation_entries =
        round
            .alpha
            .len()
            .checked_mul(lane_count)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation rank-one entries",
            ))?;
    let linear_is_valid = match round.linear_mode {
        0 => true,
        1 => !round.source_offsets.is_empty(),
        2 => round.alpha.len() == 1 && round.linear_values.len() == round.live_lane_count,
        _ => false,
    };
    if round.e_first.is_empty()
        || round.e_second.is_empty()
        || !round.e_first.len().is_power_of_two()
        || !round.e_second.len().is_power_of_two()
        || equality_entries != pair_count
        || round.alpha.is_empty()
        || !round.alpha.len().is_power_of_two()
        || lane_count == 0
        || !lane_count.is_power_of_two()
        || relation_entries != current_len
        || round.live_lane_count > lane_count
        || (fold_lane_weights && round.alpha.len() != 1)
        || !linear_is_valid
        || round
            .additional_pairs
            .iter()
            .any(|pair| pair.parent >= pair_count as u64)
    {
        return Err(MetalCommitError::UnsupportedShape(
            "direct relation factors do not match the round geometry".into(),
        ));
    }
    let additional_workgroups = if round.additional_pairs.is_empty() {
        1
    } else {
        direct_range_workgroups(round.additional_pairs.len())
    };
    Ok(DirectRelationParams {
        live_len: u64::try_from(session.live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation live length"))?,
        current_len: u64::try_from(current_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation current length"))?,
        pair_count: u64::try_from(pair_count)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation pair count"))?,
        num_first: u64::try_from(round.e_first.len())
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation first equality table"))?,
        num_second: u64::try_from(round.e_second.len()).map_err(|_| {
            MetalCommitError::ShapeOverflow("direct relation second equality table")
        })?,
        workgroups: u64::try_from(direct_range_workgroups(pair_count))
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation workgroups"))?,
        current_coeff_count: u64::try_from(round.alpha.len())
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation coefficient count"))?,
        live_lane_count: u64::try_from(round.live_lane_count)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation live lanes"))?,
        prefix_size: 1,
        materialize_prefix: 0,
        linear_mode: round.linear_mode as u64,
        additional_pair_count: u64::try_from(round.additional_pairs.len()).map_err(|_| {
            MetalCommitError::ShapeOverflow("direct relation additional pair count")
        })?,
        additional_workgroups: u64::try_from(additional_workgroups).map_err(|_| {
            MetalCommitError::ShapeOverflow("direct relation additional workgroups")
        })?,
        fold_lane_weights: u64::from(fold_lane_weights),
    })
}

fn encode_direct_range_reduction(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    partials: &Buffer,
    output: &Buffer,
    params: &DirectRangeParams,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_label("Akita fp128 direct range partial reduction");
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
