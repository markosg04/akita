use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::time::{Duration, Instant};

use akita_prover::PACKED_ONEHOT_BUFFER_ALIGNMENT;
use metal::objc::rc::autoreleasepool;
use metal::objc::{runtime::Sel, Message};
use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use crate::error::metal_status::CommandStatus;
use crate::field::Fp128Limbs;
use crate::MetalCommitError;

const DIRECT_KERNEL_NAME: &str = "akita_onehot_commit_gather";
const BLOCK_BATCHED_KERNEL_NAME: &str = "akita_onehot_commit_block_batched";
const PACKED_FP128_D512_PANELS_KERNEL_NAME: &str = "akita_packed_onehot_commit_fp128_d512_panels";
const PACKED_PARTIAL_REDUCTION_KERNEL_NAME: &str = "akita_packed_onehot_reduce_partials";
const KERNEL_SOURCE: &str = include_str!("kernels/onehot.metal");
const FP128_D512_THREADS: usize = 1_024;
const FP128_D512_TASKS_PER_STREAM: usize = 32;
const FP128_D512_TILE_FIELD_ELEMENTS: usize = 2_048;
const FP128_D512_THREADGROUP_BYTES: usize =
    FP128_D512_TILE_FIELD_ELEMENTS * size_of::<Fp128Limbs>();
const FP128_D512_COEFFICIENT_BANDS: usize = 2;

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
    pub(crate) output_coefficients: u64,
    pub(crate) columns_per_threadgroup: u64,
    pub(crate) position_partials_per_block: u64,
    pub(crate) positions_per_partial: u64,
    pub(crate) log_ring_d: u64,
}

const _: [(); 144] = [(); size_of::<PackedOneHotCommitParams>()];

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

struct PackedLaneBuffer<'a> {
    buffer: Buffer,
    zero_copy: bool,
    marker: PhantomData<&'a [u8]>,
}

pub(crate) struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    direct_pipeline: ComputePipelineState,
    block_batched_pipeline: ComputePipelineState,
    packed_fp128_d512_pipeline: ComputePipelineState,
    packed_partial_reduction_pipeline: ComputePipelineState,
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
        Ok(Self {
            queue: device.new_command_queue(),
            direct_pipeline: pipeline(DIRECT_KERNEL_NAME)?,
            block_batched_pipeline: pipeline(BLOCK_BATCHED_KERNEL_NAME)?,
            packed_fp128_d512_pipeline: pipeline(PACKED_FP128_D512_PANELS_KERNEL_NAME)?,
            packed_partial_reduction_pipeline: pipeline(PACKED_PARTIAL_REDUCTION_KERNEL_NAME)?,
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

    pub(crate) fn dispatch_packed_onehot(
        &self,
        matrix: &Buffer,
        lanes: &[u8],
        mut params: PackedOneHotCommitParams,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let lane_count = params
                .num_rows
                .checked_mul(params.lane_stride)
                .ok_or(MetalCommitError::ShapeOverflow("packed lane count"))?;
            let expected_tasks = params
                .num_columns
                .checked_mul(params.blocks_per_column)
                .ok_or(MetalCommitError::ShapeOverflow("packed task count"))?;
            let expected_output = params
                .column_capacity
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(params.n_a))
                .and_then(|count| count.checked_mul(params.ring_d))
                .ok_or(MetalCommitError::ShapeOverflow("packed output"))?;
            if lane_count != lanes.len() as u64
                || params.ring_d != 512
                || params.onehot_k != 256
                || params.column_capacity != 32
                || !matches!(params.num_columns, 25 | 28 | 32)
                || params.lane_stride != params.num_columns
                || params.n_a != 1
                || params.num_digits_inner != 1
                || params.position_partials_per_block != 4
                || !params.positions_per_partial.is_multiple_of(4)
                || !matches!(params.blocks_per_column, 32 | 64 | 128 | 256)
                || params.full_blocks_per_column != params.blocks_per_column
                || params.boundary_columns != 0
                || params.num_blocks != expected_tasks
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
            if matrix.length() != expected_matrix_bytes {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D512 matrix length does not match the plan".into(),
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
            let lanes = self.packed_lane_buffer(lanes)?;
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
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita packed fp128 D512 root commitment");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita packed fp128 D512 coefficient bands");
            encoder.set_compute_pipeline_state(&self.packed_fp128_d512_pipeline);
            encoder.set_buffer(0, Some(matrix), 0);
            encoder.set_buffer(1, Some(&lanes.buffer), 0);
            encoder.set_buffer(2, Some(&partials), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(FP128_D512_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            let reduction = command.new_compute_command_encoder();
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
                kernel: MetalOneHotKernel::PackedFp128D512Panels,
                blocks_per_threadgroup: FP128_D512_TASKS_PER_STREAM,
                columns_per_threadgroup: 1,
                matrix_block_streams: matrix_block_streams as usize,
                scratch_bytes,
                input_zero_copy: lanes.zero_copy,
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
