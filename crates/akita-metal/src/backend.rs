use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::split_eq::GruenSplitEq;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{AkitaError, ExtField, MulBaseUnreduced};
use akita_prover::compute::{
    BalancedDigitRequest, CompressionComputeBackend, CompressionRowsProducts, ComputeBackendSetup,
    ComputeExecutionDomain, CyclicRowsComputeBackend, DecomposeFoldBatchPlan, DecomposeFoldChunk,
    DecomposeFoldChunkSink, DecomposeFoldPlan, DigitRowsComputeBackend, DigitRowsProducts,
    OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, TensorPackedWitness,
    TensorProjectionKernel,
};
use akita_prover::{
    CpuBackend, CpuPreparedSetup, DirectDigitRangeProofBackend, DirectDigitRangeProofInput,
    DirectLinearLayout, DirectLinearSource, DirectRelationRangePreparationInput,
    DirectRelationRangeProofBackend, DirectRelationRangeProofState, NttCacheOwnerId,
    NttOperationCluster, PackedOneHotView, RecursiveFoldBatchView, RecursiveFoldView,
    RelationRangeImageProver, SuffixWitnessBatchView, SuffixWitnessView,
};
use akita_sumcheck::{
    CompressedUniPoly, EqFactoredSumcheckProof, EqFactoredUniPoly, SumcheckProof,
};
use akita_transcript::{
    labels, sample_ext_challenge, Blake2b512CheckpointMode, Transcript,
    TranscriptExecutionCheckpoint,
};
use akita_types::{AkitaExpandedSetup, AkitaStage1Proof, AkitaStage1StageProof, NttCacheKey};
use metal::Device;

use crate::field::{MetalField, F};
use crate::packed_onehot_fp128_d512::{opening_index_plan, DeferredFoldIndex};
use crate::prepared::MetalPreparedSetup;
use crate::runtime::{
    DigitRowsParams, DirectRangeRoundOutcome, DirectRelationAdditionalPair,
    DirectRelationLinearSegment, DirectRelationLinearSourceInput, DirectRelationRoundData,
    DirectRelationRoundOutcome, DirectRelationScalars, MetalDeviceCapabilities, MetalOneHotKernel,
    MetalRuntime, PackedDecomposeFoldParams, PackedFp128D512FoldIndex, PackedFp128D512FoldSource,
    FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL,
};
use crate::{MetalCommitError, MetalExecutionPolicy, OpeningAccelerationPolicy};

type DirectRelationProofOutput = (SumcheckProof<F>, Vec<F>, RelationRangeImageProver<F>);

struct PreparedMetalDirectRelation {
    session: crate::runtime::DirectRelationSession,
    setup_timings: crate::runtime::DispatchTimings,
    linear_layout: DirectLinearLayout<F>,
    domain_len: usize,
    coefficient_rounds: usize,
}

/// Opaque challenge-independent state for one direct Stage-2 proof.
#[doc(hidden)]
pub struct MetalDirectRelationPreparation {
    prepared: Option<Box<PreparedMetalDirectRelation>>,
}

/// Timings and structural counters from the most recent Metal one-hot dispatch.
#[derive(Clone, Debug, Default)]
pub struct MetalCommitMetrics {
    /// Kernel variant selected for the dispatch.
    pub kernel: MetalOneHotKernel,
    /// Independent source blocks processed by one threadgroup.
    pub blocks_per_threadgroup: usize,
    /// Packed trace columns processed by one threadgroup.
    pub columns_per_threadgroup: usize,
    /// Packed trace blocks assigned to the CPU worker.
    pub cpu_blocks: usize,
    /// Packed trace blocks assigned to Metal.
    pub metal_blocks: usize,
    /// Metal blocks with every live column enabled.
    pub metal_full_blocks: usize,
    /// Live Metal columns in the optional boundary block.
    pub metal_boundary_columns: usize,
    /// Live packed columns assigned to the CPU worker.
    pub cpu_columns: usize,
    /// Live packed columns assigned to Metal.
    pub metal_columns: usize,
    /// Atomic scheduled units assigned to the CPU worker.
    ///
    /// Spatial splits count `(column, block)` pairs; rank splits count
    /// `(column, block, A row)` tuples.
    pub cpu_work_units: usize,
    /// Atomic scheduled units assigned to Metal, using the same convention as
    /// [`Self::cpu_work_units`].
    pub metal_work_units: usize,
    /// A-matrix rows assigned to the CPU worker.
    pub cpu_rank_rows: usize,
    /// A-matrix rows assigned to Metal.
    pub metal_rank_rows: usize,
    /// Number of committed sources.
    pub num_sources: usize,
    /// Number of nonzero one-hot chunks across all sources.
    pub hot_entries: usize,
    /// SIMD ballots used to compact dense packed-lane probes.
    pub lane_scan_ballots: u64,
    /// Nonzero packed lanes broadcast after SIMD compaction.
    pub selected_lane_broadcasts: u64,
    /// Number of field additions represented by the sparse operation.
    pub field_additions: u64,
    /// Extra canonical field additions in a device-side partial reduction.
    pub reduction_field_additions: u64,
    /// Conservative direct-gather A traffic.
    pub gathered_matrix_bytes: u64,
    /// Modeled global matrix traffic for the selected kernel.
    pub modeled_matrix_read_bytes: u64,
    /// Modeled global packed-lane traffic for the selected kernel.
    pub modeled_lane_read_bytes: u64,
    /// Packed hot-index bytes submitted for this call.
    pub index_bytes: usize,
    /// Whether the input buffer wraps the source allocation without copying.
    pub input_zero_copy: bool,
    /// Canonical output bytes returned by this call.
    pub output_bytes: usize,
    /// Device-only transient bytes allocated by this call.
    pub scratch_bytes: usize,
    /// Exact resident A-prefix bytes.
    pub matrix_bytes: usize,
    /// Whether the A-prefix device buffer was already resident.
    pub matrix_cache_hit: bool,
    /// A-prefix packing and allocation time on a miss.
    pub matrix_prepare_time: Duration,
    /// Hot-index packing time.
    pub index_pack_time: Duration,
    /// Commit-time construction of a retained packed-opening index.
    pub opening_index_time: Duration,
    /// GPU timestamp interval for retained opening-index construction.
    pub opening_index_gpu_time: Duration,
    /// Bytes retained by the packed-opening index.
    pub opening_index_bytes: usize,
    /// Per-call Metal input/output buffer setup.
    pub buffer_setup_time: Duration,
    /// Wall time from command commit through completion.
    pub command_wall_time: Duration,
    /// GPU timestamp interval for the command, when reported by the device.
    pub gpu_time: Option<Duration>,
    /// Wall time of the concurrent CPU range.
    pub cpu_time: Duration,
    /// Copy from shared output storage into owned Rust limbs.
    pub readback_copy_time: Duration,
    /// Canonical field reconstruction and witness assembly.
    pub output_reconstruction_time: Duration,
    /// Time spent merging the disjoint CPU and Metal rectangles.
    pub merge_time: Duration,
    /// Complete `commit_inner_group` call wall time.
    pub total_time: Duration,
    /// Successful delegated B-row calls after the most recent inner commit.
    pub digit_rows_calls: usize,
    /// Delegated B-row calls executed by Metal.
    pub digit_rows_metal_calls: usize,
    /// Largest row count seen by a delegated B-row call.
    pub digit_rows_max_rows: usize,
    /// Largest column count seen by a delegated B-row call.
    pub digit_rows_max_columns: usize,
    /// Largest number of vectors fused into one delegated B-row dispatch.
    pub digit_rows_max_batch: usize,
    /// Cumulative wall time in delegated B-row calls.
    pub digit_rows_time: Duration,
    /// Cumulative GPU timestamp interval for delegated B-row calls.
    pub digit_rows_gpu_time: Duration,
    /// Successful delegated compression calls after the most recent inner commit.
    pub compression_calls: usize,
    /// Cumulative wall time in delegated compression calls.
    pub compression_time: Duration,
}

/// Aggregate routing and dispatch metrics for one opening proof.
#[derive(Clone, Debug, Default)]
pub struct MetalOpeningMetrics {
    /// Wall time from Metal command commits through completion.
    pub command_wall_time: Duration,
    /// Sum of GPU timestamp intervals reported by Metal.
    pub gpu_active_time: Duration,
    /// Wall time spent building packed-opening indices after commitment.
    pub opening_index_time: Duration,
    /// GPU timestamp interval spent building deferred packed-opening indices.
    pub opening_index_gpu_time: Duration,
    /// Cumulative bytes allocated for deferred packed-opening indices.
    pub opening_index_bytes: usize,
    /// Wall time of packed root decompose/fold dispatches, including streamed consumers.
    pub packed_decompose_wall_time: Duration,
    /// GPU timestamp interval of packed root decompose/fold dispatches.
    pub packed_decompose_gpu_time: Duration,
    /// Host callback time while consuming completed packed root chunks.
    pub packed_decompose_consumer_time: Duration,
    /// Host preparation before entering the packed root runtime dispatch.
    pub packed_decompose_prepare_time: Duration,
    /// Host postprocessing after the packed root runtime dispatch returns.
    pub packed_decompose_postprocess_time: Duration,
    /// Complete packed root backend-call wall time.
    pub packed_decompose_total_time: Duration,
    /// Packed root calls that used the retained or locally fused residue schedule.
    pub packed_decompose_indexed_calls: usize,
    /// Device-produced balanced digit bytes consumed by the successor commit.
    pub packed_decompose_direct_digit_bytes: usize,
    /// Recursive-commit calls that reused a transformed setup matrix.
    pub recursive_commit_matrix_cache_hits: usize,
    /// Recursive-commit calls that transformed a setup matrix.
    pub recursive_commit_matrix_cache_misses: usize,
    /// Wall time spent transforming recursive-commit setup matrices.
    pub recursive_commit_matrix_ntt_time: Duration,
    /// GPU timestamp interval spent transforming recursive-commit setup matrices.
    pub recursive_commit_matrix_ntt_gpu_time: Duration,
    /// Bytes retained by transformed recursive-commit setup matrices.
    pub recursive_commit_matrix_ntt_bytes: usize,
    /// Wall time for resident direct-relation linear-source construction.
    pub linear_source_command_wall_time: Duration,
    /// GPU timestamp interval for resident direct-relation linear-source construction.
    pub linear_source_gpu_time: Duration,
    /// Wall time for direct Stage-1 range-proof commands.
    pub direct_range_command_wall_time: Duration,
    /// GPU timestamp interval for direct Stage-1 range-proof commands.
    pub direct_range_gpu_time: Duration,
    /// Host buffer-construction time for direct Stage-1 range proofs.
    pub direct_range_buffer_setup_time: Duration,
    /// Wall time for direct Stage-2 relation-proof commands.
    pub direct_relation_command_wall_time: Duration,
    /// GPU timestamp interval for direct Stage-2 relation-proof commands.
    pub direct_relation_gpu_time: Duration,
    /// Host buffer-construction time for direct Stage-2 relation proofs.
    pub direct_relation_buffer_setup_time: Duration,
    /// Host time spent constructing and populating transient dispatch buffers.
    pub upload_time: Duration,
    /// Host time spent copying shared output storage.
    pub readback_time: Duration,
    /// Cumulative transient bytes requested by Metal dispatches.
    pub allocation_bytes: usize,
    /// Opening operations delegated wholesale to the CPU backend.
    pub cpu_fallback_calls: usize,
    /// CPU operations selected by the backend's operation route.
    pub planned_cpu_calls: usize,
    /// Scalar work units assigned to planned CPU operations.
    pub planned_cpu_work_units: usize,
    /// Scalar work units represented by delegated digit-row calls.
    pub cpu_tail_work_units: usize,
}

/// Residency result for an explicit packed fp128 D512 matrix prewarm.
#[derive(Clone, Copy, Debug)]
pub struct MetalMatrixPrewarmMetrics {
    /// Exact resident matrix bytes.
    pub matrix_bytes: usize,
    /// Whether the requested prefix was already resident.
    pub cache_hit: bool,
    /// Packing and allocation time on a miss.
    pub prepare_time: Duration,
}

struct BackendInner {
    runtime: Option<Arc<MetalRuntime>>,
    policy: MetalExecutionPolicy,
    opening_acceleration_policy: OpeningAccelerationPolicy,
    cpu: CpuBackend,
    initialization_time: Duration,
    last_metrics: Mutex<Option<MetalCommitMetrics>>,
    last_opening_metrics: Mutex<Option<MetalOpeningMetrics>>,
}

/// Hybrid commitment backend with a Metal one-hot inner kernel and CPU fallbacks.
#[derive(Clone)]
pub struct MetalCommitBackend<Field = F> {
    inner: Arc<BackendInner>,
    marker: PhantomData<fn() -> Field>,
}

impl<Field> MetalCommitBackend<Field> {
    /// Whether a system Metal device is visible.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Build a backend from the system default device.
    pub fn new(policy: MetalExecutionPolicy) -> Result<Self, MetalCommitError> {
        Self::new_with_opening_acceleration_policy(policy, OpeningAccelerationPolicy::default())
    }

    /// Build a backend with a limit on preprocessing retained after commitment.
    pub fn new_with_opening_acceleration_policy(
        policy: MetalExecutionPolicy,
        opening_acceleration_policy: OpeningAccelerationPolicy,
    ) -> Result<Self, MetalCommitError> {
        Self::new_with_cpu_backend_and_opening_acceleration_policy(
            policy,
            CpuBackend::DEFAULT,
            opening_acceleration_policy,
        )
    }

    /// Build a backend with an explicit CPU fallback resource policy.
    pub fn new_with_cpu_backend(
        policy: MetalExecutionPolicy,
        cpu: CpuBackend,
    ) -> Result<Self, MetalCommitError> {
        Self::new_with_cpu_backend_and_opening_acceleration_policy(
            policy,
            cpu,
            OpeningAccelerationPolicy::default(),
        )
    }

    fn new_with_cpu_backend_and_opening_acceleration_policy(
        policy: MetalExecutionPolicy,
        cpu: CpuBackend,
        opening_acceleration_policy: OpeningAccelerationPolicy,
    ) -> Result<Self, MetalCommitError> {
        let start = Instant::now();
        let runtime = match MetalRuntime::new() {
            Ok(runtime) => Some(Arc::new(runtime)),
            Err(_error) if policy == MetalExecutionPolicy::PreferMetal => None,
            Err(error) => return Err(error),
        };
        Ok(Self {
            inner: Arc::new(BackendInner {
                runtime,
                policy,
                opening_acceleration_policy,
                cpu,
                initialization_time: start.elapsed(),
                last_metrics: Mutex::new(None),
                last_opening_metrics: Mutex::new(None),
            }),
            marker: PhantomData,
        })
    }

    /// Build a backend around an existing Metal device.
    pub fn with_device(
        device: Device,
        policy: MetalExecutionPolicy,
    ) -> Result<Self, MetalCommitError> {
        Self::with_device_and_cpu_backend(device, policy, CpuBackend::DEFAULT)
    }

    /// Build around an existing device with an explicit CPU fallback policy.
    pub fn with_device_and_cpu_backend(
        device: Device,
        policy: MetalExecutionPolicy,
        cpu: CpuBackend,
    ) -> Result<Self, MetalCommitError> {
        let start = Instant::now();
        let runtime = Arc::new(MetalRuntime::from_device(device)?);
        Ok(Self {
            inner: Arc::new(BackendInner {
                runtime: Some(runtime),
                policy,
                opening_acceleration_policy: OpeningAccelerationPolicy::default(),
                cpu,
                initialization_time: start.elapsed(),
                last_metrics: Mutex::new(None),
                last_opening_metrics: Mutex::new(None),
            }),
            marker: PhantomData,
        })
    }

    /// Runtime compilation plus pipeline creation time.
    pub fn initialization_time(&self) -> Duration {
        self.inner.initialization_time
    }

    /// Selected execution policy.
    pub fn policy(&self) -> MetalExecutionPolicy {
        self.inner.policy
    }

    /// Packed-opening preprocessing retained after commitment.
    pub fn opening_acceleration_policy(&self) -> OpeningAccelerationPolicy {
        self.inner.opening_acceleration_policy
    }

    /// CPU backend used for non-Metal operations and preferred-mode fallback.
    pub fn cpu_backend(&self) -> CpuBackend {
        self.inner.cpu
    }

    /// Device and pipeline limits, or `None` when preferred Metal is unavailable.
    pub fn capabilities(&self) -> Option<MetalDeviceCapabilities> {
        self.inner
            .runtime
            .as_deref()
            .map(MetalRuntime::capabilities)
    }

    /// Metrics for the most recent successful Metal dispatch.
    pub fn last_commit_metrics(&self) -> Result<Option<MetalCommitMetrics>, MetalCommitError> {
        Ok(self
            .inner
            .last_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .clone())
    }

    /// Reset metrics immediately before one opening proof begins.
    pub fn begin_opening_metrics(&self) -> Result<(), MetalCommitError> {
        *self
            .inner
            .last_opening_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)? = Some(MetalOpeningMetrics::default());
        Ok(())
    }

    /// Metrics accumulated since the latest opening-proof reset.
    pub fn last_opening_metrics(&self) -> Result<Option<MetalOpeningMetrics>, MetalCommitError> {
        Ok(self
            .inner
            .last_opening_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .clone())
    }

    /// Record one deliberate CPU delegation in an adapter-owned opening operation.
    pub fn record_opening_cpu_fallback(&self, work_units: usize) -> Result<(), MetalCommitError> {
        self.update_opening_metrics(|metrics| {
            metrics.cpu_fallback_calls += 1;
            metrics.cpu_tail_work_units = metrics.cpu_tail_work_units.saturating_add(work_units);
        })
    }

    /// Record an operation whose route selects the CPU backend.
    pub fn record_opening_planned_cpu(&self, work_units: usize) -> Result<(), MetalCommitError> {
        self.update_opening_metrics(|metrics| {
            metrics.planned_cpu_calls += 1;
            metrics.planned_cpu_work_units =
                metrics.planned_cpu_work_units.saturating_add(work_units);
        })
    }

    pub(crate) fn runtime(&self) -> Option<&MetalRuntime> {
        self.inner.runtime.as_deref()
    }

    pub(crate) fn record_metrics(
        &self,
        metrics: MetalCommitMetrics,
    ) -> Result<(), MetalCommitError> {
        *self
            .inner
            .last_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)? = Some(metrics);
        Ok(())
    }

    fn update_metrics(
        &self,
        update: impl FnOnce(&mut MetalCommitMetrics),
    ) -> Result<(), MetalCommitError> {
        if let Some(metrics) = self
            .inner
            .last_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .as_mut()
        {
            update(metrics);
        }
        Ok(())
    }

    pub(crate) fn update_opening_metrics(
        &self,
        update: impl FnOnce(&mut MetalOpeningMetrics),
    ) -> Result<(), MetalCommitError> {
        if let Some(metrics) = self
            .inner
            .last_opening_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .as_mut()
        {
            update(metrics);
        }
        Ok(())
    }

    pub(crate) fn record_opening_index_preparation(
        &self,
        elapsed: Duration,
        timings: crate::runtime::DispatchTimings,
        allocation_bytes: usize,
    ) -> Result<(), MetalCommitError> {
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.opening_index_time += elapsed;
            metrics.opening_index_gpu_time += timings.gpu.unwrap_or_default();
            metrics.opening_index_bytes =
                metrics.opening_index_bytes.saturating_add(allocation_bytes);
            metrics.upload_time += timings.buffer_setup;
            metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(allocation_bytes);
        })
    }
}

impl MetalCommitBackend<F> {
    /// Make the rank-one D512 A prefix resident before a packed trace commit.
    pub fn prewarm_packed_fp128_d512_matrix(
        &self,
        prepared: &MetalPreparedSetup<F>,
        active_a_cols: usize,
    ) -> Result<MetalMatrixPrewarmMetrics, AkitaError> {
        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let matrix = prepared.matrix(runtime, 512, 1, active_a_cols)?;
        Ok(MetalMatrixPrewarmMetrics {
            matrix_bytes: matrix.bytes,
            cache_hit: matrix.cache_hit,
            prepare_time: matrix.prepare_time,
        })
    }

    /// Make the largest root-or-successor commitment matrix prefix resident.
    pub fn prewarm_packed_fp128_commitment_matrix(
        &self,
        prepared: &MetalPreparedSetup<F>,
        root_active_a_cols: usize,
        outer_rows: usize,
        outer_active_a_cols: usize,
    ) -> Result<MetalMatrixPrewarmMetrics, AkitaError> {
        let root_fields = root_active_a_cols.checked_mul(512).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("packed root matrix fields").into_akita()
        })?;
        let outer_fields = outer_rows
            .checked_mul(outer_active_a_cols)
            .and_then(|count| count.checked_mul(64))
            .ok_or_else(|| {
                MetalCommitError::ShapeOverflow("packed outer matrix fields").into_akita()
            })?;
        let (ring_d, n_a, active_a_cols) = if outer_fields > root_fields {
            (64, outer_rows, outer_active_a_cols)
        } else {
            (512, 1, root_active_a_cols)
        };
        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let matrix = prepared.matrix(runtime, ring_d, n_a, active_a_cols)?;
        Ok(MetalMatrixPrewarmMetrics {
            matrix_bytes: matrix.bytes,
            cache_hit: matrix.cache_hit,
            prepare_time: matrix.prepare_time,
        })
    }

    /// Decompose and challenge-fold a packed K256 root at D512.
    pub fn decompose_fold_packed_fp128_d512<const D: usize>(
        &self,
        source: PackedOneHotView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<akita_prover::DecomposeFoldWitness<F>, AkitaError> {
        self.decompose_fold_packed_fp128_d512_impl(source, plan, None)
    }

    /// Decompose a packed D512 root and expose ordered completed position chunks.
    pub fn decompose_fold_packed_fp128_d512_streaming<const D: usize>(
        &self,
        source: PackedOneHotView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
        sink: &mut dyn DecomposeFoldChunkSink,
    ) -> Result<akita_prover::DecomposeFoldWitness<F>, AkitaError> {
        self.decompose_fold_packed_fp128_d512_impl(source, plan, Some(sink))
    }

    fn decompose_fold_packed_fp128_d512_impl<const D: usize>(
        &self,
        source: PackedOneHotView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
        mut sink: Option<&mut dyn DecomposeFoldChunkSink>,
    ) -> Result<akita_prover::DecomposeFoldWitness<F>, AkitaError> {
        let total_start = Instant::now();
        if D != 512
            || source.onehot_k() != 256
            || source.column_capacity() != 32
            || plan.num_positions_per_block == 0
            || plan.num_digits == 0
        {
            return Err(MetalCommitError::UnsupportedShape(
                "packed opening requires D512/K256/capacity32 and nonempty output".into(),
            )
            .into_akita());
        }
        let segment_rings = source.num_rows().checked_div(2).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("packed decompose segment rings").into_akita()
        })?;
        if !segment_rings.is_multiple_of(plan.num_positions_per_block) {
            return Err(MetalCommitError::UnsupportedShape(
                "packed decompose segment is not aligned to the scheduled positions".into(),
            )
            .into_akita());
        }
        let blocks_per_column = segment_rings / plan.num_positions_per_block;
        let live_challenges = source
            .num_columns()
            .checked_mul(blocks_per_column)
            .ok_or_else(|| {
                MetalCommitError::ShapeOverflow("packed decompose live challenges").into_akita()
            })?;
        if plan.challenges.len()
            != source
                .column_capacity()
                .checked_mul(blocks_per_column)
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("packed decompose challenges").into_akita()
                })?
        {
            return Err(AkitaError::InvalidSize {
                expected: source.column_capacity() * blocks_per_column,
                actual: plan.challenges.len(),
            });
        }
        let challenge_weight = plan
            .challenges
            .first()
            .map_or(0, |challenge| challenge.positions.len());
        if challenge_weight == 0
            || plan.challenges[..live_challenges].iter().any(|challenge| {
                challenge.positions.len() != challenge_weight
                    || challenge.coeffs.len() != challenge_weight
            })
        {
            return Err(MetalCommitError::UnsupportedShape(
                "packed decompose requires a fixed nonzero challenge weight".into(),
            )
            .into_akita());
        }

        let mut positions = Vec::with_capacity(live_challenges * challenge_weight);
        let mut coefficients = Vec::with_capacity(live_challenges * challenge_weight);
        let dense_len = live_challenges.checked_mul(64).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("packed decompose dense challenges").into_akita()
        })?;
        let mut dense_subring64 = vec![0i8; dense_len];
        let mut has_subring64_embedding = true;
        for (challenge_index, challenge) in plan.challenges[..live_challenges].iter().enumerate() {
            challenge.validate::<D>()?;
            for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                positions.push(u16::try_from(position).map_err(|_| {
                    MetalCommitError::UnsupportedShape(
                        "packed decompose challenge position does not fit u16".into(),
                    )
                    .into_akita()
                })?);
                coefficients.push(coefficient);
                let dense_position = usize::try_from(position)
                    .ok()
                    .filter(|position| position.is_multiple_of(8))
                    .map(|position| position / 8)
                    .filter(|&position| position < 64);
                if let Some(dense_position) = dense_position {
                    let slot = challenge_index * 64 + dense_position;
                    if dense_subring64[slot] == 0 {
                        dense_subring64[slot] = coefficient;
                    } else {
                        has_subring64_embedding = false;
                    }
                } else {
                    has_subring64_embedding = false;
                }
            }
        }
        let dense_subring64 = has_subring64_embedding.then_some(dense_subring64);
        let output_coefficients = plan.num_positions_per_block.checked_mul(D).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("packed decompose output coefficients").into_akita()
        })?;
        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let position_chunk_len = sink
            .as_deref()
            .map_or(plan.num_positions_per_block, |consumer| {
                consumer
                    .preferred_position_chunk_len(plan.num_positions_per_block)
                    .max(1)
            });
        let balanced_digit_request = sink
            .as_deref()
            .and_then(DecomposeFoldChunkSink::balanced_digit_request);
        let indexed_eligible = plan.num_digits == 1
            && dense_subring64.as_deref().is_some_and(|dense| {
                dense
                    .iter()
                    .all(|&coefficient| (-2..=2).contains(&coefficient))
            })
            && balanced_digit_request.is_some();
        let retained_index = indexed_eligible
            .then(|| source.take_opening_acceleration::<PackedFp128D512FoldIndex>())
            .flatten();
        let deferred_requested = indexed_eligible
            && source
                .take_opening_acceleration::<DeferredFoldIndex>()
                .is_some();
        let fused_params = if retained_index.is_none() && deferred_requested {
            let opening_plan = opening_index_plan(
                source.num_rows(),
                source.num_columns(),
                plan.num_positions_per_block,
                blocks_per_column,
            )?;
            Some(opening_plan.fold)
        } else {
            None
        };
        let fold_source = retained_index
            .as_deref()
            .map(PackedFp128D512FoldSource::Retained)
            .or_else(|| fused_params.map(PackedFp128D512FoldSource::Fused));
        let dispatch_params = PackedDecomposeFoldParams {
            num_rows: source.num_rows() as u64,
            num_columns: source.num_columns() as u64,
            lane_stride: source.num_columns() as u64,
            num_positions: plan.num_positions_per_block as u64,
            position_start: 0,
            blocks_per_column: blocks_per_column as u64,
            challenge_weight: challenge_weight as u64,
            output_coefficients: output_coefficients as u64,
        };
        let mut sink_error = None;
        let mut consume_chunk =
            |position_start: usize,
             compressed_chunk: &[i32],
             balanced_digits: Option<(BalancedDigitRequest, &[i8])>| {
                let Some(consumer) = sink.as_deref_mut() else {
                    return;
                };
                let expanded =
                    expand_packed_decompose_chunk::<D>(compressed_chunk, plan.num_digits);
                match expanded.and_then(|expanded| {
                    let position_count = compressed_chunk.len() / D;
                    let chunk = DecomposeFoldChunk::new(
                        position_start,
                        position_count,
                        D,
                        plan.num_digits,
                        &expanded,
                    )?;
                    let chunk = if let Some((request, digits)) = balanced_digits {
                        chunk.with_balanced_digits(request, digits)?
                    } else {
                        chunk
                    };
                    consumer.consume(chunk)
                }) {
                    Ok(()) => {}
                    Err(error) => sink_error = Some(error),
                }
            };
        let dispatch_start = Instant::now();
        let mut indexed_route = false;
        let outcome = match (
            dense_subring64.as_deref(),
            fold_source,
            balanced_digit_request,
        ) {
            (Some(dense), Some(fold_source), Some(request)) if indexed_eligible => {
                indexed_route = true;
                runtime
                    .dispatch_packed_fp128_d512_subring64_decompose_fold_streaming(
                        source.lanes(),
                        dense,
                        fold_source,
                        dispatch_params,
                        request.num_digits,
                        request.log_basis,
                        position_chunk_len,
                        |position_start, compressed_chunk, digits| {
                            consume_chunk(
                                position_start,
                                compressed_chunk,
                                Some((request, digits)),
                            );
                        },
                    )
                    .map_err(MetalCommitError::into_akita)?
            }
            _ => runtime
                .dispatch_packed_fp128_d512_decompose_fold_streaming(
                    source.lanes(),
                    &positions,
                    &coefficients,
                    dense_subring64.as_deref(),
                    dispatch_params,
                    position_chunk_len,
                    |position_start, compressed_chunk| {
                        consume_chunk(position_start, compressed_chunk, None);
                    },
                )
                .map_err(MetalCommitError::into_akita)?,
        };
        drop(retained_index);
        let prepare_time = dispatch_start.duration_since(total_start);
        if let Some(error) = sink_error {
            return Err(error);
        }
        let postprocess_start = Instant::now();
        let timings = outcome.timings;
        let compressed = into_array_vec::<_, D>(outcome.centered_coefficients)?;
        let centered = if plan.num_digits == 1 {
            compressed
        } else {
            let expanded_len = compressed
                .len()
                .checked_mul(plan.num_digits)
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("packed decompose digit expansion").into_akita()
                })?;
            let mut expanded = Vec::with_capacity(expanded_len);
            for coefficients in compressed {
                expanded.push(coefficients);
                expanded.extend((1..plan.num_digits).map(|_| [0i32; D]));
            }
            expanded
        };
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let witness =
            akita_prover::backend::poly_helpers::build_decompose_fold_witness(centered, modulus);
        let postprocess_time = postprocess_start.elapsed();
        let total_time = total_start.elapsed();
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.packed_decompose_wall_time += timings.command_wall;
            metrics.packed_decompose_gpu_time += timings.gpu.unwrap_or_default();
            metrics.packed_decompose_consumer_time += outcome.consumer_time;
            metrics.packed_decompose_prepare_time += prepare_time;
            metrics.packed_decompose_postprocess_time += postprocess_time;
            metrics.packed_decompose_total_time += total_time;
            metrics.packed_decompose_indexed_calls += usize::from(indexed_route);
            if indexed_route {
                metrics.packed_decompose_direct_digit_bytes =
                    metrics.packed_decompose_direct_digit_bytes.saturating_add(
                        output_coefficients
                            .saturating_mul(balanced_digit_request.map_or(0, |r| r.num_digits)),
                    );
            }
            metrics.upload_time += timings.buffer_setup;
            metrics.readback_time += timings.readback_copy;
            metrics.allocation_bytes = metrics
                .allocation_bytes
                .saturating_add(outcome.allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)?;
        Ok(witness)
    }
}

fn expand_packed_decompose_chunk<const D: usize>(
    compressed: &[i32],
    num_digits: usize,
) -> Result<Vec<i32>, AkitaError> {
    if D == 0 || num_digits == 0 || !compressed.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: compressed.len(),
        });
    }
    if num_digits == 1 {
        return Ok(compressed.to_vec());
    }
    let expanded_len = compressed
        .len()
        .checked_mul(num_digits)
        .ok_or_else(|| AkitaError::InvalidSetup("packed decompose chunk overflow".into()))?;
    let mut expanded = Vec::with_capacity(expanded_len);
    let zeros = vec![0i32; D];
    for coefficients in compressed.chunks_exact(D) {
        expanded.extend_from_slice(coefficients);
        for _ in 1..num_digits {
            expanded.extend_from_slice(&zeros);
        }
    }
    Ok(expanded)
}

fn into_array_vec<T, const D: usize>(values: Vec<T>) -> Result<Vec<[T; D]>, AkitaError> {
    if D == 0 || !values.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: values.len(),
        });
    }
    let len = values.len();
    let boxed = values.into_boxed_slice();
    let data = Box::into_raw(boxed) as *mut T;
    let arrays = std::ptr::slice_from_raw_parts_mut(data.cast::<[T; D]>(), len / D);
    // SAFETY: `[T; D]` has the same alignment as `T`; the checked lengths
    // describe the same allocation and every element remains initialized.
    Ok(unsafe { Box::from_raw(arrays) }.into_vec())
}

fn direct_range_eq_tables(
    split_eq: &GruenSplitEq<F>,
) -> (Vec<crate::field::Fp128Limbs>, Vec<crate::field::Fp128Limbs>) {
    let (first, second) = split_eq.remaining_eq_tables();
    (
        first
            .iter()
            .copied()
            .map(crate::field::Fp128Limbs::from_field)
            .collect(),
        second
            .iter()
            .copied()
            .map(crate::field::Fp128Limbs::from_field)
            .collect(),
    )
}

fn blake2b_squeeze_checkpoint<T>(transcript: &T) -> Option<([u8; 64], usize)>
where
    T: Transcript<F>,
{
    let TranscriptExecutionCheckpoint::Blake2b512(checkpoint) =
        transcript.execution_checkpoint()?;
    let Blake2b512CheckpointMode::Squeeze {
        next_block,
        leftovers,
    } = checkpoint.mode
    else {
        return None;
    };
    let squeezed_bytes = next_block.checked_mul(64)?.checked_sub(leftovers.len())?;
    Some((checkpoint.chaining_value, squeezed_bytes))
}

fn direct_range_round_poly(
    outcome: &DirectRangeRoundOutcome,
    basis: usize,
) -> Result<EqFactoredUniPoly<F>, AkitaError> {
    let stored_coefficients = match basis {
        4 => 2,
        8 => 4,
        _ => {
            return Err(MetalCommitError::UnsupportedShape(
                "direct range Metal proof supports basis four or eight".into(),
            )
            .into_akita())
        }
    };
    let coeffs_except_linear_term = outcome.coefficients[..stored_coefficients]
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            value
                .into_field(index)
                .map_err(MetalCommitError::into_akita)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(EqFactoredUniPoly {
        coeffs_except_linear_term,
    })
}

struct DirectRelationHostRound {
    e_first: Vec<crate::field::Fp128Limbs>,
    e_second: Vec<crate::field::Fp128Limbs>,
    alpha: Vec<crate::field::Fp128Limbs>,
    additional_pairs: Vec<DirectRelationAdditionalPair>,
    scalars: DirectRelationScalars,
    live_lane_count: usize,
}

impl DirectRelationHostRound {
    fn as_runtime(&self) -> DirectRelationRoundData<'_> {
        DirectRelationRoundData {
            e_first: &self.e_first,
            e_second: &self.e_second,
            alpha: &self.alpha,
            additional_pairs: &self.additional_pairs,
            scalars: self.scalars,
            live_lane_count: self.live_lane_count,
        }
    }
}

#[tracing::instrument(skip_all, name = "direct_relation_host_round")]
fn direct_relation_host_round(
    state: &DirectRelationRangeProofState<F>,
) -> Result<DirectRelationHostRound, AkitaError> {
    let to_limbs = |values: &[F]| {
        values
            .iter()
            .copied()
            .map(crate::field::Fp128Limbs::from_field)
            .collect::<Vec<_>>()
    };
    let (e_first, e_second) = state.remaining_eq_tables();
    let additional = state.additional_round();
    let additional_pairs = additional
        .pairs
        .into_iter()
        .map(|pair| {
            Ok(DirectRelationAdditionalPair {
                parent: u64::try_from(pair.parent).map_err(|_| {
                    AkitaError::InvalidSetup(
                        "direct relation sparse parent does not fit the Metal ABI".into(),
                    )
                })?,
                reserved: 0,
                linear: pair.linear.map(crate::field::Fp128Limbs::from_field),
                binary: pair.binary.map(crate::field::Fp128Limbs::from_field),
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let (l_at_0, l_at_1) = state.current_linear_factor_evals();
    Ok(DirectRelationHostRound {
        e_first: to_limbs(e_first),
        e_second: to_limbs(e_second),
        alpha: to_limbs(state.common_alpha_factor()),
        additional_pairs,
        scalars: DirectRelationScalars {
            l_at_0: crate::field::Fp128Limbs::from_field(l_at_0),
            l_at_1: crate::field::Fp128Limbs::from_field(l_at_1),
            binary_batching: crate::field::Fp128Limbs::from_field(additional.binary_batching),
        },
        live_lane_count: state.current_live_lane_count(),
    })
}

fn direct_relation_reduced_source_scalars(
    alpha: F,
    ring_dimension: usize,
) -> (
    Vec<crate::field::Fp128Limbs>,
    crate::field::Fp128Limbs,
    crate::field::Fp128Limbs,
) {
    let mut power = F::one();
    let mut powers = Vec::with_capacity(ring_dimension);
    for _ in 0..ring_dimension {
        powers.push(crate::field::Fp128Limbs::from_field(power));
        power *= alpha;
    }
    (
        powers,
        crate::field::Fp128Limbs::from_field(alpha),
        crate::field::Fp128Limbs::from_field(power + F::one()),
    )
}

fn direct_relation_round_poly(
    outcome: &DirectRelationRoundOutcome,
) -> Result<CompressedUniPoly<F>, AkitaError> {
    let mut coeffs_except_linear_term = (0..3)
        .map(|index| {
            let main = outcome.coefficients[index]
                .into_field(index)
                .map_err(MetalCommitError::into_akita)?;
            let additional = outcome.additional_coefficients[index]
                .into_field(index + 3)
                .map_err(MetalCommitError::into_akita)?;
            Ok(main + additional)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    while coeffs_except_linear_term.len() > 1
        && coeffs_except_linear_term
            .last()
            .is_some_and(akita_field::Zero::is_zero)
    {
        coeffs_except_linear_term.pop();
    }
    Ok(CompressedUniPoly {
        coeffs_except_linear_term,
    })
}

fn direct_relation_prefix_round_poly(
    main_coefficients: [F; 3],
    additional_coefficients: [crate::field::Fp128Limbs; 4],
) -> Result<CompressedUniPoly<F>, AkitaError> {
    let mut coeffs_except_linear_term = main_coefficients
        .into_iter()
        .zip(additional_coefficients)
        .enumerate()
        .map(|(index, (main, additional))| {
            additional
                .into_field(index)
                .map(|value| main + value)
                .map_err(MetalCommitError::into_akita)
        })
        .collect::<Result<Vec<_>, _>>()?;
    while coeffs_except_linear_term.len() > 1
        && coeffs_except_linear_term
            .last()
            .is_some_and(akita_field::Zero::is_zero)
    {
        coeffs_except_linear_term.pop();
    }
    Ok(CompressedUniPoly {
        coeffs_except_linear_term,
    })
}

fn direct_relation_prefix_fields<const N: usize>(
    limbs: [crate::field::Fp128Limbs; N],
) -> Result<[F; N], AkitaError> {
    limbs
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .into_field(index)
                .map_err(MetalCommitError::into_akita)
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| AkitaError::InvalidProof)
}

fn direct_relation_prefix_point(values: [F; 4], point: usize) -> F {
    let [value_00, value_10, value_01, value_11] = values;
    match point {
        0 => value_00,
        1 => value_01,
        2 => value_01 - value_00,
        3 => value_10,
        4 => value_11,
        5 => value_11 - value_10,
        6 => value_10 - value_00,
        7 => value_11 - value_01,
        _ => value_11 - value_10 - value_01 + value_00,
    }
}

struct DirectRelationTwoRoundPrefixHost {
    equality_first: Vec<crate::field::Fp128Limbs>,
    equality_second: Vec<crate::field::Fp128Limbs>,
    alpha_points: Vec<crate::field::Fp128Limbs>,
    norm_omitted_corner: usize,
}

#[tracing::instrument(skip_all, name = "direct_relation_two_round_prefix_host")]
fn direct_relation_two_round_prefix_host(
    state: &DirectRelationRangeProofState<F>,
) -> Result<Option<DirectRelationTwoRoundPrefixHost>, AkitaError> {
    let Some(data) = state.two_round_prefix_data()? else {
        return Ok(None);
    };
    if !matches!(data.basis, 4 | 8)
        || data.coefficient_count < 4
        || !data.coefficient_count.is_power_of_two()
        || data.alpha.len() != data.coefficient_count
        || data.lane_weights.is_empty()
        || data.live_lane_count > data.lane_weights.len()
    {
        return Err(AkitaError::InvalidProof);
    }
    let y_quads = data.coefficient_count / 4;
    let mut alpha_points = Vec::with_capacity(8 * y_quads);
    for point in 1..9 {
        for quad in data.alpha.chunks_exact(4) {
            let values: [F; 4] = quad.try_into().map_err(|_| AkitaError::InvalidProof)?;
            alpha_points.push(crate::field::Fp128Limbs::from_field(
                direct_relation_prefix_point(values, point),
            ));
        }
    }
    Ok(Some(DirectRelationTwoRoundPrefixHost {
        equality_first: data
            .equality_first
            .into_iter()
            .map(crate::field::Fp128Limbs::from_field)
            .collect(),
        equality_second: data
            .equality_second
            .into_iter()
            .map(crate::field::Fp128Limbs::from_field)
            .collect(),
        alpha_points,
        norm_omitted_corner: data.norm_omitted_corner,
    }))
}

impl MetalCommitBackend<F> {
    fn record_direct_dispatch(
        &self,
        timings: crate::runtime::DispatchTimings,
        allocation_bytes: usize,
        stage2: bool,
    ) -> Result<(), AkitaError> {
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.upload_time += timings.buffer_setup;
            metrics.readback_time += timings.readback_copy;
            metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(allocation_bytes);
            if stage2 {
                metrics.direct_relation_command_wall_time += timings.command_wall;
                metrics.direct_relation_gpu_time += timings.gpu.unwrap_or_default();
                metrics.direct_relation_buffer_setup_time += timings.buffer_setup;
            } else {
                metrics.direct_range_command_wall_time += timings.command_wall;
                metrics.direct_range_gpu_time += timings.gpu.unwrap_or_default();
                metrics.direct_range_buffer_setup_time += timings.buffer_setup;
            }
        })
        .map_err(MetalCommitError::into_akita)
    }

    fn record_direct_range_dispatch(
        &self,
        timings: crate::runtime::DispatchTimings,
        allocation_bytes: usize,
    ) -> Result<(), AkitaError> {
        self.record_direct_dispatch(timings, allocation_bytes, false)
    }

    fn record_direct_relation_dispatch(
        &self,
        timings: crate::runtime::DispatchTimings,
        allocation_bytes: usize,
    ) -> Result<(), AkitaError> {
        self.record_direct_dispatch(timings, allocation_bytes, true)
    }

    fn record_direct_relation_source_dispatch(
        &self,
        timings: crate::runtime::DispatchTimings,
    ) -> Result<(), AkitaError> {
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.upload_time += timings.buffer_setup;
            metrics.readback_time += timings.readback_copy;
            metrics.linear_source_command_wall_time += timings.command_wall;
            metrics.linear_source_gpu_time += timings.gpu.unwrap_or_default();
            metrics.direct_relation_command_wall_time += timings.command_wall;
            metrics.direct_relation_gpu_time += timings.gpu.unwrap_or_default();
            metrics.direct_relation_buffer_setup_time += timings.buffer_setup;
        })
        .map_err(MetalCommitError::into_akita)
    }

    fn record_direct_range_session(
        &self,
        setup_time: Duration,
        allocation_bytes: usize,
    ) -> Result<(), AkitaError> {
        self.update_opening_metrics(|metrics| {
            metrics.upload_time += setup_time;
            metrics.direct_range_buffer_setup_time += setup_time;
            metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)
    }

    fn direct_range_cpu_fallback<T>(
        &self,
        prepared: &MetalPreparedSetup<F>,
        input: DirectDigitRangeProofInput<F>,
        transcript: &mut T,
    ) -> Result<(AkitaStage1Proof<F>, Vec<F>), AkitaError>
    where
        T: Transcript<F>,
    {
        self.record_opening_cpu_fallback(input.digit_witness().len())
            .map_err(MetalCommitError::into_akita)?;
        <CpuBackend as DirectDigitRangeProofBackend<F, F>>::prove_direct_digit_range(
            &self.cpu_backend(),
            &prepared.cpu,
            input,
            transcript,
        )
    }

    fn direct_relation_cpu_fallback<T>(
        &self,
        prepared: &MetalPreparedSetup<F>,
        prover: RelationRangeImageProver<F>,
        transcript: &mut T,
    ) -> Result<DirectRelationProofOutput, AkitaError>
    where
        T: Transcript<F>,
    {
        self.record_opening_cpu_fallback(1)
            .map_err(MetalCommitError::into_akita)?;
        <CpuBackend as DirectRelationRangeProofBackend<F, F>>::prove_direct_relation_range(
            &self.cpu_backend(),
            &prepared.cpu,
            prover,
            (),
            transcript,
        )
    }
}

impl DirectDigitRangeProofBackend<F, F> for MetalCommitBackend<F> {
    fn prove_direct_digit_range<T>(
        &self,
        prepared: &Self::PreparedSetup,
        input: DirectDigitRangeProofInput<F>,
        transcript: &mut T,
    ) -> Result<(AkitaStage1Proof<F>, Vec<F>), AkitaError>
    where
        T: Transcript<F>,
    {
        let num_vars = input
            .column_variable_count()
            .checked_add(input.ring_variable_count())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("direct range variable width overflow".into())
            })?;
        let shift = u32::try_from(num_vars)
            .map_err(|_| AkitaError::InvalidSetup("direct range variable width overflow".into()))?;
        let domain_len = 1usize.checked_shl(shift).ok_or_else(|| {
            AkitaError::InvalidSetup("direct range domain length overflow".into())
        })?;
        let ring_shift = u32::try_from(input.ring_variable_count())
            .map_err(|_| AkitaError::InvalidSetup("direct range ring width overflow".into()))?;
        let ring_len = 1usize
            .checked_shl(ring_shift)
            .ok_or_else(|| AkitaError::InvalidSetup("direct range ring length overflow".into()))?;
        let expected_live_len = input
            .live_column_count()
            .checked_mul(ring_len)
            .ok_or_else(|| AkitaError::InvalidSetup("direct range live length overflow".into()))?;
        if input.equality_point().len() != num_vars
            || input.digit_witness().len() != expected_live_len
        {
            return Err(AkitaError::InvalidSize {
                expected: domain_len,
                actual: input.digit_witness().len(),
            });
        }
        let runtime = match self.runtime() {
            Some(runtime) => runtime,
            None if self.policy() == MetalExecutionPolicy::PreferMetal => {
                return self.direct_range_cpu_fallback(prepared, input, transcript)
            }
            None => return Err(MetalCommitError::DeviceUnavailable.into_akita()),
        };
        const COMPACT_PREFIX_ROUNDS: usize = 3;
        let (mut session, setup_time) = match runtime.begin_fp128_direct_range(
            input.digit_witness().as_ref(),
            domain_len,
            COMPACT_PREFIX_ROUNDS,
        ) {
            Ok(session) => session,
            Err(_error) if self.policy() == MetalExecutionPolicy::PreferMetal => {
                return self.direct_range_cpu_fallback(prepared, input, transcript)
            }
            Err(error) => return Err(error.into_akita()),
        };
        let mut split_eq = GruenSplitEq::new(input.equality_point())?;
        let (first_eq, second_eq) = direct_range_eq_tables(&split_eq);
        let initial = match runtime.dispatch_fp128_direct_range_initial(
            &session,
            &first_eq,
            &second_eq,
            input.plan().basis(),
        ) {
            Ok(outcome) => outcome,
            Err(_error) if self.policy() == MetalExecutionPolicy::PreferMetal => {
                return self.direct_range_cpu_fallback(prepared, input, transcript)
            }
            Err(error) => return Err(error.into_akita()),
        };
        self.record_direct_range_session(
            setup_time,
            runtime.direct_range_session_allocation_bytes(&session),
        )?;
        self.record_direct_range_dispatch(initial.timings, initial.allocation_bytes)?;

        let mut next_poly = direct_range_round_poly(&initial, input.plan().basis())?;
        let mut round_polys = Vec::with_capacity(num_vars);
        let mut challenges = Vec::with_capacity(num_vars);
        let mut final_evaluation = None;
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &F::zero());
        for round in 0..num_vars {
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &next_poly);
            let challenge =
                sample_ext_challenge::<F, F, T>(transcript, labels::CHALLENGE_SUMCHECK_ROUND);
            challenges.push(challenge);
            round_polys.push(next_poly.clone());
            split_eq.bind(challenge);
            let prefix_weights = if challenges.len() <= COMPACT_PREFIX_ROUNDS {
                EqPolynomial::evals(&challenges)?
                    .into_iter()
                    .map(crate::field::Fp128Limbs::from_field)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let next_eq_storage = if round + 1 < num_vars {
                let (first, second) = direct_range_eq_tables(&split_eq);
                Some((first, second))
            } else {
                None
            };
            let next_eq = next_eq_storage
                .as_ref()
                .map(|(first, second)| (first.as_slice(), second.as_slice()));
            let advance = runtime
                .dispatch_fp128_direct_range_advance(
                    &mut session,
                    crate::field::Fp128Limbs::from_field(challenge),
                    next_eq,
                    &prefix_weights,
                    input.plan().basis(),
                )
                .map_err(MetalCommitError::into_akita)?;
            self.record_direct_range_dispatch(advance.timings, advance.allocation_bytes)?;
            if let Some(coefficients) = advance.next_coefficients {
                next_poly = direct_range_round_poly(
                    &DirectRangeRoundOutcome {
                        coefficients,
                        timings: Default::default(),
                        allocation_bytes: 0,
                    },
                    input.plan().basis(),
                )?;
            }
            if let Some(evaluation) = advance.final_evaluation {
                final_evaluation = Some(
                    evaluation
                        .into_field(0)
                        .map_err(MetalCommitError::into_akita)?,
                );
            }
        }
        let range_image_evaluation = final_evaluation.ok_or(AkitaError::InvalidProof)?;
        Ok((
            AkitaStage1Proof {
                stages: vec![AkitaStage1StageProof {
                    sumcheck_proof: EqFactoredSumcheckProof { round_polys },
                    child_claims: Vec::new(),
                }],
                range_image_evaluation,
                norm_proof: None,
            },
            challenges,
        ))
    }
}

impl DirectRelationRangeProofBackend<F, F> for MetalCommitBackend<F> {
    type Preparation = MetalDirectRelationPreparation;

    fn prepare_direct_relation_range(
        &self,
        prepared: &Self::PreparedSetup,
        mut input: DirectRelationRangePreparationInput<'_, F>,
    ) -> Result<Self::Preparation, AkitaError> {
        const COMPACT_PREFIX_ROUNDS: usize = 3;
        if input.coefficient_rounds() < COMPACT_PREFIX_ROUNDS
            || input.domain_len() < 16
            || !input.domain_len().is_power_of_two()
        {
            if self.policy() == MetalExecutionPolicy::PreferMetal {
                return Ok(MetalDirectRelationPreparation { prepared: None });
            }
            return Err(MetalCommitError::UnsupportedShape(
                "direct relation Metal proof requires at least three coefficient rounds".into(),
            )
            .into_akita());
        }
        let runtime = match self.runtime() {
            Some(runtime) => runtime,
            None if self.policy() == MetalExecutionPolicy::PreferMetal => {
                return Ok(MetalDirectRelationPreparation { prepared: None });
            }
            None => return Err(MetalCommitError::DeviceUnavailable.into_akita()),
        };
        let host_prepare_span = tracing::info_span!("direct_relation_host_prepare").entered();
        let linear_layout = input.linear_layout();
        let linear_segments = linear_layout
            .segments
            .iter()
            .map(|segment| {
                Ok(DirectRelationLinearSegment {
                    factor: crate::field::Fp128Limbs::from_field(segment.factor),
                    source_index: u32::try_from(segment.source_index).map_err(|_| {
                        AkitaError::InvalidSetup(
                            "direct relation source index does not fit the Metal ABI".into(),
                        )
                    })?,
                    target_lane_start: u32::try_from(segment.target_lane_start).map_err(|_| {
                        AkitaError::InvalidSetup(
                            "direct relation target lane does not fit the Metal ABI".into(),
                        )
                    })?,
                    target_lane_stride: u32::try_from(segment.target_lane_stride).map_err(
                        |_| {
                            AkitaError::InvalidSetup(
                                "direct relation target stride does not fit the Metal ABI".into(),
                            )
                        },
                    )?,
                    source_lane_start: u32::try_from(segment.source_lane_start).map_err(|_| {
                        AkitaError::InvalidSetup(
                            "direct relation source lane does not fit the Metal ABI".into(),
                        )
                    })?,
                    source_lane_stride: u32::try_from(segment.source_lane_stride).map_err(
                        |_| {
                            AkitaError::InvalidSetup(
                                "direct relation source stride does not fit the Metal ABI".into(),
                            )
                        },
                    )?,
                    lane_count: u32::try_from(segment.lane_count).map_err(|_| {
                        AkitaError::InvalidSetup(
                            "direct relation lane count does not fit the Metal ABI".into(),
                        )
                    })?,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let lane_offsets = linear_layout
            .lane_offsets
            .iter()
            .copied()
            .map(|offset| {
                u32::try_from(offset).map_err(|_| {
                    AkitaError::InvalidSetup(
                        "direct relation lane offset does not fit the Metal ABI".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lane_segments = linear_layout
            .lane_segments
            .iter()
            .copied()
            .map(|segment| {
                u32::try_from(segment).map_err(|_| {
                    AkitaError::InvalidSetup(
                        "direct relation lane segment does not fit the Metal ABI".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lane_weights = input
            .relation_lane_weights()
            .iter()
            .copied()
            .map(crate::field::Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let linear_round = if self.policy() == MetalExecutionPolicy::RequireMetal {
            input.take_linear_round()
        } else {
            input.linear_round()
        };
        let linear_dense_values = linear_round
            .dense_values
            .unwrap_or_default()
            .into_iter()
            .map(crate::field::Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let linear_sources = linear_round
            .sources
            .into_iter()
            .map(
                |source| -> Result<DirectRelationLinearSourceInput, AkitaError> {
                    Ok(match source {
                        DirectLinearSource::Values(values) => {
                            DirectRelationLinearSourceInput::Values(
                                values
                                    .into_iter()
                                    .map(crate::field::Fp128Limbs::from_field)
                                    .collect(),
                            )
                        }
                        DirectLinearSource::ReducedSetup {
                            ring_dimension,
                            row_count,
                            column_count,
                            row_weights,
                            alpha,
                        } => {
                            let matrix = prepared.shared_matrix(
                                runtime,
                                ring_dimension,
                                row_count,
                                column_count,
                            )?;
                            let (alpha_powers, alpha, wrap_correction) =
                                direct_relation_reduced_source_scalars(alpha, ring_dimension);
                            DirectRelationLinearSourceInput::ReducedSetup {
                                matrix: matrix.buffer,
                                ring_dimension,
                                row_count,
                                column_count,
                                row_weights: row_weights
                                    .into_iter()
                                    .map(crate::field::Fp128Limbs::from_field)
                                    .collect(),
                                alpha_powers,
                                alpha,
                                wrap_correction,
                            }
                        }
                        DirectLinearSource::ReducedSparse(source) => {
                            let (alpha_powers, alpha, wrap_correction) =
                                direct_relation_reduced_source_scalars(
                                    source.alpha,
                                    source.ring_dimension,
                                );
                            DirectRelationLinearSourceInput::ReducedSparse {
                                ring_dimension: source.ring_dimension,
                                challenge_count: source.challenge_count,
                                term_offsets: source.term_offsets,
                                positions: source.positions,
                                coefficients: source.coefficients,
                                alpha_powers,
                                alpha,
                                wrap_correction,
                            }
                        }
                    })
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        drop(host_prepare_span);
        let session_setup_span = tracing::info_span!("direct_relation_session_setup").entered();
        let session = runtime.begin_fp128_direct_relation(
            input.compact_witness(),
            input.domain_len(),
            COMPACT_PREFIX_ROUNDS,
            input.coefficient_rounds(),
            &lane_weights,
            &linear_segments,
            &lane_offsets,
            &lane_segments,
            &linear_sources,
            &linear_dense_values,
        );
        drop(session_setup_span);
        let (session, setup_timings) = match session {
            Ok(session) => session,
            Err(_error) if self.policy() == MetalExecutionPolicy::PreferMetal => {
                return Ok(MetalDirectRelationPreparation { prepared: None });
            }
            Err(error) => return Err(error.into_akita()),
        };
        Ok(MetalDirectRelationPreparation {
            prepared: Some(Box::new(PreparedMetalDirectRelation {
                session,
                setup_timings,
                linear_layout,
                domain_len: input.domain_len(),
                coefficient_rounds: input.coefficient_rounds(),
            })),
        })
    }

    fn prove_direct_relation_range<T>(
        &self,
        prepared: &Self::PreparedSetup,
        prover: RelationRangeImageProver<F>,
        preparation: Self::Preparation,
        transcript: &mut T,
    ) -> Result<(SumcheckProof<F>, Vec<F>, RelationRangeImageProver<F>), AkitaError>
    where
        T: Transcript<F>,
    {
        const COMPACT_PREFIX_ROUNDS: usize = 3;
        let Some(preparation) = preparation.prepared else {
            return self.direct_relation_cpu_fallback(prepared, prover, transcript);
        };
        let PreparedMetalDirectRelation {
            mut session,
            setup_timings,
            linear_layout,
            domain_len: prepared_domain_len,
            coefficient_rounds,
        } = *preparation;
        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let mut state = {
            let _span = tracing::info_span!("direct_relation_state_prepare").entered();
            DirectRelationRangeProofState::with_linear_layout(prover, linear_layout)
        };
        let num_rounds = state.num_rounds();
        let domain_len = state
            .current_coefficient_count()
            .checked_mul(state.current_lane_capacity())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("direct relation domain length overflow".into())
            })?;
        if domain_len != prepared_domain_len || state.coefficient_bits() != coefficient_rounds {
            return Err(AkitaError::InvalidSetup(
                "prepared direct relation geometry changed before Stage 2".into(),
            ));
        }
        let host_prepare_span = tracing::info_span!("direct_relation_host_prepare").entered();
        let two_round_prefix = direct_relation_two_round_prefix_host(&state)?;
        drop(host_prepare_span);
        let initial_round = direct_relation_host_round(&state)?;
        let (mut next_poly, prefix_reconstruction, initial_timings, initial_allocation_bytes) =
            if let Some(prefix) = &two_round_prefix {
                let outcome = match runtime.dispatch_fp128_direct_relation_two_round_prefix(
                    &session,
                    &prefix.equality_first,
                    &prefix.equality_second,
                    &prefix.alpha_points,
                    prefix.norm_omitted_corner,
                    initial_round.as_runtime(),
                ) {
                    Ok(outcome) => outcome,
                    Err(_error) if self.policy() == MetalExecutionPolicy::PreferMetal => {
                        return self.direct_relation_cpu_fallback(
                            prepared,
                            state.into_prover(),
                            transcript,
                        );
                    }
                    Err(error) => return Err(error.into_akita()),
                };
                let reconstruction = state.reconstruct_two_round_prefix(
                    direct_relation_prefix_fields(outcome.norm_evals_except_corner)?,
                    direct_relation_prefix_fields(outcome.relation_evals_except_corner)?,
                )?;
                let poly = direct_relation_prefix_round_poly(
                    reconstruction.round_zero_coefficients_except_linear(),
                    outcome.additional_coefficients,
                )?;
                (
                    poly,
                    Some(reconstruction),
                    outcome.timings,
                    outcome.allocation_bytes,
                )
            } else {
                let outcome = match runtime
                    .dispatch_fp128_direct_relation_initial(&session, initial_round.as_runtime())
                {
                    Ok(outcome) => outcome,
                    Err(_error) if self.policy() == MetalExecutionPolicy::PreferMetal => {
                        return self.direct_relation_cpu_fallback(
                            prepared,
                            state.into_prover(),
                            transcript,
                        );
                    }
                    Err(error) => return Err(error.into_akita()),
                };
                (
                    direct_relation_round_poly(&outcome)?,
                    None,
                    outcome.timings,
                    outcome.allocation_bytes,
                )
            };
        self.record_direct_range_session(
            Duration::ZERO,
            runtime.direct_relation_session_allocation_bytes(&session),
        )?;
        self.record_direct_relation_source_dispatch(setup_timings)?;
        self.record_direct_relation_dispatch(initial_timings, initial_allocation_bytes)?;

        let mut claim = state.input_claim();
        let mut round_polys = Vec::with_capacity(num_rounds);
        let mut challenges = Vec::with_capacity(num_rounds);
        let mut final_evaluation = None;
        let mut final_linear_evaluation = None;
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
        for round in 0..num_rounds {
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &next_poly);
            let challenge =
                sample_ext_challenge::<F, F, T>(transcript, labels::CHALLENGE_SUMCHECK_ROUND);
            claim = next_poly.eval_from_hint(&claim, &challenge);
            challenges.push(challenge);
            round_polys.push(next_poly.clone());
            {
                let _span = tracing::info_span!("direct_relation_host_bind").entered();
                state.bind_without_linear_terms(challenge);
            }
            if round == 0 {
                if let Some(prefix_reconstruction) = &prefix_reconstruction {
                    let prefix_weights = EqPolynomial::evals(&challenges)?
                        .into_iter()
                        .map(crate::field::Fp128Limbs::from_field)
                        .collect::<Vec<_>>();
                    let next_round = direct_relation_host_round(&state)?;
                    let additional = runtime
                        .dispatch_fp128_direct_relation_additional_compact_only(
                            &mut session,
                            crate::field::Fp128Limbs::from_field(challenge),
                            domain_len / 2,
                            &prefix_weights,
                            next_round.as_runtime(),
                        )
                        .map_err(MetalCommitError::into_akita)?;
                    self.record_direct_relation_dispatch(
                        additional.timings,
                        additional.allocation_bytes,
                    )?;
                    next_poly = direct_relation_prefix_round_poly(
                        prefix_reconstruction.round_one_coefficients_except_linear(challenge),
                        additional.coefficients,
                    )?;
                    continue;
                }
            }
            if round == 1 && prefix_reconstruction.is_some() {
                let prefix_weights = EqPolynomial::evals(&challenges)?
                    .into_iter()
                    .map(crate::field::Fp128Limbs::from_field)
                    .collect::<Vec<_>>();
                let next_round = direct_relation_host_round(&state)?;
                let outcome = runtime
                    .dispatch_fp128_direct_relation_resume_after_two_round_prefix(
                        &mut session,
                        crate::field::Fp128Limbs::from_field(challenge),
                        &prefix_weights,
                        next_round.as_runtime(),
                    )
                    .map_err(MetalCommitError::into_akita)?;
                self.record_direct_relation_dispatch(outcome.timings, outcome.allocation_bytes)?;
                next_poly = direct_relation_round_poly(&outcome)?;
                continue;
            }
            let prefix_weights = if challenges.len() <= COMPACT_PREFIX_ROUNDS {
                EqPolynomial::evals(&challenges)?
                    .into_iter()
                    .map(crate::field::Fp128Limbs::from_field)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let next_round = if round + 1 < num_rounds {
                Some(direct_relation_host_round(&state)?)
            } else {
                None
            };
            let advance = runtime
                .dispatch_fp128_direct_relation_advance(
                    &mut session,
                    crate::field::Fp128Limbs::from_field(challenge),
                    &prefix_weights,
                    next_round.as_ref().map(DirectRelationHostRound::as_runtime),
                )
                .map_err(MetalCommitError::into_akita)?;
            self.record_direct_relation_dispatch(advance.timings, advance.allocation_bytes)?;
            if let (Some(coefficients), Some(additional_coefficients)) = (
                advance.next_coefficients,
                advance.next_additional_coefficients,
            ) {
                next_poly = direct_relation_round_poly(&DirectRelationRoundOutcome {
                    coefficients,
                    additional_coefficients,
                    timings: Default::default(),
                    allocation_bytes: 0,
                })?;
            }
            if let Some(evaluation) = advance.final_evaluation {
                final_evaluation = Some(
                    evaluation
                        .into_field(0)
                        .map_err(MetalCommitError::into_akita)?,
                );
            }
            if let Some(evaluation) = advance.final_linear_evaluation {
                final_linear_evaluation = Some(
                    evaluation
                        .into_field(0)
                        .map_err(MetalCommitError::into_akita)?,
                );
            }
        }
        let final_evaluation = final_evaluation.ok_or(AkitaError::InvalidProof)?;
        let final_linear_evaluation = final_linear_evaluation.ok_or(AkitaError::InvalidProof)?;
        let (prover, expected_claim) =
            state.finish_with_linear_evaluation(final_evaluation, final_linear_evaluation)?;
        if claim != expected_claim {
            return Err(AkitaError::InvalidInput(
                "Metal stage-2 final claim disagrees with its folded oracle".into(),
            ));
        }
        Ok((SumcheckProof { round_polys }, challenges, prover))
    }
}

impl<Field: MetalField> ComputeBackendSetup<Field> for MetalCommitBackend<Field>
where
    CpuBackend: ComputeBackendSetup<Field, PreparedSetup = CpuPreparedSetup<Field>>,
{
    type PreparedSetup = MetalPreparedSetup<Field>;

    fn execution_domain(&self) -> ComputeExecutionDomain {
        ComputeExecutionDomain::Accelerator
    }

    fn prepare_expanded(
        &self,
        expanded: Arc<AkitaExpandedSetup<Field>>,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        let cpu = self.cpu_backend().prepare_expanded(expanded.clone())?;
        Ok(MetalPreparedSetup::new(cpu, expanded))
    }

    fn ensure_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError> {
        self.cpu_backend().ensure_ntt_slot(&prepared.cpu, key)
    }

    fn ntt_requirement_is_cached(
        &self,
        prepared: &Self::PreparedSetup,
        requirement: akita_prover::compute::RoutedNttRequirement,
    ) -> Result<bool, AkitaError> {
        if requirement.cluster == NttOperationCluster::RingSwitch && requirement.key.ring_d == 512 {
            return Ok(false);
        }
        self.cpu_backend()
            .ntt_requirement_is_cached(&prepared.cpu, requirement)
    }

    fn ntt_cache_owner_id(&self, prepared: &Self::PreparedSetup) -> NttCacheOwnerId {
        self.cpu_backend().ntt_cache_owner_id(&prepared.cpu)
    }

    fn planned_ntt_cache_entry_bytes(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<usize, AkitaError> {
        self.cpu_backend()
            .planned_ntt_cache_entry_bytes(&prepared.cpu, key)
    }

    fn prepared_expanded_setup<'a>(
        &self,
        prepared: &'a Self::PreparedSetup,
    ) -> &'a AkitaExpandedSetup<Field> {
        prepared.expanded.as_ref()
    }

    fn release_built_ntt_slots(&self, prepared: &Self::PreparedSetup) -> Result<usize, AkitaError> {
        self.cpu_backend().release_built_ntt_slots(&prepared.cpu)
    }
}

impl<Field: MetalField> CompressionComputeBackend<Field> for MetalCommitBackend<Field>
where
    CpuBackend: ComputeBackendSetup<Field, PreparedSetup = CpuPreparedSetup<Field>>
        + CompressionComputeBackend<Field>,
{
    fn compression_cache_bytes(&self, prepared: &Self::PreparedSetup) -> Option<usize> {
        self.cpu_backend().compression_cache_bytes(&prepared.cpu)
    }

    fn compression_rows_products<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Vec<CompressionRowsProducts<Field, D>>, AkitaError> {
        let start = Instant::now();
        let products = self
            .cpu_backend()
            .compression_rows_products(&prepared.cpu, digit_vectors)?;
        let elapsed = start.elapsed();
        self.update_metrics(|metrics| {
            metrics.compression_calls += 1;
            metrics.compression_time += elapsed;
        })
        .map_err(MetalCommitError::into_akita)?;
        Ok(products)
    }
}

impl MetalCommitBackend<F> {
    fn digit_rows_products_batch_impl<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup<F>,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
        retain_quotients: bool,
    ) -> Result<Vec<DigitRowsProducts<F, D>>, AkitaError> {
        if digit_vectors.is_empty() {
            return Ok(Vec::new());
        }
        let start = Instant::now();
        let num_cols = digit_vectors[0].len();
        let use_metal = D == 64
            && row_len != 0
            && num_cols != 0
            && log_basis == 3
            && digit_vectors.iter().all(|digits| digits.len() == num_cols)
            && cfg_iter!(digit_vectors).all(|digits| {
                digits
                    .iter()
                    .flatten()
                    .all(|&digit| (-4..=3).contains(&digit))
            })
            && self.runtime().is_some_and(|runtime| {
                runtime.supports_fp128_d64_digit_rows::<D>(
                    digit_vectors.len(),
                    row_len,
                    num_cols,
                    retain_quotients,
                )
            });
        let (row_batches, used_metal, metal_timings, allocation_bytes) = if use_metal {
            if let Some(runtime) = self.runtime() {
                let matrix = {
                    let span = tracing::info_span!(
                        "MetalDigitRows::matrix_prepare",
                        num_vectors = digit_vectors.len(),
                        row_len,
                        num_cols,
                        ring_dimension = D,
                    );
                    let _entered = span.enter();
                    prepared.matrix(runtime, D, row_len, num_cols)?
                };
                let output_coefficients = digit_vectors
                    .len()
                    .checked_mul(row_len)
                    .and_then(|count| count.checked_mul(D))
                    .ok_or_else(|| {
                        MetalCommitError::ShapeOverflow("digit-row output coefficients")
                            .into_akita()
                    })?;
                let outcome = runtime
                    .dispatch_fp128_d64_digit_rows(
                        matrix.buffer.as_ref(),
                        digit_vectors,
                        retain_quotients,
                        DigitRowsParams {
                            num_vectors: u64::try_from(digit_vectors.len()).map_err(|_| {
                                MetalCommitError::ShapeOverflow("digit-row vector count")
                                    .into_akita()
                            })?,
                            num_rows: u64::try_from(row_len).map_err(|_| {
                                MetalCommitError::ShapeOverflow("digit-row row count").into_akita()
                            })?,
                            num_cols: u64::try_from(num_cols).map_err(|_| {
                                MetalCommitError::ShapeOverflow("digit-row column count")
                                    .into_akita()
                            })?,
                            ring_d: D as u64,
                            output_coefficients: u64::try_from(output_coefficients).map_err(
                                |_| {
                                    MetalCommitError::ShapeOverflow("digit-row output coefficients")
                                        .into_akita()
                                },
                            )?,
                            columns_per_partial: FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64,
                            column_partials: u64::try_from(
                                num_cols.div_ceil(FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL),
                            )
                            .map_err(|_| {
                                MetalCommitError::ShapeOverflow("digit-row column partials")
                                    .into_akita()
                            })?,
                            retain_quotients: u64::from(retain_quotients),
                        },
                    )
                    .map_err(MetalCommitError::into_akita)?;
                let timings = outcome.timings;
                let digit_bytes = digit_vectors
                    .len()
                    .checked_mul(num_cols)
                    .and_then(|count| count.checked_mul(D))
                    .ok_or_else(|| {
                        MetalCommitError::ShapeOverflow("digit-row input bytes").into_akita()
                    })?;
                let product_count = 1usize + usize::from(retain_quotients);
                let output_bytes = output_coefficients
                    .checked_mul(product_count)
                    .and_then(|count| {
                        count.checked_mul(std::mem::size_of::<crate::field::Fp128Limbs>())
                    })
                    .ok_or_else(|| {
                        MetalCommitError::ShapeOverflow("digit-row output bytes").into_akita()
                    })?;
                let partial_bytes = digit_vectors
                    .len()
                    .checked_mul(row_len)
                    .and_then(|count| {
                        count.checked_mul(
                            num_cols.div_ceil(FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL),
                        )
                    })
                    .and_then(|count| count.checked_mul(D))
                    .and_then(|count| count.checked_mul(product_count))
                    .and_then(|count| {
                        count.checked_mul(std::mem::size_of::<crate::field::Fp128Limbs>())
                    })
                    .ok_or_else(|| {
                        MetalCommitError::ShapeOverflow("digit-row partial bytes").into_akita()
                    })?;
                let allocation_bytes = digit_bytes
                    .checked_add(output_bytes)
                    .and_then(|count| count.checked_add(partial_bytes))
                    .ok_or_else(|| {
                        MetalCommitError::ShapeOverflow("digit-row allocation bytes").into_akita()
                    })?;
                let products = {
                    let span = tracing::info_span!(
                        "MetalDigitRows::reconstruct",
                        coefficients = outcome.coefficients.len(),
                        num_vectors = digit_vectors.len(),
                        row_len,
                        ring_dimension = D,
                        retain_quotients,
                    );
                    let _entered = span.enter();
                    let coefficients = outcome
                        .coefficients
                        .into_iter()
                        .enumerate()
                        .map(|(index, value)| F::from_device(value, index))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(MetalCommitError::into_akita)?;
                    let coefficients_per_vector = row_len.checked_mul(D).ok_or_else(|| {
                        MetalCommitError::ShapeOverflow("digit-row vector output").into_akita()
                    })?;
                    let negacyclic_coefficients = coefficients
                        .get(..output_coefficients)
                        .ok_or(AkitaError::InvalidProof)?;
                    let row_batches = negacyclic_coefficients
                        .chunks_exact(coefficients_per_vector)
                        .map(|vector| {
                            vector
                                .chunks_exact(D)
                                .map(CyclotomicRing::from_slice)
                                .collect()
                        })
                        .collect::<Vec<_>>();
                    let quotient_batches = if retain_quotients {
                        let quotient_coefficients = coefficients
                            .get(output_coefficients..)
                            .filter(|values| values.len() == output_coefficients)
                            .ok_or(AkitaError::InvalidProof)?;
                        Some(
                            quotient_coefficients
                                .chunks_exact(coefficients_per_vector)
                                .map(|vector| {
                                    vector
                                        .chunks_exact(D)
                                        .map(CyclotomicRing::from_slice)
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>(),
                        )
                    } else {
                        None
                    };
                    row_batches
                        .into_iter()
                        .enumerate()
                        .map(|(index, negacyclic)| DigitRowsProducts {
                            negacyclic,
                            quotients: quotient_batches
                                .as_ref()
                                .and_then(|batches| batches.get(index).cloned()),
                        })
                        .collect()
                };
                (products, true, Some(timings), allocation_bytes)
            } else {
                (
                    if retain_quotients {
                        self.cpu_backend().digit_rows_products_batch(
                            &prepared.cpu,
                            row_len,
                            digit_vectors,
                            log_basis,
                        )?
                    } else {
                        self.cpu_backend()
                            .digit_rows_batch(&prepared.cpu, row_len, digit_vectors, log_basis)?
                            .into_iter()
                            .map(|negacyclic| DigitRowsProducts {
                                negacyclic,
                                quotients: None,
                            })
                            .collect()
                    },
                    false,
                    None,
                    0,
                )
            }
        } else {
            (
                if retain_quotients {
                    self.cpu_backend().digit_rows_products_batch(
                        &prepared.cpu,
                        row_len,
                        digit_vectors,
                        log_basis,
                    )?
                } else {
                    self.cpu_backend()
                        .digit_rows_batch(&prepared.cpu, row_len, digit_vectors, log_basis)?
                        .into_iter()
                        .map(|negacyclic| DigitRowsProducts {
                            negacyclic,
                            quotients: None,
                        })
                        .collect()
                },
                false,
                None,
                0,
            )
        };
        let elapsed = start.elapsed();
        self.update_metrics(|metrics| {
            metrics.digit_rows_calls += digit_vectors.len();
            metrics.digit_rows_metal_calls += digit_vectors.len() * usize::from(used_metal);
            metrics.digit_rows_max_rows = metrics.digit_rows_max_rows.max(row_len);
            metrics.digit_rows_max_columns = metrics.digit_rows_max_columns.max(num_cols);
            metrics.digit_rows_max_batch = metrics.digit_rows_max_batch.max(digit_vectors.len());
            metrics.digit_rows_time += elapsed;
            metrics.digit_rows_gpu_time += metal_timings
                .and_then(|timings| timings.gpu)
                .unwrap_or_default();
        })
        .map_err(MetalCommitError::into_akita)?;
        let cpu_work_units = digit_vectors
            .len()
            .saturating_mul(row_len)
            .saturating_mul(num_cols)
            .saturating_mul(D);
        self.update_opening_metrics(|metrics| {
            if let Some(timings) = metal_timings {
                metrics.command_wall_time += timings.command_wall;
                metrics.gpu_active_time += timings.gpu.unwrap_or_default();
                metrics.upload_time += timings.buffer_setup;
                metrics.readback_time += timings.readback_copy;
                metrics.allocation_bytes =
                    metrics.allocation_bytes.saturating_add(allocation_bytes);
            } else {
                metrics.cpu_fallback_calls += digit_vectors.len();
                metrics.cpu_tail_work_units =
                    metrics.cpu_tail_work_units.saturating_add(cpu_work_units);
            }
        })
        .map_err(MetalCommitError::into_akita)?;
        Ok(row_batches)
    }
}

impl DigitRowsComputeBackend<F> for MetalCommitBackend<F>
where
    CpuBackend: ComputeBackendSetup<F, PreparedSetup = CpuPreparedSetup<F>>
        + DigitRowsComputeBackend<F>
        + CyclicRowsComputeBackend<F>,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let mut batches =
            self.digit_rows_products_batch_impl(prepared, row_len, &[digits], log_basis, false)?;
        if batches.len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "single digit-row dispatch returned an invalid batch count".into(),
            ));
        }
        Ok(batches.remove(0).negacyclic)
    }

    fn digit_rows_batch<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        self.digit_rows_products_batch_impl(prepared, row_len, digit_vectors, log_basis, false)
            .map(|products| {
                products
                    .into_iter()
                    .map(|product| product.negacyclic)
                    .collect()
            })
    }

    fn digit_rows_products_batch<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
    ) -> Result<Vec<DigitRowsProducts<F, D>>, AkitaError> {
        self.digit_rows_products_batch_impl(prepared, row_len, digit_vectors, log_basis, true)
    }
}

impl CyclicRowsComputeBackend<F> for MetalCommitBackend<F>
where
    CpuBackend:
        ComputeBackendSetup<F, PreparedSetup = CpuPreparedSetup<F>> + CyclicRowsComputeBackend<F>,
{
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.cpu_backend()
            .cyclic_digit_rows(&prepared.cpu, row_len, digits, log_basis)
    }
}

macro_rules! delegate_opening_fold_to_cpu {
    ($view:ident) => {
        impl<Field, const D: usize> OpeningFoldKernel<$view<'_, Field, D>, Field, D>
            for MetalCommitBackend<Field>
        where
            Field: MetalField,
            CpuBackend: ComputeBackendSetup<Field, PreparedSetup = CpuPreparedSetup<Field>>
                + for<'a> OpeningFoldKernel<$view<'a, Field, D>, Field, D>,
        {
            fn evaluate_and_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $view<'_, Field, D>,
                plan: OpeningFoldPlan<'_, Field>,
            ) -> Result<OpeningFoldOutput<Field, D>, AkitaError> {
                let output = self.cpu_backend().evaluate_and_fold(
                    prepared.map(|value| &value.cpu),
                    source,
                    plan,
                )?;
                self.update_opening_metrics(|metrics| metrics.cpu_fallback_calls += 1)
                    .map_err(MetalCommitError::into_akita)?;
                Ok(output)
            }

            fn decompose_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $view<'_, Field, D>,
                plan: DecomposeFoldPlan<'_>,
            ) -> Result<akita_prover::DecomposeFoldWitness<Field>, AkitaError> {
                let output = self.cpu_backend().decompose_fold(
                    prepared.map(|value| &value.cpu),
                    source,
                    plan,
                )?;
                self.update_opening_metrics(|metrics| metrics.cpu_fallback_calls += 1)
                    .map_err(MetalCommitError::into_akita)?;
                Ok(output)
            }
        }
    };
}

macro_rules! delegate_opening_batch_to_cpu {
    ($view:ident) => {
        impl<Field, const D: usize> OpeningBatchKernel<$view<'_, Field, D>, Field, D>
            for MetalCommitBackend<Field>
        where
            Field: MetalField,
            CpuBackend: ComputeBackendSetup<Field, PreparedSetup = CpuPreparedSetup<Field>>
                + for<'a> OpeningBatchKernel<$view<'a, Field, D>, Field, D>,
        {
            fn decompose_fold_batch(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $view<'_, Field, D>,
                plan: DecomposeFoldBatchPlan<'_>,
            ) -> Result<akita_prover::BatchDecomposeFoldOutcome<Field, D>, AkitaError> {
                let output = self.cpu_backend().decompose_fold_batch(
                    prepared.map(|value| &value.cpu),
                    source,
                    plan,
                )?;
                self.update_opening_metrics(|metrics| metrics.cpu_fallback_calls += 1)
                    .map_err(MetalCommitError::into_akita)?;
                Ok(output)
            }
        }
    };
}

delegate_opening_fold_to_cpu!(RecursiveFoldView);
delegate_opening_fold_to_cpu!(SuffixWitnessView);
delegate_opening_batch_to_cpu!(RecursiveFoldBatchView);
delegate_opening_batch_to_cpu!(SuffixWitnessBatchView);

impl<S, Field, E, const D: usize> TensorProjectionKernel<S, Field, E, D>
    for MetalCommitBackend<Field>
where
    Field: MetalField,
    E: ExtField<Field>,
    CpuBackend: ComputeBackendSetup<Field, PreparedSetup = CpuPreparedSetup<Field>>
        + TensorProjectionKernel<S, Field, E, D>,
{
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<Field>,
    {
        <CpuBackend as TensorProjectionKernel<S, Field, E, D>>::column_partials(
            &self.cpu_backend(),
            prepared.map(|value| &value.cpu),
            source,
            logical_point,
        )
    }

    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        <CpuBackend as TensorProjectionKernel<S, Field, E, D>>::packed_witness(
            &self.cpu_backend(),
            prepared.map(|value| &value.cpu),
            source,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use akita_prover::{
        AkitaProverSetup, ComputeBackendSetup, DigitRowsComputeBackend,
        DirectDigitRangeProofBackend, DirectDigitRangeProofInput, DirectRelationRangeProofBackend,
        RelationRangeImageProver,
    };
    use akita_transcript::AkitaTranscript;
    use akita_types::{DigitRangePlan, SetupMatrixCapacity};

    use super::*;
    use crate::field::Fp128Limbs;

    #[test]
    fn fp128_blake2b_sumcheck_challenge_matches_transcript() {
        let mut transcript = AkitaTranscript::<F>::new(b"metal/blake2b-checkpoint");
        transcript.append_bytes(b"prefix", b"prior-message");
        let _ = transcript.challenge_scalar(b"prior-challenge");
        let TranscriptExecutionCheckpoint::Blake2b512(checkpoint) =
            transcript.execution_checkpoint().unwrap();
        let prior_squeezed_bytes = match &checkpoint.mode {
            Blake2b512CheckpointMode::Squeeze {
                next_block,
                leftovers,
            } => next_block * 64 - leftovers.len(),
            _ => panic!("checkpoint should follow a scalar challenge"),
        };

        let claim = F::from_u64(0x1234_5678);
        let poly = EqFactoredUniPoly {
            coeffs_except_linear_term: vec![
                F::from_u64(3),
                F::from_u64(5),
                F::from_u64(7),
                F::from_u64(11),
            ],
        };
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
        let expected_challenge = transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
        let TranscriptExecutionCheckpoint::Blake2b512(expected_checkpoint) =
            transcript.execution_checkpoint().unwrap();

        let runtime = MetalRuntime::new().unwrap();
        let coefficients = poly
            .coeffs_except_linear_term
            .iter()
            .copied()
            .map(Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let actual = runtime
            .dispatch_fp128_blake2b_sumcheck_challenge(
                &checkpoint.chaining_value,
                prior_squeezed_bytes,
                Some(Fp128Limbs::from_field(claim)),
                &coefficients,
            )
            .unwrap();

        assert_eq!(actual.challenge.into_field(0).unwrap(), expected_challenge);
        assert_eq!(actual.chaining_value, expected_checkpoint.chaining_value);
    }

    #[test]
    fn fp128_direct_range_proof_matches_cpu() {
        const COLUMN_VARIABLES: usize = 3;
        const RING_VARIABLES: usize = 7;
        const LIVE_COLUMNS: usize = 5;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            COLUMN_VARIABLES + RING_VARIABLES,
            1,
            SetupMatrixCapacity {
                num_field_elements: 512,
            },
        )
        .unwrap();
        let digits = (0..LIVE_COLUMNS * (1usize << RING_VARIABLES))
            .map(|index| [-4, -3, -2, -1, 0, 1, 2, 3][(index * 13 + 5) & 7])
            .collect::<Vec<_>>();
        let equality_point = (0..COLUMN_VARIABLES + RING_VARIABLES)
            .map(|index| F::from_u64((index * 37 + 11) as u64))
            .collect::<Vec<_>>();
        let input = || {
            DirectDigitRangeProofInput::new(
                Arc::from(digits.clone()),
                equality_point.clone(),
                DigitRangePlan::new(8).unwrap(),
                LIVE_COLUMNS,
                COLUMN_VARIABLES,
                RING_VARIABLES,
            )
            .unwrap()
        };

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let mut cpu_transcript = AkitaTranscript::<F>::new(b"metal/direct-range-parity");
        let expected = cpu
            .prove_direct_digit_range(&cpu_prepared, input(), &mut cpu_transcript)
            .unwrap();

        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        metal.begin_opening_metrics().unwrap();
        let mut metal_transcript = AkitaTranscript::<F>::new(b"metal/direct-range-parity");
        let actual = metal
            .prove_direct_digit_range(&metal_prepared, input(), &mut metal_transcript)
            .unwrap();

        assert_eq!(actual, expected);
        let metrics = metal.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.cpu_fallback_calls, 0);
        assert!(metrics.command_wall_time > Duration::ZERO);
        assert!(metrics.gpu_active_time > Duration::ZERO);
        assert!(metrics.allocation_bytes > 0);
    }

    #[test]
    fn fp128_direct_relation_range_proof_matches_cpu() {
        const COEFFICIENT_BITS: usize = 3;
        const LANE_BITS: usize = 4;
        const LIVE_LANES: usize = 11;
        let num_vars = COEFFICIENT_BITS + LANE_BITS;
        let coeff_count = 1usize << COEFFICIENT_BITS;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            num_vars,
            1,
            SetupMatrixCapacity {
                num_field_elements: 512,
            },
        )
        .unwrap();
        let digits = (0..LIVE_LANES * coeff_count)
            .map(|index| [-2, -1, 0, 1][(index * 7 + 3) & 3])
            .collect::<Vec<_>>();
        let stage1_point = (0..num_vars)
            .map(|index| F::from_u64((index * 29 + 7) as u64))
            .collect::<Vec<_>>();
        let equality = EqPolynomial::evals(&stage1_point).unwrap();
        let range_image_evaluation =
            digits
                .iter()
                .zip(equality)
                .fold(F::zero(), |sum, (&digit, weight)| {
                    let value = F::from_i64(i64::from(digit));
                    sum + weight * value * (value + F::one())
                });
        let prover = || {
            RelationRangeImageProver::new_virtual_only(
                Arc::from(digits.clone()),
                &stage1_point,
                range_image_evaluation,
                4,
                LIVE_LANES,
                LANE_BITS,
                COEFFICIENT_BITS,
            )
            .unwrap()
        };

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let mut cpu_transcript = AkitaTranscript::<F>::new(b"metal/direct-relation-parity");
        let mut cpu_prover = prover();
        cpu.prepare_direct_relation_range(
            &cpu_prepared,
            DirectRelationRangePreparationInput::from_prover(&mut cpu_prover).unwrap(),
        )
        .unwrap();
        let (expected_proof, expected_point, expected_prover) = cpu
            .prove_direct_relation_range(&cpu_prepared, cpu_prover, (), &mut cpu_transcript)
            .unwrap();

        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        metal.begin_opening_metrics().unwrap();
        let mut metal_transcript = AkitaTranscript::<F>::new(b"metal/direct-relation-parity");
        let mut metal_prover = prover();
        let metal_preparation = metal
            .prepare_direct_relation_range(
                &metal_prepared,
                DirectRelationRangePreparationInput::from_prover(&mut metal_prover).unwrap(),
            )
            .unwrap();
        let (actual_proof, actual_point, actual_prover) = metal
            .prove_direct_relation_range(
                &metal_prepared,
                metal_prover,
                metal_preparation,
                &mut metal_transcript,
            )
            .unwrap();

        assert_eq!(actual_proof, expected_proof);
        assert_eq!(actual_point, expected_point);
        assert_eq!(actual_prover.final_w_eval(), expected_prover.final_w_eval());
        let metrics = metal.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.cpu_fallback_calls, 0);
        assert!(metrics.command_wall_time > Duration::ZERO);
        assert!(metrics.gpu_active_time > Duration::ZERO);
        assert!(metrics.allocation_bytes > 0);
    }

    #[test]
    fn fp128_direct_relation_nonzero_terms_match_cpu() {
        const COEFFICIENT_BITS: usize = 6;
        const LANE_BITS: usize = 8;
        const LIVE_LANES: usize = 173;
        let num_vars = COEFFICIENT_BITS + LANE_BITS;
        let coeff_count = 1usize << COEFFICIENT_BITS;
        let lane_capacity = 1usize << LANE_BITS;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            num_vars,
            1,
            SetupMatrixCapacity {
                num_field_elements: 512,
            },
        )
        .unwrap();
        let digits = (0..LIVE_LANES * coeff_count)
            .map(|index| [-2, -1, 0, 1][(index * 7 + 3) & 3])
            .collect::<Vec<_>>();
        let stage1_point = (0..num_vars)
            .map(|index| F::from_u64((index * 29 + 7) as u64))
            .collect::<Vec<_>>();
        let common_alpha_factor = (0..coeff_count)
            .map(|index| F::from_u64((index * 17 + 5) as u64))
            .collect::<Vec<_>>();
        let relation_lane_weights = (0..lane_capacity)
            .map(|index| F::from_u64((index * 31 + 11) as u64))
            .collect::<Vec<_>>();
        let prover = || {
            RelationRangeImageProver::new_backend_test_instance(
                digits.clone(),
                &stage1_point,
                4,
                common_alpha_factor.clone(),
                relation_lane_weights.clone(),
                LIVE_LANES,
                LANE_BITS,
                COEFFICIENT_BITS,
            )
            .unwrap()
        };

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let mut cpu_transcript = AkitaTranscript::<F>::new(b"metal/direct-relation-nonzero");
        let mut cpu_prover = prover();
        cpu.prepare_direct_relation_range(
            &cpu_prepared,
            DirectRelationRangePreparationInput::from_prover(&mut cpu_prover).unwrap(),
        )
        .unwrap();
        let (expected_proof, expected_point, expected_prover) = cpu
            .prove_direct_relation_range(&cpu_prepared, cpu_prover, (), &mut cpu_transcript)
            .unwrap();

        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        let mut metal_transcript = AkitaTranscript::<F>::new(b"metal/direct-relation-nonzero");
        let mut metal_prover = prover();
        let metal_preparation = metal
            .prepare_direct_relation_range(
                &metal_prepared,
                DirectRelationRangePreparationInput::from_prover(&mut metal_prover).unwrap(),
            )
            .unwrap();
        let (actual_proof, actual_point, actual_prover) = metal
            .prove_direct_relation_range(
                &metal_prepared,
                metal_prover,
                metal_preparation,
                &mut metal_transcript,
            )
            .unwrap();

        assert_eq!(actual_proof, expected_proof);
        assert_eq!(actual_point, expected_point);
        assert_eq!(actual_prover.final_w_eval(), expected_prover.final_w_eval());
    }

    #[test]
    fn fp128_d64_digit_rows_admit_the_t28_two_slice_shape() {
        let metal = MetalCommitBackend::<F>::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let runtime = metal.runtime().unwrap();

        assert!(runtime.supports_fp128_d64_digit_rows::<64>(2, 1, 1_409_024, true));
    }

    #[test]
    fn fp128_d64_digit_rows_partial_width_is_maximal_i32_safe() {
        let worst_limb_sum_per_column = 64u64 * 4 * u64::from(u16::MAX);
        let width = FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64;

        assert!(width * worst_limb_sum_per_column <= i32::MAX as u64);
        assert!((width + 1) * worst_limb_sum_per_column > i32::MAX as u64);
    }

    #[test]
    fn packed_commitment_matrix_prewarm_selects_the_larger_outer_prefix() {
        const ROOT_COLUMNS: usize = 64;
        const OUTER_ROWS: usize = 2;
        const OUTER_COLUMNS: usize = 1_024;
        const OUTER_FIELDS: usize = OUTER_ROWS * OUTER_COLUMNS * 64;

        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            20,
            1,
            SetupMatrixCapacity {
                num_field_elements: OUTER_FIELDS,
            },
        )
        .unwrap();
        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let prepared = metal.prepare_setup(&setup).unwrap();

        let first = metal
            .prewarm_packed_fp128_commitment_matrix(
                &prepared,
                ROOT_COLUMNS,
                OUTER_ROWS,
                OUTER_COLUMNS,
            )
            .unwrap();
        assert!(!first.cache_hit);
        assert_eq!(first.matrix_bytes, OUTER_FIELDS * size_of::<Fp128Limbs>());
        assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);

        let outer = prepared
            .matrix(metal.runtime().unwrap(), 64, OUTER_ROWS, OUTER_COLUMNS)
            .unwrap();
        assert!(outer.cache_hit);
        assert_eq!(outer.prepare_time, Duration::ZERO);
        assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
    }

    #[test]
    fn fp128_d64_digit_rows_match_cpu() {
        const D: usize = 64;
        const ROWS: usize = 1;
        const COLUMNS: usize = 44_032;
        const ROOT_COLUMNS: usize = 8_192;

        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            20,
            1,
            SetupMatrixCapacity {
                num_field_elements: ROOT_COLUMNS * 512,
            },
        )
        .unwrap();
        let digits = (0..COLUMNS)
            .map(|column| {
                std::array::from_fn(|coefficient| {
                    const VALUES: [i8; 8] = [-4, -3, -1, 0, 1, 2, 3, 0];
                    VALUES[(column * 13 + coefficient * 5) % VALUES.len()]
                })
            })
            .collect::<Vec<_>>();

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let expected = cpu
            .digit_rows::<D>(&cpu_prepared, ROWS, &digits, 3)
            .unwrap();

        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        let root = metal
            .prewarm_packed_fp128_d512_matrix(&metal_prepared, ROOT_COLUMNS)
            .unwrap();
        assert!(!root.cache_hit);
        let actual = metal
            .digit_rows::<D>(&metal_prepared, ROWS, &digits, 3)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(metal_prepared.matrix_cache_entries().unwrap(), 1);

        let cached = metal
            .digit_rows::<D>(&metal_prepared, ROWS, &digits, 3)
            .unwrap();
        assert_eq!(cached, expected);
        assert_eq!(metal_prepared.matrix_cache_entries().unwrap(), 1);

        let second_digits = (0..COLUMNS)
            .map(|column| {
                std::array::from_fn(|coefficient| {
                    const VALUES: [i8; 8] = [3, 0, -4, 2, -1, 1, -3, 0];
                    VALUES[(column * 7 + coefficient * 11) % VALUES.len()]
                })
            })
            .collect::<Vec<_>>();
        let expected_second = cpu
            .digit_rows::<D>(&cpu_prepared, ROWS, &second_digits, 3)
            .unwrap();
        let actual_batch = metal
            .digit_rows_batch::<D>(
                &metal_prepared,
                ROWS,
                &[digits.as_slice(), second_digits.as_slice()],
                3,
            )
            .unwrap();
        assert_eq!(actual_batch, vec![expected, expected_second]);
        assert_eq!(metal_prepared.matrix_cache_entries().unwrap(), 1);

        let expected_products = cpu
            .digit_rows_products_batch::<D>(
                &cpu_prepared,
                ROWS,
                &[digits.as_slice(), second_digits.as_slice()],
                3,
            )
            .unwrap();
        let actual_products = metal
            .digit_rows_products_batch::<D>(
                &metal_prepared,
                ROWS,
                &[digits.as_slice(), second_digits.as_slice()],
                3,
            )
            .unwrap();
        for (actual, expected) in actual_products.iter().zip(&expected_products) {
            assert_eq!(actual.negacyclic, expected.negacyclic);
            assert_eq!(actual.quotients, expected.quotients);
        }
    }
}
