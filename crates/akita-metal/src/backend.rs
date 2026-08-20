use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, ExtField, MulBaseUnreduced};
use akita_prover::compute::{
    CompressionComputeBackend, CompressionRowsProducts, ComputeBackendSetup,
    CyclicRowsComputeBackend, DigitRowsComputeBackend, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan, TensorPackedWitness,
    TensorProjectionKernel,
};
use akita_prover::{CpuBackend, CpuPreparedSetup, NttCacheOwnerId};
use akita_types::{AkitaExpandedSetup, NttCacheKey};
use metal::Device;

use crate::field::{MetalField, F};
use crate::prepared::MetalPreparedSetup;
use crate::runtime::{
    DigitRowsParams, MetalDeviceCapabilities, MetalOneHotKernel, MetalRuntime,
    FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL,
};
use crate::{MetalCommitError, MetalExecutionPolicy};

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
    /// Cumulative wall time in delegated B-row calls.
    pub digit_rows_time: Duration,
    /// Cumulative GPU timestamp interval for delegated B-row calls.
    pub digit_rows_gpu_time: Duration,
    /// Successful delegated compression calls after the most recent inner commit.
    pub compression_calls: usize,
    /// Cumulative wall time in delegated compression calls.
    pub compression_time: Duration,
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
    cpu: CpuBackend,
    initialization_time: Duration,
    last_metrics: Mutex<Option<MetalCommitMetrics>>,
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
        Self::new_with_cpu_backend(policy, CpuBackend::DEFAULT)
    }

    /// Build a backend with an explicit CPU fallback resource policy.
    pub fn new_with_cpu_backend(
        policy: MetalExecutionPolicy,
        cpu: CpuBackend,
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
                cpu,
                initialization_time: start.elapsed(),
                last_metrics: Mutex::new(None),
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
                cpu,
                initialization_time: start.elapsed(),
                last_metrics: Mutex::new(None),
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
}

impl<Field: MetalField> ComputeBackendSetup<Field> for MetalCommitBackend<Field>
where
    CpuBackend: ComputeBackendSetup<Field, PreparedSetup = CpuPreparedSetup<Field>>,
{
    type PreparedSetup = MetalPreparedSetup<Field>;

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
    fn digit_rows_batch_impl<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup<F>,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        if digit_vectors.is_empty() {
            return Ok(Vec::new());
        }
        let start = Instant::now();
        let num_cols = digit_vectors[0].len();
        let use_metal = D == 64
            && row_len != 0
            && num_cols != 0
            && num_cols <= 524_288
            && log_basis == 3
            && digit_vectors.iter().all(|digits| digits.len() == num_cols)
            && digit_vectors.iter().all(|digits| {
                digits
                    .iter()
                    .flatten()
                    .all(|&digit| (-4..=3).contains(&digit))
            });
        let (row_batches, used_metal, metal_gpu_time) = if use_metal {
            if let Some(runtime) = self.runtime() {
                let matrix = prepared.matrix(runtime, D, row_len, num_cols)?;
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
                        },
                    )
                    .map_err(MetalCommitError::into_akita)?;
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
                let row_batches = coefficients
                    .chunks_exact(coefficients_per_vector)
                    .map(|vector| {
                        vector
                            .chunks_exact(D)
                            .map(CyclotomicRing::from_slice)
                            .collect()
                    })
                    .collect();
                (row_batches, true, outcome.timings.gpu)
            } else {
                (
                    digit_vectors
                        .iter()
                        .map(|digits| {
                            self.cpu_backend()
                                .digit_rows(&prepared.cpu, row_len, digits, log_basis)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    false,
                    None,
                )
            }
        } else {
            (
                digit_vectors
                    .iter()
                    .map(|digits| {
                        self.cpu_backend()
                            .digit_rows(&prepared.cpu, row_len, digits, log_basis)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                false,
                None,
            )
        };
        let elapsed = start.elapsed();
        self.update_metrics(|metrics| {
            metrics.digit_rows_calls += digit_vectors.len();
            metrics.digit_rows_metal_calls += digit_vectors.len() * usize::from(used_metal);
            metrics.digit_rows_time += elapsed;
            metrics.digit_rows_gpu_time += metal_gpu_time.unwrap_or_default();
        })
        .map_err(MetalCommitError::into_akita)?;
        Ok(row_batches)
    }
}

impl DigitRowsComputeBackend<F> for MetalCommitBackend<F>
where
    CpuBackend:
        ComputeBackendSetup<F, PreparedSetup = CpuPreparedSetup<F>> + DigitRowsComputeBackend<F>,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let mut batches = self.digit_rows_batch_impl(prepared, row_len, &[digits], log_basis)?;
        if batches.len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "single digit-row dispatch returned an invalid batch count".into(),
            ));
        }
        Ok(batches.remove(0))
    }

    fn digit_rows_batch<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        self.digit_rows_batch_impl(prepared, row_len, digit_vectors, log_basis)
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

impl<S, Field, E, const D: usize> SubringCoefficientPackingBatchKernel<S, Field, E, D>
    for MetalCommitBackend<Field>
where
    Field: MetalField,
    E: ExtField<Field>,
    CpuBackend: ComputeBackendSetup<Field, PreparedSetup = CpuPreparedSetup<Field>>
        + SubringCoefficientPackingBatchKernel<S, Field, E, D>,
{
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<Field>>, AkitaError> {
        <CpuBackend as SubringCoefficientPackingBatchKernel<S, Field, E, D>>::coefficient_packing_partials_batch(
            &self.cpu_backend(),
            prepared.map(|value| &value.cpu),
            source,
            plan,
        )
    }
}

#[cfg(test)]
mod tests {
    use akita_prover::{AkitaProverSetup, ComputeBackendSetup, DigitRowsComputeBackend};
    use akita_types::SetupMatrixCapacity;

    use super::*;

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
    }
}
