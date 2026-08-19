use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, ExtField, MulBaseUnreduced};
use akita_prover::compute::{
    CompressionComputeBackend, CompressionRowsProducts, ComputeBackendSetup,
    CyclicRowsComputeBackend, DigitRowsComputeBackend, TensorPackedWitness, TensorProjectionKernel,
};
use akita_prover::{CpuBackend, CpuPreparedSetup, NttCacheOwnerId};
use akita_types::{AkitaExpandedSetup, NttCacheKey};
use metal::Device;

use crate::field::{MetalField, F};
use crate::prepared::MetalPreparedSetup;
use crate::runtime::{MetalDeviceCapabilities, MetalOneHotKernel, MetalRuntime};
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
    /// Cumulative wall time in delegated B-row calls.
    pub digit_rows_time: Duration,
    /// Successful delegated compression calls after the most recent inner commit.
    pub compression_calls: usize,
    /// Cumulative wall time in delegated compression calls.
    pub compression_time: Duration,
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

impl<Field: MetalField> DigitRowsComputeBackend<Field> for MetalCommitBackend<Field>
where
    CpuBackend: ComputeBackendSetup<Field, PreparedSetup = CpuPreparedSetup<Field>>
        + DigitRowsComputeBackend<Field>,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<Field, D>>, AkitaError> {
        let start = Instant::now();
        let rows = self
            .cpu_backend()
            .digit_rows(&prepared.cpu, row_len, digits, log_basis)?;
        let elapsed = start.elapsed();
        self.update_metrics(|metrics| {
            metrics.digit_rows_calls += 1;
            metrics.digit_rows_time += elapsed;
        })
        .map_err(MetalCommitError::into_akita)?;
        Ok(rows)
    }
}

impl<Field: MetalField> CyclicRowsComputeBackend<Field> for MetalCommitBackend<Field>
where
    CpuBackend: ComputeBackendSetup<Field, PreparedSetup = CpuPreparedSetup<Field>>
        + CyclicRowsComputeBackend<Field>,
{
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<Field, D>>, AkitaError> {
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
