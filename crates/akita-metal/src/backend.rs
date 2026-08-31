use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_prover::compute::{CompressionComputeBackend, CompressionRowsProducts};
use akita_prover::{
    ComputeBackendSetup, CpuBackend, CyclicRowsComputeBackend, DigitRowsComputeBackend,
    NttCacheOwnerId, NttOperationCluster, RoutedNttRequirement,
};
use akita_types::{AkitaExpandedSetup, NttCacheKey};
use metal::Device;

use crate::field::F;
use crate::prepared::MetalPreparedSetup;
use crate::runtime::{MetalDeviceCapabilities, MetalOneHotKernel, MetalRuntime};
use crate::{MetalCommitError, MetalExecutionPolicy};

/// Timings and structural counters from the most recent one-hot commitment.
#[derive(Clone, Debug, Default)]
pub struct MetalCommitMetrics {
    /// Kernel variant selected for the dispatch.
    pub kernel: MetalOneHotKernel,
    /// Independent source blocks processed by one threadgroup.
    pub blocks_per_threadgroup: usize,
    /// Number of committed sources.
    pub num_sources: usize,
    /// Number of nonzero one-hot chunks across all sources.
    pub hot_entries: usize,
    /// Number of field additions represented by the sparse operation.
    pub field_additions: u64,
    /// Conservative direct-gather A-matrix traffic.
    pub gathered_matrix_bytes: u64,
    /// Canonical output bytes returned by the dispatch.
    pub output_bytes: usize,
    /// Device-only transient bytes allocated by the dispatch.
    pub scratch_bytes: usize,
    /// Resident A-prefix bytes.
    pub matrix_bytes: usize,
    /// Whether the A-prefix was already resident.
    pub matrix_cache_hit: bool,
    /// A-prefix packing and allocation time on a miss.
    pub matrix_prepare_time: Duration,
    /// Host time spent packing sparse indices.
    pub index_pack_time: Duration,
    /// Per-call Metal input/output buffer setup.
    pub buffer_setup_time: Duration,
    /// Wall time from command commit through completion.
    pub command_wall_time: Duration,
    /// GPU timestamp interval, when reported by the device.
    pub gpu_time: Option<Duration>,
    /// Copy from shared output storage into owned Rust limbs.
    pub readback_copy_time: Duration,
    /// Canonical field reconstruction and witness assembly.
    pub output_reconstruction_time: Duration,
    /// Complete kernel-boundary wall time.
    pub total_time: Duration,
}

struct BackendInner {
    runtime: Option<Arc<MetalRuntime>>,
    policy: MetalExecutionPolicy,
    cpu: CpuBackend,
    initialization_time: Duration,
    last_commit_metrics: Mutex<Option<MetalCommitMetrics>>,
}

/// Akita compute backend with explicit Metal admission and CPU delegation.
#[derive(Clone)]
pub struct MetalBackend {
    inner: Arc<BackendInner>,
}

impl MetalBackend {
    /// Whether a system Metal device is visible.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Build a backend from the system default device.
    pub fn new(policy: MetalExecutionPolicy) -> Result<Self, MetalCommitError> {
        Self::new_with_cpu_backend(policy, CpuBackend::DEFAULT)
    }

    /// Build a backend with explicit CPU resource limits for delegated work.
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
                last_commit_metrics: Mutex::new(None),
            }),
        })
    }

    /// Build a backend around an existing Metal device.
    pub fn with_device(
        device: Device,
        policy: MetalExecutionPolicy,
    ) -> Result<Self, MetalCommitError> {
        Self::with_device_and_cpu_backend(device, policy, CpuBackend::DEFAULT)
    }

    /// Build around an existing device with explicit CPU resource limits.
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
                last_commit_metrics: Mutex::new(None),
            }),
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

    /// CPU backend used for delegated setup and operations.
    pub fn cpu_backend(&self) -> CpuBackend {
        self.inner.cpu
    }

    /// Stable device properties, or `None` when preferred Metal is unavailable.
    pub fn capabilities(&self) -> Option<MetalDeviceCapabilities> {
        self.runtime().map(MetalRuntime::capabilities)
    }

    /// Metrics for the most recent successful Metal commitment dispatch.
    pub fn last_commit_metrics(&self) -> Result<Option<MetalCommitMetrics>, MetalCommitError> {
        Ok(self
            .inner
            .last_commit_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .clone())
    }

    pub(crate) fn runtime(&self) -> Option<&MetalRuntime> {
        self.inner.runtime.as_deref()
    }

    pub(crate) fn record_commit_metrics(
        &self,
        metrics: MetalCommitMetrics,
    ) -> Result<(), MetalCommitError> {
        *self
            .inner
            .last_commit_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)? = Some(metrics);
        Ok(())
    }
}

impl ComputeBackendSetup<F> for MetalBackend {
    type PreparedSetup = MetalPreparedSetup;

    fn prepare_expanded(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
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
        requirement: RoutedNttRequirement,
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
    ) -> &'a AkitaExpandedSetup<F> {
        prepared.expanded.as_ref()
    }

    fn release_built_ntt_slots(&self, prepared: &Self::PreparedSetup) -> Result<usize, AkitaError> {
        self.cpu_backend().release_built_ntt_slots(&prepared.cpu)
    }
}

impl CompressionComputeBackend<F> for MetalBackend {
    fn compression_cache_bytes(&self, prepared: &Self::PreparedSetup) -> Option<usize> {
        self.cpu_backend().compression_cache_bytes(&prepared.cpu)
    }

    fn compression_rows_products<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Vec<CompressionRowsProducts<F, D>>, AkitaError> {
        self.cpu_backend()
            .compression_rows_products(&prepared.cpu, digit_vectors)
    }
}

impl DigitRowsComputeBackend<F> for MetalBackend {
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        self.cpu_backend()
            .digit_rows(&prepared.cpu, row_len, digit_vectors, log_basis)
    }
}

impl CyclicRowsComputeBackend<F> for MetalBackend {
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
