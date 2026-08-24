use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use akita_field::{AkitaError, FieldCore};
use akita_prover::CpuPreparedSetup;
use akita_types::AkitaExpandedSetup;
use metal::Buffer;

use crate::field::{MetalField, F};
use crate::runtime::{DispatchTimings, MetalRuntime, RecursiveCommitParams, SharedSliceBuffer};
use crate::MetalCommitError;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MatrixCacheKey {
    ring_d: usize,
    n_a: usize,
    active_a_cols: usize,
}

struct PreparedMatrix {
    buffer: Arc<Buffer>,
    bytes: usize,
}

struct PreparedRecursiveMatrix {
    buffer: Arc<Buffer>,
    bytes: usize,
}

pub(crate) struct MatrixPreparation {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) bytes: usize,
    pub(crate) cache_hit: bool,
    pub(crate) prepare_time: Duration,
}

pub(crate) struct SharedMatrixPreparation<'a> {
    pub(crate) buffer: Buffer,
    pub(crate) bytes: usize,
    pub(crate) zero_copy: bool,
    pub(crate) prepare_time: Duration,
    marker: std::marker::PhantomData<&'a [F]>,
}

pub(crate) struct RecursiveMatrixPreparation {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) bytes: usize,
    pub(crate) cache_hit: bool,
    pub(crate) timings: DispatchTimings,
    pub(crate) allocation_bytes: usize,
}

/// Setup-bound CPU fallback state and lazily packed Metal matrix prefixes.
pub struct MetalPreparedSetup<Field: FieldCore = F> {
    pub(crate) cpu: CpuPreparedSetup<Field>,
    pub(crate) expanded: Arc<AkitaExpandedSetup<Field>>,
    matrices: Mutex<HashMap<MatrixCacheKey, PreparedMatrix>>,
    recursive_matrices: Mutex<HashMap<MatrixCacheKey, PreparedRecursiveMatrix>>,
}

#[expect(
    private_bounds,
    reason = "only the backend constructs prepared setups for its sealed field set"
)]
impl<Field: MetalField> MetalPreparedSetup<Field> {
    pub(crate) fn new(
        cpu: CpuPreparedSetup<Field>,
        expanded: Arc<AkitaExpandedSetup<Field>>,
    ) -> Self {
        Self {
            cpu,
            expanded,
            matrices: Mutex::new(HashMap::new()),
            recursive_matrices: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn matrix(
        &self,
        runtime: &MetalRuntime,
        ring_d: usize,
        n_a: usize,
        active_a_cols: usize,
    ) -> Result<MatrixPreparation, AkitaError> {
        self.expanded
            .shared_matrix
            .ring_view_dyn(n_a, active_a_cols, ring_d)?;
        let field_count = n_a
            .checked_mul(active_a_cols)
            .and_then(|count| count.checked_mul(ring_d))
            .ok_or_else(|| MetalCommitError::ShapeOverflow("A matrix field count").into_akita())?;
        let required_bytes = field_count
            .checked_mul(size_of::<Field::DeviceElement>())
            .ok_or_else(|| MetalCommitError::ShapeOverflow("A matrix bytes").into_akita())?;
        let key = MatrixCacheKey {
            ring_d,
            n_a,
            active_a_cols,
        };
        let mut matrices = self
            .matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock.into_akita())?;
        if let Some(prepared) = matrices.get(&key) {
            return Ok(MatrixPreparation {
                buffer: prepared.buffer.clone(),
                bytes: prepared.bytes,
                cache_hit: true,
                prepare_time: Duration::ZERO,
            });
        }
        if let Some(prepared) = matrices
            .values()
            .filter(|prepared| prepared.bytes >= required_bytes)
            .min_by_key(|prepared| prepared.bytes)
        {
            return Ok(MatrixPreparation {
                buffer: prepared.buffer.clone(),
                bytes: prepared.bytes,
                cache_hit: true,
                prepare_time: Duration::ZERO,
            });
        }

        let start = Instant::now();
        let fields = self
            .expanded
            .shared_matrix
            .as_field_slice()
            .get(..field_count)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "validated Metal A-matrix prefix is unexpectedly unavailable".into(),
                )
            })?;
        let packed = fields
            .iter()
            .copied()
            .map(Field::into_device)
            .collect::<Vec<_>>();
        let bytes = size_of_val(packed.as_slice());
        let buffer = runtime
            .private_buffer_from_slice(&packed)
            .map_err(MetalCommitError::into_akita)?;
        let buffer = Arc::new(buffer);
        let prepare_time = start.elapsed();
        matrices.insert(
            key,
            PreparedMatrix {
                buffer: buffer.clone(),
                bytes,
            },
        );
        Ok(MatrixPreparation {
            buffer,
            bytes,
            cache_hit: false,
            prepare_time,
        })
    }

    /// Number of independently allocated matrix prefixes resident on the device.
    pub fn matrix_cache_entries(&self) -> Result<usize, MetalCommitError> {
        Ok(self
            .matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .len())
    }

    /// Total bytes across independently allocated resident matrix prefixes.
    pub fn matrix_cache_bytes(&self) -> Result<usize, MetalCommitError> {
        self.matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .values()
            .try_fold(0usize, |total, matrix| {
                total
                    .checked_add(matrix.bytes)
                    .ok_or(MetalCommitError::ShapeOverflow("matrix cache bytes"))
            })
    }
}

impl MetalPreparedSetup<F> {
    pub(crate) fn recursive_commit_matrix<const D: usize>(
        &self,
        runtime: &MetalRuntime,
        params: RecursiveCommitParams,
    ) -> Result<RecursiveMatrixPreparation, AkitaError> {
        let n_a = usize::try_from(params.num_rows)
            .map_err(|_| MetalCommitError::ShapeOverflow("recursive matrix rows").into_akita())?;
        let active_a_cols = usize::try_from(params.num_cols).map_err(|_| {
            MetalCommitError::ShapeOverflow("recursive matrix columns").into_akita()
        })?;
        let key = MatrixCacheKey {
            ring_d: D,
            n_a,
            active_a_cols,
        };
        let mut matrices = self
            .recursive_matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock.into_akita())?;
        if let Some(prepared) = matrices.get(&key) {
            return Ok(RecursiveMatrixPreparation {
                buffer: prepared.buffer.clone(),
                bytes: prepared.bytes,
                cache_hit: true,
                timings: DispatchTimings::default(),
                allocation_bytes: 0,
            });
        }

        let source = self.shared_matrix(runtime, D, n_a, active_a_cols)?;
        let mut outcome = runtime
            .prepare_fp128_recursive_commit_matrix::<D>(&source.buffer, params)
            .map_err(MetalCommitError::into_akita)?;
        outcome.timings.buffer_setup += source.prepare_time;
        let allocation_bytes = outcome
            .allocation_bytes
            .saturating_add(source.bytes.saturating_mul(usize::from(!source.zero_copy)));
        let buffer = Arc::new(outcome.buffer);
        let bytes = outcome.allocation_bytes;
        matrices.insert(
            key,
            PreparedRecursiveMatrix {
                buffer: buffer.clone(),
                bytes,
            },
        );
        Ok(RecursiveMatrixPreparation {
            buffer,
            bytes,
            cache_hit: false,
            timings: outcome.timings,
            allocation_bytes,
        })
    }

    pub(crate) fn shared_matrix(
        &self,
        runtime: &MetalRuntime,
        ring_d: usize,
        n_a: usize,
        active_a_cols: usize,
    ) -> Result<SharedMatrixPreparation<'_>, AkitaError> {
        self.expanded
            .shared_matrix
            .ring_view_dyn(n_a, active_a_cols, ring_d)?;
        let field_count = n_a
            .checked_mul(active_a_cols)
            .and_then(|count| count.checked_mul(ring_d))
            .ok_or_else(|| MetalCommitError::ShapeOverflow("A matrix field count").into_akita())?;
        let fields = self
            .expanded
            .shared_matrix
            .as_field_slice()
            .get(..field_count)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "validated Metal A-matrix prefix is unexpectedly unavailable".into(),
                )
            })?;
        let start = Instant::now();
        let SharedSliceBuffer {
            buffer, zero_copy, ..
        } = runtime
            .shared_slice_buffer(fields)
            .map_err(MetalCommitError::into_akita)?;
        Ok(SharedMatrixPreparation {
            buffer,
            bytes: size_of_val(fields),
            zero_copy,
            prepare_time: start.elapsed(),
            marker: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use akita_prover::{AkitaProverSetup, ComputeBackendSetup};
    use akita_types::SetupMatrixCapacity;

    use super::*;
    use crate::{MetalCommitBackend, MetalExecutionPolicy};

    #[test]
    fn larger_resident_matrix_serves_a_smaller_reshaped_prefix() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            20,
            1,
            SetupMatrixCapacity {
                num_field_elements: 512 * 16,
            },
        )
        .unwrap();
        let backend = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let prepared = backend.prepare_setup(&setup).unwrap();
        let runtime = backend.runtime().unwrap();

        let root = prepared.matrix(runtime, 512, 1, 16).unwrap();
        assert!(!root.cache_hit);
        let outer = prepared.matrix(runtime, 64, 2, 32).unwrap();
        assert!(outer.cache_hit);
        assert_eq!(outer.prepare_time, Duration::ZERO);
        assert!(Arc::ptr_eq(&root.buffer, &outer.buffer));
        assert_eq!(prepared.matrix_cache_entries().unwrap(), 1);
        assert_eq!(prepared.matrix_cache_bytes().unwrap(), root.bytes);
    }
}
