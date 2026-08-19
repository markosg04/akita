use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use akita_field::{AkitaError, FieldCore};
use akita_prover::CpuPreparedSetup;
use akita_types::AkitaExpandedSetup;
use metal::Buffer;

use crate::field::{MetalField, F};
use crate::runtime::MetalRuntime;
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

pub(crate) struct MatrixPreparation {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) bytes: usize,
    pub(crate) cache_hit: bool,
    pub(crate) prepare_time: Duration,
}

/// Setup-bound CPU fallback state and lazily packed Metal matrix prefixes.
pub struct MetalPreparedSetup<Field: FieldCore = F> {
    pub(crate) cpu: CpuPreparedSetup<Field>,
    pub(crate) expanded: Arc<AkitaExpandedSetup<Field>>,
    matrices: Mutex<HashMap<MatrixCacheKey, PreparedMatrix>>,
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
        }
    }

    pub(crate) fn matrix(
        &self,
        runtime: &MetalRuntime,
        ring_d: usize,
        n_a: usize,
        active_a_cols: usize,
    ) -> Result<MatrixPreparation, AkitaError> {
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

        let start = Instant::now();
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

    /// Number of exact A-matrix shapes currently resident on the device.
    pub fn matrix_cache_entries(&self) -> Result<usize, MetalCommitError> {
        Ok(self
            .matrices
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .len())
    }

    /// Total resident bytes across exact A-matrix shapes.
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
