use akita_error::AkitaError;
use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
use akita_prover::{CommitInnerWitness, OneHotPoly, RootCommitSource};

use crate::backend::MetalBackend;
use crate::field::F;
use crate::prepared::MetalPreparedSetup;
use crate::{MetalCommitError, MetalExecutionPolicy};

/// Borrowed row-major selectors for the packed Metal root-commit kernel.
///
/// Byte zero denotes an absent coefficient. Values in `1..onehot_k` select
/// that coefficient within the row's one-hot chunk. Logical chunks are
/// column-major through `column_capacity`; omitted suffix columns are zero.
#[derive(Clone, Copy, Debug)]
pub struct PackedOneHotCommitView<'a> {
    lanes: &'a [u8],
    active_zero_rows: &'a [u64],
    zero_column_mask: u64,
    num_rows: usize,
    num_columns: usize,
    column_capacity: usize,
    onehot_k: usize,
    hot_entries: usize,
}

impl<'a> PackedOneHotCommitView<'a> {
    /// Validate a borrowed packed selector matrix without copying it.
    pub fn new(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
    ) -> Result<Self, AkitaError> {
        Self::new_with_active_zero_rows(onehot_k, column_capacity, num_columns, lanes, &[], 0)
    }

    /// Validate selectors whose row-zero coefficients are described by a
    /// compact active-row bitset and a fixed live-column mask.
    pub fn new_with_active_zero_rows(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
    ) -> Result<Self, AkitaError> {
        if onehot_k == 0 || onehot_k > usize::from(u8::MAX) + 1 || !onehot_k.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot K={onehot_k} must be a power of two at most 256"
            )));
        }
        if column_capacity == 0 || !column_capacity.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot column capacity {column_capacity} must be a nonzero power of two"
            )));
        }
        if num_columns == 0 || num_columns > column_capacity {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot live column count {num_columns} must be in 1..={column_capacity}"
            )));
        }
        if !lanes.len().is_multiple_of(num_columns) {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot lane count {} is not divisible by {num_columns} live columns",
                lanes.len()
            )));
        }
        let num_rows = lanes.len() / num_columns;
        if num_rows == 0 || !num_rows.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot row count {num_rows} must be a nonzero power of two"
            )));
        }
        let total_field_elements = num_rows
            .checked_mul(column_capacity)
            .and_then(|chunks| chunks.checked_mul(onehot_k))
            .ok_or_else(|| {
                AkitaError::InvalidInput("packed one-hot logical size overflow".into())
            })?;
        if !total_field_elements.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot logical field length {total_field_elements} is not a power of two"
            )));
        }
        let live_column_mask = if num_columns == u64::BITS as usize {
            u64::MAX
        } else {
            (1u64 << num_columns) - 1
        };
        if zero_column_mask & !live_column_mask != 0 {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot committed-zero mask {zero_column_mask:#x} exceeds {num_columns} live columns"
            )));
        }
        let expected_active_words = num_rows.div_ceil(u64::BITS as usize);
        if zero_column_mask == 0 {
            if !active_zero_rows.is_empty() {
                return Err(AkitaError::InvalidInput(
                    "packed one-hot active-zero rows require a nonzero column mask".into(),
                ));
            }
        } else if active_zero_rows.len() != expected_active_words {
            return Err(AkitaError::InvalidSize {
                expected: expected_active_words,
                actual: active_zero_rows.len(),
            });
        }
        let mut hot_entries = 0usize;
        for (position, &lane) in lanes.iter().enumerate() {
            if usize::from(lane) >= onehot_k {
                return Err(AkitaError::InvalidInput(format!(
                    "packed one-hot lane {lane} at byte {position} is outside K={onehot_k}"
                )));
            }
            let row = position / num_columns;
            let column = position % num_columns;
            let committed_zero = lane == 0
                && zero_column_mask & (1u64 << column) != 0
                && active_zero_rows
                    .get(row / u64::BITS as usize)
                    .is_some_and(|word| word & (1u64 << (row % u64::BITS as usize)) != 0);
            hot_entries += usize::from(lane != 0 || committed_zero);
        }
        Ok(Self {
            lanes,
            active_zero_rows,
            zero_column_mask,
            num_rows,
            num_columns,
            column_capacity,
            onehot_k,
            hot_entries,
        })
    }

    pub(crate) fn lanes(self) -> &'a [u8] {
        self.lanes
    }

    pub(crate) fn active_zero_rows(self) -> &'a [u64] {
        self.active_zero_rows
    }

    pub(crate) fn zero_column_mask(self) -> u64 {
        self.zero_column_mask
    }

    pub(crate) fn commits_zero_at(self, row: usize, column: usize) -> bool {
        self.zero_column_mask & (1u64 << column) != 0
            && self
                .active_zero_rows
                .get(row / u64::BITS as usize)
                .is_some_and(|word| word & (1u64 << (row % u64::BITS as usize)) != 0)
    }

    pub(crate) fn num_rows(self) -> usize {
        self.num_rows
    }

    pub(crate) fn num_columns(self) -> usize {
        self.num_columns
    }

    pub(crate) fn column_capacity(self) -> usize {
        self.column_capacity
    }

    pub(crate) fn onehot_k(self) -> usize {
        self.onehot_k
    }

    pub(crate) fn hot_entries(self) -> usize {
        self.hot_entries
    }
}

impl<const D: usize> RootCommitKernel<PackedOneHotCommitView<'_>, F, D> for MetalBackend {
    #[tracing::instrument(skip_all, name = "MetalBackend::packed_onehot_commit_inner")]
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<PackedOneHotCommitView<'_>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        let [source] = sources.as_slice() else {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot commitment requires exactly one physical source, got {}",
                sources.len()
            )));
        };
        self.commit_packed_onehot::<D>(prepared, *source, plan)
            .map(|witness| vec![witness])
    }
}

impl MetalBackend {
    /// Commit one resident packed source through the D512/K256 Metal kernel.
    pub fn commit_packed_onehot<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        source: PackedOneHotCommitView<'_>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        let shape = match crate::packed_onehot_fp128_d512::validate_shape::<D>(source, plan) {
            Ok(shape) => shape,
            Err(error) if self.policy() == MetalExecutionPolicy::RequireMetal => return Err(error),
            Err(_) => return self.commit_packed_onehot_cpu::<D>(prepared, source, plan),
        };
        let Some(runtime) = self.runtime() else {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => {
                    Err(MetalCommitError::DeviceUnavailable.into_akita())
                }
                MetalExecutionPolicy::PreferMetal => {
                    self.commit_packed_onehot_cpu::<D>(prepared, source, plan)
                }
            };
        };
        if !runtime.supports_packed_fp128_d512_panels() {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 panel pipeline is unavailable".into(),
                )
                .into_akita()),
                MetalExecutionPolicy::PreferMetal => {
                    self.commit_packed_onehot_cpu::<D>(prepared, source, plan)
                }
            };
        }
        crate::packed_onehot_fp128_d512::commit_validated::<D>(
            self, prepared, runtime, source, plan, shape,
        )
    }

    fn commit_packed_onehot_cpu<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        source: PackedOneHotCommitView<'_>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        let indices = (0..source.column_capacity())
            .flat_map(|column| {
                (0..source.num_rows()).map(move |row| {
                    if column >= source.num_columns() {
                        None
                    } else {
                        let lane = source.lanes()[row * source.num_columns() + column];
                        (lane != 0 || source.commits_zero_at(row, column)).then_some(lane)
                    }
                })
            })
            .collect();
        let poly = OneHotPoly::<F, u8>::new(source.onehot_k(), indices)?;
        let view = <OneHotPoly<F, u8> as RootCommitSource<F, D>>::commit_view(&poly)?;
        let mut witnesses =
            self.cpu_backend()
                .commit_inner_group(&prepared.cpu, vec![view], plan)?;
        witnesses.pop().ok_or_else(|| {
            AkitaError::InvalidSetup("CPU packed fallback returned no commitment witness".into())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_borrowed_geometry_and_lanes() {
        let lanes = vec![0, 1, 255, 7, 0, 3, 2, 0];
        let view = PackedOneHotCommitView::new(256, 4, 2, &lanes).unwrap();
        assert_eq!(view.lanes().as_ptr(), lanes.as_ptr());
        assert_eq!(view.num_rows(), 4);
        assert_eq!(view.hot_entries(), 5);

        assert!(PackedOneHotCommitView::new(256, 4, 3, &lanes).is_err());
        assert!(PackedOneHotCommitView::new(128, 4, 2, &lanes).is_err());
    }

    #[test]
    fn active_rows_distinguish_committed_zero_from_absence() {
        let lanes = vec![0, 0, 0, 7, 0, 0, 3, 0];
        let view =
            PackedOneHotCommitView::new_with_active_zero_rows(16, 4, 2, &lanes, &[0b0101], 0b01)
                .unwrap();
        assert!(view.commits_zero_at(0, 0));
        assert!(!view.commits_zero_at(1, 0));
        assert!(view.commits_zero_at(2, 0));
        assert!(!view.commits_zero_at(0, 1));
        assert_eq!(view.hot_entries(), 4);
    }
}
