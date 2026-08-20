use akita_algebra::ring::cyclotomic::WideCyclotomicRing;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::unreduced::HasCommitAccum;
use akita_field::{AkitaError, Prime128OffsetA7F7};

use super::PackedOneHotView;
use crate::compute::{CommitInnerPlan, ComputeBackendSetup, RootCommitKernel};
use crate::{CommitInnerWitness, CpuBackend, CpuPreparedSetup};

type F = Prime128OffsetA7F7;
type CommitWideRing<const D: usize> = WideCyclotomicRing<<F as HasCommitAccum>::CommitAccum, D>;

struct ValidatedPackedCommit {
    blocks_per_column: usize,
    active_a_cols: usize,
}

fn validate_packed_commit<const D: usize>(
    source: PackedOneHotView<'_, F, D>,
    plan: CommitInnerPlan,
) -> Result<ValidatedPackedCommit, AkitaError> {
    if plan.n_a == 0 || plan.num_positions_per_block == 0 || plan.num_digits_inner == 0 {
        return Err(AkitaError::InvalidSetup(
            "packed one-hot commitment needs nonzero rows, positions, and digit count".into(),
        ));
    }
    let segment_field_elems = source
        .num_rows()
        .checked_mul(source.onehot_k())
        .ok_or_else(|| AkitaError::InvalidInput("packed one-hot segment size overflow".into()))?;
    if !segment_field_elems.is_multiple_of(D) {
        return Err(AkitaError::InvalidInput(format!(
            "packed one-hot column segment has {segment_field_elems} fields, not a whole number of D={D} rings"
        )));
    }
    let segment_rings = segment_field_elems / D;
    if !segment_rings.is_multiple_of(plan.num_positions_per_block) {
        return Err(AkitaError::InvalidSetup(format!(
            "packed one-hot column segment of {segment_rings} rings must align to P={} commit blocks",
            plan.num_positions_per_block
        )));
    }
    let block_field_elems = plan
        .num_positions_per_block
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("packed one-hot block width overflow".into()))?;
    if !block_field_elems.is_multiple_of(source.onehot_k()) {
        return Err(AkitaError::InvalidSetup(format!(
            "packed one-hot block width {block_field_elems} must contain whole K={} chunks",
            source.onehot_k()
        )));
    }
    let active_a_cols = plan
        .num_positions_per_block
        .checked_mul(plan.num_digits_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".into()))?;
    Ok(ValidatedPackedCommit {
        blocks_per_column: segment_rings / plan.num_positions_per_block,
        active_a_cols,
    })
}

fn direct_block<const D: usize>(
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    source: PackedOneHotView<'_, F, D>,
    plan: CommitInnerPlan,
    column: usize,
    block_in_column: usize,
) -> Vec<CyclotomicRing<F, D>> {
    if column >= source.num_columns() {
        return vec![CyclotomicRing::zero(); plan.n_a];
    }
    let ring_start = block_in_column * plan.num_positions_per_block;
    let field_start = ring_start * D;
    let field_end = field_start + plan.num_positions_per_block * D;
    let row_start = field_start / source.onehot_k();
    let row_end = field_end / source.onehot_k();
    a_view
        .rows()
        .take(plan.n_a)
        .map(|a_row| {
            let mut wide = CommitWideRing::<D>::zero();
            let mut partial = CyclotomicRing::<F, D>::zero();
            let mut accumulated = 0usize;
            for row in row_start..row_end {
                let lane = usize::from(source.lanes()[row * source.num_columns() + column]);
                if lane == 0 {
                    continue;
                }
                if accumulated == F::MAX_COMMIT_ACCUMULATIONS {
                    partial += wide.reduce();
                    wide = CommitWideRing::zero();
                    accumulated = 0;
                }
                let local_field = row * source.onehot_k() + lane;
                let position = local_field / D - ring_start;
                let a_column = position * plan.num_digits_inner;
                CommitWideRing::from_ring(&a_row[a_column])
                    .shift_accumulate_into(&mut wide, local_field % D);
                accumulated += 1;
            }
            partial + wide.reduce()
        })
        .collect()
}

fn commit_packed_onehot<const D: usize>(
    backend: &CpuBackend,
    prepared: &CpuPreparedSetup<F>,
    source: PackedOneHotView<'_, F, D>,
    plan: CommitInnerPlan,
) -> Result<CommitInnerWitness<F>, AkitaError> {
    let shape = validate_packed_commit(source, plan)?;
    let a_view = backend
        .prepared_expanded_setup(prepared)
        .shared_matrix
        .ring_view::<D>(plan.n_a, shape.active_a_cols)?;
    let block_count = shape.blocks_per_column;
    let rows = cfg_into_iter!(0..source.column_capacity() * block_count)
        .map(|output_block| {
            direct_block(
                &a_view,
                source,
                plan,
                output_block / block_count,
                output_block % block_count,
            )
        })
        .collect();
    Ok(CommitInnerWitness::from_rows::<D>(rows))
}

impl<const D: usize> RootCommitKernel<PackedOneHotView<'_, F, D>, F, D> for CpuBackend {
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<PackedOneHotView<'_, F, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        let [source] = sources.as_slice() else {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot commitment requires exactly one physical source, got {}",
                sources.len()
            )));
        };
        commit_packed_onehot(self, prepared, *source, plan).map(|witness| vec![witness])
    }
}
