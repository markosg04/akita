//! Semantic relation-weight events and their canonical consumers.

#[path = "relation_weights/setup_columns.rs"]
mod setup_columns;

use std::ops::Range;

use crate::protocol::sumcheck::relation_range_image::{
    DirectLinearSource, DirectSparseLinearSource, NegacyclicSetupLinearSegment,
    NegacyclicSetupLinearTerms,
};
use akita_algebra::eq_poly::SplitEqEvals;
use akita_algebra::ring::scalar_powers;
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase, MulBase, MulBaseUnreduced,
};
use akita_types::{
    dispatch_for_field, gadget_row_scalars, prepare_coefficient_packing_batch_semantics,
    r_decomp_levels, AkitaExpandedSetup, CoefficientPackingBatchSemanticInputs,
    CoefficientPackingBatchSemantics, CommittedGroupParams, FpExtEncoding, OpeningClaimsLayout,
    OpeningFamily, OpeningMethod, PreparedSubringCoefficientPackingPoint, RelationAddressGeometry,
    RelationRangeImagePlan, RelationRowFamily, RelationWitnessGeometry, RingRelationInstance,
    SetupProjectionGeometry,
};
pub use akita_types::{RelationWeightContribution, RelationWeightEvent};
use setup_columns::{evaluate_setup_columns, SetupRows};

/// Source of setup-matrix relation weights for this evaluation.
#[derive(Clone, Copy)]
pub enum RelationSetupSource<'a, F: FieldCore> {
    /// Emit setup events directly from the expanded setup matrix.
    Matrix(&'a AkitaExpandedSetup<F>),
    /// Omit setup events because their complete evaluation is supplied separately.
    DeferredClaim,
}

/// Inputs to the one semantic relation-event builder.
pub struct RelationWeightEventInputs<'a, F: FieldCore, E: FieldCore> {
    pub setup: RelationSetupSource<'a, F>,
    pub instance: &'a RingRelationInstance<F>,
    pub alpha: E,
    pub level_params: &'a CommittedGroupParams,
    pub relation_row_point: &'a [E],
    pub claim_coefficients: &'a [E],
    pub opening_source_len: usize,
    pub opening_ring_dim: usize,
    pub relation_plan: &'a RelationRangeImagePlan,
    /// Method-typed prepared points for the current fold.
    pub opening_points:
        OpeningFamily<(), &'a [(usize, &'a PreparedSubringCoefficientPackingPoint<E>)]>,
}

mod events;
pub use events::{RelationWeightEvents, RelationWeightFactorization};

fn relation_d_group_width(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    relation_geometry: &RelationWitnessGeometry,
    group_index: usize,
) -> Result<usize, AkitaError> {
    let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
    let group_dims = lp.group_role_dims_geometry(opening_batch, group_index)?;
    let opening_width = relation_geometry
        .group_opening_geometry(group_index)?
        .physical_coefficient_width();
    let d_subcolumns = opening_width
        .checked_div(group_dims.d_d())
        .filter(|count| *count > 0 && opening_width.is_multiple_of(group_dims.d_d()))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("opening width does not factor the D role".into())
        })?;
    let num_claims = opening_batch.group_layout(group_index)?.num_polynomials();
    num_claims
        .checked_mul(group_lp.num_live_blocks())
        .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
        .and_then(|n| n.checked_mul(d_subcolumns))
        .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))
}

fn relation_d_column_ranges(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    relation_geometry: &RelationWitnessGeometry,
) -> Result<Vec<Range<usize>>, AkitaError> {
    let mut cursor = 0usize;
    let mut seen = vec![false; opening_batch.num_groups()];
    let mut ranges = vec![0..0; opening_batch.num_groups()];
    for group_id in opening_batch.root_group_order()? {
        let slot = seen
            .get_mut(group_id)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D group id out of range".into()))?;
        if std::mem::replace(slot, true) {
            return Err(AkitaError::InvalidSetup(
                "setup D group id appears more than once".into(),
            ));
        }
        let width = relation_d_group_width(lp, opening_batch, relation_geometry, group_id)?;
        let end = cursor
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
        ranges[group_id] = cursor..end;
        cursor = end;
    }
    if seen.iter().any(|present| !present) {
        return Err(AkitaError::InvalidSetup(
            "setup D group ids are not contiguous".into(),
        ));
    }
    Ok(ranges)
}

fn matching_row_range(
    row_families: &[RelationRowFamily],
    mut matches: impl FnMut(&RelationRowFamily) -> bool,
) -> Result<Range<usize>, AkitaError> {
    let mut matched = row_families
        .iter()
        .enumerate()
        .filter_map(|(row, family)| matches(family).then_some(row));
    let start = matched.next().ok_or(AkitaError::InvalidProof)?;
    let mut end = start + 1;
    for row in matched {
        if row != end {
            return Err(AkitaError::InvalidSetup(
                "relation row family is not contiguous".into(),
            ));
        }
        end += 1;
    }
    Ok(start..end)
}

/// Compile the complete A-role relation as direct reduced-ring Stage-2 weights.
///
/// For one setup column or sparse fold challenge `a`, source lane `j` stores
/// `eval(a * X^j mod (X^D + 1), alpha)`. This replaces the ordinary-product
/// A quotient rows for both `challenge * T` and `A * Z` without materializing
/// a dense witness-sized weight table.
pub(crate) fn build_negacyclic_setup_linear_terms<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    lp: &CommittedGroupParams,
    tau1: &[E],
    relation_plan: &RelationRangeImagePlan,
) -> Result<NegacyclicSetupLinearTerms<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F>,
{
    let opening_batch = instance.opening_batch();
    let relation_geometry =
        RelationWitnessGeometry::for_level(lp, opening_batch, instance.extension_degree())?;
    if relation_plan.relation_witness_geometry() != &relation_geometry {
        return Err(AkitaError::InvalidSetup(
            "negacyclic setup terms disagree with the relation geometry".into(),
        ));
    }
    let witness_layout = relation_plan.witness_layout();
    let coefficient_count = relation_plan
        .relation_address_geometry()
        .relation_coefficient_block_len();
    if coefficient_count == 0
        || !witness_layout
            .live_coeff_len()
            .is_multiple_of(coefficient_count)
    {
        return Err(AkitaError::InvalidSetup(
            "negacyclic setup terms require an aligned coefficient block".into(),
        ));
    }
    let live_lane_count = witness_layout.live_coeff_len() / coefficient_count;
    let row_families = relation_geometry.rhs_layout().row_families()?;
    let eq_tau1 = SplitEqEvals::new(tau1)?;
    if eq_tau1.len() < row_families.len() {
        return Err(AkitaError::InvalidProof);
    }

    let mut sources = Vec::with_capacity(opening_batch.num_groups());
    let mut segments = Vec::new();
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
        let group_dims = lp.group_role_dims_geometry(opening_batch, group_index)?;
        let ring_dimension = group_dims.d_a();
        let relation_ratio = ring_dimension
            .checked_div(coefficient_count)
            .filter(|ratio| *ratio > 0 && ring_dimension.is_multiple_of(coefficient_count))
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "A-role ring dimension does not factor the Stage-2 coefficient block".into(),
                )
            })?;
        let n_a = group_lp.a_rows_len();
        let inner_width = group_lp.a_col_len();
        let a_range = matching_row_range(
            &row_families,
            |family| matches!(family, RelationRowFamily::Inner { group_index: group, .. } if *group == group_index),
        )?;
        if a_range.len() != n_a {
            return Err(AkitaError::InvalidProof);
        }
        let row_weights = a_range
            .map(|row| eq_tau1.eval_at(row))
            .collect::<Result<Vec<_>, _>>()?;
        setup
            .shared_matrix
            .ring_view_dyn(n_a, inner_width, ring_dimension)?;
        let source_index = sources.len();
        sources.push(DirectLinearSource::ReducedSetup {
            ring_dimension,
            row_count: n_a,
            column_count: inner_width,
            row_weights: row_weights.clone(),
            alpha,
        });

        let depth_witness = group_lp.num_digits_inner();
        let depth_fold = group_lp.num_digits_fold();
        let num_positions = group_lp.num_positions_per_block();
        if inner_width
            != num_positions
                .checked_mul(depth_witness)
                .ok_or_else(|| AkitaError::InvalidSetup("A-role inner width overflow".into()))?
        {
            return Err(AkitaError::InvalidSetup(
                "A-role inner width disagrees with witness layout".into(),
            ));
        }
        let target_lane_stride = depth_fold
            .checked_mul(relation_ratio)
            .ok_or_else(|| AkitaError::InvalidSetup("negacyclic target stride overflow".into()))?;
        let fold_gadget = gadget_row_scalars::<F>(depth_fold, group_lp.log_basis_open())
            .into_iter()
            .map(E::lift_base)
            .collect::<Vec<_>>();
        for unit in witness_layout.units_for_group(group_index)? {
            for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                for role_subcolumn in 0..relation_ratio {
                    let physical_start = unit.z_coefficient_index(
                        ring_dimension,
                        num_positions,
                        depth_witness,
                        depth_fold,
                        0,
                        0,
                        fold_digit,
                        role_subcolumn * coefficient_count,
                    )?;
                    if !physical_start.is_multiple_of(coefficient_count) {
                        return Err(AkitaError::InvalidSetup(
                            "negacyclic target is not coefficient-block aligned".into(),
                        ));
                    }
                    segments.push(NegacyclicSetupLinearSegment {
                        factor: -fold,
                        source_index,
                        target_lane_start: physical_start / coefficient_count,
                        target_lane_stride,
                        source_lane_start: role_subcolumn,
                        source_lane_stride: relation_ratio,
                        lane_count: inner_width,
                    });
                }
            }
        }

        let challenges = instance.group_ambient_a_challenges(group_index)?;
        let num_claims = opening_batch.group_layout(group_index)?.num_polynomials();
        let num_live_blocks = group_lp.num_live_blocks();
        let challenge_count = num_claims
            .checked_mul(num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("A-role challenge count overflow".into()))?;
        if challenges.len() != challenge_count {
            return Err(AkitaError::InvalidProof);
        }
        let challenge_source_index = sources.len();
        let mut term_offsets = Vec::with_capacity(challenge_count + 1);
        let mut positions = Vec::new();
        let mut coefficients = Vec::new();
        term_offsets.push(0);
        dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            ring_dimension,
            |D_A| {
                for challenge in challenges.as_slice() {
                    challenge.validate::<D_A>()?;
                    positions.extend(challenge.positions.iter().copied());
                    coefficients.extend(challenge.coeffs.iter().copied());
                    term_offsets.push(u32::try_from(positions.len()).map_err(|_| {
                        AkitaError::InvalidSetup(
                            "A-role sparse challenge support does not fit u32".into(),
                        )
                    })?);
                }
                Ok::<(), AkitaError>(())
            }
        )?;
        sources.push(DirectLinearSource::ReducedSparse(
            DirectSparseLinearSource {
                ring_dimension,
                challenge_count,
                term_offsets,
                positions,
                coefficients,
                alpha,
            },
        ));

        let outer_dimension = group_dims.d_b();
        let role_subcolumns = ring_dimension
            .checked_div(outer_dimension)
            .filter(|count| *count > 0 && ring_dimension.is_multiple_of(outer_dimension))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("A-role T projection dimensions are malformed".into())
            })?;
        let outer_relation_ratio = outer_dimension
            .checked_div(coefficient_count)
            .filter(|ratio| *ratio > 0 && outer_dimension.is_multiple_of(coefficient_count))
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "A-role T dimension does not factor the Stage-2 coefficient block".into(),
                )
            })?;
        let depth_commit = group_lp.num_digits_outer();
        let commit_gadget = gadget_row_scalars::<F>(depth_commit, group_lp.log_basis_outer())
            .into_iter()
            .map(E::lift_base)
            .collect::<Vec<_>>();
        let target_lane_stride = n_a
            .checked_mul(depth_commit)
            .and_then(|stride| stride.checked_mul(relation_ratio))
            .ok_or_else(|| AkitaError::InvalidSetup("A-role T stride overflow".into()))?;
        for unit in witness_layout.units_for_group(group_index)? {
            if unit.num_live_blocks() == 0 {
                continue;
            }
            for claim in 0..num_claims {
                let first_challenge = claim
                    .checked_mul(num_live_blocks)
                    .and_then(|base| base.checked_add(unit.global_block_start()))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("A-role challenge index overflow".into())
                    })?;
                for (a_row, &row_weight) in row_weights.iter().enumerate() {
                    for (digit, &digit_weight) in commit_gadget.iter().enumerate() {
                        for role_subcolumn in 0..role_subcolumns {
                            for role_block in 0..outer_relation_ratio {
                                let physical_start = unit.t_coefficient_index(
                                    ring_dimension,
                                    outer_dimension,
                                    num_claims,
                                    n_a,
                                    depth_commit,
                                    claim,
                                    unit.global_block_start(),
                                    a_row,
                                    role_subcolumn,
                                    digit,
                                    role_block * coefficient_count,
                                )?;
                                if !physical_start.is_multiple_of(coefficient_count) {
                                    return Err(AkitaError::InvalidSetup(
                                        "A-role T target is not coefficient-block aligned".into(),
                                    ));
                                }
                                let source_role_block = role_subcolumn
                                    .checked_mul(outer_relation_ratio)
                                    .and_then(|base| base.checked_add(role_block))
                                    .ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "A-role challenge source overflow".into(),
                                        )
                                    })?;
                                let source_lane_start = first_challenge
                                    .checked_mul(relation_ratio)
                                    .and_then(|base| base.checked_add(source_role_block))
                                    .ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "A-role challenge source overflow".into(),
                                        )
                                    })?;
                                segments.push(NegacyclicSetupLinearSegment {
                                    factor: row_weight * digit_weight,
                                    source_index: challenge_source_index,
                                    target_lane_start: physical_start / coefficient_count,
                                    target_lane_stride,
                                    source_lane_start,
                                    source_lane_stride: relation_ratio,
                                    lane_count: unit.num_live_blocks(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(NegacyclicSetupLinearTerms {
        sources,
        segments,
        live_lane_count,
        coefficient_count,
    })
}

/// Emit the complete checked relation semantics for one fold.
pub(super) type RelationWeightBuild<E> = (
    RelationWeightEvents<E>,
    OpeningFamily<(), CoefficientPackingBatchSemantics<E>>,
);

#[tracing::instrument(skip_all, name = "build_relation_weight_events")]
pub fn build_relation_weight_events<F, E>(
    inputs: RelationWeightEventInputs<'_, F, E>,
) -> Result<RelationWeightBuild<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + MulBaseUnreduced<F>,
{
    let RelationWeightEventInputs {
        setup,
        instance,
        alpha,
        level_params: lp,
        relation_row_point: tau1,
        claim_coefficients: gamma,
        opening_source_len,
        opening_ring_dim,
        relation_plan,
        opening_points,
    } = inputs;
    let opening_batch = instance.opening_batch();
    lp.witness_chunk.validate()?;
    if gamma.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let role_dims = instance.role_dims();
    if role_dims != lp.role_dims() {
        return Err(AkitaError::InvalidSetup(
            "relation instance and level role dimensions disagree".into(),
        ));
    }
    let d_a = role_dims.d_a();
    let d_b = role_dims.d_b();
    let d_d = role_dims.d_d();
    let alpha_pows_b = scalar_powers(alpha, d_b);
    let alpha_pows_d = scalar_powers(alpha, d_d);
    let relation_geometry =
        RelationWitnessGeometry::for_level(lp, opening_batch, instance.extension_degree())?;
    let packing_required = matches!(
        relation_geometry.group_opening_method(0)?,
        OpeningMethod::SubringCoefficientPacking { .. }
    );
    if packing_required != matches!(opening_points, OpeningFamily::SubringCoefficientPacking(_)) {
        return Err(AkitaError::InvalidSetup(
            "relation opening family disagrees with prepared points".into(),
        ));
    }
    let relation_rhs_layout = relation_geometry.rhs_layout();
    let row_families = relation_rhs_layout.row_families()?;
    let quotient_row_dims = row_families
        .iter()
        .map(|row| row.geometry().polynomial_modulus_dimension())
        .collect::<Vec<_>>();
    let rows = quotient_row_dims.len();
    if rows == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let mut additional_quotient_alpha_powers = Vec::new();
    for &row_dim in &quotient_row_dims {
        if row_dim != d_a
            && row_dim != d_b
            && row_dim != d_d
            && additional_quotient_alpha_powers
                .iter()
                .all(|(dimension, _): &(usize, Vec<E>)| *dimension != row_dim)
        {
            additional_quotient_alpha_powers.push((row_dim, scalar_powers(alpha, row_dim)));
        }
    }
    let eq_tau1 = SplitEqEvals::new(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }
    let n_d_active = lp.open_commit_matrix.output_rank();
    let levels = r_decomp_levels::<F>(lp.log_basis_open);
    let witness_layout = instance.segment_layout(lp, None)?;
    if witness_layout.r_rows().len() != rows || witness_layout.quotient_depth() != levels {
        return Err(AkitaError::InvalidSetup(
            "relation matrix dimensions disagree with witness layout".to_string(),
        ));
    }
    for (row, family) in witness_layout.r_rows().iter().zip(&row_families) {
        if family.requires_quotient_witness() != row.is_some()
            || row
                .as_ref()
                .is_some_and(|row| row.geometry() != family.geometry())
        {
            return Err(AkitaError::InvalidSetup(
                "relation quotient dimensions disagree with witness layout".into(),
            ));
        }
    }
    let live_witness_coeff_len = witness_layout.live_coeff_len();
    let physical_field_len = opening_source_len
        .checked_mul(opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening field length overflow".into()))?;
    if live_witness_coeff_len > physical_field_len {
        return Err(AkitaError::InvalidSize {
            expected: physical_field_len,
            actual: live_witness_coeff_len,
        });
    }
    let setup_matrix = match setup {
        RelationSetupSource::Matrix(setup) => Some(setup),
        RelationSetupSource::DeferredClaim => None,
    };
    let setup_is_deferred = setup_matrix.is_none();
    let d_column_ranges = if setup_matrix.is_some() {
        relation_d_column_ranges(lp, opening_batch, &relation_geometry)?
    } else {
        Vec::new()
    };
    let relation_coefficient_block_len = RelationAddressGeometry::for_relation(
        &relation_geometry,
        opening_ring_dim,
        live_witness_coeff_len,
    )?
    .relation_coefficient_block_len();
    if relation_plan.relation_witness_geometry() != &relation_geometry
        || relation_plan.witness_layout() != &witness_layout
        || relation_plan
            .relation_address_geometry()
            .relation_coefficient_block_len()
            != relation_coefficient_block_len
    {
        return Err(AkitaError::InvalidSetup(
            "relation plan disagrees with the current ring switch".into(),
        ));
    }
    let (coefficient_packing_events, opening_semantics) = match opening_points {
        OpeningFamily::SubringCoefficientPacking(prepared_points) => {
            let (events, batch) = prepare_coefficient_packing_batch_semantics(
                CoefficientPackingBatchSemanticInputs {
                    level_params: lp,
                    opening_batch,
                    relation_plan,
                    relation: instance,
                    prepared_points,
                    alpha,
                    tau1,
                    claim_coefficients: gamma,
                },
            )?;
            (events, OpeningFamily::SubringCoefficientPacking(batch))
        }
        OpeningFamily::EvaluationTrace(()) => (Vec::new(), OpeningFamily::EvaluationTrace(())),
    };
    let coefficient_packing_groups = match &opening_semantics {
        OpeningFamily::EvaluationTrace(()) => &[][..],
        OpeningFamily::SubringCoefficientPacking(batch) => batch.groups(),
    };
    let mut relation_events = RelationWeightEvents {
        events: Vec::new(),
        alpha_powers: scalar_powers(
            alpha,
            quotient_row_dims
                .iter()
                .copied()
                .max()
                .ok_or(AkitaError::InvalidProof)?,
        ),
        relation_coefficient_block_len,
        physical_field_len,
        setup_is_deferred,
    };
    let mut packing_semantics_by_group = vec![None; opening_batch.num_groups()];
    for semantics in coefficient_packing_groups {
        let group_index = semantics.group_index();
        let slot = packing_semantics_by_group
            .get_mut(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if slot.replace(semantics).is_some() {
            return Err(AkitaError::InvalidSetup(
                "packing relation group appears more than once".into(),
            ));
        }
        if semantics.stage2_terms().physical_field_len() != live_witness_coeff_len {
            return Err(AkitaError::InvalidSetup(
                "packing relation live domain disagrees with the current ring switch".into(),
            ));
        }
        if semantics.stage2_terms().relation_coefficient_block_len()
            != relation_coefficient_block_len
        {
            return Err(AkitaError::InvalidSetup(
                "packing relation coefficient block disagrees with the current ring switch".into(),
            ));
        }
    }
    relation_events.extend_events(coefficient_packing_events)?;
    let d_view = if let Some(setup) = setup_matrix {
        let d_physical_columns = d_column_ranges
            .iter()
            .map(|range| range.end)
            .max()
            .unwrap_or(0);
        let rank = lp.open_commit_matrix.output_rank();
        Some((&setup.shared_matrix, rank, d_physical_columns))
    } else {
        None
    };
    let d_family = match &d_view {
        Some((matrix, rows, cols)) => {
            let view = matrix.ring_view_dyn(*rows, *cols, d_d)?;
            Some(SetupRows {
                rows: (0..*rows)
                    .map(|row| view.row_flat(row))
                    .collect::<Result<Vec<_>, _>>()?,
                ring_d: d_d,
            })
        }
        None => None,
    };
    let d_start = row_families
        .iter()
        .position(|row| matches!(row, akita_types::RelationRowFamily::Opening { .. }))
        .ok_or(AkitaError::InvalidProof)?;
    for (group_index, &packing_semantics) in packing_semantics_by_group.iter().enumerate() {
        let e_setup_offset = if setup_matrix.is_some() {
            d_column_ranges
                .get(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .start
        } else {
            0
        };
        let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
        let group_dims = lp.group_role_dims_geometry(opening_batch, group_index)?;
        let group_d_a = group_dims.d_a();
        let group_d_b = group_dims.d_b();
        let group_d_d = group_dims.d_d();
        let (b_ratio, _) = SetupProjectionGeometry::native_role_subcolumn_counts(group_dims)?;
        let opening_width = relation_geometry
            .group_opening_geometry(group_index)?
            .physical_coefficient_width();
        let d_ratio = opening_width
            .checked_div(group_d_d)
            .filter(|count| *count > 0 && opening_width.is_multiple_of(group_d_d))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("opening width does not factor the D role".into())
            })?;
        let group_alpha_pows_a = scalar_powers(alpha, group_d_a);
        let group_alpha_pows_b = scalar_powers(alpha, group_d_b);
        let group_alpha_pows_d = scalar_powers(alpha, group_d_d);
        let group_layout = opening_batch.group_layout(group_index)?;
        let group_id = group_index;
        let units = witness_layout.units_for_group(group_id)?;
        let k_g = group_layout.num_polynomials();
        let opening_method = relation_geometry.group_opening_method(group_index)?;
        match (opening_method, packing_semantics) {
            (OpeningMethod::EvaluationTrace, None) => {}
            (OpeningMethod::SubringCoefficientPacking { .. }, Some(semantics))
                if semantics.geometry().a_ring_dimension() == group_d_a => {}
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "packing semantic groups do not match scheduled opening methods".into(),
                ));
            }
        }
        let ring_multiplier_point = matches!(opening_method, OpeningMethod::EvaluationTrace)
            .then(|| instance.group_ring_multiplier_point(group_index))
            .transpose()?;
        let challenges = instance.group_ambient_a_challenges(group_index)?;
        if ring_multiplier_point.is_some_and(|point| {
            point.position_len() != group_lp.num_positions_per_block()
                || point.fold_len() != group_lp.num_live_blocks()
        }) {
            return Err(AkitaError::InvalidInput(
                "relation matrix col eval multiplier layout mismatch".to_string(),
            ));
        }
        let total_blocks = k_g
            .checked_mul(group_lp.num_live_blocks())
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.len() != total_blocks {
            return Err(AkitaError::InvalidProof);
        }
        let depth_witness = group_lp.num_digits_inner();
        let depth_commit = group_lp.num_digits_outer();
        let depth_open = group_lp.num_digits_open();
        let depth_fold = group_lp.num_digits_fold();
        let log_basis_inner = group_lp.log_basis_inner();
        let log_basis_open = group_lp.log_basis_open();
        let n_a = group_lp.a_rows_len();
        let physical_n_b = group_lp.b_rows_len();
        let n_b = group_lp.logical_b_rows_len()?;
        let inner_width = group_lp.a_col_len();
        // Hoist per-group geometry into `Copy` locals so the parallel closures
        // below capture scalars instead of the `!Sync` `&dyn LevelParamsLike`.
        let num_live_blocks_g = group_lp.num_live_blocks();
        let num_positions_per_block_g = group_lp.num_positions_per_block();
        let slice_geometry = akita_types::CommitmentSliceGeometry::try_new(
            group_lp.outer_slice_count(),
            num_live_blocks_g,
            k_g,
            n_a,
            depth_commit,
            group_d_a,
            group_d_b,
        )?;
        let b_width = slice_geometry.physical_input_width();
        let b_family = if let Some(setup) = setup_matrix {
            let b_view = setup
                .shared_matrix
                .ring_view_dyn(physical_n_b, b_width, group_d_b)?;
            let b_family = SetupRows {
                rows: (0..physical_n_b)
                    .map(|row| b_view.row_flat(row))
                    .collect::<Result<Vec<_>, _>>()?,
                ring_d: group_d_b,
            };
            Some(b_family)
        } else {
            None
        };
        let b_range = matching_row_range(
            &row_families,
            |family| matches!(family, RelationRowFamily::Outer { group_index: group, .. } if *group == group_index),
        )?;
        let consistency_row = row_families
            .iter()
            .position(|family| {
                matches!(family, RelationRowFamily::Consistency { group_index: group, .. } if *group == group_index)
            })
            .ok_or(AkitaError::InvalidProof)?;
        let consistency_weight = eq_tau1.eval_at(consistency_row)?;
        if b_range.end > eq_tau1.len() || b_range.len() != n_b {
            return Err(AkitaError::InvalidProof);
        }
        let g_open: Vec<E> = gadget_row_scalars::<F>(depth_open, log_basis_open)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let witness_gadget: Vec<E> = gadget_row_scalars::<F>(depth_witness, log_basis_inner)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let fold_gadget: Vec<E> = gadget_row_scalars::<F>(depth_fold, log_basis_open)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let d_setup_start = e_setup_offset;
        let d_setup_len = total_blocks
            .checked_mul(d_ratio)
            .and_then(|len| len.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))?;
        let d_setup_end = d_setup_start
            .checked_add(d_setup_len)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D extent overflow".to_string()))?;
        let d_setup_accs = if let Some(d_family) = &d_family {
            let _span = tracing::info_span!("relation_weight_d_setup_columns").entered();
            let row_weights = (0..n_d_active)
                .map(|row| Ok((row, vec![eq_tau1.eval_at(d_start + row)?])))
                .filter_map(|result| match result {
                    Ok((_, weights)) if weights[0].is_zero() => None,
                    other => Some(other),
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            Some(evaluate_setup_columns(
                d_family,
                d_setup_start..d_setup_end,
                &row_weights,
                1,
                &group_alpha_pows_d,
            )?)
        } else {
            None
        };
        let b_setup_accs = if let Some(b_family) = &b_family {
            let _span = tracing::info_span!("relation_weight_b_setup_columns").entered();
            let slice_count = group_lp.outer_slice_count().get();
            let row_weights = (0..physical_n_b)
                .map(|row| {
                    let weights = (0..slice_count)
                        .map(|slice_index| {
                            let logical_row = slice_geometry
                                .logical_row_index(slice_index, row, physical_n_b)?
                                .checked_add(b_range.start)
                                .ok_or(AkitaError::InvalidProof)?;
                            eq_tau1.eval_at(logical_row)
                        })
                        .collect::<Result<Vec<_>, AkitaError>>()?;
                    Ok((row, weights))
                })
                .filter_map(|result| match result {
                    Ok((_, ref weights)) if weights.iter().all(|weight| weight.is_zero()) => None,
                    other => Some(other),
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            Some(evaluate_setup_columns(
                b_family,
                0..b_width,
                &row_weights,
                slice_count,
                &group_alpha_pows_b,
            )?)
        } else {
            None
        };

        for claim in 0..k_g {
            for global_block in 0..num_live_blocks_g {
                let unit = witness_layout.unit_for_block(group_id, global_block)?;
                let challenge_index = claim
                    .checked_mul(num_live_blocks_g)
                    .and_then(|base| base.checked_add(global_block))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("relation challenge index overflow".into())
                    })?;
                let challenge_alpha =
                    challenges.eval_at_pows::<F, E>(challenge_index, &group_alpha_pows_a)?;
                let (slice_index, slice_block) = slice_geometry.block_coordinates(global_block)?;
                for (digit, &opening_gadget) in g_open.iter().enumerate() {
                    for role_subcol in 0..d_ratio {
                        let physical_start = unit.e_coefficient_index(
                            group_d_d,
                            k_g,
                            depth_open,
                            claim,
                            global_block,
                            role_subcol,
                            digit,
                            0,
                        )?;
                        let logical_block = claim * num_live_blocks_g + global_block;
                        let d_phys_col = logical_block
                            .checked_mul(d_ratio)
                            .and_then(|base| base.checked_add(role_subcol))
                            .and_then(|base| base.checked_mul(depth_open))
                            .and_then(|base| base.checked_add(digit))
                            .and_then(|local| e_setup_offset.checked_add(local))
                            .ok_or(AkitaError::InvalidProof)?;
                        let consistency_acc = consistency_weight * challenge_alpha * opening_gadget;
                        let setup_acc = if let Some(weights) = d_setup_accs.as_ref() {
                            let local_col = d_phys_col
                                .checked_sub(d_setup_start)
                                .ok_or(AkitaError::InvalidProof)?;
                            weights.get(0, local_col)?
                        } else {
                            E::zero()
                        };
                        if matches!(opening_method, OpeningMethod::EvaluationTrace) {
                            relation_events.push(
                                physical_start,
                                group_d_d,
                                role_subcol * group_d_d,
                                consistency_acc,
                                RelationWeightContribution::Constraint,
                            )?;
                        }
                        if d_setup_accs.is_some() {
                            relation_events.push(
                                physical_start,
                                group_d_d,
                                0,
                                setup_acc,
                                RelationWeightContribution::SetupMatrix,
                            )?;
                        }
                    }
                }
                for a_idx in 0..n_a {
                    for digit in 0..depth_commit {
                        let block_claim = slice_geometry
                            .max_blocks_per_slice()
                            .checked_mul(claim)
                            .and_then(|base| base.checked_add(slice_block))
                            .ok_or(AkitaError::InvalidProof)?;
                        let row_block_claim = n_a
                            .checked_mul(block_claim)
                            .and_then(|base| base.checked_add(a_idx))
                            .ok_or(AkitaError::InvalidProof)?;
                        for role_subcol in 0..b_ratio {
                            let local_col = row_block_claim
                                .checked_mul(b_ratio)
                                .and_then(|base| base.checked_add(role_subcol))
                                .and_then(|base| base.checked_mul(depth_commit))
                                .and_then(|base| base.checked_add(digit))
                                .ok_or(AkitaError::InvalidProof)?;
                            let physical_start = unit.t_coefficient_index(
                                group_d_a,
                                group_d_b,
                                k_g,
                                n_a,
                                depth_commit,
                                claim,
                                global_block,
                                a_idx,
                                role_subcol,
                                digit,
                                0,
                            )?;
                            let b_acc = if let Some(slice_weights) = b_setup_accs.as_ref() {
                                slice_weights.get(slice_index, local_col)?
                            } else {
                                E::zero()
                            };
                            if b_setup_accs.is_some() {
                                relation_events.push(
                                    physical_start,
                                    group_d_b,
                                    0,
                                    b_acc,
                                    RelationWeightContribution::SetupMatrix,
                                )?;
                            }
                        }
                    }
                }
            }
        }
        // These setup-column accumulators can be large and are not used by
        // the z-hat phase below. Release them at the named phase boundary.
        drop(d_setup_accs);
        drop(b_setup_accs);

        let z_constraint_bases = cfg_into_iter!(0..inner_width)
            .map(|k| {
                let block_idx = k / depth_witness;
                let digit_idx = k % depth_witness;
                let constraint = if let Some(point) = ring_multiplier_point {
                    consistency_weight
                        * point.eval_position_at::<E>(block_idx, &group_alpha_pows_a)?
                        * witness_gadget[digit_idx]
                } else {
                    E::zero()
                };
                Ok(constraint)
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        for unit in units {
            for position in 0..num_positions_per_block_g {
                for commit_digit in 0..depth_witness {
                    for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                        let phys_k = position * depth_witness + commit_digit;
                        let physical_start = unit.z_coefficient_index(
                            group_d_a,
                            num_positions_per_block_g,
                            depth_witness,
                            depth_fold,
                            position,
                            commit_digit,
                            fold_digit,
                            0,
                        )?;
                        if matches!(opening_method, OpeningMethod::EvaluationTrace) {
                            relation_events.push_native_ring(
                                physical_start,
                                group_d_a,
                                -(z_constraint_bases[phys_k] * fold),
                                RelationWeightContribution::Constraint,
                            )?;
                        }
                    }
                }
            }
        }
    }
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, lp.log_basis_open)
        .into_iter()
        .map(E::lift_base)
        .collect();
    for (row, &row_dim) in quotient_row_dims.iter().enumerate() {
        if witness_layout.r_rows().get(row).is_none_or(Option::is_none) {
            continue;
        }
        if matches!(
            row_families[row],
            RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
        ) {
            continue;
        }
        if matches!(
            row_families[row],
            RelationRowFamily::Consistency {
                opening_method: OpeningMethod::SubringCoefficientPacking { .. },
                ..
            }
        ) {
            continue;
        }
        let eq_weight = eq_tau1.eval_at(row)?;
        let row_alpha_pows = if row_dim == d_a {
            relation_events.alpha_powers.as_slice()
        } else if row_dim == d_b {
            alpha_pows_b.as_slice()
        } else if row_dim == d_d {
            alpha_pows_d.as_slice()
        } else {
            additional_quotient_alpha_powers
                .iter()
                .find_map(|(dimension, powers)| {
                    (*dimension == row_dim).then_some(powers.as_slice())
                })
                .ok_or(AkitaError::InvalidProof)?
        };
        let row_denom = row_alpha_pows[row_dim - 1] * alpha + E::one();
        for (digit, gadget) in r_gadget.iter().enumerate() {
            let physical_start = witness_layout.r_coefficient_index(row, digit, 0, 0)?;
            relation_events.push_native_ring(
                physical_start,
                row_dim,
                -(eq_weight * row_denom * *gadget),
                RelationWeightContribution::Constraint,
            )?;
        }
    }
    Ok((relation_events, opening_semantics))
}

#[cfg(test)]
#[path = "relation_weights_tests.rs"]
mod tests;
