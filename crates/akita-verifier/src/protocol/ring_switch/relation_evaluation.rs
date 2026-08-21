//! Succinct prepared relation evaluation for every role geometry.
//!
//! The common low alpha coordinates are factored once. The setup plan then
//! selects contiguous q=1 or projected-lane q>1 kernels without changing the
//! verifier formula or control flow.

use super::{prepared_relation_point::PreparedRelationPoint, RelationMatrixEvaluator};
use akita_algebra::{
    offset_eq::OffsetEqWindow,
    poly::multilinear_eval,
    ring::{eval_negacyclic_shift_sequence_into, CyclotomicRing},
};
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, MulBaseUnreduced,
};
use akita_types::{
    dispatch_for_field, gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, FpExtEncoding,
    RelationAddressGeometry, RelationRowFamily, RelationWitnessGeometry,
};

pub(super) fn evaluate_relation_at_point<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    point: &[E],
    setup: &AkitaExpandedSetup<F>,
    alpha: E,
    deferred_setup_claim: Option<E>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F> + MulBaseUnreduced<F>,
{
    let context = evaluator
        .flat_context
        .as_ref()
        .ok_or(AkitaError::InvalidProof)?;
    let relation_geometry = RelationWitnessGeometry::for_level(
        &context.level_params,
        &context.opening_batch,
        context.extension_degree,
    )?;
    let row_families = relation_geometry.rhs_layout().row_families()?;
    let quotient_row_dims = row_families
        .iter()
        .filter(|family| {
            !matches!(
                family,
                RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
            )
        })
        .map(|family| family.geometry().polynomial_modulus_dimension())
        .collect::<Vec<_>>();
    let prepared_point = PreparedRelationPoint::new(
        point,
        alpha,
        evaluator.relation_address_geometry,
        &quotient_row_dims,
    )?;
    if evaluator.relation_address_geometry != prepared_point.relation_address_geometry() {
        return Err(AkitaError::InvalidProof);
    }
    // The setup projection and flat relation point use the same common base of
    // the current commitment roles. Outgoing witness packaging affects only
    // the checked flat live length. The same plan therefore owns the mixed
    // E/T/Z contraction, direct setup scan, and deferred Stage-3 geometry.
    let fold_gadget = evaluator.setup_contribution_fold_gadget::<F>()?;
    let plan = {
        let _span = tracing::info_span!("relation_setup_plan").entered();
        let fold_gadget = fold_gadget.as_deref().unwrap_or(&[]);
        evaluator.setup_contribution_plan::<F>(
            prepared_point.relation_address().clone(),
            (!fold_gadget.is_empty()).then_some(fold_gadget),
        )?
    };
    let mut structured_evaluation = E::zero();
    {
        let _span = tracing::info_span!("relation_structured_groups").entered();
        for group in &evaluator.groups {
            structured_evaluation += plan
                .evaluate_structured_group::<F>(
                    group.group_id,
                    &group.c_alphas,
                    &group.opening_a_evals,
                    alpha,
                )
                .map_err(|error| {
                    AkitaError::InvalidInput(format!(
                        "relation group {} contraction failed: {error:?}",
                        group.group_id
                    ))
                })?;
        }
    }
    let coefficient_bits = evaluator
        .relation_address_geometry
        .relation_coefficient_variable_count();
    let coefficient_point = point
        .get(..coefficient_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let reduced_challenge_evaluation =
        evaluate_reduced_challenge_t::<F, E>(evaluator, &prepared_point, coefficient_point)?;

    let setup_evaluation = if let Some(claim) = deferred_setup_claim {
        claim
    } else {
        let _span =
            tracing::info_span!("relation_setup_scan", required = plan.required()).entered();
        plan.evaluate_setup_product_direct::<F>(setup, alpha, coefficient_point)?
    };
    let quotient_evaluation =
        evaluate_quotient_tail::<F, E>(evaluator, &prepared_point, &row_families).map_err(
            |error| AkitaError::InvalidInput(format!("relation quotient failed: {error:?}")),
        )?;

    let ordinary_evaluation =
        prepared_point.common_alpha_evaluation() * (structured_evaluation + quotient_evaluation);
    if deferred_setup_claim.is_some() {
        evaluator.cache_setup_contribution_plan(prepared_point.address_point(), plan)?;
    }
    Ok(ordinary_evaluation + reduced_challenge_evaluation + setup_evaluation)
}

fn evaluate_reduced_challenge_t<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    prepared_point: &PreparedRelationPoint<E>,
    coefficient_point: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let context = evaluator
        .flat_context
        .as_ref()
        .ok_or(AkitaError::InvalidProof)?;
    let coefficient_count = evaluator
        .relation_address_geometry
        .relation_coefficient_block_len();
    let coefficient_bits =
        u32::try_from(coefficient_point.len()).map_err(|_| AkitaError::InvalidProof)?;
    if coefficient_count != 1usize.checked_shl(coefficient_bits).unwrap_or(0) {
        return Err(AkitaError::InvalidProof);
    }
    let equality = prepared_point.relation_address().equality_window();
    let mut evaluation = E::zero();
    for group in &evaluator.groups {
        let group_params = context
            .level_params
            .group_params_geometry(&context.opening_batch, group.group_id)?;
        let role_dims = context
            .level_params
            .group_role_dims_geometry(&context.opening_batch, group.group_id)?;
        let num_live_blocks = group_params.num_live_blocks();
        let n_a = group_params.a_rows_len();
        let depth_commit = group_params.num_digits_outer();
        let a_row_end = group
            .a_row_start
            .checked_add(n_a)
            .ok_or(AkitaError::InvalidProof)?;
        let row_weights = evaluator
            .eq_tau1
            .get(group.a_row_start..a_row_end)
            .ok_or(AkitaError::InvalidProof)?;
        let commit_gadget = gadget_row_scalars::<F>(depth_commit, group_params.log_basis_outer())
            .into_iter()
            .map(E::lift_base)
            .collect::<Vec<_>>();
        if group.ambient_challenges.num_claims() != group.num_claims
            || group.ambient_challenges.num_live_blocks_per_claim() != num_live_blocks
        {
            return Err(AkitaError::InvalidProof);
        }
        dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            role_dims.d_a(),
            |D_A| {
                let outer_dimension = role_dims.d_b();
                let role_subcolumns = D_A
                    .checked_div(outer_dimension)
                    .filter(|count| *count > 0 && D_A.is_multiple_of(outer_dimension))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("reduced A-role T projection is malformed".into())
                    })?;
                let outer_relation_ratio = outer_dimension
                    .checked_div(coefficient_count)
                    .filter(|count| *count > 0 && outer_dimension.is_multiple_of(coefficient_count))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "reduced A-role T coefficient block is malformed".into(),
                        )
                    })?;
                let projected_relation_ratio = role_subcolumns
                    .checked_mul(outer_relation_ratio)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("reduced A-role relation ratio overflow".into())
                    })?;
                let relation_ratio = D_A
                    .checked_div(coefficient_count)
                    .filter(|count| *count == projected_relation_ratio)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "reduced A-role relation projection is malformed".into(),
                        )
                    })?;
                let mut shifts = vec![E::zero(); D_A];
                for unit in context.witness_layout.units_for_group(group.group_id)? {
                    for claim in 0..group.num_claims {
                        for global_block in unit.global_block_range() {
                            let challenge_index = claim
                                .checked_mul(num_live_blocks)
                                .and_then(|base| base.checked_add(global_block))
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "reduced A-role challenge index overflow".into(),
                                    )
                                })?;
                            let challenge = group
                                .ambient_challenges
                                .as_slice()
                                .get(challenge_index)
                                .ok_or(AkitaError::InvalidProof)?;
                            challenge.validate::<D_A>()?;
                            let mut challenge_ring = CyclotomicRing::<F, D_A>::zero();
                            for (&position, &coefficient) in
                                challenge.positions.iter().zip(&challenge.coeffs)
                            {
                                let position = usize::try_from(position)
                                    .map_err(|_| AkitaError::InvalidProof)?;
                                let slot = challenge_ring
                                    .coefficients_mut()
                                    .get_mut(position)
                                    .ok_or(AkitaError::InvalidProof)?;
                                *slot += F::from_i64(i64::from(coefficient));
                            }
                            eval_negacyclic_shift_sequence_into(
                                &challenge_ring,
                                prepared_point.alpha(),
                                &mut shifts,
                            );
                            for source_block in 0..relation_ratio {
                                let source_start = source_block
                                    .checked_mul(coefficient_count)
                                    .ok_or(AkitaError::InvalidProof)?;
                                let source_end = source_start
                                    .checked_add(coefficient_count)
                                    .ok_or(AkitaError::InvalidProof)?;
                                let source = shifts
                                    .get(source_start..source_end)
                                    .ok_or(AkitaError::InvalidProof)?;
                                let source_evaluation =
                                    multilinear_eval(source, coefficient_point)?;
                                let role_subcolumn = source_block / outer_relation_ratio;
                                let role_block = source_block % outer_relation_ratio;
                                for (a_row, &row_weight) in row_weights.iter().enumerate() {
                                    for (digit, &digit_weight) in commit_gadget.iter().enumerate() {
                                        let physical = unit.t_coefficient_index(
                                            D_A,
                                            outer_dimension,
                                            group.num_claims,
                                            n_a,
                                            depth_commit,
                                            claim,
                                            global_block,
                                            a_row,
                                            role_subcolumn,
                                            digit,
                                            role_block * coefficient_count,
                                        )?;
                                        let lane = canonical_relation_lane_index(
                                            evaluator.relation_address_geometry,
                                            physical,
                                        )?;
                                        evaluation += source_evaluation
                                            * row_weight
                                            * digit_weight
                                            * equality.eval(lane);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok::<(), AkitaError>(())
            }
        )?;
    }
    Ok(evaluation)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_quotient_tail<F, E>(
    evaluator: &RelationMatrixEvaluator<E>,
    prepared_point: &PreparedRelationPoint<E>,
    row_families: &[RelationRowFamily],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let context = evaluator
        .flat_context
        .as_ref()
        .ok_or(AkitaError::InvalidProof)?;
    let rows = row_families.len();
    if rows
        != context
            .level_params
            .relation_matrix_row_count(context.opening_batch.num_groups())?
    {
        return Err(AkitaError::InvalidSetup(
            "relation quotient row dimensions disagree with the matrix layout".into(),
        ));
    }
    let levels = r_decomp_levels::<F>(evaluator.log_basis);
    let quotient_gadget = gadget_row_scalars::<F>(levels, evaluator.log_basis);
    let mut evaluation = E::zero();
    for (row, family) in row_families.iter().enumerate() {
        if matches!(
            family,
            RelationRowFamily::CompressionF { .. }
                | RelationRowFamily::CompressionH { .. }
                | RelationRowFamily::Consistency {
                    opening_method: akita_types::OpeningMethod::SubringCoefficientPacking { .. },
                    ..
                }
        ) || !family.requires_quotient_witness()
        {
            continue;
        }
        let row_dimension = family.geometry().polynomial_modulus_dimension();
        let role_factors = prepared_point.for_dimension(row_dimension)?;
        let denominator = role_factors
            .powers
            .last()
            .copied()
            .ok_or(AkitaError::InvalidProof)?
            * prepared_point.alpha()
            + E::one();
        let row_weight = evaluator
            .eq_tau1
            .get(row)
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        let mut row_evaluation = E::zero();
        for (digit, &gadget) in quotient_gadget.iter().enumerate() {
            let physical_coefficient = context
                .witness_layout
                .r_coefficient_index(row, digit, 0, 0)?;
            let lane_start = canonical_relation_lane_index(
                evaluator.relation_address_geometry,
                physical_coefficient,
            )?;
            let lane_evaluation = evaluate_lane_segment(
                prepared_point.relation_address().equality_window(),
                lane_start,
                &role_factors.lane_powers,
            )?;
            row_evaluation += lane_evaluation.mul_base(gadget);
        }
        evaluation -= row_evaluation * row_weight * denominator;
    }
    Ok(evaluation)
}

fn evaluate_lane_segment<E: FieldCore>(
    equality_window: &OffsetEqWindow<E>,
    lane_start: usize,
    lane_alpha_powers: &[E],
) -> Result<E, AkitaError> {
    lane_alpha_powers
        .iter()
        .enumerate()
        .try_fold(E::zero(), |sum, (lane, &alpha_power)| {
            let index = lane_start
                .checked_add(lane)
                .ok_or_else(|| AkitaError::InvalidSetup("relation lane address overflow".into()))?;
            Ok(sum + equality_window.eval(index) * alpha_power)
        })
}

fn canonical_relation_lane_index(
    geometry: RelationAddressGeometry,
    physical_coefficient: usize,
) -> Result<usize, AkitaError> {
    let coeff_count = geometry.relation_coefficient_block_len();
    if physical_coefficient >= geometry.digit_witness_domain().live_len()
        || !physical_coefficient.is_multiple_of(coeff_count)
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(physical_coefficient / coeff_count)
}
