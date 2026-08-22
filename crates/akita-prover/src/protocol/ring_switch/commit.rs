use super::*;
use crate::api::commitment::commit_outer_slices;
use crate::compute::compression::{execute_compression_chains, CompressionExecutionInput};
use crate::compute::{CommitInnerPlan, OperationCtx, RuntimeCommitBackendFor};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::{CommitInnerWitness, SuffixWitnessView};
use akita_types::{
    dispatch_for_field, CommittedSourceEncoding, CompressionChainPlan, TerminalCommittedGroupParams,
};

/// Public state bound for the witness produced by one intermediate fold.
pub enum NextWitnessState<F: FieldCore> {
    /// Ordinary recursive edge, bound by the terminal compressed payload.
    OuterPayload(RingVec<F>),
    /// Last recursive edge, bound directly by the canonical inner `t` state.
    TerminalInnerState,
}

/// Result of preparing the next logical recursive witness and its public state.
pub struct NextWitnessStateOutput<F: FieldCore> {
    /// Physical witness representation when extension packing changes the logical witness.
    pub witness: Option<RecursiveWitnessFlat>,
    /// Transcript-bound public state for the next level.
    pub binding: NextWitnessState<F>,
    /// Prover hint for opening the physical next-level witness.
    pub hint: AkitaCommitmentHint<F>,
}

pub(crate) struct RecursiveCommitPrefix<F: FieldCore> {
    coeff_len: usize,
    known_balanced_log_basis: u32,
    inner: CommitInnerWitness<F>,
}

impl<F: FieldCore> RecursiveCommitPrefix<F> {
    pub(crate) const fn coeff_len(&self) -> usize {
        self.coeff_len
    }
}

pub(crate) fn merge_recursive_commit_prefixes<F: FieldCore>(
    prefixes: Vec<RecursiveCommitPrefix<F>>,
) -> Result<RecursiveCommitPrefix<F>, AkitaError> {
    let mut prefixes = prefixes.into_iter();
    let first = prefixes.next().ok_or_else(|| {
        AkitaError::InvalidInput("recursive commitment prefix list is empty".into())
    })?;
    let ring_dim = first.inner.ring_dim();
    let known_balanced_log_basis = first.known_balanced_log_basis;
    let mut coeff_len = first.coeff_len;
    let mut inner_coefficients = first.inner.into_inner_rows().into_coeffs();
    for prefix in prefixes {
        if prefix.known_balanced_log_basis != known_balanced_log_basis
            || prefix.inner.ring_dim() != ring_dim
        {
            return Err(AkitaError::InvalidSetup(
                "recursive commitment prefixes use inconsistent geometry".into(),
            ));
        }
        coeff_len = coeff_len.checked_add(prefix.coeff_len).ok_or_else(|| {
            AkitaError::InvalidSetup("recursive commitment prefix length overflow".into())
        })?;
        inner_coefficients.extend(prefix.inner.into_inner_rows().into_coeffs());
    }
    Ok(RecursiveCommitPrefix {
        coeff_len,
        known_balanced_log_basis,
        inner: CommitInnerWitness {
            inner_rows: RingVec::from_coeffs_with_ring_dim(inner_coefficients, ring_dim)?,
        },
    })
}

pub(crate) fn prepare_recursive_commit_prefix<Cfg, B>(
    commit_params: &CommittedGroupParams,
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    commit_ctx: &OperationCtx<'_, Cfg::Field, B>,
    digits: &[i8],
    known_balanced_log_basis: u32,
) -> Result<RecursiveCommitPrefix<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField,
    B: RuntimeCommitBackendFor<Cfg::Field, RecursiveWitnessFlat>,
{
    if commit_params.source_encoding != CommittedSourceEncoding::CanonicalCoefficientTable {
        return Err(AkitaError::InvalidSetup(
            "pipelined recursive commitment requires a canonical coefficient source".into(),
        ));
    }
    let backend = commit_ctx.backend();
    let prepared = commit_ctx.prepared();
    backend.validate_prepared_setup(prepared, expanded)?;
    let plan = CommitInnerPlan::from_level(commit_params);
    let d_a = commit_params.role_dims().d_a();
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Cfg::Field,
        d_a,
        |D_A| {
            let block_coeff_len = plan
                .num_positions_per_block
                .checked_mul(D_A)
                .ok_or_else(|| AkitaError::InvalidSetup("commit block width overflow".into()))?;
            if digits.is_empty() || !digits.len().is_multiple_of(block_coeff_len) {
                return Err(AkitaError::InvalidSetup(
                    "pipelined commitment prefix must contain complete source blocks".into(),
                ));
            }
            let block_count = digits.len() / block_coeff_len;
            let view = SuffixWitnessView::<Cfg::Field, D_A>::from_balanced_i8_digits(
                digits,
                known_balanced_log_basis,
            )?;
            let mut inner_group = backend.commit_inner_group(prepared, vec![view], plan)?;
            let inner = inner_group.pop().ok_or(AkitaError::InvalidProof)?;
            if !inner_group.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            validate_commit_inner_shape::<Cfg::Field, D_A>(&inner, block_count, plan.n_a)?;
            Ok(RecursiveCommitPrefix {
                coeff_len: digits.len(),
                known_balanced_log_basis,
                inner,
            })
        }
    )
}

/// Commit the next recursive witness under config `Cfg`.
///
/// The commitment ring dimension is schedule-owned (`commit_params.ring_dimension`).
/// This function warms the target NTT slot on the caller's D-free prepared setup,
/// dispatches locally to the typed commit kernel, and returns D-free protocol
/// storage.
///
/// # Errors
///
/// Returns an error if layout selection, commitment, cache preparation, or
/// D-erased hint construction fails.
#[inline(never)]
pub fn commit_w<Cfg, B>(
    commit_params: &CommittedGroupParams,
    fold_level: usize,
    expanded: &std::sync::Arc<AkitaExpandedSetup<Cfg::Field>>,
    commit_ctx: &OperationCtx<'_, Cfg::Field, B>,
    logical_w: &RecursiveWitnessFlat,
) -> Result<NextWitnessStateOutput<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + HalvingField,
    B: RuntimeCommitBackendFor<Cfg::Field, RecursiveWitnessFlat>,
{
    commit_w_with_prefix::<Cfg, B>(
        commit_params,
        fold_level,
        expanded,
        commit_ctx,
        logical_w,
        None,
    )
}

pub(crate) fn commit_w_with_prefix<Cfg, B>(
    commit_params: &CommittedGroupParams,
    fold_level: usize,
    expanded: &std::sync::Arc<AkitaExpandedSetup<Cfg::Field>>,
    commit_ctx: &OperationCtx<'_, Cfg::Field, B>,
    logical_w: &RecursiveWitnessFlat,
    commit_prefix: Option<RecursiveCommitPrefix<Cfg::Field>>,
) -> Result<NextWitnessStateOutput<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + HalvingField,
    B: RuntimeCommitBackendFor<Cfg::Field, RecursiveWitnessFlat>,
{
    let dims = commit_params.role_dims();
    let backend = commit_ctx.backend();
    let prepared = commit_ctx.prepared();
    backend.validate_prepared_setup(prepared, expanded.as_ref())?;
    let slice_geometry = validate_commit_level_params::<Cfg::Field>(
        commit_params,
        expanded.as_ref(),
        fold_level,
        1,
    )?;

    let (packed_witness, inner_rows, commitment, outer_relation_quotients, compression_witness) = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Cfg::Field,
        dims.d_a(),
        |D_A| {
            let packed_witness = match commit_params.source_encoding {
                CommittedSourceEncoding::CanonicalCoefficientTable => None,
                CommittedSourceEncoding::TensorSubfieldProjection { extension_degree } => {
                    if extension_degree != <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE {
                        return Err(AkitaError::InvalidSetup(
                            "recursive tensor source encoding does not match the protocol extension degree"
                                .into(),
                        ));
                    }
                    Some(tensor_pack_recursive_witness::<
                        Cfg::Field,
                        Cfg::ExtField,
                        D_A,
                    >(logical_w)?)
                }
            };
            let w = packed_witness.as_ref().unwrap_or(logical_w);
            let committed_coeff_len = w.committed_coeff_len()?;
            if !committed_coeff_len.is_multiple_of(D_A) {
                return Err(AkitaError::InvalidSize {
                    expected: D_A,
                    actual: committed_coeff_len,
                });
            }

            let num_ring_elems = committed_coeff_len / D_A;
            tracing::debug!(
                num_ring_elems,
                num_live_blocks = commit_params.num_live_blocks,
                num_positions_per_block = commit_params.num_positions_per_block,
                depth_commit = commit_params.num_digits_inner,
                depth_open = commit_params.num_digits_open,
                position_index_bits = commit_params.position_index_bits(),
                block_index_bits = commit_params.block_index_bits(),
                inner_width = commit_params.inner_width(),
                pow2_block = 1usize << commit_params.position_index_bits(),
                "commit_w layout"
            );

            let w_view = w.view::<Cfg::Field, D_A>()?;
            let plan = CommitInnerPlan::from_level(commit_params);
            let inner = if let Some(prefix) = commit_prefix {
                if packed_witness.is_some() {
                    return Err(AkitaError::InvalidSetup(
                        "a tensor-packed witness cannot use a canonical commit prefix".into(),
                    ));
                }
                let block_coeff_len =
                    plan.num_positions_per_block
                        .checked_mul(D_A)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("commit block width overflow".into())
                        })?;
                if prefix.coeff_len == 0
                    || !prefix.coeff_len.is_multiple_of(block_coeff_len)
                    || prefix.coeff_len > committed_coeff_len
                {
                    return Err(AkitaError::InvalidSetup(
                        "recursive commitment prefix does not cover complete source blocks".into(),
                    ));
                }
                let prefix_blocks = prefix.coeff_len / block_coeff_len;
                validate_commit_inner_shape::<Cfg::Field, D_A>(
                    &prefix.inner,
                    prefix_blocks,
                    plan.n_a,
                )?;
                let suffix_digits = w_view
                    .committed_i8_digits()
                    .get(prefix.coeff_len..)
                    .ok_or(AkitaError::InvalidProof)?;
                let suffix_live_coeff_len = w
                    .live_coeff_len()
                    .checked_sub(prefix.coeff_len)
                    .ok_or(AkitaError::InvalidProof)?;
                let mut coefficients = prefix.inner.into_inner_rows().into_coeffs();
                if suffix_live_coeff_len != 0 {
                    let suffix_view = SuffixWitnessView::<Cfg::Field, D_A>::
                        from_balanced_i8_digits_with_live_len(
                            suffix_digits,
                            suffix_live_coeff_len,
                            prefix.known_balanced_log_basis,
                        )?;
                    let suffix_group =
                        backend.commit_inner_group(prepared, vec![suffix_view], plan)?;
                    let [suffix] = suffix_group
                        .try_into()
                        .map_err(|_: Vec<_>| AkitaError::InvalidProof)?;
                    coefficients.extend(suffix.into_inner_rows().into_coeffs());
                }
                CommitInnerWitness {
                    inner_rows: RingVec::from_coeffs_with_ring_dim(coefficients, D_A)?,
                }
            } else {
                let inner_group = backend.commit_inner_group(prepared, vec![w_view], plan)?;
                let [inner] = inner_group
                    .try_into()
                    .map_err(|_: Vec<_>| AkitaError::InvalidProof)?;
                inner
            };
            validate_commit_inner_shape::<Cfg::Field, D_A>(
                &inner,
                commit_params.num_live_blocks,
                commit_params.inner_commit_matrix.output_rank(),
            )?;
            let n_a = commit_params.inner_commit_matrix.output_rank();
            let blocks = (0..commit_params.num_live_blocks)
                .map(|block| inner.block_rows::<D_A>(block, n_a))
                .collect::<Result<Vec<_>, _>>()?;
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                Cfg::Field,
                dims.d_b(),
                |D_B| {
                    let decomposed_inner_rows = decompose_commit_blocks_into::<Cfg::Field, D_A, D_B>(
                        &blocks,
                        commit_params.num_digits_outer,
                        commit_params.log_basis_outer,
                    )?;
                    let outer = commit_outer_slices::<Cfg::Field, _, D_B>(
                        backend,
                        prepared,
                        commit_params.outer_commit_matrix.output_rank(),
                        std::iter::once(&decomposed_inner_rows),
                        &slice_geometry,
                        commit_params.log_basis_outer,
                    )?;
                    let source = RingVec::from_ring_elems(&outer.rows);
                    let outer_relation_quotients = outer
                        .quotients
                        .as_ref()
                        .map(|quotients| RingVec::from_ring_elems(quotients));
                    if !commit_params.payload_mode.is_compressed() {
                        Ok::<_, AkitaError>((
                            packed_witness,
                            inner.into_inner_rows(),
                            source,
                            outer_relation_quotients,
                            None,
                        ))
                    } else {
                        let plan = CompressionChainPlan::for_complete_source(
                            commit_params
                                .outer_commit_matrix
                                .sis_table_key()
                                .modulus_profile,
                            source.coeff_len(),
                        )?;
                        let (mut outputs, _) = execute_compression_chains(
                            commit_ctx,
                            vec![CompressionExecutionInput {
                                id: (),
                                plan,
                                coefficients: source.into_coeffs(),
                            }],
                        )?;
                        let output = outputs.pop().ok_or(AkitaError::InvalidProof)?;
                        let terminal_ring_dim = output
                            .witness
                            .plan()
                            .maps()
                            .last()
                            .ok_or(AkitaError::InvalidProof)?
                            .ring_dimension();
                        let payload = RingVec::from_coeffs_with_ring_dim(
                            output.terminal.into_coefficients(),
                            terminal_ring_dim,
                        )?;
                        Ok::<_, AkitaError>((
                            packed_witness,
                            inner.into_inner_rows(),
                            payload,
                            outer_relation_quotients,
                            Some((output.witness, output.quotients)),
                        ))
                    }
                }
            )
        }
    )?;
    let hint = match compression_witness {
        Some((compression_witness, compression_quotients)) => {
            AkitaCommitmentHint::singleton_with_outer_compression(
                inner_rows,
                &compression_witness,
                &compression_quotients,
            )?
        }
        None => AkitaCommitmentHint::singleton(inner_rows)?,
    }
    .with_outer_relation_quotients(outer_relation_quotients)?;
    Ok(NextWitnessStateOutput {
        witness: packed_witness,
        binding: NextWitnessState::OuterPayload(commitment),
        hint,
    })
}

/// Bind the witness entering the terminal fold with its canonical inner
/// commitment state. No outer digits or outer commitment are computed.
#[inline(never)]
pub fn commit_terminal_w<Cfg, B>(
    commit_params: &TerminalCommittedGroupParams,
    expanded: &std::sync::Arc<AkitaExpandedSetup<Cfg::Field>>,
    commit_ctx: &OperationCtx<'_, Cfg::Field, B>,
    logical_w: &RecursiveWitnessFlat,
) -> Result<NextWitnessStateOutput<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling,
    B: RuntimeCommitBackendFor<Cfg::Field, RecursiveWitnessFlat>,
{
    let ring_dim = commit_params.d_a();
    let backend = commit_ctx.backend();
    let prepared = commit_ctx.prepared();
    backend.validate_prepared_setup(prepared, expanded.as_ref())?;

    let (packed_witness, t_state) = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Cfg::Field,
        ring_dim,
        |D_A| {
            let packed_witness = if <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE == 1 {
                None
            } else {
                Some(tensor_pack_recursive_witness::<
                    Cfg::Field,
                    Cfg::ExtField,
                    D_A,
                >(logical_w)?)
            };
            let witness = packed_witness.as_ref().unwrap_or(logical_w);
            let view = witness.view::<Cfg::Field, D_A>()?;
            let plan = CommitInnerPlan {
                n_a: commit_params.inner_commit_matrix.output_rank(),
                num_positions_per_block: commit_params.num_positions_per_block,
                num_digits_inner: commit_params.num_digits_inner,
                log_basis_inner: commit_params.log_basis_inner,
            };
            let inner_group = backend.commit_inner_group(prepared, vec![view], plan)?;
            let [inner] = inner_group
                .try_into()
                .map_err(|_: Vec<_>| AkitaError::InvalidProof)?;
            Ok::<_, AkitaError>((packed_witness, inner.into_inner_rows()))
        }
    )?;
    Ok(NextWitnessStateOutput {
        witness: packed_witness,
        binding: NextWitnessState::TerminalInnerState,
        hint: AkitaCommitmentHint::singleton(t_state)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    fn prefix(coeff_len: usize, start: i64) -> RecursiveCommitPrefix<F> {
        let coefficients = (0..64)
            .map(|offset| F::from_i64(start + offset as i64))
            .collect();
        RecursiveCommitPrefix {
            coeff_len,
            known_balanced_log_basis: 3,
            inner: CommitInnerWitness {
                inner_rows: RingVec::from_coeffs_with_ring_dim(coefficients, 64).unwrap(),
            },
        }
    }

    #[test]
    fn recursive_commit_prefix_merge_preserves_block_order() {
        let merged =
            merge_recursive_commit_prefixes(vec![prefix(1024, 0), prefix(2048, 100)]).unwrap();
        assert_eq!(merged.coeff_len, 3072);
        assert_eq!(merged.known_balanced_log_basis, 3);
        assert_eq!(merged.inner.ring_dim(), 64);
        let coefficients = merged.inner.into_inner_rows().into_coeffs();
        assert_eq!(coefficients[0], F::from_i64(0));
        assert_eq!(coefficients[63], F::from_i64(63));
        assert_eq!(coefficients[64], F::from_i64(100));
        assert_eq!(coefficients[127], F::from_i64(163));
    }
}
