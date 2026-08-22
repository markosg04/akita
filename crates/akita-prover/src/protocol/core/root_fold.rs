use super::*;
use crate::compute::{
    ComputeBackendSetup, ComputeExecutionDomain, DecomposeFoldChunk, DecomposeFoldChunkSink,
    DigitRowsComputeBackend, LevelProveStacks, OperationCtx, ProverComputeStack,
    RuntimeCommitBackendFor, RuntimeRingSwitchProveBackend,
};
use crate::protocol::fold_grind::FoldProbeSink;
use crate::protocol::ring_switch::{
    balanced_decompose_centered_i32_i8_into, merge_recursive_commit_prefixes,
    prepare_recursive_commit_prefix, RecursiveCommitPrefix,
};
use crate::RecursiveWitnessFlat;
use crate::{DirectDigitRangeProofBackend, DirectRelationRangeProofBackend};
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;
use akita_types::{dispatch_for_field, CommittedSourceEncoding};

const ROOT_STREAM_TARGET_CHUNKS: usize = 8;
const PIPELINE_MIN_WITNESS_COEFFS: usize = 64 * 1024 * 1024;

struct RootZCommitSink<'a, 'stack, F, C, Cfg>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
{
    commit_params: &'a CommittedGroupParams,
    expanded: &'a AkitaExpandedSetup<F>,
    commit_ctx: &'a OperationCtx<'stack, F, C>,
    root_ring_dim: usize,
    root_positions: usize,
    witness_digits: usize,
    fold_digits: usize,
    fold_log_basis: u32,
    known_balanced_log_basis: u32,
    commit_block_coeff_len: usize,
    preferred_chunk_len: usize,
    next_position: usize,
    pending_digits: Vec<i8>,
    prefixes: Vec<RecursiveCommitPrefix<F>>,
    accepted_prefix: Option<RecursiveCommitPrefix<F>>,
    _config: core::marker::PhantomData<fn() -> Cfg>,
}

impl<'a, 'stack, F, C, Cfg> RootZCommitSink<'a, 'stack, F, C, Cfg>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
{
    fn new(
        root_params: &CommittedGroupParams,
        commit_params: &'a CommittedGroupParams,
        expanded: &'a AkitaExpandedSetup<F>,
        commit_ctx: &'a OperationCtx<'stack, F, C>,
    ) -> Result<Self, AkitaError> {
        if root_params.witness_chunk.num_chunks != 1
            || commit_params.source_encoding != CommittedSourceEncoding::CanonicalCoefficientTable
        {
            return Err(AkitaError::InvalidSetup(
                "root streaming requires one response chunk and a canonical successor".into(),
            ));
        }
        let commit_block_coeff_len = commit_params
            .num_positions_per_block
            .checked_mul(commit_params.role_dims().d_a())
            .ok_or_else(|| AkitaError::InvalidSetup("commit block width overflow".into()))?;
        let root_positions = root_params.num_positions_per_block;
        let z_coeff_len = root_positions
            .checked_mul(root_params.num_digits_inner)
            .and_then(|count| count.checked_mul(root_params.num_digits_fold))
            .and_then(|count| count.checked_mul(root_params.role_dims().d_a()))
            .ok_or_else(|| AkitaError::InvalidSetup("root Z length overflow".into()))?;
        if z_coeff_len < commit_block_coeff_len {
            return Err(AkitaError::InvalidSetup(
                "root Z prefix contains no complete successor commit block".into(),
            ));
        }
        Ok(Self {
            commit_params,
            expanded,
            commit_ctx,
            root_ring_dim: root_params.role_dims().d_a(),
            root_positions,
            witness_digits: root_params.num_digits_inner,
            fold_digits: root_params.num_digits_fold,
            fold_log_basis: root_params.log_basis_open,
            known_balanced_log_basis: root_params
                .log_basis_inner
                .max(root_params.log_basis_outer)
                .max(root_params.log_basis_open),
            commit_block_coeff_len,
            preferred_chunk_len: root_positions.div_ceil(ROOT_STREAM_TARGET_CHUNKS),
            next_position: 0,
            pending_digits: Vec::new(),
            prefixes: Vec::new(),
            accepted_prefix: None,
            _config: core::marker::PhantomData,
        })
    }

    fn take_accepted_prefix(&mut self) -> Option<RecursiveCommitPrefix<F>> {
        self.accepted_prefix.take()
    }

    fn consume_typed<const D: usize>(
        &mut self,
        chunk: &DecomposeFoldChunk<'_>,
    ) -> Result<(), AkitaError>
    where
        Cfg: CommitmentConfig<Field = F>,
        C: RuntimeCommitBackendFor<F, RecursiveWitnessFlat>,
    {
        if D != self.root_ring_dim
            || chunk.witness_digits() != self.witness_digits
            || chunk.position_start() != self.next_position
            || chunk.position_start() + chunk.position_count() > self.root_positions
        {
            return Err(AkitaError::InvalidInput(
                "streamed root chunk is out of canonical order".into(),
            ));
        }
        let (centered_rows, remainder) = chunk.centered_coefficients().as_chunks::<D>();
        if !remainder.is_empty()
            || centered_rows.len() != chunk.position_count() * self.witness_digits
        {
            return Err(AkitaError::InvalidSize {
                expected: chunk.position_count() * self.witness_digits * D,
                actual: chunk.centered_coefficients().len(),
            });
        }
        let digit_rows = centered_rows
            .len()
            .checked_mul(self.fold_digits)
            .ok_or_else(|| AkitaError::InvalidSetup("streamed Z digit count overflow".into()))?;
        let mut digits = vec![[0i8; D]; digit_rows];
        for (centered, planes) in centered_rows
            .iter()
            .zip(digits.chunks_exact_mut(self.fold_digits))
        {
            balanced_decompose_centered_i32_i8_into(centered, planes, self.fold_log_basis);
        }
        self.pending_digits.extend(digits.into_flattened());
        self.next_position += chunk.position_count();

        let ready_len =
            self.pending_digits.len() / self.commit_block_coeff_len * self.commit_block_coeff_len;
        if ready_len != 0 {
            let suffix = self.pending_digits.split_off(ready_len);
            let ready = core::mem::replace(&mut self.pending_digits, suffix);
            self.prefixes
                .push(prepare_recursive_commit_prefix::<Cfg, C>(
                    self.commit_params,
                    self.expanded,
                    self.commit_ctx,
                    &ready,
                    self.known_balanced_log_basis,
                )?);
        }
        Ok(())
    }
}

impl<F, C, Cfg> DecomposeFoldChunkSink for RootZCommitSink<'_, '_, F, C, Cfg>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig<Field = F>,
    C: RuntimeCommitBackendFor<F, RecursiveWitnessFlat> + ComputeBackendSetup<F>,
{
    fn preferred_position_chunk_len(&self, total_positions: usize) -> usize {
        debug_assert_eq!(total_positions, self.root_positions);
        self.preferred_chunk_len
    }

    fn consume(&mut self, chunk: DecomposeFoldChunk<'_>) -> Result<(), AkitaError> {
        dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            chunk.ring_dimension(),
            |D| self.consume_typed::<D>(&chunk)
        )
    }
}

impl<F, C, Cfg> FoldProbeSink for RootZCommitSink<'_, '_, F, C, Cfg>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig<Field = F>,
    C: RuntimeCommitBackendFor<F, RecursiveWitnessFlat> + ComputeBackendSetup<F>,
{
    fn begin_attempt(&mut self, _nonce: u32) -> Result<(), AkitaError> {
        self.next_position = 0;
        self.pending_digits.clear();
        self.prefixes.clear();
        self.accepted_prefix = None;
        Ok(())
    }

    fn chunk_sink(&mut self) -> &mut dyn DecomposeFoldChunkSink {
        self
    }

    fn finish_attempt(&mut self, accepted: bool) -> Result<(), AkitaError> {
        if self.next_position != self.root_positions {
            return Err(AkitaError::InvalidInput(
                "streamed root attempt ended before every position completed".into(),
            ));
        }
        self.pending_digits.clear();
        if accepted {
            self.accepted_prefix = Some(merge_recursive_commit_prefixes(core::mem::take(
                &mut self.prefixes,
            ))?);
        } else {
            self.prefixes.clear();
        }
        Ok(())
    }
}

fn validate_packing_root_opening_shape<F, E>(
    ring_d: usize,
    alpha_bits: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: FpExtEncoding<F>,
{
    let ext_degree = <E as ExtField<F>>::EXT_DEGREE;
    if ext_degree == 0
        || !ring_d.is_multiple_of(ext_degree)
        || !(ring_d / ext_degree).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "extension-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }

    let packed_slots = ring_d / ext_degree;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if packed_inner_bits > alpha_bits {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: alpha_bits,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_root<'p, F, E, T, P, C, O, TS, R>(
    stack: &ProverComputeStack<'_, F, C, O, TS, R>,
    transcript: &'p mut T,
    claims: ProverOpeningData<'_, E, P, F>,
    root_params: &CommittedGroupParams,
    basis: BasisMode,
    fold_sink: Option<&'p mut dyn FoldProbeSink>,
) -> Result<PreparedFold<F, E>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    P: RootProverGroupOpening<F, E, O> + Clone,
    TS: ComputeBackendSetup<F>,
    O: DigitRowsComputeBackend<F>,
    C: ComputeBackendSetup<F>,
    R: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
{
    let opening_batch = claims.opening_layout()?;
    let opening_method = super::fold::uniform_opening_method(root_params, &opening_batch)?;
    if !matches!(
        opening_method,
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    ) || root_params.source_encoding
        != akita_types::CommittedSourceEncoding::CanonicalCoefficientTable
    {
        return Err(AkitaError::InvalidSetup(
            "root folds require canonical coefficient packing".into(),
        ));
    }
    // A-role root fold ring dimension (schedule-derived).
    let root_ring_d = root_params.role_dims().d_a();
    let alpha_bits = root_ring_d.trailing_zeros() as usize;
    prepare_single_field_fold::<F, E, T, P, _, C, O, TS, R>(
        stack,
        claims,
        false,
        transcript,
        || validate_packing_root_opening_shape::<F, E>(root_ring_d, alpha_bits),
        root_params,
        basis,
        fold_sink,
    )
}

/// Prove the folded-root proof payload for an intermediate root.
///
/// The caller owns schedule/config selection and passes the validated schedule
/// execution for level 0. This function owns root polynomial folding, public
/// root transcript setup, root ring-relation construction, and the folded-root
/// prover mechanics.
///
/// # Errors
///
/// Returns an error if root inputs are malformed, polynomial folding or
/// ring-relation construction fails, or the folded-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn prove_root<'stack, F, E, T, P, C, O, TS, R, Cfg>(
    expanded: &Arc<AkitaExpandedSetup<F>>,
    prefix_slots: &SetupPrefixProverRegistry<F>,
    stacks: &'stack impl LevelProveStacks<
        'stack,
        F,
        Commit = C,
        Opening = O,
        Tensor = TS,
        RingSwitch = R,
    >,
    transcript: &mut T,
    claims: ProverOpeningData<'_, E, P, F>,
    scheduled: &akita_types::RootFoldStep,
    next_params: super::fold::FoldSuccessorParams<'_>,
    next_witness_binding: akita_types::NextWitnessBindingPolicy,
    basis: BasisMode,
) -> Result<ProveLevelOutput<F, E>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + PseudoMersenneField
        + FromPrimitiveInt
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F>
        + ExtField<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + FromPrimitiveInt
        + MulBaseUnreduced<F>
        + AkitaSerialize,
    T: Transcript<F> + ProverTranscriptGrind<F>,
    P: RootProverGroupOpening<F, E, O> + Clone,
    C: RuntimeCommitBackendFor<F, RecursiveWitnessFlat> + ComputeBackendSetup<F> + 'stack,
    O: DigitRowsComputeBackend<F>
        + ComputeBackendSetup<F>
        + DirectDigitRangeProofBackend<F, E>
        + DirectRelationRangeProofBackend<F, E>
        + 'stack,
    TS: ComputeBackendSetup<F> + 'stack,
    R: RuntimeRingSwitchProveBackend<F>
        + DigitRowsComputeBackend<F>
        + ComputeBackendSetup<F>
        + 'stack,
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    <C as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <O as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <TS as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
    <R as ComputeBackendSetup<F>>::PreparedSetup: 'stack,
{
    let stack = stacks.prove_stack_at_level(0);
    let root_params = &scheduled.params.final_group.commitment;
    let opening_layout = claims.opening_layout()?;
    let opening_method = super::fold::uniform_opening_method(root_params, &opening_layout)?;
    if !matches!(
        opening_method,
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    ) || root_params.source_encoding
        != akita_types::CommittedSourceEncoding::CanonicalCoefficientTable
    {
        return Err(AkitaError::InvalidSetup(
            "root folds require canonical coefficient packing".into(),
        ));
    }

    // Absorb root claims through the D-free flat commitment encoder keyed on the
    // root level's B-role dimension (byte-identical to the verifier's
    // `claims.append_to_transcript` and to the former typed path; S2/S7 parity).
    claims.append_to_transcript::<T>(root_params, transcript)?;

    let streamed_commit_params = match next_params {
        super::fold::FoldSuccessorParams::Recursive(params)
            if params.witness.source_encoding
                == CommittedSourceEncoding::CanonicalCoefficientTable
                && scheduled.output_witness_len >= PIPELINE_MIN_WITNESS_COEFFS
                && opening_layout.num_groups() == 1
                && root_params.witness_chunk.num_chunks == 1
                && stack.opening().backend().execution_domain()
                    == ComputeExecutionDomain::Accelerator
                && stack.commit().backend().execution_domain() == ComputeExecutionDomain::Host =>
        {
            Some(&params.witness)
        }
        _ => None,
    };
    let mut fold_sink = streamed_commit_params
        .map(|commit_params| {
            RootZCommitSink::<F, C, Cfg>::new(
                root_params,
                commit_params,
                expanded.as_ref(),
                stack.commit(),
            )
        })
        .transpose()?;
    let prepared_fold = prepare_root::<F, E, T, P, C, O, TS, R>(
        stack,
        transcript,
        claims,
        root_params,
        basis,
        fold_sink
            .as_mut()
            .map(|sink| sink as &mut dyn FoldProbeSink),
    )
    .map_err(|err| AkitaError::InvalidInput(format!("prepare root failed: {err:?}")))?;
    let early_commit_prefix = fold_sink
        .as_mut()
        .map(|sink| {
            sink.take_accepted_prefix().ok_or_else(|| {
                AkitaError::InvalidInput(
                    "streamed root fold did not retain its accepted commitment prefix".into(),
                )
            })
        })
        .transpose()?;

    prove_fold::<F, E, T, C, O, TS, R, Cfg>(
        expanded,
        prefix_slots,
        stack,
        transcript,
        0,
        root_params,
        Some(next_params),
        Some(scheduled.output_witness_len),
        early_commit_prefix,
        Some(next_witness_binding),
        prepared_fold,
    )
    .map_err(|err| AkitaError::InvalidInput(format!("prove root fold failed: {err:?}")))
}
