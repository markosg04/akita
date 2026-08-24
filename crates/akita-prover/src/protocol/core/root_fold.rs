use super::*;
use crate::compute::{
    prewarm_ntt_requirements, BalancedDigitRequest, ComputeBackendSetup, ComputeExecutionDomain,
    DecomposeFoldChunk, DecomposeFoldChunkSink, DigitRowsComputeBackend, LevelProveStacks,
    NttExecutionRequirements, OperationCtx, ProverComputeStack, RuntimeCommitBackendFor,
    RuntimeRingSwitchProveBackend,
};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::protocol::fold_grind::FoldProbeSink;
use crate::protocol::ring_relation::RingRelationPreFoldSink;
use crate::protocol::ring_switch::{
    balanced_decompose_centered_i32_i8_into, merge_recursive_commit_prefixes,
    prepare_recursive_commit_prefix, RecursiveCommitPrefix,
};
use crate::RecursiveWitnessFlat;
use crate::{DirectDigitRangeProofBackend, DirectRelationRangeProofBackend};
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;
use akita_types::{
    dispatch_for_field, emit_witness_e_planes, emit_witness_t_planes, r_decomp_levels,
    AkitaCommitmentHint, CommittedSourceEncoding, OpeningClaimsLayout, RelationWitnessGeometry,
    WitnessLayout,
};

const ROOT_STREAM_TARGET_CHUNKS: usize = 8;
const PIPELINE_MIN_WITNESS_COEFFS: usize = 64 * 1024 * 1024;

struct RootEtCommitSink<'a, 'stack, F, C, Cfg>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
{
    commit_params: &'a CommittedGroupParams,
    expanded: &'a AkitaExpandedSetup<F>,
    commit_ctx: &'a OperationCtx<'stack, F, C>,
    prefix_start: Option<usize>,
    prefix: Option<RecursiveCommitPrefix<F>>,
    _config: core::marker::PhantomData<fn() -> Cfg>,
}

impl<'a, 'stack, F, C, Cfg> RootEtCommitSink<'a, 'stack, F, C, Cfg>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
{
    const fn new(
        commit_params: &'a CommittedGroupParams,
        expanded: &'a AkitaExpandedSetup<F>,
        commit_ctx: &'a OperationCtx<'stack, F, C>,
    ) -> Self {
        Self {
            commit_params,
            expanded,
            commit_ctx,
            prefix_start: None,
            prefix: None,
            _config: core::marker::PhantomData,
        }
    }

    fn take_prefix(&mut self) -> Option<(usize, RecursiveCommitPrefix<F>)> {
        self.prefix_start.take().zip(self.prefix.take())
    }
}

impl<F, C, Cfg> RingRelationPreFoldSink<F> for RootEtCommitSink<'_, '_, F, C, Cfg>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig<Field = F>,
    C: RuntimeCommitBackendFor<F, RecursiveWitnessFlat> + ComputeBackendSetup<F>,
{
    fn prepare(
        &mut self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        relation_geometry: &RelationWitnessGeometry,
        e_hats: &[&akita_types::DigitBlocks],
        hints: &[AkitaCommitmentHint<F>],
    ) -> Result<(), AkitaError> {
        let [e_hat] = e_hats else {
            return Err(AkitaError::InvalidSetup(
                "static E/T prefix requires one opening group".into(),
            ));
        };
        let [hint] = hints else {
            return Err(AkitaError::InvalidSetup(
                "static E/T prefix requires one commitment hint".into(),
            ));
        };
        let layout = WitnessLayout::new(
            level_params,
            opening_batch,
            relation_geometry,
            level_params.witness_chunk.num_chunks,
            r_decomp_levels::<F>(level_params.log_basis_open),
        )?;
        let [unit] = layout.units() else {
            return Err(AkitaError::InvalidSetup(
                "static E/T prefix requires one witness unit".into(),
            ));
        };
        let block_coeff_len = self
            .commit_params
            .num_positions_per_block
            .checked_mul(self.commit_params.role_dims().d_a())
            .ok_or_else(|| AkitaError::InvalidSetup("commit block width overflow".into()))?;
        let z_end = unit.z_range().end;
        let et_end = unit.t_range().end;
        if !z_end.is_multiple_of(block_coeff_len) {
            return Ok(());
        }
        let prefix_end = et_end / block_coeff_len * block_coeff_len;
        if prefix_end <= z_end {
            return Ok(());
        }

        let group_params = level_params.group_params_geometry(opening_batch, 0)?;
        let group_dims = level_params.group_role_dims_geometry(opening_batch, 0)?;
        let num_claims = opening_batch.group_layout(0)?.num_polynomials();
        if hint.ring_dim() != group_dims.d_a() || hint.inner_rows().len() != num_claims {
            return Err(AkitaError::InvalidInput(
                "static E/T prefix hint shape mismatch".into(),
            ));
        }
        let expected_rings_per_polynomial = group_params
            .num_live_blocks()
            .checked_mul(group_params.a_rows_len())
            .ok_or_else(|| AkitaError::InvalidSetup("commitment hint row count overflow".into()))?;
        let t_hat = dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            group_dims.d_a(),
            |D_A| {
                dispatch_for_field!(
                    akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Outer),
                    F,
                    group_dims.d_b(),
                    |D_B| {
                        let mut blocks =
                            Vec::with_capacity(num_claims * group_params.num_live_blocks());
                        for rows in hint.inner_rows() {
                            let typed_rows = rows.as_ring_slice::<D_A>()?;
                            if typed_rows.len() != expected_rings_per_polynomial {
                                return Err(AkitaError::InvalidSize {
                                    expected: expected_rings_per_polynomial,
                                    actual: typed_rows.len(),
                                });
                            }
                            blocks.extend(typed_rows.chunks_exact(group_params.a_rows_len()));
                        }
                        decompose_commit_blocks_into::<F, D_A, D_B>(
                            &blocks,
                            group_params.num_digits_outer(),
                            group_params.log_basis_outer(),
                        )
                    }
                )
            }
        )?;

        let mut digits = vec![0i8; et_end];
        let opening_width = unit.e_geometry().physical_coefficient_width();
        dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Opening),
            F,
            group_dims.d_d(),
            |D_D| emit_witness_e_planes::<D_D>(
                &mut digits,
                &layout,
                0,
                opening_width,
                num_claims,
                group_params.num_digits_open(),
                e_hat,
                group_params.num_live_blocks(),
            )
        )?;
        dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            group_dims.d_a(),
            |D_A| {
                dispatch_for_field!(
                    akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Outer),
                    F,
                    group_dims.d_b(),
                    |D_B| emit_witness_t_planes::<D_A, D_B>(
                        &mut digits,
                        &layout,
                        0,
                        num_claims,
                        group_params.a_rows_len(),
                        group_params.num_digits_outer(),
                        &t_hat,
                        group_params.num_live_blocks(),
                    )
                )
            }
        )?;
        let known_balanced_log_basis = group_params
            .log_basis_inner()
            .max(group_params.log_basis_outer())
            .max(group_params.log_basis_open());
        self.prefix = Some(prepare_recursive_commit_prefix::<Cfg, C>(
            self.commit_params,
            self.expanded,
            self.commit_ctx,
            &digits[z_end..prefix_end],
            known_balanced_log_basis,
        )?);
        self.prefix_start = Some(z_end);
        tracing::info!(
            prefix_start = z_end,
            prefix_end,
            prefix_bytes = prefix_end - z_end,
            "prepared static root E/T commitment prefix"
        );
        Ok(())
    }
}

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
        let _span = tracing::info_span!("root_z_digit_append").entered();
        if let Some(digits) = chunk.balanced_digits() {
            if digits.request()
                != (BalancedDigitRequest {
                    num_digits: self.fold_digits,
                    log_basis: self.fold_log_basis,
                })
            {
                return Err(AkitaError::InvalidInput(
                    "streamed root digit hint has the wrong decomposition".into(),
                ));
            }
            self.pending_digits.extend_from_slice(digits.digits());
        } else {
            let digit_rows = centered_rows
                .len()
                .checked_mul(self.fold_digits)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("streamed Z digit count overflow".into())
                })?;
            let digits = {
                let _span = tracing::info_span!("root_z_digit_decompose").entered();
                let mut digits = vec![[0i8; D]; digit_rows];
                for (centered, planes) in centered_rows
                    .iter()
                    .zip(digits.chunks_exact_mut(self.fold_digits))
                {
                    balanced_decompose_centered_i32_i8_into(centered, planes, self.fold_log_basis);
                }
                digits
            };
            self.pending_digits.extend(digits.into_flattened());
        }
        self.next_position += chunk.position_count();

        let ready_len =
            self.pending_digits.len() / self.commit_block_coeff_len * self.commit_block_coeff_len;
        if ready_len != 0 {
            let suffix = self.pending_digits.split_off(ready_len);
            let ready = core::mem::replace(&mut self.pending_digits, suffix);
            let _span = tracing::info_span!("root_z_commit_prefix").entered();
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

    fn balanced_digit_request(&self) -> Option<BalancedDigitRequest> {
        Some(BalancedDigitRequest {
            num_digits: self.fold_digits,
            log_basis: self.fold_log_basis,
        })
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
            let _span = tracing::info_span!("root_commit_prefix_merge").entered();
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
    pre_fold_task: Option<&'p mut (dyn FnMut() -> Result<(), AkitaError> + Send)>,
    pre_fold_sink: Option<&'p mut dyn crate::protocol::ring_relation::RingRelationPreFoldSink<F>>,
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
        pre_fold_task,
        pre_fold_sink,
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
    stacks: &'stack (impl LevelProveStacks<'stack, F, Commit = C, Opening = O, Tensor = TS, RingSwitch = R>
                 + Sync),
    transcript: &mut T,
    claims: ProverOpeningData<'_, E, P, F>,
    scheduled: &akita_types::RootFoldStep,
    next_params: super::fold::FoldSuccessorParams<'_>,
    next_witness_binding: akita_types::NextWitnessBindingPolicy,
    basis: BasisMode,
    deferred_ntt_requirements: Option<&NttExecutionRequirements>,
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
    let mut et_commit_sink = streamed_commit_params.map(|commit_params| {
        RootEtCommitSink::<F, C, Cfg>::new(commit_params, expanded.as_ref(), stack.commit())
    });
    let mut prewarm_ntt = || {
        let _span = tracing::info_span!("root_ntt_prewarm").entered();
        deferred_ntt_requirements.map_or(Ok(()), |requirements| {
            prewarm_ntt_requirements::<F, _>(stacks, requirements)
        })
    };
    let prepared_fold = prepare_root::<F, E, T, P, C, O, TS, R>(
        stack,
        transcript,
        claims,
        root_params,
        basis,
        fold_sink
            .as_mut()
            .map(|sink| sink as &mut dyn FoldProbeSink),
        deferred_ntt_requirements
            .map(|_| &mut prewarm_ntt as &mut (dyn FnMut() -> Result<(), AkitaError> + Send)),
        et_commit_sink
            .as_mut()
            .map(|sink| sink as &mut dyn RingRelationPreFoldSink<F>),
    )
    .map_err(|err| AkitaError::InvalidInput(format!("prepare root failed: {err:?}")))?;
    let mut early_commit_prefix = fold_sink
        .as_mut()
        .map(|sink| {
            sink.take_accepted_prefix().ok_or_else(|| {
                AkitaError::InvalidInput(
                    "streamed root fold did not retain its accepted commitment prefix".into(),
                )
            })
        })
        .transpose()?;
    if let Some((prefix_start, et_prefix)) = et_commit_sink
        .as_mut()
        .and_then(RootEtCommitSink::take_prefix)
    {
        let z_prefix = early_commit_prefix.take().ok_or_else(|| {
            AkitaError::InvalidInput(
                "static E/T commitment prefix is missing its Z predecessor".into(),
            )
        })?;
        if z_prefix.coeff_len() != prefix_start {
            return Err(AkitaError::InvalidSetup(
                "static E/T commitment prefix is not contiguous with Z".into(),
            ));
        }
        early_commit_prefix = Some(merge_recursive_commit_prefixes(vec![z_prefix, et_prefix])?);
    }

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
