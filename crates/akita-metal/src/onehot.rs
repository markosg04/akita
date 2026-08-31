use std::mem::size_of;
use std::time::Instant;

use akita_error::AkitaError;
use akita_prover::backend::OneHotView;
use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
use akita_prover::{CommitInnerWitness, OneHotIndex};
use akita_types::RingVec;

use crate::backend::{MetalBackend, MetalCommitMetrics};
use crate::field::{Fp128Limbs, F};
use crate::prepared::MetalPreparedSetup;
use crate::runtime::{MetalOneHotKernel, OneHotCommitParams};
use crate::{MetalCommitError, MetalExecutionPolicy};

const SUPPORTED_RING_DIMENSIONS: [usize; 3] = [64, 128, 256];
const NONE_INDEX: u16 = u16::MAX;
const METAL_GATHER_THRESHOLD_BYTES: usize = 64 * 1024 * 1024;

struct ValidatedShape {
    total_field_elements: usize,
    chunks_per_source: usize,
    num_blocks: usize,
    output_coefficients_per_source: usize,
    active_a_cols: usize,
}

impl<I: OneHotIndex, const D: usize> RootCommitKernel<OneHotView<'_, F, D, I>, F, D>
    for MetalBackend
{
    #[tracing::instrument(skip_all, name = "MetalBackend::onehot_commit_inner")]
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<OneHotView<'_, F, D, I>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }
        if !SUPPORTED_RING_DIMENSIONS.contains(&D) {
            return self.unsupported_or_cpu(
                prepared,
                sources,
                plan,
                format!("D={D}; supported dimensions are 64, 128, and 256"),
            );
        }
        let Some(runtime) = self.runtime() else {
            return self.unsupported_or_cpu(
                prepared,
                sources,
                plan,
                "no Metal runtime is available".into(),
            );
        };

        let total_start = Instant::now();
        let shape = validate_shape::<I, D>(&sources, plan)?;

        let index_pack_start = Instant::now();
        let (hot_indices, hot_entries) = pack_hot_indices(&sources)?;
        let index_pack_time = index_pack_start.elapsed();
        let field_additions = hot_entries
            .checked_mul(plan.n_a)
            .and_then(|count| count.checked_mul(D))
            .ok_or_else(|| MetalCommitError::ShapeOverflow("field additions").into_akita())?;
        let gathered_matrix_bytes = field_additions
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or_else(|| MetalCommitError::ShapeOverflow("gathered matrix bytes").into_akita())?;
        let output_coefficients = shape
            .output_coefficients_per_source
            .checked_mul(sources.len())
            .ok_or_else(|| {
                MetalCommitError::ShapeOverflow("group output coefficients").into_akita()
            })?;
        let block_batched_fits_u32 = [
            sources.len(),
            shape.chunks_per_source,
            sources[0].onehot_k(),
            D,
            plan.n_a,
            plan.num_positions_per_block,
            plan.num_digits_inner,
            shape.num_blocks,
            shape.total_field_elements,
            output_coefficients,
            shape
                .active_a_cols
                .checked_mul(plan.n_a)
                .and_then(|count| count.checked_mul(D))
                .unwrap_or(usize::MAX),
        ]
        .into_iter()
        .all(|value| u32::try_from(value).is_ok());
        let kernel = if gathered_matrix_bytes >= METAL_GATHER_THRESHOLD_BYTES
            && shape.num_blocks > 1
            && sources[0].onehot_k().is_power_of_two()
            && D.is_power_of_two()
            && block_batched_fits_u32
        {
            MetalOneHotKernel::BlockBatched
        } else {
            MetalOneHotKernel::DirectGather
        };
        if self.policy() == MetalExecutionPolicy::PreferMetal
            && kernel == MetalOneHotKernel::DirectGather
        {
            return self
                .cpu_backend()
                .commit_inner_group(&prepared.cpu, sources, plan);
        }
        let matrix = prepared.matrix(runtime, D, plan.n_a, shape.active_a_cols)?;
        let params = OneHotCommitParams {
            num_sources: to_u64(sources.len(), "source count")?,
            chunks_per_source: to_u64(shape.chunks_per_source, "chunks per source")?,
            onehot_k: to_u64(sources[0].onehot_k(), "one-hot K")?,
            ring_d: D as u64,
            n_a: to_u64(plan.n_a, "A row count")?,
            positions_per_block: to_u64(plan.num_positions_per_block, "positions per block")?,
            num_digits_inner: to_u64(plan.num_digits_inner, "inner digit count")?,
            num_blocks: to_u64(shape.num_blocks, "block count")?,
            total_field_elements: to_u64(shape.total_field_elements, "total field elements")?,
            output_coefficients: to_u64(output_coefficients, "output coefficients")?,
            blocks_per_threadgroup: 0,
            log_onehot_k: u64::from(sources[0].onehot_k().trailing_zeros()),
            log_ring_d: u64::from(D.trailing_zeros()),
        };

        let outcome = runtime
            .dispatch_onehot(matrix.buffer.as_ref(), &hot_indices, params, kernel)
            .map_err(MetalCommitError::into_akita)?;
        let output_reconstruction_start = Instant::now();
        let witnesses = reconstruct_witnesses::<D>(
            outcome.coefficients,
            sources.len(),
            shape.output_coefficients_per_source,
        )
        .map_err(MetalCommitError::into_akita)?;
        let output_reconstruction_time = output_reconstruction_start.elapsed();

        let metrics = MetalCommitMetrics {
            kernel: outcome.kernel,
            blocks_per_threadgroup: outcome.blocks_per_threadgroup,
            num_sources: sources.len(),
            hot_entries,
            field_additions: to_u64(field_additions, "field additions")?,
            gathered_matrix_bytes: to_u64(gathered_matrix_bytes, "gathered matrix bytes")?,
            output_bytes: output_coefficients
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or_else(|| {
                    MetalCommitError::ShapeOverflow("canonical output bytes").into_akita()
                })?,
            scratch_bytes: outcome.scratch_bytes,
            matrix_bytes: matrix.bytes,
            matrix_cache_hit: matrix.cache_hit,
            matrix_prepare_time: matrix.prepare_time,
            index_pack_time,
            buffer_setup_time: outcome.timings.buffer_setup,
            command_wall_time: outcome.timings.command_wall,
            gpu_time: outcome.timings.gpu,
            readback_copy_time: outcome.timings.readback_copy,
            output_reconstruction_time,
            total_time: total_start.elapsed(),
        };
        self.record_commit_metrics(metrics)
            .map_err(MetalCommitError::into_akita)?;
        Ok(witnesses)
    }
}

impl MetalBackend {
    fn unsupported_or_cpu<I: OneHotIndex, const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        sources: Vec<OneHotView<'_, F, D, I>>,
        plan: CommitInnerPlan,
        reason: String,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        match self.policy() {
            MetalExecutionPolicy::RequireMetal => {
                Err(MetalCommitError::UnsupportedShape(reason).into_akita())
            }
            MetalExecutionPolicy::PreferMetal => {
                self.cpu_backend()
                    .commit_inner_group(&prepared.cpu, sources, plan)
            }
        }
    }
}

fn validate_shape<I: OneHotIndex, const D: usize>(
    sources: &[OneHotView<'_, F, D, I>],
    plan: CommitInnerPlan,
) -> Result<ValidatedShape, AkitaError> {
    if plan.n_a == 0 || plan.num_positions_per_block == 0 || plan.num_digits_inner == 0 {
        return Err(MetalCommitError::UnsupportedShape(
            "A rows, positions per block, and inner digits must be nonzero".into(),
        )
        .into_akita());
    }
    let first = sources[0];
    let onehot_k = first.onehot_k();
    if onehot_k == 0 || onehot_k > NONE_INDEX as usize {
        return Err(MetalCommitError::UnsupportedShape(format!(
            "one-hot K={onehot_k} cannot use the u16 staging encoding"
        ))
        .into_akita());
    }
    let total_field_elements = 1usize
        .checked_shl(first.num_vars() as u32)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("2^num_vars").into_akita())?;
    if !total_field_elements.is_multiple_of(onehot_k) || !total_field_elements.is_multiple_of(D) {
        return Err(MetalCommitError::UnsupportedShape(format!(
            "field length {total_field_elements} is incompatible with K={onehot_k} and D={D}"
        ))
        .into_akita());
    }
    let chunks_per_source = total_field_elements / onehot_k;
    if first.indices().len() != chunks_per_source {
        return Err(AkitaError::InvalidSize {
            expected: chunks_per_source,
            actual: first.indices().len(),
        });
    }
    for source in &sources[1..] {
        if source.num_vars() != first.num_vars()
            || source.onehot_k() != onehot_k
            || source.indices().len() != chunks_per_source
        {
            return Err(MetalCommitError::UnsupportedShape(
                "one-hot commit group sources must have one shape".into(),
            )
            .into_akita());
        }
    }

    let total_ring_elements = total_field_elements / D;
    let num_blocks = total_ring_elements.div_ceil(plan.num_positions_per_block);
    let output_coefficients_per_source = num_blocks
        .checked_mul(plan.n_a)
        .and_then(|count| count.checked_mul(D))
        .ok_or_else(|| {
            MetalCommitError::ShapeOverflow("output coefficients per source").into_akita()
        })?;
    let active_a_cols = plan
        .num_positions_per_block
        .checked_mul(plan.num_digits_inner)
        .ok_or_else(|| MetalCommitError::ShapeOverflow("active A columns").into_akita())?;
    Ok(ValidatedShape {
        total_field_elements,
        chunks_per_source,
        num_blocks,
        output_coefficients_per_source,
        active_a_cols,
    })
}

fn pack_hot_indices<I: OneHotIndex, const D: usize>(
    sources: &[OneHotView<'_, F, D, I>],
) -> Result<(Vec<u16>, usize), AkitaError> {
    let total_indices = sources
        .len()
        .checked_mul(sources[0].indices().len())
        .ok_or_else(|| MetalCommitError::ShapeOverflow("hot-index count").into_akita())?;
    let mut packed = Vec::with_capacity(total_indices);
    let mut hot_entries = 0usize;
    for source in sources {
        for index in source.indices() {
            let value = match index {
                Some(index) => {
                    let value = index.as_usize();
                    if value >= source.onehot_k() {
                        return Err(AkitaError::InvalidInput(format!(
                            "one-hot index {value} exceeds K={}",
                            source.onehot_k()
                        )));
                    }
                    hot_entries += 1;
                    u16::try_from(value).map_err(|_| {
                        MetalCommitError::UnsupportedShape(
                            "one-hot index exceeds u16 staging width".into(),
                        )
                        .into_akita()
                    })?
                }
                None => NONE_INDEX,
            };
            packed.push(value);
        }
    }
    Ok((packed, hot_entries))
}

pub(crate) fn reconstruct_witnesses<const D: usize>(
    coefficients: Vec<Fp128Limbs>,
    num_sources: usize,
    coefficients_per_source: usize,
) -> Result<Vec<CommitInnerWitness<F>>, MetalCommitError> {
    let expected =
        num_sources
            .checked_mul(coefficients_per_source)
            .ok_or(MetalCommitError::ShapeOverflow(
                "reconstructed output length",
            ))?;
    if coefficients.len() != expected {
        return Err(MetalCommitError::UnsupportedShape(format!(
            "Metal returned {} coefficients, expected {expected}",
            coefficients.len()
        )));
    }
    let mut fields = Vec::with_capacity(coefficients.len());
    for (index, coefficient) in coefficients.into_iter().enumerate() {
        fields.push(coefficient.into_field(index)?);
    }
    let mut witnesses = Vec::with_capacity(num_sources);
    for source_fields in fields.chunks_exact(coefficients_per_source) {
        witnesses.push(CommitInnerWitness {
            inner_rows: RingVec::from_coeffs_with_ring_dim(source_fields.to_vec(), D).map_err(
                |error| {
                    MetalCommitError::UnsupportedShape(format!(
                        "invalid reconstructed ring storage: {error}"
                    ))
                },
            )?,
        });
    }
    Ok(witnesses)
}

pub(crate) fn to_u64(value: usize, name: &'static str) -> Result<u64, AkitaError> {
    u64::try_from(value).map_err(|_| MetalCommitError::ShapeOverflow(name).into_akita())
}

#[cfg(test)]
mod tests {
    use akita_config::{proof_optimized::fp128, CommitmentConfig};
    use akita_prover::{
        AkitaProverSetup, ComputeBackendSetup, CpuBackend, GroupContext, OneHotPoly,
        RootCommitSource, UniformProverStack,
    };
    use akita_types::{PolynomialGroupLayout, SetupMatrixCapacity};

    use super::*;

    fn assert_parity<const D: usize>(onehot_k: usize, num_vars: usize) {
        assert_parity_with_plan::<D>(
            onehot_k,
            num_vars,
            CommitInnerPlan {
                n_a: 2,
                num_positions_per_block: 8,
                num_digits_inner: 3,
                log_basis_inner: 3,
            },
        );
    }

    fn assert_parity_with_plan<const D: usize>(
        onehot_k: usize,
        num_vars: usize,
        plan: CommitInnerPlan,
    ) -> MetalCommitMetrics {
        let chunks = (1usize << num_vars) / onehot_k;
        let make_indices = |salt: usize| {
            (0..chunks)
                .map(|chunk| {
                    if (chunk + salt).is_multiple_of(11) {
                        None
                    } else {
                        Some(((chunk * 37 + salt) % onehot_k) as u16)
                    }
                })
                .collect::<Vec<_>>()
        };
        let mut first = make_indices(0);
        first[0] = Some(0);
        first[1] = Some((D - 1).min(onehot_k - 1) as u16);
        let polys = [
            OneHotPoly::<F, u16>::new(onehot_k, first).unwrap(),
            OneHotPoly::<F, u16>::new(onehot_k, make_indices(5)).unwrap(),
        ];
        let matrix_fields = plan.n_a * plan.num_positions_per_block * plan.num_digits_inner * D;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            num_vars,
            polys.len(),
            SetupMatrixCapacity {
                num_field_elements: matrix_fields,
            },
        )
        .unwrap();
        let cpu_prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();

        let cpu_views = polys
            .iter()
            .map(|poly| <OneHotPoly<F, u16> as RootCommitSource<F, D>>::commit_view(poly).unwrap())
            .collect();
        let metal_views = polys
            .iter()
            .map(|poly| <OneHotPoly<F, u16> as RootCommitSource<F, D>>::commit_view(poly).unwrap())
            .collect();
        let cpu = CpuBackend::DEFAULT
            .commit_inner_group(&cpu_prepared, cpu_views, plan)
            .unwrap();
        let gpu = metal
            .commit_inner_group(&metal_prepared, metal_views, plan)
            .unwrap();
        assert_eq!(cpu.len(), gpu.len());
        for (cpu, gpu) in cpu.iter().zip(&gpu) {
            assert_eq!(cpu.inner_rows, gpu.inner_rows);
        }

        let second_views = polys
            .iter()
            .map(|poly| <OneHotPoly<F, u16> as RootCommitSource<F, D>>::commit_view(poly).unwrap())
            .collect();
        let second = metal
            .commit_inner_group(&metal_prepared, second_views, plan)
            .unwrap();
        for (cpu, gpu) in cpu.iter().zip(&second) {
            assert_eq!(cpu.inner_rows, gpu.inner_rows);
        }
        let metrics = metal.last_commit_metrics().unwrap().unwrap();
        assert!(metrics.matrix_cache_hit);
        metrics
    }

    #[test]
    fn parity_d64_k16() {
        assert_parity::<64>(16, 12);
    }

    #[test]
    fn parity_d64_k256() {
        assert_parity::<64>(256, 12);
    }

    #[test]
    fn parity_d128_k64() {
        assert_parity::<128>(64, 13);
    }

    #[test]
    fn parity_d128_k128() {
        assert_parity::<128>(128, 13);
    }

    #[test]
    fn parity_d256_k256() {
        assert_parity::<256>(256, 14);
    }

    #[test]
    fn parity_block_batched_d256_k256() {
        let metrics = assert_parity_with_plan::<256>(
            256,
            22,
            CommitInnerPlan {
                n_a: 2,
                num_positions_per_block: 8,
                num_digits_inner: 3,
                log_basis_inner: 3,
            },
        );
        assert_eq!(metrics.kernel, MetalOneHotKernel::BlockBatched);
    }

    #[test]
    fn parity_segmented_block_batched_d64_k16() {
        let metrics = assert_parity_with_plan::<64>(
            16,
            22,
            CommitInnerPlan {
                n_a: 1,
                num_positions_per_block: 16_384,
                num_digits_inner: 1,
                log_basis_inner: 3,
            },
        );
        assert_eq!(metrics.kernel, MetalOneHotKernel::BlockBatched);
    }

    #[test]
    fn full_commit_matches_cpu() {
        type Cfg = fp128::OneHot;

        let num_vars = 16;
        let chunks = (1usize << num_vars) / 256;
        let indices = (0..chunks)
            .map(|chunk| Some((chunk.wrapping_mul(37) % 256) as u8))
            .collect();
        let polys = [OneHotPoly::<F, u8>::new(256, indices).unwrap()];
        let profile = Cfg::profile_without_precommitted_groups(PolynomialGroupLayout::new(
            num_vars,
            polys.len(),
        ))
        .unwrap();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            num_vars,
            polys.len(),
            SetupMatrixCapacity {
                num_field_elements: 1 << 20,
            },
        )
        .unwrap();

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let cpu_stack =
            UniformProverStack::uniform(&cpu, &cpu_prepared, setup.expanded.as_ref()).unwrap();
        let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        let metal_stack =
            UniformProverStack::uniform(&metal, &metal_prepared, setup.expanded.as_ref()).unwrap();
        let context = GroupContext::explicit(&profile);
        let cpu_output = akita_prover::commit::<Cfg, OneHotPoly<F, u8>, CpuBackend>(
            &polys,
            setup.expanded.as_ref(),
            &cpu_stack,
            context,
        )
        .unwrap();
        let metal_output = akita_prover::commit::<Cfg, OneHotPoly<F, u8>, MetalBackend>(
            &polys,
            setup.expanded.as_ref(),
            &metal_stack,
            context,
        )
        .unwrap();

        assert_eq!(cpu_output.committed_group, metal_output.committed_group);
        assert_eq!(cpu_output.hint, metal_output.hint);
    }
}
