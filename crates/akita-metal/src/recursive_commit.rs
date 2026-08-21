use akita_algebra::CyclotomicRing;
use akita_field::AkitaError;
use akita_prover::compute::{CommitInnerPlan, RootCommitKernel};
use akita_prover::{CommitInnerWitness, SuffixWitnessView};

use crate::field::{MetalField, F};
use crate::runtime::{RecursiveCommitParams, FP128_D512_LINEAR_RELATION_NUM_PRIMES};
use crate::{MetalCommitBackend, MetalCommitError, MetalExecutionPolicy, MetalPreparedSetup};

const BLOCKS_PER_GROUP: usize = 16;
const RHS_ABS_BOUND: u64 = 128;

impl<const D: usize> RootCommitKernel<SuffixWitnessView<'_, F, D>, F, D> for MetalCommitBackend<F> {
    #[tracing::instrument(skip_all, name = "MetalCommitBackend::recursive_witness_commit_inner")]
    fn commit_inner_group(
        &self,
        prepared: &MetalPreparedSetup<F>,
        sources: Vec<SuffixWitnessView<'_, F, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        let [source] = sources.as_slice() else {
            return self.unsupported_recursive_or_cpu(
                prepared,
                sources,
                plan,
                "recursive Metal commitment requires one source".into(),
            );
        };
        let num_blocks = source
            .live_ring_elems()
            .div_ceil(plan.num_positions_per_block);
        let expected_rings = num_blocks
            .checked_mul(plan.num_positions_per_block)
            .ok_or_else(|| {
                MetalCommitError::ShapeOverflow("recursive commit source rings").into_akita()
            })?;
        let expected_source_bytes = expected_rings.checked_mul(D).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("recursive commit source bytes").into_akita()
        })?;
        let supported_shape = plan.num_digits_inner == 1
            && plan.num_positions_per_block != 0
            && source.committed_i8_digits().len() >= expected_source_bytes;
        let Some(runtime) = self.runtime() else {
            return self.unsupported_recursive_or_cpu(
                prepared,
                sources,
                plan,
                "no Metal runtime is available".into(),
            );
        };
        if !supported_shape
            || !runtime.supports_fp128_recursive_commit::<D>(
                num_blocks,
                plan.n_a,
                plan.num_positions_per_block,
                RHS_ABS_BOUND,
            )
        {
            return self.unsupported_recursive_or_cpu(
                prepared,
                sources,
                plan,
                format!(
                    "recursive commitment D={D}, blocks={num_blocks}, rows={}, columns={}, digits={} is unsupported",
                    plan.n_a, plan.num_positions_per_block, plan.num_digits_inner,
                ),
            );
        }

        let matrix = prepared.shared_matrix(runtime, D, plan.n_a, plan.num_positions_per_block)?;
        let output_coefficients = num_blocks
            .checked_mul(plan.n_a)
            .and_then(|count| count.checked_mul(D))
            .ok_or_else(|| {
                MetalCommitError::ShapeOverflow("recursive commit output coefficients").into_akita()
            })?;
        let outcome = runtime
            .dispatch_fp128_recursive_commit::<D>(
                &matrix.buffer,
                source.committed_i8_digits(),
                RecursiveCommitParams {
                    num_blocks: num_blocks as u64,
                    blocks_per_group: BLOCKS_PER_GROUP as u64,
                    num_block_groups: num_blocks.div_ceil(BLOCKS_PER_GROUP) as u64,
                    num_rows: plan.n_a as u64,
                    num_cols: plan.num_positions_per_block as u64,
                    ring_d: D as u64,
                    num_primes: FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64,
                    matrix_rings: plan.n_a.saturating_mul(plan.num_positions_per_block) as u64,
                    output_coefficients: output_coefficients as u64,
                    rhs_abs_bound: RHS_ABS_BOUND,
                },
            )
            .map_err(MetalCommitError::into_akita)?;
        let timings = outcome.timings;
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.upload_time += timings.buffer_setup + matrix.prepare_time;
            metrics.readback_time += timings.readback_copy;
            metrics.allocation_bytes = metrics
                .allocation_bytes
                .saturating_add(outcome.allocation_bytes)
                .saturating_add(matrix.bytes.saturating_mul(usize::from(!matrix.zero_copy)));
        })
        .map_err(MetalCommitError::into_akita)?;

        let coefficients = outcome
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(index, value)| F::from_device(value, index))
            .collect::<Result<Vec<_>, _>>()
            .map_err(MetalCommitError::into_akita)?;
        let rings = coefficients
            .chunks_exact(D)
            .map(CyclotomicRing::<F, D>::from_slice)
            .collect::<Vec<_>>();
        let rows = rings
            .chunks_exact(plan.n_a)
            .map(<[_]>::to_vec)
            .collect::<Vec<_>>();
        Ok(vec![CommitInnerWitness::from_rows(rows)])
    }
}

impl MetalCommitBackend<F> {
    fn unsupported_recursive_or_cpu<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup<F>,
        sources: Vec<SuffixWitnessView<'_, F, D>>,
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

#[cfg(test)]
mod tests {
    use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend, RecursiveWitnessFlat};
    use akita_types::SetupMatrixCapacity;

    use super::*;

    fn assert_recursive_commit_matches_cpu<const D: usize>() {
        const ROWS: usize = 5;
        const COLUMNS: usize = 8;
        const BLOCKS: usize = 3;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            16,
            1,
            SetupMatrixCapacity {
                num_field_elements: ROWS * COLUMNS * D,
            },
        )
        .unwrap();
        let digits = (0..BLOCKS * COLUMNS * D)
            .map(|index| ((index * 5 + 3) % 8) as i8 - 4)
            .collect::<Vec<_>>();
        let witness = RecursiveWitnessFlat::from_i8_digits(digits);
        let plan = CommitInnerPlan {
            n_a: ROWS,
            num_positions_per_block: COLUMNS,
            num_digits_inner: 1,
            log_basis_inner: 3,
        };

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let expected = cpu
            .commit_inner_group(&cpu_prepared, vec![witness.view::<F, D>().unwrap()], plan)
            .unwrap();
        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        metal.begin_opening_metrics().unwrap();
        let actual = metal
            .commit_inner_group(&metal_prepared, vec![witness.view::<F, D>().unwrap()], plan)
            .unwrap();
        assert_eq!(actual[0].inner_rows, expected[0].inner_rows);
        let metrics = metal.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.cpu_fallback_calls, 0);
        assert!(metrics.gpu_active_time > std::time::Duration::ZERO);
    }

    #[test]
    fn recursive_commit_matches_cpu_d64() {
        assert_recursive_commit_matches_cpu::<64>();
    }

    #[test]
    fn recursive_commit_matches_cpu_d128() {
        assert_recursive_commit_matches_cpu::<128>();
    }
}
