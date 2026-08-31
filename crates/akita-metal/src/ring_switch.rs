use akita_algebra::CyclotomicRing;
use akita_field::AkitaError;
use akita_prover::backend::RingSwitchRelationView;
use akita_prover::compute::{
    RingSwitchRelationKernel, RingSwitchRelationPlan, RingSwitchRelationRows,
};

use crate::field::{MetalField, F};
use crate::runtime::D512LinearRelationParams;
use crate::{MetalCommitBackend, MetalCommitError, MetalPreparedSetup};

impl<const D: usize> RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D>
    for MetalCommitBackend<F>
{
    fn relation_rows(
        &self,
        prepared: &MetalPreparedSetup<F>,
        source: RingSwitchRelationView<'_, D>,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError> {
        let rhs_abs_bound = source
            .z_segment
            .iter()
            .flat_map(|row| row.iter())
            .map(|value| u64::from(value.unsigned_abs()))
            .max()
            .unwrap_or(0)
            .max(u64::from(source.z_folded_centered_inf_norm));
        let use_metal = D == 512
            && plan.n_d == 0
            && plan.n_b == 0
            && plan.n_a == 1
            && source.e_hat.is_empty()
            && source.t_hat.is_empty()
            && !source.z_segment.is_empty()
            && self.runtime().is_some_and(|runtime| {
                runtime.supports_fp128_d512_linear_relation(source.z_segment.len(), rhs_abs_bound)
            });
        if !use_metal {
            let work_units = plan
                .n_d
                .saturating_mul(source.e_hat.len())
                .saturating_add(plan.n_b.saturating_mul(source.t_hat.len()))
                .saturating_add(plan.n_a.saturating_mul(source.z_segment.len()))
                .saturating_mul(D);
            self.record_opening_cpu_fallback(work_units)
                .map_err(MetalCommitError::into_akita)?;
            return self
                .cpu_backend()
                .relation_rows(&prepared.cpu, source, plan);
        }

        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let matrix = prepared.shared_matrix(runtime, 512, 1, source.z_segment.len())?;
        let num_tiles = source.z_segment.len().div_ceil(64);
        let outcome = runtime
            .dispatch_fp128_d512_linear_relation(
                &matrix.buffer,
                source.z_segment,
                D512LinearRelationParams {
                    num_columns: source.z_segment.len() as u64,
                    columns_per_tile: 64,
                    num_tiles: num_tiles as u64,
                    num_primes: 6,
                    ntt_size: 1_024,
                    output_coefficients: D as u64,
                    rhs_abs_bound,
                },
            )
            .map_err(MetalCommitError::into_akita)?;
        let timings = outcome.timings;
        let coefficients = outcome
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(index, value)| F::from_device(value, index))
            .collect::<Result<Vec<_>, _>>()
            .map_err(MetalCommitError::into_akita)?;
        if coefficients.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: coefficients.len(),
            });
        }
        let quotient = CyclotomicRing::from_slice(&coefficients);
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
        Ok(RingSwitchRelationRows {
            d_negacyclic: Vec::new(),
            d_cyclic: Vec::new(),
            b_cyclic: Vec::new(),
            a_quotients: vec![quotient],
        })
    }
}

#[cfg(test)]
mod tests {
    use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend};
    use akita_types::SetupMatrixCapacity;

    use super::*;
    use crate::MetalExecutionPolicy;

    #[test]
    fn d512_linear_relation_matches_cpu_across_tiles() {
        const D: usize = 512;
        const COLUMNS: usize = 67;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            20,
            1,
            SetupMatrixCapacity {
                num_field_elements: COLUMNS * D,
            },
        )
        .unwrap();
        let z_segment: Vec<[i32; D]> = (0..COLUMNS)
            .map(|column| {
                std::array::from_fn(|coefficient| {
                    const VALUES: [i32; 11] = [-9, -4, -2, -1, 0, 1, 2, 3, 5, 8, 13];
                    VALUES[(column * 17 + coefficient * 7) % VALUES.len()]
                })
            })
            .collect::<Vec<_>>();
        let source = RingSwitchRelationView {
            e_hat: &[],
            t_hat: &[],
            z_segment: &z_segment,
            z_folded_centered_inf_norm: 13,
        };
        let plan = RingSwitchRelationPlan {
            n_d: 0,
            n_b: 0,
            n_a: 1,
            log_basis_open: 3,
            log_basis_outer: 3,
        };

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let expected = cpu.relation_rows(&cpu_prepared, source, plan).unwrap();

        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        metal.begin_opening_metrics().unwrap();
        let actual = metal.relation_rows(&metal_prepared, source, plan).unwrap();
        assert_eq!(actual, expected);
        let metrics = metal.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.cpu_fallback_calls, 0);
        assert!(metrics.gpu_active_time > std::time::Duration::ZERO);
    }
}
