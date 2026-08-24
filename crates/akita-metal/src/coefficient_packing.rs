use akita_field::AkitaError;
use akita_prover::compute::{
    RootOpeningSource, RootPolyMeta, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use akita_prover::{
    PackedOneHotView, RecursiveFoldBatchView, RecursiveFoldSource, RecursiveWitnessFlat,
    SuffixWitnessBatchView,
};
use akita_types::PreparedSubringCoefficientPackingPoint;

use crate::backend::MetalCommitBackend;
use crate::field::{Fp128Limbs, MetalField, F};
use crate::packed_onehot_fp128_d512::{opening_index_plan, DeferredCoefficientPackingIndex};
use crate::runtime::{
    CoefficientPackingDispatchOutcome, I8CoefficientPackingParams,
    PackedFp128D512CoefficientPackingIndex, PackedFp128D512CoefficientPackingSource,
    PackedOneHotCoefficientPackingParams, FP128_PACKED_COEFFICIENT_PACKING_ROWS_PER_PARTIAL,
};
use crate::{MetalCommitError, MetalExecutionPolicy};

fn checked_u64(value: usize, label: &'static str) -> Result<u64, AkitaError> {
    u64::try_from(value).map_err(|_| MetalCommitError::ShapeOverflow(label).into_akita())
}

fn combined_weights(
    point: &PreparedSubringCoefficientPackingPoint<F>,
) -> Result<Vec<Fp128Limbs>, AkitaError> {
    let geometry = point.geometry();
    if geometry.extension_degree() != 1
        || point.position_weights().len() != point.num_positions_per_block()
        || point.packing_weights().len() != geometry.subring_embedding_stride()
    {
        return Err(AkitaError::InvalidSetup(
            "fp128 Metal coefficient packing requires base-field weights".into(),
        ));
    }
    let count = point
        .num_positions_per_block()
        .checked_mul(geometry.subring_embedding_stride())
        .ok_or_else(|| AkitaError::InvalidInput("coefficient-packing weight overflow".into()))?;
    let mut weights = Vec::with_capacity(count);
    for &position in point.position_weights() {
        for &packing in point.packing_weights() {
            weights.push((position * packing).into_device());
        }
    }
    Ok(weights)
}

fn decode_outcome(
    backend: &MetalCommitBackend<F>,
    outcome: CoefficientPackingDispatchOutcome,
) -> Result<Vec<F>, AkitaError> {
    let coefficients = outcome
        .coefficients
        .into_iter()
        .enumerate()
        .map(|(index, value)| F::from_device(value, index))
        .collect::<Result<Vec<_>, _>>()
        .map_err(MetalCommitError::into_akita)?;
    backend
        .update_opening_metrics(|metrics| {
            metrics.command_wall_time += outcome.timings.command_wall;
            metrics.gpu_active_time += outcome.timings.gpu.unwrap_or_default();
            metrics.upload_time += outcome.timings.buffer_setup;
            metrics.readback_time += outcome.timings.readback_copy;
            metrics.allocation_bytes = metrics
                .allocation_bytes
                .saturating_add(outcome.allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)?;
    Ok(coefficients)
}

impl<const D: usize> SubringCoefficientPackingBatchKernel<SuffixWitnessBatchView<'_, F, D>, F, F, D>
    for MetalCommitBackend<F>
{
    #[tracing::instrument(skip_all, name = "MetalCommitBackend::suffix_coefficient_packing")]
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, F>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let witnesses = source.witnesses();
        if witnesses.is_empty() {
            return Ok(Vec::new());
        }
        let mut views = Vec::with_capacity(witnesses.len());
        for witness in witnesses {
            plan.validate::<D>(RootPolyMeta::<F>::num_vars(*witness))?;
            let view = witness.view::<F, D>()?;
            if view.live_ring_elems() != plan.point.num_live_positions() {
                return Err(AkitaError::InvalidSize {
                    expected: plan.point.num_live_positions(),
                    actual: view.live_ring_elems(),
                });
            }
            views.push(view);
        }
        let source_coefficients = views[0].committed_i8_digits().len();
        let live_coefficients = views[0].live_coeff_len();
        if views.iter().any(|view| {
            view.committed_i8_digits().len() != source_coefficients
                || view.live_coeff_len() != live_coefficients
        }) {
            return self.cpu_backend().coefficient_packing_partials_batch(
                prepared.map(|value| &value.cpu),
                source,
                plan,
            );
        }
        let Some(runtime) = self.runtime() else {
            return match self.policy() {
                MetalExecutionPolicy::RequireMetal => {
                    Err(MetalCommitError::DeviceUnavailable.into_akita())
                }
                MetalExecutionPolicy::PreferMetal => {
                    self.cpu_backend().coefficient_packing_partials_batch(
                        prepared.map(|value| &value.cpu),
                        source,
                        plan,
                    )
                }
            };
        };
        let geometry = plan.point.geometry();
        let weights = combined_weights(plan.point)?;
        let sources = views
            .iter()
            .map(|view| view.committed_i8_digits())
            .collect::<Vec<_>>();
        let output_coefficients = witnesses
            .len()
            .checked_mul(plan.point.num_live_blocks())
            .and_then(|count| count.checked_mul(geometry.partial_base_field_width()))
            .ok_or_else(|| {
                AkitaError::InvalidInput("coefficient-packing output overflow".into())
            })?;
        let outcome = runtime
            .dispatch_fp128_i8_coefficient_packing(
                &sources,
                &weights,
                I8CoefficientPackingParams {
                    num_sources: checked_u64(witnesses.len(), "coefficient-packing sources")?,
                    source_coefficients: checked_u64(
                        source_coefficients,
                        "coefficient-packing source width",
                    )?,
                    live_coefficients: checked_u64(
                        live_coefficients,
                        "coefficient-packing live width",
                    )?,
                    num_live_positions: checked_u64(
                        plan.point.num_live_positions(),
                        "coefficient-packing positions",
                    )?,
                    positions_per_block: checked_u64(
                        plan.point.num_positions_per_block(),
                        "coefficient-packing block positions",
                    )?,
                    num_blocks: checked_u64(
                        plan.point.num_live_blocks(),
                        "coefficient-packing blocks",
                    )?,
                    ring_d: checked_u64(D, "coefficient-packing ring dimension")?,
                    stride: checked_u64(
                        geometry.subring_embedding_stride(),
                        "coefficient-packing stride",
                    )?,
                    subring_dimension: checked_u64(
                        geometry.challenge_subring_dimension(),
                        "coefficient-packing subring dimension",
                    )?,
                    output_coefficients: checked_u64(
                        output_coefficients,
                        "coefficient-packing output",
                    )?,
                },
            )
            .map_err(MetalCommitError::into_akita)?;
        let coefficients = decode_outcome(self, outcome)?;
        let per_source = output_coefficients / witnesses.len();
        coefficients
            .chunks_exact(per_source)
            .map(|coordinates| {
                SubringCoefficientPackingPartials::new(
                    geometry,
                    plan.point.num_live_blocks(),
                    coordinates.to_vec(),
                )
            })
            .collect()
    }
}

impl<const D: usize> SubringCoefficientPackingBatchKernel<RecursiveFoldBatchView<'_, F, D>, F, F, D>
    for MetalCommitBackend<F>
{
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, F>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let mut witnesses = Vec::with_capacity(source.sources().len());
        for scheduled in source.sources() {
            let RecursiveFoldSource::Witness(witness) = scheduled else {
                self.record_opening_cpu_fallback(1)
                    .map_err(MetalCommitError::into_akita)?;
                return self.cpu_backend().coefficient_packing_partials_batch(
                    prepared.map(|value| &value.cpu),
                    source,
                    plan,
                );
            };
            witnesses.push(witness.as_ref());
        }
        let batch = <RecursiveWitnessFlat as RootOpeningSource<F, D>>::opening_batch(&witnesses)?;
        <Self as SubringCoefficientPackingBatchKernel<
            SuffixWitnessBatchView<'_, F, D>,
            F,
            F,
            D,
        >>::coefficient_packing_partials_batch(self, prepared, batch, plan)
    }
}

impl MetalCommitBackend<F> {
    /// Use the packed-source Metal route when the source owns opening state.
    pub fn retained_packed_onehot_coefficient_packing<const D: usize>(
        &self,
        source: PackedOneHotView<'_, F, D>,
        point: &PreparedSubringCoefficientPackingPoint<F>,
    ) -> Result<Option<SubringCoefficientPackingPartials<F>>, AkitaError> {
        if source
            .opening_acceleration::<PackedFp128D512CoefficientPackingIndex>()
            .is_none()
            && source
                .opening_acceleration::<DeferredCoefficientPackingIndex>()
                .is_none()
        {
            return Ok(None);
        }
        self.packed_onehot_coefficient_packing(source, point)
            .map(Some)
    }

    /// Project a row-major packed one-hot source into coefficient-packing partials.
    pub fn packed_onehot_coefficient_packing<const D: usize>(
        &self,
        source: PackedOneHotView<'_, F, D>,
        point: &PreparedSubringCoefficientPackingPoint<F>,
    ) -> Result<SubringCoefficientPackingPartials<F>, AkitaError> {
        let total_field_elements = source
            .num_rows()
            .checked_mul(source.column_capacity())
            .and_then(|count| count.checked_mul(source.onehot_k()))
            .ok_or_else(|| AkitaError::InvalidInput("packed source length overflow".into()))?;
        if !total_field_elements.is_power_of_two() {
            return Err(AkitaError::InvalidSize {
                expected: total_field_elements.next_power_of_two(),
                actual: total_field_elements,
            });
        }
        SubringCoefficientPackingPlan { point }
            .validate::<D>(total_field_elements.trailing_zeros() as usize)?;
        let block_field_elements = point
            .num_positions_per_block()
            .checked_mul(D)
            .ok_or_else(|| AkitaError::InvalidInput("packing block width overflow".into()))?;
        let segment_field_elements = source
            .num_rows()
            .checked_mul(source.onehot_k())
            .ok_or_else(|| AkitaError::InvalidInput("packing segment width overflow".into()))?;
        if !segment_field_elements.is_multiple_of(block_field_elements)
            || !block_field_elements.is_multiple_of(source.onehot_k())
        {
            return Err(MetalCommitError::UnsupportedShape(
                "packed Metal coefficient packing requires row-aligned blocks".into(),
            )
            .into_akita());
        }
        let blocks_per_column = segment_field_elements / block_field_elements;
        let rows_per_block = block_field_elements / source.onehot_k();
        let num_blocks = source
            .column_capacity()
            .checked_mul(blocks_per_column)
            .ok_or_else(|| AkitaError::InvalidInput("packing block count overflow".into()))?;
        if num_blocks != point.num_live_blocks()
            || point.num_live_positions().checked_mul(D) != Some(total_field_elements)
        {
            return Err(AkitaError::InvalidSetup(
                "packed source disagrees with coefficient-packing point".into(),
            ));
        }
        let Some(runtime) = self.runtime() else {
            return Err(MetalCommitError::DeviceUnavailable.into_akita());
        };
        let geometry = point.geometry();
        let weights = combined_weights(point)?;
        let retained_index =
            source.take_opening_acceleration::<PackedFp128D512CoefficientPackingIndex>();
        let deferred_requested = source
            .take_opening_acceleration::<DeferredCoefficientPackingIndex>()
            .is_some();
        let deferred_params = if retained_index.is_none() && deferred_requested {
            Some(
                opening_index_plan(
                    source.num_rows(),
                    source.num_columns(),
                    point.num_positions_per_block(),
                    blocks_per_column,
                )?
                .packing,
            )
        } else {
            None
        };
        let packing_source = retained_index
            .as_deref()
            .map(PackedFp128D512CoefficientPackingSource::Retained)
            .or_else(|| deferred_params.map(PackedFp128D512CoefficientPackingSource::Fused));
        let rows_per_partial = FP128_PACKED_COEFFICIENT_PACKING_ROWS_PER_PARTIAL;
        let row_partials_per_block = rows_per_block.div_ceil(rows_per_partial);
        let output_coefficients = num_blocks
            .checked_mul(geometry.partial_base_field_width())
            .ok_or_else(|| AkitaError::InvalidInput("packing output overflow".into()))?;
        let partial_coefficients = num_blocks
            .checked_mul(row_partials_per_block)
            .and_then(|count| count.checked_mul(geometry.challenge_subring_dimension()))
            .ok_or_else(|| AkitaError::InvalidInput("packing partial overflow".into()))?;
        let params = PackedOneHotCoefficientPackingParams {
            num_rows: checked_u64(source.num_rows(), "packing rows")?,
            num_columns: checked_u64(source.num_columns(), "packing columns")?,
            column_capacity: checked_u64(source.column_capacity(), "packing column capacity")?,
            onehot_k: checked_u64(source.onehot_k(), "packing one-hot K")?,
            ring_d: checked_u64(D, "packing ring dimension")?,
            positions_per_block: checked_u64(
                point.num_positions_per_block(),
                "packing block positions",
            )?,
            blocks_per_column: checked_u64(blocks_per_column, "packing blocks per column")?,
            rows_per_block: checked_u64(rows_per_block, "packing rows per block")?,
            rows_per_partial: checked_u64(rows_per_partial, "packing rows per partial")?,
            row_partials_per_block: checked_u64(row_partials_per_block, "packing row partials")?,
            num_blocks: checked_u64(num_blocks, "packing blocks")?,
            stride: checked_u64(geometry.subring_embedding_stride(), "packing stride")?,
            subring_dimension: checked_u64(
                geometry.challenge_subring_dimension(),
                "packing subring dimension",
            )?,
            output_coefficients: checked_u64(output_coefficients, "packing output")?,
            partial_coefficients: checked_u64(partial_coefficients, "packing partial output")?,
        };
        let outcome = if let Some(packing_source) = packing_source {
            runtime.dispatch_fp128_bucketed_packed_onehot_coefficient_packing(
                source.lanes(),
                packing_source,
                &weights,
                params,
            )
        } else {
            runtime.dispatch_fp128_packed_onehot_coefficient_packing(
                source.lanes(),
                &weights,
                params,
            )
        }
        .map_err(MetalCommitError::into_akita)?;
        drop(retained_index);
        SubringCoefficientPackingPartials::new(geometry, num_blocks, decode_outcome(self, outcome)?)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use akita_prover::compute::{
        CpuBackend, RootOpeningSource, SubringCoefficientPackingBatchKernel,
        SubringCoefficientPackingPlan,
    };
    use akita_prover::{PackedOneHotPoly, PackedOneHotView, RecursiveWitnessFlat};
    use akita_types::{
        BasisMode, PreparedSubringCoefficientPackingPoint, SubringCoefficientPackingGeometry,
    };

    use super::*;
    use crate::OpeningAccelerationPolicy;

    fn point_with_subring<const D: usize>(
        live_positions: usize,
        positions_per_block: usize,
        source_num_vars: usize,
        subring_dimension: usize,
    ) -> PreparedSubringCoefficientPackingPoint<F> {
        let geometry = SubringCoefficientPackingGeometry::try_new(1, D, subring_dimension).unwrap();
        let public = (0..source_num_vars)
            .map(|index| F::from_u64((index + 3) as u64))
            .collect::<Vec<_>>();
        PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            live_positions,
            positions_per_block,
            source_num_vars,
            &public,
        )
        .unwrap()
    }

    fn point<const D: usize>(
        live_positions: usize,
        positions_per_block: usize,
        source_num_vars: usize,
    ) -> PreparedSubringCoefficientPackingPoint<F> {
        point_with_subring::<D>(live_positions, positions_per_block, source_num_vars, 64)
    }

    #[test]
    fn suffix_coefficient_packing_matches_cpu() {
        const D: usize = 64;
        let digits = (0..8 * D)
            .map(|index| (index % 8) as i8 - 4)
            .collect::<Vec<_>>();
        let witness = RecursiveWitnessFlat::from_i8_digits(digits);
        let witnesses = [&witness];
        let batch =
            <RecursiveWitnessFlat as RootOpeningSource<F, D>>::opening_batch(&witnesses).unwrap();
        let point = point::<D>(8, 4, 9);
        let plan = SubringCoefficientPackingPlan { point: &point };
        let expected = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(None, batch, plan)
            .unwrap();
        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        metal.begin_opening_metrics().unwrap();
        let actual = metal
            .coefficient_packing_partials_batch(None, batch, plan)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            metal
                .last_opening_metrics()
                .unwrap()
                .unwrap()
                .cpu_fallback_calls,
            0
        );
    }

    #[test]
    fn packed_onehot_coefficient_packing_matches_definition() {
        const D: usize = 256;
        const ROWS: usize = 8;
        const COLUMNS: usize = 3;
        const CAPACITY: usize = 4;
        const K: usize = 256;
        let lanes = (0..ROWS * COLUMNS)
            .map(|index| ((index * 19 + 7) % (K - 1) + 1) as u8)
            .collect::<Vec<_>>();
        let source = PackedOneHotView::<F, D>::new(K, CAPACITY, COLUMNS, &lanes).unwrap();
        let point = point::<D>(ROWS * CAPACITY, ROWS, 13);
        let geometry = point.geometry();
        let mut expected = vec![F::zero(); CAPACITY * geometry.partial_base_field_width()];
        for column in 0..COLUMNS {
            for row in 0..ROWS {
                let hot = usize::from(lanes[row * COLUMNS + column]);
                let coefficient = hot % D;
                let subring = coefficient / geometry.subring_embedding_stride();
                let low = coefficient % geometry.subring_embedding_stride();
                expected[column * geometry.partial_base_field_width() + subring] +=
                    point.position_weights()[row] * point.packing_weights()[low];
            }
        }
        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        metal.begin_opening_metrics().unwrap();
        let actual = metal
            .packed_onehot_coefficient_packing(source, &point)
            .unwrap();
        assert_eq!(actual.coordinates(), expected);
        assert_eq!(actual.num_live_blocks(), CAPACITY);
    }

    #[test]
    fn packed_onehot_coefficient_packing_reduces_row_partials() {
        const D: usize = 512;
        const ROWS: usize = 16_384;
        const K: usize = 256;
        let lanes = (0..ROWS)
            .map(|row| ((row * 19 + 7) % (K - 1) + 1) as u8)
            .collect::<Vec<_>>();
        let source = PackedOneHotView::<F, D>::new(K, 1, 1, &lanes).unwrap();
        let point = point_with_subring::<D>(ROWS / 2, ROWS / 2, 22, 256);
        let geometry = point.geometry();
        let mut expected = vec![F::zero(); geometry.partial_base_field_width()];
        for (row, &hot) in lanes.iter().enumerate() {
            let field_index = row * K + usize::from(hot);
            let position = field_index / D;
            let coefficient = field_index % D;
            let subring = coefficient / geometry.subring_embedding_stride();
            let low = coefficient % geometry.subring_embedding_stride();
            expected[subring] += point.position_weights()[position] * point.packing_weights()[low];
        }
        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        metal.begin_opening_metrics().unwrap();
        let actual = metal
            .packed_onehot_coefficient_packing(source, &point)
            .unwrap();
        assert_eq!(actual.coordinates(), expected);
        assert_eq!(actual.num_live_blocks(), 1);
    }

    #[test]
    fn packed_onehot_atomic_tile_matches_maximal_bucket_load() {
        const D: usize = 256;
        const ROWS: usize = FP128_PACKED_COEFFICIENT_PACKING_ROWS_PER_PARTIAL;
        const K: usize = 256;
        let lanes = vec![7; ROWS];
        let source = PackedOneHotView::<F, D>::new(K, 1, 1, &lanes).unwrap();
        let point = point::<D>(ROWS, ROWS, 23);
        let geometry = point.geometry();
        let mut expected = vec![F::zero(); geometry.partial_base_field_width()];
        for row in 0..ROWS {
            let field_index = row * K + 7;
            let position = field_index / D;
            let coefficient = field_index % D;
            let subring = coefficient / geometry.subring_embedding_stride();
            let low = coefficient % geometry.subring_embedding_stride();
            expected[subring] += point.position_weights()[position] * point.packing_weights()[low];
        }
        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let actual = metal
            .packed_onehot_coefficient_packing(source, &point)
            .unwrap();
        assert_eq!(actual.coordinates(), expected);
    }

    fn indexed_packing_case(
        lanes: Vec<u8>,
        num_rows: usize,
        num_columns: usize,
        column_capacity: usize,
        positions_per_block: usize,
        deferred: bool,
    ) {
        const D: usize = 512;
        const K: usize = 256;
        let source = PackedOneHotPoly::<F>::new(K, column_capacity, num_columns, lanes).unwrap();
        let view = source.view::<D>().unwrap();
        let blocks_per_column = num_rows / (positions_per_block * 2);
        let num_blocks = blocks_per_column * column_capacity;
        let source_num_vars = (num_rows * column_capacity * K).ilog2() as usize;
        let point = point::<D>(
            num_blocks * positions_per_block,
            positions_per_block,
            source_num_vars,
        );
        let geometry = point.geometry();
        let mut expected = vec![F::zero(); num_blocks * geometry.partial_base_field_width()];
        for column in 0..num_columns {
            for row in 0..num_rows {
                let hot = usize::from(view.lanes()[row * num_columns + column]);
                if hot == 0 {
                    continue;
                }
                let field_index = row * K + hot;
                let block = field_index / (positions_per_block * D);
                let within_block = field_index % (positions_per_block * D);
                let position = within_block / D;
                let coefficient = within_block % D;
                let subring = coefficient / geometry.subring_embedding_stride();
                let low = coefficient % geometry.subring_embedding_stride();
                let output_block = column * blocks_per_column + block;
                expected[output_block * geometry.partial_base_field_width() + subring] +=
                    point.position_weights()[position] * point.packing_weights()[low];
            }
        }

        let metal = if deferred {
            MetalCommitBackend::new_with_opening_acceleration_policy(
                MetalExecutionPolicy::RequireMetal,
                OpeningAccelerationPolicy::RetainUpToBytes(0),
            )
            .unwrap()
        } else {
            MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap()
        };
        if deferred {
            assert!(view.retain_opening_acceleration(Arc::new(DeferredCoefficientPackingIndex,)));
        } else {
            let position_tiles = positions_per_block.div_ceil(256);
            let layouts = blocks_per_column * num_columns * 2 * position_tiles;
            let index = metal
                .runtime()
                .unwrap()
                .prepare_packed_fp128_d512_coefficient_packing_index(
                    view.lanes(),
                    crate::runtime::PackedCoefficientPackingIndexParams {
                        num_rows: num_rows as u64,
                        num_columns: num_columns as u64,
                        lane_stride: num_columns as u64,
                        num_positions: positions_per_block as u64,
                        blocks_per_column: blocks_per_column as u64,
                        position_tiles: position_tiles as u64,
                        record_slots: (layouts * 256) as u64,
                        offset_entries: (layouts * 33) as u64,
                    },
                )
                .unwrap();
            assert!(view.retain_opening_acceleration(Arc::new(index)));
        }
        metal.begin_opening_metrics().unwrap();
        let actual = metal
            .retained_packed_onehot_coefficient_packing(view, &point)
            .unwrap()
            .unwrap();
        assert_eq!(actual.coordinates(), expected);
        let metrics = metal.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.opening_index_bytes, 0);
        assert_eq!(metrics.opening_index_time, std::time::Duration::ZERO);
        assert_eq!(metrics.opening_index_gpu_time, std::time::Duration::ZERO);
        assert!(metal
            .retained_packed_onehot_coefficient_packing(view, &point)
            .unwrap()
            .is_none());
    }

    #[test]
    fn packed_onehot_coefficient_packing_routes_match_definition() {
        const ROWS: usize = 65_536;
        const COLUMNS: usize = 3;
        const CAPACITY: usize = 4;
        let lanes = (0..ROWS * COLUMNS)
            .map(|index| {
                if index % 17 == 0 {
                    0
                } else {
                    ((index * 19 + 7) % 255 + 1) as u8
                }
            })
            .collect::<Vec<_>>();
        indexed_packing_case(lanes.clone(), ROWS, COLUMNS, CAPACITY, 16_384, false);
        indexed_packing_case(lanes, ROWS, COLUMNS, CAPACITY, 16_384, true);
    }

    #[test]
    fn packed_onehot_coefficient_packing_routes_handle_maximal_skew() {
        const ROWS: usize = 256;
        let lanes = vec![7; ROWS];
        indexed_packing_case(lanes.clone(), ROWS, 1, 1, 128, false);
        indexed_packing_case(lanes, ROWS, 1, 1, 128, true);
    }

    #[test]
    fn deferred_packed_onehot_coefficient_packing_fuses_index_at_opening() {
        const ROWS: usize = 2_048;
        const COLUMNS: usize = 3;
        let lanes = (0..ROWS * COLUMNS)
            .map(|index| ((index * 19 + 7) % 255 + 1) as u8)
            .collect();
        indexed_packing_case(lanes, ROWS, COLUMNS, 4, 512, true);
    }
}
