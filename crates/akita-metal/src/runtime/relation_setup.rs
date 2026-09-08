use super::*;

impl MetalRuntime {
    #[expect(
        clippy::too_many_arguments,
        reason = "the session boundary keeps independent resident relation tables explicit"
    )]
    pub(crate) fn begin_fp128_direct_relation(
        &self,
        digits: &[i8],
        domain_len: usize,
        compact_prefix_rounds: usize,
        coefficient_rounds: usize,
        lane_weights: &[Fp128Limbs],
        linear_segments: &[DirectRelationLinearSegment],
        lane_offsets: &[u32],
        lane_segments: &[u32],
        linear_sources: &[DirectRelationLinearSourceInput],
        linear_dense_values: &[Fp128Limbs],
    ) -> Result<(DirectRelationSession, DispatchTimings), MetalCommitError> {
        let coefficient_count = 1usize
            .checked_shl(u32::try_from(coefficient_rounds).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation coefficient rounds")
            })?)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation coefficient count",
            ))?;
        let live_lane_count = digits.len() / coefficient_count;
        let linear_source_elements = linear_sources.iter().try_fold(0usize, |total, source| {
            total
                .checked_add(source.element_len().ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation linear source length",
                ))?)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation linear source length",
                ))
        })?;
        let linear_source_lane_count = linear_source_elements
            .checked_div(coefficient_count)
            .filter(|_| linear_source_elements.is_multiple_of(coefficient_count))
            .ok_or(MetalCommitError::UnsupportedShape(
                "factored linear source does not match the coefficient geometry".into(),
            ))?;
        let mut linear_source_lane_offsets = Vec::with_capacity(linear_sources.len() + 1);
        linear_source_lane_offsets.push(0u32);
        let mut source_lane_cursor = 0usize;
        for source in linear_sources {
            let source_elements = source.element_len().ok_or(MetalCommitError::ShapeOverflow(
                "direct relation linear source length",
            ))?;
            let source_lanes = source_elements
                .checked_div(coefficient_count)
                .filter(|_| source_elements.is_multiple_of(coefficient_count))
                .ok_or(MetalCommitError::UnsupportedShape(
                    "direct relation source is not coefficient aligned".into(),
                ))?;
            source_lane_cursor = source_lane_cursor.checked_add(source_lanes).ok_or(
                MetalCommitError::ShapeOverflow("direct relation linear source lanes"),
            )?;
            linear_source_lane_offsets.push(u32::try_from(source_lane_cursor).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation linear source lanes")
            })?);
        }
        let linear_mode = if !linear_dense_values.is_empty() {
            2
        } else if !linear_sources.is_empty() {
            1
        } else {
            0
        };
        let linear_shape_is_valid = match linear_mode {
            0 => linear_dense_values.is_empty(),
            1 => linear_source_elements != 0 && linear_dense_values.is_empty(),
            2 => {
                linear_sources.is_empty()
                    && coefficient_count == 1
                    && linear_dense_values.len() == live_lane_count
            }
            _ => false,
        };
        if domain_len < 16
            || !domain_len.is_power_of_two()
            || digits.len() > domain_len
            || compact_prefix_rounds == 0
            || compact_prefix_rounds > coefficient_rounds
            || coefficient_rounds > domain_len.trailing_zeros() as usize
            || !digits.len().is_multiple_of(coefficient_count)
            || lane_weights.is_empty()
            || !lane_weights.len().is_power_of_two()
            || coefficient_count.checked_mul(lane_weights.len()) != Some(domain_len)
            || lane_offsets.is_empty()
            || !linear_shape_is_valid
        {
            return Err(MetalCommitError::UnsupportedShape(
                "direct relation proof has malformed resident geometry".into(),
            ));
        }
        let setup_start = Instant::now();
        let compact_digits = self.shared_buffer_from_slice(digits)?;
        let compact_prefix_size = 1usize
            .checked_shl(u32::try_from(compact_prefix_rounds).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation compact prefix width")
            })?)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation compact prefix size",
            ))?;
        let first_table_len = digits.len().div_ceil(compact_prefix_size).max(1);
        let second_table_len = first_table_len.div_ceil(2).max(1);
        let first_table_bytes = first_table_len.checked_mul(size_of::<Fp128Limbs>()).ok_or(
            MetalCommitError::ShapeOverflow("direct relation first table bytes"),
        )?;
        let second_table_bytes = second_table_len
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation second table bytes",
            ))?;
        let tables = [
            self.private_buffer(first_table_bytes)?,
            self.private_buffer(second_table_bytes)?,
        ];
        let second_lane_count = (lane_weights.len() / 2).max(1);
        let second_lane_bytes = second_lane_count
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation second lane-weight table",
            ))?;
        let lane_weight_tables = [
            self.shared_buffer_from_slice(lane_weights)?,
            self.private_buffer(second_lane_bytes)?,
        ];
        let maximum_workgroups = direct_range_workgroups(digits.len().div_ceil(2));
        let partial_bytes = maximum_workgroups
            .checked_mul(FP128_DIRECT_RELATION_STORED_COEFFICIENTS)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation partial bytes",
            ))?;
        let output_bytes = FP128_DIRECT_RELATION_STORED_COEFFICIENTS
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation output bytes",
            ))?;
        let partials = self.private_buffer(partial_bytes)?;
        let round_output = self.shared_buffer(output_bytes)?;
        let additional_output = self.shared_buffer(output_bytes)?;
        let final_output = self.shared_buffer(size_of::<Fp128Limbs>())?;
        let linear_final_output = self.shared_buffer(size_of::<Fp128Limbs>())?;
        let two_round_prefix_max_workgroups =
            live_lane_count.div_ceil(FP128_DIRECT_RANGE_THREADS).max(1);
        let two_round_prefix_partial_bytes = two_round_prefix_max_workgroups
            .checked_mul(FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation two-round prefix partial bytes",
            ))?;
        let two_round_prefix_output_bytes = FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation two-round prefix output bytes",
            ))?;
        let two_round_prefix_partials = self.private_buffer(two_round_prefix_partial_bytes)?;
        let two_round_prefix_output = self.shared_buffer(two_round_prefix_output_bytes)?;

        let zero_segment = DirectRelationLinearSegment {
            factor: Fp128Limbs::default(),
            source_index: 0,
            target_lane_start: 0,
            target_lane_stride: 1,
            source_lane_start: 0,
            source_lane_stride: 1,
            lane_count: 0,
        };
        let zero_u32 = 0u32;
        let segment_bytes = if linear_segments.is_empty() {
            size_of::<DirectRelationLinearSegment>()
        } else {
            size_of_val(linear_segments)
        };
        let linear_segments = self.shared_buffer_from_slice(if linear_segments.is_empty() {
            std::slice::from_ref(&zero_segment)
        } else {
            linear_segments
        })?;
        let lane_offsets_buffer = self.shared_buffer_from_slice(lane_offsets)?;
        let lane_segments_buffer = self.shared_buffer_from_slice(if lane_segments.is_empty() {
            std::slice::from_ref(&zero_u32)
        } else {
            lane_segments
        })?;
        let linear_source_lane_offsets_buffer =
            self.shared_buffer_from_slice(if linear_source_lane_offsets.is_empty() {
                std::slice::from_ref(&zero_u32)
            } else {
                &linear_source_lane_offsets
            })?;
        let linear_capacity = linear_source_elements
            .max(linear_dense_values.len())
            .max(live_lane_count)
            .max(1);
        let linear_table_bytes = linear_capacity.checked_mul(size_of::<Fp128Limbs>()).ok_or(
            MetalCommitError::ShapeOverflow("direct relation linear table bytes"),
        )?;
        let linear_tables = [
            self.private_buffer(linear_table_bytes)?,
            self.private_buffer(linear_table_bytes)?,
        ];
        let command = self.queue.new_command_buffer();
        command.set_label("Akita resident direct relation linear source construction");
        let mut command_needed = false;
        let mut source_input_bytes = 0usize;
        let mut source_element_offset = 0usize;
        if !linear_dense_values.is_empty() {
            let staging = self.shared_buffer_from_slice(linear_dense_values)?;
            let encoder = command.new_blit_command_encoder();
            encoder.copy_from_buffer(
                &staging,
                0,
                &linear_tables[0],
                0,
                size_of_val(linear_dense_values) as u64,
            );
            encoder.end_encoding();
            command_needed = true;
            source_input_bytes = size_of_val(linear_dense_values);
        }
        for source in linear_sources {
            let destination_offset = source_element_offset
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation linear source offset",
                ))?;
            match source {
                DirectRelationLinearSourceInput::Values(values) => {
                    let staging = self.shared_buffer_from_slice(values)?;
                    let encoder = command.new_blit_command_encoder();
                    encoder.copy_from_buffer(
                        &staging,
                        0,
                        &linear_tables[0],
                        destination_offset as u64,
                        size_of_val(values.as_slice()) as u64,
                    );
                    encoder.end_encoding();
                    source_input_bytes = source_input_bytes
                        .checked_add(size_of_val(values.as_slice()))
                        .ok_or(MetalCommitError::ShapeOverflow(
                            "direct relation source input bytes",
                        ))?;
                }
                DirectRelationLinearSourceInput::ReducedSetup {
                    matrix,
                    ring_dimension,
                    row_count,
                    column_count,
                    row_weights,
                    alpha_powers,
                    alpha,
                    wrap_correction,
                } => {
                    if *ring_dimension < 32
                        || *ring_dimension > 512
                        || !ring_dimension.is_multiple_of(32)
                        || row_weights.len() != *row_count
                        || alpha_powers.len() != *ring_dimension
                    {
                        return Err(MetalCommitError::UnsupportedShape(
                            "reduced setup source geometry is unsupported".into(),
                        ));
                    }
                    let weights = self.shared_buffer_from_slice(row_weights)?;
                    let powers = self.shared_buffer_from_slice(alpha_powers)?;
                    let params = DirectRelationReducedSourceParams {
                        ring_dimension: *ring_dimension as u64,
                        row_count: *row_count as u64,
                        item_count: *column_count as u64,
                        reserved: 0,
                        alpha: *alpha,
                        wrap_correction: *wrap_correction,
                    };
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 reduced setup source");
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_setup_source_pipeline,
                    );
                    encoder.set_buffer(0, Some(matrix), 0);
                    encoder.set_buffer(1, Some(&weights), 0);
                    encoder.set_buffer(2, Some(&powers), 0);
                    encoder.set_buffer(3, Some(&linear_tables[0]), destination_offset as u64);
                    set_inline_bytes(encoder, 4, &params);
                    encoder.dispatch_thread_groups(
                        MTLSize::new(*column_count as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    encoder.end_encoding();
                    source_input_bytes = source_input_bytes
                        .checked_add(size_of_val(row_weights.as_slice()))
                        .and_then(|bytes| bytes.checked_add(size_of_val(alpha_powers.as_slice())))
                        .ok_or(MetalCommitError::ShapeOverflow(
                            "direct relation source input bytes",
                        ))?;
                }
                DirectRelationLinearSourceInput::ReducedSparse {
                    ring_dimension,
                    challenge_count,
                    term_offsets,
                    positions,
                    coefficients,
                    alpha_powers,
                    alpha,
                    wrap_correction,
                } => {
                    if *ring_dimension < 32
                        || *ring_dimension > 512
                        || !ring_dimension.is_multiple_of(32)
                        || term_offsets.len() != challenge_count + 1
                        || positions.len() != coefficients.len()
                        || term_offsets.last().copied().map(|offset| offset as usize)
                            != Some(positions.len())
                        || positions
                            .iter()
                            .any(|position| *position as usize >= *ring_dimension)
                        || alpha_powers.len() != *ring_dimension
                    {
                        return Err(MetalCommitError::UnsupportedShape(
                            "reduced sparse source geometry is unsupported".into(),
                        ));
                    }
                    let offsets = self.shared_buffer_from_slice(term_offsets)?;
                    let positions_buffer = self.shared_buffer_from_slice(positions)?;
                    let coefficients_buffer = self.shared_buffer_from_slice(coefficients)?;
                    let powers = self.shared_buffer_from_slice(alpha_powers)?;
                    let params = DirectRelationReducedSourceParams {
                        ring_dimension: *ring_dimension as u64,
                        row_count: 0,
                        item_count: *challenge_count as u64,
                        reserved: 0,
                        alpha: *alpha,
                        wrap_correction: *wrap_correction,
                    };
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 reduced sparse source");
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_sparse_source_pipeline,
                    );
                    encoder.set_buffer(0, Some(&offsets), 0);
                    encoder.set_buffer(1, Some(&positions_buffer), 0);
                    encoder.set_buffer(2, Some(&coefficients_buffer), 0);
                    encoder.set_buffer(3, Some(&powers), 0);
                    encoder.set_buffer(4, Some(&linear_tables[0]), destination_offset as u64);
                    set_inline_bytes(encoder, 5, &params);
                    encoder.dispatch_thread_groups(
                        MTLSize::new(*challenge_count as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    encoder.end_encoding();
                    source_input_bytes = source_input_bytes
                        .checked_add(size_of_val(term_offsets.as_slice()))
                        .and_then(|bytes| bytes.checked_add(size_of_val(positions.as_slice())))
                        .and_then(|bytes| bytes.checked_add(size_of_val(coefficients.as_slice())))
                        .and_then(|bytes| bytes.checked_add(size_of_val(alpha_powers.as_slice())))
                        .ok_or(MetalCommitError::ShapeOverflow(
                            "direct relation source input bytes",
                        ))?;
                }
            }
            source_element_offset = source_element_offset
                .checked_add(source.element_len().ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation linear source length",
                ))?)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation linear source length",
                ))?;
            command_needed = true;
        }
        let buffer_setup = setup_start.elapsed();
        let (command_wall, gpu) = if command_needed {
            complete_command(command)?
        } else {
            (Duration::ZERO, None)
        };
        let allocation_bytes = digits
            .len()
            .checked_add(first_table_bytes)
            .and_then(|bytes| bytes.checked_add(second_table_bytes))
            .and_then(|bytes| bytes.checked_add(partial_bytes))
            .and_then(|bytes| bytes.checked_add(2 * output_bytes))
            .and_then(|bytes| bytes.checked_add(size_of::<Fp128Limbs>()))
            .and_then(|bytes| bytes.checked_add(size_of::<Fp128Limbs>()))
            .and_then(|bytes| bytes.checked_add(segment_bytes))
            .and_then(|bytes| bytes.checked_add(size_of_val(lane_offsets)))
            .and_then(|bytes| bytes.checked_add(size_of_val(lane_segments)))
            .and_then(|bytes| bytes.checked_add(size_of_val(linear_source_lane_offsets.as_slice())))
            .and_then(|bytes| bytes.checked_add(2 * linear_table_bytes))
            .and_then(|bytes| bytes.checked_add(size_of_val(lane_weights)))
            .and_then(|bytes| bytes.checked_add(second_lane_bytes))
            .and_then(|bytes| bytes.checked_add(two_round_prefix_partial_bytes))
            .and_then(|bytes| bytes.checked_add(two_round_prefix_output_bytes))
            .and_then(|bytes| bytes.checked_add(source_input_bytes))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation resident allocation bytes",
            ))?;
        Ok((
            DirectRelationSession {
                compact_digits,
                tables,
                partials,
                round_output,
                additional_output,
                final_output,
                linear_final_output,
                linear_segments,
                lane_offsets: lane_offsets_buffer,
                lane_segments: lane_segments_buffer,
                linear_source_lane_offsets: linear_source_lane_offsets_buffer,
                linear_tables,
                current_linear_table: 0,
                linear_mode,
                linear_source_lane_count,
                linear_current_coeff_count: coefficient_count,
                linear_current_live_lane_count: live_lane_count,
                lane_weight_tables,
                two_round_prefix_partials,
                two_round_prefix_output,
                two_round_prefix_max_workgroups,
                live_len: digits.len(),
                current_len: domain_len,
                current_live_len: digits.len(),
                current_table: None,
                current_lane_weight_table: 0,
                current_lane_count: lane_weights.len(),
                coefficient_rounds,
                compact_prefix_rounds,
                rounds_folded: 0,
                allocation_bytes,
            },
            DispatchTimings {
                buffer_setup,
                command_wall,
                gpu,
                readback_copy: Duration::ZERO,
            },
        ))
    }
}
