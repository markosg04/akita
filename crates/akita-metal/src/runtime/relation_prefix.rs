use super::*;

impl MetalRuntime {
    pub(crate) fn dispatch_fp128_direct_relation_two_round_prefix(
        &self,
        session: &DirectRelationSession,
        equality_first: &[Fp128Limbs],
        equality_second: &[Fp128Limbs],
        alpha_points: &[Fp128Limbs],
        norm_omitted_corner: usize,
        round: DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationTwoRoundPrefixOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let coefficient_count = round.alpha.len();
            if session.rounds_folded != 0
                || session.current_table.is_some()
                || coefficient_count < 4
                || !coefficient_count.is_power_of_two()
                || norm_omitted_corner >= 4
                || !matches!(session.linear_mode, 0 | 1)
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation two-round prefix state is malformed".into(),
                ));
            }
            let y_quads = coefficient_count / 4;
            let equality_entries = equality_first
                .len()
                .checked_mul(equality_second.len())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation prefix equality entries",
                ))?;
            let expected_equality_entries = session.current_lane_count.checked_mul(y_quads).ok_or(
                MetalCommitError::ShapeOverflow("direct relation prefix domain"),
            )?;
            if equality_first.is_empty()
                || equality_second.is_empty()
                || !equality_first.len().is_power_of_two()
                || !equality_second.len().is_power_of_two()
                || equality_entries != expected_equality_entries
                || alpha_points.len()
                    != 8usize
                        .checked_mul(y_quads)
                        .ok_or(MetalCommitError::ShapeOverflow(
                            "direct relation prefix alpha points",
                        ))?
                || round.live_lane_count
                    != session.live_len.checked_div(coefficient_count).ok_or(
                        MetalCommitError::ShapeOverflow("direct relation prefix live lane count"),
                    )?
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation two-round prefix geometry does not match the session".into(),
                ));
            }
            let equality_block_lanes = equality_first.len() / y_quads;
            let lanes_per_thread = if equality_block_lanes.is_multiple_of(
                FP128_DIRECT_RANGE_THREADS * FP128_DIRECT_RELATION_PREFIX_LANES_PER_THREAD,
            ) {
                FP128_DIRECT_RELATION_PREFIX_LANES_PER_THREAD
            } else {
                1
            };
            let workgroups = round
                .live_lane_count
                .div_ceil(FP128_DIRECT_RANGE_THREADS * lanes_per_thread)
                .max(1);
            if workgroups > session.two_round_prefix_max_workgroups {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation two-round prefix exceeds its workspace".into(),
                ));
            }
            let params = DirectRelationTwoRoundPrefixParams {
                live_lane_count: u64::try_from(round.live_lane_count).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation prefix live lanes")
                })?,
                coefficient_count: u64::try_from(coefficient_count).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation prefix coefficients")
                })?,
                y_quads: u64::try_from(y_quads)
                    .map_err(|_| MetalCommitError::ShapeOverflow("direct relation prefix quads"))?,
                equality_first_len: u64::try_from(equality_first.len()).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation prefix equality split")
                })?,
                workgroups: u64::try_from(workgroups).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation prefix workgroups")
                })?,
                lanes_per_thread: lanes_per_thread as u64,
                norm_omitted_corner: norm_omitted_corner as u64,
                linear_mode: session.linear_mode as u64,
            };
            let mut additional_params = direct_relation_params(
                session,
                session.current_len,
                session.current_lane_count,
                false,
                &round,
            )?;
            additional_params.prefix_size = 1;

            let buffer_start = Instant::now();
            let buffers = self.direct_relation_round_buffers(&round)?;
            let equality_first_buffer = self.shared_buffer_from_slice(equality_first)?;
            let equality_second_buffer = self.shared_buffer_from_slice(equality_second)?;
            let alpha_points_buffer = self.shared_buffer_from_slice(alpha_points)?;
            let prefix_weight = [Fp128Limbs::from_u128(1)];
            let prefix = self.shared_buffer_from_slice(&prefix_weight)?;
            let additional_pairs = if round.additional_pairs.is_empty() {
                None
            } else {
                Some(self.shared_buffer_from_slice(round.additional_pairs)?)
            };
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct relation two-round prefix");
            let encoder = command.new_compute_command_encoder();
            encoder
                .set_compute_pipeline_state(&self.fp128_direct_relation_two_round_prefix_pipeline);
            encoder.set_buffer(0, Some(&session.compact_digits), 0);
            encoder.set_buffer(1, Some(&equality_first_buffer), 0);
            encoder.set_buffer(2, Some(&equality_second_buffer), 0);
            encoder.set_buffer(3, Some(&alpha_points_buffer), 0);
            encoder.set_buffer(
                4,
                Some(&session.lane_weight_tables[session.current_lane_weight_table]),
                0,
            );
            encoder.set_buffer(
                5,
                Some(&session.linear_tables[session.current_linear_table]),
                0,
            );
            encoder.set_buffer(6, Some(&session.linear_source_lane_offsets), 0);
            encoder.set_buffer(7, Some(&session.lane_offsets), 0);
            encoder.set_buffer(8, Some(&session.lane_segments), 0);
            encoder.set_buffer(9, Some(&session.linear_segments), 0);
            encoder.set_buffer(10, Some(&session.two_round_prefix_partials), 0);
            set_inline_bytes(encoder, 11, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    params.workgroups,
                    FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS as u64,
                    1,
                ),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.fp128_direct_relation_two_round_prefix_reduce_pipeline,
            );
            encoder.set_buffer(0, Some(&session.two_round_prefix_partials), 0);
            encoder.set_buffer(1, Some(&session.two_round_prefix_output), 0);
            set_inline_bytes(encoder, 2, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS as u64, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            if let Some(additional_pairs) = &additional_pairs {
                self.encode_direct_relation_additional_compact(
                    command,
                    session,
                    &prefix,
                    additional_pairs,
                    &round.scalars,
                    &additional_params,
                );
            }
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let (norm_evals_except_corner, relation_evals_except_corner) =
                read_direct_relation_prefix_evals(&session.two_round_prefix_output);
            let additional_coefficients = if additional_pairs.is_some() {
                read_direct_relation_coefficients(&session.additional_output)
            } else {
                [Fp128Limbs::default(); FP128_DIRECT_RELATION_STORED_COEFFICIENTS]
            };
            let readback_copy = readback_start.elapsed();
            let allocation_bytes = buffers
                .allocation_bytes
                .checked_add(size_of_val(equality_first))
                .and_then(|bytes| bytes.checked_add(size_of_val(equality_second)))
                .and_then(|bytes| bytes.checked_add(size_of_val(alpha_points)))
                .and_then(|bytes| bytes.checked_add(size_of_val(&prefix_weight)))
                .and_then(|bytes| bytes.checked_add(size_of_val(round.additional_pairs)))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation prefix dispatch allocation bytes",
                ))?;
            Ok(DirectRelationTwoRoundPrefixOutcome {
                norm_evals_except_corner,
                relation_evals_except_corner,
                additional_coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_direct_relation_additional_compact_only(
        &self,
        session: &mut DirectRelationSession,
        challenge: Fp128Limbs,
        current_len: usize,
        prefix_weights: &[Fp128Limbs],
        round: DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationAdditionalOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let buffer_start = Instant::now();
            let prefix = self.shared_buffer_from_slice(prefix_weights)?;
            let pairs = (!round.additional_pairs.is_empty())
                .then(|| self.shared_buffer_from_slice(round.additional_pairs))
                .transpose()?;
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            self.encode_direct_relation_linear_fold(command, session, challenge)?;
            if let Some(pairs) = &pairs {
                let mut params = direct_relation_params(
                    session,
                    current_len,
                    session.current_lane_count,
                    false,
                    &round,
                )?;
                params.prefix_size = u64::try_from(prefix_weights.len()).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation additional prefix")
                })?;
                self.encode_direct_relation_additional_compact(
                    command,
                    session,
                    &prefix,
                    pairs,
                    &round.scalars,
                    &params,
                );
            }
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = if pairs.is_some() {
                read_direct_relation_coefficients(&session.additional_output)
            } else {
                [Fp128Limbs::default(); FP128_DIRECT_RELATION_STORED_COEFFICIENTS]
            };
            let readback_copy = readback_start.elapsed();
            Ok(DirectRelationAdditionalOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: size_of_val(prefix_weights)
                    .checked_add(size_of_val(round.additional_pairs))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "direct relation additional dispatch allocation bytes",
                    ))?,
            })
        })
    }

    pub(crate) fn dispatch_fp128_direct_relation_resume_after_two_round_prefix(
        &self,
        session: &mut DirectRelationSession,
        challenge: Fp128Limbs,
        prefix_weights: &[Fp128Limbs],
        round: DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationRoundOutcome, MetalCommitError> {
        autoreleasepool(|| {
            if session.rounds_folded != 0
                || session.current_table.is_some()
                || prefix_weights.len() != 4
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation two-round resume state is malformed".into(),
                ));
            }
            let current_len = session.current_len / 4;
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct relation resume after two-round prefix");
            self.encode_direct_relation_linear_fold(command, session, challenge)?;
            let mut params = direct_relation_params(
                session,
                current_len,
                session.current_lane_count,
                false,
                &round,
            )?;
            params.prefix_size = 4;
            let buffer_start = Instant::now();
            let buffers = self.direct_relation_round_buffers(&round)?;
            let prefix = self.shared_buffer_from_slice(prefix_weights)?;
            let additional_pairs = if round.additional_pairs.is_empty() {
                None
            } else {
                Some(self.shared_buffer_from_slice(round.additional_pairs)?)
            };
            let buffer_setup = buffer_start.elapsed();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_direct_relation_initial_pipeline);
            encoder.set_buffer(0, Some(&session.compact_digits), 0);
            self.encode_direct_relation_round_buffers(
                encoder,
                1,
                session,
                &buffers,
                &session.lane_weight_tables[session.current_lane_weight_table],
            );
            encoder.set_buffer(10, Some(&session.partials), 0);
            encoder.set_buffer(11, Some(&prefix), 0);
            set_inline_bytes(encoder, 12, &round.scalars);
            set_inline_bytes(encoder, 13, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.workgroups, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            encode_direct_relation_reduction(
                command,
                &self.fp128_direct_range_reduce_pipeline,
                &session.partials,
                &session.round_output,
                &params,
            );
            if let Some(additional_pairs) = &additional_pairs {
                self.encode_direct_relation_additional_compact(
                    command,
                    session,
                    &prefix,
                    additional_pairs,
                    &round.scalars,
                    &params,
                );
            }
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = read_direct_relation_coefficients(&session.round_output);
            let additional_coefficients = if additional_pairs.is_some() {
                read_direct_relation_coefficients(&session.additional_output)
            } else {
                [Fp128Limbs::default(); FP128_DIRECT_RELATION_STORED_COEFFICIENTS]
            };
            let readback_copy = readback_start.elapsed();
            session.current_len = current_len;
            session.current_live_len = params.current_live_len as usize;
            session.rounds_folded = 2;
            Ok(DirectRelationRoundOutcome {
                coefficients,
                additional_coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: buffers
                    .allocation_bytes
                    .checked_add(size_of_val(prefix_weights))
                    .and_then(|bytes| bytes.checked_add(size_of_val(round.additional_pairs)))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "direct relation resume allocation bytes",
                    ))?,
            })
        })
    }
}
