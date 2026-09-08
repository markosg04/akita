use super::*;

impl MetalRuntime {
    pub(crate) fn dispatch_fp128_direct_relation_initial(
        &self,
        session: &DirectRelationSession,
        round: DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationRoundOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let mut params = direct_relation_params(
                session,
                session.current_len,
                session.current_lane_count,
                false,
                &round,
            )?;
            let prefix_weight = [Fp128Limbs::from_u128(1)];
            params.prefix_size = 1;
            let buffer_start = Instant::now();
            let buffers = self.direct_relation_round_buffers(&round)?;
            let prefix = self.shared_buffer_from_slice(&prefix_weight)?;
            let additional_pairs = if round.additional_pairs.is_empty() {
                None
            } else {
                Some(self.shared_buffer_from_slice(round.additional_pairs)?)
            };
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct relation initial round");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 direct relation initial partials");
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
            Ok(DirectRelationRoundOutcome {
                coefficients,
                additional_coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: buffers.allocation_bytes
                    + size_of_val(&prefix_weight)
                    + size_of_val(round.additional_pairs),
            })
        })
    }

    pub(crate) fn dispatch_fp128_direct_relation_advance(
        &self,
        session: &mut DirectRelationSession,
        challenge: Fp128Limbs,
        prefix_weights: &[Fp128Limbs],
        next_round: Option<DirectRelationRoundData<'_>>,
    ) -> Result<DirectRelationAdvanceOutcome, MetalCommitError> {
        autoreleasepool(|| {
            if session.current_len < 2 || !session.current_len.is_power_of_two() {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation session has no foldable table".into(),
                ));
            }
            let next_len = session.current_len / 2;
            if next_len == 1 {
                if next_round.is_some() {
                    return Err(MetalCommitError::UnsupportedShape(
                        "final direct relation fold received another round".into(),
                    ));
                }
                let current_table = session.current_table.ok_or_else(|| {
                    MetalCommitError::UnsupportedShape(
                        "direct relation compact domain is too small".into(),
                    )
                })?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita fp128 direct relation final fold");
                self.encode_direct_relation_linear_fold(command, session, challenge)?;
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.fp128_direct_range_finalize_pipeline);
                encoder.set_buffer(0, Some(&session.tables[current_table]), 0);
                encoder.set_buffer(1, Some(&session.final_output), 0);
                set_inline_bytes(encoder, 2, &challenge);
                set_inline_bytes(encoder, 3, &(session.current_live_len as u64));
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();
                if session.linear_mode != 0 {
                    let encoder = command.new_blit_command_encoder();
                    encoder.copy_from_buffer(
                        &session.linear_tables[session.current_linear_table],
                        0,
                        &session.linear_final_output,
                        0,
                        size_of::<Fp128Limbs>() as u64,
                    );
                    encoder.end_encoding();
                }
                let (command_wall, gpu) = complete_command(command)?;
                let readback_start = Instant::now();
                // SAFETY: `final_output` contains one initialized fp128 value.
                let final_evaluation =
                    unsafe { *session.final_output.contents().cast::<Fp128Limbs>() };
                let final_linear_evaluation = if session.linear_mode == 0 {
                    Fp128Limbs::default()
                } else {
                    // SAFETY: the final command copied one initialized fp128 value.
                    unsafe { *session.linear_final_output.contents().cast::<Fp128Limbs>() }
                };
                let readback_copy = readback_start.elapsed();
                session.current_len = 1;
                session.current_live_len = session.current_live_len.div_ceil(2);
                return Ok(DirectRelationAdvanceOutcome {
                    next_coefficients: None,
                    next_additional_coefficients: None,
                    final_evaluation: Some(final_evaluation),
                    final_linear_evaluation: Some(final_linear_evaluation),
                    timings: DispatchTimings {
                        buffer_setup: Duration::ZERO,
                        command_wall,
                        gpu,
                        readback_copy,
                    },
                    allocation_bytes: 0,
                });
            }

            let round = next_round.ok_or_else(|| {
                MetalCommitError::UnsupportedShape(
                    "non-final direct relation fold is missing round data".into(),
                )
            })?;
            let fold_lane_weights = session.rounds_folded >= session.coefficient_rounds;
            let next_lane_count = if fold_lane_weights {
                if session.current_lane_count < 2 || !session.current_lane_count.is_power_of_two() {
                    return Err(MetalCommitError::UnsupportedShape(
                        "direct relation lane-weight table is not foldable".into(),
                    ));
                }
                session.current_lane_count / 2
            } else {
                session.current_lane_count
            };
            let next_lane_weight_table = if fold_lane_weights {
                1 - session.current_lane_weight_table
            } else {
                session.current_lane_weight_table
            };
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct relation fold and next round");
            self.encode_direct_relation_linear_fold(command, session, challenge)?;
            let mut params = direct_relation_params(
                session,
                next_len,
                next_lane_count,
                fold_lane_weights,
                &round,
            )?;
            if params.current_len != next_len as u64 {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation round data does not match the folded table".into(),
                ));
            }
            let output_table = session.current_table.map_or(0, |current| 1 - current);
            let buffer_start = Instant::now();
            let buffers = self.direct_relation_round_buffers(&round)?;
            let prefix = if session.current_table.is_none() {
                let expected_prefix_size = 1usize
                    .checked_shl(u32::try_from(session.rounds_folded + 1).map_err(|_| {
                        MetalCommitError::ShapeOverflow("direct relation compact prefix width")
                    })?)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "direct relation compact prefix size",
                    ))?;
                if prefix_weights.len() != expected_prefix_size {
                    return Err(MetalCommitError::UnsupportedShape(
                        "direct relation compact prefix weights have the wrong length".into(),
                    ));
                }
                params.prefix_size = expected_prefix_size as u64;
                params.materialize_prefix =
                    u64::from(session.rounds_folded + 1 >= session.compact_prefix_rounds);
                Some(self.shared_buffer_from_slice(prefix_weights)?)
            } else {
                None
            };
            let additional_pairs = if round.additional_pairs.is_empty() {
                None
            } else {
                Some(self.shared_buffer_from_slice(round.additional_pairs)?)
            };
            let buffer_setup = buffer_start.elapsed();
            let encoder = command.new_compute_command_encoder();
            if let Some(current_table) = session.current_table {
                encoder.set_compute_pipeline_state(&self.fp128_direct_relation_field_fold_pipeline);
                encoder.set_buffer(0, Some(&session.tables[current_table]), 0);
                encoder.set_buffer(
                    15,
                    Some(&session.lane_weight_tables[session.current_lane_weight_table]),
                    0,
                );
            } else {
                if fold_lane_weights {
                    return Err(MetalCommitError::UnsupportedShape(
                        "direct relation cannot fold lane weights from compact witness state"
                            .into(),
                    ));
                }
                encoder
                    .set_compute_pipeline_state(&self.fp128_direct_relation_compact_fold_pipeline);
                encoder.set_buffer(0, Some(&session.compact_digits), 0);
            }
            encoder.set_buffer(1, Some(&session.tables[output_table]), 0);
            self.encode_direct_relation_round_buffers(
                encoder,
                2,
                session,
                &buffers,
                &session.lane_weight_tables[next_lane_weight_table],
            );
            encoder.set_buffer(11, Some(&session.partials), 0);
            if let Some(prefix) = &prefix {
                encoder.set_buffer(12, Some(prefix), 0);
            } else {
                set_inline_bytes(encoder, 12, &challenge);
            }
            set_inline_bytes(encoder, 13, &round.scalars);
            set_inline_bytes(encoder, 14, &params);
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
                if let Some(prefix) = &prefix {
                    self.encode_direct_relation_additional_compact(
                        command,
                        session,
                        prefix,
                        additional_pairs,
                        &round.scalars,
                        &params,
                    );
                } else {
                    self.encode_direct_relation_additional_field(
                        command,
                        &session.tables[output_table],
                        session,
                        additional_pairs,
                        &round.scalars,
                        &params,
                    );
                }
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
            if session.current_table.is_some()
                || session.rounds_folded + 1 >= session.compact_prefix_rounds
            {
                session.current_table = Some(output_table);
            }
            session.current_len = next_len;
            session.current_live_len = params.current_live_len as usize;
            session.rounds_folded += 1;
            if fold_lane_weights {
                session.current_lane_weight_table = next_lane_weight_table;
                session.current_lane_count = next_lane_count;
            }
            Ok(DirectRelationAdvanceOutcome {
                next_coefficients: Some(coefficients),
                next_additional_coefficients: Some(additional_coefficients),
                final_evaluation: None,
                final_linear_evaluation: None,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: buffers.allocation_bytes
                    + size_of_val(prefix_weights)
                    + size_of_val(round.additional_pairs),
            })
        })
    }

    pub(crate) fn direct_relation_session_allocation_bytes(
        &self,
        session: &DirectRelationSession,
    ) -> usize {
        session.allocation_bytes
    }

    pub(super) fn encode_direct_relation_linear_fold(
        &self,
        command: &CommandBufferRef,
        session: &mut DirectRelationSession,
        challenge: Fp128Limbs,
    ) -> Result<(), MetalCommitError> {
        self.encode_direct_relation_linear_fold_with_binding(
            command,
            session,
            Fp128KernelBinding::Inline(challenge),
        )
    }

    pub(super) fn encode_direct_relation_linear_fold_from_buffer(
        &self,
        command: &CommandBufferRef,
        session: &mut DirectRelationSession,
        challenges: &Buffer,
        challenge_offset: u64,
    ) -> Result<(), MetalCommitError> {
        self.encode_direct_relation_linear_fold_with_binding(
            command,
            session,
            Fp128KernelBinding::Buffer(challenges, challenge_offset),
        )
    }

    pub(super) fn encode_direct_relation_linear_fold_with_binding(
        &self,
        command: &CommandBufferRef,
        session: &mut DirectRelationSession,
        challenge: Fp128KernelBinding<'_>,
    ) -> Result<(), MetalCommitError> {
        if session.linear_mode == 0 {
            return Ok(());
        }
        let folding_coefficients = session.rounds_folded < session.coefficient_rounds;
        let (mode, output_len, next_coeff_count, next_live_lane_count) = if folding_coefficients {
            if session.linear_mode != 1 || session.linear_current_coeff_count < 2 {
                return Err(MetalCommitError::UnsupportedShape(
                    "factored direct relation source is not coefficient-foldable".into(),
                ));
            }
            let next_coeff_count = session.linear_current_coeff_count / 2;
            let output_len = session
                .linear_source_lane_count
                .checked_mul(next_coeff_count)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "direct relation folded linear source",
                ))?;
            (
                1usize,
                output_len,
                next_coeff_count,
                session.linear_current_live_lane_count,
            )
        } else {
            if session.linear_current_coeff_count != 1
                || session.linear_current_live_lane_count == 0
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct relation linear lane source is not foldable".into(),
                ));
            }
            let next_live_lane_count = session.linear_current_live_lane_count.div_ceil(2);
            let mode = if session.linear_mode == 1 { 2 } else { 3 };
            (mode, next_live_lane_count, 1, next_live_lane_count)
        };
        let params = DirectRelationLinearFoldParams {
            current_coeff_count: u64::try_from(session.linear_current_coeff_count)
                .map_err(|_| MetalCommitError::ShapeOverflow("direct relation linear width"))?,
            source_lane_count: u64::try_from(session.linear_source_lane_count).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation linear source lanes")
            })?,
            current_live_lane_count: u64::try_from(session.linear_current_live_lane_count)
                .map_err(|_| MetalCommitError::ShapeOverflow("direct relation linear lanes"))?,
            output_len: u64::try_from(output_len).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct relation folded linear output")
            })?,
            mode: mode as u64,
        };
        let output_table = 1 - session.current_linear_table;
        let encoder = command.new_compute_command_encoder();
        encoder.set_label("Akita fp128 resident direct relation linear fold");
        encoder.set_compute_pipeline_state(&self.fp128_direct_relation_linear_fold_pipeline);
        encoder.set_buffer(
            0,
            Some(&session.linear_tables[session.current_linear_table]),
            0,
        );
        encoder.set_buffer(1, Some(&session.linear_tables[output_table]), 0);
        encoder.set_buffer(2, Some(&session.linear_source_lane_offsets), 0);
        encoder.set_buffer(3, Some(&session.lane_offsets), 0);
        encoder.set_buffer(4, Some(&session.lane_segments), 0);
        encoder.set_buffer(5, Some(&session.linear_segments), 0);
        set_fp128_binding(encoder, 6, challenge);
        set_inline_bytes(encoder, 7, &params);
        encoder.dispatch_threads(
            MTLSize::new(params.output_len, 1, 1),
            MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
        );
        encoder.end_encoding();
        session.current_linear_table = output_table;
        session.linear_current_coeff_count = next_coeff_count;
        session.linear_current_live_lane_count = next_live_lane_count;
        if !folding_coefficients {
            session.linear_mode = 2;
        }
        Ok(())
    }

    pub(super) fn direct_relation_round_buffers(
        &self,
        round: &DirectRelationRoundData<'_>,
    ) -> Result<DirectRelationRoundBuffers, MetalCommitError> {
        let allocation_bytes = size_of_val(round.e_first)
            .checked_add(size_of_val(round.e_second))
            .and_then(|bytes| bytes.checked_add(size_of_val(round.alpha)))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation round allocation bytes",
            ))?;
        Ok(DirectRelationRoundBuffers {
            e_first: self.shared_buffer_from_slice(round.e_first)?,
            e_second: self.shared_buffer_from_slice(round.e_second)?,
            alpha: self.shared_buffer_from_slice(round.alpha)?,
            allocation_bytes,
        })
    }

    pub(super) fn encode_direct_relation_round_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        start: u64,
        session: &DirectRelationSession,
        buffers: &DirectRelationRoundBuffers,
        lane_weights: &Buffer,
    ) {
        encoder.set_buffer(start, Some(&buffers.e_first), 0);
        encoder.set_buffer(start + 1, Some(&buffers.e_second), 0);
        encoder.set_buffer(start + 2, Some(&buffers.alpha), 0);
        encoder.set_buffer(start + 3, Some(lane_weights), 0);
        encoder.set_buffer(
            start + 4,
            Some(&session.linear_tables[session.current_linear_table]),
            0,
        );
        encoder.set_buffer(start + 5, Some(&session.linear_source_lane_offsets), 0);
        encoder.set_buffer(start + 6, Some(&session.lane_offsets), 0);
        encoder.set_buffer(start + 7, Some(&session.lane_segments), 0);
        encoder.set_buffer(start + 8, Some(&session.linear_segments), 0);
    }

    pub(super) fn encode_direct_relation_additional_compact(
        &self,
        command: &CommandBufferRef,
        session: &DirectRelationSession,
        prefix: &Buffer,
        pairs: &Buffer,
        scalars: &DirectRelationScalars,
        params: &DirectRelationParams,
    ) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.fp128_direct_relation_additional_compact_pipeline);
        encoder.set_buffer(0, Some(&session.compact_digits), 0);
        encoder.set_buffer(1, Some(prefix), 0);
        encoder.set_buffer(2, Some(pairs), 0);
        encoder.set_buffer(3, Some(&session.partials), 0);
        set_inline_bytes(encoder, 4, scalars);
        set_inline_bytes(encoder, 5, params);
        encoder.dispatch_thread_groups(
            MTLSize::new(params.additional_workgroups, 1, 1),
            MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
        );
        encoder.end_encoding();
        let mut reduction_params = *params;
        reduction_params.workgroups = params.additional_workgroups;
        encode_direct_relation_reduction(
            command,
            &self.fp128_direct_range_reduce_pipeline,
            &session.partials,
            &session.additional_output,
            &reduction_params,
        );
    }

    pub(super) fn encode_direct_relation_additional_field(
        &self,
        command: &CommandBufferRef,
        witness: &Buffer,
        session: &DirectRelationSession,
        pairs: &Buffer,
        scalars: &DirectRelationScalars,
        params: &DirectRelationParams,
    ) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.fp128_direct_relation_additional_field_pipeline);
        encoder.set_buffer(0, Some(witness), 0);
        encoder.set_buffer(1, Some(pairs), 0);
        encoder.set_buffer(2, Some(&session.partials), 0);
        set_inline_bytes(encoder, 3, scalars);
        set_inline_bytes(encoder, 4, params);
        encoder.dispatch_thread_groups(
            MTLSize::new(params.additional_workgroups, 1, 1),
            MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
        );
        encoder.end_encoding();
        let mut reduction_params = *params;
        reduction_params.workgroups = params.additional_workgroups;
        encode_direct_relation_reduction(
            command,
            &self.fp128_direct_range_reduce_pipeline,
            &session.partials,
            &session.additional_output,
            &reduction_params,
        );
    }
}
