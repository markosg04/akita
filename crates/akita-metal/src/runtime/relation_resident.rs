use super::*;

impl MetalRuntime {
    pub(crate) fn dispatch_fp128_direct_relation_resident_suffix(
        &self,
        session: &mut DirectRelationSession,
        prefix_challenges: &[Fp128Limbs],
        equality_schedule: &[DirectRelationResidentEqRound],
        initial_round: DirectRelationRoundData<'_>,
        chaining_value: &[u8; 64],
        prior_squeezed_bytes: usize,
    ) -> Result<DirectRelationResidentOutcome, MetalCommitError> {
        let suffix_rounds = equality_schedule.len();
        if suffix_rounds == 0
            || prefix_challenges.len() != session.rounds_folded
            || session.rounds_folded != 2
            || suffix_rounds != session.current_len.trailing_zeros() as usize
            || session.current_table.is_some()
            || session.compact_prefix_rounds != 3
            || initial_round.alpha.is_empty()
            || initial_round.e_first != equality_schedule[0].e_first
            || initial_round.e_second != equality_schedule[0].e_second
        {
            return Err(MetalCommitError::UnsupportedShape(
                "resident direct relation suffix has malformed initial state".into(),
            ));
        }
        let _ = direct_relation_params(
            session,
            session.current_len,
            session.current_lane_count,
            false,
            &initial_round,
        )?;
        let additional_schedule = direct_relation_additional_fold_schedule(
            initial_round.additional_pairs,
            suffix_rounds.saturating_sub(1),
        )?;

        autoreleasepool(|| {
            let setup_start = Instant::now();
            let equality_buffers = equality_schedule
                .iter()
                .map(|round| {
                    Ok((
                        self.shared_buffer_from_slice(&round.e_first)?,
                        self.shared_buffer_from_slice(&round.e_second)?,
                    ))
                })
                .collect::<Result<Vec<_>, MetalCommitError>>()?;
            let taus = equality_schedule
                .iter()
                .map(|round| round.tau)
                .collect::<Vec<_>>();
            let tau_buffer = self.shared_buffer_from_slice(&taus)?;

            let alpha_bytes = size_of_val(initial_round.alpha);
            let alpha_tables = [
                self.shared_buffer_from_slice(initial_round.alpha)?,
                self.private_buffer(alpha_bytes.max(size_of::<Fp128Limbs>()))?,
            ];
            let scalar_values = vec![initial_round.scalars; suffix_rounds];
            let scalars = self.shared_buffer_from_slice(&scalar_values)?;

            let zero_pair = DirectRelationAdditionalPair {
                parent: 0,
                reserved: 0,
                linear: [Fp128Limbs::default(); 2],
                binary: [Fp128Limbs::default(); 2],
            };
            let initial_pair_storage = if initial_round.additional_pairs.is_empty() {
                std::slice::from_ref(&zero_pair)
            } else {
                initial_round.additional_pairs
            };
            let pair_capacity = initial_pair_storage.len();
            let pair_bytes = pair_capacity
                .checked_mul(size_of::<DirectRelationAdditionalPair>())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct relation additional pairs",
                ))?;
            let additional_pairs = [
                self.shared_buffer_from_slice(initial_pair_storage)?,
                self.private_buffer(pair_bytes)?,
            ];
            let additional_mappings = additional_schedule
                .iter()
                .map(|mappings| {
                    if mappings.is_empty() {
                        Ok(None)
                    } else {
                        self.shared_buffer_from_slice(mappings).map(Some)
                    }
                })
                .collect::<Result<Vec<_>, MetalCommitError>>()?;

            let total_rounds = prefix_challenges.len().checked_add(suffix_rounds).ok_or(
                MetalCommitError::ShapeOverflow("resident direct relation challenge count"),
            )?;
            let mut challenge_values = vec![Fp128Limbs::default(); total_rounds];
            challenge_values[..prefix_challenges.len()].copy_from_slice(prefix_challenges);
            let challenges = self.shared_buffer_from_slice(&challenge_values)?;
            let proof_coefficient_count =
                suffix_rounds
                    .checked_mul(3)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "resident direct relation proof coefficients",
                    ))?;
            let proof_bytes = proof_coefficient_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct relation proof bytes",
                ))?;
            let proof_coefficients = self.shared_buffer(proof_bytes)?;
            let count_bytes = suffix_rounds.checked_mul(size_of::<u32>()).ok_or(
                MetalCommitError::ShapeOverflow("resident direct relation coefficient counts"),
            )?;
            let coefficient_counts = self.shared_buffer(count_bytes)?;
            let state = self.shared_buffer_from_slice(chaining_value)?;
            let buffer_setup = setup_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 resident direct relation suffix");
            let mut current_len = session.current_len;
            let mut current_live_len = session.current_live_len;
            let mut current_live_lane_count = initial_round.live_lane_count;
            let mut current_alpha_len = initial_round.alpha.len();
            let mut current_alpha_table = 0usize;
            let mut current_additional_table = 0usize;
            let mut current_additional_count = initial_round.additional_pairs.len();
            let mut current_table = session.current_table;
            let mut current_lane_weight_table = session.current_lane_weight_table;
            let mut current_lane_count = session.current_lane_count;
            let mut rounds_folded = session.rounds_folded;
            let had_linear_terms = session.linear_mode != 0;

            for suffix_round in 0..suffix_rounds {
                session.rounds_folded = rounds_folded;
                session.current_live_len = current_live_len;
                let round_index = prefix_challenges.len() + suffix_round;
                let challenge_offset = round_index.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                    MetalCommitError::ShapeOverflow("resident direct relation challenge offset"),
                )? as u64;
                let proof_offset = suffix_round
                    .checked_mul(3 * size_of::<Fp128Limbs>())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "resident direct relation proof offset",
                    ))? as u64;
                let count_offset = suffix_round.checked_mul(size_of::<u32>()).ok_or(
                    MetalCommitError::ShapeOverflow("resident direct relation count offset"),
                )? as u64;
                let transcript_params = DirectRelationTranscriptParams {
                    prior_squeezed_bytes: if suffix_round == 0 {
                        u64::try_from(prior_squeezed_bytes).map_err(|_| {
                            MetalCommitError::ShapeOverflow(
                                "resident direct relation prior squeeze length",
                            )
                        })?
                    } else {
                        32
                    },
                    has_additional: u64::from(current_additional_count != 0),
                };
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 resident direct relation challenge");
                encoder.set_compute_pipeline_state(
                    &self.fp128_blake2b_relation_sumcheck_round_pipeline,
                );
                encoder.set_buffer(0, Some(&state), 0);
                encoder.set_buffer(1, Some(&session.round_output), 0);
                encoder.set_buffer(2, Some(&session.additional_output), 0);
                encoder.set_buffer(3, Some(&proof_coefficients), proof_offset);
                encoder.set_buffer(4, Some(&coefficient_counts), count_offset);
                encoder.set_buffer(5, Some(&challenges), challenge_offset);
                set_inline_bytes(encoder, 6, &transcript_params);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();

                let next_len = current_len / 2;
                let next_live_len = current_live_len.div_ceil(2);
                self.encode_direct_relation_linear_fold_from_buffer(
                    command,
                    session,
                    &challenges,
                    challenge_offset,
                )?;
                if next_len == 1 {
                    let table = current_table.ok_or_else(|| {
                        MetalCommitError::UnsupportedShape(
                            "resident direct relation compact suffix reaches final fold".into(),
                        )
                    })?;
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 resident direct relation final fold");
                    encoder.set_compute_pipeline_state(&self.fp128_direct_range_finalize_pipeline);
                    encoder.set_buffer(0, Some(&session.tables[table]), 0);
                    encoder.set_buffer(1, Some(&session.final_output), 0);
                    encoder.set_buffer(2, Some(&challenges), challenge_offset);
                    set_inline_bytes(encoder, 3, &(current_live_len as u64));
                    encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                    encoder.end_encoding();
                    if had_linear_terms {
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
                    current_len = 1;
                    current_live_len = next_live_len;
                    rounds_folded += 1;
                    break;
                }

                let fold_lane_weights = rounds_folded >= session.coefficient_rounds;
                let next_alpha_len = if fold_lane_weights {
                    current_alpha_len
                } else {
                    current_alpha_len / 2
                };
                let next_live_lane_count = if fold_lane_weights {
                    current_live_lane_count.div_ceil(2)
                } else {
                    current_live_lane_count
                };
                let next_lane_count = if fold_lane_weights {
                    current_lane_count / 2
                } else {
                    current_lane_count
                };
                let next_lane_weight_table = if fold_lane_weights {
                    1 - current_lane_weight_table
                } else {
                    current_lane_weight_table
                };

                if !fold_lane_weights {
                    let output_alpha_table = 1 - current_alpha_table;
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 resident direct relation alpha fold");
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_alpha_fold_pipeline,
                    );
                    encoder.set_buffer(0, Some(&alpha_tables[current_alpha_table]), 0);
                    encoder.set_buffer(1, Some(&alpha_tables[output_alpha_table]), 0);
                    encoder.set_buffer(2, Some(&challenges), challenge_offset);
                    set_inline_bytes(encoder, 3, &(next_alpha_len as u64));
                    encoder.dispatch_threads(
                        MTLSize::new(next_alpha_len as u64, 1, 1),
                        MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
                    );
                    encoder.end_encoding();
                    current_alpha_table = output_alpha_table;
                }

                let scalar_offset = suffix_round
                    .checked_mul(size_of::<DirectRelationScalars>())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "resident direct relation scalar offset",
                    ))? as u64;
                let next_scalar_offset = (suffix_round + 1)
                    .checked_mul(size_of::<DirectRelationScalars>())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "resident direct relation next scalar offset",
                    ))? as u64;
                let tau_offset = suffix_round.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                    MetalCommitError::ShapeOverflow("resident direct relation tau offset"),
                )? as u64;
                let next_tau_offset = (suffix_round + 1)
                    .checked_mul(size_of::<Fp128Limbs>())
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "resident direct relation next tau offset",
                    ))? as u64;
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 resident direct relation scalar advance");
                encoder.set_compute_pipeline_state(
                    &self.fp128_direct_relation_scalar_advance_pipeline,
                );
                encoder.set_buffer(0, Some(&scalars), scalar_offset);
                encoder.set_buffer(1, Some(&challenges), challenge_offset);
                encoder.set_buffer(2, Some(&tau_buffer), tau_offset);
                encoder.set_buffer(3, Some(&tau_buffer), next_tau_offset);
                encoder.set_buffer(4, Some(&scalars), next_scalar_offset);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();

                let next_mappings = &additional_schedule[suffix_round];
                let next_additional_count = next_mappings.len();
                if current_additional_count != 0 {
                    let mapping_buffer =
                        additional_mappings[suffix_round].as_ref().ok_or_else(|| {
                            MetalCommitError::UnsupportedShape(
                                "resident direct relation additional topology vanished".into(),
                            )
                        })?;
                    let output_additional_table = 1 - current_additional_table;
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 resident direct relation additional fold");
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_additional_fold_pipeline,
                    );
                    encoder.set_buffer(0, Some(&additional_pairs[current_additional_table]), 0);
                    encoder.set_buffer(1, Some(&additional_pairs[output_additional_table]), 0);
                    encoder.set_buffer(2, Some(mapping_buffer), 0);
                    encoder.set_buffer(3, Some(&challenges), challenge_offset);
                    set_inline_bytes(encoder, 4, &(next_additional_count as u64));
                    encoder.dispatch_threads(
                        MTLSize::new(next_additional_count as u64, 1, 1),
                        MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
                    );
                    encoder.end_encoding();
                    current_additional_table = output_additional_table;
                }

                let next_eq = &equality_schedule[suffix_round + 1];
                let additional_parents_in_range = next_mappings
                    .last()
                    .is_none_or(|mapping| mapping.parent < (next_len / 2) as u64);
                let mut params = direct_relation_params_shape(
                    session,
                    next_len,
                    next_lane_count,
                    fold_lane_weights,
                    next_eq.e_first.len(),
                    next_eq.e_second.len(),
                    next_alpha_len,
                    next_live_lane_count,
                    next_additional_count,
                    additional_parents_in_range,
                )?;
                let output_table = current_table.map_or(0, |current| 1 - current);
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 resident direct relation fold and partials");
                if let Some(table) = current_table {
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_field_fold_pipeline,
                    );
                    encoder.set_buffer(0, Some(&session.tables[table]), 0);
                    encoder.set_buffer(12, Some(&challenges), challenge_offset);
                    encoder.set_buffer(
                        15,
                        Some(&session.lane_weight_tables[current_lane_weight_table]),
                        0,
                    );
                } else {
                    params.prefix_size = 1u64 << (rounds_folded + 1);
                    params.materialize_prefix =
                        u64::from(rounds_folded + 1 >= session.compact_prefix_rounds);
                    params.resident_challenges = 1;
                    encoder.set_compute_pipeline_state(
                        &self.fp128_direct_relation_compact_fold_pipeline,
                    );
                    encoder.set_buffer(0, Some(&session.compact_digits), 0);
                    encoder.set_buffer(12, Some(&challenges), 0);
                }
                encoder.set_buffer(1, Some(&session.tables[output_table]), 0);
                encoder.set_buffer(2, Some(&equality_buffers[suffix_round + 1].0), 0);
                encoder.set_buffer(3, Some(&equality_buffers[suffix_round + 1].1), 0);
                encoder.set_buffer(4, Some(&alpha_tables[current_alpha_table]), 0);
                encoder.set_buffer(
                    5,
                    Some(&session.lane_weight_tables[next_lane_weight_table]),
                    0,
                );
                encoder.set_buffer(
                    6,
                    Some(&session.linear_tables[session.current_linear_table]),
                    0,
                );
                encoder.set_buffer(7, Some(&session.linear_source_lane_offsets), 0);
                encoder.set_buffer(8, Some(&session.lane_offsets), 0);
                encoder.set_buffer(9, Some(&session.lane_segments), 0);
                encoder.set_buffer(10, Some(&session.linear_segments), 0);
                encoder.set_buffer(11, Some(&session.partials), 0);
                encoder.set_buffer(13, Some(&scalars), next_scalar_offset);
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

                if next_additional_count != 0 {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 resident direct relation additional round");
                    if current_table.is_none() {
                        encoder.set_compute_pipeline_state(
                            &self.fp128_direct_relation_additional_compact_pipeline,
                        );
                        encoder.set_buffer(0, Some(&session.compact_digits), 0);
                        encoder.set_buffer(1, Some(&challenges), 0);
                        encoder.set_buffer(2, Some(&additional_pairs[current_additional_table]), 0);
                        encoder.set_buffer(3, Some(&session.partials), 0);
                        encoder.set_buffer(4, Some(&scalars), next_scalar_offset);
                        set_inline_bytes(encoder, 5, &params);
                    } else {
                        encoder.set_compute_pipeline_state(
                            &self.fp128_direct_relation_additional_field_pipeline,
                        );
                        encoder.set_buffer(0, Some(&session.tables[output_table]), 0);
                        encoder.set_buffer(1, Some(&additional_pairs[current_additional_table]), 0);
                        encoder.set_buffer(2, Some(&session.partials), 0);
                        encoder.set_buffer(3, Some(&scalars), next_scalar_offset);
                        set_inline_bytes(encoder, 4, &params);
                    }
                    encoder.dispatch_thread_groups(
                        MTLSize::new(params.additional_workgroups, 1, 1),
                        MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
                    );
                    encoder.end_encoding();
                    let mut reduction_params = params;
                    reduction_params.workgroups = params.additional_workgroups;
                    encode_direct_relation_reduction(
                        command,
                        &self.fp128_direct_range_reduce_pipeline,
                        &session.partials,
                        &session.additional_output,
                        &reduction_params,
                    );
                }

                if current_table.is_some() || rounds_folded + 1 >= session.compact_prefix_rounds {
                    current_table = Some(output_table);
                }
                current_len = next_len;
                current_live_len = next_live_len;
                current_live_lane_count = next_live_lane_count;
                current_alpha_len = next_alpha_len;
                current_additional_count = next_additional_count;
                current_lane_weight_table = next_lane_weight_table;
                current_lane_count = next_lane_count;
                rounds_folded += 1;
            }

            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: the completed command initialized all suffix proof slots.
            let proof_values = unsafe {
                std::slice::from_raw_parts(
                    proof_coefficients.contents().cast::<Fp128Limbs>(),
                    proof_coefficient_count,
                )
            };
            let round_coefficients = proof_values
                .chunks_exact(3)
                .map(|values| std::array::from_fn(|index| values[index]))
                .collect::<Vec<_>>();
            // SAFETY: the completed command initialized one count per suffix round.
            let coefficient_counts = unsafe {
                std::slice::from_raw_parts(
                    coefficient_counts.contents().cast::<u32>(),
                    suffix_rounds,
                )
            }
            .iter()
            .map(|&count| count as usize)
            .collect::<Vec<_>>();
            // SAFETY: the completed command initialized every suffix challenge.
            let challenge_values = unsafe {
                std::slice::from_raw_parts(challenges.contents().cast::<Fp128Limbs>(), total_rounds)
            }[prefix_challenges.len()..]
                .to_vec();
            // SAFETY: the final fold initialized both scalar outputs when present.
            let final_evaluation = unsafe { *session.final_output.contents().cast() };
            let final_linear_evaluation = if had_linear_terms {
                unsafe { *session.linear_final_output.contents().cast() }
            } else {
                Fp128Limbs::default()
            };
            let mut next_chaining_value = [0u8; 64];
            // SAFETY: `state` remains a 64-byte shared buffer after completion.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    state.contents().cast::<u8>(),
                    next_chaining_value.as_mut_ptr(),
                    next_chaining_value.len(),
                );
            }
            let readback_copy = readback_start.elapsed();

            session.current_len = current_len;
            session.current_live_len = current_live_len;
            session.current_table = current_table;
            session.current_lane_weight_table = current_lane_weight_table;
            session.current_lane_count = current_lane_count;
            session.rounds_folded = rounds_folded;
            let equality_bytes = equality_schedule
                .iter()
                .try_fold(0usize, |bytes, round| {
                    bytes
                        .checked_add(size_of_val(round.e_first.as_slice()))
                        .and_then(|sum| sum.checked_add(size_of_val(round.e_second.as_slice())))
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct relation equality buffers",
                ))?;
            let mapping_bytes = additional_schedule
                .iter()
                .try_fold(0usize, |bytes, mappings| {
                    bytes.checked_add(size_of_val(mappings.as_slice()))
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct relation additional mappings",
                ))?;
            let allocation_bytes = equality_bytes
                .checked_add(size_of_val(taus.as_slice()))
                .and_then(|bytes| bytes.checked_add(2 * alpha_bytes))
                .and_then(|bytes| bytes.checked_add(size_of_val(scalar_values.as_slice())))
                .and_then(|bytes| bytes.checked_add(2 * pair_bytes))
                .and_then(|bytes| bytes.checked_add(mapping_bytes))
                .and_then(|bytes| bytes.checked_add(size_of_val(challenge_values.as_slice())))
                .and_then(|bytes| bytes.checked_add(proof_bytes))
                .and_then(|bytes| bytes.checked_add(count_bytes + 64))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct relation allocation bytes",
                ))?;
            Ok(DirectRelationResidentOutcome {
                round_coefficients,
                coefficient_counts,
                challenges: challenge_values,
                final_evaluation,
                final_linear_evaluation,
                chaining_value: next_chaining_value,
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
}
