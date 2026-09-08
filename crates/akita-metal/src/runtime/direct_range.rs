use super::*;

impl MetalRuntime {
    #[cfg(test)]
    pub(crate) fn dispatch_fp128_blake2b_sumcheck_challenge(
        &self,
        chaining_value: &[u8; 64],
        prior_squeezed_bytes: usize,
        claim: Option<Fp128Limbs>,
        coefficients: &[Fp128Limbs],
    ) -> Result<Blake2bSumcheckChallengeOutcome, MetalCommitError> {
        if coefficients.is_empty() || coefficients.len() > 4 {
            return Err(MetalCommitError::UnsupportedShape(
                "Blake2b sumcheck challenge requires one to four coefficients".into(),
            ));
        }
        autoreleasepool(|| {
            let include_claim = claim.is_some();
            let state = self.shared_buffer_from_slice(chaining_value)?;
            let claim = self.shared_buffer_from_slice(&[claim.unwrap_or_default()])?;
            let coefficients = self.shared_buffer_from_slice(coefficients)?;
            let challenge = self.shared_buffer(size_of::<Fp128Limbs>())?;
            let params = Blake2bSumcheckChallengeParams {
                include_claim: u64::from(include_claim),
                coefficient_count: coefficients.length() / 16,
                prior_squeezed_bytes: u64::try_from(prior_squeezed_bytes)
                    .map_err(|_| MetalCommitError::ShapeOverflow("Blake2b prior squeeze length"))?,
                reserved: 0,
            };
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 Blake2b sumcheck challenge");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_blake2b_sumcheck_challenge_pipeline);
            encoder.set_buffer(0, Some(&state), 0);
            encoder.set_buffer(1, Some(&claim), 0);
            encoder.set_buffer(2, Some(&coefficients), 0);
            encoder.set_buffer(3, Some(&challenge), 0);
            set_inline_bytes(encoder, 4, &params);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            let _ = complete_command(command)?;

            // SAFETY: both shared output buffers were initialized by the completed kernel.
            let challenge = unsafe { *challenge.contents().cast::<Fp128Limbs>() };
            let mut next_chaining_value = [0u8; 64];
            // SAFETY: `state` is a 64-byte shared buffer valid for the destination length.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    state.contents().cast::<u8>(),
                    next_chaining_value.as_mut_ptr(),
                    next_chaining_value.len(),
                );
            }
            Ok(Blake2bSumcheckChallengeOutcome {
                challenge,
                chaining_value: next_chaining_value,
            })
        })
    }

    #[cfg(test)]
    pub(crate) fn dispatch_fp128_direct_range_resident(
        &self,
        session: &mut DirectRangeSession,
        equality_schedule: &[(Vec<Fp128Limbs>, Vec<Fp128Limbs>)],
        basis: usize,
        chaining_value: &[u8; 64],
        prior_squeezed_bytes: usize,
    ) -> Result<DirectRangeResidentOutcome, MetalCommitError> {
        let num_rounds = equality_schedule.len();
        if num_rounds == 0
            || num_rounds != session.current_len.trailing_zeros() as usize
            || session.current_table.is_some()
            || session.rounds_folded != 0
        {
            return Err(MetalCommitError::UnsupportedShape(
                "resident direct range session has malformed initial state".into(),
            ));
        }
        let coefficient_count = match basis {
            4 => 2usize,
            8 => 4usize,
            _ => {
                return Err(MetalCommitError::UnsupportedShape(
                    "resident direct range proof supports basis four or eight".into(),
                ))
            }
        };

        autoreleasepool(|| {
            let setup_start = Instant::now();
            let equality_buffers = equality_schedule
                .iter()
                .map(|(first, second)| {
                    Ok((
                        self.shared_buffer_from_slice(first)?,
                        self.shared_buffer_from_slice(second)?,
                    ))
                })
                .collect::<Result<Vec<_>, MetalCommitError>>()?;
            let round_output_bytes = num_rounds
                .checked_mul(FP128_DIRECT_RANGE_STORED_COEFFICIENTS)
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct range round outputs",
                ))?;
            let challenge_bytes = num_rounds.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("resident direct range challenges"),
            )?;
            let round_outputs = self.shared_buffer(round_output_bytes)?;
            let challenges = self.shared_buffer(challenge_bytes)?;
            let state = self.shared_buffer_from_slice(chaining_value)?;
            let claim = self.shared_buffer_from_slice(&[Fp128Limbs::default()])?;
            let buffer_setup = setup_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 resident direct range proof");
            let (first, second) = &equality_schedule[0];
            let mut params = direct_range_params(
                session.live_len,
                session.current_len,
                session.current_live_len,
                session.current_live_len,
                first,
                second,
                basis,
            )?;
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 resident direct range initial partials");
            encoder.set_compute_pipeline_state(&self.fp128_direct_range_initial_pipeline);
            encoder.set_buffer(0, Some(&session.compact_digits), 0);
            encoder.set_buffer(1, Some(&equality_buffers[0].0), 0);
            encoder.set_buffer(2, Some(&equality_buffers[0].1), 0);
            encoder.set_buffer(3, Some(&session.partials), 0);
            set_inline_bytes(encoder, 4, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.workgroups, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            encode_direct_range_reduction_at_offset(
                command,
                &self.fp128_direct_range_reduce_pipeline,
                &session.partials,
                &round_outputs,
                0,
                &params,
            );

            let mut current_len = session.current_len;
            let mut current_live_len = session.current_live_len;
            let mut current_table = session.current_table;
            let mut rounds_folded = session.rounds_folded;
            for round in 0..num_rounds {
                let transcript_params = Blake2bSumcheckChallengeParams {
                    include_claim: u64::from(round == 0),
                    coefficient_count: coefficient_count as u64,
                    prior_squeezed_bytes: if round == 0 {
                        u64::try_from(prior_squeezed_bytes).map_err(|_| {
                            MetalCommitError::ShapeOverflow(
                                "resident direct range prior squeeze length",
                            )
                        })?
                    } else {
                        32
                    },
                    reserved: 0,
                };
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 resident direct range challenge");
                encoder.set_compute_pipeline_state(&self.fp128_blake2b_sumcheck_challenge_pipeline);
                encoder.set_buffer(0, Some(&state), 0);
                encoder.set_buffer(1, Some(&claim), 0);
                encoder.set_buffer(
                    2,
                    Some(&round_outputs),
                    (round * FP128_DIRECT_RANGE_STORED_COEFFICIENTS * size_of::<Fp128Limbs>())
                        as u64,
                );
                encoder.set_buffer(
                    3,
                    Some(&challenges),
                    (round * size_of::<Fp128Limbs>()) as u64,
                );
                set_inline_bytes(encoder, 4, &transcript_params);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();

                let next_len = current_len / 2;
                let next_live_len = current_live_len.div_ceil(2);
                if next_len == 1 {
                    let table = current_table.ok_or_else(|| {
                        MetalCommitError::UnsupportedShape(
                            "resident direct range compact prefix reaches final fold".into(),
                        )
                    })?;
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_label("Akita fp128 resident direct range final fold");
                    encoder.set_compute_pipeline_state(&self.fp128_direct_range_finalize_pipeline);
                    encoder.set_buffer(0, Some(&session.tables[table]), 0);
                    encoder.set_buffer(1, Some(&session.final_output), 0);
                    encoder.set_buffer(
                        2,
                        Some(&challenges),
                        (round * size_of::<Fp128Limbs>()) as u64,
                    );
                    set_inline_bytes(encoder, 3, &(current_live_len as u64));
                    encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                    encoder.end_encoding();
                    current_len = 1;
                    current_live_len = next_live_len;
                    break;
                }

                let (first, second) = &equality_schedule[round + 1];
                params = direct_range_params(
                    session.live_len,
                    next_len,
                    next_live_len,
                    current_live_len,
                    first,
                    second,
                    basis,
                )?;
                let output_table = current_table.map_or(0, |current| 1 - current);
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 resident direct range fold and partials");
                if let Some(table) = current_table {
                    encoder
                        .set_compute_pipeline_state(&self.fp128_direct_range_field_fold_pipeline);
                    encoder.set_buffer(0, Some(&session.tables[table]), 0);
                    encoder.set_buffer(
                        5,
                        Some(&challenges),
                        (round * size_of::<Fp128Limbs>()) as u64,
                    );
                } else {
                    let prefix_size = 1usize
                        .checked_shl(u32::try_from(rounds_folded + 1).map_err(|_| {
                            MetalCommitError::ShapeOverflow(
                                "resident direct range compact prefix width",
                            )
                        })?)
                        .ok_or(MetalCommitError::ShapeOverflow(
                            "resident direct range compact prefix size",
                        ))?;
                    params.prefix_size = prefix_size as u64;
                    params.materialize_prefix =
                        u64::from(rounds_folded + 1 >= session.compact_prefix_rounds);
                    params.resident_challenges = 1;
                    encoder
                        .set_compute_pipeline_state(&self.fp128_direct_range_compact_fold_pipeline);
                    encoder.set_buffer(0, Some(&session.compact_digits), 0);
                    encoder.set_buffer(5, Some(&challenges), 0);
                }
                encoder.set_buffer(1, Some(&session.tables[output_table]), 0);
                encoder.set_buffer(2, Some(&equality_buffers[round + 1].0), 0);
                encoder.set_buffer(3, Some(&equality_buffers[round + 1].1), 0);
                encoder.set_buffer(4, Some(&session.partials), 0);
                set_inline_bytes(encoder, 6, &params);
                encoder.dispatch_thread_groups(
                    MTLSize::new(params.workgroups, 1, 1),
                    MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                encode_direct_range_reduction_at_offset(
                    command,
                    &self.fp128_direct_range_reduce_pipeline,
                    &session.partials,
                    &round_outputs,
                    ((round + 1) * FP128_DIRECT_RANGE_STORED_COEFFICIENTS * size_of::<Fp128Limbs>())
                        as u64,
                    &params,
                );

                if current_table.is_some() || rounds_folded + 1 >= session.compact_prefix_rounds {
                    current_table = Some(output_table);
                }
                current_len = next_len;
                current_live_len = next_live_len;
                rounds_folded += 1;
            }

            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: the completed command initialized all round and challenge outputs.
            let round_values = unsafe {
                std::slice::from_raw_parts(
                    round_outputs.contents().cast::<Fp128Limbs>(),
                    num_rounds * FP128_DIRECT_RANGE_STORED_COEFFICIENTS,
                )
            };
            let round_coefficients = round_values
                .chunks_exact(FP128_DIRECT_RANGE_STORED_COEFFICIENTS)
                .map(|values| std::array::from_fn(|index| values[index]))
                .collect::<Vec<_>>();
            // SAFETY: the completed command initialized exactly `num_rounds` challenges.
            let challenge_values = unsafe {
                std::slice::from_raw_parts(challenges.contents().cast::<Fp128Limbs>(), num_rounds)
            }
            .to_vec();
            // SAFETY: the completed command initialized the final scalar output.
            let final_evaluation = unsafe { *session.final_output.contents().cast::<Fp128Limbs>() };
            let mut next_chaining_value = [0u8; 64];
            // SAFETY: `state` remains a 64-byte shared buffer after command completion.
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
            session.rounds_folded = rounds_folded;
            let equality_bytes = equality_schedule
                .iter()
                .try_fold(0usize, |bytes, (first, second)| {
                    bytes
                        .checked_add(size_of_val(first.as_slice()))
                        .and_then(|sum| sum.checked_add(size_of_val(second.as_slice())))
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct range equality buffers",
                ))?;
            let allocation_bytes = equality_bytes
                .checked_add(round_output_bytes)
                .and_then(|bytes| bytes.checked_add(challenge_bytes))
                .and_then(|bytes| bytes.checked_add(64 + size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "resident direct range dispatch allocation bytes",
                ))?;
            Ok(DirectRangeResidentOutcome {
                round_coefficients,
                challenges: challenge_values,
                final_evaluation,
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

    pub(crate) fn begin_fp128_direct_range(
        &self,
        digits: &[i8],
        domain_len: usize,
        compact_prefix_rounds: usize,
    ) -> Result<(DirectRangeSession, Duration), MetalCommitError> {
        if domain_len < 4
            || !domain_len.is_power_of_two()
            || digits.len() > domain_len
            || compact_prefix_rounds == 0
            || compact_prefix_rounds >= domain_len.trailing_zeros() as usize
        {
            return Err(MetalCommitError::UnsupportedShape(
                "direct range proof requires a power-of-two domain of at least four entries".into(),
            ));
        }
        let setup_start = Instant::now();
        let compact_digits = self.shared_buffer_from_slice(digits)?;
        let compact_prefix_size = 1usize
            .checked_shl(u32::try_from(compact_prefix_rounds).map_err(|_| {
                MetalCommitError::ShapeOverflow("direct range compact prefix width")
            })?)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range compact prefix size",
            ))?;
        let first_table_len = digits.len().div_ceil(compact_prefix_size).max(1);
        let second_table_len = first_table_len.div_ceil(2).max(1);
        let first_table_bytes = first_table_len.checked_mul(size_of::<Fp128Limbs>()).ok_or(
            MetalCommitError::ShapeOverflow("direct range first table bytes"),
        )?;
        let second_table_bytes = second_table_len
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range second table bytes",
            ))?;
        let tables = [
            self.private_buffer(first_table_bytes)?,
            self.private_buffer(second_table_bytes)?,
        ];
        let maximum_pairs = digits.len().div_ceil(2);
        let maximum_workgroups = direct_range_workgroups(maximum_pairs);
        let partial_bytes = maximum_workgroups
            .checked_mul(FP128_DIRECT_RANGE_STORED_COEFFICIENTS)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range partial bytes",
            ))?;
        let round_output_bytes = FP128_DIRECT_RANGE_STORED_COEFFICIENTS
            .checked_mul(size_of::<Fp128Limbs>())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range message output bytes",
            ))?;
        let final_output_bytes = size_of::<Fp128Limbs>();
        let partials = self.private_buffer(partial_bytes)?;
        let round_output = self.shared_buffer(round_output_bytes)?;
        let final_output = self.shared_buffer(final_output_bytes)?;
        let allocation_bytes = digits
            .len()
            .checked_add(first_table_bytes)
            .and_then(|bytes| bytes.checked_add(second_table_bytes))
            .and_then(|bytes| bytes.checked_add(partial_bytes))
            .and_then(|bytes| bytes.checked_add(round_output_bytes))
            .and_then(|bytes| bytes.checked_add(final_output_bytes))
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range resident allocation bytes",
            ))?;
        Ok((
            DirectRangeSession {
                compact_digits,
                tables,
                partials,
                round_output,
                final_output,
                live_len: digits.len(),
                current_len: domain_len,
                current_live_len: digits.len(),
                current_table: None,
                compact_prefix_rounds,
                rounds_folded: 0,
                allocation_bytes,
            },
            setup_start.elapsed(),
        ))
    }

    pub(crate) fn dispatch_fp128_direct_range_initial(
        &self,
        session: &DirectRangeSession,
        e_first: &[Fp128Limbs],
        e_second: &[Fp128Limbs],
        basis: usize,
    ) -> Result<DirectRangeRoundOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let params = direct_range_params(
                session.live_len,
                session.current_len,
                session.current_live_len,
                session.current_live_len,
                e_first,
                e_second,
                basis,
            )?;
            let buffer_start = Instant::now();
            let first = self.shared_buffer_from_slice(e_first)?;
            let second = self.shared_buffer_from_slice(e_second)?;
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct range initial round");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 direct range initial partials");
            encoder.set_compute_pipeline_state(&self.fp128_direct_range_initial_pipeline);
            encoder.set_buffer(0, Some(&session.compact_digits), 0);
            encoder.set_buffer(1, Some(&first), 0);
            encoder.set_buffer(2, Some(&second), 0);
            encoder.set_buffer(3, Some(&session.partials), 0);
            set_inline_bytes(encoder, 4, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.workgroups, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            encode_direct_range_reduction(
                command,
                &self.fp128_direct_range_reduce_pipeline,
                &session.partials,
                &session.round_output,
                &params,
            );
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = read_direct_range_coefficients(&session.round_output);
            let readback_copy = readback_start.elapsed();
            Ok(DirectRangeRoundOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: size_of_val(e_first) + size_of_val(e_second),
            })
        })
    }

    pub(crate) fn dispatch_fp128_direct_range_advance(
        &self,
        session: &mut DirectRangeSession,
        challenge: Fp128Limbs,
        next_eq: Option<(&[Fp128Limbs], &[Fp128Limbs])>,
        prefix_weights: &[Fp128Limbs],
        basis: usize,
    ) -> Result<DirectRangeAdvanceOutcome, MetalCommitError> {
        autoreleasepool(|| {
            if session.current_len < 2 || !session.current_len.is_power_of_two() {
                return Err(MetalCommitError::UnsupportedShape(
                    "direct range session has no foldable table".into(),
                ));
            }
            let next_len = session.current_len / 2;
            let next_live_len = session.current_live_len.div_ceil(2);
            if next_len == 1 {
                if next_eq.is_some() {
                    return Err(MetalCommitError::UnsupportedShape(
                        "final direct range fold received a next-round equality table".into(),
                    ));
                }
                let current_table = session.current_table.ok_or_else(|| {
                    MetalCommitError::UnsupportedShape(
                        "direct range compact domain is too small for resident execution".into(),
                    )
                })?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita fp128 direct range final fold");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 direct range final fold");
                encoder.set_compute_pipeline_state(&self.fp128_direct_range_finalize_pipeline);
                encoder.set_buffer(0, Some(&session.tables[current_table]), 0);
                encoder.set_buffer(1, Some(&session.final_output), 0);
                set_inline_bytes(encoder, 2, &challenge);
                set_inline_bytes(encoder, 3, &(session.current_live_len as u64));
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();
                let (command_wall, gpu) = complete_command(command)?;
                let readback_start = Instant::now();
                // SAFETY: `final_output` is a shared buffer containing one fp128 value.
                let final_evaluation =
                    unsafe { *session.final_output.contents().cast::<Fp128Limbs>() };
                let readback_copy = readback_start.elapsed();
                session.current_len = 1;
                session.current_live_len = next_live_len;
                return Ok(DirectRangeAdvanceOutcome {
                    next_coefficients: None,
                    final_evaluation: Some(final_evaluation),
                    timings: DispatchTimings {
                        buffer_setup: Duration::ZERO,
                        command_wall,
                        gpu,
                        readback_copy,
                    },
                    allocation_bytes: 0,
                });
            }

            let (e_first, e_second) = next_eq.ok_or_else(|| {
                MetalCommitError::UnsupportedShape(
                    "non-final direct range fold is missing equality factors".into(),
                )
            })?;
            let mut params = direct_range_params(
                session.live_len,
                next_len,
                next_live_len,
                session.current_live_len,
                e_first,
                e_second,
                basis,
            )?;
            let output_table = session.current_table.map_or(0, |current| 1 - current);
            let buffer_start = Instant::now();
            let first = self.shared_buffer_from_slice(e_first)?;
            let second = self.shared_buffer_from_slice(e_second)?;
            let prefix = if session.current_table.is_none() {
                let expected_prefix_size = 1usize
                    .checked_shl(u32::try_from(session.rounds_folded + 1).map_err(|_| {
                        MetalCommitError::ShapeOverflow("direct range compact prefix width")
                    })?)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "direct range compact prefix size",
                    ))?;
                if prefix_weights.len() != expected_prefix_size {
                    return Err(MetalCommitError::UnsupportedShape(
                        "direct range compact prefix weights have the wrong length".into(),
                    ));
                }
                params.prefix_size = expected_prefix_size as u64;
                params.materialize_prefix =
                    u64::from(session.rounds_folded + 1 >= session.compact_prefix_rounds);
                Some(self.shared_buffer_from_slice(prefix_weights)?)
            } else {
                None
            };
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 direct range fold and next round");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita fp128 direct range fold and next-round partials");
            if let Some(current_table) = session.current_table {
                encoder.set_compute_pipeline_state(&self.fp128_direct_range_field_fold_pipeline);
                encoder.set_buffer(0, Some(&session.tables[current_table]), 0);
            } else {
                encoder.set_compute_pipeline_state(&self.fp128_direct_range_compact_fold_pipeline);
                encoder.set_buffer(0, Some(&session.compact_digits), 0);
            }
            encoder.set_buffer(1, Some(&session.tables[output_table]), 0);
            encoder.set_buffer(2, Some(&first), 0);
            encoder.set_buffer(3, Some(&second), 0);
            encoder.set_buffer(4, Some(&session.partials), 0);
            if let Some(prefix) = &prefix {
                encoder.set_buffer(5, Some(prefix), 0);
            } else {
                set_inline_bytes(encoder, 5, &challenge);
            }
            set_inline_bytes(encoder, 6, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.workgroups, 1, 1),
                MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            encode_direct_range_reduction(
                command,
                &self.fp128_direct_range_reduce_pipeline,
                &session.partials,
                &session.round_output,
                &params,
            );
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            let coefficients = read_direct_range_coefficients(&session.round_output);
            let readback_copy = readback_start.elapsed();
            if session.current_table.is_some()
                || session.rounds_folded + 1 >= session.compact_prefix_rounds
            {
                session.current_table = Some(output_table);
            }
            session.current_len = next_len;
            session.current_live_len = next_live_len;
            session.rounds_folded += 1;
            Ok(DirectRangeAdvanceOutcome {
                next_coefficients: Some(coefficients),
                final_evaluation: None,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes: size_of_val(e_first)
                    + size_of_val(e_second)
                    + size_of_val(prefix_weights),
            })
        })
    }

    pub(crate) fn direct_range_session_allocation_bytes(
        &self,
        session: &DirectRangeSession,
    ) -> usize {
        session.allocation_bytes
    }
}
