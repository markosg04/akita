use super::*;

pub(super) fn direct_range_workgroups(pair_count: usize) -> usize {
    pair_count
        .div_ceil(FP128_DIRECT_RANGE_THREADS)
        .clamp(1, FP128_DIRECT_RANGE_MAX_WORKGROUPS)
}

pub(super) fn direct_range_params(
    live_len: usize,
    current_len: usize,
    current_live_len: usize,
    input_live_len: usize,
    e_first: &[Fp128Limbs],
    e_second: &[Fp128Limbs],
    basis: usize,
) -> Result<DirectRangeParams, MetalCommitError> {
    let domain_pair_count = current_len / 2;
    let pair_count = current_live_len.div_ceil(2);
    let equality_entries =
        e_first
            .len()
            .checked_mul(e_second.len())
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct range equality entries",
            ))?;
    if !matches!(basis, 4 | 8)
        || e_first.is_empty()
        || e_second.is_empty()
        || !e_first.len().is_power_of_two()
        || !e_second.len().is_power_of_two()
        || equality_entries != domain_pair_count
        || current_live_len > current_len
        || pair_count > domain_pair_count
    {
        return Err(MetalCommitError::UnsupportedShape(
            "direct range equality factors do not match the round geometry".into(),
        ));
    }
    Ok(DirectRangeParams {
        live_len: u64::try_from(live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range live length"))?,
        current_len: u64::try_from(current_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range current length"))?,
        current_live_len: u64::try_from(current_live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range current live length"))?,
        input_live_len: u64::try_from(input_live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range input live length"))?,
        pair_count: u64::try_from(pair_count)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range pair count"))?,
        num_first: u64::try_from(e_first.len())
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range first equality table"))?,
        num_second: u64::try_from(e_second.len())
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range second equality table"))?,
        workgroups: u64::try_from(direct_range_workgroups(pair_count))
            .map_err(|_| MetalCommitError::ShapeOverflow("direct range workgroups"))?,
        basis: basis as u64,
        prefix_size: 1,
        materialize_prefix: 0,
        resident_challenges: 0,
    })
}

pub(super) fn direct_relation_params(
    session: &DirectRelationSession,
    current_len: usize,
    lane_count: usize,
    fold_lane_weights: bool,
    round: &DirectRelationRoundData<'_>,
) -> Result<DirectRelationParams, MetalCommitError> {
    direct_relation_params_shape(
        session,
        current_len,
        lane_count,
        fold_lane_weights,
        round.e_first.len(),
        round.e_second.len(),
        round.alpha.len(),
        round.live_lane_count,
        round.additional_pairs.len(),
        !round
            .additional_pairs
            .iter()
            .any(|pair| pair.parent >= (current_len / 2) as u64),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "resident relation rounds provide table geometry without host field values"
)]
pub(super) fn direct_relation_params_shape(
    session: &DirectRelationSession,
    current_len: usize,
    lane_count: usize,
    fold_lane_weights: bool,
    e_first_len: usize,
    e_second_len: usize,
    alpha_len: usize,
    live_lane_count: usize,
    additional_pair_count: usize,
    additional_parents_in_range: bool,
) -> Result<DirectRelationParams, MetalCommitError> {
    if current_len < 2 || !current_len.is_power_of_two() {
        return Err(MetalCommitError::UnsupportedShape(
            "direct relation round length is malformed".into(),
        ));
    }
    let domain_pair_count = current_len / 2;
    let current_live_len =
        alpha_len
            .checked_mul(live_lane_count)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation current live length",
            ))?;
    let active_pair_count = current_live_len.div_ceil(2);
    let pair_count = if fold_lane_weights {
        domain_pair_count
    } else {
        active_pair_count
    };
    let equality_entries =
        e_first_len
            .checked_mul(e_second_len)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation equality entries",
            ))?;
    let relation_entries =
        alpha_len
            .checked_mul(lane_count)
            .ok_or(MetalCommitError::ShapeOverflow(
                "direct relation rank-one entries",
            ))?;
    let linear_is_valid = match session.linear_mode {
        0 => true,
        1 => {
            session.linear_source_lane_count != 0 && session.linear_current_coeff_count == alpha_len
        }
        2 => alpha_len == 1 && session.linear_current_live_lane_count == live_lane_count,
        _ => false,
    };
    if e_first_len == 0
        || e_second_len == 0
        || !e_first_len.is_power_of_two()
        || !e_second_len.is_power_of_two()
        || equality_entries != domain_pair_count
        || alpha_len == 0
        || !alpha_len.is_power_of_two()
        || lane_count == 0
        || !lane_count.is_power_of_two()
        || relation_entries != current_len
        || current_live_len > current_len
        || active_pair_count > domain_pair_count
        || live_lane_count > lane_count
        || (fold_lane_weights && alpha_len != 1)
        || !linear_is_valid
        || !additional_parents_in_range
    {
        return Err(MetalCommitError::UnsupportedShape(
            "direct relation factors do not match the round geometry".into(),
        ));
    }
    let additional_workgroups = if additional_pair_count == 0 {
        1
    } else {
        direct_range_workgroups(additional_pair_count)
    };
    Ok(DirectRelationParams {
        live_len: u64::try_from(session.live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation live length"))?,
        current_len: u64::try_from(current_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation current length"))?,
        current_live_len: u64::try_from(current_live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation current live length"))?,
        input_live_len: u64::try_from(session.current_live_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation input live length"))?,
        pair_count: u64::try_from(pair_count)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation pair count"))?,
        num_first: u64::try_from(e_first_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation first equality table"))?,
        num_second: u64::try_from(e_second_len).map_err(|_| {
            MetalCommitError::ShapeOverflow("direct relation second equality table")
        })?,
        workgroups: u64::try_from(direct_range_workgroups(pair_count))
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation workgroups"))?,
        current_coeff_count: u64::try_from(alpha_len)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation coefficient count"))?,
        live_lane_count: u64::try_from(live_lane_count)
            .map_err(|_| MetalCommitError::ShapeOverflow("direct relation live lanes"))?,
        prefix_size: 1,
        materialize_prefix: 0,
        linear_mode: session.linear_mode as u64,
        additional_pair_count: u64::try_from(additional_pair_count).map_err(|_| {
            MetalCommitError::ShapeOverflow("direct relation additional pair count")
        })?,
        additional_workgroups: u64::try_from(additional_workgroups).map_err(|_| {
            MetalCommitError::ShapeOverflow("direct relation additional workgroups")
        })?,
        fold_lane_weights: u64::from(fold_lane_weights),
        resident_challenges: 0,
    })
}

pub(super) fn direct_relation_additional_fold_schedule(
    initial_pairs: &[DirectRelationAdditionalPair],
    transitions: usize,
) -> Result<Vec<Vec<DirectRelationAdditionalFoldMapping>>, MetalCommitError> {
    let mut parents = initial_pairs
        .iter()
        .map(|pair| pair.parent)
        .collect::<Vec<_>>();
    if parents.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(MetalCommitError::UnsupportedShape(
            "direct relation additional parents are not strictly ordered".into(),
        ));
    }
    let mut schedule = Vec::with_capacity(transitions);
    for _ in 0..transitions {
        let mut mappings = Vec::with_capacity(parents.len());
        let mut cursor = 0usize;
        while cursor < parents.len() {
            let parent = parents[cursor] >> 1;
            let mut left = u32::MAX;
            let mut right = u32::MAX;
            while cursor < parents.len() && parents[cursor] >> 1 == parent {
                let index = u32::try_from(cursor).map_err(|_| {
                    MetalCommitError::ShapeOverflow("direct relation additional topology")
                })?;
                if parents[cursor] & 1 == 0 {
                    left = index;
                } else {
                    right = index;
                }
                cursor += 1;
            }
            mappings.push(DirectRelationAdditionalFoldMapping {
                parent,
                left,
                right,
            });
        }
        parents = mappings.iter().map(|mapping| mapping.parent).collect();
        schedule.push(mappings);
    }
    Ok(schedule)
}

pub(super) fn encode_direct_range_reduction(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    partials: &Buffer,
    output: &Buffer,
    params: &DirectRangeParams,
) {
    encode_direct_range_reduction_at_offset(command, pipeline, partials, output, 0, params);
}

pub(super) fn encode_direct_range_reduction_at_offset(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    partials: &Buffer,
    output: &Buffer,
    output_offset: u64,
    params: &DirectRangeParams,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_label("Akita fp128 direct range partial reduction");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(partials), 0);
    encoder.set_buffer(1, Some(output), output_offset);
    set_inline_bytes(encoder, 2, params);
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
    );
    encoder.end_encoding();
}

pub(super) fn encode_direct_relation_reduction(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    partials: &Buffer,
    output: &Buffer,
    params: &DirectRelationParams,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(partials), 0);
    encoder.set_buffer(1, Some(output), 0);
    set_inline_bytes(encoder, 2, params);
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(FP128_DIRECT_RANGE_THREADS as u64, 1, 1),
    );
    encoder.end_encoding();
}

pub(super) fn read_direct_range_coefficients(
    output: &Buffer,
) -> [Fp128Limbs; FP128_DIRECT_RANGE_STORED_COEFFICIENTS] {
    // SAFETY: `output` is shared storage for four initialized fp128 values.
    let values = unsafe {
        std::slice::from_raw_parts(
            output.contents().cast::<Fp128Limbs>(),
            FP128_DIRECT_RANGE_STORED_COEFFICIENTS,
        )
    };
    std::array::from_fn(|index| values[index])
}

pub(super) fn read_direct_relation_coefficients(
    output: &Buffer,
) -> [Fp128Limbs; FP128_DIRECT_RELATION_STORED_COEFFICIENTS] {
    // SAFETY: `output` is shared storage for four initialized fp128 values.
    let values = unsafe {
        std::slice::from_raw_parts(
            output.contents().cast::<Fp128Limbs>(),
            FP128_DIRECT_RELATION_STORED_COEFFICIENTS,
        )
    };
    std::array::from_fn(|index| values[index])
}

pub(super) fn read_direct_relation_prefix_evals(
    output: &Buffer,
) -> ([Fp128Limbs; 8], [Fp128Limbs; 8]) {
    // SAFETY: `output` contains sixteen initialized fp128 values.
    let values = unsafe {
        std::slice::from_raw_parts(
            output.contents().cast::<Fp128Limbs>(),
            FP128_DIRECT_RELATION_TWO_ROUND_PREFIX_OUTPUTS,
        )
    };
    (
        std::array::from_fn(|index| values[index]),
        std::array::from_fn(|index| values[8 + index]),
    )
}

pub(super) fn set_inline_bytes<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<c_void>(),
    );
}

pub(super) fn set_fp128_binding(
    encoder: &ComputeCommandEncoderRef,
    index: u64,
    binding: Fp128KernelBinding<'_>,
) {
    match binding {
        Fp128KernelBinding::Inline(value) => set_inline_bytes(encoder, index, &value),
        Fp128KernelBinding::Buffer(buffer, offset) => {
            encoder.set_buffer(index, Some(buffer), offset);
        }
    }
}

pub(super) fn complete_command(
    command: &CommandBufferRef,
) -> Result<(Duration, Option<Duration>), MetalCommitError> {
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    let wall = start.elapsed();
    validate_completed_command(command)?;
    Ok((wall, completed_command_gpu_time(command)))
}

pub(super) fn validate_completed_command(
    command: &CommandBufferRef,
) -> Result<(), MetalCommitError> {
    let status = command.status();
    if status != MTLCommandBufferStatus::Completed {
        return Err(MetalCommitError::CommandFailed(map_command_status(status)));
    }
    Ok(())
}

pub(super) fn map_command_status(status: MTLCommandBufferStatus) -> CommandStatus {
    match status {
        MTLCommandBufferStatus::NotEnqueued => CommandStatus::NotEnqueued,
        MTLCommandBufferStatus::Enqueued => CommandStatus::Enqueued,
        MTLCommandBufferStatus::Committed => CommandStatus::Committed,
        MTLCommandBufferStatus::Scheduled => CommandStatus::Scheduled,
        MTLCommandBufferStatus::Completed => CommandStatus::Completed,
        MTLCommandBufferStatus::Error => CommandStatus::Error,
    }
}

pub(super) fn command_buffer_timestamp(
    command: &CommandBufferRef,
    name: &'static str,
) -> Option<f64> {
    // SAFETY: `command` is a live `MTLCommandBuffer`, and both selected
    // properties have the Objective-C signature `NSTimeInterval -> f64`.
    unsafe { command.send_message::<(), f64>(Sel::register(name), ()) }.ok()
}

pub(super) fn completed_command_gpu_time(command: &CommandBufferRef) -> Option<Duration> {
    let start = command_buffer_timestamp(command, "GPUStartTime")?;
    let end = command_buffer_timestamp(command, "GPUEndTime")?;
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return None;
    }
    Some(Duration::from_secs_f64(end - start))
}

pub(super) fn completed_commands_gpu_span(
    first: &CommandBufferRef,
    last: &CommandBufferRef,
) -> Option<Duration> {
    let start = command_buffer_timestamp(first, "GPUStartTime")?;
    let end = command_buffer_timestamp(last, "GPUEndTime")?;
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return None;
    }
    Some(Duration::from_secs_f64(end - start))
}
