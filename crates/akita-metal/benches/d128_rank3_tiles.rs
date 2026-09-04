//! E6 experiment: synthetic D128 rank-3 per-ring-element tiled accumulate.
//!
//! Measures the streaming and accumulate cost of a hypothetical D128 rank-3
//! one-hot root kernel that streams one ring element's 2 KB rows per tile,
//! against the production D512 panels kernel (`packed_onehot_commit` bench).
//! Not a commitment: the matrix and lanes are hashed on the device, and
//! `E6_VALIDATE=1` checks the output against a CPU model of the same synthetic
//! accumulate so the timed kernel cannot skip work.

#[cfg(target_os = "macos")]
fn main() {
    use std::ffi::c_void;
    use std::mem::size_of;
    use std::time::{Duration, Instant};

    use jolt_field::{CanonicalEncoding, Prime128OffsetA7F7, Zero};
    use metal::objc::runtime::Sel;
    use metal::objc::Message;
    use metal::{
        Buffer, CommandBuffer, CommandBufferRef, CompileOptions, ComputeCommandEncoderRef,
        ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };

    type F = Prime128OffsetA7F7;

    const D: u64 = 128;
    const N_A: u64 = 3;
    const ONEHOT_K: u64 = 256;
    const PARTIALS: u64 = 16;
    const TILE_POSITIONS: u64 = 16;
    const COUNT_THREADGROUPS: u64 = 4096;

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct E6Params {
        num_rows: u64,
        num_columns: u64,
        positions_per_block: u64,
        blocks_per_column: u64,
        n_a: u64,
        task_offset: u64,
        dispatch_tasks: u64,
        position_partials: u64,
        positions_per_partial: u64,
        output_coefficients: u64,
        density_percent: u64,
        element: u64,
    }
    const _: [(); 96] = [(); size_of::<E6Params>()];

    fn setting(name: &str, default: u64) -> u64 {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn mix(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn matrix_value(index: u64) -> F {
        let a = mix(index);
        let b = mix(a);
        let value = u128::from(a) | (u128::from(b & 0x7fff_ffff_ffff_ffff) << 64);
        F::from_u128_checked(value).expect("hashed matrix value is canonical")
    }

    fn lane(index: u64, density_percent: u64) -> u8 {
        let random = mix(index);
        if random % 100 < density_percent {
            (mix(random) % 255 + 1) as u8
        } else {
            0
        }
    }

    fn set_bytes<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
        encoder.set_bytes(
            index,
            size_of::<T>() as u64,
            std::ptr::from_ref(value).cast::<c_void>(),
        );
    }

    fn gpu_seconds(command: &CommandBufferRef) -> Option<f64> {
        // SAFETY: `command` is a live MTLCommandBuffer; both properties return NSTimeInterval.
        let start =
            unsafe { command.send_message::<(), f64>(Sel::register("GPUStartTime"), ()) }.ok()?;
        let end =
            unsafe { command.send_message::<(), f64>(Sel::register("GPUEndTime"), ()) }.ok()?;
        (start > 0.0 && end >= start).then_some(end - start)
    }

    let log_rows = setting("E6_LOG_ROWS", 16);
    let columns = setting("E6_COLUMNS", 5);
    let positions_per_block = setting("E6_POSITIONS_PER_BLOCK", 8192);
    let density_percent = setting("E6_DENSITY_PERCENT", 27);
    let tasks_per_simdgroup = setting("E6_TASKS_PER_SIMDGROUP", 2);
    let samples = setting("E6_SAMPLES", 1) as usize;
    let streams_per_command = setting("E6_STREAMS_PER_COMMAND", 32);
    let validate = std::env::var("E6_VALIDATE")
        .map(|v| v == "1")
        .unwrap_or(false);
    assert!(matches!(tasks_per_simdgroup, 1 | 2));
    assert!((1..=100).contains(&density_percent));

    let rows = 1u64 << log_rows;
    let positions_per_column = rows * ONEHOT_K / D;
    assert_eq!(positions_per_column % positions_per_block, 0);
    let blocks_per_column = positions_per_column / positions_per_block;
    let tasks = columns * blocks_per_column;
    let output_coefficients = tasks * N_A * D;
    let positions_per_partial = positions_per_block / PARTIALS;
    assert_eq!(positions_per_partial % TILE_POSITIONS, 0);
    let tasks_per_stream = 32 * tasks_per_simdgroup;
    let streams = tasks.div_ceil(tasks_per_stream);
    let matrix_count = N_A * positions_per_block * D;
    let lane_count = rows * columns;

    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let source = format!(
        "{}\n{}",
        include_str!("../src/kernels/onehot.metal"),
        include_str!("d128_rank3_tiles.metal")
    );
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(&source, &options)
        .expect("compile E6 library");
    let pipeline = |name: &str| -> ComputePipelineState {
        let function = library.get_function(name, None).expect(name);
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect(name)
    };
    let fill_matrix = pipeline("e6_fill_matrix");
    let fill_lanes = pipeline("e6_fill_lanes");
    let count_hot = pipeline("e6_count_hot");
    let panels = pipeline(if tasks_per_simdgroup == 1 {
        "e6_d128_rank3_panels_t1"
    } else {
        "e6_d128_rank3_panels_t2"
    });
    let reduce = pipeline("e6_reduce_partials");
    println!(
        "E6_PIPELINE variant=t{tasks_per_simdgroup} max_threads_per_threadgroup={} thread_execution_width={} static_threadgroup_memory={}",
        panels.max_total_threads_per_threadgroup(),
        panels.thread_execution_width(),
        panels.static_threadgroup_memory_length()
    );

    let max_len = device.max_buffer_length();
    let private = |bytes: u64| -> Buffer {
        assert!(
            bytes <= max_len,
            "buffer of {bytes} bytes exceeds device max {max_len}"
        );
        device.new_buffer(bytes, MTLResourceOptions::StorageModePrivate)
    };
    let shared = |bytes: u64| -> Buffer {
        assert!(bytes <= max_len);
        device.new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    };
    let matrix = private(matrix_count * 16);
    let lanes = private(lane_count);
    let partials = private(PARTIALS * output_coefficients * 16);
    let output = shared(output_coefficients * 16);
    let counts = shared(COUNT_THREADGROUPS * 4);

    let params = E6Params {
        num_rows: rows,
        num_columns: columns,
        positions_per_block,
        blocks_per_column,
        n_a: N_A,
        task_offset: 0,
        dispatch_tasks: tasks,
        position_partials: PARTIALS,
        positions_per_partial,
        output_coefficients,
        density_percent,
        element: 0,
    };

    let run = |encode: &dyn Fn(&ComputeCommandEncoderRef)| -> CommandBuffer {
        let command = queue.new_command_buffer().to_owned();
        let encoder = command.new_compute_command_encoder();
        encode(encoder);
        encoder.end_encoding();
        command.commit();
        command
    };

    let setup_start = Instant::now();
    let fill = run(&|encoder| {
        encoder.set_compute_pipeline_state(&fill_matrix);
        encoder.set_buffer(0, Some(&matrix), 0);
        set_bytes(encoder, 1, &matrix_count);
        encoder.dispatch_threads(
            MTLSize::new(matrix_count.min(1 << 24), 1, 1),
            MTLSize::new(256, 1, 1),
        );
    });
    let fill_l = run(&|encoder| {
        encoder.set_compute_pipeline_state(&fill_lanes);
        encoder.set_buffer(0, Some(&lanes), 0);
        set_bytes(encoder, 1, &params);
        encoder.dispatch_threads(MTLSize::new(1 << 24, 1, 1), MTLSize::new(256, 1, 1));
    });
    let count = run(&|encoder| {
        encoder.set_compute_pipeline_state(&count_hot);
        encoder.set_buffer(0, Some(&lanes), 0);
        encoder.set_buffer(1, Some(&counts), 0);
        set_bytes(encoder, 2, &params);
        encoder.dispatch_thread_groups(
            MTLSize::new(COUNT_THREADGROUPS, 1, 1),
            MTLSize::new(1024, 1, 1),
        );
    });
    count.wait_until_completed();
    for command in [&fill, &fill_l, &count] {
        assert_eq!(
            command.status(),
            metal::MTLCommandBufferStatus::Completed,
            "setup command failed"
        );
    }
    // SAFETY: `counts` is live shared storage of exactly COUNT_THREADGROUPS u32 values.
    let hot_entries: u64 = unsafe {
        std::slice::from_raw_parts(counts.contents().cast::<u32>(), COUNT_THREADGROUPS as usize)
    }
    .iter()
    .map(|&count| u64::from(count))
    .sum();
    let setup = setup_start.elapsed();

    let mut results = Vec::with_capacity(samples);
    for _ in 0..samples {
        let wall_start = Instant::now();
        let mut commands = Vec::new();
        for element in 0..N_A {
            for first_stream in (0..streams).step_by(streams_per_command as usize) {
                let chunk_streams = (streams - first_stream).min(streams_per_command);
                let task_offset = first_stream * tasks_per_stream;
                let dispatch_tasks = (chunk_streams * tasks_per_stream).min(tasks - task_offset);
                let dispatch_params = E6Params {
                    task_offset,
                    dispatch_tasks,
                    element,
                    ..params
                };
                let threadgroups = chunk_streams * PARTIALS;
                commands.push(run(&|encoder| {
                    encoder.set_compute_pipeline_state(&panels);
                    encoder.set_buffer(0, Some(&matrix), 0);
                    encoder.set_buffer(1, Some(&lanes), 0);
                    encoder.set_buffer(2, Some(&partials), 0);
                    set_bytes(encoder, 3, &dispatch_params);
                    encoder.dispatch_thread_groups(
                        MTLSize::new(threadgroups, 1, 1),
                        MTLSize::new(1024, 1, 1),
                    );
                }));
            }
        }
        let reduction = run(&|encoder| {
            encoder.set_compute_pipeline_state(&reduce);
            encoder.set_buffer(0, Some(&partials), 0);
            encoder.set_buffer(1, Some(&output), 0);
            set_bytes(encoder, 2, &params);
            encoder.dispatch_threads(
                MTLSize::new(output_coefficients, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        });
        reduction.wait_until_completed();
        let wall = wall_start.elapsed();
        for command in commands.iter().chain(std::iter::once(&reduction)) {
            assert_eq!(
                command.status(),
                metal::MTLCommandBufferStatus::Completed,
                "panels command failed"
            );
        }
        let panel_gpu: f64 = commands.iter().filter_map(|c| gpu_seconds(c)).sum();
        let reduce_gpu = gpu_seconds(&reduction).unwrap_or(0.0);
        // SAFETY: both commands are live completed MTLCommandBuffers.
        let first_start =
            unsafe { commands[0].send_message::<(), f64>(Sel::register("GPUStartTime"), ()) }
                .unwrap_or(0.0);
        let last_end =
            unsafe { reduction.send_message::<(), f64>(Sel::register("GPUEndTime"), ()) }
                .unwrap_or(0.0);
        let span = (last_end - first_start).max(0.0);
        results.push((wall, panel_gpu, reduce_gpu, span, commands.len() + 1));
    }

    let (wall, panel_gpu, reduce_gpu, span, command_count) = results[results.len() - 1];
    let bytes_streamed =
        N_A as f64 * (streams * PARTIALS) as f64 * positions_per_partial as f64 * D as f64 * 16.0;
    println!(
        "E6_D128R3 variant=t{tasks_per_simdgroup} log_rows={log_rows} columns={columns} positions_per_block={positions_per_block} density_percent={density_percent} tasks={tasks} streams={streams} threadgroups={} commands={command_count} hot_entries={hot_entries} setup_ms={:.1} wall_ms={:.1} panel_gpu_ms={:.1} reduce_gpu_ms={:.1} gpu_span_ms={:.1} ns_per_hot_gpu={:.3} ns_per_hot_wall={:.3} bytes_streamed_tb={:.3} samples={samples}",
        N_A * streams * PARTIALS,
        setup.as_secs_f64() * 1e3,
        wall.as_secs_f64() * 1e3,
        panel_gpu * 1e3,
        reduce_gpu * 1e3,
        span * 1e3,
        panel_gpu * 1e9 / hot_entries.max(1) as f64,
        wall.as_secs_f64() * 1e9 / hot_entries.max(1) as f64,
        bytes_streamed / 1e12,
    );
    let _ = Duration::ZERO;

    if validate {
        let cpu_start = Instant::now();
        let mut expected = vec![F::zero(); output_coefficients as usize];
        for row in 0..rows {
            for column in 0..columns {
                let hot = u64::from(lane(row * columns + column, density_percent));
                if hot == 0 {
                    continue;
                }
                let field = row * ONEHOT_K + hot;
                let position = field / D;
                let shift = field % D;
                let block = position / positions_per_block;
                let position_in_block = position % positions_per_block;
                for element in 0..N_A {
                    let base =
                        (((column * blocks_per_column + block) * N_A + element) * D) as usize;
                    for coefficient in 0..D {
                        let source = (coefficient + D - shift) % D;
                        let value = matrix_value(
                            (element * positions_per_block + position_in_block) * D + source,
                        );
                        let slot = &mut expected[base + coefficient as usize];
                        if coefficient >= shift {
                            *slot += value;
                        } else {
                            *slot -= value;
                        }
                    }
                }
            }
        }
        // SAFETY: `output` is live shared storage of exactly output_coefficients [u32; 4] values.
        let got = unsafe {
            std::slice::from_raw_parts(
                output.contents().cast::<[u32; 4]>(),
                output_coefficients as usize,
            )
        };
        let mut mismatches = 0usize;
        for (index, (limbs, want)) in got.iter().zip(&expected).enumerate() {
            let value = u128::from(limbs[0])
                | (u128::from(limbs[1]) << 32)
                | (u128::from(limbs[2]) << 64)
                | (u128::from(limbs[3]) << 96);
            if value != want.to_canonical_u128() {
                if mismatches < 5 {
                    eprintln!(
                        "mismatch at {index}: gpu {value:#034x} cpu {:#034x}",
                        want.to_canonical_u128()
                    );
                }
                mismatches += 1;
            }
        }
        println!(
            "E6_VALIDATE outputs={} mismatches={mismatches} cpu_model_ms={:.1} verdict={}",
            output_coefficients,
            cpu_start.elapsed().as_secs_f64() * 1e3,
            if mismatches == 0 { "OK" } else { "FAIL" }
        );
        assert_eq!(
            mismatches, 0,
            "synthetic accumulate diverged from the CPU model"
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the E6 D128 rank-3 tile experiment requires macOS");
}
