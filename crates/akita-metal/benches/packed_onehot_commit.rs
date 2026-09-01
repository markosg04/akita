#[cfg(target_os = "macos")]
fn main() {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use akita_metal::{
        MetalBackend, MetalExecutionPolicy, MetalOneHotKernel, PackedOneHotCommitView,
    };
    use akita_prover::compute::CommitInnerPlan;
    use akita_prover::{AkitaProverSetup, ComputeBackendSetup};
    use akita_types::SetupMatrixCapacity;
    use jolt_field::Prime128OffsetA7F7;

    const D: usize = 512;

    fn setting(name: &str, default: usize) -> usize {
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

    let log_rows = setting("AKITA_METAL_LOG_ROWS", 12);
    let onehot_k = setting("AKITA_METAL_ONEHOT_K", 256);
    let capacity = match onehot_k {
        16 => 64,
        256 => 32,
        _ => panic!("AKITA_METAL_ONEHOT_K must be 16 or 256"),
    };
    let columns = setting("AKITA_METAL_COLUMNS", 5);
    let positions_per_block = setting("AKITA_METAL_POSITIONS_PER_BLOCK", 64);
    let density_percent = setting("AKITA_METAL_DENSITY_PERCENT", 25);
    let samples = setting("AKITA_METAL_SAMPLES", 5);
    assert!((1..=capacity).contains(&columns));
    assert!((1..=100).contains(&density_percent));
    assert!(samples > 0);

    let rows = 1usize << log_rows;
    let lanes = (0..rows * columns)
        .map(|index| {
            let random = mix(index as u64);
            if random as usize % 100 < density_percent {
                (mix(random) % (onehot_k as u64 - 1) + 1) as u8
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let source = PackedOneHotCommitView::new(onehot_k, capacity, columns, &lanes).unwrap();
    let plan = CommitInnerPlan {
        n_a: 1,
        num_positions_per_block: positions_per_block,
        num_digits_inner: 1,
        log_basis_inner: 3,
    };
    let setup = AkitaProverSetup::<Prime128OffsetA7F7>::generate_with_capacity(
        log_rows + onehot_k.trailing_zeros() as usize + capacity.trailing_zeros() as usize,
        1,
        SetupMatrixCapacity {
            num_field_elements: positions_per_block * D,
        },
    )
    .unwrap();
    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = backend.prepare_setup(&setup).unwrap();

    let warm = backend
        .commit_packed_onehot::<D>(&prepared, source, plan)
        .unwrap();
    let matrix_prepare_time = backend
        .last_commit_metrics()
        .unwrap()
        .unwrap()
        .matrix_prepare_time;
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        let witness = black_box(
            backend
                .commit_packed_onehot::<D>(&prepared, source, plan)
                .unwrap(),
        );
        times.push(start.elapsed());
        assert_eq!(warm.inner_rows, witness.inner_rows);
    }
    let mean = Duration::from_secs_f64(
        times.iter().map(Duration::as_secs_f64).sum::<f64>() / samples as f64,
    );
    let metrics = backend.last_commit_metrics().unwrap().unwrap();
    assert_eq!(metrics.kernel, MetalOneHotKernel::PackedFp128D512Panels);
    assert!(metrics.matrix_cache_hit);
    println!(
        "AKITA_METAL_PACKED_COMMIT log_rows={log_rows} onehot_k={onehot_k} capacity={capacity} columns={columns} positions_per_block={positions_per_block} density_percent={density_percent} samples={samples} mean_ms={:.6} gpu_ms={} panel_active_ms={} panel_span_ms={} reduction_ms={} command_buffers={} matrix_streams={} command_ms={:.6} buffer_setup_ms={:.6} readback_ms={:.6} reconstruct_ms={:.6} total_ms={:.6} matrix_prepare_ms={:.6} lanes_bytes={} output_bytes={} scratch_bytes={} hot_entries={}",
        mean.as_secs_f64() * 1e3,
        metrics
            .gpu_time
            .map(|time| format!("{:.6}", time.as_secs_f64() * 1e3))
            .unwrap_or_else(|| "unavailable".into()),
        metrics
            .panel_gpu_active_time
            .map(|time| format!("{:.6}", time.as_secs_f64() * 1e3))
            .unwrap_or_else(|| "unavailable".into()),
        metrics
            .panel_gpu_span
            .map(|time| format!("{:.6}", time.as_secs_f64() * 1e3))
            .unwrap_or_else(|| "unavailable".into()),
        metrics
            .reduction_gpu_time
            .map(|time| format!("{:.6}", time.as_secs_f64() * 1e3))
            .unwrap_or_else(|| "unavailable".into()),
        metrics.command_buffers,
        metrics.matrix_block_streams,
        metrics.command_wall_time.as_secs_f64() * 1e3,
        metrics.buffer_setup_time.as_secs_f64() * 1e3,
        metrics.readback_copy_time.as_secs_f64() * 1e3,
        metrics.output_reconstruction_time.as_secs_f64() * 1e3,
        metrics.total_time.as_secs_f64() * 1e3,
        matrix_prepare_time.as_secs_f64() * 1e3,
        lanes.len(),
        metrics.output_bytes,
        metrics.scratch_bytes,
        metrics.hot_entries,
    );
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the packed one-hot Metal benchmark requires macOS");
}
