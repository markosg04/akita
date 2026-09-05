#[cfg(target_os = "macos")]
fn main() {
    use std::hint::black_box;
    use std::time::Instant;

    use akita_challenges::SparseChallenge;
    use akita_metal::{MetalBackend, MetalExecutionPolicy, PackedOneHotCommitView};
    use akita_prover::backend::OneHotPoly;
    use akita_prover::compute::{DecomposeFoldPlan, OpeningFoldKernel, RootOpeningSource};
    use akita_prover::CpuBackend;
    use jolt_field::Prime128OffsetA7F7;

    fn setting(name: &str, default: usize) -> usize {
        std::env::var(name)
            .map(|value| value.parse().unwrap())
            .unwrap_or(default)
    }

    fn mix(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    const D: usize = 128;
    const CAPACITY: usize = 32;
    let log_rows = setting("AKITA_METAL_LOG_ROWS", 18);
    let rows = 1usize << log_rows;
    let columns = setting("AKITA_METAL_COLUMNS", 30);
    let positions = setting("AKITA_METAL_POSITIONS_PER_BLOCK", rows / 512);
    let density = setting("AKITA_METAL_DENSITY_PERCENT", 40);
    assert!((1..=CAPACITY).contains(&columns));
    assert!((1..=100).contains(&density));
    assert!(positions >= 4 && positions.is_power_of_two());
    assert!((rows * 2).is_multiple_of(positions));
    let blocks = rows * 2 / positions;
    let lanes = (0..rows * columns)
        .map(|index| {
            let random = mix(index as u64);
            if random as usize % 100 < density {
                (mix(random) % 255 + 1) as u8
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let source = PackedOneHotCommitView::new(256, CAPACITY, columns, &lanes).unwrap();
    let challenges = (0..CAPACITY * blocks)
        .map(|index| SparseChallenge {
            positions: (0..19)
                .map(|term| (2 * ((17 * index + 7 * term) % 64)) as u32)
                .collect(),
            coeffs: (0..19)
                .map(|term| {
                    if mix((index * 19 + term) as u64) & 1 == 0 {
                        1
                    } else {
                        -1
                    }
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let plan = DecomposeFoldPlan {
        challenges: &challenges,
        num_positions_per_block: positions,
        num_digits: 1,
        log_basis: 3,
    };

    // Keep both positions of each sampled K256 row, spread across the full block.
    let checked_positions = positions.min(1024);
    let sampled_positions = (0..checked_positions)
        .map(|sample| {
            (sample / 2) * (positions / 2 - 1) / (checked_positions / 2 - 1) * 2 + sample % 2
        })
        .collect::<Vec<_>>();
    let mut indices = Vec::with_capacity(CAPACITY * blocks * checked_positions / 2);
    for column in 0..CAPACITY {
        for block in 0..blocks {
            for &position in sampled_positions.iter().step_by(2) {
                let row = (block * positions + position) / 2;
                let hot = if column < columns {
                    lanes[row * columns + column]
                } else {
                    0
                };
                indices.push((hot != 0).then_some(hot));
            }
        }
    }
    let oracle = OneHotPoly::<Prime128OffsetA7F7, u8>::new(256, indices).unwrap();
    let oracle_view = <OneHotPoly<Prime128OffsetA7F7, u8> as RootOpeningSource<
        Prime128OffsetA7F7,
        D,
    >>::opening_view(&oracle)
    .unwrap();
    let cpu_start = Instant::now();
    let expected = CpuBackend::DEFAULT
        .decompose_fold(
            None,
            oracle_view,
            DecomposeFoldPlan {
                num_positions_per_block: checked_positions,
                ..plan
            },
        )
        .unwrap();
    let cpu_sample_s = cpu_start.elapsed().as_secs_f64();
    drop(oracle);

    let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let mut times = Vec::new();
    let mut checksum = 0u64;
    for _ in 0..2 {
        backend.begin_opening_metrics().unwrap();
        let start = Instant::now();
        let actual = black_box(
            backend
                .decompose_fold_packed_onehot::<D>(source, plan)
                .unwrap(),
        );
        times.push(start.elapsed().as_secs_f64());
        for (sample, &position) in sampled_positions.iter().enumerate() {
            assert_eq!(
                actual.centered_coeffs_trusted::<D>()[position],
                expected.centered_coeffs_trusted::<D>()[sample]
            );
        }
        checksum = actual
            .centered_coeffs_flat()
            .iter()
            .flat_map(|coefficient| coefficient.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
            });
    }
    let metrics = backend.last_opening_metrics().unwrap().unwrap();
    assert_eq!(metrics.cpu_fallback_calls, 0);
    println!("AKITA_METAL_PACKED_FOLD log_rows={log_rows} columns={columns} positions={positions} density_percent={density} cold_s={:.6} warm_s={:.6} gpu_s={:.6} buffer_s={:.6} cpu_sample_s={cpu_sample_s:.6} cpu_checked_positions={checked_positions} parity={} checksum={checksum:016x}",
        times[0], times[1], metrics.gpu_active_time.as_secs_f64(), metrics.buffer_setup_time.as_secs_f64(),
        if checked_positions == positions { "full" } else { "sampled" });
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the packed one-hot fold benchmark requires macOS");
}
