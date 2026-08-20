#[cfg(target_os = "macos")]
mod implementation {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use akita_config::proof_optimized::fp128;
    use akita_field::Prime128OffsetA7F7;
    use akita_metal::{MetalCommitBackend, MetalExecutionPolicy, MetalOneHotKernel};
    use akita_prover::{
        AkitaProverSetup, ComputeBackendSetup, CpuBackend, GroupContext, PackedOneHotPoly,
        UniformProverStack,
    };
    use akita_types::{accumulate_matrix_field_elements_for_level, SetupMatrixCapacity};

    mod packed_onehot_commit_params {
        include!("support/packed_onehot_commit_params.rs");
    }

    use packed_onehot_commit_params::{full_commit_params, workload_num_vars};

    type Cfg = fp128::OneHot;
    type F = Prime128OffsetA7F7;

    const ONEHOT_K: usize = 256;
    const RING_D: usize = 512;
    const COLUMN_CAPACITY: usize = 32;
    const POSITIONS_PER_BLOCK: usize = 524_288;
    const POSITION_PARTIALS: usize = 16;
    const INNER_RANK: usize = 1;

    #[derive(Clone, Copy)]
    struct Workload {
        name: &'static str,
        log_t: usize,
        columns: usize,
        density_percent: usize,
    }

    const fn workload(
        name: &'static str,
        log_t: usize,
        columns: usize,
        density_percent: usize,
    ) -> Workload {
        Workload {
            name,
            log_t,
            columns,
            density_percent,
        }
    }

    const WORKLOADS: &[Workload] = &[
        workload("fp128_d512_k256_t25_c25_d25", 25, 25, 25),
        workload("fp128_d512_k256_t25_c25_d50", 25, 25, 50),
        workload("fp128_d512_k256_t25_c25_d75", 25, 25, 75),
        workload("fp128_d512_k256_t26_c28_d25", 26, 28, 25),
        workload("fp128_d512_k256_t26_c28_d50", 26, 28, 50),
        workload("fp128_d512_k256_t26_c28_d75", 26, 28, 75),
        workload("fp128_d512_k256_t27_c32_d25", 27, 32, 25),
        workload("fp128_d512_k256_t27_c32_d50", 27, 32, 50),
        workload("fp128_d512_k256_t27_c32_d75", 27, 32, 75),
        workload("fp128_d512_k256_t28_c25_d25", 28, 25, 25),
        workload("fp128_d512_k256_t28_c25_d50", 28, 25, 50),
        workload("fp128_d512_k256_t28_c25_d75", 28, 25, 75),
        workload("fp128_d512_k256_t28_c28_d25", 28, 28, 25),
        workload("fp128_d512_k256_t28_c28_d50", 28, 28, 50),
        workload("fp128_d512_k256_t28_c28_d75", 28, 28, 75),
        workload("fp128_d512_k256_t28_c32_d25", 28, 32, 25),
        workload("fp128_d512_k256_t28_c32_d50", 28, 32, 50),
        workload("fp128_d512_k256_t28_c32_d75", 28, 32, 75),
    ];

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = self.0;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^ (value >> 31)
        }
    }

    fn splitmix64_at(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn make_poly(workload: Workload) -> PackedOneHotPoly<F> {
        PackedOneHotPoly::<F>::from_lane_fn(
            ONEHOT_K,
            COLUMN_CAPACITY,
            workload.columns,
            1usize << workload.log_t,
            |index| {
                let index = index as u64;
                let density = splitmix64_at(0x6a09_e667_f3bc_c909 ^ index);
                if usize::try_from(density % 100).unwrap() >= workload.density_percent {
                    0
                } else {
                    let hot = splitmix64_at(0xbb67_ae85_84ca_a73b ^ index);
                    (hot % 255 + 1) as u8
                }
            },
        )
        .unwrap()
    }

    fn mean(values: &[Duration]) -> Duration {
        Duration::from_secs_f64(
            values.iter().map(Duration::as_secs_f64).sum::<f64>() / values.len() as f64,
        )
    }

    fn milliseconds(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1e3
    }

    fn hybrid_cpu_tail_blocks(populated_blocks: usize) -> usize {
        let target = populated_blocks / 12;
        if target == 0 {
            0
        } else {
            1usize << target.ilog2()
        }
    }

    fn paired_bootstrap_ratio_lcb(cpu: &[Duration], metal: &[Duration]) -> f64 {
        assert_eq!(cpu.len(), metal.len());
        let mut rng = SplitMix64(0xbb67_ae85_84ca_a73b);
        let mut ratios = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let mut cpu_sum = 0.0;
            let mut metal_sum = 0.0;
            for _ in 0..cpu.len() {
                let index = rng.next() as usize % cpu.len();
                cpu_sum += cpu[index].as_secs_f64();
                metal_sum += metal[index].as_secs_f64();
            }
            ratios.push(cpu_sum / metal_sum);
        }
        ratios.sort_by(f64::total_cmp);
        ratios[249]
    }

    fn validate_dispatch(
        workload: Workload,
        poly: &PackedOneHotPoly<F>,
        metal: &MetalCommitBackend,
    ) {
        let metrics = metal.last_commit_metrics().unwrap().unwrap();
        let blocks_per_column =
            (1usize << workload.log_t) * ONEHOT_K / RING_D / POSITIONS_PER_BLOCK;
        let cpu_blocks = hybrid_cpu_tail_blocks(blocks_per_column);
        let metal_blocks = blocks_per_column - cpu_blocks;
        let cpu_work_units = workload.columns * cpu_blocks;
        let metal_work_units = workload.columns * metal_blocks;
        let output_coefficients = COLUMN_CAPACITY * blocks_per_column * INNER_RANK * RING_D;
        let output_bytes = output_coefficients * size_of::<F>();
        let scratch_bytes = output_bytes * POSITION_PARTIALS;
        let matrix_bytes = INNER_RANK * POSITIONS_PER_BLOCK * RING_D * size_of::<F>();
        let matrix_streams = metal_work_units.div_ceil(32) * 2;
        let hot_entries = poly.lanes().iter().filter(|&&lane| lane != 0).count();

        assert_eq!(metrics.kernel, MetalOneHotKernel::PackedFp128D512Panels);
        assert!(metrics.input_zero_copy);
        assert!(metrics.matrix_cache_hit);
        assert_eq!(metrics.cpu_blocks, cpu_blocks);
        assert_eq!(metrics.cpu_columns, workload.columns);
        assert_eq!(metrics.cpu_work_units, cpu_work_units);
        assert_eq!(metrics.cpu_rank_rows, INNER_RANK);
        assert_ne!(metrics.cpu_time, Duration::ZERO);
        assert_eq!(metrics.blocks_per_threadgroup, 32);
        assert_eq!(metrics.columns_per_threadgroup, 1);
        assert_eq!(metrics.metal_blocks, metal_blocks);
        assert_eq!(metrics.metal_full_blocks, metal_blocks);
        assert_eq!(metrics.metal_boundary_columns, 0);
        assert_eq!(metrics.metal_columns, workload.columns);
        assert_eq!(metrics.metal_work_units, metal_work_units);
        assert_eq!(metrics.metal_rank_rows, INNER_RANK);
        assert_eq!(metrics.matrix_bytes, matrix_bytes);
        assert_eq!(
            metrics.modeled_matrix_read_bytes,
            (matrix_bytes * matrix_streams) as u64
        );
        assert_eq!(
            metrics.modeled_lane_read_bytes,
            (metal_work_units * POSITIONS_PER_BLOCK * 4) as u64
        );
        assert_eq!(metrics.index_bytes, poly.lanes().len());
        assert_eq!(metrics.output_bytes, output_bytes);
        assert_eq!(metrics.scratch_bytes, scratch_bytes);
        assert_eq!(metrics.hot_entries, hot_entries);
        assert_eq!(metrics.field_additions, (hot_entries * RING_D) as u64);
        assert_eq!(
            metrics.reduction_field_additions,
            (metal_work_units * RING_D * (POSITION_PARTIALS - 1)) as u64
        );
        assert_ne!(metrics.digit_rows_calls, 0);
        assert_eq!(metrics.digit_rows_metal_calls, metrics.digit_rows_calls);
    }

    fn report(
        workload: Workload,
        backend_setup_time: Duration,
        matrix_prepare_time: Duration,
        cpu_times: &[Duration],
        metal_times: &[Duration],
        poly: &PackedOneHotPoly<F>,
        metal: &MetalCommitBackend,
    ) {
        let cpu_mean = mean(cpu_times);
        let metal_mean = mean(metal_times);
        let ratio = cpu_mean.as_secs_f64() / metal_mean.as_secs_f64();
        let lcb = paired_bootstrap_ratio_lcb(cpu_times, metal_times);
        let metrics = metal.last_commit_metrics().unwrap().unwrap();
        let session = std::env::var("AKITA_VALIDATION_SESSION").unwrap_or_else(|_| "adhoc".into());
        let gpu_ms = metrics
            .gpu_time
            .map(milliseconds)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "unavailable".into());
        println!(
            "AKITA_F128_COMMIT_RESULT session={} workload={} samples={} backend_setup_ms={:.6} matrix_prepare_ms={:.6} cpu_mean_ms={:.6} metal_mean_ms={:.6} ratio={:.6} ratio_lcb95={:.6} gpu_ms={} inner_total_ms={:.6} digit_rows_ms={:.6} digit_rows_gpu_ms={:.6} digit_rows_calls={} digit_rows_metal_calls={} compression_ms={:.6} matrix_bytes={} matrix_read_bytes={} lane_bytes={} lane_read_bytes={} output_bytes={} scratch_bytes={} hot_entries={} input_zero_copy={} matrix_cache_hit={} cpu_work_units={} metal_work_units={}",
            session,
            workload.name,
            cpu_times.len(),
            milliseconds(backend_setup_time),
            milliseconds(matrix_prepare_time),
            milliseconds(cpu_mean),
            milliseconds(metal_mean),
            ratio,
            lcb,
            gpu_ms,
            milliseconds(metrics.total_time),
            milliseconds(metrics.digit_rows_time),
            milliseconds(metrics.digit_rows_gpu_time),
            metrics.digit_rows_calls,
            metrics.digit_rows_metal_calls,
            milliseconds(metrics.compression_time),
            metrics.matrix_bytes,
            metrics.modeled_matrix_read_bytes,
            poly.lanes().len(),
            metrics.modeled_lane_read_bytes,
            metrics.output_bytes,
            metrics.scratch_bytes,
            metrics.hot_entries,
            metrics.input_zero_copy,
            metrics.matrix_cache_hit,
            metrics.cpu_work_units,
            metrics.metal_work_units,
        );
        println!(
            "AKITA_F128_COMMIT_RAW session={} workload={} cpu_ns={:?} metal_ns={:?}",
            session,
            workload.name,
            cpu_times.iter().map(Duration::as_nanos).collect::<Vec<_>>(),
            metal_times
                .iter()
                .map(Duration::as_nanos)
                .collect::<Vec<_>>(),
        );
    }

    fn run_workload(workload: Workload, samples: usize) {
        let poly = make_poly(workload);
        let params = full_commit_params(workload);
        let backend_setup_start = Instant::now();
        let mut matrix_fields = 0;
        accumulate_matrix_field_elements_for_level(&params, &mut matrix_fields).unwrap();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            workload_num_vars(workload),
            1,
            SetupMatrixCapacity {
                num_field_elements: matrix_fields,
            },
        )
        .unwrap();
        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let cpu_stack =
            UniformProverStack::uniform(&cpu, &cpu_prepared, setup.expanded.as_ref()).unwrap();
        let metal = MetalCommitBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        let metal_stack =
            UniformProverStack::uniform(&metal, &metal_prepared, setup.expanded.as_ref()).unwrap();
        let backend_setup_time = backend_setup_start.elapsed();
        let polys = [poly];
        let context = GroupContext::explicit_without_precommitted_groups(&params);
        let cpu_commit = || {
            akita_prover::commit::<Cfg, PackedOneHotPoly<F>, CpuBackend>(
                &polys,
                setup.expanded.as_ref(),
                &cpu_stack,
                context,
            )
            .unwrap()
        };
        let metal_commit = || {
            akita_prover::commit::<Cfg, PackedOneHotPoly<F>, MetalCommitBackend>(
                &polys,
                setup.expanded.as_ref(),
                &metal_stack,
                context,
            )
            .unwrap()
        };

        let cpu_warm_start = Instant::now();
        let cpu_warm = cpu_commit();
        let cpu_warm_time = cpu_warm_start.elapsed();
        let metal_warm = metal_commit();
        let first_metal_metrics = metal.last_commit_metrics().unwrap().unwrap();
        assert!(!first_metal_metrics.matrix_cache_hit);
        let matrix_prepare_time = first_metal_metrics.matrix_prepare_time;
        assert_eq!(cpu_warm.committed_group, metal_warm.committed_group);
        assert_eq!(cpu_warm.hint, metal_warm.hint);
        let second_metal_start = Instant::now();
        let second_metal = metal_commit();
        let second_metal_time = second_metal_start.elapsed();
        assert_eq!(cpu_warm.committed_group, second_metal.committed_group);
        assert_eq!(cpu_warm.hint, second_metal.hint);
        validate_dispatch(workload, &polys[0], &metal);
        if samples == 0 {
            let metrics = metal.last_commit_metrics().unwrap().unwrap();
            println!(
                "AKITA_F128_COMMIT_PARITY workload={} backend_setup_ms={:.6} matrix_prepare_ms={:.6} cpu_ms={:.6} metal_ms={:.6} ratio={:.6} inner_ms={:.6} cpu_leg_ms={:.6} command_ms={:.6} gpu_ms={}",
                workload.name,
                milliseconds(backend_setup_time),
                milliseconds(matrix_prepare_time),
                milliseconds(cpu_warm_time),
                milliseconds(second_metal_time),
                cpu_warm_time.as_secs_f64() / second_metal_time.as_secs_f64(),
                milliseconds(metrics.total_time),
                milliseconds(metrics.cpu_time),
                milliseconds(metrics.command_wall_time),
                metrics
                    .gpu_time
                    .map(milliseconds)
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "unavailable".into()),
            );
            return;
        }

        let mut cpu_times = Vec::with_capacity(samples);
        let mut metal_times = Vec::with_capacity(samples);
        for sample in 0..samples {
            if sample.is_multiple_of(2) {
                let start = Instant::now();
                black_box(cpu_commit());
                cpu_times.push(start.elapsed());
                let start = Instant::now();
                black_box(metal_commit());
                metal_times.push(start.elapsed());
            } else {
                let start = Instant::now();
                black_box(metal_commit());
                metal_times.push(start.elapsed());
                let start = Instant::now();
                black_box(cpu_commit());
                cpu_times.push(start.elapsed());
            }
        }
        validate_dispatch(workload, &polys[0], &metal);
        report(
            workload,
            backend_setup_time,
            matrix_prepare_time,
            &cpu_times,
            &metal_times,
            &polys[0],
            &metal,
        );
    }

    pub(super) fn main_impl() {
        let selected =
            std::env::var("AKITA_PACKED_WORKLOAD").unwrap_or_else(|_| WORKLOADS[0].name.into());
        let workload = WORKLOADS
            .iter()
            .copied()
            .find(|workload| workload.name == selected)
            .unwrap_or_else(|| panic!("unknown workload {selected}"));
        let samples = std::env::var("AKITA_PACKED_SAMPLES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(15);
        run_workload(workload, samples);
    }
}

#[cfg(target_os = "macos")]
fn main() {
    implementation::main_impl();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the packed one-hot Metal benchmark requires macOS");
}
