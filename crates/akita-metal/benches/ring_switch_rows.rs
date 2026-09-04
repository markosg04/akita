#[cfg(target_os = "macos")]
fn main() {
    use std::hint::black_box;
    use std::time::Instant;

    use akita_metal::{MetalBackend, MetalExecutionPolicy};
    use akita_prover::backend::RingSwitchRelationView;
    use akita_prover::compute::{RingSwitchRelationKernel, RingSwitchRelationPlan};
    use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend};
    use akita_types::SetupMatrixCapacity;
    use jolt_field::Prime128OffsetA7F7;

    const D: usize = 64;
    let columns = std::env::var("AKITA_METAL_D_ROLE_COLUMNS")
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(16_384);
    let num_rows = std::env::var("AKITA_METAL_D_ROLE_ROWS")
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(3);
    assert!(columns > 0 && num_rows > 0);
    let setup = AkitaProverSetup::<Prime128OffsetA7F7>::generate_with_capacity(
        20,
        1,
        SetupMatrixCapacity {
            num_field_elements: akita_error::checked::product([columns, num_rows, D]).unwrap(),
        },
    )
    .unwrap();
    let digits = (0..columns)
        .map(|column| {
            std::array::from_fn(|coefficient| {
                let mut value = (column * D + coefficient) as u64;
                value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((value ^ (value >> 31)) & 7) as i8 - 4
            })
        })
        .collect::<Vec<[i8; D]>>();
    let source = RingSwitchRelationView {
        e_hat: &digits,
        t_hat: &[],
        z_segment: &[],
        z_folded_centered_inf_norm: 0,
    };
    let plan = RingSwitchRelationPlan {
        n_d: num_rows,
        n_b: 0,
        n_a: 0,
        log_basis_open: 3,
        log_basis_outer: 3,
    };
    let cpu = CpuBackend::DEFAULT;
    let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
    let cpu_start = Instant::now();
    let expected = cpu.relation_rows(&cpu_prepared, source, plan).unwrap();
    let cpu_s = cpu_start.elapsed().as_secs_f64();
    drop(cpu_prepared);

    let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
    let prepared = metal.prepare_setup(&setup).unwrap();
    metal.begin_opening_metrics().unwrap();
    let cold_start = Instant::now();
    let cold = metal.relation_rows(&prepared, source, plan).unwrap();
    let cold_s = cold_start.elapsed().as_secs_f64();
    assert_eq!(cold, expected);

    metal.begin_opening_metrics().unwrap();
    let start = Instant::now();
    let actual = black_box(metal.relation_rows(&prepared, source, plan).unwrap());
    let warm_s = start.elapsed().as_secs_f64();
    assert_eq!(actual, expected);
    let metrics = metal.last_opening_metrics().unwrap().unwrap();
    assert_eq!(metrics.cpu_fallback_calls, 0);
    let checksum = actual
        .d_negacyclic
        .iter()
        .chain(&actual.d_cyclic)
        .flat_map(|row| &row.coeffs)
        .flat_map(|value| value.to_canonical_u128().to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        });
    println!(
        "AKITA_METAL_D_ROLE columns={columns} rows={num_rows} cpu_s={cpu_s:.6} cold_s={cold_s:.6} warm_s={warm_s:.6} gpu_s={:.6} buffer_s={:.6} parity=true checksum={checksum:016x}",
        metrics.gpu_active_time.as_secs_f64(),
        metrics.buffer_setup_time.as_secs_f64(),
    );
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the ring-switch row benchmark requires macOS");
}
