use super::lut::{centered_prime_residue_i128, centered_prime_residue_i64};
use super::*;
use crate::ntt::prime::NttPrime;
use crate::ntt::tables::{I32_RAW_PRIMES, Q32_PRIMES};

const SYNTHETIC_I16_NUM_PRIMES: usize = 3;

fn synthetic_i16_primes() -> [NttPrime<i16>; SYNTHETIC_I16_NUM_PRIMES] {
    [
        NttPrime::compute(15361_i16),
        NttPrime::compute(13313_i16),
        NttPrime::compute(12289_i16),
    ]
}

#[test]
fn centered_prime_residue_keeps_positive_half_boundary() {
    let primes = synthetic_i16_primes();
    let prime16 = primes[0];
    let half16 = i64::from(prime16.p) / 2;
    assert_eq!(centered_prime_residue_i64(prime16, half16), half16 as i16);
    assert_eq!(
        centered_prime_residue_i64(prime16, half16 + 1),
        (half16 + 1 - i64::from(prime16.p)) as i16
    );

    let prime32 = Q32_PRIMES[0];
    let half32 = i64::from(prime32.p) / 2;
    assert_eq!(centered_prime_residue_i64(prime32, half32), half32 as i32);
    assert_eq!(
        centered_prime_residue_i64(prime32, half32 + 1),
        (half32 + 1 - i64::from(prime32.p)) as i32
    );
    assert_eq!(
        centered_prime_residue_i128(prime32, i128::from(half32)),
        half32 as i32
    );
    assert_eq!(
        centered_prime_residue_i128(prime32, i128::from(half32 + 1)),
        (half32 + 1 - i64::from(prime32.p)) as i32
    );
}

#[test]
fn centered_mont_lut_matches_centered_residue_boundary() {
    const D: usize = 64;
    let primes = synthetic_i16_primes();
    let params = CrtNttParamSet::<i16, SYNTHETIC_I16_NUM_PRIMES, D>::new(primes);
    let prime = params.primes[0];
    let half = i32::from(prime.p) / 2;
    let lut = CenteredMontLut::<i16, SYNTHETIC_I16_NUM_PRIMES>::new(&params, half + 1);

    let boundary = centered_prime_residue_i64(prime, i64::from(half));
    let past_boundary = centered_prime_residue_i64(prime, i64::from(half + 1));
    assert_eq!(boundary, half as i16);
    assert_eq!(past_boundary, (half + 1 - i32::from(prime.p)) as i16);
    assert_eq!(lut.get(0, half), Some(prime.from_canonical(boundary)));
    assert_eq!(
        lut.get(0, half + 1),
        Some(prime.from_canonical(past_boundary))
    );
}

#[test]
#[should_panic(expected = "lazy pointwise dot requires an i32 batched parameter set")]
fn lazy_pointwise_dot_rejects_non_i32_parameter_sets() {
    const D: usize = 64;
    let params = CrtNttParamSet::<i16, SYNTHETIC_I16_NUM_PRIMES, D>::new(synthetic_i16_primes());
    let lut = DigitMontLut::new_with_digit_bound(&params, 2);
    let mut accs = [CyclotomicCrtNtt::zero()];
    let matrix_row = [CyclotomicCrtNtt::zero()];
    let ntt_mat = [matrix_row.as_slice()];
    let digits = [[0i8; D]];
    let mut scratch = [[MontCoeff::from_raw(0i16); D]; I32_LAZY_DOT_BATCH];

    CyclotomicCrtNtt::add_assign_col_pointwise_dot_i8_multi_with_lut_scratch(
        &mut accs,
        &ntt_mat,
        0,
        &digits,
        &params,
        &lut,
        &mut scratch,
    );
}

#[test]
fn portable_lazy_dot_satisfies_canonical_montgomery_relation() {
    const D: usize = 64;
    type Ntt = CyclotomicCrtNtt<i32, 1, D>;
    for p in I32_RAW_PRIMES {
        let prime = NttPrime::compute(p);
        for case in 0..4 {
            let lhs: [[MontCoeff<i32>; D]; I32_LAZY_DOT_BATCH] = std::array::from_fn(|product| {
                std::array::from_fn(|lane| {
                    MontCoeff::from_raw(match case {
                        0 => 0,
                        1 => 1,
                        2 => p - 1,
                        _ => ((product as i64 * 7919 + lane as i64 * 104729) % i64::from(p)) as i32,
                    })
                })
            });
            let rhs = std::array::from_fn::<_, I32_LAZY_DOT_BATCH, _>(|product| {
                std::array::from_fn::<_, D, _>(|lane| {
                    MontCoeff::from_raw(if case == 3 {
                        p - 1 - lhs[product][lane].raw()
                    } else {
                        lhs[product][lane].raw()
                    })
                })
            });
            let initial = std::array::from_fn::<_, D, _>(|lane| {
                MontCoeff::from_raw(if lane % 2 == 0 { 0 } else { p - 1 })
            });
            for count in 0..=I32_LAZY_DOT_BATCH {
                let mut actual = initial;
                Ntt::add_assign_pointwise_dot_limb(
                    &mut actual,
                    |product| &lhs[product],
                    |product| &rhs[product],
                    count,
                    prime,
                );
                for lane in 0..D {
                    assert!((0..p).contains(&actual[lane].raw()));
                    let products: i128 = (0..count)
                        .map(|product| {
                            i128::from(lhs[product][lane].raw())
                                * i128::from(rhs[product][lane].raw())
                        })
                        .sum();
                    let delta = i128::from(actual[lane].raw()) - i128::from(initial[lane].raw());
                    assert_eq!(
                        (delta * (1i128 << 32) - products).rem_euclid(i128::from(p)),
                        0,
                        "p={p} case={case} count={count} lane={lane}"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "ntt-inline")]
#[test]
fn fused_signed_digit_transform_preserves_coefficients() {
    use super::convert::CenteredI16NttConverter;
    let params = CrtNttParamSet::<i32, 6, 64>::new(I32_RAW_PRIMES.map(NttPrime::compute));
    for case in 0..5 {
        let coefficients = std::array::from_fn::<_, 64, _>(|lane| match case {
            0 => 0,
            1 => 1,
            2 => i16::MIN,
            3 => i16::MAX,
            _ => (lane as i16).wrapping_mul(7919),
        });
        let converter = CenteredI16NttConverter::new(&params, &[coefficients]);
        let actual = converter
            .transform_inline(&coefficients)
            .centered_coefficients_with_params(&params);
        for limb in actual {
            assert_eq!(limb, coefficients.map(i32::from));
        }
    }
}
