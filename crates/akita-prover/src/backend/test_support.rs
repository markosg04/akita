use crate::DecomposeFoldWitness;
use akita_field::FieldCore;

pub(crate) fn aggregate_witnesses<F: FieldCore, const D: usize>(
    witnesses: &[DecomposeFoldWitness<F>],
) -> DecomposeFoldWitness<F> {
    let Some((first, rest)) = witnesses.split_first() else {
        panic!("aggregate_witnesses requires at least one witness");
    };
    first
        .ensure_ring_dim::<D>()
        .expect("witness ring dimension");
    let mut centered_coeffs = first.centered_coeffs_owned::<D>();

    for witness in rest {
        witness
            .ensure_ring_dim::<D>()
            .expect("witness ring dimension");
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs_trusted::<D>())
        {
            for k in 0..D {
                dst[k] = dst[k]
                    .checked_add(src[k])
                    .expect("centered coefficient overflow");
            }
        }
    }

    DecomposeFoldWitness::from_centered_coefficients(centered_coeffs)
}
