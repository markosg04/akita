//! Lazily-reduced full-width accumulator for `Fp128<P>` pseudo-Mersenne
//! fields (`P = 2^128 − C`).
//!
//! [`Fp128Lazy`] keeps a residue in `[0, 2^128)` congruent to the true value
//! mod `P`, folding every add/sub carry back in through `2^128 ≡ C (mod P)`.
//! Corrections are branchless and self-limiting, so the accumulator has no
//! accumulation cap — unlike the digit-limbed [`Fp128x8i32`], which overflows
//! after 2^15 unit-scale additions — and is half its size (16 vs 32 bytes per
//! coefficient). That halves the memory traffic of accumulator read-modify-
//! write streams, the bound resource of the one-hot commit sweep.

use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use num_traits::Zero;

use super::ReduceTo;
use crate::prime::Fp128;
use crate::{AdditiveGroup, CanonicalField};

#[cfg_attr(feature = "jolt-compat", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
/// A residue in `[0, 2^128)` congruent to the represented value mod `P`.
pub struct Fp128Lazy<const P: u128>(pub u128);

impl<const P: u128> Fp128Lazy<P> {
    /// `2^128 mod P`: what one accumulator wrap is congruent to.
    const WRAP: u128 = P.wrapping_neg();

    /// Additive identity accumulator.
    pub const ZERO: Self = Self(0);

    /// `a + b` over residues in `[0, 2^128)`.
    ///
    /// Each carry adds `2^128 ≡ WRAP`. The first correction can wrap again
    /// only when the corrected sum was within `WRAP` of `2^128`, leaving a
    /// value `< WRAP`, so the second correction cannot wrap a third time.
    #[inline(always)]
    fn lazy_add(a: u128, b: u128) -> u128 {
        let (s, c1) = a.overflowing_add(b);
        let (s, c2) = s.overflowing_add(if c1 { Self::WRAP } else { 0 });
        s.wrapping_add(if c2 { Self::WRAP } else { 0 })
    }

    /// `a - b` over residues in `[0, 2^128)`; mirror of [`Self::lazy_add`].
    #[inline(always)]
    fn lazy_sub(a: u128, b: u128) -> u128 {
        let (s, b1) = a.overflowing_sub(b);
        let (s, b2) = s.overflowing_sub(if b1 { Self::WRAP } else { 0 });
        s.wrapping_sub(if b2 { Self::WRAP } else { 0 })
    }
}

impl<const P: u128> From<Fp128<P>> for Fp128Lazy<P> {
    #[inline]
    fn from(x: Fp128<P>) -> Self {
        Self((x.0[0] as u128) | ((x.0[1] as u128) << 64))
    }
}

impl<const P: u128> ReduceTo<Fp128<P>> for Fp128Lazy<P> {
    /// The residue is `< 2^128 < 2P`, so one conditional subtract suffices.
    #[inline]
    fn reduce(self) -> Fp128<P> {
        Fp128::<P>::from_canonical_u128_reduced(self.0)
    }
}

impl<const P: u128> Add for Fp128Lazy<P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(Self::lazy_add(self.0, rhs.0))
    }
}

impl<const P: u128> AddAssign for Fp128Lazy<P> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = Self::lazy_add(self.0, rhs.0);
    }
}

impl<const P: u128> Sub for Fp128Lazy<P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(Self::lazy_sub(self.0, rhs.0))
    }
}

impl<const P: u128> SubAssign for Fp128Lazy<P> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = Self::lazy_sub(self.0, rhs.0);
    }
}

impl<const P: u128> Neg for Fp128Lazy<P> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(Self::lazy_sub(0, self.0))
    }
}

impl<const P: u128> Zero for Fp128Lazy<P> {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        // Both 0 and P represent zero; canonicalize before comparing.
        self.0 == 0 || self.0 == P
    }
}

impl<'a, const P: u128> Add<&'a Self> for Fp128Lazy<P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: &'a Self) -> Self {
        self + *rhs
    }
}

impl<'a, const P: u128> Sub<&'a Self> for Fp128Lazy<P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: &'a Self) -> Self {
        self - *rhs
    }
}

impl<const P: u128> AdditiveGroup for Fp128Lazy<P> {}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u128 = 0xfffffffffffffffffffffffffffffeed;
    type F = Fp128<P>;

    fn canonical(v: Fp128Lazy<P>) -> F {
        v.reduce()
    }

    #[test]
    fn add_sub_match_field_ops_across_wrap_boundaries() {
        // Values chosen to exercise: no carry, single carry, and the
        // double-correction band close to 2^128.
        let samples: [u128; 6] = [
            0,
            1,
            P - 1,
            P / 2 + 12345,
            u128::MAX - Fp128Lazy::<P>::WRAP,
            u128::MAX,
        ];
        for &a_raw in &samples {
            for &b_raw in &samples {
                let (a, b) = (Fp128Lazy::<P>(a_raw), Fp128Lazy::<P>(b_raw));
                let expect_add = canonical(a) + canonical(b);
                let expect_sub = canonical(a) - canonical(b);
                assert_eq!(canonical(a + b), expect_add, "add {a_raw} {b_raw}");
                assert_eq!(canonical(a - b), expect_sub, "sub {a_raw} {b_raw}");
                assert_eq!(canonical(-a), -canonical(a), "neg {a_raw}");
            }
        }
    }

    #[test]
    fn long_accumulation_stays_congruent() {
        // Far beyond the 2^15 cap of the digit-limbed wide accumulator.
        let x = <F as crate::FromPrimitiveInt>::from_u64(0xDEAD_BEEF);
        let mut acc = Fp128Lazy::<P>::zero();
        let mut expect = F::zero();
        for i in 0..200_000u64 {
            if i % 3 == 0 {
                acc -= Fp128Lazy::from(x);
                expect -= x;
            } else {
                acc += Fp128Lazy::from(x);
                expect += x;
            }
        }
        assert_eq!(canonical(acc), expect);
    }
}
