use jolt_field::{CanonicalEncoding, Field, Prime128OffsetA7F7};

use crate::MetalCommitError;

pub(crate) type F = Prime128OffsetA7F7;

pub(crate) trait MetalField: CanonicalEncoding + Field {
    type DeviceElement: Copy;

    fn into_device(self) -> Self::DeviceElement;

    fn from_device(value: Self::DeviceElement, index: usize) -> Result<Self, MetalCommitError>;
}

/// Canonical little-endian limbs shared with MSL buffers.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Fp128Limbs {
    pub(crate) limbs: [u32; 4],
}

impl Fp128Limbs {
    pub(crate) const fn from_u128(value: u128) -> Self {
        Self {
            limbs: [
                value as u32,
                (value >> 32) as u32,
                (value >> 64) as u32,
                (value >> 96) as u32,
            ],
        }
    }

    pub(crate) const fn to_u128(self) -> u128 {
        (self.limbs[0] as u128)
            | ((self.limbs[1] as u128) << 32)
            | ((self.limbs[2] as u128) << 64)
            | ((self.limbs[3] as u128) << 96)
    }

    pub(crate) fn from_field(value: F) -> Self {
        Self::from_u128(value.to_canonical_u128())
    }

    pub(crate) fn into_field(self, index: usize) -> Result<F, MetalCommitError> {
        F::from_u128_checked(self.to_u128()).ok_or(MetalCommitError::NonCanonicalOutput { index })
    }
}

impl MetalField for F {
    type DeviceElement = Fp128Limbs;

    fn into_device(self) -> Self::DeviceElement {
        Fp128Limbs::from_field(self)
    }

    fn from_device(value: Self::DeviceElement, index: usize) -> Result<Self, MetalCommitError> {
        value.into_field(index)
    }
}

const _: [(); 16] = [(); size_of::<Fp128Limbs>()];
const _: [(); 16] = [(); align_of::<Fp128Limbs>()];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_limb_boundary_round_trips() {
        let values = [
            0,
            1,
            (1u128 << 64) - 1,
            1u128 << 64,
            0xffff_ffff_ffff_ffff_ffff_ffff_0000_5808,
        ];
        for value in values {
            let field = F::from_u128_checked(value).unwrap();
            let limbs = Fp128Limbs::from_field(field);
            assert_eq!(limbs.to_u128(), value);
            assert_eq!(limbs.into_field(0).unwrap(), field);
        }
    }

    #[test]
    fn modulus_is_rejected() {
        let modulus = 0xffff_ffff_ffff_ffff_ffff_ffff_0000_5809;
        assert!(Fp128Limbs::from_u128(modulus).into_field(0).is_err());
    }

    #[test]
    fn field_storage_matches_device_limbs() {
        assert_eq!(size_of::<F>(), size_of::<Fp128Limbs>());
        let field = F::from_u128_checked(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210).unwrap();
        // SAFETY: Fp128 is transparent over two little-endian u64 limbs and
        // Fp128Limbs is the same 16-byte value split into four u32 limbs.
        let stored =
            unsafe { std::ptr::read_unaligned(std::ptr::from_ref(&field).cast::<Fp128Limbs>()) };
        assert_eq!(stored, Fp128Limbs::from_field(field));
    }
}
