//! Fallible indexed Blake2b streams for epoch-6 sparse challenge sampling.
use akita_error::AkitaError;
use akita_transcript::blake2b_stream::Blake2bStream;

/// Domain for `root[32] || claim-major coordinate index as LE-u64`.
pub const SPARSE_CHALLENGE_STREAM_DOMAIN: &[u8] = b"akita/sparse-challenge/blake2b512/v1";

#[derive(Clone)]
pub(crate) struct IndexedXofPrefix {
    seed: [u8; 32],
}
impl IndexedXofPrefix {
    pub(crate) fn new(seed: &[u8]) -> Result<Self, AkitaError> {
        let seed = seed.try_into().map_err(|_| {
            AkitaError::InvalidInput(
                "indexed sparse challenge group root must be exactly 32 bytes".into(),
            )
        })?;
        Ok(Self { seed })
    }
    fn reader(&self, index: u64) -> Result<Blake2bStream, AkitaError> {
        let mut context = self.seed.to_vec();
        context.extend_from_slice(&index.to_le_bytes());
        Blake2bStream::new(SPARSE_CHALLENGE_STREAM_DOMAIN, &context)
    }
}

pub(crate) struct XofCursor {
    stream: Option<Blake2bStream>,
}
impl XofCursor {
    pub(crate) fn new() -> Self {
        Self { stream: None }
    }
    #[cfg(test)]
    pub(crate) fn from_indexed_prefix(prefix: &IndexedXofPrefix, index: u64) -> Self {
        Self {
            stream: Some(prefix.reader(index).unwrap()),
        }
    }
    pub(crate) fn reset_indexed_prefix(
        &mut self,
        prefix: &IndexedXofPrefix,
        index: u64,
    ) -> Result<(), AkitaError> {
        self.stream = Some(prefix.reader(index)?);
        Ok(())
    }
    pub(crate) fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), AkitaError> {
        self.stream
            .as_mut()
            .ok_or_else(|| AkitaError::InvalidInput("uninitialized sparse stream".into()))?
            .read(out)
    }
    /// Exact masked rejection; every rejected trial consumes the same stream.
    pub(crate) fn next_usize_mod(&mut self, modulus: usize) -> Result<usize, AkitaError> {
        if modulus == 0 {
            return Err(AkitaError::InvalidInput("zero sampling modulus".into()));
        }
        if modulus == 1 {
            return Ok(0);
        }
        let bits = usize::BITS - (modulus - 1).leading_zeros();
        let len = if bits <= 8 {
            1
        } else if bits <= 16 {
            2
        } else if bits <= 32 {
            4
        } else {
            return Err(AkitaError::InvalidInput(
                "sampling modulus exceeds u32".into(),
            ));
        };
        let mask = u32::MAX >> (32 - bits);
        loop {
            let mut bytes = [0; 4];
            self.fill_bytes(&mut bytes[..len])?;
            let value = (u32::from_le_bytes(bytes) & mask) as usize;
            if value < modulus {
                return Ok(value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_root_and_coordinate_stream() {
        let prefix = IndexedXofPrefix::new(&[0; 32]).unwrap();
        let mut cursor = XofCursor::from_indexed_prefix(&prefix, 0);
        let mut actual = [0; 8];
        cursor.fill_bytes(&mut actual).unwrap();
        assert_eq!(actual, [0xf4, 0xa8, 0x94, 0x54, 0x46, 0x08, 0x18, 0xa6]);
        assert!(IndexedXofPrefix::new(&[0; 31]).is_err());
        assert!(IndexedXofPrefix::new(&[0; 33]).is_err());
        cursor.reset_indexed_prefix(&prefix, 1).unwrap();
        let mut other = [0; 8];
        cursor.fill_bytes(&mut other).unwrap();
        assert_ne!(actual, other);
    }
}
