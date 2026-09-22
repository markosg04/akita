//! Fallible epoch-6 matrix-entry Blake2b streams. IDs 0 and 1 are retired.
use akita_error::AkitaError;
use akita_transcript::blake2b_stream::Blake2bStream;

/// Domain for the canonical seed/label/geometry/entry context.
pub const MATRIX_ENTRY_STREAM_DOMAIN: &[u8] = b"akita/matrix-entry/blake2b512/v1";

/// Stable backend identifier; old SHAKE/AES values are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MatrixPrgBackendId {
    /// Framed Blake2b-512 counter expansion.
    Blake2b512 = 2,
}
impl TryFrom<u8> for MatrixPrgBackendId {
    type Error = AkitaError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::Blake2b512),
            _ => Err(AkitaError::InvalidInput(format!(
                "unsupported matrix PRG backend id: {value}"
            ))),
        }
    }
}
impl From<MatrixPrgBackendId> for u8 {
    fn from(value: MatrixPrgBackendId) -> Self {
        value as u8
    }
}

/// Canonical context for a legacy per-entry matrix stream, distinct from setup pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixPrgContext<'a> {
    /// Public seed.
    pub seed: &'a [u8; 32],
    /// Matrix label bytes.
    pub matrix_label: &'a [u8],
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub cols: usize,
    /// Entry row, strictly below rows.
    pub row: usize,
    /// Entry column, strictly below cols.
    pub col: usize,
}
impl MatrixPrgContext<'_> {
    /// Construct a fallible stream; no infallible RNG adapter is supplied.
    pub fn stream(&self) -> Result<Blake2bStream, AkitaError> {
        if self.row >= self.rows || self.col >= self.cols {
            return Err(AkitaError::InvalidInput(
                "matrix entry outside geometry".into(),
            ));
        }
        let mut context = self.seed.to_vec();
        let label_len = u64::try_from(self.matrix_label.len())
            .map_err(|_| AkitaError::InvalidInput("matrix label exceeds u64".into()))?;
        context.extend_from_slice(&label_len.to_le_bytes());
        context.extend_from_slice(self.matrix_label);
        for value in [self.rows, self.cols, self.row, self.col] {
            let value = u64::try_from(value)
                .map_err(|_| AkitaError::InvalidInput("matrix geometry exceeds u64".into()))?;
            context.extend_from_slice(&value.to_le_bytes());
        }
        Blake2bStream::new(MATRIX_ENTRY_STREAM_DOMAIN, &context)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_backends_and_invalid_entries_rejected() {
        assert!(MatrixPrgBackendId::try_from(0).is_err());
        assert!(MatrixPrgBackendId::try_from(1).is_err());
        assert_eq!(
            MatrixPrgBackendId::try_from(2).unwrap(),
            MatrixPrgBackendId::Blake2b512
        );
        let seed = [0; 32];
        let mut ctx = MatrixPrgContext {
            seed: &seed,
            matrix_label: b"A",
            rows: 2,
            cols: 2,
            row: 0,
            col: 0,
        };
        let mut a = [0; 64];
        ctx.stream().unwrap().read(&mut a).unwrap();
        assert_eq!(
            a,
            [
                107, 101, 250, 251, 117, 81, 122, 174, 247, 191, 95, 253, 230, 45, 129, 236, 107,
                63, 192, 161, 21, 172, 7, 249, 124, 39, 229, 223, 111, 81, 58, 63, 129, 167, 125,
                142, 127, 99, 155, 15, 250, 152, 216, 162, 228, 35, 47, 188, 141, 54, 4, 109, 195,
                141, 164, 151, 252, 199, 5, 86, 66, 38, 160, 170
            ]
        );
        ctx.col = 1;
        let mut b = [0; 64];
        ctx.stream().unwrap().read(&mut b).unwrap();
        assert_ne!(a, b);
        ctx.col = 2;
        assert!(ctx.stream().is_err());
    }
}
