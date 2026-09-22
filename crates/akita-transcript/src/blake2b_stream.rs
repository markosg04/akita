//! Portable Blake2b-512 counter expansion for protocol epoch 6.
//!
//! Block j hashes u32le(domain length) || domain || u64le(context length) ||
//! context || u64le(j). Counter exhaustion is an atomic, fallible boundary.
use akita_error::AkitaError;
use blake2::{Blake2b512, Digest};

/// Total bytes in the full u64 counter domain, including the final block.
pub const BLAKE2B_STREAM_BYTES: u128 = 1u128 << 70;

/// Framed unkeyed Blake2b-512 stream. Partial and empty reads preserve position.
#[derive(Clone)]
pub struct Blake2bStream {
    prefix: Blake2b512,
    position: u128,
    block: [u8; 64],
}

impl Blake2bStream {
    /// Initialize a domain-separated stream with checked canonical lengths.
    pub fn new(domain: &[u8], context: &[u8]) -> Result<Self, AkitaError> {
        let domain_len = u32::try_from(domain.len())
            .map_err(|_| AkitaError::InvalidInput("stream domain exceeds u32".into()))?;
        let context_len = u64::try_from(context.len())
            .map_err(|_| AkitaError::InvalidInput("stream context exceeds u64".into()))?;
        let mut prefix = Blake2b512::new();
        prefix.update(domain_len.to_le_bytes());
        prefix.update(domain);
        prefix.update(context_len.to_le_bytes());
        prefix.update(context);
        Ok(Self {
            prefix,
            position: 0,
            block: [0; 64],
        })
    }

    /// Fill all bytes, or leave output and position unchanged on exhaustion.
    pub fn read(&mut self, output: &mut [u8]) -> Result<(), AkitaError> {
        let end = self
            .position
            .checked_add(output.len() as u128)
            .filter(|&end| end <= BLAKE2B_STREAM_BYTES)
            .ok_or(AkitaError::RandomStreamExhausted)?;
        let mut written = 0;
        while written < output.len() {
            let offset = (self.position % 64) as usize;
            if offset == 0 {
                let index = u64::try_from(self.position / 64)
                    .map_err(|_| AkitaError::RandomStreamExhausted)?;
                let mut hash = self.prefix.clone();
                hash.update(index.to_le_bytes());
                self.block.copy_from_slice(&hash.finalize());
            }
            let take = (64 - offset).min(output.len() - written);
            output[written..written + take].copy_from_slice(&self.block[offset..offset + take]);
            written += take;
            self.position += take as u128;
        }
        debug_assert_eq!(self.position, end);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn decode(hex: &str) -> Vec<u8> {
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }
    #[test]
    fn canonical_python_vectors_and_split_reads() {
        let domain = b"akita/sparse-challenge/blake2b512/v1";
        let expected = decode(concat!("f4a89454460818a6568bf8472b1b6d70c3434cb5116211e148f0f14287ba1f96e0df4954023036e3380b270debb294b10764236f32106d9a8b830a8210e3c312", "06ff8479262b8c4536aec3a2e9cab61487fb7fcd746769cbc422fcd583bd1031661b3d11d78e93671828c1fc14aa388abd8c5cdebf789a663f5858ca54e84e93"));
        let mut stream = Blake2bStream::new(domain, &[0; 40]).unwrap();
        let mut actual = [0; 128];
        stream.read(&mut actual).unwrap();
        assert_eq!(actual.as_slice(), expected);
        for split in [0, 1, 63, 64, 65, 127, 128, 129] {
            let mut whole = Blake2bStream::new(domain, &[0; 40]).unwrap();
            let mut split_stream = whole.clone();
            let mut expected = [0; 256];
            let mut actual = [0; 256];
            whole.read(&mut expected).unwrap();
            split_stream.read(&mut actual[..split]).unwrap();
            split_stream.read(&mut []).unwrap();
            split_stream.read(&mut actual[split..]).unwrap();
            assert_eq!(actual, expected);
        }
        let mut context: Vec<_> = (0..32).collect();
        context.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        let mut stream = Blake2bStream::new(domain, &context).unwrap();
        let mut actual = [0; 64];
        stream.read(&mut actual).unwrap();
        assert_eq!(actual.as_slice(),decode("26a983a538744db2695a0cd5517f5562c0127d749e19d591f2b05fb2961f2daa3682d308581fc9d9acb75e85d33fb22c2c7bc97a2ac89aa6a1d092ff4d07340f"));
    }
    #[test]
    fn final_block_and_atomic_exhaustion() {
        let mut stream = Blake2bStream::new(b"end", b"context").unwrap();
        stream.position = BLAKE2B_STREAM_BYTES - 64;
        let mut too_long = [19; 65];
        assert_eq!(
            stream.read(&mut too_long),
            Err(AkitaError::RandomStreamExhausted)
        );
        assert_eq!(too_long, [19; 65]);
        let mut actual = [0; 64];
        stream.read(&mut actual).unwrap();
        let mut hash = stream.prefix.clone();
        hash.update(u64::MAX.to_le_bytes());
        assert_eq!(actual.as_slice(), hash.finalize().as_slice());
        stream.read(&mut []).unwrap();
        assert_eq!(
            stream.read(&mut [0]),
            Err(AkitaError::RandomStreamExhausted)
        );
    }
    #[test]
    fn domain_and_context_framing_separate_streams() {
        let mut outputs = Vec::new();
        for (domain, context) in [
            (b"a".as_slice(), b"bc".as_slice()),
            (b"ab", b"c"),
            (b"a", b"bd"),
        ] {
            let mut out = [0; 64];
            Blake2bStream::new(domain, context)
                .unwrap()
                .read(&mut out)
                .unwrap();
            outputs.push(out);
        }
        assert_ne!(outputs[0], outputs[1]);
        assert_ne!(outputs[0], outputs[2]);
    }
}
