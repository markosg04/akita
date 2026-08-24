use blake2::{Blake2b512, Digest};
use spongefish::DuplexSpongeInterface;

const BLOCK_SIZE: usize = 128;
const DIGEST_SIZE: usize = 64;

/// Resumable public state of Akita's Blake2b transcript sponge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blake2b512Checkpoint {
    /// Chaining value used to separate consecutive duplex operations.
    pub chaining_value: [u8; DIGEST_SIZE],
    /// Current duplex operation.
    pub mode: Blake2b512CheckpointMode,
}

/// Operation-specific portion of a Blake2b transcript checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Blake2b512CheckpointMode {
    /// No absorb or squeeze is in progress.
    Start,
    /// Bytes accumulated for the current absorb operation.
    Absorb { pending: Vec<u8> },
    /// Squeeze cursor and unused bytes from its most recent digest block.
    Squeeze {
        next_block: usize,
        leftovers: Vec<u8>,
    },
}

/// Spongefish's hash-duplex construction with an exportable public checkpoint.
#[derive(Clone)]
pub struct CheckpointBlake2b512 {
    chaining_value: [u8; DIGEST_SIZE],
    mode: Blake2b512CheckpointMode,
}

impl CheckpointBlake2b512 {
    /// Snapshot the public Fiat-Shamir state for accelerator continuation.
    pub fn checkpoint(&self) -> Blake2b512Checkpoint {
        Blake2b512Checkpoint {
            chaining_value: self.chaining_value,
            mode: self.mode.clone(),
        }
    }

    /// Resume from a checkpoint produced by [`Self::checkpoint`].
    pub fn from_checkpoint(checkpoint: Blake2b512Checkpoint) -> Self {
        Self {
            chaining_value: checkpoint.chaining_value,
            mode: checkpoint.mode,
        }
    }

    fn tagged_block(tag: u8) -> [u8; BLOCK_SIZE] {
        let mut block = [0u8; BLOCK_SIZE];
        block[BLOCK_SIZE - 1] = tag;
        block
    }

    fn digest(bytes: &[u8]) -> [u8; DIGEST_SIZE] {
        let digest = Blake2b512::digest(bytes);
        let mut out = [0u8; DIGEST_SIZE];
        out.copy_from_slice(&digest);
        out
    }

    fn end_squeeze(&mut self) {
        let Blake2b512CheckpointMode::Squeeze {
            next_block,
            leftovers,
        } = &self.mode
        else {
            return;
        };
        let byte_count = next_block
            .checked_mul(DIGEST_SIZE)
            .and_then(|count| count.checked_sub(leftovers.len()))
            .expect("valid squeeze cursor");
        let mut input = Vec::with_capacity(BLOCK_SIZE + DIGEST_SIZE + size_of::<usize>());
        input.extend_from_slice(&Self::tagged_block(2));
        input.extend_from_slice(&self.chaining_value);
        input.extend_from_slice(&byte_count.to_be_bytes());
        self.chaining_value = Self::digest(&input);
        self.mode = Blake2b512CheckpointMode::Start;
    }
}

impl Default for CheckpointBlake2b512 {
    fn default() -> Self {
        Self {
            chaining_value: [0u8; DIGEST_SIZE],
            mode: Blake2b512CheckpointMode::Start,
        }
    }
}

impl DuplexSpongeInterface for CheckpointBlake2b512 {
    type U = u8;

    fn absorb(&mut self, input: &[u8]) -> &mut Self {
        self.end_squeeze();
        if matches!(self.mode, Blake2b512CheckpointMode::Start) {
            let mut pending = Vec::with_capacity(BLOCK_SIZE + DIGEST_SIZE + input.len());
            pending.extend_from_slice(&Self::tagged_block(0));
            pending.extend_from_slice(&self.chaining_value);
            self.mode = Blake2b512CheckpointMode::Absorb { pending };
        }
        let Blake2b512CheckpointMode::Absorb { pending } = &mut self.mode else {
            unreachable!("end_squeeze leaves the sponge ready to absorb")
        };
        pending.extend_from_slice(input);
        self
    }

    fn squeeze(&mut self, output: &mut [u8]) -> &mut Self {
        if matches!(self.mode, Blake2b512CheckpointMode::Start) {
            self.mode = Blake2b512CheckpointMode::Squeeze {
                next_block: 0,
                leftovers: Vec::new(),
            };
            return self.squeeze(output);
        }
        if matches!(self.mode, Blake2b512CheckpointMode::Absorb { .. }) {
            self.ratchet();
            return self.squeeze(output);
        }
        if output.is_empty() {
            return self;
        }

        let Blake2b512CheckpointMode::Squeeze {
            next_block,
            leftovers,
        } = &mut self.mode
        else {
            unreachable!("start and absorb modes were handled above")
        };
        if !leftovers.is_empty() {
            let len = output.len().min(leftovers.len());
            output[..len].copy_from_slice(&leftovers[..len]);
            leftovers.drain(..len);
            return self.squeeze(&mut output[len..]);
        }

        let mut input = Vec::with_capacity(BLOCK_SIZE + DIGEST_SIZE + size_of::<usize>());
        input.extend_from_slice(&Self::tagged_block(1));
        input.extend_from_slice(&self.chaining_value);
        input.extend_from_slice(&next_block.to_be_bytes());
        let digest = Self::digest(&input);
        let chunk_len = output.len().min(DIGEST_SIZE);
        output[..chunk_len].copy_from_slice(&digest[..chunk_len]);
        leftovers.extend_from_slice(&digest[chunk_len..]);
        *next_block += 1;
        self.squeeze(&mut output[chunk_len..])
    }

    fn ratchet(&mut self) -> &mut Self {
        self.end_squeeze();
        let first = match &self.mode {
            Blake2b512CheckpointMode::Absorb { pending } => Self::digest(pending),
            Blake2b512CheckpointMode::Start => Self::digest(&[]),
            Blake2b512CheckpointMode::Squeeze { .. } => {
                unreachable!("end_squeeze terminates an active squeeze")
            }
        };
        self.chaining_value = Self::digest(&first);
        self.mode = Blake2b512CheckpointMode::Start;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::CheckpointBlake2b512;
    use spongefish::DuplexSpongeInterface;

    type Reference = spongefish::instantiations::Blake2b512;

    fn assert_squeeze_equal(
        candidate: &mut CheckpointBlake2b512,
        reference: &mut Reference,
        len: usize,
    ) {
        let mut candidate_out = vec![0u8; len];
        let mut reference_out = vec![0u8; len];
        candidate.squeeze(&mut candidate_out);
        reference.squeeze(&mut reference_out);
        assert_eq!(candidate_out, reference_out);
    }

    #[test]
    fn matches_spongefish_hash_duplex_transitions() {
        let mut candidate = CheckpointBlake2b512::default();
        let mut reference = Reference::default();

        candidate
            .absorb(b"protocol")
            .absorb(b"")
            .absorb(b"instance");
        reference
            .absorb(b"protocol")
            .absorb(b"")
            .absorb(b"instance");
        assert_squeeze_equal(&mut candidate, &mut reference, 0);
        assert_squeeze_equal(&mut candidate, &mut reference, 1);
        assert_squeeze_equal(&mut candidate, &mut reference, 31);
        assert_squeeze_equal(&mut candidate, &mut reference, 65);

        candidate.absorb(b"next-message").ratchet().absorb(b"tail");
        reference.absorb(b"next-message").ratchet().absorb(b"tail");
        assert_squeeze_equal(&mut candidate, &mut reference, 127);
        assert_squeeze_equal(&mut candidate, &mut reference, 2);
    }

    #[test]
    fn checkpoint_resumes_after_partial_squeeze() {
        let mut original = CheckpointBlake2b512::default();
        original.absorb(b"prefix").absorb(b"message");
        let mut partial = [0u8; 13];
        original.squeeze(&mut partial);

        let mut resumed = CheckpointBlake2b512::from_checkpoint(original.checkpoint());
        original.absorb(b"round");
        resumed.absorb(b"round");
        let mut original_out = [0u8; 96];
        let mut resumed_out = [0u8; 96];
        original.squeeze(&mut original_out[..32]);
        resumed.squeeze(&mut resumed_out[..32]);
        original
            .absorb(b"round-two")
            .squeeze(&mut original_out[32..]);
        resumed.absorb(b"round-two").squeeze(&mut resumed_out[32..]);
        assert_eq!(original_out, resumed_out);
    }
}
