//! Canonical fold-draw framing shared by native replay and constrained consumers.
use super::{
    fold_challenge_sample_label, FoldChallengeDrawDomain, SUBRING_COEFFICIENT_PACKING_DRAW_DOMAIN,
};
use crate::{sampler::MAX_STACK_RING_DIM, OperatorNormRejection, SparseChallengeConfig};
use akita_error::AkitaError;

/// Validated public framing around the private little-endian u32 fold nonce.
/// Accessors expose exactly the bytes native FoldDraw absorbs before/after it.
#[derive(Clone, Debug)]
pub struct FoldChallengeFrame {
    prefix: Vec<u8>,
    suffix: Vec<u8>,
    coordinate_count: usize,
}

impl FoldChallengeFrame {
    /// Validate draw metadata and construct its canonical nonce-independent frame.
    pub fn new(
        domain: FoldChallengeDrawDomain,
        ring_d: usize,
        group_index: usize,
        num_live_blocks: usize,
        num_claims: usize,
        cfg: &SparseChallengeConfig,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<Self, AkitaError> {
        if let FoldChallengeDrawDomain::SubringCoefficientPacking {
            challenge_subring_dimension,
        } = domain
        {
            if ring_d != challenge_subring_dimension {
                return Err(AkitaError::InvalidInput(
                    "coefficient-packing draw dimension mismatch".into(),
                ));
            }
            if rejection.is_some() {
                return Err(AkitaError::InvalidInput(
                    "coefficient-packing draws require the L-infinity security route".into(),
                ));
            }
        }
        if ring_d > MAX_STACK_RING_DIM {
            return Err(AkitaError::InvalidInput(format!(
                "ring dimension {ring_d} exceeds supported stack sampler limit ({MAX_STACK_RING_DIM})"
            )));
        }
        cfg.validate_dyn(ring_d).map_err(|e| {
            AkitaError::InvalidInput(format!("invalid sparse challenge config: {e}"))
        })?;
        if let Some(rejection) = rejection {
            rejection
                .validate(ring_d, cfg)
                .map_err(|error| AkitaError::InvalidInput(error.into()))?;
        }
        if num_live_blocks == 0 || num_claims == 0 {
            return Err(AkitaError::InvalidInput(
                "fold challenges require positive num_live_blocks and claims".to_string(),
            ));
        }

        let total = num_live_blocks.checked_mul(num_claims).ok_or_else(|| {
            AkitaError::InvalidSetup("sparse challenge count overflow".to_string())
        })?;
        let sample_label = fold_challenge_sample_label(group_index, num_live_blocks, num_claims)?;
        let domain_sep = cfg.domain_separator_bytes();
        let mut suffix = Vec::new();
        if matches!(
            domain,
            FoldChallengeDrawDomain::SubringCoefficientPacking { .. }
        ) {
            suffix.extend_from_slice(SUBRING_COEFFICIENT_PACKING_DRAW_DOMAIN);
        }
        if let Some(rejection) = rejection {
            suffix.extend_from_slice(&rejection.domain_separator_bytes());
        }
        let mut prefix =
            Vec::with_capacity(sample_label.len() + 8 + 8 + domain_sep.len() + 4 + suffix.len());
        prefix.extend_from_slice(&sample_label);
        prefix.extend_from_slice(&(total as u64).to_le_bytes());
        prefix.extend_from_slice(&(ring_d as u64).to_le_bytes());
        prefix.extend_from_slice(&domain_sep);
        Ok(Self {
            prefix,
            suffix,
            coordinate_count: total,
        })
    }

    /// Payload bytes before the nonce, without the transcript message-length prefix.
    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }
    /// Method and rejection domains appended immediately after the nonce.
    pub fn suffix(&self) -> &[u8] {
        &self.suffix
    }
    /// Checked live-block/claim product encoded in this frame.
    pub fn coordinate_count(&self) -> usize {
        self.coordinate_count
    }

    /// Consume the frame into the exact payload; reuses its prefix allocation.
    pub fn encode(mut self, nonce: u32) -> Vec<u8> {
        self.prefix.extend_from_slice(&nonce.to_le_bytes());
        self.prefix.extend_from_slice(&self.suffix);
        self.prefix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::D64_SELECTIVE_L2_CHALLENGE_CONFIG;

    #[test]
    fn independent_payload_vector_and_private_nonce_slot() {
        // Python struct.pack: domain || LE64(2,3,4) || label || LE64(12,64)
        // || 0 || LE64(31,11) || LE32(nonce) || 2 || LE32(18,48,4,600).
        let golden = "616b6974612f666f6c642d6368616c6c656e67652d726f756e642f7631020000000000000003000000000000000400000000000000616b2f632f77660c000000000000004000000000000000001f000000000000000b00000000000000785634120212000000300000000400000058020000";
        let expected: Vec<_> = golden
            .as_bytes()
            .chunks_exact(2)
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        let frame = FoldChallengeFrame::new(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            2,
            3,
            4,
            &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D64_SELECTIVE_L2),
        )
        .unwrap();
        assert_eq!(frame.coordinate_count(), 12);
        assert_eq!(frame.clone().encode(0x12345678), expected);
        let nonce_start = frame.prefix().len();
        for nonce in [0, u32::MAX] {
            let payload = frame.clone().encode(nonce);
            assert_eq!(&payload[..nonce_start], frame.prefix());
            assert_eq!(&payload[nonce_start..nonce_start + 4], nonce.to_le_bytes());
            assert_eq!(&payload[nonce_start + 4..], frame.suffix());
        }
    }

    #[test]
    fn frame_rejects_invalid_metadata_before_sampling() {
        for (blocks, claims) in [(0, 1), (1, 0), (usize::MAX, 2)] {
            assert!(FoldChallengeFrame::new(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                0,
                blocks,
                claims,
                &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
                Some(OperatorNormRejection::D64_SELECTIVE_L2)
            )
            .is_err());
        }
        assert!(FoldChallengeFrame::new(
            FoldChallengeDrawDomain::SubringCoefficientPacking {
                challenge_subring_dimension: 64
            },
            64,
            0,
            1,
            1,
            &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            Some(OperatorNormRejection::D64_SELECTIVE_L2)
        )
        .is_err());
    }
}
