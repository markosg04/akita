//! Shared setup data shapes for Akita prover and verifier APIs.

use super::setup_prefix::SetupPrefixVerifierRegistry;
use crate::FlatMatrix;
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_transcript::blake2b_stream::Blake2bStream;
#[allow(unused_imports)]
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::io::{Read, Write};
use std::sync::Arc;

/// Versioned derivation algorithm for the public field stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicMatrixDerivation {
    /// Fixed 4096-field-element Blake2b-512 pages with exact field sampling.
    Blake2b512PagedV2,
}

impl PublicMatrixDerivation {
    /// Number of field elements in one independently derived page.
    #[must_use]
    pub const fn page_field_elements(self) -> usize {
        match self {
            Self::Blake2b512PagedV2 => 4096,
        }
    }
}

/// Semantic identity of the infinite public field stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaSetupSeed {
    /// Coefficient derivation algorithm.
    pub derivation: PublicMatrixDerivation,
    /// Public entropy absorbed by the derivation algorithm.
    pub seed: [u8; 32],
}

impl AkitaSetupSeed {
    /// Construct a v2 paged Blake2b-512 public-matrix identity.
    #[must_use]
    pub const fn blake2b512_paged_v2(seed: [u8; 32]) -> Self {
        Self {
            derivation: PublicMatrixDerivation::Blake2b512PagedV2,
            seed,
        }
    }
}

impl From<[u8; 32]> for AkitaSetupSeed {
    fn from(seed: [u8; 32]) -> Self {
        Self::blake2b512_paged_v2(seed)
    }
}

/// Maximum setup matrix field elements accepted by self-describing setup
/// deserialization.
///
/// This cap protects generic verifier-facing setup decoding from allocating
/// directly from attacker-controlled seed metadata. It is not a protocol limit
/// on the deterministic public stream. Context-backed decoders should instead
/// enforce an expected shape and caller-supplied resource budget.
pub const MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS: usize = 1 << 26;

/// Domain for epoch-6 seed, field modulus, page size and page index contexts.
pub const PUBLIC_MATRIX_STREAM_DOMAIN: &[u8] = b"akita/public-matrix/blake2b512-paged/v2";

/// Exact base-field capacity of the shared public setup vector.
///
/// The setup stores one flat vector of field elements. A/B/D matrices are
/// role-local prefix views of this vector, so capacity is the maximum required
/// role footprint, not `max_rows * max_stride`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetupMatrixCapacity {
    /// Number of materialized base-field elements.
    pub num_field_elements: usize,
}

impl SetupMatrixCapacity {
    /// Smallest non-empty shared setup capacity.
    pub const fn minimum() -> Self {
        Self {
            num_field_elements: std::num::NonZeroUsize::MIN.get(),
        }
    }
}

/// Seed-only stage for deterministic setup expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaSetupDescriptor {
    /// Provisioning variable bound inherited from setup construction.
    pub max_num_vars: usize,
    /// Provisioning polynomial-count bound inherited from setup construction.
    pub max_num_batched_polys: usize,
    /// Number of materialized public-matrix field elements.
    pub num_field_elements: usize,
    /// Semantic identity of the infinite public field stream.
    pub setup_seed: AkitaSetupSeed,
}

/// Expanded setup stage containing materialized public matrices.
///
/// Base role matrices (A, B, D) are packed row/column prefix views of
/// `shared_matrix`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaExpandedSetup<F: Field> {
    /// Setup seed and runtime layout metadata.
    pub descriptor: AkitaSetupDescriptor,
    /// Shared 1D flat backing vector.
    pub shared_matrix: FlatMatrix<F>,
}

/// Verifier setup artifact derived from prover setup.
///
/// Semantic state is immutable so clones can safely share prepared matrix
/// caches. To replace a setup, construct a new value with [`Self::from_parts`].
///
/// ```compile_fail
/// use akita_types::AkitaVerifierSetup;
/// use jolt_field::Prime32Offset99;
/// fn replace(setup: &mut AkitaVerifierSetup<Prime32Offset99>,
///            other: AkitaVerifierSetup<Prime32Offset99>) {
///     setup.expanded = other.expanded;
/// }
/// ```
///
/// ```compile_fail
/// use akita_types::AkitaVerifierSetup;
/// use jolt_field::Prime32Offset99;
/// fn replace(setup: &mut AkitaVerifierSetup<Prime32Offset99>,
///            other: AkitaVerifierSetup<Prime32Offset99>) {
///     setup.prefix_slots = other.prefix_slots;
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AkitaVerifierSetup<F: Field> {
    /// Expanded matrix stage used for verification.
    expanded: Arc<AkitaExpandedSetup<F>>,
    /// Public setup-prefix commitment metadata for setup-claim offloading.
    prefix_slots: SetupPrefixVerifierRegistry<F>,
    /// Locally derived, negacyclic-only matrix prefixes for direct verifier checks.
    /// This performance cache is neither serialized nor part of setup identity.
    verifier_ntt: Arc<crate::ntt_cache::VerifierNttCache>,
}

impl<F: Field> PartialEq for AkitaVerifierSetup<F> {
    fn eq(&self, other: &Self) -> bool {
        self.expanded == other.expanded && self.prefix_slots == other.prefix_slots
    }
}

impl<F: Field> Eq for AkitaVerifierSetup<F> {}

impl<F: Field> AkitaVerifierSetup<F> {
    /// Borrow the immutable expanded matrix stage.
    pub fn expanded(&self) -> &Arc<AkitaExpandedSetup<F>> {
        &self.expanded
    }

    /// Borrow the public setup-prefix commitment metadata.
    pub fn prefix_slots(&self) -> &SetupPrefixVerifierRegistry<F> {
        &self.prefix_slots
    }

    /// Construct verifier setup state from expanded setup and structurally checked prefix metadata.
    ///
    /// This constructor binds the registry to the public matrix identity. It
    /// does not prove that each stored prefix commitment was derived from that
    /// matrix. Callers loading external registries must establish that
    /// provenance at their setup-installation boundary.
    pub fn from_parts(
        expanded: Arc<AkitaExpandedSetup<F>>,
        prefix_slots: SetupPrefixVerifierRegistry<F>,
    ) -> Result<Self, AkitaError> {
        if prefix_slots.setup_seed() != &expanded.descriptor.setup_seed {
            return Err(AkitaError::InvalidSetup(
                "setup-prefix registry belongs to a different public matrix".to_string(),
            ));
        }
        Ok(Self {
            expanded,
            prefix_slots,
            verifier_ntt: Arc::new(crate::ntt_cache::VerifierNttCache::default()),
        })
    }

    /// In-memory byte footprint of verifier NTT prefixes materialized so far.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache lock was poisoned.
    pub fn verifier_ntt_cache_bytes(&self) -> Result<usize, AkitaError> {
        self.verifier_ntt.cache_bytes()
    }
}

impl<F: Field + CanonicalEncoding> AkitaVerifierSetup<F> {
    /// Install a prepared scalar Q128 cache whose bytes have trusted provenance.
    ///
    /// The artifact format checks its setup and schedule identities, target
    /// representation, geometry, lengths, and residue ranges. It cannot prove
    /// that the transformed payload was derived from the named setup seed.
    /// Callers must bind the bytes to trusted setup provisioning or to the
    /// verifier program identity before calling this method.
    pub fn install_trusted_prepared_verifier_ntt_cache(
        &self,
        artifact: &[u8],
        schedule_row_digest: crate::ScheduleRowDigest,
    ) -> Result<(), AkitaError> {
        let metadata = crate::prepared_verifier_ntt_cache_metadata(artifact)?;
        let setup_seed_digest = crate::setup_seed_digest(&self.expanded.descriptor.setup_seed)
            .map_err(|error| AkitaError::InvalidSetup(format!("setup seed identity: {error}")))?;
        let expected_binding = crate::PreparedVerifierNttCacheBinding {
            setup_seed_digest,
            schedule_row_digest,
            setup_field_elements: self.expanded.descriptor.num_field_elements,
        };
        crate::dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            metadata.ring_dimension,
            |D| {
                let (decoded_metadata, prepared) =
                    crate::ntt_cache::decode_riscv64_scalar_q128_cache::<F, D>(
                        artifact,
                        expected_binding,
                    )?;
                self.verifier_ntt
                    .install_trusted(decoded_metadata, prepared)
            }
        )
    }

    /// Return an exact or covering negacyclic prefix, preparing it on demand.
    pub fn prepared_verifier_ntt_prefix<const D: usize>(
        &self,
        num_ring_elements: usize,
        tail_num_ring_elements: usize,
        width: usize,
        rhs_abs_bound: u64,
    ) -> Result<Arc<crate::PreparedNttCache<D>>, AkitaError> {
        let key = crate::NttCacheKey {
            ring_d: D,
            num_ring_elements,
            domain: crate::NttTransformDomain::Negacyclic,
        };
        self.verifier_ntt.prepare::<F, D>(
            &self.expanded,
            key,
            tail_num_ring_elements,
            crate::NttCacheMode::ExactNegacyclic {
                width,
                rhs_abs_bound,
            },
        )
    }
}

impl<F: Field> AkitaExpandedSetup<F> {
    /// Build an expanded setup from a trusted matrix the caller has already
    /// derived from `descriptor.setup_seed`.
    ///
    /// This constructor deliberately does not rederive or validate the matrix. Use
    /// [`Self::from_verified_parts`] for untrusted serialized setup bytes.
    #[must_use]
    pub fn from_trusted_seed_derived_parts_unchecked(
        descriptor: AkitaSetupDescriptor,
        shared_matrix: FlatMatrix<F>,
    ) -> Self {
        Self {
            descriptor,
            shared_matrix,
        }
    }

    /// Setup seed and runtime layout metadata.
    #[must_use]
    pub fn descriptor(&self) -> &AkitaSetupDescriptor {
        &self.descriptor
    }

    /// Shared coefficient-form matrix backing all setup roles.
    #[must_use]
    pub fn shared_matrix(&self) -> &FlatMatrix<F> {
        &self.shared_matrix
    }
}

impl<F> AkitaExpandedSetup<F>
where
    F: Field + CanonicalEncoding + Valid,
{
    /// Build an expanded setup from untrusted parts and verify the materialized
    /// matrix against the public seed.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if the seed/matrix shape is malformed or
    /// the matrix was not deterministically derived from the seed.
    pub fn from_verified_parts(
        descriptor: AkitaSetupDescriptor,
        shared_matrix: FlatMatrix<F>,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            descriptor,
            shared_matrix,
        };
        out.check()?;
        Ok(out)
    }
}

/// Fixed public seed for deterministic, reproducible setup.
#[must_use]
pub fn sample_akita_setup_seed() -> AkitaSetupSeed {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    AkitaSetupSeed::blake2b512_paged_v2(seed)
}

/// Derive an exact flat prefix of public field elements from a seed.
///
/// The coefficient stream uses the page size owned by the selected derivation
/// policy.
/// Ring dimensions are not absorbed into the XOF and do not affect this
/// derivation. Consumers reshape the returned field prefix into role-specific
/// ring-matrix views. Equal field-length requests therefore derive identical
/// coefficient prefixes for every schedule.
///
/// Each page owns one Blake2b stream and exact canonical rejection-sampling
/// calls consume that stream sequentially. Pages may be derived in parallel,
/// while concatenation in page-index order preserves deterministic prefix
/// semantics.
#[tracing::instrument(skip_all, name = "derive_public_matrix_prefix")]
pub fn derive_public_matrix_prefix<F: Field + CanonicalEncoding>(
    num_field_elements: usize,
    id: &AkitaSetupSeed,
) -> Result<FlatMatrix<F>, AkitaError> {
    let mut data = vec![F::zero(); num_field_elements];
    cfg_chunks_mut!(data, id.derivation.page_field_elements())
        .enumerate()
        .try_for_each(|(page_index, coeffs)| -> Result<(), AkitaError> {
            let mut page_rng = SetupSeedPageXof::new::<F>(id, page_index)?;
            for coeff in coeffs.iter_mut() {
                *coeff = page_rng.sample::<F>()?;
            }
            Ok(())
        })?;

    Ok(FlatMatrix::from_flat_data(data))
}

/// Check that a materialized public matrix has exactly the shape declared by
/// `descriptor`.
///
/// # Errors
///
/// Returns an error if either side is structurally malformed or if the matrix
/// field-element count differs from the descriptor.
pub fn validate_public_matrix_shape_matches_seed<F: Field + Valid>(
    shared_matrix: &FlatMatrix<F>,
    descriptor: &AkitaSetupDescriptor,
) -> Result<(), SerializationError> {
    descriptor.check()?;
    shared_matrix.check()?;
    if shared_matrix.num_field_elements() != descriptor.num_field_elements {
        return Err(SerializationError::InvalidData(
            "setup shared_matrix field count does not match setup descriptor".to_string(),
        ));
    }
    Ok(())
}

/// Check that a materialized public matrix is exactly the deterministic matrix
/// derived from `seed`.
///
/// # Errors
///
/// Returns an error if the matrix shape is malformed or if any coefficient
/// differs from the seed-derived public matrix.
pub fn validate_public_matrix_matches_seed<F: Field + CanonicalEncoding + Valid>(
    shared_matrix: &FlatMatrix<F>,
    descriptor: &AkitaSetupDescriptor,
) -> Result<(), SerializationError> {
    validate_public_matrix_shape_matches_seed(shared_matrix, descriptor)?;
    let mut expected = vec![F::zero(); descriptor.setup_seed.derivation.page_field_elements()];
    for (page_index, coeffs) in shared_matrix
        .as_field_slice()
        .chunks(descriptor.setup_seed.derivation.page_field_elements())
        .enumerate()
    {
        let mut page_rng = SetupSeedPageXof::new::<F>(&descriptor.setup_seed, page_index)
            .map_err(|e| SerializationError::InvalidData(e.to_string()))?;
        for value in &mut expected[..coeffs.len()] {
            *value = page_rng
                .sample::<F>()
                .map_err(|e| SerializationError::InvalidData(e.to_string()))?;
        }
        if coeffs != &expected[..coeffs.len()] {
            return Err(SerializationError::InvalidData(
                "setup shared_matrix does not match public matrix seed".to_string(),
            ));
        }
    }
    Ok(())
}

struct SetupSeedPageXof {
    stream: Blake2bStream,
}

impl SetupSeedPageXof {
    fn new<F: Field + CanonicalEncoding>(
        id: &AkitaSetupSeed,
        page_index: usize,
    ) -> Result<Self, AkitaError> {
        let mut context = id.seed.to_vec();
        context.extend_from_slice(&crate::field_modulus_be_bytes::<F>()?);
        let page_size = u64::try_from(id.derivation.page_field_elements())
            .map_err(|_| AkitaError::InvalidSetup("page size exceeds u64".into()))?;
        let page_index = u64::try_from(page_index)
            .map_err(|_| AkitaError::InvalidSetup("page index exceeds u64".into()))?;
        context.extend_from_slice(&page_size.to_le_bytes());
        context.extend_from_slice(&page_index.to_le_bytes());
        Ok(Self {
            stream: Blake2bStream::new(PUBLIC_MATRIX_STREAM_DOMAIN, &context)?,
        })
    }

    // Same canonical byte consumption as Jolt Solinas sample_uniform_below:
    // ceil(modulus_bits/8) LE bytes per attempt, high-bit mask, then reject >=p.
    // Epoch 6 applies this canonical mapping to every field, including BN254;
    // BN254 no longer interprets accepted draws as Montgomery residues.
    // No infallible RngCore adapter can suppress a stream error.
    fn sample<F: Field + CanonicalEncoding>(&mut self) -> Result<F, AkitaError> {
        let byte_len = usize::try_from(F::MODULUS_BITS.div_ceil(8))
            .map_err(|_| AkitaError::InvalidSetup("field width exceeds usize".into()))?;
        if byte_len == 0 || byte_len > F::NUM_BYTES {
            return Err(AkitaError::InvalidSetup(
                "unsupported setup field encoding".into(),
            ));
        }
        let mut bytes = vec![0; F::NUM_BYTES];
        loop {
            self.stream.read(&mut bytes[..byte_len])?;
            if F::MODULUS_BITS % 8 != 0 {
                bytes[byte_len - 1] &= (1u8 << (F::MODULUS_BITS % 8)) - 1;
            }
            if let Some(value) = F::from_bytes_le_checked(&bytes) {
                return Ok(value);
            }
        }
    }
}

impl Valid for PublicMatrixDerivation {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl AkitaSerialize for PublicMatrixDerivation {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let tag = match self {
            Self::Blake2b512PagedV2 => 2u8,
        };
        tag.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1u8.serialized_size(compress)
    }
}

impl AkitaDeserialize for PublicMatrixDerivation {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        match u8::deserialize_with_mode(&mut reader, compress, validate, &())? {
            2 => Ok(Self::Blake2b512PagedV2),
            tag => Err(SerializationError::InvalidData(format!(
                "unsupported public matrix derivation tag {tag}"
            ))),
        }
    }
}

impl Valid for AkitaSetupSeed {
    fn check(&self) -> Result<(), SerializationError> {
        self.derivation.check()
    }
}

impl AkitaSerialize for AkitaSetupSeed {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.derivation.serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.seed)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.derivation.serialized_size(compress) + self.seed.len()
    }
}

impl AkitaDeserialize for AkitaSetupSeed {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let derivation =
            PublicMatrixDerivation::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mut seed = [0u8; 32];
        reader.read_exact(&mut seed)?;
        let out = Self { derivation, seed };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for AkitaSetupDescriptor {
    fn check(&self) -> Result<(), SerializationError> {
        if self.max_num_batched_polys == 0 {
            return Err(SerializationError::InvalidData(
                "setup descriptor max_num_batched_polys must be at least 1".to_string(),
            ));
        }
        if self.num_field_elements == 0 {
            return Err(SerializationError::InvalidData(
                "setup descriptor num_field_elements must be non-zero".to_string(),
            ));
        }
        self.setup_seed.check()?;
        Ok(())
    }
}

impl AkitaSerialize for AkitaSetupDescriptor {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.max_num_vars
            .serialize_with_mode(&mut writer, compress)?;
        self.max_num_batched_polys
            .serialize_with_mode(&mut writer, compress)?;
        self.num_field_elements
            .serialize_with_mode(&mut writer, compress)?;
        self.setup_seed.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.max_num_vars.serialized_size(compress)
            + self.max_num_batched_polys.serialized_size(compress)
            + self.num_field_elements.serialized_size(compress)
            + self.setup_seed.serialized_size(compress)
    }
}

impl AkitaDeserialize for AkitaSetupDescriptor {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let max_num_vars = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let max_num_batched_polys =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let num_field_elements =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let setup_seed =
            AkitaSetupSeed::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            max_num_vars,
            max_num_batched_polys,
            num_field_elements,
            setup_seed,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Field + CanonicalEncoding + Valid> Valid for AkitaExpandedSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.descriptor.check()?;
        self.shared_matrix.check()?;
        validate_public_matrix_matches_seed(&self.shared_matrix, &self.descriptor)?;
        Ok(())
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for AkitaExpandedSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.descriptor.serialize_with_mode(&mut writer, compress)?;
        self.shared_matrix
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.descriptor.serialized_size(compress) + self.shared_matrix.serialized_size(compress)
    }
}

impl<F: Field + CanonicalEncoding + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaExpandedSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let descriptor =
            AkitaSetupDescriptor::deserialize_with_mode(&mut reader, compress, validate, &())?;
        descriptor.check()?;
        let shared_matrix = FlatMatrix::deserialize_with_expected_shape(
            &mut reader,
            compress,
            validate,
            descriptor.num_field_elements,
            MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
        )?;
        if matches!(validate, Validate::Yes) {
            Self::from_verified_parts(descriptor, shared_matrix)
        } else {
            Ok(Self::from_trusted_seed_derived_parts_unchecked(
                descriptor,
                shared_matrix,
            ))
        }
    }
}

impl<F: Field + CanonicalEncoding + Valid> Valid for AkitaVerifierSetup<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expanded.check()?;
        if self.prefix_slots.setup_seed() != &self.expanded.descriptor.setup_seed {
            return Err(SerializationError::InvalidData(
                "setup-prefix registry belongs to a different public matrix".to_string(),
            ));
        }
        self.prefix_slots.check()
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for AkitaVerifierSetup<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let mut writer = writer;
        self.expanded.serialize_with_mode(&mut writer, compress)?;
        self.prefix_slots.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expanded.serialized_size(compress) + self.prefix_slots.serialized_size(compress)
    }
}

impl<F: Field + CanonicalEncoding + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for AkitaVerifierSetup<F>
{
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut reader = reader;
        let expanded = Arc::new(AkitaExpandedSetup::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?);
        let prefix_slots =
            SetupPrefixVerifierRegistry::deserialize_with_mode(reader, compress, validate, &())?;
        Self::from_parts(expanded, prefix_slots)
            .map_err(|err| SerializationError::InvalidData(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Zero;
    use jolt_field::{
        Fp32, Fp64, Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59,
    };

    type F = Prime128OffsetA7F7;
    const D: usize = 4;
    type SmallF = Fp64<4294967197>;
    const SMALL_D: usize = 64;

    fn prefix_commitment_params(n_prefix: usize, d_setup: usize) -> crate::GroupOpenPhaseParams {
        let a_bound = *crate::sis::inner_coeff_linf_bounds(
            crate::sis::SisModulusProfileId::Q128OffsetA7F7,
            u32::try_from(d_setup).expect("test ring dimension"),
        )
        .first()
        .expect("exact prefix A bounds");
        let inner_commit_matrix = crate::InnerCommitMatrixParams::try_new_with_min_rank(
            crate::SisTableKey {
                policy: crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: crate::sis::SisTableDigest::CURRENT,
                modulus_profile: crate::sis::SisModulusProfileId::Q128OffsetA7F7,
                role: crate::sis::SisMatrixRole::Inner,
                ring_dimension: u32::try_from(d_setup).expect("test ring dimension"),
                coeff_linf_bound: a_bound,
            },
            1,
        )
        .expect("audited prefix A matrix");
        let outer_commit_matrix = crate::OuterCommitMatrixParams::try_new_with_min_rank(
            crate::SisTableKey {
                policy: crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                table_digest: crate::sis::SisTableDigest::CURRENT,
                modulus_profile: crate::sis::SisModulusProfileId::Q128OffsetA7F7,
                role: crate::sis::SisMatrixRole::Outer,
                ring_dimension: u32::try_from(d_setup).expect("test ring dimension"),
                coeff_linf_bound: 3,
            },
            inner_commit_matrix.output_rank() * (n_prefix / d_setup),
        )
        .expect("audited prefix B matrix");
        crate::GroupOpenPhaseParams {
            setup_natural_len: None,
            profile: crate::GroupCommitPhaseParams {
                version: crate::GroupCommitPhaseParams::VERSION,
                group: crate::PolynomialGroupLayout::singleton(n_prefix.trailing_zeros() as usize),
                blocks: crate::BlockGeometry::new(n_prefix / d_setup, 1, n_prefix / d_setup),
                outer_slice_count: crate::CommitmentSliceCount::ONE,
                inner: crate::RoleParams::new(crate::GadgetDigits::new(1, 1), inner_commit_matrix),
                outer: crate::RoleParams::new(crate::GadgetDigits::new(1, 1), outer_commit_matrix),
            },
            opening: crate::GroupOpeningPlan::evaluation_trace(
                akita_challenges::SparseChallengeConfig::pm1_only(0),
                1,
                1,
                1,
            ),
        }
    }

    fn seed(public_matrix_seed: [u8; 32]) -> AkitaSetupDescriptor {
        AkitaSetupDescriptor {
            max_num_vars: 8,
            max_num_batched_polys: 1,
            num_field_elements: 2 * D,
            setup_seed: public_matrix_seed.into(),
        }
    }

    #[test]
    fn verifier_setup_prefix_slots_roundtrip() {
        use crate::proof::{RingVec, SetupPrefixPublicCommitment, SetupPrefixVerifierSlot};

        let setup_seed = seed([7u8; 32]);
        let shared_matrix =
            derive_public_matrix_prefix::<F>(2 * D, &setup_seed.setup_seed).unwrap();
        let mut prefix_slots = SetupPrefixVerifierRegistry::new(setup_seed.setup_seed.clone());
        let d_setup = 64;
        let commitment_params = prefix_commitment_params(d_setup, d_setup);
        let matrix = &commitment_params.profile.outer.matrix;
        let payload_coefficients = crate::CompressionChainPlan::for_complete_source(
            matrix.sis_modulus_profile(),
            matrix.output_rank() * matrix.ring_dimension(),
        )
        .expect("setup-prefix compression plan")
        .terminal_coefficients();
        let slot = SetupPrefixVerifierSlot {
            id: crate::scheduled_setup_prefix(d_setup - 1, commitment_params)
                .slot_id()
                .expect("setup prefix group"),
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![F::zero(); payload_coefficients])],
            },
        };
        prefix_slots.insert(slot).expect("insert prefix slot");
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    setup_seed,
                    shared_matrix,
                ),
            ),
            prefix_slots,
            verifier_ntt: Arc::new(crate::ntt_cache::VerifierNttCache::default()),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaVerifierSetup::<F>::deserialize_compressed_exact(&bytes, &())
            .expect("deserialize");

        assert_eq!(decoded.prefix_slots.len(), 1);
        assert_eq!(decoded, setup);

        for suffix in [0, 0xa5] {
            let mut suffixed = bytes.clone();
            suffixed.push(suffix);
            assert!(AkitaVerifierSetup::<F>::deserialize_compressed_exact(&suffixed, &()).is_err());
        }

        let mut expanded_bytes = Vec::new();
        setup
            .expanded
            .serialize_compressed(&mut expanded_bytes)
            .expect("serialize expanded setup");
        assert!(
            AkitaExpandedSetup::<F>::deserialize_compressed_exact(&expanded_bytes, &()).is_ok()
        );
        for suffix in [0, 0xa5] {
            let mut suffixed = expanded_bytes.clone();
            suffixed.push(suffix);
            assert!(AkitaExpandedSetup::<F>::deserialize_compressed_exact(&suffixed, &()).is_err());
        }
    }

    #[test]
    fn verifier_setup_rejects_prefix_registry_from_another_public_matrix() {
        let setup_seed = seed([7u8; 32]);
        let shared_matrix =
            derive_public_matrix_prefix::<F>(2 * D, &setup_seed.setup_seed).unwrap();
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                setup_seed,
                shared_matrix,
            ),
        );
        let foreign_registry =
            SetupPrefixVerifierRegistry::new(AkitaSetupSeed::blake2b512_paged_v2([9u8; 32]));

        let err = AkitaVerifierSetup::from_parts(expanded, foreign_registry)
            .expect_err("cross-seed prefix registry must be rejected");
        assert!(err.to_string().contains("different public matrix"));
    }

    #[test]
    fn strict_verifier_setup_decode_rejects_matrix_not_derived_from_seed() {
        let descriptor = seed([7u8; 32]);
        let setup_seed = descriptor.setup_seed.clone();
        let wrong_seed = AkitaSetupSeed::blake2b512_paged_v2([9u8; 32]);
        let wrong_matrix = derive_public_matrix_prefix::<F>(2 * D, &wrong_seed).unwrap();
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor,
                    wrong_matrix,
                ),
            ),
            prefix_slots: SetupPrefixVerifierRegistry::new(setup_seed),
            verifier_ntt: Arc::new(crate::ntt_cache::VerifierNttCache::default()),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("setup shared_matrix does not match public matrix seed"));
    }

    #[test]
    fn strict_verifier_setup_decode_rejects_truncated_seed_prefix_matrix() {
        let descriptor = seed([7u8; 32]);
        let setup_seed = descriptor.setup_seed.clone();
        let short_matrix = derive_public_matrix_prefix::<F>(D, &setup_seed).unwrap();
        let setup = AkitaVerifierSetup {
            expanded: Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor,
                    short_matrix,
                ),
            ),
            prefix_slots: SetupPrefixVerifierRegistry::new(setup_seed),
            verifier_ntt: Arc::new(crate::ntt_cache::VerifierNttCache::default()),
        };

        let mut bytes = Vec::new();
        setup.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaVerifierSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("flat matrix field count does not match expected setup shape"));
    }

    #[test]
    fn strict_setup_decode_rejects_matrix_shape_before_payload() {
        let setup_seed = seed([7u8; 32]);
        let mut bytes = Vec::new();
        setup_seed.serialize_compressed(&mut bytes).unwrap();
        usize::MAX.serialize_compressed(&mut bytes).unwrap();
        let err = AkitaExpandedSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(err
            .to_string()
            .contains("flat matrix field count does not match expected setup shape"));
    }

    #[test]
    fn setup_seed_validity_is_not_the_generic_decode_allocation_cap() {
        let setup_seed = AkitaSetupDescriptor {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            num_field_elements: MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS + 1,
            setup_seed: [7u8; 32].into(),
        };

        setup_seed.check().unwrap();
        assert!(setup_seed.num_field_elements > MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS);
    }

    #[test]
    fn generic_setup_decode_still_rejects_shapes_above_allocation_cap() {
        let setup_seed = AkitaSetupDescriptor {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            num_field_elements: MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS + 1,
            setup_seed: [7u8; 32].into(),
        };
        let mut bytes = Vec::new();
        setup_seed.serialize_compressed(&mut bytes).unwrap();

        let err = AkitaExpandedSetup::<F>::deserialize_compressed(&bytes[..], &()).unwrap_err();

        assert!(matches!(
            err,
            SerializationError::LengthLimitExceeded { max, .. }
                if max == MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS
        ));
    }

    #[test]
    fn flat_derivation_is_deterministic_for_same_seed() {
        let seed = AkitaSetupSeed::blake2b512_paged_v2([42u8; 32]);
        let m1 = derive_public_matrix_prefix::<SmallF>(15 * SMALL_D, &seed).unwrap();
        let m2 = derive_public_matrix_prefix::<SmallF>(15 * SMALL_D, &seed).unwrap();
        assert_eq!(m1, m2);
    }

    #[test]
    fn flat_derivation_is_prefix_stable() {
        let seed = AkitaSetupSeed::blake2b512_paged_v2([7u8; 32]);
        let small = derive_public_matrix_prefix::<SmallF>(6 * SMALL_D, &seed).unwrap();
        let large = derive_public_matrix_prefix::<SmallF>(24 * SMALL_D, &seed).unwrap();
        let small_view = small.ring_view::<SMALL_D>(1, 6).unwrap();
        let large_view = large.ring_view::<SMALL_D>(1, 6).unwrap();
        for c in 0..6 {
            assert_eq!(small_view.row(0).unwrap()[c], large_view.row(0).unwrap()[c]);
        }
    }

    #[test]
    fn flat_derivation_matches_sequential_page_stream() {
        let seed = AkitaSetupSeed::blake2b512_paged_v2([5u8; 32]);
        let got = derive_public_matrix_prefix::<SmallF>(6 * SMALL_D, &seed).unwrap();
        let mut page = SetupSeedPageXof::new::<SmallF>(&seed, 0).unwrap();
        let expected = (0..6 * SMALL_D)
            .map(|_| page.sample::<SmallF>().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(got.as_field_slice(), expected.as_slice());
    }

    #[test]
    fn flat_derivation_is_independent_of_ring_dimension() {
        let seed = AkitaSetupSeed::blake2b512_paged_v2([17u8; 32]);
        let d64 = derive_public_matrix_prefix::<SmallF>(8 * 64, &seed).unwrap();
        let d128 = derive_public_matrix_prefix::<SmallF>(4 * 128, &seed).unwrap();

        assert_eq!(d64.as_field_slice(), d128.as_field_slice());
    }

    #[test]
    fn flat_derivation_is_prefix_stable_across_page_boundary() {
        let seed = AkitaSetupSeed::blake2b512_paged_v2([23u8; 32]);
        let page_field_elements = seed.derivation.page_field_elements();
        let through_page_zero =
            derive_public_matrix_prefix::<SmallF>(page_field_elements, &seed).unwrap();
        let into_page_one =
            derive_public_matrix_prefix::<SmallF>(page_field_elements + 64, &seed).unwrap();

        assert_eq!(
            through_page_zero.as_field_slice(),
            &into_page_one.as_field_slice()[..page_field_elements]
        );
    }

    #[test]
    fn paged_derivation_golden_vector() {
        let seed = AkitaSetupSeed::blake2b512_paged_v2([31u8; 32]);
        fn samples<F: Field + CanonicalEncoding>(seed: &AkitaSetupSeed) -> [u128; 3] {
            let page_field_elements = seed.derivation.page_field_elements();
            let derived = derive_public_matrix_prefix::<F>(page_field_elements + 64, seed).unwrap();
            let canonical = derived
                .as_field_slice()
                .iter()
                .map(|value| {
                    value
                        .to_u128_checked()
                        .expect("Akita field element must fit in u128")
                })
                .collect::<Vec<_>>();
            [
                canonical[0],
                canonical[page_field_elements - 1],
                canonical[page_field_elements],
            ]
        }

        assert_eq!(
            samples::<Prime32Offset99>(&seed),
            [2_747_537_039, 2_619_083_704, 2_013_241_243]
        );
        assert_eq!(
            samples::<Prime64Offset59>(&seed),
            [
                12_648_695_004_588_512_934,
                3_344_598_516_900_523_574,
                897_125_855_502_376_385,
            ]
        );
        assert_eq!(
            samples::<Prime128OffsetA7F7>(&seed),
            [
                145_959_443_756_467_794_520_262_854_373_472_073_634,
                223_713_176_082_637_938_135_576_771_346_588_946_760,
                241_518_123_698_792_648_615_314_846_481_604_222_190,
            ]
        );
        assert_eq!(
            samples::<Prime128Offset275>(&seed),
            [
                247_240_647_133_723_316_793_221_735_485_723_619_175,
                158_173_053_057_120_683_187_633_514_183_428_704_954,
                72_128_563_880_154_482_315_034_442_191_864_010_376,
            ]
        );
    }

    #[test]
    fn page_xof_binds_the_field_modulus() {
        type OtherF = Fp32<4294967291>;

        let seed = AkitaSetupSeed::blake2b512_paged_v2([29u8; 32]);
        let mut small = SetupSeedPageXof::new::<SmallF>(&seed, 0).unwrap();
        let mut other = SetupSeedPageXof::new::<OtherF>(&seed, 0).unwrap();
        let mut small_bytes = [0u8; 32];
        let mut other_bytes = [0u8; 32];
        small.stream.read(&mut small_bytes).unwrap();
        other.stream.read(&mut other_bytes).unwrap();

        assert_ne!(small_bytes, other_bytes);
    }

    #[test]
    fn different_shapes_from_same_flat() {
        let seed = AkitaSetupSeed::blake2b512_paged_v2([13u8; 32]);
        let flat = derive_public_matrix_prefix::<SmallF>(12 * SMALL_D, &seed).unwrap();
        let view_3x4 = flat.ring_view::<SMALL_D>(3, 4).unwrap();
        let view_2x6 = flat.ring_view::<SMALL_D>(2, 6).unwrap();

        assert_eq!(view_3x4.row(0).unwrap()[0], view_2x6.row(0).unwrap()[0]);
        assert_eq!(view_3x4.row(0).unwrap()[3], view_2x6.row(0).unwrap()[3]);
        assert_ne!(view_3x4.row(1).unwrap()[0], view_2x6.row(1).unwrap()[0]);
    }
    #[test]
    fn canonical_sampler_matches_solinas_field_random() {
        use rand_core::RngCore;
        struct TestRng(Blake2bStream);
        impl RngCore for TestRng {
            fn fill_bytes(&mut self, out: &mut [u8]) {
                self.0.read(out).unwrap();
            }
            fn next_u32(&mut self) -> u32 {
                let mut b = [0; 4];
                self.fill_bytes(&mut b);
                u32::from_le_bytes(b)
            }
            fn next_u64(&mut self) -> u64 {
                let mut b = [0; 8];
                self.fill_bytes(&mut b);
                u64::from_le_bytes(b)
            }
            fn try_fill_bytes(&mut self, out: &mut [u8]) -> Result<(), rand_core::Error> {
                self.fill_bytes(out);
                Ok(())
            }
        }
        fn check<F: Field + CanonicalEncoding>() {
            let seed = AkitaSetupSeed::blake2b512_paged_v2([71; 32]);
            let mut page = SetupSeedPageXof::new::<F>(&seed, 0).unwrap();
            let mut native = TestRng(page.stream.clone());
            for _ in 0..257 {
                assert_eq!(page.sample::<F>().unwrap(), F::random(&mut native));
            }
        }
        check::<jolt_field::Prime24Offset3>();
        check::<Prime32Offset99>();
        check::<Prime64Offset59>();
        check::<Prime128Offset275>();
        check::<Prime128OffsetA7F7>();
    }
    #[test]
    fn production_a7f7_sampling_consumption_vector() {
        let seed = AkitaSetupSeed::blake2b512_paged_v2([71; 32]);
        let mut page = SetupSeedPageXof::new::<Prime128OffsetA7F7>(&seed, 0).unwrap();
        // Independent hashlib vector: eight accepted 16-byte draws, then suffix.
        let expected = [
            291_247_137_895_573_119_295_029_205_577_813_119_110,
            68_153_673_745_160_040_577_167_392_256_230_836_239,
            322_881_561_755_709_032_477_739_425_028_150_445_207,
            256_231_412_712_432_632_834_658_260_240_473_882_569,
            22_036_653_535_127_250_014_525_428_159_420_818_066,
            5_224_142_567_341_477_832_753_305_482_197_751_646,
            256_599_641_076_248_507_490_914_711_564_464_170_851,
            14_580_395_030_421_277_202_459_756_261_464_921_725,
        ];
        for value in expected {
            assert_eq!(
                page.sample::<Prime128OffsetA7F7>()
                    .unwrap()
                    .to_u128_checked(),
                Some(value)
            );
        }
        let mut suffix = [0; 16];
        page.stream.read(&mut suffix).unwrap();
        assert_eq!(
            suffix,
            [
                0x65, 0xba, 0x23, 0xce, 0x8a, 0xf3, 0xcd, 0xd2, 0x88, 0xff, 0x84, 0x52, 0x5e, 0xc3,
                0x95, 0xf1
            ]
        );
    }
    #[test]
    fn bn254_canonical_sampling_vector_and_rejected_draw_consumption() {
        use jolt_field::Fr;
        let seed = AkitaSetupSeed::blake2b512_paged_v2([71; 32]);
        let mut page = SetupSeedPageXof::new::<Fr>(&seed, 0).unwrap();
        // Independent hashlib vector: eight accepts consume nine 32-byte draws.
        let expected = [
            "ec97d172dc9d293a507d1c07d96c8adfa2daef9e0c1aa200154a2139a73b0400",
            "83c63e5d497ae61f482d54ebb93243e0ced13578b7f32d1c445608d727e34f08",
            "e8b93770986657dd93bcebf277184cff07b524314b2ab35d01e4069d9754c42b",
            "d7b87b850e5aad8a5842e2f7279c1e9c35c9f415604895164e1bb968f164f405",
            "1ac91ed87787de6529e8c1fcbf4c89fc58a11bc991fc4d420bbba3c7fd6a6d20",
            "95690b2ab3056187ef04e921f992f682f343a54daa28916b5ece95eefbb62b0b",
            "6f4252bd0eab3c123779ba8149e4ae3754b3079b78a3620fa153511bd3363428",
            "7a5dae878d089bca2cf5e7485f19db81cf152989a27c657a024965196606540d",
        ];
        let decode = |hex: &str| {
            hex.as_bytes()
                .chunks_exact(2)
                .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect::<Vec<_>>()
        };
        for hex in expected {
            let canonical = Fr::from_bytes_le_checked(&decode(hex)).unwrap();
            assert_eq!(page.sample::<Fr>().unwrap(), canonical);
        }
        let mut suffix = [0; 32];
        page.stream.read(&mut suffix).unwrap();
        assert_eq!(
            suffix.as_slice(),
            decode("eeaf15d72efdd3335d1bf8791a56549aeae65173ebb8fdb6d76f20db1a00ae5c")
        );
    }
    #[test]
    fn old_setup_derivation_tag_is_rejected() {
        assert!(PublicMatrixDerivation::deserialize_compressed(&[1u8][..], &()).is_err());
        assert_eq!(
            PublicMatrixDerivation::deserialize_compressed(&[2u8][..], &()).unwrap(),
            PublicMatrixDerivation::Blake2b512PagedV2
        );
    }
}
