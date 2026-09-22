//! Prepared binary catalogs retain the JSON catalog's semantic audit and identity.

use super::{
    catalog_digest, policy_digest, AkitaError, CatalogCoverage, PlannerPolicy,
    ScheduleCatalogArtifactRowV1, SparseChallengeConfig, ValidatedScheduleCatalog,
    AKITA_INSTANCE_DESCRIPTOR_VERSION, MAX_TRUSTED_CATALOG_ROWS,
    MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
};
use akita_types::{OpeningScheduleSelection, ScheduleRowDigest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const BINARY_MAGIC: [u8; 8] = *b"AKSCBIN1";
const VERIFIER_MAGIC: [u8; 8] = *b"AKSCVFY1";

#[derive(Serialize, Deserialize)]
struct BinaryVerifierCatalog {
    magic: [u8; 8],
    protocol_epoch: u32,
    policy_digest: [u8; 32],
    family_name: String,
    row_digests: Vec<[u8; 32]>,
    rows: Vec<ScheduleCatalogArtifactRowV1>,
}

#[derive(Serialize, Deserialize)]
struct BinaryCatalog {
    magic: [u8; 8],
    protocol_epoch: u32,
    policy_digest: [u8; 32],
    family_name: String,
    rows: Vec<ScheduleCatalogArtifactRowV1>,
}

impl ValidatedScheduleCatalog {
    /// Prepare a validated catalog for guest loading without JSON parsing.
    /// The binary representation is setup data, never a proof field.
    pub fn to_artifact_binary(&self) -> Result<Vec<u8>, AkitaError> {
        if matches!(self.coverage, CatalogCoverage::Selected { .. }) {
            let selections = self.rows().map(|row| row.selection()).collect::<Vec<_>>();
            return self.to_verifier_artifact_binary(&selections);
        }
        let artifact = BinaryCatalog {
            magic: BINARY_MAGIC,
            protocol_epoch: AKITA_INSTANCE_DESCRIPTOR_VERSION,
            policy_digest: self.policy_digest,
            family_name: self.family_name.clone(),
            rows: self
                .rows_by_digest
                .iter()
                .map(|row| ScheduleCatalogArtifactRowV1 {
                    schedule: row.schedule().clone(),
                })
                .collect(),
        };
        let bytes = bincode::serde::encode_to_vec(&artifact, bincode::config::standard()).map_err(
            |error| AkitaError::InvalidSetup(format!("cannot encode binary catalog: {error}")),
        )?;
        if bytes.len() > MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES {
            return Err(AkitaError::InvalidSetup(
                "binary catalog exceeds byte limit".to_string(),
            ));
        }
        Ok(bytes)
    }

    /// Load a prepared catalog supplied by the application's trusted setup path.
    /// Checks format, configuration binding, and every available row's semantic
    /// invariants. Selected-row views additionally check commitment membership;
    /// omitted identities do not resolve to parameters. The caller owns provenance.
    pub fn from_trusted_artifact_binary(
        bytes: &[u8],
        expected_family_name: &str,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ) -> Result<Self, AkitaError> {
        if bytes.is_empty() || bytes.len() > MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES {
            return Err(AkitaError::InvalidSetup(
                "binary catalog is outside byte limits".to_string(),
            ));
        }
        if bytes.starts_with(&VERIFIER_MAGIC) {
            return Self::from_verifier_artifact_binary(
                bytes,
                expected_family_name,
                policy,
                ring_challenge_config,
            );
        }
        let (artifact, consumed): (BinaryCatalog, usize) = bincode::serde::decode_from_slice(
            bytes,
            bincode::config::standard().with_limit::<MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES>(),
        )
        .map_err(|error| AkitaError::InvalidSetup(format!("invalid binary catalog: {error}")))?;
        if consumed != bytes.len()
            || artifact.magic != BINARY_MAGIC
            || artifact.protocol_epoch != AKITA_INSTANCE_DESCRIPTOR_VERSION
        {
            return Err(AkitaError::InvalidSetup(
                "unsupported binary catalog format or trailing bytes".to_string(),
            ));
        }
        if artifact.family_name != expected_family_name
            || artifact.policy_digest != policy_digest(policy)
        {
            return Err(AkitaError::InvalidSetup(
                "binary catalog does not match runtime config".to_string(),
            ));
        }
        if artifact.rows.is_empty() || artifact.rows.len() > MAX_TRUSTED_CATALOG_ROWS {
            return Err(AkitaError::InvalidSetup(
                "binary catalog is outside row limits".to_string(),
            ));
        }
        let rows = artifact
            .rows
            .into_iter()
            .map(ScheduleCatalogArtifactRowV1::into_profile_and_schedule)
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_new(artifact.family_name, rows, policy, ring_challenge_config)
    }
}

impl ValidatedScheduleCatalog {
    /// Retain only the requested rows while preserving the complete catalog's
    /// transcript identity. Omitted rows remain opaque digest commitments.
    /// The result serves verification with an existing prepared key; it cannot
    /// size a new setup or be exported as a complete JSON catalog.
    pub fn to_verifier_artifact_binary(
        &self,
        selections: &[OpeningScheduleSelection],
    ) -> Result<Vec<u8>, AkitaError> {
        if selections.is_empty() || selections.len() > MAX_TRUSTED_CATALOG_ROWS {
            return Err(AkitaError::InvalidSetup(
                "verifier catalog selection is outside row limits".into(),
            ));
        }
        let selected = selections
            .iter()
            .map(|selection| selection.row_digest)
            .collect::<BTreeSet<_>>();
        let rows = selected
            .into_iter()
            .map(|row_digest| {
                self.resolve_selection(OpeningScheduleSelection { row_digest })
                    .map(|row| ScheduleCatalogArtifactRowV1 {
                        schedule: row.schedule().clone(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let row_digests = match &self.coverage {
            CatalogCoverage::Complete => self
                .rows()
                .map(|row| *row.selection().row_digest.as_bytes())
                .collect(),
            CatalogCoverage::Selected { row_digests } => row_digests.clone(),
        };
        let artifact = BinaryVerifierCatalog {
            magic: VERIFIER_MAGIC,
            protocol_epoch: AKITA_INSTANCE_DESCRIPTOR_VERSION,
            policy_digest: self.policy_digest,
            family_name: self.family_name.clone(),
            row_digests,
            rows,
        };
        let bytes = bincode::serde::encode_to_vec(&artifact, bincode::config::standard()).map_err(
            |error| AkitaError::InvalidSetup(format!("cannot encode verifier catalog: {error}")),
        )?;
        if bytes.len() > MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES {
            return Err(AkitaError::InvalidSetup(
                "verifier catalog exceeds byte limit".into(),
            ));
        }
        Ok(bytes)
    }

    fn from_verifier_artifact_binary(
        bytes: &[u8],
        expected_family_name: &str,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ) -> Result<Self, AkitaError> {
        let (artifact, consumed): (BinaryVerifierCatalog, usize) =
            bincode::serde::decode_from_slice(
                bytes,
                bincode::config::standard().with_limit::<MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES>(),
            )
            .map_err(|error| {
                AkitaError::InvalidSetup(format!("invalid verifier catalog: {error}"))
            })?;
        if consumed != bytes.len()
            || artifact.magic != VERIFIER_MAGIC
            || artifact.protocol_epoch != AKITA_INSTANCE_DESCRIPTOR_VERSION
        {
            return Err(AkitaError::InvalidSetup(
                "unsupported verifier catalog format or trailing bytes".into(),
            ));
        }
        if artifact.family_name != expected_family_name
            || artifact.policy_digest != policy_digest(policy)
        {
            return Err(AkitaError::InvalidSetup(
                "verifier catalog does not match runtime config".into(),
            ));
        }
        if artifact.row_digests.is_empty()
            || artifact.row_digests.len() > MAX_TRUSTED_CATALOG_ROWS
            || artifact.rows.is_empty()
            || artifact.rows.len() > artifact.row_digests.len()
            || artifact
                .row_digests
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(AkitaError::InvalidSetup(
                "verifier catalog identities are not a bounded canonical list".into(),
            ));
        }
        let rows = artifact
            .rows
            .into_iter()
            .map(ScheduleCatalogArtifactRowV1::into_profile_and_schedule)
            .collect::<Result<Vec<_>, _>>()?;
        // The same semantic audit owns both formats. No supplied identity can
        // replace the digest computed from an available row's validated contents.
        let mut catalog = Self::try_new(artifact.family_name, rows, policy, ring_challenge_config)?;
        for row in catalog.rows() {
            if artifact
                .row_digests
                .binary_search(row.selection().row_digest.as_bytes())
                .is_err()
            {
                return Err(AkitaError::InvalidSetup(
                    "selected row is absent from the catalog commitment".into(),
                ));
            }
        }
        catalog.catalog_digest = catalog_digest(
            &catalog.family_name,
            catalog.policy_digest,
            artifact
                .row_digests
                .iter()
                .copied()
                .map(ScheduleRowDigest::from_bytes),
        );
        catalog.coverage = CatalogCoverage::Selected {
            row_digests: artifact.row_digests,
        };
        Ok(catalog)
    }
}
