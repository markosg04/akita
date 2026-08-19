use akita_config::{policy_of, CommitmentConfig};
use akita_types::{
    ChunkedWitnessCfg, CommitmentPayloadMode, CommitmentSliceCount, CommitmentSliceGeometry,
    CommittedGroupParams, CommittedSourceEncoding, InnerCommitMatrixParams,
    OpenCommitMatrixParams, OpeningMethod, OuterCommitMatrixParams, SisMatrixRole,
    SisModulusProfileId, SisTableKey,
};

use super::{Cfg, Workload, COLUMN_CAPACITY, INNER_RANK, ONEHOT_K, POSITIONS_PER_BLOCK, RING_D};

pub(super) fn workload_num_vars(workload: Workload) -> usize {
    workload.log_t + ONEHOT_K.trailing_zeros() as usize + COLUMN_CAPACITY.trailing_zeros() as usize
}

pub(super) fn full_commit_params(workload: Workload) -> CommittedGroupParams {
    let num_vars = workload_num_vars(workload);
    let num_live_ring_elements = (1usize << num_vars) / RING_D;
    let num_live_blocks = num_live_ring_elements / POSITIONS_PER_BLOCK;
    let mut decomposition = Cfg::decomposition();
    decomposition.log_basis = 3;
    let outer_digits = akita_types::sis::num_digits_open(decomposition);
    let policy = policy_of::<Cfg>();
    let profile = SisModulusProfileId::Q128OffsetA7F7;
    let challenge = Cfg::ring_challenge_config(RING_D).unwrap();
    let a_bucket = akita_types::sis::rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        profile,
        RING_D,
        3,
        &challenge,
        3,
    )
    .unwrap();
    let inner_commit_matrix = InnerCommitMatrixParams::try_new(
        policy.sis_security_policy,
        policy.sis_table_digest,
        profile,
        INNER_RANK,
        POSITIONS_PER_BLOCK,
        a_bucket,
        RING_D,
    )
    .unwrap();
    let outer_width = CommitmentSliceGeometry::try_new(
        CommitmentSliceCount::EIGHT,
        num_live_blocks,
        1,
        INNER_RANK,
        outer_digits,
        RING_D,
        64,
    )
    .unwrap()
    .physical_input_width();
    let outer_bucket = akita_types::sis::rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        profile,
        SisMatrixRole::Outer,
        64,
        3,
    )
    .unwrap();
    let outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        SisTableKey {
            policy: policy.sis_security_policy,
            table_digest: policy.sis_table_digest,
            modulus_profile: profile,
            role: SisMatrixRole::Outer,
            ring_dimension: 64,
            coeff_linf_bound: outer_bucket,
        },
        outer_width,
    )
    .unwrap();
    let opening_method = OpeningMethod::EvaluationTrace;
    let open_width = akita_types::opening_d_segment_width(
        opening_method,
        1,
        RING_D,
        128,
        outer_digits,
        num_live_blocks,
        1,
    )
    .unwrap();
    let open_bucket = akita_types::sis::rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        profile,
        SisMatrixRole::Open,
        128,
        3,
    )
    .unwrap();
    let open_commit_matrix = OpenCommitMatrixParams::try_new_with_min_rank(
        SisTableKey {
            policy: policy.sis_security_policy,
            table_digest: policy.sis_table_digest,
            modulus_profile: profile,
            role: SisMatrixRole::Open,
            ring_dimension: 128,
            coeff_linf_bound: open_bucket,
        },
        open_width,
    )
    .unwrap();
    let params = CommittedGroupParams {
        payload_mode: CommitmentPayloadMode::Compressed,
        source_encoding: CommittedSourceEncoding::CanonicalCoefficientTable,
        opening_method,
        log_basis_inner: 3,
        log_basis_outer: 3,
        log_basis_open: 3,
        inner_commit_matrix,
        outer_commit_matrix,
        open_commit_matrix,
        num_live_ring_elements_per_claim: num_live_ring_elements,
        num_positions_per_block: POSITIONS_PER_BLOCK,
        num_live_blocks,
        outer_slice_count: CommitmentSliceCount::EIGHT,
        fold_challenge_config: challenge,
        num_digits_inner: 1,
        num_digits_outer: outer_digits,
        num_digits_open: outer_digits,
        num_digits_fold: 3,
        witness_chunk: ChunkedWitnessCfg::default(),
        precommitted_groups: Vec::new(),
        setup_prefix: None,
    };
    let blocks_per_column = (1usize << workload.log_t) * ONEHOT_K / RING_D / POSITIONS_PER_BLOCK;
    assert_eq!(params.num_live_blocks, COLUMN_CAPACITY * blocks_per_column);
    assert_eq!(params.inner_commit_matrix.output_rank(), INNER_RANK);
    assert_eq!(params.inner_commit_matrix.ring_dimension(), RING_D);
    assert_eq!(params.inner_commit_matrix.coeff_linf_bound(), Some(65_535));
    assert_eq!(params.outer_commit_matrix.output_rank(), 1);
    assert_eq!(params.outer_commit_matrix.ring_dimension(), 64);
    assert_eq!(params.open_commit_matrix.output_rank(), 1);
    assert_eq!(params.open_commit_matrix.ring_dimension(), 128);
    params
}
