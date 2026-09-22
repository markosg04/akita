# Native Blake-only protocol epoch 6

This packet migrates native Akita from base
`252abb895046cc1d5b9955a26a2ad2318148ac26`. It does not implement a verifier
circuit or establish the security of that circuit.

## Canonical ownership

`akita-transcript::blake2b_stream::Blake2bStream` owns the unkeyed Blake2b512
stream: block j hashes `u32le(domain.len) || domain || u64le(context.len) ||
context || u64le(j)`. Counter zero is first. Each block has 64 bytes; all
2^64 blocks are available. Reads retain partial-block suffixes. Empty reads do
nothing. A read exceeding 2^70 total bytes fails atomically with
`AkitaError::RandomStreamExhausted`; it never wraps, fabricates bytes, or panics.

Public domain constants are owned next to their context encoders:

| Owner | Domain | Context |
|---|---|---|
| akita-challenges sampler | `akita/sparse-challenge/blake2b512/v1` | root[32], coordinate u64le |
| akita-types proof/setup | `akita/public-matrix/blake2b512-paged/v2` | seed[32], modulus[32] big endian, page size u64le, page index u64le |
| akita-prover protocol/prg | `akita/matrix-entry/blake2b512/v1` | seed[32], label length u64le, label, rows/cols/row/col u64le |

The first two have live production callers. The third preserves the existing
external matrix-entry facility with a fallible stream API; it has no current
internal production caller. Matrix geometry is checked before framing. The
old infallible matrix RNG/backend traits are removed.

`akita-transcript::PROTOCOL_TAG` is the padded 64-byte
`akita-pcs/transcript/v2/blake2b`. `SESSION_DOMAIN_TAG` is now exported and
unchanged. The existing spongefish Blake2b duplex state machine and session
length framing are unchanged. Constrained callers must import these constants.
No Keccak alias remains active; selecting its old feature fails compilation.

## Sampling semantics

Sparse Fisher–Yates masking, accepted/rejected draw consumption, signs, norm
predicate and tables are unchanged. Every rejected integer draw consumes its
bytes. The integer rejection loop has no deterministic finite attempt bound;
the existing 4096 norm-candidate cap is a different outer bound. No rejected
draw is skipped and no new sampling cap is added. Stream exhaustion propagates
through sparse sampling and setup expansion as a typed error.

**Epoch 6 defines canonical field sampling for every field.** Each attempt
reads ceil(MODULUS_BITS/8) bytes, masks unused high bits, then calls the field's
canonical little-endian decoder. Invalid candidates are rejected. This is
uniform over canonical prime-field residues. The generic Field +
CanonicalEncoding API remains available. All supported Solinas variants retain
the exact existing Field::random byte-to-field mapping. BN254 changes from
native Montgomery-residue sampling to canonical decoding; its deterministic
mapping and potentially rejection consumption differ from Field::random.
This difference is intentional, versioned, and tested using independent Python
vectors. The production path no longer uses an infallible RngCore adapter.

## Identity and migration audit

- Instance descriptor version is 6. Checked decoding (`Validate::Yes`) and
  `Valid::check` reject every other version; unchecked decoding does not.
  Production verification constructs a fresh v6 descriptor from active
  configuration, setup and call before transcript replay. Callers using
  unchecked parsing remain responsible for validation.
- Public setup derivation wire tag is 2; the decoder rejects the retired tag 1.
  The default seed conversion selects the new derivation. Public expansion and
  validated deserialization both use the same new fallible implementation.
- Matrix backend ID is 2; IDs 0 and 1 are rejected.
- Setup seed digest includes serialized derivation tag. Prepared-key/cache
  bindings include that seed digest. Expanded setup validation recomputes the
  new matrix, so old actual setup coefficients are not accepted as new ones.
- Schedule artifacts' protocol_epoch changes from 5 to 6. Their schedule rows,
  geometry and security parameters are unchanged: row digests describe those
  inputs, not public-matrix coefficient bytes or stream implementation. The
  schedule loader checks the epoch separately. No old commitment, expanded
  setup, proof, or persisted key has been relabeled as new data.

Runtime sparse/setup/matrix-entry SHAKE implementations and dependencies were
removed. Remaining SHA3 use is an artifact test and an offline SIS estimator
example, outside the proof runtime; akita-types SHA3 is dev-only. The transitive
spongefish implementation has other backends but Akita names Blake2b512
explicitly. Old Keccak fixture branches are unreachable because its feature is
rejected. No runtime Poseidon was found.

## Reproduction and evidence

`python3 scripts/blake_only_vectors.py` emits independent Python hashlib
stream blocks, setup boundary samples (including the production A7F7 modulus
2^128 - 2^32 + 22537 at indices 0/4095/4096), A7F7 candidate consumption and
following suffix, BN254 canonical samples and consumption,
and v2 duplex challenges (including the label-schedule fixture). Rust tests pin those values. The initial label-schedule Python oracle omitted
public-message framing and failed. It was corrected from the production
`FramedBytes::encode` rule (`u64le(message.len) || message`) for each individual
C, O, RS and SC1 message; no native output was copied into the Python oracle.
The initial mismatch log is preserved in the handoff evidence.

`specs/blake-only-fold-vectors.json` records full sparse draw goldens from the
migrated production implementation, not an independent oracle. Their exact
inputs and assertions live in `fold_draw::tests::indexed_fold_challenge_golden_vectors`.
Reproduce with `cargo nextest run -p akita-challenges --lib indexed_fold_challenge_golden_vectors`.
They preserve end-to-end regression signals for d64 evaluation/packing and
d64/d128 norm-rejected draws.
The historical epoch-5 documents are retained as explicitly marked history.

Validation commands and exact outcomes are recorded in the implementation
handoff. Native proof roundtrip used the existing fp32_onehot integration test
(nv14 and nv16); no new Jolt proof was generated. Full workspace release feature
matrices, regenerated Jolt fixtures, recursion, and circuit constraints remain
integration obligations. No performance improvement is claimed.
