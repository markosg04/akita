//! Optional Apple Metal backend for Akita prover compute operations.
//!
//! The first accelerated operation is the fp128 one-hot inner commitment.
//! Commitment operations outside that kernel remain on Akita's CPU backend.

mod error;

pub use error::MetalCommitError;

/// Policy for selecting Metal at an operation boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalExecutionPolicy {
    /// Reject unavailable or unsupported Metal execution.
    RequireMetal,
    /// Select CPU before execution when Metal is unavailable or not profitable.
    PreferMetal,
}

#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
mod field;
#[cfg(target_os = "macos")]
mod onehot;
#[cfg(target_os = "macos")]
mod packed_onehot;
#[cfg(target_os = "macos")]
mod packed_onehot_fp128_d512;
#[cfg(target_os = "macos")]
mod prepared;
#[cfg(target_os = "macos")]
mod runtime;

#[cfg(target_os = "macos")]
pub use backend::{MetalCommitBackend, MetalCommitMetrics};
#[cfg(target_os = "macos")]
pub use prepared::MetalPreparedSetup;
#[cfg(target_os = "macos")]
pub use runtime::{MetalDeviceCapabilities, MetalOneHotKernel};

#[cfg(not(target_os = "macos"))]
mod unsupported {
    use std::marker::PhantomData;

    use akita_field::Prime128OffsetA7F7;
    use akita_prover::CpuBackend;

    use super::{MetalCommitError, MetalExecutionPolicy};

    /// Unavailable Metal backend placeholder on non-macOS targets.
    #[derive(Clone, Copy, Debug)]
    pub struct MetalCommitBackend<Field = Prime128OffsetA7F7> {
        marker: PhantomData<fn() -> Field>,
    }

    impl<Field> MetalCommitBackend<Field> {
        /// Report that this target has no Metal runtime.
        pub const fn is_available() -> bool {
            false
        }

        /// Constructing the backend on a non-macOS target always fails.
        pub fn new(_policy: MetalExecutionPolicy) -> Result<Self, MetalCommitError> {
            Err(MetalCommitError::UnsupportedPlatform)
        }

        /// Constructing the backend on a non-macOS target always fails.
        pub fn new_with_cpu_backend(
            _policy: MetalExecutionPolicy,
            _cpu: CpuBackend,
        ) -> Result<Self, MetalCommitError> {
            Err(MetalCommitError::UnsupportedPlatform)
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use unsupported::MetalCommitBackend;
