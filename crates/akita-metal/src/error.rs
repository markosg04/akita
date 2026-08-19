use akita_field::AkitaError;

/// Errors raised before or during Metal commitment execution.
#[derive(Debug, thiserror::Error)]
pub enum MetalCommitError {
    /// The crate was invoked on a target without Apple Metal.
    #[error("the Akita Metal backend requires macOS")]
    UnsupportedPlatform,
    /// No system Metal device is available.
    #[error("no Metal device is available")]
    DeviceUnavailable,
    /// Runtime MSL compilation failed.
    #[error("failed to compile the Metal commitment library: {0}")]
    LibraryCompilation(String),
    /// A named MSL function was not found.
    #[error("Metal entry point {name} was not found: {message}")]
    FunctionLookup {
        /// Static function name.
        name: &'static str,
        /// Metal runtime diagnostic.
        message: String,
    },
    /// Compute pipeline creation failed.
    #[error("failed to compile Metal entry point {name}: {message}")]
    PipelineCompilation {
        /// Static function name.
        name: &'static str,
        /// Metal runtime diagnostic.
        message: String,
    },
    /// A requested allocation exceeds a device or host size bound.
    #[error("Metal buffer length {requested} exceeds limit {maximum}")]
    BufferTooLong {
        /// Requested byte length.
        requested: u64,
        /// Maximum byte length.
        maximum: u64,
    },
    /// Shape arithmetic overflowed.
    #[error("Metal commitment shape overflow: {0}")]
    ShapeOverflow(&'static str),
    /// The operation is outside the first backend's support envelope.
    #[error("unsupported Metal commitment shape: {0}")]
    UnsupportedShape(String),
    /// A backend mutex was poisoned.
    #[error("Metal commitment state lock was poisoned")]
    PoisonedLock,
    /// The GPU command did not complete successfully.
    #[error("Metal command failed with status {0:?}")]
    CommandFailed(metal_status::CommandStatus),
    /// A returned field element was not canonical.
    #[error("Metal output coefficient {index} is not canonical")]
    NonCanonicalOutput {
        /// Coefficient index in the flat output.
        index: usize,
    },
}

impl MetalCommitError {
    pub(crate) fn into_akita(self) -> AkitaError {
        AkitaError::InvalidInput(format!("Metal commit backend: {self}"))
    }
}

pub(crate) mod metal_status {
    /// Stable command status copied from the platform value for diagnostics.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum CommandStatus {
        /// Command buffer was never enqueued.
        NotEnqueued,
        /// Command buffer was enqueued.
        Enqueued,
        /// Command buffer was committed.
        Committed,
        /// Command buffer was scheduled.
        Scheduled,
        /// Command buffer completed.
        Completed,
        /// Command buffer failed.
        Error,
    }
}
