//! Packed row-major one-hot source used by trace-oriented commitment kernels.

mod commit;
mod poly;

pub use poly::{
    PackedOneHotPoly, PackedOneHotStreamWriter, PackedOneHotView, StreamingPackedOneHotPoly,
    StreamingPackedOneHotView, PACKED_ONEHOT_BUFFER_ALIGNMENT,
};

#[cfg(test)]
mod tests;
