//! Packed row-major one-hot source used by trace-oriented commitment kernels.

mod commit;
mod poly;

pub use poly::{PackedOneHotPoly, PackedOneHotView, PACKED_ONEHOT_BUFFER_ALIGNMENT};

#[cfg(test)]
mod tests;
