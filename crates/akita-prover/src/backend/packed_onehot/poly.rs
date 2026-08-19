use std::alloc::{alloc, alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::{copy_nonoverlapping, NonNull};
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use akita_field::{parallel::*, AkitaError, FieldCore};

use crate::compute::{RootCommitSource, RootPolyMeta, RootPolyShape};

/// Alignment used by packed owners that can back a no-copy device buffer.
pub const PACKED_ONEHOT_BUFFER_ALIGNMENT: usize = 16 * 1024;
const PACKED_ONEHOT_META_RING_D: usize = 512;

struct AlignedBytes {
    ptr: NonNull<u8>,
    len: usize,
}

impl AlignedBytes {
    fn copy_from(bytes: &[u8]) -> Self {
        debug_assert!(!bytes.is_empty());
        // SAFETY: the alignment is a nonzero power of two, and a Rust slice
        // cannot exceed the maximum allocation size accepted by `Layout`.
        let layout = unsafe {
            Layout::from_size_align_unchecked(bytes.len(), PACKED_ONEHOT_BUFFER_ALIGNMENT)
        };
        // SAFETY: `layout` has nonzero size and valid alignment.
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));
        // SAFETY: both regions hold `bytes.len()` bytes and do not overlap.
        unsafe { copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len()) };
        Self {
            ptr,
            len: bytes.len(),
        }
    }

    fn zeroed(len: usize) -> Result<Self, AkitaError> {
        debug_assert_ne!(len, 0);
        let layout =
            Layout::from_size_align(len, PACKED_ONEHOT_BUFFER_ALIGNMENT).map_err(|_| {
                AkitaError::InvalidInput("packed one-hot allocation layout is too large".into())
            })?;
        // SAFETY: `layout` has nonzero size and valid alignment.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));
        Ok(Self { ptr, len })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` exclusively owns `len` initialized bytes.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn layout(&self) -> Layout {
        // SAFETY: this is the validated layout used for this allocation.
        unsafe { Layout::from_size_align_unchecked(self.len, PACKED_ONEHOT_BUFFER_ALIGNMENT) }
    }
}

impl fmt::Debug for AlignedBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlignedBytes")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl Deref for AlignedBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: `ptr` owns `len` initialized bytes until `drop`.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

// SAFETY: the allocation is owned; mutable access is exclusive to construction.
unsafe impl Send for AlignedBytes {}
// SAFETY: the allocation is exposed only through immutable slices after construction.
unsafe impl Sync for AlignedBytes {}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated with this layout and has not been freed.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout()) };
    }
}

/// A packed row-major trace whose nonzero bytes select one-hot lanes.
///
/// Storage contains `num_rows * num_columns` bytes in row-major order. Byte
/// zero denotes an all-zero chunk; values in `1..onehot_k` select the unit
/// coefficient within that chunk. Logical chunks are ordered column-major and
/// extend through `column_capacity`, with omitted suffix columns equal to zero.
#[derive(Debug, Clone)]
pub struct PackedOneHotPoly<F: FieldCore> {
    lanes: Arc<AlignedBytes>,
    num_rows: usize,
    num_columns: usize,
    column_capacity: usize,
    onehot_k: usize,
    hot_entries: usize,
    num_vars: usize,
    marker: PhantomData<F>,
}

/// Borrowed packed-row source for a commitment kernel at ring dimension `D`.
#[derive(Debug, Clone, Copy)]
pub struct PackedOneHotView<'a, F: FieldCore, const D: usize> {
    lanes: &'a [u8],
    num_rows: usize,
    num_columns: usize,
    column_capacity: usize,
    onehot_k: usize,
    hot_entries: usize,
    marker: PhantomData<F>,
}

fn validate_geometry_shape(
    onehot_k: usize,
    column_capacity: usize,
    num_columns: usize,
    lane_count: usize,
) -> Result<(usize, usize), AkitaError> {
    if onehot_k == 0 || onehot_k > usize::from(u8::MAX) + 1 || !onehot_k.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "packed one-hot K={onehot_k} must be a power of two at most 256"
        )));
    }
    if column_capacity == 0 || !column_capacity.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "packed one-hot column capacity {column_capacity} must be a nonzero power of two"
        )));
    }
    if num_columns == 0 || num_columns > column_capacity {
        return Err(AkitaError::InvalidInput(format!(
            "packed one-hot live column count {num_columns} must be in 1..={column_capacity}"
        )));
    }
    if !lane_count.is_multiple_of(num_columns) {
        return Err(AkitaError::InvalidInput(format!(
            "packed one-hot lane count {lane_count} is not divisible by {num_columns} live columns"
        )));
    }
    let num_rows = lane_count / num_columns;
    if num_rows == 0 || !num_rows.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "packed one-hot row count {num_rows} must be a nonzero power of two"
        )));
    }
    let total_field_elems = num_rows
        .checked_mul(column_capacity)
        .and_then(|chunks| chunks.checked_mul(onehot_k))
        .ok_or_else(|| AkitaError::InvalidInput("packed one-hot logical size overflow".into()))?;
    if !total_field_elems.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "packed one-hot logical field length {total_field_elems} is not a power of two"
        )));
    }
    Ok((num_rows, total_field_elems))
}

fn validate_lanes(onehot_k: usize, lanes: &[u8]) -> Result<usize, AkitaError> {
    let mut hot_entries = 0usize;
    for (position, &lane) in lanes.iter().enumerate() {
        if usize::from(lane) >= onehot_k {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot lane {lane} at byte {position} is outside K={onehot_k}"
            )));
        }
        hot_entries += usize::from(lane != 0);
    }
    Ok(hot_entries)
}

fn invalid_lane_error(onehot_k: usize, lanes: &[u8], position: usize) -> AkitaError {
    AkitaError::InvalidInput(format!(
        "packed one-hot lane {} at byte {position} is outside K={onehot_k}",
        lanes[position]
    ))
}

fn validate_geometry(
    onehot_k: usize,
    column_capacity: usize,
    num_columns: usize,
    lanes: &[u8],
) -> Result<(usize, usize, usize), AkitaError> {
    let (num_rows, total_field_elems) =
        validate_geometry_shape(onehot_k, column_capacity, num_columns, lanes.len())?;
    let hot_entries = validate_lanes(onehot_k, lanes)?;
    Ok((num_rows, total_field_elems, hot_entries))
}

fn validate_ring_dimension(
    onehot_k: usize,
    num_vars: usize,
    ring_d: usize,
) -> Result<(), AkitaError> {
    if ring_d == 0 || !ring_d.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "packed one-hot D={ring_d} must be a nonzero power of two"
        )));
    }
    if !(onehot_k.is_multiple_of(ring_d) || ring_d.is_multiple_of(onehot_k)) {
        return Err(AkitaError::InvalidInput(format!(
            "packed one-hot K={onehot_k} and D={ring_d} must divide one another"
        )));
    }
    let total_field_elems = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "packed one-hot logical size 2^{num_vars} does not fit usize"
        ))
    })?;
    if !total_field_elems.is_multiple_of(ring_d) {
        return Err(AkitaError::InvalidInput(format!(
            "packed one-hot logical field length {total_field_elems} is not divisible by D={ring_d}"
        )));
    }
    Ok(())
}

impl<F: FieldCore> PackedOneHotPoly<F> {
    /// Construct an aligned packed trace source.
    pub fn new(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: Vec<u8>,
    ) -> Result<Self, AkitaError> {
        let (num_rows, total_field_elems, hot_entries) =
            validate_geometry(onehot_k, column_capacity, num_columns, &lanes)?;
        Ok(Self {
            lanes: Arc::new(AlignedBytes::copy_from(&lanes)),
            num_rows,
            num_columns,
            column_capacity,
            onehot_k,
            hot_entries,
            num_vars: total_field_elems.trailing_zeros() as usize,
            marker: PhantomData,
        })
    }

    /// Construct directly in aligned storage from a deterministic row-major lane function.
    pub fn from_lane_fn<G>(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        num_rows: usize,
        lane: G,
    ) -> Result<Self, AkitaError>
    where
        G: Fn(usize) -> u8 + Sync,
    {
        let lane_count = num_rows
            .checked_mul(num_columns)
            .ok_or_else(|| AkitaError::InvalidInput("packed one-hot lane count overflow".into()))?;
        let (validated_rows, total_field_elems) =
            validate_geometry_shape(onehot_k, column_capacity, num_columns, lane_count)?;
        let mut lanes = AlignedBytes::zeroed(lane_count)?;
        let invalid_position = AtomicUsize::new(usize::MAX);
        let hot_entries = cfg_iter_mut!(lanes.as_mut_slice())
            .enumerate()
            .map(|(index, value)| {
                *value = lane(index);
                if usize::from(*value) >= onehot_k {
                    invalid_position.fetch_min(index, Ordering::Relaxed);
                }
                usize::from(*value != 0)
            })
            .sum();
        let invalid_position = invalid_position.load(Ordering::Relaxed);
        if invalid_position != usize::MAX {
            return Err(invalid_lane_error(onehot_k, &lanes, invalid_position));
        }
        Ok(Self {
            lanes: Arc::new(lanes),
            num_rows: validated_rows,
            num_columns,
            column_capacity,
            onehot_k,
            hot_entries,
            num_vars: total_field_elems.trailing_zeros() as usize,
            marker: PhantomData,
        })
    }

    /// Construct directly in aligned storage from a row-major fill function.
    pub fn from_row_fn<G>(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        num_rows: usize,
        fill_row: G,
    ) -> Result<Self, AkitaError>
    where
        G: Fn(usize, &mut [u8]) + Sync,
    {
        let lane_count = num_rows
            .checked_mul(num_columns)
            .ok_or_else(|| AkitaError::InvalidInput("packed one-hot lane count overflow".into()))?;
        let (validated_rows, total_field_elems) =
            validate_geometry_shape(onehot_k, column_capacity, num_columns, lane_count)?;
        let mut lanes = AlignedBytes::zeroed(lane_count)?;
        let invalid_position = AtomicUsize::new(usize::MAX);
        let hot_entries = cfg_chunks_mut!(lanes.as_mut_slice(), num_columns)
            .enumerate()
            .map(|(row, row_lanes)| {
                fill_row(row, row_lanes);
                row_lanes
                    .iter()
                    .enumerate()
                    .map(|(column, &value)| {
                        if usize::from(value) >= onehot_k {
                            invalid_position
                                .fetch_min(row * num_columns + column, Ordering::Relaxed);
                        }
                        usize::from(value != 0)
                    })
                    .sum::<usize>()
            })
            .sum();
        let invalid_position = invalid_position.load(Ordering::Relaxed);
        if invalid_position != usize::MAX {
            return Err(invalid_lane_error(onehot_k, &lanes, invalid_position));
        }
        Ok(Self {
            lanes: Arc::new(lanes),
            num_rows: validated_rows,
            num_columns,
            column_capacity,
            onehot_k,
            hot_entries,
            num_vars: total_field_elems.trailing_zeros() as usize,
            marker: PhantomData,
        })
    }

    #[must_use]
    pub fn lanes(&self) -> &[u8] {
        &self.lanes
    }

    #[must_use]
    pub const fn num_rows(&self) -> usize {
        self.num_rows
    }

    #[must_use]
    pub const fn num_columns(&self) -> usize {
        self.num_columns
    }

    #[must_use]
    pub const fn column_capacity(&self) -> usize {
        self.column_capacity
    }

    #[must_use]
    pub const fn onehot_k(&self) -> usize {
        self.onehot_k
    }

    #[must_use]
    pub const fn hot_entries(&self) -> usize {
        self.hot_entries
    }
}

impl<'a, F: FieldCore, const D: usize> PackedOneHotView<'a, F, D> {
    /// Borrow a checked packed trace buffer without transferring ownership.
    pub fn new(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        lanes: &'a [u8],
    ) -> Result<Self, AkitaError> {
        let (num_rows, total_field_elems, hot_entries) =
            validate_geometry(onehot_k, column_capacity, num_columns, lanes)?;
        validate_ring_dimension(onehot_k, total_field_elems.trailing_zeros() as usize, D)?;
        Ok(Self {
            lanes,
            num_rows,
            num_columns,
            column_capacity,
            onehot_k,
            hot_entries,
            marker: PhantomData,
        })
    }

    #[must_use]
    pub const fn lanes(self) -> &'a [u8] {
        self.lanes
    }

    #[must_use]
    pub const fn num_rows(self) -> usize {
        self.num_rows
    }

    #[must_use]
    pub const fn num_columns(self) -> usize {
        self.num_columns
    }

    #[must_use]
    pub const fn column_capacity(self) -> usize {
        self.column_capacity
    }

    #[must_use]
    pub const fn onehot_k(self) -> usize {
        self.onehot_k
    }

    #[must_use]
    pub const fn hot_entries(self) -> usize {
        self.hot_entries
    }
}

impl<F: FieldCore> RootPolyMeta<F> for PackedOneHotPoly<F> {
    fn num_ring_elems(&self) -> usize {
        (1usize << self.num_vars) / PACKED_ONEHOT_META_RING_D
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.onehot_k)
    }
}

impl<F: FieldCore, const D: usize> RootPolyShape<F, D> for PackedOneHotPoly<F> {
    fn num_ring_elems(&self) -> usize {
        (1usize << self.num_vars) / D
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.onehot_k)
    }
}

impl<F: FieldCore, const D: usize> RootCommitSource<F, D> for PackedOneHotPoly<F> {
    type CommitView<'a>
        = PackedOneHotView<'a, F, D>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        validate_ring_dimension(self.onehot_k, self.num_vars, D)?;
        Ok(PackedOneHotView {
            lanes: &self.lanes,
            num_rows: self.num_rows,
            num_columns: self.num_columns,
            column_capacity: self.column_capacity,
            onehot_k: self.onehot_k,
            hot_entries: self.hot_entries,
            marker: PhantomData,
        })
    }
}
