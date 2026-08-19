use std::alloc::{alloc, alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Deref, Range};
use std::ptr::{copy_nonoverlapping, NonNull};
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

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

    fn uninit(len: usize) -> Result<Self, AkitaError> {
        debug_assert_ne!(len, 0);
        let layout =
            Layout::from_size_align(len, PACKED_ONEHOT_BUFFER_ALIGNMENT).map_err(|_| {
                AkitaError::InvalidInput("packed one-hot allocation layout is too large".into())
            })?;
        // SAFETY: `layout` has nonzero size and valid alignment. Streaming
        // publication never exposes a byte before its row has been written.
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));
        Ok(Self { ptr, len })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` exclusively owns `len` initialized bytes.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    unsafe fn slice(&self, range: Range<usize>) -> &[u8] {
        debug_assert!(range.start <= range.end && range.end <= self.len);
        // SAFETY: the caller guarantees that this initialized range is not
        // mutated for the returned borrow's lifetime.
        unsafe {
            slice::from_raw_parts(self.ptr.as_ptr().add(range.start), range.end - range.start)
        }
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

// SAFETY: the allocation is owned; mutable access is exclusive to construction
// or to disjoint, not-yet-published ranges of a streaming writer.
unsafe impl Send for AlignedBytes {}
// SAFETY: streaming publication synchronizes through `StreamingProgress`; a
// published range is immutable, and the sole writer advances into its suffix.
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

#[derive(Debug)]
struct StreamingProgress {
    completed_rows: usize,
    hot_entries: usize,
    finished: bool,
    failure: Option<String>,
}

#[derive(Debug)]
struct StreamingPackedOneHotInner<F: FieldCore> {
    lanes: Arc<AlignedBytes>,
    num_rows: usize,
    num_columns: usize,
    column_capacity: usize,
    onehot_k: usize,
    num_vars: usize,
    progress: Mutex<StreamingProgress>,
    ready: Condvar,
    marker: PhantomData<F>,
}

/// A packed one-hot source whose completed row prefix can be consumed while
/// its remaining rows are still being generated.
#[derive(Debug, Clone)]
pub struct StreamingPackedOneHotPoly<F: FieldCore> {
    inner: Arc<StreamingPackedOneHotInner<F>>,
}

/// The unique sequential producer for a [`StreamingPackedOneHotPoly`].
#[derive(Debug)]
pub struct PackedOneHotStreamWriter<F: FieldCore> {
    inner: Arc<StreamingPackedOneHotInner<F>>,
    next_row: usize,
    closed: bool,
}

/// An owned streaming commit view at ring dimension `D`.
#[derive(Debug, Clone)]
pub struct StreamingPackedOneHotView<F: FieldCore, const D: usize> {
    inner: Arc<StreamingPackedOneHotInner<F>>,
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

impl<F: FieldCore> StreamingPackedOneHotPoly<F> {
    /// Allocate aligned row-major storage and return its source and unique writer.
    pub fn new(
        onehot_k: usize,
        column_capacity: usize,
        num_columns: usize,
        num_rows: usize,
    ) -> Result<(Self, PackedOneHotStreamWriter<F>), AkitaError> {
        let lane_count = num_rows
            .checked_mul(num_columns)
            .ok_or_else(|| AkitaError::InvalidInput("packed one-hot lane count overflow".into()))?;
        let (validated_rows, total_field_elems) =
            validate_geometry_shape(onehot_k, column_capacity, num_columns, lane_count)?;
        let inner = Arc::new(StreamingPackedOneHotInner {
            lanes: Arc::new(AlignedBytes::uninit(lane_count)?),
            num_rows: validated_rows,
            num_columns,
            column_capacity,
            onehot_k,
            num_vars: total_field_elems.trailing_zeros() as usize,
            progress: Mutex::new(StreamingProgress {
                completed_rows: 0,
                hot_entries: 0,
                finished: false,
                failure: None,
            }),
            ready: Condvar::new(),
            marker: PhantomData,
        });
        Ok((
            Self {
                inner: inner.clone(),
            },
            PackedOneHotStreamWriter {
                inner,
                next_row: 0,
                closed: false,
            },
        ))
    }

    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.inner.num_rows
    }

    #[must_use]
    pub fn num_columns(&self) -> usize {
        self.inner.num_columns
    }

    #[must_use]
    pub fn column_capacity(&self) -> usize {
        self.inner.column_capacity
    }

    #[must_use]
    pub fn onehot_k(&self) -> usize {
        self.inner.onehot_k
    }

    /// Wait for all rows and return the immutable packed owner without copying.
    pub fn finalize(&self) -> Result<PackedOneHotPoly<F>, AkitaError> {
        let hot_entries = wait_for_rows(&self.inner, self.inner.num_rows, true)?;
        Ok(PackedOneHotPoly {
            lanes: self.inner.lanes.clone(),
            num_rows: self.inner.num_rows,
            num_columns: self.inner.num_columns,
            column_capacity: self.inner.column_capacity,
            onehot_k: self.inner.onehot_k,
            hot_entries,
            num_vars: self.inner.num_vars,
            marker: PhantomData,
        })
    }
}

impl<F: FieldCore> PackedOneHotStreamWriter<F> {
    /// Fill and publish the next contiguous row range.
    ///
    /// Publication occurs only after every row in the range was filled and
    /// validated, so a concurrent consumer never observes partial rows.
    pub fn fill_next_rows<const N: usize, G>(
        &mut self,
        row_count: usize,
        fill_row: G,
    ) -> Result<(), AkitaError>
    where
        G: Fn(usize) -> Result<[u8; N], String> + Sync,
    {
        if self.closed {
            return Err(AkitaError::InvalidInput(
                "packed one-hot stream writer is already closed".into(),
            ));
        }
        if row_count == 0 {
            return Err(AkitaError::InvalidInput(
                "packed one-hot stream row batch must be nonempty".into(),
            ));
        }
        if N < self.inner.num_columns {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot row output width {N} is smaller than {} live columns",
                self.inner.num_columns
            )));
        }
        let row_end = self
            .next_row
            .checked_add(row_count)
            .ok_or_else(|| AkitaError::InvalidInput("packed one-hot row range overflow".into()))?;
        if row_end > self.inner.num_rows {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot stream row range {}..{row_end} exceeds {} rows",
                self.next_row, self.inner.num_rows
            )));
        }
        {
            let progress = self
                .inner
                .progress
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(failure) = &progress.failure {
                return Err(stream_failure(failure));
            }
            if progress.finished || progress.completed_rows != self.next_row {
                return Err(AkitaError::InvalidInput(
                    "packed one-hot stream writer progress is inconsistent".into(),
                ));
            }
        }

        let first_lane = self
            .next_row
            .checked_mul(self.inner.num_columns)
            .ok_or_else(|| AkitaError::InvalidInput("packed first lane overflow".into()))?;
        let final_lane = row_end
            .checked_mul(self.inner.num_columns)
            .ok_or_else(|| AkitaError::InvalidInput("packed final lane overflow".into()))?;
        // SAFETY: `MaybeUninit<u8>` may represent the unpublished allocation.
        // This unique writer owns this suffix, and publication advances only
        // after every row below has been initialized.
        let lanes: &mut [MaybeUninit<u8>] = unsafe {
            slice::from_raw_parts_mut(
                self.inner
                    .lanes
                    .ptr
                    .as_ptr()
                    .add(first_lane)
                    .cast::<MaybeUninit<u8>>(),
                final_lane - first_lane,
            )
        };
        let invalid_position = AtomicUsize::new(usize::MAX);
        let fill_failure = Mutex::new(None::<(usize, String)>);
        let hot_entries = cfg_chunks_mut!(lanes, self.inner.num_columns)
            .enumerate()
            .map(|(row_offset, row_lanes)| {
                let row = self.next_row + row_offset;
                let values = match fill_row(row) {
                    Ok(values) => values,
                    Err(message) => {
                        let mut failure = fill_failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if failure
                            .as_ref()
                            .is_none_or(|(failed_row, _)| row < *failed_row)
                        {
                            *failure = Some((row, message));
                        }
                        return 0;
                    }
                };
                values[..self.inner.num_columns]
                    .iter()
                    .enumerate()
                    .map(|(column, &value)| {
                        if usize::from(value) >= self.inner.onehot_k {
                            invalid_position.fetch_min(
                                row * self.inner.num_columns + column,
                                Ordering::Relaxed,
                            );
                        }
                        row_lanes[column].write(value);
                        usize::from(value != 0)
                    })
                    .sum::<usize>()
            })
            .sum::<usize>();

        let failure = fill_failure
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .map(|(row, message)| format!("packed one-hot row {row} fill failed: {message}"));
        let invalid_position = invalid_position.load(Ordering::Relaxed);
        let failure = failure.or_else(|| {
            (invalid_position != usize::MAX).then(|| {
                // SAFETY: an invalid value came from a successfully generated
                // row; the parallel fill completed all row writes before here.
                let value = unsafe { *self.inner.lanes.ptr.as_ptr().add(invalid_position) };
                format!(
                    "packed one-hot lane {value} at byte {invalid_position} is outside K={}",
                    self.inner.onehot_k
                )
            })
        });
        if let Some(failure) = failure {
            self.mark_failed(failure.clone());
            return Err(stream_failure(&failure));
        }

        let mut progress = self
            .inner
            .progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        progress.completed_rows = row_end;
        progress.hot_entries = progress
            .hot_entries
            .checked_add(hot_entries)
            .ok_or_else(|| AkitaError::InvalidInput("packed hot-entry count overflow".into()))?;
        self.next_row = row_end;
        drop(progress);
        self.inner.ready.notify_all();
        Ok(())
    }

    /// Mark a fully generated stream complete and wake all consumers.
    pub fn finish(mut self) -> Result<(), AkitaError> {
        if self.next_row != self.inner.num_rows {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot stream finished after {} of {} rows",
                self.next_row, self.inner.num_rows
            )));
        }
        let mut progress = self
            .inner
            .progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(failure) = &progress.failure {
            return Err(stream_failure(failure));
        }
        progress.finished = true;
        self.closed = true;
        drop(progress);
        self.inner.ready.notify_all();
        Ok(())
    }

    fn mark_failed(&mut self, failure: String) {
        let mut progress = self
            .inner
            .progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _ = progress.failure.get_or_insert(failure);
        self.closed = true;
        drop(progress);
        self.inner.ready.notify_all();
    }
}

impl<F: FieldCore> Drop for PackedOneHotStreamWriter<F> {
    fn drop(&mut self) {
        if !self.closed {
            self.mark_failed("packed one-hot stream writer dropped before completion".into());
        }
    }
}

impl<F: FieldCore, const D: usize> StreamingPackedOneHotView<F, D> {
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.inner.num_rows
    }

    #[must_use]
    pub fn num_columns(&self) -> usize {
        self.inner.num_columns
    }

    #[must_use]
    pub fn column_capacity(&self) -> usize {
        self.inner.column_capacity
    }

    #[must_use]
    pub fn onehot_k(&self) -> usize {
        self.inner.onehot_k
    }

    #[must_use]
    pub fn lane_count(&self) -> usize {
        self.inner.lanes.len
    }

    /// Wait until `rows` is immutable, then borrow its contiguous lane bytes.
    pub fn wait_lanes(&self, rows: Range<usize>) -> Result<&[u8], AkitaError> {
        if rows.start > rows.end || rows.end > self.inner.num_rows {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot ready range {}..{} exceeds {} rows",
                rows.start, rows.end, self.inner.num_rows
            )));
        }
        let _ = wait_for_rows(&self.inner, rows.end, false)?;
        let first_lane = rows
            .start
            .checked_mul(self.inner.num_columns)
            .ok_or_else(|| AkitaError::InvalidInput("packed first lane overflow".into()))?;
        let final_lane = rows
            .end
            .checked_mul(self.inner.num_columns)
            .ok_or_else(|| AkitaError::InvalidInput("packed final lane overflow".into()))?;
        // SAFETY: `wait_for_rows` synchronizes with publication. The sole
        // writer never mutates rows below its published prefix afterward.
        Ok(unsafe { self.inner.lanes.slice(first_lane..final_lane) })
    }

    /// Wait for the producer and return the exact nonzero lane count.
    pub fn wait_hot_entries(&self) -> Result<usize, AkitaError> {
        wait_for_rows(&self.inner, self.inner.num_rows, true)
    }
}

fn wait_for_rows<F: FieldCore>(
    inner: &StreamingPackedOneHotInner<F>,
    required_rows: usize,
    require_finished: bool,
) -> Result<usize, AkitaError> {
    let mut progress = inner
        .progress
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if let Some(failure) = &progress.failure {
            return Err(stream_failure(failure));
        }
        if progress.completed_rows >= required_rows && (!require_finished || progress.finished) {
            return Ok(progress.hot_entries);
        }
        if progress.finished {
            return Err(AkitaError::InvalidInput(format!(
                "packed one-hot stream ended at {} rows before required row {required_rows}",
                progress.completed_rows
            )));
        }
        progress = inner
            .ready
            .wait(progress)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn stream_failure(failure: &str) -> AkitaError {
    AkitaError::InvalidInput(format!("packed one-hot stream failed: {failure}"))
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

impl<F: FieldCore> RootPolyMeta<F> for StreamingPackedOneHotPoly<F> {
    fn num_ring_elems(&self) -> usize {
        (1usize << self.inner.num_vars) / PACKED_ONEHOT_META_RING_D
    }

    fn num_vars(&self) -> usize {
        self.inner.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.inner.onehot_k)
    }
}

impl<F: FieldCore, const D: usize> RootPolyShape<F, D> for StreamingPackedOneHotPoly<F> {
    fn num_ring_elems(&self) -> usize {
        (1usize << self.inner.num_vars) / D
    }

    fn num_vars(&self) -> usize {
        self.inner.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.inner.onehot_k)
    }
}

impl<F: FieldCore, const D: usize> RootCommitSource<F, D> for StreamingPackedOneHotPoly<F> {
    type CommitView<'a>
        = StreamingPackedOneHotView<F, D>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        validate_ring_dimension(self.inner.onehot_k, self.inner.num_vars, D)?;
        Ok(StreamingPackedOneHotView {
            inner: self.inner.clone(),
        })
    }
}
