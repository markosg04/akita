use akita_field::Prime128OffsetA7F7;
use akita_types::SetupMatrixCapacity;

use super::*;
use crate::compute::{
    CommitInnerPlan, ComputeBackendSetup, RootCommitKernel, RootCommitSource, RootPolyMeta,
};
use crate::{AkitaProverSetup, CpuBackend, OneHotPoly};

type F = Prime128OffsetA7F7;

fn assert_commit_parity<const D: usize>(
    onehot_k: usize,
    rows: usize,
    columns: usize,
    capacity: usize,
    positions_per_block: usize,
    lanes: Vec<u8>,
) {
    let packed = PackedOneHotPoly::<F>::new(onehot_k, capacity, columns, lanes.clone()).unwrap();
    let indices: Vec<Option<u8>> = (0..capacity)
        .flat_map(|column| {
            let lanes = &lanes;
            (0..rows).map(move |row| {
                if column >= columns {
                    None
                } else {
                    let lane = lanes[row * columns + column];
                    (lane != 0).then_some(lane)
                }
            })
        })
        .collect();
    let generic = OneHotPoly::<F, u8>::new(onehot_k, D, indices).unwrap();
    let plan = CommitInnerPlan {
        n_a: 3,
        num_positions_per_block: positions_per_block,
        num_digits_inner: 1,
        log_basis_inner: 3,
    };
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        RootPolyMeta::<F>::num_vars(&packed),
        1,
        SetupMatrixCapacity {
            num_field_elements: plan.n_a * positions_per_block * D,
        },
    )
    .unwrap();
    let cpu = CpuBackend::DEFAULT;
    let prepared = cpu.prepare_setup(&setup).unwrap();
    let packed_output = cpu
        .commit_inner_group(
            &prepared,
            vec![RootCommitSource::<F, D>::commit_view(&packed).unwrap()],
            plan,
        )
        .unwrap();
    let generic_output = cpu
        .commit_inner_group(
            &prepared,
            vec![RootCommitSource::<F, D>::commit_view(&generic).unwrap()],
            plan,
        )
        .unwrap();
    assert_eq!(packed_output[0].inner_rows, generic_output[0].inner_rows);
}

#[test]
fn constructor_checks_packed_geometry() {
    assert!(PackedOneHotPoly::<F>::new(16, 4, 3, vec![0; 7]).is_err());
    assert!(PackedOneHotPoly::<F>::new(16, 3, 3, vec![0; 24]).is_err());
    assert!(PackedOneHotPoly::<F>::new(16, 4, 5, vec![0; 40]).is_err());

    let mut invalid_lane = vec![0; 24];
    invalid_lane[7] = 16;
    assert!(PackedOneHotPoly::<F>::new(16, 4, 3, invalid_lane).is_err());
}

#[test]
fn borrowed_view_checks_without_copying() {
    let lanes = vec![0; 24];
    let view = PackedOneHotView::<F, 64>::new(16, 4, 3, &lanes).unwrap();
    assert_eq!(view.lanes().as_ptr(), lanes.as_ptr());
    assert_eq!(view.num_rows(), 8);
    assert!(PackedOneHotView::<F, 64>::new(16, 4, 3, &lanes[..7]).is_err());
}

#[test]
fn owned_packed_storage_is_device_buffer_aligned() {
    let packed = PackedOneHotPoly::<F>::new(16, 4, 3, vec![0; 24]).unwrap();
    assert_eq!(
        packed.lanes().as_ptr().addr() % PACKED_ONEHOT_BUFFER_ALIGNMENT,
        0
    );
}

#[test]
fn generated_storage_matches_vec_constructor() {
    let lane = |index: usize| match index % 7 {
        0 | 1 => 0,
        2 => 1,
        3 => 15,
        _ => ((index * 5 + 3) % 15 + 1) as u8,
    };
    let expected = PackedOneHotPoly::<F>::new(16, 4, 3, (0..24).map(lane).collect()).unwrap();
    let generated = PackedOneHotPoly::<F>::from_lane_fn(16, 4, 3, 8, lane).unwrap();
    assert_eq!(generated.lanes(), expected.lanes());
    assert_eq!(generated.hot_entries(), expected.hot_entries());
    assert_eq!(
        generated.lanes().as_ptr().addr() % PACKED_ONEHOT_BUFFER_ALIGNMENT,
        0
    );
}

#[test]
fn row_generated_storage_matches_vec_constructor() {
    let lanes = (0..24)
        .map(|index| ((index * 5 + 3) % 16) as u8)
        .collect::<Vec<_>>();
    let expected = PackedOneHotPoly::<F>::new(16, 4, 3, lanes.clone()).unwrap();
    let generated = PackedOneHotPoly::<F>::from_row_fn(16, 4, 3, 8, |row, output| {
        output.copy_from_slice(&lanes[row * 3..row * 3 + 3]);
    })
    .unwrap();
    assert_eq!(generated.lanes(), expected.lanes());
    assert_eq!(generated.hot_entries(), expected.hot_entries());
    assert_eq!(
        generated.lanes().as_ptr().addr() % PACKED_ONEHOT_BUFFER_ALIGNMENT,
        0
    );
}

#[test]
fn generated_storage_rejects_out_of_range_lanes() {
    let lane_error =
        PackedOneHotPoly::<F>::from_lane_fn(16, 4, 3, 8, |index| if index == 7 { 16 } else { 0 })
            .unwrap_err();
    assert!(lane_error.to_string().contains("lane 16 at byte 7"));

    let row_error = PackedOneHotPoly::<F>::from_row_fn(16, 4, 3, 8, |row, output| {
        if row == 2 {
            output[1] = 16;
        }
    })
    .unwrap_err();
    assert!(row_error.to_string().contains("lane 16 at byte 7"));
}

#[test]
fn streaming_storage_publishes_prefixes_and_finalizes_without_copying() {
    let (stream, mut writer) = StreamingPackedOneHotPoly::<F>::new(16, 4, 3, 8).unwrap();
    let view = RootCommitSource::<F, 64>::commit_view(&stream).unwrap();
    let lanes_ptr = std::thread::scope(|scope| {
        let consumer = scope.spawn(move || {
            let prefix = view.wait_lanes(0..4).unwrap().to_vec();
            assert_eq!(view.wait_hot_entries().unwrap(), 23);
            prefix
        });
        writer
            .fill_next_rows(4, |row| {
                Ok::<[u8; 3], String>(std::array::from_fn(|column| ((row + column) % 16) as u8))
            })
            .unwrap();
        writer
            .fill_next_rows(4, |row| {
                Ok::<[u8; 3], String>(std::array::from_fn(|column| ((row + column) % 16) as u8))
            })
            .unwrap();
        writer.finish().unwrap();
        assert_eq!(consumer.join().unwrap().len(), 12);
        stream.finalize().unwrap().lanes().as_ptr()
    });
    let packed = stream.finalize().unwrap();
    assert_eq!(packed.lanes().as_ptr(), lanes_ptr);
    assert_eq!(packed.hot_entries(), 23);
}

#[test]
fn streaming_storage_failure_wakes_consumers() {
    let (stream, mut writer) = StreamingPackedOneHotPoly::<F>::new(16, 4, 3, 8).unwrap();
    let view = RootCommitSource::<F, 64>::commit_view(&stream).unwrap();
    let error = writer
        .fill_next_rows(4, |row| {
            let mut lanes = [0; 3];
            if row == 2 {
                lanes[1] = 16;
            }
            Ok(lanes)
        })
        .unwrap_err();
    assert!(error.to_string().contains("lane 16 at byte 7"));
    assert!(view.wait_lanes(0..4).is_err());
    assert!(stream.finalize().is_err());
}

#[test]
fn packed_commit_matches_cared_geometries_and_sentinels() {
    for (onehot_k, rows, columns, capacity, positions) in [(16, 16, 3, 4, 4), (256, 8, 3, 4, 4)] {
        let patterns = [
            vec![0; rows * columns],
            vec![1; rows * columns],
            vec![(onehot_k - 1) as u8; rows * columns],
            (0..rows * columns)
                .map(|index| {
                    if index % 4 == 0 {
                        0
                    } else {
                        ((index * 73 + 19) % (onehot_k - 1) + 1) as u8
                    }
                })
                .collect(),
        ];
        for lanes in patterns {
            if onehot_k == 16 {
                assert_commit_parity::<64>(onehot_k, rows, columns, capacity, positions, lanes);
            } else {
                assert_commit_parity::<64>(
                    onehot_k,
                    rows,
                    columns,
                    capacity,
                    positions,
                    lanes.clone(),
                );
                assert_commit_parity::<128>(
                    onehot_k,
                    rows,
                    columns,
                    capacity,
                    positions,
                    lanes.clone(),
                );
                assert_commit_parity::<512>(onehot_k, rows, columns, capacity, positions, lanes);
            }
        }
    }
}
