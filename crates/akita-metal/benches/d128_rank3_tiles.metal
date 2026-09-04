// E6 experiment: synthetic per-ring-element tiled accumulate for a D128 rank-3
// one-hot root. Appended after kernels/onehot.metal at bench compile time so it
// can reuse AkitaFp128, the transposed accumulator, and the D512 gather/add
// helpers (whose tile plane stride, 2048 elements, equals 16 positions x 128).
//
// Mapping (K = 256, D = 128): local field = row * 256 + lane, position = field / 128
// (a row spans two positions), shift = field % 128. Each hot entry adds the
// negacyclic rotation of one 128-coefficient row of each of the n_a = 3 matrix
// elements into the (column, block, element) accumulator.

#define E6_D 128u
#define E6_TILE_POSITIONS 16u
#define E6_TILE_ELEMENTS 2048u
#define E6_ROWS_PER_TILE 8u

static_assert(E6_TILE_ELEMENTS == PACKED_FP128_D512_PANEL_TILE_ELEMENTS,
              "E6 tile must match the D512 panel tile plane stride");

struct E6Params {
    ulong num_rows;
    ulong num_columns;
    ulong positions_per_block;
    ulong blocks_per_column;
    ulong n_a;
    ulong task_offset;
    ulong dispatch_tasks;
    ulong position_partials;
    ulong positions_per_partial;
    ulong output_coefficients;
    ulong density_percent;
    ulong element;
};

inline ulong e6_mix(ulong value) {
    value += 0x9e3779b97f4a7c15ul;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
    return value ^ (value >> 31);
}

kernel void e6_fill_matrix(
    device AkitaFp128 *matrix [[buffer(0)]],
    constant ulong &count [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint grid [[threads_per_grid]])
{
    for (ulong index = gid; index < count; index += grid) {
        ulong a = e6_mix(index);
        ulong b = e6_mix(a);
        AkitaFp128 value;
        value.limb = uint4(
            (uint)a, (uint)(a >> 32), (uint)b, (uint)(b >> 32) & 0x7fffffffu);
        matrix[index] = value;
    }
}

kernel void e6_fill_lanes(
    device uchar *lanes [[buffer(0)]],
    constant E6Params &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint grid [[threads_per_grid]])
{
    ulong total = params.num_rows * params.num_columns;
    for (ulong index = gid; index < total; index += grid) {
        ulong random = e6_mix(index);
        uchar lane = 0;
        if (random % 100ul < params.density_percent) {
            lane = (uchar)(e6_mix(random) % 255ul + 1ul);
        }
        lanes[index] = lane;
    }
}

kernel void e6_count_hot(
    device const uchar *lanes [[buffer(0)]],
    device uint *counts [[buffer(1)]],
    constant E6Params &params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]],
    uint3 threadgroups [[threadgroups_per_grid]])
{
    threadgroup uint partial_counts[32];
    ulong total = params.num_rows * params.num_columns;
    ulong stride = (ulong)threadgroups.x * 1024ul;
    ulong start = (ulong)threadgroup_index.x * 1024ul + thread_index;
    uint count = 0u;
    for (ulong index = start; index < total; index += stride) {
        count += lanes[index] != 0 ? 1u : 0u;
    }
    count = simd_sum(count);
    if ((thread_index & 31u) == 0u) {
        partial_counts[thread_index >> 5u] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u) {
        uint sum = 0u;
        for (uint i = 0u; i < 32u; ++i) {
            sum += partial_counts[i];
        }
        counts[threadgroup_index.x] = sum;
    }
}

// One task's contribution from the tile currently resident in shared memory.
inline void e6_accumulate_task_tile(
    thread AkitaTransposedFp128Accumulator &accumulator,
    threadgroup const uint *shared_matrix,
    device const uchar *lanes,
    constant E6Params &params,
    ulong tile_row_base,
    uint task_column,
    uint simd_lane)
{
    uint local_hot = 0u;
    bool local_selected = false;
    if (simd_lane < E6_ROWS_PER_TILE) {
        local_hot = (uint)lanes[
            (tile_row_base + (ulong)simd_lane) * params.num_columns + (ulong)task_column];
        local_selected = local_hot != 0u;
    }
    uint selected = uint(simd_ballot(local_selected).operator unsigned long());
    uint4 coefficients = uint4(simd_lane, simd_lane + 32u, simd_lane + 64u, simd_lane + 96u);
    while (selected != 0u) {
        uint selected_lane = ctz(selected);
        uint selected_hot = simd_shuffle(local_hot, selected_lane);
        uint local_position = 2u * selected_lane + (selected_hot >> 7u);
        uint4 shift = uint4(selected_hot & 127u);
        akita_fp128_d512_accumulate_mixed(
            accumulator, shared_matrix, local_position * E6_D,
            (coefficients - shift) & uint4(127u), coefficients >= shift);
        selected &= selected - 1u;
    }
}

inline void e6_store_task(
    device AkitaFp128 *partials,
    AkitaTransposedFp128Accumulator accumulator,
    constant E6Params &params,
    uint task_column,
    uint task_block,
    uint element,
    uint position_partial,
    uint simd_lane)
{
    ulong block = (ulong)task_column * params.blocks_per_column + (ulong)task_block;
    ulong output_base = (block * params.n_a + (ulong)element) * (ulong)E6_D;
    ulong partial_base =
        (ulong)position_partial * params.output_coefficients + output_base;
    partials[partial_base + simd_lane] = akita_reduce_transposed_fp128(accumulator, 0u);
    partials[partial_base + simd_lane + 32ul] = akita_reduce_transposed_fp128(accumulator, 1u);
    partials[partial_base + simd_lane + 64ul] = akita_reduce_transposed_fp128(accumulator, 2u);
    partials[partial_base + simd_lane + 96ul] = akita_reduce_transposed_fp128(accumulator, 3u);
}

// TPS = tasks per SIMD group (1 or 2). Threadgroup = 32 SIMD groups = 32 * TPS
// tasks sharing one streamed tile of `params.element`'s rows.
#define E6_PANELS_KERNEL(NAME, TPS)                                                   \
kernel void NAME(                                                                     \
    device const AkitaFp128 *matrix [[buffer(0)]],                                    \
    device const uchar *lanes [[buffer(1)]],                                          \
    device AkitaFp128 *partials [[buffer(2)]],                                        \
    constant E6Params &params [[buffer(3)]],                                          \
    uint thread_index [[thread_index_in_threadgroup]],                                \
    uint3 threadgroup_index [[threadgroup_position_in_grid]])                         \
{                                                                                     \
    threadgroup uint shared_matrix[E6_TILE_ELEMENTS * 4];                             \
    constexpr uint tasks_per_stream = 32u * (TPS);                                    \
    uint num_tasks = (uint)params.dispatch_tasks;                                     \
    uint streams = (num_tasks + tasks_per_stream - 1u) / tasks_per_stream;            \
    uint simd_lane = thread_index & 31u;                                              \
    uint simdgroup = thread_index >> 5u;                                              \
    uint position_partials = (uint)params.position_partials;                          \
    uint stream = threadgroup_index.x % streams;                                      \
    uint position_partial = threadgroup_index.x / streams;                            \
    uint element = (uint)params.element;                                              \
    uint positions_per_partial = (uint)params.positions_per_partial;                  \
    uint partial_start = position_partial * positions_per_partial;                    \
    ulong rows_per_partial = (ulong)positions_per_partial / 2ul;                      \
    ulong rows_per_block = params.positions_per_block / 2ul;                          \
    uint live_columns = (uint)params.num_columns;                                     \
    uint dispatch_task_0 = stream * tasks_per_stream + simdgroup * (TPS);             \
    bool active_0 = dispatch_task_0 < num_tasks;                                      \
    uint global_0 = (uint)params.task_offset + dispatch_task_0;                       \
    uint block_0 = global_0 / live_columns;                                           \
    uint column_0 = global_0 % live_columns;                                          \
    bool active_1 = (TPS) > 1 && dispatch_task_0 + 1u < num_tasks;                    \
    uint global_1 = global_0 + 1u;                                                    \
    uint block_1 = global_1 / live_columns;                                           \
    uint column_1 = global_1 % live_columns;                                          \
    ulong matrix_cursor =                                                             \
        ((ulong)element * params.positions_per_block + (ulong)partial_start)          \
        * (ulong)E6_D;                                                                \
    AkitaTransposedFp128Accumulator accumulator_0 = akita_transposed_fp128_zero();    \
    AkitaTransposedFp128Accumulator accumulator_1 = akita_transposed_fp128_zero();    \
    uint tile_count = positions_per_partial / E6_TILE_POSITIONS;                      \
    for (uint tile = 0u; tile < tile_count; ++tile) {                                 \
        for (uint shared_index = thread_index;                                        \
             shared_index < E6_TILE_ELEMENTS;                                         \
             shared_index += 1024u) {                                                 \
            AkitaFp128 value = matrix[matrix_cursor + (ulong)shared_index];           \
            shared_matrix[shared_index] = value.limb[0];                              \
            shared_matrix[E6_TILE_ELEMENTS + shared_index] = value.limb[1];           \
            shared_matrix[E6_TILE_ELEMENTS * 2u + shared_index] = value.limb[2];      \
            shared_matrix[E6_TILE_ELEMENTS * 3u + shared_index] = value.limb[3];      \
        }                                                                             \
        threadgroup_barrier(mem_flags::mem_threadgroup);                              \
        ulong tile_rows = (ulong)position_partial * rows_per_partial                  \
            + (ulong)tile * (ulong)E6_ROWS_PER_TILE;                                  \
        if (active_0) {                                                               \
            e6_accumulate_task_tile(                                                  \
                accumulator_0, shared_matrix, lanes, params,                          \
                (ulong)block_0 * rows_per_block + tile_rows, column_0, simd_lane);    \
        }                                                                             \
        if (active_1) {                                                               \
            e6_accumulate_task_tile(                                                  \
                accumulator_1, shared_matrix, lanes, params,                          \
                (ulong)block_1 * rows_per_block + tile_rows, column_1, simd_lane);    \
        }                                                                             \
        threadgroup_barrier(mem_flags::mem_threadgroup);                              \
        matrix_cursor += (ulong)E6_TILE_ELEMENTS;                                     \
    }                                                                                 \
    if (active_0) {                                                                   \
        e6_store_task(partials, accumulator_0, params, column_0, block_0,             \
                      element, position_partial, simd_lane);                          \
    }                                                                                 \
    if (active_1) {                                                                   \
        e6_store_task(partials, accumulator_1, params, column_1, block_1,             \
                      element, position_partial, simd_lane);                          \
    }                                                                                 \
}

E6_PANELS_KERNEL(e6_d128_rank3_panels_t1, 1)
E6_PANELS_KERNEL(e6_d128_rank3_panels_t2, 2)

kernel void e6_reduce_partials(
    device const AkitaFp128 *partials [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant E6Params &params [[buffer(2)]],
    uint output_index [[thread_position_in_grid]])
{
    if ((ulong)output_index >= params.output_coefficients) {
        return;
    }
    AkitaFp128 accumulator = partials[output_index];
    for (ulong partial = 1ul; partial < params.position_partials; ++partial) {
        accumulator = akita_add(
            accumulator, partials[partial * params.output_coefficients + (ulong)output_index]);
    }
    output[output_index] = accumulator;
}
