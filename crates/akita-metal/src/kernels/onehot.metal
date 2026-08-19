#include <metal_stdlib>
using namespace metal;

#define AKITA_OFFSET 0xffffa7f7u
#define NONE_INDEX 0xffffu
#define MAX_DEFERRED_ACCUMULATIONS 32768u
#define PACKED_FP128_D512_PANEL_TILE_ELEMENTS 2048u

struct AkitaFp128 {
    uint4 limb;
};

static_assert(
    sizeof(uint) * PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 4 == 32768,
    "fp128 D512 panel tile must occupy 32 KiB");

struct AkitaCorrection {
    AkitaFp128 value;
    uint carry;
};

struct AkitaWideAccumulator {
    int4 low_digits;
    int4 high_digits;
};

struct AkitaTransposedFp128Accumulator {
    uint4 word_0;
    uint4 word_1;
    uint4 word_2;
    uint4 word_3;
    int4 wraps;
};

struct OneHotCommitParams {
    ulong num_sources;
    ulong chunks_per_source;
    ulong onehot_k;
    ulong ring_d;
    ulong n_a;
    ulong positions_per_block;
    ulong num_digits_inner;
    ulong num_blocks;
    ulong total_field_elements;
    ulong output_coefficients;
    ulong blocks_per_threadgroup;
    ulong log_onehot_k;
    ulong log_ring_d;
};

struct PackedOneHotCommitParams {
    ulong num_rows;
    ulong num_columns;
    ulong lane_stride;
    ulong column_capacity;
    ulong onehot_k;
    ulong ring_d;
    ulong n_a;
    ulong positions_per_block;
    ulong num_digits_inner;
    ulong blocks_per_column;
    ulong full_blocks_per_column;
    ulong boundary_columns;
    ulong num_blocks;
    ulong output_coefficients;
    ulong columns_per_threadgroup;
    ulong position_partials_per_block;
    ulong positions_per_partial;
    ulong log_ring_d;
};

inline AkitaFp128 akita_zero() {
    AkitaFp128 result;
    result.limb = uint4(0u);
    return result;
}

inline AkitaFp128 akita_select(bool take_lhs, AkitaFp128 lhs, AkitaFp128 rhs) {
    AkitaFp128 result;
    uint mask = take_lhs ? 0xffffffffu : 0u;
    result.limb = (lhs.limb & mask) | (rhs.limb & ~mask);
    return result;
}

inline AkitaCorrection akita_add_offset(AkitaFp128 value) {
    AkitaCorrection result;
    ulong sum = (ulong)value.limb[0] + (ulong)AKITA_OFFSET;
    result.value.limb[0] = (uint)sum;
    ulong carry = sum >> 32;
    for (uint i = 1; i < 4; ++i) {
        sum = (ulong)value.limb[i] + carry;
        result.value.limb[i] = (uint)sum;
        carry = sum >> 32;
    }
    result.carry = (uint)carry;
    return result;
}

inline AkitaFp128 akita_sub_offset(AkitaFp128 value) {
    AkitaFp128 result;
    ulong subtrahend = (ulong)AKITA_OFFSET;
    for (uint i = 0; i < 4; ++i) {
        ulong word = (ulong)value.limb[i];
        result.limb[i] = (uint)(word - subtrahend);
        subtrahend = word < subtrahend ? 1ul : 0ul;
    }
    return result;
}

inline AkitaFp128 akita_add(AkitaFp128 lhs, AkitaFp128 rhs) {
    AkitaFp128 sum;
    ulong carry = 0ul;
    for (uint i = 0; i < 4; ++i) {
        ulong word = (ulong)lhs.limb[i] + (ulong)rhs.limb[i] + carry;
        sum.limb[i] = (uint)word;
        carry = word >> 32;
    }
    AkitaCorrection corrected = akita_add_offset(sum);
    return akita_select(carry != 0ul || corrected.carry != 0u, corrected.value, sum);
}

inline AkitaFp128 akita_sub(AkitaFp128 lhs, AkitaFp128 rhs) {
    AkitaFp128 difference;
    ulong borrow = 0ul;
    for (uint i = 0; i < 4; ++i) {
        ulong subtrahend = (ulong)rhs.limb[i] + borrow;
        ulong word = (ulong)lhs.limb[i];
        difference.limb[i] = (uint)(word - subtrahend);
        borrow = word < subtrahend ? 1ul : 0ul;
    }
    AkitaFp128 corrected = akita_sub_offset(difference);
    return akita_select(borrow != 0ul, corrected, difference);
}

inline AkitaWideAccumulator akita_wide_zero() {
    AkitaWideAccumulator result;
    result.low_digits = int4(0);
    result.high_digits = int4(0);
    return result;
}

inline void akita_wide_accumulate(
    thread AkitaWideAccumulator &accumulator,
    AkitaFp128 value,
    bool positive)
{
    int sign = positive ? 1 : -1;
    accumulator.low_digits += sign * int4(value.limb & uint4(0xffffu));
    accumulator.high_digits += sign * int4(value.limb >> 16u);
}

inline AkitaFp128 akita_reduce_wide(AkitaWideAccumulator accumulator) {
    AkitaFp128 base;
    long carry = 0l;
    for (uint word = 0u; word < 4u; ++word) {
        long low = (long)accumulator.low_digits[word] + carry;
        uint low_digit = (uint)low & 0xffffu;
        carry = low >> 16;
        long high = (long)accumulator.high_digits[word] + carry;
        uint high_digit = (uint)high & 0xffffu;
        carry = high >> 16;
        base.limb[word] = low_digit | (high_digit << 16u);
    }

    AkitaCorrection canonical = akita_add_offset(base);
    base = akita_select(canonical.carry != 0u, canonical.value, base);
    if (carry == 0l) {
        return base;
    }

    ulong magnitude = (ulong)(carry > 0l ? carry : -carry);
    ulong correction_word = magnitude * (ulong)AKITA_OFFSET;
    AkitaFp128 correction = akita_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32u);
    return carry > 0l
        ? akita_add(base, correction)
        : akita_sub(base, correction);
}

inline AkitaTransposedFp128Accumulator akita_transposed_fp128_zero() {
    AkitaTransposedFp128Accumulator result;
    result.word_0 = uint4(0u);
    result.word_1 = uint4(0u);
    result.word_2 = uint4(0u);
    result.word_3 = uint4(0u);
    result.wraps = int4(0);
    return result;
}

inline AkitaFp128 akita_reduce_transposed_fp128(
    AkitaTransposedFp128Accumulator accumulator,
    uint component)
{
    AkitaFp128 base;
    base.limb = uint4(
        accumulator.word_0[component],
        accumulator.word_1[component],
        accumulator.word_2[component],
        accumulator.word_3[component]);
    AkitaCorrection canonical = akita_add_offset(base);
    base = akita_select(canonical.carry != 0u, canonical.value, base);

    long wraps = (long)accumulator.wraps[component];
    if (wraps == 0l) {
        return base;
    }
    ulong magnitude = (ulong)(wraps > 0l ? wraps : -wraps);
    ulong correction_word = magnitude * (ulong)AKITA_OFFSET;
    AkitaFp128 correction = akita_zero();
    correction.limb[0] = (uint)correction_word;
    correction.limb[1] = (uint)(correction_word >> 32u);
    return wraps > 0l
        ? akita_add(base, correction)
        : akita_sub(base, correction);
}

inline uint4 akita_add_transposed_word(
    uint4 lhs,
    uint4 rhs,
    thread uint4 &carry)
{
    uint4 base = lhs + rhs;
    uint4 sum = base + carry;
    carry = select(
        uint4(0u),
        uint4(1u),
        (base < lhs) | (sum < base));
    return sum;
}

kernel void akita_packed_onehot_reduce_partials(
    device const AkitaFp128 *partials [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant PackedOneHotCommitParams &params [[buffer(2)]],
    uint output_index [[thread_position_in_grid]])
{
    if ((ulong)output_index >= params.output_coefficients) {
        return;
    }
    ulong coefficients_per_block = params.n_a * params.ring_d;
    ulong block = (ulong)output_index / coefficients_per_block;
    ulong column = block / params.blocks_per_column;
    if (column >= params.num_columns) {
        output[output_index] = akita_zero();
        return;
    }
    AkitaFp128 accumulator = partials[output_index];
    for (ulong partial = 1ul;
         partial < params.position_partials_per_block;
         ++partial) {
        ulong partial_index =
            partial * params.output_coefficients + (ulong)output_index;
        accumulator = akita_add(accumulator, partials[partial_index]);
    }
    output[output_index] = accumulator;
}

inline uint4 akita_fp128_d512_gather_word(
    threadgroup const uint *matrix,
    uint word,
    uint matrix_base,
    uint4 sources)
{
    uint plane_base = word * PACKED_FP128_D512_PANEL_TILE_ELEMENTS;
    return uint4(
        matrix[plane_base + matrix_base + sources[0]],
        matrix[plane_base + matrix_base + sources[1]],
        matrix[plane_base + matrix_base + sources[2]],
        matrix[plane_base + matrix_base + sources[3]]);
}

inline void akita_fp128_d512_accumulate_value(
    thread AkitaTransposedFp128Accumulator &accumulator,
    uint4 value_0,
    uint4 value_1,
    uint4 value_2,
    uint4 value_3,
    bool4 positive)
{
    uint4 carry = select(uint4(1u), uint4(0u), positive);
    accumulator.word_0 = akita_add_transposed_word(
        accumulator.word_0, select(~value_0, value_0, positive), carry);
    accumulator.word_1 = akita_add_transposed_word(
        accumulator.word_1, select(~value_1, value_1, positive), carry);
    accumulator.word_2 = akita_add_transposed_word(
        accumulator.word_2, select(~value_2, value_2, positive), carry);
    accumulator.word_3 = akita_add_transposed_word(
        accumulator.word_3, select(~value_3, value_3, positive), carry);
    int4 carry_words = int4(carry);
    accumulator.wraps += select(carry_words - int4(1), carry_words, positive);
}

inline void akita_fp128_d512_accumulate_group(
    thread AkitaTransposedFp128Accumulator &accumulator,
    threadgroup const uint *matrix,
    uint simd_lane,
    uint coefficient_group,
    uint local_position,
    uint local_shift,
    bool odd_row)
{
    uint coefficient_base = coefficient_group * 128u;
    uint4 coefficients = uint4(
        simd_lane + coefficient_base,
        simd_lane + coefficient_base + 32u,
        simd_lane + coefficient_base + 64u,
        simd_lane + coefficient_base + 96u);
    uint4 sources;
    bool4 positive;
    if (!odd_row) {
        if (coefficient_group < 2u) {
            sources = (coefficients - uint4(local_shift)) & uint4(511u);
            positive = coefficients >= uint4(local_shift);
        } else {
            sources = coefficients - uint4(local_shift);
            positive = bool4(true);
        }
    } else {
        if (coefficient_group < 2u) {
            sources = coefficients + uint4(256u) - uint4(local_shift);
            positive = bool4(false);
        } else {
            uint shift = 256u + local_shift;
            sources = (coefficients - uint4(shift)) & uint4(511u);
            positive = coefficients >= uint4(shift);
        }
    }
    uint matrix_base = local_position * 512u;
    akita_fp128_d512_accumulate_value(
        accumulator,
        akita_fp128_d512_gather_word(matrix, 0u, matrix_base, sources),
        akita_fp128_d512_gather_word(matrix, 1u, matrix_base, sources),
        akita_fp128_d512_gather_word(matrix, 2u, matrix_base, sources),
        akita_fp128_d512_gather_word(matrix, 3u, matrix_base, sources),
        positive);
}

inline void akita_store_fp128_d512_group(
    device AkitaFp128 *partials,
    AkitaTransposedFp128Accumulator accumulator,
    uint coefficient_group,
    uint column,
    uint block_in_column,
    uint blocks_per_column,
    uint n_a,
    uint a_row,
    uint position_partial,
    uint output_coefficients,
    uint simd_lane)
{
    uint block = column * blocks_per_column + block_in_column;
    uint output_base = (block * n_a + a_row) * 512u + coefficient_group * 128u;
    uint partial_base = position_partial * output_coefficients + output_base;
    partials[partial_base + simd_lane] =
        akita_reduce_transposed_fp128(accumulator, 0u);
    partials[partial_base + simd_lane + 32u] =
        akita_reduce_transposed_fp128(accumulator, 1u);
    partials[partial_base + simd_lane + 64u] =
        akita_reduce_transposed_fp128(accumulator, 2u);
    partials[partial_base + simd_lane + 96u] =
        akita_reduce_transposed_fp128(accumulator, 3u);
}

kernel void akita_packed_onehot_commit_fp128_d512_panels(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const uchar *lanes [[buffer(1)]],
    device AkitaFp128 *partials [[buffer(2)]],
    constant PackedOneHotCommitParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup uint shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 4];

    constexpr uint tasks_per_stream = 32u;
    constexpr uint threads_per_threadgroup = 1024u;
    constexpr uint positions_per_tile = 4u;
    constexpr uint rows_per_tile = 8u;
    uint live_columns = (uint)params.num_columns;
    uint num_tasks = (uint)params.num_blocks;
    uint blocks_per_column = (uint)params.blocks_per_column;
    uint streams = (num_tasks + tasks_per_stream - 1u) / tasks_per_stream;
    uint simd_lane = thread_index & 31u;
    uint simdgroup = thread_index >> 5u;
    uint position_partials = (uint)params.position_partials_per_block;
    uint groups_per_band = streams * position_partials * (uint)params.n_a;
    uint coefficient_band = threadgroup_index.x / groups_per_band;
    uint group = threadgroup_index.x % groups_per_band;
    uint stream = group % streams;
    uint partial_group = group / streams;
    uint position_partial = partial_group % position_partials;
    uint a_row = partial_group / position_partials;
    uint positions_per_partial = (uint)params.positions_per_partial;
    uint partial_start = position_partial * positions_per_partial;
    uint rows_per_partial = positions_per_partial * 2u;
    uint rows_per_block = (uint)params.positions_per_block * 2u;
    uint output_coefficients = (uint)params.output_coefficients;
    uint global_task = stream * tasks_per_stream + simdgroup;
    bool simdgroup_active = global_task < num_tasks;
    uint task_column = global_task / blocks_per_column;
    uint task_block = global_task % blocks_per_column;
    ulong matrix_cursor =
        ((ulong)a_row * params.positions_per_block + (ulong)partial_start) * 512ul;

    AkitaTransposedFp128Accumulator accumulator_0 = akita_transposed_fp128_zero();
    AkitaTransposedFp128Accumulator accumulator_1 = akita_transposed_fp128_zero();
    uint coefficient_group_0 = coefficient_band * 2u;
    uint coefficient_group_1 = coefficient_group_0 + 1u;

    uint tile_count = positions_per_partial / positions_per_tile;
    for (uint tile = 0u; tile < tile_count; ++tile) {
        for (uint shared_index = thread_index;
             shared_index < PACKED_FP128_D512_PANEL_TILE_ELEMENTS;
             shared_index += threads_per_threadgroup) {
            AkitaFp128 value = matrix[matrix_cursor + (ulong)shared_index];
            shared_matrix[shared_index] = value.limb[0];
            shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS + shared_index] =
                value.limb[1];
            shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 2u + shared_index] =
                value.limb[2];
            shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 3u + shared_index] =
                value.limb[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint local_hot = 0u;
        if (simdgroup_active && simd_lane < rows_per_tile) {
            ulong trace_row = (ulong)task_block * (ulong)rows_per_block
                + (ulong)position_partial * (ulong)rows_per_partial
                + (ulong)tile * (ulong)rows_per_tile
                + (ulong)simd_lane;
            local_hot = (uint)lanes[
                trace_row * params.lane_stride + (ulong)task_column];
        }
        uint selected = uint(
            simd_ballot(local_hot != 0u).operator unsigned long());
        while (selected != 0u) {
            uint selected_lane = ctz(selected);
            uint selected_hot = simd_shuffle(local_hot, selected_lane);
            uint local_position = selected_lane >> 1u;
            bool odd_row = (selected_lane & 1u) != 0u;
            akita_fp128_d512_accumulate_group(
                accumulator_0, shared_matrix, simd_lane, coefficient_group_0,
                local_position, selected_hot, odd_row);
            akita_fp128_d512_accumulate_group(
                accumulator_1, shared_matrix, simd_lane, coefficient_group_1,
                local_position, selected_hot, odd_row);
            selected &= selected - 1u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matrix_cursor += (ulong)PACKED_FP128_D512_PANEL_TILE_ELEMENTS;
    }

    if (simdgroup_active) {
        akita_store_fp128_d512_group(
            partials, accumulator_0, coefficient_group_0, task_column, task_block,
            blocks_per_column, (uint)params.n_a, a_row, position_partial,
            output_coefficients, simd_lane);
        akita_store_fp128_d512_group(
            partials, accumulator_1, coefficient_group_1, task_column, task_block,
            blocks_per_column, (uint)params.n_a, a_row, position_partial,
            output_coefficients, simd_lane);
    }
}

kernel void akita_onehot_commit_block_batched(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const ushort *hot_indices [[buffer(1)]],
    device AkitaFp128 *output [[buffer(2)]],
    constant OneHotCommitParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    uint ring_d = (uint)params.ring_d;
    uint log_ring_d = (uint)params.log_ring_d;
    uint onehot_k = (uint)params.onehot_k;
    uint log_onehot_k = (uint)params.log_onehot_k;
    uint positions_per_block = (uint)params.positions_per_block;
    uint blocks_per_threadgroup = (uint)params.blocks_per_threadgroup;
    uint num_blocks = (uint)params.num_blocks;
    uint n_a = (uint)params.n_a;
    uint coefficient = thread_index & (ring_d - 1u);
    uint block_lane = thread_index >> log_ring_d;
    uint block_groups =
        (num_blocks + blocks_per_threadgroup - 1u) / blocks_per_threadgroup;
    uint group = threadgroup_index.x;
    uint block_group = group % block_groups;
    uint source_row = group / block_groups;
    uint row = source_row % n_a;
    uint source = source_row / n_a;
    uint block = block_group * blocks_per_threadgroup + block_lane;
    if (source >= (uint)params.num_sources || block >= num_blocks) {
        return;
    }

    uint total_field_elements = (uint)params.total_field_elements;
    uint block_field_start = block * positions_per_block * ring_d;
    uint block_field_end = min(
        block_field_start + positions_per_block * ring_d,
        total_field_elements);
    uint chunk_start = block_field_start >> log_onehot_k;
    uint chunk_end =
        (block_field_end + onehot_k - 1u) >> log_onehot_k;
    uint source_chunk_base = source * (uint)params.chunks_per_source;
    uint active_a_cols = positions_per_block * (uint)params.num_digits_inner;
    uint block_ring_start = block * positions_per_block;

    AkitaFp128 accumulator = akita_zero();
    AkitaWideAccumulator deferred = akita_wide_zero();
    uint deferred_count = 0u;
    bool has_reduced_segment = false;
    for (uint chunk = chunk_start; chunk < chunk_end; ++chunk) {
        uint hot = hot_indices[source_chunk_base + chunk];
        if (hot == NONE_INDEX) {
            continue;
        }
        uint field_position = (chunk << log_onehot_k) + hot;
        uint ring_index = field_position >> log_ring_d;
        if (ring_index < block_ring_start
            || ring_index >= block_ring_start + positions_per_block) {
            continue;
        }
        uint position = ring_index - block_ring_start;
        uint shift = field_position & (ring_d - 1u);
        bool positive = coefficient >= shift;
        uint source_coefficient = (coefficient - shift) & (ring_d - 1u);
        uint column = position * (uint)params.num_digits_inner;
        uint matrix_index =
            ((row * active_a_cols + column) << log_ring_d)
            + source_coefficient;
        AkitaFp128 value = matrix[matrix_index];
        akita_wide_accumulate(deferred, value, positive);
        ++deferred_count;
        if (deferred_count == MAX_DEFERRED_ACCUMULATIONS) {
            AkitaFp128 segment = akita_reduce_wide(deferred);
            accumulator = has_reduced_segment
                ? akita_add(accumulator, segment)
                : segment;
            deferred = akita_wide_zero();
            deferred_count = 0u;
            has_reduced_segment = true;
        }
    }

    if (deferred_count != 0u) {
        AkitaFp128 segment = akita_reduce_wide(deferred);
        accumulator = has_reduced_segment
            ? akita_add(accumulator, segment)
            : segment;
    }

    uint output_index =
        ((source * num_blocks + block) * n_a + row) * ring_d
        + coefficient;
    output[output_index] = accumulator;
}

kernel void akita_onehot_commit_gather(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const ushort *hot_indices [[buffer(1)]],
    device AkitaFp128 *output [[buffer(2)]],
    constant OneHotCommitParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    ulong output_index = (ulong)gid;
    if (output_index >= params.output_coefficients) {
        return;
    }

    ulong outputs_per_source = params.num_blocks * params.n_a * params.ring_d;
    ulong source = output_index / outputs_per_source;
    ulong source_local = output_index - source * outputs_per_source;
    ulong block = source_local / (params.n_a * params.ring_d);
    ulong row_coefficient = source_local - block * params.n_a * params.ring_d;
    ulong row = row_coefficient / params.ring_d;
    ulong coefficient = row_coefficient - row * params.ring_d;

    ulong block_field_start = block * params.positions_per_block * params.ring_d;
    ulong block_field_end = min(
        block_field_start + params.positions_per_block * params.ring_d,
        params.total_field_elements);
    ulong chunk_start = block_field_start / params.onehot_k;
    ulong chunk_end = (block_field_end + params.onehot_k - 1ul) / params.onehot_k;

    AkitaFp128 accumulator = akita_zero();
    ulong source_chunk_base = source * params.chunks_per_source;
    ulong active_a_cols = params.positions_per_block * params.num_digits_inner;
    for (ulong chunk = chunk_start; chunk < chunk_end; ++chunk) {
        uint hot = hot_indices[source_chunk_base + chunk];
        if (hot == NONE_INDEX) {
            continue;
        }

        ulong field_position = chunk * params.onehot_k + (ulong)hot;
        ulong ring_index = field_position / params.ring_d;
        if (ring_index / params.positions_per_block != block) {
            continue;
        }
        ulong position = ring_index % params.positions_per_block;
        ulong shift = field_position % params.ring_d;
        bool positive = coefficient >= shift;
        ulong source_coefficient = positive
            ? coefficient - shift
            : params.ring_d + coefficient - shift;
        ulong column = position * params.num_digits_inner;
        ulong matrix_index =
            ((row * active_a_cols + column) * params.ring_d) + source_coefficient;
        AkitaFp128 value = matrix[matrix_index];
        accumulator = positive
            ? akita_add(accumulator, value)
            : akita_sub(accumulator, value);
    }
    output[output_index] = accumulator;
}
