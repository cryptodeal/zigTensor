const std = @import("std");
const af = @import("../../../bindings/af/arrayfire.zig");
const zt_idx = @import("../../index.zig");
const assert = std.debug.assert;

const IndexType = zt_idx.IndexType;

pub fn condenseIndices(
    allocator: std.mem.Allocator,
    arr: *af.Array,
    keep_dims: bool,
    idx_types: ?[]IndexType,
    is_flat: bool,
) !struct { arr: *af.Array, modified: bool } {
    // Fast path - return the Array as is if keepDims - don't consolidate
    if (keep_dims) {
        return .{ .arr = arr, .modified = false };
    }
    // Fast path - Array has zero elements or a dim of size zero
    if (try arr.getElements() == 0) {
        return .{ .arr = arr, .modified = false };
    }

    const dims = try arr.getDims();
    var newDims = af.Dim4{};
    var newDimIdx: usize = 0;
    for (0..@intCast(af.AF_MAX_DIMS)) |i| {
        // If we're doing an index op (indexTypes is non-empty), then only collapse
        // the dimension if it contains an index literal and we aren't doing flat
        // indexing (which collapses all dims)
        if (dims.dims[i] == 1 and idx_types != null and idx_types.?.len > i and idx_types.?[i] != .Literal and !is_flat) {
            newDims.dims[newDimIdx] = 1;
            newDimIdx += 1;
        } else if (dims.dims[i] != 1) {
            // found a non-1 dim size - populate newDims.
            newDims.dims[newDimIdx] = dims.dims[i];
            newDimIdx += 1;
        }
    }

    // Only change dims if condensing is possible
    if (!std.mem.eql(c_longlong, &newDims.dims, &dims.dims)) {
        return .{ .arr = try arr.moddims(allocator, @intCast(newDims.ndims()), newDims), .modified = true };
    } else {
        return .{ .arr = arr, .modified = false };
    }
}

pub fn gForDim(indices: []af.af_index_t) i32 {
    for (0..indices.len) |i| {
        if (indices[i].isBatch) return @intCast(i);
    }
    return -1;
}

fn hasEnd(seq: *const af.af_seq) bool {
    return (seq.begin <= -1 or seq.end <= -1);
}

fn isSpan(seq: *const af.af_seq) bool {
    return (seq.step == 0 and seq.begin == 1 and seq.end == 1);
}

fn seqElements(seq: *const af.af_seq) usize {
    var out: usize = 0;
    if (seq.step > std.math.floatMin(f64)) {
        out = @intFromFloat(((seq.end - seq.begin) / @abs(seq.step)) + 1);
    } else if (seq.step < -(std.math.floatMin(f64))) {
        out = @intFromFloat(((seq.begin - seq.end) / @abs(seq.step)) + 1);
    } else {
        out = std.math.maxInt(usize);
    }
    return out;
}

fn calcDim(seq: *const af.af_seq, parent_dim: af.dim_t) !af.dim_t {
    var out_dim: af.dim_t = 1;
    if (isSpan(seq)) {
        out_dim = parent_dim;
    } else if (hasEnd(seq)) {
        var tmp: af.af_seq = .{ .begin = seq.begin, .end = seq.end, .step = seq.step };
        if (seq.begin < 0) tmp.begin += @floatFromInt(parent_dim);
        if (seq.end < 0) tmp.end += @floatFromInt(parent_dim);
        out_dim = @intCast(seqElements(&tmp));
    } else {
        // TODO: if (!(seq.begin >= -(std.math.floatMin(f64)) and seq.begin < @as(f64, @floatFromInt(parent_dim)))) {
        // Throw error??
        // }
        // TODO: if (!(seq.end < @as(f64, @floatFromInt(parent_dim)))) {
        // Throw error??
        // }
        out_dim = @intCast(seqElements(seq));
    }
    return out_dim;
}

pub fn seqToDims(indices: []af.af_index_t, parent_dims: af.Dim4, reorder: bool) !af.Dim4 {
    var odims = af.Dim4{};
    for (0..@intCast(af.AF_MAX_DIMS)) |i| {
        if (indices[i].isSeq) {
            odims.dims[i] = try calcDim(&indices[i].idx.seq, parent_dims.dims[i]);
        } else {
            var elems: af.dim_t = 0;
            var af_type: af.af_dtype = undefined;
            try af.AF_CHECK(af.af_get_type(&af_type, indices[i].idx.arr), @src());
            if (af_type == af.b8) {
                var tmp_arr: af.af_array = undefined;
                try af.AF_CHECK(af.af_where(&tmp_arr, indices[i].idx.arr), @src());
                try af.AF_CHECK(af.af_get_elements(&elems, tmp_arr), @src());
                try af.AF_CHECK(af.af_release_array(tmp_arr), @src());
            } else {
                try af.AF_CHECK(af.af_get_elements(&elems, indices[i].idx.arr), @src());
            }
            odims.dims[i] = elems;
        }
    }

    // Change the dimensions if inside GFOR
    if (reorder) {
        for (0..@intCast(af.AF_MAX_DIMS)) |i| {
            if (indices[i].isBatch) {
                var tmp = odims.dims[i];
                odims.dims[i] = odims.dims[3];
                odims.dims[3] = tmp;
                break;
            }
        }
    }
    return odims;
}

pub fn gForReorder(in: af.af_array, dim: c_uint) !af.af_array {
    if (dim > 3) {
        std.log.err("GFor: Dimension is invalid", .{});
        return error.ArrayFireInvalidDimError;
    }
    var order = [@intCast(af.AF_MAX_DIMS)]c_uint{ 0, 1, 2, dim };
    order[@intCast(dim)] = 3;
    var out: af.af_array = undefined;
    try af.AF_CHECK(af.af_reorder(&out, in, order[0], order[1], order[2], order[3]), @src());
    return out;
}
