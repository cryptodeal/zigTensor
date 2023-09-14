const std = @import("std");
const af = @import("../../../bindings/af/ArrayFire.zig");
const zt_idx = @import("../../Index.zig");

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
        return .{ .arr = try arr.modDims(allocator, @intCast(newDims.ndims()), dims), .modified = true };
    } else {
        return .{ .arr = arr, .modified = false };
    }
}
