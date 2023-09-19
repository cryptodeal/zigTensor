const af = @import("../../../bindings/af/ArrayFire.zig");
const std = @import("std");
const af_tensor = @import("ArrayFireTensor.zig");

const Tensor = @import("../../TensorBase.zig").Tensor;
const TensorAdapterBase = @import("../../TensorAdapter.zig").TensorAdapterBase;
const ArrayFireTensor = af_tensor.ArrayFireTensor;
const toArray = af_tensor.toArray;

const condenseIndices = @import("Utils.zig").condenseIndices;

pub fn afReduceAxes(
    allocator: std.mem.Allocator,
    input: *af.Array,
    axes: std.ArrayList(i32),
    comptime T: type,
    func: T,
    keep_dims: bool,
) !*af.Array {
    var arr = input;
    for (axes.items, 0..) |dim, i| {
        var og_arr: *af.Array = arr;
        defer if (i != 0 and i != axes.items.len - 1) og_arr.deinit();
        arr = try func(allocator, arr, dim);
    }
    const res = try condenseIndices(allocator, arr, keep_dims, null, false);
    defer if (res.modified) arr.deinit();
    return res.arr;
}

pub fn getReducedNumDims(comptime T: type, in_size: T, axis_size: T, keep_dims: bool) T {
    if (keep_dims) {
        return in_size;
    } else {
        if (in_size < axis_size) {
            return 0;
        } else {
            return in_size - axis_size;
        }
    }
}

fn lessThan(_: void, a: i32, b: i32) bool {
    return a < b;
}

pub fn isAllAxisReduction(allocator: std.mem.Allocator, input: Tensor, axes: std.ArrayList(i32)) !bool {
    const inputNumDims = try input.ndim(allocator);
    if (inputNumDims == 0 or axes.items.len == 0) {
        return true;
    }
    if (inputNumDims != axes.items.len) {
        return false;
    }
    // Check that all dims are present
    var _axes = axes;
    std.mem.sort(i32, _axes.items, {}, lessThan);
    for (0.._axes.items.len) |i| {
        if (_axes.items[i] != @as(i32, @intCast(i))) {
            return false;
        }
    }
    return true;
}

pub const reduceFunc_t = *const fn (allocator: std.mem.Allocator, in: *const af.Array, dim: i32) anyerror!*af.Array;
