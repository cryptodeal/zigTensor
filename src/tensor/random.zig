const std = @import("std");
const zt = @import("../zt.zig");

const defaultTensorBackend = zt.tensor.defaultTensorBackend;
const DType = zt.tensor.DType;
const Shape = zt.tensor.shape.Shape;
const Tensor = zt.tensor.Tensor;

pub fn setSeed(allocator: std.mem.Allocator, seed: u64) !void {
    var backend = try defaultTensorBackend(allocator);
    try backend.setSeed(seed);
}

pub fn randn(allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
    var backend = try defaultTensorBackend(allocator);
    return backend.randn(allocator, shape, dtype);
}

pub fn rand(allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
    var backend = try defaultTensorBackend(allocator);
    return backend.rand(allocator, shape, dtype);
}
