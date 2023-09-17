const std = @import("std");

const Tensor = @import("TensorBase.zig").Tensor;
const defaultTensorBackend = @import("DefaultTensorType.zig").defaultTensorBackend;
const Shape = @import("Shape.zig").Shape;
const DType = @import("Types.zig").DType;

pub fn setSeed(allocator: std.mem.Allocator, seed: u64) !void {
    var backend = try defaultTensorBackend(allocator);
    try backend.setSeed(allocator, seed);
}

pub fn randn(allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) !Tensor {
    var backend = try defaultTensorBackend(allocator);
    return backend.randn(allocator, shape, dtype);
}

pub fn rand(allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) !Tensor {
    var backend = try defaultTensorBackend(allocator);
    return backend.rand(allocator, shape, dtype);
}
