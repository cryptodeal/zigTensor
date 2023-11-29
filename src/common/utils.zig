const std = @import("std");
const zt = @import("../zt.zig");

pub fn f16Supported(allocator: std.mem.Allocator) !bool {
    return (try zt.tensor.defaultTensorBackend(allocator)).isDataTypeSupported(.f16);
}

pub fn f64Supported(allocator: std.mem.Allocator) !bool {
    return (try zt.tensor.defaultTensorBackend(allocator)).isDataTypeSupported(.f64);
}
