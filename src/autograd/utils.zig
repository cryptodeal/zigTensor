const std = @import("std");
const zt = @import("../zt.zig");

pub fn allClose(allocator: std.mem.Allocator, a: *zt.autograd.Variable, b: *zt.autograd.Variable, abs_tolerance: f64) !bool {
    return zt.tensor.allClose(allocator, a.tensor(), b.tensor(), abs_tolerance);
}
