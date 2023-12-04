const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;

/// An implementation of mean square error meter, which measures
/// the mean square error between targets and predictions made
/// by the model.
pub const MSEMeter = struct {
    cur_value: f64 = 0,
    cur_n: i64 = 0,

    /// Sets all the counters to 0.
    pub fn reset(self: *MSEMeter) void {
        self.cur_value = 0;
        self.cur_n = 0;
    }

    /// Computes mean square error between two tensors `output` and
    /// `target` and updates the counters. Note that the shape of the
    /// two input tensors must be identical.
    pub fn add(self: *MSEMeter, allocator: std.mem.Allocator, output: Tensor, target: Tensor) !void {
        if (try output.ndim(allocator) != target.ndim(allocator)) {
            std.debug.print("dimension mismatch in MSEMeter\n", .{});
            return error.DimsMismatch;
        }
        self.cur_n += 1;
        var tmp1 = try zt.tensor.sub(allocator, Tensor, output, Tensor, target);
        defer tmp1.deinit();
        try tmp1.inPlaceMul(allocator, Tensor, tmp1);
        try tmp1.inPlaceAdd(allocator, f64, self.cur_value * @as(f64, @floatFromInt(self.cur_n - 1)));
        self.cur_value = try tmp1.asScalar(allocator, f64) / @as(f64, @floatFromInt(self.cur_n));
    }

    /// Returns a single value of mean square error.
    pub fn value(self: *const MSEMeter) f64 {
        return self.cur_value;
    }
};
