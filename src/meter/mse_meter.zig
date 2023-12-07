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
        if (try output.ndim(allocator) != try target.ndim(allocator)) {
            std.debug.print("dimension mismatch in MSEMeter\n", .{});
            return error.DimsMismatch;
        }
        self.cur_n += 1;
        var tmp = try zt.tensor.sub(allocator, Tensor, output, Tensor, target);
        defer tmp.deinit();
        try tmp.inPlaceMul(allocator, Tensor, tmp);
        var res = try zt.tensor.sum(allocator, tmp, &.{}, false);
        defer res.deinit();
        try res.inPlaceAdd(allocator, f64, self.cur_value * @as(f64, @floatFromInt(self.cur_n - 1)));
        self.cur_value = try res.asScalar(allocator, f64) / @as(f64, @floatFromInt(self.cur_n));
    }

    /// Returns a single value of mean square error.
    pub fn value(self: *const MSEMeter) f64 {
        return self.cur_value;
    }
};

test "MeterTest -> MSEMeter" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();
    var meter: MSEMeter = .{};
    const a = try Tensor.fromSlice(allocator, &.{5}, i32, &.{ 1, 2, 3, 4, 5 }, .s32);
    defer a.deinit();
    const b = try Tensor.fromSlice(allocator, &.{5}, i32, &.{ 4, 5, 6, 7, 8 }, .s32);
    defer b.deinit();
    try meter.add(allocator, a, b);
    try std.testing.expectEqual(@as(f64, 45), meter.value());
}
