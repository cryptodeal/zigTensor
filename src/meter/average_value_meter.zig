const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;

/// An implementation of average value meter, which measures the mean and
/// variance of a sequence of values.
pub const AverageValueMeter = struct {
    cur_mean: f64 = 0,
    cur_mean_squared_sum: f64 = 0,
    cur_weight_sum: f64 = 0,
    cur_weight_sum_squared: f64 = 0,

    /// Sets all the counters to 0.
    pub fn reset(self: *AverageValueMeter) void {
        self.cur_mean = 0;
        self.cur_mean_squared_sum = 0;
        self.cur_weight_sum = 0;
        self.cur_weight_sum_squared = 0;
    }

    /// Updates counters with the given value `val` with weight `w`.
    pub fn add(self: *AverageValueMeter, val: f64, w: f64) void {
        self.cur_weight_sum += w;
        self.cur_weight_sum_squared += w * w;

        if (self.cur_weight_sum == 0) {
            return;
        }

        self.cur_mean = self.cur_mean + w * (val - self.cur_mean) / self.cur_weight_sum;
        self.cur_mean_squared_sum = self.cur_mean_squared_sum + w * (val * val - self.cur_mean_squared_sum) / self.cur_weight_sum;
    }

    /// Updates counters with all values in `vals` with equal weights.
    pub fn addTensor(self: *AverageValueMeter, allocator: std.mem.Allocator, vals: Tensor) !void {
        const w: f64 = @as(f64, @floatFromInt(try vals.elements(allocator)));
        self.cur_weight_sum += w;
        self.cur_weight_sum_squared += w;

        if (self.cur_weight_sum == 0) {
            return;
        }

        var tmp1 = try zt.tensor.sum(allocator, vals, &.{}, false);
        self.cur_mean = self.cur_mean + (try tmp1.asScalar(allocator, f64) - w * self.cur_mean) / self.cur_weight_sum;
        tmp1.deinit();

        tmp1 = try zt.tensor.mul(allocator, Tensor, vals, Tensor, vals);
        var tmp2 = try zt.tensor.sum(allocator, tmp1, &.{}, false);
        defer tmp2.deinit();
        tmp1.deinit();
        self.cur_mean_squared_sum = self.cur_mean_squared_sum + (try tmp2.asScalar(allocator, f64) - w * self.cur_mean_squared_sum) / self.cur_weight_sum;
    }

    /// Returns an array of four values: unbiased mean, unbiased variance,
    /// weight sum, and weight sum squared.
    pub fn value(self: *AverageValueMeter) [4]f64 {
        const variance = (self.cur_mean_squared_sum - self.cur_mean * self.cur_mean) / (1 - self.cur_weight_sum_squared / (self.cur_weight_sum * self.cur_weight_sum));
        return [4]f64{ self.cur_mean, variance, self.cur_weight_sum, self.cur_weight_sum_squared };
    }
};

test "MeterTest -> AverageValueMeter" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var meter: AverageValueMeter = .{};
    meter.add(1, 0);
    meter.add(2, 1);
    meter.add(3, 1);
    meter.add(4, 1);
    var val = meter.value();
    try std.testing.expectEqual(@as(f64, 3), val[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 1), val[1], 1e-10);
    try std.testing.expectEqual(@as(f64, 3), val[2]);
    const a = try Tensor.fromSlice(allocator, &.{3}, f32, &.{ 2, 3, 4 }, .f32);
    defer a.deinit();
    try meter.addTensor(allocator, a);
    val = meter.value();
    try std.testing.expectEqual(@as(f64, 3), val[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), val[1], 1e-10);
    try std.testing.expectEqual(@as(f64, 6), val[2]);
}
