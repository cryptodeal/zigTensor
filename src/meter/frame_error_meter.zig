const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;

/// An implementation of frame error meter, which measures the frame-level or
/// element-level mismatch between targets and predictions made by the model.
pub const FrameErrorMeter = struct {
    n: i64 = 0,
    sum: i64 = 0,
    accuracy: bool = false,

    /// Flag `accuracy` indicates if the meter computes and returns accuracy
    /// or error rate instead.
    pub fn init(accuracy: bool) FrameErrorMeter {
        return .{ .accuracy = accuracy };
    }

    /// Sets all the counters to 0.
    pub fn reset(self: *FrameErrorMeter) void {
        self.n = 0;
        self.sum = 0;
    }

    /// Computes frame-level mismatch between two tensors `output` and `target`
    /// and updates the counters. Note that the shape of the two input tensors
    /// should be identical.
    pub fn add(self: *FrameErrorMeter, allocator: std.mem.Allocator, output: Tensor, target: Tensor) !void {
        if (!zt.tensor.shape.eql(try output.shape(allocator), try target.shape(allocator))) {
            std.debug.print("dimension mismatch in FrameErrorMeter\n", .{});
            return error.DimsMismatch;
        }
        if (try target.ndim(allocator) != 1) {
            std.debug.print("output/target must be 1-dimensional for FrameErrorMeter\n", .{});
            return error.Exceeds1Dim;
        }
        var tmp1 = try zt.tensor.neq(allocator, Tensor, output, Tensor, target);
        defer tmp1.deinit();
        var tmp2 = try zt.tensor.countNonzero(allocator, tmp1, &.{}, false);
        defer tmp2.deinit();
        self.sum += try tmp2.scalar(allocator, u32);
        self.n += try target.dim(allocator, 0);
    }

    /// Returns a single value in percentage. If `accuracy` is `True`, the value
    /// returned is accuracy, error otherwise.
    pub fn value(self: *const FrameErrorMeter) f64 {
        const err: f64 = if (self.n > 0) @as(f64, @floatFromInt(self.sum * 100)) / @as(f64, @floatFromInt(self.n)) else 0;
        return if (self.accuracy) 100 - err else err;
    }
};

test "MeterTest -> FrameErrorMeter" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var meter: FrameErrorMeter = .{};
    const a_data: []const i32 = &.{ 1, 2, 3, 4, 5 };
    const b_data: []const i32 = &.{ 1, 1, 3, 3, 5, 6 };
    const a = try Tensor.fromSlice(allocator, &.{5}, i32, a_data, .s32);
    defer a.deinit();
    const b = try Tensor.fromSlice(allocator, &.{5}, i32, b_data, .s32);
    defer b.deinit();
    try meter.add(allocator, a, b);
    try std.testing.expectEqual(@as(f64, 40), meter.value()); // 2 / 5
    const a2 = try Tensor.fromSlice(allocator, &.{4}, i32, a_data[1..], .s32);
    defer a2.deinit();
    const b2 = try Tensor.fromSlice(allocator, &.{4}, i32, b_data[2..], .s32);
    defer b2.deinit();
    try meter.add(allocator, a2, b2);
    try std.testing.expect(@abs(55.5555555 - meter.value()) < 1e-5); // 2 + 3 / 5 + 4

}
