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
