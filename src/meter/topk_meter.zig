const std = @import("std");
const zt = @import("../zt.zig");

const Timer = std.time.Timer;
const Tensor = zt.tensor.Tensor;

/// TopKMeter computes the accuracy of the model outputs predicting the target
/// label in the top k predictions.
pub const TopKMeter = struct {
    k: u32,
    correct: i32 = 0,
    n: i32 = 0,

    pub fn init(k: i32) TopKMeter {
        return .{ .k = k };
    }

    pub fn add(self: *TopKMeter, allocator: std.mem.Allocator, output: Tensor, target: Tensor) !void {
        if (try output.dim(allocator, 1) != try target.dim(allocator, 0)) {
            std.debug.print("dimension mismatch in TopKMeter\n", .{});
            return error.DimsMismatch;
        }
        if (try target.ndim(allocator) != 1) {
            std.debug.print("output/target must be 1-dimensional for TopKMeter\n", .{});
            return error.Required1D;
        }
        const max_vals = try Tensor.initEmpty(allocator);
        defer max_vals.deinit();
        const max_ids = try Tensor.initEmpty(allocator);
        defer max_ids.deinit();
        try zt.tensor.topk(allocator, max_vals, max_ids, output, @intCast(self.k), 0, .Descending);
        const tmp = try zt.tensor.reshape(allocator, target, &.{ 1, try target.dim(allocator, 0), 1, 1 });
        defer tmp.deinit();
        const match = try zt.tensor.eq(allocator, Tensor, max_ids, Tensor, tmp);
        defer match.deinit();
        const correct = try zt.tensor.any(allocator, match, &.{0}, false);
        defer correct.deinit();
        const nonzero = try zt.tensor.countNonzero(allocator, correct, &.{}, false);
        defer nonzero.deinit();
        self.correct += try nonzero.asScalar(allocator, i32);
        const batch_size = try target.dim(0);
        self.n += @intCast(batch_size);
    }

    pub fn reset(self: *TopKMeter) void {
        self.correct = 0;
        self.n = 0;
    }

    pub fn value(self: *const TopKMeter) [2]i32 {
        return [2]i32{ self.correct, self.n };
    }

    pub fn set(self: *TopKMeter, correct: i32, n: i32) void {
        self.correct = correct;
        self.n = n;
    }
};
