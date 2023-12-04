const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;

/// An implementation of count meter, which measures the total
/// value of each category.
pub fn CountMeter(comptime num: usize) type {
    return struct {
        const Self = @This();
        counts: [num]i64 = [_]i64{0} ** num,

        /// Adds value `val` to category `id`. Note that `id`
        /// should be in range [0, `num` - 1].
        pub fn add(self: *Self, id: usize, val: i64) void {
            self.counts[id] += val;
        }

        /// Returns an array of `num` values, representing the total
        /// value of each category.
        pub fn value(self: *const Self) [num]i64 {
            return self.counts;
        }

        /// Sets the value of each category to 0.
        pub fn reset(self: *Self) void {
            @memset(&self.counts, 0);
        }
    };
}
