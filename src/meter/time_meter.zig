const std = @import("std");
const zt = @import("../zt.zig");

const Timer = std.time.Timer;
const Tensor = zt.tensor.Tensor;

/// An implementation of timer, which measures the wall clock time.
pub const TimeMeter = struct {
    timer: Timer = undefined,
    cur_value: f64 = 0,
    cur_n: i64 = 0,
    is_stopped: bool = true,
    use_unit: bool = false,

    pub fn init(unit: bool) TimeMeter {
        return .{ .use_unit = unit };
    }

    pub fn reset(self: *TimeMeter) void {
        self.cur_value = 0;
        self.cur_n = 0;
        self.is_stopped = true;
    }

    pub fn set(self: *TimeMeter, val: f64, num: i64) !void {
        self.cur_value = val;
        self.cur_n = num;
        self.timer = try Timer.start();
    }

    pub fn value(self: *TimeMeter) !f64 {
        var val = self.cur_value;
        if (!self.is_stopped) {
            // duration in seconds
            const duration: f64 = @as(f64, @floatFromInt(self.timer.read())) / @as(f64, std.time.ns_per_s);
            val += duration;
        }
        if (self.use_unit) {
            val = if (self.cur_n > 0) val / @as(f64, self.cur_n) else 0;
        }
        return val;
    }

    pub fn stop(self: *TimeMeter) void {
        if (self.is_stopped) {
            return;
        }
        // duration in seconds
        const duration: f64 = @as(f64, @floatFromInt(self.timer.read())) / @as(f64, std.time.ns_per_s);
        self.cur_value += duration;
        self.is_stopped = true;
    }

    pub fn @"resume"(self: *TimeMeter) !void {
        if (!self.is_stopped) {
            return;
        }
        self.timer = try Timer.start();
        self.is_stopped = false;
    }

    pub fn incUnit(self: *TimeMeter, num: i64) void {
        self.cur_n += num;
    }

    pub fn stopAndIncUnit(self: *TimeMeter, num: i64) void {
        self.stop();
        self.incUnit(num);
    }
};
