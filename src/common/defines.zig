const std = @import("std");

/// Optimization levels in zigTensor. These determine the computation behavior
/// of autograd operator computation as well as how inputs and outputs of
/// operators are cast.
///
/// Operator precision roughly follows those found in NVIDIA Apex:
/// - https://bit.ly/33UpSWp
/// - https://bit.ly/30Zv2OS
/// - https://bit.ly/310k8Z6
pub const OptimLevel = enum(u8) {
    /// All operations occur in default (f32 or f64) precision.
    DEFAULT = 0,
    /// Operations that perform reduction accumulation, including layer/batch
    /// normalization are performed in f32 - all other operations are in fp16.
    /// To be used in a standard mixed-precision training setup.
    O1 = 1,
    /// Only batch and layer normalization occur in f32 - all other operations
    /// occur in f16.
    O2 = 2,
    /// All operations that support it use fp16.
    O3 = 3,
};

var optim_mode_singleton = OptimMode{};

/// OptimMode used for storing the current optimization level (`OptimLevel`) for
/// zigTensor as a global singleton.
pub const OptimMode = struct {
    optim_level: OptimLevel = .DEFAULT,

    const kStringToOptimLevel = std.ComptimeStringMap(OptimLevel, .{
        .{ "DEFAULT", .DEFAULT },
        .{ "O1", .O1 },
        .{ "O2", .O2 },
        .{ "O3", .O3 },
    });

    /// Returns the OptimMode singleton.
    pub fn get() OptimMode {
        return optim_mode_singleton;
    }

    /// Gets the current optimization level. Not thread safe.
    pub fn getOptimLevel(self: *const OptimMode) OptimLevel {
        return self.optim_level;
    }

    pub fn setOptimLevel(self: *OptimMode, level: OptimLevel) void {
        self.optim_level = level;
    }

    pub fn toOptimLevel(self: *const OptimMode, in: []const u8) !OptimLevel {
        var l: ?OptimLevel = self.kStringToOptimLevel.get(in);
        if (l == null) {
            std.debug.print("OptimMode::toOptimLevel - no matching optim level for given string.\n", .{});
            return error.FailedToOptimLevel;
        }
    }
};
