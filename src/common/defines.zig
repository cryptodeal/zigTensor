const std = @import("std");

/// Reduction mode to used for CrossEntropy, AdaptiveSoftMax, etc...
pub const ReduceMode = enum(u8) {
    None = 0,
    Mean = 1,
    Sum = 2,
};

/// Pooling method to be used
pub const PoolingMode = enum(u8) {
    /// Use maximum value inside the pooling window
    Max = 0,
    /// Use average value (including padding) inside the pooling window
    AvgIncludePadding = 1,
    /// Use average value (excluding padding) inside the pooling window
    AvgExcludePadding = 2,
};

/// RNN network type
pub const RnnMode = enum(u8) {
    Relu = 0,
    Tanh = 1,
    Lstm = 2,
    Gru = 3,
};

pub const PaddingMode = enum(i8) {
    /// Use smallest possible padding such that out_size = ceil(in_size/stride)
    Same = -1,
};

pub const DistributedBackend = enum(u8) {
    /// https://github.com/facebookincubator/gloo
    Gloo = 0,
    /// https://developer.nvidia.com/nccl
    Nccl = 1,
    Stub = 2,
};

pub const DistributedInit = enum(u8) {
    Mpi = 0,
    FileSystem = 1,
};

pub const DistributedConstants = struct {
    pub const kMaxDevicePerNode: []const u8 = "MAX_DEVICE_PER_NODE";
    pub const kFilePath: []const u8 = "FILE_PATH";
    pub const kCoalesceCacheSize: usize = 20 << 20; // 20 MB
};

pub const kDynamicBenchmarkDefaultCount: usize = 10;
pub const kAmpMinimumScaleFactorValue: f64 = 1e-4;

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
    Default = 0,
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
    optim_level: OptimLevel = .Default,

    const kStringToOptimLevel = std.ComptimeStringMap(OptimLevel, .{
        .{ "Default", .Default },
        .{ "O1", .O1 },
        .{ "O2", .O2 },
        .{ "O3", .O3 },
    });

    /// Returns the OptimMode singleton.
    pub fn get() *OptimMode {
        return &optim_mode_singleton;
    }

    /// Gets the current optimization level. Not thread safe.
    pub fn getOptimLevel(self: *const OptimMode) OptimLevel {
        return self.optim_level;
    }

    /// Sets the current optimization level. Not thread safe.
    pub fn setOptimLevel(self: *OptimMode, level: OptimLevel) void {
        self.optim_level = level;
    }

    /// Converts OptimLevel
    pub fn toOptimLevel(self: *const OptimMode, in: []const u8) !OptimLevel {
        var l: ?OptimLevel = self.kStringToOptimLevel.get(in);
        if (l == null) {
            std.debug.print("OptimMode::toOptimLevel - no matching optim level for given string.\n", .{});
            return error.FailedToOptimLevel;
        }
    }
};
