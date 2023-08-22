const std = @import("std");
const af = @import("../../../backends/ArrayFire.zig");
const base = @import("../../TensorBase.zig");
const zt_types = @import("../../Types.zig");
const build_options = @import("build_options");

const toArray = @import("ArrayFireTensor.zig").toArray;
const ZT_BACKEND_CUDA = build_options.ZT_BACKEND_CUDA;
const ZT_ARRAYFIRE_USE_CUDA = build_options.ZT_ARRAYFIRE_USE_CUDA;
const ZT_BACKEND_CPU = build_options.ZT_BACKEND_CPU;
const TensorBackendType = base.TensorBackendType;
const DType = zt_types.DType;
const Tensor = base.Tensor;

const AF_CHECK = @import("Utils.zig").AF_CHECK;

var memoryInitFlag = std.once(init);

fn init() void {
    // TODO: remove this temporary workaround for TextDatasetTest crash on CPU
    // backend when tearing down the test environment. This is possibly due to
    // AF race conditions when tearing down our custom memory manager.
    // TODO: remove this temporary workaround for crashes when using custom
    // opencl kernels.
    if (ZT_BACKEND_CUDA) {
        // TODO: install memory manager
    }
}

// TODO: add ArrayFire CUDA support
/// A tensor backend implementation of the ArrayFire tensor library.
///
/// Since ArrayFire has an internal DeviceManager singleton to manage
/// its global state, nothing is stored here as those internals are
/// opaquely handled. ArrayFireBackend simply dispatches operations
/// on global tensor functions to their ArrayFire counterparts.
pub const ArrayFireBackend = struct {
    allocator: std.mem.Allocator,
    nativeIdToId_: std.AutoHashMap(c_int, c_int),
    idToNativeId_: std.AutoHashMap(c_int, c_int),

    pub fn init(allocator: std.mem.Allocator) !*ArrayFireBackend {
        var self = try allocator.create(ArrayFireBackend);
        self.* = .{
            .allocator = allocator,
            .nativeIdToId_ = std.AutoHashMap(c_int, c_int).init(allocator),
            .idToNativeId_ = std.AutoHashMap(c_int, c_int).init(allocator),
        };
        try AF_CHECK(af.af_init(), @src());
        memoryInitFlag.call();

        // segfaults here
        var device_count: c_int = undefined;
        try AF_CHECK(af.af_get_device_count(&device_count), @src());
        for (0..@intCast(device_count)) |i| {
            const id: c_int = @intCast(i);
            // TODO investigate how OpenCL fits into this.
            var native_id: c_int = id;
            if (ZT_ARRAYFIRE_USE_CUDA) {
                // TODO: native_id = try AF_CHECK(af.afcu_get_native_id(&native_id, id));
            }
            try self.nativeIdToId_.put(native_id, id);
            try self.idToNativeId_.put(id, native_id);
        }

        // TODO: finish implementation
        std.log.err("Returning ArrayFireBackendImpl\n", .{});
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *ArrayFireBackend) void {
        self.idToNativeId_.deinit();
        self.nativeIdToId_.deinit();
        self.allocator.destroy(self);
    }

    pub fn getInstance(self: *ArrayFireBackend) ArrayFireBackend {
        const instance = self.*;
        return instance;
    }

    pub fn backendType(_: *ArrayFireBackend) TensorBackendType {
        return .ArrayFire;
    }

    // -------------------------- Compute Functions --------------------------

    pub fn eval(tensor: *Tensor) !void {
        try AF_CHECK(af.af_eval(try toArray(tensor)), @src());
    }

    // TODO: pub fn getStreamOfArray(arr: af.af_array) *Stream {}

    pub fn supportsDataType(_: *ArrayFireBackend, dtype: DType) !bool {
        return switch (dtype) {
            .f16 => {
                var device: c_int = undefined;
                try AF_CHECK(af.af_get_device(&device), @src());
                var half_support: bool = undefined;
                try AF_CHECK(af.af_get_half_support(&half_support, device), @src());
                return half_support and !ZT_BACKEND_CPU;
            },
            else => true,
        };
    }

    // TODO: pub fn getMemMgrInfo()

    // TODO: pub fn setMemMgrLogStream()

    // TODO: pub fn setMemMgrLoggingEnabled()

    // TODO: pub fn setMemMgrFlushInterval()

    // -------------------------- Rand Functions --------------------------
    pub fn setSeed(seed: u64) void {
        af.af_set_seed(@intCast(seed));
    }

    // TODO: pub fn randn()

    // TODO: pub fn rand()

    // --------------------------- Tensor Operators ---------------------------

    // use comptime type param for `template` semantics
    // TODO: pub fn fromScalar()

    // TODO: pub fn full()

    // TODO: pub fn identity()

    // TODO: pub fn arange()

    // TODO: pub fn iota()

    // TODO: pub fn where()

    // TODO: pub fn topk()

    // TODO: pub fn sort()

    // TODO: pub fn sort2()

    // TODO: pub fn argsort()

};

test "ArrayFireBackend supportsDataType" {
    var allocator = std.testing.allocator;
    var backend = try ArrayFireBackend.init(allocator);
    defer backend.deinit();

    try std.testing.expect(try backend.supportsDataType(.f16));
}
