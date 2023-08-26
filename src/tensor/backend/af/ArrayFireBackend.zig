const std = @import("std");
const af = @import("../../../backends/ArrayFire.zig");
const base = @import("../../TensorBase.zig");
const zt_types = @import("../../Types.zig");
const build_options = @import("build_options");
const rt_stream = @import("../../../runtime/Stream.zig");
const rt_device_manager = @import("../../../runtime/DeviceManager.zig");
const zigrc = @import("zigrc");
const rt_device_type = @import("../../../runtime/DeviceType.zig");
const getDeviceTypes = rt_device_type.getDeviceTypes;

const ArrayFireCPUStream = @import("ArrayFireCPUStream.zig").ArrayFireCPUStream;
const Arc = zigrc.Arc;
const toArray = @import("ArrayFireTensor.zig").toArray;
const ZT_BACKEND_CUDA = build_options.ZT_BACKEND_CUDA;
const ZT_ARRAYFIRE_USE_CUDA = build_options.ZT_ARRAYFIRE_USE_CUDA;
const ZT_ARRAYFIRE_USE_CPU = build_options.ZT_ARRAYFIRE_USE_CPU;
const ZT_BACKEND_CPU = build_options.ZT_BACKEND_CPU;
const TensorBackendType = base.TensorBackendType;
const Stream = rt_stream.Stream;
const DType = zt_types.DType;
const DeviceManager = rt_device_manager.DeviceManager;
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

fn getOrWrapAfDeviceStream(allocator: std.mem.Allocator, afId: c_int, nativeId: c_int, afIdToStream: *std.AutoHashMap(c_int, Arc(Stream))) !*Stream {
    _ = nativeId;
    var iter = afIdToStream.get(afId);
    if (iter != null) {
        return iter.?.value;
    }

    if (ZT_ARRAYFIRE_USE_CPU) {
        var stream = try ArrayFireCPUStream.create(allocator);
        try afIdToStream.put(afId, stream);
        return stream.value;
    }
    //  else if (ZT_ARRAYFIRE_USE_CUDA) {
    //      TODO: add CUDA support
    //  }
    else {
        std.log.err("ArrayFireBackend was not compiled with support for CPU or GPU\n", .{});
        return error.ArrayFireBackendNoDeviceType;
    }
}

fn setActiveCallback(data: ?*anyopaque, id: c_int) !void {
    var self: *ArrayFireBackend = @ptrCast(@alignCast(data.?));
    const afId = self.nativeIdToId_.get(id).?;
    try AF_CHECK(af.af_set_device(afId), @src());
    // this is the latest point we can lazily wrap the AF stream, which may get
    // lazily intialized anytime in AF internally, e.g., via tensor computation.
    _ = try getOrWrapAfDeviceStream(self.allocator, afId, id, self.afIdToStream_.value);
    std.log.err("executed `setActiveCallback` after activating device\n", .{});
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
    afIdToStream_: Arc(std.AutoHashMap(c_int, Arc(Stream))),

    pub fn init(allocator: std.mem.Allocator) !*ArrayFireBackend {
        var self: *ArrayFireBackend = try allocator.create(ArrayFireBackend);
        var map = std.AutoHashMap(c_int, Arc(Stream)).init(allocator);
        self.* = .{
            .allocator = allocator,
            .nativeIdToId_ = std.AutoHashMap(c_int, c_int).init(allocator),
            .idToNativeId_ = std.AutoHashMap(c_int, c_int).init(allocator),
            .afIdToStream_ = try Arc(std.AutoHashMap(c_int, Arc(Stream))).init(allocator, map),
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

        var mgr = try DeviceManager.getInstance(allocator);

        // This callback ensures consistency of AF internal state on active device.
        // Capturing by value to avoid destructor race hazard for static objects.

        if (ZT_ARRAYFIRE_USE_CPU) {
            var device = try mgr.getActiveDevice(.x64);
            try device.addSetActiveCallback(setActiveCallback, self);
        } else if (ZT_ARRAYFIRE_USE_CUDA) {
            // TODO: add CUDA support
        }

        // Active device is never set explicitly, so we must wrap its stream eagerly.

        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *ArrayFireBackend) void {
        self.idToNativeId_.deinit();
        self.nativeIdToId_.deinit();
        var map = self.afIdToStream_.tryUnwrap().?;
        var iterator = map.valueIterator();
        while (iterator.next()) |stream| {
            var s = stream.tryUnwrap();
            if (s != null) s.?.deinit();
        }
        map.deinit();
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

    // access and release singleton here so test doesn't leak
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    var device_types = getDeviceTypes();
    var iterator = device_types.iterator();
    while (iterator.next()) |d_type| {
        if (mgr.isDeviceTypeAvailable(d_type)) {
            var devices = try mgr.getDevicesOfType(allocator, d_type);
            defer allocator.free(devices);
            for (devices) |dev| {
                try dev.setActive();
            }
        }
    }

    try std.testing.expect(try backend.supportsDataType(.f16));
}
