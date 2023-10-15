const std = @import("std");
const rc = @import("zigrc");
const rt_device_type = @import("device_type.zig");
const rt_stream = @import("stream.zig");
const rt_device_mgr = @import("device_manager.zig");

const assert = std.debug.assert;
const kX64DeviceId = rt_device_mgr.kX64DeviceId;
const Arc = rc.Arc;
const DeviceType = rt_device_type.DeviceType;
const getDeviceTypes = rt_device_type.getDeviceTypes;
const Stream = rt_stream.Stream;
const DeviceManager = rt_device_mgr.DeviceManager;

/// Throws an error and logs with a descriptive message
/// if the given types don't match
fn deviceImplCheck(comptime expect: type, comptime actual: type) !void {
    if (expect != actual) {
        std.debug.print("[zt.Device.impl] specified device type: [{any}] doesn't match actual device type: [{any}]\n", .{expect});
        return error.DeviceTypeMismatch;
    }
}

const CallbackCtx = struct {
    func: *const fn (data: ?*anyopaque, id: i32) anyerror!void,
    data: ?*anyopaque,
};

pub const DeviceErrors = error{DeviceMustOwnStream} || std.mem.Allocator.Error;

pub const Device = struct {
    const Self = @This();

    // The type erased pointer to the TensorBackend implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        setActiveImpl: *const fn (ctx: *anyopaque) void,
        getStreams: *const fn (ctx: *anyopaque) std.AutoHashMap(Arc(Stream), void),
        addStream: *const fn (ctx: *anyopaque, stream: Arc(Stream)) DeviceErrors!void,
        sync: *const fn (ctx: *anyopaque) anyerror!void,
        nativeId: *const fn (ctx: *anyopaque) i32,
        deviceType: *const fn (ctx: *anyopaque) DeviceType,
        setActive: *const fn (ctx: *anyopaque) anyerror!void,
        addSetActiveCallback: *const fn (ctx: *anyopaque, callback: *const fn (data: ?*anyopaque, id: i32) anyerror!void, data: ?*anyopaque) DeviceErrors!void,
        deinit: *const fn (ctx: *anyopaque) void,
    };

    /// Set this device as the active device, without worrying about the callbacks.
    pub fn setActiveImpl(ctx: *const Self) void {
        return ctx.vtable.setActiveImpl(ctx.ptr);
    }

    /// Returns an immutable list of all streams managed by this device.
    pub fn getStreams(ctx: *const Self) std.AutoHashMap(Arc(Stream), void) {
        return ctx.vtable.getStreams(ctx.ptr);
    }

    // Let this device manage given stream. Do nothing if it was already added.
    pub fn addStream(ctx: *const Self, stream: Arc(Stream)) !void {
        return ctx.vtable.addStream(ctx.ptr, stream);
    }

    /// Block calling thread and synchronize with regard to all streams on this device.
    pub fn sync(ctx: *const Self) !void {
        try ctx.vtable.sync(ctx.ptr);
    }

    /// Returns the native ID of this device (semantics are implementation-dependent).
    pub fn nativeId(ctx: *const Self) i32 {
        return ctx.vtable.nativeId(ctx.ptr);
    }

    /// Returns the type of this device.
    pub fn deviceType(ctx: *const Self) DeviceType {
        return ctx.vtable.deviceType(ctx.ptr);
    }

    /// Set this device as the active device and invokes any callbacks added.
    pub fn setActive(ctx: *const Self) !void {
        return ctx.vtable.setActive(ctx.ptr);
    }

    /// Lets this device keep track of the given callback (along with previously
    /// added ones), which will be invoked with the device's native ID after
    /// setting the device active.
    pub fn addSetActiveCallback(ctx: *const Self, callback: *const fn (data: ?*anyopaque, id: i32) anyerror!void, data: ?*anyopaque) DeviceErrors!void {
        return ctx.vtable.addSetActiveCallback(ctx.ptr, callback, data);
    }

    // Returns the underlying implementation of this device.
    //
    // Throws if the specified type does not match the actual
    // derived device type.
    pub fn getImpl(ctx: *const Self, comptime T: type) !*T {
        var actual_type = ctx.deviceType();
        if (T.type_ != actual_type) {
            std.log.debug("[zt.Device.getImpl] specified device type: [{s}] doesn't match actual device type: [{s}]\n", .{ @tagName(T.type_), @tagName(actual_type) });
            return error.FailedDeviceGetImpl;
        }
        return @ptrCast(@alignCast(ctx.ptr));
    }

    /// Frees all associated memory from the implementation.
    pub fn deinit(ctx: *const Self) void {
        return ctx.vtable.deinit(ctx.ptr);
    }

    pub fn init(device_impl: anytype) Self {
        const Ptr = @TypeOf(device_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn setActiveImpl(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.setActiveImpl();
            }

            fn getStreams(ctx: *anyopaque) std.AutoHashMap(Arc(Stream), void) {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getStreams();
            }

            fn addStream(ctx: *anyopaque, stream: Arc(Stream)) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.addStream(stream);
            }

            fn sync(ctx: *anyopaque) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.sync();
            }

            fn nativeId(ctx: *anyopaque) i32 {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.nativeId();
            }

            fn deviceType(ctx: *anyopaque) DeviceType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.deviceType();
            }

            fn setActive(ctx: *anyopaque) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.setActive();
            }

            fn addSetActiveCallback(ctx: *anyopaque, callback: *const fn (data: ?*anyopaque, id: i32) anyerror!void, data: ?*anyopaque) DeviceErrors!void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.addSetActiveCallback(callback, data);
            }

            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }
        };
        return .{
            .ptr = device_impl,
            .vtable = &.{
                .setActiveImpl = impl.setActiveImpl,
                .getStreams = impl.getStreams,
                .addStream = impl.addStream,
                .sync = impl.sync,
                .nativeId = impl.nativeId,
                .deviceType = impl.deviceType,
                .setActive = impl.setActive,
                .addSetActiveCallback = impl.addSetActiveCallback,
                .deinit = impl.deinit,
            },
        };
    }
};

pub const X64Device = struct {
    pub const type_: DeviceType = .x64;
    /// Tracks Streams from the Device.
    streams_: std.AutoHashMap(Arc(Stream), void),
    /// Used to update internal backend state for active device, thereby
    /// eliminating the `setActive --> AnyTensorBackendImpl` dependency(s).
    setActiveCallbacks_: std.ArrayList(CallbackCtx),

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*X64Device {
        var self = try allocator.create(X64Device);
        self.* = .{
            .allocator = allocator,
            .streams_ = std.AutoHashMap(Arc(Stream), void).init(allocator),
            .setActiveCallbacks_ = std.ArrayList(CallbackCtx).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *X64Device) void {
        // TODO: might need to free or decrement refcounts on streams
        self.streams_.deinit();
        self.setActiveCallbacks_.deinit();
        self.allocator.destroy(self);
    }

    pub fn setActiveImpl(_: *X64Device) void {} // no-op -- cpu is always active

    pub fn getStreams(self: *X64Device) std.AutoHashMap(Arc(Stream), void) {
        return self.streams_;
    }

    // TODO: finish implementing Stream
    pub fn addStream(self: *X64Device, stream: Arc(Stream)) DeviceErrors!void {
        if (@intFromPtr(self) != @intFromPtr((stream.value.device()).ptr)) {
            std.log.debug("[Device.addStream] Must add stream to owner device\n", .{});
            return DeviceErrors.DeviceMustOwnStream;
        }
        try self.streams_.put(stream, {});
    }

    pub fn sync(self: *X64Device) !void {
        self.setActiveImpl();
        var iterator = self.streams_.keyIterator();
        while (iterator.next()) |stream| {
            try stream.value.sync();
        }
    }

    pub fn nativeId(_: *X64Device) i32 {
        return kX64DeviceId;
    }

    pub fn deviceType(_: *X64Device) DeviceType {
        return X64Device.type_;
    }

    pub fn setActive(self: *X64Device) !void {
        self.setActiveImpl();
        for (self.setActiveCallbacks_.items) |callback| {
            try callback.func(callback.data, self.nativeId());
        }
    }

    pub fn addSetActiveCallback(self: *X64Device, callback: *const fn (data: ?*anyopaque, id: i32) anyerror!void, data: ?*anyopaque) DeviceErrors!void {
        try self.setActiveCallbacks_.append(.{ .func = callback, .data = data });
    }
};

test "Device deviceType" {
    const allocator = std.testing.allocator;
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();
    var device_types = getDeviceTypes();
    var iterator = device_types.iterator();
    while (iterator.next()) |d_type| {
        if (mgr.isDeviceTypeAvailable(d_type)) {
            var devices = try mgr.getDevicesOfType(allocator, d_type);
            defer allocator.free(devices);
            for (devices) |dev| {
                try std.testing.expect(dev.deviceType() == d_type);
            }
        }
    }
}

test "Device nativeId" {
    const allocator = std.testing.allocator;
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    var devices = try mgr.getDevicesOfType(allocator, .x64);
    defer allocator.free(devices);
    for (devices) |dev| {
        try std.testing.expect(dev.nativeId() == kX64DeviceId);
    }
}

test "Device setActive" {
    const allocator = std.testing.allocator;
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
                try std.testing.expectEqual(try mgr.getActiveDevice(d_type), dev.*);
            }
        }
    }
}

const CbCtxTest = struct {
    count: i32 = 0,
};

fn device_cb_test(data: ?*anyopaque, _: i32) !void {
    var ctx: *CbCtxTest = @ptrCast(@alignCast(data));
    ctx.count += 1;
}

test "Device addSetActiveCallback" {
    const allocator = std.testing.allocator;
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    var device_types = getDeviceTypes();
    var iterator = device_types.iterator();
    while (iterator.next()) |d_type| {
        if (mgr.isDeviceTypeAvailable(d_type)) {
            var devices = try mgr.getDevicesOfType(allocator, d_type);
            defer allocator.free(devices);
            for (devices) |dev| {
                var ctx = CbCtxTest{};
                try dev.addSetActiveCallback(device_cb_test, &ctx);
                try dev.setActive();
                try std.testing.expect(ctx.count == 1);
            }
        }
    }
}

test "Device sync" {
    const allocator = std.testing.allocator;
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    var device_types = getDeviceTypes();
    var iterator = device_types.iterator();
    while (iterator.next()) |d_type| {
        if (mgr.isDeviceTypeAvailable(d_type)) {
            var devices = try mgr.getDevicesOfType(allocator, d_type);
            defer allocator.free(devices);
            for (devices) |dev| {
                var threw_err = false;
                dev.sync() catch {
                    threw_err = true;
                };
                try std.testing.expect(!threw_err);
            }
        }
    }
}

test "Device getStreams" {
    const allocator = std.testing.allocator;
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    var device_types = getDeviceTypes();
    var iterator = device_types.iterator();
    while (iterator.next()) |d_type| {
        if (mgr.isDeviceTypeAvailable(d_type)) {
            var devices = try mgr.getDevicesOfType(allocator, d_type);
            defer allocator.free(devices);
            for (devices) |dev| {
                const streams = dev.getStreams();
                var stream_iterator = streams.keyIterator();
                while (stream_iterator.next()) |stream| {
                    var stream_dev = stream.value.device();
                    try std.testing.expect(dev.ptr == stream_dev.ptr);
                }
            }
        }
    }
}
