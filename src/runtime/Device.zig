const std = @import("std");
const rc = @import("zigrc");
const rt_device_type = @import("DeviceType.zig");
const rt_stream = @import("Stream.zig");
const rt_device_mgr = @import("DeviceManager.zig");

const assert = std.debug.assert;
const kX64DeviceId = @import("DeviceManager.zig").kX64DeviceId;
const Arc = rc.Arc;
const DeviceType = rt_device_type.DeviceType;
const getDeviceTypes = rt_device_type.getDeviceTypes;
const Stream = rt_stream.Stream;
const DeviceManager = rt_device_mgr.DeviceManager;

/// Throws an error and logs with a descriptive message
/// if the given types don't match
pub fn deviceImplCheck(comptime expect: type, comptime actual: type) !void {
    if (expect != actual) {
        std.debug.print("[zt.Device.impl] specified device type: [{any}] doesn't match actual device type: [{any}]\n", .{expect});
        return error.DeviceTypeMismatch;
    }
}

pub const Device = struct {
    const Self = @This();

    // The type erased pointer to the TensorBackend implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        setActiveImpl: *const fn (ctx: *anyopaque) void,
        getStreams: *const fn (ctx: *anyopaque) std.AutoHashMap(Arc(Stream), void),
        // addStream: *const fn (ctx: *anyopaque, stream: Arc(Stream)) StreamErrors!void,
        sync: *const fn (ctx: *anyopaque) void,
        nativeId: *const fn (ctx: *anyopaque) c_int,
        deviceType: *const fn (ctx: *anyopaque) DeviceType,
        setActive: *const fn (ctx: *anyopaque) anyerror!void,
        addSetActiveCallback: *const fn (ctx: *anyopaque, callback: *const fn (id: c_int) anyerror!void) anyerror!void,
        deinit: *const fn (ctx: *anyopaque) void,
    };

    /// Set this device as the active device, without worrying about the callbacks.
    pub fn setActiveImpl(ctx: *Self) void {
        return ctx.vtable.setActiveImpl(ctx.ptr);
    }

    /// Returns an immutable list of all streams managed by this device.
    pub fn getStreams(ctx: *Self) std.AutoHashMap(Arc(Stream), void) {
        return ctx.vtable.getStreams(ctx.ptr);
    }

    // Let this device manage given stream. Do nothing if it was already added.
    // pub fn addStream(ctx: *Self, stream: Arc(Stream)) !void {
    // return ctx.vtable.addStream(ctx.ptr, stream);
    // }

    /// Block calling thread and synchronize with regard to all streams on this device.
    pub fn sync(ctx: *Self) void {
        return ctx.vtable.sync(ctx.ptr);
    }

    /// Returns the native ID of this device (semantics are implementation-dependent).
    pub fn nativeId(ctx: *Self) c_int {
        return ctx.vtable.nativeId(ctx.ptr);
    }

    /// Returns the type of this device.
    pub fn deviceType(ctx: *Self) DeviceType {
        return ctx.vtable.deviceType(ctx.ptr);
    }

    /// Set this device as the active device and invokes any callbacks added.
    pub fn setActive(ctx: *Self) !void {
        return ctx.vtable.setActive(ctx.ptr);
    }

    /// Lets this device keep track of the given callback (along with previously
    /// added ones), which will be invoked with the device's native ID after
    /// setting the device active.
    pub fn addSetActiveCallback(ctx: *Self, callback: *const fn (id: c_int) anyerror!void) !void {
        return ctx.vtable.addSetActiveCallback(ctx.ptr, callback);
    }

    // Returns the underlying implementation of this device.
    //
    // Throws if the specified type does not match the actual
    // derived device type.
    pub fn getImpl(ctx: *Self, comptime T: type) *T {
        if (T.type_ != DeviceType.x64)
            return @ptrCast(@alignCast(ctx.ptr));
    }

    /// Frees all associated memory from the implementation.
    pub fn deinit(ctx: *Self) void {
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

            // fn addStream(ctx: *anyopaque, stream: Arc(Stream)) !void {
            // const self: Ptr = @ptrCast(@alignCast(ctx));
            // try self.addStream(stream);
            // }

            fn sync(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.sync();
            }

            fn nativeId(ctx: *anyopaque) c_int {
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

            fn addSetActiveCallback(ctx: *anyopaque, callback: *const fn (id: c_int) anyerror!void) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.addSetActiveCallback(callback);
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
                // .addStream = impl.addStream,
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

pub const StreamErrors = error{DeviceMustOwnStream} || std.mem.Allocator.Error;

pub const X64Device = struct {
    pub const type_: DeviceType = .x64;
    /// Tracks Streams from the Device.
    streams_: std.AutoHashMap(Arc(Stream), void),
    /// Used to update internal backend state for active device, thereby
    /// eliminating the `setActive --> AnyTensorBackendImpl` dependency(s).
    setActiveCallbacks_: std.ArrayList(*const fn (id: c_int) anyerror!void),

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*X64Device {
        var self = try allocator.create(X64Device);
        self.* = .{
            .allocator = allocator,
            .streams_ = std.AutoHashMap(Arc(Stream), void).init(allocator),
            .setActiveCallbacks_ = std.ArrayList(*const fn (id: c_int) anyerror!void).init(allocator),
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
    // pub fn addStream(self: *X64Device, stream: Arc(Stream)) StreamErrors!void {
    // if (!self.eql(stream.device())) {
    // std.log.err("[Device.addStream] Must add stream to owner device\n", .{});
    // return StreamErrors.DeviceMustOwnStream;
    // }
    // try self.streams_.put(stream, void);
    // }

    pub fn sync(self: *X64Device) void {
        self.setActiveImpl();
        var iterator = self.streams_.keyIterator();
        while (iterator.next()) |stream| {
            _ = stream;
            // TODO: finish implementing `Stream` interface/impl
            // stream.value.sync();
        }
    }

    pub fn nativeId(_: *X64Device) c_int {
        return kX64DeviceId;
    }

    pub fn deviceType(_: *X64Device) DeviceType {
        return X64Device.type_;
    }

    pub fn setActive(self: *X64Device) !void {
        self.setActiveImpl();
        for (self.setActiveCallbacks_.items) |callback| {
            try callback(self.nativeId());
        }
    }

    pub fn addSetActiveCallback(self: *X64Device, callback: *const fn (id: c_int) anyerror!void) StreamErrors!void {
        try self.setActiveCallbacks_.append(callback);
    }

    // pub fn getImpl(self: *X64Device, comptime T: type) !*T {
    // try deviceImplCheck(X64Device, T);
    // return self;
    // }
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
