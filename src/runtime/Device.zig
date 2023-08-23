const std = @import("std");
const rc = @import("zigrc");
const device_type = @import("DeviceType.zig");
const rt_stream = @import("Stream.zig");

const assert = std.debug.assert;
const Arc = rc.Arc;
const DeviceType = device_type.DeviceType;
const Stream = rt_stream.Stream;

/// Throws an error and logs with a descriptive message
/// if the given types don't match
pub fn deviceImplCheck(expect: DeviceType, actual: DeviceType) !void {
    if (expect != actual) {
        std.debug.print("[zt.Device.impl] specified device type: [{any}] doesn't match actual device type: [{any}]\n", .{expect});
        return error.DeviceTypeMismatch;
    }
}

pub const Device1 = struct {
    /// Tracks Streams from the Device.
    streams_: std.AutoHashMap(Arc(Stream), void),
    /// Used to update internal backend state for active device, thereby
    /// eliminating the `setActive --> AnyTensorBackendImpl` dependency(s).
    setActiveCallbacks_: std.ArrayList(*const fn (d: usize) void),

    pub fn init(allocator: std.mem.Allocator) !*Device {
        const self = try allocator.create(Device);
        self.* = .{
            .streams_ = try std.AutoHashMap(Arc(Stream), void).init(allocator),
            .setActiveCallbacks_ = try std.ArrayList(*const fn (d: usize) void).init(allocator),
        };
    }

    pub fn deinit(self: *Device) void {
        // TODO decrement keys in streams_
        self.streams_.deinit();
        self.setActiveCallbacks_.deinit();
    }

    pub fn getStreams(self: *Device) std.AutoHashMap(Arc(Stream), void) {
        return self.streams_;
    }
};

pub const Device = struct {
    const Self = @This();

    // The type erased pointer to the TensorBackend implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        setActiveImpl: *const fn (ctx: *Self) void,
        getStreams: *const fn (ctx: *Self) std.AutoHashMap(Arc(Stream), void),
        addStream: *const fn (ctx: *Self, stream: Arc(Stream)) anyerror!void,
        sync: *const fn (ctx: *Self) void,
        nativeId: *const fn (ctx: *Self) c_int,
        deviceType: *const fn (ctx: *Self) DeviceType,
        setActive: *const fn (ctx: *Self) void,
        addSetActiveCallback: *const fn (ctx: *Self, callback: *const fn (v: c_int) void) anyerror!void,
        getImpl: *const fn (ctx: *Self, comptime T: type) anyerror!anyopaque,
        deinit: *const fn (ctx: *Self) void,
    };

    /// Set this device as the active device, without worrying about the callbacks.
    pub fn setActiveImpl(ctx: *Self) void {
        ctx.vtable.setActiveImpl(ctx.ptr);
    }

    /// Returns an immutable list of all streams managed by this device.
    pub fn getStreams(ctx: *Self) std.AutoHashMap(Arc(Stream), void) {
        return ctx.vtable.getStreams(ctx.ptr);
    }

    /// Let this device manage given stream. Do nothing if it was already added.
    pub fn addStream(ctx: *Self, stream: Arc(Stream)) !void {
        try ctx.vtable.addStream(ctx.ptr, stream);
    }

    /// Block calling thread and synchronize with regard to all streams on this device.
    pub fn sync(ctx: *Self) void {
        ctx.vtable.sync(ctx.ptr);
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
    pub fn setActive(ctx: *Self) void {
        return ctx.vtable.setActive(ctx.ptr);
    }

    /// Lets this device keep track of the given callback (along with previously
    /// added ones), which will be invoked with the device's native ID after
    /// setting the device active.
    pub fn addSetActiveCallback(ctx: *Self, callback: *const fn (v: c_int) void) !void {
        try ctx.vtable.addSetActiveCallback(ctx.ptr, callback);
    }

    /// Returns the underlying implementation of this device.
    ///
    /// Throws if the specified type does not match the actual
    /// derived device type.
    pub fn getImpl(ctx: *Self, comptime T: type) !T {
        return ctx.vtable.getImpl(ctx.ptr, T);
    }

    /// Frees all associated memory from the implementation.
    pub fn deinit(ctx: *Self) void {
        ctx.vtable.deinit(ctx.ptr);
    }

    pub fn init(backend_impl: anytype) Self {
        const Ptr = @TypeOf(backend_impl);
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

            fn setActive(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.setActive();
            }

            fn addSetActiveCallback(ctx: *anyopaque, callback: *const fn (v: c_int) void) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.addSetActiveCallback(callback);
            }

            fn getImpl(ctx: *anyopaque, comptime T: type) !T {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getImpl(T);
            }

            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }
        };
        return .{
            .ptr = backend_impl,
            .vtable = &.{
                .setActiveImpl = impl.setActiveImpl,
                .getStreams = impl.getStreams,
                .addStream = impl.addStream,
                .sync = impl.sync,
                .nativeId = impl.nativeId,
                .deviceType = impl.deviceType,
                .setActive = impl.setActive,
                .addSetActiveCallback = impl.addSetActiveCallback,
                .getImpl = impl.getImpl,
                .deinit = impl.deinit,
            },
        };
    }
};
