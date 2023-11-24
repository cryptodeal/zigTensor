//! The standard Stream interface.
const std = @import("std");
const rt_device = @import("device.zig");

const assert = std.debug.assert;
const Device = rt_device.Device;

/// Enum for the type of stream.
pub const StreamType = enum {
    CUDA,
    Synchronous,
};

/// Stream ErrorSet
pub const StreamErrors = error{ StreamTypeMismatch, FailedDeviceGetImpl };

/// An abstraction for a sequence of computations that
/// must be executed synchronously on a specific device.
/// It prioritizes the synchronization of the computations,
/// while remaining agnostic to the computations themselves.
pub const Stream = struct {
    const Self = @This();

    // The type erased pointer to the TensorBackend implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        streamType: *const fn (ctx: *anyopaque) StreamType,
        device: *const fn (ctx: *anyopaque) Device,
        sync: *const fn (ctx: *anyopaque) anyerror!void,
        relativeSync: *const fn (ctx: *anyopaque, wait_on: Stream) anyerror!void,
        relativeSyncMulti: *const fn (ctx: *anyopaque, wait_ons: std.AutoHashMap(Stream, void)) anyerror!void,
        deinit: *const fn (ctx: *anyopaque) void,
    };

    // Returns the underlying implementation of this device.
    //
    // Throws if the specified type does not match the actual
    // derived device type.
    pub fn getImpl(ctx: *const Self, comptime T: type) StreamErrors!*T {
        const actual_type = ctx.streamType();
        if (T.type_ != actual_type) {
            std.log.debug("[zt.Stream.getImpl] specified stream type: [{s}] doesn't match actual stream type: [{s}]\n", .{ @tagName(T.type_), @tagName(actual_type) });
            return error.FailedDeviceGetImpl;
        }
        return @ptrCast(@alignCast(ctx.ptr));
    }

    /// Returns enum denoting the type of this stream.
    pub fn streamType(ctx: *const Self) StreamType {
        return ctx.vtable.streamType(ctx.ptr);
    }

    /// Return the owner device of this stream.
    pub fn device(ctx: *const Self) Device {
        return ctx.vtable.device(ctx.ptr);
    }

    /// Return the owner device implementation of this stream.
    pub fn deviceImpl(ctx: *const Self, comptime T: type) !*T {
        var dev = ctx.device();
        return dev.getImpl(T);
    }

    /// Block calling thread and synchronize with regard to
    /// all tasks on this stream.
    pub fn sync(ctx: *const Self) !void {
        try ctx.vtable.sync(ctx.ptr);
    }

    /// Synchronize future tasks on this stream with regard to
    /// current tasks on all given stream, i.e., the former can only
    /// start after the completion of the latter. N.B. this function
    /// may or may not block the calling thread.
    pub fn relativeSync(ctx: *const Self, wait_on: Stream) !void {
        return ctx.vtable.relativeSync(ctx.ptr, wait_on);
    }

    /// Synchronize future tasks on this stream with regard to
    /// current tasks on given stream, i.e., the former can only
    /// start after the completion of the latter. N.B. this function
    /// may or may not block the calling thread.
    pub fn relativeSyncMulti(ctx: *const Self, wait_ons: std.AutoHashMap(Stream, void)) !void {
        return ctx.vtable.relativeSyncMulti(ctx.ptr, wait_ons);
    }

    /// Frees all associated memory from the implementation.
    pub fn deinit(ctx: *const Self) void {
        return ctx.vtable.deinit(ctx.ptr);
    }

    pub fn init(backend_impl: anytype) Self {
        const Ptr = @TypeOf(backend_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn streamType(ctx: *anyopaque) StreamType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.streamType();
            }

            fn device(ctx: *anyopaque) Device {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.device();
            }

            fn sync(ctx: *anyopaque) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.sync();
            }

            fn relativeSync(ctx: *anyopaque, wait_on: Stream) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.relativeSync(wait_on);
            }

            fn relativeSyncMulti(ctx: *anyopaque, wait_ons: std.AutoHashMap(Stream, void)) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.relativeSyncMulti(wait_ons);
            }

            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }
        };
        return .{
            .ptr = backend_impl,
            .vtable = &.{
                .streamType = impl.streamType,
                .device = impl.device,
                .sync = impl.sync,
                .relativeSync = impl.relativeSync,
                .relativeSyncMulti = impl.relativeSyncMulti,
                .deinit = impl.deinit,
            },
        };
    }
};
