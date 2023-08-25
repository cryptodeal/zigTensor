//! The standard Stream interface.
const std = @import("std");
const rt_device = @import("Device.zig");

const assert = std.debug.assert;
const Device = rt_device.Device;

/// Enum for the type of stream.
pub const StreamType = enum {
    CUDA,
    Synchronous,
};

/// Stream ErrorSet
pub const StreamErrors = error{StreamTypeMismatch};

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
        device: *const fn (ctx: *anyopaque) *anyopaque,
        sync: *const fn (ctx: *anyopaque) void,
        relativeSync: *const fn (ctx: *anyopaque, wait_on: Stream) anyerror!void,
        relativeSyncMulti: *const fn (ctx: *anyopaque, wait_ons: *std.AutoHashMap(Stream, void)) anyerror!void,
        deinit: *const fn (ctx: *anyopaque) void,
    };

    /// Returns enum denoting the type of this stream.
    pub fn streamType(ctx: *Self) StreamType {
        return ctx.vtable.streamType(ctx.ptr);
    }

    /// Return the owner device of this stream.
    pub fn device(ctx: *Self) Device {
        return ctx.vtable.device(ctx.ptr);
    }

    /// Block calling thread and synchronize with regard to
    /// all tasks on this stream.
    pub fn sync(ctx: *Self) void {
        return ctx.vtable.sync(ctx.ptr);
    }

    /// Synchronize future tasks on this stream with regard to
    /// current tasks on all given stream, i.e., the former can only
    /// start after the completion of the latter. N.B. this function
    /// may or may not block the calling thread.
    pub fn relativeSync(ctx: *Self, wait_on: Stream) !void {
        return ctx.vtable.relativeSync(ctx.ptr, wait_on);
    }

    /// Synchronize future tasks on this stream with regard to
    /// current tasks on given stream, i.e., the former can only
    /// start after the completion of the latter. N.B. this function
    /// may or may not block the calling thread.
    pub fn relativeSyncMulti(ctx: *anyopaque, wait_ons: *std.AutoHashMap(Stream, void)) !void {
        return ctx.vtable.relativeSyncMulti(ctx.ptr, wait_ons);
    }

    /// Frees all associated memory from the implementation.
    pub fn deinit(ctx: *Self) void {
        return ctx.vtable.deinit(ctx.ptr);
    }

    pub fn init(comptime backend_impl: anytype) Self {
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

            fn sync(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.sync();
            }

            fn relativeSync(ctx: *anyopaque, wait_on: Stream) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.relativeSync(wait_on);
            }

            fn relativeSyncMulti(ctx: *anyopaque, wait_ons: *std.AutoHashMap(Stream, void)) !void {
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
