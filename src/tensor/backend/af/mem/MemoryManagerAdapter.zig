//! The ArrayFire MemoryManagerAdapter interface.

const std = @import("std");
const af = @import("../../../../backends/ArrayFire.zig");

const assert = std.debug.assert;

pub const kDefaultLogFlushInterval: usize = 50;

/// An interface for defining memory managers purely in zig.
///
/// The ArrayFire memory management API is defined using C callbacks; this class
/// provides a thin layer of abstraction over this callbacks and acts as an
/// adapter between derived Zig struct implementations and the ArrayFire C API. In
/// particular:
///
/// - Each instance has an associated af_memory_manager whose payload is a
///   pointer to `this` which allows callbacks to call Zig struct methods after
///   casting.
///
/// - Provides logging functions and a logging mode which logs all function calls
///   from ArrayFire and all relevant arguments. Only virtual base class methods
///   that have derived implementations are eligible for logging.
///
/// - The `MemoryManagerInstaller` provides an interface for setting implemented
///   memory managers as the active ArrayFire memory managers by setting relevant
///   callbacks on construction.
///
/// For documentation of virtual methods, see [ArrayFire's memory header](https://git.io/Jv7do)
/// for full specifications on when these methods are called by ArrayFire and the JIT.
pub const MemoryManagerAdapter = struct {
    const Self = @This();

    // The type erased pointer to the TensorBackend implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        initialize: *const fn (ctx: *anyopaque) void,
        shutdown: *const fn (ctx: *anyopaque) void,
        alloc: *const fn (ctx: *anyopaque, user_lock: bool, ndims: u32, dims: *af.dim_t, el_size: u32) anyerror!?*anyopaque,
        allocated: *const fn (ctx: *anyopaque, ptr: ?*anyopaque) usize,
        unlock: *const fn (ctx: *anyopaque, ptr: ?*anyopaque, user_lock: bool) void,
        signalMemoryCleanup: *const fn (ctx: *anyopaque) void,
        printInfo: *const fn (ctx: *anyopaque, msg: []const u8, device: c_int, writer: anytype) anyerror!void,
        userLock: *const fn (ctx: *anyopaque, ptr: *anyopaque) void,
        userUnlock: *const fn (ctx: *anyopaque, ptr: *anyopaque) void,
        isUserLocked: *const fn (ctx: *anyopaque, ptr: *anyopaque) bool,
        getMemoryPressure: *const fn (ctx: *anyopaque) f32,
        jitTreeExceedsMemoryPressure: *const fn (ctx: *anyopaque, bytes: usize) bool,
        addMemoryManagement: *const fn (ctx: *anyopaque, device: c_int) void,
        removeMemoryManagement: *const fn (ctx: *anyopaque, device: c_int) void,
        getMemSizeStep: *const fn (ctx: *anyopaque) usize,
        setMemSizeStep: *const fn (ctx: *anyopaque, size: usize) void,
        // TODO: log: *const fn () anyerror!void,
        // TODO: setLogStream: *const fn (),
        // TODO: getLogStream: *const fn (),
        setLoggingEnabled: *const fn (ctx: *anyopaque, log: bool) void,
        setLogFlushInterval: *const fn (ctx: *anyopaque, interval: usize) void,
        getHandle: *const fn (ctx: *anyopaque) anyerror!af.af_memory_manager,
    };

    pub fn initialize(self: *Self) void {
        self.vtable.initialize(self.ptr);
    }

    pub fn shutdown(self: *Self) void {
        self.vtable.shutdown(self.ptr);
    }

    pub fn alloc(self: *Self, user_lock: bool, ndims: u32, dims: *af.dim_t, el_size: u32) !?*anyopaque {
        return self.vtable.alloc(self.ptr, user_lock, ndims, dims, el_size);
    }

    pub fn allocated(self: *Self, ptr: ?*anyopaque) usize {
        return self.vtable.allocated(self.ptr, ptr);
    }

    pub fn unlock(self: *Self, ptr: ?*anyopaque, user_lock: bool) void {
        self.vtable.unlock(self.ptr, ptr, user_lock);
    }

    pub fn signalMemoryCleanup(self: *Self) void {
        self.vtable.signalMemoryCleanup(self.ptr);
    }

    pub fn printInfo(self: *Self, msg: []const u8, device: c_int, writer: anytype) !void {
        try self.vtable.printInfo(self.ptr, msg, device, writer);
    }

    pub fn userLock(self: *Self, ptr: *anyopaque) void {
        self.vtable.userLock(self.ptr, ptr);
    }

    pub fn userUnlock(self: *Self, ptr: *anyopaque) void {
        self.vtable.userUnlock(self.ptr, ptr);
    }

    pub fn isUserLocked(self: *Self, ptr: *anyopaque) bool {
        return self.vtable.isUserLocked(self.ptr, ptr);
    }

    pub fn getMemoryPressure(self: *Self) f32 {
        return self.vtable.getMemoryPressure(self.ptr);
    }

    pub fn jitTreeExceedsMemoryPressure(self: *Self, bytes: usize) bool {
        return self.vtable.jitTreeExceedsMemoryPressure(self.ptr, bytes);
    }

    pub fn addMemoryManagement(self: *Self, device: c_int) void {
        self.vtable.addMemoryManagement(self.ptr, device);
    }

    pub fn removeMemoryManagement(self: *Self, device: c_int) void {
        self.vtable.removeMemoryManagement(self.ptr, device);
    }

    pub fn getMemSizeStep(self: *Self) usize {
        return self.vtable.getMemSizeStep(self.ptr);
    }

    pub fn setMemSizeStep(self: *Self, size: usize) void {
        self.vtable.setMemSizeStep(self.ptr, size);
    }

    // TODO: pub fn log()

    // TODO: pub fn setLogStream()

    // TODO: pub fn getLogStream()

    pub fn setLoggingEnabled(self: *Self, log: bool) void {
        self.vtable.setLoggingEnabled(self.ptr, log);
    }

    pub fn setLogFlushInterval(self: *Self, interval: usize) void {
        self.vtable.setLogFlushInterval(self.ptr, interval);
    }

    pub fn getHandle(self: *Self) !af.af_memory_manager {
        return self.vtable.getHandle(self.ptr);
    }

    pub fn init(backend_impl: anytype) Self {
        const Ptr = @TypeOf(backend_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn initialize(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.initialize();
            }

            fn shutdown(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.shutdown();
            }

            fn alloc(ctx: *anyopaque, user_lock: bool, ndims: u32, dims: *af.dim_t, el_size: u32) !?*anyopaque {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.alloc(user_lock, ndims, dims, el_size);
            }

            fn allocated(ctx: *anyopaque, ptr: ?*anyopaque) usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.allocated(ptr);
            }

            fn unlock(ctx: *anyopaque, ptr: ?*anyopaque, user_lock: bool) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.unlock(ptr, user_lock);
            }

            fn signalMemoryCleanup(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.signalMemoryCleanup();
            }

            fn printInfo(ctx: *anyopaque, msg: []const u8, device: c_int, writer: anytype) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.printInfo(msg, device, writer);
            }

            fn userLock(ctx: *anyopaque, ptr: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.userLock(ptr);
            }

            fn userUnlock(ctx: *anyopaque, ptr: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.userUnlock(ptr);
            }

            fn isUserLocked(ctx: *anyopaque, ptr: *anyopaque) bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isUserLocked(ptr);
            }

            fn getMemoryPressure(ctx: *anyopaque) f32 {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getMemoryPressure();
            }

            fn jitTreeExceedsMemoryPressure(ctx: *anyopaque, bytes: usize) bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.jitTreeExceedsMemoryPressure(bytes);
            }

            fn addMemoryManagement(ctx: *anyopaque, device: c_int) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.addMemoryManagement(device);
            }

            fn removeMemoryManagement(ctx: *anyopaque, device: c_int) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.removeMemoryManagement(device);
            }

            fn getMemSizeStep(ctx: *anyopaque) usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getMemSizeStep();
            }

            fn setMemSizeStep(ctx: *anyopaque, size: usize) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.setMemSizeStep(size);
            }

            // TODO: fn log()

            // TODO: fn setLogStream()

            // TODO: fn getLogStream()

            fn setLoggingEnabled(ctx: *anyopaque, log: bool) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.setLoggingEnabled(log);
            }

            fn setLogFlushInterval(ctx: *anyopaque, interval: usize) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.setLogFlushInterval(interval);
            }

            fn getHandle(ctx: *anyopaque) !af.af_memory_manager {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getHandle();
            }
        };
        return .{
            .ptr = backend_impl,
            .vtable = &.{
                .initialize = impl.initialize,
                .shutdown = impl.shutdown,
                .alloc = impl.alloc,
                .allocated = impl.allocated,
                .unlock = impl.unlock,
                .signalMemoryCleanup = impl.signalMemoryCleanup,
                .printInfo = impl.printInfo,
                .userLock = impl.userLock,
                .userUnlock = impl.userUnlock,
                .isUserLocked = impl.isUserLocked,
                .getMemoryPressure = impl.getMemoryPressure,
                .jitTreeExceedsMemoryPressure = impl.jitTreeExceedsMemoryPressure,
                .addMemoryManagement = impl.addMemoryManagement,
                .removeMemoryManagement = impl.removeMemoryManagement,
                .getMemSizeStep = impl.getMemSizeStep,
                .setMemSizeStep = impl.setMemSizeStep,
                // TODO: .log = impl.log,
                // TODO: .setLogStream = impl.setLogStream,
                // TODO: .getLogStream = impl.getLogStream,
                .setLoggingEnabled = impl.setLoggingEnabled,
                .setLogFlushInterval = impl.setLogFlushInterval,
                .getHandle = impl.getHandle,
            },
        };
    }
};
