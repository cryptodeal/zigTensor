const std = @import("std");
const af = @import("arrayfire.zig");

/// Wraps `af.af_memory_manager`, the ArrayFire memory manager interface,
/// as a zig struct to simplify calling into the ArrayFire C API.
pub const MemoryManager = struct {
    pub const Self = @This();
    allocator: std.mem.Allocator,
    memoryManager_: af.af_memory_manager,

    pub fn initFromMemoryManager(allocator: std.mem.Allocator, mem_mgr: af.af_memory_manager) !*Self {
        const self = try allocator.create(Self);
        self.* = .{ .allocator = allocator, .memoryManager_ = mem_mgr };
        return self;
    }

    pub fn init(allocator: std.mem.Allocator) !*Self {
        return af.ops.createMemMgr(allocator);
    }

    pub fn deinit(self: *Self) void {
        self.releaseMemMgr() catch unreachable;
        self.allocator.destroy(self);
    }

    pub fn set(self: *Self) !void {
        return af.ops.setMemMgr(self);
    }

    pub fn setPinned(self: *Self) !void {
        return af.ops.setMemMgrPinned(self);
    }

    pub fn unset(self: *Self) !void {
        return af.ops.unsetMemMgr(self);
    }

    pub fn unsetPinned(self: *Self) !void {
        return af.ops.unsetMemMgrPinned(self);
    }

    pub fn getPayload(self: *Self) !?*anyopaque {
        return af.ops.memMgrGetPayload(self);
    }

    pub fn setPayload(self: *Self, payload: ?*anyopaque) !void {
        return af.ops.memMgrSetPayload(self, payload);
    }

    pub fn setInitializeFn(self: *Self, func: af.ops.MemMgrInitializeFn) !void {
        return af.ops.memMgrSetInitializeFn(self, func);
    }

    pub fn setShutdownFn(self: *Self, func: af.ops.MemMgrShutdownFn) !void {
        return af.ops.memMgrSetShutdownFn(self, func);
    }

    pub fn setAllocFn(self: *Self, func: af.ops.MemMgrAllocFn) !void {
        return af.ops.memMgrSetAllocFn(self, func);
    }

    pub fn setAllocatedFn(self: *Self, func: af.ops.MemMgrAllocatedFn) !void {
        return af.ops.memMgrSetAllocatedFn(self, func);
    }

    pub fn setSignalMemCleanupFn(self: *Self, func: af.ops.MemMgrSignalMemCleanupFn) !void {
        return af.ops.memMgrSetSignalMemCleanupFn(self, func);
    }

    pub fn setPrintInfoFn(self: *Self, func: af.ops.MemMgrPrintInfoFn) !void {
        return af.ops.memMgrSetPrintInfoFn(self, func);
    }

    pub fn setUserLockFn(self: *Self, func: af.ops.MemMgrUserLockFn) !void {
        return af.ops.memMgrSetUserLockFn(self, func);
    }

    pub fn setUserUnlockFn(self: *Self, func: af.ops.MemMgrUserUnlockFn) !void {
        return af.ops.memMgrSetUserUnlockFn(self, func);
    }

    pub fn setIsUserLockedFn(self: *Self, func: af.ops.MemMgrIsUserLockedFn) !void {
        return af.ops.memMgrSetIsUserLockedFn(self, func);
    }

    pub fn setGetMemPressureFn(self: *Self, func: af.ops.MemMgrGetMemPressureFn) !void {
        return af.ops.memMgrSetGetMemPressureFn(self, func);
    }

    pub fn setJitTreeExceedsMemPressureFn(self: *Self, func: af.ops.MemMgrJitTreeExceedsMemPressureFn) !void {
        return af.ops.memMgrSetJitTreeExceedsMemPressureFn(self, func);
    }

    pub fn setAddMemMgmtFn(self: *Self, func: af.ops.MemMgrAddMemMgmtFn) !void {
        return af.ops.memMgrSetAddMemMgmtFn(self, func);
    }

    pub fn setRemoveMemMgmtFn(self: *Self, func: af.ops.MemMgrRemoveMemMgmtFn) !void {
        return af.ops.memMgrSetRemoveMemMgmtFn(self, func);
    }

    pub fn getActiveDeviceId(self: *Self) !i32 {
        return af.ops.memMgrGetActiveDeviceId(self);
    }

    pub fn nativeAlloc(self: *Self, size: usize) !?*anyopaque {
        return af.ops.memMgrNativeAlloc(self, size);
    }

    pub fn nativeFree(self: *Self, ptr: ?*anyopaque) !void {
        return af.ops.memMgrNativeFree(self, ptr);
    }

    pub fn getMaxMemSize(self: *Self, id: i32) !usize {
        return af.ops.memMgrGetMaxMemSize(self, id);
    }

    pub fn getMemPressureThreshold(self: *Self) !f32 {
        return af.ops.memMgrGetMemPressureThreshold(self);
    }

    pub fn setMemPressureThreshold(self: *Self, value: f32) !void {
        return af.ops.memMgrSetMemPressureThreshold(self, value);
    }
};
