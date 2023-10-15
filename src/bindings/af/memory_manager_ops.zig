const std = @import("std");
const af = @import("arrayfire.zig");

/// Called after a memory manager is set and becomes active.
pub const MemMgrInitializeFn = af.af_memory_manager_initialize_fn;

/// Called after a memory manager is unset and becomes unused.
pub const MemMgrShutdownFn = af.af_memory_manager_shutdown_fn;

/// Function pointer that will be called by ArrayFire to allocate memory.
pub const MemMgrAllocFn = af.af_memory_manager_alloc_fn;

/// Checks the amount of allocated memory for a pointer.
pub const MemMgrAllocatedFn = af.af_memory_manager_allocated_fn;

/// Unlocks memory from use.
pub const MemMgrUnlockFn = af.af_memory_manager_unlock_fn;

/// Called to signal the memory manager should free memory if possible.
///
/// Called by some external functions that allocate their own memory
/// if they receive an out of memory in order to free up other memory
/// on a device.
pub const MemMgrSignalMemCleanupFn = af.af_memory_manager_signal_memory_cleanup_fn;

/// Populates a character array with human readable information about
/// the current state of the memory manager.
///
/// Prints useful information about the memory manger and its state.
/// No format is enforced and can include any information that could be
/// useful to the user. This function is only called by `printMemInfo`.
pub const MemMgrPrintInfoFn = af.af_memory_manager_print_info_fn;

/// Called to lock a buffer as user-owned memory.
pub const MemMgrUserLockFn = af.af_memory_manager_user_lock_fn;

/// Called to unlock a buffer from user-owned memory.
pub const MemMgrUserUnlockFn = af.af_memory_manager_user_unlock_fn;

/// Queries if a buffer is user locked.
pub const MemMgrIsUserLockedFn = af.af_memory_manager_is_user_locked_fn;

/// Gets memory pressure for a memory manager.
pub const MemMgrGetMemPressureFn = af.af_memory_manager_get_memory_pressure_fn;

/// Called to query if additions to the JIT tree would
/// exert too much memory pressure.
///
/// The ArrayFire JIT compiler will call this function
/// to determine if the number of bytes referenced by the
/// buffers in the JIT tree are causing too much memory
/// pressure on the system.
///
/// If the memory manager decides that the pressure is too great,
/// the JIT tree will be evaluated and this COULD result in some
/// buffers being freed if they are not referenced by other af_arrays.
/// If the memory pressure is not too great the JIT tree may not be
/// evaluated and could continue to get bigger.
///
/// The default memory manager will trigger an evaluation if the buffers
/// in the JIT tree account for half of all buffers allocated.
pub const MemMgrJitTreeExceedsMemPressureFn = af.af_memory_manager_jit_tree_exceeds_memory_pressure_fn;

/// Adds a new device to the memory manager (OpenCL only).
pub const MemMgrAddMemMgmtFn = af.af_memory_manager_add_memory_management_fn;

/// Removes a device from the memory manager (OpenCL only)
pub const MemMgrRemoveMemMgmtFn = af.af_memory_manager_remove_memory_management_fn;

/// Creates an `af.MemoryManager` handle.
pub inline fn createMemMgr(allocator: std.mem.Allocator) !*af.MemoryManager {
    var mgr: af.AF_MemoryManager = undefined;
    try af.AF_CHECK(af.af_create_memory_manager(&mgr), @src());
    return af.MemoryManager.initFromMemMgr(allocator, mgr);
}

/// Destroys an `af.MemoryManager` handle.
pub inline fn releaseMemMgr(mgr: *af.MemoryManager) !void {
    try af.AF_CHECK(af.af_release_memory_manager(mgr.memoryManager_), @src());
}

/// Sets an `af.MemoryManager` to be the default memory manager
/// for non-pinned memory allocations in ArrayFire.
pub inline fn setMemMgr(mgr: *af.MemoryManager) !void {
    try af.AF_CHECK(af.af_set_memory_manager(mgr.memoryManager_), @src());
}

/// Sets an `af.MemoryManager` to be the default memory manager
/// for pinned memory allocations in ArrayFire.
pub inline fn setMemMgrPinned(mgr: *af.MemoryManager) !void {
    try af.AF_CHECK(af.af_set_memory_manager_pinned(mgr.memoryManager_), @src());
}

/// Reset the memory manager being used in ArrayFire to the default
/// memory manager, shutting down the existing memory manager.
pub inline fn unsetMemMgr() !void {
    try af.AF_CHECK(af.af_unset_memory_manager(), @src());
}

/// Reset the pinned memory manager being used in ArrayFire to the
/// default memory manager, shutting down the existing pinned memory
/// manager.
pub inline fn unsetMemMgrPinned() !void {
    try af.AF_CHECK(af.af_unset_memory_manager_pinned(), @src());
}

/// Returns the payload ptr from an `af.MemoryManager`.
pub inline fn memMgrGetPayload(handle: *af.MemoryManager) !?*anyopaque {
    var payload: ?*anyopaque = undefined;
    try af.AF_CHECK(af.af_memory_manager_get_payload(&payload, handle.memoryManager_), @src());
    return payload;
}

/// Sets the payload ptr from an `af.MemoryManager`.
pub inline fn memMgrSetPayload(handle: *af.MemoryManager, payload: ?*anyopaque) !void {
    try af.AF_CHECK(af.af_memory_manager_set_payload(handle.memoryManager_, payload), @src());
}

/// Sets an `MemMgrInitializeFn` for a memory manager.
pub inline fn memMgrSetInitializeFn(handle: *af.MemoryManager, func: MemMgrInitializeFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_initialize_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrShutdownFn` for a memory manager.
pub inline fn memMgrSetShutdownFn(handle: *af.MemoryManager, func: MemMgrShutdownFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_shutdown_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrAllocFn` for a memory manager.
pub inline fn memMgrSetAllocFn(handle: *af.MemoryManager, func: MemMgrAllocFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_alloc_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrAllocatedFn` for a memory manager.
pub inline fn memMgrSetAllocatedFn(handle: *af.MemoryManager, func: MemMgrAllocatedFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_allocated_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrUnlockFn` for a memory manager.
pub inline fn memMgrSetUnlockFn(handle: *af.MemoryManager, func: MemMgrUnlockFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_unlock_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrSignalMemCleanupFn` for a memory manager.
pub inline fn memMgrSetSignalMemCleanupFn(handle: *af.MemoryManager, func: MemMgrSignalMemCleanupFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_signal_memory_cleanup_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrPrintInfoFn` for a memory manager.
pub inline fn memMgrSetPrintInfoFn(handle: *af.MemoryManager, func: MemMgrPrintInfoFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_print_info_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrUserLockFn` for a memory manager.
pub inline fn memMgrSetUserLockFn(handle: *af.MemoryManager, func: MemMgrUserLockFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_user_lock_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrUserUnlockFn` for a memory manager.
pub inline fn memMgrSetUserUnlockFn(handle: *af.MemoryManager, func: MemMgrUserUnlockFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_user_unlock_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrIsUserLockedFn` for a memory manager.
pub inline fn memMgrSetIsUserLockedFn(handle: *af.MemoryManager, func: MemMgrIsUserLockedFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_is_user_locked_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrGetMemPressureFn` for a memory manager.
pub inline fn memMgrSetGetMemPressureFn(handle: *af.MemoryManager, func: MemMgrGetMemPressureFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_get_memory_pressure_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrJitTreeExceedsMemPressureFn` for a memory manager.
pub inline fn memMgrSetJitTreeExceedsMemPressureFn(handle: *af.MemoryManager, func: MemMgrJitTreeExceedsMemPressureFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrAddMemMgmtFn` for a memory manager.
pub inline fn memMgrSetAddMemMgmtFn(handle: *af.MemoryManager, func: MemMgrAddMemMgmtFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_add_memory_management_fn(handle.memoryManager_, func), @src());
}

/// Sets an `MemMgrRemoveMemMgmtFn` for a memory manager.
pub inline fn memMgrSetRemoveMemMgmtFn(handle: *af.MemoryManager, func: MemMgrRemoveMemMgmtFn) !void {
    try af.AF_CHECK(af.af_memory_manager_set_remove_memory_management_fn(handle.memoryManager_, func), @src());
}

/// Returns the id of the currently-active device.
pub inline fn memMgrGetActiveDeviceId(handle: *af.MemoryManager) !i32 {
    var id: c_int = undefined;
    try af.AF_CHECK(af.af_memory_manager_get_active_device_id(&id, handle.memoryManager_), @src());
    return @intCast(id);
}

/// Allocates memory with a native memory function for the active backend.
pub inline fn memMgrNativeAlloc(handle: *af.MemoryManager, size: usize) !?*anyopaque {
    var ptr: ?*anyopaque = undefined;
    try af.AF_CHECK(af.af_memory_manager_native_alloc(&ptr, handle.memoryManager_, size), @src());
    return ptr;
}

/// Frees a pointer with a native memory function for the active backend.
pub inline fn memMgrNativeFree(handle: *af.MemoryManager, ptr: ?*anyopaque) !void {
    try af.AF_CHECK(af.af_memory_manager_native_free(handle.memoryManager_, ptr), @src());
}

/// Returns the maximum memory size for a managed device.
pub inline fn memMgrGetMaxMemSize(handle: *af.MemoryManager, id: i32) !usize {
    var size: usize = undefined;
    try af.AF_CHECK(af.af_memory_manager_get_max_memory_size(&size, handle.memoryManager_, @intCast(id)), @src());
    return size;
}

/// Returns the memory pressure threshold for a memory manager.
pub inline fn memMgrGetMemPressureThreshold(handle: *af.MemoryManager) !f32 {
    var threshold: f32 = undefined;
    try af.AF_CHECK(af.af_memory_manager_get_memory_pressure_threshold(handle.memoryManager_, threshold), @src());
    return threshold;
}

/// Sets the memory pressure threshold for a memory manager.
pub inline fn memMgrSetMemPressureThreshold(handle: *af.MemoryManager, value: f32) !void {
    try af.AF_CHECK(af.af_memory_manager_set_memory_pressure_threshold(handle.memoryManager_, value), @src());
}
