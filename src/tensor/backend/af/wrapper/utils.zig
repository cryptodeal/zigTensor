const std = @import("std");
const af = @import("../../../../backends/ArrayFire.zig");
const af_utils = @import("../Utils.zig");
const zt_type = @import("../../../Types.zig");

const AF_CHECK = af_utils.AF_CHECK;
const ztToAfType = af_utils.ztToAfType;
const DType = zt_type.DType;

/// Display ArrayFire and device info.
pub inline fn info() !void {
    try AF_CHECK(af.af_info(), @src());
}

pub inline fn init() !void {
    try AF_CHECK(af.af_init(), @src());
}

/// Gets the value of `info` as a string.
pub inline fn infoString(info_: [*c]u8, verbose: bool) !void {
    try AF_CHECK(af.af_info_string(&info_, verbose), @src());
}

/// Gets the information about device and platform as strings.
pub inline fn deviceInfo(d_name: [*c]u8, d_platform: [*c]u8, d_toolkit: [*c]u8, d_compute: [*c]u8) !void {
    try AF_CHECK(af.af_device_info(d_name, d_platform, d_toolkit, d_compute), @src());
}

/// Returns the number of compute devices on the system.
pub inline fn getDeviceCount() !usize {
    var numDevices: c_int = 0;
    try AF_CHECK(af.af_get_device_count(&numDevices), @src());
    return @intCast(numDevices);
}

/// Returns true if the device supports double precision operations; else false.
pub inline fn getDoubleSupport() !bool {
    var support: bool = undefined;
    try AF_CHECK(af.af_get_dbl_support(&support), @src());
    return support;
}

/// Returns true if the device supports half precision operations; else false.
pub inline fn getHalfSupport() !bool {
    var support: bool = undefined;
    try AF_CHECK(af.af_get_half_support(&support), @src());
    return support;
}

/// Sets the current device.
pub inline fn setDevice(id: i32) !void {
    try AF_CHECK(af.af_set_device(@intCast(id)), @src());
}

/// Returns the id of the current device.
pub inline fn getDevice() !i32 {
    var device: c_int = 0;
    try AF_CHECK(af.af_get_device(&device), @src());
    return @intCast(device);
}

/// Blocks the calling thread until all of the device operations have finished.
pub inline fn sync(id: i32) !void {
    try AF_CHECK(af.af_sync(@intCast(id)), @src());
}

/// Deprecated: use `allocDeviceV2`; Allocates memory using ArrayFire's
/// memory manager.
pub inline fn allocDevice(ptr: ?*anyopaque, bytes: usize) !void {
    try AF_CHECK(af.af_alloc_device(&ptr, @intCast(bytes)), @src());
}

/// Deprecated: use `freeDeviceV2`; Free a device pointer even if
/// it has been previously locked.
pub inline fn freeDevice(ptr: ?*anyopaque) !void {
    try AF_CHECK(af.af_free_device(ptr), @src());
}

/// Allocates memory using ArrayFire's memory manager.
pub inline fn allocDeviceV2(ptr: ?*anyopaque, bytes: usize) !void {
    try AF_CHECK(af.af_alloc_device_v2(&ptr, @intCast(bytes)), @src());
}

/// Free a device pointer even if it has been previously locked.
pub inline fn freeDeviceV2(ptr: ?*anyopaque) !void {
    try AF_CHECK(af.af_free_device_v2(ptr), @src());
}

/// Allocate pinned memory using ArrayFire's memory manager.
/// These functions allocate page locked memory. This type of
/// memory has better performance characteristics but require
/// additional care because they are a limited resource.
pub inline fn allocPinned(ptr: ?*anyopaque, bytes: usize) !void {
    try AF_CHECK(af.af_alloc_pinned(&ptr, @intCast(bytes)), @src());
}

/// Free pinned memory allocated by ArrayFire's memory manager.
/// These calls free the pinned memory on host. These functions
/// need to be called on pointers allocated using pinned function.
pub inline fn freePinned(ptr: ?*anyopaque) !void {
    try AF_CHECK(af.af_free_pinned(ptr), @src());
}

/// Allocate memory on host. This function is used for allocating
/// regular memory on host. This is useful where the compiler version
/// of ArrayFire library is different from the executable's compiler version.
///
/// It does not use ArrayFire's memory manager.
pub inline fn allocHost(ptr: ?*anyopaque, bytes: usize) !void {
    try AF_CHECK(af.af_alloc_host(&ptr, @intCast(bytes)), @src());
}

/// Free memory allocated on host internally by ArrayFire.
/// This function is used for freeing memory on host that
/// was allocated within ArrayFire. This is useful where the
/// compiler version of ArrayFire library is different from
/// the executable's compiler version.
///
/// It does not use ArrayFire's memory manager.
pub inline fn freeHost(ptr: ?*anyopaque) !void {
    try AF_CHECK(af.af_free_host(ptr), @src());
}

/// Data structure that holds memory information from the memory manager.
pub const AFDeviceMemInfo = struct {
    alloc_bytes: usize = undefined,
    alloc_buffers: usize = undefined,
    lock_bytes: usize = undefined,
    lock_buffers: usize = undefined,
};

/// Get memory information from the memory manager.
pub inline fn deviceMemInfo() !AFDeviceMemInfo {
    var mem = AFDeviceMemInfo{};
    try AF_CHECK(af.af_device_mem_info(&mem.alloc_bytes, &mem.alloc_buffers, &mem.lock_bytes, &mem.lock_buffers), @src());
    return mem;
}

/// Prints buffer details from the ArrayFire Device Manager.
pub inline fn printMemInfo(msg: [*c]const u8, device_id: i32) !void {
    try AF_CHECK(af.af_print_mem_info(msg, @intCast(device_id)), @src());
}

/// Call the garbage collection routine.
pub inline fn deviceGC() !void {
    try AF_CHECK(af.af_device_gc(), @src());
}

/// Set the resolution of memory chunks.
pub inline fn setMemStepSize(step_bytes: usize) !void {
    try AF_CHECK(af.af_set_mem_step_size(step_bytes), @src());
}

/// Get the resolution of memory chunks.
pub inline fn getMemStepSize() !usize {
    var step_bytes: usize = undefined;
    try AF_CHECK(af.af_get_mem_step_size(&step_bytes), @src());
    return step_bytes;
}

/// Sets the path where the kernels generated at runtime will be cached.
pub inline fn setKernelCacheDir(path: []const u8, override_env: i32) !void {
    try AF_CHECK(af.af_set_kernel_cache_directory(path.ptr, @intCast(override_env)), @src());
}

/// Gets the path where the kernels generated at runtime will be cached.
pub inline fn getKernelCacheDir() ![]const u8 {
    var path: [*c]u8 = undefined;
    var length: usize = undefined;
    try AF_CHECK(af.af_get_kernel_cache_directory(&length, path), @src());
    return path[0..length];
}

/// Returns the last error message that occurred and its error message.
pub inline fn getLastError() ![]const u8 {
    var err: [*c]u8 = undefined;
    var len: usize = undefined;
    try AF_CHECK(af.af_get_last_error(&len, &err), @src());
    return err[0..len];
}

/// Converts the af_err error code to its string representation.
pub inline fn errToString(err: af.af_err) ![]const u8 {
    var str = af.af_err_to_string(err);
    return std.mem.span(str);
}

/// Create a new af_seq object.
pub inline fn makeSeq(begin: f64, end: f64, step: f64) af.af_seq {
    return .{ .begin = begin, .end = end, .step = step };
}
