const std = @import("std");
const af = @import("arrayfire.zig");
const zt_idx = @import("../../tensor/index.zig");
const zt_base = @import("../../tensor/tensor_base.zig");
const zt_shape = @import("../../tensor/shape.zig");

const ArrayFireTensor = @import("../../tensor/backend/af/arrayfire_tensor.zig").ArrayFireTensor;
const Shape = zt_shape.Shape;
const DType = @import("../../tensor/types.zig").DType;
const Dim = zt_shape.Dim;
const SortMode = zt_base.SortMode;
const Location = zt_base.Location;
const StorageType = zt_base.StorageType;
const MatrixProperty = zt_base.MatrixProperty;
const PadType = zt_base.PadType;
const Index = zt_idx.Index;
const Range = zt_idx.Range;

pub inline fn getFNSD(comptime T: type, comptime dimT: type, dim: dimT, dims: af.Dim4) T {
    if (dim >= 0) return dim;

    var fNSD: T = 0;
    for (0..4) |i| {
        if (dims.dims[i] > 1) {
            fNSD = @intCast(i);
            break;
        }
    }
    return fNSD;
}

pub fn ztToAfDims(shape: Shape) !af.Dim4 {
    if (zt_shape.ndim(shape) > 4) {
        std.log.debug("ztToAfDims: ArrayFire shapes can't be more than 4 dimensions\n", .{});
        return error.ArrayFireCannotExceed4Dimensions;
    }
    var af_Dim4 = af.Dim4{};
    for (0..zt_shape.ndim(shape)) |i| af_Dim4.dims[i] = @intCast(shape[i]);
    return af_Dim4;
}

pub fn ztToAfType(self: DType) af.Dtype {
    return switch (self) {
        .f16 => af.Dtype.f16,
        .f32 => af.Dtype.f32,
        .f64 => af.Dtype.f64,
        .b8 => af.Dtype.b8,
        .s16 => af.Dtype.s16,
        .s32 => af.Dtype.s32,
        .s64 => af.Dtype.s64,
        .u8 => af.Dtype.u8,
        .u16 => af.Dtype.u16,
        .u32 => af.Dtype.u32,
        .u64 => af.Dtype.u64,
    };
}

pub fn ztToAfMatrixProperty(property: MatrixProperty) af.MatProp {
    return switch (property) {
        .None => af.MatProp.None,
        .Transpose => af.MatProp.Trans,
    };
}

pub fn ztToAfStorageType(storage_type: StorageType) af.Storage {
    return switch (storage_type) {
        .Dense => af.Storage.Dense,
        .CSR => af.Storage.CSR,
        .CSC => af.Storage.CSC,
        .COO => af.Storage.COO,
    };
}

pub fn ztToAfBorderType(self: PadType) af.BorderType {
    return switch (self) {
        .Constant => af.BorderType.PadZero,
        .Edge => af.BorderType.PadClampToEdge,
        .Symmetric => af.BorderType.PadSym,
    };
}

pub fn ztToAfTopKSortMode(sort_mode: SortMode) af.TopkFn {
    return switch (sort_mode) {
        .Descending => .Max,
        .Ascending => .Min,
    };
}

pub fn ztToAfLocation(location: Location) af.Source {
    return switch (location) {
        .Host => af.Source.Host,
        .Device => af.Source.Device,
    };
}

pub fn ztRangeToAfSeq(range: Range) af.af_seq {
    const endOpt = range.end();
    const end = if (endOpt != null) endOpt.? - 1 else -1;
    return makeSeq(@floatFromInt(range.start()), @floatFromInt(end), @floatFromInt(range.stride()));
}

pub fn ztToAfIndex(idx: Index) af.af_index_t {
    return switch (idx.idxType()) {
        .Tensor => {
            var tensor = idx.index_.Tensor;
            const adapter: *ArrayFireTensor = tensor.getAdapter(ArrayFireTensor);
            return .{
                .isSeq = false,
                .isBatch = false,
                .idx = .{ .arr = adapter.arrayHandle_.value.*.array_ },
            };
        },
        .Span => .{
            .isSeq = true,
            .isBatch = false,
            .idx = .{ .seq = af.af_span },
        },
        .Range => .{
            .isSeq = true,
            .isBatch = false,
            .idx = .{ .seq = ztRangeToAfSeq(idx.index_.Range) },
        },
        .Literal => .{
            .isSeq = true,
            .isBatch = false,
            .idx = .{
                .seq = makeSeq(@floatFromInt(idx.index_.Literal), @floatFromInt(idx.index_.Literal), 1),
            },
        },
    };
}

/// Display ArrayFire and device info.
pub inline fn info() !void {
    try af.AF_CHECK(af.af_info(), @src());
}

pub inline fn init() !void {
    try af.AF_CHECK(af.af_init(), @src());
}

/// Gets the value of `info` as a string.
pub inline fn infoString(verbose: bool) ![]const u8 {
    const info_: [*c]u8 = undefined;
    try af.AF_CHECK(af.af_info_string(info_, verbose), @src());
    return std.mem.span(info_);
}

/// Returns struct containing the following fields:
/// - `d_name`: Name of the device.
/// - `d_platform`: Platform of the device.
/// - `d_toolkit`: Toolkit version of the device.
/// - `d_compute`: Compute version of the device.
pub inline fn deviceInfo() !struct {
    d_name: []const u8,
    d_platform: []const u8,
    d_toolkit: []const u8,
    d_compute: []const u8,
} {
    const d_name: [*c]u8 = undefined;
    const d_platform: [*c]u8 = undefined;
    const d_toolkit: [*c]u8 = undefined;
    const d_compute: [*c]u8 = undefined;
    try af.AF_CHECK(af.af_device_info(d_name, d_platform, d_toolkit, d_compute), @src());
    return .{
        .d_name = std.mem.span(d_name),
        .d_platform = std.mem.span(d_platform),
        .d_toolkit = std.mem.span(d_toolkit),
        .d_compute = std.mem.span(d_compute),
    };
}

/// Returns the number of compute devices on the system.
pub inline fn getDeviceCount() !usize {
    var numDevices: c_int = 0;
    try af.AF_CHECK(af.af_get_device_count(&numDevices), @src());
    return @intCast(numDevices);
}

/// Returns true if the device supports double precision operations; else false.
pub inline fn getDoubleSupport() !bool {
    var support: bool = undefined;
    try af.AF_CHECK(af.af_get_dbl_support(&support), @src());
    return support;
}

/// Returns true if the device supports half precision operations; else false.
pub inline fn getHalfSupport(device: i32) !bool {
    var support: bool = undefined;
    try af.AF_CHECK(af.af_get_half_support(&support, device), @src());
    return support;
}

/// Sets the current device.
pub inline fn setDevice(id: i32) !void {
    try af.AF_CHECK(af.af_set_device(@intCast(id)), @src());
}

/// Returns the id of the current device.
pub inline fn getDevice() !i32 {
    var device: c_int = 0;
    try af.AF_CHECK(af.af_get_device(&device), @src());
    return @intCast(device);
}

/// Blocks the calling thread until all of the device operations have finished.
pub inline fn sync(device: i32) !void {
    try af.AF_CHECK(af.af_sync(@intCast(device)), @src());
}

/// Deprecated: use `allocDeviceV2`.
///
/// Allocates memory using ArrayFire's memory manager.
pub inline fn allocDevice(ptr: ?*anyopaque, bytes: usize) !void {
    try af.AF_CHECK(af.af_alloc_device(&ptr, @intCast(bytes)), @src());
}

/// Deprecated: use `freeDeviceV2`.
///
/// Free a device pointer even if it has been previously locked.
pub inline fn freeDevice(ptr: ?*anyopaque) !void {
    try af.AF_CHECK(af.af_free_device(ptr), @src());
}

/// Allocates memory using ArrayFire's memory manager.
pub inline fn allocDeviceV2(ptr: ?*anyopaque, bytes: usize) !void {
    try af.AF_CHECK(af.af_alloc_device_v2(&ptr, @intCast(bytes)), @src());
}

/// Free a device pointer even if it has been previously locked.
pub inline fn freeDeviceV2(ptr: ?*anyopaque) !void {
    try af.AF_CHECK(af.af_free_device_v2(ptr), @src());
}

/// Allocate pinned memory using ArrayFire's memory manager.
/// These functions allocate page locked memory. This type of
/// memory has better performance characteristics but require
/// additional care because they are a limited resource.
pub inline fn allocPinned(ptr: ?*anyopaque, bytes: usize) !void {
    try af.AF_CHECK(af.af_alloc_pinned(&ptr, @intCast(bytes)), @src());
}

/// Free pinned memory allocated by ArrayFire's memory manager.
/// These calls free the pinned memory on host. These functions
/// need to be called on pointers allocated using pinned function.
pub inline fn freePinned(ptr: ?*anyopaque) !void {
    try af.AF_CHECK(af.af_free_pinned(ptr), @src());
}

/// Allocate memory on host. This function is used for allocating
/// regular memory on host. This is useful where the compiler version
/// of ArrayFire library is different from the executable's compiler version.
///
/// It does not use ArrayFire's memory manager.
pub inline fn allocHost(ptr: ?*anyopaque, bytes: usize) !void {
    try af.AF_CHECK(af.af_alloc_host(&ptr, @intCast(bytes)), @src());
}

/// Free memory allocated on host internally by ArrayFire.
/// This function is used for freeing memory on host that
/// was allocated within ArrayFire. This is useful where the
/// compiler version of ArrayFire library is different from
/// the executable's compiler version.
///
/// It does not use ArrayFire's memory manager.
pub inline fn freeHost(ptr: ?*anyopaque) !void {
    try af.AF_CHECK(af.af_free_host(ptr), @src());
}

/// Data structure that holds memory information from the memory manager.
pub const DeviceMemInfo = struct {
    alloc_bytes: usize = undefined,
    alloc_buffers: usize = undefined,
    lock_bytes: usize = undefined,
    lock_buffers: usize = undefined,
};

/// Get memory information from the memory manager.
pub inline fn deviceMemInfo() !DeviceMemInfo {
    var mem = DeviceMemInfo{};
    try af.AF_CHECK(af.af_device_mem_info(&mem.alloc_bytes, &mem.alloc_buffers, &mem.lock_bytes, &mem.lock_buffers), @src());
    return mem;
}

/// Prints buffer details from the ArrayFire Device Manager.
pub inline fn printMemInfo(msg: [*c]const u8, device_id: i32) !void {
    try af.AF_CHECK(af.af_print_mem_info(msg, @intCast(device_id)), @src());
}

/// Call the garbage collection routine.
pub inline fn deviceGC() !void {
    try af.AF_CHECK(af.af_device_gc(), @src());
}

/// Set the resolution of memory chunks.
pub inline fn setMemStepSize(step_bytes: usize) !void {
    try af.AF_CHECK(af.af_set_mem_step_size(step_bytes), @src());
}

/// Get the resolution of memory chunks.
pub inline fn getMemStepSize() !usize {
    var step_bytes: usize = undefined;
    try af.AF_CHECK(af.af_get_mem_step_size(&step_bytes), @src());
    return step_bytes;
}

/// Sets the path where the kernels generated at runtime will be cached.
pub inline fn setKernelCacheDir(path: []const u8, override_env: i32) !void {
    try af.AF_CHECK(af.af_set_kernel_cache_directory(path.ptr, @intCast(override_env)), @src());
}

/// Gets the path where the kernels generated at runtime will be cached.
pub inline fn getKernelCacheDir() ![]const u8 {
    var path: [*c]u8 = undefined;
    var length: usize = undefined;
    try af.AF_CHECK(af.af_get_kernel_cache_directory(&length, path), @src());
    return path[0..length];
}

/// Returns the last error message that occurred and its error message.
pub inline fn getLastError() []const u8 {
    var err: [*c]u8 = undefined;
    var len: af.dim_t = undefined;
    af.af_get_last_error(&err, &len);
    return err[0..@intCast(len)];
}

/// Converts the af_err error code to its string representation.
pub inline fn errToString(err: af.af_err) []const u8 {
    const str = af.af_err_to_string(err);
    return std.mem.span(str);
}

/// Create a new `af.af_seq` object.
pub inline fn makeSeq(begin: f64, end: f64, step: f64) af.af_seq {
    return af.af_seq{ .begin = begin, .end = end, .step = step };
}

/// Create an quadruple of `af.af_index_t` array.
pub inline fn createIndexers() ![]af.af_index_t {
    var indexers: [*c]af.af_index_t = undefined;
    try af.AF_CHECK(af.af_create_indexers(&indexers), @src());
    return indexers[0..4];
}

/// Set dim to given indexer `af.Array` idx.
pub inline fn setArrayIndexer(
    indexer: []af.af_index_t,
    idx: *const af.Array,
    dim: i64,
) !void {
    try af.AF_CHECK(
        af.af_set_array_indexer(
            indexer.ptr,
            idx.array_,
            @intCast(dim),
        ),
        @src(),
    );
}

/// Set dim to given indexer `af.af_array` idx.
///
/// This function is similar to `setArrayIndexer` in terms of
/// functionality except that this version accepts object of type
/// `af.af_seq` instead of `af.Array`.
pub inline fn setSeqIndexer(
    indexer: []af.af_index_t,
    idx: *const af.af_seq,
    dim: i64,
    is_batch: bool,
) !void {
    try af.AF_CHECK(
        af.af_set_seq_indexer(
            indexer.ptr,
            idx,
            @intCast(dim),
            is_batch,
        ),
        @src(),
    );
}

/// Set dim to given indexer `af.af_array` idx.
///
/// This function is alternative to `setSeqIndexer` where instead
/// of passing in an already prepared af_seq object, you pass the
/// arguments necessary for creating an `af.af_seq` directly.
pub inline fn setSeqParamIndexer(
    indexer: []af.af_index_t,
    begin: f64,
    end: f64,
    step: f64,
    dim: i64,
    is_batch: bool,
) !void {
    try af.AF_CHECK(
        af.af_set_seq_param_indexer(
            indexer.ptr,
            begin,
            end,
            step,
            @intCast(dim),
            is_batch,
        ),
        @src(),
    );
}

/// Releases the memory resources used by the quadruple `af.af_index_t` array.
pub inline fn releaseIndexers(indexer: []af.af_index_t) !void {
    try af.AF_CHECK(af.af_release_indexers(indexer.ptr), @src());
}

/// Data structure holding ArrayFire library version information.
pub const Version = struct {
    major: i32 = undefined,
    minor: i32 = undefined,
    patch: i32 = undefined,
};

/// Returns the version information of the library.
pub inline fn getVersion() !Version {
    var major: c_int = undefined;
    var minor: c_int = undefined;
    var patch: c_int = undefined;
    try af.AF_CHECK(
        af.af_get_version(
            &major,
            &minor,
            &patch,
        ),
        @src(),
    );
    return .{
        .major = @intCast(major),
        .minor = @intCast(minor),
        .patch = @intCast(patch),
    };
}

/// Get the revision (commit) information of the library.
/// This returns a constant string from compile time and
/// should not be freed by the user.
pub inline fn getRevision() ![]const u8 {
    const msg = af.af_get_revision();
    return std.mem.span(msg);
}

/// Get the size of the type represented by the `af.Dtype` enum.
pub inline fn getSizeOf(dtype: af.Dtype) !usize {
    var size: usize = undefined;
    try af.AF_CHECK(af.af_get_size_of(&size, dtype.value()), @src());
    return size;
}

/// Enable(default) or disable error messages that display the stacktrace.
pub inline fn setEnableStackTrace(is_enabled: i32) !void {
    try af.AF_CHECK(af.af_set_enable_stacktrace(@intCast(is_enabled)), @src());
}

/// Turn the manual eval flag on or off.
pub inline fn setManualEvalFlag(flag: bool) !void {
    try af.AF_CHECK(af.af_set_manual_eval_flag(flag), @src());
}

/// Get the manual eval flag.
pub inline fn getManualEvalFlag() !bool {
    var flag: bool = undefined;
    try af.AF_CHECK(af.af_get_manual_eval_flag(&flag), @src());
    return flag;
}

/// Set the current backend when using Unified backend.
///
/// This is a noop when using one of CPU, CUDA, or OpenCL backend.
///
/// However, when using on of those 3 but trying to set it to a different
/// backend the function will throw an error.
pub inline fn setBackend(bknd: af.Backend) !void {
    try af.AF_CHECK(af.af_set_backend(bknd.value()), @src());
}

/// Get the number of backends whose libraries were successfully loaded.
/// This will be between 0-3. 0 being no backends were loaded and 3 being
/// all backends loaded successfully.
pub inline fn getBackendCount() !u32 {
    var num_backends: c_int = undefined;
    try af.AF_CHECK(af.af_get_backend_count(&num_backends), @src());
    return @intCast(num_backends);
}

/// Returns a flag of all available backends.
pub inline fn getAvailableBackends() i32 {
    var backends: c_int = undefined;
    try af.AF_CHECK(af.af_get_available_backends(&backends), @src());
    return @intCast(backends);
}

/// Returns the `AfBackend` enum for the active backend.
pub inline fn getActiveBackend() !af.Backend {
    var backend: af.af_backend = undefined;
    try af.AF_CHECK(af.af_get_active_backend(&backend), @src());
    return @enumFromInt(backend);
}

/// Sets plan cache size.
pub inline fn setFftPlanCacheSize(cache_size: usize) !void {
    try af.AF_CHECK(af.af_set_fft_plan_cache_size(cache_size), @src());
}

/// Returns true is ArrayFire is compiled with LAPACK support.
pub inline fn isLAPACKAvailable() !bool {
    var res: bool = undefined;
    try af.AF_CHECK(af.af_is_lapack_available(&res), @src());
    return res;
}

test "ztToAfBorderType" {
    try std.testing.expect(ztToAfBorderType(.Constant) == af.BorderType.PadZero);
    try std.testing.expect(ztToAfBorderType(.Edge) == af.BorderType.PadClampToEdge);
    try std.testing.expect(ztToAfBorderType(.Symmetric) == af.BorderType.PadSym);
}

test "ztToAfLocation" {
    try std.testing.expect(ztToAfLocation(.Host) == af.Source.Host);
    try std.testing.expect(ztToAfLocation(.Device) == af.Source.Device);
}

test "ztToAfStorageType" {
    try std.testing.expect(ztToAfStorageType(.Dense) == af.Storage.Dense);
    try std.testing.expect(ztToAfStorageType(.CSR) == af.Storage.CSR);
    try std.testing.expect(ztToAfStorageType(.CSC) == af.Storage.CSC);
    try std.testing.expect(ztToAfStorageType(.COO) == af.Storage.COO);
}

test "ztToAfMatrixProperty" {
    try std.testing.expect(ztToAfMatrixProperty(.None) == af.MatProp.None);
    try std.testing.expect(ztToAfMatrixProperty(.Transpose) == af.MatProp.Trans);
}

test "ztToAfDims" {
    var res1 = try ztToAfDims(&.{2});
    var exp1 = [_]c_longlong{ 2, 1, 1, 1 };
    try std.testing.expectEqualSlices(c_longlong, &exp1, &res1.dims);

    var res2 = try ztToAfDims(&.{ 2, 3 });
    var exp2 = [_]c_longlong{ 2, 3, 1, 1 };
    try std.testing.expectEqualSlices(c_longlong, &exp2, &res2.dims);

    var res3 = try ztToAfDims(&.{ 2, 3, 4 });
    var exp3 = [_]c_longlong{ 2, 3, 4, 1 };
    try std.testing.expectEqualSlices(c_longlong, &exp3, &res3.dims);

    var res4 = try ztToAfDims(&.{ 2, 3, 4, 5 });
    var exp4 = [_]c_longlong{ 2, 3, 4, 5 };
    try std.testing.expectEqualSlices(c_longlong, &exp4, &res4.dims);
}

test "ztToAfType" {
    try std.testing.expect(ztToAfType(.f16) == af.Dtype.f16);
    try std.testing.expect(ztToAfType(.f32) == af.Dtype.f32);
    try std.testing.expect(ztToAfType(.f64) == af.Dtype.f64);
    try std.testing.expect(ztToAfType(.b8) == af.Dtype.b8);
    try std.testing.expect(ztToAfType(.s16) == af.Dtype.s16);
    try std.testing.expect(ztToAfType(.s32) == af.Dtype.s32);
    try std.testing.expect(ztToAfType(.s64) == af.Dtype.s64);
    try std.testing.expect(ztToAfType(.u8) == af.Dtype.u8);
    try std.testing.expect(ztToAfType(.u16) == af.Dtype.u16);
    try std.testing.expect(ztToAfType(.u32) == af.Dtype.u32);
    try std.testing.expect(ztToAfType(.u64) == af.Dtype.u64);
}
