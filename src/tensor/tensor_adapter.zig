const std = @import("std");
const zigrc = @import("zigrc");
const zt = @import("../zt.zig");

const assert = std.debug.assert;
const DType = zt.tensor.DType;
const Index = zt.tensor.Index;
const Location = zt.tensor.Location;
const Shape = zt.tensor.shape.Shape;
const Tensor = zt.tensor.Tensor;
const TensorBackend = zt.tensor.TensorBackend;
const TensorBackendType = zt.tensor.TensorBackendType;
const Stream = zt.runtime.Stream;

pub const TensorAdapterBase = struct {
    const Self = @This();
    // The type erased pointer to the Tensor implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
        clone: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!Self,
        copy: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!Tensor,
        shallowCopy: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!Tensor,
        backendType: *const fn (ctx: *anyopaque) TensorBackendType,
        backend: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!TensorBackend,
        shape: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!Shape,
        dtype: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!DType,
        isSparse: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!bool,
        location: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!Location,
        scalar: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, out: ?*anyopaque) anyerror!void,
        device: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, out: *?*anyopaque) anyerror!void,
        host: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, out: ?*anyopaque) anyerror!void,
        unlock: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!void,
        isLocked: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!bool,
        isContiguous: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!bool,
        strides: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!Shape,
        stream: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!Stream,
        astype: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, dType: DType) anyerror!Tensor,
        index: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, indices: []const Index) anyerror!Tensor,
        flatten: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!Tensor,
        flat: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, idx: Index) anyerror!Tensor,
        asContiguousTensor: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!Tensor,
        setContext: *const fn (ctx: *anyopaque, context: ?*anyopaque) anyerror!void,
        getContext: *const fn (ctx: *anyopaque) anyerror!?*anyopaque,
        toString: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]const u8,
        // TODO: format: *const fn (value: anyopaque, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) anyerror!void,
    };

    /// Free all associated memory.
    pub fn deinit(self: *const Self) void {
        self.vtable.deinit(self.ptr);
    }

    /// Copies the tensor adapter. The copy is not required to be eager -- the
    /// implementation can use copy-on-write semantics if desirable. (TODO: verify this
    /// works as documented).
    pub fn clone(self: *const Self, allocator: std.mem.Allocator) !Self {
        return self.vtable.clone(self.ptr, allocator);
    }

    /// Deep copy the tensor, including underlying data.
    pub fn copy(self: *const Self, allocator: std.mem.Allocator) !Tensor {
        return self.vtable.copy(self.ptr, allocator);
    }

    /// Shallow copy the tensor -- returns a tensor that points
    /// to the same underlying data.
    pub fn shallowCopy(self: *const Self, allocator: std.mem.Allocator) !Tensor {
        return self.vtable.shallowCopy(self.ptr, allocator);
    }

    /// Returns the TensorBackendType enum associated with the backend.
    pub fn backendType(self: *const Self) TensorBackendType {
        return self.vtable.backendType(self.ptr);
    }

    /// Returns the backend for a tensor with this adapter implementation.
    pub fn backend(self: *const Self, allocator: std.mem.Allocator) !TensorBackend {
        return self.vtable.backend(self.ptr, allocator);
    }

    /// Returns the shape of the tensor.
    pub fn shape(self: *const Self, allocator: std.mem.Allocator) !Shape {
        return self.vtable.shape(self.ptr, allocator);
    }

    /// Returns the data type (DType) of the tensor.
    pub fn dtype(self: *const Self, allocator: std.mem.Allocator) !DType {
        return self.vtable.dtype(self.ptr, allocator);
    }

    /// Returns true if the tensor is sparse, else false.
    pub fn isSparse(self: *const Self, allocator: std.mem.Allocator) !bool {
        return self.vtable.isSparse(self.ptr, allocator);
    }

    /// Returns the tensor's location -- host or some device.
    pub fn location(self: *const Self, allocator: std.mem.Allocator) !Location {
        return self.vtable.location(self.ptr, allocator);
    }

    /// Populate a pointer with a scalar for the first element of the tensor.
    pub fn scalar(self: *const Self, allocator: std.mem.Allocator, comptime T: type) !T {
        var res: T = undefined;
        try self.vtable.scalar(self.ptr, allocator, &res);
        return res;
    }

    /// Returns a pointer to the tensor in device memory.
    pub fn device(self: *const Self, allocator: std.mem.Allocator, out: *?*anyopaque) !void {
        return self.vtable.device(self.ptr, allocator, out);
    }

    /// Populates a pointer with a pointer value in memory pointing
    /// to a host buffer containing tensor data.
    pub fn host(self: *const Self, allocator: std.mem.Allocator, out: ?*anyopaque) !void {
        return self.vtable.host(self.ptr, allocator, out);
    }

    /// Unlocks any device memory associated with the tensor that was
    /// acquired with `Tensor.device`, making it eligible to be freed.
    pub fn unlock(self: *const Self, allocator: std.mem.Allocator) !void {
        return self.vtable.unlock(self.ptr, allocator);
    }

    /// Returns true if the tensor has been memory-locked per a call to `Tensor.device`.
    pub fn isLocked(self: *const Self, allocator: std.mem.Allocator) !bool {
        return self.vtable.isLocked(self.ptr, allocator);
    }

    /// Returns a bool based on tensor contiguousness in memory.
    pub fn isContiguous(self: *const Self, allocator: std.mem.Allocator) !bool {
        return self.vtable.isContiguous(self.ptr, allocator);
    }

    /// Returns the dimension-wise strides for this tensor -- the number of bytes
    /// to step in each direction when traversing.
    pub fn strides(self: *const Self, allocator: std.mem.Allocator) !Shape {
        return self.vtable.strides(self.ptr, allocator);
    }

    /// Returns an immutable reference to the stream that contains, or did contain,
    /// the computation required to realize an up-to-date value for this tensor.
    /// E.g. `device()` may not yield a pointer to the up-to-date value -- to use
    /// this pointer, `Stream.sync` or `Stream.relativeSync` is required.
    pub fn stream(self: *const Self, allocator: std.mem.Allocator) !Stream {
        return self.vtable.stream(self.ptr, allocator);
    }

    /// Returns a tensor with elements cast as a particular type.
    pub fn astype(self: *const Self, allocator: std.mem.Allocator, dType: DType) !Tensor {
        return self.vtable.astype(self.ptr, allocator, dType);
    }

    /// Index into a tensor with a variable number of indices.
    ///
    /// Returns an indexed Tensor.
    pub fn index(self: *const Self, allocator: std.mem.Allocator, indices: []const Index) !Tensor {
        return self.vtable.index(self.ptr, allocator, indices);
    }

    /// Returns a representation of the tensor in 1 dimension.
    pub fn flatten(self: *const Self, allocator: std.mem.Allocator) !Tensor {
        return self.vtable.flatten(self.ptr, allocator);
    }

    /// Returns a representation of the tensor in 1 dimension.
    pub fn flat(self: *const Self, allocator: std.mem.Allocator, idx: Index) !Tensor {
        return self.vtable.flat(self.ptr, allocator, idx);
    }

    /// Returns a copy of the tensor that is contiguous in memory.
    pub fn asContiguousTensor(self: *const Self, allocator: std.mem.Allocator) !Tensor {
        return self.vtable.asContiguousTensor(self.ptr, allocator);
    }

    /// Sets arbitrary data on a tensor. May be a no-op for some backends.
    pub fn setContext(self: *const Self, context: ?*anyopaque) !void {
        return self.vtable.setContext(self.ptr, context);
    }

    /// Returns arbitrary data on a tensor. May be a no-op for some backends.
    pub fn getContext(self: *const Self) !?*anyopaque {
        return self.vtable.getContext(self.ptr);
    }

    /// Returns a string representation of a tensor. Not intended to be
    /// portable across backends.
    pub fn toString(self: *const Self, allocator: std.mem.Allocator) ![]const u8 {
        return self.vtable.toString(self.ptr, allocator);
    }

    /// Initializes a new `TensorAdapter`.
    pub fn init(tensor_impl: anytype) Self {
        const Ptr = @TypeOf(tensor_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }

            fn clone(ctx: *anyopaque, allocator: std.mem.Allocator) !Self {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.clone(allocator);
            }

            fn copy(ctx: *anyopaque, allocator: std.mem.Allocator) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.copy(allocator);
            }

            fn shallowCopy(ctx: *anyopaque, allocator: std.mem.Allocator) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.shallowCopy(allocator);
            }

            fn backend(ctx: *anyopaque, allocator: std.mem.Allocator) !TensorBackend {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.backend(allocator);
            }

            fn backendType(ctx: *anyopaque) TensorBackendType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.backendType();
            }

            fn shape(ctx: *anyopaque, allocator: std.mem.Allocator) !Shape {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.shape(allocator);
            }

            fn dtype(ctx: *anyopaque, allocator: std.mem.Allocator) !DType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.dtype(allocator);
            }

            fn isSparse(ctx: *anyopaque, allocator: std.mem.Allocator) !bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isSparse(allocator);
            }

            fn location(ctx: *anyopaque, allocator: std.mem.Allocator) !Location {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.location(allocator);
            }

            fn scalar(ctx: *anyopaque, allocator: std.mem.Allocator, out: ?*anyopaque) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.scalar(allocator, out);
            }

            fn device(ctx: *anyopaque, allocator: std.mem.Allocator, out: *?*anyopaque) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.device(allocator, out);
            }

            fn host(ctx: *anyopaque, allocator: std.mem.Allocator, out: ?*anyopaque) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.host(allocator, out);
            }

            fn unlock(ctx: *anyopaque, allocator: std.mem.Allocator) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.unlock(allocator);
            }

            fn isLocked(ctx: *anyopaque, allocator: std.mem.Allocator) !bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isLocked(allocator);
            }

            fn isContiguous(ctx: *anyopaque, allocator: std.mem.Allocator) !bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isContiguous(allocator);
            }

            fn strides(ctx: *anyopaque, allocator: std.mem.Allocator) !Shape {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.strides(allocator);
            }

            fn stream(ctx: *anyopaque, allocator: std.mem.Allocator) !Stream {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.stream(allocator);
            }

            fn astype(ctx: *anyopaque, allocator: std.mem.Allocator, dType: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.astype(allocator, dType);
            }

            fn index(ctx: *anyopaque, allocator: std.mem.Allocator, indices: []const Index) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.index(allocator, indices);
            }

            fn flatten(ctx: *anyopaque, allocator: std.mem.Allocator) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.flatten(allocator);
            }

            fn flat(ctx: *anyopaque, allocator: std.mem.Allocator, idx: Index) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.flat(allocator, idx);
            }

            fn asContiguousTensor(ctx: *anyopaque, allocator: std.mem.Allocator) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.asContiguousTensor(allocator);
            }

            fn setContext(ctx: *anyopaque, context: ?*anyopaque) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                try self.setContext(context);
            }

            fn getContext(ctx: *anyopaque) !?*anyopaque {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getContext();
            }

            fn toString(ctx: *anyopaque, allocator: std.mem.Allocator) ![]const u8 {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.toString(allocator);
            }
        };
        return .{
            .ptr = tensor_impl,
            .vtable = &.{
                .deinit = impl.deinit,
                .clone = impl.clone,
                .backendType = impl.backendType,
                .backend = impl.backend,
                .copy = impl.copy,
                .shallowCopy = impl.shallowCopy,
                .shape = impl.shape,
                .dtype = impl.dtype,
                .isSparse = impl.isSparse,
                .location = impl.location,
                .scalar = impl.scalar,
                .device = impl.device,
                .host = impl.host,
                .unlock = impl.unlock,
                .isLocked = impl.isLocked,
                .isContiguous = impl.isContiguous,
                .strides = impl.strides,
                .stream = impl.stream,
                .astype = impl.astype,
                .index = impl.index,
                .flatten = impl.flatten,
                .flat = impl.flat,
                .asContiguousTensor = impl.asContiguousTensor,
                .setContext = impl.setContext,
                .getContext = impl.getContext,
                .toString = impl.toString,
            },
        };
    }
};
