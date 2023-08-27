const std = @import("std");
const zt_base = @import("TensorBase.zig");
const zt_shape = @import("Shape.zig");
const zt_tensor_backend = @import("TensorBackend.zig");
const zigrc = @import("zigrc");
const zt_type = @import("Types.zig");
const rt_stream = @import("../runtime/Stream.zig");

const assert = std.debug.assert;
const Shape = zt_shape.Shape;
const Tensor = zt_base.Tensor;
const Location = zt_base.Location;
const TensorBackendType = zt_base.TensorBackendType;
const TensorBackend = zt_tensor_backend.TensorBackend;
const DType = zt_type.DType;
const Stream = rt_stream.Stream;

pub const TensorAdapterBase = struct {
    const Self = @This();
    // The type erased pointer to the TensorBackend implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
        clone: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) zigrc.Arc(Self),
        backendType: *const fn (ctx: *anyopaque) TensorBackendType,
        backend: *const fn (ctx: *anyopaque) TensorBackend,
        copy: *const fn (ctx: *anyopaque) Tensor,
        shallowCopy: *const fn (ctx: *anyopaque) Tensor,
        shape: *const fn (ctx: *anyopaque) *const Shape,
        dtype: *const fn (ctx: *anyopaque) DType,
        isSparse: *const fn (ctx: *anyopaque) bool,
        location: *const fn (ctx: *anyopaque) Location,
        scalar: *const fn (ctx: *anyopaque, out: *anyopaque) void,
        device: *const fn (ctx: *anyopaque, out: **anyopaque) void,
        host: *const fn (ctx: *anyopaque, out: *anyopaque) void,
        unlock: *const fn (ctx: *anyopaque) void,
        isLocked: *const fn (ctx: *anyopaque) bool,
        isContiguous: *const fn (ctx: *anyopaque) bool,
        strides: *const fn (ctx: *anyopaque) Shape,
        stream: *const fn (ctx: *anyopaque) *Stream,
        // TODO: `astype` might need allocator passed in?
        astype: *const fn (ctx: *anyopaque, data_type: DType) Tensor,
        // TODO - might need allocator: index: *const fn (ctx: *anyopaque, indices: *std.ArrayList(Index)) Tensor
        flatten: *const fn (ctx: *anyopaque) Tensor,
        // TODO - might need allocator: flat: *const fn (ctx: *anyopaque, idx: *Index) Tensor
        // TODO: might need allocator
        asContiguousTensor: *const fn (ctx: *anyopaque) Tensor,
        setContext: *const fn (ctx: *anyopaque, comptime T: type, context: *anyopaque) void,
        getContext: *const fn (ctx: *anyopaque, comptime T: type) ?*anyopaque,
        toString: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]const u8,
        // TODO: format: *const fn (value: *anyopaque, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) anyerror!void,
        assign: *const fn (ctx: *anyopaque, comptime T: type, value: anytype) void,
        inPlaceAdd: *const fn (ctx: *anyopaque, comptime T: type, value: anytype) void,
        inPlaceSubtract: *const fn (ctx: *anyopaque, comptime T: type, value: anytype) void,
        inPlaceMultiply: *const fn (ctx: *anyopaque, comptime T: type, value: anytype) void,
        inPlaceDivide: *const fn (ctx: *anyopaque, comptime T: type, value: anytype) void,
    };

    /// Free all associated memory.
    pub fn deinit(self: *Self) void {
        self.vtable.deinit(self.ptr);
    }

    /// Copies the tensor adapter. The copy is not required to be eager -- the
    /// implementation can use copy-on-write semantics if desirable. (TODO: verify this
    /// works as documented).
    pub fn clone(self: *Self, allocator: std.mem.Allocator) !zigrc.Arc(Self) {
        return self.vtable.clone(self.ptr, allocator);
    }

    /// Returns the TensorBackendType enum associated with the backend.
    pub fn backendType(self: *Self) TensorBackendType {
        return self.vtable.backendType(self.ptr);
    }

    /// Returns the backend for a tensor with this adapter implementation.
    pub fn backend(self: *Self) TensorBackend {
        return self.vtable.backend(self.ptr);
    }

    /// Deep copy the tensor, including underlying data.
    pub fn copy(self: *Self) Tensor {
        return self.vtable.copy(self.ptr);
    }

    /// Shallow copy the tensor -- returns a tensor that points
    /// to the same underlying data.
    pub fn shallowCopy(self: *Self) Tensor {
        return self.vtable.shallowCopy(self.ptr);
    }

    /// Returns the shape of the tensor.
    pub fn shape(self: *Self) *const Shape {
        return self.vtable.shape(self.ptr);
    }

    /// Returns the data type (DType) of the tensor.
    pub fn dtype(self: *Self) DType {
        return self.vtable.dtype(self.ptr);
    }

    /// Returns true if the tensor is sparse, else false.
    pub fn isSparse(self: *Self) bool {
        return self.vtable.isSparse(self.ptr);
    }

    /// Returns the tensor's location -- host or some device.
    pub fn location(self: *Self) Location {
        return self.vtable.location(self.ptr);
    }

    /// Populate a pointer with a scalar for the first element of the tensor.
    pub fn scalar(self: *Self, out: *anyopaque) void {
        return self.vtable.scalar(self.ptr, out);
    }

    /// Returns a pointer to the tensor in device memory.
    pub fn device(self: *Self, out: **anyopaque) void {
        return self.vtable.device(self.ptr, out);
    }

    /// Populates a pointer with a pointer value in memory pointing
    /// to a host buffer containing tensor data.
    pub fn host(self: *Self, out: *anyopaque) void {
        return self.vtable.host(self.ptr, out);
    }

    /// Unlocks any device memory associated with the tensor that was
    /// acquired with `Tensor.device`, making it eligible to be freed.
    pub fn unlock(self: *Self) void {
        return self.vtable.unlock(self.ptr);
    }

    /// Returns true if the tensor has been memory-locked per a call to `Tensor.device`.
    pub fn isLocked(self: *Self) bool {
        return self.vtable.isLocked(self.ptr);
    }

    /// Returns a bool based on tensor contiguousness in memory.
    pub fn isContiguous(self: *Self) bool {
        return self.vtable.isContiguous(self.ptr);
    }

    /// Returns the dimension-wise strides for this tensor -- the number of bytes
    /// to step in each direction when traversing.
    pub fn strides(self: *Self) Shape {
        return self.vtable.strides(self.ptr);
    }

    /// Returns an immutable reference to the stream that contains, or did contain,
    /// the computation required to realize an up-to-date value for this tensor.
    /// E.g. `device()` may not yield a pointer to the up-to-date value -- to use
    /// this pointer, `Stream.sync` or `Stream.relativeSync` is required.
    pub fn stream(self: *Self) *Stream {
        return self.vtable.stream(self.ptr);
    }

    // TODO: might need to pass in allocator
    /// Returns a tensor with elements cast as a particular type.
    pub fn astype(self: *Self, data_type: DType) Tensor {
        return self.vtable.astype(self.ptr, data_type);
    }

    // TODO: might need to pass in allocator
    /// Returns a representation of the tensor in 1 dimension.
    pub fn flatten(self: *Self) Tensor {
        return self.vtable.flatten(self.ptr);
    }

    /// Returns a copy of the tensor that is contiguous in memory.
    pub fn asContiguousTensor(self: *Self) Tensor {
        return self.vtable.asContiguousTensor(self.ptr);
    }

    /// Sets arbitrary data on a tensor. May be a no-op for some backends.
    pub fn setContext(self: *Self, comptime T: type, context: *T) void {
        return self.vtable.setContext(self.ptr, T, context);
    }

    /// Returns arbitrary data on a tensor. May be a no-op for some backends.
    pub fn getContext(self: *Self, comptime T: type) ?*T {
        return self.vtable.getContext(self.ptr, T);
    }

    /// Returns a string representation of a tensor. Not intended to be
    /// portable across backends.
    pub fn toString(self: *Self, allocator: std.mem.Allocator) anyerror![]const u8 {
        return self.vtable.toString(self.ptr, allocator);
    }

    /// Assigns value to given tensor (sets tensor as equal to specified value).
    pub fn assign(self: *Self, comptime T: type, value: T) void {
        return self.vtable.assign(self.ptr, T, value);
    }

    /// Adds the value from the tensor in place.
    pub fn inPlaceAdd(self: *Self, comptime T: type, value: T) void {
        return self.vtable.inPlaceAdd(self.ptr, T, value);
    }

    /// Subtracts the value from the tensor in place.
    pub fn inPlaceSubtract(self: *Self, comptime T: type, value: T) void {
        return self.vtable.inPlaceSubtract(self.ptr, T, value);
    }

    /// Multiplies the value with the tensor in place.
    pub fn inPlaceMultiply(self: *Self, comptime T: type, value: T) void {
        return self.vtable.inPlaceMultiply(self.ptr, T, value);
    }

    /// Divides the tensor in place by the value.
    pub fn inPlaceDivide(self: *Self, comptime T: type, value: T) void {
        return self.vtable.inPlaceDivide(self.ptr, T, value);
    }

    /// Initializes a new `TensorAdapter`.
    pub fn init(backend_impl: anytype) Self {
        const Ptr = @TypeOf(backend_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }

            fn clone(ctx: *anyopaque, allocator: std.mem.Allocator) zigrc.Arc(Self) {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return zigrc.Arc(Self).init(allocator, Self.init(self.clone(allocator)));
            }

            fn backendType(ctx: *anyopaque) TensorBackendType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.trialRunStarted();
            }

            fn backend(ctx: *anyopaque) TensorBackend {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.backend();
            }

            fn copy(ctx: *anyopaque) Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.copy();
            }

            fn shallowCopy(ctx: *anyopaque) Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.shallowCopy();
            }

            fn shape(ctx: *anyopaque) *const Shape {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.shape();
            }

            fn dtype(ctx: *anyopaque) DType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.dtype();
            }

            fn isSparse(ctx: *anyopaque) bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isSparse();
            }

            fn location(ctx: *anyopaque) Location {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.location();
            }

            fn scalar(ctx: *anyopaque, out: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.scalar(out);
            }

            fn isLocked(ctx: *anyopaque) bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isLocked();
            }

            fn isContiguous(ctx: *anyopaque) bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isContiguous();
            }

            fn strides(ctx: *anyopaque) Shape {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.strides();
            }

            fn stream(ctx: *anyopaque) *Stream {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.stream();
            }

            fn astype(ctx: *anyopaque, data_type: DType) Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.astype(data_type);
            }

            fn flatten(ctx: *anyopaque) Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.flatten();
            }

            fn asContiguousTensor(ctx: *anyopaque) Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.asContiguousTensor();
            }

            fn setContext(ctx: *anyopaque, comptime T: type, context: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.setContext(T, context);
            }

            fn getContext(ctx: *anyopaque, comptime T: type) *T {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getContext(T);
            }

            fn toString(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]const u8 {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.toString(allocator);
            }

            fn assign(ctx: *anyopaque, comptime T: type, value: T) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.assign(T, value);
            }

            fn inPlaceAdd(ctx: *anyopaque, comptime T: type, value: T) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.inPlaceAdd(T, value);
            }

            fn inPlaceSubtract(ctx: *anyopaque, comptime T: type, value: T) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.inPlaceSubtract(T, value);
            }

            fn inPlaceMultiply(ctx: *anyopaque, comptime T: type, value: T) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.inPlaceMultiply(T, value);
            }

            fn inPlaceDivide(ctx: *anyopaque, comptime T: type, value: T) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.inPlaceDivide(T, value);
            }
        };
        return .{
            .ptr = backend_impl,
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
                .isLocked = impl.isLocked,
                .isContiguous = impl.isContiguous,
                .strides = impl.strides,
                .stream = impl.stream,
                .astype = impl.astype,
                .flatten = impl.flatten,
                .asContiguousTensor = impl.asContiguousTensor,
                .setContext = impl.setContext,
                .getContext = impl.getContext,
                .toString = impl.toString,
                .assign = impl.assign,
            },
        };
    }
};
