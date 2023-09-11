const std = @import("std");
const adapter = @import("TensorAdapter.zig");
const zt_shape = @import("Shape.zig");
const zt_types = @import("Types.zig");
const zt_tensor_backend = @import("TensorBackend.zig");
const rt_stream = @import("../runtime/Stream.zig");
const af = @import("../bindings/af/ArrayFire.zig");
const zt_idx = @import("Index.zig");

const Index = zt_idx.Index;
const Stream = rt_stream.Stream;
const TensorBackend = zt_tensor_backend.TensorBackend;
const Shape = zt_shape.Shape;
const DType = zt_types.DType;
const Dim = zt_shape.Dim;
const TensorAdapterBase = adapter.TensorAdapterBase;

/// Enum for various tensor backends.
pub const TensorBackendType = enum(u8) { Stub, Tracer, ArrayFire, OneDnn, Jit };

/// Location of memory or tensors.
pub const Location = enum(u8) { Host, Device };

/// Alias to make it semantically clearer when referring to buffer location
const MemoryLocation = Location;

/// Tensor storage types.
pub const StorageType = enum(u8) { Dense = 0, CSR = 1, CSC = 2, COO = 3 };

/// Transformations to apply to Tensors (i.e. matrices) before applying certain
/// operations (i.e. matmul).
pub const MatrixProperty = enum(u8) { None = 0, Transpose = 1 };

/// Sorting mode for sorting-related functions.
pub const SortMode = enum(u8) { Descending = 0, Ascending = 1 };

/// Padding types for the pad operator.
pub const PadType = enum(u8) {
    /// pad with a constant zero value.
    Constant,
    /// pad with the values at the edges of the tensor.
    Edge,
    /// pad with a reflection of the tensor mirrored along each edge.
    Symmetric,
};

pub const Tensor = struct {
    impl_: TensorAdapterBase,

    pub fn init(impl: TensorAdapterBase) Tensor {
        return Tensor{ .impl_ = impl };
    }

    pub fn deinit(self: *Tensor) void {
        self.impl_.deinit();
    }

    pub fn copy(self: *Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.copy(allocator);
    }

    pub fn shallowCopy(self: *Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.shallowCopy(allocator);
    }

    pub fn shape(self: *Tensor, allocator: std.mem.Allocator) !Shape {
        return self.impl_.shape(allocator);
    }

    pub fn location(self: *Tensor, allocator: std.mem.Allocator) !Location {
        return self.impl_.location(allocator);
    }

    pub fn elements(self: *Tensor, allocator: std.mem.Allocator) !Dim {
        var shape_ = try self.shape(allocator);
        return shape_.elements();
    }

    pub fn dim(self: *Tensor, allocator: std.mem.Allocator, dimension: usize) !Dim {
        var shape_ = try self.shape(allocator);
        return shape_.dim(dimension);
    }

    pub fn ndim(self: *Tensor, allocator: std.mem.Allocator) !usize {
        var shape_ = try self.shape(allocator);
        return shape_.ndim();
    }

    pub fn isEmpty(self: *Tensor, allocator: std.mem.Allocator) !bool {
        var elements_ = try self.elements(allocator);
        return elements_ == 0;
    }

    // TODO: implement
    // pub fn hasAdapter(self: *Tensor) bool {
    // return self.impl_.get() != null;
    // }

    pub fn bytes(self: *Tensor, allocator: std.mem.Allocator) !usize {
        var elements_ = try self.elements(allocator);
        var dtype_ = try self.dtype(allocator);
        return elements_ * dtype_.getSize();
    }

    pub fn dtype(self: *Tensor, allocator: std.mem.Allocator) !DType {
        return self.impl_.dtype(allocator);
    }

    pub fn isSparse(self: *Tensor, allocator: std.mem.Allocator) !bool {
        return self.impl_.isSparse(allocator);
    }

    pub fn astype(self: *Tensor, allocator: std.mem.Allocator, new_type: DType) !Tensor {
        return self.impl_.astype(allocator, new_type);
    }

    pub fn index(self: *Tensor, allocator: std.mem.Allocator, indices: std.ArrayList(Index)) !Tensor {
        return self.impl_.index(allocator, indices);
    }

    pub fn flatten(self: *Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.flatten(allocator);
    }

    // TODO: implement
    // pub fn flat(self: *Tensor) Tensor {
    // return self.impl_.flat();
    // }

    pub fn asContiguousTensor(self: *Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.asContiguousTensor(allocator);
    }

    pub fn backendType(self: *const Tensor) TensorBackendType {
        return self.impl_.backendType();
    }

    pub fn getAdapter(self: *const Tensor, comptime T: type) *T {
        return @ptrCast(@alignCast(self.impl_.ptr));
    }

    pub fn backend(self: *Tensor, allocator: std.mem.Allocator) !TensorBackend {
        return self.impl_.backend(allocator);
    }

    // TODO: FL_CREATE_MEMORY_OPS macro equivalent

    pub fn unlock(self: *Tensor, allocator: std.mem.Allocator) !void {
        return self.impl_.unlock(allocator);
    }

    pub fn isLocked(self: *Tensor, allocator: std.mem.Allocator) !bool {
        return self.impl_.isLocked(allocator);
    }

    pub fn isContiguous(self: *Tensor, allocator: std.mem.Allocator) !bool {
        return self.impl_.isContiguous(allocator);
    }

    pub fn strides(self: *Tensor, allocator: std.mem.Allocator) !Shape {
        return self.impl_.strides(allocator);
    }

    pub fn stream(self: *Tensor, allocator: std.mem.Allocator) !Stream {
        return self.impl_.stream(allocator);
    }

    pub fn setContext(self: *Tensor, context: ?*anyopaque) !void {
        return self.impl_.setContext(context);
    }

    pub fn getContext(self: *Tensor) !?*anyopaque {
        return self.impl_.getContext();
    }

    pub fn toString(self: *Tensor, allocator: std.mem.Allocator) ![]const u8 {
        return self.impl_.toString(allocator);
    }

    // TODO: equivalent of assignment operators

};

// TODO: pub fn arange()

// TODO: pub fn iota()

//************************ Shaping and Indexing *************************//

pub fn reshape(tensor: *const Tensor, shape: *const Shape) Tensor {
    return (tensor.backend()).reshape(tensor, shape);
}

pub fn transpose(tensor: *const Tensor, axes: *const Shape) Tensor {
    return (tensor.backend()).transpose(tensor, axes);
}

pub fn tile(tensor: *const Tensor, shape: *const Shape) Tensor {
    return (tensor.backend()).tile(tensor, shape);
}

pub fn concatenate(tensors: *const std.ArrayList(Tensor), axis: u32) !Tensor {
    if (tensors.items.len == 0) {
        std.log.err("concatenate: called on empty set of tensors\n", .{});
        return error.ConcatFailedZeroTensors;
    }

    // ensure that all tensors have the same backend
    const b: TensorBackendType = tensors.items[0].backendType();
    var matches = true;
    for (tensors.items) |t| {
        if (t.backendType() != b) matches = false;
    }
    if (!matches) {
        std.log.err("concatenate: tried to concatenate tensors of different backends\n", .{});
        return error.ConcatFailedBackendMismatch;
    }
    return (tensors.items[0].backend()).concatenate(tensors, axis);
}

pub fn nonzero(tensor: *const Tensor) Tensor {
    return (tensor.backend()).nonzero(tensor);
}
