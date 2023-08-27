const std = @import("std");
const adapter = @import("TensorAdapter.zig");
const zt_shape = @import("Shape.zig");
const zt_types = @import("Types.zig");
const zt_tensor_backend = @import("TensorBackend.zig");
const rt_stream = @import("../runtime/Stream.zig");

const Stream = rt_stream.Stream;
const TensorBackend = zt_tensor_backend.TensorBackend;
const Shape = zt_shape.Shape;
const DType = zt_types.DType;
const Dim = zt_shape.Dim;
const TensorAdapterBase = adapter.TensorAdapterBase;

/// Enum for various tensor backends.
pub const TensorBackendType = enum { ArrayFire };

/// Location of memory or tensors.
pub const Location = enum { Host, Device };

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
pub const PadType = enum {
    /// pad with a constant zero value.
    Constant,
    /// pad with the values at the edges of the tensor.
    Edge,
    /// pad with a reflection of the tensor mirrored along each edge.
    Symmetric,
};

pub const Tensor = struct {
    impl_: *TensorAdapterBase,

    // TODO: various Tensor constructors

    pub fn copy(self: *Tensor) Tensor {
        return self.impl_.copy();
    }

    pub fn shallowCopy(self: *Tensor) Tensor {
        return self.impl_.shallowCopy();
    }

    pub fn shape(self: *Tensor) *const Shape {
        return self.impl_.shape();
    }

    pub fn location(self: *Tensor) Location {
        return self.impl_.location();
    }

    pub fn elements(self: *Tensor) usize {
        return (self.shape()).elements();
    }

    pub fn dim(self: *Tensor, dimension: usize) Dim {
        return (self.shape()).dim(dimension);
    }

    pub fn ndim(self: *Tensor) usize {
        return (self.shape()).ndim();
    }

    pub fn isEmpty(self: *Tensor) bool {
        return self.elements() == 0;
    }

    pub fn hasAdapter(self: *Tensor) bool {
        return self.impl_.get() != null;
    }

    pub fn bytes(self: *Tensor) usize {
        return self.elements() * (self.dtype()).getSize();
    }

    pub fn dtype(self: *Tensor) DType {
        return self.impl_.dtype();
    }

    pub fn isSparse(self: *Tensor) bool {
        return self.impl_.isSparse();
    }

    pub fn astype(self: *Tensor, new_type: DType) Tensor {
        return self.impl_.astype(new_type);
    }

    // TODO: equivalent of operator for indexing

    pub fn flatten(self: *Tensor) Tensor {
        return self.impl_.flatten();
    }

    pub fn flat(self: *Tensor) Tensor {
        return self.impl_.flat();
    }

    pub fn asContiguousTensor(self: *Tensor) Tensor {
        return self.impl_.asContiguousTensor();
    }

    pub fn backendType(self: *Tensor) TensorBackendType {
        return self.impl_.backendType();
    }

    pub fn getAdapter(self: *Tensor, comptime T: type) *T {
        return @ptrCast(self.impl_.get());
    }

    pub fn backend(self: *Tensor) TensorBackend {
        return self.impl_.backend();
    }

    // TODO: FL_CREATE_MEMORY_OPS macro equivalent

    pub fn unlock(self: *Tensor) void {
        return self.impl_.unlock();
    }

    pub fn isLocked(self: *Tensor) bool {
        return self.impl_.isLocked();
    }

    pub fn isContiguous(self: *Tensor) bool {
        return self.impl_.isContiguous();
    }

    pub fn strides(self: *Tensor) Shape {
        return self.impl_.strides();
    }

    pub fn stream(self: *Tensor) Stream {
        return self.impl_.stream();
    }

    pub fn setContext(self: *Tensor, context: ?*anyopaque) void {
        return self.impl_.setContext(context);
    }

    pub fn getContext(self: *Tensor) ?*anyopaque {
        return self.impl_.getContext();
    }

    // TODO: pub fn toString(self: *Tensor, allocator: std.mem.Allocator) ![]const u8 {}

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
