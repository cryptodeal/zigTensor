const std = @import("std");
const adapter = @import("TensorAdapter.zig");
const zt_shape = @import("Shape.zig");
const zt_types = @import("Types.zig");
const zt_tensor_backend = @import("TensorBackend.zig");
const rt_stream = @import("../runtime/Stream.zig");
const af = @import("../bindings/af/ArrayFire.zig");
const zt_idx = @import("Index.zig");

const defaultTensorBackend = @import("DefaultTensorType.zig").defaultTensorBackend;
const TopkRes = zt_tensor_backend.TopkRes;
const SortIndexRes = zt_tensor_backend.SortIndexRes;
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

    pub fn deinit(self: *const Tensor) void {
        self.impl_.deinit();
    }

    pub fn copy(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.copy(allocator);
    }

    pub fn shallowCopy(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.shallowCopy(allocator);
    }

    pub fn shape(self: *const Tensor, allocator: std.mem.Allocator) !Shape {
        return self.impl_.shape(allocator);
    }

    pub fn location(self: *const Tensor, allocator: std.mem.Allocator) !Location {
        return self.impl_.location(allocator);
    }

    pub fn elements(self: *const Tensor, allocator: std.mem.Allocator) !Dim {
        var shape_ = try self.shape(allocator);
        return shape_.elements();
    }

    pub fn dim(self: *const Tensor, allocator: std.mem.Allocator, dimension: usize) !Dim {
        var shape_ = try self.shape(allocator);
        return shape_.dim(dimension);
    }

    pub fn ndim(self: *const Tensor, allocator: std.mem.Allocator) !usize {
        var shape_ = try self.shape(allocator);
        return shape_.ndim();
    }

    pub fn isEmpty(self: *const Tensor, allocator: std.mem.Allocator) !bool {
        var elements_ = try self.elements(allocator);
        return elements_ == 0;
    }

    // TODO: implement
    // pub fn hasAdapter(self: *const Tensor) bool {
    // return self.impl_.get() != null;
    // }

    pub fn bytes(self: *const Tensor, allocator: std.mem.Allocator) !usize {
        var elements_: usize = @intCast(try self.elements(allocator));
        var dtype_ = try self.dtype(allocator);
        return elements_ * dtype_.getSize();
    }

    pub fn dtype(self: *const Tensor, allocator: std.mem.Allocator) !DType {
        return self.impl_.dtype(allocator);
    }

    pub fn isSparse(self: *const Tensor, allocator: std.mem.Allocator) !bool {
        return self.impl_.isSparse(allocator);
    }

    pub fn astype(self: *const Tensor, allocator: std.mem.Allocator, new_type: DType) !Tensor {
        return self.impl_.astype(allocator, new_type);
    }

    pub fn index(self: *const Tensor, allocator: std.mem.Allocator, indices: std.ArrayList(Index)) !Tensor {
        return self.impl_.index(allocator, indices);
    }

    pub fn flatten(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.flatten(allocator);
    }

    // TODO: implement
    // pub fn flat(self: *const Tensor) Tensor {
    // return self.impl_.flat();
    // }

    pub fn asContiguousTensor(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.asContiguousTensor(allocator);
    }

    pub fn backendType(self: *const Tensor) TensorBackendType {
        return self.impl_.backendType();
    }

    pub fn getAdapter(self: *const Tensor, comptime T: type) *T {
        return @ptrCast(@alignCast(self.impl_.ptr));
    }

    pub fn backend(self: *const Tensor, allocator: std.mem.Allocator) !TensorBackend {
        return self.impl_.backend(allocator);
    }

    pub fn allocHost(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type) !?[]T {
        if (try self.isEmpty(allocator)) {
            return null;
        }
        var res: []T = try allocator.alloc(T, try self.bytes(allocator) / @sizeOf(T));
        try self.impl_.host(allocator, res.ptr);
        return res;
    }

    // TODO: FL_CREATE_MEMORY_OPS macro equivalent

    pub fn unlock(self: *const Tensor, allocator: std.mem.Allocator) !void {
        return self.impl_.unlock(allocator);
    }

    pub fn isLocked(self: *const Tensor, allocator: std.mem.Allocator) !bool {
        return self.impl_.isLocked(allocator);
    }

    pub fn isContiguous(self: *const Tensor, allocator: std.mem.Allocator) !bool {
        return self.impl_.isContiguous(allocator);
    }

    pub fn strides(self: *const Tensor, allocator: std.mem.Allocator) !Shape {
        return self.impl_.strides(allocator);
    }

    pub fn stream(self: *const Tensor, allocator: std.mem.Allocator) !Stream {
        return self.impl_.stream(allocator);
    }

    pub fn setContext(self: *const Tensor, context: ?*anyopaque) !void {
        return self.impl_.setContext(context);
    }

    pub fn getContext(self: *const Tensor) !?*anyopaque {
        return self.impl_.getContext();
    }

    pub fn toString(self: *const Tensor, allocator: std.mem.Allocator) ![]const u8 {
        return self.impl_.toString(allocator);
    }

    // TODO: equivalent of assignment operators

};

pub fn fromScalar(allocator: std.mem.Allocator, comptime T: type, value: T, dtype: DType) !Tensor {
    return (try defaultTensorBackend(allocator)).fromScalar(allocator, T, value, dtype);
}

pub fn full(allocator: std.mem.Allocator, shape: *const Shape, comptime T: type, value: T, dtype: DType) !Tensor {
    return (try defaultTensorBackend(allocator)).full(allocator, shape, T, value, dtype);
}

pub fn arange(allocator: std.mem.Allocator, shape: *const Shape, seq_dim: Dim, dtype: DType) !Tensor {
    return (try defaultTensorBackend(allocator)).arange(allocator, shape, seq_dim, dtype);
}

pub fn iota(allocator: std.mem.Allocator, shape: *const Shape, tile_dims: *const Shape, dtype: DType) !Tensor {
    return (try defaultTensorBackend(allocator)).iota(allocator, shape, tile_dims, dtype);
}

//************************ Shaping and Indexing *************************//

pub fn reshape(allocator: std.mem.Allocator, tensor: Tensor, shape: *const Shape) Tensor {
    return (try tensor.backend(allocator)).reshape(tensor, shape);
}

pub fn transpose(allocator: std.mem.Allocator, tensor: Tensor, axes: *const Shape) Tensor {
    return (try tensor.backend(allocator)).transpose(tensor, axes);
}

pub fn tile(allocator: std.mem.Allocator, tensor: Tensor, shape: *const Shape) Tensor {
    return (try tensor.backend(allocator)).tile(tensor, shape);
}

pub fn concatenate(allocator: std.mem.Allocator, tensors: *const std.ArrayList(Tensor), axis: u32) !Tensor {
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
    return (try tensors.items[0].backend(allocator)).concatenate(tensors, axis);
}

pub fn nonzero(allocator: std.mem.Allocator, tensor: Tensor) Tensor {
    return (try tensor.backend(allocator)).nonzero(tensor);
}

pub fn pad(allocator: std.mem.Allocator, tensor: Tensor, pad_widths: *const std.ArrayList([2]i32), pad_type: PadType) !Tensor {
    return (try tensor.backend(allocator)).pad(allocator, tensor, pad_widths, pad_type);
}

//************************** Unary Operators ***************************//

pub fn exp(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).exp(allocator, tensor);
}

pub fn log(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).log(allocator, tensor);
}

pub fn negative(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).negative(allocator, tensor);
}

pub fn logicalNot(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).logicalNot(allocator, tensor);
}

pub fn log1p(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).log1p(allocator, tensor);
}

pub fn sin(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).sin(allocator, tensor);
}

pub fn cos(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).cos(allocator, tensor);
}

pub fn sqrt(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).sqrt(allocator, tensor);
}

pub fn tanh(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).tanh(allocator, tensor);
}

pub fn floor(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).floor(allocator, tensor);
}

pub fn ceil(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).ceil(allocator, tensor);
}

pub fn rint(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).rint(allocator, tensor);
}

pub fn absolute(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).absolute(allocator, tensor);
}

pub fn sigmoid(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).sigmoid(allocator, tensor);
}

pub fn erf(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).erf(allocator, tensor);
}

pub fn flip(allocator: std.mem.Allocator, tensor: Tensor, dim: u32) !Tensor {
    return (try tensor.backend(allocator)).flip(allocator, tensor, dim);
}

pub fn clip(allocator: std.mem.Allocator, tensor: Tensor, comptime low_T: type, low: low_T, comptime high_T: type, high: high_T) !Tensor {
    if ((low_T != Tensor or low_T != f64) or (high_T != Tensor or high_T != f64)) {
        @compileError("clip: low or high must be a Tensor or f64");
    }
    var backend = try tensor.backend(allocator);
    var lowTensor: Tensor = undefined;
    var lowTensorInit = false;
    defer if (lowTensorInit) lowTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var highTensor: Tensor = undefined;
    var highTensorInit = false;
    defer if (highTensorInit) highTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (low_T == Tensor) {
        lowTensor = low;
    } else if (low_T == f64) {
        var shape = try tensor.shape(allocator);
        lowTensor = try backend.full(allocator, &shape, f64, low, .f32);
        lowTensorInit = true;
    }
    if (high_T == Tensor) {
        highTensor = high;
    } else if (high_T == f64) {
        var shape = try tensor.shape(allocator);
        highTensor = try backend.full(allocator, &shape, f64, high, .f32);
        highTensorInit = true;
    }
    return backend.clip(allocator, tensor, lowTensor, highTensor);
}

pub fn roll(allocator: std.mem.Allocator, tensor: Tensor, shift: Dim, axis: usize) !Tensor {
    return (try tensor.backend(allocator)).roll(allocator, tensor, shift, axis);
}

pub fn isnan(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).isnan(allocator, tensor);
}

pub fn isinf(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).isinf(allocator, tensor);
}

// TODO: pub fn sign()

pub fn tril(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).tril(allocator, tensor);
}

pub fn triu(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).triu(allocator, tensor);
}

pub fn where(
    allocator: std.mem.Allocator,
    condition: Tensor,
    comptime x_T: type,
    x: x_T,
    comptime y_T: type,
    y: y_T,
) !Tensor {
    if (x_T != Tensor and y_T != Tensor) {
        @compileError("where: either lhs or rhs must be a Tensor");
    }
    var backend = try condition.backend(allocator);
    var xTensor: Tensor = undefined;
    var xTensorInit = false;
    defer if (xTensorInit) xTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var yTensor: Tensor = undefined;
    var yTensorInit = false;
    defer if (yTensorInit) yTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (x_T == Tensor) {
        xTensor = x;
    } else if (x_T == f64) {
        var shape = try condition.shape(allocator);
        xTensor = try backend.full(allocator, &shape, f64, x, .f32);
        xTensorInit = true;
    } else {
        @compileError("where: x must be either a Tensor or f64");
    }
    if (y_T == Tensor) {
        yTensor = y;
    } else if (y_T == f64) {
        var shape = try condition.shape(allocator);
        yTensor = try backend.full(allocator, &shape, f64, y, .f32);
        yTensorInit = true;
    } else {
        @compileError("where: y must be either a Tensor or f64");
    }
    return backend.where(allocator, condition, x, y);
}

pub fn topk(allocator: std.mem.Allocator, input: Tensor, k: u32, axis: Dim, sort_mode: SortMode) !TopkRes {
    return (try input.backend(allocator)).topk(allocator, input, k, axis, sort_mode);
}

pub fn sort(allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
    return (try input.backend(allocator)).sort(allocator, input, axis, sort_mode);
}

pub fn sortIndex(allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !SortIndexRes {
    return (try input.backend(allocator)).sortIndex(allocator, input, axis, sort_mode);
}

pub fn argsort(allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
    return (try input.backend(allocator)).argsort(allocator, input, axis, sort_mode);
}

pub fn add(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("add: either lhs or rhs must be a Tensor");
    }
    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.add(allocator, lhs, rhs);
}

pub fn sub(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("sub: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.sub(allocator, lhs, rhs);
}

pub fn mul(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("mul: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.mul(allocator, lhs, rhs);
}

pub fn div(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("div: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.div(allocator, lhs, rhs);
}

pub fn eq(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("eq: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.eq(allocator, lhs, rhs);
}

pub fn neq(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("neq: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.neq(allocator, lhs, rhs);
}

pub fn lessThan(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("lessThan: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.lessThan(allocator, lhs, rhs);
}

pub fn lessThanEqual(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("lessThanEqual: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.lessThanEqual(allocator, lhs, rhs);
}

pub fn greaterThan(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("greaterThan: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.greaterThan(allocator, lhs, rhs);
}

pub fn greaterThanEqual(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("greaterThanEqual: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.greaterThanEqual(allocator, lhs, rhs);
}

pub fn logicalOr(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("logicalOr: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.logicalOr(allocator, lhs, rhs);
}

pub fn logicalAnd(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("logicalAnd: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.logicalAnd(allocator, lhs, rhs);
}

pub fn mod(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("mod: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.mod(allocator, lhs, rhs);
}

pub fn bitwiseAnd(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("bitwiseAnd: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.bitwiseAnd(allocator, lhs, rhs);
}

pub fn bitwiseOr(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("bitwiseOr: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.bitwiseOr(allocator, lhs, rhs);
}

pub fn bitwiseXor(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("bitwiseXor: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.bitwiseXor(allocator, lhs, rhs);
}

pub fn lShift(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("lShift: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.lShift(allocator, lhs, rhs);
}

pub fn rShift(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("rShift: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    return backend.rShift(allocator, lhs, rhs);
}

pub fn minimum(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("minimum: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else if (lhs_T == f64) {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, .f32, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    } else {
        @compileError("minimum: lhs must be a Tensor or f64");
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else if (rhs_T == f64) {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, .f32, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    } else {
        @compileError("minimum: rhs must be a Tensor or f64");
    }
    return backend.minimum(allocator, lhs, rhs);
}

pub fn maximum(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("maximum: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else if (lhs_T == f64) {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, .f32, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    } else {
        @compileError("maximum: lhs must be a Tensor or f64");
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else if (rhs_T == f64) {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, .f32, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    } else {
        @compileError("maximum: rhs must be a Tensor or f64");
    }
    return backend.maximum(allocator, lhs, rhs);
}

pub fn power(allocator: std.mem.Allocator, comptime lhs_T: type, lhs: lhs_T, comptime rhs_T: type, rhs: rhs_T) !Tensor {
    if (lhs_T != Tensor and rhs_T != Tensor) {
        @compileError("power: either lhs or rhs must be a Tensor");
    }

    var backend: TensorBackend = undefined;
    var lhsTensor: Tensor = undefined;
    var lhsTensorInit = false;
    defer if (lhsTensorInit) lhsTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    var rhsTensor: Tensor = undefined;
    var rhsTensorInit = false;
    defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    if (lhs_T == Tensor) {
        lhsTensor = lhs;
        backend = try lhs.backend(allocator);
    } else if (lhs_T == f64) {
        var shape = try rhs.shape(allocator);
        backend = try rhs.backend(allocator);
        lhsTensor = try backend.full(allocator, &shape, .f32, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    } else {
        @compileError("power: lhs must be a Tensor or f64");
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else if (rhs_T == f64) {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, &shape, .f32, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    } else {
        @compileError("power: rhs must be a Tensor or f64");
    }
    return backend.power(allocator, lhs, rhs);
}
