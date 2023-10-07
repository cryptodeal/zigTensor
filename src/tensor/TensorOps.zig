const std = @import("std");
const tensor_ = @import("tensor.zig");

const defaultTensorBackend = tensor_.defaultTensorBackend;
const Dim = tensor_.shape.Dim;
const DType = tensor_.DType;
const MatrixProperty = tensor_.MatrixProperty;
const PadType = tensor_.PadType;
const Shape = tensor_.shape.Shape;
const SortMode = tensor_.SortMode;
const Tensor = tensor_.Tensor;
const TensorBackend = tensor_.TensorBackend;
const TensorBackendType = tensor_.TensorBackendType;
const DefaultTensorType_t = tensor_.DefaultTensorType_t;
const TensorAdapterBase = tensor_.TensorAdapterBase;

pub inline fn ztTensorBackendsMatch(fn_name: []const u8, tensors: []const Tensor) !void {
    if (tensors.len <= 1) return;
    var backend_type = tensors[0].backendType();
    for (tensors[1..]) |t| {
        if (t.backendType() != backend_type) {
            std.log.debug("{s} called with tensors of different backends.\n", .{fn_name});
            return error.TensorBackendMismatch;
        }
    }
}

pub fn fromScalar(allocator: std.mem.Allocator, comptime T: type, value: T, dtype: DType) !Tensor {
    return (try defaultTensorBackend(allocator)).fromScalar(allocator, T, value, dtype);
}

pub fn full(allocator: std.mem.Allocator, shape: Shape, comptime T: type, value: T, dtype: DType) !Tensor {
    return (try defaultTensorBackend(allocator)).full(allocator, shape, T, value, dtype);
}

pub fn identity(allocator: std.mem.Allocator, dim: Dim, dtype: DType) !Tensor {
    return (try defaultTensorBackend(allocator)).identity(allocator, dim, dtype);
}

pub fn arange(allocator: std.mem.Allocator, shape: Shape, seq_dim: Dim, dtype: DType) !Tensor {
    return (try defaultTensorBackend(allocator)).arange(allocator, shape, seq_dim, dtype);
}

pub fn iota(allocator: std.mem.Allocator, shape: Shape, tile_dims: Shape, dtype: DType) !Tensor {
    return (try defaultTensorBackend(allocator)).iota(allocator, shape, tile_dims, dtype);
}

//************************ Shaping and Indexing *************************//

pub fn reshape(allocator: std.mem.Allocator, tensor: Tensor, shape: Shape) !Tensor {
    return (try tensor.backend(allocator)).reshape(allocator, tensor, shape);
}

pub fn transpose(allocator: std.mem.Allocator, tensor: Tensor, axes: Shape) !Tensor {
    return (try tensor.backend(allocator)).transpose(allocator, tensor, axes);
}

pub fn tile(allocator: std.mem.Allocator, tensor: Tensor, shape: Shape) !Tensor {
    return (try tensor.backend(allocator)).tile(allocator, tensor, shape);
}

pub fn concatenate(allocator: std.mem.Allocator, tensors: []const Tensor, axis: u32) !Tensor {
    if (tensors.len == 0) {
        std.log.debug("concatenate: called on empty set of tensors\n", .{});
        return error.ConcatFailedZeroTensors;
    }

    // ensure that all tensors have the same backend
    const b: TensorBackendType = tensors[0].backendType();
    for (tensors) |t| {
        if (t.backendType() != b) {
            std.log.debug("concatenate: tried to concatenate tensors of different backends\n", .{});
            return error.ConcatFailedBackendMismatch;
        }
    }
    return (try tensors[0].backend(allocator)).concatenate(allocator, tensors, axis);
}

pub fn nonzero(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).nonzero(allocator, tensor);
}

pub fn pad(allocator: std.mem.Allocator, tensor: Tensor, pad_widths: []const [2]i64, pad_type: PadType) !Tensor {
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

pub inline fn abs(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return absolute(allocator, tensor);
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
    try ztTensorBackendsMatch(@src().fn_name, &.{ tensor, lowTensor, highTensor });
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

pub fn sign(allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
    return (try tensor.backend(allocator)).sign(allocator, tensor);
}

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
    defer if (xTensorInit) {
        std.debug.print("deinit xTensor\n", .{});
        xTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    };
    var yTensor: Tensor = undefined;
    var yTensorInit = false;
    defer if (yTensorInit) {
        std.debug.print("deinit yTensor\n", .{});
        yTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
    };
    if (x_T == Tensor) {
        xTensor = x;
    } else if (x_T == f64) {
        var shape = try condition.shape(allocator);
        xTensor = try backend.full(allocator, shape, f64, x, try y.dtype(allocator));
        xTensorInit = true;
    } else {
        @compileError("where: x must be either a Tensor or f64");
    }
    if (y_T == Tensor) {
        yTensor = y;
    } else if (y_T == f64) {
        var shape = try condition.shape(allocator);
        yTensor = try backend.full(allocator, shape, f64, y, try x.dtype(allocator));
        yTensorInit = true;
    } else {
        @compileError("where: y must be either a Tensor or f64");
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ condition, xTensor, yTensor });
    return backend.where(allocator, condition, xTensor, yTensor);
}

pub fn topk(allocator: std.mem.Allocator, values: Tensor, indices: Tensor, input: Tensor, k: u32, axis: Dim, sort_mode: SortMode) !void {
    return (try input.backend(allocator)).topk(allocator, values, indices, input, k, axis, sort_mode);
}

pub fn sort(allocator: std.mem.Allocator, values: Tensor, indices: ?Tensor, input: Tensor, axis: Dim, sort_mode: SortMode) !void {
    return (try input.backend(allocator)).sort(allocator, values, indices, input, axis, sort_mode);
}

pub fn argsort(allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
    return (try input.backend(allocator)).argsort(allocator, input, axis, sort_mode);
}

//************************** Binary Operators ***************************//

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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.add(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.sub(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.mul(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.div(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.eq(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.neq(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.lessThan(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.lessThanEqual(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.greaterThan(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.greaterThanEqual(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.logicalOr(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.logicalAnd(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.mod(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.bitwiseAnd(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.bitwiseOr(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.bitwiseXor(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.lShift(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, lhs_T, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, rhs_T, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.rShift(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, .f32, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    } else {
        @compileError("minimum: lhs must be a Tensor or f64");
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else if (rhs_T == f64) {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, .f32, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    } else {
        @compileError("minimum: rhs must be a Tensor or f64");
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.minimum(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, .f32, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    } else {
        @compileError("maximum: lhs must be a Tensor or f64");
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else if (rhs_T == f64) {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, .f32, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    } else {
        @compileError("maximum: rhs must be a Tensor or f64");
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.maximum(allocator, lhsTensor, rhsTensor);
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
        lhsTensor = try backend.full(allocator, shape, .f32, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    } else {
        @compileError("power: lhs must be a Tensor or f64");
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else if (rhs_T == f64) {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, .f32, rhs, try lhs.dtype(allocator));
        rhsTensorInit = true;
    } else {
        @compileError("power: rhs must be a Tensor or f64");
    }
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhsTensor, rhsTensor });
    return backend.power(allocator, lhsTensor, rhsTensor);
}

//******************************* BLAS ********************************//
pub fn matmul(allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, lhs_prop: MatrixProperty, rhs_prop: MatrixProperty) !Tensor {
    try ztTensorBackendsMatch(@src().fn_name, &.{ lhs, rhs });
    return (try lhs.backend(allocator)).matmul(allocator, lhs, rhs, lhs_prop, rhs_prop);
}

//************************** Reductions ***************************//

pub fn amin(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).amin(allocator, input, axes, keep_dims);
}

pub fn amax(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).amax(allocator, input, axes, keep_dims);
}

pub fn min(allocator: std.mem.Allocator, values: Tensor, indices: Tensor, input: Tensor, axis: u32, keep_dims: bool) !void {
    return (try input.backend(allocator)).min(allocator, values, indices, input, axis, keep_dims);
}

pub fn max(allocator: std.mem.Allocator, values: Tensor, indices: Tensor, input: Tensor, axis: u32, keep_dims: bool) !void {
    return (try input.backend(allocator)).max(allocator, values, indices, input, axis, keep_dims);
}

pub fn sum(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).sum(allocator, input, axes, keep_dims);
}

pub fn cumsum(allocator: std.mem.Allocator, input: Tensor, axis: u32) !Tensor {
    return (try input.backend(allocator)).cumsum(allocator, input, axis);
}

pub fn argmax(allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).argmax(allocator, input, axis, keep_dims);
}

pub fn argmin(allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).argmin(allocator, input, axis, keep_dims);
}

pub fn mean(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).mean(allocator, input, axes, keep_dims);
}

pub fn median(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).median(allocator, input, axes, keep_dims);
}

/// Variance of an tensor over given axes. If axes is left empty, computes the
/// variance along all axes.
///
/// Returns a Tensor containing the variance(s).
pub fn variance(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, bias: bool, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).variance(allocator, input, axes, bias, keep_dims);
}

pub fn stdev(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).stdev(allocator, input, axes, keep_dims);
}

pub fn norm(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, p: f64, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).norm(allocator, input, axes, p, keep_dims);
}

pub fn countNonzero(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).countNonzero(allocator, input, axes, keep_dims);
}

pub fn any(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).any(allocator, input, axes, keep_dims);
}

pub fn all(allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
    return (try input.backend(allocator)).all(allocator, input, axes, keep_dims);
}

//************************** Utilities ***************************//
pub fn print(allocator: std.mem.Allocator, input: Tensor) !void {
    return (try input.backend(allocator)).print(allocator, input);
}

pub fn allClose(allocator: std.mem.Allocator, a: Tensor, b: Tensor, abs_tolerance: f64) !bool {
    if (try a.dtype(allocator) != try b.dtype(allocator)) {
        std.debug.print("DType mismatch: a.dtype: {s} vs b.dtype: {s}\n", .{ @tagName(try a.dtype(allocator)), @tagName(try b.dtype(allocator)) });
        return false;
    }

    if (!tensor_.shape.eql(try a.shape(allocator), try b.shape(allocator))) {
        std.debug.print("Shape mismatch: a.shape: {any}, b.dtype: {any}\n", .{ try a.shape(allocator), try b.shape(allocator) });
        return false;
    }
    if (try a.elements(allocator) == 0 and try b.elements(allocator) == 0) {
        return true;
    }
    var r1 = try sub(allocator, Tensor, a, Tensor, b);
    defer r1.deinit();
    var r2 = try abs(allocator, r1);
    defer r2.deinit();
    var r3 = try amax(allocator, r2, &.{}, false);
    defer r3.deinit();
    var r4 = try r3.astype(allocator, .f64);
    defer r4.deinit();
    var res = try r4.scalar(allocator, f64);
    return res < abs_tolerance;
}

pub fn allEqual(allocator: std.mem.Allocator, a: Tensor, b: Tensor) !bool {
    if (try a.dtype(allocator) != try b.dtype(allocator)) {
        return false;
    }
    if (!tensor_.shape.eql(try a.shape(allocator), try b.shape(allocator))) {
        return false;
    }
    if (try a.elements(allocator) == 0 and try b.elements(allocator) == 0) {
        return true;
    }
    var r1 = try sub(allocator, Tensor, a, Tensor, b);
    defer r1.deinit();
    var r2 = try abs(allocator, r1);
    defer r2.deinit();
    var r3 = try amax(allocator, r2, &.{}, false);
    defer r3.deinit();
    var r4 = try r3.astype(allocator, .f64);
    defer r4.deinit();
    var res = try r4.scalar(allocator, f64);
    return res == 0;
}

//************************** Unit Tests **************************//

test "TensorBase -> assign" {
    const rand = @import("Random.zig").rand;
    const deinit = @import("Init.zig").deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 5, 5 }, f64, 2, .f32);
    defer a.deinit();
    var b = try rand(allocator, &.{ 5, 5 }, .f32);
    defer b.deinit();

    // a = b;
    try a.assign(allocator, Tensor, b);
    try std.testing.expect(try allEqual(allocator, a, b));

    // a = 2;
    try a.assign(allocator, f64, 10);

    var expected = try full(allocator, &.{ 5, 5 }, f64, 10, .f32);
    defer expected.deinit();
    try std.testing.expect(try allEqual(allocator, a, expected));

    // TODO: more extensive testing (mirror Flashlight's tests)
}

test "TensorBase -> inPlaceAdd" {
    const deinit = @import("Init.zig").deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 5, 5 }, f64, 2, .f32);
    defer a.deinit();
    try a.inPlaceAdd(allocator, f64, 2);

    var expected = try full(allocator, &.{ 5, 5 }, f64, 4, .f32);
    defer expected.deinit();
    try std.testing.expect(try allEqual(allocator, a, expected));

    // TODO: more extensive testing (mirror Flashlight's tests)
}

test "TensorBase -> inPlaceSub" {
    const deinit = @import("Init.zig").deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 5, 5 }, f64, 6, .f32);
    defer a.deinit();
    try a.inPlaceSub(allocator, f64, 3);

    var expected = try full(allocator, &.{ 5, 5 }, f64, 3, .f32);
    defer expected.deinit();
    try std.testing.expect(try allEqual(allocator, a, expected));

    // TODO: more extensive testing (mirror Flashlight's tests)
}

test "TensorBase -> inPlaceMul" {
    const deinit = @import("Init.zig").deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 5, 5 }, f64, 5, .f32);
    defer a.deinit();
    try a.inPlaceMul(allocator, f64, 2);

    var expected = try full(allocator, &.{ 5, 5 }, f64, 10, .f32);
    defer expected.deinit();
    try std.testing.expect(try allEqual(allocator, a, expected));

    // TODO: more extensive testing (mirror Flashlight's tests)
}

test "TensorBase -> inPlaceDiv" {
    const deinit = @import("Init.zig").deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 5, 5 }, f64, 10, .f32);
    defer a.deinit();
    try a.inPlaceDiv(allocator, f64, 2);

    var expected = try full(allocator, &.{ 5, 5 }, f64, 5, .f32);
    defer expected.deinit();
    try std.testing.expect(try allEqual(allocator, a, expected));

    // TODO: more extensive testing (mirror Flashlight's tests)
}

test "TensorBase -> sign" {
    const rand = @import("Random.zig").rand;
    const deinit = @import("Init.zig").deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var init_rand = try rand(allocator, &.{ 5, 5 }, .f32);
    defer init_rand.deinit();
    var vals = try sub(allocator, Tensor, init_rand, f32, 0.5);
    defer vals.deinit();
    // TODO: need to finish writing this test
}
