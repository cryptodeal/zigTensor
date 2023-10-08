const std = @import("std");
const tensor_ = @import("tensor.zig");

const defaultTensorBackend = tensor_.defaultTensorBackend;
const Dim = tensor_.shape.Dim;
const DType = tensor_.DType;
const dtypeTraits = tensor_.dtypeTraits;
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

pub fn arange2(allocator: std.mem.Allocator, comptime T: type, start: T, end: T, step: T) !Tensor {
    const ctype = comptime dtypeTraits(T).ctype; // throws compileError if invalid type
    var dim: Dim = switch (T) {
        f16, f32, f64 => @intFromFloat((end - start) / step),
        else => @intCast(@divTrunc(end - start, step)),
    };
    var res = try arange(
        allocator,
        &.{dim},
        0,
        ctype,
    );
    try res.inPlaceMul(allocator, T, step);
    try res.inPlaceAdd(allocator, T, start);
    return res;
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
        xTensor.deinit(); // if initializing lhsTensor, defer freeing associated mem
    };
    var yTensor: Tensor = undefined;
    var yTensorInit = false;
    defer if (yTensorInit) {
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
        lhsTensor = try backend.full(allocator, shape, f64, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    } else {
        @compileError("minimum: lhs must be a Tensor or f64");
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else if (rhs_T == f64) {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, f64, rhs, try lhs.dtype(allocator));
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
        lhsTensor = try backend.full(allocator, shape, f64, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    } else {
        @compileError("maximum: lhs must be a Tensor or f64");
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else if (rhs_T == f64) {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, f64, rhs, try lhs.dtype(allocator));
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
        lhsTensor = try backend.full(allocator, shape, f64, lhs, try rhs.dtype(allocator));
        lhsTensorInit = true;
    } else {
        @compileError("power: lhs must be a Tensor or f64");
    }

    if (rhs_T == Tensor) {
        rhsTensor = rhs;
    } else if (rhs_T == f64) {
        var shape = try lhs.shape(allocator);
        rhsTensor = try backend.full(allocator, shape, f64, rhs, try lhs.dtype(allocator));
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
fn assertTensorScalarBinop(
    allocator: std.mem.Allocator,
    in: Tensor,
    comptime ScalarType: type,
    scalar: ScalarType,
    comptime OpType: type,
    op: OpType,
    expect_out: Tensor,
) !void {
    var res = try op(allocator, Tensor, in, ScalarType, scalar);
    defer res.deinit();
    var expect = try expect_out.astype(allocator, try res.dtype(allocator));
    defer expect.deinit();
    try std.testing.expect(try allClose(allocator, res, expect, 1e-5));
}

fn assertScalarTensorBinop(
    allocator: std.mem.Allocator,
    comptime ScalarType: type,
    scalar: ScalarType,
    in: Tensor,
    comptime OpType: type,
    op: OpType,
    expect_out: Tensor,
) !void {
    var res = try op(allocator, ScalarType, scalar, Tensor, in);
    defer res.deinit();
    var expect = try expect_out.astype(allocator, try res.dtype(allocator));
    defer expect.deinit();
    try std.testing.expect(try allClose(allocator, res, expect, 1e-5));
}

fn assertScalarTensorCommutativeBinop(
    allocator: std.mem.Allocator,
    scalar: i8,
    in: Tensor,
    comptime OpType: type,
    op: OpType,
    expect_out: Tensor,
) !void {
    try assertTensorScalarBinop(allocator, in, i8, scalar, OpType, op, expect_out);
    try assertScalarTensorBinop(allocator, i8, scalar, in, OpType, op, expect_out);
}

fn assertCommutativeBinop(
    allocator: std.mem.Allocator,
    in1: Tensor,
    in2: Tensor,
    comptime OpType: type,
    op: OpType,
    out: Tensor,
) !void {
    var res1 = try op(allocator, Tensor, in1, Tensor, in2);
    defer res1.deinit();
    try std.testing.expect(try allClose(allocator, res1, out, 1e-5));
    var res2 = try op(allocator, Tensor, in2, Tensor, in1);
    defer res2.deinit();
    try std.testing.expect(try allClose(allocator, res2, out, 1e-5));
}

const TestOpType = *const fn (allocator: std.mem.Allocator, comptime lhs_T: type, lhs: anytype, comptime rhs_T: type, rhs: anytype) anyerror!Tensor;

fn testArithmeticBinops(allocator: std.mem.Allocator, dtype: DType) !void {
    var tmp_a = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 0, 1, 2, 3 }, .f32);
    defer tmp_a.deinit();
    var a = try tmp_a.astype(allocator, dtype);
    defer a.deinit();
    var tmp_b = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 1, 2, 3, 4 }, .f32);
    defer tmp_b.deinit();
    var b = try tmp_b.astype(allocator, dtype);
    defer b.deinit();
    var tmp_c = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 1, 3, 5, 7 }, .f32);
    defer tmp_c.deinit();
    var c = try tmp_c.astype(allocator, dtype);
    defer c.deinit();
    var tmp_d = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 1, 6, 15, 28 }, .f32);
    defer tmp_d.deinit();
    var d = try tmp_d.astype(allocator, dtype);
    defer d.deinit();
    var tmp_e = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 3, 2, 1, 0 }, .f32);
    defer tmp_e.deinit();
    var e = try tmp_e.astype(allocator, dtype);
    defer e.deinit();
    var tmp_f = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 2, 4, 6, 8 }, .f32);
    defer tmp_f.deinit();
    var f = try tmp_f.astype(allocator, dtype);
    defer f.deinit();
    var z = try full(allocator, &.{ 2, 2 }, f64, 0, dtype);
    defer z.deinit();

    try assertCommutativeBinop(allocator, a, z, TestOpType, add, a);
    try assertCommutativeBinop(allocator, a, b, TestOpType, add, c);
    try assertScalarTensorCommutativeBinop(allocator, 1, a, TestOpType, add, b);
    try assertScalarTensorCommutativeBinop(allocator, 0, a, TestOpType, add, a);

    var tmp1 = try sub(allocator, Tensor, c, Tensor, z);
    defer tmp1.deinit();
    try std.testing.expect(try allClose(allocator, tmp1, c, 1e-5));
    if (dtype != .u8 and dtype != .u16 and dtype != .u32 and dtype != .u64) {
        var tmp2 = try sub(allocator, Tensor, z, Tensor, c);
        defer tmp2.deinit();
        var exp1 = try negative(allocator, c);
        defer exp1.deinit();
        try std.testing.expect(try allClose(allocator, tmp2, exp1, 1e-5));
    }
    var tmp3 = try sub(allocator, Tensor, c, Tensor, b);
    defer tmp3.deinit();
    try std.testing.expect(try allClose(allocator, tmp3, a, 1e-5));
    try assertTensorScalarBinop(allocator, b, i8, 1, TestOpType, sub, a);
    try assertScalarTensorBinop(allocator, i8, 3, a, TestOpType, sub, e);
    try assertTensorScalarBinop(allocator, a, i8, 0, TestOpType, sub, a);

    try assertCommutativeBinop(allocator, c, z, TestOpType, mul, z);
    try assertCommutativeBinop(allocator, c, b, TestOpType, mul, d);
    try assertScalarTensorCommutativeBinop(allocator, 0, a, TestOpType, mul, z);
    try assertScalarTensorCommutativeBinop(allocator, 1, a, TestOpType, mul, a);
    try assertScalarTensorCommutativeBinop(allocator, 2, b, TestOpType, mul, f);

    var tmp4 = try div(allocator, Tensor, z, Tensor, b);
    defer tmp4.deinit();
    try std.testing.expect(try allClose(allocator, tmp4, z, 1e-5));
    var tmp5 = try div(allocator, Tensor, d, Tensor, b);
    defer tmp5.deinit();
    try std.testing.expect(try allClose(allocator, tmp5, c, 1e-5));
    try assertTensorScalarBinop(allocator, z, i8, 1, TestOpType, div, z);
    try assertTensorScalarBinop(allocator, a, i8, 1, TestOpType, div, a);
    try assertTensorScalarBinop(allocator, f, i8, 2, TestOpType, div, b);
    // TODO division by zero doesn't always fail.
    // e.g., ArrayFire yields max value of dtype
}

fn testComparisonBinops(allocator: std.mem.Allocator, dtype: DType) !void {
    // expected values for comparison op results
    var falses = try full(allocator, &.{ 2, 2 }, i8, 0, .b8);
    defer falses.deinit();
    var trues = try full(allocator, &.{ 2, 2 }, i8, 1, .b8);
    defer trues.deinit();
    var tmp_false_trues = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 0, 1, 0, 1 }, .f32);
    defer tmp_false_trues.deinit();
    var false_trues = try tmp_false_trues.astype(allocator, .b8);
    defer false_trues.deinit();
    var tmp_true_falses = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 1, 0, 1, 0 }, .f32);
    defer tmp_true_falses.deinit();
    var true_falses = try tmp_true_falses.astype(allocator, .b8);
    defer true_falses.deinit();

    // values used for testing
    var tmp_a = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 0, 1, 2, 3 }, .f32);
    defer tmp_a.deinit();
    var a = try tmp_a.astype(allocator, dtype);
    defer a.deinit();
    var tmp_b = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 0, 0, 2, 0 }, .f32);
    defer tmp_b.deinit();
    var b = try tmp_b.astype(allocator, dtype);
    defer b.deinit();
    var tmp_c = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 2, 3, 4, 5 }, .f32);
    defer tmp_c.deinit();
    var c = try tmp_c.astype(allocator, dtype);
    defer c.deinit();
    var tmp_d = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 0, 4, 2, 6 }, .f32);
    defer tmp_d.deinit();
    var d = try tmp_d.astype(allocator, dtype);
    defer d.deinit();
    var tmp_e = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 0, 1, 0, 1 }, .f32);
    defer tmp_e.deinit();
    var e = try tmp_e.astype(allocator, dtype);
    defer e.deinit();

    // test equality comparison (==)
    var tmp1 = try eq(allocator, Tensor, a, Tensor, a);
    defer tmp1.deinit();
    try std.testing.expect(try allClose(allocator, tmp1, trues, 1e-5));
    try assertCommutativeBinop(allocator, a, b, TestOpType, eq, true_falses);
    try assertCommutativeBinop(allocator, a, c, TestOpType, eq, falses);
    try assertScalarTensorCommutativeBinop(allocator, 4, a, TestOpType, eq, falses);
    try assertScalarTensorCommutativeBinop(allocator, 1, e, TestOpType, eq, false_trues);

    // test inequality comparison (!=)
    var tmp2 = try neq(allocator, Tensor, a, Tensor, a);
    defer tmp2.deinit();
    try std.testing.expect(try allClose(allocator, tmp2, falses, 1e-5));
    try assertCommutativeBinop(allocator, a, b, TestOpType, neq, false_trues);
    try assertCommutativeBinop(allocator, a, c, TestOpType, neq, trues);
    try assertScalarTensorCommutativeBinop(allocator, 4, a, TestOpType, neq, trues);
    try assertScalarTensorCommutativeBinop(allocator, 1, e, TestOpType, neq, true_falses);

    // test greater than comparison (>)
    var tmp3 = try greaterThan(allocator, Tensor, a, Tensor, a);
    defer tmp3.deinit();
    try std.testing.expect(try allClose(allocator, tmp3, falses, 1e-5));
    var tmp4 = try greaterThan(allocator, Tensor, c, Tensor, a);
    defer tmp4.deinit();
    try std.testing.expect(try allClose(allocator, tmp4, trues, 1e-5));
    var tmp5 = try greaterThan(allocator, Tensor, d, Tensor, a);
    defer tmp5.deinit();
    try std.testing.expect(try allClose(allocator, tmp5, false_trues, 1e-5));
    var tmp6 = try greaterThan(allocator, Tensor, a, Tensor, d);
    defer tmp6.deinit();
    try std.testing.expect(try allClose(allocator, tmp6, falses, 1e-5));
    try assertTensorScalarBinop(allocator, c, i8, 1, TestOpType, greaterThan, trues);
    try assertScalarTensorBinop(allocator, i8, 0, c, TestOpType, greaterThan, falses);
    try assertTensorScalarBinop(allocator, d, i8, 3, TestOpType, greaterThan, false_trues);
    try assertScalarTensorBinop(allocator, i8, 3, d, TestOpType, greaterThan, true_falses);

    // test less than comparison (<)
    var tmp7 = try lessThan(allocator, Tensor, a, Tensor, a);
    defer tmp7.deinit();
    try std.testing.expect(try allClose(allocator, tmp7, falses, 1e-5));
    var tmp8 = try lessThan(allocator, Tensor, c, Tensor, a);
    defer tmp8.deinit();
    try std.testing.expect(try allClose(allocator, tmp8, falses, 1e-5));
    var tmp9 = try lessThan(allocator, Tensor, d, Tensor, a);
    defer tmp9.deinit();
    try std.testing.expect(try allClose(allocator, tmp9, falses, 1e-5));
    var tmp10 = try lessThan(allocator, Tensor, a, Tensor, d);
    defer tmp10.deinit();
    try std.testing.expect(try allClose(allocator, tmp10, false_trues, 1e-5));
    try assertTensorScalarBinop(allocator, c, i8, 1, TestOpType, lessThan, falses);
    try assertScalarTensorBinop(allocator, i8, 0, c, TestOpType, lessThan, trues);
    try assertTensorScalarBinop(allocator, d, i8, 3, TestOpType, lessThan, true_falses);
    try assertScalarTensorBinop(allocator, i8, 3, d, TestOpType, lessThan, false_trues);

    // test greater than or equal comparison (>=)
    var tmp11 = try greaterThanEqual(allocator, Tensor, a, Tensor, a);
    defer tmp11.deinit();
    try std.testing.expect(try allClose(allocator, tmp11, trues, 1e-5));
    var tmp12 = try greaterThanEqual(allocator, Tensor, c, Tensor, a);
    defer tmp12.deinit();
    try std.testing.expect(try allClose(allocator, tmp12, trues, 1e-5));
    var tmp13 = try greaterThanEqual(allocator, Tensor, d, Tensor, a);
    defer tmp13.deinit();
    try std.testing.expect(try allClose(allocator, tmp13, trues, 1e-5));
    var tmp14 = try greaterThanEqual(allocator, Tensor, a, Tensor, d);
    defer tmp14.deinit();
    try std.testing.expect(try allClose(allocator, tmp14, true_falses, 1e-5));
    try assertTensorScalarBinop(allocator, c, i8, 2, TestOpType, greaterThanEqual, trues);
    try assertScalarTensorBinop(allocator, i8, 1, c, TestOpType, greaterThanEqual, falses);
    try assertTensorScalarBinop(allocator, d, i8, 3, TestOpType, greaterThanEqual, false_trues);
    try assertScalarTensorBinop(allocator, i8, 3, d, TestOpType, greaterThanEqual, true_falses);

    // test less than or equal comparison (<=)
    var tmp15 = try lessThanEqual(allocator, Tensor, a, Tensor, a);
    defer tmp15.deinit();
    try std.testing.expect(try allClose(allocator, tmp15, trues, 1e-5));
    var tmp16 = try lessThanEqual(allocator, Tensor, c, Tensor, a);
    defer tmp16.deinit();
    try std.testing.expect(try allClose(allocator, tmp16, falses, 1e-5));
    var tmp17 = try lessThanEqual(allocator, Tensor, d, Tensor, a);
    defer tmp17.deinit();
    try std.testing.expect(try allClose(allocator, tmp17, true_falses, 1e-5));
    var tmp18 = try lessThanEqual(allocator, Tensor, a, Tensor, d);
    defer tmp18.deinit();
    try std.testing.expect(try allClose(allocator, tmp18, trues, 1e-5));
    try assertTensorScalarBinop(allocator, c, i8, 1, TestOpType, lessThanEqual, falses);
    try assertScalarTensorBinop(allocator, i8, 2, c, TestOpType, lessThanEqual, trues);
    try assertTensorScalarBinop(allocator, d, i8, 3, TestOpType, lessThanEqual, true_falses);
    try assertScalarTensorBinop(allocator, i8, 3, d, TestOpType, lessThanEqual, false_trues);
}

fn testLogicalBinops(allocator: std.mem.Allocator, dtype: DType) !void {
    // expected values for comparison
    var falses = try full(allocator, &.{ 2, 2 }, f64, 0, .b8);
    defer falses.deinit();
    var trues = try full(allocator, &.{ 2, 2 }, f64, 1, .b8);
    defer trues.deinit();
    var tmp_false_trues = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 0, 1, 0, 1 }, .f32);
    defer tmp_false_trues.deinit();
    var false_trues = try tmp_false_trues.astype(allocator, .b8);
    defer false_trues.deinit();

    // values used for testing
    var tmp_a = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 0, 1, 0, 3 }, .f32);
    defer tmp_a.deinit();
    var a = try tmp_a.astype(allocator, dtype);
    defer a.deinit();
    var tmp_b = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 2, 3, 4, 5 }, .f32);
    defer tmp_b.deinit();
    var b = try tmp_b.astype(allocator, dtype);
    defer b.deinit();
    var z = try full(allocator, &.{ 2, 2 }, f64, 0, dtype);
    defer z.deinit();

    // test logicalOr (or)
    var tmp1 = try logicalOr(allocator, Tensor, z, Tensor, z);
    defer tmp1.deinit();
    try std.testing.expect(try allClose(allocator, tmp1, falses, 1e-5));
    try assertCommutativeBinop(allocator, a, z, TestOpType, logicalOr, false_trues);
    try assertCommutativeBinop(allocator, z, b, TestOpType, logicalOr, trues);
    try assertCommutativeBinop(allocator, a, b, TestOpType, logicalOr, trues);
    try assertScalarTensorCommutativeBinop(allocator, 0, a, TestOpType, logicalOr, false_trues);
    try assertScalarTensorCommutativeBinop(allocator, 2, z, TestOpType, logicalOr, trues);

    // test logicalAnd (and)
    var tmp2 = try logicalAnd(allocator, Tensor, z, Tensor, z);
    defer tmp2.deinit();
    try std.testing.expect(try allClose(allocator, tmp2, falses, 1e-5));
    try assertCommutativeBinop(allocator, a, z, TestOpType, logicalAnd, falses);
    try assertCommutativeBinop(allocator, z, b, TestOpType, logicalAnd, falses);
    try assertCommutativeBinop(allocator, a, b, TestOpType, logicalAnd, false_trues);
    try assertScalarTensorCommutativeBinop(allocator, 0, a, TestOpType, logicalAnd, falses);
    try assertScalarTensorCommutativeBinop(allocator, 2, a, TestOpType, logicalAnd, false_trues);
}

fn testModuloBinop(allocator: std.mem.Allocator, dtype: DType) !void {
    var tmp_a = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 1, 2, 3, 4 }, .f32);
    defer tmp_a.deinit();
    var a = try tmp_a.astype(allocator, dtype);
    defer a.deinit();
    var tmp_b = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 2, 3, 5, 7 }, .f32);
    defer tmp_b.deinit();
    var b = try tmp_b.astype(allocator, dtype);
    defer b.deinit();
    var tmp_c = try Tensor.fromSlice(allocator, &.{ 2, 2 }, f32, &.{ 0, 1, 2, 3 }, .f32);
    defer tmp_c.deinit();
    var c = try tmp_c.astype(allocator, dtype);
    defer c.deinit();
    var z = try full(allocator, &.{ 2, 2 }, f64, 0, dtype);
    defer z.deinit();

    var tmp1 = try mod(allocator, Tensor, z, Tensor, b);
    defer tmp1.deinit();
    try std.testing.expect(try allClose(allocator, tmp1, z, 1e-5));
    var tmp2 = try mod(allocator, Tensor, a, Tensor, a);
    defer tmp2.deinit();
    try std.testing.expect(try allClose(allocator, tmp2, z, 1e-5));
    var tmp3 = try mod(allocator, Tensor, a, Tensor, b);
    defer tmp3.deinit();
    try std.testing.expect(try allClose(allocator, tmp3, a, 1e-5));
    var tmp4 = try mod(allocator, Tensor, b, Tensor, a);
    defer tmp4.deinit();
    try std.testing.expect(try allClose(allocator, tmp4, c, 1e-5));

    try assertScalarTensorBinop(allocator, i8, 0, a, TestOpType, mod, z);
    try assertScalarTensorBinop(allocator, i8, 11, a, TestOpType, mod, c);
    try assertTensorScalarBinop(allocator, a, i8, 1, TestOpType, mod, z);
    try assertTensorScalarBinop(allocator, a, i8, 5, TestOpType, mod, a);
}

fn testBitBinops(allocator: std.mem.Allocator, dtype: DType) !void {
    var tmp_a = try Tensor.fromSlice(allocator, &.{ 2, 1 }, f32, &.{ 0b0001, 0b1000 }, .f32);
    defer tmp_a.deinit();
    var a = try tmp_a.astype(allocator, dtype);
    defer a.deinit();
    var tmp_b = try Tensor.fromSlice(allocator, &.{ 2, 1 }, f32, &.{ 0b0010, 0b0100 }, .f32);
    defer tmp_b.deinit();
    var b = try tmp_b.astype(allocator, dtype);
    defer b.deinit();
    var tmp_c = try Tensor.fromSlice(allocator, &.{ 2, 1 }, f32, &.{ 0b0011, 0b1100 }, .f32);
    defer tmp_c.deinit();
    var c = try tmp_c.astype(allocator, dtype);
    defer c.deinit();
    var tmp_d = try Tensor.fromSlice(allocator, &.{ 2, 1 }, f32, &.{ 0b0110, 0b0110 }, .f32);
    defer tmp_d.deinit();
    var d = try tmp_d.astype(allocator, dtype);
    defer d.deinit();
    var tmp_e = try Tensor.fromSlice(allocator, &.{ 2, 1 }, f32, &.{ 0b1000, 0b0001 }, .f32);
    defer tmp_e.deinit();
    var e = try tmp_e.astype(allocator, dtype);
    defer e.deinit();
    var tmp_g = try Tensor.fromSlice(allocator, &.{ 2, 1 }, f32, &.{ 2, 1 }, .f32);
    defer tmp_g.deinit();
    var g = try tmp_g.astype(allocator, dtype);
    defer g.deinit();
    var tmp_h = try Tensor.fromSlice(allocator, &.{ 2, 1 }, f32, &.{ 0b1000, 0b1000 }, .f32);
    defer tmp_h.deinit();
    var h = try tmp_h.astype(allocator, dtype);
    defer h.deinit();
    var tmp_z = try Tensor.fromSlice(allocator, &.{ 2, 1 }, f32, &.{ 0b0000, 0b0000 }, .f32);
    defer tmp_z.deinit();
    var z = try tmp_z.astype(allocator, dtype);
    defer z.deinit();

    // test bitwiseAnd (&)
    var tmp1 = try bitwiseAnd(allocator, Tensor, z, Tensor, z);
    defer tmp1.deinit();
    try std.testing.expect(try allClose(allocator, tmp1, z, 1e-5));
    try assertCommutativeBinop(allocator, a, b, TestOpType, bitwiseAnd, z);
    try assertCommutativeBinop(allocator, z, b, TestOpType, bitwiseAnd, z);
    try assertCommutativeBinop(allocator, d, b, TestOpType, bitwiseAnd, b);
    try assertScalarTensorCommutativeBinop(allocator, 0b0000, b, TestOpType, bitwiseAnd, z);
    try assertScalarTensorCommutativeBinop(allocator, 0b0110, b, TestOpType, bitwiseAnd, b);

    // test bitwiseOr (|)
    var tmp2 = try bitwiseOr(allocator, Tensor, z, Tensor, z);
    defer tmp2.deinit();
    try std.testing.expect(try allClose(allocator, tmp2, z, 1e-5));
    try assertCommutativeBinop(allocator, a, z, TestOpType, bitwiseOr, a);
    try assertCommutativeBinop(allocator, z, b, TestOpType, bitwiseOr, b);
    try assertCommutativeBinop(allocator, a, b, TestOpType, bitwiseOr, c);
    try assertScalarTensorCommutativeBinop(allocator, 0b0000, b, TestOpType, bitwiseOr, b);
    try assertScalarTensorCommutativeBinop(allocator, 0b0110, b, TestOpType, bitwiseOr, d);

    // test bitwiseXor (^)
    var tmp3 = try bitwiseXor(allocator, Tensor, z, Tensor, z);
    defer tmp3.deinit();
    try std.testing.expect(try allClose(allocator, tmp3, z, 1e-5));
    try assertCommutativeBinop(allocator, a, z, TestOpType, bitwiseXor, a);
    try assertCommutativeBinop(allocator, z, b, TestOpType, bitwiseXor, b);
    try assertCommutativeBinop(allocator, a, b, TestOpType, bitwiseXor, c);
    try assertCommutativeBinop(allocator, c, c, TestOpType, bitwiseXor, z);
    try assertScalarTensorCommutativeBinop(allocator, 0b0000, b, TestOpType, bitwiseXor, b);
    try assertScalarTensorCommutativeBinop(allocator, 0b1001, a, TestOpType, bitwiseXor, e);

    // TODO test scalar input (need right/left_shift operator)
    var tmp4 = try lShift(allocator, Tensor, z, Tensor, z);
    defer tmp4.deinit();
    try std.testing.expect(try allClose(allocator, tmp4, z, 1e-5));
    var tmp5 = try lShift(allocator, Tensor, a, Tensor, z);
    defer tmp5.deinit();
    try std.testing.expect(try allClose(allocator, tmp5, a, 1e-5));
    var tmp6 = try lShift(allocator, Tensor, z, Tensor, a);
    defer tmp6.deinit();
    try std.testing.expect(try allClose(allocator, tmp6, z, 1e-5));
    var tmp7 = try lShift(allocator, Tensor, b, Tensor, g);
    defer tmp7.deinit();
    try std.testing.expect(try allClose(allocator, tmp7, h, 1e-5));

    var tmp8 = try rShift(allocator, Tensor, z, Tensor, z);
    defer tmp8.deinit();
    try std.testing.expect(try allClose(allocator, tmp8, z, 1e-5));
    var tmp9 = try rShift(allocator, Tensor, a, Tensor, z);
    defer tmp9.deinit();
    try std.testing.expect(try allClose(allocator, tmp9, a, 1e-5));
    var tmp10 = try rShift(allocator, Tensor, z, Tensor, a);
    defer tmp10.deinit();
    try std.testing.expect(try allClose(allocator, tmp10, z, 1e-5));
    var tmp11 = try rShift(allocator, Tensor, h, Tensor, g);
    defer tmp11.deinit();
    try std.testing.expect(try allClose(allocator, tmp11, b, 1e-5));
}

fn testTensorIncompatibleShapes(allocator: std.mem.Allocator, dtype: DType, lhs: Tensor, rhs: Tensor) !void {
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        add(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        sub(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        mul(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        div(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        eq(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        neq(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        lessThan(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        lessThanEqual(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        greaterThan(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        greaterThanEqual(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        logicalOr(allocator, Tensor, lhs, Tensor, rhs),
    );
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        logicalAnd(allocator, Tensor, lhs, Tensor, rhs),
    );
    // TODO ArrayFire needs software impl for fp16 modulo on CUDA backend;
    // remove this test when zigTensor supports ArrayFire CUDA backend
    // if (dtype != .f16) {
    try std.testing.expectError(
        error.FailedBinaryOpOrBroadcast,
        mod(allocator, Tensor, lhs, Tensor, rhs),
    );
    // }
    // these operators are generally not well-defined for fps
    if (dtype != .f16 and dtype != .f32 and dtype != .f64) {
        try std.testing.expectError(
            error.FailedBinaryOpOrBroadcast,
            bitwiseAnd(allocator, Tensor, lhs, Tensor, rhs),
        );
        try std.testing.expectError(
            error.FailedBinaryOpOrBroadcast,
            bitwiseOr(allocator, Tensor, lhs, Tensor, rhs),
        );
        try std.testing.expectError(
            error.FailedBinaryOpOrBroadcast,
            bitwiseXor(allocator, Tensor, lhs, Tensor, rhs),
        );
        try std.testing.expectError(
            error.FailedBinaryOpOrBroadcast,
            lShift(allocator, Tensor, lhs, Tensor, rhs),
        );
        try std.testing.expectError(
            error.FailedBinaryOpOrBroadcast,
            rShift(allocator, Tensor, lhs, Tensor, rhs),
        );
    }
}

fn testTensorIncompatibleShapesForType(allocator: std.mem.Allocator, dtype: DType) !void {
    const rand = @import("Random.zig").rand;
    var a = try rand(allocator, &.{ 2, 2 }, dtype);
    defer a.deinit();
    var too_many_axises = try rand(allocator, &.{ 4, 5, 6 }, dtype);
    defer too_many_axises.deinit();
    var too_few_axises = try rand(allocator, &.{3}, dtype);
    defer too_few_axises.deinit();
    var diffDim = try rand(allocator, &.{ 2, 3 }, dtype);
    defer diffDim.deinit();
    try testTensorIncompatibleShapes(allocator, dtype, a, too_many_axises);
    try testTensorIncompatibleShapes(allocator, dtype, a, too_few_axises);
    try testTensorIncompatibleShapes(allocator, dtype, a, diffDim);
}

fn releaseBinaryOpRes(res: [2]Tensor) void {
    res[0].deinit();
    res[1].deinit();
}

fn doBinaryOp(allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, tile_shape_lhs: Shape, tile_shape_rhs: Shape, comptime func: TestOpType) ![2]Tensor {
    std.debug.assert(try lhs.ndim(allocator) <= try rhs.ndim(allocator));
    var tile1 = try tile(allocator, lhs, tile_shape_lhs);
    defer tile1.deinit();
    var tile2 = try tile(allocator, rhs, tile_shape_rhs);
    defer tile2.deinit();
    return [2]Tensor{ try func(allocator, Tensor, lhs, Tensor, rhs), try func(allocator, Tensor, tile1, Tensor, tile2) };
}

fn computeBroadcastShape(allocator: std.mem.Allocator, lhs_shape: Shape, rhs_shape: Shape) !Shape {
    const max_ndim: usize = @max(tensor_.shape.ndim(lhs_shape), tensor_.shape.ndim(rhs_shape));
    var out_shape = try allocator.alloc(Dim, max_ndim);
    for (0..max_ndim) |i| {
        if (i > tensor_.shape.ndim(lhs_shape) - 1) {
            out_shape[i] = rhs_shape[i];
        } else if (i > tensor_.shape.ndim(rhs_shape) - 1) {
            out_shape[i] = lhs_shape[i];
        } else if (lhs_shape[i] == 1) {
            out_shape[i] = rhs_shape[i];
        } else if (rhs_shape[i] == 1) {
            out_shape[i] = lhs_shape[i];
        } else if (lhs_shape[i] == rhs_shape[i]) {
            out_shape[i] = lhs_shape[i];
        } else {
            return error.FailedComputeBroadcastShape;
        }
    }
    return out_shape;
}

const TestFunc = *const fn (allocator: std.mem.Allocator, dtype: DType) anyerror!void;

fn applyToAllFpDtypes(allocator: std.mem.Allocator, func: TestFunc) !void {
    try func(allocator, .f16);
    try func(allocator, .f32);
    try func(allocator, .f64);
}

fn applyToAllIntegralDtypes(allocator: std.mem.Allocator, func: TestFunc) !void {
    // TODO casting to `b8` clips values to 0 and 1, which breaks the fixtures
    // func(allocator, .b8);
    try func(allocator, .u8);
    try func(allocator, .s16);
    try func(allocator, .u16);
    try func(allocator, .s32);
    try func(allocator, .u32);
    try func(allocator, .s64);
    try func(allocator, .u64);
}

fn applyToAllDtypes(allocator: std.mem.Allocator, func: TestFunc) !void {
    try applyToAllFpDtypes(allocator, func);
    try applyToAllIntegralDtypes(allocator, func);
}

test "TensorBinaryOpsTest -> ArithmeticBinaryOperators" {
    const allocator = std.testing.allocator;
    defer tensor_.deinit(); // deinit global singletons
    try applyToAllDtypes(allocator, testArithmeticBinops);
}

test "TensorBinaryOpsTest -> ComparisonBinaryOperators" {
    const allocator = std.testing.allocator;
    defer tensor_.deinit(); // deinit global singletons
    try applyToAllDtypes(allocator, testComparisonBinops);
}

test "TensorBinaryOpsTest -> LogicalBinaryOperators" {
    const allocator = std.testing.allocator;
    defer tensor_.deinit(); // deinit global singletons
    try applyToAllDtypes(allocator, testLogicalBinops);
}

test "TensorBinaryOpsTest -> ModuloBinaryOperators" {
    const allocator = std.testing.allocator;
    defer tensor_.deinit(); // deinit global singletons

    // TODO: once ArrayFire CUDA support is added, skip testing for `.f16`
    // as currently lacking software impl for fp16 modulo on CUDA backend
    try applyToAllDtypes(allocator, testModuloBinop);
}

test "TensorBinaryOpsTest -> BitBinaryOperators" {
    const allocator = std.testing.allocator;
    defer tensor_.deinit(); // deinit global singletons

    // ArrayFire doesn't support bit ops for floating point types
    try applyToAllIntegralDtypes(allocator, testBitBinops);
}

test "TensorBinaryOpsTest -> BinaryOperatorIncompatibleShapes" {
    const allocator = std.testing.allocator;
    defer tensor_.deinit(); // deinit global singletons
    try applyToAllDtypes(allocator, testTensorIncompatibleShapesForType);
}

test "TensorBinaryOpsTest -> minimum" {
    const allocator = std.testing.allocator;
    defer tensor_.deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 3, 3 }, f64, 1, .f32);
    defer a.deinit();
    var b = try full(allocator, &.{ 3, 3 }, f64, 2, .f32);
    defer b.deinit();
    var c = try minimum(allocator, Tensor, a, Tensor, b);
    defer c.deinit();
    try std.testing.expect(try a.dtype(allocator) == try b.dtype(allocator));
    try std.testing.expect(try allClose(allocator, a, c, 1e-5));
    var tmp = try minimum(allocator, f64, 1, Tensor, b);
    defer tmp.deinit();
    var tmp1 = try tmp.astype(allocator, try a.dtype(allocator));
    defer tmp1.deinit();
    try std.testing.expect(try allClose(allocator, tmp1, a, 1e-5));
    var tmp2 = try minimum(allocator, Tensor, b, f64, 1);
    defer tmp2.deinit();
    var tmp3 = try tmp2.astype(allocator, try a.dtype(allocator));
    defer tmp3.deinit();
    try std.testing.expect(try allClose(allocator, tmp3, a, 1e-5));
}

test "TensorBinaryOpsTest -> maximum" {
    const allocator = std.testing.allocator;
    defer tensor_.deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 3, 3 }, f64, 1, .f32);
    defer a.deinit();
    var b = try full(allocator, &.{ 3, 3 }, f64, 2, .f32);
    defer b.deinit();
    var c = try maximum(allocator, Tensor, a, Tensor, b);
    defer c.deinit();
    try std.testing.expect(try b.dtype(allocator) == try c.dtype(allocator));
    try std.testing.expect(try allClose(allocator, b, c, 1e-5));
    var tmp = try maximum(allocator, f64, 1, Tensor, b);
    defer tmp.deinit();
    var tmp1 = try tmp.astype(allocator, try a.dtype(allocator));
    defer tmp1.deinit();
    try std.testing.expect(try allClose(allocator, tmp1, b, 1e-5));
    var tmp2 = try maximum(allocator, Tensor, b, f64, 1);
    defer tmp2.deinit();
    var tmp3 = try tmp2.astype(allocator, try a.dtype(allocator));
    defer tmp3.deinit();
    try std.testing.expect(try allClose(allocator, tmp3, b, 1e-5));
}

test "TensorBinaryOpsTest -> broadcasting" {
    const allocator = std.testing.allocator;
    const rand = @import("Random.zig").rand;
    defer tensor_.deinit(); // deinit global singletons

    // Collection of {lhs, rhs, tileShapeLhs, tileShapeRhs} corresponding to
    // broadcasting [lhs] to [rhs] by tiling by the the respective tileShapes
    const ShapeData = struct {
        lhs: Shape, // broadcast from
        rhs: Shape, // broadcast to
        tile_shape_lhs: Shape,
        tile_shape_rhs: Shape,
    };
    var shapes: []const ShapeData = &.{
        .{ .lhs = &.{ 3, 1 }, .rhs = &.{ 3, 3 }, .tile_shape_lhs = &.{ 1, 3 }, .tile_shape_rhs = &.{ 1, 1 } },
        .{ .lhs = &.{3}, .rhs = &.{ 3, 3 }, .tile_shape_lhs = &.{ 1, 3 }, .tile_shape_rhs = &.{ 1, 1 } },
        .{ .lhs = &.{ 3, 1, 4 }, .rhs = &.{ 3, 6, 4 }, .tile_shape_lhs = &.{ 1, 6, 1 }, .tile_shape_rhs = &.{ 1, 1, 1 } },
        .{ .lhs = &.{ 3, 1, 4, 1 }, .rhs = &.{ 3, 2, 4, 5 }, .tile_shape_lhs = &.{ 1, 2, 1, 5 }, .tile_shape_rhs = &.{ 1, 1, 1, 1 } },
        .{ .lhs = &.{ 1, 10 }, .rhs = &.{ 8, 10 }, .tile_shape_lhs = &.{ 8, 1 }, .tile_shape_rhs = &.{ 1, 1 } },
        .{ .lhs = &.{ 2, 1, 5, 1 }, .rhs = &.{ 2, 3, 5, 3 }, .tile_shape_lhs = &.{ 1, 3, 1, 3 }, .tile_shape_rhs = &.{ 1, 1, 1, 1 } },
        .{ .lhs = &.{ 3, 1, 2, 1 }, .rhs = &.{ 1, 4, 1, 5 }, .tile_shape_lhs = &.{ 1, 4, 1, 5 }, .tile_shape_rhs = &.{ 3, 1, 2, 1 } },
        .{ .lhs = &.{ 3, 2, 1 }, .rhs = &.{ 3, 1, 4, 1 }, .tile_shape_lhs = &.{ 1, 1, 4 }, .tile_shape_rhs = &.{ 1, 2, 1, 1 } },
    };
    const functions = comptime [_]TestOpType{
        minimum,
        maximum,
        power,
        add,
        sub,
        mul,
        div,
        eq,
        neq,
        lessThan,
        lessThanEqual,
        greaterThan,
        greaterThanEqual,
        logicalOr,
        logicalAnd,
        mod,
        bitwiseAnd,
        bitwiseOr,
        bitwiseXor,
        lShift,
        rShift,
    };
    inline for (functions) |funcp| {
        for (shapes) |shape_data| {
            var tmp_lhs = try rand(allocator, shape_data.lhs, .f32);
            defer tmp_lhs.deinit();
            try tmp_lhs.inPlaceAdd(allocator, f64, 1);
            try tmp_lhs.inPlaceMul(allocator, f64, 10);
            var lhs = try tmp_lhs.astype(allocator, .s32);
            defer lhs.deinit();
            var tmp_rhs = try rand(allocator, shape_data.rhs, .f32);
            defer tmp_rhs.deinit();
            try tmp_rhs.inPlaceAdd(allocator, f64, 1);
            try tmp_rhs.inPlaceMul(allocator, f64, 10);
            var rhs = try tmp_rhs.astype(allocator, .s32);
            defer rhs.deinit();

            var res = try doBinaryOp(allocator, lhs, rhs, shape_data.tile_shape_lhs, shape_data.tile_shape_rhs, funcp);
            defer releaseBinaryOpRes(res);
            var expected_shape = try computeBroadcastShape(allocator, shape_data.lhs, shape_data.rhs);
            defer allocator.free(expected_shape);
            try std.testing.expect(tensor_.shape.eql(try res[0].shape(allocator), expected_shape));
            try std.testing.expect(try allClose(allocator, res[0], res[1], 1e-5));
        }

        // Scalar broadcasting
        const scalar_val: f64 = 4;
        const in_shape: Shape = &.{ 2, 3, 4 };
        var tmp_lhs = try rand(allocator, in_shape, .f32);
        defer tmp_lhs.deinit();
        var lhs = try tmp_lhs.astype(allocator, .s32);
        defer lhs.deinit();
        var rhs = try fromScalar(allocator, f64, 4, .s32);
        defer rhs.deinit();
        var rhs_tiled = try full(allocator, in_shape, f64, scalar_val, .s32);
        defer rhs_tiled.deinit();
        var res = try funcp(allocator, Tensor, lhs, Tensor, rhs);
        defer res.deinit();
        var exp_ = try funcp(allocator, Tensor, lhs, Tensor, rhs_tiled);
        defer exp_.deinit();
        try std.testing.expect(try allClose(allocator, res, exp_, 1e-5));
    }
}

test "TensorBinaryOpsTest -> power" {
    const allocator = std.testing.allocator;
    defer tensor_.deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 3, 3 }, f64, 2, .f32);
    defer a.deinit();
    var b = try full(allocator, &.{ 3, 3 }, f64, 2, .f32);
    defer b.deinit();
    var res = try power(allocator, Tensor, a, Tensor, b);
    defer res.deinit();
    var exp_ = try mul(allocator, Tensor, a, Tensor, b);
    defer exp_.deinit();
    try std.testing.expect(try allClose(allocator, res, exp_, 1e-5));
}

test "TensorBinaryOpsTest -> powerDouble" {
    const allocator = std.testing.allocator;
    defer tensor_.deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 3, 3 }, f64, 2, .f32);
    defer a.deinit();
    var res = try power(allocator, Tensor, a, f64, 3);
    defer res.deinit();
    var tmp = try mul(allocator, Tensor, a, Tensor, a);
    defer tmp.deinit();
    var exp1 = try mul(allocator, Tensor, tmp, Tensor, a);
    defer exp1.deinit();
    try std.testing.expect(try allClose(allocator, res, exp1, 1e-5));

    var res2 = try power(allocator, f64, 3, Tensor, a);
    defer res2.deinit();
    var exp2 = try full(allocator, try a.shape(allocator), f64, 3 * 3, .f32);
    defer exp2.deinit();
    try std.testing.expect(try allClose(allocator, res2, exp2, 1e-5));
}

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

test "TensorBinaryOpsTest -> inPlaceAdd" {
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

test "TensorBinaryOpsTest -> inPlaceSub" {
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

test "TensorBinaryOpsTest -> inPlaceMul" {
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

test "TensorBinaryOpsTest -> inPlaceDiv" {
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
