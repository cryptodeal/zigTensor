//! The standard TensorBackend implementation.
const std = @import("std");
const zigrc = @import("zigrc");
const tensor_ = @import("tensor.zig");

const assert = std.debug.assert;
const Tensor = tensor_.Tensor;
const TensorBackendType = tensor_.TensorBackendType;
const DType = tensor_.DType;
const Shape = tensor_.shape.Shape;
const Dim = tensor_.shape.Dim;
const SortMode = tensor_.SortMode;
const MatrixProperty = tensor_.MatrixProperty;
const PadType = tensor_.PadType;
const Index = tensor_.Index;

pub fn areBackendsEqual(self: TensorBackend, other: TensorBackend) bool {
    return self.backendType() == other.backendType();
}

pub const ValIdxRes = struct { values: Tensor, indices: Tensor };

pub const SortIndexRes = struct { out: Tensor, idx: Tensor };

/// A Tensor backend that can be used to store global state associated with a
/// particular tensor implementation.
///
/// This abstraction facilitates adherence to the implementation requirements for
/// global operators that operate on tensors (e.g. those functions that are not
/// members of `zt.Tensor`.
///
/// zigTensor Tensors dispatch to their corresponding backends using
/// `zt.Tensor.backend() --> typeToBackend (see below) to grab the correct
/// instance.
pub const TensorBackend = struct {
    const Self = @This();

    // The type erased pointer to the TensorBackend implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
        backendType: *const fn (ctx: *anyopaque) TensorBackendType,
        eval: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!void,
        supportsDataType: *const fn (ctx: *anyopaque, data_type: DType) anyerror!bool,
        // TODO: getMemMgrInfo: *const fn (ctx: *anyopaque) ,
        // TODO: setMemMgrLogStream: *const fn (ctx: *anyopaque) ,
        // TODO: setMemMgrLoggingEnabled: *const fn (ctx: *anyopaque) ,
        // TODO: setMemMgrFlushInterval: *const fn (ctx: *anyopaque) ,
        setSeed: *const fn (ctx: *anyopaque, seed: u64) anyerror!void,
        randn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, shape: Shape, dtype: DType) anyerror!Tensor,
        rand: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, shape: Shape, dtype: DType) anyerror!Tensor,
        fromScalar: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, value: f64, dtype: DType) anyerror!Tensor,
        full: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, shape: Shape, value: f64, dtype: DType) anyerror!Tensor,
        identity: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, dim: Dim, dtype: DType) anyerror!Tensor,
        arange: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, shape: Shape, seq_dim: Dim, dtype: DType) anyerror!Tensor,
        iota: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, dims: Shape, tile_dims: Shape, dtype: DType) anyerror!Tensor,
        where: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, condition: Tensor, x: Tensor, y: Tensor) anyerror!Tensor,
        topk: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, k: u32, axis: Dim, sort_mode: SortMode) anyerror!ValIdxRes,
        sort: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) anyerror!Tensor,
        sortIndex: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) anyerror!SortIndexRes,
        argsort: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) anyerror!Tensor,
        matmul: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, lhs_prop: MatrixProperty, rhs_prop: MatrixProperty) anyerror!Tensor,
        reshape: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, shape: Shape) anyerror!Tensor,
        transpose: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, axes: Shape) anyerror!Tensor,
        tile: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, shape: Shape) anyerror!Tensor,
        concatenate: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensors: std.ArrayList(Tensor), axis: u32) anyerror!Tensor,
        nonzero: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        pad: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, pad_widths: *const std.ArrayList([2]i32), pad_type: PadType) anyerror!Tensor,
        exp: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        log: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        negative: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        logicalNot: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        log1p: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        sin: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        cos: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        sqrt: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        tanh: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        floor: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        ceil: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        rint: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        absolute: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        sigmoid: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        erf: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        flip: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, dim: u32) anyerror!Tensor,
        clip: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, low: Tensor, high: Tensor) anyerror!Tensor,
        roll: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, shift: Dim, axis: usize) anyerror!Tensor,
        isnan: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        isinf: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        sign: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        tril: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        triu: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!Tensor,
        amin: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) anyerror!Tensor,
        amax: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) anyerror!Tensor,
        min: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) anyerror!ValIdxRes,
        max: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) anyerror!ValIdxRes,
        sum: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) anyerror!Tensor,
        cumsum: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: u32) anyerror!Tensor,
        argmax: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) anyerror!Tensor,
        argmin: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) anyerror!Tensor,
        mean: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) anyerror!Tensor,
        median: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) anyerror!Tensor,
        variance: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, bias: bool, keep_dims: bool) anyerror!Tensor,
        stdev: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) anyerror!Tensor,
        norm: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, p: f64, keep_dims: bool) anyerror!Tensor,
        countNonzero: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) anyerror!Tensor,
        any: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) anyerror!Tensor,
        all: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) anyerror!Tensor,
        assign: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!void,
        getIndexAssignShape: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, target: Tensor, indices: []const Index) anyerror![]Dim, // util used to create constant from numeric literal for index assign
        indexAssign: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) anyerror!void,
        indexAdd: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) anyerror!void,
        indexSub: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) anyerror!void,
        indexMul: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) anyerror!void,
        indexDiv: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) anyerror!void,
        add: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        inPlaceAdd: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!void,
        sub: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        inPlaceSub: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!void,
        mul: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        inPlaceMul: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!void,
        div: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        inPlaceDiv: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!void,
        eq: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        neq: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        lessThan: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        lessThanEqual: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        greaterThan: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        greaterThanEqual: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        logicalOr: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        logicalAnd: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        mod: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        bitwiseAnd: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        bitwiseOr: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        bitwiseXor: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        lShift: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        rShift: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        minimum: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        maximum: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        power: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) anyerror!Tensor,
        print: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) anyerror!void,
    };

    pub fn deinit(self: *const Self) void {
        return self.vtable.deinit(self.ptr);
    }

    pub fn backendType(self: *const Self) TensorBackendType {
        return self.vtable.backendType(self.ptr);
    }

    pub fn eval(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !void {
        return self.vtable.eval(self.ptr, allocator, tensor);
    }

    pub fn supportsDataType(self: *const Self, data_type: DType) !bool {
        return self.vtable.supportsDataType(self.ptr, data_type);
    }

    // TODO: pub fn getMemMgrInfo()

    // TODO: pub fn setMemMgrLogStream()

    // TODO: pub fn setMemMgrLoggingEnabled()

    // TODO: pub fn setMemMgrFlushInterval()

    pub fn setSeed(self: *const Self, seed: u64) !void {
        return self.vtable.setSeed(self.ptr, seed);
    }

    pub fn randn(self: *const Self, allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
        return self.vtable.randn(self.ptr, allocator, shape, dtype);
    }

    pub fn rand(self: *const Self, allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
        return self.vtable.rand(self.ptr, allocator, shape, dtype);
    }

    pub fn fromScalar(self: *const Self, allocator: std.mem.Allocator, comptime T: type, value: T, dtype: DType) !Tensor {
        var f: f64 = undefined;
        switch (@typeInfo(T)) {
            .Float => f = @floatCast(value),
            .Int => f = @floatFromInt(value),
            else => return error.InvalidTypePassedToFromScalar,
        }
        return self.vtable.fromScalar(self.ptr, allocator, f, dtype);
    }

    pub fn full(self: *const Self, allocator: std.mem.Allocator, shape: Shape, comptime T: type, value: T, dtype: DType) !Tensor {
        var f: f64 = undefined;
        switch (@typeInfo(T)) {
            .Float => f = @floatCast(value),
            .Int => f = @floatFromInt(value),
            else => return error.InvalidTypePassedToFromScalar,
        }
        return self.vtable.full(self.ptr, allocator, shape, f, dtype);
    }

    fn getIndexAssignShape(self: *const Self, allocator: std.mem.Allocator, target: Tensor, indices: []const Index) ![]Dim {
        return self.vtable.getIndexAssignShape(self.ptr, allocator, target, indices);
    }

    fn validateDimMatch(_: *const Self, allocator: std.mem.Allocator, og_shape: Shape, target_shape: Shape) !bool {
        // TODO: find a better method of validating this
        var no_empty_dims = std.ArrayList(Dim).init(allocator);
        defer no_empty_dims.deinit();
        for (target_shape) |v| {
            if (v != 1) try no_empty_dims.append(v);
        }
        return std.mem.eql(Dim, no_empty_dims.items, og_shape);
    }

    fn modifyIdxAssignShape(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor, tensor_shape: Shape, target_shape: Shape) !Tensor {
        if (tensor_.shape.elements(target_shape) != tensor_.shape.elements(tensor_shape) or !try self.validateDimMatch(allocator, tensor_shape, target_shape)) {
            std.debug.print("tensor_shape: {any} vs target_shape: {any}\n", .{ tensor_shape, target_shape });
            return error.FailedToModifyIndexAssignShape;
        }
        return self.reshape(allocator, tensor, target_shape);
    }

    pub fn indexAssign(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, comptime T: type, rhs: T, indices: []const Index) !void {
        var init_rhs_tensor = false;
        var rhsTensor: Tensor = undefined;
        defer if (init_rhs_tensor) rhsTensor.deinit();
        var shape = try self.getIndexAssignShape(allocator, lhs, indices);
        defer allocator.free(shape);
        if (T == Tensor) {
            var rhs_shape = try rhs.shape(allocator);
            if (tensor_.shape.eql(shape, rhs_shape)) {
                rhsTensor = rhs;
            } else {
                rhsTensor = try self.modifyIdxAssignShape(allocator, rhs, rhs_shape, shape);
                init_rhs_tensor = true;
            }
        } else {
            rhsTensor = try self.full(allocator, shape, T, rhs, try lhs.dtype(allocator));
            init_rhs_tensor = true;
        }
        return self.vtable.indexAssign(self.ptr, allocator, lhs, rhsTensor, indices);
    }

    pub fn indexAdd(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, comptime T: type, rhs: T, indices: []const Index) !void {
        var init_rhs_tensor = false;
        var rhsTensor: Tensor = undefined;
        defer if (init_rhs_tensor) rhsTensor.deinit();
        var shape = try self.getIndexAssignShape(allocator, lhs, indices);
        defer allocator.free(shape);
        if (T == Tensor) {
            var rhs_shape = try rhs.shape(allocator);
            if (tensor_.shape.eql(shape, rhs_shape)) {
                rhsTensor = rhs;
            } else {
                rhsTensor = try self.modifyIdxAssignShape(allocator, rhs, rhs_shape, shape);
                init_rhs_tensor = true;
            }
        } else {
            rhsTensor = try self.full(allocator, shape, T, rhs, try lhs.dtype(allocator));
            init_rhs_tensor = true;
        }
        return self.vtable.indexAdd(self.ptr, allocator, lhs, rhsTensor, indices);
    }

    pub fn indexSub(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, comptime T: type, rhs: T, indices: []const Index) !void {
        var init_rhs_tensor = false;
        var rhsTensor: Tensor = undefined;
        defer if (init_rhs_tensor) rhsTensor.deinit();
        var shape = try self.getIndexAssignShape(allocator, lhs, indices);
        defer allocator.free(shape);
        if (T == Tensor) {
            var rhs_shape = try rhs.shape(allocator);
            if (tensor_.shape.eql(shape, rhs_shape)) {
                rhsTensor = rhs;
            } else {
                rhsTensor = try self.modifyIdxAssignShape(allocator, rhs, rhs_shape, shape);
                init_rhs_tensor = true;
            }
        } else {
            rhsTensor = try self.full(allocator, &shape, T, rhs, try lhs.dtype(allocator));
            init_rhs_tensor = true;
        }
        return self.vtable.indexSub(self.ptr, allocator, lhs, rhsTensor, indices);
    }

    pub fn indexMul(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, comptime T: type, rhs: T, indices: []const Index) !void {
        var init_rhs_tensor = false;
        var rhsTensor: Tensor = undefined;
        defer if (init_rhs_tensor) rhsTensor.deinit();
        var shape = try self.getIndexAssignShape(allocator, lhs, indices);
        defer allocator.free(shape);
        if (T == Tensor) {
            var rhs_shape = try rhs.shape(allocator);
            if (tensor_.shape.eql(shape, rhs_shape)) {
                rhsTensor = rhs;
            } else {
                rhsTensor = try self.modifyIdxAssignShape(allocator, rhs, rhs_shape, shape);
                init_rhs_tensor = true;
            }
        } else {
            rhsTensor = try self.full(allocator, shape, T, rhs, try lhs.dtype(allocator));
            init_rhs_tensor = true;
        }
        return self.vtable.indexMul(self.ptr, allocator, lhs, rhsTensor, indices);
    }

    pub fn indexDiv(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, comptime T: type, rhs: T, indices: []const Index) !void {
        var init_rhs_tensor = false;
        var rhsTensor: Tensor = undefined;
        defer if (init_rhs_tensor) rhsTensor.deinit();
        var shape = try self.getIndexAssignShape(allocator, lhs, indices);
        defer allocator.free(shape);
        if (T == Tensor) {
            var rhs_shape = try rhs.shape(allocator);
            if (tensor_.shape.eql(shape, rhs_shape)) {
                rhsTensor = rhs;
            } else {
                rhsTensor = try self.modifyIdxAssignShape(allocator, rhs, rhs_shape, &shape);
                init_rhs_tensor = true;
            }
        } else {
            rhsTensor = try self.full(allocator, &shape, T, rhs, try lhs.dtype(allocator));
            init_rhs_tensor = true;
        }
        return self.vtable.indexDiv(self.ptr, allocator, lhs, rhsTensor, indices);
    }

    pub fn identity(self: *const Self, allocator: std.mem.Allocator, dim: Dim, dtype: DType) !Tensor {
        return self.vtable.identity(self.ptr, allocator, dim, dtype);
    }

    pub fn arange(self: *const Self, allocator: std.mem.Allocator, shape: Shape, seq_dim: Dim, dtype: DType) !Tensor {
        return self.vtable.arange(self.ptr, allocator, shape, seq_dim, dtype);
    }

    pub fn iota(self: *const Self, allocator: std.mem.Allocator, dims: Shape, tile_dims: Shape, dtype: DType) !Tensor {
        return self.vtable.iota(self.ptr, allocator, dims, tile_dims, dtype);
    }

    pub fn topk(self: *const Self, allocator: std.mem.Allocator, input: Tensor, k: u32, axis: Dim, sort_mode: SortMode) !ValIdxRes {
        return self.vtable.topk(self.ptr, allocator, input, k, axis, sort_mode);
    }

    pub fn where(self: *const Self, allocator: std.mem.Allocator, condition: Tensor, x: Tensor, y: Tensor) !Tensor {
        return self.vtable.where(self.ptr, allocator, condition, x, y);
    }

    pub fn sort(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
        return self.vtable.sort(self.ptr, allocator, input, axis, sort_mode);
    }

    pub fn sortIndex(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !SortIndexRes {
        return self.vtable.sortIndex(self.ptr, allocator, input, axis, sort_mode);
    }

    pub fn argsort(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
        return self.vtable.argsort(self.ptr, allocator, input, axis, sort_mode);
    }

    pub fn matmul(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, lhs_prop: MatrixProperty, rhs_prop: MatrixProperty) !Tensor {
        return self.vtable.matmul(self.ptr, allocator, lhs, rhs, lhs_prop, rhs_prop);
    }

    pub fn reshape(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor, shape: Shape) !Tensor {
        return self.vtable.reshape(self.ptr, allocator, tensor, shape);
    }

    pub fn transpose(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor, axes: Shape) !Tensor {
        return self.vtable.transpose(self.ptr, allocator, tensor, axes);
    }

    pub fn tile(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor, shape: Shape) !Tensor {
        return self.vtable.tile(self.ptr, allocator, tensor, shape);
    }

    pub fn concatenate(self: *const Self, allocator: std.mem.Allocator, tensors: std.ArrayList(Tensor), axis: u32) !Tensor {
        return self.vtable.concatenate(self.ptr, allocator, tensors, axis);
    }

    pub fn nonzero(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.nonzero(self.ptr, allocator, tensor);
    }

    pub fn pad(self: *const Self, allocator: std.mem.Allocator, input: Tensor, pad_widths: *const std.ArrayList([2]i32), pad_type: PadType) !Tensor {
        return self.vtable.pad(self.ptr, allocator, input, pad_widths, pad_type);
    }

    pub fn exp(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.exp(self.ptr, allocator, tensor);
    }

    pub fn log(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.log(self.ptr, allocator, tensor);
    }

    pub fn negative(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.negative(self.ptr, allocator, tensor);
    }

    pub fn logicalNot(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.logicalNot(self.ptr, allocator, tensor);
    }

    pub fn log1p(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.log1p(self.ptr, allocator, tensor);
    }

    pub fn sin(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.sin(self.ptr, allocator, tensor);
    }

    pub fn cos(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.cos(self.ptr, allocator, tensor);
    }

    pub fn sqrt(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.sqrt(self.ptr, allocator, tensor);
    }

    pub fn tanh(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.tanh(self.ptr, allocator, tensor);
    }

    pub fn floor(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.floor(self.ptr, allocator, tensor);
    }

    pub fn ceil(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.ceil(self.ptr, allocator, tensor);
    }

    pub fn rint(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.rint(self.ptr, allocator, tensor);
    }

    pub fn absolute(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.absolute(self.ptr, allocator, tensor);
    }

    pub fn sigmoid(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.sigmoid(self.ptr, allocator, tensor);
    }

    pub fn erf(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.erf(self.ptr, allocator, tensor);
    }

    pub fn flip(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor, dim: u32) !Tensor {
        return self.vtable.flip(self.ptr, allocator, tensor, dim);
    }

    pub fn clip(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor, low: Tensor, high: Tensor) !Tensor {
        return self.vtable.clip(self.ptr, allocator, tensor, low, high);
    }

    pub fn roll(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor, shift: Dim, axis: usize) !Tensor {
        return self.vtable.roll(self.ptr, allocator, tensor, shift, axis);
    }

    pub fn isnan(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.isnan(self.ptr, allocator, tensor);
    }

    pub fn isinf(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.isinf(self.ptr, allocator, tensor);
    }

    pub fn sign(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.sign(self.ptr, allocator, tensor);
    }

    pub fn tril(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.tril(self.ptr, allocator, tensor);
    }

    pub fn triu(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        return self.vtable.triu(self.ptr, allocator, tensor);
    }

    pub fn amin(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        return self.vtable.amin(self.ptr, allocator, input, axes, keep_dims);
    }

    pub fn amax(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        return self.vtable.amax(self.ptr, allocator, input, axes, keep_dims);
    }

    pub fn min(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !ValIdxRes {
        return self.vtable.min(self.ptr, allocator, input, axis, keep_dims);
    }

    pub fn max(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !ValIdxRes {
        return self.vtable.max(self.ptr, allocator, input, axis, keep_dims);
    }

    pub fn sum(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        return self.vtable.sum(self.ptr, allocator, input, axes, keep_dims);
    }

    pub fn cumsum(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axis: u32) !Tensor {
        return self.vtable.cumsum(self.ptr, allocator, input, axis);
    }

    pub fn argmax(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !Tensor {
        return self.vtable.argmax(self.ptr, allocator, input, axis, keep_dims);
    }

    pub fn argmin(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !Tensor {
        return self.vtable.argmin(self.ptr, allocator, input, axis, keep_dims);
    }

    pub fn mean(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        return self.vtable.mean(self.ptr, allocator, input, axes, keep_dims);
    }

    pub fn median(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        return self.vtable.median(self.ptr, allocator, input, axes, keep_dims);
    }

    pub fn variance(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, bias: bool, keep_dims: bool) !Tensor {
        return self.vtable.variance(self.ptr, allocator, input, axes, bias, keep_dims);
    }

    pub fn stdev(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        return self.vtable.stdev(self.ptr, allocator, input, axes, keep_dims);
    }

    pub fn norm(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, p: f64, keep_dims: bool) !Tensor {
        return self.vtable.norm(self.ptr, allocator, input, axes, p, keep_dims);
    }

    pub fn countNonzero(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        return self.vtable.countNonzero(self.ptr, allocator, input, axes, keep_dims);
    }

    pub fn any(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        return self.vtable.any(self.ptr, allocator, input, axes, keep_dims);
    }

    pub fn all(self: *const Self, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        return self.vtable.all(self.ptr, allocator, input, axes, keep_dims);
    }

    pub fn assign(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
        return self.vtable.assign(self.ptr, allocator, lhs, rhs);
    }

    pub fn add(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.add(self.ptr, allocator, lhs, rhs);
    }

    pub fn inPlaceAdd(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
        return self.vtable.inPlaceAdd(self.ptr, allocator, lhs, rhs);
    }

    pub fn sub(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.sub(self.ptr, allocator, lhs, rhs);
    }

    pub fn inPlaceSub(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
        return self.vtable.inPlaceSub(self.ptr, allocator, lhs, rhs);
    }

    pub fn mul(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.mul(self.ptr, allocator, lhs, rhs);
    }

    pub fn inPlaceMul(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
        return self.vtable.inPlaceMul(self.ptr, allocator, lhs, rhs);
    }

    pub fn div(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.div(self.ptr, allocator, lhs, rhs);
    }

    pub fn inPlaceDiv(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
        return self.vtable.inPlaceDiv(self.ptr, allocator, lhs, rhs);
    }

    pub fn eq(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.eq(self.ptr, allocator, lhs, rhs);
    }

    pub fn neq(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.neq(self.ptr, allocator, lhs, rhs);
    }

    pub fn lessThan(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.lessThan(self.ptr, allocator, lhs, rhs);
    }

    pub fn lessThanEqual(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.lessThanEqual(self.ptr, allocator, lhs, rhs);
    }

    pub fn greaterThan(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.greaterThan(self.ptr, allocator, lhs, rhs);
    }

    pub fn greaterThanEqual(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.greaterThanEqual(self.ptr, allocator, lhs, rhs);
    }

    pub fn logicalOr(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.logicalOr(self.ptr, allocator, lhs, rhs);
    }

    pub fn logicalAnd(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.logicalAnd(self.ptr, allocator, lhs, rhs);
    }

    pub fn mod(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.mod(self.ptr, allocator, lhs, rhs);
    }

    pub fn bitwiseAnd(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.bitwiseAnd(self.ptr, allocator, lhs, rhs);
    }

    pub fn bitwiseOr(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.bitwiseOr(self.ptr, allocator, lhs, rhs);
    }

    pub fn bitwiseXor(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.bitwiseXor(self.ptr, allocator, lhs, rhs);
    }

    pub fn lShift(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.lShift(self.ptr, allocator, lhs, rhs);
    }

    pub fn rShift(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.rShift(self.ptr, allocator, lhs, rhs);
    }

    pub fn minimum(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.minimum(self.ptr, allocator, lhs, rhs);
    }

    pub fn maximum(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.maximum(self.ptr, allocator, lhs, rhs);
    }

    pub fn power(self: *const Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return self.vtable.power(self.ptr, allocator, lhs, rhs);
    }

    pub fn print(self: *const Self, allocator: std.mem.Allocator, tensor: Tensor) !void {
        return self.vtable.print(self.ptr, allocator, tensor);
    }

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

            fn backendType(ctx: *anyopaque) TensorBackendType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.backendType();
            }

            fn eval(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.eval(allocator, tensor);
            }

            fn supportsDataType(ctx: *anyopaque, data_type: DType) !bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.supportsDataType(data_type);
            }

            // TODO: fn getMemMgrInfo()

            // TODO: fn setMemMgrLogStream()

            // TODO: fn setMemMgrLoggingEnabled()

            // TODO: fn setMemMgrFlushInterval()

            fn setSeed(ctx: *anyopaque, seed: u64) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.setSeed(seed);
            }

            fn randn(ctx: *anyopaque, allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.randn(allocator, shape, dtype);
            }

            fn rand(ctx: *anyopaque, allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.rand(allocator, shape, dtype);
            }

            fn fromScalar(ctx: *anyopaque, allocator: std.mem.Allocator, value: f64, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.fromScalar(allocator, value, dtype);
            }

            fn full(ctx: *anyopaque, allocator: std.mem.Allocator, shape: Shape, value: f64, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.full(allocator, shape, value, dtype);
            }

            fn identity(ctx: *anyopaque, allocator: std.mem.Allocator, dim: Dim, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.identity(allocator, dim, dtype);
            }

            fn arange(ctx: *anyopaque, allocator: std.mem.Allocator, shape: Shape, seq_dim: Dim, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.arange(allocator, shape, seq_dim, dtype);
            }

            fn iota(ctx: *anyopaque, allocator: std.mem.Allocator, dims: Shape, tile_dims: Shape, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.iota(allocator, dims, tile_dims, dtype);
            }

            fn where(ctx: *anyopaque, allocator: std.mem.Allocator, condition: Tensor, x: Tensor, y: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.where(allocator, condition, x, y);
            }

            fn topk(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, k: u32, axis: Dim, sort_mode: SortMode) !ValIdxRes {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.topk(allocator, input, k, axis, sort_mode);
            }

            fn sort(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.sort(allocator, input, axis, sort_mode);
            }

            fn sortIndex(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !SortIndexRes {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.sortIndex(allocator, input, axis, sort_mode);
            }

            fn argsort(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.argsort(allocator, input, axis, sort_mode);
            }

            fn matmul(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, lhs_prop: MatrixProperty, rhs_prop: MatrixProperty) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.matmul(allocator, lhs, rhs, lhs_prop, rhs_prop);
            }

            fn reshape(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, shape: Shape) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.reshape(allocator, tensor, shape);
            }

            fn transpose(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, axes: Shape) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.transpose(allocator, tensor, axes);
            }

            fn tile(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, shape: Shape) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.tile(allocator, tensor, shape);
            }

            fn concatenate(ctx: *anyopaque, allocator: std.mem.Allocator, tensors: std.ArrayList(Tensor), axis: u32) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.concatenate(allocator, tensors, axis);
            }

            fn nonzero(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.nonzero(allocator, tensor);
            }

            fn pad(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, pad_widths: *const std.ArrayList([2]i32), pad_type: PadType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.pad(allocator, input, pad_widths, pad_type);
            }

            fn exp(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.exp(allocator, tensor);
            }

            fn log(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.log(allocator, tensor);
            }

            fn negative(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.negative(allocator, tensor);
            }

            fn logicalNot(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.logicalNot(allocator, tensor);
            }

            fn log1p(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.log1p(allocator, tensor);
            }

            fn sin(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.sin(allocator, tensor);
            }

            fn cos(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.cos(allocator, tensor);
            }

            fn sqrt(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.sqrt(allocator, tensor);
            }

            fn tanh(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.tanh(allocator, tensor);
            }

            fn floor(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.floor(allocator, tensor);
            }

            fn ceil(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.ceil(allocator, tensor);
            }

            fn rint(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.rint(allocator, tensor);
            }

            fn absolute(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.absolute(allocator, tensor);
            }

            fn sigmoid(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.sigmoid(allocator, tensor);
            }

            fn erf(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.erf(allocator, tensor);
            }

            fn flip(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, dim: u32) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.flip(allocator, tensor, dim);
            }

            fn clip(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, low: Tensor, high: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.clip(allocator, tensor, low, high);
            }

            fn roll(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, shift: Dim, axis: usize) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.roll(allocator, tensor, shift, axis);
            }

            fn isnan(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isnan(allocator, tensor);
            }

            fn isinf(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isinf(allocator, tensor);
            }

            fn sign(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.sign(allocator, tensor);
            }

            fn tril(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.tril(allocator, tensor);
            }

            fn triu(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.triu(allocator, tensor);
            }

            fn amin(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.amin(allocator, input, axes, keep_dims);
            }

            fn amax(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.amax(allocator, input, axes, keep_dims);
            }

            fn min(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !ValIdxRes {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.min(allocator, input, axis, keep_dims);
            }

            fn max(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !ValIdxRes {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.max(allocator, input, axis, keep_dims);
            }

            fn sum(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.sum(allocator, input, axes, keep_dims);
            }

            fn cumsum(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: u32) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.cumsum(allocator, input, axis);
            }

            fn argmax(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.argmax(allocator, input, axis, keep_dims);
            }

            fn argmin(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.argmin(allocator, input, axis, keep_dims);
            }

            fn mean(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.mean(allocator, input, axes, keep_dims);
            }

            fn median(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.median(allocator, input, axes, keep_dims);
            }

            fn variance(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, bias: bool, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.variance(allocator, input, axes, bias, keep_dims);
            }

            fn stdev(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.stdev(allocator, input, axes, keep_dims);
            }

            fn norm(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, p: f64, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.norm(allocator, input, axes, p, keep_dims);
            }

            fn countNonzero(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.countNonzero(allocator, input, axes, keep_dims);
            }

            fn any(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.any(allocator, input, axes, keep_dims);
            }

            fn all(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.all(allocator, input, axes, keep_dims);
            }

            fn assign(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.assign(allocator, lhs, rhs);
            }

            fn getIndexAssignShape(ctx: *anyopaque, allocator: std.mem.Allocator, target: Tensor, indices: []const Index) ![]Dim {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getIndexAssignShape(allocator, target, indices);
            }

            fn indexAssign(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.indexAssign(allocator, lhs, rhs, indices);
            }

            fn indexAdd(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.indexAdd(allocator, lhs, rhs, indices);
            }

            fn indexSub(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.indexSub(allocator, lhs, rhs, indices);
            }

            fn indexMul(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.indexMul(allocator, lhs, rhs, indices);
            }

            fn indexDiv(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.indexDiv(allocator, lhs, rhs, indices);
            }

            fn add(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.add(allocator, lhs, rhs);
            }

            fn inPlaceAdd(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.inPlaceAdd(allocator, lhs, rhs);
            }

            fn sub(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.sub(allocator, lhs, rhs);
            }

            fn inPlaceSub(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.inPlaceSub(allocator, lhs, rhs);
            }

            fn mul(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.mul(allocator, lhs, rhs);
            }

            fn inPlaceMul(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.inPlaceMul(allocator, lhs, rhs);
            }

            fn div(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.div(allocator, lhs, rhs);
            }

            fn inPlaceDiv(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.inPlaceDiv(allocator, lhs, rhs);
            }

            fn eq(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.eq(allocator, lhs, rhs);
            }

            fn neq(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.neq(allocator, lhs, rhs);
            }

            fn lessThan(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.lessThan(allocator, lhs, rhs);
            }

            fn lessThanEqual(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.lessThanEqual(allocator, lhs, rhs);
            }

            fn greaterThan(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.greaterThan(allocator, lhs, rhs);
            }

            fn greaterThanEqual(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.greaterThanEqual(allocator, lhs, rhs);
            }

            fn logicalOr(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.logicalOr(allocator, lhs, rhs);
            }

            fn logicalAnd(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.logicalAnd(allocator, lhs, rhs);
            }

            fn mod(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.mod(allocator, lhs, rhs);
            }

            fn bitwiseAnd(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.bitwiseAnd(allocator, lhs, rhs);
            }

            fn bitwiseOr(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.bitwiseOr(allocator, lhs, rhs);
            }

            fn bitwiseXor(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.bitwiseXor(allocator, lhs, rhs);
            }

            fn lShift(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.lShift(allocator, lhs, rhs);
            }

            fn rShift(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.rShift(allocator, lhs, rhs);
            }

            fn minimum(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.minimum(allocator, lhs, rhs);
            }

            fn maximum(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.maximum(allocator, lhs, rhs);
            }

            fn power(ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.power(allocator, lhs, rhs);
            }

            fn print(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.print(allocator, tensor);
            }
        };
        return .{
            .ptr = backend_impl,
            .vtable = &.{
                .deinit = impl.deinit,
                .backendType = impl.backendType,
                .eval = impl.eval,
                .supportsDataType = impl.supportsDataType,
                // TODO: .getMemMgrInfo = impl.getMemMgrInfo,
                // TODO: .setMemMgrLogStream = impl.setMemMgrLogStream,
                // TODO: .setMemMgrLoggingEnabled = impl.setMemMgrLoggingEnabled,
                // TODO: .setMemMgrFlushInterval = impl.setMemMgrFlushInterval,
                .setSeed = impl.setSeed,
                .randn = impl.randn,
                .rand = impl.rand,
                .fromScalar = impl.fromScalar,
                .full = impl.full,
                .identity = impl.identity,
                .arange = impl.arange,
                .iota = impl.iota,
                .where = impl.where,
                .topk = impl.topk,
                .sort = impl.sort,
                .sortIndex = impl.sortIndex,
                .argsort = impl.argsort,
                .matmul = impl.matmul,
                .reshape = impl.reshape,
                .transpose = impl.transpose,
                .tile = impl.tile,
                .concatenate = impl.concatenate,
                .nonzero = impl.nonzero,
                .pad = impl.pad,
                .exp = impl.exp,
                .log = impl.log,
                .negative = impl.negative,
                .logicalNot = impl.logicalNot,
                .log1p = impl.log1p,
                .sin = impl.sin,
                .cos = impl.cos,
                .sqrt = impl.sqrt,
                .tanh = impl.tanh,
                .floor = impl.floor,
                .ceil = impl.ceil,
                .rint = impl.rint,
                .absolute = impl.absolute,
                .sigmoid = impl.sigmoid,
                .erf = impl.erf,
                .flip = impl.flip,
                .clip = impl.clip,
                .roll = impl.roll,
                .isnan = impl.isnan,
                .isinf = impl.isinf,
                .sign = impl.sign,
                .tril = impl.tril,
                .triu = impl.triu,
                .amin = impl.amin,
                .amax = impl.amax,
                .min = impl.min,
                .max = impl.max,
                .sum = impl.sum,
                .cumsum = impl.cumsum,
                .argmax = impl.argmax,
                .argmin = impl.argmin,
                .mean = impl.mean,
                .median = impl.median,
                .variance = impl.variance,
                .stdev = impl.stdev,
                .norm = impl.norm,
                .countNonzero = impl.countNonzero,
                .any = impl.any,
                .all = impl.all,
                .assign = impl.assign,
                .getIndexAssignShape = impl.getIndexAssignShape,
                .indexAssign = impl.indexAssign,
                .indexAdd = impl.indexAdd,
                .indexSub = impl.indexSub,
                .indexMul = impl.indexMul,
                .indexDiv = impl.indexDiv,
                .add = impl.add,
                .inPlaceAdd = impl.inPlaceAdd,
                .sub = impl.sub,
                .inPlaceSub = impl.inPlaceSub,
                .mul = impl.mul,
                .inPlaceMul = impl.inPlaceMul,
                .div = impl.div,
                .inPlaceDiv = impl.inPlaceDiv,
                .eq = impl.eq,
                .neq = impl.neq,
                .lessThan = impl.lessThan,
                .lessThanEqual = impl.lessThanEqual,
                .greaterThan = impl.greaterThan,
                .greaterThanEqual = impl.greaterThanEqual,
                .logicalOr = impl.logicalOr,
                .logicalAnd = impl.logicalAnd,
                .mod = impl.mod,
                .bitwiseAnd = impl.bitwiseAnd,
                .bitwiseOr = impl.bitwiseOr,
                .bitwiseXor = impl.bitwiseXor,
                .lShift = impl.lShift,
                .rShift = impl.rShift,
                .minimum = impl.minimum,
                .maximum = impl.maximum,
                .power = impl.power,
                .print = impl.print,
            },
        };
    }
};
