//! The standard TensorBackend implementation.

const std = @import("std");
const zigrc = @import("zigrc");
const DType = @import("Types.zig").DType;
const base = @import("TensorBase.zig");
const zt_shape = @import("Shape.zig");

const assert = std.debug.assert;
const Tensor = base.Tensor;
const SortMode = base.SortMode;
const MatrixProperty = base.MatrixProperty;
const Shape = zt_shape.Shape;
const Dim = zt_shape.Dim;
const TensorBackendType = base.TensorBackendType;

pub fn areBackendsEqual(self: TensorBackend, other: TensorBackend) bool {
    return self.backendType() == other.backendType();
}

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
        randn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) anyerror!Tensor,
        rand: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) anyerror!Tensor,
        constant: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, shape: ?*const Shape, value: f64, dtype: DType) anyerror!Tensor,
        identity: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, dim: Dim, dtype: DType) anyerror!Tensor,
        arange: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, shape: *const Shape, seq_dim: Dim, dtype: DType) anyerror!Tensor,
        iota: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, dims: *const Shape, tile_dims: *const Shape, dtype: DType) anyerror!Tensor,
        where: *const fn (ctx: *anyopaque, condition: Tensor, x: Tensor, y: Tensor) anyerror!Tensor,
        topk: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, k: u32, axis: Dim, sort_mode: SortMode) anyerror!struct { values: Tensor, indices: Tensor },
        sort: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) anyerror!Tensor,
        sortIndex: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) anyerror!struct { out: Tensor, idx: Tensor },
        argsort: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) anyerror!Tensor,
        matmul: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, lhs_prop: MatrixProperty, rhs_prop: MatrixProperty) anyerror!Tensor,
        reshape: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, shape: *const Shape) anyerror!Tensor,
    };

    pub fn fromScalar(self: *Self, allocator: std.mem.Allocator, comptime T: type, value: T, dtype: DType) !Tensor {
        var f: f64 = undefined;
        switch (@typeInfo(T)) {
            .Float => f = @floatCast(value),
            .Int => f = @floatFromInt(value),
            else => return error.InvalidTypePassedToFromScalar,
        }
        return self.vtable.constant(self.ptr, allocator, null, f, dtype);
    }

    pub fn full(self: *Self, allocator: std.mem.Allocator, shape: *const Shape, comptime T: type, value: T, dtype: DType) !Tensor {
        var f: f64 = undefined;
        switch (@typeInfo(T)) {
            .Float => f = @floatCast(value),
            .Int => f = @floatFromInt(value),
            else => return error.InvalidTypePassedToFromScalar,
        }
        return self.vtable.constant(self.ptr, allocator, shape, f, dtype);
    }

    pub fn deinit(self: *Self) void {
        return self.vtable.deinit(self.ptr);
    }

    pub fn backendType(self: *Self) TensorBackendType {
        return self.vtable.backendType(self.ptr);
    }

    pub fn eval(self: *Self, allocator: std.mem.Allocator, tensor: Tensor) !void {
        return self.vtable.eval(self.ptr, allocator, tensor);
    }

    pub fn supportsDataType(self: *Self, data_type: DType) !bool {
        return self.vtable.supportsDataType(self.ptr, data_type);
    }

    pub fn setSeed(self: *Self, seed: u64) !void {
        return self.vtable.setSeed(self.ptr, seed);
    }

    pub fn randn(self: *Self, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) !Tensor {
        return self.vtable.randn(self.ptr, allocator, shape, dtype);
    }

    pub fn rand(self: *Self, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) !Tensor {
        return self.vtable.rand(self.ptr, allocator, shape, dtype);
    }

    pub fn constant(self: *Self, allocator: std.mem.Allocator, shape: ?*const Shape, value: f64, dtype: DType) !Tensor {
        return self.vtable.constant(self.ptr, allocator, shape, value, dtype);
    }

    pub fn constantI64(self: *Self, allocator: std.mem.Allocator, shape: ?*const Shape, value: i64, dtype: DType) !Tensor {
        return self.vtable.constantI64(self.ptr, allocator, shape, value, dtype);
    }

    pub fn constantU64(self: *Self, allocator: std.mem.Allocator, shape: ?*const Shape, value: u64, dtype: DType) !Tensor {
        return self.vtable.constantU64(self.ptr, allocator, shape, value, dtype);
    }

    pub fn identity(self: *Self, allocator: std.mem.Allocator, dim: Dim, dtype: DType) !Tensor {
        return self.vtable.identity(self.ptr, allocator, dim, dtype);
    }

    pub fn arange(self: *Self, allocator: std.mem.Allocator, shape: *const Shape, seq_dim: Dim, dtype: DType) !Tensor {
        return self.vtable.arange(self.ptr, allocator, shape, seq_dim, dtype);
    }

    pub fn iota(self: *Self, allocator: std.mem.Allocator, dims: *const Shape, tile_dims: *const Shape, dtype: DType) !Tensor {
        return self.vtable.iota(self.ptr, allocator, dims, tile_dims, dtype);
    }

    pub fn topk(self: *Self, allocator: std.mem.Allocator, input: Tensor, k: u32, axis: Dim, sort_mode: SortMode) !struct { values: Tensor, indices: Tensor } {
        return self.vtable.topk(self.ptr, allocator, input, k, axis, sort_mode);
    }

    pub fn where(self: *Self, condition: Tensor, x: Tensor, y: Tensor) !Tensor {
        return self.vtable.where(self.ptr, condition, x, y);
    }

    pub fn sort(self: *Self, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
        return self.vtable.sort(self.ptr, allocator, input, axis, sort_mode);
    }

    pub fn sortIndex(self: *Self, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !struct { out: Tensor, idx: Tensor } {
        return self.vtable.sortIndex(self.ptr, allocator, input, axis, sort_mode);
    }

    pub fn argsort(self: *Self, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
        return self.vtable.argsort(self.ptr, allocator, input, axis, sort_mode);
    }

    pub fn matmul(self: *Self, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, lhs_prop: MatrixProperty, rhs_prop: MatrixProperty) !Tensor {
        return self.vtable.matmul(self.ptr, allocator, lhs, rhs, lhs_prop, rhs_prop);
    }

    pub fn reshape(self: *Self, allocator: std.mem.Allocator, tensor: Tensor, shape: *const Shape) !Tensor {
        return self.vtable.reshape(self.ptr, allocator, tensor, shape);
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

            fn setSeed(ctx: *anyopaque, seed: u64) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.setSeed(seed);
            }

            fn randn(ctx: *anyopaque, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.randn(allocator, shape, dtype);
            }

            fn rand(ctx: *anyopaque, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.rand(allocator, shape, dtype);
            }

            fn constant(ctx: *anyopaque, allocator: std.mem.Allocator, shape: ?*const Shape, value: f64, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.constant(allocator, shape, value, dtype);
            }

            fn identity(ctx: *anyopaque, allocator: std.mem.Allocator, dim: Dim, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.identity(allocator, dim, dtype);
            }

            fn arange(ctx: *anyopaque, allocator: std.mem.Allocator, shape: *const Shape, seq_dim: Dim, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.arange(allocator, shape, seq_dim, dtype);
            }

            fn iota(ctx: *anyopaque, allocator: std.mem.Allocator, dims: *const Shape, tile_dims: *const Shape, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.iota(allocator, dims, tile_dims, dtype);
            }

            fn where(ctx: *anyopaque, condition: Tensor, x: Tensor, y: Tensor) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.where(condition, x, y);
            }

            fn topk(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, k: u32, axis: Dim, sort_mode: SortMode) !struct { values: Tensor, indices: Tensor } {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.topk(allocator, input, k, axis, sort_mode);
            }

            fn sort(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.sort(allocator, input, axis, sort_mode);
            }

            fn sortIndex(ctx: *anyopaque, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !struct { out: Tensor, idx: Tensor } {
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

            fn reshape(ctx: *anyopaque, allocator: std.mem.Allocator, tensor: Tensor, shape: *const Shape) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.reshape(allocator, tensor, shape);
            }
        };
        return .{
            .ptr = backend_impl,
            .vtable = &.{
                .deinit = impl.deinit,
                .backendType = impl.backendType,
                .eval = impl.eval,
                .supportsDataType = impl.supportsDataType,
                .setSeed = impl.setSeed,
                .randn = impl.randn,
                .rand = impl.rand,
                .constant = impl.constant,
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
            },
        };
    }
};
