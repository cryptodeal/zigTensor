//! The standard TensorBackend implementation.

const std = @import("std");
const zigrc = @import("zigrc");
const DType = @import("types.zig").DType;
const base = @import("TensorBase.zig");
const zt_shape = @import("Shape.zig");

const assert = std.debug.assert;
const Tensor = base.Tensor;
const Shape = zt_shape.Shape;
const Dim = zt_shape.Dim;
const TensorBackendType = base.TensorBackendType;

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
        backendType: *const fn (ctx: *const anyopaque) TensorBackendType,
        eval: *const fn (ctx: *const anyopaque) anyerror!void,
        supportsDataType: *const fn (ctx: *const anyopaque, data_type: DType) anyerror!bool,
        // TODO: getMemMgrInfo: *const fn (ctx: *anyopaque) ,
        // TODO: setMemMgrLogStream: *const fn (ctx: *anyopaque) ,
        // TODO: setMemMgrLoggingEnabled: *const fn (ctx: *anyopaque) ,
        // TODO: setMemMgrFlushInterval: *const fn (ctx: *anyopaque) ,
        setSeed: *const fn (ctx: *const anyopaque, seed: u64) anyerror!void,
        randn: *const fn (ctx: *const anyopaque, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) anyerror!Tensor,
        rand: *const fn (ctx: *const anyopaque, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) anyerror!Tensor,
        // TODO: fromScalar: *const fn (ctx: *const anyopaque, allocator: std.mem.Allocator, ) ,
        // TODO: full: *const fn (ctx: *const anyopaque, allocator: std.mem.Allocator, ) ,
        identity: *const fn (ctx: *const anyopaque, allocator: std.mem.Allocator, dim: Dim, dtype: DType) anyerror!Tensor,
        arange: *const fn (ctx: *const anyopaque, allocator: std.mem.Allocator, shape: *const Shape, seq_dim: Dim, dtype: DType) anyerror!Tensor,
        iota: *const fn (ctx: *const anyopaque, allocator: std.mem.Allocator, dims: *const Shape, tile_dims: *const Shape, dtype: DType) anyerror!Tensor,
    };

    pub fn deinit(self: *Self) void {
        return self.vtable.deinit(self.ptr);
    }

    pub fn backendType(self: *Self) TensorBackendType {
        return self.vtable.backendType(self.ptr);
    }

    pub fn eval(self: *Self) !void {
        return self.vtable.eval(self.ptr);
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

    pub fn identity(self: *Self, allocator: std.mem.Allocator, dim: Dim, dtype: DType) !Tensor {
        return self.vtable.identity(self.ptr, allocator, dim, dtype);
    }

    pub fn arange(self: *Self, allocator: std.mem.Allocator, shape: *const Shape, seq_dim: Dim, dtype: DType) !Tensor {
        return self.vtable.arange(self.ptr, allocator, shape, seq_dim, dtype);
    }

    pub fn iota(self: *Self, allocator: std.mem.Allocator, dims: *const Shape, tile_dims: *const Shape, dtype: DType) !Tensor {
        return self.vtable.iota(self.ptr, allocator, dims, tile_dims, dtype);
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

            fn backendType(ctx: *const anyopaque) TensorBackendType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.trialRunStarted();
            }

            fn eval(ctx: *const anyopaque) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.trialRunFinished();
            }

            fn supportsDataType(ctx: *const anyopaque, data_type: DType) !bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.supportsDataType(data_type);
            }

            fn setSeed(ctx: *const anyopaque, seed: u64) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.setSeed(seed);
            }

            fn randn(ctx: *const anyopaque, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.randn(allocator, shape, dtype);
            }

            fn rand(ctx: *const anyopaque, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.rand(allocator, shape, dtype);
            }

            fn identity(ctx: *const anyopaque, allocator: std.mem.Allocator, dim: Dim, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.identity(allocator, dim, dtype);
            }

            fn arange(ctx: *const anyopaque, allocator: std.mem.Allocator, shape: *const Shape, seq_dim: Dim, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.arange(allocator, shape, seq_dim, dtype);
            }

            fn iota(ctx: *const anyopaque, allocator: std.mem.Allocator, dims: *const Shape, tile_dims: *const Shape, dtype: DType) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.iota(allocator, dims, tile_dims, dtype);
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
                .identity = impl.identity,
                .arange = impl.arange,
                .iota = impl.iota,
            },
        };
    }
};
