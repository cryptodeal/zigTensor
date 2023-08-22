//! The standard TensorBackend implementation.

const std = @import("std");
const zigrc = @import("zigrc");
const DType = @import("types.zig");
const base = @import("TensorBase.zig");

const assert = std.debug.assert;
const Tensor = base.Tensor;
const TensorBackendType = base.TensorBackendType;

pub const TensorBackend = struct {
    const Self = @This();

    // The type erased pointer to the TensorBackend implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        clone: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) Self,
        backendType: *const fn (ctx: *anyopaque) TensorBackendType,
        eval: *const fn (ctx: *anyopaque) void,
        supportsDataType: *const fn (ctx: *anyopaque, data_type: DType) bool,
    };

    pub fn clone(self: *Self, allocator: std.mem.Allocator) zigrc.Arc(Self) {
        return self.vtable.clone(self.ptr, allocator);
    }

    pub fn backendType(self: *Self) TensorBackendType {
        return self.vtable.backendType(self.ptr);
    }

    pub fn eval(self: *Self) void {
        self.vtable.eval(self.ptr);
    }

    pub fn supportsDataType(self: *Self, data_type: DType) bool {
        return self.vtable.supportsDataType(self.ptr, data_type);
    }

    pub fn init(backend_impl: anytype) Self {
        const Ptr = @TypeOf(backend_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn clone(ctx: *anyopaque, allocator: std.mem.Allocator) zigrc.Arc(Self) {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return zigrc.Arc(Self).init(allocator, Self.init(self.clone(allocator)));
            }

            fn backendType(ctx: *anyopaque) TensorBackendType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.trialRunStarted();
            }

            fn eval(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.trialRunFinished();
            }

            fn supportsDataType(ctx: *anyopaque, data_type: DType) bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.supportsDataType(data_type);
            }
        };
        return .{
            .ptr = backend_impl,
            .vtable = &.{
                .clone = impl.clone,
                .backendType = impl.backendType,
                .eval = impl.eval,
                .supportsDataType = impl.supportsDataType,
            },
        };
    }
};
