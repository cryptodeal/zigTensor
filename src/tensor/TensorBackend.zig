//! The standard TensorBackend implementation.

const std = @import("std");
const DType = @import("types.zig");
const base = @import("TensorBase.zig");

const assert = std.debug.assert;
const Tensor = base.Tensor;
const TensorBackendType = base.TensorBackendType;

const TensorBackend = struct {
    const Self = @This();

    // The type erased pointer to the TensorBackend implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        backendType: *const fn (ctx: *anyopaque) TensorBackendType,
        eval: *const fn (ctx: *anyopaque) void,
        supportsDataType: *const fn (ctx: *anyopaque, data_type: DType) bool,
    };

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
                .backendType = impl.backendType,
                .eval = impl.eval,
                .supportsDataType = impl.supportsDataType,
            },
        };
    }
};
