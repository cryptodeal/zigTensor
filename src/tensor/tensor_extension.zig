const std = @import("std");
const assert = std.debug.assert;

/// A runtime type denoting the tensor extension.
pub const TensorExtensionType = enum(u8) {
    Generic, // placeholder
    Autograd,
    Vision,
    JitOptimizer,
};

pub const TensorExtension = struct {
    const Self = @This();
    // The type erased pointer to the DynamicBenchmarkOptions implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
        getExtensionType: *const fn (ctx: *anyopaque) TensorExtensionType,
    };

    /// Free all associated memory.
    pub fn deinit(self: *const Self) void {
        self.vtable.deinit(self.ptr);
    }

    /// Returns the type of the extension.
    pub fn getExtensionType(self: *const Self) TensorExtensionType {
        return self.vtable.getExtensionType(self.ptr);
    }

    pub fn init(ext_impl: anytype) Self {
        const Ptr = @TypeOf(ext_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }

            fn getExtensionType(ctx: *anyopaque) TensorExtensionType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getExtensionType();
            }
        };
        return .{
            .ptr = ext_impl,
            .vtable = &.{
                .deinit = impl.deinit,
            },
        };
    }
};
