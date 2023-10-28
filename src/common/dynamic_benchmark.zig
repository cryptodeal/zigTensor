const std = @import("std");

const assert = std.debug.assert;

pub const DynamicBenchmarkOptionsBase = struct {
    const Self = @This();
    // The type erased pointer to the DynamicBenchmarkOptions implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
    };

    /// Free all associated memory.
    pub fn deinit(self: *const Self) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn init(opt_impl: anytype) Self {
        const Ptr = @TypeOf(opt_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }
        };
        return .{
            .ptr = opt_impl,
            .vtable = &.{
                .deinit = impl.deinit,
            },
        };
    }
};
