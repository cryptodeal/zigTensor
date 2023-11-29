const std = @import("std");

const assert = std.debug.assert;

pub const FirstOrderOptimizer = struct {
    const Self = @This();

    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
        zeroGrad: *const fn (ctx: *anyopaque) void,
        getLr: *const fn (ctx: *anyopaque) f32,
        setLr: *const fn (ctx: *anyopaque, lr: f32) void,
        step: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!void,
        prettyString: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]const u8,
    };

    /// Free all associated memory.
    pub fn deinit(self: *const Self) void {
        self.vtable.deinit(self.ptr);
    }

    /// Zero the gradients for all the parameters being optimized. Typically
    /// called after every call to step().
    pub fn zeroGrad(self: *const Self) void {
        self.vtable.zeroGrad(self.ptr);
    }

    /// Get the learning rate.
    pub fn getLr(self: *const Self) f32 {
        return self.vtable.getLr(self.ptr);
    }

    /// Set the learning rate.
    pub fn setLr(self: *const Self, lr: f32) void {
        self.vtable.setLr(self.ptr, lr);
    }

    pub fn step(self: *const Self, allocator: std.mem.Allocator) !void {
        return self.vtable.step(self.ptr, allocator);
    }

    /// Generates a stringified representation of the optimizer.
    pub fn prettyString(self: *const Self, allocator: std.mem.Allocator) ![]const u8 {
        return self.vtable.prettyString(self.ptr, allocator);
    }

    pub fn init(ext_impl: anytype) Self {
        const Ptr = @TypeOf(ext_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        const PtrChild = PtrInfo.Pointer.child;
        assert(@typeInfo(PtrChild) == .Struct); // Must point to a struct
        const impl = struct {
            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }

            fn zeroGrad(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.zeroGrad();
            }

            fn getLr(ctx: *anyopaque) f32 {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getLr();
            }

            fn setLr(ctx: *anyopaque, lr: f32) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.setLr(lr);
            }

            fn step(ctx: *anyopaque, allocator: std.mem.Allocator) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.step(allocator);
            }

            fn prettyString(ctx: *anyopaque, allocator: std.mem.Allocator) ![]const u8 {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.prettyString(allocator);
            }
        };
        return .{
            .ptr = ext_impl,
            .vtable = &.{
                .deinit = impl.deinit,
                .zeroGrad = impl.zeroGrad,
                .getLr = impl.getLr,
                .setLr = impl.setLr,
                .step = impl.step,
                .prettyString = impl.prettyString,
            },
        };
    }
};
