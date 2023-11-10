const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;

pub const DevicePtr = struct {
    allocator: std.mem.Allocator = undefined,
    ptr_: ?*anyopaque = null,
    tensor_: ?Tensor = null,

    pub fn init(allocator: std.mem.Allocator, in: Tensor) !DevicePtr {
        var self: DevicePtr = .{
            .allocator = allocator,
            .tensor_ = try in.shallowCopy(allocator),
        };
        errdefer self.tensor_.?.deinit();
        if (!try self.tensor_.?.isEmpty(allocator)) {
            if (!try self.tensor_.?.isContiguous(allocator)) {
                std.debug.print("can't get device pointer of non-contiguous Tensor\n", .{});
                return error.DevicePtrNonContiguousTensor;
            }
            self.ptr_ = try self.tensor_.?.device(allocator, ?*anyopaque);
        }
        return self;
    }

    pub fn initDevicePtr(allocator: std.mem.Allocator, d: *DevicePtr) !DevicePtr {
        var self: DevicePtr = .{
            .allocator = allocator,
            .ptr_ = d.ptr_,
            .tensor_ = d.tensor_,
        };
        d.ptr_ = null;
        d.tensor_ = null;
        return self;
    }

    pub fn deinit(self: *DevicePtr) void {
        if (self.ptr_ != null) {
            self.tensor_.?.unlock(self.allocator) catch unreachable;
        }
        if (self.tensor_ != null) {
            self.tensor_.?.deinit();
        }
    }

    pub fn assign(self: *DevicePtr, allocator: std.mem.Allocator, other: *DevicePtr) !void {
        if (self.ptr_ != null) {
            try self.tensor_.?.unlock(allocator);
        }
        if (self.tensor_ != null) {
            self.tensor_.?.deinit();
        }
        self.tensor_ = other.tensor_;
        other.tensor_ = null;
        self.ptr_ = other.ptr_;
        other.ptr_ = null;
    }

    pub fn get(self: *const DevicePtr) ?*anyopaque {
        return self.ptr_;
    }
};

test "DevicePtr -> Null" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    var x = try Tensor.initEmpty(allocator);
    defer x.deinit();
    var xp = try DevicePtr.init(allocator, x);
    defer xp.deinit();
    try std.testing.expect(xp.get() == null);
}

test "DevicePtr -> NoCopy" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    var a = try zt.tensor.full(allocator, &.{ 3, 3 }, f64, 5, .f32);
    defer a.deinit();
    var device_ptr_loc: ?*anyopaque = null;
    var p = try DevicePtr.init(allocator, a);
    defer p.deinit();
    device_ptr_loc = p.get();
    try std.testing.expect(device_ptr_loc == try a.device(allocator, ?*anyopaque));
    try a.unlock(allocator);
}

test "DevicePtr -> Locking" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    var x = try Tensor.initHandle(allocator, &.{ 3, 3 }, .f32);
    defer x.deinit();
    try std.testing.expect(try x.isLocked(allocator) == false);
    {
        var xp = try DevicePtr.init(allocator, x);
        defer xp.deinit();
        try std.testing.expect(try x.isLocked(allocator));
    }
    try std.testing.expect(try x.isLocked(allocator) == false);
}

test "DevicePtr -> Move" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    var x = try Tensor.initHandle(allocator, &.{ 3, 3 }, .f32);
    defer x.deinit();
    var y = try Tensor.initHandle(allocator, &.{ 4, 4 }, .f32);
    defer y.deinit();

    var yp = try DevicePtr.init(allocator, y);
    defer yp.deinit();
    try std.testing.expect(try x.isLocked(allocator) == false);
    try std.testing.expect(try y.isLocked(allocator));

    var tmp = try DevicePtr.init(allocator, x);
    try yp.assign(allocator, &tmp);
    try std.testing.expect(try x.isLocked(allocator));
    try std.testing.expect(try y.isLocked(allocator) == false);

    tmp = DevicePtr{};
    try yp.assign(allocator, &tmp);
    try std.testing.expect(try x.isLocked(allocator) == false);
    try std.testing.expect(try y.isLocked(allocator) == false);
}
