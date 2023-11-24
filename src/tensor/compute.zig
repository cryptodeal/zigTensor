const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;
const Stream = zt.runtime.Stream;
const DeviceManager = zt.runtime.DeviceManager;
const Device = zt.runtime.Device;
const DeviceType = zt.runtime.DeviceType;
const kDefaultDeviceType = zt.runtime.kDefaultDeviceType;

fn tensorsToUniqueStreams(allocator: std.mem.Allocator, tensors: []const Tensor) !std.AutoHashMap(Stream, void) {
    var unique_streams = std.AutoHashMap(Stream, void).init(allocator);
    for (tensors) |tensor| {
        try unique_streams.put(try tensor.stream(allocator), {});
    }
    return unique_streams;
}

pub fn sync(allocator: std.mem.Allocator) !void {
    var device_manager = try DeviceManager.getInstance(allocator);
    return (try device_manager.getActiveDevice(kDefaultDeviceType)).sync();
}

pub fn syncDevice(allocator: std.mem.Allocator, device_id: i32) !void {
    var device_manager = try DeviceManager.getInstance(allocator);
    return (try device_manager.getDevice(kDefaultDeviceType, device_id)).sync();
}

pub fn syncDeviceTypes(allocator: std.mem.Allocator, types: std.EnumSet(DeviceType)) !void {
    var device_manager = try DeviceManager.getInstance(allocator);
    // TODO: consider launching these `Device::sync` calls non-blockingly
    var iterator = types.iterator();
    while (iterator.next()) |t| {
        try (try device_manager.getActiveDevice(t)).sync();
    }
}

pub fn syncDevices(devices: std.AutoHashMap(Device, void)) !void {
    // TODO: consider launching these `Device::sync` calls non-blockingly
    var iterator = devices.keyIterator();
    while (iterator.next()) |device| {
        try device.sync();
    }
}

pub fn relativeSync(allocator: std.mem.Allocator, wait: Stream, wait_ons: []const Tensor) !void {
    // ensure computations are launched
    for (wait_ons) |tensor| {
        try (try tensor.backend(allocator)).eval(allocator, tensor);
    }
    var unique_streams = try tensorsToUniqueStreams(allocator, wait_ons);
    defer unique_streams.deinit();
    try wait.relativeSyncMulti(unique_streams);
}

pub fn relativeSync2(allocator: std.mem.Allocator, waits: []const Tensor, wait_on: Stream) !void {
    var unique_streams = try tensorsToUniqueStreams(allocator, waits);
    defer unique_streams.deinit();
    var iterator = unique_streams.keyIterator();
    while (iterator.next()) |stream| {
        try stream.relativeSync(wait_on);
    }
}

pub fn eval(allocator: std.mem.Allocator, tensor: Tensor) !void {
    return (try tensor.backend(allocator)).eval(allocator, tensor);
}

pub fn getDevice(allocator: std.mem.Allocator) !i32 {
    var device_manager = try DeviceManager.getInstance(allocator);
    return (try device_manager.getActiveDevice(kDefaultDeviceType)).nativeId();
}

pub fn setDevice(allocator: std.mem.Allocator, device_id: i32) !void {
    var device_manager = try DeviceManager.getInstance(allocator);
    return (try device_manager.getDevice(kDefaultDeviceType, device_id)).setActive();
}

pub fn getDeviceCount(allocator: std.mem.Allocator) !usize {
    var device_manager = try DeviceManager.getInstance(allocator);
    return device_manager.getDeviceCount(kDefaultDeviceType);
}

// TODO: pub fn getMemMgrInfo() !void {}

// TODO: pub fn setMemMgrLogStream() !void {}

// TODO: pub fn setMemMgrLoggingEnabled() !void {}

// TODO: pub fn setMemMgrFlushInterval() !void {}

test "TensorComputeTest -> sync" {
    const full = @import("tensor.zig").full;
    const add = @import("tensor.zig").add;
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    // Testing whether a value is ready isn't meaningful since any function to
    // inspect its state will implicitly synchronize -- this test simply ensures
    // sync runs
    var t1 = try full(allocator, &.{ 10, 10 }, i8, 1, .f32);
    defer t1.deinit();
    var t2 = try full(allocator, &.{ 10, 10 }, i8, 2, .f32);
    defer t2.deinit();
    var t3 = try add(allocator, Tensor, t1, Tensor, t2);
    defer t3.deinit();
    try sync(allocator);

    const device_id = try getDevice(allocator);
    var t4 = try add(allocator, Tensor, t1, Tensor, t2);
    defer t4.deinit();
    try t4.inPlaceAdd(allocator, Tensor, t3);
    try syncDevice(allocator, device_id);
}

test "TensorComputeTest -> eval" {
    const full = @import("tensor.zig").full;
    const mul = @import("tensor.zig").mul;
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    // Testing whether a value is ready isn't meaningful since any function to
    // inspect its state will implicitly synchronize -- this test simply ensures
    // eval runs
    var t1 = try full(allocator, &.{ 10, 10 }, i8, 3, .f32);
    defer t1.deinit();
    var t2 = try full(allocator, &.{ 10, 10 }, i8, 4, .f32);
    defer t2.deinit();
    var t3 = try mul(allocator, Tensor, t1, Tensor, t2);
    defer t3.deinit();
    try eval(allocator, t3);
}
