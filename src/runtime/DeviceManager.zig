const std = @import("std");
const rc = @import("zigrc");
const rt_device = @import("Device.zig");
const rt_device_type = @import("DeviceType.zig");
// const cuda = @import("CUDAUtils.zig");
const ZT_BACKEND_CUDA = @import("build_options").ZT_BACKEND_CUDA;

const Device = rt_device.Device;
const Arc = rc.Arc;
const X64Device = rt_device.X64Device;
const DeviceType = rt_device_type.DeviceType;
const getDeviceTypes = rt_device_type.getDeviceTypes;

/// Device id for the single CPU device.
pub const kX64DeviceId: i32 = 0;

fn getActiveDeviceId(device_type: DeviceType) !i32 {
    return switch (device_type) {
        .x64 => kX64DeviceId,
        .CUDA => {
            // TODO: support CUDA backend
            // if (ZT_BACKEND_CUDA) {
            // return cuda.getActiveDeviceId();
            // }
            std.log.err("CUDA is not supported\n", .{});
            return error.CUDABackendUnsupported;
        },
    };
}

var deviceManagerSingleton: ?*DeviceManager = null;

pub fn deinitDeviceManager() void {
    if (deviceManagerSingleton != null) {
        deviceManagerSingleton.?.deinit();
    }
}

pub const DeviceManager = struct {
    pub const DeviceTypeInfo = std.AutoHashMap(i32, Device);

    deviceTypeToInfo_: std.EnumMap(DeviceType, DeviceTypeInfo),
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) !*DeviceManager {
        var x64Info = DeviceTypeInfo.init(allocator);
        var x64Device = Device.init(try X64Device.init(allocator));
        try x64Info.put(kX64DeviceId, x64Device);
        var deviceTypeToInfo_ = std.EnumMap(DeviceType, DeviceTypeInfo){};
        deviceTypeToInfo_.put(.x64, x64Info);
        if (ZT_BACKEND_CUDA) {
            // TODO: add cuda device to deviceTypeToInfo_
        }

        var self = try allocator.create(DeviceManager);
        self.* = .{
            .allocator = allocator,
            .deviceTypeToInfo_ = deviceTypeToInfo_,
        };
        return self;
    }

    pub fn deinit(self: *DeviceManager) void {
        var type_info_iterator = self.deviceTypeToInfo_.iterator();
        while (type_info_iterator.next()) |type_info| {
            var device_iterator = type_info.value.valueIterator();
            while (device_iterator.next()) |dev| {
                dev.deinit();
            }
            type_info.value.deinit();
        }
        self.allocator.destroy(self);
        deviceManagerSingleton = null;
    }

    pub fn enforceDeviceTypeAvailable(self: *DeviceManager, error_prefix: []const u8, device_type: DeviceType) !void {
        if (!self.isDeviceTypeAvailable(device_type)) {
            std.log.err("{s} device type `{s}` unavailable\n", .{ error_prefix, @tagName(device_type) });
            return error.DeviceTypeUnavailable;
        }
    }

    pub fn getInstance(allocator: std.mem.Allocator) !*DeviceManager {
        if (deviceManagerSingleton == null) {
            deviceManagerSingleton = try DeviceManager.init(allocator);
        }
        return deviceManagerSingleton.?;
    }

    pub fn isDeviceTypeAvailable(self: *DeviceManager, device_type: DeviceType) bool {
        return self.deviceTypeToInfo_.contains(device_type);
    }

    pub fn getDeviceCount(self: *DeviceManager, device_type: DeviceType) !u32 {
        try self.enforceDeviceTypeAvailable("[DeviceManager.getDeviceCount]", device_type);
        return self.deviceTypeToInfo_.get(device_type).?.count();
    }

    pub fn getDevicesOfType(self: *DeviceManager, allocator: std.mem.Allocator, device_type: DeviceType) ![]*Device {
        try self.enforceDeviceTypeAvailable("[DeviceManager.getDevicesOfType]", device_type);
        var device_list = std.ArrayList(*Device).init(allocator);
        var devices = self.deviceTypeToInfo_.get(device_type).?;
        var device_iterator = devices.valueIterator();
        while (device_iterator.next()) |d| {
            try device_list.append(d);
        }
        return device_list.toOwnedSlice();
    }

    pub fn getDevice(self: *DeviceManager, device_type: DeviceType, id: i32) !Device {
        try self.enforceDeviceTypeAvailable("[DeviceManager.getDevice]", device_type);
        var idToDevice = self.deviceTypeToInfo_.get(device_type).?;
        if (!idToDevice.contains(id)) {
            std.log.err("[DeviceManager::getDevice] unknown device id: [{d}]\n", .{id});
            return error.DeviceNotFound;
        }
        return idToDevice.get(id).?;
    }

    pub fn getActiveDevice(self: *DeviceManager, device_type: DeviceType) !Device {
        try self.enforceDeviceTypeAvailable("[DeviceManager.getActiveDevice]", device_type);
        const active_device_id = try getActiveDeviceId(device_type);
        return self.deviceTypeToInfo_.get(device_type).?.get(active_device_id).?;
    }
};

test "DeviceManager getInstance" {
    var allocator = std.testing.allocator;
    var mgr1 = try DeviceManager.getInstance(allocator);
    defer mgr1.deinit();
    try std.testing.expectEqual(mgr1, try DeviceManager.getInstance(allocator));
}

test "DeviceManager isDeviceTypeAvailable" {
    const allocator = std.testing.allocator;
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    // x64 (CPU) should be always available
    try std.testing.expect(mgr.isDeviceTypeAvailable(.x64));
    // CUDA availability depends on compilation
    try std.testing.expect(mgr.isDeviceTypeAvailable(.CUDA) == ZT_BACKEND_CUDA);
}

test "DeviceManager getDeviceCount" {
    const allocator = std.testing.allocator;
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    // For now we always treat CPU as a single device
    try std.testing.expect(try mgr.getDeviceCount(.x64) == 1);

    if (mgr.isDeviceTypeAvailable(.CUDA)) {
        try std.testing.expect(try mgr.getDeviceCount(.CUDA) != 0);
    } else {
        try std.testing.expectError(error.DeviceTypeUnavailable, mgr.getDeviceCount(.CUDA));
    }
}

test "DeviceManager getDevicesOfType" {
    const allocator = std.testing.allocator;
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    // For now we always treat CPU as a single device
    var devices = try mgr.getDevicesOfType(allocator, .x64);
    defer allocator.free(devices);

    try std.testing.expect(devices.len == 1);
    var device_type_set = getDeviceTypes();
    var type_iterator = device_type_set.iterator();
    while (type_iterator.next()) |t| {
        if (mgr.isDeviceTypeAvailable(.CUDA)) {
            var dev_list = try mgr.getDevicesOfType(allocator, t);
            for (dev_list) |dev| {
                try std.testing.expect(dev.deviceType() == t);
            }
        } else {
            try std.testing.expectError(error.DeviceTypeUnavailable, mgr.getDeviceCount(.CUDA));
        }
    }
}

test "DeviceManager getDevice" {
    const allocator = std.testing.allocator;
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    var x64Device = try mgr.getDevice(.x64, kX64DeviceId);
    try std.testing.expect(x64Device.deviceType() == .x64);
}

test "DeviceManager getActiveDevice" {
    const allocator = std.testing.allocator;
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    var device_type_set = getDeviceTypes();
    var type_iterator = device_type_set.iterator();
    while (type_iterator.next()) |t| {
        if (mgr.isDeviceTypeAvailable(t)) {
            var dev = try mgr.getActiveDevice(t);
            try std.testing.expect(dev.deviceType() == t);
        } else {
            try std.testing.expectError(error.DeviceTypeUnavailable, mgr.getActiveDevice(.CUDA));
        }
    }
}
