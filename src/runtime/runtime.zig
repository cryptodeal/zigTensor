const std = @import("std");
const device_type = @import("DeviceType.zig");
const device_manager = @import("DeviceManager.zig");

// device manager exports
pub const kX64DeviceId = device_manager.kX64DeviceId;
pub const DeviceManager = device_manager.DeviceManager;

/// device type exports
pub const DeviceType = device_type.DeviceType;
pub const kDefaultDeviceType = device_type.kDefaultDeviceType;
pub const deviceTypeToString = device_type.deviceTypeToString;
pub const getDeviceTypes = device_type.getDeviceTypes;

test {
    std.testing.refAllDecls(@This());
}
