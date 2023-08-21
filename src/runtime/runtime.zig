const std = @import("std");
const device_type = @import("DeviceType.zig");

pub const DeviceType = device_type.DeviceType;
pub const kDefaultDeviceType = device_type.kDefaultDeviceType;
pub const deviceTypeToString = device_type.deviceTypeToString;
pub const getDeviceTypes = device_type.getDeviceTypes;

test {
    std.testing.refAllDecls(@This());
}
