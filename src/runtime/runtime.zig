const std = @import("std");
const device_type = @import("DeviceType.zig");
const device_manager = @import("DeviceManager.zig");
const device = @import("Device.zig");
const stream = @import("Stream.zig");
const sync_stream = @import("SynchronousStream.zig");

// device manager exports
pub const kX64DeviceId = device_manager.kX64DeviceId;
pub const DeviceManager = device_manager.DeviceManager;

// device manager exports
pub const Device = device.Device;
pub const X64Device = device.X64Device;
pub const DeviceErrors = device.DeviceErrors;

/// device type exports
pub const DeviceType = device_type.DeviceType;
pub const kDefaultDeviceType = device_type.kDefaultDeviceType;
pub const deviceTypeToString = device_type.deviceTypeToString;
pub const getDeviceTypes = device_type.getDeviceTypes;

// stream exports
pub const Stream = stream.Stream;
pub const StreamErrors = stream.StreamErrors;
pub const SynchronousStream = sync_stream.SynchronousStream;

test {
    std.testing.refAllDecls(@This());
}
