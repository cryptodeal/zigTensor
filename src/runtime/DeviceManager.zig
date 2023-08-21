const std = @import("std");
const rc = @import("zigrc");
const device = @import("Device.zig");
const deviceType = @import("DeviceType.zig");
const cuda = @import("CUDAUtils.zig");
const ZT_BACKEND_CUDA = @import("build_options").ZT_BACKEND_CUDA;

const Device = device.Device;
const DeviceType = deviceType.DeviceType;

/// Device id for the single CPU device.
pub const kX64DeviceId: usize = 0;

fn getActiveDeviceId(device_type: DeviceType) !usize {
    return switch (device_type) {
        .x64 => kX64DeviceId,
        .CUDA => {
            // TODO: support CUDA backend
            // if (ZT_BACKEND_CUDA) {
            // return cuda.getActiveDeviceId();
            // } else {
            std.log.err("CUDA is not supported\n", .{});
            return error.CUDABackendUnsupported;
            //}
        },
    };
}

pub const DeviceManager = struct {
    pub const DeviceTypeInfo = std.AutoHashMap(usize, rc.Arc(*Device));

    deviceTypeToInfo_: std.AutoHashMap(DeviceType, DeviceTypeInfo),

    pub fn init(allocator: std.mem.Allocator) DeviceManager {
        var x64Info = DeviceTypeInfo.init(allocator);
        _ = x64Info;
        //try x64Info.put()
        return .{};
    }
};
