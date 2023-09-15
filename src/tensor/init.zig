const std = @import("std");
const build_options = @import("build_options");
const default_tensor_type = @import("DefaultTensorType.zig");

const ZT_USE_ARRAYFIRE = build_options.ZT_USE_ARRAYFIRE;
const defaultTensorBackend = default_tensor_type.defaultTensorBackend;

const deinitBackend = if (ZT_USE_ARRAYFIRE) @import("backend/af/ArrayFireBackend.zig").deinitBackend else @compileError("Must specify backend as build argument.");
const deinitDeviceManager = @import("../runtime/DeviceManager.zig").deinitDeviceManager;
pub const init = std.once(initFn);

fn initFn() void {
    defaultTensorBackend() catch unreachable;
    // TODO: initLogging();
}

pub fn deinit() void {
    deinitBackend();
    deinitDeviceManager();
}
