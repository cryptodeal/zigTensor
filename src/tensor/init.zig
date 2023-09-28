const std = @import("std");
const build_options = @import("build_options");

const defaultTensorBackend = @import("tensor.zig").defaultTensorBackend;

const ZT_USE_ARRAYFIRE = build_options.ZT_USE_ARRAYFIRE;

// TODO: update import paths imports
const deinitBackend = if (ZT_USE_ARRAYFIRE) @import("backend/af/ArrayFireBackend.zig").deinitBackend else @compileError("Must specify backend as build argument.");
const deinitDeviceManager = @import("../runtime/DeviceManager.zig").deinitDeviceManager;

pub const init = std.once(initFn);
pub fn deinit() void {
    deinitBackend();
    deinitDeviceManager();
}

fn initFn() void {
    defaultTensorBackend() catch unreachable;
    // TODO: initLogging();
}
