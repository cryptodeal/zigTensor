const std = @import("std");
const default_tensor_type = @import("DefaultTensorType.zig");

const defaultTensorBackend = default_tensor_type.defaultTensorBackend;

pub const init = std.once(initFn);

fn initFn() void {
    defaultTensorBackend();
    // TODO: initLogging();
}
