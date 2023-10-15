const std = @import("std");

pub usingnamespace @import("arrayfire_backend.zig");
pub usingnamespace @import("arrayfire_tensor.zig");

test {
    std.testing.refAllDecls(@This());
}
