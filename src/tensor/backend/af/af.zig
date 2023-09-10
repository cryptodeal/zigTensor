const std = @import("std");
pub usingnamespace @import("ArrayFireBackend.zig");
pub usingnamespace @import("ArrayFireTensor.zig");

test {
    std.testing.refAllDecls(@This());
}
