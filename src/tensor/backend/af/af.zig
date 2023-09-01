const std = @import("std");
const af_backend = @import("ArrayFireBackend.zig");

pub const ArrayFireBackend = af_backend.ArrayFireBackend;

test {
    std.testing.refAllDecls(@This());
}
