const std = @import("std");
const utils = @import("Utils.zig");
const af_backend = @import("ArrayFireBackend.zig");

pub const ArrayFireBackend = af_backend.ArrayFireBackend;
pub const AF_CHECK = utils.AF_CHECK;
pub const ztToAfType = utils.ztToAfType;

test {
    std.testing.refAllDecls(@This());
}
