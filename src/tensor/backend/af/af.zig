const std = @import("std");
const utils = @import("Utils.zig");

pub const AF_CHECK = utils.AF_CHECK;
pub const ztToAfType = utils.ztToAfType;

test {
    std.testing.refAllDecls(@This());
}
