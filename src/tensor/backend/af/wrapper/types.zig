const std = @import("std");
const af = @import("../../../../backends/ArrayFire.zig");

pub const BinaryOp = enum(af.af_binary_op) {
    BinaryAdd = af.AF_BINARY_ADD,
    BinaryMul = af.AF_BINARY_MUL,
    BinaryMin = af.AF_BINARY_MIN,
    BinaryMax = af.AF_BINARY_MAX,

    pub fn value(self: *BinaryOp) af.af_binary_op {
        return @intFromEnum(self);
    }
};
