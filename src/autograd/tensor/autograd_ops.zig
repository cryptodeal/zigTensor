const std = @import("std");
const zt = @import("../../zt.zig");

const Tensor = zt.tensor.Tensor;

pub const RnnGradData = struct {
    dy: Tensor,
    dhy: Tensor,
    dcy: Tensor,
};
