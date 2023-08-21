const std = @import("std");
const zt_shape = @import("Shape.zig");

pub const backend = @import("backend/backend.zig");
pub const Shape = zt_shape.Shape;
pub const ShapeErrors = zt_shape.ShapeErrors;
pub const Dim = zt_shape.Dim;

test {
    std.testing.refAllDecls(@This());
}
