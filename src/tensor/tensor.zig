const std = @import("std");
const zt_shape = @import("Shape.zig");
const zt_index = @import("index.zig");

pub const backend = @import("backend/backend.zig");
pub const Shape = zt_shape.Shape;
pub const ShapeErrors = zt_shape.ShapeErrors;
pub const Dim = zt_shape.Dim;

pub const Index = zt_index.Index;
pub const Range = zt_index.Range;

test {
    std.testing.refAllDecls(@This());
}
