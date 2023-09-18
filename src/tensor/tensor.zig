const std = @import("std");
pub usingnamespace @import("DefaultTensorType.zig");
pub usingnamespace @import("Index.zig");
pub usingnamespace @import("Init.zig");
pub usingnamespace @import("Random.zig");
pub usingnamespace @import("Shape.zig");
pub usingnamespace @import("TensorAdapter.zig");
pub usingnamespace @import("TensorBackend.zig");
pub usingnamespace @import("TensorBase.zig");

pub const backend = @import("backend/backend.zig");

test {
    std.testing.refAllDecls(@This());
}
