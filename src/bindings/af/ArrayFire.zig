const std = @import("std");
const af = @cImport({
    @cInclude("arrayfire.h");
    @cInclude("af/internal.h");
});

pub usingnamespace af;

pub const ops = struct {
    pub usingnamespace @import("array_ops.zig");
    pub usingnamespace @import("event_ops.zig");
    pub usingnamespace @import("features_ops.zig");
    pub usingnamespace @import("image_ops.zig");
    pub usingnamespace @import("random_ops.zig");
    pub usingnamespace @import("memory_manager_ops.zig");
    pub usingnamespace @import("utils.zig");
    pub usingnamespace @import("window_ops.zig");
};
pub usingnamespace @import("Array.zig");
pub usingnamespace @import("Cell.zig");
pub usingnamespace @import("Event.zig");
pub usingnamespace @import("Features.zig");
pub usingnamespace @import("MemoryManager.zig");
pub usingnamespace @import("RandomEngine.zig");
pub usingnamespace @import("types.zig");
pub usingnamespace @import("Window.zig");
pub usingnamespace @import("gFor.zig");

pub const Errors = error{ArrayFireError};

/// Utility function for handling `af.af_err` when calling
/// directly into ArrayFire's C API from Zig.
pub inline fn AF_CHECK(v: af.af_err, src: std.builtin.SourceLocation) !void {
    if (v != af.AF_SUCCESS) {
        std.debug.print(
            "ArrayFire error: {s}:{d} - {s}:\n{s}\n",
            .{ src.file, src.line, ops.errToString(v), ops.getLastError() },
        );
        return error.ArrayFireError;
    }
}

test {
    std.testing.refAllDecls(ops);
}
