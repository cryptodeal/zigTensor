const af = @import("ArrayFire.zig");

pub const Cell = struct {
    row: i32,
    col: i32,
    title: []const u8,
    cmap: af.ColorMap,

    pub fn value(self: *const Cell) af.af_cell {
        return af.af_cell{
            .row = @intCast(self.row),
            .col = @intCast(self.col),
            .title = self.title.ptr,
            .cmap = self.cmap.value(),
        };
    }
};
