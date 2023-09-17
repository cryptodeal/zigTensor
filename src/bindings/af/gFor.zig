const std = @import("std");
const af = @import("ArrayFire.zig");

threadlocal var gforStatus: bool = undefined;

pub fn gforGet() bool {
    return gforStatus;
}

pub fn gforSet(val: bool) void {
    gforStatus = val;
}

pub fn gforToggle() bool {
    var status: u1 = @intFromBool(gforGet());
    status ^= 1;
    const res = status != 0;
    gforSet(res);
    return res;
}

pub const batchFunc_t = *const fn (allocator: std.mem.Allocator, lhs: *const af.Array, rhs: *const af.Array, batch: bool) anyerror!*af.Array;

pub fn batchFunc(allocator: std.mem.Allocator, lhs: *const af.Array, rhs: *const af.Array, func: batchFunc_t) !*af.Array {
    if (gforGet()) {
        std.log.err("batchFunc can not be used inside GFOR\n", .{});
        return error.BatchFuncFailed;
    }
    gforSet(true);
    var res: *af.Array = try func(allocator, lhs, rhs, gforGet());
    errdefer gforSet(false); // if errors, still need to set false
    gforSet(false); // if no error, set false
    return res;
}

test "gforToggle" {
    var lastStatus = false;
    for (0..10) |_| {
        var status = gforToggle();
        try std.testing.expect(status != lastStatus);
        lastStatus = status;
    }
}
