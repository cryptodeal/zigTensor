const std = @import("std");
const af = @import("../../../bindings/af/ArrayFire.zig");

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

pub const batchFunc_t = fn (lhs: af.af_array, rhs: af.af_array) af.af_array;
pub fn batchFunc(lhs: af.af_array, rhs: af.af_array, func: *const batchFunc_t) !af.af_array {
    if (gforGet()) {
        std.log.err("batchFunc can not be used inside GFOR\n", .{});
        return error.BatchFuncFailed;
    }
    gforSet(true);
    var res: af.af_array = func(lhs, rhs);
    gforSet(false);
    return res;
}

test "gforToggle" {
    for (0..10) |_| {
        var status = gforToggle();
        std.debug.print("status: {any}\n", .{status});
    }
}
