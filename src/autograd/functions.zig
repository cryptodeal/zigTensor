const zt = @import("../zt.zig");
const std = @import("std");

const Variable = @import("autograd.zig").Variable;
const Tensor = zt.tensor.Tensor;
const Shape = zt.tensor.shape.Shape;
const Dim = zt.tensor.shape.Dim;

pub const detail = struct {
    pub fn tileAs(allocator: std.mem.Allocator, input: Tensor, rdims: Shape) !Tensor {
        // Scalar tensor
        if (try input.ndim(allocator) == 0) {
            return zt.tensor.tile(allocator, input, rdims);
        }
        var dims = try allocator.alloc(Dim, zt.tensor.shape.ndim(rdims));
        defer allocator.free(dims);
        @memset(dims, 1);
        var idims = try input.shape(allocator);
        for (0..zt.tensor.shape.ndim(rdims)) |i| {
            var idims_size: Dim = if (i + 1 > zt.tensor.shape.ndim(idims)) 1 else idims[i];
            if (@mod(rdims[i], idims_size) != 0) {
                std.debug.print("Invalid dims for tileAs for input dims {any} to output dims {any}\n", .{ idims, rdims });
                return error.TileAsInvalidDims;
            }
            dims[i] = @divTrunc(rdims[i], idims_size);
        }
        return zt.tensor.tile(allocator, input, dims);
    }

    pub fn sumAs(allocator: std.mem.Allocator, input: Tensor, rdims: Shape) !Tensor {
        var idims = try input.shape(allocator);
        var result = try Tensor.initAssign(allocator, input);
        defer result.deinit();
        for (0..try input.ndim(allocator)) |i| {
            if (i + 1 > zt.tensor.shape.ndim(rdims) or idims[i] != rdims[i]) {
                var tmp_sum = try zt.tensor.sum(allocator, result, &.{@as(Dim, @intCast(i))}, true);
                defer tmp_sum.deinit();
                try result.assign(allocator, Tensor, tmp_sum);
            }
        }
        var out = try result.astype(allocator, try input.dtype(allocator));
        defer out.deinit();
        return zt.tensor.reshape(allocator, out, rdims);
    }

    // TODO: pub fn expandedShapeFromReducedDims()

    // TODO: pub fn expandFromReduction()

    pub fn areVariableTypesEqual(allocator: std.mem.Allocator, a: *Variable, b: *Variable) !bool {
        return (try a.dtype(allocator) == try b.dtype(allocator));
    }
};

inline fn ztVariableDTypesMatch(allocator: std.mem.Allocator, fn_name: []const u8, vars: []const Variable) !void {
    if (vars.len <= 1) return;
    var var1 = vars[0];
    for (vars[1..]) |v| {
        if (!detail.areVariableTypesEqual(allocator, &var1, &v)) {
            std.log.debug("{s}  doesn't support binary operations with Variables of different types\n", .{fn_name});
            return error.TensorBackendMismatch;
        }
    }
}

fn bothAddGradFunc(allocator: std.mem.Allocator, inputs: []Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    try inputs[0].addGrad(try Variable.init(allocator, grad_output.tensor(), false));
    try inputs[1].addGrad(try Variable.init(allocator, grad_output.tensor(), false));
}

fn lhsAddGradFunc(allocator: std.mem.Allocator, inputs: []Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    try inputs[0].addGrad(try Variable.init(allocator, grad_output.tensor(), false));
}

pub fn add(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !Variable {
    if ((LhsT != Variable and LhsT != f64) or (RhsT != Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.add only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == Variable and RhsT == Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.add(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.initWithInputs(allocator, result, &.{ try lhs.withoutData(allocator), try rhs.withoutData(allocator) }, bothAddGradFunc, null, null);
    }
    if (LhsT == Variable and RhsT == f64) {
        var result = try zt.tensor.add(allocator, Tensor, lhs.tensor(), f64, rhs);
        defer result.deinit();
        return Variable.initWithInputs(allocator, try result.astype(allocator, try lhs.dtype(allocator)), &.{try lhs.withoutData(allocator)}, lhsAddGradFunc, null, null);
    }
    if (LhsT == f64 and RhsT == Variable) {
        return add(allocator, RhsT, rhs, LhsT, lhs);
    }
}

// TODO: pub fn sub(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !Variable {}

fn negateGradFunc(allocator: std.mem.Allocator, inputs: []Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    _ = allocator;
    _ = inputs;
    _ = grad_output;
}

pub fn negate(allocator: std.mem.Allocator, input: *const Variable) !Variable {
    var result = try zt.tensor.sub(allocator, f64, 0, Tensor, input.tensor());
    _ = result;
}

fn reciprocalGradFunc(allocator: std.mem.Allocator, inputs: []Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    _ = allocator;
    _ = inputs;
    _ = grad_output;
}

pub fn reciprocal(allocator: std.mem.Allocator, input: *const Variable) !Variable {
    _ = allocator;
    _ = input;
}
