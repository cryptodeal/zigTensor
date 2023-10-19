const zt = @import("../zt.zig");
const std = @import("std");

const Variable = @import("autograd.zig").Variable;
const Tensor = zt.tensor.Tensor;
const Shape = zt.tensor.shape.Shape;
const Dim = zt.tensor.shape.Dim;

pub const detail = struct {
    pub fn adjustInputType(allocator: std.mem.Allocator, comptime T: type, in: T, func_name: []const u8) !struct { res: T, allocated: bool } {
        const optim_level: zt.OptimLevel = zt.OptimMode.get().getOptimLevel();
        // Fastpath - DEFAULT mode never casts tensors
        if (optim_level == .DEFAULT) {
            return .{ .res = in, .allocated = false };
        }

        if (!zt.kOptimLevelTypeExclusionMappings(optim_level, func_name)) {
            // Not in the excluded list - cast to f16
            return .{ .res = in.astype(allocator, .f16), .allocated = true };
        } else {
            // Upcast to f32 only if we have an f16 input - otherwise, leave as is
            if (try in.dtype(allocator) == .f16) {
                return .{ .res = in.astype(allocator, .f32), .allocated = true };
            } else {
                return .{ .res = in, .allocated = false };
            }
        }
    }

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

    pub fn areVariableTypesEqual(allocator: std.mem.Allocator, a: *const Variable, b: *const Variable) !bool {
        return (try a.dtype(allocator) == try b.dtype(allocator));
    }
};

inline fn ztVariableDTypesMatch(allocator: std.mem.Allocator, fn_name: []const u8, vars: []const *Variable) !void {
    if (vars.len <= 1) return;
    var var1 = vars[0];
    for (vars[1..]) |v| {
        if (!try detail.areVariableTypesEqual(allocator, var1, v)) {
            std.log.debug("{s}  doesn't support binary operations with Variables of different types\n", .{fn_name});
            return error.VariableDTypeMismatch;
        }
    }
}

fn bothAddGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    try inputs[0].addGrad(allocator, try Variable.init(allocator, grad_output.tensor(), false));
    try inputs[1].addGrad(allocator, try Variable.init(allocator, grad_output.tensor(), false));
}

fn lhsAddGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    try inputs[0].addGrad(allocator, try Variable.init(allocator, grad_output.tensor(), false));
}

pub fn add(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.add only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.add(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.initWithInputs(allocator, result, &.{ try lhs.withoutData(allocator), try rhs.withoutData(allocator) }, bothAddGradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.add(allocator, Tensor, lhs.tensor(), f64, rhs);
        defer result.deinit();
        return Variable.initWithInputs(allocator, try result.astype(allocator, try lhs.dtype(allocator)), &.{try lhs.withoutData(allocator)}, lhsAddGradFunc, null, null);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return add(allocator, RhsT, rhs, LhsT, lhs);
    }
}

fn bothSubGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    try inputs[0].addGrad(allocator, try Variable.init(allocator, grad_output.tensor(), false));
    var neg = try negate(allocator, grad_output);
    defer neg.deinit();
    try inputs[1].addGrad(allocator, try Variable.init(allocator, neg.tensor(), false));
}

fn lhsSubGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    try inputs[0].addGrad(allocator, try Variable.init(allocator, grad_output.tensor(), false));
}

fn rhsSubGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var neg = try negate(allocator, &grad_output);
    defer neg.deinit();
    try inputs[0].addGrad(allocator, try Variable.init(allocator, neg.tensor(), false));
}

pub fn sub(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.sub only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.sub(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.initWithInputs(allocator, result, &.{ try lhs.withoutData(allocator), try rhs.withoutData(allocator) }, bothSubGradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.sub(allocator, Tensor, lhs.tensor(), f64, rhs);
        return Variable.initWithInputs(allocator, result, &.{try lhs.withoutData(allocator)}, lhsSubGradFunc, null, null);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        var result = try zt.tensor.sub(allocator, f64, lhs, Tensor, rhs.tensor());
        return Variable.initWithInputs(allocator, result, &.{try rhs.withoutData(allocator)}, rhsSubGradFunc, null, null);
    }
}

fn bothMulGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    if (inputs[0].isCalcGrad()) {
        try inputs[0].addGrad(allocator, try Variable.init(allocator, try zt.tensor.mul(allocator, Tensor, grad_output.tensor(), Tensor, inputs[1].tensor()), false));
    }
    if (inputs[1].isCalcGrad()) {
        try inputs[1].addGrad(allocator, try Variable.init(allocator, try zt.tensor.mul(allocator, Tensor, grad_output.tensor(), Tensor, inputs[0].tensor()), false));
    }
}

const F64GradFuncCtx = struct { val: f64 };

fn freeF64GradFuncCtx(allocator: std.mem.Allocator, ctx: ?*anyopaque) void {
    var grad_ctx: F64GradFuncCtx = @ptrCast(@alignCast(ctx));
    allocator.destroy(grad_ctx);
}

fn lhsMulGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: F64GradFuncCtx = @ptrCast(@alignCast(ctx));
    try inputs[0].addGrad(allocator, try Variable.init(allocator, try zt.tensor.mul(allocator, Tensor, grad_output.tensor(), f64, grad_ctx.val), false));
}

pub fn mul(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.mul only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.mul(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        var in1: *Variable = if (rhs.isCalcGrad()) lhs else try lhs.withoutData(allocator);
        var in2: *Variable = if (lhs.isCalcGrad()) rhs else try rhs.withoutData(allocator);
        return Variable.initWithInputs(allocator, result, &.{ in1, in2 }, bothMulGradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.mul(allocator, Tensor, lhs.tensor(), f64, rhs);
        var ctx = try allocator.create(F64GradFuncCtx);
        ctx.* = .{ .val = rhs };
        return Variable.initWithInputs(allocator, result, &.{try lhs.withoutData(allocator)}, lhsMulGradFunc, ctx, freeF64GradFuncCtx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return mul(allocator, RhsT, rhs, LhsT, lhs);
    }
}

fn bothDivGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var inputs1_rec = try negate(allocator, inputs[1]);
    defer inputs1_rec.deinitAll();
    var grad_input0 = try mul(allocator, *Variable, grad_output, *Variable, inputs1_rec);
    defer grad_input0.deinit();
    if (inputs[0].isCalcGrad()) {
        try inputs[0].addGrad(allocator, try Variable.init(allocator, grad_input0.tensor(), false));
    }
    if (inputs[1].isCalcGrad()) {
        var tmp_neg = try negate(allocator, inputs[0]);
        defer tmp_neg.deinitAll();
        var tmp = try mul(allocator, *Variable, grad_input0, *Variable, tmp_neg);
        defer tmp.deinitAll();
        var tmp2 = try mul(allocator, *Variable, tmp, *Variable, inputs1_rec);
        defer tmp2.deinit();
        try inputs[1].addGrad(allocator, try Variable.init(allocator, tmp2.tensor(), false));
    }
}

fn lhsDivGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: F64GradFuncCtx = @ptrCast(@alignCast(ctx));
    var tmp = try div(allocator, Variable, grad_output, f64, grad_ctx.val);
    defer tmp.deinit();
    try inputs[0].addGrad(allocator, try Variable.init(allocator, tmp.tensor(), false));
}

fn rhsDivGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: F64GradFuncCtx = @ptrCast(@alignCast(ctx));
    var tmp = try mul(allocator, Variable, grad_output, f64, -grad_ctx.val);
    defer tmp.deinitAll();
    var tmp2 = try mul(allocator, Variable, inputs[0], Variable, inputs[0]);
    defer tmp2.deinitAll();
    var res = try div(allocator, Variable, tmp, Variable, tmp2);
    defer res.deinit();
    try inputs[0].addGrad(allocator, try Variable.init(allocator, res.tensor(), false));
}

pub fn div(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.div only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.div(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        var in1: *Variable = if (rhs.isCalcGrad()) lhs else try lhs.withoutData(allocator);
        return Variable.initWithInputs(allocator, result, &.{ in1, rhs }, bothDivGradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.div(allocator, Tensor, lhs.tensor(), f64, rhs);
        var ctx = try allocator.create(F64GradFuncCtx);
        ctx.* = .{ .val = rhs };
        return Variable.initWithInputs(allocator, result, &.{try lhs.withoutData(allocator)}, lhsDivGradFunc, ctx, freeF64GradFuncCtx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        var result = try zt.tensor.div(allocator, f64, lhs, Tensor, rhs.tensor());
        var ctx = try allocator.create(F64GradFuncCtx);
        ctx.* = .{ .val = lhs };
        return Variable.initWithInputs(allocator, result, &.{rhs}, rhsDivGradFunc, ctx, freeF64GradFuncCtx);
    }
}

pub fn greaterThan(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.greaterThan only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.greaterThan(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.greaterThan(allocator, Tensor, lhs.tensor(), f64, rhs);
        return Variable.init(allocator, result, false);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        var result = try zt.tensor.greaterThan(allocator, f64, lhs, Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
}

pub fn lessThan(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.lessThan only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.lessThan(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.lessThan(allocator, Tensor, lhs.tensor(), f64, rhs);
        return Variable.init(allocator, result, false);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        var result = try zt.tensor.lessThan(allocator, f64, lhs, Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
}

pub fn greaterThanEqual(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.greaterThanEqual only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.greaterThanEqual(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.greaterThanEqual(allocator, Tensor, lhs.tensor(), f64, rhs);
        return Variable.init(allocator, result, false);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        var result = try zt.tensor.greaterThanEqual(allocator, f64, lhs, Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
}

pub fn lessThanEqual(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.lessThanEqual only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.lessThanEqual(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.lessThanEqual(allocator, Tensor, lhs.tensor(), f64, rhs);
        return Variable.init(allocator, result, false);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        var result = try zt.tensor.lessThanEqual(allocator, f64, lhs, Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
}

pub fn logicalAnd(allocator: std.mem.Allocator, lhs: *Variable, rhs: *Variable) !*Variable {
    try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
    var result = try zt.tensor.logicalAnd(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
    return Variable.init(allocator, result, false);
}

pub fn logicalNot(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.logicalNot(allocator, input.tensor());
    return Variable.init(allocator, result, false);
}

fn bothMaxGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var tmp_mask = try zt.tensor.greaterThan(allocator, Tensor, inputs[0].tensor(), Tensor, inputs[1].tensor());
    defer tmp_mask.deinit();
    var mask_tensor = try tmp_mask.astype(allocator, try grad_output.dtype(allocator));
    var mask = try Variable.init(allocator, mask_tensor, false);
    defer mask.deinitAll();
    var tmp1 = try mul(allocator, *Variable, mask, *Variable, grad_output);
    defer tmp1.deinit();
    try inputs[0].addGrad(allocator, try Variable.init(allocator, tmp1.tensor(), false));
    var tmp_not = try logicalNot(allocator, mask);
    defer tmp_not.deinitAll();
    var tmp2 = try mul(allocator, *Variable, tmp_not, *Variable, grad_output);
    defer tmp2.deinit();
    try inputs[1].addGrad(allocator, try Variable.init(allocator, tmp2.tensor(), false));
}

fn lhsMaxGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: F64GradFuncCtx = @ptrCast(@alignCast(ctx));
    var tmp_mask = try zt.tensor.greaterThan(allocator, Tensor, inputs[0].tensor(), f64, grad_ctx.val);
    defer tmp_mask.deinit();
    var mask_tensor = try tmp_mask.astype(allocator, try grad_output.dtype(allocator));
    var mask = try Variable.init(allocator, mask_tensor, false);
    defer mask.deinitAll();
    var tmp1 = try mul(allocator, *Variable, mask, *Variable, grad_output);
    defer tmp1.deinit();
    try inputs[0].addGrad(allocator, try Variable.init(allocator, tmp1.tensor(), false));
}

pub fn max(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.max only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.maximum(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.initWithInputs(allocator, result, &.{ lhs, rhs }, bothMaxGradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.maximum(allocator, Tensor, lhs.tensor(), f64, rhs.tensor());
        var ctx = try allocator.create(F64GradFuncCtx);
        ctx.* = .{ .val = rhs };
        return Variable.initWithInputs(allocator, result, &.{lhs}, lhsMaxGradFunc, ctx, freeF64GradFuncCtx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return max(allocator, RhsT, rhs, LhsT, lhs);
    }
}

fn bothMinGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var tmp_mask = try zt.tensor.lessThan(allocator, Tensor, inputs[0].tensor(), Tensor, inputs[1].tensor());
    defer tmp_mask.deinit();
    var mask_tensor = try tmp_mask.astype(allocator, try grad_output.dtype(allocator));
    var mask = try Variable.init(allocator, mask_tensor, false);
    defer mask.deinitAll();
    var tmp1 = try mul(allocator, *Variable, mask, *Variable, grad_output);
    defer tmp1.deinit();
    try inputs[0].addGrad(allocator, try Variable.init(allocator, tmp1.tensor(), false));
    var tmp_not = try logicalNot(allocator, mask);
    defer tmp_not.deinitAll();
    var tmp2 = try mul(allocator, *Variable, tmp_not, *Variable, grad_output);
    defer tmp2.deinit();
    try inputs[1].addGrad(allocator, try Variable.init(allocator, tmp2.tensor(), false));
}

fn lhsMinGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: F64GradFuncCtx = @ptrCast(@alignCast(ctx));
    var tmp_mask = try zt.tensor.lessThan(allocator, Tensor, inputs[0].tensor(), f64, grad_ctx.val);
    defer tmp_mask.deinit();
    var mask_tensor = try tmp_mask.astype(allocator, try grad_output.dtype(allocator));
    var mask = try Variable.init(allocator, mask_tensor, false);
    defer mask.deinitAll();
    var tmp1 = try mul(allocator, *Variable, mask, *Variable, grad_output);
    defer tmp1.deinit();
    try inputs[0].addGrad(allocator, try Variable.init(allocator, tmp1.tensor(), false));
}

pub fn min(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.min only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.minimum(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.initWithInputs(allocator, result, &.{ lhs, rhs }, bothMinGradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.minimum(allocator, Tensor, lhs.tensor(), f64, rhs.tensor());
        var ctx = try allocator.create(F64GradFuncCtx);
        ctx.* = .{ .val = rhs };
        return Variable.initWithInputs(allocator, result, &.{lhs}, lhsMinGradFunc, ctx, freeF64GradFuncCtx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return min(allocator, RhsT, rhs, LhsT, lhs);
    }
}

fn negateGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var tmp_neg = try negate(allocator, grad_output);
    defer tmp_neg.deinit();
    try inputs[0].addGrad(allocator, try Variable.init(allocator, tmp_neg.tensor(), false));
}

pub fn negate(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.sub(allocator, f64, 0, Tensor, input.tensor());
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, negateGradFunc, null, null);
}

fn reciprocalGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var res = try reciprocal(allocator, inputs[0]);
    var tmp_neg = try negate(allocator, grad_output);
    defer tmp_neg.deinitAll();
    var tmp1 = try mul(allocator, *Variable, tmp_neg, *Variable, res);
    defer tmp1.deinitAll();
    var tmp2 = try mul(allocator, *Variable, tmp1, *Variable, res);
    defer tmp2.deinit();
    try inputs[0].addGrad(allocator, try Variable.init(allocator, tmp2.tensor(), false));
}

pub fn reciprocal(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.div(allocator, f64, 1, Tensor, adj.res);
    return Variable.initWithInputs(allocator, result, &.{input}, reciprocalGradFunc, null, null);
}

fn expGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var tmp1 = try zt.tensor.exp(allocator, inputs[0].tensor());
    defer tmp1.deinit();
    var res = try zt.tensor.mul(allocator, Tensor, grad_output.tensor(), Tensor, tmp1);
    try inputs[0].addGrad(allocator, try Variable.init(allocator, res, false));
}

pub fn exp(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.exp(allocator, adj.res);
    return Variable.initWithInputs(allocator, result, &.{input}, expGradFunc, null, null);
}

fn logGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var res = try zt.tensor.div(allocator, Tensor, grad_output.tensor(), Tensor, inputs[0].tensor());
    try inputs[0].addGrad(allocator, try Variable.init(allocator, res, false));
}

pub fn log(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.log(allocator, adj.res);
    return Variable.initWithInputs(allocator, result, &.{input}, logGradFunc, null, null);
}

fn log1pGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var tmp = try zt.tensor.add(allocator, f64, 1, Tensor, inputs[0].tensor());
    defer tmp.deinit();
    var res = try zt.tensor.div(allocator, Tensor, grad_output.tensor(), Tensor, tmp);
    try inputs[0].addGrad(allocator, try Variable.init(allocator, res, false));
}

pub fn log1p(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.log1p(allocator, adj.res);
    return Variable.initWithInputs(allocator, result, &.{input}, log1pGradFunc, null, null);
}

fn powGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: F64GradFuncCtx = @ptrCast(@alignCast(ctx));
    var grad = try zt.tensor.power(allocator, Tensor, inputs[0].tensor(), f64, grad_ctx.val - 1);
    try grad.inPlaceMul(allocator, f64, grad_ctx.val);
    try grad.inPlaceMul(allocator, Tensor, grad_output.tensor());
    try inputs[0].addGrad(allocator, try Variable.init(allocator, grad, false));
}

pub fn pow(allocator: std.mem.Allocator, input: *Variable, p: f64) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.power(allocator, Tensor, adj.res, f64, p);
    var ctx = try allocator.create(F64GradFuncCtx);
    ctx.* = .{ .val = p };
    return Variable.initWithInputs(allocator, result, &.{input}, powGradFunc, ctx, freeF64GradFuncCtx);
}

fn sinGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var res = try zt.tensor.cos(allocator, inputs[0].tensor());
    try res.inPlaceMul(allocator, Tensor, grad_output.tensor());
    try inputs[0].addGrad(allocator, try Variable.init(allocator, res, false));
}

pub fn sin(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.sin(allocator, input.tensor());
    return Variable.initWithInputs(allocator, result, &.{input}, sinGradFunc, null, null);
}

fn cosGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var tmp = try zt.tensor.sin(allocator, inputs[0].tensor());
    defer tmp.deinit();
    var res = try zt.tensor.negative(allocator, tmp);
    try res.inPlaceMul(allocator, Tensor, grad_output.tensor());
    try inputs[0].addGrad(allocator, try Variable.init(allocator, res, false));
}

pub fn cos(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.cos(allocator, input.tensor());
    return Variable.initWithInputs(allocator, result, &.{input}, cosGradFunc, null, null);
}

const TensorGradFuncCtx = struct { val: Tensor };
fn freeTensorGradFuncCtx(allocator: std.mem.Allocator, ctx: ?*anyopaque) void {
    var grad_ctx: TensorGradFuncCtx = @ptrCast(@alignCast(ctx));
    allocator.destroy(grad_ctx);
}

fn tanhGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: TensorGradFuncCtx = @ptrCast(@alignCast(ctx));
    var tmp1 = try zt.tensor.mul(allocator, Tensor, grad_ctx.val, Tensor, grad_ctx.val);
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.sub(allocator, i8, 1, Tensor, tmp1);
    try tmp2.inPlaceMul(allocator, Tensor, grad_output.tensor());
    try inputs[0].addGrad(allocator, try Variable.init(allocator, tmp2, false));
}

pub fn tanh(allocator: std.mem.Allocator, input: *const Variable) !*Variable {
    var result = try zt.tensor.tanh(allocator, input.tensor());
    var ctx = try allocator.create(TensorGradFuncCtx);
    ctx.* = .{ .val = result };
    return Variable.initWithInputs(allocator, result, &.{input}, tanhGradFunc, ctx, freeTensorGradFuncCtx);
}

fn clampGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    _ = allocator;
    _ = inputs;
    _ = grad_output;
    _ = ctx;
}

test "AutogradTest -> AutogradOperatorTypeCompatibility" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons
    if (!try zt.f16Supported(allocator)) {
        return error.SkipZigTest;
    }

    var half = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 2 }, .f16), true);
    defer half.deinitAll();
    var float = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 2 }, .f32), true);
    defer float.deinitAll();
    try std.testing.expectError(error.VariableDTypeMismatch, add(allocator, *Variable, half, *Variable, float));
    try std.testing.expectError(error.VariableDTypeMismatch, sub(allocator, *Variable, half, *Variable, float));
    try std.testing.expectError(error.VariableDTypeMismatch, mul(allocator, *Variable, half, *Variable, float));
    try std.testing.expectError(error.VariableDTypeMismatch, div(allocator, *Variable, half, *Variable, float));
    try std.testing.expectError(error.VariableDTypeMismatch, greaterThan(allocator, *Variable, half, *Variable, float));
    try std.testing.expectError(error.VariableDTypeMismatch, lessThan(allocator, *Variable, half, *Variable, float));
    try std.testing.expectError(error.VariableDTypeMismatch, greaterThanEqual(allocator, *Variable, half, *Variable, float));
    try std.testing.expectError(error.VariableDTypeMismatch, lessThanEqual(allocator, *Variable, half, *Variable, float));
    try std.testing.expectError(error.VariableDTypeMismatch, logicalAnd(allocator, half, float));
    try std.testing.expectError(error.VariableDTypeMismatch, max(allocator, *Variable, half, *Variable, float));
    try std.testing.expectError(error.VariableDTypeMismatch, min(allocator, *Variable, half, *Variable, float));
}
