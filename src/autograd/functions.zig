const zt = @import("../zt.zig");
const std = @import("std");

const Variable = @import("autograd.zig").Variable;
const Index = zt.tensor.Index;
const Range = zt.tensor.Range;
const Tensor = zt.tensor.Tensor;
const Shape = zt.tensor.shape.Shape;
const Dim = zt.tensor.shape.Dim;

pub const detail = struct {
    /// Performs type conversion based on the optim level. Operations that lack
    /// sufficient precision are automatically upcast to f32 before computation.
    /// These are typically operations that require accumulations or reductions.
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

/// Checks if a variadic number of Variables have the same types.
pub inline fn ztVariableDTypesMatch(allocator: std.mem.Allocator, fn_name: []const u8, vars: []const *Variable) !void {
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
    var var1 = try Variable.initSharedData(allocator, grad_output.shared_data.retain(), false);
    defer var1.deinit();
    var var2 = try Variable.initSharedData(allocator, grad_output.shared_data.retain(), false);
    defer var2.deinit();
    try inputs[0].addGrad(allocator, var1);
    try inputs[1].addGrad(allocator, var2);
}

fn lhsAddGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    try inputs[0].addGrad(allocator, try Variable.initSharedData(allocator, grad_output.shared_data.retain(), false));
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
        return Variable.initWithInputs(allocator, result, &.{try lhs.withoutData(allocator)}, lhsAddGradFunc, null, null);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return add(allocator, RhsT, rhs, LhsT, lhs);
    }
}

fn bothSubGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var var1 = try Variable.initSharedData(allocator, grad_output.shared_data.retain(), false);
    defer var1.deinit();
    try inputs[0].addGrad(allocator, var1);
    var neg = try negate(allocator, grad_output);
    defer neg.deinit();
    var var2 = try Variable.initSharedData(allocator, neg.shared_data.retain(), false);
    defer var2.deinit();
    try inputs[1].addGrad(allocator, var2);
}

fn lhsSubGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var var1 = try Variable.initSharedData(allocator, grad_output.shared_data.retain(), false);
    defer var1.deinit();
    try inputs[0].addGrad(allocator, var1);
}

fn rhsSubGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var neg = try negate(allocator, &grad_output);
    defer neg.deinit();
    var var1 = try Variable.initSharedData(allocator, neg.shared_data.retain(), false);
    defer var1.deinit();
    try inputs[0].addGrad(allocator, var1);
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
        var tmp_var = try Variable.init(allocator, try zt.tensor.mul(allocator, Tensor, grad_output.tensor(), Tensor, inputs[1].tensor()), false);
        defer tmp_var.deinit();
        try inputs[0].addGrad(allocator, tmp_var);
    }
    if (inputs[1].isCalcGrad()) {
        var tmp_var = try Variable.init(allocator, try zt.tensor.mul(allocator, Tensor, grad_output.tensor(), Tensor, inputs[0].tensor()), false);
        defer tmp_var.deinit();
        try inputs[1].addGrad(allocator, tmp_var);
    }
}

const F64Ctx = struct { val: f64 };

fn freeF64Ctx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
    allocator.destroy(grad_ctx);
}

fn lhsMulGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
    var tmp_var = try Variable.init(allocator, try zt.tensor.mul(allocator, Tensor, grad_output.tensor(), f64, grad_ctx.val), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn mul(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.mul only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.mul(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        var in1: *Variable = if (rhs.isCalcGrad()) try lhs.clone(allocator) else try lhs.withoutData(allocator);
        var in2: *Variable = if (lhs.isCalcGrad()) try rhs.clone(allocator) else try rhs.withoutData(allocator);
        return Variable.initWithInputs(allocator, result, &.{ in1, in2 }, bothMulGradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.mul(allocator, Tensor, lhs.tensor(), f64, rhs);
        var ctx = try allocator.create(F64Ctx);
        ctx.* = .{ .val = rhs };
        return Variable.initWithInputs(allocator, result, &.{try lhs.withoutData(allocator)}, lhsMulGradFunc, ctx, freeF64Ctx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return mul(allocator, RhsT, rhs, LhsT, lhs);
    }
}

fn bothDivGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var inputs1_rec = try negate(allocator, inputs[1]);
    defer inputs1_rec.deinit();
    var grad_input0 = try mul(allocator, *Variable, grad_output, *Variable, inputs1_rec);
    defer grad_input0.deinit();
    if (inputs[0].isCalcGrad()) {
        var tmp_var = try Variable.initSharedData(allocator, grad_input0.shared_data.retain(), false);
        defer tmp_var.deinit();
        try inputs[0].addGrad(allocator, tmp_var);
    }
    if (inputs[1].isCalcGrad()) {
        var tmp_neg = try negate(allocator, inputs[0]);
        defer tmp_neg.deinit();
        var tmp = try mul(allocator, *Variable, grad_input0, *Variable, tmp_neg);
        defer tmp.deinit();
        var tmp2 = try mul(allocator, *Variable, tmp, *Variable, inputs1_rec);
        defer tmp2.deinit();
        var tmp_var = try Variable.initSharedData(allocator, tmp2.shared_data.retain(), false);
        defer tmp_var.deinit();
        try inputs[1].addGrad(allocator, tmp_var);
    }
}

fn lhsDivGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
    var tmp = try div(allocator, Variable, grad_output, f64, grad_ctx.val);
    defer tmp.deinit();
    var tmp_var = try Variable.initSharedData(allocator, tmp.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

fn rhsDivGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
    var tmp = try mul(allocator, Variable, grad_output, f64, -grad_ctx.val);
    defer tmp.deinit();
    var tmp2 = try mul(allocator, Variable, inputs[0], Variable, inputs[0]);
    defer tmp2.deinit();
    var res = try div(allocator, Variable, tmp, Variable, tmp2);
    defer res.deinit();
    var tmp_var = try Variable.initSharedData(allocator, res.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn div(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.div only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.div(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        var in1: *Variable = if (rhs.isCalcGrad()) try lhs.clone(allocator) else try lhs.withoutData(allocator);
        return Variable.initWithInputs(allocator, result, &.{ in1, try rhs.clone(allocator) }, bothDivGradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.div(allocator, Tensor, lhs.tensor(), f64, rhs);
        var ctx = try allocator.create(F64Ctx);
        ctx.* = .{ .val = rhs };
        return Variable.initWithInputs(allocator, result, &.{try lhs.withoutData(allocator)}, lhsDivGradFunc, ctx, freeF64Ctx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        var result = try zt.tensor.div(allocator, f64, lhs, Tensor, rhs.tensor());
        var ctx = try allocator.create(F64Ctx);
        ctx.* = .{ .val = lhs };
        return Variable.initWithInputs(allocator, result, &.{try rhs.clone(allocator)}, rhsDivGradFunc, ctx, freeF64Ctx);
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
    var mask = try Variable.init(allocator, try tmp_mask.astype(allocator, try grad_output.dtype(allocator)), false);
    defer mask.deinit();
    var tmp1 = try mul(allocator, *Variable, mask, *Variable, grad_output);
    defer tmp1.deinit();
    var tmp_var1 = try Variable.initSharedData(allocator, tmp1.shared_data.retain(), false);
    defer tmp_var1.deinit();
    try inputs[0].addGrad(allocator, tmp_var1);
    var tmp_not = try logicalNot(allocator, mask);
    defer tmp_not.deinit();
    var tmp2 = try mul(allocator, *Variable, tmp_not, *Variable, grad_output);
    defer tmp2.deinit();
    var tmp_var2 = try Variable.initSharedData(allocator, tmp2.shared_data.retain(), false);
    defer tmp_var2.deinit();
    try inputs[1].addGrad(allocator, tmp_var2);
}

fn lhsMaxGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
    var tmp_mask = try zt.tensor.greaterThan(allocator, Tensor, inputs[0].tensor(), f64, grad_ctx.val);
    defer tmp_mask.deinit();
    var mask = try Variable.init(allocator, try tmp_mask.astype(allocator, try grad_output.dtype(allocator)), false);
    defer mask.deinit();
    var tmp1 = try mul(allocator, *Variable, mask, *Variable, grad_output);
    defer tmp1.deinit();
    var tmp_var = try Variable.initSharedData(allocator, tmp1.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn max(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.max only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.maximum(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.initWithInputs(allocator, result, &.{ try lhs.clone(allocator), try rhs.clone(allocator) }, bothMaxGradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.maximum(allocator, Tensor, lhs.tensor(), f64, rhs.tensor());
        var ctx = try allocator.create(F64Ctx);
        ctx.* = .{ .val = rhs };
        return Variable.initWithInputs(allocator, result, &.{try lhs.clone(allocator)}, lhsMaxGradFunc, ctx, freeF64Ctx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return max(allocator, RhsT, rhs, LhsT, lhs);
    }
}

fn bothMinGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var tmp_mask = try zt.tensor.lessThan(allocator, Tensor, inputs[0].tensor(), Tensor, inputs[1].tensor());
    defer tmp_mask.deinit();
    var mask = try Variable.init(allocator, try tmp_mask.astype(allocator, try grad_output.dtype(allocator)), false);
    defer mask.deinit();
    var tmp1 = try mul(allocator, *Variable, mask, *Variable, grad_output);
    defer tmp1.deinit();
    var tmp_var1 = try Variable.initSharedData(allocator, tmp1.shared_data.retain(), false);
    defer tmp_var1.deinit();
    try inputs[0].addGrad(allocator, tmp_var1);
    var tmp_not = try logicalNot(allocator, mask);
    defer tmp_not.deinit();
    var tmp2 = try mul(allocator, *Variable, tmp_not, *Variable, grad_output);
    defer tmp2.deinit();
    var tmp_var2 = try Variable.initSharedData(allocator, tmp2.shared_data.retain(), false);
    defer tmp_var2.deinit();
    try inputs[1].addGrad(allocator, tmp_var2);
}

fn lhsMinGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
    var tmp_mask = try zt.tensor.lessThan(allocator, Tensor, inputs[0].tensor(), f64, grad_ctx.val);
    defer tmp_mask.deinit();
    var mask = try Variable.init(allocator, try tmp_mask.astype(allocator, try grad_output.dtype(allocator)), false);
    defer mask.deinit();
    var tmp1 = try mul(allocator, *Variable, mask, *Variable, grad_output);
    defer tmp1.deinit();
    var tmp_var = try Variable.initSharedData(allocator, tmp1.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn min(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.min only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        var result = try zt.tensor.minimum(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.initWithInputs(allocator, result, &.{ try lhs.clone(allocator), try rhs.clone(allocator) }, bothMinGradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        var result = try zt.tensor.minimum(allocator, Tensor, lhs.tensor(), f64, rhs.tensor());
        var ctx = try allocator.create(F64Ctx);
        ctx.* = .{ .val = rhs };
        return Variable.initWithInputs(allocator, result, &.{try lhs.clone(allocator)}, lhsMinGradFunc, ctx, freeF64Ctx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return min(allocator, RhsT, rhs, LhsT, lhs);
    }
}

fn negateGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var tmp_neg = try negate(allocator, grad_output);
    defer tmp_neg.deinit();
    var tmp_var = try Variable.initSharedData(allocator, tmp_neg.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn negate(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.sub(allocator, f64, 0, Tensor, input.tensor());
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, negateGradFunc, null, null);
}

fn reciprocalGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var res = try reciprocal(allocator, inputs[0]);
    var tmp_neg = try negate(allocator, grad_output);
    defer tmp_neg.deinit();
    var tmp1 = try mul(allocator, *Variable, tmp_neg, *Variable, res);
    defer tmp1.deinit();
    var tmp2 = try mul(allocator, *Variable, tmp1, *Variable, res);
    defer tmp2.deinit();
    var tmp_var = try Variable.initSharedData(allocator, tmp2.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn reciprocal(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.div(allocator, f64, 1, Tensor, adj.res);
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, reciprocalGradFunc, null, null);
}

fn expGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var res = try zt.tensor.exp(allocator, inputs[0].tensor());
    try res.inPlaceMul(allocator, Tensor, grad_output.tensor());
    var tmp_var = try Variable.init(allocator, res, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn exp(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.exp(allocator, adj.res);
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, expGradFunc, null, null);
}

fn logGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var res = try zt.tensor.div(allocator, Tensor, grad_output.tensor(), Tensor, inputs[0].tensor());
    var tmp_var = try Variable.init(allocator, res, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn log(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.log(allocator, adj.res);
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, logGradFunc, null, null);
}

fn log1pGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var tmp = try zt.tensor.add(allocator, f64, 1, Tensor, inputs[0].tensor());
    defer tmp.deinit();
    var res = try zt.tensor.div(allocator, Tensor, grad_output.tensor(), Tensor, tmp);
    var tmp_var = try Variable.init(allocator, res, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn log1p(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var res = try zt.tensor.log1p(allocator, adj.res);
    return Variable.initWithInputs(allocator, res, &.{try input.clone(allocator)}, log1pGradFunc, null, null);
}

fn powGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
    var grad = try zt.tensor.power(allocator, Tensor, inputs[0].tensor(), f64, grad_ctx.val - 1);
    try grad.inPlaceMul(allocator, f64, grad_ctx.val);
    try grad.inPlaceMul(allocator, Tensor, grad_output.tensor());
    var tmp_var = try Variable.init(allocator, grad, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn pow(allocator: std.mem.Allocator, input: *Variable, p: f64) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.power(allocator, Tensor, adj.res, f64, p);
    var ctx = try allocator.create(F64Ctx);
    ctx.* = .{ .val = p };
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, powGradFunc, ctx, freeF64Ctx);
}

fn sinGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var res = try zt.tensor.cos(allocator, inputs[0].tensor());
    try res.inPlaceMul(allocator, Tensor, grad_output.tensor());
    var tmp_var = try Variable.init(allocator, res, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn sin(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.sin(allocator, input.tensor());
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, sinGradFunc, null, null);
}

fn cosGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var tmp = try zt.tensor.sin(allocator, inputs[0].tensor());
    defer tmp.deinit();
    var res = try zt.tensor.negative(allocator, tmp);
    try res.inPlaceMul(allocator, Tensor, grad_output.tensor());
    var tmp_var = try Variable.init(allocator, res, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn cos(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.cos(allocator, input.tensor());
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, cosGradFunc, null, null);
}

const TensorCtx = struct { val: Tensor };
fn freeTensorCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var grad_ctx: *TensorCtx = @ptrCast(@alignCast(ctx));
    allocator.destroy(grad_ctx);
}

fn tanhGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: *TensorCtx = @ptrCast(@alignCast(ctx));
    var tmp1 = try zt.tensor.mul(allocator, Tensor, grad_ctx.val, Tensor, grad_ctx.val);
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.sub(allocator, i8, 1, Tensor, tmp1);
    try tmp2.inPlaceMul(allocator, Tensor, grad_output.tensor());
    var tmp_var = try Variable.init(allocator, tmp2, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn tanh(allocator: std.mem.Allocator, input: *const Variable) !*Variable {
    var result = try zt.tensor.tanh(allocator, input.tensor());
    var ctx = try allocator.create(TensorCtx);
    ctx.* = .{ .val = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, tanhGradFunc, ctx, freeTensorCtx);
}

const ClampCtx = struct { lo: f64, hi: f64, result: Tensor };

fn freeClampCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var grad_ctx: *ClampCtx = @ptrCast(@alignCast(ctx));
    allocator.destroy(grad_ctx);
}

fn clampGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: *ClampCtx = @ptrCast(@alignCast(ctx));
    var tmp1 = try Tensor.initAssign(allocator, grad_output.tensor());
    var tmp2 = try zt.tensor.greaterThan(allocator, Tensor, grad_ctx.result, f64, grad_ctx.lo);
    var tmp3 = try zt.tensor.lessThan(allocator, Tensor, grad_ctx.result, f64, grad_ctx.hi);
    var tmp4 = try zt.tensor.logicalAnd(allocator, Tensor, tmp1, Tensor, tmp2);
    tmp2.deinit();
    tmp3.deinit();
    var grad_mask = try zt.tensor.where(allocator, tmp3, Tensor, tmp1, f64, 0);
    tmp4.deinit();
    var tmp_var = try Variable.init(allocator, grad_mask, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn clamp(allocator: std.mem.Allocator, input: *Variable, lo: f64, hi: f64) !*Variable {
    var result = try zt.tensor.clip(allocator, input.tensor(), f64, lo, f64, hi);
    var ctx = try allocator.create(ClampCtx);
    ctx.* = .{ .lo = lo, .hi = hi, .result = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, clampGradFunc, ctx, freeClampCtx);
}

fn sqrtGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: *TensorCtx = @ptrCast(@alignCast(ctx));
    var output = try Variable.init(allocator, grad_ctx.val, false);
    var tmp = try mul(allocator, f64, 2, *Variable, output);
    output.deinit();
    var res = try div(allocator, *Variable, grad_output, *Variable, tmp);
    defer res.deinit();
    tmp.deinit();
    var tmp_var = try Variable.initSharedData(allocator, res.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn sqrt(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.sqrt(allocator, input.tensor());
    var ctx = try allocator.create(TensorCtx);
    ctx.* = .{ .val = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, sqrtGradFunc, ctx, freeTensorCtx);
}

fn sigmoidGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var grad_ctx: *TensorCtx = @ptrCast(@alignCast(ctx));
    var tmp = try zt.tensor.sub(allocator, f64, 1, Tensor, grad_ctx.val);
    try tmp.inPlaceMul(allocator, Tensor, grad_ctx.val);
    var grad = try zt.tensor.mul(allocator, Tensor, grad_output.tensor(), Tensor, tmp);
    tmp.deinit();
    var tmp_var = try Variable.init(allocator, grad, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn sigmoid(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.sigmoid(allocator, input.tensor());
    var ctx = try allocator.create(TensorCtx);
    ctx.* = .{ .val = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, sigmoidGradFunc, ctx, freeTensorCtx);
}

pub fn swish(allocator: std.mem.Allocator, input: *Variable, beta: f64) !*Variable {
    var tmp1 = try mul(allocator, f64, beta, *Variable, input);
    defer tmp1.deinit();
    var tmp2 = try sigmoid(allocator, tmp1);
    defer tmp2.deinit();
    return mul(allocator, *Variable, input, *Variable, tmp2);
}

fn erfGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var x = inputs[0].tensor();
    var grad = try zt.tensor.mul(allocator, grad_output.tensor(), f64, 2);
    try grad.inPlaceDiv(allocator, f64, @sqrt(@as(f64, std.math.pi)));
    var tmp1 = try zt.tensor.mul(allocator, Tensor, x, Tensor, x);
    defer tmp1.deinit();
    try tmp1.inPlaceMul(allocator, f64, -1);
    var tmp2 = try zt.tensor.exp(allocator, tmp1);
    defer tmp2.deinit();
    try grad.inPlaceMul(allocator, Tensor, tmp2);
    var tmp_var = try Variable.init(allocator, grad, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn erf(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.erf(allocator, adj.res);
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, erfGradFunc, null, null);
}

const DimsCtx = struct { dims: []Dim };

fn freeDimsCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var transpose_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
    allocator.free(transpose_ctx.dims);
    allocator.destroy(transpose_ctx);
}

fn transposeGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var transpose_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
    var reverse_shape = transpose_ctx.dims;
    if (zt.tensor.shape.ndim(transpose_ctx.dims) != 0) {
        // Reverse if transposing all dims (empty arg)
        std.mem.reverse(Dim, reverse_shape);
    }
    for (0..zt.tensor.shape.ndim(reverse_shape)) |i| {
        reverse_shape[@intCast(transpose_ctx.dims[i])] = i;
    }
    var tmp_var = try Variable.init(
        allocator,
        try zt.tensor.transpose(allocator, grad_output.tensor(), reverse_shape),
    );
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn transpose(allocator: std.mem.Allocator, input: *Variable, dims: []const Dim) !*Variable {
    var result = try zt.tensor.transpose(allocator, input.tensor(), dims);
    var ctx = try allocator.create(DimsCtx);
    ctx.* = .{ .dims = try allocator.alloc(Dim, dims.len) };
    @memcpy(ctx.dims, dims);
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, transposeGradFunc, ctx, freeDimsCtx);
}

fn tileAsGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var tile_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
    var tmp_sum = try sumAs(allocator, grad_output, tile_ctx.dims);
    defer tmp_sum.deinit();
    var tmp_var = try Variable.init(allocator, try tmp_sum.tensor().astype(allocator, try inputs[0].dtype(allocator)), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

fn tileAs(allocator: std.mem.Allocator, input: *Variable, comptime T: type, ref: T) !*Variable {
    return switch (T) {
        *Variable => tileAs(allocator, input, Shape, try ref.shape(allocator)),
        Shape => {
            var result = try detail.tileAs(allocator, input.tensor(), ref);
            var in_dims = try input.shape(allocator);
            var ctx = try allocator.create(DimsCtx);
            ctx.* = .{ .dims = try allocator.alloc(Dim, in_dims.len) };
            @memcpy(ctx.dims, in_dims);
            return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, tileAsGradFunc, ctx, freeDimsCtx);
        },
        else => @compileError("autograd.tileAs only supports ref value of *Variable or Shape"),
    };
}

fn sumAsGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var sum_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
    var tmp_tiled = try tileAs(allocator, grad_output, sum_ctx.dims);
    defer tmp_tiled.deinit();
    var tmp_var = try Variable.initSharedData(allocator, tmp_tiled.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

fn sumAs(allocator: std.mem.Allocator, input: *Variable, comptime T: type, ref: T) !*Variable {
    return switch (T) {
        *Variable => sumAs(allocator, input, Shape, try ref.shape(allocator)),
        Shape => {
            var result = try detail.sumAs(allocator, input.tensor(), ref);
            var in_dims = try input.shape(allocator);
            var ctx = try allocator.create(DimsCtx);
            ctx.* = .{ .dims = try allocator.alloc(Dim, in_dims.len) };
            @memcpy(ctx.dims, in_dims);
            return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, sumAsGradFunc, ctx, freeDimsCtx);
        },
        else => @compileError("autograd.tileAs only supports ref value of *Variable or Shape"),
    };
}

const ConcatCtx = struct {
    dim: Dim,
    in_dims: []Shape,
    numdims: usize,
};

fn freeConcatCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var concat_ctx: *ConcatCtx = @ptrCast(@alignCast(ctx));
    allocator.free(concat_ctx.in_dims);
    allocator.destroy(concat_ctx);
}

fn concatGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var concat_ctx: *ConcatCtx = @ptrCast(@alignCast(ctx));
    var sx = try allocator.alloc(Index, concat_ctx.numdims);
    defer allocator.free(sx);
    @memset(sx, zt.tensor.span);
    var s: Dim = 0;
    for (0..inputs.len) |i| {
        sx[@intCast(concat_ctx.dim)] = Index.initRange(Range.init(s, .{ .dim = s + concat_ctx.in_dims[i][@intCast(concat_ctx.dim)] }));
        var tmp_var = try Variable.init(allocator, try grad_output.tensor().index(allocator, sx), false);
        defer tmp_var.deinit();
        try inputs[i].addGrad(allocator, tmp_var);
        s += concat_ctx.in_dims[i][@intCast(concat_ctx.dim)];
    }
}

pub fn concatenate(allocator: std.mem.Allocator, concat_inputs: []const *Variable, dim: Dim) !*Variable {
    if (concat_inputs.len == 0) {
        std.debug.print("cannot concatenate zero variables\n", .{});
        return error.ConcatenateZeroVariables;
    }
    if (concat_inputs.len == 1) {
        return concat_inputs[0].clone(allocator);
    }
    // All Variables must be of the same type and have the same number of dims
    var dtype = try concat_inputs[0].dtype(allocator);
    var numdims = try concat_inputs[0].ndim(allocator);
    for (concat_inputs[1..]) |v| {
        if (try v.dtype(allocator) != dtype) {
            std.debug.print("concatenate: all input Variables must be of the same type\n", .{});
            return error.ConcatenateInputVariableTypeMismatch;
        } else if (try v.ndim(allocator) != numdims) {
            std.debug.print("concatenate: all input Variables must have the same number of dimensions\n", .{});
            return error.ConcatenateInputVariableDimsMismatch;
        }
    }

    // All Variables must have the same size when indexed along the dim not being
    // concatenated along
    var dims = try concat_inputs[0].shape(allocator);
    var concat_size = dims[@intCast(dim)];
    for (1..concat_inputs.len) |i| {
        concat_size += try concat_inputs[i].dim(allocator, i);
        for (0..numdims) |d| {
            if (dim != @as(Dim, @intCast(d)) and try concat_inputs[i].dim(allocator, d) != dims[@intCast(d)]) {
                std.debug.print("mismatch in dimension not being concatenated\n", .{});
                return error.ConcatenateDimsMismatch;
            }
        }
    }
    var new_dims = try allocator.alloc(Dim, dims.len);
    defer allocator.free(new_dims);
    @memcpy(new_dims, dims);
    new_dims[@intCast(dim)] = concat_size;
    var result = try Tensor.initHandle(allocator, new_dims, dtype);
    var slice = try allocator.alloc(Index, numdims);
    defer allocator.free(slice);
    @memset(slice, zt.tensor.span);
    var start: Dim = 0;
    var inputs_no_data = std.ArrayList(*Variable).init(allocator);
    defer inputs_no_data.deinit();
    var in_dims = std.ArrayList(Shape).init(allocator);
    for (concat_inputs) |input| {
        slice[@intCast(dim)] = Index.initRange(Range.init(start, .{ .dim = start + try input.dim(allocator, @intCast(dim)) }));
        try result.indexAssign(allocator, Tensor, input.tensor(), slice);
        start += try input.dim(allocator, @intCast(dim));
        try inputs_no_data.append(try input.withoutData(allocator));
        try in_dims.append(try input.shape(allocator));
    }

    var ctx = try allocator.create(ConcatCtx);
    ctx.* = .{ .dim = dim, .in_dims = try in_dims.toOwnedSlice(), .numdims = numdims };
    return Variable.initWithInputs(allocator, result, inputs_no_data.items, concatGradFunc, ctx, freeConcatCtx);
}

// TODO pub fn split() !*Variable {}

fn tileGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var tile_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
    var tmp_sum = try sumAs(allocator, grad_output, Shape, tile_ctx.dims);
    defer tmp_sum.deinit();
    var tmp_var = try Variable.init(allocator, try tmp_sum.tensor().astype(allocator, try inputs[0].dtype(allocator)), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn tile(allocator: std.mem.Allocator, input: *Variable, dims: Shape) !*Variable {
    var result = try zt.tensor.tile(allocator, input.tensor(), dims);
    var idims = try input.shape(allocator);
    var ctx = try allocator.create(DimsCtx);
    ctx.* = .{ .dims = try allocator.alloc(Dim, idims.len) };
    @memcpy(ctx.dims, idims);
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, tileGradFunc, ctx, freeDimsCtx);
}

// Unit tests
const JacobianFunc = *const fn (allocator: std.mem.Allocator, input: *Variable) anyerror!*Variable;

inline fn jacobianTestImpl(
    allocator: std.mem.Allocator,
    func: JacobianFunc,
    input: *Variable,
    precision: f32,
    perturbation: f32,
) !bool {
    var fwd_jacobian_variable = try func(allocator, input);
    defer fwd_jacobian_variable.deinit();
    var fwd_jacobian = try Tensor.initHandle(allocator, &.{ try fwd_jacobian_variable.elements(allocator), try input.elements(allocator) }, .f32);
    defer fwd_jacobian.deinit();

    for (0..@intCast(try input.elements(allocator))) |i| {
        var tmp_flat = try input.tensor().flatten(allocator);
        defer tmp_flat.deinit();
        var orig = try tmp_flat.index(allocator, &.{Index.initDim(@intCast(i))});
        defer orig.deinit();
        var assign1 = try zt.tensor.sub(allocator, Tensor, orig, f32, perturbation);
        defer assign1.deinit();
        try input.tensor().flatAssign(allocator, Tensor, assign1, Index.initDim(@intCast(i)));
        var outa_variable: *Variable = try func(allocator, input);
        defer outa_variable.deinit();
        var outa = outa_variable.tensor();

        var assign2 = try zt.tensor.add(allocator, Tensor, orig, f32, perturbation);
        defer assign2.deinit();
        try input.tensor().flatAssign(allocator, Tensor, assign2, Index.initDim(@intCast(i)));
        var outb_variable: *Variable = try func(allocator, input);
        defer outb_variable.deinit();
        var outb = outb_variable.tensor();
        try input.tensor().flatAssign(allocator, Tensor, orig, Index.initDim(@intCast(i)));

        var tmp = try zt.tensor.sub(allocator, Tensor, outb, Tensor, outa);
        defer tmp.deinit();
        var assign3 = try zt.tensor.reshape(allocator, tmp, &.{try outa.elements(allocator)});
        defer assign3.deinit();
        try assign3.inPlaceMul(allocator, f64, 0.5);
        try assign3.inPlaceDiv(allocator, f32, perturbation);
        try fwd_jacobian.indexAssign(allocator, Tensor, assign3, &.{ Index.initRange(zt.tensor.span), Index.initDim(@intCast(i)) });
    }
    var bwd_jacobian_variable = try func(allocator, input);
    defer bwd_jacobian_variable.deinit();
    var bwd_jacobian = try Tensor.initHandle(allocator, &.{ try bwd_jacobian_variable.elements(allocator), try input.elements(allocator) }, .f32);
    defer bwd_jacobian.deinit();
    var dout_jacobian_var = try func(allocator, input);
    defer dout_jacobian_var.deinit();
    var dout = try Variable.init(allocator, try zt.tensor.full(allocator, try dout_jacobian_var.shape(allocator), f64, 0, try dout_jacobian_var.dtype(allocator)), false);
    defer dout.deinit();

    for (0..@intCast(try dout.elements(allocator))) |i| {
        try dout.tensor().flatAssign(allocator, f64, 1, Index.initDim(@intCast(i)));
        input.zeroGrad();
        var out: *Variable = try func(allocator, input);
        try out.backwardAddGrad(allocator, dout, false);

        var in_grad = try input.grad();
        var assign = try zt.tensor.reshape(allocator, in_grad.tensor(), &.{try input.elements(allocator)});
        defer assign.deinit();
        try bwd_jacobian.indexAssign(allocator, Tensor, assign, &.{Index.initDim(@intCast(i))});
        try dout.tensor().flatAssign(allocator, f64, 0, Index.initDim(@intCast(i)));
    }
    return zt.tensor.allClose(allocator, fwd_jacobian, bwd_jacobian, @floatCast(precision));
}

fn funcIdx(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var tmp_idx1 = try input.index(allocator, &.{ Index.initDim(0), Index.initDim(0) });
    defer tmp_idx1.deinit();
    var tmp_idx2 = try input.index(allocator, &.{ Index.initDim(0), Index.initDim(1) });
    defer tmp_idx2.deinit();
    return add(allocator, *Variable, tmp_idx1, *Variable, tmp_idx2);
}

test "AutogradTest -> AutogradVariableIndex" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons
    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 1, 3, 3 }, .f64), true);
    defer x.deinit();
    var tmp_x_idx1 = try x.index(allocator, &.{ Index.initDim(0), Index.initDim(0) });
    defer tmp_x_idx1.deinit();
    var tmp_x_idx2 = try x.index(allocator, &.{ Index.initDim(0), Index.initDim(1) });
    defer tmp_x_idx2.deinit();
    var y = try add(allocator, *Variable, tmp_x_idx1, *Variable, tmp_x_idx2);
    defer y.deinit();
    try std.testing.expect(try jacobianTestImpl(allocator, funcIdx, x, 1e-5, 1e-4));
}

test "AutogradTest -> AutogradOperatorTypeCompatibility" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons
    if (!try zt.f16Supported(allocator)) {
        return error.SkipZigTest;
    }

    var half = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 2 }, .f16), true);
    defer half.deinit();
    var float = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 2 }, .f32), true);
    defer float.deinit();
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

    // TODO: finish tests
}

test "AutogradTest -> CastingAsDifferentGradTypes" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons
    if (!try zt.f16Supported(allocator)) {
        return error.SkipZigTest;
    }
    var half = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 5 }, .f16), true);
    defer half.deinit();
    var float = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 5 }, .f32), true);
    defer float.deinit();
    // Computing gradients with mixed types fails when the op is applied
    try std.testing.expectError(error.VariableDTypeMismatch, add(allocator, *Variable, half, *Variable, float));
}

test "AutogradTest -> CastingAs" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons
    if (!try zt.f16Supported(allocator)) {
        return error.SkipZigTest;
    }

    var var_f32 = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 5 }, .f32), true);
    defer var_f32.deinit();
    var var_f16 = try var_f32.astype(allocator, .f16);
    defer var_f16.deinit();
    try std.testing.expect(try var_f32.dtype(allocator) == .f32);
    try std.testing.expect(try var_f16.dtype(allocator) == .f16);
    var comp_tensor = try var_f32.tensor().astype(allocator, .f16);
    defer comp_tensor.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, var_f16.tensor(), comp_tensor, 1e-5));
}
