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
        // Fastpath - Default mode never casts tensors
        if (optim_level == .Default) {
            return .{ .res = in, .allocated = false };
        }

        if (!zt.kOptimLevelTypeExclusionMappings(optim_level, func_name)) {
            // Not in the excluded list - cast to f16
            return .{ .res = try in.astype(allocator, .f16), .allocated = true };
        } else {
            // Upcast to f32 only if we have an f16 input - otherwise, leave as is
            if (try in.dtype(allocator) == .f16) {
                return .{ .res = try in.astype(allocator, .f32), .allocated = true };
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
            if (@rem(rdims[i], idims_size) != 0) {
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

    const ExpandedShapeRes = struct { shape: Shape, allocated: bool };
    pub fn expandedShapeFromReducedDims(allocator: std.mem.Allocator, input: Tensor, axes: []const Dim, keep_dims: bool) !ExpandedShapeRes {
        // Fast path - tensor already retained its shape
        if (keep_dims) {
            return .{ .shape = try input.shape(allocator), .allocated = false };
        }
        // If we output a scalar,
        if (try input.ndim(allocator) == 0) {
            return .{ .shape = &.{}, .allocated = false };
        }

        const pre_ndims = try input.ndim(allocator) + axes.len;
        var new_shape = try allocator.alloc(Dim, pre_ndims);
        @memset(new_shape, 1);
        var axes_idx: usize = 0;
        var input_idx: usize = 0;
        for (0..pre_ndims) |i| {
            if (i == axes[axes_idx]) {
                // This dim was reduced over, leave as 1 in the new shape
                axes_idx += 1;
            } else {
                // Dim wasn't reduced over - add the shape from the new tensor
                new_shape[i] = try input.dim(allocator, input_idx);
                input_idx += 1;
            }
        }
        return .{ .shape = new_shape, .allocated = true };
    }

    pub fn expandFromReduction(allocator: std.mem.Allocator, comptime T: type, input: T, axes: []const Dim, keep_dims: bool) !T {
        return switch (T) {
            Tensor => {
                var o = try expandedShapeFromReducedDims(allocator, input, axes, keep_dims);
                defer if (o.allocated) allocator.free(o.shape);
                return zt.tensor.reshape(allocator, input, o.shape);
            },
            *Variable => {
                var o = try expandedShapeFromReducedDims(allocator, input.tensor(), axes, keep_dims);
                defer if (o.allocated) allocator.free(o.shape);
                return moddims(allocator, input, o.shape);
            },
            else => @compileError("autograd.detail.expandFromReduction only supports inputs of type Tensor or *Variable"),
        };
    }

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
    var tmp_var = try Variable.initSharedData(allocator, grad_output.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
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
    var neg = try negate(allocator, grad_output);
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
    var tmp_sum = try sumAs(allocator, grad_output, Shape, tile_ctx.dims);
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
    var tmp_tiled = try tileAs(allocator, grad_output, Shape, sum_ctx.dims);
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
    @memset(sx, Index.initRange(zt.tensor.span));
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
        concat_size += try concat_inputs[i].dim(allocator, @intCast(dim));
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
    @memset(slice, Index.initRange(zt.tensor.span));
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

pub fn split(allocator: std.mem.Allocator, input: *Variable, comptime T: type, splits: T, dim: Dim) ![]*Variable {
    return switch (T) {
        i64 => {
            if (splits <= 0) {
                std.debug.print("autograd.split: splits must be a positive integer\n", .{});
                return error.InvalidSplitSize;
            }
            var dim_size = try input.dim(allocator, @intCast(dim));
            var split_sizes = try std.ArrayList(i64).initCapacity(allocator, @intCast(@divTrunc(dim_size, splits)));
            defer split_sizes.deinit();
            split_sizes.appendNTimesAssumeCapacity(splits, @intCast(@divTrunc(dim_size, splits)));

            if (@mod(dim_size, splits) > 0) {
                try split_sizes.append(@mod(dim_size, splits));
            }
            return split(allocator, input, []const i64, split_sizes.items, dim);
        },
        []const i64 => {
            if (dim >= @as(i64, @intCast(try input.ndim(allocator)))) {
                std.debug.print("autograd.split: passed dim is larger than the number of dimensions of the input.\n", .{});
                return error.SplitDimExceedsNumDims;
            }
            var dim_size = try input.dim(allocator, @intCast(dim));
            var N = splits.len;
            var outputs = try allocator.alloc(*Variable, N);
            errdefer {
                for (outputs) |v| v.deinit();
                allocator.free(outputs);
            }
            var sel = try allocator.alloc(Index, try input.ndim(allocator));
            defer allocator.free(sel);
            @memset(sel, Index.initRange(zt.tensor.span));
            var start: Dim = 0;
            for (0..N) |i| {
                if (splits[i] <= 0) {
                    std.debug.print("autograd.split: elements in splits must be positive\n", .{});
                    return error.InvalidSplitSizes;
                }
                var end: Dim = start + splits[i];
                sel[@intCast(dim)] = Index.initRange(Range.init(start, .{ .dim = end }));
                outputs[i] = try input.index(allocator, sel);
                start = end;
            }
            if (start != dim_size) {
                std.debug.print("autograd.split: sum of split sizes must match split dim\n", .{});
                return error.SumOfSplitMismatch;
            }
            return outputs;
        },
        else => @compileError("autograd.split: splits must be of type []const i64 or i64"),
    };
}

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

const ReductionCtx = struct {
    in_dims: Shape,
    axes: []i64,
    keep_dims: bool,
};

fn freeSumCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var sum_ctx: *ReductionCtx = @ptrCast(@alignCast(ctx));
    allocator.free(sum_ctx.in_dims);
    allocator.free(sum_ctx.axes);
    allocator.destroy(sum_ctx);
}

fn sumGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var sum_ctx: *ReductionCtx = @ptrCast(@alignCast(ctx));
    var tmp = try detail.expandFromReduction(allocator, Tensor, grad_output.tensor(), sum_ctx.axes, sum_ctx.keep_dims);
    defer tmp.deinit();
    var tmp_var = try Variable.init(allocator, try detail.tileAs(allocator, tmp, sum_ctx.in_dims), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn sum(allocator: std.mem.Allocator, input: *Variable, axes: []const i64, keep_dims: bool) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.sum(allocator, adj.res, axes, keep_dims);
    var indims = try input.shape(allocator);
    var ctx = try allocator.create(ReductionCtx);
    ctx.* = .{ .in_dims = try allocator.alloc(Dim, indims.len), .axes = try allocator.alloc(i64, axes.len), .keep_dims = keep_dims };
    @memcpy(ctx.in_dims, indims);
    @memcpy(ctx.axes, axes);
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, sumGradFunc, ctx, freeSumCtx);
}

fn meanGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var mean_ctx: *ReductionCtx = @ptrCast(@alignCast(ctx));
    var odims = try grad_output.shape(allocator);
    var count: Dim = 1;
    for (0..zt.tensor.shape.ndim(try mean_ctx.in_dims)) |i| {
        const odim_size: Dim = if (i + 1 > zt.tensor.shape.ndim(odims)) 1 else odims[i];
        count *= @divTrunc(ctx.in_dims[i], odim_size);
    }
    var tmp = try detail.expandFromReduction(allocator, Tensor, grad_output.tensor(), mean_ctx.axes, mean_ctx.keep_dims);
    defer tmp.deinit();
    var grad = try detail.tileAs(allocator, tmp, mean_ctx.in_dims);
    try grad.inPlaceDiv(allocator, Dim, count);
    var tmp_var = try Variable.init(allocator, grad, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn mean(allocator: std.mem.Allocator, input: *Variable, axes: []const Dim, keep_dims: bool) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var result = try zt.tensor.mean(allocator, adj.res, axes, keep_dims);
    var indims = try input.shape(allocator);
    var ctx = try allocator.create(ReductionCtx);
    ctx.* = .{ .in_dims = try allocator.alloc(Dim, indims.len), .axes = try allocator.alloc(i64, axes.len), .keep_dims = keep_dims };
    @memcpy(ctx.in_dims, indims);
    @memcpy(ctx.axes, axes);
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, meanGradFunc, ctx, freeSumCtx);
}

const VarCtx = struct {
    val: f64,
    axes: []const Dim,
};

fn freeVarCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var var_ctx: *VarCtx = @ptrCast(@alignCast(ctx));
    allocator.free(var_ctx.axes);
    allocator.destroy(var_ctx);
}

fn varianceGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var var_ctx: *VarCtx = @ptrCast(@alignCast(ctx));
    var expanded_dims = try allocator.alloc(Dim, try inputs[0].shape(allocator));
    defer allocator.free(expanded_dims);
    @memcpy(expanded_dims, try inputs[0].shape(allocator));
    var tile_dims = try allocator.alloc(Dim, try inputs[0].shape(allocator));
    defer allocator.free(tile_dims);
    @memcpy(tile_dims, try inputs[0].shape(allocator));
    for (var_ctx.axes) |ax| {
        tile_dims[@intCast(ax)] = try inputs[0].dim(allocator, @intCast(ax));
        expanded_dims[@intCast(ax)] = 1;
    }

    var tmp1 = try moddims(allocator, grad_output, expanded_dims);
    defer tmp1.deinit();
    var tmp2 = try tileAs(allocator, tmp1, Shape, tile_dims);
    defer tmp2.deinit();
    var lhs = try mul(allocator, f64, 2 * var_ctx.val, *Variable, tmp2);
    defer lhs.deinit();
    var tmp3 = try mean(allocator, inputs[0], var_ctx.axes, false);
    defer tmp3.deinit();
    var tmp4 = try moddims(allocator, tmp3, expanded_dims);
    defer tmp4.deinit();
    var tmp5 = try tileAs(allocator, tmp4, Shape, tile_dims);
    defer tmp5.deinit();
    var rhs = try sub(allocator, *Variable, inputs[0], *Variable, tmp5);
    defer rhs.deinit();
    var res = try mul(allocator, *Variable, lhs, *Variable, rhs);
    defer res.deinit();
    var tmp_var = try Variable.initSharedData(allocator, res.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn variance(allocator: std.mem.Allocator, in: *Variable, axes: []const Dim, is_biased: bool, keep_dims: bool) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, in.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var input = try zt.tensor.mul(allocator, Tensor, adj.res, Tensor, adj.res);
    defer input.deinit();
    var result = try zt.tensor.sum(allocator, input, axes, keep_dims);

    var avg = try zt.tensor.mean(allocator, input, axes, keep_dims);
    defer avg.deinit();
    var n: Dim = 1;
    for (axes) |ax| {
        n *= try input.dim(allocator, @intCast(ax));
    }
    if (!is_biased and n == 1) {
        std.debug.print("autograd.variance: cannot compute unbiased variance with only one sample\n", .{});
        return error.VarianceFailedOneSample;
    }

    var val: f64 = 1 / (if (is_biased) @as(f64, @floatFromInt(n)) else @as(f64, @floatFromInt(n)) - 1);
    try result.inPlaceSub(allocator, Dim, n);
    try result.inPlaceMul(allocator, Tensor, avg);
    try result.inPlaceMul(allocator, Tensor, avg);
    try result.inPlaceMul(allocator, f64, val);
    var ctx = try allocator.create(VarCtx);
    ctx.* = .{ .val = val, .axes = try allocator.alloc(Dim, axes.len) };
    @memcpy(ctx.axes, axes);
    return Variable.initWithInputs(allocator, result, &.{try in.withoutData(allocator)}, varianceGradFunc, ctx, freeVarCtx);
}

const NormCtx = struct {
    sumap: Tensor,
    p: f64,
    axes: []const Dim,
    keep_dims: bool,
};

fn freeNormCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var norm_ctx: *NormCtx = @ptrCast(@alignCast(ctx));
    norm_ctx.sumap.deinit();
    allocator.free(norm_ctx.axes);
    allocator.destroy(norm_ctx);
}

fn normGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var norm_ctx: *NormCtx = @ptrCast(@alignCast(ctx));
    var gvar = try Variable.init(allocator, try zt.tensor.power(allocator, Tensor, norm_ctx.sumap, f64, 1 - 1 / norm_ctx.p), false);
    defer gvar.deinit();
    var tmp1 = try abs(allocator, inputs[0]);
    defer tmp1.deinit();
    var tmp2 = try pow(allocator, tmp1, norm_ctx.p - 2);
    defer tmp2.deinit();
    var lhs = try zt.tensor.mul(allocator, Tensor, inputs[0].tensor(), Tensor, tmp2.tensor());
    defer lhs.deinit();
    var tmp3 = try detail.expandFromReduction(allocator, Tensor, grad_output.tensor(), norm_ctx.axes, norm_ctx.keep_dims);
    defer tmp3.deinit();
    try tmp3.inPlaceDiv(allocator, Tensor, gvar.tensor());
    var rhs = try detail.tileAs(allocator, tmp3, try inputs[0].shape(allocator));
    defer rhs.deinit();
    var tmp_var = try Variable.init(allocator, try zt.tensor.mul(allocator, Tensor, lhs, Tensor, rhs), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn norm(allocator: std.mem.Allocator, input: *Variable, axes: []const Dim, p: f64, keep_dims: bool) !*Variable {
    if (p <= 0) {
        std.debug.print("Lp norm: p must be > 0\n", .{});
        return error.InvalidNormP;
    }
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var tmp1 = try zt.tensor.abs(allocator, adj.res);
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.power(allocator, Tensor, tmp1, f64, p);
    defer tmp2.deinit();
    var tmp4 = try zt.tensor.sum(allocator, tmp2, axes, keep_dims);
    defer tmp4.deinit();

    var sumap = try detail.expandFromReduction(allocator, Tensor, tmp4, axes, keep_dims);
    var result = try zt.tensor.power(allocator, Tensor, tmp4, f64, 1 / p);
    try zt.tensor.eval(allocator, result);
    var ctx = try allocator.create(NormCtx);
    ctx.* = .{ .sumap = sumap, .p = p, .axes = try allocator.alloc(Dim, axes.len), .keep_dims = keep_dims };
    @memcpy(ctx.axes, axes);
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, normGradFunc, ctx, freeNormCtx);
}

pub fn normalize(allocator: std.mem.Allocator, in: *Variable, axes: []const Dim, p: f64, eps: f64) !*Variable {
    var adj = try detail.adjustInputType(allocator, *Variable, in, @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var tmp_norm = try norm(allocator, adj.res, axes, p);
    defer tmp_norm.deinit();
    var inv_scale = try max(allocator, *Variable, tmp_norm, f64, eps);
    defer inv_scale.deinit();
    var tmp = try tileAs(allocator, inv_scale, *Variable, adj.res);
    defer tmp.deinit();
    return div(allocator, *Variable, adj.res, *Variable, tmp);
}

fn matmulGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    if (inputs[0].isCalcGrad()) {
        var _lhs = grad_output.tensor();
        if (try _lhs.ndim(allocator) == 1) {
            _lhs = try zt.tensor.reshape(allocator, _lhs, &.{ 1, try _lhs.dim(allocator, 0) });
        }
        var _rhs = inputs[1].tensor();
        if (try _rhs.ndim(allocator) == 1) {
            _rhs = try zt.tensor.reshape(allocator, _rhs, &.{ try _rhs.dim(allocator, 0), 1 });
        }

        // matmulNT(gradOutput, inputs[1])
        // -- matmulNT([M, K], [N, K])
        // -- matmul([M, K], [K, N]) -- [M, K]
        var val = try zt.tensor.matmul(allocator, _lhs, _rhs, .None, .Transpose);
        defer val.deinit();
        var tmp = try Variable.init(allocator, try detail.sumAs(allocator, val, try inputs[0].shape(allocator)), false);
        defer tmp.deinit();
        try inputs[0].addGrad(allocator, tmp);
    }
    if (inputs[1].isCalcGrad()) {
        var _lhs = inputs[0].tensor();
        if (try _lhs.ndim(allocator) == 1) {
            _lhs = try zt.tensor.reshape(allocator, _lhs, &.{ 1, try _lhs.dim(allocator, 0) });
        }
        var _rhs = grad_output.tensor();
        if (try _rhs.ndim(allocator) == 1) {
            _rhs = try zt.tensor.reshape(allocator, _rhs, &.{ try _rhs.dim(allocator, 0), 1 });
        }

        // matmulTN(inputs[0], gradOutput)
        // -- matmulTN([M, N], [M, K])
        // -- matmul([N, M], [M, K]) -- [N, K]
        var val = try zt.tensor.matmul(allocator, _lhs, _rhs, .Transpose, .None);
        defer val.deinit();
        var tmp = try Variable.init(allocator, try detail.sumAs(allocator, val, try inputs[1].shape(allocator)), false);
        defer tmp.deinit();
        try inputs[1].addGrad(allocator, tmp);
    }
}

pub fn matmul(allocator: std.mem.Allocator, lhs: *Variable, rhs: *Variable) !*Variable {
    try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
    // lhs:Input[0] -- [M, N]
    // rhs:Input[1] -- [N, K]
    // matmul(lhs, rhs)
    // -- matmul([M, N], [N, K]) --  [M, K]
    // result:gradOutput -- [M, K]
    var result = try zt.tensor.matmul(allocator, lhs.tensor(), rhs.tensor(), .None, .None);
    return Variable.initWithInputs(allocator, result, &.{ try lhs.clone(allocator), try rhs.clone(allocator) }, matmulGradFunc, null, null);
}

fn matmulTNGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    if (inputs[0].isCalcGrad()) {
        // matmulNT(inputs[1], gradOutput)
        // -- matmulNT([N, K], [M, K])
        // -- matmul([N, K], [K, M]) -- [N, M]
        var val = try zt.tensor.matmul(allocator, inputs[1].tensor(), grad_output.tensor(), .None, .Transpose);
        defer val.deinit();
        var tmp = try Variable.init(allocator, try detail.sumAs(allocator, val, try inputs[0].shape(allocator)), false);
        defer tmp.deinit();
        try inputs[0].addGrad(allocator, tmp);
    }
    if (inputs[1].isCalcGrad()) {
        // matmul(inputs[0], gradOutput)
        // -- matmulNT([N, M], [M, K]) -- [N, K]
        var val = try zt.tensor.matmul(allocator, inputs[0].tensor(), grad_output.tensor(), .None, .None);
        defer val.deinit();
        var tmp = try Variable.init(allocator, try detail.sumAs(allocator, val, try inputs[1].shape(allocator)), false);
        defer tmp.deinit();
        try inputs[1].addGrad(allocator, tmp);
    }
}

pub fn matmulTN(allocator: std.mem.Allocator, lhs: *Variable, rhs: *Variable) !*Variable {
    try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
    // lhs:Input[0] -- [N, M]
    // rhs:Input[1] -- [N, K]
    // matmulTN(lhs, rhs)
    // -- matmulTN([N, M], [N, K])
    // -- matmul([M, N], [N, K]) -- [M, K]
    // result:gradOutput -- [M, K]
    var result = try zt.tensor.matmul(allocator, lhs.tensor(), rhs.tensor(), .Transpose, .None);
    return Variable.initWithInputs(allocator, result, &.{ try lhs.clone(allocator), try rhs.clone(allocator) }, matmulTNGradFunc, null, null);
}

fn matmulNTGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    if (inputs[0].isCalcGrad()) {
        // matmul(gradOutput, inputs[1])
        // -- matmul([M, K], [K, N]) -- [M, N]
        var val = try zt.tensor.matmul(allocator, grad_output.tensor(), inputs[1].tensor(), .None, .None);
        defer val.deinit();
        var tmp = try Variable.init(allocator, try detail.sumAs(allocator, val, try inputs[0].shape(allocator)), false);
        defer tmp.deinit();
        try inputs[0].addGrad(allocator, tmp);
    }
    if (inputs[1].isCalcGrad()) {
        // matmulTN(gradOutput, inputs[0])
        // -- matmulTN([M, K], [M, N])
        // -- matmul([K, M], [M, N]) -- [K, N]
        var val = try zt.tensor.matmul(allocator, grad_output.tensor(), inputs[0].tensor(), .Transpose, .None);
        defer val.deinit();
        var tmp = try Variable.init(allocator, try detail.sumAs(allocator, val, try inputs[1].shape(allocator)), false);
        defer tmp.deinit();
        try inputs[1].addGrad(allocator, tmp);
    }
}

pub fn matmulNT(allocator: std.mem.Allocator, lhs: *Variable, rhs: *Variable) !*Variable {
    try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
    // lhs:Input[0] -- [M, N]
    // rhs:Input[1] -- [K, N]
    // matmulNT(lhs, rhs)
    // -- matmulNT([M, N], [K, N])
    // -- matmul([M, N], [N, K]) -- [M, K]
    // result:gradOutput -- [M, K]
    var result = try zt.tensor.matmul(allocator, lhs.tensor(), rhs.tensor(), .None, .Transpose);
    return Variable.initWithInputs(allocator, result, &.{ try lhs.clone(allocator), try rhs.clone(allocator) }, matmulNTGradFunc, null, null);
}

fn absGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    // Convert it into -1, 0, 1
    var sign = try zt.tensor.sign(allocator, inputs[0].tensor());
    try sign.inPlaceMul(allocator, Tensor, grad_output.tensor());
    var tmp_var = try Variable.init(allocator, sign, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn abs(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.abs(allocator, input.tensor());
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, absGradFunc, null, null);
}

fn flatGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var flat_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
    var tmp = try Variable.init(allocator, try zt.tensor.reshape(allocator, grad_output.tensor(), flat_ctx.dims), false);
    defer tmp.deinit();
    try inputs[0].addGrad(allocator, tmp);
}

pub fn flat(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try input.tensor().flatten(allocator);
    var idims = try input.shape(allocator);
    var ctx = try allocator.create(DimsCtx);
    ctx.* = .{ .dims = try allocator.alloc(Dim, idims.len) };
    @memcpy(ctx.dims, idims);
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, flatGradFunc, ctx, freeDimsCtx);
}

fn moddimsGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var moddims_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
    var tmp_mod = try moddims(allocator, grad_output, moddims_ctx.dims);
    defer tmp_mod.deinit();
    var tmp_var = try Variable.initSharedData(allocator, tmp_mod.shared_data.retain(), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn moddims(allocator: std.mem.Allocator, input: *Variable, dims: Shape) !*Variable {
    if (try input.ndim(allocator) == 0) {
        return input.clone(allocator);
    }
    var infer_dims = try allocator.alloc(Dim, dims.len);
    defer allocator.free(infer_dims);
    @memcpy(infer_dims, dims);
    var max_ndims: usize = @max(try input.ndim(allocator), zt.tensor.shape.ndim(dims));

    // Check for inferred dims that are beyond the input's number of dims
    for (0..max_ndims) |i| {
        if (i >= try input.ndim(allocator) and infer_dims[i] == 0) {
            std.debug.print("autograd.moddims: tried to infer dimension {d} which exceeds the number of dimensions of the input.\n", .{i});
            return error.InferredDimsExceedNumDims;
        }
    }

    // Infer any 0 dim
    for (0..max_ndims) |i| {
        if (i < zt.tensor.shape.ndim(infer_dims) and infer_dims[i] == 0) {
            infer_dims[i] = try input.dim(allocator, i);
        }
    }

    // Infer any -1 dim
    var n_infer: usize = 0;
    for (0..max_ndims) |i| {
        if (i < zt.tensor.shape.ndim(infer_dims) and infer_dims[i] == -1) {
            n_infer += 1;
            infer_dims[i] = -(@divTrunc(try input.elements(allocator), zt.tensor.shape.elements(infer_dims)));
        }
    }

    if (n_infer > 1) {
        std.debug.print("autograd.moddims: too many dimensions infer\n", .{});
        return error.TooManyDimsInferred;
    }

    if (zt.tensor.shape.elements(infer_dims) != try input.elements(allocator)) {
        std.debug.print("autograd.moddims: mismatched # of elements\n", .{});
        return error.MismatchedNumElements;
    }

    var result = try zt.tensor.reshape(allocator, input.tensor(), infer_dims);
    var in_dims = try input.shape(allocator);
    var ctx = try allocator.create(DimsCtx);
    ctx.* = .{ .dims = try allocator.alloc(Dim, in_dims.len) };
    @memcpy(ctx.dims, in_dims);
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, moddimsGradFunc, ctx, freeDimsCtx);
}

const SoftmaxCtx = struct {
    dim: Dim,
    tile_dims: []const Dim,
    result: Tensor,
};

fn freeSoftmaxCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var softmax_ctx: *SoftmaxCtx = @ptrCast(@alignCast(ctx));
    allocator.free(softmax_ctx.tile_dims);
    allocator.destroy(softmax_ctx);
}

fn softmaxGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var softmax_ctx: *SoftmaxCtx = @ptrCast(@alignCast(ctx));
    var rbyg = try zt.tensor.mul(allocator, Tensor, grad_output.tensor(), Tensor, softmax_ctx.result);
    defer rbyg.deinit();
    var tmp1 = try zt.tensor.sum(allocator, rbyg, &.{softmax_ctx.dim}, true);
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.tile(allocator, tmp1, softmax_ctx.tile_dims);
    defer tmp2.deinit();
    var tmp3 = try zt.tensor.mul(allocator, Tensor, softmax_ctx.result, Tensor, tmp2);
    defer tmp3.deinit();
    var grad_sm = try zt.tensor.sub(allocator, Tensor, rbyg, Tensor, tmp3);
    defer grad_sm.deinit();
    var tmp_var = try Variable.init(allocator, try grad_sm.astype(allocator, try inputs[0].dtype(allocator)), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn softmax(allocator: std.mem.Allocator, input: *Variable, dim: Dim) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var maxvals = try zt.tensor.amax(allocator, adj.res, &.{dim}, true);
    defer maxvals.deinit();
    var tile_dims = try allocator.alloc(Dim, try input.ndim(allocator));
    @memset(tile_dims, 1);
    tile_dims[@intCast(dim)] = try input.dim(allocator, @intCast(dim));

    var tmp1 = try zt.tensor.tile(allocator, maxvals, tile_dims);
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.sub(allocator, Tensor, adj.res, Tensor, tmp1);
    defer tmp2.deinit();
    var exp_input = try zt.tensor.exp(allocator, tmp2);
    defer exp_input.deinit();
    var tmp3 = try zt.tensor.sum(allocator, exp_input, &.{dim}, true);
    defer tmp3.deinit();
    var tmp4 = try zt.tensor.tile(allocator, tmp3, tile_dims);
    defer tmp4.deinit();
    var result = try zt.tensor.div(allocator, Tensor, exp_input, Tensor, tmp4);
    try zt.tensor.eval(allocator, result);
    var ctx = try allocator.create(SoftmaxCtx);
    ctx.* = .{ .dim = dim, .tile_dims = tile_dims, .result = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, softmaxGradFunc, ctx, freeSoftmaxCtx);
}

fn logSoftmaxGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var softmax_ctx: *SoftmaxCtx = @ptrCast(@alignCast(ctx));
    var tmp1 = try zt.tensor.sum(allocator, grad_output.tensor(), &.{softmax_ctx.dim}, true);
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.tile(allocator, tmp1, softmax_ctx.tile_dims);
    defer tmp2.deinit();
    var tmp3 = try zt.tensor.exp(allocator, softmax_ctx.result);
    defer tmp3.deinit();
    try tmp3.inPlaceMul(allocator, Tensor, tmp2);
    var tmp_var = try Variable.init(allocator, try zt.tensor.sub(allocator, Tensor, grad_output.tensor(), Tensor, tmp3), false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn logSoftmax(allocator: std.mem.Allocator, input: *Variable, dim: Dim) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var maxvals = try zt.tensor.amax(allocator, adj.res, &.{dim}, true);
    defer maxvals.deinit();
    var tile_dims = try allocator.alloc(Dim, try input.ndim(allocator));
    @memset(tile_dims, 1);
    tile_dims[@intCast(dim)] = try input.dim(allocator, @intCast(dim));

    var tmp1 = try zt.tensor.tile(allocator, maxvals, tile_dims);
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.sub(allocator, Tensor, adj.res, Tensor, tmp1);
    defer tmp2.deinit();
    var tmp3 = try zt.tensor.exp(allocator, tmp2);
    defer tmp3.deinit();
    var tmp4 = try zt.tensor.sum(allocator, tmp3, &.{dim}, true);
    defer tmp4.deinit();
    var tmp5 = try zt.tensor.log(allocator, tmp4);
    defer tmp5.deinit();
    try tmp5.inPlaceAdd(allocator, Tensor, maxvals);
    var tmp6 = try zt.tensor.tile(allocator, tmp5, tile_dims);
    defer tmp6.deinit();
    var result = try zt.tensor.sub(allocator, Tensor, adj.res, Tensor, tmp6);
    try zt.tensor.eval(allocator, result);
    var ctx = try allocator.create(SoftmaxCtx);
    ctx.* = .{ .dim = dim, .tile_dims = tile_dims, .result = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, logSoftmaxGradFunc, ctx, freeSoftmaxCtx);
}

pub fn binaryCrossEntropy(allocator: std.mem.Allocator, inputs: *Variable, targets: *Variable) !*Variable {
    var targets_typed = try targets.astype(allocator, try inputs.dtype(allocator));
    defer targets_typed.deinit();
    var tmp1 = try sub(allocator, f64, 1, *Variable, targets_typed);
    defer tmp1.deinit();
    var tmp2 = try log(allocator, inputs);
    defer tmp2.deinit();
    var tmp3 = try mul(allocator, *Variable, targets_typed, *Variable, tmp2);
    defer tmp3.deinit();
    var tmp4 = try sub(allocator, f64, 1, *Variable, inputs);
    defer tmp4.deinit();
    var tmp5 = try log(allocator, tmp4);
    defer tmp5.deinit();
    var tmp6 = try mul(allocator, *Variable, tmp1, *Variable, tmp5);
    defer tmp6.deinit();
    var tmp7 = try add(allocator, *Variable, tmp3, *Variable, tmp6);
    defer tmp7.deinit();
    return negate(allocator, tmp7);
}

const CatCrossEntropyCtx = struct {
    C: Dim,
    X: Dim,
    mask: Tensor,
    ignore_mask: Tensor,
    denominator: Tensor = undefined,
    reduction: zt.common.ReduceMode,
    input_dims: []Dim,
};

fn freeCatCrossEntropyCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var cat_ctx: *CatCrossEntropyCtx = @ptrCast(@alignCast(ctx));
    cat_ctx.mask.deinit();
    cat_ctx.ignore_mask.deinit();
    if (cat_ctx.reduction == .Mean) {
        cat_ctx.denominator.deinit();
    }
    allocator.free(cat_ctx.input_dims);
    allocator.destroy(cat_ctx);
}

fn categoricalCrossEntropyGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var cat_ctx: *CatCrossEntropyCtx = @ptrCast(@alignCast(ctx));
    var tmp_grad1: Tensor = undefined;
    switch (cat_ctx.reduction) {
        .None => tmp_grad1 = try zt.tensor.reshape(allocator, grad_output.tensor(), &.{cat_ctx.X}),
        .Mean => {
            var tmp = try zt.tensor.div(allocator, Tensor, grad_output.tensor(), Tensor, cat_ctx.denominator);
            defer tmp.deinit();
            tmp_grad1 = try zt.tensor.tile(allocator, tmp, &.{cat_ctx.X});
        },
        .Sum => tmp_grad1 = try zt.tensor.tile(allocator, grad_output.tensor(), &.{cat_ctx.X}),
    }
    // [1 X]
    try tmp_grad1.indexAssign(allocator, f64, 0, &.{Index.initTensor(cat_ctx.ignore_mask)});
    var tmp_grad2 = try zt.tensor.reshape(allocator, tmp_grad1, &.{ 1, cat_ctx.X });
    tmp_grad1.deinit();
    var grad = try zt.tensor.tile(allocator, tmp_grad2, &.{cat_ctx.C});
    tmp_grad2.deinit();
    try grad.inPlaceMul(allocator, Tensor, cat_ctx.mask);
    var tmp_var = try Variable.init(allocator, try zt.tensor.reshape(allocator, grad, cat_ctx.input_dims), false);
    defer tmp_var.deinit();
    grad.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn categoricalCrossEntropy(allocator: std.mem.Allocator, in: *Variable, targets: *Variable, reduction: zt.common.ReduceMode, ignore_idx: i64) !*Variable {
    var adj = try detail.adjustInputType(allocator, *Variable, in, @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var input: *Variable = adj.res;
    // input -- [C, X1, X2, X3]
    // target -- [X1, X2, X3, 1]
    if (try input.ndim(allocator) != try targets.ndim(allocator) + 1) {
        std.debug.print("dimension mismatch in categorical cross entropy: target must have one fewer dimension than input\n", .{});
        return error.CatCrossEntropyDimsMismatch;
    }
    for (1..try input.ndim(allocator)) |i| {
        if (try input.dim(allocator, i) != try targets.dim(allocator, i - 1)) {
            std.debug.print("dimension mismatch in categorical cross entropy\n", .{});
            return error.CatCrossEntropyDimsMismatch;
        }
    }

    const C = try input.dim(allocator, 0);
    const X = try targets.elements(allocator);
    var cond1 = try zt.tensor.lessThan(allocator, Tensor, targets.tensor(), f64, 0);
    var cond2 = try zt.tensor.greaterThanEqual(allocator, Tensor, targets.tensor(), i64, C);
    var or_cond = try zt.tensor.logicalOr(allocator, Tensor, cond1, Tensor, cond2);
    defer or_cond.deinit();
    cond1.deinit();
    cond2.deinit();
    var cond3 = try zt.tensor.neq(allocator, Tensor, targets.tensor(), i64, ignore_idx);
    var and_cond = try zt.tensor.logicalAnd(allocator, Tensor, or_cond, Tensor, cond3);
    cond3.deinit();
    var cond_check = try zt.tensor.all(allocator, and_cond, &.{}, false);
    and_cond.deinit();
    if (try cond_check.scalar(allocator, i8) != 0) {
        std.debug.print("target contains elements out of valid range [0, num_categories) in categorical cross entropy\n", .{});
        return error.CatCrossEntropyInvalidTargetElement;
    }
    cond_check.deinit();

    var x = try zt.tensor.reshape(allocator, input.tensor(), &.{ C, X });
    var y = try zt.tensor.reshape(allocator, targets.tensor(), &.{ 1, X });

    var A = try zt.tensor.arange(allocator, &.{ C, X }, 0, .f32);
    var B = try zt.tensor.tile(allocator, y, &.{C});
    var mask = try zt.tensor.eq(allocator, Tensor, A, Tensor, B);
    A.deinit();
    B.deinit();
    try mask.inPlaceMul(allocator, f64, -1); // [C X]

    var tmp_res1 = try zt.tensor.mul(allocator, Tensor, mask, Tensor, x);
    x.deinit();
    var tmp_ignore_mask = try zt.tensor.eq(allocator, Tensor, y, i64, ignore_idx);
    y.deinit();
    var ignore_mask = try tmp_ignore_mask.flatten(allocator);
    tmp_ignore_mask.deinit();
    var tmp_res2 = try zt.tensor.sum(allocator, tmp_res1, &.{0}, false);
    tmp_res1.deinit();
    var tmp_res3 = try tmp_res2.flatten(allocator); // [X, 1]
    defer tmp_res3.deinit();
    tmp_res2.deinit();
    try tmp_res3.indexAssign(allocator, f64, 0, &.{Index.initTensor(ignore_mask)});

    var result: Tensor = undefined;
    var denominator: Tensor = undefined;
    switch (reduction) {
        .None => result = try zt.tensor.reshape(allocator, tmp_res3, try targets.shape(allocator)), // [X1 X2 X3]
        .Mean => {
            var tmp_ignore_mask1 = try zt.tensor.logicalNot(allocator, ignore_mask);
            var tmp_ignore_mask2 = try tmp_ignore_mask1.astype(allocator, .s32);
            tmp_ignore_mask1.deinit();
            denominator = try zt.tensor.sum(allocator, tmp_ignore_mask2, &.{0}, false);
            tmp_ignore_mask2.deinit();
            result = try zt.tensor.sum(allocator, tmp_res3, &.{0}, false);
            try result.inPlaceDiv(allocator, Tensor, denominator); // [1]
        },
        .Sum => result = try zt.tensor.sum(allocator, tmp_res3, &.{0}, false), // [1]
    }

    var input_dims = try input.shape(allocator);
    var ctx = try allocator.create(CatCrossEntropyCtx);
    ctx.* = .{ .X = X, .C = C, .mask = mask, .ignore_mask = ignore_mask, .reduction = reduction, .input_dims = try allocator.alloc(Dim, input_dims.len) };
    @memcpy(ctx.input_dims, input_dims);
    if (reduction == .Mean) ctx.denominator = denominator;

    return Variable.initWithInputs(allocator, result, &.{ try input.withoutData(allocator), try targets.clone(allocator) }, categoricalCrossEntropyGradFunc, ctx, freeCatCrossEntropyCtx);
}

const WeightCatCrossEntropyCtx = struct {
    C: Dim,
    X: Dim,
    mask: Tensor,
    ignore_mask: Tensor,
    denominator: *Variable,
    input_dims: Shape,
};

fn freeWeightedCatCrossEntropyCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var cat_ctx: *WeightCatCrossEntropyCtx = @ptrCast(@alignCast(ctx));
    cat_ctx.mask.deinit();
    cat_ctx.ignore_mask.deinit();
    cat_ctx.denominator.deinit();
    allocator.free(cat_ctx.input_dims);
    allocator.destroy(cat_ctx);
}

fn weightedCategoricalCrossEntropyGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var cat_ctx: *WeightCatCrossEntropyCtx = @ptrCast(@alignCast(ctx));
    var tmp_grad1 = zt.tensor.div(allocator, Tensor, grad_output.tensor(), Tensor, cat_ctx.denominator.tensor());
    var tmp_grad2 = try zt.tensor.tile(allocator, tmp_grad1, &.{ 1, cat_ctx.X });
    tmp_grad1.deinit();

    var weight_tensor = cat_ctx.weight.tensor();
    try tmp_grad2.inPlaceMul(allocator, Tensor, cat_ctx.ignore_mask);
    var tmp_grad3 = try zt.tensor.tile(allocator, tmp_grad2, &.{cat_ctx.C});
    tmp_grad2.deinit();
    try tmp_grad3.inPlaceMul(allocator, Tensor, cat_ctx.mask);
    var tmp_grad4 = try zt.tensor.reshape(allocator, tmp_grad3, cat_ctx.input_dims);
    tmp_grad3.deinit();
    tmp_grad4.inPlaceMul(allocator, Tensor, weight_tensor);
    var tmp_var = try Variable.init(allocator, tmp_grad4, false);
    defer tmp_var.deinit();
    try inputs[0].addGrad(allocator, tmp_var);
}

pub fn weightedCategoricalCrossEntropy(allocator: std.mem.Allocator, input: *Variable, targets: *Variable, weight: *Variable, ignore_idx: i64) !*Variable {
    // input -- [C, X1, X2, X3]
    // target -- [X1, X2, X3]
    if (try input.ndim(allocator) < try targets.ndim(allocator) - 1) {
        std.debug.print("{s}: input must have one more than the number of target dimensions minus 1\n", .{@src().fn_name});
        return error.WeightedCatCrossEntropyDimsMismatch;
    }

    for (1..try targets.ndim(allocator) - 2) |i| {
        if (try input.dim(allocator, i) != targets.dim(allocator, i - 1)) {
            std.debug.print("{s}: dimension mismatch in categorical cross entropy\n", .{@src().fn_name});
            return error.WeightedCatCrossEntropyDimsMismatch;
        }
    }
    if (try weight.dim(allocator, 0) != try input.dim(allocator, 0)) {
        std.debug.print("{s}: dimension mismatch in categorical cross entropy\n", .{@src().fn_name});
        return error.WeightedCatCrossEntropyDimsMismatch;
    }

    const C = try input.dim(allocator, 0);
    const X = try targets.elements(allocator);
    var cond1 = try zt.tensor.lessThan(allocator, Tensor, targets.tensor(), f64, 0);
    var cond2 = try zt.tensor.greaterThanEqual(allocator, Tensor, targets.tensor(), i64, C);
    var or_cond = try zt.tensor.logicalOr(allocator, Tensor, cond1, Tensor, cond2);
    cond1.deinit();
    cond2.deinit();
    var cond_check = try zt.tensor.any(or_cond, &.{}, false);
    or_cond.deinit();
    if (try cond_check.scalar(allocator, i8) != 0) {
        std.debug.print("{s}: target contains elements out of valid range [0, num_categories) in categorical cross entropy\n", .{@src().fn_name});
        return error.WeightedCatCrossEntropyInvalidTargetElement;
    }
    cond_check.deinit();

    var x = try zt.tensor.reshape(allocator, input.tensor(), &.{ C, X });
    var y = try zt.tensor.reshape(allocator, targets.tensor(), &.{ 1, X });

    var A = try zt.tensor.arange(allocator, &.{ C, X }, 0, .f32);
    var B = try zt.tensor.tile(allocator, y, &.{C});
    var mask = try zt.tensor.eq(allocator, Tensor, A, Tensor, B);
    A.deinit();
    B.deinit();
    try mask.inPlaceMul(allocator, f64, -1); // [C X]

    var weight_sum = try zt.tensor.tile(allocator, weight.tensor(), &.{ 1, X });
    try weight_sum.inPlaceMul(allocator, Tensor, mask);
    try weight_sum.inPlaceMul(allocator, f64, -1);
    var denominator = try Variable.init(allocator, try zt.tensor.sum(allocator, weight_sum, &.{ 0, 1 }, false), false);
    weight_sum.deinit();

    var tmp_res1 = try zt.tensor.mul(allocator, Tensor, mask, Tensor, x);
    x.deinit();
    var tmp_ignore_mask = try zt.tensor.neq(allocator, Tensor, y, i64, ignore_idx);
    y.deinit();
    var ignore_mask = try tmp_ignore_mask.astype(allocator, .s32); // [1, X]
    tmp_ignore_mask.deinit();
    var tmp_res2 = try zt.tensor.sum(allocator, tmp_res1, &.{0}, true); // [1, X]
    tmp_res1.deinit();
    try tmp_res2.inPlaceMul(allocator, Tensor, ignore_mask);
    var result = try zt.tensor.sum(allocator, tmp_res2, &.{1}, true);
    tmp_res2.deinit();

    var input_dims = try input.shape(allocator);
    var ctx = try allocator.create(WeightCatCrossEntropyCtx);
    ctx.* = .{ .C = C, .X = X, .mask = mask, .ignore_mask = ignore_mask, .denominator = denominator, .input_dims = try allocator.alloc(Dim, input_dims.len) };
    @memcpy(ctx.input_dims, input_dims);
    return Variable.initWithInputs(allocator, result, &.{ try input.withoutData(allocator), try targets.clone(allocator), try weight.clone(allocator) }, weightedCategoricalCrossEntropyGradFunc, ctx, freeWeightedCatCrossEntropyCtx);
}

// Unit tests
const JacobianFunc = *const fn (allocator: std.mem.Allocator, input: *Variable, ctx: ?*anyopaque) anyerror!*Variable;

inline fn jacobianTestImpl(
    allocator: std.mem.Allocator,
    func: JacobianFunc,
    input: *Variable,
    ctx: ?*anyopaque,
    precision: f32,
    perturbation: f32,
) !bool {
    var fwd_var = try func(allocator, input, ctx);
    defer fwd_var.deinit();
    var fwd_jacobian = try Tensor.initHandle(allocator, &.{ try fwd_var.elements(allocator), try input.elements(allocator) }, .f32);
    defer fwd_jacobian.deinit();
    for (0..@intCast(try input.elements(allocator))) |i| {
        var tmp_flat = try input.tensor().flatten(allocator);
        defer tmp_flat.deinit();
        var orig = try tmp_flat.index(allocator, &.{Index.initDim(@intCast(i))});
        defer orig.deinit();
        var assign1 = try zt.tensor.sub(allocator, Tensor, orig, f32, perturbation);
        defer assign1.deinit();
        try input.tensor().flatAssign(allocator, Tensor, assign1, Index.initDim(@intCast(i)));
        var outa_var: *Variable = try func(allocator, input, ctx);
        defer outa_var.deinit();
        var outa = outa_var.tensor();
        _ = try outa.getAdapter(zt.tensor.DefaultTensorType_t).getHandle(allocator); // tensor was a view, run indexing and promote

        var assign2 = try zt.tensor.add(allocator, Tensor, orig, f32, perturbation);
        defer assign2.deinit();
        try input.tensor().flatAssign(allocator, Tensor, assign2, Index.initDim(@intCast(i)));
        var outb_var: *Variable = try func(allocator, input, ctx);
        defer outb_var.deinit();
        var outb = outb_var.tensor();
        _ = try outb.getAdapter(zt.tensor.DefaultTensorType_t).getHandle(allocator); // tensor was a view, run indexing and promote
        try input.tensor().flatAssign(allocator, Tensor, orig, Index.initDim(@intCast(i)));

        var tmp = try zt.tensor.sub(allocator, Tensor, outb, Tensor, outa);
        defer tmp.deinit();
        var assign3 = try zt.tensor.reshape(allocator, tmp, &.{try outa.elements(allocator)});
        defer assign3.deinit();
        try assign3.inPlaceMul(allocator, f64, 0.5);
        try assign3.inPlaceDiv(allocator, f32, perturbation);
        try fwd_jacobian.indexAssign(allocator, Tensor, assign3, &.{ Index.initRange(zt.tensor.span), Index.initDim(@intCast(i)) });
    }
    var bwd_var = try func(allocator, input, ctx);
    defer bwd_var.deinit();
    var bwd_jacobian = try Tensor.initHandle(allocator, &.{ try bwd_var.elements(allocator), try input.elements(allocator) }, .f32);
    defer bwd_jacobian.deinit();
    var dout = try Variable.init(allocator, try zt.tensor.full(allocator, try bwd_var.shape(allocator), f64, 0, try bwd_var.dtype(allocator)), false);
    defer dout.deinit();

    for (0..@intCast(try dout.elements(allocator))) |i| {
        try dout.tensor().flatAssign(allocator, f64, 1, Index.initDim(@intCast(i))); // element in 1D view
        input.zeroGrad();
        var out: *Variable = try func(allocator, input, ctx);
        defer out.deinit();
        try out.backwardWithGrad(allocator, dout, false);

        var assign = try zt.tensor.reshape(allocator, (try input.grad()).tensor(), &.{try input.elements(allocator)});
        defer assign.deinit();
        try bwd_jacobian.indexAssign(allocator, Tensor, assign, &.{Index.initDim(@intCast(i))});
        try dout.tensor().flatAssign(allocator, f64, 0, Index.initDim(@intCast(i)));
    }
    return zt.tensor.allClose(allocator, fwd_jacobian, bwd_jacobian, @floatCast(precision));
}

fn funcIdx(allocator: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
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
    //  try std.testing.expect(try jacobianTestImpl(allocator, funcIdx, x, null, 1e-5, 1e-4));
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

    // Binary operators
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
    try std.testing.expectError(error.VariableDTypeMismatch, matmul(allocator, half, float));
    try std.testing.expectError(error.VariableDTypeMismatch, matmulTN(allocator, half, float));
    try std.testing.expectError(error.VariableDTypeMismatch, matmulNT(allocator, half, float));

    // expect no throw
    var res1 = try binaryCrossEntropy(allocator, half, float);
    res1.deinit();

    var cat_input = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 7, 10, 4 }, .f16), true);
    defer cat_input.deinit();
    var tmp_cat_target1 = try zt.tensor.rand(allocator, &.{ 10, 4 }, .u32);
    defer tmp_cat_target1.deinit();
    var tmp_cat_target2 = try zt.tensor.mod(allocator, Tensor, tmp_cat_target1, f64, 7);
    defer tmp_cat_target2.deinit();
    var cat_target = try Variable.init(allocator, try tmp_cat_target2.astype(allocator, .s32), false);
    defer cat_target.deinit();
    var res2 = try categoricalCrossEntropy(allocator, cat_input, cat_target, .Mean, -1);
    res2.deinit();

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

test "AutogradTest -> CastingAsBackward" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons
    if (!try zt.f16Supported(allocator)) {
        return error.SkipZigTest;
    }

    var a = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 4, 4 }, .f16), true);
    defer a.deinit();
    var b = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 4, 4 }, .f16), false);
    defer b.deinit();
    var c = try add(allocator, *Variable, b, *Variable, a);
    defer c.deinit();
    try c.backward(allocator, false);
    try std.testing.expect(try (try a.grad()).dtype(allocator) == .f16);
    try std.testing.expect(try (try c.grad()).dtype(allocator) == .f16);
    var d = try a.astype(allocator, .f32);
    defer d.deinit();
    try std.testing.expect(!d.isGradAvailable());
}

test "AutogradTest -> CastingAsGrad" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons
    if (!try zt.f16Supported(allocator)) {
        return error.SkipZigTest;
    }

    // compare to f32 case
    var x = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 2, .f32), true);
    defer x.deinit();
    var y = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 3, .f32), true);
    defer y.deinit();
    var tmp1 = try mul(allocator, *Variable, x, *Variable, x);
    defer tmp1.deinit();
    var tmp2 = try mul(allocator, *Variable, x, *Variable, y);
    defer tmp2.deinit();
    var tmp3 = try mul(allocator, *Variable, y, *Variable, y);
    defer tmp3.deinit();
    var tmp4 = try add(allocator, *Variable, tmp1, *Variable, tmp2);
    defer tmp4.deinit();
    var z = try add(allocator, *Variable, tmp4, *Variable, tmp3);
    defer z.deinit();
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dx = try x.grad();
    var dy = try y.grad();

    // f16 -- cast gradients in both directions
    var x32 = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 2, .f32), true);
    defer x32.deinit();
    var y32 = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 3, .f32), true);
    defer y32.deinit();
    var xf16 = try x32.astype(allocator, .f16);
    defer xf16.deinit();
    var yf16 = try y32.astype(allocator, .f16);
    defer yf16.deinit();
    var tmp5 = try mul(allocator, *Variable, xf16, *Variable, xf16);
    defer tmp5.deinit();
    var tmp6 = try mul(allocator, *Variable, xf16, *Variable, yf16);
    defer tmp6.deinit();
    var tmp7 = try mul(allocator, *Variable, yf16, *Variable, yf16);
    defer tmp7.deinit();
    var tmp8 = try add(allocator, *Variable, tmp5, *Variable, tmp6);
    defer tmp8.deinit();
    var zf16 = try add(allocator, *Variable, tmp8, *Variable, tmp7);
    defer zf16.deinit();
    var zf32 = try zf16.astype(allocator, .f32);
    defer zf32.deinit();
    try zf32.backwardWithGrad(allocator, dz, false);

    try std.testing.expect(try (try xf16.grad()).dtype(allocator) == .f16);
    try std.testing.expect(try (try yf16.grad()).dtype(allocator) == .f16);
    try std.testing.expect(try (try zf16.grad()).dtype(allocator) == .f16);
    try std.testing.expect(try (try x32.grad()).dtype(allocator) == .f32);
    try std.testing.expect(try (try y32.grad()).dtype(allocator) == .f32);
    var exp1 = try (try xf16.grad()).tensor().astype(allocator, .f32);
    defer exp1.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), exp1, 1e-5));
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), (try y32.grad()).tensor(), 1e-5));
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), (try x32.grad()).tensor(), 1e-5));
}

test "AutogradTest -> NoCalcGrad" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), false);
    defer x.deinit();
    var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer y.deinit();
    var tmp1 = try mul(allocator, *Variable, x, *Variable, x);
    defer tmp1.deinit();
    var tmp2 = try mul(allocator, *Variable, x, *Variable, y);
    defer tmp2.deinit();
    var tmp3 = try mul(allocator, *Variable, y, *Variable, y);
    defer tmp3.deinit();
    var tmp4 = try add(allocator, *Variable, tmp1, *Variable, tmp2);
    defer tmp4.deinit();
    var z = try add(allocator, *Variable, tmp4, *Variable, tmp3);
    defer z.deinit();
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dy = try y.grad();
    var exp1 = try zt.tensor.mul(allocator, i8, 2, Tensor, y.tensor());
    defer exp1.deinit();
    try exp1.inPlaceAdd(allocator, Tensor, x.tensor());
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), exp1, 1e-5));
    try std.testing.expectError(error.GradientCalcDisabled, x.grad());
}

const ConcatT1Ctx = struct { x2: *Variable, x3: *Variable, x4: *Variable };
fn funcConcatenateT1(allocator: std.mem.Allocator, input: *Variable, ctx: ?*anyopaque) !*Variable {
    var c: *ConcatT1Ctx = @ptrCast(@alignCast(ctx.?));
    return concatenate(allocator, &.{ input, c.x2, c.x3, c.x4 }, 2);
}

const ConcatT2Ctx = struct { x1: *Variable, x2: *Variable, x4: *Variable };
fn funcConcatenateT2(allocator: std.mem.Allocator, input: *Variable, ctx: ?*anyopaque) !*Variable {
    var c: *ConcatT2Ctx = @ptrCast(@alignCast(ctx.?));
    return concatenate(allocator, &.{ c.x1, c.x2, input, c.x4 }, 2);
}

test "AutogradTest -> Concatenate" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    var x1 = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 3, 1, 2 }, .f64), true);
    defer x1.deinit();
    var x2 = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 3, 3, 2 }, .f64), true);
    defer x2.deinit();
    var x3 = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 3, 1, 2 }, .f64), true);
    defer x3.deinit();
    var x4 = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 3, 7, 2 }, .f64), true);
    defer x4.deinit();
    var output = try concatenate(allocator, &.{ x1, x2, x3, x4 }, 2);
    defer output.deinit();

    try std.testing.expect(zt.tensor.shape.eql(try output.shape(allocator), &.{ 2, 3, 12, 2 }));
    var concat_t1_ctx = try allocator.create(ConcatT1Ctx);
    defer allocator.destroy(concat_t1_ctx);
    concat_t1_ctx.* = .{ .x2 = x2, .x3 = x3, .x4 = x4 };
    try std.testing.expect(try jacobianTestImpl(allocator, funcConcatenateT1, x1, concat_t1_ctx, 1e-5, 1e-4));

    var concat_t2_ctx = try allocator.create(ConcatT2Ctx);
    defer allocator.destroy(concat_t2_ctx);
    concat_t2_ctx.* = .{ .x1 = x1, .x2 = x2, .x4 = x4 };
    try std.testing.expect(try jacobianTestImpl(allocator, funcConcatenateT2, x3, concat_t2_ctx, 1e-5, 1e-4));
}

fn funcSplit(allocator: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
    var tmp = try split(allocator, input, i64, 2, 1);
    var res = tmp[0];
    defer {
        for (tmp, 0..) |v, i| {
            if (i == 0) continue;
            v.deinit();
        }
        allocator.free(tmp);
    }
    return res;
}

test "AutogradTest -> Split" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    // check output
    var x = try Variable.init(allocator, try zt.tensor.arange(allocator, &.{ 7, 2 }, 0, .f32), true);
    defer x.deinit();
    var y_split = try split(allocator, x, i64, 1, 0);
    defer {
        for (y_split) |y| y.deinit();
        allocator.free(y_split);
    }
    try std.testing.expect(y_split.len == 7);
    try std.testing.expect(zt.tensor.shape.eql(try y_split[0].shape(allocator), &.{ 1, 2 }));
    try std.testing.expect(zt.tensor.shape.eql(try y_split[2].shape(allocator), &.{ 1, 2 }));
    var tmp_eq = try zt.tensor.eq(allocator, Tensor, y_split[6].tensor(), f64, 6);
    defer tmp_eq.deinit();
    var tmp_all = try zt.tensor.all(allocator, tmp_eq, &.{}, false);
    defer tmp_all.deinit();
    try std.testing.expect(try tmp_all.scalar(allocator, i8) != 0);

    var a = try Variable.init(allocator, try zt.tensor.arange(allocator, &.{ 5, 3 }, 1, .f32), true);
    defer a.deinit();
    var b_split = try split(allocator, a, []const i64, &.{ 2, 1 }, 1);
    defer {
        for (b_split) |b| b.deinit();
        allocator.free(b_split);
    }
    try std.testing.expect(b_split.len == 2);
    try std.testing.expect(zt.tensor.shape.eql(try b_split[0].shape(allocator), &.{ 5, 2 }));
    try std.testing.expect(zt.tensor.shape.eql(try b_split[1].shape(allocator), &.{ 5, 1 }));
    var tmp_comp = try zt.tensor.arange(allocator, &.{ 5, 2 }, 1, .f32);
    defer tmp_comp.deinit();
    var tmp_eq1 = try zt.tensor.eq(allocator, Tensor, b_split[0].tensor(), Tensor, tmp_comp);
    defer tmp_eq1.deinit();
    var tmp_all1 = try zt.tensor.all(allocator, tmp_eq1, &.{}, false);
    defer tmp_all1.deinit();
    try std.testing.expect(try tmp_all1.scalar(allocator, i8) != 0);
    var tmp_eq2 = try zt.tensor.eq(allocator, Tensor, b_split[1].tensor(), f64, 2);
    defer tmp_eq2.deinit();
    var tmp_all2 = try zt.tensor.all(allocator, tmp_eq2, &.{}, false);
    defer tmp_all2.deinit();
    try std.testing.expect(try tmp_all2.scalar(allocator, i8) != 0);

    // check exception handling
    try std.testing.expectError(error.SumOfSplitMismatch, split(allocator, a, []const i64, &.{ 2, 2 }, 0));

    // check gradient
    var input = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 3 }, .f64), true);
    defer input.deinit();
    try std.testing.expect(try jacobianTestImpl(allocator, funcSplit, input, null, 1e-5, 1e-4));
}

fn funcTile(allocator: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
    return tile(allocator, input, &.{ 1, 2 });
}

test "AutogradTest -> Tile" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{6}, .f32), true);
    defer x.deinit();
    var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 6, 3 }, .f32), true);
    defer y.deinit();
    var tmp = try tile(allocator, x, &.{ 1, 3 });
    defer tmp.deinit();
    var z = try mul(allocator, *Variable, y, *Variable, tmp);
    defer z.deinit();
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{ 6, 3 }, f64, 1, .f32), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dy = try y.grad();
    var dx = try x.grad();
    var exp1 = try zt.tensor.tile(allocator, x.tensor(), &.{ 1, 3 });
    defer exp1.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), exp1, 1e-5));
    var exp2 = try zt.tensor.sum(allocator, y.tensor(), &.{1}, false);
    defer exp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), exp2, 1e-5));

    // Jacobian
    var input = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 10, 1, 5 }, .f32), true);
    defer input.deinit();
    try std.testing.expect(try jacobianTestImpl(allocator, funcTile, input, null, 1e-4, 1e-3));
}

test "AutogradTest -> TileAs" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer x.deinit();
    var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 2 }, .f32), true);
    defer y.deinit();
    var tmp = try tileAs(allocator, x, *Variable, y);
    defer tmp.deinit();
    var z = try mul(allocator, *Variable, y, *Variable, tmp);
    defer z.deinit();
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{ 5, 2 }, f64, 1, .f32), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dy = try y.grad();
    var dx = try x.grad();
    var exp1 = try zt.tensor.tile(allocator, x.tensor(), &.{ 1, 2 });
    defer exp1.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), exp1, 1e-5));
    var exp2 = try zt.tensor.sum(allocator, y.tensor(), &.{1}, false);
    defer exp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), exp2, 1e-5));
}

// TODO: test "AutogradTest -> TileAsF16" {}

test "AutogradTest -> TileAs2" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{10}, .f32), true);
    defer x.deinit();
    var z = try tileAs(allocator, x, Shape, &.{ 10, 3 });
    defer z.deinit();
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{ 10, 3 }, f64, 1, .f32), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dx = try x.grad();
    var exp1 = try zt.tensor.full(allocator, try x.shape(allocator), f64, 3, .f32);
    defer exp1.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), exp1, 1e-5));
}

fn funcCol(allocator: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
    return input.index(allocator, &.{ Index.initRange(zt.tensor.span), Index.initDim(4) });
}

fn funcRow(allocator: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
    return input.index(allocator, &.{Index.initDim(4)});
}

fn funcSlice(allocator: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
    return input.index(allocator, &.{ Index.initRange(zt.tensor.span), Index.initRange(zt.tensor.span), Index.initDim(4) });
}

fn funcCols(allocator: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
    return input.index(allocator, &.{ Index.initRange(zt.tensor.span), Index.initRange(Range.init(2, .{ .dim = 5 })) });
}

fn funcRows(allocator: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
    return input.index(allocator, &.{Index.initRange(Range.init(2, .{ .dim = 5 }))});
}

fn funcSlices(allocator: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
    return input.index(allocator, &.{ Index.initRange(zt.tensor.span), Index.initRange(zt.tensor.span), Index.initRange(Range.init(2, .{ .dim = 5 })) });
}

fn funcFlat(allocator: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
    return input.flat(allocator, Index.initRange(Range.init(4, .{ .dim = 100 })));
}

test "AutogradTest -> Indexing" {
    const allocator = std.testing.allocator;
    defer zt.tensor.deinit(); // deinit global singletons

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 6, 7, 4 }, .f64), true);
    defer x.deinit();
    try std.testing.expect(try jacobianTestImpl(allocator, funcCol, x, null, 1e-5, 1e-4));
    try std.testing.expect(try jacobianTestImpl(allocator, funcRow, x, null, 1e-5, 1e-4));
    try std.testing.expect(try jacobianTestImpl(allocator, funcSlice, x, null, 1e-5, 1e-4));
    try std.testing.expect(try jacobianTestImpl(allocator, funcCols, x, null, 1e-5, 1e-4));
    try std.testing.expect(try jacobianTestImpl(allocator, funcRows, x, null, 1e-5, 1e-4));
    try std.testing.expect(try jacobianTestImpl(allocator, funcSlices, x, null, 1e-5, 1e-4));
    try std.testing.expect(try jacobianTestImpl(allocator, funcFlat, x, null, 1e-5, 1e-4));
}
