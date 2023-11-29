const zt = @import("../zt.zig");
const zigrc = @import("zigrc");
const std = @import("std");

const ConvBenchmarks = zt.common.ConvBenchmarks;
const DynamicBenchmark = zt.common.DynamicBenchmark;
const Index = zt.tensor.Index;
const Range = zt.tensor.Range;
const Tensor = zt.tensor.Tensor;
const Shape = zt.tensor.shape.Shape;
const Dim = zt.tensor.shape.Dim;
const PoolingMode = zt.common.PoolingMode;
const AutogradPayload = zt.autograd.AutogradPayload;
const AutogradPayloadData = zt.autograd.AutogradPayloadData;
const Variable = zt.autograd.Variable;
const AutogradExtension = zt.autograd.AutogradExtension;

pub const detail = struct {
    /// Performs type conversion based on the optim level. Operations that lack
    /// sufficient precision are automatically upcast to f32 before computation.
    /// These are typically operations that require accumulations or reductions.
    pub fn adjustInputType(allocator: std.mem.Allocator, comptime T: type, in: T, func_name: []const u8) !struct { res: T, allocated: bool } {
        const optim_level: zt.common.OptimLevel = zt.common.OptimMode.get().getOptimLevel();
        // Fastpath - Default mode never casts tensors
        if (optim_level == .Default) {
            return .{ .res = in, .allocated = false };
        }

        if (!zt.common.kOptimLevelTypeExclusionMappings(optim_level, func_name)) {
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

    pub fn createAutogradPayload(allocator: std.mem.Allocator, args: []const *Variable) !?zigrc.Arc(AutogradPayload) {
        for (args) |a| {
            if (a.isCalcGrad()) {
                return try zigrc.Arc(AutogradPayload).init(allocator, .{ .data = try zigrc.Arc(AutogradPayloadData).init(allocator, null) });
            }
        }
        return null;
    }

    pub fn tileAs(allocator: std.mem.Allocator, input: Tensor, rdims: Shape) !Tensor {
        // Scalar tensor
        if (try input.ndim(allocator) == 0) {
            return zt.tensor.tile(allocator, input, rdims);
        }
        var dims = try allocator.alloc(Dim, zt.tensor.shape.ndim(rdims));
        defer allocator.free(dims);
        @memset(dims, 1);
        const idims = try input.shape(allocator);
        for (0..zt.tensor.shape.ndim(rdims)) |i| {
            const idims_size: Dim = if (i + 1 > zt.tensor.shape.ndim(idims)) 1 else idims[i];
            if (@rem(rdims[i], idims_size) != 0) {
                std.debug.print("Invalid dims for tileAs for input dims {any} to output dims {any}\n", .{ idims, rdims });
                return error.TileAsInvalidDims;
            }
            dims[i] = @divTrunc(rdims[i], idims_size);
        }
        return zt.tensor.tile(allocator, input, dims);
    }

    pub fn sumAs(allocator: std.mem.Allocator, input: Tensor, rdims: Shape) !Tensor {
        const idims = try input.shape(allocator);
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
                const o = try expandedShapeFromReducedDims(allocator, input, axes, keep_dims);
                defer if (o.allocated) allocator.free(o.shape);
                return zt.tensor.reshape(allocator, input, o.shape);
            },
            *Variable => {
                const o = try expandedShapeFromReducedDims(allocator, input.tensor(), axes, keep_dims);
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
    const var1 = vars[0];
    for (vars[1..]) |v| {
        if (!try detail.areVariableTypesEqual(allocator, var1, v)) {
            std.log.debug("{s}  doesn't support binary operations with Variables of different types\n", .{fn_name});
            return error.VariableDTypeMismatch;
        }
    }
}

pub fn add(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.add only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        const result = try zt.tensor.add(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
                var var1 = try Variable.initSharedData(alloc, grad_output.shared_data.retain(), false);
                defer var1.deinit();
                var var2 = try Variable.initSharedData(alloc, grad_output.shared_data.retain(), false);
                defer var2.deinit();
                try inputs[0].addGrad(alloc, var1);
                try inputs[1].addGrad(alloc, var2);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{ try lhs.withoutData(allocator), try rhs.withoutData(allocator) }, gradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        const result = try zt.tensor.add(allocator, Tensor, lhs.tensor(), f64, rhs);
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
                var tmp_var = try Variable.initSharedData(alloc, grad_output.shared_data.retain(), false);
                defer tmp_var.deinit();
                try inputs[0].addGrad(alloc, tmp_var);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{try lhs.withoutData(allocator)}, gradFunc, null, null);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return add(allocator, RhsT, rhs, LhsT, lhs);
    }
}

pub fn sub(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.sub only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        const result = try zt.tensor.sub(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
                var var1 = try Variable.initSharedData(alloc, grad_output.shared_data.retain(), false);
                defer var1.deinit();
                try inputs[0].addGrad(alloc, var1);
                var neg = try negate(alloc, grad_output);
                defer neg.deinit();
                var var2 = try Variable.initSharedData(alloc, neg.shared_data.retain(), false);
                defer var2.deinit();
                try inputs[1].addGrad(alloc, var2);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{ try lhs.withoutData(allocator), try rhs.withoutData(allocator) }, gradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        const result = try zt.tensor.sub(allocator, Tensor, lhs.tensor(), f64, rhs);
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
                var var1 = try Variable.initSharedData(alloc, grad_output.shared_data.retain(), false);
                defer var1.deinit();
                try inputs[0].addGrad(alloc, var1);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{try lhs.withoutData(allocator)}, gradFunc, null, null);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        const result = try zt.tensor.sub(allocator, f64, lhs, Tensor, rhs.tensor());
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
                var neg = try negate(alloc, grad_output);
                defer neg.deinit();
                var var1 = try Variable.initSharedData(alloc, neg.shared_data.retain(), false);
                defer var1.deinit();
                try inputs[0].addGrad(alloc, var1);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{try rhs.withoutData(allocator)}, gradFunc, null, null);
    }
}

pub fn mul(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.mul only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        const result = try zt.tensor.mul(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        const in1: *Variable = if (rhs.isCalcGrad()) try lhs.clone(allocator) else try lhs.withoutData(allocator);
        const in2: *Variable = if (lhs.isCalcGrad()) try rhs.clone(allocator) else try rhs.withoutData(allocator);
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
                if (inputs[0].isCalcGrad()) {
                    var tmp_var = try Variable.init(alloc, try zt.tensor.mul(alloc, Tensor, grad_output.tensor(), Tensor, inputs[1].tensor()), false);
                    defer tmp_var.deinit();
                    try inputs[0].addGrad(alloc, tmp_var);
                }
                if (inputs[1].isCalcGrad()) {
                    var tmp_var = try Variable.init(alloc, try zt.tensor.mul(alloc, Tensor, grad_output.tensor(), Tensor, inputs[0].tensor()), false);
                    defer tmp_var.deinit();
                    try inputs[1].addGrad(alloc, tmp_var);
                }
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{ in1, in2 }, gradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        const result = try zt.tensor.mul(allocator, Tensor, lhs.tensor(), f64, rhs);
        const F64Ctx = struct { val: f64 };
        const freeCtx = (struct {
            pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
                const grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
                alloc.destroy(grad_ctx);
            }
        }).call;
        const ctx = try allocator.create(F64Ctx);
        ctx.* = .{ .val = rhs };
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, gc: ?*anyopaque) !void {
                const grad_ctx: *F64Ctx = @ptrCast(@alignCast(gc));
                var tmp_var = try Variable.init(alloc, try zt.tensor.mul(alloc, Tensor, grad_output.tensor(), f64, grad_ctx.val), false);
                defer tmp_var.deinit();
                try inputs[0].addGrad(alloc, tmp_var);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{try lhs.withoutData(allocator)}, gradFunc, ctx, freeCtx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return mul(allocator, RhsT, rhs, LhsT, lhs);
    }
}

pub fn div(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.div only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        const result = try zt.tensor.div(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        const in1: *Variable = if (rhs.isCalcGrad()) try lhs.clone(allocator) else try lhs.withoutData(allocator);
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
                var inputs1_rec = try reciprocal(alloc, inputs[1]);
                defer inputs1_rec.deinit();
                var grad_input0 = try mul(alloc, *Variable, grad_output, *Variable, inputs1_rec);
                defer grad_input0.deinit();
                if (inputs[0].isCalcGrad()) {
                    var tmp_var = try Variable.initSharedData(alloc, grad_input0.shared_data.retain(), false);
                    defer tmp_var.deinit();
                    try inputs[0].addGrad(alloc, tmp_var);
                }
                if (inputs[1].isCalcGrad()) {
                    var tmp_neg = try negate(alloc, inputs[0]);
                    defer tmp_neg.deinit();
                    var tmp = try mul(alloc, *Variable, grad_input0, *Variable, tmp_neg);
                    defer tmp.deinit();
                    var tmp2 = try mul(alloc, *Variable, tmp, *Variable, inputs1_rec);
                    defer tmp2.deinit();
                    var tmp_var = try Variable.initSharedData(alloc, tmp2.shared_data.retain(), false);
                    defer tmp_var.deinit();
                    try inputs[1].addGrad(alloc, tmp_var);
                }
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{ in1, try rhs.clone(allocator) }, gradFunc, null, null);
    }
    const F64Ctx = struct { val: f64 };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
            alloc.destroy(grad_ctx);
        }
    }).call;
    if (LhsT == *Variable and RhsT == f64) {
        const result = try zt.tensor.div(allocator, Tensor, lhs.tensor(), f64, rhs);

        const ctx = try allocator.create(F64Ctx);
        ctx.* = .{ .val = rhs };
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
                const grad_ctx: *F64Ctx = @ptrCast(@alignCast(c));
                var tmp = try div(alloc, *Variable, grad_output, f64, grad_ctx.val);
                defer tmp.deinit();
                var tmp_var = try Variable.initSharedData(alloc, tmp.shared_data.retain(), false);
                defer tmp_var.deinit();
                try inputs[0].addGrad(alloc, tmp_var);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{try lhs.withoutData(allocator)}, gradFunc, ctx, freeCtx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        const result = try zt.tensor.div(allocator, f64, lhs, Tensor, rhs.tensor());
        const ctx = try allocator.create(F64Ctx);
        ctx.* = .{ .val = lhs };
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
                const grad_ctx: *F64Ctx = @ptrCast(@alignCast(c));
                var tmp = try mul(alloc, *Variable, grad_output, f64, -grad_ctx.val);
                defer tmp.deinit();
                var tmp2 = try mul(alloc, *Variable, inputs[0], *Variable, inputs[0]);
                defer tmp2.deinit();
                var res = try div(alloc, *Variable, tmp, *Variable, tmp2);
                defer res.deinit();
                var tmp_var = try Variable.initSharedData(alloc, res.shared_data.retain(), false);
                defer tmp_var.deinit();
                try inputs[0].addGrad(alloc, tmp_var);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{try rhs.clone(allocator)}, gradFunc, ctx, freeCtx);
    }
}

pub fn greaterThan(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.greaterThan only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        const result = try zt.tensor.greaterThan(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
    if (LhsT == *Variable and RhsT == f64) {
        const result = try zt.tensor.greaterThan(allocator, Tensor, lhs.tensor(), f64, rhs);
        return Variable.init(allocator, result, false);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        const result = try zt.tensor.greaterThan(allocator, f64, lhs, Tensor, rhs.tensor());
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
        const result = try zt.tensor.lessThan(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
    if (LhsT == *Variable and RhsT == f64) {
        const result = try zt.tensor.lessThan(allocator, Tensor, lhs.tensor(), f64, rhs);
        return Variable.init(allocator, result, false);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        const result = try zt.tensor.lessThan(allocator, f64, lhs, Tensor, rhs.tensor());
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
        const result = try zt.tensor.greaterThanEqual(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
    if (LhsT == *Variable and RhsT == f64) {
        const result = try zt.tensor.greaterThanEqual(allocator, Tensor, lhs.tensor(), f64, rhs);
        return Variable.init(allocator, result, false);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        const result = try zt.tensor.greaterThanEqual(allocator, f64, lhs, Tensor, rhs.tensor());
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
        const result = try zt.tensor.lessThanEqual(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
    if (LhsT == *Variable and RhsT == f64) {
        const result = try zt.tensor.lessThanEqual(allocator, Tensor, lhs.tensor(), f64, rhs);
        return Variable.init(allocator, result, false);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        const result = try zt.tensor.lessThanEqual(allocator, f64, lhs, Tensor, rhs.tensor());
        return Variable.init(allocator, result, false);
    }
}

pub fn logicalAnd(allocator: std.mem.Allocator, lhs: *Variable, rhs: *Variable) !*Variable {
    try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
    return Variable.init(allocator, try zt.tensor.logicalAnd(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor()), false);
}

pub fn logicalNot(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var result = try zt.tensor.logicalNot(allocator, input.tensor());
    defer result.deinit();
    return Variable.init(allocator, try result.astype(allocator, try input.dtype(allocator)), false);
}

pub fn max(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.max only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        const result = try zt.tensor.maximum(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
                var tmp_mask = try zt.tensor.greaterThan(alloc, Tensor, inputs[0].tensor(), Tensor, inputs[1].tensor());
                defer tmp_mask.deinit();
                var mask = try Variable.init(alloc, try tmp_mask.astype(alloc, try grad_output.dtype(alloc)), false);
                defer mask.deinit();
                var tmp1 = try mul(alloc, *Variable, mask, *Variable, grad_output);
                defer tmp1.deinit();
                var tmp_var1 = try Variable.initSharedData(alloc, tmp1.shared_data.retain(), false);
                defer tmp_var1.deinit();
                try inputs[0].addGrad(alloc, tmp_var1);
                var tmp_not = try logicalNot(alloc, mask);
                defer tmp_not.deinit();
                var tmp2 = try mul(alloc, *Variable, tmp_not, *Variable, grad_output);
                defer tmp2.deinit();
                var tmp_var2 = try Variable.initSharedData(alloc, tmp2.shared_data.retain(), false);
                defer tmp_var2.deinit();
                try inputs[1].addGrad(alloc, tmp_var2);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{ try lhs.clone(allocator), try rhs.clone(allocator) }, gradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        const result = try zt.tensor.maximum(allocator, Tensor, lhs.tensor(), f64, rhs);
        const F64Ctx = struct { val: f64 };
        const freeCtx = (struct {
            pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
                const grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
                alloc.destroy(grad_ctx);
            }
        }).call;
        const ctx = try allocator.create(F64Ctx);
        ctx.* = .{ .val = rhs };
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
                const grad_ctx: *F64Ctx = @ptrCast(@alignCast(c));
                var tmp_mask = try zt.tensor.greaterThan(alloc, Tensor, inputs[0].tensor(), f64, grad_ctx.val);
                defer tmp_mask.deinit();
                var mask = try Variable.init(alloc, try tmp_mask.astype(alloc, try grad_output.dtype(alloc)), false);
                defer mask.deinit();
                var tmp1 = try mul(alloc, *Variable, mask, *Variable, grad_output);
                defer tmp1.deinit();
                var tmp_var = try Variable.initSharedData(alloc, tmp1.shared_data.retain(), false);
                defer tmp_var.deinit();
                try inputs[0].addGrad(alloc, tmp_var);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{try lhs.clone(allocator)}, gradFunc, ctx, freeCtx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return max(allocator, RhsT, rhs, LhsT, lhs);
    }
}

pub fn min(allocator: std.mem.Allocator, comptime LhsT: type, lhs: LhsT, comptime RhsT: type, rhs: RhsT) !*Variable {
    if ((LhsT != *Variable and LhsT != f64) or (RhsT != *Variable and RhsT != f64) or (LhsT == f64 and RhsT == f64)) {
        std.log.debug("autograd.min only supports Variables and f64\n", .{});
        return error.VariableOpInvalidArgTypes;
    }
    if (LhsT == *Variable and RhsT == *Variable) {
        try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
        const result = try zt.tensor.minimum(allocator, Tensor, lhs.tensor(), Tensor, rhs.tensor());
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
                var tmp_mask = try zt.tensor.lessThan(alloc, Tensor, inputs[0].tensor(), Tensor, inputs[1].tensor());
                defer tmp_mask.deinit();
                var mask = try Variable.init(alloc, try tmp_mask.astype(alloc, try grad_output.dtype(alloc)), false);
                defer mask.deinit();
                var tmp1 = try mul(alloc, *Variable, mask, *Variable, grad_output);
                defer tmp1.deinit();
                var tmp_var1 = try Variable.initSharedData(alloc, tmp1.shared_data.retain(), false);
                defer tmp_var1.deinit();
                try inputs[0].addGrad(alloc, tmp_var1);
                var tmp_not = try logicalNot(alloc, mask);
                defer tmp_not.deinit();
                var tmp2 = try mul(alloc, *Variable, tmp_not, *Variable, grad_output);
                defer tmp2.deinit();
                var tmp_var2 = try Variable.initSharedData(alloc, tmp2.shared_data.retain(), false);
                defer tmp_var2.deinit();
                try inputs[1].addGrad(alloc, tmp_var2);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{ try lhs.clone(allocator), try rhs.clone(allocator) }, gradFunc, null, null);
    }
    if (LhsT == *Variable and RhsT == f64) {
        const result = try zt.tensor.minimum(allocator, Tensor, lhs.tensor(), f64, rhs);
        const F64Ctx = struct { val: f64 };
        const freeCtx = (struct {
            pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
                const grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
                alloc.destroy(grad_ctx);
            }
        }).call;
        const ctx = try allocator.create(F64Ctx);
        ctx.* = .{ .val = rhs };
        const gradFunc = (struct {
            pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
                const grad_ctx: *F64Ctx = @ptrCast(@alignCast(c));
                var tmp_mask = try zt.tensor.lessThan(alloc, Tensor, inputs[0].tensor(), f64, grad_ctx.val);
                defer tmp_mask.deinit();
                var mask = try Variable.init(alloc, try tmp_mask.astype(alloc, try grad_output.dtype(alloc)), false);
                defer mask.deinit();
                var tmp1 = try mul(alloc, *Variable, mask, *Variable, grad_output);
                defer tmp1.deinit();
                var tmp_var = try Variable.initSharedData(alloc, tmp1.shared_data.retain(), false);
                defer tmp_var.deinit();
                try inputs[0].addGrad(alloc, tmp_var);
            }
        }).call;
        return Variable.initWithInputs(allocator, result, &.{try lhs.clone(allocator)}, gradFunc, ctx, freeCtx);
    }
    if (LhsT == f64 and RhsT == *Variable) {
        return min(allocator, RhsT, rhs, LhsT, lhs);
    }
}

pub fn negate(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    const result = try zt.tensor.sub(allocator, f64, 0, Tensor, input.tensor());
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            var tmp_neg = try negate(alloc, grad_output);
            defer tmp_neg.deinit();
            var tmp_var = try Variable.initSharedData(alloc, tmp_neg.shared_data.retain(), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, null, null);
}

pub fn reciprocal(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            var res = try reciprocal(alloc, inputs[0]);
            defer res.deinit();
            var tmp_neg = try negate(alloc, grad_output);
            defer tmp_neg.deinit();
            var tmp1 = try mul(alloc, *Variable, tmp_neg, *Variable, res);
            defer tmp1.deinit();
            var tmp2 = try mul(alloc, *Variable, tmp1, *Variable, res);
            defer tmp2.deinit();
            var tmp_var = try Variable.initSharedData(alloc, tmp2.shared_data.retain(), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    return Variable.initWithInputs(allocator, try zt.tensor.div(allocator, f64, 1, Tensor, adj.res), &.{try input.clone(allocator)}, gradFunc, null, null);
}

pub fn exp(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            var res = try zt.tensor.exp(alloc, inputs[0].tensor());
            try res.inPlaceMul(alloc, Tensor, grad_output.tensor());
            var tmp_var = try Variable.init(alloc, res, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    return Variable.initWithInputs(allocator, try zt.tensor.exp(allocator, adj.res), &.{try input.clone(allocator)}, gradFunc, null, null);
}

pub fn log(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            const res = try zt.tensor.div(alloc, Tensor, grad_output.tensor(), Tensor, inputs[0].tensor());
            var tmp_var = try Variable.init(alloc, res, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    return Variable.initWithInputs(allocator, try zt.tensor.log(allocator, adj.res), &.{try input.clone(allocator)}, gradFunc, null, null);
}

pub fn log1p(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            var tmp = try zt.tensor.add(alloc, f64, 1, Tensor, inputs[0].tensor());
            defer tmp.deinit();
            const res = try zt.tensor.div(alloc, Tensor, grad_output.tensor(), Tensor, tmp);
            var tmp_var = try Variable.init(alloc, res, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    return Variable.initWithInputs(allocator, try zt.tensor.log1p(allocator, adj.res), &.{try input.clone(allocator)}, gradFunc, null, null);
}

pub fn pow(allocator: std.mem.Allocator, input: *Variable, p: f64) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    const F64Ctx = struct { val: f64 };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const grad_ctx: *F64Ctx = @ptrCast(@alignCast(ctx));
            alloc.destroy(grad_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const grad_ctx: *F64Ctx = @ptrCast(@alignCast(c));
            var grad = try zt.tensor.power(alloc, Tensor, inputs[0].tensor(), f64, grad_ctx.val - 1);
            try grad.inPlaceMul(alloc, f64, grad_ctx.val);
            try grad.inPlaceMul(alloc, Tensor, grad_output.tensor());
            var tmp_var = try Variable.init(alloc, grad, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    const ctx = try allocator.create(F64Ctx);
    ctx.* = .{ .val = p };
    return Variable.initWithInputs(allocator, try zt.tensor.power(allocator, Tensor, adj.res, f64, p), &.{try input.clone(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn sin(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            var res = try zt.tensor.cos(alloc, inputs[0].tensor());
            try res.inPlaceMul(alloc, Tensor, grad_output.tensor());
            var tmp_var = try Variable.init(alloc, res, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    return Variable.initWithInputs(allocator, try zt.tensor.sin(allocator, input.tensor()), &.{try input.clone(allocator)}, gradFunc, null, null);
}

pub fn cos(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            var tmp = try zt.tensor.sin(alloc, inputs[0].tensor());
            defer tmp.deinit();
            var res = try zt.tensor.negative(alloc, tmp);
            try res.inPlaceMul(alloc, Tensor, grad_output.tensor());
            var tmp_var = try Variable.init(alloc, res, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    return Variable.initWithInputs(allocator, try zt.tensor.cos(allocator, input.tensor()), &.{try input.clone(allocator)}, gradFunc, null, null);
}

pub fn tanh(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    const TensorCtx = struct { val: Tensor };
    const freeCtx = (struct {
        fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const grad_ctx: *TensorCtx = @ptrCast(@alignCast(ctx));
            alloc.destroy(grad_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const grad_ctx: *TensorCtx = @ptrCast(@alignCast(c));
            var tmp1 = try zt.tensor.mul(alloc, Tensor, grad_ctx.val, Tensor, grad_ctx.val);
            defer tmp1.deinit();
            var tmp2 = try zt.tensor.sub(alloc, i8, 1, Tensor, tmp1);
            try tmp2.inPlaceMul(alloc, Tensor, grad_output.tensor());
            var tmp_var = try Variable.init(alloc, tmp2, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    const result = try zt.tensor.tanh(allocator, input.tensor());
    const ctx = try allocator.create(TensorCtx);
    ctx.* = .{ .val = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn clamp(allocator: std.mem.Allocator, input: *Variable, lo: f64, hi: f64) !*Variable {
    const ClampCtx = struct { lo: f64, hi: f64, result: Tensor };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const grad_ctx: *ClampCtx = @ptrCast(@alignCast(ctx));
            alloc.destroy(grad_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const grad_ctx: *ClampCtx = @ptrCast(@alignCast(c));
            var tmp1 = try zt.tensor.greaterThan(alloc, Tensor, grad_ctx.result, f64, grad_ctx.lo);
            defer tmp1.deinit();
            var tmp2 = try zt.tensor.lessThan(alloc, Tensor, grad_ctx.result, f64, grad_ctx.hi);
            defer tmp2.deinit();
            var tmp3 = try zt.tensor.logicalAnd(alloc, Tensor, tmp1, Tensor, tmp2);
            defer tmp3.deinit();
            var tmp_var = try Variable.init(alloc, try zt.tensor.where(alloc, tmp3, Tensor, grad_output.tensor(), f64, 0), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    const result = try zt.tensor.clip(allocator, input.tensor(), f64, lo, f64, hi);
    const ctx = try allocator.create(ClampCtx);
    ctx.* = .{ .lo = lo, .hi = hi, .result = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn sqrt(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    const TensorCtx = struct { val: Tensor };
    const freeCtx = (struct {
        fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const grad_ctx: *TensorCtx = @ptrCast(@alignCast(ctx));
            alloc.destroy(grad_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const grad_ctx: *TensorCtx = @ptrCast(@alignCast(c));
            var output = try Variable.init(alloc, try Tensor.initAssign(alloc, grad_ctx.val), false);
            defer output.deinit();
            var tmp = try mul(alloc, f64, 2, *Variable, output);
            defer tmp.deinit();
            var res = try div(alloc, *Variable, grad_output, *Variable, tmp);
            defer res.deinit();
            var tmp_var = try Variable.initSharedData(alloc, res.shared_data.retain(), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    const result = try zt.tensor.sqrt(allocator, input.tensor());
    const ctx = try allocator.create(TensorCtx);
    ctx.* = .{ .val = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn sigmoid(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    const TensorCtx = struct { val: Tensor };
    const freeCtx = (struct {
        fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const grad_ctx: *TensorCtx = @ptrCast(@alignCast(ctx));
            alloc.destroy(grad_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const grad_ctx: *TensorCtx = @ptrCast(@alignCast(c));
            var tmp = try zt.tensor.sub(alloc, f64, 1, Tensor, grad_ctx.val);
            try tmp.inPlaceMul(alloc, Tensor, grad_ctx.val);
            const grad = try zt.tensor.mul(alloc, Tensor, grad_output.tensor(), Tensor, tmp);
            tmp.deinit();
            var tmp_var = try Variable.init(alloc, grad, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    const result = try zt.tensor.sigmoid(allocator, input.tensor());
    const ctx = try allocator.create(TensorCtx);
    ctx.* = .{ .val = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn swish(allocator: std.mem.Allocator, input: *Variable, beta: f64) !*Variable {
    var tmp1 = try mul(allocator, f64, beta, *Variable, input);
    defer tmp1.deinit();
    var tmp2 = try sigmoid(allocator, tmp1);
    defer tmp2.deinit();
    return mul(allocator, *Variable, input, *Variable, tmp2);
}

pub fn erf(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            const x = inputs[0].tensor();
            var grad = try zt.tensor.mul(alloc, Tensor, grad_output.tensor(), f64, 2);
            try grad.inPlaceDiv(alloc, f64, @sqrt(@as(f64, std.math.pi)));
            var tmp1 = try zt.tensor.mul(alloc, Tensor, x, Tensor, x);
            defer tmp1.deinit();
            try tmp1.inPlaceMul(alloc, f64, -1);
            var tmp2 = try zt.tensor.exp(alloc, tmp1);
            defer tmp2.deinit();
            try grad.inPlaceMul(alloc, Tensor, tmp2);
            var tmp_var = try Variable.init(alloc, grad, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    return Variable.initWithInputs(allocator, try zt.tensor.erf(allocator, adj.res), &.{try input.clone(allocator)}, gradFunc, null, null);
}

pub fn transpose(allocator: std.mem.Allocator, input: *Variable, dims: []const Dim) !*Variable {
    const DimsCtx = struct { dims: []Dim };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const transpose_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
            alloc.free(transpose_ctx.dims);
            alloc.destroy(transpose_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const transpose_ctx: *DimsCtx = @ptrCast(@alignCast(c));
            var reverse_shape = try alloc.alloc(Dim, transpose_ctx.dims.len);
            defer alloc.free(reverse_shape);
            @memcpy(reverse_shape, transpose_ctx.dims);
            if (zt.tensor.shape.ndim(transpose_ctx.dims) != 0) {
                // Reverse if transposing all dims (empty arg)
                std.mem.reverse(Dim, reverse_shape);
            }
            for (0..zt.tensor.shape.ndim(reverse_shape)) |i| {
                reverse_shape[@intCast(transpose_ctx.dims[i])] = @intCast(i);
            }
            var tmp_var = try Variable.init(alloc, try zt.tensor.transpose(alloc, grad_output.tensor(), reverse_shape), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    const ctx = try allocator.create(DimsCtx);
    ctx.* = .{ .dims = try allocator.alloc(Dim, dims.len) };
    @memcpy(ctx.dims, dims);
    return Variable.initWithInputs(allocator, try zt.tensor.transpose(allocator, input.tensor(), dims), &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

fn tileAs(allocator: std.mem.Allocator, input: *Variable, comptime T: type, ref: T) !*Variable {
    return switch (T) {
        *Variable => tileAs(allocator, input, Shape, try ref.shape(allocator)),
        Shape => {
            const DimsCtx = struct { dims: []Dim };
            const freeCtx = (struct {
                pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
                    const transpose_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
                    alloc.free(transpose_ctx.dims);
                    alloc.destroy(transpose_ctx);
                }
            }).call;
            const gradFunc = (struct {
                pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
                    const tile_ctx: *DimsCtx = @ptrCast(@alignCast(c));
                    var tmp_sum = try sumAs(alloc, grad_output, Shape, tile_ctx.dims);
                    defer tmp_sum.deinit();
                    var tmp_var = try Variable.init(alloc, try tmp_sum.tensor().astype(alloc, try inputs[0].dtype(alloc)), false);
                    defer tmp_var.deinit();
                    try inputs[0].addGrad(alloc, tmp_var);
                }
            }).call;
            const in_dims = try input.shape(allocator);
            const ctx = try allocator.create(DimsCtx);
            ctx.* = .{ .dims = try allocator.alloc(Dim, in_dims.len) };
            @memcpy(ctx.dims, in_dims);
            return Variable.initWithInputs(allocator, try detail.tileAs(allocator, input.tensor(), ref), &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
        },
        else => @compileError("autograd.tileAs only supports ref value of *Variable or Shape"),
    };
}

fn sumAs(allocator: std.mem.Allocator, input: *Variable, comptime T: type, ref: T) !*Variable {
    return switch (T) {
        *Variable => sumAs(allocator, input, Shape, try ref.shape(allocator)),
        Shape => {
            const DimsCtx = struct { dims: []Dim };
            const freeCtx = (struct {
                pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
                    const transpose_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
                    alloc.free(transpose_ctx.dims);
                    alloc.destroy(transpose_ctx);
                }
            }).call;
            const gradFunc = (struct {
                pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
                    const sum_ctx: *DimsCtx = @ptrCast(@alignCast(c));
                    var tmp_tiled = try tileAs(alloc, grad_output, Shape, sum_ctx.dims);
                    defer tmp_tiled.deinit();
                    var tmp_var = try Variable.initSharedData(alloc, tmp_tiled.shared_data.retain(), false);
                    defer tmp_var.deinit();
                    try inputs[0].addGrad(alloc, tmp_var);
                }
            }).call;
            const in_dims = try input.shape(allocator);
            const ctx = try allocator.create(DimsCtx);
            ctx.* = .{ .dims = try allocator.alloc(Dim, in_dims.len) };
            @memcpy(ctx.dims, in_dims);
            return Variable.initWithInputs(allocator, try detail.sumAs(allocator, input.tensor(), ref), &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
        },
        else => @compileError("autograd.tileAs only supports ref value of *Variable or Shape"),
    };
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
    const dtype = try concat_inputs[0].dtype(allocator);
    const numdims = try concat_inputs[0].ndim(allocator);
    for (concat_inputs[1..]) |v| {
        if (try v.dtype(allocator) != dtype) {
            std.debug.print("concatenate: all input Variables must be of the same type\n", .{});
            return error.ConcatInputVariableDTypeMismatch;
        } else if (try v.ndim(allocator) != numdims) {
            std.debug.print("concatenate: all input Variables must have the same number of dimensions\n", .{});
            return error.ConcatenateInputVariableDimsMismatch;
        }
    }

    // All Variables must have the same size when indexed along the dim not being
    // concatenated along
    const dims = try concat_inputs[0].shape(allocator);
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

    const ConcatCtx = struct { dim: Dim, in_dims: []Shape, numdims: usize };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, c: *anyopaque) void {
            const concat_ctx: *ConcatCtx = @ptrCast(@alignCast(c));
            alloc.free(concat_ctx.in_dims);
            alloc.destroy(concat_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const concat_ctx: *ConcatCtx = @ptrCast(@alignCast(c));
            var sx = try alloc.alloc(Index, concat_ctx.numdims);
            defer alloc.free(sx);
            @memset(sx, Index.initRange(zt.tensor.span));
            var s: Dim = 0;
            for (0..inputs.len) |i| {
                sx[@intCast(concat_ctx.dim)] = Index.initRange(Range.init(s, .{ .dim = s + concat_ctx.in_dims[i][@intCast(concat_ctx.dim)] }));
                var tmp_var = try Variable.init(alloc, try grad_output.tensor().index(alloc, sx), false);
                defer tmp_var.deinit();
                try inputs[i].addGrad(alloc, tmp_var);
                s += concat_ctx.in_dims[i][@intCast(concat_ctx.dim)];
            }
        }
    }).call;

    const ctx = try allocator.create(ConcatCtx);
    ctx.* = .{ .dim = dim, .in_dims = try in_dims.toOwnedSlice(), .numdims = numdims };
    return Variable.initWithInputs(allocator, result, inputs_no_data.items, gradFunc, ctx, freeCtx);
}

pub fn split(allocator: std.mem.Allocator, input: *Variable, comptime T: type, splits: T, dim: Dim) ![]*Variable {
    return switch (T) {
        i64 => {
            if (splits <= 0) {
                std.debug.print("autograd.split: splits must be a positive integer\n", .{});
                return error.InvalidSplitSize;
            }
            const dim_size = try input.dim(allocator, @intCast(dim));
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
            const dim_size = try input.dim(allocator, @intCast(dim));
            const N = splits.len;
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
                const end: Dim = start + splits[i];
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

pub fn tile(allocator: std.mem.Allocator, input: *Variable, dims: Shape) !*Variable {
    const DimsCtx = struct { dims: []Dim };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const transpose_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
            alloc.free(transpose_ctx.dims);
            alloc.destroy(transpose_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const tile_ctx: *DimsCtx = @ptrCast(@alignCast(c));
            var tmp_sum = try sumAs(alloc, grad_output, Shape, tile_ctx.dims);
            defer tmp_sum.deinit();
            var tmp_var = try Variable.init(alloc, try tmp_sum.tensor().astype(alloc, try inputs[0].dtype(alloc)), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    const idims = try input.shape(allocator);
    const ctx = try allocator.create(DimsCtx);
    ctx.* = .{ .dims = try allocator.alloc(Dim, idims.len) };
    @memcpy(ctx.dims, idims);
    return Variable.initWithInputs(allocator, try zt.tensor.tile(allocator, input.tensor(), dims), &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn sum(allocator: std.mem.Allocator, input: *Variable, axes: []const i64, keep_dims: bool) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    const ReductionCtx = struct { in_dims: []Dim, axes: []Dim, keep_dims: bool };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const sum_ctx: *ReductionCtx = @ptrCast(@alignCast(ctx));
            alloc.free(sum_ctx.in_dims);
            alloc.free(sum_ctx.axes);
            alloc.destroy(sum_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const sum_ctx: *ReductionCtx = @ptrCast(@alignCast(c));
            var tmp = try detail.expandFromReduction(alloc, Tensor, grad_output.tensor(), sum_ctx.axes, sum_ctx.keep_dims);
            defer tmp.deinit();
            var tmp_var = try Variable.init(alloc, try detail.tileAs(alloc, tmp, sum_ctx.in_dims), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    const result = try zt.tensor.sum(allocator, adj.res, axes, keep_dims);
    const indims = try input.shape(allocator);
    const ctx = try allocator.create(ReductionCtx);
    ctx.* = .{ .in_dims = try allocator.alloc(Dim, indims.len), .axes = try allocator.alloc(Dim, axes.len), .keep_dims = keep_dims };
    @memcpy(ctx.in_dims, indims);
    @memcpy(ctx.axes, axes);
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn mean(allocator: std.mem.Allocator, input: *Variable, axes: []const Dim, keep_dims: bool) !*Variable {
    var adj = try detail.adjustInputType(allocator, Tensor, input.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    const ReductionCtx = struct { in_dims: []Dim, axes: []Dim, keep_dims: bool };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const sum_ctx: *ReductionCtx = @ptrCast(@alignCast(ctx));
            alloc.free(sum_ctx.in_dims);
            alloc.free(sum_ctx.axes);
            alloc.destroy(sum_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const mean_ctx: *ReductionCtx = @ptrCast(@alignCast(c));
            const odims = try grad_output.shape(alloc);
            var count: Dim = 1;
            for (0..zt.tensor.shape.ndim(mean_ctx.in_dims)) |i| {
                const odim_size: Dim = if (i + 1 > zt.tensor.shape.ndim(odims)) 1 else odims[i];
                count *= @divTrunc(mean_ctx.in_dims[i], odim_size);
            }
            var tmp = try detail.expandFromReduction(alloc, Tensor, grad_output.tensor(), mean_ctx.axes, mean_ctx.keep_dims);
            defer tmp.deinit();
            var grad = try detail.tileAs(alloc, tmp, mean_ctx.in_dims);
            try grad.inPlaceDiv(alloc, Dim, count);
            var tmp_var = try Variable.init(alloc, grad, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    const result = try zt.tensor.mean(allocator, adj.res, axes, keep_dims);
    const indims = try input.shape(allocator);
    const ctx = try allocator.create(ReductionCtx);
    ctx.* = .{ .in_dims = try allocator.alloc(Dim, indims.len), .axes = try allocator.alloc(i64, axes.len), .keep_dims = keep_dims };
    @memcpy(ctx.in_dims, indims);
    @memcpy(ctx.axes, axes);
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn variance(allocator: std.mem.Allocator, in: *Variable, axes: []const Dim, is_biased: bool, keep_dims: bool) !*Variable {
    const VarCtx = struct { val: f64, axes: []Dim };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, c: *anyopaque) void {
            const var_ctx: *VarCtx = @ptrCast(@alignCast(c));
            alloc.free(var_ctx.axes);
            alloc.destroy(var_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const var_ctx: *VarCtx = @ptrCast(@alignCast(c));
            var expanded_dims = try alloc.alloc(Dim, try inputs[0].ndim(alloc));
            defer alloc.free(expanded_dims);
            @memcpy(expanded_dims, try inputs[0].shape(alloc));
            var tile_dims = try alloc.alloc(Dim, try inputs[0].ndim(alloc));
            defer alloc.free(tile_dims);
            @memcpy(tile_dims, try inputs[0].shape(alloc));
            for (var_ctx.axes) |ax| {
                tile_dims[@intCast(ax)] = try inputs[0].dim(alloc, @intCast(ax));
                expanded_dims[@intCast(ax)] = 1;
            }

            var tmp1 = try moddims(alloc, grad_output, expanded_dims);
            defer tmp1.deinit();
            var tmp2 = try tileAs(alloc, tmp1, Shape, tile_dims);
            defer tmp2.deinit();
            var lhs = try mul(alloc, f64, 2 * var_ctx.val, *Variable, tmp2);
            defer lhs.deinit();
            var tmp3 = try mean(alloc, inputs[0], var_ctx.axes, false);
            defer tmp3.deinit();
            var tmp4 = try moddims(alloc, tmp3, expanded_dims);
            defer tmp4.deinit();
            var tmp5 = try tileAs(alloc, tmp4, Shape, tile_dims);
            defer tmp5.deinit();
            var rhs = try sub(alloc, *Variable, inputs[0], *Variable, tmp5);
            defer rhs.deinit();
            var res = try mul(alloc, *Variable, lhs, *Variable, rhs);
            defer res.deinit();
            var tmp_var = try Variable.initSharedData(alloc, res.shared_data.retain(), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    var adj = try detail.adjustInputType(allocator, Tensor, in.tensor(), @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var input = adj.res;
    var tmp = try zt.tensor.mul(allocator, Tensor, adj.res, Tensor, adj.res);
    defer tmp.deinit();
    var result = try zt.tensor.sum(allocator, tmp, axes, keep_dims);

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

    const val: f64 = 1 / (if (is_biased) @as(f64, @floatFromInt(n)) else @as(f64, @floatFromInt(n)) - 1);
    var tmp1 = try zt.tensor.mul(allocator, Tensor, avg, Tensor, avg);
    defer tmp1.deinit();
    try tmp1.inPlaceMul(allocator, i64, n);
    try result.inPlaceSub(allocator, Tensor, tmp1);
    try result.inPlaceMul(allocator, f64, val);
    const ctx = try allocator.create(VarCtx);
    ctx.* = .{ .val = val, .axes = try allocator.alloc(Dim, axes.len) };
    @memcpy(ctx.axes, axes);
    return Variable.initWithInputs(allocator, result, &.{try in.clone(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn norm(allocator: std.mem.Allocator, input: *Variable, axes: []const Dim, p: f64, keep_dims: bool) !*Variable {
    const NormCtx = struct { sumap: Tensor, p: f64, axes: []Dim, keep_dims: bool };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            var norm_ctx: *NormCtx = @ptrCast(@alignCast(ctx));
            norm_ctx.sumap.deinit();
            alloc.free(norm_ctx.axes);
            alloc.destroy(norm_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const norm_ctx: *NormCtx = @ptrCast(@alignCast(c));
            var gvar = try Variable.init(alloc, try zt.tensor.power(alloc, Tensor, norm_ctx.sumap, f64, 1 - 1 / norm_ctx.p), false);
            defer gvar.deinit();
            var tmp1 = try abs(alloc, inputs[0]);
            defer tmp1.deinit();
            var tmp2 = try pow(alloc, tmp1, norm_ctx.p - 2);
            defer tmp2.deinit();
            var lhs = try zt.tensor.mul(alloc, Tensor, inputs[0].tensor(), Tensor, tmp2.tensor());
            defer lhs.deinit();
            var tmp3 = try detail.expandFromReduction(alloc, Tensor, grad_output.tensor(), norm_ctx.axes, norm_ctx.keep_dims);
            defer tmp3.deinit();
            try tmp3.inPlaceDiv(alloc, Tensor, gvar.tensor());
            var rhs = try detail.tileAs(alloc, tmp3, try inputs[0].shape(alloc));
            defer rhs.deinit();
            var tmp_var = try Variable.init(alloc, try zt.tensor.mul(alloc, Tensor, lhs, Tensor, rhs), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
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
    var tmp3 = try zt.tensor.sum(allocator, tmp2, axes, keep_dims);
    defer tmp3.deinit();

    const sumap = try detail.expandFromReduction(allocator, Tensor, tmp3, axes, keep_dims);
    const result = try zt.tensor.power(allocator, Tensor, tmp3, f64, 1 / p);
    try zt.tensor.eval(allocator, result);
    const ctx = try allocator.create(NormCtx);
    ctx.* = .{ .sumap = sumap, .p = p, .axes = try allocator.alloc(Dim, axes.len), .keep_dims = keep_dims };
    @memcpy(ctx.axes, axes);
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn normalize(allocator: std.mem.Allocator, in: *Variable, axes: []const Dim, p: f64, eps: f64) !*Variable {
    var adj = try detail.adjustInputType(allocator, *Variable, in, @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    var tmp_norm = try norm(allocator, adj.res, axes, p, false);
    defer tmp_norm.deinit();
    var inv_scale = try max(allocator, *Variable, tmp_norm, f64, eps);
    defer inv_scale.deinit();
    var tmp = try tileAs(allocator, inv_scale, *Variable, adj.res);
    defer tmp.deinit();
    return div(allocator, *Variable, adj.res, *Variable, tmp);
}

pub fn matmul(allocator: std.mem.Allocator, lhs: *Variable, rhs: *Variable) !*Variable {
    try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
    // lhs:Input[0] -- [M, N]
    // rhs:Input[1] -- [N, K]
    // matmul(lhs, rhs)
    // -- matmul([M, N], [N, K]) --  [M, K]
    // result:gradOutput -- [M, K]
    const result = try zt.tensor.matmul(allocator, lhs.tensor(), rhs.tensor(), .None, .None);
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            if (inputs[0].isCalcGrad()) {
                var lhs_init = false;
                var _lhs = grad_output.tensor();
                defer if (lhs_init) _lhs.deinit();
                if (try _lhs.ndim(alloc) == 1) {
                    _lhs = try zt.tensor.reshape(alloc, _lhs, &.{ 1, try _lhs.dim(alloc, 0) });
                    lhs_init = true;
                }
                var rhs_init = false;
                var _rhs = inputs[1].tensor();
                defer if (rhs_init) _rhs.deinit();
                if (try _rhs.ndim(alloc) == 1) {
                    _rhs = try zt.tensor.reshape(alloc, _rhs, &.{ try _rhs.dim(alloc, 0), 1 });
                    rhs_init = true;
                }

                // matmulNT(gradOutput, inputs[1])
                // -- matmulNT([M, K], [N, K])
                // -- matmul([M, K], [K, N]) -- [M, K]
                var val = try zt.tensor.matmul(alloc, _lhs, _rhs, .None, .Transpose);
                defer val.deinit();
                var tmp = try Variable.init(alloc, try detail.sumAs(alloc, val, try inputs[0].shape(alloc)), false);
                defer tmp.deinit();
                try inputs[0].addGrad(alloc, tmp);
            }
            if (inputs[1].isCalcGrad()) {
                var lhs_init = false;
                var _lhs = inputs[0].tensor();
                defer if (lhs_init) _lhs.deinit();
                if (try _lhs.ndim(alloc) == 1) {
                    _lhs = try zt.tensor.reshape(alloc, _lhs, &.{ 1, try _lhs.dim(alloc, 0) });
                    lhs_init = true;
                }
                var rhs_init = false;
                var _rhs = grad_output.tensor();
                defer if (rhs_init) _rhs.deinit();
                if (try _rhs.ndim(alloc) == 1) {
                    _rhs = try zt.tensor.reshape(alloc, _rhs, &.{ try _rhs.dim(alloc, 0), 1 });
                    rhs_init = true;
                }

                // matmulTN(inputs[0], gradOutput)
                // -- matmulTN([M, N], [M, K])
                // -- matmul([N, M], [M, K]) -- [N, K]
                var val = try zt.tensor.matmul(alloc, _lhs, _rhs, .Transpose, .None);
                defer val.deinit();
                var tmp = try Variable.init(alloc, try detail.sumAs(alloc, val, try inputs[1].shape(alloc)), false);
                defer tmp.deinit();
                try inputs[1].addGrad(alloc, tmp);
            }
        }
    }).call;
    return Variable.initWithInputs(allocator, result, &.{ try lhs.clone(allocator), try rhs.clone(allocator) }, gradFunc, null, null);
}

pub fn matmulTN(allocator: std.mem.Allocator, lhs: *Variable, rhs: *Variable) !*Variable {
    try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
    // lhs:Input[0] -- [N, M]
    // rhs:Input[1] -- [N, K]
    // matmulTN(lhs, rhs)
    // -- matmulTN([N, M], [N, K])
    // -- matmul([M, N], [N, K]) -- [M, K]
    // result:gradOutput -- [M, K]
    const result = try zt.tensor.matmul(allocator, lhs.tensor(), rhs.tensor(), .Transpose, .None);
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            if (inputs[0].isCalcGrad()) {
                // matmulNT(inputs[1], gradOutput)
                // -- matmulNT([N, K], [M, K])
                // -- matmul([N, K], [K, M]) -- [N, M]
                var val = try zt.tensor.matmul(alloc, inputs[1].tensor(), grad_output.tensor(), .None, .Transpose);
                defer val.deinit();
                var tmp = try Variable.init(alloc, try detail.sumAs(alloc, val, try inputs[0].shape(alloc)), false);
                defer tmp.deinit();
                try inputs[0].addGrad(alloc, tmp);
            }
            if (inputs[1].isCalcGrad()) {
                // matmul(inputs[0], gradOutput)
                // -- matmulNT([N, M], [M, K]) -- [N, K]
                var val = try zt.tensor.matmul(alloc, inputs[0].tensor(), grad_output.tensor(), .None, .None);
                defer val.deinit();
                var tmp = try Variable.init(alloc, try detail.sumAs(alloc, val, try inputs[1].shape(alloc)), false);
                defer tmp.deinit();
                try inputs[1].addGrad(alloc, tmp);
            }
        }
    }).call;
    return Variable.initWithInputs(allocator, result, &.{ try lhs.clone(allocator), try rhs.clone(allocator) }, gradFunc, null, null);
}

pub fn matmulNT(allocator: std.mem.Allocator, lhs: *Variable, rhs: *Variable) !*Variable {
    try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ lhs, rhs });
    // lhs:Input[0] -- [M, N]
    // rhs:Input[1] -- [K, N]
    // matmulNT(lhs, rhs)
    // -- matmulNT([M, N], [K, N])
    // -- matmul([M, N], [N, K]) -- [M, K]
    // result:gradOutput -- [M, K]
    const result = try zt.tensor.matmul(allocator, lhs.tensor(), rhs.tensor(), .None, .Transpose);
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            if (inputs[0].isCalcGrad()) {
                // matmul(gradOutput, inputs[1])
                // -- matmul([M, K], [K, N]) -- [M, N]
                var val = try zt.tensor.matmul(alloc, grad_output.tensor(), inputs[1].tensor(), .None, .None);
                defer val.deinit();
                var tmp = try Variable.init(alloc, try detail.sumAs(alloc, val, try inputs[0].shape(alloc)), false);
                defer tmp.deinit();
                try inputs[0].addGrad(alloc, tmp);
            }
            if (inputs[1].isCalcGrad()) {
                // matmulTN(gradOutput, inputs[0])
                // -- matmulTN([M, K], [M, N])
                // -- matmul([K, M], [M, N]) -- [K, N]
                var val = try zt.tensor.matmul(alloc, grad_output.tensor(), inputs[0].tensor(), .Transpose, .None);
                defer val.deinit();
                var tmp = try Variable.init(alloc, try detail.sumAs(alloc, val, try inputs[1].shape(alloc)), false);
                defer tmp.deinit();
                try inputs[1].addGrad(alloc, tmp);
            }
        }
    }).call;
    return Variable.initWithInputs(allocator, result, &.{ try lhs.clone(allocator), try rhs.clone(allocator) }, gradFunc, null, null);
}

pub fn abs(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            // Convert it into -1, 0, 1
            var sign = try zt.tensor.sign(alloc, inputs[0].tensor());
            try sign.inPlaceMul(alloc, Tensor, grad_output.tensor());
            var tmp_var = try Variable.init(alloc, sign, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    return Variable.initWithInputs(allocator, try zt.tensor.abs(allocator, input.tensor()), &.{try input.clone(allocator)}, gradFunc, null, null);
}

pub fn flat(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    const DimsCtx = struct { dims: []Dim };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const transpose_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
            alloc.free(transpose_ctx.dims);
            alloc.destroy(transpose_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const flat_ctx: *DimsCtx = @ptrCast(@alignCast(c));
            var tmp = try Variable.init(allocator, try zt.tensor.reshape(alloc, grad_output.tensor(), flat_ctx.dims), false);
            defer tmp.deinit();
            try inputs[0].addGrad(alloc, tmp);
        }
    }).call;
    const result = try input.tensor().flatten(allocator);
    const idims = try input.shape(allocator);
    const ctx = try allocator.create(DimsCtx);
    ctx.* = .{ .dims = try allocator.alloc(Dim, idims.len) };
    @memcpy(ctx.dims, idims);
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn moddims(allocator: std.mem.Allocator, input: *Variable, dims: Shape) !*Variable {
    if (try input.ndim(allocator) == 0) {
        return input.clone(allocator);
    }
    var infer_dims = try allocator.alloc(Dim, dims.len);
    defer allocator.free(infer_dims);
    @memcpy(infer_dims, dims);
    const max_ndims: usize = @max(try input.ndim(allocator), zt.tensor.shape.ndim(dims));

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

    const DimsCtx = struct { dims: []Dim };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const transpose_ctx: *DimsCtx = @ptrCast(@alignCast(ctx));
            alloc.free(transpose_ctx.dims);
            alloc.destroy(transpose_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const moddims_ctx: *DimsCtx = @ptrCast(@alignCast(c));
            var tmp_mod = try moddims(alloc, grad_output, moddims_ctx.dims);
            defer tmp_mod.deinit();
            var tmp_var = try Variable.initSharedData(alloc, tmp_mod.shared_data.retain(), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;

    const result = try zt.tensor.reshape(allocator, input.tensor(), infer_dims);
    const in_dims = try input.shape(allocator);
    const ctx = try allocator.create(DimsCtx);
    ctx.* = .{ .dims = try allocator.alloc(Dim, in_dims.len) };
    @memcpy(ctx.dims, in_dims);
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn softmax(allocator: std.mem.Allocator, input: *Variable, dim: Dim) !*Variable {
    const SoftmaxCtx = struct { dim: Dim, tile_dims: []const Dim, result: Tensor };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const softmax_ctx: *SoftmaxCtx = @ptrCast(@alignCast(ctx));
            alloc.free(softmax_ctx.tile_dims);
            alloc.destroy(softmax_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const softmax_ctx: *SoftmaxCtx = @ptrCast(@alignCast(c));
            var rbyg = try zt.tensor.mul(alloc, Tensor, grad_output.tensor(), Tensor, softmax_ctx.result);
            defer rbyg.deinit();
            var tmp1 = try zt.tensor.sum(alloc, rbyg, &.{softmax_ctx.dim}, true);
            defer tmp1.deinit();
            var tmp2 = try zt.tensor.tile(alloc, tmp1, softmax_ctx.tile_dims);
            defer tmp2.deinit();
            var tmp3 = try zt.tensor.mul(alloc, Tensor, softmax_ctx.result, Tensor, tmp2);
            defer tmp3.deinit();
            var grad_sm = try zt.tensor.sub(alloc, Tensor, rbyg, Tensor, tmp3);
            defer grad_sm.deinit();
            var tmp_var = try Variable.init(alloc, try grad_sm.astype(alloc, try inputs[0].dtype(alloc)), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;

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
    const result = try zt.tensor.div(allocator, Tensor, exp_input, Tensor, tmp4);
    try zt.tensor.eval(allocator, result);
    const ctx = try allocator.create(SoftmaxCtx);
    ctx.* = .{ .dim = dim, .tile_dims = tile_dims, .result = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn logSoftmax(allocator: std.mem.Allocator, input: *Variable, dim: Dim) !*Variable {
    const SoftmaxCtx = struct { dim: Dim, tile_dims: []const Dim, result: Tensor };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const softmax_ctx: *SoftmaxCtx = @ptrCast(@alignCast(ctx));
            alloc.free(softmax_ctx.tile_dims);
            alloc.destroy(softmax_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const softmax_ctx: *SoftmaxCtx = @ptrCast(@alignCast(c));
            var tmp1 = try zt.tensor.sum(alloc, grad_output.tensor(), &.{softmax_ctx.dim}, true);
            defer tmp1.deinit();
            var tmp2 = try zt.tensor.tile(alloc, tmp1, softmax_ctx.tile_dims);
            defer tmp2.deinit();
            var tmp3 = try zt.tensor.exp(alloc, softmax_ctx.result);
            defer tmp3.deinit();
            try tmp3.inPlaceMul(alloc, Tensor, tmp2);
            var tmp4 = try zt.tensor.sub(alloc, Tensor, grad_output.tensor(), Tensor, tmp3);
            defer tmp4.deinit();
            var tmp_var = try Variable.init(alloc, try tmp4.astype(alloc, try inputs[0].dtype(alloc)), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
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
    const result = try zt.tensor.sub(allocator, Tensor, adj.res, Tensor, tmp6);
    try zt.tensor.eval(allocator, result);
    const ctx = try allocator.create(SoftmaxCtx);
    ctx.* = .{ .dim = dim, .tile_dims = tile_dims, .result = result };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
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

pub fn categoricalCrossEntropy(allocator: std.mem.Allocator, in: *Variable, targets: *Variable, reduction: zt.common.ReduceMode, ignore_idx: i64) !*Variable {
    const CatCrossEntropyCtx = struct {
        C: Dim,
        X: Dim,
        mask: Tensor,
        ignore_mask: Tensor,
        denominator: Tensor = undefined,
        reduction: zt.common.ReduceMode,
        input_dims: []Dim,
    };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            var cat_ctx: *CatCrossEntropyCtx = @ptrCast(@alignCast(ctx));
            cat_ctx.mask.deinit();
            cat_ctx.ignore_mask.deinit();
            if (cat_ctx.reduction == .Mean) {
                cat_ctx.denominator.deinit();
            }
            alloc.free(cat_ctx.input_dims);
            alloc.destroy(cat_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
            const cat_ctx: *CatCrossEntropyCtx = @ptrCast(@alignCast(ctx));
            var tmp_grad1: Tensor = undefined;
            switch (cat_ctx.reduction) {
                .None => tmp_grad1 = try zt.tensor.reshape(alloc, grad_output.tensor(), &.{cat_ctx.X}),
                .Mean => {
                    var tmp = try zt.tensor.div(alloc, Tensor, grad_output.tensor(), Tensor, cat_ctx.denominator);
                    defer tmp.deinit();
                    tmp_grad1 = try zt.tensor.tile(alloc, tmp, &.{cat_ctx.X});
                },
                .Sum => tmp_grad1 = try zt.tensor.tile(alloc, grad_output.tensor(), &.{cat_ctx.X}),
            }
            // [1 X]
            try tmp_grad1.indexAssign(alloc, f64, 0, &.{Index.initTensor(cat_ctx.ignore_mask)});
            var tmp_grad2 = try zt.tensor.reshape(alloc, tmp_grad1, &.{ 1, cat_ctx.X });
            tmp_grad1.deinit();
            var grad = try zt.tensor.tile(alloc, tmp_grad2, &.{cat_ctx.C});
            tmp_grad2.deinit();
            try grad.inPlaceMul(alloc, Tensor, cat_ctx.mask);
            var tmp_var = try Variable.init(alloc, try zt.tensor.reshape(alloc, grad, cat_ctx.input_dims), false);
            defer tmp_var.deinit();
            grad.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
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
    const ignore_mask = try tmp_ignore_mask.flatten(allocator);
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

    const input_dims = try input.shape(allocator);
    var ctx = try allocator.create(CatCrossEntropyCtx);
    ctx.* = .{ .X = X, .C = C, .mask = mask, .ignore_mask = ignore_mask, .reduction = reduction, .input_dims = try allocator.alloc(Dim, input_dims.len) };
    @memcpy(ctx.input_dims, input_dims);
    if (reduction == .Mean) ctx.denominator = denominator;

    return Variable.initWithInputs(allocator, result, &.{ try input.withoutData(allocator), try targets.clone(allocator) }, gradFunc, ctx, freeCtx);
}

pub fn weightedCategoricalCrossEntropy(allocator: std.mem.Allocator, input: *Variable, targets: *Variable, weight: *Variable, ignore_idx: i64) !*Variable {
    const WeightCatCrossEntropyCtx = struct {
        C: Dim,
        X: Dim,
        mask: Tensor,
        ignore_mask: Tensor,
        denominator: *Variable,
        input_dims: Shape,
    };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            var cat_ctx: *WeightCatCrossEntropyCtx = @ptrCast(@alignCast(ctx));
            cat_ctx.mask.deinit();
            cat_ctx.ignore_mask.deinit();
            cat_ctx.denominator.deinit();
            alloc.free(cat_ctx.input_dims);
            alloc.destroy(cat_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
            var cat_ctx: *WeightCatCrossEntropyCtx = @ptrCast(@alignCast(ctx));
            var tmp_grad1 = zt.tensor.div(alloc, Tensor, grad_output.tensor(), Tensor, cat_ctx.denominator.tensor());
            var tmp_grad2 = try zt.tensor.tile(alloc, tmp_grad1, &.{ 1, cat_ctx.X });
            tmp_grad1.deinit();

            const weight_tensor = cat_ctx.weight.tensor();
            try tmp_grad2.inPlaceMul(alloc, Tensor, cat_ctx.ignore_mask);
            var tmp_grad3 = try zt.tensor.tile(alloc, tmp_grad2, &.{cat_ctx.C});
            tmp_grad2.deinit();
            try tmp_grad3.inPlaceMul(alloc, Tensor, cat_ctx.mask);
            var tmp_grad4 = try zt.tensor.reshape(alloc, tmp_grad3, cat_ctx.input_dims);
            tmp_grad3.deinit();
            tmp_grad4.inPlaceMul(alloc, Tensor, weight_tensor);
            var tmp_var = try Variable.init(alloc, tmp_grad4, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
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
    const denominator = try Variable.init(allocator, try zt.tensor.sum(allocator, weight_sum, &.{ 0, 1 }, false), false);
    weight_sum.deinit();

    var tmp_res1 = try zt.tensor.mul(allocator, Tensor, mask, Tensor, x);
    x.deinit();
    var tmp_ignore_mask = try zt.tensor.neq(allocator, Tensor, y, i64, ignore_idx);
    y.deinit();
    const ignore_mask = try tmp_ignore_mask.astype(allocator, .s32); // [1, X]
    tmp_ignore_mask.deinit();
    var tmp_res2 = try zt.tensor.sum(allocator, tmp_res1, &.{0}, true); // [1, X]
    tmp_res1.deinit();
    try tmp_res2.inPlaceMul(allocator, Tensor, ignore_mask);
    const result = try zt.tensor.sum(allocator, tmp_res2, &.{1}, true);
    tmp_res2.deinit();

    const input_dims = try input.shape(allocator);
    const ctx = try allocator.create(WeightCatCrossEntropyCtx);
    ctx.* = .{ .C = C, .X = X, .mask = mask, .ignore_mask = ignore_mask, .denominator = denominator, .input_dims = try allocator.alloc(Dim, input_dims.len) };
    @memcpy(ctx.input_dims, input_dims);
    return Variable.initWithInputs(allocator, result, &.{ try input.withoutData(allocator), try targets.clone(allocator), try weight.clone(allocator) }, gradFunc, ctx, freeCtx);
}

pub fn reorder(allocator: std.mem.Allocator, input: *Variable, shape: Shape) !*Variable {
    const DimGradTuple = std.meta.Tuple(&.{ Dim, usize });
    const ReorderCtx = struct { dim_grad: []DimGradTuple };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const reorder_ctx: *ReorderCtx = @ptrCast(@alignCast(ctx));
            alloc.free(reorder_ctx.dim_grad);
            alloc.destroy(reorder_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
            const reorder_ctx: *ReorderCtx = @ptrCast(@alignCast(ctx));
            const dim_grad = reorder_ctx.dim_grad;
            var reordered = try alloc.alloc(Dim, dim_grad.len);
            defer alloc.free(reordered);
            for (0..dim_grad.len) |i| {
                reordered[i] = @intCast(dim_grad[i][1]);
            }
            var tmp_var = try Variable.init(alloc, try zt.tensor.transpose(alloc, grad_output.tensor(), reordered), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    var result = try zt.tensor.transpose(allocator, input.tensor(), shape);
    if (!try result.isContiguous(allocator)) {
        var tmp = result;
        defer tmp.deinit();
        result = try tmp.asContiguousTensor(allocator);
    }

    var dim_grad = try allocator.alloc(DimGradTuple, zt.tensor.shape.ndim(shape));
    for (0..zt.tensor.shape.ndim(shape)) |i| {
        dim_grad[i] = .{ try zt.tensor.shape.dim(shape, i), i };
    }

    std.mem.sort(DimGradTuple, dim_grad, {}, (struct {
        pub fn lessThan(_: void, a: DimGradTuple, b: DimGradTuple) bool {
            return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]);
        }
    }).lessThan);
    const ctx = try allocator.create(ReorderCtx);
    ctx.* = .{ .dim_grad = dim_grad };

    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn linearNoBias(allocator: std.mem.Allocator, input: *Variable, weight: *Variable) !*Variable {
    var tmp = try Tensor.initEmpty(allocator);
    defer tmp.deinit();
    var dummy_bias = try Variable.init(allocator, try tmp.astype(allocator, try input.dtype(allocator)), false);
    defer dummy_bias.deinit();
    return linear(allocator, input, weight, dummy_bias);
}

pub fn linear(allocator: std.mem.Allocator, in: *Variable, wt: *Variable, bs: *Variable) !*Variable {
    try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ in, wt, bs });
    const LinearCtx = struct { has_bias: bool };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const linear_ctx: *LinearCtx = @ptrCast(@alignCast(ctx));
            alloc.destroy(linear_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
            const linear_ctx: *LinearCtx = @ptrCast(@alignCast(ctx));
            const has_bias = linear_ctx.has_bias;
            var in_ = inputs[0];
            var wt_ = inputs[1];
            var wt_tensor = wt_.tensor();

            const nframes: Dim = @divTrunc(try in_.elements(alloc), try in_.dim(alloc, 0));

            if (has_bias and inputs[2].isCalcGrad()) {
                var bs_ = inputs[2];
                var bias_grad = try sumAs(alloc, grad_output, *Variable, bs_);
                defer bias_grad.deinit();
                var tmp_var = try Variable.initSharedData(alloc, bias_grad.shared_data.retain(), false);
                defer tmp_var.deinit();
                try bs_.addGrad(alloc, tmp_var);
            }
            if (in_.isCalcGrad()) {
                const to2dout: Shape = &.{ try wt_tensor.dim(alloc, 0), nframes };
                var tmp1 = try moddims(alloc, grad_output, to2dout);
                defer tmp1.deinit();
                var tmp2 = try matmulTN(alloc, wt_, tmp1);
                defer tmp2.deinit();
                var tmp3 = try moddims(alloc, tmp2, try in_.shape(alloc));
                defer tmp3.deinit();
                var tmp_var = try Variable.initSharedData(alloc, tmp3.shared_data.retain(), false);
                defer tmp_var.deinit();
                try in_.addGrad(alloc, tmp_var);
            }
            if (wt_.isCalcGrad()) {
                const to2din: Shape = &.{ try wt_tensor.dim(alloc, 1), nframes };
                const to2dout: Shape = &.{ try wt_tensor.dim(alloc, 0), nframes };
                var lhs = try moddims(alloc, grad_output, to2dout);
                defer lhs.deinit();
                var rhs = try moddims(alloc, in_, to2din);
                defer rhs.deinit();
                var wt_grad = try matmulNT(alloc, lhs, rhs);
                defer wt_grad.deinit();
                var tmp_var = try Variable.initSharedData(alloc, wt_grad.shared_data.retain(), false);
                defer tmp_var.deinit();
                try wt_.addGrad(alloc, tmp_var);
            }
        }
    }).call;
    const adj_in = try detail.adjustInputType(allocator, *Variable, in, @src().fn_name);
    var input = adj_in.res;
    defer if (adj_in.allocated) input.deinit();
    const adj_wt = try detail.adjustInputType(allocator, *Variable, wt, @src().fn_name);
    var weight = adj_wt.res;
    defer if (adj_wt.allocated) weight.deinit();
    const adj_bs = try detail.adjustInputType(allocator, *Variable, bs, @src().fn_name);
    var bias = adj_bs.res;
    defer if (adj_bs.allocated) bias.deinit();

    const to2d: Shape = &.{ try input.dim(allocator, 0), @divTrunc(try input.elements(allocator), try input.dim(allocator, 0)) };
    var to4d = try allocator.alloc(Dim, try input.ndim(allocator));
    defer allocator.free(to4d);
    @memcpy(to4d, try input.shape(allocator));
    to4d[0] = try weight.tensor().dim(allocator, 0);

    var matmul_rhs = try zt.tensor.reshape(allocator, input.tensor(), to2d);
    var tmp_matmul = try zt.tensor.matmul(allocator, weight.tensor(), matmul_rhs, .None, .None);
    matmul_rhs.deinit();
    var output = try zt.tensor.reshape(allocator, tmp_matmul, to4d);
    tmp_matmul.deinit();

    const has_bias = try bias.elements(allocator) > 0;
    if (has_bias) {
        const out_shape = try output.shape(allocator);
        var tile_dims = try allocator.alloc(Dim, out_shape.len);
        defer allocator.free(tile_dims);
        @memcpy(tile_dims, out_shape);
        tile_dims[0] = 1;
        var tmp_tile = try zt.tensor.tile(allocator, bias.tensor(), tile_dims);
        defer tmp_tile.deinit();
        try output.inPlaceAdd(allocator, Tensor, tmp_tile);
    }

    const ctx = try allocator.create(LinearCtx);
    ctx.* = .{ .has_bias = has_bias };
    if (has_bias) {
        return Variable.initWithInputs(allocator, output, &.{ try input.clone(allocator), try weight.clone(allocator), try bias.clone(allocator) }, gradFunc, ctx, freeCtx);
    }
    return Variable.initWithInputs(allocator, output, &.{ try input.clone(allocator), try weight.clone(allocator) }, gradFunc, ctx, freeCtx);
}

pub fn conv2dNoBias(
    allocator: std.mem.Allocator,
    in: *Variable,
    wt: *Variable,
    sx: i64,
    sy: i64,
    px: i64,
    py: i64,
    dx: i64,
    dy: i64,
    groups: i64,
    benchmarks: ?zigrc.Arc(ConvBenchmarks),
) !*Variable {
    var dummy_bias = try Variable.init(allocator, try Tensor.initHandle(allocator, &.{0}, try in.dtype(allocator)), false);
    defer dummy_bias.deinit();
    return conv2d(allocator, in, wt, dummy_bias, sx, sy, px, py, dx, dy, groups, benchmarks);
}

pub fn conv2d(
    allocator: std.mem.Allocator,
    in: *Variable,
    wt: *Variable,
    bs: *Variable,
    sx: i64,
    sy: i64,
    px: i64,
    py: i64,
    dx: i64,
    dy: i64,
    groups: i64,
    benchmarks: ?zigrc.Arc(ConvBenchmarks),
) !*Variable {
    try ztVariableDTypesMatch(allocator, @src().fn_name, &.{ in, wt, bs });

    var payload = try detail.createAutogradPayload(allocator, &.{ in, wt, bs });
    defer if (payload) |p| p.release();

    const has_bias = !try bs.isEmpty(allocator);

    var adj_in = try detail.adjustInputType(allocator, *Variable, in, @src().fn_name);
    defer if (adj_in.allocated) adj_in.res.deinit();
    var input = adj_in.res;
    var adj_wt = try detail.adjustInputType(allocator, *Variable, wt, @src().fn_name);
    defer if (adj_wt.allocated) adj_wt.res.deinit();
    var weights = adj_wt.res;
    var adj_bs = try detail.adjustInputType(allocator, *Variable, bs, @src().fn_name);
    defer if (adj_bs.allocated) adj_bs.res.deinit();
    var bias = adj_bs.res;

    const output = try zt.autograd.op_details.conv2d(
        allocator,
        input.tensor(),
        weights.tensor(),
        bias.tensor(),
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        groups,
        payload,
    );
    const Ctx = struct { sx: i64, sy: i64, px: i64, py: i64, dx: i64, dy: i64, has_bias: bool, groups: i64, benchmarks: ?zigrc.Arc(ConvBenchmarks), payload: ?zigrc.Arc(AutogradPayload) };
    const ctx = try allocator.create(Ctx);
    ctx.* = .{ .sx = sx, .sy = sy, .px = px, .py = py, .dx = dx, .dy = dy, .has_bias = has_bias, .groups = groups, .benchmarks = benchmarks, .payload = if (payload) |*p| p.retain() else null };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, c: *anyopaque) void {
            const context: *Ctx = @ptrCast(@alignCast(c));
            if (context.payload) |p| p.release();
            alloc.destroy(context);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, c: ?*anyopaque) !void {
            const context: *Ctx = @ptrCast(@alignCast(c));

            const autograd_extension = try (try inputs[0].tensor().backend(alloc)).getExtension(alloc, AutogradExtension);
            _ = autograd_extension;

            // TODO: Create benchmarks if needed
            const data_bench: ?zigrc.Arc(DynamicBenchmark) = null;
            const filter_bench: ?zigrc.Arc(DynamicBenchmark) = null;
            const bias_bench: ?zigrc.Arc(DynamicBenchmark) = null;

            // Bias gradients
            const compute_bias_grad = inputs.len > 2 and inputs[2].isCalcGrad();
            var bs_: Tensor = undefined;
            var bs_alloc = false;
            if (context.has_bias and compute_bias_grad) {
                bs_ = inputs[2].tensor();
            } else {
                bs_ = try Tensor.initEmpty(alloc);
                bs_alloc = true;
            }
            defer if (bs_alloc) bs_.deinit();

            const in_ = inputs[0].tensor();
            const wt_ = inputs[1].tensor();

            // Data (input) gradients
            if (inputs[0].isCalcGrad()) {
                const data_grad = try (try (try in_.backend(alloc)).getExtension(alloc, AutogradExtension)).conv2dBackwardData(
                    alloc,
                    grad_output.tensor(),
                    in_,
                    wt_,
                    context.sx,
                    context.sy,
                    context.px,
                    context.py,
                    context.dx,
                    context.dy,
                    context.groups,
                    data_bench,
                    context.payload,
                );
                var tmp_var = try Variable.init(alloc, data_grad, false);
                defer tmp_var.deinit();
                try inputs[0].addGrad(alloc, tmp_var);
            }

            // Filter (weight) and bias gradients
            if (inputs[1].isCalcGrad() or compute_bias_grad) {
                const tmp_res = try (try (try wt_.backend(alloc)).getExtension(alloc, AutogradExtension)).conv2dBackwardFilterBias(
                    alloc,
                    grad_output.tensor(),
                    in_,
                    wt_,
                    bs_,
                    context.sx,
                    context.sy,
                    context.px,
                    context.py,
                    context.dx,
                    context.dy,
                    context.groups,
                    filter_bench,
                    bias_bench,
                    context.payload,
                );
                var filter_grad = try Variable.init(alloc, tmp_res[0], false); // filter/weight
                defer filter_grad.deinit();
                if (inputs[1].isCalcGrad()) {
                    try inputs[1].addGrad(alloc, filter_grad);
                }
                var bias_grad = try Variable.init(alloc, tmp_res[1], false); // filter/weight
                defer bias_grad.deinit();
                if (compute_bias_grad) {
                    try inputs[2].addGrad(alloc, bias_grad);
                }
            }
        }
    }).call;
    if (has_bias) {
        return Variable.initWithInputs(allocator, output, &.{ try input.clone(allocator), try weights.clone(allocator), try bias.clone(allocator) }, gradFunc, ctx, freeCtx);
    }
    return Variable.initWithInputs(allocator, output, &.{ try input.clone(allocator), try weights.clone(allocator) }, gradFunc, ctx, freeCtx);
}

pub fn pool2d(
    allocator: std.mem.Allocator,
    input: *Variable,
    wx: i64,
    wy: i64,
    sx: i64,
    sy: i64,
    px: i64,
    py: i64,
    mode: PoolingMode,
) !*Variable {
    const payload = try detail.createAutogradPayload(allocator, &.{input});
    const output = try zt.autograd.op_details.pool2d(allocator, input.tensor(), wx, wy, sx, sy, px, py, mode, payload);
    const Ctx = struct { wx: i64, wy: i64, sx: i64, sy: i64, px: i64, py: i64, mode: PoolingMode, output: Tensor, payload: ?zigrc.Arc(AutogradPayload) };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, context: *anyopaque) void {
            const c: *Ctx = @ptrCast(@alignCast(context));
            alloc.destroy(c);
        }
    }).call;
    const ctx = try allocator.create(Ctx);
    ctx.* = .{ .wx = wx, .wy = wy, .sx = sx, .sy = sy, .px = px, .py = py, .mode = mode, .output = output, .payload = payload };
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, context: ?*anyopaque) !void {
            var in = inputs[0];
            if (!in.isCalcGrad()) {
                return;
            }

            const c: *Ctx = @ptrCast(@alignCast(context));

            var autograd_extension = try (try in.tensor().backend(alloc)).getExtension(alloc, AutogradExtension);
            var tmp_var = try Variable.init(alloc, try autograd_extension.pool2dBackward(alloc, grad_output.tensor(), in.tensor(), c.output, c.wx, c.wy, c.sx, c.sy, c.px, c.py, c.mode, c.payload), false);
            defer tmp_var.deinit();
            try in.addGrad(alloc, tmp_var);
        }
    }).call;
    return Variable.initWithInputs(allocator, output, &.{try input.clone(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn batchnorm(
    allocator: std.mem.Allocator,
    _input: *Variable,
    weight: *Variable,
    bias: *Variable,
    running_mean: *Variable,
    running_var: *Variable,
    axes: []const i64,
    train: bool,
    momentum: f64,
    epsilon: f64,
) !*Variable {
    const payload = try detail.createAutogradPayload(allocator, &.{ _input, weight, bias });
    errdefer if (payload) |p| p.release();
    var adj_in = try detail.adjustInputType(allocator, *Variable, _input, @src().fn_name);
    defer if (adj_in.allocated) adj_in.res.deinit();
    var input = adj_in.res;

    var save_mean = try Tensor.initEmpty(allocator);
    errdefer save_mean.deinit();
    var save_var = try Tensor.initEmpty(allocator);
    errdefer save_var.deinit();

    const output = try zt.autograd.op_details.batchnorm(
        allocator,
        save_mean,
        save_var,
        input.tensor(),
        weight.tensor(),
        bias.tensor(),
        running_mean.tensor(),
        running_var.tensor(),
        axes,
        train,
        momentum,
        @floatCast(epsilon),
        payload,
    );
    const Ctx = struct {
        save_mean: Tensor,
        save_var: Tensor,
        train: bool,
        axes: []const i64,
        epsilon: f64,
        payload: ?zigrc.Arc(AutogradPayload),
    };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            var c: *Ctx = @ptrCast(@alignCast(ctx));
            c.save_mean.deinit();
            c.save_var.deinit();
            // TODO: need to free payload if it is not null
            alloc.destroy(c);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, _grad_output: *Variable, ctx: ?*anyopaque) !void {
            const c: *Ctx = @ptrCast(@alignCast(ctx));
            var in = inputs[0];
            var wt = inputs[1];
            var bs = inputs[2];

            var adj_grad_output = try detail.adjustInputType(alloc, *Variable, _grad_output, "batchnorm");
            defer if (adj_grad_output.allocated) adj_grad_output.res.deinit();
            var grad_output = adj_grad_output.res;

            if (!in.isCalcGrad() and !wt.isCalcGrad() and !bs.isCalcGrad()) {
                return;
            }

            var adj_grad_in = try detail.adjustInputType(alloc, Tensor, in.tensor(), "batchnorm");
            defer if (adj_grad_in.allocated) adj_grad_in.res.deinit();
            const adj_grad_in_tensor = adj_grad_in.res;

            const res = try (try (try in.tensor().backend(alloc)).getExtension(alloc, AutogradExtension)).batchnormBackward(
                alloc,
                grad_output.tensor(),
                c.save_mean,
                c.save_var,
                adj_grad_in_tensor,
                wt.tensor(),
                c.axes,
                c.train,
                @floatCast(c.epsilon),
                c.payload,
            );
            var grad_in = res[0];
            defer grad_in.deinit();
            var grad_wt = res[1];
            defer grad_wt.deinit();
            var grad_bs = res[2];
            defer grad_bs.deinit();

            var tmp_in_var = try Variable.init(alloc, try grad_in.astype(alloc, try in.dtype(alloc)), false);
            defer tmp_in_var.deinit();
            try in.addGrad(alloc, tmp_in_var);
            var tmp_wt_var = try Variable.init(alloc, try grad_wt.astype(alloc, try wt.dtype(alloc)), false);
            defer tmp_wt_var.deinit();
            try wt.addGrad(alloc, tmp_wt_var);
            if (!try bs.isEmpty(alloc)) {
                var tmp_bs_var = try Variable.init(alloc, try grad_bs.astype(alloc, try bs.dtype(alloc)), false);
                defer tmp_bs_var.deinit();
                try bs.addGrad(alloc, tmp_bs_var);
            }
        }
    }).call;

    const ctx = try allocator.create(Ctx);
    ctx.* = .{ .save_mean = save_mean, .save_var = save_var, .train = train, .axes = axes, .epsilon = epsilon, .payload = payload };

    return Variable.initWithInputs(allocator, output, &.{ try input.clone(allocator), try weight.clone(allocator), try bias.clone(allocator) }, gradFunc, ctx, freeCtx);
}

pub fn gatedlinearunit(allocator: std.mem.Allocator, input: *Variable, dim: usize) !*Variable {
    const GLUCtx = struct {
        fhalf: []Index,
        shalf: []Index,
        fhalfout: Tensor,
        shalfout: Tensor,
        in_dims: []Dim,
        in_type: zt.tensor.DType,
    };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            var c: *GLUCtx = @ptrCast(@alignCast(ctx));
            alloc.free(c.fhalf);
            alloc.free(c.shalf);
            c.fhalfout.deinit();
            c.shalfout.deinit();
            alloc.free(c.in_dims);
            alloc.destroy(c);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
            const c: *GLUCtx = @ptrCast(@alignCast(ctx));
            var grad_glu = try Tensor.initHandle(alloc, c.in_dims, c.in_type);
            var tmp1 = try zt.tensor.mul(alloc, Tensor, c.shalfout, Tensor, grad_output.tensor());
            defer tmp1.deinit();
            try grad_glu.indexAssign(alloc, Tensor, tmp1, c.fhalf);
            var tmp2 = try zt.tensor.sub(alloc, f64, 1, Tensor, c.shalfout);
            defer tmp2.deinit();
            try tmp2.inPlaceMul(alloc, Tensor, c.shalfout);
            try tmp2.inPlaceMul(alloc, Tensor, c.fhalfout);
            try tmp2.inPlaceMul(alloc, Tensor, grad_output.tensor());
            try grad_glu.indexAssign(alloc, Tensor, tmp2, c.shalf);
            var tmp_var = try Variable.init(alloc, grad_glu, false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    if (dim >= try input.ndim(allocator)) {
        std.debug.print("gatedlinearunit - passed dim is great than the number of dimensions of the input.\n", .{});
        return error.GLUInvalidDim;
    }

    const in_dims = try input.shape(allocator);
    const in_type = try input.dtype(allocator);
    const in_size = in_dims[dim];
    if (@rem(in_size, 2) == 1) {
        std.debug.print("halving dimension must be even for GLU\n", .{});
        return error.GLUHalvingDimensionNotEven;
    }

    var fhalf = try allocator.alloc(Index, try input.ndim(allocator));
    @memset(fhalf, Index.initRange(zt.tensor.span));
    var shalf = try allocator.alloc(Index, try input.ndim(allocator));
    @memset(shalf, Index.initRange(zt.tensor.span));
    fhalf[dim] = Index.initRange(Range.initEnd(@divTrunc(in_size, 2)));
    shalf[dim] = Index.initRange(Range.init(@divTrunc(in_size, 2), .{ .dim = in_size }));

    const fhalfout = try input.tensor().index(allocator, fhalf);
    var tmp_shalfout1 = try input.tensor().index(allocator, shalf);
    defer tmp_shalfout1.deinit();

    const shalfout = try zt.tensor.sigmoid(allocator, tmp_shalfout1);
    const ctx = try allocator.create(GLUCtx);
    ctx.* = .{
        .fhalf = fhalf,
        .shalf = shalf,
        .fhalfout = fhalfout,
        .shalfout = shalfout,
        .in_dims = try allocator.alloc(Dim, in_dims.len),
        .in_type = in_type,
    };
    @memcpy(ctx.in_dims, in_dims);

    return Variable.initWithInputs(
        allocator,
        try zt.tensor.mul(allocator, Tensor, fhalfout, Tensor, shalfout),
        &.{try input.withoutData(allocator)},
        gradFunc,
        ctx,
        freeCtx,
    );
}

// TODO: pub fn rnn() !*Variable {}

pub fn embedding(allocator: std.mem.Allocator, input: *Variable, embeddings: *Variable) !*Variable {
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
            var w = inputs[1];
            if (!w.isCalcGrad()) {
                return;
            }

            var ip = try inputs[0].tensor().flatten(alloc);
            defer ip.deinit();
            const size = try ip.elements(alloc);
            var deltas = try zt.tensor.reshape(alloc, grad_output.tensor(), &.{ try w.dim(alloc, 0), size });
            defer deltas.deinit();

            var sp_vals = try zt.tensor.full(alloc, &.{size}, f64, 1, try deltas.dtype(alloc));
            defer sp_vals.deinit();
            var sp_row_idxs = try zt.tensor.arange(alloc, &.{size + 1}, 0, .s32);
            defer sp_row_idxs.deinit();
            var sp_col_idxs = try ip.astype(alloc, .s32);
            defer sp_col_idxs.deinit();

            var sp = try Tensor.initSparse(
                alloc,
                try ip.elements(alloc),
                try w.dim(alloc, 1),
                sp_vals,
                sp_row_idxs,
                sp_col_idxs,
                .CSR,
            );
            defer sp.deinit();
            var rhs = try zt.tensor.transpose(alloc, deltas, &.{});
            defer rhs.deinit();
            var tmp = try zt.tensor.matmul(alloc, sp, rhs, .Transpose, .None);
            defer tmp.deinit();
            var tmp_var = try Variable.init(alloc, try zt.tensor.transpose(alloc, tmp, &.{}), false);
            defer tmp_var.deinit();
            try w.addGrad(alloc, tmp_var);
        }
    }).call;
    // TODO: {zt.tensor.Tensor}{4-dims} - relax this
    if (try input.ndim(allocator) >= 4) {
        std.debug.print("embedding input must have 3 or fewer dims\n", .{});
        return error.EmbeddingInputExceedsMaxDims;
    }

    var idxs = try input.tensor().flatten(allocator);
    defer idxs.deinit();
    const in_dims = try input.shape(allocator);
    var r_dims = try allocator.alloc(Dim, try input.ndim(allocator) + 1);
    defer allocator.free(r_dims);
    r_dims[0] = try embeddings.dim(allocator, 0);
    for (1..try input.ndim(allocator) + 1) |i| {
        r_dims[i] = in_dims[i - 1];
    }
    var tmp = try embeddings.tensor().index(allocator, &.{ Index.initRange(zt.tensor.span), Index.initTensor(idxs) });
    defer tmp.deinit();
    const result = try zt.tensor.reshape(allocator, tmp, r_dims);
    return Variable.initWithInputs(
        allocator,
        result,
        &.{ try input.clone(allocator), try embeddings.clone(allocator) },
        gradFunc,
        null,
        null,
    );
}

pub fn padding(allocator: std.mem.Allocator, input: *Variable, pad: []const [2]Dim, val: f64) !*Variable {
    const PaddingCtx = struct { in_seq: []Index };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const padding_ctx: *PaddingCtx = @ptrCast(@alignCast(ctx));
            alloc.free(padding_ctx.in_seq);
            alloc.destroy(padding_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
            const padding_ctx: *PaddingCtx = @ptrCast(@alignCast(ctx));
            var tmp_var = try Variable.init(alloc, try grad_output.tensor().index(alloc, padding_ctx.in_seq), false);
            defer tmp_var.deinit();
            try inputs[0].addGrad(alloc, tmp_var);
        }
    }).call;
    if (pad.len > try input.ndim(allocator)) {
        std.debug.print("padding: number of padding dimensions exceeds number of input dimensions\n", .{});
        return error.PaddingDimsExceedsInputDims;
    }

    var op_dims = try allocator.alloc(Dim, try input.ndim(allocator));
    defer allocator.free(op_dims);
    @memcpy(op_dims, try input.shape(allocator));
    var in_seq = try allocator.alloc(Index, try input.ndim(allocator));
    @memset(in_seq, Index.initRange(zt.tensor.span));
    for (pad, 0..) |p, i| {
        op_dims[i] += (p[0] + p[1]);
        in_seq[i] = Index.initRange(Range.init(p[0], .{ .dim = op_dims[i] - p[1] }));
    }
    var result = try zt.tensor.full(allocator, op_dims, f64, val, try input.dtype(allocator));
    try result.indexAssign(allocator, Tensor, input.tensor(), in_seq);
    const ctx = try allocator.create(PaddingCtx);
    ctx.* = .{ .in_seq = in_seq };
    return Variable.initWithInputs(allocator, result, &.{try input.withoutData(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn dropout(allocator: std.mem.Allocator, input: *Variable, p: f64) !*Variable {
    if (p > 0) {
        var tmp_rand = try zt.tensor.rand(allocator, try input.shape(allocator), try input.dtype(allocator));
        defer tmp_rand.deinit();
        var comp = try zt.tensor.greaterThan(allocator, Tensor, tmp_rand, f64, p);
        defer comp.deinit();
        var mask = try Variable.init(allocator, try comp.astype(allocator, try input.dtype(allocator)), false);
        defer mask.deinit();
        var tmp = try mul(allocator, f64, 1 / (1 - p), *Variable, mask);
        defer tmp.deinit();
        return mul(allocator, *Variable, tmp, *Variable, input);
    } else {
        return input.clone(allocator);
    }
}

pub fn relu(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    return max(allocator, *Variable, input, f64, 0);
}

pub fn gelu(allocator: std.mem.Allocator, in: *Variable) !*Variable {
    var adj = try detail.adjustInputType(allocator, *Variable, in, @src().fn_name);
    defer if (adj.allocated) adj.res.deinit();
    const input = adj.res;
    var tmp1 = try mul(allocator, *Variable, input, *Variable, input);
    defer tmp1.deinit();
    var tmp2 = try mul(allocator, *Variable, input, *Variable, tmp1);
    defer tmp2.deinit();
    var tmp3 = try mul(allocator, f64, 0.044715, *Variable, tmp2);
    defer tmp3.deinit();
    var tmp4 = try add(allocator, *Variable, input, *Variable, tmp3);
    defer tmp4.deinit();
    var tmp5 = try mul(allocator, f64, 0.7978845608, *Variable, tmp4);
    defer tmp5.deinit();
    var tmp6 = try tanh(allocator, tmp5);
    defer tmp6.deinit();
    var rhs = try add(allocator, f64, 1, *Variable, tmp6);
    defer rhs.deinit();
    var lhs = try mul(allocator, f64, 0.5, *Variable, input);
    defer lhs.deinit();
    return mul(allocator, *Variable, lhs, *Variable, rhs);
}

pub fn relativePositionEmbeddingRotate(allocator: std.mem.Allocator, input: *Variable) !*Variable {
    const EmbeddingRotateCtx = struct { d0: Dim, d1: Dim, d2: Dim };
    const freeCtx = (struct {
        pub fn call(alloc: std.mem.Allocator, ctx: *anyopaque) void {
            const embedding_rotate_ctx: *EmbeddingRotateCtx = @ptrCast(@alignCast(ctx));
            alloc.destroy(embedding_rotate_ctx);
        }
    }).call;
    const gradFunc = (struct {
        pub fn call(alloc: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
            const c: *EmbeddingRotateCtx = @ptrCast(@alignCast(ctx));
            const d0 = c.d0;
            const d1 = c.d1;
            const d2 = c.d2;
            var tmp1 = try zt.tensor.reshape(alloc, grad_output.tensor(), &.{ (d0 + d1 - 1) * d1, 1, d2 });
            defer tmp1.deinit();
            var tmp2 = try zt.tensor.full(alloc, &.{ d1, 1, d2 }, f64, 0, try grad_output.dtype(alloc));
            defer tmp2.deinit();
            var tmp3 = try zt.tensor.concatenate(alloc, &.{ grad_output.tensor(), tmp2 }, 0);
            defer tmp3.deinit();
            var tmp_var1 = try Variable.init(alloc, try zt.tensor.reshape(alloc, tmp3, &.{ d0 + d1, d1, d2 }), false);
            defer tmp_var1.deinit();
            var tmp_var2 = try tmp_var1.index(alloc, &.{Index.initRange(Range.init(0, .{ .dim = d0 }))});
            defer tmp_var2.deinit();
            var used_var = try Variable.initSharedData(alloc, tmp_var2.shared_data.retain(), false);
            defer used_var.deinit();
            try inputs[0].addGrad(alloc, used_var);
        }
    }).call;
    if (try input.ndim(allocator) != 3) {
        std.debug.print("relativePositionEmbeddingRotate - input tensor must have 3 dimensions\n", .{});
        return error.InputMustHave3Dims;
    }

    const d0 = try input.dim(allocator, 0);
    const d1 = try input.dim(allocator, 1);
    const d2 = try input.dim(allocator, 2);
    var tmp1 = try zt.tensor.full(allocator, &.{ d1, d1, d2 }, f64, 0, try input.dtype());
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.concatenate(allocator, &.{ input.tensor(), tmp1 }, 0);
    defer tmp2.deinit();
    var tmp3 = try zt.tensor.reshape(allocator, tmp2, &.{ (d0 + d1) * d1, 1, d2 });
    defer tmp3.deinit();
    var tmp4 = try tmp3.index(allocator, &.{Index.initRange(Range.init(0, .{ .dim = (d1 + d0 - 1) * d1 }))});
    defer tmp4.deinit();
    const result = try zt.tensor.reshape(allocator, tmp4, &.{ d0 + d1 - 1, d1, d2 });
    const ctx = try allocator.create(EmbeddingRotateCtx);
    ctx.* = .{ .d0 = d0, .d1 = d1, .d2 = d2 };
    return Variable.initWithInputs(allocator, result, &.{try input.clone(allocator)}, gradFunc, ctx, freeCtx);
}

pub fn multiheadAttention(
    allocator: std.mem.Allocator,
    query: *Variable,
    key: *Variable,
    value: *Variable,
    pos_emb: *Variable,
    mask: *Variable,
    pad_mask: *Variable,
    n_heads: i64,
    p_dropout: f64,
    offset: i64,
) !*Variable {
    if (try query.ndim(allocator) != 3) {
        std.debug.print("multiheadAttention - query input tensor should be 3 dimensions: Time x (n_heads * head_dim) x B\n", .{});
        return error.QueryRequires3Dims;
    }
    if (try key.ndim(allocator) != 3) {
        std.debug.print("multiheadAttention - key input tensor should be 3 dimensions: Time x (n_heads * head_dim) x B\n", .{});
        return error.KeyRequires3Dims;
    }
    if (try value.ndim(allocator) != 3) {
        std.debug.print("multiheadAttention - value input tensor should be 3 dimensions: Time x (n_heads * head_dim) x B\n", .{});
        return error.ValueRequires3Dims;
    }

    const bsz = try query.dim(allocator, 2);
    const model_dim = try query.dim(allocator, 1);
    const head_dim = @divTrunc(model_dim, n_heads);

    var tmp_q = try moddims(allocator, query, &.{ -1, head_dim, n_heads * bsz });
    defer tmp_q.deinit();
    var k = try moddims(allocator, key, &.{ -1, head_dim, n_heads * bsz });
    defer k.deinit();
    var v = try moddims(allocator, value, &.{ -1, head_dim, n_heads * bsz });
    defer v.deinit();

    var q = try div(allocator, *Variable, tmp_q, f64, @sqrt(@as(f64, @floatFromInt(head_dim))));
    defer q.deinit();
    var scores = try matmulNT(allocator, q, k);
    defer scores.deinit();
    if (!try pos_emb.isEmpty(allocator)) {
        const n = @divTrunc(try pos_emb.dim(allocator, 0), 2) - offset;
        var tmp1 = try pos_emb.astype(allocator, try q.dtype(allocator));
        defer tmp1.deinit();
        var tmp2 = try matmulNT(allocator, tmp1, q);
        defer tmp2.deinit();
        var pscores = try relativePositionEmbeddingRotate(allocator, tmp2);
        defer pscores.deinit();
        var tmp_scores = scores;
        defer tmp_scores.deinit();
        var tmp3 = try pscores.index(allocator, &.{Index.initRange(Range.init(n, .{ .dim = n + try k.dim(allocator, 0) }))});
        defer tmp3.deinit();
        var tmp4 = try transpose(allocator, tmp3, &.{ 1, 0, 2 });
        defer tmp4.deinit();
        scores = try add(allocator, *Variable, scores, *Variable, tmp4);
    }
    if (!try mask.isEmpty(allocator)) {
        var tmp_scores = scores;
        defer tmp_scores.deinit();
        var tmp1 = try mask.astype(allocator, try scores.dtype(allocator));
        defer tmp1.deinit();
        var tmp2 = try tileAs(allocator, tmp1, *Variable, scores);
        defer tmp2.deinit();
        scores = try add(allocator, *Variable, scores, *Variable, tmp2);
    }
    if (!try pad_mask.isEmpty(allocator)) {
        if (try pad_mask.dim(allocator, 0) != try query.dim(allocator, 0)) {
            std.debug.print("multiheadAttention: invalid padding mask size\n", .{});
            return error.InvalidPaddingMaskSize;
        }
        var tmp1 = try moddims(allocator, pad_mask, &.{ 1, try pad_mask.dim(allocator, 0), 1, bsz });
        defer tmp1.deinit();
        var tmp2 = try tileAs(allocator, tmp1, Shape, &.{ try pad_mask.dim(allocator, 0), try pad_mask.dim(allocator, 0), n_heads, bsz });
        defer tmp2.deinit();
        var tmp_scores = scores;
        defer tmp_scores.deinit();
        var tmp3 = try tmp2.astype(allocator, try scores.dtype(allocator));
        defer tmp3.deinit();
        var tmp4 = try moddims(allocator, tmp3, &.{ try pad_mask.dim(allocator, 0), try pad_mask.dim(allocator, 0), n_heads * bsz });
        defer tmp4.deinit();
        scores = try add(allocator, *Variable, scores, *Variable, tmp4);
    }
    var tmp1 = try softmax(allocator, scores, 1);
    defer tmp1.deinit();
    var attn = try dropout(allocator, tmp1, p_dropout);
    defer attn.deinit();
    var tmp2 = try attn.astype(allocator, try v.dtype(allocator));
    defer tmp2.deinit();
    var tmp_result = try matmul(allocator, tmp2, v);
    defer tmp_result.deinit();
    return moddims(allocator, tmp_result, &.{ -1, head_dim * n_heads, bsz });
}

// Unit tests
const AutogradTestF16 = struct {
    pub fn setUp() void {
        zt.common.OptimMode.get().setOptimLevel(.O3);
    }

    pub fn tearDown() void {
        zt.common.OptimMode.get().setOptimLevel(.Default);
    }
};

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

test "AutogradTest -> AutogradVariableIndex" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 1, 3, 3 }, .f64), true);
    defer x.deinit();
    var tmp_x_idx1 = try x.index(allocator, &.{ Index.initDim(0), Index.initDim(0) });
    defer tmp_x_idx1.deinit();
    var tmp_x_idx2 = try x.index(allocator, &.{ Index.initDim(0), Index.initDim(1) });
    defer tmp_x_idx2.deinit();
    var y = try add(allocator, *Variable, tmp_x_idx1, *Variable, tmp_x_idx2);
    defer y.deinit();
    const funcIdx = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            var tmp_idx1 = try input.index(alloc, &.{ Index.initDim(0), Index.initDim(0) });
            defer tmp_idx1.deinit();
            var tmp_idx2 = try input.index(alloc, &.{ Index.initDim(0), Index.initDim(1) });
            defer tmp_idx2.deinit();
            return add(alloc, *Variable, tmp_idx1, *Variable, tmp_idx2);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcIdx, x, null, 1e-5, 1e-4));
}

test "AutogradTest -> AutogradOperatorTypeCompatibility" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f16Supported(allocator)) {
        return error.SkipZigTest;
    }

    var f16_ = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 2 }, .f16), true);
    defer f16_.deinit();
    var f32_ = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 2 }, .f32), true);
    defer f32_.deinit();

    // Binary operators
    try std.testing.expectError(error.VariableDTypeMismatch, add(allocator, *Variable, f16_, *Variable, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, sub(allocator, *Variable, f16_, *Variable, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, mul(allocator, *Variable, f16_, *Variable, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, div(allocator, *Variable, f16_, *Variable, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, greaterThan(allocator, *Variable, f16_, *Variable, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, lessThan(allocator, *Variable, f16_, *Variable, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, greaterThanEqual(allocator, *Variable, f16_, *Variable, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, lessThanEqual(allocator, *Variable, f16_, *Variable, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, logicalAnd(allocator, f16_, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, max(allocator, *Variable, f16_, *Variable, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, min(allocator, *Variable, f16_, *Variable, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, matmul(allocator, f16_, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, matmulTN(allocator, f16_, f32_));
    try std.testing.expectError(error.VariableDTypeMismatch, matmulNT(allocator, f16_, f32_));

    // expect no throw
    var res1 = try binaryCrossEntropy(allocator, f16_, f32_);
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

    if (@import("build_options").ZT_USE_ONEDNN and @import("builtin") or @import("build_options").ZT_USE_CUDNN) {
        var pool2d_res = try pool2d(allocator, f16_, 1, 1, 1, 1, 1, 1, .Max);
        pool2d_res.deinit();
    }

    var res3 = try embedding(allocator, f16_, f32_);
    res3.deinit();

    // Ternary operators
    var f32_2 = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 2 }, .f32), true);
    defer f32_2.deinit();
    var f16_2 = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 2 }, .f16), true);
    defer f16_2.deinit();
    try std.testing.expectError(error.VariableDTypeMismatch, linear(allocator, f16_, f32_, f16_2));
    try std.testing.expectError(error.VariableDTypeMismatch, linear(allocator, f16_, f32_, f32_2));

    if (@import("build_options").ZT_USE_ONEDNN or @import("build_options").ZT_USE_CUDNN) {
        var w = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{1}, .f32), true);
        defer w.deinit();
        var b = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{1}, .f32), true);
        defer b.deinit();

        try std.testing.expectError(error.OneDNNBatchnormNoMomentum, batchnorm(allocator, f16_, f32_, f32_2, w, b, &.{1}, true, 0.01, 0.01));

        try std.testing.expectError(error.OneDNNBatchnormNoMomentum, batchnorm(allocator, f16_, f32_, f16_2, w, b, &.{1}, true, 0.01, 0.01));
    }

    try std.testing.expectError(error.VariableDTypeMismatch, conv2d(allocator, f16_, f32_, f16_2, 1, 1, 0, 0, 1, 1, 1, null));

    // Quaternary operators

    // TODO: test `rnn` throws

    // Variadic operators
    const concat_inputs: []const *Variable = &.{ f16_, f32_, f16_2, f32_2 };
    try std.testing.expectError(error.ConcatInputVariableDTypeMismatch, concatenate(allocator, concat_inputs, 0));
}

test "AutogradTest -> CastingAsDifferentGradTypes" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f16Supported(allocator)) {
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
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f16Supported(allocator)) {
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
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f16Supported(allocator)) {
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
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f16Supported(allocator)) {
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
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

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

test "AutogradTest -> Concatenate" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    const ConcatT1Ctx = struct { x2: *Variable, x3: *Variable, x4: *Variable };
    const ConcatT2Ctx = struct { x1: *Variable, x2: *Variable, x4: *Variable };

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
    const concat_t1_ctx = try allocator.create(ConcatT1Ctx);
    defer allocator.destroy(concat_t1_ctx);
    concat_t1_ctx.* = .{ .x2 = x2, .x3 = x3, .x4 = x4 };
    const funcConcatenateT1 = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, ctx: ?*anyopaque) !*Variable {
            const c: *ConcatT1Ctx = @ptrCast(@alignCast(ctx.?));
            return concatenate(alloc, &.{ input, c.x2, c.x3, c.x4 }, 2);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcConcatenateT1, x1, concat_t1_ctx, 1e-5, 1e-4));

    const concat_t2_ctx = try allocator.create(ConcatT2Ctx);
    defer allocator.destroy(concat_t2_ctx);
    concat_t2_ctx.* = .{ .x1 = x1, .x2 = x2, .x4 = x4 };
    const funcConcatenateT2 = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, ctx: ?*anyopaque) !*Variable {
            const c: *ConcatT2Ctx = @ptrCast(@alignCast(ctx.?));
            return concatenate(alloc, &.{ c.x1, c.x2, input, c.x4 }, 2);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcConcatenateT2, x3, concat_t2_ctx, 1e-5, 1e-4));
}

test "AutogradTest -> Split" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

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
    const funcSplit = (struct {
        pub fn call(alloc: std.mem.Allocator, in: *Variable, _: ?*anyopaque) !*Variable {
            var tmp = try split(alloc, in, i64, 2, 1);
            defer alloc.free(tmp);
            defer for (1..tmp.len) |i| tmp[i].deinit();
            return tmp[0];
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcSplit, input, null, 1e-5, 1e-4));
}

test "AutogradTest -> Tile" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

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
    const funcTile = (struct {
        pub fn call(alloc: std.mem.Allocator, in: *Variable, _: ?*anyopaque) !*Variable {
            return tile(alloc, in, &.{ 1, 2 });
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcTile, input, null, 1e-4, 1e-3));
}

test "AutogradTest -> TileAs" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

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

test "AutogradTestF16 -> TileAsF16" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f16Supported(allocator)) {
        return error.SkipZigTest;
    }
    AutogradTestF16.setUp(); // set optim mode

    defer AutogradTestF16.tearDown(); // reset optim mode

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f16), true);
    defer x.deinit();
    var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 2 }, .f16), true);
    defer y.deinit();
    var tmp = try tileAs(allocator, x, *Variable, y);
    defer tmp.deinit();
    var z = try mul(allocator, *Variable, y, *Variable, tmp);
    defer z.deinit();
    try std.testing.expect(try x.dtype(allocator) == try z.dtype(allocator));
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{ 5, 2 }, f64, 1, .f16), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dy = try y.grad();
    var dx = try x.grad();
    var tmp_exp1 = try zt.tensor.tile(allocator, x.tensor(), &.{ 1, 2 });
    defer tmp_exp1.deinit();
    var exp1 = try tmp_exp1.astype(allocator, try dx.dtype(allocator));
    defer exp1.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), exp1, 1e-2));
    var tmp_exp2 = try zt.tensor.sum(allocator, y.tensor(), &.{1}, false);
    defer tmp_exp2.deinit();
    var exp2 = try tmp_exp2.astype(allocator, try dx.dtype(allocator));
    defer exp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), exp2, 1e-2));
}

test "AutogradTest -> TileAs2" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

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

test "AutogradTest -> Indexing" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 6, 7, 4 }, .f64), true);
    defer x.deinit();
    const funcCol = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return input.index(alloc, &.{ Index.initRange(zt.tensor.span), Index.initDim(4) });
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcCol, x, null, 1e-5, 1e-4));
    const funcRow = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return input.index(alloc, &.{Index.initDim(4)});
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcRow, x, null, 1e-5, 1e-4));
    const funcSlice = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return input.index(alloc, &.{ Index.initRange(zt.tensor.span), Index.initRange(zt.tensor.span), Index.initDim(4) });
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcSlice, x, null, 1e-5, 1e-4));
    const funcCols = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return input.index(alloc, &.{ Index.initRange(zt.tensor.span), Index.initRange(Range.init(2, .{ .dim = 5 })) });
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcCols, x, null, 1e-5, 1e-4));
    const funcRows = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return input.index(alloc, &.{Index.initRange(Range.init(2, .{ .dim = 5 }))});
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcRows, x, null, 1e-5, 1e-4));
    const funcSlices = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return input.index(alloc, &.{ Index.initRange(zt.tensor.span), Index.initRange(zt.tensor.span), Index.initRange(Range.init(2, .{ .dim = 5 })) });
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcSlices, x, null, 1e-5, 1e-4));
    const funcFlat = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return input.flat(alloc, Index.initRange(Range.init(4, .{ .dim = 100 })));
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcFlat, x, null, 1e-5, 1e-4));
}

test "AutogradTest -> Padding" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 3, 3 }, .f32), true);
    defer in.deinit();
    const funcPad = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return padding(alloc, input, &.{ [2]Dim{ 1, 2 }, [2]Dim{ 0, 1 } }, -1);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcPad, in, null, 1e-3, 1e-4));
}

// TODO: test "AutogradTest -> Pooling" {}

// TODO: test "AutogradTestF16 -> PoolingF16" {}

test "AutogradTest -> Reorder" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var in_tensor = try zt.tensor.rand(allocator, &.{ 3, 1, 4, 1 }, .f32);
    try in_tensor.inPlaceMul(allocator, f64, 2);
    var in = try Variable.init(allocator, in_tensor, true);
    defer in.deinit();
    const funcReorder = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return reorder(alloc, input, &.{ 2, 0, 3, 1 });
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcReorder, in, null, 1e-3, 1e-4));
}

test "AutogradTest -> Embedding" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    const n_words: i64 = 10;
    var input_tensor = try zt.tensor.rand(allocator, &.{ 4, 2 }, .f32);
    try input_tensor.inPlaceMul(allocator, i64, n_words);
    var input = try Variable.init(allocator, input_tensor, true);
    defer input.deinit();
    var weights = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 4, n_words }, .f64), true);
    defer weights.deinit();
    const funcEmbed = (struct {
        pub fn call(alloc: std.mem.Allocator, w: *Variable, ctx: ?*anyopaque) !*Variable {
            const in: *Variable = @ptrCast(@alignCast(ctx.?));
            return embedding(alloc, in, w);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcEmbed, weights, input, 1e-5, 1e-4));
}

// TODO (once CUDA is supported): test "AutogradTest -> GetAdvancedIndex" {}

const FuncVar = *const fn (allocator: std.mem.Allocator, lhs: *Variable, rhs: *Variable) anyerror!*Variable;
const FuncScalarL = *const fn (allocator: std.mem.Allocator, lhs: f64, rhs: *Variable) anyerror!*Variable;
const FuncScalarR = *const fn (allocator: std.mem.Allocator, lhs: *Variable, rhs: f64) anyerror!*Variable;

fn TestImpl(comptime fn1: FuncVar, comptime fn2: FuncScalarL, comptime fn3: FuncScalarR) type {
    return struct {
        pub fn fnArrL(allocator: std.mem.Allocator, in: *Variable, ctx: ?*anyopaque) !*Variable {
            const temp: *Variable = @ptrCast(@alignCast(ctx.?));
            return fn1(allocator, in, temp);
        }

        pub fn fnArrR(allocator: std.mem.Allocator, in: *Variable, ctx: ?*anyopaque) !*Variable {
            const temp: *Variable = @ptrCast(@alignCast(ctx.?));
            return fn1(allocator, temp, in);
        }

        pub fn fnScalarL(allocator: std.mem.Allocator, in: *Variable, _: ?*anyopaque) !*Variable {
            return fn2(allocator, 1.414, in);
        }

        pub fn fnScalarR(allocator: std.mem.Allocator, in: *Variable, _: ?*anyopaque) !*Variable {
            return fn3(allocator, in, 1.732);
        }

        pub fn execTests(allocator: std.mem.Allocator) !void {
            var in_tensor = try zt.tensor.rand(allocator, &.{ 3, 4, 5, 6 }, .f64);
            try in_tensor.inPlaceAdd(allocator, f64, 1);
            var input = try Variable.init(allocator, in_tensor, true);
            defer input.deinit();
            var temp_tensor = try zt.tensor.rand(allocator, &.{ 3, 4, 5, 6 }, .f64);
            try temp_tensor.inPlaceSub(allocator, f64, 2);
            var temp = try Variable.init(allocator, temp_tensor, true);
            defer temp.deinit();

            try std.testing.expect(try jacobianTestImpl(allocator, fnArrL, input, temp, 1e-5, 1e-4));
            try std.testing.expect(try jacobianTestImpl(allocator, fnArrR, input, temp, 1e-5, 1e-4));
            try std.testing.expect(try jacobianTestImpl(allocator, fnScalarL, input, null, 1e-5, 1e-7));
            try std.testing.expect(try jacobianTestImpl(allocator, fnScalarR, input, null, 1e-5, 1e-7));
        }
    };
}

test "AutogradBinaryOpsTest -> BasicOps" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    // test add
    const funcAdd1 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: *Variable) !*Variable {
            return add(alloc, *Variable, a, *Variable, b);
        }
    }).call;
    const funcAdd2 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: f64, b: *Variable) !*Variable {
            return add(alloc, f64, a, *Variable, b);
        }
    }).call;
    const funcAdd3 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: f64) !*Variable {
            return add(alloc, *Variable, a, f64, b);
        }
    }).call;
    try TestImpl(funcAdd1, funcAdd2, funcAdd3).execTests(allocator);

    // test sub
    const funcSub1 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: *Variable) !*Variable {
            return sub(alloc, *Variable, a, *Variable, b);
        }
    }).call;
    const funcSub2 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: f64, b: *Variable) !*Variable {
            return sub(alloc, f64, a, *Variable, b);
        }
    }).call;
    const funcSub3 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: f64) !*Variable {
            return sub(alloc, *Variable, a, f64, b);
        }
    }).call;
    try TestImpl(funcSub1, funcSub2, funcSub3).execTests(allocator);

    // test mul
    const funcMul1 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: *Variable) !*Variable {
            return mul(alloc, *Variable, a, *Variable, b);
        }
    }).call;
    const funcMul2 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: f64, b: *Variable) !*Variable {
            return mul(alloc, f64, a, *Variable, b);
        }
    }).call;
    const funcMul3 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: f64) !*Variable {
            return mul(alloc, *Variable, a, f64, b);
        }
    }).call;
    try TestImpl(funcMul1, funcMul2, funcMul3).execTests(allocator);

    // test div
    const funcDiv1 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: *Variable) !*Variable {
            return div(alloc, *Variable, a, *Variable, b);
        }
    }).call;
    const funcDiv2 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: f64, b: *Variable) !*Variable {
            return div(alloc, f64, a, *Variable, b);
        }
    }).call;
    const funcDiv3 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: f64) !*Variable {
            return div(alloc, *Variable, a, f64, b);
        }
    }).call;
    try TestImpl(funcDiv1, funcDiv2, funcDiv3).execTests(allocator);

    // test min
    const funcMin1 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: *Variable) !*Variable {
            return min(alloc, *Variable, a, *Variable, b);
        }
    }).call;
    const funcMin2 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: f64, b: *Variable) !*Variable {
            return min(alloc, f64, a, *Variable, b);
        }
    }).call;
    const funcMin3 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: f64) !*Variable {
            return min(alloc, *Variable, a, f64, b);
        }
    }).call;
    try TestImpl(funcMin1, funcMin2, funcMin3).execTests(allocator);

    // test max
    const funcMax1 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: *Variable) !*Variable {
            return max(alloc, *Variable, a, *Variable, b);
        }
    }).call;
    const funcMax2 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: f64, b: *Variable) !*Variable {
            return max(alloc, f64, a, *Variable, b);
        }
    }).call;
    const funcMax3 = (struct {
        pub fn call(alloc: std.mem.Allocator, a: *Variable, b: f64) !*Variable {
            return max(alloc, *Variable, a, f64, b);
        }
    }).call;
    try TestImpl(funcMax1, funcMax2, funcMax3).execTests(allocator);
}

test "AutogradBinaryOpsTest -> BinaryCrossEntropy" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{10}, .f32), true);
    defer x.deinit();
    var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{10}, .f32), true);
    defer y.deinit();
    var loss = try binaryCrossEntropy(allocator, x, y);
    defer loss.deinit();

    // loss should be positive
    var comp = try zt.tensor.greaterThan(allocator, Tensor, loss.tensor(), f64, 0);
    defer comp.deinit();
    var res = try zt.tensor.all(allocator, comp, &.{}, false);
    defer res.deinit();
    try std.testing.expect(try res.scalar(allocator, i8) != 0);
}

test "AutogradBinaryOpsTest -> CrossEntropy" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    const CrossEntropyCtx = struct { y: *Variable, mode: zt.common.ReduceMode = undefined, ignore_idx: i64 };

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 7, 10, 4 }, .f64), true);
    defer x.deinit();
    var tmp_y1 = try zt.tensor.rand(allocator, &.{ 10, 4 }, .u32);
    defer tmp_y1.deinit();
    var tmp_y2 = try zt.tensor.mod(allocator, Tensor, tmp_y1, f64, 7);
    defer tmp_y2.deinit();
    var y = try Variable.init(allocator, try tmp_y2.astype(allocator, .s32), false);
    defer y.deinit();
    var ignore_idx_var = try y.index(allocator, &.{ Index.initDim(0), Index.initDim(0) });
    defer ignore_idx_var.deinit();
    const ignore_idx: i64 = @intCast(try ignore_idx_var.scalar(allocator, i32));

    const modes: []const zt.common.ReduceMode = &.{ .None, .Sum, .Mean };
    var ctx = try allocator.create(CrossEntropyCtx);
    defer allocator.destroy(ctx);
    ctx.* = .{ .y = y, .ignore_idx = ignore_idx };
    for (modes) |mode| {
        ctx.mode = mode;
        const func = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const cc: *CrossEntropyCtx = @ptrCast(@alignCast(c.?));
                return categoricalCrossEntropy(alloc, input, cc.y, cc.mode, -1);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, func, x, ctx, 1e-5, 1e-4));
        const funcIgnore = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const cc: *CrossEntropyCtx = @ptrCast(@alignCast(c.?));
                return categoricalCrossEntropy(alloc, input, cc.y, cc.mode, cc.ignore_idx);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, funcIgnore, x, ctx, 1e-5, 1e-4));
    }

    var loss_sum = try categoricalCrossEntropy(allocator, x, y, .Sum, -1);
    defer loss_sum.deinit();
    var loss_mean = try categoricalCrossEntropy(allocator, x, y, .Mean, -1);
    defer loss_mean.deinit();
    var actual_var = try div(allocator, *Variable, loss_sum, *Variable, loss_mean);
    defer actual_var.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 40), try actual_var.scalar(allocator, f64), 1e-5);

    var loss_sum_ignore = try categoricalCrossEntropy(allocator, x, y, .Sum, ignore_idx);
    defer loss_sum_ignore.deinit();
    var loss_mean_ignore = try categoricalCrossEntropy(allocator, x, y, .Mean, ignore_idx);
    defer loss_mean_ignore.deinit();
    var tmp_eq = try zt.tensor.eq(allocator, Tensor, y.tensor(), i64, ignore_idx);
    defer tmp_eq.deinit();
    var ignore_count = try zt.tensor.sum(allocator, tmp_eq, &.{}, false);
    defer ignore_count.deinit();
    var actual_var2 = try div(allocator, *Variable, loss_sum_ignore, *Variable, loss_mean_ignore);
    defer actual_var2.deinit();
    try std.testing.expectApproxEqAbs(
        @as(f64, 40) - @as(f64, @floatFromInt(try ignore_count.scalar(allocator, u32))),
        try actual_var2.scalar(allocator, f64),
        1e-5,
    );

    var a = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 4, 5, 6 }, .f32), false);
    defer a.deinit();
    var b = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 8 }, .f32), false);
    defer b.deinit();
    try std.testing.expectError(error.CatCrossEntropyDimsMismatch, categoricalCrossEntropy(allocator, a, b, .Mean, -1));

    var c = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 4, 5, 6 }, .f32), false);
    defer c.deinit();
    var d = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), false);
    defer d.deinit();
    try std.testing.expectError(error.CatCrossEntropyDimsMismatch, categoricalCrossEntropy(allocator, c, d, .Mean, -1));
}

test "AutogradBinaryOpsTest -> Linear" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    const LinkCtx = struct { in: *Variable, wt: *Variable, bs: *Variable };

    const batch_sizes: []const Dim = &.{ 1, 5 };
    for (batch_sizes) |b| {
        var in_tensor = try zt.tensor.rand(allocator, &.{ 3, 4, b }, .f64);
        try in_tensor.inPlaceMul(allocator, f64, 4);
        try in_tensor.inPlaceSub(allocator, f64, 1);
        var in = try Variable.init(allocator, in_tensor, true);
        defer in.deinit();
        var wt_tensor = try zt.tensor.rand(allocator, &.{ 6, 3 }, .f64);
        try wt_tensor.inPlaceMul(allocator, f64, 2);
        try wt_tensor.inPlaceSub(allocator, f64, 1);
        var wt = try Variable.init(allocator, wt_tensor, true);
        defer wt.deinit();
        var bs_tensor = try zt.tensor.rand(allocator, &.{6}, .f64);
        try bs_tensor.inPlaceMul(allocator, f64, 2);
        try bs_tensor.inPlaceSub(allocator, f64, 1);
        var bs = try Variable.init(allocator, bs_tensor, true);
        defer bs.deinit();
        const ctx = try allocator.create(LinkCtx);
        defer allocator.destroy(ctx);
        ctx.* = .{ .in = in, .wt = wt, .bs = bs };
        const funcLinIn = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const lin_c: *LinkCtx = @ptrCast(@alignCast(c.?));
                return linear(alloc, input, lin_c.wt, lin_c.bs);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, funcLinIn, in, ctx, 1e-8, 1e-4));
        const funcLinWt = (struct {
            pub fn call(alloc: std.mem.Allocator, weight: *Variable, c: ?*anyopaque) !*Variable {
                const lin_c: *LinkCtx = @ptrCast(@alignCast(c.?));
                return linear(alloc, lin_c.in, weight, lin_c.bs);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, funcLinWt, wt, ctx, 1e-8, 1e-4));
        const funcLinBs = (struct {
            pub fn call(alloc: std.mem.Allocator, bias: *Variable, c: ?*anyopaque) !*Variable {
                const lin_c: *LinkCtx = @ptrCast(@alignCast(c.?));
                return linear(alloc, lin_c.in, lin_c.wt, bias);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, funcLinBs, bs, ctx, 1e-8, 1e-4));
    }
}

test "AutogradTestF16 -> LinearF16" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f16Supported(allocator)) {
        return error.SkipZigTest;
    }
    const LinkCtx = struct { in: *Variable, wt: *Variable, bs: *Variable };
    AutogradTestF16.setUp(); // set optim mode

    defer AutogradTestF16.tearDown(); // reset optim mode

    const batch_sizes: []const Dim = &.{ 1, 5 };
    const scale: f32 = 4;
    for (batch_sizes) |b| {
        var in_tensor = try zt.tensor.rand(allocator, &.{ 2, 2, b }, .f16);
        try in_tensor.inPlaceMul(allocator, f32, scale);
        var in = try Variable.init(allocator, in_tensor, true);
        defer in.deinit();
        var wt_tensor = try zt.tensor.rand(allocator, &.{ 2, 2 }, .f16);
        try wt_tensor.inPlaceMul(allocator, f32, scale);
        var wt = try Variable.init(allocator, wt_tensor, true);
        defer wt.deinit();
        var bs_tensor = try zt.tensor.rand(allocator, &.{2}, .f16);
        try bs_tensor.inPlaceMul(allocator, f32, scale);
        var bs = try Variable.init(allocator, bs_tensor, true);
        defer bs.deinit();
        const ctx = try allocator.create(LinkCtx);
        defer allocator.destroy(ctx);
        ctx.* = .{ .in = in, .wt = wt, .bs = bs };
        const funcLinIn = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const lin_c: *LinkCtx = @ptrCast(@alignCast(c.?));
                return linear(alloc, input, lin_c.wt, lin_c.bs);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, funcLinIn, in, ctx, 5e-2, 5e-1));
        const funcLinWt = (struct {
            pub fn call(alloc: std.mem.Allocator, weight: *Variable, c: ?*anyopaque) !*Variable {
                const lin_c: *LinkCtx = @ptrCast(@alignCast(c.?));
                return linear(alloc, lin_c.in, weight, lin_c.bs);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, funcLinWt, wt, ctx, 5e-2, 5e-1));
        const funcLinBs = (struct {
            pub fn call(alloc: std.mem.Allocator, bias: *Variable, c: ?*anyopaque) !*Variable {
                const lin_c: *LinkCtx = @ptrCast(@alignCast(c.?));
                return linear(alloc, lin_c.in, lin_c.wt, bias);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, funcLinBs, bs, ctx, 5e-2, 5e-1));
    }
}

test "AutogradBinaryOpsTest -> Multiply" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer x.deinit();
    var y = try mul(allocator, *Variable, x, *Variable, x);
    defer y.deinit();
    var dy = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dy.deinit();
    try y.backwardWithGrad(allocator, dy, false);
    var dx = try x.grad();
    var comp = try zt.tensor.mul(allocator, Tensor, x.tensor(), f64, 2);
    defer comp.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), comp, 1e-5));
}

test "AutogradBinaryOpsTest -> MultiplyAdd" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
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
    var dx = try x.grad();
    var dy = try y.grad();
    var comp1 = try zt.tensor.mul(allocator, f64, 2, Tensor, x.tensor());
    defer comp1.deinit();
    try comp1.inPlaceAdd(allocator, Tensor, y.tensor());
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), comp1, 1e-5));
    var comp2 = try zt.tensor.mul(allocator, f64, 2, Tensor, y.tensor());
    defer comp2.deinit();
    try comp2.inPlaceAdd(allocator, Tensor, x.tensor());
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), comp2, 1e-5));
}

test "AutogradBinaryOpsTest -> MultiplyAddScalar" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer x.deinit();
    var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer y.deinit();
    var tmp1 = try mul(allocator, f64, 2, *Variable, x);
    defer tmp1.deinit();
    var tmp2 = try mul(allocator, *Variable, x, *Variable, y);
    defer tmp2.deinit();
    var tmp3 = try add(allocator, *Variable, tmp1, *Variable, tmp2);
    defer tmp3.deinit();
    var z = try add(allocator, *Variable, tmp3, *Variable, y);
    defer z.deinit();
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dx = try x.grad();
    var dy = try y.grad();
    var comp1 = try zt.tensor.add(allocator, f64, 2, Tensor, y.tensor());
    defer comp1.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), comp1, 1e-5));
    var comp2 = try zt.tensor.add(allocator, f64, 1, Tensor, x.tensor());
    defer comp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), comp2, 1e-5));
}

test "AutogradBinaryOpsTest -> MultiplySub" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer x.deinit();
    var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer y.deinit();
    var lhs = try mul(allocator, *Variable, x, *Variable, x);
    defer lhs.deinit();
    var rhs = try mul(allocator, *Variable, x, *Variable, y);
    defer rhs.deinit();
    var z = try sub(allocator, *Variable, lhs, *Variable, rhs);
    defer z.deinit();
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dx = try x.grad();
    var dy = try y.grad();
    var comp1 = try zt.tensor.mul(allocator, f64, 2, Tensor, x.tensor());
    defer comp1.deinit();
    try comp1.inPlaceSub(allocator, Tensor, y.tensor());
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), comp1, 1e-5));
    var comp2 = try zt.tensor.negative(allocator, x.tensor());
    defer comp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), comp2, 1e-5));
}

test "AutogradBinaryOpsTest -> DivideAdd" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f64), true);
    defer x.deinit();
    var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f64), true);
    defer y.deinit();
    var tmp1 = try div(allocator, *Variable, x, *Variable, y);
    defer tmp1.deinit();
    var tmp2 = try add(allocator, *Variable, x, *Variable, tmp1);
    defer tmp2.deinit();
    var z = try add(allocator, *Variable, tmp2, *Variable, y);
    defer z.deinit();
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f64), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dx = try x.grad();
    var dy = try y.grad();
    try std.testing.expect(try z.dtype(allocator) == .f64);
    var comp1 = try zt.tensor.div(allocator, f64, 1, Tensor, y.tensor());
    defer comp1.deinit();
    try comp1.inPlaceAdd(allocator, f64, 1);
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), comp1, 1e-5));
    var tmp3 = try zt.tensor.mul(allocator, Tensor, y.tensor(), Tensor, y.tensor());
    defer tmp3.deinit();
    var tmp4 = try zt.tensor.div(allocator, Tensor, x.tensor(), Tensor, tmp3);
    defer tmp4.deinit();
    var comp2 = try zt.tensor.sub(allocator, f64, 1, Tensor, tmp4);
    defer comp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), comp2, 1e-5));
}

test "AutogradBinaryOpsTest -> matmul" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    const MatMulCtx = struct { other: *Variable };

    const M: Dim = 10;
    const K: Dim = 12;
    const N: Dim = 14;
    const b2: Dim = 2;
    const b3: Dim = 4;
    const mk: Shape = &.{ M, K };
    const mkb2: Shape = &.{ M, K, b2 }; // 1 batch dim
    const mkb2b3: Shape = &.{ M, K, b2, b3 }; // 2 batch dims
    const kn: Shape = &.{ K, N };
    const knb2: Shape = &.{ K, N, b2 }; // 1 batch dim
    const knb2b3: Shape = &.{ K, N, b2, b3 }; // 2 batch dims

    // lhs, rhs
    const inputs: []const std.meta.Tuple(&.{ Shape, Shape }) = &.{
        .{ mk, kn },
        .{ mk, knb2 },
        .{ mk, knb2b3 },
        .{ mkb2, kn },
        .{ mkb2, knb2 },
        .{ mkb2b3, kn },
        .{ mkb2b3, knb2b3 },
    };

    const trFirstTwoDims = (struct {
        pub fn call(alloc: std.mem.Allocator, in: Shape) !Shape {
            var out = try alloc.alloc(Dim, in.len);
            @memcpy(out, in);
            const out1 = out[1];
            out[1] = out[0];
            out[0] = out1;
            return out;
        }
    }).call;

    const ctx = try allocator.create(MatMulCtx);
    defer allocator.destroy(ctx);
    for (inputs) |pair| {
        const a_shape = pair[0];
        const b_shape = pair[1];

        var a_tensor = try zt.tensor.rand(allocator, a_shape, .f64);
        try a_tensor.inPlaceMul(allocator, f64, 2);
        try a_tensor.inPlaceSub(allocator, f64, 1);
        var a = try Variable.init(allocator, a_tensor, true);
        defer a.deinit();
        var b_tensor = try zt.tensor.rand(allocator, b_shape, .f64);
        try b_tensor.inPlaceMul(allocator, f64, 2);
        try b_tensor.inPlaceSub(allocator, f64, 1);
        var b = try Variable.init(allocator, b_tensor, true);
        defer b.deinit();

        const aT_shape = try trFirstTwoDims(allocator, a_shape);
        defer allocator.free(aT_shape);
        var aT = try Variable.init(allocator, try zt.tensor.rand(allocator, aT_shape, .f64), true);
        defer aT.deinit();
        const bT_shape = try trFirstTwoDims(allocator, b_shape);
        defer allocator.free(bT_shape);
        var bT = try Variable.init(allocator, try zt.tensor.rand(allocator, bT_shape, .f64), true);
        defer bT.deinit();

        // matmul
        const funcMatmulLhs = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const m_ctx: *MatMulCtx = @ptrCast(@alignCast(c.?));
                return matmul(alloc, input, m_ctx.other);
            }
        }).call;
        ctx.* = .{ .other = b };
        try std.testing.expect(try jacobianTestImpl(allocator, funcMatmulLhs, a, ctx, 1e-6, 1e-4));
        const funcMatmulRhs = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const m_ctx: *MatMulCtx = @ptrCast(@alignCast(c.?));
                return matmul(alloc, m_ctx.other, input);
            }
        }).call;
        ctx.* = .{ .other = a };
        try std.testing.expect(try jacobianTestImpl(allocator, funcMatmulRhs, b, ctx, 1e-6, 1e-4));

        // matmulTN
        const funcMatmulTNLhs = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const m_ctx: *MatMulCtx = @ptrCast(@alignCast(c.?));
                return matmulTN(alloc, input, m_ctx.other);
            }
        }).call;
        ctx.* = .{ .other = b };
        try std.testing.expect(try jacobianTestImpl(allocator, funcMatmulTNLhs, aT, ctx, 1e-6, 1e-4));
        const funcMatmulTNRhs = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const m_ctx: *MatMulCtx = @ptrCast(@alignCast(c.?));
                return matmulTN(alloc, m_ctx.other, input);
            }
        }).call;
        ctx.* = .{ .other = aT };
        try std.testing.expect(try jacobianTestImpl(allocator, funcMatmulTNRhs, b, ctx, 1e-6, 1e-4));

        // matmulNT
        const funcMatmulNTLhs = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const m_ctx: *MatMulCtx = @ptrCast(@alignCast(c.?));
                return matmulNT(alloc, input, m_ctx.other);
            }
        }).call;
        ctx.* = .{ .other = bT };
        try std.testing.expect(try jacobianTestImpl(allocator, funcMatmulNTLhs, a, ctx, 1e-6, 1e-4));
        const funcMatmulNTRhs = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const m_ctx: *MatMulCtx = @ptrCast(@alignCast(c.?));
                return matmulNT(alloc, m_ctx.other, input);
            }
        }).call;
        ctx.* = .{ .other = a };
        try std.testing.expect(try jacobianTestImpl(allocator, funcMatmulNTRhs, bT, ctx, 1e-6, 1e-4));
    }
}

test "AutogradBinaryOpsTest -> WeightNormLinear" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var v = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 3, 2 }, .f32), true);
    defer v.deinit();
    const norm_dim: Shape = &.{1};
    var tmp_g = try norm(allocator, v, norm_dim, 2, false);
    defer tmp_g.deinit();
    var g = try Variable.initSharedData(allocator, tmp_g.shared_data.retain(), true);
    defer g.deinit();
    var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 2, 3 }, .f32), true);
    defer in.deinit();

    const WCtx = struct {
        v: *Variable,
        g: *Variable,
        norm_dim: Shape,
        in: *Variable,
    };
    const ctx = try allocator.create(WCtx);
    defer allocator.destroy(ctx);
    ctx.* = .{ .v = v, .g = g, .norm_dim = norm_dim, .in = in };
    const funcWeightNormIn = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
            const w_ctx: *WCtx = @ptrCast(@alignCast(c.?));
            var tmp1 = try norm(alloc, w_ctx.v, w_ctx.norm_dim, 2, false);
            defer tmp1.deinit();
            var tmp2 = try div(alloc, *Variable, w_ctx.g, *Variable, tmp1);
            defer tmp2.deinit();
            var tmp3 = try tileAs(alloc, tmp2, *Variable, w_ctx.v);
            defer tmp3.deinit();
            var w = try mul(alloc, *Variable, w_ctx.v, *Variable, tmp3);
            defer w.deinit();
            return matmul(allocator, w, input);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcWeightNormIn, in, ctx, 1e-3, 1e-4));

    const funcWeightNormV = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
            const w_ctx: *WCtx = @ptrCast(@alignCast(c.?));
            var tmp1 = try norm(alloc, input, w_ctx.norm_dim, 2, false);
            defer tmp1.deinit();
            var tmp2 = try div(alloc, *Variable, w_ctx.g, *Variable, tmp1);
            defer tmp2.deinit();
            var tmp3 = try tileAs(alloc, tmp2, *Variable, input);
            defer tmp3.deinit();
            var w = try mul(alloc, *Variable, input, *Variable, tmp3);
            defer w.deinit();
            return matmul(allocator, w, w_ctx.in);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcWeightNormV, v, ctx, 1e-2, 1e-4));

    const funcWeightNormG = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
            const w_ctx: *WCtx = @ptrCast(@alignCast(c.?));
            var tmp1 = try norm(alloc, w_ctx.v, w_ctx.norm_dim, 2, false);
            defer tmp1.deinit();
            var tmp2 = try div(alloc, *Variable, input, *Variable, tmp1);
            defer tmp2.deinit();
            var tmp3 = try tileAs(alloc, tmp2, *Variable, w_ctx.v);
            defer tmp3.deinit();
            var w = try mul(alloc, *Variable, w_ctx.v, *Variable, tmp3);
            defer w.deinit();
            return matmul(allocator, w, w_ctx.in);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcWeightNormG, g, ctx, 5e-3, 1e-4));
}

test "AutogradConv2DTest -> Convolve" {
    if (!@import("build_options").ZT_USE_ONEDNN and !@import("build_options").ZT_USE_CUDNN) {
        return error.SkipZigTest;
    }
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 10, 9, 8, 7 }, .f32), true);
    defer in.deinit();
    var wt = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 4, 3, 8, 6 }, .f32), true);
    defer wt.deinit();
    var bs = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 1, 1, 6, 1 }, .f32), true);
    defer bs.deinit();
    const px: i64 = 2;
    const py: i64 = 1;
    const sx: i64 = 1;
    const sy: i64 = 1;
    const dx: i64 = 1;
    const dy: i64 = 1;
    const benchmarks: ?zigrc.Arc(ConvBenchmarks) = null;
    const Ctx = struct {
        wt: *Variable,
        bs: *Variable = undefined,
        sx: i64,
        sy: i64,
        dx: i64,
        dy: i64,
        px: i64,
        py: i64,
        benchmarks: ?zigrc.Arc(ConvBenchmarks) = null,
    };
    const ctx = try allocator.create(Ctx);
    ctx.* = .{ .wt = wt, .bs = bs, .sx = sx, .sy = sy, .dx = dx, .dy = dy, .px = px, .py = py, .benchmarks = benchmarks };
    defer allocator.destroy(ctx);
    const funcConvIn = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, context: ?*anyopaque) !*Variable {
            const c: *Ctx = @ptrCast(@alignCast(context.?));
            return conv2dNoBias(alloc, input, c.wt, c.sx, c.sy, c.px, c.py, c.dx, c.dy, 1, c.benchmarks);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcConvIn, in, ctx, 0.06, 1e-4));
}

// TODO: test "AutogradTestF16 -> ConvolveF16" {}

// TODO: test "AutogradConv2DTest -> ConvolveFilterGroups" {}

// TODO: test "AutogradConv2DTest -> ConvolveDilation" {}

// TODO: test "AutogradConv2DTest -> WeightNormConv" {}

test "AutogradNormalizationTest -> Normalize" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 3 }, .f64), true);
    defer x.deinit();
    const funcNormalize2 = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return normalize(alloc, input, &.{1}, 2, 1e-12);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcNormalize2, x, null, 1e-5, 1e-4));
    var ys = try funcNormalize2(allocator, x, null);
    defer ys.deinit();
    var tmp1 = try zt.tensor.mul(allocator, Tensor, ys.tensor(), Tensor, ys.tensor());
    defer tmp1.deinit();
    var a = try zt.tensor.sum(allocator, tmp1, &.{1}, false);
    defer a.deinit();
    var b = try zt.tensor.full(allocator, &.{5}, f64, 1, .f64);
    defer b.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, a, b, 1e-5));
    var yb = try normalize(allocator, x, &.{1}, 2, 1);
    defer yb.deinit();
    var tmp2 = try zt.tensor.mul(allocator, Tensor, yb.tensor(), Tensor, yb.tensor());
    defer tmp2.deinit();
    var tmp3 = try zt.tensor.sum(allocator, tmp2, &.{1}, false);
    defer tmp3.deinit();
    var tmp4 = try zt.tensor.sqrt(allocator, tmp3);
    defer tmp4.deinit();
    var tmp5 = try zt.tensor.lessThanEqual(allocator, Tensor, tmp4, f64, 1);
    defer tmp5.deinit();
    var res = try zt.tensor.all(allocator, tmp5, &.{}, false);
    defer res.deinit();
    try std.testing.expect(try res.scalar(allocator, i8) != 0);
}

// TODO: test "AutogradNormalizationTest -> BatchNormEvalModeOutputSingleAxis" {}

// TODO: test "AutogradNormalizationTest -> BatchNormEvalModeOutputMultipleAxis" {}

// TODO: test "AutogradNormalizationTest -> BatchNormTrainModeOutputSingleAxis" {}

// TODO: test "AutogradNormalizationTest -> BatchNormTrainModeOutputMultipleAxis" {}

// TODO: test "AutogradNormalizationTest -> BatchNormJacobian" {}

// TODO: test "AutogradTestF16 -> BatchNormJacobianF16" {}

// TODO: test "AutogradNormalizationTest -> BatchNormJacobianMultipleAxes" {}

// TODO: test "AutogradTestF16 -> BatchNormJacobianMultipleAxesF16" {}

// TODO: debug - test hangs/never finished (on OSx aarch64)
test "AutogradNormalizationTest -> LayerNormJacobian" {
    if (!@import("build_options").ZT_USE_ONEDNN and !@import("build_options").ZT_USE_CUDNN) {
        return error.SkipZigTest;
    }
    const allocator = std.heap.c_allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    const feat_axes: []const i64 = &.{ 0, 1, 2, 3 };
    var input = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 7, 7, 3, 10 }, .f32), true);
    defer input.deinit();
    var n_features: i64 = 1;
    for (feat_axes) |ax| {
        n_features *= try input.dim(allocator, @intCast(ax));
    }
    var running_mean = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{n_features}, .f32), false);
    defer running_mean.deinit();
    var running_var = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{n_features}, .f32), false);
    defer running_var.deinit();
    var weight = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{n_features}, .f32), true);
    defer weight.deinit();
    var bias = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{n_features}, .f32), true);
    defer bias.deinit();
    const Ctx = struct {
        running_mean: *Variable,
        running_var: *Variable,
        weight: *Variable,
        bias: *Variable,
        feat_axes: []i64,
    };
    const funcLnIn = (struct {
        pub fn call(alloc: std.mem.Allocator, in: *Variable, ctx: ?*anyopaque) !*Variable {
            const c: *Ctx = @ptrCast(@alignCast(ctx.?));
            return batchnorm(alloc, in, c.weight, c.bias, c.running_mean, c.running_var, c.feat_axes, true, 0, 1e-5);
        }
    }).call;
    const ctx = try allocator.create(Ctx);
    defer {
        allocator.free(ctx.feat_axes);
        allocator.destroy(ctx);
    }
    ctx.* = .{ .running_mean = running_mean, .running_var = running_var, .weight = weight, .bias = bias, .feat_axes = try allocator.alloc(i64, feat_axes.len) };
    @memcpy(ctx.feat_axes, feat_axes);
    try std.testing.expect(try jacobianTestImpl(allocator, funcLnIn, input, ctx, 1e-2, 1e-4));
}

// TODO: test "AutogradTestF16 -> LayerNormJacobianF16" {}

test "AutogradReductionTest -> Sum" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    const handle_dims: []const bool = &.{ false, true };
    for (handle_dims) |keep_dims| {
        const s: Shape = if (keep_dims) &.{ 6, 1 } else &.{6};

        var x = try Variable.init(allocator, try zt.tensor.rand(allocator, s, .f32), true);
        defer x.deinit();
        var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 6, 3 }, .f32), true);
        defer y.deinit();

        var tmp1 = try sum(allocator, y, &.{1}, keep_dims);
        defer tmp1.deinit();
        var z = try mul(allocator, *Variable, x, *Variable, tmp1);
        defer z.deinit();
        var dz = try Variable.init(allocator, try zt.tensor.full(allocator, s, f64, 1, .f32), false);
        defer dz.deinit();
        try z.backwardWithGrad(allocator, dz, false);

        var dy = try y.grad();
        var dx = try x.grad();
        var tmp2 = try zt.tensor.tile(allocator, x.tensor(), &.{ 1, 3 });
        defer tmp2.deinit();
        try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), tmp2, 1e-5));
        var tmp3 = try zt.tensor.sum(allocator, y.tensor(), &.{1}, keep_dims);
        defer tmp3.deinit();
        try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), tmp3, 1e-5));
        const TestCtx = struct { keep_dims: bool };
        const ctx = try allocator.create(TestCtx);
        defer allocator.destroy(ctx);
        ctx.* = .{ .keep_dims = keep_dims };

        const funcSum = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const c_ctx: *TestCtx = @ptrCast(@alignCast(c.?));
                return sum(alloc, input, &.{0}, c_ctx.keep_dims);
            }
        }).call;
        // Reduce over 1-dim input
        var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{6}, .f32), true);
        defer in.deinit();
        try std.testing.expect(try jacobianTestImpl(allocator, funcSum, in, ctx, 5e-3, 1e-4));
        // Reduce over scalar input
        var in_scalar = try Variable.init(allocator, try zt.tensor.fromScalar(allocator, f64, 3.14, .f32), true);
        defer in_scalar.deinit();
        try std.testing.expect(try jacobianTestImpl(allocator, funcSum, in_scalar, ctx, 5e-3, 1e-4));
    }

    var r = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 6, 7, 8 }, .f32), true);
    defer r.deinit();
    var r_out = try sum(allocator, r, &.{ 1, 2 }, false);
    defer r_out.deinit();
    var r_out_tensor = try zt.tensor.sum(allocator, r.tensor(), &.{ 1, 2 }, false);
    defer r_out_tensor.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, r_out.tensor(), r_out_tensor, 1e-5));
}

test "AutogradReductionTest -> SumAs" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer x.deinit();
    var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 2 }, .f32), true);
    defer y.deinit();
    var tmp1 = try sumAs(allocator, y, *Variable, x);
    defer tmp1.deinit();
    var z = try mul(allocator, *Variable, x, *Variable, tmp1);
    defer z.deinit();
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dy = try y.grad();
    var dx = try x.grad();
    var tmp2 = try zt.tensor.tile(allocator, x.tensor(), &.{ 1, 2 });
    defer tmp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), tmp2, 1e-5));
    var tmp3 = try zt.tensor.sum(allocator, y.tensor(), &.{1}, false);
    defer tmp3.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), tmp3, 1e-5));
}

test "AutogradReductionTest -> SumAs2" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 2 }, .f32), true);
    defer y.deinit();
    var z = try sumAs(allocator, y, Shape, &.{5});
    defer z.deinit();
    var dz = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dz.deinit();
    try z.backwardWithGrad(allocator, dz, false);
    var dy = try y.grad();
    var tmp = try zt.tensor.full(allocator, &.{ 5, 2 }, f64, 1, .f32);
    defer tmp.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), tmp, 1e-5));
}

test "AutogradReductionTest -> Mean" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    const TestCtx = struct { keep_dims: bool };

    const handle_dims: []const bool = &.{ false, true };
    for (handle_dims) |keep_dims| {
        const x_shape: Shape = if (keep_dims) &.{ 5, 1, 1 } else &.{5};
        var x = try Variable.init(allocator, try zt.tensor.rand(allocator, x_shape, .f32), true);
        defer x.deinit();
        var y = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 3, 2 }, .f32), true);
        defer y.deinit();
        var var_out = try mean(allocator, y, &.{ 1, 2 }, keep_dims);
        defer var_out.deinit();
        var tmp = try mean(allocator, y, &.{ 1, 2 }, keep_dims);
        defer tmp.deinit();
        var z = try mul(allocator, *Variable, x, *Variable, tmp);
        defer z.deinit();
        var dz = try Variable.init(allocator, try zt.tensor.full(allocator, try x.shape(allocator), f64, 1, .f32), false);
        defer dz.deinit();
        try z.backwardWithGrad(allocator, dz, false);
        var dy = try y.grad();
        var dx = try x.grad();
        var tmp1 = try zt.tensor.tile(allocator, x.tensor(), &.{ 1, 3, 2 });
        defer tmp1.deinit();
        try tmp1.inPlaceDiv(allocator, f64, 6);
        try std.testing.expect(try zt.tensor.allClose(allocator, dy.tensor(), tmp1, 1e-5));
        var tmp2 = try zt.tensor.mean(allocator, y.tensor(), &.{ 1, 2 }, keep_dims);
        defer tmp2.deinit();
        try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), tmp2, 1e-5));

        var a = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 3, 2 }, .f64), true);
        defer a.deinit();
        const funcMean = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const m_ctx: *TestCtx = @ptrCast(@alignCast(c.?));
                return mean(alloc, input, &.{ 1, 2 }, m_ctx.keep_dims);
            }
        }).call;
        const ctx = try allocator.create(TestCtx);
        defer allocator.destroy(ctx);
        ctx.* = .{ .keep_dims = keep_dims };
        try std.testing.expect(try jacobianTestImpl(allocator, funcMean, a, ctx, 1e-4, 1e-4));

        var q = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 6, 7, 8 }, .f32), false);
        defer q.deinit();
        var q_out = try mean(allocator, q, &.{ 1, 2 }, keep_dims);
        defer q_out.deinit();
        var q_out_tensor = try zt.tensor.mean(allocator, q.tensor(), &.{ 1, 2 }, keep_dims);
        defer q_out_tensor.deinit();
        try std.testing.expect(try zt.tensor.allClose(allocator, q_out.tensor(), q_out_tensor, 1e-5));

        const funcMean0 = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const m_ctx: *TestCtx = @ptrCast(@alignCast(c.?));
                return mean(alloc, input, &.{0}, m_ctx.keep_dims);
            }
        }).call;
        // Reduce over 1-dim input
        var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{6}, .f32), true);
        defer in.deinit();
        try std.testing.expect(try jacobianTestImpl(allocator, funcMean0, in, ctx, 5e-3, 1e-4));
        // Reduce over scalar input
        var in_scalar = try Variable.init(allocator, try zt.tensor.fromScalar(allocator, f64, 3.14, .f32), true);
        defer in_scalar.deinit();
        try std.testing.expect(try jacobianTestImpl(allocator, funcMean0, in_scalar, ctx, 5e-3, 1e-4));
    }
}

test "AutogradReductionTest -> Variance" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    const TestCtx = struct { b: bool, keep_dims: bool };

    const biased: []const bool = &.{ true, false };
    const handle_dims: []const bool = &.{ false, true };
    for (biased) |b| {
        for (handle_dims) |keep_dims| {
            var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 6, 7, 8 }, .f64), true);
            defer x.deinit();

            // TODO:{zt.tensor.Tensor} -- enforce AF versioning and remediate
            // Behavior of the bias parameter in af::var was changed in
            // https://git.io/Jv5gF and is different in ArrayFire v3.7. If isbiased is
            // true, sample variance rather than population variance is used. The
            // zigTensor API implements the opposite behavior to be consistent with
            // other libraries.
            const af_var_bias_arg = !b;

            var expected_var = try zt.tensor.variance(allocator, x.tensor(), &.{1}, af_var_bias_arg, keep_dims);
            defer expected_var.deinit();
            var calculated_var = try variance(allocator, x, &.{1}, b, keep_dims);
            defer calculated_var.deinit();
            try std.testing.expect(try zt.tensor.allClose(allocator, expected_var, calculated_var.tensor(), 1e-5));
            const ctx = try allocator.create(TestCtx);
            defer allocator.destroy(ctx);
            ctx.* = .{ .b = b, .keep_dims = keep_dims };

            const funcVar = (struct {
                pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                    const v_ctx: *TestCtx = @ptrCast(@alignCast(c.?));
                    return variance(alloc, input, &.{ 1, 2 }, v_ctx.b, v_ctx.keep_dims);
                }
            }).call;
            try std.testing.expect(try jacobianTestImpl(allocator, funcVar, x, ctx, 1e-5, 1e-5));
        }
    }
}

test "AutogradReductionTest -> Norm" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    const TestCtx = struct { keep_dims: bool };

    const handle_dims: []const bool = &.{ false, true };
    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 3 }, .f64), true);
    defer x.deinit();
    const ctx = try allocator.create(TestCtx);
    defer allocator.destroy(ctx);
    for (handle_dims) |keep_dims| {
        ctx.* = .{ .keep_dims = keep_dims };
        const funcNorm2 = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const n_ctx: *TestCtx = @ptrCast(@alignCast(c.?));
                return norm(alloc, input, &.{1}, 2, n_ctx.keep_dims);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, funcNorm2, x, ctx, 1e-4, 1e-4));
        const funcNorm1 = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const n_ctx: *TestCtx = @ptrCast(@alignCast(c.?));
                return norm(alloc, input, &.{1}, 1, n_ctx.keep_dims);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, funcNorm1, x, ctx, 1e-4, 1e-4));
        const funcNorm3 = (struct {
            pub fn call(alloc: std.mem.Allocator, input: *Variable, c: ?*anyopaque) !*Variable {
                const n_ctx: *TestCtx = @ptrCast(@alignCast(c.?));
                return norm(alloc, input, &.{1}, 3, n_ctx.keep_dims);
            }
        }).call;
        try std.testing.expect(try jacobianTestImpl(allocator, funcNorm3, x, ctx, 1e-4, 1e-4));
    }
}

// TODO: fn testRnnImpl() !void {}

// TODO: test "AutogradRnnTest -> Rnn" {}

// TODO: test "AutogradRnnTest -> Lstm" {}

// TODO: test "AutogradRnnTest -> Gru" {}

// TODO: test "AutogradTestF16 -> RnnF16" {}

// TODO: test "AutogradTestF16 -> LstmF16" {}

// TODO: test "AutogradTestF16 -> GruF16" {}

test "AutogradUnaryOpsTest -> Clamp" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    var in_tensor = try zt.tensor.rand(allocator, &.{ 5, 6, 7, 4 }, .f64);
    try in_tensor.inPlaceMul(allocator, f64, 3);
    var input = try Variable.init(allocator, in_tensor, true);
    defer input.deinit();
    const lo: f64 = -1;
    const hi: f64 = 1;
    const perturb: f32 = 1e-5;
    // Need to do this as gradient is not continuous when input = lo / hi.
    var inarr = input.tensor();
    var tmp1 = try zt.tensor.sub(allocator, Tensor, inarr, f64, lo);
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.abs(allocator, tmp1);
    defer tmp2.deinit();
    var tmp3 = try zt.tensor.greaterThan(allocator, Tensor, tmp2, f32, perturb);
    defer tmp3.deinit();
    var tmp4 = try zt.tensor.where(allocator, tmp3, Tensor, inarr, f64, lo + 10 * @as(f64, @floatCast(perturb)));
    defer tmp4.deinit();
    try inarr.assign(allocator, Tensor, tmp4);
    var tmp5 = try zt.tensor.sub(allocator, Tensor, inarr, f64, hi);
    defer tmp5.deinit();
    var tmp6 = try zt.tensor.abs(allocator, tmp5);
    defer tmp6.deinit();
    var tmp7 = try zt.tensor.greaterThan(allocator, Tensor, tmp6, f32, perturb);
    defer tmp7.deinit();
    var tmp8 = try zt.tensor.where(allocator, tmp7, Tensor, inarr, f64, hi + 10 * @as(f64, @floatCast(perturb)));
    defer tmp8.deinit();
    try inarr.assign(allocator, Tensor, tmp8);

    const TestCtx = struct { lo: f64, hi: f64 };
    const ctx = try allocator.create(TestCtx);
    defer allocator.destroy(ctx);
    ctx.* = .{ .lo = lo, .hi = hi };
    const fnCol = (struct {
        pub fn call(alloc: std.mem.Allocator, in: *Variable, c: ?*anyopaque) !*Variable {
            const c_ctx: *TestCtx = @ptrCast(@alignCast(c.?));
            return clamp(alloc, in, c_ctx.lo, c_ctx.hi);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, fnCol, input, ctx, 1e-10, perturb));
}

test "AutogradUnaryOpsTest -> Glu" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 3, 4, 5 }, .f64), true);
    defer in.deinit();
    const funcGlu = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return gatedlinearunit(alloc, input, 1);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcGlu, in, null, 1e-5, 1e-4));
}

test "AutogradUnaryOpsTest -> Sigmoid" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer x.deinit();
    var y = try sigmoid(allocator, x);
    defer y.deinit();
    var dy = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dy.deinit();
    try y.backwardWithGrad(allocator, dy, false);
    var dx = try x.grad();
    var tmp1 = try zt.tensor.sub(allocator, f64, 1, Tensor, y.tensor());
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.mul(allocator, Tensor, y.tensor(), Tensor, tmp1);
    defer tmp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), tmp2, 1e-5));
    var tmp3 = try zt.tensor.sigmoid(allocator, x.tensor());
    defer tmp3.deinit();
    var tmp4 = try zt.tensor.sub(allocator, f64, 1, Tensor, tmp3);
    defer tmp4.deinit();
    var tmp5 = try zt.tensor.sigmoid(allocator, x.tensor());
    defer tmp5.deinit();
    var tmp6 = try zt.tensor.mul(allocator, Tensor, tmp5, Tensor, tmp4);
    defer tmp6.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), tmp6, 1e-5));
}

test "AutogradUnaryOpsTest -> Erf" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer x.deinit();
    var y = try erf(allocator, x);
    defer y.deinit();
    var tmp1 = try zt.tensor.erf(allocator, x.tensor());
    defer tmp1.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, y.tensor(), tmp1, 1e-5));

    var dy = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dy.deinit();
    try y.backwardWithGrad(allocator, dy, false);
    var tmp2 = try mul(allocator, *Variable, x, *Variable, x);
    defer tmp2.deinit();
    var tmp3 = try negate(allocator, tmp2);
    defer tmp3.deinit();
    var tmp4 = try exp(allocator, tmp3);
    defer tmp4.deinit();
    var target_grads = try mul(allocator, f64, 2 / @sqrt(@as(f64, std.math.pi)), *Variable, tmp4);
    defer target_grads.deinit();
    var dx = try x.grad();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), target_grads.tensor(), 1e-5));

    const funcErf = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return erf(alloc, input);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcErf, x, null, 5e-4, 1e-4));
}

test "AutogradUnaryOpsTest -> Tanh" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer x.deinit();
    var y = try tanh(allocator, x);
    defer y.deinit();
    var dy = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dy.deinit();
    try y.backwardWithGrad(allocator, dy, false);
    var dx = try x.grad();
    var tmp1 = try zt.tensor.mul(allocator, Tensor, y.tensor(), Tensor, y.tensor());
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.sub(allocator, f64, 1, Tensor, tmp1);
    defer tmp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), tmp2, 1e-5));
    var lhs = try zt.tensor.tanh(allocator, x.tensor());
    defer lhs.deinit();
    try lhs.inPlaceAdd(allocator, f64, 1);
    var tmp3 = try zt.tensor.tanh(allocator, x.tensor());
    defer tmp3.deinit();
    var rhs = try zt.tensor.sub(allocator, f64, 1, Tensor, tmp3);
    defer rhs.deinit();
    var tmp4 = try zt.tensor.mul(allocator, Tensor, lhs, Tensor, rhs);
    defer tmp4.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), tmp4, 1e-5));
}

test "AutogradUnaryOpsTest -> Transpose" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 6, 7, 8 }, .f32), true);
    defer in.deinit();
    var out = try transpose(allocator, in, &.{ 2, 0, 1, 3 });
    defer out.deinit();
    try out.backward(allocator, false);
    try std.testing.expect(zt.tensor.shape.eql(try (try in.grad()).shape(allocator), &.{ 5, 6, 7, 8 }));

    const funcTranspose = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return transpose(alloc, input, &.{ 1, 3, 2, 0 });
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcTranspose, in, null, 5e-4, 1e-4));

    var in2 = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 6, 7, 8, 9 }, .f32), true);
    defer in2.deinit();
    var out2 = try transpose(allocator, in2, &.{});
    defer out2.deinit();
    try out2.backward(allocator, false);
    try std.testing.expect(zt.tensor.shape.eql(try (try in2.grad()).shape(allocator), &.{ 6, 7, 8, 9 }));

    const funcTranspose2 = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return transpose(alloc, input, &.{});
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcTranspose2, in2, null, 5e-4, 1e-4));
}

test "AutogradUnaryOpsTest -> Exp" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer x.deinit();
    var y = try exp(allocator, x);
    defer y.deinit();
    var dy = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
    defer dy.deinit();
    try y.backwardWithGrad(allocator, dy, false);
    var dx = try x.grad();
    var tmp = try zt.tensor.exp(allocator, x.tensor());
    defer tmp.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), tmp, 1e-5));
}

test "AutogradUnaryOpsTest -> Log1p" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
    defer x.deinit();
    var y = try log1p(allocator, x);
    defer y.deinit();

    var x_copy = try Variable.initSharedData(allocator, x.shared_data.retain(), true);
    defer x_copy.deinit();
    var tmp1 = try add(allocator, f64, 1, *Variable, x_copy);
    defer tmp1.deinit();
    var y_exp = try log(allocator, tmp1);
    defer y_exp.deinit();

    try y.backward(allocator, false);
    try y_exp.backward(allocator, false);

    try std.testing.expect(try zt.tensor.allClose(allocator, y.tensor(), y_exp.tensor(), 1e-5));
    try std.testing.expect(try zt.tensor.allClose(allocator, (try y.grad()).tensor(), (try y_exp.grad()).tensor(), 1e-5));
    try std.testing.expect(try zt.tensor.allClose(allocator, (try x.grad()).tensor(), (try x_copy.grad()).tensor(), 1e-5));
}

test "AutogradUnaryOpsTest -> Softmax" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 3, 5, 1 }, .f64), true);
    defer in.deinit();
    const funcSm = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return softmax(alloc, input, 0);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcSm, in, null, 1e-5, 1e-4));
}

test "AutogradTestF16 -> SoftmaxF16" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f16Supported(allocator)) {
        return error.SkipZigTest;
    }
    AutogradTestF16.setUp(); // set optim mode

    defer AutogradTestF16.tearDown(); // reset optim mode

    var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 3, 5, 1 }, .f16), true);
    defer in.deinit();
    const funcSm = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return softmax(alloc, input, 0);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcSm, in, null, 1e-2, 1e-1));
}

test "AutogradUnaryOpsTest -> LogSoftmax" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 3, 5, 1 }, .f64), true);
    defer in.deinit();
    const funcLsm = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return logSoftmax(alloc, input, 0);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcLsm, in, null, 1e-5, 1e-4));
}

test "AutogradTestF16 -> LogSoftmaxF16" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f16Supported(allocator)) {
        return error.SkipZigTest;
    }
    AutogradTestF16.setUp(); // set optim mode

    defer AutogradTestF16.tearDown(); // reset optim mode

    var in = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 3, 5, 1 }, .f16), true);
    defer in.deinit();
    const funcLsm = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return logSoftmax(alloc, input, 0);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcLsm, in, null, 1e-2, 1e-1));
}

test "AutogradUnaryOpsTest -> Pow" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    {
        var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
        defer x.deinit();
        var y = try pow(allocator, x, 2);
        defer y.deinit();
        var dy = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 2, .f32), false);
        defer dy.deinit();
        try y.backwardWithGrad(allocator, dy, false);
        var dx = try x.grad();
        var tmp1 = try zt.tensor.mul(allocator, Tensor, x.tensor(), f64, 2);
        defer tmp1.deinit();
        try tmp1.inPlaceMul(allocator, f64, 2);
        try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), tmp1, 1e-5));
    }
    {
        var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{5}, .f32), true);
        defer x.deinit();
        var y = try pow(allocator, x, 3);
        defer y.deinit();
        var dy = try Variable.init(allocator, try zt.tensor.full(allocator, &.{5}, f64, 1, .f32), false);
        defer dy.deinit();
        try y.backwardWithGrad(allocator, dy, false);
        var dx = try x.grad();
        var tmp1 = try zt.tensor.power(allocator, Tensor, x.tensor(), f64, 2);
        defer tmp1.deinit();
        try tmp1.inPlaceMul(allocator, f64, 3);
        try std.testing.expect(try zt.tensor.allClose(allocator, dx.tensor(), tmp1, 1e-5));
    }
}

test "AutogradUnaryOpsTest -> Sqrt" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    if (!try zt.common.f64Supported(allocator)) {
        return error.SkipZigTest;
    }

    var x = try Variable.init(allocator, try zt.tensor.rand(allocator, &.{ 5, 3 }, .f64), true);
    defer x.deinit();
    const funcSqrt = (struct {
        pub fn call(alloc: std.mem.Allocator, input: *Variable, _: ?*anyopaque) !*Variable {
            return sqrt(alloc, input);
        }
    }).call;
    try std.testing.expect(try jacobianTestImpl(allocator, funcSqrt, x, null, 1e-3, 1e-4));
}
