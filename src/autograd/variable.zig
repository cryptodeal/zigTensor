const std = @import("std");
const zt = @import("../zt.zig");
const Arc = @import("zigrc").Arc;

const assert = std.debug.assert;
const Shape = zt.tensor.shape.Shape;
const Tensor = zt.tensor.Tensor;
const DType = zt.tensor.DType;
const Index = zt.tensor.Index;
const Dim = zt.tensor.shape.Dim;

/// Variable wraps an Arrayfire array and facilitates easy backpropagation.
///
/// Variable is a wrapper around Arrayfire array and supports many operations
/// (Functions) on arrays. When a Function is applied on input Variable(s), the
/// output Variable(s) records it's inputs and a gradient function which
/// can be used to compute gradient propagated from output to each of its
/// inputs.
///
/// Thus, Variable and Functions build a computation graph which is a DAG.
/// Following chain rule, backpropagation through this graph can be done easily
/// by traversing in a topologically sorted order starting from output
/// Variable(s).
pub const Variable = struct {
    pub const GradFunc = *const fn (allocator: std.mem.Allocator, inputs: []const *Variable, grad_ouput: *Variable, ctx: ?*anyopaque) anyerror!void;
    pub const FreeGradFuncCtx = *const fn (allocator: std.mem.Allocator, ctx: *anyopaque) void;
    pub const GradHook = *const fn (allocator: std.mem.Allocator, grad: *Variable) anyerror!void;
    pub const DAG = []*Variable;

    pub const SharedData = struct {
        /// Array wrapped by this variable
        data: Tensor = undefined,

        pub fn deinit(self: SharedData) void {
            self.data.deinit();
        }

        // TODO: ztSaveLoad
    };

    pub const SharedGrad = struct {
        /// Whether the gradient should be computed for this Variable
        calc_grad: bool = false,
        /// Inputs of this Variable
        inputs: []*Variable = &[_]*Variable{},
        /// Gradient with respect to this Variable
        grad: ?*Variable = null,
        /// Function for calculating the gradient of the input Variables
        grad_func: ?GradFunc = null,
        /// Grad function context
        grad_func_ctx: ?*anyopaque = null,
        /// Function for deinitializing grad function context
        deinit_grad_func_ctx: ?FreeGradFuncCtx = null,
        /// Function applied to gradient after it's computed during bwd pass
        on_grad_available: ?GradHook = null,
        allocator: std.mem.Allocator = undefined,

        pub fn deinit(self: SharedGrad) void {
            if (self.grad != null) {
                self.grad.?.deinit();
            }
            if (self.inputs.len > 0) {
                for (self.inputs) |in| in.deinit();
                self.allocator.free(self.inputs);
            }
            if (self.grad_func_ctx != null and self.deinit_grad_func_ctx != null) {
                self.deinit_grad_func_ctx.?(self.allocator, self.grad_func_ctx.?);
            }
        }
    };

    shared_data: Arc(SharedData),
    shared_grad: Arc(SharedGrad),
    allocator: std.mem.Allocator = undefined,

    pub fn initEmpty(allocator: std.mem.Allocator) !*Variable {
        var self = try allocator.create(Variable);
        self.* = .{
            .shared_data = try Arc(SharedData).init(allocator, .{ .data = try Tensor.initEmpty(allocator) }),
            .shared_grad = try Arc(SharedGrad).init(allocator, .{}),
            .allocator = allocator,
        };
        return self;
    }

    pub fn init(allocator: std.mem.Allocator, data: Tensor, calc_grad: bool) !*Variable {
        var self = try allocator.create(Variable);
        self.* = .{
            .shared_data = try Arc(SharedData).init(allocator, .{ .data = data }),
            .shared_grad = try Arc(SharedGrad).init(allocator, .{ .calc_grad = calc_grad }),
            .allocator = allocator,
        };
        return self;
    }

    pub fn initSharedData(allocator: std.mem.Allocator, data: Arc(SharedData), calc_grad: bool) !*Variable {
        var self = try allocator.create(Variable);
        self.* = .{
            .shared_data = data,
            .shared_grad = try Arc(SharedGrad).init(allocator, .{ .calc_grad = calc_grad }),
            .allocator = allocator,
        };
        return self;
    }

    pub fn initWithInputs(
        allocator: std.mem.Allocator,
        data: Tensor,
        inputs: []const *Variable,
        grad_func: GradFunc,
        grad_func_ctx: ?*anyopaque,
        deinit_grad_func_ctx: ?FreeGradFuncCtx,
    ) !*Variable {
        var self = try allocator.create(Variable);
        self.* = .{
            .shared_data = try Arc(SharedData).init(allocator, .{ .data = data }),
            .shared_grad = try Arc(SharedGrad).init(allocator, .{}),
            .allocator = allocator,
        };
        var is_calc_required = false;
        for (inputs) |in| {
            if (in.isCalcGrad()) is_calc_required = true;
        }
        if (is_calc_required) {
            self.shared_grad.value.calc_grad = true;
            self.shared_grad.value.allocator = allocator;
            self.shared_grad.value.inputs = try allocator.alloc(*Variable, inputs.len);
            @memcpy(self.shared_grad.value.inputs, inputs);
            self.shared_grad.value.grad_func = grad_func;
            self.shared_grad.value.grad_func_ctx = grad_func_ctx;
            self.shared_grad.value.deinit_grad_func_ctx = deinit_grad_func_ctx;
        } else {
            for (inputs) |in| {
                in.deinit();
            }
            if (grad_func_ctx != null and deinit_grad_func_ctx != null) {
                deinit_grad_func_ctx.?(allocator, grad_func_ctx.?);
            }
        }
        return self;
    }

    pub fn deinit(self: *Variable) void {
        self.shared_data.releaseWithFn(SharedData.deinit);
        self.shared_grad.releaseWithFn(SharedGrad.deinit);
        self.allocator.destroy(self);
    }

    pub fn clone(self: *Variable, allocator: std.mem.Allocator) !*Variable {
        var new_var = try allocator.create(Variable);
        new_var.* = .{
            .shared_data = self.shared_data.retain(),
            .shared_grad = self.shared_grad.retain(),
            .allocator = allocator,
        };
        return new_var;
    }

    pub fn index(self: *Variable, allocator: std.mem.Allocator, indices: []const Index) !*Variable {
        var result = try self.tensor().index(allocator, indices);
        var idx_ctx = try allocator.create(IndexCtx);
        idx_ctx.* = .{
            .indices = try allocator.alloc(Index, indices.len),
            .in_dims = try self.shape(allocator),
            .in_type = try self.dtype(allocator),
        };
        @memcpy(idx_ctx.indices, indices);
        return Variable.initWithInputs(allocator, result, &.{try self.withoutData(allocator)}, indexGradFunc, idx_ctx, freeIndexCtx);
    }

    pub fn flat(self: *const Variable, allocator: std.mem.Allocator, idx: Index) !*Variable {
        var result = try (self.tensor()).flat(allocator, idx);
        var flat_ctx = try allocator.create(FlatCtx);
        flat_ctx.* = .{
            .index = idx,
            .in_dims = try self.shape(allocator),
            .in_type = try self.dtype(allocator),
        };
        return Variable.initWithInputs(allocator, result, &.{try self.withoutData(allocator)}, flatGradFunc, flat_ctx, freeFlatCtx);
    }

    pub fn tensor(self: *const Variable) Tensor {
        return self.shared_data.value.data;
    }

    pub fn astype(self: *Variable, allocator: std.mem.Allocator, new_type: DType) !*Variable {
        var output = try self.tensor().astype(allocator, new_type);
        return Variable.initWithInputs(allocator, output, &.{try self.withoutData(allocator)}, astypeGradFunc, null, null);
    }

    pub fn grad(self: *const Variable) !*Variable {
        if (!self.shared_grad.value.calc_grad) {
            std.debug.print("gradient calculation disabled for this Variable\n", .{});
            return error.GradientCalcDisabled;
        }
        if (self.shared_grad.value.grad == null) {
            std.debug.print("gradient not calculated yet for this Variable\n", .{});
            return error.GradientNotCalculated;
        }
        return self.shared_grad.value.grad.?;
    }

    pub fn getInputs(self: *const Variable) []*Variable {
        return self.shared_grad.value.inputs;
    }

    pub fn isCalcGrad(self: *const Variable) bool {
        return self.shared_grad.value.calc_grad;
    }

    pub fn isGradAvailable(self: *const Variable) bool {
        if (!self.shared_grad.value.calc_grad) {
            return false;
        }
        return self.shared_grad.value.grad != null;
    }

    pub fn shape(self: *const Variable, allocator: std.mem.Allocator) !Shape {
        return (self.tensor()).shape(allocator);
    }

    pub fn isEmpty(self: *const Variable, allocator: std.mem.Allocator) !bool {
        return (self.tensor()).isEmpty(allocator);
    }

    pub fn isContiguous(self: *const Variable, allocator: std.mem.Allocator) !bool {
        return (self.tensor()).isContiguous(allocator);
    }

    // TODO: might not want this done in place
    pub fn asContiguous(self: *const Variable, allocator: std.mem.Allocator) !*Variable {
        if (!try self.isEmpty(allocator) and !try self.isContiguous(allocator)) {
            var tmp_contig = try self.tensor().asContiguousTensor(allocator);
            defer tmp_contig.deinit();
            try self.tensor().assign(allocator, Tensor, tmp_contig);
        }
        return self;
    }

    pub fn dtype(self: *const Variable, allocator: std.mem.Allocator) !DType {
        return self.tensor().dtype(allocator);
    }

    pub fn elements(self: *const Variable, allocator: std.mem.Allocator) !Dim {
        return self.tensor().elements(allocator);
    }

    pub fn bytes(self: *const Variable, allocator: std.mem.Allocator) !usize {
        return self.tensor().bytes(allocator);
    }

    pub fn ndim(self: *const Variable, allocator: std.mem.Allocator) !usize {
        return self.tensor().ndim(allocator);
    }

    pub fn dim(self: *const Variable, allocator: std.mem.Allocator, dimension: usize) !Dim {
        return self.tensor().dim(allocator, dimension);
    }

    pub fn eval(self: *const Variable, allocator: std.mem.Allocator) !void {
        try zt.tensor.eval(allocator, self.tensor());
    }

    pub fn zeroGrad(self: *Variable) void {
        if (self.shared_grad.value.grad != null) {
            self.shared_grad.value.grad.?.deinit();
            self.shared_grad.value.grad = null;
        }
    }

    pub fn setCalcGrad(self: *Variable, calc_grad: bool) void {
        self.shared_grad.value.calc_grad = calc_grad;
        if (!calc_grad) {
            self.shared_grad.value.grad_func = null;
            for (self.shared_grad.value.inputs) |in| {
                in.deinit();
            }
            self.shared_grad.value.allocator.free(self.shared_grad.value.inputs);
            self.shared_grad.value.inputs = &[_]*Variable{};
            self.shared_grad.value.grad.?.deinit();
            self.shared_grad.value.grad = null;
        }
    }

    pub fn addGrad(self: *Variable, allocator: std.mem.Allocator, child_grad: *Variable) !void {
        if (self.shared_grad.value.calc_grad) {
            // Ensure the type of the child grad is the same as the type of this
            // Variable (and transitively, that it's the same type as an existing grad)
            if (try child_grad.dtype(allocator) != try self.dtype(allocator)) {
                std.debug.print(
                    "Variable.addGrad: attempted to add child gradient of type {s}  to a Variable of type {s}. You might be performing an operation with two inputs of different types.\n",
                    .{ @tagName(try child_grad.dtype(allocator)), @tagName(try self.dtype(allocator)) },
                );
                return error.GradientTypeMismatch;
            }
            if (!zt.tensor.shape.eql(try child_grad.shape(allocator), try self.shape(allocator))) {
                std.debug.print(
                    "Variable.addGrad: given gradient has dimensions not equal to this Variable's dimensions: this variable has shape {any} whereas the child gradient has dimensions {any}\n",
                    .{ try self.shape(allocator), try child_grad.shape(allocator) },
                );
                return error.GradientDimsMismatch;
            }
            if (self.shared_grad.value.grad != null) {
                var tmp = self.shared_grad.value.grad.?;
                defer tmp.deinit();
                // Prevent increment of array refcount to avoid a copy
                // if getting a device pointer. See
                // https://git.io/fp9oM for more
                self.shared_grad.value.grad = try Variable.init(
                    allocator,
                    try zt.tensor.add(allocator, Tensor, tmp.tensor(), Tensor, child_grad.tensor()),
                    false,
                );
            } else {
                // Copy the child_grad Variable so as to share a reference
                // to the underlying child_grad.tensor() rather than copying
                // the tensor into a new variable
                self.shared_grad.value.grad = try child_grad.clone(allocator);
            }
        }
    }

    pub fn registerGradHook(self: *Variable, hook: GradHook) !void {
        self.shared_grad.value.on_grad_available = hook;
    }

    pub fn clearGradHook(self: *Variable) !void {
        self.shared_grad.value.on_grad_available = null;
    }

    pub fn applyGradHook(self: *Variable, allocator: std.mem.Allocator) !void {
        if (self.shared_grad.value.on_grad_available != null) {
            assert(self.shared_grad.value.grad != null);
            try self.shared_grad.value.on_grad_available.?(allocator, self.shared_grad.value.grad.?);
        }
    }

    pub fn calcGradInputs(self: *Variable, allocator: std.mem.Allocator) !void {
        if (self.shared_grad.value.grad_func != null) {
            if (self.shared_grad.value.grad == null) {
                std.debug.print("gradient was not propagated to this Variable\n", .{});
                return error.GradientNotPropagated;
            }

            try self.shared_grad.value.grad_func.?(allocator, self.shared_grad.value.inputs, self.shared_grad.value.grad.?, self.shared_grad.value.grad_func_ctx);
        }
    }

    pub fn backwardWithGrad(self: *Variable, allocator: std.mem.Allocator, grad_: *Variable, retain_graph: bool) !void {
        try self.addGrad(allocator, grad_);
        var dag = try self.build(allocator);
        defer allocator.free(dag);
        var i: usize = dag.len;
        while (i > 0) {
            i -= 1;
            var iter = dag[i];
            try iter.calcGradInputs(allocator);
            try iter.applyGradHook(allocator);
            if (!retain_graph) {
                defer iter.deinit();
            }
        }
    }

    pub fn backward(self: *Variable, allocator: std.mem.Allocator, retain_graph: bool) !void {
        var ones = try Variable.init(
            allocator,
            try zt.tensor.full(
                allocator,
                try self.shape(allocator),
                i8,
                1,
                try self.dtype(allocator),
            ),
            false,
        );
        defer ones.deinit();
        try self.backwardWithGrad(allocator, ones, retain_graph);
    }

    pub fn withoutData(self: *Variable, allocator: std.mem.Allocator) !*Variable {
        var other = try allocator.create(Variable);
        other.* = .{
            .shared_data = try Arc(SharedData).init(allocator, .{ .data = try Tensor.initHandle(allocator, try self.shape(allocator), try self.dtype(allocator)) }),
            .shared_grad = self.shared_grad.retain(),
            .allocator = allocator,
        };
        return other;
    }

    pub fn build(self: *Variable, allocator: std.mem.Allocator) !DAG {
        var cache = std.AutoHashMap(*SharedData, void).init(allocator);
        defer cache.deinit();
        var dag = std.ArrayList(*Variable).init(allocator);
        try recurse(allocator, self, &cache, &dag);
        return dag.toOwnedSlice();
    }
};

// Topological sort
fn recurse(allocator: std.mem.Allocator, variable: *Variable, cache: *std.AutoHashMap(*Variable.SharedData, void), dag: *std.ArrayList(*Variable)) !void {
    var id = variable.shared_data.value;
    if (cache.contains(id)) return;
    for (variable.getInputs()) |input| {
        try recurse(allocator, input, cache, dag);
    }
    try cache.put(id, {});
    try dag.append(try variable.clone(allocator));
}

const IndexCtx = struct {
    indices: []Index,
    in_dims: Shape,
    in_type: DType,
};

const FlatCtx = struct {
    index: Index,
    in_dims: Shape,
    in_type: DType,
};

fn freeIndexCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var idx_ctx: *IndexCtx = @ptrCast(@alignCast(ctx));
    allocator.free(idx_ctx.indices);
    allocator.destroy(idx_ctx);
}

fn freeFlatCtx(allocator: std.mem.Allocator, ctx: *anyopaque) void {
    var flat_ctx: *FlatCtx = @ptrCast(@alignCast(ctx));
    allocator.destroy(flat_ctx);
}

fn indexGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var idx_ctx: *IndexCtx = @ptrCast(@alignCast(ctx));
    if (!inputs[0].isGradAvailable()) {
        var grad = try zt.tensor.full(allocator, idx_ctx.in_dims, i8, 0, idx_ctx.in_type);
        var grad_var = try Variable.init(allocator, grad, false);
        defer grad_var.deinit();
        try inputs[0].addGrad(allocator, grad_var);
    }
    var grad = (try inputs[0].grad()).tensor();
    try grad.indexAdd(allocator, Tensor, grad_output.tensor(), idx_ctx.indices);
}

fn flatGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, ctx: ?*anyopaque) !void {
    var idx_ctx: *FlatCtx = @ptrCast(@alignCast(ctx));
    if (!inputs[0].isGradAvailable()) {
        var tmp_var = try Variable.init(
            allocator,
            try zt.tensor.full(allocator, idx_ctx.in_dims, i8, 0, idx_ctx.in_type),
            false,
        );
        defer tmp_var.deinit();
        try inputs[0].addGrad(allocator, tmp_var);
    }
    var grad = (try inputs[0].grad()).tensor();
    try grad.flatAdd(allocator, Tensor, grad_output.tensor(), idx_ctx.index);
}

fn astypeGradFunc(allocator: std.mem.Allocator, inputs: []const *Variable, grad_output: *Variable, _: ?*anyopaque) !void {
    var input = inputs[0];
    // Cast the grad output to match the type of the input's grad
    var tmp_var = try Variable.init(allocator, try grad_output.tensor().astype(allocator, try input.dtype(allocator)), false);
    defer tmp_var.deinit();
    try input.addGrad(allocator, tmp_var);
}
