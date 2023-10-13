const std = @import("std");
const zt = @import("../zt.zig");
const Arc = @import("zigrc").Arc;

const Shape = zt.tensor.shape.Shape;
const Tensor = zt.tensor.Tensor;

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
    pub const GradFunc = *const fn (allocator: std.mem.Allocator, inputs: []const Variable, grad_ouput: Variable) anyerror!void;
    pub const FreeGradFuncCtx = *const fn (allocator: std.mem.Allocator, ctx: *anyopaque) void;

    pub const GradHook = *const fn (allocator: std.mem.Allocator, grad: Variable) anyerror!void;

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
        inputs: []Variable = &[_]Variable{},
        /// Gradient with respect to this Variable
        grad: ?Variable = null,
        /// Function for calculating the gradient of the input Variables
        grad_func: ?GradFunc = null,
        /// Grad function context
        grad_func_ctx: ?*anyopaque = null,
        /// Function for deinitializing grad function context
        deinit_grad_func_ctx: ?FreeGradFuncCtx = null,
        /// Function applied to gradient after it's computed during bwd pass
        grad_hook: ?GradHook = null,
        allocator: std.mem.Allocator = undefined,

        pub fn deinit(self: *const SharedGrad) void {
            if (self.inputs.len > 0) {
                self.allocator.free(self.inputs);
            }
            if (self.grad_func_ctx != null and self.deinit_grad_func_ctx != null) {
                self.deinit_grad_func_ctx.?(self.allocator, self.grad_func_ctx);
            }
        }
    };

    shared_data: Arc(SharedData) = undefined,
    shared_grad: Arc(SharedGrad) = undefined,

    pub fn initEmpty(allocator: std.mem.Allocator) !Variable {
        return .{
            .shared_data = try Arc(SharedData).init(allocator, .{ .data = try Tensor.initEmpty(allocator) }),
            .shared_grad = try Arc(SharedGrad).init(allocator, .{}),
        };
    }

    pub fn init(allocator: std.mem.Allocator, data: Tensor, calc_grad: bool) !Variable {
        return .{
            .shared_data = try Arc(SharedData).init(allocator, .{ .data = data }),
            .shared_grad = try Arc(SharedGrad).init(allocator, .{ .calc_grad = calc_grad }),
        };
    }

    pub fn initWithInputs(allocator: std.mem.Allocator, data: Tensor, inputs: []const Variable, grad_func: GradFunc) !Variable {
        var self: Variable = .{
            .shared_data = try Arc(SharedData).init(allocator, .{ .data = data }),
            .shared_grad = try Arc(SharedGrad).init(allocator, .{}),
        };
        for (inputs) |in| {
            if (!in.isCalcGrad()) continue;
            self.shared_grad.calc_grad = true;
            self.shared_grad.allocator = allocator;
            self.shared_grad.inputs = try allocator.alloc(Variable, inputs.len);
            @memcpy(self.shared_grad.inputs, inputs);
            self.shared_grad.grad_func = grad_func;
            break;
        }
        return self;
    }

    pub fn deinit(self: *const Variable) void {
        self.shared_data.deinit();
        self.shared_grad.deinit();
    }

    // TODO: pub fn index() !Variable {}

    // TODO: pub fn flat() !Variable {}

    pub fn tensor(self: *const Variable) Tensor {
        return self.shared_data.data;
    }

    // TODO: pub fn astype() !Variable {}

    pub fn grad(self: *const Variable) !*Variable {
        if (!self.shared_grad.calc_grad) {
            std.debug.print("gradient calculation disabled for this Variable\n", .{});
            return error.GradientCalcDisabled;
        }
        if (self.shared_grad.grad == null) {
            std.debug.print("gradient not calculated yet for this Variable\n", .{});
            return error.GradientNotCalculated;
        }
        return self.shared_grad.value;
    }

    pub fn getInputs(self: *const Variable) []Variable {
        return self.shared_grad.inputs;
    }

    pub fn isCalcGrad(self: *const Variable) bool {
        return self.shared_grad.calc_grad;
    }

    pub fn isGradAvailable(self: *const Variable) bool {
        if (!self.shared_grad.calc_grad) {
            return false;
        }
        return self.shared_grad.grad != null;
    }

    pub fn shape(self: *const Variable, allocator: std.mem.Allocator) !Shape {
        return (self.tensor()).shape(allocator);
    }

    pub fn isEmpty(self: *const Variable, allocator: std.mem.Allocator) !bool {
        return (self.tensor()).isEmpty(allocator);
    }
};
