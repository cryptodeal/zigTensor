const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;
const Variable = zt.autograd.Variable;

/// An implementation of the Adadelta optimizer.
///
/// For more details see the paper [Adadelta: An Adaptive Learning Rate Method](https://arxiv.org/pdf/1212.5701.pdf).
const AdadeltaOptimizer = struct {
    allocator: std.mem.Allocator,
    parameters_: []*Variable,
    lr_: f64,
    rho_: f32,
    eps_: f32,
    wd_: f32,
    acc_grad_: []Tensor,
    acc_delta_: []Tensor,

    pub fn init(
        allocator: std.mem.Allocator,
        parameters: []const *Variable,
        learning_rate: f64,
        rho: f32,
        epsilon: f32,
        weight_decay: f32,
    ) !*AdadeltaOptimizer {
        const self = try allocator.create(AdadeltaOptimizer);
        self.* = .{
            .allocator = allocator,
            .parameters_ = try allocator.alloc(*Variable, parameters.len),
            .lr_ = learning_rate,
            .rho_ = rho,
            .eps_ = epsilon,
            .wd_ = weight_decay,
            .acc_grad_ = try allocator.alloc(Tensor, parameters.len),
            .acc_delta_ = try allocator.alloc(Tensor, parameters.len),
        };
        @memcpy(self.parameters_, parameters);
        for (self.parameters_, 0..) |parameter, i| {
            self.acc_grad_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));
            self.acc_delta_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));

            try zt.tensor.eval(allocator, self.acc_grad_[i]);
            try zt.tensor.eval(allocator, self.acc_delta_[i]);
        }
        return self;
    }

    pub fn deinit(self: *AdadeltaOptimizer) void {
        for (self.parameters_) |v| v.deinit();
        self.allocator.free(self.parameters_);
        for (self.acc_grad_) |t| t.deinit();
        self.allocator.free(self.acc_grad_);
        for (self.acc_delta_) |t| t.deinit();
        self.allocator.free(self.acc_delta_);
        self.allocator.free(self);
    }

    /// Zero the gradients for all the parameters being optimized. Typically
    /// called after every call to step().
    pub fn zeroGrad(self: *AdadeltaOptimizer) void {
        for (self.parameters_) |parameter| {
            parameter.zeroGrad();
        }
    }

    /// Get the learning rate.
    pub fn getLr(self: *const AdadeltaOptimizer) f64 {
        return self.lr_;
    }

    /// Set the learning rate.
    pub fn setLr(self: *AdadeltaOptimizer, lr: f64) void {
        self.lr_ = lr;
    }

    pub fn step(self: *AdadeltaOptimizer, allocator: std.mem.Allocator) !void {
        for (0..self.parameters_.len) |i| {
            if (!self.parameters_[i].isGradAvailable()) {
                continue;
            }

            const grad = (try self.parameters_[i].grad()).tensor();
            var data = self.parameters_[i].tensor();

            if (self.wd_ != 0) {
                // Weight decay term
                var tmp = try zt.tensor.mul(allocator, f32, self.wd_, Tensor, data);
                defer tmp.deinit();
                try data.inPlaceSub(allocator, tmp);
            }

            const acc_grad = self.acc_grad_[i];
            const acc_delta = self.acc_delta_[i];

            var tmp1 = try zt.tensor.mul(allocator, f32, 1 - self.rho_, Tensor, grad);
            try tmp1.inPlaceMul(allocator, grad);
            try acc_grad.inPlaceMul(allocator, f32, self.rho_);
            try acc_grad.inPlaceAdd(allocator, tmp1);
            tmp1.deinit();
            try zt.tensor.eval(allocator, acc_grad);

            tmp1 = try zt.tensor.add(allocator, Tensor, acc_grad, f32, self.eps_);
            var tmp2 = try zt.tensor.sqrt(allocator, tmp1);
            tmp1.deinit();
            try tmp2.inPlaceMul(allocator, Tensor, grad);
            tmp1 = try zt.tensor.add(allocator, Tensor, acc_delta, f32, self.eps_);
            const tmp3 = try zt.tensor.sqrt(allocator, tmp1);
            tmp1.deinit();
            const delta = try zt.tensor.div(allocator, tmp3, tmp2);
            defer delta.deinit();
            tmp2.deinit();
            tmp3.deinit();

            tmp1 = try zt.tensor.mul(allocator, f64, self.lr_, Tensor, delta);
            try data.inPlaceSub(allocator, tmp1);
            tmp1.deinit();
            try zt.tensor.eval(allocator, data);

            tmp1 = try zt.tensor.mul(allocator, f32, 1 - self.rho_, Tensor, delta);
            try tmp1.inPlaceMul(allocator, delta);
            try acc_delta.inPlaceMul(allocator, f32, self.rho_);
            try acc_delta.inPlaceAdd(allocator, tmp1);
            tmp1.deinit();
            try zt.tensor.eval(allocator, acc_delta);
        }
    }

    /// Generates a stringified representation of the optimizer.
    pub fn prettyString(self: *AdadeltaOptimizer, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().write("Adadelta");
        if (self.wd_ != 0) {
            try buffer.writer().print(" (weight decay={d});", .{self.wd_});
        }
        try buffer.writer().print(" (rho={d});", .{self.rho_});
        if (self.eps_ != 0) {
            try buffer.writer().print(" (epsilon={d})", .{self.eps_});
        }
        return buffer.toOwnedSlice();
    }
};
