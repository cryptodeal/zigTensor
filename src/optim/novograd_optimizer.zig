const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;
const Variable = zt.autograd.Variable;

/// An implementation of the Novograd optimizer.
///
/// For more details see the paper
/// [Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks](https://arxiv.org/abs/1905.11286).
pub const NovogradOptimizer = struct {
    allocator: std.mem.Allocator,
    parameters_: []*Variable,
    lr_: f64,
    beta1_: f32,
    beta2_: f32,
    eps_: f32,
    wd_: f32,
    acc_grad_norm_: []f64,
    acc_grad_: []Tensor,

    pub fn init(
        allocator: std.mem.Allocator,
        parameters: []const *Variable,
        learning_rate: f64,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) !*NovogradOptimizer {
        const self = try allocator.create(NovogradOptimizer);
        self.* = .{
            .allocator = allocator,
            .parameters_ = try allocator.alloc(*Variable, parameters.len),
            .lr_ = learning_rate,
            .beta1_ = beta1,
            .beta2_ = beta2,
            .eps_ = epsilon,
            .wd_ = weight_decay,
            .acc_grad_norm_ = try allocator.alloc(f64, parameters.len),
            .acc_grad_ = try allocator.alloc(Tensor, parameters.len),
        };
        @memcpy(self.parameters_, parameters);
        for (self.parameters_, 0..) |parameter, i| {
            self.acc_grad_norm_[i] = 0;
            self.acc_grad_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));
            try zt.tensor.eval(allocator, self.acc_grad_[i]);
        }
        return self;
    }

    pub fn deinit(self: *NovogradOptimizer) void {
        for (self.parameters_) |v| v.deinit();
        self.allocator.free(self.parameters_);
        self.allocator.free(self.acc_grad_norm_);
        for (self.acc_grad_) |t| t.deinit();
        self.allocator.free(self.acc_grad_);
        self.allocator.destroy(self);
    }

    /// Zero the gradients for all the parameters being optimized. Typically
    /// called after every call to step().
    pub fn zeroGrad(self: *NovogradOptimizer) void {
        for (self.parameters_) |parameter| {
            parameter.zeroGrad();
        }
    }

    /// Get the learning rate.
    pub fn getLr(self: *const NovogradOptimizer) f64 {
        return self.lr_;
    }

    /// Set the learning rate.
    pub fn setLr(self: *NovogradOptimizer, lr: f64) void {
        self.lr_ = lr;
    }

    pub fn step(self: *NovogradOptimizer, allocator: std.mem.Allocator) !void {
        for (0..self.parameters_.len) |i| {
            if (!self.parameters_[i].isGradAvailable()) {
                continue;
            }

            const grad = (try self.parameters_[i].grad()).tensor();
            const data = self.parameters_[i].tensor();
            const acc_grad = self.acc_grad_[i];

            var tmp1 = try zt.tensor.mul(allocator, Tensor, grad, Tensor, grad);
            var tmp2 = try zt.tensor.sum(allocator, tmp1, &.{}, false);
            tmp1.deinit();
            const grad_norm = try tmp2.asScalar(allocator, f64);
            tmp2.deinit();
            self.acc_grad_norm_[i] = @as(f64, @floatCast(self.beta2_)) * self.acc_grad_norm_[i] + (1 - @as(f64, @floatCast(self.beta2_))) * grad_norm;

            tmp1 = try zt.tensor.div(allocator, Tensor, grad, f32, @as(f32, @floatCast(@sqrt(self.acc_grad_norm_[i]))) + self.eps_);
            tmp2 = try zt.tensor.mul(allocator, f32, self.wd_, Tensor, data);
            try tmp1.inPlaceAdd(allocator, Tensor, tmp2);
            tmp2.deinit();
            try tmp1.inPlaceMul(allocator, f32, 1 - self.beta1_);
            try acc_grad.inPlaceMul(allocator, f32, self.beta1_);
            try acc_grad.inPlaceAdd(allocator, Tensor, tmp1);
            tmp1.deinit();

            try zt.tensor.eval(allocator, acc_grad);

            tmp1 = try zt.tensor.mul(allocator, f64, self.lr_, Tensor, acc_grad);
            try data.inPlaceSub(allocator, Tensor, tmp1);

            try zt.tensor.eval(allocator, data);
        }
    }

    pub fn prettyString(self: *NovogradOptimizer, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().write("Novograd");
        if (self.wd_ != 0) {
            try buffer.writer().print(" (weight decay={d})", .{self.wd_});
        }
        return buffer.toOwnedSlice();
    }
};
