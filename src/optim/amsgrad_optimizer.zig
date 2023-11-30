const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;
const Variable = zt.autograd.Variable;

/// An implementation of the AMSgrad optimizer.
///
/// For more details see the paper [On the Convergence of Adam and Beyond](https://openreview.net/pdf?id=ryQu7f-RZ).
pub const AMSgradOptimizer = struct {
    allocator: std.mem.Allocator,
    parameters_: []*Variable,
    lr_: f64,
    beta1_: f32,
    beta2_: f32,
    eps_: f32,
    wd_: f32,
    biased_first_: []Tensor,
    biased_second_: []Tensor,
    max_exp_avg_sq_: []Tensor,

    pub fn init(
        allocator: std.mem.Allocator,
        parameters: []const *Variable,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) !*AMSgradOptimizer {
        const self = try allocator.create(AMSgradOptimizer);
        self.* = .{
            .allocator = allocator,
            .parameters_ = try allocator.alloc(*Variable, parameters.len),
            .lr_ = @floatCast(learning_rate),
            .beta1_ = beta1,
            .beta2_ = beta2,
            .eps_ = epsilon,
            .wd_ = weight_decay,
            .biased_first_ = try std.ArrayList(Tensor).initCapacity(allocator, parameters.len),
            .biased_second_ = try std.ArrayList(Tensor).initCapacity(allocator, parameters.len),
            .max_exp_avg_sq_ = try std.ArrayList(Tensor).initCapacity(allocator, parameters.len),
        };
        @memcpy(self.parameters_, parameters);
        for (self.parameters_, 0..) |parameter, i| {
            self.biased_first_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));
            self.biased_second_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));
            self.max_exp_avg_sq_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));

            try zt.tensor.eval(allocator, self.biased_first[i]);
            try zt.tensor.eval(allocator, self.biased_second[i]);
            try zt.tensor.eval(allocator, self.max_exp_avg_sq[i]);
        }
        return self;
    }

    pub fn deinit(self: *AMSgradOptimizer) void {
        for (self.parameters_) |v| v.deinit();
        self.allocator.free(self.parameters_);
        for (self.biased_first_) |t| t.deinit();
        self.allocator.free(self.biased_first_);
        for (self.biased_second_) |t| t.deinit();
        self.allocator.free(self.biased_second_);
        for (self.max_exp_avg_sq_) |t| t.deinit();
        self.allocator.free(self.max_exp_avg_sq_);
        self.allocator.destroy(self);
    }

    /// Zero the gradients for all the parameters being optimized. Typically
    /// called after every call to step().
    pub fn zeroGrad(self: *AMSgradOptimizer) void {
        for (self.parameters_) |parameter| {
            parameter.zeroGrad();
        }
    }

    /// Get the learning rate.
    pub fn getLr(self: *const AMSgradOptimizer) f64 {
        return self.lr_;
    }

    /// Set the learning rate.
    pub fn setLr(self: *AMSgradOptimizer, lr: f64) void {
        self.lr_ = lr;
    }

    pub fn step(self: *AMSgradOptimizer, allocator: std.mem.Allocator) !void {
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
                try data.inPlaceSub(allocator, Tensor, tmp);
            }

            const biased_first = self.biased_first_[i];
            const biased_second = self.biased_second_[i];
            const max_exp_avg_sq = self.max_exp_avg_sq_[i];

            var tmp = try zt.tensor.mul(allocator, f32, 1 - self.beta1_, Tensor, grad);
            try biased_first.inPlaceMul(allocator, f32, self.beta1_);
            try biased_first.inPlaceAdd(allocator, Tensor, tmp);
            tmp.deinit();
            tmp = try zt.tensor.mul(allocator, f32, 1 - self.beta2_, Tensor, grad);
            try tmp.inPlaceMul(allocator, Tensor, grad);
            try biased_second.inPlaceMul(allocator, f32, self.beta2_);
            try biased_second.inPlaceAdd(allocator, Tensor, tmp);
            tmp.deinit();
            tmp = try zt.tensor.maximum(allocator, Tensor, max_exp_avg_sq, Tensor, biased_second);
            try max_exp_avg_sq.assign(allocator, Tensor, tmp);
            tmp.deinit();
            try zt.tensor.eval(allocator, biased_first);
            try zt.tensor.eval(allocator, biased_second);
            try zt.tensor.eval(allocator, max_exp_avg_sq);

            tmp = try zt.tensor.sqrt(allocator, max_exp_avg_sq);
            try tmp.inPlaceAdd(allocator, f32, self.eps_);
            var tmp2 = try zt.tensor.mul(allocator, f64, self.lr_, Tensor, biased_first);
            defer tmp2.deinit();
            try tmp2.inPlaceDiv(allocator, Tensor, tmp);
            tmp.deinit();
            try data.inPlaceSub(allocator, Tensor, tmp2);

            try zt.tensor.eval(allocator, data);
        }
    }

    /// Generates a stringified representation of the optimizer.
    pub fn prettyString(self: *AMSgradOptimizer, allocator: std.mem.Allocator) []const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().writeAll("AMSgrad from ");
        if (self.wd_ != 0) {
            try buffer.writer().print(" (weight decay={d})", .{self.wd_});
        }
        return buffer.toOwnedSlice();
    }
};
