const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;
const Variable = zt.autograd.Variable;

/// An implementation of the Adam optimizer.
///
/// For more details see the paper
/// [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
pub const AdamOptimizer = struct {
    allocator: std.mem.Allocator,
    parameters_: []*Variable,
    lr_: f64,
    beta1_: f32,
    beta2_: f32,
    eps_: f32,
    wd_: f32,
    count_: usize = 0,
    biased_first_: []Tensor,
    biased_second_: []Tensor,

    pub fn init(
        allocator: std.mem.Allocator,
        parameters: []const *Variable,
        learning_rate: f64,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) !*AdamOptimizer {
        const self = try allocator.create(AdamOptimizer);
        self.* = .{
            .allocator = allocator,
            .parameters_ = try allocator.alloc(*Variable, parameters.len),
            .lr_ = learning_rate,
            .beta1_ = beta1,
            .beta2_ = beta2,
            .eps_ = epsilon,
            .wd_ = weight_decay,
            .biased_first_ = try allocator.alloc(Tensor, parameters.len),
            .biased_second_ = try allocator.alloc(Tensor, parameters.len),
        };
        @memcpy(self.parameters_, parameters);
        for (self.parameters_, 0..) |parameter, i| {
            self.biased_first_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));
            self.biased_second_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));

            try zt.tensor.eval(allocator, self.biased_first_[i]);
            try zt.tensor.eval(allocator, self.biased_second_[i]);
        }
        return self;
    }

    pub fn deinit(self: *AdamOptimizer) void {
        for (self.parameters_) |v| v.deinit();
        self.allocator.free(self.parameters_);
        for (self.biased_first_) |t| t.deinit();
        self.allocator.free(self.biased_first_);
        for (self.biased_second_) |t| t.deinit();
        self.allocator.free(self.biased_second_);
        self.allocator.destroy(self);
    }

    /// Zero the gradients for all the parameters being optimized. Typically
    /// called after every call to step().
    pub fn zeroGrad(self: *AdamOptimizer) void {
        for (self.parameters_) |parameter| {
            parameter.zeroGrad();
        }
    }

    /// Get the learning rate.
    pub fn getLr(self: *const AdamOptimizer) f64 {
        return self.lr_;
    }

    /// Set the learning rate.
    pub fn setLr(self: *AdamOptimizer, lr: f64) void {
        self.lr_ = lr;
    }

    pub fn step(self: *AdamOptimizer, allocator: std.mem.Allocator) !void {
        self.count_ += 1;
        const corrected_bias1 = 1 - std.math.pow(f32, self.beta1_, @floatFromInt(self.count_));
        const corrected_bias2 = 1 - std.math.pow(f32, self.beta2_, @floatFromInt(self.count_));
        const corrected_lr: f64 = self.lr_ * @as(f64, @floatCast(@sqrt(corrected_bias2) / corrected_bias1));

        for (0..self.parameters_.len) |i| {
            if (!self.parameters_[i].isGradAvailable()) {
                continue;
            }

            const grad = (try self.parameters_[i].grad()).tensor();
            const data = self.parameters_[i].tensor();

            if (self.wd_ != 0) {
                // Weight decay term
                var tmp = try zt.tensor.mul(allocator, f64, @as(f64, @floatCast(self.wd_)) * self.lr_, Tensor, data);
                defer tmp.deinit();
                try data.inPlaceSub(allocator, Tensor, tmp);
            }

            const biased_first = self.biased_first_[i];
            const biased_second = self.biased_second_[i];

            var tmp1 = try zt.tensor.mul(allocator, f32, 1 - self.beta1_, Tensor, grad);
            try biased_first.inPlaceMul(allocator, f32, self.beta1_);
            try biased_first.inPlaceAdd(allocator, Tensor, tmp1);
            tmp1.deinit();
            tmp1 = try zt.tensor.mul(allocator, f32, 1 - self.beta2_, Tensor, grad);
            try tmp1.inPlaceMul(allocator, Tensor, grad);
            try biased_second.inPlaceMul(allocator, f32, self.beta2_);
            try biased_second.inPlaceAdd(allocator, Tensor, tmp1);
            tmp1.deinit();

            try zt.tensor.eval(allocator, biased_first);
            try zt.tensor.eval(allocator, biased_second);

            tmp1 = try zt.tensor.sqrt(allocator, biased_second);
            try tmp1.inPlaceAdd(allocator, f32, self.eps_);
            var tmp2 = try zt.tensor.mul(allocator, f64, corrected_lr, Tensor, biased_first);
            defer tmp2.deinit();
            try tmp2.inPlaceDiv(allocator, Tensor, tmp1);
            tmp1.deinit();
            try data.inPlaceSub(allocator, Tensor, tmp2);

            try zt.tensor.eval(allocator, data);
        }
    }

    /// Generates a stringified representation of the optimizer.
    pub fn prettyString(self: *AdamOptimizer, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().write("Adam");
        if (self.wd_ != 0) {
            try buffer.writer().print(" (weight decay={d})", .{self.wd_});
        }
        return buffer.toOwnedSlice();
    }
};
