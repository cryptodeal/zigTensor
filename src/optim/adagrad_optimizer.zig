const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;
const Variable = zt.autograd.Variable;

/// An implementation of the Adagrad optimizer.
///
/// For more details see the paper
/// [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
pub const AdagradOptimizer = struct {
    allocator: std.mem.Allocator,
    parameters_: []*Variable,
    lr_: f64,
    eps_: f32,
    wd_: f32,
    variance_: []Tensor,

    pub fn init(allocator: std.mem.Allocator, parameters: []const *Variable, learning_rate: f64, epsilon: f32, weight_decay: f32) !*AdagradOptimizer {
        const self = try allocator.create(AdagradOptimizer);
        self.* = .{
            .allocator = allocator,
            .parameters_ = try allocator.alloc(*Variable, parameters.len),
            .lr_ = learning_rate,
            .eps_ = epsilon,
            .wd_ = weight_decay,
            .variance_ = try allocator.alloc(Tensor, parameters.len),
        };
        @memcpy(self.parameters_, parameters);
        for (self.parameters_, 0..) |parameter, i| {
            self.variance_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));
            try zt.tensor.eval(allocator, self.variance_[i]);
        }
        return self;
    }

    pub fn deinit(self: *AdagradOptimizer) void {
        for (self.parameters_) |v| v.deinit();
        self.allocator.free(self.parameters_);
        for (self.variance_) |t| t.deinit();
        self.allocator.free(self.variance_);
        self.allocator.destroy(self);
    }

    /// Zero the gradients for all the parameters being optimized. Typically
    /// called after every call to step().
    pub fn zeroGrad(self: *AdagradOptimizer) void {
        for (self.parameters_) |parameter| {
            parameter.zeroGrad();
        }
    }

    /// Get the learning rate.
    pub fn getLr(self: *const AdagradOptimizer) f64 {
        return self.lr_;
    }

    /// Set the learning rate.
    pub fn setLr(self: *AdagradOptimizer, lr: f64) void {
        self.lr_ = lr;
    }

    pub fn step(self: *AdagradOptimizer, allocator: std.mem.Allocator) !void {
        for (0..self.parameters_.len) |i| {
            if (!self.parameters_[i].isGradAvailable()) {
                continue;
            }

            const grad = (try self.parameters_[i].grad(allocator)).tensor();
            const data = self.parameters_[i].tensor();
            const variance = self.variance_[i];

            if (self.wd_ != 0) {
                // Weight decay term
                var tmp = try zt.tensor.mul(allocator, f32, self.wd_, Tensor, data);
                defer tmp.deinit();
                try data.inPlaceSub(allocator, Tensor, tmp);
            }

            var tmp1 = try zt.tensor.mul(allocator, Tensor, grad, Tensor, grad);
            try variance.inPlaceAdd(allocator, Tensor, tmp1);
            tmp1.deinit();
            try zt.tensor.eval(allocator, tmp1);

            tmp1 = try zt.tensor.sqrt(allocator, variance);
            try tmp1.inPlaceAdd(allocator, f32, self.eps_);
            const tmp2 = try zt.tensor.mul(allocator, f64, self.lr_, Tensor, grad);
            const tmp3 = try zt.tensor.div(allocator, Tensor, tmp2, Tensor, tmp1);
            defer tmp3.deinit();
            tmp1.deinit();
            tmp2.deinit();
            try data.inPlaceSub(allocator, Tensor, tmp3);
            try zt.tensor.eval(allocator, data);
        }
    }

    /// Generates a stringified representation of the optimizer.
    pub fn prettyString(self: *AdagradOptimizer, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().writeAll("Adagrad");
        if (self.eps_ != 0) {
            try buffer.writer().print(" (epsilon={d})", .{self.eps_});
        }
        return buffer.toOwnedSlice();
    }
};
