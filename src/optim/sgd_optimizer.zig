const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;
const Variable = zt.autograd.Variable;

/// A Stochastic Gradient Descent (SGD) optimizer.
///
/// Reference for SGD and Momentum: http://cs231n.github.io/neural-networks-3/#sgd
pub const SGDOptimizer = struct {
    allocator: std.mem.Allocator,
    parameters_: []*Variable,
    lr_: f64,
    use_nesterov_: bool,
    mu_: f32,
    wd_: f32,
    velocities_: []Tensor = &[_]Tensor{},

    pub fn init(
        allocator: std.mem.Allocator,
        parameters: []const *Variable,
        learning_rate: f64,
        momentum: f32,
        weight_decay: f32,
        use_nesterov: bool,
    ) !*SGDOptimizer {
        const self = try allocator.create(SGDOptimizer);
        self.* = .{
            .allocator = allocator,
            .parameters_ = try allocator.alloc(*Variable, parameters.len),
            .lr_ = learning_rate,
            .use_nesterov_ = use_nesterov,
            .mu_ = momentum,
            .wd_ = weight_decay,
        };
        @memcpy(self.parameters_, parameters);
        if (momentum != 0) {
            self.velocities_ = try allocator.alloc(Tensor, parameters.len);
            for (self.parameters_, 0..) |parameter, i| {
                self.velocities_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));
                try zt.tensor.eval(allocator, self.velocities_[i]);
            }
        }
        return self;
    }

    pub fn deinit(self: *SGDOptimizer) void {
        for (self.parameters_) |v| v.deinit();
        self.allocator.free(self.parameters_);
        for (self.velocities_) |t| t.deinit();
        self.allocator.free(self.velocities_);
        self.allocator.destroy(self);
    }

    /// Zero the gradients for all the parameters being optimized. Typically
    /// called after every call to step().
    pub fn zeroGrad(self: *SGDOptimizer) void {
        for (self.parameters_) |parameter| {
            parameter.zeroGrad();
        }
    }

    /// Get the learning rate.
    pub fn getLr(self: *const SGDOptimizer) f64 {
        return self.lr_;
    }

    /// Set the learning rate.
    pub fn setLr(self: *SGDOptimizer, lr: f64) void {
        self.lr_ = lr;
    }

    pub fn step(self: *SGDOptimizer, allocator: std.mem.Allocator) !void {
        for (0..self.parameters_.len) |i| {
            if (!self.parameters_[i].isGradAvailable()) {
                continue;
            }

            const grad = (try self.parameters_[i].grad()).tensor();
            const data = self.parameters_[i].tensor();

            if (self.wd_ != 0) {
                var tmp = try zt.tensor.mul(allocator, f32, self.wd_, Tensor, data);
                defer tmp.deinit();
                try grad.inPlaceAdd(allocator, Tensor, tmp);
            }

            if (self.mu_ != 0) {
                var velocity = self.velocities_[i];

                // Regular momentum
                try velocity.inPlaceMul(allocator, f32, self.mu_);
                try velocity.inPlaceAdd(allocator, Tensor, grad);
                try zt.tensor.eval(allocator, velocity);

                if (self.use_nesterov_) {
                    // Update for nesterov momentum
                    var tmp = try zt.tensor.mul(allocator, Tensor, velocity, f32, self.mu_);
                    defer tmp.deinit();
                    try grad.inPlaceAdd(allocator, Tensor, tmp);
                } else {
                    try grad.assign(allocator, Tensor, velocity);
                }
            }
            var tmp = try zt.tensor.mul(allocator, f64, self.lr_, Tensor, grad);
            defer tmp.deinit();
            try data.inPlaceSub(allocator, Tensor, tmp);
            try zt.tensor.eval(allocator, data);
        }
    }

    pub fn prettyString(self: *SGDOptimizer, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().writeAll("SGD");
        if (self.wd_ != 0) {
            try buffer.writer().print(" (weight decay={d})", .{self.wd_});
        }
        if (self.use_nesterov_ and self.mu_ != 0) {
            try buffer.writer().print("; (Nesterov momentum={d})", .{self.mu_});
        } else if (self.mu_ != 0) {
            try buffer.writer().print("; (momentum={d})", .{self.mu_});
        }
        return buffer.toOwnedSlice();
    }
};
