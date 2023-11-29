const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;
const Variable = zt.autograd.Variable;

/// Nesterov Accelerated Gradient with modification for the changeable lr
/// through time. Implements the version from
/// https://github.com/pytorch/fairseq/blob/e75cff5f2c1d62f12dc911e0bf420025eb1a4e33/fairseq/optim/nag.py#L43
pub const NAGOptimizer = struct {
    allocator: std.mem.Allocator,
    parameters_: []*Variable,
    lr_: f64,
    mu_: f32,
    wd_: f32,
    velocities_: []Tensor,
    old_lr_: f64,

    pub fn init(allocator: std.mem.Allocator, parameters: []const *Variable, learning_rate: f64, momentum: f32, weight_decay: f32) !*NAGOptimizer {
        if (momentum <= 0) {
            std.debug.print("Invalid momentum for NAG optimizer, it should be > 0", .{});
            return error.NAGOptimizerInvalidMomentum;
        }
        const self = try allocator.create(NAGOptimizer);
        self.* = .{
            .allocator = allocator,
            .parameters_ = try allocator.alloc(*Variable, parameters.len),
            .lr_ = learning_rate,
            .mu_ = momentum,
            .wd_ = weight_decay,
            .velocities_ = try allocator.alloc(Tensor, parameters.len),
            .old_lr_ = learning_rate,
        };
        @memcpy(self.parameters_, parameters);
        for (self.parameters_, 0..) |parameter, i| {
            self.velocities_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));
            try zt.tensor.eval(allocator, self.velocities_[i]);
        }
        return self;
    }

    pub fn deinit(self: *NAGOptimizer) void {
        for (self.parameters_) |v| v.deinit();
        self.allocator.free(self.parameters_);
        for (self.velocities_) |t| t.deinit();
        self.allocator.free(self.velocities_);
        self.allocator.destroy(self);
    }

    /// Zero the gradients for all the parameters being optimized. Typically
    /// called after every call to step().
    pub fn zeroGrad(self: *NAGOptimizer) void {
        for (self.parameters_) |parameter| {
            parameter.zeroGrad();
        }
    }

    /// Get the learning rate.
    pub fn getLr(self: *const NAGOptimizer) f64 {
        return self.lr_;
    }

    /// Set the learning rate.
    pub fn setLr(self: *NAGOptimizer, lr: f64) void {
        self.lr_ = lr;
    }

    pub fn step(self: *NAGOptimizer, allocator: std.mem.Allocator) !void {
        const corrected_lr = self.lr_ / self.old_lr_;

        for (0..self.parameters_.len) |i| {
            if (!self.parameters_[i].isGradAvailable()) {
                continue;
            }

            const grad = (try self.parameters_[i].grad()).tensor();
            const data = self.parameters_[i].tensor();

            if (self.wd_ != 0) {
                // Weight decay term
                try data.inPlaceMul(allocator, f64, 1 - self.lr_ * @as(f64, @floatCast(self.wd_)));
            }
            const velocity = self.velocities_[i];
            // this velocity corresponds to fairseq velocity * -1
            var tmp1 = try zt.tensor.mul(allocator, f64, self.lr_, Tensor, grad);
            try velocity.inPlaceMul(allocator, f32, self.mu_);
            try velocity.inPlaceMul(allocator, f64, corrected_lr);
            try velocity.inPlaceAdd(allocator, Tensor, tmp1);
            tmp1.deinit();
            try zt.tensor.eval(allocator, velocity);

            tmp1 = try zt.tensor.mul(allocator, f64, self.lr_, Tensor, velocity);
            try grad.inPlaceMul(allocator, f64, self.lr_);
            try grad.inPlaceAdd(allocator, Tensor, tmp1);
            tmp1.deinit();
            try data.inPlaceSub(allocator, Tensor, grad);
            try zt.tensor.eval(allocator, data);
        }
        self.old_lr_ = self.lr_;
    }

    pub fn prettyString(self: *NAGOptimizer, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().print("NAG (lr={d})); (previous lr={d});", .{ self.lr_, self.old_lr_ });
        if (self.wd_ != 0) {
            try buffer.writer().print(" (weight decay={d});", .{self.wd_});
        }
        try buffer.writer().print(" (Nesterov momentum={d})", .{self.mu_});
        return buffer.toOwnedSlice();
    }
};
