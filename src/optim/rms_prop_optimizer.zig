const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;
const Variable = zt.autograd.Variable;

/// An implementation of the RMSProp optimizer. For more details see Geoff
/// Hinton's [lecture slides](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
/// and https://arxiv.org/pdf/1308.0850v5.pdf.
pub const RMSPropOptimizer = struct {
    allocator: std.mem.Allocator,
    parameters_: []*Variable,
    lr_: f64,
    use_first_: bool,
    rho_: f32,
    eps_: f32,
    wd_: f32,
    first_: []Tensor = &[_]Tensor{},
    second_: []Tensor,

    pub fn init(
        allocator: std.mem.Allocator,
        parameters: []const *Variable,
        learning_rate: f64,
        rho: f32,
        epsilon: f32,
        weight_decay: f32,
        use_first: bool,
    ) !*RMSPropOptimizer {
        const self = try allocator.create(RMSPropOptimizer);
        self.* = .{
            .allocator = allocator,
            .parameters_ = try allocator.alloc(*Variable, parameters.len),
            .lr_ = learning_rate,
            .use_first_ = use_first,
            .rho_ = rho,
            .eps_ = epsilon,
            .wd_ = weight_decay,
            .second_ = try allocator.alloc(Tensor, parameters.len),
        };
        @memcpy(self.parameters_, parameters);
        if (self.use_first_) {
            self.first_ = try allocator.alloc(Tensor, parameters.len);
        }
        for (self.parameters_, 0..) |parameter, i| {
            if (self.use_first_) {
                self.first_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));
                try zt.tensor.eval(allocator, self.first_[i]);
            }

            self.second_[i] = try zt.tensor.full(allocator, try parameter.shape(allocator), f64, 0, try parameter.dtype(allocator));
            try zt.tensor.eval(allocator, self.second_[i]);
        }
        return self;
    }

    pub fn deinit(self: *RMSPropOptimizer) void {
        for (self.parameters_) |v| v.deinit();
        self.allocator.free(self.parameters_);
        for (self.first_) |t| t.deinit();
        self.allocator.free(self.first_);
        for (self.second_) |t| t.deinit();
        self.allocator.free(self.second_);
        self.allocator.destroy(self);
    }

    /// Zero the gradients for all the parameters being optimized. Typically
    /// called after every call to step().
    pub fn zeroGrad(self: *RMSPropOptimizer) void {
        for (self.parameters_) |parameter| {
            parameter.zeroGrad();
        }
    }

    /// Get the learning rate.
    pub fn getLr(self: *const RMSPropOptimizer) f64 {
        return self.lr_;
    }

    /// Set the learning rate.
    pub fn setLr(self: *RMSPropOptimizer, lr: f64) void {
        self.lr_ = lr;
    }

    pub fn step(self: *RMSPropOptimizer, allocator: std.mem.Allocator) !void {
        for (0..self.parameters_.len) |i| {
            if (!self.parameters_[i].isGradAvailable()) {
                continue;
            }

            const grad = (try self.parameters_[i].grad()).tensor();
            const data = self.parameters_[i].tensor();

            if (self.wd_ != 0) {
                // Weight decay term
                var tmp = try zt.tensor.mul(allocator, f32, self.wd_, Tensor, data);
                defer tmp.deinit();
                try data.inPlaceSub(allocator, Tensor, tmp);
            }

            const second = self.second_[i];
            var tmp1 = try zt.tensor.mul(allocator, Tensor, grad, Tensor, grad);
            try tmp1.inPlaceMul(allocator, f32, 1 - self.rho_);
            try second.inPlaceMul(allocator, f32, self.rho_);
            try second.inPlaceAdd(allocator, Tensor, tmp1);
            tmp1.deinit();
            try zt.tensor.eval(allocator, second);

            // Create shallow copy of second so that we don't update
            // "second" below
            var moments = try Tensor.initAssign(allocator, second);
            defer moments.deinit();
            if (self.use_first_) {
                var first = self.first_[i];
                tmp1 = try zt.tensor.mul(allocator, f32, 1 - self.rho_, Tensor, grad);
                try first.inPlaceMul(allocator, f32, self.rho_);
                try first.inPlaceAdd(allocator, Tensor, tmp1);
                tmp1.deinit();
                tmp1 = try zt.tensor.mul(allocator, Tensor, first, Tensor, first);
                try moments.inPlaceSub(allocator, Tensor, tmp1);
                tmp1.deinit();
                try zt.tensor.eval(allocator, first);
            }

            tmp1 = try zt.tensor.sqrt(allocator, moments);
            try tmp1.inPlaceAdd(allocator, f32, self.eps_);
            var tmp2 = try zt.tensor.mul(allocator, f64, self.lr_, Tensor, grad);
            try tmp2.inPlaceDiv(allocator, Tensor, tmp1);
            tmp1.deinit();
            try data.inPlaceSub(allocator, Tensor, tmp2);
            tmp2.deinit();

            try zt.tensor.eval(allocator, data);
        }
    }

    pub fn prettyString(self: *RMSPropOptimizer, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().writeAll("RMSProp");
        if (self.wd_ != 0) {
            try buffer.writer().print(" (weight decay={d})", .{self.wd_});
        }
        if (self.use_first_) {
            try buffer.writer().writeAll("; (use first moment)");
        }
        return buffer.toOwnedSlice();
    }
};
