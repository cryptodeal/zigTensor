const std = @import("std");
const zigrc = @import("zigrc");
const zt = @import("../../zt.zig");
const OneDnnAutogradExtension = @import("backend/onednn/onednn_autograd_extension.zig").OneDnnAutogradExtension;

const PoolingMode = zt.common.PoolingMode;
const AutogradPayload = zt.autograd.AutogradPayload;
const AutogradExtension = zt.autograd.AutogradExtension;

const Tensor = zt.tensor.Tensor;

pub const RnnGradData = struct {
    dy: Tensor,
    dhy: Tensor,
    dcy: Tensor,
};

pub const op_details = struct {
    pub fn conv2d(
        allocator: std.mem.Allocator,
        input: Tensor,
        weights: Tensor,
        bias: Tensor,
        sx: i64,
        sy: i64,
        px: i64,
        py: i64,
        dx: i64,
        dy: i64,
        groups: i64,
        payload: ?zigrc.Arc(AutogradPayload),
    ) !Tensor {
        var ext = try (try input.backend(allocator)).getExtension(allocator, AutogradExtension);
        return ext.conv2d(allocator, input, weights, bias, sx, sy, px, py, dx, dy, groups, payload);
    }

    pub fn pool2d(
        allocator: std.mem.Allocator,
        input: Tensor,
        wx: i64,
        wy: i64,
        sx: i64,
        sy: i64,
        px: i64,
        py: i64,
        mode: PoolingMode,
        payload: ?zigrc.Arc(AutogradPayload),
    ) !Tensor {
        var ext = try (try input.backend(allocator)).getExtension(allocator, AutogradExtension);
        return ext.pool2d(allocator, input, wx, wy, sx, sy, px, py, mode, payload);
    }

    pub fn batchnorm(
        allocator: std.mem.Allocator,
        save_mean: Tensor,
        save_var: Tensor,
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        axes: []const i64,
        train: bool,
        momentum: f64,
        epsilon: f64,
        payload: ?zigrc.Arc(AutogradPayload),
    ) !Tensor {
        var ext = try (try input.backend(allocator)).getExtension(allocator, AutogradExtension);
        return ext.batchnorm(
            allocator,
            save_mean,
            save_var,
            input,
            weight,
            bias,
            running_mean,
            running_var,
            axes,
            train,
            momentum,
            epsilon,
            payload,
        );
    }
};
