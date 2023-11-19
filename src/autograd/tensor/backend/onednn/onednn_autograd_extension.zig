const std = @import("std");
const zt = @import("../../../../zt.zig");
const zigrc = @import("zigrc");
const dnnl = @import("../../../../bindings/onednn/onednn.zig");
const conv2d_impl = @import("conv2d.zig");
const pool2d_impl = @import("pool2d.zig");
const batchnorm_impl = @import("batchnorm.zig");
const rnn_impl = @import("rnn.zig");

const DynamicBenchmark = zt.common.DynamicBenchmark;
const RnnGradData = zt.autograd.RnnGradData;
const AutogradPayload = zt.autograd.AutogradPayload;
const PoolingMode = zt.common.PoolingMode;
const DType = zt.tensor.DType;
const RnnMode = zt.common.RnnMode;
const Tensor = zt.tensor.Tensor;
const TensorExtensionType = zt.tensor.TensorExtensionType;

pub const OneDnnAutogradExtension = struct {
    const Self = @This();
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*Self {
        var self = try allocator.create(Self);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn getExtensionType(_: *const Self) TensorExtensionType {
        return .Autograd;
    }

    pub fn isDataTypeSupported(_: *const Self, dtype: DType) bool {
        // fp16 computation is not supported with onednn
        return dtype != .f16;
    }

    // TODO: test to verify freeing memory allocated by onednn
    pub fn conv2d(
        _: *const Self,
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
        return conv2d_impl.conv2d(allocator, input, weights, bias, sx, sy, px, py, dx, dy, groups, payload);
    }

    pub fn pool2d(
        _: *const Self,
        allocator: std.mem.Allocator,
        input: Tensor,
        wx: i64,
        wy: i64,
        sx: i64,
        sy: i64,
        px: i64,
        py: i64,
        mode: PoolingMode,
        autograd_payload: ?zigrc.Arc(AutogradPayload),
    ) !Tensor {
        return pool2d_impl.pool2d(allocator, input, wx, wy, sx, sy, px, py, mode, autograd_payload);
    }

    pub fn batchnorm(
        _: *const Self,
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
        return batchnorm_impl.batchnorm(
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

    pub fn rnn(
        _: *const Self,
        allocator: std.mem.Allocator,
        input: Tensor,
        hidden_state: Tensor,
        cell_state: Tensor,
        weights: Tensor,
        hidden_size: i64,
        num_layers: i64,
        mode: RnnMode,
        bidirectional: bool,
        dropout: f32,
        autograd_payload: ?zigrc.Arc(AutogradPayload),
    ) !std.meta.Tuple(&.{ Tensor, Tensor, Tensor }) {
        return rnn_impl.rnn(
            allocator,
            input,
            hidden_state,
            cell_state,
            weights,
            hidden_size,
            num_layers,
            mode,
            bidirectional,
            dropout,
            autograd_payload,
        );
    }

    // TODO: test to verify freeing memory allocated by onednn
    pub fn conv2dBackwardData(
        _: *const Self,
        allocator: std.mem.Allocator,
        grad_output: Tensor,
        input: Tensor,
        weights: Tensor,
        sx: i64,
        sy: i64,
        px: i64,
        py: i64,
        dx: i64,
        dy: i64,
        groups: i64,
        data_grad_benchmark: ?zigrc.Arc(DynamicBenchmark),
        payload: ?zigrc.Arc(AutogradPayload),
    ) !Tensor {
        return conv2d_impl.conv2dBackwardData(
            allocator,
            grad_output,
            input,
            weights,
            sx,
            sy,
            px,
            py,
            dx,
            dy,
            groups,
            data_grad_benchmark,
            payload,
        );
    }

    // TODO: test to verify freeing memory allocated by onednn
    pub fn conv2dBackwardFilterBias(
        _: *const Self,
        allocator: std.mem.Allocator,
        grad_output: Tensor,
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
        filter_bench: ?zigrc.Arc(DynamicBenchmark),
        bias_bench: ?zigrc.Arc(DynamicBenchmark),
        payload: ?zigrc.Arc(AutogradPayload),
    ) !std.meta.Tuple(&.{ Tensor, Tensor }) {
        return conv2d_impl.conv2dBackwardFilterBias(
            allocator,
            grad_output,
            input,
            weights,
            bias,
            sx,
            sy,
            px,
            py,
            dx,
            dy,
            groups,
            filter_bench,
            bias_bench,
            payload,
        );
    }

    pub fn pool2dBackward(
        _: *const Self,
        allocator: std.mem.Allocator,
        grad_output: Tensor,
        input: Tensor,
        pool_output: Tensor,
        wx: i64,
        wy: i64,
        sx: i64,
        sy: i64,
        px: i64,
        py: i64,
        mode: PoolingMode,
        payload: ?zigrc.Arc(AutogradPayload),
    ) !Tensor {
        return pool2d_impl.pool2dBackward(
            allocator,
            grad_output,
            input,
            pool_output,
            wx,
            wy,
            sx,
            sy,
            px,
            py,
            mode,
            payload,
        );
    }

    pub fn batchnormBackward(
        _: *const Self,
        allocator: std.mem.Allocator,
        grad_output: Tensor,
        save_mean: Tensor,
        save_var: Tensor,
        input: Tensor,
        weight: Tensor,
        axes: []const i64,
        train: bool,
        epsilon: f32,
        payload: ?zigrc.Arc(AutogradPayload),
    ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor, Tensor }) {
        return batchnorm_impl.batchnormBackward(
            allocator,
            grad_output,
            save_mean,
            save_var,
            input,
            weight,
            axes,
            train,
            epsilon,
            payload,
        );
    }

    pub fn rnnBackward(
        _: *const Self,
        _: std.mem.Allocator,
        _: Tensor,
        _: Tensor,
        _: Tensor,
        _: Tensor,
        _: zigrc.Arc(RnnGradData),
        _: Tensor,
        _: i64,
        _: i64,
        _: RnnMode,
        _: bool,
        _: f32,
        _: zigrc.Arc(AutogradPayload),
    ) !std.meta.Tuple(&.{ Tensor, Tensor, Tensor, Tensor }) {
        std.debug.print("onednn RNN: Gradient computation not yet supported\n", .{});
        return error.OneDNNDoesNotSupportRNNGradientComputation;
    }
};
