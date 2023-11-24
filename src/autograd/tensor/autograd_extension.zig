const std = @import("std");
const zigrc = @import("zigrc");
const zt = @import("../../zt.zig");

const assert = std.debug.assert;
const Dim = zt.tensor.shape.Dim;
const DType = zt.tensor.DType;
const DynamicBenchmark = zt.common.DynamicBenchmark;
const TensorExtensionType = zt.tensor.TensorExtensionType;
const Tensor = zt.tensor.Tensor;
const PoolingMode = zt.common.PoolingMode;
const RnnMode = zt.common.RnnMode;
const RnnGradData = zt.autograd.RnnGradData;

pub const AutogradPayloadData = ?*anyopaque;

pub const AutogradPayload = struct { data: zigrc.Arc(AutogradPayloadData) };

pub const AutogradExtension = struct {
    const Self = @This();

    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
        isDataTypeSupported: *const fn (ctx: *anyopaque, dtype: DType) bool,
        getExtensionType: *const fn (ctx: *anyopaque) TensorExtensionType,
        //**************************** Forward ****************************//
        conv2d: *const fn (
            ctx: *anyopaque,
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
        ) anyerror!Tensor,
        pool2d: *const fn (
            ctx: *anyopaque,
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
        ) anyerror!Tensor,
        batchnorm: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            save_mean: Tensor,
            save_var: Tensor,
            input: Tensor,
            weight: Tensor,
            bias: Tensor,
            running_mean: Tensor,
            running_var: Tensor,
            axes: []const Dim,
            train: bool,
            momentum: f64,
            epsilon: f64,
            payload: ?zigrc.Arc(AutogradPayload),
        ) anyerror!Tensor,
        rnn: *const fn (
            ctx: *anyopaque,
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
            payload: ?zigrc.Arc(AutogradPayload),
        ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor, Tensor }),
        //**************************** Backward ****************************//
        // ]----- conv2d
        conv2dBackwardData: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            grad_output: Tensor,
            input: Tensor,
            weight: Tensor,
            sx: i64,
            sy: i64,
            px: i64,
            py: i64,
            dx: i64,
            dy: i64,
            groups: i64,
            data_grad_benchmark: ?zigrc.Arc(DynamicBenchmark),
            payload: ?zigrc.Arc(AutogradPayload),
        ) anyerror!Tensor,
        conv2dBackwardFilterBias: *const fn (
            ctx: *anyopaque,
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
            autograd_payload: ?zigrc.Arc(AutogradPayload),
        ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor }),
        // ]----- pool2D
        pool2dBackward: *const fn (
            ctx: *anyopaque,
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
        ) anyerror!Tensor,
        // ]----- batchnorm
        batchnormBackward: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            grad_output: Tensor,
            save_mean: Tensor,
            save_var: Tensor,
            input: Tensor,
            weight: Tensor,
            axes: []const Dim,
            train: bool,
            epsilon: f32,
            payload: ?zigrc.Arc(AutogradPayload),
        ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor, Tensor }),
        // ]----- rnn
        rnnBackward: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            input: Tensor,
            hidden_state: Tensor,
            cell_state: Tensor,
            weights: Tensor,
            grad_data: zigrc.Arc(RnnGradData),
            output: Tensor,
            num_layers: i64,
            hidden_size: i64,
            mode: RnnMode,
            bidirectional: bool,
            drop_prob: f32,
            payload: zigrc.Arc(AutogradPayload),
        ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor, Tensor, Tensor }),
    };

    /// Free all associated memory.
    pub fn deinit(self: *const Self) void {
        self.vtable.deinit(self.ptr);
    }

    /// Returns the type of the extension.
    pub fn getExtensionType(self: *const Self) TensorExtensionType {
        return self.vtable.getExtensionType(self.ptr);
    }

    pub fn extensionType() TensorExtensionType {
        return .Autograd;
    }

    pub fn isDataTypeSupported(self: *const Self, dtype: DType) bool {
        return self.vtable.isDataTypeSupported(self.ptr, dtype);
    }

    //**************************** Forward ****************************//

    pub fn conv2d(
        self: *const Self,
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
        return self.vtable.conv2d(
            self.ptr,
            allocator,
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
            payload,
        );
    }

    pub fn pool2d(
        self: *const Self,
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
        return self.vtable.pool2d(
            self.ptr,
            allocator,
            input,
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

    pub fn batchnorm(
        self: *const Self,
        allocator: std.mem.Allocator,
        save_mean: Tensor,
        save_var: Tensor,
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        axes: []const Dim,
        train: bool,
        momentum: f64,
        epsilon: f64,
        payload: ?zigrc.Arc(AutogradPayload),
    ) !Tensor {
        return self.vtable.batchnorm(
            self.ptr,
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
        self: *const Self,
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
        payload: zigrc.Arc(AutogradPayload),
    ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor, Tensor }) {
        return self.vtable.rnn(
            self.ptr,
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
            payload,
        );
    }

    //**************************** Backward ****************************//
    // ]----- conv2d
    pub fn conv2dBackwardData(
        self: *const Self,
        allocator: std.mem.Allocator,
        grad_output: Tensor,
        input: Tensor,
        weight: Tensor,
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
        return self.vtable.conv2dBackwardData(
            self.ptr,
            allocator,
            grad_output,
            input,
            weight,
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

    pub fn conv2dBackwardFilterBias(
        self: *const Self,
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
    ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor }) {
        return self.vtable.conv2dBackwardFilterBias(
            self.ptr,
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

    // ]----- pool2D
    pub fn pool2dBackward(
        self: *const Self,
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
        return self.vtable.pool2dBackward(
            self.ptr,
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

    // ]----- batchnorm
    pub fn batchnormBackward(
        self: *const Self,
        allocator: std.mem.Allocator,
        grad_output: Tensor,
        save_mean: Tensor,
        save_var: Tensor,
        input: Tensor,
        weight: Tensor,
        axes: []const Dim,
        train: bool,
        epsilon: f32,
        payload: ?zigrc.Arc(AutogradPayload),
    ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor, Tensor }) {
        return self.vtable.batchnormBackward(
            self.ptr,
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

    // ]----- rnn
    pub fn rnnBackward(
        self: *const Self,
        allocator: std.mem.Allocator,
        input: Tensor,
        hidden_state: Tensor,
        cell_state: Tensor,
        weights: Tensor,
        grad_data: zigrc.Arc(RnnGradData),
        output: Tensor,
        num_layers: i64,
        hidden_size: i64,
        mode: RnnMode,
        bidirectional: bool,
        drop_prob: f32,
        payload: zigrc.Arc(AutogradPayload),
    ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor, Tensor, Tensor }) {
        return self.vtable.rnnBackward(
            self.ptr,
            allocator,
            input,
            hidden_state,
            cell_state,
            weights,
            grad_data,
            output,
            num_layers,
            hidden_size,
            mode,
            bidirectional,
            drop_prob,
            payload,
        );
    }

    pub fn init(ext_impl: anytype) Self {
        const Ptr = @TypeOf(ext_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }

            fn isDataTypeSupported(ctx: *anyopaque, dtype: DType) bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isDataTypeSupported(dtype);
            }

            fn getExtensionType(ctx: *anyopaque) TensorExtensionType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getExtensionType();
            }

            //**************************** Forward ****************************//
            fn conv2d(
                ctx: *anyopaque,
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
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.conv2d(allocator, input, weights, bias, sx, sy, px, py, dx, dy, groups, payload);
            }

            fn pool2d(
                ctx: *anyopaque,
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
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.pool2d(
                    allocator,
                    input,
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

            fn batchnorm(
                ctx: *anyopaque,
                allocator: std.mem.Allocator,
                save_mean: Tensor,
                save_var: Tensor,
                input: Tensor,
                weight: Tensor,
                bias: Tensor,
                running_mean: Tensor,
                running_var: Tensor,
                axes: []const Dim,
                train: bool,
                momentum: f64,
                epsilon: f64,
                payload: ?zigrc.Arc(AutogradPayload),
            ) !Tensor {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.batchnorm(
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

            fn rnn(
                ctx: *anyopaque,
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
                payload: ?zigrc.Arc(AutogradPayload),
            ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor, Tensor }) {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.rnn(
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
                    payload,
                );
            }

            //**************************** Backward ****************************//
            // ]----- conv2d
            fn conv2dBackwardData(
                ctx: *anyopaque,
                allocator: std.mem.Allocator,
                grad_output: Tensor,
                input: Tensor,
                weight: Tensor,
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
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.conv2dBackwardData(
                    allocator,
                    grad_output,
                    input,
                    weight,
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

            fn conv2dBackwardFilterBias(
                ctx: *anyopaque,
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
                autograd_payload: ?zigrc.Arc(AutogradPayload),
            ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor }) {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.conv2dBackwardFilterBias(
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
                    autograd_payload,
                );
            }

            // ]----- pool2D
            fn pool2dBackward(
                ctx: *anyopaque,
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
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.pool2dBackward(
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

            // ]----- batchnorm
            fn batchnormBackward(
                ctx: *anyopaque,
                allocator: std.mem.Allocator,
                grad_output: Tensor,
                save_mean: Tensor,
                save_var: Tensor,
                input: Tensor,
                weight: Tensor,
                axes: []const Dim,
                train: bool,
                epsilon: f32,
                payload: ?zigrc.Arc(AutogradPayload),
            ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor, Tensor }) {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.batchnormBackward(
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

            // ]----- rnn
            fn rnnBackward(
                ctx: *anyopaque,
                allocator: std.mem.Allocator,
                input: Tensor,
                hidden_state: Tensor,
                cell_state: Tensor,
                weights: Tensor,
                grad_data: zigrc.Arc(RnnGradData),
                output: Tensor,
                num_layers: i64,
                hidden_size: i64,
                mode: RnnMode,
                bidirectional: bool,
                drop_prob: f32,
                payload: zigrc.Arc(AutogradPayload),
            ) anyerror!std.meta.Tuple(&.{ Tensor, Tensor, Tensor, Tensor }) {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.rnnBackward(
                    allocator,
                    input,
                    hidden_state,
                    cell_state,
                    weights,
                    grad_data,
                    output,
                    num_layers,
                    hidden_size,
                    mode,
                    bidirectional,
                    drop_prob,
                    payload,
                );
            }
        };
        return .{
            .ptr = ext_impl,
            .vtable = &.{
                .deinit = impl.deinit,
                .getExtensionType = impl.getExtensionType,
                .isDataTypeSupported = impl.isDataTypeSupported,
                .conv2d = impl.conv2d,
                .pool2d = impl.pool2d,
                .batchnorm = impl.batchnorm,
                .rnn = impl.rnn,
                .conv2dBackwardData = impl.conv2dBackwardData,
                .conv2dBackwardFilterBias = impl.conv2dBackwardFilterBias,
                .pool2dBackward = impl.pool2dBackward,
                .batchnormBackward = impl.batchnormBackward,
                .rnnBackward = impl.rnnBackward,
            },
        };
    }
};
