const zt = @import("../../../../zt.zig");
const zigrc = @import("zigrc");
const dnnl = @import("../../../../bindings/onednn/onednn.zig");
const std = @import("std");
const dnnl_utils = @import("dnnl_utils.zig");

const AutogradPayload = zt.autograd.AutogradPayload;
const dnnlMapToType = dnnl_utils.dnnlMapToType;
const convertToDnnlDims = dnnl_utils.convertToDnnlDims;
const createArgs = dnnl_utils.createArgs;
const executeNetwork = dnnl_utils.executeNetwork;
const DnnlEngine = dnnl_utils.DnnlEngine;
const DnnlMemoryWrapper = dnnl_utils.DnnlMemoryWrapper;
const Shape = zt.tensor.shape.Shape;
const Tensor = zt.tensor.Tensor;

// zigTensor accept HWCN order according to docs
const k_h_idx: usize = 0;
const k_w_idx: usize = 1;
const k_channel_size_idx: usize = 2;
const k_batch_size_idx: usize = 3;

// Use `dnnl.dnnl_format_tag_any` for memory formatting even if pool
// inputs are shaped in a particular way.
const format_any: dnnl.dnnl_format_tag_t = dnnl.dnnl_format_tag_any;
const format_nchw: dnnl.dnnl_format_tag_t = dnnl.dnnl_nchw;
const format_x: dnnl.dnnl_format_tag_t = dnnl.dnnl_x;
const format_2d: dnnl.dnnl_format_tag_t = dnnl.dnnl_nc;

fn getNFeatures(input_shape: Shape, axes: []const i64) !i64 {
    var n_features: i64 = 1;
    for (axes) |ax| {
        n_features *= try zt.tensor.shape.dim(input_shape, @intCast(ax));
    }
    return n_features;
}

fn getInputOutputDims(allocator: std.mem.Allocator, min_axis: i64, max_axis: i64, input: Tensor, n_features: i64) ![]i64 {
    var in_desc_dims: [4]i64 = undefined;
    if (min_axis == 0) {
        in_desc_dims = [_]i64{
            1,
            1,
            @intCast(n_features),
            @divTrunc(try input.elements(allocator), n_features),
        };
    } else {
        var batch_sz: i64 = 1;
        for (@as(usize, @intCast(max_axis)) + 1..try input.ndim(allocator)) |i| {
            batch_sz *= try input.dim(allocator, i);
        }
        in_desc_dims = [_]i64{
            1,
            @divTrunc(try input.elements(allocator), (n_features * batch_sz)),
            @intCast(n_features),
            batch_sz,
        };
    }

    var input_output_dims = try allocator.alloc(i64, 4);
    input_output_dims[0] = in_desc_dims[k_batch_size_idx];
    input_output_dims[1] = in_desc_dims[k_channel_size_idx];
    input_output_dims[2] = in_desc_dims[k_h_idx];
    input_output_dims[3] = in_desc_dims[k_w_idx];
    return input_output_dims;
}

pub const OneDnnBatchNormPayload = struct {
    allocator: std.mem.Allocator,
    fwd_prim_desc: dnnl.dnnl_primitive_desc_t = null,
    weights: Tensor = undefined, // combined weight and bias
    bias: Tensor = undefined,
    weights_dims: []i64 = &[_]i64{},
    bias_dims: []i64 = &[_]i64{},
    output_memory_desc: dnnl.dnnl_memory_desc_t = null,
    mean_memory: dnnl.dnnl_memory_t = null,
    var_memory: dnnl.dnnl_memory_t = null,
    weights_memory: dnnl.dnnl_memory_t = null,
    bias_memory: dnnl.dnnl_memory_t = null,

    pub fn init(allocator: std.mem.Allocator) !*OneDnnBatchNormPayload {
        const self = try allocator.create(OneDnnBatchNormPayload);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *OneDnnBatchNormPayload) void {
        dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(self.fwd_prim_desc), @src()) catch unreachable;
        self.weights.deinit();
        self.bias.deinit();
        self.allocator.free(self.weights_dims);
        self.allocator.free(self.bias_dims);
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_destroy(self.output_memory_desc), @src()) catch unreachable;
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(self.mean_memory), @src()) catch unreachable;
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(self.var_memory), @src()) catch unreachable;
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(self.weights_memory), @src()) catch unreachable;
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(self.bias_memory), @src()) catch unreachable;
        self.allocator.destroy(self);
    }
};

pub inline fn batchnorm(
    allocator: std.mem.Allocator,
    _: Tensor,
    _: Tensor,
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    axes: []const i64,
    train: bool,
    momentum: f64,
    epsilon: f64,
    autograd_payload: ?zigrc.Arc(AutogradPayload),
) !Tensor {
    if (momentum != 0) {
        std.debug.print("OneDNN batchnorm op doesn't support momentum.\n", .{});
        return error.OneDNNBatchnormNoMomentum;
    }
    if (try input.dtype(allocator) == .f16) {
        std.debug.print("OneDNN batchnorm op - f16 inputs not supported.\n", .{});
        return error.OneDNNBatchnormNoF16;
    }
    var payload_data = try OneDnnBatchNormPayload.init(allocator);
    var payload = try zigrc.Arc(?*anyopaque).init(allocator, payload_data);
    const freeCtx = (struct {
        pub fn call(v: ?*anyopaque) void {
            var ctx: *OneDnnBatchNormPayload = @ptrCast(@alignCast(v));
            ctx.deinit();
        }
    }).call;
    errdefer payload.releaseWithFn(freeCtx);
    if (train and autograd_payload != null) {
        autograd_payload.?.value.data = payload;
    }

    const output = try Tensor.initHandle(allocator, try input.shape(allocator), try input.dtype(allocator));
    const n_features = try getNFeatures(try input.shape(allocator), axes);

    if (try running_var.isEmpty(allocator)) {
        var tmp = try zt.tensor.full(allocator, &.{n_features}, f64, 1, try input.dtype(allocator));
        defer tmp.deinit();
        try running_var.assign(allocator, Tensor, tmp);
    }

    if (try running_mean.isEmpty(allocator)) {
        var tmp = try zt.tensor.full(allocator, &.{n_features}, f64, 0, try input.dtype(allocator));
        defer tmp.deinit();
        try running_mean.assign(allocator, Tensor, tmp);
    }

    // Check if axes are valid
    const min_axis, const max_axis = std.mem.minMax(i64, axes);
    const axes_continuous = (axes.len == (max_axis - min_axis + 1));
    if (!axes_continuous) {
        std.debug.print("axis array should be continuous\n", .{});
        return error.AxisArrayNotContinuous;
    }

    _ = try dnnlMapToType(try input.dtype(allocator));
    const dnnl_engine = (try DnnlEngine.getInstance(allocator)).getEngine();

    // Prepare combined weights
    // If empty, user specifies affine to false. Both not trainable.
    const weight_non_empty = if (try weight.isEmpty(allocator)) try zt.tensor.full(allocator, &.{n_features}, f64, 1, .f32) else weight;
    const bias_non_empty = if (try bias.isEmpty(allocator)) try zt.tensor.full(allocator, &.{n_features}, f64, 0, .f32) else bias;

    // DNNL only accepts weight and bias as a combined input.
    // https://git.io/JLn9X
    payload_data.weights = weight_non_empty;
    payload_data.bias = bias_non_empty;
    payload_data.weights_dims = try convertToDnnlDims(allocator, &.{n_features});
    payload_data.bias_dims = try convertToDnnlDims(allocator, &.{n_features});
    const input_output_dims = try getInputOutputDims(allocator, min_axis, max_axis, input, n_features);
    defer allocator.free(input_output_dims);

    // Memory for forward
    var input_memory = try DnnlMemoryWrapper.init(allocator, input, input_output_dims, format_nchw);
    defer input_memory.deinit();
    var output_memory = try DnnlMemoryWrapper.init(allocator, output, input_output_dims, format_nchw);
    defer output_memory.deinit();
    var mean_memory = try DnnlMemoryWrapper.init(allocator, running_mean, &.{try running_mean.dim(allocator, 0)}, format_x);
    defer mean_memory.deinit();
    var var_memory = try DnnlMemoryWrapper.init(allocator, running_var, &.{try running_var.dim(allocator, 0)}, format_x);
    defer var_memory.deinit();
    // combined scale and shift (weight and bias)
    var weights_memory = try DnnlMemoryWrapper.init(allocator, payload_data.weights, payload_data.weights_dims, format_x);
    defer weights_memory.deinit();
    var bias_memory = try DnnlMemoryWrapper.init(allocator, payload_data.bias, payload_data.bias_dims, format_x);
    defer bias_memory.deinit();
    payload_data.mean_memory = mean_memory.getMemory();
    payload_data.var_memory = var_memory.getMemory();
    payload_data.weights_memory = weights_memory.getMemory();
    payload_data.bias_memory = bias_memory.getMemory();
    // Primitives and descriptors
    const kind: dnnl.dnnl_prop_kind_t = if (train) dnnl.dnnl_forward_training else dnnl.dnnl_forward_inference;
    // https://fburl.com/6latj733
    var flag: dnnl.dnnl_normalization_flags_t = if (train) dnnl.dnnl_normalization_flags_none else dnnl.dnnl_use_global_stats;
    flag = flag | dnnl.dnnl_use_scale | dnnl.dnnl_use_shift;
    try dnnl.DNNL_CHECK(
        dnnl.dnnl_batch_normalization_forward_primitive_desc_create(
            &payload_data.fwd_prim_desc,
            dnnl_engine,
            kind,
            input_memory.getDescriptor(),
            output_memory.getDescriptor(),
            @floatCast(epsilon),
            flag,
            null,
        ),
        @src(),
    );
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_clone(&payload_data.output_memory_desc, output_memory.getDescriptor()), @src());
    var bn: dnnl.dnnl_primitive_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&bn, payload_data.fwd_prim_desc), @src());
    var network = std.ArrayList(dnnl.dnnl_primitive_t).init(allocator);
    defer {
        for (network.items) |p| dnnl.DNNL_CHECK(dnnl.dnnl_primitive_destroy(p), @src()) catch unreachable;
        network.deinit();
    }
    try network.append(bn);
    var fwd_args = std.ArrayList([]dnnl.dnnl_exec_arg_t).init(allocator);
    defer {
        for (fwd_args.items) |a| allocator.free(a);
        fwd_args.deinit();
    }
    try fwd_args.append(try createArgs(
        allocator,
        &.{
            .{ dnnl.DNNL_ARG_SRC, input_memory.getMemory() },
            .{ dnnl.DNNL_ARG_MEAN, mean_memory.getMemory() },
            .{ dnnl.DNNL_ARG_VARIANCE, var_memory.getMemory() },
            .{ dnnl.DNNL_ARG_DST, output_memory.getMemory() },
            .{ dnnl.DNNL_ARG_SCALE, weights_memory.getMemory() },
            .{ dnnl.DNNL_ARG_SHIFT, bias_memory.getMemory() },
        },
        null,
    ));

    try executeNetwork(allocator, &network, &fwd_args);

    return output;
}

pub inline fn batchnormBackward(
    allocator: std.mem.Allocator,
    grad_output: Tensor,
    save_mean: Tensor,
    save_var: Tensor,
    input: Tensor,
    weight: Tensor,
    axes: []const i64,
    train: bool,
    epsilon: f64,
    autograd_payload: ?zigrc.Arc(AutogradPayload),
) !std.meta.Tuple(&.{ Tensor, Tensor, Tensor }) {
    _ = save_mean;
    _ = save_var;
    _ = weight;
    _ = train;
    if (autograd_payload == null) {
        std.debug.print("OneDnnAutogradExtension.{s} given null AutogradPayload\n", .{@src().fn_name});
        return error.NullAutogradPayload;
    }
    var payload: *OneDnnBatchNormPayload = @ptrCast(@alignCast(autograd_payload.?.value.data.value.*));
    _ = try dnnlMapToType(try input.dtype(allocator));

    const dnnl_engine = (try DnnlEngine.getInstance(allocator)).getEngine();
    const min_axis, const max_axis = std.mem.minMax(i64, axes);
    const axes_continuous = (axes.len == (max_axis - min_axis + 1));
    if (!axes_continuous) {
        std.debug.print("axis array should be continuous\n", .{});
        return error.AxisArrayNotContinuous;
    }

    const n_features = try getNFeatures(try input.shape(allocator), axes);
    const input_output_dims = try getInputOutputDims(allocator, min_axis, max_axis, input, n_features);
    defer allocator.free(input_output_dims);

    const grad_input = try Tensor.initHandle(allocator, try input.shape(allocator), try input.dtype(allocator));
    const grad_weights = try Tensor.initHandle(allocator, try payload.weights.shape(allocator), try payload.weights.dtype(allocator));
    const grad_bias = try Tensor.initHandle(allocator, try payload.bias.shape(allocator), try payload.bias.dtype(allocator));

    var input_memory = try DnnlMemoryWrapper.init(allocator, input, input_output_dims, format_nchw);
    defer input_memory.deinit();

    // Memory for gradient computation
    var grad_output_mem = try DnnlMemoryWrapper.init(allocator, grad_output, input_output_dims, format_nchw);
    defer grad_output_mem.deinit();
    var grad_input_mem = try DnnlMemoryWrapper.init(allocator, grad_input, input_output_dims, format_nchw);
    defer grad_input_mem.deinit();
    var grad_weights_mem = try DnnlMemoryWrapper.init(allocator, grad_weights, payload.weights_dims, format_x);
    defer grad_weights_mem.deinit();
    var grad_bias_mem = try DnnlMemoryWrapper.init(allocator, grad_bias, payload.bias_dims, format_x);
    defer grad_bias_mem.deinit();

    // Primitives and descriptors
    var bwd_prim_desc: dnnl.dnnl_primitive_desc_t = null;
    try dnnl.DNNL_CHECK(
        dnnl.dnnl_batch_normalization_backward_primitive_desc_create(
            &bwd_prim_desc,
            dnnl_engine,
            dnnl.dnnl_backward,
            grad_output_mem.getDescriptor(),
            payload.output_memory_desc,
            grad_output_mem.getDescriptor(),
            @floatCast(epsilon),
            dnnl.dnnl_use_scale | dnnl.dnnl_use_shift,
            payload.fwd_prim_desc,
            null,
        ),
        @src(),
    );
    var bwd_prim: dnnl.dnnl_primitive_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&bwd_prim, bwd_prim_desc), @src());
    defer dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(bwd_prim_desc), @src()) catch unreachable;

    // Execute
    var network_backwards = std.ArrayList(dnnl.dnnl_primitive_t).init(allocator);
    defer {
        for (network_backwards.items) |p| dnnl.DNNL_CHECK(dnnl.dnnl_primitive_destroy(p), @src()) catch unreachable;
        network_backwards.deinit();
    }
    try network_backwards.append(bwd_prim);
    var bwd_args = std.ArrayList([]dnnl.dnnl_exec_arg_t).init(allocator);
    defer {
        for (bwd_args.items) |a| allocator.free(a);
        bwd_args.deinit();
    }
    try bwd_args.append(try createArgs(allocator, &.{
        .{ dnnl.DNNL_ARG_SRC, input_memory.getMemory() },
        .{ dnnl.DNNL_ARG_MEAN, payload.mean_memory },
        .{ dnnl.DNNL_ARG_VARIANCE, payload.var_memory },
        .{ dnnl.DNNL_ARG_SCALE, payload.weights_memory },
        .{ dnnl.DNNL_ARG_SHIFT, payload.bias_memory },
        .{ dnnl.DNNL_ARG_DIFF_SRC, grad_input_mem.getMemory() },
        .{ dnnl.DNNL_ARG_DIFF_DST, grad_output_mem.getMemory() },
        .{ dnnl.DNNL_ARG_DIFF_SCALE, grad_weights_mem.getMemory() },
        .{ dnnl.DNNL_ARG_DIFF_SHIFT, grad_bias_mem.getMemory() },
    }, null));

    try executeNetwork(allocator, &network_backwards, &bwd_args);

    return .{ grad_input, grad_weights, grad_bias };
}
