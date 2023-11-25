const std = @import("std");
const zt = @import("../../../../zt.zig");
const dnnl = @import("../../../../bindings/onednn/onednn.zig");
const dnnl_utils = @import("dnnl_utils.zig");
const zigrc = @import("zigrc");

const DynamicBenchmark = zt.common.DynamicBenchmark;
const AutogradPayload = zt.autograd.AutogradPayload;
const dnnlMapToType = dnnl_utils.dnnlMapToType;
const convertToDnnlDims = dnnl_utils.convertToDnnlDims;
const DnnlEngine = dnnl_utils.DnnlEngine;
const DnnlMemoryWrapper = dnnl_utils.DnnlMemoryWrapper;
const dnnlAlignOrdering = dnnl_utils.dnnlAlignOrdering;
const createArgs = dnnl_utils.createArgs;
const executeNetwork = dnnl_utils.executeNetwork;
const Dim = zt.tensor.shape.Dim;
const Shape = zt.tensor.shape.Shape;
const Tensor = zt.tensor.Tensor;
const DType = zt.tensor.DType;

// Input, output: WHCN; weights: WHIO
pub const k_w_idx: usize = 0;
pub const k_h_idx: usize = 1;
pub const k_i_o_channel_size_idx: usize = 2;
pub const k_i_o_batch_size_idx: usize = 3;
pub const k_weight_output_channel_size_idx: usize = 3;

pub const format_any = dnnl.dnnl_format_tag_any;
pub const format_nchw = dnnl.dnnl_nchw;
pub const format_bias = dnnl.dnnl_x;

pub const OneDnnConv2DData = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    input_dims: []Dim = &[_]Dim{},
    weight_dims: []Dim = &[_]Dim{},
    output_dims: []Dim = &[_]Dim{},
    bias_dims: []Dim = &[_]Dim{},
    stride_dims: []Dim = &[_]Dim{},
    dilation_dims: []Dim = &[_]Dim{},
    padding_dims: []Dim = &[_]Dim{},
    // Memory descriptors
    input_mem_desc: dnnl.dnnl_memory_desc_t = null,
    output_mem_desc: dnnl.dnnl_memory_desc_t = null,
    weight_mem_desc: dnnl.dnnl_memory_desc_t = null,
    bias_mem_desc: dnnl.dnnl_memory_desc_t = null,
    // used for creating a backward desc
    fwd_prim_desc: dnnl.dnnl_primitive_desc_t = null,

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.input_dims);
        self.allocator.free(self.weight_dims);
        self.allocator.free(self.output_dims);
        self.allocator.free(self.bias_dims);
        self.allocator.free(self.stride_dims);
        self.allocator.free(self.dilation_dims);
        self.allocator.free(self.padding_dims);
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_destroy(self.input_mem_desc), @src()) catch unreachable;
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_destroy(self.output_mem_desc), @src()) catch unreachable;
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_destroy(self.weight_mem_desc), @src()) catch unreachable;
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_destroy(self.bias_mem_desc), @src()) catch unreachable;
        dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(self.fwd_prim_desc), @src()) catch unreachable;
    }

    pub fn init(
        allocator: std.mem.Allocator,
        input_type: DType,
        input_shape: Shape,
        weights_shape: Shape,
        bias_shape: Shape,
        output_shape: Shape,
        sx: i64,
        sy: i64,
        px: i64,
        py: i64,
        dx: i64,
        dy: i64,
        groups: i64,
    ) !Self {
        const data_type: dnnl.dnnl_data_type_t = try dnnlMapToType(input_type);
        const format_weight: dnnl.dnnl_format_tag_t = if (groups == 1) dnnl.dnnl_oihw else dnnl.dnnl_goihw;
        const has_bias = zt.tensor.shape.elements(bias_shape) > 0;

        var out: Self = .{ .allocator = allocator };
        // Create memory dims
        out.input_dims = try convertToDnnlDims(allocator, &.{
            try zt.tensor.shape.dim(input_shape, k_i_o_batch_size_idx),
            try zt.tensor.shape.dim(input_shape, k_i_o_channel_size_idx),
            try zt.tensor.shape.dim(input_shape, k_h_idx),
            try zt.tensor.shape.dim(input_shape, k_w_idx),
        });
        if (groups == 1) {
            out.weight_dims = try convertToDnnlDims(allocator, &.{
                try zt.tensor.shape.dim(weights_shape, k_weight_output_channel_size_idx),
                try zt.tensor.shape.dim(input_shape, k_i_o_channel_size_idx),
                try zt.tensor.shape.dim(weights_shape, k_h_idx),
                try zt.tensor.shape.dim(weights_shape, k_w_idx),
            });
        } else {
            out.weight_dims = try convertToDnnlDims(allocator, &.{
                groups,
                @divTrunc(try zt.tensor.shape.dim(weights_shape, k_weight_output_channel_size_idx), groups),
                @divTrunc(try zt.tensor.shape.dim(input_shape, k_i_o_channel_size_idx), groups),
                try zt.tensor.shape.dim(weights_shape, k_h_idx),
                try zt.tensor.shape.dim(weights_shape, k_w_idx),
            });
        }
        out.output_dims = try convertToDnnlDims(allocator, &.{
            try zt.tensor.shape.dim(input_shape, k_i_o_batch_size_idx),
            try zt.tensor.shape.dim(weights_shape, k_weight_output_channel_size_idx),
            try zt.tensor.shape.dim(output_shape, k_h_idx),
            try zt.tensor.shape.dim(output_shape, k_w_idx),
        });
        out.bias_dims = try convertToDnnlDims(allocator, &.{try zt.tensor.shape.dim(weights_shape, k_weight_output_channel_size_idx)});
        out.stride_dims = try convertToDnnlDims(allocator, &.{ sy, sx });
        out.padding_dims = try convertToDnnlDims(allocator, &.{ py, px });
        // NB: DNNL treats a dilation of 0 as a standard convolution and indexes
        // larger dilations accordingly. See https://git.io/fhAT2 for more.
        out.dilation_dims = try convertToDnnlDims(allocator, &.{ dy - 1, dx - 1 });
        // Create memory descriptors. using `dnnl.dnnl_format_tag_any` gives the best performance
        try dnnl.DNNL_CHECK(
            dnnl.dnnl_memory_desc_create_with_tag(
                &out.input_mem_desc,
                @intCast(out.input_dims.len),
                out.input_dims.ptr,
                data_type,
                format_any,
            ),
            @src(),
        );
        try dnnl.DNNL_CHECK(
            dnnl.dnnl_memory_desc_create_with_tag(
                &out.output_mem_desc,
                @intCast(out.output_dims.len),
                out.output_dims.ptr,
                data_type,
                format_any,
            ),
            @src(),
        );
        try dnnl.DNNL_CHECK(
            dnnl.dnnl_memory_desc_create_with_tag(
                &out.weight_mem_desc,
                @intCast(out.weight_dims.len),
                out.weight_dims.ptr,
                data_type,
                format_weight,
            ),
            @src(),
        );
        try dnnl.DNNL_CHECK(
            dnnl.dnnl_memory_desc_create_with_tag(
                &out.bias_mem_desc,
                @intCast(out.bias_dims.len),
                out.bias_dims.ptr,
                data_type,
                format_any,
            ),
            @src(),
        );

        const forward_mode: dnnl.dnnl_prop_kind_t = dnnl.dnnl_forward_training;
        // TODO: determine train mode/assess perf impact of always choosing training
        // (primitive cache storage overhead?)
        // const forward_mode: dnnl.dnnl_prop_kind_t = if (train) dnnl.dnnl_forward_training else dnnl.dnnl_forward_inference;
        try dnnl.DNNL_CHECK(
            dnnl.dnnl_convolution_forward_primitive_desc_create(
                &out.fwd_prim_desc,
                (try DnnlEngine.getInstance(allocator)).getEngine(),
                forward_mode,
                dnnl.dnnl_convolution_direct,
                out.input_mem_desc,
                out.weight_mem_desc,
                if (has_bias) out.bias_mem_desc else null,
                out.output_mem_desc,
                out.stride_dims.ptr,
                out.dilation_dims.ptr,
                out.padding_dims.ptr,
                out.padding_dims.ptr,
                null,
            ),
            @src(),
        );

        return out;
    }
};

// TODO: test to verify freeing memory allocated by onednn
pub inline fn conv2d(
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
    _: ?zigrc.Arc(AutogradPayload),
) !Tensor {
    if (try input.dtype(allocator) == .f16) {
        std.debug.print("Half precision is not supported in CPU.\n", .{});
        return error.AutogradExtensionUnsupportedType;
    }

    // zigTensor input, weight, and output shapes in column-major:
    // - Input is WHCN
    // - Weights are WHIO
    // - Output is WHCN
    // Since ArrayFire is column major, getting a raw pointer (1D
    // representation) of these shapes and viewing as if the representation is
    // row major transposes along all axis into NCHW for the input and output
    // and OIHW for the weights
    var output = try Tensor.initHandle(
        allocator,
        &.{
            1 + @divTrunc((try input.dim(allocator, k_w_idx) + (2 * px) - (1 + (try weights.dim(allocator, k_w_idx) - 1) * dx)), sx),
            1 + @divTrunc((try input.dim(allocator, k_h_idx) + (2 * py) - (1 + (try weights.dim(allocator, k_h_idx) - 1) * dy)), sy),
            try weights.dim(allocator, k_weight_output_channel_size_idx),
            try input.dim(allocator, k_i_o_batch_size_idx),
        },
        try input.dtype(allocator),
    );
    const has_bias = try bias.elements(allocator) > 0;
    _ = try dnnlMapToType(try input.dtype(allocator));
    const format_weight: dnnl.dnnl_format_tag_t = if (groups == 1) dnnl.dnnl_oihw else dnnl.dnnl_goihw;
    const dnnl_engine = (try DnnlEngine.getInstance(allocator)).getEngine();

    //********************************* Forward *******************************//
    var conv2d_data = try OneDnnConv2DData.init(
        allocator,
        try input.dtype(allocator),
        try input.shape(allocator),
        try weights.shape(allocator),
        try bias.shape(allocator),
        try output.shape(allocator),
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        groups,
    );
    defer conv2d_data.deinit();

    // Create memory
    var input_mem_init = try DnnlMemoryWrapper.init(allocator, input, conv2d_data.input_dims, format_nchw);
    defer input_mem_init.deinitAll();
    var output_mem_init = try DnnlMemoryWrapper.init(allocator, output, conv2d_data.output_dims, format_nchw);
    defer output_mem_init.deinitAll();
    var weights_mem = try DnnlMemoryWrapper.init(allocator, weights, conv2d_data.weight_dims, format_weight);
    defer weights_mem.deinitAll();

    // Network for execution
    var network = std.ArrayList(dnnl.dnnl_primitive_t).init(allocator);
    defer {
        for (network.items) |p| dnnl.DNNL_CHECK(dnnl.dnnl_primitive_destroy(p), @src()) catch unreachable;
        network.deinit();
    }
    var fwd_args = std.ArrayList([]dnnl.dnnl_exec_arg_t).init(allocator);
    defer {
        for (fwd_args.items) |a| allocator.free(a);
        fwd_args.deinit();
    }

    // DNNL suggests checking if the layout requested for the convolution
    // is different from NCHW/OIHW (even if specified), and reordering if
    // necessary, since the convolution itself may request a different
    // ordering
    const input_desc: dnnl.const_dnnl_memory_desc_t = dnnl.dnnl_primitive_desc_query_md(conv2d_data.fwd_prim_desc, dnnl.dnnl_query_src_md, 0);
    const weights_desc: dnnl.const_dnnl_memory_desc_t = dnnl.dnnl_primitive_desc_query_md(conv2d_data.fwd_prim_desc, dnnl.dnnl_query_weights_md, 0);
    const output_desc: dnnl.const_dnnl_memory_desc_t = dnnl.dnnl_primitive_desc_query_md(conv2d_data.fwd_prim_desc, dnnl.dnnl_query_dst_md, 0);
    // Input
    const in_mem_allocated, const input_memory = try dnnlAlignOrdering(allocator, &network, &fwd_args, input_mem_init.getMemory(), input_desc);
    defer if (in_mem_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(input_memory), @src()) catch unreachable;
    const weights_mem_allocated, const weights_memory = try dnnlAlignOrdering(allocator, &network, &fwd_args, weights_mem.getMemory(), weights_desc);
    defer if (weights_mem_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(weights_memory), @src()) catch unreachable;
    // Output - adds a reorder after the conv if needed
    var output_memory_allocated = false;
    var output_memory = output_mem_init.getMemory();
    defer if (output_memory_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(output_memory), @src()) catch unreachable;
    var out_desc: dnnl.const_dnnl_memory_desc_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_get_memory_desc(output_memory, &out_desc), @src());
    if (dnnl.dnnl_memory_desc_equal(out_desc, output_desc) == 0) {
        try dnnl.DNNL_CHECK(dnnl.dnnl_memory_create(&output_memory, output_desc, dnnl_engine, dnnl.DNNL_MEMORY_ALLOCATE), @src());
        output_memory_allocated = true;
    }

    // Create convolution
    var conv: dnnl.dnnl_primitive_t = null;
    var bias_memory = try DnnlMemoryWrapper.init(allocator, bias, conv2d_data.bias_dims, format_bias);
    defer bias_memory.deinit();
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&conv, conv2d_data.fwd_prim_desc), @src());

    try network.append(conv);

    // Conv fwd args
    var conv_fwd_args = try createArgs(
        allocator,
        &.{
            .{ dnnl.DNNL_ARG_SRC, input_memory },
            .{ dnnl.DNNL_ARG_WEIGHTS, weights_memory },
            .{ dnnl.DNNL_ARG_DST, output_memory },
        },
        if (has_bias) 4 else null,
    );
    if (has_bias) {
        conv_fwd_args[3] = .{ .arg = dnnl.DNNL_ARG_BIAS, .memory = bias_memory.getMemory() };
    }
    try fwd_args.append(conv_fwd_args);

    // Add output reordering if needed
    if (output_memory != output_mem_init.getMemory()) {
        var reorder_pd: dnnl.dnnl_primitive_desc_t = null;
        var src_desc: dnnl.const_dnnl_memory_desc_t = null;
        try dnnl.DNNL_CHECK(dnnl.dnnl_memory_get_memory_desc(output_memory, &src_desc), @src());
        try dnnl.DNNL_CHECK(dnnl.dnnl_reorder_primitive_desc_create(&reorder_pd, src_desc, dnnl_engine, output_mem_init.getDescriptor(), dnnl_engine, null), @src());
        var reorder: dnnl.dnnl_primitive_t = null;
        try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&reorder, reorder_pd), @src());
        try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(reorder_pd), @src());
        try network.append(reorder);
        try fwd_args.append(try createArgs(allocator, &.{ .{ dnnl.DNNL_ARG_FROM, output_memory }, .{ dnnl.DNNL_ARG_TO, output_mem_init.getMemory() } }, null));
    }

    // Run
    try executeNetwork(allocator, &network, &fwd_args);

    return output;
}

// TODO: test to verify freeing memory allocated by onednn
pub inline fn conv2dBackwardData(
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
    _: ?zigrc.Arc(DynamicBenchmark),
    _: ?zigrc.Arc(AutogradPayload),
) !Tensor {
    const grad_input = try Tensor.initHandle(allocator, try input.shape(allocator), try input.dtype(allocator)); // result

    _ = try dnnlMapToType(try input.dtype(allocator));
    const format_weight: dnnl.dnnl_format_tag_t = if (groups == 1) dnnl.dnnl_oihw else dnnl.dnnl_goihw;
    const dnnl_engine_bwd = (try DnnlEngine.getInstance(allocator)).getEngine();

    var bias = try Tensor.initEmpty(allocator); // dummy
    defer bias.deinit();
    var conv2d_data = try OneDnnConv2DData.init(
        allocator,
        try input.dtype(allocator),
        try input.shape(allocator),
        try weights.shape(allocator),
        try bias.shape(allocator),
        try grad_output.shape(allocator), // has the same shape as the Conv output
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        groups,
    );
    defer conv2d_data.deinit();

    // Backward descriptor
    var bwd_data_primitive_desc: dnnl.dnnl_primitive_desc_t = null;
    try dnnl.DNNL_CHECK(
        dnnl.dnnl_convolution_backward_data_primitive_desc_create(
            &bwd_data_primitive_desc,
            dnnl_engine_bwd,
            dnnl.dnnl_convolution_direct,
            conv2d_data.input_mem_desc,
            conv2d_data.weight_mem_desc,
            conv2d_data.output_mem_desc,
            conv2d_data.stride_dims.ptr,
            conv2d_data.dilation_dims.ptr,
            conv2d_data.padding_dims.ptr,
            conv2d_data.padding_dims.ptr,
            conv2d_data.fwd_prim_desc,
            null,
        ),
        @src(),
    );
    defer dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(bwd_data_primitive_desc), @src()) catch unreachable;

    // Create memory
    var grad_output_mem_init = try DnnlMemoryWrapper.init(allocator, grad_output, conv2d_data.output_dims, format_nchw);
    defer grad_output_mem_init.deinitAll();
    var grad_input_mem_init = try DnnlMemoryWrapper.init(allocator, grad_input, conv2d_data.input_dims, format_nchw);
    defer grad_input_mem_init.deinitAll();
    var weights_mem_init_bwd = try DnnlMemoryWrapper.init(allocator, weights, conv2d_data.weight_dims, format_weight);
    defer weights_mem_init_bwd.deinitAll();

    var network_backwards = std.ArrayList(dnnl.dnnl_primitive_t).init(allocator);
    defer {
        for (network_backwards.items) |p| dnnl.DNNL_CHECK(dnnl.dnnl_primitive_destroy(p), @src()) catch unreachable;
        network_backwards.deinit();
    }
    var bwd_data_args = std.ArrayList([]dnnl.dnnl_exec_arg_t).init(allocator);
    defer {
        for (bwd_data_args.items) |a| allocator.free(a);
        bwd_data_args.deinit();
    }

    // Check for reorderings
    const grad_output_desc: dnnl.const_dnnl_memory_desc_t = dnnl.dnnl_primitive_desc_query_md(bwd_data_primitive_desc, dnnl.dnnl_query_diff_dst_md, 0);
    const weights_desc: dnnl.const_dnnl_memory_desc_t = dnnl.dnnl_primitive_desc_query_md(bwd_data_primitive_desc, dnnl.dnnl_query_weights_md, 0);
    const grad_input_desc: dnnl.const_dnnl_memory_desc_t = dnnl.dnnl_primitive_desc_query_md(bwd_data_primitive_desc, dnnl.dnnl_query_diff_src_md, 0);
    const grad_out_allocated, const grad_output_memory = try dnnlAlignOrdering(allocator, &network_backwards, &bwd_data_args, grad_output_mem_init.getMemory(), grad_output_desc);
    defer if (grad_out_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(grad_output_memory), @src()) catch unreachable;
    const weights_mem_allocated, const weights_memory_backward = try dnnlAlignOrdering(allocator, &network_backwards, &bwd_data_args, weights_mem_init_bwd.getMemory(), weights_desc);
    defer if (weights_mem_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(weights_memory_backward), @src()) catch unreachable;
    var grad_input_memory_allocated = false;
    var grad_input_memory = grad_input_mem_init.getMemory();
    defer if (grad_input_memory_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(grad_input_memory), @src()) catch unreachable;
    // Don't reorder the gradient until after the conv
    if (dnnl.dnnl_memory_desc_equal(grad_input_mem_init.getDescriptor(), grad_input_desc) == 0) {
        try dnnl.DNNL_CHECK(dnnl.dnnl_memory_create(&grad_input_memory, grad_input_desc, dnnl_engine_bwd, dnnl.DNNL_MEMORY_ALLOCATE), @src());
        grad_input_memory_allocated = true;
    }

    // Convolution backwards primitive
    var conv_bwd_data: dnnl.dnnl_primitive_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&conv_bwd_data, bwd_data_primitive_desc), @src());
    try bwd_data_args.append(try createArgs(allocator, &.{
        .{ dnnl.DNNL_ARG_DIFF_SRC, grad_input_memory },
        .{ dnnl.DNNL_ARG_WEIGHTS, weights_memory_backward },
        .{ dnnl.DNNL_ARG_DIFF_DST, grad_output_memory },
    }, null));
    try network_backwards.append(conv_bwd_data);

    // Reorder the output (which is grad_input here) if necessary
    if (grad_input_memory != grad_input_mem_init.getMemory()) {
        var reorder_pd: dnnl.dnnl_primitive_desc_t = null;
        var tmp_input_desc: dnnl.dnnl_memory_desc_t = null;
        try dnnl.DNNL_CHECK(dnnl.dnnl_memory_get_memory_desc(grad_input_memory, &tmp_input_desc), @src());
        try dnnl.DNNL_CHECK(dnnl.dnnl_reorder_primitive_desc_create(&reorder_pd, tmp_input_desc, dnnl_engine_bwd, grad_input_mem_init.getDescriptor(), dnnl_engine_bwd, null), @src());
        var reorder: dnnl.dnnl_primitive_t = null;
        try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&reorder, reorder_pd), @src());
        try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(reorder_pd), @src());
        try network_backwards.append(reorder);
        try bwd_data_args.append(try createArgs(allocator, &.{
            .{ dnnl.DNNL_ARG_FROM, grad_input_memory },
            .{ dnnl.DNNL_ARG_TO, grad_input_mem_init.getMemory() },
        }, null));
    }

    // Run
    try executeNetwork(allocator, &network_backwards, &bwd_data_args);

    return grad_input;
}

// TODO: test to verify freeing memory allocated by onednn
pub inline fn conv2dBackwardFilterBias(
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
    _: ?zigrc.Arc(DynamicBenchmark),
    _: ?zigrc.Arc(DynamicBenchmark),
    _: ?zigrc.Arc(AutogradPayload),
) !std.meta.Tuple(&.{ Tensor, Tensor }) {
    const grad_weights = try Tensor.initHandle(allocator, try weights.shape(allocator), try weights.dtype(allocator));

    _ = try dnnlMapToType(try input.dtype(allocator));
    const format_weight: dnnl.dnnl_format_tag_t = if (groups == 1) dnnl.dnnl_oihw else dnnl.dnnl_goihw;
    const dnnl_engine_bwd = (try DnnlEngine.getInstance(allocator)).getEngine();

    var conv2d_data = try OneDnnConv2DData.init(
        allocator,
        try input.dtype(allocator),
        try input.shape(allocator),
        try weights.shape(allocator),
        try bias.shape(allocator),
        try grad_output.shape(allocator), // has the same shape as the Conv output
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        groups,
    );
    defer conv2d_data.deinit();

    var bias_ndims: dnnl.dnnl_dim_t = undefined;
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_query(conv2d_data.bias_mem_desc, dnnl.dnnl_query_ndims_s32, &bias_ndims), @src());
    const compute_bias_grad = !try bias.isEmpty(allocator) and bias_ndims != 0;
    const grad_bias = if (compute_bias_grad) try Tensor.initHandle(allocator, try bias.shape(allocator), try bias.dtype(allocator)) else try Tensor.initEmpty(allocator);

    // Weight backward primitive descriptor
    var bwd_weight_primitive_desc: dnnl.dnnl_primitive_desc_t = null;
    try dnnl.DNNL_CHECK(
        dnnl.dnnl_convolution_backward_weights_primitive_desc_create(
            &bwd_weight_primitive_desc,
            dnnl_engine_bwd,
            dnnl.dnnl_convolution_direct,
            conv2d_data.input_mem_desc,
            conv2d_data.weight_mem_desc,
            if (compute_bias_grad) conv2d_data.bias_mem_desc else null,
            conv2d_data.output_mem_desc,
            conv2d_data.stride_dims.ptr,
            conv2d_data.dilation_dims.ptr,
            conv2d_data.padding_dims.ptr,
            conv2d_data.padding_dims.ptr,
            conv2d_data.fwd_prim_desc,
            null,
        ),
        @src(),
    );

    // Weight backward primitive
    var bwd_weights: dnnl.dnnl_primitive_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&bwd_weights, bwd_weight_primitive_desc), @src());
    defer dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(bwd_weight_primitive_desc), @src()) catch unreachable;

    // Create memory
    var input_raw_mem_init = try DnnlMemoryWrapper.init(allocator, input, conv2d_data.input_dims, format_nchw);
    defer input_raw_mem_init.deinitAll();
    var grad_output_mem_init = try DnnlMemoryWrapper.init(allocator, grad_output, conv2d_data.output_dims, format_nchw);
    defer grad_output_mem_init.deinitAll();
    var grad_weights_mem_init = try DnnlMemoryWrapper.init(allocator, grad_weights, conv2d_data.weight_dims, format_weight);
    defer grad_weights_mem_init.deinitAll();

    var network_backwards = std.ArrayList(dnnl.dnnl_primitive_t).init(allocator);
    defer {
        for (network_backwards.items) |p| dnnl.DNNL_CHECK(dnnl.dnnl_primitive_destroy(p), @src()) catch unreachable;
        network_backwards.deinit();
    }
    var bwd_weights_args = std.ArrayList([]dnnl.dnnl_exec_arg_t).init(allocator);
    defer {
        for (bwd_weights_args.items) |a| allocator.free(a);
        bwd_weights_args.deinit();
    }

    // Check for reorderings, reorder if needed
    const input_desc = dnnl.dnnl_primitive_desc_query_md(bwd_weight_primitive_desc, dnnl.dnnl_query_src_md, 0);
    const grad_output_desc = dnnl.dnnl_primitive_desc_query_md(bwd_weight_primitive_desc, dnnl.dnnl_query_diff_dst_md, 0);
    const grad_weights_desc = dnnl.dnnl_primitive_desc_query_md(bwd_weight_primitive_desc, dnnl.dnnl_query_diff_weights_md, 0);
    const input_memory_bwds_allocated, const input_memory_backwards = try dnnlAlignOrdering(allocator, &network_backwards, &bwd_weights_args, input_raw_mem_init.getMemory(), input_desc);
    defer if (input_memory_bwds_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(input_memory_backwards), @src()) catch unreachable;
    const grad_output_memory_allocated, const grad_output_memory = try dnnlAlignOrdering(allocator, &network_backwards, &bwd_weights_args, grad_output_mem_init.getMemory(), grad_output_desc);
    defer if (grad_output_memory_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(grad_output_memory), @src()) catch unreachable;
    var grad_bias_mem: DnnlMemoryWrapper = if (compute_bias_grad) try DnnlMemoryWrapper.init(allocator, grad_bias, conv2d_data.bias_dims, format_bias) else .{};
    defer grad_bias_mem.deinitAll();
    // Don't reorder the grads until after the conv bwd
    var grad_weights_memory_allocated = false;
    var grad_weights_memory = grad_weights_mem_init.getMemory();
    defer if (grad_weights_memory_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(grad_weights_memory), @src()) catch unreachable;
    if (dnnl.dnnl_memory_desc_equal(grad_weights_mem_init.getDescriptor(), grad_weights_desc) == 0) {
        try dnnl.DNNL_CHECK(dnnl.dnnl_memory_create(&grad_weights_memory, grad_weights_desc, dnnl_engine_bwd, dnnl.DNNL_MEMORY_ALLOCATE), @src());
        grad_weights_memory_allocated = true;
    }

    // Create the convolution backward weight
    var bwd_conv_weights_args = try createArgs(
        allocator,
        &.{
            .{ dnnl.DNNL_ARG_SRC, input_memory_backwards },
            .{ dnnl.DNNL_ARG_DIFF_WEIGHTS, grad_weights_memory },
            .{ dnnl.DNNL_ARG_DIFF_DST, grad_output_memory },
        },
        if (compute_bias_grad) 4 else null,
    );
    if (compute_bias_grad) {
        bwd_conv_weights_args[3] = .{ .arg = dnnl.DNNL_ARG_DIFF_BIAS, .memory = grad_bias_mem.getMemory() };
    }

    try network_backwards.append(bwd_weights);
    try bwd_weights_args.append(bwd_conv_weights_args);

    // Reorder weight gradients if necessary
    var grad_weights_md: dnnl.dnnl_memory_desc_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_get_memory_desc(grad_weights_memory, &grad_weights_md), @src());
    if (grad_weights_memory != grad_weights_mem_init.getMemory()) {
        var reorder: dnnl.dnnl_primitive_t = null;
        var reorder_pd: dnnl.dnnl_primitive_desc_t = null;
        var grad_weights_new_md: dnnl.const_dnnl_memory_desc_t = null;
        try dnnl.DNNL_CHECK(dnnl.dnnl_memory_get_memory_desc(grad_weights_memory, &grad_weights_new_md), @src());
        try dnnl.DNNL_CHECK(dnnl.dnnl_reorder_primitive_desc_create(&reorder_pd, grad_weights_new_md, (try DnnlEngine.getInstance(allocator)).getEngine(), grad_weights_mem_init.getDescriptor(), (try DnnlEngine.getInstance(allocator)).getEngine(), null), @src());
        try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&reorder, reorder_pd), @src());
        try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(reorder_pd), @src());
        try network_backwards.append(reorder);
        try bwd_weights_args.append(try createArgs(allocator, &.{ .{ dnnl.DNNL_ARG_FROM, grad_weights_memory }, .{ dnnl.DNNL_ARG_TO, grad_weights_mem_init.getMemory() } }, null));
    }

    // Run
    try executeNetwork(allocator, &network_backwards, &bwd_weights_args);

    return .{ grad_weights, grad_bias };
}
