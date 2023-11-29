const std = @import("std");
const zt = @import("../../../../zt.zig");
const dnnl = @import("../../../../bindings/onednn/onednn.zig");
const dnnl_utils = @import("dnnl_utils.zig");
const zigrc = @import("zigrc");

const AutogradPayload = zt.autograd.AutogradPayload;
const Shape = zt.tensor.shape.Shape;
const Tensor = zt.tensor.Tensor;
const PoolingMode = zt.common.PoolingMode;
const createArgs = dnnl_utils.createArgs;
const convertToDnnlDims = dnnl_utils.convertToDnnlDims;
const dnnlMapToType = dnnl_utils.dnnlMapToType;
const dnnlMapToPoolingMode = dnnl_utils.dnnlMapToPoolingMode;
const dnnlAlignOrdering = dnnl_utils.dnnlAlignOrdering;
const executeNetwork = dnnl_utils.executeNetwork;
const DnnlEngine = dnnl_utils.DnnlEngine;
const DnnlMemoryWrapper = dnnl_utils.DnnlMemoryWrapper;

const k_w_idx: usize = 0;
const k_h_idx: usize = 1;
const k_channel_size_idx: usize = 2;
const k_batch_size_idx: usize = 3;

// Use `dnnl.dnnl_format_tag_any` for memory formatting even if pool
// inputs are shaped in a particular way.
const format_any: dnnl.dnnl_format_tag_t = dnnl.dnnl_format_tag_any;
const format_nchw: dnnl.dnnl_format_tag_t = dnnl.dnnl_nchw;

const DimsData = struct {
    allocator: std.mem.Allocator,
    input_dims: []i64 = &[_]i64{},
    output_dims: []i64 = &[_]i64{},
    window_dims: []i64 = &[_]i64{},
    stride_dims: []i64 = &[_]i64{},
    padding_dims: []i64 = &[_]i64{},
    dilation_dims: []i64 = &[_]i64{},

    pub fn init(
        allocator: std.mem.Allocator,
        input: Shape,
        output: Shape,
        wx: i64,
        wy: i64,
        sx: i64,
        sy: i64,
        px: i64,
        py: i64,
    ) !DimsData {
        var d: DimsData = .{ .allocator = allocator };
        d.input_dims = try convertToDnnlDims(allocator, &.{
            try zt.tensor.shape.dim(input, k_batch_size_idx),
            try zt.tensor.shape.dim(input, k_channel_size_idx),
            try zt.tensor.shape.dim(input, k_h_idx),
            try zt.tensor.shape.dim(input, k_w_idx),
        });
        d.output_dims = try convertToDnnlDims(allocator, &.{
            try zt.tensor.shape.dim(input, k_batch_size_idx),
            try zt.tensor.shape.dim(input, k_channel_size_idx),
            try zt.tensor.shape.dim(output, k_h_idx),
            try zt.tensor.shape.dim(output, k_w_idx),
        });
        d.window_dims = try convertToDnnlDims(allocator, &.{ wy, wx });
        d.stride_dims = try convertToDnnlDims(allocator, &.{ sy, sx });
        d.padding_dims = try convertToDnnlDims(allocator, &.{ py, px });
        d.dilation_dims = try convertToDnnlDims(allocator, &.{ 1, 1 });
        return d;
    }

    pub fn deinit(self: *DimsData) void {
        self.allocator.free(self.input_dims);
        self.allocator.free(self.output_dims);
        self.allocator.free(self.window_dims);
        self.allocator.free(self.stride_dims);
        self.allocator.free(self.padding_dims);
        self.allocator.free(self.dilation_dims);
    }
};

const OneDnnPool2DPayload = struct {
    allocator: std.mem.Allocator,
    workspace: dnnl.dnnl_memory_t = null,
    output_memory: dnnl.dnnl_memory_t = null,
    dims_data: *DimsData = undefined,
    pooling_fwd_prim_desc: dnnl.dnnl_primitive_desc_t = null,

    pub fn init(allocator: std.mem.Allocator) !*OneDnnPool2DPayload {
        const self = try allocator.create(OneDnnPool2DPayload);
        self.* = .{ .allocator = allocator, .dims_data = try allocator.create(DimsData) };
        return self;
    }

    pub fn deinit(self: *OneDnnPool2DPayload) void {
        if (self.workspace != null) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(self.workspace), @src()) catch unreachable;
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(self.output_memory), @src()) catch unreachable;
        self.dims_data.deinit();
        dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(self.pooling_fwd_prim_desc), @src()) catch unreachable;
        self.allocator.destroy(self);
    }
};

// TODO: test to verify freeing memory allocated by onednn
pub inline fn pool2d(
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
    const train = autograd_payload != null;
    var payload_data = try OneDnnPool2DPayload.init(allocator);
    const payload = try zigrc.Arc(?*anyopaque).init(allocator, payload_data);
    if (train) {
        autograd_payload.?.value.data = payload;
    }

    // inputX x inputY x channels x batch
    const ix = try input.dim(allocator, k_w_idx);
    const iy = if (try input.ndim(allocator) > k_h_idx) try input.dim(allocator, k_h_idx) else 1;
    const c = if (try input.ndim(allocator) > k_channel_size_idx) try input.dim(allocator, k_channel_size_idx) else 1;
    const b = if (try input.ndim(allocator) > k_batch_size_idx) try input.dim(allocator, k_batch_size_idx) else 1;

    var output = try Tensor.initHandle(
        allocator,
        &.{
            1 + @divTrunc(ix + 2 * px - wx, sx),
            1 + @divTrunc(iy + 2 * py - wy, sy),
            c,
            b,
        },
        try input.dtype(allocator),
    );

    payload_data.dims_data.* = try DimsData.init(allocator, &.{ ix, iy, c, b }, try output.shape(allocator), wx, wy, sx, sy, px, py);
    const d = payload_data.dims_data;
    const data_type = try dnnlMapToType(try input.dtype(allocator));

    // Memory desc
    var input_md: dnnl.dnnl_memory_desc_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_create_with_tag(&input_md, @intCast(d.input_dims.len), d.input_dims.ptr, data_type, format_nchw), @src());
    defer dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_destroy(input_md), @src()) catch unreachable;
    var output_md: dnnl.dnnl_memory_desc_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_create_with_tag(&output_md, @intCast(d.output_dims.len), d.output_dims.ptr, data_type, format_any), @src());
    defer dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_destroy(output_md), @src()) catch unreachable;

    // Memory
    const dnnl_engine = (try DnnlEngine.getInstance(allocator)).getEngine();
    var input_mem_init = try DnnlMemoryWrapper.init(allocator, input, d.input_dims, format_nchw);
    defer input_mem_init.deinit();
    var output_mem_init = try DnnlMemoryWrapper.init(allocator, output, d.output_dims, format_nchw);
    defer output_mem_init.deinit();

    // Choose a mode based on whether gradients are needed
    const forward_mode: dnnl.dnnl_prop_kind_t = if (train) dnnl.dnnl_forward else dnnl.dnnl_forward_inference;

    // Descriptors
    const pooling_mode: dnnl.dnnl_alg_kind_t = dnnlMapToPoolingMode(mode);
    try dnnl.DNNL_CHECK(
        dnnl.dnnl_pooling_forward_primitive_desc_create(
            &payload_data.pooling_fwd_prim_desc,
            dnnl_engine,
            forward_mode,
            pooling_mode,
            input_md,
            output_md,
            d.stride_dims.ptr,
            d.window_dims.ptr,
            d.dilation_dims.ptr, // dilation -- TODO: add to API (hardcoded to { 1, 1 } at present)
            d.padding_dims.ptr,
            d.padding_dims.ptr,
            null,
        ),
        @src(),
    );
    const prim_desc = payload_data.pooling_fwd_prim_desc;

    // Network
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

    // Reorder if needed
    const input_desc: dnnl.const_dnnl_memory_desc_t = dnnl.dnnl_primitive_desc_query_md(prim_desc, dnnl.dnnl_query_src_md, 0);
    const output_desc: dnnl.const_dnnl_memory_desc_t = dnnl.dnnl_primitive_desc_query_md(prim_desc, dnnl.dnnl_query_dst_md, 0);
    const input_memory_allocated, const input_memory = try dnnlAlignOrdering(allocator, &network, &fwd_args, input_mem_init.getMemory(), input_desc);
    defer if (input_memory_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(input_memory), @src()) catch unreachable;
    payload_data.output_memory = output_mem_init.getMemory();
    if (dnnl.dnnl_memory_desc_equal(output_mem_init.getDescriptor(), output_desc) == 0) {
        try dnnl.DNNL_CHECK(dnnl.dnnl_memory_create(&payload_data.output_memory, output_desc, dnnl_engine, dnnl.DNNL_MEMORY_ALLOCATE), @src());
    }

    // Workspace and layer (only training mode requires a workspace)
    var pooling: dnnl.dnnl_primitive_t = null;
    var fwd_pooling_args = try createArgs(allocator, &.{ .{ dnnl.DNNL_ARG_SRC, input_memory }, .{ dnnl.DNNL_ARG_DST, payload_data.output_memory } }, if (train) 3 else null);
    if (train) {
        const workspace_desc: dnnl.const_dnnl_memory_desc_t = dnnl.dnnl_primitive_desc_query_md(prim_desc, dnnl.dnnl_query_workspace_md, 0);
        try dnnl.DNNL_CHECK(dnnl.dnnl_memory_create(&payload_data.workspace, workspace_desc, dnnl_engine, dnnl.DNNL_MEMORY_ALLOCATE), @src());
        fwd_pooling_args[2] = .{ .arg = dnnl.DNNL_ARG_WORKSPACE, .memory = payload_data.workspace };
    }
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&pooling, prim_desc), @src());
    try network.append(pooling);
    try fwd_args.append(fwd_pooling_args);

    // Add output reordering if needed
    if (payload_data.output_memory != output_mem_init.getMemory()) {
        var reorder: dnnl.dnnl_primitive_t = null;
        var reorder_pd: dnnl.dnnl_primitive_desc_t = null;
        try dnnl.DNNL_CHECK(dnnl.dnnl_reorder_primitive_desc_create(&reorder_pd, output_desc, dnnl_engine, output_mem_init.getDescriptor(), dnnl_engine, null), @src());
        try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&reorder, reorder_pd), @src());
        try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(reorder_pd), @src());
        try network.append(reorder);
        try fwd_args.append(try createArgs(allocator, &.{ .{ dnnl.DNNL_ARG_FROM, payload_data.output_memory }, .{ dnnl.DNNL_ARG_TO, output_mem_init.getMemory() } }, null));
    }

    try executeNetwork(allocator, &network, &fwd_args);
    return output;
}

pub inline fn pool2dBackward(
    allocator: std.mem.Allocator,
    grad_output: Tensor,
    input: Tensor,
    _: Tensor,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    mode: PoolingMode,
    autograd_payload: ?zigrc.Arc(AutogradPayload),
) !Tensor {
    if (autograd_payload == null) {
        std.debug.print("OneDnnAutogradExtension.{s}: given null AutogradPayload\n", .{@src().fn_name});
        return error.NullAutogradPayload;
    }
    const payload: *OneDnnPool2DPayload = @ptrCast(@alignCast(autograd_payload.?.value.data.value));

    const grad_input = try Tensor.initHandle(allocator, try input.shape(allocator), .f32);
    const dnnl_engine_bwd = (try DnnlEngine.getInstance(allocator)).getEngine();

    const d = payload.dims_data;
    const pooling_mode = dnnlMapToPoolingMode(mode);
    _ = try dnnlMapToType(try input.dtype(allocator));

    // Memory
    var grad_input_mem_init = try DnnlMemoryWrapper.init(allocator, grad_input, d.input_dims, format_nchw);
    defer grad_input_mem_init.deinitAll();
    var grad_output_mem_init = try DnnlMemoryWrapper.init(allocator, grad_output, d.output_dims, format_nchw);
    defer grad_output_mem_init.deinitAll();

    // Descriptors
    // Memory descriptors from initialized memory must be used since
    // pooling_backward descriptors require an ordering
    var bwd_primitive_desc: dnnl.dnnl_primitive_desc_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_pooling_backward_primitive_desc_create(
        &bwd_primitive_desc,
        dnnl_engine_bwd,
        pooling_mode,
        grad_input_mem_init.getDescriptor(),
        grad_output_mem_init.getDescriptor(),
        d.stride_dims.ptr,
        d.window_dims.ptr,
        d.dilation_dims.ptr,
        d.padding_dims.ptr,
        d.padding_dims.ptr,
        payload.pooling_fwd_prim_desc, // hint
        null,
    ), @src());
    defer dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(bwd_primitive_desc), @src()) catch unreachable;

    var network_backward = std.ArrayList(dnnl.dnnl_primitive_t).init(allocator);
    defer {
        for (network_backward.items) |p| dnnl.DNNL_CHECK(dnnl.dnnl_primitive_destroy(p), @src()) catch unreachable;
        network_backward.deinit();
    }
    var bwd_args = std.ArrayList([]dnnl.dnnl_exec_arg_t).init(allocator);
    defer {
        for (bwd_args.items) |a| allocator.free(a);
        bwd_args.deinit();
    }
    // Reorder output memory if required
    var output_md: dnnl.const_dnnl_memory_desc_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_get_memory_desc(payload.output_memory, &output_md), @src());
    const grad_output_mem_allocated, const grad_output_memory = try dnnlAlignOrdering(allocator, &network_backward, &bwd_args, grad_output_mem_init.getMemory(), output_md);
    defer if (grad_output_mem_allocated) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(grad_output_memory), @src()) catch unreachable;

    var pool_bwd: dnnl.dnnl_primitive_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&pool_bwd, bwd_primitive_desc), @src());
    try bwd_args.append(try createArgs(allocator, &.{
        .{ dnnl.DNNL_ARG_DIFF_SRC, grad_input_mem_init.getMemory() },
        .{ dnnl.DNNL_ARG_DIFF_DST, grad_output_memory },
        .{ dnnl.DNNL_ARG_WORKSPACE, payload.workspace },
    }, null));
    try network_backward.append(pool_bwd);

    try executeNetwork(allocator, &network_backward, &bwd_args);

    return grad_input;
}
