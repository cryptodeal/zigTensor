const zt = @import("../../../../zt.zig");
const zigrc = @import("zigrc");
const dnnl = @import("../../../../bindings/onednn/onednn.zig");
const std = @import("std");
const dnnl_utils = @import("dnnl_utils.zig");

const dnnlMapToType = dnnl_utils.dnnlMapToType;
const convertToDnnlDims = dnnl_utils.convertToDnnlDims;
const createArgs = dnnl_utils.createArgs;
const executeNetwork = dnnl_utils.executeNetwork;
const DnnlEngine = dnnl_utils.DnnlEngine;
const DnnlMemoryWrapper = dnnl_utils.DnnlMemoryWrapper;
const Index = zt.tensor.Index;
const Range = zt.tensor.Range;
const Shape = zt.tensor.shape.Shape;
const Tensor = zt.tensor.Tensor;
const RnnMode = zt.common.RnnMode;
const AutogradPayload = zt.autograd.AutogradPayload;

pub const ParsedWeightsAndBias = struct {
    allocator: std.mem.Allocator,
    // First layer - will be empty if inSize == hiddenSize
    weights_input_1l: Tensor = undefined,
    weights_hidden_1l: Tensor = undefined,
    bias_1l: Tensor = undefined,
    // All other layers
    weights_input: Tensor = undefined,
    weights_hidden: Tensor = undefined,
    bias: Tensor = undefined,

    pub fn init(
        allocator: std.mem.Allocator,
        weights: Tensor,
        mode: RnnMode,
        num_layers: i64,
        direction_mult: i64,
        in_size: i64,
        num_gates: i64,
        hidden_size: i64,
    ) !*ParsedWeightsAndBias {
        var out = try allocator.create(ParsedWeightsAndBias);
        out.* = .{ .allocator = allocator };

        // Per-layer sizes for weights_input and weights_hidden.
        // If in_size == hidden_size, then weights_input_size == weights_hidden_size for all
        // layers, else all but the first layer
        const weights_input_size_1l = direction_mult * in_size * num_gates * hidden_size;
        const weights_hidden_size = direction_mult * hidden_size * num_gates * hidden_size;
        const weights_input_size = weights_hidden_size;
        const lbr_gru_bias: i64 = if (mode == .Gru) 1 else 0;
        const bias_size = num_layers * direction_mult * (num_gates + lbr_gru_bias) * hidden_size;

        const first_layer_different = in_size != hidden_size;
        // Adjusted if skipping first layer parsing
        const num_weights_layers = if (first_layer_different) num_layers - 1 else num_layers;
        const weights_offset = if (first_layer_different) weights_input_size_1l + weights_hidden_size else 0;
        // If skipping the first layer, parse then skip over the first layer
        // weights and parse the remaining layers. Parsing all bias layers is still
        // fine since biases for each layer have the same size
        if (first_layer_different) {
            out.weights_input_1l = try weights.flat(allocator, Index.initRange(Range.initEnd(weights_input_size_1l)));
            out.weights_hidden_1l = try weights.flat(allocator, Index.initRange(Range.init(weights_input_size_1l, .{ .dim = weights_input_size_1l + weights_hidden_size })));

            if (mode == .Gru) {
                var tmp1 = try reorderLbrGruWeights(allocator, in_size, hidden_size, out.weights_input_1l);
                defer tmp1.deinit();
                try out.weights_input_1l.assign(allocator, Tensor, tmp1);
                var tmp2 = try reorderLbrGruWeights(allocator, hidden_size, hidden_size, out.weights_hidden_1l);
                defer tmp2.deinit();
                try out.weights_hidden_1l.assign(allocator, Tensor, tmp2);
            }
        }

        var weights_flat = try weights.flatten(allocator);
        defer weights_flat.deinit();
        // cuDNN RNN weights, for each layer, are arranged with a chunk of
        // input-hidden weights for each layer followed by a chunk of hidden-hidden
        // weights for each layer:
        // {[layers x [hidden_size, input_size]], [layers x  [hidden_size, hidden_size]] }
        // Rearrange this to what oneDNN expects (or will reorder if not optimal),
        // which is numLayers chunks of two chunks containing input-hidden and
        // hidden-hidden:
        // {[layers x [[hidden_size x in_size], [hidden_size x hidden_size]]]}
        // Note that the loop is over the total number of layers in case we'r doing a
        // single-layer operation where input size and hidden size are different but
        // we'll call another primitive with the output of that first layer as the
        // input to the next layers
        var weights_input = try Tensor.initHandle(allocator, &.{0}, try weights.dtype(allocator));
        var weights_hidden = try Tensor.initHandle(allocator, &.{0}, try weights.dtype(allocator));
        var weights_flat_offset = try weights_flat.flat(allocator, Index.initRange(Range.init(weights_offset, .{ .end = zt.tensor.end })));
        defer weights_flat_offset.deinit();
        // Specifically ignore the first layer's weights, so in_size == hidden_size
        for (0..@intCast(num_weights_layers)) |i| {
            // number of input/hidden weights
            // TODO: Will change for bidirectional
            const chunk_size = hidden_size * hidden_size * num_gates;
            // weights per layer
            const layer_chunk_size = chunk_size + chunk_size;

            // Grab input-hidden weights and chunk them together
            var input_weights_chunk = try weights_flat_offset.flat(allocator, Index.initRange(Range.init(
                layer_chunk_size * @as(i64, @intCast(i)),
                .{ .dim = layer_chunk_size * @as(i64, @intCast(i)) + chunk_size },
            )));
            defer input_weights_chunk.deinit();
            // Grab hidden-hidden weights and chunk them together
            var input_hidden_chunk = try weights_flat_offset.flat(allocator, Index.initRange(Range.init(
                layer_chunk_size * @as(i64, @intCast(i)) + chunk_size,
                .{ .dim = layer_chunk_size * @as(i64, @intCast(i)) + chunk_size + chunk_size },
            )));
            defer input_hidden_chunk.deinit();

            if (mode == .Gru) {
                var tmp1 = try reorderLbrGruWeights(allocator, hidden_size, hidden_size, input_weights_chunk);
                defer tmp1.deinit();
                try input_weights_chunk.assign(allocator, Tensor, tmp1);
                var tmp2 = try reorderLbrGruWeights(allocator, hidden_size, hidden_size, input_hidden_chunk);
                defer tmp2.deinit();
                try input_hidden_chunk.assign(allocator, Tensor, tmp2);
            }

            var tmp1 = try zt.tensor.concatenate(allocator, &.{ weights_input, input_weights_chunk }, 2);
            defer tmp1.deinit();
            try weights_input.assign(allocator, Tensor, tmp1);
            var tmp2 = try zt.tensor.concatenate(allocator, &.{ weights_hidden, input_hidden_chunk }, 2);
            defer tmp2.deinit();
            try weights_hidden.assign(allocator, Tensor, tmp2);
        }
        out.weights_input = weights_input;
        out.weights_hidden = weights_hidden;

        // Reduce the weights to form biases. cuDNN uses two separate bias terms:
        // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNMode_t -
        // oneDNN expects only one bias term. Sum together the coefficients for both
        // bias terms to get a single bias term for oneDNN. The gradients for
        // each term can be computed as one since the gradients with respect to
        // the bias subarrays will simply be half of the computed gradient with
        // oneDNN
        var bias = try Tensor.initHandle(allocator, &.{0}, try weights.dtype(allocator));
        const bias_start_offset = num_layers * weights_hidden_size + (num_layers - 1) * weights_input_size + weights_input_size_1l;
        // In vanilla RNN modes, the biases can be simply added:
        // two biases for each bias in zt cuDNN with CUDNN_RNN_DOUBLE_BIAS (default)
        const num_biases: i64 = 2;
        // First, grab a subarray which contains only both bias terms; then add them
        var bias_flat = try weights_flat.flat(allocator, Index.initRange(Range.init(bias_start_offset, .{ .end = zt.tensor.end })));
        defer bias_flat.deinit();
        // Layout is: {num_layers x [num_biases x [bias shape]]}
        for (0..@intCast(num_layers)) |i| {
            if (mode == .Gru) {
                const lbr_gru_chunk_size = hidden_size * 6;
                // In the case of the LBR GRU, there's an extra bias term which shouldn't
                // be combined with the first two pairs of biases. Six chunks total.
                // cuDNN --> oneDNN transformation for ordering:
                // r1, u1, o, r2, u2, u' --> u1 + u2, r1 + r2, o, u'
                const base = @as(i64, @intCast(i)) * lbr_gru_chunk_size;

                // u1_ -- [1, 2]
                var concat_tmp1 = try bias_flat.flat(allocator, Index.initRange(Range.init(base + hidden_size * 1, .{ .dim = base + hidden_size * 2 })));
                // r1 -- [0, 1]
                var concat_tmp2 = try bias_flat.flat(allocator, Index.initRange(Range.init(base + hidden_size * 0, .{ .dim = base + hidden_size * 1 })));
                // o -- [2, 3]
                var concat_tmp3 = try bias_flat.flat(allocator, Index.initRange(Range.init(base + hidden_size * 2, .{ .dim = base + hidden_size * 3 })));
                // u' -- [5, 6]
                var concat_tmp4 = try bias_flat.flat(allocator, Index.initRange(Range.init(base + hidden_size * 5, .{ .dim = base + hidden_size * 6 })));
                // The sum of the following tensors yields the correct bias
                // u1, r1, o, u'
                var biases1 = try zt.tensor.concatenate(allocator, &.{ concat_tmp1, concat_tmp2, concat_tmp3, concat_tmp4 }, 0);
                defer biases1.deinit();
                concat_tmp1.deinit();
                concat_tmp2.deinit();
                concat_tmp3.deinit();
                concat_tmp4.deinit();

                // u2 -- [4, 5]
                concat_tmp1 = try bias_flat.flat(allocator, Index.initRange(Range.init(base + hidden_size * 4, .{ .dim = base + hidden_size * 5 })));
                // r2 -- [3, 4]
                concat_tmp2 = try bias_flat.flat(allocator, Index.initRange(Range.init(base + hidden_size * 3, .{ .dim = base + hidden_size * 4 })));
                // zeroes to add to o and u'
                concat_tmp3 = try zt.tensor.full(allocator, &.{hidden_size * 2}, f64, 0, try bias_flat.dtype(allocator));
                // u2, r2, 0, 0
                var biases2 = try zt.tensor.concatenate(allocator, &.{ concat_tmp1, concat_tmp2, concat_tmp3 }, 0);
                defer biases2.deinit();
                concat_tmp1.deinit();
                concat_tmp2.deinit();
                concat_tmp3.deinit();

                var layer_bias_combined = try zt.tensor.add(allocator, Tensor, biases1, Tensor, biases2);
                defer layer_bias_combined.deinit();
                var tmp_res = try zt.tensor.concatenate(allocator, &.{ bias, layer_bias_combined }, 0);
                defer tmp_res.deinit();
                try bias.assign(allocator, Tensor, tmp_res);
            } else {
                // The number of bias terms in the tensor per-layer
                const layer_stride = @divTrunc(bias_size, num_layers * num_biases);
                var biases1 = try bias_flat.index(allocator, &.{Index.initRange(Range.init(layer_stride * @as(i64, @intCast(i)), .{ .dim = layer_stride * @as(i64, @intCast(i)) + @divTrunc(layer_stride, num_biases) }))});
                defer biases1.deinit();
                var biases2 = try bias_flat.index(allocator, &.{Index.initRange(Range.init(layer_stride * @as(i64, @intCast(i)) + @divTrunc(layer_stride, num_biases), .{ .dim = layer_stride * (@as(i64, @intCast(i)) + 1) }))});
                defer biases2.deinit();
                var layer_bias_combined = try zt.tensor.add(allocator, Tensor, biases1, Tensor, biases2);
                defer layer_bias_combined.deinit();
                var tmp_res = try zt.tensor.concatenate(allocator, &.{ bias, layer_bias_combined }, 0);
                defer tmp_res.deinit();
                try bias.assign(allocator, Tensor, tmp_res);
            }
        }

        if (first_layer_different) {
            out.bias_1l = try bias.flat(allocator, Index.initRange(Range.initEnd(@divTrunc(bias_size, num_layers))));
            if (num_layers > 1) {
                // bias for the second --> last layer
                var tmp = try bias.flat(allocator, Index.initRange(Range.init(@divTrunc(bias_size, num_layers), .{ .end = zt.tensor.end })));
                defer tmp.deinit();
                try bias.assign(allocator, Tensor, tmp);
            }
        }
        out.bias = bias;

        // Case for a single layer of different in/hidden size
        if (first_layer_different and num_layers == 1) {
            var tmp1 = try out.weights_input_1l.copy(allocator);
            defer tmp1.deinit();
            try out.weights_input.assign(allocator, Tensor, tmp1);
            var tmp2 = try out.weights_hidden_1l.copy(allocator);
            defer tmp2.deinit();
            try out.weights_hidden.assign(allocator, Tensor, tmp2);
            var tmp3 = try out.bias_1l.copy(allocator);
            defer tmp3.deinit();
            try out.bias.assign(allocator, Tensor, tmp3);
        }

        return out;
    }

    pub fn deinit(self: *ParsedWeightsAndBias) void {
        self.weights_input_1l.deinit();
        self.weights_hidden_1l.deinit();
        self.bias_1l.deinit();
        self.weights_input.deinit();
        self.weights_hidden.deinit();
        self.bias.deinit();
        self.allocator.destroy(self);
    }
};

// Each gate's weights have dimensions d1 x d2
fn reorderLbrGruWeights(allocator: std.mem.Allocator, d1: i64, d2: i64, weights: Tensor) !Tensor {
    // LBR GRU requires switch the given the r, u, o gate order from cuDNN to u,
    // r, o as required by oneDNN (this from empirical verification)
    const weights_size = d1 * d2;
    if (try weights.elements(allocator) != weights_size * 3) {
        std.debug.print("RNN {s} given invalid weights tensor or dims - weights of size {d}  which should be exactly {d}\n", .{ @src().fn_name, try weights.elements(allocator), weights_size * 3 });
        return error.RNNInvalidWeightsTensorOrDims;
    }

    var concat1 = try weights.flat(allocator, Index.initRange(Range.init(weights_size, .{ .dim = weights_size * 2 })));
    defer concat1.deinit();
    var concat2 = try weights.flat(allocator, Index.initRange(Range.init(0, .{ .dim = weights_size })));
    defer concat2.deinit();
    var concat3 = try weights.flat(allocator, Index.initRange(Range.init(2 * weights_size, .{ .end = zt.tensor.end })));
    defer concat3.deinit();
    return zt.tensor.concatenate(allocator, &.{ concat1, concat2, concat3 }, 0);
}

pub const RnnResult = struct {
    workspace: dnnl.dnnl_memory_t = null,
    y: Tensor = undefined, // output
    hy: Tensor = undefined, // hidden output
    cy: Tensor = undefined, // cell output

    pub fn deinit(self: RnnResult) void {
        dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(self.workspace), @src()) catch unreachable;
        self.y.deinit();
        self.hy.deinit();
        self.cy.deinit();
    }
};

/// Does forward for a single onednn RNN primitive
fn rnnImpl(
    allocator: std.mem.Allocator,
    input: Tensor,
    hidden_state: Tensor,
    cell_state: Tensor,
    weights_input: Tensor,
    weights_hidden: Tensor,
    bias: Tensor,
    hidden_size: i64,
    num_layers: i64,
    mode: RnnMode,
    activation: dnnl.dnnl_alg_kind_t,
    num_gates: i64,
    direction: dnnl.dnnl_rnn_direction_t,
    direction_mult: i64,
    kind: dnnl.dnnl_prop_kind_t,
    dropout: f32,
) !RnnResult {
    _ = dropout;
    var result: RnnResult = .{};
    const dnnl_engine = (try DnnlEngine.getInstance(allocator)).getEngine();

    // Dimensions
    const in_size = try input.dim(allocator, 0);
    const batch_size: i64 = if (try input.ndim(allocator) < 2) 1 else try input.dim(allocator, 1);
    const seq_length: i64 = if (try input.ndim(allocator) < 3) 1 else try input.dim(allocator, 2);
    const input_dims: []const i64 = &.{ seq_length, batch_size, in_size };
    const output_dims: []const i64 = &.{ seq_length, batch_size, hidden_size * direction_mult };
    const d_type = try dnnlMapToType(try input.dtype(allocator));
    const total_layers = num_layers;
    const out_size = hidden_size;
    const h_dims: []const i64 = &.{ total_layers, direction_mult, batch_size, hidden_size };
    const c_dims: []const i64 = &.{ total_layers, direction_mult, batch_size, hidden_size };
    const extra_bias: i64 = if (mode == .Gru) 1 else 0; // for LBR GRU
    const bias_dims: []const i64 = &.{ num_layers, direction_mult, num_gates + extra_bias, hidden_size };
    // ldigo
    const weights_input_dims: []const i64 = &.{ num_layers, direction_mult, in_size, num_gates, hidden_size };
    const weights_hidden_dims: []const i64 = &.{ num_layers, direction_mult, hidden_size, num_gates, hidden_size };

    // Out tensors: output (y), hidden state output (hy), cell state output (cy)
    const y = try Tensor.initHandle(allocator, &.{ out_size, batch_size, seq_length }, try input.dtype(allocator));
    var hy = try Tensor.initHandle(allocator, &.{ hidden_size, batch_size, total_layers }, try input.dtype(allocator));
    const cy: Tensor = if (mode == .Lstm) try Tensor.initHandle(allocator, try hy.shape(allocator), try input.dtype(allocator)) else try Tensor.initEmpty(allocator);

    // Memory for forward
    const tnc: dnnl.dnnl_format_tag_t = dnnl.dnnl_tnc;
    const ldnc: dnnl.dnnl_format_tag_t = dnnl.dnnl_ldnc;
    const ldgoi: dnnl.dnnl_format_tag_t = dnnl.dnnl_ldgoi;
    const ldgo: dnnl.dnnl_format_tag_t = dnnl.dnnl_ldgo;
    var in_contig = try input.asContiguousTensor(allocator);
    defer in_contig.deinit();
    var input_mem_init = try DnnlMemoryWrapper.init(allocator, in_contig, input_dims, tnc);
    defer input_mem_init.deinit();
    var output_mem_init = try DnnlMemoryWrapper.init(allocator, y, output_dims, tnc);
    defer output_mem_init.deinit();
    var hidden_in_mem_init = DnnlMemoryWrapper{};
    var hidden_state_contig: Tensor = undefined;
    if (!try hidden_state.isEmpty(allocator)) {
        hidden_state_contig = try hidden_state.asContiguousTensor(allocator);
        hidden_in_mem_init = try DnnlMemoryWrapper.init(allocator, hidden_state_contig, h_dims, ldnc);
    }
    defer hidden_state_contig.deinit();
    var hidden_out_mem_init = try DnnlMemoryWrapper.init(allocator, hy, h_dims, ldnc);
    defer hidden_out_mem_init.deinit();
    var weights_input_contig = try weights_input.asContiguousTensor(allocator);
    defer weights_input_contig.deinit();
    var weights_input_mem_raw_init = try DnnlMemoryWrapper.init(allocator, weights_input_contig, weights_input_dims, ldgoi);
    defer weights_input_mem_raw_init.deinit();
    var weights_hidden_contig = try weights_hidden.asContiguousTensor(allocator);
    defer weights_hidden_contig.deinit();
    var weights_hidden_mem_raw_init = try DnnlMemoryWrapper.init(allocator, weights_hidden_contig, weights_hidden_dims, ldgoi);
    defer weights_hidden_mem_raw_init.deinit();
    var bias_contig = try bias.asContiguousTensor(allocator);
    defer bias_contig.deinit();
    var bias_mem_init = try DnnlMemoryWrapper.init(allocator, bias_contig, bias_dims, ldgo);
    defer bias_mem_init.deinit();
    var cell_in_mem_init = DnnlMemoryWrapper{};
    defer cell_in_mem_init.deinit();
    var cell_out_mem_init = DnnlMemoryWrapper{};
    defer cell_out_mem_init.deinit();

    // TODO: don't force a format tag - use any and do a reorder based
    // on the format of the primitive - what it says - like you're supposed to
    // Primitive for reordering input weights: ldgoi --> ldigo
    var weights_input_mem_desc: dnnl.dnnl_memory_desc_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_create_with_tag(&weights_input_mem_desc, @intCast(weights_input_dims.len), weights_input_dims.ptr, d_type, dnnl.dnnl_ldigo), @src());
    var weights_input_mem_init: dnnl.dnnl_memory_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_create(&weights_input_mem_init, weights_input_mem_desc, dnnl_engine, dnnl.DNNL_MEMORY_ALLOCATE), @src());
    // Primitive for reordering iter/hidden weights: ldgoi --> ldigo
    var weights_hidden_mem_desc: dnnl.dnnl_memory_desc_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_create_with_tag(&weights_hidden_mem_desc, @intCast(weights_hidden_dims.len), weights_hidden_dims.ptr, d_type, dnnl.dnnl_ldigo), @src());
    var weights_hidden_mem_init: dnnl.dnnl_memory_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_create(&weights_hidden_mem_init, weights_hidden_mem_desc, dnnl_engine, dnnl.DNNL_MEMORY_ALLOCATE), @src());

    // Add arguments
    var rnn_fwd_args = std.ArrayList(dnnl.dnnl_exec_arg_t).fromOwnedSlice(allocator, try createArgs(allocator, &.{
        .{ dnnl.DNNL_ARG_SRC_LAYER, input_mem_init.getMemory() },
        .{ dnnl.DNNL_ARG_SRC_ITER, hidden_in_mem_init.getMemory() },
        .{ dnnl.DNNL_ARG_WEIGHTS_LAYER, weights_input_mem_init },
        .{ dnnl.DNNL_ARG_WEIGHTS_ITER, weights_hidden_mem_init },
        .{ dnnl.DNNL_ARG_BIAS, bias_mem_init.getMemory() },
        .{ dnnl.DNNL_ARG_DST_LAYER, output_mem_init.getMemory() },
        .{ dnnl.DNNL_ARG_DST_ITER, hidden_out_mem_init.getMemory() },
    }, null));

    // Workspace memory, if needed
    var workspace: dnnl.dnnl_memory_t = null;
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

    // reorder input weights
    var reorder_prim_desc: dnnl.dnnl_primitive_desc_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_reorder_primitive_desc_create(&reorder_prim_desc, weights_input_mem_raw_init.getDescriptor(), dnnl_engine, weights_input_mem_desc, dnnl_engine, null), @src());
    var reorder_prim1: dnnl.dnnl_primitive_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&reorder_prim1, reorder_prim_desc), @src());
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(reorder_prim_desc), @src());
    try network.append(reorder_prim1);
    try fwd_args.append(try createArgs(allocator, &.{
        .{ dnnl.DNNL_ARG_FROM, weights_input_mem_raw_init.getMemory() },
        .{ dnnl.DNNL_ARG_TO, weights_input_mem_init },
    }, null));
    // reorder iter weights
    reorder_prim_desc = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_reorder_primitive_desc_create(&reorder_prim_desc, weights_hidden_mem_raw_init.getDescriptor(), dnnl_engine, weights_hidden_mem_desc, dnnl_engine, null), @src());
    var reorder_prim2: dnnl.dnnl_primitive_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&reorder_prim2, reorder_prim_desc), @src());
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(reorder_prim_desc), @src());
    try network.append(reorder_prim2);
    try fwd_args.append(try createArgs(allocator, &.{
        .{ dnnl.DNNL_ARG_FROM, weights_hidden_mem_raw_init.getMemory() },
        .{ dnnl.DNNL_ARG_TO, weights_hidden_mem_init },
    }, null));

    // Initialize descriptors
    var primitive_desc: dnnl.dnnl_primitive_desc_t = null;
    if (mode == .Relu or mode == .Tanh) {
        try dnnl.DNNL_CHECK(dnnl.dnnl_vanilla_rnn_forward_primitive_desc_create(
            &primitive_desc,
            dnnl_engine,
            kind,
            activation,
            direction,
            input_mem_init.getDescriptor(),
            hidden_in_mem_init.getDescriptor(),
            weights_input_mem_desc, // weights "layer"
            weights_hidden_mem_desc, // weights "iter"
            bias_mem_init.getDescriptor(),
            output_mem_init.getDescriptor(),
            hidden_out_mem_init.getDescriptor(),
            dnnl.dnnl_rnn_flags_undef,
            0,
            0,
            null,
        ), @src());
    } else if (mode == .Lstm) {
        // LSTM-only
        // input cell state
        // TODO: function that takes the array and
        // returns the desciptor and memory -- takes an argument for
        // which determines whether or not it's ok to return empty
        // descriptors if the array is empty
        if (!try cell_state.isEmpty(allocator)) {
            var cell_state_contig = try cell_state.asContiguousTensor(allocator);
            defer cell_state_contig.deinit();
            cell_in_mem_init = try DnnlMemoryWrapper.init(allocator, cell_state_contig, c_dims, ldnc);
        }
        // output cell state
        cell_out_mem_init = try DnnlMemoryWrapper.init(allocator, cy, c_dims, ldnc);

        try dnnl.DNNL_CHECK(dnnl.dnnl_lstm_forward_primitive_desc_create(
            &primitive_desc,
            dnnl_engine,
            kind,
            direction,
            input_mem_init.getDescriptor(),
            hidden_in_mem_init.getDescriptor(),
            cell_in_mem_init.getDescriptor(),
            weights_input_mem_desc, // weights "layer"
            weights_hidden_mem_desc, // weights "iter"
            null,
            null,
            bias_mem_init.getDescriptor(),
            output_mem_init.getDescriptor(),
            hidden_out_mem_init.getDescriptor(),
            cell_out_mem_init.getDescriptor(),
            dnnl.dnnl_rnn_flags_undef,
            null,
        ), @src());

        try rnn_fwd_args.append(.{ .arg = dnnl.DNNL_ARG_SRC_ITER_C, .memory = cell_in_mem_init.getMemory() });
        try rnn_fwd_args.append(.{ .arg = dnnl.DNNL_ARG_DST_ITER_C, .memory = cell_out_mem_init.getMemory() });
    } else if (mode == .Gru) {
        // Use a linear-before-reset GRU so we can have parity with cuDNN
        try dnnl.DNNL_CHECK(dnnl.dnnl_lbr_gru_forward_primitive_desc_create(
            &primitive_desc,
            dnnl_engine,
            kind,
            direction,
            input_mem_init.getDescriptor(),
            hidden_in_mem_init.getDescriptor(),
            weights_input_mem_desc,
            weights_hidden_mem_desc,
            bias_mem_init.getDescriptor(),
            output_mem_init.getDescriptor(),
            hidden_out_mem_init.getDescriptor(),
            dnnl.dnnl_rnn_flags_undef,
            null,
        ), @src());
    }
    var primitive: dnnl.dnnl_primitive_t = null;
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&primitive, primitive_desc), @src());
    try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(primitive_desc), @src());
    try network.append(primitive);

    const workspace_md: dnnl.const_dnnl_memory_desc_t = dnnl.dnnl_primitive_desc_query_md(primitive_desc, dnnl.dnnl_query_workspace_md, 0);
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_create(&workspace, workspace_md, dnnl_engine, dnnl.DNNL_MEMORY_ALLOCATE), @src());
    try rnn_fwd_args.append(.{ .arg = dnnl.DNNL_ARG_WORKSPACE, .memory = workspace });
    try fwd_args.append(try rnn_fwd_args.toOwnedSlice());

    try executeNetwork(allocator, &network, &fwd_args);

    result.y = y;
    result.hy = hy;
    result.cy = cy;
    result.workspace = workspace;
    return result;
}

pub inline fn rnn(
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
    if (dropout > 0) {
        std.debug.print("onednn RNN: dropout > 0.0 unsupported\n", .{});
        return error.RNNUnsupportedDropout;
    }
    if (bidirectional) {
        std.debug.print("onednn RNN: bidirectional not yet supported\n", .{});
        return error.RNNUnsupportedBidirectional;
    }

    const train = autograd_payload != null;

    // Constants
    const direction: dnnl.dnnl_rnn_direction_t = if (bidirectional) dnnl.dnnl_bidirectional_concat else dnnl.dnnl_unidirectional_left2right;
    const direction_mult: i64 = if (bidirectional) 2 else 1;
    const kind: dnnl.dnnl_prop_kind_t = if (train) dnnl.dnnl_forward_training else dnnl.dnnl_forward_inference;
    var num_gates: i64 = 1;
    var activation: dnnl.dnnl_alg_kind_t = dnnl.dnnl_alg_kind_undef;
    switch (mode) {
        .Lstm => {
            num_gates = 4;
        },
        .Gru => {
            num_gates = 3;
        },
        .Relu => {
            activation = dnnl.dnnl_eltwise_relu;
        },
        .Tanh => {
            activation = dnnl.dnnl_eltwise_tanh;
        },
    }

    const in_size = try input.dim(allocator, 0);

    // In zigTensor, all RNN weights are stored as one contiguous tensor, so we
    // have to parse out the input weights, input biases, hidden weights, and
    // hidden biases from one tensor. Order doesn't matter since the arrangement
    // is a black box
    const parsed_weights = try ParsedWeightsAndBias.init(allocator, weights, mode, num_layers, direction_mult, in_size, num_gates, hidden_size);

    var result: RnnResult = undefined;
    // The oneDNN RNN primitive has an API limitation where input size and
    // hidden size can only differ if the primitive has exactly one layer.
    // Therefore, for computations for more than one layer, first do the
    // operation for one layer, which gives an output vector of size [hidden
    // size, batch size, sequence length * number of directions], then use
    // that output as the input for layers [2, L]. Since the input size dim 0
    // is now the hidden size, the primitive can fuse computation for
    // arbitrarily-many layers.
    if (try input.dim(allocator, 0) == hidden_size or num_layers == 1) {
        // Input and hidden size are the same, or we only have one layer, which
        // means we can call the impl as is and parse weights "normally"
        result = try rnnImpl(
            allocator,
            input,
            hidden_state,
            cell_state,
            parsed_weights.weights_input,
            parsed_weights.weights_hidden,
            parsed_weights.bias,
            hidden_size,
            num_layers,
            mode,
            activation,
            num_gates,
            direction,
            direction_mult,
            kind,
            dropout,
        );
    } else {
        // We require more than one layer with different input and hidden states -
        // see the above. Seek to the first layer's hidden/cell state, weights, and
        // bias
        var tmp_hidden_state = try hidden_state.index(allocator, &.{ Index.initRange(zt.tensor.span), Index.initRange(zt.tensor.span), Index.initDim(0) });
        var tmp_cell_state = try cell_state.index(allocator, &.{ Index.initRange(zt.tensor.span), Index.initRange(zt.tensor.span), Index.initDim(0) });
        var result_l1 = try rnnImpl(
            allocator,
            input,
            tmp_hidden_state,
            tmp_cell_state,
            parsed_weights.weights_input_1l,
            parsed_weights.weights_hidden_1l,
            parsed_weights.bias_1l,
            hidden_size,
            1,
            mode,
            activation,
            num_gates,
            direction,
            direction_mult,
            kind,
            dropout,
        );
        defer result_l1.deinit();
        tmp_hidden_state.deinit();
        tmp_cell_state.deinit();

        // Layers [2..N] //
        // Seek  past the first layer's hidden/cell state, weights, and bias
        tmp_hidden_state = try hidden_state.index(allocator, &.{ Index.initRange(zt.tensor.span), Index.initRange(zt.tensor.span), Index.initRange(Range.init(1, .{ .end = zt.tensor.end })) });
        tmp_cell_state = try cell_state.index(allocator, &.{ Index.initRange(zt.tensor.span), Index.initRange(zt.tensor.span), Index.initRange(Range.init(1, .{ .end = zt.tensor.end })) });
        var result_l2n = try rnnImpl(
            allocator,
            result_l1.y,
            tmp_hidden_state,
            tmp_cell_state,
            parsed_weights.weights_input,
            parsed_weights.weights_hidden,
            parsed_weights.bias,
            hidden_size,
            num_layers - 1, // layers [2..N]
            mode,
            activation,
            num_gates,
            direction,
            direction_mult,
            kind,
            dropout,
        );
        tmp_hidden_state.deinit();
        tmp_cell_state.deinit();
        defer result_l2n.hy.deinit();
        defer result_l2n.cy.deinit();
        if (result_l2n.workspace != null) try dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(result_l2n.workspace), @src());

        result.y = result_l2n.y;
        result.hy = try zt.tensor.concatenate(allocator, &.{ result_l1.hy, result_l2n.hy }, 2);
        result.cy = try zt.tensor.concatenate(allocator, &.{ result_l1.cy, result_l2n.cy }, 2);
    }

    if (result.workspace != null) try dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(result.workspace), @src());
    return .{ result.y, result.hy, result.cy };
}
