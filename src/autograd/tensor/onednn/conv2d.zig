const std = @import("std");
const zt = @import("../../../zt.zig");
const dnnl = @import("../../../bindings/onednn/onednn.zig");
const dnnl_utils = @import("dnnl_utils.zig");

const dnnlMapToType = dnnl_utils.dnnlMapToType;
const convertToDnnlDims = dnnl_utils.convertToDnnlDims;
const DnnlEngine = dnnl_utils.DnnlEngine;

const Dim = zt.tensor.shape.Dim;
const Shape = zt.tensor.shape.Shape;

const DType = zt.tensor.DType;

// Input, output: WHCN; weights: WHIO
pub const kWIdx: usize = 0;
pub const kHIdx: usize = 1;
pub const kIOChannelSizeIdx: usize = 2;
pub const kIOBatchSizeIdx: usize = 3;
pub const kWeightOutputChannelSizeIdx: usize = 3;

pub const formatAny = dnnl.dnnl_format_tag_any;
pub const formatNCHW = dnnl.dnnl_abcd;
pub const formatBias = dnnl.dnnl_a;

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
    input_mem_desc: dnnl.dnnl_memory_desc_t,
    output_mem_desc: dnnl.dnnl_memory_desc_t,
    weight_mem_desc: dnnl.dnnl_memory_desc_t,
    bias_mem_desc: dnnl.dnnl_memory_desc_t,
    // used for creating a backward desc
    fwd_prim_desc: dnnl.dnnl_primitive_desc_t,

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
        _ = bias_shape;
        const data_type: dnnl.dnnl_data_type_t = try dnnlMapToType(input_type);
        const format_weight: dnnl.dnnl_format_tag_t = if (groups == 1) dnnl.dnnl_abcd else dnnl.dnnl_abcde;

        var out: Self = .{ .allocator = allocator };
        // Create memory dims
        out.input_dims = try convertToDnnlDims(allocator, &.{
            zt.tensor.shape.dim(input_shape, kIOBatchSizeIdx),
            zt.tensor.shape.dim(input_shape, kIOChannelSizeIdx),
            zt.tensor.shape.dim(input_shape, kHIdx),
            zt.tensor.shape.dim(input_shape, kWIdx),
        });
        if (groups == 1) {
            out.weight_dims = try convertToDnnlDims(allocator, &.{
                zt.tensor.shape.dim(weights_shape, kWeightOutputChannelSizeIdx),
                zt.tensor.shape.dim(input_shape, kIOChannelSizeIdx),
                zt.tensor.shape.dim(weights_shape, kHIdx),
                zt.tensor.shape.dim(weights_shape, kWIdx),
            });
        } else {
            out.weight_dims = try convertToDnnlDims(allocator, &.{
                groups,
                @divTrunc(zt.tensor.shape.dim(weights_shape, kWeightOutputChannelSizeIdx), groups),
                @divTrunc(zt.tensor.shape.dim(input_shape, kIOChannelSizeIdx), groups),
                zt.tensor.shape.dim(weights_shape, kHIdx),
                zt.tensor.shape.dim(weights_shape, kWIdx),
            });
        }
        out.output_dims = try convertToDnnlDims(allocator, &.{
            zt.tensor.shape.dim(input_shape, kIOBatchSizeIdx),
            zt.tensor.shape.dim(weights_shape, kWeightOutputChannelSizeIdx),
            zt.tensor.shape.dim(output_shape, kHIdx),
            zt.tensor.shape.dim(output_shape, kWIdx),
        });
        out.bias_dims = try convertToDnnlDims(allocator, &.{zt.tensor.shape.dim(weights_shape, kWeightOutputChannelSizeIdx)});
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
                formatAny,
            ),
            @src(),
        );
        try dnnl.DNNL_CHECK(
            dnnl.dnnl_memory_desc_create_with_tag(
                &out.output_mem_desc,
                @intCast(out.output_dims.len),
                out.output_dims.ptr,
                data_type,
                formatAny,
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
                formatAny,
            ),
            @src(),
        );

        const forward_mode: dnnl.dnnl_prop_kind_t = dnnl.dnnl_forward_training;
        // TODO: determine train mode/assess perf impact of always choosing training
        // (primitive cache storage overhead?)
        // const forward_mode: dnnl.dnnl_prop_kind_t = if (train) dnnl.dnnl_forward_training else dnnl.dnnl_forward_inference;
        var dnnl_engine = (try DnnlEngine.getInstance(allocator)).getEngine();
        try dnnl.DNNL_CHECK(
            dnnl.dnnl_convolution_forward_primitive_desc_create(
                &out.fwd_prim_desc,
                dnnl_engine,
                forward_mode,
                dnnl.dnnl_deconvolution_direct,
                out.input_mem_desc,
                out.weight_mem_desc,
                out.bias_mem_desc,
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
