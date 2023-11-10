const std = @import("std");
const zt = @import("../../../zt.zig");
const zigrc = @import("zigrc");
const dnnl = @import("../../../bindings/onednn/onednn.zig");
const AutogradPayload = @import("../autograd_extension.zig").AutogradPayload;
const conv2d = @import("conv2d.zig");
const dnnl_utils = @import("dnnl_utils.zig");

const dnnlMapToType = dnnl_utils.dnnlMapToType;
const DnnlEngine = dnnl_utils.DnnlEngine;
const OneDnnConv2DData = conv2d.OneDnnConv2DData;
const kWIdx = conv2d.kWIdx;
const kHIdx = conv2d.kHIdx;
const kWeightOutputChannelSizeIdx = conv2d.kWeightOutputChannelSizeIdx;
const kIOBatchSizeIdx = conv2d.kIOBatchSizeIdx;

const DType = zt.tensor.DType;
const Tensor = zt.tensor.Tensor;

pub const OneDnnAutogradExtension = struct {
    const Self = @This();

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) *Self {
        var self = try allocator.create(Self);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn isDataTypeSupported(dtype: DType) bool {
        // fp16 computation is not supported with onednn
        return dtype != .f16;
    }

    pub fn conv2d(
        self: *Self,
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
        _: zigrc.Arc(AutogradPayload),
    ) !Tensor {
        _ = self;
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
                1 + @divTrunc((try input.dim(allocator, kWIdx) + (2 * px) - (1 + (try weights.dim(allocator, kWIdx) - 1) * dx)), sx),
                1 + @divTrunc((try input.dim(allocator, kHIdx) + (2 * py) - (1 + (try weights.dim(allocator, kHIdx) - 1) * dy)), sy),
                try weights.dim(allocator, kWeightOutputChannelSizeIdx),
                try input.dim(allocator, kIOBatchSizeIdx),
            },
            try input.dtype(allocator),
        );
        const has_bias = try bias.elements(allocator) > 0;
        _ = has_bias;

        const data_type: dnnl.dnnl_data_type_t = try dnnlMapToType(try input.dtype(allocator));
        _ = data_type;
        const format_weight: dnnl.dnnl_format_tag_t = if (groups == 1) dnnl.dnnl_abcd else dnnl.dnnl_abcde;
        _ = format_weight;
        var dnnl_engine = (try DnnlEngine.getInstance(allocator)).getEngine();
        _ = dnnl_engine;

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

    }
};
