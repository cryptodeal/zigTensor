const std = @import("std");
const zt = @import("../../../zt.zig");
const dnnl = @import("../../../bindings/onednn/onednn.zig");
const build_options = @import("build_options");

const DType = zt.tensor.DType;
const Shape = zt.tensor.shape.Shape;
const Dim = zt.tensor.shape.Dim;
const PoolingMode = zt.common.PoolingMode;

var dnnlStreamSingleton: ?*DnnlStream = null;

pub const DnnlStream = struct {
    allocator: std.mem.Allocator,
    stream_: dnnl.dnnl_stream_t = null,

    pub fn init(allocator: std.mem.Allocator, engine: dnnl.dnnl_engine_t) !*DnnlStream {
        var self = try allocator.create(DnnlStream);
        self.* = .{ .allocator = allocator };
        if (build_options.ZT_BACKEND_OPENCL) {
            // TODO: self.stream_
        } else {
            try dnnl.DNNL_CHECK(dnnl.dnnl_stream_create(&self.stream_, engine, dnnl.dnnl_stream_default_flags), @src());
        }
        return self;
    }

    pub fn deinit(self: *DnnlStream) void {
        dnnl.DNNL_CHECK(dnnl.dnnl_stream_destroy(self.stream_), @src()) catch unreachable;
        self.allocator.destroy(self);
    }

    pub fn getInstance(allocator: std.mem.Allocator) !*DnnlEngine {
        if (dnnlStreamSingleton == null) {
            dnnlStreamSingleton = try DnnlStream.init(allocator, (try DnnlEngine.getInstance(allocator)).getEngine());
        }
        return dnnlEngineSingleton.?;
    }

    pub fn releaseInstance() void {
        if (dnnlStreamSingleton != null) {
            dnnlStreamSingleton.*.deinit();
            dnnlStreamSingleton = null;
        }
    }

    pub fn getStream(self: *const DnnlStream) dnnl.dnnl_stream_t {
        return self.stream_;
    }
};

var dnnlEngineSingleton: ?*DnnlEngine = null;

/// Struct that contains a static instance of a dnnl.dnnl_engine_t
pub const DnnlEngine = struct {
    allocator: std.mem.Allocator,
    engine_: dnnl.dnnl_engine_t = null,

    pub fn init(allocator: std.mem.Allocator) !*DnnlEngine {
        var self = try allocator.create(DnnlEngine);
        self.* = .{ .allocator = allocator };
        if (build_options.ZT_BACKEND_OPENCL) {
            // TODO:
        } else {
            try dnnl.DNNL_CHECK(dnnl.dnnl_engine_create(&self.engine_, dnnl.dnnl_cpu, 0), @src());
        }
        return self;
    }

    pub fn getInstance(allocator: std.mem.Allocator) !*DnnlEngine {
        if (dnnlEngineSingleton == null) {
            dnnlEngineSingleton = try DnnlEngine.init(allocator);
        }
        return dnnlEngineSingleton.?;
    }

    pub fn freeInstance() void {
        if (dnnlEngineSingleton != null) {
            dnnlEngineSingleton.*.deinit();
            dnnlEngineSingleton = null;
        }
    }

    pub fn deinit(self: *DnnlEngine) void {
        dnnl.DNNL_CHECK(dnnl.dnnl_engine_destroy(self.engine_), @src()) catch unreachable;
        self.allocator.destroy(self);
    }

    pub fn getEngine(self: *const DnnlEngine) dnnl.dnnl_engine_t {
        return self.engine_;
    }
};

pub fn convertToDnnlDims(allocator: std.mem.Allocator, shape: Shape) ![]i64 {
    var dims = try allocator.alloc(i64, shape.dims.len);
    @memcpy(dims, shape);
    return dims;
}

pub fn dnnlMapToPoolingMode(mode: PoolingMode) dnnl.dnnl_alg_kind_t {
    return switch (mode) {
        .Max => dnnl.dnnl_pooling_max,
        .AvgIncludePadding => dnnl.dnnl_pooling_avg_include_padding,
        .AvgExcludePadding => dnnl.dnnl_pooling_avg_exclude_padding,
    };
}

pub inline fn dnnlMapToType(t: DType) !dnnl.dnnl_data_type_t {
    return switch (t) {
        .f16 => dnnl.dnnl_f16,
        .f32 => dnnl.dnnl_f32,
        else => {
            std.debug.print("data type {s} is not supported by DNNL\n", .{@tagName(t)});
            return error.DNNLInvalidDataType;
        },
    };
}

test "DnnlUtils -> dnnlMapToType" {
    try std.testing.expect(try dnnlMapToType(.f16) == dnnl.dnnl_f16);
    try std.testing.expect(try dnnlMapToType(.f32) == dnnl.dnnl_f32);
    try std.testing.expectError(error.DNNLInvalidDataType, dnnlMapToType(.f64));
}
