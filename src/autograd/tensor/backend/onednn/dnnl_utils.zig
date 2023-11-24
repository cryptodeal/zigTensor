const std = @import("std");
const zt = @import("../../../../zt.zig");
const dnnl = @import("../../../../bindings/onednn/onednn.zig");
const build_options = @import("build_options");

const DType = zt.tensor.DType;
const Shape = zt.tensor.shape.Shape;
const Tensor = zt.tensor.Tensor;
const Dim = zt.tensor.shape.Dim;
const PoolingMode = zt.common.PoolingMode;
const DevicePtr = zt.common.DevicePtr;

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

    pub fn getInstance(allocator: std.mem.Allocator) !*DnnlStream {
        if (dnnlStreamSingleton == null) {
            dnnlStreamSingleton = try DnnlStream.init(allocator, (try DnnlEngine.getInstance(allocator)).getEngine());
        }
        return dnnlStreamSingleton.?;
    }

    pub fn freeInstance() void {
        if (dnnlStreamSingleton) |stream| {
            stream.deinit();
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
        if (dnnlEngineSingleton) |engine| {
            engine.deinit();
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

pub const DnnlMemoryWrapper = struct {
    allocator: std.mem.Allocator = undefined,
    dims_: []i64 = &[_]i64{},
    descriptor_: dnnl.dnnl_memory_desc_t = null,
    memory_: dnnl.dnnl_memory_t = null,
    device_ptr_: ?DevicePtr = null,

    pub fn init(allocator: std.mem.Allocator, tensor: Tensor, dims: []const i64, format: dnnl.dnnl_format_tag_t) !DnnlMemoryWrapper {
        var self: DnnlMemoryWrapper = .{
            .allocator = allocator,
            .dims_ = try allocator.alloc(i64, dims.len),
        };
        @memcpy(self.dims_, dims);
        var buffer: ?*anyopaque = null;
        if (build_options.ZT_BACKEND_OPENCL) {
            // TODO: implement
        } else {
            self.device_ptr_ = try DevicePtr.init(allocator, tensor);
            buffer = self.device_ptr_.?.get();
        }
        try dnnl.DNNL_CHECK(
            dnnl.dnnl_memory_desc_create_with_tag(&self.descriptor_, @intCast(self.dims_.len), self.dims_.ptr, try dnnlMapToType(try tensor.dtype(allocator)), format),
            @src(),
        );
        try dnnl.DNNL_CHECK(dnnl.dnnl_memory_create(&self.memory_, self.descriptor_, (try DnnlEngine.getInstance(allocator)).getEngine(), buffer), @src());
        return self;
    }

    pub fn deinitAll(self: *DnnlMemoryWrapper) void {
        if (self.dims_.len > 0) self.allocator.free(self.dims_);
        if (self.memory_ != null) dnnl.DNNL_CHECK(dnnl.dnnl_memory_destroy(self.memory_), @src()) catch unreachable;
        if (self.descriptor_ != null) dnnl.DNNL_CHECK(dnnl.dnnl_memory_desc_destroy(self.descriptor_), @src()) catch unreachable;
    }

    pub fn deinit(self: *DnnlMemoryWrapper) void {
        if (self.dims_.len > 0) self.allocator.free(self.dims_);
        if (self.device_ptr_ != null) self.device_ptr_.?.deinit();
    }

    pub fn assign(self: *DnnlMemoryWrapper, other: *DnnlMemoryWrapper) void {
        _ = other;
        _ = self;
        // TODO: implement
    }

    pub fn getMemory(self: *const DnnlMemoryWrapper) dnnl.dnnl_memory_t {
        return self.memory_;
    }

    pub fn getDescriptor(self: *const DnnlMemoryWrapper) dnnl.dnnl_memory_desc_t {
        return self.descriptor_;
    }
};

pub fn convertToDnnlDims(allocator: std.mem.Allocator, shape: Shape) ![]i64 {
    const dims = try allocator.alloc(i64, shape.len);
    @memcpy(dims, shape);
    return dims;
}

pub fn createArgs(allocator: std.mem.Allocator, args: []const std.meta.Tuple(&.{ c_int, dnnl.dnnl_memory_t }), nargs: ?usize) ![]dnnl.dnnl_exec_arg_t {
    var exec_args = try allocator.alloc(dnnl.dnnl_exec_arg_t, nargs orelse args.len);
    for (args, 0..) |a, i| {
        exec_args[i] = .{
            .arg = a[0],
            .memory = a[1],
        };
    }
    return exec_args;
}

pub fn dnnlAlignOrdering(
    allocator: std.mem.Allocator,
    net: *std.ArrayList(dnnl.dnnl_primitive_t),
    net_args: *std.ArrayList([]dnnl.dnnl_exec_arg_t),
    memory: dnnl.dnnl_memory_t,
    desc: dnnl.const_dnnl_memory_desc_t,
) !struct { bool, dnnl.dnnl_memory_t } {
    var memory_out = memory;
    var mem_out_desc: dnnl.const_dnnl_memory_desc_t = null;
    var allocated = false;
    // rewrite
    try dnnl.DNNL_CHECK(dnnl.dnnl_memory_get_memory_desc(memory, &mem_out_desc), @src());
    if (dnnl.dnnl_memory_desc_equal(mem_out_desc, desc) == 0) {
        // use the ordering requested by the descriptor
        try dnnl.DNNL_CHECK(dnnl.dnnl_memory_create(&memory_out, desc, (try DnnlEngine.getInstance(allocator)).getEngine(), dnnl.DNNL_MEMORY_ALLOCATE), @src());
        allocated = true;
        var reorder_pd: dnnl.dnnl_primitive_desc_t = null;
        try dnnl.DNNL_CHECK(dnnl.dnnl_reorder_primitive_desc_create(&reorder_pd, mem_out_desc, (try DnnlEngine.getInstance(allocator)).getEngine(), desc, (try DnnlEngine.getInstance(allocator)).getEngine(), null), @src());
        var reorder: dnnl.dnnl_primitive_t = null;
        try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_create(&reorder, reorder_pd), @src());
        try dnnl.DNNL_CHECK(dnnl.dnnl_primitive_desc_destroy(reorder_pd), @src());
        try net.append(reorder);
        try net_args.append(try createArgs(allocator, &.{ .{ dnnl.DNNL_ARG_FROM, memory }, .{ dnnl.DNNL_ARG_TO, memory_out } }, null));
    }
    return .{ allocated, memory_out };
}

pub fn executeNetwork(
    allocator: std.mem.Allocator,
    net: *std.ArrayList(dnnl.dnnl_primitive_t),
    net_args: *std.ArrayList([]dnnl.dnnl_exec_arg_t),
) !void {
    if (net.items.len != net_args.items.len) {
        std.debug.print("executeNetwork - given different size nets and net_args\n", .{});
        return error.FailedExecuteNetwork;
    }

    // TODO: {zt.Tensor} -- improve this to work with other backend interop
    // If on the CPU backend, there isn't a AF computation stream that facilitates
    // enforcing that inputs to computation are ready; we're required to wait
    // until all AF operations are done
    if (build_options.ZT_BACKEND_CPU) {
        try zt.tensor.sync(allocator);
    }

    for (0..net.items.len) |i| {
        try dnnl.DNNL_CHECK(
            dnnl.dnnl_primitive_execute(
                net.items[i],
                (try DnnlStream.getInstance(allocator)).getStream(),
                @intCast(net_args.items[i].len),
                net_args.items[i].ptr,
            ),
            @src(),
        );
    }

    try dnnl.DNNL_CHECK(dnnl.dnnl_stream_wait((try DnnlStream.getInstance(allocator)).getStream()), @src());

    // TODO: {zt.Tensor} -- improve this to work with other backend interop
    if (build_options.ZT_BACKEND_CPU) {
        // Block the executing thread until the work is complete
        try dnnl.DNNL_CHECK(dnnl.dnnl_stream_wait((try DnnlStream.getInstance(allocator)).getStream()), @src());
    }
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

test "DnnlUtils -> dnnlMapToPoolingMode" {
    try std.testing.expect(dnnlMapToPoolingMode(.Max) == dnnl.dnnl_pooling_max);
    try std.testing.expect(dnnlMapToPoolingMode(.AvgIncludePadding) == dnnl.dnnl_pooling_avg_include_padding);
    try std.testing.expect(dnnlMapToPoolingMode(.AvgExcludePadding) == dnnl.dnnl_pooling_avg_exclude_padding);
}

test "DnnlUtils -> dnnlMapToType" {
    try std.testing.expect(try dnnlMapToType(.f16) == dnnl.dnnl_f16);
    try std.testing.expect(try dnnlMapToType(.f32) == dnnl.dnnl_f32);
    try std.testing.expectError(error.DNNLInvalidDataType, dnnlMapToType(.f64));
}
