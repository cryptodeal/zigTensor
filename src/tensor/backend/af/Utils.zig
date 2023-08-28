const std = @import("std");
const af = @import("../../../backends/ArrayFire.zig");
const tensor_base = @import("../../TensorBase.zig");
const zt_shape = @import("../../Shape.zig");
const zt_index = @import("../../Index.zig");

const IndexType = zt_index.IndexType;
const Shape = zt_shape.Shape;
const Dim = zt_shape.Dim;
const DType = @import("../../Types.zig").DType;
const MatrixProperty = tensor_base.MatrixProperty;
const StorageType = tensor_base.StorageType;
const SortMode = tensor_base.SortMode;
const Location = tensor_base.Location;
const PadType = tensor_base.PadType;

pub const AfDims4 = struct {
    dims: [4]af.dim_t = [_]af.dim_t{1} ** 4,

    pub fn init(dims: ?[4]af.dim_t) AfDims4 {
        var self: AfDims4 = .{};
        if (dims != null) {
            self.dims = dims.?;
        }
        return self;
    }

    pub fn elements(self: AfDims4) usize {
        return @intCast(self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3]);
    }
};

pub fn AF_CHECK(v: af.af_err, src: std.builtin.SourceLocation) !void {
    if (v != af.AF_SUCCESS) {
        var err_string: [*c]const u8 = af.af_err_to_string(v);
        std.debug.print("ArrayFire error: {s}:{d} - {s}\n", .{ src.file, src.line, std.mem.span(err_string) });

        return error.ArrayFireError;
    }
}

pub const AfDType = enum(af.af_dtype) {
    f32,
    c32,
    f64,
    c64,
    b8,
    s32,
    u32,
    u8,
    s64,
    u64,
    s16,
    u16,
    f16,

    pub fn value(self: *AfDType) af.af_dtype {
        return @intFromEnum(self);
    }
};

pub fn ztToAfType(data_type: DType) AfDType {
    return switch (data_type) {
        .f16 => AfDType.f16,
        .f32 => AfDType.f32,
        .f64 => AfDType.f64,
        .b8 => AfDType.b8,
        .s16 => AfDType.s16,
        .s32 => AfDType.s32,
        .s64 => AfDType.s64,
        .u8 => AfDType.u8,
        .u16 => AfDType.u16,
        .u32 => AfDType.u32,
        .u64 => AfDType.u64,
    };
}

pub fn afToZtType(data_type: AfDType) !DType {
    return switch (data_type) {
        .f16 => DType.f16,
        .f32 => DType.f32,
        .f64 => DType.f64,
        .b8 => DType.b8,
        .s16 => DType.s16,
        .s32 => DType.s32,
        .s64 => DType.s64,
        .u8 => DType.u8,
        .u16 => DType.u16,
        .u32 => DType.u32,
        .u64 => DType.u64,
        else => return error.UnsupportedArrayFireType,
    };
}

pub fn ztToAfMatrixProperty(property: MatrixProperty) af.af_mat_prop {
    return switch (property) {
        .None => af.AF_MAT_NONE,
        .Transpose => af.AF_MAT_TRANS,
    };
}

pub fn ztToAfStorageType(storage_type: StorageType) af.af_storage {
    return switch (storage_type) {
        .Dense => af.AF_STORAGE_DENSE,
        .CSR => af.AF_STORAGE_CSR,
        .CSC => af.AF_STORAGE_CSC,
        .COO => af.AF_STORAGE_COO,
    };
}

pub fn ztToAfTopKSortMode(sort_mode: SortMode) af.af_topk_function {
    return switch (sort_mode) {
        .Descending => af.AF_TOPK_MAX,
        .Ascending => af.AF_TOPK_MIN,
    };
}

pub fn ztToAfDims(shape: *const Shape) !AfDims4 {
    if (shape.ndim() > 4) {
        std.log.err("ztToAfDims: ArrayFire shapes can't be more than 4 dimensions\n", .{});
        return error.ArrayFireCannotExceed4Dimensions;
    }
    var af_dims4 = AfDims4.init([_]af.dim_t{1} ** 4);
    for (0..shape.ndim()) |i| {
        af_dims4.dims[i] = @intCast(try shape.dim(i));
    }
    return af_dims4;
}

pub fn afToZtDimsRaw(d: AfDims4, num_dims: usize, s: *Shape) !void {
    if (num_dims > @as(usize, @intCast(af.AF_MAX_DIMS))) {
        std.log.err("afToZtDims: num_dims ({d}) > af.AF_MAX_DIMS ({d} )", .{ num_dims, af.AF_MAX_DIMS });
    }
    var storage = s.get();

    // num_dims constraint is enforced by the internal API per condenseDims
    if (num_dims == 1 and d.elements() == 0) {
        // Empty tensor
        try storage.resize(1);
        s.dims_.items[0] = 0;
        return;
    }

    // num_dims == 0 --> scalar tensor
    if (num_dims == 0) {
        try storage.resize(0);
        return;
    }

    try storage.resize(num_dims);
    for (0..num_dims) |i| {
        storage.items[i] = @intCast(d.dims[i]);
    }
}

pub fn afToZtDims(allocator: std.mem.Allocator, d: AfDims4, num_dims: usize) !Shape {
    var shape = try Shape.initRaw(allocator);
    try afToZtDimsRaw(d, num_dims, &shape);
    return shape;
}

// TODO: ztRangeToAfSeq
// pub fn ztRangeToAfSeq(range: *Range) af.af_seq

// TODO: ztToAfIndex
// pub fn ztToAfIndex(idx: *Index) af.index

pub fn condenseDims(dims: AfDims4) AfDims4 {
    if (dims.elements() == 0) {
        return AfDims4.init([_]af.dim_t{ 0, 1, 1, 1 });
    }

    // Find the condensed shape
    var new_dims = AfDims4.init([_]af.dim_t{ 1, 1, 1, 1 });
    var new_dim_idx: usize = 0;
    for (0..af.AF_MAX_DIMS) |i| {
        if (dims.dims[i] != 1) {
            // found a non-1 dim size - populate new_dims
            new_dims.dims[new_dim_idx] = dims.dims[i];
            new_dim_idx += 1;
        }
    }
    return new_dims;
}

// TODO: condenseIndices
pub fn condenseIndices(arr: af.af_array, keep_dims: bool, index_types: ?*std.ArrayList(IndexType), is_flat: bool) af.af_array {
    _ = index_types;
    _ = is_flat;
    // Fast path - return the Array as is if keepDims - don't consolidate
    if (keep_dims) {
        return arr;
    }

    // Fast path - Array has zero elements or a dim of size zero
    var elements: af.dim_t = undefined;
    try AF_CHECK(af.af_get_elements(&elements, arr), @src());
    if (elements == 0) {
        return arr;
    }
}

pub fn ztToAfLocation(location: Location) af.af_source {
    return switch (location) {
        .Host => af.afHost,
        .Device => af.afDevice,
    };
}

pub fn fromZtData(shape: *const Shape, ptr: ?*const anyopaque, dtype: DType) !af.af_array {
    var dims = try ztToAfDims(shape);
    var af_dtype = ztToAfType(dtype);
    var res: af.af_array = undefined;
    switch (dtype) {
        .f32, .f64, .s32, .u32, .s64, .u64, .s16, .u16, .b8, .u8 => try AF_CHECK(af.af_create_array(
            &res,
            ptr,
            @intCast(shape.ndim()),
            &dims.dims,
            @intFromEnum(af_dtype),
        ), @src()),
        else => {
            std.log.err("fromZtData: can't construct ArrayFire array from given type.\n", .{});
            return error.UnsupportedArrayFireType;
        },
    }
    return res;
}

pub fn ztToAfPadType(pad_type: PadType) af.af_border_type {
    return switch (pad_type) {
        .Constant => af.AF_PAD_ZERO,
        .Edge => af.AF_PAD_CLAMP_TO_EDGE,
        .Symmetric => af.AF_PAD_SYM,
    };
}

pub fn createAfIndexers() ![]af.af_index_t {
    var indexers: [*c]af.af_index_t = undefined;
    try AF_CHECK(af.af_create_indexers(&indexers), @src());
    return indexers[0..3];
}

test "af.af_create_array" {
    const allocator = std.testing.allocator;
    var data = try allocator.alloc(f32, 100);
    defer allocator.free(data);
    @memset(data, 100);
    var dim = [_]Dim{100};
    var shape = try Shape.init(allocator, &dim);
    defer shape.deinit();

    var arr_ptr = try fromZtData(&shape, data.ptr, DType.f32);
    try AF_CHECK(af.af_release_array(arr_ptr), @src());
}

test "af.af_create_indexers" {
    var indexers = try createAfIndexers();
    try AF_CHECK(af.af_release_indexers(indexers.ptr), @src());
}

test "ztToAfType" {
    try std.testing.expect(ztToAfType(.f16) == AfDType.f16);
    try std.testing.expect(ztToAfType(.f32) == AfDType.f32);
    try std.testing.expect(ztToAfType(.f64) == AfDType.f64);
    try std.testing.expect(ztToAfType(.b8) == AfDType.b8);
    try std.testing.expect(ztToAfType(.s16) == AfDType.s16);
    try std.testing.expect(ztToAfType(.s32) == AfDType.s32);
    try std.testing.expect(ztToAfType(.s64) == AfDType.s64);
    try std.testing.expect(ztToAfType(.u8) == AfDType.u8);
    try std.testing.expect(ztToAfType(.u16) == AfDType.u16);
    try std.testing.expect(ztToAfType(.u32) == AfDType.u32);
    try std.testing.expect(ztToAfType(.u64) == AfDType.u64);
}

test "afToZtType" {
    try std.testing.expect(try afToZtType(.f16) == DType.f16);
    try std.testing.expect(try afToZtType(.f32) == DType.f32);
    try std.testing.expect(try afToZtType(.f64) == DType.f64);
    try std.testing.expect(try afToZtType(.b8) == DType.b8);
    try std.testing.expect(try afToZtType(.s16) == DType.s16);
    try std.testing.expect(try afToZtType(.s32) == DType.s32);
    try std.testing.expect(try afToZtType(.s64) == DType.s64);
    try std.testing.expect(try afToZtType(.u8) == DType.u8);
    try std.testing.expect(try afToZtType(.u16) == DType.u16);
    try std.testing.expect(try afToZtType(.u32) == DType.u32);
    try std.testing.expect(try afToZtType(.u64) == DType.u64);
}

test "ztToAfMatrixProperty" {
    try std.testing.expect(ztToAfMatrixProperty(.None) == af.AF_MAT_NONE);
    try std.testing.expect(ztToAfMatrixProperty(.Transpose) == af.AF_MAT_TRANS);
}

test "ztToAfStorageType" {
    try std.testing.expect(ztToAfStorageType(.Dense) == af.AF_STORAGE_DENSE);
    try std.testing.expect(ztToAfStorageType(.CSR) == af.AF_STORAGE_CSR);
    try std.testing.expect(ztToAfStorageType(.CSC) == af.AF_STORAGE_CSC);
    try std.testing.expect(ztToAfStorageType(.COO) == af.AF_STORAGE_COO);
}

test "ztToAfTopKSortMode" {
    try std.testing.expect(ztToAfTopKSortMode(.Descending) == af.AF_TOPK_MAX);
    try std.testing.expect(ztToAfTopKSortMode(.Ascending) == af.AF_TOPK_MIN);
}

test "ztToAfDims" {
    const allocator = std.testing.allocator;

    var dims1 = [_]Dim{2};
    var shape = try Shape.init(allocator, &dims1);
    var res1 = try ztToAfDims(&shape);
    var exp1 = [_]af.dim_t{ 2, 1, 1, 1 };
    try std.testing.expectEqualSlices(af.dim_t, &exp1, &res1.dims);
    shape.deinit();

    var dims2 = [_]Dim{ 2, 3 };
    shape = try Shape.init(allocator, &dims2);
    var res2 = try ztToAfDims(&shape);
    var exp2 = [_]af.dim_t{ 2, 3, 1, 1 };
    try std.testing.expectEqualSlices(af.dim_t, &exp2, &res2.dims);
    shape.deinit();

    var dims3 = [_]Dim{ 2, 3, 4 };
    shape = try Shape.init(allocator, &dims3);
    var res3 = try ztToAfDims(&shape);
    var exp3 = [_]af.dim_t{ 2, 3, 4, 1 };
    try std.testing.expectEqualSlices(af.dim_t, &exp3, &res3.dims);
    shape.deinit();

    var dims4 = [_]Dim{ 2, 3, 4, 5 };
    shape = try Shape.init(allocator, &dims4);
    var res4 = try ztToAfDims(&shape);
    var exp4 = [_]af.dim_t{ 2, 3, 4, 5 };
    try std.testing.expectEqualSlices(af.dim_t, &exp4, &res4.dims);
    shape.deinit();
}

test "afToZtDims" {
    const allocator = std.testing.allocator;

    var dims1 = AfDims4.init([_]af.dim_t{ 2, 1, 1, 1 });
    var shape = try afToZtDims(allocator, dims1, 1);
    var exp1 = [_]Dim{2};
    try std.testing.expectEqualSlices(Dim, &exp1, shape.dims_.items);
    shape.deinit();

    var dims2 = AfDims4.init([_]af.dim_t{ 2, 3, 1, 1 });
    shape = try afToZtDims(allocator, dims2, 2);
    var exp2 = [_]Dim{ 2, 3 };
    try std.testing.expectEqualSlices(Dim, &exp2, shape.dims_.items);
    shape.deinit();

    var dims3 = AfDims4.init([_]af.dim_t{ 2, 3, 4, 1 });
    shape = try afToZtDims(allocator, dims3, 3);
    var exp3 = [_]Dim{ 2, 3, 4 };
    try std.testing.expectEqualSlices(Dim, &exp3, shape.dims_.items);
    shape.deinit();

    var dims4 = AfDims4.init([_]af.dim_t{ 2, 3, 4, 5 });
    shape = try afToZtDims(allocator, dims4, 4);
    var exp4 = [_]Dim{ 2, 3, 4, 5 };
    try std.testing.expectEqualSlices(Dim, &exp4, shape.dims_.items);
    shape.deinit();
}

test "ztToAfLocation" {
    try std.testing.expect(ztToAfLocation(.Host) == af.afHost);
    try std.testing.expect(ztToAfLocation(.Device) == af.afDevice);
}

test "ztToAfPadType" {
    try std.testing.expect(ztToAfPadType(.Constant) == af.AF_PAD_ZERO);
    try std.testing.expect(ztToAfPadType(.Edge) == af.AF_PAD_CLAMP_TO_EDGE);
    try std.testing.expect(ztToAfPadType(.Symmetric) == af.AF_PAD_SYM);
}
