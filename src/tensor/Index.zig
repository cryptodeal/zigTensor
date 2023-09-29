const std = @import("std");
const tensor_ = @import("tensor.zig");

const Dim = tensor_.Dim;
const Tensor = tensor_.Tensor;

/// Represents the imaginary index after the last index along
/// an axis of a tensor. This special case is used because
/// `Range.end` is exclusive.
pub const end_t = void;

/// A static instance of `end_t` for convenience; e.g. can be used
/// for `Range(0, end)` to index all elements along a certain axis.
pub const end: end_t = {};

/// Allowed indexing operators.
pub const IndexType = enum(u8) { Tensor = 0, Range = 1, Literal = 2, Span = 3 };

pub const RangeIdxTypeTag = enum(u8) { end, dim };

pub const RangeError = error{FailedToGetEndVal};

/// An entity representing a contiguous or strided sequence of indices.
///
/// Assuming an axis has N elements, this is the mapping from negative to
/// positive indices:
///
/// -------------------------
/// | -N | -N+1 | ... |  -1 |
/// -------------------------
/// |  0 |    1 | ... | N-1 |
/// -------------------------
pub const Range = struct {
    pub const index = union(RangeIdxTypeTag) { end: end_t, dim: Dim };

    pub const kDefaultStride: Dim = 1;

    start_: Dim = 0,
    /// end is exclusive; null means including the last element.
    end_: ?Dim = null,

    stride_: Dim = kDefaultStride,

    /// Initializes a Range with the indices [0, idx) (i.e. [0, idx - 1]).
    pub fn initEnd(idx: Dim) Range {
        return Range.init(0, .{ .dim = idx });
    }

    /// Initializes a range with the indices [start_idx, end_idx) (i.e. [start_idx, end_idx - 1]).
    pub fn init(start_idx: Dim, end_idx: index) Range {
        return Range.initWithStride(start_idx, end_idx, kDefaultStride);
    }

    /// Construct a range with the indices [start, end) (i.e. [start, end - 1])
    /// with the given stride.
    pub fn initWithStride(start_idx: Dim, end_idx: index, stride_val: Dim) Range {
        var self = Range{
            .start_ = start_idx,
            .stride_ = stride_val,
        };
        if (@as(RangeIdxTypeTag, end_idx) == .dim) {
            self.end_ = end_idx.dim;
        }
        return self;
    }

    pub fn start(self: *const Range) Dim {
        return self.start_;
    }

    pub fn end(self: *const Range) ?Dim {
        return self.end_;
    }

    pub fn endVal(self: *const Range) !Dim {
        if (self.end_ != null) {
            return self.end_.?;
        }
        std.log.debug("[Range.endVal] end is `end_t`\n", .{});
        return RangeError.FailedToGetEndVal;
    }

    pub fn stride(self: *const Range) Dim {
        return self.stride_;
    }

    pub fn eql(self: *const Range, other: *const Range) bool {
        return std.meta.eql(self.*, other.*);
    }
};

pub const span: Range = Range.initWithStride(-1, .{ .dim = -1 }, 0);

pub const IndexErrors = error{IndexGetTypeMismatch};

pub const IndexTypeTag = enum(u8) { Literal, Range, Tensor };

/// An entity used to index a tensor.
///
/// An index can be of a few different types, which are
/// implicitly converted via Index's constructors:
///
/// - zt.Tensor if doing advanced indexing, where elements
/// from a tensor are indexed based on values in the
/// indexing tensor.
///
/// - zt.Range, which refers to a contiguous (or strided)
/// sequence of indices.
///
/// - An index literal, which refers to a single subtensor
/// of the tensor being indexed.
pub const Index = struct {
    pub const IndexVariant = union { Literal: Dim, Range: Range, Tensor: Tensor };

    /// The type of Indexing operator.
    type_: IndexType,

    /// Underlying data referred to by the index.
    index_: IndexVariant,

    fn initTensor(tensor: Tensor) Index {
        return .{
            .type_ = .Tensor,
            .index_ = .{ .Tensor = tensor },
        };
    }

    pub fn initRange(range: Range) Index {
        return .{
            .type_ = if (std.meta.eql(range, span)) .Span else .Range,
            .index_ = .{ .Range = range },
        };
    }

    pub fn initDim(idx: Dim) Index {
        return .{
            .type_ = .Literal,
            .index_ = .{ .Literal = idx },
        };
    }

    pub fn initCopy(other: Index) Index {
        return .{
            .type_ = other.type_,
            .index_ = other.index_,
        };
    }

    /// Returns the index type for this index.
    pub fn idxType(self: *const Index) IndexType {
        return self.type_;
    }

    pub fn isSpan(self: *const Index) bool {
        return self.type_ == .Span;
    }

    pub fn get(self: *Index, comptime T: type) !T {
        switch (self.type_) {
            .Literal => {
                if (T == Dim) {
                    return self.index_.Literal;
                }
            },
            .Range, .Span => {
                if (T == Range) {
                    return self.index_.Range;
                }
            },
            .Tensor => {
                if (T == Tensor) {
                    return self.index_.Tensor;
                }
            },
        }
        std.log.debug("[Index.get] type `{s}` doesn't match IndexTypeTag `{s}`\n", .{ @typeName(T), @tagName(self.type_) });
        return IndexErrors.IndexGetTypeMismatch;
    }

    pub fn getVariant(self: *Index) IndexVariant {
        return self.index_;
    }
};

test "IndexTest -> Range" {
    var s1 = Range.initEnd(3);
    try std.testing.expect(s1.start() == 0);
    try std.testing.expect(try s1.endVal() == 3);
    try std.testing.expect(s1.stride() == 1);

    var s2 = Range.init(4, .{ .dim = 5 });
    try std.testing.expect(s2.start() == 4);
    try std.testing.expect(try s2.endVal() == 5);
    try std.testing.expect(s2.stride() == 1);

    var s3 = Range.initWithStride(7, .{ .dim = 8 }, 9);
    try std.testing.expect(s3.stride() == 9);

    var s4 = Range.initWithStride(1, .{ .end = end }, 2);
    try std.testing.expect(s4.start() == 1);
    try std.testing.expect(s4.end() == null);
    try std.testing.expect(s4.stride() == 2);
}

test "IndexTest -> Range.eql" {
    var r1 = Range.initEnd(4);
    var r2 = Range.initEnd(4);
    try std.testing.expect(r1.eql(&r2));

    r1 = Range.init(2, .{ .dim = 3 });
    r2 = Range.init(2, .{ .dim = 3 });
    try std.testing.expect(r1.eql(&r2));

    r1 = Range.initWithStride(5, .{ .dim = 6 }, 7);
    r2 = Range.initWithStride(5, .{ .dim = 6 }, 7);
    try std.testing.expect(r1.eql(&r2));

    r1 = Range.initWithStride(8, .{ .end = end }, 9);
    r2 = Range.initWithStride(8, .{ .end = end }, 9);
    try std.testing.expect(r1.eql(&r2));
}

test "IndexTest -> Index.idxType" {
    const idx1 = Index.initDim(1);
    try std.testing.expect(idx1.idxType() == .Literal);

    const idx2 = Index.initRange(Range.initEnd(3));
    try std.testing.expect(idx2.idxType() == .Range);

    const idx3 = Index.initRange(span);
    try std.testing.expect(idx3.idxType() == .Span);
    try std.testing.expect(idx3.isSpan());
}

test "IndexTest -> ArrayFireMaxIndex" {
    const full = tensor_.full;
    const Shape = @import("Shape.zig").Shape;
    const deinit = @import("Init.zig").deinit;
    defer deinit(); // deinit global singletons

    const allocator = std.testing.allocator;
    var dims = [_]Dim{ 2, 3, 4, 5 };
    var shape = try Shape.init(allocator, &dims);
    defer shape.deinit();
    var t = try full(allocator, &shape, f64, 6, .f32);
    defer t.deinit();
    if (t.backendType() != .ArrayFire) {
        return error.SkipZigTest;
    }
    var indices = [_]Index{ Index.initDim(1), Index.initDim(2), Index.initDim(3), Index.initDim(4), Index.initDim(5) };
    try std.testing.expectError(
        error.IndicesExceedMaxDims,
        t.index(allocator, &indices),
    );
}

test "IndexTest -> Shape" {
    const full = tensor_.full;
    const Shape = tensor_.Shape;
    const deinit = @import("Init.zig").deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var dims = [_]Dim{ 4, 4 };
    var shape = try Shape.init(allocator, &dims);
    defer shape.deinit();
    var t = try full(allocator, &shape, f64, 3, .f32);
    defer t.deinit();

    var indices1 = [_]Index{ Index.initDim(2), Index.initDim(2) };
    var res1 = try t.index(allocator, &indices1);
    defer res1.deinit();
    var res1_dims = [_]Dim{1};
    var res1_shape = try Shape.init(allocator, &res1_dims);
    defer res1_shape.deinit();
    var real_shape = try res1.shape(allocator);
    try std.testing.expect(real_shape.eql(&res1_shape));

    var indices2 = [_]Index{ Index.initDim(2), Index.initRange(span) };
    var res2 = try t.index(allocator, &indices2);
    defer res2.deinit();
    var res2_dims = [_]Dim{4};
    var res2_shape = try Shape.init(allocator, &res2_dims);
    defer res2_shape.deinit();
    real_shape = try res2.shape(allocator);
    try std.testing.expect(real_shape.eql(&res2_shape));

    var indices3 = [1]Index{Index.initDim(2)};
    var res3 = try t.index(allocator, &indices3);
    defer res3.deinit();
    var res3_dims = [_]Dim{4};
    var res3_shape = try Shape.init(allocator, &res3_dims);
    defer res3_shape.deinit();
    real_shape = try res3.shape(allocator);
    try std.testing.expect(real_shape.eql(&res3_shape));

    var indices4 = [1]Index{Index.initRange(Range.initEnd(3))};
    var res4 = try t.index(allocator, &indices4);
    defer res4.deinit();
    var res4_dims = [_]Dim{ 3, 4 };
    var res4_shape = try Shape.init(allocator, &res4_dims);
    defer res4_shape.deinit();
    real_shape = try res4.shape(allocator);
    try std.testing.expect(real_shape.eql(&res4_shape));

    var indices5 = [1]Index{Index.initRange(Range.init(1, .{ .dim = 2 }))};
    var res5 = try t.index(allocator, &indices5);
    defer res5.deinit();
    var res5_dims = [_]Dim{ 1, 4 };
    var res5_shape = try Shape.init(allocator, &res5_dims);
    defer res5_shape.deinit();
    real_shape = try res5.shape(allocator);
    try std.testing.expect(real_shape.eql(&res5_shape));

    var indices6 = [2]Index{
        Index.initRange(Range.init(1, .{ .dim = 2 })),
        Index.initRange(Range.init(1, .{ .dim = 2 })),
    };
    var res6 = try t.index(allocator, &indices6);
    defer res6.deinit();
    var res6_dims = [_]Dim{ 1, 1 };
    var res6_shape = try Shape.init(allocator, &res6_dims);
    defer res6_shape.deinit();
    real_shape = try res6.shape(allocator);
    try std.testing.expect(real_shape.eql(&res6_shape));

    var indices7 = [1]Index{Index.initRange(Range.init(0, .{ .end = end }))};
    var res7 = try t.index(allocator, &indices7);
    defer res7.deinit();
    var res7_dims = [_]Dim{ 4, 4 };
    var res7_shape = try Shape.init(allocator, &res7_dims);
    defer res7_shape.deinit();
    real_shape = try res7.shape(allocator);
    try std.testing.expect(real_shape.eql(&res7_shape));

    var indices8 = [1]Index{Index.initRange(Range.initWithStride(0, .{ .end = end }, 2))};
    var res8 = try t.index(allocator, &indices8);
    defer res8.deinit();
    var res8_dims = [_]Dim{ 2, 4 };
    var res8_shape = try Shape.init(allocator, &res8_dims);
    defer res8_shape.deinit();
    real_shape = try res8.shape(allocator);
    try std.testing.expect(real_shape.eql(&res8_shape));

    var t1_dims = [_]Dim{ 5, 6, 7, 8 };
    var t1_shape = try Shape.init(allocator, &t1_dims);
    defer t1_shape.deinit();
    var t1 = try full(allocator, &t1_shape, f64, 3, .f32);
    defer t1.deinit();

    var indices9 = [_]Index{ Index.initDim(2), Index.initRange(Range.init(2, .{ .dim = 4 })), Index.initRange(span), Index.initDim(3) };
    var t1_res = try t1.index(allocator, &indices9);
    defer t1_res.deinit();
    var res9_dims = [_]Dim{ 2, 7 };
    var res9_shape = try Shape.init(allocator, &res9_dims);
    defer res9_shape.deinit();
    real_shape = try t1_res.shape(allocator);
    try std.testing.expect(real_shape.eql(&res9_shape));

    var indices10 = [_]Index{ Index.initRange(span), Index.initDim(3), Index.initRange(span), Index.initRange(span) };
    var t1_res2 = try t1.index(allocator, &indices10);
    defer t1_res2.deinit();
    var res10_dims = [_]Dim{ 5, 7, 8 };
    var res10_shape = try Shape.init(allocator, &res10_dims);
    defer res10_shape.deinit();
    real_shape = try t1_res2.shape(allocator);
    try std.testing.expect(real_shape.eql(&res10_shape));

    var indices11 = [_]Index{ Index.initRange(span), Index.initRange(Range.init(1, .{ .dim = 2 })), Index.initRange(span), Index.initRange(span) };
    var t1_res3 = try t1.index(allocator, &indices11);
    defer t1_res3.deinit();
    var res11_dims = [_]Dim{ 5, 1, 7, 8 };
    var res11_shape = try Shape.init(allocator, &res11_dims);
    defer res11_shape.deinit();
    real_shape = try t1_res3.shape(allocator);
    try std.testing.expect(real_shape.eql(&res11_shape));
}

test "IndexTest -> IndexAssignment" {
    const full = tensor_.full;
    const Shape = tensor_.Shape;
    const allClose = tensor_.allClose;
    const deinit = tensor_.deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var t_dims = [_]Dim{ 4, 4 };
    var t_shape = try Shape.init(allocator, &t_dims);
    defer t_shape.deinit();
    var t = try full(allocator, &t_shape, f64, 0, .s32);
    defer t.deinit();

    var indices = [2]Index{ Index.initRange(span), Index.initDim(0) };
    try t.indexAssign(allocator, i32, 1, &indices);
    indices = [2]Index{ Index.initRange(span), Index.initDim(1) };
    try t.indexAdd(allocator, i32, 1, &indices);
    indices = [2]Index{ Index.initRange(span), Index.initRange(Range.init(2, .{ .end = end })) };
    try t.indexAdd(allocator, i32, 1, &indices);
    indices = [2]Index{ Index.initRange(span), Index.initRange(span) };
    try t.indexMul(allocator, i32, 7, &indices);
    try t.inPlaceDiv(allocator, i32, 7);
    var expected = try full(allocator, &t_shape, f64, 1, .s32);
    defer expected.deinit();
    try std.testing.expect(try allClose(allocator, t, expected, 1));

    var a_dims = [_]Dim{ 6, 6 };
    var a_shape = try Shape.init(allocator, &a_dims);
    defer a_shape.deinit();
    var a = try full(allocator, &a_shape, f64, 0, .f32);
    defer a.deinit();
    indices = [2]Index{ Index.initDim(3), Index.initDim(4) };
    try a.indexAssign(allocator, f64, 4, &indices);
    var expected2_dims = [_]Dim{1};
    var expected2_shape = try Shape.init(allocator, &expected2_dims);
    defer expected2_shape.deinit();
    var expected2 = try full(allocator, &expected2_shape, f64, 4, .f32);
    defer expected2.deinit();
    var a_idx = try a.index(allocator, &indices);
    defer a_idx.deinit();
    try std.testing.expect(try allClose(allocator, a_idx, expected2, 1e-5));

    var a_assign_dims = [_]Dim{6};
    var a_assign_shape = try Shape.init(allocator, &a_assign_dims);
    defer a_assign_shape.deinit();
    var a_assign_val = try full(allocator, &a_assign_shape, f64, 8, .f32);
    defer a_assign_val.deinit();
    var indices1 = [1]Index{Index.initDim(2)};
    try a.indexAssign(allocator, Tensor, a_assign_val, &indices1);
    var a_idx2 = try a.index(allocator, &indices1);
    defer a_idx2.deinit();
    try std.testing.expect(try allClose(allocator, a_idx2, a_assign_val, 1e-5));

    var b_dims = [_]Dim{ 3, 3 };
    var b_shape = try Shape.init(allocator, &b_dims);
    defer b_shape.deinit();
    var b = try full(allocator, &b_shape, f64, 1, .f32);
    defer b.deinit();

    // TODO: implement fn so `var c = b;` works (initAssign??)

    // TODO: finish this test
}

// TODO: test "IndexTest -> IndexInPlaceOps" {}

test "IndexTest -> flat" {
    const rand = tensor_.rand;
    const full = tensor_.full;
    const Shape = tensor_.Shape;
    const allClose = tensor_.allClose;
    const deinit = tensor_.deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var m_dims = [_]Dim{ 4, 6 };
    var m_shape = try Shape.init(allocator, &m_dims);
    defer m_shape.deinit();
    var m = try rand(allocator, &m_shape, .f32);
    defer m.deinit();
    for (0..@intCast(try m.elements(allocator))) |i| {
        var m_flat = try m.flat(allocator, Index.initDim(@intCast(i)));
        defer m_flat.deinit();
        var indices = [_]Index{
            Index.initDim(@intCast(@rem(i, 4))),
            Index.initDim(@intCast(@divTrunc(i, 4))),
        };
        var m_indexed = try m.index(allocator, &indices);
        defer m_indexed.deinit();
        try std.testing.expect(try allClose(allocator, m_flat, m_indexed, 1e-5));
    }

    var n_dims = [_]Dim{ 4, 6, 8 };
    var n_shape = try Shape.init(allocator, &n_dims);
    defer n_shape.deinit();
    var n = try rand(allocator, &n_shape, .f32);
    defer n.deinit();
    for (0..@intCast(try n.elements(allocator))) |i| {
        var n_flat = try n.flat(allocator, Index.initDim(@intCast(i)));
        defer n_flat.deinit();
        var indices = [_]Index{
            Index.initDim(@intCast(@rem(i, 4))),
            Index.initDim(@intCast(@mod(@divTrunc(i, 4), 6))),
            Index.initDim(@intCast(@mod(@divTrunc(i, 4 * 6), 8))),
        };
        var n_indexed = try n.index(allocator, &indices);
        defer n_indexed.deinit();
        try std.testing.expect(try allClose(allocator, n_flat, n_indexed, 1e-5));
    }

    var a_dims = [_]Dim{ 5, 6, 7, 8 };
    var a_shape = try Shape.init(allocator, &a_dims);
    defer a_shape.deinit();
    var a = try full(allocator, &a_shape, f64, 9, .f32);
    defer a.deinit();
    var test_indices = [_]Dim{ 0, 1, 4, 11, 62, 104, 288 };
    for (test_indices) |i| {
        var a_flat = try a.flat(allocator, Index.initDim(i));
        defer a_flat.deinit();
        try std.testing.expect(try a_flat.scalar(allocator, f32) == 9);
    }

    // TODO finish test w assignment
}

test "IndexTest -> TensorIndex" {
    const full = tensor_.full;
    const Shape = tensor_.Shape;
    const deinit = tensor_.deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var idxs = [_]Dim{ 0, 1, 4, 9, 11, 13, 16, 91 };
    var size = idxs.len;
    var indice_dims = [_]Dim{@intCast(size)};
    var indices_shape = try Shape.init(allocator, &indice_dims);
    defer indices_shape.deinit();
    var indices = try full(allocator, &indices_shape, f64, 0, .f32);
    defer indices.deinit();
    for (0..idxs.len) |i| {
        var idx = [_]Index{Index.initDim(@intCast(i))};
        try indices.indexAssign(allocator, Dim, idxs[i], &idx);
    }
}

// TODO: test "IndexTest -> ExpressionIndex" {}
