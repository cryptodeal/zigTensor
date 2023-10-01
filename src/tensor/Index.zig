const std = @import("std");
const tensor_ = @import("tensor.zig");

const Dim = tensor_.shape.Dim;
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
        std.log.debug(
            "[Index.get] type `{s}` doesn't match IndexTypeTag `{s}`\n",
            .{ @typeName(T), @tagName(self.type_) },
        );
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
    const deinit = @import("Init.zig").deinit;
    defer deinit(); // deinit global singletons
    const allocator = std.testing.allocator;

    var t = try full(allocator, &.{ 2, 3, 4, 5 }, f64, 6, .f32);
    defer t.deinit();
    if (t.backendType() != .ArrayFire) {
        return error.SkipZigTest;
    }
    try std.testing.expectError(
        error.IndicesExceedMaxDims,
        t.index(
            allocator,
            &.{ Index.initDim(1), Index.initDim(2), Index.initDim(3), Index.initDim(4), Index.initDim(5) },
        ),
    );
}

test "IndexTest -> Shape" {
    const full = tensor_.full;
    const deinit = @import("Init.zig").deinit;
    const allocator = std.testing.allocator;
    const shape = tensor_.shape;
    defer deinit(); // deinit global singletons

    var t = try full(allocator, &.{ 4, 4 }, f64, 3, .f32);
    defer t.deinit();

    var res1 = try t.index(allocator, &.{ Index.initDim(2), Index.initDim(2) });
    defer res1.deinit();
    try std.testing.expect(shape.eql(try res1.shape(allocator), &.{1}));

    var res2 = try t.index(allocator, &.{ Index.initDim(2), Index.initRange(span) });
    defer res2.deinit();
    try std.testing.expect(shape.eql(try res2.shape(allocator), &.{4}));

    var res3 = try t.index(allocator, &.{Index.initDim(2)});
    defer res3.deinit();
    try std.testing.expect(shape.eql(try res3.shape(allocator), &.{4}));

    var res4 = try t.index(allocator, &.{Index.initRange(Range.initEnd(3))});
    defer res4.deinit();
    try std.testing.expect(shape.eql(try res4.shape(allocator), &.{ 3, 4 }));

    var res5 = try t.index(allocator, &.{Index.initRange(Range.init(1, .{ .dim = 2 }))});
    defer res5.deinit();
    try std.testing.expect(shape.eql(try res5.shape(allocator), &.{ 1, 4 }));

    var res6 = try t.index(
        allocator,
        &.{
            Index.initRange(Range.init(1, .{ .dim = 2 })),
            Index.initRange(Range.init(1, .{ .dim = 2 })),
        },
    );
    defer res6.deinit();
    try std.testing.expect(shape.eql(try res6.shape(allocator), &.{ 1, 1 }));

    var res7 = try t.index(allocator, &.{Index.initRange(Range.init(0, .{ .end = end }))});
    defer res7.deinit();
    try std.testing.expect(shape.eql(try res7.shape(allocator), &.{ 4, 4 }));

    var res8 = try t.index(
        allocator,
        &.{Index.initRange(Range.initWithStride(0, .{ .end = end }, 2))},
    );
    defer res8.deinit();
    try std.testing.expect(shape.eql(try res8.shape(allocator), &.{ 2, 4 }));

    var t1 = try full(allocator, &.{ 5, 6, 7, 8 }, f64, 3, .f32);
    defer t1.deinit();
    var t1_res = try t1.index(
        allocator,
        &.{ Index.initDim(2), Index.initRange(Range.init(2, .{ .dim = 4 })), Index.initRange(span), Index.initDim(3) },
    );
    defer t1_res.deinit();
    try std.testing.expect(shape.eql(try t1_res.shape(allocator), &.{ 2, 7 }));

    var t1_res2 = try t1.index(
        allocator,
        &.{ Index.initRange(span), Index.initDim(3), Index.initRange(span), Index.initRange(span) },
    );
    defer t1_res2.deinit();
    try std.testing.expect(shape.eql(try t1_res2.shape(allocator), &.{ 5, 7, 8 }));

    var t1_res3 = try t1.index(
        allocator,
        &.{
            Index.initRange(span),
            Index.initRange(Range.init(1, .{ .dim = 2 })),
            Index.initRange(span),
            Index.initRange(span),
        },
    );
    defer t1_res3.deinit();
    try std.testing.expect(shape.eql(try t1_res3.shape(allocator), &.{ 5, 1, 7, 8 }));
}

test "IndexTest -> IndexAssignment" {
    const full = tensor_.full;
    const rand = tensor_.rand;
    const arange = tensor_.arange;
    const initAssign = tensor_.initAssign;
    const allClose = tensor_.allClose;
    const deinit = tensor_.deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var t = try full(allocator, &.{ 4, 4 }, f64, 0, .s32);
    defer t.deinit();
    try t.indexAssign(allocator, i32, 1, &.{ Index.initRange(span), Index.initDim(0) });
    try t.indexAdd(allocator, i32, 1, &.{ Index.initRange(span), Index.initDim(1) });
    try t.indexAdd(
        allocator,
        i32,
        1,
        &.{ Index.initRange(span), Index.initRange(Range.init(2, .{ .end = end })) },
    );
    try t.indexMul(allocator, i32, 7, &.{ Index.initRange(span), Index.initRange(span) });
    try t.inPlaceDiv(allocator, i32, 7);
    var expected = try full(allocator, &.{ 4, 4 }, f64, 1, .s32);
    defer expected.deinit();
    try std.testing.expect(try allClose(allocator, t, expected, 1));

    var a = try full(allocator, &.{ 6, 6 }, f64, 0, .f32);
    defer a.deinit();
    try a.indexAssign(allocator, f64, 4, &.{ Index.initDim(3), Index.initDim(4) });
    var expected2 = try full(allocator, &.{1}, f64, 4, .f32);
    defer expected2.deinit();
    var a_idx = try a.index(allocator, &.{ Index.initDim(3), Index.initDim(4) });
    defer a_idx.deinit();
    try std.testing.expect(try allClose(allocator, a_idx, expected2, 1e-5));

    var a_assign_val = try full(allocator, &.{6}, f64, 8, .f32);
    defer a_assign_val.deinit();
    try a.indexAssign(allocator, Tensor, a_assign_val, &.{Index.initDim(2)});
    var a_idx2 = try a.index(allocator, &.{Index.initDim(2)});
    defer a_idx2.deinit();
    try std.testing.expect(try allClose(allocator, a_idx2, a_assign_val, 1e-5));

    var b = try full(allocator, &.{ 3, 3 }, f64, 1, .f32);
    defer b.deinit();
    var c = try initAssign(allocator, b);
    defer c.deinit();
    try b.inPlaceAdd(allocator, f64, 1);
    var b_expected = try full(allocator, &.{ 3, 3 }, f64, 2, .f32);
    defer b_expected.deinit();
    try std.testing.expect(try allClose(allocator, b, b_expected, 1e-5));

    var c_expected = try full(allocator, &.{ 3, 3 }, f64, 1, .f32);
    defer c_expected.deinit();
    try std.testing.expect(try allClose(allocator, c, c_expected, 1e-5));

    var q = try full(allocator, &.{ 4, 4 }, f64, 2, .f32);
    defer q.deinit();
    var r = try full(allocator, &.{4}, f64, 3, .f32);
    defer r.deinit();
    try q.indexAssign(allocator, Tensor, r, &.{Index.initDim(0)});
    var q_idx = try q.index(allocator, &.{Index.initDim(0)});
    defer q_idx.deinit();
    try std.testing.expect(try allClose(allocator, q_idx, r, 1e-5));

    var q_exp = try full(allocator, &.{ 3, 4 }, f64, 2, .f32);
    defer q_exp.deinit();
    var q_idx2 = try q.index(allocator, &.{Index.initRange(Range.init(1, .{ .end = end }))});
    defer q_idx2.deinit();
    try std.testing.expect(try allClose(allocator, q_idx2, q_exp, 1e-5));

    var k = try rand(allocator, &.{ 100, 200 }, .f32);
    defer k.deinit();
    var k_assign = try full(allocator, &.{200}, f64, 0, .f32);
    defer k_assign.deinit();
    try k.indexAssign(allocator, Tensor, k_assign, &.{Index.initDim(3)});
    var k_idx = try k.index(allocator, &.{Index.initDim(3)});
    defer k_idx.deinit();
    try std.testing.expect(try allClose(allocator, k_idx, k_assign, 1e-5));

    // weak ref
    var g = try rand(allocator, &.{ 3, 4, 5 }, .f32);
    defer g.deinit();
    var gC = try g.copy(allocator);
    defer gC.deinit();
    var gI = try g.index(
        allocator,
        &.{ Index.initRange(span), Index.initRange(Range.init(0, .{ .dim = 3 })) },
    );
    defer gI.deinit();
    try g.indexAdd(
        allocator,
        f64,
        3,
        &.{ Index.initRange(span), Index.initRange(Range.init(0, .{ .dim = 3 })) },
    );
    try gI.inPlaceSub(allocator, f64, 3);
    var gC_idx = try gC.index(
        allocator,
        &.{ Index.initRange(span), Index.initRange(Range.init(0, .{ .dim = 3 })) },
    );
    defer gC_idx.deinit();
    try std.testing.expect(try allClose(allocator, gC_idx, gI, 1e-5));

    var x = try rand(allocator, &.{ 5, 6, 7, 8 }, .f32);
    defer x.deinit();
    var x_assign = try full(allocator, &.{ 6, 7, 8 }, f64, 0, .f32);
    defer x_assign.deinit();
    try x.indexAssign(allocator, Tensor, x_assign, &.{Index.initDim(3)});
    var x_idx = try x.index(allocator, &.{Index.initDim(3)});
    defer x_idx.deinit();
    try std.testing.expect(try allClose(allocator, x_idx, x_assign, 1e-5));

    var x_assign2 = try full(allocator, &.{ 5, 6, 8 }, f64, 3, .f32);
    defer x_assign2.deinit();
    try x.indexAssign(allocator, Tensor, x_assign2, &.{ Index.initRange(span), Index.initRange(span), Index.initDim(2) });
    var x_idx2 = try x.index(allocator, &.{ Index.initRange(span), Index.initRange(span), Index.initDim(2) });
    defer x_idx2.deinit();
    try std.testing.expect(try allClose(allocator, x_idx2, x_assign2, 1e-5));

    var x_assign3 = try full(allocator, &.{ 5, 2, 7, 8 }, f64, 2, .f32);
    defer x_assign3.deinit();
    try x.indexAssign(
        allocator,
        Tensor,
        x_assign3,
        &.{ Index.initRange(span), Index.initRange(Range.init(1, .{ .dim = 3 })), Index.initRange(span) },
    );
    var x_idx3 = try x.index(
        allocator,
        &.{ Index.initRange(span), Index.initRange(Range.init(1, .{ .dim = 3 })), Index.initRange(span) },
    );
    defer x_idx3.deinit();
    try std.testing.expect(try allClose(allocator, x_idx3, x_assign3, 1e-5));

    var x_assign4 = try full(allocator, &.{ 5, 5, 7, 5 }, f64, 2, .f32);
    defer x_assign4.deinit();
    var arange_tensor = try arange(allocator, &.{5}, 0, .f32);
    defer arange_tensor.deinit();
    try x.indexAssign(
        allocator,
        Tensor,
        x_assign4,
        &.{ Index.initRange(span), Index.initTensor(arange_tensor), Index.initRange(span), Index.initTensor(arange_tensor) },
    );
    var x_idx4 = try x.index(
        allocator,
        &.{ Index.initRange(span), Index.initTensor(arange_tensor), Index.initRange(span), Index.initTensor(arange_tensor) },
    );
    defer x_idx4.deinit();
    try std.testing.expect(try allClose(allocator, x_idx4, x_assign4, 1e-5));
}

test "IndexTest -> IndexInPlaceOps" {
    const full = tensor_.full;
    const allClose = tensor_.allClose;
    const deinit = tensor_.deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 4, 5, 6 }, f64, 0, .f32);
    defer a.deinit();
    var b = try full(allocator, &.{ 5, 6 }, f64, 1, .f32);
    defer b.deinit();
    try a.indexAdd(allocator, Tensor, b, &.{Index.initDim(2)});
    var a_idx = try a.index(allocator, &.{Index.initDim(2)});
    defer a_idx.deinit();
    try std.testing.expect(try allClose(allocator, a_idx, b, 1e-5));
    try a.indexSub(allocator, Tensor, b, &.{Index.initDim(2)});
    var a_expected = try full(allocator, &.{ 4, 5, 6 }, f64, 0, .f32);
    defer a_expected.deinit();
    try std.testing.expect(try allClose(allocator, a, a_expected, 1e-5));

    var f = try full(allocator, &.{ 1, 3, 3 }, f64, 4, .f32);
    defer f.deinit();
    var d = try full(allocator, &.{3}, f64, 6, .f32);
    defer d.deinit();
    try f.indexAdd(allocator, Tensor, d, &.{ Index.initDim(0), Index.initDim(1) });
    var f_idx = try f.index(allocator, &.{ Index.initDim(0), Index.initDim(1) });
    defer f_idx.deinit();
    try d.inPlaceAdd(allocator, f64, 4);
    try std.testing.expect(try allClose(allocator, f_idx, d, 1e-5));

    var s = try full(allocator, &.{ 4, 5, 6 }, f64, 5, .s32);
    defer s.deinit();
    var sA = try full(allocator, &.{6}, f64, 3, .s32);
    defer sA.deinit();
    try s.indexAdd(allocator, Tensor, sA, &.{ Index.initDim(0), Index.initDim(1) });
    var s_idx = try s.index(allocator, &.{ Index.initDim(0), Index.initDim(1) });
    defer s_idx.deinit();
    try sA.inPlaceAdd(allocator, f64, 5);
    try std.testing.expect(try allClose(allocator, s_idx, sA, 1e-5));
}

test "IndexTest -> flat" {
    const rand = tensor_.rand;
    const full = tensor_.full;
    const allClose = tensor_.allClose;
    const deinit = tensor_.deinit;
    const shape = tensor_.shape;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var m = try rand(allocator, &.{ 4, 6 }, .f32);
    defer m.deinit();
    for (0..@intCast(try m.elements(allocator))) |i| {
        var m_flat = try m.flat(allocator, Index.initDim(@intCast(i)));
        defer m_flat.deinit();
        var m_indexed = try m.index(
            allocator,
            &.{
                Index.initDim(@intCast(@mod(i, 4))),
                Index.initDim(@intCast(@divTrunc(i, 4))),
            },
        );
        defer m_indexed.deinit();
        try std.testing.expect(try allClose(allocator, m_flat, m_indexed, 1e-5));
    }

    var n = try rand(allocator, &.{ 4, 6, 8 }, .f32);
    defer n.deinit();
    for (0..@intCast(try n.elements(allocator))) |i| {
        var n_flat = try n.flat(allocator, Index.initDim(@intCast(i)));
        defer n_flat.deinit();
        var n_indexed = try n.index(
            allocator,
            &.{
                Index.initDim(@intCast(@mod(i, 4))),
                Index.initDim(@intCast(@mod(@divTrunc(i, 4), 6))),
                Index.initDim(@intCast(@mod(@divTrunc(i, 4 * 6), 8))),
            },
        );
        defer n_indexed.deinit();
        try std.testing.expect(try allClose(allocator, n_flat, n_indexed, 1e-5));
    }

    var a = try full(allocator, &.{ 5, 6, 7, 8 }, f64, 9, .f32);
    defer a.deinit();
    var test_indices = [_]Dim{ 0, 1, 4, 11, 62, 104, 288 };
    for (test_indices) |i| {
        var a_flat = try a.flat(allocator, Index.initDim(i));
        defer a_flat.deinit();
        try std.testing.expect(try a_flat.scalar(allocator, f32) == 9);
    }

    try a.flatAssign(allocator, f64, 5, Index.initDim(8));
    var a_flat = try a.flat(allocator, Index.initDim(8));
    defer a_flat.deinit();
    try std.testing.expect(try a_flat.scalar(allocator, f32) == 5);

    for (test_indices) |i| {
        try a.flatAssign(allocator, f64, @floatFromInt(i + 1), Index.initDim(@intCast(i)));
    }
    for (test_indices) |i| {
        var tmp_idx = try a.index(allocator, &.{
            Index.initDim(@intCast(@mod(i, 5))),
            Index.initDim(@intCast(@mod(@divTrunc(i, 5), 6))),
            Index.initDim(@intCast(@mod(@divTrunc(i, 5 * 6), 7))),
            Index.initDim(@intCast(@mod(@divTrunc(i, 5 * 6 * 7), 8))),
        });
        defer tmp_idx.deinit();
        try std.testing.expect(try tmp_idx.scalar(allocator, f32) == @as(f32, @floatFromInt(i + 1)));
    }

    // Tensor assignment
    var tmp_assign = try full(allocator, &.{1}, f64, 7.4, .f32);
    defer tmp_assign.deinit();
    try a.flatAssign(allocator, Tensor, tmp_assign, Index.initDim(32));
    // In-place
    try a.flatAdd(allocator, f64, 33, Index.initDim(100));
    var a_flattened = try a.flatten(allocator);
    defer a_flattened.deinit();
    var a_flattened_idx = try a_flattened.index(allocator, &.{Index.initDim(100)});
    defer a_flattened_idx.deinit();
    var expected = try full(allocator, &.{1}, f64, 33 + 9, .f32);
    defer expected.deinit();
    try std.testing.expect(try allClose(allocator, a_flattened_idx, expected, 1e-5));

    // TODO: Tensor indexing
    // TODO: need to add method to init Tensor from slice

    // Range flat assignment
    var rA = try rand(allocator, &.{6}, .f32);
    defer rA.deinit();
    try a.flatAssign(allocator, Tensor, rA, Index.initRange(Range.init(1, .{ .dim = 7 })));
    var a_flattened2 = try a.flatten(allocator);
    defer a_flattened2.deinit();
    var a_flattened2_idx = try a_flattened2.index(
        allocator,
        &.{Index.initRange(Range.init(1, .{ .dim = 7 }))},
    );
    defer a_flattened2_idx.deinit();
    try std.testing.expect(try allClose(allocator, a_flattened2_idx, rA, 1e-5));

    // With leading singleton dims
    var b = try rand(allocator, &.{ 1, 1, 10 }, .f32);
    defer b.deinit();
    var b_flat = try b.flat(allocator, Index.initRange(Range.initEnd(3)));
    defer b_flat.deinit();
    try std.testing.expect(shape.eql(try b_flat.shape(allocator), &.{3}));
    var b_assign = try full(allocator, &.{3}, f64, 6, .f32);
    defer b_assign.deinit();
    try b.flatAssign(allocator, Tensor, b_assign, Index.initRange(Range.initEnd(3)));
    var b_flattened = try b.flatten(allocator);
    defer b_flattened.deinit();
    var b_flattened_idx = try b_flattened.index(allocator, &.{Index.initRange(Range.initEnd(3))});
    defer b_flattened_idx.deinit();
    try std.testing.expect(try allClose(allocator, b_flattened_idx, b_assign, 1e-5));
}

test "IndexTest -> TensorIndex" {
    const full = tensor_.full;
    const rand = tensor_.rand;
    const shape = tensor_.shape;
    const allClose = tensor_.allClose;
    const initAssign = tensor_.initAssign;
    const arange = tensor_.arange;
    const add = tensor_.add;
    const deinit = tensor_.deinit;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var idxs = [_]Dim{ 0, 1, 4, 9, 11, 13, 16, 91 };
    var size = idxs.len;
    var tensor_indices = try full(allocator, &.{@as(i64, @intCast(size))}, f64, 0, .f32);
    defer tensor_indices.deinit();
    for (0..size) |i| {
        try tensor_indices.indexAssign(
            allocator,
            Dim,
            idxs[i],
            &.{Index.initDim(@intCast(i))},
        );
    }

    var a = try rand(allocator, &.{100}, .f32);
    defer a.deinit();
    var indexed = try a.index(allocator, &.{Index.initTensor(tensor_indices)});
    defer indexed.deinit();
    for (0..size) |i| {
        var indexed_idx = try indexed.index(allocator, &.{Index.initDim(@intCast(i))});
        defer indexed_idx.deinit();
        var a_idx = try a.index(allocator, &.{Index.initDim(idxs[i])});
        defer a_idx.deinit();
        try std.testing.expect(try allClose(allocator, indexed_idx, a_idx, 1e-5));
    }

    try a.indexAssign(allocator, f64, 5, &.{Index.initTensor(tensor_indices)});
    var expected1 = try full(allocator, &.{@as(i64, @intCast(size))}, f64, 5, .f32);
    defer expected1.deinit();
    var a_idx1 = try a.index(allocator, &.{Index.initTensor(tensor_indices)});
    defer a_idx1.deinit();
    try std.testing.expect(try allClose(allocator, a_idx1, expected1, 1e-5));

    // Out of range indices
    var i = try arange(allocator, &.{10}, 0, .u32);
    defer i.deinit();
    var b = try rand(allocator, &.{ 20, 20 }, .f32);
    defer b.deinit();
    var ref = try initAssign(allocator, b);
    defer ref.deinit();
    var b_idx = try b.index(allocator, &.{Index.initTensor(i)});
    defer b_idx.deinit();
    var b_idx2 = try b.index(allocator, &.{Index.initRange(Range.initEnd(10))});
    defer b_idx2.deinit();
    try std.testing.expect(shape.eql(try b_idx.shape(allocator), try b_idx2.shape(allocator)));
    try std.testing.expect(try allClose(allocator, b_idx, b_idx2, 1e-5));

    try b.indexAdd(allocator, f64, 3, &.{Index.initTensor(i)});
    var b_idx3 = try b.index(allocator, &.{Index.initTensor(i)});
    defer b_idx3.deinit();
    var b_idx4 = try b.index(allocator, &.{Index.initRange(Range.initEnd(10))});
    defer b_idx4.deinit();
    try std.testing.expect(try allClose(allocator, b_idx3, b_idx4, 1e-5));
    var ref_add = try add(allocator, Tensor, ref, f64, 3);
    defer ref_add.deinit();
    var b_idx5 = try b.index(allocator, &.{Index.initTensor(i)});
    defer b_idx5.deinit();
    var ref_exp1 = try ref_add.index(allocator, &.{Index.initTensor(i)});
    defer ref_exp1.deinit();
    try std.testing.expect(try allClose(allocator, b_idx5, ref_exp1, 1e-5));

    var b_rhs = try full(
        allocator,
        &.{ @as(i64, @intCast(try i.elements(allocator))), try b.dim(allocator, 1) },
        f64,
        10,
        .f32,
    );
    defer b_rhs.deinit();
    try b.indexAdd(allocator, Tensor, b_rhs, &.{Index.initTensor(i)});
    var b_idx6 = try b.index(allocator, &.{Index.initTensor(i)});
    defer b_idx6.deinit();
    var ref_add2 = try add(allocator, Tensor, ref, f64, 13);
    defer ref_add2.deinit();
    var ref_exp2 = try ref_add2.index(allocator, &.{Index.initTensor(i)});
    defer ref_exp2.deinit();
    try std.testing.expect(shape.eql(try b_idx6.shape(allocator), try ref_exp2.shape(allocator)));
    try std.testing.expect(try allClose(allocator, b_idx6, ref_exp2, 1e-5));

    // Tensor index a > 1D tensor
    var c = try rand(allocator, &.{ 10, 10, 10 }, .f32);
    defer c.deinit();
    var c_arange = try arange(allocator, &.{5}, 0, .f32);
    defer c_arange.deinit();
    var c_idx = try c.index(allocator, &.{Index.initTensor(c_arange)});
    defer c_idx.deinit();
    try std.testing.expect(shape.eql(try c_idx.shape(allocator), &.{ 5, 10, 10 }));
}

// TODO: test "IndexTest -> ExpressionIndex" {}
