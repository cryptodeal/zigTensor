const std = @import("std");
const zt_shape = @import("Shape.zig");
const zt_base = @import("TensorBase.zig");

const Dim = zt_shape.Dim;
const Tensor = zt_base.Tensor;

/// Represents the imaginary index after the last index along
/// an axis of a tensor. This special case is used because
/// `Range.end` is exclusive.
pub const end_t = struct {};

/// A static instance of `end_t` for convenience; e.g. can be used
/// for `Range(0, end)` to index all elements along a certain axis.
pub const end: end_t = .{};

/// Allowed indexing operators.
pub const IndexType = enum(u8) { Tensor = 0, Range = 1, Literal = 2, Span = 3 };

pub const RangeIdxTypeTag = enum { end, dim };

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

    pub fn start(self: *Range) Dim {
        return self.start_;
    }

    pub fn end(self: *Range) ?Dim {
        return self.end_;
    }

    pub fn endVal(self: *Range) !Dim {
        if (self.end_ != null) {
            return self.end_.?;
        }
        std.log.err("[Range.endVal] end is `end_t`\n", .{});
        return RangeError.FailedToGetEndVal;
    }

    pub fn stride(self: *Range) Dim {
        return self.stride_;
    }

    pub fn eql(self: *Range, other: *Range) bool {
        return std.meta.eql(self.*, other.*);
    }
};

pub const span: Range = Range.initWithStride(-1, .{ .dim = -1 }, 0);

pub const IndexErrors = error{IndexGetTypeMismatch};

pub const IndexTypeTag = enum { Literal, Range, Tensor };

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
    pub const IndexVariant = union(IndexTypeTag) { Literal: Dim, Range: Range, Tensor: Tensor };

    /// The type of Indexing operator.
    type_: IndexType,

    /// Underlying data referred to by the index.
    index_: IndexVariant,

    fn initTensor(tensor: *Tensor) Index {
        return .{
            .type_ = .Tensor,
            .index_ = .{ .Tensor = tensor },
        };
    }

    pub fn initRange(comptime range: Range) Index {
        return .{
            .type_ = if (std.meta.eql(range, span)) .Span else .Range,
            .index_ = .{ .Range = range },
        };
    }

    pub fn initDim(comptime idx: Dim) Index {
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
    pub fn idxType(comptime self: *const Index) IndexType {
        return self.type_;
    }

    pub fn isSpan(comptime self: *const Index) bool {
        return self.type_ == .Span;
    }

    pub fn get(self: *Index, comptime T: type) !T {
        switch (self.index) {
            .Literal => {
                if (T == Dim) {
                    return self.index_.Literal;
                }
            },
            .Range => {
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
        std.log.err("[Index.get] type `{s}` doesn't match IndexTypeTag `{s}`\n", .{ @typeName(T), @tagName(self.type_) });
        return IndexErrors.IndexGetTypeMismatch;
    }

    pub fn getVariant(self: *Index) IndexVariant {
        return self.index_;
    }
};

test "IndexTest Range" {
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

test "IndexTest Range.eql" {
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

test "IndexTest Index.idxType" {
    const idx1 = Index.initDim(1);
    try std.testing.expect(idx1.idxType() == .Literal);

    const idx2 = Index.initRange(Range.initEnd(3));
    try std.testing.expect(idx2.idxType() == .Range);

    const idx3 = Index.initRange(span);
    try std.testing.expect(idx3.idxType() == .Span);
    try std.testing.expect(idx3.isSpan());
}
