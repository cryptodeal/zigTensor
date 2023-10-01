const std = @import("std");

/// The type of a dimension.
pub const Dim = i64;

pub const ShapeErrors = error{InvalidDimension};

const kEmptyShapeNumberOfElements: Dim = 1;

/// Data structure describing the dimensions of a tensor.
///
/// Shape has explicit dimensions and sizes, which contrasts
/// some Tensor libraries that rely on implicit dimensions
/// (i.e. ignore 1). Accordingly, zigTensor Shapes can be of
/// arbitrary size where 1-dimensions makes each distinguishable.
///
/// Shapes dimensions should be >= 1 in size. Shapes with a zero
/// dimension have 0 elements (even if other dimensions are nonzero
/// in size). E.g. a Shape of {0} has zero elements just as a Shape
/// with dimensions {1, 2, 3, 0} does.
///
/// Different tensor backends implement different shape and
/// dimension semantics. As a result, these need to be converted
/// bidirectionally from zigTensor Shapes. Having a shared set
/// of behaviors in this API gaurantees that tensors and their shapes
/// can be manipulated across tensor backends.
pub const Shape = []const Dim;

/// Check if a dimension is valid (i.e. in bounds) given the
/// current size of the shape. If not valid, throws an error.
fn checkDimsOrThrow(shape: Shape, dimension: usize) !void {
    if (dimension > ndim(shape) - 1) {
        std.log.debug("Shape index {d} out of bounds for shape with {d} dimensions.\n", .{ dimension, shape.len });
        return ShapeErrors.InvalidDimension;
    }
}

/// Returns the number of dimensions in the shape.
pub fn ndim(shape: Shape) usize {
    return shape.len;
}

/// Get the size of a given dimension in the number of arguments.
/// Throws if the given dimension is larger than the number of
/// dimensions.
///
/// Returns the number of elements at the given dimension.
pub fn dim(shape: Shape, dimension: usize) !Dim {
    try checkDimsOrThrow(shape, dimension);
    return shape[dimension];
}

/// Returns the number of elements in a tensor of this shape.
pub fn elements(shape: Shape) Dim {
    if (shape.len == 0) return kEmptyShapeNumberOfElements;
    var count: Dim = 1;
    for (shape) |d| count *= d;
    return count;
}

/// Compares two shapes. Returns true if their dimensions are equal.
pub fn eql(a: Shape, b: Shape) bool {
    return std.mem.eql(Dim, a, b);
}

test "Shape basic" {
    try std.testing.expect(ndim(&.{ 3, 4 }) == 2);
    try std.testing.expect(try dim(&.{ 3, 4 }, 0) == 3);
    try std.testing.expect(try dim(&.{ 3, 4 }, 1) == 4);
    try std.testing.expectError(ShapeErrors.InvalidDimension, dim(&.{ 3, 4 }, 5));
}

test "Shape many dimensions" {
    try std.testing.expect(ndim(&.{ 1, 2, 3, 4, 5, 6, 7 }) == 7);
    try std.testing.expect(try dim(&.{ 1, 2, 3, 4, 5, 6, 7 }, 5) == 6);
}

test "Shape ndim" {
    try std.testing.expect(ndim(&.{}) == 0);
    try std.testing.expect(ndim(&.{ 1, 0, 1 }) == 3);
    try std.testing.expect(ndim(&.{ 1, 1, 1 }) == 3);
    try std.testing.expect(ndim(&.{ 5, 2, 3 }) == 3);
    try std.testing.expect(ndim(&.{ 1, 2, 3, 6 }) == 4);
    try std.testing.expect(ndim(&.{ 1, 2, 3, 1, 1, 1 }) == 6);
    try std.testing.expect(ndim(&.{ 1, 2, 3, 1, 1, 1, 5 }) == 7);
    try std.testing.expect(ndim(&.{ 4, 2, 3, 1, 1, 1, 5 }) == 7);
}

test "Shape elements" {
    try std.testing.expect(elements(&.{}) == 1);
    try std.testing.expect(elements(&.{0}) == 0); // empty tensor
    try std.testing.expect(elements(&.{ 1, 1, 1, 1 }) == 1);
    try std.testing.expect(elements(&.{ 1, 2, 3, 4 }) == 24);
    try std.testing.expect(elements(&.{ 1, 2, 3, 0 }) == 0);
}

test "Shape equality" {
    try std.testing.expect(eql(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 4 }));
    try std.testing.expect(!eql(&.{ 1, 2, 3, 4 }, &.{ 4, 3, 4 }));
    try std.testing.expect(!eql(&.{ 1, 2 }, &.{ 1, 1, 1, 2 }));
    try std.testing.expect(!eql(&.{ 5, 2, 3 }, &.{ 5, 2, 3, 1 }));
    try std.testing.expect(eql(&.{ 5, 2, 3, 1 }, &.{ 5, 2, 3, 1 }));
    try std.testing.expect(!eql(&.{ 5, 2, 1, 1 }, &.{ 5, 2, 1, 4 }));
}
