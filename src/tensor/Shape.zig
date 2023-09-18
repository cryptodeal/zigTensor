const std = @import("std");
const af = @import("../bindings/af/ArrayFire.zig");

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
pub const Shape = struct {
    /// Storage for the dimension values. Defaults to an empty
    /// Shape {0}, whereas {} is a scalar shape.
    dims_: std.ArrayList(Dim),
    /// Gives the maximum number of dimensions a tensor of a particular
    /// shape can have.
    ///
    /// If the maximum size can be arbitrarily high, `std.math.maxInt(usize)`
    /// should be used.
    kMaxDims: usize = std.math.maxInt(usize),

    /// Initialize a Shape via a slice of `Dim` values.
    pub fn initRaw(allocator: std.mem.Allocator) Shape {
        return .{
            .dims_ = std.ArrayList(Dim).init(allocator),
        };
    }

    /// Initialize a Shape via a slice of `Dim` values.
    pub fn init(allocator: std.mem.Allocator, d: []Dim) !Shape {
        var self = Shape.initRaw(allocator);
        try self.dims_.appendSlice(d);
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *const Shape) void {
        self.dims_.deinit();
    }

    /// Check if a dimension is valid (i.e. in bounds) given the
    /// current size of the shape. If not valid, throws an error.
    pub fn checkDimsOrThrow(self: *const Shape, dimension: usize) !void {
        if (dimension > self.ndim() - 1) {
            std.log.debug("Shape index {d} out of bounds for shape with {d} dimensions.\n", .{ dimension, self.dims_.items.len });
            return ShapeErrors.InvalidDimension;
        }
    }

    /// Returns the number of elements in a tensor of this shape.
    pub fn elements(self: *const Shape) Dim {
        if (self.dims_.items.len == 0) {
            return kEmptyShapeNumberOfElements;
        }
        var count: Dim = 1;
        for (self.dims_.items) |d| count *= d;
        return count;
    }

    /// Returns the number of dimensions in the shape.
    pub fn ndim(self: *const Shape) usize {
        return self.dims_.items.len;
    }

    /// Get the size of a given dimension in the number of arguments.
    /// Throws if the given dimension is larger than the number of
    /// dimensions.
    ///
    /// Returns the number of elements at the given dimension.
    pub fn dim(self: *const Shape, dimension: usize) !Dim {
        try self.checkDimsOrThrow(dimension);
        return self.dims_.items[dimension];
    }

    /// Compares two shapes. Returns true if their dimensions are equal.
    pub fn eql(self: *const Shape, other: *const Shape) bool {
        return std.mem.eql(Dim, self.dims_.items, other.dims_.items);
    }

    /// Gets a reference to the underlying dims ArrayList.
    pub fn get(self: *Shape) *std.ArrayList(Dim) {
        return &self.dims_;
    }

    /// Formats Shape for printing to writer.
    pub fn format(value: Shape, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        var tmp_shape = @constCast(&value);
        try writer.print("(", .{});
        for (value.dims_.items, 0..) |v, i| {
            try writer.print("{d}", .{v});
            if (i < tmp_shape.ndim() - 1) try writer.print(", ", .{});
        }
        return writer.print(")", .{});
    }
};

test "Shape basic" {
    var allocator = std.testing.allocator;

    var dims = [_]Dim{ 3, 4 };
    var s = try Shape.init(allocator, &dims);
    defer s.deinit();
    try std.testing.expect(s.ndim() == 2);
    try std.testing.expect(try s.dim(0) == 3);
    try std.testing.expect(try s.dim(1) == 4);
    try std.testing.expectError(ShapeErrors.InvalidDimension, s.dim(5));
}

test "Shape many dimensions" {
    var allocator = std.testing.allocator;
    var dims = [_]Dim{ 1, 2, 3, 4, 5, 6, 7 };
    var many = try Shape.init(allocator, &dims);
    defer many.deinit();
    try std.testing.expect(many.ndim() == 7);
    try std.testing.expect(try many.dim(5) == 6);
}

test "Shape ndim" {
    var allocator = std.testing.allocator;

    var dims1 = [_]Dim{};
    var s = try Shape.init(allocator, &dims1);
    try std.testing.expect(s.ndim() == 0);
    s.deinit();

    var dims2 = [_]Dim{ 1, 0, 1 };
    s = try Shape.init(allocator, &dims2);
    try std.testing.expect(s.ndim() == 3);
    s.deinit();

    var dims3 = [_]Dim{ 1, 1, 1 };
    s = try Shape.init(allocator, &dims3);
    try std.testing.expect(s.ndim() == 3);
    s.deinit();

    var Dim4 = [_]Dim{ 5, 2, 3 };
    s = try Shape.init(allocator, &Dim4);
    try std.testing.expect(s.ndim() == 3);
    s.deinit();

    var dims5 = [_]Dim{ 1, 2, 3, 6 };
    s = try Shape.init(allocator, &dims5);
    try std.testing.expect(s.ndim() == 4);
    s.deinit();

    var dims6 = [_]Dim{ 1, 2, 3, 1, 1, 1 };
    s = try Shape.init(allocator, &dims6);
    try std.testing.expect(s.ndim() == 6);
    s.deinit();

    var dims7 = [_]Dim{ 1, 2, 3, 1, 1, 1, 5 };
    s = try Shape.init(allocator, &dims7);
    try std.testing.expect(s.ndim() == 7);
    s.deinit();

    var dims8 = [_]Dim{ 4, 2, 3, 1, 1, 1, 5 };
    s = try Shape.init(allocator, &dims8);
    try std.testing.expect(s.ndim() == 7);
    s.deinit();
}

test "Shape elements" {
    var allocator = std.testing.allocator;

    var dims1 = [_]Dim{};
    var s = try Shape.init(allocator, &dims1);
    try std.testing.expect(s.elements() == 1);
    s.deinit();

    var dims2 = [_]Dim{0};
    s = try Shape.init(allocator, &dims2);
    try std.testing.expect(s.elements() == 0); // empty tensor
    s.deinit();

    var dims3 = [_]Dim{ 1, 1, 1, 1 };
    s = try Shape.init(allocator, &dims3);
    try std.testing.expect(s.elements() == 1);
    s.deinit();

    var Dim4 = [_]Dim{ 1, 2, 3, 4 };
    s = try Shape.init(allocator, &Dim4);
    try std.testing.expect(s.elements() == 24);
    s.deinit();

    var dims5 = [_]Dim{ 1, 2, 3, 0 };
    s = try Shape.init(allocator, &dims5);
    try std.testing.expect(s.elements() == 0);
    s.deinit();
}

test "Shape equality" {
    var allocator = std.testing.allocator;

    var base_dims = [_]Dim{ 1, 2, 3, 4 };
    var a = try Shape.init(allocator, &base_dims);
    var b = try Shape.init(allocator, &base_dims);
    try std.testing.expect(a.eql(&b));
    b.deinit();

    var dims1 = [_]Dim{ 4, 3, 4 };
    b = try Shape.init(allocator, &dims1);
    try std.testing.expect(!a.eql(&b));
    a.deinit();
    b.deinit();

    var dims2 = [_]Dim{ 1, 2 };
    a = try Shape.init(allocator, &dims2);
    var dims3 = [_]Dim{ 1, 1, 1, 2 };
    b = try Shape.init(allocator, &dims3);
    try std.testing.expect(!a.eql(&b));
    a.deinit();
    b.deinit();

    var Dim4 = [_]Dim{ 5, 2, 3 };
    a = try Shape.init(allocator, &Dim4);
    var dims5 = [_]Dim{ 5, 2, 3, 1 };
    b = try Shape.init(allocator, &dims5);
    try std.testing.expect(!a.eql(&b));
    a.deinit();

    a = try Shape.init(allocator, &dims5);
    try std.testing.expect(a.eql(&b));
    a.deinit();
    b.deinit();

    var dims6 = [_]Dim{ 5, 2, 1, 1 };
    a = try Shape.init(allocator, &dims6);
    var dims7 = [_]Dim{ 5, 2, 1, 4 };
    b = try Shape.init(allocator, &dims7);
    try std.testing.expect(!a.eql(&b));
    a.deinit();
    b.deinit();
}

test "Shape string" {
    var allocator = std.testing.allocator;

    var dims1 = [_]Dim{ 3, 4, 7, 9 };
    var s = try Shape.init(allocator, &dims1);
    var str = std.ArrayList(u8).init(allocator);
    try str.writer().print("{any}", .{s});
    try std.testing.expectEqualStrings("(3, 4, 7, 9)", str.items);
    s.deinit();
    str.clearAndFree();

    var dims2 = [_]Dim{};
    s = try Shape.init(allocator, &dims2);
    try str.writer().print("{any}", .{s});
    try std.testing.expectEqualStrings("()", str.items);
    s.deinit();
    str.clearAndFree();

    var dims3 = [_]Dim{0};
    s = try Shape.init(allocator, &dims3);
    try str.writer().print("{any}", .{s});
    try std.testing.expectEqualStrings("(0)", str.items);
    s.deinit();
    str.clearAndFree();

    var Dim4 = [_]Dim{ 7, 7, 7, 7, 7, 7, 7 };
    s = try Shape.init(allocator, &Dim4);
    try str.writer().print("{any}", .{s});
    try std.testing.expectEqualStrings("(7, 7, 7, 7, 7, 7, 7)", str.items);
    s.deinit();
    str.deinit();
}
