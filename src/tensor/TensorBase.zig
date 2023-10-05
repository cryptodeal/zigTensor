const std = @import("std");
const tensor = @import("tensor.zig");

const Dim = tensor.shape.Dim;
const DType = tensor.DType;
const defaultTensorBackend = tensor.defaultTensorBackend;
const DefaultTensorType_t = tensor.DefaultTensorType_t;
const dtypeTraits = tensor.dtypeTraits;
const Index = tensor.Index;
const Shape = tensor.shape.Shape;
const Stream = @import("../runtime/Stream.zig");
const TensorAdapterBase = tensor.TensorAdapterBase;
const TensorBackend = tensor.TensorBackend;
const ztTensorBackendsMatch = tensor.ztTensorBackendsMatch;

/// Enum for various tensor backends.
pub const TensorBackendType = enum(u8) { Stub, Tracer, ArrayFire, OneDnn, Jit };

/// Location of memory or tensors.
pub const Location = enum(u8) { Host, Device };

/// Alias to make it semantically clearer when referring to buffer location
const MemoryLocation = Location;

/// Tensor storage types.
pub const StorageType = enum(u8) { Dense = 0, CSR = 1, CSC = 2, COO = 3 };

/// Transformations to apply to Tensors (i.e. matrices) before applying certain
/// operations (i.e. matmul).
pub const MatrixProperty = enum(u8) { None = 0, Transpose = 1 };

/// Sorting mode for sorting-related functions.
pub const SortMode = enum(u8) { Descending = 0, Ascending = 1 };

/// Padding types for the pad operator.
pub const PadType = enum(u8) {
    /// pad with a constant zero value.
    Constant,
    /// pad with the values at the edges of the tensor.
    Edge,
    /// pad with a reflection of the tensor mirrored along each edge.
    Symmetric,
};

/// A Tensor that can be used for computations.
///
/// The underlying implementations of tensor operations are contained in
/// types implementing the `TensorAdapterBase` interface; these implementations
/// also contain the state associated with the tensor. Tensor stores the vTable
/// struct, which holds a pointer to the underlying implementation, which can be
/// swapped out if the implementation of the backend changes.
///
/// `TensorAdapterBase` implementations may differ across tensor libraries,
/// hardware-specific libraries or compilers and DSLs.
pub const Tensor = struct {
    /// The vTable interface, which contains the pointer to the underlying
    /// tensor implementation.
    impl_: TensorAdapterBase,

    pub fn init(impl: TensorAdapterBase) Tensor {
        return Tensor{ .impl_ = impl };
    }

    pub fn initEmpty(allocator: std.mem.Allocator) !Tensor {
        return Tensor.init(TensorAdapterBase.init(try DefaultTensorType_t.initEmpty(allocator)));
    }

    pub fn fromSlice(allocator: std.mem.Allocator, s: Shape, comptime T: type, data: []const T, data_type: DType) !Tensor {
        var backend_ = try defaultTensorBackend(allocator);
        return switch (T) {
            // TODO: handle more types
            // TODO: use `@ptrCast` to coerce to c types???
            f16, f32, f64 => backend_.fromSlice(allocator, s, @constCast(data.ptr), data_type),
            i16 => backend_.fromSlice(allocator, s, @as([*]c_short, @ptrCast(@alignCast(@constCast(data.ptr)))), data_type),
            u16 => backend_.fromSlice(allocator, s, @as([*]c_ushort, @ptrCast(@alignCast(@constCast(data.ptr)))), data_type),
            i32 => backend_.fromSlice(allocator, s, @as([*]c_int, @ptrCast(@alignCast(@constCast(data.ptr)))), data_type),
            u32 => backend_.fromSlice(allocator, s, @as([*]c_uint, @ptrCast(@alignCast(@constCast(data.ptr)))), data_type),
            i64 => backend_.fromSlice(allocator, s, @as([*]c_longlong, @ptrCast(@alignCast(@constCast(data.ptr)))), data_type),
            u64 => backend_.fromSlice(allocator, s, @as([*]c_ulonglong, @ptrCast(@alignCast(@constCast(data.ptr)))), data_type),
            else => @compileError("Unsupported type passed to fromSlice"),
        };
    }

    pub fn deinit(self: *const Tensor) void {
        self.impl_.deinit();
    }

    pub fn copy(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.copy(allocator);
    }

    pub fn shallowCopy(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.shallowCopy(allocator);
    }

    pub fn shape(self: *const Tensor, allocator: std.mem.Allocator) !Shape {
        return self.impl_.shape(allocator);
    }

    pub fn location(self: *const Tensor, allocator: std.mem.Allocator) !Location {
        return self.impl_.location(allocator);
    }

    pub fn elements(self: *const Tensor, allocator: std.mem.Allocator) !Dim {
        return tensor.shape.elements(try self.shape(allocator));
    }

    pub fn dim(self: *const Tensor, allocator: std.mem.Allocator, dimension: usize) !Dim {
        return tensor.shape.dim(try self.shape(allocator), dimension);
    }

    pub fn ndim(self: *const Tensor, allocator: std.mem.Allocator) !usize {
        return tensor.shape.ndim(try self.shape(allocator));
    }

    pub fn isEmpty(self: *const Tensor, allocator: std.mem.Allocator) !bool {
        var elements_ = try self.elements(allocator);
        return elements_ == 0;
    }

    // TODO: implement
    // pub fn hasAdapter(self: *const Tensor) bool {
    // return self.impl_.get() != null;
    // }

    pub fn bytes(self: *const Tensor, allocator: std.mem.Allocator) !usize {
        var elements_: usize = @intCast(try self.elements(allocator));
        var dtype_ = try self.dtype(allocator);
        return elements_ * dtype_.getSize();
    }

    pub fn dtype(self: *const Tensor, allocator: std.mem.Allocator) !DType {
        return self.impl_.dtype(allocator);
    }

    pub fn isSparse(self: *const Tensor, allocator: std.mem.Allocator) !bool {
        return self.impl_.isSparse(allocator);
    }

    pub fn astype(self: *const Tensor, allocator: std.mem.Allocator, new_type: DType) !Tensor {
        return self.impl_.astype(allocator, new_type);
    }

    pub fn index(self: *const Tensor, allocator: std.mem.Allocator, indices: []const Index) !Tensor {
        return self.impl_.index(allocator, indices);
    }

    pub fn flatten(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.flatten(allocator);
    }

    pub fn flat(self: *const Tensor, allocator: std.mem.Allocator, idx: Index) !Tensor {
        return self.impl_.flat(allocator, idx);
    }

    pub fn asContiguousTensor(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.impl_.asContiguousTensor(allocator);
    }

    pub fn backendType(self: *const Tensor) TensorBackendType {
        return self.impl_.backendType();
    }

    pub fn getAdapter(self: *const Tensor, comptime T: type) *T {
        return @ptrCast(@alignCast(self.impl_.ptr));
    }

    pub fn backend(self: *const Tensor, allocator: std.mem.Allocator) !TensorBackend {
        return self.impl_.backend(allocator);
    }

    // TODO: FL_CREATE_MEMORY_OPS macro equivalent

    pub fn scalar(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type) !T {
        if (try self.isEmpty(allocator)) {
            std.log.debug("Tensor.scalar called on empty tensor\n", .{});
            return error.ScalarFailedEmptyTensor;
        }
        var type_trait = comptime dtypeTraits(T);
        var self_dtype = try self.dtype(allocator);
        if (self_dtype != type_trait.zt_type) {
            std.log.debug(
                "Tensor.scalar: requested type of {s} doesn't match tensor type, which is {s}\n",
                .{ type_trait.string, self_dtype.toString() },
            );
        }
        return self.impl_.scalar(allocator, T);
    }

    pub fn allocHost(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type) !?[]T {
        if (try self.isEmpty(allocator)) {
            return null;
        }
        var res: []T = try allocator.alloc(T, try self.bytes(allocator) / @sizeOf(T));
        try self.impl_.host(allocator, res.ptr);
        return res;
    }

    pub fn unlock(self: *const Tensor, allocator: std.mem.Allocator) !void {
        return self.impl_.unlock(allocator);
    }

    pub fn isLocked(self: *const Tensor, allocator: std.mem.Allocator) !bool {
        return self.impl_.isLocked(allocator);
    }

    pub fn isContiguous(self: *const Tensor, allocator: std.mem.Allocator) !bool {
        return self.impl_.isContiguous(allocator);
    }

    pub fn strides(self: *const Tensor, allocator: std.mem.Allocator) !Shape {
        return self.impl_.strides(allocator);
    }

    pub fn stream(self: *const Tensor, allocator: std.mem.Allocator) !Stream {
        return self.impl_.stream(allocator);
    }

    pub fn setContext(self: *const Tensor, context: ?*anyopaque) !void {
        return self.impl_.setContext(context);
    }

    pub fn getContext(self: *const Tensor) !?*anyopaque {
        return self.impl_.getContext();
    }

    pub fn toString(self: *const Tensor, allocator: std.mem.Allocator) ![]const u8 {
        return self.impl_.toString(allocator);
    }

    // in-place operations/assignment operators
    pub fn assign(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T) !void {
        var bknd = try self.backend(allocator);
        var rhsTensor: Tensor = undefined;
        var rhsTensorInit = false;
        defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
        if (T == Tensor) {
            rhsTensor = rhs;
        } else {
            rhsTensor = try bknd.full(allocator, try self.shape(allocator), T, rhs, try self.dtype(allocator));
            rhsTensorInit = true;
        }
        var check = [_]Tensor{ self.*, rhsTensor };
        try ztTensorBackendsMatch(@src().fn_name, &check);
        return bknd.assign(allocator, self.*, rhsTensor);
    }

    pub fn flatAssign(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, idx: Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.flatAssign(allocator, self.*, T, rhs, idx);
    }

    pub fn flatAdd(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, idx: Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.flatAdd(allocator, self.*, T, rhs, idx);
    }

    pub fn flatSub(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, idx: Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.flatSub(allocator, self.*, T, rhs, idx);
    }

    pub fn flatMul(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, idx: Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.flatMul(allocator, self.*, T, rhs, idx);
    }

    pub fn flatDiv(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, idx: Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.flatDiv(allocator, self.*, T, rhs, idx);
    }

    pub fn indexAssign(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, indices: []const Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.indexAssign(allocator, self.*, T, rhs, indices);
    }

    pub fn indexAdd(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, indices: []const Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.indexAdd(allocator, self.*, T, rhs, indices);
    }

    pub fn indexSub(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, indices: []const Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.indexSub(allocator, self.*, T, rhs, indices);
    }

    pub fn indexMul(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, indices: []const Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.indexMul(allocator, self.*, T, rhs, indices);
    }

    pub fn indexDiv(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, indices: []const Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.indexDiv(allocator, self.*, T, rhs, indices);
    }

    pub fn inPlaceAdd(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T) !void {
        var bknd = try self.backend(allocator);
        var rhsTensor: Tensor = undefined;
        var rhsTensorInit = false;
        defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
        if (T == Tensor) {
            rhsTensor = rhs;
        } else {
            var used_shape = try self.shape(allocator);
            rhsTensor = try bknd.full(allocator, used_shape, T, rhs, try self.dtype(allocator));
            rhsTensorInit = true;
        }
        var check = [_]Tensor{ self.*, rhsTensor };
        try ztTensorBackendsMatch(@src().fn_name, &check);
        return bknd.inPlaceAdd(allocator, self.*, rhsTensor);
    }

    pub fn inPlaceSub(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T) !void {
        var bknd = try self.backend(allocator);
        var rhsTensor: Tensor = undefined;
        var rhsTensorInit = false;
        defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
        if (T == Tensor) {
            rhsTensor = rhs;
        } else {
            var used_shape = try self.shape(allocator);
            rhsTensor = try bknd.full(allocator, used_shape, T, rhs, try self.dtype(allocator));
            rhsTensorInit = true;
        }
        var check = [_]Tensor{ self.*, rhsTensor };
        try ztTensorBackendsMatch(@src().fn_name, &check);
        return bknd.inPlaceSub(allocator, self.*, rhsTensor);
    }

    pub fn inPlaceMul(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T) !void {
        var bknd = try self.backend(allocator);
        var rhsTensor: Tensor = undefined;
        var rhsTensorInit = false;
        defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
        if (T == Tensor) {
            rhsTensor = rhs;
        } else {
            var used_shape = try self.shape(allocator);
            rhsTensor = try bknd.full(allocator, used_shape, T, rhs, try self.dtype(allocator));
            rhsTensorInit = true;
        }
        var check = [_]Tensor{ self.*, rhsTensor };
        try ztTensorBackendsMatch(@src().fn_name, &check);
        return bknd.inPlaceMul(allocator, self.*, rhsTensor);
    }

    pub fn inPlaceDiv(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T) !void {
        var bknd = try self.backend(allocator);
        var rhsTensor: Tensor = undefined;
        var rhsTensorInit = false;
        defer if (rhsTensorInit) rhsTensor.deinit(); // if initializing rhsTensor, defer freeing associated mem
        if (T == Tensor) {
            rhsTensor = rhs;
        } else {
            var used_shape = try self.shape(allocator);
            rhsTensor = try bknd.full(allocator, used_shape, T, rhs, try self.dtype(allocator));
            rhsTensorInit = true;
        }
        var check = [_]Tensor{ self.*, rhsTensor };
        try ztTensorBackendsMatch(@src().fn_name, &check);
        return bknd.inPlaceDiv(allocator, self.*, rhsTensor);
    }
};

// Unit tests/helpers

fn matmulRef(allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
    // (M x N) x (N x K) --> (M x K)
    var M = try lhs.dim(allocator, 0);
    var N = try lhs.dim(allocator, 1);
    var K = try rhs.dim(allocator, 1);

    var out = try tensor.full(allocator, &.{ M, K }, f32, 0, .f32);
    for (0..@intCast(M)) |i| {
        for (0..@intCast(K)) |j| {
            for (0..@intCast(N)) |k| {
                var tmp_lhs_idx = try lhs.index(allocator, &.{ Index.initDim(@intCast(i)), Index.initDim(@intCast(k)) });
                defer tmp_lhs_idx.deinit();
                var tmp_rhs_idx = try rhs.index(allocator, &.{ Index.initDim(@intCast(k)), Index.initDim(@intCast(j)) });
                defer tmp_rhs_idx.deinit();
                var tmp_mul = try tensor.mul(allocator, Tensor, tmp_lhs_idx, Tensor, tmp_rhs_idx);
                defer tmp_mul.deinit();
                try out.indexAdd(allocator, Tensor, tmp_mul, &.{ Index.initDim(@intCast(i)), Index.initDim(@intCast(j)) });
            }
        }
    }
    return out;
}

test "TensorBLASTest -> matmul" {
    const allocator = std.testing.allocator;
    defer tensor.deinit(); // deinit global singletons

    var i: Dim = 10;
    var j: Dim = 20;
    var k: Dim = 12;

    var a = try tensor.rand(allocator, &.{ i, j }, .f32);
    defer a.deinit();
    var b = try tensor.rand(allocator, &.{ j, k }, .f32);
    defer b.deinit();
    var ref = try matmulRef(allocator, a, b);
    defer ref.deinit();
    var res = try tensor.matmul(allocator, a, b, .None, .None);
    defer res.deinit();
    try std.testing.expect(try tensor.allClose(allocator, res, ref, 1e-5));

    var b_transposed = try tensor.transpose(allocator, b, &.{});
    defer b_transposed.deinit();
    var res2 = try tensor.matmul(allocator, a, b_transposed, .None, .Transpose);
    defer res2.deinit();
    try std.testing.expect(try tensor.allClose(allocator, res2, ref, 1e-5));

    var a_transposed = try tensor.transpose(allocator, a, &.{});
    defer a_transposed.deinit();
    var res3 = try tensor.matmul(allocator, a_transposed, b, .Transpose, .None);
    defer res3.deinit();
    try std.testing.expect(try tensor.allClose(allocator, res3, ref, 1e-5));
}

test "TensorBLASTest -> matmulShapes" {
    const allocator = std.testing.allocator;
    defer tensor.deinit(); // deinit global singletons

    // Matrix/vector/scalar multiplies
    var rand1 = try tensor.rand(allocator, &.{10}, .f32);
    defer rand1.deinit();
    var rand2 = try tensor.rand(allocator, &.{10}, .f32);
    defer rand2.deinit();
    var res1 = try tensor.matmul(allocator, rand1, rand2, .None, .None);
    defer res1.deinit();
    try std.testing.expect(tensor.shape.eql(try res1.shape(allocator), &.{1}));

    var res2 = try tensor.matmul(allocator, rand1, rand2, .Transpose, .None);
    defer res2.deinit();
    try std.testing.expect(tensor.shape.eql(try res2.shape(allocator), &.{1}));

    var res3 = try tensor.matmul(allocator, rand1, rand2, .Transpose, .Transpose);
    defer res3.deinit();
    try std.testing.expect(tensor.shape.eql(try res3.shape(allocator), &.{1}));

    var res4 = try tensor.matmul(allocator, rand1, rand2, .None, .Transpose);
    defer res4.deinit();
    try std.testing.expect(tensor.shape.eql(try res4.shape(allocator), &.{1}));

    var rand3 = try tensor.rand(allocator, &.{ 1, 10 }, .f32);
    defer rand3.deinit();
    var res5 = try tensor.matmul(allocator, rand3, rand1, .None, .None);
    defer res5.deinit();
    try std.testing.expect(tensor.shape.eql(try res5.shape(allocator), &.{1}));

    var rand4 = try tensor.rand(allocator, &.{1}, .f32);
    defer rand4.deinit();
    var res6 = try tensor.matmul(allocator, rand4, rand3, .None, .None);
    defer res6.deinit();
    try std.testing.expect(tensor.shape.eql(try res6.shape(allocator), &.{10}));

    var rand5 = try tensor.rand(allocator, &.{ 3, 4 }, .f32);
    defer rand5.deinit();
    var rand6 = try tensor.rand(allocator, &.{4}, .f32);
    defer rand6.deinit();
    var res7 = try tensor.matmul(allocator, rand5, rand6, .None, .None);
    defer res7.deinit();
    try std.testing.expect(tensor.shape.eql(try res7.shape(allocator), &.{3}));

    var rand7 = try tensor.rand(allocator, &.{5}, .f32);
    defer rand7.deinit();
    var rand8 = try tensor.rand(allocator, &.{ 5, 7 }, .f32);
    defer rand8.deinit();
    var res8 = try tensor.matmul(allocator, rand7, rand8, .None, .None);
    defer res8.deinit();
    try std.testing.expect(tensor.shape.eql(try res8.shape(allocator), &.{7}));

    try std.testing.expectError(
        error.ArrayFireError,
        tensor.matmul(
            allocator,
            rand4,
            rand1,
            .None,
            .None,
        ),
    );

    var rand9 = try tensor.rand(allocator, &.{3}, .f32);
    defer rand9.deinit();
    try std.testing.expectError(
        error.ArrayFireError,
        tensor.matmul(
            allocator,
            rand9,
            rand8,
            .None,
            .None,
        ),
    );

    // Batch matrix multiply
    var M: Dim = 10;
    var K: Dim = 12;
    var N: Dim = 14;
    var b2: Dim = 2;
    var b3: Dim = 4;

    var rand10 = try tensor.rand(allocator, &.{ M, K }, .f32);
    defer rand10.deinit();
    var rand11 = try tensor.rand(allocator, &.{ K, N }, .f32);
    defer rand11.deinit();
    var res9 = try tensor.matmul(allocator, rand10, rand11, .None, .None);
    defer res9.deinit();
    try std.testing.expect(tensor.shape.eql(try res9.shape(allocator), &.{ M, N }));

    var rand12 = try tensor.rand(allocator, &.{ M, K, b2 }, .f32);
    defer rand12.deinit();
    var rand13 = try tensor.rand(allocator, &.{ K, N, b2 }, .f32);
    defer rand13.deinit();
    var res10 = try tensor.matmul(allocator, rand12, rand13, .None, .None);
    defer res10.deinit();
    try std.testing.expect(tensor.shape.eql(try res10.shape(allocator), &.{ M, N, b2 }));

    var rand14 = try tensor.rand(allocator, &.{ M, K, b2, b3 }, .f32);
    defer rand14.deinit();
    var rand15 = try tensor.rand(allocator, &.{ K, N, b2, b3 }, .f32);
    defer rand15.deinit();
    var res11 = try tensor.matmul(allocator, rand14, rand15, .None, .None);
    defer res11.deinit();
    try std.testing.expect(tensor.shape.eql(try res11.shape(allocator), &.{ M, N, b2, b3 }));

    var res12 = try tensor.matmul(allocator, rand14, rand11, .None, .None);
    defer res12.deinit();
    try std.testing.expect(tensor.shape.eql(try res12.shape(allocator), &.{ M, N, b2, b3 }));

    var res13 = try tensor.matmul(allocator, rand10, rand15, .None, .None);
    defer res13.deinit();
    try std.testing.expect(tensor.shape.eql(try res13.shape(allocator), &.{ M, N, b2, b3 }));

    // Batch matrix multiply with transpose
    var rand16 = try tensor.rand(allocator, &.{ K, M }, .f32);
    defer rand16.deinit();
    var res14 = try tensor.matmul(allocator, rand16, rand11, .Transpose, .None);
    defer res14.deinit();
    try std.testing.expect(tensor.shape.eql(try res14.shape(allocator), &.{ M, N }));

    var rand17 = try tensor.rand(allocator, &.{ N, K }, .f32);
    defer rand17.deinit();
    var res15 = try tensor.matmul(allocator, rand10, rand17, .None, .Transpose);
    defer res15.deinit();
    try std.testing.expect(tensor.shape.eql(try res15.shape(allocator), &.{ M, N }));

    // b2 transpose
    var rand18 = try tensor.rand(allocator, &.{ K, M, b2 }, .f32);
    defer rand18.deinit();
    var res16 = try tensor.matmul(allocator, rand18, rand11, .Transpose, .None);
    defer res16.deinit();
    try std.testing.expect(tensor.shape.eql(try res16.shape(allocator), &.{ M, N, b2 }));

    var res17 = try tensor.matmul(allocator, rand12, rand17, .None, .Transpose);
    defer res17.deinit();
    try std.testing.expect(tensor.shape.eql(try res17.shape(allocator), &.{ M, N, b2 }));

    var res18 = try tensor.matmul(allocator, rand16, rand13, .Transpose, .None);
    defer res18.deinit();
    try std.testing.expect(tensor.shape.eql(try res18.shape(allocator), &.{ M, N, b2 }));

    var rand19 = try tensor.rand(allocator, &.{ N, K, b2 }, .f32);
    defer rand19.deinit();
    var res19 = try tensor.matmul(allocator, rand10, rand19, .None, .Transpose);
    defer res19.deinit();
    try std.testing.expect(tensor.shape.eql(try res19.shape(allocator), &.{ M, N, b2 }));

    var res20 = try tensor.matmul(allocator, rand18, rand13, .Transpose, .None);
    defer res20.deinit();
    try std.testing.expect(tensor.shape.eql(try res20.shape(allocator), &.{ M, N, b2 }));

    var res21 = try tensor.matmul(allocator, rand12, rand19, .None, .Transpose);
    defer res21.deinit();
    try std.testing.expect(tensor.shape.eql(try res21.shape(allocator), &.{ M, N, b2 }));

    // TODO: b2, b3 transpose
    var rand20 = try tensor.rand(allocator, &.{ K, M, b2, b3 }, .f32);
    defer rand20.deinit();
    var res22 = try tensor.matmul(allocator, rand20, rand11, .Transpose, .None);
    defer res22.deinit();
    try std.testing.expect(tensor.shape.eql(try res22.shape(allocator), &.{ M, N, b2, b3 }));

    var res23 = try tensor.matmul(allocator, rand14, rand17, .None, .Transpose);
    defer res23.deinit();
    try std.testing.expect(tensor.shape.eql(try res23.shape(allocator), &.{ M, N, b2, b3 }));

    var res24 = try tensor.matmul(allocator, rand16, rand15, .Transpose, .None);
    defer res24.deinit();
    try std.testing.expect(tensor.shape.eql(try res24.shape(allocator), &.{ M, N, b2, b3 }));

    var rand21 = try tensor.rand(allocator, &.{ N, K, b2, b3 }, .f32);
    defer rand21.deinit();
    var res25 = try tensor.matmul(allocator, rand10, rand21, .None, .Transpose);
    defer res25.deinit();
    try std.testing.expect(tensor.shape.eql(try res25.shape(allocator), &.{ M, N, b2, b3 }));

    var res26 = try tensor.matmul(allocator, rand20, rand15, .Transpose, .None);
    defer res26.deinit();
    try std.testing.expect(tensor.shape.eql(try res26.shape(allocator), &.{ M, N, b2, b3 }));

    var res27 = try tensor.matmul(allocator, rand14, rand21, .None, .Transpose);
    defer res27.deinit();
    try std.testing.expect(tensor.shape.eql(try res27.shape(allocator), &.{ M, N, b2, b3 }));

    var rand22 = try tensor.rand(allocator, &.{ 256, 200, 2 }, .f32);
    defer rand22.deinit();
    var rand23 = try tensor.rand(allocator, &.{ 256, 200, 2 }, .f32);
    defer rand23.deinit();
    var res28 = try tensor.matmul(allocator, rand22, rand23, .None, .Transpose);
    defer res28.deinit();
    try std.testing.expect(tensor.shape.eql(try res28.shape(allocator), &.{ 256, 256, 2 }));
}

test "TensorBaseTest -> Metadata" {
    const allocator = std.testing.allocator;
    defer tensor.deinit(); // deinit global singletons
    const s: Dim = 9;
    var t = try tensor.rand(allocator, &.{ s, s }, .f32);
    defer t.deinit();
    try std.testing.expect(try t.elements(allocator) == s * s);
    try std.testing.expect(!try t.isEmpty(allocator));
    try std.testing.expect(try t.bytes(allocator) == s * s * @sizeOf(f32));

    var e = try Tensor.initEmpty(allocator);
    defer e.deinit();
    try std.testing.expect(try e.elements(allocator) == 0);
    try std.testing.expect(try e.isEmpty(allocator));
    try std.testing.expect(!try e.isSparse(allocator));
    try std.testing.expect(!try e.isLocked(allocator));
}

// TODO: test "TensorBaseTest -> hasAdapter" {}

test "TensorBaseTest -> fromScalar" {
    const allocator = std.testing.allocator;
    defer tensor.deinit(); // deinit global singletons
    var a = try tensor.fromScalar(allocator, f32, 3.14, .f32);
    defer a.deinit();
    try std.testing.expect(try a.elements(allocator) == 1);
    try std.testing.expect(try a.ndim(allocator) == 0);
    try std.testing.expect(!try a.isEmpty(allocator));
    try std.testing.expect(tensor.shape.eql(try a.shape(allocator), &.{}));
}

// TODO: test "TensorBaseTest -> string" {}

test "TensorBaseTest -> AssignmentOperators" {
    const allocator = std.testing.allocator;
    defer tensor.deinit(); // deinit global singletons

    // TODO: add support to evaluate as comptime string?
    // e.g. try zt.eval("a += b", .{ a, b });

    var a = try tensor.full(allocator, &.{ 3, 3 }, f64, 1, .f32);
    defer a.deinit();
    try a.inPlaceAdd(allocator, f64, 2);
    var exp = try tensor.full(allocator, &.{ 3, 3 }, f64, 3, .f32);
    // no call to deinit; we'll manually do so w/o defer to allow reuse of var
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));
    exp.deinit();

    try a.inPlaceSub(allocator, f64, 1);
    exp = try tensor.full(allocator, &.{ 3, 3 }, f64, 2, .f32);
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));
    exp.deinit();

    try a.inPlaceMul(allocator, f64, 8);
    exp = try tensor.full(allocator, &.{ 3, 3 }, f64, 16, .f32);
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));
    exp.deinit();

    try a.inPlaceDiv(allocator, f64, 4);
    exp = try tensor.full(allocator, &.{ 3, 3 }, f64, 4, .f32);
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));
    exp.deinit();

    exp = try tensor.full(allocator, &.{ 4, 4 }, f64, 7, .f32);
    try a.assign(allocator, Tensor, exp);
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));

    var b = try tensor.initAssign(allocator, a);
    defer b.deinit();
    try std.testing.expect(try tensor.allClose(allocator, b, exp, 1e-5));
    exp.deinit();

    try a.assign(allocator, f64, 6);
    exp = try tensor.full(allocator, &.{ 4, 4 }, f64, 6, .f32);
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));
    exp.deinit();

    exp = try tensor.full(allocator, &.{ 5, 6, 7 }, f64, 8, .f32);
    defer exp.deinit();
    try a.assign(allocator, Tensor, exp);
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));
}

test "TensorBaseTest -> CopyOperators" {
    const allocator = std.testing.allocator;
    defer tensor.deinit(); // deinit global singletons

    var a = try tensor.full(allocator, &.{ 3, 3 }, f64, 1, .f32);
    defer a.deinit();
    var b = try tensor.initAssign(allocator, a);
    defer b.deinit();
    try a.inPlaceAdd(allocator, f32, 1);

    var exp = try tensor.full(allocator, &.{ 3, 3 }, f64, 1, .f32);
    try std.testing.expect(try tensor.allClose(allocator, b, exp, 1e-5));
    exp.deinit();
    exp = try tensor.full(allocator, &.{ 3, 3 }, f64, 2, .f32);
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));

    var c = try a.copy(allocator);
    defer c.deinit();
    try a.inPlaceAdd(allocator, f32, 1);
    try std.testing.expect(try tensor.allClose(allocator, c, exp, 1e-5));
    exp.deinit();
    exp = try tensor.full(allocator, &.{ 3, 3 }, f64, 3, .f32);
    defer exp.deinit();
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));
}

test "TensorBaseTest -> ConstructFromData" {
    const allocator = std.testing.allocator;
    defer tensor.deinit(); // deinit global singletons

    const val: f32 = 3;
    var vec = [_]f32{val} ** 100;
    var s: Shape = &.{ 10, 10 };

    var a = try Tensor.fromSlice(allocator, s, f32, &vec, .f32);
    defer a.deinit();
    var exp = try tensor.full(allocator, s, f64, val, .f32);
    defer exp.deinit();
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));

    var ascending: []const f32 = &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    var t = try Tensor.fromSlice(allocator, &.{ 3, 4 }, f32, ascending, .f32);
    defer t.deinit();
    try std.testing.expect(try t.dtype(allocator) == .f32);
    for (ascending, 0..) |v, i| {
        var t_idx = try t.index(allocator, &.{
            Index.initDim(@intCast(@mod(i, 3))),
            Index.initDim(@intCast(@divTrunc(i, 3))),
        });
        defer t_idx.deinit();
        try std.testing.expect(try t_idx.scalar(allocator, f32) == v);
    }

    // TODO: add fixtures/check stuff
    var intV = try Tensor.fromSlice(allocator, &.{3}, i32, &.{ 1, 2, 3 }, .s32);
    defer intV.deinit();
    try std.testing.expect(try intV.dtype(allocator) == .s32);
}

test "TensorBaseTest -> reshape" {
    const allocator = std.testing.allocator;
    defer tensor.deinit(); // deinit global singletons

    var a = try tensor.full(allocator, &.{ 4, 4 }, f64, 3, .f32);
    defer a.deinit();
    var b = try tensor.reshape(allocator, a, &.{ 8, 2 });
    defer b.deinit();
    try std.testing.expect(tensor.shape.eql(try b.shape(allocator), &.{ 8, 2 }));
    var exp = try tensor.reshape(allocator, b, &.{ 4, 4 });
    defer exp.deinit();
    try std.testing.expect(try tensor.allClose(allocator, a, exp, 1e-5));
    try std.testing.expectError(error.ArrayFireError, tensor.reshape(allocator, a, &.{}));
}

test "TensorBaseTest -> transpose" {
    const allocator = std.testing.allocator;
    defer tensor.deinit(); // deinit global singletons

    // TODO: expand to check els
    var og = try tensor.full(allocator, &.{ 3, 4 }, f64, 3, .f32);
    var exp = try tensor.full(allocator, &.{ 4, 3 }, f64, 3, .f32);
    var res = try tensor.transpose(allocator, og, &.{});
    og.deinit();
    try std.testing.expect(try tensor.allClose(allocator, res, exp, 1e-5));
    exp.deinit();
    res.deinit();

    og = try tensor.full(allocator, &.{ 4, 5, 6, 7 }, f64, 3, .f32);
    exp = try tensor.full(allocator, &.{ 6, 4, 5, 7 }, f64, 3, .f32);
    res = try tensor.transpose(allocator, og, &.{ 2, 0, 1, 3 });
    og.deinit();
    try std.testing.expect(try tensor.allClose(allocator, res, exp, 1e-5));
    exp.deinit();
    res.deinit();

    og = try tensor.rand(allocator, &.{ 3, 4, 5 }, .f32);
    try std.testing.expectError(
        error.ArrayFireTransposeFailed,
        tensor.transpose(allocator, og, &.{ 0, 1 }),
    );
    og.deinit();

    og = try tensor.rand(allocator, &.{ 2, 4, 6, 8 }, .f32);
    try std.testing.expectError(
        error.ArrayFireTransposeFailed,
        tensor.transpose(allocator, og, &.{ 1, 0, 2 }),
    );
    og.deinit();

    og = try tensor.rand(allocator, &.{ 2, 4, 6, 8 }, .f32);
    try std.testing.expectError(
        error.ArrayFireTransposeFailed,
        tensor.transpose(allocator, og, &.{ 1, 0, 2, 4 }),
    );
    og.deinit();

    og = try tensor.rand(allocator, &.{4}, .f32);
    try std.testing.expect(try tensor.allClose(
        allocator,
        try tensor.transpose(allocator, og, &.{}), // 1D tensor returns itself (no new alloc)
        og,
        1e-5,
    ));
    og.deinit();

    og = try tensor.rand(allocator, &.{ 5, 6, 7 }, .f32);
    res = try tensor.transpose(allocator, og, &.{});
    og.deinit();
    try std.testing.expect(tensor.shape.eql(try res.shape(allocator), &.{ 7, 6, 5 }));
    res.deinit();

    og = try tensor.rand(allocator, &.{ 5, 6, 1, 7 }, .f32);
    res = try tensor.transpose(allocator, og, &.{});
    og.deinit();
    try std.testing.expect(tensor.shape.eql(try res.shape(allocator), &.{ 7, 1, 6, 5 }));
    res.deinit();

    og = try tensor.rand(allocator, &.{ 1, 1 }, .f32);
    res = try tensor.transpose(allocator, og, &.{});
    og.deinit();
    try std.testing.expect(tensor.shape.eql(try res.shape(allocator), &.{ 1, 1 }));
    res.deinit();

    og = try tensor.rand(allocator, &.{ 7, 2, 1, 3 }, .f32);
    res = try tensor.transpose(allocator, og, &.{ 0, 2, 1, 3 });
    og.deinit();
    try std.testing.expect(tensor.shape.eql(try res.shape(allocator), &.{ 7, 1, 2, 3 }));
    res.deinit();
}
