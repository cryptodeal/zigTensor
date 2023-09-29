const std = @import("std");
const tensor = @import("tensor.zig");

const Dim = tensor.Dim;
const DType = tensor.DType;
const dtypeTraits = tensor.dtypeTraits;
const Index = tensor.Index;
const Shape = tensor.Shape;
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
        var shape_ = try self.shape(allocator);
        return shape_.elements();
    }

    pub fn dim(self: *const Tensor, allocator: std.mem.Allocator, dimension: usize) !Dim {
        var shape_ = try self.shape(allocator);
        return shape_.dim(dimension);
    }

    pub fn ndim(self: *const Tensor, allocator: std.mem.Allocator) !usize {
        var shape_ = try self.shape(allocator);
        return shape_.ndim();
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

    pub fn index(self: *const Tensor, allocator: std.mem.Allocator, indices: []Index) !Tensor {
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
            var used_shape = try self.shape(allocator);
            rhsTensor = try bknd.full(allocator, &used_shape, T, rhs, try self.dtype(allocator));
            rhsTensorInit = true;
        }
        var check = [_]Tensor{ self.*, rhsTensor };
        try ztTensorBackendsMatch(@src().fn_name, &check);
        return bknd.assign(allocator, self.*, rhsTensor);
    }

    pub fn indexAssign(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, indices: []Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.indexAssign(allocator, self.*, T, rhs, indices);
    }

    pub fn indexAdd(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, indices: []Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.indexAdd(allocator, self.*, T, rhs, indices);
    }

    pub fn indexSub(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, indices: []Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.indexSub(allocator, self.*, T, rhs, indices);
    }

    pub fn indexMul(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, indices: []Index) !void {
        var bknd = try self.backend(allocator);
        return bknd.indexMul(allocator, self.*, T, rhs, indices);
    }

    pub fn indexDiv(self: *const Tensor, allocator: std.mem.Allocator, comptime T: type, rhs: T, indices: []Index) !void {
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
            rhsTensor = try bknd.full(allocator, &used_shape, T, rhs, try self.dtype(allocator));
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
            rhsTensor = try bknd.full(allocator, &used_shape, T, rhs, try self.dtype(allocator));
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
            rhsTensor = try bknd.full(allocator, &used_shape, T, rhs, try self.dtype(allocator));
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
            rhsTensor = try bknd.full(allocator, &used_shape, T, rhs, try self.dtype(allocator));
            rhsTensorInit = true;
        }
        var check = [_]Tensor{ self.*, rhsTensor };
        try ztTensorBackendsMatch(@src().fn_name, &check);
        return bknd.inPlaceDiv(allocator, self.*, rhsTensor);
    }
};
