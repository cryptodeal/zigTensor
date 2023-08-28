const std = @import("std");
const zt_base = @import("../../TensorBase.zig");
const af = @import("../../../backends/ArrayFire.zig");
const zt_types = @import("../../Types.zig");
const zt_idx = @import("../../Index.zig");
const zt_shape = @import("../../Shape.zig");
const zigrc = @import("zigrc");
const af_utils = @import("Utils.zig");

const fromZtData = af_utils.fromZtData;
const createAfIndexers = af_utils.createAfIndexers;
const AF_CHECK = af_utils.AF_CHECK;
const Tensor = zt_base.Tensor;
const Location = zt_base.Location;
const TensorBackendType = zt_base.TensorBackendType;
const Index = zt_idx.Index;
const IndexType = zt_idx.IndexType;
const Shape = zt_shape.Shape;
const Dim = zt_shape.Dim;
const DType = zt_types.DType;
const Arc = zigrc.Arc;

pub fn toArray(tensor: *Tensor) !af.af_array {
    if (tensor.backendType() != .ArrayFire) {
        std.log.err("toArray: tensor is not ArrayFire-backed\n", .{});
        return error.TensorNotArrayFireBacked;
    }
    return tensor.getAdapter(ArrayFireTensor).getHandle();
}

/// Tensor adapter for the internal ArrayFire array. Maps operations
/// expressed in zigTensor tensors to ArrayFire.
pub const ArrayFireTensor = struct {
    pub const ComponentTypeTag = enum { array, indexedArray };

    /// To be visited when this tensor is to be indexed. Indexes the underlying
    /// af.af_array, and returns the proxy to be used as a temporary lvalue.
    pub const IndexedArrayComponent = struct {
        isFlat_: bool,

        pub fn init(_isFlat: bool) IndexedArrayComponent {
            return .{ .isFlat = _isFlat };
        }

        pub fn get(inst: *const ArrayFireTensor) !af.af_array {
            var res: af.af_array = undefined;
            try AF_CHECK(af.af_index_gen(&res, try inst.getHandle(), @intCast(inst.numDims_), inst.indices_.?.ptr), @src());
            return res;
        }
    };

    /// To be visited when this tensor is holding an array without needing
    /// indexing. Passthrough - returns the array directly.
    pub const ArrayComponent = struct {
        pub fn get(inst: *const ArrayFireTensor) af.af_array {
            return inst.arrayHandle_.value.*;
        }
    };

    /// A pointer to the internal ArrayFire array. Shared
    /// amongst tensors that are shallow-copied.
    arrayHandle_: Arc(af.af_array),

    /// Indices in the event that this tensor is about to be indexed. Cleared
    /// the next time this array handle is acquired (see `getHandle`).
    indices_: ?[]af.af_index_t = null,

    /// Necessary to maintain the types of each index, as ArrayFire
    /// doesn't distinguish between an integer index literal and
    /// an af.af_seq of size one; both have slightly different
    /// behavior with zt.Tensor.
    indexTypes_: ?std.ArrayList(IndexType) = null,

    /// A zigTensor Shape that mirrors ArrayFire dims.
    ///
    /// N.B. this shape is only updated on calls to
    /// `ArrayFireTensor.shape()` so as to satisfy API
    /// requirements per returning a const reference.
    /// `af.af_get_numdims` should be used for internal
    /// computation where shape/dimensions are needed.
    shape_: Shape = undefined,

    /// The number of dimensions in this ArrayFire tensor
    /// that are "expected" per interoperability with other
    /// tensors. Because ArrayFire doesn't distinguish between
    /// singleton dimensions that are defaults and those that
    /// are explicitly specified, this must be explicitly tracked.
    ///
    /// the `zt.Tensor` default tensor shape is {0} - the default
    /// number of `num_dims` is thus 1. Scalars have `num_dims` == 0.
    numDims_: usize = 1,

    /// The TensorBackendType enum value for the ArrayFireTensor implementation.
    tensorBackendType_: TensorBackendType = .ArrayFire,

    // An interface to visit when getting an array handle. Indexes lazily
    // because we can't store an af::array::proxy as an lvalue. See getHandle().
    handle_: union(ComponentTypeTag) { array: ArrayComponent, indexedArray: IndexedArrayComponent },

    /// Initializes a new ArrayFireTensor that will be lazily indexed.
    /// Intended for internal use only.
    pub fn initLazyIndexing(allocator: std.mem.Allocator, handle: Arc(af.af_array), af_indices: std.ArrayList(*Index), index_types: std.ArrayList(IndexType), num_dims: usize, is_flat: bool) !*ArrayFireTensor {
        _ = is_flat;
        _ = num_dims;
        _ = index_types;
        _ = af_indices;
        _ = handle;
        _ = allocator;
    }

    /// Initializes a new ArrayFireTensor from an ArrayFire array
    /// handle without copying the handle. Used for creating
    /// gauranteed shallow-copies.
    pub fn initFromShared(allocator: std.mem.Allocator, arr: Arc(af.af_array), num_dims: usize) !*ArrayFireTensor {
        _ = num_dims;
        _ = arr;
        _ = allocator;
    }

    pub fn initFromArray(allocator: std.mem.Allocator, arr: af.af_array, num_dims: usize) !*ArrayFireTensor {
        _ = allocator;
        _ = arr;
        _ = num_dims;
    }

    pub fn initRaw(allocator: std.mem.Allocator) !*ArrayFireTensor {
        var self = try allocator.create(ArrayFireTensor);
        self.* = .{};
        return self;
    }

    pub fn init(allocator: std.mem.Allocator, shape: *const Shape, data_type: DType, ptr: ?*const anyopaque) !*ArrayFireTensor {
        var self = try allocator.create(ArrayFireTensor);
        self.* = .{
            .arrayHandle_ = try Arc(af.af_array).init(allocator, try fromZtData(shape, ptr, data_type)),
            .numDims_ = shape.ndim(),
            .handle_ = .{ .array = ArrayComponent{} },
        };
        return self;
    }

    pub fn initComplex(allocator: std.mem.Allocator, n_rows: Dim, n_cols: Dim, values: *Tensor, row_idx: *Tensor, col_idx: *Tensor) !*ArrayFireTensor {
        _ = col_idx;
        _ = row_idx;
        _ = values;
        _ = n_cols;
        _ = n_rows;
        _ = allocator;
    }

    pub fn getHandle(self: *ArrayFireTensor) !af.af_array {
        _ = self;
    }

    pub fn numDims(self: *ArrayFireTensor) usize {
        return self.numDims_;
    }

    // TODO: pub fn clone

    // TODO: pub fn copy

    // TODO: pub fn shallowCopy

    pub fn backendType(_: *ArrayFireTensor) TensorBackendType {
        return .ArrayFire;
    }

    // TODO: pub fn backend

    // TODO: pub fn shape(self: *ArrayFireTensor) !Shape {}

    // TODO: pub fn type()

    // TODO: pub fn isSparse()

    // TODO: pub fn afHandleType()

    // TODO: pub fn location()

    // TODO: pub fn scalar()

    // TODO: pub fn device()

    // TODO: pub fn host()

    // TODO: pub fn unlock()

    // TODO: pub fn isLocked()

    // TODO: pub fn isContiguous()

    // TODO: pub fn strides()

    // TODO: pub fn stream()

    // TODO: pub fn astype()

    // TODO: pub fn index()

    // TODO: pub fn flatten()

    // TODO: pub fn flat()

    // TODO: pub fn asContiguousTensor()

    pub fn setContext(self: *ArrayFireTensor, comptime T: type, context: af.af_array) !void {
        // no-op
        _ = self;
        _ = T;
        _ = context;
    }

    pub fn getContext(self: *ArrayFireTensor, comptime T: type) ?T {
        _ = self;
        // no-op
        return null;
    }

    // TODO: pub fn toString()
};
