const std = @import("std");
const zt_base = @import("../../TensorBase.zig");
const af = @import("../../../bindings/af/ArrayFire.zig");
const zt_types = @import("../../Types.zig");
const zt_idx = @import("../../Index.zig");
const zt_shape = @import("../../Shape.zig");
const zigrc = @import("zigrc");
const build_options = @import("build_options");
const runtime = @import("../../../runtime/runtime.zig");

const assert = std.debug.assert;
const TensorAdapterBase = @import("../../TensorAdapter.zig").TensorAdapterBase;
const ArrayFireBackend = @import("ArrayFireBackend.zig").ArrayFireBackend;
const TensorBackend = @import("../../TensorBackend.zig").TensorBackend;
const condenseIndices = @import("Utils.zig").condenseIndices;
const ZT_BACKEND_CUDA = build_options.ZT_BACKEND_CUDA;
const ZT_BACKEND_CPU = build_options.ZT_BACKEND_CPU;
const ZT_BACKEND_OPENCL = build_options.ZT_BACKEND_OPENCL;
const Tensor = zt_base.Tensor;
const StorageType = zt_base.StorageType;
const Location = zt_base.Location;
const TensorBackendType = zt_base.TensorBackendType;
const Index = zt_idx.Index;
const IndexType = zt_idx.IndexType;
const Shape = zt_shape.Shape;
const Dim = zt_shape.Dim;
const DType = zt_types.DType;
const Arc = zigrc.Arc;

const GetHandleErrors = std.mem.Allocator.Error || af.Errors;

pub fn toArray(allocator: std.mem.Allocator, tensor: Tensor) !*af.Array {
    if (tensor.backendType() != .ArrayFire) {
        std.log.err("toArray: tensor is not ArrayFire-backed\n", .{});
        return error.TensorNotArrayFireBacked;
    }
    return tensor.getAdapter(ArrayFireTensor).getHandle(allocator);
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

        pub fn get(_: *const IndexedArrayComponent, allocator: std.mem.Allocator, inst: *ArrayFireTensor) !*af.Array {
            return af.ops.indexGen(
                allocator,
                try inst.getHandle(allocator),
                @intCast(inst.numDims_),
                inst.indices_.?,
            );
        }
    };

    /// To be visited when this tensor is holding an array without needing
    /// indexing. Passthrough - returns the array directly.
    pub const ArrayComponent = struct {
        pub fn get(_: *const ArrayComponent, inst: *const ArrayFireTensor) *af.Array {
            return inst.arrayHandle_.value.*;
        }
    };

    pub const HandleUnion = union(ComponentTypeTag) { array: ArrayComponent, indexedArray: IndexedArrayComponent };

    /// A pointer to the internal ArrayFire array. Shared
    /// amongst tensors that are shallow-copied.
    arrayHandle_: Arc(*af.Array) = undefined,

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
    handle_: HandleUnion = .{ .array = ArrayComponent{} },

    allocator: std.mem.Allocator,

    /// Initializes a new ArrayFireTensor that will be lazily indexed.
    /// Intended for internal use only.
    pub fn initLazyIndexing(
        allocator: std.mem.Allocator,
        handle: Arc(*af.Array),
        af_indices: []af.af_index_t,
        index_types: std.ArrayList(IndexType),
        num_dims: usize,
        is_flat: bool,
    ) !*ArrayFireTensor {
        var self = try allocator.create(ArrayFireTensor);
        self.* = .{
            .arrayHandle_ = handle,
            .indices_ = af_indices,
            .indexTypes_ = index_types,
            .handle_ = .{ .indexedArray = IndexedArrayComponent.init(is_flat) },
            .numDims_ = num_dims,
            .allocator = allocator,
        };
        return self;
    }

    /// Initializes a new ArrayFireTensor from an ArrayFire array
    /// handle without copying the handle. Used for creating
    /// gauranteed shallow-copies.
    pub fn initFromShared(allocator: std.mem.Allocator, arr: Arc(*af.Array), num_dims: usize) !*ArrayFireTensor {
        var self = try allocator.create(ArrayFireTensor);
        var afDims = try arr.value.*.getDims();
        self.* = .{
            .arrayHandle_ = arr,
            .numDims_ = num_dims,
            .allocator = allocator,
            .shape_ = try afDims.toZtShape(allocator, afDims.ndims()),
        };
        return self;
    }

    pub fn initFromArray(allocator: std.mem.Allocator, arr: *af.Array, num_dims: usize) !*ArrayFireTensor {
        var self = try allocator.create(ArrayFireTensor);
        var afDims = try arr.getDims();
        self.* = .{
            .arrayHandle_ = try Arc(*af.Array).init(allocator, arr),
            .numDims_ = num_dims,
            .shape_ = try afDims.toZtShape(allocator, afDims.ndims()),
            .allocator = allocator,
        };
        return self;
    }

    pub fn initRaw(allocator: std.mem.Allocator) !*ArrayFireTensor {
        return allocator.create(ArrayFireTensor);
    }

    pub fn init(
        allocator: std.mem.Allocator,
        _shape: Shape,
        data_type: DType,
        ptr: ?*anyopaque,
        loc: Location,
    ) !*ArrayFireTensor {
        var self = try allocator.create(ArrayFireTensor);
        var arr: *af.Array = undefined;
        switch (loc) {
            .Host => arr = try af.Array.initFromPtr(
                allocator,
                ptr,
                @intCast(_shape.ndim()),
                try _shape.toAfDims(),
                data_type.toAfDtype(),
            ),
            .Device => arr = try af.Array.initFromDevicePtr(
                allocator,
                ptr,
                @intCast(_shape.ndim()),
                try _shape.toAfDims(),
                data_type.toAfDtype(),
            ),
        }
        self.* = .{
            .arrayHandle_ = try Arc(*af.Array).init(allocator, arr),
            .numDims_ = _shape.ndim(),
            .shape_ = _shape,
            .allocator = allocator,
        };
        return self;
    }

    pub fn initSparse(allocator: std.mem.Allocator, n_rows: Dim, n_cols: Dim, values: Tensor, row_idx: Tensor, col_idx: Tensor, storage_type: StorageType) !*ArrayFireTensor {
        _ = storage_type;
        _ = col_idx;
        _ = row_idx;
        _ = values;
        _ = n_cols;
        _ = n_rows;
        _ = allocator;
    }

    pub fn deinit(self: *ArrayFireTensor) void {
        self.arrayHandle_.releaseWithFn(af.Array.deinit);
        // remove indices
        if (self.indices_ != null) {
            af.ops.releaseIndexers(@constCast(self.indices_.?)) catch unreachable;
        }
        // remove IndexTypes
        if (self.indexTypes_ != null) {
            self.indexTypes_.?.deinit();
        }
        self.shape_.deinit();
        self.allocator.destroy(self);
    }

    pub fn getHandle(self: *ArrayFireTensor, allocator: std.mem.Allocator) GetHandleErrors!*af.Array {
        if (@as(ComponentTypeTag, self.handle_) != .array) {
            const idxComp: IndexedArrayComponent = self.handle_.indexedArray;
            var oldHandle = self.arrayHandle_;
            defer oldHandle.releaseWithFn(af.Array.deinit);
            var idxTypes: ?[]IndexType = if (self.indexTypes_ != null) self.indexTypes_.?.items else null;
            self.arrayHandle_ = try Arc(*af.Array).init(
                allocator,
                try condenseIndices(
                    allocator,
                    try idxComp.get(allocator, self),
                    false,
                    idxTypes,
                    idxComp.isFlat_,
                ),
            );
            // Clear state
            self.handle_ = .{ .array = ArrayComponent{} }; // set to passthrough
            // remove indices
            if (self.indices_ != null) {
                try af.ops.releaseIndexers(@constCast(self.indices_.?));
                self.indices_ = null;
            }
            // remove IndexTypes
            if (idxTypes != null) {
                self.indexTypes_.?.deinit();
                self.indexTypes_ = null;
            }
        }
        return self.arrayHandle_.value.*;
    }

    pub fn numDims(self: *ArrayFireTensor) usize {
        return self.numDims_;
    }

    pub fn clone(self: *ArrayFireTensor, allocator: std.mem.Allocator) !TensorAdapterBase {
        var arr = try self.getHandle(allocator); // increment internal AF refcount
        return TensorAdapterBase.init(try ArrayFireTensor.initFromArray(
            allocator,
            arr,
            self.numDims(),
        ));
    }

    pub fn copy(self: *ArrayFireTensor, allocator: std.mem.Allocator) !Tensor {
        _ = try self.getHandle(allocator); // if this tensor was a view, run indexing and promote
        var copiedArr = try self.arrayHandle_.value.*.copy(allocator);
        return Tensor.init(TensorAdapterBase.init(try ArrayFireTensor.initFromArray(allocator, copiedArr, self.numDims())));
    }

    pub fn shallowCopy(self: *ArrayFireTensor, allocator: std.mem.Allocator) !Tensor {
        _ = try self.getHandle(allocator); // if this tensor was a view, run indexing and promote
        var sharedArr = self.arrayHandle_.retain();
        return Tensor.init(TensorAdapterBase.init(try ArrayFireTensor.initFromShared(allocator, sharedArr, self.numDims())));
    }

    pub fn backendType(_: *ArrayFireTensor) TensorBackendType {
        return .ArrayFire;
    }

    pub fn backend(_: *const ArrayFireTensor, allocator: std.mem.Allocator) !TensorBackend {
        return TensorBackend.init(try ArrayFireBackend.getInstance(allocator));
    }

    pub fn shape(self: *ArrayFireTensor, allocator: std.mem.Allocator) !Shape {
        // Update the Shape in-place. Doesn't change any underlying data; only the
        // mirrored Shape metadata.
        const afDims: af.Dim4 = try (try self.getHandle(allocator)).getDims();
        try afDims.toZtShapeRaw(self.numDims(), &self.shape_);
        return self.shape_;
    }

    pub fn dtype(self: *ArrayFireTensor, allocator: std.mem.Allocator) !DType {
        var arr = try self.getHandle(allocator);
        var afType = try arr.getType();
        return afType.toZtDType();
    }

    pub fn isSparse(self: *ArrayFireTensor, allocator: std.mem.Allocator) !bool {
        var arr = try self.getHandle(allocator);
        return arr.isSparse();
    }

    // TODO: pub fn afHandleType()

    pub fn location(self: *ArrayFireTensor, allocator: std.mem.Allocator) !Location {
        return switch (try af.ops.getBackendId(try self.getHandle(allocator))) {
            .CUDA, .OpenCL => .Device,
            .CPU => .Host,
            else => error.UnsupportedBackend,
        };
    }

    pub fn scalar(self: *ArrayFireTensor, allocator: std.mem.Allocator, out: ?*anyopaque) !void {
        var arr = try self.getHandle(allocator);
        try af.AF_CHECK(af.af_get_scalar(out, arr.array_), @src());
    }

    pub fn device(self: *ArrayFireTensor, allocator: std.mem.Allocator, out: *?*anyopaque) !void {
        var arr = try self.getHandle(allocator);
        try af.AF_CHECK(af.af_get_device_ptr(out, arr.array_), @src());
    }

    pub fn host(self: *ArrayFireTensor, allocator: std.mem.Allocator, out: ?*anyopaque) !void {
        var arr = try self.getHandle(allocator);
        try arr.getDataPtr(out);
    }

    pub fn unlock(self: *ArrayFireTensor, allocator: std.mem.Allocator) !void {
        var arr = try self.getHandle(allocator);
        try arr.unlock();
    }

    pub fn isLocked(self: *ArrayFireTensor, allocator: std.mem.Allocator) !bool {
        var arr = try self.getHandle(allocator);
        return arr.isLocked();
    }

    pub fn isContiguous(self: *ArrayFireTensor, allocator: std.mem.Allocator) !bool {
        var arr = try self.getHandle(allocator);
        return arr.isLinear();
    }

    pub fn strides(self: *ArrayFireTensor, allocator: std.mem.Allocator) !Shape {
        var arr = try self.getHandle(allocator);
        var afStrides = try arr.getStrides();
        return afStrides.toZtShape(allocator, self.numDims());
    }

    pub fn stream(self: *ArrayFireTensor, allocator: std.mem.Allocator) !runtime.Stream {
        // TODO indexing is unlikely to change the stream associated with a tensor.
        // But if it can, we need to call `getHandle()` here.
        var bknd = try ArrayFireBackend.getInstance(allocator);
        return bknd.getStreamOfArray(allocator, self.arrayHandle_.value.*);
    }

    pub fn astype(self: *ArrayFireTensor, allocator: std.mem.Allocator, dType: DType) !Tensor {
        var arr = try self.getHandle(allocator);
        var convertedArr = try arr.cast(allocator, dType.toAfDtype());
        return Tensor.init(TensorAdapterBase.init(try ArrayFireTensor.initFromArray(allocator, convertedArr, self.numDims())));
    }

    pub fn index(self: *ArrayFireTensor, allocator: std.mem.Allocator, indices: std.ArrayList(Index)) !Tensor {
        if (indices.items.len > @as(usize, @intCast(af.AF_MAX_DIMS))) {
            std.log.err("ArrayFire-backed tensor was indexed with > 4 elements: ArrayFire tensors support up to 4 dimensions.\n", .{});
            return error.IndicesExceedMaxDims;
        }

        // TODO: vet and stress test this a lot more/add proper support for
        // multi-tensor
        // If indexing by a single element and it's a tensor with the same number of
        // indices as the array being indexed, do a flat index as this is probably a
        // filter-based index (for example: a(a < 5)).
        const completeTensorIndex = indices.items.len == 1 and indices.items[0].idxType() == .Tensor and try indices.items[0].Tensor.elements() == @as(usize, @intCast((try self.getHandle()).getElements()));
        var afIndices = try af.ops.createIndexers(); // this creates implicit spans for up to maxDims
        if (completeTensorIndex) {
            try af.ops.setSeqIndexer(afIndices, &af.af_seq{ .begin = 0, .end = 0, .step = 1 }, 0, false);
        }

        if (indices.items.len > afIndices.len) {
            std.log.err("ArrayFireTensor.index internal error - passed indices is larger than the number of af indices.\n", .{});
            return error.PassedIndicesLargerThanAfIndices;
        }

        // Fill in corresponding index types for each af index
        var indexTypes = try std.ArrayList(IndexType).initCapacity(allocator, afIndices.len);
        var i: usize = 0;
        while (i < indices.items.len) : (i += 1) {
            indexTypes.appendAssumeCapacity(indices.items[i].idxType());
            afIndices[i] = af.ops.ztToAfIndex(indices.items[i]);
        }

        // If we're adding implicit spans, fill those indexTypes in
        while (i < afIndices.len) : (i += 1) {
            indexTypes.appendAssumeCapacity(.Span);
        }

        _ = try self.getHandle(allocator); // if this tensor was a view, run indexing and promote

        assert(afIndices.len == indexTypes.items.len);
        // Compute numDims for the new Tensor
        var newNumDims = self.numDims();

        if (completeTensorIndex) {
            // TODO/FIXME: compute this based on the number of els in the indexing
            // tensor(s)
            newNumDims = 1;
        } else {
            for (indexTypes.items) |iType| {
                if (iType == .Literal) newNumDims -= 1;
            }
        }
        newNumDims = @max(newNumDims, 1);

        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initLazyIndexing(
                    allocator,
                    self.arrayHandle_.retain(),
                    afIndices,
                    indexTypes,
                    newNumDims,
                    false,
                ),
            ),
        );
    }

    pub fn flatten(self: *ArrayFireTensor, allocator: std.mem.Allocator) !Tensor {
        var arr = try self.getHandle(allocator);
        var flattenedArr = try arr.flat(allocator);
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    flattenedArr,
                    1,
                ),
            ),
        );
    }

    pub fn flat(self: *ArrayFireTensor, allocator: std.mem.Allocator, idx: Index) !Tensor {
        _ = try self.getHandle(allocator); // if this tensor was a view, run indexing and promote
        // Return a lazy indexing operation. Indexing with a single index on an
        // ArrayFire tensor (with a type that is not an af::array) ends up doing
        // flat indexing, so all index assignment operators will work as they are.
        var indices = try af.ops.createIndexers();
        indices[0] = af.ops.ztToAfIndex(idx);
        var index_types = std.ArrayList(IndexType).init(allocator);
        try index_types.append(idx.idxType());
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initLazyIndexing(
                    allocator,
                    self.arrayHandle_.retain(),
                    indices,
                    index_types,
                    1,
                    true,
                ),
            ),
        );
    }

    pub fn asContiguousTensor(self: *ArrayFireTensor, allocator: std.mem.Allocator) !Tensor {
        if (try self.isContiguous(allocator)) {
            _ = try self.getHandle(allocator);
            var other = self.arrayHandle_.retain();
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromShared(
                        allocator,
                        other,
                        self.numDims(),
                    ),
                ),
            );
        }

        var arr = try self.getHandle(allocator);
        var newDims = af.Dim4{};
        newDims.dims[0] = @as(af.dim_t, @intCast(try arr.getElements()));
        var linearArr = try af.Array.initHandle(allocator, 1, newDims, try arr.getType());
        var indices = try af.ops.createIndexers();
        try af.ops.setSeqIndexer(indices, &af.af_span, 0, false);
        try af.ops.copy(linearArr, arr, indices);
        try af.ops.releaseIndexers(indices);
        return Tensor.init(TensorAdapterBase.init(try ArrayFireTensor.initFromArray(allocator, linearArr, 1)));
    }

    pub fn setContext(self: *ArrayFireTensor, context: ?*anyopaque) !void {
        // no-op
        _ = self;
        _ = context;
    }

    pub fn getContext(self: *ArrayFireTensor) ?*anyopaque {
        _ = self;
        // no-op
        return null;
    }

    pub fn toString(self: *ArrayFireTensor, allocator: std.mem.Allocator) ![]const u8 {
        var arr = try self.getHandle(allocator);
        return arr.toString("ArrayFireTensor", 4, true);
    }

    fn adjustInPlaceOperandDims(self: *ArrayFireTensor, allocator: std.mem.Allocator, operand: *Tensor) !*af.Array {
        // optimstically try to moddims the operand's singleton dims
        const preIdxDims: af.Dim4 = try self.arrayHandle_.value.*.getDims();
        const operandArr: *af.Array = toArray(allocator, operand);

        // dims to which to try to modify the input if doing indexing
        var newDims: af.Dim4 = undefined;
        const operandDims: af.Dim4 = try operandArr.getDims();

        if (self.indices_ != null and self.indices_.?.len == 1) {
            // This case is only reachable via tensor-based indexing or indexing on a
            // tensor via Tensor.flat()
            if (self.numDims_ != 1) {
                std.log.err("ArrayFireTensor.adjustInPlaceOperandDims index size was 1 but tensor has greater than 1 dimension.\n", .{});
                return error.IndexSizeTensorDimsMismatch;
            }
        } else if (self.indices_ != null and self.indices_.?.len > 0) {
            // All other indexing operations
            const indices: []af.af_index_t = self.indices_.?;
            const indexTypes: []IndexType = self.indexTypes_.?.items;
            if (indices.len != indexTypes.len) {
                std.log.err("ArrayFireTensor.adjustInPlaceOperandDims - passed indices and indexTypes are of different sizes.\n", .{});
                return error.IndicesAndIndexTypesSizeMismatch;
            }

            // If the dimensions being indexed are 1 and collapsing them yields the same
            // shape as the operand, we can safely moddims, the operand, else there's a
            // dimension mismatch. For example:
            // {4, 5, 6, 7}(span, span, 5) --> {4, 5, 1, 7} --> {4, 5, 7}
            // {4, 5, 6, 7}(4) --> {1, 5, 1, 7} --> {5, 1, 7, 1}
            var indicesToCompress = std.ArrayList(u32).init(allocator);
            defer indicesToCompress.deinit();
            for (0..indices.len) |i| {
                // If an index literal, the corresponding dimension in the indexed array
                // is 1, then we indexed the input to a dim of 1, so we can condense that
                // index
                if (indexTypes[i] == .Literal) {
                    try indicesToCompress.append(i);
                }
            }

            var condensedDims = af.Dim4{};
            var postIdxDims = preIdxDims;
            var outDimIdx: usize = 0;
            var compressIdx: usize = 0;
            for (0..@as(usize, @intCast(af.AF_MAX_DIMS))) |i| {
                if (compressIdx < indicesToCompress.items.len and @as(u32, @intCast(i)) == indicesToCompress.items[compressIdx]) {
                    compressIdx += 1;
                    postIdxDims.dims[i] = 1;
                } else {
                    // Use the size of the dim post-indexing. Span uses the preIdx dim
                    // and literals are pushed to 1.
                    if (i < indexTypes.len) {
                        if (indexTypes[i] == .Tensor) {
                            var size: af.dim_t = undefined;
                            try af.AF_CHECK(af.af_get_elements(&size, indices[i].idx.arr));
                            postIdxDims.dims[i] = size;
                        } else if (indexTypes[i] == .Range) {
                            const seq = indices[i].idx.seq;
                            postIdxDims.dims[i] = if (seq.step == 0) 0 else @as(af.dim_t, @intCast(@fabs((seq.end - seq.begin) / seq.step))) + 1;
                        } else if (indexTypes[i] == .Literal) {
                            postIdxDims.dims[i] = 1;
                        }
                    }
                    condensedDims.dims[outDimIdx] = postIdxDims[i];
                    outDimIdx += 1;
                }
            }

            // Can modify the operand to work with the proxy or array input only by
            // removing singleton dimensions
            if (std.meta.eql(af.dim_t, &condensedDims.dims, &operandDims.dims)) {
                newDims = postIdxDims;
            } else {
                std.log.err("ArrayFireTensor.adjustInPlaceOperandDims: can't apply operation in-place to indexed ArrayFireTensor - dimensions don't match.", .{});
                return error.CannotApplyOperationDimsMismatch;
            }
        } else {
            // No indexing so no change in dimensions required
            newDims = operandDims;
        }

        // moddims involves an eval. This will be fixed in AF 3.8.1/3.8.2
        const doModdims = !std.meta.eql(af.dim_t, &(try operandArr.getDims()).dims, &newDims.dims);
        if (doModdims) {
            return operandArr.modDims(allocator, newDims.ndims(), newDims);
        } else {
            return operandArr;
        }
    }
};
