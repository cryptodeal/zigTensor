const std = @import("std");
const zt_base = @import("../../TensorBase.zig");
const af = @import("../../../bindings/af/ArrayFire.zig");
const zt_types = @import("../../Types.zig");
const zt_idx = @import("../../Index.zig");
const zt_shape = @import("../../Shape.zig");
const zigrc = @import("zigrc");
const build_options = @import("build_options");
const runtime = @import("../../../runtime/runtime.zig");
const af_utils = @import("Utils.zig");

const growCapacity = @import("../utils.zig").growCapacity;
const assert = std.debug.assert;
const deinit = @import("../../Init.zig").deinit;
const TensorAdapterBase = @import("../../TensorAdapter.zig").TensorAdapterBase;
const ArrayFireBackend = @import("ArrayFireBackend.zig").ArrayFireBackend;
const TensorBackend = @import("../../TensorBackend.zig").TensorBackend;
const condenseIndices = af_utils.condenseIndices;
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
        std.log.debug("toArray: tensor is not ArrayFire-backed\n", .{});
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

        pub fn init(isFlat: bool) IndexedArrayComponent {
            return .{ .isFlat_ = isFlat };
        }

        pub fn get(self: *const IndexedArrayComponent, allocator: std.mem.Allocator, inst: *ArrayFireTensor) !*af.Array {
            var arr = inst.arrayHandle_.value.*;
            var n_dims: i64 = if (!self.isFlat_) @intCast(try arr.getNumDims()) else 1;
            return af.ops.indexGen(
                allocator,
                arr,
                n_dims,
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
    shape_: []Dim = &[_]Dim{},

    capacity: usize = 0,

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

    // utils for modifying the underlying shape (reduce allocations)

    /// Extend the shape by 1 element. Allocates more memory as necessary.
    /// Invalidates pointers if additional memory is needed.
    pub fn append(self: *ArrayFireTensor, item: Dim) !void {
        const new_item_ptr = try self.addOne();
        new_item_ptr.* = item;
    }

    /// Append the slice of dims to the list. Allocates more
    /// memory as necessary.
    /// Invalidates pointers if additional memory is needed.
    pub fn appendSlice(self: *ArrayFireTensor, items: []const Dim) !void {
        try self.ensureUnusedCapacity(items.len);
        self.appendSliceAssumeCapacity(items);
    }

    /// Adjust the shape's length to `new_len`.
    /// Does not initialize added dims if any.
    /// Invalidates pointers if additional memory is needed.
    pub fn resize(self: *ArrayFireTensor, new_len: usize) !void {
        try self.ensureTotalCapacity(new_len);
        self.shape_.len = new_len;
    }

    /// Modify the array so that it can hold at least `additional_count` **more** items.
    /// Invalidates pointers if additional memory is needed.
    pub fn ensureUnusedCapacity(self: *ArrayFireTensor, additional_count: usize) !void {
        return self.ensureTotalCapacity(self.items.len + additional_count);
    }

    /// Modify the shape so that it can hold at least `new_capacity` dims.
    /// Invalidates pointers if additional memory is needed.
    pub fn ensureTotalCapacity(self: *ArrayFireTensor, new_capacity: usize) !void {
        if (self.capacity >= new_capacity) return;

        const better_capacity = growCapacity(self.capacity, new_capacity);
        return self.ensureTotalCapacityPrecise(better_capacity);
    }

    /// Modify the shape so that it can hold `new_capacity` dims.
    /// Like `ensureTotalCapacity`, but the resulting capacity is guaranteed
    /// to be equal to `new_capacity`.
    /// Invalidates pointers if additional memory is needed.
    pub fn ensureTotalCapacityPrecise(self: *ArrayFireTensor, new_capacity: usize) !void {
        if (self.capacity >= new_capacity) return;

        // Here we avoid copying allocated but unused bytes by
        // attempting a resize in place, and falling back to allocating
        // a new buffer and doing our own copy. With a realloc() call,
        // the allocator implementation would pointlessly copy our
        // extra capacity.
        const old_memory = self.allocatedSlice();
        if (self.allocator.resize(old_memory, new_capacity)) {
            self.capacity = new_capacity;
        } else {
            const new_memory = try self.allocator.alignedAlloc(Dim, null, new_capacity);
            @memcpy(new_memory[0..self.shape_.len], self.shape_);
            self.allocator.free(old_memory);
            self.shape_.ptr = new_memory.ptr;
            self.capacity = new_memory.len;
        }
    }

    /// Returns a slice of all the dims plus the extra capacity, whose memory
    /// contents are `undefined`.
    pub fn allocatedSlice(self: ArrayFireTensor) []Dim {
        // `shape_.len` is the length, not the capacity.
        return self.shape_.ptr[0..self.capacity];
    }

    /// Increase length by 1, returning pointer to the new item.
    /// The returned pointer becomes invalid when the list resized.
    pub fn addOne(self: *ArrayFireTensor) !*Dim {
        try self.ensureTotalCapacity(self.shape_.len + 1);
        return self.addOneAssumeCapacity();
    }

    pub fn afDimsToZtShape(self: *ArrayFireTensor, af_dims: *const af.Dim4) !void {
        var num_dims = self.numDims();
        if (num_dims > @as(usize, @intCast(af.AF_MAX_DIMS))) {
            std.log.debug("afToZtDims: num_dims ({d}) > af.AF_MAX_DIMS ({d} )", .{ num_dims, af.AF_MAX_DIMS });
            return error.ExceedsArrayFireMaxDims;
        }

        // num_dims constraint is enforced by the internal API per condenseDims
        if (num_dims == 1 and af_dims.elements() == 0) {
            // Empty tensor
            try self.resize(1);
            self.shape_[0] = 0;
            return;
        }

        // num_dims == 0 --> scalar tensor
        if (num_dims == 0) {
            try self.resize(0);
            return;
        }

        try self.resize(num_dims);
        for (0..num_dims) |i| self.shape_[i] = @intCast(af_dims.dims[i]);
    }

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
        var self: *ArrayFireTensor = try allocator.create(ArrayFireTensor);
        var afDims = try handle.value.*.getDims();
        self.* = .{
            .arrayHandle_ = handle,
            .indices_ = af_indices,
            .indexTypes_ = index_types,
            .handle_ = .{ .indexedArray = IndexedArrayComponent.init(is_flat) },
            .numDims_ = num_dims,
            .allocator = allocator,
        };
        try self.afDimsToZtShape(&afDims);

        return self;
    }

    /// Initializes a new ArrayFireTensor from an ArrayFire array
    /// handle without copying the handle. Used for creating
    /// gauranteed shallow-copies.
    pub fn initFromShared(allocator: std.mem.Allocator, arr: Arc(*af.Array), num_dims: usize) !*ArrayFireTensor {
        var self: *ArrayFireTensor = try allocator.create(ArrayFireTensor);
        var afDims = try arr.value.*.getDims();
        self.* = .{
            .arrayHandle_ = arr,
            .numDims_ = num_dims,
            .allocator = allocator,
        };
        try self.afDimsToZtShape(&afDims);
        return self;
    }

    pub fn initFromArray(allocator: std.mem.Allocator, arr: *af.Array, num_dims: usize) !*ArrayFireTensor {
        var self: *ArrayFireTensor = try allocator.create(ArrayFireTensor);
        var afDims = try arr.getDims();
        self.* = .{
            .arrayHandle_ = try Arc(*af.Array).init(allocator, arr),
            .numDims_ = num_dims,
            .allocator = allocator,
        };
        try self.afDimsToZtShape(&afDims);
        return self;
    }

    pub fn initRaw(allocator: std.mem.Allocator) !*ArrayFireTensor {
        var arr = try af.Array.initHandle(allocator, 0, af.Dim4{}, af.Dtype.f32);
        return ArrayFireTensor.initFromArray(allocator, arr, 0);
    }

    pub fn initEmpty(allocator: std.mem.Allocator) !*ArrayFireTensor {
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
                try af.ops.ztToAfDims(_shape),
                af.ops.ztToAfType(data_type),
            ),
            .Device => arr = try af.Array.initFromDevicePtr(
                allocator,
                ptr,
                @intCast(_shape.ndim()),
                try af.ops.ztToAfDims(_shape),
                af.ops.ztToAfType(data_type),
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
            af.ops.releaseIndexers(self.indices_.?) catch unreachable;
        }
        // remove IndexTypes
        if (self.indexTypes_ != null) {
            self.indexTypes_.?.deinit();
        }
        self.allocator.free(self.allocatedSlice());
        self.allocator.destroy(self);
    }

    pub fn getHandle(self: *ArrayFireTensor, allocator: std.mem.Allocator) GetHandleErrors!*af.Array {
        if (@as(ComponentTypeTag, self.handle_) != .array) {
            const idxComp: IndexedArrayComponent = self.handle_.indexedArray;
            var idxTypes: ?[]IndexType = if (self.indexTypes_ != null) self.indexTypes_.?.items else null;

            //std.debug.print("attempting IndexedArrayComponent.get()\n", .{});
            var oldHandle = self.arrayHandle_;
            defer oldHandle.releaseWithFn(af.Array.deinit);
            var used_arr = try idxComp.get(allocator, self);
            var condensed = try condenseIndices(
                allocator,
                used_arr,
                false,
                idxTypes,
                idxComp.isFlat_,
            );
            defer if (condensed.modified) used_arr.deinit();

            // assign the new handle
            self.arrayHandle_ = try Arc(*af.Array).init(
                allocator,
                condensed.arr,
            );

            // Clear state
            self.handle_ = .{ .array = ArrayComponent{} }; // set to passthrough
            // remove indices
            if (self.indices_ != null) {
                try af.ops.releaseIndexers(self.indices_.?);
                self.indices_ = null;
            }
            // remove IndexTypes
            if (self.indexTypes_ != null) {
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
        try self.afDimsToZtShape(&afDims);
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

    pub fn strides(self: *ArrayFireTensor, allocator: std.mem.Allocator) ![]Dim {
        var arr = try self.getHandle(allocator);
        var afStrides = try arr.getStrides();
        return afStrides.dimsToOwnedShape(allocator);
    }

    pub fn stream(self: *ArrayFireTensor, allocator: std.mem.Allocator) !runtime.Stream {
        // TODO indexing is unlikely to change the stream associated with a tensor.
        // But if it can, we need to call `getHandle()` here.
        var bknd = try ArrayFireBackend.getInstance(allocator);
        return bknd.getStreamOfArray(allocator, self.arrayHandle_.value.*);
    }

    pub fn astype(self: *ArrayFireTensor, allocator: std.mem.Allocator, dType: DType) !Tensor {
        var arr = try self.getHandle(allocator);
        var convertedArr = try arr.cast(allocator, af.ops.ztToAfType(dType));
        return Tensor.init(TensorAdapterBase.init(try ArrayFireTensor.initFromArray(allocator, convertedArr, self.numDims())));
    }

    pub fn index(self: *ArrayFireTensor, allocator: std.mem.Allocator, indices: []Index) !Tensor {
        if (indices.len > @as(usize, @intCast(af.AF_MAX_DIMS))) {
            std.log.debug(
                "ArrayFire-backed tensor was indexed with > 4 elements: ArrayFire tensors support up to 4 dimensions.\n",
                .{},
            );
            return error.IndicesExceedMaxDims;
        }

        // TODO: vet and stress test this a lot more/add proper support for
        // multi-tensor
        // If indexing by a single element and it's a tensor with the same number of
        // indices as the array being indexed, do a flat index as this is probably a
        // filter-based index (for example: a(a < 5)).
        const completeTensorIndex = indices.len == 1 and indices[0].idxType() == .Tensor and try indices[0].index_.Tensor.elements(allocator) == @as(usize, @intCast(try (try self.getHandle(allocator)).getElements()));
        var afIndices = try af.ops.createIndexers(); // this creates implicit spans for up to maxDims
        if (completeTensorIndex) {
            // TODO: verify this is correct; needs tests
            try af.ops.setSeqParamIndexer(afIndices, 0, 0, 1, 0, false);
        }

        if (indices.len > afIndices.len) {
            std.log.debug("ArrayFireTensor.index internal error - passed indices is larger than the number of af indices.\n", .{});
            return error.PassedIndicesLargerThanAfIndices;
        }

        // Fill in corresponding index types for each af index
        var indexTypes = try std.ArrayList(IndexType).initCapacity(allocator, afIndices.len);
        var i: usize = 0;
        while (i < indices.len) : (i += 1) {
            indexTypes.appendAssumeCapacity(indices[i].idxType());
            afIndices[i] = af.ops.ztToAfIndex(indices[i]);
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
        newNumDims = @max(newNumDims, 1); // can never index to a 0 dim tensor

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
                std.log.debug("ArrayFireTensor.adjustInPlaceOperandDims index size was 1 but tensor has greater than 1 dimension.\n", .{});
                return error.IndexSizeTensorDimsMismatch;
            }
        } else if (self.indices_ != null and self.indices_.?.len > 0) {
            // All other indexing operations
            const indices: []af.af_index_t = self.indices_.?;
            const indexTypes: []IndexType = self.indexTypes_.?.items;
            if (indices.len != indexTypes.len) {
                std.log.debug("ArrayFireTensor.adjustInPlaceOperandDims - passed indices and indexTypes are of different sizes.\n", .{});
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
                            postIdxDims.dims[i] = if (seq.step == 0) 0 else @as(af.dim_t, @intCast(@abs((seq.end - seq.begin) / seq.step))) + 1;
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
                std.log.debug("ArrayFireTensor.adjustInPlaceOperandDims: can't apply operation in-place to indexed ArrayFireTensor - dimensions don't match.", .{});
                return error.CannotApplyOperationDimsMismatch;
            }
        } else {
            // No indexing so no change in dimensions required
            newDims = operandDims;
        }

        // moddims involves an eval. This will be fixed in AF 3.8.1/3.8.2
        const doModdims = !std.meta.eql(af.dim_t, &(try operandArr.getDims()).dims, &newDims.dims);
        if (doModdims) {
            return operandArr.moddims(allocator, newDims.ndims(), newDims);
        } else {
            return operandArr;
        }
    }
};

// unit tests/test utility functions

fn getRefCount(arr: *af.Array, sync: bool) !i32 {
    if (sync) {
        try arr.eval();
        try af.ops.sync(-1);
    }
    return arr.getDataRefCount();
}

fn allClose(allocator: std.mem.Allocator, a: *const af.Array, b: *const af.Array, abs_tolerance: f64) !bool {
    if (try a.getType() != try b.getType()) {
        return false;
    }
    if (!std.mem.eql(af.dim_t, &(try a.getDims()).dims, &(try b.getDims()).dims)) {
        return false;
    }
    if (try a.isEmpty() and try b.isEmpty()) {
        return true;
    }
    var sub = try af.ops.sub(allocator, a, b, false);
    defer sub.deinit();
    var abs = try af.ops.abs(allocator, sub);
    defer abs.deinit();
    var max = try af.ops.maxAll(abs);
    return max.real < abs_tolerance;
}

test "ArrayFireTensorBaseTest -> AfRefCountBasic" {
    const allocator = std.testing.allocator;
    // Sanity check that af.af_array moved into zt.Tensors don't have their
    // refcount inrcremented/show proper usage of refs in tensor ops
    var qDims = af.Dim4{};
    qDims.dims[0] = 2;
    qDims.dims[1] = 2;
    defer deinit(); // deinit global singletons
    var q = try af.ops.constant(allocator, 1, 2, qDims, af.Dtype.f32);
    defer q.deinit();
    // without eval/sync, no refcount
    try std.testing.expect(try getRefCount(q, false) == 0);

    var a = try af.ops.constant(allocator, 1, 2, qDims, af.Dtype.f32);
    try std.testing.expect(try getRefCount(q, true) == 1);

    var tensor = Tensor.init(
        TensorAdapterBase.init(
            try ArrayFireTensor.initFromArray(allocator, a, 2),
        ),
    );
    defer tensor.deinit();
    var aRef = try toArray(allocator, tensor);
    try std.testing.expect(try getRefCount(aRef, true) == 1);

    // TODO: verify copy works and increments ref count
}

test "ArrayFireTensorBaseTest -> AfRefCountModify" {
    const full = @import("../../tensor.zig").full;
    const add = @import("../../tensor.zig").add;
    const mul = @import("../../tensor.zig").mul;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    // Compositional operations don't increment refcount
    var a = try full(allocator, &.{ 2, 2 }, f64, 1, .f32);
    defer a.deinit();
    var b = try full(allocator, &.{ 2, 2 }, f64, 1, .f32);
    defer b.deinit();

    var res1 = try add(allocator, Tensor, a, Tensor, b);
    defer res1.deinit();
    try std.testing.expect(try getRefCount(try toArray(allocator, a), true) == 1);
    try std.testing.expect(try getRefCount(try toArray(allocator, b), true) == 1);

    var res1Data = try res1.allocHost(allocator, f32);
    defer allocator.free(res1Data.?);
    for (res1Data.?) |v| try std.testing.expect(v == 2);

    var c = try full(allocator, &.{ 2, 2 }, f64, 1, .f32);
    defer c.deinit();
    var d = try full(allocator, &.{ 2, 2 }, f64, 1, .f32);
    defer d.deinit();

    var res2_a = try mul(allocator, Tensor, c, Tensor, c);
    defer res2_a.deinit();
    var res2_b = try mul(allocator, Tensor, d, Tensor, d);
    defer res2_b.deinit();
    var res2 = try add(allocator, Tensor, res2_a, Tensor, res2_b);
    defer res2.deinit();
    try std.testing.expect(try getRefCount(try toArray(allocator, c), true) == 1);
    try std.testing.expect(try getRefCount(try toArray(allocator, d), true) == 1);
}

test "ArrayFireTensorBaseTest -> astypeRefcount" {
    const allocator = std.testing.allocator;
    const rand = @import("../../tensor.zig").rand;
    defer deinit(); // deinit global singletons

    var t = try rand(allocator, &.{ 5, 5 }, .f32);
    defer t.deinit();

    try std.testing.expect(try getRefCount(try toArray(allocator, t), true) == 1);

    var t64 = try t.astype(allocator, .f64);
    defer t64.deinit();
    try std.testing.expect(try getRefCount(try toArray(allocator, t64), true) == 1);
    try std.testing.expect(try t64.dtype(allocator) == .f64);
}

// TODO: test "astypeInPlaceRefcount" {}

test "ArrayFireTensorBaseTest -> BackendInterop" {
    const allocator = std.testing.allocator;
    const rand = @import("../../tensor.zig").rand;
    defer deinit(); // deinit global singletons
    var a = try rand(allocator, &.{ 10, 12 }, .f32);
    defer a.deinit();
    try std.testing.expect(a.backendType() == .ArrayFire);
}

// TODO: test "withTensorType" {}

// TODO: test "ArrayFireAssignmentOperators" {}

test "ArrayFireTensorBaseTest -> BinaryOperators" {
    const full = @import("../../tensor.zig").full;
    const eq = @import("../../tensor.zig").eq;
    const add = @import("../../tensor.zig").add;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try full(allocator, &.{ 2, 2 }, f64, 1, .f32);
    defer a.deinit();
    var b = try full(allocator, &.{ 2, 2 }, f64, 2, .f32);
    defer b.deinit();
    var c = try full(allocator, &.{ 2, 2 }, f64, 3, .f32);
    defer c.deinit();

    // test equality of tensors vs equality of underlying ArrayFire arrays
    var tensorEq = try eq(allocator, Tensor, a, Tensor, b);
    defer tensorEq.deinit();
    var arrEq = try af.ops.eq(allocator, try toArray(allocator, a), try toArray(allocator, b), false);
    defer arrEq.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, tensorEq), arrEq, 1e-5));

    // test addition of tensors vs addition of underlying ArrayFire arrays
    var tensorAdd = try add(allocator, Tensor, a, Tensor, b);
    defer tensorAdd.deinit();
    var arrAdd = try af.ops.add(allocator, try toArray(allocator, a), try toArray(allocator, b), false);
    defer arrAdd.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, tensorAdd), arrAdd, 1e-5));
}

test "ArrayFireTensorBaseTest -> full" {
    const full = @import("../../tensor.zig").full;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    // TODO: expand with fixtures for each type
    var a = try full(allocator, &.{ 3, 4 }, f64, 3, .f32);
    defer a.deinit();

    try std.testing.expect(zt_shape.eql(&.{ 3, 4 }, try a.shape(allocator)));
    try std.testing.expect(try a.dtype(allocator) == .f32);
    var aAfDim = af.Dim4{};
    aAfDim.dims[0] = 3;
    aAfDim.dims[1] = 4;
    var aArray = try af.ops.constant(allocator, 3, 2, aAfDim, .f32);
    defer aArray.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, a), aArray, 1e-5));

    var b = try full(allocator, &.{ 1, 1, 5, 4 }, f64, 4.5, .f32);
    defer b.deinit();
    try std.testing.expect(zt_shape.eql(&.{ 1, 1, 5, 4 }, try b.shape(allocator)));
    try std.testing.expect(try b.dtype(allocator) == .f32);
    var bAfDim = af.Dim4.init([4]af.dim_t{ 1, 1, 5, 4 });
    var bArray = try af.ops.constant(allocator, 4.5, 4, bAfDim, .f32);
    defer bArray.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, b), bArray, 1e-5));
}

test "ArrayFireTensorBaseTest -> identity" {
    const identity = @import("../../tensor.zig").identity;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try identity(allocator, 6, .f32);
    defer a.deinit();

    try std.testing.expect(zt_shape.eql(&.{ 6, 6 }, try a.shape(allocator)));
    try std.testing.expect(try a.dtype(allocator) == .f32);
    var afDim = af.Dim4{};
    afDim.dims[0] = 6;
    afDim.dims[1] = 6;
    var arrIdentity = try af.ops.identity(allocator, 2, afDim, .f32);
    defer arrIdentity.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, a), arrIdentity, 1e-5));

    var f64Tensor = try identity(allocator, 6, .f64);
    defer f64Tensor.deinit();
    try std.testing.expect(try f64Tensor.dtype(allocator) == .f64);
}

test "ArrayFireTensorBaseTest -> randn" {
    const randn = @import("../../tensor.zig").randn;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    const s: Dim = 30;
    var a = try randn(allocator, &.{ s, s }, .f32);
    defer a.deinit();

    try std.testing.expect(zt_shape.eql(&.{ s, s }, try a.shape(allocator)));
    try std.testing.expect(try a.dtype(allocator) == .f32);
    var afDim = af.Dim4{};
    afDim.dims[0] = @intCast(s * s);
    var moddimsArr = try af.ops.moddims(allocator, try toArray(allocator, a), 1, afDim);
    defer moddimsArr.deinit();
    var meanArr = try af.ops.mean(allocator, moddimsArr, -1);
    defer meanArr.deinit();
    var absArr = try af.ops.abs(allocator, meanArr);
    defer absArr.deinit();
    var tmpConst = try af.ops.constant(allocator, 2, 1, af.Dim4{}, .f32);
    defer tmpConst.deinit();
    var ltArr = try af.ops.lt(allocator, absArr, tmpConst, false);
    defer ltArr.deinit();
    try std.testing.expect(@as(u1, @intFromFloat((try af.ops.allTrueAll(ltArr)).real)) != 0);
}

test "ArrayFireTensorBaseTest -> rand" {
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    const s: Dim = 30;
    var a = try rand(allocator, &.{ s, s }, .f32);
    defer a.deinit();
    try std.testing.expect(zt_shape.eql(&.{ s, s }, try a.shape(allocator)));
    try std.testing.expect(try a.dtype(allocator) == .f32);

    var afDims = af.Dim4{};
    afDims.dims[0] = @intCast(s);
    afDims.dims[1] = @intCast(s);
    var tmpConst1 = try af.ops.constant(allocator, 1, 2, afDims, .f32);
    defer tmpConst1.deinit();
    var lteArr = try af.ops.le(allocator, try toArray(allocator, a), tmpConst1, false);
    defer lteArr.deinit();
    try std.testing.expect(@as(u1, @intFromFloat((try af.ops.allTrueAll(lteArr)).real)) != 0);
    var tmpConst2 = try af.ops.constant(allocator, 0, 2, afDims, .f32);
    defer tmpConst2.deinit();
    var gteArr = try af.ops.ge(allocator, try toArray(allocator, a), tmpConst2, false);
    defer gteArr.deinit();
    try std.testing.expect(@as(u1, @intFromFloat((try af.ops.allTrueAll(gteArr)).real)) != 0);

    var b = try rand(allocator, &.{1}, .f64);
    defer b.deinit();
    try std.testing.expect(try b.dtype(allocator) == .f64);
}

test "ArrayFireTensorBaseTest -> amin" {
    const rand = @import("../../tensor.zig").rand;
    const amin = @import("../../tensor.zig").amin;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try rand(allocator, &.{ 3, 3 }, .f32);
    defer a.deinit();

    var axes = std.ArrayList(i32).init(allocator);
    defer axes.deinit();
    var tensorRes1 = try amin(allocator, a, axes, false);
    defer tensorRes1.deinit();

    var tensorScalar1 = try tensorRes1.scalar(allocator, f32);
    var afRes1 = try af.ops.minAll(try toArray(allocator, a));
    try std.testing.expect(tensorScalar1 == @as(f32, @floatCast(afRes1.real)));

    try axes.append(0);
    var tensorRes2 = try amin(allocator, a, axes, false);
    defer tensorRes2.deinit();
    var afRes2 = try af.ops.min(allocator, try toArray(allocator, a), 0);
    defer afRes2.deinit();
    var afRes3 = try condenseIndices(allocator, afRes2, false, null, false);
    defer if (afRes3.modified) afRes3.arr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, tensorRes2), afRes3.arr, 1e-5));
}

test "ArrayFireTensorBaseTest -> amax" {
    const rand = @import("../../tensor.zig").rand;
    const amax = @import("../../tensor.zig").amax;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try rand(allocator, &.{ 3, 3 }, .f32);
    defer a.deinit();

    var axes = std.ArrayList(i32).init(allocator);
    defer axes.deinit();
    var tensorRes1 = try amax(allocator, a, axes, false);
    defer tensorRes1.deinit();

    var tensorScalar1 = try tensorRes1.scalar(allocator, f32);
    var afRes1 = try af.ops.maxAll(try toArray(allocator, a));
    try std.testing.expect(tensorScalar1 == @as(f32, @floatCast(afRes1.real)));

    try axes.append(0);
    var tensorRes2 = try amax(allocator, a, axes, false);
    defer tensorRes2.deinit();
    var afRes2 = try af.ops.max(allocator, try toArray(allocator, a), 0);
    defer afRes2.deinit();
    var afRes3 = try condenseIndices(allocator, afRes2, false, null, false);
    defer if (afRes3.modified) afRes3.arr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, tensorRes2), afRes3.arr, 1e-5));
}

test "ArrayFireTensorBaseTest -> sum" {
    const rand = @import("../../tensor.zig").rand;
    const sum = @import("../../tensor.zig").sum;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try rand(allocator, &.{ 3, 3 }, .f32);
    defer a.deinit();

    var axes = std.ArrayList(i32).init(allocator);
    defer axes.deinit();
    var aSum = try sum(allocator, a, axes, false);
    defer aSum.deinit();

    var aSumScalar = try aSum.scalar(allocator, f32);
    var aAfSum = try af.ops.sumAll(try toArray(allocator, a));
    try std.testing.expectApproxEqAbs(aSumScalar, @as(f32, @floatCast(aAfSum.real)), 1e-5);

    try axes.append(0);
    var tensorSum = try sum(allocator, a, axes, false);
    defer tensorSum.deinit();
    var afRes2 = try af.ops.sum(allocator, try toArray(allocator, a), 0);
    defer afRes2.deinit();
    var afRes3 = try condenseIndices(allocator, afRes2, false, null, false);
    defer if (afRes3.modified) afRes3.arr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, tensorSum), afRes3.arr, 1e-5));

    var b = try rand(allocator, &.{ 5, 6, 7, 8 }, .f32);
    defer b.deinit();

    var bAxes = std.ArrayList(i32).init(allocator);
    defer bAxes.deinit();
    var bSum = try sum(allocator, b, bAxes, false);
    defer bSum.deinit();

    var bSumScalar = try bSum.scalar(allocator, f32);
    var bAfSum = try af.ops.sumAll(try toArray(allocator, b));
    try std.testing.expectApproxEqAbs(bSumScalar, @as(f32, @floatCast(bAfSum.real)), 1e-5);

    try bAxes.append(1);
    try bAxes.append(2);
    var bTensorSum = try sum(allocator, b, bAxes, false);
    defer bTensorSum.deinit();
    var afRes4 = try af.ops.sum(allocator, try toArray(allocator, b), 1);
    defer afRes4.deinit();
    var afRes5 = try af.ops.sum(allocator, afRes4, 2);
    defer afRes5.deinit();
    var afRes6 = try condenseIndices(allocator, afRes5, false, null, false);
    defer if (afRes6.modified) afRes6.arr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, bTensorSum), afRes6.arr, 1e-5));
}

test "ArrayFireTensorBaseTest -> exp" {
    const exp = @import("../../tensor.zig").exp;
    const full = @import("../../tensor.zig").full;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var in = try full(allocator, &.{ 3, 3 }, f64, 4, .f32);
    defer in.deinit();

    var expTensor = try exp(allocator, in);
    defer expTensor.deinit();
    var expArr = try af.ops.exp(allocator, try toArray(allocator, in));
    defer expArr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, expTensor), expArr, 1e-5));
}

test "ArrayFireTensorBaseTest -> log" {
    const log = @import("../../tensor.zig").log;
    const full = @import("../../tensor.zig").full;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var in = try full(allocator, &.{ 3, 3 }, f64, 2, .f32);
    defer in.deinit();

    var logTensor = try log(allocator, in);
    defer logTensor.deinit();
    var logArr = try af.ops.log(allocator, try toArray(allocator, in));
    defer logArr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, logTensor), logArr, 1e-5));
}

test "ArrayFireTensorBaseTest -> log1p" {
    const log = @import("../../tensor.zig").log;
    const log1p = @import("../../tensor.zig").log1p;
    const add = @import("../../tensor.zig").add;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var in = try rand(allocator, &.{ 3, 3 }, .f32);
    defer in.deinit();

    var log1pTensor = try log1p(allocator, in);
    defer log1pTensor.deinit();
    var add1Tensor = try add(allocator, Tensor, in, f32, 1);
    defer add1Tensor.deinit();
    var add1LogTensor = try log(allocator, add1Tensor);
    defer add1LogTensor.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, log1pTensor), try toArray(allocator, add1LogTensor), 1e-5));
}

test "ArrayFireTensorBaseTest -> sin" {
    const sin = @import("../../tensor.zig").sin;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var in = try rand(allocator, &.{ 3, 3 }, .f32);
    defer in.deinit();

    var sinTensor = try sin(allocator, in);
    defer sinTensor.deinit();
    var sinArr = try af.ops.sin(allocator, try toArray(allocator, in));
    defer sinArr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, sinTensor), sinArr, 1e-5));
}

test "ArrayFireTensorBaseTest -> cos" {
    const cos = @import("../../tensor.zig").cos;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var in = try rand(allocator, &.{ 3, 3 }, .f32);
    defer in.deinit();

    var cosTensor = try cos(allocator, in);
    defer cosTensor.deinit();
    var cosArr = try af.ops.cos(allocator, try toArray(allocator, in));
    defer cosArr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, cosTensor), cosArr, 1e-5));
}

test "ArrayFireTensorBaseTest -> sqrt" {
    const sqrt = @import("../../tensor.zig").sqrt;
    const full = @import("../../tensor.zig").full;
    const div = @import("../../tensor.zig").div;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var in = try full(allocator, &.{ 3, 3 }, f64, 4, .f32);
    defer in.deinit();

    var sqrtTensor = try sqrt(allocator, in);
    defer sqrtTensor.deinit();
    var expected = try div(allocator, Tensor, in, f32, 2);
    defer expected.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, sqrtTensor), try toArray(allocator, expected), 1e-5));
}

test "ArrayFireTensorBaseTest -> tanh" {
    const tanh = @import("../../tensor.zig").tanh;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var in = try rand(allocator, &.{ 3, 3 }, .f32);
    defer in.deinit();

    var tanhTensor = try tanh(allocator, in);
    defer tanhTensor.deinit();
    var tanhArr = try af.ops.tanh(allocator, try toArray(allocator, in));
    defer tanhArr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, tanhTensor), tanhArr, 1e-5));
}

test "ArrayFireTensorBaseTest -> absolute" {
    const absolute = @import("../../tensor.zig").absolute;
    const full = @import("../../tensor.zig").full;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    const val: f64 = -3.1;

    var a = try full(allocator, &.{ 3, 3 }, f64, val, .f32);
    defer a.deinit();
    var absA = try absolute(allocator, a);
    defer absA.deinit();

    var comp = try full(allocator, &.{ 3, 3 }, f64, -val, .f32);
    defer comp.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, absA), try toArray(allocator, comp), 1e-5));
}

test "ArrayFireTensorBaseTest -> erf" {
    const erf = @import("../../tensor.zig").erf;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var in = try rand(allocator, &.{ 3, 3 }, .f32);
    defer in.deinit();

    var erfTensor = try erf(allocator, in);
    defer erfTensor.deinit();
    var erfArr = try af.ops.erf(allocator, try toArray(allocator, in));
    defer erfArr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, erfTensor), erfArr, 1e-5));
}

test "ArrayFireTensorBaseTest -> mean" {
    const mean = @import("../../tensor.zig").mean;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try rand(allocator, &.{ 3, 50 }, .f32);
    defer a.deinit();

    var axes = std.ArrayList(i32).init(allocator);
    defer axes.deinit();
    var meanTensor = try mean(allocator, a, axes, false);
    defer meanTensor.deinit();

    try std.testing.expectApproxEqAbs(
        try meanTensor.scalar(allocator, f32),
        @floatCast((try af.ops.meanAll(try toArray(allocator, a))).real),
        1e-4,
    );

    try axes.append(0);
    var meanTensor2 = try mean(allocator, a, axes, false);
    defer meanTensor2.deinit();
    var meanArr = try af.ops.mean(allocator, try toArray(allocator, a), 0);
    defer meanArr.deinit();
    var condensedMean = try condenseIndices(allocator, meanArr, false, null, false);
    defer if (condensedMean.modified) condensedMean.arr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, meanTensor2), condensedMean.arr, 1e-5));
}

test "ArrayFireTensorBaseTest -> median" {
    const median = @import("../../tensor.zig").median;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try rand(allocator, &.{ 3, 50 }, .f32);
    defer a.deinit();

    var axes = std.ArrayList(i32).init(allocator);
    defer axes.deinit();
    var medianTensor = try median(allocator, a, axes, false);
    defer medianTensor.deinit();

    try std.testing.expectApproxEqAbs(
        @as(f32, @floatCast((try af.ops.medianAll(try toArray(allocator, a))).real)),
        try medianTensor.scalar(allocator, f32),
        1e-3,
    );

    try axes.append(0);
    var medianTensor2 = try median(allocator, a, axes, false);
    defer medianTensor2.deinit();
    var medianArr = try af.ops.median(allocator, try toArray(allocator, a), 0);
    defer medianArr.deinit();
    var condensedMedian = try condenseIndices(allocator, medianArr, false, null, false);
    defer if (condensedMedian.modified) condensedMedian.arr.deinit();
    try std.testing.expect(try allClose(
        allocator,
        try toArray(allocator, medianTensor2),
        condensedMedian.arr,
        1e-5,
    ));
}

test "ArrayFireTensorBaseTest -> variance" {
    const variance = @import("../../tensor.zig").variance;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    const bias = false;
    const bias_mode: af.VarBias = if (bias) .Sample else .Population;

    var a = try rand(allocator, &.{ 3, 3 }, .f32);
    defer a.deinit();
    var axes = std.ArrayList(i32).init(allocator);
    defer axes.deinit();

    var aVarTensor = try variance(allocator, a, axes, false, false);
    defer aVarTensor.deinit();
    var aVarScalar = try aVarTensor.scalar(allocator, f32);
    try std.testing.expect(
        @as(f32, @floatCast(
            (try af.ops.varAllV2(
                try toArray(allocator, a),
                bias_mode,
            )).real,
        )) == aVarScalar,
    );

    try axes.append(0);
    var aVarTensor2 = try variance(allocator, a, axes, false, false);
    defer aVarTensor2.deinit();
    var aVarArr = try af.ops.varV2(allocator, try toArray(allocator, a), bias_mode, 0);
    defer aVarArr.deinit();
    var condensedVar = try condenseIndices(
        allocator,
        aVarArr,
        false,
        null,
        false,
    );
    defer if (condensedVar.modified) condensedVar.arr.deinit();
    try std.testing.expect(try allClose(
        allocator,
        try toArray(allocator, aVarTensor2),
        condensedVar.arr,
        1e-5,
    ));

    axes.items[0] = 1;
    var aVarTensor3 = try variance(allocator, a, axes, false, false);
    defer aVarTensor3.deinit();
    var aVarArr2 = try af.ops.varV2(
        allocator,
        try toArray(allocator, a),
        bias_mode,
        1,
    );
    defer aVarArr2.deinit();
    var condensedVar2 = try condenseIndices(
        allocator,
        aVarArr2,
        false,
        null,
        false,
    );
    defer if (condensedVar2.modified) condensedVar2.arr.deinit();
    try std.testing.expect(try allClose(
        allocator,
        try toArray(allocator, aVarTensor3),
        condensedVar2.arr,
        1e-5,
    ));

    // Make sure multidimension matches computing for all
    try std.testing.expectEqual(
        (try toArray(allocator, aVarTensor)).getScalar(f32),
        @floatCast(
            (try af.ops.varAllV2(try toArray(allocator, a), bias_mode)).real,
        ),
    );
    try axes.insert(0, 0);
    var biasedTensor = try variance(allocator, a, axes, true, false);
    defer biasedTensor.deinit();
    var biasedScalar = try biasedTensor.scalar(allocator, f32);
    try std.testing.expectEqual(
        @as(
            f32,
            @floatCast(
                (try af.ops.varAllV2(try toArray(allocator, a), .Sample)).real,
            ),
        ),
        biasedScalar,
    );
}

test "ArrayFireTensorBaseTest -> stdev" {
    const stdev = @import("../../tensor.zig").stdev;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try rand(allocator, &.{ 3, 3 }, .f32);
    defer a.deinit();

    var axes = std.ArrayList(i32).init(allocator);
    defer axes.deinit();
    try axes.append(0);
    var tensor_res_1 = try stdev(allocator, a, axes, true);
    defer tensor_res_1.deinit();
    var arr_res_1 = try af.ops.stdevV2(
        allocator,
        try toArray(allocator, a),
        .Population,
        0,
    );
    defer arr_res_1.deinit();
    try std.testing.expect(try allClose(
        allocator,
        try toArray(allocator, tensor_res_1),
        arr_res_1,
        1e-5,
    ));

    axes.items[0] = 1;
    var tensor_res_2 = try stdev(allocator, a, axes, true);
    defer tensor_res_2.deinit();
    var arr_res_2 = try af.ops.stdevV2(
        allocator,
        try toArray(allocator, a),
        .Population,
        1,
    );
    defer arr_res_2.deinit();
    try std.testing.expect(try allClose(
        allocator,
        try toArray(allocator, tensor_res_2),
        arr_res_2,
        1e-5,
    ));

    try axes.insert(0, 0);
    var tensor_res_3 = try stdev(allocator, a, axes, false);
    defer tensor_res_3.deinit();
    var tensor_res_3_scalar = try tensor_res_3.scalar(allocator, f32);
    var expected = @sqrt((try af.ops.varAllV2(try toArray(allocator, a), .Population)).real);
    try std.testing.expectEqual(tensor_res_3_scalar, @floatCast(expected));
}

test "ArrayFireTensorBaseTest -> norm" {
    const norm = @import("../../tensor.zig").norm;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try rand(allocator, &.{ 3, 3 }, .f32);
    defer a.deinit();

    var axes = std.ArrayList(i32).init(allocator);
    var res_tensor = try norm(allocator, a, axes, 2, false);
    defer res_tensor.deinit();

    var arr_norm = try af.ops.norm(try toArray(allocator, a), af.NormType.Vector2, 1, 1);
    try std.testing.expectApproxEqAbs(
        @as(f32, @floatCast(arr_norm)),
        try res_tensor.scalar(allocator, f32),
        1e-4,
    );
}

test "ArrayFireTensorBaseTest -> tile" {
    const tile = @import("../../tensor.zig").tile;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try rand(allocator, &.{ 3, 3 }, .f32);
    defer a.deinit();

    var tile_tensor = try tile(allocator, a, &.{ 4, 5, 6 });
    defer tile_tensor.deinit();
    var tile_arr = try af.ops.tile(allocator, try toArray(allocator, a), 4, 5, 6, 1);
    defer tile_arr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, tile_tensor), tile_arr, 1e-5));
}

test "ArrayFireTensorBaseTest -> nonzero" {
    const nonzero = @import("../../tensor.zig").nonzero;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try rand(allocator, &.{ 3, 3 }, .u32);
    defer a.deinit();

    var nz = try nonzero(allocator, a);
    defer nz.deinit();
    var arr = try af.ops.where(allocator, try toArray(allocator, a));
    defer arr.deinit();
    try std.testing.expect(try allClose(allocator, try toArray(allocator, nz), arr, 1e-5));
}

test "ArrayFireTensorBaseTest -> transpose" {
    const transpose = @import("../../tensor.zig").transpose;
    const rand = @import("../../tensor.zig").rand;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons

    var a = try rand(allocator, &.{ 3, 5 }, .f32);
    defer a.deinit();

    try std.testing.expectError(
        error.ArrayFireTransposeFailed,
        transpose(allocator, a, &.{ 0, 1, 2, 3, 4 }),
    );

    var a_trans_tensor = try transpose(allocator, a, &.{});
    defer a_trans_tensor.deinit();
    var a_trans_arr = try af.ops.transpose(allocator, try toArray(allocator, a), false);
    defer a_trans_arr.deinit();
    try std.testing.expect(try allClose(
        allocator,
        try toArray(allocator, a_trans_tensor),
        a_trans_arr,
        1e-5,
    ));

    var b = try rand(allocator, &.{ 3, 5, 4, 8 }, .f32);
    defer b.deinit();

    var b_trans_tensor = try transpose(allocator, b, &.{ 2, 0, 1, 3 });
    defer b_trans_tensor.deinit();
    var b_trans_arr = try af.ops.reorder(allocator, try toArray(allocator, b), 2, 0, 1, 3);
    defer b_trans_arr.deinit();
    try std.testing.expect(try allClose(
        allocator,
        try toArray(allocator, b_trans_tensor),
        b_trans_arr,
        1e-5,
    ));
}

test "ArrayFireTensorBaseTest -> concatenate" {
    const concatenate = @import("../../tensor.zig").concatenate;
    const allocator = std.testing.allocator;
    defer deinit(); // deinit global singletons
    var tensors = try std.ArrayList(Tensor).initCapacity(allocator, 11);
    defer tensors.deinit();
    try std.testing.expectError(
        error.ConcatFailedZeroTensors,
        concatenate(allocator, tensors, 0),
    );
}

// TODO: test "ArrayFireTensorBaseTest -> device" {}

// TODO: test "ArrayFireTensorBaseTest -> defaultConstructor" {}

// TODO: test "ArrayFireTensorBaseTest -> emptyRangeIndexing" {}
