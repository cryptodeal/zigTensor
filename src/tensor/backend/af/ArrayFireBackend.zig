const std = @import("std");
const af = @import("../../../bindings/af/ArrayFire.zig");
const base = @import("../../TensorBase.zig");
const zt_shape = @import("../../Shape.zig");
const zt_types = @import("../../Types.zig");
const build_options = @import("build_options");
const rt_stream = @import("../../../runtime/Stream.zig");
const rt_device_manager = @import("../../../runtime/DeviceManager.zig");
const zigrc = @import("zigrc");
const rt_device_type = @import("../../../runtime/DeviceType.zig");
const reductions = @import("reductions.zig");
const zt_backend = @import("../../TensorBackend.zig");

const TopkRes = zt_backend.TopkRes;
const SortIndexRes = zt_backend.SortIndexRes;
const condenseIndices = @import("Utils.zig").condenseIndices;
const isAllAxisReduction = reductions.isAllAxisReduction;
const afReduceAxes = reductions.afReduceAxes;
const reduceFunc_t = reductions.reduceFunc_t;
const getReducedNumDims = reductions.getReducedNumDims;
const getDeviceTypes = rt_device_type.getDeviceTypes;
const ArrayFireCPUStream = @import("ArrayFireCPUStream.zig").ArrayFireCPUStream;
const Arc = zigrc.Arc;
const ArrayFireTensor = @import("ArrayFireTensor.zig").ArrayFireTensor;
const toArray = @import("ArrayFireTensor.zig").toArray;
const ZT_BACKEND_CUDA = build_options.ZT_BACKEND_CUDA;
const ZT_BACKEND_CPU = build_options.ZT_BACKEND_CPU;
const ZT_ARRAYFIRE_USE_CUDA = build_options.ZT_ARRAYFIRE_USE_CUDA;
const ZT_ARRAYFIRE_USE_CPU = build_options.ZT_ARRAYFIRE_USE_CPU;
const SortMode = base.SortMode;
const PadType = base.PadType;
const MatrixProperty = base.MatrixProperty;
const TensorBackendType = base.TensorBackendType;
const TensorAdapterBase = @import("../../TensorAdapter.zig").TensorAdapterBase;
const Stream = rt_stream.Stream;
const DType = zt_types.DType;
const DeviceManager = rt_device_manager.DeviceManager;
const Tensor = base.Tensor;
const Shape = zt_shape.Shape;
const Dim = zt_shape.Dim;

var memoryInitFlag = std.once(init);

// Intentionally private. Only one instance should exist/it should be accessed
// via getInstance().
fn init() void {
    // TODO: remove this temporary workaround for TextDatasetTest crash on CPU
    // backend when tearing down the test environment. This is possibly due to
    // AF race conditions when tearing down our custom memory manager.
    // TODO: remove this temporary workaround for crashes when using custom
    // opencl kernels.
    if (ZT_BACKEND_CUDA) {
        // TODO: install memory manager
    }
}

/// Get the stream associated with given device in the given map; if it's not in
/// the map, initialize it (by wrapping or creating) and put it into the map.
fn getOrWrapAfDeviceStream(
    allocator: std.mem.Allocator,
    afId: i32,
    nativeId: i32,
    afIdToStream: *std.AutoHashMap(i32, Arc(Stream)),
) !Stream {
    _ = nativeId;
    var iter = afIdToStream.get(afId);
    if (iter != null) {
        return iter.?.value.*;
    }

    if (ZT_ARRAYFIRE_USE_CPU) {
        var stream = try ArrayFireCPUStream.create(allocator);
        try afIdToStream.put(afId, stream);
        return stream.value.*;
    }
    //  else if (ZT_ARRAYFIRE_USE_CUDA) {
    //      TODO: add CUDA support
    //  }
    else {
        std.log.err("ArrayFireBackend was not compiled with support for CPU or GPU\n", .{});
        return error.ArrayFireBackendNoDeviceType;
    }
}

fn setActiveCallback(data: ?*anyopaque, id: c_int) !void {
    var self: *ArrayFireBackend = @ptrCast(@alignCast(data.?));
    const afId = self.nativeIdToId_.get(id).?;
    try af.ops.setDevice(afId);
    // this is the latest point we can lazily wrap the AF stream, which may get
    // lazily intialized anytime in AF internally, e.g., via tensor computation.
    _ = try getOrWrapAfDeviceStream(self.allocator, afId, id, self.afIdToStream_.value);
}

var ArrayFireBackendSingleton: ?*ArrayFireBackend = null;

// TODO: add ArrayFire CUDA support

/// A tensor backend implementation of the ArrayFire tensor library.
///
/// Since ArrayFire has an internal DeviceManager singleton to manage
/// its global state, nothing is stored here as those internals are
/// opaquely handled. ArrayFireBackend simply dispatches operations
/// on global tensor functions to their ArrayFire counterparts.
pub const ArrayFireBackend = struct {
    allocator: std.mem.Allocator,
    /// Maps ArrayFire Native Device ID to zigTensor Device ID.
    nativeIdToId_: std.AutoHashMap(i32, i32),
    /// Maps zigTensor Device ID to ArrayFire Native Device ID.
    idToNativeId_: std.AutoHashMap(i32, i32),
    /// Tracks the individual active stream on each ArrayFire device
    /// N.B. using a shared pointer see `zigrc` to allow its capture
    /// in setActive callback; see constructor for details.
    afIdToStream_: Arc(std.AutoHashMap(i32, Arc(Stream))),

    /// Private function to initialize a new ArrayFireBackend instance.
    /// Should not be called directly as only one instance should exist;
    /// use `ArrayFireBackend.getInstance` instead.
    fn init(allocator: std.mem.Allocator) !*ArrayFireBackend {
        var self: *ArrayFireBackend = try allocator.create(ArrayFireBackend);
        var map = std.AutoHashMap(i32, Arc(Stream)).init(allocator);
        self.* = .{
            .allocator = allocator,
            .nativeIdToId_ = std.AutoHashMap(i32, i32).init(allocator),
            .idToNativeId_ = std.AutoHashMap(i32, i32).init(allocator),
            .afIdToStream_ = try Arc(std.AutoHashMap(i32, Arc(Stream))).init(allocator, map),
        };
        try af.ops.init();
        memoryInitFlag.call();

        // segfaults here
        const device_count = try af.ops.getDeviceCount();
        for (0..device_count) |i| {
            const id: i32 = @intCast(i);
            // TODO investigate how OpenCL fits into this.
            var native_id: i32 = id;
            if (ZT_ARRAYFIRE_USE_CUDA) {
                // TODO: native_id = try AF_CHECK(af.afcu_get_native_id(&native_id, id));
            }
            try self.nativeIdToId_.put(native_id, id);
            try self.idToNativeId_.put(id, native_id);
        }

        var mgr = try DeviceManager.getInstance(allocator);

        // This callback ensures consistency of AF internal state on active device.
        // Capturing by value to avoid destructor race hazard for static objects.
        if (ZT_ARRAYFIRE_USE_CPU) {
            var device = try mgr.getActiveDevice(.x64);
            try device.addSetActiveCallback(setActiveCallback, self);
        } else if (ZT_ARRAYFIRE_USE_CUDA) {
            // TODO: add CUDA support
        }

        // Active device is never set explicitly, so we must wrap its stream eagerly.
        var activeAfId: i32 = try af.ops.getDevice();
        _ = try getOrWrapAfDeviceStream(allocator, activeAfId, self.idToNativeId_.get(activeAfId).?, self.afIdToStream_.value);

        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *ArrayFireBackend) void {
        self.idToNativeId_.deinit();
        self.nativeIdToId_.deinit();
        var map = self.afIdToStream_.tryUnwrap().?;
        var iterator = map.valueIterator();
        while (iterator.next()) |stream| {
            var s = stream.tryUnwrap();
            if (s != null) {
                s.?.deinit();
            }
        }
        map.deinit();
        self.allocator.destroy(self);
        ArrayFireBackendSingleton = null;
    }

    /// Returns the singleton instance of the ArrayFireBackend; if
    /// no instance exists, initializes a new one.
    pub fn getInstance(allocator: std.mem.Allocator) !*ArrayFireBackend {
        if (ArrayFireBackendSingleton == null) {
            ArrayFireBackendSingleton = try ArrayFireBackend.init(allocator);
        }
        return ArrayFireBackendSingleton.?;
    }

    /// Returns the enum value indicating the backend type.
    pub fn backendType(_: *const ArrayFireBackend) TensorBackendType {
        return .ArrayFire;
    }

    // -------------------------- Compute Functions --------------------------

    /// Evaluate any expressions in the ArrayFire array backing the tensor.
    pub fn eval(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !void {
        try af.ops.eval(try toArray(allocator, tensor));
    }

    /// Returns the stream from which the given array was created.
    pub fn getStreamOfArray(self: *ArrayFireBackend, allocator: std.mem.Allocator, arr: *af.Array) !Stream {
        // TODO once we enforce integrate Device.setDevice into fl.setDevice, each
        // array's stream should always be wrapped already (via setDevice callback).
        const afId = try arr.getDeviceId();
        const nativeId = self.idToNativeId_.get(afId).?;
        return getOrWrapAfDeviceStream(allocator, afId, nativeId, self.afIdToStream_.value);
    }

    pub fn supportsDataType(_: *const ArrayFireBackend, dtype: DType) !bool {
        return switch (dtype) {
            .f16 => {
                var half_support: bool = try af.ops.getHalfSupport(try af.ops.getDevice());
                // f16 isn't (yet) supported with the CPU backend per OneDNN
                // limitations
                return half_support and !ZT_BACKEND_CPU;
            },
            else => true,
        };
    }

    // TODO: pub fn getMemMgrInfo()

    // TODO: pub fn setMemMgrLogStream()

    // TODO: pub fn setMemMgrLoggingEnabled()

    // TODO: pub fn setMemMgrFlushInterval()

    // -------------------------- Rand Functions --------------------------
    pub fn setSeed(_: *const ArrayFireBackend, seed: u64) !void {
        try af.ops.setSeed(seed);
    }

    pub fn randn(_: *const ArrayFireBackend, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) !Tensor {
        var dims = try af.ops.ztToAfDims(shape);
        const ndims = shape.ndim();
        var arr = try af.Array.randn(allocator, @intCast(ndims), dims, af.ops.ztToAfType(dtype));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    ndims,
                ),
            ),
        );
    }

    pub fn rand(_: *const ArrayFireBackend, allocator: std.mem.Allocator, shape: *const Shape, dtype: DType) !Tensor {
        var dims = try af.ops.ztToAfDims(shape);
        const ndims = shape.ndim();
        var arr = try af.Array.randu(allocator, @intCast(ndims), dims, af.ops.ztToAfType(dtype));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    ndims,
                ),
            ),
        );
    }

    // --------------------------- Tensor Operators ---------------------------
    pub fn constant(_: *const ArrayFireBackend, allocator: std.mem.Allocator, shape: ?*const Shape, value: f64, dtype: DType) !Tensor {
        const ndim: usize = if (shape != null) shape.?.ndim() else 0;
        const afNdim: usize = if (shape != null) shape.?.ndim() else 1;
        const afDim = if (shape != null) try af.ops.ztToAfDims(shape.?) else af.Dim4{};
        var arr: *af.Array = try af.Array.constant(
            allocator,
            value,
            @intCast(afNdim),
            afDim,
            af.ops.ztToAfType(dtype),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    ndim,
                ),
            ),
        );
    }

    pub fn identity(_: *const ArrayFireBackend, allocator: std.mem.Allocator, dim: Dim, dtype: DType) !Tensor {
        const dims = af.Dim4{ .dims = [_]af.dim_t{ @intCast(dim), @intCast(dim), 1, 1 } };
        var arr = try af.ops.identity(allocator, 2, dims, af.ops.ztToAfType(dtype));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    2,
                ),
            ),
        );
    }

    pub fn arange(_: *const ArrayFireBackend, allocator: std.mem.Allocator, shape: *const Shape, seq_dim: Dim, dtype: DType) !Tensor {
        var dims = try af.ops.ztToAfDims(shape);
        const ndims = shape.ndim();
        var arr = try af.ops.range(
            allocator,
            @intCast(ndims),
            dims,
            @intCast(seq_dim),
            af.ops.ztToAfType(dtype),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    ndims,
                ),
            ),
        );
    }

    pub fn iota(_: *const ArrayFireBackend, allocator: std.mem.Allocator, dims: *const Shape, tile_dims: *const Shape, dtype: DType) !Tensor {
        var afDims = try af.ops.ztToAfDims(dims);
        var afTileDims = try af.ops.ztToAfDims(tile_dims);
        var arr = try af.ops.iota(
            allocator,
            @intCast(afDims.ndims()),
            afDims,
            @intCast(afTileDims.ndims()),
            afTileDims,
            af.ops.ztToAfType(dtype),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    @max(afDims.ndims(), afTileDims.ndims()),
                ),
            ),
        );
    }

    pub fn where(_: *const ArrayFireBackend, allocator: std.mem.Allocator, condition: Tensor, x: Tensor, y: Tensor) !Tensor {
        var orig: Tensor = x;
        try af.ops.replace(try toArray(allocator, orig), try toArray(allocator, condition), try toArray(allocator, y));
        return orig;
    }

    pub fn topk(
        _: *const ArrayFireBackend,
        allocator: std.mem.Allocator,
        input: Tensor,
        k: u32,
        axis: Dim,
        sort_mode: SortMode,
    ) !TopkRes {
        var output = try af.ops.topk(
            allocator,
            try toArray(allocator, input),
            @intCast(k),
            @intCast(axis),
            af.ops.ztToAfTopKSortMode(sort_mode),
        );
        return .{
            .values = Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        output.values,
                        try input.ndim(allocator),
                    ),
                ),
            ),
            .indices = Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        output.indices,
                        try input.ndim(allocator),
                    ),
                ),
            ),
        };
    }

    pub fn sort(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axis: Dim, sort_mode: SortMode) !Tensor {
        if (sort_mode != .Descending and sort_mode != .Ascending) {
            std.log.err(
                "Cannot sort ArrayFire tensor with given SortMode: only Descending and Ascending supported.\n",
                .{},
            );
            return error.UnsupportedSortMode;
        }
        var arr = try af.ops.sort(allocator, try toArray(allocator, input), @intCast(axis), sort_mode == .Ascending);
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try input.ndim(allocator),
                ),
            ),
        );
    }

    pub fn sortIndex(
        _: *const ArrayFireBackend,
        allocator: std.mem.Allocator,
        input: Tensor,
        axis: Dim,
        sort_mode: SortMode,
    ) !SortIndexRes {
        if (sort_mode != .Descending and sort_mode != .Ascending) {
            std.log.err(
                "Cannot sort ArrayFire tensor with given SortMode: only Descending and Ascending supported.\n",
                .{},
            );
            return error.UnsupportedSortMode;
        }
        var output = try af.ops.sortIndex(allocator, try toArray(allocator, input), @intCast(axis), sort_mode == .Ascending);
        return .{
            .out = Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        output.out,
                        try input.ndim(allocator),
                    ),
                ),
            ),
            .idx = Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        output.idx,
                        try input.ndim(allocator),
                    ),
                ),
            ),
        };
    }

    pub fn argsort(
        _: *const ArrayFireBackend,
        allocator: std.mem.Allocator,
        input: Tensor,
        axis: Dim,
        sort_mode: SortMode,
    ) !Tensor {
        if (sort_mode != .Descending and sort_mode != .Ascending) {
            std.log.err(
                "Cannot sort ArrayFire tensor with given SortMode: only Descending and Ascending supported.\n",
                .{},
            );
            return error.UnsupportedSortMode;
        }
        var output = try af.ops.sortIndex(allocator, try toArray(allocator, input), @intCast(axis), sort_mode == .Ascending);
        defer output.out.deinit();
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    output.idx,
                    try input.ndim(allocator),
                ),
            ),
        );
    }

    pub fn matmul(
        _: *const ArrayFireBackend,
        allocator: std.mem.Allocator,
        lhs: Tensor,
        rhs: Tensor,
        lhs_prop: MatrixProperty,
        rhs_prop: MatrixProperty,
    ) !Tensor {
        var lhsProp = lhs_prop;
        var rhsProp = rhs_prop;
        var lhsNumDims = try lhs.ndim(allocator);
        var rhsNumDims = try rhs.ndim(allocator);
        var numDims = @max(lhsNumDims, rhsNumDims);
        if ((lhsNumDims == 1 or rhsNumDims == 1) and numDims > 1) {
            numDims -= 1;
        }

        var lhsArray = try toArray(allocator, lhs);
        var modLhsArray = false;
        defer if (modLhsArray) lhsArray.deinit();
        var rhsArray = try toArray(allocator, rhs);
        var modRhsArray = false;
        defer if (modRhsArray) rhsArray.deinit();

        if (lhsNumDims == 1 and rhsNumDims == 1) {
            // Simulate a dot product by transposing the lhs:
            // (1, k) x (k, 1) --> (1, 1) --> reshape to (1)
            // Ignore other transposes since 1D tensors are the transpose of themselves.
            // ArrayFire would otherwise transpose a (k) tensor to (1, k) since (k) =
            // (k, 1, 1, 1) and ArrayFire transpose transposes the first two dimensions.
            lhsProp = .Transpose;
            rhsProp = .None;
            numDims = 1;
        } else {
            if (rhsNumDims == 1) {
                var dims = af.Dim4{};
                dims.dims[0] = @intCast(try rhs.dim(allocator, 0));
                rhsArray = try af.ops.modDims(allocator, rhsArray, 2, dims);
                modRhsArray = true;
            }
            if (lhsNumDims == 1) {
                var dims = af.Dim4{};
                dims.dims[0] = @intCast(try lhs.dim(allocator, 0));
                lhsArray = try af.ops.modDims(allocator, lhsArray, 2, dims);
                modLhsArray = true;
            }
        }

        var arr = try af.ops.matmul(
            allocator,
            lhsArray,
            rhsArray,
            af.ops.ztToAfMatrixProperty(lhsProp),
            af.ops.ztToAfMatrixProperty(rhsProp),
        );

        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    numDims,
                ),
            ),
        );
    }

    pub fn reshape(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor, shape: *const Shape) !Tensor {
        var arr = try af.ops.modDims(
            allocator,
            try toArray(allocator, tensor),
            @intCast(shape.ndim()),
            try af.ops.ztToAfDims(shape),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    shape.ndim(),
                ),
            ),
        );
    }

    pub fn transpose(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor, axes: *const Shape) !Tensor {
        var currentDims = try tensor.ndim(allocator);
        if (currentDims == 1) {
            return tensor;
        } else if (currentDims == 2 and (axes.ndim() == 0 or (axes.dims_.items.len == 2 and axes.dims_.items[0] == 1 and axes.dims_.items[1] == 0))) {
            // fastpath for matrices
            var arr = try af.ops.transpose(allocator, try toArray(allocator, tensor), false);
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        arr,
                        currentDims,
                    ),
                ),
            );
        } else if (axes.ndim() == 0) {
            var dims = std.simd.iota(u32, @intCast(af.AF_MAX_DIMS));
            // Compute the reversed dimensions for as many ndims as are in the input
            for (0..currentDims) |i| dims[i] = @intCast(currentDims - 1 - i);

            // flip all dimensions
            var arr = try af.ops.reorder(allocator, try toArray(allocator, tensor), dims[0], dims[1], dims[2], dims[3]);
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        arr,
                        currentDims,
                    ),
                ),
            );
        } else {
            if (axes.ndim() > @as(usize, @intCast(af.AF_MAX_DIMS))) {
                std.log.err("ArrayFire tensor transpose was given permutation dims with > 4 axes\n", .{});
                return error.ArrayFireTransposeFailed;
            }
            if (axes.ndim() != currentDims) {
                std.log.err("ArrayFire tensor transpose axes don't match tensor's for permutation - axes must have the same number of dimensions as the tensor\n", .{});
                return error.ArrayFireTransposeFailed;
            }
            // reorder based on specified dimensions
            var d = std.simd.iota(u32, @intCast(af.AF_MAX_DIMS));
            for (0..axes.ndim()) |i| {
                if (axes.dims_.items[i] > currentDims - 1) {
                    std.log.err("ArrayFireBackend.transpose - given dimension is larger than the number of dimensions in the tensor\n", .{});
                    return error.ArrayFireTransposeFailed;
                }
                d[i] = @intCast(axes.dims_.items[i]);
            }
            var arr = try af.ops.reorder(allocator, try toArray(allocator, tensor), d[0], d[1], d[2], d[3]);
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        arr,
                        currentDims,
                    ),
                ),
            );
        }
    }

    pub fn tile(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor, shape: *const Shape) !Tensor {
        var afDims = try af.ops.ztToAfDims(shape);
        var arr = try af.ops.tile(
            allocator,
            try toArray(allocator, tensor),
            @intCast(afDims.dims[0]),
            @intCast(afDims.dims[1]),
            @intCast(afDims.dims[2]),
            @intCast(afDims.dims[3]),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    @max(try tensor.ndim(allocator), shape.ndim()), // TODO: check
                ),
            ),
        );
    }

    pub fn concatenate(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensors: *const std.ArrayList(Tensor), axis: u32) !Tensor {
        var arrays = try allocator.alloc(af.af_array, tensors.items.len);
        defer allocator.free(arrays);
        for (tensors.items, 0..) |t, i| {
            var arr = try toArray(allocator, t);
            arrays[i] = arr.array_;
        }
        var afArray: af.af_array = undefined;
        try af.AF_CHECK(af.af_join_many(&afArray, @intCast(axis), @intCast(arrays.len), arrays.ptr), @src());
        var arr = try af.Array.init(allocator, afArray);
        var numDims = try tensors.items[0].ndim(allocator);
        if (axis > @max(@as(u32, @intCast(numDims - 1)), @as(u32, 0))) {
            numDims = @intCast(axis + 1);
        }

        // All tensors have the same numdims else AF would throw
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    numDims,
                ),
            ),
        );
    }

    pub fn nonzero(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.where(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    1,
                ),
            ),
        );
    }

    pub fn pad(
        _: *const ArrayFireBackend,
        allocator: std.mem.Allocator,
        input: Tensor,
        pad_widths: *const std.ArrayList([2]i32),
        pad_type: PadType,
    ) !Tensor {
        if (pad_widths.items.len > @as(usize, @intCast(af.AF_MAX_DIMS))) {
            std.log.err("ArrayFireBackend::pad - given padWidths for more than 4 dimensions\n", .{});
            return error.ArrayFirePadFailed;
        }

        // convert ((begin_1, end_1), ..., (begin_k, end_k)) to ((begin_1, ...,
        // begin_k), (end_1, ..., end_k)) for ArrayFire
        var beginPadding = af.Dim4{};
        var endPadding = af.Dim4{};
        for (pad_widths.items, 0..) |w, i| {
            beginPadding.dims[i] = @intCast(w[0]);
            endPadding.dims[i] = @intCast(w[1]);
        }

        var arr = try af.ops.pad(
            allocator,
            try toArray(allocator, input),
            @intCast(pad_widths.items.len),
            beginPadding,
            @intCast(pad_widths.items.len),
            endPadding,
            af.ops.ztToAfBorderType(pad_type),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    @max(try input.ndim(allocator), pad_widths.items.len), // TODO: check
                ),
            ),
        );
    }

    pub fn exp(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.exp(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn log(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.log(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    // TODO: pub fn negative(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {}

    pub fn logicalNot(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.not(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn log1p(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.log1p(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn sin(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.sin(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn cos(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.cos(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn sqrt(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.sqrt(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn tanh(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.tanh(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn floor(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.floor(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn ceil(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.ceil(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn rint(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.round(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn absolute(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.abs(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn sigmoid(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.sigmoid(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn erf(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.erf(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn flip(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor, dim: u32) !Tensor {
        var arr = try af.ops.flip(allocator, try toArray(allocator, tensor), dim);
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn clip(
        _: *const ArrayFireBackend,
        allocator: std.mem.Allocator,
        tensor: Tensor,
        low: Tensor,
        high: Tensor,
        batch: bool,
    ) !Tensor {
        var arr = try af.ops.clamp(
            allocator,
            try toArray(allocator, tensor),
            try toArray(allocator, low),
            try toArray(allocator, high),
            batch,
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn roll(
        _: *const ArrayFireBackend,
        allocator: std.mem.Allocator,
        tensor: Tensor,
        shift: Dim,
        axis: usize,
    ) !Tensor {
        var shifts = [_]i32{0} ** @as(usize, @intCast(af.AF_MAX_DIMS));
        shifts[axis] = @intCast(shift);
        var arr = try af.ops.shift(
            allocator,
            try toArray(allocator, tensor),
            shifts[0],
            shifts[1],
            shifts[2],
            shifts[3],
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn isnan(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.isNan(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    pub fn isinf(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.isInf(allocator, try toArray(allocator, tensor));
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
    }

    // TODO: pub fn sign(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {}

    pub fn tril(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.lower(allocator, try toArray(allocator, tensor), false);
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator), // TODO: check
                ),
            ),
        );
    }

    pub fn triu(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var arr = try af.ops.upper(allocator, try toArray(allocator, tensor), false);
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try tensor.ndim(allocator), // TODO: check
                ),
            ),
        );
    }

    pub fn amin(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: std.ArrayList(i32), keep_dims: bool) !Tensor {
        if (try isAllAxisReduction(allocator, input, axes)) {
            // Reduce along all axes returning a singleton tensor
            var arr = try af.ops.minAllArray(allocator, try toArray(allocator, input));
            var res = try condenseIndices(allocator, arr, false, null, false);
            defer if (res.modified) arr.deinit();
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        res.arr,
                        0,
                    ),
                ),
            );
        } else {
            var arr = try afReduceAxes(
                allocator,
                try toArray(allocator, input),
                axes,
                reduceFunc_t,
                af.ops.min,
                keep_dims,
            );
            var numDims = getReducedNumDims(usize, try input.ndim(allocator), axes.items.len, keep_dims);
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        arr,
                        numDims,
                    ),
                ),
            );
        }
    }

    pub fn amax(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: std.ArrayList(i32), keep_dims: bool) !Tensor {
        if (try isAllAxisReduction(allocator, input, axes)) {
            // Reduce along all axes returning a singleton tensor
            var arr = try af.ops.maxAllArray(allocator, try toArray(allocator, input));
            var res = try condenseIndices(allocator, arr, false, null, false);
            defer if (res.modified) arr.deinit();
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        res.arr,
                        0,
                    ),
                ),
            );
        } else {
            var arr = try afReduceAxes(
                allocator,
                try toArray(allocator, input),
                axes,
                reduceFunc_t,
                af.ops.max,
                keep_dims,
            );
            var numDims = getReducedNumDims(usize, try input.ndim(allocator), axes.items.len, keep_dims);
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        arr,
                        numDims,
                    ),
                ),
            );
        }
    }
};

pub fn canBroadcast(lhs: *const Shape, rhs: *const Shape) bool {
    var nDim: usize = @max(lhs.ndim(), rhs.ndim());
    for (0..nDim) |i| {
        if (i + 1 > lhs.ndim() or i + 1 > rhs.ndim()) {
            // One Shape has more dimensions than the other - will broadcast to the
            // smaller tensor
            continue;
        }
        if (lhs.dims_.items[i] != rhs.dims_.items[i] and lhs.dims_.items[i] != 1 and rhs.dims_.items[i] != 1) {
            return false;
        }
    }
    return true;
}

test "ArrayFireBackend supportsDataType" {
    var allocator = std.testing.allocator;
    var backend = try ArrayFireBackend.getInstance(allocator);
    defer backend.deinit();

    // access and release DeviceManager singleton here so test doesn't leak
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    var device_types = getDeviceTypes();
    var iterator = device_types.iterator();
    while (iterator.next()) |d_type| {
        if (mgr.isDeviceTypeAvailable(d_type)) {
            var devices = try mgr.getDevicesOfType(allocator, d_type);
            defer allocator.free(devices);
            for (devices) |dev| {
                try dev.setActive();
            }
        }
    }

    try std.testing.expect(try backend.supportsDataType(.f16));
}

test "ArrayFireBackend getInstance (singleton)" {
    const allocator = std.testing.allocator;
    var b1 = try ArrayFireBackend.getInstance(allocator);
    defer b1.deinit();

    // access and release DeviceManager singleton here so test doesn't leak
    var mgr = try DeviceManager.getInstance(allocator);
    defer mgr.deinit();

    // b2 doesn't need `deinit` called as it's a singleton
    var b2 = try ArrayFireBackend.getInstance(allocator);
    try std.testing.expect(b1 == b2);
}
