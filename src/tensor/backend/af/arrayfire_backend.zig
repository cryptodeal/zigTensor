const std = @import("std");
const af = @import("../../../bindings/af/arrayfire.zig");
const base = @import("../../tensor_base.zig");
const zt_shape = @import("../../shape.zig");
const zt_types = @import("../../types.zig");
const build_options = @import("build_options");
const rt_stream = @import("../../../runtime/stream.zig");
const rt_device_manager = @import("../../../runtime/device_manager.zig");
const zigrc = @import("zigrc");
const rt_device_type = @import("../../../runtime/device_type.zig");
const reductions = @import("reductions.zig");
const zt_backend = @import("../../tensor_backend.zig");
const zt_index = @import("../../index.zig");
const af_utils = @import("utils.zig");

const assert = std.debug.assert;
const gforGet = af.gforGet;
const batchFunc = af.batchFunc;
const batchFunc_t = af.batchFunc_t;
const inPlaceBatchFunc_t = af.inPlaceBatchFunc_t;
const inPlaceBatchFunc = af.inPlaceBatchFunc;
const Index = zt_index.Index;
const IndexType = zt_index.IndexType;
const deinit = @import("../../init.zig").deinit;
const condenseIndices = af_utils.condenseIndices;
const gForDim = af_utils.gForDim;
const seqToDims = af_utils.seqToDims;
const gForReorder = af_utils.gForReorder;
const isAllAxisReduction = reductions.isAllAxisReduction;
const afReduceAxes = reductions.afReduceAxes;
const reduceFunc_t = reductions.reduceFunc_t;
const getReducedNumDims = reductions.getReducedNumDims;
const getDeviceTypes = rt_device_type.getDeviceTypes;
const ArrayFireCPUStream = @import("arrayfire_cpu_stream.zig").ArrayFireCPUStream;
const Arc = zigrc.Arc;
const ArrayFireTensor = @import("arrayfire_tensor.zig").ArrayFireTensor;
const toArray = @import("arrayfire_tensor.zig").toArray;
const ZT_BACKEND_CUDA = build_options.ZT_BACKEND_CUDA;
const ZT_BACKEND_CPU = build_options.ZT_BACKEND_CPU;
const ZT_ARRAYFIRE_USE_CUDA = build_options.ZT_ARRAYFIRE_USE_CUDA;
const ZT_ARRAYFIRE_USE_CPU = build_options.ZT_ARRAYFIRE_USE_CPU;
const SortMode = base.SortMode;
const PadType = base.PadType;
const MatrixProperty = base.MatrixProperty;
const TensorBackendType = base.TensorBackendType;
const TensorAdapterBase = @import("../../tensor_adapter.zig").TensorAdapterBase;
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
        std.log.debug("ArrayFireBackend was not compiled with support for CPU or GPU\n", .{});
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

pub fn deinitBackend() void {
    if (ArrayFireBackendSingleton != null) {
        ArrayFireBackendSingleton.?.deinit();
    }
}

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

    pub fn randn(_: *const ArrayFireBackend, allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
        var dims = try af.ops.ztToAfDims(shape);
        const ndims = zt_shape.ndim(shape);
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

    pub fn rand(_: *const ArrayFireBackend, allocator: std.mem.Allocator, shape: Shape, dtype: DType) !Tensor {
        var dims = try af.ops.ztToAfDims(shape);
        const ndims = zt_shape.ndim(shape);
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
    pub fn fromScalar(_: *const ArrayFireBackend, allocator: std.mem.Allocator, value: f64, dtype: DType) !Tensor {
        var arr: *af.Array = try af.Array.constant(
            allocator,
            value,
            1,
            af.Dim4{},
            af.ops.ztToAfType(dtype),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    0,
                ),
            ),
        );
    }

    pub fn fromScalarI64(_: *const ArrayFireBackend, allocator: std.mem.Allocator, value: i64) !Tensor {
        var arr: *af.Array = try af.Array.constantI64(
            allocator,
            value,
            1,
            af.Dim4{},
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    0,
                ),
            ),
        );
    }

    pub fn fromScalarU64(_: *const ArrayFireBackend, allocator: std.mem.Allocator, value: u64) !Tensor {
        var arr: *af.Array = try af.Array.constantU64(
            allocator,
            value,
            1,
            af.Dim4{},
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    0,
                ),
            ),
        );
    }

    pub fn fromSlice(_: *const ArrayFireBackend, allocator: std.mem.Allocator, s: Shape, data: ?*anyopaque, dtype: DType) !Tensor {
        var arr: *af.Array = try af.ops.createArray(
            allocator,
            data,
            @intCast(zt_shape.ndim(s)),
            try af.ops.ztToAfDims(s),
            af.ops.ztToAfType(dtype),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    zt_shape.ndim(s),
                ),
            ),
        );
    }

    pub fn full(_: *const ArrayFireBackend, allocator: std.mem.Allocator, shape: Shape, value: f64, dtype: DType) !Tensor {
        var arr: *af.Array = try af.Array.constant(
            allocator,
            value,
            @intCast(zt_shape.ndim(shape)),
            try af.ops.ztToAfDims(shape),
            af.ops.ztToAfType(dtype),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    zt_shape.ndim(shape),
                ),
            ),
        );
    }

    pub fn fullI64(_: *const ArrayFireBackend, allocator: std.mem.Allocator, shape: Shape, value: i64) !Tensor {
        var arr: *af.Array = try af.Array.constantI64(
            allocator,
            value,
            @intCast(zt_shape.ndim(shape)),
            try af.ops.ztToAfDims(shape),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    zt_shape.ndim(shape),
                ),
            ),
        );
    }

    pub fn fullU64(_: *const ArrayFireBackend, allocator: std.mem.Allocator, shape: Shape, value: u64) !Tensor {
        var arr: *af.Array = try af.Array.constantU64(
            allocator,
            value,
            @intCast(zt_shape.ndim(shape)),
            try af.ops.ztToAfDims(shape),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    zt_shape.ndim(shape),
                ),
            ),
        );
    }

    pub fn identity(_: *const ArrayFireBackend, allocator: std.mem.Allocator, dim: Dim, dtype: DType) !Tensor {
        const dims = af.Dim4.init(&.{ @as(af.dim_t, @intCast(dim)), @as(af.dim_t, @intCast(dim)) });
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

    pub fn arange(_: *const ArrayFireBackend, allocator: std.mem.Allocator, shape: Shape, seq_dim: Dim, dtype: DType) !Tensor {
        var dims = try af.ops.ztToAfDims(shape);
        const ndims = zt_shape.ndim(shape);
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

    pub fn iota(_: *const ArrayFireBackend, allocator: std.mem.Allocator, dims: Shape, tile_dims: Shape, dtype: DType) !Tensor {
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
        var orig = try Tensor.initAssign(allocator, x);
        try af.ops.replace(try toArray(allocator, orig), try toArray(allocator, condition), try toArray(allocator, y));
        return orig;
    }

    pub fn topk(
        _: *const ArrayFireBackend,
        allocator: std.mem.Allocator,
        values: Tensor,
        indices: Tensor,
        input: Tensor,
        k: u32,
        axis: Dim,
        sort_mode: SortMode,
    ) !void {
        var output = try af.ops.topk(
            allocator,
            try toArray(allocator, input),
            @intCast(k),
            @intCast(axis),
            af.ops.ztToAfTopKSortMode(sort_mode),
        );
        var tmp_vals = Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    output.values,
                    try input.ndim(allocator),
                ),
            ),
        );
        defer tmp_vals.deinit();
        try values.assign(allocator, Tensor, tmp_vals);
        var tmp_indices = Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    output.indices,
                    try input.ndim(allocator),
                ),
            ),
        );
        defer tmp_indices.deinit();
        try indices.assign(allocator, Tensor, tmp_indices);
    }

    pub fn sort(_: *const ArrayFireBackend, allocator: std.mem.Allocator, values: Tensor, indices: ?Tensor, input: Tensor, axis: Dim, sort_mode: SortMode) !void {
        if (sort_mode != .Descending and sort_mode != .Ascending) {
            std.log.debug(
                "Cannot sort ArrayFire tensor with given SortMode: only Descending and Ascending supported.\n",
                .{},
            );
            return error.UnsupportedSortMode;
        }
        if (indices == null) {
            var arr = try af.ops.sort(allocator, try toArray(allocator, input), @intCast(axis), sort_mode == .Ascending);
            var tmp_values = Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        arr,
                        try input.ndim(allocator),
                    ),
                ),
            );
            defer tmp_values.deinit();
            try values.assign(allocator, Tensor, tmp_values);
        } else {
            var output = try af.ops.sortIndex(allocator, try toArray(allocator, input), @intCast(axis), sort_mode == .Ascending);
            var tmp_vals = Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        output.out,
                        try input.ndim(allocator),
                    ),
                ),
            );
            defer tmp_vals.deinit();
            try values.assign(allocator, Tensor, tmp_vals);
            var tmp_indices = Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        output.idx,
                        try input.ndim(allocator),
                    ),
                ),
            );
            defer tmp_indices.deinit();
            try indices.?.assign(allocator, Tensor, tmp_indices);
        }
    }

    pub fn argsort(
        _: *const ArrayFireBackend,
        allocator: std.mem.Allocator,
        input: Tensor,
        axis: Dim,
        sort_mode: SortMode,
    ) !Tensor {
        if (sort_mode != .Descending and sort_mode != .Ascending) {
            std.log.debug(
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
        var numDims = @max(try lhs.ndim(allocator), try rhs.ndim(allocator));
        if ((try lhs.ndim(allocator) == 1 or try rhs.ndim(allocator) == 1) and numDims > 1) {
            numDims -= 1;
        }

        var lhsArray = try toArray(allocator, lhs);
        var modLhsArray = false;
        defer if (modLhsArray) lhsArray.deinit();
        var rhsArray = try toArray(allocator, rhs);
        var modRhsArray = false;
        defer if (modRhsArray) rhsArray.deinit();

        if (try lhs.ndim(allocator) == 1 and try rhs.ndim(allocator) == 1) {
            // Simulate a dot product by transposing the lhs:
            // (1, k) x (k, 1) --> (1, 1) --> reshape to (1)
            // Ignore other transposes since 1D tensors are the transpose of themselves.
            // ArrayFire would otherwise transpose a (k) tensor to (1, k) since (k) =
            // (k, 1, 1, 1) and ArrayFire transpose transposes the first two dimensions.
            lhsProp = .Transpose;
            rhsProp = .None;
            numDims = 1;
        } else {
            if (try rhs.ndim(allocator) == 1) {
                var dims = af.Dim4.init(&.{@as(af.dim_t, @intCast(try rhs.dim(allocator, 0)))});
                rhsArray = try af.ops.moddims(allocator, rhsArray, 2, dims);
                modRhsArray = true;
            }
            if (try lhs.ndim(allocator) == 1) {
                var dims = af.Dim4.init(&.{ 1, @as(af.dim_t, @intCast(try lhs.dim(allocator, 0))) });
                lhsArray = try af.ops.moddims(allocator, lhsArray, 2, dims);
                modLhsArray = true;
            }
        }

        var res = try af.ops.matmul(
            allocator,
            lhsArray,
            rhsArray,
            af.ops.ztToAfMatrixProperty(lhsProp),
            af.ops.ztToAfMatrixProperty(rhsProp),
        );

        var arr = res;
        if (try lhs.ndim(allocator) == 1 and try rhs.ndim(allocator) == 2) {
            var current_dims = try res.getDims();
            var new_dims = af.Dim4.init(&.{current_dims.dims[1]});
            arr = try af.ops.moddims(allocator, res, 1, new_dims);
            res.deinit();
        }

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

    pub fn reshape(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor, shape: Shape) !Tensor {
        var af_dims = try af.ops.ztToAfDims(shape);
        var arr = try af.ops.moddims(
            allocator,
            try toArray(allocator, tensor),
            @intCast(af_dims.ndims()),
            af_dims,
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    zt_shape.ndim(shape),
                ),
            ),
        );
    }

    pub fn transpose(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor, axes: Shape) !Tensor {
        var currentDims = try tensor.ndim(allocator);
        if (currentDims == 1) {
            return tensor;
        } else if (currentDims == 2 and (zt_shape.ndim(axes) == 0 or (axes.len == 2 and axes[0] == 1 and axes[1] == 0))) {
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
        } else if (zt_shape.ndim(axes) == 0) {
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
            if (zt_shape.ndim(axes) > @as(usize, @intCast(af.AF_MAX_DIMS))) {
                std.log.debug("ArrayFire tensor transpose was given permutation dims with > 4 axes\n", .{});
                return error.ArrayFireTransposeFailed;
            }
            if (zt_shape.ndim(axes) != currentDims) {
                std.log.debug("ArrayFire tensor transpose axes don't match tensor's for permutation - axes must have the same number of dimensions as the tensor\n", .{});
                return error.ArrayFireTransposeFailed;
            }
            // reorder based on specified dimensions
            var d = std.simd.iota(u32, @intCast(af.AF_MAX_DIMS));
            for (0..zt_shape.ndim(axes)) |i| {
                if (axes[i] > currentDims - 1) {
                    std.log.debug("ArrayFireBackend.transpose - given dimension is larger than the number of dimensions in the tensor\n", .{});
                    return error.ArrayFireTransposeFailed;
                }
                d[i] = @intCast(axes[i]);
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

    pub fn tile(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor, shape: Shape) !Tensor {
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
                    @max(try tensor.ndim(allocator), zt_shape.ndim(shape)), // TODO: check
                ),
            ),
        );
    }

    pub fn concatenate(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensors: []const Tensor, axis: u32) !Tensor {
        var arrays = try allocator.alloc(af.af_array, tensors.len);
        defer allocator.free(arrays);
        for (tensors, 0..) |t, i| {
            var arr = try toArray(allocator, t);
            arrays[i] = arr.array_;
        }
        var afArray: af.af_array = undefined;
        try af.AF_CHECK(af.af_join_many(&afArray, @intCast(axis), @intCast(arrays.len), arrays.ptr), @src());
        var arr = try af.Array.init(allocator, afArray);
        var numDims = try tensors[0].ndim(allocator);
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
        pad_widths: []const [2]i64,
        pad_type: PadType,
    ) !Tensor {
        if (pad_widths.len > @as(usize, @intCast(af.AF_MAX_DIMS))) {
            std.log.debug("ArrayFireBackend::pad - given padWidths for more than 4 dimensions\n", .{});
            return error.ArrayFirePadFailed;
        }

        // convert ((begin_1, end_1), ..., (begin_k, end_k)) to ((begin_1, ...,
        // begin_k), (end_1, ..., end_k)) for ArrayFire
        var beginPadding = af.Dim4{};
        var endPadding = af.Dim4{};
        for (pad_widths, 0..) |w, i| {
            beginPadding.dims[i] = @intCast(w[0]);
            endPadding.dims[i] = @intCast(w[1]);
        }

        var arr = try af.ops.pad(
            allocator,
            try toArray(allocator, input),
            @intCast(pad_widths.len),
            beginPadding,
            @intCast(pad_widths.len),
            endPadding,
            af.ops.ztToAfBorderType(pad_type),
        );
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    @max(try input.ndim(allocator), pad_widths.len), // TODO: check
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

    pub fn negative(self: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var shape = try tensor.shape(allocator);
        var neg_const = try self.full(allocator, shape, -1, try tensor.dtype(allocator));
        defer neg_const.deinit();
        return self.mul(allocator, tensor, neg_const);
    }

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
    ) !Tensor {
        var arr = try af.ops.clamp(
            allocator,
            try toArray(allocator, tensor),
            try toArray(allocator, low),
            try toArray(allocator, high),
            gforGet(),
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

    pub fn sign(self: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !Tensor {
        var sign_arr = try af.ops.sign(allocator, try toArray(allocator, tensor));
        var tmp = Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    sign_arr,
                    try tensor.ndim(allocator),
                ),
            ),
        );
        try tmp.inPlaceMul(allocator, f64, 2);
        var ones = try self.full(allocator, try tensor.shape(allocator), 1, try tensor.dtype(allocator));
        var sign_tensor = try self.sub(allocator, ones, tmp);
        tmp.deinit();
        ones.deinit();
        var zero_const = try self.full(allocator, try tensor.shape(allocator), 0, try tensor.dtype(allocator));
        var eql = try self.eq(allocator, tensor, zero_const);
        defer eql.deinit();
        zero_const.deinit();
        try sign_tensor.indexAssign(allocator, f64, 0, &.{Index.initTensor(eql)});
        return sign_tensor;
    }

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

    pub fn amin(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
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
            var numDims = getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims);
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

    pub fn amax(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
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
            var numDims = getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims);
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

    pub fn min(_: *const ArrayFireBackend, allocator: std.mem.Allocator, values: Tensor, indices: Tensor, input: Tensor, axis: u32, keep_dims: bool) !void {
        var res = try af.ops.imin(allocator, try toArray(allocator, input), @intCast(axis));
        var cond_vals = try condenseIndices(allocator, res.out, keep_dims, null, false);
        defer if (cond_vals.modified) res.out.deinit();
        var cond_idx = try condenseIndices(allocator, res.idx, keep_dims, null, false);
        defer if (cond_idx.modified) res.idx.deinit();
        var tmp_vals = Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    cond_vals.arr,
                    getReducedNumDims(usize, try input.ndim(allocator), 1, keep_dims),
                ),
            ),
        );
        defer tmp_vals.deinit();
        try values.assign(allocator, Tensor, tmp_vals);
        var tmp_indices = Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    cond_idx.arr,
                    getReducedNumDims(usize, try input.ndim(allocator), 1, keep_dims),
                ),
            ),
        );
        defer tmp_indices.deinit();
        try indices.assign(allocator, Tensor, tmp_indices);
    }

    pub fn max(_: *const ArrayFireBackend, allocator: std.mem.Allocator, values: Tensor, indices: Tensor, input: Tensor, axis: u32, keep_dims: bool) !void {
        var res = try af.ops.imax(allocator, try toArray(allocator, input), @intCast(axis));
        var cond_vals = try condenseIndices(allocator, res.out, keep_dims, null, false);
        defer if (cond_vals.modified) res.out.deinit();
        var cond_idx = try condenseIndices(allocator, res.idx, keep_dims, null, false);
        defer if (cond_idx.modified) res.idx.deinit();
        var tmp_vals = Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    cond_vals.arr,
                    getReducedNumDims(usize, try input.ndim(allocator), 1, keep_dims),
                ),
            ),
        );
        defer tmp_vals.deinit();
        try values.assign(allocator, Tensor, tmp_vals);
        var tmp_indices = Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    cond_idx.arr,
                    getReducedNumDims(usize, try input.ndim(allocator), 1, keep_dims),
                ),
            ),
        );
        defer tmp_indices.deinit();
        try indices.assign(allocator, Tensor, tmp_indices);
    }

    pub fn sum(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        if (try isAllAxisReduction(allocator, input, axes)) {
            // std.debug.print("isAllAxisReduction = true\n", .{});
            var arr = try af.ops.sumAllArray(allocator, try toArray(allocator, input));
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
            // std.debug.print("isAllAxisReduction = false\n", .{});
            var arr = try afReduceAxes(
                allocator,
                try toArray(allocator, input),
                axes,
                reduceFunc_t,
                af.ops.sum,
                keep_dims,
            );
            var numDims = getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims);
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

    pub fn cumsum(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axis: u32) !Tensor {
        var arr = try af.ops.accum(allocator, try toArray(allocator, input), @intCast(axis));
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

    pub fn argmax(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !Tensor {
        var res = try af.ops.imax(allocator, try toArray(allocator, input), @intCast(axis));
        defer res.out.deinit();
        var cond_idx = try condenseIndices(allocator, res.idx, keep_dims, null, false);
        defer if (cond_idx.modified) res.idx.deinit();
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    cond_idx.arr,
                    getReducedNumDims(usize, try input.ndim(allocator), 1, keep_dims),
                ),
            ),
        );
    }

    pub fn argmin(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axis: u32, keep_dims: bool) !Tensor {
        var res = try af.ops.imin(allocator, try toArray(allocator, input), @intCast(axis));
        defer res.out.deinit();
        var cond_idx = try condenseIndices(allocator, res.idx, keep_dims, null, false);
        defer if (cond_idx.modified) res.idx.deinit();
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    cond_idx.arr,
                    getReducedNumDims(usize, try input.ndim(allocator), 1, keep_dims),
                ),
            ),
        );
    }

    pub fn mean(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        if (try isAllAxisReduction(allocator, input, axes)) {
            var arr = try toArray(allocator, input);
            for (0..@intCast(af.AF_MAX_DIMS)) |i| {
                var og_arr = arr;
                defer if (i != 0) og_arr.deinit();
                arr = try af.ops.mean(allocator, arr, -1);
            }
            var res = try condenseIndices(allocator, arr, keep_dims, null, false);
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
                *const fn (allocator: std.mem.Allocator, in: *const af.Array, dim: i64) anyerror!*af.Array,
                af.ops.mean,
                keep_dims,
            );
            var numDims = getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims);
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

    pub fn median(self: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        if (try isAllAxisReduction(allocator, input, axes)) {
            // Reduce along all axes returning a singleton tensor
            // TODO: modify this to `medianAllArray` to take advantage of the
            // ArrayFire reduce_all kernels once available
            var res = try af.ops.medianAll(try toArray(allocator, input));
            return self.fromScalar(allocator, res.real, .f32);
        } else {
            var arr = try afReduceAxes(
                allocator,
                try toArray(allocator, input),
                axes,
                *const fn (allocator: std.mem.Allocator, in: *const af.Array, dim: i64) anyerror!*af.Array,
                af.ops.median,
                keep_dims,
            );
            var numDims = getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims);
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

    pub fn variance(self: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, bias: bool, keep_dims: bool) !Tensor {
        var bias_mode: af.VarBias = if (bias) .Sample else .Population;
        // Use ArrayFire default for one dimension which may be optimized
        var arr = try toArray(allocator, input);
        // Reduce along all axes returning a singleton tensor
        // TODO: modify this to af_var_all_array_v2 to take advantage of the
        // ArrayFire reduce_all kernels once available
        if (try isAllAxisReduction(allocator, input, axes)) {
            var out = try af.ops.varAllV2(arr, bias_mode);
            return self.fromScalar(allocator, out.real, .f32);
        } else if (axes.len == 1) {
            var var_arr = try af.ops.varV2(allocator, arr, bias_mode, @intCast(axes[0]));
            var res = try condenseIndices(allocator, var_arr, keep_dims, null, false);
            defer if (res.modified) var_arr.deinit();
            var num_dims = getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims);
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        res.arr,
                        num_dims,
                    ),
                ),
            );
        } else {
            var mean_arr = try self.mean(allocator, input, axes, true);
            defer mean_arr.deinit();
            var x = try batchFunc(allocator, arr, try toArray(allocator, mean_arr), af.ops.sub);
            defer x.deinit();
            var x_squared = try af.ops.pow2(allocator, x);
            defer x_squared.deinit();
            var x_reduced = try afReduceAxes(allocator, x_squared, axes, reduceFunc_t, af.ops.sum, true);
            defer x_reduced.deinit();

            var denominator: i64 = 1;
            var dims = try arr.getDims();
            for (axes) |dim| {
                denominator *= @intCast(dims.dims[@intCast(dim)]);
            }
            if (bias) denominator -= 1;
            var rhs_arr = try af.ops.constant(
                allocator,
                @floatFromInt(denominator),
                try x_reduced.getNumDims(),
                try x_reduced.getDims(),
                try x_reduced.getType(),
            );
            defer rhs_arr.deinit();
            var div_arr = try af.ops.div(allocator, x_reduced, rhs_arr, false);
            var res = try condenseIndices(allocator, div_arr, keep_dims, null, false);
            defer if (res.modified) div_arr.deinit();
            var num_dims = getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims);
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        res.arr,
                        num_dims,
                    ),
                ),
            );
        }
    }

    pub fn stdev(self: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        const bias = false; // TODO: make this configurable
        const bias_mode: af.VarBias = if (bias) .Sample else .Population;
        if (try isAllAxisReduction(allocator, input, axes)) {
            // TODO: update to af.af_std_dev_all_array_v2 once specialization is available
            var out = try af.ops.stdevAllV2(try toArray(allocator, input), bias_mode);
            return self.fromScalar(allocator, out.real, .f32);
        } else if (axes.len == 1) {
            // Use arrayfire default for one dimension which may be optimized
            // TODO: update this? stddev is deprecated.
            var arr = try af.ops.stdevV2(allocator, try toArray(allocator, input), bias_mode, @intCast(axes[0]));
            var cond_arr = try condenseIndices(allocator, arr, keep_dims, null, false);
            defer if (cond_arr.modified) arr.deinit();
            var num_dims = getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims);
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        cond_arr.arr,
                        num_dims,
                    ),
                ),
            );
        }
        var var_tensor = try self.variance(allocator, input, axes, bias, keep_dims);
        defer var_tensor.deinit();
        return self.sqrt(allocator, var_tensor);
    }

    pub fn norm(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, p: f64, keep_dims: bool) !Tensor {
        if (try isAllAxisReduction(allocator, input, axes)) {
            // TODO: update to af_norm_all_array if device-side specialization is
            // available. Either that or use the all-axis specializations with the below
            // implementation
            var flat_arr = try af.ops.flat(allocator, try toArray(allocator, input));
            defer flat_arr.deinit();
            var abs_arr = try af.ops.abs(allocator, flat_arr);
            defer abs_arr.deinit();
            var const_arr = try af.ops.constant(allocator, p, try abs_arr.getNumDims(), try abs_arr.getDims(), try abs_arr.getType());
            defer const_arr.deinit();
            var pow_arr = try af.ops.pow(allocator, abs_arr, const_arr, false);
            defer pow_arr.deinit();
            var sum_arr = try af.ops.sumAllArray(allocator, pow_arr);
            defer sum_arr.deinit();
            var denom_arr = try af.ops.constant(allocator, 1 / p, try sum_arr.getNumDims(), try sum_arr.getDims(), try sum_arr.getType());
            defer denom_arr.deinit();
            var res1 = try af.ops.pow(allocator, sum_arr, denom_arr, false);
            var condensed = try condenseIndices(allocator, res1, false, null, false);
            defer if (condensed.modified) res1.deinit();
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        condensed.arr,
                        0,
                    ),
                ),
            );
        } else {
            var abs_arr = try af.ops.abs(allocator, try toArray(allocator, input));
            defer abs_arr.deinit();
            var const_arr = try af.ops.constant(allocator, p, try abs_arr.getNumDims(), try abs_arr.getDims(), try abs_arr.getType());
            defer const_arr.deinit();
            var pow_arr = try af.ops.pow(allocator, abs_arr, const_arr, false);
            defer pow_arr.deinit();
            var reduced_arr = try afReduceAxes(allocator, pow_arr, axes, reduceFunc_t, af.ops.sum, keep_dims);
            defer reduced_arr.deinit();
            var denom_arr = try af.ops.constant(allocator, 1 / p, try reduced_arr.getNumDims(), try reduced_arr.getDims(), try reduced_arr.getType());
            defer denom_arr.deinit();
            var res = try af.ops.pow(allocator, reduced_arr, denom_arr, false);
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        res,
                        getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims),
                    ),
                ),
            );
        }
    }

    pub fn countNonzero(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        var arr = try toArray(allocator, input);
        var num_dims: usize = undefined;
        var out: *af.Array = undefined;
        if (try isAllAxisReduction(allocator, input, axes)) {
            var count_arr = try af.ops.count(allocator, arr, -1);
            defer count_arr.deinit();
            var sum_arr = try af.ops.sumAllArray(allocator, count_arr);
            var condensed = try condenseIndices(allocator, sum_arr, keep_dims, null, false);
            defer if (condensed.modified) sum_arr.deinit();
            out = condensed.arr;
            num_dims = 0;
        } else if (axes.len == 1) {
            out = try af.ops.count(allocator, arr, @intCast(axes[0]));
            num_dims = getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims);
        } else {
            var count_arr = try af.ops.count(allocator, arr, @intCast(axes[0]));
            defer count_arr.deinit();
            out = try afReduceAxes(allocator, count_arr, axes[1..], reduceFunc_t, af.ops.sum, keep_dims);
            num_dims = getReducedNumDims(usize, try input.ndim(allocator), axes.len, keep_dims);
        }
        var condensed = try condenseIndices(allocator, out, keep_dims, null, false);
        defer if (condensed.modified) out.deinit();
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    condensed.arr,
                    num_dims,
                ),
            ),
        );
    }

    pub fn any(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        if (try isAllAxisReduction(allocator, input, axes)) {
            // Reduce along all axes returning a singleton tensor
            var arr = try af.ops.anyTrueAllArray(allocator, try toArray(allocator, input));
            var condensed = try condenseIndices(allocator, arr, false, null, false);
            defer if (condensed.modified) arr.deinit();
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        condensed.arr,
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
                af.ops.anyTrue,
                keep_dims,
            );
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        arr,
                        getReducedNumDims(
                            usize,
                            try input.ndim(allocator),
                            axes.len,
                            keep_dims,
                        ),
                    ),
                ),
            );
        }
    }

    pub fn all(_: *const ArrayFireBackend, allocator: std.mem.Allocator, input: Tensor, axes: []const i64, keep_dims: bool) !Tensor {
        if (try isAllAxisReduction(allocator, input, axes)) {
            // Reduce along all axes returning a singleton tensor
            var arr = try af.ops.allTrueAllArray(allocator, try toArray(allocator, input));
            var condensed = try condenseIndices(allocator, arr, false, null, false);
            defer if (condensed.modified) arr.deinit();
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        condensed.arr,
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
                af.ops.allTrue,
                keep_dims,
            );
            return Tensor.init(
                TensorAdapterBase.init(
                    try ArrayFireTensor.initFromArray(
                        allocator,
                        arr,
                        getReducedNumDims(
                            usize,
                            try input.ndim(allocator),
                            axes.len,
                            keep_dims,
                        ),
                    ),
                ),
            );
        }
    }

    pub fn assign(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
        (lhs.getAdapter(ArrayFireTensor)).numDims_ = try rhs.ndim(allocator);
        return af.ops.assign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
        );
    }

    pub fn getIndexAssignShape(_: *const ArrayFireBackend, allocator: std.mem.Allocator, target: Tensor, indices: []const Index) ![]Dim {
        if (indices.len > @as(usize, @intCast(af.AF_MAX_DIMS))) {
            std.log.debug(
                "ArrayFire-backed tensor was indexed with > 4 elements: ArrayFire tensors support up to 4 dimensions.\n",
                .{},
            );
            return error.IndicesExceedMaxDims;
        }

        const completeTensorIndex = indices.len == 1 and indices[0].idxType() == .Tensor and try indices[0].index_.Tensor.elements(allocator) == @as(usize, @intCast(try (try toArray(allocator, target)).getElements()));
        var afIndices = try af.ops.createIndexers(); // this creates implicit spans for up to maxDims
        defer af.ops.releaseIndexers(afIndices) catch unreachable;
        if (completeTensorIndex) {
            // TODO: verify this is correct; needs tests
            try af.ops.setSeqParamIndexer(afIndices, 0, 0, 1, 0, false);
        }

        if (indices.len > afIndices.len) {
            std.log.debug("ArrayFireTensor.index internal error - passed indices is larger than the number of af indices.\n", .{});
            return error.PassedIndicesLargerThanAfIndices;
        }

        // Fill in corresponding index types for each af index
        var i: usize = 0;
        while (i < indices.len) : (i += 1) {
            afIndices[i] = af.ops.ztToAfIndex(indices[i]);
        }

        var target_array = try toArray(allocator, target);
        var p_dims = try target_array.getDims();
        if (completeTensorIndex) {
            p_dims = af.Dim4.init(&.{@as(af.dim_t, @intCast(p_dims.elements()))});
        }
        var dims = try seqToDims(afIndices, p_dims, true);
        return dims.dimsToOwnedShape(allocator);
    }

    fn flatOpAssign(_: *const ArrayFireBackend, impl: *af.Array, other: *af.Array, indices: []af.af_index_t, comptime op: []const u8) !void {
        var nd: u32 = 1;
        var this_dims = try impl.getDims();
        var other_dims = try other.getDims();
        var dim = gForDim(indices);
        var other_arr = other.get();

        var tmp_idx: af.af_array = null;
        if (!indices[0].isSeq) {
            var idx_type: af.af_dtype = undefined;
            try af.AF_CHECK(af.af_get_type(&idx_type, indices[0].idx.arr), @src());
            if (idx_type == af.b8) {
                try af.AF_CHECK(af.af_where(&tmp_idx, indices[0].idx.arr), @src());
                indices[0].idx.arr = tmp_idx;
            }
        }

        var batch_assign = false;
        var is_reordered = false;
        if (dim >= 0) {
            // FIXME: Figure out a faster, cleaner way to do this
            var out_dims = try seqToDims(indices, this_dims, false);

            batch_assign = true;
            for (0..@intCast(af.AF_MAX_DIMS)) |i| {
                if (indices[i].isBatch) {
                    var tmp_batch_assign = @intFromBool(batch_assign);
                    tmp_batch_assign &= @intFromBool(other_dims.dims[i] == 1);
                    batch_assign = tmp_batch_assign != 0;
                } else {
                    var tmp_batch_assign = @intFromBool(batch_assign);
                    tmp_batch_assign &= @intFromBool(other_dims.dims[i] == out_dims.dims[i]);
                    batch_assign = tmp_batch_assign != 0;
                }
            }

            if (batch_assign) {
                var out: af.af_array = undefined;
                try af.AF_CHECK(
                    af.af_tile(
                        &out,
                        other_arr,
                        @intCast(@divTrunc(out_dims.dims[0], other_dims.dims[0])),
                        @intCast(@divTrunc(out_dims.dims[1], other_dims.dims[1])),
                        @intCast(@divTrunc(out_dims.dims[2], other_dims.dims[2])),
                        @intCast(@divTrunc(out_dims.dims[3], other_dims.dims[3])),
                    ),
                    @src(),
                );
                other_arr = out;
            } else if (!std.mem.eql(af.dim_t, &out_dims.dims, &other_dims.dims)) {
                // HACK: This is a quick check to see if other has been reordered
                // inside gfor
                // TODO: Figure out if this breaks and implement a cleaner
                // method
                other_arr = try gForReorder(other_arr, @intCast(dim));
                is_reordered = true;
            }
        }

        var par_arr = impl.get();
        var tmp_lhs: af.af_array = null;
        try af.AF_CHECK(af.af_index_gen(&tmp_lhs, par_arr, @intCast(nd), indices.ptr), @src());
        var op_res: af.af_array = null;
        if (comptime std.mem.eql(u8, op, "+=")) {
            try af.AF_CHECK(af.af_add(&op_res, tmp_lhs, other_arr, false), @src());
        } else if (comptime std.mem.eql(u8, op, "-=")) {
            try af.AF_CHECK(af.af_sub(&op_res, tmp_lhs, other_arr, false), @src());
        } else if (comptime std.mem.eql(u8, op, "*=")) {
            try af.AF_CHECK(af.af_mul(&op_res, tmp_lhs, other_arr, false), @src());
        } else if (comptime std.mem.eql(u8, op, "/=")) {
            try af.AF_CHECK(af.af_div(&op_res, tmp_lhs, other_arr, false), @src());
        } else {
            @compileError("Unsupported operation passed to idxOpAssign\n");
        }

        var res: af.af_array = null;
        try af.AF_CHECK(
            af.af_assign_gen(
                &res,
                par_arr,
                @intCast(nd),
                indices.ptr,
                op_res,
            ),
            @src(),
        );
        try af.AF_CHECK(af.af_release_array(tmp_lhs), @src());
        try af.AF_CHECK(af.af_release_array(op_res), @src());

        try impl.set(res);
        if (dim >= 0 and (is_reordered or batch_assign)) {
            if (other_arr != null) try af.AF_CHECK(af.af_release_array(other_arr), @src());
        }
        if (tmp_idx != null) try af.AF_CHECK(af.af_release_array(tmp_idx), @src());
    }

    fn flatIdxAssign(_: *const ArrayFireBackend, impl: *af.Array, other: *af.Array, indices: []af.af_index_t) !void {
        var nd: u32 = 1;
        var this_dims = try impl.getDims();
        var other_dims = try other.getDims();
        var dim = gForDim(indices);
        var other_arr = other.get();

        var tmp_idx: af.af_array = null;
        if (!indices[0].isSeq) {
            var idx_type: af.af_dtype = undefined;
            try af.AF_CHECK(af.af_get_type(&idx_type, indices[0].idx.arr), @src());
            if (idx_type == af.b8) {
                try af.AF_CHECK(af.af_where(&tmp_idx, indices[0].idx.arr), @src());
                indices[0].idx.arr = tmp_idx;
            }
        }

        var batch_assign = false;
        var is_reordered = false;
        if (dim >= 0) {
            // FIXME: Figure out a faster, cleaner way to do this
            var out_dims = try seqToDims(indices, this_dims, false);

            batch_assign = true;
            for (0..@intCast(af.AF_MAX_DIMS)) |i| {
                if (indices[i].isBatch) {
                    var tmp_batch_assign = @intFromBool(batch_assign);
                    tmp_batch_assign &= @intFromBool(other_dims.dims[i] == 1);
                    batch_assign = tmp_batch_assign != 0;
                } else {
                    var tmp_batch_assign = @intFromBool(batch_assign);
                    tmp_batch_assign &= @intFromBool(other_dims.dims[i] == out_dims.dims[i]);
                    batch_assign = tmp_batch_assign != 0;
                }
            }

            if (batch_assign) {
                var out: af.af_array = undefined;
                try af.AF_CHECK(
                    af.af_tile(
                        &out,
                        other_arr,
                        @intCast(@divTrunc(out_dims.dims[0], other_dims.dims[0])),
                        @intCast(@divTrunc(out_dims.dims[1], other_dims.dims[1])),
                        @intCast(@divTrunc(out_dims.dims[2], other_dims.dims[2])),
                        @intCast(@divTrunc(out_dims.dims[3], other_dims.dims[3])),
                    ),
                    @src(),
                );
                other_arr = out;
            } else if (!std.mem.eql(af.dim_t, &out_dims.dims, &other_dims.dims)) {
                // HACK: This is a quick check to see if other has been reordered
                // inside gfor
                // TODO: Figure out if this breaks and implement a cleaner
                // method
                other_arr = try gForReorder(other_arr, @intCast(dim));
                is_reordered = true;
            }
        }

        var par_arr = impl.get();
        var res: af.af_array = null;
        try af.AF_CHECK(
            af.af_assign_gen(
                &res,
                par_arr,
                @intCast(nd),
                indices.ptr,
                other_arr,
            ),
            @src(),
        );

        try impl.set(res);
        if (dim >= 0 and (is_reordered or batch_assign)) {
            if (other_arr != null) try af.AF_CHECK(af.af_release_array(other_arr), @src());
        }
        if (tmp_idx != null) try af.AF_CHECK(af.af_release_array(tmp_idx), @src());
    }

    pub fn flatAssign(self: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, idx: Index) !void {
        // Return a lazy indexing operation. Indexing with a single index on an
        // ArrayFire tensor (with a type that is not an af::array) ends up doing
        // flat indexing, so all index assignment operators will work as they are.
        var indices = try af.ops.createIndexers();
        indices[0] = af.ops.ztToAfIndex(idx);

        try self.flatIdxAssign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
            indices,
        );
        try af.ops.releaseIndexers(indices);
    }

    pub fn flatAdd(self: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, idx: Index) !void {
        // Return a lazy indexing operation. Indexing with a single index on an
        // ArrayFire tensor (with a type that is not an af::array) ends up doing
        // flat indexing, so all index assignment operators will work as they are.
        var indices = try af.ops.createIndexers();
        indices[0] = af.ops.ztToAfIndex(idx);

        try self.flatOpAssign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
            indices,
            "+=",
        );
        try af.ops.releaseIndexers(indices);
    }

    pub fn flatSub(self: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, idx: Index) !void {
        // Return a lazy indexing operation. Indexing with a single index on an
        // ArrayFire tensor (with a type that is not an af::array) ends up doing
        // flat indexing, so all index assignment operators will work as they are.
        var indices = try af.ops.createIndexers();
        indices[0] = af.ops.ztToAfIndex(idx);

        try self.flatOpAssign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
            indices,
            "-=",
        );
        try af.ops.releaseIndexers(indices);
    }

    pub fn flatMul(self: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, idx: Index) !void {
        // Return a lazy indexing operation. Indexing with a single index on an
        // ArrayFire tensor (with a type that is not an af::array) ends up doing
        // flat indexing, so all index assignment operators will work as they are.
        var indices = try af.ops.createIndexers();
        indices[0] = af.ops.ztToAfIndex(idx);

        try self.flatOpAssign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
            indices,
            "*=",
        );
        try af.ops.releaseIndexers(indices);
    }

    pub fn flatDiv(self: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, idx: Index) !void {
        // Return a lazy indexing operation. Indexing with a single index on an
        // ArrayFire tensor (with a type that is not an af::array) ends up doing
        // flat indexing, so all index assignment operators will work as they are.
        var indices = try af.ops.createIndexers();
        indices[0] = af.ops.ztToAfIndex(idx);

        try self.flatOpAssign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
            indices,
            "/=",
        );
        try af.ops.releaseIndexers(indices);
    }

    fn idxOpAssign(_: *const ArrayFireBackend, impl: *af.Array, other: *af.Array, indices: []af.af_index_t, is_linear: bool, comptime op: []const u8) !void {
        var nd = try impl.getNumDims();
        var this_dims = try impl.getDims();
        var other_dims = try other.getDims();
        var dim = gForDim(indices);
        var other_arr = other.get();

        var tmp_arrs = [_]af.af_array{null} ** 4;
        // if idx == bool type array, use where to get indices
        for (indices, 0..) |v, i| {
            if (v.isSeq) continue;
            var idx_type: af.af_dtype = undefined;
            try af.AF_CHECK(af.af_get_type(&idx_type, v.idx.arr), @src());
            if (idx_type == af.b8) {
                try af.AF_CHECK(af.af_where(&tmp_arrs[i], v.idx.arr), @src());
                indices[i].idx.arr = tmp_arrs[i];
            }
        }

        var batch_assign = false;
        var is_reordered = false;
        if (dim >= 0) {
            // FIXME: Figure out a faster, cleaner way to do this
            var out_dims = try seqToDims(indices, this_dims, false);

            batch_assign = true;
            for (0..@intCast(af.AF_MAX_DIMS)) |i| {
                if (indices[i].isBatch) {
                    var tmp_batch_assign = @intFromBool(batch_assign);
                    tmp_batch_assign &= @intFromBool(other_dims.dims[i] == 1);
                    batch_assign = tmp_batch_assign != 0;
                } else {
                    var tmp_batch_assign = @intFromBool(batch_assign);
                    tmp_batch_assign &= @intFromBool(other_dims.dims[i] == out_dims.dims[i]);
                    batch_assign = tmp_batch_assign != 0;
                }
            }

            if (batch_assign) {
                var out: af.af_array = undefined;
                try af.AF_CHECK(
                    af.af_tile(
                        &out,
                        other_arr,
                        @intCast(@divTrunc(out_dims.dims[0], other_dims.dims[0])),
                        @intCast(@divTrunc(out_dims.dims[1], other_dims.dims[1])),
                        @intCast(@divTrunc(out_dims.dims[2], other_dims.dims[2])),
                        @intCast(@divTrunc(out_dims.dims[3], other_dims.dims[3])),
                    ),
                    @src(),
                );
                other_arr = out;
            } else if (!std.mem.eql(af.dim_t, &out_dims.dims, &other_dims.dims)) {
                // HACK: This is a quick check to see if other has been reordered
                // inside gfor
                // TODO: Figure out if this breaks and implement a cleaner
                // method
                other_arr = try gForReorder(other_arr, @intCast(dim));
                is_reordered = true;
            }
        }

        var par_arr: af.af_array = null;

        var parent_dims = try impl.getDims();
        if (is_linear) {
            try af.AF_CHECK(af.af_flat(&par_arr, impl.get()), @src());

            // The set call will dereference the impl->parent_ array. We are doing
            // this because the af_flat call above increases the reference count of
            // the parent array which triggers a copy operation. This triggers a
            // copy operation inside the af_assign_gen function below. The parent
            // array will be reverted to the original array and shape later in the
            // code.
            try impl.release(); // also sets underlying array to null
            nd = 1;
        } else {
            par_arr = impl.get();
        }

        var tmp_lhs: af.af_array = null;
        try af.AF_CHECK(af.af_index_gen(&tmp_lhs, par_arr, @intCast(nd), indices.ptr), @src());
        var op_res: af.af_array = null;
        if (comptime std.mem.eql(u8, op, "+=")) {
            try af.AF_CHECK(af.af_add(&op_res, tmp_lhs, other_arr, false), @src());
        } else if (comptime std.mem.eql(u8, op, "-=")) {
            try af.AF_CHECK(af.af_sub(&op_res, tmp_lhs, other_arr, false), @src());
        } else if (comptime std.mem.eql(u8, op, "*=")) {
            try af.AF_CHECK(af.af_mul(&op_res, tmp_lhs, other_arr, false), @src());
        } else if (comptime std.mem.eql(u8, op, "/=")) {
            try af.AF_CHECK(af.af_div(&op_res, tmp_lhs, other_arr, false), @src());
        } else {
            @compileError("Unsupported operation passed to idxOpAssign\n");
        }

        var flat_res: af.af_array = null;
        try af.AF_CHECK(
            af.af_assign_gen(
                &flat_res,
                par_arr,
                @intCast(nd),
                indices.ptr,
                op_res,
            ),
            @src(),
        );
        try af.AF_CHECK(af.af_release_array(tmp_lhs), @src());
        try af.AF_CHECK(af.af_release_array(op_res), @src());

        var res: af.af_array = null;
        var unflattened: af.af_array = null;
        if (is_linear) {
            try af.AF_CHECK(
                af.af_moddims(
                    &res,
                    flat_res,
                    @intCast(this_dims.ndims()),
                    &this_dims.dims,
                ),
                @src(),
            );

            // Unflatten the af_array and reset the original reference
            try af.AF_CHECK(
                af.af_moddims(
                    &unflattened,
                    par_arr,
                    @intCast(parent_dims.ndims()),
                    &parent_dims.dims,
                ),
                @src(),
            );
            try impl.set(unflattened);
            try af.AF_CHECK(af.af_release_array(par_arr), @src());
            try af.AF_CHECK(af.af_release_array(flat_res), @src());
        } else {
            res = flat_res;
        }

        try impl.set(res);
        if (dim >= 0 and (is_reordered or batch_assign)) {
            if (other_arr != null) try af.AF_CHECK(af.af_release_array(other_arr), @src());
        }
        for (tmp_arrs) |v| if (v != null) try af.AF_CHECK(af.af_release_array(v), @src());
    }

    fn idxAssign(_: *const ArrayFireBackend, impl: *af.Array, other: *af.Array, indices: []af.af_index_t, is_linear: bool) !void {
        var nd = try impl.getNumDims();
        var this_dims = try impl.getDims();
        var other_dims = try other.getDims();
        var dim = gForDim(indices);
        var other_arr = other.get();

        var tmp_arrs = [_]af.af_array{null} ** 4;
        // if idx == bool type array, use where to get indices
        for (indices, 0..) |v, i| {
            if (v.isSeq) continue;
            var idx_type: af.af_dtype = undefined;
            try af.AF_CHECK(af.af_get_type(&idx_type, v.idx.arr), @src());
            if (idx_type == af.b8) {
                try af.AF_CHECK(af.af_where(&tmp_arrs[i], v.idx.arr), @src());
                indices[i].idx.arr = tmp_arrs[i];
            }
        }

        var batch_assign = false;
        var is_reordered = false;
        if (dim >= 0) {
            // FIXME: Figure out a faster, cleaner way to do this
            var out_dims = try seqToDims(indices, this_dims, false);

            batch_assign = true;
            for (0..@intCast(af.AF_MAX_DIMS)) |i| {
                if (indices[i].isBatch) {
                    var tmp_batch_assign = @intFromBool(batch_assign);
                    tmp_batch_assign &= @intFromBool(other_dims.dims[i] == 1);
                    batch_assign = tmp_batch_assign != 0;
                } else {
                    var tmp_batch_assign = @intFromBool(batch_assign);
                    tmp_batch_assign &= @intFromBool(other_dims.dims[i] == out_dims.dims[i]);
                    batch_assign = tmp_batch_assign != 0;
                }
            }

            if (batch_assign) {
                var out: af.af_array = undefined;
                try af.AF_CHECK(
                    af.af_tile(
                        &out,
                        other_arr,
                        @intCast(@divTrunc(out_dims.dims[0], other_dims.dims[0])),
                        @intCast(@divTrunc(out_dims.dims[1], other_dims.dims[1])),
                        @intCast(@divTrunc(out_dims.dims[2], other_dims.dims[2])),
                        @intCast(@divTrunc(out_dims.dims[3], other_dims.dims[3])),
                    ),
                    @src(),
                );
                other_arr = out;
            } else if (!std.mem.eql(af.dim_t, &out_dims.dims, &other_dims.dims)) {
                // HACK: This is a quick check to see if other has been reordered
                // inside gfor
                // TODO: Figure out if this breaks and implement a cleaner
                // method
                other_arr = try gForReorder(other_arr, @intCast(dim));
                is_reordered = true;
            }
        }

        var par_arr: af.af_array = null;

        var parent_dims = try impl.getDims();
        if (is_linear) {
            try af.AF_CHECK(af.af_flat(&par_arr, impl.get()), @src());

            // The set call will dereference the impl->parent_ array. We are doing
            // this because the af_flat call above increases the reference count of
            // the parent array which triggers a copy operation. This triggers a
            // copy operation inside the af_assign_gen function below. The parent
            // array will be reverted to the original array and shape later in the
            // code.
            try impl.release(); // also sets underlying array to null
            nd = 1;
        } else {
            par_arr = impl.get();
        }

        var flat_res: af.af_array = null;
        try af.AF_CHECK(
            af.af_assign_gen(
                &flat_res,
                par_arr,
                @intCast(nd),
                indices.ptr,
                other_arr,
            ),
            @src(),
        );

        var res: af.af_array = null;
        var unflattened: af.af_array = null;
        if (is_linear) {
            try af.AF_CHECK(
                af.af_moddims(
                    &res,
                    flat_res,
                    @intCast(this_dims.ndims()),
                    (&this_dims.dims).ptr,
                ),
                @src(),
            );
            // Unflatten the af_array and reset the original reference
            try af.AF_CHECK(
                af.af_moddims(
                    &unflattened,
                    par_arr,
                    @intCast(parent_dims.ndims()),
                    (&parent_dims.dims).ptr,
                ),
                @src(),
            );
            try impl.set(unflattened);
            try af.AF_CHECK(af.af_release_array(par_arr), @src());
            try af.AF_CHECK(af.af_release_array(flat_res), @src());
        } else {
            res = flat_res;
        }

        try impl.set(res);
        if (dim >= 0 and (is_reordered or batch_assign)) {
            if (other_arr != null) try af.AF_CHECK(af.af_release_array(other_arr), @src());
        }
        for (tmp_arrs) |v| if (v != null) try af.AF_CHECK(af.af_release_array(v), @src());
    }

    pub fn indexAssign(self: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) !void {
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
        const completeTensorIndex = indices.len == 1 and indices[0].idxType() == .Tensor and try indices[0].index_.Tensor.elements(allocator) == @as(usize, @intCast(try (try toArray(allocator, lhs)).getElements()));
        var afIndices = try af.ops.createIndexers(); // this creates implicit spans for up to maxDims
        if (completeTensorIndex) {
            // TODO: verify this is correct; needs tests
            try af.ops.setSeqParamIndexer(afIndices, 0, 0, 1, 0, false);
        }

        if (indices.len > afIndices.len) {
            std.log.debug("ArrayFireTensor.indexAssign internal error - passed indices is larger than the number of af indices.\n", .{});
            return error.PassedIndicesLargerThanAfIndices;
        }

        // Fill in corresponding index types for each af index
        var i: usize = 0;
        while (i < indices.len) : (i += 1) {
            afIndices[i] = af.ops.ztToAfIndex(indices[i]);
        }

        try self.idxAssign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
            afIndices,
            completeTensorIndex, // verify this is correct
        );
        try af.ops.releaseIndexers(afIndices);
    }

    pub fn indexAdd(self: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) !void {
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
        const completeTensorIndex = indices.len == 1 and indices[0].idxType() == .Tensor and try indices[0].index_.Tensor.elements(allocator) == @as(usize, @intCast(try (try toArray(allocator, lhs)).getElements()));
        var afIndices = try af.ops.createIndexers(); // this creates implicit spans for up to maxDims
        if (completeTensorIndex) {
            // TODO: verify this is correct; needs tests
            try af.ops.setSeqParamIndexer(afIndices, 0, 0, 1, 0, false);
        }

        if (indices.len > afIndices.len) {
            std.log.debug("ArrayFireTensor.indexAdd internal error - passed indices is larger than the number of af indices.\n", .{});
            return error.PassedIndicesLargerThanAfIndices;
        }

        // Fill in corresponding index types for each af index
        var i: usize = 0;
        while (i < indices.len) : (i += 1) {
            afIndices[i] = af.ops.ztToAfIndex(indices[i]);
        }

        try self.idxOpAssign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
            afIndices,
            completeTensorIndex, // verify this is correct
            "+=",
        );
        try af.ops.releaseIndexers(afIndices);
    }

    pub fn indexSub(self: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) !void {
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
        const completeTensorIndex = indices.len == 1 and indices[0].idxType() == .Tensor and (try indices[0].index_.Tensor.elements(allocator) == @as(usize, @intCast(try (try toArray(allocator, lhs)).getElements())) or try indices[0].index_.Tensor.dtype(allocator) == .b8);
        var afIndices = try af.ops.createIndexers(); // this creates implicit spans for up to maxDims
        if (completeTensorIndex) {
            // TODO: verify this is correct; needs tests
            try af.ops.setSeqParamIndexer(afIndices, 0, 0, 1, 0, false);
        }

        if (indices.len > afIndices.len) {
            std.log.debug("ArrayFireTensor.indexSub internal error - passed indices is larger than the number of af indices.\n", .{});
            return error.PassedIndicesLargerThanAfIndices;
        }

        // Fill in corresponding index types for each af index
        var i: usize = 0;
        while (i < indices.len) : (i += 1) {
            afIndices[i] = af.ops.ztToAfIndex(indices[i]);
        }

        try self.idxOpAssign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
            afIndices,
            completeTensorIndex, // verify this is correct
            "-=",
        );
        try af.ops.releaseIndexers(afIndices);
    }

    pub fn indexMul(self: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) !void {
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
        const completeTensorIndex = indices.len == 1 and indices[0].idxType() == .Tensor and try indices[0].index_.Tensor.elements(allocator) == @as(usize, @intCast(try (try toArray(allocator, lhs)).getElements()));
        var afIndices = try af.ops.createIndexers(); // this creates implicit spans for up to maxDims
        if (completeTensorIndex) {
            // TODO: verify this is correct; needs tests
            try af.ops.setSeqParamIndexer(afIndices, 0, 0, 1, 0, false);
        }

        if (indices.len > afIndices.len) {
            std.log.debug("ArrayFireTensor.indexMul internal error - passed indices is larger than the number of af indices.\n", .{});
            return error.PassedIndicesLargerThanAfIndices;
        }

        // Fill in corresponding index types for each af index
        var i: usize = 0;
        while (i < indices.len) : (i += 1) {
            afIndices[i] = af.ops.ztToAfIndex(indices[i]);
        }

        try self.idxOpAssign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
            afIndices,
            completeTensorIndex, // verify this is correct
            "*=",
        );
        try af.ops.releaseIndexers(afIndices);
    }

    pub fn indexDiv(self: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, indices: []const Index) !void {
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
        const completeTensorIndex = indices.len == 1 and indices[0].idxType() == .Tensor and try indices[0].index_.Tensor.elements(allocator) == @as(usize, @intCast(try (try toArray(allocator, lhs)).getElements()));
        var afIndices = try af.ops.createIndexers(); // this creates implicit spans for up to maxDims
        if (completeTensorIndex) {
            // TODO: verify this is correct; needs tests
            try af.ops.setSeqParamIndexer(afIndices, 0, 0, 1, 0, false);
        }

        if (indices.len > afIndices.len) {
            std.log.debug("ArrayFireTensor.indexDiv internal error - passed indices is larger than the number of af indices.\n", .{});
            return error.PassedIndicesLargerThanAfIndices;
        }

        // Fill in corresponding index types for each af index
        var i: usize = 0;
        while (i < indices.len) : (i += 1) {
            afIndices[i] = af.ops.ztToAfIndex(indices[i]);
        }

        try self.idxOpAssign(
            try toArray(allocator, lhs),
            try toArray(allocator, rhs),
            afIndices,
            completeTensorIndex, // verify this is correct
            "/=",
        );
        try af.ops.releaseIndexers(afIndices);
    }

    pub fn add(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.add);
    }

    // TODO: handle advanced index case for `inPlaceAdd` (only impacts ArrayFire CUDA Backend)
    pub fn inPlaceAdd(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
        return doBinaryOpOrBroadcastInPlace(allocator, lhs, rhs, af.ops.addInplace);
    }

    pub fn sub(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.sub);
    }

    pub fn inPlaceSub(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
        return doBinaryOpOrBroadcastInPlace(allocator, lhs, rhs, af.ops.subInplace);
    }

    pub fn mul(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.mul);
    }

    pub fn inPlaceMul(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
        return doBinaryOpOrBroadcastInPlace(allocator, lhs, rhs, af.ops.mulInplace);
    }

    pub fn div(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.div);
    }

    pub fn inPlaceDiv(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !void {
        return doBinaryOpOrBroadcastInPlace(allocator, lhs, rhs, af.ops.divInplace);
    }

    pub fn eq(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.eq);
    }

    pub fn neq(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.neq);
    }

    pub fn lessThan(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.lt);
    }

    pub fn lessThanEqual(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.le);
    }

    pub fn greaterThan(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.gt);
    }

    pub fn greaterThanEqual(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.ge);
    }

    pub fn logicalOr(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.or_);
    }

    pub fn logicalAnd(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.and_);
    }

    pub fn mod(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.mod);
    }

    pub fn bitwiseAnd(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.bitAnd);
    }

    pub fn bitwiseOr(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.bitOr);
    }

    pub fn bitwiseXor(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.bitXor);
    }

    pub fn lShift(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.bitShiftL);
    }

    pub fn rShift(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.bitShiftR);
    }

    pub fn minimum(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.minOf);
    }

    pub fn maximum(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.maxOf);
    }

    pub fn power(_: *const ArrayFireBackend, allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor) !Tensor {
        return doBinaryOpOrBroadcast(allocator, lhs, rhs, af.ops.pow);
    }

    pub fn print(_: *const ArrayFireBackend, allocator: std.mem.Allocator, tensor: Tensor) !void {
        var arr_string = try (try toArray(allocator, tensor)).toString("ArrayFireTensor", 10, true);
        defer af.ops.freeHost(@ptrCast(@constCast(arr_string.ptr))) catch unreachable;
        std.debug.print("{s}\n", .{arr_string});
    }
};

pub fn canBroadcast(lhs: Shape, rhs: Shape) bool {
    var nDim: usize = @max(zt_shape.ndim(lhs), zt_shape.ndim(rhs));
    for (0..nDim) |i| {
        if (i + 1 > zt_shape.ndim(lhs) or i + 1 > zt_shape.ndim(rhs)) {
            // One Shape has more dimensions than the other - will broadcast to the
            // smaller tensor
            continue;
        }
        if (lhs[i] != rhs[i] and lhs[i] != 1 and rhs[i] != 1) {
            return false;
        }
    }
    return true;
}

pub fn doBinaryOpOrBroadcast(allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, func: batchFunc_t) !Tensor {
    var lhsShape = try lhs.shape(allocator);
    var rhsShape = try rhs.shape(allocator);

    // Dims are the same or scalar <> 1-el tensor - no broadcasting
    if (zt_shape.eql(lhsShape, rhsShape) or (zt_shape.elements(lhsShape) <= 1 and zt_shape.elements(rhsShape) <= 1)) {
        var arr = try func(allocator, try toArray(allocator, lhs), try toArray(allocator, rhs), gforGet());
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    try lhs.ndim(allocator),
                ),
            ),
        );
    }

    if (canBroadcast(lhsShape, rhsShape)) {
        var arr = try batchFunc(allocator, try toArray(allocator, lhs), try toArray(allocator, rhs), func);
        return Tensor.init(
            TensorAdapterBase.init(
                try ArrayFireTensor.initFromArray(
                    allocator,
                    arr,
                    @max(try lhs.ndim(allocator), try rhs.ndim(allocator)),
                ),
            ),
        );
    } else {
        std.log.debug(
            "doBinaryOpOrBroadcast: cannot perform operation or broadcasting with tensors of shapes {any} and {any}  - dimension mismatch.\n",
            .{ lhsShape, rhsShape },
        );
        return error.FailedBinaryOpOrBroadcast;
    }
}

pub fn doBinaryOpOrBroadcastInPlace(allocator: std.mem.Allocator, lhs: Tensor, rhs: Tensor, func: inPlaceBatchFunc_t) !void {
    var lhsShape = try lhs.shape(allocator);
    var rhsShape = try rhs.shape(allocator);

    // Dims are the same or scalar <> 1-el tensor - no broadcasting
    if (zt_shape.eql(lhsShape, rhsShape) or (zt_shape.elements(lhsShape) <= 1 and zt_shape.elements(rhsShape) <= 1)) {
        try func(try toArray(allocator, lhs), try toArray(allocator, rhs), gforGet());
        return;
    }

    if (canBroadcast(lhsShape, rhsShape)) {
        try inPlaceBatchFunc(try toArray(allocator, lhs), try toArray(allocator, rhs), func);
        return;
    } else {
        std.log.debug(
            "doBinaryOpOrBroadcastInPlace: cannot perform operation or broadcasting with tensors of shapes {any} and {any}  - dimension mismatch.\n",
            .{ lhsShape, rhsShape },
        );
        return error.FailedBinaryOpOrBroadcast;
    }
}

test "ArrayFireBackend supportsDataType" {
    var allocator = std.testing.allocator;
    var backend = try ArrayFireBackend.getInstance(allocator);

    // access and release DeviceManager singleton here so test doesn't leak
    var mgr = try DeviceManager.getInstance(allocator);

    // frees both backend and mgr
    defer deinit();

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

    // frees both backend and mgr
    defer deinit();

    // b2 doesn't need `deinit` called as it's a singleton
    var b2 = try ArrayFireBackend.getInstance(allocator);
    try std.testing.expect(b1 == b2);
}
