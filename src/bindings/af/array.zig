const std = @import("std");
const af = @import("arrayfire.zig");
const zt_types = @import("../../tensor/types.zig");
const zt_shape = @import("../../tensor/shape.zig");

const Location = @import("../../tensor/tensor_base.zig").Location;
const Shape = zt_shape.Shape;
const DType = zt_types.DType;

/// Wraps `af.af_array`, the ArrayFire multi dimensional data container,
/// as a zig struct to simplify calling into the ArrayFire C API.
pub const Array = struct {
    const Self = @This();

    array_: af.af_array,
    allocator: std.mem.Allocator,

    pub fn get(self: *const Self) af.af_array {
        return self.array_;
    }

    pub fn set(self: *Self, arr: af.af_array) !void {
        try self.release();
        self.array_ = arr;
    }

    pub fn init(allocator: std.mem.Allocator, arr: af.af_array) !*Self {
        var self = try allocator.create(Self);
        self.* = .{ .array_ = arr, .allocator = allocator };
        return self;
    }

    pub fn initHandle(
        allocator: std.mem.Allocator,
        ndims: u32,
        dims: af.Dim4,
        dtype: af.Dtype,
    ) !*Self {
        return af.ops.createHandle(allocator, ndims, dims, dtype);
    }

    /// Initializes a new instance of `af.Array` from device memory.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn initFromDevicePtr(
        allocator: std.mem.Allocator,
        data: ?*anyopaque,
        ndims: u32,
        dims: af.Dim4,
        dtype: af.Dtype,
    ) !*Self {
        return af.ops.deviceArray(
            allocator,
            data,
            ndims,
            dims,
            dtype,
        );
    }

    /// Returns ptr to an `af.Array` initialized with user defined data.
    pub fn initFromPtr(
        allocator: std.mem.Allocator,
        data: ?*const anyopaque,
        ndims: u32,
        dims: af.Dim4,
        dtype: af.Dtype,
    ) !*Self {
        return af.ops.createArray(
            allocator,
            data,
            ndims,
            dims,
            dtype,
        );
    }

    /// Reads an `af.Array` saved in files using the index
    /// in the file (0-indexed).
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn readIndex(allocator: std.mem.Allocator, filename: []const u8, idx: u32) !*Self {
        return af.ops.readArray(allocator, filename, idx);
    }

    /// Reads an `af.Array` saved in files using the key
    /// that was used along with the `af.Array`.
    ///
    /// Note that if there are multiple `af.Array`s with the
    /// same key, only the first one will be read.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn readKey(allocator: std.mem.Allocator, filename: []const u8, key: []const u8) !*Self {
        return af.ops.readArrayKey(allocator, filename, key);
    }

    /// Create an `af.Array` from a scalar input value.
    ///
    /// The `af.Array` created has the same value at all locations.
    pub fn constant(
        allocator: std.mem.Allocator,
        val: f64,
        ndims: u32,
        dims: af.Dim4,
        dtype: af.Dtype,
    ) !*Self {
        return af.ops.constant(
            allocator,
            val,
            ndims,
            dims,
            dtype,
        );
    }

    /// Create an `af.Array` of type s64 from a scalar input value.
    ///
    /// The `af.Array` created has the same value at all locations.
    pub fn constantI64(
        allocator: std.mem.Allocator,
        val: i64,
        ndims: u32,
        dims: af.Dim4,
    ) !*Self {
        return af.ops.constantI64(allocator, val, ndims, dims);
    }

    /// Create an `af.Array` of type u64 from a scalar input value.
    ///
    /// The `af.Array` created has the same value at all locations.
    pub fn constantU64(
        allocator: std.mem.Allocator,
        val: u64,
        ndims: u32,
        dims: af.Dim4,
    ) !*Self {
        return af.ops.constantU64(allocator, val, ndims, dims);
    }

    /// Creates an `af.Array` with [0, n-1] values along the `seq_dim` dimension
    /// and tiled across other dimensions specified by an array of `ndims` dimensions.
    pub fn range(
        allocator: std.mem.Allocator,
        ndims: u32,
        dims: af.Dim4,
        seq_dim: i32,
        dtype: af.Dtype,
    ) !*Self {
        return af.ops.range(
            allocator,
            ndims,
            dims,
            seq_dim,
            dtype,
        );
    }

    /// Create an sequence [0, dims.elements() - 1] and modify to
    /// specified dimensions dims and then tile it according to `tdims`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn iota(
        allocator: std.mem.Allocator,
        ndims: u32,
        dims: af.Dim4,
        t_ndims: u32,
        tdims: af.Dim4,
        dtype: af.Dtype,
    ) !*Self {
        return af.ops.iota(
            allocator,
            ndims,
            dims,
            t_ndims,
            tdims,
            dtype,
        );
    }

    /// Returns ptr to an identity `af.Array` with diagonal values 1.
    pub fn identity(
        allocator: std.mem.Allocator,
        ndims: u32,
        dims: af.Dim4,
        dtype: af.Dtype,
    ) !*Self {
        return af.ops.identity(allocator, ndims, dims, dtype);
    }

    /// Returns ptr to an `af.Array` of uniform numbers
    /// using a random engine.
    pub fn randomUniform(
        allocator: std.mem.Allocator,
        ndims: u32,
        dims: af.Dim4,
        dtype: af.Dtype,
        engine: *af.RandomEngine,
    ) !*Self {
        return af.ops.randomUniform(allocator, ndims, dims, dtype, engine);
    }

    /// Returns ptr to an `af.Array` of normal numbers
    /// using a random engine.
    pub fn randomNormal(
        allocator: std.mem.Allocator,
        ndims: u32,
        dims: af.Dim4,
        dtype: af.Dtype,
        engine: *af.RandomEngine,
    ) !*Self {
        return af.ops.randomNormal(allocator, ndims, dims, dtype, engine);
    }

    /// Returns ptr to an `af.Array` of uniform numbers
    /// using a random engine.
    pub fn randu(allocator: std.mem.Allocator, ndims: u32, dims: af.Dim4, dtype: af.Dtype) !*Self {
        return af.ops.randu(allocator, ndims, dims, dtype);
    }

    /// Returns ptr to an `af.Array` of normal numbers
    /// using a random engine.
    pub fn randn(allocator: std.mem.Allocator, ndims: u32, dims: af.Dim4, dtype: af.Dtype) !*Self {
        return af.ops.randn(allocator, ndims, dims, dtype);
    }

    /// Converts `af.Array` of values, row indices and column
    /// indices into a sparse `af.Array`.
    pub fn createSparseArray(
        allocator: std.mem.Allocator,
        nRows: i64,
        nCols: i64,
        values: *const Self,
        rowIdx: *const Self,
        colIdx: *const Self,
        stype: af.Storage,
    ) !*Self {
        return af.ops.createSparseArray(
            allocator,
            nRows,
            nCols,
            values,
            rowIdx,
            colIdx,
            stype,
        );
    }

    pub fn assign(self: *Self, other: *const Self) !void {
        return af.ops.assign(self, other);
    }

    /// Converts a dense `af.Array` into a sparse array.
    pub fn createSparseArrayFromDense(allocator: std.mem.Allocator, dense: *const Self, stype: af.Storage) !*Self {
        return af.ops.createSparseArrayFromDense(allocator, dense, stype);
    }

    /// Convert an existing sparse `af.Array` into a different storage format.
    /// Converting storage formats is allowed between `af.Storage.CSR`, `af.Storage.COO`
    /// and `af.Storage.Dense`.
    ///
    /// When converting to `af.Storage.Dense`, a dense array is returned.
    ///
    /// N.B. `af.Storage.CSC` is currently not supported.
    pub fn sparseConvertTo(allocator: std.mem.Allocator, in: *const Self, destStorage: af.Storage) !*Self {
        return af.ops.sparseConvertTo(allocator, in, destStorage);
    }

    /// Returns a dense `af.Array` from a sparse `af.Array`.
    pub fn sparseToDense(allocator: std.mem.Allocator, sparse: *const Self) !*Self {
        return af.ops.sparseToDense(allocator, sparse);
    }

    /// Frees all associated memory;
    /// releases underlying `af.af_array`.
    pub fn deinit(self: *Self) void {
        self.release() catch unreachable;
        self.allocator.destroy(self);
    }

    /// Returns the sum of the elements in this `af.Array`
    /// along the given dimension.
    pub fn sum(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.sum(allocator, self, dim);
    }

    /// Returns the sum of the elements in this `af.Array` along
    /// the given dimension, replacing NaN values with `nanval`.
    pub fn sumNan(self: *const Self, allocator: std.mem.Allocator, dim: i32, nanval: f64) !*Self {
        return af.ops.sumNan(allocator, self, dim, nanval);
    }

    /// Returns the sum of the elements in this `af.Array` by
    /// key along the given dimension according to key.
    pub fn sumByKey(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: i32,
    ) !struct { keys_out: *Self, vals_out: *Self } {
        return af.ops.sumByKey(allocator, keys, self, dim);
    }

    /// Returns the sum of the elements in this `af.Array` by key along
    /// the given dimension according to key; replaces NaN values with
    /// the specified `nanval`.
    pub fn sumByKeyNan(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: i32,
        nanval: f64,
    ) !*Self {
        return af.ops.sumByKeyNan(
            allocator,
            keys,
            self,
            dim,
            nanval,
        );
    }

    /// Returns the product of all values in this `af.Array`
    /// along the specified dimension.
    pub fn product(self: *const Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return af.ops.product(allocator, self, dim);
    }

    /// Returns the product of all values in this `af.Array` along the
    /// specified dimension; replaces NaN values with specified `nanval`.
    pub fn productNan(
        self: *const Self,
        allocator: std.mem.Allocator,
        dim: usize,
        nanval: f64,
    ) !*Self {
        return af.ops.productNan(allocator, self, dim, nanval);
    }

    /// Returns the product of all values in this `af.Array`
    /// along the specified dimension according to key.
    pub fn productByKey(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: i32,
    ) !*Self {
        return af.ops.productByKey(allocator, keys, self, dim);
    }

    /// Returns the sum of the elements in this `af.Array` by key along
    /// the given dimension according to key; replaces NaN values with
    /// the specified `nanval`.
    pub fn productByKeyNan(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: i32,
        nanval: f64,
    ) !*Self {
        return af.ops.productByKeyNan(
            allocator,
            keys,
            self,
            dim,
            nanval,
        );
    }

    /// Returns the minimum of all values in this `af.Array`
    /// along the specified dimension.
    pub fn min(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.min(allocator, self, dim);
    }

    /// Returns the minimum of all values in this `af.Array`
    /// along the specified dimension according to key.
    pub fn minByKey(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: i32,
    ) !*Self {
        return af.ops.minByKey(allocator, keys, self, dim);
    }

    /// Returns the maximum of all values in this `af.Array`
    /// along the specified dimension.
    pub fn max(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.max(allocator, self, dim);
    }

    /// Returns the maximum of all values of this `af.Array`
    /// along the specified dimension according to key.
    pub fn maxByKey(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: i32,
    ) !*Self {
        return af.ops.maxByKey(allocator, keys, self, dim);
    }

    /// Finds ragged max values in this `af.Array`; uses an additional input
    /// `af.Array` to determine the number of elements to use along the
    /// reduction axis.
    ///
    /// Returns struct containing the following fields:
    /// - `val`: `af.Array` containing the maximum ragged values in
    /// this `af.Array` along the specified dimension according
    /// `ragged_len`.
    /// - `idx`: `af.Array` containing the locations of the maximum
    /// ragged values in this `af.Array` along the specified
    /// dimension according to `ragged_len`.
    pub fn maxRagged(
        self: *const Self,
        allocator: std.mem.Allocator,
        ragged_len: *const Self,
        dim: i32,
    ) !*Self {
        return af.ops.maxRagged(allocator, self, ragged_len, dim);
    }

    /// Tests if all values in this `af.Array` along the
    /// specified dimension are true.
    pub fn allTrue(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.allTrue(allocator, self, dim);
    }

    /// Tests if all values in this `af.Array` along the
    /// specified dimension are true accord to key.
    pub fn allTrueByKey(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: i32,
    ) !*Self {
        return af.ops.allTrueByKey(allocator, keys, self, dim);
    }

    /// Tests if any values in this `af.Array` along the
    /// specified dimension are true.
    pub fn anyTrue(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.anyTrue(allocator, self, dim);
    }

    /// Tests if any values in this `af.Array` along the
    /// specified dimension are true according to key.
    pub fn anyTrueByKey(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: i32,
    ) !*Self {
        return af.ops.anyTrueByKey(allocator, keys, self, dim);
    }

    /// Count the number of non-zero elements in this `af.Array`.
    ///
    /// Return type is `af.Dtype.u32` for all input types.
    ///
    /// This function performs the operation across all batches present
    /// in the input simultaneously.
    pub fn count(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.count(allocator, self, dim);
    }

    /// Counts the non-zero values in this `af.Array` according to an
    /// `af.Array` of keys.
    ///
    /// All non-zero values corresponding to each group of consecutive equal
    /// keys will be counted. Keys can repeat, however only consecutive key
    /// values will be considered for each reduction. If a key value is repeated
    /// somewhere else in the keys array it will be considered the start of a new
    /// reduction. There are two outputs: the reduced set of consecutive keys and
    /// the corresponding final reduced values.
    pub fn countByKey(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: i32,
    ) !*Self {
        return af.ops.countByKey(allocator, keys, self, dim);
    }

    /// Returns the sum of all elements in this `af.Array`.
    pub fn sumAll(self: *const Self) !af.ops.ComplexParts {
        return af.ops.sumAll(self);
    }

    /// Returns the sum of all elements in this `af.Array`
    /// replacing NaN values with `nanval`.
    pub fn sumNanAll(self: *const Self, nanval: f64) !af.ops.ComplexParts {
        return af.ops.sumNanAll(self, nanval);
    }

    /// Returns the product of all elements in this `af.Array`.
    pub fn productAll(self: *const Self) !af.ops.ComplexParts {
        return af.ops.productAll(self);
    }

    /// Returns the product of all elements in this `af.Array`
    /// replacing NaN values with `nanval`.
    pub fn productNanAll(self: *const Self, nanval: f64) !af.ops.ComplexParts {
        return af.ops.productNanAll(self, nanval);
    }

    /// Returns the minimum value of all elements in this `af.Array`.
    pub fn minAll(self: *const Self) !af.ops.ComplexParts {
        return af.ops.minAll(self);
    }

    /// Returns the maximum value of all elements in this `af.Array`.
    pub fn maxAll(self: *const Self) !af.ops.ComplexParts {
        return af.ops.maxAll(self);
    }

    /// Returns whether all elements in this `af.Array` are true.
    pub fn allTrueAll(self: *const Self) !af.ops.ComplexParts {
        return af.ops.allTrueAll(self);
    }

    /// Returns whether any elements in this `af.Array` are true.
    pub fn anyTrueAll(self: *const Self) !af.ops.ComplexParts {
        return af.ops.anyTrueAll(self);
    }

    /// Returns the number of non-zero elements in this `af.Array`.
    pub fn countAll(self: *const Self) !af.ops.ComplexParts {
        return af.ops.countAll(self);
    }

    /// Find the minimum values and their locations.
    ///
    /// This function performs the operation across all
    /// batches present in the input simultaneously.
    ///
    /// Returns struct containing the following fields:
    /// - `out`: `af.Array` containing the minimum of all values
    /// in this `af.Array` along the specified dimension.
    /// - `idx`: `af.Array` containg the location of the minimum of
    /// all values in this `af.Array` along the specified dimension.
    pub fn imin(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.imin(allocator, self, dim);
    }

    /// Find the maximum value and its location.
    ///
    /// This function performs the operation across all
    /// batches present in the input simultaneously.
    ///
    /// Returns struct containing the following fields:
    /// - `out`: `af.Array` containing the maximum of all values
    /// in this `af.Array` along the specified dimension.
    /// - `idx`: `af.Array` containg the location of the maximum of
    /// all values in this `af.Array` along the specified dimension.
    pub fn imax(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.imax(allocator, self, dim);
    }

    /// Returns minimum value and its location from the entirety of
    /// this `af.Array`.
    pub fn iminAll(self: *const Self) !struct { real: f64, imag: f64, idx: u32 } {
        return af.ops.iminAll(self);
    }

    /// Returns maximum value and its location from the entirety of
    /// this `af.Array`.
    pub fn imaxAll(self: *const Self) !struct { real: f64, imag: f64, idx: u32 } {
        return af.ops.imaxAll(self);
    }

    /// Returns the computed cumulative sum (inclusive) of this `af.Array`.
    pub fn accum(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.accum(allocator, self, dim);
    }

    /// Generalized scan of this `af.Array`.
    ///
    /// Returns ptr to an `af.Array` containing scan of the input.
    pub fn scan(
        self: *const Self,
        allocator: std.mem.Allocator,
        dim: i32,
        op: af.BinaryOp,
        inclusive_scan: bool,
    ) !*Self {
        return af.ops.scan(
            allocator,
            self,
            dim,
            op,
            inclusive_scan,
        );
    }

    //// Generalized scan by key of this `af.Array`.
    ///
    /// Returns ptr to an `af.Array` containing scan of the input
    /// by key.
    pub fn scanByKey(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: i32,
        op: af.BinaryOp,
        inclusive_scan: bool,
    ) !*Self {
        return af.ops.scanByKey(
            allocator,
            keys,
            self,
            dim,
            op,
            inclusive_scan,
        );
    }

    /// Locate the indices of non-zero elements in this
    /// `af.Array`.
    ///
    /// Return type is u32 for all input types.
    ///
    /// The locations are provided by flattening this
    /// `af.Array` into a linear `af.Array`.
    pub fn where(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.where(allocator, self);
    }

    /// First order numerical difference along specified dimension
    /// of this `af.Array`.
    ///
    /// This function performs the operation across all batches
    /// present in this `af.Array` simultaneously.
    pub fn diff1(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.diff1(allocator, self, dim);
    }

    /// Second order numerical difference along specified dimension
    /// of this `af.Array`.
    ///
    /// This function performs the operation across all batches
    /// present in this `af.Array` simultaneously.
    pub fn diff2(self: *const Self, allocator: std.mem.Allocator, dim: i32) !*Self {
        return af.ops.diff2(allocator, self, dim);
    }

    /// Sort this `af.Array`.
    ///
    /// Returns ptr to `af.Array` containing sorted output.
    pub fn sort(
        self: *const Self,
        allocator: std.mem.Allocator,
        dim: u32,
        is_ascending: bool,
    ) !*Self {
        return af.ops.sort(allocator, self, dim, is_ascending);
    }

    /// Sort this `af.Array` and return sorted indices.
    ///
    /// Index `af.Array` is of `af.Dtype.u32`.
    ///
    /// Returns struct containing the following fields:
    /// - `out`: `af.Array` containing sorted output.
    /// - `indices`: `af.Array` containing the indices
    /// in the original input.
    pub fn sortIndex(
        self: *const Self,
        allocator: std.mem.Allocator,
        dim: u32,
        is_ascending: bool,
    ) !struct { out: *Self, idx: *Self } {
        return af.ops.sortIndex(allocator, self, dim, is_ascending);
    }

    /// Sort this `af.Array` based on keys.
    ///
    /// Returns struct containing the following fields:
    /// - `out_keys`: `af.Array` containing the keys based
    /// on sorted values.
    /// - `out_values`: `af.Array` containing the sorted values.
    pub fn sortByKey(
        self: *const Self,
        allocator: std.mem.Allocator,
        keys: *const Self,
        dim: u32,
        is_ascending: bool,
    ) !struct { out_keys: *Self, out_values: *Self } {
        return af.ops.sortByKey(
            allocator,
            keys,
            self,
            dim,
            is_ascending,
        );
    }

    /// Finds unique values from this `af.Array`.
    ///
    /// The input must be a one-dimensional `af.Array`. Batching
    /// is not currently supported.
    pub fn setUnique(self: *const Self, allocator: std.mem.Allocator, is_sorted: bool) !*Self {
        return af.ops.setUnique(allocator, self, is_sorted);
    }

    /// Find the union of this `af.Array` and another
    /// `af.Array`.
    ///
    /// The inputs must be one-dimensional `af.Array`s.
    ///
    /// Batching is not currently supported.
    pub fn setUnion(
        self: *const Self,
        allocator: std.mem.Allocator,
        second: *const Self,
        is_unique: bool,
    ) !*Self {
        return af.ops.setUnion(
            allocator,
            self,
            second,
            is_unique,
        );
    }
    /// Find the intersection of this `af.Array` and another
    /// `af.Array`.
    ///
    /// The inputs must be one-dimensional `af.Array`s.
    /// Batching is not currently supported.
    pub fn setIntersect(
        self: *const Self,
        allocator: std.mem.Allocator,
        second: *const Self,
        is_unique: bool,
    ) !*Self {
        return af.ops.setIntersect(
            allocator,
            self,
            second,
            is_unique,
        );
    }

    /// Performs element wise addition on this `af.Array` and
    /// another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn add(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.add(allocator, self, rhs, batch);
    }

    /// Performs in-place element wise addition on this `af.Array` and
    /// another `af.Array`.
    pub fn addInplace(self: *Self, rhs: *const Self, batch: bool) !void {
        try af.ops.addInplace(self, rhs, batch);
    }

    /// Performs element wise subtraction on this `af.Array` and
    /// another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn sub(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.sub(allocator, self, rhs, batch);
    }

    /// Performs in-place element wise subtraction on this `af.Array` and
    /// another `af.Array`.
    pub fn subInplace(self: *Self, rhs: *const Self, batch: bool) !void {
        try af.ops.subInplace(self, rhs, batch);
    }

    /// Performs element wise multiplication on this `af.Array` and
    /// another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn mul(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.mul(allocator, self, rhs, batch);
    }

    /// Performs in-place element wise multiplication on this `af.Array` and
    /// another `af.Array`.
    pub fn mulInplace(self: *Self, rhs: *const Self, batch: bool) !void {
        try af.ops.mulInplace(self, rhs, batch);
    }

    /// Performs element wise division on this `af.Array` and
    /// another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn div(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.div(allocator, self, rhs, batch);
    }

    /// Performs in-place element wise division on this `af.Array` and
    /// another `af.Array`.
    pub fn divInplace(self: *Self, rhs: *const Self, batch: bool) !void {
        try af.ops.divInplace(self, rhs, batch);
    }

    /// Perform a less-than comparison between
    /// corresponding elements of this `af.Array` and
    /// another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn lt(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.lt(allocator, self, rhs, batch);
    }

    /// Perform a greater-than comparison between
    /// corresponding elements of this `af.Array` and
    /// another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn gt(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.gt(allocator, self, rhs, batch);
    }

    /// Perform a less-than-or-equal comparison between
    /// corresponding elements of this `af.Array` and
    /// another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn le(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.le(allocator, self, rhs, batch);
    }

    /// Perform a greater-than-or-equal comparison between
    /// corresponding elements of this `af.Array` and
    /// another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn ge(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.ge(allocator, self, rhs, batch);
    }

    /// Checks if corresponding elements of this `af.Array`
    /// and another `af.Array` are equal.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn eq(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.eq(allocator, self, rhs, batch);
    }

    /// Checks if corresponding elements of this `af.Array`
    /// and another `af.Array` are not equal.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn neq(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.neq(allocator, self, rhs, batch);
    }

    /// Evaluate the logical AND of this `af.Array`
    /// and another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn and_(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.and_(allocator, self, rhs, batch);
    }

    /// Evaluate the logical OR of this `af.Array`
    /// and another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn or_(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.or_(allocator, self, rhs, batch);
    }

    /// Evaluate the logical NOT of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn not(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.not(allocator, self);
    }

    /// Evaluate the bitwise NOT of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn bitNot(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.bitNot(allocator, self);
    }

    /// Evaluate the bitwise AND of this `af.Array`
    /// and another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn bitAnd(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.bitAnd(allocator, self, rhs, batch);
    }

    /// Evaluate the bitwise OR of this `af.Array`
    /// and another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn bitOr(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.bitOr(allocator, self, rhs, batch);
    }

    /// Evaluate the bitwise XOR of this `af.Array`
    /// and another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn bitXor(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.bitXor(allocator, self, rhs, batch);
    }

    /// Shift the bits of this integer `af.Array` left.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn bitShiftL(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.bitShiftL(allocator, self, rhs, batch);
    }

    /// Shift the bits of this integer `af.Array` right.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn bitShiftR(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.bitShiftR(allocator, self, rhs, batch);
    }

    /// Casts this `af.Array` from one type to another.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn cast(self: *const Self, allocator: std.mem.Allocator, dtype: af.Dtype) !*Self {
        return af.ops.cast(allocator, self, dtype);
    }

    /// Returns the elementwise minimum between this `af.Array`
    /// and another `af.Array`.
    pub fn minOf(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.minOf(allocator, self, rhs, batch);
    }

    /// Returns the elementwise maximum between this `af.Array`
    /// and another `af.Array`.
    pub fn maxOf(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.maxOf(allocator, self, rhs, batch);
    }

    /// Clamp this `af.Array` between an upper and a lower limit.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn clamp(
        self: *const Self,
        allocator: std.mem.Allocator,
        lo: *const Self,
        hi: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.clamp(allocator, self, lo, hi, batch);
    }

    /// Calculate the remainder of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn rem(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.rem(allocator, self, rhs, batch);
    }

    /// Calculate the modulus of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn mod(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.mod(allocator, self, rhs, batch);
    }

    /// Calculate the absolute value of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn abs(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.abs(allocator, self);
    }

    /// Calculate the phase angle (in radians) of this complex `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn arg(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.arg(allocator, self);
    }

    /// Calculate the sign of elements in this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn sign(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sign(allocator, self);
    }

    /// Round numbers in this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn round(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.round(allocator, self);
    }

    /// Truncate numbers in this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn trunc(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.trunc(allocator, self);
    }

    /// Floor numbers in this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn floor(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.floor(allocator, self);
    }

    /// Ceil numbers in this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn ceil(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.ceil(allocator, self);
    }

    /// Calculate the length of the hypotenuse of this `af.Array`
    /// and another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn hypot(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.hypot(allocator, self, rhs, batch);
    }

    /// Evaluate the sine function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn sin(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sin(allocator, self);
    }

    /// Evaluate the cosine function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn cos(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.cos(allocator, self);
    }

    /// Evaluate the tangent function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn tan(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.tan(allocator, self);
    }

    /// Evaluate the inverse sine function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn asin(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.asin(allocator, self);
    }

    /// Evaluate the inverse cosine function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn acos(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.acos(allocator, self);
    }

    /// Evaluate the inverse tangent function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn atan(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.atan(allocator, self);
    }

    /// Evaluate the inverse tangent function of this `af.Array`
    /// and another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn atan2(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.atan2(allocator, self, rhs, batch);
    }

    /// Evaluate the hyperbolic sine function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn sinh(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sinh(allocator, self);
    }

    /// Evaluate the hyperbolic cosine function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn cosh(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.cosh(allocator, self);
    }

    /// Evaluate the hyperbolic tangent function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn tanh(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.tanh(allocator, self);
    }

    /// Evaluate the inverse hyperbolic sine function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn asinh(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.asinh(allocator, self);
    }

    /// Evaluate the inverse hyperbolic cosine function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn acosh(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.acosh(allocator, self);
    }

    /// Evaluate the inverse hyperbolic tangent function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn atanh(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.atanh(allocator, self);
    }

    /// Initializes and returns a complex `af.Array` from this
    /// single real `af.Array`.
    pub fn cplx(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.cplx(allocator, self);
    }

    /// Initializes and returns a complex `af.Array` from
    /// two real `af.Array`s.
    pub fn cplx2(
        self: *const Self,
        allocator: std.mem.Allocator,
        imag_: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.cplx2(allocator, self, imag_, batch);
    }

    /// Returns the real part of this complex `af.Array`.
    pub fn real(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.real(allocator, self);
    }

    /// Returns the imaginary part of this complex `af.Array`.
    pub fn imag(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.imag(allocator, self);
    }

    /// Evaluate the complex conjugate of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn conjg(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.conjg(allocator, self);
    }

    /// Evaluate the nth root of this `af.Array` and another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn root(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        batch: bool,
    ) !*Self {
        return af.ops.root(allocator, self, rhs, batch);
    }

    /// Raise this `af.Array` base to a power (or exponent).
    ///
    /// Returns ptr the resulting `af.Array`.
    pub fn pow(self: *const Self, allocator: std.mem.Allocator, rhs: *const Self, batch: bool) !*Self {
        return af.ops.pow(allocator, self, rhs, batch);
    }

    /// Raise 2 to a power (or exponent).
    ///
    /// Returns ptr the resulting `af.Array`.
    pub fn pow2(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.pow2(allocator, self);
    }

    /// Evaluate the logistical sigmoid function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn sigmoid(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sigmoid(allocator, self);
    }

    /// Evaluate the exponential of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn exp(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.exp(allocator, self);
    }

    /// Evaluate the exponential of this `af.Array` minus 1, exp(in) - 1.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn expm1(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.expm1(allocator, self);
    }

    /// Evaluate the error function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn erf(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.erf(allocator, self);
    }

    /// Evaluate the complementary error function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn erfc(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.erfc(allocator, self);
    }

    /// Evaluate the natural logarithm of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn log(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.log(allocator, self);
    }

    /// Evaluate the natural logarithm of this `af.Array` plus 1, ln(1+in).
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn log1p(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.log1p(allocator, self);
    }

    /// Evaluate the base 10 logarithm of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn log10(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.log10(allocator, self);
    }

    /// Evaluate the base 2 logarithm of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn log2(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.log2(allocator, self);
    }

    /// Evaluate the square root of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn sqrt(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sqrt(allocator, self);
    }

    /// Evaluate the reciprocal square root of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn rsqrt(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.rsqrt(allocator, self);
    }

    /// Evaluate the cube root of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn cbrt(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.cbrt(allocator, self);
    }

    /// Calculate the factorial of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn factorial(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.factorial(allocator, self);
    }

    /// Evaluate the gamma function of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn tgamma(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.tgamma(allocator, self);
    }

    /// Evaluate the logarithm of the absolute value of the gamma function
    /// of this `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn lgamma(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.lgamma(allocator, self);
    }

    /// Check if values of this `af.Array` are zero.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn isZero(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.isZero(allocator, self);
    }

    /// Check if values of this `af.Array` are infinite.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn isInf(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.isInf(allocator, self);
    }

    /// Check if values of this `af.Array` are NaN.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn isNaN(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.isNaN(allocator, self);
    }

    /// Lock the device buffer in the memory manager.
    pub fn lockDevicePtr(self: *const Self) !void {
        return af.ops.lockDevicePtr(self);
    }

    /// Unlock the device buffer in the memory manager.
    pub fn unlockDevicePtr(self: *const Self) !void {
        return af.ops.unlockDevicePtr(self);
    }

    /// Lock the device buffer in the memory manager.
    pub fn lock(self: *const Self) !void {
        return af.ops.lockArray(self);
    }

    /// Unlock the device buffer in the memory manager.
    pub fn unlock(self: *const Self) !void {
        return af.ops.unlockArray(self);
    }

    /// Query if `af.Array` has been locked by the user.
    pub fn isLocked(self: *const Self) !bool {
        return af.ops.isLockedArray(self);
    }

    /// Get the device pointer and lock the buffer in memory manager.
    pub fn getDevicePtr(self: *const Self) !?*anyopaque {
        return af.ops.getDevicePtr(self);
    }

    /// Lookup the values of this `af.Array` based on sequences.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn index(
        self: *const Self,
        allocator: std.mem.Allocator,
        ndims: u32,
        idx: *const af.af_seq,
    ) !*Self {
        return af.ops.index(allocator, self, ndims, idx);
    }

    /// Lookup the values of this `af.Array` by indexing
    /// with another `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn lookup(
        self: *const Self,
        allocator: std.mem.Allocator,
        indices: *const Self,
        dim: u32,
    ) !*Self {
        return af.ops.lookup(allocator, self, indices, dim);
    }

    /// Copy and write values in the locations specified
    /// by the sequences.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn assignSeq(
        self: *const Self,
        allocator: std.mem.Allocator,
        ndims: u32,
        indices: *const af.af_seq,
        rhs: *const Self,
    ) !*Self {
        return af.ops.assignSeq(
            allocator,
            self,
            ndims,
            indices,
            rhs,
        );
    }

    /// Indexing this `af.Array` using `af.af_seq`, or `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn indexGen(
        self: *const Self,
        allocator: std.mem.Allocator,
        ndims: i64,
        indices: *const af.af_index_t,
    ) !*Self {
        return af.ops.indexGen(allocator, self, ndims, indices);
    }

    /// Assignment of this `af.Array` using `af.af_seq`, or `af.Array`.
    ///
    /// Generalized assignment function that accepts either `af.Array`
    /// or `af.af_seq` along a dimension to assign elements from this
    /// `af.Array` to an output `af.Array`.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn assignGen(
        self: *Self,
        allocator: std.mem.Allocator,
        ndims: i64,
        indices: []af.af_index_t,
        rhs: *const af.Array,
    ) !*Self {
        return af.ops.assignGen(
            allocator,
            self,
            ndims,
            indices,
            rhs,
        );
    }

    /// Print this `af.Array` and dimensions to screen.
    pub fn print(self: *const Self) !void {
        return af.ops.printArray(self);
    }

    /// Print the expression, `af.Array`, and dimensions to screen.
    pub fn printArrayGen(self: *const Self, expr: []const u8, precision: i32) !void {
        return af.ops.printArrayGen(expr, self, precision);
    }

    /// Save this `af.Array` to a binary file.
    ///
    /// The `save` and `read` functions are designed
    /// to provide store and read access to `af.Array`s using files
    /// written to disk.
    ///
    /// Returns the index location of the `af.Array` in the file.
    pub fn save(self: *const Self, key: []const u8, filename: []const u8, append: bool) !i32 {
        return af.ops.saveArray(self, key, filename, append);
    }

    /// Checks for the key's existence in the file. Returns
    /// the index of the array in the file if the key is
    /// found; else returns -1 if key is not found.
    ///
    /// When calling `readArrayKey`, it may be a good idea
    /// to run this function first to check for the key
    /// and then call the `readArrayIndex` using the index.
    ///
    /// This will avoid exceptions in case of key not found.
    pub fn readKeyCheck(filename: []const u8, key: []const u8) !i32 {
        return af.ops.readArrayKeyCheck(filename, key);
    }

    /// Print this `af.Array` to a string instead of the screen.
    ///
    /// Returns slice containing the resulting string.
    ///
    /// N.B. The memory for output is allocated by the function.
    /// The user is responsible for deleting the memory using
    /// `af.freeHost`.
    pub fn toString(
        self: *const Self,
        expr: []const u8,
        precision: i32,
        trans: bool,
    ) ![]const u8 {
        return af.ops.arrayToString(expr, self, precision, trans);
    }

    /// Deep copy this `af.Array` to another.
    pub fn copy(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.copyArray(allocator, self);
    }

    /// Copy data from a C pointer (host/device) to the underlying `af.af_array`.
    pub fn write(
        self: *Self,
        data: ?*const anyopaque,
        bytes: usize,
        src: af.Source,
    ) !void {
        return af.ops.writeArray(self, data, bytes, src);
    }

    /// Copy data from the underlying `af.af_array` to a pointer.
    pub fn getDataPtr(self: *const Self, data: ?*anyopaque) !void {
        return af.ops.getDataPtr(data, self);
    }

    /// Reduce the reference count of this `af.Array`.
    pub fn release(self: *Self) !void {
        try af.ops.releaseArray(self);
        self.array_ = null;
    }

    /// Increments this `af.Array` reference count.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn retain(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.retainArray(allocator, self);
    }

    /// Returns the reference count of this `af.Array`.
    pub fn getDataRefCount(self: *const Self) !i32 {
        return af.ops.getDataRefCount(self);
    }

    /// Evaluate any expressions in this `af.Array`.
    pub fn eval(self: *Self) !void {
        return af.ops.eval(self);
    }

    /// Returns the total number of elements across all dimensions
    /// of this `af.Array`.
    pub fn getElements(self: *const Self) !i64 {
        return af.ops.getElements(self);
    }

    /// Returns the `af.Dtype` of this `af.Array`.
    pub fn getType(self: *const Self) !af.Dtype {
        return af.ops.getType(self);
    }

    /// Returns the dimensions of this `af.Array`.
    pub fn getDims(self: *const Self) !af.Dim4 {
        return af.ops.getDims(self);
    }

    /// Returns the number of dimensions of this `af.Array`.
    pub fn getNumDims(self: *const Self) !u32 {
        return af.ops.getNumDims(self);
    }

    /// Check if this `af.Array` is empty.
    pub fn isEmpty(self: *const Self) !bool {
        return af.ops.isEmpty(self);
    }

    /// Check if this `af.Array` is a scalar.
    pub fn isScalar(self: *const Self) !bool {
        return af.ops.isScalar(self);
    }

    /// Check if this `af.Array` is a row vector.
    pub fn isRow(self: *const Self) !bool {
        return af.ops.isRow(self);
    }

    /// Check if this `af.Array` is a column vector.
    pub fn isColumn(self: *const Self) !bool {
        return af.ops.isColumn(self);
    }

    /// Check if this `af.Array` is a vector.
    pub fn isVector(self: *const Self) !bool {
        return af.ops.isVector(self);
    }

    /// Check if this `af.Array` is a complex type.
    pub fn isComplex(self: *const Self) !bool {
        return af.ops.isComplex(self);
    }

    /// Check if this `af.Array` is a real type.
    pub fn isReal(self: *const Self) !bool {
        return af.ops.isReal(self);
    }

    /// Check if this `af.Array` is double precision type.
    pub fn isDouble(self: *const Self) !bool {
        return af.ops.isDouble(self);
    }

    /// Check if this `af.Array` is single precision type.
    pub fn isSingle(self: *const Self) !bool {
        return af.ops.isSingle(self);
    }

    /// Check if this `af.Aarray` is a 16 bit floating point type.
    pub fn isHalf(self: *const Self) !bool {
        return af.ops.isHalf(self);
    }

    /// Check if this `af.Array` is a real floating point type.
    pub fn isRealFloating(self: *const Self) !bool {
        return af.ops.isRealFloating(self);
    }

    /// Check if this `af.Array` is floating precision type.
    pub fn isFloating(self: *const Self) !bool {
        return af.ops.isFloating(self);
    }

    /// Check if this `af.Array` is integer type.
    pub fn isInteger(self: *const Self) !bool {
        return af.ops.isInteger(self);
    }

    /// Check if this `af.Array` is bool type.
    pub fn isBool(self: *const Self) !bool {
        return af.ops.isBool(self);
    }

    /// Check if this `af.Array` is sparse.
    pub fn isSparse(self: *const Self) !bool {
        return af.ops.isSparse(self);
    }

    /// Returns the first element from this `af.Array`.
    pub fn getScalar(self: *const Self, comptime T: type) !T {
        return af.ops.getScalar(T, self);
    }

    /// Get's the backend enum for this `af.Array`.
    ///
    /// This will return one of the values from the `af.Backend` enum.
    /// The return value specifies which backend the `af.Array` was created on.
    pub fn getBackendId(self: *const Self) !af.Backend {
        return af.ops.getBackendId(self);
    }

    /// Returns the id of the device this `af.Array` was created on.
    pub fn getDeviceId(self: *const Self) !i32 {
        return af.ops.getDeviceId(self);
    }

    /// Performs a matrix multiplication on this `af.Array` and
    /// another `af.Array`.
    ///
    /// N.B. The following applies for Sparse-Dense matrix multiplication.
    /// This function can be used with one sparse input. The sparse input
    /// must always be the lhs and the dense matrix must be rhs. The sparse
    /// array can only be of `af.Storage.CSR` format. The returned array is
    /// always dense. optLhs can only be one of `af.MatProp.None`, `af.MatProp.Trans`,
    /// `af.MatProp.CTrans`. optRhs can only be `af.MatProp.None`.
    pub fn matmul(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        optLhs: af.MatProp,
        optRhs: af.MatProp,
    ) !*Self {
        return af.ops.matmul(
            allocator,
            self,
            rhs,
            optLhs,
            optRhs,
        );
    }

    /// Scalar dot product between two vectors.
    ///
    /// Also referred to as the inner product.
    pub fn dot(
        self: *const Self,
        allocator: std.mem.Allocator,
        rhs: *const Self,
        optLhs: af.MatProp,
        optRhs: af.MatProp,
    ) !*Self {
        return af.ops.dot(
            allocator,
            self,
            rhs,
            optLhs,
            optRhs,
        );
    }

    /// Scalar dot product between two vectors (this `af.Array`
    /// and another `af.Array`).
    ///
    /// Also referred to as the inner product. Returns the result
    /// as a host scalar.
    pub fn dotAll(
        self: *const Self,
        rhs: *const Self,
        optLhs: af.MatProp,
        optRhs: af.MatProp,
    ) !af.ops.ComplexParts {
        return af.ops.dotAll(self, rhs, optLhs, optRhs);
    }

    /// Transpose a matrix.
    pub fn transpose(self: *Self, allocator: std.mem.Allocator, conjugate: bool) !*Self {
        return af.ops.transpose(allocator, self, conjugate);
    }

    /// Transpose this `af.Array` matrix in-place.
    pub fn transposeInplace(self: *Self, conjugate: bool) !void {
        return af.ops.transposeInplace(self, conjugate);
    }

    /// Create a diagonal matrix from this `af.Array` matrix.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn diagCreate(self: *const Self, allocator: std.mem.Allocator, num: i32) !*Self {
        return af.ops.diagCreate(allocator, self, num);
    }

    /// Extract diagonal from this `af.Array` matrix.
    ///
    /// Returns ptr to the resulting `af.Array`.
    pub fn diagExtract(self: *const Self, allocator: std.mem.Allocator, num: i32) !*Self {
        return af.ops.diagExtract(allocator, self, num);
    }

    /// Joins this `af.Array` and another `af.Array` along the
    /// specified dimension.
    pub fn join(
        self: *const Self,
        allocator: std.mem.Allocator,
        dim: i32,
        second: *const Self,
    ) !*Self {
        return af.ops.join(allocator, self, dim, second);
    }

    /// Repeat the contents of this `af.Array` along the
    /// specified dimensions.
    pub fn tile(
        self: *const Self,
        allocator: std.mem.Allocator,
        x: u32,
        y: u32,
        z: u32,
        w: u32,
    ) !*Self {
        return af.ops.tile(allocator, self, x, y, z, w);
    }

    /// Reorder this `af.Array` according to the specified dimensions.
    pub fn reorder(
        self: *const Self,
        allocator: std.mem.Allocator,
        x: u32,
        y: u32,
        z: u32,
        w: u32,
    ) !*Self {
        return af.ops.reorder(allocator, self, x, y, z, w);
    }

    /// Circular shift this `af.Array` according to the specified
    /// dimensions.
    pub fn shift(
        self: *const Self,
        allocator: std.mem.Allocator,
        x: i32,
        y: i32,
        z: i32,
        w: i32,
    ) !*Self {
        return af.ops.shift(allocator, self, x, y, z, w);
    }

    /// Modifies the dimensions of this `af.Array` to the shape specified
    /// by an array of ndims dimensions.
    pub fn moddims(
        self: *const Self,
        allocator: std.mem.Allocator,
        ndims: u32,
        dims: af.Dim4,
    ) !*Self {
        return af.ops.moddims(allocator, self, ndims, dims);
    }

    /// Flatten this `af.Array` to a single dimension.
    pub fn flat(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.flat(allocator, self);
    }

    /// Flip this `af.Array` along the specified dimension.
    ///
    /// Mirrors the `af.Array` along the specified dimensions.
    pub fn flip(self: *const Self, allocator: std.mem.Allocator, dim: u32) !*Self {
        return af.ops.flip(allocator, self, dim);
    }

    /// Create a lower triangular matrix from this `af.Array`.
    pub fn lower(self: *const Self, allocator: std.mem.Allocator, is_unit_diag: bool) !*Self {
        return af.ops.lower(allocator, self, is_unit_diag);
    }

    /// Create an upper triangular matrix from this `af.Array`.
    pub fn upper(self: *const Self, allocator: std.mem.Allocator, is_unit_diag: bool) !*Self {
        return af.ops.upper(allocator, self, is_unit_diag);
    }

    /// Selects elements from this `af.Array` and another based on the values of a
    /// binary conditional `af.Array`.
    ///
    /// Creates a new `af.Array` that is composed of values either from
    /// this `af.Array` or `af.Array` b, based on a third conditional
    /// `af.Array`. For all non-zero elements in the conditional `af.Array`,
    /// the output `af.Array` will contain values from this. Otherwise the output
    /// will contain values from b.
    pub fn select(
        self: *const Self,
        allocator: std.mem.Allocator,
        cond: *const Self,
        b: *const Self,
    ) !*Self {
        return af.ops.select(allocator, cond, self, b);
    }

    /// Returns ptr to an `af.Array` containing elements of this
    /// `af.Array` when cond is true else scalar value b.
    pub fn selectScalarR(
        self: *const Self,
        allocator: std.mem.Allocator,
        cond: *const Self,
        b: f64,
    ) !*Self {
        return af.ops.selectScalarR(allocator, cond, self, b);
    }

    /// Returns ptr to an `af.Array` containing elements of this
    /// `af.Array` when cond is false else scalar value a.
    pub fn selectScalarL(
        self: *const Self,
        allocator: std.mem.Allocator,
        cond: *const Self,
        a: f64,
    ) !*Self {
        return af.ops.selectScalarL(allocator, cond, a, self);
    }

    /// Replace elements of this `af.Array` based on a conditional `af.Array`.
    ///
    /// - Input values are retained when corresponding elements from condition array are true.
    /// - Input values are replaced when corresponding elements from condition array are false.
    pub fn replace(self: *Self, cond: *const Self, b: *const Self) !void {
        return af.ops.replace(self, cond, b);
    }

    /// Replace elements of this `af.Array` based on a conditional `af.Array`.
    ///
    /// N.B. Values of this `af.Array` are replaced with corresponding values of b,
    /// when cond is false.
    pub fn replaceScalar(self: *Self, cond: *const Self, b: f64) !void {
        return af.ops.replaceScalar(self, cond, b);
    }

    /// Pad this `af.Array`.
    ///
    /// Pad this `af.Array` using a constant or values from input along border
    pub fn pad(
        self: *const Self,
        allocator: std.mem.Allocator,
        begin_ndims: u32,
        begin_dims: af.Dim4,
        end_ndims: u32,
        end_dims: af.Dim4,
        pad_fill_type: af.BorderType,
    ) !*Self {
        return af.ops.pad(
            allocator,
            self,
            begin_ndims,
            begin_dims,
            end_ndims,
            end_dims,
            pad_fill_type,
        );
    }

    /// Calculate the gradients of this `af.Array`.
    ///
    /// The gradients along the first and second dimensions
    /// are calculated simultaneously.
    ///
    /// Return struct containing the following fields:
    /// - `dx`: `af.Array` containing the gradient along the
    /// 1st dimension of this `af.Array`.
    /// - `dy`: `af.Array` containing the gradient along the
    /// 2nd dimension of this `af.Array`.
    pub fn gradient(self: *const Self, allocator: std.mem.Allocator) !struct {
        dx: *Self,
        dy: *Self,
    } {
        return af.ops.gradient(allocator, self);
    }

    /// Computes the singular value decomposition of this `af.Array` matrix.
    ///
    /// This function factorizes a matrix A into two unitary matrices,
    /// U and VT, and a diagonal matrix S, such that A=USVT. If A has
    /// M rows and N columns ( M×N), then U will be M×M, V will be N×N,
    /// and S will be M×N. However, for S, this function only returns the
    /// non-zero diagonal elements as a sorted (in descending order) 1D `af.Array`.
    pub fn svd(self: *const Self, allocator: std.mem.Allocator) !struct {
        u: *Self,
        s: *Self,
        vt: *Self,
    } {
        return af.ops.svd(allocator, self);
    }

    /// Computes the singular value decomposition of this `af.Array` matrix in-place.
    pub fn svdInplace(self: *Self, allocator: std.mem.Allocator) !struct {
        u: *Self,
        s: *Self,
        vt: *Self,
    } {
        return af.ops.svdInplace(allocator, self);
    }

    /// Perform LU decomposition on this `af.Array`.
    ///
    /// This function decomposes input matrix A into a
    /// lower triangle L and upper triangle U.
    pub fn lu(self: *const Self, allocator: std.mem.Allocator) !struct {
        lower: *Self,
        upper: *Self,
    } {
        return af.ops.lu(allocator, self);
    }

    /// In-place LU decomposition.
    pub fn luInplace(self: *Self, allocator: std.mem.Allocator, is_lapack_piv: bool) !*Self {
        return af.ops.luInplace(allocator, self, is_lapack_piv);
    }

    /// Perform QR decomposition on this `af.Array`.
    ///
    /// This function decomposes input matrix A into an orthogonal
    /// matrix Q and an upper triangular matrix R.
    pub fn qr(self: *const Self, allocator: std.mem.Allocator) !struct {
        q: *af.Array,
        r: *af.Array,
        tau: *af.Array,
    } {
        return af.ops.qr(allocator, self);
    }

    /// In-place QR decomposition on this `af.Array`.
    pub fn qrInplace(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.qrInplace(allocator, self);
    }

    /// Perform Cholesky decomposition on this `af.Array`.
    ///
    /// This function decomposes a positive definite matrix A into
    /// two triangular matrices.
    ///
    /// Returns a struct containing the following fields:
    /// -`out`: `af.Array` containing the triangular matrix.
    /// Multiply out with it conjugate transpose reproduces
    /// the input `af.Array`.
    /// -`info`: is 0 if cholesky decomposition passes, if not
    /// it returns the rank at which the decomposition failed.
    pub fn cholesky(self: *const Self, allocator: std.mem.Allocator, is_upper: bool) !struct {
        out: *Self,
        info: i32,
    } {
        return af.ops.cholesky(allocator, self, is_upper);
    }

    /// In-place Cholesky decomposition on this `af.Array`.
    ///
    /// Returns 0 if cholesky decomposition passes, if not
    /// it returns the rank at which the decomposition failed.
    pub fn choleskyInplace(self: *Self, is_upper: bool) !i32 {
        return af.ops.choleskyInplace(self, is_upper);
    }

    /// Solve a system of equations.
    ///
    /// This function takes a co-efficient matrix A and an output
    /// matrix B as inputs to solve the following equation for X.
    pub fn solve(
        self: *const Self,
        allocator: std.mem.Allocator,
        b: *const Self,
        options: af.MatProp,
    ) !*Self {
        return af.ops.solve(allocator, self, b, options);
    }

    /// Solve a system of equations.
    ///
    /// This function takes a co-efficient matrix A and an output
    /// matrix B as inputs to solve the following equation for X.
    pub fn solveLU(
        self: *const Self,
        allocator: std.mem.Allocator,
        piv: *const Self,
        b: *const Self,
        options: af.MatProp,
    ) !*Self {
        return af.ops.solveLU(allocator, self, piv, b, options);
    }

    /// Invert this `af.Array` matrix.
    ///
    /// This function inverts a square `af.Array` matrix.
    pub fn inverse(self: *const Self, allocator: std.mem.Allocator, options: af.MatProp) !*Self {
        return af.ops.inverse(allocator, self, options);
    }

    /// Pseudo-invert this `af.Array` matrix.
    ///
    /// This function calculates the Moore-Penrose pseudoinverse
    /// of a matrix A, using `svd` at its core. If A is of size M×N,
    /// then its pseudoinverse A+ will be of size N×M.
    ///
    /// This calculation can be batched if the input array is three or
    /// four-dimensional (M×N×P×Q, with Q=1 for only three dimensions).
    /// Each M×N slice along the third dimension will have its own pseudoinverse,
    /// for a total of P×Q pseudoinverses in the output array (N×M×P×Q).
    pub fn pInverse(self: *const Self, allocator: std.mem.Allocator, tol: f64, options: af.MatProp) !*Self {
        return af.ops.pInverse(allocator, self, tol, options);
    }

    /// Returns the rank of this `af.Array` matrix.
    ///
    /// This function uses `af.ops.qr` to find the rank of this
    /// `af.Array` matrix within the given tolerance.
    pub fn rank(self: *const Self, tol: f64) !u32 {
        return af.ops.rank(self, tol);
    }

    /// Returns the determinant of this `af.Array` matrix.
    pub fn det(self: *const Self) !struct { det_real: f64, det_imag: f64 } {
        return af.ops.det(self);
    }

    /// Returns the norm of this `af.Array` matrix.
    ///
    /// This function can return the norm using various
    /// metrics based on the type parameter.
    pub fn norm(self: *const Self, norm_type: af.NormType, p: f64, q: f64) !f64 {
        return af.ops.norm(self, norm_type, p, q);
    }

    /// Signals interpolation on one dimensional signals.
    /// Returns ptr to a new `af.Array`.
    pub fn approx1(
        self: *const Self,
        allocator: std.mem.Allocator,
        pos: *const Self,
        method: af.InterpType,
        off_grid: f32,
    ) !*Self {
        return af.ops.approx1(allocator, self, pos, method, off_grid);
    }

    /// Signals interpolation on one dimensional signals; accepts
    /// a pre-allocated array, `out`, where results are written.
    pub fn approx1V2(
        self: *const Self,
        out: *Self,
        pos: *const Self,
        method: af.InterpType,
        off_grid: f32,
    ) !void {
        return af.ops.approx1V2(out, self, pos, method, off_grid);
    }

    /// Signals interpolation on two dimensional signals.
    /// Returns ptr to a new `af.Array`.
    pub fn approx2(
        self: *const af.Array,
        pos0: *const Self,
        pos1: *const Self,
        method: af.InterpType,
        off_grid: f32,
    ) !*Self {
        return af.ops.approx2(self, pos0, pos1, method, off_grid);
    }

    /// Signals interpolation on two dimensional signals; accepts
    /// a pre-allocated array, `out`, where results are written.
    pub fn approx2V2(
        self: *const Self,
        out: *Self,
        pos0: *const Self,
        pos1: *const Self,
        method: af.InterpType,
        off_grid: f32,
    ) !void {
        return af.ops.approx2V2(out, self, pos0, pos1, method, off_grid);
    }

    /// Signals interpolation on one dimensional signals along specified dimension.
    ///
    /// `approx1Uniform` accepts the dimension to perform the interpolation along
    /// the input. It also accepts start and step values which define the uniform
    /// range of corresponding indices.
    pub fn approx1Uniform(
        self: *const Self,
        allocator: std.mem.Allocator,
        pos: *const Self,
        interp_dim: i32,
        idx_start: f64,
        idx_step: f64,
        method: af.InterpType,
        off_grid: f32,
    ) !*Self {
        return af.ops.approx1Uniform(
            allocator,
            self,
            pos,
            interp_dim,
            idx_start,
            idx_step,
            method,
            off_grid,
        );
    }

    /// Signals interpolation on one dimensional signals along specified dimension;
    /// accepts a pre-allocated array, `out`, where results are written.
    ///
    /// `approx1Uniform` accepts the dimension to perform the interpolation along
    /// the input. It also accepts start and step values which define the uniform
    /// range of corresponding indices.
    pub fn approx1UniformV2(
        self: *const Self,
        out: *Self,
        pos: *const Self,
        interp_dim: i32,
        idx_start: f64,
        idx_step: f64,
        method: af.InterpType,
        off_grid: f32,
    ) !void {
        return af.ops.approx1UniformV2(
            out,
            self,
            pos,
            interp_dim,
            idx_start,
            idx_step,
            method,
            off_grid,
        );
    }

    /// Signals interpolation on two dimensional signals
    /// along specified dimensions.
    ///
    /// `approx2Uniform` accepts two dimensions to perform the interpolation
    /// along the input. It also accepts start and step values which define
    /// the uniform range of corresponding indices.
    pub fn approx2Uniform(
        self: *const Self,
        allocator: std.mem.Allocator,
        pos0: *const Self,
        interp_dim0: i32,
        idx_start_dim0: f64,
        idx_step_dim0: f64,
        pos1: *const Self,
        interp_dim1: i32,
        idx_start_dim1: f64,
        idx_step_dim1: f64,
        method: af.InterpType,
        off_grid: f32,
    ) !*Self {
        return af.ops.approx2Uniform(
            allocator,
            self,
            pos0,
            interp_dim0,
            idx_start_dim0,
            idx_step_dim0,
            pos1,
            interp_dim1,
            idx_start_dim1,
            idx_step_dim1,
            method,
            off_grid,
        );
    }

    /// Signals interpolation on two dimensional signals
    /// along specified dimensions; accepts a pre-allocated array, `out`,
    /// where results are written.
    ///
    /// `approx2UniformV2` accepts two dimensions to perform the interpolation
    /// along the input. It also accepts start and step values which define
    /// the uniform range of corresponding indices.
    pub fn approx2UniformV2(
        self: *const Self,
        out: *Self,
        pos0: *const Self,
        interp_dim0: *const Self,
        idx_start_dim0: f64,
        idx_step_dim0: f64,
        pos1: *const Self,
        interp_dim1: i32,
        idx_start_dim1: f64,
        idx_step_dim1: f64,
        method: af.InterpType,
        off_grid: f32,
    ) !void {
        return af.ops.approx2UniformV2(
            out,
            self,
            pos0,
            interp_dim0,
            idx_start_dim0,
            idx_step_dim0,
            pos1,
            interp_dim1,
            idx_start_dim1,
            idx_step_dim1,
            method,
            off_grid,
        );
    }

    /// Fast fourier transform on one dimensional signals.
    pub fn fft(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        odim0: f64,
    ) !*Self {
        return af.ops.fft(allocator, self, norm_factor, odim0);
    }

    /// In-place fast fourier transform on one dimensional signals.
    pub fn fftInplace(self: *Self, norm_factor: f64) !void {
        return af.ops.fftInplace(self, norm_factor);
    }

    /// Fast fourier transform on two dimensional signals.
    pub fn fft2(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        odim0: i64,
        odim1: i64,
    ) !*Self {
        return af.ops.fft2(allocator, self, norm_factor, odim0, odim1);
    }

    /// In-place fast fourier transform on two dimensional signals.
    pub fn fft2Inplace(self: *Self, norm_factor: f64) !void {
        return af.ops.fft2Inplace(self, norm_factor);
    }

    /// Fast fourier transform on three dimensional signals.
    pub fn fft3(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        odim0: i64,
        odim1: i64,
        odim2: i64,
    ) !*Self {
        return af.ops.fft3(
            allocator,
            self,
            norm_factor,
            odim0,
            odim1,
            odim2,
        );
    }

    /// In-place fast fourier transform on three dimensional signals.
    pub fn fft3Inplace(self: *Self, norm_factor: f64) !void {
        return af.ops.fft3Inplace(self, norm_factor);
    }

    /// Inverse fast fourier transform on one dimensional signals.
    pub fn ifft(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        odim0: i64,
    ) !*Self {
        return af.ops.ifft(allocator, self, norm_factor, odim0);
    }

    /// In-place inverse fast fourier transform on one dimensional signals.
    pub fn ifftInplace(self: *Self, norm_factor: f64) !void {
        return af.ops.ifftInplace(self, norm_factor);
    }

    /// Inverse fast fourier transform on two dimensional signals.
    pub fn ifft2(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        odim0: i64,
        odim1: i64,
    ) !*Self {
        return af.ops.ifft2(allocator, self, norm_factor, odim0, odim1);
    }

    /// In-place inverse fast fourier transform on two dimensional signals.
    pub fn ifft2Inplace(self: *Self, norm_factor: f64) !void {
        return af.ops.ifft2Inplace(self, norm_factor);
    }

    /// Inverse fast fourier transform on three dimensional signals.
    pub fn ifft3(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        odim0: i64,
        odim1: i64,
        odim2: i64,
    ) !*Self {
        return af.ops.ifft3(allocator, self, norm_factor, odim0, odim1, odim2);
    }

    /// In-place inverse fast fourier transform on three dimensional signals.
    pub fn ifft3Inplace(self: *Self, norm_factor: f64) !void {
        return af.ops.ifft3Inplace(self, norm_factor);
    }

    /// Real to complex fast fourier transform for one dimensional signals.
    pub fn fftR2C(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        pad0: i64,
    ) !*Self {
        return af.ops.fftR2C(allocator, self, norm_factor, pad0);
    }

    /// Real to complex fast fourier transform for two dimensional signals.
    pub fn fft2R2C(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        pad0: i64,
        pad1: i64,
    ) !*Self {
        return af.ops.fft2R2C(allocator, self, norm_factor, pad0, pad1);
    }

    /// Real to complex fast fourier transform for three dimensional signals.
    pub fn fft3R2C(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        pad0: i64,
        pad1: i64,
    ) !*Self {
        return af.ops.fft3R2C(allocator, self, norm_factor, pad0, pad1);
    }

    /// Complex to real fast fourier transform for one dimensional signals.
    pub fn fftC2R(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        is_odd: bool,
    ) !*Self {
        return af.ops.fftC2R(allocator, self, norm_factor, is_odd);
    }

    /// Complex to real fast fourier transform for two dimensional signals.
    pub fn fft2C2R(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        is_odd: bool,
    ) !*Self {
        return af.ops.fft2C2R(allocator, self, norm_factor, is_odd);
    }

    /// Complex to real fast fourier transform for three dimensional signals.
    pub fn fft3C2R(
        self: *const Self,
        allocator: std.mem.Allocator,
        norm_factor: f64,
        is_odd: bool,
    ) !*Self {
        return af.ops.fft3C2R(allocator, self, norm_factor, is_odd);
    }

    /// Convolution on one dimensional signals.
    pub fn convolve1(
        self: *const Self,
        allocator: std.mem.Allocator,
        filter: *const Self,
        mode: af.ConvMode,
        domain: af.ConvDomain,
    ) !*Self {
        return af.ops.convolve1(allocator, self, filter, mode, domain);
    }

    /// Convolution on two dimensional signals.
    pub fn convolve2(
        self: *const Self,
        allocator: std.mem.Allocator,
        filter: *const Self,
        mode: af.ConvMode,
        domain: af.ConvDomain,
    ) !*Self {
        return af.ops.convolve2(allocator, self, filter, mode, domain);
    }

    /// 2D convolution.
    ///
    /// This version of convolution is consistent with the
    /// machine learning formulation that will spatially convolve
    /// a filter on 2-dimensions against a signal. Multiple signals
    /// and filters can be batched against each other. Furthermore,
    /// the signals and filters can be multi-dimensional however
    /// their dimensions must match.
    ///
    /// Example: Signals with dimensions: d0 x d1 x d2 x Ns
    /// Filters with dimensions: d0 x d1 x d2 x Nf
    ///
    /// Resulting Convolution: d0 x d1 x Nf x Ns
    pub fn convolve2NN(
        self: *const Self,
        allocator: std.mem.Allocator,
        filter: *const Self,
        strid_dims: u32,
        strides: af.Dim4,
        padding_dims: u32,
        padding: af.Dim4,
        dilation_dims: u32,
        dilations: af.Dim4,
    ) !*Self {
        return af.ops.convolve2NN(
            allocator,
            self,
            filter,
            strid_dims,
            strides,
            padding_dims,
            padding,
            dilation_dims,
            dilations,
        );
    }

    /// Convolution on three dimensional signals.
    pub fn convolve3(
        self: *const Self,
        allocator: std.mem.Allocator,
        filter: *const Self,
        mode: af.ConvMode,
        domain: af.ConvDomain,
    ) !*Self {
        return af.ops.convolve3(allocator, self, filter, mode, domain);
    }

    /// Separable convolution on two dimensional signals.
    pub fn convolve2Sep(
        self: *const Self,
        allocator: std.mem.Allocator,
        col_filter: *const Self,
        row_filter: *Self,
        mode: af.ConvMode,
    ) !*Self {
        return af.ops.convolve2Sep(allocator, self, col_filter, row_filter, mode);
    }

    /// Convolution on 1D signals using FFT.
    pub fn fftConvolve1(
        self: *const Self,
        allocator: std.mem.Allocator,
        filter: *const Self,
        mode: af.ConvMode,
    ) !*Self {
        return af.ops.fftConvolve1(allocator, self, filter, mode);
    }

    /// Convolution on 2D signals using FFT.
    pub fn fftConvolve2(
        self: *const Self,
        allocator: std.mem.Allocator,
        filter: *const Self,
        mode: af.ConvMode,
    ) !*Self {
        return af.ops.fftConvolve2(allocator, self, filter, mode);
    }

    /// Convolution on 3D signals using FFT.
    pub fn fftConvolve3(
        self: *const Self,
        allocator: std.mem.Allocator,
        filter: *const Self,
        mode: af.ConvMode,
    ) !*Self {
        return af.ops.fftConvolve3(allocator, self, filter, mode);
    }

    /// Finite impulse response filter.
    pub fn fir(self: *const Self, allocator: std.mem.Allocator, x: *const Self) !*Self {
        return af.ops.fir(allocator, self, x);
    }

    /// Infinite impulse response filter.
    pub fn iir(self: *const Self, allocator: std.mem.Allocator, a: *const Self, x: *const Self) !*Self {
        return af.ops.iir(allocator, self, a, x);
    }

    /// Median filter.
    pub fn medFilt(
        self: *const Self,
        allocator: std.mem.Allocator,
        wind_length: i64,
        wind_width: i64,
        edge_pad: af.BorderType,
    ) !*Self {
        return af.ops.medFilt(allocator, self, wind_length, wind_width, edge_pad);
    }

    /// 1D median filter.
    pub fn medFilt1(
        self: *const Self,
        allocator: std.mem.Allocator,
        wind_width: i64,
        edge_pad: af.BorderType,
    ) !*Self {
        return af.ops.medFilt1(allocator, self, wind_width, edge_pad);
    }

    /// 2D median filter.
    pub fn medFilt2(
        self: *const Self,
        allocator: std.mem.Allocator,
        wind_length: i64,
        wind_width: i64,
        edge_pad: af.BorderType,
    ) !*Self {
        return af.ops.medFilt2(allocator, self, wind_length, wind_width, edge_pad);
    }

    /// Returns reference to components of the input sparse `af.Array`.
    ///
    /// Returns reference to values, row indices, column indices and
    /// storage format of an input sparse `af.Array`.
    pub fn sparseGetInfo(self: *const Self, allocator: std.mem.Allocator) !struct {
        values: *af.Array,
        rowIdx: *af.Array,
        colIdx: *af.Array,
        stype: af.Storage,
    } {
        return af.ops.sparseGetInfo(allocator, self);
    }

    /// Returns reference to the values component of the sparse `af.Array`.
    ///
    /// Values is the `af.Array` containing the non-zero elements of the
    /// dense matrix.
    pub fn sparseGetValues(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sparseGetValues(allocator, self);
    }

    /// Returns reference to the row indices component of the sparse `af.Array`.
    ///
    /// Row indices is the `af.Array` containing the row indices of the sparse array.
    pub fn sparseGetRowIdx(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sparseGetRowIdx(allocator, self);
    }

    /// Returns reference to the column indices component of the sparse `af.Array`.
    ///
    /// Column indices is the `af.Array` containing the column indices of the sparse array.
    pub fn sparseGetColIdx(self: *const Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sparseGetColIdx(allocator, self);
    }

    /// Returns the number of non zero elements in the sparse `af.Array`.
    ///
    /// This is always equal to the size of the values `af.Array`.
    pub fn sparseGetNNZ(self: *const Self) !i64 {
        return af.ops.sparseGetNNZ(self);
    }

    /// Returns the storage type of a sparse `af.Array`.
    ///
    /// The `af.Storage` type of the format of data storage
    /// in the sparse `af.Array`.
    pub fn sparseGetStorage(self: *const Self) !af.Storage {
        return af.ops.sparseGetStorage(self);
    }

    /// Returns the mean of the input `af.Array` along the specified dimension.
    pub fn mean(self: *const Self, allocator: std.mem.Allocator, dim: i64) !*Self {
        return af.ops.mean(allocator, self, dim);
    }

    /// Returns the mean of the weighted input `af.Array` along the specified dimension.
    pub fn meanWeighted(
        self: *const Self,
        allocator: std.mem.Allocator,
        weights: *const Self,
        dim: i64,
    ) !*Self {
        return af.ops.meanWeighted(allocator, self, weights, dim);
    }

    /// Returns the variance of the input `af.Array` along the specified dimension.
    pub fn var_(
        self: *const Self,
        allocator: std.mem.Allocator,
        isBiased: bool,
        dim: i64,
    ) !*Self {
        return af.ops.var_(allocator, self, isBiased, dim);
    }

    /// Returns the variance of the input `af.Array` along the specified dimension.
    /// Type of bias specified using `af.VarBias` enum.
    pub fn varV2(
        self: *const Self,
        allocator: std.mem.Allocator,
        bias: af.VarBias,
        dim: i64,
    ) !*Self {
        return af.ops.varV2(allocator, self, bias, dim);
    }

    /// Returns the variance of the weighted input `af.Array` along the specified dimension.
    pub fn varWeighted(
        self: *const Self,
        allocator: std.mem.Allocator,
        weights: *const Self,
        dim: i64,
    ) !*Self {
        return af.ops.varWeighted(allocator, self, weights, dim);
    }

    /// Returns the mean and variance of the input `af.Array` along the specified dimension.
    pub fn meanVar(
        self: *const Self,
        allocator: std.mem.Allocator,
        weights: *const Self,
        bias: *const Self,
        dim: i64,
    ) !struct {
        mean: *af.Array,
        variance: *af.Array,
    } {
        return af.ops.meanVar(allocator, self, weights, bias, dim);
    }

    /// Returns the standard deviation of the input `af.Array` along the specified dimension.
    pub fn stdev(self: *const Self, allocator: std.mem.Allocator, dim: i64) !*Self {
        return af.ops.stdev(allocator, self, dim);
    }

    /// Returns the standard deviation of the input `af.Array` along the specified dimension.
    /// Type of bias used for variance calculation is specified with `af.VarBias` enum.
    pub fn stdevV2(
        self: *const Self,
        allocator: std.mem.Allocator,
        bias: af.VarBias,
        dim: i64,
    ) !*Self {
        return af.ops.stdevV2(allocator, self, bias, dim);
    }

    /// Returns the covariance of the input `af.Array`s along the specified dimension.
    pub fn cov(
        self: *const Self,
        allocator: std.mem.Allocator,
        Y: *const Self,
        isBiased: bool,
    ) !*Self {
        return af.ops.cov(allocator, self, Y, isBiased);
    }

    /// Returns the covariance of the input `af.Array`s along the specified dimension.
    /// Type of bias used for variance calculation is specified with `af.VarBias` enum.
    pub fn covV2(
        self: *const Self,
        allocator: std.mem.Allocator,
        Y: *const Self,
        bias: af.VarBias,
    ) !*Self {
        return af.ops.covV2(allocator, self, Y, bias);
    }

    /// Returns the median of the input `af.Array` across the specified dimension.
    pub fn median(self: *const Self, allocator: std.mem.Allocator, dim: i64) !*Self {
        return af.ops.median(allocator, self, dim);
    }

    /// Returns both the real part and imaginary part of the mean
    /// of the entire input `af.Array`.
    pub fn meanAll(self: *const Self) !af.ops.ComplexParts {
        return af.ops.meanAll(self);
    }

    /// Returns both the real part and imaginary part of the mean
    /// of the entire weighted input `af.Array`.
    pub fn meanAllWeighted(self: *const Self, weights: *const Self) !af.ops.ComplexParts {
        return af.ops.meanAllWeighted(self, weights);
    }

    /// Returns both the real part and imaginary part of the variance
    /// of the entire weighted input `af.Array`.
    pub fn varAll(self: *const Self, isBiased: bool) !af.ops.ComplexParts {
        return af.ops.varAll(self, isBiased);
    }

    /// Returns both the real part and imaginary part of the variance
    /// of the entire weighted input `af.Array`.
    ///
    /// Type of bias used for variance calculation is specified with
    /// `af.VarBias` enum.
    pub fn varAllV2(self: *const Self, bias: af.VarBias) !af.ops.ComplexParts {
        return af.ops.varAllV2(self, bias);
    }

    /// Returns both the real part and imaginary part of the variance
    /// of the entire weighted input `af.Array`.
    pub fn varAllWeighted(self: *const Self, weights: *const Self) !af.ops.ComplexParts {
        return af.ops.varAllWeighted(self, weights);
    }

    /// Returns both the real part and imaginary part of the standard
    /// deviation of the entire input `af.Array`.
    pub fn stdevAll(self: *const Self) !af.ops.ComplexParts {
        return af.ops.stdevAll(self);
    }

    /// Returns both the real part and imaginary part of the standard
    /// deviation of the entire input `af.Array`.
    ///
    /// Type of bias used for variance calculation is specified with
    /// `af.VarBias` enum.
    pub fn stdevAllV2(self: *const Self, bias: af.VarBias) !af.ops.ComplexParts {
        return af.ops.stdevAllV2(self, bias);
    }

    /// Returns both the real part and imaginary part of the median
    /// of the entire input `af.Array`.
    pub fn medianAll(self: *const Self) !af.ops.ComplexParts {
        return af.ops.medianAll(self);
    }

    /// Returns both the real part and imaginary part of the correlation
    /// coefficient of the input `af.Array`s.
    pub fn corrCoef(self: *const Self, Y: *const Self) !af.ops.ComplexParts {
        return af.ops.corrCoef(self, Y);
    }

    /// This function returns the top k values along a given dimension
    /// of the input `af.Array`.
    ///
    /// The indices along with their values are returned. If the input is a
    /// multi-dimensional `af.Array`, the indices will be the index of the
    /// value in that dimension. Order of duplicate values are not preserved.
    /// This function is optimized for small values of k.
    ///
    /// This function performs the operation across all dimensions of the input array.
    pub fn topk(self: *const Self, allocator: std.mem.Allocator, k: i32, dim: i32, order: af.TopkFn) !struct {
        values: *Self,
        indices: *Self,
    } {
        return af.ops.topk(allocator, self, k, dim, order);
    }

    /// Returns `af.Features` struct containing arrays for x and y coordinates
    /// and score, while array orientation is set to 0 as FAST does not compute
    /// orientation, and size is set to 1 as FAST does not compute multiple scales.
    pub fn fast(
        self: *const Self,
        allocator: std.mem.Allocator,
        thr: f32,
        arc_length: u32,
        non_max: bool,
        feature_ratio: f32,
        edge: u32,
    ) !*af.Features {
        return af.ops.fast(
            allocator,
            self,
            thr,
            arc_length,
            non_max,
            feature_ratio,
            edge,
        );
    }

    /// Returns `af.Features` struct containing arrays for x and y coordinates
    /// and score (Harris response), while arrays orientation and size are set
    /// to 0 and 1, respectively, because Harris does not compute that information.
    pub fn harris(
        self: *const Self,
        allocator: std.mem.Allocator,
        max_corners: u32,
        min_response: f32,
        sigma: f32,
        block_size: u32,
        k_thr: f32,
    ) !*af.Features {
        return af.ops.harris(
            allocator,
            self,
            max_corners,
            min_response,
            sigma,
            block_size,
            k_thr,
        );
    }

    /// Returns struct containing the following fields:
    /// - `feat`: `af.Features` composed of arrays for x and y coordinates,
    /// score, orientation and size of selected features.
    /// - `desc`: Nx8 `af.Array` containing extracted
    /// descriptors, where N is the number of selected features.
    pub fn orb(
        self: *const Self,
        allocator: std.mem.Allocator,
        fast_thr: f32,
        max_feat: u32,
        scl_fctr: f32,
        levels: u32,
        blur_img: bool,
    ) !struct {
        feat: *af.Features,
        desc: *Self,
    } {
        return af.ops.orb(
            allocator,
            self,
            fast_thr,
            max_feat,
            scl_fctr,
            levels,
            blur_img,
        );
    }

    /// Returns `FeatDesc` struct containing the following fields:
    /// - `feat`: `af.Features` composed of arrays for x and y coordinates,
    /// score, orientation and size of selected features.
    /// - `desc`: Nx128 `af.Array` containing extracted descriptors,
    /// where N is the number of features found by SIFT.
    pub fn sift(
        self: *const Self,
        allocator: std.mem.Allocator,
        n_layers: u32,
        contrast_thr: f32,
        edge_thr: f32,
        init_sigma: f32,
        double_input: bool,
        intensity_scale: f32,
        feature_ratio: f32,
    ) !struct {
        feat: *af.Features,
        desc: *Self,
    } {
        return af.ops.sift(
            allocator,
            self,
            n_layers,
            contrast_thr,
            edge_thr,
            init_sigma,
            double_input,
            intensity_scale,
            feature_ratio,
        );
    }

    /// Returns struct containing the following fields:
    /// - `feat`: `af.Features` composed of arrays for x and y coordinates,
    /// score, orientation and size of selected features.
    /// - `desc`: Nx272 `af.Array` containing extracted GLOH descriptors,
    /// where N is the number of features found by SIFT.
    pub fn gloh(
        self: *const Self,
        allocator: std.mem.Allocator,
        n_layers: u32,
        contrast_thr: f32,
        edge_thr: f32,
        init_sigma: f32,
        double_input: bool,
        intensity_scale: f32,
        feature_ratio: f32,
    ) !struct {
        feat: *af.Features,
        desc: *Self,
    } {
        return af.ops.gloh(
            allocator,
            self,
            n_layers,
            contrast_thr,
            edge_thr,
            init_sigma,
            double_input,
            intensity_scale,
            feature_ratio,
        );
    }

    /// Calculates Hamming distances between two 2-dimensional `af.Array`s containing
    /// features, one of the `af.Array`s containing the training data and the other
    /// the query data. One of the dimensions of the both `af.Array`s must be equal
    /// among them, identifying the length of each feature. The other dimension
    /// indicates the total number of features in each of the training and query
    /// `af.Array`s. Two 1-dimensional `af.Array`s are created as results, one containg
    /// the smallest N distances of the query `af.Array` and another containing the indices
    /// of these distances in the training `af.Array`s. The resulting 1-dimensional `af.Array`s
    /// have length equal to the number of features contained in the query `af.Array`.
    ///
    /// Returns struct containing the fields:
    /// - `idx`: `af.Array` of MxN size, where M is equal to the number
    /// of query features and N is equal to n_dist. The value at position
    /// IxJ indicates the index of the Jth smallest distance to the Ith
    /// query value in the train data array. the index of the Ith smallest
    /// distance of the Mth query.
    /// - `dist`: `af.Array` of MxN size, where M is equal to the number
    /// of query features and N is equal to n_dist. The value at position
    /// IxJ indicates the Hamming distance of the Jth smallest distance
    /// to the Ith query value in the train data `af.Array`.
    pub fn hammingMatcher(
        self: *const Self,
        allocator: std.mem.Allocator,
        train: *const Self,
        dist_dim: i64,
        n_dist: u32,
    ) !struct {
        idx: *Self,
        dist: *Self,
    } {
        return af.ops.hammingMatcher(
            allocator,
            self,
            train,
            dist_dim,
            n_dist,
        );
    }

    /// Determines the nearest neighbouring points to a given set of points.
    ///
    /// Returns struct containing the fields:
    /// - `idx`: `af.Array` of M×N size, where M is n_dist and N is the
    /// number of queries. The value at position i,j is the index of the point
    /// in train along dim1 (if dist_dim is 0) or along dim 0 (if dist_dim is 1),
    /// with the ith smallest distance to the jth query point.
    /// - `dist`: `af.Array` of M×N size, where M is n_dist and N is the number
    /// of queries. The value at position i,j is the distance from the jth query
    /// point to the point in train referred to by idx( i,j). This distance is
    /// computed according to the dist_type chosen.
    pub fn nearestNeighbor(
        self: *const Self,
        allocator: std.mem.Allocator,
        train: *const Self,
        dist_dim: i64,
        n_dist: u32,
        dist_type: af.MatchType,
    ) !struct {
        idx: *Self,
        dist: *Self,
    } {
        return af.ops.nearestNeighbor(
            allocator,
            self,
            train,
            dist_dim,
            n_dist,
            dist_type,
        );
    }

    /// Template matching is an image processing technique to find small patches
    /// of an image which match a given template image. A more in depth discussion
    /// on the topic can be found [here](https://en.wikipedia.org/wiki/Template_matching).
    ///
    /// Returns an `af.Array` containing disparity values for the window starting at
    /// corresponding pixel position.
    pub fn matchTemplate(
        self: *const Self,
        allocator: std.mem.Allocator,
        template_img: *const Self,
        m_type: af.MatchType,
    ) !*Self {
        return af.ops.matchTemplate(allocator, self, template_img, m_type);
    }

    /// SUSAN corner detector.
    ///
    /// Returns `af.Features` struct composed of `af.Array`s for x and y coordinates,
    /// score, orientation and size of selected features.
    pub fn susan(
        self: *const Self,
        allocator: std.mem.Allocator,
        radius: u32,
        diff_thr: f32,
        geom_thr: f32,
        feature_ratio: f32,
        edge: u32,
    ) !*af.Features {
        return af.ops.susan(
            allocator,
            self,
            radius,
            diff_thr,
            geom_thr,
            feature_ratio,
            edge,
        );
    }

    /// Difference of Gaussians.
    ///
    /// Given an image, this function computes two different versions
    /// of smoothed input image using the difference smoothing parameters
    /// and subtracts one from the other and returns the result.
    ///
    /// Returns an `af.Array` containing the calculated difference of smoothed inputs.
    pub fn dog(self: *const Self, allocator: std.mem.Allocator, radius1: i32, radius2: i32) !*Self {
        return af.ops.dog(allocator, self, radius1, radius2);
    }

    /// Get strides of underlying data.
    pub fn getStrides(self: *const Self) !af.Dim4 {
        return af.ops.getStrides(self);
    }

    /// Returns bool indicating whether all elements in an `af.Array` are contiguous.
    pub fn isLinear(self: *const Self) !bool {
        return af.ops.isLinear(self);
    }
};
