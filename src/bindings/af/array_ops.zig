const std = @import("std");
const af = @import("ArrayFire.zig");
const zt_types = @import("../../tensor/Types.zig");
const zt_base = @import("../../tensor/TensorBase.zig");
const zt_shape = @import("../../tensor/Shape.zig");

const Location = zt_base.Location;
const Shape = zt_shape.Shape;
const Dim = zt_shape.Dim;

/// Returns the sum of the elements of the input `af.Array`
/// along the given dimension.
pub inline fn sum(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sum(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the sum of the elements of the input `af.Array` along
/// the given dimension, replacing NaN values with `nanval`.
pub inline fn sumNan(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
    nanval: f64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sum_nan(
            &arr,
            in.array_,
            @intCast(dim),
            nanval,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the sum of the elements in the input `af.Array` by
/// key along the given dimension according to key.
pub inline fn sumByKey(
    allocator: std.mem.Allocator,
    keys: *const af.Array,
    vals: *const af.Array,
    dim: i32,
) !struct { keys_out: *af.Array, vals_out: *af.Array } {
    var keys_out: af.af_array = undefined;
    var vals_out: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sum_by_key(
            &keys_out,
            &vals_out,
            keys.array_,
            vals.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .keys_out = try af.Array.init(allocator, keys_out),
        .vals_out = try af.Array.init(allocator, vals_out),
    };
}

/// Returns the sum of the elements in an `af.Array` by key along
/// the given dimension according to key; replaces NaN values with
/// the specified `nanval`.
pub inline fn sumByKeyNan(
    allocator: std.mem.Allocator,
    keys: *const af.Array,
    vals: *const af.Array,
    dim: i32,
    nanval: f64,
) !struct { keys_out: *af.Array, vals_out: *af.Array } {
    var keys_out: af.af_array = undefined;
    var vals_out: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sum_by_key_nan(
            &keys_out,
            &vals_out,
            keys.array_,
            vals.array_,
            @intCast(dim),
            nanval,
        ),
        @src(),
    );
    return .{
        .keys_out = try af.Array.init(allocator, keys_out),
        .vals_out = try af.Array.init(allocator, vals_out),
    };
}

/// Returns the product of all values in the input `af.Array`
/// along the specified dimension.
pub inline fn product(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_product(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the product of all values in the input `af.Array` along the
/// specified dimension; replaces NaN values with specified `nanval`.
pub inline fn productNan(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
    nanval: f64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_product_nan(
            &arr,
            in.array_,
            @intCast(dim),
            nanval,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the product of all values in the input `af.Array`
/// along the specified dimension according to key.
pub inline fn productByKey(
    allocator: std.mem.Allocator,
    keys: *const af.Array,
    vals: *const af.Array,
    dim: i32,
) !struct { keys_out: *af.Array, vals_out: *af.Array } {
    var keys_out: af.af_array = undefined;
    var vals_out: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_product_by_key(
            &keys_out,
            &vals_out,
            keys.array_,
            vals.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .keys_out = try af.Array.init(allocator, keys_out),
        .vals_out = try af.Array.init(allocator, vals_out),
    };
}

/// Returns the sum of the elements in the input `af.Array` by key along
/// the given dimension according to key; replaces NaN values with
/// the specified `nanval`.
pub inline fn productByKeyNan(
    allocator: std.mem.Allocator,
    keys: *const af.Array,
    vals: *const af.Array,
    dim: i32,
    nanval: f64,
) !struct { keys_out: *af.Array, vals_out: *af.Array } {
    var keys_out: af.af_array = undefined;
    var vals_out: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_product_by_key_nan(
            &keys_out,
            &vals_out,
            keys.array_,
            vals.array_,
            @intCast(dim),
            nanval,
        ),
        @src(),
    );
    return .{
        .keys_out = try af.Array.init(allocator, keys_out),
        .vals_out = try af.Array.init(allocator, vals_out),
    };
}

/// Returns the minimum of all values in the input `af.Array`
/// along the specified dimension.
pub inline fn min(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_min(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the minimum of all values in the input `af.Array`
/// along the specified dimension according to key.
pub inline fn minByKey(
    allocator: std.mem.Allocator,
    keys: *const af.Array,
    vals: *const af.Array,
    dim: i32,
) !struct { keys_out: *af.Array, vals_out: *af.Array } {
    var keys_out: af.af_array = undefined;
    var vals_out: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_min_by_key(
            &keys_out,
            &vals_out,
            keys.array_,
            vals.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .keys_out = try af.Array.init(allocator, keys_out),
        .vals_out = try af.Array.init(allocator, vals_out),
    };
}

/// Returns the maximum of all values in the input `af.Array`
/// along the specified dimension.
pub inline fn max(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_max(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the maximum of all values in the input `af.Array`
/// along the specified dimension according to key.
pub inline fn maxByKey(
    allocator: std.mem.Allocator,
    keys: *const af.Array,
    vals: *const af.Array,
    dim: i32,
) !struct { keys_out: *af.Array, vals_out: *af.Array } {
    var keys_out: af.af_array = undefined;
    var vals_out: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_max_by_key(
            &keys_out,
            &vals_out,
            keys.array_,
            vals.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .keys_out = try af.Array.init(allocator, keys_out),
        .vals_out = try af.Array.init(allocator, vals_out),
    };
}

/// Finds ragged max values in an `af.Array`; uses an additional input
/// `af.Array` to determine the number of elements to use along the
/// reduction axis.
///
/// Returns struct containing the following fields:
/// - `val`: `af.Array` containing the maximum ragged values in
/// the input `af.Array` along the specified dimension according
/// `ragged_len`.
/// - `idx`: `af.Array` containing the locations of the maximum
/// ragged values in the input `af.Array` along the specified
/// dimension according to `ragged_len`.
pub inline fn maxRagged(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    ragged_len: *const af.Array,
    dim: i32,
) !struct { val: *af.Array, idx: *af.Array } {
    var val: af.af_array = undefined;
    var idx: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_max_ragged(
            &val,
            &idx,
            in.array_,
            ragged_len.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .val = try af.Array.init(allocator, val),
        .idx = try af.Array.init(allocator, idx),
    };
}

/// Tests if all values in the input `af.Array` along the
/// specified dimension are true.
pub inline fn allTrue(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_all_true(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Tests if all values in the input `af.Array` along the
/// specified dimension are true accord to key.
pub inline fn allTrueByKey(
    allocator: std.mem.Allocator,
    keys: *const af.Array,
    vals: *const af.Array,
    dim: i32,
) !struct { keys_out: *af.Array, vals_out: *af.Array } {
    var keys_out: af.af_array = undefined;
    var vals_out: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_all_true_by_key(
            &keys_out,
            &vals_out,
            keys.array_,
            vals.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .keys_out = try af.Array.init(allocator, keys_out),
        .vals_out = try af.Array.init(allocator, vals_out),
    };
}

/// Tests if any values in the input `af.Array` along the
/// specified dimension are true.
pub inline fn anyTrue(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_any_true(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Tests if any values in the input `af.Array` along the
/// specified dimension are true according to key.
pub inline fn anyTrueByKey(
    allocator: std.mem.Allocator,
    keys: *const af.Array,
    vals: *const af.Array,
    dim: i32,
) !struct { keys_out: *af.Array, vals_out: *af.Array } {
    var keys_out: af.af_array = undefined;
    var vals_out: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_any_true_by_key(
            &keys_out,
            &vals_out,
            keys.array_,
            vals.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .keys_out = try af.Array.init(allocator, keys_out),
        .vals_out = try af.Array.init(allocator, vals_out),
    };
}

/// Count the number of non-zero elements in the input `af.Array`.
///
/// Return type is `af.Dtype.u32` for all input types.
///
/// This function performs the operation across all batches present
/// in the input simultaneously.
pub inline fn count(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_count(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Counts the non-zero values of an input `af.Array` according to an
/// `af.Array` of keys.
///
/// All non-zero values corresponding to each group of consecutive equal
/// keys will be counted. Keys can repeat, however only consecutive key
/// values will be considered for each reduction. If a key value is repeated
/// somewhere else in the keys array it will be considered the start of a new
/// reduction. There are two outputs: the reduced set of consecutive keys and
/// the corresponding final reduced values.
pub inline fn countByKey(
    allocator: std.mem.Allocator,
    keys: *const af.Array,
    vals: *const af.Array,
    dim: i32,
) !struct { keys_out: *af.Array, vals_out: *af.Array } {
    var keys_out: af.af_array = undefined;
    var vals_out: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_count_by_key(
            &keys_out,
            &vals_out,
            keys.array_,
            vals.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .keys_out = try af.Array.init(allocator, keys_out),
        .vals_out = try af.Array.init(allocator, vals_out),
    };
}

/// Returns the sum of all elements in an input `af.Array`.
pub inline fn sumAll(in: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_sum_all(
            &res.real,
            &res.imag,
            in.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns the sum of all elements in an input `af.Array`
/// replacing NaN values with `nanval`.
pub inline fn sumNanAll(in: *const af.Array, nanval: f64) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_sum_nan_all(
            &res.real,
            &res.imag,
            in.array_,
            nanval,
        ),
        @src(),
    );
    return res;
}

/// Returns the product of all elements in an input `af.Array`.
pub inline fn productAll(in: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_product_all(
            &res.real,
            &res.imag,
            in.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns the product of all elements in an input `af.Array`
/// replacing NaN values with `nanval`.
pub inline fn productNanAll(in: *const af.Array, nanval: f64) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_product_nan_all(
            &res.real,
            &res.imag,
            in.array_,
            nanval,
        ),
        @src(),
    );
    return res;
}

/// Returns the minimum value of all elements in an input `af.Array`.
pub inline fn minAll(in: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_min_all(
            &res.real,
            &res.imag,
            in.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns the maximum value of all elements in an input `af.Array`.
pub inline fn maxAll(in: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_max_all(
            &res.real,
            &res.imag,
            in.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns whether all elements in an input `af.Array` are true.
pub inline fn allTrueAll(in: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_all_true_all(
            &res.real,
            &res.imag,
            in.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns whether any elements in an input `af.Array` are true.
pub inline fn anyTrueAll(in: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_any_true_all(
            &res.real,
            &res.imag,
            in.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns the number of non-zero elements in an input `af.Array`.
pub inline fn countAll(in: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_count_all(
            &res.real,
            &res.imag,
            in.array_,
        ),
        @src(),
    );
    return res;
}

/// Find the minimum values and their locations.
///
/// This function performs the operation across all
/// batches present in the input simultaneously.
///
/// Returns struct containing the following fields:
/// - `out`: `af.Array` containing the minimum of all values
/// in the input `af.Array` along the specified dimension.
/// - `idx`: `af.Array` containg the location of the minimum of
/// all values in the input `af.Array` along the specified dimension.
pub inline fn imin(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
) !struct { out: *af.Array, idx: *af.Array } {
    var out: af.af_array = undefined;
    var idx: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_imin(
            &out,
            &idx,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .out = try af.Array.init(allocator, out),
        .idx = try af.Array.init(allocator, idx),
    };
}

/// Find the maximum value and its location.
///
/// This function performs the operation across all
/// batches present in the input simultaneously.
///
/// Returns struct containing the following fields:
/// - `out`: `af.Array` containing the maximum of all values
/// in the input `af.Array` along the specified dimension.
/// - `idx`: `af.Array` containg the location of the maximum of
/// all values in the input `af.Array` along the specified dimension.
pub inline fn imax(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
) !struct { out: *af.Array, idx: *af.Array } {
    var out: af.af_array = undefined;
    var idx: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_imax(
            &out,
            &idx,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .out = try af.Array.init(allocator, out),
        .idx = try af.Array.init(allocator, idx),
    };
}

/// Returns minimum value and its location from the entire array.
pub inline fn iminAll(in: *const af.Array) !struct { real: f64, imag: f64, idx: u32 } {
    var res: struct { real: f64, imag: f64, idx: u32 } = .{
        .real = undefined,
        .imag = undefined,
        .idx = undefined,
    };
    var idx: c_uint = undefined;
    try af.AF_CHECK(
        af.af_imin_all(
            &res.real,
            &res.imag,
            &idx,
            in.array_,
        ),
        @src(),
    );
    res.idx = @intCast(idx);
    return res;
}

/// Returns maximum value and its location from the entire array.
pub inline fn imaxAll(in: *const af.Array) !struct { real: f64, imag: f64, idx: u32 } {
    var res: struct { real: f64, imag: f64, idx: i32 } = .{
        .real = undefined,
        .imag = undefined,
        .idx = undefined,
    };
    var idx: c_uint = undefined;
    try af.AF_CHECK(
        af.af_imax_all(
            &res.real,
            &res.imag,
            &idx,
            in.array_,
        ),
        @src(),
    );
    res.idx = @intCast(idx);
    return res;
}

/// Returns the computed cumulative sum (inclusive) of an `af.Array`.
pub inline fn accum(allocator: std.mem.Allocator, in: *const af.Array, dim: i32) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_accum(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Generalized scan of the input `af.Array`.
///
/// Returns ptr to an `af.Array` containing scan of the input.
pub inline fn scan(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
    op: af.BinaryOp,
    inclusive_scan: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_scan(
            &arr,
            in.array_,
            @intCast(dim),
            op.value(),
            inclusive_scan,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

//// Generalized scan by key of the input `af.Array`.
///
/// Returns ptr to an `af.Array` containing scan of the input
/// by key.
pub inline fn scanByKey(
    allocator: std.mem.Allocator,
    key: *const af.Array,
    in: *const af.Array,
    dim: i32,
    op: af.BinaryOp,
    inclusive_scan: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_scan_by_key(
            &arr,
            key.array_,
            in.array_,
            @intCast(dim),
            op.value(),
            inclusive_scan,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Locate the indices of non-zero elements in the input
/// `af.Array`.
///
/// Return type is u32 for all input types
///
/// The locations are provided by flattening the input
/// `af.Array` into a linear `af.Array`.
pub inline fn where(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_where(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// First order numerical difference along specified dimension.
///
/// This function performs the operation across all batches
/// present in the input `af.Array` simultaneously.
pub inline fn diff1(allocator: std.mem.Allocator, in: *const af.Array, dim: i32) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_diff1(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Second order numerical difference along specified dimension.
///
/// This function performs the operation across all batches
/// present in the input `af.Array` simultaneously.
pub inline fn diff2(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_diff2(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Sort a multidimensional input `af.Array`.
///
/// Returns ptr to `af.Array` containing sorted output.
pub inline fn sort(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: u32,
    is_ascending: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sort(
            &arr,
            in.array_,
            @intCast(dim),
            is_ascending,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Sort a multi dimensional `af.Array` and return sorted indices.
///
/// Index `af.Array` is of `af.Dtype.u32`.
///
/// Returns struct containing the following fields:
/// - `out`: `af.Array` containing sorted output.
/// - `indices`: `af.Array` containing the indices
/// in the original input.
pub inline fn sortIndex(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: u32,
    is_ascending: bool,
) !struct { out: *af.Array, idx: *af.Array } {
    var out: af.af_array = undefined;
    var indices: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sort_index(
            &out,
            &indices,
            in.array_,
            @intCast(dim),
            is_ascending,
        ),
        @src(),
    );
    return .{ .out = try af.Array.init(allocator, out), .idx = try af.Array.init(allocator, indices) };
}

/// Sort a multi dimensional `af.Array` based on keys.
///
/// Returns struct containing the following fields:
/// - `out_keys`: `af.Array` containing the keys based
/// on sorted values.
/// - `out_values`: `af.Array` containing the sorted values.
pub inline fn sortByKey(
    allocator: std.mem.Allocator,
    keys: *const af.Array,
    values: *const af.Array,
    dim: u32,
    is_ascending: bool,
) !struct { out_keys: *af.Array, out_values: *af.Array } {
    var out_keys: af.af_array = undefined;
    var out_values: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sort_by_key(
            &out_keys,
            &out_values,
            keys.array_,
            values.array_,
            @intCast(dim),
            is_ascending,
        ),
        @src(),
    );
    return .{
        .out_keys = try af.Array.init(allocator, out_keys),
        .out_values = try af.Array.init(allocator, out_values),
    };
}

/// Finds unique values from an input `af.Array`.
///
/// The input must be a one-dimensional `af.Array`. Batching
/// is not currently supported.
pub inline fn setUnique(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    is_sorted: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_set_unique(
            &arr,
            in.array_,
            is_sorted,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Find the union of two `af.Array`s.
///
/// The inputs must be one-dimensional `af.Array`s.
/// Batching is not currently supported.
pub inline fn setUnion(
    allocator: std.mem.Allocator,
    first: *const af.Array,
    second: *const af.Array,
    is_unique: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_set_union(
            &arr,
            first.array_,
            second.array_,
            is_unique,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Find the intersection of two `af.Array`s.
///
/// The inputs must be one-dimensional `af.Array`s.
/// Batching is not currently supported.
pub inline fn setIntersect(
    allocator: std.mem.Allocator,
    first: *const af.Array,
    second: *const af.Array,
    is_unique: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_set_intersect(
            &arr,
            first.array_,
            second.array_,
            is_unique,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Performs element wise addition on two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn add(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_add(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Performs element wise subtraction on two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn sub(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sub(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Performs element wise multiplication on two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn mul(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_mul(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Performs element wise division on two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn div(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_div(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Perform a less-than comparison between
/// corresponding elements of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn lt(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_lt(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Perform a greater-than comparison between
/// corresponding elements of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn gt(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_gt(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Perform a less-than-or-equal comparison between
/// corresponding elements of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn le(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_le(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Perform a greater-than-or-equal comparison between
/// corresponding elements of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn ge(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_ge(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Checks if corresponding elements of two `af.Array`s
/// are equal.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn eq(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_eq(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Checks if corresponding elements of two `af.Array`s
/// are not equal.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn neq(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_neq(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the logical AND of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn and_(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_and(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the logical OR of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn or_(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_or(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the logical NOT of the input `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn not(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_not(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the bitwise NOT of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn bitNot(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_bitnot(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the bitwise AND of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn bitAnd(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_bitand(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the bitwise OR of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn bitOr(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_bitor(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the bitwise XOR of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn bitXor(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_bitxor(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Shift the bits of integer `af.Array` left.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn bitShiftL(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_bitshiftl(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Shift the bits of integer `af.Array` right.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn bitShiftR(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_bitshiftr(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Casts an `af.Array` from one type to another.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn cast(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dtype: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_cast(
            &arr,
            in.array_,
            dtype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the elementwise minimum between two `af.Array`s.
pub inline fn minOf(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_minof(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the elementwise maximum between two `af.Array`s.
pub inline fn maxOf(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_maxof(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Clamp an `af.Array` between an upper and a lower limit.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn clamp(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    lo: *const af.Array,
    hi: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_clamp(
            &arr,
            in.array_,
            lo.array_,
            hi.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Calculate the remainder.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn rem(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_rem(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Calculate the modulus.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn mod(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_mod(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Calculate the absolute value.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn abs(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_abs(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Calculate the phase angle (in radians) of a complex `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn arg(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fabs(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Calculate the sign of elements in an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn sign(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sign(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Round numbers in an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn round(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_round(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Truncate numbers in an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn trunc(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_trunc(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Floor numbers in an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn floor(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_floor(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Ceil numbers in an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn ceil(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_ceil(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Calculate the length of the hypotenuse of two input
/// `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn hypot(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_hypot(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the sine function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn sin(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sin(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the cosine function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn cos(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_cos(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the tangent function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn tan(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_tan(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the inverse sine function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn asin(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_asin(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the inverse cosine function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn acos(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_acos(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the inverse tangent function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn atan(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_atan(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the inverse tangent function of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn atan2(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_atan2(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the hyperbolic sine function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn sinh(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sinh(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the hyperbolic cosine function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn cosh(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_cosh(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the hyperbolic tangent function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn tanh(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_tanh(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the inverse hyperbolic sine function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn asinh(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_asinh(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the inverse hyperbolic cosine function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn acosh(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_acosh(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the inverse hyperbolic tangent function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn atanh(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_atanh(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Initializes and returns a complex `af.Array` from a
/// single real `af.Array`.
pub inline fn cplx(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_cplx(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Initializes and returns a complex `af.Array` from
/// two real `af.Array`s.
pub inline fn cplx2(
    allocator: std.mem.Allocator,
    real_: *const af.Array,
    imag_: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_cplx2(
            &arr,
            real_.array_,
            imag_.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the real part of a complex `af.Array`.
pub inline fn real(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_real(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the imaginary part of a complex `af.Array`.
pub inline fn imag(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_imag(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the complex conjugate of an input `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn conjg(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_conjg(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the nth root of two `af.Array`s.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn root(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_root(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Raise a base to a power (or exponent).
///
/// Returns ptr the resulting `af.Array`.
pub inline fn pow(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    batch: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_pow(
            &arr,
            lhs.array_,
            rhs.array_,
            batch,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Raise 2 to a power (or exponent).
///
/// Returns ptr the resulting `af.Array`.
pub inline fn pow2(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_pow2(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the logistical sigmoid function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn sigmoid(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sigmoid(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the exponential of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn exp(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_exp(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the exponential of an `af.Array` minus 1, exp(in) - 1.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn expm1(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_expm1(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the error function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn erf(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_erf(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the complementary error function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn erfc(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_erfc(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the natural logarithm of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn log(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_log(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the natural logarithm of an `af.Array` plus 1, ln(1+in).
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn log1p(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_log1p(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the base 10 logarithm of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn log10(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_log10(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the base 2 logarithm of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn log2(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_log2(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the square root of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn sqrt(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sqrt(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the reciprocal square root of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn rsqrt(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_rsqrt(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the cube root of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn cbrt(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_cbrt(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Calculate the factorial of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn factorial(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_factorial(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the gamma function of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn tgamma(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_tgamma(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Evaluate the logarithm of the absolute value of the gamma function
/// of an `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn lgamma(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_lgamma(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Check if values of an `af.Array` are zero.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn isZero(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_iszero(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Check if values of an `af.Array` are infinite.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn isInf(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_isinf(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Check if values of an `af.Array` are NaN.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn isNan(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_isnan(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Initializes an `af.Array` from device memory.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn deviceArray(
    allocator: std.mem.Allocator,
    data: ?*anyopaque,
    ndims: u32,
    dims: af.Dim4,
    dtype: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_device_array(
            &arr,
            data,
            @intCast(ndims),
            &dims.dims,
            dtype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Lock the device buffer in the memory manager.
pub inline fn lockDevicePtr(arr: *const af.Array) !void {
    try af.AF_CHECK(af.af_lock_device_ptr(arr.array_), @src());
}

/// Unlock device buffer in the memory manager.
pub inline fn unlockDevicePtr(arr: *const af.Array) !void {
    try af.AF_CHECK(af.af_unlock_device_ptr(arr.array_), @src());
}

/// Lock the device buffer in the memory manager.
pub inline fn lockArray(arr: *const af.Array) !void {
    try af.AF_CHECK(af.af_lock_array(arr.array_), @src());
}

/// Unlock device buffer in the memory manager.
pub inline fn unlockArray(arr: *const af.Array) !void {
    try af.AF_CHECK(af.af_unlock_array(arr.array_), @src());
}

/// Query if the `af.Array` has been locked by the user.
pub inline fn isLockedArray(arr: *const af.Array) !bool {
    var is_locked: bool = undefined;
    try af.AF_CHECK(af.af_is_locked_array(&is_locked, arr.array_), @src());
    return is_locked;
}

/// Get the device pointer and lock the buffer in memory manager.
pub inline fn getDevicePtr(arr: *const af.Array) !?*anyopaque {
    var ptr: ?*anyopaque = undefined;
    try af.AF_CHECK(af.af_get_device_ptr(&ptr, arr.array_), @src());
    return ptr;
}

/// Lookup the values of the input `af.Array` based on sequences.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn index(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    ndims: u32,
    idx: *const af.af_seq,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_index(
            &arr,
            in.array_,
            @intCast(ndims),
            idx,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Lookup the values of an input `af.Array` by indexing
/// with another `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn lookup(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    indices: *const af.Array,
    dim: u32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_lookup(
            &arr,
            in.array_,
            indices.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Copy and write values in the locations specified
/// by the sequences.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn assignSeq(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    ndims: u32,
    indices: *const af.af_seq,
    rhs: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_assign_seq(
            &arr,
            lhs.array_,
            @intCast(ndims),
            indices,
            rhs.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Indexing an array using `af.af_seq`, or `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn indexGen(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    ndims: i64,
    indices: []const af.af_index_t,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_index_gen(
            &arr,
            in.array_,
            @intCast(ndims),
            indices.ptr,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Assignment of an `af.Array` using `af.af_seq`, or `af.Array`.
///
/// Generalized assignment function that accepts either `af.Array`
/// or `af.af_seq` along a dimension to assign elements from an
/// input `af.Array` to an output `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn assignGen(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    ndims: i64,
    indices: *const af.af_index_t,
    rhs: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_assign_gen(
            &arr,
            lhs.array_,
            @intCast(ndims),
            indices,
            rhs.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

// TODO: fix as not working
/// Print the `af.Array` and dimensions to screen.
pub inline fn printArray(arr: *const af.Array) !void {
    try af.AF_CHECK(af.af_print_array(arr.array_), @src());
}

/// Print the expression, `af.Array`, and dimensions to screen.
pub inline fn printArrayGen(expr: []const u8, arr: *const af.Array, precision: i32) !void {
    try af.AF_CHECK(
        af.af_print_array_gen(
            expr.ptr,
            arr.array_,
            @intCast(precision),
        ),
        @src(),
    );
}

/// Save an `af.Array` to a binary file.
///
/// The `saveArray` and `readArray` functions are designed
/// to provide store and read access to `af.Array`s using files
/// written to disk.
///
/// Returns the index location of the `af.Array` in the file.
pub inline fn saveArray(
    key: []const u8,
    arr: *const af.Array,
    filename: []const u8,
    append: bool,
) !i32 {
    var idx: c_int = undefined;
    try af.AF_CHECK(
        af.af_save_array(
            &idx,
            key.ptr,
            arr.array_,
            filename.ptr,
            append,
        ),
        @src(),
    );
    return @intCast(idx);
}

/// Reads an `af.Array` saved in files using the index
/// in the file (0-indexed).
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn readArrayIndex(
    allocator: std.mem.Allocator,
    filename: []const u8,
    idx: u32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_read_array_index(
            &arr,
            filename.ptr,
            @intCast(idx),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Reads an `af.Array` saved in files using the key
/// that was used along with the `af.Array`.
///
/// Note that if there are multiple `af.Array`s with the
/// same key, only the first one will be read.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn readArrayKey(
    allocator: std.mem.Allocator,
    filename: []const u8,
    key: []const u8,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_read_array_key(
            &arr,
            filename.ptr,
            key.ptr,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
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
pub inline fn readArrayKeyCheck(filename: []const u8, key: []const u8) !i32 {
    var idx: c_int = undefined;
    try af.AF_CHECK(
        af.af_read_array_key_check(
            &idx,
            filename.ptr,
            key.ptr,
        ),
        @src(),
    );
    return @intCast(idx);
}

/// Print the `af.Array` to a string instead of the screen.
///
/// Returns slice containing the resulting string.
///
/// N.B. The memory for output is allocated by the function.
/// The user is responsible for deleting the memory using
/// `af.freeHost`.
pub inline fn arrayToString(
    expr: []const u8,
    arr: *const af.Array,
    precision: i32,
    trans: bool,
) ![]const u8 {
    var output: [*c]u8 = undefined;
    try af.AF_CHECK(
        af.af_array_to_string(
            &output,
            expr.ptr,
            arr.array_,
            @intCast(precision),
            trans,
        ),
        @src(),
    );
    return std.mem.span(output);
}

/// Returns ptr to an `af.Array` initialized with user defined data.
pub inline fn createArray(
    allocator: std.mem.Allocator,
    data: ?*const anyopaque,
    ndims: u32,
    dims: af.Dim4,
    dtype: af.Dtype,
) !*af.Array {
    var res: af.af_array = undefined;
    switch (dtype) {
        .f32, .f64, .s32, .u32, .s64, .u64, .s16, .u16, .b8, .u8 => try af.AF_CHECK(
            af.af_create_array(
                &res,
                data,
                @intCast(ndims),
                &dims.dims,
                dtype.value(),
            ),
            @src(),
        ),
        else => {
            std.log.err(
                "createArray: can't construct ArrayFire array from given type.\n",
                .{},
            );
            return error.UnsupportedArrayFireType;
        },
    }
    return af.Array.init(allocator, res);
}

/// Returns ptr to an empty `af.Array`.
pub inline fn createHandle(
    allocator: std.mem.Allocator,
    ndims: u32,
    dims: af.Dim4,
    dtype: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_create_handle(
            &arr,
            @intCast(ndims),
            &dims.dims,
            dtype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Deep copy an `af.Array` to another.
pub inline fn copyArray(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_copy_array(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Copy data from a pointer (host/device) to an existing `af.Array`.
pub inline fn writeArray(
    arr: *af.Array,
    data: ?*const anyopaque,
    bytes: usize,
    src: af.Source,
) !void {
    try af.AF_CHECK(
        af.af_write_array(
            arr.array_,
            data,
            bytes,
            src.value(),
        ),
        @src(),
    );
}

/// Copy data from an `af.Array` to a pointer.
pub inline fn getDataPtr(data: ?*anyopaque, arr: *const af.Array) !void {
    try af.AF_CHECK(
        af.af_get_data_ptr(
            data,
            arr.array_,
        ),
        @src(),
    );
}

/// Reduce the reference count of the `af.Array`.
pub inline fn releaseArray(arr: *af.Array) !void {
    try af.AF_CHECK(af.af_release_array(arr.array_), @src());
}

/// Increments an `af.Array` reference count.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn retainArray(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_retain_array(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Get the reference count of an `af.Array`.
pub inline fn getDataRefCount(arr: *const af.Array) !i32 {
    var refCount: c_int = undefined;
    try af.AF_CHECK(af.af_get_data_ref_count(&refCount, arr.array_), @src());
    return @intCast(refCount);
}

/// Evaluate any expressions in the `af.Array`.
pub inline fn eval(in: *af.Array) !void {
    try af.AF_CHECK(af.af_eval(in.array_), @src());
}

/// Evaluate a slice of `af.Array`s together.
pub inline fn evalMultiple(arrays: []*af.Array) !void {
    try af.AF_CHECK(
        af.af_eval_multiple(
            @intCast(arrays.len),
            arrays.ptr,
        ),
        @src(),
    );
}

/// Returns the total number of elements across all dimensions
/// of the `af.Array`.
pub inline fn getElements(arr: *const af.Array) !i64 {
    var elements: af.dim_t = undefined;
    try af.AF_CHECK(
        af.af_get_elements(
            &elements,
            arr.array_,
        ),
        @src(),
    );
    return @intCast(elements);
}

/// Returns the type of an `af.Array`.
pub inline fn getType(arr: *const af.Array) !af.Dtype {
    var dtype: af.af_dtype = undefined;
    try af.AF_CHECK(af.af_get_type(&dtype, arr.array_), @src());
    return @enumFromInt(dtype);
}

/// Returns the dimensions of an `af.Array`.
pub inline fn getDims(arr: *const af.Array) !af.Dim4 {
    var dims: af.Dim4 = .{};
    try af.AF_CHECK(
        af.af_get_dims(
            &dims.dims[0],
            &dims.dims[1],
            &dims.dims[2],
            &dims.dims[3],
            arr.array_,
        ),
        @src(),
    );
    return dims;
}

/// Returns the number of dimensions of an `af.Array`.
pub inline fn getNumDims(arr: *const af.Array) !u32 {
    var numDims: c_uint = undefined;
    try af.AF_CHECK(af.af_get_numdims(&numDims, arr.array_), @src());
    return @intCast(numDims);
}

/// Returns bool indicating whether an `af.Array` is empty.
pub inline fn isEmpty(arr: *const af.Array) !bool {
    var empty: bool = undefined;
    try af.AF_CHECK(af.af_is_empty(&empty, arr.array_), @src());
    return empty;
}

/// Returns bool indicating whether an `af.Array` is scalar.
pub inline fn isScalar(arr: *const af.Array) !bool {
    var scalar: bool = undefined;
    try af.AF_CHECK(af.af_is_scalar(&scalar, arr.array_), @src());
    return scalar;
}

/// Returns bool indicating whether an `af.Array` is a row vector.
pub inline fn isRow(arr: *const af.Array) !bool {
    var row: bool = undefined;
    try af.AF_CHECK(af.af_is_row(&row, arr.array_), @src());
    return row;
}

/// Returns bool indicating whether an `af.Array` is a column vector.
pub inline fn isColumn(arr: *const af.Array) !bool {
    var col: bool = undefined;
    try af.AF_CHECK(af.af_is_column(&col, arr.array_), @src());
    return col;
}

/// Returns bool indicating whether an `af.Array` is a vector.
pub inline fn isVector(arr: *af.Array) !bool {
    var vec: bool = undefined;
    try af.AF_CHECK(af.af_is_vector(&vec, arr.array_), @src());
    return vec;
}

/// Returns bool indicating whether an `af.Array` is complex type.
pub inline fn isComplex(arr: *const af.Array) !bool {
    var complex: bool = undefined;
    try af.AF_CHECK(af.af_is_complex(&complex, arr.array_), @src());
    return complex;
}

/// Returns bool indicating whether an `af.Array` is a real type.
pub inline fn isReal(arr: *const af.Array) !bool {
    var is_real: bool = undefined;
    try af.AF_CHECK(af.af_is_real(&is_real, arr.array_), @src());
    return is_real;
}

/// Returns bool indicating whether an `af.Array` is double
/// precision type.
pub inline fn isDouble(arr: *const af.Array) !bool {
    var is_double: bool = undefined;
    try af.AF_CHECK(af.af_is_double(&is_double, arr.array_), @src());
    return is_double;
}

/// Returns bool indicating whether an `af.Array` is single
/// precision type.
pub inline fn isSingle(arr: *const af.Array) !bool {
    var is_single: bool = undefined;
    try af.AF_CHECK(af.af_is_single(&is_single, arr.array_), @src());
    return is_single;
}

/// Returns bool indicating whether an `af.Array` is a 16 bit
/// floating point type.
pub inline fn isHalf(arr: *const af.Array) !bool {
    var is_half: bool = undefined;
    try af.AF_CHECK(af.af_is_half(&is_half, arr.array_), @src());
    return is_half;
}

/// Returns bool indicating whether an `af.Array` is a real
/// floating point type.
pub inline fn isRealFloating(arr: *const af.Array) !bool {
    var is_real_floating: bool = undefined;
    try af.AF_CHECK(af.af_is_realfloating(&is_real_floating, arr.array_), @src());
    return is_real_floating;
}

/// Returns bool indicating whether an `af.Array` is floating
/// precision type.
pub inline fn isFloating(arr: *const af.Array) !bool {
    var is_floating: bool = undefined;
    try af.AF_CHECK(af.af_is_floating(&is_floating, arr.array_), @src());
    return is_floating;
}

/// Returns bool indicating whether an `af.Array` is integer type.
pub inline fn isInteger(arr: *const af.Array) !bool {
    var is_integer: bool = undefined;
    try af.AF_CHECK(af.af_is_integer(&is_integer, arr.array_), @src());
    return is_integer;
}

/// Returns bool indicating whether an `af.Array` is bool type.
pub inline fn isBool(arr: *const af.Array) !bool {
    var is_bool: bool = undefined;
    try af.AF_CHECK(af.af_is_bool(&is_bool, arr.array_), @src());
    return is_bool;
}

/// Returns bool indicating whether an `af.Array` is sparse.
pub inline fn isSparse(arr: *const af.Array) !bool {
    var is_sparse: bool = undefined;
    try af.AF_CHECK(af.af_is_sparse(&is_sparse, arr.array_), @src());
    return is_sparse;
}

/// Get first element from an `af.Array`.
pub inline fn getScalar(comptime T: type, arr: *const af.Array) !T {
    var res: T = undefined;
    try af.AF_CHECK(af.af_get_scalar(&res, arr.array_), @src());
    return res;
}

/// Get's the backend enum for an `af.Array`.
///
/// This will return one of the values from the `af.Backend` enum.
/// The return value specifies which backend the `af.Array` was created on.
pub inline fn getBackendId(arr: *const af.Array) !af.Backend {
    var backend: af.af_backend = undefined;
    try af.AF_CHECK(af.af_get_backend_id(&backend, arr.array_), @src());
    return @enumFromInt(backend);
}

/// Returns the id of the device an `af.Array` was created on.
pub inline fn getDeviceId(arr: *const af.Array) !i32 {
    var device: c_int = undefined;
    try af.AF_CHECK(af.af_get_device_id(&device, arr.array_), @src());
    return @intCast(device);
}

// TODO: gemm should optionally allocate based on the `C` param

/// BLAS general matrix multiply (GEMM) of two `af.Array` objects.
///
/// This provides a general interface to the BLAS level 3 general
/// matrix multiply (GEMM), which is generally defined as:
///
/// C=α∗opA(A)opB(B)+β∗C
///
/// Where α (alpha) and β (beta) are both scalars; A and B are the matrix
/// multiply operands; and opA and opB are noop (if `af.MatProp.None`) or transpose
/// (if `af.MatProp.Trans`) operations on A or B before the actual GEMM operation.
/// Batched GEMM is supported if at least either A or B have more than two dimensions
/// (see `matmul` for more details on broadcasting). However, only one alpha and one
/// beta can be used for all of the batched matrix operands.
///
/// The `af.Array` that out points to can be used both as an input and output.
/// An allocation will be performed if you pass a null `af.Array` handle (i.e. af_array c = 0;).
/// If a valid `af.Array` is passed as C, the operation will be performed on that `af.Array` itself.
/// The C af_array must be the correct type and shape; otherwise, an error will be thrown.
pub inline fn gemm(
    comptime T: type,
    C: *af.Array,
    opA: af.MatProp,
    opB: af.MatProp,
    alpha: T,
    A: *const af.Array,
    B: *const af.Array,
    beta: T,
) !void {
    try af.AF_CHECK(
        af.af_gemm(
            C,
            opA.value(),
            opB.value(),
            &alpha,
            A.array_,
            B.array_,
            &beta,
        ),
        @src(),
    );
}

/// Performs a matrix multiplication on two `af.Array`s (lhs, rhs).
///
/// N.B. The following applies for Sparse-Dense matrix multiplication.
/// This function can be used with one sparse input. The sparse input
/// must always be the lhs and the dense matrix must be rhs. The sparse
/// array can only be of `af.Storage.CSR` format. The returned array is
/// always dense. optLhs can only be one of `af.MatProp.None`, `af.MatProp.Trans`,
/// `af.MatProp.CTrans`. optRhs can only be `af.MatProp.None`.
pub inline fn matmul(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    optLhs: af.MatProp,
    optRhs: af.MatProp,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_matmul(
            &arr,
            lhs.array_,
            rhs.array_,
            optLhs.value(),
            optRhs.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Scalar dot product between two vectors.
///
/// Also referred to as the inner product.
pub inline fn dot(
    allocator: std.mem.Allocator,
    lhs: *const af.Array,
    rhs: *const af.Array,
    optLhs: af.MatProp,
    optRhs: af.MatProp,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_dot(
            &arr,
            lhs.array_,
            rhs.array_,
            optLhs.value(),
            optRhs.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Scalar dot product between two vectors.
///
/// Also referred to as the inner product. Returns the result
/// as a host scalar.
pub inline fn dotAll(
    lhs: *const af.Array,
    rhs: *const af.Array,
    optLhs: af.MatProp,
    optRhs: af.MatProp,
) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_dot_all(
            &res.real,
            &res.imag,
            lhs.array_,
            rhs.array_,
            optLhs.value(),
            optRhs.value(),
        ),
        @src(),
    );
    return res;
}

/// Transpose a matrix.
pub inline fn transpose(
    allocator: std.mem.Allocator,
    in: *af.Array,
    conjugate: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_transpose(
            &arr,
            in.array_,
            conjugate,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Tranpose a matrix in-place.
pub inline fn transposeInplace(in: *af.Array, conjugate: bool) !void {
    try af.AF_CHECK(af.af_transpose_inplace(in.array_, conjugate), @src());
}

/// Create an `af.Array` from a scalar input value.
///
/// The `af.Array` created has the same value at all locations.
pub inline fn constant(
    allocator: std.mem.Allocator,
    val: f64,
    ndims: u32,
    dims: af.Dim4,
    dtype: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_constant(
            &arr,
            val,
            @intCast(ndims),
            &dims.dims,
            dtype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Create a complex type `af.Array` from a scalar input value.
///
/// The `af.Array` created has the same value at all locations.
pub inline fn constantComplex(
    allocator: std.mem.Allocator,
    real_: f64,
    imag_: f64,
    ndims: u32,
    dims: af.Dim4,
    dtype: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_constant_complex(
            &arr,
            real_,
            imag_,
            @intCast(ndims),
            &dims.dims,
            dtype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Create an `af.Array` of type s64 from a scalar input value.
///
/// The `af.Array` created has the same value at all locations.
pub inline fn constantI64(
    allocator: std.mem.Allocator,
    val: i64,
    ndims: u32,
    dims: af.Dim4,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_constant_long(
            &arr,
            @intCast(val),
            @intCast(ndims),
            &dims.dims,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Create an `af.Array` of type u64 from a scalar input value.
///
/// The `af.Array` created has the same value at all locations.
pub inline fn constantU64(
    allocator: std.mem.Allocator,
    val: u64,
    ndims: u32,
    dims: af.Dim4,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_constant_ulong(
            &arr,
            @intCast(val),
            @intCast(ndims),
            &dims.dims,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Creates an `af.Array` with [0, n-1] values along the `seq_dim` dimension
/// and tiled across other dimensions specified by an array of `ndims` dimensions.
pub inline fn range(
    allocator: std.mem.Allocator,
    ndims: u32,
    dims: af.Dim4,
    seq_dim: i32,
    dtype: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_range(
            &arr,
            @intCast(ndims),
            &dims.dims,
            @intCast(seq_dim),
            dtype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Create an sequence [0, dims.elements() - 1] and modify to
/// specified dimensions dims and then tile it according to `tdims`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn iota(
    allocator: std.mem.Allocator,
    ndims: u32,
    dims: af.Dim4,
    t_ndims: u32,
    tdims: af.Dim4,
    dtype: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_iota(
            &arr,
            @intCast(ndims),
            &dims.dims,
            @intCast(t_ndims),
            &tdims.dims,
            dtype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns ptr to an identity `af.Array` with diagonal values 1.
pub inline fn identity(
    allocator: std.mem.Allocator,
    ndims: u32,
    dims: af.Dim4,
    dtype: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_identity(
            &arr,
            @intCast(ndims),
            &dims.dims,
            dtype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Create a diagonal matrix from input `af.Array`.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn diagCreate(allocator: std.mem.Allocator, in: *const af.Array, num: i32) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_diag_create(
            &arr,
            in.array_,
            @intCast(num),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Extract diagonal from a matrix.
///
/// Returns ptr to the resulting `af.Array`.
pub inline fn diagExtract(allocator: std.mem.Allocator, in: *const af.Array, num: i32) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_diag_extract(
            &arr,
            in.array_,
            @intCast(num),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Joins 2 `af.Array`s along `dim`.
pub inline fn join(
    allocator: std.mem.Allocator,
    dim: i32,
    first: *const af.Array,
    second: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_join(
            &arr,
            @intCast(dim),
            first.array_,
            second.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Join many `af.Array`s along dim.
///
/// Current limit is set to 10 `af.Array`s.
pub inline fn joinMany(
    allocator: std.mem.Allocator,
    dim: i32,
    inputs: []const *af.Array,
) !*af.Array {
    var in_arrays = try allocator.alloc(af.af_array, inputs.len);
    defer allocator.free(in_arrays);
    for (inputs, 0..) |array, i| in_arrays[i] = array.array_;
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_join_many(
            &arr,
            @intCast(dim),
            @intCast(inputs.len),
            in_arrays.ptr,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Repeat the contents of the input `af.Array` along the
/// specified dimensions.
pub inline fn tile(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    x: u32,
    y: u32,
    z: u32,
    w: u32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_tile(
            &arr,
            in.array_,
            @intCast(x),
            @intCast(y),
            @intCast(z),
            @intCast(w),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Reorder an `af.Array` according to the specified dimensions.
pub inline fn reorder(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    x: u32,
    y: u32,
    z: u32,
    w: u32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_reorder(
            &arr,
            in.array_,
            @intCast(x),
            @intCast(y),
            @intCast(z),
            @intCast(w),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Circular shift along specified dimensions.
pub inline fn shift(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    x: i32,
    y: i32,
    z: i32,
    w: i32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_shift(
            &arr,
            in.array_,
            @intCast(x),
            @intCast(y),
            @intCast(z),
            @intCast(w),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Modifies the dimensions of an input `af.Array` to the shape specified
/// by an array of ndims dimensions.
pub inline fn modDims(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    ndims: u32,
    dims: af.Dim4,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_moddims(
            &arr,
            in.array_,
            @intCast(ndims),
            &dims.dims,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Flatten the input to a single dimension.
pub inline fn flat(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_flat(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Flip the input along specified dimension.
///
/// Mirrors the `af.Array` along the specified dimensions.
pub inline fn flip(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: u32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_flip(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Create a lower triangular matrix from input `af.Array`.
pub inline fn lower(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    is_unit_diag: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_lower(
            &arr,
            in.array_,
            is_unit_diag,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Create an upper triangular matrix from input `af.Array`.
pub inline fn upper(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    is_unit_diag: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_upper(
            &arr,
            in.array_,
            is_unit_diag,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Selects elements from two `af.Array`s based on the values of a
/// binary conditional `af.Array`.
///
/// Creates a new `af.Array` that is composed of values either from
/// `af.Array` a or `af.Array` b, based on a third conditional
/// `af.Array`. For all non-zero elements in the conditional `af.Array`,
/// the output `af.Array` will contain values from a. Otherwise the output
/// will contain values from b.
pub inline fn select(
    allocator: std.mem.Allocator,
    cond: *const af.Array,
    a: *const af.Array,
    b: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_select(
            &arr,
            cond.array_,
            a.array_,
            b.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns ptr to an `af.Array` containing elements of a
/// when cond is true else scalar value b.
pub inline fn selectScalarR(
    allocator: std.mem.Allocator,
    cond: *const af.Array,
    a: *const af.Array,
    b: f64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_select_scalar_r(
            &arr,
            cond.array_,
            a.array_,
            b,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns ptr to an `af.Array` containing elements of b
/// when cond is false else scalar value a.
pub inline fn selectScalarL(
    allocator: std.mem.Allocator,
    cond: *const af.Array,
    a: f64,
    b: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_select_scalar_r(
            &arr,
            cond.array_,
            a,
            b.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Replace elements of an `af.Array` based on a conditional `af.Array`.
///
/// - Input values are retained when corresponding elements from condition array are true.
/// - Input values are replaced when corresponding elements from condition array are false.
pub inline fn replace(
    a: *af.Array,
    cond: *const af.Array,
    b: *const af.Array,
) !void {
    try af.AF_CHECK(af.af_replace(a.array_, cond.array_, b.array_), @src());
}

/// Replace elements of an `af.Array` based on a conditional `af.Array`.
///
/// N.B. Values of a are replaced with corresponding values of b, when cond is false.
pub inline fn replaceScalar(a: *af.Array, cond: *const af.Array, b: f64) !void {
    try af.AF_CHECK(af.af_replace_scalar(a.array_, cond.array_, b), @src());
}

/// Pad an `af.Array`.
///
/// Pad the input `af.Array` using a constant or values from input along border
pub inline fn pad(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    begin_ndims: u32,
    begin_dims: af.Dim4,
    end_ndims: u32,
    end_dims: af.Dim4,
    pad_fill_type: af.BorderType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_pad(
            &arr,
            in.array_,
            @intCast(begin_ndims),
            &begin_dims.dims,
            @intCast(end_ndims),
            &end_dims.dims,
            pad_fill_type.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Calculate the gradients of the input `af.Array`.
///
/// The gradients along the first and second dimensions
/// are calculated simultaneously.
///
/// Return struct containing the following fields:
/// - `dx`: `af.Array` containing the gradient along the
/// 1st dimension of the input `af.Array`.
/// - `dy`: `af.Array` containing the gradient along the
/// 2nd dimension of the input `af.Array`.
pub inline fn gradient(allocator: std.mem.Allocator, in: *const af.Array) !struct {
    dx: *af.Array,
    dy: *af.Array,
} {
    var dx: af.af_array = undefined;
    var dy: af.af_array = undefined;
    try af.AF_CHECK(af.af_gradient(&dx, &dy, in.array_), @src());
    return .{
        .dx = try af.Array.init(allocator, dx),
        .dy = try af.Array.init(allocator, dy),
    };
}

/// Computes the singular value decomposition of a matrix.
///
/// This function factorizes a matrix A into two unitary matrices,
/// U and VT, and a diagonal matrix S, such that A=USVT. If A has
/// M rows and N columns ( M×N), then U will be M×M, V will be N×N,
/// and S will be M×N. However, for S, this function only returns the
/// non-zero diagonal elements as a sorted (in descending order) 1D `af.Array`.
pub inline fn svd(allocator: std.mem.Allocator, in: *const af.Array) !struct {
    u: *af.Array,
    s: *af.Array,
    vt: *af.Array,
} {
    var u: af.af_array = undefined;
    var s: af.af_array = undefined;
    var vt: af.af_array = undefined;
    try af.AF_CHECK(af.af_svd(&u, &s, &vt, in.array_), @src());
    return .{
        .u = try af.Array.init(allocator, u),
        .s = try af.Array.init(allocator, s),
        .vt = try af.Array.init(allocator, vt),
    };
}

/// Computes the singular value decomposition of a matrix in-place.
pub inline fn svdInplace(allocator: std.mem.Allocator, in: *af.Array) !struct {
    u: *af.Array,
    s: *af.Array,
    vt: *af.Array,
} {
    var u: af.af_array = undefined;
    var s: af.af_array = undefined;
    var vt: af.af_array = undefined;
    try af.AF_CHECK(af.af_svd_inplace(&u, &s, &vt, in.array_), @src());
    return .{
        .u = try af.Array.init(allocator, u),
        .s = try af.Array.init(allocator, s),
        .vt = try af.Array.init(allocator, vt),
    };
}

/// Perform LU decomposition.
///
/// This function decomposes input matrix A into a
/// lower triangle L and upper triangle U.
pub inline fn lu(allocator: std.mem.Allocator, in: *const af.Array) !struct {
    lower: *af.Array,
    upper: *af.Array,
} {
    var lower_: af.af_array = undefined;
    var upper_: af.af_array = undefined;
    try af.AF_CHECK(af.af_lu(&lower_, &upper_, in.array_), @src());
    return .{
        .lower = try af.Array.init(allocator, lower_),
        .upper = try af.Array.init(allocator, upper_),
    };
}

/// In-place LU decomposition.
pub inline fn luInplace(
    allocator: std.mem.Allocator,
    in: *af.Array,
    is_lapack_piv: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_lu_inplace(
            &arr,
            in.array_,
            is_lapack_piv,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Perform QR decomposition.
///
/// This function decomposes input matrix A into an orthogonal
/// matrix Q and an upper triangular matrix R.
pub inline fn qr(allocator: std.mem.Allocator, in: *const af.Array) !struct {
    q: *af.Array,
    r: *af.Array,
    tau: *af.Array,
} {
    var q: af.af_array = undefined;
    var r: af.af_array = undefined;
    var tau: af.af_array = undefined;
    try af.AF_CHECK(af.af_qr(&q, &r, &tau, in.array_), @src());
    return .{
        .q = try af.Array.init(allocator, q),
        .r = try af.Array.init(allocator, r),
        .tau = try af.Array.init(allocator, tau),
    };
}

/// In-place QR decomposition.
pub inline fn qrInplace(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_qr_inplace(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Perform Cholesky decomposition.
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
pub inline fn cholesky(allocator: std.mem.Allocator, in: *const af.Array, is_upper: bool) !struct {
    out: *af.Array,
    info: i32,
} {
    var out: af.af_array = undefined;
    var info: c_int = undefined;
    try af.AF_CHECK(
        af.af_cholesky(
            &out,
            &info,
            in.array_,
            is_upper,
        ),
        @src(),
    );
    return .{
        .out = try af.Array.init(allocator, out),
        .info = @intCast(info),
    };
}

/// In-place Cholesky decomposition.
///
/// Returns 0 if cholesky decomposition passes, if not
/// it returns the rank at which the decomposition failed.
pub inline fn choleskyInplace(in: *af.Array, is_upper: bool) !i32 {
    var info: c_int = undefined;
    try af.AF_CHECK(
        af.af_cholesky_inplace(
            &info,
            in.array_,
            is_upper,
        ),
        @src(),
    );
    return @intCast(info);
}

/// Solve a system of equations.
///
/// This function takes a co-efficient matrix A and an output
/// matrix B as inputs to solve the following equation for X.
pub inline fn solve(
    allocator: std.mem.Allocator,
    a: *const af.Array,
    b: *const af.Array,
    options: af.MatProp,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_solve(
            &arr,
            a.array_,
            b.array_,
            options.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Solve a system of equations.
///
/// This function takes a co-efficient matrix A and an output
/// matrix B as inputs to solve the following equation for X.
pub inline fn solveLU(
    allocator: std.mem.Allocator,
    a: *const af.Array,
    piv: *const af.Array,
    b: *const af.Array,
    options: af.MatProp,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_solve_lu(
            &arr,
            a.array_,
            piv.array_,
            b.array_,
            options.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}
/// Invert a matrix.
///
/// This function inverts a square matrix A.
pub inline fn inverse(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    options: af.MatProp,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_inverse(
            &arr,
            in.array_,
            options.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Pseudo-invert a matrix.
///
/// This function calculates the Moore-Penrose pseudoinverse
/// of a matrix A, using `svd` at its core. If A is of size M×N,
/// then its pseudoinverse A+ will be of size N×M.
///
/// This calculation can be batched if the input array is three or
/// four-dimensional (M×N×P×Q, with Q=1 for only three dimensions).
/// Each M×N slice along the third dimension will have its own pseudoinverse,
/// for a total of P×Q pseudoinverses in the output array (N×M×P×Q).
pub inline fn pInverse(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    tol: f64,
    options: af.MatProp,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_pinverse(
            &arr,
            in.array_,
            tol,
            options.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the rank of the input matrix.
///
/// This function uses `qr` to find the rank of the input
/// matrix within the given tolerance.
pub inline fn rank(in: *const af.Array, tol: f64) !u32 {
    var res: c_uint = undefined;
    try af.AF_CHECK(
        af.af_rank(
            &res,
            in.array_,
            tol,
        ),
        @src(),
    );
    return @intCast(res);
}

/// Returns the determinant of the input matrix.
pub inline fn det(in: *const af.Array) !struct { det_real: f64, det_imag: f64 } {
    var det_real: f64 = undefined;
    var det_imag: f64 = undefined;
    try af.AF_CHECK(
        af.af_det(
            &det_real,
            &det_imag,
            in.array_,
        ),
        @src(),
    );
    return .{ .det_real = det_real, .det_imag = det_imag };
}

/// Returns the norm of the input matrix.
///
/// This function can return the norm using various metrics based
/// on the type paramter.
pub inline fn norm(in: *const af.Array, norm_type: af.NormType, p: f64, q: f64) !f64 {
    var res: f64 = undefined;
    try af.AF_CHECK(
        af.af_norm(
            &res,
            in.array_,
            norm_type.value(),
            p,
            q,
        ),
        @src(),
    );
    return res;
}

/// Calculates backward pass gradient of 2D convolution.
///
/// This function calculates the gradient with respect to the output
/// of the `convolve2NN` function that uses the machine learning formulation
/// for the dimensions of the signals and filters.
pub inline fn convolve2GradientNN(
    allocator: std.mem.Allocator,
    incoming_gradient: *const af.Array,
    original_signal: *const af.Array,
    original_filter: *const af.Array,
    convolved_output: *const af.Array,
    stride_dims: u32,
    strides: af.Dim4,
    padding_dims: u32,
    paddings: af.Dim4,
    dilation_dims: u32,
    dilations: af.Dim4,
    grad_type: af.ConvGradientType,
) !*af.Array {
    var res: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_convolve2_gradient_nn(
            &res,
            incoming_gradient.array_,
            original_signal.array_,
            original_filter.array_,
            convolved_output.array_,
            @intCast(stride_dims),
            &strides.dims,
            @intCast(padding_dims),
            &paddings.dims,
            @intCast(dilation_dims),
            &dilations.dims,
            grad_type.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, res);
}

/// Returns pointer to an `af.Array` of uniform numbers
/// using a random engine.
pub inline fn randomUniform(
    allocator: std.mem.Allocator,
    ndims: u32,
    dims: af.Dim4,
    dtype: af.Dtype,
    engine: *af.RandomEngine,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_random_uniform(
            &arr,
            @intCast(ndims),
            &dims.dims,
            dtype.value(),
            engine.engine_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns pointer to an `af.Array` of normal numbers
/// using a random engine.
pub inline fn randomNormal(
    allocator: std.mem.Allocator,
    ndims: u32,
    dims: af.Dim4,
    dtype: af.Dtype,
    engine: *af.RandomEngine,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_random_normal(
            &arr,
            @intCast(ndims),
            &dims.dims,
            dtype.value(),
            engine.engine_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns an `af.Array` of uniform numbers using a random engine.
pub inline fn randu(
    allocator: std.mem.Allocator,
    ndims: u32,
    dims: af.Dim4,
    dtype: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_randu(
            &arr,
            @intCast(ndims),
            &dims.dims,
            dtype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns an `af.Array` of normal numbers using a random engine.
pub inline fn randn(
    allocator: std.mem.Allocator,
    ndims: u32,
    dims: af.Dim4,
    dtype: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_randn(
            &arr,
            @intCast(ndims),
            &dims.dims,
            dtype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Signals interpolation on one dimensional signals.
/// Returns the result as a new `af.Array`.
pub inline fn approx1(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    pos: *const af.Array,
    method: af.InterpType,
    off_grid: f32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_approx1(
            &arr,
            in.array_,
            pos.array_,
            method.value(),
            off_grid,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Signals interpolation on one dimensional signals; accepts
/// a pre-allocated array, `out`, where results are written.
pub inline fn approx1V2(
    out: *af.Array,
    in: *const af.Array,
    pos: *const af.Array,
    method: af.InterpType,
    off_grid: f32,
) !void {
    try af.AF_CHECK(
        af.af_approx1_v2(
            &out.array_,
            in.array_,
            pos.array_,
            method.value(),
            off_grid,
        ),
        @src(),
    );
}

/// Signals interpolation on two dimensional signals.
/// Returns the result as a new `af.Array`.
pub inline fn approx2(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    pos0: *const af.Array,
    pos1: *const af.Array,
    method: af.InterpType,
    off_grid: f32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_approx2(
            &arr,
            in.array_,
            pos0.array_,
            pos1.array_,
            method.value(),
            off_grid,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Signals interpolation on two dimensional signals; accepts
/// a pre-allocated array, `out`, where results are written.
pub inline fn approx2V2(
    out: *af.Array,
    in: *const af.Array,
    pos0: *const af.Array,
    pos1: *const af.Array,
    method: af.InterpType,
    off_grid: f32,
) !void {
    try af.AF_CHECK(
        af.af_approx2_v2(
            &out.array_,
            in.array_,
            pos0.array_,
            pos1.array_,
            method.value(),
            off_grid,
        ),
        @src(),
    );
}

/// Signals interpolation on one dimensional signals along specified dimension.
///
/// `approx1Uniform` accepts the dimension to perform the interpolation along
/// the input. It also accepts start and step values which define the uniform
/// range of corresponding indices.
pub inline fn approx1Uniform(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    pos: *const af.Array,
    interp_dim: i32,
    idx_start: f64,
    idx_step: f64,
    method: af.InterpType,
    off_grid: f32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_approx1_uniform(
            &arr,
            in.array_,
            pos.array_,
            @intCast(interp_dim),
            idx_start,
            idx_step,
            method.value(),
            off_grid,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Signals interpolation on one dimensional signals along specified dimension;
/// accepts a pre-allocated array, `out`, where results are written.
///
/// `approx1Uniform` accepts the dimension to perform the interpolation along
/// the input. It also accepts start and step values which define the uniform
/// range of corresponding indices.
pub inline fn approx1UniformV2(
    out: *af.Array,
    in: *const af.Array,
    pos: *const af.Array,
    interp_dim: i32,
    idx_start: f64,
    idx_step: f64,
    method: af.InterpType,
    off_grid: f32,
) !void {
    try af.AF_CHECK(
        af.af_approx1_uniform_v2(
            &out.array_,
            in.array_,
            pos.array_,
            @intCast(interp_dim),
            idx_start,
            idx_step,
            method.value(),
            off_grid,
        ),
        @src(),
    );
}

/// Signals interpolation on two dimensional signals
/// along specified dimensions.
///
/// `approx2Uniform` accepts two dimensions to perform the interpolation
/// along the input. It also accepts start and step values which define
/// the uniform range of corresponding indices.
pub inline fn approx2Uniform(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    pos0: *const af.Array,
    interp_dim0: i32,
    idx_start_dim0: f64,
    idx_step_dim0: f64,
    pos1: *const af.Array,
    interp_dim1: i32,
    idx_start_dim1: f64,
    idx_step_dim1: f64,
    method: af.InterpType,
    off_grid: f32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_approx2_uniform(
        &arr,
        in.array_,
        pos0.array_,
        @intCast(interp_dim0),
        idx_start_dim0,
        idx_step_dim0,
        pos1.array_,
        @intCast(interp_dim1),
        idx_start_dim1,
        idx_step_dim1,
        method.value(),
        off_grid,
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Signals interpolation on two dimensional signals
/// along specified dimensions; accepts a pre-allocated array, `out`,
/// where results are written.
///
/// `approx2UniformV2` accepts two dimensions to perform the interpolation
/// along the input. It also accepts start and step values which define
/// the uniform range of corresponding indices.
pub inline fn approx2UniformV2(
    out: *af.Array,
    in: *const af.Array,
    pos0: *const af.Array,
    interp_dim0: i32,
    idx_start_dim0: f64,
    idx_step_dim0: f64,
    pos1: *const af.Array,
    interp_dim1: i32,
    idx_start_dim1: f64,
    idx_step_dim1: f64,
    method: af.InterpType,
    off_grid: f32,
) !void {
    try af.AF_CHECK(
        af.af_approx2_uniform_v2(
            &out.array_,
            in.array_,
            pos0.array_,
            @intCast(interp_dim0),
            idx_start_dim0,
            idx_step_dim0,
            pos1.array_,
            @intCast(interp_dim1),
            idx_start_dim1,
            idx_step_dim1,
            method.value(),
            off_grid,
        ),
        @src(),
    );
}

/// Fast fourier transform on one dimensional signals.
pub inline fn fft(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    odim0: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft(
            &arr,
            in.array_,
            norm_factor,
            @intCast(odim0),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// In-place fast fourier transform on one dimensional signals.
pub inline fn fftInplace(in: *af.Array, norm_factor: f64) !void {
    try af.AF_CHECK(
        af.af_fft_inplace(
            in.array_,
            norm_factor,
        ),
        @src(),
    );
}

/// Fast fourier transform on two dimensional signals.
pub inline fn fft2(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    odim0: i64,
    odim1: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft2(
            &arr,
            in.array_,
            norm_factor,
            @intCast(odim0),
            @intCast(odim1),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// In-place fast fourier transform on two dimensional signals.
pub inline fn fft2Inplace(in: *af.Array, norm_factor: f64) !void {
    try af.AF_CHECK(
        af.af_fft2_inplace(
            in.array_,
            norm_factor,
        ),
        @src(),
    );
}

/// Fast fourier transform on three dimensional signals.
pub inline fn fft3(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    odim0: i64,
    odim1: i64,
    odim2: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft3(
            &arr,
            in.array_,
            norm_factor,
            @intCast(odim0),
            @intCast(odim1),
            @intCast(odim2),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// In-place fast fourier transform on three dimensional signals.
pub inline fn fft3Inplace(in: *af.Array, norm_factor: f64) !void {
    try af.AF_CHECK(
        af.af_fft3_inplace(
            in.array_,
            norm_factor,
        ),
        @src(),
    );
}

/// Inverse fast fourier transform on one dimensional signals.
pub inline fn ifft(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    odim0: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_ifft(
            &arr,
            in.array_,
            norm_factor,
            @intCast(odim0),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// In-place inverse fast fourier transform on one dimensional signals.
pub inline fn ifftInplace(in: *af.Array, norm_factor: f64) !void {
    try af.AF_CHECK(
        af.af_ifft_inplace(
            in.array_,
            norm_factor,
        ),
        @src(),
    );
}

/// Inverse fast fourier transform on two dimensional signals.
pub inline fn ifft2(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    odim0: i64,
    odim1: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_ifft2(
            &arr,
            in.array_,
            norm_factor,
            @intCast(odim0),
            @intCast(odim1),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// In-place inverse fast fourier transform on two dimensional signals.
pub inline fn ifft2Inplace(in: *af.Array, norm_factor: f64) !void {
    try af.AF_CHECK(
        af.af_ifft2_inplace(
            in.array_,
            norm_factor,
        ),
        @src(),
    );
}

/// Inverse fast fourier transform on three dimensional signals.
pub inline fn ifft3(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    odim0: i64,
    odim1: i64,
    odim2: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_ifft3(
            &arr,
            in.array_,
            norm_factor,
            @intCast(odim0),
            @intCast(odim1),
            @intCast(odim2),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// In-place inverse fast fourier transform on three dimensional signals.
pub inline fn ifft3Inplace(in: *af.Array, norm_factor: f64) !void {
    try af.AF_CHECK(
        af.af_ifft3_inplace(
            in.array_,
            norm_factor,
        ),
        @src(),
    );
}

/// Real to complex fast fourier transform for one dimensional signals.
pub inline fn fftR2C(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    pad0: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft_r2c(
            &arr,
            in.array_,
            norm_factor,
            @intCast(pad0),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Real to complex fast fourier transform for two dimensional signals.
pub inline fn fft2R2C(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    pad0: i64,
    pad1: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft2_r2c(
            &arr,
            in.array_,
            norm_factor,
            @intCast(pad0),
            @intCast(pad1),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Real to complex fast fourier transform for three dimensional signals.
pub inline fn fft3R2C(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    pad0: i64,
    pad1: i64,
    pad2: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft3_r2c(
            &arr,
            in.array_,
            norm_factor,
            @intCast(pad0),
            @intCast(pad1),
            @intCast(pad2),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Complex to real fast fourier transform for one dimensional signals.
pub inline fn fftC2R(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    is_odd: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft_c2r(
            &arr,
            in.array_,
            norm_factor,
            is_odd,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Complex to real fast fourier transform for two dimensional signals.
pub inline fn fft2C2R(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    is_odd: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft2_c2r(
            &arr,
            in.array_,
            norm_factor,
            is_odd,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Complex to real fast fourier transform for three dimensional signals.
pub inline fn fft3C2R(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    norm_factor: f64,
    is_odd: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft2_c2r(
            &arr,
            in.array_,
            norm_factor,
            is_odd,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Convolution on one dimensional signals.
pub inline fn convolve1(
    allocator: std.mem.Allocator,
    signal: *const af.Array,
    filter: *const af.Array,
    mode: af.ConvMode,
    domain: af.ConvDomain,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_convolve1(
            &arr,
            signal.array_,
            filter.array_,
            mode.value(),
            domain.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Convolution on two dimensional signals.
pub inline fn convolve2(
    allocator: std.mem.Allocator,
    signal: *const af.Array,
    filter: *const af.Array,
    mode: af.ConvMode,
    domain: af.ConvDomain,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_convolve2(
            &arr,
            signal.array_,
            filter.array_,
            mode.value(),
            domain.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
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
pub inline fn convolve2NN(
    allocator: std.mem.Allocator,
    signal: *const af.Array,
    filter: *const af.Array,
    stride_dims: u32,
    strides: af.Dim4,
    padding_dims: u32,
    paddings: af.Dim4,
    dilation_dims: u32,
    dilations: af.Dim4,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_convolve2_nn(
            &arr,
            signal.array_,
            filter.array_,
            @intCast(stride_dims),
            &strides.dims,
            @intCast(padding_dims),
            &paddings.dims,
            @intCast(dilation_dims),
            &dilations.dims,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Convolution on three dimensional signals.
pub inline fn convolve3(
    allocator: std.mem.Allocator,
    signal: *const af.Array,
    filter: *const af.Array,
    mode: af.ConvMode,
    domain: af.ConvDomain,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_convolve3(
            &arr,
            signal.array_,
            filter.array_,
            mode.value(),
            domain.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Separable convolution on two dimensional signals.
pub inline fn convolve2Sep(
    allocator: std.mem.Allocator,
    col_filter: *const af.Array,
    row_filter: *const af.Array,
    signal: *const af.Array,
    mode: af.ConvMode,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_convolve2_sep(
            &arr,
            col_filter.array_,
            row_filter.array_,
            signal.array_,
            mode.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Convolution on 1D signals using FFT.
pub inline fn fftConvolve1(
    allocator: std.mem.Allocator,
    signal: *const af.Array,
    filter: *const af.Array,
    mode: af.ConvMode,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft_convolve1(
            &arr,
            signal.array_,
            filter.array_,
            mode.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Convolution on 2D signals using FFT.
pub inline fn fftConvolve2(
    allocator: std.mem.Allocator,
    signal: *const af.Array,
    filter: *const af.Array,
    mode: af.ConvMode,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft_convolve2(
            &arr,
            signal.array_,
            filter.array_,
            mode.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Convolution on 3D signals using FFT.
pub inline fn fftConvolve3(
    allocator: std.mem.Allocator,
    signal: *const af.Array,
    filter: *const af.Array,
    mode: af.ConvMode,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fft_convolve3(
            &arr,
            signal.array_,
            filter.array_,
            mode.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Finite impulse response filter.
pub inline fn fir(
    allocator: std.mem.Allocator,
    b: *const af.Array,
    x: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_fir(
            &arr,
            b.array_,
            x.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Infinite impulse response filter.
pub inline fn iir(
    allocator: std.mem.Allocator,
    b: *const af.Array,
    a: *const af.Array,
    x: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_iir(
            &arr,
            b.array_,
            a.array_,
            x.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Median filter.
pub inline fn medfilt(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    wind_length: i64,
    wind_width: i64,
    edge_pad: af.BorderType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_medFilt(
            &arr,
            in.array_,
            @intCast(wind_length),
            @intCast(wind_width),
            edge_pad.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// 1D median filter.
pub inline fn medFilt1(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    wind_width: i64,
    edge_pad: af.BorderType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_medfilt1(
            &arr,
            in.array_,
            @intCast(wind_width),
            edge_pad.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// 2D median filter.
pub inline fn medFilt2(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    wind_length: i64,
    wind_width: i64,
    edge_pad: af.BorderType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_medfilt2(
            &arr,
            in.array_,
            @intCast(wind_length),
            @intCast(wind_width),
            edge_pad.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Converts `af.Array` of values, row indices and column indices into a sparse array.
pub inline fn createSparseArray(
    allocator: std.mem.Allocator,
    nRows: i64,
    nCols: i64,
    values: *const af.Array,
    rowIdx: *const af.Array,
    colIdx: *const af.Array,
    stype: af.Storage,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_create_sparse_array(
            &arr,
            @intCast(nRows),
            @intCast(nCols),
            values.array_,
            rowIdx.array_,
            colIdx.array_,
            stype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn createSparseArrayFromPtr()

/// Converts a dense `af.Array` into a sparse array.
pub inline fn createSparseArrayFromDense(
    allocator: std.mem.Allocator,
    dense: *const af.Array,
    stype: af.Storage,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_create_sparse_array_from_dense(
            &arr,
            dense.array_,
            stype.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Convert an existing sparse `af.Array` into a different storage format.
/// Converting storage formats is allowed between `af.Storage.CSR`, `af.Storage.COO`
/// and `af.Storage.Dense`.
///
/// When converting to `af.Storage.Dense`, a dense array is returned.
///
/// N.B. `af.Storage.CSC` is currently not supported.
pub inline fn sparseConvertTo(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    destStorage: af.Storage,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sparse_convert_to(
            &arr,
            in.array_,
            destStorage.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns a dense `af.Array` from a sparse `af.Array`.
pub inline fn sparseToDense(allocator: std.mem.Allocator, sparse: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sparse_to_dense(
            &arr,
            sparse.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns reference to components of the input sparse `af.Array`.
///
/// Returns reference to values, row indices, column indices and
/// storage format of an input sparse `af.Array`.
pub inline fn sparseGetInfo(allocator: std.mem.Allocator, in: *const af.Array) !struct {
    values: *af.Array,
    rowIdx: *af.Array,
    colIdx: *af.Array,
    stype: af.Storage,
} {
    var values: af.af_array = undefined;
    var rowIdx: af.af_array = undefined;
    var colIdx: af.af_array = undefined;
    var stype: af.af_storage = undefined;
    try af.AF_CHECK(
        af.af_sparse_get_info(
            &values,
            &rowIdx,
            &colIdx,
            &stype,
            in.array_,
        ),
        @src(),
    );
    return .{
        .values = try af.Array.init(allocator, values),
        .rowIdx = try af.Array.init(allocator, rowIdx),
        .colIdx = try af.Array.init(allocator, colIdx),
        .stype = @enumFromInt(stype),
    };
}

/// Returns reference to the values component of the sparse `af.Array`.
///
/// Values is the `af.Array` containing the non-zero elements of the
/// dense matrix.
pub inline fn sparseGetValues(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sparse_get_values(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns reference to the row indices component of the sparse `af.Array`.
///
/// Row indices is the `af.Array` containing the row indices of the sparse array.
pub inline fn sparseGetRowIdx(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sparse_get_row_idx(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns reference to the column indices component of the sparse `af.Array`.
///
/// Column indices is the `af.Array` containing the column indices of the sparse array.
pub inline fn sparseGetColIdx(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sparse_get_col_idx(
            &arr,
            in.array_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the number of non zero elements in the sparse `af.Array`.
///
/// This is always equal to the size of the values `af.Array`.
pub inline fn sparseGetNNZ(in: *const af.Array) !i64 {
    var nnz: af.dim_t = undefined;
    try af.AF_CHECK(
        af.af_sparse_get_nnz(
            &nnz,
            in.array_,
        ),
        @src(),
    );
    return @intCast(nnz);
}

/// Returns the storage type of a sparse `af.Array`.
///
/// The `af.Storage` type of the format of data storage
/// in the sparse `af.Array`.
pub inline fn sparseGetStorage(in: *const af.Array) !af.Storage {
    var stype: af.af_storage = undefined;
    try af.AF_CHECK(
        af.af_sparse_get_storage(
            &stype,
            in.array_,
        ),
        @src(),
    );
    return @enumFromInt(stype);
}

/// Returns the mean of the input `af.Array` along the specified dimension.
pub inline fn mean(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_mean(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the mean of the weighted input `af.Array` along the specified dimension.
pub inline fn meanWeighted(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    weights: *const af.Array,
    dim: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_mean_weighted(
            &arr,
            in.array_,
            weights.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the variance of the input `af.Array` along the specified dimension.
pub inline fn var_(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    isBiased: bool,
    dim: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_var(
            &arr,
            in.array_,
            isBiased,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the variance of the input `af.Array` along the specified dimension.
/// Type of bias specified using `af.VarBias` enum.
pub inline fn varV2(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    bias: af.VarBias,
    dim: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_var_v2(
            &arr,
            in.array_,
            bias.value(),
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the vairance of the weighted input `af.Array` along the specified dimension.
pub inline fn varWeighted(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    weights: *const af.Array,
    dim: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_var_weighted(
            &arr,
            in.array_,
            weights.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the mean and variance of the input `af.Array` along the specified dimension.
pub inline fn meanVar(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    weights: *const af.Array,
    bias: *const af.Array,
    dim: i64,
) !struct {
    mean: *af.Array,
    variance: *af.Array,
} {
    var mean_: af.af_array = undefined;
    var variance: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_meanvar(
            &mean_,
            &variance,
            in.array_,
            weights.array_,
            bias.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return .{
        .mean = try af.Array.init(allocator, mean_),
        .variance = try af.Array.init(allocator, variance),
    };
}

/// Returns the standard deviation of the input `af.Array` along the specified dimension.
pub inline fn stdev(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_stdev(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the standard deviation of the input `af.Array` along the specified dimension.
/// Type of bias used for variance calculation is specified with `af.VarBias` enum.
pub inline fn stdevV2(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    bias: af.VarBias,
    dim: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_stdev_v2(
            &arr,
            in.array_,
            bias.value(),
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the covariance of the input `af.Array`s along the specified dimension.
pub inline fn cov(
    allocator: std.mem.Allocator,
    X: *const af.Array,
    Y: *const af.Array,
    isBiased: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_cov(
            &arr,
            X.array_,
            Y.array_,
            isBiased,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the covariance of the input `af.Array`s along the specified dimension.
/// Type of bias used for variance calculation is specified with `af.VarBias` enum.
pub inline fn covV2(
    allocator: std.mem.Allocator,
    X: *const af.Array,
    Y: *const af.Array,
    bias: af.VarBias,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_cov_v2(
            &arr,
            X.array_,
            Y.array_,
            bias.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns the median of the input `af.Array` across the specified dimension.
pub inline fn median(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    dim: i64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_median(
            &arr,
            in.array_,
            @intCast(dim),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Data structure holding real and imaginary parts
/// of functions that return both.
pub const ComplexParts = struct {
    real: f64 = undefined,
    imaginary: f64 = undefined,
};

/// Returns both the real part and imaginary part of the mean
/// of the entire input `af.Array`.
pub fn meanAll(in: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_mean_all(
            &res.real,
            &res.imaginary,
            in.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns both the real part and imaginary part of the mean
/// of the entire weighted input `af.Array`.
pub fn meanAllWeighted(
    in: *const af.Array,
    weights: *const af.Array,
) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_mean_all_weighted(
            &res.real,
            &res.imaginary,
            in.array_,
            weights.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns both the real part and imaginary part of the variance
/// of the entire weighted input `af.Array`.
pub inline fn varAll(
    in: *const af.Array,
    isBiased: bool,
) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_var_all(
            &res.real,
            &res.imaginary,
            in.array_,
            isBiased,
        ),
        @src(),
    );
    return res;
}

/// Returns both the real part and imaginary part of the variance
/// of the entire weighted input `af.Array`.
///
/// Type of bias used for variance calculation is specified with
/// `af.VarBias` enum.
pub inline fn varAllV2(
    in: *const af.Array,
    bias: af.VarBias,
) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_var_all_v2(
            &res.real,
            &res.imaginary,
            in.array_,
            bias.value(),
        ),
        @src(),
    );
    return res;
}

/// Returns both the real part and imaginary part of the variance
/// of the entire weighted input `af.Array`.
pub inline fn varAllWeighted(in: *const af.Array, weights: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_var_all_weighted(
            &res.real,
            &res.imaginary,
            in.array_,
            weights.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns both the real part and imaginary part of the standard
/// deviation of the entire input `af.Array`.
pub inline fn stdevAll(in: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_stdev_all(
            &res.real,
            &res.imaginary,
            in.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns both the real part and imaginary part of the standard
/// deviation of the entire input `af.Array`.
///
/// Type of bias used for variance calculation is specified with
/// `af.VarBias` enum.
pub inline fn stdevAllV2(in: *const af.Array, bias: af.VarBias) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_stdev_all_v2(
            &res.real,
            &res.imaginary,
            in.array_,
            bias.value(),
        ),
        @src(),
    );
    return res;
}

/// Returns both the real part and imaginary part of the median
/// of the entire input `af.Array`.
pub inline fn medianAll(in: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_median_all(
            &res.real,
            &res.imaginary,
            in.array_,
        ),
        @src(),
    );
    return res;
}

/// Returns both the real part and imaginary part of the correlation
/// coefficient of the input `af.Array`s.
pub inline fn corrCoef(X: *const af.Array, Y: *const af.Array) !ComplexParts {
    var res = ComplexParts{};
    try af.AF_CHECK(
        af.af_corrcoef(
            &res.real,
            &res.imaginary,
            X.array_,
            Y.array_,
        ),
        @src(),
    );
    return res;
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
pub inline fn topk(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    k: i32,
    dim: i32,
    order: af.TopkFn,
) !struct { values: *af.Array, indices: *af.Array } {
    var values: af.af_array = undefined;
    var indices: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_topk(
            &values,
            &indices,
            in.array_,
            @intCast(k),
            @intCast(dim),
            order.value(),
        ),
        @src(),
    );
    return .{
        .values = try af.Array.init(allocator, values),
        .indices = try af.Array.init(allocator, indices),
    };
}

/// Returns `af.Features` struct containing arrays for x and y coordinates
/// and score, while array orientation is set to 0 as FAST does not compute
/// orientation, and size is set to 1 as FAST does not compute multiple scales.
pub inline fn fast(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    thr: f32,
    arc_length: u32,
    non_max: bool,
    feature_ratio: f32,
    edge: u32,
) !*af.Features {
    var feat: af.af_features = undefined;
    try af.AF_CHECK(
        af.af_fast(
            &feat,
            in.array_,
            thr,
            @intCast(arc_length),
            non_max,
            feature_ratio,
            @intCast(edge),
        ),
        @src(),
    );
    return af.Features.init(allocator, feat);
}

/// Returns `af.Features` struct containing arrays for x and y coordinates
/// and score (Harris response), while arrays orientation and size are set
/// to 0 and 1, respectively, because Harris does not compute that information.
pub inline fn harris(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    max_corners: u32,
    min_response: f32,
    sigma: f32,
    block_size: u32,
    k_thr: f32,
) !*af.Features {
    var feat: af.af_features = undefined;
    try af.AF_CHECK(
        af.af_harris(
            &feat,
            in.array_,
            @intCast(max_corners),
            min_response,
            sigma,
            @intCast(block_size),
            k_thr,
        ),
        @src(),
    );
    return af.Features.init(allocator, feat);
}

/// Returns struct containing the following fields:
/// - `feat`: `af.Features` composed of arrays for x and y coordinates,
/// score, orientation and size of selected features.
/// - `desc`: Nx8 `af.Array` containing extracted
/// descriptors, where N is the number of selected features.
pub inline fn orb(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    fast_thr: f32,
    max_feat: u32,
    scl_fctr: f32,
    levels: u32,
    blur_img: bool,
) !struct { feat: *af.Features, desc: *af.Array } {
    var feat: af.af_features = undefined;
    var desc: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_orb(
            &feat,
            &desc,
            in.array_,
            fast_thr,
            @intCast(max_feat),
            scl_fctr,
            @intCast(levels),
            blur_img,
        ),
        @src(),
    );
    return .{
        .feat = try af.Features.init(allocator, feat),
        .desc = try af.Array.init(allocator, desc),
    };
}

/// Returns struct containing the following fields:
/// - `feat`: `af.Features` composed of arrays for x and y coordinates,
/// score, orientation and size of selected features.
/// - `desc`: Nx128 `af.Array` containing extracted descriptors,
/// where N is the number of features found by SIFT.
pub inline fn sift(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    n_layers: u32,
    contrast_thr: f32,
    edge_thr: f32,
    init_sigma: f32,
    double_input: bool,
    intensity_scale: f32,
    feature_ratio: f32,
) !struct { feat: *af.Features, desc: *af.Array } {
    var feat: af.af_features = undefined;
    var desc: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_sift(
            &feat,
            &desc,
            in.array_,
            @intCast(n_layers),
            contrast_thr,
            edge_thr,
            init_sigma,
            double_input,
            intensity_scale,
            feature_ratio,
        ),
        @src(),
    );
    return .{
        .feat = try af.Features.init(allocator, feat),
        .desc = try af.Array.init(allocator, desc),
    };
}

/// Returns struct containing the following fields:
/// - `feat`: `af.Features` composed of arrays for x and y coordinates,
/// score, orientation and size of selected features.
/// - `desc`: Nx272 `af.Array` containing extracted GLOH descriptors,
/// where N is the number of features found by SIFT.
pub inline fn gloh(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    n_layers: u32,
    contrast_thr: f32,
    edge_thr: f32,
    init_sigma: f32,
    double_input: bool,
    intensity_scale: f32,
    feature_ratio: f32,
) !struct { feat: *af.Features, desc: *af.Array } {
    var feat: af.af_features = undefined;
    var desc: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_gloh(
            &feat,
            &desc,
            in.array_,
            @intCast(n_layers),
            contrast_thr,
            edge_thr,
            init_sigma,
            double_input,
            intensity_scale,
            feature_ratio,
        ),
        @src(),
    );
    return .{
        .feat = try af.Features.init(allocator, feat),
        .desc = try af.Array.init(allocator, desc),
    };
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
pub inline fn hammingMatcher(
    allocator: std.mem.Allocator,
    query: *const af.Array,
    train: *const af.Array,
    dist_dim: i64,
    n_dist: u32,
) !struct { idx: *af.Array, dist: *af.Array } {
    var idx: af.af_array = undefined;
    var dist: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_hamming_matcher(
            &idx,
            &dist,
            query.array_,
            train.array_,
            @intCast(dist_dim),
            @intCast(n_dist),
        ),
        @src(),
    );
    return .{
        .idx = try af.Array.init(allocator, idx),
        .dist = try af.Array.init(allocator, dist),
    };
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
pub inline fn nearestNeighbor(
    allocator: std.mem.Allocator,
    query: *const af.Array,
    train: *const af.Array,
    dist_dim: i64,
    n_dist: u32,
    dist_type: af.MatchType,
) !struct { idx: *af.Array, dist: *af.Array } {
    var idx: af.af_array = undefined;
    var dist: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_nearest_neighbour(
            &idx,
            &dist,
            query.array_,
            train.array_,
            @intCast(dist_dim),
            @intCast(n_dist),
            dist_type.value(),
        ),
        @src(),
    );
    return .{
        .idx = try af.Array.init(allocator, idx),
        .dist = try af.Array.init(allocator, dist),
    };
}

/// Template matching is an image processing technique to find small patches
/// of an image which match a given template image. A more in depth discussion
/// on the topic can be found [here](https://en.wikipedia.org/wiki/Template_matching).
///
/// Returns an `af.Array` containing disparity values for the window starting at
/// corresponding pixel position.
pub inline fn matchTemplate(
    allocator: std.mem.Allocator,
    search_img: *const af.Array,
    template_img: *const af.Array,
    m_type: af.MatchType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_match_template(
            &arr,
            search_img.array_,
            template_img.array_,
            m_type.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// SUSAN corner detector.
///
/// Returns `af.Features` struct composed of `af.Array`s for x and y coordinates,
/// score, orientation and size of selected features.
pub inline fn susan(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    radius: u32,
    diff_thr: f32,
    geom_thr: f32,
    feature_ratio: f32,
    edge: u32,
) !*af.Features {
    var feat: af.af_features = undefined;
    try af.AF_CHECK(
        af.af_susan(
            &feat,
            in.array_,
            @intCast(radius),
            diff_thr,
            geom_thr,
            feature_ratio,
            @intCast(edge),
        ),
        @src(),
    );
    return af.Features.init(allocator, feat);
}

/// Difference of Gaussians.
///
/// Given an image, this function computes two different versions
/// of smoothed input image using the difference smoothing parameters
/// and subtracts one from the other and returns the result.
///
/// Returns an `af.Array` containing the calculated difference of smoothed inputs.
pub inline fn dog(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    radius1: i32,
    radius2: i32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_dog(&arr, in.array_, @intCast(radius1), @intCast(radius2)), @src());
    return af.Array.init(allocator, arr);
}

/// Homography estimation find a perspective transform between two sets
/// of 2D points. Currently, two methods are supported for the estimation,
/// RANSAC (RANdom SAmple Consensus) and LMedS (Least Median of Squares).
/// Both methods work by randomly selecting a subset of 4 points of the set
/// of source points, computing the eigenvectors of that set and finding the
/// perspective transform. The process is repeated several times, a maximum
/// of times given by the value passed to the iterations arguments for RANSAC
/// (for the CPU backend, usually less than that, depending on the quality of
/// the dataset, but for CUDA and OpenCL backends the transformation will be
/// computed exactly the amount of times passed via the iterations parameter),
/// the returned value is the one that matches the best number of inliers, which
/// are all of the points that fall within a maximum L2 distance from the value
/// passed to the inlier_thr argument.
///
/// Returns struct composed of the following fields:
/// - `out`: a 3x3 `af.Array` containing the estimated homography.
/// - `inliers`: he number of inliers that the homography was estimated
/// to comprise, in the case that htype is AF_HOMOGRAPHY_RANSAC, a higher
/// inlier_thr value will increase the estimated inliers. Note that if the number
/// of inliers is too low, it is likely that a bad homography will be returned.
pub inline fn homography(
    allocator: std.mem.Allocator,
    inliers: *const af.Array,
    x_src: *const af.Array,
    y_src: *const af.Array,
    x_dst: *const af.Array,
    y_dst: *const af.Array,
    htype: af.HomographyType,
    inlier_thr: f32,
    iterations: u32,
    otype: af.Dtype,
) !struct { out: *af.Array, inliers: i32 } {
    var arr: af.af_array = undefined;
    var inliers_: c_int = undefined;
    try af.AF_CHECK(
        af.af_homography(
            &arr,
            &inliers_,
            inliers.array_,
            x_src.array_,
            y_src.array_,
            x_dst.array_,
            y_dst.array_,
            htype.value(),
            inlier_thr,
            @intCast(iterations),
            otype.value(),
        ),
        @src(),
    );
    return .{
        .out = try af.Array.init(allocator, arr),
        .inliers = @intCast(inliers_),
    };
}

/// Create an `af.Array` with specified strides and offset.
pub inline fn createStridedArray(
    allocator: std.mem.Allocator,
    data: ?*const anyopaque,
    offset: i64,
    ndims: u32,
    dims: af.Dim4,
    strides: af.Dim4,
    ty: af.Dtype,
    location: af.Source,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_create_strided_array(
            &arr,
            data,
            @intCast(offset),
            @intCast(ndims),
            &dims.dims,
            &strides.dims,
            ty.value(),
            location.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Get strides of underlying data.
pub inline fn getStrides(arr: *const af.Array) !af.Dim4 {
    var dims = af.Dim4{};
    try af.AF_CHECK(
        af.af_get_strides(
            &dims.dims[0],
            &dims.dims[1],
            &dims.dims[2],
            &dims.dims[3],
            arr.array_,
        ),
        @src(),
    );
    return dims;
}

/// Returns bool indicating whether all elements in an `af.Array` are contiguous.
pub inline fn isLinear(arr: *const af.Array) !bool {
    var res: bool = undefined;
    try af.AF_CHECK(af.af_is_linear(&res, arr.array_), @src());
    return res;
}

// unit tests

test "createArray" {
    const allocator = std.testing.allocator;
    var data = try allocator.alloc(f32, 100);
    defer allocator.free(data);
    @memset(data, 100);
    var dim = [_]Dim{100};
    var shape = try Shape.init(allocator, &dim);
    defer shape.deinit();

    var arr = try createArray(
        allocator,
        data.ptr,
        @intCast(shape.ndim()),
        try shape.toAfDims(),
        .f32,
    );
    defer arr.deinit();
    const elements = try arr.getElements();
    try std.testing.expect(elements == 100);
}

test "constant" {
    const allocator = std.testing.allocator;
    var d = [4]af.dim_t{ 100, 1, 1, 1 };
    var dims = af.Dim4.init(d);
    var arr = try constant(allocator, 5, 4, dims, .f64);
    defer arr.deinit();

    try std.testing.expect(try getScalar(f64, arr) == 5);
    try std.testing.expect(try getElements(arr) == 100);
}
