const std = @import("std");
const af = @import("ArrayFire.zig");
const zt_types = @import("../../tensor/Types.zig");
const zt_base = @import("../../tensor/TensorBase.zig");
const zt_shape = @import("../../tensor/Shape.zig");

const Location = zt_base.Location;
const DType = zt_types.DType;
const Shape = zt_shape.Shape;
const Dim = zt_shape.Dim;

/// Returns the sum of the elements of the `af.Array` along
/// the given dimension.
pub inline fn sum(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_sum(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

/// Returns the sum of the elements of the `af.Array` along
/// the given dimension, replacing NaN values with `nanval`.
pub inline fn sumNan(allocator: std.mem.Allocator, in: *af.Array, dim: usize, nanval: f64) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_sum_nan(&arr, in.array_, dim, nanval), @src());
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn sumByKey

// TODO: pub inline fn sumByKeyNan

pub inline fn product(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_product(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn productNan(allocator: std.mem.Allocator, in: *af.Array, dim: usize, nanval: f64) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_product_nan(&arr, in.array_, @intCast(dim), nanval), @src());
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn productByKey

// TODO: pub inline fn productByKeyNan

pub inline fn min(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_min(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn minByKey

pub inline fn max(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_max(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn maxByKey

// TODO: pub inline fn maxRagged

pub inline fn allTrue(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_all_true(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn allTrueByKey

pub inline fn anyTrue(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_any_true(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn anyTrueByKey

pub inline fn count(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_count(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn countByKey

// TODO: pub inline fn sumAll

// TODO: pub inline fn sumNanAll

// TODO: pub inline fn productAll

// TODO: pub inline fn productNanAll

// TODO: pub inline fn minAll

// TODO: pub inline fn maxAll

// TODO: pub inline fn allTrueAll

// TODO: pub inline fn anyTrueAll

// TODO: pub inline fn countAll

// TODO: pub inline fn imin

// TODO: pub inline fn imax

// TODO: pub inline fn iminAll

// TODO: pub inline fn imaxAll

pub inline fn accum(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_accum(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn scan(allocator: std.mem.Allocator, in: *af.Array, dim: usize, op: af.BinaryOp, inclusive_scan: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_scan(&arr, in.array_, @intCast(dim), op, inclusive_scan), @src());
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn scanByKey

pub inline fn where(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_where(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn diff1(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_diff1(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn diff2(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_diff2(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn sort(allocator: std.mem.Allocator, in: *af.Array, dim: usize, is_ascending: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_sort(&arr, in.array_, @intCast(dim), is_ascending), @src());
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn sortIndex

// TODO: pub inline fn sortByKey

pub inline fn setUnique(allocator: std.mem.Allocator, in: *af.Array, is_sorted: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_set_unique(&arr, in.array_, is_sorted), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn setUnion(allocator: std.mem.Allocator, first: *af.Array, second: *af.Array, is_unique: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_set_union(&arr, first.array_, second.array_, is_unique), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn setIntersect(allocator: std.mem.Allocator, first: *af.Array, second: *af.Array, is_unique: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_set_intersect(&arr, first.array_, second.array_, is_unique), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn add(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_add(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn sub(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_sub(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn mul(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_mul(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn div(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_div(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn lt(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_lt(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn gt(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_gt(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn le(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_le(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn ge(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_ge(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn eq(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_eq(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn neq(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_neq(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn and_(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_and(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn or_(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_or(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn not(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_not(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn bitNot(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_bitnot(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn bitAnd(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_bitand(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn bitOr(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_bitor(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn bitXor(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_bitxor(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn bitShiftL(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_bitshiftl(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn bitShiftR(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_bitshiftr(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn cast(allocator: std.mem.Allocator, in: *af.Array, dtype: DType) !*af.Array {
    const af_type = dtype.toAfDtype();
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_cast(&arr, in.array_, af_type), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn minOf(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_minof(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn maxOf(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_maxof(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn clamp(allocator: std.mem.Allocator, in: *af.Array, lo: *af.Array, hi: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_clamp(&arr, in.array_, lo.array_, hi.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn rem(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_rem(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn mod(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_mod(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn abs(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_abs(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn arg(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_fabs(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn sign(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_sign(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn round(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_round(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn trunc(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_trunc(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn floor(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_floor(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn ceil(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_ceil(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn hypot(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_hypot(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn sin(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_sin(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn cos(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_cos(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn tan(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_tan(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn asin(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_asin(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn acos(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_acos(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn atan(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_atan(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn atan2(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_atan2(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn sinh(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_sinh(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn cosh(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_cosh(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn tanh(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_tanh(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn asinh(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_asinh(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn acosh(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_acosh(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn atanh(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_atanh(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn cplx(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_cplx(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn cplx2(allocator: std.mem.Allocator, real_: *af.Array, imag_: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_cplx2(&arr, real_.array_, imag_.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn real(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_real(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn imag(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_imag(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn conjg(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_conjg(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn root(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_root(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn pow(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, batch: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_pow(&arr, lhs.array_, rhs.array_, batch), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn pow2(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_pow2(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn sigmoid(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_sigmoid(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn exp(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_exp(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn expm1(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_expm1(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn erf(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_erf(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn erfc(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_erfc(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn log(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_log(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn log1p(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_log1p(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn log10(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_log10(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn log2(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_log2(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn sqrt(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_sqrt(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn rsqrt(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_rsqrt(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn cbrt(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_cbrt(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn factorial(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_factorial(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn tgamma(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_tgamma(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn lgamma(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_lgamma(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn isZero(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_iszero(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn isInf(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_isinf(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn isNan(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_isnan(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Initializes an af.Array from device memory.
pub inline fn deviceArray(allocator: std.mem.Allocator, data: ?*anyopaque, ndims: usize, dims: af.Dims4, dtype: DType) !af.af_array {
    var af_type = dtype.toAfDtype();
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_device_array(&arr, data, @intCast(ndims), &dims.dims, af_type), @src());
    return af.Array.init(allocator, arr);
}

/// Lock the device buffer in the memory manager.
pub inline fn lockDevicePtr(arr: *af.Array) !void {
    try af.AF_CHECK(af.af_lock_device_ptr(arr.array_), @src());
}

/// Unlock device buffer in the memory manager.
pub inline fn unlockDevicePtr(arr: *af.Array) !void {
    try af.AF_CHECK(af.af_unlock_device_ptr(arr.array_), @src());
}

/// Lock the device buffer in the memory manager.
pub inline fn lockArray(arr: *af.Array) !void {
    try af.AF_CHECK(af.af_lock_array(arr.array_), @src());
}

/// Unlock device buffer in the memory manager.
pub inline fn unlockArray(arr: *af.Array) !void {
    try af.AF_CHECK(af.af_unlock_array(arr.array_), @src());
}

/// Query if the array has been locked by the user.
pub inline fn isLockedArray(arr: *af.Array) !bool {
    var is_locked: bool = undefined;
    try af.AF_CHECK(af.af_is_locked_array(&is_locked, arr.array_), @src());
    return is_locked;
}

/// Get the device pointer and lock the buffer in memory manager.
pub inline fn getDevicePtr(arr: *af.Array) !?*anyopaque {
    var ptr: ?*anyopaque = undefined;
    try af.AF_CHECK(af.af_get_device_ptr(&ptr, arr.array_), @src());
    return ptr;
}

/// Lookup the values of input array based on sequences.
pub inline fn index(allocator: std.mem.Allocator, in: *af.Array, ndims: usize, idx: *const af.af_seq) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_index(&arr, in.array_, @intCast(ndims), idx), @src());
    return af.Array.init(allocator, arr);
}

/// Lookup the values of an input array by indexing with another array.
pub inline fn lookup(allocator: std.mem.Allocator, in: *af.Array, indices: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_lookup(&arr, in.array_, indices.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

/// Copy and write values in the locations specified by the sequences.
pub inline fn assignSeq(allocator: std.mem.Allocator, lhs: *af.Array, ndims: usize, indices: *const af.af_seq, rhs: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_assign_seq(&arr, lhs.array_, @intCast(ndims), indices, rhs.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Indexing an array using `af.af_seq`, or `af.Array`.
pub inline fn indexGen(allocator: std.mem.Allocator, in: *af.Array, ndims: i64, indices: *const af.af_index_t) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_index_gen(&arr, in.array_, @intCast(ndims), indices), @src());
    return af.Array.init(allocator, arr);
}

// TODO: fix as not working
/// Print the array and dimensions to screen.
pub inline fn printArray(arr: *af.Array) !void {
    try af.AF_CHECK(af.af_print_array(arr.array_), @src());
}

pub inline fn printArrayGen(arr: *af.Array, precision: i32) ![]const u8 {
    var msg: [*c]const u8 = undefined;
    try af.AF_CHECK(af.af_print_array_gen(msg, arr.array_, @intCast(precision)), @src());
    return std.mem.span(msg);
}

// TODO: pub inline fn saveArray()

// TODO: pub inline fn readArrayIndex()

// TODO: pub inline fn readArrayKey()

// TODO: pub inline fn readArrayKeyCheck()

// TODO: pub inline fn arrayToString()

// TODO: pub inline fn exampleFunction()???

/// Create an `af.Array` handle initialized with user defined data.
pub inline fn createArray(allocator: std.mem.Allocator, shape: *const Shape, ptr: ?*const anyopaque, dtype: DType) !*af.Array {
    var dims = try shape.toAfDims();
    var af_dtype = dtype.toAfDtype();
    var res: af.af_array = undefined;
    switch (dtype) {
        .f32, .f64, .s32, .u32, .s64, .u64, .s16, .u16, .b8, .u8 => try af.AF_CHECK(af.af_create_array(
            &res,
            ptr,
            @intCast(shape.ndim()),
            &dims.dims,
            af_dtype.value(),
        ), @src()),
        else => {
            std.log.err("createArray: can't construct ArrayFire array from given type.\n", .{});
            return error.UnsupportedArrayFireType;
        },
    }
    return af.Array.init(allocator, res);
}

/// Create `af.Array` handle.
pub inline fn createHandle(allocator: std.mem.Allocator, ndims: usize, dims: af.Dims4, dtype: af.Dtype) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_create_handle(&arr, @intCast(ndims), &dims.dims, @intFromEnum(dtype)), @src());
    return af.Array.init(allocator, arr);
}

/// Deep copy an array to another.
pub inline fn copyArray(allocator: std.mem.Allocator, in: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_copy_array(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Copy data from a C pointer (host/device) to an existing array.
pub inline fn writeArray(arr: *af.Array, data: ?*const anyopaque, bytes: usize, src: Location) !void {
    try af.AF_CHECK(af.af_write_array(arr.array_, data, bytes, src.toAfSource()), @src());
}

/// Copy data from an `af.Array` to a C pointer.
pub inline fn getDataPtr(data: ?*anyopaque, arr: *af.Array) !void {
    try af.AF_CHECK(af.af_get_data_ptr(data, arr.array_), @src());
}

/// Reduce the reference count of the `af.Array`.
pub inline fn releaseArray(arr: *af.Array) !void {
    try af.AF_CHECK(af.af_release_array(arr.array_), @src());
}

/// Get the reference count of `af.Array`.
pub inline fn getDataRefCount(arr: *af.Array) !i32 {
    var refCount: i32 = undefined;
    try af.AF_CHECK(af.af_get_data_ref_count(&refCount, arr.array_), @src());
    return refCount;
}

/// Evaluate any expressions in the `af.Array`.
pub inline fn eval(in: *af.Array) !void {
    try af.AF_CHECK(af.af_eval(in.array_), @src());
}

/// Evaluate a slice of `af.Array`s together.
pub inline fn evalMultiple(arrays: []*af.Array) !void {
    try af.AF_CHECK(af.af_eval_multiple(@intCast(arrays.len), arrays.ptr), @src());
}

/// Get the total number of elements across all dimensions of the `af.Array`.
pub inline fn getElements(arr: *af.Array) !usize {
    var elements: af.dim_t = undefined;
    try af.AF_CHECK(af.af_get_elements(&elements, arr.array_), @src());
    return @intCast(elements);
}

/// Gets the type of an `af.Array`.
pub inline fn getType(arr: *af.Array) !af.Dtype {
    var dtype: af.af_dtype = undefined;
    try af.AF_CHECK(af.af_get_type(&dtype, arr.array_), @src());
    return @enumFromInt(dtype);
}

/// Gets the dimensions of an `af.Array`.
pub inline fn getDims(arr: *af.Array) !af.Dims4 {
    var dims: af.Dims4 = .{};
    try af.AF_CHECK(af.af_get_dims(&dims.dims[0], &dims.dims[1], &dims.dims[2], &dims.dims[3], arr.array_), @src());
    return dims;
}

/// Gets the number of dimensions of an `af.Array`.
pub inline fn getNumDims(arr: *af.Array) !usize {
    var numDims: c_uint = undefined;
    try af.AF_CHECK(af.af_get_numdims(&numDims, arr.array_), @src());
    return @intCast(numDims);
}

/// Check if an `af.Array` is empty.
pub inline fn isEmpty(arr: *af.Array) !bool {
    var empty: bool = undefined;
    try af.AF_CHECK(af.af_is_empty(&empty, arr.array_), @src());
    return empty;
}

/// Check if an `af.Array` is scalar.
pub inline fn isScalar(arr: *af.Array) !bool {
    var scalar: bool = undefined;
    try af.AF_CHECK(af.af_is_scalar(&scalar, arr.array_), @src());
    return scalar;
}

/// Check if an `af.Array` is row vector.
pub inline fn isRow(arr: *af.Array) !bool {
    var row: bool = undefined;
    try af.AF_CHECK(af.af_is_row(&row, arr.array_), @src());
    return row;
}

/// Check if an `af.Array` is a column vector.
pub inline fn isColumn(arr: *af.Array) !bool {
    var col: bool = undefined;
    try af.AF_CHECK(af.af_is_column(&col, arr.array_), @src());
    return col;
}

/// Check if an `af.Array` is a vector.
pub inline fn isVector(arr: *af.Array) !bool {
    var vec: bool = undefined;
    try af.AF_CHECK(af.af_is_vector(&vec, arr.array_), @src());
    return vec;
}

/// Check if an `af.Array` is complex type.
pub inline fn isComplex(arr: *af.Array) !bool {
    var complex: bool = undefined;
    try af.AF_CHECK(af.af_is_complex(&complex, arr.array_), @src());
    return complex;
}

/// Check if an `af.Array` is a real type.
pub inline fn isReal(arr: *af.Array) !bool {
    var is_real: bool = undefined;
    try af.AF_CHECK(af.af_is_real(&is_real, arr.array_), @src());
    return is_real;
}

/// Check if an `af.Array` is double precision type.
pub inline fn isDouble(arr: *af.Array) !bool {
    var is_double: bool = undefined;
    try af.AF_CHECK(af.af_is_double(&is_double, arr.array_), @src());
    return is_double;
}

/// Check if an `af.Array` is single precision type.
pub inline fn isSingle(arr: *af.Array) !bool {
    var is_single: bool = undefined;
    try af.AF_CHECK(af.af_is_single(&is_single, arr.array_), @src());
    return is_single;
}

/// Check if an `af.Array` is a 16 bit floating point type.
pub inline fn isHalf(arr: *af.Array) !bool {
    var is_half: bool = undefined;
    try af.AF_CHECK(af.af_is_half(&is_half, arr.array_), @src());
    return is_half;
}

/// Check if an `af.Array` is a real floating point type.
pub inline fn isRealFloating(arr: *af.Array) !bool {
    var is_real_floating: bool = undefined;
    try af.AF_CHECK(af.af_is_realfloating(&is_real_floating, arr.array_), @src());
    return is_real_floating;
}

/// Check if an `af.Array` is floating precision type.
pub inline fn isFloating(arr: *af.Array) !bool {
    var is_floating: bool = undefined;
    try af.AF_CHECK(af.af_is_floating(&is_floating, arr.array_), @src());
    return is_floating;
}

/// Check if an `af.Array` is integer type.
pub inline fn isInteger(arr: *af.Array) !bool {
    var is_integer: bool = undefined;
    try af.AF_CHECK(af.af_is_integer(&is_integer, arr.array_), @src());
    return is_integer;
}

/// Check if an `af.Array` is bool type.
pub inline fn isBool(arr: *af.Array) !bool {
    var is_bool: bool = undefined;
    try af.AF_CHECK(af.af_is_bool(&is_bool, arr.array_), @src());
    return is_bool;
}

/// Check if an `af.Array` is sparse.
pub inline fn isSparse(arr: *af.Array) !bool {
    var is_sparse: bool = undefined;
    try af.AF_CHECK(af.af_is_sparse(&is_sparse, arr.array_), @src());
    return is_sparse;
}

/// Get first element from an `af.Array`.
pub inline fn getScalar(comptime T: type, arr: *af.Array) !T {
    var res: T = undefined;
    try af.AF_CHECK(af.af_get_scalar(&res, arr.array_), @src());
    return res;
}

pub const AFMatProp = enum(af.af_mat_prop) {
    None = af.AF_MAT_NONE,
    Trans = af.AF_MAT_TRANS,
    CTrans = af.AF_MAT_CTRANS,
    Conj = af.AF_MAT_CONJ,
    Upper = af.AF_MAT_UPPER,
    Lower = af.AF_MAT_LOWER,
    DiagUnit = af.AF_MAT_DIAG_UNIT,
    Sym = af.AF_MAT_SYM,
    PosDef = af.AF_MAT_POSDEF,
    Orthog = af.AF_MAT_ORTHOG,
    TriDiag = af.AF_MAT_TRI_DIAG,
    BlockDiag = af.AF_MAT_BLOCK_DIAG,

    pub fn value(self: *AFMatProp) c_uint {
        return @intFromEnum(self);
    }
};

// TODO: gemm should optionally allocate based on the `C` param

/// BLAS general matrix multiply (GEMM) of two `af.Array` objects.
///
/// This provides a general interface to the BLAS level 3 general
/// matrix multiply (GEMM), which is generally defined as:
///
/// C=α∗opA(A)opB(B)+β∗C
///
/// Where α (alpha) and β (beta) are both scalars; A and B are the matrix
/// multiply operands; and opA and opB are noop (if `AFMatProp.None`) or transpose
/// (if `AFMatProp.Trans`) operations on A or B before the actual GEMM operation.
/// Batched GEMM is supported if at least either A or B have more than two dimensions
/// (see `matmul` for more details on broadcasting). However, only one alpha and one
/// beta can be used for all of the batched matrix operands.
///
/// The `af.Array` that out points to can be used both as an input and output.
/// An allocation will be performed if you pass a null `af.Array` handle (i.e. af_array c = 0;).
/// If a valid `af.Array` is passed as C, the operation will be performed on that `af.Array` itself.
/// The C af_array must be the correct type and shape; otherwise, an error will be thrown.
pub inline fn gemm(comptime T: type, C: *af.Array, opA: AFMatProp, opB: AFMatProp, alpha: T, A: *af.Array, B: *af.Array, beta: T) !void {
    try af.AF_CHECK(af.af_gemm(C, opA, opB, &alpha, A.array_, B.array_, &beta), @src());
}

/// Performs a matrix multiplication on two `af.Array`s (lhs, rhs).
///
/// N.B. The following applies for Sparse-Dense matrix multiplication.
/// This function can be used with one sparse input. The sparse input
/// must always be the lhs and the dense matrix must be rhs. The sparse
/// array can only be of AF_STORAGE_CSR format. The returned array is
/// always dense. optLhs an only be one of `AFMatProp.None`, `AFMatProp.Trans`,
/// `AFMatProp.CTrans`. optRhs can only be AF_MAT_NONE.
pub inline fn matmul(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, optLhs: AFMatProp, optRhs: AFMatProp) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_matmul(&arr, lhs.array_, rhs.array_, optLhs, optRhs), @src());
    return af.Array.init(allocator, arr);
}

/// Scalar dot product between two vectors.
///
/// Also referred to as the inner product.
pub inline fn dot(allocator: std.mem.Allocator, lhs: *af.Array, rhs: *af.Array, optLhs: AFMatProp, optRhs: AFMatProp) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_dot(&arr, lhs.array_, rhs.array_, optLhs, optRhs), @src());
    return af.Array.init(allocator, arr);
}

// TODO: pub inline fn dotAll()

/// Transpose a matrix.
pub inline fn transpose(allocator: std.mem.Allocator, in: *af.Array, conjugate: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_transpose(&arr, in.array_, conjugate), @src());
    return af.Array.init(allocator, arr);
}

/// Tranpose a matrix in-place.
pub inline fn transposeInPlace(in: *af.Array, conjugate: bool) !void {
    try af.AF_CHECK(af.af_transpose_inplace(in.array_, conjugate), @src());
}

/// Create an `af.Array` from a scalar input value.
///
/// The `af.Array` created has the same value at all locations.
pub inline fn constant(allocator: std.mem.Allocator, val: f64, ndims: usize, dims: af.Dims4, dtype: DType) !*af.Array {
    var arr: af.af_array = undefined;
    var af_type = dtype.toAfDtype();
    try af.AF_CHECK(af.af_constant(&arr, val, @intCast(ndims), &dims.dims, af_type.value()), @src());
    return af.Array.init(allocator, arr);
}

/// Create a complex type `af.Array` from a scalar input value.
///
/// The `af.Array` created has the same value at all locations.
pub inline fn constantComplex(allocator: std.mem.Allocator, real_: f64, imag_: f64, ndims: usize, dims: af.Dims4, dtype: DType) !*af.Array {
    var arr: af.af_array = undefined;
    var af_type = dtype.toAfDtype();
    try af.AF_CHECK(af.af_constant_complex(&arr, real_, imag_, @intCast(ndims), &dims.dims, af_type.value()), @src());
    return af.Array.init(allocator, arr);
}

/// Create an `af.Array` of type s64 from a scalar input value.
///
/// The `af.Array` created has the same value at all locations.
pub inline fn constantLong(allocator: std.mem.Allocator, val: i64, ndims: usize, dims: af.Dims4) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_constant_long(&arr, @intCast(val), @intCast(ndims), &dims.dims), @src());
    return af.Array.init(allocator, arr);
}

/// Create an `af.Array` of type u64 from a scalar input value.
///
/// The `af.Array` created has the same value at all locations.
pub inline fn constantUlong(allocator: std.mem.Allocator, val: u64, ndims: usize, dims: af.Dims4) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_constant_ulong(&arr, @intCast(val), @intCast(ndims), &dims.dims), @src());
    return af.Array.init(allocator, arr);
}

/// Creates an `af.Array` with [0, n-1] values along the `seq_dim` dimension
/// and tiled across other dimensions specified by an array of `ndims` dimensions.
pub inline fn range(allocator: std.mem.Allocator, ndims: usize, dims: af.Dims4, seq_dim: i32, dtype: DType) !*af.Array {
    var arr: af.af_array = undefined;
    var af_type = dtype.toAfDtype();
    try af.AF_CHECK(af.af_range(&arr, @intCast(ndims), &dims.dims, seq_dim, af_type), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn iota(allocator: std.mem.Allocator, ndims: usize, dims: af.Dims4, t_ndims: usize, tdims: af.Dims4, dtype: DType) !*af.Array {
    var arr: af.af_array = undefined;
    var af_type = dtype.toAfDtype();
    try af.AF_CHECK(af.af_iota(&arr, @intCast(ndims), &dims.dims, @intCast(t_ndims), &tdims.dims, af_type), @src());
    return af.Array.init(allocator, arr);
}

/// Returns an identity `af.Array` with diagonal values 1.
pub inline fn identity(allocator: std.mem.Allocator, ndims: usize, dims: af.Dims4, dtype: DType) !*af.Array {
    var arr: af.af_array = undefined;
    var af_type = dtype.toAfDtype();
    try af.AF_CHECK(af.af_identity(&arr, @intCast(ndims), &dims.dims, af_type), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn diagCreate(allocator: std.mem.Allocator, in: *af.Array, num: i32) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_diag_create(&arr, in.array_, @intCast(num)), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn diagExtract(allocator: std.mem.Allocator, in: *af.Array, num: i32) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_diag_extract(&arr, in.array_, @intCast(num)), @src());
    return af.Array.init(allocator, arr);
}

/// Joins 2 `af.Array`s along `dim`.
pub inline fn join(allocator: std.mem.Allocator, dim: i32, first: *af.Array, second: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_join(&arr, dim, first.array_, second.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Join many `af.Array`s along dim.
///
/// Current limit is set to 10 `af.Array`s.
pub inline fn joinMany(allocator: std.mem.Allocator, dim: i32, n_arrays: usize, inputs: []const *af.Array) !*af.Array {
    var in_arrays = try allocator.alloc(af.af_array, n_arrays);
    defer allocator.free(in_arrays);
    for (inputs, 0..) |array, i| in_arrays[i] = array.array_;
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_join_many(&arr, dim, @intCast(n_arrays), in_arrays.ptr), @src());
    return af.Array.init(allocator, arr);
}

/// Repeat the contents of the input `af.Array` along the specified dimensions.
pub inline fn tile(allocator: std.mem.Allocator, in: *af.Array, x: usize, y: usize, z: usize, w: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_tile(&arr, in.array_, @intCast(x), @intCast(y), @intCast(z), @intCast(w)), @src());
    return af.Array.init(allocator, arr);
}

/// Reorder an `af.Array` according to the specified dimensions.
pub inline fn reorder(allocator: std.mem.Allocator, in: *af.Array, x: usize, y: usize, z: usize, w: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_reorder(&arr, in.array_, @intCast(x), @intCast(y), @intCast(z), @intCast(w)), @src());
    return af.Array.init(allocator, arr);
}

/// Circular shift along specified dimensions.
pub inline fn shift(allocator: std.mem.Allocator, in: *af.Array, x: i32, y: i32, z: i32, w: i32) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_shift(&arr, in.array_, @intCast(x), @intCast(y), @intCast(z), @intCast(w)), @src());
    return af.Array.init(allocator, arr);
}

/// Modifies the dimensions of an input `af.Array` to the shape specified
/// by an array of ndims dimensions.
pub inline fn moddims(allocator: std.mem.Allocator, in: *af.Array, ndims: usize, dims: af.Dims4) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_moddims(&arr, in.array_, @intCast(ndims), &dims.dims), @src());
    return af.Array.init(allocator, arr);
}

/// Flatten the input to a single dimension.
pub inline fn flat(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_flat(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Flip the input along specified dimension.
///
/// Mirrors the `af.Array` along the specified dimensions.
pub inline fn flip(allocator: std.mem.Allocator, in: *af.Array, dim: usize) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_flip(&arr, in.array_, @intCast(dim)), @src());
    return af.Array.init(allocator, arr);
}

/// Create a lower triangular matrix from input `af.Array`.
pub inline fn lower(allocator: std.mem.Allocator, in: *const af.Array, is_unit_diag: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_lower(&arr, in.array_, is_unit_diag), @src());
    return af.Array.init(allocator, arr);
}

/// Create an upper triangular matrix from input `af.Array`.
pub inline fn upper(allocator: std.mem.Allocator, in: *const af.Array, is_unit_diag: bool) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_upper(&arr, in.array_, is_unit_diag), @src());
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
pub inline fn select(allocator: std.mem.Allocator, cond: *af.Array, a: *af.Array, b: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_select(&arr, cond.array_, a.array_, b.array_), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn selectScalarR(allocator: std.mem.Allocator, cond: *af.Array, a: *af.Array, b: f64) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_select_scalar_r(&arr, cond.array_, a.array_, b), @src());
    return af.Array.init(allocator, arr);
}

pub inline fn selectScalarL(allocator: std.mem.Allocator, cond: *af.Array, a: f64, b: *af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_select_scalar_r(&arr, cond.array_, a, b.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Replace elements of an `af.Array` based on a conditional `af.Array`.
///
/// - Input values are retained when corresponding elements from condition array are true.
/// - Input values are replaced when corresponding elements from condition array are false.
pub inline fn replace(a: *af.Array, cond: *af.Array, b: *af.Array) !void {
    try af.AF_CHECK(af.af_replace(a.array_, cond.array_, b.array_), @src());
}

/// Replace elements of an `af.Array` based on a conditional `af.Array`.
///
/// N.B. Values of a are replaced with corresponding values of b, when cond is false.
pub inline fn replaceScalar(a: *af.Array, cond: *af.Array, b: f64) !void {
    try af.AF_CHECK(af.af_replace_scalar(a.array_, cond.array_, b), @src());
}

pub const AFBorderType = enum(af.af_border_type) {
    PadZero = af.AF_PAD_ZERO,
    PadSym = af.AF_PAD_SYM,
    PadClampToEdge = af.AF_PAD_CLAMP_TO_EDGE,
    PadPeriodic = af.AF_PAD_PERIODIC,

    pub fn value(self: AFBorderType) af.af_border_type {
        return @intFromEnum(self);
    }
};

/// Pad an `af.Array`.
///
/// Pad the input `af.Array` using a constant or values from input along border
pub inline fn pad(
    allocator: std.mem.Allocator,
    in: *af.Array,
    begin_ndims: usize,
    begin_dims: af.Dims4,
    end_ndims: usize,
    end_dims: af.Dims4,
    pad_fill_type: AFBorderType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_pad(&arr, in.array_, @intCast(begin_ndims), &begin_dims.dims, @intCast(end_ndims), &end_dims.dims, pad_fill_type), @src());
    return af.Array.init(allocator, arr);
}

pub const AFGradientType = enum(af.af_conv_gradient_type) {
    Default = af.AF_CONV_GRADIENT_DEFAULT,
    Filter = af.AF_CONV_GRADIENT_FILTER,
    Data = af.AF_CONV_GRADIENT_DATA,
    Bias = af.AF_CONV_GRADIENT_BIAS,

    pub fn value(self: AFGradientType) af.af_conv_gradient_type {
        return @intFromEnum(self);
    }
};

/// Calculates backward pass gradient of 2D convolution.
///
/// This function calculates the gradient with respect to the output
/// of the `convolve2NN` function that uses the machine learning formulation
/// for the dimensions of the signals and filters.
pub inline fn convolve2GradientNN(
    allocator: std.mem.Allocator,
    incoming_gradient: *af.Array,
    original_signal: *af.Array,
    original_filter: *af.Array,
    convolved_output: *af.Array,
    stride_dims: usize,
    strides: *af.Dims4,
    padding_dims: usize,
    paddings: af.Dims4,
    dilation_dims: usize,
    dilations: af.Dims4,
    grad_type: AFGradientType,
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
            grad_type,
        ),
        @src(),
    );
    return af.Array.init(allocator, res);
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

    var arr = try createArray(allocator, &shape, data.ptr, DType.f32);
    defer arr.deinit();
    const elements = try arr.getElements();
    try std.testing.expect(elements == 100);
}

test "constant" {
    const allocator = std.testing.allocator;
    var d = [4]af.dim_t{ 100, 1, 1, 1 };
    var dims = af.Dims4.init(d);
    var arr = try constant(allocator, 5, 4, dims, .f64);
    defer arr.deinit();

    try std.testing.expect(try getScalar(f64, arr) == 5);
    try std.testing.expect(try getElements(arr) == 100);
}
