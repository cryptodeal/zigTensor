const std = @import("std");
const af_utils = @import("../Utils.zig");
const af = @import("../../../../backends/ArrayFire.zig");
const af_types = @import("types.zig");
const zt_types = @import("../../../Types.zig");

const ArrayFireArray = @import("array.zig").ArrayFireArray;
const DType = zt_types.DType;
const BinaryOp = af_types.BinaryOp;
const AfDims4 = af_utils.AfDims4;
const AF_CHECK = af_utils.AF_CHECK;
const ztToAfType = af_utils.ztToAfType;

pub inline fn sum(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_sum(&arr, in.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn sumNan(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_sum_nan(&arr, in.af_array, dim, std.math.nan(f64)), @src());
    return ArrayFireArray.init(allocator, arr);
}

// TODO: pub inline fn sumByKey

// TODO: pub inline fn sumByKeyNan

pub inline fn product(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_product(&arr, in.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn productNan(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_product_nan(&arr, in.af_array, @intCast(dim), std.math.nan(f64)), @src());
    return ArrayFireArray.init(allocator, arr);
}

// TODO: pub inline fn productByKey

// TODO: pub inline fn productByKeyNan

pub inline fn min(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_min(&arr, in.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
}

// TODO: pub inline fn minByKey

pub inline fn max(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_max(&arr, in.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
}

// TODO: pub inline fn maxByKey

// TODO: pub inline fn maxRagged

pub inline fn allTrue(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_all_true(&arr, in.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
}

// TODO: pub inline fn allTrueByKey

pub inline fn anyTrue(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_any_true(&arr, in.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
}

// TODO: pub inline fn anyTrueByKey

pub inline fn count(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_count(&arr, in.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
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

pub inline fn accum(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_accum(&arr, in.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn scan(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize, op: BinaryOp, inclusive_scan: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_scan(&arr, in.af_array, @intCast(dim), op.value(), inclusive_scan), @src());
    return ArrayFireArray.init(allocator, arr);
}

// TODO: pub inline fn scanByKey

pub inline fn where(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_where(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn diff1(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_diff1(&arr, in.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn diff2(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_diff2(&arr, in.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn sort(allocator: std.mem.Allocator, in: *ArrayFireArray, dim: usize, is_ascending: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_sort(&arr, in.af_array, @intCast(dim), is_ascending), @src());
    return ArrayFireArray.init(allocator, arr);
}

// TODO: pub inline fn sortIndex

// TODO: pub inline fn sortByKey

pub inline fn setUnique(allocator: std.mem.Allocator, in: *ArrayFireArray, is_sorted: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_set_unique(&arr, in.af_array, is_sorted), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn setUnion(allocator: std.mem.Allocator, first: *ArrayFireArray, second: *ArrayFireArray, is_unique: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_set_union(&arr, first.af_array, second.af_array, is_unique), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn setIntersect(allocator: std.mem.Allocator, first: *ArrayFireArray, second: *ArrayFireArray, is_unique: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_set_intersect(&arr, first.af_array, second.af_array, is_unique), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn add(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_add(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn sub(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_sub(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn mul(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_mul(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn div(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_div(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn lt(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_lt(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn gt(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_gt(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn le(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_le(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn ge(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_ge(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn eq(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_eq(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn neq(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_neq(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn and_(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_and(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn or_(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_or(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn not(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_not(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn bitNot(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_bitnot(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn bitAnd(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_bitand(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn bitOr(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_bitor(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn bitXor(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_bitxor(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn bitShiftL(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_bitshiftl(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn bitShiftR(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_bitshiftr(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn cast(allocator: std.mem.Allocator, in: *ArrayFireArray, dtype: DType) !*ArrayFireArray {
    const af_type = ztToAfType(dtype);
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_cast(&arr, in.af_array, af_type.value()), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn minOf(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_minof(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn maxOf(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_maxof(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn clamp(allocator: std.mem.Allocator, in: *ArrayFireArray, lo: *ArrayFireArray, hi: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_clamp(&arr, in.af_array, lo.af_array, hi.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn rem(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_rem(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn mod(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_mod(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn abs(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_abs(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn arg(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_fabs(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn sign(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_sign(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn round(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_round(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn trunc(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_trunc(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn floor(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_floor(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn ceil(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_ceil(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn hypot(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_hypot(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn sin(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_sin(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn cos(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_cos(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn tan(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_tan(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn asin(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_asin(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn acos(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_acos(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn atan(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_atan(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn atan2(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_atan2(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn sinh(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_sinh(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn cosh(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_cosh(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn tanh(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_tanh(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn asinh(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_asinh(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn acosh(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_acosh(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn atanh(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_atanh(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn cplx(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_cplx(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn cplx2(allocator: std.mem.Allocator, real_: *ArrayFireArray, imag_: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_cplx2(&arr, real_.af_array, imag_.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn real(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_real(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn imag(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_imag(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn conjg(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_conjg(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn root(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_root(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn pow(allocator: std.mem.Allocator, lhs: *ArrayFireArray, rhs: *ArrayFireArray, batch: bool) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_pow(&arr, lhs.af_array, rhs.af_array, batch), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn pow2(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_pow2(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn sigmoid(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_sigmoid(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn exp(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_exp(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn expm1(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_expm1(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn erf(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_erf(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn erfc(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_erfc(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn log(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_log(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn log1p(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_log1p(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn log10(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_log10(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn log2(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_log2(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn sqrt(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_sqrt(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn rsqrt(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_rsqrt(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn cbrt(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_cbrt(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn factorial(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_factorial(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn tgamma(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_tgamma(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn lgamma(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_lgamma(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn isZero(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_iszero(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn isInf(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_isinf(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

pub inline fn isNan(allocator: std.mem.Allocator, in: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_isnan(&arr, in.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

/// Initializes an ArrayFireArray from device memory.
pub inline fn deviceArray(allocator: std.mem.Allocator, data: ?*anyopaque, ndims: usize, dims: AfDims4, dtype: DType) !af.af_array {
    var af_type = ztToAfType(dtype);
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_device_array(&arr, data, @intCast(ndims), &dims.dims, af_type.value()), @src());
    return ArrayFireArray.init(allocator, arr);
}

/// Lock the device buffer in the memory manager.
pub inline fn lockDevicePtr(arr: *ArrayFireArray) !void {
    try AF_CHECK(af.af_lock_device_ptr(arr.af_array), @src());
}

/// Unlock device buffer in the memory manager.
pub inline fn unlockDevicePtr(arr: *ArrayFireArray) !void {
    try AF_CHECK(af.af_unlock_device_ptr(arr.af_array), @src());
}

/// Lock the device buffer in the memory manager.
pub inline fn lockArray(arr: *ArrayFireArray) !void {
    try AF_CHECK(af.af_lock_array(arr.af_array), @src());
}

/// Unlock device buffer in the memory manager.
pub inline fn unlockArray(arr: *ArrayFireArray) !void {
    try AF_CHECK(af.af_unlock_array(arr.af_array), @src());
}

/// Query if the array has been locked by the user.
pub inline fn isLockedArray(arr: *ArrayFireArray) !bool {
    var is_locked: bool = undefined;
    try AF_CHECK(af.af_is_locked_array(&is_locked, arr.af_array), @src());
    return is_locked;
}

/// Get the device pointer and lock the buffer in memory manager.
pub inline fn getDevicePtr(arr: *ArrayFireArray) !?*anyopaque {
    var ptr: ?*anyopaque = undefined;
    try AF_CHECK(af.af_get_device_ptr(&ptr, arr.af_array), @src());
    return ptr;
}

/// Lookup the values of input array based on sequences.
pub inline fn index(allocator: std.mem.Allocator, in: *ArrayFireArray, ndims: usize, idx: *const af.af_seq) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_index(&arr, in.af_array, @intCast(ndims), idx), @src());
    return ArrayFireArray.init(allocator, arr);
}

/// Lookup the values of an input array by indexing with another array.
pub inline fn lookup(allocator: std.mem.Allocator, in: *ArrayFireArray, indices: *ArrayFireArray, dim: usize) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_lookup(&arr, in.af_array, indices.af_array, @intCast(dim)), @src());
    return ArrayFireArray.init(allocator, arr);
}

/// Copy and write values in the locations specified by the sequences.
pub inline fn assignSeq(allocator: std.mem.Allocator, lhs: *ArrayFireArray, ndims: usize, indices: *const af.af_seq, rhs: *ArrayFireArray) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_assign_seq(&arr, lhs.af_array, @intCast(ndims), indices, rhs.af_array), @src());
    return ArrayFireArray.init(allocator, arr);
}

/// Indexing an array using `af.af_seq`, or `af.af_array`.
pub inline fn indexGen(allocator: std.mem.Allocator, in: *ArrayFireArray, ndims: i64, indices: *const af.af_index_t) !*ArrayFireArray {
    var arr: af.af_array = undefined;
    try AF_CHECK(af.af_index_gen(&arr, in.af_array, @intCast(ndims), indices), @src());
    return ArrayFireArray.init(allocator, arr);
}
