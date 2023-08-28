const std = @import("std");
const af_utils = @import("../Utils.zig");
const af = @import("../../../../backends/ArrayFire.zig");
const af_types = @import("types.zig");
const zt_types = @import("../../../Types.zig");
const ops = @import("array_ops.zig");

const DType = zt_types.DType;
const BinaryOp = af_types.BinaryOp;
const AfDims4 = af_utils.AfDims4;
const AF_CHECK = af_utils.AF_CHECK;
const ztToAfType = af_utils.ztToAfType;

pub const ArrayFireArray = struct {
    const Self = @This();

    af_array: af.af_array,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, arr: af.af_array) !*Self {
        var self = try allocator.create(Self);
        self.* = .{ .af_array = arr, .allocator = allocator };
        return self;
    }

    pub fn initFromDevicePtr(allocator: std.mem.Allocator, data: ?*anyopaque, ndims: usize, dims: AfDims4, dtype: DType) !*Self {
        return ops.deviceArray(allocator, data, ndims, dims, dtype);
    }

    pub fn deinit(self: *Self) void {
        AF_CHECK(af.af_release_array(self.af_array), @src()) catch unreachable;
        self.allocator.destroy(self);
    }

    pub fn sum(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.sum(allocator, self, dim);
    }

    pub fn sumNan(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.sumNan(allocator, self, dim);
    }

    // TODO: pub fn sumByKey

    // TODO: pub fn sumByKeyNan

    pub fn product(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.product(allocator, self, dim);
    }

    pub fn productNan(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.productNan(allocator, self, dim);
    }

    // TODO: pub fn productByKey

    // TODO: pub fn productByKeyNan

    pub fn min(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.min(allocator, self, dim);
    }

    // TODO: pub fn minByKey

    pub fn max(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.max(allocator, self, dim);
    }

    // TODO: pub fn maxByKey

    // TODO: pub fn maxRagged

    pub fn allTrue(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.allTrue(allocator, self, dim);
    }

    // TODO: pub fn allTrueByKey

    pub fn anyTrue(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.anyTrue(allocator, self, dim);
    }

    // TODO: pub fn anyTrueByKey

    pub fn count(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.count(allocator, self, dim);
    }

    // TODO: pub fn countByKey

    // TODO: pub fn sumAll

    // TODO: pub fn sumNanAll

    // TODO: pub fn productAll

    // TODO: pub fn productNanAll

    // TODO: pub fn minAll

    // TODO: pub fn maxAll

    // TODO: pub fn allTrueAll

    // TODO: pub fn anyTrueAll

    // TODO: pub fn countAll

    // TODO: pub fn imin

    // TODO: pub fn imax

    // TODO: pub fn iminAll

    // TODO: pub fn imaxAll

    pub fn accum(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.accum(allocator, self, dim);
    }

    pub fn scan(self: *Self, allocator: std.mem.Allocator, dim: usize, op: BinaryOp, inclusive_scan: bool) !*Self {
        return ops.scan(allocator, self, dim, op, inclusive_scan);
    }

    // TODO: pub fn scanByKey

    pub fn where(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.where(allocator, self);
    }

    pub fn diff1(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.diff1(allocator, self, dim);
    }

    pub fn diff2(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return ops.diff2(allocator, self, dim);
    }

    pub fn sort(self: *Self, allocator: std.mem.Allocator, dim: usize, is_ascending: bool) !*Self {
        return ops.sort(allocator, self, dim, is_ascending);
    }

    // TODO: pub fn sortIndex

    // TODO: pub fn sortByKey

    pub fn setUnique(self: *Self, allocator: std.mem.Allocator, is_sorted: bool) !*Self {
        return ops.setUnique(allocator, self, is_sorted);
    }

    pub fn setUnion(self: *Self, allocator: std.mem.Allocator, second: *Self, is_unique: bool) !*Self {
        return ops.setUnion(allocator, self, second, is_unique);
    }

    pub fn setIntersect(self: *Self, allocator: std.mem.Allocator, second: *Self, is_unique: bool) !*Self {
        return ops.setIntersect(allocator, self, second, is_unique);
    }

    pub fn add(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.add(allocator, self, rhs, batch);
    }

    pub fn sub(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.sub(allocator, self, rhs, batch);
    }

    pub fn mul(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.mul(allocator, self, rhs, batch);
    }

    pub fn div(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.div(allocator, self, rhs, batch);
    }

    pub fn lt(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.lt(allocator, self, rhs, batch);
    }

    pub fn gt(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.gt(allocator, self, rhs, batch);
    }

    pub fn le(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.le(allocator, self, rhs, batch);
    }

    pub fn ge(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.ge(allocator, self, rhs, batch);
    }

    pub fn eq(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.eq(allocator, self, rhs, batch);
    }

    pub fn neq(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.neq(allocator, self, rhs, batch);
    }

    pub fn and_(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.and_(allocator, self, rhs, batch);
    }

    pub fn or_(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.or_(allocator, self, rhs, batch);
    }

    pub fn not(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.not(allocator, self);
    }

    pub fn bitNot(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.bitNot(allocator, self);
    }

    pub fn bitAnd(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.bitAnd(allocator, self, rhs, batch);
    }

    pub fn bitOr(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.bitOr(allocator, self, rhs, batch);
    }

    pub fn bitXor(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.bitXor(allocator, self, rhs, batch);
    }

    pub fn bitShiftL(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.bitShiftL(allocator, self, rhs, batch);
    }

    pub fn bitShiftR(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.bitShiftR(allocator, self, rhs, batch);
    }

    pub fn cast(self: *Self, allocator: std.mem.Allocator, dtype: DType) !*Self {
        return ops.cast(allocator, self, dtype);
    }

    pub fn minOf(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.minOf(allocator, self, rhs, batch);
    }

    pub fn maxOf(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.maxOf(allocator, self, rhs, batch);
    }

    pub fn clamp(self: *Self, allocator: std.mem.Allocator, lo: *Self, hi: *Self, batch: bool) !*Self {
        return ops.clamp(allocator, self, lo, hi, batch);
    }

    pub fn rem(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.rem(allocator, self, rhs, batch);
    }

    pub fn mod(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.mod(allocator, self, rhs, batch);
    }

    pub fn abs(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.abs(allocator, self);
    }

    pub fn arg(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.arg(allocator, self);
    }

    pub fn sign(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.sign(allocator, self);
    }

    pub fn round(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.round(allocator, self);
    }

    pub fn trunc(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.trunc(allocator, self);
    }

    pub fn floor(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.floor(allocator, self);
    }

    pub fn ceil(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.ceil(allocator, self);
    }

    pub fn hypot(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.hypot(allocator, self, rhs, batch);
    }

    pub fn sin(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.sin(allocator, self);
    }

    pub fn cos(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.cos(allocator, self);
    }

    pub fn tan(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.tan(allocator, self);
    }

    pub fn asin(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.asin(allocator, self);
    }

    pub fn acos(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.acos(allocator, self);
    }

    pub fn atan(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.atan(allocator, self);
    }

    pub fn atan2(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.atan2(allocator, self, rhs, batch);
    }

    pub fn sinh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.sinh(allocator, self);
    }

    pub fn cosh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.cosh(allocator, self);
    }

    pub fn tanh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.tanh(allocator, self);
    }

    pub fn asinh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.asinh(allocator, self);
    }

    pub fn acosh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.acosh(allocator, self);
    }

    pub fn atanh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.atanh(allocator, self);
    }

    pub fn cplx(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.cplx(allocator, self);
    }

    pub fn cplx2(self: *Self, allocator: std.mem.Allocator, imag_: *Self, batch: bool) !*Self {
        return ops.cplx2(allocator, self, imag_, batch);
    }

    pub fn real(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.real(allocator, self);
    }

    pub fn imag(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.imag(allocator, self);
    }

    pub fn conjg(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.conjg(allocator, self);
    }

    pub fn root(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.root(allocator, self, rhs, batch);
    }

    pub fn pow(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return ops.pow(allocator, self, rhs, batch);
    }

    pub fn pow2(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.pow2(allocator, self);
    }

    pub fn sigmoid(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.sigmoid(allocator, self);
    }

    pub fn exp(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.exp(allocator, self);
    }

    pub fn expm1(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.expm1(allocator, self);
    }

    pub fn erf(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.erf(allocator, self);
    }

    pub fn erfc(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.erfc(allocator, self);
    }

    pub fn log(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.log(allocator, self);
    }

    pub fn log1p(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.log1p(allocator, self);
    }

    pub fn log10(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.log10(allocator, self);
    }

    pub fn log2(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.log2(allocator, self);
    }

    pub fn sqrt(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.sqrt(allocator, self);
    }

    pub fn rsqrt(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.rsqrt(allocator, self);
    }

    pub fn cbrt(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.cbrt(allocator, self);
    }

    pub fn factorial(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.factorial(allocator, self);
    }

    pub fn tgamma(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.tgamma(allocator, self);
    }

    pub fn lgamma(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.lgamma(allocator, self);
    }

    pub fn isZero(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.isZero(allocator, self);
    }

    pub fn isInf(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.isInf(allocator, self);
    }

    pub fn isNaN(self: *Self, allocator: std.mem.Allocator) !*Self {
        return ops.isNaN(allocator, self);
    }

    pub fn lockDevicePtr(self: *Self) !void {
        return ops.lockDevicePtr(self);
    }

    pub fn unlockDevicePtr(self: *Self) !void {
        return ops.unlockDevicePtr(self);
    }

    pub fn lockArray(self: *Self) !void {
        return ops.lockArray(self);
    }

    pub fn unlockArray(self: *Self) !void {
        return ops.unlockArray(self);
    }

    pub fn isLockedArray(self: *Self) !bool {
        return ops.isLockedArray(self);
    }

    pub fn getDevicePtr(self: *Self) !?*anyopaque {
        return ops.getDevicePtr(self);
    }

    pub fn index(self: *Self, allocator: std.mem.Allocator, ndims: usize, idx: *const af.af_seq) !*Self {
        return ops.index(allocator, self, ndims, idx);
    }

    pub fn lookup(self: *Self, allocator: std.mem.Allocator, indices: *Self, dim: usize) !*Self {
        return ops.lookup(allocator, self, indices, dim);
    }

    pub fn assignSeq(self: *Self, allocator: std.mem.Allocator, ndims: usize, indices: *const af.af_seq, rhs: *Self) !*Self {
        return ops.assignSeq(allocator, self, ndims, indices, rhs);
    }

    pub fn indexGen(self: *Self, allocator: std.mem.Allocator, ndims: i64, indices: *const af.af_index_t) !*Self {
        return ops.indexGen(allocator, self, ndims, indices);
    }
};
