const std = @import("std");
const af = @import("ArrayFire.zig");
const zt_types = @import("../../tensor/Types.zig");
const zt_shape = @import("../../tensor/Shape.zig");

const Location = @import("../../tensor/TensorBase.zig").Location;
const Shape = zt_shape.Shape;
const DType = zt_types.DType;

/// Wraps `af.af_array`, the ArrayFire multi dimensional data container,
/// as a zig struct to simplify calling into the ArrayFire C API.
pub const Array = struct {
    const Self = @This();

    array_: af.af_array,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, arr: af.af_array) !*Self {
        var self = try allocator.create(Self);
        self.* = .{ .array_ = arr, .allocator = allocator };
        return self;
    }

    pub fn initFromDevicePtr(allocator: std.mem.Allocator, data: ?*anyopaque, ndims: usize, dims: af.Dims4, dtype: DType) !*Self {
        return af.ops.deviceArray(allocator, data, ndims, dims, dtype);
    }

    pub fn initFromPtr(allocator: std.mem.Allocator, shape: *const Shape, ptr: ?*const anyopaque, dtype: DType) !*Self {
        return af.ops.createArray(allocator, shape, ptr, dtype);
    }

    pub fn deinit(self: *Self) void {
        self.releaseArray() catch unreachable;
        self.allocator.destroy(self);
    }

    pub fn sum(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return af.ops.sum(allocator, self, dim);
    }

    pub fn sumNan(self: *Self, allocator: std.mem.Allocator, dim: usize, nanval: f64) !*Self {
        return af.ops.sumNan(allocator, self, dim, nanval);
    }

    // TODO: pub fn sumByKey

    // TODO: pub fn sumByKeyNan

    pub fn product(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return af.ops.product(allocator, self, dim);
    }

    pub fn productNan(self: *Self, allocator: std.mem.Allocator, dim: usize, nanval: f64) !*Self {
        return af.ops.productNan(allocator, self, dim, nanval);
    }

    // TODO: pub fn productByKey

    // TODO: pub fn productByKeyNan

    pub fn min(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return af.ops.min(allocator, self, dim);
    }

    // TODO: pub fn minByKey

    pub fn max(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return af.ops.max(allocator, self, dim);
    }

    // TODO: pub fn maxByKey

    // TODO: pub fn maxRagged

    pub fn allTrue(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return af.ops.allTrue(allocator, self, dim);
    }

    // TODO: pub fn allTrueByKey

    pub fn anyTrue(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return af.ops.anyTrue(allocator, self, dim);
    }

    // TODO: pub fn anyTrueByKey

    pub fn count(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return af.ops.count(allocator, self, dim);
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
        return af.ops.accum(allocator, self, dim);
    }

    pub fn scan(self: *Self, allocator: std.mem.Allocator, dim: usize, op: af.BinaryOp, inclusive_scan: bool) !*Self {
        return af.ops.scan(allocator, self, dim, op, inclusive_scan);
    }

    // TODO: pub fn scanByKey

    pub fn where(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.where(allocator, self);
    }

    pub fn diff1(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return af.ops.diff1(allocator, self, dim);
    }

    pub fn diff2(self: *Self, allocator: std.mem.Allocator, dim: usize) !*Self {
        return af.ops.diff2(allocator, self, dim);
    }

    pub fn sort(self: *Self, allocator: std.mem.Allocator, dim: usize, is_ascending: bool) !*Self {
        return af.ops.sort(allocator, self, dim, is_ascending);
    }

    // TODO: pub fn sortIndex

    // TODO: pub fn sortByKey

    pub fn setUnique(self: *Self, allocator: std.mem.Allocator, is_sorted: bool) !*Self {
        return af.ops.setUnique(allocator, self, is_sorted);
    }

    pub fn setUnion(self: *Self, allocator: std.mem.Allocator, second: *Self, is_unique: bool) !*Self {
        return af.ops.setUnion(allocator, self, second, is_unique);
    }

    pub fn setIntersect(self: *Self, allocator: std.mem.Allocator, second: *Self, is_unique: bool) !*Self {
        return af.ops.setIntersect(allocator, self, second, is_unique);
    }

    pub fn add(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.add(allocator, self, rhs, batch);
    }

    pub fn sub(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.sub(allocator, self, rhs, batch);
    }

    pub fn mul(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.mul(allocator, self, rhs, batch);
    }

    pub fn div(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.div(allocator, self, rhs, batch);
    }

    pub fn lt(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.lt(allocator, self, rhs, batch);
    }

    pub fn gt(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.gt(allocator, self, rhs, batch);
    }

    pub fn le(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.le(allocator, self, rhs, batch);
    }

    pub fn ge(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.ge(allocator, self, rhs, batch);
    }

    pub fn eq(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.eq(allocator, self, rhs, batch);
    }

    pub fn neq(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.neq(allocator, self, rhs, batch);
    }

    pub fn and_(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.and_(allocator, self, rhs, batch);
    }

    pub fn or_(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.or_(allocator, self, rhs, batch);
    }

    pub fn not(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.not(allocator, self);
    }

    pub fn bitNot(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.bitNot(allocator, self);
    }

    pub fn bitAnd(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.bitAnd(allocator, self, rhs, batch);
    }

    pub fn bitOr(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.bitOr(allocator, self, rhs, batch);
    }

    pub fn bitXor(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.bitXor(allocator, self, rhs, batch);
    }

    pub fn bitShiftL(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.bitShiftL(allocator, self, rhs, batch);
    }

    pub fn bitShiftR(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.bitShiftR(allocator, self, rhs, batch);
    }

    pub fn cast(self: *Self, allocator: std.mem.Allocator, dtype: DType) !*Self {
        return af.ops.cast(allocator, self, dtype);
    }

    pub fn minOf(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.minOf(allocator, self, rhs, batch);
    }

    pub fn maxOf(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.maxOf(allocator, self, rhs, batch);
    }

    pub fn clamp(self: *Self, allocator: std.mem.Allocator, lo: *Self, hi: *Self, batch: bool) !*Self {
        return af.ops.clamp(allocator, self, lo, hi, batch);
    }

    pub fn rem(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.rem(allocator, self, rhs, batch);
    }

    pub fn mod(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.mod(allocator, self, rhs, batch);
    }

    pub fn abs(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.abs(allocator, self);
    }

    pub fn arg(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.arg(allocator, self);
    }

    pub fn sign(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sign(allocator, self);
    }

    pub fn round(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.round(allocator, self);
    }

    pub fn trunc(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.trunc(allocator, self);
    }

    pub fn floor(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.floor(allocator, self);
    }

    pub fn ceil(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.ceil(allocator, self);
    }

    pub fn hypot(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.hypot(allocator, self, rhs, batch);
    }

    pub fn sin(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sin(allocator, self);
    }

    pub fn cos(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.cos(allocator, self);
    }

    pub fn tan(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.tan(allocator, self);
    }

    pub fn asin(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.asin(allocator, self);
    }

    pub fn acos(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.acos(allocator, self);
    }

    pub fn atan(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.atan(allocator, self);
    }

    pub fn atan2(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.atan2(allocator, self, rhs, batch);
    }

    pub fn sinh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sinh(allocator, self);
    }

    pub fn cosh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.cosh(allocator, self);
    }

    pub fn tanh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.tanh(allocator, self);
    }

    pub fn asinh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.asinh(allocator, self);
    }

    pub fn acosh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.acosh(allocator, self);
    }

    pub fn atanh(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.atanh(allocator, self);
    }

    pub fn cplx(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.cplx(allocator, self);
    }

    pub fn cplx2(self: *Self, allocator: std.mem.Allocator, imag_: *Self, batch: bool) !*Self {
        return af.ops.cplx2(allocator, self, imag_, batch);
    }

    pub fn real(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.real(allocator, self);
    }

    pub fn imag(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.imag(allocator, self);
    }

    pub fn conjg(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.conjg(allocator, self);
    }

    pub fn root(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.root(allocator, self, rhs, batch);
    }

    pub fn pow(self: *Self, allocator: std.mem.Allocator, rhs: *Self, batch: bool) !*Self {
        return af.ops.pow(allocator, self, rhs, batch);
    }

    pub fn pow2(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.pow2(allocator, self);
    }

    pub fn sigmoid(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sigmoid(allocator, self);
    }

    pub fn exp(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.exp(allocator, self);
    }

    pub fn expm1(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.expm1(allocator, self);
    }

    pub fn erf(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.erf(allocator, self);
    }

    pub fn erfc(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.erfc(allocator, self);
    }

    pub fn log(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.log(allocator, self);
    }

    pub fn log1p(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.log1p(allocator, self);
    }

    pub fn log10(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.log10(allocator, self);
    }

    pub fn log2(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.log2(allocator, self);
    }

    pub fn sqrt(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.sqrt(allocator, self);
    }

    pub fn rsqrt(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.rsqrt(allocator, self);
    }

    pub fn cbrt(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.cbrt(allocator, self);
    }

    pub fn factorial(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.factorial(allocator, self);
    }

    pub fn tgamma(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.tgamma(allocator, self);
    }

    pub fn lgamma(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.lgamma(allocator, self);
    }

    pub fn isZero(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.isZero(allocator, self);
    }

    pub fn isInf(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.isInf(allocator, self);
    }

    pub fn isNaN(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.isNaN(allocator, self);
    }

    pub fn lockDevicePtr(self: *Self) !void {
        return af.ops.lockDevicePtr(self);
    }

    pub fn unlockDevicePtr(self: *Self) !void {
        return af.ops.unlockDevicePtr(self);
    }

    pub fn lockArray(self: *Self) !void {
        return af.ops.lockArray(self);
    }

    pub fn unlockArray(self: *Self) !void {
        return af.ops.unlockArray(self);
    }

    pub fn isLockedArray(self: *Self) !bool {
        return af.ops.isLockedArray(self);
    }

    pub fn getDevicePtr(self: *Self) !?*anyopaque {
        return af.ops.getDevicePtr(self);
    }

    pub fn index(self: *Self, allocator: std.mem.Allocator, ndims: usize, idx: *const af.af_seq) !*Self {
        return af.ops.index(allocator, self, ndims, idx);
    }

    pub fn lookup(self: *Self, allocator: std.mem.Allocator, indices: *Self, dim: usize) !*Self {
        return af.ops.lookup(allocator, self, indices, dim);
    }

    pub fn assignSeq(self: *Self, allocator: std.mem.Allocator, ndims: usize, indices: *const af.af_seq, rhs: *Self) !*Self {
        return af.ops.assignSeq(allocator, self, ndims, indices, rhs);
    }

    pub fn indexGen(self: *Self, allocator: std.mem.Allocator, ndims: i64, indices: *const af.af_index_t) !*Self {
        return af.ops.indexGen(allocator, self, ndims, indices);
    }

    /// Deep copy the underlying `af.af_array` to another.
    pub fn copyArray(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.copyArray(allocator, self);
    }

    /// Copy data from a C pointer (host/device) to the underlying `af.af_array`.
    pub fn writeArray(self: *Self, data: ?*const anyopaque, bytes: usize, src: Location) !void {
        return af.ops.writeArray(self, data, bytes, src);
    }

    /// Copy data from the underlying `af.af_array` to a C pointer.
    pub fn getDataPtr(self: *Self, data: ?*anyopaque) !void {
        return af.ops.getDataPtr(self, data);
    }

    /// Reduce the reference count of the underlying `af.af_array`.
    pub fn releaseArray(self: *Self) !void {
        return af.ops.releaseArray(self);
    }

    /// Returns the reference count of the underlying `af.af_array`.
    pub fn getDataRefCount(self: *Self) !i32 {
        return af.ops.getDataRefCount(self);
    }

    /// Evaluate any expressions in the underlying `af.af_array`.
    pub fn eval(self: *Self) !void {
        return af.ops.eval(self);
    }

    /// Returns the total number of elements across all dimensions of the underlying `af.af_array`.
    pub fn getElements(self: *Self) !i64 {
        return af.ops.getElements(self);
    }

    /// Returns the `DType` of the underlying `af.af_array`.
    pub fn getType(self: *Self) !DType {
        return af.ops.getType(self);
    }

    /// Returns the dimensions of the underlying `af.af_array`.
    pub fn getDims(self: *Self) !af.Dims4 {
        return af.ops.getDims(self);
    }

    /// Returns the number of dimensions of the underlying `af.af_array`.
    pub fn getNumDims(self: *Self) !u32 {
        return af.ops.getNumDims(self);
    }

    /// Check if the underlying `af.af_array` is empty.
    pub fn isEmpty(self: *Self) !bool {
        return af.ops.isEmpty(self);
    }

    /// Check if the underlying `af.af_array` is a scalar.
    pub fn isScalar(self: *Self) !bool {
        return af.ops.isScalar(self);
    }

    /// Check if the underlying `af.af_array` is a row vector.
    pub fn isRow(self: *Self) !bool {
        return af.ops.isRow(self);
    }

    /// Check if the underlying `af.af_array` is a column vector.
    pub fn isColumn(self: *Self) !bool {
        return af.ops.isColumn(self);
    }

    /// Check if the underlying `af.af_array` is a vector.
    pub fn isVector(self: *Self) !bool {
        return af.ops.isVector(self);
    }

    /// Check if the underlying `af.af_array` is a complex type.
    pub fn isComplex(self: *Self) !bool {
        return af.ops.isComplex(self);
    }

    /// Check if the underlying `af.af_array` is a real type.
    pub fn isReal(self: *Self) !bool {
        return af.ops.isReal(self);
    }

    /// Check if the underlying `af.af_array` is double precision type.
    pub fn isDouble(self: *Self) !bool {
        return af.ops.isDouble(self);
    }

    /// Check if the underlying `af.af_array` is single precision type.
    pub fn isSingle(self: *Self) !bool {
        return af.ops.isSingle(self);
    }

    /// Check if the underlying `af.af_array` is a 16 bit floating point type.
    pub fn isHalf(self: *Self) !bool {
        return af.ops.isHalf(self);
    }

    /// Check if the underlying `af.af_array` is a real floating point type.
    pub fn isRealFloating(self: *Self) !bool {
        return af.ops.isRealFloating(self);
    }

    /// Check if the underlying `af.af_array` is floating precision type.
    pub fn isFloating(self: *Self) !bool {
        return af.ops.isFloating(self);
    }

    /// Check if the underlying `af.af_array` is integer type.
    pub fn isInteger(self: *Self) !bool {
        return af.ops.isInteger(self);
    }

    /// Check if the underlying `af.af_array` is bool type.
    pub fn isBool(self: *Self) !bool {
        return af.ops.isBool(self);
    }

    /// Check if the underlying `af.af_array` is sparse.
    pub fn isSparse(self: *Self) !bool {
        return af.ops.isSparse(self);
    }

    /// Returns the first element from the underlying `af.af_array`.
    pub fn getScalar(self: *Self, comptime T: type) !T {
        return af.ops.getScalar(self, T);
    }
};
