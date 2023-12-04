const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;

/// An implementation of edit distance meter, which measures the edit distance
/// between targets and predictions made by the model.
pub const EditDistanceMeter = struct {
    n: i64 = 0,
    ndel: i64 = 0,
    nins: i64 = 0,
    nsub: i64 = 0,

    /// A structure storing number of different type of errors
    /// when computing edit distance.
    pub const ErrorState = struct {
        /// Number of deletion error
        ndel: i64 = 0,
        /// Number of insertion error
        nins: i64 = 0,
        /// Number of substitution error
        nsub: i64 = 0,

        /// Sums up all the errors.
        pub fn sum(self: *const ErrorState) i64 {
            return self.ndel + self.nins + self.nsub;
        }
    };

    /// Sets all the counters to 0.
    pub fn reset(self: *EditDistanceMeter) void {
        self.n = 0;
        self.ndel = 0;
        self.nins = 0;
        self.nsub = 0;
    }

    /// Computes edit distance between two tensors `output` and `target`
    /// and updates the counters.
    pub fn addTensor(self: *EditDistanceMeter, allocator: std.mem.Allocator, output: Tensor, target: Tensor) !void {
        if (try target.ndim(allocator) != 1) {
            std.debug.print("target must be 1-dimensional for EditDistanceMeter\n", .{});
            return error.InvalidTargetNdims;
        }
        if (try output.ndim(allocator) != 1) {
            std.debug.print("output must be 1-dimensional for EditDistanceMeter\n", .{});
            return error.InvalidTargetNdims;
        }

        const in1_raw = try output.allocHost(allocator, i32);
        defer allocator.free(in1_raw);
        const in2_raw = try target.allocHost(allocator, i32);
        defer allocator.free(in2_raw);

        const err_state = try levensteinDistance(allocator, i32, in1_raw, in2_raw);
        self.addErrorState(&err_state, try target.dim(allocator, 0));
    }

    /// Updates all the counters with an `ErrorState`.
    pub fn addErrorState(self: *EditDistanceMeter, es: *const ErrorState, n: i64) void {
        self.add(n, es.ndel, es.nins, es.nsub);
    }

    //// Updates all the counters with inputs sharing the same meaning.
    pub fn add(self: *EditDistanceMeter, n: i64, ndel: i64, nins: i64, nsub: i64) void {
        self.n += n;
        self.ndel += ndel;
        self.nins += nins;
        self.nsub += nsub;
    }

    /// Computes edit distance between two slices `output` and `target`,
    /// and updates the counters.
    pub fn addSlice(self: *EditDistanceMeter, allocator: std.mem.Allocator, comptime T: type, output: []T, target: []T) !void {
        const err_state = try levensteinDistance(allocator, T, output, target);
        self.addErrorState(&err_state, target.len);
    }

    /// Returns an array of five values: edit distance, total length,
    /// number of deletions, number of insertions, number of substitutions.
    pub fn value(self: *const EditDistanceMeter) [5]i64 {
        return [_]i64{ self.sumErr(), self.n, self.ndel, self.nins, self.nsub };
    }

    fn sumErr(self: *const EditDistanceMeter) i64 {
        return self.ndel + self.nins + self.nsub;
    }

    /// Returns an array of five values: error rate, total length,
    /// deletion rate, insertion rate, substitution rate.
    pub fn errorRate(self: *const EditDistanceMeter) [5]f64 {
        const val: f64 = undefined;
        const val_del: f64 = undefined;
        const val_ins: f64 = undefined;
        const val_sub: f64 = undefined;
        if (self.n > 0) {
            val = @as(f64, @floatFromInt(self.sumErr() * 100)) / @as(f64, @floatFromInt(self.n));
            val_del = @as(f64, @floatFromInt(self.ndel * 100)) / @as(f64, @floatFromInt(self.n));
            val_ins = @as(f64, @floatFromInt(self.nins * 100)) / @as(f64, @floatFromInt(self.n));
            val_sub = @as(f64, @floatFromInt(self.nsub * 100)) / @as(f64, @floatFromInt(self.n));
        } else {
            val = if (self.sumErr() > 0) std.math.inf(f64) else 0;
            val_del = if (self.ndel > 0) std.math.inf(f64) else 0;
            val_ins = if (self.nins > 0) std.math.inf(f64) else 0;
            val_sub = if (self.nsub > 0) std.math.inf(f64) else 0;
        }
        return [5]f64{ val, @floatFromInt(self.n), val_del, val_ins, val_sub };
    }

    fn levensteinDistance(_: *const EditDistanceMeter, allocator: std.mem.Allocator, comptime T: type, in1_begin: []T, in2_begin: []T) !ErrorState {
        var column = try allocator.alloc(ErrorState, in1_begin.len + 1);
        defer allocator.free(column);
        @memset(column, .{});
        var i: usize = 0;
        while (i <= in2_begin.len) : (i += 1) {
            column[i].nins = @intCast(i);
        }

        var curin2: usize = 0;
        var x: usize = 1;
        while (x <= in2_begin.len) : (x += 1) {
            var last_diagonal = column[0];
            column[0].ndel = @intCast(x);
            var curin1: usize = 0;
            var y: usize = 1;
            while (y <= in1_begin.len) : (y += 1) {
                const old_diagonal = column[y];
                var possibilities = [_]i64{ column[y].sum() + 1, column[y - 1].sum() + 1, last_diagonal.sum() + (if (in1_begin[curin1] == in2_begin[curin2]) 0 else 1) };
                const min_it = std.mem.min(i64, &possibilities);
                var distance: usize = 0;
                for (possibilities) |p| {
                    if (p == min_it) {
                        break;
                    }
                    distance += 1;
                }
                if (distance == 0) {
                    // deletion error
                    column[y].ndel += 1;
                } else if (distance == 1) {
                    // insertion error
                    column[y] = column[y - 1];
                    column[y].nins += 1;
                } else {
                    column[y] = last_diagonal;
                    if (in1_begin[curin1] != in2_begin[curin2]) {
                        // substitution error
                        column[y].nsub += 1;
                    }
                }

                last_diagonal = old_diagonal;
                curin1 += 1;
            }
            curin2 += 1;
        }

        return column[in1_begin.len];
    }
};
