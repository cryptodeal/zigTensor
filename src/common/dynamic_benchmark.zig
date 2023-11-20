const std = @import("std");
const zigrc = @import("zigrc");
const zt = @import("../zt.zig");

const assert = std.debug.assert;

pub const DynamicBenchmarkOptionsBase = struct {
    const Self = @This();
    // The type erased pointer to the DynamicBenchmarkOptions implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
        timingsComplete: *const fn (ctx: *anyopaque) bool,
        accumulateTimeToCurrentOption: *const fn (ctx: *anyopaque, time: f64, increment_count: bool) anyerror!void,
        reset: *const fn (ctx: *anyopaque) anyerror!void,
    };

    /// Free all associated memory.
    pub fn deinit(self: Self) void {
        self.vtable.deinit(self.ptr);
    }

    pub fn timingsComplete(self: *const Self) bool {
        return self.vtable.timingsComplete(self.ptr);
    }

    pub fn accumulateTimeToCurrentOption(self: *const Self, time: f64, increment_count: bool) !void {
        return self.vtable.accumulateTimeToCurrentOption(self.ptr, time, increment_count);
    }

    pub fn reset(self: *const Self) !void {
        return self.vtable.reset(self.ptr);
    }

    pub fn init(opt_impl: anytype) Self {
        const Ptr = @TypeOf(opt_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }

            fn timingsComplete(ctx: *anyopaque) bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.timingsComplete();
            }

            fn accumulateTimeToCurrentOption(ctx: *anyopaque, time: f64, increment_count: bool) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.accumulateTimeToCurrentOption(time, increment_count);
            }

            fn reset(ctx: *anyopaque) !void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.reset();
            }
        };
        return .{
            .ptr = opt_impl,
            .vtable = &.{
                .deinit = impl.deinit,
                .timingsComplete = impl.timingsComplete,
                .accumulateTimeToCurrentOption = impl.accumulateTimeToCurrentOption,
                .reset = impl.reset,
            },
        };
    }
};

pub fn DynamicBenchmarkOptions(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        options_: []T,
        bench_count_: usize = 0,
        timings_complete: bool = false,
        current_option_idx: usize = 0, // first option is the default
        // Number of times the option at each index has been timed
        counts: std.AutoHashMap(usize, usize),
        // Accumulated times for each option
        times: std.AutoHashMap(usize, f64),

        /// Constructs an instance given a slice of options of specified type. The
        /// options are assumed to be distinct since benchmarks options are
        /// determined by index.
        pub fn initFromSlice(allocator: std.mem.Allocator, options: []const T, bench_count: usize) !*Self {
            var self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .options_ = try allocator.alloc(T, options.len),
                .bench_count_ = bench_count,
                .counts = std.AutoHashMap(usize, usize).init(allocator),
                .times = std.AutoHashMap(usize, f64).init(allocator),
            };
            @memcpy(self.options_, options);
            try self.reset();
            return self;
        }

        /// Constructs an instance given a set of options.
        pub fn initFromIterator(allocator: std.mem.Allocator, options: std.AutoHashMap(T, void).KeyIterator, bench_count: usize) !*Self {
            var self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .options_ = try allocator.alloc(T, options.len),
                .bench_count_ = bench_count,
                .counts = std.AutoHashMap(usize, usize).init(allocator),
                .times = std.AutoHashMap(usize, f64).init(allocator),
            };
            var idx: usize = 0;
            while (options.next()) |o| {
                self.options_[idx] = o;
                idx += 1;
            }
            try self.reset();
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.options_);
            self.counts.deinit();
            self.times.deinit();
            self.allocator.destroy(self);
        }

        /// Gets the current option; updates the current state.
        ///
        /// If each option hasn't been used/timed as many times as the max count, pick
        /// the first option that hasn't been timed to the maximum count. If all
        /// timings are complete, choose the optimal timing.
        pub fn updateState(self: *Self) T {
            if (!self.timings_complete) {
                for (self.options_, 0..) |o, i| {
                    if (self.counts.get(i).? < self.bench_count_) {
                        self.current_option_idx = i;
                        return o;
                    }
                }
                self.timings_complete = true;

                // All options have been benchmarked with the max count - pick the one
                // with the lowest time
                var min_time_option_idx: usize = 0;
                for (0..self.options_.len) |i| {
                    if (self.times.get(i).? < self.times.get(min_time_option_idx).?) {
                        min_time_option_idx = i;
                    }
                }
                self.current_option_idx = min_time_option_idx;
            }
            return self.options_[self.current_option_idx];
        }

        /// Gets the options' current value. This is deterministically computed and
        /// only changes as per calls to `accumulateTimeToCurrentOption` that may
        /// increment the count
        pub fn currentOption(self: *Self) T {
            return self.updateState();
        }

        /// Returns whether or not this options' timings are complete.
        pub fn timingsComplete(self: *Self) bool {
            _ = self.updateState();
            return self.timings_complete;
        }

        /// Adds time to the current option tally.
        pub fn accumulateTimeToCurrentOption(self: *Self, time: f64, increment_count: bool) !void {
            if (self.timingsComplete()) {
                std.debug.print("Options.{s}: Tried to accumulate time when benchmarking is complete\n", .{@src().fn_name});
                return error.AccumulateTimeAfterBenchmarksComplete;
            }
            _ = self.updateState();
            try self.times.put(self.current_option_idx, (self.times.get(self.current_option_idx) orelse 0) + time);
            if (increment_count) {
                try self.counts.put(self.current_option_idx, (self.counts.get(self.current_option_idx) orelse 0) + 1);
            }
        }

        /// Resets options state to the default. Clears timings and counts.
        pub fn reset(self: *Self) !void {
            for (0..self.options_.len) |i| {
                try self.counts.put(i, 0);
                try self.times.put(i, 0);
            }
            self.timings_complete = false;
            self.current_option_idx = 0;
        }
    };
}

// Global zt benchmark mode - if off, no benchmarks will run, and audited
// functions will be run directly without timings
var benchmark_mode = false;

pub const DynamicBenchmark = struct {
    const Self = @This();

    options_: zigrc.Arc(DynamicBenchmarkOptionsBase),
    // Timer for current benchmark iteration
    timer: std.time.Timer = undefined,

    pub fn init(options: zigrc.Arc(DynamicBenchmarkOptionsBase)) Self {
        return .{
            .options_ = options,
        };
    }

    pub fn deinit(self: Self) void {
        self.options_.releaseWithFn(DynamicBenchmarkOptionsBase.deinit);
    }

    pub fn audit(self: *Self, allocator: std.mem.Allocator, function: *const fn (ctx: ?*anyopaque) anyerror!void, ctx: ?*anyopaque, increment_count: bool) !void {
        // Only run the benchmarking components if some options are yet to be
        // fully-timed and benchmark mode is on - otherwise, only run the passed
        // lambda
        if (self.options_.value.timingsComplete() or !benchmark_mode) {
            try function(ctx);
        } else {
            try self.start(allocator);
            try function(ctx);
            try self.stop(allocator, increment_count);
        }
    }

    pub fn getOptions(self: *Self, comptime T: type) T {
        // TODO: find a way to retain ref count for `self.options_` underlying value
        // without preventing release of self.options_ itself, but ensuring no double free
        // on underlying resource
        // _ = self.options_.retain();
        return @ptrCast(@alignCast(self.options_.value.ptr));
    }

    pub fn start(self: *Self, allocator: std.mem.Allocator) !void {
        try zt.tensor.sync(allocator);
        self.timer = try std.time.Timer.start();
    }

    pub fn stop(self: *Self, allocator: std.mem.Allocator, increment_count: bool) !void {
        try zt.tensor.sync(allocator);
        const time: f64 = @as(f64, @floatFromInt(self.timer.lap())) / std.time.ns_per_s;
        try self.options_.value.accumulateTimeToCurrentOption(time, increment_count);
    }

    pub fn setBenchmarkMode(mode: bool) void {
        benchmark_mode = mode;
    }

    pub fn getBenchmarkMode() bool {
        return benchmark_mode;
    }
};

pub const ConvBenchmarks = struct {
    bwd_filter_benchmark: zigrc.Arc(DynamicBenchmark),
    bwd_data_benchmark: zigrc.Arc(DynamicBenchmark),
    bwd_bias_benchmark: zigrc.Arc(DynamicBenchmark),
};

test "DynamicBenchmark -> OptionsStateBasic" {
    benchmark_mode = true;
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();
    const max_count: usize = 5;
    const ops: []const i32 = &.{ 1, 2, 3, 4, 5 };

    var options = try DynamicBenchmarkOptions(i32).initFromSlice(allocator, ops, max_count);
    defer options.deinit();

    try std.testing.expect(options.timingsComplete() == false);
    try std.testing.expect(options.currentOption() == 1);
    for (0..max_count * ops.len) |_| {
        try options.accumulateTimeToCurrentOption(1, true);
    }
    try std.testing.expect(options.timingsComplete());
    try std.testing.expect(options.currentOption() == 1); // best idx should never have changed
}

test "DynamicBenchmark -> OptionscurrentOptionUnchangedWithNoCountIncrement" {
    benchmark_mode = true;
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();
    const ops: []const i32 = &.{ 1, 2, 3, 4, 5 };

    var options = try DynamicBenchmarkOptions(i32).initFromSlice(allocator, ops, 3);
    defer options.deinit();

    var state = options.currentOption();
    try options.accumulateTimeToCurrentOption(3, false);
    try options.accumulateTimeToCurrentOption(4, false);
    try std.testing.expect(state == options.currentOption());
}

test "DynamicBenchmark -> OptionsStateTimed" {
    benchmark_mode = true;
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();
    const max_count: usize = 5;
    const ops: []const i32 = &.{ 1, 2, 3, 4, 5 };

    var options = try DynamicBenchmarkOptions(i32).initFromSlice(allocator, ops, max_count);
    defer options.deinit();

    for (0..max_count * ops.len) |i| {
        // option 4 is faster
        if (options.currentOption() == 4) {
            try options.accumulateTimeToCurrentOption(1, true);
        } else {
            try options.accumulateTimeToCurrentOption(10 * (@as(f64, @floatFromInt(i)) + 1), false);
            try options.accumulateTimeToCurrentOption(10 * (@as(f64, @floatFromInt(i)) + 1), true);
        }
    }
    try std.testing.expect(options.timingsComplete());
    try std.testing.expect(options.currentOption() == 4); // fastest
    try std.testing.expect(options.currentOption() == 4); // no state change (timings are complete)
}

test "DynamicBenchmark -> DynamicBenchmarkSimple" {
    benchmark_mode = true;
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();
    const max_count: usize = 5;
    const sleep_times: []const u64 = &.{ 4, 2, 6 };

    var options = try DynamicBenchmarkOptions(u64).initFromSlice(allocator, sleep_times, max_count);
    var dynamic_bench = try zigrc.Arc(DynamicBenchmark).init(allocator, DynamicBenchmark.init(try zigrc.Arc(DynamicBenchmarkOptionsBase).init(allocator, DynamicBenchmarkOptionsBase.init(options))));
    defer dynamic_bench.releaseWithFn(DynamicBenchmark.deinit);

    const Ctx = struct { sleep_time: u64 };
    var ctx = try allocator.create(Ctx);
    defer allocator.destroy(ctx);

    for (0..max_count * sleep_times.len) |_| {
        ctx.* = .{ .sleep_time = options.currentOption() * std.time.ns_per_ms };
        const cb = (struct {
            pub fn call(c: ?*anyopaque) !void {
                var ctx_: *Ctx = @ptrCast(@alignCast(c));
                std.time.sleep(ctx_.sleep_time);
            }
        }).call;
        try dynamic_bench.value.audit(allocator, cb, ctx, true);
    }
    try std.testing.expect(options.timingsComplete());
    // sleeping for fewer miliseconds is faster
    try std.testing.expect(options.currentOption() == 2);
}

test "DynamicBenchmark -> DynamicBenchmarkDisjointLambdas" {
    benchmark_mode = true;
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();
    const max_count: usize = 5;
    const sleep_times: []const u64 = &.{ 4, 2, 6 };

    var options = try DynamicBenchmarkOptions(u64).initFromSlice(allocator, sleep_times, max_count);
    var dynamic_bench = try zigrc.Arc(DynamicBenchmark).init(allocator, DynamicBenchmark.init(try zigrc.Arc(DynamicBenchmarkOptionsBase).init(allocator, DynamicBenchmarkOptionsBase.init(options))));
    defer dynamic_bench.releaseWithFn(DynamicBenchmark.deinit);

    const Ctx = struct { sleep_time: u64 };
    var ctx = try allocator.create(Ctx);
    defer allocator.destroy(ctx);

    for (0..max_count * sleep_times.len) |_| {
        var sleep_time: u64 = options.currentOption() * std.time.ns_per_ms;
        ctx.* = .{ .sleep_time = sleep_time };
        const cb = (struct {
            pub fn call(c: ?*anyopaque) !void {
                var ctx_: *Ctx = @ptrCast(@alignCast(c));
                std.time.sleep(ctx_.sleep_time);
            }
        }).call;
        try dynamic_bench.value.audit(allocator, cb, ctx, false);

        // intermediate sleep is inversely proportional to the audit sleep time:
        // 4, 2, 6 --> 18, 24, 12
        // total duration disregarding the audit is therefore:
        // 18 + 2 * 4, 24 + 2 * 2, 12 + 2 * 6 ---> 26, 28, 24
        var intermediate_sleep_time: u64 = (30 - (3 * options.currentOption())) * std.time.ns_per_ms;
        std.time.sleep(intermediate_sleep_time);
        try dynamic_bench.value.audit(allocator, cb, ctx, true);
    }
    try std.testing.expect(options.timingsComplete());
    // option 2 is still fastest disregarding intermediate time
    try std.testing.expect(options.currentOption() == 2);
}

test "DynamicBenchmark -> DynamicBenchmarkMatmul" {
    benchmark_mode = true;
    const Dim = zt.tensor.shape.Dim;
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    const max_count: usize = 5;
    // n x n arrays of different sizes
    const array_sizes: []const Dim = &.{ 256, 8, 2048 };

    var dynamic_bench = try zigrc.Arc(DynamicBenchmark).init(
        allocator,
        DynamicBenchmark.init(
            try zigrc.Arc(DynamicBenchmarkOptionsBase).init(
                allocator,
                DynamicBenchmarkOptionsBase.init(
                    try DynamicBenchmarkOptions(Dim).initFromSlice(
                        allocator,
                        array_sizes,
                        max_count,
                    ),
                ),
            ),
        ),
    );
    defer dynamic_bench.releaseWithFn(DynamicBenchmark.deinit);

    const Ctx = struct { size: Dim, allocator: std.mem.Allocator };
    var ctx = try allocator.create(Ctx);
    defer allocator.destroy(ctx);

    for (0..max_count * array_sizes.len) |_| {
        ctx.* = .{ .size = dynamic_bench.value.getOptions(*DynamicBenchmarkOptions(Dim)).currentOption(), .allocator = allocator };
        const cb = (struct {
            pub fn call(context: ?*anyopaque) !void {
                var ctx_: *Ctx = @ptrCast(@alignCast(context));
                const size = ctx_.size;
                const alloc = ctx_.allocator;
                var a = try zt.tensor.rand(alloc, &.{ size, size }, .f32);
                defer a.deinit();
                var b = try zt.tensor.rand(alloc, &.{ size, size }, .f32);
                defer b.deinit();
                var c = try zt.tensor.matmul(alloc, a, b, .None, .None);
                defer c.deinit();
                try zt.tensor.eval(allocator, c);
            }
        }).call;
        try dynamic_bench.value.audit(allocator, cb, ctx, true);
    }
    var ops = dynamic_bench.value.getOptions(*DynamicBenchmarkOptions(Dim));
    try std.testing.expect(ops.timingsComplete());
    try std.testing.expect(ops.currentOption() == std.mem.min(Dim, array_sizes));
}
