const std = @import("std");
const zt = @import("../zt.zig");
const Variable = zt.autograd.Variable;

pub const AdadeltaOptimizer = @import("adadelta_optimizer.zig").AdadeltaOptimizer;
pub const AdagradOptimizer = @import("adagrad_optimizer.zig").AdagradOptimizer;
pub const AdamOptimizer = @import("adam_optimizer.zig").AdamOptimizer;
pub const AMSgradOptimizer = @import("amsgrad_optimizer.zig").AMSgradOptimizer;
pub const NAGOptimizer = @import("nag_optimizer.zig").NAGOptimizer;
pub const NovogradOptimizer = @import("novograd_optimizer.zig").NovogradOptimizer;
pub const RMSPropOptimizer = @import("rms_prop_optimizer.zig").RMSPropOptimizer;
pub const SGDOptimizer = @import("sgd_optimizer.zig").SGDOptimizer;
pub const FirstOrderOptimizer = @import("optimizers.zig").FirstOrderOptimizer;
pub usingnamespace @import("utils.zig");

// Benchmarking
fn time(alloc: std.mem.Allocator, fn_name: []const u8, func: *const fn (allocator: std.mem.Allocator) anyerror!f64) !void {
    std.debug.print("Timing {s} ... ", .{fn_name});
    std.debug.print("{d:.5} msec\n", .{try func(alloc)});
}

const Ctx = struct {
    opt: FirstOrderOptimizer,
    w: *Variable,
    input: *Variable,
};

fn timeit(allocator: std.mem.Allocator, func: *const fn (allocator: std.mem.Allocator, ctx: *Ctx) anyerror!void, ctx: *Ctx) !f64 {
    // warmup
    for (0..10) |_| {
        try func(allocator, ctx);
    }
    try zt.tensor.sync(allocator);

    const num_iters: usize = 100;
    try zt.tensor.sync(allocator);
    var start = try std.time.Timer.start();
    for (0..num_iters) |_| {
        try func(allocator, ctx);
    }
    try zt.tensor.sync(allocator);
    return (@as(f64, @floatFromInt(start.lap())) / std.time.ns_per_s) / @as(f64, @floatFromInt(num_iters));
}

fn optloop(allocator: std.mem.Allocator, opt: FirstOrderOptimizer, w: *Variable) !f64 {
    var input = try Variable.init(allocator, try zt.tensor.randn(allocator, &.{ 10, 10 }, .f32), false);
    defer input.deinit();
    var ctx: Ctx = .{ .opt = opt, .w = w, .input = input };
    const bench_fn = (struct {
        pub fn call(alloc: std.mem.Allocator, c: *Ctx) !void {
            for (0..100) |_| {
                c.opt.zeroGrad();
                const loss = try zt.autograd.matmul(alloc, c.w, c.input);
                defer loss.deinit();
                try loss.backward(alloc, false);
                try c.opt.step(alloc);
            }
        }
    }).call;
    return timeit(allocator, bench_fn, &ctx);
}

fn sgd(allocator: std.mem.Allocator) !f64 {
    const w = try Variable.init(allocator, try zt.tensor.randn(allocator, &.{ 1, 10 }, .f32), true);
    defer w.deinit();
    const opt = FirstOrderOptimizer.init(try SGDOptimizer.init(allocator, &.{try w.clone(allocator)}, 1e-3, 0, 0, false));
    defer opt.deinit();
    return optloop(allocator, opt, w);
}

fn adam(allocator: std.mem.Allocator) !f64 {
    const w = try Variable.init(allocator, try zt.tensor.randn(allocator, &.{ 1, 10 }, .f32), true);
    defer w.deinit();
    const opt = FirstOrderOptimizer.init(try AdamOptimizer.init(allocator, &.{try w.clone(allocator)}, 1e-3, 0.9, 0.999, 1e-8, 0));
    defer opt.deinit();
    return optloop(allocator, opt, w);
}

fn rmsprop(allocator: std.mem.Allocator) !f64 {
    const w = try Variable.init(allocator, try zt.tensor.randn(allocator, &.{ 1, 10 }, .f32), true);
    defer w.deinit();
    const opt = FirstOrderOptimizer.init(try RMSPropOptimizer.init(allocator, &.{try w.clone(allocator)}, 1e-3, 0.99, 1e-8, 0, false));
    defer opt.deinit();
    return optloop(allocator, opt, w);
}

fn adadelta(allocator: std.mem.Allocator) !f64 {
    const w = try Variable.init(allocator, try zt.tensor.randn(allocator, &.{ 1, 10 }, .f32), true);
    defer w.deinit();
    const opt = FirstOrderOptimizer.init(try AdadeltaOptimizer.init(allocator, &.{try w.clone(allocator)}, 1, 0.9, 1e-8, 0));
    defer opt.deinit();
    return optloop(allocator, opt, w);
}

fn nag(allocator: std.mem.Allocator) !f64 {
    const w = try Variable.init(allocator, try zt.tensor.randn(allocator, &.{ 1, 10 }, .f32), true);
    defer w.deinit();
    const opt = FirstOrderOptimizer.init(try NAGOptimizer.init(allocator, &.{try w.clone(allocator)}, 1e-3, 0.99, 0));
    defer opt.deinit();
    return optloop(allocator, opt, w);
}

test "OptimBenchmark" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();
    try time(allocator, "sgd", sgd);
    try time(allocator, "adam", adam);
    try time(allocator, "rmsprop", rmsprop);
    try time(allocator, "adadelta", adadelta);
    try time(allocator, "nag", nag);
}
