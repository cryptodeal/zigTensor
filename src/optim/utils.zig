const std = @import("std");
const zt = @import("../zt.zig");

const Tensor = zt.tensor.Tensor;
const Variable = zt.autograd.Variable;

pub fn clipGradNorm(allocator: std.mem.Allocator, parameters: []const *Variable, max_norm: f64) !f64 {
    var grad_norm: f64 = 0;
    for (parameters) |p| {
        if (!p.isGradAvailable()) {
            continue;
        }
        const grad = (try p.grad()).tensor();
        var tmp1 = try zt.tensor.mul(allocator, Tensor, grad, Tensor, grad);
        defer tmp1.deinit();
        var tmp2 = try zt.tensor.sum(allocator, tmp1, &.{}, false);
        defer tmp2.deinit();
        grad_norm += try tmp2.asScalar(allocator, f64);
    }
    grad_norm = @sqrt(grad_norm);
    const scale: f64 = max_norm / (grad_norm + 1e-6);
    if (scale >= 1) {
        return grad_norm;
    }
    for (parameters) |p| {
        if (!p.isGradAvailable()) {
            continue;
        }
        try (try p.grad()).tensor().inPlaceMul(allocator, f64, scale);
    }
    return grad_norm;
}

test "OptimTest -> GradNorm" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();

    var parameters = std.ArrayList(*Variable).init(allocator);
    defer {
        for (parameters.items) |v| v.deinit();
        parameters.deinit();
    }
    for (0..5) |_| {
        var tmp = try Variable.init(allocator, try zt.tensor.randn(allocator, &.{ 10, 10, 10 }, .f32), true);
        var v = try tmp.astype(allocator, .f64);
        tmp.deinit();
        tmp = try Variable.init(allocator, try zt.tensor.randn(allocator, &.{ 10, 10, 10 }, .f64), false);
        defer tmp.deinit();
        try v.addGrad(allocator, tmp);
        try parameters.append(v);
    }
    const max_norm: f64 = 5;
    _ = try clipGradNorm(allocator, parameters.items, max_norm);

    var clipped: f64 = 0;
    for (parameters.items) |v| {
        const g: Tensor = (try v.grad()).tensor();
        var tmp1 = try zt.tensor.mul(allocator, Tensor, g, Tensor, g);
        defer tmp1.deinit();
        var tmp2 = try zt.tensor.sum(allocator, tmp1, &.{}, false);
        defer tmp2.deinit();
        clipped += try tmp2.asScalar(allocator, f64);
    }
    clipped = @sqrt(clipped);
    var tmp1 = try zt.tensor.full(allocator, &.{1}, f64, max_norm, .f32);
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.full(allocator, &.{1}, f64, clipped, .f32);
    defer tmp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, tmp1, tmp2, 1e-5));
}

test "OptimTest -> GradNormF16" {
    const allocator = std.testing.allocator;
    zt.tensor.init(allocator);
    defer zt.tensor.deinit();
    if (!try zt.common.f16Supported(allocator)) {
        return error.SkipZigTest;
    }

    var parameters = std.ArrayList(*Variable).init(allocator);
    defer {
        for (parameters.items) |v| v.deinit();
        parameters.deinit();
    }
    for (0..5) |_| {
        var tmp = try Variable.init(allocator, try zt.tensor.randn(allocator, &.{ 10, 10, 10 }, .f32), true);
        var v = try tmp.astype(allocator, .f16);
        tmp.deinit();
        tmp = try Variable.init(allocator, try zt.tensor.randn(allocator, &.{ 10, 10, 10 }, .f16), false);
        defer tmp.deinit();
        try v.addGrad(allocator, tmp);
        try parameters.append(v);
    }
    const max_norm: f64 = 5;
    _ = try clipGradNorm(allocator, parameters.items, max_norm);

    var clipped: f64 = 0;
    for (parameters.items) |v| {
        const g: Tensor = (try v.grad()).tensor();
        var tmp1 = try zt.tensor.mul(allocator, Tensor, g, Tensor, g);
        defer tmp1.deinit();
        var tmp2 = try zt.tensor.sum(allocator, tmp1, &.{}, false);
        defer tmp2.deinit();
        clipped += try tmp2.asScalar(allocator, f64);
    }
    clipped = @sqrt(clipped);
    var tmp1 = try zt.tensor.full(allocator, &.{1}, f64, max_norm, .f32);
    defer tmp1.deinit();
    var tmp2 = try zt.tensor.full(allocator, &.{1}, f64, clipped, .f32);
    defer tmp2.deinit();
    try std.testing.expect(try zt.tensor.allClose(allocator, tmp1, tmp2, 1e-2));
}
