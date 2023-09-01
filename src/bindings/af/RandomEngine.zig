const std = @import("std");
const af = @import("ArrayFire.zig");

pub const RandomEngine = struct {
    const Self = @This();

    engine_: af.af_random_engine,
    allocator: std.mem.Allocator,

    pub fn initFromRandomEngine(allocator: std.mem.Allocator, rand_engine: af.af_random_engine) !*Self {
        var self = try allocator.create(Self);
        self.* = .{ .engine_ = rand_engine, .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *Self) void {
        af.ops.releaseRandEngine(self) catch unreachable;
        self.allocator.destroy(self);
    }
};
