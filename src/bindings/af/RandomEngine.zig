const std = @import("std");
const af = @import("ArrayFire.zig");

pub const RandomEngine = struct {
    const Self = @This();

    randomEngine_: af.AF_RandomEngine,
    allocator: std.mem.Allocator,

    pub fn initFromRandomEngine(allocator: std.mem.Allocator, rand_engine: af.AF_RandomEngine) !*Self {
        var self = try allocator.create(Self);
        self.* = .{ .randomEngine_ = rand_engine, .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *Self) void {
        af.releaseRandEngine(self) catch unreachable;
        self.allocator.destroy(self);
    }
};
