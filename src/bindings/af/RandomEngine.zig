const std = @import("std");
const af = @import("ArrayFire.zig");

/// Wraps `af.af_random_engine`, the ArrayFire randomEngine class,
/// as a zig struct to simplify calling into the ArrayFire C API.
pub const RandomEngine = struct {
    const Self = @This();

    engine_: af.af_random_engine,
    allocator: std.mem.Allocator,

    pub fn initFromRandomEngine(allocator: std.mem.Allocator, rand_engine: af.af_random_engine) !*Self {
        var self = try allocator.create(Self);
        self.* = .{ .engine_ = rand_engine, .allocator = allocator };
        return self;
    }

    pub fn init(allocator: std.mem.Allocator, rtype: af.RandomEngineType, seed: u64) !*Self {
        return af.ops.createRandEngine(allocator, rtype, seed);
    }

    pub fn initDefault(allocator: std.mem.Allocator) !*Self {
        return af.ops.getDefaultRandEngine(allocator);
    }

    pub fn deinit(self: *Self) void {
        self.release() catch unreachable;
        self.allocator.destroy(self);
    }

    pub fn retain(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.retainRandEngine(allocator, self);
    }

    pub fn setType(self: *Self, rtype: af.RandomEngineType) !void {
        try af.ops.randEngineSetType(self, rtype);
    }

    pub fn getType(self: *Self) !af.RandomEngineType {
        return af.ops.randEngineGetType(self);
    }

    pub fn setSeed(self: *Self, seed: u64) !void {
        try af.ops.randEngineSetSeed(self, seed);
    }

    pub fn getSeed(self: *Self) !u64 {
        return af.ops.randEngineGetSeed(self);
    }

    pub fn release(self: *Self) !void {
        try af.ops.releaseRandEngine(self);
    }
};
