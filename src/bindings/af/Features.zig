const std = @import("std");
const af = @import("ArrayFire.zig");

pub const Features = struct {
    const Self = @This();

    feats_: af.AF_Features,
    allocator: std.mem.Allocator,

    pub fn initFromFeatures(allocator: std.mem.Allocator, features: af.AF_Features) !*Self {
        var self = try allocator.create(Self);
        self.* = .{ .feats_ = features, .allocator = allocator };
        return self;
    }

    pub fn init(allocator: std.mem.Allocator, num: i64) !*Self {
        return af.ops.createFeatures(allocator, num);
    }

    pub fn deinit(self: *Self) void {
        self.releaseFeatures() catch unreachable;
        self.allocator.destroy(self);
    }

    pub fn retainFeatures(self: *Self, allocator: std.mem.Allocator) !*Self {
        return af.ops.retainFeatures(allocator, self);
    }

    pub fn getFeaturesNum(self: *Self) !i64 {
        return af.ops.getFeaturesNum(self);
    }

    pub fn getFeaturesXPos(self: *Self, allocator: std.mem.Allocator) !*af.Array {
        return af.ops.getFeaturesXPos(allocator, self);
    }

    pub fn getFeaturesYPos(self: *Self, allocator: std.mem.Allocator) !*af.Array {
        return af.ops.getFeaturesYPos(allocator, self);
    }

    pub fn getFeaturesScore(self: *Self, allocator: std.mem.Allocator) !*af.Array {
        return af.ops.getFeaturesScore(allocator, self);
    }

    pub fn getFeaturesOrientation(self: *Self, allocator: std.mem.Allocator) !*af.Array {
        return af.ops.getFeaturesOrientation(allocator, self);
    }

    pub fn getFeaturesSize(self: *Self, allocator: std.mem.Allocator) !*af.Array {
        return af.ops.getFeaturesSize(allocator, self);
    }

    pub fn releaseFeatures(self: *Self) !*af.Array {
        return af.ops.releaseFeatures(self);
    }
};
