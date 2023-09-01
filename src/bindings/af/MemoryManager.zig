const std = @import("std");
const af = @import("ArrayFire.zig");

pub const MemoryManager = struct {
    pub const Self = @This();
    allocator: std.mem.Allocator,
    memMgr_: af.AF_MemoryManager,

    pub fn initFromMemMgr(allocator: std.mem.Allocator, mem_mgr: af.AF_MemoryManager) !*Self {
        var self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
            .memMgr_ = mem_mgr,
        };
        return self;
    }
};
