const std = @import("std");
const af = @import("arrayfire.zig");

/// Wraps `af.af_event`, the ArrayFire Event class, as a zig
/// struct to simplify calling into the ArrayFire C API.
pub const Event = struct {
    event_: af.af_event,
    allocator: std.mem.Allocator,

    pub fn initFromEvent(allocator: std.mem.Allocator, event: af.af_event) !*Event {
        var self = try allocator.create(Event);
        self.* = .{ .event_ = event, .allocator = allocator };
        return self;
    }

    pub fn init(allocator: std.mem.Allocator) !*Event {
        return initFromEvent(allocator, try af.ops.createEvent());
    }

    pub fn deinit(self: *Event) void {
        self.deleteEvent() catch unreachable;
        self.allocator.destroy(self);
    }

    pub fn deleteEvent(self: *Event) !void {
        try af.ops.deleteEvent(self);
    }

    pub fn markEvent(self: *Event) !void {
        try af.ops.markEvent(self);
    }

    pub fn enqueueWaitEvent(self: *Event) !void {
        try af.ops.enqueueWaitEvent(self);
    }

    pub fn blockEvent(self: *Event) !void {
        try af.ops.blockEvent(self);
    }
};
