const std = @import("std");
const af = @import("arrayfire.zig");

/// Returns a new `af.Event` handle.
pub fn createEvent(allocator: std.mem.Allocator) !*af.Event {
    var event: af.af_event = undefined;
    try af.AF_CHECK(af.af_create_event(&event), @src());
    return af.Event.initFromEvent(allocator, event);
}

/// Release the `af.Event` handle.
pub fn deleteEvent(event: *af.Event) !void {
    try af.AF_CHECK(af.af_delete_event(event.event_), @src());
}

/// Marks the `af.Event` on the active computation stream.
pub fn markEvent(event: *const af.Event) !void {
    try af.AF_CHECK(af.af_mark_event(event.event_), @src());
}

/// Enqueues the `af.Event` and all enqueued events on the active stream.
pub fn enqueueWaitEvent(event: *const af.Event) !void {
    try af.AF_CHECK(af.af_enqueue_wait_event(event.event_), @src());
}

/// Blocks the calling thread on events until all events on the
/// computation stream before mark was called are complete.
pub fn blockEvent(event: *const af.Event) !void {
    try af.AF_CHECK(af.af_block_event(event.event_), @src());
}
