const af = @import("ArrayFire.zig");

/// Returns a new `af.Event` handle.
pub inline fn createEvent() !af.Event {
    var event: af.af_event = undefined;
    try af.AF_CHECK(af.af_create_event(&event), @src());
    return event;
}

/// Release the `af.Event` handle.
pub inline fn deleteEvent(event: af.Event) !void {
    try af.AF_CHECK(af.af_delete_event(event), @src());
}

/// Marks the `af.Event` on the active computation stream.
pub inline fn markEvent(event: af.Event) !void {
    try af.AF_CHECK(af.af_mark_event(event), @src());
}

/// Enqueues the `af.Event` and all enqueued events on the active stream.
pub inline fn enqueueWaitEvent(event: af.Event) !void {
    try af.AF_CHECK(af.af_enqueue_wait_event(event), @src());
}

/// Blocks the calling thread on events until all events on the
/// computation stream before mark was called are complete.
pub inline fn blockEvent(event: af.Event) !void {
    try af.AF_CHECK(af.af_block_event(event), @src());
}
