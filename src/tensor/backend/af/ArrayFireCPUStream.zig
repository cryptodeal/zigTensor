const std = @import("std");
const zigrc = @import("zigrc");
const af = @import("../../../bindings/af/ArrayFire.zig");
const rt_stream = @import("../../../runtime/Stream.zig");
const rt_device = @import("../../../runtime/Device.zig");
const rt_sync_stream = @import("../../../runtime/SynchronousStream.zig");

const Arc = zigrc.Arc;
const SynchronousStream = rt_sync_stream.SynchronousStream;
const StreamType = rt_stream.StreamType;
const Stream = rt_stream.Stream;
const StreamErrors = rt_stream.StreamErrors;
const Device = rt_device.Device;
const X64Device = rt_device.X64Device;

pub const ArrayFireCPUStream = @This();

pub const type_: StreamType = .Synchronous;

syncImpl_: *SynchronousStream,
allocator: std.mem.Allocator,

pub fn create(allocator: std.mem.Allocator) !Arc(Stream) {
    var self = try allocator.create(ArrayFireCPUStream);
    self.* = .{
        .allocator = allocator,
        .syncImpl_ = try SynchronousStream.init(allocator),
    };
    var stream_obj = Stream.init(self);
    var stream = try Arc(Stream).init(allocator, stream_obj);
    try self.syncImpl_.device_.addStream(stream);
    return stream;
}

pub fn deinit(self: *ArrayFireCPUStream) void {
    self.syncImpl_.deinit();
    self.allocator.destroy(self);
}

pub fn streamType(self: *ArrayFireCPUStream) StreamType {
    return self.syncImpl_.streamType();
}

pub fn device(self: *ArrayFireCPUStream) Device {
    return self.syncImpl_.device();
}

pub fn sync(_: *ArrayFireCPUStream) !void {
    try af.ops.sync(try af.ops.getDevice());
}

pub fn relativeSync(_: *ArrayFireCPUStream, wait_on: Stream) !void {
    switch (wait_on.streamType()) {
        .Synchronous => {
            var stream = try wait_on.getImpl(ArrayFireCPUStream);
            try stream.sync();
        },
        else => {
            std.log.debug("[zt.Stream.relativeSync] Unsupported for different types of streams\n", .{});
            return StreamErrors.StreamTypeMismatch;
        },
    }
}

pub fn relativeSyncMulti(self: *ArrayFireCPUStream, wait_ons: std.AutoHashMap(Stream, void)) !void {
    var iterator = wait_ons.keyIterator();
    while (iterator.next()) |stream| {
        try self.relativeSync(stream.*);
    }
}
