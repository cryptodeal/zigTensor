const std = @import("std");
const zigrc = @import("zigrc");
const zt = @import("../../../zt.zig");
const af = @import("../../../bindings/af/arrayfire.zig");

const Arc = zigrc.Arc;
const SynchronousStream = zt.runtime.SynchronousStream;
const StreamType = zt.runtime.StreamType;
const Stream = zt.runtime.Stream;
const StreamErrors = zt.runtime.StreamErrors;
const Device = zt.runtime.Device;
const X64Device = zt.runtime.X64Device;

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
    const stream_obj = Stream.init(self);
    const stream = try Arc(Stream).init(allocator, stream_obj);
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
