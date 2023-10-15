const std = @import("std");
const rt_stream = @import("stream.zig");
const rt_device = @import("device.zig");
const rt_device_mgr = @import("device_manager.zig");

const X64Device = rt_device.X64Device;
const Device = rt_device.Device;
const StreamType = rt_stream.StreamType;
const Stream = rt_stream.Stream;
const StreamErrors = rt_stream.StreamErrors;
const DeviceManager = rt_device_mgr.DeviceManager;

/// An abstraction for a synchronous stream. The word "synchronous"
/// describes the relative synchronization strategy, i.e., it merely
/// delegates to `sync`.
pub const SynchronousStream = struct {
    const Self = @This();

    pub const type_: StreamType = .Synchronous;

    device_: *X64Device,
    type_: StreamType = .Synchronous,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*Self {
        var self = try allocator.create(Self);
        var mgr = try DeviceManager.getInstance(allocator);
        var dev = try mgr.getActiveDevice(.x64);
        self.* = .{
            .allocator = allocator,
            .device_ = try dev.getImpl(X64Device),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn streamType(_: *Self) StreamType {
        return Self.type_;
    }

    pub fn device(self: *Self) Device {
        return Device.init(self.device_);
    }

    pub fn sync(_: *Self) !void {} // no-op

    pub fn relativeSync(_: *Self, wait_on: Stream) !void {
        switch (wait_on.streamType()) {
            .Synchronous => {
                var stream = try wait_on.getImpl(SynchronousStream);
                try stream.sync();
            },
            else => {
                std.log.debug("[zt.Stream.relativeSync] Unsupported for different types of streams\n", .{});
                return StreamErrors.StreamTypeMismatch;
            },
        }
    }

    pub fn relativeSyncMulti(self: *Self, wait_ons: std.AutoHashMap(Stream, void)) !void {
        var iterator = wait_ons.keyIterator();
        while (iterator.next()) |stream| {
            try self.relativeSync(stream);
        }
    }
};
