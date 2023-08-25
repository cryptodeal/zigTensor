const std = @import("std");
const rt_stream = @import("Stream.zig");
const rt_device = @import("Device.zig");
const rt_device_mgr = @import("DeviceManager.zig");

const X64Device = rt_device.X64Device;
const StreamType = rt_stream.StreamType;
const DeviceManager = rt_device_mgr.DeviceManager;

/// An abstraction for a synchronous stream. The word "synchronous"
/// describes the relative synchronization strategy, i.e., it merely
/// delegates to `sync`.
pub const SynchronousStream = struct {
    const Self = @This();

    pub const type_: StreamType = .Synchronous;

    device_: *X64Device,
    type_: StreamType = .Synchronous,

    pub fn init(allocator: std.mem.Allocator) !*Self {
        var self = try allocator.alloc(Self);
        var mgr = try DeviceManager.getInstance(allocator);
        var device = try mgr.getActiveDevice(.x64);
        self.* = .{
            .device_ = try device.getImpl(X64Device),
        };
        return self;
    }

    pub fn streamType(_: *Self) StreamType {
        return Self.type_;
    }
};
