const std = @import("std");
const rc = @import("zigrc");
const device_type = @import("DeviceType.zig");
const stream = @import("Stream.zig");

const DeviceType = device_type.DeviceType;
const Stream = stream.Stream;

/// Throws an error and logs with a descriptive message
/// if the given types don't match
pub fn deviceImplCheck(expect: DeviceType, actual: DeviceType) !void {
    if (expect != actual) {
        std.debug.print("[zt.Device.impl] specified device type: [{any}] doesn't match actual device type: [{any}]\n", .{expect});
        return error.DeviceTypeMismatch;
    }
}

pub const Device = struct {
    /// Tracks Streams from the Device.
    streams_: std.AutoHashMap(rc.Arc(*Stream), void),
    /// Used to update internal backend state for active device, thereby
    /// eliminating the `setActive --> AnyTensorBackendImpl` dependency(s).
    setActiveCallbacks_: std.ArrayList(*const fn (d: usize) void),

    pub fn init(allocator: std.mem.Allocator) !*Device {
        const self = try allocator.create(Device);
        self.* = .{
            .streams_ = try std.AutoHashMap(rc.Arc(*Stream), void).init(allocator),
            .setActiveCallbacks_ = try std.ArrayList(*const fn (d: usize) void).init(allocator),
        };
    }

    pub fn deinit(self: *Device) void {
        // TODO decrement keys in streams_
        self.streams_.deinit();
        self.setActiveCallbacks_.deinit();
    }

    pub fn getStreams(self: *Device) std.AutoHashMap(rc.Arc(*Stream), void) {
        return self.streams_;
    }
};
