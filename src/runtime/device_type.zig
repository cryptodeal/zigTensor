const std = @import("std");
const rc = @import("zigrc");
const ZT_BACKEND_CUDA = @import("build_options").ZT_BACKEND_CUDA;

pub const DeviceType = enum {
    x64,
    CUDA,

    /// Formats DeviceType for printing to writer.
    pub fn format(value: DeviceType, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        return writer.print("{s}", .{@tagName(value)});
    }
};

pub const kDefaultDeviceType: DeviceType = if (ZT_BACKEND_CUDA) .CUDA else .x64;

pub fn deviceTypeToString(device_type: DeviceType) []const u8 {
    return @tagName(device_type);
}

pub fn getDeviceTypes() std.EnumSet(DeviceType) {
    return std.EnumSet(DeviceType).initFull();
}

test "DeviceType getDeviceTypes" {
    const device_types = getDeviceTypes();
    try std.testing.expect(device_types.contains(.x64));
    try std.testing.expect(device_types.contains(.CUDA));
    try std.testing.expect(device_types.count() == 2);
}

test "DeviceType deviceTypeToString" {
    try std.testing.expectEqualStrings("x64", deviceTypeToString(.x64));
    try std.testing.expectEqualStrings("CUDA", deviceTypeToString(.CUDA));
}
