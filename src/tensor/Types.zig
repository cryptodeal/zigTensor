const std = @import("std");
const af = @import("../bindings/af/ArrayFire.zig");

const StringMapType: type = struct { key: []const u8, value: DType };

const kStringToType = std.ComptimeStringMap(usize, []StringMapType{
    .{ .key = "f16", .value = DType.f16 },
    .{ .key = "f32", .value = DType.f32 },
    .{ .key = "f64", .value = DType.f64 },
    .{ .key = "b8", .value = DType.b8 },
    .{ .key = "s16", .value = DType.s16 },
    .{ .key = "s32", .value = DType.s32 },
    .{ .key = "s64", .value = DType.s64 },
    .{ .key = "u8", .value = DType.u8 },
    .{ .key = "u16", .value = DType.u16 },
    .{ .key = "u32", .value = DType.u32 },
    .{ .key = "u64", .value = DType.u64 },
});

pub const DTypeError = error{ InvalidStringInput, InvalidTypeQueried };

pub const DType = enum(u8) {
    f16 = 0, // 16-bit float
    f32 = 1, // 32-bit float
    f64 = 2, // 64-bit float
    b8 = 3, // 8-bit boolean
    s16 = 4, // 16-bit signed integer
    s32 = 5, // 32-bit signed integer
    s64 = 6, // 64-bit signed integer
    u8 = 7, // 8-bit unsigned integer
    u16 = 8, // 16-bit unsigned integer
    u32 = 9, // 32-bit unsigned integer
    u64 = 10, // 64-bit unsigned integer

    pub fn toString(self: *DType) []const u8 {
        return switch (self) {
            .f16 => "f16",
            .f32 => "f32",
            .f64 => "f64",
            .b8 => "b8",
            .s16 => "s16",
            .s32 => "s32",
            .s64 => "s64",
            .u8 => "u8",
            .u16 => "u16",
            .u32 => "u32",
            .u64 => "u64",
        };
    }

    pub fn toAfDtype(self: DType) af.Dtype {
        return switch (self) {
            .f16 => af.Dtype.f16,
            .f32 => af.Dtype.f32,
            .f64 => af.Dtype.f64,
            .b8 => af.Dtype.b8,
            .s16 => af.Dtype.s16,
            .s32 => af.Dtype.s32,
            .s64 => af.Dtype.s64,
            .u8 => af.Dtype.u8,
            .u16 => af.Dtype.u16,
            .u32 => af.Dtype.u32,
            .u64 => af.Dtype.u64,
        };
    }

    pub fn getSize(self: *DType) usize {
        return switch (self) {
            .f16 => @sizeOf(f16),
            .f32 => @sizeOf(f32),
            .f64 => @sizeOf(f64),
            .b8 => @sizeOf(u8),
            .s16 => @sizeOf(i16),
            .s32 => @sizeOf(i32),
            .s64 => @sizeOf(i64),
            .u8 => @sizeOf(u8),
            .u16 => @sizeOf(u16),
            .u32 => @sizeOf(u32),
            .u64 => @sizeOf(u64),
        };
    }

    pub fn stringToDtype(_: *DType, string: []const u8) !DType {
        if (kStringToType.has(string)) {
            return kStringToType.get(string).?;
        }
        return DTypeError.InvalidStringInput;
    }
};

test "DType.toAfDtype" {
    try std.testing.expect(DType.f16.toAfDtype() == af.Dtype.f16);
    try std.testing.expect(DType.f32.toAfDtype() == af.Dtype.f32);
    try std.testing.expect(DType.f64.toAfDtype() == af.Dtype.f64);
    try std.testing.expect(DType.b8.toAfDtype() == af.Dtype.b8);
    try std.testing.expect(DType.s16.toAfDtype() == af.Dtype.s16);
    try std.testing.expect(DType.s32.toAfDtype() == af.Dtype.s32);
    try std.testing.expect(DType.s64.toAfDtype() == af.Dtype.s64);
    try std.testing.expect(DType.u8.toAfDtype() == af.Dtype.u8);
    try std.testing.expect(DType.u16.toAfDtype() == af.Dtype.u16);
    try std.testing.expect(DType.u32.toAfDtype() == af.Dtype.u32);
    try std.testing.expect(DType.u64.toAfDtype() == af.Dtype.u64);
}
