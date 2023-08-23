const std = @import("std");

/// all sizes are rounded to at least 512 bytes.
pub const kMinBlockSize: usize = 512;

/// largest "small" allocation is 1 MiB.
pub const kSmallSize: usize = 1048576;

/// "small" allocations are packed in 2 MiB blocks.
pub const kSmallBuffer: usize = 2097152;

/// "large" allocations may be packed in 20 MiB blocks.
pub const kLargeBuffer: usize = 20971520;

/// allocations between 1 and 10 MiB may use kLargeBuffer.
pub const kMinLargeAlloc: usize = 10485760;

/// round up large allocs to 2 MiB.
pub const kRoundLarge: usize = 2097152;

/// Environment variables names, specifying number of mega bytes as floats.
pub const kMemRecyclingSize: []const u8 = "ZT_MEM_RECYCLING_SIZE_MB";
pub const kMemSplitSize: []const u8 = "ZT_MEM_SPLIT_SIZE_MB";
pub const kMB: f64 = @floatFromInt(@shlExact(@as(u32, 1), 20));

fn sizeUp(size: usize) usize {
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
}

fn getAllocationSize(size: usize) usize {
    if (size <= kSmallSize) {
        return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
        return kLargeBuffer;
    } else {
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
}

/// Returns number of bytes as represented by the named environment variable.
/// The variable is interperested as a float string specifying value in MBs.
/// Returns `default_val` on failure to read the variable or parse its value.
pub fn getEnvAsBytesFromFloatMb(name: []const u8, default_val: usize) !usize {
    const env = std.os.getenv(name);
    if (env != null) {
        const mb: f64 = std.fmt.parseFloat(f64, env.?) catch {
            std.log.err("getEnvAsBytesFromFloatMb: Invalid environment variable value: name={s} value={s}\n", .{ name, env });
            return error.FailedGetEnvAsBytesFromFloatMb;
        };
        return @round(mb * kMB);
    }
    return default_val;
}
