const std = @import("std");
const OptimLevel = @import("defines.zig").OptimLevel;

pub fn kOptimLevelTypeExclusionMappings(level: OptimLevel, func_name: []const u8) bool {
    const default_map = std.ComptimeStringMap(void, .{}); // unused
    const o1_map = std.ComptimeStringMap(void, .{
        // Perform all operations in fp16 except for:
        .{"batchnorm"},
        .{"reciprocal"},
        .{"erf"},
        .{"exp"},
        .{"log"},
        .{"log1p"},
        .{"pow"},
        .{"sum"},
        .{"mean"},
        .{"variance"},
        .{"norm"},
        .{"normalize"},
        .{"softmax"},
        .{"logSoftmax"},
        .{"categoricalCrossEntropy"},
        .{"gelu"},
    });
    const o2_map = std.ComptimeStringMap(void, .{
        // Perform all operations in fp16 except for:
        .{"batchnorm"},
    });
    const o3_map = std.ComptimeStringMap(void, .{}); // Perform all operations in f16
    return switch (level) {
        .DEFAULT => return default_map.has(func_name),
        .O1 => return o1_map.has(func_name),
        .O2 => return o2_map.has(func_name),
        .O3 => return o3_map.has(func_name),
    };
}
