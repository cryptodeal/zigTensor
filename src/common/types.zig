const std = @import("std");
const OptimLevel = @import("defines.zig").OptimLevel;

pub fn kOptimLevelTypeExclusionMappings(level: OptimLevel, func_name: []const u8) bool {
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
    return switch (level) {
        .Default => false, // unused
        .O1 => o1_map.has(func_name),
        .O2 => o2_map.has(func_name),
        .O3 => false, // unused
    };
}
