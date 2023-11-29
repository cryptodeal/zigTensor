const types = @import("types.zig");
const ZT_BACKEND_OPENCL = @import("build_options").ZT_BACKEND_OPENCL;
pub const ocl = if (ZT_BACKEND_OPENCL) @import("opencl_utils.zig") else undefined;

pub usingnamespace @import("defines.zig");
pub const kOptimLevelTypeExclusionMappings = types.kOptimLevelTypeExclusionMappings;

pub usingnamespace @import("utils.zig");
pub usingnamespace @import("dynamic_benchmark.zig");
pub usingnamespace @import("device_ptr.zig");
