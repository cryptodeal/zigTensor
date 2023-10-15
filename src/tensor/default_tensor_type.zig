const std = @import("std");
const build_options = @import("build_options");

const TensorBackend = @import("tensor.zig").TensorBackend;

const ZT_USE_ARRAYFIRE = build_options.ZT_USE_ARRAYFIRE;
// TODO: const ZT_USE_ONEDNN = build_options.ZT_USE_ONEDNN;

pub const DefaultTensorType_t = if (ZT_USE_ARRAYFIRE) @import("backend/af/arrayfire_tensor.zig").ArrayFireTensor else @compileError("Must specify backend as compile flag");

pub const DefaultTensorBackend_t = if (ZT_USE_ARRAYFIRE) @import("backend/af/arrayfire_backend.zig").ArrayFireBackend else @compileError("Must specify backend as compile flag");

pub fn defaultTensorBackend(allocator: std.mem.Allocator) !TensorBackend {
    var tensor = DefaultTensorType_t{};
    return tensor.backend(allocator);
}
