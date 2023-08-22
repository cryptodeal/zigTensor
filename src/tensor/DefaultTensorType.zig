const std = @import("std");
const TensorBackend = @import("TensorBackend.zig").TensorBackend;
const build_options = @import("build_options");
const ArrayFireBackend = @import("backend/af/ArrayFireBackend.zig").ArrayFireBackend;

const ZT_USE_ARRAYFIRE = build_options.ZT_USE_ARRAYFIRE;
const ZT_USE_ONEDNN = build_options.ZT_USE_ONEDNN;
const ZT_USE_TENSOR_STUB = build_options.ZT_USE_TENSOR_STUB;

pub var DefaultTensorType: ?TensorBackend = null;

pub fn defaultTensorBackend(allocator: std.mem.Allocator) !TensorBackend {
    if (DefaultTensorType != null) {
        return DefaultTensorType.?;
    }
    var af_backend = try ArrayFireBackend.init(allocator);
    DefaultTensorType = TensorBackend.init(af_backend);
    return DefaultTensorType.?;
}
