pub const tensor = @import("tensor/tensor.zig");
pub const common = @import("common/common.zig");
pub const runtime = @import("runtime/runtime.zig");
pub usingnamespace @import("common/common.zig");
pub const autograd = @import("autograd/autograd.zig");

test {
    const allocator = @import("std").testing.allocator;
    tensor.init(allocator);
    defer tensor.deinit();
    _ = autograd;
    _ = tensor;
    _ = runtime;
}
