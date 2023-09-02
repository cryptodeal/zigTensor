pub const tensor = @import("tensor/tensor.zig");
pub const runtime = @import("runtime/runtime.zig");
pub const af = @import("bindings/af/ArrayFire.zig");

test {
    _ = tensor;
    _ = af;
    _ = runtime;
}
