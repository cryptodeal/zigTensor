const zt = @import("../../zt.zig");
const std = @import("std");
const build_options = @import("build_options");
const ZT_USE_ONEDNN = build_options.ZT_USE_ONEDNN;
const ZT_ARRAYFIRE_USE_CPU = build_options.ZT_ARRAYFIRE_USE_CPU;
const ZT_ARRAYFIRE_USE_OPENCL = build_options.ZT_ARRAYFIRE_USE_OPENCL;
const ZT_USE_ARRAYFIRE = build_options.ZT_USE_ARRAYFIRE;
const OneDnnAutogradExtension = if (ZT_USE_ONEDNN) @import("backend/onednn/onednn_autograd_extension.zig").OneDnnAutogradExtension else undefined;

const TensorExtensionRegistrer = zt.tensor.TensorExtensionRegistrar;
const TensorExtension = zt.tensor.TensorExtension;

pub fn registerAutogradExtensions(ext: *TensorExtensionRegistrer) std.mem.Allocator.Error!void {
    // TODO: register CUDNN extension
    if (ZT_USE_ONEDNN) {
        // TODO: OneDNN backend can transparently use its autograd extension

        if (ZT_USE_ARRAYFIRE and (ZT_ARRAYFIRE_USE_CPU or ZT_ARRAYFIRE_USE_OPENCL)) {
            const registerFn = (struct {
                pub fn call(alloc: std.mem.Allocator) !TensorExtension {
                    return TensorExtension.init(try OneDnnAutogradExtension.init(alloc));
                }
            }).call;
            _ = try ext.registerTensorExtension(.ArrayFire, .Autograd, registerFn);
        }
    }
}
