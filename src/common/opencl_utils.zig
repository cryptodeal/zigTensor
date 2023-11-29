const opencl = @import("../bindings/opencl.zig");
const build_options = @import("build_options");
const ZT_ARRAYFIRE_USE_OPENCL = build_options.ZT_ARRAYFIRE_USE_OPENCL;

pub fn getQueue(retain: bool) !opencl.cl_command_queue {
    var queue: opencl.cl_command_queue = null;
    if (ZT_ARRAYFIRE_USE_OPENCL) {
        const af = @import("../bindings/af/arrayfire.zig");
        try af.AF_CHECK(af.afcl_get_queue(@ptrCast(&queue), retain), @src());
    }
    return queue;
}

pub fn getContext(retain: bool) !opencl.cl_context {
    var context: opencl.cl_context = null;
    if (ZT_ARRAYFIRE_USE_OPENCL) {
        const af = @import("../bindings/af/arrayfire.zig");
        try af.AF_CHECK(af.afcl_get_context(@ptrCast(&context), retain), @src());
    }
    return context;
}

pub fn getDeviceId() !opencl.cl_device_id {
    var device_id: opencl.cl_device_id = null;
    if (ZT_ARRAYFIRE_USE_OPENCL) {
        const af = @import("../bindings/af/arrayfire.zig");
        try af.AF_CHECK(af.afcl_get_device_id(@ptrCast(&device_id)), @src());
    }
    return device_id;
}
