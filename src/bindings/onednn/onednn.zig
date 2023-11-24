const std = @import("std");

const dnnl = @cImport({
    @cInclude("oneapi/dnnl/dnnl.h");
    // conditionally import DNNL OpenCL headers
    if (@import("build_options").ZT_BACKEND_OPENCL) {
        @cInclude("oneapi/dnnl/dnnl_ocl.h");
    }
});

pub usingnamespace dnnl;

const DnnlError = error{
    DnnlOutOfMemory,
    DnnlInvalidArguments,
    DnnlUnimplemented,
    DnnlLastImplReached,
    DnnlRuntimeError,
    DnnlNotRequired,
    DnnlInvalidGraph,
    DnnlInvalidGraphOp,
    DnnlInvalidShape,
    DnnlInvalidDataType,
};

/// Utility function for handling `af.af_err` when calling
/// directly into ArrayFire's C API from Zig.
pub inline fn DNNL_CHECK(v: dnnl.dnnl_status_t, src: std.builtin.SourceLocation) !void {
    if (v != dnnl.dnnl_success) {
        const err: DnnlError = switch (v) {
            dnnl.dnnl_out_of_memory => DnnlError.DnnlOutOfMemory,
            dnnl.dnnl_invalid_arguments => DnnlError.DnnlInvalidArguments,
            dnnl.dnnl_unimplemented => DnnlError.DnnlUnimplemented,
            dnnl.dnnl_last_impl_reached => DnnlError.DnnlLastImplReached,
            dnnl.dnnl_runtime_error => DnnlError.DnnlRuntimeError,
            dnnl.dnnl_not_required => DnnlError.DnnlNotRequired,
            dnnl.dnnl_invalid_graph => DnnlError.DnnlInvalidGraph,
            dnnl.dnnl_invalid_graph_op => DnnlError.DnnlInvalidGraphOp,
            dnnl.dnnl_invalid_shape => DnnlError.DnnlInvalidShape,
            dnnl.dnnl_invalid_data_type => DnnlError.DnnlInvalidDataType,
            else => unreachable,
        };
        std.debug.print("DNNL error: {s}:{d} - {s}\n", .{ src.file, src.line, @errorName(err) });
        return err;
    }
}
