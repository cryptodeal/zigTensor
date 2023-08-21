const TensorBackend = @import("TensorBackend.zig").TensorBackend;
const ZT_BACKEND_CUDA = @import("build_options").ZT_BACKEND_CUDA;

pub fn defaultTensorBackend() TensorBackend {}
