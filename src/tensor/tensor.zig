const default_tensor_type = @import("default_tensor_type.zig");
const index = @import("index.zig");
const init_ = @import("init.zig");
const base = @import("tensor_base.zig");
const types = @import("types.zig");
const tensor_backend = @import("tensor_backend.zig");
const tensor_extension = @import("tensor_extension.zig");

// default_tensor_type.zig exports
pub const DefaultTensorType_t = default_tensor_type.DefaultTensorType_t;
pub const DefaultTensorBackend_t = default_tensor_type.DefaultTensorBackend_t;
pub const defaultTensorBackend = default_tensor_type.defaultTensorBackend;

// index.zig exports
pub const end_t = index.end_t;
pub const end = index.end;
pub const span = index.span;
pub const IndexType = index.IndexType;
pub const RangeIdxTypeTag = index.RangeIdxTypeTag;
pub const RangeError = index.RangeError;
pub const Range = index.Range;
pub const Index = index.Index;

// init.zig exports
pub const init = init_.init;
pub const deinit = init_.deinit;

// tensor_adapter.zig exports
pub const TensorAdapterBase = @import("tensor_adapter.zig").TensorAdapterBase;

// tensor_backend.zig exports
pub const areBackendsEqual = tensor_backend.areBackendsEqual;
pub const TensorBackend = tensor_backend.TensorBackend;

// tensor_base.zig exports
pub const TensorBackendType = base.TensorBackendType;
pub const Location = base.Location;
pub const StorageType = base.StorageType;
pub const MatrixProperty = base.MatrixProperty;
pub const SortMode = base.SortMode;
pub const PadType = base.PadType;
pub const Tensor = base.Tensor;

// shape.zig exports
pub const shape = @import("shape.zig");

// types.zig exports
pub const DTypeError = types.DTypeError;
pub const DType = types.DType;
pub const dtypeTraits = types.dtypeTraits;

// tensor_extension.zig exports
pub const TensorExtension = tensor_extension.TensorExtension;
pub const TensorExtensionType = tensor_extension.TensorExtensionType;
pub const TensorExtensionRegistrar = tensor_extension.TensorExtensionRegistrar;
pub const deinitExtensionRegistrar = tensor_extension.deinitExtensionRegistrar;

pub usingnamespace @import("random.zig");
pub usingnamespace @import("tensor_ops.zig");
pub usingnamespace @import("compute.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
