const default_tensor_type = @import("DefaultTensorType.zig");
const index = @import("Index.zig");
const init_ = @import("Init.zig");
const base = @import("TensorBase.zig");
const shape = @import("Shape.zig");
const types = @import("Types.zig");
const tensor_backend = @import("TensorBackend.zig");

// DefaultTensorType.zig exports
pub const DefaultTensorType_t = default_tensor_type.DefaultTensorType_t;
pub const DefaultTensorBackend_t = default_tensor_type.DefaultTensorBackend_t;
pub const defaultTensorBackend = default_tensor_type.defaultTensorBackend;

// Index.zig exports
pub const end_t = index.end_t;
pub const end = index.end;
pub const IndexType = index.IndexType;
pub const RangeIdxTypeTag = index.RangeIdxTypeTag;
pub const RangeError = index.RangeError;
pub const Range = index.Range;
pub const Index = index.Index;

// Init.zig exports
pub const init = init_.init;
pub const deinit = init_.deinit;

// TensorAdapter.zig exports
pub const TensorAdapterBase = @import("TensorAdapter.zig").TensorAdapterBase;

// TensorBackend.zig exports
pub const areBackendsEqual = tensor_backend.areBackendsEqual;
pub const ValIdxRes = tensor_backend.ValIdxRes;
pub const SortIndexRes = tensor_backend.SortIndexRes;
pub const TensorBackend = tensor_backend.TensorBackend;

// TensorBase.zig exports
pub const TensorBackendType = base.TensorBackendType;
pub const Location = base.Location;
pub const StorageType = base.StorageType;
pub const MatrixProperty = base.MatrixProperty;
pub const SortMode = base.SortMode;
pub const PadType = base.PadType;
pub const Tensor = base.Tensor;

// Shape.zig exports
pub const Dim = shape.Dim;
pub const ShapeErrors = shape.ShapeErrors;
pub const Shape = shape.Shape;

// Types.zig exports
pub const DTypeError = types.DTypeError;
pub const DType = types.DType;
pub const dtypeTraits = types.dtypeTraits;

pub usingnamespace @import("Random.zig");
pub usingnamespace @import("TensorOps.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
