const std = @import("std");
const zt = @import("../zt.zig");
const assert = std.debug.assert;

const DType = zt.tensor.DType;
const TensorBackendType = zt.tensor.TensorBackendType;

/// A runtime type denoting the tensor extension.
pub const TensorExtensionType = enum(u8) {
    Generic, // placeholder
    Autograd,
    Vision,
    JitOptimizer,
};

pub const TensorExtensionCallback = *const fn (allocator: std.mem.Allocator) anyerror!TensorExtension;

var tensorExtensionRegistrarSingleton: ?*TensorExtensionRegistrar = null;

pub fn deinitExtensionRegistrar() void {
    if (tensorExtensionRegistrarSingleton != null) {
        tensorExtensionRegistrarSingleton.?.deinit();
    }
}

/// Employ an extensible factory singleton pattern to handle creation callbacks
/// for creating specific TensorExtension instances.
pub const TensorExtensionRegistrar = struct {
    const CallbackMap = std.AutoHashMap(TensorExtensionType, TensorExtensionCallback);
    allocator: std.mem.Allocator,
    extensions_: std.AutoHashMap(TensorBackendType, CallbackMap),

    pub fn init(allocator: std.mem.Allocator) !*TensorExtensionRegistrar {
        var self = try allocator.create(TensorExtensionRegistrar);
        self.* = .{
            .allocator = allocator,
            .extensions_ = std.AutoHashMap(TensorBackendType, CallbackMap).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *TensorExtensionRegistrar) void {
        var iterator = self.extensions_.valueIterator();
        while (iterator.next()) |m| {
            m.deinit();
        }
        self.extensions_.deinit();
        self.allocator.destroy(self);
    }

    pub fn getInstance(allocator: std.mem.Allocator) !*TensorExtensionRegistrar {
        if (tensorExtensionRegistrarSingleton == null) {
            tensorExtensionRegistrarSingleton = try TensorExtensionRegistrar.init(allocator);
        }
        return tensorExtensionRegistrarSingleton.?;
    }

    pub fn registerTensorExtension(self: *TensorExtensionRegistrar, allocator: std.mem.Allocator, backend: TensorBackendType, extension_type: TensorExtensionType, creation_func: TensorExtensionCallback) !bool {
        var inner_map: TensorExtensionRegistrar.CallbackMap = undefined;
        if (!self.extensions_.contains(backend)) {
            inner_map = CallbackMap.init(allocator);
        } else {
            inner_map = self.extensions_.get(backend).?;
        }
        // Add extension to registry
        try inner_map.putNoClobber(extension_type, creation_func);
        return true;
    }

    pub fn isTensorExtensionRegistered(self: *TensorExtensionRegistrar, backend: TensorBackendType, extension_type: TensorExtensionType) bool {
        var entry = self.extensions.get(backend);
        return entry != null and entry.contains(extension_type);
    }

    pub fn getTensorExtensionCreationFunc(self: *TensorExtensionRegistrar, backend: TensorBackendType, extension_type: TensorExtensionType) !TensorExtensionCallback {
        if (!self.extensions_.contains(backend)) {
            std.debug.print("TensorExtensionRegistrar.{s}: no tensor extensions registered for given backend ({s}).\n", .{ @src().fn_name, @tagName(backend) });
            return error.NoExtensionsRegisteredForBackend;
        }
        var _extensions: TensorExtensionRegistrar.CallbackMap = self.extensions_.get(backend).?;
        if (!_extensions.contains(extension_type)) {
            std.debug.print("TensorExtensionRegistrar.{s}: the specified extension ({s}) has not been registered for given backend ({s}).\n", .{ @src().fn_name, @tagName(extension_type), @tagName(backend) });
            return error.ExtensionNotRegisteredForBackend;
        }
        return _extensions.get(extension_type).?;
    }
};

pub const TensorExtension = struct {
    const Self = @This();

    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (ctx: *anyopaque) void,
        getExtensionType: *const fn (ctx: *anyopaque) TensorExtensionType,
        isDataTypeSupported: *const fn (ctx: *anyopaque, dtype: DType) bool,
    };

    /// Free all associated memory.
    pub fn deinit(self: *const Self) void {
        self.vtable.deinit(self.ptr);
    }

    /// Returns the type of the extension.
    pub fn getExtensionType(self: *const Self) TensorExtensionType {
        return self.vtable.getExtensionType(self.ptr);
    }

    pub fn isDataTypeSupported(self: *const Self, dtype: DType) bool {
        return self.vtable.isDataTypeSupported(self.ptr, dtype);
    }

    pub fn init(ext_impl: anytype) Self {
        const Ptr = @TypeOf(ext_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                self.deinit();
            }

            fn getExtensionType(ctx: *anyopaque) TensorExtensionType {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.getExtensionType();
            }

            fn isDataTypeSupported(ctx: *anyopaque, dtype: DType) bool {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.isDataTypeSupported(dtype);
            }

            fn getUnderlyingPtr(ctx: *anyopaque) Ptr {
                return @ptrCast(@alignCast(ctx));
            }
        };
        return .{
            .ptr = ext_impl,
            .vtable = &.{
                .deinit = impl.deinit,
                .getExtensionType = impl.getExtensionType,
                .isDataTypeSupported = impl.isDataTypeSupported,
            },
            .getUnderlyingPtr = impl.getUnderlyingPtr,
        };
    }
};
