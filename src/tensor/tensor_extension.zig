const std = @import("std");
const zt = @import("../zt.zig");
const assert = std.debug.assert;

const DType = zt.tensor.DType;
const TensorBackendType = zt.tensor.TensorBackendType;
const AutogradExtension = zt.autograd.AutogradExtension;

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
        tensorExtensionRegistrarSingleton = null;
    }
}

/// Employ an extensible factory singleton pattern to handle creation callbacks
/// for creating specific TensorExtension instances.
pub const TensorExtensionRegistrar = struct {
    const CallbackMap = std.EnumMap(TensorExtensionType, TensorExtensionCallback);
    allocator: std.mem.Allocator,
    extensions_: std.EnumMap(TensorBackendType, CallbackMap) = .{},

    pub fn init(allocator: std.mem.Allocator) !*TensorExtensionRegistrar {
        var self = try allocator.create(TensorExtensionRegistrar);
        self.* = .{
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *TensorExtensionRegistrar) void {
        self.allocator.destroy(self);
    }

    pub fn getInstance(allocator: std.mem.Allocator) !*TensorExtensionRegistrar {
        if (tensorExtensionRegistrarSingleton == null) {
            tensorExtensionRegistrarSingleton = try TensorExtensionRegistrar.init(allocator);
        }
        return tensorExtensionRegistrarSingleton.?;
    }

    pub fn releaseInstance() void {
        if (tensorExtensionRegistrarSingleton != null) {
            tensorExtensionRegistrarSingleton.?.deinit();
        }
    }

    pub fn registerTensorExtension(self: *TensorExtensionRegistrar, backend: TensorBackendType, extension_type: TensorExtensionType, creation_func: TensorExtensionCallback) bool {
        if (!self.extensions_.contains(backend)) {
            self.extensions_.put(backend, CallbackMap{});
        }
        var inner_map = self.extensions_.get(backend).?;
        // Add extension to registry
        inner_map.put(extension_type, creation_func);
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
        getAutogradExtension: *const fn (ctx: *anyopaque) anyerror!AutogradExtension,
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

    pub fn getExtension(self: *const Self, comptime T: type) !T {
        return switch (T) {
            AutogradExtension => self.vtable.getAutogradExtension(self.ptr),
            // TODO: VisionExtension => self.vtable.getVisionExtension(self.ptr),
            // TODO: JitOptimizerExtension => self.vtable.getJitOptimizerExtension(self.ptr),
            else => @compileError("TensorExtension.getExtension: must be of type `AutogradExtension`, `VisionExtension`, or `JitOptimizerExtension`.\n"),
        };
    }

    pub fn init(ext_impl: anytype) Self {
        const Ptr = @TypeOf(ext_impl);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        const PtrChild = PtrInfo.Pointer.child;
        assert(@typeInfo(PtrChild) == .Struct); // Must point to a struct
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

            fn getAutogradExtension(ctx: *anyopaque) !AutogradExtension {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                if (self.getExtensionType() != .Autograd) {
                    return error.WrongExtensionType;
                }
                return AutogradExtension.init(self);
            }
        };
        return .{
            .ptr = ext_impl,
            .vtable = &.{
                .deinit = impl.deinit,
                .getExtensionType = impl.getExtensionType,
                .isDataTypeSupported = impl.isDataTypeSupported,
                .getAutogradExtension = impl.getAutogradExtension,
            },
        };
    }
};
