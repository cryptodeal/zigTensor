const std = @import("std");
const build_options = @import("build_options");
const zt = @import("../zt.zig");

const defaultTensorBackend = zt.tensor.defaultTensorBackend;

const ZT_USE_ARRAYFIRE = build_options.ZT_USE_ARRAYFIRE;

// TODO: update import paths imports
const deinitBackend = if (ZT_USE_ARRAYFIRE) @import("backend/af/arrayfire_backend.zig").deinitBackend else @compileError("Must specify backend as build argument.");

pub fn Once(comptime f: fn (allocator: std.mem.Allocator) void) type {
    return struct {
        done: bool = false,
        mutex: std.Thread.Mutex = std.Thread.Mutex{},

        /// Call the function `f`.
        /// If `call` is invoked multiple times `f` will be executed only the
        /// first time.
        /// The invocations are thread-safe.
        pub fn call(self: *@This(), allocator: std.mem.Allocator) void {
            if (@atomicLoad(bool, &self.done, .Acquire))
                return;

            return self.callSlow(allocator);
        }

        fn callSlow(self: *@This(), allocator: std.mem.Allocator) void {
            @setCold(true);

            self.mutex.lock();
            defer self.mutex.unlock();

            // The first thread to acquire the mutex gets to run the initializer
            if (!self.done) {
                f(allocator);
                @atomicStore(bool, &self.done, true, .Release);
            }
        }
    };
}

var init_once = Once(initFn){};

pub fn init(allocator: std.mem.Allocator) void {
    if (@import("builtin").is_test) {
        initFn(allocator);
    } else {
        init_once.call(allocator);
    }
}

pub fn deinit() void {
    deinitBackend();
    zt.runtime.deinitDeviceManager();
    zt.tensor.deinitExtensionRegistrar();
}

fn initFn(allocator: std.mem.Allocator) void {
    _ = defaultTensorBackend(allocator) catch unreachable;
    const ext = zt.tensor.TensorExtensionRegistrar.getInstance(allocator) catch unreachable;
    zt.autograd.registerAutogradExtensions(ext) catch unreachable;
    // TODO: initLogging();
}
