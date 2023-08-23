const std = @import("std");
const zigrc = @import("zigrc");
const mem_dev_interface = @import("MemoryManagerDeviceInterface.zig");

const Arc = zigrc.Arc;
const MemoryManagerDeviceInterface = mem_dev_interface.MemoryManagerDeviceInterface;

inline fn divup(comptime T: type, a: T, b: T) T {
    return (((a) + (b) - 1) / (b));
}

/// A clone of ArrayFire's default memory manager that implements the
/// `MemoryManagerAdapter` interface and can be used with zigTensor
/// abstractions, facilitating logging and inspection of internal
/// memory manager state during runs.
///
/// Additionally, provides a simple starting point for other memory
/// manager implementations.
pub const DefaultMemoryManager = struct {
    pub const MAX_BUFFERS: u32 = 1000;
    pub const ONE_GB: usize = @shlExact(@as(usize, 1), 30);

    deviceInterface: Arc(MemoryManagerDeviceInterface),
    memStepSize: usize,
    maxBuffers: u32,
    debugMode: bool,

    pub const LockedInfo = struct {
        managerLock: bool,
        userLock: bool,
        bytes: usize,
    };

    pub const locked_t = std.AutoHashMap(usize, LockedInfo);
    pub const locked_iter = locked_t.Iterator;

    pub const free_t = std.AutoHashMap(usize, std.ArrayList(?*anyopaque));
    pub const free_iter = free_t.Iterator;

    // TODO: need zig equivalent for next line
    // pub const uptr_t = std::unique_ptr<void, std::function<void(void*)>>;

    pub const MemoryInfo = struct {
        lockedMap: locked_t,
        freeMap: free_t,

        lockBytes: usize,
        lockBuffers: usize,
        totalBytes: usize,
        totalBuffers: usize,
        maxBytes: usize,

        pub fn init(allocator: std.mem.Allocator) MemoryInfo {
            return .{
                .lockedMap = locked_t.init(allocator),
                .freeMap = free_t.init(allocator),
                .lockBytes = 0,
                .lockBuffers = 0,
                .totalBytes = 0,
                .totalBuffers = 0,
                .maxBytes = ONE_GB,
            };
        }

        pub fn deinit(self: *MemoryInfo) void {
            self.lockedMap.deinit();
            self.freeMap.deinit();
        }
    };

    pub fn getCurrentMemoryInfo(self: *DefaultMemoryManager) MemoryInfo {
        _ = self;
    }
};
