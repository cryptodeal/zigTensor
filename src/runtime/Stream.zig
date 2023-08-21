const std = @import("std");

/// Enum for the type of stream.
pub const StreamType = enum {
    CUDA,
    Synchronous,
};

/// Stream ErrorSet
pub const StreamErrors = error{StreamTypeMismatch};

/// An abstraction for a sequence of computations that
/// must be executed synchronously on a specific device.
/// It prioritizes the synchronization of the computations,
/// while remaining agnostic to the computations themselves.
pub const Stream = struct {
    /// Get the underlying implementation of this stream.
    ///
    /// Throws error if specified type does not match the
    /// actual derived stream type.
    ///
    /// Returns an immutable reference to the specified stream type.
    pub fn impl(self: *Stream, T: type) StreamErrors!T {
        if (T != @TypeOf(self)) {
            std.debug.print("[zt.Stream.impl] specified stream type doesn't match actual stream type.\n", .{});
            return StreamErrors.StreamTypeMismatch;
        }
        return (@as(T, @ptrCast(self))).*;
    }
};
