const std = @import("std");
const af = @import("ArrayFire.zig");

/// Returns a new `af.RandomEngine`.
pub inline fn createRandEngine(allocator: std.mem.Allocator, rtype: af.RandomEngineType, seed: u64) !*af.RandomEngine {
    var rand_engine: af.af_random_engine = undefined;
    try af.AF_CHECK(af.af_create_random_engine(&rand_engine, rtype, @intCast(seed)), @src());
    return af.RandomEngine.initFromRandomEngine(allocator, rand_engine);
}

/// Increment the underlying `af.af_random_engine` reference count.
///
/// Returns a new `af.RandomEngine` wrapping the new `af.af_random_engine`
/// handle.
pub inline fn retainRandEngine(allocator: std.mem.Allocator, engine: *af.RandomEngine) !*af.RandomEngine {
    var rand_engine: af.af_random_engine = undefined;
    try af.AF_CHECK(af.af_retain_random_engine(&rand_engine, engine.randomEngine_), @src());
    std.debug.print("new ptr: {any}; old ptr: {any}\n", .{ rand_engine, engine.randomEngine_ });
    return af.RandomEngine.initFromRandomEngine(allocator, rand_engine);
}

/// Sets the type of the `af.RandomEngine`.
pub inline fn randEngineSetType(engine: *af.RandomEngine, rtype: af.RandomEngineType) !void {
    try af.AF_CHECK(af.af_random_engine_set_type(&engine.randomEngine_, rtype), @src());
}

/// Returns the type of the `af.RandomEngine`.
pub inline fn randEngineGetType(engine: *af.RandomEngine) !af.RandomEngineType {
    var rtype: af.af_random_engine_type = undefined;
    try af.AF_CHECK(af.af_random_engine_get_type(&rtype, engine.randomEngine_), @src());
    return @enumFromInt(rtype);
}

pub inline fn releaseRandEngine(engine: *af.RandomEngine) !void {
    try af.AF_CHECK(af.af_release_random_engine(engine.randomEngine_), @src());
}
