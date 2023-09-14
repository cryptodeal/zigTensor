const std = @import("std");
const af = @import("ArrayFire.zig");

/// Returns a new `af.RandomEngine`.
pub fn createRandEngine(allocator: std.mem.Allocator, rtype: af.RandomEngineType, seed: u64) !*af.RandomEngine {
    var rand_engine: af.af_random_engine = undefined;
    try af.AF_CHECK(af.af_create_random_engine(&rand_engine, rtype.value(), @intCast(seed)), @src());
    return af.RandomEngine.initFromRandomEngine(allocator, rand_engine);
}

/// Increment the underlying `af.af_random_engine` reference count.
///
/// Returns a new `af.RandomEngine` wrapping the new `af.af_random_engine`
/// handle.
pub fn retainRandEngine(allocator: std.mem.Allocator, engine: *af.RandomEngine) !*af.RandomEngine {
    var rand_engine: af.af_random_engine = undefined;
    try af.AF_CHECK(af.af_retain_random_engine(&rand_engine, engine.engine_), @src());
    return af.RandomEngine.initFromRandomEngine(allocator, rand_engine);
}

/// Sets the type of the `af.RandomEngine`.
pub fn randEngineSetType(engine: *af.RandomEngine, rtype: af.RandomEngineType) !void {
    try af.AF_CHECK(af.af_random_engine_set_type(&engine.engine_, rtype.value()), @src());
}

/// Returns the type of the `af.RandomEngine`.
pub fn randEngineGetType(engine: *af.RandomEngine) !af.RandomEngineType {
    var rtype: af.af_random_engine_type = undefined;
    try af.AF_CHECK(af.af_random_engine_get_type(&rtype, engine.engine_), @src());
    return @enumFromInt(rtype);
}

/// Sets the seed of the `af.RandomEngine`.
pub fn randEngineSetSeed(engine: *af.RandomEngine, seed: u64) !void {
    try af.AF_CHECK(af.af_random_engine_set_seed(&engine.engine_, @intCast(seed)), @src());
}

/// Returns a new instance of the Default Random Engine.
pub fn getDefaultRandEngine(allocator: std.mem.Allocator) !*af.RandomEngine {
    var rand_engine: af.af_random_engine = undefined;
    try af.AF_CHECK(af.af_get_default_random_engine(&rand_engine), @src());
    return af.RandomEngine.initFromRandomEngine(allocator, rand_engine);
}

/// Sets the type of the Default Random Engine.
pub fn setDefaultRandEngineType(rtype: af.RandomEngineType) !void {
    try af.AF_CHECK(af.af_set_default_random_engine_type(rtype.value()), @src());
}

/// Returns the current seed value of the given `af.RandomEngine`.
pub fn randEngineGetSeed(engine: *af.RandomEngine) !u64 {
    var seed: c_ulonglong = undefined;
    try af.AF_CHECK(af.af_random_engine_get_seed(&seed, engine.engine_), @src());
    return @intCast(seed);
}

/// Releases the underlying `af.af_random_engine` handle.
pub fn releaseRandEngine(engine: *af.RandomEngine) !void {
    try af.AF_CHECK(af.af_release_random_engine(engine.engine_), @src());
}

/// Sets the seed for the current default random engine.
pub fn setSeed(seed: u64) !void {
    try af.AF_CHECK(af.af_set_seed(@intCast(seed)), @src());
}

/// Returns the current seed value of the default random engine.
pub fn getSeed() !u64 {
    var seed: c_ulonglong = undefined;
    try af.AF_CHECK(af.af_get_seed(&seed), @src());
    return @intCast(seed);
}
