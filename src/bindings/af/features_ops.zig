const std = @import("std");
const af = @import("ArrayFire.zig");

/// Returns a new `AFFeatures` object with `num` features.
pub inline fn createFeatures(allocator: std.mem.Allocator, num: i64) !*af.Features {
    var feats: af.af_features = undefined;
    try af.AF_CHECK(
        af.af_create_features(
            &feats,
            @intCast(num),
        ),
        @src(),
    );
    return af.Features.initFromFeatures(allocator, feats);
}

/// Increases the reference count of the `af.Features` and all
/// of its associated arrays.
///
/// Returns the reference to the incremented `af.Features`.
pub inline fn retainFeatures(
    allocator: std.mem.Allocator,
    features: *const af.Features,
) !*af.Features {
    var feats: af.af_features = undefined;
    try af.AF_CHECK(
        af.af_retain_features(
            &feats,
            features.feats_,
        ),
        @src(),
    );
    return af.Features.initFromFeatures(allocator, feats);
}

/// Returns the number of features associated with this object.
pub inline fn getFeaturesNum(feat: *const af.Features) !i64 {
    var num: af.dim_t = undefined;
    try af.AF_CHECK(
        af.af_get_features_num(
            &num,
            feat.feats_,
        ),
        @src(),
    );
    return @intCast(num);
}

/// Returns an `af.Array` with all x positions of the `af.Features`.
pub inline fn getFeaturesXPos(
    allocator: std.mem.Allocator,
    feat: *const af.Features,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_get_features_xpos(
            &arr,
            feat.feats_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns an `af.Array` with all y positions of the features.
pub inline fn getFeaturesYPos(allocator: std.mem.Allocator, feat: *const af.Features) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_get_features_ypos(
            &arr,
            feat.feats_,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Returns an `af.Array` with the scores of the features.
pub inline fn getFeaturesScore(allocator: std.mem.Allocator, feat: *const af.Features) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_get_features_score(&arr, feat.feats_), @src());
    return af.Array.init(allocator, arr);
}

/// Returns an `af.Array` with the orientations of the features.
pub inline fn getFeaturesOrientation(allocator: std.mem.Allocator, feat: *const af.Features) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_get_features_orientation(&arr, feat.feats_), @src());
    return af.Array.init(allocator, arr);
}

/// Returns an `af.Array` with the sizes of the features.
pub inline fn getFeaturesSize(allocator: std.mem.Allocator, feat: *const af.Features) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_get_features_size(&arr, feat.feats_), @src());
    return af.Array.init(allocator, arr);
}

/// Reduces the reference count of each of the features.
pub inline fn releaseFeatures(feat: *af.Features) !void {
    try af.AF_CHECK(af.af_release_features(feat.feats_), @src());
}
