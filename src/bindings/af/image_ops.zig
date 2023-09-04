const std = @import("std");
const af = @import("ArrayFire.zig");

/// Load an image from disk to an `af.Array`.
pub inline fn loadImage(
    allocator: std.mem.Allocator,
    filename: []const u8,
    isColor: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_load_image(
            &arr,
            filename.ptr,
            isColor,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Save an `af.Array` to disk as an image.
///
/// Supported formats include JPG, PNG, PPM and other
/// formats supported by freeimage.
pub inline fn saveImage(filename: []const u8, in: *const af.Array) !void {
    try af.AF_CHECK(af.af_save_image(filename.ptr, in.array_), @src());
}

/// Load an image from memory which is stored as a FreeImage
/// stream (FIMEMORY).
///
/// Supported formats include JPG, PNG, PPM and other formats
/// supported by freeimage.
pub inline fn loadImageMem(allocator: std.mem.Allocator, ptr: ?*const anyopaque) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_load_image_memory(&arr, ptr), @src());
    return af.Array.init(allocator, arr);
}

/// Save an `af.Array` to memory as an image using FreeImage
/// stream (FIMEMORY).
///
/// Supported formats include JPG, PNG, PPM and other formats
/// supported by freeimage.
pub inline fn saveImageMem(in: *const af.Array, format: af.ImageFormat) !?*anyopaque {
    var ptr: ?*anyopaque = undefined;
    try af.AF_CHECK(
        af.af_save_image_memory(
            &ptr,
            in.array_,
            format.value(),
        ),
        @src(),
    );
    return ptr;
}

/// Delete memory created by `saveImageMem` function.
///
/// This internally calls FreeImage_CloseMemory.
///
/// Supported formats include JPG, PNG, PPM and other formats
/// supported by freeimage.
pub inline fn deleteImageMem(ptr: ?*anyopaque) !void {
    try af.AF_CHECK(af.af_delete_image_memory(ptr), @src());
}

/// Loads an image as is original type; returns ptr to the
/// resulting `af.Array`.
///
/// This load image function allows you to load images as
/// u8, u16 or f32 depending on the type of input image
/// as shown by the table below:
/// ------------------------------------------------------------------------
/// | Bits per Color (Gray/RGB/RGBA Bits Per Pixel) | Array Type | Range   |
/// ------------------------------------------------------------------------
/// | 8 ( 8/24/32 BPP)                              | u8         | 0-255   |
/// ------------------------------------------------------------------------
/// | 16 (16/48/64 BPP)                             | u16        | 0-65535 |
/// ------------------------------------------------------------------------
/// | 32 (32/96/128 BPP)                            | f32        | 0-1     |
/// ------------------------------------------------------------------------
pub inline fn loadImageNative(allocator: std.mem.Allocator, filename: []const u8) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_load_image_native(&arr, filename.ptr), @src());
    return af.Array.init(allocator, arr);
}

/// Saving an image without modifications.
/// This function only accepts u8, u16, f32 `af.Array`s.
/// These arrays are saved to images without any modifications.
///
/// You must also note that note all image type support 16 or 32 bit images.
///
/// The best options for 16 bit images are PNG, PPM and TIFF.
/// The best option for 32 bit images is TIFF. These allow lossless storage.
///
/// The images stored have the following properties:
/// ------------------------------------------------------------------------
/// | Array Type | Bits per Color (Gray/RGB/RGBA Bits Per Pixel) | Range   |
/// ------------------------------------------------------------------------
/// | u8         | 8 ( 8/24/32 BPP)                              | 0-255   |
/// ------------------------------------------------------------------------
/// | u16        | 16 (16/48/64 BPP)                             | 0-65535 |
/// ------------------------------------------------------------------------
/// | f32        | 32 (32/96/128 BPP)                            | f32     |
/// ------------------------------------------------------------------------
pub inline fn saveImageNative(filename: []const u8, in: *const af.Array) !void {
    try af.AF_CHECK(
        af.af_save_image_native(
            filename.ptr,
            in.array_,
        ),
        @src(),
    );
}

/// Returns true if ArrayFire was compiled with ImageIO (FreeImage) support.
pub inline fn isImageIoAvailable() !bool {
    var out: bool = false;
    try af.AF_CHECK(af.af_is_image_io_available(&out), @src());
    return out;
}

/// Resize an input image; returns ptr to the resulting `af.Array`.
///
/// Resizing an input image can be done using either `af.InterpType.Nearest`,
/// `af.InterpType.Bilinear` or `af.InterpType.Lower`, interpolations.
/// Nearest interpolation will pick the nearest value to the location,
/// bilinear interpolation will do a weighted interpolation for calculate
/// the new size and lower interpolation is similar to the nearest, except
/// it will use the floor function to get the lower neighbor.
///
/// This function does not differentiate between images and data. As long as
/// the `af.Array` is defined and the output dimensions are not 0, it will
/// resize any type or size of `af.Array`.
pub inline fn resize(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    odim0: i64,
    odim1: i64,
    method: af.InterpType,
) !*af.Array {
    var out: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_resize(
            &out,
            in.array_,
            @intCast(odim0),
            @intCast(odim1),
            method.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, out);
}

/// Transform an input image.
///
/// The `transform` function uses an affine or perspective
/// transform matrix to tranform an input image into a new one.
///
/// If matrix tf is is a 3x2 matrix, an affine transformation will
/// be performed. The matrix operation is applied to each location (x, y)
/// that is then transformed to (x', y') of the new array. Hence the
/// transformation is an element-wise operation.
///
/// The operation is as below:
/// tf = [r00 r10
/// r01 r11
/// t0 t1]
///
/// x' = x * r00 + y * r01 + t0;
/// y' = x * r10 + y * r11 + t1;
///
/// If matrix tf is is a 3x3 matrix, a perspective transformation will be performed.
///
/// The operation is as below.
///
/// tf = [r00 r10 r20
/// r01 r11 r21
/// t0 t1 t2]
///
/// x' = (x * r00 + y * r01 + t0) / (x * r20 + y * r21 + t2);
/// y' = (x * r10 + y * r11 + t1) / (x * r20 + y * r21 + t2);
///
/// The transformation matrix tf should always be of type f32.
///
/// Interpolation types of `af.InterpType.Nearest`, `af.InterpType.Bilinear`,
/// and `af.InterpType.Lower` are allowed.
///
/// Affine transforms can be used for various purposes. `translate`, `scale`,
/// and `skew` are specializations of the transform function.
pub inline fn transform(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    trans: *const af.Array,
    odim0: i64,
    odim1: i64,
    method: af.InterpType,
    inverse: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_transform(
            &arr,
            in.array_,
            trans.array_,
            @intCast(odim0),
            @intCast(odim1),
            method.value(),
            inverse,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}
