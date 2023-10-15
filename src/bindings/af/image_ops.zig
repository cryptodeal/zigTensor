const std = @import("std");
const af = @import("arrayfire.zig");

/// Load an image from disk to an `af.Array`.
pub fn loadImage(
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
pub fn saveImage(filename: []const u8, in: *const af.Array) !void {
    try af.AF_CHECK(af.af_save_image(filename.ptr, in.array_), @src());
}

/// Load an image from memory which is stored as a FreeImage
/// stream (FIMEMORY).
///
/// Supported formats include JPG, PNG, PPM and other formats
/// supported by freeimage.
pub fn loadImageMem(allocator: std.mem.Allocator, ptr: ?*const anyopaque) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_load_image_memory(&arr, ptr), @src());
    return af.Array.init(allocator, arr);
}

/// Save an `af.Array` to memory as an image using FreeImage
/// stream (FIMEMORY).
///
/// Supported formats include JPG, PNG, PPM and other formats
/// supported by freeimage.
pub fn saveImageMem(in: *const af.Array, format: af.ImageFormat) !?*anyopaque {
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
pub fn deleteImageMem(ptr: ?*anyopaque) !void {
    try af.AF_CHECK(af.af_delete_image_memory(ptr), @src());
}

/// Loads an image as is original type; returns ptr to the
/// resulting `af.Array`.
///
/// This load image function allows you to load images as
/// u8, u16 or f32 depending on the type of input image
/// as shown by the table below:
///
/// ------------------------------------------------------------------------
/// | Bits per Color (Gray/RGB/RGBA Bits Per Pixel) | Array Type | Range   |
/// | 8 ( 8/24/32 BPP)                              | u8         | 0-255   |
/// | 16 (16/48/64 BPP)                             | u16        | 0-65535 |
/// | 32 (32/96/128 BPP)                            | f32        | 0-1     |
/// ------------------------------------------------------------------------
pub fn loadImageNative(allocator: std.mem.Allocator, filename: []const u8) !*af.Array {
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
///
/// ------------------------------------------------------------------------
/// | Array Type | Bits per Color (Gray/RGB/RGBA Bits Per Pixel) | Range   |
/// | u8         | 8 ( 8/24/32 BPP)                              | 0-255   |
/// | u16        | 16 (16/48/64 BPP)                             | 0-65535 |
/// | f32        | 32 (32/96/128 BPP)                            | 0-1     |
pub fn saveImageNative(filename: []const u8, in: *const af.Array) !void {
    try af.AF_CHECK(
        af.af_save_image_native(
            filename.ptr,
            in.array_,
        ),
        @src(),
    );
}

/// Returns true if ArrayFire was compiled with ImageIO (FreeImage) support.
pub fn isImageIoAvailable() !bool {
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
pub fn resize(
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
///
/// r01 r11
///
/// t0 t1]
///
/// x' = x * r00 + y * r01 + t0;
///
/// y' = x * r10 + y * r11 + t1;
///
/// If matrix tf is is a 3x3 matrix, a perspective transformation will be performed.
///
/// The operation is as below.
///
/// tf = [r00 r10 r20
///
/// r01 r11 r21
///
/// t0 t1 t2]
///
/// x' = (x * r00 + y * r01 + t0) / (x * r20 + y * r21 + t2);
///
/// y' = (x * r10 + y * r11 + t1) / (x * r20 + y * r21 + t2);
///
/// The transformation matrix tf should always be of type f32.
///
/// Interpolation types of `af.InterpType.Nearest`, `af.InterpType.Bilinear`,
/// and `af.InterpType.Lower` are allowed.
///
/// Affine transforms can be used for various purposes. `translate`, `scale`,
/// and `skew` are specializations of the transform function.
pub fn transform(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    trans: *const af.Array,
    odim0: i64,
    odim1: i64,
    method: af.InterpType,
    inv: bool,
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
            inv,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// `transform` function, but requires a pre-allocated array.
///
/// See docs for `transform` for more information.
pub fn transformV2(
    out: *af.Array,
    in: *const af.Array,
    trans: *const af.Array,
    odim0: i64,
    odim1: i64,
    method: af.InterpType,
    inv: bool,
) !void {
    return af.AF_CHECK(
        af.af_transform_v2(
            &out.array_,
            in.array_,
            trans.array_,
            @intCast(odim0),
            @intCast(odim1),
            method.value(),
            inv,
        ),
        @src(),
    );
}

/// Transform coordinates.
pub fn transformCoordinates(
    allocator: std.mem.Allocator,
    tf: *const af.Array,
    d0: f32,
    d1: f32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_transform_coordinates(
            &arr,
            tf.array_,
            d0,
            d1,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Rotate an input image or `af.Array`.
///
/// The rotation is done counter-clockwise, with an angle
/// theta (in radians), using a specified method of interpolation
/// to determine the values of the output array. Six types of
/// interpolation are currently supported:
/// - `af.InterpType.Nearest`: nearest value to the location
/// - `af.InterpType.Bilinear`: weighted interpolation
/// - `af.InterpType.BilinearCosine`: bilinear interpolation with cosine smoothing
/// - `af.InterpType.Bicubic`: bicubic interpolation
/// - `af.InterpType.BicubicSpline`: bicubic interpolation with Catmull-Rom splines
/// - `af.InterpType.Lower`: floor indexed
///
/// Since the output image still needs to be an upright box, crop determines how to
/// bound the output image, given the now-rotated image.
pub fn rotate(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    theta: f32,
    crop: bool,
    method: af.InterpType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_rotate(
            &arr,
            in.array_,
            theta,
            crop,
            method.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Translate an input image.
///
/// Translating an image is moving it along 1st and 2nd dimensions
/// by trans0 and trans1. Positive values of these will move the
/// data towards negative x and negative y whereas negative values
/// of these will move the positive right and positive down.
///
/// To specify an output dimension, use the odim0 and odim1 for dim0
/// and dim1 respectively. The size of 2rd and 3rd dimension is same
/// as input. If odim0 and odim1 and not defined, then the output
/// dimensions are same as the input dimensions and the data out of
/// bounds will be discarded.
///
/// All new values that do not map to a location of the input `af.Array`
/// are set to 0.
///
/// Translate is a special case of the `transform` function.
pub fn translate(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    trans0: f32,
    trans1: f32,
    odim0: i64,
    odim1: i64,
    method: af.InterpType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_translate(
            &arr,
            in.array_,
            trans0,
            trans1,
            @intCast(odim0),
            @intCast(odim1),
            method.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Scale an input image.
///
/// Scale is the same functionality as `resize` except that
/// the scale function uses the transform kernels. The other
/// difference is that scale does not set boundary values to
/// be the boundary of the input `af.Array`. Instead these are
/// set to 0.
///
/// Scale is a special case of the `transform` function.
pub fn scale(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    scale0: f32,
    scale1: f32,
    odim0: i64,
    odim1: i64,
    method: af.InterpType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_scale(
            &arr,
            in.array_,
            scale0,
            scale1,
            @intCast(odim0),
            @intCast(odim1),
            method.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Skew an input image.
///
/// Skew function skews the input `af.Array` along dim0
/// by skew0 and along dim1 by skew1. The skew arguments
/// are in radians. Skewing the data means the data remains
/// parallel along 1 dimensions but the other dimensions
/// gets moved along based on the angle. If both skew0 and
/// skew1 are specified, then the data will be skewed along
/// both directions.
///
/// Explicit output dimensions can be specified using odim0
/// and odim1.
///
/// All new values that do not map to a location of the input
/// `af.Array` are set to 0.
///
/// Skew is a special case of the `transform` function.
pub fn skew(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    skew0: f32,
    skew1: f32,
    odim0: i64,
    odim1: i64,
    method: af.InterpType,
    inv: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_skew(
            &arr,
            in.array_,
            skew0,
            skew1,
            @intCast(odim0),
            @intCast(odim1),
            method.value(),
            inv,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Histogram of input data.
///
/// A histogram is a representation of the distribution of
/// given data. This representation is essentially a graph
/// consisting of the data range or domain on one axis and
/// frequency of occurence on the other axis. All the data
/// in the domain is counted in the appropriate bin. The
/// total number of elements belonging to each bin is known
/// as the bin's frequency.
///
/// The regular histogram function creates bins of equal size
/// between the minimum and maximum of the input data (min and
/// max are calculated internally). The histogram min-max function
/// takes input parameters minimum and maximum, and divides the bins
/// into equal sizes within the range specified by min and max
/// parameters. All values less than min in the data range are placed
/// in the first (min) bin and all values greater than max will be
/// placed in the last (max) bin.
pub fn histogram(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    nbins: u32,
    minval: f64,
    maxval: f64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_histogram(
            &arr,
            in.array_,
            nbins,
            minval,
            maxval,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Dilation(morphological operator) for images.
///
/// The dilation function takes two pieces of data as inputs.
/// The first is the input image to be morphed, and the second
/// is the mask indicating the neighborhood around each pixel
/// to match.
///
/// In dilation, for each pixel, the mask is centered at the pixel.
/// If the center pixel of the mask matches the corresponding pixel
/// on the image, then the mask is accepted. If the center pixels do
/// not match, then the mask is ignored and no changes are made.
///
/// For further reference, see: [Dilation (morphology)](https://en.wikipedia.org/wiki/Dilation_(morphology)).
pub fn dilate(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    mask: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_dilate(&arr, in.array_, mask.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Dilation(morphological operator) for volumes.
///
/// Dilation for a volume is similar to the way dilation
/// works on an image. Only difference is that the masking
/// operation is performed on a volume instead of a rectangular region.
///
/// For further reference, see: [Dilation (morphology)](https://en.wikipedia.org/wiki/Dilation_(morphology)).
pub fn dilate3(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    mask: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_dilate3(&arr, in.array_, mask.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Erosion(morphological operator) for images.
///
/// The erosion function is a morphological transformation
/// on an image that requires two inputs. The first is the
/// image to be morphed, and the second is the mask indicating
/// neighborhood that must be white in order to preserve each pixel.
///
/// In erode, for each pixel, the mask is centered at the pixel.
/// If each pixel of the mask matches the corresponding pixel on the
/// image, then no change is made. If there is at least one mismatch,
/// then pixels are changed to the background color (black).
///
/// For further reference, see: [Erosion (morphology)](https://en.wikipedia.org/wiki/Erosion_(morphology)).
pub fn erode(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    mask: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_erode(&arr, in.array_, mask.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Erosion(morphological operator) for volumes.
///
/// Erosion for a volume is similar to the way erosion works
/// on an image. Only difference is that the masking operation
/// is performed on a volume instead of a rectangular region.
///
/// For further reference, see: [Erosion (morphology)](https://en.wikipedia.org/wiki/Erosion_(morphology)).
pub fn erode3(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    mask: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_erode3(&arr, in.array_, mask.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Bilateral Filter.
///
/// A bilateral filter is a edge-preserving filter that
/// reduces noise in an image. The intensity of each pixel
/// is replaced by a weighted average of the intensities of
/// nearby pixels. The weights follow a Gaussian distribution
/// and depend on the distance as well as the color distance.
///
/// The bilateral filter requires the size of the filter
/// (in pixels) and the upper bound on color values, N, where
/// pixel values range from 0–N inclusively.
///
/// The return type of the array is f64 for f64 input, f32 for
/// all other input types.
pub fn bilateral(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    spatial_sigma: f32,
    chromatic_sigma: f32,
    isColor: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_bilateral(
        &arr,
        in.array_,
        spatial_sigma,
        chromatic_sigma,
        isColor,
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Meanshift Filter.
///
/// A meanshift filter is an edge-preserving smoothing filter
/// commonly used in object tracking and image segmentation.
///
/// This filter replaces each pixel in the image with the mean
/// of the values within a given given color and spatial radius.
/// The meanshift filter is an iterative algorithm that continues
/// until a maxium number of iterations is met or until the value
/// of the means no longer changes.
pub fn meanShift(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    spatial_sigma: f32,
    chromatic_sigma: f32,
    iter: u32,
    is_color: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_mean_shift(
        &arr,
        in.array_,
        spatial_sigma,
        chromatic_sigma,
        iter,
        is_color,
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Find minimum value from a window.
///
/// `minfilt` finds the smallest value from a 2D window
/// and assigns it to the current pixel.
pub fn minfilt(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    wind_length: i64,
    wind_width: i64,
    edge_pad: af.BorderType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_minfilt(
        &arr,
        in.array_,
        @intCast(wind_length),
        @intCast(wind_width),
        edge_pad.value(),
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Find maximum value from a window.
///
/// `maxfilt` finds the maximum value from a 2D window
/// and assigns it to the current pixel.
pub fn maxfilt(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    wind_length: i64,
    wind_width: i64,
    edge_pad: af.BorderType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_maxfilt(
        &arr,
        in.array_,
        @intCast(wind_length),
        @intCast(wind_width),
        edge_pad.value(),
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Find blobs in given image.
///
/// Given a binary image (with zero representing background pixels),
/// regions computes a floating point image where each connected
/// component is labeled from 1 to N, the total number of components
/// in the image.
///
/// A component is defined as one or more nonzero pixels that are connected
/// by the specified connectivity (either 4-way(AF_CONNECTIVITY_4) or
/// 8-way(AF_CONNECTIVITY_8)) in two dimensions.
pub fn regions(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    connectivity: af.Connectivity,
    ty: af.Dtype,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_regions(
        &arr,
        in.array_,
        connectivity.value(),
        ty.value(),
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Sobel Operators.
///
/// Sobel operators perform a 2-D spatial gradient measurement
/// on an image to emphasize the regions of high spatial frequency,
/// namely edges. A more in depth discussion on it can be found [here](https://en.wikipedia.org/wiki/Sobel_operator).
pub fn sobelOperator(
    allocator: std.mem.Allocator,
    img: *const af.Array,
    ker_size: u32,
) !struct { dx: *af.Array, dy: *af.Array } {
    var dx: af.af_array = undefined;
    var dy: af.af_array = undefined;
    try af.AF_CHECK(af.af_sobel_operator(
        &dx,
        &dy,
        img.array_,
        ker_size,
    ), @src());
    return .{
        .dx = try af.Array.init(allocator, dx),
        .dy = try af.Array.init(allocator, dy),
    };
}

/// RGB to Grayscale colorspace converter.
///
/// RGB (Red, Green, Blue) is the most common format used
/// in computer imaging. RGB stores individual values for
/// red, green and blue, and hence the 3 values per pixel.
/// A combination of these three values produces the gamut
/// of unique colors.
///
/// Grayscale is a single channel color space where pixel
/// value ranges from 0 to 1. Zero represents black, one
/// represent white and any value between zero & one is a
/// gray value.
pub fn rgb2Gray(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    rPercent: f32,
    gPercent: f32,
    bPercent: f32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_rgb2gray(
        &arr,
        in.array_,
        rPercent,
        gPercent,
        bPercent,
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Grayscale to RGB colorspace converter.
///
/// Grayscale is a single channel color space where pixel
/// value ranges from 0 to 1. Zero represents black, one
/// represent white and any value between zero & one is
/// a gray value.
///
/// RGB (Red, Green, Blue) is the most common format used
/// in computer imaging. RGB stores individual values for
/// red, green and blue, and hence the 3 values per pixel.
/// A combination of these three values produces the gamut
/// of unique colors.
pub fn gray2Rgb(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    rFactor: f32,
    gFactor: f32,
    bFactor: f32,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_gray2rgb(
        &arr,
        in.array_,
        rFactor,
        gFactor,
        bFactor,
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Histogram equalization of input image.
///
/// Histogram equalization is a method in image processing
/// of contrast adjustment using the image's histogram.
///
/// Data normalization via histogram equalization.
pub fn histEqual(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    hist: *const af.Array,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_hist_equal(
        &arr,
        in.array_,
        hist.array_,
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Creates a Gaussian Kernel.
///
/// This function creates a kernel of a specified size
/// that contains a Gaussian distribution. This distribution
/// is normalized to one. This is most commonly used when
/// performing a Gaussian blur on an image. The function takes
/// two sets of arguments, the size of the kernel (width and
/// height in pixels) and the sigma parameters (for row and column)
/// which effect the distribution of the weights in the y and x
/// directions, respectively.
///
/// Changing sigma causes the weights in each direction to vary.
/// Sigma is calculated internally as (0.25 * rows + 0.75) for rows
/// and similarly for columns.
pub fn gaussianKernel(
    allocator: std.mem.Allocator,
    rows: i32,
    cols: i32,
    sigma_r: f64,
    sigma_c: f64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_gaussian_kernel(
        &arr,
        rows,
        cols,
        sigma_r,
        sigma_c,
    ), @src());
    return af.Array.init(allocator, arr);
}

/// HSV to RGB colorspace converter.
///
/// HSV (Hue, Saturation, Value), also known as HSB (hue, saturation,
/// brightness), is often used by artists because it is more natural
/// to think about a color in terms of hue and saturation than in terms
/// of additive or subtractive color components (as in RGB). HSV is a
/// transformation of RGB colorspace; its components and colorimetry are
/// relative to the RGB colorspace from which it was derived. Like RGB,
/// HSV also uses 3 values per pixel.
///
/// RGB (Red, Green, Blue) is the most common format used in computer
/// imaging. RGB stores individual values for red, green and blue, and
/// hence the 3 values per pixel. A combination of these three values
/// produces the gamut of unique colors.
pub fn hsv2Rgb(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_hsv2rgb(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

/// RGB to HSV colorspace converter.
///
/// RGB (Red, Green, Blue) is the most common format used in
/// computer imaging. RGB stores individual values for red,
/// green and blue, and hence the 3 values per pixel. A combination
/// of these three values produces the gamut of unique colors.
///
/// HSV (Hue, Saturation, Value), also known as HSB (hue, saturation,
/// brightness), is often used by artists because it is more natural
/// to think about a color in terms of hue and saturation than in terms
/// of additive or subtractive color components (as in RGB). HSV is a
/// transformation of RGB colorspace; its components and colorimetry are
/// relative to the RGB colorspace from which it was derived. Like RGB,
/// HSV also uses 3 values per pixel.
pub fn rgb2Hsv(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_rgb2hsv(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

/// Colorspace conversion function.
///
/// RGB (Red, Green, Blue) is the most common format used
/// in computer imaging. RGB stores individual values for red,
/// green and blue, and hence the 3 values per pixel. A combination
/// of these three values produces the gamut of unique colors.
///
/// HSV (Hue, Saturation, Value), also known as HSB (hue, saturation,
/// brightness), is often used by artists because it is more natural
/// to think about a color in terms of hue and saturation than in terms
/// of additive or subtractive color components (as in RGB). HSV is a
/// transformation of RGB colorspace; its components and colorimetry are
/// relative to the RGB colorspace from which it was derived. Like RGB,
/// HSV also uses 3 values per pixel.
///
/// Grayscale is a single channel color space where pixel value ranges
/// from 0 to 1. Zero represents black, one represent white and any value
/// between zero & one is a gray value.
///
/// Supported conversions:
///
/// -----------------
/// | From  | To    |
/// | .RGB  | .Gray |
/// | .Gray | .RGB  |
/// | .RGB  | .HSV  |
/// | .HSV  | .RGB  |
/// -----------------
pub fn colorSpace(
    allocator: std.mem.Allocator,
    image: *const af.Array,
    to: af.CSpaceT,
    from: af.CSpaceT,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_color_space(
        &arr,
        image.array_,
        to.value(),
        from.value(),
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Rearrange windowed sections of an `af.Array` into columns (or rows).
///
/// A moving window of size wx × wy captures sections of the input `af.Array`,
/// and flattens them into columns (or rows if is_column is false) of the output
/// `af.Array`. It starts at the top-left section of the input array and moves in
/// column-major order, each time moving in strides of sx units along the column
/// and sy units along the row, whenever it exhausts a column. When the remainder
/// of the column or row is not big enough to accomodate the window, that remainder
/// is skipped and the window moves on.
///
/// Optionally, one can specify that the input image's border be padded (with zeros)
/// before the moving window starts capturing sections. The width of the padding is
/// defined by px for the top and bottom and py for the left and right sides, with
/// maximum values of wx-1 and wy-1, respectively. The moving window then captures
/// sections as if the padding is part of the input image, and thus the padding also
/// becomes part of the output array's columns.
pub fn unwrap(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    wx: i64,
    wy: i64,
    sx: i64,
    sy: i64,
    px: i64,
    py: i64,
    is_column: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_unwrap(
        &arr,
        in.array_,
        @intCast(wx),
        @intCast(wy),
        @intCast(sx),
        @intCast(sy),
        @intCast(px),
        @intCast(py),
        is_column,
    ), @src());
    return af.Array.init(allocator, arr);
}

/// Performs the opposite of `unwrap`.
///
/// More specifically, wrap takes each column (or row if is_column
/// is false) of the m×n input `af.Array` and reshapes them into wx × wy
/// patches (where m= wx × wy) of the ox × oy output `af.Array`. Wrap is
/// typically used on an `af.Array` that has been previously unwrapped - for
/// example, in the case of image processing, one can unwrap an image,
/// process the unwrapped `af.Array`, and then compose it back into an image
/// using wrap.
///
/// The process can be visualized as a moving window taking a column from
/// the input, reshaping it into a patch, and then placing that patch on
/// its corresponding position in the output. It starts placing a patch
/// on the output's top-left corner, then moves sx units along the column,
/// and sy units along the row whenever it exhausts a column. If padding
/// exists in the input `af.Array` (gray-filled boxes), which typically happens
/// when padding was applied on the previous unwrap, then px and py must be
/// specified in order for the padding to be removed on the output `af.Array`.
pub fn wrap(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    ox: i64,
    oy: i64,
    wx: i64,
    wy: i64,
    sx: i64,
    sy: i64,
    px: i64,
    py: i64,
    is_column: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_wrap(
        &arr,
        in.array_,
        @intCast(ox),
        @intCast(oy),
        @intCast(wx),
        @intCast(wy),
        @intCast(sx),
        @intCast(sy),
        @intCast(px),
        @intCast(py),
        is_column,
    ), @src());
    return af.Array.init(allocator, arr);
}

/// `wrap` function, but requires a pre-allocated array.
///
/// See docs for `wrap` for more information.
pub fn wrapV2(
    out: *af.Array,
    in: *const af.Array,
    ox: i64,
    oy: i64,
    wx: i64,
    wy: i64,
    sx: i64,
    sy: i64,
    px: i64,
    py: i64,
    is_column: bool,
) !void {
    try af.AF_CHECK(af.af_wrap_v2(
        &out.array_,
        in.array_,
        @intCast(ox),
        @intCast(oy),
        @intCast(wx),
        @intCast(wy),
        @intCast(sx),
        @intCast(sy),
        @intCast(px),
        @intCast(py),
        is_column,
    ), @src());
}

/// Summed Area Tables; returns ptr to the resulting `af.Array`.
pub fn sat(allocator: std.mem.Allocator, in: *const af.Array) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(af.af_sat(&arr, in.array_), @src());
    return af.Array.init(allocator, arr);
}

/// YCbCr to RGB colorspace converter.
///
/// YCbCr is a family of color spaces used as a part of
/// the color image pipeline in video and digital photography
/// systems where Y is luma component and Cb & Cr are the
/// blue-difference and red-difference chroma components.
///
/// RGB (Red, Green, Blue) is the most common format used in
/// computer imaging. RGB stores individual values for red,
/// green and blue, and hence the 3 values per pixel. A combination
/// of these three values produces the gamut of unique colors.
///
/// Input `af.Array` to this function should be of real data with
/// the following range in their respective channels.
/// - Y -> [16,219]
/// - Cb -> [16,240]
/// - Cr -> [16,240]
pub fn ycbcr2Rgb(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    standard: af.YCCStandard,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_ycbcr2rgb(
            &arr,
            in.array_,
            standard.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// RGB to YCbCr colorspace converter.
///
/// RGB (Red, Green, Blue) is the most common format used in
/// computer imaging. RGB stores individual values for red,
/// green and blue, and hence the 3 values per pixel. A combination
/// of these three values produces the gamut of unique colors.
///
/// YCbCr is a family of color spaces used as a part of the color image
/// pipeline in video and digital photography systems where Y is luma
/// component and Cb & Cr are the blue-difference and red-difference
/// chroma components.
///
/// Input `af.Array` to this function should be of real data in the range [0,1].
pub fn rgb2Ycbcr(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    standard: af.YCCStandard,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_rgb2ycbcr(
            &arr,
            in.array_,
            standard.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// The `moments` function allows for finding different
/// properties of image regions.
///
/// Currently, ArrayFire calculates all first order moments.
/// The moments are defined within the `af.MomentType` enum.
pub fn moments(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    moment: af.MomentType,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_moments(
            &arr,
            in.array_,
            moment.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

// TODO: possibly returns `[]f64`; pending triage.

/// Calculating image moment(s) of a single image.
pub fn momentsAll(in: *const af.Array, moment: af.MomentType) !f64 {
    var res: f64 = undefined;
    try af.AF_CHECK(
        af.af_moments_all(
            &res,
            in.array_,
            moment.value(),
        ),
        @src(),
    );
    return res;
}

/// Canny Edge Detector.
///
/// The Canny edge detector is an edge detection operator
/// that uses a multi-stage algorithm to detect a wide
/// range of edges in images. A more in depth discussion
/// on it can be found [here](https://en.wikipedia.org/wiki/Canny_edge_detector).
pub fn canny(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    threshold_type: af.CannyThreshold,
    low_threshold_ratio: f32,
    high_threshold_ratio: f32,
    sobel_window_length: u32,
    is_fast: bool,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_canny(
            &arr,
            in.array_,
            threshold_type.value(),
            low_threshold_ratio,
            high_threshold_ratio,
            sobel_window_length,
            is_fast,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Anisotropic Smoothing Filter.
///
/// Anisotropic diffusion algorithm aims at removing noise
/// in the images while preserving important features such
/// as edges. The algorithm essentially creates a scale space
/// representation of the original image, where image from
/// previous step is used to create a new version of blurred
/// image using the diffusion process. Standard isotropic
/// diffusion methods such as gaussian blur, doesn't take into
/// account the local content(smaller neighborhood of current
/// processing pixel) while removing noise. Anisotropic diffusion
/// uses the flux equations given below to achieve that. Flux
/// equation is the formula used by the diffusion process to
/// determine how much a pixel in neighborhood should contribute
/// to the blurring operation being done at the current pixel at
/// a given iteration.
///
/// The flux function can be either exponential or quadratic.
pub fn anisotropicDiffusion(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    timestep: f32,
    conductance: f32,
    iterations: u32,
    fftype: af.FluxFunction,
    diffusion_kind: af.DiffusionEq,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_anisotropic_diffusion(
            &arr,
            in.array_,
            timestep,
            conductance,
            @intCast(iterations),
            fftype.value(),
            diffusion_kind.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Iterative Deconvolution.
///
/// Iterative deconvolution function accepts `af.Array` of the following
/// types only:
/// - f32
/// - s16
/// - u16
/// - u8
pub fn iterativeDeconv(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    ker: *const af.Array,
    iterations: u32,
    relax_factor: f32,
    algo: af.IterativeDeconvAlgo,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_iterative_deconv(
            &arr,
            in.array_,
            ker.array_,
            @intCast(iterations),
            relax_factor,
            algo.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Inverse Deconvolution.
///
/// Inverse deconvolution is an linear algorithm i.e. they
/// are non-iterative in nature and usually faster than
/// iterative deconvolution algorithms.
///
/// Depending on the values passed on to the enum `af.InverseDeconvAlgo`,
/// different equations are used to compute the final result.
///
/// Inverse deconvolution function accepts `af.Array` of the following
/// types only:
/// - f32
/// - s16
/// - u16
/// - u8
pub fn inverseDeconv(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    psf: *const af.Array,
    gamma: f32,
    algo: af.InverseDeconvAlgo,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_inverse_deconv(
            &arr,
            in.array_,
            psf.array_,
            gamma,
            algo.value(),
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}

/// Segment image based on similar pixel characteristics.
///
/// This filter is similar to `regions` (connected components)
/// with additional criteria for segmentation. In `regions`, all
/// connected (`af.Connectivity`) pixels connected are considered
/// to be a single component. In this variation of connected components,
/// pixels having similar pixel statistics of the neighborhoods around
/// a given set of seed points are grouped together.
///
/// The parameter radius determines the size of neighborhood around a seed
/// point.
///
/// Mean (μ) and Variance (σ2) are the pixel statistics that are computed
/// across all neighborhoods around the given set of seed points. The pixels
/// which are connected to seed points and lie in the confidence interval
/// ([μ−α∗σ,μ+α∗σ] where α is the parameter multiplier) are grouped.
/// Multiplier can be used to control the width of the confidence interval.
///
/// This filter follows an iterative approach for fine tuning the segmentation.
/// An initial segmenetation followed by a finite number (iter) of segmentations
/// are performed. The user provided parameter iter is only a request and the
/// algorithm can preempt the execution if σ2 approaches zero. The initial
/// segmentation uses the mean and variance calculated from the neighborhoods of
/// all the seed points. For subsequent segmentations, all pixels in the previous
/// segmentation are used to re-calculate the mean and variance (as opposed to using
/// the pixels in the neighborhood of the seed point).
pub fn confidenceCC(
    allocator: std.mem.Allocator,
    in: *const af.Array,
    seedx: *const af.Array,
    seedy: *const af.Array,
    radius: u32,
    multiplier: u32,
    iter: i32,
    segmented_value: f64,
) !*af.Array {
    var arr: af.af_array = undefined;
    try af.AF_CHECK(
        af.af_confidence_cc(
            &arr,
            in.array_,
            seedx.array_,
            seedy.array_,
            @intCast(radius),
            @intCast(multiplier),
            @intCast(iter),
            segmented_value,
        ),
        @src(),
    );
    return af.Array.init(allocator, arr);
}
