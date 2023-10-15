const std = @import("std");
const af = @import("arrayfire.zig");
const zt_shape = @import("../../tensor/shape.zig");
const zt_base = @import("../../tensor/tensor_base.zig");

const DType = @import("../../tensor/types.zig").DType;
const Shape = zt_shape.Shape;
const Dim = zt_shape.Dim;
const MatrixProperty = zt_base.MatrixProperty;
const StorageType = zt_base.StorageType;
const Location = zt_base.Location;
const SortMode = zt_base.SortMode;
const PadType = zt_base.PadType;

/// ArrayFire's DType wrapped as zig enum.
pub const Dtype = enum(af.af_dtype) {
    /// 32-bit floating point values.
    f32,
    /// 32-bit complex floating point values.
    c32,
    /// 64-bit floating point values.
    f64,
    /// 64-bit complex floating point values.
    c64,
    /// 8-bit boolean values.
    b8,
    /// 32-bit signed integral values.
    s32,
    /// 32-bit unsigned integral values.
    u32,
    /// 8-bit unsigned integral values.
    u8,
    /// 64-bit signed integral values.
    s64,
    /// 64-bit unsigned integral values.
    u64,
    /// 16-bit signed integral values.
    s16,
    /// 16-bit unsigned integral values.
    u16,
    /// 16-bit floating point values.
    f16,

    /// Returns value as `af.af_dtype` for calling
    /// directly into ArrayFire.
    pub fn value(self: Dtype) af.af_dtype {
        return @intFromEnum(self);
    }

    /// Returns the equivalent DType.
    pub fn toZtDType(self: Dtype) !DType {
        return switch (self) {
            .f16 => DType.f16,
            .f32 => DType.f32,
            .f64 => DType.f64,
            .b8 => DType.b8,
            .s16 => DType.s16,
            .s32 => DType.s32,
            .s64 => DType.s64,
            .u8 => DType.u8,
            .u16 => DType.u16,
            .u32 => DType.u32,
            .u64 => DType.u64,
            else => error.UnsupportedDType,
        };
    }
};

/// ArrayFire's Source Type wrapped as zig enum.
pub const Source = enum(af.af_source) {
    Device = af.afDevice,
    Host = af.afHost,

    /// Returns the equivalent Location.
    pub fn toZtLocation(self: Source) Location {
        return switch (self) {
            Source.Host => Location.Host,
            Source.Device => Location.Device,
        };
    }

    /// Returns value as `af.af_source` for calling
    /// directly into ArrayFire.
    pub fn value(self: Source) af.af_source {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Interpolation Type type wrapped as zig enum.
pub const InterpType = enum(af.af_interp_type) {
    /// Nearest Interpolation.
    Nearest = af.AF_INTERP_NEAREST,
    /// Linear Interpolation.
    Linear = af.AF_INTERP_LINEAR,
    /// Bilinear Interpolation.
    Bilinear = af.AF_INTERP_BILINEAR,
    /// Cubic Interpolation.
    Cubic = af.AF_INTERP_CUBIC,
    /// Floor Indexed.
    Lower = af.AF_INTERP_LOWER,
    /// Linear Interpolation with cosine smoothing.
    LinearCosine = af.AF_INTERP_LINEAR_COSINE,
    /// Bilinear Interpolation with cosine smoothing.
    BilinearCosine = af.AF_INTERP_BILINEAR_COSINE,
    /// Bicubic Interpolation.
    Bicubic = af.AF_INTERP_BICUBIC,
    /// Cubic Interpolation with Catmull-Rom splines.
    CubicSpline = af.AF_INTERP_CUBIC_SPLINE,
    /// Bicubic Interpolation with Catmull-Rom splines.
    BicubicSpline = af.AF_INTERP_BICUBIC_SPLINE,

    /// Returns value as `af.af_interp_type` for calling
    /// directly into ArrayFire.
    pub fn value(self: InterpType) af.af_interp_type {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Border Type type wrapped as zig enum.
pub const BorderType = enum(af.af_border_type) {
    /// Out of bound values are 0.
    PadZero = af.AF_PAD_ZERO,
    /// Out of bound values are symmetric over the edge.
    PadSym = af.AF_PAD_SYM,
    /// Out of bound values are clamped to the edge.
    PadClampToEdge = af.AF_PAD_CLAMP_TO_EDGE,
    /// Out of bound values are mapped to range of the dimension in cyclic fashion.
    PadPeriodic = af.AF_PAD_PERIODIC,

    /// Returns PadType equivalent.
    pub fn toZtPadType(self: BorderType) !PadType {
        return switch (self) {
            BorderType.Zero => PadType.Constant,
            BorderType.ClampToEdge => PadType.Edge,
            BorderType.Sym => PadType.Symmetric,
            else => error.UnsupportedPadType,
        };
    }

    /// Returns value as `af.af_border_type` for calling
    /// directly into ArrayFire.
    pub fn value(self: BorderType) af.af_border_type {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Connectivity type wrapped as zig enum.
pub const Connectivity = enum(af.af_connectivity) {
    /// Connectivity includes neighbors, North, East,
    /// South and West of current pixel.
    Connectivity4 = af.AF_CONNECTIVITY_4,
    /// Connectivity includes 4-connectivity neigbors
    /// and also those on Northeast, Northwest, Southeast
    /// and Southwest.
    Connectivity8 = af.AF_CONNECTIVITY_8,

    /// Returns value as `af.af_connectivity` for calling
    /// directly into ArrayFire.
    pub fn value(self: Connectivity) af.af_connectivity {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Conv Mode type wrapped as zig enum.
pub const ConvMode = enum(af.af_conv_mode) {
    /// Output of the convolution is the same size as input.
    Default = af.AF_CONV_DEFAULT,
    /// Output of the convolution is signal_len + filter_len - 1.
    Expand = af.AF_CONV_EXPAND,

    /// Returns value as `af.af_conv_mode` for calling
    /// directly into ArrayFire.
    pub fn value(self: ConvMode) af.af_conv_mode {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Conv Domain type wrapped as zig enum.
pub const ConvDomain = enum(af.af_conv_domain) {
    /// ArrayFire automatically picks the right
    /// convolution algorithm.
    Auto = af.AF_CONV_AUTO,
    /// Perform convolution in spatial domain.
    Spatial = af.AF_CONV_SPATIAL,
    /// Perform convolution in frequency domain.
    Freq = af.AF_CONV_FREQ,

    /// Returns value as `af.af_conv_domain` for calling
    /// directly into ArrayFire.
    pub fn value(self: ConvDomain) af.af_conv_domain {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Match Type type wrapped as zig enum.
pub const MatchType = enum(af.af_match_type) {
    /// Match based on Sum of Absolute Differences (SAD).
    SAD = af.AF_SAD,
    /// Match based on Zero mean SAD.
    ZSAD = af.AF_ZSAD,
    /// Match based on Locally scaled SAD.
    LSAD = af.AF_LSAD,
    /// Match based on Sum of Squared Differences (SSD).
    SSD = af.AF_SSD,
    /// Match based on Zero mean SSD.
    ZSSD = af.AF_ZSSD,
    /// Match based on Locally scaled SSD.
    LSSD = af.AF_LSSD,
    /// Match based on Normalized Cross Correlation (NCC).
    NCC = af.AF_NCC,
    /// Match based on Zero mean NCC.
    ZNCC = af.AF_ZNCC,
    /// Match based on Sum of Hamming Distances (SHD).
    SHD = af.AF_SHD,

    /// Returns value as `af.af_match_type` for calling
    /// directly into ArrayFire.
    pub fn value(self: MatchType) af.af_match_type {
        return @intFromEnum(self);
    }
};

/// ArrayFires YCC Standard type wrapped as zig enum.
///
/// Specifies the ITU-R BT "xyz" standard which determines
/// the Kb, Kr values used in colorspace conversion equation.
pub const YCCStandard = enum(af.af_ycc_std) {
    /// ITU-R BT.601 (formerly CCIR 601) standard.
    YCC601 = af.AF_YCC_601,
    /// ITU-R BT.709 standard.
    YCC709 = af.AF_YCC_709,
    /// ITU-R BT.2020 standard.
    YCC2020 = af.AF_YCC_2020,

    /// Returns value as `af.af_ycc_std` for calling
    /// directly into ArrayFire.
    pub fn value(self: YCCStandard) af.af_ycc_std {
        return @intFromEnum(self);
    }
};

/// ArrayFire's CSpace T wrapped as zig enum.
pub const CSpaceT = enum(af.af_cspace_t) {
    /// Grayscale.
    Gray = af.AF_GRAY,
    /// 3-channel RGB.
    RGB = af.AF_RGB,
    /// 3-channel HSV
    HSV = af.AF_HSV,
    /// 3-channel YCbCr
    YCbCr = af.AF_YCbCr,

    /// Returns value as `af.af_cspace_t` for calling
    /// directly into ArrayFire.
    pub fn value(self: CSpaceT) af.af_cspace_t {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Mat Prop type wrapped as zig enum.
pub const MatProp = enum(af.af_mat_prop) {
    /// Default.
    None = af.AF_MAT_NONE,
    /// Data needs to be transposed.
    Trans = af.AF_MAT_TRANS,
    /// Data needs to be conjugate transposed.
    CTrans = af.AF_MAT_CTRANS,
    /// Data needs to be conjugate.
    Conj = af.AF_MAT_CONJ,
    /// Matrix is upper triangular.
    Upper = af.AF_MAT_UPPER,
    /// Matrix is lower triangular.
    Lower = af.AF_MAT_LOWER,
    /// Matrix diagonal contains unitary values.
    DiagUnit = af.AF_MAT_DIAG_UNIT,
    /// Matrix is symmetric.
    Sym = af.AF_MAT_SYM,
    /// Matrix is positive definite.
    PosDef = af.AF_MAT_POSDEF,
    /// Matrix is orthogonal.
    Orthog = af.AF_MAT_ORTHOG,
    /// Matrix is tri diagonal.
    TriDiag = af.AF_MAT_TRI_DIAG,
    /// Matrix is block diagonal.
    BlockDiag = af.AF_MAT_BLOCK_DIAG,

    /// Returns the equivalent MatrixProperty.
    pub fn toZtMatrixProperty(self: MatProp) !MatrixProperty {
        return switch (self) {
            .None => MatrixProperty.None,
            .Trans => MatrixProperty.Transpose,
            else => error.UnsupportedMatrixProperty,
        };
    }

    /// Returns value as `af.af_mat_prop` for calling
    /// directly into ArrayFire.
    pub fn value(self: MatProp) af.af_mat_prop {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Norm type wrapped as zig enum.
pub const NormType = enum(af.af_norm_type) {
    /// Treats the input as a vector and returns
    /// the sum of absolute values.
    Vector1 = af.AF_NORM_VECTOR_1,
    /// Treats the input as a vector and returns
    /// the max of absolute values.
    VectorInf = af.AF_NORM_VECTOR_INF,
    /// Treats the input as a vector and returns
    /// euclidean norm.
    Vector2 = af.AF_NORM_VECTOR_2,
    /// Treats the input as a vector and returns
    /// the p-norm.
    VectorP = af.AF_NORM_VECTOR_P,
    /// Return the max of column sums.
    Matrix1 = af.AF_NORM_MATRIX_1,
    /// Return the max of row sums.
    MatrixInf = af.AF_NORM_MATRIX_INF,
    /// Returns the max singular value).
    /// Currently NOT SUPPORTED!
    Matrix2 = af.AF_NORM_MATRIX_2,
    /// Returns Lpq-norm
    MatrixLPQ = af.AF_NORM_MATRIX_L_PQ,

    /// Returns value as `af.af_norm_type` for calling
    /// directly into ArrayFire.
    pub fn value(self: NormType) af.af_norm_type {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Image Format type wrapped as zig enum.
pub const ImageFormat = enum(af.af_image_format) {
    /// FreeImage Enum for Bitmap File.
    BMP = af.AF_FIF_BMP,
    ICO = af.AF_FIF_ICO,
    JPEG = af.AF_FIF_JPEG,
    JNG = af.AF_FIF_JNG,
    PNG = af.AF_FIF_PNG,
    PPM = af.AF_FIF_PPM,
    PPMRAW = af.AF_FIF_PPMRAW,
    TIFF = af.AF_FIF_TIFF,
    PSD = af.AF_FIF_PSD,
    HDR = af.AF_FIF_HDR,
    EXR = af.AF_FIF_EXR,
    JP2 = af.AF_FIF_JP2,
    RAW = af.AF_FIF_RAW,

    /// Returns value as `af.af_image_format` for calling
    /// directly into ArrayFire.
    pub fn value(self: ImageFormat) af.af_image_format {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Moment Type type wrapped as zig enum.
pub const MomentType = enum(af.af_moment_type) {
    M00 = af.AF_MOMENT_M00,
    M01 = af.AF_MOMENT_M01,
    M10 = af.AF_MOMENT_M10,
    M11 = af.AF_MOMENT_M11,
    FirstOrder = af.AF_MOMENT_FIRST_ORDER,

    /// Returns value as `af.af_moment_type` for calling
    /// directly into ArrayFire.
    pub fn value(self: MomentType) af.af_moment_type {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Homography Type type wrapped as zig enum.
pub const HomographyType = enum(af.af_homography_type) {
    /// Computes homography using RANSAC.
    RANSAC = af.AF_HOMOGRAPHY_RANSAC,
    /// Computes homography using Least Median of Squares.
    LMEDS = af.AF_HOMOGRAPHY_LMEDS,

    /// Returns value as `af.af_homography_type` for calling
    /// directly into ArrayFire.
    pub fn value(self: HomographyType) af.af_homography_type {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Backend type wrapped as zig enum.
pub const Backend = enum(af.af_backend) {
    Default = af.AF_BACKEND_DEFAULT,
    CPU = af.AF_BACKEND_CPU,
    CUDA = af.AF_BACKEND_CUDA,
    OpenCL = af.AF_BACKEND_OPENCL,

    /// Returns value as `af.af_backend` for calling
    /// directly into ArrayFire.
    pub fn value(self: Backend) af.af_backend {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Binary Op type wrapped as zig enum.
pub const BinaryOp = enum(af.af_binary_op) {
    Add = af.AF_BINARY_ADD,
    Mul = af.AF_BINARY_MUL,
    Min = af.AF_BINARY_MIN,
    Max = af.AF_BINARY_MAX,

    /// Returns value as `af.af_binary_op` for calling
    /// directly into ArrayFire.
    pub fn value(self: BinaryOp) af.af_binary_op {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Random Engine Type type wrapped as zig enum.
pub const RandomEngineType = enum(af.af_random_engine_type) {
    DEFAULT = af.AF_RANDOM_ENGINE_DEFAULT,
    THREEFRY = af.AF_RANDOM_ENGINE_THREEFRY,
    MERSENNE = af.AF_RANDOM_ENGINE_MERSENNE,

    /// Returns value as `af.af_random_engine_type` for calling
    /// directly into ArrayFire.
    pub fn value(self: RandomEngineType) af.af_random_engine_type {
        return @intFromEnum(self);
    }
};

/// ArrayFire's ColorMap type wrapped as zig enum.
pub const ColorMap = enum(af.af_colormap) {
    /// Default grayscale map.
    Default = af.AF_COLORMAP_DEFAULT,
    /// Spectrum map (390nm-830nm, in sRGB colorspace).
    Spectrum = af.AF_COLORMAP_SPECTRUM,
    /// Colors, aka. Rainbow.
    Colors = af.AF_COLORMAP_COLORS,
    /// Red hue map.
    Red = af.AF_COLORMAP_RED,
    /// Mood map.
    Mood = af.AF_COLORMAP_MOOD,
    /// Heat map.
    Heat = af.AF_COLORMAP_HEAT,
    /// Blue hue map.
    Blue = af.AF_COLORMAP_BLUE,
    /// Perceptually uniform shades of black-red-yellow.
    Inferno = af.AF_COLORMAP_INFERNO,
    /// Perceptually uniform shades of black-red-white.
    Magma = af.AF_COLORMAP_MAGMA,
    /// Perceptually uniform shades of blue-red-yellow.
    Plasma = af.AF_COLORMAP_PLASMA,
    /// Perceptually uniform shades of blue-green-yellow.
    Viridis = af.AF_COLORMAP_VIRIDIS,

    /// Returns value as `af.af_colormap` for calling
    /// directly into ArrayFire.
    pub fn value(self: ColorMap) af.af_colormap {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Marker Type type wrapped as zig enum.
pub const MarkerType = enum(af.af_marker_type) {
    None = af.AF_MARKER_NONE,
    Point = af.AF_MARKER_POINT,
    Circle = af.AF_MARKER_CIRCLE,
    Square = af.AF_MARKER_SQUARE,
    Triangle = af.AF_MARKER_TRIANGLE,
    Cross = af.AF_MARKER_CROSS,
    Plus = af.AF_MARKER_PLUS,
    Star = af.AF_MARKER_STAR,

    /// Returns value as `af.af_marker_type` for calling
    /// directly into ArrayFire.
    pub fn value(self: MarkerType) af.af_marker_type {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Canny Threshold wrapped as zig enum.
pub const CannyThreshold = enum(af.af_canny_threshold) {
    Manual = af.AF_CANNY_THRESHOLD_MANUAL,
    Otsu = af.AF_CANNY_THRESHOLD_AUTO_OTSU,

    /// Returns value as `af.af_canny_threshold` for calling
    /// directly into ArrayFire.
    pub fn value(self: CannyThreshold) af.af_canny_threshold {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Storage type wrapped as zig enum.
pub const Storage = enum(af.af_storage) {
    /// Storage type is dense.
    Dense = af.AF_STORAGE_DENSE,
    /// Storage type is CSR.
    CSR = af.AF_STORAGE_CSR,
    /// Storage type is CSC.
    CSC = af.AF_STORAGE_CSC,
    /// Storage type is COO.
    COO = af.AF_STORAGE_COO,

    pub fn toZtStorageType(self: Storage) StorageType {
        return switch (self) {
            .Dense => StorageType.Dense,
            .CSR => StorageType.CSR,
            .CSC => StorageType.CSC,
            .COO => StorageType.COO,
        };
    }

    /// Returns value as `af.af_storage` for calling
    /// directly into ArrayFire.
    pub fn value(self: Storage) af.af_storage {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Flux Function type wrapped as zig enum.
pub const FluxFunction = enum(af.af_flux_function) {
    /// Quadratic flux function.
    Quadratic = af.AF_FLUX_QUADRATIC,
    /// Exponential flux function.
    Exp = af.AF_FLUX_EXPONENTIAL,
    /// Default flux function is exponential.
    Default = af.AF_FLUX_DEFAULT,

    /// Returns value as `af.af_topk_function` for calling
    /// directly into ArrayFire.
    pub fn value(self: FluxFunction) af.af_flux_function {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Diffusion Eq type wrapped as zig enum.
pub const DiffusionEq = enum(af.af_diffusion_eq) {
    /// Gradient diffusion equation.
    Grad = af.AF_DIFFUSION_GRAD,
    /// Modified curvature diffusion equation.
    MCDE = af.AF_DIFFUSION_MCDE,
    /// Default option is same as `AfDiffusionEq.Grad`.
    Default = af.AF_DIFFUSION_DEFAULT,

    /// Returns value as `af.af_topk_function` for calling
    /// directly into ArrayFire.
    pub fn value(self: DiffusionEq) af.af_diffusion_eq {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Topk Function type wrapped as zig enum.
pub const TopkFn = enum(af.af_topk_function) {
    /// Top k min values.
    Min = af.AF_TOPK_MIN,
    /// Top k max values.
    Max = af.AF_TOPK_MAX,
    /// Top k default values, which is the same as `AfTopkFn.Max`.
    Default = af.AF_TOPK_DEFAULT,

    /// Returns the equivalent SortMode.
    pub fn toZtSortMode(self: TopkFn) SortMode {
        return switch (self) {
            .Min => SortMode.Ascending,
            .Max => SortMode.Descending,
            .Default => SortMode.Descending,
        };
    }

    /// Returns value as `af.af_topk_function` for calling
    /// directly into ArrayFire.
    pub fn value(self: TopkFn) af.af_topk_function {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Var Bias type wrapped as zig enum.
pub const VarBias = enum(af.af_var_bias) {
    Default = af.AF_VARIANCE_DEFAULT,
    Sample = af.AF_VARIANCE_SAMPLE,
    Population = af.AF_VARIANCE_POPULATION,

    /// Returns value as `af.af_var_bias` for calling
    /// directly into ArrayFire.
    pub fn value(self: VarBias) af.af_var_bias {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Iterative Deconv Algo type wrapped as zig enum.
pub const IterativeDeconvAlgo = enum(af.af_iterative_deconv_algo) {
    /// Landweber Deconvolution.
    Landweber,
    /// Richardson-Lucy Deconvolution.
    RichardsonLucy,
    /// Richardson-Lucy Deconvolution.
    Default,

    /// Returns value as `af.af_iterative_deconv_algo` for calling
    /// directly into ArrayFire.
    pub fn value(self: IterativeDeconvAlgo) af.af_iterative_deconv_algo {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Inverse Deconv Algo type wrapped as zig enum.
pub const InverseDeconvAlgo = enum(af.af_inverse_deconv_algo) {
    Tikhonov = af.AF_INVERSE_DECONV_TIKHONOV,
    Default = af.AF_INVERSE_DECONV_DEFAULT,

    /// Returns value as `af.af_inverse_deconv_algo` for calling
    /// directly into ArrayFire.
    pub fn value(self: InverseDeconvAlgo) af.af_inverse_deconv_algo {
        return @intFromEnum(self);
    }
};

/// ArrayFire's Conv Gradient Type type wrapped as zig enum.
pub const ConvGradientType = enum(af.af_conv_gradient_type) {
    Default = af.AF_CONV_GRADIENT_DEFAULT,
    Filter = af.AF_CONV_GRADIENT_FILTER,
    Data = af.AF_CONV_GRADIENT_DATA,
    Bias = af.AF_CONV_GRADIENT_BIAS,

    /// Returns value as `af.af_conv_gradient_type` for calling
    /// directly into ArrayFire.
    pub fn value(self: ConvGradientType) af.af_conv_gradient_type {
        return @intFromEnum(self);
    }
};

/// ArrayFire's
pub const Dim4 = struct {
    dims: [4]af.dim_t = [_]af.dim_t{1} ** 4,

    pub fn init(dims: []const af.dim_t) Dim4 {
        var self: Dim4 = .{};
        @memcpy(self.dims[0..dims.len], dims);
        return self;
    }

    pub fn ndims(self: *const Dim4) usize {
        const num = self.elements();
        if (num == 0) return 0;
        if (num == 1) return 1;

        if (self.dims[3] != 1) return 4;
        if (self.dims[2] != 1) return 3;
        if (self.dims[1] != 1) return 2;

        return 1;
    }

    pub fn condenseDims(self: Dim4) Dim4 {
        if (self.elements() == 0) {
            return Dim4.init([_]af.dim_t{ 0, 1, 1, 1 });
        }

        // Find the condensed shape
        var new_dims = Dim4.init([_]af.dim_t{ 1, 1, 1, 1 });
        var new_dim_idx: usize = 0;
        for (0..af.AF_MAX_DIMS) |i| {
            if (self.dims[i] != 1) {
                // found a non-1 dim size - populate new_dims
                new_dims.dims[new_dim_idx] = self.dims[i];
                new_dim_idx += 1;
            }
        }
        return new_dims;
    }

    pub fn elements(self: *const Dim4) usize {
        return @intCast(self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3]);
    }

    pub fn getOwnedStrides(self: *const af.Dim4, allocator: std.mem.Allocator, num_dims: usize) ![]Dim {
        var out_shape = std.ArrayList(Dim).init(allocator);
        if (num_dims > @as(usize, @intCast(af.AF_MAX_DIMS))) {
            std.log.debug("afToZtDims: num_dims ({d}) > af.AF_MAX_DIMS ({d} )", .{ num_dims, af.AF_MAX_DIMS });
            return error.ExceedsArrayFireMaxDims;
        }

        // num_dims constraint is enforced by the internal API per condenseDims
        if (num_dims == 1 and self.elements() == 0) {
            // Empty tensor
            try out_shape.resize(1);
            out_shape.items[0] = 0;
            return out_shape.toOwnedSlice();
        }

        // num_dims == 0 --> scalar tensor
        if (num_dims == 0) {
            try out_shape.resize(0);
            return out_shape.toOwnedSlice();
        }

        try out_shape.resize(num_dims);
        for (0..num_dims) |i| out_shape.items[i] = @intCast(self.dims[i]);
        return out_shape.toOwnedSlice();
    }

    pub fn dimsToOwnedShape(self: *const af.Dim4, allocator: std.mem.Allocator) ![]Dim {
        var num_dims = self.ndims();
        var out_shape = std.ArrayList(Dim).init(allocator);
        if (num_dims > @as(usize, @intCast(af.AF_MAX_DIMS))) {
            std.log.debug("afToZtDims: num_dims ({d}) > af.AF_MAX_DIMS ({d} )", .{ num_dims, af.AF_MAX_DIMS });
            return error.ExceedsArrayFireMaxDims;
        }

        // num_dims constraint is enforced by the internal API per condenseDims
        if (num_dims == 1 and self.elements() == 0) {
            // Empty tensor
            try out_shape.resize(1);
            out_shape.items[0] = 0;
            return out_shape.toOwnedSlice();
        }

        // num_dims == 0 --> scalar tensor
        if (num_dims == 0) {
            try out_shape.resize(0);
            return out_shape.toOwnedSlice();
        }

        try out_shape.resize(num_dims);
        for (0..num_dims) |i| out_shape.items[i] = @intCast(self.dims[i]);
        return out_shape.toOwnedSlice();
    }

    /// Formats Shape for printing to writer.
    pub fn format(value: Dim4, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("(", .{});
        for (value.dims, 0..) |v, i| {
            try writer.print("{d}", .{v});
            if (i < value.ndims() - 1) try writer.print(", ", .{});
            if (i == value.ndims() - 1) break;
        }
        return writer.print(")", .{});
    }
};

test "AfDtype.toZtType" {
    try std.testing.expect(try Dtype.f16.toZtDType() == DType.f16);
    try std.testing.expect(try Dtype.f32.toZtDType() == DType.f32);
    try std.testing.expect(try Dtype.f64.toZtDType() == DType.f64);
    try std.testing.expect(try Dtype.b8.toZtDType() == DType.b8);
    try std.testing.expect(try Dtype.s16.toZtDType() == DType.s16);
    try std.testing.expect(try Dtype.s32.toZtDType() == DType.s32);
    try std.testing.expect(try Dtype.s64.toZtDType() == DType.s64);
    try std.testing.expect(try Dtype.u8.toZtDType() == DType.u8);
    try std.testing.expect(try Dtype.u16.toZtDType() == DType.u16);
    try std.testing.expect(try Dtype.u32.toZtDType() == DType.u32);
    try std.testing.expect(try Dtype.u64.toZtDType() == DType.u64);
}
