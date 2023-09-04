const std = @import("std");
const af = @import("ArrayFire.zig");

/// Creates an `af.Window` struct with specified `width`, `height`, and `title`.
pub inline fn createWindow(
    allocator: std.mem.Allocator,
    width: i32,
    height: i32,
    title: []const u8,
) !*af.Window {
    var out: af.af_window = undefined;
    try af.AF_CHECK(
        af.af_create_window(
            &out,
            @intCast(width),
            @intCast(height),
            title.ptr,
        ),
        @src(),
    );
    return af.Window.initFromWindow(allocator, out);
}

/// Set the start position where the window will appear.
pub inline fn setPosition(wind: *const af.Window, x: u32, y: u32) !void {
    try af.AF_CHECK(
        af.af_set_position(
            wind.window_,
            @intCast(x),
            @intCast(y),
        ),
        @src(),
    );
}

/// Set the title of the window.
pub inline fn setTitle(wind: *const af.Window, title: []const u8) !void {
    try af.AF_CHECK(
        af.af_set_title(
            wind.window_,
            title.ptr,
        ),
        @src(),
    );
}

/// Set the window size.
pub inline fn setSize(wind: *const af.Window, w: u32, h: u32) !void {
    try af.AF_CHECK(
        af.af_set_size(
            wind.window_,
            @intCast(w),
            @intCast(h),
        ),
        @src(),
    );
}

/// Draws the input `af.Array` as an image to the window.
pub inline fn drawImage(wind: *const af.Window, in: *const af.Array, props: *const af.Cell) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_image(
            wind.window_,
            in.array_,
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array` as a plot to the window.
pub inline fn drawPlot(
    wind: *const af.Window,
    X: *const af.Array,
    Y: *const af.Array,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_plot(
            wind.window_,
            X.array_,
            Y.array_,
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array` as a 3D line plot to the window.
pub inline fn drawPlot3(
    wind: *const af.Window,
    P: *const af.Array,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_plot3(
            wind.window_,
            P.array_,
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array` as as a 2D or 3D plot to the window.
pub inline fn drawPlotNd(
    wind: *const af.Window,
    P: *const af.Array,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_plot_nd(
            wind.window_,
            P.array_,
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array`s as as a 2D plot to the window.
pub inline fn drawPlot2d(
    wind: *const af.Window,
    X: *const af.Array,
    Y: *const af.Array,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_plot2(
            wind.window_,
            X.array_,
            Y.array_,
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array`s as as a 3D plot to the window.
pub inline fn drawPlot3d(
    wind: *const af.Window,
    X: *const af.Array,
    Y: *const af.Array,
    Z: *const af.Array,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_plot3(
            wind.window_,
            X.array_,
            Y.array_,
            Z.array_,
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array`s as as a scatter plot to the window.
pub inline fn drawScatter(
    wind: *const af.Window,
    X: *const af.Array,
    Y: *const af.Array,
    marker: af.MarkerType,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_scatter(
            wind.window_,
            X.array_,
            Y.array_,
            marker.value(),
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array` as as a scatter plot to the window.
pub inline fn drawScatter3(
    wind: *const af.Window,
    P: *const af.Array,
    marker: af.MarkerType,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_scatter(
            wind.window_,
            P.array_,
            marker.value(),
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array` as as a 2D or 3D scatter plot to the window.
pub inline fn drawScatterNd(
    wind: *const af.Window,
    P: *const af.Array,
    marker: af.MarkerType,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_scatter_nd(
            wind.window_,
            P.array_,
            marker.value(),
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array`s as as a 2D scatter plot to the window.
pub inline fn drawScatter2d(
    wind: *const af.Window,
    X: *const af.Array,
    Y: *const af.Array,
    marker: af.MarkerType,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_scatter_2d(
            wind.window_,
            X.array_,
            Y.array_,
            marker.value(),
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array`s as as a 3D scatter plot to the window.
pub inline fn drawScatter3d(
    wind: *const af.Window,
    X: *const af.Array,
    Y: *const af.Array,
    Z: *const af.Array,
    marker: af.MarkerType,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_scatter_3d(
            wind.window_,
            X.array_,
            Y.array_,
            Z.array_,
            marker.value(),
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array` as a histogram to the window.
pub inline fn drawHist(
    wind: *const af.Window,
    X: *const af.Array,
    minval: f64,
    maxval: f64,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_hist(
            wind.window_,
            X.array_,
            minval,
            maxval,
            &cell,
        ),
        @src(),
    );
}

/// Renders the input `af.Array`s as a 3D surface plot to the window.
pub inline fn drawSurface(
    wind: *const af.Window,
    xVals: *const af.Array,
    yVals: *const af.Array,
    S: *const af.Array,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_surface(
            wind.window_,
            xVals.array_,
            yVals.array_,
            S.array_,
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array`s as a 2D or 3D vector field to the window.
pub inline fn drawVectorFieldNd(
    wind: *const af.Window,
    points: *const af.Array,
    directions: *const af.Array,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_vector_field_nd(
            wind.window_,
            points.array_,
            directions.array_,
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array`s as a 3D vector field to the window.
pub inline fn drawVectorField3d(
    wind: *const af.Window,
    xPoints: *const af.Array,
    yPoints: *const af.Array,
    zPoints: *const af.Array,
    xDirs: *const af.Array,
    yDirs: *const af.Array,
    zDirs: *const af.Array,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_vector_field_3d(
            wind.window_,
            xPoints.array_,
            yPoints.array_,
            zPoints.array_,
            xDirs.array_,
            yDirs.array_,
            zDirs.array_,
            &cell,
        ),
        @src(),
    );
}

/// Draws the input `af.Array`s as a 2D vector field to the window.
pub inline fn drawVectorField2d(
    wind: *const af.Window,
    xPoints: *const af.Array,
    yPoints: *const af.Array,
    xDirs: *const af.Array,
    yDirs: *const af.Array,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_draw_vector_field_3d(
            wind.window_,
            xPoints.array_,
            yPoints.array_,
            xDirs.array_,
            yDirs.array_,
            &cell,
        ),
        @src(),
    );
}

/// Setup grid layout for multiview mode in a window.
pub inline fn grid(wind: *const af.Window, rows: i32, cols: i32) !void {
    try af.AF_CHECK(
        af.af_grid(
            wind.window_,
            @intCast(rows),
            @intCast(cols),
        ),
        @src(),
    );
}

/// Sets axes limits for a histogram/plot/surface/vector field.
pub inline fn setAxesLimitsCompute(
    wind: *const af.Window,
    x: *const af.Array,
    y: *const af.Array,
    z: *const af.Array,
    exact: bool,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_set_axes_limits_compute(
            wind.window_,
            x.array_,
            y.array_,
            z.array_,
            exact,
            &cell,
        ),
        @src(),
    );
}

/// Sets axes limits for a 2D histogram/plot/vector field.
pub inline fn setAxesLimits2d(
    wind: *const af.Window,
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    exact: bool,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_set_axes_limits_2d(
            wind.window_,
            xmin,
            xmax,
            ymin,
            ymax,
            exact,
            &cell,
        ),
        @src(),
    );
}

/// Sets axes limits for a 3D histogram/plot/vector field.
pub inline fn setAxesLimits3d(
    wind: *const af.Window,
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    zmin: f32,
    zmax: f32,
    exact: bool,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_set_axes_limits_3d(
            wind.window_,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            exact,
            &cell,
        ),
        @src(),
    );
}

/// Sets axes titles for histogram/plot/surface/vector field.
pub inline fn setAxesTitles(
    wind: *const af.Window,
    xtitle: []const u8,
    ytitle: []const u8,
    ztitle: []const u8,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_set_axes_titles(
            wind.window_,
            xtitle.ptr,
            ytitle.ptr,
            ztitle.ptr,
            &cell,
        ),
        @src(),
    );
}

/// Sets axes labels formats for charts.
pub inline fn setAxesLabelFormat(
    wind: *const af.Window,
    xformat: []const u8,
    yformat: []const u8,
    zformat: []const u8,
    props: *const af.Cell,
) !void {
    const cell = props.value();
    try af.AF_CHECK(
        af.af_set_axes_label_format(
            wind.window_,
            xformat.ptr,
            yformat.ptr,
            zformat.ptr,
            &cell,
        ),
        @src(),
    );
}

/// Shows the window.
pub inline fn show(wind: *const af.Window) !void {
    try af.AF_CHECK(af.af_show(wind.window_), @src());
}

/// Checks whether window is marked for close.
pub inline fn isWindowClosed(wind: *const af.Window) !bool {
    var out: bool = undefined;
    try af.AF_CHECK(af.af_is_window_closed(&out, wind.window_), @src());
    return out;
}

/// Hide/Show a window.
pub inline fn setVisibility(wind: *const af.Window, is_visible: bool) !void {
    try af.AF_CHECK(
        af.af_set_visibility(
            wind.window_,
            is_visible,
        ),
        @src(),
    );
}

/// Destroys a window handle.
pub inline fn destroyWindow(wind: *af.Window) !void {
    try af.AF_CHECK(af.af_destroy_window(wind.window_), @src());
}
