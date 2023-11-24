const std = @import("std");
const af = @import("arrayfire.zig");

pub const Window = struct {
    allocator: std.mem.Allocator,
    window_: af.af_window,

    pub fn initFromWindow(allocator: std.mem.Allocator, window: af.af_window) !*Window {
        const self = try allocator.create(Window);
        self.* = .{ .allocator = allocator, .window_ = window };
        return self;
    }

    pub fn init(allocator: std.mem.Allocator, width: i32, height: i32, title: []const u8) !*Window {
        return af.ops.createWindow(allocator, width, height, title);
    }

    pub fn deinit(self: *Window) void {
        self.destroy() catch unreachable;
        self.allocator.destroy(self);
    }

    pub fn setPosition(self: *Window, x: u32, y: u32) !void {
        try af.ops.setPosition(self, x, y);
    }

    pub fn setTitle(self: *Window, title: []const u8) !void {
        try af.ops.setTitle(self, title);
    }

    pub fn setSize(self: *Window, w: u32, h: u32) !void {
        try af.ops.setSize(self, w, h);
    }

    pub fn drawImage(self: *Window, in: *const af.Array, props: af.Cell) !void {
        try af.ops.drawImage(self, in, &props);
    }

    pub fn drawPlot(self: *Window, X: *const af.Array, Y: *const af.Array, props: af.Cell) !void {
        try af.ops.drawPlot(self, X, Y, &props);
    }

    pub fn drawPlot3(self: *Window, P: *const af.Array, props: af.Cell) !void {
        try af.ops.drawPlot3(self, P, &props);
    }

    pub fn drawPlotNd(self: *Window, P: *const af.Array, props: af.Cell) !void {
        try af.ops.drawPlotNd(self, P, &props);
    }

    pub fn drawPlot2d(self: *Window, X: *const af.Array, Y: *const af.Array, props: af.Cell) !void {
        try af.ops.drawPlot2d(self, X, Y, &props);
    }

    pub fn drawPlot3d(self: *Window, P: *const af.Array, props: af.Cell) !void {
        try af.ops.drawPlot3d(self, P, &props);
    }

    pub fn drawScatter(self: *Window, X: *const af.Array, Y: *const af.Array, marker: af.MarkerType, props: af.Cell) !void {
        try af.ops.drawScatter(self, X, Y, marker, &props);
    }

    pub fn drawScatter3(self: *Window, P: *const af.Array, marker: af.MarkerType, props: af.Cell) !void {
        try af.ops.drawScatter3(self, P, marker, &props);
    }

    pub fn drawScatterNd(self: *Window, P: *const af.Array, marker: af.MarkerType, props: af.Cell) !void {
        try af.ops.drawScatterNd(self, P, marker, &props);
    }

    pub fn drawScatter2d(self: *Window, X: *const af.Array, Y: *const af.Array, marker: af.MarkerType, props: af.Cell) !void {
        try af.ops.drawScatter2d(self, X, Y, marker, &props);
    }

    pub fn drawScatter3d(self: *Window, X: *const af.Array, Y: *const af.Array, Z: *const af.Array, marker: af.MarkerType, props: af.Cell) !void {
        try af.ops.drawScatter3d(self, X, Y, Z, marker, &props);
    }

    pub fn drawHist(self: *Window, X: *const af.Array, minval: f64, maxval: f64, props: af.Cell) !void {
        try af.ops.drawHist(self, X, minval, maxval, &props);
    }

    pub fn drawVectorFieldNd(self: *Window, points: *const af.Array, directions: *const af.Array, props: af.Cell) !void {
        try af.ops.drawVectorFieldNd(self, points, directions, &props);
    }

    pub fn drawVectorField3d(self: *Window, xPoints: *const af.Array, yPoints: *const af.Array, zPoints: *const af.Array, xDirs: *const af.Aray, yDirs: *const af.Array, zDirs: *const af.Array, props: af.Cell) !void {
        try af.ops.drawVectorField3d(self, xPoints, yPoints, zPoints, xDirs, yDirs, zDirs, &props);
    }

    pub fn drawVectorField2d(self: *Window, xPoints: *const af.Array, yPoints: *const af.Array, xDirs: *const af.Array, yDirs: *const af.Array, props: af.Cell) !void {
        try af.ops.drawVectorField2d(self, xPoints, yPoints, xDirs, yDirs, &props);
    }

    pub fn grid(self: *Window, rows: i32, cols: i32) !void {
        try af.ops.grid(self, rows, cols);
    }

    pub fn setAxesLimitsCompute(self: *Window, x: *const af.Array, y: *const af.Array, z: *const af.Array, exact: bool, props: af.Cell) !void {
        try af.ops.setAxesLimitsCompute(self, x, y, z, exact, &props);
    }

    pub fn setAxesLimits2d(self: *Window, xmin: f32, xmax: f32, ymin: f32, ymax: f32, exact: bool, props: af.Cell) !void {
        try af.ops.setAxesLimits2d(self, xmin, xmax, ymin, ymax, exact, &props);
    }

    pub fn setAxesLimits3d(self: *Window, xmin: f32, xmax: f32, ymin: f32, ymax: f32, zmin: f32, zmax: f32, exact: bool, props: af.Cell) !void {
        try af.ops.setAxesLimits3d(self, xmin, xmax, ymin, ymax, zmin, zmax, exact, &props);
    }

    pub fn setAxesTitles(self: *Window, xtitle: []const u8, ytitle: []const u8, ztitle: []const u8, props: af.Cell) !void {
        try af.ops.setAxesTitles(self, xtitle, ytitle, ztitle, &props);
    }

    pub fn setAxesLabelFormat(self: *Window, xformat: []const u8, yformat: []const u8, zformat: []const u8, props: af.Cell) !void {
        try af.ops.setAxesLabelFormat(self, xformat, yformat, zformat, &props);
    }

    pub fn show(self: *Window) !void {
        try af.ops.show(self);
    }

    pub fn isClosed(self: *Window) !bool {
        return af.ops.isWindowClosed(self);
    }

    pub fn setVisibility(self: *Window, is_visible: bool) !void {
        try af.ops.setVisibility(self, is_visible);
    }

    pub fn destroy(self: *Window) !void {
        try af.ops.destroyWindow(self);
    }
};
