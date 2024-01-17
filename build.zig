const std = @import("std");

// None indicates this backend is not in use
const BackendType = enum { None, Cpu, Cuda, OpenCL };

const BackendOptions = struct {
    library_path: []u8 = &[_]u8{},
    include_path: []u8 = &[_]u8{},
    backend_type: BackendType = .None,
};

const LinkerOpts = struct {
    arrayfire: BackendOptions = .{},
    onednn: BackendOptions = .{},
};

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) !void {
    // capture build cli flags
    var backend_opts: LinkerOpts = .{};
    // library/include paths
    const ZT_ARRAYFIRE_INCLUDE_PATH = b.option([]const u8, "ZT_ARRAYFIRE_INCLUDE_PATH", "Arrayfire include path pointing to headers") orelse &.{};
    const ZT_ARRAYFIRE_LIBRARY_PATH = b.option([]const u8, "ZT_ARRAYFIRE_LIBRARY_PATH", "Arrayfire libary path") orelse &.{};
    const ZT_ONEDNN_INCLUDE_PATH = b.option([]const u8, "ZT_ONEDNN_INCLUDE_PATH", "OneDNN include path pointing to headers") orelse &.{};
    const ZT_ONEDNN_LIBRARY_PATH = b.option([]const u8, "ZT_ONEDNN_LIBRARY_PATH", "OneDNN libary path") orelse &.{};
    // backend option flags
    const ZT_ARRAYFIRE_USE_CPU = b.option(bool, "ZT_ARRAYFIRE_USE_CPU", "Use ArrayFire with CPU backend") orelse false;
    const ZT_ARRAYFIRE_USE_OPENCL = b.option(bool, "ZT_ARRAYFIRE_USE_OPENCL", "Use ArrayFire with OpenCL backend") orelse false;
    const ZT_ARRAYFIRE_USE_CUDA = b.option(bool, "ZT_ARRAYFIRE_USE_CUDA", "Use ArrayFire with CUDA backend") orelse false;
    const ZT_USE_ARRAYFIRE = if (ZT_ARRAYFIRE_USE_CUDA or ZT_ARRAYFIRE_USE_OPENCL or ZT_ARRAYFIRE_USE_CPU) true else false;
    const ZT_USE_ONEDNN = b.option(bool, "ZT_USE_ONEDNN", "Use OneDNN backend") orelse false;
    const ZT_USE_CUDNN = b.option(bool, "ZT_USE_CUDNN", "Use cuDNN backend") orelse false;
    // zigTensor backend options (explicit or inferred from backend option flags)
    const ZT_BACKEND_CUDA = b.option(bool, "ZT_BACKEND_CUDA", "Use CUDA backend") orelse ZT_ARRAYFIRE_USE_CUDA;
    const ZT_BACKEND_CPU = b.option(bool, "ZT_BACKEND_CPU", "Use CPU backend") orelse ZT_ARRAYFIRE_USE_CPU;
    const ZT_BACKEND_OPENCL = b.option(bool, "ZT_BACKEND_OPENCL", "Use OpenCL backend") orelse ZT_ARRAYFIRE_USE_OPENCL;
    if (ZT_USE_ARRAYFIRE) {
        backend_opts.arrayfire = .{
            .library_path = @constCast(ZT_ARRAYFIRE_LIBRARY_PATH),
            .include_path = @constCast(ZT_ARRAYFIRE_INCLUDE_PATH),
            .backend_type = if (ZT_ARRAYFIRE_USE_CUDA) .Cuda else if (ZT_ARRAYFIRE_USE_OPENCL) .OpenCL else .Cpu,
        };
    }
    if (ZT_USE_ONEDNN) {
        backend_opts.onednn = .{
            .library_path = @constCast(ZT_ONEDNN_LIBRARY_PATH),
            .include_path = @constCast(ZT_ONEDNN_INCLUDE_PATH),
            .backend_type = .Cpu,
        };
    }

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const opts = .{ .target = target, .optimize = optimize };
    const zigrc_module = b.dependency("zigrc", opts).module("zigrc");

    var shared_opts = b.addOptions();
    shared_opts.addOption(bool, "ZT_BACKEND_CUDA", ZT_BACKEND_CUDA);
    shared_opts.addOption(bool, "ZT_BACKEND_CPU", ZT_BACKEND_CPU);
    shared_opts.addOption(bool, "ZT_BACKEND_OPENCL", ZT_BACKEND_OPENCL);
    shared_opts.addOption(bool, "ZT_ARRAYFIRE_USE_CPU", ZT_ARRAYFIRE_USE_CPU);
    shared_opts.addOption(bool, "ZT_ARRAYFIRE_USE_OPENCL", ZT_ARRAYFIRE_USE_OPENCL);
    shared_opts.addOption(bool, "ZT_ARRAYFIRE_USE_CUDA", ZT_ARRAYFIRE_USE_CUDA);
    shared_opts.addOption(bool, "ZT_USE_ARRAYFIRE", ZT_USE_ARRAYFIRE);
    shared_opts.addOption(bool, "ZT_USE_ONEDNN", ZT_USE_ONEDNN);
    shared_opts.addOption(bool, "ZT_USE_CUDNN", ZT_USE_CUDNN);

    // TODO: add optional deps based on build flags (e.g. link to backend (ArrayFire))

    const main_module = b.addModule("zigTensor", .{ .root_source_file = .{ .path = "src/zt.zig" }, .imports = &.{ .{ .name = "build_options", .module = shared_opts.createModule() }, .{ .name = "zigrc", .module = zigrc_module } } });

    const lib = b.addStaticLibrary(.{
        .name = "zigTensor",
        .root_source_file = main_module.root_source_file,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    try linkBackends(b.allocator, lib, &backend_opts);
    b.installArtifact(lib);

    // Unit Tests
    const main_tests = b.addTest(.{
        .root_source_file = main_module.root_source_file.?,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    try linkBackends(b.allocator, main_tests, &backend_opts);

    main_tests.root_module.addOptions("build_options", shared_opts);
    main_tests.root_module.addImport("zigrc", zigrc_module);

    const run_main_tests = b.addRunArtifact(main_tests);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_main_tests.step);

    // Docs
    const zigTensor_docs = b.addStaticLibrary(.{
        .name = "zigTensor",
        .root_source_file = std.Build.LazyPath.relative("src/zt.zig"),
        .target = target,
        .optimize = optimize,
    });
    zigTensor_docs.root_module.addOptions("build_options", shared_opts);
    zigTensor_docs.root_module.addImport("zigrc", zigrc_module);
    try linkBackends(b.allocator, zigTensor_docs, &backend_opts);

    const build_docs = b.addInstallDirectory(.{
        .source_dir = zigTensor_docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "../docs",
    });
    const build_docs_step = b.step("docs", "Build the zigTensor library docs");
    build_docs_step.dependOn(&build_docs.step);
}

const BindingMaps = struct {
    include: std.StringHashMap(void),
    library: std.StringHashMap(void),
    rpath: std.StringHashMap(void),

    pub fn init(allocator: std.mem.Allocator) BindingMaps {
        return .{
            .include = std.StringHashMap(void).init(allocator),
            .library = std.StringHashMap(void).init(allocator),
            .rpath = std.StringHashMap(void).init(allocator),
        };
    }

    pub fn deinit(self: *BindingMaps) void {
        self.include.deinit();
        self.library.deinit();
        self.rpath.deinit();
    }
};

fn addDefaultPaths(target: std.Target, opts: *BackendOptions) void {
    if (target.isDarwin()) {
        if (opts.include_path.len == 0) {
            // default to homebrew installations on OSx
            if (target.cpu.arch == .aarch64) {
                opts.include_path = @constCast("/opt/homebrew/include");
            } else {
                opts.include_path = @constCast("/usr/local/include");
            }
        }
        if (opts.library_path.len == 0) {
            // default to homebrew installations on OSx
            if (target.cpu.arch == .aarch64) {
                opts.library_path = @constCast("/opt/homebrew/lib");
            } else {
                opts.library_path = @constCast("/usr/local/lib");
            }
        }
    }
}

fn linkBackends(allocator: std.mem.Allocator, compile: *std.Build.Step.Compile, opts: *LinkerOpts) !void {
    var maps = BindingMaps.init(allocator);
    defer maps.deinit();
    const target = compile.rootModuleTarget();
    if (opts.arrayfire.backend_type == .OpenCL or opts.onednn.backend_type == .OpenCL) {
        compile.addIncludePath(std.Build.LazyPath.relative("headers"));
    }
    if (opts.arrayfire.backend_type != .None) {
        const lib_name: []const u8 = if (opts.arrayfire.backend_type == .Cuda) "afcuda" else if (opts.arrayfire.backend_type == .OpenCL) "afopencl" else "afcpu";
        addDefaultPaths(target, &opts.arrayfire);
        try linkBackend(compile, opts.arrayfire, lib_name, &maps);
    }
    if (opts.onednn.backend_type != .None) {
        addDefaultPaths(target, &opts.onednn);
        try linkBackend(compile, opts.onednn, "dnnl", &maps);
    }
}

fn linkBackend(compile: *std.Build.Step.Compile, opts: BackendOptions, lib_name: []const u8, maps: *BindingMaps) !void {
    if (opts.include_path.len == 0) {
        std.debug.print("include path must be specified (OSx defaults to homebrew installation)", .{});
        return error.BackendMissingIncludePaths;
    }
    if (opts.library_path.len == 0) {
        std.debug.print("library path must be specified (OSx defaults to homebrew installation)", .{});
        return error.BackendMissingLibraryPaths;
    }
    if (!maps.include.contains(opts.include_path)) {
        compile.addIncludePath(.{ .path = opts.include_path });
        try maps.include.put(opts.include_path, {});
    }
    if (!maps.library.contains(opts.library_path)) {
        compile.addLibraryPath(.{ .path = opts.library_path });
        try maps.library.put(opts.library_path, {});
    }
    if (!maps.rpath.contains(opts.library_path)) {
        compile.addRPath(.{ .path = opts.library_path });
        try maps.rpath.put(opts.library_path, {});
    }
    compile.linkSystemLibrary(lib_name);
}
