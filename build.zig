const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) !void {
    // capture build cli flags
    const ZT_BACKEND_CUDA = b.option(bool, "ZT_BACKEND_CUDA", "Use CUDA backend") orelse false;
    const ZT_BACKEND_CPU = b.option(bool, "ZT_BACKEND_CPU", "Use CPU backend") orelse false;
    const ZT_BACKEND_OPENCL = b.option(bool, "ZT_BACKEND_OPENCL", "Use OpenCL backend") orelse false;
    const ZT_ARRAYFIRE_USE_CPU = b.option(bool, "ZT_ARRAYFIRE_USE_CPU", "Use ArrayFire with CPU backend") orelse false;
    const ZT_ARRAYFIRE_USE_OPENCL = b.option(bool, "ZT_ARRAYFIRE_USE_OPENCL", "Use ArrayFire with OpenCL backend") orelse false;
    const ZT_ARRAYFIRE_USE_CUDA = b.option(bool, "ZT_ARRAYFIRE_USE_CUDA", "Use ArrayFire with CUDA backend") orelse false;
    const ZT_USE_ARRAYFIRE = if (ZT_ARRAYFIRE_USE_CUDA or ZT_ARRAYFIRE_USE_OPENCL or ZT_ARRAYFIRE_USE_CPU) true else false;
    const ZT_USE_ONEDNN = b.option(bool, "ZT_USE_ONEDNN", "Use ArrayFire with CUDA backend") orelse false;

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

    var dependencies = std.ArrayList(std.Build.ModuleDependency).init(b.allocator);
    defer dependencies.deinit();
    try dependencies.append(.{ .name = "build_options", .module = shared_opts.createModule() });
    try dependencies.append(.{ .name = "zigrc", .module = zigrc_module });

    // TODO: add optional deps based on build flags (e.g. link to backend (ArrayFire))

    const main_module = b.addModule("zigTensor", .{
        .source_file = .{ .path = "src/zt.zig" },
        .dependencies = dependencies.items,
    });

    const lib = b.addStaticLibrary(.{
        .name = "zigTensor",
        .root_source_file = main_module.source_file,
        .target = target,
        .optimize = optimize,
    });
    lib.linkLibC();
    linkBackend(lib);
    b.installArtifact(lib);

    // Unit Tests
    const main_tests = b.addTest(.{
        .root_source_file = main_module.source_file,
        .target = target,
        .optimize = optimize,
    });
    main_tests.linkLibC();
    linkBackend(main_tests);

    main_tests.addOptions("build_options", shared_opts);
    main_tests.addModule("zigrc", zigrc_module);

    const run_main_tests = b.addRunArtifact(main_tests);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_main_tests.step);

    // Docs
    const zigTensor_docs = main_tests;
    const build_docs = b.addInstallDirectory(.{
        .source_dir = zigTensor_docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "../docs",
    });
    const build_docs_step = b.step("docs", "Build the zigTensor library docs");
    build_docs_step.dependOn(&build_docs.step);
}

// TODO: add flexibility, enable linking various backends
fn linkBackend(compile: *std.Build.Step.Compile) void {
    const target = (std.zig.system.NativeTargetInfo.detect(compile.target) catch @panic("failed to detect native target info!")).target;
    if (target.os.tag == .linux) {
        // TODO: support linux
    } else if (target.os.tag == .windows) {
        // TODO: support windows
    } else if (target.isDarwin()) {
        if (target.cpu.arch == .aarch64) {
            compile.addIncludePath(.{ .path = "/opt/homebrew/include" });
            compile.addLibraryPath(.{ .path = "/opt/homebrew/lib" });
        } else {
            compile.addIncludePath(.{ .path = "/usr/local/include" });
            compile.addLibraryPath(.{ .path = "/usr/local/lib" });
        }
        compile.linkSystemLibrary("afcpu");
    }
}
