const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) !void {
    const use_cuda = b.option(bool, "ZT_BACKEND_CUDA", "Use CUDA backend") orelse false;
    const af_use_opencl = b.option(bool, "ZT_ARRAYFIRE_USE_OPENCL", "Use ArrayFire with OpenCL backend") orelse false;

    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const opts = .{ .target = target, .optimize = optimize };
    const zigrc_module = b.dependency("zigrc", opts).module("zigrc");

    var shared_opts = b.addOptions();
    shared_opts.addOption(bool, "ZT_BACKEND_CUDA", use_cuda);
    shared_opts.addOption(bool, "ZT_ARRAYFIRE_USE_OPENCL", af_use_opencl);

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
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = main_module.source_file,
        .target = target,
        .optimize = optimize,
    });
    lib.linkLibC();
    linkBackend(lib);
    b.installArtifact(lib);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const main_tests = b.addTest(.{
        .root_source_file = main_module.source_file,
        .target = target,
        .optimize = optimize,
    });
    // TODO: these should o
    main_tests.linkLibC();
    linkBackend(main_tests);

    main_tests.addOptions("build_options", shared_opts);
    main_tests.addModule("zigrc", zigrc_module);

    const run_main_tests = b.addRunArtifact(main_tests);

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build test`
    // This will evaluate the `test` step rather than the default, which is "install".
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
        compile.addIncludePath(.{ .path = "/opt/homebrew/include" });
        compile.addLibraryPath(.{ .path = "/opt/homebrew/lib" });
        compile.linkSystemLibrary("afcpu");
    }
}
