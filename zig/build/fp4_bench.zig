//! The fp4 decode experiment's steps: `test-fp4-kernels` (variants against the fp64 reference with
//! a negative control) and `tf-fp4-bench` (the timing binary, built for campaign/heavy.py).
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const optimize: std.builtin.OptimizeMode = .fast;
    const metal = b.createModule(.{
        .root_source_file = b.path("zig/src/metal/metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    metal.linkFramework("Metal", .{});
    metal.linkFramework("Foundation", .{});
    metal.linkFramework("IOSurface", .{});
    metal.linkSystemLibrary("objc", .{});

    const here = "zig/src/core/metal_fp4/";
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path(here ++ "fp4_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "metal", .module = metal }},
    }) });
    const run_tests = b.addRunArtifact(tests);
    run_tests.has_side_effects = true;
    const check = b.step("test-fp4-kernels", "fp4 decode variants against the fp64 dequant on Inkling decode shapes, with a corrupted-scale negative control");
    check.dependOn(&run_tests.step);
    test_step.dependOn(&run_tests.step);

    const split_exe = b.addExecutable(.{ .name = "tf-split-bf16-bench", .root_module = b.createModule(.{
        .root_source_file = b.path(here ++ "split_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "metal", .module = metal }},
    }) });
    const split_install = b.addInstallArtifact(split_exe, .{});
    const split_bench = b.step("tf-split-bf16-bench", "Build the split-bf16 TensorOps experiment");
    split_bench.dependOn(&split_install.step);

    const exe = b.addExecutable(.{ .name = "tf-fp4-bench", .root_module = b.createModule(.{
        .root_source_file = b.path(here ++ "fp4_bench_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "metal", .module = metal }},
    }) });
    const install = b.addInstallArtifact(exe, .{});
    const bench = b.step("tf-fp4-bench", "Build tf-fp4-bench (run it through campaign/heavy.py)");
    bench.dependOn(&install.step);
}
