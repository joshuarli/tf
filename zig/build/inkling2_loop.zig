//! Inkling 2 Mini (`inkling2_loop`) in the build: the host package's checks, the native Metal family, and the
//! teacher-forced runner the campaign's parity check drives.
const std = @import("std");

fn module(b: *std.Build, target: std.Build.ResolvedTarget, root: []const u8) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = .Debug, .link_libc = true });
}

pub fn tests(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const core = "zig/src/core/";
    const tokenizer = module(b, target, core ++ "tokenizer/tokenizer.zig");
    const template = module(b, target, core ++ "template/template.zig");
    const small = module(b, target, "zig/src/families/inkling2/config.zig");
    const small_prompt = module(b, target, "zig/src/families/inkling2/prompt.zig");
    small_prompt.addImport("tokenizer", tokenizer);
    small_prompt.addImport("template", template);
    const readers = module(b, target, core ++ "root.zig");
    readers.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    readers.addImport("tokenizer", tokenizer);
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/inkling2_loop/host_test.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "tokenizer", .module = tokenizer },
            .{ .name = "template", .module = template },
            .{ .name = "inkling2_config", .module = small },
            .{ .name = "inkling2_prompt", .module = small_prompt },
            .{ .name = "core", .module = readers },
        },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    run.has_side_effects = true; // the tokenizer test reads the checkpoint outside the build's inputs
    const step = b.step("test-inkling2-loop-host", "Inkling 2 Mini family package: config, quant geometry, weight map, chat template, tokenizer and refusals");
    step.dependOn(&run.step);
    test_step.dependOn(&run.step);
}

/// The family's own module over the Metal runtime, the core readers and the shared Inkling 2 config.
pub fn engineModule(b: *std.Build, target: std.Build.ResolvedTarget, metal: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/inkling2_loop/native.zig"),
        .target = target,
        .optimize = .fast,
        .link_libc = true,
    });
    m.addImport("metal", metal);
    m.addImport("core", core);
    m.addImport("inkling2_config", b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/config.zig"), .target = target, .optimize = .fast }));
    return m;
}

/// The family's GPU-side tests, which open the real checkpoint only when `TF_INKLING2_LOOP_DIR` names it.
pub fn metalTests(b: *std.Build, target: std.Build.ResolvedTarget, metal: *std.Build.Module, core: *std.Build.Module, test_step: *std.Build.Step) void {
    const m = engineModule(b, target, metal, core);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    run.has_side_effects = true;
    const step = b.step("test-inkling2-loop", "Inkling 2 Mini's native Metal forward against the checkpoint and its own state");
    step.dependOn(&run.step);
    test_step.dependOn(&run.step);
}

/// `tf-inkling2-loop-run <model> <tokens.json> logits <out>`: every position's float32 logits, teacher-forced.
pub fn runner(b: *std.Build, target: std.Build.ResolvedTarget, metal: *std.Build.Module, core: *std.Build.Module) void {
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/inkling2_loop/run.zig"),
        .target = target,
        .optimize = .fast,
        .link_libc = true,
    });
    m.addImport("metal", metal);
    m.addImport("core", core);
    m.addImport("engine_api", b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = .fast, .link_libc = true }));
    m.addImport("inkling2_config", b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/config.zig"), .target = target, .optimize = .fast }));
    const exe = b.addExecutable(.{ .name = "tf-inkling2-loop-run", .root_module = m });
    const install = b.addInstallArtifact(exe, .{});
    b.step("tf-inkling2-loop-run", "Inkling 2 Mini teacher-forced logits on Metal, for the campaign's parity check").dependOn(&install.step);
}

test {
    _ = std;
}
