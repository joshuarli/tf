//! Inkling 2 Mini (`inkling2_loop`) host checks (no GPU).
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
