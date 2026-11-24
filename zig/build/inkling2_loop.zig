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
    const prefix_tests = module(b, target, "zig/src/native/inkling2_loop_prefix.zig");
    const prefix_run = b.addRunArtifact(b.addTest(.{ .root_module = prefix_tests }));
    step.dependOn(&prefix_run.step);
    test_step.dependOn(&run.step);
    test_step.dependOn(&prefix_run.step);
    if (target.result.os.tag == .macos) {
        const metal = b.createModule(.{
            .root_source_file = b.path("zig/src/metal/metal.zig"),
            .target = target,
            .optimize = .Debug,
            .link_libc = true,
        });
        metal.linkFramework("Metal", .{});
        metal.linkFramework("Foundation", .{});
        metal.linkFramework("IOSurface", .{});
        metal.linkSystemLibrary("objc", .{});
        const family = engineModule(b, target, metal, readers);
        const facade_file = b.addWriteFiles().add("inkling_host_facade.zig", "pub const inkling2_loop = @import(\"inkling_family\");\n");
        const facade = b.createModule(.{ .root_source_file = facade_file, .target = target, .optimize = .Debug, .link_libc = true });
        facade.addImport("inkling_family", family);
        const lanes = module(b, target, "zig/src/core/lanes/lanes.zig");
        const api = module(b, target, "zig/src/core/engine_api.zig");
        api.addImport("lanes", lanes);
        const host = module(b, target, "zig/src/native/inkling2_loop_host.zig");
        host.addImport("metal", metal);
        host.addImport("tensorfold", facade);
        host.addImport("engine_api", api);
        const host_run = b.addRunArtifact(b.addTest(.{ .root_module = host, .filters = &.{"inkling scheduler:"} }));
        step.dependOn(&host_run.step);
        test_step.dependOn(&host_run.step);
        metalTests(b, target, metal, readers, test_step);
    }
}

/// The family's own module over the Metal runtime, the core readers and the shared Inkling 2 config.
pub fn engineModule(b: *std.Build, target: std.Build.ResolvedTarget, metal: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/inkling2_loop/family.zig"),
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
    const check = b.createModule(.{
        .root_source_file = b.path("zig/src/native/inkling2_loop_prefix_check.zig"),
        .target = target,
        .optimize = .fast,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "inkling", .module = m } },
    });
    const exe = b.addExecutable(.{ .name = "tf-inkling2-prefix-check", .root_module = check });
    b.step("tf-inkling2-prefix-check", "Compare retained state to identical replay and full prefill").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const family_test = b.addTest(.{ .root_module = m });
    const run = b.addRunArtifact(family_test);
    run.has_side_effects = true;
    const compile = b.step("compile-inkling2-loop", "Compile the Inkling 2 Mini Metal test binaries without running them on the GPU");
    compile.dependOn(&family_test.step);
    const step = b.step("test-inkling2-loop", "Inkling 2 Mini's native Metal forward against the checkpoint and its own state");
    step.dependOn(&run.step);
    test_step.dependOn(&run.step);
    const d = b.createModule(.{
        .root_source_file = b.path("zig/src/families/inkling2_loop/dense_test.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    });
    d.addImport("metal", metal);
    d.addImport("core", core);
    d.addImport("inkling2_config", b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/config.zig"), .target = target, .optimize = .Debug }));
    const dense_test = b.addTest(.{ .root_module = d });
    compile.dependOn(&dense_test.step);
    const dense_run = b.addRunArtifact(dense_test);
    dense_run.has_side_effects = true;
    step.dependOn(&dense_run.step);
    test_step.dependOn(&dense_run.step);
    const affine = module(b, target, "zig/src/families/inkling2_loop/affine_test.zig");
    affine.addImport("metal", metal);
    affine.addImport("core", core);
    affine.addImport("inkling2_config", b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/config.zig"), .target = target, .optimize = .Debug }));
    const affine_test = b.addTest(.{ .root_module = affine });
    compile.dependOn(&affine_test.step);
    const affine_run = b.addRunArtifact(affine_test);
    affine_run.has_side_effects = true;
    b.step("test-inkling2-loop-affine", "Compare affine dense and expert kernels with float64").dependOn(&affine_run.step);
    step.dependOn(&affine_run.step);
    test_step.dependOn(&affine_run.step);
    const dflash_test = b.addTest(.{ .root_module = dflashModule(b, target, "zig/src/families/inkling2_loop/dflash_test.zig") });
    compile.dependOn(&dflash_test.step);
    const dflash_run = b.addRunArtifact(dflash_test);
    dflash_run.has_side_effects = true;
    b.step("test-inkling2-loop-dflash2", "Inkling 2 Mini's DFlash2 drafter: config refusals and greedy speculation against plain decoding").dependOn(&dflash_run.step);
    step.dependOn(&dflash_run.step);
    test_step.dependOn(&dflash_run.step);
}

/// The DFlash2 drafter reads the shared `qwen27/dflash` runtime, so its tests import the engine, not the family.
fn dflashModule(b: *std.Build, target: std.Build.ResolvedTarget, root: []const u8) *std.Build.Module {
    const ids = b.addWriteFiles();
    _ = ids.addCopyFile(b.path("zig/src/families/nemotron/draft_ids.txt"), "draft_ids.txt");
    const draft_ids = b.createModule(.{ .root_source_file = ids.add("draft_ids.zig", "pub const text = @embedFile(\"draft_ids.txt\");\n") });
    const mods = @import("metal_targets.zig").metalEngineModules(b, target, .fast, draft_ids);
    mods.sources.addAnonymousImport("nemotron_packed_metallib", .{ .root_source_file = b.addWriteFiles().add("inkling-dflash2-unused.zig", "pub const bytes = \"\";\n") });
    return b.createModule(.{
        .root_source_file = b.path(root),
        .target = target,
        .optimize = .fast,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = mods.metal }, .{ .name = "tensorfold", .module = mods.engine } },
    });
}

/// `tf-inkling2-loop-run`: every position's float32 logits, teacher-forced, for the campaign's parity check.
pub fn runner(b: *std.Build, target: std.Build.ResolvedTarget, metal: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
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
    return m;
}

test {
    _ = std;
}
