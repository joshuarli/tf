//! Inkling 2 Mini's Metal kernel sources, compiled when a model opens.
//!
//! `glue` is the dense graph: one token at a time, f32 activations, the checkpoint's bf16 weights widened per read.
//! `experts` runs the routed projections in the two stored formats, decoding the checkpoint's own bytes. Both mirror
//! the graph of llama.cpp `40b7a546`.
//! `dense` re-runs the bf16 matvecs (`i2_mv`, `i2_mv_cap`) with the activation staged once a threadgroup and
//! vectorized weight loads, plus a fused form for projections sharing one input. `forward.zig` compiles it beside
//! the model library and falls back to `glue` when that compile fails.
const std = @import("std");
const mtl = @import("metal");

pub const glue = @embedFile("kernels/glue.metal");

pub const prefill = @embedFile("kernels/prefill.metal");
pub const prefill_expert_wide = @embedFile("kernels/prefill_expert_wide.metal");
pub const affine_stage = @embedFile("kernels/affine_stage.metal");
pub const float_mpp = @embedFile("kernels/float_mpp.metal");
pub const prefill_smallrow = @embedFile("kernels/prefill_smallrow.metal");
pub const prefill_gemm_mpp = @embedFile("kernels/prefill_gemm_mpp.metal");
pub const verify_nax = @embedFile("kernels/verify_nax.metal");

pub fn prefillSource(comptime mxfp8: bool) []const u8 {
    return if (mxfp8) "#define I2P_MXFP8 1\n" ++ prefill else prefill;
}

/// Split-K decode attention: the window's positions partitioned over threadgroups, then one combine pass.
pub const attention = @embedFile("kernels/attention.metal");

pub const dense = @embedFile("kernels/dense.metal");
pub const affine = @embedFile("kernels/affine.metal");

pub const AffinePipes = struct {
    dense: mtl.Pipeline,
    projections: mtl.Pipeline,
    batch_projections: mtl.Pipeline,
    dense_rows: mtl.Pipeline,
    float: mtl.Pipeline,
    float_control: mtl.Pipeline,
    batch_float_control: mtl.Pipeline,
    embed: mtl.Pipeline,
    up: mtl.Pipeline,
    down: mtl.Pipeline,
    group_up: mtl.Pipeline,
    group_down: mtl.Pipeline,

    pub fn init(device: mtl.Device) !AffinePipes {
        const lib = try mtl.Library.fromSource(device, affine, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const dense_pipe = try mtl.Pipeline.init(device, lib, "i2a_dense", false);
        errdefer dense_pipe.deinit();
        const projections_pipe = try mtl.Pipeline.init(device, lib, "i2a_projections", false);
        errdefer projections_pipe.deinit();
        const batch_projections_pipe = try mtl.Pipeline.init(device, lib, "i2a_batch_projections", false);
        errdefer batch_projections_pipe.deinit();
        const dense_rows_pipe = try mtl.Pipeline.init(device, lib, "i2a_dense_rows", false);
        errdefer dense_rows_pipe.deinit();
        const float_pipe = try mtl.Pipeline.init(device, lib, "i2a_float", false);
        errdefer float_pipe.deinit();
        const float_control_pipe = try mtl.Pipeline.init(device, lib, "i2a_float_control", false);
        errdefer float_control_pipe.deinit();
        const batch_float_control_pipe = try mtl.Pipeline.init(device, lib, "i2a_batch_float_control", false);
        errdefer batch_float_control_pipe.deinit();
        const embed_pipe = try mtl.Pipeline.init(device, lib, "i2a_embed", false);
        errdefer embed_pipe.deinit();
        const up_pipe = try mtl.Pipeline.init(device, lib, "i2a_expert_up", false);
        errdefer up_pipe.deinit();
        const down_pipe = try mtl.Pipeline.init(device, lib, "i2a_expert_down", false);
        errdefer down_pipe.deinit();
        const group_up_pipe = try mtl.Pipeline.init(device, lib, "i2a_group_up", false);
        errdefer group_up_pipe.deinit();
        const group_down_pipe = try mtl.Pipeline.init(device, lib, "i2a_group_down", false);
        return .{ .group_up = group_up_pipe, .group_down = group_down_pipe, .dense = dense_pipe, .projections = projections_pipe, .batch_projections = batch_projections_pipe, .dense_rows = dense_rows_pipe, .float = float_pipe, .float_control = float_control_pipe, .batch_float_control = batch_float_control_pipe, .embed = embed_pipe, .up = up_pipe, .down = down_pipe };
    }

    pub fn deinit(p: AffinePipes) void {
        p.dense.deinit();
        p.projections.deinit();
        p.batch_projections.deinit();
        p.dense_rows.deinit();
        p.float.deinit();
        p.float_control.deinit();
        p.batch_float_control.deinit();
        p.embed.deinit();
        p.up.deinit();
        p.down.deinit();
        p.group_up.deinit();
        p.group_down.deinit();
    }
};

pub const dense_names = [_][:0]const u8{ "i2_dmv", "i2_dmv_cap", "i2_dmv_fused", "i2_dmv_fusedx" };

pub const Dense = struct { mv: mtl.Pipeline, mv_cap: mtl.Pipeline, fused: mtl.Pipeline, fusedx: mtl.Pipeline };

var dense_owner: ?mtl.Device = null;
var dense_pipes: ?Dense = null;

pub fn densePipes(device: mtl.Device) !Dense {
    if (dense_pipes) |p| if (dense_owner.?.id == device.id) return p;
    const lib = try mtl.Library.fromSource(device, dense, mtl.CompileOptions.mlx());
    const pipes = Dense{
        .mv = try mtl.Pipeline.init(device, lib, dense_names[0], false),
        .mv_cap = try mtl.Pipeline.init(device, lib, dense_names[1], false),
        .fused = try mtl.Pipeline.init(device, lib, dense_names[2], false),
        .fusedx = try mtl.Pipeline.init(device, lib, dense_names[3], false),
    };
    dense_owner = device;
    dense_pipes = pipes;
    return pipes;
}

/// The routed experts' source for one stored format: MXFP8 defines its decoder, NVFP4 is the default.
pub fn expertSource(comptime mxfp8: bool) []const u8 {
    const text = @embedFile("kernels/experts.metal");
    return if (mxfp8) "#define I2_MXFP8 1\n" ++ text else text;
}

test "the sources name the kernels the model loads" {
    try std.testing.expect(std.mem.indexOf(u8, glue, "i2_attn") != null);
    try std.testing.expect(std.mem.indexOf(u8, attention, "i2_attn_split") != null);
    try std.testing.expect(std.mem.indexOf(u8, attention, "i2_attn_combine") != null);
    try std.testing.expect(std.mem.indexOf(u8, glue, "i2_conv_store") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(false), "I2_MXFP8 1") == null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(true), "I2_MXFP8 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(false), "i2_gu") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(false), "i2_down") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(false), "i2_gu_table") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(false), "i2_down_table") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(true), "i2_gu_table") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(true), "i2_down_thread") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_dense") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_attn") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_gu") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_down") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_cap_exact") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_gemm") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_moe_sort") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_moe_offsets") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_moe_count") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_moe_zero") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_down_grouped") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefill, "i2p_gu_grouped") != null);
    try std.testing.expect(std.mem.indexOf(u8, prefillSource(false), "I2P_MXFP8 1") == null);
    try std.testing.expect(std.mem.indexOf(u8, prefillSource(true), "I2P_MXFP8 1") != null);
}
