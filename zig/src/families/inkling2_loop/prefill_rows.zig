const std = @import("std");
const mtl = @import("metal");

pub const source = @embedFile("kernels/prefill_rows.metal");

pub const threads = 256;
const per = 9;
const max_depth = 5;
const max_queries = 4;
const max_norms = 3;

pub const Pipes = struct {
    mix: mtl.Pipeline,
    batch_mix: mtl.Pipeline,
    batch_mix2: mtl.Pipeline,
    clips: mtl.Pipeline,
    ring: mtl.Pipeline,
    kv: mtl.Pipeline,
    conv_add: mtl.Pipeline,

    pub fn init(device: mtl.Device) !Pipes {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const mix = try mtl.Pipeline.init(device, lib, "i2pr_mix_norm", false);
        errdefer mix.deinit();
        const batch_mix = try mtl.Pipeline.init(device, lib, "i2pr_batch_mix", false);
        errdefer batch_mix.deinit();
        const batch_mix2 = try mtl.Pipeline.init(device, lib, "i2pr_batch_mix2", false);
        errdefer batch_mix2.deinit();
        const clips = try mtl.Pipeline.init(device, lib, "i2pr_clips", false);
        errdefer clips.deinit();
        const ring = try mtl.Pipeline.init(device, lib, "i2pr_ring_store", false);
        errdefer ring.deinit();
        const kv = try mtl.Pipeline.init(device, lib, "i2pr_kv_store", false);
        errdefer kv.deinit();
        return .{ .mix = mix, .batch_mix = batch_mix, .batch_mix2 = batch_mix2, .clips = clips, .ring = ring, .kv = kv, .conv_add = try mtl.Pipeline.init(device, lib, "i2pr_conv_add", false) };
    }

    pub fn deinit(p: Pipes) void {
        p.mix.deinit();
        p.batch_mix.deinit();
        p.batch_mix2.deinit();
        p.clips.deinit();
        p.ring.deinit();
        p.kv.deinit();
        p.conv_add.deinit();
    }
};

pub const RingArgs = extern struct { channels: u32, ring: u32, base: u32, rows: u32, tail: u32 };
pub const KvArgs = extern struct { count: u32, base: u32, rows: u32, wring: u32 };
pub const ConvArgs = extern struct { channels: u32, heads: u32, ring: u32, base: u32, rows: u32, layout: u32, norm: f32, out: u32 };

pub const ClipArgs = extern struct { heads: u32, kv: u32, items: u32, q_threshold: f32, rel_threshold: f32, eps: f32 };

pub const clip_dim = 128;
pub const clip_rel = 16;
pub const clip_items = 8;

pub const MixArgs = extern struct { width: u32, depth: u32, stride: u32, k: u32, norm: u32, scale: f32, eps: f32 };

pub fn mixFits(width: usize, depth: usize, queries: usize, norms: usize) bool {
    if (width != threads * per or depth == 0 or depth > max_depth) return false;
    return queries >= 1 and queries <= max_queries and (norms == 0 or (norms == queries and norms <= max_norms));
}

test "the fused mix takes the served width and refuses what its registers cannot hold" {
    try std.testing.expect(mixFits(2304, 5, 3, 3));
    try std.testing.expect(mixFits(2304, 1, 1, 0));
    try std.testing.expect(!mixFits(2304, 6, 1, 1));
    try std.testing.expect(!mixFits(2048, 2, 1, 1));
    try std.testing.expect(mixFits(2304, 2, 4, 0));
    try std.testing.expect(!mixFits(2304, 2, 4, 4));
    try std.testing.expect(!mixFits(2304, 2, 5, 0));
    try std.testing.expect(!mixFits(2304, 2, 3, 1));
}
