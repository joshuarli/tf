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

    pub fn init(device: mtl.Device) !Pipes {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{ .mix = try mtl.Pipeline.init(device, lib, "i2pr_mix_norm", false) };
    }

    pub fn deinit(p: Pipes) void {
        p.mix.deinit();
    }
};

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
