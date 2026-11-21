//! Deterministic synthetic codes, scales and activations in the checkpoint's exact layout: no
//! weight file is read, and every variant sees the same bytes.
const std = @import("std");
const shapes = @import("fp4_shapes.zig");

pub fn seedFor(fmt: shapes.Format, proj: shapes.Projection, pair: usize) u64 {
    return 0x4644_0000_0000_0000 ^ (@as(u64, @intFromEnum(fmt)) << 40) ^ (@as(u64, @intFromEnum(proj)) << 32) ^ pair +% 1;
}

/// Fills one expert's packed codes and scales; the NVFP4 per-tensor global is returned.
pub fn fillPair(rng: std.Random, fmt: shapes.Format, codes: []u8, scales: []u8) f64 {
    switch (fmt) {
        .nvfp4 => {
            rng.bytes(codes);
            for (scales) |*s| {
                const magnitude: u8 = 0x30 + rng.uintLessThan(u8, 24);
                s.* = magnitude | (if (rng.boolean()) @as(u8, 0x80) else 0);
            }
            const amax = 0.1 + 0.25 * rng.float(f32);
            return @as(f64, amax) / 2688.0;
        },
        .mxfp8 => {
            for (codes) |*c| {
                var b: u8 = 0x28 + rng.uintLessThan(u8, 32);
                if (rng.boolean()) b |= 0x80;
                c.* = b;
            }
            for (scales) |*s| s.* = 118 + rng.uintLessThan(u8, 8);
            // MXFP8 has no per-tensor global: the kernel's AMAX/DEN pair must stay 1.
            return 1;
        },
    }
}

pub fn fillData(rng: std.Random, out: []f32) void {
    for (out) |*v| v.* = rng.float(f32) * 2.0 - 1.0;
}

test "synthetic bytes stay inside the decodable ranges" {
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();
    var codes: [4096]u8 = undefined;
    var scales: [256]u8 = undefined;
    const global = fillPair(rng, .mxfp8, &codes, &scales);
    try std.testing.expectEqual(@as(f64, 1), global);
    for (codes) |c| try std.testing.expect(c & 0x7f != 0x7f);
    for (scales) |s| try std.testing.expect(s >= 118 and s < 126);
    const g2 = fillPair(rng, .nvfp4, &codes, &scales);
    try std.testing.expect(g2 > 0.1 / 2688.0 and g2 < 0.35 / 2688.0);
    for (scales) |s| try std.testing.expect(s & 0x7f >= 0x30 and s & 0x7f < 0x48);
}
