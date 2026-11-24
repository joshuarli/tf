//! bf16 tap rows `[rows][count * hidden]`: the block residual entering listed visits, and the final norm (id 48).
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

pub const max_taps = 8;

const Args = extern struct { width: u32, rows: u32, stride: u32, column: u32 };

pub const Taps = struct {
    buffer: mtl.Buffer,
    copy: mtl.Pipeline,
    rows: usize,
    hidden: u32,
    visits: [max_taps]u32 = @splat(0),
    count: u32,
    final: u32,

    pub fn init(device: mtl.Device, config: *const cfg.Config, visits: []const u32, rows: usize) !Taps {
        if (visits.len == 0 or visits.len > max_taps or rows == 0) return error.BadTaps;
        for (visits, 0..) |v, i| {
            if (v > config.slots()) return error.BadTaps;
            for (visits[0..i]) |prior| if (prior == v) return error.BadTaps;
        }
        const lib = try mtl.Library.fromSource(device, @embedFile("kernels/taps.metal"), mtl.CompileOptions.mlx());
        defer lib.deinit();
        const copy = try mtl.Pipeline.init(device, lib, "i2t_tap", false);
        errdefer copy.deinit();
        const size = rows * visits.len * config.text.hidden * 2;
        var t = Taps{ .buffer = try device.buffer(size, mtl.ResourceOptions.shared), .copy = copy, .rows = rows, .hidden = config.text.hidden, .count = @intCast(visits.len), .final = config.slots() };
        @memcpy(t.visits[0..visits.len], visits);
        return t;
    }

    pub fn deinit(t: *Taps) void {
        t.copy.deinit();
        t.buffer.deinit();
    }

    pub fn rowBytes(t: *const Taps) usize {
        return @as(usize, t.count) * t.hidden * 2;
    }

    pub fn bytes(t: *const Taps, first: usize, n: usize) []const u8 {
        return t.buffer.contents()[first * t.rowBytes() ..][0 .. n * t.rowBytes()];
    }

    pub fn encode(t: *const Taps, enc: mtl.ComputeEncoder, visit: u32, src: mtl.Buffer, offset: usize, rows: u32) void {
        for (t.visits[0..t.count], 0..) |v, i| {
            if (v != visit) continue;
            enc.setPipeline(t.copy);
            enc.setBuffer(src, offset, 0);
            enc.setValue(Args{ .width = t.hidden, .rows = rows, .stride = t.count * t.hidden, .column = @as(u32, @intCast(i)) * t.hidden }, 1);
            enc.setBuffer(t.buffer, 0, 2);
            enc.dispatchThreads(mtl.Size.of(t.hidden, rows, 1), mtl.Size.of(256, 1, 1));
        }
    }
};

/// The bf16 a tap row holds for `x`: round to nearest even, a NaN kept quiet.
pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (u & 0x7f800000 == 0x7f800000 and u & 0x007fffff != 0) return @intCast((u >> 16) | 0x40);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

test "tap rows round to nearest even and keep NaN quiet" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expectEqual(@as(u16, 0x3f81), bf16(@bitCast(@as(u32, 0x3f808001))));
    try std.testing.expectEqual(@as(u16, 0x7f80), bf16(std.math.inf(f32)));
    try std.testing.expect(bf16(std.math.nan(f32)) & 0x7f80 == 0x7f80 and bf16(std.math.nan(f32)) & 0x7f != 0);
}
