//! bf16 tap rows `[rows][count * hidden]`: the block residual entering listed visits, and the final norm (id 48).
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

pub const max_taps = 8;

const Args = extern struct { width: u32, rows: u32, stride: u32, column: u32 };
const RingArgs = extern struct { width: u32, rows: u32, stride: u32, column: u32, skip: u32, base: u32, window: u32 };

pub const Taps = struct {
    buffer: mtl.Buffer,
    copy: mtl.Pipeline,
    copy_ring: mtl.Pipeline,
    rows: usize,
    hidden: u32,
    visits: [max_taps]u32 = @splat(0),
    count: u32,
    final: u32,

    pub const RingWrite = struct {
        taps: *const Taps,
        ring: mtl.Buffer,
        skip: u32,
        rows: u32,
        base: u32,
        window: u32,

        pub fn encode(w: RingWrite, enc: mtl.ComputeEncoder, visit: u32, src: mtl.Buffer, offset: usize) void {
            if (w.rows == 0) return;
            for (w.taps.visits[0..w.taps.count], 0..) |v, i| {
                if (v != visit) continue;
                enc.setPipeline(w.taps.copy_ring);
                enc.setBuffer(src, offset, 0);
                enc.setValue(RingArgs{
                    .width = w.taps.hidden,
                    .rows = w.rows,
                    .stride = w.taps.count * w.taps.hidden,
                    .column = @as(u32, @intCast(i)) * w.taps.hidden,
                    .skip = w.skip,
                    .base = w.base,
                    .window = w.window,
                }, 1);
                enc.setBuffer(w.ring, 0, 2);
                enc.dispatchThreads(mtl.Size.of(w.taps.hidden, w.rows, 1), mtl.Size.of(256, 1, 1));
            }
        }
    };

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
        const copy_ring = try mtl.Pipeline.init(device, lib, "i2t_tap_ring", false);
        errdefer copy_ring.deinit();
        const size = rows * visits.len * config.text.hidden * 2;
        var t = Taps{ .buffer = try device.buffer(size, mtl.ResourceOptions.shared), .copy = copy, .copy_ring = copy_ring, .rows = rows, .hidden = config.text.hidden, .count = @intCast(visits.len), .final = config.slots() };
        @memcpy(t.visits[0..visits.len], visits);
        return t;
    }

    pub fn deinit(t: *Taps) void {
        t.copy_ring.deinit();
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

test "ring tap capture writes skipped and wrapped rows identically to flat taps without touching other slots" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var c: cfg.Config = undefined;
    c.text.hidden = 16;
    c.text.layers = 4;
    c.loop = .{ .core_start = 1, .core_end = 3, .passes = 1, .gate_rank = 16 };
    var taps = try Taps.init(device, &c, &.{ 1, 4 }, 6);
    defer taps.deinit();
    const src = try device.buffer(6 * 16 * 4, mtl.ResourceOptions.shared);
    defer src.deinit();
    for (src.slice(f32, 6 * 16), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 40)) * 0.125;
    const window: u32 = 5;
    const ring = try device.buffer(window * taps.rowBytes(), mtl.ResourceOptions.shared);
    defer ring.deinit();
    const sentinel: u16 = 0x5a5a;
    @memset(ring.slice(u16, window * 2 * 16), sentinel);
    @memset(taps.buffer.slice(u16, 6 * 2 * 16), 0);
    const rw = Taps.RingWrite{ .taps = &taps, .ring = ring, .skip = 2, .rows = 4, .base = 3, .window = window };
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    taps.encode(enc, 1, src, 0, 6);
    taps.encode(enc, 4, src, 0, 6);
    rw.encode(enc, 1, src, 0);
    rw.encode(enc, 4, src, 0);
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    const flat_bytes = taps.bytes(0, 6);
    const ring_bytes = ring.contents();
    const row_bytes = taps.rowBytes();
    for (0..4) |r| {
        const dst = (3 + r) % window;
        try std.testing.expectEqualSlices(u8, flat_bytes[(2 + r) * row_bytes ..][0..row_bytes], ring_bytes[dst * row_bytes ..][0..row_bytes]);
    }
    const untouched = ring.slice(u16, window * 2 * 16)[2 * 32 ..][0..32];
    for (untouched) |word| try std.testing.expectEqual(sentinel, word);
}
