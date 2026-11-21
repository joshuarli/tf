//! A stream's per-position state and the scratch one token's forward needs. Each executed slot (a physical layer, or a
//! core layer in one extra pass) owns its keys, values and four convolution rings; the weights are shared.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

/// Positions a convolution ring keeps: the three tokens a kernel-4 tap reaches, plus the slot being written.
pub const ring = 4;
/// The most states one depth mix folds: the token's four block boundaries and itself.
pub const mix_states = 5;

pub const Cache = struct {
    keys: mtl.Buffer,
    values: mtl.Buffer,
    k_conv: mtl.Buffer,
    v_conv: mtl.Buffer,
    attn_conv: mtl.Buffer,
    mlp_conv: mtl.Buffer,
    capacity: usize,
    len: usize = 0,

    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize) !Cache {
        const kv: usize = c.text.local.kv * c.text.local.dim;
        const h: usize = c.text.hidden;
        const slots: usize = c.slots();
        var out = Cache{ .keys = undefined, .values = undefined, .k_conv = undefined, .v_conv = undefined, .attn_conv = undefined, .mlp_conv = undefined, .capacity = capacity };
        out.keys = try device.buffer(@max(slots * capacity * kv * 4, 16), mtl.ResourceOptions.private);
        errdefer out.keys.deinit();
        out.values = try device.buffer(@max(slots * capacity * kv * 4, 16), mtl.ResourceOptions.private);
        errdefer out.values.deinit();
        out.k_conv = try device.buffer(@max(slots * ring * kv * 4, 16), mtl.ResourceOptions.private);
        errdefer out.k_conv.deinit();
        out.v_conv = try device.buffer(@max(slots * ring * kv * 4, 16), mtl.ResourceOptions.private);
        errdefer out.v_conv.deinit();
        out.attn_conv = try device.buffer(@max(slots * ring * h * 4, 16), mtl.ResourceOptions.private);
        errdefer out.attn_conv.deinit();
        out.mlp_conv = try device.buffer(@max(slots * ring * h * 4, 16), mtl.ResourceOptions.private);
        errdefer out.mlp_conv.deinit();
        return out;
    }

    pub fn deinit(c: *Cache) void {
        c.keys.deinit();
        c.values.deinit();
        c.k_conv.deinit();
        c.v_conv.deinit();
        c.attn_conv.deinit();
        c.mlp_conv.deinit();
    }

    /// Slot `s`'s row for position `at` in the key or value buffer, in bytes.
    pub fn row(c: *const Cache, c0: *const cfg.Config, s: usize, at: usize) usize {
        const slot: usize = c.capacity * c0.text.local.kv * c0.text.local.dim * 4;
        return s * slot + at * c0.text.local.kv * c0.text.local.dim * 4;
    }

    /// Slot `s`'s convolution ring for a site `channels` wide, in bytes.
    pub fn convAt(_: *const Cache, s: usize, channels: usize) usize {
        return s * ring * channels * 4;
    }
};

pub const Scratch = struct {
    ids: mtl.Buffer, // [1] u32, shared so the host can set it
    x: mtl.Buffer, // the residual [h]
    prelude: mtl.Buffer, // the prelude's output, restored by every extra pass [h]
    passout: mtl.Buffer, // a finished pass's final residual, the next pass's bank output [h]
    bank: mtl.Buffer, // the finished pass's core blocks after the first [mix_states][h]
    ffnin: mtl.Buffer, // the FFN sub-layer's residual [h]
    xn: mtl.Buffer, // [3][h] the normed attention inputs
    q: mtl.Buffer, // [q_width]
    k: mtl.Buffer, // [kv_width] the raw K projection
    v: mtl.Buffer,
    r: mtl.Buffer, // [heads][d_rel]
    kc: mtl.Buffer, // the K convolution's output [kv_width]
    vc: mtl.Buffer,
    kdyn: mtl.Buffer, // [kv][taps] the full layers' per-head dynamic taps
    vdyn: mtl.Buffer,
    sink: mtl.Buffer, // [heads]
    relbias: mtl.Buffer, // [heads][extent + 1]
    attn: mtl.Buffer, // [q_width]
    gate: mtl.Buffer, // [q_width]
    projout: mtl.Buffer, // [h] an attention output projection
    o: mtl.Buffer, // [h] the attention sub-layer's output
    aconv: mtl.Buffer, // [h] the FFN convolution tail
    blocks: mtl.Buffer, // [mix_states][h] the token's block boundaries, oldest first
    states: mtl.Buffer, // [mix_states][h] one mix's states, newest first then cur
    mixs: mtl.Buffer, // [mix_states][h] the same, normalised
    mixout: mtl.Buffer, // [4][h] one mix's outputs
    msg: mtl.Buffer, // [4][h] the cross-pass messages
    normed: mtl.Buffer, // [rank]
    h2: mtl.Buffer, // [2][h] a controller's concat
    tanh: mtl.Buffer, // [2][h] a controller's gains and forgets
    router: mtl.Buffer, // [experts]
    picks: mtl.Buffer, // [topk] int
    wts: mtl.Buffer, // [topk]
    offsets: mtl.Buffer, // [experts + 1] int
    order: mtl.Buffer, // [topk] int
    act: mtl.Buffer, // [topk][inter]
    down: mtl.Buffer, // [topk][h]
    moe: mtl.Buffer, // [h]
    logits: mtl.Buffer, // [unpadded vocab], shared so the host can read it

    pub fn init(device: mtl.Device, c: *const cfg.Config) !Scratch {
        const h: usize = c.text.hidden;
        const qw: usize = c.text.local.q * c.text.local.dim;
        const kvw: usize = c.text.local.kv * c.text.local.dim;
        const heads: usize = c.text.local.q;
        const kv: usize = c.text.local.kv;
        const extent: usize = @max(c.text.rel_extent, c.text.window) + 1;
        const inter: usize = c.text.expert_inter;
        const topk: usize = c.text.topk;
        const rank: usize = c.loop.gate_rank;
        const experts: usize = c.text.experts;
        var s: Scratch = undefined;
        const alloc = [_]struct { ptr: *mtl.Buffer, bytes: usize }{
            .{ .ptr = &s.x, .bytes = h * 4 },
            .{ .ptr = &s.prelude, .bytes = h * 4 },
            .{ .ptr = &s.passout, .bytes = h * 4 },
            .{ .ptr = &s.bank, .bytes = mix_states * h * 4 },
            .{ .ptr = &s.ffnin, .bytes = h * 4 },
            .{ .ptr = &s.xn, .bytes = 3 * h * 4 },
            .{ .ptr = &s.q, .bytes = qw * 4 },
            .{ .ptr = &s.k, .bytes = kvw * 4 },
            .{ .ptr = &s.v, .bytes = kvw * 4 },
            .{ .ptr = &s.r, .bytes = heads * c.text.d_rel * 4 },
            .{ .ptr = &s.kc, .bytes = kvw * 4 },
            .{ .ptr = &s.vc, .bytes = kvw * 4 },
            .{ .ptr = &s.kdyn, .bytes = kv * 4 * 4 },
            .{ .ptr = &s.vdyn, .bytes = kv * 4 * 4 },
            .{ .ptr = &s.sink, .bytes = heads * 4 },
            .{ .ptr = &s.relbias, .bytes = heads * extent * 4 },
            .{ .ptr = &s.attn, .bytes = qw * 4 },
            .{ .ptr = &s.gate, .bytes = qw * 4 },
            .{ .ptr = &s.projout, .bytes = h * 4 },
            .{ .ptr = &s.o, .bytes = h * 4 },
            .{ .ptr = &s.aconv, .bytes = h * 4 },
            .{ .ptr = &s.blocks, .bytes = mix_states * h * 4 },
            .{ .ptr = &s.states, .bytes = mix_states * h * 4 },
            .{ .ptr = &s.mixs, .bytes = mix_states * h * 4 },
            .{ .ptr = &s.mixout, .bytes = 4 * h * 4 },
            .{ .ptr = &s.msg, .bytes = 4 * h * 4 },
            .{ .ptr = &s.normed, .bytes = rank * 4 },
            .{ .ptr = &s.h2, .bytes = 2 * h * 4 },
            .{ .ptr = &s.tanh, .bytes = 2 * h * 4 },
            .{ .ptr = &s.router, .bytes = experts * 4 },
            .{ .ptr = &s.picks, .bytes = topk * 4 },
            .{ .ptr = &s.wts, .bytes = topk * 4 },
            .{ .ptr = &s.offsets, .bytes = (experts + 1) * 4 },
            .{ .ptr = &s.order, .bytes = topk * 4 },
            .{ .ptr = &s.act, .bytes = topk * inter * 4 },
            .{ .ptr = &s.down, .bytes = topk * h * 4 },
            .{ .ptr = &s.moe, .bytes = h * 4 },
        };
        var made: usize = 0;
        errdefer for (alloc[0..made]) |f| f.ptr.deinit();
        for (alloc) |f| {
            f.ptr.* = try device.buffer(@max(f.bytes, 16), mtl.ResourceOptions.private);
            made += 1;
        }
        s.ids = try device.buffer(16, mtl.ResourceOptions.shared);
        s.logits = try device.buffer(@max(c.text.unpadded_vocab * 4, 16), mtl.ResourceOptions.shared);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for ([_][]const u8{"ids", "x", "prelude", "passout", "bank", "ffnin", "xn", "q", "k", "v", "r", "kc", "vc", "kdyn", "vdyn", "sink", "relbias", "attn", "gate", "projout", "o", "aconv", "blocks", "states", "mixs", "mixout", "msg", "normed", "h2", "tanh", "router", "picks", "wts", "offsets", "order", "act", "down", "moe", "logits"}) |f| @field(s, f).deinit();
    }
};

test "the state's rings and mix lists carry the graph's widths" {
    try std.testing.expectEqual(@as(usize, 4), ring);
    try std.testing.expectEqual(@as(usize, 5), mix_states);
}
