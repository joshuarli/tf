//! A stream's per-position state and the scratch one token's forward needs. Each executed slot (a physical layer, or a
//! core layer in one extra pass) owns its keys, values and four convolution rings; the weights are shared.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const kernels = @import("kernels.zig");

/// Positions a convolution ring keeps: the three tokens a kernel-4 tap reaches, plus the slot being written.
pub const ring = 4;
/// The most states one depth mix folds: the token's four block boundaries and itself.
pub const mix_states = 5;
/// Split-K decode attention: a chunk walks at most this many window positions.
pub const attn_chunk = 64;
/// Split-K decode attention: the most chunks one window splits into.
pub const attn_splits = 16;

/// The split of `n` window positions into contiguous chunks: the fewest chunks of at most attn_chunk positions.
pub fn attnPartition(n: usize) struct { splits: u32, chunk: u32 } {
    const splits: u32 = @intCast(@max(1, @min(attn_splits, (n + attn_chunk - 1) / attn_chunk)));
    return .{ .splits = splits, .chunk = @intCast((n + splits - 1) / splits) };
}

pub const Cache = struct {
    keys: mtl.Buffer,
    values: mtl.Buffer,
    k_conv: mtl.Buffer,
    v_conv: mtl.Buffer,
    attn_conv: mtl.Buffer,
    mlp_conv: mtl.Buffer,
    capacity: usize,
    len: usize = 0,

    pub fn isLocalSlot(c0: *const cfg.Config, s: usize) bool {
        const phys = c0.loop.physicalLayer(c0.text.layers, @intCast(s));
        return !c0.isFull(phys);
    }

    pub fn slotRows(c0: *const cfg.Config, s: usize, capacity: usize) usize {
        return if (isLocalSlot(c0, s)) c0.text.window else capacity;
    }

    pub fn globalSlots(c0: *const cfg.Config) usize {
        var n: usize = 0;
        for (0..c0.slots()) |s| {
            if (!isLocalSlot(c0, s)) n += 1;
        }
        return n;
    }

    pub fn localSlots(c0: *const cfg.Config) usize {
        return c0.slots() - globalSlots(c0);
    }

    pub fn keysBytes(c0: *const cfg.Config, capacity: usize) usize {
        const kv: usize = c0.text.local.kv * c0.text.local.dim;
        var total: usize = 0;
        for (0..c0.slots()) |s| total += slotRows(c0, s, capacity) * kv * 4;
        return total;
    }

    pub fn bytesPerToken(c0: *const cfg.Config) usize {
        const kv: usize = c0.text.local.kv * c0.text.local.dim;
        return globalSlots(c0) * kv * 4 * 2;
    }

    pub fn fixedBytes(c0: *const cfg.Config) usize {
        const kv: usize = c0.text.local.kv * c0.text.local.dim;
        const h: usize = c0.text.hidden;
        const slots: usize = c0.slots();
        const rings = localSlots(c0) * c0.text.window * kv * 4 * 2;
        const conv = slots * ring * kv * 4 * 2 + slots * ring * h * 4 * 2;
        return rings + conv;
    }

    pub fn longestFitting(c0: *const cfg.Config, budget: u64, most: usize) usize {
        const per: u64 = bytesPerToken(c0);
        const fixed: u64 = fixedBytes(c0);
        if (per == 0 or budget <= fixed) return 0;
        return @min(most, (budget - fixed) / per);
    }

    pub fn baseOffset(c: *const Cache, c0: *const cfg.Config, s: usize) usize {
        const kv: usize = c0.text.local.kv * c0.text.local.dim;
        var base: usize = 0;
        for (0..s) |t| base += slotRows(c0, t, c.capacity) * kv * 4;
        return base;
    }

    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize) !Cache {
        const kv: usize = c.text.local.kv * c.text.local.dim;
        const h: usize = c.text.hidden;
        const slots: usize = c.slots();
        var out = Cache{ .keys = undefined, .values = undefined, .k_conv = undefined, .v_conv = undefined, .attn_conv = undefined, .mlp_conv = undefined, .capacity = capacity };
        const total = keysBytes(c, capacity);
        out.keys = try device.buffer(@max(total, 16), mtl.ResourceOptions.private);
        errdefer out.keys.deinit();
        out.values = try device.buffer(@max(total, 16), mtl.ResourceOptions.private);
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

    /// Slot `s`'s row for position `at` in the key or value buffer, in bytes. Local slots keep a ring of
    /// `window` rows addressed by `at % window`; global slots keep the full capacity.
    pub fn row(c: *const Cache, c0: *const cfg.Config, s: usize, at: usize) usize {
        const kvw: usize = c0.text.local.kv * c0.text.local.dim;
        const r: usize = if (isLocalSlot(c0, s)) at % c0.text.window else at;
        return c.baseOffset(c0, s) + r * kvw * 4;
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
    passout: mtl.Buffer,
    bank: mtl.Buffer,
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
    attn_part: mtl.Buffer, // [heads][attn_splits][dim + 2] the split attention's per-chunk partials
    attn_split: mtl.Pipeline,
    attn_combine: mtl.Pipeline,
    gate: mtl.Buffer, // [q_width]
    projout: mtl.Buffer, // [h] an attention output projection
    o: mtl.Buffer, // [h] the attention sub-layer's output
    aconv: mtl.Buffer, // [h] the FFN convolution tail
    blocks: mtl.Buffer,
    mixs: mtl.Buffer,
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
    capture: mtl.Buffer, // [slots][3][h], shared: attn_out, ffn_out and l_out an executed slot, for the trace

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
        const aheads: usize = @max(c.text.local.q, c.text.global.q);
        const adim: usize = @max(c.text.local.dim, c.text.global.dim);
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
            .{ .ptr = &s.attn_part, .bytes = aheads * attn_splits * (adim + 2) * 4 },
            .{ .ptr = &s.gate, .bytes = qw * 4 },
            .{ .ptr = &s.projout, .bytes = h * 4 },
            .{ .ptr = &s.o, .bytes = h * 4 },
            .{ .ptr = &s.aconv, .bytes = h * 4 },
            .{ .ptr = &s.blocks, .bytes = mix_states * h * 4 },
            .{ .ptr = &s.mixs, .bytes = h * 4 },
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
        s.capture = try device.buffer(@max(c.slots() * trace_rows * h * 4, 16), mtl.ResourceOptions.shared);
        const attn_lib = try mtl.Library.fromSource(device, kernels.attention, mtl.CompileOptions.mlx());
        defer attn_lib.deinit();
        s.attn_split = try mtl.Pipeline.init(device, attn_lib, "i2_attn_split", false);
        errdefer s.attn_split.deinit();
        s.attn_combine = try mtl.Pipeline.init(device, attn_lib, "i2_attn_combine", false);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for ([_][]const u8{"ids", "x", "prelude", "passout", "bank", "ffnin", "xn", "q", "k", "v", "r", "kc", "vc", "kdyn", "vdyn", "sink", "relbias", "attn", "attn_part", "attn_split", "attn_combine", "gate", "projout", "o", "aconv", "blocks", "mixs", "mixout", "msg", "normed", "h2", "tanh", "router", "picks", "wts", "offsets", "order", "act", "down", "moe", "logits", "capture"}) |f| @field(s, f).deinit();
    }
};

test "the state's rings and mix lists carry the graph's widths" {
    try std.testing.expectEqual(@as(usize, 4), ring);
    try std.testing.expectEqual(@as(usize, 5), mix_states);
    for ([_]struct { n: usize, splits: u32, chunk: u32 }{
        .{ .n = 1, .splits = 1, .chunk = 1 },
        .{ .n = 64, .splits = 1, .chunk = 64 },
        .{ .n = 65, .splits = 2, .chunk = 33 },
        .{ .n = 1024, .splits = 16, .chunk = 64 },
        .{ .n = 1335, .splits = 16, .chunk = 84 },
    }) |c| {
        const p = attnPartition(c.n);
        try std.testing.expectEqual(c.splits, p.splits);
        try std.testing.expectEqual(c.chunk, p.chunk);
    }
}

test "local slots ring while global slots keep the full capacity" {
    var c: cfg.Config = std.mem.zeroes(cfg.Config);
    c.text.layers = 24;
    c.text.hidden = 2304;
    c.text.local = .{ .q = 24, .kv = 8, .dim = 128 };
    c.text.window = 1024;
    c.loop = .{ .core_start = 6, .core_end = 18, .passes = 3, .gate_rank = 64 };
    for (0..24) |i| c.local[i] = true;
    for ([_]u32{ 5, 11, 17, 23 }) |g| c.local[g] = false;
    try std.testing.expectEqual(@as(u32, 48), c.slots());
    try std.testing.expectEqual(@as(usize, 8), Cache.globalSlots(&c));
    try std.testing.expectEqual(@as(usize, 40), Cache.localSlots(&c));
    const kvw: usize = 8 * 128;
    try std.testing.expectEqual(8 * kvw * 4 * 2, Cache.bytesPerToken(&c));
    const cap: usize = 9000;
    try std.testing.expectEqual((40 * 1024 + 8 * cap) * kvw * 4, Cache.keysBytes(&c, cap));
    var probe = Cache{ .keys = undefined, .values = undefined, .k_conv = undefined, .v_conv = undefined, .attn_conv = undefined, .mlp_conv = undefined, .capacity = cap };
    for (0..c.slots()) |s| {
        const want = Cache.slotRows(&c, s, cap) * kvw * 4;
        try std.testing.expectEqual(want, probe.baseOffset(&c, s + 1) - probe.baseOffset(&c, s));
    }
    var local: usize = 0;
    while (local < c.slots() and !Cache.isLocalSlot(&c, local)) local += 1;
    try std.testing.expectEqual(probe.row(&c, local, 0), probe.row(&c, local, 1024));
    try std.testing.expectEqual(probe.row(&c, local, 1), probe.row(&c, local, 1025));
    try std.testing.expectEqual(probe.row(&c, local, 2047 % 1024), probe.row(&c, local, 2047));
    try std.testing.expect(probe.row(&c, local, 0) != probe.row(&c, local, 1));
    var global: usize = 0;
    while (global < c.slots() and Cache.isLocalSlot(&c, global)) global += 1;
    try std.testing.expectEqual(probe.baseOffset(&c, global) + 2048 * kvw * 4, probe.row(&c, global, 2048));
    const budget: u64 = Cache.fixedBytes(&c) + 8259 * Cache.bytesPerToken(&c);
    try std.testing.expectEqual(@as(usize, 8259), Cache.longestFitting(&c, budget, 1 << 20));
}

/// Rows one executed slot writes into the capture buffer, in the reference's order: the three mixed inputs, the three
/// normed ones, the attention output, the FFN's input and the slot's output.
pub const trace_rows = 9;
/// The summaries the trace keeps for one tensor: count, mean, standard deviation, min, max.
const Summary = struct { n: usize = 0, mean: f64 = 0, m2: f64 = 0, lo: f32 = std.math.inf(f32), hi: f32 = -std.math.inf(f32) };

/// Per-tensor summaries in the reference's execution order. The reference summarizes a whole tensor, which for a
/// teacher-forced pass spans every position, so each row folds in every position's values as the pass walks them.
pub const Trace = struct {
    gpa: std.mem.Allocator,
    slots: usize,
    h: usize,
    names: std.ArrayList([]const u8),
    sums: std.ArrayList(Summary),

    pub fn init(gpa: std.mem.Allocator, c: *const cfg.Config) !Trace {
        var t = Trace{ .gpa = gpa, .slots = c.slots(), .h = c.text.hidden, .names = .empty, .sums = .empty };
        try t.reserve(c);
        return t;
    }

    /// The names in the order they will be written: the reference's per-slot order, for every executed slot.
    fn reserve(t: *Trace, c: *const cfg.Config) !void {
        var slot: u32 = 0;
        while (slot < c.slots()) : (slot += 1) {
            for (0..3) |j| {
                const name = try std.fmt.allocPrint(t.gpa, "mixin-{d}-{d}", .{ slot, j });
                try t.names.append(t.gpa, name);
            }
            for (0..3) |j| {
                const name = try std.fmt.allocPrint(t.gpa, "normin-{d}-{d}", .{ slot, j });
                try t.names.append(t.gpa, name);
            }
            inline for (.{ "attn_out", "ffn_inp", "l_out" }) |prefix| {
                const name = try std.fmt.allocPrint(t.gpa, "{s}-{d}", .{ prefix, slot });
                try t.names.append(t.gpa, name);
            }
        }
        try t.sums.appendNTimes(t.gpa, .{}, t.names.items.len);
    }

    pub fn deinit(t: *Trace) void {
        for (t.names.items) |name| t.gpa.free(name);
        t.names.deinit(t.gpa);
        t.sums.deinit(t.gpa);
    }

    /// Fold this position's rows in. `capture` holds `[slots][trace_rows][h]` as the forward wrote it.
    pub fn position(t: *Trace, capture: mtl.Buffer, slots: []const u32, _: usize) !void {
        const rows = capture.slice(f32, t.slots * trace_rows * t.h);
        for (slots) |slot| {
            const base: usize = @as(usize, slot) * trace_rows * t.h;
            for (0..trace_rows) |r| t.fold(@as(usize, slot) * trace_rows + r, rows[base + r * t.h ..][0..t.h]);
        }
    }

    fn fold(t: *Trace, at: usize, x: []const f32) void {
        var s = &t.sums.items[at];
        for (x) |v| {
            s.n += 1;
            const d = @as(f64, v) - s.mean;
            s.mean += d / @as(f64, @floatFromInt(s.n));
            s.m2 += d * (@as(f64, v) - s.mean);
            if (v < s.lo) s.lo = v;
            if (v > s.hi) s.hi = v;
        }
    }

    pub fn write(t: *Trace, io: std.Io, path: []const u8) !void {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(t.gpa);
        for (t.names.items, 0..) |name, i| {
            const s = t.sums.items[i];
            if (s.n == 0) continue;
            const sd = @sqrt(s.m2 / @as(f64, @floatFromInt(s.n)));
            try out.print(t.gpa, "{s} {d:.9} {d:.9} {d:.9} {d:.9}\n", .{ name, s.mean, sd, s.lo, s.hi });
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
    }
};
