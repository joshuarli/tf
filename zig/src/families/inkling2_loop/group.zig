//! Grouped verification: streams' newest tokens and drafts in one batch step, each row's bits a batch step's.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const Tensor = @import("model.zig").Tensor;
const state = @import("state.zig");
const Cache = state.Cache;
const prefill = @import("prefill.zig");
const batch_mod = @import("batch.zig");

pub const source = @import("kernels.zig").attention ++ "\n" ++ @embedFile("kernels/verify.metal");

/// The batch step's few-row kernels keep each row's arithmetic independent of the step's rows up to eight.
pub const max_rows: usize = 8;

const DecodeParts = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, splits: u32, chunk: u32, scale: f32, floor: f32, alpha: f32 };
const BatchRow = extern struct { keys: u64, values: u64, args: DecodeParts };
const GroupRow = extern struct { keys: u64, values: u64, args: DecodeParts, first: u32, gpos: u32 };
const RelArgs = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, scale: f32, floor: f32, alpha: f32, eps: f32 };
const ConvRow = extern struct { ring: u64, pos: u32, first: u32 };
const ConvArgs = extern struct { channels: u32, heads: u32, ring: u32, pos: u32, layout: u32, norm: f32, residual: u32 };
const CommitRow = extern struct { keys: u64, values: u64, conv: [4]u64, pos: u32, capacity: u32, keep: u32, conv_keep: u32 };
const CommitSlot = extern struct { local: u32, locals_before: u32, globals_before: u32, pad: u32 = 0 };
const CommitArgs = extern struct { kvw: u32, hidden: u32, rows: u32, stride: u32, slots: u32, window: u32, ring: u32 };
const RoundRow = extern struct { keys: u64, values: u64, conv: [4]u64, pos: u32, capacity: u32, owner: u32, j: u32 };
const Stream = extern struct { first: u32, size: u32 };

const names = [_][:0]const u8{ "i2v_conv", "i2v_store_rel", "i2v_split", "i2v_commit_kv", "i2v_commit_conv", "i2v_argmax", "i2v_accept", "i2v_round_kv", "i2v_round_conv" };
const Kernel = enum { conv, store_rel, split, commit_kv, commit_conv, argmax, accept, round_kv, round_conv };

pub const Finish = enum { picks, commit };
pub const Decided = struct { picks: []const u32, accepted: []const u32 };

pub const Verifier = struct {
    kernels: [names.len]mtl.Pipeline,
    kv: mtl.Buffer,
    conv: prefill.ConvCapture,
    targets: mtl.Buffer,
    accepted: mtl.Buffer,
    slots: [64]CommitSlot,
    slot_count: usize,

    pub fn init(device: mtl.Device, config: *const @import("config.zig").Config) !Verifier {
        var v: Verifier = undefined;
        if (config.slots() > v.slots.len) return error.TooManySlots;
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var made: usize = 0;
        errdefer for (v.kernels[0..made]) |p| p.deinit();
        for (names, 0..) |name, i| {
            v.kernels[i] = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        const kvw: usize = config.text.local.kv * config.text.local.dim;
        v.kv = try device.buffer(config.slots() * 2 * max_rows * kvw * 2, mtl.ResourceOptions.private);
        errdefer v.kv.deinit();
        v.conv = try prefill.ConvCapture.init(device, config, max_rows);
        errdefer v.conv.deinit();
        v.targets = try device.buffer(max_rows * 4, mtl.ResourceOptions.shared);
        errdefer v.targets.deinit();
        v.accepted = try device.buffer(max_rows * 4, mtl.ResourceOptions.shared);
        v.slot_count = config.slots();
        var locals: u32 = 0;
        var globals: u32 = 0;
        for (0..config.slots()) |s| {
            const local = Cache.isLocalSlot(config, s);
            v.slots[s] = .{ .local = @intFromBool(local), .locals_before = locals, .globals_before = globals };
            if (local) locals += 1 else globals += 1;
        }
        return v;
    }

    pub fn deinit(v: *Verifier) void {
        for (v.kernels) |p| p.deinit();
        v.kv.deinit();
        v.conv.deinit();
        v.targets.deinit();
        v.accepted.deinit();
    }

    fn pipe(v: *const Verifier, k: Kernel) mtl.Pipeline {
        return v.kernels[@intFromEnum(k)];
    }

    pub fn group(v: *const Verifier, caches: []const *Cache, sizes: []const usize) !Group {
        if (caches.len == 0 or caches.len != sizes.len) return error.BadGroup;
        var g = Group{ .verifier = v, .caches = caches, .sizes = sizes, .rows = 0 };
        for (caches, sizes, 0..) |cache, size, i| {
            if (size == 0) return error.BadGroup;
            if (cache.len + size > cache.capacity) return error.ContextFull;
            for (caches[0..i]) |prior| if (prior == cache) return error.DuplicateCache;
            for (0..size) |j| {
                if (g.rows == max_rows) return error.BadGroup;
                g.owner[g.rows] = @intCast(i);
                g.pos[g.rows] = @intCast(cache.len + j);
                g.first[g.rows] = @intCast(g.rows - j);
                g.rows += 1;
            }
        }
        return g;
    }

    pub fn forward(_: *const Verifier, m: *const Model, batch: *batch_mod.BatchScratch, g: *const Group, ids: []const u32) !void {
        if (ids.len != g.rows or ids.len > batch.rows.chunk) return error.BadGroup;
        try prefill.groupStep(m, batch, g, ids);
    }

    pub fn decide(v: *const Verifier, m: *const Model, batch: *batch_mod.BatchScratch, g: *const Group, ids: []const u32, finish: Finish) !Decided {
        if (ids.len != g.rows or ids.len > batch.rows.chunk) return error.BadGroup;
        try prefill.groupFinish(m, batch, g, ids, finish);
        const accepted = v.accepted.slice(u32, g.caches.len);
        if (finish == .commit) for (g.caches, accepted) |cache, a| {
            cache.len += a + 1;
        };
        return .{ .picks = v.targets.slice(u32, g.rows), .accepted = accepted };
    }

    pub fn encodeFinish(v: *const Verifier, enc: mtl.ComputeEncoder, m: *const Model, ids: mtl.Buffer, logits: mtl.Buffer, g: *const Group, finish: Finish) void {
        enc.setPipeline(v.pipe(.argmax));
        enc.setBuffer(logits, 0, 0);
        enc.setValue(@as(u32, m.config.text.unpadded_vocab), 1);
        enc.setBuffer(v.targets, 0, 2);
        enc.dispatchGroups(mtl.Size.of(g.rows, 1, 1), mtl.Size.of(1024, 1, 1));
        if (finish == .picks) return;
        var streams: [max_rows]Stream = undefined;
        var at: u32 = 0;
        for (g.sizes, 0..) |size, i| {
            streams[i] = .{ .first = at, .size = @intCast(size) };
            at += @intCast(size);
        }
        enc.setPipeline(v.pipe(.accept));
        enc.setBuffer(ids, 0, 0);
        enc.setBuffer(v.targets, 0, 1);
        enc.setBytes(std.mem.sliceAsBytes(streams[0..g.caches.len]), 2);
        enc.setValue(@as(u32, @intCast(g.caches.len)), 3);
        enc.setBuffer(v.accepted, 0, 4);
        enc.dispatchThreads(mtl.Size.of(g.caches.len, 1, 1), mtl.Size.of(32, 1, 1));
        const c = &m.config;
        var rows: [max_rows]RoundRow = undefined;
        for (0..g.rows) |r| {
            const cache = g.caches[g.owner[r]];
            rows[r] = .{
                .keys = cache.keys.gpuAddress(),
                .values = cache.values.gpuAddress(),
                .conv = .{ cache.k_conv.gpuAddress(), cache.v_conv.gpuAddress(), cache.attn_conv.gpuAddress(), cache.mlp_conv.gpuAddress() },
                .pos = g.pos[r],
                .capacity = @intCast(cache.capacity),
                .owner = g.owner[r],
                .j = @intCast(r - g.first[r]),
            };
        }
        const kvw: usize = c.text.local.kv * c.text.local.dim;
        const args = CommitArgs{ .kvw = @intCast(kvw), .hidden = c.text.hidden, .rows = @intCast(g.rows), .stride = max_rows, .slots = @intCast(v.slot_count), .window = c.text.window, .ring = state.ring };
        for (g.caches) |cache| for ([_]mtl.Buffer{ cache.keys, cache.values, cache.k_conv, cache.v_conv, cache.attn_conv, cache.mlp_conv }) |buf| enc.useResource(buf, mtl.ResourceUsage.write);
        enc.setPipeline(v.pipe(.round_kv));
        enc.setBuffer(v.kv, 0, 0);
        enc.setValue(args, 1);
        enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 2);
        enc.setBytes(std.mem.sliceAsBytes(v.slots[0..v.slot_count]), 3);
        enc.setBuffer(v.accepted, 0, 4);
        enc.dispatchThreads(mtl.Size.of(kvw, g.rows, v.slot_count * 2), mtl.Size.of(256, 1, 1));
        enc.setPipeline(v.pipe(.round_conv));
        enc.setBuffer(v.conv.buffer, 0, 0);
        enc.setValue(args, 1);
        enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 2);
        enc.setBuffer(v.accepted, 0, 3);
        enc.dispatchThreads(mtl.Size.of(@max(kvw, c.text.hidden), g.rows, v.slot_count * 4), mtl.Size.of(256, 1, 1));
    }

    pub fn picks(v: *const Verifier, m: *const Model, logits: mtl.Buffer, rows: usize) ![]const u32 {
        if (rows == 0 or rows > max_rows) return error.BadGroup;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(v.pipe(.argmax));
        enc.setBuffer(logits, 0, 0);
        enc.setValue(@as(u32, m.config.text.unpadded_vocab), 1);
        enc.setBuffer(v.targets, 0, 2);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling grouped picks: {s}", .{message});
            return error.GpuFailed;
        }
        return v.targets.slice(u32, rows);
    }

    pub fn commit(v: *const Verifier, m: *const Model, g: *const Group, accepted: []const usize) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const enc = cb.compute(.serial);
        try v.encodeCommit(enc, m, g, accepted);
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling grouped commit: {s}", .{message});
            return error.GpuFailed;
        }
        committed(g, accepted);
    }

    pub fn encodeCommit(v: *const Verifier, enc: mtl.ComputeEncoder, m: *const Model, g: *const Group, accepted: []const usize) !void {
        if (accepted.len != g.caches.len) return error.BadAcceptedPrefix;
        for (accepted, g.sizes) |a, size| if (a >= size) return error.BadAcceptedPrefix;
        const c = &m.config;
        var rows: [max_rows]CommitRow = undefined;
        for (0..g.rows) |r| {
            const i = g.owner[r];
            const cache = g.caches[i];
            const j = r - g.first[r];
            const kept = accepted[i] + 1;
            rows[r] = .{
                .keys = cache.keys.gpuAddress(),
                .values = cache.values.gpuAddress(),
                .conv = .{ cache.k_conv.gpuAddress(), cache.v_conv.gpuAddress(), cache.attn_conv.gpuAddress(), cache.mlp_conv.gpuAddress() },
                .pos = g.pos[r],
                .capacity = @intCast(cache.capacity),
                .keep = @intFromBool(j < kept),
                .conv_keep = @intFromBool(j + state.ring >= kept),
            };
        }
        const kvw: usize = c.text.local.kv * c.text.local.dim;
        const args = CommitArgs{ .kvw = @intCast(kvw), .hidden = c.text.hidden, .rows = @intCast(g.rows), .stride = max_rows, .slots = @intCast(v.slot_count), .window = c.text.window, .ring = state.ring };
        for (g.caches) |cache| for ([_]mtl.Buffer{ cache.keys, cache.values, cache.k_conv, cache.v_conv, cache.attn_conv, cache.mlp_conv }) |buf| enc.useResource(buf, mtl.ResourceUsage.write);
        enc.setPipeline(v.pipe(.commit_kv));
        enc.setBuffer(v.kv, 0, 0);
        enc.setValue(args, 1);
        enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 2);
        enc.setBytes(std.mem.sliceAsBytes(v.slots[0..v.slot_count]), 3);
        enc.dispatchThreads(mtl.Size.of(kvw, g.rows, v.slot_count * 2), mtl.Size.of(256, 1, 1));
        enc.setPipeline(v.pipe(.commit_conv));
        enc.setBuffer(v.conv.buffer, 0, 0);
        enc.setValue(args, 1);
        enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 2);
        enc.dispatchThreads(mtl.Size.of(@max(kvw, c.text.hidden), g.rows, v.slot_count * 4), mtl.Size.of(256, 1, 1));
    }
};

pub fn committed(g: *const Group, accepted: []const usize) void {
    for (g.caches, accepted) |cache, a| cache.len += a + 1;
}

pub const Group = struct {
    verifier: *const Verifier,
    caches: []const *Cache,
    sizes: []const usize,
    rows: usize,
    owner: [max_rows]u8 = @splat(0),
    pos: [max_rows]u32 = @splat(0),
    first: [max_rows]u32 = @splat(0),

    pub fn conv(g: *const Group, enc: mtl.ComputeEncoder, x: mtl.Buffer, x_off: usize, weight: Tensor, dyn: ?mtl.Buffer, dyn_off: usize, out: mtl.Buffer, out_off: usize, channels: u32, heads: u32, layout: u32, norm: f32, slot: u32, site: prefill.ConvSite) void {
        const v = g.verifier;
        var rows: [max_rows]ConvRow = undefined;
        for (0..g.rows) |r| {
            const cache = g.caches[g.owner[r]];
            const buf = switch (site) {
                .key => cache.k_conv,
                .value => cache.v_conv,
                .attention => cache.attn_conv,
                .mlp => cache.mlp_conv,
            };
            rows[r] = .{ .ring = buf.gpuAddress() + cache.convAt(slot, channels), .pos = g.pos[r], .first = g.first[r] };
            enc.useResource(buf, mtl.ResourceUsage.read);
        }
        enc.setPipeline(v.pipe(.conv));
        enc.setBuffer(x, x_off, 0);
        enc.setBuffer(weight.buffer, weight.offset, 1);
        enc.setBuffer(dyn orelse x, if (dyn != null) dyn_off else x_off, 2);
        enc.setValue(ConvArgs{ .channels = channels, .heads = heads, .ring = state.ring, .pos = 0, .layout = layout, .norm = norm, .residual = @intFromBool(site == .mlp) }, 3);
        enc.setBuffer(out, out_off, 4);
        enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 5);
        enc.setBuffer(v.conv.buffer, v.conv.offset(slot, site, 0), 6);
        enc.dispatchThreads(mtl.Size.of(channels, g.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub const AttnBuffers = struct { q: mtl.Buffer, k: mtl.Buffer, v: mtl.Buffer, r: mtl.Buffer, sink: mtl.Buffer, relbias: mtl.Buffer, out: mtl.Buffer, parts: mtl.Buffer, combine: mtl.Pipeline };

    pub fn attend(g: *const Group, enc: mtl.ComputeEncoder, m: *const Model, rel_bias: Tensor, slot: u32, full: bool, extent: u32, bufs: AttnBuffers) void {
        const v = g.verifier;
        const config = &m.config;
        const text = &config.text;
        const heads: u32 = text.local.q;
        const kv: u32 = text.local.kv;
        const dim: u32 = text.local.dim;
        const kvw: usize = @as(usize, kv) * dim;
        const win: u32 = if (full) 0 else text.window;
        var rows: [max_rows]GroupRow = undefined;
        var plain: [max_rows]BatchRow = undefined;
        var positions: [max_rows]u32 = undefined;
        var max_splits: u32 = 0;
        for (0..g.rows) |r| {
            const cache = g.caches[g.owner[r]];
            const pos = g.pos[r];
            const lo: u32 = if (win != 0 and pos + 1 > win) pos + 1 - win else 0;
            const part = state.attnPartition(pos + 1 - lo);
            const parts = DecodeParts{ .heads = heads, .kv = kv, .dim = dim, .rel = text.d_rel, .extent = extent, .window = win, .pos = pos, .tau = @intFromBool(full), .splits = part.splits, .chunk = part.chunk, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim))), .floor = @floatFromInt(text.log_floor), .alpha = text.log_alpha };
            const base = cache.row(config, slot, 0);
            rows[r] = .{ .keys = cache.keys.gpuAddress() + base, .values = cache.values.gpuAddress() + base, .args = parts, .first = g.first[r], .gpos = pos - (@as(u32, @intCast(r)) - g.first[r]) };
            plain[r] = .{ .keys = rows[r].keys, .values = rows[r].values, .args = parts };
            positions[r] = pos;
            max_splits = @max(max_splits, part.splits);
            enc.useResource(cache.keys, mtl.ResourceUsage.read);
            enc.useResource(cache.values, mtl.ResourceUsage.read);
        }
        const kh = (@as(usize, slot) * 2 + 0) * max_rows * kvw * 2;
        const vh = (@as(usize, slot) * 2 + 1) * max_rows * kvw * 2;
        enc.setPipeline(v.pipe(.store_rel));
        enc.setBuffer(bufs.k, 0, 0);
        enc.setBuffer(bufs.v, 0, 1);
        enc.setBuffer(bufs.r, 0, 2);
        enc.setBuffer(rel_bias.buffer, rel_bias.offset, 3);
        enc.setValue(RelArgs{ .heads = heads, .kv = kv, .dim = dim, .rel = text.d_rel, .extent = extent, .window = 0, .pos = 0, .tau = @intFromBool(full), .scale = 0, .floor = @floatFromInt(text.log_floor), .alpha = text.log_alpha, .eps = text.eps }, 4);
        enc.setBuffer(bufs.relbias, 0, 5);
        enc.setBytes(std.mem.sliceAsBytes(positions[0..g.rows]), 6);
        enc.setBuffer(v.kv, kh, 7);
        enc.setBuffer(v.kv, vh, 8);
        enc.dispatchGroups(mtl.Size.of(kv + (heads * (extent + 1) + 255) / 256, g.rows, 1), mtl.Size.of(256, 1, 1));
        enc.setPipeline(v.pipe(.split));
        enc.setBuffer(bufs.q, 0, 0);
        enc.setBuffer(bufs.relbias, 0, 1);
        enc.setBuffer(bufs.sink, 0, 2);
        enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 3);
        enc.setBuffer(bufs.parts, 0, 4);
        enc.setBuffer(v.kv, kh, 5);
        enc.setBuffer(v.kv, vh, 6);
        enc.dispatchGroups(mtl.Size.of(kv * max_splits, g.rows, 1), mtl.Size.of((heads / kv) * 32, 1, 1));
        enc.setPipeline(bufs.combine);
        enc.setBuffer(bufs.parts, 0, 0);
        enc.setBytes(std.mem.sliceAsBytes(plain[0..g.rows]), 1);
        enc.setBuffer(bufs.out, 0, 2);
        enc.dispatchGroups(mtl.Size.of(heads, g.rows, 1), mtl.Size.of(32, 1, 1));
    }
};

pub fn acceptedDrafts(drafts: []const u32, targets: []const u32) usize {
    var n: usize = 0;
    while (n < drafts.len and n < targets.len and drafts[n] == targets[n]) n += 1;
    return n;
}

test "the grouped verification kernels compile with the attention they twin" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    for (names) |name| {
        const p = try mtl.Pipeline.init(device, lib, name, false);
        p.deinit();
    }
}

test "drafts are accepted while each equals the target's pick before it" {
    try std.testing.expectEqual(@as(usize, 0), acceptedDrafts(&.{ 5, 6 }, &.{ 4, 6, 7 }));
    try std.testing.expectEqual(@as(usize, 1), acceptedDrafts(&.{ 5, 6 }, &.{ 5, 7, 7 }));
    try std.testing.expectEqual(@as(usize, 2), acceptedDrafts(&.{ 5, 6 }, &.{ 5, 6, 7 }));
    try std.testing.expectEqual(@as(usize, 0), acceptedDrafts(&.{}, &.{9}));
}
