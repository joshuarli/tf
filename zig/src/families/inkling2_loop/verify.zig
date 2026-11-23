const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const Cache = @import("state.zig").Cache;
const ring = @import("state.zig").ring;
const prefill = @import("prefill.zig");

/// The most copy drafts a decode round verifies at once: a verify of 8 rows costs about four decode steps.
pub const hybrid_drafts: usize = 7;

pub const RoundTimes = struct { save: u64 = 0, forward: u64 = 0, argmax: u64 = 0, restore: u64 = 0, replay: u64 = 0 };
pub threadlocal var round_times: ?*RoundTimes = null;

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const Verification = struct {
    model: *const Model,
    cache: *Cache,
    scratch: *prefill.PrefillScratch,
    snapshot: mtl.Buffer,
    conv: prefill.ConvCapture,
    start: usize,
    rows: usize,

    pub fn begin(model: *const Model, scratch: *prefill.PrefillScratch, cache: *Cache, ids: []const u32, argmax: []u32) !Verification {
        const started = if (round_times != null) now() else 0;
        if (ids.len == 0 or ids.len > scratch.chunk or ids.len > model.config.text.window or argmax.len != ids.len) return error.BadChunk;
        const c = &model.config;
        const row_bytes: usize = c.text.local.kv * c.text.local.dim * 2;
        const conv_bytes = cache.k_conv.length() + cache.v_conv.length() + cache.attn_conv.length() + cache.mlp_conv.length();
        const bytes = conv_bytes + Cache.localSlots(c) * ids.len * row_bytes * 2;
        const snapshot = try model.device.buffer(@max(bytes, 16), mtl.ResourceOptions.private);
        errdefer snapshot.deinit();
        var conv = try prefill.ConvCapture.init(model.device, c, ids.len);
        errdefer conv.deinit();
        if (scratch.verify_conv != null) return error.VerificationActive;
        var v = Verification{ .model = model, .scratch = scratch, .cache = cache, .snapshot = snapshot, .conv = conv, .start = cache.len, .rows = ids.len };
        try v.copy(true);
        const saved = if (round_times != null) now() else 0;
        scratch.verify_conv = &v.conv;
        defer scratch.verify_conv = null;
        try prefill.prefillWant(model, scratch, cache, ids, .all);
        const forwarded = if (round_times != null) now() else 0;
        const vocab: usize = c.text.unpadded_vocab;
        const logits = scratch.logits.slice(f32, ids.len * vocab);
        for (argmax, 0..) |*out, row| {
            const values = logits[row * vocab ..][0..vocab];
            var best: usize = 0;
            for (values[1..], 1..) |value, i| if (value > values[best]) { best = i; };
            out.* = @intCast(best);
        }
        if (round_times) |t| {
            t.save = saved - started;
            t.forward = forwarded - saved;
            t.argmax = now() - forwarded;
        }
        return v;
    }

    pub fn finish(v: *Verification, ids: []const u32, accepted: usize) !void {
        defer v.snapshot.deinit();
        defer v.conv.deinit();
        if (ids.len != v.rows or accepted > ids.len) return error.BadAcceptedPrefix;
        if (accepted == ids.len) return;
        const started = if (round_times != null) now() else 0;
        if (accepted == 0) {
            try v.copy(false);
            v.cache.len = v.start;
        } else {
            try v.commit(accepted);
        }
        const restored = if (round_times != null) now() else 0;
        if (round_times) |t| {
            t.restore = restored - started;
            t.replay = now() - restored;
        }
    }

    fn commit(v: *Verification, accepted: usize) !void {
        const c = &v.model.config;
        const row_bytes: usize = c.text.local.kv * c.text.local.dim * 2;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = v.model.queue.commandBuffer();
        const blit = cb.blit();
        const rings = [_]mtl.Buffer{ v.cache.k_conv, v.cache.v_conv, v.cache.attn_conv, v.cache.mlp_conv };
        const sites = [_]prefill.ConvSite{ .key, .value, .attention, .mlp };
        var at: usize = 0;
        for (rings) |buf| {
            blit.copy(v.snapshot, at, buf, 0, buf.length());
            at += buf.length();
        }
        for (0..c.slots()) |slot| {
            if (!Cache.isLocalSlot(c, slot)) continue;
            for (0..v.rows) |row| {
                const off = v.cache.row(c, slot, v.start + row);
                for ([_]mtl.Buffer{ v.cache.keys, v.cache.values }) |buf| {
                    if (row >= accepted) blit.copy(v.snapshot, at, buf, off, row_bytes);
                    at += row_bytes;
                }
            }
        }
        for (0..c.slots()) |slot| {
            for (rings, sites) |buf, site| {
                const width = v.conv.width(site);
                for (accepted - @min(accepted, ring)..accepted) |row| {
                    const off = v.cache.convAt(slot, width) + ((v.start + row) % ring) * width * 4;
                    blit.copy(v.conv.buffer, v.conv.offset(slot, site, row), buf, off, width * 4);
                }
            }
        }
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling verify commit: {s}", .{message});
            return error.GpuFailed;
        }
        v.cache.len = v.start + accepted;
    }

    pub fn reset(v: *Verification) !void {
        try v.copy(false);
        v.cache.len = v.start;
    }

    fn copy(v: *Verification, save: bool) !void {
        const c = &v.model.config;
        const row_bytes: usize = c.text.local.kv * c.text.local.dim * 2;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = v.model.queue.commandBuffer();
        const blit = cb.blit();
        var at: usize = 0;
        for ([_]mtl.Buffer{ v.cache.k_conv, v.cache.v_conv, v.cache.attn_conv, v.cache.mlp_conv }) |buf| {
            if (save) blit.copy(buf, 0, v.snapshot, at, buf.length()) else blit.copy(v.snapshot, at, buf, 0, buf.length());
            at += buf.length();
        }
        for (0..c.slots()) |slot| {
            if (!Cache.isLocalSlot(c, slot)) continue;
            for (0..v.rows) |row| {
                const off = v.cache.row(c, slot, v.start + row);
                for ([_]mtl.Buffer{ v.cache.keys, v.cache.values }) |buf| {
                    if (save) blit.copy(buf, off, v.snapshot, at, row_bytes) else blit.copy(v.snapshot, at, buf, off, row_bytes);
                    at += row_bytes;
                }
            }
        }
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling verify snapshot: {s}", .{message});
            return error.GpuFailed;
        }
    }
};

const batch_mod = @import("batch.zig");
const state = @import("state.zig");
const Layer = @import("model.zig").Layer;
const Tensor = @import("model.zig").Tensor;

pub const group_source = @embedFile("kernels/verify.metal");

/// Rows one grouped verification carries at most: the batch step's few-row kernels, whose per-row arithmetic does not
/// depend on how many rows share the step.
pub const group_rows: usize = 8;

const DecodeParts = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, splits: u32, chunk: u32, scale: f32, floor: f32, alpha: f32 };
const BatchRow = extern struct { keys: u64, values: u64, args: DecodeParts };
const GroupRow = extern struct { keys: u64, values: u64, args: DecodeParts, first: u32, gpos: u32 };
const RelArgs = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, scale: f32, floor: f32, alpha: f32, eps: f32 };
const ConvRow = extern struct { ring: u64, pos: u32, first: u32 };
const ConvArgs = extern struct { channels: u32, heads: u32, ring: u32, pos: u32, layout: u32, norm: f32, residual: u32 };
const CommitRow = extern struct { keys: u64, values: u64, conv: [4]u64, pos: u32, capacity: u32, keep: u32, conv_keep: u32 };
const CommitSlot = extern struct { local: u32, locals_before: u32, globals_before: u32, pad: u32 = 0 };
const CommitArgs = extern struct { kvw: u32, hidden: u32, rows: u32, stride: u32, slots: u32, window: u32, ring: u32 };

const group_names = [_][:0]const u8{ "i2v_conv", "i2v_store_rel", "i2v_split", "i2v_commit_kv", "i2v_commit_conv" };
const GK = enum { conv, store_rel, split, commit_kv, commit_conv };

/// Grouped verification: one batch step whose rows are several streams' consecutive positions, the first of each the
/// stream's newest token and the rest its drafts. Every row computes what an ordinary batch step at that position
/// computes, and the forward leaves the caches as they were; `commit` then writes the accepted rows, so the caches
/// equal those of ordinary decoding of the accepted tokens.
pub const Verifier = struct {
    kernels: [group_names.len]mtl.Pipeline,
    kv: mtl.Buffer,
    conv: prefill.ConvCapture,
    slots: [64]CommitSlot,
    slot_count: usize,

    pub fn init(device: mtl.Device, config: *const @import("config.zig").Config) !Verifier {
        var v: Verifier = undefined;
        if (config.slots() > v.slots.len) return error.TooManySlots;
        const lib = try mtl.Library.fromSource(device, group_source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var made: usize = 0;
        errdefer for (v.kernels[0..made]) |p| p.deinit();
        for (group_names, 0..) |name, i| {
            v.kernels[i] = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        const kvw: usize = config.text.local.kv * config.text.local.dim;
        v.kv = try device.buffer(config.slots() * 2 * group_rows * kvw * 2, mtl.ResourceOptions.private);
        errdefer v.kv.deinit();
        v.conv = try prefill.ConvCapture.init(device, config, group_rows);
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
    }

    fn pipe(v: *const Verifier, k: GK) mtl.Pipeline {
        return v.kernels[@intFromEnum(k)];
    }

    /// The round's streams: `sizes[i]` consecutive rows for `caches[i]`, starting at its next position.
    pub fn group(v: *const Verifier, caches: []const *Cache, sizes: []const usize) !Group {
        if (caches.len == 0 or caches.len != sizes.len) return error.BadGroup;
        var g = Group{ .verifier = v, .caches = caches, .sizes = sizes, .rows = 0 };
        for (caches, sizes, 0..) |cache, size, i| {
            if (size == 0) return error.BadGroup;
            if (cache.len + size > cache.capacity) return error.ContextFull;
            for (caches[0..i]) |prior| if (prior == cache) return error.DuplicateCache;
            for (0..size) |j| {
                if (g.rows == group_rows) return error.BadGroup;
                g.owner[g.rows] = @intCast(i);
                g.pos[g.rows] = @intCast(cache.len + j);
                g.first[g.rows] = @intCast(g.rows - j);
                g.rows += 1;
            }
        }
        return g;
    }

    /// The verification forward over `ids` (every group's rows in order); the logits of every row land in
    /// `batch.rows.logits`. The caches are unchanged.
    pub fn forward(v: *const Verifier, m: *const Model, batch: *batch_mod.BatchScratch, g: *const Group, ids: []const u32) !void {
        _ = v;
        if (ids.len != g.rows or ids.len > batch.rows.chunk) return error.BadGroup;
        try prefill.groupStep(m, batch, g, ids);
    }

    /// Commit `accepted[i] + 1` rows of each group (its newest token and the drafts accepted after it) and advance its
    /// cache by that many positions.
    pub fn commit(v: *const Verifier, m: *const Model, g: *const Group, accepted: []const usize) !void {
        if (accepted.len != g.caches.len) return error.BadAcceptedPrefix;
        for (accepted, g.sizes) |a, size| if (a >= size) return error.BadAcceptedPrefix;
        const c = &m.config;
        var rows: [group_rows]CommitRow = undefined;
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
        const args = CommitArgs{ .kvw = @intCast(kvw), .hidden = c.text.hidden, .rows = @intCast(g.rows), .stride = group_rows, .slots = @intCast(v.slot_count), .window = c.text.window, .ring = state.ring };
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const enc = cb.compute(.serial);
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
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling grouped commit: {s}", .{message});
            return error.GpuFailed;
        }
        for (g.caches, accepted) |cache, a| cache.len += a + 1;
    }
};

/// One round's rows: the stream each belongs to (`owner`), its position and the chunk row its group starts at.
pub const Group = struct {
    verifier: *const Verifier,
    caches: []const *Cache,
    sizes: []const usize,
    rows: usize,
    owner: [group_rows]u8 = @splat(0),
    pos: [group_rows]u32 = @splat(0),
    first: [group_rows]u32 = @splat(0),

    /// The batch convolution for grouped rows (`i2v_conv`), capturing each row's input for the commit.
    pub fn conv(g: *const Group, enc: mtl.ComputeEncoder, x: mtl.Buffer, x_off: usize, weight: Tensor, dyn: ?mtl.Buffer, out: mtl.Buffer, out_off: usize, channels: u32, heads: u32, layout: u32, norm: f32, slot: u32, site: prefill.ConvSite) void {
        const v = g.verifier;
        var rows: [group_rows]ConvRow = undefined;
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
        enc.setBuffer(dyn orelse x, if (dyn != null) 0 else x_off, 2);
        enc.setValue(ConvArgs{ .channels = channels, .heads = heads, .ring = state.ring, .pos = 0, .layout = layout, .norm = norm, .residual = @intFromBool(site == .mlp) }, 3);
        enc.setBuffer(out, out_off, 4);
        enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 5);
        enc.setBuffer(v.conv.buffer, v.conv.offset(slot, site, 0), 6);
        enc.dispatchThreads(mtl.Size.of(channels, g.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub const AttnBuffers = struct { q: mtl.Buffer, k: mtl.Buffer, v: mtl.Buffer, r: mtl.Buffer, sink: mtl.Buffer, relbias: mtl.Buffer, out: mtl.Buffer, parts: mtl.Buffer, combine: mtl.Pipeline };

    /// The batch attention for grouped rows: keys and values into the slot's capture (`i2v_store_rel`), the split
    /// attention reading the group's own rows from it (`i2v_split`), and the batch combine.
    pub fn attend(g: *const Group, enc: mtl.ComputeEncoder, m: *const Model, rel_bias: Tensor, slot: u32, full: bool, extent: u32, bufs: AttnBuffers) void {
        const v = g.verifier;
        const config = &m.config;
        const text = &config.text;
        const heads: u32 = text.local.q;
        const kv: u32 = text.local.kv;
        const dim: u32 = text.local.dim;
        const kvw: usize = @as(usize, kv) * dim;
        const win: u32 = if (full) 0 else text.window;
        var rows: [group_rows]GroupRow = undefined;
        var plain: [group_rows]BatchRow = undefined;
        var positions: [group_rows]u32 = undefined;
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
        const kh = (@as(usize, slot) * 2 + 0) * group_rows * kvw * 2;
        const vh = (@as(usize, slot) * 2 + 1) * group_rows * kvw * 2;
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

test "grouped verification kernels compile" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const lib = try mtl.Library.fromSource(device, group_source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    for (group_names) |name| {
        const p = try mtl.Pipeline.init(device, lib, name, false);
        p.deinit();
    }
}
