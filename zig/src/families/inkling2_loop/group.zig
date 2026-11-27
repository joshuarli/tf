//! Grouped verification: streams' trees of drafts in one batch step, each row's bits its path's one-row steps'.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const Tensor = @import("model.zig").Tensor;
const Config = @import("config.zig").Config;
const state = @import("state.zig");
const Cache = state.Cache;
const prefill = @import("prefill.zig");
const batch_mod = @import("batch.zig");

pub const source = @import("kernels.zig").attention ++ "\n" ++ @embedFile("kernels/verify.metal");

/// The most rows one grouped verify takes: a stream's newest token and a tree of drafts under it.
pub const max_rows: usize = 16;
/// The most rows where not every projection is affine: past eight its batch step leaves the few-row kernels.
pub const plain_rows: usize = 8;
const none: u8 = 0xff;

const DecodeParts = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, splits: u32, chunk: u32, scale: f32, floor: f32, alpha: f32 };
const BatchRow = extern struct { keys: u64, values: u64, args: DecodeParts };
const GroupRow = extern struct { keys: u64, values: u64, args: DecodeParts, gpos: u32, path: [max_rows]u8 };
const NaxAttnTile = extern struct {
    keys: u64,
    values: u64,
    stream_first: u32,
    stream_size: u32,
    row0: u32,
    count: u32,
    gpos: u32,
    lo: u32,
    hi: u32,
    splits: u32,
    chunk: u32,
    heads: u32,
    kv: u32,
    dim: u32,
    extent: u32,
    window: u32,
    tau: u32,
    scale: f32,
    floor: f32,
    alpha: f32,
    anc_mask: [16]u32,
    depth: [32]u8,
};
const RelArgs = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, scale: f32, floor: f32, alpha: f32, eps: f32 };
const ConvRow = extern struct { ring: u64, pos: u32, start: u32, back: [4]u8 };
const ConvArgs = extern struct { channels: u32, heads: u32, ring: u32, pos: u32, layout: u32, norm: f32, residual: u32, topk: u32 };
pub const ConvInto = struct { residual: bool, topk: u32, wts: mtl.Buffer };
const CommitRow = extern struct { keys: u64, values: u64, conv: [4]u64, pos: u32, capacity: u32, keep: u32, conv_keep: u32 };
const CommitSlot = extern struct { local: u32, locals_before: u32, globals_before: u32, pad: u32 = 0 };
const CommitArgs = extern struct { kvw: u32, hidden: u32, rows: u32, stride: u32, slots: u32, window: u32, ring: u32 };
const RoundRow = extern struct { keys: u64, values: u64, conv: [4]u64, pos: u32, capacity: u32, owner: u32, depth: u32 };
const Stream = extern struct { first: u32, size: u32 };

const names = [_][:0]const u8{ "i2v_conv", "i2v_store_rel", "i2v_split", "i2v_commit_kv", "i2v_commit_conv", "i2v_argmax", "i2v_argmax_p1", "i2v_argmax_p2", "i2v_accept", "i2v_round_kv", "i2v_round_conv", "i2v_qkv" };
const Kernel = enum { conv, store_rel, split, commit_kv, commit_conv, argmax, argmax_p1, argmax_p2, accept, round_kv, round_conv, qkv };
const QkvRow = extern struct { k_ring: u64, v_ring: u64, pos: u32, start: u32, back: [4]u8, pad: u32 = 0 };
pub const QkvArgs = extern struct { kv: u32, heads: u32, dim: u32, rel: u32, extent: u32, layout: u32, dyn: u32, tau: u32, norm: f32, q_threshold: f32, rel_threshold: f32, eps: f32, floor: f32, alpha: f32 };
pub const QkvWeights = struct { k_conv: Tensor, v_conv: Tensor, q_scale: Tensor, r_scale: Tensor, rel_bias: Tensor };
pub const QkvBuffers = struct { k: mtl.Buffer, v: mtl.Buffer, kdyn: mtl.Buffer, vdyn: mtl.Buffer, q: mtl.Buffer, r: mtl.Buffer, relbias: mtl.Buffer, kc: mtl.Buffer, vc: mtl.Buffer };

pub const Finish = enum { picks, commit };
/// `last[i]`: stream i's deepest accepted row, counted from its first row; on a chain, its accepted drafts.
pub const Decided = struct { picks: []const u32, last: []const u32 };

pub const Verifier = struct {
    kernels: [names.len]mtl.Pipeline,
    kv: mtl.Buffer,
    conv: prefill.ConvCapture,
    targets: mtl.Buffer,
    argmax_v: mtl.Buffer,
    argmax_i: mtl.Buffer,
    accepted: mtl.Buffer,
    keep: mtl.Buffer,
    kept: mtl.Buffer,
    slots: [64]CommitSlot,
    slot_count: usize,
    /// The most rows a group may hold: `max_rows` when every projection is affine, else `plain_rows`.
    limit: usize,

    pub fn init(device: mtl.Device, config: *const Config) !Verifier {
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
        v.argmax_v = try device.buffer(max_rows * 64 * 4, mtl.ResourceOptions.private);
        errdefer v.argmax_v.deinit();
        v.argmax_i = try device.buffer(max_rows * 64 * 4, mtl.ResourceOptions.private);
        errdefer v.argmax_i.deinit();
        v.accepted = try device.buffer(max_rows * 4, mtl.ResourceOptions.shared);
        errdefer v.accepted.deinit();
        v.keep = try device.buffer(max_rows * 4, mtl.ResourceOptions.shared);
        errdefer v.keep.deinit();
        v.kept = try device.buffer(max_rows * 4, mtl.ResourceOptions.shared);
        v.slot_count = config.slots();
        v.limit = if (config.checkpoint == .mlx) max_rows else plain_rows;
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
        v.argmax_v.deinit();
        v.argmax_i.deinit();
        v.accepted.deinit();
        v.keep.deinit();
        v.kept.deinit();
    }

    fn pipe(v: *const Verifier, k: Kernel) mtl.Pipeline {
        return v.kernels[@intFromEnum(k)];
    }

    /// Streams of chains: stream i's `sizes[i]` rows, its newest token and then each draft after the one before.
    pub fn group(v: *const Verifier, caches: []const *Cache, sizes: []const usize) !Group {
        return v.tree(caches, sizes, null);
    }

    /// Streams of trees; `parents` concatenates each stream's row j + 1 parents (from its first row, below j + 1).
    pub fn tree(v: *const Verifier, caches: []const *Cache, sizes: []const usize, parents: ?[]const u8) !Group {
        if (caches.len == 0 or caches.len != sizes.len) return error.BadGroup;
        var g = Group{ .verifier = v, .caches = caches, .sizes = sizes, .rows = 0 };
        var given: usize = 0;
        for (caches, sizes, 0..) |cache, size, i| {
            if (size == 0) return error.BadGroup;
            if (g.rows + size > v.limit) return error.BadGroup;
            for (caches[0..i]) |prior| if (prior == cache) return error.DuplicateCache;
            const first = g.rows;
            for (0..size) |j| {
                const r = first + j;
                var depth: usize = 0;
                if (j > 0) {
                    const p: usize = if (parents) |list| list[given] else j - 1;
                    if (parents != null) given += 1;
                    if (p >= j) return error.BadDraftParents;
                    g.parent[r] = @intCast(first + p);
                    depth = g.depth[first + p] + 1;
                    @memcpy(g.path[r][0..depth], g.path[first + p][0..depth]);
                } else g.parent[r] = none;
                g.path[r][depth] = @intCast(r);
                g.depth[r] = @intCast(depth);
                g.owner[r] = @intCast(i);
                g.first[r] = @intCast(first);
                g.pos[r] = @intCast(cache.len + depth);
            }
            var deepest: usize = 0;
            for (first..first + size) |r| deepest = @max(deepest, g.depth[r]);
            if (cache.len + deepest + 1 > cache.capacity) return error.ContextFull;
            g.rows += size;
        }
        if (parents) |list| if (given != list.len) return error.BadDraftParents;
        return g;
    }

    pub fn forward(_: *const Verifier, m: *const Model, batch: *batch_mod.BatchScratch, g: *const Group, ids: []const u32) !void {
        if (ids.len != g.rows or ids.len > batch.rows.chunk) return error.BadGroup;
        try prefill.groupStep(m, batch, g, ids);
    }

    pub fn decide(v: *const Verifier, m: *const Model, batch: *batch_mod.BatchScratch, g: *const Group, ids: []const u32, finish: Finish) !Decided {
        if (ids.len != g.rows or ids.len > batch.rows.chunk) return error.BadGroup;
        try prefill.groupFinish(m, batch, g, ids, finish);
        const last = v.accepted.slice(u32, g.caches.len);
        if (finish == .commit) for (g.caches, last, 0..) |cache, end, i| {
            cache.len += g.kept(i, end);
        };
        return .{ .picks = v.targets.slice(u32, g.rows), .last = last };
    }

    fn encodeArgmax(v: *const Verifier, enc: mtl.ComputeEncoder, vocab: u32, logits: mtl.Buffer, rows: usize) void {
        std.debug.assert(rows > 0);
        std.debug.assert(rows <= max_rows);
        std.debug.assert(vocab > 0);
        enc.setPipeline(v.pipe(.argmax_p1));
        enc.setBuffer(logits, 0, 0);
        enc.setValue(vocab, 1);
        enc.setBuffer(v.argmax_v, 0, 2);
        enc.setBuffer(v.argmax_i, 0, 3);
        enc.dispatchGroups(mtl.Size.of(64, rows, 1), mtl.Size.of(256, 1, 1));
        enc.setPipeline(v.pipe(.argmax_p2));
        enc.setBuffer(logits, 0, 0);
        enc.setBuffer(v.argmax_v, 0, 1);
        enc.setBuffer(v.argmax_i, 0, 2);
        enc.setValue(vocab, 3);
        enc.setBuffer(v.targets, 0, 4);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(64, 1, 1));
    }

    pub fn encodeFinish(v: *const Verifier, enc: mtl.ComputeEncoder, m: *const Model, ids: mtl.Buffer, logits: mtl.Buffer, g: *const Group, finish: Finish) void {
        std.debug.assert(g.rows > 0);
        std.debug.assert(g.rows <= max_rows);
        std.debug.assert(g.caches.len > 0);
        std.debug.assert(g.caches.len <= max_rows);
        v.encodeArgmax(enc, m.config.text.unpadded_vocab, logits, g.rows);
        if (finish == .picks) return;
        var streams: [max_rows]Stream = undefined;
        var at: u32 = 0;
        for (g.sizes, 0..) |size, i| {
            streams[i] = .{ .first = at, .size = @intCast(size) };
            at += @intCast(size);
        }
        var parents: [max_rows]u32 = undefined;
        for (0..g.rows) |r| parents[r] = if (g.parent[r] == none) std.math.maxInt(u32) else g.parent[r];
        enc.setPipeline(v.pipe(.accept));
        enc.setBuffer(ids, 0, 0);
        enc.setBuffer(v.targets, 0, 1);
        enc.setBytes(std.mem.sliceAsBytes(streams[0..g.caches.len]), 2);
        enc.setValue(@as(u32, @intCast(g.caches.len)), 3);
        enc.setBuffer(v.accepted, 0, 4);
        enc.setBytes(std.mem.sliceAsBytes(parents[0..g.rows]), 5);
        enc.setBuffer(v.keep, 0, 6);
        enc.setBuffer(v.kept, 0, 7);
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
                .depth = g.depth[r],
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
        enc.setBuffer(v.keep, 0, 4);
        enc.dispatchThreads(mtl.Size.of(kvw, g.rows, v.slot_count * 2), mtl.Size.of(256, 1, 1));
        enc.setPipeline(v.pipe(.round_conv));
        enc.setBuffer(v.conv.buffer, 0, 0);
        enc.setValue(args, 1);
        enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 2);
        enc.setBuffer(v.keep, 0, 3);
        enc.setBuffer(v.kept, 0, 4);
        enc.dispatchThreads(mtl.Size.of(@max(kvw, c.text.hidden), g.rows, v.slot_count * 4), mtl.Size.of(256, 1, 1));
    }

    pub fn picks(v: *const Verifier, m: *const Model, logits: mtl.Buffer, rows: usize) ![]const u32 {
        if (rows == 0 or rows > max_rows) return error.BadGroup;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const enc = cb.compute(.serial);
        v.encodeArgmax(enc, m.config.text.unpadded_vocab, logits, rows);
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling grouped picks: {s}", .{message});
            return error.GpuFailed;
        }
        return v.targets.slice(u32, rows);
    }

    /// Commits each stream's path to its row `last[i]` (from its first) where decoding it would have written.
    pub fn commit(v: *const Verifier, m: *const Model, g: *const Group, last: []const usize) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const enc = cb.compute(.serial);
        try v.encodeCommit(enc, m, g, last);
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling grouped commit: {s}", .{message});
            return error.GpuFailed;
        }
        committed(g, last);
    }

    pub fn encodeCommit(v: *const Verifier, enc: mtl.ComputeEncoder, m: *const Model, g: *const Group, last: []const usize) !void {
        if (last.len != g.caches.len) return error.BadAcceptedPrefix;
        for (last, g.sizes) |a, size| if (a >= size) return error.BadAcceptedPrefix;
        const c = &m.config;
        var rows: [max_rows]CommitRow = undefined;
        for (0..g.rows) |r| {
            const i = g.owner[r];
            const cache = g.caches[i];
            const end = g.first[r] + last[i];
            const kept = g.depth[end] + 1;
            rows[r] = .{
                .keys = cache.keys.gpuAddress(),
                .values = cache.values.gpuAddress(),
                .conv = .{ cache.k_conv.gpuAddress(), cache.v_conv.gpuAddress(), cache.attn_conv.gpuAddress(), cache.mlp_conv.gpuAddress() },
                .pos = g.pos[r],
                .capacity = @intCast(cache.capacity),
                .keep = @intFromBool(g.ancestor(r, end)),
                .conv_keep = @intFromBool(g.depth[r] + state.ring >= kept),
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

pub fn committed(g: *const Group, last: []const usize) void {
    for (g.caches, last, 0..) |cache, end, i| cache.len += g.kept(i, end);
}

/// A batch scratch for verifies of up to `max_rows` rows whose every width keeps the few-row kernels (`half` off).
pub fn scratch(device: mtl.Device, config: *const Config) !batch_mod.BatchScratch {
    const capacity = max_rows;
    var rows = try prefill.PrefillScratch.init(device, config, capacity);
    errdefer rows.deinit();
    rows.half = false;
    var decode = try state.Scratch.init(device, config);
    errdefer decode.deinit();
    const heads = @max(config.text.local.q, config.text.global.q);
    const dim = @max(config.text.local.dim, config.text.global.dim);
    const parts = try device.buffer(capacity * heads * state.attn_splits * (dim + 2) * 4, mtl.ResourceOptions.private);
    errdefer parts.deinit();
    const controls_input = try device.buffer(capacity * 3 * 2 * config.text.hidden * 4, mtl.ResourceOptions.private);
    errdefer controls_input.deinit();
    const controls_hidden = try device.buffer(6 * capacity * 3 * config.loop.gate_rank * 4, mtl.ResourceOptions.private);
    errdefer controls_hidden.deinit();
    const lib = try mtl.Library.fromSource(device, @import("kernels.zig").attention, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const split = try mtl.Pipeline.init(device, lib, "i2_batch_split", false);
    errdefer split.deinit();
    const combine = try mtl.Pipeline.init(device, lib, "i2_batch_combine", false);
    errdefer combine.deinit();
    const combine_sig_mul = try mtl.Pipeline.init(device, lib, "i2_batch_combine_sig_mul", false);
    errdefer combine_sig_mul.deinit();
    const controls_norm = try mtl.Pipeline.init(device, lib, "i2_controls_norm", false);
    errdefer controls_norm.deinit();
    const controls_down = try mtl.Pipeline.init(device, lib, "i2_controls_down", false);
    errdefer controls_down.deinit();
    const controls_down_bf16 = try mtl.Pipeline.init(device, lib, "i2_controls_down_bf16", false);
    errdefer controls_down_bf16.deinit();
    const controls_apply = try mtl.Pipeline.init(device, lib, "i2_controls_apply", false);
    errdefer controls_apply.deinit();
    const controls_apply_bf16 = try mtl.Pipeline.init(device, lib, "i2_controls_apply_bf16", false);
    errdefer controls_apply_bf16.deinit();
    const mix = try mtl.Pipeline.init(device, lib, "i2_batch_mix", false);
    errdefer mix.deinit();
    const mix4 = try mtl.Pipeline.init(device, lib, "i2_batch_mix4", false);
    errdefer mix4.deinit();
    const qkv = try mtl.Pipeline.init(device, lib, "i2_batch_qkv", false);
    errdefer qkv.deinit();
    const controls_cols = try mtl.Pipeline.init(device, lib, "i2_controls_apply_rows", false);
    errdefer controls_cols.deinit();
    const controls_cols_bf16 = try mtl.Pipeline.init(device, lib, "i2_controls_apply_rows_bf16", false);
    errdefer controls_cols_bf16.deinit();
    const shared = try @import("rows.zig").Pipes.init(device);
    return .{ .rows = rows, .decode = decode, .parts = parts, .split = split, .combine = combine, .combine_sig_mul = combine_sig_mul, .controls_norm = controls_norm, .controls_down = controls_down, .controls_down_bf16 = controls_down_bf16, .controls_apply = controls_apply, .controls_apply_bf16 = controls_apply_bf16, .mix = mix, .mix4 = mix4, .qkv = qkv, .controls_cols = controls_cols, .controls_cols_bf16 = controls_cols_bf16, .shared = shared, .controls_input = controls_input, .controls_hidden = controls_hidden };
}

pub const Group = struct {
    verifier: *const Verifier,
    caches: []const *Cache,
    sizes: []const usize,
    rows: usize,
    owner: [max_rows]u8 = @splat(0),
    pos: [max_rows]u32 = @splat(0),
    first: [max_rows]u32 = @splat(0),
    depth: [max_rows]u8 = @splat(0),
    parent: [max_rows]u8 = @splat(none),
    /// `path[r][d]`: row r's ancestor at depth d, and r itself at its own depth.
    path: [max_rows][max_rows]u8 = undefined,

    /// Whether row `a` lies on the path from its stream's first row to row `r`.
    pub fn ancestor(g: *const Group, a: usize, r: usize) bool {
        return g.owner[a] == g.owner[r] and g.depth[a] <= g.depth[r] and g.path[r][g.depth[a]] == a;
    }

    /// Rows a commit of stream `i` to its row `last` (counted from its first) keeps: the path's length.
    pub fn kept(g: *const Group, i: usize, last: usize) usize {
        var first: usize = 0;
        for (g.sizes[0..i]) |size| first += size;
        return @as(usize, g.depth[first + last]) + 1;
    }

    /// The rows from stream `i`'s first row to its row `last`, counted from the first, in depth order.
    pub fn line(g: *const Group, i: usize, last: usize, out: []usize) []usize {
        var first: usize = 0;
        for (g.sizes[0..i]) |size| first += size;
        const end = first + last;
        const n = @as(usize, g.depth[end]) + 1;
        for (out[0..n], g.path[end][0..n]) |*o, r| o.* = r - first;
        return out[0..n];
    }

    pub fn conv(g: *const Group, enc: mtl.ComputeEncoder, x: mtl.Buffer, x_off: usize, weight: Tensor, dyn: ?mtl.Buffer, dyn_off: usize, out: mtl.Buffer, out_off: usize, channels: u32, heads: u32, layout: u32, norm: f32, slot: u32, site: prefill.ConvSite, into: ConvInto) void {
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
            var back: [4]u8 = @splat(none);
            const depth = g.depth[r];
            for (1..state.ring) |k| if (k <= depth) {
                back[k - 1] = g.path[r][depth - k];
            };
            rows[r] = .{ .ring = buf.gpuAddress() + cache.convAt(slot, channels), .pos = g.pos[r], .start = @intCast(cache.len), .back = back };
            enc.useResource(buf, mtl.ResourceUsage.read);
        }
        enc.setPipeline(v.pipe(.conv));
        enc.setBuffer(x, x_off, 0);
        enc.setBuffer(weight.buffer, weight.offset, 1);
        enc.setBuffer(dyn orelse x, if (dyn != null) dyn_off else x_off, 2);
        enc.setValue(ConvArgs{ .channels = channels, .heads = heads, .ring = state.ring, .pos = 0, .layout = layout, .norm = norm, .residual = @intFromBool(into.residual), .topk = into.topk }, 3);
        enc.setBuffer(out, out_off, 4);
        enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 5);
        enc.setBuffer(v.conv.buffer, v.conv.offset(slot, site, 0), 6);
        if (into.topk > 0) enc.setBuffer(into.wts, 0, 7) else enc.setBuffer(x, x_off, 7);
        const vec4 = (channels & 3) == 0 and (heads == 0 or ((channels / heads) & 3) == 0);
        enc.dispatchThreads(mtl.Size.of(if (vec4) channels / 4 else channels, g.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn qkv(g: *const Group, enc: mtl.ComputeEncoder, args: QkvArgs, w: QkvWeights, bufs: QkvBuffers, slot: u32) void {
        const v = g.verifier;
        const kvw: usize = @as(usize, args.kv) * args.dim;
        var rows: [max_rows]QkvRow = undefined;
        for (0..g.rows) |r| {
            const cache = g.caches[g.owner[r]];
            var back: [4]u8 = @splat(none);
            const depth = g.depth[r];
            for (1..state.ring) |k| if (k <= depth) {
                back[k - 1] = g.path[r][depth - k];
            };
            rows[r] = .{ .k_ring = cache.k_conv.gpuAddress() + cache.convAt(slot, kvw), .v_ring = cache.v_conv.gpuAddress() + cache.convAt(slot, kvw), .pos = g.pos[r], .start = @intCast(cache.len), .back = back };
            enc.useResource(cache.k_conv, mtl.ResourceUsage.read);
            enc.useResource(cache.v_conv, mtl.ResourceUsage.read);
        }
        enc.setPipeline(v.pipe(.qkv));
        for ([_]mtl.Buffer{ bufs.k, bufs.v }, 0..) |buf, i| enc.setBuffer(buf, 0, i);
        for ([_]Tensor{ w.k_conv, w.v_conv }, 2..) |t, i| enc.setBuffer(t.buffer, t.offset, i);
        enc.setBuffer(bufs.kdyn, 0, 4);
        enc.setBuffer(bufs.vdyn, 0, 5);
        enc.setBuffer(bufs.q, 0, 6);
        enc.setBuffer(w.q_scale.buffer, w.q_scale.offset, 7);
        enc.setBuffer(bufs.r, 0, 8);
        enc.setBuffer(w.r_scale.buffer, w.r_scale.offset, 9);
        enc.setBuffer(w.rel_bias.buffer, w.rel_bias.offset, 10);
        enc.setValue(args, 11);
        enc.setBuffer(bufs.relbias, 0, 12);
        enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 13);
        enc.setBuffer(bufs.kc, 0, 14);
        enc.setBuffer(bufs.vc, 0, 15);
        enc.setBuffer(v.conv.buffer, v.conv.offset(slot, .key, 0), 16);
        enc.setBuffer(v.conv.buffer, v.conv.offset(slot, .value, 0), 17);
        enc.setBuffer(v.kv, (@as(usize, slot) * 2 + 0) * max_rows * kvw * 2, 18);
        enc.setBuffer(v.kv, (@as(usize, slot) * 2 + 1) * max_rows * kvw * 2, 19);
        enc.dispatchGroups(mtl.Size.of(args.kv + args.heads + (args.heads * (args.extent + 1) + 255) / 256, g.rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// Combine stage after split attention: `plain` expects `i2_batch_combine`, `gated` expects `i2_batch_combine_sig_mul`.
    pub const AttnCombine = union(enum) {
        plain: mtl.Pipeline,
        gated: struct { pipe: mtl.Pipeline, gate: mtl.Buffer },
    };

    pub const AttnBuffers = struct { q: mtl.Buffer, k: mtl.Buffer, v: mtl.Buffer, r: mtl.Buffer, sink: mtl.Buffer, relbias: mtl.Buffer, out: mtl.Buffer, parts: mtl.Buffer, combine: AttnCombine, nax_split: ?mtl.Pipeline = null };

    pub fn attend(g: *const Group, stores: ?mtl.ComputeEncoder, enc: mtl.ComputeEncoder, m: *const Model, rel_bias: Tensor, slot: u32, full: bool, extent: u32, bufs: AttnBuffers) void {
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
            rows[r] = .{ .keys = cache.keys.gpuAddress() + base, .values = cache.values.gpuAddress() + base, .args = parts, .gpos = @intCast(cache.len), .path = g.path[r] };
            plain[r] = .{ .keys = rows[r].keys, .values = rows[r].values, .args = parts };
            positions[r] = pos;
            max_splits = @max(max_splits, part.splits);
            enc.useResource(cache.keys, mtl.ResourceUsage.read);
            enc.useResource(cache.values, mtl.ResourceUsage.read);
        }
        const kh = (@as(usize, slot) * 2 + 0) * max_rows * kvw * 2;
        const vh = (@as(usize, slot) * 2 + 1) * max_rows * kvw * 2;
        if (stores) |store| {
            store.setPipeline(v.pipe(.store_rel));
            store.setBuffer(bufs.k, 0, 0);
            store.setBuffer(bufs.v, 0, 1);
            store.setBuffer(bufs.r, 0, 2);
            store.setBuffer(rel_bias.buffer, rel_bias.offset, 3);
            store.setValue(RelArgs{ .heads = heads, .kv = kv, .dim = dim, .rel = text.d_rel, .extent = extent, .window = 0, .pos = 0, .tau = @intFromBool(full), .scale = 0, .floor = @floatFromInt(text.log_floor), .alpha = text.log_alpha, .eps = text.eps }, 4);
            store.setBuffer(bufs.relbias, 0, 5);
            store.setBytes(std.mem.sliceAsBytes(positions[0..g.rows]), 6);
            store.setBuffer(v.kv, kh, 7);
            store.setBuffer(v.kv, vh, 8);
            store.dispatchGroups(mtl.Size.of(kv + (heads * (extent + 1) + 255) / 256, g.rows, 1), mtl.Size.of(256, 1, 1));
        }
        var max_stream: usize = 0;
        var max_cached: u32 = 0;
        for (g.caches, g.sizes) |cache, sz| {
            max_stream = @max(max_stream, sz);
            const gpos: u32 = @intCast(cache.len);
            const lo_min: u32 = if (win != 0 and gpos + 1 > win) gpos + 1 - win else 0;
            max_cached = @max(max_cached, if (gpos > lo_min) gpos - lo_min else 0);
        }
        const use_nax_split = bufs.nax_split != null and heads == 24 and kv == 8 and dim == 128 and g.rows <= 32 and
            (max_stream >= 5 or (max_stream >= 3 and max_cached >= 2048));
        if (use_nax_split) {
            var tiles: [20]NaxAttnTile = undefined;
            var n_tiles: usize = 0;
            var tile_max_splits: u32 = 1;
            var stream_first: usize = 0;
            for (g.caches, g.sizes) |cache, size| {
                const gpos: u32 = @intCast(cache.len);
                const base = cache.row(config, slot, 0);
                const keys_addr = cache.keys.gpuAddress() + base;
                const values_addr = cache.values.gpuAddress() + base;
                var depth_arr: [32]u8 = @splat(0);
                var max_depth: u32 = 0;
                for (0..size) |j| {
                    const d = g.depth[stream_first + j];
                    depth_arr[j] = d;
                    max_depth = @max(max_depth, d);
                }
                const lo_min: u32 = if (win != 0 and gpos + 1 > win) gpos + 1 - win else 0;
                const hi_max: u32 = gpos + max_depth + 1;
                const part = state.attnPartition(hi_max - lo_min);
                const cached_len: u32 = if (gpos > lo_min) gpos - lo_min else 0;
                const target_splits: u32 = if (g.caches.len == 1 and cached_len >= 512 and cached_len <= 2048) 10 else part.splits;
                const raw_chunk: u32 = if (cached_len > 0) (cached_len + target_splits - 1) / target_splits else 0;
                const chunk32: u32 = if (raw_chunk > 0) ((raw_chunk + 31) / 32) * 32 else 0;
                const actual_splits: u32 = if (chunk32 > 0) @max(1, (cached_len + chunk32 - 1) / chunk32) else 1;
                tile_max_splits = @max(tile_max_splits, actual_splits);
                for (0..size) |j| plain[stream_first + j].args.splits = actual_splits;
                var off: usize = 0;
                while (off < size) : (off += 16) {
                    const cnt = @min(16, size - off);
                    var anc_mask: [16]u32 = @splat(0);
                    for (0..cnt) |lr| {
                        const r = stream_first + off + lr;
                        var mask: u32 = 0;
                        const d_len: usize = @as(usize, g.depth[r]) + 1;
                        for (0..d_len) |di| {
                            const anc_r = g.path[r][di];
                            mask |= @as(u32, 1) << @intCast(anc_r - stream_first);
                        }
                        anc_mask[lr] = mask;
                    }
                    tiles[n_tiles] = .{
                        .keys = keys_addr,
                        .values = values_addr,
                        .stream_first = @intCast(stream_first),
                        .stream_size = @intCast(size),
                        .row0 = @intCast(stream_first + off),
                        .count = @intCast(cnt),
                        .gpos = gpos,
                        .lo = lo_min,
                        .hi = hi_max,
                        .splits = actual_splits,
                        .chunk = chunk32,
                        .heads = heads,
                        .kv = kv,
                        .dim = dim,
                        .extent = extent,
                        .window = win,
                        .tau = @intFromBool(full),
                        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim))),
                        .floor = @floatFromInt(text.log_floor),
                        .alpha = text.log_alpha,
                        .anc_mask = anc_mask,
                        .depth = depth_arr,
                    };
                    n_tiles += 1;
                }
                stream_first += size;
            }
            enc.setPipeline(bufs.nax_split.?);
            enc.setBuffer(bufs.q, 0, 0);
            enc.setBuffer(bufs.relbias, 0, 1);
            enc.setBuffer(bufs.sink, 0, 2);
            enc.setBytes(std.mem.sliceAsBytes(tiles[0..n_tiles]), 3);
            enc.setBuffer(bufs.parts, 0, 4);
            enc.setBuffer(v.kv, kh, 5);
            enc.setBuffer(v.kv, vh, 6);
            enc.dispatchGroups(mtl.Size.of(kv, tile_max_splits, n_tiles), mtl.Size.of(96, 1, 1));
        } else {
            enc.setPipeline(v.pipe(.split));
            enc.setBuffer(bufs.q, 0, 0);
            enc.setBuffer(bufs.relbias, 0, 1);
            enc.setBuffer(bufs.sink, 0, 2);
            enc.setBytes(std.mem.sliceAsBytes(rows[0..g.rows]), 3);
            enc.setBuffer(bufs.parts, 0, 4);
            enc.setBuffer(v.kv, kh, 5);
            enc.setBuffer(v.kv, vh, 6);
            enc.dispatchGroups(mtl.Size.of(kv * max_splits, g.rows, 1), mtl.Size.of((heads / kv) * 32, 1, 1));
        }
        switch (bufs.combine) {
            .plain => |pipe| {
                enc.setPipeline(pipe);
                enc.setBuffer(bufs.parts, 0, 0);
                enc.setBytes(std.mem.sliceAsBytes(plain[0..g.rows]), 1);
                enc.setBuffer(bufs.out, 0, 2);
            },
            .gated => |gated| {
                enc.setPipeline(gated.pipe);
                enc.setBuffer(bufs.parts, 0, 0);
                enc.setBytes(std.mem.sliceAsBytes(plain[0..g.rows]), 1);
                enc.setBuffer(gated.gate, 0, 2);
                enc.setBuffer(bufs.out, 0, 3);
            },
        }
        enc.dispatchGroups(mtl.Size.of(heads, g.rows, 1), mtl.Size.of(32, 1, 1));
    }
};

pub fn acceptedDrafts(drafts: []const u32, targets: []const u32) usize {
    var n: usize = 0;
    while (n < drafts.len and n < targets.len and drafts[n] == targets[n]) n += 1;
    return n;
}

/// The deepest accepted row of one tree, as `i2v_accept` walks it; `parents[j]` is row j + 1's parent.
pub fn walk(parents: []const u8, ids: []const u32, picks: []const u32) usize {
    var at: usize = 0;
    while (true) {
        var next: ?usize = null;
        for (at + 1..ids.len) |c| if (parents[c - 1] == at and ids[c] == picks[at]) {
            next = c;
            break;
        };
        at = next orelse return at;
    }
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

test "a tree's accepted path follows the child whose token is its parent's pick, the lowest-numbered on a tie" {
    const parents = [_]u8{ 0, 0, 1, 1, 3, 0, 4 };
    const ids = [_]u32{ 9, 5, 6, 7, 8, 1, 6, 2 };
    try std.testing.expectEqual(@as(usize, 4), walk(&parents, &ids, &.{ 5, 8, 0, 0, 0, 0, 0, 0 }));
    try std.testing.expectEqual(@as(usize, 7), walk(&parents, &ids, &.{ 5, 8, 0, 0, 2, 0, 0, 0 }));
    try std.testing.expectEqual(@as(usize, 2), walk(&parents, &ids, &.{ 6, 0, 0, 0, 0, 0, 0, 0 }));
    try std.testing.expectEqual(@as(usize, 0), walk(&parents, &ids, &.{ 3, 0, 0, 0, 0, 0, 0, 0 }));
    const chain = [_]u8{ 0, 1, 2 };
    const drafts = [_]u32{ 4, 5, 6, 7 };
    for (0..4) |n| {
        var picks = [_]u32{ 5, 6, 7, 8 };
        if (n < 3) picks[n] = 0;
        try std.testing.expectEqual(acceptedDrafts(drafts[1..], &picks), walk(&chain, &drafts, &picks));
    }
}

fn scalarArgmax(values: []const f32) u32 {
    var best: usize = 0;
    for (values[1..], 1..) |v, i| if (v > values[best]) {
        best = i;
    };
    return @intCast(best);
}

test "two-stage gpu argmax matches host scan on boundary vocabs, ties, infinities, and NaNs" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const p1 = try mtl.Pipeline.init(device, lib, "i2v_argmax_p1", false);
    defer p1.deinit();
    const p2 = try mtl.Pipeline.init(device, lib, "i2v_argmax_p2", false);
    defer p2.deinit();
    const pv = try device.buffer(max_rows * 64 * 4, mtl.ResourceOptions.private);
    defer pv.deinit();
    const pi = try device.buffer(max_rows * 64 * 4, mtl.ResourceOptions.private);
    defer pi.deinit();
    const out = try device.buffer(max_rows * 4, mtl.ResourceOptions.shared);
    defer out.deinit();
    var prng = std.Random.DefaultPrng.init(0x6172_676d);
    const rng = prng.random();
    const nan = std.math.nan(f32);
    const neg_inf = -std.math.inf(f32);
    for ([_]u32{ 1, 31, 64, 16384, 16385, 32771 }) |vocab| {
        const logits = try device.buffer(max_rows * @as(usize, vocab) * 4, mtl.ResourceOptions.shared);
        defer logits.deinit();
        for ([_]usize{ 1, 7, max_rows }) |rows| {
            const slice = logits.slice(f32, rows * vocab);
            for (slice) |*v| v.* = rng.float(f32) * 20 - 10;
            if (vocab > 1) {
                slice[0] = 50.0;
                slice[vocab - 1] = 50.0;
            }
            if (rows > 1) {
                const r1 = slice[vocab..][0..vocab];
                @memset(r1, neg_inf);
                if (vocab > 2) r1[vocab / 2] = nan;
            }
            if (rows > 2) {
                const r2 = slice[2 * vocab ..][0..vocab];
                r2[0] = nan;
                if (vocab > 1) r2[vocab - 1] = 999.0;
            }
            if (rows > 3 and vocab > 5) {
                const r3 = slice[3 * vocab ..][0..vocab];
                r3[2] = 100.0;
                r3[vocab - 2] = 100.0;
                r3[vocab / 2] = nan;
            }
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(p1);
            enc.setBuffer(logits, 0, 0);
            enc.setValue(vocab, 1);
            enc.setBuffer(pv, 0, 2);
            enc.setBuffer(pi, 0, 3);
            enc.dispatchGroups(mtl.Size.of(64, rows, 1), mtl.Size.of(256, 1, 1));
            enc.setPipeline(p2);
            enc.setBuffer(logits, 0, 0);
            enc.setBuffer(pv, 0, 1);
            enc.setBuffer(pi, 0, 2);
            enc.setValue(vocab, 3);
            enc.setBuffer(out, 0, 4);
            enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(64, 1, 1));
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.GpuFailed;
            const got = out.slice(u32, rows);
            for (0..rows) |r| {
                const expected = scalarArgmax(slice[r * vocab ..][0..vocab]);
                try std.testing.expectEqual(expected, got[r]);
            }
        }
    }
}