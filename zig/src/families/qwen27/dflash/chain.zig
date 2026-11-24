//! A draft encoded before the verify it follows returns: filled with the verify's path and committed as it returns.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const op = @import("operators.zig");
const execution = @import("execution.zig");
const selector = @import("selector.zig");
const Model = @import("runtime_model.zig").Model;
const Frame = @import("runtime_frame.zig").Frame;
const Backend = @import("runtime_backend.zig").Backend;
pub const Ref = @import("runtime_backend.zig").Ref;
const context_mod = @import("runtime_context.zig");

/// The most path rows a chained draft absorbs: a grouped verify's rows.
pub const max_rows = 16;

const Params = extern struct { count: u32, context: u32, begin: u32, own: u32, path: [max_rows]u32 };
const Rows = extern struct { base: u64 = 0, width: u32, rows: u32 = 0, window: u32 = 0, block: u32 = 0, keys: u32 = 0, pad: u32 = 0 };

const names = [_][:0]const u8{ "dfc_gather", "dfc_append", "dfc_context", "dfc_own" };
const Kernel = enum { gather, append, context, own };

/// A chained block's inputs while it encodes: the embed, rope and attention callbacks read these buffers.
pub const Follow = struct {
    chain: *const Chain,
    ids: Ref,
    positions: Ref,
    mask: Ref,
    params: Ref,
    /// Keys the attention reads: the context with every path row kept, then the block.
    keys: u32,
    /// The ring slot an unwrapped context starts at for every path length; null: each layer merges its keys.
    at: ?u32,
    split: bool,
};

/// An encoded draft: the drafter's context it follows, the buffers `fill` writes and the views `land` reads.
pub const Ticket = struct {
    slot: u1,
    context: op.Context,
    next: op.Context = undefined,
    rows: u32,
    keys: u32,
    at: ?u32,
    params: Ref,
    ids: Ref,
    positions: Ref,
    mask: Ref,
    head: op.Head,
    projected: op.View,
    topk: op.View,
    anchor: u32 = 0,
    filled: bool = false,
};

/// Two frames, so one draft encodes while the other runs.
pub const Chain = struct {
    pipes: [names.len]mtl.Pipeline,
    frames: [2]Frame,
    scratch: []u8,
    allocator: std.mem.Allocator,

    pub fn init(m: *Model) !Chain {
        const device = m.backend.target.device;
        const c = m.graph.config;
        const lib = try mtl.Library.fromSource(device, @embedFile("chain.metal"), mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Chain = .{ .pipes = undefined, .frames = undefined, .scratch = &.{}, .allocator = m.allocator };
        var made: usize = 0;
        errdefer for (out.pipes[0..made]) |p| p.deinit();
        for (&out.pipes, names) |*p, name| {
            p.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        out.scratch = try m.allocator.alloc(u8, @as(usize, m.tree_block) * (c.window + m.tree_block));
        for (&out.frames) |*f| {
            f.* = .{ .allocator = m.allocator, .device = device, .queue = m.backend.target.queue, .whole = true };
            f.setResident(m.backend.residency);
        }
        return out;
    }

    pub fn deinit(c: *Chain) void {
        for (&c.frames) |*f| f.deinit();
        c.allocator.free(c.scratch);
        for (c.pipes) |p| p.deinit();
    }

    fn pipe(c: *const Chain, k: Kernel) mtl.Pipeline {
        return c.pipes[@intFromEnum(k)];
    }

    /// Up to `rows` path rows of `taps` absorbed onto `context`, then the block; null if attention would vary.
    pub fn encode(c: *Chain, m: *Model, context: op.Context, taps: Ref, rows: u32) !?Ticket {
        const b = &m.backend;
        const cf = m.graph.config;
        const block = m.tree_block;
        if (rows == 0 or rows > max_rows or b.failed or b.follow != null or block < 2 or block > 16) return error.BadDraftOptions;
        if (context.begin > context.end or context.end - context.begin > cf.window - 1) return error.BadDraftContext;
        const shortest = after(context, 1, cf.window);
        const longest = after(context, rows, cf.window);
        const least: u32 = @intCast(shortest.end - shortest.begin + block);
        const keys: u32 = @intCast(longest.end - longest.begin + block);
        if (least <= context_mod.split_keys and keys > context_mod.split_keys) return null;
        const slot: u1 = if (!c.frames[0].active) 0 else if (!c.frames[1].active) 1 else return error.DraftFrameBusy;
        const start: u32 = @intCast(context.begin % cf.window);
        const at: ?u32 = if (longest.begin == context.begin and start + keys - block <= cf.window and start + keys <= cf.window + 16) start else null;
        std.mem.swap(Frame, &b.frame, &c.frames[slot]);
        defer std.mem.swap(Frame, &b.frame, &c.frames[slot]);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        return c.encodeIn(m, context, taps, rows, slot, keys, at, least > context_mod.split_keys) catch |err| {
            b.follow = null;
            b.frame.release();
            return err;
        };
    }

    fn encodeIn(c: *const Chain, m: *Model, context: op.Context, taps: Ref, rows: u32, slot: u1, keys: u32, at: ?u32, split: bool) !Ticket {
        const b = &m.backend;
        const g = &m.graph;
        const cf = g.config;
        const block = m.tree_block;
        const ops = b.ops();
        const frame = try ops.vtable.begin(ops.ptr);
        const params = try b.frame.ref(try b.frame.alloc(1, @sizeOf(Params), .u8));
        const ids = try b.frame.ref(try b.frame.alloc(block, 1, .u32));
        const positions = try b.frame.ref(try b.frame.alloc(block, 1, .u64));
        const mask = try b.frame.ref(try b.frame.alloc(block, keys, .u8));
        const staged = try b.frame.alloc(rows, cf.tapWidth(), .bf16);
        const e = try b.encoder(frame);
        const to = try b.frame.ref(staged);
        const width = cf.tapWidth() * 2 / 16;
        e.setPipeline(c.pipe(.gather));
        e.setBuffer(taps.buf, taps.off, 0);
        e.setBuffer(params.buf, params.off, 1);
        e.setValue(Rows{ .width = width, .rows = rows }, 2);
        e.setBuffer(to.buf, to.off, 3);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
        const follow = Follow{ .chain = c, .ids = ids, .positions = positions, .mask = mask, .params = params, .keys = keys, .at = at, .split = split };
        var absorbed: [max_rows]u64 = undefined;
        for (absorbed[0..rows], 0..) |*p, i| p.* = context.end + i;
        try execution.absorb(ops, g, m.preparation, staged, absorbed[0..rows], frame, &follow);
        b.follow = &follow;
        const output = try execution.encodeStreams(ops, g, m.preparation, &.{.{ .context = after(context, rows, cf.window), .anchor = cf.mask }}, frame, block, &.{});
        b.follow = null;
        const topk = try b.frame.alloc(output.head.logits.rows, @sizeOf(core.bf16_topk.Row), .u8);
        _ = try b.encoder(frame);
        const logits = try b.frame.ref(output.head.logits);
        const picked = try b.frame.ref(topk);
        try b.topk.encode(b.frame.encoder, .{ .buf = logits.buf, .off = logits.off }, .{ .buf = picked.buf, .off = picked.off }, .{ .rows = output.head.logits.rows, .vocab = output.head.logits.width, .k = cf.topk, .stride = output.head.logits.stride });
        b.frame.encoder.barrier();
        try b.frame.seal();
        return .{ .slot = slot, .context = context, .rows = rows, .keys = keys, .at = at, .params = params, .ids = ids, .positions = positions, .mask = mask, .head = output.head, .projected = output.projected, .topk = topk };
    }

    /// The verify's accepted path, as its rows in the tap buffer from the newest token, and the target's pick after it.
    pub fn fill(c: *Chain, m: *const Model, t: *Ticket, path: []const u32, anchor: u32) !void {
        const cf = m.graph.config;
        const block = m.tree_block;
        if (path.len == 0 or path.len > t.rows or anchor >= cf.vocab or c.frames[t.slot].submitted) return error.BadDraftOptions;
        const next = after(t.context, path.len, cf.window);
        const context: u32 = @intCast(next.end - next.begin);
        var p = Params{ .count = @intCast(path.len), .context = context, .begin = @intCast(next.begin % cf.window), .own = (t.at orelse 0) + context, .path = undefined };
        for (&p.path, 0..) |*row, i| row.* = path[@min(i, path.len - 1)];
        @memcpy(bytesOf(t.params, @sizeOf(Params)), std.mem.asBytes(&p));
        const ids = std.mem.bytesAsSlice(u32, bytesOf(t.ids, @as(usize, block) * 4));
        const positions = std.mem.bytesAsSlice(u64, bytesOf(t.positions, @as(usize, block) * 8));
        for (ids, positions, 0..) |*id, *at, q| {
            id.* = if (q == 0) anchor else cf.mask;
            at.* = next.end + q;
        }
        const visible = context + block;
        const scratch = c.scratch[0 .. @as(usize, block) * visible];
        try context_mod.fillMask(scratch, context, block, cf.window, cf.block);
        const mask = bytesOf(t.mask, @as(usize, block) * t.keys);
        for (0..block) |q| {
            @memcpy(mask[q * t.keys ..][0..visible], scratch[q * visible ..][0..visible]);
            @memset(mask[q * t.keys + visible .. (q + 1) * t.keys], 0);
        }
        t.next = next;
        t.anchor = anchor;
        t.filled = true;
    }

    pub fn submit(c: *Chain, t: *const Ticket) !void {
        if (!t.filled) return error.BadDraftOptions;
        try c.frames[t.slot].submit();
    }

    pub fn frameOf(c: *Chain, t: *const Ticket) *Frame {
        return &c.frames[t.slot];
    }

    /// Waits for a submitted draft: the drafter's context then holds the path, and the tree is the block's selection.
    pub fn land(c: *Chain, m: *Model, t: *const Ticket, options: selector.Options) !selector.Tree {
        const f = &c.frames[t.slot];
        defer f.release();
        try c.settle(m, t);
        var lattice = try @import("runtime_lattice.zig").read(m.allocator, f, t.topk, t.projected, m.graph.config, t.head);
        defer lattice.deinit();
        const metric = mtl.pass_metrics.phase(.selector);
        defer metric.end();
        return selector.propose(m.allocator, &m.graph, lattice, t.anchor, t.next.end + 1, options);
    }

    /// Lets a draft go unread: a submitted one is waited for and its path stays in the drafter's context.
    pub fn drop(c: *Chain, m: *Model, t: *const Ticket) void {
        const f = &c.frames[t.slot];
        defer f.release();
        if (f.submitted) c.settle(m, t) catch {};
    }

    fn settle(c: *Chain, m: *Model, t: *const Ticket) !void {
        const f = &c.frames[t.slot];
        f.land() catch |err| {
            m.backend.failed = true;
            return err;
        };
        if (m.committed_end != t.next.end or !std.meta.eql(m.session.context, t.context) or !std.meta.eql(m.backend.context, t.context)) {
            m.backend.failed = true;
            return error.BadDraftContext;
        }
        m.session.context = t.next;
        m.session.ready = true;
        m.backend.context = t.next;
    }
};

/// The drafter's context after `kept` more committed rows, as `Session.admit` advances it.
pub fn after(c: op.Context, kept: u64, window: u32) op.Context {
    const end = c.end + kept;
    return .{ .cache = c.cache, .begin = @max(c.begin, end -| (@as(u64, window) - 1)), .end = end };
}

fn bytesOf(r: Ref, n: usize) []u8 {
    return r.buf.contents()[r.off..][0..n];
}

/// A chained frame's path keys and values of `layer` into the ring, below the path's filled length only.
pub fn append(ptr: *anyopaque, epoch: u64, f: *const Follow, layer: u32, keys: op.View, values: op.View, positions: []const u64) !void {
    const b: *Backend = @ptrCast(@alignCast(ptr));
    const e = try b.encoder(epoch);
    const cf = b.config;
    if (layer >= cf.layers or positions.len == 0 or positions.len > max_rows) return error.BadDraftContext;
    try keys.check(@intCast(positions.len), cf.kvWidth(), .bf16);
    try values.check(keys.rows, keys.width, .bf16);
    if (keys.stride != keys.width or values.stride != values.width) return error.DraftOperatorShape;
    const k = try b.frame.ref(keys);
    const v = try b.frame.ref(values);
    const width = cf.kvWidth() * 2 / 16;
    e.setPipeline(f.chain.pipe(.append));
    e.setBuffer(k.buf, k.off, 0);
    e.setBuffer(v.buf, v.off, 1);
    e.setBuffer(f.params.buf, f.params.off, 2);
    e.setValue(Rows{ .base = positions[0], .width = width, .rows = @intCast(positions.len), .window = cf.window }, 3);
    e.setBuffer(b.caches[layer].keys, 0, 4);
    e.setBuffer(b.caches[layer].values, 0, 5);
    e.dispatchThreads(mtl.Size.of(width, positions.len, 1), mtl.Size.of(128, 1, 1));
    e.barrier();
}

/// The block's attention in a chained frame: the ring's rows `fill` places, then the block's own, under its mask.
pub fn attend(b: *Backend, e: mtl.ComputeEncoder, p: op.Attention, f: *const Follow) !op.View {
    const cf = b.config;
    if (p.layer >= cf.layers or p.streams.len != 0 or p.mask != .all_slots) return error.BadDraftContext;
    if (p.keys.stride != p.keys.width or p.values.stride != p.values.width) return error.DraftOperatorShape;
    const rows = p.query.rows;
    const width = cf.kvWidth() * 2 / 16;
    const cache = b.caches[p.layer];
    const own = [2]Ref{ try b.frame.ref(p.keys), try b.frame.ref(p.values) };
    const rings = [2]mtl.Buffer{ cache.keys, cache.values };
    const result = try b.frame.alloc(rows, cf.qWidth(), .bf16);
    var kv: [2]Ref = undefined;
    if (f.at) |at| {
        for (own, rings, &kv) |o, ring, *out| {
            e.setPipeline(f.chain.pipe(.own));
            e.setBuffer(o.buf, o.off, 0);
            e.setBuffer(f.params.buf, f.params.off, 1);
            e.setValue(Rows{ .width = width, .block = rows }, 2);
            e.setBuffer(ring, 0, 3);
            e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(128, 1, 1));
            out.* = .{ .buf = ring, .off = @as(usize, at) * cf.kvWidth() * 2 };
        }
    } else {
        for (own, rings, &kv) |o, ring, *out| {
            out.* = try b.frame.ref(try b.frame.alloc(f.keys, cf.kvWidth(), .bf16));
            e.setPipeline(f.chain.pipe(.context));
            e.setBuffer(ring, 0, 0);
            e.setBuffer(o.buf, o.off, 1);
            e.setBuffer(f.params.buf, f.params.off, 2);
            e.setValue(Rows{ .width = width, .window = cf.window, .block = rows, .keys = f.keys }, 3);
            e.setBuffer(out.buf, out.off, 4);
            e.dispatchThreads(mtl.Size.of(width, f.keys, 1), mtl.Size.of(128, 1, 1));
        }
    }
    e.barrier();
    const q = try b.frame.ref(p.query);
    const y = try b.frame.ref(result);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cf.head_dim)));
    const R = core.shared_attention.Ref;
    const views = [_]R{ .{ .buf = q.buf, .off = q.off }, .{ .buf = kv[0].buf, .off = kv[0].off }, .{ .buf = kv[1].buf, .off = kv[1].off }, .{ .buf = f.mask.buf, .off = f.mask.off }, .{ .buf = y.buf, .off = y.off } };
    if (f.split) try b.attention.encodeSplit(e, rows, f.keys, f.keys, scale, views[0], views[1], views[2], views[3], views[4]) else try b.attention.encode(e, rows, f.keys, f.keys, scale, views[0], views[1], views[2], views[3], views[4]);
    e.barrier();
    return result;
}

test "a chained draft's context after its path advances as the session's does" {
    const c = op.Context{ .cache = 1, .begin = 10, .end = 2050 };
    try std.testing.expectEqual(op.Context{ .cache = 1, .begin = 10, .end = 2051 }, after(c, 1, 2048));
    try std.testing.expectEqual(op.Context{ .cache = 1, .begin = 12, .end = 2059 }, after(c, 9, 2048));
    try std.testing.expectEqual(op.Context{ .cache = 1, .begin = 0, .end = 16 }, after(.{ .cache = 1, .begin = 0, .end = 0 }, 16, 2048));
}
