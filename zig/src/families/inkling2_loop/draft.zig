//! The checkpoint's MTP head on Metal. A block of `block` rows (the stream's newest token, the anchor, then the trained
//! mask embedding) attends the stream's tapped target states and itself in both directions; the normalized mask rows
//! propose through the target's unembedding plus a recurrent Markov bias, one token after another. Only the target's
//! verification decides what is emitted, so the drafter's arithmetic affects acceptance, never output.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("model.zig");
const prefill = @import("prefill.zig");

const Model = wts.Model;
const Tensor = wts.Tensor;

pub const source = @embedFile("kernels/mtp.metal");

/// Context positions one attention split walks at most, and the most splits a context takes.
pub const attn_chunk: usize = 64;
pub const max_splits: usize = 32;

/// A stream's drafter context: every draft layer's keys and values for the target positions it has seen, f16
/// `[layers][capacity][kv_heads][dim]`, and the Markov head's mixed tap of the newest one. Positions `[start, len)`
/// hold keys; `len` is the next anchor's position.
pub const Context = struct {
    keys: mtl.Buffer,
    values: mtl.Buffer,
    frnn: mtl.Buffer,
    capacity: usize,
    start: usize = 0,
    len: usize = 0,

    pub fn init(device: mtl.Device, config: *const cfg.Config, capacity: usize) !Context {
        const head = config.mtp orelse return error.NoDraftHead;
        const size = @as(usize, head.layers) * capacity * head.kv_heads * head.head_dim * 2;
        const keys = try device.buffer(@max(size, 16), mtl.ResourceOptions.private);
        errdefer keys.deinit();
        const values = try device.buffer(@max(size, 16), mtl.ResourceOptions.private);
        errdefer values.deinit();
        const frnn = try device.buffer(@as(usize, config.text.hidden) * 4, mtl.ResourceOptions.private);
        return .{ .keys = keys, .values = values, .frnn = frnn, .capacity = capacity };
    }

    pub fn deinit(c: *Context) void {
        c.keys.deinit();
        c.values.deinit();
        c.frnn.deinit();
    }

    /// Drafter memory for one stream of `capacity` positions.
    pub fn bytes(config: *const cfg.Config, capacity: usize) usize {
        const head = config.mtp orelse return 0;
        return 2 * @as(usize, head.layers) * capacity * head.kv_heads * head.head_dim * 2 + @as(usize, config.text.hidden) * 4;
    }

    /// Forget every position: the next extend starts the context at `at`.
    pub fn restart(c: *Context, at: usize) void {
        c.start = at;
        c.len = at;
    }

    /// Positions `[0, len)` of `from`, which must hold them, become this context's.
    pub fn copyFrom(c: *Context, queue: mtl.Queue, config: *const cfg.Config, from: *const Context, len: usize) !void {
        if (from.start != 0 or from.len < len or len > c.capacity) return error.BadDraftContext;
        const head = config.mtp orelse return error.NoDraftHead;
        const row: usize = @as(usize, head.kv_heads) * head.head_dim * 2;
        if (len > 0) {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = queue.commandBuffer();
            const blit = cb.blit();
            for (0..head.layers) |l| {
                blit.copy(from.keys, l * from.capacity * row, c.keys, l * c.capacity * row, len * row);
                blit.copy(from.values, l * from.capacity * row, c.values, l * c.capacity * row, len * row);
            }
            blit.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |_| return error.GpuFailed;
        }
        c.start = 0;
        c.len = len;
    }
};

/// One tapped row of the last forward for `extend`: the context it joins, at that context's next position, unless it
/// is dropped (a rejected draft).
pub const Row = struct { ctx: *Context, keep: bool = true };

const MixArgs = extern struct { hidden: u32, rows: u32, taps: u32, tap_rows: u32, row0: u32, mixes: u32, eps: f32, norms: [4]u32, alpha: [32]f32 };
const CtxRow = extern struct { keys: u64, values: u64, pos: u32, capacity: u32, keep: u32, pad: u32 = 0 };
const CtxArgs = extern struct { kv: u32, dim: u32, rows: u32, layers: u32, eps: f32, log_theta: f32, norms: [8]u32 };
const BlockArgs = extern struct { hidden: u32, block: u32, rows: u32 };
const QkArgs = extern struct { q_heads: u32, kv_heads: u32, dim: u32, block: u32, eps: f32, log_theta: f32, q_norm: u32, k_norm: u32 };
const AttnSlot = extern struct { keys: u64, values: u64, lo: u32, hi: u32, splits: u32, chunk: u32 };
const AttnArgs = extern struct { q_heads: u32, kv_heads: u32, dim: u32, block: u32, max_splits: u32, scale: f32 };
const Count = extern struct { count: u32 };
const RnnCtxArgs = extern struct { hidden: u32, rank: u32, block: u32, steps: u32, rows: u32 };
const StepArgs = extern struct { rank: u32, width: u32, steps: u32, step: u32 };
const FoldArgs = extern struct { vocab: u32, rank: u32, slots: u32, block: u32, steps: u32, step: u32, tiles: u32, dump: u32, cap: f32 };
const PickArgs = extern struct { tiles: u32, steps: u32, step: u32 };
const BuildArgs = extern struct { vocab: u32, hidden: u32, rank: u32, bits: u32 };
const Dense = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, cap: f32 };
const AffineDense = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, bits: u32, cap: f32 };
const EmbedArgs = extern struct { width: u32, rows: u32, eps: f32 };
const RmsArgs = extern struct { width: u32, rows: u32, xs: u32, xb: u32, ys: u32, yb: u32, wb: u32, eps: f32 };

const kernel_names = [_][:0]const u8{ "i2m_mix", "i2m_ctx_store", "i2m_block", "i2m_qk", "i2m_attn_split", "i2m_attn_combine", "i2m_silu_mul", "i2m_rnn_ctx", "i2m_rnn_step", "i2m_fold", "i2m_pick", "i2m_zero", "i2m_fold_build" };
const K = enum { mix, ctx_store, block, qk, attn_split, attn_combine, silu_mul, rnn_ctx, rnn_step, fold, pick, zero, fold_build };

const Bind = struct { buf: mtl.Buffer, off: usize = 0 };

fn bt(x: Tensor) Bind {
    return .{ .buf = x.buffer, .off = x.offset };
}

fn bb(buf: mtl.Buffer) Bind {
    return .{ .buf = buf };
}

fn bo(buf: mtl.Buffer, off: usize) Bind {
    return .{ .buf = buf, .off = off };
}

/// Per-step timings of the last `propose` and `extend`, in nanoseconds, when the caller asks for them.
pub const Times = struct { extend: u64 = 0, propose: u64 = 0 };

pub const Drafter = struct {
    model: *const Model,
    pipes: *const prefill.Pipes,
    kernels: [kernel_names.len]mtl.Pipeline,
    head: cfg.Mtp,
    hidden: usize,
    vocab: usize,
    slots: usize,
    block: usize,
    rows: usize,
    taps: prefill.Taps,
    alpha: [4][8]f32,
    mixed: mtl.Buffer,
    kctx: mtl.Buffer,
    vctx: mtl.Buffer,
    rowtab: mtl.Buffer,
    h: mtl.Buffer,
    xn: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    attn: mtl.Buffer,
    o: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    anchor: mtl.Buffer,
    ids: mtl.Buffer,
    pos: mtl.Buffer,
    prev: mtl.Buffer,
    part: mtl.Buffer,
    base: mtl.Buffer,
    rnn_ctx: mtl.Buffer,
    frnn: mtl.Buffer,
    state: mtl.Buffer,
    latent: mtl.Buffer,
    fold_part: mtl.Buffer,
    drafts: mtl.Buffer,
    dump: ?mtl.Buffer = null,
    times: ?*Times = null,

    /// The drafter for up to `slots` streams a proposal and `rows` tapped rows an extend. `pipes` lends the prefill
    /// path's projection, norm and embedding pipelines; the model must have loaded its draft head.
    pub fn init(m: *Model, pipes: *const prefill.Pipes, slots: usize, rows: usize) !Drafter {
        const head = m.config.mtp orelse return error.NoDraftHead;
        if (m.draft == null) return error.NoDraftHead;
        if (slots == 0 or slots > 8 or rows == 0) return error.BadDrafter;
        if (head.rank != 256 or head.block < 2 or head.block * (head.q_heads / head.kv_heads) > 32 or head.head_dim != 128 or head.layers > 3 or head.taps_n > 8) return error.BadDraftGeometry;
        var d: Drafter = undefined;
        d.model = m;
        d.pipes = pipes;
        d.head = head;
        d.hidden = m.config.text.hidden;
        d.vocab = m.config.text.unpadded_vocab;
        d.slots = slots;
        d.block = head.block;
        d.rows = rows;
        d.dump = null;
        d.times = null;
        const lib = try mtl.Library.fromSource(m.device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var made: usize = 0;
        errdefer for (d.kernels[0..made]) |p| p.deinit();
        for (kernel_names, 0..) |name, i| {
            d.kernels[i] = try mtl.Pipeline.init(m.device, lib, name, false);
            made += 1;
        }
        const w = m.draft.?;
        const fuse = w.fuse.buffer.contents()[w.fuse.offset..][0..w.fuse.bytes];
        for (0..head.layers + 1) |mix| {
            var top: f32 = -std.math.inf(f32);
            var e: [8]f32 = @splat(0);
            for (0..head.taps_n) |t| top = @max(top, bf16At(fuse, mix * head.taps_n + t));
            var sum: f32 = 0;
            for (0..head.taps_n) |t| {
                e[t] = @exp(bf16At(fuse, mix * head.taps_n + t) - top);
                sum += e[t];
            }
            for (0..8) |t| d.alpha[@min(mix, 3)][t] = if (t < head.taps_n) roundBf16(e[t] / sum) else 0;
        }
        const h = d.hidden;
        const qw: usize = head.q_heads * head.head_dim;
        const kvw: usize = head.kv_heads * head.head_dim;
        const r = slots * d.block;
        const steps = d.block - 1;
        const tiles = (d.vocab + 31) / 32;
        const private = mtl.ResourceOptions.private;
        const shared = mtl.ResourceOptions.shared;
        const alloc = [_]struct { ptr: *mtl.Buffer, bytes: usize, opts: usize }{
            .{ .ptr = &d.taps.buffer, .bytes = @as(usize, head.taps_n) * rows * h * 4, .opts = private },
            .{ .ptr = &d.mixed, .bytes = 4 * rows * h * 4, .opts = private },
            .{ .ptr = &d.kctx, .bytes = @as(usize, head.layers) * rows * kvw * 4, .opts = private },
            .{ .ptr = &d.vctx, .bytes = @as(usize, head.layers) * rows * kvw * 4, .opts = private },
            .{ .ptr = &d.rowtab, .bytes = rows * @sizeOf(CtxRow), .opts = shared },
            .{ .ptr = &d.h, .bytes = r * h * 4, .opts = private },
            .{ .ptr = &d.xn, .bytes = r * h * 4, .opts = private },
            .{ .ptr = &d.q, .bytes = r * qw * 4, .opts = private },
            .{ .ptr = &d.k, .bytes = r * kvw * 4, .opts = private },
            .{ .ptr = &d.v, .bytes = r * kvw * 4, .opts = private },
            .{ .ptr = &d.attn, .bytes = r * qw * 4, .opts = private },
            .{ .ptr = &d.o, .bytes = r * h * 4, .opts = private },
            .{ .ptr = &d.gate, .bytes = r * @as(usize, head.inter) * 4, .opts = private },
            .{ .ptr = &d.up, .bytes = r * @as(usize, head.inter) * 4, .opts = private },
            .{ .ptr = &d.anchor, .bytes = slots * h * 4, .opts = private },
            .{ .ptr = &d.ids, .bytes = slots * 4, .opts = shared },
            .{ .ptr = &d.pos, .bytes = slots * 4, .opts = shared },
            .{ .ptr = &d.prev, .bytes = slots * 4, .opts = shared },
            .{ .ptr = &d.part, .bytes = slots * @as(usize, head.q_heads) * d.block * (max_splits + 1) * (@as(usize, head.head_dim) + 2) * 4, .opts = private },
            .{ .ptr = &d.base, .bytes = r * d.vocab * 4, .opts = private },
            .{ .ptr = &d.rnn_ctx, .bytes = slots * steps * 3 * @as(usize, head.rank) * 4, .opts = private },
            .{ .ptr = &d.frnn, .bytes = slots * h * 4, .opts = private },
            .{ .ptr = &d.state, .bytes = slots * @as(usize, head.rank) * 4, .opts = private },
            .{ .ptr = &d.latent, .bytes = slots * @as(usize, head.rank) * 4, .opts = private },
            .{ .ptr = &d.fold_part, .bytes = slots * tiles * 8, .opts = private },
            .{ .ptr = &d.drafts, .bytes = slots * steps * 4, .opts = shared },
        };
        var done: usize = 0;
        errdefer for (alloc[0..done]) |f| f.ptr.deinit();
        for (alloc) |f| {
            f.ptr.* = try m.device.buffer(@max(f.bytes, 16), f.opts);
            done += 1;
        }
        d.taps.rows = rows;
        d.taps.count = head.taps_n;
        d.taps.final = m.config.slots();
        d.taps.visits = @splat(0);
        for (head.taps[0..head.taps_n], 0..) |tap, i| d.taps.visits[i] = tap;
        if (m.draft.?.fold == null) try buildFold(m, d.kernels[@intFromEnum(K.fold_build)]);
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        for (d.kernels) |p| p.deinit();
        inline for ([_][]const u8{ "mixed", "kctx", "vctx", "rowtab", "h", "xn", "q", "k", "v", "attn", "o", "gate", "up", "anchor", "ids", "pos", "prev", "part", "base", "rnn_ctx", "frnn", "state", "latent", "fold_part", "drafts" }) |f| @field(d, f).deinit();
        d.taps.buffer.deinit();
        if (d.dump) |buf| buf.deinit();
    }

    /// The most tokens one block proposes.
    pub fn depth(d: *const Drafter) usize {
        return d.block - 1;
    }

    fn pipe(d: *const Drafter, k: K) mtl.Pipeline {
        return d.kernels[@intFromEnum(k)];
    }

    fn set(enc: mtl.ComputeEncoder, p: mtl.Pipeline, args: anytype, args_at: usize, bufs: []const Bind) void {
        enc.setPipeline(p);
        var at: usize = 0;
        for (bufs) |x| {
            if (at == args_at) at += 1;
            enc.setBuffer(x.buf, x.off, at);
            at += 1;
        }
        enc.setValue(args, args_at);
    }

    /// x [rows] @ w^T into out, f32 with w's bf16 rows: the few-row kernel up to 16 rows, the tiled one above.
    fn linear(d: *const Drafter, enc: mtl.ComputeEncoder, w: Tensor, x: Bind, rows: usize, out: Bind) void {
        const n = w.shape[0];
        const args = Dense{ .n = @intCast(n), .k = @intCast(w.shape[1]), .rows = @intCast(rows), .xs = 1, .xb = 0, .cap = 0 };
        if (rows <= 16) {
            set(enc, d.pipes.smallrow, args, 2, &.{ x, bt(w), out });
            enc.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        } else {
            set(enc, d.pipes.gemm, args, 2, &.{ x, bt(w), out });
            enc.dispatchGroups(mtl.Size.of((n + 63) / 64, (rows + 31) / 32, 1), mtl.Size.of(256, 1, 1));
        }
    }

    fn rms(d: *const Drafter, enc: mtl.ComputeEncoder, x: Bind, w: Tensor, rows: usize, out: Bind) void {
        const args = RmsArgs{ .width = @intCast(d.hidden), .rows = @intCast(rows), .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = d.model.config.text.eps };
        set(enc, d.pipes.rms, args, 2, &.{ x, bt(w), out });
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn add(d: *const Drafter, enc: mtl.ComputeEncoder, dst: Bind, src: Bind, count: usize) void {
        set(enc, d.model.pipe(.add), Count{ .count = @intCast(count) }, 2, &.{ dst, src });
        enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn copy(d: *const Drafter, enc: mtl.ComputeEncoder, src: Bind, dst: Bind, count: usize) void {
        set(enc, d.model.pipe(.copy), Count{ .count = @intCast(count) }, 2, &.{ src, dst });
        enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn finish(cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder, what: []const u8) !void {
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling drafter {s}: {s}", .{ what, message });
            return error.GpuFailed;
        }
    }

    /// Append the last forward's tapped rows to their contexts: row i of `d.taps` joins `rows[i].ctx` at its next
    /// position. A context's Markov tap becomes its newest kept row's.
    pub fn extend(d: *Drafter, rows: []const Row) !void {
        if (rows.len == 0) return;
        const started = if (d.times != null) now() else 0;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.model.queue.commandBuffer();
        const enc = cb.compute(.serial);
        try d.encodeExtend(enc, rows);
        try finish(cb, enc, "extend");
        if (d.times) |t| t.extend += now() - started;
    }

    /// `extend`'s dispatches, for an encoder the caller ends before the taps change; the contexts advance now.
    pub fn encodeExtend(d: *Drafter, enc: mtl.ComputeEncoder, rows: []const Row) !void {
        if (rows.len == 0) return;
        if (rows.len > d.rows) return error.TooManyRows;
        const m = d.model;
        const w = m.draft.?;
        const head = d.head;
        const h = d.hidden;
        const kvw: usize = head.kv_heads * head.head_dim;
        const table = d.rowtab.slice(CtxRow, rows.len);
        var any = false;
        for (rows, 0..) |row, i| {
            const ctx = row.ctx;
            if (row.keep and ctx.len >= ctx.capacity) return error.DraftContextFull;
            table[i] = .{ .keys = ctx.keys.gpuAddress(), .values = ctx.values.gpuAddress(), .pos = @intCast(ctx.len), .capacity = @intCast(ctx.capacity), .keep = @intFromBool(row.keep) };
            if (row.keep) {
                ctx.len += 1;
                any = true;
            }
        }
        if (!any) return;
        var mix = MixArgs{ .hidden = @intCast(h), .rows = @intCast(rows.len), .taps = head.taps_n, .tap_rows = @intCast(d.rows), .row0 = 0, .mixes = head.layers + 1, .eps = m.config.text.eps, .norms = @splat(0), .alpha = @splat(0) };
        for (0..head.layers) |l| mix.norms[l] = @intCast(w.layers[l].mix_norm.offset / 2);
        mix.norms[head.layers] = @intCast(w.markov_norm.offset / 2);
        for (0..head.layers + 1) |x| for (0..8) |t| {
            mix.alpha[x * 8 + t] = d.alpha[x][t];
        };
        set(enc, d.pipe(.mix), mix, 2, &.{ bb(d.taps.buffer), bb(w.buffer), bb(d.mixed) });
        enc.dispatchGroups(mtl.Size.of(rows.len, head.layers + 1, 1), mtl.Size.of(256, 1, 1));
        for (w.layers, 0..) |layer, l| {
            const input = bo(d.mixed, l * rows.len * h * 4);
            d.linear(enc, layer.k_ctx, input, rows.len, bo(d.kctx, l * rows.len * kvw * 4));
            d.linear(enc, layer.v_ctx, input, rows.len, bo(d.vctx, l * rows.len * kvw * 4));
        }
        var store = CtxArgs{ .kv = head.kv_heads, .dim = head.head_dim, .rows = @intCast(rows.len), .layers = head.layers, .eps = m.config.text.eps, .log_theta = std.math.log2(head.rope_theta), .norms = @splat(0) };
        for (w.layers, 0..) |layer, l| store.norms[l] = @intCast(layer.k_norm_ctx.offset / 2);
        for (rows) |row| {
            enc.useResource(row.ctx.keys, mtl.ResourceUsage.write);
            enc.useResource(row.ctx.values, mtl.ResourceUsage.write);
        }
        set(enc, d.pipe(.ctx_store), store, 3, &.{ bb(d.kctx), bb(d.vctx), bb(w.buffer), bb(d.rowtab) });
        enc.dispatchGroups(mtl.Size.of(rows.len, head.kv_heads, head.layers), mtl.Size.of(head.head_dim / 2, 1, 1));
        for (rows, 0..) |row, i| {
            if (!row.keep) continue;
            var last = true;
            for (rows[i + 1 ..]) |later| {
                if (later.keep and later.ctx == row.ctx) last = false;
            }
            if (last) d.copy(enc, bo(d.mixed, (@as(usize, head.layers) * rows.len + i) * h * 4), bb(row.ctx.frnn), h);
        }
    }

    /// Propose up to `counts[i]` tokens after `anchors[i]` for each context into `out[i]`; a count of zero drafts
    /// nothing for that stream. The anchor is the token at the context's next position.
    pub fn propose(d: *Drafter, ctxs: []const *Context, anchors: []const u32, counts: []const usize, out: []const []u32) !void {
        const n = ctxs.len;
        if (n == 0) return;
        if (n > d.slots or anchors.len != n or counts.len != n or out.len != n) return error.BadProposal;
        var steps: usize = 0;
        for (counts, out) |c, o| {
            if (c > d.depth() or o.len < c) return error.BadProposal;
            steps = @max(steps, c);
        }
        if (steps == 0) return;
        const started = if (d.times != null) now() else 0;
        const m = d.model;
        const w = m.draft.?;
        const head = d.head;
        const h = d.hidden;
        const kvw: usize = head.kv_heads * head.head_dim;
        const inter: usize = head.inter;
        const rows = n * d.block;
        const tiles = (d.vocab + 31) / 32;
        const all_steps = d.block - 1;
        @memcpy(d.ids.slice(u32, n), anchors);
        @memcpy(d.prev.slice(u32, n), anchors);
        for (ctxs, 0..) |ctx, i| d.pos.slice(u32, n)[i] = @intCast(ctx.len);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const enc = cb.compute(.serial);
        const eps = m.config.text.eps;
        if (m.weights.embed.affine) |aq| {
            set(enc, m.affine_pipes.?.embed, EmbedArgs{ .width = @intCast(h), .rows = @intCast(n), .eps = eps }, 5, &.{ bb(d.ids), bt(m.weights.embed), bo(aq.scales, aq.scale_offset), bo(aq.biases, aq.bias_offset), bt(m.weights.embed_norm), bb(d.anchor) });
        } else {
            set(enc, d.pipes.embed, EmbedArgs{ .width = @intCast(h), .rows = @intCast(n), .eps = eps }, 3, &.{ bb(d.ids), bt(m.weights.embed), bt(m.weights.embed_norm), bb(d.anchor) });
        }
        enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        set(enc, d.pipe(.block), BlockArgs{ .hidden = @intCast(h), .block = @intCast(d.block), .rows = @intCast(rows) }, 2, &.{ bb(d.anchor), bt(w.mask), bb(d.h) });
        enc.dispatchThreads(mtl.Size.of(h, rows, 1), mtl.Size.of(256, 1, 1));
        for (ctxs) |ctx| {
            enc.useResource(ctx.keys, mtl.ResourceUsage.read);
            enc.useResource(ctx.values, mtl.ResourceUsage.read);
        }
        const window: usize = head.window;
        for (w.layers, 0..) |layer, l| {
            d.rms(enc, bb(d.h), layer.input_norm, rows, bb(d.xn));
            d.linear(enc, layer.q, bb(d.xn), rows, bb(d.q));
            d.linear(enc, layer.k, bb(d.xn), rows, bb(d.k));
            d.linear(enc, layer.v, bb(d.xn), rows, bb(d.v));
            set(enc, d.pipe(.qk), QkArgs{ .q_heads = head.q_heads, .kv_heads = head.kv_heads, .dim = head.head_dim, .block = @intCast(d.block), .eps = eps, .log_theta = std.math.log2(head.rope_theta), .q_norm = @intCast(layer.q_norm.offset / 2), .k_norm = @intCast(layer.k_norm.offset / 2) }, 4, &.{ bb(d.q), bb(d.k), bb(w.buffer), bb(d.pos) });
            enc.dispatchGroups(mtl.Size.of(rows, head.q_heads + head.kv_heads, 1), mtl.Size.of(head.head_dim / 2, 1, 1));
            var table: [8]AttnSlot = undefined;
            var widest: u32 = 0;
            for (ctxs, 0..) |ctx, i| {
                const lo = if (head.local[l]) @max(ctx.start, ctx.len -| window) else ctx.start;
                const span = ctx.len - lo;
                const splits: usize = if (span == 0) 0 else @min(max_splits, (span + attn_chunk - 1) / attn_chunk);
                const chunk: usize = if (splits == 0) 0 else (span + splits - 1) / splits;
                const at = l * ctx.capacity * kvw * 2;
                table[i] = .{ .keys = ctx.keys.gpuAddress() + at, .values = ctx.values.gpuAddress() + at, .lo = @intCast(lo), .hi = @intCast(ctx.len), .splits = @intCast(splits), .chunk = @intCast(chunk) };
                widest = @max(widest, @as(u32, @intCast(splits)));
            }
            const attn_args = AttnArgs{ .q_heads = head.q_heads, .kv_heads = head.kv_heads, .dim = head.head_dim, .block = @intCast(d.block), .max_splits = widest, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head.head_dim))) };
            enc.setPipeline(d.pipe(.attn_split));
            enc.setBuffer(d.q, 0, 0);
            enc.setBuffer(d.k, 0, 1);
            enc.setBuffer(d.v, 0, 2);
            enc.setValue(attn_args, 3);
            enc.setBytes(std.mem.sliceAsBytes(table[0..n]), 4);
            enc.setBuffer(d.part, 0, 5);
            enc.dispatchGroups(mtl.Size.of(widest + 1, head.kv_heads, n), mtl.Size.of(32 * d.block * (head.q_heads / head.kv_heads), 1, 1));
            set(enc, d.pipe(.attn_combine), attn_args, 1, &.{ bb(d.part), bb(d.attn) });
            enc.dispatchGroups(mtl.Size.of(head.q_heads, rows, 1), mtl.Size.of(32, 1, 1));
            d.linear(enc, layer.o, bb(d.attn), rows, bb(d.o));
            d.add(enc, bb(d.h), bb(d.o), rows * h);
            d.rms(enc, bb(d.h), layer.post_norm, rows, bb(d.xn));
            d.linear(enc, layer.gate, bb(d.xn), rows, bb(d.gate));
            d.linear(enc, layer.up, bb(d.xn), rows, bb(d.up));
            set(enc, d.pipe(.silu_mul), Count{ .count = @intCast(rows * inter) }, 2, &.{ bb(d.gate), bb(d.up) });
            enc.dispatchThreads(mtl.Size.of(rows * inter, 1, 1), mtl.Size.of(256, 1, 1));
            d.linear(enc, layer.down, bb(d.gate), rows, bb(d.o));
            d.add(enc, bb(d.h), bb(d.o), rows * h);
        }
        d.rms(enc, bb(d.h), w.norm, rows, bb(d.xn));
        const unembed = m.weights.unembed;
        if (unembed.affine) |aq| {
            set(enc, m.affine_pipes.?.dense_rows, AffineDense{ .n = @intCast(d.vocab), .k = @intCast(h), .rows = @intCast(rows), .xs = 1, .xb = 0, .bits = aq.bits(), .cap = 0 }, 4, &.{ bb(d.xn), bt(unembed), bo(aq.scales, aq.scale_offset), bo(aq.biases, aq.bias_offset), bb(d.base) });
            enc.dispatchGroups(mtl.Size.of((d.vocab + 63) / 64, (rows + 31) / 32, 1), mtl.Size.of(256, 1, 1));
        } else {
            var head_rows = unembed;
            head_rows.shape[0] = d.vocab;
            d.linear(enc, head_rows, bb(d.xn), rows, bb(d.base));
        }
        for (ctxs, 0..) |ctx, i| d.copy(enc, bb(ctx.frnn), bo(d.frnn, i * h * 4), h);
        const rank: usize = head.rank;
        set(enc, d.pipe(.rnn_ctx), RnnCtxArgs{ .hidden = @intCast(h), .rank = head.rank, .block = @intCast(d.block), .steps = @intCast(all_steps), .rows = @intCast(n * all_steps) }, 4, &.{ bb(d.xn), bb(d.frnn), bt(w.rnn), bt(w.rnn_bias), bb(d.rnn_ctx) });
        enc.dispatchGroups(mtl.Size.of((3 * rank + 7) / 8, n * all_steps, 1), mtl.Size.of(256, 1, 1));
        set(enc, d.pipe(.zero), Count{ .count = @intCast(n * rank) }, 1, &.{bb(d.state)});
        enc.dispatchThreads(mtl.Size.of(n * rank, 1, 1), mtl.Size.of(256, 1, 1));
        const fold = w.fold.?;
        const fold_bias = w.fold_bias.?;
        const dump = d.dump orelse d.base;
        for (0..steps) |step| {
            set(enc, d.pipe(.rnn_step), StepArgs{ .rank = head.rank, .width = @intCast(2 * rank + 2 * h), .steps = @intCast(all_steps), .step = @intCast(step) }, 6, &.{ bt(w.rnn), bt(w.markov_w1), bb(d.rnn_ctx), bb(d.prev), bb(d.state), bb(d.latent) });
            enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(1024, 1, 1));
            set(enc, d.pipe(.fold), FoldArgs{ .vocab = @intCast(d.vocab), .rank = head.rank, .slots = @intCast(n), .block = @intCast(d.block), .steps = @intCast(all_steps), .step = @intCast(step), .tiles = @intCast(tiles), .dump = @intFromBool(d.dump != null), .cap = m.config.text.logit_cap }, 4, &.{ bb(d.base), bt(fold), bt(fold_bias), bb(d.latent), bb(d.fold_part), bb(dump) });
            enc.dispatchGroups(mtl.Size.of(tiles, 1, 1), mtl.Size.of(256, 1, 1));
            set(enc, d.pipe(.pick), PickArgs{ .tiles = @intCast(tiles), .steps = @intCast(all_steps), .step = @intCast(step) }, 1, &.{ bb(d.fold_part), bb(d.prev), bb(d.drafts) });
            enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(1024, 1, 1));
        }
        try finish(cb, enc, "propose");
        const drafted = d.drafts.slice(u32, n * all_steps);
        for (counts, out, 0..) |c, o, i| @memcpy(o[0..c], drafted[i * all_steps ..][0..c]);
        if (d.times) |t| t.propose += now() - started;
    }
};

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn bf16At(bytes: []const u8, i: usize) f32 {
    const bits: u32 = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
    return @bitCast(bits << 16);
}

/// Round to the nearest bf16, ties to even: the head's tap mixtures are stored at that precision by both references.
pub fn roundBf16(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    const lsb = (bits >> 16) & 1;
    const rounded = (bits +% 0x7fff +% lsb) & 0xffff0000;
    return @bitCast(rounded);
}

/// The unembedding folded through the Markov head's latent projection, built once into buffers the model frees.
fn buildFold(m: *Model, kernel: mtl.Pipeline) !void {
    const w = &m.draft.?;
    const vocab: usize = m.config.text.unpadded_vocab;
    const rank: usize = w.hidden_out.shape[1];
    const h: usize = m.config.text.hidden;
    const fold_buf = try m.ownBuffer(vocab * rank * 2);
    const bias_buf = try m.ownBuffer(vocab * 2);
    const unembed = m.weights.unembed;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(kernel);
    if (unembed.affine) |aq| {
        enc.setBuffer(unembed.buffer, unembed.offset, 0);
        enc.setBuffer(aq.scales, aq.scale_offset, 1);
        enc.setBuffer(aq.biases, aq.bias_offset, 2);
        enc.setBuffer(unembed.buffer, unembed.offset, 3);
    } else {
        if (unembed.dtype != .bf16) return error.BadTensor;
        for (0..4) |i| enc.setBuffer(unembed.buffer, unembed.offset, i);
    }
    enc.setBuffer(w.hidden_out.buffer, w.hidden_out.offset, 4);
    enc.setBuffer(w.hidden_out_bias.buffer, w.hidden_out_bias.offset, 5);
    enc.setValue(BuildArgs{ .vocab = @intCast(vocab), .hidden = @intCast(h), .rank = @intCast(rank), .bits = if (unembed.affine) |aq| aq.bits() else 0 }, 6);
    enc.setBuffer(fold_buf, 0, 7);
    enc.setBuffer(bias_buf, 0, 8);
    enc.dispatchGroups(mtl.Size.of((vocab + 7) / 8, 1, 1), mtl.Size.of(rank, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.log.err("inkling drafter fold: {s}", .{message});
        return error.GpuFailed;
    }
    w.fold = .{ .buffer = fold_buf, .offset = 0, .bytes = vocab * rank * 2, .dtype = .bf16, .rank = 2, .shape = .{ vocab, rank, 1, 1, 1 } };
    w.fold_bias = .{ .buffer = bias_buf, .offset = 0, .bytes = vocab * 2, .dtype = .bf16, .rank = 1, .shape = .{ vocab, 1, 1, 1, 1 } };
}

test "bf16 rounding is to nearest, ties to even" {
    try std.testing.expectEqual(@as(f32, 1.0), roundBf16(1.0));
    try std.testing.expectEqual(@as(f32, 1.0), roundBf16(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(f32, @bitCast(@as(u32, 0x3f820000))), roundBf16(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expectEqual(@as(f32, @bitCast(@as(u32, 0x3f810000))), roundBf16(@bitCast(@as(u32, 0x3f808001))));
}

test "the drafter's kernels compile" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    for (kernel_names) |name| {
        const p = try mtl.Pipeline.init(device, lib, name, false);
        p.deinit();
    }
}

/// The campaign's rendered chat prompts: the agent workload's first turn, then three short requests.
pub const prompts_json = @embedFile("fixtures/mtp_prompts.json");
pub const Prompt = struct { name: []const u8, token_ids: []const u32 };

fn readShared(m: *const Model, source_buf: mtl.Buffer, offset: usize, bytes: usize) !mtl.Buffer {
    const out = try m.device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared);
    errdefer out.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    blit.copy(source_buf, offset, out, 0, bytes);
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    return out;
}

fn argmaxOf(x: []const f32) u32 {
    var best: usize = 0;
    for (x[1..], 1..) |v, i| if (v > x[best]) {
        best = i;
    };
    return @intCast(best);
}

test "the drafter's proposals follow its own logits, repeat bit for bit and batch by stream" {
    const path = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return error.SkipZigTest;
    const dir = std.mem.span(path);
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const m = try Model.load(gpa, io, dir);
    defer m.deinit();
    try m.loadDraft(dir);
    var ps = try prefill.PrefillScratch.init(m.device, &m.config, 512);
    defer ps.deinit();
    var d = try Drafter.init(m, &ps.pipes, 2, 512);
    defer d.deinit();
    const parsed = try std.json.parseFromSlice([]const Prompt, gpa, prompts_json, .{});
    defer parsed.deinit();
    const prompt = parsed.value[0].token_ids;
    const vocab = m.config.text.unpadded_vocab;
    const h = m.config.text.hidden;
    const taps_n: usize = m.config.mtp.?.taps_n;
    var cache = try @import("state.zig").Cache.init(m.device, &m.config, 4096);
    defer cache.deinit();
    var one = try Context.init(m.device, &m.config, 4096);
    defer one.deinit();
    var two = try Context.init(m.device, &m.config, 4096);
    defer two.deinit();
    var at: usize = 0;
    while (at < prompt.len) {
        const take = @min(512, prompt.len - at);
        ps.taps = &d.taps;
        defer ps.taps = null;
        try prefill.prefillWant(m, &ps, &cache, prompt[at..][0..take], if (at + take == prompt.len) .last else .none);
        var rows: [512]Row = undefined;
        for (rows[0..take]) |*row| row.* = .{ .ctx = &one };
        try d.extend(rows[0..take]);
        for (rows[0..take]) |*row| row.* = .{ .ctx = &two };
        try d.extend(rows[0..take]);
        at += take;
    }
    try std.testing.expectEqual(prompt.len, one.len);
    const anchor = argmaxOf(ps.logits.slice(f32, vocab));
    const steps = d.depth();
    d.dump = try m.device.buffer(2 * steps * vocab * 4, mtl.ResourceOptions.shared);
    var first: [7]u32 = undefined;
    try d.propose(&.{&one}, &.{anchor}, &.{steps}, &.{first[0..steps]});
    const logits = try gpa.dupe(f32, d.dump.?.slice(f32, steps * vocab));
    defer gpa.free(logits);
    for (0..steps) |s| {
        const row = logits[s * vocab ..][0..vocab];
        for (row) |v| try std.testing.expect(std.math.isFinite(v));
        try std.testing.expectEqual(argmaxOf(row), first[s]);
    }
    var again: [7]u32 = undefined;
    try d.propose(&.{&one}, &.{anchor}, &.{steps}, &.{again[0..steps]});
    try std.testing.expectEqualSlices(u32, first[0..steps], again[0..steps]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(d.dump.?.slice(f32, steps * vocab)));
    var pair: [2][7]u32 = undefined;
    try d.propose(&.{ &one, &two }, &.{ anchor, anchor }, &.{ steps, 3 }, &.{ pair[0][0..steps], pair[1][0..3] });
    try std.testing.expectEqualSlices(u32, first[0..steps], pair[0][0..steps]);
    try std.testing.expectEqualSlices(u32, first[0..3], pair[1][0..3]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(d.dump.?.slice(f32, steps * vocab)));
    std.debug.print("drafter anchor={d} drafts={any}\n", .{ anchor, first[0..steps] });
    if (std.c.getenv("TF_INKLING2_MTP_DUMP")) |prefix_z| {
        const prefix = std.mem.span(prefix_z);
        for (parsed.value, 0..) |case, index| {
            var cache_n = try @import("state.zig").Cache.init(m.device, &m.config, 4096);
            defer cache_n.deinit();
            var ctx = try Context.init(m.device, &m.config, 4096);
            defer ctx.deinit();
            const ids = case.token_ids;
            const taps = try gpa.alloc(f32, ids.len * taps_n * h);
            defer gpa.free(taps);
            var pos: usize = 0;
            while (pos < ids.len) {
                const take = @min(512, ids.len - pos);
                ps.taps = &d.taps;
                defer ps.taps = null;
                try prefill.prefillWant(m, &ps, &cache_n, ids[pos..][0..take], if (pos + take == ids.len) .last else .none);
                const tapped = try readShared(m, d.taps.buffer, 0, taps_n * d.taps.rows * h * 4);
                defer tapped.deinit();
                const view = tapped.slice(f32, taps_n * d.taps.rows * h);
                for (0..take) |r| for (0..taps_n) |t| {
                    @memcpy(taps[((pos + r) * taps_n + t) * h ..][0..h], view[(t * d.taps.rows + r) * h ..][0..h]);
                };
                var rows: [512]Row = undefined;
                for (rows[0..take]) |*row| row.* = .{ .ctx = &ctx };
                try d.extend(rows[0..take]);
                pos += take;
            }
            const start = argmaxOf(ps.logits.slice(f32, vocab));
            var drafts: [7]u32 = undefined;
            try d.propose(&.{&ctx}, &.{start}, &.{steps}, &.{drafts[0..steps]});
            var name: [512]u8 = undefined;
            const stem = if (index == 0) prefix else try std.fmt.bufPrint(&name, "{s}-{s}", .{ prefix, case.name });
            var file: [600]u8 = undefined;
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&file, "{s}.taps.f32", .{stem}), .data = std.mem.sliceAsBytes(taps) });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&file, "{s}.logits.f32", .{stem}), .data = std.mem.sliceAsBytes(d.dump.?.slice(f32, steps * vocab)) });
            var meta: [512]u8 = undefined;
            var list: [128]u8 = undefined;
            var n: usize = 0;
            for (drafts[0..steps], 0..) |token, i| n += (try std.fmt.bufPrint(list[n..], "{s}{d}", .{ if (i == 0) "" else ",", token })).len;
            const text = try std.fmt.bufPrint(&meta, "{{\"positions\":{d},\"anchor\":{d},\"steps\":{d},\"vocab\":{d},\"hidden\":{d},\"taps\":{d},\"drafts\":[{s}]}}\n", .{ ids.len, start, steps, vocab, h, taps_n, list[0..n] });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&file, "{s}.json", .{stem}), .data = text });
        }
    }
}
