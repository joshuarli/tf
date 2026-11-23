//! Inkling 2 Mini's teacher-forced forward on Metal: the 24 physical layers, the core loop's three passes each with its
//! own KV and convolution state, and the depth mixes. One token a call, one command buffer a forward, in the
//! reference's order (`campaign/ref/inkling2_loop_ref.py`, a transcription of llama.cpp `40b7a546`).
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const wts = @import("model.zig");
const kernels = @import("kernels.zig");

const Model = wts.Model;
const Cache = st.Cache;
const Scratch = st.Scratch;
const Tensor = wts.Tensor;
const Layer = wts.Layer;
const Controller = wts.Controller;

/// The kernels' argument structs; each mirrors one `struct` in the `.metal` sources.
const Mv = extern struct { m: u32, k: u32, cap: f32 };
const Dm = extern struct { m: u32, k: u32, cap: f32 };
const Dmg = extern struct { k: u32, m0: u32, m1: u32, m2: u32, m3: u32 };
const Dmgx = extern struct { k0: u32, k1: u32, k2: u32, m: [8]u32, x: [8]u32 };
const Rms = extern struct { width: u32, blocks: u32, eps: f32 };
const Mix = extern struct { width: u32, depth: u32, k: u32, scale: f32 };
const Clip = extern struct { heads: u32, width: u32, threshold: f32 };
const Conv = extern struct { channels: u32, heads: u32, ring: u32, pos: u32, layout: u32, norm: f32 };
const Attn = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, scale: f32, floor: f32, alpha: f32 };
const AttnParts = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, splits: u32, chunk: u32, scale: f32, floor: f32, alpha: f32 };
const Pair = extern struct { count: u32 };
const Route = extern struct { experts: u32, topk: u32, route_scale: f32 };
const Moe = extern struct { k: u32, n: u32, rows: u32, gate_cap: f32, up_cap: f32 };
const Combine = extern struct { width: u32, topk: u32 };
const Plan = extern struct { slots: u32, experts: u32 };
const AffineDense = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, bits: u32, cap: f32 };
const AffineExpert = extern struct { k: u32, n: u32, slots: u32, topk: u32, gate_cap: f32, up_cap: f32 };
const AffineEmbed = extern struct { width: u32, rows: u32, eps: f32 };

const B = struct { buf: mtl.Buffer, off: usize = 0, args: bool = false };
const A = B{ .buf = undefined, .args = true };

fn b(buf: mtl.Buffer) B {
    return .{ .buf = buf };
}

fn bo(buf: mtl.Buffer, off: usize) B {
    return .{ .buf = buf, .off = off };
}

fn t(x: Tensor) B {
    return .{ .buf = x.buffer, .off = x.offset };
}

fn opt(x: ?Tensor) B {
    return if (x) |v| t(v) else A;
}

fn width(x: Tensor) usize {
    return if (x.rank >= 2) x.shape[1] else x.shape[0];
}

fn nrows(x: Tensor) usize {
    return if (x.rank >= 2) x.shape[0] else 1;
}

/// `count` rows of `x` from row `at`, as a tensor of their own.
fn rows(x: Tensor, at: usize, count: usize) Tensor {
    const k = width(x);
    var out = x;
    out.offset = x.offset + at * k * x.dtype.size();
    out.bytes = count * k * x.dtype.size();
    out.rank = 2;
    out.shape = .{ count, k, 1, 1, 1 };
    return out;
}

pub const Forward = struct {
    m: *const Model,
    s: *Scratch,
    cache: *Cache,
    enc: mtl.ComputeEncoder,
    h: u32,
    qw: u32,
    kvw: u32,
    heads: u32,
    kv: u32,
    dim: u32,
    drel: u32,
    pos: u32,
    blocks: [st.mix_states]B = @splat(b(undefined)),
    count: usize = 0,
    capture: ?*st.Trace = null,

    fn set(f: Forward, g: wts.Glue, args: anytype, bufs: []const B) void {
        f.setRaw(f.m.pipe(g), args, bufs);
    }

    fn setRaw(f: Forward, pipe: mtl.Pipeline, args: anytype, bufs: []const B) void {
        f.enc.setPipeline(pipe);
        for (bufs, 0..) |buf, i| {
            if (buf.args) f.enc.setValue(args, i) else f.enc.setBuffer(buf.buf, buf.off, i);
        }
    }

    fn setParts(f: Forward, pipe: mtl.Pipeline, args: AttnParts, bufs: []const B) void {
        f.enc.setPipeline(pipe);
        for (bufs, 0..) |buf, i| {
            if (buf.args) f.enc.setValue(args, i) else f.enc.setBuffer(buf.buf, buf.off, i);
        }
    }

    fn dense(f: Forward) ?kernels.Dense {
        return kernels.densePipes(f.m.device) catch null;
    }

    fn setExpert(f: Forward, mx: bool, e: wts.Experts, args: anytype, bufs: []const B) void {
        f.enc.setPipeline(f.m.expert(mx, e));
        for (bufs, 0..) |buf, i| {
            if (buf.args) f.enc.setValue(args, i) else f.enc.setBuffer(buf.buf, buf.off, i);
        }
    }

    /// out[rows] = bf16 W[rows, k] . x[k]; a warp an output row.
    fn lin(f: Forward, w: Tensor, x: B, out: B) void {
        const m: usize = nrows(w);
        if (w.affine != null or w.dtype == .f32) {
            f.affineLin(w, x, out, @intCast(m), 0);
            return;
        }
        if (f.dense()) |p| {
            f.setRaw(p.mv, Dm{ .m = @intCast(m), .k = @intCast(width(w)), .cap = 0 }, &.{ x, t(w), A, out });
            f.enc.dispatchGroups(mtl.Size.of((m + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        f.set(.mv, Mv{ .m = @intCast(m), .k = @intCast(width(w)), .cap = 0 }, &.{ x, t(w), A, out });
        f.enc.dispatchGroups(mtl.Size.of((m + 31) / 32, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    /// The unembedding: the soft-capped logits over the unpadded vocabulary.
    fn headModel(f: Forward, w: Tensor, x: B, out: B, vocab: u32) void {
        if (w.affine != null) {
            f.affineLin(w, x, out, vocab, f.m.config.text.logit_cap);
            return;
        }
        if (f.dense()) |p| {
            f.setRaw(p.mv_cap, Dm{ .m = vocab, .k = @intCast(width(w)), .cap = f.m.config.text.logit_cap }, &.{ x, t(w), A, out });
            f.enc.dispatchGroups(mtl.Size.of((vocab + 31) / 32, 1, 1), mtl.Size.of(1024, 1, 1));
            return;
        }
        f.set(.mv_cap, Mv{ .m = vocab, .k = @intCast(width(w)), .cap = f.m.config.text.logit_cap }, &.{ x, t(w), A, out });
        f.enc.dispatchGroups(mtl.Size.of((vocab + 31) / 32, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    fn affineLin(f: Forward, w: Tensor, x: B, out: B, n: u32, cap: f32) void {
        const p = f.m.affine_pipes.?;
        const args = AffineDense{ .n = n, .k = @intCast(width(w)), .rows = 1, .xs = 1, .xb = 0, .bits = if (w.affine) |q| q.bits() else 0, .cap = cap };
        if (w.affine) |q| {
            f.setRaw(p.dense, args, &.{ x, t(w), bo(q.scales, q.scale_offset), bo(q.biases, q.bias_offset), A, out });
        } else f.setRaw(p.float, args, &.{ x, t(w), A, out });
        f.enc.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn fusedx(f: Forward, p: kernels.Dense, xs: [3]B, ks: [3]u32, ws: [8]Tensor, os: [8]B, ms: [8]u32, xsel: [8]u32) void {
        var total: usize = 0;
        for (ms) |r| total += r;
        f.setRaw(p.fusedx, Dmgx{ .k0 = ks[0], .k1 = ks[1], .k2 = ks[2], .m = ms, .x = xsel }, &.{
            xs[0], xs[1], xs[2],
            t(ws[0]), os[0], t(ws[1]), os[1], t(ws[2]), os[2], t(ws[3]), os[3],
            t(ws[4]), os[4], t(ws[5]), os[5], t(ws[6]), os[6], t(ws[7]), os[7], A,
        });
        f.enc.dispatchGroups(mtl.Size.of((total + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn fused2(f: Forward, p: kernels.Dense, x: B, w0: Tensor, o0: B, w1: Tensor, o1: B) void {
        const m0: usize = nrows(w0);
        const m1: usize = nrows(w1);
        f.setRaw(p.fused, Dmg{ .k = @intCast(width(w0)), .m0 = @intCast(m0), .m1 = @intCast(m1), .m2 = 0, .m3 = 0 }, &.{ x, t(w0), o0, t(w1), o1, t(w0), o0, t(w0), o0, A });
        f.enc.dispatchGroups(mtl.Size.of((m0 + m1 + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn projGroup(f: Forward, p: kernels.Dense, lay: *const Layer, inputs: [3]B, full: bool) void {
        const a = &lay.attn;
        const s = f.s;
        const kd: Tensor = if (full) a.k_dyn.? else a.k;
        const vd: Tensor = if (full) a.v_dyn.? else a.v;
        f.fusedx(p, inputs, .{ @intCast(width(a.q)), @intCast(width(a.k)), @intCast(width(a.v)) }, .{ a.q, a.r, a.sink, a.gate, a.k, kd, a.v, vd }, .{ b(s.q), b(s.r), b(s.sink), b(s.gate), b(s.k), if (full) b(s.kdyn) else b(s.k), b(s.v), if (full) b(s.vdyn) else b(s.v) }, .{ @intCast(nrows(a.q)), @intCast(nrows(a.r)), @intCast(nrows(a.sink)), @intCast(nrows(a.gate)), @intCast(nrows(a.k)), if (full) @intCast(nrows(a.k_dyn.?)) else 0, @intCast(nrows(a.v)), if (full) @intCast(nrows(a.v_dyn.?)) else 0 }, .{ 0, 0, 0, 0, 1, 1, 2, 2 });
    }


    /// rms_norm of `count` rows `w` wide; a null weight leaves them unweighted (the depth mix).
    fn rms(f: Forward, x: B, weight: ?B, out: B, count: u32, w: u32, eps: f32) void {
        const wb = weight orelse b(f.s.x);
        f.set(.rms, Rms{ .width = w, .blocks = if (weight == null) 0 else count, .eps = eps }, &.{ x, wb, A, out });
        f.enc.dispatchGroups(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn rms1(f: Forward, x: B, weight: ?B, out: B) void {
        f.rms(x, weight, out, 1, f.h, f.m.config.text.eps);
    }

    fn flat(f: Forward, g: wts.Glue, args: anytype, bufs: []const B, count: usize) void {
        f.set(g, args, bufs);
        f.enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn copyN(f: Forward, x: B, out: B, count: usize) void {
        f.flat(.copy, Pair{ .count = @intCast(count) }, &.{ x, out, A }, count);
    }

    fn addN(f: Forward, dst: B, src: B, count: usize) void {
        f.flat(.add, Pair{ .count = @intCast(count) }, &.{ dst, src, A }, count);
    }

    /// The depth mix: `states` newest first, then `cur`, over `query`'s rows.
    fn mix(f: Forward, states: []const B, cur: B, query: Tensor, out: B) void {
        const s = f.s;
        var depth: u32 = 0;
        for (states) |state| {
            f.copyN(state, bo(s.states, depth * f.h * 4), f.h);
            depth += 1;
        }
        f.copyN(cur, bo(s.states, depth * f.h * 4), f.h);
        depth += 1;
        f.rms(b(s.states), null, b(s.mixs), depth, f.h, 1e-6);
        const n: u32 = @intCast(nrows(query));
        f.set(.mix, Mix{ .width = f.h, .depth = depth, .k = n, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(f.h))) }, &.{ b(s.mixs), t(query), A, out });
        f.enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// The mix over this token's block states newest first, then `cur`.
    fn mixBlocks(f: Forward, cur: B, query: Tensor, out: B) void {
        var ordered: [st.mix_states]B = undefined;
        for (0..f.count) |i| ordered[i] = f.blocks[f.count - 1 - i];
        f.mix(ordered[0..f.count], cur, query, out);
    }

    fn pushBlock(f: *Forward, state: B) void {
        const at = bo(f.s.blocks, f.count * f.h * 4);
        f.copyN(state, at, f.h);
        f.blocks[f.count] = at;
        f.count += 1;
    }

    fn convolveDynPre(f: Forward, x: B, out: B, weight: Tensor, dyn_out: B, ring: B, channels: u32, conv_heads: u32, layout: u32, dyn: bool) void {
        const taps: u32 = cfg.conv_taps;
        const mul: u32 = if (dyn) 2 else 1;
        const norm: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(mul * taps + 1)));
        const args = Conv{ .channels = channels, .heads = conv_heads, .ring = st.ring, .pos = f.pos, .layout = layout, .norm = norm };
        f.set(.conv, args, &.{ x, ring, t(weight), dyn_out, A, out });
        f.enc.dispatchThreads(mtl.Size.of(channels, 1, 1), mtl.Size.of(256, 1, 1));
        f.set(.conv_store, args, &.{ x, ring, A });
        f.enc.dispatchThreads(mtl.Size.of(channels, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn rings(f: Forward, slot: u32) [4]B {
        const kv: usize = f.kvw;
        return .{
            bo(f.cache.k_conv, f.cache.convAt(slot, kv)),
            bo(f.cache.v_conv, f.cache.convAt(slot, kv)),
            bo(f.cache.attn_conv, f.cache.convAt(slot, f.h)),
            bo(f.cache.mlp_conv, f.cache.convAt(slot, f.h)),
        };
    }

    /// One convolution site over a token: the taps, then the ring's update. `dyn_in` is the projection's input row and
    /// `x` the convolved row; they differ for the keys and values, whose taps come from a separate projection.
    fn convolve(f: Forward, x: B, out: B, weight: Tensor, dyn: ?Tensor, dyn_in: B, dyn_out: B, ring: B, channels: u32, conv_heads: u32, layout: u32) void {
        const taps: u32 = cfg.conv_taps;
        const mul: u32 = if (dyn != null) 2 else 1;
        const norm: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(mul * taps + 1)));
        if (dyn) |dw| f.lin(dw, dyn_in, dyn_out);
        const args = Conv{ .channels = channels, .heads = conv_heads, .ring = st.ring, .pos = f.pos, .layout = layout, .norm = norm };
        f.set(.conv, args, &.{ x, ring, t(weight), if (dyn != null) dyn_out else A, A, out });
        f.enc.dispatchThreads(mtl.Size.of(channels, 1, 1), mtl.Size.of(256, 1, 1));
        f.set(.conv_store, args, &.{ x, ring, A });
        f.enc.dispatchThreads(mtl.Size.of(channels, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// One executed slot. `ip` is its executed position (for the block boundaries), `slot` its state.
    fn layer(f: *Forward, slot: u32, ip: u32, bank: ?Bank) !void {
        const m = f.m;
        const c = &m.config;
        const text = &c.text;
        const index = c.loop.physicalLayer(text.layers, slot);
        const lay: *const Layer = &m.weights.layers[index];
        const full = c.isFull(index);
        const s = f.s;
        const inp: B = b(s.x);

        var inputs = [3]B{ inp, inp, inp };
        if (index > 0) {
            f.mixBlocks(inp, m.weights.attn_queries[index - 1], bo(s.mixout, 0));
            inputs = if (full)
                .{ bo(s.mixout, 0), bo(s.mixout, f.h * 4), bo(s.mixout, 2 * f.h * 4) }
            else
                .{ bo(s.mixout, 0), bo(s.mixout, 0), bo(s.mixout, 0) };
        }

        var msg: ?B = null;
        if (bank) |bk| {
            msg = f.messages(bk, ip);
            for (0..3) |j| {
                const ctl = lay.controllers[j] orelse return error.MissingController;
                f.control(ctl, inputs[j], bo(msg.?.buf, msg.?.off + j * f.h * 4), bo(s.xn, j * f.h * 4));
            }
            inputs = .{ bo(s.xn, 0), bo(s.xn, f.h * 4), bo(s.xn, 2 * f.h * 4) };
        }

        if (f.capture) |cap| for (0..3) |j| f.record(cap, slot, j, inputs[j], f.h);

        if (full) {
            f.rms1(inputs[0], t(lay.q_norm), bo(s.xn, 0));
            f.rms1(inputs[1], t(lay.k_norm), bo(s.xn, f.h * 4));
            f.rms1(inputs[2], t(lay.v_norm), bo(s.xn, 2 * f.h * 4));
            inputs = .{ bo(s.xn, 0), bo(s.xn, f.h * 4), bo(s.xn, 2 * f.h * 4) };
        } else if (bank != null) {
            for (0..3) |j| f.rms1(inputs[j], t(lay.q_norm), bo(s.xn, j * f.h * 4));
            inputs = .{ bo(s.xn, 0), bo(s.xn, f.h * 4), bo(s.xn, 2 * f.h * 4) };
        } else {
            f.rms1(inputs[0], t(lay.q_norm), bo(s.xn, 0));
            inputs = .{ bo(s.xn, 0), bo(s.xn, 0), bo(s.xn, 0) };
        }

        if (f.capture) |cap| for (0..3) |j| f.record(cap, slot, 3 + j, inputs[j], f.h);
        f.attention(lay, slot, full, inputs);
        if (f.capture) |cap| f.record(cap, slot, 6, b(s.o), f.h);
        if (ip % text.block == 0) {
            f.pushBlock(inp);
            f.copyN(b(s.o), b(s.x), f.h);
        } else f.addN(b(s.x), b(s.o), f.h);

        f.copyN(b(s.x), b(s.ffnin), f.h);
        if (f.capture) |cap| f.record(cap, slot, 7, b(s.ffnin), f.h);
        f.mixBlocks(b(s.x), m.weights.mlp_queries[index], bo(s.mixout, 0));
        if (msg) |got| {
            const ctl = lay.controllers[3] orelse return error.MissingController;
            f.control(ctl, bo(s.mixout, 0), bo(got.buf, got.off + 3 * f.h * 4), bo(s.mixout, 0));
        }
        f.rms1(bo(s.mixout, 0), t(lay.mlp_norm), bo(s.xn, 0));
        f.experts(lay, bo(s.xn, 0), slot);
        const slots = f.rings(slot);
        f.convolve(b(s.moe), b(s.aconv), lay.mlp_post, null, A, A, slots[3], f.h, 0, 0);
        f.copyN(b(s.ffnin), b(s.x), f.h);
        f.addN(b(s.x), b(s.aconv), f.h);
        if (f.capture) |cap| f.record(cap, slot, 8, b(s.x), f.h);
    }

    fn record(f: Forward, _: *st.Trace, slot: u32, which: usize, src: B, cols: u32) void {
        const at = (@as(usize, slot) * st.trace_rows + which) * f.h * 4;
        f.copyN(src, bo(f.s.capture, at), cols);
    }

    /// The four messages of one core layer: its stacked cross-pass query rows over the bank, newest state first.
    fn messages(f: Forward, bk: Bank, ip: u32) B {
        const s = f.s;
        const at: usize = ip - f.m.config.loop.core_start;
        var depth: u32 = 0;
        var i: usize = bk.states.len;
        while (i > 0) {
            i -= 1;
            f.copyN(bk.states[i], bo(s.states, depth * f.h * 4), f.h);
            depth += 1;
        }
        f.copyN(bk.out, bo(s.states, depth * f.h * 4), f.h);
        depth += 1;
        f.rms(b(s.states), null, b(s.mixs), depth, f.h, 1e-6);
        f.set(.mix4, Mix{ .width = f.h, .depth = depth, .k = 4, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(f.h))) }, &.{
            b(s.mixs),
            t(rows(f.m.weights.cross_pass_q, at * 2, 1)),
            t(rows(f.m.weights.cross_pass_kv, at * 2, 1)),
            t(rows(f.m.weights.cross_pass_kv, at * 2 + 1, 1)),
            t(rows(f.m.weights.cross_pass_q, at * 2 + 1, 1)),
            A,
            b(s.msg),
        });
        f.enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
        return b(s.msg);
    }

    /// A loop controller's output: cur - forget * cur + gain * message.
    fn control(f: Forward, ctl: Controller, cur: B, message: B, out: B) void {
        const s = f.s;
        f.rms1(cur, null, bo(s.h2, 0));
        f.rms1(message, null, bo(s.h2, f.h * 4));
        if (ctl.gain.dtype == .bf16 and f.dense() != null) {
            const p = f.dense().?;
            f.lin(ctl.down, b(s.h2), b(s.normed));
            f.fused2(p, b(s.normed), ctl.gain, bo(s.tanh, 0), ctl.forget, bo(s.tanh, f.h * 4));
        } else {
            f.lin(ctl.down, b(s.h2), b(s.normed));
            f.lin(ctl.gain, b(s.normed), bo(s.tanh, 0));
            f.lin(ctl.forget, b(s.normed), bo(s.tanh, f.h * 4));
        }
        f.flat(.ctl, Pair{ .count = f.h }, &.{ cur, message, bo(s.tanh, 0), bo(s.tanh, f.h * 4), A, out }, f.h);
    }

    fn attention(f: Forward, lay: *const Layer, slot: u32, full: bool, inputs: [3]B) void {
        const s = f.s;
        const a = &lay.attn;
        const text = &f.m.config.text;
        const extent = f.extentOf(full);
        const ring = f.rings(slot);
        if (a.q.affine == null and f.dense() != null) {
            const p = f.dense().?;
            f.projGroup(p, lay, inputs, full);
            f.convolveDynPre(b(s.k), b(s.kc), a.k_conv, b(s.kdyn), ring[0], f.kvw, if (full) f.kv else 0, @intFromBool(full), full);
            f.convolveDynPre(b(s.v), b(s.vc), a.v_conv, b(s.vdyn), ring[1], f.kvw, if (full) f.kv else 0, @intFromBool(full), full);
        } else {
            f.lin(a.q, inputs[0], b(s.q));
            f.lin(a.r, inputs[0], b(s.r));
            f.lin(a.sink, inputs[0], b(s.sink));
            f.lin(a.gate, inputs[0], b(s.gate));
            f.lin(a.k, inputs[1], b(s.k));
            f.convolve(b(s.k), b(s.kc), a.k_conv, a.k_dyn, inputs[1], b(s.kdyn), ring[0], f.kvw, if (full) f.kv else 0, @intFromBool(full));
            f.lin(a.v, inputs[2], b(s.v));
            f.convolve(b(s.v), b(s.vc), a.v_conv, a.v_dyn, inputs[2], b(s.vdyn), ring[1], f.kvw, if (full) f.kv else 0, @intFromBool(full));
        }
        f.clipScaled(b(s.q), a.q_scale, f.heads, f.dim, text.qk_clip);
        f.clipScaled(b(s.r), a.r_scale, f.heads, f.drel, text.rel_clip);
        f.clipPlain(b(s.kc), f.kv, f.dim, text.qk_clip);
        f.rms(b(s.vc), null, b(s.vc), f.kv, f.dim, text.eps);
        f.kvStore(slot);
        f.relative(lay, b(s.r), extent, full);
        const win: u32 = if (full) 0 else text.window;
        const lo: u32 = if (win != 0 and f.pos + 1 > win) f.pos + 1 - win else 0;
        const part = st.attnPartition(f.pos + 1 - lo);
        const parts = AttnParts{
            .heads = f.heads,
            .kv = f.kv,
            .dim = f.dim,
            .rel = f.drel,
            .extent = extent,
            .window = win,
            .pos = f.pos,
            .tau = @intFromBool(full),
            .splits = part.splits,
            .chunk = part.chunk,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(f.dim))),
            .floor = @floatFromInt(text.log_floor),
            .alpha = text.log_alpha,
        };
        const keybase = bo(f.cache.keys, f.cache.row(&f.m.config, slot, 0));
        const valbase = bo(f.cache.values, f.cache.row(&f.m.config, slot, 0));
        f.setParts(f.s.attn_split, parts, &.{ b(s.q), keybase, valbase, b(s.relbias), b(s.sink), A, b(s.attn_part) });
        f.enc.dispatchGroups(mtl.Size.of(f.kv * part.splits, 1, 1), mtl.Size.of((f.heads / f.kv) * 32, 1, 1));
        f.setParts(f.s.attn_combine, parts, &.{ b(s.attn_part), A, b(s.attn) });
        f.enc.dispatchGroups(mtl.Size.of(f.heads, 1, 1), mtl.Size.of(32, 1, 1));
        f.flat(.sig_mul, Pair{ .count = f.qw }, &.{ b(s.attn), b(s.gate), A, b(s.attn) }, f.qw);
        f.lin(a.o, b(s.attn), b(s.projout));
        f.convolve(b(s.projout), b(s.o), lay.attn_post, null, A, A, ring[2], f.h, 0, 0);
    }

    fn extentOf(f: Forward, full: bool) u32 {
        return if (full) f.m.config.text.rel_extent else f.m.config.text.window;
    }

    /// This token's convolved key and normed value into its slot's row, for the positions after it.
    fn kvStore(f: Forward, slot: u32) void {
        const at = f.cache.row(&f.m.config, slot, f.pos);
        f.flat(.kv_store, Pair{ .count = f.kvw }, &.{ b(f.s.kc), b(f.s.vc), bo(f.cache.keys, at), bo(f.cache.values, at), A }, f.kvw);
    }

    fn clipScaled(f: Forward, x: B, scale: Tensor, heads: u32, w: u32, threshold: f32) void {
        f.set(.clip_scaled, Clip{ .heads = heads, .width = w, .threshold = threshold }, &.{ x, t(scale), A, x });
        f.enc.dispatchGroups(mtl.Size.of(heads, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn clipPlain(f: Forward, x: B, heads: u32, w: u32, threshold: f32) void {
        f.set(.clip, Clip{ .heads = heads, .width = w, .threshold = threshold }, &.{ x, A, x });
        f.enc.dispatchGroups(mtl.Size.of(heads, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn relative(f: Forward, lay: *const Layer, r: B, extent: u32, full: bool) void {
        const text = &f.m.config.text;
        f.set(.relbias, Attn{ .heads = f.heads, .kv = f.kv, .dim = f.dim, .rel = f.drel, .extent = extent, .window = 0, .pos = f.pos, .tau = @intFromBool(full), .scale = 0, .floor = @floatFromInt(text.log_floor), .alpha = text.log_alpha }, &.{ t(lay.attn.rel_bias), r, A, b(f.s.relbias) });
        f.enc.dispatchThreads(mtl.Size.of(f.heads * (extent + 1), 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn experts(f: Forward, lay: *const Layer, input: B, slot: u32) void {
        const m = f.m;
        const c = &m.config;
        const s = f.s;
        const E: u32 = c.text.experts;
        const K: u32 = c.text.topk;
        const inter: u32 = c.text.expert_inter;
        f.lin(lay.router, input, b(s.router));
        f.set(.route, Route{ .experts = E, .topk = K, .route_scale = c.text.route_scale }, &.{ b(s.router), t(lay.router_bias), A, b(s.picks), b(s.wts) });
        f.enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
        if (f.capture != null) {
            f.copyN(b(s.picks), bo(s.capture, (c.slots() * st.trace_rows * f.h + slot * K) * 4), K);
            const start = c.slots() * (st.trace_rows * f.h + K) + slot * E * 2;
            f.copyN(b(s.router), bo(s.capture, start * 4), E);
            f.copyN(t(lay.router_bias), bo(s.capture, (start + E) * 4), E);
        }
        if (lay.precision == .affine4) {
            const p = m.affine_pipes.?;
            const args = AffineExpert{ .k = f.h, .n = inter, .slots = K, .topk = K, .gate_cap = c.text.gate_cap, .up_cap = c.text.up_cap };
            f.setRaw(p.up, args, &.{ input, t(lay.w13), t(lay.w13_scale), t(lay.w13_bias.?), b(s.picks), A, b(s.act) });
            f.enc.dispatchGroups(mtl.Size.of((inter + 7) / 8, K, 1), mtl.Size.of(256, 1, 1));
            f.setRaw(p.down, AffineExpert{ .k = inter, .n = f.h, .slots = K, .topk = K, .gate_cap = 0, .up_cap = 0 }, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), t(lay.w2_bias.?), b(s.picks), A, b(s.down) });
            f.enc.dispatchGroups(mtl.Size.of((f.h + 7) / 8, K, 1), mtl.Size.of(256, 1, 1));
            f.set(.combine, Combine{ .width = f.h, .topk = K }, &.{ b(s.down), b(s.wts), A, b(s.moe) });
            f.enc.dispatchThreads(mtl.Size.of(f.h, 1, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        const codes = b(if (lay.precision == .mxfp8) m.t256 else m.t16);
        f.setExpert(lay.precision == .mxfp8, .gu, Moe{ .k = f.h, .n = inter, .rows = 2 * inter, .gate_cap = c.text.gate_cap, .up_cap = c.text.up_cap }, &.{
            input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.picks), b(s.order), A, b(s.act), codes,
        });
        f.enc.dispatchGroups(mtl.Size.of(inter, K, 1), mtl.Size.of(32, 1, 1));
        const down = Moe{ .k = inter, .n = f.h, .rows = f.h, .gate_cap = 0, .up_cap = 0 };
        if (lay.precision == .mxfp8) {
            f.setExpert(true, .down, down, &.{
                b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.picks), b(s.order), A, b(s.down),
            });
            f.enc.dispatchGroups(mtl.Size.of((f.h + 63) / 64, K, 1), mtl.Size.of(64, 1, 1));
        } else {
            f.setExpert(false, .down, down, &.{
                b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.picks), b(s.order), A, b(s.down), codes,
            });
            f.enc.dispatchGroups(mtl.Size.of(f.h, K, 1), mtl.Size.of(32, 1, 1));
        }
        f.set(.combine, Combine{ .width = f.h, .topk = K }, &.{ b(s.down), b(s.wts), A, b(s.moe) });
        f.enc.dispatchThreads(mtl.Size.of(f.h, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn embed(f: Forward, id: u32) void {
        const m = f.m;
        f.s.ids.slice(u32, 1)[0] = id;
        if (m.weights.embed.affine) |q| {
            f.setRaw(m.affine_pipes.?.embed, AffineEmbed{ .width = f.h, .rows = 1, .eps = m.config.text.eps }, &.{ b(f.s.ids), t(m.weights.embed), bo(q.scales, q.scale_offset), bo(q.biases, q.bias_offset), t(m.weights.embed_norm), A, b(f.s.x) });
        } else f.set(.embed, Rms{ .width = f.h, .blocks = 1, .eps = m.config.text.eps }, &.{ b(f.s.ids), t(m.weights.embed), t(m.weights.embed_norm), A, b(f.s.x) });
        f.enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
    }
};

/// A pass's bank: the previous pass's core blocks after the first, and its final output.
pub const Bank = struct { states: []const B, out: B };

/// One token's forward: `cache.len` positions precede it; the logits land in `s.logits`.
pub fn position(m: *const Model, s: *Scratch, cache: *Cache, id: u32) !void {
    return positionTrace(m, s, cache, id, null);
}

/// The same pass, recording the executed slots' rows into `t` when it is given.
pub fn positionTrace(m: *const Model, s: *Scratch, cache: *Cache, id: u32, trace: ?*st.Trace) !void {
    const c = &m.config;
    const text = &c.text;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    var f = Forward{
        .m = m,
        .s = s,
        .cache = cache,
        .enc = cb.compute(.serial),
        .h = text.hidden,
        .qw = text.local.q * text.local.dim,
        .kvw = text.local.kv * text.local.dim,
        .heads = text.local.q,
        .kv = text.local.kv,
        .dim = text.local.dim,
        .drel = text.d_rel,
        .pos = @intCast(cache.len),
        .capture = trace,
    };
    f.embed(id);
    for (0..c.loop.core_start) |i| try f.layer(@intCast(i), @intCast(i), null);
    const prelude = f.count;
    f.copyN(b(s.x), b(s.prelude), f.h);
    const span = c.loop.core_end - c.loop.core_start;
    var bank_states: [st.mix_states]B = undefined;
    var pass: u32 = 0;
    while (pass < c.loop.passes) : (pass += 1) {
        var bank: ?Bank = null;
        if (pass > 0) {
            var n: usize = 0;
            for (prelude + 1..f.count) |i| {
                f.copyN(f.blocks[i], bo(s.bank, n * f.h * 4), f.h);
                bank_states[n] = bo(s.bank, n * f.h * 4);
                n += 1;
            }
            f.copyN(b(s.x), b(s.passout), f.h);
            bank = .{ .states = bank_states[0..n], .out = b(s.passout) };
            f.count = prelude;
            f.copyN(b(s.prelude), b(s.x), f.h);
        }
        for (c.loop.core_start..c.loop.core_end) |ip| {
            const slot: u32 = if (pass == 0) @intCast(ip) else @intCast(text.layers + (pass - 1) * span + (ip - c.loop.core_start));
            try f.layer(slot, @intCast(ip), bank);
        }
    }
    for (c.loop.core_end..text.layers) |i| try f.layer(@intCast(i), @intCast(i), null);
    f.mixBlocks(b(s.x), m.weights.output_query, bo(s.mixout, 0));
    f.rms1(bo(s.mixout, 0), t(m.weights.norm), bo(s.xn, 0));
    f.headModel(m.weights.unembed, bo(s.xn, 0), b(s.logits), text.unpadded_vocab);
    f.enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop forward: {s}", .{msg});
        return error.GpuFailed;
    }
    if (trace) |tr| {
        var slots: [cfg.max_layers]u32 = undefined;
        var n: usize = 0;
        for (0..c.loop.core_start) |i| {
            slots[n] = @intCast(i);
            n += 1;
        }
        const core = c.loop.core_end - c.loop.core_start;
        for (0..c.loop.passes) |p| for (c.loop.core_start..c.loop.core_end) |ip| {
            slots[n] = if (p == 0) @intCast(ip) else @intCast(text.layers + (p - 1) * core + (ip - c.loop.core_start));
            n += 1;
        };
        for (c.loop.core_end..text.layers) |i| {
            slots[n] = @intCast(i);
            n += 1;
        }
        try tr.position(s.capture, slots[0..n], cache.len);
    }
    cache.len += 1;
}

test {
    _ = st;
    _ = @import("attention_test.zig");
}
