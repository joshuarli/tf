const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const wts = @import("model.zig");
const kernels = @import("kernels.zig");

const Model = wts.Model;
const Cache = st.Cache;
const Tensor = wts.Tensor;
const Layer = wts.Layer;
const Controller = wts.Controller;

pub const default_chunk: usize = 512;

pub const WantLogits = enum { none, last, all };

const Dense = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32 };
const Cap = extern struct { n: u32, k: u32, rows: u32, cap: f32 };
const RouterA = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32 };
const Gemm = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, cap: f32 };
const Pick = extern struct { experts: u32, topk: u32, rows: u32, route_scale: f32 };
const EmbedA = extern struct { width: u32, rows: u32, eps: f32 };
const RmsB = extern struct { width: u32, rows: u32, xs: u32, xb: u32, ys: u32, yb: u32, wb: u32, eps: f32 };
const Gather = extern struct { width: u32, depth: u32, stride: u32 };
const MixB = extern struct { width: u32, depth: u32, k: u32, rows: u32, scale: f32 };
const CtlB = extern struct { h: u32, rows: u32, cxs: u32, cxb: u32, mxs: u32, mxb: u32, ys: u32, yb: u32 };
const ClipB = extern struct { heads: u32, width: u32, rows: u32, threshold: f32 };
const ConvB = extern struct { channels: u32, heads: u32, ring: u32, base: u32, rows: u32, layout: u32, norm: f32 };
const CStore = extern struct { channels: u32, ring: u32, base: u32, rows: u32 };
const KvB = extern struct { count: u32, base: u32, rows: u32, wring: u32 };
const RelB = extern struct { heads: u32, rel: u32, extent: u32, rows: u32, base: u32, tau: u32, floor: f32, alpha: f32 };
const AttnB = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, base: u32, rows: u32, rmode: u32, scale: f32, floor: f32, alpha: f32 };
const MoeB = extern struct { k: u32, n: u32, slots: u32, topk: u32, gate_cap: f32, up_cap: f32 };
const Moeg = extern struct { k: u32, n: u32, slots: u32, topk: u32, experts: u32, gate_cap: f32, up_cap: f32 };
const CombineB = extern struct { width: u32, topk: u32, rows: u32 };
const Rms = extern struct { width: u32, blocks: u32, eps: f32 };
const Pair = extern struct { count: u32 };

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

fn srows(x: Tensor, at: usize, count: usize) Tensor {
    const k = width(x);
    var out = x;
    out.offset = x.offset + at * k * x.dtype.size();
    out.bytes = count * k * x.dtype.size();
    out.rank = 2;
    out.shape = .{ count, k, 1, 1, 1 };
    return out;
}

pub const Pipes = struct {
    dense: mtl.Pipeline,
    cap: mtl.Pipeline,
    capexact: mtl.Pipeline,
    router: mtl.Pipeline,
    gemm: mtl.Pipeline,
    smallrow: mtl.Pipeline,
    gemm_mpp: ?mtl.Pipeline,
    pick: mtl.Pipeline,
    embed: mtl.Pipeline,
    rms: mtl.Pipeline,
    gather: mtl.Pipeline,
    mix: mtl.Pipeline,
    mix4: mtl.Pipeline,
    ctl: mtl.Pipeline,
    clips: mtl.Pipeline,
    clip: mtl.Pipeline,
    conv: mtl.Pipeline,
    cstore: mtl.Pipeline,
    kv: mtl.Pipeline,
    rel: mtl.Pipeline,
    attn: mtl.Pipeline,
    attn_tiled: mtl.Pipeline,
    gu_nv: mtl.Pipeline,
    down_nv: mtl.Pipeline,
    gu_mx: mtl.Pipeline,
    down_mx: mtl.Pipeline,
    gu_g_nv: mtl.Pipeline,
    down_g_nv: mtl.Pipeline,
    gu_g_mx: mtl.Pipeline,
    down_g_mx: mtl.Pipeline,
    expert_matrix_nv: mtl.Pipeline,
    expert_matrix_mx: mtl.Pipeline,
    moe_zero: mtl.Pipeline,
    moe_count: mtl.Pipeline,
    moe_offsets: mtl.Pipeline,
    moe_sort: mtl.Pipeline,
    combine: mtl.Pipeline,

    pub fn deinit(p: *Pipes) void {
        if (p.gemm_mpp) |pipe| pipe.deinit();
        inline for ([_][]const u8{ "dense", "cap", "capexact", "router", "gemm", "smallrow", "pick", "gu_g_nv", "down_g_nv", "gu_g_mx", "down_g_mx", "expert_matrix_nv", "expert_matrix_mx", "moe_zero", "moe_count", "moe_offsets", "moe_sort", "embed", "rms", "gather", "mix", "mix4", "ctl", "clips", "clip", "conv", "cstore", "kv", "rel", "attn", "attn_tiled", "gu_nv", "down_nv", "gu_mx", "down_mx", "combine" }) |f| @field(p, f).deinit();
    }
};

pub const PrefillScratch = struct {
    pipes: Pipes,
    chunk: usize,
    h: u32,
    heads: u32,
    kv: u32,
    dim: u32,
    drel: u32,
    ids: mtl.Buffer,
    x: mtl.Buffer,
    prelude: mtl.Buffer,
    passout: mtl.Buffer,
    bank: mtl.Buffer,
    ffnin: mtl.Buffer,
    n0: mtl.Buffer,
    n1: mtl.Buffer,
    n2: mtl.Buffer,
    m0: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    r: mtl.Buffer,
    kc: mtl.Buffer,
    vc: mtl.Buffer,
    kdyn: mtl.Buffer,
    vdyn: mtl.Buffer,
    sink: mtl.Buffer,
    relbias: mtl.Buffer,
    attn: mtl.Buffer,
    gate: mtl.Buffer,
    projout: mtl.Buffer,
    o: mtl.Buffer,
    aconv: mtl.Buffer,
    moe: mtl.Buffer,
    blocks: mtl.Buffer,
    staged: mtl.Buffer,
    mixout: mtl.Buffer,
    msg: mtl.Buffer,
    h2in: mtl.Buffer,
    normed: mtl.Buffer,
    tanhg: mtl.Buffer,
    tanhf: mtl.Buffer,
    router: mtl.Buffer,
    picks: mtl.Buffer,
    wts: mtl.Buffer,
    act: mtl.Buffer,
    down: mtl.Buffer,
    moe_cnt: mtl.Buffer,
    moe_off: mtl.Buffer,
    moe_cur: mtl.Buffer,
    moe_sorted: mtl.Buffer,
    logits: mtl.Buffer,
    capture: ?mtl.Buffer,
    capture_row: usize,
    routes: ?mtl.Buffer,

    pub fn init(device: mtl.Device, c: *const cfg.Config, chunk: usize) !PrefillScratch {
        const h: usize = c.text.hidden;
        const qw: usize = @max(c.text.local.q * c.text.local.dim, c.text.global.q * c.text.global.dim);
        const kvw: usize = @max(c.text.local.kv * c.text.local.dim, c.text.global.kv * c.text.global.dim);
        const mq: usize = @max(c.text.local.q, c.text.global.q);
        const mkv: usize = @max(c.text.local.kv, c.text.global.kv);
        const extent: usize = @max(c.text.rel_extent, c.text.window) + 1;
        const inter: usize = c.text.expert_inter;
        const topk: usize = c.text.topk;
        const rank: usize = c.loop.gate_rank;
        const experts: usize = c.text.experts;
        const vocab: usize = c.text.unpadded_vocab;
        const S: usize = st.mix_states;
        var s: PrefillScratch = undefined;
        s.capture = null;
        s.capture_row = 0;
        s.routes = null;
        s.chunk = chunk;
        s.h = @intCast(h);
        s.heads = c.text.local.q;
        s.kv = c.text.local.kv;
        s.dim = c.text.local.dim;
        s.drel = c.text.d_rel;
        const alloc = [_]struct { ptr: *mtl.Buffer, bytes: usize }{
            .{ .ptr = &s.x, .bytes = chunk * h * 4 },
            .{ .ptr = &s.prelude, .bytes = chunk * h * 4 },
            .{ .ptr = &s.passout, .bytes = chunk * h * 4 },
            .{ .ptr = &s.bank, .bytes = chunk * S * h * 4 },
            .{ .ptr = &s.ffnin, .bytes = chunk * h * 4 },
            .{ .ptr = &s.n0, .bytes = chunk * h * 4 },
            .{ .ptr = &s.n1, .bytes = chunk * h * 4 },
            .{ .ptr = &s.n2, .bytes = chunk * h * 4 },
            .{ .ptr = &s.m0, .bytes = chunk * h * 4 },
            .{ .ptr = &s.q, .bytes = chunk * qw * 4 },
            .{ .ptr = &s.k, .bytes = chunk * kvw * 4 },
            .{ .ptr = &s.v, .bytes = chunk * kvw * 4 },
            .{ .ptr = &s.r, .bytes = chunk * mq * c.text.d_rel * 4 },
            .{ .ptr = &s.kc, .bytes = chunk * kvw * 4 },
            .{ .ptr = &s.vc, .bytes = chunk * kvw * 4 },
            .{ .ptr = &s.kdyn, .bytes = chunk * mkv * 4 * 4 },
            .{ .ptr = &s.vdyn, .bytes = chunk * mkv * 4 * 4 },
            .{ .ptr = &s.sink, .bytes = chunk * mq * 4 },
            .{ .ptr = &s.relbias, .bytes = chunk * mq * extent * 4 },
            .{ .ptr = &s.attn, .bytes = chunk * qw * 4 },
            .{ .ptr = &s.gate, .bytes = chunk * qw * 4 },
            .{ .ptr = &s.projout, .bytes = chunk * h * 4 },
            .{ .ptr = &s.o, .bytes = chunk * h * 4 },
            .{ .ptr = &s.aconv, .bytes = chunk * h * 4 },
            .{ .ptr = &s.moe, .bytes = chunk * h * 4 },
            .{ .ptr = &s.blocks, .bytes = chunk * S * h * 4 },
            .{ .ptr = &s.staged, .bytes = chunk * S * h * 4 },
            .{ .ptr = &s.mixout, .bytes = chunk * 4 * h * 4 },
            .{ .ptr = &s.msg, .bytes = chunk * 4 * h * 4 },
            .{ .ptr = &s.h2in, .bytes = chunk * 2 * h * 4 },
            .{ .ptr = &s.normed, .bytes = chunk * rank * 4 },
            .{ .ptr = &s.tanhg, .bytes = chunk * h * 4 },
            .{ .ptr = &s.tanhf, .bytes = chunk * h * 4 },
            .{ .ptr = &s.router, .bytes = chunk * experts * 4 },
            .{ .ptr = &s.picks, .bytes = chunk * topk * 4 },
            .{ .ptr = &s.wts, .bytes = chunk * topk * 4 },
            .{ .ptr = &s.act, .bytes = chunk * topk * inter * 4 },
            .{ .ptr = &s.down, .bytes = chunk * topk * h * 4 },
            .{ .ptr = &s.moe_cnt, .bytes = @max(experts * 4, 16) },
            .{ .ptr = &s.moe_off, .bytes = @max((experts + 1) * 4, 16) },
            .{ .ptr = &s.moe_cur, .bytes = @max(experts * 4, 16) },
            .{ .ptr = &s.moe_sorted, .bytes = @max(chunk * topk * 4, 16) },
        };
        var made: usize = 0;
        errdefer for (alloc[0..made]) |f| f.ptr.deinit();
        for (alloc) |f| {
            f.ptr.* = try device.buffer(@max(f.bytes, 16), mtl.ResourceOptions.private);
            made += 1;
        }
        s.ids = try device.buffer(@max(chunk * 4, 16), mtl.ResourceOptions.shared);
        errdefer s.ids.deinit();
        s.logits = try device.buffer(@max(chunk * vocab * 4, 16), mtl.ResourceOptions.shared);
        const opts = mtl.CompileOptions.mlx();
        const lib = try mtl.Library.fromSource(device, kernels.prefill, opts);
        defer lib.deinit();
        const libmx = try mtl.Library.fromSource(device, kernels.prefillSource(true), opts);
        defer libmx.deinit();
        s.pipes.dense = try mtl.Pipeline.init(device, lib, "i2p_dense", false);
        errdefer s.pipes.dense.deinit();
        s.pipes.cap = try mtl.Pipeline.init(device, lib, "i2p_dense_cap", false);
        errdefer s.pipes.cap.deinit();
        s.pipes.capexact = try mtl.Pipeline.init(device, lib, "i2p_cap_exact", false);
        errdefer s.pipes.capexact.deinit();
        s.pipes.router = try mtl.Pipeline.init(device, lib, "i2p_router_exact", false);
        errdefer s.pipes.router.deinit();
        s.pipes.gemm = try mtl.Pipeline.init(device, lib, "i2p_gemm", false);
        errdefer s.pipes.gemm.deinit();
        const small_lib = try mtl.Library.fromSource(device, kernels.prefill_smallrow, opts);
        defer small_lib.deinit();
        s.pipes.smallrow = try mtl.Pipeline.init(device, small_lib, "i2p_small", false);
        errdefer s.pipes.smallrow.deinit();
        s.pipes.gemm_mpp = if (device.tensorUnits()) blk: {
            const mpp_lib = try mtl.Library.fromSource(device, kernels.prefill_gemm_mpp, opts);
            defer mpp_lib.deinit();
            break :blk try mtl.Pipeline.init(device, mpp_lib, "i2p_gemm_mpp", false);
        } else null;
        errdefer if (s.pipes.gemm_mpp) |pipe| pipe.deinit();
        s.pipes.pick = try mtl.Pipeline.init(device, lib, "i2p_routepick", false);
        errdefer s.pipes.pick.deinit();
        s.pipes.embed = try mtl.Pipeline.init(device, lib, "i2p_embed", false);
        errdefer s.pipes.embed.deinit();
        s.pipes.rms = try mtl.Pipeline.init(device, lib, "i2p_rms", false);
        errdefer s.pipes.rms.deinit();
        s.pipes.gather = try mtl.Pipeline.init(device, lib, "i2p_gather", false);
        errdefer s.pipes.gather.deinit();
        s.pipes.mix = try mtl.Pipeline.init(device, lib, "i2p_mix", false);
        errdefer s.pipes.mix.deinit();
        s.pipes.mix4 = try mtl.Pipeline.init(device, lib, "i2p_mix4", false);
        errdefer s.pipes.mix4.deinit();
        s.pipes.ctl = try mtl.Pipeline.init(device, lib, "i2p_ctl", false);
        errdefer s.pipes.ctl.deinit();
        s.pipes.clips = try mtl.Pipeline.init(device, lib, "i2p_clip_scaled", false);
        errdefer s.pipes.clips.deinit();
        s.pipes.clip = try mtl.Pipeline.init(device, lib, "i2p_clip", false);
        errdefer s.pipes.clip.deinit();
        s.pipes.conv = try mtl.Pipeline.init(device, lib, "i2p_conv", false);
        errdefer s.pipes.conv.deinit();
        s.pipes.cstore = try mtl.Pipeline.init(device, lib, "i2p_conv_store", false);
        errdefer s.pipes.cstore.deinit();
        s.pipes.kv = try mtl.Pipeline.init(device, lib, "i2p_kv_store", false);
        errdefer s.pipes.kv.deinit();
        s.pipes.rel = try mtl.Pipeline.init(device, lib, "i2p_relbias", false);
        errdefer s.pipes.rel.deinit();
        s.pipes.attn = try mtl.Pipeline.init(device, lib, "i2p_attn", false);
        errdefer s.pipes.attn.deinit();
        s.pipes.attn_tiled = try mtl.Pipeline.init(device, lib, "i2p_attn_tiled", false);
        errdefer s.pipes.attn_tiled.deinit();
        s.pipes.gu_nv = try mtl.Pipeline.init(device, lib, "i2p_gu", false);
        errdefer s.pipes.gu_nv.deinit();
        s.pipes.down_nv = try mtl.Pipeline.init(device, lib, "i2p_down", false);
        errdefer s.pipes.down_nv.deinit();
        s.pipes.gu_mx = try mtl.Pipeline.init(device, libmx, "i2p_gu", false);
        errdefer s.pipes.gu_mx.deinit();
        s.pipes.down_mx = try mtl.Pipeline.init(device, libmx, "i2p_down", false);
        errdefer s.pipes.down_mx.deinit();
        s.pipes.gu_g_nv = try mtl.Pipeline.init(device, lib, "i2p_gu_grouped", false);
        errdefer s.pipes.gu_g_nv.deinit();
        s.pipes.down_g_nv = try mtl.Pipeline.init(device, lib, "i2p_down_grouped", false);
        errdefer s.pipes.down_g_nv.deinit();
        s.pipes.gu_g_mx = try mtl.Pipeline.init(device, libmx, "i2p_gu_grouped", false);
        errdefer s.pipes.gu_g_mx.deinit();
        s.pipes.down_g_mx = try mtl.Pipeline.init(device, libmx, "i2p_down_grouped", false);
        errdefer s.pipes.down_g_mx.deinit();
        s.pipes.expert_matrix_nv = try mtl.Pipeline.init(device, lib, "i2p_expert_matrix", false);
        errdefer s.pipes.expert_matrix_nv.deinit();
        s.pipes.expert_matrix_mx = try mtl.Pipeline.init(device, libmx, "i2p_expert_matrix", false);
        errdefer s.pipes.expert_matrix_mx.deinit();
        s.pipes.moe_zero = try mtl.Pipeline.init(device, lib, "i2p_moe_zero", false);
        errdefer s.pipes.moe_zero.deinit();
        s.pipes.moe_count = try mtl.Pipeline.init(device, lib, "i2p_moe_count", false);
        errdefer s.pipes.moe_count.deinit();
        s.pipes.moe_offsets = try mtl.Pipeline.init(device, lib, "i2p_moe_offsets", false);
        errdefer s.pipes.moe_offsets.deinit();
        s.pipes.moe_sort = try mtl.Pipeline.init(device, lib, "i2p_moe_sort", false);
        errdefer s.pipes.moe_sort.deinit();
        s.pipes.combine = try mtl.Pipeline.init(device, lib, "i2p_combine", false);
        return s;
    }

    pub fn deinit(s: *PrefillScratch) void {
        s.pipes.deinit();
        inline for ([_][]const u8{ "ids", "x", "prelude", "passout", "bank", "ffnin", "n0", "n1", "n2", "m0", "q", "k", "v", "r", "kc", "vc", "kdyn", "vdyn", "sink", "relbias", "attn", "gate", "projout", "o", "aconv", "moe", "blocks", "staged", "mixout", "msg", "h2in", "normed", "tanhg", "tanhf", "router", "picks", "wts", "act", "down", "moe_cnt", "moe_off", "moe_cur", "moe_sorted", "logits" }) |f| @field(s, f).deinit();
    }
};

const In = struct { buf: B, xs: u32, xb: u32 };

const Chunk = struct {
    m: *const Model,
    s: *PrefillScratch,
    cache: *Cache,
    enc: mtl.ComputeEncoder,
    n: u32,
    base: u32,
    h: u32,
    qw: u32,
    kvw: u32,
    heads: u32,
    kv: u32,
    dim: u32,
    drel: u32,
    ring: bool = false,
    count: usize = 0,

    fn isLocal(c: Chunk, slot: usize) bool {
        const m = c.m;
        const phys = m.config.loop.physicalLayer(m.config.text.layers, @intCast(slot));
        return !m.config.isFull(phys);
    }

    fn slotBase(c: Chunk, slot: usize) usize {
        const m = c.m;
        if (!c.ring) return c.cache.row(&m.config, slot, 0);
        const kvw: usize = m.config.text.local.kv * m.config.text.local.dim;
        var base: usize = 0;
        for (0..slot) |s| {
            const phys = m.config.loop.physicalLayer(m.config.text.layers, @intCast(s));
            const rows: usize = if (!m.config.isFull(phys)) m.config.text.window else c.cache.capacity;
            base += rows * kvw * 4;
        }
        return base;
    }

    fn slotRing(c: Chunk, slot: usize) u32 {
        if (!c.ring) return 0;
        if (!c.isLocal(slot)) return 0;
        return c.m.config.text.window;
    }

    fn setP(c: Chunk, pipe: mtl.Pipeline, args: anytype, bufs: []const B) void {
        c.enc.setPipeline(pipe);
        for (bufs, 0..) |buf, i| {
            if (buf.args) c.enc.setValue(args, i) else c.enc.setBuffer(buf.buf, buf.off, i);
        }
    }

    fn setG(c: Chunk, g: wts.Glue, args: anytype, bufs: []const B) void {
        c.setP(c.m.pipe(g), args, bufs);
    }

    fn lin(c: Chunk, w: Tensor, x: In, out: B) void {
        const m: usize = nrows(w);
        const args = Gemm{ .n = @intCast(m), .k = @intCast(width(w)), .rows = c.n, .xs = x.xs, .xb = x.xb, .cap = 0 };
        if (c.n <= 8) {
            c.setP(c.s.pipes.smallrow, args, &.{ x.buf, t(w), A, out });
            c.enc.dispatchGroups(mtl.Size.of((m + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        } else if (c.s.pipes.gemm_mpp != null and m >= 384 and args.k >= 2304 and c.n >= 16) {
            c.setP(c.s.pipes.gemm_mpp.?, args, &.{ x.buf, t(w), A, out });
            c.enc.dispatchGroups(mtl.Size.of((m + 31) / 32, (c.n + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        } else {
            c.setP(c.s.pipes.gemm, args, &.{ x.buf, t(w), A, out });
            c.enc.dispatchGroups(mtl.Size.of((m + 63) / 64, (c.n + 31) / 32, 1), mtl.Size.of(256, 1, 1));
        }
    }

    fn headModel(c: Chunk, w: Tensor, x: In, out: B, vocab: u32) void {
        const args = Gemm{ .n = vocab, .k = @intCast(width(w)), .rows = c.n, .xs = x.xs, .xb = x.xb, .cap = c.m.config.text.logit_cap };
        if (c.n <= 8) {
            c.setP(c.s.pipes.smallrow, args, &.{ x.buf, t(w), A, out });
            c.enc.dispatchGroups(mtl.Size.of((vocab + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        } else if (c.s.pipes.gemm_mpp != null and c.n >= 16 and args.k >= 2304) {
            c.setP(c.s.pipes.gemm_mpp.?, args, &.{ x.buf, t(w), A, out });
            c.enc.dispatchGroups(mtl.Size.of((vocab + 31) / 32, (c.n + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        } else {
            c.setP(c.s.pipes.gemm, args, &.{ x.buf, t(w), A, out });
            c.enc.dispatchGroups(mtl.Size.of((vocab + 63) / 64, (c.n + 31) / 32, 1), mtl.Size.of(256, 1, 1));
        }
    }

    fn headModelLast(c: Chunk, w: Tensor, xoff: B, out: B, vocab: u32) void {
        c.setP(c.s.pipes.capexact, Cap{ .n = vocab, .k = @intCast(width(w)), .rows = 1, .cap = c.m.config.text.logit_cap }, &.{ xoff, t(w), A, out });
        c.enc.dispatchGroups(mtl.Size.of((vocab + 31) / 32, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    fn rmsW(c: Chunk, x: In, weight: B, out: B, ys: u32, yb: u32, w: u32, eps: f32) void {
        c.setP(c.s.pipes.rms, RmsB{ .width = w, .rows = c.n, .xs = x.xs, .xb = x.xb, .ys = ys, .yb = yb, .wb = 1, .eps = eps }, &.{ x.buf, weight, A, out });
        c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn rmsU2(c: Chunk, x: In, out: B, ys: u32, yb: u32, w: u32, eps: f32) void {
        c.setP(c.s.pipes.rms, RmsB{ .width = w, .rows = c.n, .xs = x.xs, .xb = x.xb, .ys = ys, .yb = yb, .wb = 0, .eps = eps }, &.{ x.buf, b(c.s.ids), A, out });
        c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn rmsU(c: Chunk, x: B, out: B, count: usize, w: u32, eps: f32) void {
        c.setG(.rms, Rms{ .width = w, .blocks = 0, .eps = eps }, &.{ x, out, A, out });
        c.enc.dispatchGroups(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn flat(c: Chunk, g: wts.Glue, args: anytype, bufs: []const B, count: usize) void {
        c.setG(g, args, bufs);
        c.enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn copyN(c: Chunk, x: B, out: B, count: usize) void {
        c.flat(.copy, Pair{ .count = @intCast(count) }, &.{ x, out, A }, count);
    }

    fn addN(c: Chunk, dst: B, src: B, count: usize) void {
        c.flat(.add, Pair{ .count = @intCast(count) }, &.{ dst, src, A }, count);
    }

    fn gatherM(c: Chunk, blk: B, cur: B, depth: u32) void {
        c.setP(c.s.pipes.gather, Gather{ .width = c.h, .depth = depth, .stride = st.mix_states }, &.{ blk, cur, A, b(c.s.staged) });
        c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn mixQ(c: Chunk, query: Tensor, out: B) void {
        const depth = c.count + 1;
        c.rmsU(b(c.s.staged), b(c.s.staged), @as(usize, c.n) * depth, c.h, 1e-6);
        const k: u32 = @intCast(nrows(query));
        c.setP(c.s.pipes.mix, MixB{ .width = c.h, .depth = @intCast(depth), .k = k, .rows = c.n, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.h))) }, &.{ b(c.s.staged), t(query), A, out });
        c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn mixBlocks(c: Chunk, cur: B, query: Tensor, out: B) void {
        c.gatherM(b(c.s.blocks), cur, @intCast(c.count + 1));
        c.mixQ(query, out);
    }

    fn pushBlock(c: *Chunk) void {
        for (0..c.n) |tt| {
            const src = bo(c.s.x, tt * c.h * 4);
            const dst = bo(c.s.blocks, (tt * st.mix_states + c.count) * c.h * 4);
            c.copyN(src, dst, c.h);
        }
        c.count += 1;
    }

    fn rings(c: Chunk, slot: u32) [4]B {
        const kv: usize = c.kvw;
        return .{
            bo(c.cache.k_conv, c.cache.convAt(slot, kv)),
            bo(c.cache.v_conv, c.cache.convAt(slot, kv)),
            bo(c.cache.attn_conv, c.cache.convAt(slot, c.h)),
            bo(c.cache.mlp_conv, c.cache.convAt(slot, c.h)),
        };
    }

    fn convolve(c: Chunk, x: B, out: B, weight: Tensor, dyn: ?Tensor, dyn_in: In, dyn_out: B, ring: B, channels: u32, conv_heads: u32, layout: u32) void {
        const taps: u32 = cfg.conv_taps;
        const mul: u32 = if (dyn != null) 2 else 1;
        const norm: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(mul * taps + 1)));
        if (dyn) |dw| c.lin(dw, dyn_in, dyn_out);
        c.setP(c.s.pipes.conv, ConvB{ .channels = channels, .heads = conv_heads, .ring = st.ring, .base = c.base, .rows = c.n, .layout = layout, .norm = norm }, &.{ x, ring, t(weight), if (dyn != null) dyn_out else A, A, out });
        c.enc.dispatchThreads(mtl.Size.of(channels, c.n, 1), mtl.Size.of(256, 1, 1));
        c.setP(c.s.pipes.cstore, CStore{ .channels = channels, .ring = st.ring, .base = c.base, .rows = c.n }, &.{ x, ring, A });
        c.enc.dispatchThreads(mtl.Size.of(channels, c.n, 1), mtl.Size.of(256, 1, 1));
    }

    fn kvStore(c: Chunk, slot: u32) void {
        const at = c.slotBase(slot);
        c.setP(c.s.pipes.kv, KvB{ .count = c.kvw, .base = c.base, .rows = c.n, .wring = c.slotRing(slot) }, &.{ b(c.s.kc), b(c.s.vc), bo(c.cache.keys, at), bo(c.cache.values, at), A });
        c.enc.dispatchThreads(mtl.Size.of(c.kvw, c.n, 1), mtl.Size.of(256, 1, 1));
    }

    fn clipScaled(c: Chunk, x: B, scale: Tensor, heads: u32, w: u32, threshold: f32) void {
        c.setP(c.s.pipes.clips, ClipB{ .heads = heads, .width = w, .rows = c.n, .threshold = threshold }, &.{ x, t(scale), A, x });
        c.enc.dispatchGroups(mtl.Size.of(heads * c.n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn clipPlain(c: Chunk, x: B, heads: u32, w: u32, threshold: f32) void {
        c.setP(c.s.pipes.clip, ClipB{ .heads = heads, .width = w, .rows = c.n, .threshold = threshold }, &.{ x, A, x });
        c.enc.dispatchGroups(mtl.Size.of(heads * c.n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn relative(c: Chunk, lay: *const Layer, r: B, extent: u32, full: bool) void {
        const text = &c.m.config.text;
        c.setP(c.s.pipes.rel, RelB{ .heads = c.heads, .rel = c.drel, .extent = extent, .rows = c.n, .base = c.base, .tau = @intFromBool(full), .floor = @floatFromInt(text.log_floor), .alpha = text.log_alpha }, &.{ t(lay.attn.rel_bias), r, A, b(c.s.relbias) });
        c.enc.dispatchThreads(mtl.Size.of(c.heads * (extent + 1), c.n, 1), mtl.Size.of(256, 1, 1));
    }

    fn control(c: Chunk, ctl: Controller, cur: In, msg: In, out: B, ys: u32, yb: u32) void {
        const s = c.s;
        c.rmsU2(cur, b(s.h2in), 2, 0, c.h, 1e-6);
        c.rmsU2(msg, b(s.h2in), 2, 1, c.h, 1e-6);
        c.lin(ctl.down, .{ .buf = b(s.h2in), .xs = 1, .xb = 0 }, b(s.normed));
        c.lin(ctl.gain, .{ .buf = b(s.normed), .xs = 1, .xb = 0 }, b(s.tanhg));
        c.lin(ctl.forget, .{ .buf = b(s.normed), .xs = 1, .xb = 0 }, b(s.tanhf));
        c.setP(s.pipes.ctl, CtlB{ .h = c.h, .rows = c.n, .cxs = cur.xs, .cxb = cur.xb, .mxs = msg.xs, .mxb = msg.xb, .ys = ys, .yb = yb }, &.{ cur.buf, msg.buf, b(s.tanhg), b(s.tanhf), A, out });
        c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
    }

    fn messages(c: Chunk, bk: Bank, ip: u32) B {
        const m = c.m;
        const s = c.s;
        const at: usize = ip - m.config.loop.core_start;
        c.gatherM(bk.states, bk.out, @intCast(bk.n + 1));
        c.rmsU(b(s.staged), b(s.staged), @as(usize, c.n) * (bk.n + 1), c.h, 1e-6);
        c.setP(s.pipes.mix4, MixB{ .width = c.h, .depth = @intCast(bk.n + 1), .k = 4, .rows = c.n, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.h))) }, &.{
            b(s.staged),
            t(srows(m.weights.cross_pass_q, at * 2, 1)),
            t(srows(m.weights.cross_pass_kv, at * 2, 1)),
            t(srows(m.weights.cross_pass_kv, at * 2 + 1, 1)),
            t(srows(m.weights.cross_pass_q, at * 2 + 1, 1)),
            A,
            b(s.msg),
        });
        c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(256, 1, 1));
        return b(s.msg);
    }

    fn attention(c: Chunk, lay: *const Layer, slot: u32, full: bool, ins: [3]In) void {
        const s = c.s;
        const text = &c.m.config.text;
        const extent: u32 = if (full) text.rel_extent else text.window;
        c.lin(lay.attn.q, ins[0], b(s.q));
        c.clipScaled(b(s.q), lay.attn.q_scale, c.heads, c.dim, text.qk_clip);
        c.lin(lay.attn.r, ins[0], b(s.r));
        c.clipScaled(b(s.r), lay.attn.r_scale, c.heads, c.drel, text.rel_clip);
        const ring = c.rings(slot);
        c.lin(lay.attn.k, ins[1], b(s.k));
        c.convolve(b(s.k), b(s.kc), lay.attn.k_conv, lay.attn.k_dyn, ins[1], b(s.kdyn), ring[0], c.kvw, if (full) c.kv else 0, @intFromBool(full));
        c.clipPlain(b(s.kc), c.kv, c.dim, text.qk_clip);
        c.lin(lay.attn.v, ins[2], b(s.v));
        c.convolve(b(s.v), b(s.vc), lay.attn.v_conv, lay.attn.v_dyn, ins[2], b(s.vdyn), ring[1], c.kvw, if (full) c.kv else 0, @intFromBool(full));
        c.rmsU(b(s.vc), b(s.vc), @as(usize, c.n) * c.kv, c.dim, text.eps);
        c.lin(lay.attn.sink, ins[0], b(s.sink));
        c.relative(lay, b(s.r), extent, full);
        const win: u32 = if (full) 0 else text.window;
        const keybase = bo(c.cache.keys, c.slotBase(slot));
        const valbase = bo(c.cache.values, c.slotBase(slot));
        const tiled = c.heads == 24 and c.kv == 8 and c.dim == 128;
        c.setP(if (tiled) s.pipes.attn_tiled else s.pipes.attn, AttnB{ .heads = c.heads, .kv = c.kv, .dim = c.dim, .rel = c.drel, .extent = extent, .window = win, .base = c.base, .rows = c.n, .rmode = @intFromBool(c.ring), .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.dim))), .floor = @floatFromInt(text.log_floor), .alpha = text.log_alpha }, &.{ b(s.q), keybase, valbase, b(s.relbias), b(s.sink), A, b(s.attn), b(s.kc), b(s.vc) });
        if (tiled) c.enc.dispatchGroups(mtl.Size.of(c.kv, (c.n + 7) / 8, 1), mtl.Size.of(768, 1, 1)) else c.enc.dispatchGroups(mtl.Size.of(c.heads, c.n, 1), mtl.Size.of(32, 1, 1));
        c.kvStore(slot);
        c.lin(lay.attn.gate, ins[0], b(s.gate));
        c.flat(.sig_mul, Pair{ .count = c.qw * c.n }, &.{ b(s.attn), b(s.gate), A, b(s.attn) }, @as(usize, c.qw) * c.n);
        c.lin(lay.attn.o, .{ .buf = b(s.attn), .xs = 1, .xb = 0 }, b(s.projout));
        c.convolve(b(s.projout), b(s.o), lay.attn_post, null, undefined, b(s.projout), ring[2], c.h, 0, 0);
    }

    fn experts(c: Chunk, lay: *const Layer, input: B, slot: u32) void {
        const m = c.m;
        const cc = &m.config;
        const s = c.s;
        const E: u32 = cc.text.experts;
        const K: u32 = cc.text.topk;
        const inter: u32 = cc.text.expert_inter;
        c.setP(s.pipes.router, RouterA{ .n = E, .k = c.h, .rows = c.n, .xs = 1, .xb = 0 }, &.{ input, t(lay.router), A, b(s.router) });
        c.enc.dispatchGroups(mtl.Size.of((E + 31) / 32, c.n, 1), mtl.Size.of(1024, 1, 1));
        c.setP(s.pipes.pick, Pick{ .experts = E, .topk = K, .rows = c.n, .route_scale = cc.text.route_scale }, &.{ b(s.router), t(lay.router_bias), A, b(s.picks), b(s.wts) });
        c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(32, 1, 1));
        if (s.capture) |capture| {
            c.copyN(bo(s.picks, s.capture_row * K * 4), bo(capture, (cc.slots() * st.trace_rows * c.h + slot * K) * 4), K);
            const start = cc.slots() * (st.trace_rows * c.h + K) + slot * E * 2;
            c.copyN(bo(s.router, s.capture_row * E * 4), bo(capture, start * 4), E);
            c.copyN(t(lay.router_bias), bo(capture, (start + E) * 4), E);
        }
        if (s.routes) |routes| {
            const start = @as(usize, slot) * (@as(usize, c.n) * (K + E) + E);
            c.copyN(b(s.picks), bo(routes, start * 4), @as(usize, c.n) * K);
            c.copyN(b(s.router), bo(routes, (start + @as(usize, c.n) * K) * 4), @as(usize, c.n) * E);
            c.copyN(t(lay.router_bias), bo(routes, (start + @as(usize, c.n) * (K + E)) * 4), E);
        }
        const slots: u32 = c.n * K;
        if (c.n <= 8) {
            const codes = b(if (lay.mxfp8) m.t256 else m.t16);
            const gu = if (lay.mxfp8) s.pipes.gu_mx else s.pipes.gu_nv;
            const down = if (lay.mxfp8) s.pipes.down_mx else s.pipes.down_nv;
            c.setP(gu, MoeB{ .k = c.h, .n = inter, .slots = slots, .topk = K, .gate_cap = cc.text.gate_cap, .up_cap = cc.text.up_cap }, &.{ input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.picks), A, b(s.act), codes });
            c.enc.dispatchGroups(mtl.Size.of(inter, slots, 1), mtl.Size.of(32, 1, 1));
            c.setP(down, MoeB{ .k = inter, .n = c.h, .slots = slots, .topk = K, .gate_cap = 0, .up_cap = 0 }, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.picks), A, b(s.down), codes });
            c.enc.dispatchGroups(mtl.Size.of(c.h, slots, 1), mtl.Size.of(32, 1, 1));
            c.setP(s.pipes.combine, CombineB{ .width = c.h, .topk = K, .rows = c.n }, &.{ b(s.down), b(s.wts), A, b(s.moe) });
            c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        c.setP(s.pipes.moe_zero, E, &.{ b(s.moe_cnt), A });
        c.enc.dispatchThreads(mtl.Size.of(E, 1, 1), mtl.Size.of(32, 1, 1));
        c.setP(s.pipes.moe_count, slots, &.{ b(s.picks), b(s.moe_cnt), A });
        c.enc.dispatchThreads(mtl.Size.of(slots, 1, 1), mtl.Size.of(32, 1, 1));
        c.setP(s.pipes.moe_offsets, E, &.{ b(s.moe_cnt), b(s.moe_off), b(s.moe_cur), A });
        c.enc.dispatchThreads(mtl.Size.of(128, 1, 1), mtl.Size.of(128, 1, 1));
        c.setP(s.pipes.moe_sort, slots, &.{ b(s.picks), b(s.moe_cur), b(s.moe_sorted), A });
        c.enc.dispatchThreads(mtl.Size.of(slots, 1, 1), mtl.Size.of(32, 1, 1));
        const codes = b(if (lay.mxfp8) m.t256 else m.t16);
        const gu = if (lay.mxfp8) s.pipes.gu_g_mx else s.pipes.gu_g_nv;
        const down = if (lay.mxfp8) s.pipes.down_g_mx else s.pipes.down_g_nv;
        const matrix = if (lay.mxfp8) s.pipes.expert_matrix_mx else s.pipes.expert_matrix_nv;
        c.setP(gu, Moeg{ .k = c.h, .n = inter, .slots = slots, .topk = K, .experts = E, .gate_cap = cc.text.gate_cap, .up_cap = cc.text.up_cap }, &.{ input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.moe_off), b(s.moe_sorted), A, b(s.act), codes });
        c.enc.dispatchGroups(mtl.Size.of(inter, E, 1), mtl.Size.of(32, 1, 1));
        c.setP(matrix, Moeg{ .k = c.h, .n = inter, .slots = slots, .topk = K, .experts = E, .gate_cap = cc.text.gate_cap, .up_cap = cc.text.up_cap }, &.{ input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.moe_off), b(s.moe_sorted), A, b(s.act), codes });
        c.enc.dispatchGroups(mtl.Size.of((inter * 2 + 63) / 64, E, (c.n + 31) / 32), mtl.Size.of(256, 1, 1));
        c.setP(down, Moeg{ .k = inter, .n = c.h, .slots = slots, .topk = K, .experts = E, .gate_cap = 0, .up_cap = 0 }, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.moe_off), b(s.moe_sorted), A, b(s.down), codes });
        c.enc.dispatchGroups(mtl.Size.of(c.h, E, 1), mtl.Size.of(32, 1, 1));
        c.setP(matrix, Moeg{ .k = inter, .n = c.h, .slots = slots, .topk = K, .experts = E, .gate_cap = 0, .up_cap = 0 }, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.moe_off), b(s.moe_sorted), A, b(s.down), codes });
        c.enc.dispatchGroups(mtl.Size.of((c.h + 63) / 64, E, (c.n + 31) / 32), mtl.Size.of(256, 1, 1));
        c.setP(s.pipes.combine, CombineB{ .width = c.h, .topk = K, .rows = c.n }, &.{ b(s.down), b(s.wts), A, b(s.moe) });
        c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
    }

    fn layer(c: *Chunk, slot: u32, ip: u32, bank: ?Bank) !void {
        const m = c.m;
        const config = &m.config;
        const text = &config.text;
        const index = config.loop.physicalLayer(text.layers, slot);
        const lay: *const Layer = &m.weights.layers[index];
        const full = config.isFull(index);
        const s = c.s;

        var ins: [3]In = .{ .{ .buf = b(s.x), .xs = 1, .xb = 0 }, .{ .buf = b(s.x), .xs = 1, .xb = 0 }, .{ .buf = b(s.x), .xs = 1, .xb = 0 } };
        if (index > 0) {
            const q = m.weights.attn_queries[index - 1];
            c.mixBlocks(b(s.x), q, bo(s.mixout, 0));
            const k: u32 = @intCast(nrows(q));
            if (full) {
                ins = .{ .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 0 }, .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 1 }, .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 2 } };
            } else {
                ins = .{ .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 0 }, .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 0 }, .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 0 } };
            }
        }

        if (bank) |bk| {
            const msg = c.messages(bk, ip);
            for (0..3) |j| {
                const ctl = lay.controllers[j] orelse return error.MissingController;
                const nbuf = [_]B{ b(s.n0), b(s.n1), b(s.n2) };
                c.control(ctl, ins[j], .{ .buf = msg, .xs = 4, .xb = @intCast(j) }, nbuf[j], 1, 0);
                ins[j] = .{ .buf = nbuf[j], .xs = 1, .xb = 0 };
            }
        }

        for (0..3) |j| c.record(slot, j, ins[j]);
        if (full) {
            c.rmsW(ins[0], t(lay.q_norm), b(s.n0), 1, 0, c.h, text.eps);
            c.rmsW(ins[1], t(lay.k_norm), b(s.n1), 1, 0, c.h, text.eps);
            c.rmsW(ins[2], t(lay.v_norm), b(s.n2), 1, 0, c.h, text.eps);
            ins = .{ .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n1), .xs = 1, .xb = 0 }, .{ .buf = b(s.n2), .xs = 1, .xb = 0 } };
        } else if (bank != null) {
            c.rmsW(ins[0], t(lay.q_norm), b(s.n0), 1, 0, c.h, text.eps);
            c.rmsW(ins[1], t(lay.q_norm), b(s.n1), 1, 0, c.h, text.eps);
            c.rmsW(ins[2], t(lay.q_norm), b(s.n2), 1, 0, c.h, text.eps);
            ins = .{ .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n1), .xs = 1, .xb = 0 }, .{ .buf = b(s.n2), .xs = 1, .xb = 0 } };
        } else {
            c.rmsW(ins[0], t(lay.q_norm), b(s.n0), 1, 0, c.h, text.eps);
            ins = .{ .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n0), .xs = 1, .xb = 0 } };
        }

        for (0..3) |j| c.record(slot, 3 + j, ins[j]);
        c.attention(lay, slot, full, ins);
        c.record(slot, 6, .{ .buf = b(s.o), .xs = 1, .xb = 0 });
        if (ip % text.block == 0) {
            c.pushBlock();
            c.copyN(b(s.o), b(s.x), @as(usize, c.n) * c.h);
        } else c.addN(b(s.x), b(s.o), @as(usize, c.n) * c.h);

        c.copyN(b(s.x), b(s.ffnin), @as(usize, c.n) * c.h);
        c.record(slot, 7, .{ .buf = b(s.ffnin), .xs = 1, .xb = 0 });
        c.mixBlocks(b(s.x), m.weights.mlp_queries[index], bo(s.mixout, 0));
        if (bank) |_| {
            const ctl = lay.controllers[3] orelse return error.MissingController;
            c.control(ctl, .{ .buf = bo(s.mixout, 0), .xs = 1, .xb = 0 }, .{ .buf = b(s.msg), .xs = 4, .xb = 3 }, bo(s.mixout, 0), 1, 0);
        }
        c.rmsW(.{ .buf = bo(s.mixout, 0), .xs = 1, .xb = 0 }, t(lay.mlp_norm), b(s.m0), 1, 0, c.h, text.eps);
        c.experts(lay, b(s.m0), slot);
        const slots = c.rings(slot);
        c.convolve(b(s.moe), b(s.aconv), lay.mlp_post, null, undefined, b(s.moe), slots[3], c.h, 0, 0);
        c.copyN(b(s.ffnin), b(s.x), @as(usize, c.n) * c.h);
        c.addN(b(s.x), b(s.aconv), @as(usize, c.n) * c.h);
        c.record(slot, 8, .{ .buf = b(s.x), .xs = 1, .xb = 0 });
    }

    fn record(c: Chunk, slot: u32, which: usize, input: In) void {
        if (c.s.capture) |capture| {
            const src = bo(input.buf.buf, input.buf.off + (c.s.capture_row * input.xs + input.xb) * c.h * 4);
            c.copyN(src, bo(capture, (@as(usize, slot) * st.trace_rows + which) * c.h * 4), c.h);
        }
    }
};

const ProfileRounds = struct {
    queue: mtl.Queue,
    cb: mtl.CommandBuffer,
    enabled: bool,

    fn next(p: *ProfileRounds, c: *Chunk, name: []const u8) !void {
        if (!p.enabled) return;
        c.enc.end();
        p.cb.commit();
        p.cb.wait();
        if (p.cb.failure()) |msg| {
            std.log.err("inkling2_loop prefill: {s}", .{msg});
            return error.GpuFailed;
        }
        p.queue.roundEnd();
        p.queue.roundBegin(name);
        p.cb = p.queue.commandBuffer();
        c.enc = p.cb.compute(.serial);
    }
};

fn profileOccupancy(rounds: *ProfileRounds, c: *Chunk, slot: u32, name: []const u8) !void {
    const requested = std.c.getenv("TF_INKLING_PREFILL_OCCUPANCY_BASE") orelse return;
    const base = std.fmt.parseInt(usize, std.mem.span(requested), 10) catch return;
    if (!rounds.enabled or c.base != base) return;
    try rounds.next(c, name);
    std.debug.print("inkling2 occupancy base={d} slot={d} rows={d} counts=", .{ c.base, slot, c.n });
    for (c.s.moe_cnt.slice(u32, c.m.config.text.experts)) |count| std.debug.print("{d},", .{count});
    std.debug.print("\n", .{});
}

pub const Bank = struct { states: B, out: B, n: usize };

pub fn prefill(m: *const Model, s: *PrefillScratch, cache: *Cache, ids: []const u32) !void {
    return prefillWant(m, s, cache, ids, .all);
}

pub fn prefillWant(m: *const Model, s: *PrefillScratch, cache: *Cache, ids: []const u32, want: WantLogits) !void {
    if (ids.len == 0 or ids.len > s.chunk) return error.BadChunk;
    if (ids.len + cache.len > cache.capacity) return error.ContextFull;
    if (ids.len > m.config.text.window) return error.ChunkTooLarge;
    @memcpy(s.ids.slice(u32, ids.len), ids);
    const kvw: usize = m.config.text.local.kv * m.config.text.local.dim;
    const linear: usize = m.config.slots() * cache.capacity * kvw * 4;
    const ring = cache.keys.length() != linear;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    m.queue.roundBegin("inkling2-prefill");
    defer m.queue.roundEnd();
    const threshold = if (std.c.getenv("TF_INKLING_PREFILL_PROFILE_BASE")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0 else null;
    const profiling = if (threshold) |base| cache.len >= base else false;
    var rounds = ProfileRounds{ .queue = m.queue, .cb = m.queue.commandBuffer(), .enabled = profiling };
    const config = &m.config;
    const text = &config.text;
    var c = Chunk{
        .m = m,
        .s = s,
        .cache = cache,
        .enc = rounds.cb.compute(.serial),
        .n = @intCast(ids.len),
        .base = @intCast(cache.len),
        .h = s.h,
        .qw = text.local.q * text.local.dim,
        .kvw = text.local.kv * text.local.dim,
        .heads = text.local.q,
        .kv = text.local.kv,
        .dim = text.local.dim,
        .drel = text.d_rel,
        .ring = ring,
    };
    c.setP(s.pipes.embed, EmbedA{ .width = s.h, .rows = c.n, .eps = text.eps }, &.{ b(s.ids), t(m.weights.embed), t(m.weights.embed_norm), A, b(s.x) });
    c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(256, 1, 1));
    var profile_name: [64]u8 = undefined;
    for (0..config.loop.core_start) |i| {
        if (profiling) try rounds.next(&c, try std.fmt.bufPrint(&profile_name, "inkling2-slot-{d}", .{i}));
        try c.layer(@intCast(i), @intCast(i), null);
        try profileOccupancy(&rounds, &c, @intCast(i), "inkling2-occupancy");
    }
    const prelude = c.count;
    c.copyN(b(s.x), b(s.prelude), @as(usize, c.n) * c.h);
    const span = config.loop.core_end - config.loop.core_start;
    var pass: u32 = 0;
    while (pass < config.loop.passes) : (pass += 1) {
        var bk: ?Bank = null;
        var stored: Bank = undefined;
        if (pass > 0) {
            const bn = c.count - (prelude + 1);
            for (0..c.n) |tt| {
                if (profiling and tt % 16 == 0) try rounds.next(&c, try std.fmt.bufPrint(&profile_name, "inkling2-pass-{d}-copy-{d}", .{ pass, tt / 16 }));
                for (0..bn) |i| c.copyN(bo(s.blocks, (tt * st.mix_states + prelude + 1 + i) * c.h * 4), bo(s.bank, (tt * st.mix_states + i) * c.h * 4), c.h);
                c.copyN(bo(s.x, tt * c.h * 4), bo(s.passout, tt * c.h * 4), c.h);
            }
            stored = .{ .states = b(s.bank), .out = b(s.passout), .n = bn };
            bk = stored;
            c.count = prelude;
            c.copyN(b(s.prelude), b(s.x), @as(usize, c.n) * c.h);
        }
        for (config.loop.core_start..config.loop.core_end) |ip| {
            const slot: u32 = if (pass == 0) @intCast(ip) else @intCast(text.layers + (pass - 1) * span + (ip - config.loop.core_start));
            if (profiling) try rounds.next(&c, try std.fmt.bufPrint(&profile_name, "inkling2-slot-{d}", .{slot}));
            try c.layer(slot, @intCast(ip), bk);
            try profileOccupancy(&rounds, &c, slot, "inkling2-occupancy");
        }
    }
    for (config.loop.core_end..text.layers) |i| {
        if (profiling) try rounds.next(&c, try std.fmt.bufPrint(&profile_name, "inkling2-slot-{d}", .{i}));
        try c.layer(@intCast(i), @intCast(i), null);
        try profileOccupancy(&rounds, &c, @intCast(i), "inkling2-occupancy");
    }
    if (profiling) try rounds.next(&c, "inkling2-head");
    c.mixBlocks(b(s.x), m.weights.output_query, bo(s.mixout, 0));
    c.rmsW(.{ .buf = bo(s.mixout, 0), .xs = 1, .xb = 0 }, t(m.weights.norm), b(s.n0), 1, 0, c.h, text.eps);
    if (profiling) try rounds.next(&c, "inkling2-vocabulary");
    switch (want) {
        .none => {},
        .last => c.headModelLast(m.weights.unembed, bo(s.n0, (@as(usize, c.n) - 1) * c.h * 4), b(s.logits), text.unpadded_vocab),
        .all => c.headModel(m.weights.unembed, .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, b(s.logits), text.unpadded_vocab),
    }
    c.enc.end();
    rounds.cb.commit();
    rounds.cb.wait();
    if (rounds.cb.failure()) |msg| {
        std.log.err("inkling2_loop prefill: {s}", .{msg});
        return error.GpuFailed;
    }
    cache.len += ids.len;
}
