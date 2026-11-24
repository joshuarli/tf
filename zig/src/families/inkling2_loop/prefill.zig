const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const wts = @import("model.zig");
const kernels = @import("kernels.zig");
const staged_mod = @import("prefill_staged.zig");
const rows_mod = @import("prefill_rows.zig");

const Model = wts.Model;
const Dispatch = @import("rows.zig").Dispatch;
const Cache = st.Cache;
const Tensor = wts.Tensor;
const Layer = wts.Layer;
const Controller = wts.Controller;

pub const default_chunk: usize = 1024;

/// The half-operand expert library: gate/up matrices from 14 rows an expert, down from 8, grouped f32 kernels below.
pub const half_source = "#define I2P_HALF_OPERANDS 1\n#define I2P_MATRIX_THRESHOLD 14\n#define I2P_DOWN_MATRIX_THRESHOLD 8\n" ++ kernels.prefill ++ kernels.prefill_expert_wide;
/// The same on tensor units: the matrices there, and half activations from grouped gate/up for down's matrices.
pub const units_source = "#define I2P_TENSOR_UNITS 1\n#define I2P_HALF_ACT 1\n" ++ half_source ++ @embedFile("kernels/prefill_units.metal") ++ @embedFile("kernels/prefill_attn.metal");
const units_rows_source = "#define I2P_TENSOR_UNITS 1\n#define I2P_HALF_ACT 1\n#define I2P_HALF_OPERANDS 1\n#define I2P_MATRIX_THRESHOLD 1\n#define I2P_DOWN_MATRIX_THRESHOLD 1\n" ++ kernels.prefill ++ kernels.prefill_expert_wide ++ @embedFile("kernels/prefill_units.metal") ++ @embedFile("kernels/prefill_attn.metal");

pub fn profile(text: ?[]const u8) error{BadInklingProfile}!bool {
    const value = text orelse return false;
    if (std.mem.eql(u8, value, "f32")) return false;
    if (std.mem.eql(u8, value, "fast")) return true;
    return error.BadInklingProfile;
}

test "profile accepts only the documented values" {
    try std.testing.expectEqual(false, try profile(null));
    try std.testing.expectEqual(false, try profile("f32"));
    try std.testing.expectEqual(true, try profile("fast"));
    try std.testing.expectError(error.BadInklingProfile, profile(""));
    try std.testing.expectError(error.BadInklingProfile, profile("FAST"));
}

pub const WantLogits = enum { none, last, all };

const Dense = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32 };
const Cap = extern struct { n: u32, k: u32, rows: u32, cap: f32 };
const RouterA = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32 };
const Gemm = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, cap: f32 };
const Pick = extern struct { experts: u32, topk: u32, rows: u32, route_scale: f32 };
const EmbedA = extern struct { width: u32, rows: u32, eps: f32 };
const RmsB = extern struct { width: u32, rows: u32, xs: u32, xb: u32, ys: u32, yb: u32, wb: u32, eps: f32 };
const Rms3 = extern struct { width: u32, xs: [3]u32, xb: [3]u32, eps: f32 };
const PassCopy = extern struct { width: u32, stride: u32, start: u32, count: u32 };
const Scatter = extern struct { width: u32, stride: u32, slot: u32 };
const Gather = extern struct { width: u32, depth: u32, stride: u32 };
const MixB = extern struct { width: u32, depth: u32, k: u32, rows: u32, scale: f32 };
const ControlsArgs = extern struct { h: u32, rank: u32, rows: u32, xs: [3]u32, xb: [3]u32 };
const CtlB = extern struct { h: u32, rows: u32, cxs: u32, cxb: u32, mxs: u32, mxb: u32, ys: u32, yb: u32 };
const GateCtl = extern struct { h: u32, rows: u32, k: u32, cxs: u32, cxb: u32, mxs: u32, mxb: u32, ys: u32, yb: u32 };
const Clip3B = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, q_threshold: f32, rel_threshold: f32 };
const ClipB = extern struct { heads: u32, width: u32, rows: u32, threshold: f32 };
const ConvB = extern struct { channels: u32, heads: u32, ring: u32, base: u32, rows: u32, layout: u32, norm: f32 };
const CStore = extern struct { channels: u32, ring: u32, base: u32, rows: u32 };
const KvB = extern struct { count: u32, base: u32, rows: u32, wring: u32 };
const RelB = extern struct { heads: u32, rel: u32, extent: u32, rows: u32, base: u32, tau: u32, floor: f32, alpha: f32 };
const AttnB = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, base: u32, rows: u32, rmode: u32, scale: f32, floor: f32, alpha: f32 };
const DecodeMoe = extern struct { k: u32, n: u32, rows: u32, gate_cap: f32, up_cap: f32 };
const MoeB = extern struct { k: u32, n: u32, slots: u32, topk: u32, gate_cap: f32, up_cap: f32 };
const Moeg = extern struct { k: u32, n: u32, slots: u32, topk: u32, experts: u32, gate_cap: f32, up_cap: f32 };
const CombineB = extern struct { width: u32, topk: u32, rows: u32 };
const ConvRow = extern struct { ring: u64, pos: u32, pad: u32 = 0 };
const DecodeConv = extern struct { channels: u32, heads: u32, ring: u32, pos: u32, layout: u32, norm: f32, residual: u32 };
const DecodeAttn = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, scale: f32, floor: f32, alpha: f32, eps: f32 };
const DecodeParts = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, splits: u32, chunk: u32, scale: f32, floor: f32, alpha: f32 };
const BatchStoreRow = extern struct { keys: u64, values: u64, pos: u32, pad: u32 = 0 };
const BatchRow = extern struct { keys: u64, values: u64, args: DecodeParts };
const Rms = extern struct { width: u32, blocks: u32, eps: f32 };
const Pair = extern struct { count: u32 };
const AffineBatchControl = extern struct { k: u32, m0: u32, m1: u32, h: u32, cxs: u32, cxb: u32, mxs: u32, mxb: u32, ys: u32, yb: u32 };
const RmsPair = extern struct { width: u32, xs0: u32, xb0: u32, xs1: u32, xb1: u32, eps: f32 };
const AffineProjections = extern struct { k: u32, full: u32, m: [8]u32 };
const SharedProjections = extern struct { k: u32, full: u32, rows: u32, m: [8]u32 };
const AffineDense = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, bits: u32, cap: f32 };

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
    affine_half_rows: ?mtl.Pipeline,
    affine_stage_prepare: ?mtl.Pipeline,
    affine_stage_gemm: ?mtl.Pipeline,
    affine_stage_split: ?mtl.Pipeline,
    affine_stage_gemm_split: ?mtl.Pipeline,
    float_mpp: ?mtl.Pipeline,
    gate_ctl: ?mtl.Pipeline,
    affine_half_up: ?mtl.Pipeline,
    affine_half_down: ?mtl.Pipeline,
    bf16_convert: ?mtl.Pipeline,
    bf16_gemm_mpp: ?mtl.Pipeline,
    pick: mtl.Pipeline,
    embed: mtl.Pipeline,
    rms: mtl.Pipeline,
    rms3: mtl.Pipeline,
    rms_pair: mtl.Pipeline,
    gather_rms: mtl.Pipeline,
    scatter: mtl.Pipeline,
    pass_copy: mtl.Pipeline,
    mix: mtl.Pipeline,
    mix4: mtl.Pipeline,
    ctl: mtl.Pipeline,
    clips: mtl.Pipeline,
    clip3: mtl.Pipeline,
    clip: mtl.Pipeline,
    conv: mtl.Pipeline,
    batch_conv: mtl.Pipeline,
    batch_store_rel: mtl.Pipeline,
    cstore: mtl.Pipeline,
    kv: mtl.Pipeline,
    rel: mtl.Pipeline,
    rel_rows: mtl.Pipeline,
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
    half_attn: ?mtl.Pipeline,
    half_expert_nv: ?mtl.Pipeline,
    wide_expert_nv: ?mtl.Pipeline,
    wide_gu_nv: ?mtl.Pipeline,
    wide_down_nv: ?mtl.Pipeline,
    half_expert_mx: ?mtl.Pipeline,
    units_up: ?mtl.Pipeline,
    units_down: ?mtl.Pipeline,
    affine_units_up: ?mtl.Pipeline,
    affine_units_down: ?mtl.Pipeline,
    paired: [4]?mtl.Pipeline,
    attn_units: ?mtl.Pipeline,
    attn_units64: ?mtl.Pipeline,
    half_rows: ?mtl.Pipeline,
    staged: ?staged_mod.Pipes,
    rows: rows_mod.Pipes,
    moe_zero: mtl.Pipeline,
    moe_count: mtl.Pipeline,
    moe_offsets: mtl.Pipeline,
    moe_sort: mtl.Pipeline,
    combine: mtl.Pipeline,

    pub fn deinit(p: *Pipes) void {
        if (p.gemm_mpp) |pipe| pipe.deinit();
        if (p.affine_half_rows) |pipe| pipe.deinit();
        if (p.affine_stage_prepare) |pipe| pipe.deinit();
        if (p.affine_stage_gemm) |pipe| pipe.deinit();
        if (p.affine_stage_split) |pipe| pipe.deinit();
        if (p.affine_stage_gemm_split) |pipe| pipe.deinit();
        if (p.float_mpp) |pipe| pipe.deinit();
        if (p.gate_ctl) |pipe| pipe.deinit();
        if (p.affine_half_up) |pipe| pipe.deinit();
        if (p.affine_half_down) |pipe| pipe.deinit();
        if (p.bf16_convert) |pipe| pipe.deinit();
        if (p.bf16_gemm_mpp) |pipe| pipe.deinit();
        if (p.half_attn) |pipe| pipe.deinit();
        if (p.half_expert_nv) |pipe| pipe.deinit();
        if (p.wide_expert_nv) |pipe| pipe.deinit();
        if (p.wide_gu_nv) |pipe| pipe.deinit();
        if (p.wide_down_nv) |pipe| pipe.deinit();
        if (p.half_expert_mx) |pipe| pipe.deinit();
        if (p.units_up) |pipe| pipe.deinit();
        if (p.units_down) |pipe| pipe.deinit();
        if (p.affine_units_up) |pipe| pipe.deinit();
        if (p.affine_units_down) |pipe| pipe.deinit();
        for (p.paired) |paired| if (paired) |pipe| pipe.deinit();
        if (p.attn_units) |pipe| pipe.deinit();
        if (p.attn_units64) |pipe| pipe.deinit();
        if (p.half_rows) |pipe| pipe.deinit();
        if (p.staged) |pipes| pipes.deinit();
        p.rows.deinit();
        inline for ([_][]const u8{ "dense", "cap", "capexact", "router", "gemm", "smallrow", "pick", "gu_g_nv", "down_g_nv", "gu_g_mx", "down_g_mx", "expert_matrix_nv", "expert_matrix_mx", "moe_zero", "moe_count", "moe_offsets", "moe_sort", "embed", "rms", "rms3", "rms_pair", "gather_rms", "scatter", "pass_copy", "mix", "mix4", "ctl", "clips", "clip3", "clip", "conv", "batch_conv", "batch_store_rel", "cstore", "kv", "rel", "rel_rows", "attn", "attn_tiled", "gu_nv", "down_nv", "gu_mx", "down_mx", "combine" }) |f| @field(p, f).deinit();
    }
};

pub const Rounding = struct {
    experts: bool = false,
    inputs: bool = false,

    pub fn parse(text: ?[]const u8) error{BadInklingPrefillHalf}!Rounding {
        const value = text orelse return .{};
        if (std.mem.eql(u8, value, "experts")) return .{ .experts = true };
        if (std.mem.eql(u8, value, "inputs")) return .{ .inputs = true };
        if (std.mem.eql(u8, value, "both")) return .{ .experts = true, .inputs = true };
        return error.BadInklingPrefillHalf;
    }
};

test "half rounding accepts only the documented values" {
    try std.testing.expectEqual(Rounding{}, try Rounding.parse(null));
    try std.testing.expectEqual(Rounding{ .experts = true }, try Rounding.parse("experts"));
    try std.testing.expectEqual(Rounding{ .inputs = true }, try Rounding.parse("inputs"));
    try std.testing.expectEqual(Rounding{ .experts = true, .inputs = true }, try Rounding.parse("both"));
    try std.testing.expectError(error.BadInklingPrefillHalf, Rounding.parse(""));
    try std.testing.expectError(error.BadInklingPrefillHalf, Rounding.parse("Experts"));
}

pub const Tiles = struct {
    tall: bool = false,
    paired: bool = false,
    attn64: bool = false,

    pub const default: Tiles = .{ .paired = true, .attn64 = true };

    pub fn parse(text: ?[]const u8) error{BadInklingPrefillTile}!Tiles {
        const value = text orelse return default;
        var out: Tiles = .{};
        if (std.mem.eql(u8, value, "none")) return out;
        var words = std.mem.splitScalar(u8, value, ',');
        while (words.next()) |word| {
            if (std.mem.eql(u8, word, "tall")) {
                out.tall = true;
            } else if (std.mem.eql(u8, word, "paired")) {
                out.paired = true;
            } else if (std.mem.eql(u8, word, "attn64")) {
                out.attn64 = true;
            } else return error.BadInklingPrefillTile;
        }
        return out;
    }
};

test "tile options list the shapes in use" {
    try std.testing.expectEqual(Tiles.default, try Tiles.parse(null));
    try std.testing.expectEqual(Tiles{}, try Tiles.parse("none"));
    try std.testing.expectEqual(Tiles{ .tall = true, .attn64 = true }, try Tiles.parse("tall,attn64"));
    try std.testing.expectEqual(Tiles{ .paired = true }, try Tiles.parse("paired"));
    try std.testing.expectError(error.BadInklingPrefillTile, Tiles.parse(""));
    try std.testing.expectError(error.BadInklingPrefillTile, Tiles.parse("paired,"));
    try std.testing.expectError(error.BadInklingPrefillTile, Tiles.parse("paired,wide"));
}

pub const ConvSite = enum { key, value, attention, mlp };

pub const ConvCapture = struct {
    buffer: mtl.Buffer,
    rows: usize,
    kv_width: usize,
    hidden: usize,

    pub fn init(device: mtl.Device, config: *const cfg.Config, rows: usize) !ConvCapture {
        const kv_width: usize = config.text.local.kv * config.text.local.dim;
        const hidden: usize = config.text.hidden;
        const bytes = config.slots() * rows * (2 * kv_width + 2 * hidden) * 4;
        return .{ .buffer = try device.buffer(@max(bytes, 16), mtl.ResourceOptions.private), .rows = rows, .kv_width = kv_width, .hidden = hidden };
    }

    pub fn width(c: ConvCapture, site: ConvSite) usize {
        return switch (site) { .key, .value => c.kv_width, .attention, .mlp => c.hidden };
    }

    pub fn offset(c: ConvCapture, slot: usize, site: ConvSite, row: usize) usize {
        const site_width: usize = switch (site) { .key => 0, .value => c.kv_width, .attention => 2 * c.kv_width, .mlp => 2 * c.kv_width + c.hidden };
        return (slot * c.rows * (2 * c.kv_width + 2 * c.hidden) + site_width * c.rows + row * c.width(site)) * 4;
    }

    pub fn deinit(c: *ConvCapture) void {
        c.buffer.deinit();
    }
};

pub const PrefillScratch = struct {
    pipes: Pipes,
    chunk: usize,
    /// Rows of logits `.all` can return; a chunk asking for more is refused.
    logit_rows: usize,
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
    kh: mtl.Buffer,
    vh: mtl.Buffer,
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
    bf16_input: ?mtl.Buffer,
    half_overflow: ?mtl.Buffer,
    xh: ?mtl.Buffer,
    act_h: ?mtl.Buffer,
    affine_stage_w: ?mtl.Buffer,
    /// A chunk's staged-projection input as hi and lo halves, [2][rows][k].
    affine_stage_x: ?mtl.Buffer,
    affine_stage_extent: usize,
    fast: bool,
    half: bool,
    /// The half path's expert matrices, attention and staged projections on tensor units; off is the shader cores.
    units: bool,
    fuse: bool,
    rounding: Rounding,
    tiles: Tiles,
    capture: ?mtl.Buffer,
    capture_row: usize,
    routes: ?mtl.Buffer,
    verify_conv: ?*ConvCapture,
    taps: ?*const @import("taps.zig").Taps,
    gpu_seconds: f64,

    pub fn init(device: mtl.Device, c: *const cfg.Config, chunk: usize) !PrefillScratch {
        return initRows(device, c, chunk, chunk);
    }

    /// Scratch for chunks of up to `chunk` rows that keeps logits for `logit_rows` of them (1 for `.last` only).
    pub fn initRows(device: mtl.Device, c: *const cfg.Config, chunk: usize, logit_rows: usize) !PrefillScratch {
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
        const profile_env: ?[]const u8 = if (std.c.getenv("INKLING_PROFILE")) |v| std.mem.span(v) else null;
        const fast = try profile(profile_env);
        s.fast = fast;
        s.half = true;
        s.units = true;
        s.fuse = true;
        s.rounding = try Rounding.parse(if (std.c.getenv("INKLING_PREFILL_HALF")) |v| std.mem.span(v) else null);
        s.tiles = try Tiles.parse(if (std.c.getenv("INKLING_PREFILL_TILE")) |v| std.mem.span(v) else null);
        s.bf16_input = if (fast and device.tensorUnits()) try device.buffer(@max(chunk * @max(2 * h, @max(qw, @max(kvw, rank))) * 2, 16), mtl.ResourceOptions.private) else null;
        errdefer if (s.bf16_input) |buf| buf.deinit();
        s.half_overflow = if (fast or device.tensorUnits()) try device.buffer(16, mtl.ResourceOptions.shared) else null;
        errdefer if (s.half_overflow) |buf| buf.deinit();
        s.xh = if (device.tensorUnits()) try device.buffer(@max(chunk * h * 2, 16), mtl.ResourceOptions.private) else null;
        errdefer if (s.xh) |buf| buf.deinit();
        s.act_h = if (device.tensorUnits()) try device.buffer(@max(chunk * topk * inter * 2, 16), mtl.ResourceOptions.private) else null;
        errdefer if (s.act_h) |buf| buf.deinit();
        s.affine_stage_extent = @max(2 * h, @max(qw, @max(kvw, rank)));
        s.affine_stage_w = if (c.checkpoint == .mlx and device.tensorUnits()) try device.buffer(@max(s.affine_stage_extent * s.affine_stage_extent * 2, 16), mtl.ResourceOptions.private) else null;
        errdefer if (s.affine_stage_w) |buf| buf.deinit();
        s.affine_stage_x = if (s.affine_stage_w != null) try device.buffer(@max(chunk * s.affine_stage_extent * 4, 16), mtl.ResourceOptions.private) else null;
        errdefer if (s.affine_stage_x) |buf| buf.deinit();
        s.capture = null;
        s.capture_row = 0;
        s.routes = null;
        s.verify_conv = null;
        s.taps = null;
        s.gpu_seconds = 0;
        s.chunk = chunk;
        s.logit_rows = @min(@max(logit_rows, 1), chunk);
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
            .{ .ptr = &s.kh, .bytes = chunk * kvw * 2 },
            .{ .ptr = &s.vh, .bytes = chunk * kvw * 2 },
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
        s.logits = try device.buffer(@max(s.logit_rows * vocab * 4, 16), mtl.ResourceOptions.shared);
        errdefer s.logits.deinit();
        try s.commit(device);
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
        s.pipes.gate_ctl = null;
        s.pipes.float_mpp = if (s.affine_stage_w != null) blk: {
            const float_lib = try mtl.Library.fromSource(device, kernels.float_mpp, opts);
            defer float_lib.deinit();
            s.pipes.gate_ctl = try mtl.Pipeline.init(device, float_lib, "i2af_gate_ctl", false);
            errdefer s.pipes.gate_ctl.?.deinit();
            break :blk try mtl.Pipeline.init(device, float_lib, "i2af_gemm", false);
        } else null;
        errdefer if (s.pipes.float_mpp) |pipe| pipe.deinit();
        errdefer if (s.pipes.gate_ctl) |pipe| pipe.deinit();
        s.pipes.affine_stage_prepare = null;
        s.pipes.affine_stage_gemm = null;
        s.pipes.affine_stage_split = null;
        s.pipes.affine_stage_gemm_split = null;
        if (s.affine_stage_w != null) {
            const stage_lib = try mtl.Library.fromSource(device, kernels.affine_stage, opts);
            defer stage_lib.deinit();
            s.pipes.affine_stage_prepare = try mtl.Pipeline.init(device, stage_lib, "i2as_prepare", false);
            errdefer s.pipes.affine_stage_prepare.?.deinit();
            s.pipes.affine_stage_gemm = try mtl.Pipeline.init(device, stage_lib, "i2as_gemm", false);
            errdefer s.pipes.affine_stage_gemm.?.deinit();
            s.pipes.affine_stage_split = try mtl.Pipeline.init(device, stage_lib, "i2as_split", false);
            errdefer s.pipes.affine_stage_split.?.deinit();
            s.pipes.affine_stage_gemm_split = try mtl.Pipeline.init(device, stage_lib, "i2as_gemm_split", false);
        }
        errdefer if (s.pipes.affine_stage_prepare) |pipe| pipe.deinit();
        errdefer if (s.pipes.affine_stage_gemm) |pipe| pipe.deinit();
        errdefer if (s.pipes.affine_stage_split) |pipe| pipe.deinit();
        errdefer if (s.pipes.affine_stage_gemm_split) |pipe| pipe.deinit();
        s.pipes.staged = if (s.affine_stage_w != null) try staged_mod.Pipes.init(device) else null;
        errdefer if (s.pipes.staged) |pipes| pipes.deinit();
        s.pipes.rows = try rows_mod.Pipes.init(device);
        errdefer s.pipes.rows.deinit();
        s.pipes.affine_half_rows = null;
        s.pipes.affine_half_up = null;
        s.pipes.affine_half_down = null;
        if (device.tensorUnits()) {
            const affine_lib = try mtl.Library.fromSource(device, "#define I2A_HALF_OPERANDS 1\n" ++ kernels.affine, opts);
            defer affine_lib.deinit();
            s.pipes.affine_half_rows = try mtl.Pipeline.init(device, affine_lib, "i2a_dense_half_rows", false);
            errdefer s.pipes.affine_half_rows.?.deinit();
            s.pipes.affine_half_up = try mtl.Pipeline.init(device, affine_lib, "i2a_half_group_up", false);
            errdefer s.pipes.affine_half_up.?.deinit();
            s.pipes.affine_half_down = try mtl.Pipeline.init(device, affine_lib, "i2a_half_group_down", false);
        }
        errdefer if (s.pipes.affine_half_rows) |pipe| pipe.deinit();
        errdefer if (s.pipes.affine_half_up) |pipe| pipe.deinit();
        errdefer if (s.pipes.affine_half_down) |pipe| pipe.deinit();
        s.pipes.bf16_convert = null;
        s.pipes.bf16_gemm_mpp = null;
        s.pipes.gemm_mpp = if (device.tensorUnits()) blk: {
            const mpp_lib = try mtl.Library.fromSource(device, kernels.prefill_gemm_mpp, opts);
            defer mpp_lib.deinit();
            if (fast) {
                s.pipes.bf16_convert = try mtl.Pipeline.init(device, mpp_lib, "i2p_bf16_convert", false);
                errdefer s.pipes.bf16_convert.?.deinit();
                s.pipes.bf16_gemm_mpp = try mtl.Pipeline.init(device, mpp_lib, "i2p_gemm_bf16_mpp", false);
                errdefer s.pipes.bf16_gemm_mpp.?.deinit();
            }
            break :blk try mtl.Pipeline.init(device, mpp_lib, "i2p_gemm_mpp", false);
        } else null;
        errdefer if (s.pipes.gemm_mpp) |pipe| pipe.deinit();
        errdefer if (s.pipes.bf16_convert) |pipe| pipe.deinit();
        errdefer if (s.pipes.bf16_gemm_mpp) |pipe| pipe.deinit();
        s.pipes.pick = try mtl.Pipeline.init(device, lib, "i2p_routepick", false);
        errdefer s.pipes.pick.deinit();
        s.pipes.embed = try mtl.Pipeline.init(device, lib, "i2p_embed", false);
        errdefer s.pipes.embed.deinit();
        s.pipes.rms = try mtl.Pipeline.init(device, lib, "i2p_rms", false);
        errdefer s.pipes.rms.deinit();
        s.pipes.rms3 = try mtl.Pipeline.init(device, lib, "i2p_rms3", false);
        errdefer s.pipes.rms3.deinit();
        s.pipes.rms_pair = try mtl.Pipeline.init(device, lib, "i2p_rms_pair", false);
        errdefer s.pipes.rms_pair.deinit();
        s.pipes.gather_rms = try mtl.Pipeline.init(device, lib, "i2p_gather_rms", false);
        errdefer s.pipes.gather_rms.deinit();
        s.pipes.scatter = try mtl.Pipeline.init(device, lib, "i2p_scatter", false);
        errdefer s.pipes.scatter.deinit();
        s.pipes.pass_copy = try mtl.Pipeline.init(device, lib, "i2p_pass_copy", false);
        errdefer s.pipes.pass_copy.deinit();
        s.pipes.mix = try mtl.Pipeline.init(device, lib, "i2p_mix", false);
        errdefer s.pipes.mix.deinit();
        s.pipes.mix4 = try mtl.Pipeline.init(device, lib, "i2p_mix4", false);
        errdefer s.pipes.mix4.deinit();
        s.pipes.ctl = try mtl.Pipeline.init(device, lib, "i2p_ctl", false);
        errdefer s.pipes.ctl.deinit();
        s.pipes.clips = try mtl.Pipeline.init(device, lib, "i2p_clip_scaled", false);
        errdefer s.pipes.clips.deinit();
        s.pipes.clip3 = try mtl.Pipeline.init(device, lib, "i2p_clip3", false);
        errdefer s.pipes.clip3.deinit();
        s.pipes.clip = try mtl.Pipeline.init(device, lib, "i2p_clip", false);
        errdefer s.pipes.clip.deinit();
        s.pipes.conv = try mtl.Pipeline.init(device, lib, "i2p_conv", false);
        errdefer s.pipes.conv.deinit();
        s.pipes.batch_conv = try mtl.Pipeline.init(device, lib, "i2p_batch_conv", false);
        errdefer s.pipes.batch_conv.deinit();
        s.pipes.batch_store_rel = try mtl.Pipeline.init(device, lib, "i2p_batch_store_rel", false);
        errdefer s.pipes.batch_store_rel.deinit();
        s.pipes.cstore = try mtl.Pipeline.init(device, lib, "i2p_conv_store", false);
        errdefer s.pipes.cstore.deinit();
        s.pipes.kv = try mtl.Pipeline.init(device, lib, "i2p_kv_store", false);
        errdefer s.pipes.kv.deinit();
        s.pipes.rel = try mtl.Pipeline.init(device, lib, "i2p_relbias", false);
        errdefer s.pipes.rel.deinit();
        s.pipes.rel_rows = try mtl.Pipeline.init(device, lib, "i2p_relbias_rows", false);
        errdefer s.pipes.rel_rows.deinit();
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
        s.pipes.half_attn = null;
        s.pipes.half_expert_nv = null;
        s.pipes.wide_expert_nv = null;
        s.pipes.wide_gu_nv = null;
        s.pipes.wide_down_nv = null;
        s.pipes.half_expert_mx = null;
        s.pipes.units_up = null;
        s.pipes.units_down = null;
        s.pipes.affine_units_up = null;
        s.pipes.affine_units_down = null;
        s.pipes.paired = @splat(null);
        s.pipes.attn_units = null;
        s.pipes.attn_units64 = null;
        s.pipes.half_rows = null;
        if (fast or device.tensorUnits()) {
            const half_lib = try mtl.Library.fromSource(device, if (!device.tensorUnits()) half_source else if (s.rounding.experts) units_rows_source else units_source, opts);
            defer half_lib.deinit();
            if (device.tensorUnits()) {
                s.pipes.wide_expert_nv = try mtl.Pipeline.init(device, half_lib, "i2p_expert_matrix_wide", false);
                errdefer s.pipes.wide_expert_nv.?.deinit();
                s.pipes.wide_gu_nv = try mtl.Pipeline.init(device, half_lib, "i2p_gu_grouped", false);
                errdefer s.pipes.wide_gu_nv.?.deinit();
                s.pipes.wide_down_nv = try mtl.Pipeline.init(device, half_lib, "i2p_down_grouped", false);
                errdefer s.pipes.wide_down_nv.?.deinit();
                s.pipes.units_up = try mtl.Pipeline.init(device, half_lib, "i2p_expert_units_up", false);
                errdefer s.pipes.units_up.?.deinit();
                s.pipes.units_down = try mtl.Pipeline.init(device, half_lib, "i2p_expert_units_down", false);
                errdefer s.pipes.units_down.?.deinit();
                s.pipes.affine_units_up = try mtl.Pipeline.init(device, half_lib, "i2p_affine_units_up", false);
                errdefer s.pipes.affine_units_up.?.deinit();
                s.pipes.affine_units_down = try mtl.Pipeline.init(device, half_lib, "i2p_affine_units_down", false);
                errdefer s.pipes.affine_units_down.?.deinit();
                for ([_][:0]const u8{ "i2pu_units_up", "i2pu_units_down", "i2pu_affine_up", "i2pu_affine_down" }, 0..) |name, i| {
                    errdefer for (s.pipes.paired[0..i]) |pipe| pipe.?.deinit();
                    s.pipes.paired[i] = try mtl.Pipeline.init(device, half_lib, name, false);
                }
                errdefer for (s.pipes.paired) |pipe| pipe.?.deinit();
                s.pipes.attn_units = try mtl.Pipeline.init(device, half_lib, "i2p_attn_units", false);
                errdefer s.pipes.attn_units.?.deinit();
                s.pipes.attn_units64 = try mtl.Pipeline.init(device, half_lib, "i2pa_attn_units64", false);
                errdefer s.pipes.attn_units64.?.deinit();
                s.pipes.half_rows = try mtl.Pipeline.init(device, half_lib, "i2p_half_rows", false);
            }
            errdefer if (s.pipes.wide_expert_nv) |pipe| pipe.deinit();
            errdefer if (s.pipes.wide_gu_nv) |pipe| pipe.deinit();
            errdefer if (s.pipes.wide_down_nv) |pipe| pipe.deinit();
            errdefer if (s.pipes.units_up) |pipe| pipe.deinit();
            errdefer if (s.pipes.units_down) |pipe| pipe.deinit();
            errdefer if (s.pipes.affine_units_up) |pipe| pipe.deinit();
            errdefer if (s.pipes.affine_units_down) |pipe| pipe.deinit();
            errdefer if (s.pipes.attn_units) |pipe| pipe.deinit();
            errdefer if (s.pipes.attn_units64) |pipe| pipe.deinit();
            errdefer if (s.pipes.half_rows) |pipe| pipe.deinit();
            s.pipes.half_attn = try mtl.Pipeline.init(device, half_lib, "i2p_attn_half", false);
            errdefer s.pipes.half_attn.?.deinit();
            if (fast) {
                const fast_lib = try mtl.Library.fromSource(device, "#define I2P_HALF_OPERANDS 1\n" ++ kernels.prefill, opts);
                defer fast_lib.deinit();
                s.pipes.half_expert_nv = try mtl.Pipeline.init(device, fast_lib, "i2p_expert_matrix", false);
                errdefer s.pipes.half_expert_nv.?.deinit();
                const half_mx_lib = try mtl.Library.fromSource(device, "#define I2P_HALF_OPERANDS 1\n#define I2P_MXFP8 1\n" ++ kernels.prefill, opts);
                defer half_mx_lib.deinit();
                s.pipes.half_expert_mx = try mtl.Pipeline.init(device, half_mx_lib, "i2p_expert_matrix", false);
            }
        }
        errdefer if (s.pipes.half_attn) |pipe| pipe.deinit();
        errdefer if (s.pipes.half_expert_nv) |pipe| pipe.deinit();
        errdefer if (s.pipes.wide_expert_nv) |pipe| pipe.deinit();
        errdefer if (s.pipes.wide_gu_nv) |pipe| pipe.deinit();
        errdefer if (s.pipes.wide_down_nv) |pipe| pipe.deinit();
        errdefer if (s.pipes.half_expert_mx) |pipe| pipe.deinit();
        errdefer if (s.pipes.units_up) |pipe| pipe.deinit();
        errdefer if (s.pipes.units_down) |pipe| pipe.deinit();
        errdefer if (s.pipes.affine_units_up) |pipe| pipe.deinit();
        errdefer if (s.pipes.affine_units_down) |pipe| pipe.deinit();
        errdefer if (s.pipes.attn_units) |pipe| pipe.deinit();
        errdefer if (s.pipes.attn_units64) |pipe| pipe.deinit();
        errdefer if (s.pipes.half_rows) |pipe| pipe.deinit();
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

    fn commit(s: *PrefillScratch, device: mtl.Device) !void {
        const queue = try device.queue();
        defer queue.deinit();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const blit = cb.blit();
        inline for (buffer_names) |f| blit.fill(@field(s, f), 0, @field(s, f).length(), 0);
        for ([_]?mtl.Buffer{ s.bf16_input, s.xh, s.act_h, s.affine_stage_w, s.affine_stage_x }) |optional| {
            if (optional) |buf| blit.fill(buf, 0, buf.length(), 0);
        }
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
    }

    const buffer_names = [_][]const u8{ "ids", "x", "prelude", "passout", "bank", "n0", "n1", "n2", "m0", "q", "k", "v", "r", "kc", "vc", "kh", "vh", "kdyn", "vdyn", "sink", "relbias", "attn", "gate", "projout", "o", "aconv", "moe", "blocks", "staged", "mixout", "msg", "h2in", "normed", "tanhg", "tanhf", "router", "picks", "wts", "act", "down", "moe_cnt", "moe_off", "moe_cur", "moe_sorted", "logits" };

    pub fn deinit(s: *PrefillScratch) void {
        if (s.bf16_input) |buf| buf.deinit();
        if (s.half_overflow) |buf| buf.deinit();
        if (s.xh) |buf| buf.deinit();
        if (s.act_h) |buf| buf.deinit();
        if (s.affine_stage_w) |buf| buf.deinit();
        if (s.affine_stage_x) |buf| buf.deinit();
        s.pipes.deinit();
        inline for (buffer_names) |f| @field(s, f).deinit();
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
    batch: ?*BatchContext = null,

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
            base += rows * kvw * 2;
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
        if (w.affine != null or w.dtype == .f32) {
            c.affineLin(w, x, out, @intCast(m), c.n, 0);
            return;
        }
        const args = Gemm{ .n = @intCast(m), .k = @intCast(width(w)), .rows = c.n, .xs = x.xs, .xb = x.xb, .cap = 0 };
        if (c.n <= 8) {
            c.setP(c.s.pipes.smallrow, args, &.{ x.buf, t(w), A, out });
            c.enc.dispatchGroups(mtl.Size.of((m + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        } else if (c.s.pipes.gemm_mpp != null and m >= 384 and args.k >= 2304 and c.n >= 16) {
            if (c.s.pipes.bf16_gemm_mpp) |pipe| {
                const input = b(c.s.bf16_input.?);
                c.setP(c.s.pipes.bf16_convert.?, args, &.{ x.buf, A, input });
                c.enc.dispatchThreads(mtl.Size.of(@as(usize, c.n) * args.k, 1, 1), mtl.Size.of(256, 1, 1));
                c.setP(pipe, args, &.{ input, t(w), A, out });
            } else {
                c.setP(c.s.pipes.gemm_mpp.?, args, &.{ x.buf, t(w), A, out });
            }
            c.enc.dispatchGroups(mtl.Size.of((m + 31) / 32, (c.n + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        } else {
            c.setP(c.s.pipes.gemm, args, &.{ x.buf, t(w), A, out });
            c.enc.dispatchGroups(mtl.Size.of((m + 63) / 64, (c.n + 31) / 32, 1), mtl.Size.of(256, 1, 1));
        }
    }

    fn headModel(c: Chunk, w: Tensor, x: In, out: B, vocab: u32) void {
        if (w.affine != null) {
            c.affineLin(w, x, out, vocab, c.n, c.m.config.text.logit_cap);
            return;
        }
        const args = Gemm{ .n = vocab, .k = @intCast(width(w)), .rows = c.n, .xs = x.xs, .xb = x.xb, .cap = c.m.config.text.logit_cap };
        if (c.n <= 8) {
            c.setP(c.s.pipes.smallrow, args, &.{ x.buf, t(w), A, out });
            c.enc.dispatchGroups(mtl.Size.of((vocab + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        } else if (c.s.pipes.gemm_mpp != null and c.n >= 16 and args.k >= 2304) {
            if (c.s.pipes.bf16_gemm_mpp) |pipe| {
                const input = b(c.s.bf16_input.?);
                c.setP(c.s.pipes.bf16_convert.?, args, &.{ x.buf, A, input });
                c.enc.dispatchThreads(mtl.Size.of(@as(usize, c.n) * args.k, 1, 1), mtl.Size.of(256, 1, 1));
                c.setP(pipe, args, &.{ input, t(w), A, out });
            } else {
                c.setP(c.s.pipes.gemm_mpp.?, args, &.{ x.buf, t(w), A, out });
            }
            c.enc.dispatchGroups(mtl.Size.of((vocab + 31) / 32, (c.n + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        } else {
            c.setP(c.s.pipes.gemm, args, &.{ x.buf, t(w), A, out });
            c.enc.dispatchGroups(mtl.Size.of((vocab + 63) / 64, (c.n + 31) / 32, 1), mtl.Size.of(256, 1, 1));
        }
    }

    fn headModelLast(c: Chunk, w: Tensor, xoff: B, out: B, vocab: u32) void {
        if (w.affine != null) {
            c.affineLin(w, .{ .buf = xoff, .xs = 1, .xb = 0 }, out, vocab, 1, c.m.config.text.logit_cap);
            return;
        }
        c.setP(c.s.pipes.capexact, Cap{ .n = vocab, .k = @intCast(width(w)), .rows = 1, .cap = c.m.config.text.logit_cap }, &.{ xoff, t(w), A, out });
        c.enc.dispatchGroups(mtl.Size.of((vocab + 31) / 32, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    fn affineLin(c: Chunk, w: Tensor, x: In, out: B, n: u32, rows: u32, cap: f32) void {
        const p = c.m.affine_pipes.?;
        const args = AffineDense{ .n = n, .k = @intCast(width(w)), .rows = rows, .xs = x.xs, .xb = x.xb, .bits = if (w.affine) |q| q.bits() else 0, .cap = cap };
        if (w.affine) |q| {
            if (rows > 8 and c.s.half and c.s.pipes.affine_stage_gemm != null and n <= c.s.affine_stage_extent and args.k <= c.s.affine_stage_extent) {
                const staged_w = b(c.s.affine_stage_w.?);
                c.setP(c.s.pipes.affine_stage_prepare.?, args, &.{ t(w), bo(q.scales, q.scale_offset), bo(q.biases, q.bias_offset), A, staged_w, b(c.s.half_overflow.?) });
                c.enc.dispatchThreads(mtl.Size.of(@as(usize, n) * args.k / 4, 1, 1), mtl.Size.of(256, 1, 1));
                if (c.s.units and c.s.pipes.affine_stage_gemm_split != null and args.k % 4 == 0) {
                    const split = b(c.s.affine_stage_x.?);
                    c.setP(c.s.pipes.affine_stage_split.?, args, &.{ x.buf, A, split, b(c.s.half_overflow.?) });
                    c.enc.dispatchThreads(mtl.Size.of(@as(usize, rows) * args.k / 4, 1, 1), mtl.Size.of(256, 1, 1));
                    c.setP(c.s.pipes.affine_stage_gemm_split.?, args, &.{ split, staged_w, A, out });
                    c.enc.dispatchGroups(mtl.Size.of((n + 63) / 64, (rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
                    return;
                }
                c.setP(c.s.pipes.affine_stage_gemm.?, args, &.{ x.buf, staged_w, A, out });
                c.enc.dispatchGroups(mtl.Size.of((n + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
                return;
            }
            if (rows > 8 and c.s.half and c.s.pipes.affine_half_rows != null) {
                c.setP(c.s.pipes.affine_half_rows.?, args, &.{ x.buf, t(w), bo(q.scales, q.scale_offset), bo(q.biases, q.bias_offset), A, out, b(c.s.half_overflow.?) });
                c.enc.dispatchGroups(mtl.Size.of((n + 63) / 64, (rows + 31) / 32, 1), mtl.Size.of(256, 1, 1));
                return;
            } else if (c.sharedDense(w, q, x, out, args)) {
                return;
            } else {
                c.setP(p.dense, args, &.{ x.buf, t(w), bo(q.scales, q.scale_offset), bo(q.biases, q.bias_offset), A, out });
            }
        } else {
            if (rows > 8 and c.s.half and c.s.pipes.float_mpp != null) {
                c.setP(c.s.pipes.float_mpp.?, args, &.{ x.buf, t(w), A, out });
                c.enc.dispatchGroups(mtl.Size.of((n + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
                return;
            }
            c.setP(p.float, args, &.{ x.buf, t(w), A, out });
        }
        c.enc.dispatchGroups(mtl.Size.of((n + 7) / 8, rows, 1), mtl.Size.of(256, 1, 1));
    }

    fn sharedDense(c: Chunk, w: Tensor, q: wts.Affine, x: In, out: B, args: AffineDense) bool {
        const batch = c.batch orelse return false;
        const d = batch.scratch.shared.dense(args.rows, args.n, args.k, args.bits) orelse return false;
        c.setP(d.pipe, args, &.{ x.buf, t(w), bo(q.scales, q.scale_offset), bo(q.biases, q.bias_offset), A, out });
        c.enc.dispatchGroups(d.groups, d.threads);
        return true;
    }

    fn affineProjections(c: Chunk, lay: *const Layer, inputs: [3]In, full: bool) void {
        const a = &lay.attn;
        const s = c.s;
        const k = if (full) a.k else a.q;
        const v = if (full) a.v else a.q;
        const qf = a.q.affine.?;
        const kf = k.affine.?;
        const vf = v.affine.?;
        const ms: [8]u32 = .{ @intCast(nrows(a.q)), @intCast(nrows(a.k)), @intCast(nrows(a.v)), @intCast(nrows(a.r)), @intCast(nrows(a.gate)), @intCast(nrows(a.sink)), if (full) @intCast(nrows(a.k_dyn.?)) else 0, if (full) @intCast(nrows(a.v_dyn.?)) else 0 };
        var total: usize = 0;
        for (ms) |n| total += n;
        const bufs = [_]B{
            inputs[0].buf, inputs[1].buf, inputs[2].buf, t(a.q), bo(qf.scales, qf.scale_offset), bo(qf.biases, qf.bias_offset),
            t(k), bo(kf.scales, kf.scale_offset), bo(kf.biases, kf.bias_offset),
            t(v), bo(vf.scales, vf.scale_offset), bo(vf.biases, vf.bias_offset),
            b(s.q), b(s.k), b(s.v), b(s.r), b(s.gate), b(s.sink), b(s.kdyn), b(s.vdyn), A,
        };
        if (c.batch) |batch| if (batch.scratch.shared.fused(c.n, ms, @intCast(width(a.q)))) |d| {
            c.setP(d.pipe, SharedProjections{ .k = @intCast(width(a.q)), .full = @intFromBool(full), .rows = c.n, .m = ms }, &bufs);
            c.enc.dispatchGroups(d.groups, d.threads);
            return;
        };
        c.setP(c.m.affine_pipes.?.batch_projections, AffineProjections{ .k = @intCast(width(a.q)), .full = @intFromBool(full), .m = ms }, &bufs);
        c.enc.dispatchGroups(mtl.Size.of((total + 7) / 8, c.n, 1), mtl.Size.of(256, 1, 1));
    }

    fn rmsW(c: Chunk, x: In, weight: B, out: B, ys: u32, yb: u32, w: u32, eps: f32) void {
        c.setP(c.s.pipes.rms, RmsB{ .width = w, .rows = c.n, .xs = x.xs, .xb = x.xb, .ys = ys, .yb = yb, .wb = 1, .eps = eps }, &.{ x.buf, weight, A, out });
        c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn flat(c: Chunk, g: wts.Glue, args: anytype, bufs: []const B, count: usize) void {
        c.setG(g, args, bufs);
        c.enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn tap(c: Chunk, visit: u32) void {
        if (c.s.taps) |taps| taps.encode(c.enc, visit, c.s.x, 0, c.n);
    }

    fn copyN(c: Chunk, x: B, out: B, count: usize) void {
        c.flat(.copy, Pair{ .count = @intCast(count) }, &.{ x, out, A }, count);
    }

    fn addN(c: Chunk, dst: B, src: B, count: usize) void {
        c.flat(.add, Pair{ .count = @intCast(count) }, &.{ dst, src, A }, count);
    }

    fn gatherM(c: Chunk, blk: B, cur: B, depth: u32) void {
        c.setP(c.s.pipes.gather_rms, Gather{ .width = c.h, .depth = depth, .stride = st.mix_states }, &.{ blk, cur, A, b(c.s.staged) });
        c.enc.dispatchGroups(mtl.Size.of(depth * c.n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn mixQ(c: Chunk, query: Tensor, out: B) void {
        const depth = c.count + 1;
        const k: u32 = @intCast(nrows(query));
        const pipe = if (c.batch != null and k <= 4 and depth <= st.mix_states) c.batch.?.scratch.mix else c.s.pipes.mix;
        c.setP(pipe, MixB{ .width = c.h, .depth = @intCast(depth), .k = k, .rows = c.n, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.h))) }, &.{ b(c.s.staged), t(query), A, out });
        c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn mixBlocks(c: Chunk, cur: B, query: Tensor, out: B) void {
        c.gatherM(b(c.s.blocks), cur, @intCast(c.count + 1));
        c.mixQ(query, out);
    }

    fn mixNorm(c: Chunk, cur: B, query: Tensor, out: B, norms: []const Tensor, outs: [3]B) bool {
        var queries: [4]B = @splat(t(query));
        const k = nrows(query);
        for (0..@min(k, 4)) |i| queries[i] = bo(query.buffer, query.offset + i * c.h * 4);
        return c.mixStates(b(c.s.blocks), cur, c.count + 1, queries, k, out, norms, outs);
    }

    fn mixStates(c: Chunk, states: B, cur: B, depth: usize, queries: [4]B, k: usize, out: B, norms: []const Tensor, outs: [3]B) bool {
        if (c.batch != null or !c.s.fuse or !rows_mod.mixFits(c.h, depth, k, norms.len)) return false;
        const args = rows_mod.MixArgs{ .width = c.h, .depth = @intCast(depth), .stride = st.mix_states, .k = @intCast(k), .norm = @intFromBool(norms.len > 0), .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.h))), .eps = c.m.config.text.eps };
        var weights: [3]B = @splat(queries[0]);
        for (norms, 0..) |w, i| weights[i] = t(w);
        c.setP(c.s.pipes.rows.mix, args, &.{ states, cur, queries[0], queries[1], queries[2], queries[3], A, out, weights[0], weights[1], weights[2], outs[0], outs[1], outs[2] });
        c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(rows_mod.threads, 1, 1));
        return true;
    }

    fn pushBlock(c: *Chunk) void {
        c.setP(c.s.pipes.scatter, Scatter{ .width = c.h, .stride = st.mix_states, .slot = @intCast(c.count) }, &.{ b(c.s.x), A, b(c.s.blocks), b(c.s.o) });
        c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
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

    fn convolve(c: Chunk, x: B, out: B, weight: Tensor, dyn: ?Tensor, dyn_in: In, dyn_out: B, ring: B, channels: u32, conv_heads: u32, layout: u32, slot: u32, site: ConvSite, dyn_ready: bool) void {
        const taps: u32 = cfg.conv_taps;
        const mul: u32 = if (dyn != null) 2 else 1;
        const norm: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(mul * taps + 1)));
        if (!dyn_ready) if (dyn) |dw| c.lin(dw, dyn_in, dyn_out);
        if (c.batch) |batch| {
            if (batch.group) |g| {
                g.conv(c.enc, x.buf, x.off, weight, if (dyn != null) dyn_out.buf else null, dyn_out.off, if (site == .mlp) c.s.x else out.buf, if (site == .mlp) 0 else out.off, channels, conv_heads, layout, norm, slot, site);
                return;
            }
            var rows: [8]ConvRow = undefined;
            for (batch.caches, 0..) |cache, row| {
                const buf = switch (site) {
                    .key => cache.k_conv,
                    .value => cache.v_conv,
                    .attention => cache.attn_conv,
                    .mlp => cache.mlp_conv,
                };
                rows[row] = .{ .ring = buf.gpuAddress() + cache.convAt(slot, channels), .pos = @intCast(cache.len) };
                c.enc.useResource(buf, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
            }
            const args = DecodeConv{ .channels = channels, .heads = conv_heads, .ring = st.ring, .pos = 0, .layout = layout, .norm = norm, .residual = @intFromBool(site == .mlp) };
            c.setP(c.s.pipes.batch_conv, args, &.{ x, t(weight), if (dyn != null) dyn_out else A, A, if (site == .mlp) b(c.s.x) else out });
            c.enc.setBytes(std.mem.sliceAsBytes(rows[0..batch.caches.len]), 5);
            c.enc.dispatchThreads(mtl.Size.of(channels, c.n, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        c.setP(c.s.pipes.conv, ConvB{ .channels = channels, .heads = conv_heads, .ring = st.ring, .base = c.base, .rows = c.n, .layout = layout, .norm = norm }, &.{ x, ring, t(weight), if (dyn != null) dyn_out else A, A, out });
        c.enc.dispatchThreads(mtl.Size.of(channels, c.n, 1), mtl.Size.of(256, 1, 1));
        c.ringStore(x, ring, channels);
        if (c.s.verify_conv) |capture| c.copyN(x, bo(capture.buffer, capture.offset(slot, site, 0)), @as(usize, c.n) * channels);
    }

    fn ringStore(c: Chunk, x: B, ring: B, channels: u32) void {
        if (!c.s.fuse) {
            c.setP(c.s.pipes.cstore, CStore{ .channels = channels, .ring = st.ring, .base = c.base, .rows = c.n }, &.{ x, ring, A });
            c.enc.dispatchThreads(mtl.Size.of(channels, c.n, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        const tail: u32 = @min(c.n, st.ring);
        c.setP(c.s.pipes.rows.ring, rows_mod.RingArgs{ .channels = channels, .ring = st.ring, .base = c.base, .rows = c.n, .tail = tail }, &.{ x, ring, A });
        c.enc.dispatchThreads(mtl.Size.of(channels, tail, 1), mtl.Size.of(256, 1, 1));
    }

    fn convolveAdd(c: Chunk, x: B, out: ?B, weight: Tensor, ring: B, slot: u32, site: ConvSite, residual: B) bool {
        if (c.batch != null or !c.s.fuse) return false;
        const norm: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.conv_taps + 1)));
        c.setP(c.s.pipes.rows.conv_add, rows_mod.ConvArgs{ .channels = c.h, .heads = 0, .ring = st.ring, .base = c.base, .rows = c.n, .layout = 0, .norm = norm, .out = @intFromBool(out != null) }, &.{ x, ring, t(weight), A, out orelse residual, residual });
        c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
        c.ringStore(x, ring, c.h);
        if (c.s.verify_conv) |capture| c.copyN(x, bo(capture.buffer, capture.offset(slot, site, 0)), @as(usize, c.n) * c.h);
        return true;
    }

    fn kvStore(c: Chunk, slot: u32) void {
        const at = c.slotBase(slot);
        if (c.s.fuse and c.kvw % 4 == 0) {
            c.setP(c.s.pipes.rows.kv, rows_mod.KvArgs{ .count = c.kvw, .base = c.base, .rows = c.n, .wring = c.slotRing(slot) }, &.{ b(c.s.kh), b(c.s.vh), bo(c.cache.keys, at), bo(c.cache.values, at), A });
            c.enc.dispatchThreads(mtl.Size.of(c.kvw / 4, c.n, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        c.setP(c.s.pipes.kv, KvB{ .count = c.kvw, .base = c.base, .rows = c.n, .wring = c.slotRing(slot) }, &.{ b(c.s.kh), b(c.s.vh), bo(c.cache.keys, at), bo(c.cache.values, at), A });
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
        const args = RelB{ .heads = c.heads, .rel = c.drel, .extent = extent, .rows = c.n, .base = c.base, .tau = @intFromBool(full), .floor = @floatFromInt(text.log_floor), .alpha = text.log_alpha };
        if (c.heads * c.drel <= 1024 and c.drel <= 32) {
            c.setP(c.s.pipes.rel_rows, args, &.{ t(lay.attn.rel_bias), r, A, b(c.s.relbias) });
            c.enc.dispatchGroups(mtl.Size.of((extent + 1 + 255) / 256, c.n, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        c.setP(c.s.pipes.rel, args, &.{ t(lay.attn.rel_bias), r, A, b(c.s.relbias) });
        c.enc.dispatchThreads(mtl.Size.of(c.heads * (extent + 1), c.n, 1), mtl.Size.of(256, 1, 1));
    }

    fn batchAttention(c: Chunk, lay: *const Layer, slot: u32, full: bool, extent: u32, batch: *BatchContext) void {
        if (batch.group) |g| {
            g.attend(c.enc, c.m, lay.attn.rel_bias, slot, full, extent, .{ .q = c.s.q, .k = c.s.kc, .v = c.s.vc, .r = c.s.r, .sink = c.s.sink, .relbias = c.s.relbias, .out = c.s.attn, .parts = batch.scratch.parts, .combine = batch.scratch.combine });
            return;
        }
        const s = c.s;
        const text = &c.m.config.text;
        const win: u32 = if (full) 0 else text.window;
        var rows: [8]BatchRow = undefined;
        var store_rows: [8]BatchStoreRow = undefined;
        var max_splits: u32 = 0;
        for (batch.caches, 0..) |cache, row| {
            const pos: u32 = @intCast(cache.len);
            const at = cache.row(&c.m.config, slot, pos);
            store_rows[row] = .{ .keys = cache.keys.gpuAddress() + at, .values = cache.values.gpuAddress() + at, .pos = pos };
            const lo: u32 = if (win != 0 and pos + 1 > win) pos + 1 - win else 0;
            const part = st.attnPartition(pos + 1 - lo);
            const parts = DecodeParts{ .heads = c.heads, .kv = c.kv, .dim = c.dim, .rel = c.drel, .extent = extent, .window = win, .pos = pos, .tau = @intFromBool(full), .splits = part.splits, .chunk = part.chunk, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.dim))), .floor = @floatFromInt(text.log_floor), .alpha = text.log_alpha };
            rows[row] = .{ .keys = cache.keys.gpuAddress() + cache.row(&c.m.config, slot, 0), .values = cache.values.gpuAddress() + cache.row(&c.m.config, slot, 0), .args = parts };
            max_splits = @max(max_splits, part.splits);
            c.enc.useResource(cache.keys, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
            c.enc.useResource(cache.values, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        }
        const rel_args = DecodeAttn{ .heads = c.heads, .kv = c.kv, .dim = c.dim, .rel = c.drel, .extent = extent, .window = 0, .pos = 0, .tau = @intFromBool(full), .scale = 0, .floor = @floatFromInt(text.log_floor), .alpha = text.log_alpha, .eps = text.eps };
        c.setP(s.pipes.batch_store_rel, rel_args, &.{ b(s.kc), b(s.vc), b(s.r), t(lay.attn.rel_bias), A, b(s.relbias) });
        c.enc.setBytes(std.mem.sliceAsBytes(store_rows[0..batch.caches.len]), 6);
        c.enc.dispatchGroups(mtl.Size.of(c.kv + (c.heads * (extent + 1) + 255) / 256, c.n, 1), mtl.Size.of(256, 1, 1));
        c.enc.setPipeline(batch.scratch.split);
        c.enc.setBuffer(s.q, 0, 0);
        c.enc.setBuffer(s.relbias, 0, 1);
        c.enc.setBuffer(s.sink, 0, 2);
        c.enc.setBytes(std.mem.sliceAsBytes(rows[0..batch.caches.len]), 3);
        c.enc.setBuffer(batch.scratch.parts, 0, 4);
        c.enc.dispatchGroups(mtl.Size.of(c.kv * max_splits, c.n, 1), mtl.Size.of((c.heads / c.kv) * 32, 1, 1));
        c.enc.setPipeline(batch.scratch.combine);
        c.enc.setBuffer(batch.scratch.parts, 0, 0);
        c.enc.setBytes(std.mem.sliceAsBytes(rows[0..batch.caches.len]), 1);
        c.enc.setBuffer(s.attn, 0, 2);
        c.enc.dispatchGroups(mtl.Size.of(c.heads, c.n, 1), mtl.Size.of(32, 1, 1));
    }

    fn control(c: Chunk, ctl: Controller, cur: In, msg: In, out: B, ys: u32, yb: u32) void {
        const s = c.s;
        c.setP(s.pipes.rms_pair, RmsPair{ .width = c.h, .xs0 = cur.xs, .xb0 = cur.xb, .xs1 = msg.xs, .xb1 = msg.xb, .eps = 1e-6 }, &.{ cur.buf, msg.buf, A, b(s.h2in) });
        c.enc.dispatchGroups(mtl.Size.of(2 * c.n, 1, 1), mtl.Size.of(256, 1, 1));
        c.lin(ctl.down, .{ .buf = b(s.h2in), .xs = 1, .xb = 0 }, b(s.normed));
        if (c.batch != null and ctl.gain.dtype == .f32 and ctl.forget.dtype == .f32 and c.m.affine_pipes != null) {
            const n: u32 = @intCast(nrows(ctl.gain));
            c.setP(c.m.affine_pipes.?.batch_float_control, AffineBatchControl{ .k = @intCast(width(ctl.gain)), .m0 = n, .m1 = @intCast(nrows(ctl.forget)), .h = c.h, .cxs = cur.xs, .cxb = cur.xb, .mxs = msg.xs, .mxb = msg.xb, .ys = ys, .yb = yb }, &.{ b(s.normed), t(ctl.gain), t(ctl.forget), cur.buf, msg.buf, out, A });
            c.enc.dispatchGroups(mtl.Size.of((n + 7) / 8, c.n, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        const rank: u32 = @intCast(width(ctl.gain));
        const fusable = c.batch == null and s.fuse and c.n > 8 and s.half and s.pipes.float_mpp != null and ctl.gain.dtype == .f32 and ctl.forget.dtype == .f32 and nrows(ctl.gain) == c.h and nrows(ctl.forget) == c.h and width(ctl.forget) == rank;
        if (fusable) if (s.pipes.gate_ctl) |pipe| {
            c.setP(pipe, GateCtl{ .h = c.h, .rows = c.n, .k = rank, .cxs = cur.xs, .cxb = cur.xb, .mxs = msg.xs, .mxb = msg.xb, .ys = ys, .yb = yb }, &.{ b(s.normed), t(ctl.gain), t(ctl.forget), A, cur.buf, msg.buf, out });
            c.enc.dispatchGroups(mtl.Size.of((c.h + 31) / 32, (c.n + 15) / 16, 1), mtl.Size.of(128, 1, 1));
            return;
        };
        c.lin(ctl.gain, .{ .buf = b(s.normed), .xs = 1, .xb = 0 }, b(s.tanhg));
        c.lin(ctl.forget, .{ .buf = b(s.normed), .xs = 1, .xb = 0 }, b(s.tanhf));
        c.setP(s.pipes.ctl, CtlB{ .h = c.h, .rows = c.n, .cxs = cur.xs, .cxb = cur.xb, .mxs = msg.xs, .mxb = msg.xb, .ys = ys, .yb = yb }, &.{ cur.buf, msg.buf, b(s.tanhg), b(s.tanhf), A, out });
        c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
    }

    fn controls(c: Chunk, controllers: [3]Controller, ins: [3]In, msg: B, batch: *BatchContext) void {
        const scratch = batch.scratch;
        const rank: u32 = @intCast(nrows(controllers[0].down));
        const args = ControlsArgs{ .h = c.h, .rank = rank, .rows = c.n, .xs = .{ ins[0].xs, ins[1].xs, ins[2].xs }, .xb = .{ ins[0].xb, ins[1].xb, ins[2].xb } };
        c.setP(scratch.controls_norm, args, &.{ ins[0].buf, ins[1].buf, ins[2].buf, msg, A, b(scratch.controls_input) });
        c.enc.dispatchGroups(mtl.Size.of(6 * c.n, 1, 1), mtl.Size.of(256, 1, 1));
        c.setP(scratch.controls_down, args, &.{ b(scratch.controls_input), t(controllers[0].down), t(controllers[1].down), t(controllers[2].down), A, b(scratch.controls_hidden) });
        c.enc.dispatchGroups(mtl.Size.of((rank + 7) / 8, c.n, 3), mtl.Size.of(256, 1, 1));
        c.setP(scratch.controls_apply, args, &.{ b(scratch.controls_hidden), t(controllers[0].gain), t(controllers[1].gain), t(controllers[2].gain), t(controllers[0].forget), t(controllers[1].forget), t(controllers[2].forget), ins[0].buf, ins[1].buf, ins[2].buf, msg, b(c.s.n0), b(c.s.n1), b(c.s.n2), A });
        c.enc.dispatchGroups(mtl.Size.of((c.h + 7) / 8, c.n, 3), mtl.Size.of(256, 1, 1));
    }

    fn messages(c: Chunk, bk: Bank, ip: u32) B {
        const m = c.m;
        const s = c.s;
        const at: usize = ip - m.config.loop.core_start;
        const queries = [4]B{ t(srows(m.weights.cross_pass_q, at * 2, 1)), t(srows(m.weights.cross_pass_kv, at * 2, 1)), t(srows(m.weights.cross_pass_kv, at * 2 + 1, 1)), t(srows(m.weights.cross_pass_q, at * 2 + 1, 1)) };
        if (c.mixStates(bk.states, bk.out, bk.n + 1, queries, 4, b(s.msg), &.{}, @splat(b(s.msg)))) return b(s.msg);
        c.gatherM(bk.states, bk.out, @intCast(bk.n + 1));
        const pipe = if (c.batch != null and bk.n + 1 <= st.mix_states) c.batch.?.scratch.mix4 else s.pipes.mix4;
        c.setP(pipe, MixB{ .width = c.h, .depth = @intCast(bk.n + 1), .k = 4, .rows = c.n, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.h))) }, &.{
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

    fn stage(c: Chunk) ?staged_mod.Stage {
        const s = c.s;
        if (c.batch != null or c.n <= 8 or !s.half or !s.units or !s.fuse) return null;
        return .{
            .enc = c.enc,
            .prepare = s.pipes.affine_stage_prepare orelse return null,
            .split = s.pipes.affine_stage_split orelse return null,
            .gemm_split = s.pipes.affine_stage_gemm_split orelse return null,
            .pipes = s.pipes.staged orelse return null,
            .weights = s.affine_stage_w orelse return null,
            .inputs = s.affine_stage_x orelse return null,
            .overflow = s.half_overflow orelse return null,
            .rows = c.n,
            .half_inputs = s.rounding.inputs,
            .tall = s.tiles.tall,
        };
    }

    fn stagedProjections(c: Chunk, lay: *const Layer, ins: [3]In, full: bool) bool {
        const st_ = c.stage() orelse return false;
        const a = &lay.attn;
        const s = c.s;
        var inputs: [3]staged_mod.Input = undefined;
        for (ins, 0..) |in, i| inputs[i] = .{ .buf = in.buf.buf, .off = in.buf.off, .xs = in.xs, .xb = in.xb };
        if (full) {
            const kd = a.k_dyn orelse return false;
            const vd = a.v_dyn orelse return false;
            const qs = staged_mod.joined(&.{ a.q, a.r, a.gate, a.sink }) orelse return false;
            const ks = staged_mod.joined(&.{ a.k, kd }) orelse return false;
            const vs = staged_mod.joined(&.{ a.v, vd }) orelse return false;
            return staged_mod.encode(st_, &.{ qs, ks, vs }, &.{
                .{ .view = a.q, .out = s.q, .input = 0 },    .{ .view = a.r, .out = s.r, .input = 0 },
                .{ .view = a.gate, .out = s.gate, .input = 0 }, .{ .view = a.sink, .out = s.sink, .input = 0 },
                .{ .view = a.k, .out = s.k, .input = 1 },    .{ .view = kd, .out = s.kdyn, .input = 1 },
                .{ .view = a.v, .out = s.v, .input = 2 },    .{ .view = vd, .out = s.vdyn, .input = 2 },
            }, inputs);
        }
        const all = staged_mod.joined(&.{ a.q, a.k, a.v, a.r, a.gate, a.sink }) orelse return false;
        return staged_mod.encode(st_, &.{all}, &.{
            .{ .view = a.q, .out = s.q, .input = 0 },    .{ .view = a.k, .out = s.k, .input = 1 },
            .{ .view = a.v, .out = s.v, .input = 2 },    .{ .view = a.r, .out = s.r, .input = 0 },
            .{ .view = a.gate, .out = s.gate, .input = 0 }, .{ .view = a.sink, .out = s.sink, .input = 0 },
        }, inputs);
    }

    fn attention(c: Chunk, lay: *const Layer, slot: u32, full: bool, ins: [3]In, push: bool) bool {
        const s = c.s;
        const text = &c.m.config.text;
        const extent: u32 = if (full) text.rel_extent else text.window;
        const fused = c.batch != null and lay.attn.q.affine != null;
        const staged = !fused and c.stagedProjections(lay, ins, full);
        const ready = fused or staged;
        const clips = staged and c.dim == rows_mod.clip_dim and c.drel == rows_mod.clip_rel;
        if (fused) c.affineProjections(lay, ins, full) else if (!staged) c.lin(lay.attn.q, ins[0], b(s.q));
        if (c.batch == null and !clips) c.clipScaled(b(s.q), lay.attn.q_scale, c.heads, c.dim, text.qk_clip);
        if (!ready) c.lin(lay.attn.r, ins[0], b(s.r));
        if (c.batch == null and !clips) c.clipScaled(b(s.r), lay.attn.r_scale, c.heads, c.drel, text.rel_clip);
        const ring = c.rings(slot);
        if (!ready) c.lin(lay.attn.k, ins[1], b(s.k));
        c.convolve(b(s.k), b(s.kc), lay.attn.k_conv, lay.attn.k_dyn, ins[1], b(s.kdyn), ring[0], c.kvw, if (full) c.kv else 0, @intFromBool(full), slot, .key, ready);
        if (c.batch != null) {
            c.setP(s.pipes.clip3, Clip3B{ .heads = c.heads, .kv = c.kv, .dim = c.dim, .rel = c.drel, .q_threshold = text.qk_clip, .rel_threshold = text.rel_clip }, &.{ b(s.q), t(lay.attn.q_scale), b(s.r), t(lay.attn.r_scale), b(s.kc), A });
            c.enc.dispatchGroups(mtl.Size.of((2 * c.heads + c.kv) * c.n, 1, 1), mtl.Size.of(256, 1, 1));
        } else if (!clips) c.clipPlain(b(s.kc), c.kv, c.dim, text.qk_clip);
        if (!ready) c.lin(lay.attn.v, ins[2], b(s.v));
        c.convolve(b(s.v), b(s.vc), lay.attn.v_conv, lay.attn.v_dyn, ins[2], b(s.vdyn), ring[1], c.kvw, if (full) c.kv else 0, @intFromBool(full), slot, .value, ready);
        if (!ready) c.lin(lay.attn.sink, ins[0], b(s.sink));
        if (c.batch) |batch| {
            c.batchAttention(lay, slot, full, extent, batch);
        } else {
        if (clips) {
            const items = 2 * c.heads + c.kv;
            c.setP(s.pipes.rows.clips, rows_mod.ClipArgs{ .heads = c.heads, .kv = c.kv, .items = items, .q_threshold = text.qk_clip, .rel_threshold = text.rel_clip, .eps = text.eps }, &.{ b(s.q), t(lay.attn.q_scale), b(s.r), t(lay.attn.r_scale), b(s.kc), b(s.vc), b(s.kh), b(s.vh), A });
            c.enc.dispatchGroups(mtl.Size.of((items + rows_mod.clip_items - 1) / rows_mod.clip_items, c.n, 1), mtl.Size.of(32 * rows_mod.clip_items, 1, 1));
        } else {
            c.setG(.rms_kv_store, Rms{ .width = c.dim, .blocks = 0, .eps = text.eps }, &.{ b(s.kc), b(s.vc), b(s.kh), b(s.vh), A });
            c.enc.dispatchGroups(mtl.Size.of(c.kv * c.n, 1, 1), mtl.Size.of(256, 1, 1));
        }
        c.relative(lay, b(s.r), extent, full);
        const win: u32 = if (full) 0 else text.window;
        const keybase = bo(c.cache.keys, c.slotBase(slot));
        const valbase = bo(c.cache.values, c.slotBase(slot));
        const tiled = c.heads == 24 and c.kv == 8 and c.dim == 128;
        const attn_args = AttnB{ .heads = c.heads, .kv = c.kv, .dim = c.dim, .rel = c.drel, .extent = extent, .window = win, .base = c.base, .rows = c.n, .rmode = @intFromBool(c.ring), .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.dim))), .floor = @floatFromInt(text.log_floor), .alpha = text.log_alpha };
        if (tiled and s.half and s.pipes.half_attn != null and (s.fast or (c.n > 8 and c.m.affine_pipes != null))) {
            if (s.units and s.fuse and s.tiles.attn64 and s.pipes.attn_units64 != null) {
                c.setP(s.pipes.attn_units64.?, attn_args, &.{ b(s.q), keybase, valbase, b(s.relbias), b(s.sink), A, b(s.attn), b(s.kh), b(s.vh), b(s.half_overflow.?) });
                c.enc.dispatchGroups(mtl.Size.of(c.kv, (c.n + 63) / 64, 1), mtl.Size.of(384, 1, 1));
            } else if (s.units and s.pipes.attn_units != null) {
                c.setP(s.pipes.attn_units.?, attn_args, &.{ b(s.q), keybase, valbase, b(s.relbias), b(s.sink), A, b(s.attn), b(s.kh), b(s.vh), b(s.half_overflow.?) });
                c.enc.dispatchGroups(mtl.Size.of(c.kv, (c.n + 31) / 32, 1), mtl.Size.of(192, 1, 1));
            } else {
                c.setP(s.pipes.half_attn.?, attn_args, &.{ b(s.q), keybase, valbase, b(s.relbias), b(s.sink), A, b(s.attn), b(s.kh), b(s.vh), b(s.half_overflow.?) });
                c.enc.dispatchGroups(mtl.Size.of(c.heads, (c.n + 31) / 32, 1), mtl.Size.of(128, 1, 1));
            }
        } else {
            c.setP(if (tiled) s.pipes.attn_tiled else s.pipes.attn, attn_args, &.{ b(s.q), keybase, valbase, b(s.relbias), b(s.sink), A, b(s.attn), b(s.kh), b(s.vh) });
            if (tiled) c.enc.dispatchGroups(mtl.Size.of(c.kv, (c.n + 7) / 8, 1), mtl.Size.of(768, 1, 1)) else c.enc.dispatchGroups(mtl.Size.of(c.heads, c.n, 1), mtl.Size.of(32, 1, 1));
        }
        c.kvStore(slot);
        }
        if (!ready) c.lin(lay.attn.gate, ins[0], b(s.gate));
        const gated = if (c.stage()) |st_| staged_mod.gatedProjection(st_, lay.attn.o, s.projout, s.attn, s.gate) else false;
        if (!gated) {
            c.flat(.sig_mul, Pair{ .count = c.qw * c.n }, &.{ b(s.attn), b(s.gate), A, b(s.attn) }, @as(usize, c.qw) * c.n);
            c.lin(lay.attn.o, .{ .buf = b(s.attn), .xs = 1, .xb = 0 }, b(s.projout));
        }
        if (!push and c.convolveAdd(b(s.projout), if (s.capture != null) b(s.o) else null, lay.attn_post, ring[2], slot, .attention, b(s.x))) return true;
        c.convolve(b(s.projout), b(s.o), lay.attn_post, null, undefined, b(s.projout), ring[2], c.h, 0, 0, slot, .attention, false);
        return false;
    }

    fn experts(c: Chunk, lay: *const Layer, input: B, slot: u32) void {
        const m = c.m;
        const cc = &m.config;
        const s = c.s;
        const E: u32 = cc.text.experts;
        const K: u32 = cc.text.topk;
        const inter: u32 = cc.text.expert_inter;
        if (c.batch != null or lay.router.dtype == .f32) {
            c.lin(lay.router, .{ .buf = input, .xs = 1, .xb = 0 }, b(s.router));
        } else {
            c.setP(s.pipes.router, RouterA{ .n = E, .k = c.h, .rows = c.n, .xs = 1, .xb = 0 }, &.{ input, t(lay.router), A, b(s.router) });
            c.enc.dispatchGroups(mtl.Size.of((E + 31) / 32, c.n, 1), mtl.Size.of(1024, 1, 1));
        }
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
        if (lay.precision == .affine4) {
            const p = m.affine_pipes.?;
            if (c.n > 8 and s.half) {
                c.setP(s.pipes.moe_zero, E, &.{ b(s.moe_cnt), A });
                c.enc.dispatchThreads(mtl.Size.of(E, 1, 1), mtl.Size.of(32, 1, 1));
                c.setP(s.pipes.moe_count, slots, &.{ b(s.picks), b(s.moe_cnt), A });
                c.enc.dispatchThreads(mtl.Size.of(slots, 1, 1), mtl.Size.of(32, 1, 1));
                c.setP(s.pipes.moe_offsets, E, &.{ b(s.moe_cnt), b(s.moe_off), b(s.moe_cur), A });
                c.enc.dispatchThreads(mtl.Size.of(128, 1, 1), mtl.Size.of(128, 1, 1));
                c.setP(s.pipes.moe_sort, slots, &.{ b(s.picks), b(s.moe_cur), b(s.moe_sorted), A });
                c.enc.dispatchThreads(mtl.Size.of(slots, 1, 1), mtl.Size.of(32, 1, 1));
                const up_args = Moeg{ .k = c.h, .n = inter, .slots = slots, .topk = K, .experts = E, .gate_cap = cc.text.gate_cap, .up_cap = cc.text.up_cap };
                if (s.units and s.pipes.affine_units_up != null and c.h % 64 == 0 and inter % 64 == 0) {
                    c.affineUnitExperts(lay, input, up_args);
                    return;
                }
                if (if (s.half) s.pipes.affine_half_up else null) |pipe| {
                    c.setP(pipe, up_args, &.{ input, t(lay.w13), t(lay.w13_scale), t(lay.w13_bias.?), b(s.moe_off), b(s.moe_sorted), A, b(s.act), b(s.half_overflow.?) });
                } else c.setP(p.group_up, up_args, &.{ input, t(lay.w13), t(lay.w13_scale), t(lay.w13_bias.?), b(s.moe_off), b(s.moe_sorted), A, b(s.act) });
                c.enc.dispatchGroups(mtl.Size.of((inter * 2 + 63) / 64, E, (c.n + 31) / 32), mtl.Size.of(256, 1, 1));
                const down_args = Moeg{ .k = inter, .n = c.h, .slots = slots, .topk = K, .experts = E, .gate_cap = 0, .up_cap = 0 };
                if (if (s.half) s.pipes.affine_half_down else null) |pipe| {
                    c.setP(pipe, down_args, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), t(lay.w2_bias.?), b(s.moe_off), b(s.moe_sorted), A, b(s.down), b(s.half_overflow.?) });
                } else c.setP(p.group_down, down_args, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), t(lay.w2_bias.?), b(s.moe_off), b(s.moe_sorted), A, b(s.down) });
                c.enc.dispatchGroups(mtl.Size.of((c.h + 63) / 64, E, (c.n + 31) / 32), mtl.Size.of(256, 1, 1));
                c.setP(s.pipes.combine, CombineB{ .width = c.h, .topk = K, .rows = c.n }, &.{ b(s.down), b(s.wts), A, b(s.moe) });
                c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
                return;
            }
            c.setP(p.up, MoeB{ .k = c.h, .n = inter, .slots = slots, .topk = K, .gate_cap = cc.text.gate_cap, .up_cap = cc.text.up_cap }, &.{ input, t(lay.w13), t(lay.w13_scale), t(lay.w13_bias.?), b(s.picks), A, b(s.act) });
            c.enc.dispatchGroups(mtl.Size.of((inter + 7) / 8, slots, 1), mtl.Size.of(256, 1, 1));
            c.setP(p.down, MoeB{ .k = inter, .n = c.h, .slots = slots, .topk = K, .gate_cap = 0, .up_cap = 0 }, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), t(lay.w2_bias.?), b(s.picks), A, b(s.down) });
            c.enc.dispatchGroups(mtl.Size.of((c.h + 7) / 8, slots, 1), mtl.Size.of(256, 1, 1));
            c.setP(s.pipes.combine, CombineB{ .width = c.h, .topk = K, .rows = c.n }, &.{ b(s.down), b(s.wts), A, b(s.moe) });
            c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        if (c.n <= 8 or !s.half) {
            const codes = b(if (lay.precision == .mxfp8) m.t256 else m.t16);
            if (lay.precision == .nvfp4) if (c.batch) |batch| if (c.gatheredExperts(lay, input, batch, codes)) return;
            const gu = if (lay.precision == .mxfp8) s.pipes.gu_mx else s.pipes.gu_nv;
            const down = if (lay.precision == .mxfp8) s.pipes.down_mx else s.pipes.down_nv;
            c.setP(gu, MoeB{ .k = c.h, .n = inter, .slots = slots, .topk = K, .gate_cap = cc.text.gate_cap, .up_cap = cc.text.up_cap }, &.{ input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.picks), A, b(s.act), codes });
            c.enc.dispatchGroups(mtl.Size.of(inter, slots, 1), mtl.Size.of(32, 1, 1));
            if (lay.precision == .mxfp8) {
                c.setP(m.expert(true, .down), DecodeMoe{ .k = inter, .n = c.h, .rows = c.h, .gate_cap = 0, .up_cap = 0 }, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.picks), b(s.down), A, b(s.down) });
                c.enc.dispatchGroups(mtl.Size.of((c.h + 63) / 64, slots, 1), mtl.Size.of(64, 1, 1));
            } else {
                c.setP(down, MoeB{ .k = inter, .n = c.h, .slots = slots, .topk = K, .gate_cap = 0, .up_cap = 0 }, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.picks), A, b(s.down), codes });
                c.enc.dispatchGroups(mtl.Size.of(c.h, slots, 1), mtl.Size.of(32, 1, 1));
            }
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
        const codes = b(if (lay.precision == .mxfp8) m.t256 else m.t16);
        const wide_matrix = s.half and lay.precision != .mxfp8 and !s.fast and m.affine_pipes != null and s.pipes.wide_expert_nv != null;
        const half_matrix = if (!s.half) null else if (lay.precision == .mxfp8) s.pipes.half_expert_mx else if (wide_matrix) s.pipes.wide_expert_nv else if (s.fast) s.pipes.half_expert_nv else null;
        const gu = if (lay.precision == .mxfp8) s.pipes.gu_g_mx else if (wide_matrix) s.pipes.wide_gu_nv.? else s.pipes.gu_g_nv;
        const down = if (lay.precision == .mxfp8) s.pipes.down_g_mx else if (wide_matrix) s.pipes.wide_down_nv.? else s.pipes.down_g_nv;
        const matrix = half_matrix orelse if (lay.precision == .mxfp8) s.pipes.expert_matrix_mx else s.pipes.expert_matrix_nv;
        const args_gu = Moeg{ .k = c.h, .n = inter, .slots = slots, .topk = K, .experts = E, .gate_cap = cc.text.gate_cap, .up_cap = cc.text.up_cap };
        if (wide_matrix and s.units and s.pipes.units_up != null and c.h % 64 == 0 and inter % 64 == 0) {
            c.unitExperts(lay, input, args_gu, codes);
            return;
        }
        if (wide_matrix) {
            c.setP(gu, args_gu, &.{ input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.moe_off), b(s.moe_sorted), A, b(s.act), codes, b(s.act_h.?), b(s.half_overflow.?) });
        } else c.setP(gu, args_gu, &.{ input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.moe_off), b(s.moe_sorted), A, b(s.act), codes });
        c.enc.dispatchGroups(mtl.Size.of(inter, E, 1), mtl.Size.of(32, 1, 1));
        if (half_matrix != null) {
            const overflow = s.half_overflow.?;
            c.setP(matrix, args_gu, &.{ input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.moe_off), b(s.moe_sorted), A, b(s.act), codes, b(overflow) });
        } else {
            c.setP(matrix, args_gu, &.{ input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.moe_off), b(s.moe_sorted), A, b(s.act), codes });
        }
        c.enc.dispatchGroups(mtl.Size.of(if (wide_matrix or half_matrix == null) (inter * 2 + 63) / 64 else (inter * 2 + 31) / 32, E, (c.n + 31) / 32), mtl.Size.of(if (wide_matrix or half_matrix == null) 256 else 128, 1, 1));
        c.setP(down, Moeg{ .k = inter, .n = c.h, .slots = slots, .topk = K, .experts = E, .gate_cap = 0, .up_cap = 0 }, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.moe_off), b(s.moe_sorted), A, b(s.down), codes });
        c.enc.dispatchGroups(mtl.Size.of(c.h, E, 1), mtl.Size.of(32, 1, 1));
        const args_down = Moeg{ .k = inter, .n = c.h, .slots = slots, .topk = K, .experts = E, .gate_cap = 0, .up_cap = 0 };
        if (half_matrix != null) {
            const overflow = s.half_overflow.?;
            c.setP(matrix, args_down, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.moe_off), b(s.moe_sorted), A, b(s.down), codes, b(overflow) });
        } else {
            c.setP(matrix, args_down, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.moe_off), b(s.moe_sorted), A, b(s.down), codes });
        }
        c.enc.dispatchGroups(mtl.Size.of(if (wide_matrix or half_matrix == null) (c.h + 63) / 64 else (c.h + 31) / 32, E, (c.n + 31) / 32), mtl.Size.of(if (wide_matrix or half_matrix == null) 256 else 128, 1, 1));
        c.setP(s.pipes.combine, CombineB{ .width = c.h, .topk = K, .rows = c.n }, &.{ b(s.down), b(s.wts), A, b(s.moe) });
        c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
    }

    fn gatheredExperts(c: Chunk, lay: *const Layer, input: B, batch: *BatchContext, codes: B) bool {
        const s = c.s;
        const text = &c.m.config.text;
        const shared = &batch.scratch.shared;
        const slots: u32 = c.n * text.topk;
        const dn = shared.downs(c.n, c.h, text.expert_inter, slots) orelse return false;
        const gu = shared.gateUp(c.n, text.expert_inter, c.h, slots) orelse Dispatch{ .pipe = s.pipes.gu_nv, .groups = mtl.Size.of(text.expert_inter, slots, 1), .threads = mtl.Size.of(32, 1, 1) };
        c.setP(gu.pipe, MoeB{ .k = c.h, .n = text.expert_inter, .slots = slots, .topk = text.topk, .gate_cap = text.gate_cap, .up_cap = text.up_cap }, &.{ input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.picks), A, b(s.act), codes });
        c.enc.dispatchGroups(gu.groups, gu.threads);
        c.setP(dn.pipe, MoeB{ .k = text.expert_inter, .n = c.h, .slots = slots, .topk = text.topk, .gate_cap = 0, .up_cap = 0 }, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.picks), A, b(s.down), codes });
        c.enc.dispatchGroups(dn.groups, dn.threads);
        c.setP(s.pipes.combine, CombineB{ .width = c.h, .topk = text.topk, .rows = c.n }, &.{ b(s.down), b(s.wts), A, b(s.moe) });
        c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
        return true;
    }

    /// Routed NVFP4 projections on tensor units; experts under the matrix thresholds keep the grouped f32 kernels.
    fn unitExperts(c: Chunk, lay: *const Layer, input: B, up: Moeg, codes: B) void {
        const s = c.s;
        const overflow = b(s.half_overflow.?);
        const tiles: usize = (up.slots + 63) / 64 + up.experts;
        c.halfRows(input, true);
        c.setP(s.pipes.wide_gu_nv.?, up, &.{ input, t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.moe_off), b(s.moe_sorted), A, b(s.act), codes, b(s.act_h.?), overflow });
        c.enc.dispatchGroups(mtl.Size.of(up.n, up.experts, 1), mtl.Size.of(32, 1, 1));
        c.setP(c.unitPipe(s.pipes.units_up.?, 0), up, &.{ b(s.xh.?), t(lay.w13), t(lay.w13_scale), opt(lay.w13_global), opt(lay.w13_den), b(s.moe_off), b(s.moe_sorted), A, b(s.act_h.?), overflow });
        c.enc.dispatchGroups(mtl.Size.of(2 * up.n / 64, tiles, 1), mtl.Size.of(128, 1, 1));
        const down = Moeg{ .k = up.n, .n = c.h, .slots = up.slots, .topk = up.topk, .experts = up.experts, .gate_cap = 0, .up_cap = 0 };
        c.setP(s.pipes.wide_down_nv.?, down, &.{ b(s.act), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.moe_off), b(s.moe_sorted), A, b(s.down), codes });
        c.enc.dispatchGroups(mtl.Size.of(c.h, up.experts, 1), mtl.Size.of(32, 1, 1));
        c.setP(c.unitPipe(s.pipes.units_down.?, 1), down, &.{ b(s.act_h.?), t(lay.w2), t(lay.w2_scale), opt(lay.w2_global), opt(lay.w2_den), b(s.moe_off), b(s.moe_sorted), A, b(s.down), overflow });
        c.enc.dispatchGroups(mtl.Size.of(c.h / 64, tiles, 1), mtl.Size.of(128, 1, 1));
        c.setP(s.pipes.combine, CombineB{ .width = c.h, .topk = up.topk, .rows = c.n }, &.{ b(s.down), b(s.wts), A, b(s.moe) });
        c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
    }

    fn unitPipe(c: Chunk, single: mtl.Pipeline, which: usize) mtl.Pipeline {
        if (c.batch != null or !c.s.fuse or !c.s.tiles.paired) return single;
        return c.s.pipes.paired[which] orelse single;
    }

    /// The chunk's expert input rows as halves in `xh`, clamped as the NVFP4 matrices round them when `clamp`.
    fn halfRows(c: Chunk, input: B, clamp: bool) void {
        const count: u32 = c.n * c.h / 4;
        c.setP(c.s.pipes.half_rows.?, [2]u32{ count, @intFromBool(clamp) }, &.{ input, b(c.s.xh.?), b(c.s.half_overflow.?), A });
        c.enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// The routed affine 4-bit projections on the tensor units, every expert's rows as halves.
    fn affineUnitExperts(c: Chunk, lay: *const Layer, input: B, up: Moeg) void {
        const s = c.s;
        const overflow = b(s.half_overflow.?);
        const tiles: usize = (up.slots + 63) / 64 + up.experts;
        c.halfRows(input, false);
        c.setP(c.unitPipe(s.pipes.affine_units_up.?, 2), up, &.{ b(s.xh.?), t(lay.w13), t(lay.w13_scale), t(lay.w13_bias.?), A, b(s.moe_off), b(s.moe_sorted), A, b(s.act_h.?), overflow });
        c.enc.dispatchGroups(mtl.Size.of(2 * up.n / 64, tiles, 1), mtl.Size.of(128, 1, 1));
        const down = Moeg{ .k = up.n, .n = c.h, .slots = up.slots, .topk = up.topk, .experts = up.experts, .gate_cap = 0, .up_cap = 0 };
        c.setP(c.unitPipe(s.pipes.affine_units_down.?, 3), down, &.{ b(s.act_h.?), t(lay.w2), t(lay.w2_scale), t(lay.w2_bias.?), A, b(s.moe_off), b(s.moe_sorted), A, b(s.down), overflow });
        c.enc.dispatchGroups(mtl.Size.of(c.h / 64, tiles, 1), mtl.Size.of(128, 1, 1));
        c.setP(s.pipes.combine, CombineB{ .width = c.h, .topk = up.topk, .rows = c.n }, &.{ b(s.down), b(s.wts), A, b(s.moe) });
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
        const normed = [3]B{ b(s.n0), b(s.n1), b(s.n2) };
        var fused_norm = false;
        if (index > 0) {
            const q = m.weights.attn_queries[index - 1];
            const norms: []const Tensor = if (full) &.{ lay.q_norm, lay.k_norm, lay.v_norm } else &.{lay.q_norm};
            fused_norm = bank == null and c.mixNorm(b(s.x), q, bo(s.mixout, 0), norms, normed);
            if (!fused_norm and !c.mixNorm(b(s.x), q, bo(s.mixout, 0), &.{}, normed)) c.mixBlocks(b(s.x), q, bo(s.mixout, 0));
            const k: u32 = @intCast(nrows(q));
            if (full) {
                ins = .{ .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 0 }, .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 1 }, .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 2 } };
            } else {
                ins = .{ .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 0 }, .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 0 }, .{ .buf = bo(s.mixout, 0), .xs = k, .xb = 0 } };
            }
        }

        if (bank) |bk| {
            const msg = c.messages(bk, ip);
            const ctl: [3]Controller = .{ lay.controllers[0] orelse return error.MissingController, lay.controllers[1] orelse return error.MissingController, lay.controllers[2] orelse return error.MissingController };
            const use_batch = if (c.batch != null and c.m.affine_pipes != null) blk: {
                for (ctl) |item| if (item.down.dtype != .f32 or item.gain.dtype != .f32 or item.forget.dtype != .f32) break :blk false;
                break :blk true;
            } else false;
            if (use_batch) {
                c.controls(ctl, ins, msg, c.batch.?);
                ins = .{ .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n1), .xs = 1, .xb = 0 }, .{ .buf = b(s.n2), .xs = 1, .xb = 0 } };
            } else {
                for (0..3) |j| {
                    const nbuf = [_]B{ b(s.n0), b(s.n1), b(s.n2) };
                    c.control(ctl[j], ins[j], .{ .buf = msg, .xs = 4, .xb = @intCast(j) }, nbuf[j], 1, 0);
                    ins[j] = .{ .buf = nbuf[j], .xs = 1, .xb = 0 };
                }
            }
        }

        for (0..3) |j| c.record(slot, j, ins[j]);
        if (fused_norm) {
            ins = if (full) .{ .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n1), .xs = 1, .xb = 0 }, .{ .buf = b(s.n2), .xs = 1, .xb = 0 } } else .{ .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n0), .xs = 1, .xb = 0 } };
        } else if (full or bank != null) {
            c.setP(s.pipes.rms3, Rms3{ .width = c.h, .xs = .{ ins[0].xs, ins[1].xs, ins[2].xs }, .xb = .{ ins[0].xb, ins[1].xb, ins[2].xb }, .eps = text.eps }, &.{ ins[0].buf, ins[1].buf, ins[2].buf, t(lay.q_norm), t(if (full) lay.k_norm else lay.q_norm), t(if (full) lay.v_norm else lay.q_norm), A, b(s.n0), b(s.n1), b(s.n2) });
            c.enc.dispatchGroups(mtl.Size.of(3 * c.n, 1, 1), mtl.Size.of(256, 1, 1));
            ins = .{ .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n1), .xs = 1, .xb = 0 }, .{ .buf = b(s.n2), .xs = 1, .xb = 0 } };
        } else {
            c.rmsW(ins[0], t(lay.q_norm), b(s.n0), 1, 0, c.h, text.eps);
            ins = .{ .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, .{ .buf = b(s.n0), .xs = 1, .xb = 0 } };
        }

        for (0..3) |j| c.record(slot, 3 + j, ins[j]);
        const push = ip % text.block == 0;
        const added = c.attention(lay, slot, full, ins, push);
        c.record(slot, 6, .{ .buf = b(s.o), .xs = 1, .xb = 0 });
        if (push) {
            c.pushBlock();
        } else if (!added) c.addN(b(s.x), b(s.o), @as(usize, c.n) * c.h);

        c.record(slot, 7, .{ .buf = b(s.x), .xs = 1, .xb = 0 });
        const mlp_query = m.weights.mlp_queries[index];
        const mlp_fused = bank == null and c.mixNorm(b(s.x), mlp_query, bo(s.mixout, 0), &.{lay.mlp_norm}, .{ b(s.m0), b(s.m0), b(s.m0) });
        if (!mlp_fused and !c.mixNorm(b(s.x), mlp_query, bo(s.mixout, 0), &.{}, normed)) c.mixBlocks(b(s.x), mlp_query, bo(s.mixout, 0));
        if (bank) |_| {
            const ctl = lay.controllers[3] orelse return error.MissingController;
            c.control(ctl, .{ .buf = bo(s.mixout, 0), .xs = 1, .xb = 0 }, .{ .buf = b(s.msg), .xs = 4, .xb = 3 }, bo(s.mixout, 0), 1, 0);
        }
        if (!mlp_fused) c.rmsW(.{ .buf = bo(s.mixout, 0), .xs = 1, .xb = 0 }, t(lay.mlp_norm), b(s.m0), 1, 0, c.h, text.eps);
        c.experts(lay, b(s.m0), slot);
        const slots = c.rings(slot);
        if (!c.convolveAdd(b(s.moe), null, lay.mlp_post, slots[3], slot, .mlp, b(s.x))) {
            c.convolve(b(s.moe), b(s.aconv), lay.mlp_post, null, undefined, b(s.moe), slots[3], c.h, 0, 0, slot, .mlp, false);
            if (c.batch == null) c.addN(b(s.x), b(s.aconv), @as(usize, c.n) * c.h);
        }
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
    sent: [8]mtl.CommandBuffer = undefined,
    count: usize = 0,

    fn flush(p: *ProfileRounds, c: *Chunk) void {
        if (p.enabled or c.batch != null or !c.s.fuse or p.count == p.sent.len) return;
        c.enc.end();
        p.cb.commit();
        p.sent[p.count] = p.cb;
        p.count += 1;
        p.cb = p.queue.commandBuffer();
        c.enc = p.cb.compute(.serial);
    }

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

const BatchContext = struct {
    scratch: *@import("batch.zig").BatchScratch,
    caches: []const *Cache,
    group: ?*const @import("group.zig").Group = null,
    finish: ?@import("group.zig").Finish = null,
};

pub fn prefill(m: *const Model, s: *PrefillScratch, cache: *Cache, ids: []const u32) !void {
    return prefillWant(m, s, cache, ids, .all);
}

pub fn prefillWant(m: *const Model, s: *PrefillScratch, cache: *Cache, ids: []const u32, want: WantLogits) !void {
    return run(m, s, cache, ids, want, null);
}

pub fn batchStep(m: *const Model, s: *@import("batch.zig").BatchScratch, caches: []const *Cache, ids: []const u32) !void {
    var context = BatchContext{ .scratch = s, .caches = caches };
    try run(m, &s.rows, caches[0], ids, .all, &context);
}

/// The batch step over grouped rows (`group.Group`): every row's logits, the caches untouched until its commit.
pub fn groupStep(m: *const Model, s: *@import("batch.zig").BatchScratch, group: *const @import("group.zig").Group, ids: []const u32) !void {
    var context = BatchContext{ .scratch = s, .caches = group.caches, .group = group };
    try run(m, &s.rows, group.caches[0], ids, .all, &context);
}

pub fn groupFinish(m: *const Model, s: *@import("batch.zig").BatchScratch, group: *const @import("group.zig").Group, ids: []const u32, finish: @import("group.zig").Finish) !void {
    var context = BatchContext{ .scratch = s, .caches = group.caches, .group = group, .finish = finish };
    try run(m, &s.rows, group.caches[0], ids, .all, &context);
}

const group_rounds = [_][]const u8{ "inkling2-group-1", "inkling2-group-2", "inkling2-group-3", "inkling2-group-4", "inkling2-group-5", "inkling2-group-6", "inkling2-group-7", "inkling2-group-8" };

fn roundName(batch: ?*BatchContext, rows: usize) []const u8 {
    const context = batch orelse return "inkling2-prefill";
    if (context.group == null) return "inkling2-batch";
    return group_rounds[@min(rows, group_rounds.len) - 1];
}

fn run(m: *const Model, s: *PrefillScratch, cache: *Cache, ids: []const u32, want: WantLogits, batch: ?*BatchContext) !void {
    if (ids.len == 0 or ids.len > s.chunk or (want == .all and ids.len > s.logit_rows)) return error.BadChunk;
    if (batch == null and ids.len + cache.len > cache.capacity) return error.ContextFull;
    if (s.taps) |taps| if (taps.rows < ids.len) return error.TapsTooSmall;
    if (s.half_overflow) |buf| buf.slice(u32, 1)[0] = 0;
    @memcpy(s.ids.slice(u32, ids.len), ids);
    const kvw: usize = m.config.text.local.kv * m.config.text.local.dim;
    const linear: usize = m.config.slots() * cache.capacity * kvw * 2;
    const ring = cache.keys.length() != linear;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    m.queue.roundBegin(roundName(batch, ids.len));
    defer m.queue.roundEnd();
    const threshold = if (std.c.getenv("TF_INKLING_PREFILL_PROFILE_BASE")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0 else null;
    const profiling = if (threshold) |base| batch == null and cache.len >= base else false;
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
        .batch = batch,
    };
    if (m.weights.embed.affine) |q| {
        c.setP(m.affine_pipes.?.embed, EmbedA{ .width = s.h, .rows = c.n, .eps = text.eps }, &.{ b(s.ids), t(m.weights.embed), bo(q.scales, q.scale_offset), bo(q.biases, q.bias_offset), t(m.weights.embed_norm), A, b(s.x) });
    } else c.setP(s.pipes.embed, EmbedA{ .width = s.h, .rows = c.n, .eps = text.eps }, &.{ b(s.ids), t(m.weights.embed), t(m.weights.embed_norm), A, b(s.x) });
    c.enc.dispatchGroups(mtl.Size.of(c.n, 1, 1), mtl.Size.of(256, 1, 1));
    var profile_name: [64]u8 = undefined;
    for (0..config.loop.core_start) |i| {
        if (profiling) try rounds.next(&c, try std.fmt.bufPrint(&profile_name, "inkling2-slot-{d}", .{i}));
        c.tap(@intCast(i));
        try c.layer(@intCast(i), @intCast(i), null);
        try profileOccupancy(&rounds, &c, @intCast(i), "inkling2-occupancy");
        if (i % 6 == 5) rounds.flush(&c);
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
            if (profiling) try rounds.next(&c, try std.fmt.bufPrint(&profile_name, "inkling2-pass-{d}-copy", .{pass}));
            c.setP(s.pipes.pass_copy, PassCopy{ .width = c.h, .stride = st.mix_states, .start = @intCast(prelude + 1), .count = @intCast(bn) }, &.{ b(s.blocks), b(s.x), b(s.prelude), b(s.bank), b(s.passout), A });
            c.enc.dispatchThreads(mtl.Size.of(c.h, c.n, 1), mtl.Size.of(256, 1, 1));
            stored = .{ .states = b(s.bank), .out = b(s.passout), .n = bn };
            bk = stored;
            c.count = prelude;
        }
        for (config.loop.core_start..config.loop.core_end) |ip| {
            const slot: u32 = if (pass == 0) @intCast(ip) else @intCast(text.layers + (pass - 1) * span + (ip - config.loop.core_start));
            if (profiling) try rounds.next(&c, try std.fmt.bufPrint(&profile_name, "inkling2-slot-{d}", .{slot}));
            c.tap(@intCast(config.loop.core_start + pass * span + (ip - config.loop.core_start)));
            try c.layer(slot, @intCast(ip), bk);
            try profileOccupancy(&rounds, &c, slot, "inkling2-occupancy");
            if (slot % 6 == 5) rounds.flush(&c);
        }
    }
    for (config.loop.core_end..text.layers) |i| {
        if (profiling) try rounds.next(&c, try std.fmt.bufPrint(&profile_name, "inkling2-slot-{d}", .{i}));
        c.tap(@intCast(config.loop.core_start + config.loop.passes * span + (i - config.loop.core_end)));
        try c.layer(@intCast(i), @intCast(i), null);
        try profileOccupancy(&rounds, &c, @intCast(i), "inkling2-occupancy");
    }
    if (profiling) try rounds.next(&c, "inkling2-head");
    c.mixBlocks(b(s.x), m.weights.output_query, bo(s.mixout, 0));
    c.rmsW(.{ .buf = bo(s.mixout, 0), .xs = 1, .xb = 0 }, t(m.weights.norm), b(s.n0), 1, 0, c.h, text.eps);
    if (s.taps) |taps| taps.encode(c.enc, taps.final, s.n0, 0, c.n);
    if (profiling) try rounds.next(&c, "inkling2-vocabulary");
    switch (want) {
        .none => {},
        .last => c.headModelLast(m.weights.unembed, bo(s.n0, (@as(usize, c.n) - 1) * c.h * 4), b(s.logits), text.unpadded_vocab),
        .all => c.headModel(m.weights.unembed, .{ .buf = b(s.n0), .xs = 1, .xb = 0 }, b(s.logits), text.unpadded_vocab),
    }
    if (batch) |context| if (context.group) |g| if (context.finish) |finish| g.verifier.encodeFinish(c.enc, m, s.ids, s.logits, g, finish);
    c.enc.end();
    rounds.cb.commit();
    rounds.cb.wait();
    s.gpu_seconds = 0;
    for (rounds.sent[0..rounds.count]) |sent| {
        if (sent.failure()) |msg| {
            std.log.err("inkling2_loop prefill: {s}", .{msg});
            return error.GpuFailed;
        }
        s.gpu_seconds += sent.gpuSeconds();
    }
    if (rounds.cb.failure()) |msg| {
        std.log.err("inkling2_loop prefill: {s}", .{msg});
        return error.GpuFailed;
    }
    s.gpu_seconds += rounds.cb.gpuSeconds();
    if (s.half_overflow) |buf| {
        const overflow = buf.slice(u32, 1)[0];
        if (s.fast) {
            std.debug.print("inkling2 half activation overflow: {d}\n", .{overflow});
        } else if (overflow != 0) {
            cache.len = 0;
            return error.HalfActivationOverflow;
        }
    }
    if (batch) |value| {
        if (value.group == null) for (value.caches) |item| {
            item.len += 1;
        };
    } else cache.len += ids.len;
}
