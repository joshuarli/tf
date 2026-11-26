const std = @import("std");
const mtl = @import("metal");
const fmt = @import("format.zig");
const ba = @import("binary_archive.zig");
const rt_mod = @import("runtime.zig");

const mix_states: u32 = 5;
const conv_ring: u32 = 4;
const attn_splits: u32 = 16;
pub const default_chunk: usize = 1024;
pub const max_verify_rows: usize = 16;
const none_u8: u8 = 0xff;
pub const dflash_tap_visits = [_]u32{ 5, 23, 41, 48 };

const EmbedA = extern struct { width: u32, rows: u32, eps: f32 };
const PairA = extern struct { count: u32 };
const PassCopyA = extern struct { width: u32, stride: u32, start: u32, count: u32 };
const MixArgs = extern struct { width: u32, depth: u32, stride: u32, k: u32, norm: u32, scale: f32, eps: f32 };
const RmsBA = extern struct { width: u32, rows: u32, xs: u32, xb: u32, ys: u32, yb: u32, wb: u32, eps: f32 };
const Rms3A = extern struct { width: u32, xs: [3]u32, xb: [3]u32, eps: f32 };
const RmsPairA = extern struct { width: u32, xs0: u32, xb0: u32, xs1: u32, xb1: u32, eps: f32 };
const StageArgs = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, bits: u32, cap: f32 };
const PartArgs = extern struct { tile: u32, row: u32, n: u32, pad: u32 = 0 };
const GemmArgs = extern struct { k: u32, rows: u32, count: u32, pad: u32 = 0, part: [6]PartArgs };
const GateArgs = extern struct { count: u32 };
const GateCtlA = extern struct { h: u32, rows: u32, k: u32, cxs: u32, cxb: u32, mxs: u32, mxb: u32, ys: u32, yb: u32 };
const ConvBA = extern struct { channels: u32, heads: u32, ring: u32, base: u32, rows: u32, layout: u32, norm: f32 };
const RingA = extern struct { channels: u32, ring: u32, base: u32, rows: u32, tail: u32 };
const ConvAddA = extern struct { channels: u32, heads: u32, ring: u32, base: u32, rows: u32, layout: u32, norm: f32, out: u32 };
const ClipA = extern struct { heads: u32, kv: u32, items: u32, q_threshold: f32, rel_threshold: f32, eps: f32 };
const RelBA = extern struct { heads: u32, rel: u32, extent: u32, rows: u32, base: u32, tau: u32, floor: f32, alpha: f32 };
const AttnBA = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, base: u32, rows: u32, rmode: u32, scale: f32, floor: f32, alpha: f32 };
const KvA = extern struct { count: u32, base: u32, rows: u32, wring: u32 };
const ScatterA = extern struct { width: u32, stride: u32, slot: u32 };
const PickA = extern struct { experts: u32, topk: u32, rows: u32, route_scale: f32 };
const MoegA = extern struct { k: u32, n: u32, slots: u32, topk: u32, experts: u32, gate_cap: f32, up_cap: f32 };
const CombineA = extern struct { width: u32, topk: u32, rows: u32 };
const TapA = extern struct { width: u32, rows: u32, stride: u32, column: u32 };
const KvPermA = extern struct { rows: u32, stride_rows: u32, kv: u32, dim: u32 };

const TProjA = extern struct { k: u32, full: u32, rows: u32, m: [8]u32 };
const TMoeA = extern struct { k: u32, n: u32, slots: u32, topk: u32, gate_cap: f32, up_cap: f32 };
const CtlDownR = extern struct { h: u32, rank: u32, rows: u32, msg0: u32, xs: [3]u32, xb: [3]u32 };
const CtlApply3R = extern struct { h: u32, rank: u32, rows: u32, xs: [3]u32, xb: [3]u32, msg0: u32 = 0, cols: u32 = 1 };
const CtlApply1R = extern struct { k: u32, m0: u32, m1: u32, h: u32, cxs: u32, cxb: u32, mxs: u32, mxb: u32, ys: u32, yb: u32 };
const QkvArgs = extern struct { kv: u32, heads: u32, dim: u32, rel: u32, extent: u32, layout: u32, dyn: u32, tau: u32, norm: f32, q_threshold: f32, rel_threshold: f32, eps: f32, floor: f32, alpha: f32 };
const VQkvRow = extern struct { k_ring: u64, v_ring: u64, pos: u32, start: u32, back: [4]u8, pad: u32 = 0 };
const DecodeParts = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, splits: u32, chunk: u32, scale: f32, floor: f32, alpha: f32 };
const BatchRow = extern struct { keys: u64, values: u64, args: DecodeParts };
const VGroupRow = extern struct { keys: u64, values: u64, args: DecodeParts, gpos: u32, path: [max_verify_rows]u8 };
const VConvRow = extern struct { ring: u64, pos: u32, start: u32, back: [4]u8 };
const VConvArgs = extern struct { channels: u32, heads: u32, ring: u32, pos: u32, layout: u32, norm: f32, residual: u32, topk: u32 };
const VStream = extern struct { first: u32, size: u32 };
const VCommitSlot = extern struct { local: u32, locals_before: u32, globals_before: u32, pad: u32 = 0 };
const VCommitArgs = extern struct { kvw: u32, hidden: u32, rows: u32, stride: u32, slots: u32, window: u32, ring: u32 };
const VRoundRow = extern struct { keys: u64, values: u64, conv: [4]u64, pos: u32, capacity: u32, owner: u32, depth: u32 };

pub const ExtraPipes = struct {
    conv: mtl.Pipeline,
    relbias_rows: mtl.Pipeline,
    moe_zero: mtl.Pipeline,
    moe_count: mtl.Pipeline,
    moe_offsets: mtl.Pipeline,
    moe_sort: mtl.Pipeline,
    combine: mtl.Pipeline,
    half_rows: mtl.Pipeline,
    attn_units: mtl.Pipeline,
    expert_units_down: mtl.Pipeline,
    affine_units_up: mtl.Pipeline,
    affine_units_down: mtl.Pipeline,
    mix_norm: mtl.Pipeline,
    clips: mtl.Pipeline,
    ring_store: mtl.Pipeline,
    kv_store: mtl.Pipeline,
    conv_add: mtl.Pipeline,
    as_split: mtl.Pipeline,
    as_gemm_split: mtl.Pipeline,
    ps_gemm_split: mtl.Pipeline,
    ps_gemm_hi: mtl.Pipeline,
    ps_split_gated: mtl.Pipeline,
    af_gemm: mtl.Pipeline,
    af_gate_ctl: mtl.Pipeline,
    tap: mtl.Pipeline,
    as_prepare_sb: mtl.Pipeline,
    unpack_gf2: mtl.Pipeline,
    expert_units_up_packed: mtl.Pipeline,
    kv_to_perm: mtl.Pipeline,
    v_qkv: mtl.Pipeline,
    v_split: mtl.Pipeline,
    v_conv: mtl.Pipeline,
    v_accept: mtl.Pipeline,
    v_round_kv: mtl.Pipeline,
    v_round_conv: mtl.Pipeline,
    e_down: mtl.Pipeline,
    e_down16: mtl.Pipeline,
    r_controls_down: mtl.Pipeline,
    t_dense_8_2_sb: mtl.Pipeline,
    t_dense16_8_1_sb: mtl.Pipeline,
    t_proj_8_2_sb: mtl.Pipeline,
    t_proj16_8_1_sb: mtl.Pipeline,
    e_gu_packed: mtl.Pipeline,
    e_gu16_packed: mtl.Pipeline,
    ctl_apply3_gf2_rows: mtl.Pipeline,
    ctl_apply1_gf2_rows: mtl.Pipeline,

    pub fn load(device: mtl.Device, lib: mtl.Library, archive: ba.BinaryArchive, ind: bool) !ExtraPipes {
        return .{
            .conv = try archive.loadPipeline(device, lib, "i2p_conv", ind),
            .relbias_rows = try archive.loadPipeline(device, lib, "i2p_relbias_rows", ind),
            .moe_zero = try archive.loadPipeline(device, lib, "i2p_moe_zero", ind),
            .moe_count = try archive.loadPipeline(device, lib, "i2p_moe_count", ind),
            .moe_offsets = try archive.loadPipeline(device, lib, "i2p_moe_offsets", ind),
            .moe_sort = try archive.loadPipeline(device, lib, "i2p_moe_sort", ind),
            .combine = try archive.loadPipeline(device, lib, "i2p_combine", ind),
            .half_rows = try archive.loadPipeline(device, lib, "i2p_half_rows", ind),
            .attn_units = try archive.loadPipeline(device, lib, "i2p_attn_units", ind),
            .expert_units_down = try archive.loadPipeline(device, lib, "i2p_expert_units_down", ind),
            .affine_units_up = try archive.loadPipeline(device, lib, "i2p_affine_units_up", ind),
            .affine_units_down = try archive.loadPipeline(device, lib, "i2p_affine_units_down", ind),
            .mix_norm = try archive.loadPipeline(device, lib, "i2pr_mix_norm", ind),
            .clips = try archive.loadPipeline(device, lib, "i2pr_clips", ind),
            .ring_store = try archive.loadPipeline(device, lib, "i2pr_ring_store", ind),
            .kv_store = try archive.loadPipeline(device, lib, "i2pr_kv_store", ind),
            .conv_add = try archive.loadPipeline(device, lib, "i2pr_conv_add", ind),
            .as_split = try archive.loadPipeline(device, lib, "i2as_split", ind),
            .as_gemm_split = try archive.loadPipeline(device, lib, "i2as_gemm_split", ind),
            .ps_gemm_split = try archive.loadPipeline(device, lib, "i2ps_gemm_split", ind),
            .ps_gemm_hi = try archive.loadPipeline(device, lib, "i2ps_gemm_hi", ind),
            .ps_split_gated = try archive.loadPipeline(device, lib, "i2ps_split_gated", ind),
            .af_gemm = try archive.loadPipeline(device, lib, "i2af_gemm", ind),
            .af_gate_ctl = try archive.loadPipeline(device, lib, "i2af_gate_ctl", ind),
            .tap = try archive.loadPipeline(device, lib, "i2t_tap", ind),
            .as_prepare_sb = try archive.loadPipeline(device, lib, "i2_aot_as_prepare_sb", ind),
            .unpack_gf2 = try archive.loadPipeline(device, lib, "i2_aot_unpack_gf2", ind),
            .expert_units_up_packed = try archive.loadPipeline(device, lib, "i2_aot_expert_units_up_packed", ind),
            .kv_to_perm = try archive.loadPipeline(device, lib, "i2_aot_kv_to_perm", ind),
            .v_qkv = try archive.loadPipeline(device, lib, "i2v_qkv", ind),
            .v_split = try archive.loadPipeline(device, lib, "i2v_split", ind),
            .v_conv = try archive.loadPipeline(device, lib, "i2v_conv", ind),
            .v_accept = try archive.loadPipeline(device, lib, "i2v_accept", ind),
            .v_round_kv = try archive.loadPipeline(device, lib, "i2v_round_kv", ind),
            .v_round_conv = try archive.loadPipeline(device, lib, "i2v_round_conv", ind),
            .e_down = try archive.loadPipeline(device, lib, "i2e_down", ind),
            .e_down16 = try archive.loadPipeline(device, lib, "i2e_down16", ind),
            .r_controls_down = try archive.loadPipeline(device, lib, "i2r_controls_down", ind),
            .t_dense_8_2_sb = try archive.loadPipeline(device, lib, "i2_aot_t_dense_8_2_sb", ind),
            .t_dense16_8_1_sb = try archive.loadPipeline(device, lib, "i2_aot_t_dense16_8_1_sb", ind),
            .t_proj_8_2_sb = try archive.loadPipeline(device, lib, "i2_aot_t_proj_8_2_sb", ind),
            .t_proj16_8_1_sb = try archive.loadPipeline(device, lib, "i2_aot_t_proj16_8_1_sb", ind),
            .e_gu_packed = try archive.loadPipeline(device, lib, "i2_aot_e_gu_packed", ind),
            .e_gu16_packed = try archive.loadPipeline(device, lib, "i2_aot_e_gu16_packed", ind),
            .ctl_apply3_gf2_rows = try archive.loadPipeline(device, lib, "i2_aot_ctl_apply3_gf2_rows", ind),
            .ctl_apply1_gf2_rows = try archive.loadPipeline(device, lib, "i2_aot_ctl_apply1_gf2_rows", ind),
        };
    }

    pub fn deinit(e: *ExtraPipes) void {
        const names = [_][]const u8{
            "conv", "relbias_rows", "moe_zero", "moe_count", "moe_offsets", "moe_sort",
            "combine", "half_rows", "attn_units", "expert_units_down", "affine_units_up",
            "affine_units_down", "mix_norm", "clips", "ring_store", "kv_store", "conv_add",
            "as_split", "as_gemm_split", "ps_gemm_split", "ps_gemm_hi", "ps_split_gated",
            "af_gemm", "af_gate_ctl", "tap", "as_prepare_sb", "unpack_gf2",
            "expert_units_up_packed", "kv_to_perm", "v_qkv", "v_split", "v_conv",
            "v_accept", "v_round_kv", "v_round_conv", "e_down", "e_down16", "r_controls_down",
            "t_dense_8_2_sb", "t_dense16_8_1_sb", "t_proj_8_2_sb", "t_proj16_8_1_sb",
            "e_gu_packed", "e_gu16_packed", "ctl_apply3_gf2_rows", "ctl_apply1_gf2_rows",
        };
        inline for (names) |name| @field(e, name).deinit();
    }
};

pub const PrefillScratch = struct {
    chunk: usize,
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
    moe: mtl.Buffer,
    blocks: mtl.Buffer,
    mixout: mtl.Buffer,
    msg: mtl.Buffer,
    h2in: mtl.Buffer,
    normed: mtl.Buffer,
    router: mtl.Buffer,
    picks: mtl.Buffer,
    wts: mtl.Buffer,
    act: mtl.Buffer,
    act_h: mtl.Buffer,
    xh: mtl.Buffer,
    down: mtl.Buffer,
    moe_cnt: mtl.Buffer,
    moe_off: mtl.Buffer,
    moe_cur: mtl.Buffer,
    moe_sorted: mtl.Buffer,
    stage_w: mtl.Buffer,
    stage_x: mtl.Buffer,
    overflow: mtl.Buffer,
    taps: mtl.Buffer,

    pub fn init(device: mtl.Device, g: fmt.Geometry, chunk: usize) !PrefillScratch {
        const priv = mtl.ResourceOptions.private;
        const shr = mtl.ResourceOptions.shared;
        const h: usize = g.hidden;
        const qw: usize = g.heads * g.head_dim;
        const kvw: usize = g.kv_heads * g.head_dim;
        const rel_cols: usize = g.rel_extent + 1;
        const slots_max: usize = chunk * g.topk;
        return .{
            .chunk = chunk,
            .ids = try device.buffer(chunk * 4, shr),
            .x = try device.buffer(chunk * h * 4, priv),
            .prelude = try device.buffer(chunk * h * 4, priv),
            .passout = try device.buffer(chunk * h * 4, priv),
            .bank = try device.buffer(chunk * mix_states * h * 4, priv),
            .n0 = try device.buffer(chunk * h * 4, priv),
            .n1 = try device.buffer(chunk * h * 4, priv),
            .n2 = try device.buffer(chunk * h * 4, priv),
            .m0 = try device.buffer(chunk * h * 4, priv),
            .q = try device.buffer(chunk * qw * 4, priv),
            .k = try device.buffer(chunk * kvw * 4, priv),
            .v = try device.buffer(chunk * kvw * 4, priv),
            .r = try device.buffer(chunk * g.heads * g.d_rel * 4, priv),
            .kc = try device.buffer(chunk * kvw * 4, priv),
            .vc = try device.buffer(chunk * kvw * 4, priv),
            .kh = try device.buffer(chunk * kvw * 2, priv),
            .vh = try device.buffer(chunk * kvw * 2, priv),
            .kdyn = try device.buffer(chunk * g.kv_heads * 4 * 4, priv),
            .vdyn = try device.buffer(chunk * g.kv_heads * 4 * 4, priv),
            .sink = try device.buffer(chunk * g.heads * 4, priv),
            .relbias = try device.buffer(chunk * g.heads * rel_cols * 4, priv),
            .attn = try device.buffer(chunk * qw * 4, priv),
            .gate = try device.buffer(chunk * qw * 4, priv),
            .projout = try device.buffer(chunk * h * 4, priv),
            .o = try device.buffer(chunk * h * 4, priv),
            .moe = try device.buffer(chunk * h * 4, priv),
            .blocks = try device.buffer(chunk * mix_states * h * 4, priv),
            .mixout = try device.buffer(chunk * 4 * h * 4, priv),
            .msg = try device.buffer(chunk * 4 * h * 4, priv),
            .h2in = try device.buffer(chunk * 2 * h * 4, priv),
            .normed = try device.buffer(chunk * g.gate_rank * 4, priv),
            .router = try device.buffer(chunk * g.experts * 4, priv),
            .picks = try device.buffer(slots_max * 4, priv),
            .wts = try device.buffer(slots_max * 4, priv),
            .act = try device.buffer(slots_max * g.expert_inter * 4, priv),
            .act_h = try device.buffer(slots_max * g.expert_inter * 2, priv),
            .xh = try device.buffer(chunk * h * 2, priv),
            .down = try device.buffer(slots_max * h * 4, priv),
            .moe_cnt = try device.buffer(g.experts * 4, priv),
            .moe_off = try device.buffer((g.experts + 1) * 4, priv),
            .moe_cur = try device.buffer(g.experts * 4, priv),
            .moe_sorted = try device.buffer(slots_max * 4, priv),
            .stage_w = try device.buffer(9000 * qw * 2, priv),
            .stage_x = try device.buffer(2 * chunk * qw * 2, priv),
            .overflow = try device.buffer(16, shr),
            .taps = try device.buffer(chunk * dflash_tap_visits.len * h * 2, shr),
        };
    }

    pub fn deinit(s: *PrefillScratch) void {
        const names = [_][]const u8{
            "ids", "x", "prelude", "passout", "bank", "n0", "n1", "n2", "m0", "q", "k", "v", "r",
            "kc", "vc", "kh", "vh", "kdyn", "vdyn", "sink", "relbias", "attn", "gate", "projout",
            "o", "moe", "blocks", "mixout", "msg", "h2in", "normed", "router", "picks", "wts",
            "act", "act_h", "xh", "down", "moe_cnt", "moe_off", "moe_cur", "moe_sorted",
            "stage_w", "stage_x", "overflow", "taps",
        };
        inline for (names) |name| @field(s, name).deinit();
    }
};

pub const VerifyScratch = struct {
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
    kdyn: mtl.Buffer,
    vdyn: mtl.Buffer,
    sink: mtl.Buffer,
    relbias: mtl.Buffer,
    attn: mtl.Buffer,
    gate: mtl.Buffer,
    projout: mtl.Buffer,
    o: mtl.Buffer,
    blocks: mtl.Buffer,
    mixout: mtl.Buffer,
    msg: mtl.Buffer,
    ctl_hid: mtl.Buffer,
    router: mtl.Buffer,
    picks: mtl.Buffer,
    wts: mtl.Buffer,
    act: mtl.Buffer,
    down: mtl.Buffer,
    parts: mtl.Buffer,
    logits: mtl.Buffer,
    v_kv: mtl.Buffer,
    v_conv: mtl.Buffer,
    targets: mtl.Buffer,
    accepted: mtl.Buffer,
    keep: mtl.Buffer,
    kept: mtl.Buffer,
    taps: mtl.Buffer,

    pub fn init(device: mtl.Device, g: fmt.Geometry) !VerifyScratch {
        const priv = mtl.ResourceOptions.private;
        const shr = mtl.ResourceOptions.shared;
        const R: usize = max_verify_rows;
        const h: usize = g.hidden;
        const qw: usize = g.heads * g.head_dim;
        const kvw: usize = g.kv_heads * g.head_dim;
        const rel_cols: usize = g.rel_extent + 1;
        const slots_max: usize = R * g.topk;
        return .{
            .ids = try device.buffer(R * 4, shr),
            .x = try device.buffer(R * h * 4, priv),
            .prelude = try device.buffer(R * h * 4, priv),
            .passout = try device.buffer(R * h * 4, priv),
            .bank = try device.buffer(R * mix_states * h * 4, priv),
            .n0 = try device.buffer(R * h * 4, priv),
            .n1 = try device.buffer(R * h * 4, priv),
            .n2 = try device.buffer(R * h * 4, priv),
            .m0 = try device.buffer(R * h * 4, priv),
            .q = try device.buffer(R * qw * 4, priv),
            .k = try device.buffer(R * kvw * 4, priv),
            .v = try device.buffer(R * kvw * 4, priv),
            .r = try device.buffer(R * g.heads * g.d_rel * 4, priv),
            .kc = try device.buffer(R * kvw * 4, priv),
            .vc = try device.buffer(R * kvw * 4, priv),
            .kdyn = try device.buffer(R * g.kv_heads * 4 * 4, priv),
            .vdyn = try device.buffer(R * g.kv_heads * 4 * 4, priv),
            .sink = try device.buffer(R * g.heads * 4, priv),
            .relbias = try device.buffer(R * g.heads * rel_cols * 4, priv),
            .attn = try device.buffer(R * qw * 4, priv),
            .gate = try device.buffer(R * qw * 4, priv),
            .projout = try device.buffer(R * h * 4, priv),
            .o = try device.buffer(R * h * 4, priv),
            .blocks = try device.buffer(R * mix_states * h * 4, priv),
            .mixout = try device.buffer(R * 4 * h * 4, priv),
            .msg = try device.buffer(R * 4 * h * 4, priv),
            .ctl_hid = try device.buffer(3 * R * g.gate_rank * 4, priv),
            .router = try device.buffer(R * g.experts * 4, priv),
            .picks = try device.buffer(slots_max * 4, priv),
            .wts = try device.buffer(slots_max * 4, priv),
            .act = try device.buffer(slots_max * g.expert_inter * 4, priv),
            .down = try device.buffer(slots_max * h * 4, priv),
            .parts = try device.buffer(R * g.heads * attn_splits * (g.head_dim + 2) * 4, priv),
            .logits = try device.buffer(R * g.unpadded_vocab * 4, shr),
            .v_kv = try device.buffer(fmt.num_slots * 2 * R * kvw * 2, priv),
            .v_conv = try device.buffer(fmt.num_slots * R * (2 * kvw + 2 * h) * 4, priv),
            .targets = try device.buffer(R * 4, shr),
            .accepted = try device.buffer(R * 4, shr),
            .keep = try device.buffer(R * 4, shr),
            .kept = try device.buffer(R * 4, shr),
            .taps = try device.buffer(R * dflash_tap_visits.len * h * 2, shr),
        };
    }

    pub fn deinit(s: *VerifyScratch) void {
        const names = [_][]const u8{
            "ids", "x", "prelude", "passout", "bank", "n0", "n1", "n2", "m0", "q", "k", "v", "r",
            "kc", "vc", "kdyn", "vdyn", "sink", "relbias", "attn", "gate", "projout", "o",
            "blocks", "mixout", "msg", "ctl_hid", "router", "picks", "wts", "act", "down",
            "parts", "logits", "v_kv", "v_conv", "targets", "accepted", "keep", "kept", "taps",
        };
        inline for (names) |name| @field(s, name).deinit();
    }

    fn convOffset(g: fmt.Geometry, slot: u32, site: usize) usize {
        const kvw: usize = g.kv_heads * g.head_dim;
        const h: usize = g.hidden;
        const site_w: usize = switch (site) {
            0 => 0,
            1 => kvw,
            2 => 2 * kvw,
            else => 2 * kvw + h,
        };
        return (@as(usize, slot) * max_verify_rows * (2 * kvw + 2 * h) + site_w * max_verify_rows) * 4;
    }
};

pub const PrefillStats = struct {
    wall_ms: f64 = 0,
    gpu_ms: f64 = 0,
    tokens: usize = 0,
    chunks: usize = 0,
    last_token: u32 = 0,
};

pub const VerifyStats = struct {
    wall_ms: f64 = 0,
    gpu_ms: f64 = 0,
    rows: usize = 0,
    accepted_last: u32 = 0,
    kept_tokens: usize = 0,
};

fn attnPartition(n: usize) struct { splits: u32, chunk: u32 } {
    const splits: u32 = @intCast(@max(1, @min(attn_splits, (n + 63) / 64)));
    return .{ .splits = splits, .chunk = @intCast((n + splits - 1) / splits) };
}

fn encodeTap(ep: *const ExtraPipes, enc: mtl.ComputeEncoder, visit: u32, src: mtl.Buffer, taps_buf: mtl.Buffer, h: u32, rows: u32) void {
    for (dflash_tap_visits, 0..) |v, i| {
        if (v != visit) continue;
        enc.setPipeline(ep.tap);
        enc.setBuffer(src, 0, 0);
        enc.setValue(TapA{ .width = h, .rows = rows, .stride = @as(u32, @intCast(dflash_tap_visits.len)) * h, .column = @as(u32, @intCast(i)) * h }, 1);
        enc.setBuffer(taps_buf, 0, 2);
        enc.dispatchThreads(mtl.Size.of(h, rows, 1), mtl.Size.of(256, 1, 1));
    }
}

fn encodeStagedPrepare(rt: *const rt_mod.Runtime, ep: *const ExtraPipes, enc: mtl.ComputeEncoder, s: *const PrefillScratch, ref: fmt.AffineRef, dst_row: usize) void {
    const n: u32 = ref.rows;
    const k: u32 = ref.cols;
    enc.setPipeline(ep.as_prepare_sb);
    rt.setW(enc, ref.w.off, 0);
    rt.setW(enc, ref.sb.off, 1);
    enc.setValue(StageArgs{ .n = n, .k = k, .rows = 1, .xs = 1, .xb = 0, .bits = ref.bits, .cap = 0 }, 2);
    enc.setBuffer(s.stage_w, dst_row * @as(usize, k) * 2, 3);
    enc.setBuffer(s.overflow, 0, 4);
    enc.dispatchThreads(mtl.Size.of(@as(usize, n) * k / 4, 1, 1), mtl.Size.of(256, 1, 1));
}

fn encodeStagedGemmParts(
    ep: *const ExtraPipes,
    enc: mtl.ComputeEncoder,
    s: *const PrefillScratch,
    in_buf: mtl.Buffer,
    k: u32,
    rows: u32,
    xs: u32,
    xb: u32,
    half_inputs: bool,
    parts: []const struct { row: u32, n: u32, out: mtl.Buffer },
) void {
    enc.setPipeline(ep.as_split);
    enc.setBuffer(in_buf, 0, 0);
    enc.setValue(StageArgs{ .n = 0, .k = k, .rows = rows, .xs = xs, .xb = xb, .bits = 0, .cap = 0 }, 1);
    enc.setBuffer(s.stage_x, 0, 2);
    enc.setBuffer(s.overflow, 0, 3);
    enc.dispatchThreads(mtl.Size.of(@as(usize, rows) * k / 4, 1, 1), mtl.Size.of(256, 1, 1));

    var args = GemmArgs{ .k = k, .rows = rows, .count = @intCast(parts.len), .part = undefined };
    var outs: [6]mtl.Buffer = undefined;
    var tiles: u32 = 0;
    for (parts, 0..) |pt, i| {
        args.part[i] = .{ .tile = tiles, .row = pt.row, .n = pt.n };
        outs[i] = pt.out;
        tiles += (pt.n + 63) / 64;
    }
    for (parts.len..6) |i| {
        args.part[i] = .{ .tile = tiles, .row = 0, .n = 0 };
        outs[i] = outs[0];
    }
    enc.setPipeline(if (half_inputs) ep.ps_gemm_hi else ep.ps_gemm_split);
    enc.setBuffer(s.stage_x, 0, 0);
    enc.setBuffer(s.stage_w, 0, 1);
    enc.setValue(args, 2);
    for (outs, 0..) |o, i| enc.setBuffer(o, 0, 3 + i);
    enc.dispatchGroups(mtl.Size.of(tiles, (rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
}

pub fn prefillChunk(
    rt: *rt_mod.Runtime,
    ids: []const u32,
    want_last_logits: bool,
    half_inputs: bool,
    capture_taps: bool,
) !PrefillStats {
    if (ids.len == 0) return .{};
    if (rt.len + ids.len > rt.capacity) return error.ContextFull;
    const ep = &(rt.extra_pipes orelse return error.MissingPrefillPipes);
    const s = try rt.ensurePrefillScratch();
    if (ids.len <= 8) {
        const t0 = rt_mod.nowNs();
        var gpu_ms: f64 = 0;
        var last_tok: u32 = 0;
        for (ids, 0..) |id, i| {
            const st = try rt.stepDirect(id, want_last_logits and i + 1 == ids.len, false);
            gpu_ms += st.gpu_ms;
            if (i + 1 == ids.len) last_tok = st.token;
        }
        return .{ .wall_ms = @as(f64, @floatFromInt(rt_mod.nowNs() - t0)) / 1e6, .gpu_ms = gpu_ms, .tokens = ids.len, .chunks = 1, .last_token = last_tok };
    }

    const t0 = rt_mod.nowNs();
    var total_gpu: f64 = 0;
    var chunks: usize = 0;
    var offset: usize = 0;
    var last_tok: u32 = 0;

    while (offset < ids.len) {
        const take = @min(s.chunk, ids.len - offset);
        const is_last_chunk = offset + take == ids.len;
        if (take <= 8) {
            for (ids[offset .. offset + take], 0..) |id, k| {
                const st = try rt.stepDirect(id, want_last_logits and is_last_chunk and k + 1 == take, false);
                total_gpu += st.gpu_ms;
                if (is_last_chunk and k + 1 == take) last_tok = st.token;
            }
            offset += take;
            chunks += 1;
            continue;
        }
        const gpu_c = try prefillOneChunk(rt, ep, s, ids[offset .. offset + take], want_last_logits and is_last_chunk, half_inputs, capture_taps);
        total_gpu += gpu_c;
        offset += take;
        chunks += 1;
    }
    if (want_last_logits) {
        last_tok = rt_mod.cpuArgmax(rt.logitsSlice());
    }
    return .{
        .wall_ms = @as(f64, @floatFromInt(rt_mod.nowNs() - t0)) / 1e6,
        .gpu_ms = total_gpu,
        .tokens = ids.len,
        .chunks = chunks,
        .last_token = last_tok,
    };
}

fn prefillOneChunk(
    rt: *rt_mod.Runtime,
    ep: *const ExtraPipes,
    s: *const PrefillScratch,
    ids: []const u32,
    want_last_logits: bool,
    half_inputs: bool,
    capture_taps: bool,
) !f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const g = rt.sched.geom;
    const p = &rt.pipes;
    const n: u32 = @intCast(ids.len);
    const base: u32 = @intCast(rt.len);
    const h: u32 = g.hidden;
    const qw: u32 = g.heads * g.head_dim;
    const kvw: u32 = g.kv_heads * g.head_dim;
    const scale_h: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
    const scale_attn: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(g.head_dim)));
    const norm_post: f32 = 1.0 / @sqrt(@as(f32, 5.0));

    s.overflow.slice(u32, 1)[0] = 0;
    @memcpy(s.ids.slice(u32, ids.len), ids);

    var sent: [12]mtl.CommandBuffer = undefined;
    var sent_n: usize = 0;
    var cb = rt.queue.commandBuffer();
    var enc = cb.compute(.serial);

    enc.setPipeline(p.aot_embed_sb.?);
    enc.setBuffer(s.ids, 0, 0);
    rt.setW(enc, rt.sched.globals.embed.w.off, 1);
    rt.setW(enc, rt.sched.globals.embed.sb.off, 2);
    rt.setW(enc, rt.sched.globals.embed_norm.off, 3);
    enc.setValue(EmbedA{ .width = h, .rows = n, .eps = g.eps }, 4);
    enc.setBuffer(s.x, 0, 5);
    enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));

    for (rt.sched.slots) |sl| {
        const lay = &rt.sched.layers[sl.phys];
        const full = sl.is_full != 0;
        const bank = sl.is_bank != 0;
        const push = sl.push_block != 0;

        if (sl.is_pass_start != 0) {
            enc.setPipeline(p.pass_copy);
            enc.setBuffer(s.blocks, 0, 0);
            enc.setBuffer(s.x, 0, 1);
            enc.setBuffer(s.prelude, 0, 2);
            enc.setBuffer(s.bank, 0, 3);
            enc.setBuffer(s.passout, 0, 4);
            enc.setValue(PassCopyA{ .width = h, .stride = mix_states, .start = 2, .count = sl.bank_n }, 5);
            enc.dispatchThreads(mtl.Size.of(h, n, 1), mtl.Size.of(256, 1, 1));
        }

        if (capture_taps) encodeTap(ep, enc, sl.visit, s.x, s.taps, h, n);

        var in0 = s.x;
        var in1 = s.x;
        var in2 = s.x;
        var fused_norm = false;

        if (sl.phys > 0) {
            const qref = rt.sched.globals.attn_queries[sl.phys - 1];
            const kq: u32 = if (full) 3 else 1;
            const q0_off = qref.off;
            const q1_off = if (full) q0_off + @as(u64, h) * 4 else q0_off;
            const q2_off = if (full) q0_off + @as(u64, 2 * h) * 4 else q0_off;
            const do_norm = !bank;
            enc.setPipeline(ep.mix_norm);
            enc.setBuffer(s.blocks, 0, 0);
            enc.setBuffer(s.x, 0, 1);
            rt.setW(enc, q0_off, 2);
            rt.setW(enc, q1_off, 3);
            rt.setW(enc, q2_off, 4);
            rt.setW(enc, q0_off, 5);
            enc.setValue(MixArgs{ .width = h, .depth = sl.block_count_before + 1, .stride = mix_states, .k = kq, .norm = @intFromBool(do_norm), .scale = scale_h, .eps = g.eps }, 6);
            enc.setBuffer(s.mixout, 0, 7);
            rt.setW(enc, if (do_norm) lay.q_norm.off else q0_off, 8);
            rt.setW(enc, if (do_norm and full) lay.k_norm.off else q0_off, 9);
            rt.setW(enc, if (do_norm and full) lay.v_norm.off else q0_off, 10);
            enc.setBuffer(s.n0, 0, 11);
            enc.setBuffer(s.n1, 0, 12);
            enc.setBuffer(s.n2, 0, 13);
            enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
            if (do_norm) {
                in0 = s.n0;
                in1 = if (full) s.n1 else s.n0;
                in2 = if (full) s.n2 else s.n0;
                fused_norm = true;
            }
        }

        if (bank) {
            const at: u64 = sl.cross_pass_at;
            const cpq = rt.sched.globals.cross_pass_q.off;
            const cpkv = rt.sched.globals.cross_pass_kv.off;
            enc.setPipeline(ep.mix_norm);
            enc.setBuffer(s.bank, 0, 0);
            enc.setBuffer(s.passout, 0, 1);
            rt.setW(enc, cpq + (at * 2) * h * 4, 2);
            rt.setW(enc, cpkv + (at * 2) * h * 4, 3);
            rt.setW(enc, cpkv + (at * 2 + 1) * h * 4, 4);
            rt.setW(enc, cpq + (at * 2 + 1) * h * 4, 5);
            enc.setValue(MixArgs{ .width = h, .depth = sl.bank_n + 1, .stride = mix_states, .k = 4, .norm = 0, .scale = scale_h, .eps = g.eps }, 6);
            enc.setBuffer(s.msg, 0, 7);
            rt.setW(enc, cpq, 8);
            rt.setW(enc, cpq, 9);
            rt.setW(enc, cpq, 10);
            enc.setBuffer(s.msg, 0, 11);
            enc.setBuffer(s.msg, 0, 12);
            enc.setBuffer(s.msg, 0, 13);
            enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));

            const kq: u32 = if (full) 3 else 1;
            const nbufs = [_]mtl.Buffer{ s.n0, s.n1, s.n2 };
            const gf_count: u32 = h * g.gate_rank;
            const gf_off: usize = @as(usize, gf_count) * 4;
            for (0..3) |j| {
                const xb: u32 = if (full) @intCast(j) else 0;
                enc.setPipeline(p.rms_pair);
                enc.setBuffer(s.mixout, 0, 0);
                enc.setBuffer(s.msg, 0, 1);
                enc.setValue(RmsPairA{ .width = h, .xs0 = kq, .xb0 = xb, .xs1 = 4, .xb1 = @intCast(j), .eps = g.eps }, 2);
                enc.setBuffer(s.h2in, 0, 3);
                enc.dispatchGroups(mtl.Size.of(2 * n, 1, 1), mtl.Size.of(256, 1, 1));

                enc.setPipeline(ep.af_gemm);
                enc.setBuffer(s.h2in, 0, 0);
                rt.setW(enc, lay.controllers[j].down.off, 1);
                enc.setValue(StageArgs{ .n = g.gate_rank, .k = 2 * h, .rows = n, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, 2);
                enc.setBuffer(s.normed, 0, 3);
                enc.dispatchGroups(mtl.Size.of((g.gate_rank + 31) / 32, (n + 15) / 16, 1), mtl.Size.of(128, 1, 1));

                enc.setPipeline(ep.unpack_gf2);
                rt.setW(enc, lay.controllers[j].gain.off, 0);
                enc.setValue(gf_count, 1);
                enc.setBuffer(s.stage_w, 0, 2);
                enc.setBuffer(s.stage_w, gf_off, 3);
                enc.dispatchThreads(mtl.Size.of(gf_count, 1, 1), mtl.Size.of(256, 1, 1));

                enc.setPipeline(ep.af_gate_ctl);
                enc.setBuffer(s.normed, 0, 0);
                enc.setBuffer(s.stage_w, 0, 1);
                enc.setBuffer(s.stage_w, gf_off, 2);
                enc.setValue(GateCtlA{ .h = h, .rows = n, .k = g.gate_rank, .cxs = kq, .cxb = xb, .mxs = 4, .mxb = @intCast(j), .ys = 1, .yb = 0 }, 3);
                enc.setBuffer(s.mixout, 0, 4);
                enc.setBuffer(s.msg, 0, 5);
                enc.setBuffer(nbufs[j], 0, 6);
                enc.dispatchGroups(mtl.Size.of((h + 31) / 32, (n + 15) / 16, 1), mtl.Size.of(128, 1, 1));
            }
            enc.setPipeline(p.rms3);
            enc.setBuffer(s.n0, 0, 0);
            enc.setBuffer(s.n1, 0, 1);
            enc.setBuffer(s.n2, 0, 2);
            rt.setW(enc, lay.q_norm.off, 3);
            rt.setW(enc, if (full) lay.k_norm.off else lay.q_norm.off, 4);
            rt.setW(enc, if (full) lay.v_norm.off else lay.q_norm.off, 5);
            enc.setValue(Rms3A{ .width = h, .xs = .{ 1, 1, 1 }, .xb = .{ 0, 0, 0 }, .eps = g.eps }, 6);
            enc.setBuffer(s.n0, 0, 7);
            enc.setBuffer(s.n1, 0, 8);
            enc.setBuffer(s.n2, 0, 9);
            enc.dispatchGroups(mtl.Size.of(3 * n, 1, 1), mtl.Size.of(256, 1, 1));
            in0 = s.n0;
            in1 = s.n1;
            in2 = s.n2;
            fused_norm = true;
        }

        if (!fused_norm) {
            enc.setPipeline(p.rms);
            enc.setBuffer(s.x, 0, 0);
            rt.setW(enc, lay.q_norm.off, 1);
            enc.setValue(RmsBA{ .width = h, .rows = n, .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = g.eps }, 2);
            enc.setBuffer(s.n0, 0, 3);
            enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
            in0 = s.n0;
            in1 = s.n0;
            in2 = s.n0;
        }

        if (full) {
            const rq = lay.proj_q.rows;
            const rk = lay.proj_k.rows;
            encodeStagedPrepare(rt, ep, enc, s, lay.proj_q, 0);
            encodeStagedPrepare(rt, ep, enc, s, lay.proj_k, rq);
            encodeStagedPrepare(rt, ep, enc, s, lay.proj_v, rq + rk);
            const m = lay.proj_ms;
            encodeStagedGemmParts(ep, enc, s, in0, h, n, 1, 0, half_inputs, &.{
                .{ .row = 0, .n = m[0], .out = s.q },
                .{ .row = m[0], .n = m[3], .out = s.r },
                .{ .row = m[0] + m[3], .n = m[4], .out = s.gate },
                .{ .row = m[0] + m[3] + m[4], .n = m[5], .out = s.sink },
            });
            encodeStagedGemmParts(ep, enc, s, in1, h, n, 1, 0, half_inputs, &.{
                .{ .row = rq, .n = m[1], .out = s.k },
                .{ .row = rq + m[1], .n = m[6], .out = s.kdyn },
            });
            encodeStagedGemmParts(ep, enc, s, in2, h, n, 1, 0, half_inputs, &.{
                .{ .row = rq + rk, .n = m[2], .out = s.v },
                .{ .row = rq + rk + m[2], .n = m[7], .out = s.vdyn },
            });
        } else {
            encodeStagedPrepare(rt, ep, enc, s, lay.proj_q, 0);
            const m = lay.proj_ms;
            if (in0.id == in1.id and in0.id == in2.id) {
                encodeStagedGemmParts(ep, enc, s, in0, h, n, 1, 0, half_inputs, &.{
                    .{ .row = 0, .n = m[0], .out = s.q },
                    .{ .row = m[0], .n = m[1], .out = s.k },
                    .{ .row = m[0] + m[1], .n = m[2], .out = s.v },
                    .{ .row = m[0] + m[1] + m[2], .n = m[3], .out = s.r },
                    .{ .row = m[0] + m[1] + m[2] + m[3], .n = m[4], .out = s.gate },
                    .{ .row = m[0] + m[1] + m[2] + m[3] + m[4], .n = m[5], .out = s.sink },
                });
            } else {
                encodeStagedGemmParts(ep, enc, s, in0, h, n, 1, 0, half_inputs, &.{
                    .{ .row = 0, .n = m[0], .out = s.q },
                    .{ .row = m[0] + m[1] + m[2], .n = m[3], .out = s.r },
                    .{ .row = m[0] + m[1] + m[2] + m[3], .n = m[4], .out = s.gate },
                    .{ .row = m[0] + m[1] + m[2] + m[3] + m[4], .n = m[5], .out = s.sink },
                });
                encodeStagedGemmParts(ep, enc, s, in1, h, n, 1, 0, half_inputs, &.{
                    .{ .row = m[0], .n = m[1], .out = s.k },
                });
                encodeStagedGemmParts(ep, enc, s, in2, h, n, 1, 0, half_inputs, &.{
                    .{ .row = m[0] + m[1], .n = m[2], .out = s.v },
                });
            }
        }

        const k_ring_off = @as(usize, sl.slot) * conv_ring * kvw * 4;
        const kv_conv_norm: f32 = 1.0 / @sqrt(@as(f32, if (full) 9.0 else 5.0));
        const tail: u32 = @min(n, conv_ring);
        enc.setPipeline(ep.conv);
        enc.setBuffer(s.k, 0, 0);
        enc.setBuffer(rt.k_conv, k_ring_off, 1);
        rt.setW(enc, lay.k_conv.off, 2);
        if (full) enc.setBuffer(s.kdyn, 0, 3) else enc.setValue(ConvBA{ .channels = kvw, .heads = 0, .ring = conv_ring, .base = base, .rows = n, .layout = 0, .norm = kv_conv_norm }, 3);
        enc.setValue(ConvBA{ .channels = kvw, .heads = if (full) g.kv_heads else 0, .ring = conv_ring, .base = base, .rows = n, .layout = sl.is_full, .norm = kv_conv_norm }, 4);
        enc.setBuffer(s.kc, 0, 5);
        enc.dispatchThreads(mtl.Size.of(kvw, n, 1), mtl.Size.of(256, 1, 1));

        enc.setPipeline(ep.ring_store);
        enc.setBuffer(s.k, 0, 0);
        enc.setBuffer(rt.k_conv, k_ring_off, 1);
        enc.setValue(RingA{ .channels = kvw, .ring = conv_ring, .base = base, .rows = n, .tail = tail }, 2);
        enc.dispatchThreads(mtl.Size.of(kvw, tail, 1), mtl.Size.of(256, 1, 1));

        enc.setPipeline(ep.conv);
        enc.setBuffer(s.v, 0, 0);
        enc.setBuffer(rt.v_conv, k_ring_off, 1);
        rt.setW(enc, lay.v_conv.off, 2);
        if (full) enc.setBuffer(s.vdyn, 0, 3) else enc.setValue(ConvBA{ .channels = kvw, .heads = 0, .ring = conv_ring, .base = base, .rows = n, .layout = 0, .norm = kv_conv_norm }, 3);
        enc.setValue(ConvBA{ .channels = kvw, .heads = if (full) g.kv_heads else 0, .ring = conv_ring, .base = base, .rows = n, .layout = sl.is_full, .norm = kv_conv_norm }, 4);
        enc.setBuffer(s.vc, 0, 5);
        enc.dispatchThreads(mtl.Size.of(kvw, n, 1), mtl.Size.of(256, 1, 1));

        enc.setPipeline(ep.ring_store);
        enc.setBuffer(s.v, 0, 0);
        enc.setBuffer(rt.v_conv, k_ring_off, 1);
        enc.setValue(RingA{ .channels = kvw, .ring = conv_ring, .base = base, .rows = n, .tail = tail }, 2);
        enc.dispatchThreads(mtl.Size.of(kvw, tail, 1), mtl.Size.of(256, 1, 1));

        const clip_items = 2 * g.heads + g.kv_heads;
        enc.setPipeline(ep.clips);
        enc.setBuffer(s.q, 0, 0);
        rt.setW(enc, lay.q_scale.off, 1);
        enc.setBuffer(s.r, 0, 2);
        rt.setW(enc, lay.r_scale.off, 3);
        enc.setBuffer(s.kc, 0, 4);
        enc.setBuffer(s.vc, 0, 5);
        enc.setBuffer(s.kh, 0, 6);
        enc.setBuffer(s.vh, 0, 7);
        enc.setValue(ClipA{ .heads = g.heads, .kv = g.kv_heads, .items = clip_items, .q_threshold = g.qk_clip, .rel_threshold = g.rel_clip, .eps = g.eps }, 8);
        enc.dispatchGroups(mtl.Size.of((clip_items + 7) / 8, n, 1), mtl.Size.of(256, 1, 1));

        enc.setPipeline(ep.relbias_rows);
        rt.setW(enc, lay.rel_bias.off, 0);
        enc.setBuffer(s.r, 0, 1);
        enc.setValue(RelBA{ .heads = g.heads, .rel = g.d_rel, .extent = lay.extent, .rows = n, .base = base, .tau = sl.is_full, .floor = @floatFromInt(g.log_floor), .alpha = g.log_alpha }, 2);
        enc.setBuffer(s.relbias, 0, 3);
        enc.dispatchGroups(mtl.Size.of((lay.extent + 1 + 255) / 256, n, 1), mtl.Size.of(256, 1, 1));

        const kv_off = rt.slot_kv_off[sl.slot];
        const win: u32 = if (full) 0 else g.window;
        enc.setPipeline(ep.attn_units);
        enc.setBuffer(s.q, 0, 0);
        enc.setBuffer(rt.keys, kv_off, 1);
        enc.setBuffer(rt.values, kv_off, 2);
        enc.setBuffer(s.relbias, 0, 3);
        enc.setBuffer(s.sink, 0, 4);
        enc.setValue(AttnBA{ .heads = g.heads, .kv = g.kv_heads, .dim = g.head_dim, .rel = g.d_rel, .extent = lay.extent, .window = win, .base = base, .rows = n, .rmode = 1, .scale = scale_attn, .floor = @floatFromInt(g.log_floor), .alpha = g.log_alpha }, 5);
        enc.setBuffer(s.attn, 0, 6);
        enc.setBuffer(s.kh, 0, 7);
        enc.setBuffer(s.vh, 0, 8);
        enc.setBuffer(s.overflow, 0, 9);
        enc.dispatchGroups(mtl.Size.of(g.kv_heads, (n + 31) / 32, 1), mtl.Size.of(192, 1, 1));

        enc.setPipeline(ep.kv_store);
        enc.setBuffer(s.kh, 0, 0);
        enc.setBuffer(s.vh, 0, 1);
        enc.setBuffer(rt.keys, kv_off, 2);
        enc.setBuffer(rt.values, kv_off, 3);
        enc.setValue(KvA{ .count = kvw, .base = base, .rows = n, .wring = win }, 4);
        enc.dispatchThreads(mtl.Size.of(kvw / 4, n, 1), mtl.Size.of(256, 1, 1));

        encodeStagedPrepare(rt, ep, enc, s, lay.proj_o, 0);
        const o_count: u32 = n * qw;
        enc.setPipeline(ep.ps_split_gated);
        enc.setBuffer(s.attn, 0, 0);
        enc.setBuffer(s.gate, 0, 1);
        enc.setValue(GateArgs{ .count = o_count }, 2);
        enc.setBuffer(s.stage_x, 0, 3);
        enc.setBuffer(s.overflow, 0, 4);
        enc.dispatchThreads(mtl.Size.of(o_count / 4, 1, 1), mtl.Size.of(256, 1, 1));

        if (half_inputs) {
            var oargs = GemmArgs{ .k = qw, .rows = n, .count = 1, .part = undefined };
            oargs.part[0] = .{ .tile = 0, .row = 0, .n = h };
            for (1..6) |i| oargs.part[i] = .{ .tile = (h + 63) / 64, .row = 0, .n = 0 };
            enc.setPipeline(ep.ps_gemm_hi);
            enc.setBuffer(s.stage_x, 0, 0);
            enc.setBuffer(s.stage_w, 0, 1);
            enc.setValue(oargs, 2);
            for (0..6) |i| enc.setBuffer(s.projout, 0, 3 + i);
            enc.dispatchGroups(mtl.Size.of((h + 63) / 64, (n + 63) / 64, 1), mtl.Size.of(128, 1, 1));
        } else {
            enc.setPipeline(ep.as_gemm_split);
            enc.setBuffer(s.stage_x, 0, 0);
            enc.setBuffer(s.stage_w, 0, 1);
            enc.setValue(StageArgs{ .n = h, .k = qw, .rows = n, .xs = 1, .xb = 0, .bits = 8, .cap = 0 }, 2);
            enc.setBuffer(s.projout, 0, 3);
            enc.dispatchGroups(mtl.Size.of((h + 63) / 64, (n + 63) / 64, 1), mtl.Size.of(128, 1, 1));
        }

        const a_ring_off = @as(usize, sl.slot) * conv_ring * h * 4;
        if (!push) {
            enc.setPipeline(ep.conv_add);
            enc.setBuffer(s.projout, 0, 0);
            enc.setBuffer(rt.attn_conv, a_ring_off, 1);
            rt.setW(enc, lay.attn_post.off, 2);
            enc.setValue(ConvAddA{ .channels = h, .heads = 0, .ring = conv_ring, .base = base, .rows = n, .layout = 0, .norm = norm_post, .out = 0 }, 3);
            enc.setBuffer(s.x, 0, 4);
            enc.setBuffer(s.x, 0, 5);
            enc.dispatchThreads(mtl.Size.of(h, n, 1), mtl.Size.of(256, 1, 1));
        } else {
            enc.setPipeline(ep.conv);
            enc.setBuffer(s.projout, 0, 0);
            enc.setBuffer(rt.attn_conv, a_ring_off, 1);
            rt.setW(enc, lay.attn_post.off, 2);
            enc.setValue(ConvBA{ .channels = h, .heads = 0, .ring = conv_ring, .base = base, .rows = n, .layout = 0, .norm = norm_post }, 3);
            enc.setValue(ConvBA{ .channels = h, .heads = 0, .ring = conv_ring, .base = base, .rows = n, .layout = 0, .norm = norm_post }, 4);
            enc.setBuffer(s.o, 0, 5);
            enc.dispatchThreads(mtl.Size.of(h, n, 1), mtl.Size.of(256, 1, 1));
        }
        enc.setPipeline(ep.ring_store);
        enc.setBuffer(s.projout, 0, 0);
        enc.setBuffer(rt.attn_conv, a_ring_off, 1);
        enc.setValue(RingA{ .channels = h, .ring = conv_ring, .base = base, .rows = n, .tail = tail }, 2);
        enc.dispatchThreads(mtl.Size.of(h, tail, 1), mtl.Size.of(256, 1, 1));

        if (push) {
            enc.setPipeline(p.scatter);
            enc.setBuffer(s.x, 0, 0);
            enc.setValue(ScatterA{ .width = h, .stride = mix_states, .slot = sl.block_count_before }, 1);
            enc.setBuffer(s.blocks, 0, 2);
            enc.setBuffer(s.o, 0, 3);
            enc.dispatchThreads(mtl.Size.of(h, n, 1), mtl.Size.of(256, 1, 1));
        }

        {
            const mq_off = rt.sched.globals.mlp_queries[sl.phys].off;
            enc.setPipeline(ep.mix_norm);
            enc.setBuffer(s.blocks, 0, 0);
            enc.setBuffer(s.x, 0, 1);
            rt.setW(enc, mq_off, 2);
            rt.setW(enc, mq_off, 3);
            rt.setW(enc, mq_off, 4);
            rt.setW(enc, mq_off, 5);
            enc.setValue(MixArgs{ .width = h, .depth = sl.block_count_after_attn + 1, .stride = mix_states, .k = 1, .norm = @intFromBool(!bank), .scale = scale_h, .eps = g.eps }, 6);
            enc.setBuffer(s.mixout, 0, 7);
            rt.setW(enc, if (!bank) lay.mlp_norm.off else mq_off, 8);
            rt.setW(enc, mq_off, 9);
            rt.setW(enc, mq_off, 10);
            enc.setBuffer(if (!bank) s.m0 else s.n0, 0, 11);
            enc.setBuffer(if (!bank) s.m0 else s.n1, 0, 12);
            enc.setBuffer(if (!bank) s.m0 else s.n2, 0, 13);
            enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        }

        if (bank) {
            const ctl = lay.controllers[3];
            const gf_count: u32 = h * g.gate_rank;
            const gf_off: usize = @as(usize, gf_count) * 4;
            enc.setPipeline(p.rms_pair);
            enc.setBuffer(s.mixout, 0, 0);
            enc.setBuffer(s.msg, 0, 1);
            enc.setValue(RmsPairA{ .width = h, .xs0 = 1, .xb0 = 0, .xs1 = 4, .xb1 = 3, .eps = g.eps }, 2);
            enc.setBuffer(s.h2in, 0, 3);
            enc.dispatchGroups(mtl.Size.of(2 * n, 1, 1), mtl.Size.of(256, 1, 1));

            enc.setPipeline(ep.af_gemm);
            enc.setBuffer(s.h2in, 0, 0);
            rt.setW(enc, ctl.down.off, 1);
            enc.setValue(StageArgs{ .n = g.gate_rank, .k = 2 * h, .rows = n, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, 2);
            enc.setBuffer(s.normed, 0, 3);
            enc.dispatchGroups(mtl.Size.of((g.gate_rank + 31) / 32, (n + 15) / 16, 1), mtl.Size.of(128, 1, 1));

            enc.setPipeline(ep.unpack_gf2);
            rt.setW(enc, ctl.gain.off, 0);
            enc.setValue(gf_count, 1);
            enc.setBuffer(s.stage_w, 0, 2);
            enc.setBuffer(s.stage_w, gf_off, 3);
            enc.dispatchThreads(mtl.Size.of(gf_count, 1, 1), mtl.Size.of(256, 1, 1));

            enc.setPipeline(ep.af_gate_ctl);
            enc.setBuffer(s.normed, 0, 0);
            enc.setBuffer(s.stage_w, 0, 1);
            enc.setBuffer(s.stage_w, gf_off, 2);
            enc.setValue(GateCtlA{ .h = h, .rows = n, .k = g.gate_rank, .cxs = 1, .cxb = 0, .mxs = 4, .mxb = 3, .ys = 1, .yb = 0 }, 3);
            enc.setBuffer(s.mixout, 0, 4);
            enc.setBuffer(s.msg, 0, 5);
            enc.setBuffer(s.mixout, 0, 6);
            enc.dispatchGroups(mtl.Size.of((h + 31) / 32, (n + 15) / 16, 1), mtl.Size.of(128, 1, 1));

            enc.setPipeline(p.rms);
            enc.setBuffer(s.mixout, 0, 0);
            rt.setW(enc, lay.mlp_norm.off, 1);
            enc.setValue(RmsBA{ .width = h, .rows = n, .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = g.eps }, 2);
            enc.setBuffer(s.m0, 0, 3);
            enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        }

        const E: u32 = g.experts;
        const K: u32 = g.topk;
        const slots_n: u32 = n * K;
        enc.setPipeline(ep.af_gemm);
        enc.setBuffer(s.m0, 0, 0);
        rt.setW(enc, lay.router.off, 1);
        enc.setValue(StageArgs{ .n = E, .k = h, .rows = n, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, 2);
        enc.setBuffer(s.router, 0, 3);
        enc.dispatchGroups(mtl.Size.of((E + 31) / 32, (n + 15) / 16, 1), mtl.Size.of(128, 1, 1));

        enc.setPipeline(p.routepick);
        enc.setBuffer(s.router, 0, 0);
        rt.setW(enc, lay.router_bias.off, 1);
        enc.setValue(PickA{ .experts = E, .topk = K, .rows = n, .route_scale = g.route_scale }, 2);
        enc.setBuffer(s.picks, 0, 3);
        enc.setBuffer(s.wts, 0, 4);
        enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));

        enc.setPipeline(ep.moe_zero);
        enc.setBuffer(s.moe_cnt, 0, 0);
        enc.setValue(E, 1);
        enc.dispatchThreads(mtl.Size.of(E, 1, 1), mtl.Size.of(32, 1, 1));

        enc.setPipeline(ep.moe_count);
        enc.setBuffer(s.picks, 0, 0);
        enc.setBuffer(s.moe_cnt, 0, 1);
        enc.setValue(slots_n, 2);
        enc.dispatchThreads(mtl.Size.of(slots_n, 1, 1), mtl.Size.of(32, 1, 1));

        enc.setPipeline(ep.moe_offsets);
        enc.setBuffer(s.moe_cnt, 0, 0);
        enc.setBuffer(s.moe_off, 0, 1);
        enc.setBuffer(s.moe_cur, 0, 2);
        enc.setValue(E, 3);
        enc.dispatchThreads(mtl.Size.of(128, 1, 1), mtl.Size.of(128, 1, 1));

        enc.setPipeline(ep.moe_sort);
        enc.setBuffer(s.picks, 0, 0);
        enc.setBuffer(s.moe_cur, 0, 1);
        enc.setBuffer(s.moe_sorted, 0, 2);
        enc.setValue(slots_n, 3);
        enc.dispatchThreads(mtl.Size.of(slots_n, 1, 1), mtl.Size.of(32, 1, 1));

        const tiles: usize = (slots_n + 63) / 64 + E;
        const hcount: u32 = n * h / 4;
        enc.setPipeline(ep.half_rows);
        enc.setBuffer(s.m0, 0, 0);
        enc.setBuffer(s.xh, 0, 1);
        enc.setBuffer(s.overflow, 0, 2);
        enc.setValue([2]u32{ hcount, @intFromBool(lay.is_affine4 == 0) }, 3);
        enc.dispatchThreads(mtl.Size.of(hcount, 1, 1), mtl.Size.of(256, 1, 1));

        const up_args = MoegA{ .k = h, .n = g.expert_inter, .slots = slots_n, .topk = K, .experts = E, .gate_cap = g.gate_cap, .up_cap = g.up_cap };
        const down_args = MoegA{ .k = g.expert_inter, .n = h, .slots = slots_n, .topk = K, .experts = E, .gate_cap = 0, .up_cap = 0 };
        if (lay.is_affine4 != 0) {
            enc.setPipeline(ep.affine_units_up);
            enc.setBuffer(s.xh, 0, 0);
            rt.setW(enc, lay.w13.off, 1);
            rt.setW(enc, lay.w13_scale.off, 2);
            rt.setW(enc, lay.w13_bias.off, 3);
            enc.setValue(up_args, 4);
            enc.setBuffer(s.moe_off, 0, 5);
            enc.setBuffer(s.moe_sorted, 0, 6);
            enc.setValue(up_args, 7);
            enc.setBuffer(s.act_h, 0, 8);
            enc.setBuffer(s.overflow, 0, 9);
            enc.dispatchGroups(mtl.Size.of((2 * g.expert_inter) / 64, tiles, 1), mtl.Size.of(128, 1, 1));

            enc.setPipeline(ep.affine_units_down);
            enc.setBuffer(s.act_h, 0, 0);
            rt.setW(enc, lay.w2.off, 1);
            rt.setW(enc, lay.w2_scale.off, 2);
            rt.setW(enc, lay.w2_bias.off, 3);
            enc.setValue(down_args, 4);
            enc.setBuffer(s.moe_off, 0, 5);
            enc.setBuffer(s.moe_sorted, 0, 6);
            enc.setValue(down_args, 7);
            enc.setBuffer(s.down, 0, 8);
            enc.setBuffer(s.overflow, 0, 9);
            enc.dispatchGroups(mtl.Size.of(h / 64, tiles, 1), mtl.Size.of(128, 1, 1));
        } else {
            enc.setPipeline(ep.expert_units_up_packed);
            enc.setBuffer(s.xh, 0, 0);
            rt.setW(enc, lay.w13.off, 1);
            rt.setW(enc, lay.w13_scale.off, 2);
            rt.setW(enc, lay.w13_global.off, 3);
            rt.setW(enc, rt.sched.globals.nvfp4_den.off, 4);
            enc.setBuffer(s.moe_off, 0, 5);
            enc.setBuffer(s.moe_sorted, 0, 6);
            enc.setValue(up_args, 7);
            enc.setBuffer(s.act_h, 0, 8);
            enc.setBuffer(s.overflow, 0, 9);
            enc.dispatchGroups(mtl.Size.of((2 * g.expert_inter) / 64, tiles, 1), mtl.Size.of(128, 1, 1));

            enc.setPipeline(ep.expert_units_down);
            enc.setBuffer(s.act_h, 0, 0);
            rt.setW(enc, lay.w2.off, 1);
            rt.setW(enc, lay.w2_scale.off, 2);
            rt.setW(enc, lay.w2_global.off, 3);
            rt.setW(enc, rt.sched.globals.nvfp4_den.off, 4);
            enc.setBuffer(s.moe_off, 0, 5);
            enc.setBuffer(s.moe_sorted, 0, 6);
            enc.setValue(down_args, 7);
            enc.setBuffer(s.down, 0, 8);
            enc.setBuffer(s.overflow, 0, 9);
            enc.dispatchGroups(mtl.Size.of(h / 64, tiles, 1), mtl.Size.of(128, 1, 1));
        }

        enc.setPipeline(ep.combine);
        enc.setBuffer(s.down, 0, 0);
        enc.setBuffer(s.wts, 0, 1);
        enc.setValue(CombineA{ .width = h, .topk = K, .rows = n }, 2);
        enc.setBuffer(s.moe, 0, 3);
        enc.dispatchThreads(mtl.Size.of(h, n, 1), mtl.Size.of(256, 1, 1));

        enc.setPipeline(ep.conv_add);
        enc.setBuffer(s.moe, 0, 0);
        enc.setBuffer(rt.mlp_conv, a_ring_off, 1);
        rt.setW(enc, lay.mlp_post.off, 2);
        enc.setValue(ConvAddA{ .channels = h, .heads = 0, .ring = conv_ring, .base = base, .rows = n, .layout = 0, .norm = norm_post, .out = 0 }, 3);
        enc.setBuffer(s.x, 0, 4);
        enc.setBuffer(s.x, 0, 5);
        enc.dispatchThreads(mtl.Size.of(h, n, 1), mtl.Size.of(256, 1, 1));

        enc.setPipeline(ep.ring_store);
        enc.setBuffer(s.moe, 0, 0);
        enc.setBuffer(rt.mlp_conv, a_ring_off, 1);
        enc.setValue(RingA{ .channels = h, .ring = conv_ring, .base = base, .rows = n, .tail = tail }, 2);
        enc.dispatchThreads(mtl.Size.of(h, tail, 1), mtl.Size.of(256, 1, 1));

        if (sl.is_prelude_end != 0) {
            enc.setPipeline(p.copy);
            enc.setBuffer(s.x, 0, 0);
            enc.setBuffer(s.prelude, 0, 1);
            enc.setValue(PairA{ .count = n * h }, 2);
            enc.dispatchThreads(mtl.Size.of(@as(usize, n) * h, 1, 1), mtl.Size.of(256, 1, 1));
        }

        if (sl.is_flush_after != 0 and sent_n < sent.len) {
            enc.end();
            cb.commit();
            sent[sent_n] = cb;
            sent_n += 1;
            cb = rt.queue.commandBuffer();
            enc = cb.compute(.serial);
        }
    }

    if (want_last_logits or capture_taps) {
        const oq_off = rt.sched.globals.output_query.off;
        enc.setPipeline(ep.mix_norm);
        enc.setBuffer(s.blocks, 0, 0);
        enc.setBuffer(s.x, 0, 1);
        rt.setW(enc, oq_off, 2);
        rt.setW(enc, oq_off, 3);
        rt.setW(enc, oq_off, 4);
        rt.setW(enc, oq_off, 5);
        enc.setValue(MixArgs{ .width = h, .depth = 5, .stride = mix_states, .k = 1, .norm = 1, .scale = scale_h, .eps = g.eps }, 6);
        enc.setBuffer(s.mixout, 0, 7);
        rt.setW(enc, rt.sched.globals.norm.off, 8);
        rt.setW(enc, oq_off, 9);
        rt.setW(enc, oq_off, 10);
        enc.setBuffer(s.n0, 0, 11);
        enc.setBuffer(s.n0, 0, 12);
        enc.setBuffer(s.n0, 0, 13);
        enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));

        if (capture_taps) encodeTap(ep, enc, fmt.num_slots, s.n0, s.taps, h, n);

        if (want_last_logits) {
            enc.setPipeline(p.aot_dense8_sb.?);
            enc.setBuffer(s.n0, @as(usize, n - 1) * h * 4, 0);
            rt.setW(enc, rt.sched.globals.unembed.w.off, 1);
            rt.setW(enc, rt.sched.globals.unembed.sb.off, 2);
            enc.setValue(StageArgs{ .n = g.unpadded_vocab, .k = h, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = g.logit_cap }, 3);
            enc.setBuffer(rt.logits, 0, 4);
            enc.dispatchGroups(mtl.Size.of((g.unpadded_vocab + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        }
    }

    enc.end();
    cb.commit();
    cb.wait();
    var gpu_ms: f64 = 0;
    for (sent[0..sent_n]) |scb| {
        if (scb.failure()) |msg| {
            std.log.err("aot prefill cb failed: {s}", .{msg});
            return error.GpuFailed;
        }
        gpu_ms += scb.gpuSeconds() * 1000.0;
    }
    if (cb.failure()) |msg| {
        std.log.err("aot prefill final cb failed: {s}", .{msg});
        return error.GpuFailed;
    }
    gpu_ms += cb.gpuSeconds() * 1000.0;
    if (s.overflow.slice(u32, 1)[0] != 0) return error.HalfActivationOverflow;
    rt.len += ids.len;
    return gpu_ms;
}

pub fn convertKvToPerm(rt: *rt_mod.Runtime) !void {
    const ep = &(rt.extra_pipes orelse return error.MissingPrefillPipes);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const g = rt.sched.geom;
    const priv = mtl.ResourceOptions.private;
    const tmp_k = try rt.device.buffer(rt.keys.length(), priv);
    defer tmp_k.deinit();
    const tmp_v = try rt.device.buffer(rt.values.length(), priv);
    defer tmp_v.deinit();

    const cb = rt.queue.commandBuffer();
    const enc = cb.compute(.serial);
    for (rt.sched.slots) |sl| {
        const full = sl.is_full != 0;
        const stride_rows: u32 = @intCast(if (full) rt.capacity else g.window);
        const rows: u32 = @intCast(if (full) rt.len else @min(rt.len, g.window));
        if (rows == 0) continue;
        const off = rt.slot_kv_off[sl.slot];
        const args = KvPermA{ .rows = rows, .stride_rows = stride_rows, .kv = g.kv_heads, .dim = g.head_dim };
        enc.setPipeline(ep.kv_to_perm);
        enc.setBuffer(rt.keys, off, 0);
        enc.setBuffer(tmp_k, off, 1);
        enc.setValue(args, 2);
        enc.dispatchThreads(mtl.Size.of(g.head_dim, g.kv_heads, rows), mtl.Size.of(128, 1, 1));

        enc.setPipeline(ep.kv_to_perm);
        enc.setBuffer(rt.values, off, 0);
        enc.setBuffer(tmp_v, off, 1);
        enc.setValue(args, 2);
        enc.dispatchThreads(mtl.Size.of(g.head_dim, g.kv_heads, rows), mtl.Size.of(128, 1, 1));
    }
    enc.end();
    const bl = cb.blit();
    bl.copy(tmp_k, 0, rt.keys, 0, rt.keys.length());
    bl.copy(tmp_v, 0, rt.values, 0, rt.values.length());
    bl.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
}

pub fn verifyTree(
    rt: *rt_mod.Runtime,
    ids: []const u32,
    parents_in: ?[]const u8,
    commit: bool,
    capture_taps: bool,
) !VerifyStats {
    const R = ids.len;
    if (R == 0 or R > max_verify_rows) return error.InvalidVerifyRows;
    const ep = &(rt.extra_pipes orelse return error.MissingPrefillPipes);
    const vs = try rt.ensureVerifyScratch();
    const g = rt.sched.geom;
    const p = &rt.pipes;
    const n: u32 = @intCast(R);
    const h: u32 = g.hidden;
    const qw: u32 = g.heads * g.head_dim;
    const kvw: u32 = g.kv_heads * g.head_dim;
    const scale_h: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
    const scale_attn: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(g.head_dim)));
    const norm_post: f32 = 1.0 / @sqrt(@as(f32, 5.0));

    var parent: [max_verify_rows]u8 = @splat(none_u8);
    var depth: [max_verify_rows]u8 = @splat(0);
    var path: [max_verify_rows][max_verify_rows]u8 = @splat(@splat(0));
    var pos_arr: [max_verify_rows]u32 = @splat(0);
    var max_depth: usize = 0;
    for (0..R) |r| {
        if (r > 0) {
            const par: u8 = if (parents_in) |pl| pl[r - 1] else @intCast(r - 1);
            if (par >= r) return error.BadDraftParents;
            parent[r] = par;
            depth[r] = depth[par] + 1;
            @memcpy(path[r][0..depth[r]], path[par][0..depth[r]]);
        }
        path[r][depth[r]] = @intCast(r);
        pos_arr[r] = @intCast(rt.len + depth[r]);
        max_depth = @max(max_depth, depth[r]);
    }
    if (rt.len + max_depth + 1 > rt.capacity) return error.ContextFull;

    const t0 = rt_mod.nowNs();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    @memcpy(vs.ids.slice(u32, R), ids);

    var cb = rt.queue.commandBuffer();
    var enc = cb.compute(.serial);
    var cb_first: ?mtl.CommandBuffer = null;

    enc.setPipeline(p.aot_embed_sb.?);
    enc.setBuffer(vs.ids, 0, 0);
    rt.setW(enc, rt.sched.globals.embed.w.off, 1);
    rt.setW(enc, rt.sched.globals.embed.sb.off, 2);
    rt.setW(enc, rt.sched.globals.embed_norm.off, 3);
    enc.setValue(EmbedA{ .width = h, .rows = n, .eps = g.eps }, 4);
    enc.setBuffer(vs.x, 0, 5);
    enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));

    const use_t16 = R > 8;
    const t_dense = if (use_t16) ep.t_dense16_8_1_sb else ep.t_dense_8_2_sb;
    const t_proj = if (use_t16) ep.t_proj16_8_1_sb else ep.t_proj_8_2_sb;
    const t_step: u32 = if (use_t16) 8 else 16;

    for (rt.sched.slots) |sl| {
        const lay = &rt.sched.layers[sl.phys];
        const full = sl.is_full != 0;
        const bank = sl.is_bank != 0;
        const push = sl.push_block != 0;

        if (sl.is_pass_start != 0) {
            enc.setPipeline(p.pass_copy);
            enc.setBuffer(vs.blocks, 0, 0);
            enc.setBuffer(vs.x, 0, 1);
            enc.setBuffer(vs.prelude, 0, 2);
            enc.setBuffer(vs.bank, 0, 3);
            enc.setBuffer(vs.passout, 0, 4);
            enc.setValue(PassCopyA{ .width = h, .stride = mix_states, .start = 2, .count = sl.bank_n }, 5);
            enc.dispatchThreads(mtl.Size.of(h, n, 1), mtl.Size.of(256, 1, 1));
        }

        if (capture_taps) encodeTap(ep, enc, sl.visit, vs.x, vs.taps, h, n);

        var in0 = vs.x;
        var in1 = vs.x;
        var in2 = vs.x;
        var fused_norm = false;

        if (sl.phys > 0) {
            const qref = rt.sched.globals.attn_queries[sl.phys - 1];
            const kq: u32 = if (full) 3 else 1;
            const q0_off = qref.off;
            const q1_off = if (full) q0_off + @as(u64, h) * 4 else q0_off;
            const q2_off = if (full) q0_off + @as(u64, 2 * h) * 4 else q0_off;
            if (bank) {
                const at: u64 = sl.cross_pass_at;
                const cpq = rt.sched.globals.cross_pass_q.off;
                const cpkv = rt.sched.globals.cross_pass_kv.off;
                const args2 = [2]MixArgs{
                    .{ .width = h, .depth = sl.block_count_before + 1, .stride = mix_states, .k = kq, .norm = 0, .scale = scale_h, .eps = g.eps },
                    .{ .width = h, .depth = sl.bank_n + 1, .stride = mix_states, .k = 4, .norm = 0, .scale = scale_h, .eps = g.eps },
                };
                enc.setPipeline(p.batch_mix2);
                enc.setBuffer(vs.blocks, 0, 0);
                enc.setBuffer(vs.x, 0, 1);
                rt.setW(enc, q0_off, 2);
                rt.setW(enc, q1_off, 3);
                rt.setW(enc, q2_off, 4);
                rt.setW(enc, q0_off, 5);
                enc.setBuffer(vs.mixout, 0, 6);
                enc.setBuffer(vs.bank, 0, 7);
                enc.setBuffer(vs.passout, 0, 8);
                rt.setW(enc, cpq + (at * 2) * h * 4, 9);
                rt.setW(enc, cpkv + (at * 2) * h * 4, 10);
                rt.setW(enc, cpkv + (at * 2 + 1) * h * 4, 11);
                rt.setW(enc, cpq + (at * 2 + 1) * h * 4, 12);
                enc.setBuffer(vs.msg, 0, 13);
                enc.setBytes(std.mem.sliceAsBytes(&args2), 14);
                enc.dispatchGroups(mtl.Size.of(n, 2, 1), mtl.Size.of(256, 1, 1));

                const xs3: [3]u32 = .{ kq, kq, kq };
                const xb3: [3]u32 = if (full) .{ 0, 1, 2 } else .{ 0, 0, 0 };
                enc.setPipeline(ep.r_controls_down);
                enc.setBuffer(vs.mixout, 0, 0);
                enc.setBuffer(vs.mixout, 0, 1);
                enc.setBuffer(vs.mixout, 0, 2);
                enc.setBuffer(vs.msg, 0, 3);
                rt.setW(enc, lay.controllers[0].down.off, 4);
                rt.setW(enc, lay.controllers[1].down.off, 5);
                rt.setW(enc, lay.controllers[2].down.off, 6);
                enc.setValue(CtlDownR{ .h = h, .rank = g.gate_rank, .rows = n, .msg0 = 0, .xs = xs3, .xb = xb3 }, 7);
                enc.setBuffer(vs.ctl_hid, 0, 8);
                enc.dispatchGroups(mtl.Size.of((g.gate_rank + 7) / 8, n, 3), mtl.Size.of(256, 1, 1));

                enc.setPipeline(ep.ctl_apply3_gf2_rows);
                enc.setBuffer(vs.ctl_hid, 0, 0);
                rt.setW(enc, lay.controllers[0].gain.off, 1);
                rt.setW(enc, lay.controllers[1].gain.off, 2);
                rt.setW(enc, lay.controllers[2].gain.off, 3);
                enc.setBuffer(vs.mixout, 0, 4);
                enc.setBuffer(vs.mixout, 0, 5);
                enc.setBuffer(vs.mixout, 0, 6);
                enc.setBuffer(vs.msg, 0, 7);
                enc.setBuffer(vs.n0, 0, 8);
                enc.setBuffer(vs.n1, 0, 9);
                enc.setBuffer(vs.n2, 0, 10);
                enc.setValue(CtlApply3R{ .h = h, .rank = g.gate_rank, .rows = n, .xs = xs3, .xb = xb3, .msg0 = 0 }, 11);
                enc.dispatchGroups(mtl.Size.of((h + 7) / 8, n, 3), mtl.Size.of(256, 1, 1));

                enc.setPipeline(p.rms3);
                enc.setBuffer(vs.n0, 0, 0);
                enc.setBuffer(vs.n1, 0, 1);
                enc.setBuffer(vs.n2, 0, 2);
                rt.setW(enc, lay.q_norm.off, 3);
                rt.setW(enc, if (full) lay.k_norm.off else lay.q_norm.off, 4);
                rt.setW(enc, if (full) lay.v_norm.off else lay.q_norm.off, 5);
                enc.setValue(Rms3A{ .width = h, .xs = .{ 1, 1, 1 }, .xb = .{ 0, 0, 0 }, .eps = g.eps }, 6);
                enc.setBuffer(vs.n0, 0, 7);
                enc.setBuffer(vs.n1, 0, 8);
                enc.setBuffer(vs.n2, 0, 9);
                enc.dispatchGroups(mtl.Size.of(3 * n, 1, 1), mtl.Size.of(256, 1, 1));
                in0 = vs.n0;
                in1 = vs.n1;
                in2 = vs.n2;
                fused_norm = true;
            } else {
                enc.setPipeline(p.batch_mix);
                enc.setBuffer(vs.blocks, 0, 0);
                enc.setBuffer(vs.x, 0, 1);
                rt.setW(enc, q0_off, 2);
                rt.setW(enc, q1_off, 3);
                rt.setW(enc, q2_off, 4);
                rt.setW(enc, q0_off, 5);
                enc.setValue(MixArgs{ .width = h, .depth = sl.block_count_before + 1, .stride = mix_states, .k = kq, .norm = 1, .scale = scale_h, .eps = g.eps }, 6);
                enc.setBuffer(vs.mixout, 0, 7);
                rt.setW(enc, lay.q_norm.off, 8);
                rt.setW(enc, if (full) lay.k_norm.off else lay.q_norm.off, 9);
                rt.setW(enc, if (full) lay.v_norm.off else lay.q_norm.off, 10);
                enc.setBuffer(vs.n0, 0, 11);
                enc.setBuffer(vs.n1, 0, 12);
                enc.setBuffer(vs.n2, 0, 13);
                enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
                in0 = vs.n0;
                in1 = if (full) vs.n1 else vs.n0;
                in2 = if (full) vs.n2 else vs.n0;
                fused_norm = true;
            }
        }

        if (!fused_norm) {
            enc.setPipeline(p.rms);
            enc.setBuffer(vs.x, 0, 0);
            rt.setW(enc, lay.q_norm.off, 1);
            enc.setValue(RmsBA{ .width = h, .rows = n, .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = g.eps }, 2);
            enc.setBuffer(vs.n0, 0, 3);
            enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
            in0 = vs.n0;
            in1 = vs.n0;
            in2 = vs.n0;
        }

        if (R == 1) {
            var total_proj: u32 = 0;
            for (lay.proj_ms) |m_| total_proj += m_;
            enc.setPipeline(p.aot_batch_proj8_sb.?);
            enc.setBuffer(in0, 0, 0);
            enc.setBuffer(in1, 0, 1);
            enc.setBuffer(in2, 0, 2);
            rt.setW(enc, lay.proj_q.w.off, 3);
            rt.setW(enc, lay.proj_q.sb.off, 4);
            rt.setW(enc, lay.proj_k.w.off, 5);
            rt.setW(enc, lay.proj_k.sb.off, 6);
            rt.setW(enc, lay.proj_v.w.off, 7);
            rt.setW(enc, lay.proj_v.sb.off, 8);
            enc.setBuffer(vs.q, 0, 9);
            enc.setBuffer(vs.k, 0, 10);
            enc.setBuffer(vs.v, 0, 11);
            enc.setBuffer(vs.r, 0, 12);
            enc.setBuffer(vs.gate, 0, 13);
            enc.setBuffer(vs.sink, 0, 14);
            enc.setBuffer(vs.kdyn, 0, 15);
            enc.setBuffer(vs.vdyn, 0, 16);
            enc.setValue(extern struct { k: u32, full: u32, m: [8]u32 }{ .k = h, .full = sl.is_full, .m = lay.proj_ms }, 17);
            enc.dispatchGroups(mtl.Size.of((total_proj + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        } else {
            var groups: u32 = 0;
            for (lay.proj_ms) |m_| groups += (m_ + t_step - 1) / t_step;
            enc.setPipeline(t_proj);
            enc.setBuffer(in0, 0, 0);
            enc.setBuffer(in1, 0, 1);
            enc.setBuffer(in2, 0, 2);
            rt.setW(enc, lay.proj_q.w.off, 3);
            rt.setW(enc, lay.proj_q.sb.off, 4);
            rt.setW(enc, lay.proj_k.w.off, 5);
            rt.setW(enc, lay.proj_k.sb.off, 6);
            rt.setW(enc, lay.proj_v.w.off, 7);
            rt.setW(enc, lay.proj_v.sb.off, 8);
            enc.setBuffer(vs.q, 0, 9);
            enc.setBuffer(vs.k, 0, 10);
            enc.setBuffer(vs.v, 0, 11);
            enc.setBuffer(vs.r, 0, 12);
            enc.setBuffer(vs.gate, 0, 13);
            enc.setBuffer(vs.sink, 0, 14);
            enc.setBuffer(vs.kdyn, 0, 15);
            enc.setBuffer(vs.vdyn, 0, 16);
            enc.setValue(TProjA{ .k = h, .full = sl.is_full, .rows = n, .m = lay.proj_ms }, 17);
            enc.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(256, 1, 1));
        }

        const kh_off = (@as(usize, sl.slot) * 2 + 0) * max_verify_rows * kvw * 2;
        const vh_off = (@as(usize, sl.slot) * 2 + 1) * max_verify_rows * kvw * 2;
        {
            var qrows: [max_verify_rows]VQkvRow = undefined;
            for (0..R) |r| {
                var back: [4]u8 = @splat(none_u8);
                const d = depth[r];
                for (1..conv_ring) |k| if (k <= d) {
                    back[k - 1] = path[r][d - k];
                };
                qrows[r] = .{
                    .k_ring = rt.k_conv.gpuAddress() + @as(usize, sl.slot) * conv_ring * kvw * 4,
                    .v_ring = rt.v_conv.gpuAddress() + @as(usize, sl.slot) * conv_ring * kvw * 4,
                    .pos = pos_arr[r],
                    .start = @intCast(rt.len),
                    .back = back,
                };
            }
            enc.useResource(rt.k_conv, mtl.ResourceUsage.read);
            enc.useResource(rt.v_conv, mtl.ResourceUsage.read);
            const qkv_args = QkvArgs{
                .kv = g.kv_heads,
                .heads = g.heads,
                .dim = g.head_dim,
                .rel = g.d_rel,
                .extent = lay.extent,
                .layout = sl.is_full,
                .dyn = sl.is_full,
                .tau = sl.is_full,
                .norm = 1.0 / @sqrt(@as(f32, if (full) 9.0 else 5.0)),
                .q_threshold = g.qk_clip,
                .rel_threshold = g.rel_clip,
                .eps = g.eps,
                .floor = @floatFromInt(g.log_floor),
                .alpha = g.log_alpha,
            };
            enc.setPipeline(ep.v_qkv);
            enc.setBuffer(vs.k, 0, 0);
            enc.setBuffer(vs.v, 0, 1);
            rt.setW(enc, lay.k_conv.off, 2);
            rt.setW(enc, lay.v_conv.off, 3);
            enc.setBuffer(vs.kdyn, 0, 4);
            enc.setBuffer(vs.vdyn, 0, 5);
            enc.setBuffer(vs.q, 0, 6);
            rt.setW(enc, lay.q_scale.off, 7);
            enc.setBuffer(vs.r, 0, 8);
            rt.setW(enc, lay.r_scale.off, 9);
            rt.setW(enc, lay.rel_bias.off, 10);
            enc.setValue(qkv_args, 11);
            enc.setBuffer(vs.relbias, 0, 12);
            enc.setBytes(std.mem.sliceAsBytes(qrows[0..R]), 13);
            enc.setBuffer(vs.kc, 0, 14);
            enc.setBuffer(vs.vc, 0, 15);
            enc.setBuffer(vs.v_conv, VerifyScratch.convOffset(g, sl.slot, 0), 16);
            enc.setBuffer(vs.v_conv, VerifyScratch.convOffset(g, sl.slot, 1), 17);
            enc.setBuffer(vs.v_kv, kh_off, 18);
            enc.setBuffer(vs.v_kv, vh_off, 19);
            enc.dispatchGroups(mtl.Size.of(g.kv_heads + g.heads + (g.heads * (lay.extent + 1) + 255) / 256, n, 1), mtl.Size.of(256, 1, 1));
        }

        {
            const win: u32 = if (full) 0 else g.window;
            var grows: [max_verify_rows]VGroupRow = undefined;
            var prows: [max_verify_rows]BatchRow = undefined;
            var max_splits: u32 = 0;
            const kbase = rt.keys.gpuAddress() + rt.slot_kv_off[sl.slot];
            const vbase = rt.values.gpuAddress() + rt.slot_kv_off[sl.slot];
            for (0..R) |r| {
                const pos = pos_arr[r];
                const lo: u32 = if (win != 0 and pos + 1 > win) pos + 1 - win else 0;
                const part = attnPartition(pos + 1 - lo);
                const dp = DecodeParts{ .heads = g.heads, .kv = g.kv_heads, .dim = g.head_dim, .rel = g.d_rel, .extent = lay.extent, .window = win, .pos = pos, .tau = sl.is_full, .splits = part.splits, .chunk = part.chunk, .scale = scale_attn, .floor = @floatFromInt(g.log_floor), .alpha = g.log_alpha };
                grows[r] = .{ .keys = kbase, .values = vbase, .args = dp, .gpos = @intCast(rt.len), .path = path[r] };
                prows[r] = .{ .keys = kbase, .values = vbase, .args = dp };
                max_splits = @max(max_splits, part.splits);
            }
            enc.useResource(rt.keys, mtl.ResourceUsage.read);
            enc.useResource(rt.values, mtl.ResourceUsage.read);
            enc.setPipeline(ep.v_split);
            enc.setBuffer(vs.q, 0, 0);
            enc.setBuffer(vs.relbias, 0, 1);
            enc.setBuffer(vs.sink, 0, 2);
            enc.setBytes(std.mem.sliceAsBytes(grows[0..R]), 3);
            enc.setBuffer(vs.parts, 0, 4);
            enc.setBuffer(vs.v_kv, kh_off, 5);
            enc.setBuffer(vs.v_kv, vh_off, 6);
            enc.dispatchGroups(mtl.Size.of(g.kv_heads * max_splits, n, 1), mtl.Size.of((g.heads / g.kv_heads) * 32, 1, 1));

            enc.setPipeline(p.aot_combine_sig_mul.?);
            enc.setBuffer(vs.parts, 0, 0);
            enc.setBytes(std.mem.sliceAsBytes(prows[0..R]), 1);
            enc.setBuffer(vs.gate, 0, 2);
            enc.setBuffer(vs.attn, 0, 3);
            enc.dispatchGroups(mtl.Size.of(g.heads, n, 1), mtl.Size.of(32, 1, 1));
        }

        if (R == 1) {
            enc.setPipeline(p.aot_dense8_sb.?);
            enc.setBuffer(vs.attn, 0, 0);
            rt.setW(enc, lay.proj_o.w.off, 1);
            rt.setW(enc, lay.proj_o.sb.off, 2);
            enc.setValue(StageArgs{ .n = h, .k = qw, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = 0 }, 3);
            enc.setBuffer(vs.projout, 0, 4);
            enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        } else {
            enc.setPipeline(t_dense);
            enc.setBuffer(vs.attn, 0, 0);
            rt.setW(enc, lay.proj_o.w.off, 1);
            rt.setW(enc, lay.proj_o.sb.off, 2);
            enc.setValue(StageArgs{ .n = h, .k = qw, .rows = n, .xs = 1, .xb = 0, .bits = 8, .cap = 0 }, 3);
            enc.setBuffer(vs.projout, 0, 4);
            enc.dispatchGroups(mtl.Size.of((h + t_step - 1) / t_step, 1, 1), mtl.Size.of(256, 1, 1));
        }

        {
            var crows: [max_verify_rows]VConvRow = undefined;
            for (0..R) |r| {
                var back: [4]u8 = @splat(none_u8);
                const d = depth[r];
                for (1..conv_ring) |k| if (k <= d) {
                    back[k - 1] = path[r][d - k];
                };
                crows[r] = .{ .ring = rt.attn_conv.gpuAddress() + @as(usize, sl.slot) * conv_ring * h * 4, .pos = pos_arr[r], .start = @intCast(rt.len), .back = back };
            }
            enc.useResource(rt.attn_conv, mtl.ResourceUsage.read);
            enc.setPipeline(ep.v_conv);
            enc.setBuffer(vs.projout, 0, 0);
            rt.setW(enc, lay.attn_post.off, 1);
            enc.setBuffer(vs.projout, 0, 2);
            enc.setValue(VConvArgs{ .channels = h, .heads = 0, .ring = conv_ring, .pos = 0, .layout = 0, .norm = norm_post, .residual = @intFromBool(!push), .topk = 0 }, 3);
            enc.setBuffer(if (!push) vs.x else vs.o, 0, 4);
            enc.setBytes(std.mem.sliceAsBytes(crows[0..R]), 5);
            enc.setBuffer(vs.v_conv, VerifyScratch.convOffset(g, sl.slot, 2), 6);
            enc.setBuffer(vs.projout, 0, 7);
            enc.dispatchThreads(mtl.Size.of(h, n, 1), mtl.Size.of(256, 1, 1));
        }

        if (push) {
            enc.setPipeline(p.scatter);
            enc.setBuffer(vs.x, 0, 0);
            enc.setValue(ScatterA{ .width = h, .stride = mix_states, .slot = sl.block_count_before }, 1);
            enc.setBuffer(vs.blocks, 0, 2);
            enc.setBuffer(vs.o, 0, 3);
            enc.dispatchThreads(mtl.Size.of(h, n, 1), mtl.Size.of(256, 1, 1));
        }

        {
            const mq_off = rt.sched.globals.mlp_queries[sl.phys].off;
            enc.setPipeline(p.batch_mix);
            enc.setBuffer(vs.blocks, 0, 0);
            enc.setBuffer(vs.x, 0, 1);
            rt.setW(enc, mq_off, 2);
            rt.setW(enc, mq_off, 3);
            rt.setW(enc, mq_off, 4);
            rt.setW(enc, mq_off, 5);
            enc.setValue(MixArgs{ .width = h, .depth = sl.block_count_after_attn + 1, .stride = mix_states, .k = 1, .norm = @intFromBool(!bank), .scale = scale_h, .eps = g.eps }, 6);
            enc.setBuffer(vs.mixout, 0, 7);
            rt.setW(enc, if (!bank) lay.mlp_norm.off else mq_off, 8);
            rt.setW(enc, mq_off, 9);
            rt.setW(enc, mq_off, 10);
            enc.setBuffer(if (!bank) vs.m0 else vs.n0, 0, 11);
            enc.setBuffer(if (!bank) vs.m0 else vs.n1, 0, 12);
            enc.setBuffer(if (!bank) vs.m0 else vs.n2, 0, 13);
            enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        }

        if (bank) {
            const ctl = lay.controllers[3];
            enc.setPipeline(ep.r_controls_down);
            enc.setBuffer(vs.mixout, 0, 0);
            enc.setBuffer(vs.mixout, 0, 1);
            enc.setBuffer(vs.mixout, 0, 2);
            enc.setBuffer(vs.msg, 0, 3);
            rt.setW(enc, ctl.down.off, 4);
            rt.setW(enc, ctl.down.off, 5);
            rt.setW(enc, ctl.down.off, 6);
            enc.setValue(CtlDownR{ .h = h, .rank = g.gate_rank, .rows = n, .msg0 = 3, .xs = .{ 1, 1, 1 }, .xb = .{ 0, 0, 0 } }, 7);
            enc.setBuffer(vs.ctl_hid, 0, 8);
            enc.dispatchGroups(mtl.Size.of((g.gate_rank + 7) / 8, n, 1), mtl.Size.of(256, 1, 1));

            enc.setPipeline(ep.ctl_apply1_gf2_rows);
            enc.setBuffer(vs.ctl_hid, 0, 0);
            rt.setW(enc, ctl.gain.off, 1);
            enc.setBuffer(vs.mixout, 0, 2);
            enc.setBuffer(vs.msg, 0, 3);
            enc.setBuffer(vs.mixout, 0, 4);
            enc.setValue(CtlApply1R{ .k = g.gate_rank, .m0 = h, .m1 = h, .h = h, .cxs = 1, .cxb = 0, .mxs = 4, .mxb = 3, .ys = 1, .yb = 0 }, 5);
            enc.dispatchGroups(mtl.Size.of((h + 7) / 8, n, 1), mtl.Size.of(256, 1, 1));

            enc.setPipeline(p.rms);
            enc.setBuffer(vs.mixout, 0, 0);
            rt.setW(enc, lay.mlp_norm.off, 1);
            enc.setValue(RmsBA{ .width = h, .rows = n, .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = g.eps }, 2);
            enc.setBuffer(vs.m0, 0, 3);
            enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        }

        const slots_n: u32 = n * g.topk;
        enc.setPipeline(p.float_mv);
        enc.setBuffer(vs.m0, 0, 0);
        rt.setW(enc, lay.router.off, 1);
        enc.setValue(StageArgs{ .n = g.experts, .k = h, .rows = n, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, 2);
        enc.setBuffer(vs.router, 0, 3);
        enc.dispatchGroups(mtl.Size.of((g.experts + 7) / 8, n, 1), mtl.Size.of(256, 1, 1));

        enc.setPipeline(p.routepick);
        enc.setBuffer(vs.router, 0, 0);
        rt.setW(enc, lay.router_bias.off, 1);
        enc.setValue(PickA{ .experts = g.experts, .topk = g.topk, .rows = n, .route_scale = g.route_scale }, 2);
        enc.setBuffer(vs.picks, 0, 3);
        enc.setBuffer(vs.wts, 0, 4);
        enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));

        if (lay.is_affine4 != 0) {
            enc.setPipeline(p.aff_up);
            enc.setBuffer(vs.m0, 0, 0);
            rt.setW(enc, lay.w13.off, 1);
            rt.setW(enc, lay.w13_scale.off, 2);
            rt.setW(enc, lay.w13_bias.off, 3);
            enc.setBuffer(vs.picks, 0, 4);
            enc.setValue(TMoeA{ .k = h, .n = g.expert_inter, .slots = slots_n, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }, 5);
            enc.setBuffer(vs.act, 0, 6);
            enc.dispatchGroups(mtl.Size.of((g.expert_inter + 7) / 8, slots_n, 1), mtl.Size.of(256, 1, 1));

            enc.setPipeline(p.aff_down);
            enc.setBuffer(vs.act, 0, 0);
            rt.setW(enc, lay.w2.off, 1);
            rt.setW(enc, lay.w2_scale.off, 2);
            rt.setW(enc, lay.w2_bias.off, 3);
            enc.setBuffer(vs.picks, 0, 4);
            enc.setValue(TMoeA{ .k = g.expert_inter, .n = h, .slots = slots_n, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }, 5);
            enc.setBuffer(vs.down, 0, 6);
            enc.dispatchGroups(mtl.Size.of((h + 7) / 8, slots_n, 1), mtl.Size.of(256, 1, 1));
        } else if (R == 1) {
            enc.setPipeline(p.aot_gu_nv_packed.?);
            enc.setBuffer(vs.m0, 0, 0);
            rt.setW(enc, lay.w13.off, 1);
            rt.setW(enc, lay.w13_scale.off, 2);
            rt.setW(enc, lay.w13_global.off, 3);
            enc.setBuffer(vs.picks, 0, 4);
            enc.setValue(TMoeA{ .k = h, .n = g.expert_inter, .slots = slots_n, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }, 5);
            enc.setBuffer(vs.act, 0, 6);
            rt.setW(enc, rt.sched.globals.t16.off, 7);
            rt.setW(enc, rt.sched.globals.t256.off, 8);
            enc.dispatchGroups(mtl.Size.of((g.expert_inter + 7) / 8, slots_n, 1), mtl.Size.of(256, 1, 1));

            enc.setPipeline(p.aot_down1_nv.?);
            enc.setBuffer(vs.act, 0, 0);
            rt.setW(enc, lay.w2.off, 1);
            rt.setW(enc, lay.w2_scale.off, 2);
            rt.setW(enc, lay.w2_global.off, 3);
            enc.setBuffer(vs.picks, 0, 4);
            enc.setValue(TMoeA{ .k = g.expert_inter, .n = h, .slots = slots_n, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }, 5);
            enc.setBuffer(vs.down, 0, 6);
            rt.setW(enc, rt.sched.globals.t16.off, 7);
            rt.setW(enc, rt.sched.globals.t256.off, 8);
            enc.dispatchGroups(mtl.Size.of((h + 31) / 32, slots_n, 1), mtl.Size.of(256, 1, 1));
        } else {
            enc.setPipeline(if (use_t16) ep.e_gu16_packed else ep.e_gu_packed);
            enc.setBuffer(vs.m0, 0, 0);
            rt.setW(enc, lay.w13.off, 1);
            rt.setW(enc, lay.w13_scale.off, 2);
            rt.setW(enc, lay.w13_global.off, 3);
            enc.setBuffer(vs.picks, 0, 4);
            enc.setValue(TMoeA{ .k = h, .n = g.expert_inter, .slots = slots_n, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }, 5);
            enc.setBuffer(vs.act, 0, 6);
            rt.setW(enc, rt.sched.globals.t16.off, 7);
            rt.setW(enc, rt.sched.globals.t256.off, 8);
            enc.dispatchGroups(mtl.Size.of((2 * g.expert_inter + 15) / 16, slots_n, 1), mtl.Size.of(128, 1, 1));

            enc.setPipeline(if (use_t16) ep.e_down16 else ep.e_down);
            enc.setBuffer(vs.act, 0, 0);
            rt.setW(enc, lay.w2.off, 1);
            rt.setW(enc, lay.w2_scale.off, 2);
            rt.setW(enc, lay.w2_global.off, 3);
            rt.setW(enc, rt.sched.globals.nvfp4_den.off, 4);
            enc.setBuffer(vs.picks, 0, 5);
            enc.setValue(TMoeA{ .k = g.expert_inter, .n = h, .slots = slots_n, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }, 6);
            enc.setBuffer(vs.down, 0, 7);
            rt.setW(enc, rt.sched.globals.t16.off, 8);
            enc.dispatchGroups(mtl.Size.of((h + 31) / 32, slots_n, 1), mtl.Size.of(256, 1, 1));
        }

        {
            var mrows: [max_verify_rows]VConvRow = undefined;
            for (0..R) |r| {
                var back: [4]u8 = @splat(none_u8);
                const d = depth[r];
                for (1..conv_ring) |k| if (k <= d) {
                    back[k - 1] = path[r][d - k];
                };
                mrows[r] = .{ .ring = rt.mlp_conv.gpuAddress() + @as(usize, sl.slot) * conv_ring * h * 4, .pos = pos_arr[r], .start = @intCast(rt.len), .back = back };
            }
            enc.useResource(rt.mlp_conv, mtl.ResourceUsage.read);
            enc.setPipeline(ep.v_conv);
            enc.setBuffer(vs.down, 0, 0);
            rt.setW(enc, lay.mlp_post.off, 1);
            enc.setBuffer(vs.down, 0, 2);
            enc.setValue(VConvArgs{ .channels = h, .heads = 0, .ring = conv_ring, .pos = 0, .layout = 0, .norm = norm_post, .residual = 1, .topk = g.topk }, 3);
            enc.setBuffer(vs.x, 0, 4);
            enc.setBytes(std.mem.sliceAsBytes(mrows[0..R]), 5);
            enc.setBuffer(vs.v_conv, VerifyScratch.convOffset(g, sl.slot, 3), 6);
            enc.setBuffer(vs.wts, 0, 7);
            enc.dispatchThreads(mtl.Size.of(h, n, 1), mtl.Size.of(256, 1, 1));
        }

        if (sl.is_prelude_end != 0) {
            enc.setPipeline(p.copy);
            enc.setBuffer(vs.x, 0, 0);
            enc.setBuffer(vs.prelude, 0, 1);
            enc.setValue(PairA{ .count = n * h }, 2);
            enc.dispatchThreads(mtl.Size.of(@as(usize, n) * h, 1, 1), mtl.Size.of(256, 1, 1));

            enc.end();
            cb.commit();
            cb_first = cb;
            cb = rt.queue.commandBuffer();
            enc = cb.compute(.serial);
        }
    }

    {
        const oq_off = rt.sched.globals.output_query.off;
        enc.setPipeline(p.batch_mix);
        enc.setBuffer(vs.blocks, 0, 0);
        enc.setBuffer(vs.x, 0, 1);
        rt.setW(enc, oq_off, 2);
        rt.setW(enc, oq_off, 3);
        rt.setW(enc, oq_off, 4);
        rt.setW(enc, oq_off, 5);
        enc.setValue(MixArgs{ .width = h, .depth = 5, .stride = mix_states, .k = 1, .norm = 1, .scale = scale_h, .eps = g.eps }, 6);
        enc.setBuffer(vs.mixout, 0, 7);
        rt.setW(enc, rt.sched.globals.norm.off, 8);
        rt.setW(enc, oq_off, 9);
        rt.setW(enc, oq_off, 10);
        enc.setBuffer(vs.n0, 0, 11);
        enc.setBuffer(vs.n0, 0, 12);
        enc.setBuffer(vs.n0, 0, 13);
        enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));

        if (capture_taps) encodeTap(ep, enc, fmt.num_slots, vs.n0, vs.taps, h, n);

        if (R == 1) {
            enc.setPipeline(p.aot_dense8_sb.?);
            enc.setBuffer(vs.n0, 0, 0);
            rt.setW(enc, rt.sched.globals.unembed.w.off, 1);
            rt.setW(enc, rt.sched.globals.unembed.sb.off, 2);
            enc.setValue(StageArgs{ .n = g.unpadded_vocab, .k = h, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = g.logit_cap }, 3);
            enc.setBuffer(vs.logits, 0, 4);
            enc.dispatchGroups(mtl.Size.of((g.unpadded_vocab + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        } else {
            enc.setPipeline(t_dense);
            enc.setBuffer(vs.n0, 0, 0);
            rt.setW(enc, rt.sched.globals.unembed.w.off, 1);
            rt.setW(enc, rt.sched.globals.unembed.sb.off, 2);
            enc.setValue(StageArgs{ .n = g.unpadded_vocab, .k = h, .rows = n, .xs = 1, .xb = 0, .bits = 8, .cap = g.logit_cap }, 3);
            enc.setBuffer(vs.logits, 0, 4);
            enc.dispatchGroups(mtl.Size.of((g.unpadded_vocab + t_step - 1) / t_step, 1, 1), mtl.Size.of(256, 1, 1));
        }
    }

    if (commit) {
        enc.setPipeline(p.argmax);
        enc.setBuffer(vs.logits, 0, 0);
        enc.setValue(g.unpadded_vocab, 1);
        enc.setBuffer(vs.targets, 0, 2);
        enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(1024, 1, 1));

        const streams = [_]VStream{.{ .first = 0, .size = n }};
        var par32: [max_verify_rows]u32 = undefined;
        for (0..R) |r| par32[r] = if (parent[r] == none_u8) std.math.maxInt(u32) else parent[r];
        enc.setPipeline(ep.v_accept);
        enc.setBuffer(vs.ids, 0, 0);
        enc.setBuffer(vs.targets, 0, 1);
        enc.setBytes(std.mem.sliceAsBytes(&streams), 2);
        enc.setValue(@as(u32, 1), 3);
        enc.setBuffer(vs.accepted, 0, 4);
        enc.setBytes(std.mem.sliceAsBytes(par32[0..R]), 5);
        enc.setBuffer(vs.keep, 0, 6);
        enc.setBuffer(vs.kept, 0, 7);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));

        var rrows: [max_verify_rows]VRoundRow = undefined;
        for (0..R) |r| {
            rrows[r] = .{
                .keys = rt.keys.gpuAddress(),
                .values = rt.values.gpuAddress(),
                .conv = .{ rt.k_conv.gpuAddress(), rt.v_conv.gpuAddress(), rt.attn_conv.gpuAddress(), rt.mlp_conv.gpuAddress() },
                .pos = pos_arr[r],
                .capacity = @intCast(rt.capacity),
                .owner = 0,
                .depth = depth[r],
            };
        }
        var cslots: [fmt.num_slots]VCommitSlot = undefined;
        var locs: u32 = 0;
        var globs: u32 = 0;
        for (0..fmt.num_slots) |sidx| {
            const is_full = rt.slotIsFull(@intCast(sidx));
            cslots[sidx] = .{ .local = @intFromBool(!is_full), .locals_before = locs, .globals_before = globs };
            if (is_full) globs += 1 else locs += 1;
        }
        const cargs = VCommitArgs{ .kvw = kvw, .hidden = h, .rows = n, .stride = max_verify_rows, .slots = fmt.num_slots, .window = g.window, .ring = conv_ring };
        for ([_]mtl.Buffer{ rt.keys, rt.values, rt.k_conv, rt.v_conv, rt.attn_conv, rt.mlp_conv }) |buf| enc.useResource(buf, mtl.ResourceUsage.write);
        enc.setPipeline(ep.v_round_kv);
        enc.setBuffer(vs.v_kv, 0, 0);
        enc.setValue(cargs, 1);
        enc.setBytes(std.mem.sliceAsBytes(rrows[0..R]), 2);
        enc.setBytes(std.mem.sliceAsBytes(&cslots), 3);
        enc.setBuffer(vs.keep, 0, 4);
        enc.dispatchThreads(mtl.Size.of(kvw, n, fmt.num_slots * 2), mtl.Size.of(256, 1, 1));

        enc.setPipeline(ep.v_round_conv);
        enc.setBuffer(vs.v_conv, 0, 0);
        enc.setValue(cargs, 1);
        enc.setBytes(std.mem.sliceAsBytes(rrows[0..R]), 2);
        enc.setBuffer(vs.keep, 0, 3);
        enc.setBuffer(vs.kept, 0, 4);
        enc.dispatchThreads(mtl.Size.of(@max(kvw, h), n, fmt.num_slots * 4), mtl.Size.of(256, 1, 1));
    }

    enc.end();
    cb.commit();
    cb.wait();
    if (cb_first) |c0| {
        if (c0.failure()) |msg| {
            std.log.err("aot verify cb0 failed: {s}", .{msg});
            return error.GpuFailed;
        }
    }
    if (cb.failure()) |msg| {
        std.log.err("aot verify cb1 failed: {s}", .{msg});
        return error.GpuFailed;
    }
    const gpu_ms = (if (cb_first) |c0| c0.gpuSeconds() else 0) * 1000.0 + cb.gpuSeconds() * 1000.0;
    const wall_ms = @as(f64, @floatFromInt(rt_mod.nowNs() - t0)) / 1e6;
    var acc_last: u32 = 0;
    var kept_n: usize = 0;
    if (commit) {
        acc_last = vs.accepted.slice(u32, 1)[0];
        kept_n = @as(usize, depth[acc_last]) + 1;
        rt.len += kept_n;
    }
    return .{ .wall_ms = wall_ms, .gpu_ms = gpu_ms, .rows = R, .accepted_last = acc_last, .kept_tokens = kept_n };
}
