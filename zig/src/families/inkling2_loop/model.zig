//! Inkling 2 Mini's checkpoint as Metal buffers, and the pipelines its forward runs.
//!
//! Expert weights stay in the checkpoint's own layout (NVFP4 nibble pairs or MXFP8 codes with their block scales) and
//! the kernels decode them, so no float32 materialisation is ever resident. The dense weights are bf16 or f32 as
//! stored, read in place from the file's mapping.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const st = core.safetensors;
const cfg = @import("config.zig");
const kernels = @import("kernels.zig");
const name_map = @import("names.zig");
const quant = @import("quant.zig");

pub const Affine = struct {
    scales: mtl.Buffer,
    scale_offset: usize,
    biases: mtl.Buffer,
    bias_offset: usize,
    kind: quant.Kind,

    pub fn bits(self: Affine) u32 {
        return switch (self.kind) {
            .affine4 => 4,
            .affine8 => 8,
            else => unreachable,
        };
    }
};

pub const Tensor = struct {
    buffer: mtl.Buffer,
    offset: usize,
    bytes: usize,
    dtype: st.DType,
    shape: [st.max_rank]usize = @splat(1),
    rank: usize = 0,
    affine: ?Affine = null,

    pub fn count(self: Tensor) usize {
        return self.bytes / self.dtype.size();
    }
};

const Shard = struct { map: ?mtl.MappedFile, buffer: mtl.Buffer, data: usize, header: st.Header };

pub const Attn = struct {
    q: Tensor, // wq_du [q_width, hidden]
    k: Tensor, // wk_dv [kv_width, hidden]
    v: Tensor,
    o: Tensor, // wo_ud [hidden, q_width]
    r: Tensor, // wr_du [heads * d_rel, hidden]
    gate: Tensor, // gate_attn [q_width, hidden]
    sink: Tensor, // attn_sink_proj [heads, hidden]
    q_scale: Tensor, // q_l2_clip.scale [heads, head_dim]
    r_scale: Tensor, // rel_l2_clip.scale [heads, d_rel]
    rel_bias: Tensor, // rel_logits_proj.proj [d_rel, extent]
    k_conv: Tensor, // [taps][kv_width], the kernel's own tap order
    v_conv: Tensor,
    k_dyn: ?Tensor = null, // full layers' per-token taps [kv_width * taps, hidden]
    v_dyn: ?Tensor = null,
};

pub const Controller = struct { down: Tensor, gain: Tensor, forget: Tensor };

pub const Layer = struct {
    q_norm: Tensor, // the SWA attn_norm, or the full layers' q_input_norm
    k_norm: Tensor,
    v_norm: Tensor,
    attn: Attn,
    attn_post: Tensor, // attn_sconv [taps][hidden]
    mlp_norm: Tensor,
    mlp_post: Tensor, // mlp_sconv [taps][hidden]
    router: Tensor, // mlp.gate.weight [experts, hidden] bf16
    router_bias: Tensor, // mlp.gate.bias [experts] f32
    w13: Tensor, // packed codes [experts, 2 * inter, ...]
    w13_scale: Tensor,
    w13_global: ?Tensor = null, // the per-expert amax and the tensor's denominator, NVFP4 only
    w13_den: ?Tensor = null,
    w2: Tensor,
    w2_scale: Tensor,
    w2_global: ?Tensor = null,
    w2_den: ?Tensor = null,
    w13_bias: ?Tensor = null,
    w2_bias: ?Tensor = null,
    precision: quant.Kind,
    controllers: [4]?Controller = @splat(null), // attn, k, v, mlp on the core layers
};

pub const Weights = struct {
    embed: Tensor,
    embed_norm: Tensor,
    norm: Tensor,
    unembed: Tensor,
    output_query: Tensor,
    attn_queries: []Tensor, // layer - 1 -> [3, hidden] on full layers, [hidden] elsewhere
    mlp_queries: []Tensor, // layer -> [hidden]
    cross_pass_q: Tensor, // [2 * core_layers, hidden]
    cross_pass_kv: Tensor,
    layers: []Layer,
};

/// One block-drafter layer of `mtp.safetensors`, every tensor bf16 as stored.
pub const DraftLayer = struct {
    input_norm: Tensor, // input_layernorm [hidden]
    post_norm: Tensor, // post_attention_layernorm [hidden]
    mix_norm: Tensor, // dflare_norm [hidden], the norm of this layer's tap mixture
    q: Tensor, // q_proj [q_heads * dim, hidden]
    k: Tensor, // k_proj [kv_heads * dim, hidden]
    v: Tensor,
    k_ctx: Tensor, // k_proj_ctx [kv_heads * dim, hidden], the context rows' own projections
    v_ctx: Tensor,
    o: Tensor, // o_proj [hidden, q_heads * dim]
    q_norm: Tensor, // [dim]
    k_norm: Tensor,
    k_norm_ctx: Tensor,
    gate: Tensor, // mlp.gate_proj [inter, hidden]
    up: Tensor,
    down: Tensor, // mlp.down_proj [hidden, inter]
};

/// The checkpoint's MTP head (`mtp.safetensors`): a block drafter whose context is the target's tapped hidden states,
/// and its recurrent Markov head. `fold` and `fold_bias` are the target's unembedding times the head's latent
/// projection, built once on the GPU by the drafter.
pub const DraftWeights = struct {
    buffer: mtl.Buffer,
    layers: []DraftLayer,
    norm: Tensor, // [hidden]
    mask: Tensor, // mask_embed [hidden], the block's masked rows
    fuse: Tensor, // fuse_weight [layers + 1, taps]: each layer's tap mixture, then the Markov head's
    markov_w1: Tensor, // [vocab, rank]
    rnn: Tensor, // rnn_proj.weight [3 * rank, 2 * rank + 2 * hidden]
    rnn_bias: Tensor, // [3 * rank]
    hidden_out: Tensor, // [hidden, rank]
    hidden_out_bias: Tensor, // [hidden]
    markov_norm: Tensor, // markov_head.dflare_norm [hidden]
    fold: ?Tensor = null, // [unpadded vocab, rank] bf16
    fold_bias: ?Tensor = null, // [unpadded vocab] bf16
};

pub const glue_names = [_][:0]const u8{
    "i2_mv",         "i2_mv_cap",  "i2_embed",    "i2_rms",       "i2_mix",  "i2_mix4", "i2_clip_scaled",
    "i2_clip",       "i2_conv",    "i2_conv_store", "i2_kv_store", "i2_attn", "i2_relbias", "i2_sig_mul",
    "i2_add",        "i2_sub",     "i2_mul",      "i2_ctl",  "i2_copy", "i2_bcast",
    "i2_zero",       "i2_route",   "i2_to_bf16",  "i2_combine",   "i2_plan", "i2_rms_gather", "i2_rms_weighted3", "i2_rms_kv_store", "i2_clip3",
};

pub const Glue = enum {
    mv,
    mv_cap,
    embed,
    rms,
    mix,
    mix4,
    clip_scaled,
    clip,
    conv,
    conv_store,
    kv_store,
    attn,
    relbias,
    sig_mul,
    add,
    sub,
    mul,
    ctl,
    copy,
    bcast,
    zero,
    route,
    to_bf16,
    combine,
    plan,
    rms_gather,
    rms_weighted3,
    rms_kv_store,
    clip3,
};

pub const expert_names = [_][:0]const u8{ "i2_gu", "i2_down" };
pub const Experts = enum { gu, down };

fn expertKernels(mx: bool) [2][:0]const u8 {
    if (!mx) return .{ "i2_gu_table", "i2_down_table" };
    return .{ "i2_gu_table", "i2_down_thread" };
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    arena: std.heap.ArenaAllocator,
    shards: []Shard,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    weights: Weights,
    glue: [glue_names.len]mtl.Pipeline,
    experts: [2][2]mtl.Pipeline, // [nvfp4, mxfp8][gate/up, down]
    affine_pipes: ?kernels.AffinePipes = null,
    draft: ?DraftWeights = null,
    /// The checkpoint directory, for weights loaded on request.
    dir: []const u8 = "",
    t16: mtl.Buffer,
    t256: mtl.Buffer,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.owned = .empty;
        m.affine_pipes = null;
        m.draft = null;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        m.config = try readConfig(arena.allocator(), io, dir);
        m.arena = arena;
        m.dir = try m.arena.allocator().dupe(u8, dir);
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        errdefer for (m.owned.items) |buf| buf.deinit();
        const names = try cfg.shards(m.arena.allocator(), io, dir);
        m.shards = try m.arena.allocator().alloc(Shard, names.len);
        var mapped: usize = 0;
        errdefer for (m.shards[0..mapped]) |s| {
            s.buffer.deinit();
            if (s.map) |map| map.deinit();
        };
        for (names, 0..) |name, i| {
            m.shards[i] = try mapShard(m, dir, name);
            mapped += 1;
        }
        if (m.config.checkpoint == .modelopt) {
            if (m.shards.len != 1) return error.MultipleShards;
            const formats = try m.arena.allocator().alloc(cfg.Expert, m.config.text.layers);
            try cfg.expertFormats(m.shards[0].header, m.config.text.layers, formats);
            m.weights = try m.mapWeights(formats);
        } else m.weights = try m.mapMlxWeights();
        m.t16 = try m.device.buffer(16 * 4, mtl.ResourceOptions.shared);
        errdefer m.t16.deinit();
        m.t256 = try m.device.buffer(256 * 4, mtl.ResourceOptions.shared);
        errdefer m.t256.deinit();
        for (0..16) |i| m.t16.slice(f32, 16)[i] = @floatCast(core.quant.nvfp4.e2m1(@intCast(i)));
        for (0..256) |i| m.t256.slice(f32, 256)[i] = @floatCast(core.quant.fp8.e4m3(@intCast(i)));
        try m.buildPipelines();
        errdefer {
            for (m.glue) |p| p.deinit();
            for (m.experts) |row| for (row) |p| p.deinit();
        }
        if (m.config.checkpoint == .mlx) m.affine_pipes = try kernels.AffinePipes.init(m.device);
        return m;
    }

    pub fn deinit(m: *Model) void {
        if (m.affine_pipes) |p| p.deinit();
        for (m.glue) |p| p.deinit();
        for (m.experts) |row| for (row) |p| p.deinit();
        m.t16.deinit();
        m.t256.deinit();
        for (m.owned.items) |buf| buf.deinit();
        m.unmapShards();
        m.arena.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }

    fn unmapShards(m: *Model) void {
        for (m.shards) |s| {
            s.buffer.deinit();
            if (s.map) |map| map.deinit();
        }
    }

    fn mapShard(m: *Model, dir: []const u8, file: []const u8) !Shard {
        const a = m.arena.allocator();
        const path = try std.fs.path.joinZ(a, &.{ dir, file });
        const map = try mtl.MappedFile.open(path.ptr);
        errdefer map.deinit();
        if (map.size < 8) return error.BadSafetensors;
        const n = try st.headerSize(std.mem.readInt(u64, map.bytes[0..8], .little), map.size);
        const data = 8 + n;
        const prefixes: []const []const u8 = if (m.config.checkpoint == .mlx) &.{ "model.", "lm_head." } else &.{"model.llm."};
        const header_bytes = if (m.config.checkpoint == .mlx) try a.dupe(u8, map.bytes[8..data]) else map.bytes[8..data];
        const header = try st.parseHeaderPrefixes(a, header_bytes, map.size - data, prefixes);
        if (m.config.checkpoint == .mlx) {
            for (header.values()) |entry| if (entry.begin % 4 != 0) return error.UnalignedTensor;
            const buffer = try m.device.buffer(map.size - data, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            @memcpy(buffer.contents()[0 .. map.size - data], map.bytes[data..map.size]);
            map.deinit();
            return .{ .map = null, .buffer = buffer, .data = 0, .header = header };
        }
        const buffer = try m.device.bufferNoCopy(map.bytes.ptr, map.bytes.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        return .{ .map = map, .buffer = buffer, .data = data, .header = header };
    }

    /// `config.json` and the shard's header, as one parse: the shared package needs the loop block it drops.
    fn readConfig(a: std.mem.Allocator, io: std.Io, dir: []const u8) !cfg.Config {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
        var why: cfg.Why = .{};
        return cfg.parse(a, text, &why) catch |e| {
            std.log.err("inkling2_loop: {s}", .{why.text()});
            return e;
        };
    }

    pub fn get(m: *Model, name: []const u8) !Tensor {
        for (m.shards) |s| if (s.header.get(name)) |e| {
            var t = Tensor{ .buffer = s.buffer, .offset = s.data + e.begin, .bytes = e.end - e.begin, .dtype = e.dtype, .rank = e.rank };
            for (0..e.rank) |i| t.shape[i] = e.shape[i];
            return t;
        };
        std.log.err("inkling2_loop: the checkpoint has no {s}", .{name});
        return error.MissingTensor;
    }

    fn want(m: *Model, name: []const u8, dtype: st.DType, dims: []const usize) !Tensor {
        const t = try m.get(name);
        if (t.dtype != dtype or t.rank != dims.len) {
            std.log.err("inkling2_loop: {s} is {s} {any}, expected {s} {any}", .{ name, @tagName(t.dtype), t.shape[0..t.rank], @tagName(dtype), dims });
            return error.BadTensor;
        }
        for (dims, t.shape[0..t.rank]) |w, g| if (w != g) {
            std.log.err("inkling2_loop: {s} has shape {any}, expected {any}", .{ name, t.shape[0..t.rank], dims });
            return error.BadTensor;
        };
        return t;
    }

    fn ownedTensor(m: *Model, rows: usize, cols: usize) !Tensor {
        const buf = try m.device.buffer(rows * cols * 4, mtl.ResourceOptions.shared);
        errdefer buf.deinit();
        try m.owned.append(m.arena.allocator(), buf);
        return .{ .buffer = buf, .offset = 0, .bytes = rows * cols * 4, .dtype = .f32, .shape = .{ rows, cols, 1, 1, 1 }, .rank = 2 };
    }

    /// Read `<dir>/mtp.safetensors` into one GPU buffer and check every tensor against `mtp_config`. The text graph never
    /// reads these weights, so a model that serves without drafts never loads them.
    pub fn loadDraft(m: *Model, dir: []const u8) !void {
        if (m.draft != null) return;
        const head = m.config.mtp orelse return error.NoDraftHead;
        const a = m.arena.allocator();
        const path = try std.fs.path.joinZ(a, &.{ dir, "mtp.safetensors" });
        const map = try mtl.MappedFile.open(path.ptr);
        defer map.deinit();
        if (map.size < 8) return error.BadSafetensors;
        const n = try st.headerSize(std.mem.readInt(u64, map.bytes[0..8], .little), map.size);
        const data = 8 + n;
        const header = try st.parseHeader(a, try a.dupe(u8, map.bytes[8..data]), map.size - data);
        for (header.values()) |entry| if (entry.begin % 16 != 0) return error.UnalignedTensor;
        const buffer = try m.device.buffer(@max(map.size - data, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buffer.deinit();
        @memcpy(buffer.contents()[0 .. map.size - data], map.bytes[data..map.size]);
        const Find = struct {
            header: *const st.Header,
            buffer: mtl.Buffer,
            fn want(f: @This(), name: []const u8, dims: []const usize) !Tensor {
                const e = f.header.get(name) orelse {
                    std.log.err("inkling2_loop: mtp.safetensors has no {s}", .{name});
                    return error.MissingTensor;
                };
                var ok = e.dtype == .bf16 and e.rank == dims.len;
                if (ok) for (dims, e.shape[0..e.rank]) |w, g| {
                    if (w != g) ok = false;
                };
                if (!ok) {
                    std.log.err("inkling2_loop: mtp.safetensors {s} is {s} {any}, expected bf16 {any}", .{ name, @tagName(e.dtype), e.shape[0..e.rank], dims });
                    return error.BadTensor;
                }
                var t = Tensor{ .buffer = f.buffer, .offset = e.begin, .bytes = e.end - e.begin, .dtype = e.dtype, .rank = e.rank };
                for (0..e.rank) |i| t.shape[i] = e.shape[i];
                return t;
            }
        };
        const f = Find{ .header = &header, .buffer = buffer };
        const h: usize = head.hidden;
        const qw: usize = head.q_heads * head.head_dim;
        const kvw: usize = head.kv_heads * head.head_dim;
        const inter: usize = head.inter;
        const rank: usize = head.rank;
        const taps: usize = head.taps_n;
        if (h != m.config.text.hidden or head.layers == 0) return error.BadDraftGeometry;
        const layers = try a.alloc(DraftLayer, head.layers);
        for (layers, 0..) |*l, i| {
            const p = try std.fmt.allocPrint(a, "layers.{d}.", .{i});
            l.* = .{
                .input_norm = try f.want(sub(a, p, "input_layernorm.weight"), &.{h}),
                .post_norm = try f.want(sub(a, p, "post_attention_layernorm.weight"), &.{h}),
                .mix_norm = try f.want(sub(a, p, "dflare_norm.weight"), &.{h}),
                .q = try f.want(sub(a, p, "self_attn.q_proj.weight"), &.{ qw, h }),
                .k = try f.want(sub(a, p, "self_attn.k_proj.weight"), &.{ kvw, h }),
                .v = try f.want(sub(a, p, "self_attn.v_proj.weight"), &.{ kvw, h }),
                .k_ctx = try f.want(sub(a, p, "self_attn.k_proj_ctx.weight"), &.{ kvw, h }),
                .v_ctx = try f.want(sub(a, p, "self_attn.v_proj_ctx.weight"), &.{ kvw, h }),
                .o = try f.want(sub(a, p, "self_attn.o_proj.weight"), &.{ h, qw }),
                .q_norm = try f.want(sub(a, p, "self_attn.q_norm.weight"), &.{head.head_dim}),
                .k_norm = try f.want(sub(a, p, "self_attn.k_norm.weight"), &.{head.head_dim}),
                .k_norm_ctx = try f.want(sub(a, p, "self_attn.k_norm_ctx.weight"), &.{head.head_dim}),
                .gate = try f.want(sub(a, p, "mlp.gate_proj.weight"), &.{ inter, h }),
                .up = try f.want(sub(a, p, "mlp.up_proj.weight"), &.{ inter, h }),
                .down = try f.want(sub(a, p, "mlp.down_proj.weight"), &.{ h, inter }),
            };
        }
        const vocab: usize = m.config.text.vocab;
        const weights = DraftWeights{
            .buffer = buffer,
            .layers = layers,
            .norm = try f.want("norm.weight", &.{h}),
            .mask = try f.want("mask_embed", &.{h}),
            .fuse = try f.want("fuse_weight", &.{ head.layers + 1, taps }),
            .markov_w1 = try f.want("markov_head.markov_w1.weight", &.{ vocab, rank }),
            .rnn = try f.want("markov_head.rnn_proj.weight", &.{ 3 * rank, 2 * rank + 2 * h }),
            .rnn_bias = try f.want("markov_head.rnn_proj.bias", &.{3 * rank}),
            .hidden_out = try f.want("markov_head.hidden_out.weight", &.{ h, rank }),
            .hidden_out_bias = try f.want("markov_head.hidden_out.bias", &.{h}),
            .markov_norm = try f.want("markov_head.dflare_norm.weight", &.{h}),
        };
        try m.owned.append(a, buffer);
        m.draft = weights;
    }

    /// A buffer the model frees with its weights, for tensors derived once from them.
    pub fn ownBuffer(m: *Model, bytes: usize) !mtl.Buffer {
        const buf = try m.device.buffer(@max(bytes, 16), mtl.ResourceOptions.private);
        errdefer buf.deinit();
        try m.owned.append(m.arena.allocator(), buf);
        return buf;
    }

    fn affineWeight(m: *Model, base: []const u8, rows: usize, cols: usize, bits: u32) !Tensor {
        if ((bits != 4 and bits != 8) or cols % 64 != 0) return error.BadGeometry;
        const a = m.arena.allocator();
        var words = try m.want(try std.fmt.allocPrint(a, "{s}.weight", .{base}), .u32, &.{ rows, cols / (32 / bits) });
        const scales = try m.want(try std.fmt.allocPrint(a, "{s}.scales", .{base}), .bf16, &.{ rows, cols / 64 });
        const biases = try m.want(try std.fmt.allocPrint(a, "{s}.biases", .{base}), .bf16, &.{ rows, cols / 64 });
        words.shape[1] = cols;
        words.affine = .{ .scales = scales.buffer, .scale_offset = scales.offset, .biases = biases.buffer, .bias_offset = biases.offset, .kind = if (bits == 8) .affine8 else .affine4 };
        return words;
    }

    fn mlxConv(m: *Model, name: []const u8, channels: usize, full: bool) !Tensor {
        var weight = if (full) try m.want(name, .bf16, &.{ 4, channels }) else try m.want(name, .bf16, &.{ channels, 4 });
        weight.shape = .{ 4, channels, 1, 1, 1 };
        return weight;
    }

    /// A convolution weight as [taps][channels], the view the kernel's two layouts name. SWA layers store
    /// [channels, 1, taps] and full layers `static_w` [taps, channels], which the reference reverses.
    fn convWeight(m: *Model, name: []const u8, channels: usize, swa: bool) !Tensor {
        const t = try m.get(name);
        if (t.dtype != .bf16) return error.BadTensor;
        if (swa) {
            if (t.rank != 3 or t.shape[0] != channels or t.shape[1] != 1 or t.shape[2] != 4) return error.BadTensor;
            // [channels, 1, taps] and [taps][channels] name the same bytes: index tap * channels + channel
            return .{ .buffer = t.buffer, .offset = t.offset, .bytes = t.bytes, .dtype = t.dtype, .rank = 2, .shape = .{ 4, channels, 1, 1, 1 } };
        }
        if (t.rank != 2 or t.shape[0] != 4 or t.shape[1] != channels) return error.BadTensor;
        return t;
    }

    fn mapWeights(m: *Model, formats: []const cfg.Expert) !Weights {
        const c = &m.config;
        const a = m.arena.allocator();
        const h: usize = c.text.hidden;
        const L: usize = c.text.layers;
        var w: Weights = undefined;
        w.embed = try m.want("model.llm.embed.weight", .bf16, &.{ c.text.vocab, h });
        w.embed_norm = try m.want("model.llm.embed_norm.weight", .bf16, &.{h});
        w.norm = try m.want("model.llm.norm.weight", .bf16, &.{h});
        w.unembed = try m.want("model.llm.unembed.weight", .bf16, &.{ c.text.vocab, h });
        w.output_query = try m.want("model.llm.attn_res.output_query", .f32, &.{h});
        w.cross_pass_q = try m.want("model.llm.middle_cycle_cross_pass_queries", .f32, &.{ 2 * (c.loop.core_end - c.loop.core_start), h });
        w.cross_pass_kv = try m.want("model.llm.middle_cycle_cross_pass_kv_queries", .f32, &.{ 2 * (c.loop.core_end - c.loop.core_start), h });
        w.layers = try a.alloc(Layer, L);
        w.attn_queries = try a.alloc(Tensor, L - 1);
        w.mlp_queries = try a.alloc(Tensor, L);
        for (0..L) |i| {
            const name = try std.fmt.allocPrint(a, "model.llm.attn_res.mlp_queries.{d}", .{i});
            w.mlp_queries[i] = try m.want(name, .f32, &.{h});
        }
        for (0..L - 1) |i| {
            const name = try std.fmt.allocPrint(a, "model.llm.attn_res.attn_queries.{d}", .{i});
            w.attn_queries[i] = if (c.isFull(@intCast(i + 1)))
                try m.want(name, .f32, &.{ 3, h })
            else
                try m.want(name, .f32, &.{h});
        }
        for (0..L) |i| w.layers[i] = try mapLayer(m, @intCast(i), formats[i]);
        return w;
    }

    fn mapMlxWeights(m: *Model) !Weights {
        const c = &m.config;
        const a = m.arena.allocator();
        const h: usize = c.text.hidden;
        const layers: usize = c.text.layers;
        var w: Weights = undefined;
        w.embed = try m.affineWeight("model.embed", c.text.vocab, h, 8);
        w.embed_norm = try m.want("model.embed_norm.weight", .bf16, &.{h});
        w.norm = try m.want("model.norm.weight", .bf16, &.{h});
        w.unembed = try m.affineWeight("lm_head", c.text.vocab, h, 8);
        w.output_query = try m.want("model.final_input.query", .f32, &.{ 1, h });
        const core_layers = c.loop.core_end - c.loop.core_start;
        w.cross_pass_q = try m.ownedTensor(2 * core_layers, h);
        w.cross_pass_kv = try m.ownedTensor(2 * core_layers, h);
        for (0..core_layers) |i| {
            const name = try std.fmt.allocPrint(a, "model.layers.{d}.loop_message.query", .{i + c.loop.core_start});
            const src = try m.want(name, .f32, &.{ 4, h });
            const input = src.buffer.contents()[src.offset..][0..src.bytes];
            const q = w.cross_pass_q.buffer.contents();
            const kv = w.cross_pass_kv.buffer.contents();
            @memcpy(q[(2 * i) * h * 4 ..][0 .. h * 4], input[0 .. h * 4]);
            @memcpy(kv[(2 * i) * h * 4 ..][0 .. 2 * h * 4], input[h * 4 ..][0 .. 2 * h * 4]);
            @memcpy(q[(2 * i + 1) * h * 4 ..][0 .. h * 4], input[3 * h * 4 ..][0 .. h * 4]);
        }
        w.layers = try a.alloc(Layer, layers);
        w.attn_queries = try a.alloc(Tensor, layers - 1);
        w.mlp_queries = try a.alloc(Tensor, layers);
        for (0..layers) |i| {
            const mlp = try std.fmt.allocPrint(a, "model.layers.{d}.mlp_input.query", .{i});
            w.mlp_queries[i] = try m.want(mlp, .f32, &.{ 1, h });
            if (i > 0) {
                const attn = try std.fmt.allocPrint(a, "model.layers.{d}.attn_input.query", .{i});
                w.attn_queries[i - 1] = try m.want(attn, .f32, &.{ if (c.isFull(@intCast(i))) 3 else 1, h });
            }
            w.layers[i] = try name_map.mapMlxLayer(MlxLoader{ .model = m, .config = c, .arena = &m.arena }, @intCast(i));
        }
        return w;
    }

    pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
        return m.glue[@backingInt(g)];
    }

    pub fn expert(m: *const Model, mx: bool, e: Experts) mtl.Pipeline {
        return m.experts[@intFromBool(mx)][@backingInt(e)];
    }

    fn buildPipelines(m: *Model) !void {
        const opts = mtl.CompileOptions.mlx();
        {
            const lib = try mtl.Library.fromSource(m.device, kernels.glue, opts);
            defer lib.deinit();
            for (glue_names, 0..) |n, i| m.glue[i] = try mtl.Pipeline.init(m.device, lib, n, false);
        }
        for ([_]bool{ false, true }, 0..) |mx, i| {
            const source = if (mx) kernels.expertSource(true) else kernels.expertSource(false);
            const lib = try mtl.Library.fromSource(m.device, source, opts);
            defer lib.deinit();
            for (expertKernels(mx), 0..) |n, j| m.experts[i][j] = try mtl.Pipeline.init(m.device, lib, n, false);
        }
    }
};

fn layerPrefix(a: std.mem.Allocator, i: u32) ![]const u8 {
    return std.fmt.allocPrint(a, "model.llm.layers.{d}.", .{i});
}

fn sub(a: std.mem.Allocator, prefix: []const u8, suffix: []const u8) []const u8 {
    return std.fmt.allocPrint(a, "{s}{s}", .{ prefix, suffix }) catch unreachable;
}

const MlxLoader = struct {
    model: *Model,
    config: *const cfg.Config,
    arena: *std.heap.ArenaAllocator,

    pub fn want(l: MlxLoader, name: []const u8, dtype: st.DType, dims: []const usize) !Tensor {
        return l.model.want(name, dtype, dims);
    }
    pub fn ownedTensor(l: MlxLoader, rows: usize, cols: usize) !Tensor {
        return l.model.ownedTensor(rows, cols);
    }
    pub fn affineWeight(l: MlxLoader, name: []const u8, rows: usize, cols: usize, bits: u32) !Tensor {
        return l.model.affineWeight(name, rows, cols, bits);
    }
    pub fn mlxConv(l: MlxLoader, name: []const u8, channels: usize, full: bool) !Tensor {
        return l.model.mlxConv(name, channels, full);
    }
    pub fn controllers(l: MlxLoader, prefix: []const u8, out: *Layer) !void {
        return mapMlxControllers(l.model, prefix, out);
    }
};

fn mapMlxControllers(m: *Model, prefix: []const u8, out: *Layer) !void {
    const h: usize = m.config.text.hidden;
    const rank: usize = m.config.loop.gate_rank;
    const a = m.arena.allocator();
    const down = try m.want(sub(a, prefix, "loop_down"), .f32, &.{ 4, 2, h, rank });
    const gates = try m.want(sub(a, prefix, "loop_gain_forget"), .f32, &.{ 4, rank, 2 * h });
    const ds = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(down.buffer.contents()[down.offset..][0..down.bytes])));
    const gs = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(gates.buffer.contents()[gates.offset..][0..gates.bytes])));
    for (0..4) |j| {
        const d = try m.ownedTensor(rank, 2 * h);
        const g = try m.ownedTensor(h, rank);
        const f = try m.ownedTensor(h, rank);
        const dv = d.buffer.slice(f32, rank * 2 * h);
        const gv = g.buffer.slice(f32, h * rank);
        const fv = f.buffer.slice(f32, h * rank);
        for (0..rank) |r| for (0..2 * h) |col| {
            dv[r * 2 * h + col] = ds[(j * 2 * h + col) * rank + r];
        };
        for (0..h) |col| for (0..rank) |r| {
            gv[col * rank + r] = gs[(j * rank + r) * 2 * h + col];
            fv[col * rank + r] = gs[(j * rank + r) * 2 * h + h + col];
        };
        out.controllers[j] = .{ .down = d, .gain = g, .forget = f };
    }
}

fn mapLayer(m: *Model, i: u32, prec: cfg.Expert) !Layer {
    const c = &m.config;
    const a = m.arena.allocator();
    const h: usize = c.text.hidden;
    const full = c.isFull(i);
    const heads = c.heads(i);
    const qw: usize = heads.q * heads.dim;
    const kvw: usize = heads.kv * heads.dim;
    const p = try layerPrefix(a, i);
    const extent: usize = c.extent(i);
    const inter: usize = c.text.expert_inter;
    const E: usize = c.text.experts;
    var out: Layer = undefined;
    out.precision = if (prec == .mxfp8) .mxfp8 else .nvfp4;
    out.w13_bias = null;
    out.w2_bias = null;
    if (full) {
        out.q_norm = try m.want(sub(a, p, "q_input_norm.weight"), .bf16, &.{h});
        out.k_norm = try m.want(sub(a, p, "k_input_norm.weight"), .bf16, &.{h});
        out.v_norm = try m.want(sub(a, p, "v_input_norm.weight"), .bf16, &.{h});
    } else {
        out.q_norm = try m.want(sub(a, p, "attn_norm.weight"), .bf16, &.{h});
        out.k_norm = out.q_norm;
        out.v_norm = out.q_norm;
    }
    out.attn = .{
        .q = try m.want(sub(a, p, "attn.wq_du.weight"), .bf16, &.{ qw, h }),
        .k = try m.want(sub(a, p, "attn.wk_dv.weight"), .bf16, &.{ kvw, h }),
        .v = try m.want(sub(a, p, "attn.wv_dv.weight"), .bf16, &.{ kvw, h }),
        .o = try m.want(sub(a, p, "attn.wo_ud.weight"), .bf16, &.{ h, qw }),
        .r = try m.want(sub(a, p, "attn.wr_du.weight"), .bf16, &.{ heads.q * c.text.d_rel, h }),
        .gate = try m.want(sub(a, p, "attn.gate_attn.weight"), .bf16, &.{ qw, h }),
        .sink = try m.want(sub(a, p, "attn.attn_sink_proj.weight"), .bf16, &.{ heads.q, h }),
        .q_scale = try m.want(sub(a, p, "attn.q_l2_clip.scale"), .bf16, &.{ heads.q, heads.dim }),
        .r_scale = try m.want(sub(a, p, "attn.rel_l2_clip.scale"), .bf16, &.{ heads.q, c.text.d_rel }),
        .rel_bias = try m.want(sub(a, p, "attn.rel_logits_proj.proj"), .bf16, &.{ c.text.d_rel, extent }),
        .k_conv = try m.convWeight(sub(a, p, if (full) "attn.k_sconv.static_w" else "attn.k_sconv.weight"), kvw, !full),
        .v_conv = try m.convWeight(sub(a, p, if (full) "attn.v_sconv.static_w" else "attn.v_sconv.weight"), kvw, !full),
    };
    if (full) {
        const kv: usize = heads.kv;
        out.attn.k_dyn = try m.want(sub(a, p, "attn.k_sconv.weight_proj"), .bf16, &.{ kv * 4, h });
        out.attn.v_dyn = try m.want(sub(a, p, "attn.v_sconv.weight_proj"), .bf16, &.{ kv * 4, h });
    }
    out.attn_post = try m.convWeight(sub(a, p, "attn_sconv.weight"), h, true);
    out.mlp_norm = try m.want(sub(a, p, "mlp_norm.weight"), .bf16, &.{h});
    out.mlp_post = try m.convWeight(sub(a, p, "mlp_sconv.weight"), h, true);
    out.router = try m.want(sub(a, p, "mlp.gate.weight"), .bf16, &.{ E, h });
    out.router_bias = try m.want(sub(a, p, "mlp.gate.bias"), .f32, &.{E});
    switch (prec) {
        .nvfp4 => {
            out.w13 = try m.want(sub(a, p, "mlp.experts.w13_weight"), .u8, &.{ E, 2 * inter, h / 2 });
            out.w13_scale = try m.want(sub(a, p, "mlp.experts.w13_weight_scale"), .f8_e4m3, &.{ E, 2 * inter, h / 16 });
            out.w13_global = try m.want(sub(a, p, "mlp.experts.w13_weight_amax"), .f32, &.{E});
            out.w13_den = try m.want(sub(a, p, "mlp.experts.w13_weight_outer_scale_denominator"), .f32, &.{1});
            out.w2 = try m.want(sub(a, p, "mlp.experts.w2_weight"), .u8, &.{ E, h, inter / 2 });
            out.w2_scale = try m.want(sub(a, p, "mlp.experts.w2_weight_scale"), .f8_e4m3, &.{ E, h, inter / 16 });
            out.w2_global = try m.want(sub(a, p, "mlp.experts.w2_weight_amax"), .f32, &.{E});
            out.w2_den = try m.want(sub(a, p, "mlp.experts.w2_weight_outer_scale_denominator"), .f32, &.{1});
        },
        .mxfp8 => {
            out.w13 = try m.want(sub(a, p, "mlp.experts.w13_weight"), .f8_e4m3, &.{ E, 2 * inter, h });
            out.w13_scale = try m.want(sub(a, p, "mlp.experts.w13_weight_scale"), .u8, &.{ E, 2 * inter, h / 32 });
            out.w2 = try m.want(sub(a, p, "mlp.experts.w2_weight"), .f8_e4m3, &.{ E, h, inter });
            out.w2_scale = try m.want(sub(a, p, "mlp.experts.w2_weight_scale"), .u8, &.{ E, h, inter / 32 });
        },
    }
    if (i >= c.loop.core_start and i < c.loop.core_end) {
        const rank: usize = c.loop.gate_rank;
        inline for (.{ "attn", "k", "v", "mlp" }, 0..) |name, j| {
            out.controllers[j] = .{
                .down = try m.want(sub(a, p, "controllers." ++ name ++ ".down_weight"), .bf16, &.{ rank, 2 * h }),
                .gain = try m.want(sub(a, p, "controllers." ++ name ++ ".gain_weight"), .bf16, &.{ h, rank }),
                .forget = try m.want(sub(a, p, "controllers." ++ name ++ ".forget_weight"), .bf16, &.{ h, rank }),
            };
        }
    }
    return out;
}

fn buildPipelines(m: *Model) !void {
    const opts = mtl.CompileOptions.mlx();
    {
        const lib = try mtl.Library.fromSource(m.device, kernels.glue, opts);
        defer lib.deinit();
        for (glue_names, 0..) |n, i| m.glue[i] = try mtl.Pipeline.init(m.device, lib, n, false);
    }
    for ([_]bool{ false, true }, 0..) |mx, i| {
        const lib = try mtl.Library.fromSource(m.device, kernels.expertSource(mx), opts);
        defer lib.deinit();
        for (expertKernels(mx), 0..) |n, j| m.experts[i][j] = try mtl.Pipeline.init(m.device, lib, n, false);
    }
}

pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
    return m.glue[@backingInt(g)];
}

pub fn expert(m: *const Model, mx: bool, e: Experts) mtl.Pipeline {
    return m.experts[@intFromBool(mx)][@backingInt(e)];
}

test {
    _ = kernels;
}
