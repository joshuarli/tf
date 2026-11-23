//! Inkling 2 Mini's weight map: every tensor the config implies, with its dtype, shape and typed role, checked
//! against the checkpoint headers. Expert geometry is checked by the shared NVFP4 reader and the MXFP8 block rule.
const std = @import("std");
const core = @import("core");
const st = core.safetensors;
const config = @import("config.zig");
const quant = @import("quant.zig");

pub const DType = st.DType;
pub const File = enum { model, mtp };

/// A tensor's job in the graph, or that it belongs to the refused vision tower.
pub const Role = enum {
    embed,
    embed_norm,
    final_norm,
    unembed,
    attn_res_attn_query,
    attn_res_mlp_query,
    attn_res_output_query,
    cross_pass_query,
    cross_pass_kv_query,
    attn_q,
    attn_k,
    attn_v,
    attn_o,
    attn_r,
    attn_gate,
    attn_sink,
    attn_q_scale,
    attn_r_scale,
    attn_rel_bias,
    sconv_static,
    sconv_dynamic,
    sconv_post,
    input_norm,
    expert_w13,
    expert_w13_scale,
    expert_w13_amax,
    expert_w13_denominator,
    expert_w2,
    expert_w2_scale,
    expert_w2_amax,
    expert_w2_denominator,
    router,
    router_bias,
    controller_down,
    controller_gain,
    controller_forget,
    vision_embedding,
    vision_norm,
    vision_attention,
    vision_mlp,
    vision_projection,
    draft_head,
};

pub const Precision = quant.Kind;

const Text = @import("inkling2_config").Text;
const Mtp = config.Mtp;

fn put(sink: anytype, comptime fmt: []const u8, args: anytype, dtype: DType, dims: []const u64, role: Role) !void {
    var buf: [128]u8 = undefined;
    try sink.put(try std.fmt.bufPrint(&buf, fmt, args), dtype, dims, role);
}

fn attention(c: *const config.Config, i: u32, sink: anytype) !void {
    const h: u64 = c.text.hidden;
    const hd = c.heads(i);
    const q: u64 = @as(u64, hd.q) * hd.dim;
    const kv: u64 = @as(u64, hd.kv) * hd.dim;
    const k: u64 = config.conv_taps;
    const L = .{i};
    try put(sink, "model.llm.layers.{d}.attn.wq_du.weight", L, .bf16, &.{ q, h }, .attn_q);
    try put(sink, "model.llm.layers.{d}.attn.wk_dv.weight", L, .bf16, &.{ kv, h }, .attn_k);
    try put(sink, "model.llm.layers.{d}.attn.wv_dv.weight", L, .bf16, &.{ kv, h }, .attn_v);
    try put(sink, "model.llm.layers.{d}.attn.wr_du.weight", L, .bf16, &.{ @as(u64, hd.q) * c.text.d_rel, h }, .attn_r);
    try put(sink, "model.llm.layers.{d}.attn.wo_ud.weight", L, .bf16, &.{ h, q }, .attn_o);
    try put(sink, "model.llm.layers.{d}.attn.gate_attn.weight", L, .bf16, &.{ q, h }, .attn_gate);
    try put(sink, "model.llm.layers.{d}.attn.attn_sink_proj.weight", L, .bf16, &.{ hd.q, h }, .attn_sink);
    try put(sink, "model.llm.layers.{d}.attn.q_l2_clip.scale", L, .bf16, &.{ hd.q, hd.dim }, .attn_q_scale);
    try put(sink, "model.llm.layers.{d}.attn.rel_l2_clip.scale", L, .bf16, &.{ hd.q, c.text.d_rel }, .attn_r_scale);
    try put(sink, "model.llm.layers.{d}.attn.rel_logits_proj.proj", L, .bf16, &.{ c.text.d_rel, c.extent(i) }, .attn_rel_bias);
    inline for (.{ "k", "v" }) |kind| {
        if (c.isFull(i)) {
            try put(sink, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.static_w", L, .bf16, &.{ k, kv }, .sconv_static);
            try put(sink, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.weight_proj", L, .bf16, &.{ @as(u64, hd.kv) * k, h }, .sconv_dynamic);
        } else {
            try put(sink, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.weight", L, .bf16, &.{ kv, 1, k }, .sconv_post);
        }
    }
    if (c.isFull(i)) {
        inline for (.{ "q", "k", "v" }) |kind| try put(sink, "model.llm.layers.{d}." ++ kind ++ "_input_norm.weight", L, .bf16, &.{h}, .input_norm);
    } else {
        try put(sink, "model.llm.layers.{d}.attn_norm.weight", L, .bf16, &.{h}, .input_norm);
    }
}

fn controllers(c: *const config.Config, i: u32, sink: anytype) !void {
    if (i < c.loop.core_start or i >= c.loop.core_end) return;
    const h: u64 = c.text.hidden;
    const rank: u64 = c.loop.gate_rank;
    inline for (.{ "attn", "k", "v", "mlp" }) |name| {
        try put(sink, "model.llm.layers.{d}.controllers." ++ name ++ ".down_weight", .{i}, .bf16, &.{ rank, 2 * h }, .controller_down);
        try put(sink, "model.llm.layers.{d}.controllers." ++ name ++ ".gain_weight", .{i}, .bf16, &.{ h, rank }, .controller_gain);
        try put(sink, "model.llm.layers.{d}.controllers." ++ name ++ ".forget_weight", .{i}, .bf16, &.{ h, rank }, .controller_forget);
    }
}

fn experts(c: *const config.Config, i: u32, prec: Precision, sink: anytype) !void {
    const h: u64 = c.text.hidden;
    const m: u64 = c.text.expert_inter;
    const e: u64 = c.text.experts;
    const L = .{i};
    try put(sink, "model.llm.layers.{d}.mlp.gate.weight", L, .bf16, &.{ e, h }, .router);
    try put(sink, "model.llm.layers.{d}.mlp.gate.bias", L, .f32, &.{e}, .router_bias);
    const rows = [2][2]u64{ .{ 2 * m, h }, .{ h, m } };
    inline for (.{ "w13", "w2" }, 0..) |w, n| {
        const r = rows[n][0];
        const k = rows[n][1];
        switch (prec) {
            .nvfp4 => {
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight", L, .u8, &.{ e, r, k / 2 }, if (n == 0) .expert_w13 else .expert_w2);
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight_scale", L, .f8_e4m3, &.{ e, r, k / core.nvfp4.group }, if (n == 0) .expert_w13_scale else .expert_w2_scale);
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight_amax", L, .f32, &.{e}, if (n == 0) .expert_w13_amax else .expert_w2_amax);
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight_outer_scale_denominator", L, .f32, &.{1}, if (n == 0) .expert_w13_denominator else .expert_w2_denominator);
            },
            .mxfp8 => {
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight", L, .f8_e4m3, &.{ e, r, k }, if (n == 0) .expert_w13 else .expert_w2);
                try put(sink, "model.llm.layers.{d}.mlp.experts." ++ w ++ "_weight_scale", L, .u8, &.{ e, r, k / mxfp8_group }, if (n == 0) .expert_w13_scale else .expert_w2_scale);
            },
            .affine4, .affine8 => return error.Unsupported,
        }
    }
}

fn mlxSub(a: std.mem.Allocator, prefix: []const u8, suffix: []const u8) []const u8 {
    return std.fmt.allocPrint(a, "{s}{s}", .{ prefix, suffix }) catch unreachable;
}

fn affineRows(t: anytype, start: usize, count: usize) @TypeOf(t) {
    var out = t;
    const q = out.affine.?;
    const k = t.shape[1];
    out.offset += start * k * q.bits() / 8;
    out.bytes = count * k * q.bits() / 8;
    out.shape[0] = count;
    out.affine.?.scale_offset += start * (k / 64) * 2;
    out.affine.?.bias_offset += start * (k / 64) * 2;
    return out;
}

pub fn mapMlxLayer(m: anytype, i: u32) !@TypeOf(m.model.weights.layers[0]) {
    const c = m.config;
    const a = m.arena.allocator();
    const h: usize = c.text.hidden;
    const hd = c.heads(i);
    const q: usize = hd.q * hd.dim;
    const kv: usize = hd.kv * hd.dim;
    const rel: usize = hd.q * c.text.d_rel;
    const inter: usize = c.text.expert_inter;
    const expert_count: usize = c.text.experts;
    const full = c.isFull(i);
    const p = try std.fmt.allocPrint(a, "model.layers.{d}.", .{i});
    var out: @TypeOf(m.model.weights.layers[0]) = undefined;
    out.attn.k_dyn = null;
    out.attn.v_dyn = null;
    out.w13_global = null;
    out.w13_den = null;
    out.w2_global = null;
    out.w2_den = null;
    out.w13_bias = null;
    out.w2_bias = null;
    out.precision = if (i < 20) .nvfp4 else .affine4;
    out.controllers = @splat(null);
    if (full) {
        out.q_norm = try m.want(mlxSub(a, p, "q_input_norm.weight"), .bf16, &.{h});
        out.k_norm = try m.want(mlxSub(a, p, "k_input_norm.weight"), .bf16, &.{h});
        out.v_norm = try m.want(mlxSub(a, p, "v_input_norm.weight"), .bf16, &.{h});
    } else {
        out.q_norm = try m.want(mlxSub(a, p, "attn_norm.weight"), .bf16, &.{h});
        out.k_norm = out.q_norm;
        out.v_norm = out.q_norm;
    }
    const projection = try m.affineWeight(mlxSub(a, p, if (full) "attn.q_proj" else "attn.in_proj"), q + (if (full) @as(usize, 0) else 2 * kv) + rel + q + hd.q, h, 8);
    var at: usize = 0;
    out.attn.q = affineRows(projection, at, q); at += q;
    if (!full) {
        out.attn.k = affineRows(projection, at, kv); at += kv;
        out.attn.v = affineRows(projection, at, kv); at += kv;
    } else {
        const kp = try m.affineWeight(mlxSub(a, p, "attn.k_proj"), kv + hd.kv * 4, h, 8);
        const vp = try m.affineWeight(mlxSub(a, p, "attn.v_proj"), kv + hd.kv * 4, h, 8);
        out.attn.k = affineRows(kp, 0, kv);
        out.attn.k_dyn = affineRows(kp, kv, hd.kv * 4);
        out.attn.v = affineRows(vp, 0, kv);
        out.attn.v_dyn = affineRows(vp, kv, hd.kv * 4);
    }
    out.attn.r = affineRows(projection, at, rel); at += rel;
    out.attn.gate = affineRows(projection, at, q); at += q;
    out.attn.sink = affineRows(projection, at, hd.q);
    out.attn.o = try m.affineWeight(mlxSub(a, p, "attn.o_proj"), h, q, 8);
    out.attn.q_scale = try m.want(mlxSub(a, p, "attn.q_scale"), .bf16, &.{ hd.q, hd.dim });
    out.attn.r_scale = try m.want(mlxSub(a, p, "attn.r_scale"), .bf16, &.{ hd.q, c.text.d_rel });
    out.attn.rel_bias = try m.want(mlxSub(a, p, "attn.rel_proj"), .bf16, &.{ c.text.d_rel, c.extent(i) });
    out.attn.k_conv = try m.mlxConv(mlxSub(a, p, if (full) "attn.k_static" else "attn.k_conv"), kv, full);
    out.attn.v_conv = try m.mlxConv(mlxSub(a, p, if (full) "attn.v_static" else "attn.v_conv"), kv, full);
    out.attn_post = try m.mlxConv(mlxSub(a, p, "attn_conv"), h, false);
    out.mlp_norm = try m.want(mlxSub(a, p, "mlp_norm.weight"), .bf16, &.{h});
    out.mlp_post = try m.mlxConv(mlxSub(a, p, "mlp_conv"), h, false);
    out.router = try m.want(mlxSub(a, p, "mlp.router_weight"), .f32, &.{ expert_count, h });
    out.router_bias = try m.want(mlxSub(a, p, "mlp.router_bias"), .f32, &.{expert_count});
    const w13 = mlxSub(a, p, "mlp.experts.w13");
    const w2 = mlxSub(a, p, "mlp.experts.w2");
    out.w13 = try m.want(mlxSub(a, w13, ".weight"), .u32, &.{ expert_count, 2 * inter, h / 8 });
    out.w2 = try m.want(mlxSub(a, w2, ".weight"), .u32, &.{ expert_count, h, inter / 8 });
    if (i < 20) {
        out.w13_scale = try m.want(mlxSub(a, w13, ".scales"), .u8, &.{ expert_count, 2 * inter, h / 16 });
        out.w2_scale = try m.want(mlxSub(a, w2, ".scales"), .u8, &.{ expert_count, h, inter / 16 });
        out.w13_global = try m.want(mlxSub(a, w13, ".global_scale"), .f32, &.{expert_count});
        out.w2_global = try m.want(mlxSub(a, w2, ".global_scale"), .f32, &.{expert_count});
        const den = try m.ownedTensor(1, 1);
        den.buffer.slice(f32, 1)[0] = 2688;
        out.w13_den = den;
        out.w2_den = den;
    } else {
        out.w13_scale = try m.want(mlxSub(a, w13, ".scales"), .bf16, &.{ expert_count, 2 * inter, h / 64 });
        out.w2_scale = try m.want(mlxSub(a, w2, ".scales"), .bf16, &.{ expert_count, h, inter / 64 });
        out.w13_bias = try m.want(mlxSub(a, w13, ".biases"), .bf16, &.{ expert_count, 2 * inter, h / 64 });
        out.w2_bias = try m.want(mlxSub(a, w2, ".biases"), .bf16, &.{ expert_count, h, inter / 64 });
    }
    if (i >= c.loop.core_start and i < c.loop.core_end) try m.controllers(p, &out);
    return out;
}

fn mlxLinear(sink: anytype, name: []const u8, rows: u64, cols: u64, bits: u64, role: Role) !void {
    var buf: [160]u8 = undefined;
    try sink.put(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), .u32, &.{ rows, cols * bits / 32 }, role);
    try sink.put(try std.fmt.bufPrint(&buf, "{s}.scales", .{name}), .bf16, &.{ rows, cols / 64 }, role);
    try sink.put(try std.fmt.bufPrint(&buf, "{s}.biases", .{name}), .bf16, &.{ rows, cols / 64 }, role);
}

fn mlxBackbone(c: *const config.Config, sink: anytype) !void {
    const h: u64 = c.text.hidden;
    const inter: u64 = c.text.expert_inter;
    const e: u64 = c.text.experts;
    try mlxLinear(sink, "model.embed", c.text.vocab, h, 8, .embed);
    try mlxLinear(sink, "lm_head", c.text.vocab, h, 8, .unembed);
    try put(sink, "model.embed_norm.weight", .{}, .bf16, &.{h}, .embed_norm);
    try put(sink, "model.norm.weight", .{}, .bf16, &.{h}, .final_norm);
    try put(sink, "model.final_input.query", .{}, .f32, &.{ 1, h }, .attn_res_output_query);
    for (0..c.text.layers) |n| {
        const i: u32 = @intCast(n);
        const heads = c.heads(i);
        const q: u64 = @as(u64, heads.q) * heads.dim;
        const kv: u64 = @as(u64, heads.kv) * heads.dim;
        const rel: u64 = @as(u64, heads.q) * c.text.d_rel;
        const full = c.isFull(i);
        var buf: [128]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&buf, "model.layers.{d}.attn.{s}", .{ i, if (full) "q_proj" else "in_proj" });
        try mlxLinear(sink, prefix, q + (if (full) @as(u64, 0) else 2 * kv) + rel + q + heads.q, h, 8, .attn_q);
        if (full) {
            inline for (.{ "k", "v" }) |kind| {
                const name = try std.fmt.bufPrint(&buf, "model.layers.{d}.attn." ++ kind ++ "_proj", .{i});
                try mlxLinear(sink, name, kv + @as(u64, heads.kv) * 4, h, 8, if (kind[0] == 'k') .attn_k else .attn_v);
                try put(sink, "model.layers.{d}.attn." ++ kind ++ "_static", .{i}, .bf16, &.{ 4, kv }, .sconv_static);
                try put(sink, "model.layers.{d}." ++ kind ++ "_input_norm.weight", .{i}, .bf16, &.{h}, .input_norm);
            }
        } else {
            inline for (.{ "k", "v" }) |kind| try put(sink, "model.layers.{d}.attn." ++ kind ++ "_conv", .{i}, .bf16, &.{ kv, 4 }, .sconv_post);
            try put(sink, "model.layers.{d}.attn_norm.weight", .{i}, .bf16, &.{h}, .input_norm);
        }
        if (full) try put(sink, "model.layers.{d}.q_input_norm.weight", .{i}, .bf16, &.{h}, .input_norm);
        try mlxLinear(sink, try std.fmt.bufPrint(&buf, "model.layers.{d}.attn.o_proj", .{i}), h, q, 8, .attn_o);
        try put(sink, "model.layers.{d}.attn.q_scale", .{i}, .bf16, &.{ heads.q, heads.dim }, .attn_q_scale);
        try put(sink, "model.layers.{d}.attn.r_scale", .{i}, .bf16, &.{ heads.q, c.text.d_rel }, .attn_r_scale);
        try put(sink, "model.layers.{d}.attn.rel_proj", .{i}, .bf16, &.{ c.text.d_rel, c.extent(i) }, .attn_rel_bias);
        try put(sink, "model.layers.{d}.attn_conv", .{i}, .bf16, &.{ h, 4 }, .sconv_post);
        try put(sink, "model.layers.{d}.mlp_conv", .{i}, .bf16, &.{ h, 4 }, .sconv_post);
        try put(sink, "model.layers.{d}.mlp_norm.weight", .{i}, .bf16, &.{h}, .input_norm);
        try put(sink, "model.layers.{d}.mlp.router_weight", .{i}, .f32, &.{ e, h }, .router);
        try put(sink, "model.layers.{d}.mlp.router_bias", .{i}, .f32, &.{e}, .router_bias);
        try put(sink, "model.layers.{d}.mlp_input.query", .{i}, .f32, &.{ 1, h }, .attn_res_mlp_query);
        if (i > 0) try put(sink, "model.layers.{d}.attn_input.query", .{i}, .f32, &.{ if (full) 3 else 1, h }, .attn_res_attn_query);
        if (i >= c.loop.core_start and i < c.loop.core_end) {
            try put(sink, "model.layers.{d}.loop_down", .{i}, .f32, &.{ 4, 2, h, c.loop.gate_rank }, .controller_down);
            try put(sink, "model.layers.{d}.loop_gain_forget", .{i}, .f32, &.{ 4, c.loop.gate_rank, 2 * h }, .controller_gain);
            try put(sink, "model.layers.{d}.loop_message.query", .{i}, .f32, &.{ 4, h }, .cross_pass_query);
        }
        inline for (.{ "w13", "w2" }, 0..) |w, j| {
            const rows: u64 = if (j == 0) 2 * inter else h;
            const cols: u64 = if (j == 0) h else inter;
            try put(sink, "model.layers.{d}.mlp.experts." ++ w ++ ".weight", .{i}, .u32, &.{ e, rows, cols / 8 }, if (j == 0) .expert_w13 else .expert_w2);
            if (i < 20) {
                try put(sink, "model.layers.{d}.mlp.experts." ++ w ++ ".scales", .{i}, .u8, &.{ e, rows, cols / 16 }, if (j == 0) .expert_w13_scale else .expert_w2_scale);
                try put(sink, "model.layers.{d}.mlp.experts." ++ w ++ ".global_scale", .{i}, .f32, &.{e}, if (j == 0) .expert_w13_amax else .expert_w2_amax);
            } else {
                try put(sink, "model.layers.{d}.mlp.experts." ++ w ++ ".scales", .{i}, .bf16, &.{ e, rows, cols / 64 }, if (j == 0) .expert_w13_scale else .expert_w2_scale);
                try put(sink, "model.layers.{d}.mlp.experts." ++ w ++ ".biases", .{i}, .bf16, &.{ e, rows, cols / 64 }, if (j == 0) .expert_w13_scale else .expert_w2_scale);
            }
        }
    }
}

fn backbone(c: *const config.Config, prec: []const Precision, sink: anytype) !void {
    const h: u64 = c.text.hidden;
    try put(sink, "model.llm.embed.weight", .{}, .bf16, &.{ c.text.vocab, h }, .embed);
    try put(sink, "model.llm.embed_norm.weight", .{}, .bf16, &.{h}, .embed_norm);
    try put(sink, "model.llm.norm.weight", .{}, .bf16, &.{h}, .final_norm);
    try put(sink, "model.llm.unembed.weight", .{}, .bf16, &.{ c.text.vocab, h }, .unembed);
    try put(sink, "model.llm.attn_res.output_query", .{}, .f32, &.{h}, .attn_res_output_query);
    try put(sink, "model.llm.middle_cycle_cross_pass_queries", .{}, .f32, &.{ 2 * (c.loop.core_end - c.loop.core_start), h }, .cross_pass_query);
    try put(sink, "model.llm.middle_cycle_cross_pass_kv_queries", .{}, .f32, &.{ 2 * (c.loop.core_end - c.loop.core_start), h }, .cross_pass_kv_query);
    for (0..c.text.layers) |k| {
        const i: u32 = @intCast(k);
        try put(sink, "model.llm.attn_res.mlp_queries.{d}", .{i}, .f32, &.{h}, .attn_res_mlp_query);
        if (i + 1 < c.text.layers) {
            const dims: []const u64 = if (c.isFull(i + 1)) &.{ 3, h } else &.{h};
            try put(sink, "model.llm.attn_res.attn_queries.{d}", .{i}, .f32, dims, .attn_res_attn_query);
        }
        try attention(c, i, sink);
        try put(sink, "model.llm.layers.{d}.attn_sconv.weight", .{i}, .bf16, &.{ h, 1, config.conv_taps }, .sconv_post);
        try put(sink, "model.llm.layers.{d}.mlp_norm.weight", .{i}, .bf16, &.{h}, .input_norm);
        try put(sink, "model.llm.layers.{d}.mlp_sconv.weight", .{i}, .bf16, &.{ h, 1, config.conv_taps }, .sconv_post);
        try experts(c, i, prec[i], sink);
        try controllers(c, i, sink);
    }
}

fn vision(v: *const @import("inkling2_config").Vision, sink: anytype) !void {
    const h: u64 = v.hidden;
    const ff: u64 = h * v.mlp_ratio;
    try put(sink, "model.visual.backbone.embeddings.class_embedding", .{}, .bf16, &.{h}, .vision_embedding);
    try put(sink, "model.visual.backbone.embeddings.cls_position_embedding", .{}, .bf16, &.{h}, .vision_embedding);
    try put(sink, "model.visual.backbone.embeddings.patch_embedding.weight", .{}, .bf16, &.{ h, 3 * @as(u64, v.patch) * v.patch }, .vision_embedding);
    try put(sink, "model.visual.backbone.embeddings.position_embedding", .{}, .bf16, &.{ @as(u64, v.grid) * v.grid, h }, .vision_embedding);
    inline for (.{ "weight", "bias" }) |p| try put(sink, "model.visual.backbone.ln_pre." ++ p, .{}, .bf16, &.{h}, .vision_norm);
    for (0..v.layers) |k| {
        const L = .{k};
        const pre = "model.visual.backbone.layers.{d}.";
        inline for (.{ "ln_1", "ln_2" }) |n| inline for (.{ "weight", "bias" }) |p| try put(sink, pre ++ n ++ "." ++ p, L, .bf16, &.{h}, .vision_norm);
        inline for (.{ "ls_1", "ls_2" }) |n| try put(sink, pre ++ n ++ ".gamma", L, .bf16, &.{h}, .vision_norm);
        inline for (.{ "q_proj", "k_proj", "v_proj", "out_proj" }) |n| {
            try put(sink, pre ++ "self_attn." ++ n ++ ".weight", L, .bf16, &.{ h, h }, .vision_attention);
            try put(sink, pre ++ "self_attn." ++ n ++ ".bias", L, .bf16, &.{h}, .vision_attention);
        }
        try put(sink, pre ++ "mlp.c_fc.weight", L, .bf16, &.{ ff, h }, .vision_mlp);
        try put(sink, pre ++ "mlp.c_fc.bias", L, .bf16, &.{ff}, .vision_mlp);
        try put(sink, pre ++ "mlp.c_proj.weight", L, .bf16, &.{ h, ff }, .vision_mlp);
        try put(sink, pre ++ "mlp.c_proj.bias", L, .bf16, &.{h}, .vision_mlp);
    }
    try put(sink, "model.visual.proj_in.weight", .{}, .bf16, &.{ v.proj_hidden, h * v.merge * v.merge }, .vision_projection);
    try put(sink, "model.visual.proj_out.weight", .{}, .bf16, &.{ v.text_hidden, v.proj_hidden }, .vision_projection);
}

fn draft(m: Mtp, text: *const Text, sink: anytype) !void {
    const h: u64 = text.hidden;
    const q: u64 = @as(u64, m.q_heads) * m.head_dim;
    const kv: u64 = @as(u64, m.kv_heads) * m.head_dim;
    const rank: u64 = m.rank;
    try put(sink, "fuse_weight", .{}, .bf16, &.{ m.taps_n, m.taps_n }, .draft_head);
    try put(sink, "mask_embed", .{}, .bf16, &.{h}, .draft_head);
    try put(sink, "norm.weight", .{}, .bf16, &.{h}, .draft_head);
    try put(sink, "markov_head.dflare_norm.weight", .{}, .bf16, &.{h}, .draft_head);
    try put(sink, "markov_head.hidden_out.bias", .{}, .bf16, &.{h}, .draft_head);
    try put(sink, "markov_head.hidden_out.weight", .{}, .bf16, &.{ h, rank }, .draft_head);
    try put(sink, "markov_head.markov_w1.weight", .{}, .bf16, &.{ text.vocab, rank }, .draft_head);
    try put(sink, "markov_head.rnn_proj.bias", .{}, .bf16, &.{3 * rank}, .draft_head);
    try put(sink, "markov_head.rnn_proj.weight", .{}, .bf16, &.{ 3 * rank, 2 * h + 2 * rank }, .draft_head);
    for (0..m.layers) |k| {
        const L = .{k};
        const pre = "layers.{d}.";
        inline for (.{ "dflare_norm", "input_layernorm", "post_attention_layernorm" }) |n| try put(sink, pre ++ n ++ ".weight", L, .bf16, &.{h}, .draft_head);
        inline for (.{ "k_norm", "k_norm_ctx", "q_norm" }) |n| try put(sink, pre ++ "self_attn." ++ n ++ ".weight", L, .bf16, &.{m.head_dim}, .draft_head);
        inline for (.{ "k_proj", "k_proj_ctx", "v_proj", "v_proj_ctx" }) |n| try put(sink, pre ++ "self_attn." ++ n ++ ".weight", L, .bf16, &.{ kv, h }, .draft_head);
        try put(sink, pre ++ "self_attn.q_proj.weight", L, .bf16, &.{ q, h }, .draft_head);
        try put(sink, pre ++ "self_attn.o_proj.weight", L, .bf16, &.{ h, q }, .draft_head);
        try put(sink, pre ++ "mlp.gate_proj.weight", L, .bf16, &.{ m.inter, h }, .draft_head);
        try put(sink, pre ++ "mlp.up_proj.weight", L, .bf16, &.{ m.inter, h }, .draft_head);
        try put(sink, pre ++ "mlp.down_proj.weight", L, .bf16, &.{ h, m.inter }, .draft_head);
    }
}

/// Every tensor of `file` the config implies, as sink.put(name, dtype, dims, role).
pub fn all(c: *const config.Config, prec: []const Precision, file: File, sink: anytype) !void {
    switch (file) {
        .model => {
            if (c.checkpoint == .mlx) {
                try mlxBackbone(c, sink);
            } else {
                try backbone(c, prec, sink);
                try vision(&c.vision, sink);
            }
        },
        .mtp => try draft(c.mtp orelse return error.NoDraftHead, &c.text, sink),
    }
}

/// Each layer's expert precision from its w13 tensor: packed U8 is NVFP4, FP8 is MXFP8, anything else is refused.
pub fn precisions(entries: st.Header, layers: u32, out: []Precision) !void {
    if (out.len < layers) return error.PrecisionCount;
    for (0..layers) |i| {
        var buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.mlp.experts.w13_weight", .{i});
        const hit = entries.get(name) orelse return error.NoExperts;
        out[i] = switch (hit.dtype) {
            .u8 => .nvfp4,
            .f8_e4m3 => .mxfp8,
            else => return error.ExpertFormat,
        };
    }
}

pub fn mxfp8Value(weight: u8, scale: u8) f64 {
    const exponent: f64 = @floatFromInt(@as(i32, scale) - 127);
    return core.quant.fp8.e4m3(weight) * std.math.pow(f64, 2.0, exponent);
}

pub const mxfp8_group = 32;

fn get(entries: st.Header, name: []const u8, why: *config.Why) !st.Entry {
    return entries.get(name) orelse {
        why.set("{s}: missing", .{name});
        return error.BadGeometry;
    };
}

fn same(e: st.Entry, dtype: DType, dims: []const u64) bool {
    if (e.dtype != dtype or e.rank != dims.len) return false;
    for (dims, e.shape[0..@as(usize, e.rank)]) |want, got| if (want != got) return false;
    return true;
}

fn checkNvfp4(entries: st.Header, prefix: []const u8, n: u64, k: u64, why: *config.Why) !void {
    var wb: [160]u8 = undefined;
    var sb: [160]u8 = undefined;
    var db: [160]u8 = undefined;
    var gb: [160]u8 = undefined;
    const wn = try std.fmt.bufPrint(&wb, "{s}_weight", .{prefix});
    const sn = try std.fmt.bufPrint(&sb, "{s}_weight_scale", .{prefix});
    const dn = try std.fmt.bufPrint(&db, "{s}_weight_amax", .{prefix});
    const gn = try std.fmt.bufPrint(&gb, "{s}_weight_outer_scale_denominator", .{prefix});
    const w = try get(entries, wn, why);
    const s = try get(entries, sn, why);
    const d = try get(entries, dn, why);
    const g = try get(entries, gn, why);
    const dims = core.nvfp4.dims(
        .{ .dtype = w.dtype, .shape = w.shape[1..@as(usize, w.rank)] },
        .{ .dtype = s.dtype, .shape = s.shape[1..@as(usize, s.rank)] },
        .{ .dtype = g.dtype, .shape = g.shape[0..@as(usize, g.rank)] },
        null,
    ) catch {
        why.set("{s}: not ModelOpt NVFP4 (codes U8 [E,N,K/2], scales F8_E4M3 [E,N,K/16], a scalar denominator)", .{prefix});
        return error.BadGeometry;
    };
    const stack: u64 = @intCast(w.shape[0]);
    if (d.dtype != .f32 or d.rank != 1 or d.shape[0] != stack or dims.n != n or dims.k != k) {
        why.set("{s}: NVFP4 geometry is [{d},{d},{d}] codes and [{d},{d}] scales, expected [E,{d},{d}] and [E,{d},{d}]", .{ prefix, stack, dims.n, dims.k / 2, stack, dims.k / core.nvfp4.group, n, k, n, k / core.nvfp4.group });
        return error.BadGeometry;
    }
}

fn checkMxfp8(entries: st.Header, prefix: []const u8, n: u64, k: u64, why: *config.Why) !void {
    var wb: [160]u8 = undefined;
    var sb: [160]u8 = undefined;
    const wn = try std.fmt.bufPrint(&wb, "{s}_weight", .{prefix});
    const sn = try std.fmt.bufPrint(&sb, "{s}_weight_scale", .{prefix});
    const w = try get(entries, wn, why);
    const s = try get(entries, sn, why);
    const stack: u64 = @intCast(w.shape[0]);
    if (!same(w, .f8_e4m3, &.{ stack, n, k }) or !same(s, .u8, &.{ stack, n, k / mxfp8_group })) {
        why.set("{s}: MXFP8 geometry is [E,{d},{d}] e4m3 codes and [E,{d},{d}] ue8m0 scales", .{ prefix, n, k, n, k / mxfp8_group });
        return error.BadGeometry;
    }
}

/// Every dotted expert layer's NVFP4 or MXFP8 geometry; a corrupted shape or scale is refused by tensor name.
pub fn geometry(c: *const config.Config, entries: st.Header, why: *config.Why) !void {
    if (c.checkpoint == .mlx) {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const report = try check(arena.allocator(), c, entries, .model);
        if (report.missing.items.len != 0) {
            why.set("{s}: missing", .{report.missing.items[0]});
            return error.BadGeometry;
        }
        if (report.wrong.items.len != 0) {
            why.set("{s}: wrong dtype or shape", .{report.wrong.items[0]});
            return error.BadGeometry;
        }
        return;
    }
    var prec: [config.max_layers]Precision = undefined;
    precisions(entries, c.text.layers, &prec) catch |e| {
        why.set("expert tensors do not say NVFP4 (U8) or MXFP8 (F8_E4M3): {s}", .{@errorName(e)});
        return e;
    };
    const h: u64 = c.text.hidden;
    const m: u64 = c.text.expert_inter;
    for (0..c.text.layers) |i| {
        var w13: [96]u8 = undefined;
        var w2: [96]u8 = undefined;
        const p13 = try std.fmt.bufPrint(&w13, "model.llm.layers.{d}.mlp.experts.w13", .{i});
        const p2 = try std.fmt.bufPrint(&w2, "model.llm.layers.{d}.mlp.experts.w2", .{i});
        switch (prec[i]) {
            .nvfp4 => {
                try checkNvfp4(entries, p13, 2 * m, h, why);
                try checkNvfp4(entries, p2, h, m, why);
            },
            .mxfp8 => {
                try checkMxfp8(entries, p13, 2 * m, h, why);
                try checkMxfp8(entries, p2, h, m, why);
            },
            .affine4, .affine8 => return error.Unsupported,
        }
    }
}

pub const Report = struct {
    expected: usize = 0,
    missing: std.ArrayList([]const u8) = .empty,
    wrong: std.ArrayList([]const u8) = .empty,
    unknown: std.ArrayList([]const u8) = .empty,

    pub fn clean(r: *const Report) bool {
        return r.missing.items.len + r.wrong.items.len + r.unknown.items.len == 0;
    }
};

const Checker = struct {
    arena: std.mem.Allocator,
    entries: st.Header,
    seen: []bool,
    report: *Report,

    pub fn put(k: *Checker, name: []const u8, dtype: DType, dims: []const u64, role: Role) !void {
        _ = role;
        k.report.expected += 1;
        const at = k.entries.getIndex(name) orelse return k.report.missing.append(k.arena, try k.arena.dupe(u8, name));
        if (k.seen[at]) return error.DuplicateName;
        k.seen[at] = true;
        const e = k.entries.values()[at];
        var ok = e.dtype == dtype and e.rank == dims.len;
        if (ok) for (dims, e.shape[0..e.rank]) |want, got| {
            if (want != got) ok = false;
        };
        if (!ok) try k.report.wrong.append(k.arena, try k.arena.dupe(u8, name));
    }
};

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// `entries` against the map of `file`; expert precisions come from the entries themselves (lists live in `arena`).
pub fn check(arena: std.mem.Allocator, c: *const config.Config, entries: st.Header, file: File) !Report {
    var prec: [config.max_layers]Precision = undefined;
    if (file == .model and c.checkpoint == .modelopt) try precisions(entries, c.text.layers, &prec);
    const seen = try arena.alloc(bool, entries.count());
    @memset(seen, false);
    var r: Report = .{};
    var k: Checker = .{ .arena = arena, .entries = entries, .seen = seen, .report = &r };
    try all(c, &prec, file, &k);
    for (entries.keys(), seen) |name, was| if (!was and !(c.checkpoint == .mlx and std.mem.startsWith(u8, name, "vision_tower."))) try r.unknown.append(arena, name);
    std.mem.sort([]const u8, r.unknown.items, {}, lessThan);
    return r;
}

const Roles = struct {
    arena: std.mem.Allocator,
    map: std.StringHashMapUnmanaged(Role) = .empty,

    pub fn put(r: *Roles, name: []const u8, dtype: DType, dims: []const u64, role: Role) !void {
        _ = dtype;
        _ = dims;
        try r.map.put(r.arena, try r.arena.dupe(u8, name), role);
    }
};

/// Every mapped tensor's role by name, for checking a checkpoint index before its headers are read.
pub fn roles(arena: std.mem.Allocator, c: *const config.Config, file: File) !std.StringHashMapUnmanaged(Role) {
    var prec: [config.max_layers]Precision = undefined;
    if (file == .model) @memset(prec[0..c.text.layers], .nvfp4);
    var r: Roles = .{ .arena = arena };
    try all(c, &prec, file, &r);
    return r.map;
}
