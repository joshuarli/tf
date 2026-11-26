const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const st = core.safetensors;
const cfg = @import("../config.zig");
const kernels = @import("../kernels.zig");
const fmt = @import("format.zig");
const ba = @import("binary_archive.zig");

const rows_metal = @embedFile("../kernels/rows.metal");
const experts_metal = @embedFile("../kernels/experts.metal");
const verify_metal = @embedFile("../kernels/verify.metal");
const prefill_rows_metal = @embedFile("../kernels/prefill_rows.metal");
const taps_metal = @embedFile("../kernels/taps.metal");

const decode_pso_names = [_][:0]const u8{
    "i2a_embed",
    "i2a_batch_projections",
    "i2a_dense",
    "i2a_float",
    "i2a_batch_float_control",
    "i2a_expert_up",
    "i2a_expert_down",
    "i2_batch_qkv",
    "i2_batch_split",
    "i2_batch_combine",
    "i2_controls_norm",
    "i2_controls_down",
    "i2_controls_apply",
    "i2_controls_apply_rows",
    "i2v_argmax",
    "i2_copy",
    "i2_sig_mul",
    "i2_add",
    "i2p_pass_copy",
    "i2p_rms",
    "i2p_rms3",
    "i2p_rms_pair",
    "i2p_batch_conv",
    "i2p_scatter",
    "i2p_routepick",
    "i2p_gu",
    "i2pr_batch_mix",
    "i2pr_batch_mix2",
    "i2e_down1",
    "i2r_controls_down",
    "i2t_tap",
};

const Shard = struct {
    map: mtl.MappedFile,
    data: usize,
    header: st.Header,
};

const WeightWriter = struct {
    file: std.Io.File,
    io: std.Io,
    hasher: std.crypto.hash.sha2.Sha256,
    pos: u64 = 0,
    zeros: [16384]u8 = @splat(0),

    fn init(io: std.Io, path: []const u8) !WeightWriter {
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        return .{ .file = file, .io = io, .hasher = std.crypto.hash.sha2.Sha256.init(.{}) };
    }

    fn deinit(w: *WeightWriter) void {
        w.file.close(w.io);
    }

    fn padTo(w: *WeightWriter, alignment: u64) !void {
        const target = fmt.alignUp(w.pos, alignment);
        var rem = target - w.pos;
        while (rem > 0) {
            const take: usize = @intCast(@min(rem, w.zeros.len));
            try w.writeRaw(w.zeros[0..take]);
            rem -= take;
        }
    }

    fn writeRaw(w: *WeightWriter, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try w.file.writeStreamingAll(w.io, bytes);
        w.hasher.update(bytes);
        w.pos += bytes.len;
    }

    fn append(w: *WeightWriter, bytes: []const u8, alignment: u64) !fmt.TensorRef {
        try w.padTo(alignment);
        const off = w.pos;
        try w.writeRaw(bytes);
        return .{ .off = off, .bytes = bytes.len };
    }

    fn finish(w: *WeightWriter) ![32]u8 {
        try w.padTo(fmt.page_align);
        return w.hasher.finalResult();
    }
};

fn findTensor(shards: []const Shard, name: []const u8) !struct { bytes: []const u8, entry: st.Entry } {
    for (shards) |s| {
        if (s.header.get(name)) |e| {
            return .{ .bytes = s.map.bytes[s.data + e.begin .. s.data + e.end], .entry = e };
        }
    }
    std.log.err("aotc: missing tensor {s}", .{name});
    return error.MissingTensor;
}

fn copyTensor(w: *WeightWriter, shards: []const Shard, name: []const u8, align_bytes: u64) !fmt.TensorRef {
    const t = try findTensor(shards, name);
    return w.append(t.bytes, align_bytes);
}

fn interleaveSB(gpa: std.mem.Allocator, scales: []const u8, biases: []const u8) ![]u8 {
    if (scales.len != biases.len or scales.len % 2 != 0) return error.BadShape;
    const n = scales.len / 2;
    const out = try gpa.alloc(u8, scales.len * 2);
    for (0..n) |i| {
        out[i * 4 + 0] = scales[i * 2 + 0];
        out[i * 4 + 1] = scales[i * 2 + 1];
        out[i * 4 + 2] = biases[i * 2 + 0];
        out[i * 4 + 3] = biases[i * 2 + 1];
    }
    return out;
}

fn packAffine(gpa: std.mem.Allocator, w: *WeightWriter, shards: []const Shard, base: []const u8, rows: u32, cols: u32, bits: u32, trim_rows: ?u32, interleave: bool) !fmt.AffineRef {
    var nb: [192]u8 = undefined;
    const wt = try findTensor(shards, try std.fmt.bufPrint(&nb, "{s}.weight", .{base}));
    const st_ = try findTensor(shards, try std.fmt.bufPrint(&nb, "{s}.scales", .{base}));
    const bt = try findTensor(shards, try std.fmt.bufPrint(&nb, "{s}.biases", .{base}));
    const use_rows: u32 = trim_rows orelse rows;
    const w_row_bytes: usize = @as(usize, cols) * bits / 8;
    const sb_row_bytes: usize = @as(usize, cols) / 64 * 2;
    const w_bytes = wt.bytes[0 .. @as(usize, use_rows) * w_row_bytes];
    const s_bytes = st_.bytes[0 .. @as(usize, use_rows) * sb_row_bytes];
    const b_bytes = bt.bytes[0 .. @as(usize, use_rows) * sb_row_bytes];

    const w_ref = try w.append(w_bytes, 256);
    var ref = fmt.AffineRef{
        .w = w_ref,
        .rows = use_rows,
        .cols = cols,
        .bits = bits,
    };
    if (interleave) {
        const packed_sb = try interleaveSB(gpa, s_bytes, b_bytes);
        defer gpa.free(packed_sb);
        ref.sb = try w.append(packed_sb, 256);
    }
    ref.s = try w.append(s_bytes, 256);
    ref.b = try w.append(b_bytes, 256);
    return ref;
}

fn sha256File(io: std.Io, path: []const u8) !struct { size: u64, digest: [32]u8 } {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [1 << 20]u8 = undefined;
    var total: u64 = 0;
    while (true) {
        const n = file.readStreaming(io, &.{&buf}) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        if (n == 0) break;
        hasher.update(buf[0..n]);
        total += n;
    }
    return .{ .size = total, .digest = hasher.finalResult() };
}

fn runCmd(gpa: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
    const res = try std.process.run(gpa, io, .{ .argv = argv, .stdout_limit = .limited(4 << 20), .stderr_limit = .limited(4 << 20) });
    defer gpa.free(res.stdout);
    defer gpa.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code == 0) return else {
            std.log.err("aotc command failed ({d}): {s}", .{ code, res.stderr });
            return error.CompilerFailed;
        },
        else => {
            std.log.err("aotc command terminated abnormally: {s}", .{res.stderr});
            return error.CompilerFailed;
        },
    }
}

fn compileMetallib(gpa: std.mem.Allocator, io: std.Io, out_dir: []const u8, extra_metal: ?[]const u8) ![]const u8 {
    const tmp_dir = try std.fs.path.join(gpa, &.{ out_dir, ".aotc_air" });
    defer gpa.free(tmp_dir);
    try std.Io.Dir.cwd().createDirPath(io, tmp_dir);
    defer std.Io.Dir.cwd().deleteTree(io, tmp_dir) catch {};

    const Unit = struct { name: []const u8, src: []const u8 };
    const attn_verify = try std.fmt.allocPrint(gpa, "{s}\n{s}\n", .{ kernels.attention, verify_metal });
    defer gpa.free(attn_verify);
    const rows_exp = try std.fmt.allocPrint(gpa, "{s}\n#define I2T_ROUTED 1\n{s}\n", .{ rows_metal, experts_metal });
    defer gpa.free(rows_exp);

    var units: std.ArrayList(Unit) = .empty;
    defer units.deinit(gpa);
    try units.appendSlice(gpa, &.{
        .{ .name = "affine", .src = kernels.affine },
        .{ .name = "attention_verify", .src = attn_verify },
        .{ .name = "glue", .src = kernels.glue },
        .{ .name = "prefill", .src = kernels.prefill },
        .{ .name = "prefill_rows", .src = prefill_rows_metal },
        .{ .name = "rows_experts", .src = rows_exp },
        .{ .name = "taps", .src = taps_metal },
    });
    if (extra_metal) |src| try units.append(gpa, .{ .name = "aot_fused", .src = src });

    var airs: std.ArrayList([]const u8) = .empty;
    defer {
        for (airs.items) |p| gpa.free(p);
        airs.deinit(gpa);
    }

    for (units.items) |u| {
        const mpath = try std.fmt.allocPrint(gpa, "{s}/{s}.metal", .{ tmp_dir, u.name });
        defer gpa.free(mpath);
        const apath = try std.fmt.allocPrint(gpa, "{s}/{s}.air", .{ tmp_dir, u.name });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = mpath, .data = u.src });
        try runCmd(gpa, io, &.{ "xcrun", "-sdk", "macosx", "metal", "-c", "-std=metal4.1", "-fmetal-math-mode=safe", "-O3", mpath, "-o", apath });
        try airs.append(gpa, apath);
    }

    const metallib_path = try std.fs.path.join(gpa, &.{ out_dir, "kernels.metallib" });
    var link_argv: std.ArrayList([]const u8) = .empty;
    defer link_argv.deinit(gpa);
    try link_argv.appendSlice(gpa, &.{ "xcrun", "-sdk", "macosx", "metallib" });
    try link_argv.appendSlice(gpa, airs.items);
    try link_argv.appendSlice(gpa, &.{ "-o", metallib_path });
    try runCmd(gpa, io, link_argv.items);
    return metallib_path;
}

fn copyFileIfPresent(io: std.Io, src: []const u8, dst: []const u8) !bool {
    const in_file = std.Io.Dir.cwd().openFile(io, src, .{}) catch return false;
    defer in_file.close(io);
    const out_file = try std.Io.Dir.cwd().createFile(io, dst, .{});
    defer out_file.close(io);
    var buf: [1 << 20]u8 = undefined;
    while (true) {
        const n = in_file.readStreaming(io, &.{&buf}) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        if (n == 0) break;
        try out_file.writeStreamingAll(io, buf[0..n]);
    }
    return true;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3) {
        std.debug.print("usage: tf-inkling2-aotc <checkpoint-dir> <out-dir> [--repack=none|phase_c] [--runner-bin PATH]\n", .{});
        return error.ExpectedArgs;
    }
    const ckpt_dir = args[1];
    const out_dir = args[2];
    var repack_c = false;
    var runner_bin: ?[]const u8 = null;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--repack=phase_c")) {
            repack_c = true;
        } else if (std.mem.eql(u8, args[i], "--repack=none")) {
            repack_c = false;
        } else if (std.mem.eql(u8, args[i], "--runner-bin") and i + 1 < args.len) {
            i += 1;
            runner_bin = args[i];
        } else return error.UnknownArg;
    }

    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try std.Io.Dir.cwd().createDirPath(io, out_dir);

    const cfg_path = try std.fs.path.join(arena, &.{ ckpt_dir, "config.json" });
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, arena, .limited(16 << 20));
    var why: cfg.Why = .{};
    const model_cfg = try cfg.parse(arena, cfg_text, &why);
    if (model_cfg.checkpoint != .mlx) return error.UnsupportedCheckpoint;

    const shard_names = try cfg.shards(arena, io, ckpt_dir);
    var shards = try arena.alloc(Shard, shard_names.len);
    var mapped: usize = 0;
    defer for (shards[0..mapped]) |s| s.map.deinit();
    for (shard_names, 0..) |name, si| {
        const path = try std.fs.path.joinZ(arena, &.{ ckpt_dir, name });
        const map = try mtl.MappedFile.open(path.ptr);
        const hdr_size = try st.headerSize(std.mem.readInt(u64, map.bytes[0..8], .little), map.size);
        const data = 8 + hdr_size;
        const header = try st.parseHeaderPrefixes(arena, map.bytes[8..data], map.size - data, &.{ "model.", "lm_head." });
        shards[si] = .{ .map = map, .data = data, .header = header };
        mapped += 1;
    }

    const weights_path = try std.fs.path.join(arena, &.{ out_dir, "weights.aot" });
    var ww = try WeightWriter.init(io, weights_path);
    defer ww.deinit();

    var sched = fmt.Schedule{};
    if (repack_c) {
        sched.layout_flags = fmt.layout_interleaved_sb | fmt.layout_predivided_nvfp4 | fmt.layout_trimmed_lm_head | fmt.layout_colocated_nvfp4;
    }
    const h: u32 = model_cfg.text.hidden;
    const rank: u32 = model_cfg.loop.gate_rank;
    const inter: u32 = model_cfg.text.expert_inter;
    const E: u32 = model_cfg.text.experts;

    sched.globals.embed = try packAffine(gpa, &ww, shards, "model.embed", model_cfg.text.vocab, h, 8, null, repack_c);
    sched.globals.embed_norm = try copyTensor(&ww, shards, "model.embed_norm.weight", 256);
    sched.globals.norm = try copyTensor(&ww, shards, "model.norm.weight", 256);
    sched.globals.unembed = try packAffine(gpa, &ww, shards, "lm_head", model_cfg.text.vocab, h, 8, if (repack_c) model_cfg.text.unpadded_vocab else null, repack_c);
    sched.globals.output_query = try copyTensor(&ww, shards, "model.final_input.query", 256);

    const cp_q = try gpa.alloc(u8, 2 * fmt.num_core_layers * h * 4);
    defer gpa.free(cp_q);
    const cp_kv = try gpa.alloc(u8, 2 * fmt.num_core_layers * h * 4);
    defer gpa.free(cp_kv);
    for (0..fmt.num_core_layers) |ci| {
        var nb: [128]u8 = undefined;
        const t = try findTensor(shards, try std.fmt.bufPrint(&nb, "model.layers.{d}.loop_message.query", .{ci + fmt.core_start}));
        @memcpy(cp_q[(2 * ci) * h * 4 ..][0 .. h * 4], t.bytes[0 .. h * 4]);
        @memcpy(cp_kv[(2 * ci) * h * 4 ..][0 .. 2 * h * 4], t.bytes[h * 4 ..][0 .. 2 * h * 4]);
        @memcpy(cp_q[(2 * ci + 1) * h * 4 ..][0 .. h * 4], t.bytes[3 * h * 4 ..][0 .. h * 4]);
    }
    sched.globals.cross_pass_q = try ww.append(cp_q, 256);
    sched.globals.cross_pass_kv = try ww.append(cp_kv, 256);

    for (0..fmt.num_layers) |li| {
        var nb: [128]u8 = undefined;
        sched.globals.mlp_queries[li] = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp_input.query", .{li}), 256);
        if (li > 0) {
            sched.globals.attn_queries[li - 1] = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "model.layers.{d}.attn_input.query", .{li}), 256);
        }
    }

    var t16_vals: [16]f32 = undefined;
    for (0..16) |k| t16_vals[k] = @floatCast(core.quant.nvfp4.e2m1(@intCast(k)));
    sched.globals.t16 = try ww.append(std.mem.sliceAsBytes(&t16_vals), 256);
    var t256_vals: [256]f32 = undefined;
    for (0..256) |k| t256_vals[k] = @floatCast(core.quant.fp8.e4m3(@intCast(k)));
    sched.globals.t256 = try ww.append(std.mem.sliceAsBytes(&t256_vals), 256);
    const den_val = [1]f32{2688.0};
    sched.globals.nvfp4_den = try ww.append(std.mem.sliceAsBytes(&den_val), 256);

    const ctl_down_buf = try gpa.alloc(f32, rank * 2 * h);
    defer gpa.free(ctl_down_buf);
    const ctl_gain_buf = try gpa.alloc(f32, h * rank);
    defer gpa.free(ctl_gain_buf);
    const ctl_forget_buf = try gpa.alloc(f32, h * rank);
    defer gpa.free(ctl_forget_buf);
    const prediv_buf = try gpa.alloc(f32, E);
    defer gpa.free(prediv_buf);

    for (0..fmt.num_layers) |li| {
        const layer_idx: u32 = @intCast(li);
        const full = model_cfg.isFull(layer_idx);
        const hd = model_cfg.heads(layer_idx);
        const qw: u32 = hd.q * hd.dim;
        const kvw: u32 = hd.kv * hd.dim;
        const rel: u32 = hd.q * model_cfg.text.d_rel;
        var lp = fmt.LayerPlan{
            .is_full = @intFromBool(full),
            .is_affine4 = @intFromBool(li >= 20),
            .has_controllers = @intFromBool(li >= fmt.core_start and li < fmt.core_end),
            .extent = model_cfg.extent(layer_idx),
        };
        var pbuf: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pbuf, "model.layers.{d}.", .{li});
        var nb: [160]u8 = undefined;

        if (full) {
            lp.q_norm = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}q_input_norm.weight", .{p}), 256);
            lp.k_norm = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}k_input_norm.weight", .{p}), 256);
            lp.v_norm = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}v_input_norm.weight", .{p}), 256);
            const q_rows = qw + rel + qw + hd.q;
            const kv_rows = kvw + hd.kv * 4;
            lp.proj_q = try packAffine(gpa, &ww, shards, try std.fmt.bufPrint(&nb, "{s}attn.q_proj", .{p}), q_rows, h, 8, null, repack_c);
            lp.proj_k = try packAffine(gpa, &ww, shards, try std.fmt.bufPrint(&nb, "{s}attn.k_proj", .{p}), kv_rows, h, 8, null, repack_c);
            lp.proj_v = try packAffine(gpa, &ww, shards, try std.fmt.bufPrint(&nb, "{s}attn.v_proj", .{p}), kv_rows, h, 8, null, repack_c);
            lp.proj_ms = .{ qw, kvw, kvw, rel, qw, hd.q, hd.kv * 4, hd.kv * 4 };
        } else {
            lp.q_norm = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}attn_norm.weight", .{p}), 256);
            lp.k_norm = lp.q_norm;
            lp.v_norm = lp.q_norm;
            const in_rows = qw + 2 * kvw + rel + qw + hd.q;
            lp.proj_q = try packAffine(gpa, &ww, shards, try std.fmt.bufPrint(&nb, "{s}attn.in_proj", .{p}), in_rows, h, 8, null, repack_c);
            lp.proj_k = lp.proj_q;
            lp.proj_v = lp.proj_q;
            lp.proj_ms = .{ qw, kvw, kvw, rel, qw, hd.q, 0, 0 };
        }

        lp.proj_o = try packAffine(gpa, &ww, shards, try std.fmt.bufPrint(&nb, "{s}attn.o_proj", .{p}), h, qw, 8, null, repack_c);
        lp.q_scale = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}attn.q_scale", .{p}), 256);
        lp.r_scale = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}attn.r_scale", .{p}), 256);
        lp.rel_bias = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}attn.rel_proj", .{p}), 256);
        lp.k_conv = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}attn.{s}", .{ p, if (full) "k_static" else "k_conv" }), 256);
        lp.v_conv = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}attn.{s}", .{ p, if (full) "v_static" else "v_conv" }), 256);
        lp.attn_post = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}attn_conv", .{p}), 256);
        lp.mlp_norm = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}mlp_norm.weight", .{p}), 256);
        lp.mlp_post = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}mlp_conv", .{p}), 256);
        lp.router = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}mlp.router_weight", .{p}), 256);
        lp.router_bias = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}mlp.router_bias", .{p}), 256);

        lp.w13 = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}mlp.experts.w13.weight", .{p}), 4096);
        lp.w13_scale = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}mlp.experts.w13.scales", .{p}), 256);
        if (li < 20) {
            const g13 = try findTensor(shards, try std.fmt.bufPrint(&nb, "{s}mlp.experts.w13.global_scale", .{p}));
            if (repack_c) {
                const src = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(g13.bytes)));
                for (src, 0..) |v, ei| prediv_buf[ei] = v / 2688.0;
                lp.w13_global = try ww.append(std.mem.sliceAsBytes(prediv_buf), 256);
            } else {
                lp.w13_global = try ww.append(g13.bytes, 256);
            }
        } else {
            const b13 = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}mlp.experts.w13.biases", .{p}), 256);
            lp.w13_bias = b13;
        }

        lp.w2 = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}mlp.experts.w2.weight", .{p}), 4096);
        lp.w2_scale = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}mlp.experts.w2.scales", .{p}), 256);
        if (li < 20) {
            const g2 = try findTensor(shards, try std.fmt.bufPrint(&nb, "{s}mlp.experts.w2.global_scale", .{p}));
            if (repack_c) {
                const src = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(g2.bytes)));
                for (src, 0..) |v, ei| prediv_buf[ei] = v / 2688.0;
                lp.w2_global = try ww.append(std.mem.sliceAsBytes(prediv_buf), 256);
            } else {
                lp.w2_global = try ww.append(g2.bytes, 256);
            }
        } else {
            const b2 = try copyTensor(&ww, shards, try std.fmt.bufPrint(&nb, "{s}mlp.experts.w2.biases", .{p}), 256);
            lp.w2_bias = b2;
        }

        if (lp.has_controllers != 0) {
            const down = try findTensor(shards, try std.fmt.bufPrint(&nb, "{s}loop_down", .{p}));
            const gates = try findTensor(shards, try std.fmt.bufPrint(&nb, "{s}loop_gain_forget", .{p}));
            const ds = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(down.bytes)));
            const gs = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(gates.bytes)));
            for (0..4) |j| {
                for (0..rank) |r| for (0..2 * h) |col| {
                    ctl_down_buf[r * 2 * h + col] = ds[(j * 2 * h + col) * rank + r];
                };
                for (0..h) |col| for (0..rank) |r| {
                    ctl_gain_buf[col * rank + r] = gs[(j * rank + r) * 2 * h + col];
                    ctl_forget_buf[col * rank + r] = gs[(j * rank + r) * 2 * h + h + col];
                };
                lp.controllers[j] = .{
                    .down = try ww.append(std.mem.sliceAsBytes(ctl_down_buf), 256),
                    .gain = try ww.append(std.mem.sliceAsBytes(ctl_gain_buf), 256),
                    .forget = try ww.append(std.mem.sliceAsBytes(ctl_forget_buf), 256),
                };
            }
        }
        _ = inter;
        sched.layers[li] = lp;
    }

    sched.weights_sha256 = try ww.finish();
    sched.weights_size = ww.pos;
    sched.slots = fmt.buildSlots(&sched.layers);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var arch_buf: [32]u8 = undefined;
    const arch_name = ba.deviceArch(device, &arch_buf);
    sched.setArchName(arch_name);
    sched.min_generation = device.generation();

    const metallib_path = try compileMetallib(gpa, io, out_dir, null);
    defer gpa.free(metallib_path);

    const archive_path = try std.fs.path.join(arena, &.{ out_dir, "pipelines.mtlarchive" });
    {
        const lib = try mtl.Library.fromFile(device, metallib_path);
        defer lib.deinit();
        const archive = try ba.BinaryArchive.create(device);
        defer archive.deinit();
        for (decode_pso_names) |name| {
            try archive.addCompute(lib, name, true);
        }
        try archive.serialize(archive_path);
    }

    const sched_path = try std.fs.path.join(arena, &.{ out_dir, "schedule.bin" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = sched_path, .data = std.mem.asBytes(&sched) });

    const tok_src = try std.fs.path.join(arena, &.{ ckpt_dir, "tokenizer.json" });
    const tok_dst = try std.fs.path.join(arena, &.{ out_dir, "tokenizer.json" });
    const has_tok = try copyFileIfPresent(io, tok_src, tok_dst);

    var has_runner = false;
    const runner_dst = try std.fs.path.join(arena, &.{ out_dir, "tf-inkling2-mini-aot" });
    if (runner_bin) |rb| {
        has_runner = try copyFileIfPresent(io, rb, runner_dst);
        if (has_runner) {
            const cpath = try arena.alloc(u8, runner_dst.len + 1);
            @memcpy(cpath[0..runner_dst.len], runner_dst);
            cpath[runner_dst.len] = 0;
            _ = std.c.chmod(@ptrCast(cpath.ptr), 0o755);
        }
    }

    var entries: std.ArrayList(fmt.FileEntry) = .empty;
    defer entries.deinit(gpa);
    const fixed_names = [_][]const u8{ "schedule.bin", "kernels.metallib", "pipelines.mtlarchive" };
    for (fixed_names) |name| {
        const p = try std.fs.path.join(arena, &.{ out_dir, name });
        const h_ = try sha256File(io, p);
        const hex = fmt.hexEncode(h_.digest);
        try entries.append(gpa, .{ .name = name, .size = h_.size, .sha256 = try arena.dupe(u8, &hex) });
    }
    const w_hex = fmt.hexEncode(sched.weights_sha256);
    try entries.append(gpa, .{ .name = "weights.aot", .size = sched.weights_size, .sha256 = try arena.dupe(u8, &w_hex) });
    if (has_tok) {
        const h_ = try sha256File(io, tok_dst);
        const hex = fmt.hexEncode(h_.digest);
        try entries.append(gpa, .{ .name = "tokenizer.json", .size = h_.size, .sha256 = try arena.dupe(u8, &hex) });
    }
    if (has_runner) {
        const h_ = try sha256File(io, runner_dst);
        const hex = fmt.hexEncode(h_.digest);
        try entries.append(gpa, .{ .name = "tf-inkling2-mini-aot", .size = h_.size, .sha256 = try arena.dupe(u8, &hex) });
    }

    const manifest = fmt.Manifest{
        .target_arch = arch_name,
        .min_generation = sched.min_generation,
        .context_cap = sched.geom.context_cap,
        .layout_flags = sched.layout_flags,
        .files = entries.items,
    };
    var mw: std.Io.Writer.Allocating = .init(gpa);
    defer mw.deinit();
    try std.json.Stringify.value(manifest, .{ .whitespace = .indent_2 }, &mw.writer);
    try mw.writer.writeByte('\n');
    const manifest_path = try std.fs.path.join(arena, &.{ out_dir, "manifest.json" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = manifest_path, .data = mw.written() });

    const elapsed_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
    std.debug.print("aotc: compiled {s} -> {s} in {d:.1} ms (arch={s}, weights={d:.2} GB, repack_c={}, psos={d})\n", .{
        ckpt_dir, out_dir, elapsed_ms, arch_name, @as(f64, @floatFromInt(sched.weights_size)) / 1e9, repack_c, decode_pso_names.len,
    });
}
