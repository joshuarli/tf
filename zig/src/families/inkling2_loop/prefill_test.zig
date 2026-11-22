const std = @import("std");
const mtl = @import("metal");
const model_mod = @import("model.zig");
const state_mod = @import("state.zig");
const fwd = @import("forward.zig");
const prefill_mod = @import("prefill.zig");

const t = std.testing;

fn tinyDir(a: std.mem.Allocator) !?[]const u8 {
    if (std.c.getenv("TF_INKLING2_LOOP_TINY")) |p| return try a.dupe(u8, std.mem.span(p));
    const home = std.c.getenv("HOME") orelse return null;
    return try std.fmt.allocPrint(a, "{s}/models/tiny-Inkling2LoopForConditionalGeneration@5e761216759d", .{std.mem.span(home)});
}

fn writeShim(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, real: []const u8) !void {
    const cfg_path = try std.fs.path.join(gpa, &.{ real, "config.json" });
    defer gpa.free(cfg_path);
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20));
    defer gpa.free(cfg_text);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, cfg_text, .{});
    if (doc != .object) return error.BadConfig;
    var root = doc.object;
    var quant: std.json.ObjectMap = .empty;
    try quant.put(a, "quant_method", .{ .string = "ModelOpt" });
    try quant.put(a, "quant_algo", .{ .string = "NVFP4" });
    try root.put(a, "quantization_config", .{ .object = quant });
    const tc = root.get("text_config") orelse return error.BadConfig;
    if (tc != .object) return error.BadConfig;
    var text = tc.object;
    try text.put(a, "intermediate_size", .{ .integer = 32 });
    try text.put(a, "sconv_kernel_size", .{ .integer = 4 });
    try text.put(a, "model_max_length", .{ .integer = 1048576 });
    try text.put(a, "pre_norm_pre_affine", .{ .bool = false });
    try text.put(a, "final_norm_pre_affine", .{ .bool = false });
    try root.put(a, "text_config", .{ .object = text });
    const mtp = root.get("mtp_config") orelse return error.BadConfig;
    if (mtp != .object) return error.BadConfig;
    var mtp_o = mtp.object;
    var taps: std.json.Array = .init(a);
    for ([_]i64{ 5, 11, 13, 15 }) |v| try taps.append(.{ .integer = v });
    try mtp_o.put(a, "target_layer_ids", .{ .array = taps });
    const th = text.get("hidden_size") orelse return error.BadConfig;
    try mtp_o.put(a, "hidden_size", .{ .integer = th.integer });
    try root.put(a, "mtp_config", .{ .object = mtp_o });
    const vis = root.get("vision_config") orelse return error.BadConfig;
    if (vis != .object) return error.BadConfig;
    var vis_o = vis.object;
    try vis_o.put(a, "native_patch_size", .{ .integer = 20 });
    try vis_o.put(a, "adapter_merge_size", .{ .integer = 2 });
    try vis_o.put(a, "projector_bias", .{ .bool = false });
    try root.put(a, "vision_config", .{ .object = vis_o });
    var arch: std.json.Array = .init(a);
    try arch.append(.{ .string = "Inkling2LoopForConditionalGeneration" });
    try root.put(a, "architectures", .{ .array = arch });
    doc = .{ .object = root };
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try std.json.Stringify.value(doc, .{}, &out.writer);
    const shim_cfg = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(shim_cfg);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = shim_cfg, .data = out.written() });

    const st_path = try std.fs.path.join(gpa, &.{ real, "model.safetensors" });
    defer gpa.free(st_path);
    const whole = try std.Io.Dir.cwd().readFileAlloc(io, st_path, gpa, .limited(64 << 20));
    defer gpa.free(whole);
    if (whole.len < 8) return error.BadSafetensors;
    const n = std.mem.readInt(u64, whole[0..8], .little);
    const hdr = try std.json.parseFromSliceLeaky(std.json.Value, a, whole[8..][0..@intCast(n)], .{});
    if (hdr != .object) return error.BadSafetensors;
    var w: std.Io.Writer.Allocating = .init(gpa);
    defer w.deinit();
    var wr = &w.writer;
    try wr.writeAll("{\"weight_map\":{");
    var first = true;
    var it = hdr.object.iterator();
    while (it.next()) |kv| {
        if (!std.mem.startsWith(u8, kv.key_ptr.*, "model.llm.")) continue;
        if (!first) try wr.writeAll(",");
        first = false;
        try wr.writeAll("\"");
        try wr.writeAll(kv.key_ptr.*);
        try wr.writeAll("\":\"model.safetensors\"");
    }
    try wr.writeAll("}}");
    const shim_idx = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(shim_idx);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = shim_idx, .data = w.written() });

    const link = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(link);
    const target = try std.fs.path.join(gpa, &.{ real, "model.safetensors" });
    defer gpa.free(target);
    std.Io.Dir.cwd().deleteFile(t.io, link) catch {};
    try std.Io.Dir.symLinkAbsolute(t.io, target, link, .{});
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| {
        const d = @abs(x - y);
        if (d > m) m = d;
    }
    return m;
}

fn maxAbs(x: []const f32) f32 {
    var m: f32 = 0;
    for (x) |v| {
        const d = @abs(v);
        if (d > m) m = d;
    }
    return m;
}

fn kvRowOff(cfg: anytype, cache: anytype, slot: usize, at: usize, kvw: usize) usize {
    const linear: usize = cfg.slots() * cache.capacity * kvw * 4;
    if (cache.keys.length() == linear) return slot * cache.capacity * kvw * 4 + at * kvw * 4;
    const phys = cfg.loop.physicalLayer(cfg.text.layers, @intCast(slot));
    const win: usize = cfg.text.window;
    var base: usize = 0;
    for (0..slot) |s| {
        const p = cfg.loop.physicalLayer(cfg.text.layers, @intCast(s));
        base += (if (!cfg.isFull(p)) win else cache.capacity) * kvw * 4;
    }
    const r: usize = if (!cfg.isFull(phys)) at % win else at;
    return base + r * kvw * 4;
}

fn kvStateDiff(cfg: anytype, cache: anytype, a: []const f32, b: []const f32, len: usize) f32 {
    const kvw: usize = cfg.text.local.kv * cfg.text.local.dim;
    var m: f32 = 0;
    for (0..cfg.slots()) |s| {
        for (0..len) |j| {
            const off = kvRowOff(cfg, cache, s, j, kvw) / 4;
            for (0..kvw) |i| {
                const d = @abs(a[off + i] - b[off + i]);
                if (d > m) m = d;
            }
        }
    }
    return m;
}

const StageBufs = struct {
    bytes: usize,
    conv_bytes: usize,
    mlp_bytes: usize,
    keys: mtl.Buffer,
    values: mtl.Buffer,
    kc: mtl.Buffer,
    vc: mtl.Buffer,
    ac: mtl.Buffer,
    mc: mtl.Buffer,
};

fn snapshot(m: *const model_mod.Model, cache: *const state_mod.Cache, stage: *StageBufs) !void {
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    blit.copy(cache.keys, 0, stage.keys, 0, stage.bytes);
    blit.copy(cache.values, 0, stage.values, 0, stage.bytes);
    blit.copy(cache.k_conv, 0, stage.kc, 0, stage.conv_bytes);
    blit.copy(cache.v_conv, 0, stage.vc, 0, stage.conv_bytes);
    blit.copy(cache.attn_conv, 0, stage.ac, 0, stage.mlp_bytes);
    blit.copy(cache.mlp_conv, 0, stage.mc, 0, stage.mlp_bytes);
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("snapshot: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

fn stageInit(device: mtl.Device, kv_bytes: usize, conv_bytes: usize, mlp_bytes: usize) !StageBufs {
    return .{
        .bytes = kv_bytes,
        .conv_bytes = conv_bytes,
        .mlp_bytes = mlp_bytes,
        .keys = try device.buffer(@max(kv_bytes, 16), mtl.ResourceOptions.shared),
        .values = try device.buffer(@max(kv_bytes, 16), mtl.ResourceOptions.shared),
        .kc = try device.buffer(@max(conv_bytes, 16), mtl.ResourceOptions.shared),
        .vc = try device.buffer(@max(conv_bytes, 16), mtl.ResourceOptions.shared),
        .ac = try device.buffer(@max(mlp_bytes, 16), mtl.ResourceOptions.shared),
        .mc = try device.buffer(@max(mlp_bytes, 16), mtl.ResourceOptions.shared),
    };
}

fn stageDeinit(stage: *StageBufs) void {
    stage.keys.deinit();
    stage.values.deinit();
    stage.kc.deinit();
    stage.vc.deinit();
    stage.ac.deinit();
    stage.mc.deinit();
}

test "chunk self-consistency n=4 vs 4x1" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const gpa = t.allocator;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tinyDir(a) orelse return error.SkipZigTest;
    const tmpbase = std.mem.span(std.c.getenv("TMPDIR") orelse "/tmp");
    const shim = try std.fs.path.join(a, &.{ tmpbase, "i2-prefill-shim" });
    std.Io.Dir.cwd().createDirPath(io, shim) catch {};
    try writeShim(gpa, io, shim, dir);
    var m = try model_mod.Model.load(gpa, io, shim);
    defer m.deinit();
    const ids = [_]u32{ 100, 107, 114, 121 };
    const vocab: usize = m.config.text.unpadded_vocab;
    var cache_a = try state_mod.Cache.init(device, &m.config, 8);
    defer cache_a.deinit();
    var cache_b = try state_mod.Cache.init(device, &m.config, 8);
    defer cache_b.deinit();
    var ps_a = try prefill_mod.PrefillScratch.init(device, &m.config, 8);
    defer ps_a.deinit();
    var ps_b = try prefill_mod.PrefillScratch.init(device, &m.config, 8);
    defer ps_b.deinit();
    try prefill_mod.prefill(m, &ps_a, &cache_a, &ids);
    const la = try gpa.alloc(f32, 4 * vocab);
    defer gpa.free(la);
    @memcpy(la, ps_a.logits.slice(f32, 4 * vocab));
    const lb = try gpa.alloc(f32, 4 * vocab);
    defer gpa.free(lb);
    for (ids, 0..) |id, i| {
        try prefill_mod.prefill(m, &ps_b, &cache_b, ([_]u32{id})[0..1]);
        @memcpy(lb[i * vocab ..][0..vocab], ps_b.logits.slice(f32, vocab));
    }
    const d0 = maxAbsDiff(la[0 * vocab ..][0..vocab], lb[0 * vocab ..][0..vocab]);
    const d1 = maxAbsDiff(la[1 * vocab ..][0..vocab], lb[1 * vocab ..][0..vocab]);
    const d2 = maxAbsDiff(la[2 * vocab ..][0..vocab], lb[2 * vocab ..][0..vocab]);
    const d3 = maxAbsDiff(la[3 * vocab ..][0..vocab], lb[3 * vocab ..][0..vocab]);
    std.debug.print("self-consistency row diffs {e} {e} {e} {e}\n", .{ d0, d1, d2, d3 });
    const kvw: usize = m.config.text.local.kv * m.config.text.local.dim;
    const kv_bytes = cache_a.keys.length();
    const fa = cache_a.keys.slice(f32, 0);
    _ = fa;
    var sa = try stageInit(device, kv_bytes, 16, 16);
    defer stageDeinit(&sa);
    var sb = try stageInit(device, kv_bytes, 16, 16);
    defer stageDeinit(&sb);
    try snapshot(m, &cache_a, &sa);
    try snapshot(m, &cache_b, &sb);
    const xa = sa.keys.slice(f32, kv_bytes / 4);
    const xb = sb.keys.slice(f32, kv_bytes / 4);
    for (0..m.config.slots()) |s| {
        for (0..4) |j| {
            const off = kvRowOff(&m.config, &cache_a, s, j, kvw) / 4;
            const dd = maxAbsDiff(xa[off..][0..kvw], xb[off..][0..kvw]);
            if (dd != 0) std.debug.print("self slot {d} pos {d} kdiff {e}\n", .{ s, j, dd });
        }
    }
    const h: usize = m.config.text.hidden;
    const kc_bytes = m.config.slots() * state_mod.ring * kvw * 4;
    const ac_bytes = m.config.slots() * state_mod.ring * h * 4;
    const ka = try device.buffer(@max(kc_bytes, 16), mtl.ResourceOptions.shared);
    defer ka.deinit();
    const kb2 = try device.buffer(@max(kc_bytes, 16), mtl.ResourceOptions.shared);
    defer kb2.deinit();
    const oa = try device.buffer(@max(ac_bytes, 16), mtl.ResourceOptions.shared);
    defer oa.deinit();
    const ob2 = try device.buffer(@max(ac_bytes, 16), mtl.ResourceOptions.shared);
    defer ob2.deinit();
    {
        const cb = m.queue.commandBuffer();
        const blit = cb.blit();
        blit.copy(cache_a.k_conv, 0, ka, 0, kc_bytes);
        blit.copy(cache_b.k_conv, 0, kb2, 0, kc_bytes);
        blit.copy(cache_a.mlp_conv, 0, oa, 0, ac_bytes);
        blit.copy(cache_b.mlp_conv, 0, ob2, 0, ac_bytes);
        blit.end();
        cb.commit();
        cb.wait();
    }
    const ra = ka.slice(f32, kc_bytes / 4);
    const rb = kb2.slice(f32, kc_bytes / 4);
    const ma = oa.slice(f32, ac_bytes / 4);
    const mb = ob2.slice(f32, ac_bytes / 4);
    for (0..m.config.slots()) |s| {
        for (0..state_mod.ring) |r| {
            const off = (s * state_mod.ring + r) * kvw;
            const dd = maxAbsDiff(ra[off..][0..kvw], rb[off..][0..kvw]);
            if (dd != 0) std.debug.print("self slot {d} ring {d} kconvdiff {e}\n", .{ s, r, dd });
        }
        for (0..state_mod.ring) |r| {
            const off = (s * state_mod.ring + r) * h;
            const dd = maxAbsDiff(ma[off..][0..h], mb[off..][0..h]);
            if (dd != 0) std.debug.print("self slot {d} ring {d} mconvdiff {e}\n", .{ s, r, dd });
        }
    }
    try t.expect(d0 == 0);
    try t.expect(d1 == 0);
    try t.expect(d2 == 0);
    try t.expect(d3 == 0);
}

test "chunked prefill matches per-token position on the tiny fixture" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const gpa = t.allocator;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tinyDir(a) orelse return error.SkipZigTest;
    {
        const probe = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
        var pf = std.Io.Dir.cwd().openFile(io, probe, .{}) catch return error.SkipZigTest;
        pf.close(io);
    }
    const tmpbase = std.mem.span(std.c.getenv("TMPDIR") orelse "/tmp");
    const shim = try std.fs.path.join(a, &.{ tmpbase, "i2-prefill-shim" });
    std.Io.Dir.cwd().createDirPath(io, shim) catch {};
    try writeShim(gpa, io, shim, dir);
    var m = try model_mod.Model.load(gpa, io, shim);
    defer m.deinit();
    const N: usize = 10;
    const C: usize = 8;
    const K: usize = m.config.text.topk;
    var ids: [N]u32 = undefined;
    for (0..N) |i| ids[i] = @intCast(100 + i * 7);
    const vocab: usize = m.config.text.unpadded_vocab;
    const ref_logits = try gpa.alloc(f32, N * vocab);
    defer gpa.free(ref_logits);
    const got_logits = try gpa.alloc(f32, N * vocab);
    defer gpa.free(got_logits);
    const ref_picks = try gpa.alloc(i32, N * K);
    defer gpa.free(ref_picks);
    const got_picks = try gpa.alloc(i32, N * K);
    defer gpa.free(got_picks);

    var cache_a = try state_mod.Cache.init(device, &m.config, N + 1);
    defer cache_a.deinit();
    var scratch = try state_mod.Scratch.init(device, &m.config);
    defer scratch.deinit();
    for (ids, 0..) |id, i| {
        try fwd.position(m, &scratch, &cache_a, id);
        @memcpy(ref_logits[i * vocab ..][0..vocab], scratch.logits.slice(f32, vocab));
        @memcpy(ref_picks[i * K ..][0..K], scratch.picks.slice(i32, K));
    }
    const kvw: usize = m.config.text.local.kv * m.config.text.local.dim;
    const h: usize = m.config.text.hidden;
    const conv_elems = m.config.slots() * state_mod.ring * kvw;
    const mlp_elems = m.config.slots() * state_mod.ring * h;
    const kv_bytes = cache_a.keys.length();
    var stage_a = try stageInit(device, kv_bytes, conv_elems * 4, mlp_elems * 4);
    defer stageDeinit(&stage_a);
    try snapshot(m, &cache_a, &stage_a);

    for ([_]usize{ 1, 4, 5 }) |chunk| {
        var cache_b = try state_mod.Cache.init(device, &m.config, N + 1);
        defer cache_b.deinit();
        var ps = try prefill_mod.PrefillScratch.init(device, &m.config, C);
        defer ps.deinit();
        var at: usize = 0;
        while (at < N) {
            const take = @min(chunk, N - at);
            try prefill_mod.prefill(m, &ps, &cache_b, ids[at..][0..take]);
            @memcpy(got_logits[at * vocab ..][0..take * vocab], ps.logits.slice(f32, take * vocab));
            @memcpy(got_picks[at * K ..][0..take * K], ps.picks.slice(i32, take * K));
            at += take;
        }
        try t.expectEqual(cache_a.len, cache_b.len);
        var stage_b = try stageInit(device, kv_bytes, conv_elems * 4, mlp_elems * 4);
        defer stageDeinit(&stage_b);
        try snapshot(m, &cache_b, &stage_b);
        const bound = @max(1.0, maxAbs(ref_logits)) * 1.5e-5;
        const kd = kvStateDiff(&m.config, &cache_a, stage_a.keys.slice(f32, kv_bytes / 4), stage_b.keys.slice(f32, kv_bytes / 4), N);
        const vd = kvStateDiff(&m.config, &cache_a, stage_a.values.slice(f32, kv_bytes / 4), stage_b.values.slice(f32, kv_bytes / 4), N);
        std.debug.print("chunk {d}: logits {e} keys {e} values {e} conv {e} mlp {e} bound {e}\n", .{
            chunk,
            maxAbsDiff(ref_logits, got_logits),
            kd,
            vd,
            maxAbsDiff(stage_a.kc.slice(f32, conv_elems), stage_b.kc.slice(f32, conv_elems)),
            maxAbsDiff(stage_a.ac.slice(f32, mlp_elems), stage_b.ac.slice(f32, mlp_elems)),
            bound,
        });
        for (0..N) |r| {
            const d = maxAbsDiff(ref_logits[r * vocab ..][0..vocab], got_logits[r * vocab ..][0..vocab]);
            if (d > 1e-6) std.debug.print("chunk {d} row {d} logit diff {e}\n", .{ chunk, r, d });
        }
        try t.expectEqualSlices(i32, ref_picks, got_picks);
        try t.expect(maxAbsDiff(ref_logits, got_logits) <= bound);
        try t.expect(kd <= bound);
        try t.expect(vd <= bound);
        try t.expect(maxAbsDiff(stage_a.kc.slice(f32, conv_elems), stage_b.kc.slice(f32, conv_elems)) <= bound);
        try t.expect(maxAbsDiff(stage_a.vc.slice(f32, conv_elems), stage_b.vc.slice(f32, conv_elems)) <= bound);
        try t.expect(maxAbsDiff(stage_a.ac.slice(f32, mlp_elems), stage_b.ac.slice(f32, mlp_elems)) <= bound);
        try t.expect(maxAbsDiff(stage_a.mc.slice(f32, mlp_elems), stage_b.mc.slice(f32, mlp_elems)) <= bound);
    }

    var cache_c = try state_mod.Cache.init(device, &m.config, N + 2);
    defer cache_c.deinit();
    var cache_d = try state_mod.Cache.init(device, &m.config, N + 2);
    defer cache_d.deinit();
    var ps = try prefill_mod.PrefillScratch.init(device, &m.config, C);
    defer ps.deinit();
    for (ids) |id| try fwd.position(m, &scratch, &cache_c, id);
    var at: usize = 0;
    while (at < N) {
        const take = @min(4, N - at);
        try prefill_mod.prefill(m, &ps, &cache_d, ids[at..][0..take]);
        at += take;
    }
    try fwd.position(m, &scratch, &cache_c, 42);
    @memcpy(ref_logits[0..vocab], scratch.logits.slice(f32, vocab));
    try prefill_mod.prefill(m, &ps, &cache_d, &.{42});
    @memcpy(got_logits[0..vocab], ps.logits.slice(f32, vocab));
    try t.expect(maxAbsDiff(ref_logits[0..vocab], got_logits[0..vocab]) <= @max(1.0, maxAbs(ref_logits[0..vocab])) * 1.5e-5);
}

const legacyGemm =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\struct i2p_gemm_args { uint n, k, rows, xs, xb; float cap; };
    \\[[kernel]] void i2p_gemm(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    \\    constant i2p_gemm_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    \\    uint3 tg [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]]) {
    \\  constexpr int BM = 8;
    \\  constexpr int BN = 64;
    \\  constexpr int SG_N = BN / 8;
    \\  constexpr int TG_THREADS = SG_N * 32;
    \\  constexpr int A_PAD = 16;
    \\  constexpr int B_PAD = BN + 8;
    \\  constexpr int W_PAD = BN + 8;
    \\  if (TG_THREADS > 256) return;
    \\  const int sg_id = int(ti) / 32;
    \\  const int sg_n = sg_id % SG_N;
    \\  const int m0 = int(tg.y) * BM;
    \\  const int n_base = int(tg.x) * BN;
    \\  threadgroup float xs_pad[BM * A_PAD];
    \\  threadgroup float ws_pad[8 * B_PAD];
    \\  threadgroup float wb[BM * W_PAD];
    \\  simdgroup_matrix<float, 8, 8> acc(0.0f);
    \\  const bool m_in = m0 < int(a.rows);
    \\  if (m_in) {
    \\    for (int k0 = 0; k0 < int(a.k); k0 += 8) {
    \\      for (int e = int(ti); e < BM * 8; e += TG_THREADS) {
    \\        const int r = e / 8;
    \\        const int c = e % 8;
    \\        const int gr = m0 + r;
    \\        const int gc = k0 + c;
    \\        float v = 0.0f;
    \\        if (gr < int(a.rows) && gc < int(a.k)) {
    \\          const uint xr = a.xb + uint(gr) * a.xs;
    \\          v = X[size_t(xr) * size_t(a.k) + uint(gc)];
    \\        }
    \\        xs_pad[r * A_PAD + c] = v;
    \\      }
    \\      for (int e = int(ti); e < 8 * BN; e += TG_THREADS) {
    \\        const int kk = e / BN;
    \\        const int nn = e % BN;
    \\        const int gn = n_base + nn;
    \\        const int gk = k0 + kk;
    \\        float v = 0.0f;
    \\        if (gn < int(a.n) && gk < int(a.k)) v = float(W[size_t(gn) * size_t(a.k) + uint(gk)]);
    \\        ws_pad[kk * B_PAD + nn] = v;
    \\      }
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      simdgroup_matrix<float, 8, 8> amat;
    \\      simdgroup_matrix<float, 8, 8> bmat;
    \\      simdgroup_load(amat, xs_pad, ulong(A_PAD));
    \\      simdgroup_load(bmat, ws_pad + sg_n * 8, ulong(B_PAD), ulong2(0, 0), false);
    \\      simdgroup_multiply_accumulate(acc, amat, bmat, acc);
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    simdgroup_store(acc, wb + sg_n * 8, ulong(W_PAD));
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (m_in) {
    \\    for (int e = int(ti); e < BM * BN; e += TG_THREADS) {
    \\      const int rr = e / BN;
    \\      const int cc = e % BN;
    \\      const int om = int(tg.y) * BM + rr;
    \\      const int on = int(tg.x) * BN + cc;
    \\      if (om < int(a.rows) && on < int(a.n)) {
    \\        float v = wb[rr * W_PAD + cc];
    \\        if (a.cap > 0.0f) v = a.cap * precise::tanh(v / a.cap);
    \\        O[size_t(om) * size_t(a.n) + uint(on)] = v;
    \\      }
    \\    }
    \\  }
    \\}
    \\
;

const GemmArgs = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, cap: f32 };

test "tiled dense projections match the original gemm" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @import("kernels.zig").prefill, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const ref_lib = try mtl.Library.fromSource(device, legacyGemm, mtl.CompileOptions.mlx());
    defer ref_lib.deinit();
    const candidate = try mtl.Pipeline.init(device, lib, "i2p_gemm", false);
    defer candidate.deinit();
    const reference = try mtl.Pipeline.init(device, ref_lib, "i2p_gemm", false);
    defer reference.deinit();
    const shapes = [_][2]usize{ .{ 3072, 2304 }, .{ 384, 2304 }, .{ 24, 2304 }, .{ 1024, 2304 }, .{ 32, 2304 }, .{ 2304, 3072 }, .{ 128, 2304 }, .{ 64, 4608 }, .{ 2304, 64 }, .{ 200058, 2304 } };
    for (shapes) |shape| {
        const n = shape[0];
        const k = shape[1];
        const w = try device.buffer(n * k * 2, mtl.ResourceOptions.shared);
        defer w.deinit();
        const ws = w.slice(u16, n * k);
        for (ws, 0..) |*v, i| v.* = @as(u16, @intCast(0x3e00 + ((i *% 37 + i / k *% 13) % 256))) | (if (i % 3 == 0) @as(u16, 0x8000) else 0);
        for ([_]usize{ 1, 7, 128 }) |rows| {
            const x = try device.buffer((rows * 2 + 1) * k * 4, mtl.ResourceOptions.shared);
            defer x.deinit();
            for (x.slice(f32, (rows * 2 + 1) * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i *% 17 + i / k *% 31) % 251)) - 125)) / 127.0;
            const got = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared);
            defer got.deinit();
            const ref = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared);
            defer ref.deinit();
            const args = GemmArgs{ .n = @intCast(n), .k = @intCast(k), .rows = @intCast(rows), .xs = 2, .xb = 1, .cap = 0 };
            for ([_]mtl.Pipeline{ candidate, reference }, [_]mtl.Buffer{ got, ref }, [_]usize{ 32, 8 }) |pipe, output, bm| {
                const cb = queue.commandBuffer();
                const enc = cb.compute(.serial);
                enc.setPipeline(pipe);
                enc.setBuffer(x, 0, 0);
                enc.setBuffer(w, 0, 1);
                enc.setValue(args, 2);
                enc.setBuffer(output, 0, 3);
                enc.dispatchGroups(mtl.Size.of((n + 63) / 64, (rows + bm - 1) / bm, 1), mtl.Size.of(256, 1, 1));
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| { std.debug.print("gemm: {s}\n", .{msg}); return error.GpuFailed; }
            }
            const actual = got.slice(f32, rows * n);
            const expected = ref.slice(f32, rows * n);
            const diff = maxAbsDiff(actual, expected);
            std.debug.print("gemm n={d} k={d} rows={d} maxdiff={e}\n", .{ n, k, rows, diff });
            const bound = @max(1.0, maxAbs(expected)) * 1.5e-5;
            for (actual, expected) |a, e| try t.expect(std.math.isFinite(a) and std.math.isFinite(e) and @abs(a - e) <= bound);
        }
    }
}

const ExpertArgs = extern struct { k: u32, n: u32, slots: u32, topk: u32, experts: u32, gate_cap: f32, up_cap: f32 };

fn expertE4(code: u8) f64 {
    const exp: i32 = @intCast((code >> 3) & 15);
    const man: f64 = @floatFromInt(code & 7);
    const value = if (exp == 0) man / 512.0 else (1.0 + man / 8.0) * std.math.pow(f64, 2.0, @floatFromInt(exp - 7));
    return if (code & 128 != 0) -value else value;
}

fn expertCode(code: u8, mx: bool) f64 {
    if (mx) return expertE4(code);
    const values = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const value = values[code & 7];
    return if (code & 8 != 0) -value else value;
}

test "routed expert projections meet fp64 bounds at occupancy boundaries" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    for ([_]bool{ false, true }) |mx| {
        const source = if (mx) @import("kernels.zig").prefillSource(true) else @import("kernels.zig").prefillSource(false);
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const matrix = try mtl.Pipeline.init(device, lib, "i2p_expert_matrix", false);
        defer matrix.deinit();
        const codes = try device.buffer(256 * 4, mtl.ResourceOptions.shared);
        defer codes.deinit();
        for (codes.slice(f32, 256), 0..) |*v, i| v.* = @floatCast(expertCode(@intCast(if (mx and i % 128 == 127) 0 else i % (if (mx) @as(usize, 256) else 16)), mx));
        const globals = try device.buffer(16, mtl.ResourceOptions.shared);
        defer globals.deinit();
        globals.slice(f32, 1)[0] = 0.73;
        const den = try device.buffer(16, mtl.ResourceOptions.shared);
        defer den.deinit();
        den.slice(f32, 1)[0] = 1.3;
        const off = try device.buffer(16, mtl.ResourceOptions.shared);
        defer off.deinit();
        const row_of = try device.buffer(128 * 4, mtl.ResourceOptions.shared);
        defer row_of.deinit();
        for (row_of.slice(i32, 128), 0..) |*v, i| v.* = @intCast(127 - i);
        for ([_]bool{ true, false }) |gu| {
            const k: usize = if (gu) 2304 else 1152;
            const n: usize = 64;
            const cols = n * (if (gu) @as(usize, 2) else 1);
            const group: usize = if (mx) 32 else 16;
            const code_stride = if (mx) k else k / 2;
            const w = try device.buffer(cols * code_stride, mtl.ResourceOptions.shared);
            defer w.deinit();
            const scales = try device.buffer(cols * (k / group), mtl.ResourceOptions.shared);
            defer scales.deinit();
            for (w.slice(u8, cols * code_stride), 0..) |*v, i| v.* = if (mx) @intCast((i *% 13 + i / code_stride *% 7) % 112 + 128 * ((i / 5 + i / code_stride) % 2)) else @intCast((i *% 37 + i / code_stride *% 13) % 256);
            for (scales.slice(u8, cols * (k / group)), 0..) |*v, i| v.* = if (mx) @intCast(123 + i % 5) else @intCast(40 + i % 16);
            const x = try device.buffer(128 * k * 4, mtl.ResourceOptions.shared);
            defer x.deinit();
            for (x.slice(f32, 128 * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i *% 17 + i / k *% 31) % 251)) - 125)) / 127.0;
            const out = try device.buffer(128 * n * 4, mtl.ResourceOptions.shared);
            defer out.deinit();
            const grouped = try mtl.Pipeline.init(device, lib, if (gu) "i2p_gu_grouped" else "i2p_down_grouped", false);
            defer grouped.deinit();
            for ([_]usize{ 1, 3, 8, 15, 16, 32, 128 }) |rows| {
                off.slice(i32, 2)[0] = 0;
                off.slice(i32, 2)[1] = @intCast(rows);
                const args = ExpertArgs{ .k = @intCast(k), .n = n, .slots = 128, .topk = 8, .experts = 1, .gate_cap = if (gu) 7 else 0, .up_cap = if (gu) 7 else 0 };
                const cb = queue.commandBuffer();
                const enc = cb.compute(.serial);
                for ([_]mtl.Pipeline{ grouped, matrix }, 0..) |pipe, p| {
                    enc.setPipeline(pipe);
                    enc.setBuffer(x, 0, 0);
                    enc.setBuffer(w, 0, 1);
                    enc.setBuffer(scales, 0, 2);
                    enc.setBuffer(globals, 0, 3);
                    enc.setBuffer(den, 0, 4);
                    enc.setBuffer(off, 0, 5);
                    enc.setBuffer(row_of, 0, 6);
                    enc.setValue(args, 7);
                    enc.setBuffer(out, 0, 8);
                    enc.setBuffer(codes, 0, 9);
                    if (p == 0) enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1)) else enc.dispatchGroups(mtl.Size.of((cols + 63) / 64, 1, (rows + 31) / 32), mtl.Size.of(256, 1, 1));
                }
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| { std.debug.print("expert: {s}\n", .{msg}); return error.GpuFailed; }
                var worst: f64 = 0;
                for (0..rows) |r| {
                    const slot = 127 - r;
                    for (0..n) |c| {
                        var sums = [_]f64{ 0, 0 };
                        var magnitudes = [_]f64{ 0, 0 };
                        for (0..if (gu) @as(usize, 2) else 1) |part| {
                            const wr = if (gu) 2 * c + part else c;
                            for (0..k) |j| {
                                const byte_code = w.slice(u8, cols * code_stride)[wr * code_stride + (if (mx) j else j / 2)];
                                const code = if (mx) byte_code else (byte_code >> @as(u3, @intCast((j % 2) * 4))) & 15;
                                const sc = scales.slice(u8, cols * (k / group))[wr * (k / group) + j / group];
                                const scale = if (mx) std.math.pow(f64, 2, @as(f64, @floatFromInt(sc)) - 127) else expertE4(sc);
                                const input_row = if (gu) slot / 8 else slot;
                                const term = expertCode(code, mx) * scale * @as(f64, x.slice(f32, 128 * k)[input_row * k + j]);
                                sums[part] += term;
                                magnitudes[part] += @abs(term);
                            }
                        }
                        const global: f64 = if (mx) 1 else @as(f64, globals.slice(f32, 1)[0] / den.slice(f32, 1)[0]);
                        const gate = sums[0] * global;
                        const up = sums[1] * global;
                        const expected = if (gu) (7.0 * std.math.tanh(gate / 7.0) / (1.0 + @exp(-gate))) * (7.0 * std.math.tanh(up / 7.0)) else gate;
                        const bound = (if (gu) @as(f64, 16.0) else 1.0) * (@as(f64, @floatFromInt(k)) * 0x1p-24 / (1.0 - @as(f64, @floatFromInt(k)) * 0x1p-24)) * (magnitudes[0] + magnitudes[1]) * global + 1e-6;
                        const actual: f64 = out.slice(f32, 128 * n)[slot * n + c];
                        const err = @abs(actual - expected);
                        worst = @max(worst, err);
                        if (!std.math.isFinite(actual) or err > bound) {
                            std.debug.print("expert mx={} gu={} rows={d} slot={d} col={d} got={e} ref={e} error={e} bound={e}\n", .{ mx, gu, rows, slot, c, actual, expected, err, bound });
                            return error.ExpertReferenceMismatch;
                        }
                    }
                }
                std.debug.print("expert mx={} gu={} rows={d} worst={e}\n", .{ mx, gu, rows, worst });

const AttnArgs = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, base: u32, rows: u32, rmode: u32, scale: f32, floor: f32, alpha: f32 };

fn attnKey(j: usize, kv: usize, d: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast((j * 17 + kv * 7) % 101)) - 50)) * @as(f32, @floatFromInt(@as(i32, @intCast(d % 17)) - 8)) / 512;
}

fn attnValue(j: usize, kv: usize, d: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast((j * 13 + kv * 11) % 97)) - 48)) * @as(f32, @floatFromInt(@as(i32, @intCast(d % 17)) - 8)) / 512;
}

test "tiled attention matches fp64 and original attention" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @import("kernels.zig").prefill, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const tiled = try mtl.Pipeline.init(device, lib, "i2p_attn_tiled", false);
    defer tiled.deinit();
    const original = try mtl.Pipeline.init(device, lib, "i2p_attn", false);
    defer original.deinit();
    const heads = 24; const kv = 8; const dim = 128; const extent = 64;
    for ([_]usize{ 0, 900, 5000 }) |base| {
        for ([_]usize{ 0, 1024 }) |window| {
            const cache_rows = if (window > 0) window else @max(base, 1);
            const keys = try device.buffer(cache_rows * kv * dim * 4, mtl.ResourceOptions.shared);
            defer keys.deinit();
            const values = try device.buffer(cache_rows * kv * dim * 4, mtl.ResourceOptions.shared);
            defer values.deinit();
            const start = if (window > 0 and base > window) base - window else 0;
            for (start..base) |j| for (0..kv) |h| for (0..dim) |d| {
                const at = ((if (window > 0) j % window else j) * kv + h) * dim + d;
                keys.slice(f32, cache_rows * kv * dim)[at] = attnKey(j, h, d);
                values.slice(f32, cache_rows * kv * dim)[at] = attnValue(j, h, d);
            };
            for ([_]usize{ 35, 1, 7, 9, 17, 33, 63, 100, 128 }) |rows| {
                if (base == 5000 and rows != 1 and rows != 7 and rows != 128) continue;
                const q = try device.buffer(rows * heads * dim * 4, mtl.ResourceOptions.shared); defer q.deinit();
                const kc = try device.buffer(rows * kv * dim * 4, mtl.ResourceOptions.shared); defer kc.deinit();
                const vc = try device.buffer(rows * kv * dim * 4, mtl.ResourceOptions.shared); defer vc.deinit();
                const bias = try device.buffer(rows * heads * (extent + 1) * 4, mtl.ResourceOptions.shared); defer bias.deinit();
                const sink = try device.buffer(rows * heads * 4, mtl.ResourceOptions.shared); defer sink.deinit();
                const got = try device.buffer(rows * heads * dim * 4, mtl.ResourceOptions.shared); defer got.deinit();
                const ref = try device.buffer(rows * heads * dim * 4, mtl.ResourceOptions.shared); defer ref.deinit();
                for (q.slice(f32, rows * heads * dim), 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + i / dim * 3) % 67)) - 33)) / 64;
                for (0..rows) |r| for (0..kv) |h| for (0..dim) |d| {
                    const at = (r * kv + h) * dim + d;
                    kc.slice(f32, rows * kv * dim)[at] = attnKey(base + r, h, d);
                    vc.slice(f32, rows * kv * dim)[at] = attnValue(base + r, h, d);
                };
                for (bias.slice(f32, rows * heads * (extent + 1)), 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 19) % 61)) - 30)) / 128;
                for (sink.slice(f32, rows * heads), 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 16;
                const args = AttnArgs{ .heads = heads, .kv = kv, .dim = dim, .rel = 0, .extent = extent, .window = @intCast(window), .base = @intCast(base), .rows = @intCast(rows), .rmode = @intFromBool(window > 0), .scale = 1.0 / @sqrt(@as(f32, dim)), .floor = 128, .alpha = 0.25 };
                for ([_]mtl.Pipeline{ tiled, original }, [_]mtl.Buffer{ got, ref }, [_]bool{ true, false }) |pipe, output, is_tiled| {
                    const cb = queue.commandBuffer(); const enc = cb.compute(.serial);
                    enc.setPipeline(pipe); enc.setBuffer(q, 0, 0); enc.setBuffer(keys, 0, 1); enc.setBuffer(values, 0, 2);
                    enc.setBuffer(bias, 0, 3); enc.setBuffer(sink, 0, 4); enc.setValue(args, 5); enc.setBuffer(output, 0, 6); enc.setBuffer(kc, 0, 7); enc.setBuffer(vc, 0, 8);
                    if (is_tiled) enc.dispatchGroups(mtl.Size.of(kv, (rows + 7) / 8, 1), mtl.Size.of(768, 1, 1)) else enc.dispatchGroups(mtl.Size.of(heads, rows, 1), mtl.Size.of(32, 1, 1));
                    enc.end(); cb.commit(); cb.wait();
                    if (cb.failure()) |msg| { std.debug.print("attention: {s}\n", .{msg}); return error.GpuFailed; }
                }
                var max_fp64: f64 = 0; var max_original: f32 = 0;
                for (0..rows) |r| for (0..heads) |h| {
                    var qsum: f64 = 0;
                    for (0..dim) |d| qsum += @as(f64, q.slice(f32, rows * heads * dim)[(r * heads + h) * dim + d]) * @as(f64, @floatFromInt(@as(i32, @intCast(d % 17)) - 8));
                    const pos = base + r;
                    const lo = if (window > 0 and pos + 1 > window) pos + 1 - window else 0;
                    const tau: f64 = if (window == 0) 1 + @as(f64, args.alpha) * @log(@max(1, @as(f64, @floatFromInt(pos + 1)) / @as(f64, args.floor))) else 1;
                    var mass: f64 = 1; var numerator: f64 = 0;
                    for (lo..pos + 1) |j| {
                        const score = qsum * @as(f64, args.scale) * tau * @as(f64, attnKey(j, h / 3, 9)) + @as(f64, bias.slice(f32, rows * heads * (extent + 1))[(r * heads + h) * (extent + 1) + @min(pos - j, extent)]) - @as(f64, sink.slice(f32, rows * heads)[r * heads + h]);
                        const w = @exp(score); mass += w;
                        numerator += w * @as(f64, attnValue(j, h / 3, 9));
                    }
                    for (0..dim) |d| {
                        const expected = numerator / mass * @as(f64, @floatFromInt(@as(i32, @intCast(d % 17)) - 8));
                        const at = (r * heads + h) * dim + d;
                        const actual = got.slice(f32, rows * heads * dim)[at];
                        try t.expect(std.math.isFinite(actual) and std.math.isFinite(ref.slice(f32, rows * heads * dim)[at]));
                        max_fp64 = @max(max_fp64, @abs(@as(f64, actual) - expected));
                        max_original = @max(max_original, @abs(actual - ref.slice(f32, rows * heads * dim)[at]));
                    }
                };
                std.debug.print("attention base={d} window={d} rows={d} fp64={e} original={e}\n", .{ base, window, rows, max_fp64, max_original });
                try t.expect(max_fp64 <= 0.00005 and max_original == 0);
            }
        }
    }
}
