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
    const linear: usize = cfg.slots() * cache.capacity * kvw * 2;
    if (cache.keys.length() == linear) return slot * cache.capacity * kvw * 2 + at * kvw * 2;
    const phys = cfg.loop.physicalLayer(cfg.text.layers, @intCast(slot));
    const win: usize = cfg.text.window;
    var base: usize = 0;
    for (0..slot) |s| {
        const p = cfg.loop.physicalLayer(cfg.text.layers, @intCast(s));
        base += (if (!cfg.isFull(p)) win else cache.capacity) * kvw * 2;
    }
    const r: usize = if (!cfg.isFull(phys)) at % win else at;
    return base + r * kvw * 2;
}

fn kvStateDiff(cfg: anytype, cache: anytype, a: []const f16, b: []const f16, len: usize) f32 {
    const kvw: usize = cfg.text.local.kv * cfg.text.local.dim;
    var m: f32 = 0;
    for (0..cfg.slots()) |s| {
        for (0..len) |j| {
            const off = kvRowOff(cfg, cache, s, j, kvw) / 2;
            for (0..kvw) |i| {
                const d = @abs(@as(f32, a[off + i]) - @as(f32, b[off + i]));
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
    const xa = sa.keys.slice(f16, kv_bytes / 2);
    const xb = sb.keys.slice(f16, kv_bytes / 2);
    for (0..m.config.slots()) |s| {
        for (0..4) |j| {
            const off = kvRowOff(&m.config, &cache_a, s, j, kvw) / 2;
            var dd: f32 = 0;
            for (xa[off..][0..kvw], xb[off..][0..kvw]) |x, y| dd = @max(dd, @abs(@as(f32, x) - @as(f32, y)));
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
        const kd = kvStateDiff(&m.config, &cache_a, stage_a.keys.slice(f16, kv_bytes / 2), stage_b.keys.slice(f16, kv_bytes / 2), N);
        const vd = kvStateDiff(&m.config, &cache_a, stage_a.values.slice(f16, kv_bytes / 2), stage_b.values.slice(f16, kv_bytes / 2), N);
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

test "chunks longer than the local window leave the state of window-bounded chunks on the tiny fixture" {
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
    const window: usize = m.config.text.window;
    const N: usize = 3 * window;
    const bounded = window - 4;
    try t.expect(N <= 64 and bounded > 8);
    const K: usize = m.config.text.topk;
    var ids: [64]u32 = undefined;
    for (0..N) |i| ids[i] = @intCast(100 + (i * 7) % 61);
    const vocab: usize = m.config.text.unpadded_vocab;
    const kvw: usize = m.config.text.local.kv * m.config.text.local.dim;
    const h: usize = m.config.text.hidden;
    const conv_elems = m.config.slots() * state_mod.ring * kvw;
    const mlp_elems = m.config.slots() * state_mod.ring * h;
    var ps = try prefill_mod.PrefillScratch.init(device, &m.config, N);
    defer ps.deinit();
    ps.half = false;
    var logits: [4][]f32 = undefined;
    var picks: [4][]i32 = undefined;
    var stages: [4]StageBufs = undefined;
    var kv_bytes: usize = 0;
    const chunks = [_]usize{ bounded, window + 1, window + 8, N };
    for (chunks, 0..) |chunk, side| {
        var cache = try state_mod.Cache.init(device, &m.config, N + 1);
        defer cache.deinit();
        kv_bytes = cache.keys.length();
        logits[side] = try gpa.alloc(f32, N * vocab);
        picks[side] = try gpa.alloc(i32, N * K);
        var at: usize = 0;
        while (at < N) {
            const take = @min(chunk, N - at);
            try t.expect(take > 8);
            try prefill_mod.prefill(m, &ps, &cache, ids[at..][0..take]);
            @memcpy(logits[side][at * vocab ..][0 .. take * vocab], ps.logits.slice(f32, take * vocab));
            @memcpy(picks[side][at * K ..][0 .. take * K], ps.picks.slice(i32, take * K));
            at += take;
        }
        try t.expectEqual(N, cache.len);
        stages[side] = try stageInit(device, kv_bytes, conv_elems * 4, mlp_elems * 4);
        try snapshot(m, &cache, &stages[side]);
    }
    defer for (0..chunks.len) |side| {
        gpa.free(logits[side]);
        gpa.free(picks[side]);
        stageDeinit(&stages[side]);
    };
    var probe = try state_mod.Cache.init(device, &m.config, N + 1);
    defer probe.deinit();
    for (1..chunks.len) |side| {
        const kd = kvStateDiff(&m.config, &probe, stages[0].keys.slice(f16, kv_bytes / 2), stages[side].keys.slice(f16, kv_bytes / 2), N);
        const vd = kvStateDiff(&m.config, &probe, stages[0].values.slice(f16, kv_bytes / 2), stages[side].values.slice(f16, kv_bytes / 2), N);
        std.debug.print("long chunk {d} vs {d}: logits {e} keys {e} values {e}\n", .{ chunks[side], bounded, maxAbsDiff(logits[0], logits[side]), kd, vd });
        try t.expectEqualSlices(i32, picks[0], picks[side]);
        try t.expectEqualSlices(f32, logits[0], logits[side]);
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(stages[0].keys.slice(f16, kv_bytes / 2)), std.mem.sliceAsBytes(stages[side].keys.slice(f16, kv_bytes / 2)));
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(stages[0].values.slice(f16, kv_bytes / 2)), std.mem.sliceAsBytes(stages[side].values.slice(f16, kv_bytes / 2)));
        try t.expectEqualSlices(f32, stages[0].kc.slice(f32, conv_elems), stages[side].kc.slice(f32, conv_elems));
        try t.expectEqualSlices(f32, stages[0].vc.slice(f32, conv_elems), stages[side].vc.slice(f32, conv_elems));
        try t.expectEqualSlices(f32, stages[0].ac.slice(f32, mlp_elems), stages[side].ac.slice(f32, mlp_elems));
        try t.expectEqualSlices(f32, stages[0].mc.slice(f32, mlp_elems), stages[side].mc.slice(f32, mlp_elems));
    }
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

test "bf16 tensorops dense projection accumulates rounded operands" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @import("kernels.zig").prefill_gemm_mpp, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const cast = try mtl.Pipeline.init(device, lib, "i2p_bf16_convert", false);
    defer cast.deinit();
    const gemm = try mtl.Pipeline.init(device, lib, "i2p_gemm_bf16_mpp", false);
    defer gemm.deinit();
    const rows = 17; const n = 35; const k = 32;
    const args = GemmArgs{ .n = n, .k = k, .rows = rows, .xs = 2, .xb = 1, .cap = 0 };
    const x = try device.buffer((rows * 2 + 1) * k * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    @memset(x.slice(f32, (rows * 2 + 1) * k), 1.003);
    const w = try device.buffer(n * k * 2, mtl.ResourceOptions.shared);
    defer w.deinit();
    @memset(w.slice(u16, n * k), 0x3f80);
    const rounded = try device.buffer(rows * k * 2, mtl.ResourceOptions.shared);
    defer rounded.deinit();
    const out = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared);
    defer out.deinit();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(cast); enc.setBuffer(x, 0, 0); enc.setValue(args, 1); enc.setBuffer(rounded, 0, 2);
    enc.dispatchThreads(mtl.Size.of(rows * k, 1, 1), mtl.Size.of(256, 1, 1));
    enc.setPipeline(gemm); enc.setBuffer(rounded, 0, 0); enc.setBuffer(w, 0, 1); enc.setValue(args, 2); enc.setBuffer(out, 0, 3);
    enc.dispatchGroups(mtl.Size.of((n + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
    enc.end(); cb.commit(); cb.wait();
    if (cb.failure()) |msg| { std.debug.print("bf16 tensorops: {s}\n", .{msg}); return error.GpuFailed; }
    const reference: f64 = 32.0;
    try t.expect(@abs(reference - @as(f64, @floatCast(@as(f32, 1.003))) * k) > 0.01);
    for (out.slice(f32, rows * n)) |value| try t.expectApproxEqAbs(reference, @as(f64, @floatCast(value)), 0.0001);
}

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


test "dense projections match fp64" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @embedFile("kernels/prefill_gemm_mpp.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const ref_lib = try mtl.Library.fromSource(device, legacyGemm, mtl.CompileOptions.mlx());
    defer ref_lib.deinit();
    const candidate = try mtl.Pipeline.init(device, lib, "i2p_gemm_mpp", false);
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
        for ([_]usize{ 1, 7, 35, 128, 512 }) |rows| {
            const x = try device.buffer((rows * 2 + 1) * k * 4, mtl.ResourceOptions.shared);
            defer x.deinit();
            for (x.slice(f32, (rows * 2 + 1) * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i *% 17 + i / k *% 31) % 251)) - 125)) / 127.0;
            const got = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared);
            defer got.deinit();
            const ref = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared);
            defer ref.deinit();
            @memset(got.slice(f32, rows * n), std.math.nan(f32));
            @memset(ref.slice(f32, rows * n), std.math.nan(f32));
            const args = GemmArgs{ .n = @intCast(n), .k = @intCast(k), .rows = @intCast(rows), .xs = 2, .xb = 1, .cap = if (n == 200058) 30 else 0 };
            for ([_]mtl.Pipeline{ candidate, reference }, [_]mtl.Buffer{ got, ref }, [_]usize{ 16, 8 }) |pipe, output, bm| {
                const cb = queue.commandBuffer();
                const enc = cb.compute(.serial);
                enc.setPipeline(pipe);
                enc.setBuffer(x, 0, 0);
                enc.setBuffer(w, 0, 1);
                enc.setValue(args, 2);
                enc.setBuffer(output, 0, 3);
                enc.dispatchGroups(mtl.Size.of((n + (if (bm == 16) @as(usize, 31) else 63)) / (if (bm == 16) @as(usize, 32) else 64), (rows + bm - 1) / bm, 1), mtl.Size.of(if (bm == 16) 128 else 256, 1, 1));
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
            const xs = x.slice(f32, (rows * 2 + 1) * k);
            var worst: f64 = 0;
            for (0..@min(rows, 5)) |ri| {
                const row = if (rows <= 5) ri else ri * (rows - 1) / 4;
                for (0..@min(n, 64)) |ni| {
                    const col = if (n <= 64) ni else ni * (n - 1) / 63;
                    var sum: f64 = 0;
                    var magnitude: f64 = 0;
                    for (0..k) |ki| {
                        const bits: u32 = @as(u32, ws[col * k + ki]) << 16;
                        const weight: f32 = @bitCast(bits);
                        const product = @as(f64, xs[(1 + row * 2) * k + ki]) * @as(f64, weight);
                        sum += product;
                        magnitude += @abs(product);
                    }
                    const result = if (args.cap > 0) @as(f64, args.cap) * std.math.tanh(sum / @as(f64, args.cap)) else sum;
                    const error_value = @abs(@as(f64, actual[row * n + col]) - result);
                    const fp32_bound = 4.0 * @as(f64, @floatFromInt(k + 64)) * 0x1p-24 * magnitude + 8.0 * 0x1p-24 * @max(1.0, @abs(result));
                    worst = @max(worst, error_value / @max(fp32_bound, 1e-30));
                    try t.expect(error_value <= fp32_bound);
                }
            }
            std.debug.print("gemm fp64 n={d} k={d} rows={d} worst_bound_ratio={e}\n", .{ n, k, rows, worst });
        }
    }
}

test "few row dense projections match fp64" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @embedFile("kernels/prefill_smallrow.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const ref_lib = try mtl.Library.fromSource(device, legacyGemm, mtl.CompileOptions.mlx());
    defer ref_lib.deinit();
    const candidate = try mtl.Pipeline.init(device, lib, "i2p_small", false);
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
        for ([_]usize{ 1, 2, 4, 8, 16 }) |rows| {
            const x = try device.buffer((rows * 2 + 1) * k * 4, mtl.ResourceOptions.shared);
            defer x.deinit();
            for (x.slice(f32, (rows * 2 + 1) * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i *% 17 + i / k *% 31) % 251)) - 125)) / 127.0;
            const got = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared);
            defer got.deinit();
            const ref = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared);
            defer ref.deinit();
            @memset(got.slice(f32, rows * n), std.math.nan(f32));
            @memset(ref.slice(f32, rows * n), std.math.nan(f32));
            const args = GemmArgs{ .n = @intCast(n), .k = @intCast(k), .rows = @intCast(rows), .xs = 2, .xb = 1, .cap = if (n == 200058) 30 else 0 };
            for ([_]mtl.Pipeline{ candidate, reference }, [_]mtl.Buffer{ got, ref }, [_]usize{ 1, 8 }) |pipe, output, bm| {
                const cb = queue.commandBuffer();
                const enc = cb.compute(.serial);
                enc.setPipeline(pipe);
                enc.setBuffer(x, 0, 0);
                enc.setBuffer(w, 0, 1);
                enc.setValue(args, 2);
                enc.setBuffer(output, 0, 3);
                enc.dispatchGroups(mtl.Size.of(if (bm == 1) (n + 7) / 8 else (n + 63) / 64, if (bm == 1) 1 else (rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
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
            const xs = x.slice(f32, (rows * 2 + 1) * k);
            var worst: f64 = 0;
            for (0..rows) |row| {
                for (0..@min(n, 64)) |ni| {
                    const col = if (n <= 64) ni else ni * (n - 1) / 63;
                    var sum: f64 = 0;
                    var magnitude: f64 = 0;
                    for (0..k) |ki| {
                        const bits: u32 = @as(u32, ws[col * k + ki]) << 16;
                        const weight: f32 = @bitCast(bits);
                        const product = @as(f64, xs[(1 + row * 2) * k + ki]) * @as(f64, weight);
                        sum += product;
                        magnitude += @abs(product);
                    }
                    const result = if (args.cap > 0) @as(f64, args.cap) * std.math.tanh(sum / @as(f64, args.cap)) else sum;
                    const error_value = @abs(@as(f64, actual[row * n + col]) - result);
                    const fp32_bound = 4.0 * @as(f64, @floatFromInt(k + 64)) * 0x1p-24 * magnitude + 8.0 * 0x1p-24 * @max(1.0, @abs(result));
                    worst = @max(worst, error_value / @max(fp32_bound, 1e-30));
                    try t.expect(error_value <= fp32_bound);
                }
            }
            std.debug.print("gemm fp64 n={d} k={d} rows={d} worst_bound_ratio={e}\n", .{ n, k, rows, worst });
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

test "half routed expert matrix rounds operands before f32 accumulation" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    for ([_]bool{ false, true }) |mx| {
        const source = if (mx) "#define I2P_HALF_OPERANDS 1\n#define I2P_MXFP8 1\n" ++ @import("kernels.zig").prefill else "#define I2P_HALF_OPERANDS 1\n" ++ @import("kernels.zig").prefill;
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "i2p_expert_matrix", false);
        defer pipe.deinit();
        const rows = 16;
        const n = 64;
        const k = 32;
        const x = try device.buffer(rows * k * 4, mtl.ResourceOptions.shared);
        defer x.deinit();
        @memset(x.slice(f32, rows * k), 1.0006);
        const w = try device.buffer(n * k, mtl.ResourceOptions.shared);
        defer w.deinit();
        @memset(w.slice(u8, n * k), if (mx) 0x38 else 0x22);
        const scales = try device.buffer(n * 2, mtl.ResourceOptions.shared);
        defer scales.deinit();
        @memset(scales.slice(u8, n * 2), if (mx) 127 else 0x38);
        const global = try device.buffer(16, mtl.ResourceOptions.shared);
        defer global.deinit();
        global.slice(f32, 1)[0] = 1;
        const off = try device.buffer(16, mtl.ResourceOptions.shared);
        defer off.deinit();
        off.slice(i32, 2)[0] = 0;
        off.slice(i32, 2)[1] = rows;
        const row_of = try device.buffer(rows * 4, mtl.ResourceOptions.shared);
        defer row_of.deinit();
        for (row_of.slice(i32, rows), 0..) |*value, i| value.* = @intCast(rows - 1 - i);
        const out = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared);
        defer out.deinit();
        @memset(out.slice(f32, rows * n), std.math.nan(f32));
        const codes = try device.buffer(256 * 4, mtl.ResourceOptions.shared);
        defer codes.deinit();
        for (codes.slice(f32, 256), 0..) |*value, i| value.* = @floatCast(expertCode(@intCast(if (mx and i % 128 == 127) 0 else i % (if (mx) @as(usize, 256) else 16)), mx));
        const args = ExpertArgs{ .k = k, .n = n, .slots = rows, .topk = 1, .experts = 1, .gate_cap = 0, .up_cap = 0 };
        const overflow = try device.buffer(16, mtl.ResourceOptions.shared);
        defer overflow.deinit();
        overflow.slice(u32, 1)[0] = 0;
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipe);
        enc.setBuffer(x, 0, 0);
        enc.setBuffer(w, 0, 1);
        enc.setBuffer(scales, 0, 2);
        enc.setBuffer(global, 0, 3);
        enc.setBuffer(global, 0, 4);
        enc.setBuffer(off, 0, 5);
        enc.setBuffer(row_of, 0, 6);
        enc.setValue(args, 7);
        enc.setBuffer(out, 0, 8);
        enc.setBuffer(codes, 0, 9);
        enc.setBuffer(overflow, 0, 10);
        enc.dispatchGroups(mtl.Size.of(2, 1, 1), mtl.Size.of(128, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| { std.debug.print("half expert: {s}\n", .{msg}); return error.GpuFailed; }
        try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
        const rounded: f64 = @floatCast(@as(f16, @floatCast(@as(f32, 1.0006))));
        const reference = rounded * 32.0;
        const unrounded_reference = @as(f64, @floatCast(@as(f32, 1.0006))) * 32.0;
        try t.expect(@abs(reference - unrounded_reference) > 0.001);
        for (out.slice(f32, rows * n)) |value| try t.expectApproxEqAbs(reference, @as(f64, @floatCast(value)), 0.0001);
        x.slice(f32, rows * k)[0] = 65505;
        const cb_overflow = queue.commandBuffer();
        const enc_overflow = cb_overflow.compute(.serial);
        enc_overflow.setPipeline(pipe);
        enc_overflow.setBuffer(x, 0, 0);
        enc_overflow.setBuffer(w, 0, 1);
        enc_overflow.setBuffer(scales, 0, 2);
        enc_overflow.setBuffer(global, 0, 3);
        enc_overflow.setBuffer(global, 0, 4);
        enc_overflow.setBuffer(off, 0, 5);
        enc_overflow.setBuffer(row_of, 0, 6);
        enc_overflow.setValue(args, 7);
        enc_overflow.setBuffer(out, 0, 8);
        enc_overflow.setBuffer(codes, 0, 9);
        enc_overflow.setBuffer(overflow, 0, 10);
        enc_overflow.dispatchGroups(mtl.Size.of(2, 1, 1), mtl.Size.of(128, 1, 1));
        enc_overflow.end();
        cb_overflow.commit();
        cb_overflow.wait();
        if (cb_overflow.failure()) |msg| { std.debug.print("half expert overflow: {s}\n", .{msg}); return error.GpuFailed; }
        try t.expectEqual(@as(u32, 1), overflow.slice(u32, 1)[0]);
        try t.expectApproxEqAbs(65504.0 + 31.0 * rounded, @as(f64, @floatCast(out.slice(f32, rows * n)[0])), 0.1);
    }
}

test "half routed expert fragments match varied fp64 products" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init(); defer device.deinit();
    const queue = try device.queue(); defer queue.deinit();
    const rows = 17; const n = 70; const k = 64;
    for ([_]bool{ false, true }) |mx| {
        const source = if (mx) "#define I2P_HALF_OPERANDS 1\n#define I2P_MXFP8 1\n" ++ @import("kernels.zig").prefill else "#define I2P_HALF_OPERANDS 1\n" ++ @import("kernels.zig").prefill;
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx()); defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "i2p_expert_matrix", false); defer pipe.deinit();
        const x = try device.buffer(rows * k * 4, mtl.ResourceOptions.shared); defer x.deinit();
        for (x.slice(f32, rows * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + i / k * 11) % 97)) - 48)) / 31.0;
        const weights = try device.buffer(n * k, mtl.ResourceOptions.shared); defer weights.deinit();
        const wb = weights.slice(u8, n * k);
        if (mx) {
            for (wb, 0..) |*v, i| v.* = @intCast((i * 13 + i / k * 7) % 120);
        } else {
            for (wb[0 .. n * k / 2], 0..) |*v, i| {
                const lo: u8 = @intCast((i * 7 + i / (k / 2) * 3) % 16);
                const hi: u8 = @intCast((i * 11 + i / (k / 2) * 5) % 16);
                v.* = lo | (hi << 4);
            }
        }
        const scales = try device.buffer(n * (k / 16), mtl.ResourceOptions.shared); defer scales.deinit();
        @memset(scales.slice(u8, n * (k / 16)), if (mx) 127 else 0x38);
        const global = try device.buffer(16, mtl.ResourceOptions.shared); defer global.deinit(); global.slice(f32, 1)[0] = 1;
        const off = try device.buffer(16, mtl.ResourceOptions.shared); defer off.deinit(); off.slice(i32, 2)[0] = 0; off.slice(i32, 2)[1] = rows;
        const route = try device.buffer(rows * 4, mtl.ResourceOptions.shared); defer route.deinit();
        for (route.slice(i32, rows), 0..) |*v, i| v.* = @intCast(rows - 1 - i);
        const out = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared); defer out.deinit();
        @memset(out.slice(f32, rows * n), std.math.nan(f32));
        const codes = try device.buffer(256 * 4, mtl.ResourceOptions.shared); defer codes.deinit();
        for (codes.slice(f32, 256), 0..) |*v, i| v.* = @floatCast(expertCode(@intCast(if (mx and i % 128 == 127) 0 else i % (if (mx) @as(usize, 256) else 16)), mx));
        const overflow = try device.buffer(16, mtl.ResourceOptions.shared); defer overflow.deinit(); overflow.slice(u32, 1)[0] = 0;
        const args = ExpertArgs{ .k = k, .n = n, .slots = rows, .topk = 1, .experts = 1, .gate_cap = 0, .up_cap = 0 };
        const cb = queue.commandBuffer(); const enc = cb.compute(.serial);
        enc.setPipeline(pipe); enc.setBuffer(x, 0, 0); enc.setBuffer(weights, 0, 1); enc.setBuffer(scales, 0, 2);
        enc.setBuffer(global, 0, 3); enc.setBuffer(global, 0, 4); enc.setBuffer(off, 0, 5); enc.setBuffer(route, 0, 6);
        enc.setValue(args, 7); enc.setBuffer(out, 0, 8); enc.setBuffer(codes, 0, 9); enc.setBuffer(overflow, 0, 10);
        enc.dispatchGroups(mtl.Size.of((n + 31) / 32, 1, 1), mtl.Size.of(128, 1, 1));
        enc.end(); cb.commit(); cb.wait();
        if (cb.failure()) |msg| { std.debug.print("half expert varied: {s}\n", .{msg}); return error.GpuFailed; }
        for (0..rows) |row| for (0..n) |col| {
            var expected: f64 = 0; var magnitude: f64 = 0;
            for (0..k) |j| {
                const code: u8 = if (mx) wb[col * k + j] else (wb[col * (k / 2) + j / 2] >> @as(u3, @intCast((j % 2) * 4))) & 15;
                const operand: f64 = @floatCast(@as(f16, @floatCast(x.slice(f32, rows * k)[row * k + j])));
                const weight: f64 = @floatCast(@as(f16, @floatCast(expertCode(code, mx))));
                expected += operand * weight;
                magnitude += @abs(operand * weight);
            }
            const bound = 4.0 * k * 0x1p-24 * magnitude + 8.0 * 0x1p-24 * @max(1.0, @abs(expected));
            try t.expect(@abs(@as(f64, @floatCast(out.slice(f32, rows * n)[row * n + col])) - expected) <= bound);
        };
        try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
    }
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
            }
        }
    }
}

const ExpertSlotArgs = extern struct { k: u32, n: u32, slots: u32, topk: u32, gate_cap: f32, up_cap: f32 };

test "per slot expert projections meet fp64 bounds" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    for ([_]bool{ false, true }) |mx| {
        const source = if (mx) @import("kernels.zig").prefillSource(true) else @import("kernels.zig").prefillSource(false);
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const codes = try device.buffer(256 * 4, mtl.ResourceOptions.shared);
        defer codes.deinit();
        for (codes.slice(f32, 256), 0..) |*v, i| v.* = @floatCast(expertCode(@intCast(if (mx and i % 128 == 127) 0 else i % (if (mx) @as(usize, 256) else 16)), mx));
        const globals = try device.buffer(16, mtl.ResourceOptions.shared);
        defer globals.deinit();
        globals.slice(f32, 1)[0] = 0.73;
        const den = try device.buffer(16, mtl.ResourceOptions.shared);
        defer den.deinit();
        den.slice(f32, 1)[0] = 1.3;
        const picks = try device.buffer(128 * 4, mtl.ResourceOptions.shared);
        defer picks.deinit();
        @memset(picks.slice(i32, 128), 0);
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
            const direct = try mtl.Pipeline.init(device, lib, if (gu) "i2p_gu" else "i2p_down", false);
            defer direct.deinit();
            for ([_]usize{ 1, 2, 4, 8 }) |rows| {
                const slots = rows * 8;
                const args = ExpertSlotArgs{ .k = @intCast(k), .n = n, .slots = @intCast(slots), .topk = 8, .gate_cap = if (gu) 7 else 0, .up_cap = if (gu) 7 else 0 };
                const cb = queue.commandBuffer();
                const enc = cb.compute(.serial);
                enc.setPipeline(direct);
                enc.setBuffer(x, 0, 0);
                enc.setBuffer(w, 0, 1);
                enc.setBuffer(scales, 0, 2);
                enc.setBuffer(globals, 0, 3);
                enc.setBuffer(den, 0, 4);
                enc.setBuffer(picks, 0, 5);
                enc.setValue(args, 6);
                enc.setBuffer(out, 0, 7);
                enc.setBuffer(codes, 0, 8);
                enc.dispatchGroups(mtl.Size.of(n, slots, 1), mtl.Size.of(32, 1, 1));
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| { std.debug.print("expert: {s}\n", .{msg}); return error.GpuFailed; }
                var worst: f64 = 0;
                for (0..slots) |slot| {
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
            }
        }
    }
}

const AttnArgs = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, base: u32, rows: u32, rmode: u32, scale: f32, floor: f32, alpha: f32 };

fn attnKey(j: usize, kv: usize, d: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast((j * 17 + kv * 7) % 101)) - 50)) * @as(f32, @floatFromInt(@as(i32, @intCast(d % 17)) - 8)) / 512;
}

fn attnValue(j: usize, kv: usize, d: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast((j * 13 + kv * 11) % 97)) - 48)) * @as(f32, @floatFromInt(@as(i32, @intCast(d % 17)) - 8)) / 512;
}

test "half attention uses rounded query key probability and value" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init(); defer device.deinit();
    const queue = try device.queue(); defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, "#define I2P_HALF_OPERANDS 1\n" ++ @import("kernels.zig").prefill, mtl.CompileOptions.mlx()); defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "i2p_attn_half", false); defer pipe.deinit();
    const heads = 3; const dim = 128;
    const q = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared); defer q.deinit();
    const k = try device.buffer(dim * 2, mtl.ResourceOptions.shared); defer k.deinit();
    const v = try device.buffer(dim * 2, mtl.ResourceOptions.shared); defer v.deinit();
    const bias = try device.buffer(heads * 4, mtl.ResourceOptions.shared); defer bias.deinit();
    const sink = try device.buffer(heads * 4, mtl.ResourceOptions.shared); defer sink.deinit();
    const out = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared); defer out.deinit();
    const overflow = try device.buffer(16, mtl.ResourceOptions.shared); defer overflow.deinit();
    @memset(q.slice(f32, heads * dim), 1.0006); @memset(k.slice(f16, dim), 0.01);
    @memset(v.slice(f16, dim), 0.5003); @memset(bias.slice(f32, heads), 0); @memset(sink.slice(f32, heads), 0);
    overflow.slice(u32, 1)[0] = 0;
    const args = AttnArgs{ .heads = heads, .kv = 1, .dim = dim, .rel = 0, .extent = 0, .window = 0, .base = 0, .rows = 1, .rmode = 0, .scale = 0.125, .floor = 128, .alpha = 0 };
    const cb = queue.commandBuffer(); const enc = cb.compute(.serial);
    enc.setPipeline(pipe); enc.setBuffer(q, 0, 0); enc.setBuffer(k, 0, 1); enc.setBuffer(v, 0, 2);
    enc.setBuffer(bias, 0, 3); enc.setBuffer(sink, 0, 4); enc.setValue(args, 5); enc.setBuffer(out, 0, 6);
    enc.setBuffer(k, 0, 7); enc.setBuffer(v, 0, 8); enc.setBuffer(overflow, 0, 9);
    enc.dispatchGroups(mtl.Size.of(heads, 1, 1), mtl.Size.of(128, 1, 1));
    enc.end(); cb.commit(); cb.wait();
    if (cb.failure()) |msg| { std.debug.print("half attention: {s}\n", .{msg}); return error.GpuFailed; }
    const qr: f64 = @floatCast(@as(f16, @floatCast(@as(f32, 1.0006))));
    const kr: f64 = @floatCast(@as(f16, @floatCast(@as(f32, 0.01))));
    const vr: f64 = @floatCast(@as(f16, @floatCast(@as(f32, 0.5003))));
    const weight = @exp(qr * kr * dim * @as(f64, args.scale));
    const norm = @max(1.0, weight);
    const rounded_weight: f64 = @floatCast(@as(f16, @floatCast(weight / norm)));
    const expected = rounded_weight * vr / (1.0 / norm + weight / norm);
    try t.expect(@abs(qr - @as(f64, @floatCast(@as(f32, 1.0006)))) > 0.0001);
    try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
    for (out.slice(f32, heads * dim)) |value| try t.expectApproxEqAbs(expected, @as(f64, @floatCast(value)), 0.0001);
}

test "half attention masks future and expired keys" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init(); defer device.deinit();
    const queue = try device.queue(); defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, "#define I2P_HALF_OPERANDS 1\n" ++ @import("kernels.zig").prefill, mtl.CompileOptions.mlx()); defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "i2p_attn_half", false); defer pipe.deinit();
    const heads = 3; const dim = 128; const rows = 35; const window = 3;
    const q = try device.buffer(rows * heads * dim * 4, mtl.ResourceOptions.shared); defer q.deinit();
    const k = try device.buffer(rows * dim * 2, mtl.ResourceOptions.shared); defer k.deinit();
    const v = try device.buffer(rows * dim * 2, mtl.ResourceOptions.shared); defer v.deinit();
    const bias = try device.buffer(rows * heads * 4, mtl.ResourceOptions.shared); defer bias.deinit();
    const sink = try device.buffer(rows * heads * 4, mtl.ResourceOptions.shared); defer sink.deinit();
    const out = try device.buffer(rows * heads * dim * 4, mtl.ResourceOptions.shared); defer out.deinit();
    const overflow = try device.buffer(16, mtl.ResourceOptions.shared); defer overflow.deinit();
    @memset(q.slice(f32, rows * heads * dim), 0); @memset(k.slice(f16, rows * dim), 0);
    @memset(bias.slice(f32, rows * heads), 0); @memset(sink.slice(f32, rows * heads), 0);
    for (0..rows) |row| @memset(v.slice(f16, rows * dim)[row * dim ..][0..dim], @floatCast(0.5003 + @as(f32, @floatFromInt(row)) * 0.125));
    overflow.slice(u32, 1)[0] = 0;
    const args = AttnArgs{ .heads = heads, .kv = 1, .dim = dim, .rel = 0, .extent = 0, .window = window, .base = 0, .rows = rows, .rmode = 1, .scale = 0.125, .floor = 128, .alpha = 0 };
    const cb = queue.commandBuffer(); const enc = cb.compute(.serial);
    enc.setPipeline(pipe); enc.setBuffer(q, 0, 0); enc.setBuffer(k, 0, 1); enc.setBuffer(v, 0, 2);
    enc.setBuffer(bias, 0, 3); enc.setBuffer(sink, 0, 4); enc.setValue(args, 5); enc.setBuffer(out, 0, 6);
    enc.setBuffer(k, 0, 7); enc.setBuffer(v, 0, 8); enc.setBuffer(overflow, 0, 9);
    enc.dispatchGroups(mtl.Size.of(heads, (rows + 31) / 32, 1), mtl.Size.of(128, 1, 1));
    enc.end(); cb.commit(); cb.wait();
    if (cb.failure()) |msg| { std.debug.print("half attention window: {s}\n", .{msg}); return error.GpuFailed; }
    for (0..rows) |row| {
        const first = if (row + 1 > window) row + 1 - window else 0;
        var total: f64 = 0;
        for (first..row + 1) |j| {
            const value: f32 = 0.5003 + @as(f32, @floatFromInt(j)) * 0.125;
            total += @as(f64, @floatCast(@as(f16, @floatCast(value))));
        }
        const expected = total / @as(f64, @floatFromInt(1 + row + 1 - first));
        for (out.slice(f32, rows * heads * dim)[row * heads * dim ..][0 .. heads * dim]) |value|
            try t.expectApproxEqAbs(expected, @as(f64, @floatCast(value)), 0.00001);
    }
    try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
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
            const keys = try device.buffer(cache_rows * kv * dim * 2, mtl.ResourceOptions.shared);
            defer keys.deinit();
            const values = try device.buffer(cache_rows * kv * dim * 2, mtl.ResourceOptions.shared);
            defer values.deinit();
            const start = if (window > 0 and base > window) base - window else 0;
            for (start..base) |j| for (0..kv) |h| for (0..dim) |d| {
                const at = ((if (window > 0) j % window else j) * kv + h) * dim + d;
                keys.slice(f16, cache_rows * kv * dim)[at] = @floatCast(attnKey(j, h, d));
                values.slice(f16, cache_rows * kv * dim)[at] = @floatCast(attnValue(j, h, d));
            };
            for ([_]usize{ 35, 1, 7, 9, 17, 33, 63, 100, 128 }) |rows| {
                if (base == 5000 and rows != 1 and rows != 7 and rows != 128) continue;
                const q = try device.buffer(rows * heads * dim * 4, mtl.ResourceOptions.shared); defer q.deinit();
                const kc = try device.buffer(rows * kv * dim * 2, mtl.ResourceOptions.shared); defer kc.deinit();
                const vc = try device.buffer(rows * kv * dim * 2, mtl.ResourceOptions.shared); defer vc.deinit();
                const bias = try device.buffer(rows * heads * (extent + 1) * 4, mtl.ResourceOptions.shared); defer bias.deinit();
                const sink = try device.buffer(rows * heads * 4, mtl.ResourceOptions.shared); defer sink.deinit();
                const got = try device.buffer(rows * heads * dim * 4, mtl.ResourceOptions.shared); defer got.deinit();
                const ref = try device.buffer(rows * heads * dim * 4, mtl.ResourceOptions.shared); defer ref.deinit();
                for (q.slice(f32, rows * heads * dim), 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + i / dim * 3) % 67)) - 33)) / 64;
                for (0..rows) |r| for (0..kv) |h| for (0..dim) |d| {
                    const at = (r * kv + h) * dim + d;
                    kc.slice(f16, rows * kv * dim)[at] = @floatCast(attnKey(base + r, h, d));
                    vc.slice(f16, rows * kv * dim)[at] = @floatCast(attnValue(base + r, h, d));
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

const serialRoute =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\struct Args { uint experts, topk, rows; float route_scale; };
    \\[[kernel]] void route(const device float* L [[buffer(0)]], const device float* bias [[buffer(1)]],
    \\    constant Args& a [[buffer(2)]], device int* picks [[buffer(3)]], device float* wts [[buffer(4)]],
    \\    uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    \\  if (lane != 0u || t >= a.rows) return;
    \\  const uint E = a.experts, K = a.topk;
    \\  const device float* l = L + (size_t)t * E;
    \\  float probs[128], sel[128];
    \\  for (uint e = 0; e < E; ++e) { probs[e] = 1.0f / (1.0f + precise::exp(-l[e])); sel[e] = probs[e] + bias[e]; }
    \\  float denom = 0.0f;
    \\  for (uint k = 0; k < K; ++k) {
    \\    uint best = E;
    \\    for (uint e = 0; e < E; ++e) {
    \\      if (isnan(sel[e])) continue;
    \\      if (best == E || sel[e] > sel[best]) best = e;
    \\    }
    \\    picks[(size_t)t * K + k] = int(best);
    \\    if (best != E) { denom += probs[best]; sel[best] = NAN; }
    \\  }
    \\  for (uint k = 0; k < K; ++k) wts[(size_t)t * K + k] = probs[uint(picks[(size_t)t * K + k])] / (denom + 1e-20f) * a.route_scale;
    \\}
;

const RouteArgs = extern struct { experts: u32, topk: u32, rows: u32, route_scale: f32 };

const RelArgs = extern struct { heads: u32, rel: u32, extent: u32, rows: u32, base: u32, tau: u32, floor: f32, alpha: f32 };

test "row-staged relative bias matches the per-output kernel bit for bit" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @import("kernels.zig").prefill, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const each = try mtl.Pipeline.init(device, lib, "i2p_relbias", false);
    defer each.deinit();
    const rows_pipe = try mtl.Pipeline.init(device, lib, "i2p_relbias_rows", false);
    defer rows_pipe.deinit();
    for ([_]RelArgs{
        .{ .heads = 24, .rel = 16, .extent = 1024, .rows = 37, .base = 9000, .tau = 1, .floor = 8192, .alpha = 0.1 },
        .{ .heads = 24, .rel = 16, .extent = 1024, .rows = 5, .base = 3, .tau = 0, .floor = 8192, .alpha = 0.1 },
        .{ .heads = 4, .rel = 4, .extent = 16, .rows = 9, .base = 20, .tau = 1, .floor = 8, .alpha = 0.25 },
    }) |args| {
        const cols: usize = args.extent + 1;
        const p_buf = try device.buffer(args.rel * args.extent * 2, mtl.ResourceOptions.shared);
        defer p_buf.deinit();
        for (p_buf.slice(u16, args.rel * args.extent), 0..) |*v, i| {
            const value: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 37) % 201)) - 100)) / 53.0;
            v.* = @truncate(@as(u32, @bitCast(value)) >> 16);
        }
        const r_buf = try device.buffer(args.rows * args.heads * args.rel * 4, mtl.ResourceOptions.shared);
        defer r_buf.deinit();
        for (r_buf.slice(f32, args.rows * args.heads * args.rel), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 29 + i / 16) % 173)) - 86)) / 41.0;
        var outs: [2]mtl.Buffer = undefined;
        for (&outs, 0..) |*out, side| {
            out.* = try device.buffer(args.rows * args.heads * cols * 4, mtl.ResourceOptions.shared);
            @memset(out.slice(f32, args.rows * args.heads * cols), std.math.nan(f32));
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(if (side == 0) each else rows_pipe);
            enc.setBuffer(p_buf, 0, 0);
            enc.setBuffer(r_buf, 0, 1);
            enc.setValue(args, 2);
            enc.setBuffer(out.*, 0, 3);
            if (side == 0) enc.dispatchThreads(mtl.Size.of(args.heads * cols, args.rows, 1), mtl.Size.of(256, 1, 1)) else enc.dispatchGroups(mtl.Size.of((cols + 255) / 256, args.rows, 1), mtl.Size.of(256, 1, 1));
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("relbias: {s}\n", .{msg});
                return error.GpuFailed;
            }
        }
        defer for (outs) |out| out.deinit();
        const want = outs[0].slice(f32, args.rows * args.heads * cols);
        for (want) |v| try t.expect(std.math.isFinite(v));
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(outs[1].slice(f32, args.rows * args.heads * cols)));
    }
}

test "parallel route selection matches serial picks and weights" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @import("kernels.zig").prefill, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const ref_lib = try mtl.Library.fromSource(device, serialRoute, mtl.CompileOptions.mlx());
    defer ref_lib.deinit();
    const candidate = try mtl.Pipeline.init(device, lib, "i2p_routepick", false);
    defer candidate.deinit();
    const reference = try mtl.Pipeline.init(device, ref_lib, "route", false);
    defer reference.deinit();
    const logits = try device.buffer(8 * 128 * 4, mtl.ResourceOptions.shared);
    defer logits.deinit();
    const bias = try device.buffer(128 * 4, mtl.ResourceOptions.shared);
    defer bias.deinit();
    const actual_picks = try device.buffer(8 * 128 * 4, mtl.ResourceOptions.shared);
    defer actual_picks.deinit();
    const expected_picks = try device.buffer(8 * 128 * 4, mtl.ResourceOptions.shared);
    defer expected_picks.deinit();
    const actual_weights = try device.buffer(8 * 128 * 4, mtl.ResourceOptions.shared);
    defer actual_weights.deinit();
    const expected_weights = try device.buffer(8 * 128 * 4, mtl.ResourceOptions.shared);
    defer expected_weights.deinit();
    for ([_]usize{ 96, 128 }) |experts| for ([_]usize{ 8, 24 }) |topk| for ([_]usize{ 1, 4, 8 }) |rows| {
        for (logits.slice(f32, rows * experts), 0..) |*v, i| v.* = if (i / experts == 0) 0 else @as(f32, @floatFromInt(@as(i32, @intCast((i * 17) % 37)) - 18)) / 8;
        for (bias.slice(f32, experts), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) / 16;
        const args = RouteArgs{ .experts = @intCast(experts), .topk = @intCast(topk), .rows = @intCast(rows), .route_scale = 8 };
        for ([_]mtl.Pipeline{ candidate, reference }, [_]mtl.Buffer{ actual_picks, expected_picks }, [_]mtl.Buffer{ actual_weights, expected_weights }) |pipe, picks, weights| {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(pipe);
            enc.setBuffer(logits, 0, 0);
            enc.setBuffer(bias, 0, 1);
            enc.setValue(args, 2);
            enc.setBuffer(picks, 0, 3);
            enc.setBuffer(weights, 0, 4);
            enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| { std.debug.print("route: {s}\n", .{msg}); return error.GpuFailed; }
        }
        try t.expectEqualSlices(i32, expected_picks.slice(i32, rows * topk), actual_picks.slice(i32, rows * topk));
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(expected_weights.slice(f32, rows * topk)), std.mem.sliceAsBytes(actual_weights.slice(f32, rows * topk)));
    };
}

test "wide half expert preserves routed gate and down products" {
    const device = try mtl.Device.init(); defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue(); defer queue.deinit();
    const source = "#define I2P_HALF_OPERANDS 1\n" ++ @import("kernels.zig").prefill ++ @import("kernels.zig").prefill_expert_wide;
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx()); defer lib.deinit();
    const narrow = try mtl.Pipeline.init(device, lib, "i2p_expert_matrix", false); defer narrow.deinit();
    const wide = try mtl.Pipeline.init(device, lib, "i2p_expert_matrix_wide", false); defer wide.deinit();
    const rows = 65; const n = 70; const k = 128;
    const x = try device.buffer(rows * k * 4, mtl.ResourceOptions.shared); defer x.deinit();
    for (x.slice(f32, rows * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + i / k * 11) % 97)) - 48)) / 31.0;
    const weights = try device.buffer(n * k, mtl.ResourceOptions.shared); defer weights.deinit();
    for (weights.slice(u8, n * k), 0..) |*v, i| {
        const lo: u8 = @intCast((i * 7 + i / (k / 2) * 3) % 16);
        const hi: u8 = @intCast((i * 11 + i / (k / 2) * 5) % 16);
        v.* = lo | (hi << 4);
    }
    const scales = try device.buffer(2 * n * (k / 16), mtl.ResourceOptions.shared); defer scales.deinit();
    @memset(scales.slice(u8, 2 * n * (k / 16)), 0x38);
    const global = try device.buffer(16, mtl.ResourceOptions.shared); defer global.deinit(); global.slice(f32, 1)[0] = 1;
    const off = try device.buffer(16, mtl.ResourceOptions.shared); defer off.deinit(); off.slice(i32, 2)[0] = 0; off.slice(i32, 2)[1] = rows;
    const route = try device.buffer(rows * 4, mtl.ResourceOptions.shared); defer route.deinit();
    for (route.slice(i32, rows), 0..) |*v, i| v.* = @intCast(rows - 1 - i);
    const reference = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared); defer reference.deinit();
    const actual = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared); defer actual.deinit();
    const codes = try device.buffer(16 * 4, mtl.ResourceOptions.shared); defer codes.deinit();
    for (codes.slice(f32, 16), 0..) |*v, i| v.* = @floatCast(expertCode(@intCast(i), false));
    const overflow = try device.buffer(16, mtl.ResourceOptions.shared); defer overflow.deinit();
    for ([_]bool{ false, true }) |up| {
        const args = ExpertArgs{ .k = k, .n = n, .slots = rows, .topk = 1, .experts = 1, .gate_cap = if (up) 4 else 0, .up_cap = if (up) 25 else 0 };
        for ([_]mtl.Pipeline{ narrow, wide }, [_]mtl.Buffer{ reference, actual }, [_]bool{ false, true }) |pipe, out, is_wide| {
            overflow.slice(u32, 1)[0] = 0;
            @memset(out.slice(f32, rows * n), std.math.nan(f32));
            const cb = queue.commandBuffer(); const enc = cb.compute(.serial);
            enc.setPipeline(pipe); enc.setBuffer(x, 0, 0); enc.setBuffer(weights, 0, 1); enc.setBuffer(scales, 0, 2);
            enc.setBuffer(global, 0, 3); enc.setBuffer(global, 0, 4); enc.setBuffer(off, 0, 5); enc.setBuffer(route, 0, 6);
            enc.setValue(args, 7); enc.setBuffer(out, 0, 8); enc.setBuffer(codes, 0, 9); enc.setBuffer(overflow, 0, 10);
            const columns: usize = if (up) 2 * n else n;
            enc.dispatchGroups(mtl.Size.of(if (is_wide) (columns + 63) / 64 else (columns + 31) / 32, 1, (rows + 31) / 32), mtl.Size.of(if (is_wide) 256 else 128, 1, 1));
            enc.end(); cb.commit(); cb.wait();
            if (cb.failure()) |_| return error.GpuFailed;
            try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
        }
        for (reference.slice(f32, rows * n), actual.slice(f32, rows * n)) |want, got| {
            try t.expect(std.math.isFinite(want) and std.math.isFinite(got));
            try t.expectApproxEqAbs(want, got, 0.0001);
        }
    }
}

test "wide expert and grouped fallback partition threshold occupancy" {
    const device = try mtl.Device.init(); defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue(); defer queue.deinit();
    const source = "#define I2P_HALF_OPERANDS 1\n#define I2P_MATRIX_THRESHOLD 14\n#define I2P_DOWN_MATRIX_THRESHOLD 8\n" ++ @import("kernels.zig").prefill ++ @import("kernels.zig").prefill_expert_wide;
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx()); defer lib.deinit();
    const wide = try mtl.Pipeline.init(device, lib, "i2p_expert_matrix_wide", false); defer wide.deinit();
    const low_up = try mtl.Pipeline.init(device, lib, "i2p_gu_grouped", false); defer low_up.deinit();
    const low_down = try mtl.Pipeline.init(device, lib, "i2p_down_grouped", false); defer low_down.deinit();
    const rows = 15; const n = 8; const k = 64;
    const x = try device.buffer(rows * k * 4, mtl.ResourceOptions.shared); defer x.deinit(); @memset(x.slice(f32, rows * k), 0.25);
    const weights = try device.buffer(n * k, mtl.ResourceOptions.shared); defer weights.deinit(); @memset(weights.slice(u8, n * k), 0x22);
    const scales = try device.buffer(2 * n * (k / 16), mtl.ResourceOptions.shared); defer scales.deinit(); @memset(scales.slice(u8, 2 * n * (k / 16)), 0x38);
    const global = try device.buffer(16, mtl.ResourceOptions.shared); defer global.deinit(); global.slice(f32, 1)[0] = 1;
    const off = try device.buffer(16, mtl.ResourceOptions.shared); defer off.deinit(); off.slice(i32, 2)[0] = 0;
    const route = try device.buffer(rows * 4, mtl.ResourceOptions.shared); defer route.deinit(); for (route.slice(i32, rows), 0..) |*v, i| v.* = @intCast(i);
    const out = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared); defer out.deinit();
    const codes = try device.buffer(16 * 4, mtl.ResourceOptions.shared); defer codes.deinit(); for (codes.slice(f32, 16), 0..) |*v, i| v.* = @floatCast(expertCode(@intCast(i), false));
    const overflow = try device.buffer(16, mtl.ResourceOptions.shared); defer overflow.deinit();
    for ([_]u32{ 7, 8, 13, 14, 15 }) |count| for ([_]bool{ false, true }) |up| for ([_]bool{ false, true }) |is_wide| {
        off.slice(i32, 2)[1] = @intCast(count);
        overflow.slice(u32, 1)[0] = 0;
        @memset(out.slice(f32, rows * n), std.math.nan(f32));
        const args = ExpertArgs{ .k = k, .n = n, .slots = count, .topk = 1, .experts = 1, .gate_cap = if (up) 4 else 0, .up_cap = if (up) 25 else 0 };
        const cb = queue.commandBuffer(); const enc = cb.compute(.serial);
        enc.setPipeline(if (is_wide) wide else if (up) low_up else low_down);
        enc.setBuffer(x, 0, 0); enc.setBuffer(weights, 0, 1); enc.setBuffer(scales, 0, 2); enc.setBuffer(global, 0, 3); enc.setBuffer(global, 0, 4);
        enc.setBuffer(off, 0, 5); enc.setBuffer(route, 0, 6); enc.setValue(args, 7); enc.setBuffer(out, 0, 8); enc.setBuffer(codes, 0, 9);
        if (is_wide) enc.setBuffer(overflow, 0, 10);
        enc.dispatchGroups(if (is_wide) mtl.Size.of(1, 1, 1) else mtl.Size.of(n, 1, 1), mtl.Size.of(if (is_wide) 256 else 32, 1, 1));
        enc.end(); cb.commit(); cb.wait(); if (cb.failure()) |_| return error.GpuFailed;
        const active = (count >= (if (up) @as(u32, 14) else 8)) == is_wide;
        const expected: f32 = if (up) 4 * std.math.tanh(@as(f32, 16) / 4) / (1 + @exp(@as(f32, -16))) * 25 * std.math.tanh(@as(f32, 16) / 25) else 16;
        for (out.slice(f32, rows * n)[0 .. @as(usize, count) * n]) |value| {
            if (active) try t.expectApproxEqAbs(expected, value, 0.001) else try t.expect(std.math.isNan(value));
        }
        try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
    };
}

test {
    _ = @import("prefill_staged_test.zig");
}
