const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;

const Dense = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, bits: u32, cap: f32 };
const Proj = extern struct { k: u32, full: u32, m: [8]u32 };
const SharedProj = extern struct { k: u32, full: u32, rows: u32, m: [8]u32 };
const Moe = extern struct { k: u32, n: u32, slots: u32, topk: u32, gate_cap: f32, up_cap: f32 };

const Kind = enum { head, oproj, proj, gate_up, down };
const widest = inkling.group.max_rows;
const slots_a_visit = widest * 8;
const Io = struct { x: mtl.Buffer, act: mtl.Buffer, y: mtl.Buffer, outs: [8]mtl.Buffer, picks: mtl.Buffer, visits: usize };
const Case = struct { pipe: mtl.Pipeline, groups: mtl.Size, threads: mtl.Size };

/// The served few-row kernels against the one-row kernels at 1 to 16 rows, each served output checked byte for byte.
pub fn run(gpa: std.mem.Allocator, m: *inkling.Model, reps: usize, picks_path: ?[]const u8, io_ctx: std.Io, out: *std.ArrayList(u8)) !void {
    const c = &m.config;
    var bs = try inkling.BatchScratch.init(m.device, c, 8);
    defer bs.deinit();
    const vocab: usize = c.text.unpadded_vocab;
    const x = try filled(m, widest * 3072, 1);
    defer x.deinit();
    const act = try filled(m, slots_a_visit * c.text.expert_inter, 2);
    defer act.deinit();
    const y = try m.device.buffer(widest * vocab * 4, mtl.ResourceOptions.private);
    defer y.deinit();
    var checks: [2]mtl.Buffer = undefined;
    for (&checks) |*b| b.* = try m.device.buffer(widest * vocab * 4, mtl.ResourceOptions.shared);
    defer for (checks) |b| b.deinit();
    var outs: [8]mtl.Buffer = undefined;
    for (&outs) |*b| b.* = try m.device.buffer(widest * 3072 * 4, mtl.ResourceOptions.private);
    defer for (outs) |b| b.deinit();
    const visits: usize = 48;
    const picks = try m.device.buffer(visits * slots_a_visit * 4, mtl.ResourceOptions.shared);
    defer picks.deinit();
    const pv = picks.slice(i32, visits * slots_a_visit);
    for (0..visits) |v| for (0..widest) |r| for (0..8) |j| {
        pv[v * slots_a_visit + r * 8 + j] = @intCast((r * 4 + j) % 32);
    };
    var picked_rows: usize = widest;
    if (picks_path) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io_ctx, path, gpa, .limited(4 << 20));
        defer gpa.free(bytes);
        const parsed = try std.json.parseFromSlice([]const i32, gpa, bytes, .{});
        defer parsed.deinit();
        if (parsed.value.len % (visits * 8) != 0 or parsed.value.len > visits * slots_a_visit) return error.BadPicks;
        const per = parsed.value.len / visits;
        picked_rows = per / 8;
        for (0..visits) |v| @memcpy(pv[v * slots_a_visit ..][0..per], parsed.value[v * per ..][0..per]);
    }
    const io = Io{ .x = x, .act = act, .y = y, .outs = outs, .picks = picks, .visits = visits };
    const affine = m.affine_pipes.?;
    const shared = &bs.shared;
    const text = &c.text;
    const local = localLayer(m, 0);
    const ms: [8]u32 = .{ @intCast(local.attn.q.shape[0]), @intCast(local.attn.k.shape[0]), @intCast(local.attn.v.shape[0]), @intCast(local.attn.r.shape[0]), @intCast(local.attn.gate.shape[0]), @intCast(local.attn.sink.shape[0]), 0, 0 };
    var total: usize = 0;
    for (ms) |n| total += n;
    const warm = [_]Case{
        .{ .pipe = affine.dense, .groups = mtl.Size.of((vocab + 7) / 8, 1, 1), .threads = mtl.Size.of(256, 1, 1) },
        .{ .pipe = affine.dense, .groups = mtl.Size.of((text.hidden + 7) / 8, 1, 1), .threads = mtl.Size.of(256, 1, 1) },
        .{ .pipe = affine.batch_projections, .groups = mtl.Size.of((total + 7) / 8, 1, 1), .threads = mtl.Size.of(256, 1, 1) },
        .{ .pipe = bs.rows.pipes.gu_nv, .groups = mtl.Size.of(text.expert_inter, 8, 1), .threads = mtl.Size.of(32, 1, 1) },
        .{ .pipe = bs.rows.pipes.down_nv, .groups = mtl.Size.of(text.hidden, 8, 1), .threads = mtl.Size.of(32, 1, 1) },
    };
    for (0..3) |_| for (warm, [_]Kind{ .head, .oproj, .proj, .gate_up, .down }) |cs, kind| {
        _ = try timeIt(m, reps, kind, cs, false, io, 1, io.y);
    };
    for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16 }) |rows| for ([_]Kind{ .head, .oproj, .proj, .gate_up, .down }) |kind| {
        if (rows > picked_rows) continue;
        const slots = rows * text.topk;
        const one: Case = switch (kind) {
            .head => .{ .pipe = affine.dense, .groups = mtl.Size.of((vocab + 7) / 8, rows, 1), .threads = mtl.Size.of(256, 1, 1) },
            .oproj => .{ .pipe = affine.dense, .groups = mtl.Size.of((text.hidden + 7) / 8, rows, 1), .threads = mtl.Size.of(256, 1, 1) },
            .proj => .{ .pipe = affine.batch_projections, .groups = mtl.Size.of((total + 7) / 8, rows, 1), .threads = mtl.Size.of(256, 1, 1) },
            .gate_up => .{ .pipe = bs.rows.pipes.gu_nv, .groups = mtl.Size.of(text.expert_inter, slots, 1), .threads = mtl.Size.of(32, 1, 1) },
            .down => .{ .pipe = bs.rows.pipes.down_nv, .groups = mtl.Size.of(text.hidden, slots, 1), .threads = mtl.Size.of(32, 1, 1) },
        };
        const d = switch (kind) {
            .head => shared.dense(rows, text.unpadded_vocab, text.hidden, 8),
            .oproj => shared.dense(rows, text.hidden, @intCast(local.attn.o.shape[1]), 8),
            .proj => shared.fused(rows, ms, text.hidden),
            .gate_up => shared.gateUp(rows, text.expert_inter, text.hidden, slots),
            .down => shared.downs(rows, text.hidden, text.expert_inter, slots),
        };
        const served: Case = if (d) |v| .{ .pipe = v.pipe, .groups = v.groups, .threads = v.threads } else one;
        var times: [2][5]f64 = undefined;
        for (0..5) |round| for ([_]Case{ one, served }, 0..) |cs, side| {
            times[side][round] = try timeIt(m, reps, kind, cs, side == 1 and d != null, io, rows, io.y);
        };
        for (&times) |*t| std.mem.sort(f64, t, {}, std.sort.asc(f64));
        const equal = if (kind == .proj or d == null) true else try sameOut(m, kind, one, served, io, rows, checks);
        try out.print(gpa, "{{\"kind\":\"kernel\",\"rows\":{d},\"op\":\"{s}\",\"kernel\":\"served\",\"one_row_us\":{d:.2},\"kernel_us\":{d:.2},\"shared_kernel\":{},\"equal\":{}}}\n", .{ rows, @tagName(kind), times[0][2], times[1][2], d != null, equal });
        std.debug.print("kernel rows={d} {s}: one-row {d:.1} us, served {d:.1} us{s}\n", .{ rows, @tagName(kind), times[0][2], times[1][2], if (equal) "" else " BYTES DIFFER" });
        if (!equal) return error.KernelDiffers;
    };
}

fn sameOut(m: *inkling.Model, kind: Kind, one: Case, cs: Case, io: Io, rows: u32, checks: [2]mtl.Buffer) !bool {
    for ([_]Case{ one, cs }, [_]bool{ false, true }, checks) |case, s, buf| _ = try timeIt(m, 1, kind, case, s, io, rows, buf);
    const n: usize = switch (kind) {
        .head => m.config.text.unpadded_vocab * @as(usize, rows),
        .oproj => m.config.text.hidden * @as(usize, rows),
        .gate_up => m.config.text.expert_inter * @as(usize, rows) * 8,
        .down => m.config.text.hidden * @as(usize, rows) * 8,
        .proj => 0,
    };
    return std.mem.eql(u8, std.mem.sliceAsBytes(checks[0].slice(f32, n)), std.mem.sliceAsBytes(checks[1].slice(f32, n)));
}

fn filled(m: *inkling.Model, count: usize, seed: u64) !mtl.Buffer {
    const buf = try m.device.buffer(count * 4, mtl.ResourceOptions.shared);
    var rng = std.Random.DefaultPrng.init(seed);
    for (buf.slice(f32, count)) |*v| v.* = rng.random().float(f32) * 2 - 1;
    return buf;
}

fn localLayer(m: *inkling.Model, i: usize) *const inkling.Layer {
    var seen: usize = 0;
    while (true) for (m.weights.layers, 0..) |*lay, index| {
        if (m.config.isFull(@intCast(index))) continue;
        if (seen == i % 20) return lay;
        seen += 1;
    };
}

fn encode(enc: mtl.ComputeEncoder, m: *inkling.Model, kind: Kind, i: usize, shared: bool, io: Io, rows: u32, y: mtl.Buffer) void {
    const text = &m.config.text;
    switch (kind) {
        .head, .oproj => {
            const w = if (kind == .head) m.weights.unembed else m.weights.layers[i % m.weights.layers.len].attn.o;
            const q = w.affine.?;
            enc.setBuffer(io.x, 0, 0);
            enc.setBuffer(w.buffer, w.offset, 1);
            enc.setBuffer(q.scales, q.scale_offset, 2);
            enc.setBuffer(q.biases, q.bias_offset, 3);
            enc.setValue(Dense{ .n = if (kind == .head) text.unpadded_vocab else @intCast(w.shape[0]), .k = @intCast(w.shape[1]), .rows = rows, .xs = 1, .xb = 0, .bits = 8, .cap = if (kind == .head) text.logit_cap else 0 }, 4);
            enc.setBuffer(y, 0, 5);
        },
        .proj => {
            const a = &localLayer(m, i).attn;
            const qf = a.q.affine.?;
            for (0..3) |j| enc.setBuffer(io.x, 0, j);
            for (0..3) |j| {
                enc.setBuffer(a.q.buffer, a.q.offset, 3 + 3 * j);
                enc.setBuffer(qf.scales, qf.scale_offset, 4 + 3 * j);
                enc.setBuffer(qf.biases, qf.bias_offset, 5 + 3 * j);
            }
            for (io.outs, 0..) |b, j| enc.setBuffer(b, 0, 12 + j);
            const ms: [8]u32 = .{ @intCast(a.q.shape[0]), @intCast(a.k.shape[0]), @intCast(a.v.shape[0]), @intCast(a.r.shape[0]), @intCast(a.gate.shape[0]), @intCast(a.sink.shape[0]), 0, 0 };
            if (shared) enc.setValue(SharedProj{ .k = text.hidden, .full = 0, .rows = rows, .m = ms }, 20) else enc.setValue(Proj{ .k = text.hidden, .full = 0, .m = ms }, 20);
        },
        .gate_up, .down => {
            const lay = &m.weights.layers[i % 20];
            const up = kind == .gate_up;
            enc.setBuffer(if (up) io.x else io.act, 0, 0);
            for ([_]inkling.Tensor{ if (up) lay.w13 else lay.w2, if (up) lay.w13_scale else lay.w2_scale, if (up) lay.w13_global.? else lay.w2_global.?, if (up) lay.w13_den.? else lay.w2_den.? }, 1..) |t, j| enc.setBuffer(t.buffer, t.offset, j);
            enc.setBuffer(io.picks, (i % io.visits) * slots_a_visit * 4, 5);
            enc.setValue(Moe{ .k = if (up) text.hidden else text.expert_inter, .n = if (up) text.expert_inter else text.hidden, .slots = rows * 8, .topk = 8, .gate_cap = if (up) text.gate_cap else 0, .up_cap = if (up) text.up_cap else 0 }, 6);
            enc.setBuffer(y, 0, 7);
            enc.setBuffer(m.t16, 0, 8);
        },
    }
}

fn timeIt(m: *inkling.Model, reps: usize, kind: Kind, cs: Case, shared: bool, io: Io, rows: u32, y: mtl.Buffer) !f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = cb.compute(.serial);
    for (0..reps) |i| {
        enc.setPipeline(cs.pipe);
        encode(enc, m, kind, i, shared, io, rows, y);
        enc.dispatchGroups(cs.groups, cs.threads);
    }
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("kernel bench: {s}", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(reps));
}

const spec = @import("dflash_spec.zig");
const Phases = @TypeOf(@as(inkling.PrefillScratch, undefined).phases);
const Phase = Phases.Key;

pub const Prompt = struct { name: []const u8, ids: []const u32 };

fn phaseSet(list: []const Phase) Phases {
    var set = Phases.initOne(.edges);
    for (list) |p| set.insert(p);
    return set;
}

fn without(list: []const Phase) Phases {
    var set = Phases.full;
    for (list) |p| set.remove(p);
    return set;
}

fn setName(set: Phases, buf: []u8) ![]const u8 {
    if (set.eql(.full)) return "all";
    var w = std.Io.Writer.fixed(buf);
    const left = Phases.full.differenceWith(set);
    if (left.count() < set.count()) {
        try w.writeAll("all");
        var it = left.iterator();
        while (it.next()) |p| try w.print("-{s}", .{@tagName(p)});
        return w.buffered();
    }
    var it = set.iterator();
    var first = true;
    while (it.next()) |p| {
        try w.print("{s}{s}", .{ if (first) "" else "+", @tagName(p) });
        first = false;
    }
    return w.buffered();
}

pub fn phases(gpa: std.mem.Allocator, m: *inkling.Model, prompts: []const Prompt, reps: usize, widths: []const usize, out: *std.ArrayList(u8)) !void {
    var verifier = try inkling.group.Verifier.init(m.device, &m.config);
    defer verifier.deinit();
    var bs = try inkling.group.scratch(m.device, &m.config);
    defer bs.deinit();
    var ps = try inkling.PrefillScratch.init(m.device, &m.config, spec.prefill_chunk);
    defer ps.deinit();
    var rollback = try inkling.Cache.Rollback.initRows(m.device, &m.config, inkling.group.max_rows);
    defer rollback.deinit();
    const vocab: usize = m.config.text.unpadded_vocab;
    const logits = try gpa.alloc(f32, vocab);
    defer gpa.free(logits);
    var sets: [2 * Phases.len + 6]Phases = undefined;
    sets[0] = phaseSet(&.{});
    var count: usize = 1;
    var all = Phases.full;
    var each = all.iterator();
    while (each.next()) |p| if (p != .edges) {
        sets[count] = phaseSet(&.{p});
        sets[count + 1] = without(&.{p});
        count += 2;
    };
    sets[count] = phaseSet(&.{ .router, .experts });
    sets[count + 1] = phaseSet(&.{ .router, .affine_experts });
    sets[count + 2] = without(&.{ .router, .experts });
    sets[count + 3] = without(&.{ .experts, .affine_experts });
    sets[count + 4] = without(&.{ .router, .experts, .affine_experts });
    sets[count + 5] = .full;
    count += 6;
    const samples = try gpa.alloc(std.ArrayList(f64), count);
    defer gpa.free(samples);
    for (samples) |*s| s.* = .empty;
    defer for (samples) |*s| s.deinit(gpa);
    defer bs.rows.phases = .full;
    for (prompts) |prompt| {
        var cache = try inkling.Cache.init(m.device, &m.config, prompt.ids.len + inkling.group.max_rows);
        defer cache.deinit();
        const first = try spec.prefillPlain(m, &ps, &cache, prompt.ids);
        var rows: [inkling.group.max_rows]u32 = undefined;
        rows[0] = first;
        try rollback.copy(m.queue, &m.config, &cache, true);
        for (0..rows.len - 1) |i| {
            try inkling.batchStep(m, &bs, &.{&cache}, &.{rows[i]}, logits);
            rows[i + 1] = spec.argmax(logits);
        }
        try rollback.copy(m.queue, &m.config, &cache, false);
        for (widths) |width| {
            for (samples) |*s| s.clearRetainingCapacity();
            for (0..reps + 1) |rep| for (0..count) |k| {
                const i = (k + rep) % count;
                bs.rows.phases = .full;
                _ = try phaseStep(m, &verifier, &bs, &cache, &rollback, rows[0..@max(width, 1)], width == 0, logits);
                bs.rows.phases = sets[i];
                const ms = try phaseStep(m, &verifier, &bs, &cache, &rollback, rows[0..@max(width, 1)], width == 0, logits);
                if (rep > 0) try samples[i].append(gpa, ms);
            };
            bs.rows.phases = .full;
            for (sets[0..count], samples) |set, *list| {
                std.mem.sort(f64, list.items, {}, std.sort.asc(f64));
                var name: [256]u8 = undefined;
                const label = try setName(set, &name);
                const p50 = list.items[list.items.len / 2];
                try out.print(gpa, "{{\"kind\":\"phase\",\"prompt\":\"{s}\",\"context\":{d},\"width\":{d},\"path\":\"{s}\",\"set\":\"{s}\",\"reps\":{d},\"gpu_ms_p50\":{d:.4},\"gpu_ms_min\":{d:.4},\"gpu_ms_max\":{d:.4}}}\n", .{ prompt.name, cache.len, @max(width, 1), if (width == 0) "batch_step" else "grouped", label, reps, p50, list.items[0], list.items[list.items.len - 1] });
                std.debug.print("phases {s}: context={d} width={d} {s}: gpu p50={d:.3} ms\n", .{ prompt.name, cache.len, width, label, p50 });
            }
        }
    }
}

fn phaseStep(m: *inkling.Model, verifier: *inkling.group.Verifier, bs: *inkling.BatchScratch, cache: *inkling.Cache, rollback: *inkling.Cache.Rollback, rows: []const u32, single: bool, logits: []f32) !f64 {
    if (single) {
        try rollback.copy(m.queue, &m.config, cache, true);
        try inkling.batchStep(m, bs, &.{cache}, rows[0..1], logits);
        const ms = bs.rows.gpu_seconds * 1e3;
        try rollback.copy(m.queue, &m.config, cache, false);
        return ms;
    }
    const g = try verifier.group(&.{cache}, &.{rows.len});
    try verifier.forward(m, bs, &g, rows);
    return bs.rows.gpu_seconds * 1e3;
}

fn digestBuffer(m: *inkling.Model, buf: mtl.Buffer, len: usize) ![32]u8 {
    const host = try m.device.buffer(@max(len, 16), mtl.ResourceOptions.shared);
    defer host.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    blit.copy(buf, 0, host, 0, len);
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(host.contents()[0..len], &out, .{});
    return out;
}

fn digestBytes(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

fn hashLine(gpa: std.mem.Allocator, out: *std.ArrayList(u8), prompt: []const u8, what: []const u8, width: usize, digest: [32]u8) !void {
    try out.print(gpa, "{{\"kind\":\"hash\",\"prompt\":\"{s}\",\"what\":\"{s}\",\"width\":{d},\"sha256\":\"{s}\"}}\n", .{ prompt, what, width, &std.fmt.bytesToHex(digest, .lower) });
}

pub fn hashes(gpa: std.mem.Allocator, m: *inkling.Model, prompts: []const Prompt, widths: []const usize, out: *std.ArrayList(u8)) !void {
    var verifier = try inkling.group.Verifier.init(m.device, &m.config);
    defer verifier.deinit();
    var bs = try inkling.group.scratch(m.device, &m.config);
    defer bs.deinit();
    var ps = try inkling.PrefillScratch.init(m.device, &m.config, spec.prefill_chunk);
    defer ps.deinit();
    var rollback = try inkling.Cache.Rollback.initRows(m.device, &m.config, inkling.group.max_rows);
    defer rollback.deinit();
    const vocab: usize = m.config.text.unpadded_vocab;
    const steps = try gpa.alloc(f32, inkling.group.max_rows * vocab);
    defer gpa.free(steps);
    for (prompts) |prompt| {
        var head = try inkling.Cache.init(m.device, &m.config, spec.prefill_chunk);
        defer head.deinit();
        const take = @min(prompt.ids.len, spec.prefill_chunk);
        try inkling.prefillWant(m, &ps, &head, prompt.ids[0..take], .all);
        try hashLine(gpa, out, prompt.name, "prefill_chunk_logits", take, digestBytes(std.mem.sliceAsBytes(ps.logits.slice(f32, take * vocab))));
        var cache = try inkling.Cache.init(m.device, &m.config, prompt.ids.len + inkling.group.max_rows);
        defer cache.deinit();
        const first = try spec.prefillPlain(m, &ps, &cache, prompt.ids);
        try hashLine(gpa, out, prompt.name, "prefill_last_logits", prompt.ids.len, digestBytes(std.mem.sliceAsBytes(ps.logits.slice(f32, vocab))));
        const names = [_][]const u8{ "prefill_keys", "prefill_values", "prefill_k_conv", "prefill_v_conv", "prefill_attn_conv", "prefill_mlp_conv" };
        for ([_]mtl.Buffer{ cache.keys, cache.values, cache.k_conv, cache.v_conv, cache.attn_conv, cache.mlp_conv }, names) |buf, name| try hashLine(gpa, out, prompt.name, name, prompt.ids.len, try digestBuffer(m, buf, buf.length()));
        var rows: [inkling.group.max_rows]u32 = undefined;
        rows[0] = first;
        try rollback.copy(m.queue, &m.config, &cache, true);
        for (0..rows.len) |i| {
            try inkling.batchStep(m, &bs, &.{&cache}, &.{rows[i]}, steps[i * vocab ..][0..vocab]);
            if (i + 1 < rows.len) rows[i + 1] = spec.argmax(steps[i * vocab ..][0..vocab]);
        }
        try hashLine(gpa, out, prompt.name, "step_logits", rows.len, digestBytes(std.mem.sliceAsBytes(steps)));
        try hashLine(gpa, out, prompt.name, "step_tokens", rows.len, digestBytes(std.mem.sliceAsBytes(&rows)));
        const names2 = [_][]const u8{ "step_keys", "step_values", "step_k_conv", "step_v_conv", "step_attn_conv", "step_mlp_conv" };
        for ([_]mtl.Buffer{ cache.keys, cache.values, cache.k_conv, cache.v_conv, cache.attn_conv, cache.mlp_conv }, names2) |buf, name| try hashLine(gpa, out, prompt.name, name, rows.len, try digestBuffer(m, buf, buf.length()));
        try rollback.copy(m.queue, &m.config, &cache, false);
        for (widths) |width| {
            if (width == 0) continue;
            const g = try verifier.group(&.{&cache}, &.{width});
            try verifier.forward(m, &bs, &g, rows[0..width]);
            const got = bs.rows.logits.slice(f32, width * vocab);
            try hashLine(gpa, out, prompt.name, "verify_logits", width, digestBytes(std.mem.sliceAsBytes(got)));
            const equal = std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(steps[0 .. width * vocab]));
            try out.print(gpa, "{{\"kind\":\"exact\",\"prompt\":\"{s}\",\"context\":{d},\"width\":{d},\"equal_steps\":{}}}\n", .{ prompt.name, cache.len, width, equal });
            std.debug.print("hashes {s}: width={d} verify rows equal one-row steps: {}\n", .{ prompt.name, width, equal });
            if (!equal) return error.GroupedDiffersFromPlain;
        }
    }
}
