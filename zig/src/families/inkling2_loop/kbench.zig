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

/// A candidate few-row kernel in `rows.source`: simdgroups, 8-row weight tiles and batch halves a group, its widths.
const Variant = struct { kind: Kind, name: [:0]const u8, sg: u32, t: u32, h: u32 = 1, j: u32 = 1, least: u32, most: u32 };

const variants = [_]Variant{
    .{ .kind = .gate_up, .name = "i2e_gate_up_f3", .sg = 4, .t = 2, .least = 2, .most = 8 },
    .{ .kind = .gate_up, .name = "i2e_gate_up_f4", .sg = 4, .t = 2, .least = 2, .most = 8 },
    .{ .kind = .gate_up, .name = "i2e_gate_up_r4", .sg = 4, .t = 1, .least = 2, .most = 8 },
    .{ .kind = .gate_up, .name = "i2e_gate_up_r4_t2", .sg = 4, .t = 2, .least = 2, .most = 8 },
    .{ .kind = .gate_up, .name = "i2e_gate_up_r4_s8", .sg = 8, .t = 2, .least = 2, .most = 8 },
    .{ .kind = .gate_up, .name = "i2e_gate_up16_f3", .sg = 4, .t = 2, .least = 2, .most = 16 },
    .{ .kind = .gate_up, .name = "i2e_gate_up16_f4", .sg = 4, .t = 2, .least = 2, .most = 16 },
    .{ .kind = .gate_up, .name = "i2e_gate_up16_r4", .sg = 4, .t = 1, .least = 2, .most = 16 },
    .{ .kind = .gate_up, .name = "i2e_gate_up16_r4_t2", .sg = 4, .t = 2, .least = 2, .most = 16 },
    .{ .kind = .gate_up, .name = "i2e_gate_up16_r4_s8", .sg = 8, .t = 2, .least = 2, .most = 16 },
    .{ .kind = .gate_up, .name = "i2e_gate_up16_f3_j2", .sg = 4, .t = 2, .j = 2, .least = 2, .most = 16 },
    .{ .kind = .down, .name = "i2e_down_f1", .sg = 8, .t = 4, .least = 1, .most = 8 },
    .{ .kind = .down, .name = "i2e_down_f3", .sg = 8, .t = 4, .least = 1, .most = 8 },
    .{ .kind = .down, .name = "i2e_down_r4", .sg = 8, .t = 2, .least = 1, .most = 8 },
    .{ .kind = .down, .name = "i2e_down_r4_t4", .sg = 8, .t = 4, .least = 1, .most = 8 },
    .{ .kind = .down, .name = "i2e_down16_f3", .sg = 8, .t = 4, .least = 1, .most = 16 },
    .{ .kind = .down, .name = "i2e_down16_r4", .sg = 8, .t = 2, .least = 1, .most = 16 },
    .{ .kind = .down, .name = "i2e_down16_r4_t4", .sg = 8, .t = 4, .least = 1, .most = 16 },
    .{ .kind = .down, .name = "i2e_down16_f3_j2", .sg = 8, .t = 4, .j = 2, .least = 1, .most = 16 },
    .{ .kind = .head, .name = "i2t_dense_8_2", .sg = 8, .t = 2, .least = 2, .most = 8 },
    .{ .kind = .head, .name = "i2t_dense_16_2", .sg = 16, .t = 2, .least = 2, .most = 8 },
    .{ .kind = .head, .name = "i2t_dense_8_4", .sg = 8, .t = 4, .least = 2, .most = 8 },
    .{ .kind = .head, .name = "i2t_dense_16_4", .sg = 16, .t = 4, .least = 2, .most = 8 },
    .{ .kind = .head, .name = "i2t_dense16_8_2", .sg = 8, .t = 2, .least = 9, .most = 16 },
    .{ .kind = .head, .name = "i2t_dense16_16_2", .sg = 16, .t = 2, .least = 9, .most = 16 },
    .{ .kind = .head, .name = "i2t_dense16_16_4", .sg = 16, .t = 4, .least = 9, .most = 16 },
    .{ .kind = .head, .name = "i2t_dense16_8_1", .sg = 8, .t = 1, .least = 9, .most = 16 },
    .{ .kind = .head, .name = "i2t_dense16_h8_2", .sg = 8, .t = 2, .h = 2, .least = 9, .most = 16 },
    .{ .kind = .head, .name = "i2t_dense16_h16_2", .sg = 16, .t = 2, .h = 2, .least = 9, .most = 16 },
    .{ .kind = .oproj, .name = "i2t_dense_8_2", .sg = 8, .t = 2, .least = 2, .most = 8 },
    .{ .kind = .oproj, .name = "i2t_dense_16_2", .sg = 16, .t = 2, .least = 2, .most = 8 },
    .{ .kind = .oproj, .name = "i2t_dense_8_4", .sg = 8, .t = 4, .least = 2, .most = 8 },
    .{ .kind = .oproj, .name = "i2t_dense_16_4", .sg = 16, .t = 4, .least = 2, .most = 8 },
    .{ .kind = .oproj, .name = "i2t_dense16_8_2", .sg = 8, .t = 2, .least = 9, .most = 16 },
    .{ .kind = .oproj, .name = "i2t_dense16_16_2", .sg = 16, .t = 2, .least = 9, .most = 16 },
    .{ .kind = .oproj, .name = "i2t_dense16_16_4", .sg = 16, .t = 4, .least = 9, .most = 16 },
    .{ .kind = .oproj, .name = "i2t_dense16_8_1", .sg = 8, .t = 1, .least = 9, .most = 16 },
    .{ .kind = .oproj, .name = "i2t_dense16_h8_2", .sg = 8, .t = 2, .h = 2, .least = 9, .most = 16 },
    .{ .kind = .oproj, .name = "i2t_dense16_h16_2", .sg = 16, .t = 2, .h = 2, .least = 9, .most = 16 },
    .{ .kind = .proj, .name = "i2t_projections_8_2", .sg = 8, .t = 2, .least = 3, .most = 8 },
    .{ .kind = .proj, .name = "i2t_projections_16_2", .sg = 16, .t = 2, .least = 3, .most = 8 },
    .{ .kind = .proj, .name = "i2t_projections_16_4", .sg = 16, .t = 4, .least = 3, .most = 8 },
    .{ .kind = .proj, .name = "i2t_projections16_8_2", .sg = 8, .t = 2, .least = 9, .most = 16 },
    .{ .kind = .proj, .name = "i2t_projections16_16_2", .sg = 16, .t = 2, .least = 9, .most = 16 },
    .{ .kind = .proj, .name = "i2t_projections16_16_4", .sg = 16, .t = 4, .least = 9, .most = 16 },
    .{ .kind = .proj, .name = "i2t_projections16_h8_2", .sg = 8, .t = 2, .h = 2, .least = 9, .most = 16 },
    .{ .kind = .proj, .name = "i2t_projections16_h16_2", .sg = 16, .t = 2, .h = 2, .least = 9, .most = 16 },
};

/// `chosen`: "all", "none" or comma-separated variant names to time beside the one-row and the served kernels.
pub fn run(gpa: std.mem.Allocator, m: *inkling.Model, reps: usize, picks_path: ?[]const u8, chosen: []const u8, io_ctx: std.Io, out: *std.ArrayList(u8)) !void {
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
    const lib = try mtl.Library.fromSource(m.device, inkling.rows.source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var pipes: [variants.len]?mtl.Pipeline = @splat(null);
    defer for (pipes) |p| if (p) |pipe| pipe.deinit();
    for (variants, 0..) |v, i| if (picked(chosen, v.name)) {
        pipes[i] = mtl.Pipeline.init(m.device, lib, v.name, false) catch |e| blk: {
            std.debug.print("kernel bench: {s} skipped: {t}\n", .{ v.name, e });
            break :blk null;
        };
    };
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
        var cases: [2 + variants.len]Case = undefined;
        var names: [2 + variants.len][]const u8 = undefined;
        cases[0] = one;
        names[0] = "one-row";
        cases[1] = if (d) |v| .{ .pipe = v.pipe, .groups = v.groups, .threads = v.threads } else one;
        names[1] = "served";
        var count: usize = 2;
        for (variants, pipes) |v, p| if (p) |pipe| if (v.kind == kind and rows >= v.least and rows <= v.most) {
            const n: usize = switch (kind) {
                .head => vocab,
                .oproj => text.hidden,
                .gate_up => 2 * @as(usize, text.expert_inter),
                .down => text.hidden,
                .proj => 0,
            };
            var blocks: usize = (n + 8 * v.t - 1) / (8 * v.t);
            if (kind == .proj) {
                blocks = 0;
                for (ms) |part| blocks += (part + 8 * v.t - 1) / (8 * v.t);
            }
            const moe = kind == .gate_up or kind == .down;
            cases[count] = .{ .pipe = pipe, .groups = mtl.Size.of(blocks * v.h, if (moe) (slots + v.j - 1) / v.j else 1, 1), .threads = mtl.Size.of(32 * v.sg, 1, 1) };
            names[count] = v.name;
            count += 1;
        };
        var times: [2 + variants.len][5]f64 = undefined;
        for (0..5) |round| for (cases[0..count], 0..) |cs, side| {
            times[side][round] = try timeIt(m, reps, kind, cs, side > 0 and (side > 1 or d != null), io, rows, io.y);
        };
        for (times[0..count]) |*t| std.mem.sort(f64, t, {}, std.sort.asc(f64));
        for (cases[1..count], names[1..count], times[1..count], 1..) |cs, name, t, side| {
            const equal = if (kind == .proj) true else try sameOut(m, kind, one, cs, side > 1 or d != null, io, rows, checks);
            try out.print(gpa, "{{\"kind\":\"kernel\",\"rows\":{d},\"op\":\"{s}\",\"kernel\":\"{s}\",\"one_row_us\":{d:.2},\"kernel_us\":{d:.2},\"equal\":{}}}\n", .{ rows, @tagName(kind), name, times[0][2], t[2], equal });
            std.debug.print("kernel rows={d} {s}: one-row {d:.1} us, {s} {d:.1} us{s}\n", .{ rows, @tagName(kind), times[0][2], name, t[2], if (equal) "" else " BYTES DIFFER" });
            if (!equal) return error.KernelDiffers;
        }
    };
}

fn picked(chosen: []const u8, name: []const u8) bool {
    if (std.mem.eql(u8, chosen, "all")) return true;
    var it = std.mem.splitScalar(u8, chosen, ',');
    while (it.next()) |part| if (std.mem.eql(u8, part, name)) return true;
    return false;
}

fn sameOut(m: *inkling.Model, kind: Kind, one: Case, cs: Case, shared: bool, io: Io, rows: u32, checks: [2]mtl.Buffer) !bool {
    for ([_]Case{ one, cs }, [_]bool{ false, shared }, checks) |case, s, buf| _ = try timeIt(m, 1, kind, case, s, io, rows, buf);
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
