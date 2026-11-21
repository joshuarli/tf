const std = @import("std");
const mtl = @import("metal");
const fam = @import("family.zig");
const model = @import("model.zig");

const t = std.testing;

fn checkpoint(a: std.mem.Allocator) ![]const u8 {
    if (std.c.getenv("TF_INKLING2_LOOP_CKPT")) |p| return std.mem.span(p);
    const home = std.c.getenv("HOME") orelse return error.SkipZigTest;
    return try std.fmt.allocPrint(a, "{s}/models/Inkling-2-Mini@ed40e7da625f", .{std.mem.span(home)});
}

fn bf16ToF64(b: u16) f64 {
    const f: f32 = @bitCast(@as(u32, b) << 16);
    return @floatCast(f);
}

const Check = struct {
    worst_ratio: f64 = 0,
    worst_err: f64 = 0,
    failed: usize = 0,
    n: usize = 0,

    fn add(self: *Check, got: f32, want: f64, bound: f64) void {
        self.n += 1;
        const err = @abs(@as(f64, got) - want);
        if (err > self.worst_err) self.worst_err = err;
        const ratio = err / (bound + 1e-30);
        if (ratio > self.worst_ratio) self.worst_ratio = ratio;
        if (err > bound) self.failed += 1;
    }
};

fn boundOf(terms: f64, y: f64, k: usize) f64 {
    const u = std.math.ldexp(@as(f64, 1), -24);
    const gamma = @as(f64, @floatFromInt(k + 64)) * u;
    return 4.0 * gamma * terms + 2.0 * u * @abs(y);
}

fn fillX(xs: []f32) void {
    var s: u64 = 0x9e3779b97f4a7c15;
    for (xs) |*v| {
        s = s *% 6364136223846793005 +% 1442695040888963407;
        v.* = @as(f32, @floatFromInt((s >> 33) % 2000)) / 1000.0 - 1.0;
    }
}

fn weightBytes(w: model.Tensor) []const u8 {
    return w.buffer.contents()[w.offset..][0..w.bytes];
}

fn refRows(w: model.Tensor, k: usize, xs: []const f32, rows: []const usize, out: []f64, bound: []f64) void {
    const bytes = weightBytes(w);
    for (rows, 0..) |r, i| {
        var y: f64 = 0;
        var terms: f64 = 0;
        for (0..k) |c| {
            const b = std.mem.readInt(u16, bytes[(r * k + c) * 2 ..][0..2], .little);
            const wv = bf16ToF64(b);
            const xv: f64 = xs[c];
            y += wv * xv;
            terms += @abs(wv * xv);
        }
        out[i] = y;
        bound[i] = boundOf(terms, y, k);
    }
}

const MvArgs = extern struct { m: u32, k: u32, cap: f32 };
const FusedArgs = extern struct { k: u32, m0: u32, m1: u32, m2: u32, m3: u32 };
const FusedxArgs = extern struct { k0: u32, k1: u32, k2: u32, m: [8]u32, x: [8]u32 };

fn runMv(m: *fam.Model, pipe: mtl.Pipeline, args: MvArgs, x: mtl.Buffer, w: model.Tensor, o: mtl.Buffer, groups: usize, threads: usize) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipe);
    enc.setBuffer(x, 0, 0);
    enc.setBuffer(w.buffer, w.offset, 1);
    enc.setValue(args, 2);
    enc.setBuffer(o, 0, 3);
    enc.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(threads, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("dense_test mv: {s}", .{msg});
        return error.GpuFailed;
    }
}

fn runFused(m: *fam.Model, pipe: mtl.Pipeline, args: FusedArgs, x: mtl.Buffer, ws: [4]model.Tensor, os: [4]mtl.Buffer, groups: usize) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipe);
    enc.setBuffer(x, 0, 0);
    for (ws, os, 0..) |w, o, i| {
        enc.setBuffer(w.buffer, w.offset, 1 + i * 2);
        enc.setBuffer(o, 0, 2 + i * 2);
    }
    enc.setValue(args, 9);
    enc.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(256, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("dense_test fused: {s}", .{msg});
        return error.GpuFailed;
    }
}

fn runFusedx(m: *fam.Model, pipe: mtl.Pipeline, args: FusedxArgs, xs: [3]mtl.Buffer, ws: [8]model.Tensor, os: [8]mtl.Buffer, groups: usize) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipe);
    for (xs, 0..) |x, i| enc.setBuffer(x, 0, i);
    for (ws, os, 0..) |w, o, i| {
        enc.setBuffer(w.buffer, w.offset, 3 + i * 2);
        enc.setBuffer(o, 0, 4 + i * 2);
    }
    enc.setValue(args, 19);
    enc.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(256, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("dense_test fusedx: {s}", .{msg});
        return error.GpuFailed;
    }
}

fn checkCase(m: *fam.Model, pipes: fam.kernels.Dense, name: []const u8, w: model.Tensor, k: usize, cap: f32, sample_every: usize) !void {
    const rows: usize = if (w.rank >= 2) w.shape[0] else 1;
    const xb = try m.device.buffer(@max(k * 4, 16), mtl.ResourceOptions.shared);
    defer xb.deinit();
    const old = try m.device.buffer(@max(rows * 4, 16), mtl.ResourceOptions.shared);
    defer old.deinit();
    const new_a = try m.device.buffer(@max(rows * 4, 16), mtl.ResourceOptions.shared);
    defer new_a.deinit();
    const new_b = try m.device.buffer(@max(rows * 4, 16), mtl.ResourceOptions.shared);
    defer new_b.deinit();
    const xs = xb.slice(f32, k);
    fillX(xs);
    const args = MvArgs{ .m = @intCast(rows), .k = @intCast(k), .cap = cap };
    const old_pipe = if (cap > 0) m.pipe(.mv_cap) else m.pipe(.mv);
    const new_pipe = if (cap > 0) pipes.mv_cap else pipes.mv;
    const new_groups = if (cap > 0) (rows + 31) / 32 else (rows + 7) / 8;
    const new_threads: usize = if (cap > 0) 1024 else 256;
    try runMv(m, old_pipe, args, xb, w, old, (rows + 31) / 32, 1024);
    try runMv(m, new_pipe, args, xb, w, new_a, new_groups, new_threads);
    try runMv(m, new_pipe, args, xb, w, new_b, new_groups, new_threads);
    for (new_a.slice(f32, rows), new_b.slice(f32, rows)) |a, b| try t.expectEqual(a, b);
    var list: std.ArrayList(usize) = .empty;
    defer list.deinit(t.allocator);
    var r: usize = 0;
    while (r < rows) : (r += sample_every) try list.append(t.allocator, r);
    if (list.items.len == 0 or list.items[list.items.len - 1] != rows - 1) try list.append(t.allocator, rows - 1);
    const got_new = new_a.slice(f32, rows);
    const got_old = old.slice(f32, rows);
    const ref = try t.allocator.alloc(f64, list.items.len);
    defer t.allocator.free(ref);
    const bnd = try t.allocator.alloc(f64, list.items.len);
    defer t.allocator.free(bnd);
    refRows(w, k, xs, list.items, ref, bnd);
    var ck_new: Check = .{};
    var ck_old: Check = .{};
    var worst_drift: f64 = 0;
    for (list.items, 0..) |row, i| {
        var b = bnd[i];
        var want = ref[i];
        if (cap > 0) {
            want = cap * std.math.tanh(want / cap);
            b += 4.0 * std.math.ldexp(@as(f64, 1), -24) * (@abs(want) + 1.0);
        }
        ck_new.add(got_new[row], want, b);
        ck_old.add(got_old[row], want, b);
    }
    for (got_new, got_old) |a, b| {
        const d = @abs(@as(f64, a) - @as(f64, b));
        if (d > worst_drift) worst_drift = d;
    }
    std.debug.print("  {s} [{d}x{d}]{s}: {d} sampled rows, new worst err {e:.3} ratio {d:.4} failed {d}, old worst ratio {d:.4} failed {d}, old/new drift {e:.3}\n", .{ name, rows, k, if (cap > 0) " cap" else "", list.items.len, ck_new.worst_err, ck_new.worst_ratio, ck_new.failed, ck_old.worst_ratio, ck_old.failed, worst_drift });
    try t.expectEqual(@as(usize, 0), ck_new.failed);
    try t.expectEqual(@as(usize, 0), ck_old.failed);
}

fn checkFusedGF(m: *fam.Model, pipes: fam.kernels.Dense, x: mtl.Buffer, ws: [4]model.Tensor, h: usize, k: usize) !void {
    var outs: [4]mtl.Buffer = undefined;
    for ([_]usize{ h, h, 0, 0 }, 0..) |rows, i| outs[i] = try m.device.buffer(@max(rows * 4, 16), mtl.ResourceOptions.shared);
    defer for (outs) |o| o.deinit();
    try runFused(m, pipes.fused, .{ .k = @intCast(k), .m0 = @intCast(h), .m1 = @intCast(h), .m2 = 0, .m3 = 0 }, x, ws, outs, (2 * h + 7) / 8);
    var singles: [2]mtl.Buffer = undefined;
    for (0..2) |i| {
        singles[i] = try m.device.buffer(@max(h * 4, 16), mtl.ResourceOptions.shared);
        try runMv(m, pipes.mv, .{ .m = @intCast(h), .k = @intCast(k), .cap = 0 }, x, ws[i], singles[i], (h + 7) / 8, 256);
    }
    defer for (singles) |o| o.deinit();
    var ck: Check = .{};
    const xs = x.slice(f32, k);
    for (ws[0..2], outs[0..2], singles) |w, o, s| {
        const got = o.slice(f32, h);
        const want1 = s.slice(f32, h);
        for (got, want1) |a, b| try t.expectEqual(a, b);
        var r: usize = 0;
        while (r < h) : (r += 64) {
            const one: [1]usize = .{r};
            var ref: [1]f64 = undefined;
            var bnd: [1]f64 = undefined;
            refRows(w, k, xs, &one, &ref, &bnd);
            ck.add(got[r], ref[0], bnd[0]);
        }
    }
    std.debug.print("  fused-gf: matches singles bit for bit over {d} rows, bound worst ratio {d:.4} failed {d}\n", .{ 2 * h, ck.worst_ratio, ck.failed });
    try t.expectEqual(@as(usize, 0), ck.failed);
}

fn checkFusedx(m: *fam.Model, pipes: fam.kernels.Dense, name: []const u8, xs: [3]mtl.Buffer, xss: [3][]const f32, ks: [3]usize, ws: [8]model.Tensor, ms: [8]usize, xsel: [8]u32) !void {
    var outs: [8]mtl.Buffer = undefined;
    var outs2: [8]mtl.Buffer = undefined;
    var singles: [8]mtl.Buffer = undefined;
    for (ms, 0..) |rows, i| {
        outs[i] = try m.device.buffer(@max(rows * 4, 16), mtl.ResourceOptions.shared);
        outs2[i] = try m.device.buffer(@max(rows * 4, 16), mtl.ResourceOptions.shared);
        singles[i] = try m.device.buffer(@max(rows * 4, 16), mtl.ResourceOptions.shared);
    }
    defer for (outs, outs2, singles) |a, b, c| {
        a.deinit();
        b.deinit();
        c.deinit();
    };
    var total: usize = 0;
    var mar: [8]u32 = undefined;
    for (ms, 0..) |rows, i| {
        mar[i] = @intCast(rows);
        total += rows;
    }
    const args = FusedxArgs{ .k0 = @intCast(ks[0]), .k1 = @intCast(ks[1]), .k2 = @intCast(ks[2]), .m = mar, .x = xsel };
    try runFusedx(m, pipes.fusedx, args, xs, ws, outs, (total + 7) / 8);
    try runFusedx(m, pipes.fusedx, args, xs, ws, outs2, (total + 7) / 8);
    var ck: Check = .{};
    for (ws, outs, outs2, singles, ms, xsel, 0..) |w, o, o2, sgl, rows, x, i| {
        if (rows == 0) continue;
        const k = ks[x];
        try runMv(m, pipes.mv, .{ .m = @intCast(rows), .k = @intCast(k), .cap = 0 }, xs[x], w, sgl, (rows + 7) / 8, 256);
        const got = o.slice(f32, rows);
        const got2 = o2.slice(f32, rows);
        const ref1 = sgl.slice(f32, rows);
        for (got, got2, ref1) |a, b, c| {
            try t.expectEqual(a, b);
            try t.expectEqual(a, c);
        }
        const every: usize = if (rows > 1024) 64 else 1;
        var r: usize = 0;
        while (r < rows) : (r += every) {
            const one: [1]usize = .{r};
            var ref: [1]f64 = undefined;
            var bnd: [1]f64 = undefined;
            refRows(w, k, xss[x], &one, &ref, &bnd);
            ck.add(got[r], ref[0], bnd[0]);
        }
        _ = i;
    }
    std.debug.print("  {s}: fusedx matches singles bit for bit over {d} rows, rerun identical, bound worst ratio {d:.4} failed {d}\n", .{ name, total, ck.worst_ratio, ck.failed });
    try t.expectEqual(@as(usize, 0), ck.failed);
}

test "the dense kernels match i2_mv and i2_mv_cap on the real projection shapes within the fp32 bound" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const dir = try checkpoint(arena.allocator());
    const m = fam.Model.load(arena.allocator(), t.io, dir) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer m.deinit();
    const pipes = try fam.kernels.densePipes(m.device);
    const l0 = m.weights.layers[0].attn;
    const l5 = m.weights.layers[5];
    const ctl = m.weights.layers[6].controllers[0] orelse return error.MissingController;
    const h: usize = m.config.text.hidden;
    try checkCase(m, pipes, "q", l0.q, h, 0, 1);
    try checkCase(m, pipes, "r", l0.r, h, 0, 1);
    try checkCase(m, pipes, "k", l0.k, h, 0, 1);
    try checkCase(m, pipes, "v", l0.v, h, 0, 1);
    try checkCase(m, pipes, "sink", l0.sink, h, 0, 1);
    try checkCase(m, pipes, "gate", l0.gate, h, 0, 1);
    try checkCase(m, pipes, "o", l0.o, m.config.text.local.q * m.config.text.local.dim, 0, 1);
    try checkCase(m, pipes, "kdyn", l5.attn.k_dyn.?, h, 0, 1);
    try checkCase(m, pipes, "down", ctl.down, 2 * h, 0, 1);
    try checkCase(m, pipes, "gain", ctl.gain, m.config.loop.gate_rank, 0, 1);
    try checkCase(m, pipes, "forget", ctl.forget, m.config.loop.gate_rank, 0, 1);
    try checkCase(m, pipes, "router", m.weights.layers[0].router, h, 0, 1);
    try checkCase(m, pipes, "unembed", m.weights.unembed, h, m.config.text.logit_cap, 64);

    const xb = try m.device.buffer(@max(h * 4, 16), mtl.ResourceOptions.shared);
    defer xb.deinit();
    fillX(xb.slice(f32, h));
    const rank: usize = m.config.loop.gate_rank;
    const xbs = xb.slice(f32, h);
    const qx: [8]model.Tensor = .{ l0.q, l0.r, l0.sink, l0.gate, l0.q, l0.q, l0.q, l0.q };
    try checkFusedx(m, pipes, "fusedx-qrsg", .{ xb, xb, xb }, .{ xbs, xbs, xbs }, .{ h, h, h }, qx, .{ l0.q.shape[0], l0.r.shape[0], l0.sink.shape[0], l0.gate.shape[0], 0, 0, 0, 0 }, .{ 0, 0, 0, 0, 0, 0, 0, 0 });
    const fx: [8]model.Tensor = .{ l0.q, l0.r, l0.sink, l0.gate, l0.k, l5.attn.k_dyn.?, l0.v, l5.attn.v_dyn.? };
    try checkFusedx(m, pipes, "fusedx-full8", .{ xb, xb, xb }, .{ xbs, xbs, xbs }, .{ h, h, h }, fx, .{ l0.q.shape[0], l0.r.shape[0], l0.sink.shape[0], l0.gate.shape[0], l0.k.shape[0], l5.attn.k_dyn.?.shape[0], l0.v.shape[0], l5.attn.v_dyn.?.shape[0] }, .{ 0, 0, 0, 0, 1, 1, 2, 2 });
    const gf4: [4]model.Tensor = .{ ctl.gain, ctl.forget, ctl.gain, ctl.gain };
    try checkFusedGF(m, pipes, xb, gf4, h, rank);
    const gx: [8]model.Tensor = .{ ctl.gain, ctl.forget, ctl.gain, ctl.gain, ctl.gain, ctl.gain, ctl.gain, ctl.gain };
    const xbr = try m.device.buffer(@max(rank * 4, 16), mtl.ResourceOptions.shared);
    defer xbr.deinit();
    fillX(xbr.slice(f32, rank));
    const xrs = xbr.slice(f32, rank);
    try checkFusedx(m, pipes, "fusedx-gf", .{ xbr, xbr, xbr }, .{ xrs, xrs, xrs }, .{ rank, rank, rank }, gx, .{ h, h, 0, 0, 0, 0, 0, 0 }, .{ 0, 0, 0, 0, 0, 0, 0, 0 });
}
