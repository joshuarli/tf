//! Split-K decode attention against the serial kernel: random Q/K/V/bias at positions crossing the 1,024
//! window, in both the sliding (tau off) and full (tau on) modes, within a derived fp32 bound.
const std = @import("std");
const mtl = @import("metal");
const kernels = @import("kernels.zig");
const st = @import("state.zig");

const heads: u32 = 24;
const kv: u32 = 8;
const dim: u32 = 128;
const drel: u32 = 16;
const extent: u32 = 1024;
const window: u32 = 1024;
const floor: f32 = 8192;
const alpha: f32 = 0.1;
const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, dim)));

const Attn = extern struct {
    heads: u32,
    kv: u32,
    dim: u32,
    rel: u32,
    extent: u32,
    window: u32,
    pos: u32,
    tau: u32,
    scale: f32,
    floor: f32,
    alpha: f32,
};

const AttnParts = extern struct {
    heads: u32,
    kv: u32,
    dim: u32,
    rel: u32,
    extent: u32,
    window: u32,
    pos: u32,
    tau: u32,
    splits: u32,
    chunk: u32,
    scale: f32,
    floor: f32,
    alpha: f32,
};

fn fill(rng: std.Random, x: []f32) void {
    for (x) |*v| v.* = rng.float(f32) * 4.0 - 2.0;
}

fn maxAbs(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

fn maxOf(x: []const f32) f32 {
    var m: f32 = 0;
    for (x) |v| m = @max(m, @abs(v));
    return m;
}

fn reference(out: []f32, q: []const f32, k: []const f32, v: []const f32, b: []const f32, sink: []const f32, pos: u32, full: bool) void {
    const tau: f64 = if (full) 1.0 + @as(f64, alpha) * @log(@max(1.0, (@as(f64, pos) + 1.0) / @as(f64, floor))) else 1.0;
    const lo: u32 = if (!full and pos + 1 > window) pos + 1 - window else 0;
    const g: u32 = heads / kv;
    for (0..heads) |h| {
        const kg: usize = h / g;
        var top: f64 = 0;
        var mass: f64 = 1;
        var acc: [128]f64 = @splat(0);
        for (lo..pos + 1) |j| {
            var dot: f64 = 0;
            for (0..dim) |d| dot += @as(f64, q[h * dim + d]) * @as(f64, k[(j * kv + kg) * dim + d]);
            const col: usize = @min(pos - j, extent);
            const score = dot * @as(f64, scale) * tau + @as(f64, b[h * (extent + 1) + col]) - @as(f64, sink[h]);
            const peak = @max(top, score);
            const r = @exp(top - peak);
            const w = @exp(score - peak);
            mass = mass * r + w;
            for (0..dim) |d| acc[d] = acc[d] * r + w * @as(f64, v[(j * kv + kg) * dim + d]);
            top = peak;
        }
        for (0..dim) |d| out[h * dim + d] = @floatCast(acc[d] / mass);
    }
}

test "split-K decode attention matches i2_attn across the sliding window" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, kernels.attention, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const glue_lib = try mtl.Library.fromSource(device, kernels.glue, mtl.CompileOptions.mlx());
    defer glue_lib.deinit();
    const old = try mtl.Pipeline.init(device, glue_lib, "i2_attn", false);
    defer old.deinit();
    const split = try mtl.Pipeline.init(device, lib, "i2_attn_split", false);
    defer split.deinit();
    const combine = try mtl.Pipeline.init(device, lib, "i2_attn_combine", false);
    defer combine.deinit();

    var prng = std.Random.DefaultPrng.init(0x696e6b6c);
    const rng = prng.random();
    var worst_old: f32 = 0;
    var worst_new: f32 = 0;
    var worst_pair: f32 = 0;
    var worst_bound: f32 = 0;
    for ([_]bool{ false, true }) |full| {
        for ([_]u32{ 1, 31, 1023, 1024, 1500, 4100, 16384 }) |pos| {
            if (!full and pos > 4100) continue;
            const rows: usize = @as(usize, pos) + 1;
            const n: usize = if (!full) @min(rows, window) else rows;
            const q = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared);
            defer q.deinit();
            const kk = try device.buffer(rows * kv * dim * 4, mtl.ResourceOptions.shared);
            defer kk.deinit();
            const vv = try device.buffer(rows * kv * dim * 4, mtl.ResourceOptions.shared);
            defer vv.deinit();
            const bb = try device.buffer(heads * (extent + 1) * 4, mtl.ResourceOptions.shared);
            defer bb.deinit();
            const sk = try device.buffer(heads * 4, mtl.ResourceOptions.shared);
            defer sk.deinit();
            fill(rng, q.slice(f32, heads * dim));
            fill(rng, kk.slice(f32, rows * kv * dim));
            fill(rng, vv.slice(f32, rows * kv * dim));
            fill(rng, bb.slice(f32, heads * (extent + 1)));
            fill(rng, sk.slice(f32, heads));
            const p = st.attnPartition(n);
            const part = try device.buffer(heads * p.splits * (dim + 2) * 4, mtl.ResourceOptions.shared);
            defer part.deinit();
            const out_old = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared);
            defer out_old.deinit();
            const out_new = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared);
            defer out_new.deinit();
            var kr: ?mtl.Buffer = null;
            defer if (kr) |b| b.deinit();
            var vr: ?mtl.Buffer = null;
            defer if (vr) |b| b.deinit();
            if (!full) {
                const rbuf = try device.buffer(window * kv * dim * 4, mtl.ResourceOptions.shared);
                kr = rbuf;
                const vbuf = try device.buffer(window * kv * dim * 4, mtl.ResourceOptions.shared);
                vr = vbuf;
                const ks = kk.slice(f32, rows * kv * dim);
                const vs = vv.slice(f32, rows * kv * dim);
                const kd = rbuf.slice(f32, window * kv * dim);
                const vd = vbuf.slice(f32, window * kv * dim);
                @memset(kd, 0);
                @memset(vd, 0);
                const lo: usize = if (rows > window) rows - window else 0;
                for (lo..rows) |j| {
                    @memcpy(kd[(j % window) * kv * dim ..][0 .. kv * dim], ks[j * kv * dim ..][0 .. kv * dim]);
                    @memcpy(vd[(j % window) * kv * dim ..][0 .. kv * dim], vs[j * kv * dim ..][0 .. kv * dim]);
                }
            }
            const k_new = kr orelse kk;
            const v_new = vr orelse vv;

            const args = Attn{ .heads = heads, .kv = kv, .dim = dim, .rel = drel, .extent = extent, .window = if (full) 0 else window, .pos = pos, .tau = @intFromBool(full), .scale = scale, .floor = floor, .alpha = alpha };
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(old);
            enc.setBuffer(q, 0, 0);
            enc.setBuffer(kk, 0, 1);
            enc.setBuffer(vv, 0, 2);
            enc.setBuffer(bb, 0, 3);
            enc.setBuffer(sk, 0, 4);
            enc.setValue(args, 5);
            enc.setBuffer(out_old, 0, 6);
            enc.dispatchGroups(mtl.Size.of(heads, 1, 1), mtl.Size.of(32, 1, 1));
            const pargs = AttnParts{ .heads = heads, .kv = kv, .dim = dim, .rel = drel, .extent = extent, .window = if (full) 0 else window, .pos = pos, .tau = @intFromBool(full), .splits = p.splits, .chunk = p.chunk, .scale = scale, .floor = floor, .alpha = alpha };
            enc.setPipeline(split);
            enc.setBuffer(q, 0, 0);
            enc.setBuffer(k_new, 0, 1);
            enc.setBuffer(v_new, 0, 2);
            enc.setBuffer(bb, 0, 3);
            enc.setBuffer(sk, 0, 4);
            enc.setValue(pargs, 5);
            enc.setBuffer(part, 0, 6);
            enc.dispatchGroups(mtl.Size.of(kv * p.splits, 1, 1), mtl.Size.of((heads / kv) * 32, 1, 1));
            enc.setPipeline(combine);
            enc.setBuffer(part, 0, 0);
            enc.setValue(pargs, 1);
            enc.setBuffer(out_new, 0, 2);
            enc.dispatchGroups(mtl.Size.of(heads, 1, 1), mtl.Size.of(32, 1, 1));
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("attention test: {s}\n", .{msg});
                return error.GpuFailed;
            }

            const got_old = out_old.slice(f32, heads * dim);
            const got_new = out_new.slice(f32, heads * dim);
            const want = try std.testing.allocator.alloc(f32, heads * dim);
            defer std.testing.allocator.free(want);
            reference(want, q.slice(f32, heads * dim), kk.slice(f32, rows * kv * dim), vv.slice(f32, rows * kv * dim), bb.slice(f32, heads * (extent + 1)), sk.slice(f32, heads), pos, full);
            const d_old = maxAbs(got_old, want);
            const d_new = maxAbs(got_new, want);
            const d_pair = maxAbs(got_old, got_new);
            const vmax = @max(maxOf(vv.slice(f32, rows * kv * dim)), 1.0);
            const bound: f32 = 16 * 1.1920929e-7 * @as(f32, @floatCast(@sqrt(@as(f64, @floatFromInt(n * dim))))) * vmax;
            worst_old = @max(worst_old, d_old);
            worst_new = @max(worst_new, d_new);
            worst_pair = @max(worst_pair, d_pair);
            worst_bound = @max(worst_bound, bound);
            if (d_old > bound or d_new > bound or d_pair > bound) {
                std.debug.print("attention test full={} pos={d} n={d}: old-vs-ref {e}, new-vs-ref {e}, old-vs-new {e}, bound {e}\n", .{ full, pos, n, d_old, d_new, d_pair, bound });
                return error.TestUnexpectedResult;
            }
        }
    }
    std.debug.print("attention test: worst old-vs-ref {e}, new-vs-ref {e}, old-vs-new {e}, bound {e}\n", .{ worst_old, worst_new, worst_pair, worst_bound });
}

test "ring-backed local attention equals the full cache across the wrap" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, kernels.attention, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const glue_lib = try mtl.Library.fromSource(device, kernels.glue, mtl.CompileOptions.mlx());
    defer glue_lib.deinit();
    const old = try mtl.Pipeline.init(device, glue_lib, "i2_attn", false);
    defer old.deinit();
    const split = try mtl.Pipeline.init(device, lib, "i2_attn_split", false);
    defer split.deinit();
    const combine = try mtl.Pipeline.init(device, lib, "i2_attn_combine", false);
    defer combine.deinit();

    var prng = std.Random.DefaultPrng.init(0x72696e67);
    const rng = prng.random();
    var worst_pair: f32 = 0;
    var worst_ref: f32 = 0;
    var worst_bound: f32 = 0;
    for ([_]u32{ 0, 1023, 1024, 1025, 2047, 2048, 3000 }) |pos| {
        for ([_]bool{ false, true }) |full| {
            const rows: usize = @as(usize, pos) + 1;
            const n: usize = if (!full) @min(rows, window) else rows;
            const lo: usize = if (!full and rows > window) rows - window else 0;
            const q = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared);
            defer q.deinit();
            const kf = try device.buffer(rows * kv * dim * 4, mtl.ResourceOptions.shared);
            defer kf.deinit();
            const vf = try device.buffer(rows * kv * dim * 4, mtl.ResourceOptions.shared);
            defer vf.deinit();
            const bb = try device.buffer(heads * (extent + 1) * 4, mtl.ResourceOptions.shared);
            defer bb.deinit();
            const sk = try device.buffer(heads * 4, mtl.ResourceOptions.shared);
            defer sk.deinit();
            fill(rng, q.slice(f32, heads * dim));
            fill(rng, kf.slice(f32, rows * kv * dim));
            fill(rng, vf.slice(f32, rows * kv * dim));
            fill(rng, bb.slice(f32, heads * (extent + 1)));
            fill(rng, sk.slice(f32, heads));
            const p = st.attnPartition(n);
            const args = Attn{ .heads = heads, .kv = kv, .dim = dim, .rel = drel, .extent = extent, .window = if (full) 0 else window, .pos = pos, .tau = @intFromBool(full), .scale = scale, .floor = floor, .alpha = alpha };
            const pargs = AttnParts{ .heads = heads, .kv = kv, .dim = dim, .rel = drel, .extent = extent, .window = if (full) 0 else window, .pos = pos, .tau = @intFromBool(full), .splits = p.splits, .chunk = p.chunk, .scale = scale, .floor = floor, .alpha = alpha };
            const out_old = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared);
            defer out_old.deinit();
            const out_new = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared);
            defer out_new.deinit();
            var kr: ?mtl.Buffer = null;
            defer if (kr) |b| b.deinit();
            var vr: ?mtl.Buffer = null;
            defer if (vr) |b| b.deinit();
            var part_lin: ?mtl.Buffer = null;
            defer if (part_lin) |b| b.deinit();
            var out_lin: ?mtl.Buffer = null;
            defer if (out_lin) |b| b.deinit();
            if (!full) {
                const rbuf = try device.buffer(window * kv * dim * 4, mtl.ResourceOptions.shared);
                kr = rbuf;
                const vbuf = try device.buffer(window * kv * dim * 4, mtl.ResourceOptions.shared);
                vr = vbuf;
                const ks = kf.slice(f32, rows * kv * dim);
                const vs = vf.slice(f32, rows * kv * dim);
                const kd = rbuf.slice(f32, window * kv * dim);
                const vd = vbuf.slice(f32, window * kv * dim);
                @memset(kd, 0);
                @memset(vd, 0);
                for (lo..rows) |j| {
                    @memcpy(kd[(j % window) * kv * dim ..][0 .. kv * dim], ks[j * kv * dim ..][0 .. kv * dim]);
                    @memcpy(vd[(j % window) * kv * dim ..][0 .. kv * dim], vs[j * kv * dim ..][0 .. kv * dim]);
                }
                if (rows <= window) {
                    part_lin = try device.buffer(heads * p.splits * (dim + 2) * 4, mtl.ResourceOptions.shared);
                    out_lin = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared);
                }
            }
            const part = try device.buffer(heads * p.splits * (dim + 2) * 4, mtl.ResourceOptions.shared);
            defer part.deinit();
            const k_new = kr orelse kf;
            const v_new = vr orelse vf;
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(old);
            enc.setBuffer(q, 0, 0);
            enc.setBuffer(kf, 0, 1);
            enc.setBuffer(vf, 0, 2);
            enc.setBuffer(bb, 0, 3);
            enc.setBuffer(sk, 0, 4);
            enc.setValue(args, 5);
            enc.setBuffer(out_old, 0, 6);
            enc.dispatchGroups(mtl.Size.of(heads, 1, 1), mtl.Size.of(32, 1, 1));
            enc.setPipeline(split);
            enc.setBuffer(q, 0, 0);
            enc.setBuffer(k_new, 0, 1);
            enc.setBuffer(v_new, 0, 2);
            enc.setBuffer(bb, 0, 3);
            enc.setBuffer(sk, 0, 4);
            enc.setValue(pargs, 5);
            enc.setBuffer(part, 0, 6);
            enc.dispatchGroups(mtl.Size.of(kv * p.splits, 1, 1), mtl.Size.of((heads / kv) * 32, 1, 1));
            enc.setPipeline(combine);
            enc.setBuffer(part, 0, 0);
            enc.setValue(pargs, 1);
            enc.setBuffer(out_new, 0, 2);
            enc.dispatchGroups(mtl.Size.of(heads, 1, 1), mtl.Size.of(32, 1, 1));
            if (part_lin) |pl| {
                enc.setPipeline(split);
                enc.setBuffer(q, 0, 0);
                enc.setBuffer(kf, 0, 1);
                enc.setBuffer(vf, 0, 2);
                enc.setBuffer(bb, 0, 3);
                enc.setBuffer(sk, 0, 4);
                enc.setValue(pargs, 5);
                enc.setBuffer(pl, 0, 6);
                enc.dispatchGroups(mtl.Size.of(kv * p.splits, 1, 1), mtl.Size.of((heads / kv) * 32, 1, 1));
                enc.setPipeline(combine);
                enc.setBuffer(pl, 0, 0);
                enc.setValue(pargs, 1);
                enc.setBuffer(out_lin.?, 0, 2);
                enc.dispatchGroups(mtl.Size.of(heads, 1, 1), mtl.Size.of(32, 1, 1));
            }
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("ring attention test: {s}\n", .{msg});
                return error.GpuFailed;
            }
            const got_old = out_old.slice(f32, heads * dim);
            const got_new = out_new.slice(f32, heads * dim);
            const want = try std.testing.allocator.alloc(f32, heads * dim);
            defer std.testing.allocator.free(want);
            reference(want, q.slice(f32, heads * dim), kf.slice(f32, rows * kv * dim), vf.slice(f32, rows * kv * dim), bb.slice(f32, heads * (extent + 1)), sk.slice(f32, heads), pos, full);
            const vmax = @max(maxOf(vf.slice(f32, rows * kv * dim)), 1.0);
            const bound: f32 = 16 * 1.1920929e-7 * @as(f32, @floatCast(@sqrt(@as(f64, @floatFromInt(n * dim))))) * vmax;
            const d_old = maxAbs(got_old, want);
            const d_new = maxAbs(got_new, want);
            const d_pair = maxAbs(got_old, got_new);
            worst_pair = @max(worst_pair, d_pair);
            worst_ref = @max(worst_ref, @max(d_old, d_new));
            worst_bound = @max(worst_bound, bound);
            if (d_old > bound or d_new > bound or d_pair > bound) {
                std.debug.print("ring test full={} pos={d} n={d}: old-vs-ref {e}, new-vs-ref {e}, old-vs-new {e}, bound {e}\n", .{ full, pos, n, d_old, d_new, d_pair, bound });
                return error.TestUnexpectedResult;
            }
            if (out_lin) |ol| {
                const d_bit = maxAbs(ol.slice(f32, heads * dim), got_new);
                if (d_bit != 0) {
                    std.debug.print("ring test local pos={d}: linear-vs-ring {e}, want bitwise 0\n", .{ pos, d_bit });
                    return error.TestUnexpectedResult;
                }
            }
        }
    }
    std.debug.print("ring test: worst pair {e}, worst vs-ref {e}, bound {e}\n", .{ worst_pair, worst_ref, worst_bound });
}
