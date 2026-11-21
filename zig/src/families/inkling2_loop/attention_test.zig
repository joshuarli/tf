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
            enc.setBuffer(kk, 0, 1);
            enc.setBuffer(vv, 0, 2);
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
