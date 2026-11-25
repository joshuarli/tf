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
            const kh = try device.buffer(rows * kv * dim * 2, mtl.ResourceOptions.shared);
            defer kh.deinit();
            const vh = try device.buffer(rows * kv * dim * 2, mtl.ResourceOptions.shared);
            defer vh.deinit();
            for (kk.slice(f32, rows * kv * dim), kh.slice(f16, rows * kv * dim)) |*src, *dst| { dst.* = @floatCast(src.*); src.* = dst.*; }
            for (vv.slice(f32, rows * kv * dim), vh.slice(f16, rows * kv * dim)) |*src, *dst| { dst.* = @floatCast(src.*); src.* = dst.*; }
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
                const rbuf = try device.buffer(window * kv * dim * 2, mtl.ResourceOptions.shared);
                kr = rbuf;
                const vbuf = try device.buffer(window * kv * dim * 2, mtl.ResourceOptions.shared);
                vr = vbuf;
                const ks = kh.slice(f16, rows * kv * dim);
                const vs = vh.slice(f16, rows * kv * dim);
                const kd = rbuf.slice(f16, window * kv * dim);
                const vd = vbuf.slice(f16, window * kv * dim);
                @memset(kd, 0);
                @memset(vd, 0);
                const lo: usize = if (rows > window) rows - window else 0;
                for (lo..rows) |j| {
                    @memcpy(kd[(j % window) * kv * dim ..][0 .. kv * dim], ks[j * kv * dim ..][0 .. kv * dim]);
                    @memcpy(vd[(j % window) * kv * dim ..][0 .. kv * dim], vs[j * kv * dim ..][0 .. kv * dim]);
                }
            }
            const k_new = kr orelse kh;
            const v_new = vr orelse vh;

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
            const kh = try device.buffer(rows * kv * dim * 2, mtl.ResourceOptions.shared);
            defer kh.deinit();
            const vh = try device.buffer(rows * kv * dim * 2, mtl.ResourceOptions.shared);
            defer vh.deinit();
            for (kf.slice(f32, rows * kv * dim), kh.slice(f16, rows * kv * dim)) |*src, *dst| { dst.* = @floatCast(src.*); src.* = dst.*; }
            for (vf.slice(f32, rows * kv * dim), vh.slice(f16, rows * kv * dim)) |*src, *dst| { dst.* = @floatCast(src.*); src.* = dst.*; }
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
                const rbuf = try device.buffer(window * kv * dim * 2, mtl.ResourceOptions.shared);
                kr = rbuf;
                const vbuf = try device.buffer(window * kv * dim * 2, mtl.ResourceOptions.shared);
                vr = vbuf;
                const ks = kh.slice(f16, rows * kv * dim);
                const vs = vh.slice(f16, rows * kv * dim);
                const kd = rbuf.slice(f16, window * kv * dim);
                const vd = vbuf.slice(f16, window * kv * dim);
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
            const k_new = kr orelse kh;
            const v_new = vr orelse vh;
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
                enc.setBuffer(kh, 0, 1);
                enc.setBuffer(vh, 0, 2);
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

/// The split and merge kernels before the split staged blocks of rows: one row at a time through threadgroup memory. The staged kernel
/// must keep its bytes, since prefill's attention and every saved decode result follow this per-row order.
const serial_split =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\struct i2_attn_parts { uint heads, kv, dim, rel, extent, window, pos, tau, splits, chunk; float scale, floor, alpha; };
    \\
    \\[[kernel]] void i2_attn_split(const device float* Q [[buffer(0)]], const device half* KEYS [[buffer(1)]],
    \\    const device half* VALUES [[buffer(2)]], const device float* B [[buffer(3)]], const device float* sink [[buffer(4)]],
    \\    constant i2_attn_parts& a [[buffer(5)]], device float* PART [[buffer(6)]],
    \\    uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    \\    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint nth [[threads_per_threadgroup]]) {
    \\  const uint G = a.heads / a.kv;
    \\  const uint D = a.dim, per = max(1u, D / 32u);
    \\  const uint kv = tgp / a.splits, s = tgp % a.splits, h = kv * G + sg;
    \\  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(a.pos) + 1.0f) / a.floor)) : 1.0f;
    \\  float qv[4];
    \\  for (uint j = 0; j < per; ++j) qv[j] = (lane + 32u * j < D) ? Q[h * D + lane + 32u * j] * a.scale * tau : 0.0f;
    \\  const float sk = sink[h];
    \\  const uint lo = a.window ? (a.pos + 1u > a.window ? a.pos + 1u - a.window : 0u) : 0u;
    \\  const size_t base = ((size_t)h * a.splits + s) * (D + 2u);
    \\  const uint js = lo + s * a.chunk, je = min(js + a.chunk, a.pos + 1u);
    \\  if (js >= je) {
    \\    if (lane == 0u) {
    \\      PART[base] = -INFINITY;
    \\      PART[base + 1u] = 0.0f;
    \\    }
    \\    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) PART[base + 2u + lane + 32u * u] = 0.0f;
    \\    return;
    \\  }
    \\  threadgroup float4 KB4[32];
    \\  threadgroup float4 VB4[32];
    \\  threadgroup float* KB = (threadgroup float*)KB4;
    \\  threadgroup float* VB = (threadgroup float*)VB4;
    \\  float top = -INFINITY, mass = 0.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    \\  for (uint j = js; j < je; ++j) {
    \\    const uint col = min(a.pos - j, a.extent);
    \\    const uint rj = a.window ? (j % a.window) : j;
    \\    const size_t at = ((size_t)rj * a.kv + kv) * D;
    \\    const float bias = B[h * (a.extent + 1u) + col] - sk;
    \\    for (uint v = tid; v < D / 4u; v += nth) {
    \\      KB4[v] = float4(*reinterpret_cast<const device packed_half4*>(KEYS + at + v * 4u));
    \\      VB4[v] = float4(*reinterpret_cast<const device packed_half4*>(VALUES + at + v * 4u));
    \\    }
    \\    for (uint i = (D / 4u) * 4u + tid; i < D; i += nth) {
    \\      KB[i] = KEYS[at + i];
    \\      VB[i] = VALUES[at + i];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    float dot = 0.0f;
    \\    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) dot = fma(qv[u], KB[lane + 32u * u], dot);
    \\    const float score = simd_sum(dot) + bias;
    \\    const float peak = max(top, score);
    \\    const float rescale = precise::exp(top - peak);
    \\    const float w = precise::exp(score - peak);
    \\    mass = mass * rescale + w;
    \\    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) acc[u] = acc[u] * rescale + w * VB[lane + 32u * u];
    \\    top = peak;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  }
    \\  if (lane == 0u) {
    \\    PART[base] = top;
    \\    PART[base + 1u] = mass;
    \\  }
    \\  for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) PART[base + 2u + lane + 32u * u] = acc[u];
    \\}
    \\
    \\[[kernel]] void i2_attn_combine(const device float* PART [[buffer(0)]],
    \\    constant i2_attn_parts& a [[buffer(1)]], device float* O [[buffer(2)]],
    \\    uint h [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    \\  const uint D = a.dim, S = a.splits, per = max(1u, D / 32u);
    \\  float top = 0.0f, mass = 1.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    \\  for (uint s = 0; s < S; ++s) {
    \\    const size_t base = ((size_t)h * S + s) * (D + 2u);
    \\    const float ptop = PART[base];
    \\    const float pmass = PART[base + 1u];
    \\    if (pmass == 0.0f) continue;
    \\    const float peak = max(top, ptop);
    \\    const float r0 = precise::exp(top - peak);
    \\    const float r1 = precise::exp(ptop - peak);
    \\    mass = mass * r0 + pmass * r1;
    \\    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) acc[u] = acc[u] * r0 + r1 * PART[base + 2u + lane + 32u * u];
    \\    top = peak;
    \\  }
    \\  for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) O[h * D + lane + 32u * u] = acc[u] / mass;
    \\}
    \\
;

const BatchRow = extern struct { keys: u64, values: u64, args: AttnParts };

fn splitInputs(rng: std.Random, device: mtl.Device, rows: usize) ![5]mtl.Buffer {
    const q = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared);
    errdefer q.deinit();
    const kh = try device.buffer(rows * kv * dim * 2, mtl.ResourceOptions.shared);
    errdefer kh.deinit();
    const vh = try device.buffer(rows * kv * dim * 2, mtl.ResourceOptions.shared);
    errdefer vh.deinit();
    const bb = try device.buffer(heads * (extent + 1) * 4, mtl.ResourceOptions.shared);
    errdefer bb.deinit();
    const sk = try device.buffer(heads * 4, mtl.ResourceOptions.shared);
    fill(rng, q.slice(f32, heads * dim));
    for (kh.slice(f16, rows * kv * dim)) |*x| x.* = @floatCast(rng.float(f32) * 4.0 - 2.0);
    for (vh.slice(f16, rows * kv * dim)) |*x| x.* = @floatCast(rng.float(f32) * 4.0 - 2.0);
    fill(rng, bb.slice(f32, heads * (extent + 1)));
    fill(rng, sk.slice(f32, heads));
    return .{ q, kh, vh, bb, sk };
}

fn sameBits(a: []const f32, b: []const f32) bool {
    for (a, b) |x, y| if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) return false;
    return true;
}

test "staged split attention keeps the serial kernel's bytes, single and batched" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, kernels.attention, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const old_lib = try mtl.Library.fromSource(device, serial_split, mtl.CompileOptions.mlx());
    defer old_lib.deinit();
    const old = try mtl.Pipeline.init(device, old_lib, "i2_attn_split", false);
    defer old.deinit();
    const old_combine = try mtl.Pipeline.init(device, old_lib, "i2_attn_combine", false);
    defer old_combine.deinit();
    const split = try mtl.Pipeline.init(device, lib, "i2_attn_split", false);
    defer split.deinit();
    const combine = try mtl.Pipeline.init(device, lib, "i2_attn_combine", false);
    defer combine.deinit();
    const batch_split = try mtl.Pipeline.init(device, lib, "i2_batch_split", false);
    defer batch_split.deinit();
    const batch_combine = try mtl.Pipeline.init(device, lib, "i2_batch_combine", false);
    defer batch_combine.deinit();

    var prng = std.Random.DefaultPrng.init(0x73746167);
    const rng = prng.random();
    const positions = [_]u32{ 0, 1, 30, 31, 32, 33, 63, 64, 65, 95, 96, 127, 128, 1023, 1024, 1025, 1100, 2047, 2499, 4100 };
    var checked: usize = 0;
    for ([_]bool{ false, true }) |full| {
        for (positions) |pos| {
            const rows: usize = if (full) @as(usize, pos) + 1 else window;
            const n: usize = if (full) rows else @min(@as(usize, pos) + 1, window);
            const inputs = try splitInputs(rng, device, rows);
            defer for (inputs) |buf| buf.deinit();
            const p = st.attnPartition(n);
            const pargs = AttnParts{ .heads = heads, .kv = kv, .dim = dim, .rel = drel, .extent = extent, .window = if (full) 0 else window, .pos = pos, .tau = @intFromBool(full), .splits = p.splits, .chunk = p.chunk, .scale = scale, .floor = floor, .alpha = alpha };
            var parts: [2]mtl.Buffer = undefined;
            var outs: [2]mtl.Buffer = undefined;
            for (0..2) |i| {
                parts[i] = try device.buffer(heads * p.splits * (dim + 2) * 4, mtl.ResourceOptions.shared);
                outs[i] = try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared);
            }
            defer for (0..2) |i| {
                parts[i].deinit();
                outs[i].deinit();
            };
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            for ([_]mtl.Pipeline{ old, split }, 0..) |pipe, i| {
                enc.setPipeline(pipe);
                for (inputs, 0..) |buf, at| enc.setBuffer(buf, 0, at);
                enc.setValue(pargs, 5);
                enc.setBuffer(parts[i], 0, 6);
                enc.dispatchGroups(mtl.Size.of(kv * p.splits, 1, 1), mtl.Size.of((heads / kv) * 32, 1, 1));
                enc.setPipeline(if (i == 0) old_combine else combine);
                enc.setBuffer(parts[i], 0, 0);
                enc.setValue(pargs, 1);
                enc.setBuffer(outs[i], 0, 2);
                enc.dispatchGroups(mtl.Size.of(heads, 1, 1), mtl.Size.of(32, 1, 1));
            }
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("staged split test: {s}\n", .{msg});
                return error.GpuFailed;
            }
            const count = heads * p.splits * (dim + 2);
            if (!sameBits(parts[0].slice(f32, count), parts[1].slice(f32, count)) or !sameBits(outs[0].slice(f32, heads * dim), outs[1].slice(f32, heads * dim))) {
                std.debug.print("staged split test full={} pos={d}: bytes differ from the serial kernel, worst {e}\n", .{ full, pos, maxAbs(outs[0].slice(f32, heads * dim), outs[1].slice(f32, heads * dim)) });
                return error.TestUnexpectedResult;
            }
            checked += 1;
        }
    }

    const width = 3;
    for ([_]bool{ false, true }) |full| {
        const picks = [_][width]u32{ .{ 5, 64, 2499 }, .{ 1023, 1500, 31 }, .{ 0, 1024, 4100 } };
        for (picks) |at| {
            var rowbufs: [width][5]mtl.Buffer = undefined;
            var made: usize = 0;
            defer for (rowbufs[0..made]) |set| for (set) |buf| buf.deinit();
            var rows: [width]BatchRow = undefined;
            var argsets: [width]AttnParts = undefined;
            var most: u32 = 0;
            for (at, 0..) |pos, r| {
                const cap: usize = if (full) @as(usize, pos) + 1 else window;
                rowbufs[r] = try splitInputs(rng, device, cap);
                made += 1;
                const n: usize = if (full) cap else @min(@as(usize, pos) + 1, window);
                const p = st.attnPartition(n);
                argsets[r] = .{ .heads = heads, .kv = kv, .dim = dim, .rel = drel, .extent = extent, .window = if (full) 0 else window, .pos = pos, .tau = @intFromBool(full), .splits = p.splits, .chunk = p.chunk, .scale = scale, .floor = floor, .alpha = alpha };
                rows[r] = .{ .keys = rowbufs[r][1].gpuAddress(), .values = rowbufs[r][2].gpuAddress(), .args = argsets[r] };
                most = @max(most, p.splits);
            }
            const q = try device.buffer(width * heads * dim * 4, mtl.ResourceOptions.shared);
            defer q.deinit();
            const bb = try device.buffer(width * heads * (extent + 1) * 4, mtl.ResourceOptions.shared);
            defer bb.deinit();
            const sk = try device.buffer(width * heads * 4, mtl.ResourceOptions.shared);
            defer sk.deinit();
            for (0..width) |r| {
                @memcpy(q.slice(f32, width * heads * dim)[r * heads * dim ..][0 .. heads * dim], rowbufs[r][0].slice(f32, heads * dim));
                @memcpy(bb.slice(f32, width * heads * (extent + 1))[r * heads * (extent + 1) ..][0 .. heads * (extent + 1)], rowbufs[r][3].slice(f32, heads * (extent + 1)));
                @memcpy(sk.slice(f32, width * heads)[r * heads ..][0..heads], rowbufs[r][4].slice(f32, heads));
            }
            const bpart = try device.buffer(width * heads * st.attn_splits * (dim + 2) * 4, mtl.ResourceOptions.shared);
            defer bpart.deinit();
            const bout = try device.buffer(width * heads * dim * 4, mtl.ResourceOptions.shared);
            defer bout.deinit();
            var singles: [width][2]mtl.Buffer = undefined;
            var single_made: usize = 0;
            defer for (singles[0..single_made]) |pair| for (pair) |buf| buf.deinit();
            for (0..width) |r| {
                singles[r] = .{ try device.buffer(heads * argsets[r].splits * (dim + 2) * 4, mtl.ResourceOptions.shared), try device.buffer(heads * dim * 4, mtl.ResourceOptions.shared) };
                single_made += 1;
            }
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(batch_split);
            enc.setBuffer(q, 0, 0);
            enc.setBuffer(bb, 0, 1);
            enc.setBuffer(sk, 0, 2);
            enc.setBytes(std.mem.sliceAsBytes(rows[0..]), 3);
            enc.setBuffer(bpart, 0, 4);
            for (0..width) |r| {
                enc.useResource(rowbufs[r][1], mtl.ResourceUsage.read);
                enc.useResource(rowbufs[r][2], mtl.ResourceUsage.read);
            }
            enc.dispatchGroups(mtl.Size.of(kv * most, width, 1), mtl.Size.of((heads / kv) * 32, 1, 1));
            enc.setPipeline(batch_combine);
            enc.setBuffer(bpart, 0, 0);
            enc.setBytes(std.mem.sliceAsBytes(rows[0..]), 1);
            enc.setBuffer(bout, 0, 2);
            enc.dispatchGroups(mtl.Size.of(heads, width, 1), mtl.Size.of(32, 1, 1));
            for (0..width) |r| {
                enc.setPipeline(split);
                for (rowbufs[r], 0..) |buf, i| enc.setBuffer(buf, 0, i);
                enc.setValue(argsets[r], 5);
                enc.setBuffer(singles[r][0], 0, 6);
                enc.dispatchGroups(mtl.Size.of(kv * argsets[r].splits, 1, 1), mtl.Size.of((heads / kv) * 32, 1, 1));
                enc.setPipeline(combine);
                enc.setBuffer(singles[r][0], 0, 0);
                enc.setValue(argsets[r], 1);
                enc.setBuffer(singles[r][1], 0, 2);
                enc.dispatchGroups(mtl.Size.of(heads, 1, 1), mtl.Size.of(32, 1, 1));
            }
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("staged batch test: {s}\n", .{msg});
                return error.GpuFailed;
            }
            for (0..width) |r| {
                const got = bout.slice(f32, width * heads * dim)[r * heads * dim ..][0 .. heads * dim];
                if (!sameBits(got, singles[r][1].slice(f32, heads * dim))) {
                    std.debug.print("staged batch test full={} pos={d}: batch row differs from its single-stream bytes\n", .{ full, at[r] });
                    return error.TestUnexpectedResult;
                }
            }
            checked += width;
        }
    }
    std.debug.print("staged split test: {d} cases byte-equal to the serial kernel and across batch rows\n", .{checked});
}

const MixArgs = extern struct { width: u32, depth: u32, k: u32, rows: u32, scale: f32 };

test "batched depth mixes keep the prefill mixes' bytes" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, kernels.attention, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const prefill_lib = try mtl.Library.fromSource(device, kernels.prefillSource(false), mtl.CompileOptions.mlx());
    defer prefill_lib.deinit();
    const pipes = [_]mtl.Pipeline{
        try mtl.Pipeline.init(device, prefill_lib, "i2p_mix", false),
        try mtl.Pipeline.init(device, lib, "i2_batch_mix", false),
        try mtl.Pipeline.init(device, prefill_lib, "i2p_mix4", false),
        try mtl.Pipeline.init(device, lib, "i2_batch_mix4", false),
    };
    defer for (pipes) |p| p.deinit();
    const width = 2304;
    const rows = 5;
    var prng = std.Random.DefaultPrng.init(0x6d6978);
    const rng = prng.random();
    const states = try device.buffer(rows * 5 * width * 4, mtl.ResourceOptions.shared);
    defer states.deinit();
    const queries = try device.buffer(4 * width * 4, mtl.ResourceOptions.shared);
    defer queries.deinit();
    fill(rng, states.slice(f32, rows * 5 * width));
    fill(rng, queries.slice(f32, 4 * width));
    var outs: [2]mtl.Buffer = undefined;
    for (&outs) |*o| o.* = try device.buffer(rows * 4 * width * 4, mtl.ResourceOptions.shared);
    defer for (outs) |o| o.deinit();
    var checked: usize = 0;
    for (1..6) |depth| for (0..5) |kq| {
        const four = kq == 4;
        const k: u32 = if (four) 4 else @intCast(kq + 1);
        const args = MixArgs{ .width = width, .depth = @intCast(depth), .k = k, .rows = rows, .scale = 1.0 / @sqrt(@as(f32, width)) };
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..2) |side| {
            enc.setPipeline(pipes[(if (four) @as(usize, 2) else 0) + side]);
            enc.setBuffer(states, 0, 0);
            if (four) {
                for (0..4) |q| enc.setBuffer(queries, q * width * 4, 1 + q);
                enc.setValue(args, 5);
                enc.setBuffer(outs[side], 0, 6);
            } else {
                enc.setBuffer(queries, 0, 1);
                enc.setValue(args, 2);
                enc.setBuffer(outs[side], 0, 3);
            }
            enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
        }
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("mix test: {s}\n", .{msg});
            return error.GpuFailed;
        }
        const count = rows * k * width;
        if (!sameBits(outs[0].slice(f32, count), outs[1].slice(f32, count))) {
            std.debug.print("mix test depth={d} k={d} four={}: bytes differ\n", .{ depth, k, four });
            return error.TestUnexpectedResult;
        }
        checked += 1;
    };
    std.debug.print("batched depth mix test: {d} cases byte-equal to the prefill mixes\n", .{checked});
}

const MoeArgs = extern struct { k: u32, n: u32, slots: u32, topk: u32, gate_cap: f32, up_cap: f32 };

test "gathered expert projections keep the per-slot kernels' bytes" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var shared = try @import("rows.zig").Pipes.init(device);
    defer shared.deinit();
    const prefill_lib = try mtl.Library.fromSource(device, kernels.prefillSource(false), mtl.CompileOptions.mlx());
    defer prefill_lib.deinit();
    const gu = try mtl.Pipeline.init(device, prefill_lib, "i2p_gu", false);
    defer gu.deinit();
    const down = try mtl.Pipeline.init(device, prefill_lib, "i2p_down", false);
    defer down.deinit();
    const experts = 24;
    const h = 2304;
    const inter = 1152;
    const topk = 8;
    const widest = @import("rows.zig").max_rows;
    const most = widest * topk;
    var prng = std.Random.DefaultPrng.init(0x6578_7073);
    const rng = prng.random();
    const w13 = try device.buffer(experts * 2 * inter * h / 2, mtl.ResourceOptions.shared);
    defer w13.deinit();
    const s13 = try device.buffer(experts * 2 * inter * h / 16, mtl.ResourceOptions.shared);
    defer s13.deinit();
    const w2 = try device.buffer(experts * h * inter / 2, mtl.ResourceOptions.shared);
    defer w2.deinit();
    const s2 = try device.buffer(experts * h * inter / 16, mtl.ResourceOptions.shared);
    defer s2.deinit();
    rng.bytes(w13.slice(u8, experts * 2 * inter * h / 2));
    rng.bytes(w2.slice(u8, experts * h * inter / 2));
    for (s13.slice(u8, experts * 2 * inter * h / 16)) |*v| v.* = 0x20 + rng.uintLessThan(u8, 0x30);
    for (s2.slice(u8, experts * h * inter / 16)) |*v| v.* = 0x20 + rng.uintLessThan(u8, 0x30);
    const amax = try device.buffer(experts * 4, mtl.ResourceOptions.shared);
    defer amax.deinit();
    for (amax.slice(f32, experts)) |*v| v.* = rng.float(f32) * 4 + 1;
    const den = try device.buffer(16, mtl.ResourceOptions.shared);
    defer den.deinit();
    den.slice(f32, 1)[0] = 448.0 * 6.0;
    const codes = try device.buffer(16 * 4, mtl.ResourceOptions.shared);
    defer codes.deinit();
    for (codes.slice(f32, 16), [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 }) |*v, c| v.* = c;
    const x = try device.buffer(widest * h * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    fill(rng, x.slice(f32, widest * h));
    const act = try device.buffer(most * inter * 4, mtl.ResourceOptions.shared);
    defer act.deinit();
    fill(rng, act.slice(f32, most * inter));
    const picks = try device.buffer(most * 4, mtl.ResourceOptions.shared);
    defer picks.deinit();
    var outs: [2]mtl.Buffer = undefined;
    for (&outs) |*o| o.* = try device.buffer(most * h * 4, mtl.ResourceOptions.shared);
    defer for (outs) |o| o.deinit();
    var checked: usize = 0;
    var seen: std.StaticBitSet(widest + 1) = .empty;
    for (0..4 * widest) |case| {
        const rows = case % widest + 1;
        const family = case / widest;
        const slots = rows * topk;
        const ps = picks.slice(i32, most);
        if (family >= 2) {
            countedPicks(rng, rows, family == 3, experts, ps[0..slots]);
        } else for (0..rows) |r| {
            var order: [experts]i32 = undefined;
            for (&order, 0..) |*o, i| o.* = @intCast(i);
            if (family == 0) rng.shuffle(i32, order[0..12]);
            for (0..topk) |k| ps[r * topk + k] = order[k];
        }
        var members: [experts]usize = @splat(0);
        for (ps[0..slots]) |e| members[@intCast(e)] += 1;
        for (members) |n| if (n > 0) seen.set(n);
        for ([_]bool{ true, false }) |up| {
            const args = if (up) MoeArgs{ .k = h, .n = inter, .slots = @intCast(slots), .topk = topk, .gate_cap = 30, .up_cap = 20 } else MoeArgs{ .k = inter, .n = h, .slots = @intCast(slots), .topk = topk, .gate_cap = 0, .up_cap = 0 };
            const gathered = (if (up) shared.gateUp(rows, inter, h, @intCast(slots)) else shared.downs(rows, h, inter, @intCast(slots))) orelse {
                if (up and rows == 1) continue;
                return error.TestUnexpectedResult;
            };
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            for (0..2) |side| {
                enc.setPipeline(if (side == 1) gathered.pipe else if (up) gu else down);
                enc.setBuffer(if (up) x else act, 0, 0);
                enc.setBuffer(if (up) w13 else w2, 0, 1);
                enc.setBuffer(if (up) s13 else s2, 0, 2);
                enc.setBuffer(amax, 0, 3);
                enc.setBuffer(den, 0, 4);
                enc.setBuffer(picks, 0, 5);
                enc.setValue(args, 6);
                enc.setBuffer(outs[side], 0, 7);
                enc.setBuffer(codes, 0, 8);
                if (side == 1) enc.dispatchGroups(gathered.groups, gathered.threads) else enc.dispatchGroups(mtl.Size.of(if (up) inter else h, slots, 1), mtl.Size.of(32, 1, 1));
            }
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("expert test: {s}\n", .{msg});
                return error.GpuFailed;
            }
            const count = slots * (if (up) @as(usize, inter) else h);
            if (!sameBits(outs[0].slice(f32, count), outs[1].slice(f32, count))) {
                std.debug.print("expert test rows={d} up={}: bytes differ\n", .{ rows, up });
                return error.TestUnexpectedResult;
            }
            checked += 1;
        }
    }
    for (1..widest + 1) |n| try std.testing.expect(seen.isSet(n));
    std.debug.print("gathered expert test: {d} cases byte-equal to the per-slot kernels, experts of 1 to {d} members\n", .{ checked, widest });
}

/// Picks where the experts' member counts run from `rows` down to 1 (or up from 1) and repeat, each row 8 distinct.
fn countedPicks(rng: std.Random, rows: usize, ascending: bool, comptime experts: usize, out: []i32) void {
    var counts: [experts]usize = undefined;
    var n: usize = 0;
    var total: usize = 0;
    var c: usize = if (ascending) 1 else rows;
    while (total < rows * 8) : (n += 1) {
        counts[n] = @min(c, rows * 8 - total);
        total += counts[n];
        c = if (ascending) (if (c == rows) 1 else c + 1) else (if (c == 1) rows else c - 1);
    }
    std.mem.sort(usize, counts[0..n], {}, std.sort.desc(usize));
    var ids: [experts]i32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    rng.shuffle(i32, &ids);
    var used: [16]usize = @splat(0);
    for (counts[0..n], ids[0..n]) |count, id| {
        var order: [16]usize = undefined;
        for (order[0..rows], 0..) |*o, r| o.* = r;
        rng.shuffle(usize, order[0..rows]);
        std.mem.sort(usize, order[0..rows], &used, struct {
            fn less(f: *const [16]usize, a: usize, b: usize) bool {
                return f[a] < f[b];
            }
        }.less);
        for (order[0..count]) |r| {
            out[r * 8 + used[r]] = id;
            used[r] += 1;
        }
    }
    for (0..rows) |r| {
        std.debug.assert(used[r] == 8);
        rng.shuffle(i32, out[r * 8 ..][0..8]);
    }
}

fn bf16Bits(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

const QkvArgs = extern struct { kv: u32, heads: u32, dim: u32, rel: u32, extent: u32, layout: u32, dyn: u32, tau: u32, norm: f32, q_threshold: f32, rel_threshold: f32, eps: f32, floor: f32, alpha: f32 };
const ConvArgs = extern struct { channels: u32, heads: u32, ring: u32, pos: u32, layout: u32, norm: f32, residual: u32, topk: u32 };
const Clip3Args = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, q_threshold: f32, rel_threshold: f32 };
const RelArgs = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, scale: f32, floor: f32, alpha: f32, eps: f32 };

const QkvRig = struct {
    rows: usize,
    bufs: [17]mtl.Buffer,

    const names = [_][]const u8{ "k", "v", "kw", "vw", "kd", "vd", "q", "qs", "r", "rs", "p", "rel", "kc", "vc", "rings", "kv", "captures" };

    fn init(device: mtl.Device, rng: std.Random, rows: usize) !QkvRig {
        const c = kv * dim;
        const sizes = [17]usize{ rows * c * 4, rows * c * 4, c * 4 * 2, c * 4 * 2, rows * kv * 4 * 4, rows * kv * 4 * 4, rows * heads * dim * 4, heads * dim * 2, rows * heads * drel * 4, heads * drel * 2, drel * extent * 2, rows * heads * (extent + 1) * 4, rows * c * 4, rows * c * 4, rows * 2 * 4 * c * 4, rows * 2 * c * 2, rows * 2 * c * 4 };
        var g: QkvRig = .{ .rows = rows, .bufs = undefined };
        var made: usize = 0;
        errdefer for (g.bufs[0..made]) |buf| buf.deinit();
        for (&g.bufs, sizes, 0..) |*buf, size, i| {
            buf.* = try device.buffer(size, mtl.ResourceOptions.shared);
            made += 1;
            switch (i) {
                2, 3, 7, 9, 10 => {
                    for (buf.slice(u16, size / 2)) |*v| v.* = bf16Bits(rng.float(f32) * 2 - 1);
                },
                11, 12, 13, 15, 16 => @memset(buf.slice(u8, size), 0),
                else => fill(rng, buf.slice(f32, size / 4)),
            }
        }
        return g;
    }

    fn deinit(g: *QkvRig) void {
        for (g.bufs) |buf| buf.deinit();
    }

    fn copy(g: *QkvRig, queue: mtl.Queue, from: *const QkvRig) !void {
        const cb = queue.commandBuffer();
        const blit = cb.blit();
        for (g.bufs, from.bufs) |dst, src| blit.copy(src, 0, dst, 0, src.length());
        blit.end();
        try finishQkv(cb);
    }

    fn ring(g: *const QkvRig, r: usize, site: usize) u64 {
        return g.bufs[14].gpuAddress() + (r * 2 + site) * 4 * kv * dim * 4;
    }

    fn row(g: *const QkvRig, r: usize, site: usize) u64 {
        return g.bufs[15].gpuAddress() + (r * 2 + site) * kv * dim * 2;
    }
};

fn finishQkv(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("qkv test: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

fn qkvArgs(full: bool) QkvArgs {
    const mul: u32 = if (full) 2 else 1;
    return .{ .kv = kv, .heads = heads, .dim = dim, .rel = drel, .extent = extent, .layout = @intFromBool(full), .dyn = @intFromBool(full), .tau = @intFromBool(full), .norm = 1.0 / @sqrt(@as(f32, @floatFromInt(mul * 4 + 1))), .q_threshold = 22.6, .rel_threshold = 8, .eps = 1e-6, .floor = floor, .alpha = alpha };
}

test "the joined key/value prologue keeps the convolutions', clips' and stores' bytes" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const opts = mtl.CompileOptions.mlx();
    const plib = try mtl.Library.fromSource(device, kernels.prefill, opts);
    defer plib.deinit();
    const glib = try mtl.Library.fromSource(device, @import("group.zig").source, opts);
    defer glib.deinit();
    var pipes: [6]mtl.Pipeline = undefined;
    const sources = [6]mtl.Library{ plib, plib, plib, glib, glib, glib };
    const kernel_names = [6][:0]const u8{ "i2p_batch_conv", "i2p_clip3", "i2p_batch_store_rel", "i2_batch_qkv", "i2v_conv", "i2v_qkv" };
    var made: usize = 0;
    defer for (pipes[0..made]) |p| p.deinit();
    for (&pipes, sources, kernel_names) |*p, lib, name| {
        p.* = try mtl.Pipeline.init(device, lib, name, false);
        made += 1;
    }
    const store_rel = try mtl.Pipeline.init(device, glib, "i2v_store_rel", false);
    defer store_rel.deinit();
    var prng = std.Random.DefaultPrng.init(0x716b_7673);
    const rng = prng.random();
    var checked: usize = 0;
    for ([_]usize{ 1, 3, 6 }) |rows| for ([_]bool{ false, true }) |full| for ([_]u32{ 0, 1, 2, 5, 1500 }) |start| for ([_]bool{ false, true }) |verify| {
        var base = try QkvRig.init(device, rng, rows);
        defer base.deinit();
        var sides: [2]QkvRig = undefined;
        sides[0] = try QkvRig.init(device, rng, rows);
        defer sides[0].deinit();
        sides[1] = try QkvRig.init(device, rng, rows);
        defer sides[1].deinit();
        for (&sides) |*g| try g.copy(queue, &base);
        const a = qkvArgs(full);
        const c: u32 = kv * dim;
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (sides, 0..) |g, side| {
            const B = g.bufs;
            for ([_]mtl.Buffer{ B[14], B[15], B[16] }) |buf| enc.useResource(buf, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
            if (side == 1) {
                enc.setPipeline(if (verify) pipes[5] else pipes[3]);
                for (0..11) |i| enc.setBuffer(B[i], 0, i);
                enc.setValue(a, 11);
                enc.setBuffer(B[11], 0, 12);
                if (verify) {
                    var vrows: [8]extern struct { k_ring: u64, v_ring: u64, pos: u32, start: u32, back: [4]u8, pad: u32 = 0 } = undefined;
                    for (0..rows) |r| {
                        var back: [4]u8 = @splat(0xff);
                        for (1..4) |k| if (k <= r) {
                            back[k - 1] = @intCast(r - k);
                        };
                        vrows[r] = .{ .k_ring = g.ring(0, 0), .v_ring = g.ring(0, 1), .pos = start + @as(u32, @intCast(r)), .start = start, .back = back };
                    }
                    enc.setBytes(std.mem.sliceAsBytes(vrows[0..rows]), 13);
                    enc.setBuffer(B[12], 0, 14);
                    enc.setBuffer(B[13], 0, 15);
                    enc.setBuffer(B[16], 0, 16);
                    enc.setBuffer(B[16], rows * c * 4, 17);
                    enc.setBuffer(B[15], 0, 18);
                    enc.setBuffer(B[15], rows * c * 2, 19);
                } else {
                    var brows: [8]extern struct { k_ring: u64, v_ring: u64, keys: u64, values: u64, pos: u32, pad: u32 = 0 } = undefined;
                    for (0..rows) |r| brows[r] = .{ .k_ring = g.ring(r, 0), .v_ring = g.ring(r, 1), .keys = g.row(r, 0), .values = g.row(r, 1), .pos = start + @as(u32, @intCast(r)) * 7 };
                    enc.setBytes(std.mem.sliceAsBytes(brows[0..rows]), 13);
                    enc.setBuffer(B[12], 0, 14);
                    enc.setBuffer(B[13], 0, 15);
                }
                enc.dispatchGroups(mtl.Size.of(kv + heads + (heads * (extent + 1) + 255) / 256, rows, 1), mtl.Size.of(256, 1, 1));
                continue;
            }
            for (0..2) |site| {
                const conv = ConvArgs{ .channels = c, .heads = if (full) kv else 0, .ring = 4, .pos = 0, .layout = @intFromBool(full), .norm = a.norm, .residual = 0, .topk = 0 };
                enc.setPipeline(if (verify) pipes[4] else pipes[0]);
                enc.setBuffer(B[site], 0, 0);
                enc.setBuffer(B[2 + site], 0, 1);
                enc.setBuffer(B[4 + site], 0, 2);
                enc.setValue(conv, 3);
                enc.setBuffer(B[12 + site], 0, 4);
                if (verify) {
                    var vrows: [8]extern struct { ring: u64, pos: u32, start: u32, back: [4]u8 } = undefined;
                    for (0..rows) |r| {
                        var back: [4]u8 = @splat(0xff);
                        for (1..4) |k| if (k <= r) {
                            back[k - 1] = @intCast(r - k);
                        };
                        vrows[r] = .{ .ring = g.ring(0, site), .pos = start + @as(u32, @intCast(r)), .start = start, .back = back };
                    }
                    enc.setBytes(std.mem.sliceAsBytes(vrows[0..rows]), 5);
                    enc.setBuffer(B[16], site * rows * c * 4, 6);
                } else {
                    var brows: [8]extern struct { ring: u64, pos: u32, pad: u32 = 0 } = undefined;
                    for (0..rows) |r| brows[r] = .{ .ring = g.ring(r, site), .pos = start + @as(u32, @intCast(r)) * 7 };
                    enc.setBytes(std.mem.sliceAsBytes(brows[0..rows]), 5);
                }
                enc.setBuffer(B[site], 0, if (verify) 7 else 6);
                enc.dispatchThreads(mtl.Size.of(c, rows, 1), mtl.Size.of(256, 1, 1));
                if (site == 1) continue;
                enc.setPipeline(pipes[1]);
                for ([_]usize{ 6, 7, 8, 9, 12 }, 0..) |i, j| enc.setBuffer(B[i], 0, j);
                enc.setValue(Clip3Args{ .heads = heads, .kv = kv, .dim = dim, .rel = drel, .q_threshold = a.q_threshold, .rel_threshold = a.rel_threshold }, 5);
                enc.dispatchGroups(mtl.Size.of((2 * heads + kv) * rows, 1, 1), mtl.Size.of(256, 1, 1));
            }
            const rel = RelArgs{ .heads = heads, .kv = kv, .dim = dim, .rel = drel, .extent = extent, .window = 0, .pos = 0, .tau = @intFromBool(full), .scale = 0, .floor = floor, .alpha = alpha, .eps = a.eps };
            enc.setPipeline(if (verify) store_rel else pipes[2]);
            for ([_]usize{ 12, 13, 8, 10 }, 0..) |i, j| enc.setBuffer(B[i], 0, j);
            enc.setValue(rel, 4);
            enc.setBuffer(B[11], 0, 5);
            if (verify) {
                var positions: [8]u32 = undefined;
                for (0..rows) |r| positions[r] = start + @as(u32, @intCast(r));
                enc.setBytes(std.mem.sliceAsBytes(positions[0..rows]), 6);
                enc.setBuffer(B[15], 0, 7);
                enc.setBuffer(B[15], rows * c * 2, 8);
            } else {
                var srows: [8]extern struct { keys: u64, values: u64, pos: u32, pad: u32 = 0 } = undefined;
                for (0..rows) |r| srows[r] = .{ .keys = g.row(r, 0), .values = g.row(r, 1), .pos = start + @as(u32, @intCast(r)) * 7 };
                enc.setBytes(std.mem.sliceAsBytes(srows[0..rows]), 6);
            }
            enc.dispatchGroups(mtl.Size.of(kv + (heads * (extent + 1) + 255) / 256, rows, 1), mtl.Size.of(256, 1, 1));
        }
        enc.end();
        try finishQkv(cb);
        for ([_]usize{ 6, 11, 12, 13, 14, 15, 16 }) |i| {
            const bytes = sides[0].bufs[i].length();
            if (!std.mem.eql(u8, sides[0].bufs[i].slice(u8, bytes), sides[1].bufs[i].slice(u8, bytes))) {
                std.debug.print("qkv test rows={d} full={} start={d} verify={}: {s} differs\n", .{ rows, full, start, verify, QkvRig.names[i] });
                return error.TestUnexpectedResult;
            }
        }
        checked += 1;
    };
    std.debug.print("joined key/value prologue: {d} cases (batch and verify, local and full, positions from 0) byte-equal to the separate kernels\n", .{checked});
}
