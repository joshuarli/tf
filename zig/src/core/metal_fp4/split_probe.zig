const std = @import("std");
const mtl = @import("metal");
const ref = @import("fp4_ref.zig");
const shapes = @import("fp4_shapes.zig");
const synth = @import("fp4_synth.zig");

pub const names = [_][]const u8{ "baseline", "float", "one_pre", "two_pre", "three_pre", "one_packed", "two_packed", "three_packed", "half_half", "half_mxfp8", "fp8_mxfp8", "half_nvfp4" };
const Args = extern struct { k: u32, n: u32, slots: u32, topk: u32, experts: u32, gate_cap: f32, up_cap: f32 };
const opts = mtl.ResourceOptions.shared;

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}
fn unbf(v: u16) f32 { return @bitCast(@as(u32, v) << 16); }

pub const Fixture = struct {
    device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, baseline_lib: mtl.Library, fast_lib: mtl.Library, more_lib: mtl.Library,
    convert: mtl.Pipeline, float_pipe: mtl.Pipeline, pre: mtl.Pipeline, packed_pipe: mtl.Pipeline, baseline: mtl.Pipeline,
    half_convert: mtl.Pipeline, half_half: mtl.Pipeline, half_mxfp8: mtl.Pipeline,
    fp8_quant: mtl.Pipeline, fp8_mxfp8: mtl.Pipeline, half_mask: mtl.Pipeline, half_nvfp4: mtl.Pipeline,
    x: mtl.Buffer, parts: mtl.Buffer, w: mtl.Buffer, padded_w: mtl.Buffer, s: mtl.Buffer, bw: mtl.Buffer, y: mtl.Buffer, half_x: mtl.Buffer, half_w: mtl.Buffer, qx: mtl.Buffer, qscale: mtl.Buffer,
    off: mtl.Buffer, order: mtl.Buffer, amax: mtl.Buffer, den: mtl.Buffer, codes: mtl.Buffer,
    k: u32, n: u32, rows: u32, fmt: shapes.Format, half_inexact: usize,

    pub fn init(k: u32, rows: u32, fmt: shapes.Format, compiled: bool) !Fixture {
        const n: u32 = 2304;
        const device = try mtl.Device.init();
        const queue = try device.queue();
        const lib = if (compiled) try mtl.Library.fromFile(device, ".work/tmp/split_tensorops.metallib") else try mtl.Library.fromSource(device, @embedFile("split_tensorops.metal"), mtl.CompileOptions.mlx());
        const bs = if (fmt == .mxfp8) "#define I2P_MXFP8 1\n" ++ @embedFile("split_baseline.metal") else @embedFile("split_baseline.metal");
        const baseline_lib = if (compiled) try mtl.Library.fromFile(device, if (fmt == .mxfp8) ".work/tmp/split_mx.metallib" else ".work/tmp/split_nv.metallib") else try mtl.Library.fromSource(device, bs, mtl.CompileOptions.mlx());
        const fast_lib = if (compiled) try mtl.Library.fromFile(device, ".work/tmp/split_fast.metallib") else try mtl.Library.fromSource(device, @embedFile("split_fast.metal"), mtl.CompileOptions.mlx());
        const more_lib = if (compiled) try mtl.Library.fromFile(device, ".work/tmp/split_more.metallib") else try mtl.Library.fromSource(device, @embedFile("split_more.metal"), mtl.CompileOptions.mlx());
        const grouped = fmt == .nvfp4 and rows < 16;
        var p = Fixture{
            .device = device, .queue = queue, .lib = lib, .baseline_lib = baseline_lib, .fast_lib = fast_lib, .more_lib = more_lib,
            .convert = try mtl.Pipeline.init(device, lib, "split_convert", false),
            .float_pipe = try mtl.Pipeline.init(device, lib, "split_float", false),
            .pre = try mtl.Pipeline.init(device, lib, "split_preconverted", false),
            .packed_pipe = try mtl.Pipeline.init(device, lib, "split_packed", false),
            .baseline = try mtl.Pipeline.init(device, baseline_lib, if (grouped) "i2p_down_grouped" else "i2p_expert_matrix", false),
            .half_convert = try mtl.Pipeline.init(device, fast_lib, "fast_half_convert", false),
            .half_half = try mtl.Pipeline.init(device, fast_lib, "fast_half_half", false),
            .half_mxfp8 = try mtl.Pipeline.init(device, fast_lib, "fast_half_mxfp8", false),
            .fp8_quant = try mtl.Pipeline.init(device, more_lib, "fast_fp8_quant", false),
            .fp8_mxfp8 = try mtl.Pipeline.init(device, more_lib, "fast_fp8_mxfp8", false),
            .half_mask = try mtl.Pipeline.init(device, more_lib, "fast_half_mask", false),
            .half_nvfp4 = try mtl.Pipeline.init(device, more_lib, "fast_half_nvfp4", false),
            .x = try device.buffer(@as(usize, k) * rows * 4, opts),
            .parts = try device.buffer(@as(usize, k) * rows * 6, opts),
            .w = try device.buffer(@as(usize, k) * n / (if (fmt == .nvfp4) @as(usize, 2) else 1), opts),
            .padded_w = try device.buffer(@as(usize, (k + 255) & ~@as(u32, 255)) * n / 2, opts),
            .s = try device.buffer(@as(usize, k) * n / (if (fmt == .nvfp4) @as(usize, 16) else 32), opts),
            .bw = try device.buffer(@as(usize, k) * n * 2, opts),
            .y = try device.buffer(@as(usize, n) * rows * 4, opts),
            .half_x = try device.buffer(@as(usize, k) * rows * 4, opts),
            .half_w = try device.buffer(@as(usize, k) * n * 2, opts),
            .qx = try device.buffer(@as(usize, k) * rows, opts),
            .qscale = try device.buffer(@as(usize, k) * rows / 32 * 4, opts),
            .off = try device.buffer(8, opts), .order = try device.buffer(@as(usize, rows) * 4, opts),
            .amax = try device.buffer(4, opts), .den = try device.buffer(4, opts),
            .codes = try device.buffer(256 * 4, opts), .k = k, .n = n, .rows = rows, .fmt = fmt, .half_inexact = 0,
        };
        p.off.slice(i32, 2)[0] = 0; p.off.slice(i32, 2)[1] = @intCast(rows);
        for (p.order.slice(i32, rows), 0..) |*v, i| v.* = @intCast(i);
        p.amax.slice(f32, 1)[0] = 1; p.den.slice(f32, 1)[0] = 1;
        for (p.codes.slice(f32, 256), 0..) |*v, i| v.* = @floatCast(if (fmt == .nvfp4) ref.nvfp4_codes[i & 15] else ref.e4m3(@intCast(i)));
        var random = std.Random.DefaultPrng.init(0x5c00 + k + @as(u64, @intFromEnum(fmt)) * 911);
        const code_count = @as(usize, k) * n / (if (fmt == .nvfp4) @as(usize, 2) else 1);
        const scale_count = @as(usize, k) * n / (if (fmt == .nvfp4) @as(usize, 16) else 32);
        const w = p.w.slice(u8, code_count); const s = p.s.slice(u8, scale_count);
        _ = synth.fillPair(random.random(), fmt, w, s);
        if (fmt == .nvfp4) {
            const stride = @as(usize, (k + 255) & ~@as(u32, 255)) / 2;
            const padded = p.padded_w.slice(u8, stride * n);
            @memset(padded, 0);
            for (0..n) |col| @memcpy(padded[col * stride ..][0 .. k / 2], w[col * (k / 2) ..][0 .. k / 2]);
        }
        const bw = p.bw.slice(u16, @as(usize, k) * n);
        for (0..n) |col| for (0..k) |i| {
            const weight = if (fmt == .nvfp4) blk: {
                const code = w[col * (k / 2) + i / 2];
                break :blk ref.e2m1(if (i % 2 == 0) code & 15 else code >> 4) * ref.e4m3(s[col * (k / 16) + i / 16]);
            } else ref.e4m3(w[col * k + i]) * ref.e8m0(s[col * (k / 32) + i / 32]);
            const b = bf16(@floatCast(weight));
            if (@as(f64, unbf(b)) != weight) return error.InexactWeight;
            bw[col * k + i] = b;
            const h: f16 = @floatCast(weight);
            if (@as(f64, h) != weight) p.half_inexact += 1;
            p.half_w.slice(f16, @as(usize, k) * n)[col * k + i] = h;
        };
        return p;
    }
    pub fn deinit(p: *Fixture) void {
        for ([_]mtl.Buffer{ p.x, p.parts, p.w, p.padded_w, p.s, p.bw, p.y, p.off, p.order, p.amax, p.den, p.codes, p.half_x, p.half_w, p.qx, p.qscale }) |b| b.deinit();
        p.convert.deinit(); p.float_pipe.deinit(); p.pre.deinit(); p.packed_pipe.deinit(); p.baseline.deinit();
        p.half_convert.deinit(); p.half_half.deinit(); p.half_mxfp8.deinit();
        p.fp8_quant.deinit(); p.fp8_mxfp8.deinit(); p.half_mask.deinit(); p.half_nvfp4.deinit();
        p.lib.deinit(); p.baseline_lib.deinit(); p.fast_lib.deinit(); p.more_lib.deinit(); p.queue.deinit(); p.device.deinit();
    }
    pub fn supported(p: *Fixture, arm: usize) bool {
        return switch (arm) { 9, 10 => p.fmt == .mxfp8, 11 => p.fmt == .nvfp4, else => true };
    }
    pub fn fill(p: *Fixture, exact: bool) void {
        var random = std.Random.DefaultPrng.init(0x7350 + p.k + p.rows);
        for (p.x.slice(f32, @as(usize, p.k) * p.rows)) |*v| {
            const x = random.random().float(f32) * 2 - 1;
            v.* = if (exact) unbf(bf16(x)) else x;
        }
    }
    pub fn run(p: *Fixture, arm: usize) !f64 {
        const pool = mtl.objc.Pool.push(); defer pool.pop();
        const cb = p.queue.commandBuffer();
        const terms: u32 = if (arm >= 2 and arm < 8) @intCast((arm - 2) % 3 + 1) else 0;
        if (arm >= 2 and arm < 8) {
            const e = cb.compute(.serial); e.setPipeline(p.convert); e.setBuffer(p.x, 0, 0); e.setBuffer(p.parts, 0, 1);
            e.setValue(p.k * p.rows, 2); e.setValue(terms, 3);
            e.dispatchThreads(mtl.Size.of(p.k * p.rows, 1, 1), mtl.Size.of(256, 1, 1)); e.end();
        }
        if (arm == 8 or arm == 9 or arm == 11) {
            const e = cb.compute(.serial); e.setPipeline(if (arm == 11) p.half_mask else p.half_convert); e.setBuffer(p.x, 0, 0); e.setBuffer(p.half_x, 0, 1);
            e.setValue(p.k * p.rows, 2); e.dispatchThreads(mtl.Size.of(p.k * p.rows, 1, 1), mtl.Size.of(256, 1, 1)); e.end();
        }
        if (arm == 10) {
            const e = cb.compute(.serial); e.setPipeline(p.fp8_quant); e.setBuffer(p.x, 0, 0); e.setBuffer(p.qx, 0, 1); e.setBuffer(p.qscale, 0, 2);
            e.setValue(p.k, 3); e.setValue(p.rows, 4); e.dispatchThreads(mtl.Size.of(p.k * p.rows / 32, 1, 1), mtl.Size.of(256, 1, 1)); e.end();
        }
        const e = cb.compute(.serial);
        if (arm == 0) {
            e.setPipeline(p.baseline); e.setBuffer(p.x, 0, 0); e.setBuffer(p.w, 0, 1); e.setBuffer(p.s, 0, 2);
            e.setBuffer(p.amax, 0, 3); e.setBuffer(p.den, 0, 4); e.setBuffer(p.off, 0, 5); e.setBuffer(p.order, 0, 6);
            e.setValue(Args{ .k = p.k, .n = p.n, .slots = p.rows, .topk = 1, .experts = 1, .gate_cap = 0, .up_cap = 0 }, 7);
            e.setBuffer(p.y, 0, 8); e.setBuffer(p.codes, 0, 9);
            if (p.fmt == .nvfp4 and p.rows < 16) e.dispatchGroups(mtl.Size.of(p.n, 1, 1), mtl.Size.of(32, 1, 1)) else e.dispatchGroups(mtl.Size.of((p.n + 63) / 64, 1, (p.rows + 31) / 32), mtl.Size.of(256, 1, 1));
        } else if (arm == 1 or arm < 5) {
            e.setPipeline(if (arm == 1) p.float_pipe else p.pre);
            e.setBuffer(if (arm == 1) p.x else p.parts, 0, 0); e.setBuffer(p.bw, 0, 1); e.setBuffer(p.y, 0, 2);
            e.setValue(p.k, 3); e.setValue(p.n, 4); e.setValue(p.rows, 5); if (arm > 1) e.setValue(terms, 6);
            e.dispatchGroups(mtl.Size.of((p.n + 31) / 32, (p.rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        } else if (arm < 8) {
            e.setPipeline(p.packed_pipe); e.setBuffer(p.parts, 0, 0); e.setBuffer(p.w, 0, 1); e.setBuffer(p.s, 0, 2); e.setBuffer(p.y, 0, 3);
            e.setValue(p.k, 4); e.setValue(p.n, 5); e.setValue(p.rows, 6); e.setValue(terms, 7); e.setValue(@as(u32, @intFromBool(p.fmt == .mxfp8)), 8);
            e.dispatchGroups(mtl.Size.of((p.n + 31) / 32, (p.rows + 15) / 16, 1), mtl.Size.of(32, 1, 1));
        } else if (arm == 8) {
            e.setPipeline(p.half_half); e.setBuffer(p.half_x, 0, 0); e.setBuffer(p.half_w, 0, 1); e.setBuffer(p.y, 0, 2);
            e.setValue(p.k, 3); e.setValue(p.n, 4); e.setValue(p.rows, 5);
            e.dispatchGroups(mtl.Size.of((p.n + 31) / 32, (p.rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        } else if (arm == 9 or arm == 11) {
            e.setPipeline(if (arm == 9) p.half_mxfp8 else p.half_nvfp4); e.setBuffer(p.half_x, 0, 0); e.setBuffer(if (arm == 9) p.w else p.padded_w, 0, 1); e.setBuffer(p.s, 0, 2); e.setBuffer(p.y, 0, 3);
            e.setValue(p.k, 4); e.setValue(p.n, 5); e.setValue(p.rows, 6);
            e.dispatchGroups(mtl.Size.of((p.n + 31) / 32, (p.rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        } else {
            e.setPipeline(p.fp8_mxfp8); e.setBuffer(p.qx, 0, 0); e.setBuffer(p.w, 0, 1); e.setBuffer(p.s, 0, 2); e.setBuffer(p.qscale, 0, 3); e.setBuffer(p.y, 0, 4);
            e.setValue(p.k, 5); e.setValue(p.n, 6); e.setValue(p.rows, 7);
            e.dispatchGroups(mtl.Size.of((p.n + 31) / 32, (p.rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        }
        e.end(); cb.commit(); cb.wait();
        if (cb.failure()) |msg| { std.debug.print("GPU: {s}\n", .{msg}); return error.GpuFailure; }
        return cb.gpuSeconds() * 1e6;
    }
    pub fn net(p: *Fixture) ref.Net {
        return .{ .fmt = p.fmt, .k = p.k, .n = p.n, .rows = p.n,
            .codes = p.w.slice(u8, @as(usize, p.k) * p.n / (if (p.fmt == .nvfp4) @as(usize, 2) else 1)),
            .scales = p.s.slice(u8, @as(usize, p.k) * p.n / (if (p.fmt == .nvfp4) @as(usize, 16) else 32)), .global = 1 };
    }
};

pub const ErrorStats = struct { abs: f64 = 0, rel: f64 = 0, ratio: f64 = 0, failed: usize = 0 };
pub fn evaluate(p: *Fixture, outputs: []const f32, stats: *[names.len]ErrorStats) !void {
    const a = std.heap.c_allocator;
    const want = try a.alloc(f64, p.n); defer a.free(want);
    const bound = try a.alloc(f64, p.n); defer a.free(bound);
    const count = @as(usize, p.n) * p.rows;
    for (0..p.rows) |row| {
        ref.down(p.net(), p.x.slice(f32, @as(usize, p.k) * p.rows)[row * p.k ..][0..p.k], want, bound);
        for (0..names.len) |arm| for (0..p.n) |col| {
            if (!p.supported(arm)) continue;
            const got: f64 = outputs[arm * count + row * p.n + col];
            const err = @abs(got - want[col]);
            if (!std.math.isFinite(got)) return error.NonFinite;
            const s = &stats[arm]; s.abs = @max(s.abs, err); s.rel = @max(s.rel, err / @max(@abs(want[col]), 1e-30));
            s.ratio = @max(s.ratio, err / @max(bound[col], 1e-30)); if (err > bound[col]) s.failed += 1;
        };
    }
}

test "split arms meet fp64 bounds on exact bf16 inputs and detect corrupt weights" {
    const pool = mtl.objc.Pool.push(); defer pool.pop();
    const a = std.testing.allocator;
    for ([_]shapes.Format{ .nvfp4, .mxfp8 }) |fmt| for ([_]u32{ 1152, 2304 }) |k| for ([_]u32{ 8, 32 }) |rows| {
        var p = try Fixture.init(k, rows, fmt, false); defer p.deinit();
        p.fill(true);
        const count = @as(usize, p.n) * p.rows;
        const outputs = try a.alloc(f32, count * names.len); defer a.free(outputs);
        for (0..names.len) |arm| {
            if (!p.supported(arm)) continue;
            _ = try p.run(arm);
            @memcpy(outputs[arm * count ..][0..count], p.y.slice(f32, count));
        }
        var stats: [names.len]ErrorStats = @splat(.{});
        try evaluate(&p, outputs, &stats);
        for (stats, 0..) |s, arm| {
            if (arm == 10) continue;
            if (s.failed != 0) std.debug.print("exact fmt={s} k={d} arm={s} failed={d} ratio={d}\n", .{ fmt.name(), k, names[arm], s.failed, s.ratio });
            try std.testing.expectEqual(@as(usize, 0), s.failed);
        }
        p.fill(false);
        for (0..names.len) |arm| {
            if (!p.supported(arm)) continue;
            _ = try p.run(arm);
            @memcpy(outputs[arm * count ..][0..count], p.y.slice(f32, count));
        }
        stats = @splat(.{});
        try evaluate(&p, outputs, &stats);
        for (stats, 0..) |s, arm| {
            if (arm == 2 or arm == 5 or arm >= 8) continue;
            if (s.failed != 0) std.debug.print("random fmt={s} k={d} rows={d} arm={s} failed={d} ratio={d}\n", .{ fmt.name(), k, rows, names[arm], s.failed, s.ratio });
            try std.testing.expectEqual(@as(usize, 0), s.failed);
        }
        if (fmt == .mxfp8) {
            const quant_x = try a.alloc(f32, k); defer a.free(quant_x);
            const quant_want = try a.alloc(f64, p.n); defer a.free(quant_want);
            const quant_bound = try a.alloc(f64, p.n); defer a.free(quant_bound);
            const qx = p.qx.slice(u8, @as(usize, k) * rows);
            const scales = p.qscale.slice(f32, @as(usize, k) * rows / 32);
            for (0..rows) |row| {
                for (quant_x, 0..) |*v, i| {
                    const scale = scales[row * (k / 32) + i / 32];
                    v.* = @floatCast(ref.e4m3(qx[row * k + i]) * @as(f64, scale));
                    try std.testing.expect(@abs(v.* - p.x.slice(f32, @as(usize, k) * rows)[row * k + i]) <= 17 * scale);
                }
                ref.down(p.net(), quant_x, quant_want, quant_bound);
                for (0..p.n) |i| try std.testing.expect(@abs(@as(f64, outputs[10 * count + row * p.n + i]) - quant_want[i]) <= quant_bound[i]);
            }
        }
        for (p.bw.slice(u16, @as(usize, k) * p.n)) |*w| w.* = bf16(unbf(w.*) * 2);
        _ = try p.run(1);
        const want = try a.alloc(f64, p.n); defer a.free(want);
        const bound = try a.alloc(f64, p.n); defer a.free(bound);
        ref.down(p.net(), p.x.slice(f32, @as(usize, p.k) * p.rows)[0..k], want, bound);
        var failures: usize = 0;
        for (0..p.n) |i| if (@abs(@as(f64, p.y.slice(f32, count)[i]) - want[i]) > bound[i]) { failures += 1; };
        try std.testing.expect(failures > p.n * 3 / 4);
    };
}
