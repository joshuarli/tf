const std = @import("std");
const mtl = @import("metal");
const ref = @import("fp4_ref.zig");
const shapes = @import("fp4_shapes.zig");
const synth = @import("fp4_synth.zig");

pub const names = [_][]const u8{ "baseline", "float", "one_pre", "two_pre", "three_pre", "one_packed", "two_packed", "three_packed" };
const Args = extern struct { k: u32, n: u32, slots: u32, topk: u32, experts: u32, gate_cap: f32, up_cap: f32 };
const opts = mtl.ResourceOptions.shared;

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}
fn unbf(v: u16) f32 { return @bitCast(@as(u32, v) << 16); }

pub const Fixture = struct {
    device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, baseline_lib: mtl.Library,
    convert: mtl.Pipeline, float_pipe: mtl.Pipeline, pre: mtl.Pipeline, packed_pipe: mtl.Pipeline, baseline: mtl.Pipeline,
    x: mtl.Buffer, parts: mtl.Buffer, w: mtl.Buffer, s: mtl.Buffer, bw: mtl.Buffer, y: mtl.Buffer,
    off: mtl.Buffer, order: mtl.Buffer, amax: mtl.Buffer, den: mtl.Buffer, codes: mtl.Buffer,
    k: u32, n: u32, rows: u32, fmt: shapes.Format,

    pub fn init(k: u32, rows: u32, fmt: shapes.Format, compiled: bool) !Fixture {
        const n: u32 = 2304;
        const device = try mtl.Device.init();
        const queue = try device.queue();
        const lib = if (compiled) try mtl.Library.fromFile(device, ".work/tmp/split_tensorops.metallib") else try mtl.Library.fromSource(device, @embedFile("split_tensorops.metal"), mtl.CompileOptions.mlx());
        const bs = if (fmt == .mxfp8) "#define I2P_MXFP8 1\n" ++ @embedFile("split_baseline.metal") else @embedFile("split_baseline.metal");
        const baseline_lib = if (compiled) try mtl.Library.fromFile(device, if (fmt == .mxfp8) ".work/tmp/split_mx.metallib" else ".work/tmp/split_nv.metallib") else try mtl.Library.fromSource(device, bs, mtl.CompileOptions.mlx());
        const grouped = fmt == .nvfp4 and rows < 16;
        var p = Fixture{
            .device = device, .queue = queue, .lib = lib, .baseline_lib = baseline_lib,
            .convert = try mtl.Pipeline.init(device, lib, "split_convert", false),
            .float_pipe = try mtl.Pipeline.init(device, lib, "split_float", false),
            .pre = try mtl.Pipeline.init(device, lib, "split_preconverted", false),
            .packed_pipe = try mtl.Pipeline.init(device, lib, "split_packed", false),
            .baseline = try mtl.Pipeline.init(device, baseline_lib, if (grouped) "i2p_down_grouped" else "i2p_expert_matrix", false),
            .x = try device.buffer(@as(usize, k) * rows * 4, opts),
            .parts = try device.buffer(@as(usize, k) * rows * 6, opts),
            .w = try device.buffer(@as(usize, k) * n / (if (fmt == .nvfp4) @as(usize, 2) else 1), opts),
            .s = try device.buffer(@as(usize, k) * n / (if (fmt == .nvfp4) @as(usize, 16) else 32), opts),
            .bw = try device.buffer(@as(usize, k) * n * 2, opts),
            .y = try device.buffer(@as(usize, n) * rows * 4, opts),
            .off = try device.buffer(8, opts), .order = try device.buffer(@as(usize, rows) * 4, opts),
            .amax = try device.buffer(4, opts), .den = try device.buffer(4, opts),
            .codes = try device.buffer(256 * 4, opts), .k = k, .n = n, .rows = rows, .fmt = fmt,
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
        const bw = p.bw.slice(u16, @as(usize, k) * n);
        for (0..n) |col| for (0..k) |i| {
            const weight = if (fmt == .nvfp4) blk: {
                const code = w[col * (k / 2) + i / 2];
                break :blk ref.e2m1(if (i % 2 == 0) code & 15 else code >> 4) * ref.e4m3(s[col * (k / 16) + i / 16]);
            } else ref.e4m3(w[col * k + i]) * ref.e8m0(s[col * (k / 32) + i / 32]);
            const b = bf16(@floatCast(weight));
            if (@as(f64, unbf(b)) != weight) return error.InexactWeight;
            bw[col * k + i] = b;
        };
        return p;
    }
    pub fn deinit(p: *Fixture) void {
        for ([_]mtl.Buffer{ p.x, p.parts, p.w, p.s, p.bw, p.y, p.off, p.order, p.amax, p.den, p.codes }) |b| b.deinit();
        p.convert.deinit(); p.float_pipe.deinit(); p.pre.deinit(); p.packed_pipe.deinit(); p.baseline.deinit();
        p.lib.deinit(); p.baseline_lib.deinit(); p.queue.deinit(); p.device.deinit();
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
        const terms: u32 = if (arm >= 2) @intCast((arm - 2) % 3 + 1) else 0;
        if (arm >= 2) {
            const e = cb.compute(.serial); e.setPipeline(p.convert); e.setBuffer(p.x, 0, 0); e.setBuffer(p.parts, 0, 1);
            e.setValue(p.k * p.rows, 2); e.setValue(terms, 3);
            e.dispatchThreads(mtl.Size.of(p.k * p.rows, 1, 1), mtl.Size.of(256, 1, 1)); e.end();
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
        } else {
            e.setPipeline(p.packed_pipe); e.setBuffer(p.parts, 0, 0); e.setBuffer(p.w, 0, 1); e.setBuffer(p.s, 0, 2); e.setBuffer(p.y, 0, 3);
            e.setValue(p.k, 4); e.setValue(p.n, 5); e.setValue(p.rows, 6); e.setValue(terms, 7); e.setValue(@as(u32, @intFromBool(p.fmt == .mxfp8)), 8);
            e.dispatchGroups(mtl.Size.of((p.n + 31) / 32, (p.rows + 15) / 16, 1), mtl.Size.of(32, 1, 1));
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
            _ = try p.run(arm);
            @memcpy(outputs[arm * count ..][0..count], p.y.slice(f32, count));
        }
        var stats: [names.len]ErrorStats = @splat(.{});
        try evaluate(&p, outputs, &stats);
        for (stats, 0..) |s, arm| {
            if (s.failed != 0) std.debug.print("exact fmt={s} k={d} arm={s} failed={d} ratio={d}\n", .{ fmt.name(), k, names[arm], s.failed, s.ratio });
            try std.testing.expectEqual(@as(usize, 0), s.failed);
        }
        p.fill(false);
        for (0..names.len) |arm| {
            _ = try p.run(arm);
            @memcpy(outputs[arm * count ..][0..count], p.y.slice(f32, count));
        }
        stats = @splat(.{});
        try evaluate(&p, outputs, &stats);
        for (stats, 0..) |s, arm| {
            if (arm == 2 or arm == 5) continue;
            if (s.failed != 0) std.debug.print("random fmt={s} k={d} rows={d} arm={s} failed={d} ratio={d}\n", .{ fmt.name(), k, rows, names[arm], s.failed, s.ratio });
            try std.testing.expectEqual(@as(usize, 0), s.failed);
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
