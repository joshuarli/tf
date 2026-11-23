const std = @import("std");
const mtl = @import("metal");
const prefill = @import("prefill.zig");

const t = std.testing;

fn half(x: f64) f64 {
    return @floatCast(@as(f16, @floatCast(x)));
}

fn bind(enc: mtl.ComputeEncoder, bufs: []const mtl.Buffer) void {
    for (bufs, 0..) |buf, i| enc.setBuffer(buf, 0, i);
}

fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("attention units: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

const AttnArgs = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, base: u32, rows: u32, rmode: u32, scale: f32, floor: f32, alpha: f32 };

test "tensor-unit attention meets fp64 bounds across windows, cached keys and row tails" {
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, prefill.units_source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const units = try mtl.Pipeline.init(device, lib, "i2p_attn_units", false);
    defer units.deinit();
    const simd = try mtl.Pipeline.init(device, lib, "i2p_attn_half", false);
    defer simd.deinit();
    const heads = 24;
    const kv = 8;
    const dim = 128;
    const gamma = struct {
        fn of(k: usize) f64 {
            const u = @as(f64, @floatFromInt(k + 2)) * 0x1p-24;
            return u / (1.0 - u);
        }
    }.of;
    for ([_]AttnArgs{
        .{ .heads = heads, .kv = kv, .dim = dim, .rel = 16, .extent = 48, .window = 48, .base = 100, .rows = 45, .rmode = 1, .scale = 1.0 / @sqrt(128.0), .floor = 16, .alpha = 0.1 },
        .{ .heads = heads, .kv = kv, .dim = dim, .rel = 16, .extent = 64, .window = 0, .base = 70, .rows = 37, .rmode = 0, .scale = 1.0 / @sqrt(128.0), .floor = 16, .alpha = 0.1 },
    }) |args| {
        const rows: usize = args.rows;
        const base: usize = args.base;
        const cols: usize = args.extent + 1;
        const cache_rows: usize = if (args.window > 0) args.window else base;
        const q = try device.buffer(rows * heads * dim * 4, mtl.ResourceOptions.shared);
        defer q.deinit();
        for (q.slice(f32, rows * heads * dim), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i *% 31 + i / dim *% 7) % 197)) - 98)) / 23.0;
        const keys = try device.buffer(cache_rows * kv * dim * 2, mtl.ResourceOptions.shared);
        defer keys.deinit();
        const values = try device.buffer(cache_rows * kv * dim * 2, mtl.ResourceOptions.shared);
        defer values.deinit();
        const kc = try device.buffer(rows * kv * dim * 2, mtl.ResourceOptions.shared);
        defer kc.deinit();
        const vc = try device.buffer(rows * kv * dim * 2, mtl.ResourceOptions.shared);
        defer vc.deinit();
        const keyAt = struct {
            fn of(j: usize, g: usize, d: usize, value: bool) f16 {
                const salt: usize = if (value) 13 else 0;
                return @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((j *% 17 + g *% 5 + d *% 3 + salt) % 89)) - 44)) / 29.0);
            }
        }.of;
        for (0..base + rows) |j| for (0..kv) |g| for (0..dim) |d| {
            if (j < base) {
                if (args.window > 0 and j + args.window < base) continue;
                const r = if (args.rmode != 0) j % args.window else j;
                keys.slice(f16, cache_rows * kv * dim)[(r * kv + g) * dim + d] = keyAt(j, g, d, false);
                values.slice(f16, cache_rows * kv * dim)[(r * kv + g) * dim + d] = keyAt(j, g, d, true);
            } else {
                kc.slice(f16, rows * kv * dim)[((j - base) * kv + g) * dim + d] = keyAt(j, g, d, false);
                vc.slice(f16, rows * kv * dim)[((j - base) * kv + g) * dim + d] = keyAt(j, g, d, true);
            }
        };
        const bias = try device.buffer(rows * heads * cols * 4, mtl.ResourceOptions.shared);
        defer bias.deinit();
        for (bias.slice(f32, rows * heads * cols), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i *% 11) % 61)) - 30)) / 37.0;
        const sink = try device.buffer(rows * heads * 4, mtl.ResourceOptions.shared);
        defer sink.deinit();
        for (sink.slice(f32, rows * heads), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 23)) - 11)) / 5.0;
        const overflow = try device.buffer(16, mtl.ResourceOptions.shared);
        defer overflow.deinit();
        var outs: [2]mtl.Buffer = undefined;
        for (&outs, 0..) |*out, side| {
            out.* = try device.buffer(rows * heads * dim * 4, mtl.ResourceOptions.shared);
            @memset(out.slice(f32, rows * heads * dim), std.math.nan(f32));
            overflow.slice(u32, 1)[0] = 0;
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(if (side == 0) units else simd);
            bind(enc, &.{ q, keys, values, bias, sink });
            enc.setValue(args, 5);
            enc.setBuffer(out.*, 0, 6);
            enc.setBuffer(kc, 0, 7);
            enc.setBuffer(vc, 0, 8);
            enc.setBuffer(overflow, 0, 9);
            if (side == 0) enc.dispatchGroups(mtl.Size.of(kv, (rows + 31) / 32, 1), mtl.Size.of(192, 1, 1)) else enc.dispatchGroups(mtl.Size.of(heads, (rows + 31) / 32, 1), mtl.Size.of(128, 1, 1));
            enc.end();
            try finish(cb);
            try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
        }
        defer for (outs) |out| out.deinit();
        var worst = [_]f64{ 0, 0 };
        for (0..rows) |row| for (0..heads) |h| {
            const g = h / (heads / kv);
            const pos = base + row;
            const tau: f64 = if (args.window > 0) 1 else 1 + @as(f64, args.alpha) * @log(@max(1.0, (@as(f64, @floatFromInt(pos)) + 1) / @as(f64, args.floor)));
            const lo = if (args.window > 0 and pos + 1 > args.window) pos + 1 - args.window else 0;
            var top: f64 = 0;
            var scores: [256]f64 = undefined;
            var score_error: f64 = 0;
            for (lo..pos + 1) |j| {
                var dot: f64 = 0;
                var mag: f64 = 0;
                for (0..dim) |d| {
                    const term = half(q.slice(f32, rows * heads * dim)[(row * heads + h) * dim + d]) * @as(f64, @floatCast(keyAt(j, g, d, false)));
                    dot += term;
                    mag += @abs(term);
                }
                const dist = @min(pos - j, args.extent);
                const sc = dot * @as(f64, args.scale) * tau + @as(f64, bias.slice(f32, rows * heads * cols)[(row * heads + h) * cols + dist]) - @as(f64, sink.slice(f32, rows * heads)[row * heads + h]);
                scores[j - lo] = sc;
                top = @max(top, sc);
                score_error = @max(score_error, gamma(dim) * mag * @as(f64, args.scale) * tau + 8 * 0x1p-24 * (@abs(sc) + mag * @as(f64, args.scale) * tau));
            }
            var den: f64 = @exp(-top);
            for (lo..pos + 1) |j| den += @exp(scores[j - lo] - top);
            var vmax: f64 = 0;
            for (0..dim) |d| {
                var num: f64 = 0;
                for (lo..pos + 1) |j| {
                    const v: f64 = @floatCast(keyAt(j, g, d, true));
                    vmax = @max(vmax, @abs(v));
                    num += @exp(scores[j - lo] - top) * v;
                }
                const want = num / den;
                for (outs, 0..) |out, side| {
                    const got: f64 = out.slice(f32, rows * heads * dim)[(row * heads + h) * dim + d];
                    worst[side] = @max(worst[side], @abs(got - want));
                    const bound = 2 * (0x1p-11 + score_error) * vmax + 1e-5;
                    if (!(@abs(got - want) <= bound)) {
                        std.debug.print("attention side={d} base={d} window={d} row={d} head={d} dim={d} got={e} want={e} bound={e}\n", .{ side, base, args.window, row, h, d, got, want, bound });
                        return error.AttentionReferenceMismatch;
                    }
                }
            }
        };
        std.debug.print("attention base={d} window={d} rows={d}: units worst {e}, simdgroup worst {e}\n", .{ base, args.window, rows, worst[0], worst[1] });
    }
}
