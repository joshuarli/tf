const std = @import("std");
const mtl = @import("metal");
const kernels = @import("kernels.zig");
const rows_mod = @import("rows.zig");

const Dense = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, bits: u32, cap: f32 };
const Projections = extern struct { k: u32, full: u32, m: [8]u32 };
const Shared = extern struct { k: u32, full: u32, rows: u32, m: [8]u32 };

fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

fn affine(device: mtl.Device, rng: std.Random, n: usize, k: usize) ![3]mtl.Buffer {
    const w = try device.buffer(n * k, mtl.ResourceOptions.shared);
    errdefer w.deinit();
    const s = try device.buffer(n * (k / 64) * 2, mtl.ResourceOptions.shared);
    errdefer s.deinit();
    const b = try device.buffer(n * (k / 64) * 2, mtl.ResourceOptions.shared);
    for (w.slice(u8, n * k)) |*v| v.* = rng.int(u8);
    for (s.slice(u16, n * (k / 64))) |*v| v.* = bf16(rng.float(f32) * 0.02 + 0.001);
    for (b.slice(u16, n * (k / 64))) |*v| v.* = bf16(rng.float(f32) * 0.4 - 0.2);
    return .{ w, s, b };
}

fn inputs(device: mtl.Device, rng: std.Random, count: usize) !mtl.Buffer {
    const x = try device.buffer(count * 4, mtl.ResourceOptions.shared);
    for (x.slice(f32, count)) |*v| v.* = rng.float(f32) * 4 - 2;
    return x;
}

fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("few-row kernel test: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

fn sameBits(a: []const f32, b: []const f32) bool {
    for (a, b) |x, y| if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) return false;
    return true;
}

test "few-row affine projections give each row the one-row kernel's bytes" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const one = try kernels.AffinePipes.init(device);
    defer one.deinit();
    var shared = try rows_mod.Pipes.init(device);
    defer shared.deinit();
    var prng = std.Random.DefaultPrng.init(0x726f_7773);
    const rng = prng.random();
    var checked: usize = 0;
    for ([_][3]u32{ .{ 2312, 3072, 0 }, .{ 65600, 1024, 30 }, .{ 1034, 2304, 30 }, .{ 2306, 3072, 0 } }) |shape| {
        const n = shape[0];
        const k = shape[1];
        const cap: f32 = @floatFromInt(shape[2]);
        const w = try affine(device, rng, n, k);
        defer for (w) |buf| buf.deinit();
        const x = try inputs(device, rng, (rows_mod.max_rows + 1) * k);
        defer x.deinit();
        var outs: [2]mtl.Buffer = undefined;
        for (&outs) |*o| o.* = try device.buffer(rows_mod.max_rows * n * 4, mtl.ResourceOptions.shared);
        defer for (outs) |o| o.deinit();
        for (2..rows_mod.max_rows + 1) |rows| for ([_]u32{ 0, 1 }) |xb| {
            const args = Dense{ .n = n, .k = k, .rows = @intCast(rows), .xs = 1, .xb = xb, .bits = 8, .cap = cap };
            const d = shared.dense(rows, n, k, 8) orelse return error.TestUnexpectedResult;
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            for (0..2) |side| {
                enc.setPipeline(if (side == 0) one.dense else d.pipe);
                enc.setBuffer(x, 0, 0);
                for (w, 1..) |buf, i| enc.setBuffer(buf, 0, i);
                enc.setValue(args, 4);
                enc.setBuffer(outs[side], 0, 5);
                if (side == 0) enc.dispatchGroups(mtl.Size.of((n + 7) / 8, rows, 1), mtl.Size.of(256, 1, 1)) else enc.dispatchGroups(d.groups, d.threads);
            }
            enc.end();
            try finish(cb);
            if (!sameBits(outs[0].slice(f32, rows * n), outs[1].slice(f32, rows * n))) {
                std.debug.print("few-row dense n={d} k={d} rows={d} xb={d}: bytes differ\n", .{ n, k, rows, xb });
                return error.TestUnexpectedResult;
            }
            checked += 1;
        };
    }
    std.debug.print("few-row dense test: {d} cases byte-equal to the one-row kernel\n", .{checked});
}

test "few-row fused attention projections give each row the one-row kernel's bytes" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const one = try kernels.AffinePipes.init(device);
    defer one.deinit();
    var shared = try rows_mod.Pipes.init(device);
    defer shared.deinit();
    var prng = std.Random.DefaultPrng.init(0x7072_6f6a);
    const rng = prng.random();
    const k = 2304;
    var checked: usize = 0;
    for ([_]bool{ false, true }) |full| {
        const m: [8]u32 = if (full) .{ 3072, 1024, 1024, 384, 3072, 24, 32, 32 } else .{ 3072, 1024, 1024, 384, 3072, 24, 0, 0 };
        const rows_q: usize = if (full) 3072 + 384 + 3072 + 24 else 8600;
        const wq = try affine(device, rng, rows_q, k);
        defer for (wq) |buf| buf.deinit();
        const wk = try affine(device, rng, 1056, k);
        defer for (wk) |buf| buf.deinit();
        const wv = try affine(device, rng, 1056, k);
        defer for (wv) |buf| buf.deinit();
        var xs: [3]mtl.Buffer = undefined;
        for (&xs) |*x| x.* = try inputs(device, rng, rows_mod.max_rows * k);
        defer for (xs) |x| x.deinit();
        var outs: [2][8]mtl.Buffer = undefined;
        for (&outs) |*side| for (side, m) |*o, rows_m| {
            o.* = try device.buffer(@max(rows_mod.max_rows * rows_m * 4, 16), mtl.ResourceOptions.shared);
        };
        defer for (outs) |side| for (side) |o| o.deinit();
        var total: usize = 0;
        for (m) |v| total += v;
        for (2..rows_mod.max_rows + 1) |rows| {
            const d = shared.fused(rows, m, k) orelse continue;
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            for (0..2) |side| {
                enc.setPipeline(if (side == 0) one.batch_projections else d.pipe);
                for (xs, 0..) |x, i| enc.setBuffer(if (full) x else xs[0], 0, i);
                for ([_][3]mtl.Buffer{ wq, wk, wv }, 0..) |set, j| for (set, 0..) |buf, i| enc.setBuffer(buf, 0, 3 + 3 * j + i);
                for (outs[side], 0..) |o, i| enc.setBuffer(o, 0, 12 + i);
                if (side == 0) {
                    enc.setValue(Projections{ .k = k, .full = @intFromBool(full), .m = m }, 20);
                    enc.dispatchGroups(mtl.Size.of((total + 7) / 8, rows, 1), mtl.Size.of(256, 1, 1));
                } else {
                    enc.setValue(Shared{ .k = k, .full = @intFromBool(full), .rows = @intCast(rows), .m = m }, 20);
                    enc.dispatchGroups(d.groups, d.threads);
                }
            }
            enc.end();
            try finish(cb);
            for (m, 0..) |rows_m, i| {
                const count = rows * rows_m;
                if (!sameBits(outs[0][i].slice(f32, count), outs[1][i].slice(f32, count))) {
                    std.debug.print("few-row projections full={} rows={d} part={d}: bytes differ\n", .{ full, rows, i });
                    return error.TestUnexpectedResult;
                }
            }
            checked += 1;
        }
    }
    std.debug.print("few-row projections test: {d} cases byte-equal to the one-row kernel\n", .{checked});
}
