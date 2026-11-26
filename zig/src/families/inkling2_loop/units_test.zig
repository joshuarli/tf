const std = @import("std");
const mtl = @import("metal");
const prefill = @import("prefill.zig");

const t = std.testing;

const Moeg = extern struct { k: u32, n: u32, slots: u32, topk: u32, experts: u32, gate_cap: f32, up_cap: f32 };

fn e4m3(code: u8) f64 {
    const exp: i32 = @intCast((code >> 3) & 15);
    const man: f64 = @floatFromInt(code & 7);
    const value = if (exp == 0) man / 512.0 else (1.0 + man / 8.0) * std.math.pow(f64, 2.0, @floatFromInt(exp - 7));
    return if (code & 128 != 0) -value else value;
}

fn e2m1(code: u8) f64 {
    const values = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return if (code & 8 != 0) -values[code & 7] else values[code & 7];
}

fn half(x: f64) f64 {
    return @floatCast(@as(f16, @floatCast(x)));
}

fn situ(g: f64, u: f64, gate_cap: f64, up_cap: f64) f64 {
    return gate_cap * std.math.tanh(g / gate_cap) / (1.0 + @exp(-g)) * up_cap * std.math.tanh(u / up_cap);
}

/// Routed experts with `counts` sorted rows each, `cols` weight rows of depth `k` an expert, NVFP4 coded.
const Fixture = struct {
    counts: []const usize,
    slots: usize,
    k: usize,
    cols: usize,
    off: mtl.Buffer,
    row_of: mtl.Buffer,
    w: mtl.Buffer,
    s: mtl.Buffer,
    amax: mtl.Buffer,
    den: mtl.Buffer,

    fn init(device: mtl.Device, counts: []const usize, k: usize, cols: usize, salt: usize) !Fixture {
        var slots: usize = 0;
        for (counts) |n| slots += n;
        const e = counts.len;
        const f = Fixture{
            .counts = counts,
            .slots = slots,
            .k = k,
            .cols = cols,
            .off = try device.buffer((e + 1) * 4, mtl.ResourceOptions.shared),
            .row_of = try device.buffer(slots * 4, mtl.ResourceOptions.shared),
            .w = try device.buffer(e * cols * k / 2, mtl.ResourceOptions.shared),
            .s = try device.buffer(e * cols * k / 16, mtl.ResourceOptions.shared),
            .amax = try device.buffer(e * 4, mtl.ResourceOptions.shared),
            .den = try device.buffer(16, mtl.ResourceOptions.shared),
        };
        const off = f.off.slice(i32, e + 1);
        off[0] = 0;
        for (counts, 0..) |n, i| off[i + 1] = off[i] + @as(i32, @intCast(n));
        for (f.row_of.slice(i32, slots), 0..) |*v, i| v.* = @intCast((i * 7) % slots);
        for (f.w.slice(u8, e * cols * k / 2), 0..) |*v, i| v.* = @intCast((i *% 37 + i / (k / 2) *% 13 + salt) % 256);
        for (f.s.slice(u8, e * cols * k / 16), 0..) |*v, i| v.* = @intCast(40 + (i * 5 + salt) % 16);
        for (f.amax.slice(f32, e), 0..) |*v, i| v.* = 0.5 + 0.1 * @as(f32, @floatFromInt(i));
        f.den.slice(f32, 1)[0] = 1.3;
        return f;
    }

    fn deinit(f: Fixture) void {
        for ([_]mtl.Buffer{ f.off, f.row_of, f.w, f.s, f.amax, f.den }) |buf| buf.deinit();
    }

    fn expertOf(f: Fixture, sorted: usize) usize {
        var at: usize = 0;
        for (f.counts, 0..) |n, e| {
            if (sorted < at + n) return e;
            at += n;
        }
        unreachable;
    }

    /// Weight row `row` of expert `e` at depth j: the e2m1 code times its group's e4m3 scale, exact in half.
    fn weight(f: Fixture, e: usize, row: usize, j: usize) f64 {
        const at = e * f.cols + row;
        const byte = f.w.slice(u8, f.counts.len * f.cols * f.k / 2)[at * (f.k / 2) + j / 2];
        const code: u8 = (byte >> @as(u3, @intCast((j % 2) * 4))) & 15;
        return e2m1(code) * e4m3(f.s.slice(u8, f.counts.len * f.cols * f.k / 16)[at * (f.k / 16) + j / 16]);
    }

    fn global(f: Fixture, e: usize) f64 {
        return @as(f64, f.amax.slice(f32, f.counts.len)[e]) / @as(f64, f.den.slice(f32, 1)[0]);
    }

    fn tiles(f: Fixture) usize {
        return (f.slots + 63) / 64 + f.counts.len;
    }
};

fn bind(enc: mtl.ComputeEncoder, bufs: []const mtl.Buffer) void {
    for (bufs, 0..) |buf, i| enc.setBuffer(buf, 0, i);
}

fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("expert units: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

test "tensor-unit routed experts meet fp64 bounds across tiles, thresholds and column tails" {
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, prefill.units_source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const to_half = try mtl.Pipeline.init(device, lib, "i2p_half_rows", false);
    defer to_half.deinit();
    const grouped = try mtl.Pipeline.init(device, lib, "i2p_gu_grouped", false);
    defer grouped.deinit();
    const up = try mtl.Pipeline.init(device, lib, "i2p_expert_units_up", false);
    defer up.deinit();
    const down = try mtl.Pipeline.init(device, lib, "i2p_expert_units_down", false);
    defer down.deinit();
    const counts = [_]usize{ 7, 13, 64, 0, 65, 130, 14 };
    const topk = 3;
    const codes = try device.buffer(16 * 4, mtl.ResourceOptions.shared);
    defer codes.deinit();
    for (codes.slice(f32, 16), 0..) |*v, i| v.* = @floatCast(e2m1(@intCast(i)));
    const overflow = try device.buffer(16, mtl.ResourceOptions.shared);
    defer overflow.deinit();
    const gamma = struct {
        fn of(k: usize) f64 {
            const u = @as(f64, @floatFromInt(k + 2)) * 0x1p-24;
            return u / (1.0 - u);
        }
    }.of;

    const n_up = 80;
    const k_up = 192;
    const fu = try Fixture.init(device, &counts, k_up, 2 * n_up, 3);
    defer fu.deinit();
    const tokens = (fu.slots + topk - 1) / topk;
    const x = try device.buffer(tokens * k_up * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    for (x.slice(f32, tokens * k_up), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i *% 17 + i / k_up *% 31) % 251)) - 125)) / 97.0;
    const xh = try device.buffer(tokens * k_up * 2, mtl.ResourceOptions.shared);
    defer xh.deinit();
    const act = try device.buffer(fu.slots * n_up * 4, mtl.ResourceOptions.shared);
    defer act.deinit();
    const act_h = try device.buffer(fu.slots * n_up * 2, mtl.ResourceOptions.shared);
    defer act_h.deinit();
    @memset(act.slice(f32, fu.slots * n_up), std.math.nan(f32));
    @memset(act_h.slice(f16, fu.slots * n_up), std.math.nan(f16));
    overflow.slice(u32, 1)[0] = 0;
    const args_up = Moeg{ .k = k_up, .n = n_up, .slots = @intCast(fu.slots), .topk = topk, .experts = counts.len, .gate_cap = 7, .up_cap = 7 };
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        const count: u32 = @intCast(tokens * k_up / 4);
        enc.setPipeline(to_half);
        bind(enc, &.{ x, xh, overflow });
        enc.setValue([2]u32{ count, 1 }, 3);
        enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
        enc.setPipeline(grouped);
        bind(enc, &.{ x, fu.w, fu.s, fu.amax, fu.den, fu.off, fu.row_of });
        enc.setValue(args_up, 7);
        enc.setBuffer(act, 0, 8);
        enc.setBuffer(codes, 0, 9);
        enc.setBuffer(act_h, 0, 10);
        enc.setBuffer(overflow, 0, 11);
        enc.dispatchGroups(mtl.Size.of(n_up, counts.len, 1), mtl.Size.of(32, 1, 1));
        enc.setPipeline(up);
        bind(enc, &.{ xh, fu.w, fu.s, fu.amax, fu.den, fu.off, fu.row_of });
        enc.setValue(args_up, 7);
        enc.setBuffer(act_h, 0, 8);
        enc.setBuffer(overflow, 0, 9);
        enc.dispatchGroups(mtl.Size.of((fu.cols + 63) / 64, fu.tiles(), 1), mtl.Size.of(256, 1, 1));
        enc.end();
        try finish(cb);
    }
    try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
    for (x.slice(f32, tokens * k_up), xh.slice(f16, tokens * k_up)) |v, h| try t.expectEqual(@as(f16, @floatCast(v)), h);
    var worst_up: f64 = 0;
    for (0..fu.slots) |sorted| {
        const e = fu.expertOf(sorted);
        const slot: usize = @intCast(fu.row_of.slice(i32, fu.slots)[sorted]);
        const row = slot / topk;
        for (0..n_up) |c| {
            const got_h: f64 = @floatCast(act_h.slice(f16, fu.slots * n_up)[slot * n_up + c]);
            const got: f64 = act.slice(f32, fu.slots * n_up)[slot * n_up + c];
            if (counts[e] < 8) {
                try t.expect(std.math.isNan(got_h));
                continue;
            }
            if (counts[e] < 14) {
                try t.expect(std.math.isFinite(got));
                try t.expectEqual(@as(f16, @floatCast(got)), act_h.slice(f16, fu.slots * n_up)[slot * n_up + c]);
                continue;
            }
            try t.expect(std.math.isNan(got));
            var sums = [_]f64{ 0, 0 };
            var mags = [_]f64{ 0, 0 };
            for (0..2) |part| for (0..k_up) |j| {
                const term = half(x.slice(f32, tokens * k_up)[row * k_up + j]) * fu.weight(e, 2 * c + part, j);
                sums[part] += term;
                mags[part] += @abs(term);
            };
            const g = fu.global(e);
            const want = situ(sums[0] * g, sums[1] * g, 7, 7);
            const bound = 0x1p-11 * @abs(want) + 16.0 * gamma(k_up) * (mags[0] + mags[1]) * g + 0x1p-24;
            const err = @abs(got_h - want);
            worst_up = @max(worst_up, err);
            if (!(err <= bound)) {
                std.debug.print("units up expert={d} rows={d} slot={d} col={d} got={e} want={e} bound={e}\n", .{ e, counts[e], slot, c, got_h, want, bound });
                return error.ExpertReferenceMismatch;
            }
        }
    }

    const n_down = 72;
    const k_down = 128;
    const fd = try Fixture.init(device, &counts, k_down, n_down, 11);
    defer fd.deinit();
    const xd = try device.buffer(fd.slots * k_down * 2, mtl.ResourceOptions.shared);
    defer xd.deinit();
    for (xd.slice(f16, fd.slots * k_down), 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i *% 13 + i / k_down *% 29) % 211)) - 105)) / 23.0);
    const out = try device.buffer(fd.slots * n_down * 4, mtl.ResourceOptions.shared);
    defer out.deinit();
    @memset(out.slice(f32, fd.slots * n_down), std.math.nan(f32));
    const args_down = Moeg{ .k = k_down, .n = n_down, .slots = @intCast(fd.slots), .topk = topk, .experts = counts.len, .gate_cap = 0, .up_cap = 0 };
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(down);
        bind(enc, &.{ xd, fd.w, fd.s, fd.amax, fd.den, fd.off, fd.row_of });
        enc.setValue(args_down, 7);
        enc.setBuffer(out, 0, 8);
        enc.dispatchGroups(mtl.Size.of((n_down + 63) / 64, fd.tiles(), 1), mtl.Size.of(256, 1, 1));
        enc.end();
        try finish(cb);
    }
    var worst_down: f64 = 0;
    for (0..fd.slots) |sorted| {
        const e = fd.expertOf(sorted);
        const slot: usize = @intCast(fd.row_of.slice(i32, fd.slots)[sorted]);
        for (0..n_down) |c| {
            const got: f64 = out.slice(f32, fd.slots * n_down)[slot * n_down + c];
            if (counts[e] < 8) {
                try t.expect(std.math.isNan(got));
                continue;
            }
            var sum: f64 = 0;
            var mag: f64 = 0;
            for (0..k_down) |j| {
                const term = @as(f64, @floatCast(xd.slice(f16, fd.slots * k_down)[slot * k_down + j])) * fd.weight(e, c, j);
                sum += term;
                mag += @abs(term);
            }
            const g = fd.global(e);
            const want = sum * g;
            const bound = gamma(k_down) * mag * g + 0x1p-24 * @abs(want) + 1e-6;
            const err = @abs(got - want);
            worst_down = @max(worst_down, err);
            if (!(err <= bound)) {
                std.debug.print("units down expert={d} rows={d} slot={d} col={d} got={e} want={e} bound={e}\n", .{ e, counts[e], slot, c, got, want, bound });
                return error.ExpertReferenceMismatch;
            }
        }
    }
    std.debug.print("expert units worst up={e} down={e}\n", .{ worst_up, worst_down });

    x.slice(f32, tokens * k_up)[5 * k_up + 3] = 70000;
    x.slice(f32, tokens * k_up)[6 * k_up + 4] = -70000;
    overflow.slice(u32, 1)[0] = 0;
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        const count: u32 = @intCast(tokens * k_up / 4);
        enc.setPipeline(to_half);
        bind(enc, &.{ x, xh, overflow });
        enc.setValue([2]u32{ count, 1 }, 3);
        enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
        enc.end();
        try finish(cb);
    }
    try t.expectEqual(@as(u32, 2), overflow.slice(u32, 1)[0]);
    try t.expectEqual(@as(f16, 65504), xh.slice(f16, tokens * k_up)[5 * k_up + 3]);
    try t.expectEqual(@as(f16, -65504), xh.slice(f16, tokens * k_up)[6 * k_up + 4]);
}

test "tensor-unit affine experts meet fp64 bounds across tiles and column tails" {
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, prefill.units_source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const to_half = try mtl.Pipeline.init(device, lib, "i2p_half_rows", false);
    defer to_half.deinit();
    const up = try mtl.Pipeline.init(device, lib, "i2p_affine_units_up", false);
    defer up.deinit();
    const down = try mtl.Pipeline.init(device, lib, "i2p_affine_units_down", false);
    defer down.deinit();
    const counts = [_]usize{ 1, 7, 0, 64, 65 };
    var off_values: [counts.len + 1]i32 = undefined;
    off_values[0] = 0;
    for (counts, 0..) |n, i| off_values[i + 1] = off_values[i] + @as(i32, @intCast(n));
    const slots: usize = @intCast(off_values[counts.len]);
    const topk = 2;
    const tokens = (slots + topk - 1) / topk;
    const off = try device.buffer((counts.len + 1) * 4, mtl.ResourceOptions.shared);
    defer off.deinit();
    @memcpy(off.slice(i32, counts.len + 1), &off_values);
    const row_of = try device.buffer(slots * 4, mtl.ResourceOptions.shared);
    defer row_of.deinit();
    for (row_of.slice(i32, slots), 0..) |*v, i| v.* = @intCast((i * 5) % slots);
    const overflow = try device.buffer(16, mtl.ResourceOptions.shared);
    defer overflow.deinit();
    overflow.slice(u32, 1)[0] = 0;
    const Weights = struct {
        w: mtl.Buffer,
        s: mtl.Buffer,
        b: mtl.Buffer,
        rows: usize,
        k: usize,
        fn init(dev: mtl.Device, rows: usize, k: usize, salt: usize) !@This() {
            const r = @This(){ .w = try dev.buffer(rows * k / 2, mtl.ResourceOptions.shared), .s = try dev.buffer(rows * k / 32, mtl.ResourceOptions.shared), .b = try dev.buffer(rows * k / 32, mtl.ResourceOptions.shared), .rows = rows, .k = k };
            for (r.w.slice(u8, rows * k / 2), 0..) |*v, i| v.* = @intCast((i *% 29 + i / (k / 2) *% 11 + salt) % 256);
            for (r.s.slice(u16, rows * k / 64), r.b.slice(u16, rows * k / 64), 0..) |*sv, *bv, i| {
                const scale: f32 = 0.01 + 0.002 * @as(f32, @floatFromInt((i + salt) % 7));
                const bias: f32 = -0.08 + 0.01 * @as(f32, @floatFromInt((i * 3 + salt) % 9));
                sv.* = @truncate(@as(u32, @bitCast(scale)) >> 16);
                bv.* = @truncate(@as(u32, @bitCast(bias)) >> 16);
            }
            return r;
        }
        fn deinit(r: @This()) void {
            r.w.deinit();
            r.s.deinit();
            r.b.deinit();
        }
        fn weight(r: @This(), row: usize, j: usize) f64 {
            const byte = r.w.slice(u8, r.rows * r.k / 2)[row * (r.k / 2) + j / 2];
            const code: f32 = @floatFromInt((byte >> @as(u3, @intCast((j % 2) * 4))) & 15);
            const scale: f32 = @bitCast(@as(u32, r.s.slice(u16, r.rows * r.k / 64)[row * (r.k / 64) + j / 64]) << 16);
            const bias: f32 = @bitCast(@as(u32, r.b.slice(u16, r.rows * r.k / 64)[row * (r.k / 64) + j / 64]) << 16);
            return @floatCast(@as(f16, @floatCast(@mulAdd(f32, code, scale, bias))));
        }
    };
    const expertOf = struct {
        fn of(offs: []const i32, sorted: usize) usize {
            var e: usize = 0;
            while (sorted >= @as(usize, @intCast(offs[e + 1]))) e += 1;
            return e;
        }
    }.of;
    const gamma = struct {
        fn of(k: usize) f64 {
            const u = @as(f64, @floatFromInt(k + 2)) * 0x1p-24;
            return u / (1.0 - u);
        }
    }.of;
    const n_up = 80;
    const k_up = 128;
    const wu = try Weights.init(device, counts.len * 2 * n_up, k_up, 5);
    defer wu.deinit();
    const x = try device.buffer(tokens * k_up * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    for (x.slice(f32, tokens * k_up), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i *% 19 + i / k_up *% 7) % 233)) - 116)) / 61.0;
    const xh = try device.buffer(tokens * k_up * 2, mtl.ResourceOptions.shared);
    defer xh.deinit();
    const act_h = try device.buffer(slots * n_up * 2, mtl.ResourceOptions.shared);
    defer act_h.deinit();
    @memset(act_h.slice(f16, slots * n_up), std.math.nan(f16));
    const args_up = Moeg{ .k = k_up, .n = n_up, .slots = @intCast(slots), .topk = topk, .experts = counts.len, .gate_cap = 7, .up_cap = 7 };
    const tiles = (slots + 127) / 128 + counts.len;
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        const count: u32 = @intCast(tokens * k_up / 4);
        enc.setPipeline(to_half);
        bind(enc, &.{ x, xh, overflow });
        enc.setValue([2]u32{ count, 0 }, 3);
        enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
        enc.setPipeline(up);
        bind(enc, &.{ xh, wu.w, wu.s, wu.b, overflow, off, row_of });
        enc.setValue(args_up, 7);
        enc.setBuffer(act_h, 0, 8);
        enc.setBuffer(overflow, 0, 9);
        enc.dispatchGroups(mtl.Size.of((2 * n_up + 63) / 64, tiles, 1), mtl.Size.of(256, 1, 1));
        enc.end();
        try finish(cb);
    }
    try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
    var worst_up: f64 = 0;
    for (0..slots) |sorted| {
        const e = expertOf(&off_values, sorted);
        const slot: usize = @intCast(row_of.slice(i32, slots)[sorted]);
        for (0..n_up) |c| {
            var sums = [_]f64{ 0, 0 };
            var mags = [_]f64{ 0, 0 };
            for (0..2) |part| for (0..k_up) |j| {
                const term = half(x.slice(f32, tokens * k_up)[(slot / topk) * k_up + j]) * wu.weight(e * 2 * n_up + 2 * c + part, j);
                sums[part] += term;
                mags[part] += @abs(term);
            };
            const want = situ(sums[0], sums[1], 7, 7);
            const got: f64 = @floatCast(act_h.slice(f16, slots * n_up)[slot * n_up + c]);
            const bound = 0x1p-11 * @abs(want) + 16.0 * gamma(k_up) * (mags[0] + mags[1]) + 0x1p-24;
            worst_up = @max(worst_up, @abs(got - want));
            if (!(@abs(got - want) <= bound)) {
                std.debug.print("affine units up expert={d} slot={d} col={d} got={e} want={e} bound={e}\n", .{ e, slot, c, got, want, bound });
                return error.ExpertReferenceMismatch;
            }
        }
    }
    const n_down = 72;
    const k_down = 128;
    const wd = try Weights.init(device, counts.len * n_down, k_down, 9);
    defer wd.deinit();
    const xd = try device.buffer(slots * k_down * 2, mtl.ResourceOptions.shared);
    defer xd.deinit();
    for (xd.slice(f16, slots * k_down), 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i *% 23 + i / k_down *% 3) % 199)) - 99)) / 17.0);
    const out = try device.buffer(slots * n_down * 4, mtl.ResourceOptions.shared);
    defer out.deinit();
    @memset(out.slice(f32, slots * n_down), std.math.nan(f32));
    const args_down = Moeg{ .k = k_down, .n = n_down, .slots = @intCast(slots), .topk = topk, .experts = counts.len, .gate_cap = 0, .up_cap = 0 };
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(down);
        bind(enc, &.{ xd, wd.w, wd.s, wd.b, overflow, off, row_of });
        enc.setValue(args_down, 7);
        enc.setBuffer(out, 0, 8);
        enc.setBuffer(overflow, 0, 9);
        enc.dispatchGroups(mtl.Size.of((n_down + 63) / 64, tiles, 1), mtl.Size.of(256, 1, 1));
        enc.end();
        try finish(cb);
    }
    try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
    var worst_down: f64 = 0;
    for (0..slots) |sorted| {
        const e = expertOf(&off_values, sorted);
        const slot: usize = @intCast(row_of.slice(i32, slots)[sorted]);
        for (0..n_down) |c| {
            var sum: f64 = 0;
            var mag: f64 = 0;
            for (0..k_down) |j| {
                const term = @as(f64, @floatCast(xd.slice(f16, slots * k_down)[slot * k_down + j])) * wd.weight(e * n_down + c, j);
                sum += term;
                mag += @abs(term);
            }
            const got: f64 = out.slice(f32, slots * n_down)[slot * n_down + c];
            const bound = gamma(k_down) * mag + 0x1p-24 * @abs(sum) + 1e-6;
            worst_down = @max(worst_down, @abs(got - sum));
            if (!(@abs(got - sum) <= bound)) {
                std.debug.print("affine units down expert={d} slot={d} col={d} got={e} want={e} bound={e}\n", .{ e, slot, c, got, sum, bound });
                return error.ExpertReferenceMismatch;
            }
        }
    }
    std.debug.print("affine units worst up={e} down={e}\n", .{ worst_up, worst_down });
}

test "oQ4e tensor-unit prefill matches the simdgroup-matrix prefill" {
    const path = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return error.SkipZigTest;
    const model = @import("model.zig");
    const state = @import("state.zig");
    const m = try model.Model.load(t.allocator, t.io, std.mem.span(path));
    defer m.deinit();
    if (!m.device.tensorUnits()) return error.SkipZigTest;
    const vocab = m.config.text.unpadded_vocab;
    var ids: [600]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1000 + (i * 7919) % 50000);
    var ps = try prefill.PrefillScratch.init(m.device, &m.config, 512);
    defer ps.deinit();
    var logits: [2][]f32 = undefined;
    for ([_]bool{ true, false }, 0..) |units, side| {
        var cache = try state.Cache.init(m.device, &m.config, ids.len);
        defer cache.deinit();
        ps.units = units;
        try prefill.prefillWant(m, &ps, &cache, ids[0..512], .none);
        try prefill.prefillWant(m, &ps, &cache, ids[512..], .last);
        logits[side] = try t.allocator.dupe(f32, ps.logits.slice(f32, vocab));
    }
    defer for (logits) |l| t.allocator.free(l);
    var worst: f32 = 0;
    var scale: f32 = 1;
    var best = [_]usize{ 0, 0 };
    for (logits[0], logits[1], 0..) |a, b, i| {
        try t.expect(std.math.isFinite(a) and std.math.isFinite(b));
        worst = @max(worst, @abs(a - b));
        scale = @max(scale, @abs(b));
        if (a > logits[0][best[0]]) best[0] = i;
        if (b > logits[1][best[1]]) best[1] = i;
    }
    std.debug.print("oq4e units vs simdgroup prefill: logits_max_diff={e} scale={e} argmax={d},{d}\n", .{ worst, scale, best[0], best[1] });
    try t.expectEqual(best[1], best[0]);
    try t.expect(worst <= 0.05 * scale);
}
