const std = @import("std");
const mtl = @import("metal");
const t = std.testing;

const Dense = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, bits: u32, cap: f32 };
const Expert = extern struct { k: u32, n: u32, slots: u32, topk: u32, gate_cap: f32, up_cap: f32 };
const Group = extern struct { k: u32, n: u32, slots: u32, topk: u32, experts: u32, gate_cap: f32, up_cap: f32 };
const Projections = extern struct { k: u32, full: u32, m: [8]u32 };
const FloatPair = extern struct { k: u32, m0: u32, m1: u32 };

fn weight(row: usize, k: usize, bits: u32) f64 {
    const mask = (@as(u32, 1) << @intCast(bits)) - 1;
    const code: f64 = @floatFromInt(@as(u32, @intCast((row * 17 + k * 7) % 256)) & mask);
    return code * 0.0625 - 0.5;
}

fn fill(w: []u32, rows: usize, width: usize, bits: u32) void {
    const per: usize = 32 / bits;
    @memset(w, 0);
    for (0..rows) |r| for (0..width) |k| {
        const mask = (@as(u32, 1) << @intCast(bits)) - 1;
        const code: u32 = @as(u32, @intCast((r * 17 + k * 7) % 256)) & mask;
        w[r * (width / per) + k / per] |= code << @intCast((k % per) * bits);
    };
}

fn launch(device: mtl.Device, queue: mtl.Queue, name: []const u8, buffers: []const mtl.Buffer, args: anytype, n: usize, rows: usize) !void {
    const lib = try mtl.Library.fromSource(device, @embedFile("kernels/affine.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, name, false);
    defer pipe.deinit();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipe);
    for (buffers[0 .. buffers.len - 1], 0..) |buf, i| enc.setBuffer(buf, 0, i);
    enc.setValue(args, buffers.len - 1);
    enc.setBuffer(buffers[buffers.len - 1], 0, buffers.len);
    const grouped = std.mem.startsWith(u8, name, "i2a_group_");
    const matrix = std.mem.eql(u8, name, "i2a_dense_rows");
    const group_up = std.mem.eql(u8, name, "i2a_group_up");
    enc.dispatchGroups(mtl.Size.of(if (grouped) (n * (if (group_up) @as(usize, 2) else 1) + 63) / 64 else if (matrix) (n + 63) / 64 else (n + 7) / 8, if (grouped) 2 else if (matrix) (rows + 31) / 32 else rows, if (grouped) (rows + 31) / 32 else 1), mtl.Size.of(256, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
}

test "fused affine projections preserve separate inputs and projection rows" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = 64;
    const capacity = 32;
    const x = try device.buffer(3 * k * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    const w = try device.buffer(capacity * k, mtl.ResourceOptions.shared);
    defer w.deinit();
    const scales = try device.buffer(capacity * 2, mtl.ResourceOptions.shared);
    defer scales.deinit();
    const biases = try device.buffer(3 * capacity * 2, mtl.ResourceOptions.shared);
    defer biases.deinit();
    const out = try device.buffer(8 * capacity * 4, mtl.ResourceOptions.shared);
    defer out.deinit();
    for (x.slice(f32, 3 * k), 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 19)) - 9)) / 16;
    fill(w.slice(u32, capacity * k / 4), capacity, k, 8);
    @memset(scales.slice(u16, capacity), 0x3d80);
    @memset(biases.slice(u16, 3 * capacity)[0..capacity], 0xbf00);
    @memset(biases.slice(u16, 3 * capacity)[capacity .. 2 * capacity], 0xbf80);
    @memset(biases.slice(u16, 3 * capacity)[2 * capacity .. 3 * capacity], 0x3e80);
    const lib = try mtl.Library.fromSource(device, @embedFile("kernels/affine.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "i2a_projections", false);
    defer pipe.deinit();
    for ([_]bool{ false, true }) |full| {
        const ms: [8]u32 = .{ 5, 4, 3, 2, 6, 3, if (full) 2 else 0, if (full) 1 else 0 };
        var total: usize = 0;
        for (ms) |n| total += n;
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipe);
        for (0..3) |i| enc.setBuffer(x, i * k * 4, i);
        for (0..3) |i| {
            enc.setBuffer(w, 0, 3 + i * 3);
            enc.setBuffer(scales, 0, 4 + i * 3);
            enc.setBuffer(biases, (if (full) i else 0) * capacity * 2, 5 + i * 3);
        }
        for (0..8) |i| enc.setBuffer(out, i * capacity * 4, 12 + i);
        enc.setValue(Projections{ .k = k, .full = @intFromBool(full), .m = ms }, 20);
        enc.dispatchGroups(mtl.Size.of((total + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        for (ms, 0..) |n, which| {
            const source: usize = if (which == 1 or which == 6) 1 else if (which == 2 or which == 7) 2 else 0;
            const base: usize = switch (which) {
                0 => 0,
                1 => if (full) 0 else ms[0],
                2 => if (full) 0 else ms[0] + ms[1],
                3 => ms[0] + (if (full) @as(u32, 0) else ms[1] + ms[2]),
                4 => ms[0] + (if (full) @as(u32, 0) else ms[1] + ms[2]) + ms[3],
                5 => ms[0] + (if (full) @as(u32, 0) else ms[1] + ms[2]) + ms[3] + ms[4],
                6 => ms[1],
                7 => ms[2],
                else => unreachable,
            };
            const bias_delta: f64 = if (!full or source == 0) 0 else if (source == 1) -0.5 else 0.75;
            for (0..n) |row| {
                var want: f64 = 0;
                for (0..k) |j| want += @as(f64, x.slice(f32, 3 * k)[source * k + j]) * (weight(base + row, j, 8) + bias_delta);
                try t.expectApproxEqAbs(want, @as(f64, out.slice(f32, 8 * capacity)[which * capacity + row]), 0.001);
            }
        }
    }
}

test "fused float controller preserves projections and in-place output" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = 131;
    const m0 = 7;
    const m1 = 11;
    const x = try device.buffer(k * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    const w0 = try device.buffer(m0 * k * 4, mtl.ResourceOptions.shared);
    defer w0.deinit();
    const w1 = try device.buffer(m1 * k * 4, mtl.ResourceOptions.shared);
    defer w1.deinit();
    const y0 = try device.buffer(m0 * 4, mtl.ResourceOptions.shared);
    defer y0.deinit();
    const y1 = try device.buffer(m1 * 4, mtl.ResourceOptions.shared);
    defer y1.deinit();
    for (x.slice(f32, k), 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 8;
    for (w0.slice(f32, m0 * k), 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 16;
    for (w1.slice(f32, m1 * k), 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 16;
    try launch(device, queue, "i2a_float", &.{ x, w0, y0 }, Dense{ .n = m0, .k = k, .rows = 1, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, m0, 1);
    try launch(device, queue, "i2a_float", &.{ x, w1, y1 }, Dense{ .n = m1, .k = k, .rows = 1, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, m1, 1);
    for ([_]mtl.Buffer{ w0, w1 }, [_]mtl.Buffer{ y0, y1 }, [_]usize{ m0, m1 }) |w, y, rows| {
        for (0..rows) |row| {
            var want: f64 = 0;
            for (0..k) |j| want += @as(f64, x.slice(f32, k)[j]) * @as(f64, w.slice(f32, rows * k)[row * k + j]);
            try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, rows)[row]), 0.001);
        }
    }
    const cur = try device.buffer(m0 * 4, mtl.ResourceOptions.shared);
    defer cur.deinit();
    const msg = try device.buffer(m0 * 4, mtl.ResourceOptions.shared);
    defer msg.deinit();
    var want: [m0]f32 = undefined;
    for (0..m0) |i| {
        cur.slice(f32, m0)[i] = @as(f32, @floatFromInt(i)) / 4;
        msg.slice(f32, m0)[i] = @as(f32, @floatFromInt(i + 1)) / 8;
        want[i] = cur.slice(f32, m0)[i] - std.math.tanh(y1.slice(f32, m1)[i]) * cur.slice(f32, m0)[i] + std.math.tanh(y0.slice(f32, m0)[i]) * msg.slice(f32, m0)[i];
    }
    const lib = try mtl.Library.fromSource(device, @embedFile("kernels/affine.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const control = try mtl.Pipeline.init(device, lib, "i2a_float_control", false);
    defer control.deinit();
    const cb2 = queue.commandBuffer();
    const enc2 = cb2.compute(.serial);
    enc2.setPipeline(control);
    for ([_]mtl.Buffer{ x, w0, w1, cur, msg, cur }, 0..) |buffer, i| enc2.setBuffer(buffer, 0, i);
    enc2.setValue(FloatPair{ .k = k, .m0 = m0, .m1 = m0 }, 6);
    enc2.dispatchGroups(mtl.Size.of((m0 + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
    enc2.end();
    cb2.commit();
    cb2.wait();
    if (cb2.failure()) |_| return error.GpuFailed;
    for (want, 0..) |value, i| try t.expectApproxEqAbs(value, cur.slice(f32, m0)[i], 0.00001);
}

test "affine dense and expert kernels agree with dequantized float64 weights" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = 64;
    const n = 68;
    const x = try device.buffer(256 * k * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    const words = try device.buffer(2 * 2 * n * k, mtl.ResourceOptions.shared);
    defer words.deinit();
    const scales = try device.buffer(2 * 2 * n * 2, mtl.ResourceOptions.shared);
    defer scales.deinit();
    const biases = try device.buffer(2 * 2 * n * 2, mtl.ResourceOptions.shared);
    defer biases.deinit();
    const picks = try device.buffer(256 * 4, mtl.ResourceOptions.shared);
    defer picks.deinit();
    const offsets = try device.buffer(3 * 4, mtl.ResourceOptions.shared);
    defer offsets.deinit();
    const sorted = try device.buffer(256 * 4, mtl.ResourceOptions.shared);
    defer sorted.deinit();
    const y = try device.buffer(256 * n * 4, mtl.ResourceOptions.shared);
    defer y.deinit();
    for (x.slice(f32, 256 * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 8;
    @memset(scales.slice(u16, 2 * 2 * n), 0x3d80);
    @memset(biases.slice(u16, 2 * 2 * n), 0xbf00);
    for (picks.slice(i32, 256), 0..) |*v, i| v.* = @intCast(i % 2);
    for ([_]usize{ 1, 4, 16, 256 }) |rows| {
    const first_count = (rows + 1) / 2;
    offsets.slice(u32, 3)[0] = 0;
    offsets.slice(u32, 3)[1] = @intCast(first_count);
    offsets.slice(u32, 3)[2] = @intCast(rows);
    for (0..rows) |row| sorted.slice(u32, 256)[if (row % 2 == 0) row / 2 else first_count + row / 2] = @intCast(row);
    fill(words.slice(u32, 2 * 2 * n * k / 4), 2 * 2 * n, k, 8);
    try launch(device, queue, "i2a_dense", &.{ x, words, scales, biases, y }, Dense{ .n = n, .k = k, .rows = @intCast(rows), .xs = 1, .xb = 0, .bits = 8, .cap = 0 }, n, rows);
    for (0..rows) |row| for (0..n) |col| {
        var want: f64 = 0;
        for (0..k) |j| want += @as(f64, x.slice(f32, 256 * k)[row * k + j]) * weight(col, j, 8);
        try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, 256 * n)[row * n + col]), 0.001);
    };
    try launch(device, queue, "i2a_dense_rows", &.{ x, words, scales, biases, y }, Dense{ .n = n, .k = k, .rows = @intCast(rows), .xs = 1, .xb = 0, .bits = 8, .cap = 0 }, n, rows);
    for (0..rows) |row| for (0..n) |col| {
        var want: f64 = 0;
        for (0..k) |j| want += @as(f64, x.slice(f32, 256 * k)[row * k + j]) * weight(col, j, 8);
        try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, 256 * n)[row * n + col]), 0.001);
    };
    fill(words.slice(u32, 2 * 2 * n * k / 4), 2 * 2 * n, k, 4);
    try launch(device, queue, "i2a_expert_up", &.{ x, words, scales, biases, picks, y }, Expert{ .k = k, .n = n, .slots = @intCast(rows), .topk = 1, .gate_cap = 4, .up_cap = 25 }, n, rows);
    for (0..rows) |row| for (0..n) |col| {
        const expert: usize = @intCast(picks.slice(i32, 256)[row]);
        var gate: f64 = 0;
        var up: f64 = 0;
        for (0..k) |j| {
            const v: f64 = x.slice(f32, 256 * k)[row * k + j];
            gate += v * weight(expert * 2 * n + 2 * col, j, 4);
            up += v * weight(expert * 2 * n + 2 * col + 1, j, 4);
        }
        const want = 4 * std.math.tanh(gate / 4) / (1 + @exp(-gate)) * 25 * std.math.tanh(up / 25);
        try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, 256 * n)[row * n + col]), 0.001);
    };
    try launch(device, queue, "i2a_group_up", &.{ x, words, scales, biases, offsets, sorted, y }, Group{ .k = k, .n = n, .slots = @intCast(rows), .topk = 1, .experts = 2, .gate_cap = 4, .up_cap = 25 }, n, rows);
    for (0..rows) |row| for (0..n) |col| {
        const expert: usize = @intCast(picks.slice(i32, 256)[row]);
        var gate: f64 = 0;
        var up: f64 = 0;
        for (0..k) |j| {
            const v: f64 = x.slice(f32, 256 * k)[row * k + j];
            gate += v * weight(expert * 2 * n + 2 * col, j, 4);
            up += v * weight(expert * 2 * n + 2 * col + 1, j, 4);
        }
        const want = 4 * std.math.tanh(gate / 4) / (1 + @exp(-gate)) * 25 * std.math.tanh(up / 25);
        try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, 256 * n)[row * n + col]), 0.001);
    };
    try launch(device, queue, "i2a_expert_down", &.{ x, words, scales, biases, picks, y }, Expert{ .k = k, .n = n, .slots = @intCast(rows), .topk = 1, .gate_cap = 0, .up_cap = 0 }, n, rows);
    for (0..rows) |row| for (0..n) |col| {
        const expert: usize = @intCast(picks.slice(i32, 256)[row]);
        var want: f64 = 0;
        for (0..k) |j| want += @as(f64, x.slice(f32, 256 * k)[row * k + j]) * weight(expert * n + col, j, 4);
        try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, 256 * n)[row * n + col]), 0.001);
    };
    try launch(device, queue, "i2a_group_down", &.{ x, words, scales, biases, offsets, sorted, y }, Group{ .k = k, .n = n, .slots = @intCast(rows), .topk = 1, .experts = 2, .gate_cap = 0, .up_cap = 0 }, n, rows);
    for (0..rows) |row| for (0..n) |col| {
        const expert: usize = @intCast(picks.slice(i32, 256)[row]);
        var want: f64 = 0;
        for (0..k) |j| want += @as(f64, x.slice(f32, 256 * k)[row * k + j]) * weight(expert * n + col, j, 4);
        try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, 256 * n)[row * n + col]), 0.001);
    };
    }
}

test "affine embedding and router match float64 references" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = 256;
    const n = 4;
    const x = try device.buffer(256 * k * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    const fw = try device.buffer(n * k * 4, mtl.ResourceOptions.shared);
    defer fw.deinit();
    const words = try device.buffer(n * k, mtl.ResourceOptions.shared);
    defer words.deinit();
    const scales = try device.buffer(n * k / 64 * 2, mtl.ResourceOptions.shared);
    defer scales.deinit();
    const biases = try device.buffer(n * k / 64 * 2, mtl.ResourceOptions.shared);
    defer biases.deinit();
    const ids = try device.buffer(256 * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    const norm = try device.buffer(k * 2, mtl.ResourceOptions.shared);
    defer norm.deinit();
    const y = try device.buffer(256 * k * 4, mtl.ResourceOptions.shared);
    defer y.deinit();
    for (x.slice(f32, 256 * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 8;
    for (fw.slice(f32, n * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 16;
    for (ids.slice(u32, 256), 0..) |*v, i| v.* = @intCast(i % n);
    @memset(norm.slice(u16, k), 0x3f80);
    @memset(scales.slice(u16, n * k / 64), 0x3d80);
    @memset(biases.slice(u16, n * k / 64), 0xbf00);
    fill(words.slice(u32, n * k / 4), n, k, 8);
    const lib = try mtl.Library.fromSource(device, @embedFile("kernels/affine.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const embed = try mtl.Pipeline.init(device, lib, "i2a_embed", false);
    defer embed.deinit();
    for ([_]usize{ 1, 4, 16, 256 }) |rows| {
        try launch(device, queue, "i2a_float", &.{ x, fw, y }, Dense{ .n = n, .k = k, .rows = @intCast(rows), .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, n, rows);
        for (0..rows) |row| for (0..n) |col| {
            var want: f64 = 0;
            for (0..k) |j| want += @as(f64, x.slice(f32, 256 * k)[row * k + j]) * @as(f64, fw.slice(f32, n * k)[col * k + j]);
            try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, 256 * k)[row * n + col]), 0.001);
        };
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(embed);
        for ([_]mtl.Buffer{ ids, words, scales, biases, norm }, 0..) |buf, i| enc.setBuffer(buf, 0, i);
        enc.setValue(extern struct { width: u32, rows: u32, eps: f32 }{ .width = k, .rows = @intCast(rows), .eps = 0.00001 }, 5);
        enc.setBuffer(y, 0, 6);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        for (0..rows) |row| {
            const id = ids.slice(u32, 256)[row];
            var sum: f64 = 0;
            for (0..k) |j| sum += weight(id, j, 8) * weight(id, j, 8);
            const inv = 1 / @sqrt(sum / k + 0.00001);
            for (0..k) |j| try t.expectApproxEqAbs(weight(id, j, 8) * inv, @as(f64, y.slice(f32, 256 * k)[row * k + j]), 0.001);
        }
    }
}

test "misaligned oQ4e shard data produces finite decode logits" {
    const path = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return error.SkipZigTest;
    const model = @import("model.zig");
    const state = @import("state.zig");
    const forward = @import("forward.zig");
    const m = try model.Model.load(t.allocator, t.io, std.mem.span(path));
    defer m.deinit();
    var cache = try state.Cache.init(m.device, &m.config, 1);
    defer cache.deinit();
    var scratch = try state.Scratch.init(m.device, &m.config);
    defer scratch.deinit();
    try forward.position(m, &scratch, &cache, 200002);
    const logits = scratch.logits.slice(f32, m.config.text.unpadded_vocab);
    var best: usize = 0;
    for (logits, 0..) |v, i| {
        try t.expect(std.math.isFinite(v));
        if (v > logits[best]) best = i;
    }
    try t.expectEqual(@as(usize, 12), best);
}
