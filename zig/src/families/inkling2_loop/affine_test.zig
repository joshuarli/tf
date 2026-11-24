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

test "half affine rows match rounded operands across tiles and strides" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, "#define I2A_HALF_OPERANDS 1\n" ++ @embedFile("kernels/affine.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "i2a_dense_half_rows", false);
    defer pipe.deinit();
    const rows = 65;
    const n = 68;
    const k = 128;
    const x = try device.buffer(rows * 2 * k * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    const words = try device.buffer(n * k, mtl.ResourceOptions.shared);
    defer words.deinit();
    const scales = try device.buffer(n * (k / 64) * 2, mtl.ResourceOptions.shared);
    defer scales.deinit();
    const biases = try device.buffer(n * (k / 64) * 2, mtl.ResourceOptions.shared);
    defer biases.deinit();
    const y = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared);
    defer y.deinit();
    const overflow = try device.buffer(16, mtl.ResourceOptions.shared);
    defer overflow.deinit();
    for (x.slice(f32, rows * 2 * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 97)) - 48)) / 37.0;
    @memset(scales.slice(u16, n * (k / 64)), 0x3b83);
    @memset(biases.slice(u16, n * (k / 64)), 0xbd0d);
    const scale: f32 = @bitCast(@as(u32, 0x3b830000));
    const bias: f32 = @bitCast(@as(u32, 0xbd0d0000));
    for ([_]u32{ 4, 8 }) |bits| {
        fill(words.slice(u32, n * k / 4), n, k, bits);
        overflow.slice(u32, 1)[0] = 0;
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipe);
        for ([_]mtl.Buffer{ x, words, scales, biases }, 0..) |buf, i| enc.setBuffer(buf, 0, i);
        enc.setValue(Dense{ .n = n, .k = k, .rows = rows, .xs = 2, .xb = 1, .bits = bits, .cap = 0 }, 4);
        enc.setBuffer(y, 0, 5);
        enc.setBuffer(overflow, 0, 6);
        enc.dispatchGroups(mtl.Size.of((n + 63) / 64, (rows + 31) / 32, 1), mtl.Size.of(256, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
        for (0..rows) |row| for (0..n) |col| {
            var want: f64 = 0;
            for (0..k) |j| {
                const code: f32 = @floatFromInt(@as(u32, @intCast((col * 17 + j * 7) % 256)) & ((@as(u32, 1) << @intCast(bits)) - 1));
                const value = @mulAdd(f32, code, scale, bias);
                const w: f16 = @floatCast(value);
                const input: f16 = @floatCast(x.slice(f32, rows * 2 * k)[(row * 2 + 1) * k + j]);
                want += @as(f64, @floatCast(w)) * @as(f64, @floatCast(input));
            }
            try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, rows * n)[row * n + col]), 0.001);
        };
    }
    x.slice(f32, rows * 2 * k)[k] = 70000;
    overflow.slice(u32, 1)[0] = 0;
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipe);
    for ([_]mtl.Buffer{ x, words, scales, biases }, 0..) |buf, i| enc.setBuffer(buf, 0, i);
    enc.setValue(Dense{ .n = n, .k = k, .rows = rows, .xs = 2, .xb = 1, .bits = 8, .cap = 0 }, 4);
    enc.setBuffer(y, 0, 5);
    enc.setBuffer(overflow, 0, 6);
    enc.dispatchGroups(mtl.Size.of((n + 63) / 64, (rows + 31) / 32, 1), mtl.Size.of(256, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    try t.expect(overflow.slice(u32, 1)[0] > 0);
}

test "half grouped affine experts honor sorted routes and tile tails" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, "#define I2A_HALF_OPERANDS 1\n" ++ @embedFile("kernels/affine.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const k = 64;
    const n = 68;
    const slots = 70;
    const x = try device.buffer(slots * k * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    const words = try device.buffer(2 * 2 * n * k / 2, mtl.ResourceOptions.shared);
    defer words.deinit();
    const scales = try device.buffer(2 * 2 * n * 2, mtl.ResourceOptions.shared);
    defer scales.deinit();
    const biases = try device.buffer(2 * 2 * n * 2, mtl.ResourceOptions.shared);
    defer biases.deinit();
    const offsets = try device.buffer(3 * 4, mtl.ResourceOptions.shared);
    defer offsets.deinit();
    const sorted = try device.buffer(slots * 4, mtl.ResourceOptions.shared);
    defer sorted.deinit();
    const y = try device.buffer(slots * n * 4, mtl.ResourceOptions.shared);
    defer y.deinit();
    const overflow = try device.buffer(16, mtl.ResourceOptions.shared);
    defer overflow.deinit();
    for (x.slice(f32, slots * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 8;
    fill(words.slice(u32, 2 * 2 * n * k / 8), 2 * 2 * n, k, 4);
    @memset(scales.slice(u16, 2 * 2 * n), 0x3d80);
    @memset(biases.slice(u16, 2 * 2 * n), 0xbf00);
    offsets.slice(u32, 3)[0] = 0;
    offsets.slice(u32, 3)[1] = 35;
    offsets.slice(u32, 3)[2] = slots;
    for (0..slots) |slot| sorted.slice(u32, slots)[if (slot % 2 == 0) slot / 2 else 35 + slot / 2] = @intCast(slot);
    for ([_]bool{ true, false }) |up| {
        const pipe = try mtl.Pipeline.init(device, lib, if (up) "i2a_half_group_up" else "i2a_half_group_down", false);
        defer pipe.deinit();
        overflow.slice(u32, 1)[0] = 0;
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipe);
        for ([_]mtl.Buffer{ x, words, scales, biases, offsets, sorted }, 0..) |buf, i| enc.setBuffer(buf, 0, i);
        enc.setValue(Group{ .k = k, .n = n, .slots = slots, .topk = 2, .experts = 2, .gate_cap = if (up) 4 else 0, .up_cap = if (up) 25 else 0 }, 6);
        enc.setBuffer(y, 0, 7);
        enc.setBuffer(overflow, 0, 8);
        enc.dispatchGroups(mtl.Size.of(((if (up) @as(usize, 2 * n) else @as(usize, n)) + 63) / 64, 2, (slots + 31) / 32), mtl.Size.of(256, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
        for (0..slots) |slot| for (0..n) |col| {
            const expert = slot % 2;
            var first: f64 = 0;
            var second: f64 = 0;
            for (0..k) |j| {
                const v: f64 = x.slice(f32, slots * k)[(if (up) slot / 2 else slot) * k + j];
                first += v * weight(if (up) expert * 2 * n + 2 * col else expert * n + col, j, 4);
                if (up) second += v * weight(expert * 2 * n + 2 * col + 1, j, 4);
            }
            const want = if (up) 4 * std.math.tanh(first / 4) / (1 + @exp(-first)) * 25 * std.math.tanh(second / 25) else first;
            try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, slots * n)[slot * n + col]), 0.001);
        };
    }
}

test "staged affine matmul honors strides tails and half weights" {
    const pool = mtl.objc.Pool.push(); defer pool.pop();
    const device = try mtl.Device.init(); defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue(); defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @embedFile("kernels/affine_stage.metal"), mtl.CompileOptions.mlx()); defer lib.deinit();
    const prepare = try mtl.Pipeline.init(device, lib, "i2as_prepare", false); defer prepare.deinit();
    const gemm = try mtl.Pipeline.init(device, lib, "i2as_gemm", false); defer gemm.deinit();
    const rows = 33; const n = 68; const k = 128;
    const x = try device.buffer(rows * 2 * k * 4, mtl.ResourceOptions.shared); defer x.deinit();
    const w = try device.buffer(n * k, mtl.ResourceOptions.shared); defer w.deinit();
    const scales = try device.buffer(n * (k / 64) * 2, mtl.ResourceOptions.shared); defer scales.deinit();
    const biases = try device.buffer(n * (k / 64) * 2, mtl.ResourceOptions.shared); defer biases.deinit();
    const staged_w = try device.buffer(n * k * 2, mtl.ResourceOptions.private); defer staged_w.deinit();
    const y = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared); defer y.deinit();
    const overflow = try device.buffer(16, mtl.ResourceOptions.shared); defer overflow.deinit();
    for (x.slice(f32, rows * 2 * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 97)) - 48)) / 37.0;
    @memset(scales.slice(u16, n * (k / 64)), 0x3b83); @memset(biases.slice(u16, n * (k / 64)), 0xbd0d);
    const scale: f32 = @bitCast(@as(u32, 0x3b830000)); const bias: f32 = @bitCast(@as(u32, 0xbd0d0000));
    for ([_]u32{ 4, 8 }) |bits| {
        fill(w.slice(u32, n * k / 4), n, k, bits);
        overflow.slice(u32, 1)[0] = 0;
        const args = Dense{ .n = n, .k = k, .rows = rows, .xs = 2, .xb = 1, .bits = bits, .cap = 0 };
        const cb = queue.commandBuffer(); const enc = cb.compute(.serial);
        enc.setPipeline(prepare); enc.setBuffer(w, 0, 0); enc.setBuffer(scales, 0, 1); enc.setBuffer(biases, 0, 2); enc.setValue(args, 3); enc.setBuffer(staged_w, 0, 4); enc.setBuffer(overflow, 0, 5);
        enc.dispatchThreads(mtl.Size.of(n * k / 4, 1, 1), mtl.Size.of(256, 1, 1));
        enc.setPipeline(gemm); enc.setBuffer(x, 0, 0); enc.setBuffer(staged_w, 0, 1); enc.setValue(args, 2); enc.setBuffer(y, 0, 3);
        enc.dispatchGroups(mtl.Size.of((n + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        enc.end(); cb.commit(); cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        try t.expectEqual(@as(u32, 0), overflow.slice(u32, 1)[0]);
        for (0..rows) |row| for (0..n) |col| {
            var want: f64 = 0;
            for (0..k) |j| {
                const code: f32 = @floatFromInt(@as(u32, @intCast((col * 17 + j * 7) % 256)) & ((@as(u32, 1) << @intCast(bits)) - 1));
                const hw: f16 = @floatCast(@mulAdd(f32, code, scale, bias));
                const input: f32 = x.slice(f32, rows * 2 * k)[(row * 2 + 1) * k + j];
                want += @as(f64, @floatCast(hw)) * @as(f64, @floatCast(input));
            }
            try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, rows * n)[row * n + col]), 0.001);
        };
    }
    @memset(scales.slice(u16, n * (k / 64)), 0x4780);
    overflow.slice(u32, 1)[0] = 0;
    const args = Dense{ .n = n, .k = k, .rows = rows, .xs = 2, .xb = 1, .bits = 8, .cap = 0 };
    const cb = queue.commandBuffer(); const enc = cb.compute(.serial);
    enc.setPipeline(prepare); enc.setBuffer(w, 0, 0); enc.setBuffer(scales, 0, 1); enc.setBuffer(biases, 0, 2); enc.setValue(args, 3); enc.setBuffer(staged_w, 0, 4); enc.setBuffer(overflow, 0, 5);
    enc.dispatchThreads(mtl.Size.of(n * k / 4, 1, 1), mtl.Size.of(256, 1, 1));
    enc.end(); cb.commit(); cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    try t.expect(overflow.slice(u32, 1)[0] > 0);
}

test "float mpp projections preserve strided inputs and tile tails" {
    const pool = mtl.objc.Pool.push(); defer pool.pop();
    const device = try mtl.Device.init(); defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    const queue = try device.queue(); defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @embedFile("kernels/float_mpp.metal"), mtl.CompileOptions.mlx()); defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "i2af_gemm", false); defer pipe.deinit();
    const rows = 33; const n = 68; const k = 128;
    const x = try device.buffer(rows * 2 * k * 4, mtl.ResourceOptions.shared); defer x.deinit();
    const w = try device.buffer(n * k * 4, mtl.ResourceOptions.shared); defer w.deinit();
    const y = try device.buffer(rows * n * 4, mtl.ResourceOptions.shared); defer y.deinit();
    for (x.slice(f32, rows * 2 * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 97)) - 48)) / 37.0;
    for (w.slice(f32, n * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13) % 83)) - 41)) / 29.0;
    const args = Dense{ .n = n, .k = k, .rows = rows, .xs = 2, .xb = 1, .bits = 0, .cap = 0 };
    const cb = queue.commandBuffer(); const enc = cb.compute(.serial);
    enc.setPipeline(pipe); enc.setBuffer(x, 0, 0); enc.setBuffer(w, 0, 1); enc.setValue(args, 2); enc.setBuffer(y, 0, 3);
    enc.dispatchGroups(mtl.Size.of((n + 31) / 32, (rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
    enc.end(); cb.commit(); cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    for (0..rows) |row| for (0..n) |col| {
        var want: f64 = 0;
        for (0..k) |j| want += @as(f64, x.slice(f32, rows * 2 * k)[(row * 2 + 1) * k + j]) * @as(f64, w.slice(f32, n * k)[col * k + j]);
        try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, rows * n)[row * n + col]), 0.0001);
    };
}

test "oQ4e prefill without half pipelines matches per-token decode" {
    const path = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return error.SkipZigTest;
    const model = @import("model.zig");
    const state = @import("state.zig");
    const forward = @import("forward.zig");
    const prefill = @import("prefill.zig");
    const m = try model.Model.load(t.allocator, t.io, std.mem.span(path));
    defer m.deinit();
    const vocab = m.config.text.unpadded_vocab;
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1000 + (i * 7919) % 50000);
    var cache = try state.Cache.init(m.device, &m.config, ids.len);
    defer cache.deinit();
    var scratch = try state.Scratch.init(m.device, &m.config);
    defer scratch.deinit();
    for (ids) |id| try forward.position(m, &scratch, &cache, id);
    const want = try t.allocator.dupe(f32, scratch.logits.slice(f32, vocab));
    defer t.allocator.free(want);
    var ps = try prefill.PrefillScratch.init(m.device, &m.config, ids.len);
    defer ps.deinit();
    for ([_]bool{ false, true }) |half| {
        var filled = try state.Cache.init(m.device, &m.config, ids.len);
        defer filled.deinit();
        ps.half = half;
        try prefill.prefillWant(m, &ps, &filled, &ids, .last);
        const got = ps.logits.slice(f32, vocab);
        var worst: f32 = 0;
        var scale: f32 = 1;
        var best_got: usize = 0;
        var best_want: usize = 0;
        for (got, want, 0..) |g, w, i| {
            try t.expect(std.math.isFinite(g));
            worst = @max(worst, @abs(g - w));
            scale = @max(scale, @abs(w));
            if (g > got[best_got]) best_got = i;
            if (w > want[best_want]) best_want = i;
        }
        std.debug.print("oq4e prefill half={} logits_max_diff={e} scale={e}\n", .{ half, worst, scale });
        try t.expectEqual(best_want, best_got);
        try t.expectEqual(ids.len, filled.len);
        if (!half) try t.expect(worst <= scale * 1.5e-5);
    }
}

test "oQ4e prefill chunks longer than the local window match window-bounded chunks" {
    const path = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return error.SkipZigTest;
    const model = @import("model.zig");
    const state = @import("state.zig");
    const forward = @import("forward.zig");
    const prefill = @import("prefill.zig");
    const m = try model.Model.load(t.allocator, t.io, std.mem.span(path));
    defer m.deinit();
    const vocab = m.config.text.unpadded_vocab;
    const n = m.config.text.window + 76;
    const ids = try t.allocator.alloc(u32, n);
    defer t.allocator.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(1000 + (i * 7919) % 50000);
    var scratch = try state.Scratch.init(m.device, &m.config);
    defer scratch.deinit();
    var best_want: usize = 0;
    {
        var cache = try state.Cache.init(m.device, &m.config, n);
        defer cache.deinit();
        for (ids) |id| try forward.position(m, &scratch, &cache, id);
        const want = scratch.logits.slice(f32, vocab);
        for (want, 0..) |w, i| if (w > want[best_want]) {
            best_want = i;
        };
    }
    var ps = try prefill.PrefillScratch.initRows(m.device, &m.config, n, 1);
    defer ps.deinit();
    for ([_]bool{ false, true }) |half| {
        ps.half = half;
        var logits: [2][]f32 = undefined;
        var next: [2][]f32 = undefined;
        for ([_]usize{ n, 512 }, 0..) |chunk, side| {
            var cache = try state.Cache.init(m.device, &m.config, n + 1);
            defer cache.deinit();
            var at: usize = 0;
            while (at < n) {
                const take = @min(chunk, n - at);
                try prefill.prefillWant(m, &ps, &cache, ids[at..][0..take], if (at + take == n) .last else .none);
                at += take;
            }
            try t.expectEqual(n, cache.len);
            logits[side] = try t.allocator.dupe(f32, ps.logits.slice(f32, vocab));
            try forward.position(m, &scratch, &cache, 4242);
            next[side] = try t.allocator.dupe(f32, scratch.logits.slice(f32, vocab));
        }
        defer for (logits, next) |a, b| {
            t.allocator.free(a);
            t.allocator.free(b);
        };
        var worst: f32 = 0;
        var best: usize = 0;
        for (logits[0], logits[1], 0..) |a, b, i| {
            try t.expect(std.math.isFinite(a) and std.math.isFinite(b));
            worst = @max(worst, @abs(a - b));
            if (a > logits[0][best]) best = i;
        }
        std.debug.print("oq4e chunk {d} vs 512 half={} logits_max_diff={e}\n", .{ n, half, worst });
        try t.expectEqual(best_want, best);
        if (!half) {
            try t.expectEqualSlices(f32, logits[1], logits[0]);
            try t.expectEqualSlices(f32, next[1], next[0]);
        }
    }
}

const SharedProjections = extern struct { k: u32, full: u32, rows: u32, m: [8]u32 };

fn sameWords(a: []const f32, b: []const f32) bool {
    for (a, b) |x, y| if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) return false;
    return true;
}

test "shared-row affine kernels give each row the per-row kernel's bytes" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @embedFile("kernels/affine.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const dense = try mtl.Pipeline.init(device, lib, "i2a_dense", false);
    defer dense.deinit();
    const dense_shared = try mtl.Pipeline.init(device, lib, "i2a_dense_shared", false);
    defer dense_shared.deinit();
    const batch = try mtl.Pipeline.init(device, lib, "i2a_batch_projections", false);
    defer batch.deinit();
    const shared = try mtl.Pipeline.init(device, lib, "i2a_shared_projections", false);
    defer shared.deinit();
    const k = 2304;
    const n = 37;
    const wrows = 64;
    var prng = std.Random.DefaultPrng.init(0x72_6f_77_73);
    const rng = prng.random();
    const x = try device.buffer(3 * 8 * k * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    for (x.slice(f32, 3 * 8 * k)) |*v| v.* = rng.float(f32) * 4 - 2;
    var w: [3]mtl.Buffer = undefined;
    var sc: [3]mtl.Buffer = undefined;
    var bi: [3]mtl.Buffer = undefined;
    for (0..3) |i| {
        w[i] = try device.buffer(wrows * k, mtl.ResourceOptions.shared);
        for (w[i].slice(u32, wrows * k / 4)) |*v| v.* = rng.int(u32);
        sc[i] = try device.buffer(wrows * (k / 64) * 2, mtl.ResourceOptions.shared);
        bi[i] = try device.buffer(wrows * (k / 64) * 2, mtl.ResourceOptions.shared);
        for (sc[i].slice(u16, wrows * (k / 64)), bi[i].slice(u16, wrows * (k / 64))) |*s, *b| {
            s.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 0.02 + 0.001)) >> 16);
            b.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 0.5 - 0.25)) >> 16);
        }
    }
    defer for (0..3) |i| {
        w[i].deinit();
        sc[i].deinit();
        bi[i].deinit();
    };
    const out_a = try device.buffer(8 * 8 * wrows * 4, mtl.ResourceOptions.shared);
    defer out_a.deinit();
    const out_b = try device.buffer(8 * 8 * wrows * 4, mtl.ResourceOptions.shared);
    defer out_b.deinit();
    var checked: usize = 0;
    for ([_]u32{ 4, 8 }) |bits| for ([_]f32{ 0, 30 }) |cap| for (2..9) |rows| {
        const args = Dense{ .n = n, .k = k, .rows = @intCast(rows), .xs = 2, .xb = 1, .bits = bits, .cap = cap };
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for ([_]mtl.Pipeline{ dense, dense_shared }, [_]mtl.Buffer{ out_a, out_b }, 0..) |pipe, out, i| {
            enc.setPipeline(pipe);
            enc.setBuffer(x, 0, 0);
            enc.setBuffer(w[0], 0, 1);
            enc.setBuffer(sc[0], 0, 2);
            enc.setBuffer(bi[0], 0, 3);
            enc.setValue(args, 4);
            enc.setBuffer(out, 0, 5);
            enc.dispatchGroups(mtl.Size.of((n + 7) / 8, if (i == 0) rows else 1, 1), mtl.Size.of(256, 1, 1));
        }
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        if (!sameWords(out_a.slice(f32, rows * n), out_b.slice(f32, rows * n))) {
            std.debug.print("shared dense bits={d} cap={d} rows={d}: bytes differ\n", .{ bits, cap, rows });
            return error.TestUnexpectedResult;
        }
        checked += 1;
    };
    for ([_]bool{ false, true }) |full| for (2..9) |rows| {
        const ms: [8]u32 = .{ 13, 9, 7, 5, 11, 3, if (full) 6 else 0, if (full) 4 else 0 };
        var total: usize = 0;
        for (ms) |m| total += m;
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for ([_]mtl.Pipeline{ batch, shared }, [_]mtl.Buffer{ out_a, out_b }, 0..) |pipe, out, i| {
            enc.setPipeline(pipe);
            for (0..3) |j| enc.setBuffer(x, j * 8 * k * 4, j);
            for (0..3) |j| {
                enc.setBuffer(w[j], 0, 3 + j * 3);
                enc.setBuffer(sc[j], 0, 4 + j * 3);
                enc.setBuffer(bi[j], 0, 5 + j * 3);
            }
            for (0..8) |j| enc.setBuffer(out, j * 8 * wrows * 4, 12 + j);
            if (i == 0) {
                enc.setValue(Projections{ .k = k, .full = @intFromBool(full), .m = ms }, 20);
                enc.dispatchGroups(mtl.Size.of((total + 7) / 8, rows, 1), mtl.Size.of(256, 1, 1));
            } else {
                enc.setValue(SharedProjections{ .k = k, .full = @intFromBool(full), .rows = @intCast(rows), .m = ms }, 20);
                enc.dispatchGroups(mtl.Size.of((total + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            }
        }
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        for (ms, 0..) |m, j| {
            const at = j * 8 * wrows;
            if (!sameWords(out_a.slice(f32, 8 * 8 * wrows)[at..][0 .. rows * m], out_b.slice(f32, 8 * 8 * wrows)[at..][0 .. rows * m])) {
                std.debug.print("shared projections full={} rows={d} output={d}: bytes differ\n", .{ full, rows, j });
                return error.TestUnexpectedResult;
            }
        }
        checked += 1;
    };
    std.debug.print("shared-row affine kernels: {d} cases byte-equal to the per-row kernels\n", .{checked});
}


/// The affine dot and float projection before their loads were issued ahead of the sums.
const serial_affine =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\struct i2a_dense_args { uint n, k, rows, xs, xb, bits; float cap; };
    \\struct i2a_expert_args { uint k, n, slots, topk; float gate_cap, up_cap; };
    \\struct i2a_embed_args { uint width, rows; float eps; };
    \\struct i2a_projection_args { uint k, full; uint m[8]; };
    \\struct i2a_pair_args { uint k, m0, m1; };
    \\
    \\inline float i2a_weight(const device uint* w, const device bfloat* s, const device bfloat* b,
    \\                        size_t row, uint k, uint width, uint bits) {
    \\  const uint per = 32u / bits;
    \\  const uint code = (w[row * (width / per) + k / per] >> ((k % per) * bits)) & ((1u << bits) - 1u);
    \\  const size_t group = row * (width / 64u) + k / 64u;
    \\  return float(code) * float(s[group]) + float(b[group]);
    \\}
    \\
    \\inline float i2a_dot(const device float* x, const device uint* w, const device bfloat* s,
    \\                     const device bfloat* b, size_t row, uint width, uint bits, uint lane) {
    \\  const uint per = 32u / bits;
    \\  const uint mask = (1u << bits) - 1u;
    \\  float acc = 0.0f;
    \\  for (uint word = lane; word < width / per; word += 32u) {
    \\    const uint packed = w[row * (width / per) + word];
    \\    const size_t group = row * (width / 64u) + (word * per) / 64u;
    \\    const float scale = float(s[group]), bias = float(b[group]);
    \\    const uint4 shift = uint4(0u, bits, bits * 2u, bits * 3u);
    \\    const float4 code0 = float4((uint4(packed) >> shift) & mask);
    \\    const float4 input0 = float4(*reinterpret_cast<const device packed_float4*>(x + word * per));
    \\    const float4 value0 = fma(code0, float4(scale), float4(bias));
    \\    acc = fma(input0.x, value0.x, acc);
    \\    acc = fma(input0.y, value0.y, acc);
    \\    acc = fma(input0.z, value0.z, acc);
    \\    acc = fma(input0.w, value0.w, acc);
    \\    if (bits == 4u) {
    \\      const float4 code1 = float4((uint4(packed >> 16u) >> shift) & mask);
    \\      const float4 input1 = float4(*reinterpret_cast<const device packed_float4*>(x + word * per + 4u));
    \\      const float4 value1 = fma(code1, float4(scale), float4(bias));
    \\      acc = fma(input1.x, value1.x, acc);
    \\      acc = fma(input1.y, value1.y, acc);
    \\      acc = fma(input1.z, value1.z, acc);
    \\      acc = fma(input1.w, value1.w, acc);
    \\    }
    \\  }
    \\  return simd_sum(acc);
    \\}
    \\
    \\[[kernel]] void i2a_float(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
    \\    constant i2a_dense_args& a [[buffer(2)]], device float* y [[buffer(3)]],
    \\    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    \\  const uint col = group.x * 8u + tid / 32u;
    \\  const uint row = group.y, lane = tid % 32u;
    \\  if (col >= a.n || row >= a.rows) return;
    \\  const device float* input = x + (size_t)(row * a.xs + a.xb) * a.k;
    \\  const device float* weight = w + (size_t)col * a.k;
    \\  float acc = 0.0f;
    \\  uint base = 0u;
    \\  for (; base + 128u <= a.k; base += 128u) {
    \\    const uint k = base + lane * 4u;
    \\    const float4 xv = float4(*reinterpret_cast<const device packed_float4*>(input + k));
    \\    const float4 wv = float4(*reinterpret_cast<const device packed_float4*>(weight + k));
    \\    acc = fma(xv.x, wv.x, acc);
    \\    acc = fma(xv.y, wv.y, acc);
    \\    acc = fma(xv.z, wv.z, acc);
    \\    acc = fma(xv.w, wv.w, acc);
    \\  }
    \\  for (uint k = base + lane; k < a.k; k += 32u) acc = fma(input[k], weight[k], acc);
    \\  acc = simd_sum(acc);
    \\  if (lane == 0u) y[(size_t)row * a.n + col] = acc;
    \\}
    \\
    \\[[kernel]] void i2a_dense(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    \\    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    \\    constant i2a_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    \\    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    \\  const uint col = group.x * 8u + tid / 32u;
    \\  const uint row = group.y;
    \\  const uint lane = tid % 32u;
    \\  if (col >= a.n || row >= a.rows) return;
    \\  const device float* input = x + (size_t)(row * a.xs + a.xb) * a.k;
    \\  const float acc = i2a_dot(input, w, s, b, col, a.k, a.bits, lane);
    \\  if (lane == 0u) y[(size_t)row * a.n + col] = a.cap > 0.0f ? a.cap * precise::tanh(acc / a.cap) : acc;
    \\}
    \\
;

test "prefetched affine and float projections keep the serial kernels' bytes" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, @embedFile("kernels/affine.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    const old_lib = try mtl.Library.fromSource(device, serial_affine, mtl.CompileOptions.mlx());
    defer old_lib.deinit();
    var prng = std.Random.DefaultPrng.init(0x70_72_65_66);
    const rng = prng.random();
    const kmax = 4608;
    const n = 41;
    const x = try device.buffer(3 * kmax * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    for (x.slice(f32, 3 * kmax)) |*v| v.* = rng.float(f32) * 4 - 2;
    const w = try device.buffer(n * kmax * 4, mtl.ResourceOptions.shared);
    defer w.deinit();
    const sc = try device.buffer(n * (kmax / 64) * 2, mtl.ResourceOptions.shared);
    defer sc.deinit();
    const bi = try device.buffer(n * (kmax / 64) * 2, mtl.ResourceOptions.shared);
    defer bi.deinit();
    for (sc.slice(u16, n * (kmax / 64)), bi.slice(u16, n * (kmax / 64))) |*s, *b| {
        s.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 0.02 + 0.001)) >> 16);
        b.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 0.5 - 0.25)) >> 16);
    }
    const outs = [_]mtl.Buffer{ try device.buffer(3 * n * 4, mtl.ResourceOptions.shared), try device.buffer(3 * n * 4, mtl.ResourceOptions.shared) };
    defer for (outs) |o| o.deinit();
    var checked: usize = 0;
    for ([_][]const u8{ "i2a_dense", "i2a_float" }) |name| {
        const pipes = [_]mtl.Pipeline{ try mtl.Pipeline.init(device, old_lib, name, false), try mtl.Pipeline.init(device, lib, name, false) };
        defer for (pipes) |p| p.deinit();
        const float = std.mem.eql(u8, name, "i2a_float");
        for ([_]u32{ 64, 2304, 3072, 4608, 4544 }) |k| for ([_]u32{ 4, 8 }) |bits| {
            if (float and bits == 8) continue;
            if (float) {
                for (w.slice(f32, n * kmax)) |*v| v.* = rng.float(f32) * 2 - 1;
            } else {
                for (w.slice(u32, n * kmax)) |*v| v.* = rng.int(u32);
            }
            const args = Dense{ .n = n, .k = k, .rows = 3, .xs = 1, .xb = 0, .bits = bits, .cap = if (k == 2304) 30 else 0 };
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            for (pipes, outs) |pipe, out| {
                enc.setPipeline(pipe);
                enc.setBuffer(x, 0, 0);
                enc.setBuffer(w, 0, 1);
                if (float) {
                    enc.setValue(args, 2);
                    enc.setBuffer(out, 0, 3);
                } else {
                    enc.setBuffer(sc, 0, 2);
                    enc.setBuffer(bi, 0, 3);
                    enc.setValue(args, 4);
                    enc.setBuffer(out, 0, 5);
                }
                enc.dispatchGroups(mtl.Size.of((n + 7) / 8, 3, 1), mtl.Size.of(256, 1, 1));
            }
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |_| return error.GpuFailed;
            if (!sameWords(outs[0].slice(f32, 3 * n), outs[1].slice(f32, 3 * n))) {
                std.debug.print("prefetched {s} k={d} bits={d}: bytes differ\n", .{ name, k, bits });
                return error.TestUnexpectedResult;
            }
            checked += 1;
        };
    }
    std.debug.print("prefetched affine projections: {d} cases byte-equal to the serial kernels\n", .{checked});
}
