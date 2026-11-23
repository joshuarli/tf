const std = @import("std");
const mtl = @import("metal");
const t = std.testing;

const Dense = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, bits: u32, cap: f32 };
const Expert = extern struct { k: u32, n: u32, slots: u32, topk: u32, gate_cap: f32, up_cap: f32 };

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
    enc.dispatchGroups(mtl.Size.of((n + 7) / 8, rows, 1), mtl.Size.of(256, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
}

test "affine dense and expert kernels agree with dequantized float64 weights" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = 64;
    const n = 4;
    const x = try device.buffer(2 * k * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    const words = try device.buffer(2 * 2 * n * k, mtl.ResourceOptions.shared);
    defer words.deinit();
    const scales = try device.buffer(2 * 2 * n * 2, mtl.ResourceOptions.shared);
    defer scales.deinit();
    const biases = try device.buffer(2 * 2 * n * 2, mtl.ResourceOptions.shared);
    defer biases.deinit();
    const picks = try device.buffer(2 * 4, mtl.ResourceOptions.shared);
    defer picks.deinit();
    const y = try device.buffer(2 * n * 4, mtl.ResourceOptions.shared);
    defer y.deinit();
    for (x.slice(f32, 2 * k), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 8;
    @memset(scales.slice(u16, 2 * 2 * n), 0x3d80);
    @memset(biases.slice(u16, 2 * 2 * n), 0xbf00);
    picks.slice(i32, 2)[0] = 1;
    picks.slice(i32, 2)[1] = 0;
    fill(words.slice(u32, 2 * 2 * n * k / 4), 2 * 2 * n, k, 8);
    try launch(device, queue, "i2a_dense", &.{ x, words, scales, biases, y }, Dense{ .n = n, .k = k, .rows = 2, .xs = 1, .xb = 0, .bits = 8, .cap = 0 }, n, 2);
    for (0..2) |row| for (0..n) |col| {
        var want: f64 = 0;
        for (0..k) |j| want += @as(f64, x.slice(f32, 2 * k)[row * k + j]) * weight(col, j, 8);
        try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, 2 * n)[row * n + col]), 0.001);
    };
    fill(words.slice(u32, 2 * 2 * n * k / 4), 2 * 2 * n, k, 4);
    try launch(device, queue, "i2a_expert_up", &.{ x, words, scales, biases, picks, y }, Expert{ .k = k, .n = n, .slots = 2, .topk = 1, .gate_cap = 4, .up_cap = 25 }, n, 2);
    for (0..2) |row| for (0..n) |col| {
        const expert: usize = @intCast(picks.slice(i32, 2)[row]);
        var gate: f64 = 0;
        var up: f64 = 0;
        for (0..k) |j| {
            const v: f64 = x.slice(f32, 2 * k)[row * k + j];
            gate += v * weight(expert * 2 * n + 2 * col, j, 4);
            up += v * weight(expert * 2 * n + 2 * col + 1, j, 4);
        }
        const want = 4 * std.math.tanh(gate / 4) / (1 + @exp(-gate)) * 25 * std.math.tanh(up / 25);
        try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, 2 * n)[row * n + col]), 0.001);
    };
    try launch(device, queue, "i2a_expert_down", &.{ x, words, scales, biases, picks, y }, Expert{ .k = k, .n = n, .slots = 2, .topk = 1, .gate_cap = 0, .up_cap = 0 }, n, 2);
    for (0..2) |row| for (0..n) |col| {
        const expert: usize = @intCast(picks.slice(i32, 2)[row]);
        var want: f64 = 0;
        for (0..k) |j| want += @as(f64, x.slice(f32, 2 * k)[row * k + j]) * weight(expert * n + col, j, 4);
        try t.expectApproxEqAbs(want, @as(f64, y.slice(f32, 2 * n)[row * n + col]), 0.001);
    };
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
