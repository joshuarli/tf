const std = @import("std");
const mtl = @import("metal");
const kernels = @import("kernels.zig");

const old_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\struct i2_mix_args { uint width, depth, k; float scale; };
    \\[[kernel]] void i2_mix(const device float* S [[buffer(0)]], const device float* Q [[buffer(1)]],
    \\    constant i2_mix_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    \\    uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
    \\  threadgroup float part[32];
    \\  threadgroup float w[4][5];
    \\  for (uint k = 0; k < a.k; ++k) {
    \\    for (uint j = 0; j < a.depth; ++j) {
    \\      float sum = 0.0f;
    \\      for (uint i = t; i < a.width; i += nth) sum = fma(S[j * a.width + i], Q[k * a.width + i], sum);
    \\      sum = simd_sum(sum);
    \\      if (t % 32u == 0u) part[t / 32u] = sum;
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      float total = 0.0f;
    \\      for (uint s = 0; s < nth / 32u; ++s) total += part[s];
    \\      if (t == 0u) w[k][j] = total * a.scale;
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (t == 0u) {
    \\      float peak = w[k][0];
    \\      for (uint j = 1; j < a.depth; ++j) peak = max(peak, w[k][j]);
    \\      float total = 0.0f;
    \\      for (uint j = 0; j < a.depth; ++j) total += precise::exp(w[k][j] - peak);
    \\      for (uint j = 0; j < a.depth; ++j) w[k][j] = precise::exp(w[k][j] - peak) / total;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = t; i < a.width; i += nth) {
    \\      float sum = 0.0f;
    \\      for (uint j = 0; j < a.depth; ++j) sum = fma(w[k][j], S[j * a.width + i], sum);
    \\      O[k * a.width + i] = sum;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  }
    \\}
    \\
    \\[[kernel]] void i2_mix4(const device float* S [[buffer(0)]], const device float* Q [[buffer(1)]],
    \\    const device float* K [[buffer(2)]], const device float* V [[buffer(3)]], const device float* F [[buffer(4)]],
    \\    constant i2_mix_args& a [[buffer(5)]], device float* O [[buffer(6)]],
    \\    uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
    \\  threadgroup float part[32];
    \\  threadgroup float w[4][5];
    \\  for (uint k = 0; k < 4u; ++k) {
    \\    const device float* q = k == 0 ? Q : k == 1 ? K : k == 2 ? V : F;
    \\    for (uint j = 0; j < a.depth; ++j) {
    \\      float sum = 0.0f;
    \\      for (uint i = t; i < a.width; i += nth) sum = fma(S[j * a.width + i], q[i], sum);
    \\      sum = simd_sum(sum);
    \\      if (t % 32u == 0u) part[t / 32u] = sum;
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      float total = 0.0f;
    \\      for (uint s = 0; s < nth / 32u; ++s) total += part[s];
    \\      if (t == 0u) w[k][j] = total * a.scale;
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (t == 0u) {
    \\      float peak = w[k][0];
    \\      for (uint j = 1; j < a.depth; ++j) peak = max(peak, w[k][j]);
    \\      float total = 0.0f;
    \\      for (uint j = 0; j < a.depth; ++j) total += precise::exp(w[k][j] - peak);
    \\      for (uint j = 0; j < a.depth; ++j) w[k][j] = precise::exp(w[k][j] - peak) / total;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = t; i < a.width; i += nth) {
    \\      float sum = 0.0f;
    \\      for (uint j = 0; j < a.depth; ++j) sum = fma(w[k][j], S[j * a.width + i], sum);
    \\      O[k * a.width + i] = sum;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  }
    \\}
    \\
    ;

const width = 256;
const Rms = extern struct { width: u32 = width, blocks: u32 = 0, eps: f32 = 1e-6 };
const Pair = extern struct { count: u32 = width };
const OldMix = extern struct { width: u32 = width, depth: u32, k: u32 = 4, scale: f32 = 0.0625 };
const Mix = extern struct { width: u32 = width, depth: u32, k: u32 = 4, scale: f32 = 0.0625, indices: [5]u32 };

fn norm(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, x: mtl.Buffer, xo: usize, y: mtl.Buffer, yo: usize, count: usize) void {
    enc.setPipeline(pipe);
    enc.setBuffer(x, xo, 0);
    enc.setBuffer(x, xo, 1);
    enc.setValue(Rms{}, 2);
    enc.setBuffer(y, yo, 3);
    enc.dispatchGroups(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}

fn copy(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, x: mtl.Buffer, xo: usize, y: mtl.Buffer, yo: usize) void {
    enc.setPipeline(pipe);
    enc.setBuffer(x, xo, 0);
    enc.setBuffer(y, yo, 1);
    enc.setValue(Pair{}, 2);
    enc.dispatchThreads(mtl.Size.of(width, 1, 1), mtl.Size.of(256, 1, 1));
}

test "published depth and rewound pass banks match packed_rows normalization bitwise" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, kernels.glue, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const oldlib = try mtl.Library.fromSource(device, old_source, mtl.CompileOptions.mlx());
    defer oldlib.deinit();
    const rms = try mtl.Pipeline.init(device, lib, "i2_rms", false);
    defer rms.deinit();
    const cp = try mtl.Pipeline.init(device, lib, "i2_copy", false);
    defer cp.deinit();
    const mix = try mtl.Pipeline.init(device, lib, "i2_mix", false);
    defer mix.deinit();
    const mix4 = try mtl.Pipeline.init(device, lib, "i2_mix4", false);
    defer mix4.deinit();
    const old = try mtl.Pipeline.init(device, oldlib, "i2_mix", false);
    defer old.deinit();
    const old4 = try mtl.Pipeline.init(device, oldlib, "i2_mix4", false);
    defer old4.deinit();
    const raw = try device.buffer(16 * width * 4, mtl.ResourceOptions.shared);
    defer raw.deinit();
    const blocks = try device.buffer(5 * width * 4, mtl.ResourceOptions.shared);
    defer blocks.deinit();
    const bank = try device.buffer(5 * width * 4, mtl.ResourceOptions.shared);
    defer bank.deinit();
    const current = try device.buffer(width * 4, mtl.ResourceOptions.shared);
    defer current.deinit();
    const packed_rows = try device.buffer(5 * width * 4, mtl.ResourceOptions.shared);
    defer packed_rows.deinit();
    const normed = try device.buffer(5 * width * 4, mtl.ResourceOptions.shared);
    defer normed.deinit();
    const query = try device.buffer(4 * width * 4, mtl.ResourceOptions.shared);
    defer query.deinit();
    const got = try device.buffer(32 * 4 * width * 4, mtl.ResourceOptions.shared);
    defer got.deinit();
    const want = try device.buffer(32 * 4 * width * 4, mtl.ResourceOptions.shared);
    defer want.deinit();
    for (raw.slice(f32, 16 * width), 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 97 + i / width * 37) % 997)) / 498.0 - 1.0;
    for (query.slice(f32, 4 * width), 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 53) % 251)) / 125.0 - 1.0;
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    const row = width * 4;
    var result: usize = 0;
    norm(enc, rms, raw, 12 * row, blocks, 0, 1);
    for (0..3) |pass| {
        if (pass > 0) {
            copy(enc, cp, blocks, 2 * row, bank, 0);
            norm(enc, rms, raw, ((pass - 1) * 4 + 3) * row, bank, row, 1);
        }
        for (0..3) |j| {
            if (j > 0) norm(enc, rms, raw, (pass * 4 + j - 1) * row, blocks, j * row, 1);
            for (0..j + 1) |k| {
                const raw_index = if (j == k) 12 else pass * 4 + j - k - 1;
                copy(enc, cp, raw, raw_index * row, packed_rows, k * row);
            }
            copy(enc, cp, raw, (pass * 4 + 3) * row, packed_rows, (j + 1) * row);
            norm(enc, rms, packed_rows, 0, normed, 0, j + 2);
            norm(enc, rms, raw, (pass * 4 + 3) * row, current, 0, 1);
            var indices: [5]u32 = @splat(0);
            for (0..j + 1) |k| indices[k] = @intCast(j - k);
            enc.setPipeline(old); enc.setBuffer(normed, 0, 0); enc.setBuffer(query, 0, 1);
            enc.setValue(OldMix{ .depth = @intCast(j + 2) }, 2); enc.setBuffer(want, result * 4 * row, 3);
            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
            enc.setPipeline(mix); enc.setBuffer(blocks, 0, 0); enc.setBuffer(query, 0, 1);
            enc.setValue(Mix{ .depth = @intCast(j + 2), .indices = indices }, 2); enc.setBuffer(got, result * 4 * row, 3); enc.setBuffer(current, 0, 4);
            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
            result += 1;
        }
        if (pass > 0) {
            copy(enc, cp, raw, ((pass - 1) * 4 + 1) * row, packed_rows, 0);
            copy(enc, cp, raw, ((pass - 1) * 4 + 3) * row, packed_rows, row);
            norm(enc, rms, packed_rows, 0, normed, 0, 2);
            enc.setPipeline(old4); enc.setBuffer(normed, 0, 0);
            for (0..4) |j| enc.setBuffer(query, j * row, j + 1);
            enc.setValue(OldMix{ .depth = 2 }, 5); enc.setBuffer(want, result * 4 * row, 6);
            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
            enc.setPipeline(mix4); enc.setBuffer(bank, 0, 0);
            for (0..4) |j| enc.setBuffer(query, j * row, j + 1);
            enc.setValue(Mix{ .depth = 2, .indices = .{ 0, 0, 0, 0, 0 } }, 5); enc.setBuffer(got, result * 4 * row, 6); enc.setBuffer(bank, row, 7);
            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
            result += 1;
        }
    }
    enc.end(); cb.commit(); cb.wait();
    if (cb.failure()) |msg| { std.log.err("depth test: {s}", .{msg}); return error.GpuFailed; }
    try std.testing.expectEqualSlices(u8, want.contents()[0 .. result * 4 * row], got.contents()[0 .. result * 4 * row]);
}
