const std = @import("std");
const mtl = @import("metal");
const model = @import("model.zig");
const state = @import("state.zig");
const prefill = @import("prefill.zig");
const forward = @import("forward.zig");
const batch = @import("batch.zig");

fn buffers(c: *state.Cache) [6]mtl.Buffer {
    return .{ c.keys, c.values, c.k_conv, c.v_conv, c.attn_conv, c.mlp_conv };
}

fn copyCache(m: *const model.Model, src: *state.Cache, dst: *state.Cache) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    for (buffers(src), buffers(dst)) |a, b| blit.copy(a, 0, b, 0, a.length());
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    dst.len = src.len;
}

fn compare(m: *const model.Model, a: mtl.Buffer, b: mtl.Buffer, half: bool) !f32 {
    const x = try m.device.buffer(a.length(), mtl.ResourceOptions.shared);
    defer x.deinit();
    const y = try m.device.buffer(b.length(), mtl.ResourceOptions.shared);
    defer y.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    blit.copy(a, 0, x, 0, a.length());
    blit.copy(b, 0, y, 0, b.length());
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    var worst: f32 = 0;
    if (half) {
        for (x.slice(f16, x.length() / 2), y.slice(f16, y.length() / 2)) |av, bv| {
            if (@as(u16, @bitCast(av)) == @as(u16, @bitCast(bv))) continue;
            const diff = @abs(@as(f32, av) - @as(f32, bv));
            if (!std.math.isFinite(diff) or diff > 1.5e-5) return error.CacheMismatch;
            worst = @max(worst, diff);
        }
        return worst;
    }
    for (x.slice(f32, x.length() / 4), y.slice(f32, y.length() / 4)) |av, bv| {
        if (@as(u32, @bitCast(av)) == @as(u32, @bitCast(bv))) continue;
        const diff = @abs(av - bv);
        if (!std.math.isFinite(diff) or diff > 1.5e-5) return error.CacheMismatch;
        worst = @max(worst, diff);
    }
    return worst;
}

fn argmax(x: []const f32) usize {
    var best: usize = 0;
    for (x[1..], 1..) |v, i| if (v > x[best]) { best = i; };
    return best;
}

test "batched independent cache steps match decode" {
    const path = std.c.getenv("TF_INKLING2_LOOP_DIR") orelse return error.SkipZigTest;
    const m = try model.Model.load(std.testing.allocator, std.testing.io, std.mem.span(path));
    defer m.deinit();
    var ps = try prefill.PrefillScratch.init(m.device, &m.config, 16);
    defer ps.deinit();
    var ds = try state.Scratch.init(m.device, &m.config);
    defer ds.deinit();
    var bs = try batch.BatchScratch.init(m.device, &m.config, 4);
    defer bs.deinit();
    const vocab = m.config.text.unpadded_vocab;
    const logits = try std.testing.allocator.alloc(f32, 4 * vocab);
    defer std.testing.allocator.free(logits);
    for ([_]usize{ 1, 2, 4, 4 }, 0..) |width, case| {
        var actual: [4]state.Cache = undefined;
        var expected: [4]state.Cache = undefined;
        var caches: [4]*state.Cache = undefined;
        var ids: [4]u32 = undefined;
        var made: usize = 0;
        defer for (0..made) |i| { actual[i].deinit(); expected[i].deinit(); };
        for (0..width) |i| {
            const capacity: usize = if (case == 3) 80 else 32;
            actual[i] = try state.Cache.init(m.device, &m.config, capacity);
            expected[i] = try state.Cache.init(m.device, &m.config, capacity);
            made += 1;
            caches[i] = &actual[i];
            const prompt = [_]u32{ 100, 125, 213, 324, 528, 731, 1023, 1101 };
            if (case == 3) {
                const len = 62 + i;
                var at: usize = 0;
                while (at < len) {
                    const take = @min(prompt.len, len - at);
                    try prefill.prefillWant(m, &ps, &actual[i], prompt[0..take], .none);
                    at += take;
                }
            } else try prefill.prefillWant(m, &ps, &actual[i], prompt[0 .. 4 + i], .none);
            try copyCache(m, &actual[i], &expected[i]);
            ids[i] = @intCast(150 + i);
        }
        try batch.step(m, &bs, caches[0..width], ids[0..width], logits[0 .. width * vocab]);
        for (0..width) |i| {
            try forward.position(m, &ds, &expected[i], ids[i]);
            const got = logits[i * vocab ..][0..vocab];
            const want = ds.logits.slice(f32, vocab);
            var worst: f32 = 0;
            for (got, want) |a, b| {
                const diff = @abs(a - b);
                try std.testing.expect(std.math.isFinite(diff) and diff <= 1.5e-5);
                worst = @max(worst, diff);
            }
            try std.testing.expectEqual(argmax(want), argmax(got));
            try std.testing.expectEqual(expected[i].len, actual[i].len);
            var state_worst: f32 = 0;
            for (buffers(&actual[i]), buffers(&expected[i]), 0..) |a, b, index| state_worst = @max(state_worst, try compare(m, a, b, index < 2));
            std.debug.print("batch width={d} row={d} logits_max={e} cache_max={e} absolute_tolerance=1.5e-5\n", .{ width, i, worst, state_worst });
        }
    }
}
