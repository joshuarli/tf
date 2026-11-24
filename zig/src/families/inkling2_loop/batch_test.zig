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
    var bs = try batch.BatchScratch.init(m.device, &m.config, 8);
    defer bs.deinit();
    const vocab = m.config.text.unpadded_vocab;
    const logits = try std.testing.allocator.alloc(f32, 8 * vocab);
    defer std.testing.allocator.free(logits);
    for ([_]usize{ 1, 2, 4, 4, 8 }, 0..) |width, case| {
        var actual: [8]state.Cache = undefined;
        var expected: [8]state.Cache = undefined;
        var caches: [8]*state.Cache = undefined;
        var ids: [8]u32 = undefined;
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
            } else try prefill.prefillWant(m, &ps, &actual[i], prompt[0..@min(4 + i, prompt.len)], .none);
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

const Prompt = struct { name: []const u8, token_ids: []const u32 };
const stream_capacity: usize = 112;

const Streams = struct {
    gpa: std.mem.Allocator,
    m: *model.Model,
    ps: prefill.PrefillScratch,
    bs: batch.BatchScratch,
    verifier: @import("group.zig").Verifier,
    base: [8]state.Cache,
    ids: [8]u32,
    logits: []f32,

    fn init(gpa: std.mem.Allocator, path: []const u8) !*Streams {
        const s = try gpa.create(Streams);
        errdefer gpa.destroy(s);
        s.gpa = gpa;
        s.m = try model.Model.load(gpa, std.testing.io, path);
        errdefer s.m.deinit();
        const config = &s.m.config;
        s.ps = try prefill.PrefillScratch.init(s.m.device, config, 128);
        errdefer s.ps.deinit();
        s.bs = try batch.BatchScratch.init(s.m.device, config, 8);
        errdefer s.bs.deinit();
        s.verifier = try @import("group.zig").Verifier.init(s.m.device, config);
        errdefer s.verifier.deinit();
        const parsed = try std.json.parseFromSlice([]const Prompt, gpa, @embedFile("fixtures/prompts.json"), .{});
        defer parsed.deinit();
        const agent = parsed.value[0].token_ids;
        const prompts = [_][]const u32{ agent[0..61], parsed.value[1].token_ids, parsed.value[2].token_ids, parsed.value[3].token_ids, agent[400..471], agent[900..953], agent[1500..1588], agent[2000..2033] };
        var made: usize = 0;
        errdefer for (s.base[0..made]) |*c| c.deinit();
        for (prompts, 0..) |prompt, i| {
            s.base[i] = try state.Cache.init(s.m.device, config, stream_capacity);
            made += 1;
            var at: usize = 0;
            while (at < prompt.len) {
                const take = @min(s.ps.chunk, prompt.len - at);
                try prefill.prefillWant(s.m, &s.ps, &s.base[i], prompt[at..][0..take], .none);
                at += take;
            }
            s.ids[i] = agent[at + 7 * i + 3];
        }
        s.logits = try gpa.alloc(f32, 8 * config.text.unpadded_vocab);
        return s;
    }

    fn deinit(s: *Streams) void {
        s.gpa.free(s.logits);
        for (&s.base) |*c| c.deinit();
        s.verifier.deinit();
        s.bs.deinit();
        s.ps.deinit();
        s.m.deinit();
        s.gpa.destroy(s);
    }

    fn row(s: *const Streams, i: usize) []const f32 {
        const vocab = s.m.config.text.unpadded_vocab;
        return s.logits[i * vocab ..][0..vocab];
    }
};

fn readback(m: *const model.Model, source: mtl.Buffer) !mtl.Buffer {
    const out = try m.device.buffer(source.length(), mtl.ResourceOptions.shared);
    errdefer out.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    blit.copy(source, 0, out, 0, source.length());
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    return out;
}

fn expectSameCache(m: *const model.Model, a: *state.Cache, b: *state.Cache) !void {
    try std.testing.expectEqual(a.len, b.len);
    for (buffers(a), buffers(b)) |x, y| {
        const xs = try readback(m, x);
        defer xs.deinit();
        const ys = try readback(m, y);
        defer ys.deinit();
        try std.testing.expectEqualSlices(u8, xs.slice(u8, xs.length()), ys.slice(u8, ys.length()));
    }
}

fn hashed(values: []const f32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(values));
}

fn checkAlone(path: []const u8) !void {
    const gpa = std.testing.allocator;
    const s = try Streams.init(gpa, path);
    defer s.deinit();
    const config = &s.m.config;
    var alone: [8]u64 = undefined;
    var after: [8]state.Cache = undefined;
    var made: usize = 0;
    defer for (after[0..made]) |*c| c.deinit();
    for (0..8) |i| {
        after[i] = try state.Cache.init(s.m.device, config, stream_capacity);
        made += 1;
        try copyCache(s.m, &s.base[i], &after[i]);
        try batch.step(s.m, &s.bs, &.{&after[i]}, &.{s.ids[i]}, s.logits[0..config.text.unpadded_vocab]);
        alone[i] = hashed(s.row(0));
    }
    var work: [8]state.Cache = undefined;
    var work_made: usize = 0;
    defer for (work[0..work_made]) |*c| c.deinit();
    for (&work) |*c| {
        c.* = try state.Cache.init(s.m.device, config, stream_capacity);
        work_made += 1;
    }
    var checked: usize = 0;
    for (2..9) |width| {
        var caches: [8]*state.Cache = undefined;
        var ids: [8]u32 = undefined;
        var order: [8]usize = undefined;
        for (0..width) |j| {
            order[j] = (width + 3 * j) % 8;
            for (order[0..j]) |prior| if (prior == order[j]) {
                order[j] = (order[j] + 1) % 8;
            };
            try copyCache(s.m, &s.base[order[j]], &work[j]);
            caches[j] = &work[j];
            ids[j] = s.ids[order[j]];
        }
        try batch.step(s.m, &s.bs, caches[0..width], ids[0..width], s.logits[0 .. width * config.text.unpadded_vocab]);
        for (0..width) |j| {
            try std.testing.expectEqual(alone[order[j]], hashed(s.row(j)));
            try expectSameCache(s.m, &work[j], &after[order[j]]);
            checked += 1;
        }
    }
    std.debug.print("rows alone and grouped {s}: {d} rows of batch steps at widths 2-8 equal their one-row steps bit for bit\n", .{ path, checked });
}

test "a row's logits and caches are bit-equal stepped alone or with one to seven other rows" {
    var ran = false;
    for ([_][*:0]const u8{ "TF_INKLING2_LOOP_DIR", "TF_INKLING2_LOOP_OQ4E_DIR" }) |name| {
        const path = std.c.getenv(name) orelse continue;
        try checkAlone(std.mem.span(path));
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}

fn checkGroupedAlone(path: []const u8) !void {
    const gpa = std.testing.allocator;
    const s = try Streams.init(gpa, path);
    defer s.deinit();
    const config = &s.m.config;
    const vocab = config.text.unpadded_vocab;
    const drafts = [8][3]u32{ .{ s.ids[0], 1101, 2024 }, .{ s.ids[1], 6, 0 }, .{ s.ids[2], 731, 0 }, .{ s.ids[3], 324, 0 }, .{ s.ids[4], 0, 0 }, .{ s.ids[5], 99, 0 }, .{ s.ids[6], 0, 0 }, .{ s.ids[7], 5000, 0 } };
    const sizes = [8]usize{ 3, 2, 2, 2, 1, 2, 1, 2 };
    var alone: [8][3]u64 = undefined;
    var probe = try state.Cache.init(s.m.device, config, stream_capacity);
    defer probe.deinit();
    for (0..8) |i| {
        try copyCache(s.m, &s.base[i], &probe);
        for (0..sizes[i]) |j| {
            try batch.step(s.m, &s.bs, &.{&probe}, &.{drafts[i][j]}, s.logits[0..vocab]);
            alone[i][j] = hashed(s.row(0));
        }
    }
    const groupings = [_][]const usize{ &.{ 0, 4 }, &.{ 4, 0, 6 }, &.{ 0, 1, 2 }, &.{ 3, 2, 0 }, &.{ 5, 6, 0, 7 }, &.{ 7, 4, 6, 0 }, &.{ 4, 6, 1, 5, 7 } };
    var checked: usize = 0;
    for (groupings) |members| {
        var caches: [8]*state.Cache = undefined;
        var group_sizes: [8]usize = undefined;
        var ids: [8]u32 = undefined;
        var n: usize = 0;
        for (members, 0..) |i, k| {
            caches[k] = &s.base[i];
            group_sizes[k] = sizes[i];
            for (0..sizes[i]) |j| {
                ids[n] = drafts[i][j];
                n += 1;
            }
        }
        const g = try s.verifier.group(caches[0..members.len], group_sizes[0..members.len]);
        try s.verifier.forward(s.m, &s.bs, &g, ids[0..n]);
        @memcpy(s.logits[0 .. n * vocab], s.bs.rows.logits.slice(f32, n * vocab));
        var at: usize = 0;
        for (members) |i| for (0..sizes[i]) |j| {
            try std.testing.expectEqual(alone[i][j], hashed(s.row(at)));
            at += 1;
            checked += 1;
        };
    }
    std.debug.print("grouped rows alone and shared {s}: {d} rows in groups of 2-8 equal their one-row steps bit for bit\n", .{ path, checked });
}

test "a grouped verify row keeps its one-row step's bits whatever streams share the round" {
    var ran = false;
    for ([_][*:0]const u8{ "TF_INKLING2_LOOP_DIR", "TF_INKLING2_LOOP_OQ4E_DIR" }) |name| {
        const path = std.c.getenv(name) orelse continue;
        try checkGroupedAlone(std.mem.span(path));
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}

fn checkConcurrent(path: []const u8) !void {
    const gpa = std.testing.allocator;
    const s = try Streams.init(gpa, path);
    defer s.deinit();
    const config = &s.m.config;
    const vocab = config.text.unpadded_vocab;
    const steps = 6;
    var solo: [8][steps]u64 = undefined;
    var tokens: [8][steps]u32 = undefined;
    var work: [8]state.Cache = undefined;
    var made: usize = 0;
    defer for (work[0..made]) |*c| c.deinit();
    for (&work) |*c| {
        c.* = try state.Cache.init(s.m.device, config, stream_capacity);
        made += 1;
    }
    for (0..8) |i| {
        try copyCache(s.m, &s.base[i], &work[i]);
        var id = s.ids[i];
        for (0..steps) |j| {
            try batch.step(s.m, &s.bs, &.{&work[i]}, &.{id}, s.logits[0..vocab]);
            solo[i][j] = hashed(s.row(0));
            id = @intCast(argmax(s.row(0)));
            tokens[i][j] = id;
        }
    }
    var next: [8]u32 = s.ids;
    for (0..8) |i| try copyCache(s.m, &s.base[i], &work[i]);
    var widths_seen: [9]bool = @splat(false);
    for (0..steps + 3) |step| {
        var caches: [8]*state.Cache = undefined;
        var ids: [8]u32 = undefined;
        var which: [8]usize = undefined;
        var count: usize = 0;
        for (0..8) |i| {
            const join = i / 2;
            if (step < join or step >= join + steps) continue;
            caches[count] = &work[i];
            ids[count] = next[i];
            which[count] = i;
            count += 1;
        }
        widths_seen[count] = true;
        try batch.step(s.m, &s.bs, caches[0..count], ids[0..count], s.logits[0 .. count * vocab]);
        for (which[0..count], 0..) |i, k| {
            const j = step - i / 2;
            try std.testing.expectEqual(solo[i][j], hashed(s.row(k)));
            next[i] = @intCast(argmax(s.row(k)));
            try std.testing.expectEqual(tokens[i][j], next[i]);
        }
    }
    for ([_]usize{ 2, 4, 6, 8 }) |w| try std.testing.expect(widths_seen[w]);
    std.debug.print("concurrent replies {s}: 8 greedy streams of {d} tokens joining and leaving equal their solo replies bit for bit\n", .{ path, steps });
}

test "concurrent greedy batch-step replies equal solo replies as streams join and leave" {
    var ran = false;
    for ([_][*:0]const u8{ "TF_INKLING2_LOOP_DIR", "TF_INKLING2_LOOP_OQ4E_DIR" }) |name| {
        const path = std.c.getenv(name) orelse continue;
        try checkConcurrent(std.mem.span(path));
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}
