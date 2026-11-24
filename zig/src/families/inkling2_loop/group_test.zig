const std = @import("std");
const mtl = @import("metal");
const model = @import("model.zig");
const state = @import("state.zig");
const prefill = @import("prefill.zig");
const batch = @import("batch.zig");
const group = @import("group.zig");
const Taps = @import("taps.zig").Taps;

const Prompt = struct { name: []const u8, token_ids: []const u32 };
const prompts_json = @embedFile("fixtures/prompts.json");

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

fn cacheBuffers(c: *state.Cache) [6]mtl.Buffer {
    return .{ c.keys, c.values, c.k_conv, c.v_conv, c.attn_conv, c.mlp_conv };
}

fn cloneCache(m: *const model.Model, src: *state.Cache, dst: *state.Cache) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    for (cacheBuffers(src), cacheBuffers(dst)) |x, y| blit.copy(x, 0, y, 0, x.length());
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    dst.len = src.len;
}

fn expectSameCache(m: *const model.Model, a: *state.Cache, b: *state.Cache) !void {
    try std.testing.expectEqual(a.len, b.len);
    for (cacheBuffers(a), cacheBuffers(b)) |x, y| {
        const xs = try readback(m, x);
        defer xs.deinit();
        const ys = try readback(m, y);
        defer ys.deinit();
        try std.testing.expectEqualSlices(u8, xs.slice(u8, xs.length()), ys.slice(u8, ys.length()));
    }
}

fn prefillTokens(m: *const model.Model, ps: *prefill.PrefillScratch, cache: *state.Cache, tokens: []const u32) !void {
    var at: usize = 0;
    while (at < tokens.len) {
        const take = @min(ps.chunk, tokens.len - at);
        try prefill.prefillWant(m, ps, cache, tokens[at..][0..take], .none);
        at += take;
    }
}

fn checkGrouped(path: []const u8) !void {
    const gpa = std.testing.allocator;
    const m = try model.Model.load(gpa, std.testing.io, path);
    defer m.deinit();
    const config = &m.config;
    const vocab = config.text.unpadded_vocab;
    const parsed = try std.json.parseFromSlice([]const Prompt, gpa, prompts_json, .{});
    defer parsed.deinit();
    const agent = parsed.value[0].token_ids;
    var ps = try prefill.PrefillScratch.init(m.device, config, 512);
    defer ps.deinit();
    var bs = try batch.BatchScratch.init(m.device, config, 8);
    defer bs.deinit();
    var verifier = try group.Verifier.init(m.device, config);
    defer verifier.deinit();
    const capacity: usize = config.text.window + 64;
    const starts = [_]usize{ config.text.window - 2, 37, 1 };
    const prompts = [_][]const u32{ agent[0..starts[0]], parsed.value[1].token_ids, parsed.value[2].token_ids[0..1] };
    var grouped: [3]state.Cache = undefined;
    var serial: [3]state.Cache = undefined;
    var accepted_only: [3]state.Cache = undefined;
    var made: usize = 0;
    defer for (0..made) |i| {
        grouped[i].deinit();
        serial[i].deinit();
        accepted_only[i].deinit();
    };
    for (0..3) |i| {
        grouped[i] = try state.Cache.init(m.device, config, capacity);
        serial[i] = try state.Cache.init(m.device, config, capacity);
        accepted_only[i] = try state.Cache.init(m.device, config, capacity);
        made += 1;
        try prefillTokens(m, &ps, &grouped[i], prompts[i]);
        try cloneCache(m, &grouped[i], &serial[i]);
        try cloneCache(m, &grouped[i], &accepted_only[i]);
    }
    const rows = [_][]const u32{ agent[starts[0] .. starts[0] + 4], &.{ 3711, 149878, 152743 }, &.{4896} };
    const sizes = [_]usize{ 4, 3, 1 };
    var ids: [8]u32 = undefined;
    var n: usize = 0;
    for (rows) |r| {
        @memcpy(ids[n..][0..r.len], r);
        n += r.len;
    }
    const caches = [_]*state.Cache{ &grouped[0], &grouped[1], &grouped[2] };
    const g = try verifier.group(&caches, &sizes);
    try verifier.forward(m, &bs, &g, ids[0..n]);
    const got = try gpa.dupe(f32, bs.rows.logits.slice(f32, n * vocab));
    defer gpa.free(got);
    const picked = try gpa.dupe(u32, try verifier.picks(m, bs.rows.logits, n));
    defer gpa.free(picked);
    for (0..3) |i| try std.testing.expectEqual(starts[i], grouped[i].len);
    const logits = try gpa.alloc(f32, 3 * vocab);
    defer gpa.free(logits);
    const first_row = [_]usize{ 0, 4, 7 };
    for (0..4) |step| {
        var live: [3]*state.Cache = undefined;
        var step_ids: [3]u32 = undefined;
        var which: [3]usize = undefined;
        var count: usize = 0;
        for (0..3) |i| if (step < sizes[i]) {
            live[count] = &serial[i];
            step_ids[count] = rows[i][step];
            which[count] = i;
            count += 1;
        };
        try batch.step(m, &bs, live[0..count], step_ids[0..count], logits[0 .. count * vocab]);
        for (0..count) |k| {
            const row = first_row[which[k]] + step;
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits[k * vocab ..][0..vocab]), std.mem.sliceAsBytes(got[row * vocab ..][0..vocab]));
            try std.testing.expectEqual(argmaxAt(logits[k * vocab ..][0..vocab]), picked[row]);
        }
    }
    const accepted = [_]usize{ 2, 1, 0 };
    try verifier.commit(m, &g, &accepted);
    for (0..3) |step| {
        var live: [3]*state.Cache = undefined;
        var step_ids: [3]u32 = undefined;
        var count: usize = 0;
        for (0..3) |i| if (step <= accepted[i]) {
            live[count] = &accepted_only[i];
            step_ids[count] = rows[i][step];
            count += 1;
        };
        try batch.step(m, &bs, live[0..count], step_ids[0..count], logits[0 .. count * vocab]);
    }
    for (0..3) |i| try expectSameCache(m, &grouped[i], &accepted_only[i]);
    const next = [_]u32{ 6, 1101, 2024 };
    try batch.step(m, &bs, &.{ &grouped[0], &grouped[1], &grouped[2] }, &next, logits);
    const expected = try gpa.dupe(f32, logits);
    defer gpa.free(expected);
    try batch.step(m, &bs, &.{ &accepted_only[0], &accepted_only[1], &accepted_only[2] }, &next, logits);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(logits));
    const singles = [_]usize{ 1, 1, 1 };
    const g1 = try verifier.group(&caches, &singles);
    try verifier.forward(m, &bs, &g1, &next);
    const single_logits = try gpa.dupe(f32, bs.rows.logits.slice(f32, 3 * vocab));
    defer gpa.free(single_logits);
    try verifier.commit(m, &g1, &.{ 0, 0, 0 });
    try batch.step(m, &bs, &.{ &accepted_only[0], &accepted_only[1], &accepted_only[2] }, &next, logits);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(single_logits));
    for (0..3) |i| try expectSameCache(m, &grouped[i], &accepted_only[i]);
    const full_rows = [_][]const u32{ &.{ 40, 41, 42 }, &.{ 43, 44 }, &.{ 45, 46, 47 } };
    const full_sizes = [_]usize{ 3, 2, 3 };
    n = 0;
    for (full_rows) |r| {
        @memcpy(ids[n..][0..r.len], r);
        n += r.len;
    }
    const g2 = try verifier.group(&caches, &full_sizes);
    try verifier.forward(m, &bs, &g2, ids[0..n]);
    try verifier.commit(m, &g2, &.{ 2, 1, 2 });
    for (0..3) |step| {
        var live: [3]*state.Cache = undefined;
        var step_ids: [3]u32 = undefined;
        var count: usize = 0;
        for (0..3) |i| if (step < full_sizes[i]) {
            live[count] = &accepted_only[i];
            step_ids[count] = full_rows[i][step];
            count += 1;
        };
        try batch.step(m, &bs, live[0..count], step_ids[0..count], logits[0 .. count * vocab]);
    }
    for (0..3) |i| try expectSameCache(m, &grouped[i], &accepted_only[i]);
    std.debug.print("grouped verification {s}: rows {d}, logits, picks and caches equal batch decoding bit for bit\n", .{ path, n });
}

fn argmaxAt(values: []const f32) u32 {
    var best: usize = 0;
    for (values[1..], 1..) |value, i| if (value > values[best]) {
        best = i;
    };
    return @intCast(best);
}

test "grouped verification equals batch decoding bit for bit and commits the same caches" {
    var ran = false;
    for ([_][*:0]const u8{ "TF_INKLING2_LOOP_DIR", "TF_INKLING2_LOOP_OQ4E_DIR" }) |name| {
        const path = std.c.getenv(name) orelse continue;
        try checkGrouped(std.mem.span(path));
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}

fn checkTaps(path: []const u8) !void {
    const gpa = std.testing.allocator;
    const m = try model.Model.load(gpa, std.testing.io, path);
    defer m.deinit();
    const config = &m.config;
    const parsed = try std.json.parseFromSlice([]const Prompt, gpa, prompts_json, .{});
    defer parsed.deinit();
    const agent = parsed.value[0].token_ids;
    var ps = try prefill.PrefillScratch.init(m.device, config, 512);
    defer ps.deinit();
    var bs = try batch.BatchScratch.init(m.device, config, 8);
    defer bs.deinit();
    var verifier = try group.Verifier.init(m.device, config);
    defer verifier.deinit();
    const visits = [_]u32{ 5, 23, 41, config.slots() };
    var grouped_taps = try Taps.init(m.device, config, &visits, 512);
    defer grouped_taps.deinit();
    var step_taps = try Taps.init(m.device, config, &visits, 8);
    defer step_taps.deinit();
    const start = config.text.window - 3;
    var cache = try state.Cache.init(m.device, config, config.text.window + 64);
    defer cache.deinit();
    var plain = try state.Cache.init(m.device, config, config.text.window + 64);
    defer plain.deinit();
    ps.taps = &grouped_taps;
    try prefillTokens(m, &ps, &cache, agent[0..start]);
    ps.taps = null;
    const tail = grouped_taps.bytes((start - 1) % 512, 1);
    try std.testing.expect(!std.mem.allEqual(u8, tail, 0));
    try cloneCache(m, &cache, &plain);
    const rows = agent[start .. start + 8];
    const g = try verifier.group(&.{&cache}, &.{8});
    bs.rows.taps = &grouped_taps;
    try verifier.forward(m, &bs, &g, rows);
    bs.rows.taps = &step_taps;
    defer bs.rows.taps = null;
    const logits = try gpa.alloc(f32, config.text.unpadded_vocab);
    defer gpa.free(logits);
    for (rows, 0..) |id, j| {
        try batch.step(m, &bs, &.{&plain}, &.{id}, logits);
        try std.testing.expectEqualSlices(u8, step_taps.bytes(0, 1), grouped_taps.bytes(j, 1));
    }
    std.debug.print("grouped taps {s}: 8 rows from position {d} equal one-row batch steps' taps bit for bit\n", .{ path, start });
}

test "a grouped row's taps equal the batch step's taps at its position, across the local ring's wrap" {
    var ran = false;
    for ([_][*:0]const u8{ "TF_INKLING2_LOOP_DIR", "TF_INKLING2_LOOP_OQ4E_DIR" }) |name| {
        const path = std.c.getenv(name) orelse continue;
        try checkTaps(std.mem.span(path));
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}

fn checkFused(path: []const u8) !void {
    const gpa = std.testing.allocator;
    const m = try model.Model.load(gpa, std.testing.io, path);
    defer m.deinit();
    const config = &m.config;
    const parsed = try std.json.parseFromSlice([]const Prompt, gpa, prompts_json, .{});
    defer parsed.deinit();
    const agent = parsed.value[0].token_ids;
    var ps = try prefill.PrefillScratch.init(m.device, config, 512);
    defer ps.deinit();
    var bs = try batch.BatchScratch.init(m.device, config, 8);
    defer bs.deinit();
    var verifier = try group.Verifier.init(m.device, config);
    defer verifier.deinit();
    const vocab = config.text.unpadded_vocab;
    const logits = try gpa.alloc(f32, vocab);
    defer gpa.free(logits);
    const start = config.text.window - 3;
    var split: [3]state.Cache = undefined;
    for (&split) |*c| c.* = try state.Cache.init(m.device, config, config.text.window + 64);
    defer for (&split) |*c| c.deinit();
    try prefillTokens(m, &ps, &split[2], agent[0..start]);
    var ids: [8]u32 = undefined;
    ids[0] = agent[start];
    var probe = try state.Cache.init(m.device, config, config.text.window + 64);
    defer probe.deinit();
    try cloneCache(m, &split[2], &probe);
    for (1..4) |j| {
        try batch.step(m, &bs, &.{&probe}, &.{ids[j - 1]}, logits);
        ids[j] = argmaxAt(logits);
    }
    for (4..8) |j| ids[j] = @intCast(1000 + j);
    const wrong = [_]u32{ ids[0], 1001, ids[2], ids[3] };
    const cases = [_][]const u32{ ids[0..8], ids[0..5], ids[0..2], &wrong };
    var seen: [3]bool = @splat(false);
    for (cases) |rows| {
        const width = rows.len;
        try cloneCache(m, &split[2], &split[0]);
        try cloneCache(m, &split[2], &split[1]);
        const g0 = try verifier.group(&.{&split[0]}, &.{width});
        try verifier.forward(m, &bs, &g0, rows);
        const picks = try gpa.dupe(u32, try verifier.picks(m, bs.rows.logits, width));
        defer gpa.free(picks);
        const kept = group.acceptedDrafts(rows[1..], picks);
        seen[if (kept == 0) 0 else if (kept == width - 1) 2 else 1] = true;
        try verifier.commit(m, &g0, &.{kept});
        const g1 = try verifier.group(&.{&split[1]}, &.{width});
        const decided = try verifier.decide(m, &bs, &g1, rows, .commit);
        try std.testing.expectEqualSlices(u32, picks, decided.picks);
        try std.testing.expectEqual(kept, decided.accepted[0]);
        try expectSameCache(m, &split[0], &split[1]);
    }
    try std.testing.expect(seen[0] and seen[1] and seen[2]);
    std.debug.print("fused verify rounds {s}: zero, partial and full acceptance pick, accept and commit as forward, picks and commit\n", .{path});
}

test "a fused verify round picks, accepts and commits what forward, picks and commit do" {
    var ran = false;
    for ([_][*:0]const u8{ "TF_INKLING2_LOOP_DIR", "TF_INKLING2_LOOP_OQ4E_DIR" }) |name| {
        const path = std.c.getenv(name) orelse continue;
        try checkFused(std.mem.span(path));
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}
