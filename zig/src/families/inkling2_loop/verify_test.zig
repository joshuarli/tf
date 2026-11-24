const std = @import("std");
const mtl = @import("metal");
const model = @import("model.zig");
const state = @import("state.zig");
const prefill = @import("prefill.zig");
const Verification = @import("verify.zig").Verification;

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

fn reconstructReference(v: *Verification, accepted: usize) !void {
    const m = v.model;
    const config = &m.config;
    const snapshot = try readback(m, v.snapshot);
    defer snapshot.deinit();
    const capture = try readback(m, v.conv.buffer);
    defer capture.deinit();
    const saved = snapshot.slice(u8, snapshot.length());
    const inputs = capture.slice(u8, capture.length());
    const rings = [_]mtl.Buffer{ v.cache.k_conv, v.cache.v_conv, v.cache.attn_conv, v.cache.mlp_conv };
    const sites = [_]prefill.ConvSite{ .key, .value, .attention, .mlp };
    var ring_base: usize = 0;
    for (rings, sites) |buf, site| {
        const width = v.conv.width(site);
        for (0..config.slots()) |slot| {
            for (accepted - @min(accepted, state.ring)..accepted) |row| {
                const dst = ring_base + v.cache.convAt(slot, width) + ((v.start + row) % state.ring) * width * 4;
                const src = v.conv.offset(slot, site, row);
                @memcpy(saved[dst..][0 .. width * 4], inputs[src..][0 .. width * 4]);
            }
        }
        ring_base += buf.length();
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    var at: usize = 0;
    for (rings) |buf| {
        blit.copy(snapshot, at, buf, 0, buf.length());
        at += buf.length();
    }
    const row_bytes: usize = config.text.local.kv * config.text.local.dim * 2;
    for (0..config.slots()) |slot| {
        if (!state.Cache.isLocalSlot(config, slot)) continue;
        for (0..v.rows) |row| {
            for ([_]mtl.Buffer{ v.cache.keys, v.cache.values }) |buf| {
                if (row >= accepted) blit.copy(snapshot, at, buf, v.cache.row(config, slot, v.start + row), row_bytes);
                at += row_bytes;
            }
        }
    }
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    v.cache.len = v.start + accepted;
}

fn expectCacheEqual(m: *const model.Model, a: *state.Cache, b: *state.Cache, start: usize, rows: usize, accepted: usize) !void {
    const config = &m.config;
    try std.testing.expectEqual(a.len, b.len);
    for ([_]mtl.Buffer{ a.k_conv, a.v_conv, a.attn_conv, a.mlp_conv }, [_]mtl.Buffer{ b.k_conv, b.v_conv, b.attn_conv, b.mlp_conv }) |lhs, rhs| {
        const x = try readback(m, lhs);
        defer x.deinit();
        const y = try readback(m, rhs);
        defer y.deinit();
        try std.testing.expectEqualSlices(u8, x.slice(u8, x.length()), y.slice(u8, y.length()));
    }
    const row_bytes: usize = config.text.local.kv * config.text.local.dim * 2;
    const bytes = config.slots() * rows * row_bytes * 2;
    const x = try m.device.buffer(bytes, mtl.ResourceOptions.shared);
    defer x.deinit();
    const y = try m.device.buffer(bytes, mtl.ResourceOptions.shared);
    defer y.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    var at: usize = 0;
    for (0..config.slots()) |slot| {
        for (0..rows) |row| {
            if (row >= accepted and !state.Cache.isLocalSlot(config, slot)) continue;
            for ([_]mtl.Buffer{ a.keys, a.values }, [_]mtl.Buffer{ b.keys, b.values }) |lhs, rhs| {
                blit.copy(lhs, a.row(config, slot, start + row), x, at, row_bytes);
                blit.copy(rhs, b.row(config, slot, start + row), y, at, row_bytes);
                at += row_bytes;
            }
        }
    }
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    try std.testing.expectEqualSlices(u8, x.slice(u8, at), y.slice(u8, at));
}

test "verification commits the same width prefix across cache boundaries" {
    const path = std.c.getenv("TF_INKLING2_LOOP_DIR") orelse return error.SkipZigTest;
    const m = try model.Model.load(std.testing.allocator, std.testing.io, std.mem.span(path));
    defer m.deinit();
    const config = &m.config;
    const base: usize = config.text.window - 1;
    var a = try state.Cache.init(m.device, config, base + 40);
    defer a.deinit();
    var b = try state.Cache.init(m.device, config, base + 40);
    defer b.deinit();
    var sa = try prefill.PrefillScratch.init(m.device, config, 512);
    defer sa.deinit();
    var sb = try prefill.PrefillScratch.init(m.device, config, 512);
    defer sb.deinit();
    var tokens: [512]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(100 + i % 71);
    var at: usize = 0;
    while (at < base) {
        const take = @min(tokens.len, base - at);
        try prefill.prefillWant(m, &sa, &a, tokens[0..take], .none);
        try prefill.prefillWant(m, &sb, &b, tokens[0..take], .none);
        at += take;
    }
    const draft = [_]u32{ 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116 };
    var anchors: [32]u32 = undefined;
    for (&anchors, 0..) |*token, i| token.* = @intCast(101 + i);
    var anchor_predictions: [32]u32 = undefined;
    var initial_a = try Verification.begin(m, &sa, &a, &anchors, &anchor_predictions);
    var initial_b = try Verification.begin(m, &sb, &b, &anchors, &anchor_predictions);
    var replay_best: [2]u32 = undefined;
    for ([_]usize{ 1, 2, 4, 8, 16 }) |width| for ([_]usize{ 0, 1, 2, 3, 4, 5, 7, 8, 15, 16 }) |accepted| {
        if (accepted > width) continue;
        try initial_a.reset();
        try initial_b.reset();
        var actual_best: [draft.len]u32 = undefined;
        var reference_best: [draft.len]u32 = undefined;
        var actual = try Verification.begin(m, &sa, &a, draft[0..width], actual_best[0..width]);
        var reference = try Verification.begin(m, &sb, &b, draft[0..width], reference_best[0..width]);
        try std.testing.expectEqualSlices(u32, actual_best[0..width], reference_best[0..width]);
        if (width == 16 and accepted == 2) @memcpy(&replay_best, actual_best[0..2]);
        try actual.finish(draft[0..width], accepted);
        try reconstructReference(&reference, accepted);
        try reference.finish(draft[0..width], width);
        try expectCacheEqual(m, &a, &b, base, width, accepted);
        const continuation = [_]u32{ 109, 110, 111, 112, 113 };
        try prefill.prefillWant(m, &sa, &a, &continuation, .all);
        try prefill.prefillWant(m, &sb, &b, &continuation, .all);
        const n = continuation.len * config.text.unpadded_vocab;
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(sa.logits.slice(f32, n)), std.mem.sliceAsBytes(sb.logits.slice(f32, n)));
        std.debug.print("verify same-width commit width={d} accepted={d} exact\n", .{ width, accepted });
    };
    try initial_b.reset();
    try prefill.prefillWant(m, &sb, &b, draft[0..2], .all);
    const vocab: usize = config.text.unpadded_vocab;
    const logits = sb.logits.slice(f32, 2 * vocab);
    for (0..2) |row| {
        const values = logits[row * vocab ..][0..vocab];
        var best: usize = 0;
        for (values[1..], 1..) |value, i| if (value > values[best]) { best = i; };
        try std.testing.expectEqual(@as(u32, @intCast(best)), replay_best[row]);
    }
    try initial_a.finish(&anchors, 0);
    try initial_b.finish(&anchors, 0);
}

const vfy = @import("verify.zig");

fn expectParentsFirst(t: *const vfy.Tree) !void {
    for (t.shape(), 1..) |p, r| {
        try std.testing.expect(p < r);
        try std.testing.expectEqual(t.depth[p] + 1, t.depth[r]);
    }
}

test "a tree plan merges the lookup chain into the drafter's tree where tokens agree, parents before children" {
    var plan = vfy.Plan.init();
    plan.verify_ms = @splat(1);
    const drafted = vfy.Drafted{ .tokens = &.{ 10, 20, 11, 30 }, .parents = &.{ -1, -1, 0, 2 }, .scores = &.{ -0.05, -2.0, -0.1, -0.2 } };
    const t = plan.build(7, drafted, &.{ 10, 11, 40, 41 }, 20, 16);
    try std.testing.expectEqual(@as(usize, 7), t.rows);
    try expectParentsFirst(&t);
    try std.testing.expectEqual(@as(u32, 7), t.tokens[0]);
    try std.testing.expectEqual(@as(usize, 4), t.looked());
    try std.testing.expectEqual(@as(u32, 20), t.match);
    var seen: [6]bool = @splat(false);
    for (1..t.rows) |r| {
        const at = std.mem.indexOfScalar(u32, &.{ 10, 20, 11, 30, 40, 41 }, t.tokens[r]).?;
        seen[at] = true;
        const chained = t.chain[r] != vfy.Plan.unlooked;
        try std.testing.expectEqual(at != 1 and at != 3, chained);
        try std.testing.expectEqual(at < 4, t.q[r] > 0);
    }
    for (seen) |s| try std.testing.expect(s);
    const again = plan.build(7, drafted, &.{ 10, 11, 40, 41 }, 20, 16);
    try std.testing.expectEqualSlices(u32, t.tokens[0..t.rows], again.tokens[0..again.rows]);
    try std.testing.expectEqualSlices(u8, t.shape(), again.shape());
}

test "a plan verifies few rows for unlikely drafts and all of a long likely chain past the drafter's depth" {
    var plan = vfy.Plan.init();
    const weak = vfy.Drafted{ .tokens = &.{ 1, 2, 3, 4, 5, 6, 7 }, .parents = &.{ -1, 0, 1, 2, 3, 4, 5 }, .scores = &.{ -2.5, -5, -7.5, -10, -12.5, -15, -17.5 } };
    try std.testing.expect(plan.build(9, weak, &.{}, 0, 16).rows <= 2);
    var chain: [15]u32 = undefined;
    for (&chain, 0..) |*c, i| c.* = @intCast(100 + i);
    const strong = plan.build(9, null, &chain, 40, 16);
    try std.testing.expectEqual(@as(usize, 16), strong.rows);
    try std.testing.expectEqual(@as(u8, 15), strong.depth[15]);
    try expectParentsFirst(&strong);
    try std.testing.expectEqual(@as(usize, 6), plan.build(9, null, &chain, 40, 6).rows);
    plan.most = 8;
    try std.testing.expectEqual(@as(usize, 8), plan.build(9, null, &chain, 40, 16).rows);
}

test "a plan learns from each draft under an accepted parent and leaves the drafter out only behind a long likely chain" {
    var plan = vfy.Plan.init();
    try std.testing.expect(plan.lookupSuffices(15, 40));
    try std.testing.expect(!plan.lookupSuffices(7, 40));
    try std.testing.expect(!plan.lookupSuffices(15, 3));
    const before = plan.estimate(0, 0, 0, 3, false);
    var chain: [15]u32 = undefined;
    for (&chain, 0..) |*c, i| c.* = @intCast(100 + i);
    const t = plan.build(9, null, chain[0..4], 3, 16);
    var picks: [16]u32 = @splat(0);
    for (1..t.rows) |r| picks[t.shape()[r - 1]] = t.tokens[r];
    for (0..20) |_| plan.learn(&t, &picks);
    try std.testing.expect(plan.estimate(0, 0, 0, 3, false) > before);
    for (&picks) |*p| p.* = 1;
    for (0..60) |_| plan.learn(&t, &picks);
    try std.testing.expect(plan.estimate(0, 0, 0, 3, false) < before);
    const fresh = vfy.Plan.init();
    try std.testing.expectEqual(fresh.estimate(0, 0, 0, 3, true), plan.estimate(0, 0, 0, 3, true));
    plan.reset();
    try std.testing.expectEqual(before, plan.estimate(0, 0, 0, 3, false));
    plan.skip_at = 0;
    try std.testing.expect(!plan.lookupSuffices(15, 40));
}
