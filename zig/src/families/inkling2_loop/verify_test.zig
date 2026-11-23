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

const batch = @import("batch.zig");
const verify = @import("verify.zig");
const drafting = @import("draft.zig");

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
    const parsed = try std.json.parseFromSlice([]const drafting.Prompt, gpa, drafting.prompts_json, .{});
    defer parsed.deinit();
    const agent = parsed.value[0].token_ids;
    var ps = try prefill.PrefillScratch.init(m.device, config, 512);
    defer ps.deinit();
    var bs = try batch.BatchScratch.init(m.device, config, 8);
    defer bs.deinit();
    var verifier = try verify.Verifier.init(m.device, config);
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
    std.debug.print("grouped verification {s}: rows {d}, logits and caches equal batch decoding bit for bit\n", .{ path, n });
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

fn argmaxAt(values: []const f32) u32 {
    var best: usize = 0;
    for (values[1..], 1..) |value, i| if (value > values[best]) {
        best = i;
    };
    return @intCast(best);
}

/// Prefill `tokens` in chunks, feeding each chunk's taps to the stream's drafter context when it has one; the
/// prompt's next token. Both decoding paths prefill through here, so their caches start equal.
fn prefillDrafted(m: *const model.Model, ps: *prefill.PrefillScratch, cache: *state.Cache, d: *drafting.Drafter, ctx: ?*drafting.Context, tokens: []const u32) !u32 {
    var at: usize = 0;
    var rows: [512]drafting.Row = undefined;
    while (at < tokens.len) {
        const take = @min(ps.chunk, tokens.len - at);
        if (ctx != null) ps.taps = &d.taps;
        defer ps.taps = null;
        try prefill.prefillWant(m, ps, cache, tokens[at..][0..take], if (at + take == tokens.len) .last else .none);
        if (ctx) |c| {
            for (rows[0..take]) |*row| row.* = .{ .ctx = c };
            try d.extend(rows[0..take]);
        }
        at += take;
    }
    return argmaxAt(ps.logits.slice(f32, m.config.text.unpadded_vocab));
}

fn checkSpeculation(path: []const u8, width: usize, steps: usize) !void {
    const gpa = std.testing.allocator;
    const m = try model.Model.load(gpa, std.testing.io, path);
    defer m.deinit();
    try m.loadDraft(path);
    const config = &m.config;
    const vocab = config.text.unpadded_vocab;
    const parsed = try std.json.parseFromSlice([]const drafting.Prompt, gpa, drafting.prompts_json, .{});
    defer parsed.deinit();
    var ps = try prefill.PrefillScratch.init(m.device, config, 512);
    defer ps.deinit();
    var bs = try batch.BatchScratch.init(m.device, config, 8);
    defer bs.deinit();
    var verifier = try verify.Verifier.init(m.device, config);
    defer verifier.deinit();
    var d = try drafting.Drafter.init(m, &ps.pipes, 4, 512);
    defer d.deinit();
    const capacity: usize = 4096;
    var plain: [4]state.Cache = undefined;
    var spec: [4]state.Cache = undefined;
    var ctxs: [4]drafting.Context = undefined;
    var made: usize = 0;
    defer for (0..made) |i| {
        plain[i].deinit();
        spec[i].deinit();
        ctxs[i].deinit();
    };
    const want = try gpa.alloc(u32, width * (steps + 1));
    defer gpa.free(want);
    const got = try gpa.alloc(u32, width * (steps + 1));
    defer gpa.free(got);
    var have: [4]usize = @splat(0);
    var streams: [4]verify.Stream = undefined;
    for (0..width) |i| {
        plain[i] = try state.Cache.init(m.device, config, capacity);
        spec[i] = try state.Cache.init(m.device, config, capacity);
        ctxs[i] = try drafting.Context.init(m.device, config, capacity);
        made += 1;
        const prompt = parsed.value[i].token_ids;
        want[i * (steps + 1)] = try prefillDrafted(m, &ps, &plain[i], &d, null, prompt);
        const first = try prefillDrafted(m, &ps, &spec[i], &d, &ctxs[i], prompt);
        got[i * (steps + 1)] = first;
        have[i] = 1;
        streams[i] = .{ .cache = &spec[i], .ctx = &ctxs[i], .anchor = first };
    }
    const logits = try gpa.alloc(f32, width * vocab);
    defer gpa.free(logits);
    var caches: [4]*state.Cache = undefined;
    var ids: [4]u32 = undefined;
    for (0..width) |i| caches[i] = &plain[i];
    for (1..steps + 1) |step| {
        for (0..width) |i| ids[i] = want[i * (steps + 1) + step - 1];
        try batch.step(m, &bs, caches[0..width], ids[0..width], logits[0 .. width * vocab]);
        for (0..width) |i| want[i * (steps + 1) + step] = argmaxAt(logits[i * vocab ..][0..vocab]);
    }
    var speculator = verify.Speculator{ .model = m, .drafter = &d, .verifier = &verifier, .batch = &bs };
    var rounds: usize = 0;
    while (true) {
        var live: [4]*verify.Stream = undefined;
        var owners: [4]usize = undefined;
        var count: usize = 0;
        for (0..width) |i| if (have[i] < steps + 1) {
            streams[i].limit = steps + 1 - have[i] - 1;
            live[count] = &streams[i];
            owners[count] = i;
            count += 1;
        };
        if (count == 0) break;
        try speculator.round(live[0..count]);
        rounds += 1;
        for (live[0..count], owners[0..count]) |st, i| {
            const n = @min(st.emitted_len, steps + 1 - have[i]);
            @memcpy(got[i * (steps + 1) + have[i] ..][0..n], st.emitted[0..n]);
            have[i] += n;
            st.anchor = st.emitted[st.emitted_len - 1];
        }
    }
    var offered: u64 = 0;
    var taken: u64 = 0;
    for (streams[0..width]) |st| {
        offered += st.offered;
        taken += st.taken;
    }
    try std.testing.expectEqualSlices(u32, want, got);
    try std.testing.expect(offered > 0);
    std.debug.print("speculation {s} width={d}: {d} tokens a stream equal plain greedy, {d} rounds, {d} of {d} drafts accepted\n", .{ path, width, steps + 1, rounds, taken, offered });
}

test "speculative decoding emits plain greedy decode's tokens on real prompts at widths one and four" {
    var ran = false;
    for ([_][*:0]const u8{ "TF_INKLING2_LOOP_DIR", "TF_INKLING2_LOOP_OQ4E_DIR" }) |name| {
        const path = std.c.getenv(name) orelse continue;
        try checkSpeculation(std.mem.span(path), 1, 128);
        try checkSpeculation(std.mem.span(path), 4, 64);
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}
