const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const cfg = tf.qwen27.dflash.config;
const dflash = @import("dflash.zig");
const spec = @import("dflash_spec.zig");

const Prompt = struct { name: []const u8, token_ids: []const u32 };
const prompts_json = @embedFile("fixtures/prompts.json");
const draft_config = @embedFile("fixtures/dflash2_config.json");

fn target(text: []const u8) !inkling.config.Config {
    var why: inkling.config.Why = .{};
    return inkling.config.parse(std.testing.allocator, text, &why);
}

test "Inkling DFlash2 binds the oQ4e target and refuses another target or config by name" {
    const a = std.testing.allocator;
    const t = try target(@embedFile("fixtures/oq4e-config.json"));
    const c = try dflash.bind(try cfg.parse(a, draft_config), &t);
    try std.testing.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, c.tapIds());
    try std.testing.expect(c.final_norm);
    try std.testing.expectEqual(@as(f32, 30), c.softcap);
    try std.testing.expectEqual(@as(u32, 200058), c.candidateVocab());
    try std.testing.expectEqual(@as(u32, 48), t.slots());
    const native = try target(@embedFile("fixtures/config.json"));
    try std.testing.expectError(error.InklingDFlash2NeedsAffineHead, dflash.bind(try cfg.parse(a, draft_config), &native));
    var capped = t;
    capped.text.logit_cap = 25;
    try std.testing.expectError(error.InklingDFlash2Softcap, dflash.bind(try cfg.parse(a, draft_config), &capped));
    var wider = t;
    wider.text.hidden = 2048;
    try std.testing.expectError(error.DraftForAnotherTarget, dflash.bind(try cfg.parse(a, draft_config), &wider));
    const edits = [_]struct { from: []const u8, to: []const u8, err: anyerror }{
        .{ .from = "\"after_correction\"", .to = "\"before_correction\"", .err = error.UnsupportedSelectorSoftcap },
        .{ .from = "      48\n    ]", .to = "      47\n    ]", .err = error.NotStockDFlash2 },
        .{ .from = "\"num_target_layers\": 48", .to = "\"num_target_layers\": 49", .err = error.NotStockDFlash2 },
        .{ .from = "\"hidden_size\": 2304", .to = "\"hidden_size\": 2048", .err = error.NotStockDFlash2 },
        .{ .from = "\"block_size\": 8", .to = "\"block_size\": 16", .err = error.NotStockDFlash2 },
        .{ .from = "\"is_causal\": false", .to = "\"is_causal\": true", .err = error.BadDraftConfig },
    };
    for (edits) |e| {
        const text = try std.mem.replaceOwned(u8, a, draft_config, e.from, e.to);
        defer a.free(text);
        try std.testing.expect(!std.mem.eql(u8, text, draft_config));
        try std.testing.expectError(e.err, cfg.parse(a, text));
    }
}

const Loaded = struct { model: *inkling.Model, drafter: *dflash.Drafter };

fn load(options: dflash.Drafter.Options) !?Loaded {
    const target_dir = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return null;
    const draft_dir = std.c.getenv("TF_INKLING2_DFLASH2_DIR") orelse return null;
    const m = try inkling.Model.load(std.testing.allocator, std.testing.io, std.mem.span(target_dir));
    errdefer m.deinit();
    const d = try dflash.Drafter.init(std.testing.allocator, std.testing.io, m, std.mem.span(draft_dir), options);
    return .{ .model = m, .drafter = d };
}

const Tally = struct {
    rounds: usize = 0,
    zero: usize = 0,
    partial: usize = 0,
    full: usize = 0,
    /// Rounds that offered fewer than a block's drafts: the output limit falls inside their block.
    short: usize = 0,
    /// Rounds whose accepted tokens an EOS cut short.
    cut: usize = 0,

    fn add(t: *Tally, rounds: []const spec.Round) void {
        for (rounds) |r| {
            t.rounds += 1;
            if (r.offered < dflash.depth) t.short += 1;
            if (r.emitted < r.accepted + 1) t.cut += 1;
            if (r.offered == 0) continue;
            if (r.accepted == 0) t.zero += 1 else if (r.accepted == r.offered) t.full += 1 else t.partial += 1;
        }
    }
};

/// Plain greedy and speculative decoding of `ids` from fresh caches: their tokens, which must match, and the rounds.
fn compare(s: *spec.Speculator, ids: []const u32, limit: usize, eos: []const u32, tally: *Tally, caches_equal: bool) !usize {
    const a = std.testing.allocator;
    const m = s.model;
    var plain_cache = try inkling.Cache.init(m.device, &m.config, ids.len + limit + 8);
    defer plain_cache.deinit();
    var spec_cache = try inkling.Cache.init(m.device, &m.config, ids.len + limit + 8);
    defer spec_cache.deinit();
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(a);
    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(a);
    var rounds: std.ArrayList(spec.Round) = .empty;
    defer rounds.deinit(a);
    const first = try spec.prefillPlain(m, &s.prefill, &plain_cache, ids);
    try want.append(a, first);
    try spec.plain(m, &s.batch, &plain_cache, first, limit, eos, a, &want, null);
    const started = try s.start(&spec_cache, ids);
    try std.testing.expectEqual(first, started.first);
    try got.append(a, started.first);
    try s.run(&spec_cache, started.first, limit, eos, a, &got, &rounds);
    try std.testing.expectEqualSlices(u32, want.items, got.items);
    try std.testing.expectEqual(spec_cache.len, s.drafter.committed());
    if (caches_equal) try expectSameCache(m, &plain_cache, &spec_cache);
    tally.add(rounds.items);
    return want.items.len;
}

fn cacheBuffers(c: *inkling.Cache) [6]mtl.Buffer {
    return .{ c.keys, c.values, c.k_conv, c.v_conv, c.attn_conv, c.mlp_conv };
}

fn expectSameCache(m: *const inkling.Model, x: *inkling.Cache, y: *inkling.Cache) !void {
    try std.testing.expectEqual(x.len, y.len);
    for (cacheBuffers(x), cacheBuffers(y)) |p, q| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const ps = try m.device.buffer(p.length(), mtl.ResourceOptions.shared);
        defer ps.deinit();
        const qs = try m.device.buffer(q.length(), mtl.ResourceOptions.shared);
        defer qs.deinit();
        const cb = m.queue.commandBuffer();
        const blit = cb.blit();
        blit.copy(p, 0, ps, 0, p.length());
        blit.copy(q, 0, qs, 0, q.length());
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        try std.testing.expectEqualSlices(u8, ps.slice(u8, ps.length()), qs.slice(u8, qs.length()));
    }
}

test "Inkling DFlash2 greedy speculation emits plain greedy decoding's tokens on real prompts" {
    const loaded = try load(.{ .bits = 4 }) orelse return error.SkipZigTest;
    defer loaded.model.deinit();
    defer loaded.drafter.deinit();
    try std.testing.expect(try loaded.drafter.fixedBytes() <= 4 << 30);
    var s = try spec.Speculator.init(loaded.model, loaded.drafter, dflash.tap_ids);
    defer s.deinit();
    const parsed = try std.json.parseFromSlice([]const Prompt, std.testing.allocator, prompts_json, .{});
    defer parsed.deinit();
    var tally: Tally = .{};
    var tokens: usize = 0;
    for (parsed.value) |p| tokens += try compare(&s, p.token_ids, 256, &.{}, &tally, false);
    std.debug.print("dflash2 speculation: {d} prompts, {d} tokens equal plain greedy; {d} rounds, {d} zero, {d} partial, {d} full\n", .{ parsed.value.len, tokens, tally.rounds, tally.zero, tally.partial, tally.full });
    try std.testing.expect(tally.partial > 0 and tally.full > 0);
}

/// Drafts from a known continuation: `wrong` replaces the token at `wrong` (0-based) in every chain with another.
const Oracle = struct {
    tokens: []const u32,
    prompt: usize,
    wrong: ?usize = null,
    vocab: u32,

    fn call(ptr: *anyopaque, pending: u32, position: u64, out: []u32) anyerror!void {
        const o: *Oracle = @ptrCast(@alignCast(ptr));
        const at: usize = @intCast(position - o.prompt);
        if (o.tokens[at] != pending) return error.OraclePosition;
        for (out, 0..) |*t, j| {
            t.* = if (at + 1 + j < o.tokens.len) o.tokens[at + 1 + j] else 0;
            if (o.wrong) |w| if (j >= w) {
                t.* = (t.* + 1) % o.vocab;
            };
        }
    }
};

test "forced drafts accept all, none or a prefix, stop at a limit or an EOS inside a block, and keep plain's caches" {
    const loaded = try load(.{}) orelse return error.SkipZigTest;
    defer loaded.model.deinit();
    defer loaded.drafter.deinit();
    const a = std.testing.allocator;
    const m = loaded.model;
    var s = try spec.Speculator.init(m, loaded.drafter, dflash.tap_ids);
    defer s.deinit();
    const parsed = try std.json.parseFromSlice([]const Prompt, a, prompts_json, .{});
    defer parsed.deinit();
    const ids = parsed.value[1].token_ids;
    var cache = try inkling.Cache.init(m.device, &m.config, ids.len + 160);
    defer cache.deinit();
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(a);
    try want.append(a, try spec.prefillPlain(m, &s.prefill, &cache, ids));
    try spec.plain(m, &s.batch, &cache, want.items[0], 150, &.{}, a, &want, null);
    var oracle = Oracle{ .tokens = want.items, .prompt = ids.len, .vocab = m.config.text.unpadded_vocab };
    s.proposer = .{ .ptr = &oracle, .call = Oracle.call };
    var tally: Tally = .{};
    _ = try compare(&s, ids, 1 + 8 * 16, &.{}, &tally, true);
    try std.testing.expectEqual(@as(usize, 16), tally.full);
    try std.testing.expectEqual(@as(usize, 16), tally.rounds);
    tally = .{};
    _ = try compare(&s, ids, 1 + 8 * 9 + 3, &.{}, &tally, false);
    try std.testing.expectEqual(@as(usize, 1), tally.short);
    try std.testing.expectEqual(@as(usize, 10), tally.full);
    oracle.wrong = 0;
    tally = .{};
    _ = try compare(&s, ids, 40, &.{}, &tally, true);
    try std.testing.expectEqual(tally.rounds, tally.zero + @as(usize, 1));
    oracle.wrong = 3;
    tally = .{};
    _ = try compare(&s, ids, 1 + 4 * 21, &.{}, &tally, true);
    try std.testing.expectEqual(@as(usize, 20), tally.partial);
    try std.testing.expectEqual(@as(usize, 1), tally.full);
    oracle.wrong = null;
    var stop: usize = 13;
    while (stop < 140 and (stop % 8 == 0 or std.mem.indexOfScalar(u32, want.items[0..stop], want.items[stop]) != null)) stop += 1;
    try std.testing.expect(stop < 140);
    tally = .{};
    const n = try compare(&s, ids, 150, &.{want.items[stop]}, &tally, false);
    try std.testing.expectEqual(stop + 1, n);
    try std.testing.expectEqual(@as(usize, 1), tally.cut);
    s.proposer = null;
    std.debug.print("dflash2 forced drafts: full, zero and partial acceptance, a limit at token {d} and an EOS at {d} inside a block, equal plain greedy and its caches\n", .{ 1 + 8 * 9 + 3, stop });
}
