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

test "a lookup takes the longest earlier suffix of three or more tokens, the most recent of that length" {
    var l = spec.Lookup.init(std.testing.allocator);
    defer l.deinit();
    var out: [7]u32 = undefined;
    try l.append(&.{ 5, 1, 2, 3, 10, 6, 1, 2, 3, 11, 5, 1, 2, 3 });
    const longest = l.match().?;
    try std.testing.expectEqual(spec.Lookup.Match{ .length = 4, .end = 3 }, longest);
    try std.testing.expectEqualSlices(u32, &.{ 10, 6, 1, 2, 3, 11, 5 }, out[0..l.proposals(longest, &out)]);
    l.reset();
    try l.append(&.{ 1, 2, 3, 10, 1, 2, 3, 11, 1, 2, 3 });
    const recent = l.match().?;
    try std.testing.expectEqual(spec.Lookup.Match{ .length = 3, .end = 6 }, recent);
    try std.testing.expectEqualSlices(u32, &.{ 11, 1, 2, 3 }, out[0..l.proposals(recent, &out)]);
    l.reset();
    try l.append(&.{ 1, 2, 3, 4, 2, 3 });
    try std.testing.expectEqual(@as(?spec.Lookup.Match, null), l.match());
    l.reset();
    try std.testing.expectEqual(@as(?spec.Lookup.Match, null), l.match());
    try std.testing.expectError(error.LookupToken, l.append(&.{1 << 21}));
}

test "a lookup finds the longest match behind hundreds of shorter recent ones, as repetitive prompts hold" {
    var l = spec.Lookup.init(std.testing.allocator);
    defer l.deinit();
    try l.append(&.{ 9, 8, 7, 1, 2, 3, 4 });
    for (0..300) |i| try l.append(&.{ @intCast(500 + i), 1, 2, 3, 5 });
    try l.append(&.{ 9, 8, 7, 1, 2, 3 });
    try std.testing.expectEqual(spec.Lookup.Match{ .length = 6, .end = 5 }, l.match().?);
}

test "a lookup drafts only from an occurrence that ends before a drafter's window" {
    var l = spec.Lookup.init(std.testing.allocator);
    defer l.deinit();
    try l.append(&.{ 7, 8, 9, 4 });
    for (0..12) |i| try l.append(&.{@intCast(100 + i)});
    try l.append(&.{ 7, 8, 9 });
    const m = l.match().?;
    try std.testing.expectEqual(@as(u32, 2), m.end);
    try std.testing.expectEqual(@as(usize, 18), l.tokens.items.len - 1);
    try std.testing.expectEqual(m, l.beyondWindow(16).?);
    try std.testing.expectEqual(@as(?spec.Lookup.Match, null), l.beyondWindow(17));
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

test "a round whose latest tokens occurred only before the drafter's window drafts what followed there, as plain greedy decodes" {
    const loaded = try load(.{ .bits = 4 }) orelse return error.SkipZigTest;
    defer loaded.model.deinit();
    defer loaded.drafter.deinit();
    const a = std.testing.allocator;
    const m = loaded.model;
    var s = try spec.Speculator.init(m, loaded.drafter, dflash.tap_ids);
    defer s.deinit();
    var lookup = spec.Lookup.init(a);
    defer lookup.deinit();
    s.lookup = &lookup;
    const parsed = try std.json.parseFromSlice([]const Prompt, a, prompts_json, .{});
    defer parsed.deinit();
    const agent = parsed.value[0].token_ids;
    const repeated = 40;
    const ids = try std.mem.concat(a, u32, &.{ agent, agent[0..repeated] });
    defer a.free(ids);
    try std.testing.expect(ids.len > repeated + loaded.drafter.config.window);
    const first = agent[repeated];
    const limit = 96;
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
    _ = try spec.prefillPlain(m, &s.prefill, &plain_cache, ids);
    try want.append(a, first);
    try spec.plain(m, &s.batch, &plain_cache, first, limit, &.{}, a, &want, null);
    _ = try s.start(&spec_cache, ids);
    try got.append(a, first);
    try s.run(&spec_cache, first, limit, &.{}, a, &got, &rounds);
    try std.testing.expectEqualSlices(u32, want.items, got.items);
    try std.testing.expectEqual(spec_cache.len, s.drafter.committed());
    try expectSameCache(m, &plain_cache, &spec_cache);
    const opening = rounds.items[0];
    try std.testing.expectEqual(spec.Source.lookup, opening.source);
    try std.testing.expectEqual(dflash.depth, opening.offered);
    try std.testing.expectEqualSlices(u32, agent[repeated + 1 ..][0..dflash.depth], opening.drafts[0..dflash.depth]);
    var lookups: usize = 0;
    var accepted: usize = 0;
    for (rounds.items) |r| if (r.source == .lookup) {
        lookups += 1;
        accepted += r.accepted;
    };
    std.debug.print("dflash2 lookup: {d} of {d} rounds drafted from before the drafter's window and accepted {d} drafts; {d} tokens equal plain greedy and its caches\n", .{ lookups, rounds.items.len, accepted, got.items.len });
}

const TreeTally = struct {
    rounds: usize = 0,
    zero: usize = 0,
    /// Rounds that verified more than eight rows, and rounds whose accepted path went past depth 7.
    wide: usize = 0,
    deep: usize = 0,
    /// Rounds whose accepted path left the first draft's subtree, and those an EOS or the limit cut short.
    branched: usize = 0,
    cut: usize = 0,
    looked: usize = 0,
    looked_accepted: usize = 0,
    most_accepted: usize = 0,
    /// Rounds that offered fewer drafts than a tree holds: the output limit falls inside their tree.
    short: usize = 0,

    fn add(t: *TreeTally, rounds: []const spec.Round) void {
        for (rounds) |r| {
            t.rounds += 1;
            t.most_accepted = @max(t.most_accepted, r.accepted);
            if (r.offered < inkling.verify.max_drafts) t.short += 1;
            if (r.offered > 0 and r.accepted == 0) t.zero += 1;
            if (r.offered + 1 > 8) t.wide += 1;
            if (r.accepted > dflash.depth) t.deep += 1;
            if (r.emitted < r.accepted + 1) t.cut += 1;
            t.looked += r.looked;
            t.looked_accepted += r.looked_accepted;
            if (r.accepted > 0) {
                var first_child: usize = 0;
                while (first_child < r.offered and r.parents[first_child] != 0) first_child += 1;
                var on: usize = 0;
                for (0..r.offered) |j| if ((r.hits >> @intCast(j)) & 1 != 0 and r.parents[j] == 0) {
                    on = j;
                    break;
                };
                if (on != first_child) t.branched += 1;
            }
        }
    }
};

fn compareTree(s: *spec.Speculator, ids: []const u32, limit: usize, eos: []const u32, tally: *TreeTally, caches_equal: bool) !usize {
    const a = std.testing.allocator;
    const m = s.model;
    var plain_cache = try inkling.Cache.init(m.device, &m.config, ids.len + limit + inkling.group.max_rows);
    defer plain_cache.deinit();
    var spec_cache = try inkling.Cache.init(m.device, &m.config, ids.len + limit + inkling.group.max_rows);
    defer spec_cache.deinit();
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(a);
    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(a);
    var rounds: std.ArrayList(spec.Round) = .empty;
    defer rounds.deinit(a);
    var bs = try inkling.BatchScratch.init(m.device, &m.config, 1);
    defer bs.deinit();
    const first = try spec.prefillPlain(m, &s.prefill, &plain_cache, ids);
    try want.append(a, first);
    try spec.plain(m, &bs, &plain_cache, first, limit, eos, a, &want, null);
    const started = try s.start(&spec_cache, ids);
    try std.testing.expectEqual(first, started.first);
    try got.append(a, started.first);
    try s.run(&spec_cache, started.first, limit, eos, a, &got, &rounds);
    try std.testing.expectEqualSlices(u32, want.items, got.items);
    try std.testing.expectEqual(spec_cache.len, s.drafter.committed());
    if (caches_equal) try expectSameCache(m, &plain_cache, &spec_cache);
    for (rounds.items) |r| try std.testing.expectEqual(spec.Source.tree, r.source);
    tally.add(rounds.items);
    return want.items.len;
}

test "Inkling DFlash2 tree speculation with lookup chains emits plain greedy decoding's tokens and caches on real prompts" {
    const loaded = try load(.{ .bits = 4 }) orelse return error.SkipZigTest;
    defer loaded.model.deinit();
    defer loaded.drafter.deinit();
    const a = std.testing.allocator;
    var s = try spec.Speculator.init(loaded.model, loaded.drafter, dflash.tap_ids);
    defer s.deinit();
    var lookup = spec.Lookup.init(a);
    defer lookup.deinit();
    s.lookup = &lookup;
    s.mode = .tree;
    try std.testing.expectEqual(inkling.group.max_rows, s.plan.most);
    const parsed = try std.json.parseFromSlice([]const Prompt, a, prompts_json, .{});
    defer parsed.deinit();
    var tally: TreeTally = .{};
    var tokens: usize = 0;
    for (parsed.value) |p| tokens += try compareTree(&s, p.token_ids, 256, &.{}, &tally, true);
    std.debug.print("dflash2 tree speculation: {d} prompts, {d} tokens and their caches equal plain greedy; {d} rounds, {d} past 8 rows, {d} accepted past depth 7, {d} took a later branch, {d} zero; {d} lookup rows, {d} accepted\n", .{ parsed.value.len, tokens, tally.rounds, tally.wide, tally.deep, tally.branched, tally.zero, tally.looked, tally.looked_accepted });
    try std.testing.expect(tally.wide > 0 and tally.deep > 0 and tally.looked_accepted > 0);
}

/// A tree round's drafter tree from a known continuation, shaped by `shape`; scores make every draft likely.
const TreeOracle = struct {
    tokens: []const u32,
    prompt: usize,
    vocab: u32,
    shape: Shape = .deep,
    out_tokens: [inkling.verify.max_drafts]u32 = undefined,
    out_parents: [inkling.verify.max_drafts]i32 = undefined,
    out_scores: [inkling.verify.max_drafts]f64 = undefined,

    const Shape = enum {
        /// A wrong first draft, then the continuation 14 deep under the newest token.
        deep,
        /// Only wrong drafts.
        none,
        /// Two drafts under the newest token, the continuation under the second.
        branch,
    };

    fn next(o: *const TreeOracle, at: usize, j: usize) u32 {
        return if (at + 1 + j < o.tokens.len) o.tokens[at + 1 + j] else 0;
    }

    fn wrong(o: *const TreeOracle, token: u32) u32 {
        return (token + 1) % o.vocab;
    }

    fn call(ptr: *anyopaque, pending: u32, position: u64) anyerror!inkling.verify.Drafted {
        const o: *TreeOracle = @ptrCast(@alignCast(ptr));
        const at: usize = @intCast(position - o.prompt);
        if (o.tokens[at] != pending) return error.OraclePosition;
        var n: usize = 0;
        switch (o.shape) {
            .deep, .none => {
                o.out_tokens[0] = o.wrong(o.next(at, 0));
                o.out_parents[0] = -1;
                o.out_scores[0] = -0.05;
                n = 1;
                for (0..14) |j| {
                    const t = o.next(at, j);
                    o.out_tokens[n] = if (o.shape == .deep) t else o.wrong(o.wrong(t));
                    o.out_parents[n] = if (j == 0) -1 else @intCast(n - 1);
                    o.out_scores[n] = -0.001 * @as(f64, @floatFromInt(j + 1));
                    n += 1;
                }
            },
            .branch => {
                o.out_tokens[0] = o.wrong(o.next(at, 0));
                o.out_parents[0] = -1;
                o.out_scores[0] = -0.01;
                o.out_tokens[1] = o.next(at, 0);
                o.out_parents[1] = -1;
                o.out_scores[1] = -0.02;
                o.out_tokens[2] = o.next(at, 1);
                o.out_parents[2] = 0;
                o.out_scores[2] = -0.03;
                n = 3;
                for (1..6) |j| {
                    o.out_tokens[n] = o.next(at, j);
                    o.out_parents[n] = if (j == 1) 1 else @intCast(n - 1);
                    o.out_scores[n] = -0.02 - 0.001 * @as(f64, @floatFromInt(j));
                    n += 1;
                }
            },
        }
        return .{ .tokens = o.out_tokens[0..n], .parents = o.out_parents[0..n], .scores = o.out_scores[0..n] };
    }
};

test "forced trees accept a 15-row path, none or a later branch, stop at a limit or an EOS inside a tree, and keep plain's caches" {
    const loaded = try load(.{}) orelse return error.SkipZigTest;
    defer loaded.model.deinit();
    defer loaded.drafter.deinit();
    const a = std.testing.allocator;
    const m = loaded.model;
    var s = try spec.Speculator.init(m, loaded.drafter, dflash.tap_ids);
    defer s.deinit();
    s.mode = .tree;
    s.plan.verify_ms = @splat(1);
    const parsed = try std.json.parseFromSlice([]const Prompt, a, prompts_json, .{});
    defer parsed.deinit();
    const ids = parsed.value[1].token_ids;
    var cache = try inkling.Cache.init(m.device, &m.config, ids.len + 400);
    defer cache.deinit();
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(a);
    try want.append(a, try spec.prefillPlain(m, &s.prefill, &cache, ids));
    try spec.plain(m, &s.batch, &cache, want.items[0], 360, &.{}, a, &want, null);
    var oracle = TreeOracle{ .tokens = want.items, .prompt = ids.len, .vocab = m.config.text.unpadded_vocab };
    s.tree_proposer = .{ .ptr = &oracle, .call = TreeOracle.call };
    var tally: TreeTally = .{};
    try std.testing.expectEqual(@as(usize, 1 + 15 * 8), try compareTree(&s, ids, 1 + 15 * 8, &.{}, &tally, true));
    try std.testing.expectEqual(inkling.verify.max_drafts - 1, tally.most_accepted);
    try std.testing.expect(tally.deep >= 7 and tally.wide >= 7);
    tally = .{};
    try std.testing.expectEqual(@as(usize, 1 + 15 * 5 + 6), try compareTree(&s, ids, 1 + 15 * 5 + 6, &.{}, &tally, false));
    try std.testing.expect(tally.deep >= 5 and tally.short >= 2);
    oracle.shape = .none;
    tally = .{};
    _ = try compareTree(&s, ids, 30, &.{}, &tally, true);
    try std.testing.expectEqual(tally.rounds, tally.zero + @as(usize, 1));
    oracle.shape = .branch;
    tally = .{};
    _ = try compareTree(&s, ids, 1 + 7 * 10, &.{}, &tally, true);
    try std.testing.expect(tally.branched >= 9);
    oracle.shape = .deep;
    var stop: usize = 20;
    while (stop < 200 and (stop % 15 == 0 or std.mem.indexOfScalar(u32, want.items[0..stop], want.items[stop]) != null)) stop += 1;
    try std.testing.expect(stop < 200);
    tally = .{};
    const n = try compareTree(&s, ids, 300, &.{want.items[stop]}, &tally, false);
    try std.testing.expectEqual(stop + 1, n);
    try std.testing.expectEqual(@as(usize, 1), tally.cut);
    s.tree_proposer = null;
    std.debug.print("dflash2 forced trees: a 15-row path, none, a later branch, a limit inside a tree and an EOS at {d} inside one, equal plain greedy and its caches\n", .{stop});
}

test "a tree round merges the lookup chain from before the drafter's window, as plain greedy decodes" {
    const loaded = try load(.{ .bits = 4 }) orelse return error.SkipZigTest;
    defer loaded.model.deinit();
    defer loaded.drafter.deinit();
    const a = std.testing.allocator;
    const m = loaded.model;
    var s = try spec.Speculator.init(m, loaded.drafter, dflash.tap_ids);
    defer s.deinit();
    var lookup = spec.Lookup.init(a);
    defer lookup.deinit();
    s.lookup = &lookup;
    s.mode = .tree;
    const parsed = try std.json.parseFromSlice([]const Prompt, a, prompts_json, .{});
    defer parsed.deinit();
    const agent = parsed.value[0].token_ids;
    const repeated = 40;
    const ids = try std.mem.concat(a, u32, &.{ agent, agent[0..repeated] });
    defer a.free(ids);
    var tally: TreeTally = .{};
    const n = try compareTree(&s, ids, 128, &.{}, &tally, true);
    std.debug.print("dflash2 tree lookup: {d} tokens and caches equal plain greedy in {d} rounds; {d} lookup rows, {d} accepted, {d} rounds accepted past depth 7\n", .{ n, tally.rounds, tally.looked, tally.looked_accepted, tally.deep });
    try std.testing.expect(tally.looked_accepted > 0);
}

/// Plain greedy steps of `ids` on `bs` from a fresh cache: the tokens, and each step's logits hashed.
fn plainOn(s: *spec.Speculator, bs: *inkling.BatchScratch, ids: []const u32, steps: usize, out: *std.ArrayList(u32), hashes: *std.ArrayList(u64)) !void {
    const a = std.testing.allocator;
    const m = s.model;
    var cache = try inkling.Cache.init(m.device, &m.config, ids.len + steps + inkling.group.max_rows);
    defer cache.deinit();
    const logits = try a.alloc(f32, m.config.text.unpadded_vocab);
    defer a.free(logits);
    try out.append(a, try spec.prefillPlain(m, &s.prefill, &cache, ids));
    while (out.items.len < steps) {
        try inkling.batchStep(m, bs, &.{&cache}, &.{out.items[out.items.len - 1]}, logits);
        try hashes.append(a, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(logits)));
        try out.append(a, spec.argmax(logits));
    }
}

test "plain greedy steps keep their logits on a fresh one-row scratch and on the verify scratch before and after speculation" {
    const loaded = try load(.{ .bits = 4 }) orelse return error.SkipZigTest;
    defer loaded.model.deinit();
    defer loaded.drafter.deinit();
    const a = std.testing.allocator;
    var s = try spec.Speculator.init(loaded.model, loaded.drafter, dflash.tap_ids);
    defer s.deinit();
    var lookup = spec.Lookup.init(a);
    defer lookup.deinit();
    s.lookup = &lookup;
    s.mode = .tree;
    const parsed = try std.json.parseFromSlice([]const Prompt, a, prompts_json, .{});
    defer parsed.deinit();
    var fresh = try inkling.BatchScratch.init(loaded.model.device, &loaded.model.config, 1);
    defer fresh.deinit();
    var tokens: [4]std.ArrayList(u32) = @splat(.empty);
    defer for (&tokens) |*t| t.deinit(a);
    var hashes: [4]std.ArrayList(u64) = @splat(.empty);
    defer for (&hashes) |*h| h.deinit(a);
    var tally: TreeTally = .{};
    for (0..4) |k| {
        if (k == 2) {
            for (parsed.value) |p| _ = try compareTree(&s, p.token_ids, 101, &.{}, &tally, false);
        }
        try plainOn(&s, if (k == 0 or k == 3) &fresh else &s.batch, parsed.value[1].token_ids, 96, &tokens[k], &hashes[k]);
    }
    var first: [4]usize = @splat(0);
    for (1..4) |k| first[k] = std.mem.indexOfDiff(u64, hashes[0].items, hashes[k].items) orelse hashes[0].items.len;
    std.debug.print("plain scratches: first logits difference from a fresh one-row scratch at step {d} (unused verify scratch), {d} (after {d} speculative rounds), {d} (fresh again), of {d}\n", .{ first[1], first[2], tally.rounds, first[3], hashes[0].items.len });
    for (1..4) |k| try std.testing.expectEqualSlices(u32, tokens[0].items, tokens[k].items);
    for (1..4) |k| try std.testing.expectEqual(hashes[0].items.len, first[k]);
}
