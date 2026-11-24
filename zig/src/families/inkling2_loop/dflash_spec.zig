//! Greedy speculative decoding of Inkling 2 Mini with its DFlash2 drafter, one grouped verify a round.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const dflash = @import("dflash.zig");

pub const prefill_chunk: usize = 512;

pub const Source = enum { drafter, lookup, tree };

/// How a round drafts: the drafter's chain (with DF2a's lookup rounds when `lookup` is set), or a tree.
pub const Mode = enum { chain, tree };

pub const max_drafts = inkling.verify.max_drafts;

pub const Round = struct {
    position: u64,
    offered: u32,
    accepted: u32,
    emitted: u32,
    drafts: [max_drafts]u32,
    /// Row j + 1's parent row; a chain's is row j.
    parents: [max_drafts]u8,
    source: Source,
    /// A tree round's lookup rows, and those its accepted path holds.
    looked: u32 = 0,
    looked_accepted: u32 = 0,
    /// The plan's expected committed tokens.
    expected: f32 = 0,
    /// Per draft: the drafter's probability, its lookup chain place and whether the target's pick at its parent is it.
    q: [max_drafts]f32 = @splat(0),
    chain: [max_drafts]u8 = @splat(inkling.verify.Plan.unlooked),
    hits: u16 = 0,
    match: u32 = 0,
    /// Whether the drafter ran: a tree round leaves it out when the lookup chain suffices.
    drafted: bool = true,
    draft_ns: u64,
    verify_ns: u64,
    select_ns: u64,
    commit_ns: u64,
    host_ns: u64,
    total_ns: u64,
};

pub const Times = tf.qwen27.dflash.runtime_frame.Times;

/// A tree round on `mtl.clock` (host seconds) and the GPU's clock, kept while `TF_INKLING_ROUND_SPLIT` is set.
pub const Split = struct {
    /// Host seconds: the round's start, the drafter's call and return, the tree built, the verify's return, the end.
    start: f64 = 0,
    draft_call: f64 = 0,
    draft_return: f64 = 0,
    planned: f64 = 0,
    verified: f64 = 0,
    done: f64 = 0,
    /// The drafter's frames (none when the round left it out) and the verify's GPU seconds.
    drafter: Times = .{},
    verify_gpu: f64 = 0,
    rows: u32 = 0,
    accepted: u32 = 0,
};

pub const Proposer = struct { ptr: *anyopaque, call: *const fn (*anyopaque, u32, u64, []u32) anyerror!void };
/// Supplies a tree round's drafter tree in place of the drafter: the slices stay valid until the next call.
pub const TreeProposer = struct { ptr: *anyopaque, call: *const fn (*anyopaque, u32, u64) anyerror!inkling.verify.Drafted };

pub const Lookup = inkling.verify.Lookup;

pub const Prefill = struct { first: u32, prefill_ns: u64, flush_ns: u64 };

pub const Speculator = struct {
    model: *inkling.Model,
    drafter: *dflash.Drafter,
    verifier: inkling.group.Verifier,
    batch: inkling.BatchScratch,
    prefill: inkling.PrefillScratch,
    taps: inkling.Taps,
    proposer: ?Proposer = null,
    tree_proposer: ?TreeProposer = null,
    lookup: ?*Lookup = null,
    depth: u32 = dflash.depth,
    mode: Mode = .chain,
    plan: inkling.verify.Plan,
    gathered: []u8,
    a: std.mem.Allocator,
    /// Tree rounds' splits, printed as `inkling2 split` lines when a run ends (null: `TF_INKLING_ROUND_SPLIT` unset).
    splits: ?std.ArrayList(Split) = null,

    pub fn init(model: *inkling.Model, drafter: *dflash.Drafter, visits: []const u32) !Speculator {
        var verifier = try inkling.group.Verifier.init(model.device, &model.config);
        errdefer verifier.deinit();
        var batch = try inkling.group.scratch(model.device, &model.config);
        errdefer batch.deinit();
        var ps = try inkling.PrefillScratch.init(model.device, &model.config, prefill_chunk);
        errdefer ps.deinit();
        var taps = try inkling.Taps.init(model.device, &model.config, visits, prefill_chunk);
        errdefer taps.deinit();
        if (taps.rowBytes() != @as(usize, drafter.config.tapWidth()) * 2) return error.TapWidth;
        const gathered = try drafter.a.alloc(u8, inkling.group.max_rows * taps.rowBytes());
        var plan = inkling.verify.Plan.init();
        plan.most = verifier.limit;
        const split = if (std.c.getenv("TF_INKLING_ROUND_SPLIT")) |v| v[0] != 0 and v[0] != '0' else false;
        return .{ .model = model, .drafter = drafter, .verifier = verifier, .batch = batch, .prefill = ps, .taps = taps, .plan = plan, .gathered = gathered, .a = drafter.a, .splits = if (split) .empty else null };
    }

    pub fn deinit(s: *Speculator) void {
        if (s.splits) |*list| list.deinit(s.a);
        s.a.free(s.gathered);
        s.taps.deinit();
        s.prefill.deinit();
        s.batch.deinit();
        s.verifier.deinit();
    }

    pub fn start(s: *Speculator, cache: *inkling.Cache, ids: []const u32) !Prefill {
        if (cache.len != 0 or ids.len == 0) return error.BadPrompt;
        try s.drafter.reset();
        s.plan.reset();
        if (s.lookup) |l| {
            l.reset();
            try l.append(ids);
        }
        const t0 = now();
        s.prefill.taps = &s.taps;
        defer s.prefill.taps = null;
        var at: usize = 0;
        while (at < ids.len) {
            const take = @min(prefill_chunk, ids.len - at);
            try inkling.prefillWant(s.model, &s.prefill, cache, ids[at..][0..take], if (at + take == ids.len) .last else .none);
            try s.drafter.absorb(s.taps.bytes(0, take), @intCast(take));
            at += take;
        }
        const first = argmax(s.prefill.logits.slice(f32, s.model.config.text.unpadded_vocab));
        const t1 = now();
        try s.drafter.flush();
        return .{ .first = first, .prefill_ns = t1 - t0, .flush_ns = now() - t1 };
    }

    pub fn run(s: *Speculator, cache: *inkling.Cache, first: u32, limit: usize, eos: []const u32, a: std.mem.Allocator, tokens: *std.ArrayList(u32), rounds: ?*std.ArrayList(Round)) !void {
        if (s.drafter.committed() != cache.len) return error.DraftContextBehind;
        if (s.depth == 0 or s.depth > dflash.depth) return error.BadDraftDepth;
        if (s.lookup) |l| if (l.tokens.items.len != cache.len) return error.LookupBehind else try l.append(&.{first});
        defer if (s.splits) |*list| {
            for (list.items) |x| std.debug.print("inkling2 split {{\"start\":{d:.7},\"draft_call\":{d:.7},\"draft_return\":{d:.7},\"planned\":{d:.7},\"verified\":{d:.7},\"done\":{d:.7},\"d_committed\":{d:.7},\"d_returned\":{d:.7},\"d_start\":{d:.7},\"d_end\":{d:.7},\"d_busy\":{d:.7},\"d_buffers\":{d},\"d_frames\":{d},\"v_gpu\":{d:.7},\"rows\":{d},\"accepted\":{d}}}\n", .{ x.start, x.draft_call, x.draft_return, x.planned, x.verified, x.done, x.drafter.committed, x.drafter.returned, x.drafter.start, x.drafter.end, x.drafter.busy, x.drafter.buffers, x.drafter.frames, x.verify_gpu, x.rows, x.accepted });
            list.clearRetainingCapacity();
        };
        var pending = first;
        while (tokens.items.len < limit and std.mem.indexOfScalar(u32, eos, pending) == null) {
            if (s.mode == .tree) {
                if (try s.treeRound(cache, &pending, limit, eos, a, tokens, rounds)) continue;
                break;
            }
            const t0 = now();
            const remaining = limit - tokens.items.len;
            if (cache.len >= cache.capacity) return error.ContextFull;
            const room = cache.capacity - cache.len - 1;
            var offered: u32 = @intCast(@min(@min(s.depth, remaining - 1), room));
            var ids: [inkling.group.max_rows]u32 = undefined;
            ids[0] = pending;
            const beyond = if (offered == 0) null else if (s.lookup) |l| l.beyondWindow(s.drafter.config.window) else null;
            const source: Source = if (beyond != null) .lookup else .drafter;
            if (beyond) |m| {
                offered = s.lookup.?.proposals(m, ids[1..][0..offered]);
            } else if (s.proposer) |p| try p.call(p.ptr, pending, cache.len, ids[1..][0..offered]) else try s.drafter.propose(pending, offered, ids[1..][0..offered]);
            const drafts = ids[1..][0..offered];
            const t1 = now();
            const position = cache.len;
            const g = try s.verifier.group(&.{cache}, &.{@as(usize, offered) + 1});
            s.batch.rows.taps = &s.taps;
            defer s.batch.rows.taps = null;
            const decided = try s.verifier.decide(s.model, &s.batch, &g, ids[0 .. offered + 1], .commit);
            const t2 = now();
            const picks = decided.picks;
            const kept: usize = decided.last[0];
            const t3 = now();
            const t4 = now();
            try s.drafter.absorb(s.taps.bytes(0, kept + 1), @intCast(kept + 1));
            var emitted: u32 = 0;
            for (0..kept + 1) |j| {
                if (tokens.items.len == limit) break;
                const token = if (j < kept) drafts[j] else picks[kept];
                try tokens.append(a, token);
                emitted += 1;
                if (std.mem.indexOfScalar(u32, eos, token) != null) break;
            }
            if (s.lookup) |l| try l.append(tokens.items[tokens.items.len - emitted ..]);
            pending = tokens.items[tokens.items.len - 1];
            const t5 = now();
            var proposed: [max_drafts]u32 = @splat(0);
            @memcpy(proposed[0..offered], drafts);
            var chain: [max_drafts]u8 = undefined;
            for (&chain, 0..) |*p, j| p.* = @intCast(j);
            if (rounds) |list| try list.append(a, .{ .position = position, .offered = offered, .accepted = @intCast(kept), .emitted = emitted, .drafts = proposed, .parents = chain, .source = source, .draft_ns = t1 - t0, .verify_ns = t2 - t1, .select_ns = t3 - t2, .commit_ns = t4 - t3, .host_ns = t5 - t4, .total_ns = t5 - t0 });
            if (emitted < kept + 1) break;
        }
    }

    fn treeRound(s: *Speculator, cache: *inkling.Cache, pending: *u32, limit: usize, eos: []const u32, a: std.mem.Allocator, tokens: *std.ArrayList(u32), rounds: ?*std.ArrayList(Round)) !bool {
        var split: Split = .{};
        const timing = s.splits != null;
        if (timing) split.start = mtl.clock.seconds();
        const t0 = now();
        const remaining = limit - tokens.items.len;
        if (cache.len >= cache.capacity) return error.ContextFull;
        const room = cache.capacity - cache.len - 1;
        const offered: usize = @min(@min(max_drafts, remaining - 1), room);
        var chain: [max_drafts]u32 = undefined;
        var chained: u32 = 0;
        var match_length: u32 = 0;
        if (offered > 0) if (s.lookup) |l| if (l.match()) |m| {
            chained = l.proposals(m, chain[0..offered]);
            match_length = m.length;
        };
        var proposal: ?tf.qwen27.dflash.selector.Tree = null;
        defer if (proposal) |*t| t.deinit();
        var drafted: ?inkling.verify.Drafted = null;
        if (offered > 0 and !s.plan.lookupSuffices(chained, match_length)) {
            if (s.tree_proposer) |p| {
                drafted = try p.call(p.ptr, pending.*, cache.len);
            } else {
                if (timing) {
                    split.draft_call = mtl.clock.seconds();
                    s.drafter.model.backend.frame.times = &split.drafter;
                }
                defer s.drafter.model.backend.frame.times = null;
                proposal = try s.drafter.proposeTree(pending.*, s.plan.nodes, s.plan.children);
                if (timing) split.draft_return = mtl.clock.seconds();
                drafted = .{ .tokens = proposal.?.tokens, .parents = proposal.?.parents, .scores = proposal.?.scores };
            }
        }
        const tree = s.plan.build(pending.*, drafted, chain[0..chained], match_length, offered + 1);
        if (timing) split.planned = mtl.clock.seconds();
        const t1 = now();
        const position = cache.len;
        const g = try s.verifier.tree(&.{cache}, &.{tree.rows}, tree.shape());
        s.batch.rows.taps = &s.taps;
        defer s.batch.rows.taps = null;
        const decided = try s.verifier.decide(s.model, &s.batch, &g, tree.tokens[0..tree.rows], .commit);
        if (timing) split.verified = mtl.clock.seconds();
        const t2 = now();
        var line: [inkling.group.max_rows]usize = undefined;
        const path = g.line(0, decided.last[0], &line);
        var hits: u16 = 0;
        for (1..tree.rows) |r| if (tree.tokens[r] == decided.picks[tree.parents[r - 1]]) {
            hits |= @as(u16, 1) << @intCast(r - 1);
        };
        const bonus = decided.picks[path[path.len - 1]];
        s.plan.learn(&tree, decided.picks);
        const t3 = now();
        const rb = s.taps.rowBytes();
        for (path, 0..) |r, i| @memcpy(s.gathered[i * rb ..][0..rb], s.taps.bytes(r, 1));
        try s.drafter.absorb(s.gathered[0 .. path.len * rb], @intCast(path.len));
        const t4 = now();
        var emitted: u32 = 0;
        var looked_accepted: u32 = 0;
        for (path[1..]) |r| looked_accepted += @intFromBool(tree.chain[r] != inkling.verify.Plan.unlooked);
        for (0..path.len) |j| {
            if (tokens.items.len == limit) break;
            const token = if (j + 1 < path.len) tree.tokens[path[j + 1]] else bonus;
            try tokens.append(a, token);
            emitted += 1;
            if (std.mem.indexOfScalar(u32, eos, token) != null) break;
        }
        if (s.lookup) |l| try l.append(tokens.items[tokens.items.len - emitted ..]);
        pending.* = tokens.items[tokens.items.len - 1];
        const t5 = now();
        if (s.splits) |*list| {
            split.done = mtl.clock.seconds();
            split.verify_gpu = s.batch.rows.gpu_seconds;
            split.rows = @intCast(tree.rows);
            split.accepted = @intCast(path.len - 1);
            try list.append(s.a, split);
        }
        if (rounds) |list| {
            var r: Round = .{ .position = position, .offered = @intCast(tree.rows - 1), .accepted = @intCast(path.len - 1), .emitted = emitted, .drafts = @splat(0), .parents = @splat(0), .source = .tree, .looked = @intCast(tree.looked()), .looked_accepted = looked_accepted, .expected = @floatCast(tree.expected), .hits = hits, .match = tree.match, .drafted = drafted != null, .draft_ns = t1 - t0, .verify_ns = t2 - t1, .select_ns = t3 - t2, .commit_ns = t4 - t3, .host_ns = t5 - t4, .total_ns = t5 - t0 };
            @memcpy(r.drafts[0 .. tree.rows - 1], tree.tokens[1..tree.rows]);
            @memcpy(r.parents[0 .. tree.rows - 1], tree.shape());
            @memcpy(r.q[0 .. tree.rows - 1], tree.q[1..tree.rows]);
            @memcpy(r.chain[0 .. tree.rows - 1], tree.chain[1..tree.rows]);
            try list.append(a, r);
        }
        return emitted == path.len;
    }
};

pub fn plain(m: *inkling.Model, bs: *inkling.BatchScratch, cache: *inkling.Cache, first: u32, limit: usize, eos: []const u32, a: std.mem.Allocator, tokens: *std.ArrayList(u32), steps_ns: ?*std.ArrayList(u64)) !void {
    const logits = try a.alloc(f32, m.config.text.unpadded_vocab);
    defer a.free(logits);
    var last = first;
    while (tokens.items.len < limit and std.mem.indexOfScalar(u32, eos, last) == null) {
        const t0 = now();
        try inkling.batchStep(m, bs, &.{cache}, &.{last}, logits);
        last = argmax(logits);
        try tokens.append(a, last);
        if (steps_ns) |list| try list.append(a, now() - t0);
    }
}

pub fn prefillPlain(m: *inkling.Model, ps: *inkling.PrefillScratch, cache: *inkling.Cache, ids: []const u32) !u32 {
    if (cache.len != 0 or ids.len == 0) return error.BadPrompt;
    var at: usize = 0;
    while (at < ids.len) {
        const take = @min(ps.chunk, ids.len - at);
        try inkling.prefillWant(m, ps, cache, ids[at..][0..take], if (at + take == ids.len) .last else .none);
        at += take;
    }
    return argmax(ps.logits.slice(f32, m.config.text.unpadded_vocab));
}

/// The host's greedy pick: the first index of the largest value, a NaN never winning.
pub fn argmax(values: []const f32) u32 {
    var best: usize = 0;
    for (values[1..], 1..) |value, i| if (value > values[best]) {
        best = i;
    };
    return @intCast(best);
}

pub fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
