//! Greedy speculative decoding of Inkling 2 Mini with its DFlash2 drafter: each round drafts a chain after the newest
//! token, verifies it in one grouped batch step, commits the accepted prefix and gives the drafter its rows' taps.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const dflash = @import("dflash.zig");

pub const prefill_chunk: usize = 512;

/// One round: its anchor's position, what it drafted and kept, and where its wall time went.
pub const Round = struct {
    position: u64,
    offered: u32,
    accepted: u32,
    emitted: u32,
    draft_ns: u64,
    verify_ns: u64,
    select_ns: u64,
    commit_ns: u64,
    host_ns: u64,
    total_ns: u64,
};

/// Drafts other than the drafter's, for checks that force acceptance: `out.len` tokens after `pending` at `position`.
pub const Proposer = struct { ptr: *anyopaque, call: *const fn (*anyopaque, u32, u64, []u32) anyerror!void };

pub const Prefill = struct { first: u32, prefill_ns: u64, flush_ns: u64 };

pub const Speculator = struct {
    model: *inkling.Model,
    drafter: *dflash.Drafter,
    verifier: inkling.group.Verifier,
    batch: inkling.BatchScratch,
    prefill: inkling.PrefillScratch,
    taps: inkling.Taps,
    proposer: ?Proposer = null,
    depth: u32 = dflash.depth,

    /// `visits`: the target rows the drafter reads, `dflash.tap_ids` but for a diagnostic of another reading.
    pub fn init(model: *inkling.Model, drafter: *dflash.Drafter, visits: []const u32) !Speculator {
        var verifier = try inkling.group.Verifier.init(model.device, &model.config);
        errdefer verifier.deinit();
        var batch = try inkling.BatchScratch.init(model.device, &model.config, inkling.group.max_rows);
        errdefer batch.deinit();
        var ps = try inkling.PrefillScratch.init(model.device, &model.config, prefill_chunk);
        errdefer ps.deinit();
        const taps = try inkling.Taps.init(model.device, &model.config, visits, prefill_chunk);
        if (taps.rowBytes() != @as(usize, drafter.config.tapWidth()) * 2) return error.TapWidth;
        return .{ .model = model, .drafter = drafter, .verifier = verifier, .batch = batch, .prefill = ps, .taps = taps };
    }

    pub fn deinit(s: *Speculator) void {
        s.taps.deinit();
        s.prefill.deinit();
        s.batch.deinit();
        s.verifier.deinit();
    }

    /// `ids` into an empty `cache`, every row's taps into a reset drafter, then the drafter's window: the first token.
    pub fn start(s: *Speculator, cache: *inkling.Cache, ids: []const u32) !Prefill {
        if (cache.len != 0 or ids.len == 0) return error.BadPrompt;
        try s.drafter.reset();
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

    /// Decodes after `first` (already in `tokens`) until `tokens` holds `limit` tokens or ends at an `eos` token.
    pub fn run(s: *Speculator, cache: *inkling.Cache, first: u32, limit: usize, eos: []const u32, a: std.mem.Allocator, tokens: *std.ArrayList(u32), rounds: ?*std.ArrayList(Round)) !void {
        if (s.drafter.committed() != cache.len) return error.DraftContextBehind;
        if (s.depth == 0 or s.depth > dflash.depth) return error.BadDraftDepth;
        var pending = first;
        while (tokens.items.len < limit and std.mem.indexOfScalar(u32, eos, pending) == null) {
            const t0 = now();
            const remaining = limit - tokens.items.len;
            if (cache.len >= cache.capacity) return error.ContextFull;
            const room = cache.capacity - cache.len - 1;
            const offered: u32 = @intCast(@min(@min(s.depth, remaining - 1), room));
            var ids: [inkling.group.max_rows]u32 = undefined;
            ids[0] = pending;
            const drafts = ids[1..][0..offered];
            if (s.proposer) |p| try p.call(p.ptr, pending, cache.len, drafts) else try s.drafter.propose(pending, offered, drafts);
            const t1 = now();
            const position = cache.len;
            const g = try s.verifier.group(&.{cache}, &.{@as(usize, offered) + 1});
            s.batch.rows.taps = &s.taps;
            defer s.batch.rows.taps = null;
            try s.verifier.forward(s.model, &s.batch, &g, ids[0 .. offered + 1]);
            const t2 = now();
            const picks = try s.verifier.picks(s.model, s.batch.rows.logits, offered + 1);
            const kept = inkling.group.acceptedDrafts(drafts, picks);
            const t3 = now();
            try s.verifier.commit(s.model, &g, &.{kept});
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
            pending = tokens.items[tokens.items.len - 1];
            const t5 = now();
            if (rounds) |list| try list.append(a, .{ .position = position, .offered = offered, .accepted = @intCast(kept), .emitted = emitted, .draft_ns = t1 - t0, .verify_ns = t2 - t1, .select_ns = t3 - t2, .commit_ns = t4 - t3, .host_ns = t5 - t4, .total_ns = t5 - t0 });
            if (emitted < kept + 1) break;
        }
    }
};

/// Plain greedy decoding: one batch step a token after `first`, until `tokens` holds `limit` or ends at an `eos` token.
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

/// A prompt prefilled without taps: its first token.
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
