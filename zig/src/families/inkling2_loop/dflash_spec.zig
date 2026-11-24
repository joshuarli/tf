//! Greedy speculative decoding of Inkling 2 Mini with its DFlash2 drafter, one grouped verify a round.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const dflash = @import("dflash.zig");

pub const prefill_chunk: usize = 512;

pub const Source = enum { drafter, lookup };

pub const Round = struct {
    position: u64,
    offered: u32,
    accepted: u32,
    emitted: u32,
    drafts: [dflash.depth]u32,
    source: Source,
    draft_ns: u64,
    verify_ns: u64,
    select_ns: u64,
    commit_ns: u64,
    host_ns: u64,
    total_ns: u64,
};

pub const Proposer = struct { ptr: *anyopaque, call: *const fn (*anyopaque, u32, u64, []u32) anyerror!void };

pub const Lookup = struct {
    a: std.mem.Allocator,
    tokens: std.ArrayList(u32) = .empty,
    latest: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    earlier: std.ArrayList(u32) = .empty,

    pub const min_match = 3;
    pub const max_match = 64;
    const max_candidates = 4096;
    const none = std.math.maxInt(u32);

    pub const Match = struct { length: u32, end: u32 };

    pub fn init(a: std.mem.Allocator) Lookup {
        return .{ .a = a };
    }

    pub fn deinit(l: *Lookup) void {
        l.tokens.deinit(l.a);
        l.latest.deinit(l.a);
        l.earlier.deinit(l.a);
    }

    pub fn reset(l: *Lookup) void {
        l.tokens.clearRetainingCapacity();
        l.latest.clearRetainingCapacity();
        l.earlier.clearRetainingCapacity();
    }

    pub fn append(l: *Lookup, ids: []const u32) !void {
        for (ids) |id| {
            if (id >= 1 << 21) return error.LookupToken;
            try l.tokens.append(l.a, id);
            const n = l.tokens.items.len;
            var prior: u32 = none;
            if (n >= min_match) {
                const t = l.tokens.items[n - min_match ..];
                const slot = try l.latest.getOrPut(l.a, (@as(u64, t[0]) << 42) | (@as(u64, t[1]) << 21) | t[2]);
                if (slot.found_existing) prior = slot.value_ptr.*;
                slot.value_ptr.* = @intCast(n - 1);
            }
            try l.earlier.append(l.a, prior);
        }
    }

    pub fn match(l: *const Lookup) ?Match {
        const t = l.tokens.items;
        if (t.len == 0) return null;
        const last = t.len - 1;
        var best: ?Match = null;
        var end = l.earlier.items[last];
        var seen: usize = 0;
        while (end != none and seen < max_candidates) : (seen += 1) {
            var length: u32 = 0;
            while (length < max_match and length <= end and t[end - length] == t[last - length]) length += 1;
            if (best == null or length > best.?.length) best = .{ .length = length, .end = end };
            if (length == max_match) break;
            end = l.earlier.items[end];
        }
        return if (best != null and best.?.length >= min_match) best else null;
    }

    pub fn beyondWindow(l: *const Lookup, window: u32) ?Match {
        const m = l.match() orelse return null;
        return if (m.end + @as(usize, window) <= l.tokens.items.len - 1) m else null;
    }

    pub fn proposals(l: *const Lookup, m: Match, out: []u32) u32 {
        const after = l.tokens.items[m.end + 1 ..];
        const n = @min(out.len, after.len);
        @memcpy(out[0..n], after[0..n]);
        return @intCast(n);
    }
};

pub const Prefill = struct { first: u32, prefill_ns: u64, flush_ns: u64 };

pub const Speculator = struct {
    model: *inkling.Model,
    drafter: *dflash.Drafter,
    verifier: inkling.group.Verifier,
    batch: inkling.BatchScratch,
    prefill: inkling.PrefillScratch,
    taps: inkling.Taps,
    proposer: ?Proposer = null,
    lookup: ?*Lookup = null,
    depth: u32 = dflash.depth,

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

    pub fn start(s: *Speculator, cache: *inkling.Cache, ids: []const u32) !Prefill {
        if (cache.len != 0 or ids.len == 0) return error.BadPrompt;
        try s.drafter.reset();
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
        var pending = first;
        while (tokens.items.len < limit and std.mem.indexOfScalar(u32, eos, pending) == null) {
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
            if (s.lookup) |l| try l.append(tokens.items[tokens.items.len - emitted ..]);
            pending = tokens.items[tokens.items.len - 1];
            const t5 = now();
            var proposed: [dflash.depth]u32 = @splat(0);
            @memcpy(proposed[0..offered], drafts);
            if (rounds) |list| try list.append(a, .{ .position = position, .offered = offered, .accepted = @intCast(kept), .emitted = emitted, .drafts = proposed, .source = source, .draft_ns = t1 - t0, .verify_ns = t2 - t1, .select_ns = t3 - t2, .commit_ns = t4 - t3, .host_ns = t5 - t4, .total_ns = t5 - t0 });
            if (emitted < kept + 1) break;
        }
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
