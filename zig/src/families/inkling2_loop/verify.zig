const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const Cache = @import("state.zig").Cache;
const ring = @import("state.zig").ring;
const prefill = @import("prefill.zig");
const max_rows = @import("group.zig").max_rows;

/// The most copy drafts a decode round verifies at once: a verify of 8 rows costs about four decode steps.
pub const hybrid_drafts: usize = 7;

pub const RoundTimes = struct { save: u64 = 0, forward: u64 = 0, argmax: u64 = 0, restore: u64 = 0, replay: u64 = 0 };
pub threadlocal var round_times: ?*RoundTimes = null;

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const Verification = struct {
    model: *const Model,
    cache: *Cache,
    scratch: *prefill.PrefillScratch,
    snapshot: mtl.Buffer,
    conv: prefill.ConvCapture,
    start: usize,
    rows: usize,

    pub fn begin(model: *const Model, scratch: *prefill.PrefillScratch, cache: *Cache, ids: []const u32, argmax: []u32) !Verification {
        const started = if (round_times != null) now() else 0;
        if (ids.len == 0 or ids.len > scratch.chunk or ids.len > model.config.text.window or argmax.len != ids.len) return error.BadChunk;
        const c = &model.config;
        const row_bytes: usize = c.text.local.kv * c.text.local.dim * 2;
        const conv_bytes = cache.k_conv.length() + cache.v_conv.length() + cache.attn_conv.length() + cache.mlp_conv.length();
        const bytes = conv_bytes + Cache.localSlots(c) * ids.len * row_bytes * 2;
        const snapshot = try model.device.buffer(@max(bytes, 16), mtl.ResourceOptions.private);
        errdefer snapshot.deinit();
        var conv = try prefill.ConvCapture.init(model.device, c, ids.len);
        errdefer conv.deinit();
        if (scratch.verify_conv != null) return error.VerificationActive;
        var v = Verification{ .model = model, .scratch = scratch, .cache = cache, .snapshot = snapshot, .conv = conv, .start = cache.len, .rows = ids.len };
        try v.copy(true);
        const saved = if (round_times != null) now() else 0;
        scratch.verify_conv = &v.conv;
        defer scratch.verify_conv = null;
        try prefill.prefillWant(model, scratch, cache, ids, .all);
        const forwarded = if (round_times != null) now() else 0;
        const vocab: usize = c.text.unpadded_vocab;
        const logits = scratch.logits.slice(f32, ids.len * vocab);
        for (argmax, 0..) |*out, row| {
            const values = logits[row * vocab ..][0..vocab];
            var best: usize = 0;
            for (values[1..], 1..) |value, i| if (value > values[best]) { best = i; };
            out.* = @intCast(best);
        }
        if (round_times) |t| {
            t.save = saved - started;
            t.forward = forwarded - saved;
            t.argmax = now() - forwarded;
        }
        return v;
    }

    pub fn finish(v: *Verification, ids: []const u32, accepted: usize) !void {
        defer v.snapshot.deinit();
        defer v.conv.deinit();
        if (ids.len != v.rows or accepted > ids.len) return error.BadAcceptedPrefix;
        if (accepted == ids.len) return;
        const started = if (round_times != null) now() else 0;
        if (accepted == 0) {
            try v.copy(false);
            v.cache.len = v.start;
        } else {
            try v.commit(accepted);
        }
        const restored = if (round_times != null) now() else 0;
        if (round_times) |t| {
            t.restore = restored - started;
            t.replay = now() - restored;
        }
    }

    fn commit(v: *Verification, accepted: usize) !void {
        const c = &v.model.config;
        const row_bytes: usize = c.text.local.kv * c.text.local.dim * 2;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = v.model.queue.commandBuffer();
        const blit = cb.blit();
        const rings = [_]mtl.Buffer{ v.cache.k_conv, v.cache.v_conv, v.cache.attn_conv, v.cache.mlp_conv };
        const sites = [_]prefill.ConvSite{ .key, .value, .attention, .mlp };
        var at: usize = 0;
        for (rings) |buf| {
            blit.copy(v.snapshot, at, buf, 0, buf.length());
            at += buf.length();
        }
        for (0..c.slots()) |slot| {
            if (!Cache.isLocalSlot(c, slot)) continue;
            for (0..v.rows) |row| {
                const off = v.cache.row(c, slot, v.start + row);
                for ([_]mtl.Buffer{ v.cache.keys, v.cache.values }) |buf| {
                    if (row >= accepted) blit.copy(v.snapshot, at, buf, off, row_bytes);
                    at += row_bytes;
                }
            }
        }
        for (0..c.slots()) |slot| {
            for (rings, sites) |buf, site| {
                const width = v.conv.width(site);
                for (accepted - @min(accepted, ring)..accepted) |row| {
                    const off = v.cache.convAt(slot, width) + ((v.start + row) % ring) * width * 4;
                    blit.copy(v.conv.buffer, v.conv.offset(slot, site, row), buf, off, width * 4);
                }
            }
        }
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling verify commit: {s}", .{message});
            return error.GpuFailed;
        }
        v.cache.len = v.start + accepted;
    }

    pub fn reset(v: *Verification) !void {
        try v.copy(false);
        v.cache.len = v.start;
    }

    fn copy(v: *Verification, save: bool) !void {
        const c = &v.model.config;
        const row_bytes: usize = c.text.local.kv * c.text.local.dim * 2;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = v.model.queue.commandBuffer();
        const blit = cb.blit();
        var at: usize = 0;
        for ([_]mtl.Buffer{ v.cache.k_conv, v.cache.v_conv, v.cache.attn_conv, v.cache.mlp_conv }) |buf| {
            if (save) blit.copy(buf, 0, v.snapshot, at, buf.length()) else blit.copy(v.snapshot, at, buf, 0, buf.length());
            at += buf.length();
        }
        for (0..c.slots()) |slot| {
            if (!Cache.isLocalSlot(c, slot)) continue;
            for (0..v.rows) |row| {
                const off = v.cache.row(c, slot, v.start + row);
                for ([_]mtl.Buffer{ v.cache.keys, v.cache.values }) |buf| {
                    if (save) blit.copy(buf, off, v.snapshot, at, row_bytes) else blit.copy(v.snapshot, at, buf, off, row_bytes);
                    at += row_bytes;
                }
            }
        }
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling verify snapshot: {s}", .{message});
            return error.GpuFailed;
        }
    }
};

/// A request's tokens indexed by trigram: a round drafts what followed an earlier occurrence of its newest tokens.
pub const Lookup = struct {
    a: std.mem.Allocator,
    tokens: std.ArrayList(u32) = .empty,
    latest: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    earlier: std.ArrayList(u32) = .empty,

    pub const min_match = 3;
    pub const max_match = 64;
    const max_candidates = 4096;
    const missing = std.math.maxInt(u32);

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
            var prior: u32 = missing;
            if (n >= min_match) {
                const t = l.tokens.items[n - min_match ..];
                const slot = try l.latest.getOrPut(l.a, (@as(u64, t[0]) << 42) | (@as(u64, t[1]) << 21) | t[2]);
                if (slot.found_existing) prior = slot.value_ptr.*;
                slot.value_ptr.* = @intCast(n - 1);
            }
            try l.earlier.append(l.a, prior);
        }
    }

    /// Appends what `tokens` holds past the indexed prefix; `tokens` must extend it.
    pub fn follow(l: *Lookup, tokens: []const u32) !void {
        const have = l.tokens.items.len;
        if (tokens.len < have) return error.LookupAhead;
        try l.append(tokens[have..]);
    }

    /// The longest earlier match of the newest tokens' suffix, `min_match` or more long, the latest of that length.
    pub fn match(l: *const Lookup) ?Match {
        const t = l.tokens.items;
        if (t.len == 0) return null;
        const last = t.len - 1;
        var best: ?Match = null;
        var end = l.earlier.items[last];
        var seen: usize = 0;
        while (end != missing and seen < max_candidates) : (seen += 1) {
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

/// The most drafts one round's tree holds under its newest token.
pub const max_drafts = max_rows - 1;

/// A round's draft tree: row 0 is the newest token, every later row a draft under an earlier one.
pub const Tree = struct {
    tokens: [max_rows]u32 = undefined,
    /// `parents[j]`: row j + 1's parent row.
    parents: [max_drafts]u8 = undefined,
    rows: usize = 1,
    depth: [max_rows]u8 = @splat(0),
    /// The drafter's probability for the row's token after its parent's, 0 for a row it did not draft.
    q: [max_rows]f32 = @splat(0),
    /// The row's place in the lookup chain, `Plan.unlooked` for a row the chain does not hold.
    chain: [max_rows]u8 = @splat(Plan.unlooked),
    /// The match length the lookup chain followed, 0 without one.
    match: u32 = 0,
    /// Whether the drafter's tree went into it.
    drafted: bool = false,
    /// Expected tokens a round of `rows` rows commits under the plan's estimates.
    expected: f64 = 1,

    pub fn shape(t: *const Tree) []const u8 {
        return t.parents[0 .. t.rows - 1];
    }

    /// Lookup rows among the drafts.
    pub fn looked(t: *const Tree) usize {
        var n: usize = 0;
        for (t.chain[1..t.rows]) |c| n += @intFromBool(c != Plan.unlooked);
        return n;
    }
};

/// The drafter's best-first tree: token i under `parents[i]` (-1: the newest token), cumulative log probability i.
pub const Drafted = struct { tokens: []const u32, parents: []const i32, scores: []const f64 };

/// Each round's tree from the drafter and the lookup chain, rows from a fixed cost table: no timing changes a tree.
pub const Plan = struct {
    /// Verify ms at 1 to 16 rows (index = rows); the drafter's ms a round.
    verify_ms: [max_rows + 1]f64 = default_verify_ms,
    draft_ms: f64 = default_draft_ms,
    /// Nodes and children a round asks the drafter's selector for.
    nodes: u32 = 32,
    children: u32 = 4,
    /// The most rows a round verifies (at most the verifier's limit).
    most: usize = max_rows,
    /// A round leaves the drafter out when its chain alone likely commits `skip_depth` drafts (0: never).
    skip_at: f64 = 0.7,
    drafter: [drafter_depths][q_bins]Count = undefined,
    lookup: [max_drafts][length_bins]Count = undefined,
    pooled: [max_drafts][length_bins]Count = undefined,
    agree: [q_bins]Count = undefined,

    pub const unlooked: u8 = 0xff;
    pub const skip_depth = 8;
    pub const drafter_depths = 7;
    pub const q_bins = q_edges.len + 1;
    pub const length_bins = length_edges.len + 1;
    const q_edges = [_]f32{ 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9 };
    const length_edges = [_]u32{ 4, 6, 10, 18 };
    const prior_weight: f64 = 12;
    const sibling_cap: f64 = 0.98;
    const Count = struct { hits: f64, trials: f64 };

    pub const default_verify_ms = [max_rows + 1]f64{ 0.0, 13.6, 17.8, 21.0, 23.6, 25.4, 27.0, 28.4, 29.8, 31.3, 32.7, 34.0, 35.3, 36.4, 37.4, 38.4, 39.4 };
    pub const default_draft_ms: f64 = 3.0;

    pub fn init() Plan {
        var p = Plan{};
        p.reset();
        return p;
    }

    /// Back to the priors, as a new request starts.
    pub fn reset(p: *Plan) void {
        for (&p.drafter, 0..) |*row, d| for (row, 0..) |*c, b| {
            c.* = .{ .hits = prior_weight * drafterPrior(d, b), .trials = prior_weight };
        };
        for (&p.lookup, &p.pooled, 0..) |*row, *all, j| for (row, all, 0..) |*c, *u, b| {
            c.* = .{ .hits = prior_weight * lookup_prior[j][b], .trials = prior_weight };
            u.* = .{ .hits = prior_weight * pooled_prior[j][b], .trials = prior_weight };
        };
        for (&p.agree, agree_prior) |*c, prior| c.* = .{ .hits = prior_weight * prior, .trials = prior_weight };
    }

    fn qBin(q: f32) usize {
        var b: usize = 0;
        while (b < q_edges.len and q >= q_edges[b]) b += 1;
        return b;
    }

    fn lengthBin(length: u32) usize {
        var b: usize = 0;
        while (b < length_edges.len and length >= length_edges[b]) b += 1;
        return b;
    }

    fn drafterPrior(depth: usize, bin: usize) f64 {
        return drafter_prior[depth][bin];
    }

    fn rate(c: Count) f64 {
        return c.hits / c.trials;
    }

    /// The estimate that a draft is accepted once its parent is: agreed, the drafter's alone, or the chain's alone.
    pub fn estimate(p: *const Plan, depth: usize, q: f32, chain: u8, match_length: u32, drafted: bool) f64 {
        if (q > 0 and chain != unlooked) return rate(p.agree[qBin(q)]);
        if (q > 0) return rate(p.drafter[@min(depth, drafter_depths - 1)][qBin(q)]);
        const table = if (drafted) &p.lookup else &p.pooled;
        return rate(table[chain][lengthBin(match_length)]);
    }

    /// Whether a lookup chain of `chained` drafts after a match of `match_length` makes the drafter not worth its time.
    pub fn lookupSuffices(p: *const Plan, chained: usize, match_length: u32) bool {
        if (p.skip_at <= 0 or chained < skip_depth) return false;
        var path: f64 = 1;
        for (p.pooled[0..skip_depth]) |row| path *= rate(row[lengthBin(match_length)]);
        return path >= p.skip_at;
    }

    const Node = struct { token: u32, parent: u8, depth: u8, q: f32, chain: u8, p: f64, path: f64 };
    const max_pool_nodes = 1 + 64 + max_drafts;

    pub const BatchInput = struct {
        plan: *const Plan,
        pending: u32,
        drafted: ?Drafted,
        chain: []const u32,
        match_length: u32,
        limit: usize,
    };

    fn collectNodes(p: *const Plan, pending: u32, drafted: ?Drafted, chain: []const u32, match_length: u32, nodes: *[max_pool_nodes]Node) usize {
        nodes[0] = .{ .token = pending, .parent = 0, .depth = 0, .q = 0, .chain = unlooked, .p = 1, .path = 1 };
        var n: usize = 1;
        if (drafted) |d| {
            var at: [64]u8 = undefined;
            for (d.tokens, d.parents, d.scores, 0..) |token, parent, score, i| {
                if (i >= at.len or parent >= @as(i32, @intCast(i)) or parent < -1) break;
                const up: usize = if (parent < 0) 0 else at[@intCast(parent)];
                const before: f64 = if (parent < 0) 0 else d.scores[@intCast(parent)];
                const depth = nodes[up].depth + 1;
                if (depth > max_drafts) break;
                const q: f32 = @max(@as(f32, @floatCast(@exp(score - before))), std.math.floatMin(f32));
                nodes[n] = .{ .token = token, .parent = @intCast(up), .depth = depth, .q = q, .chain = unlooked, .p = 0, .path = 0 };
                at[i] = @intCast(n);
                n += 1;
            }
        }
        var cur: usize = 0;
        for (chain[0..@min(chain.len, max_drafts)], 0..) |token, j| {
            var next: ?usize = null;
            for (1..n) |i| if (nodes[i].parent == cur and nodes[i].token == token) {
                next = i;
                break;
            };
            if (next) |i| {
                nodes[i].chain = @intCast(j);
            } else {
                nodes[n] = .{ .token = token, .parent = @intCast(cur), .depth = nodes[cur].depth + 1, .q = 0, .chain = @intCast(j), .p = 0, .path = 0 };
                next = n;
                n += 1;
            }
            cur = next.?;
        }
        for (nodes[1..n]) |*node| node.p = p.estimate(node.depth - 1, node.q, node.chain, match_length, drafted != null);
        for (0..n) |up| {
            var sum: f64 = 0;
            for (nodes[1..n]) |node| if (node.parent == up) {
                sum += node.p;
            };
            if (sum > sibling_cap) for (nodes[1..n]) |*node| if (node.parent == up) {
                node.p *= sibling_cap / sum;
            };
        }
        for (nodes[1..n]) |*node| node.path = nodes[node.parent].path * node.p;
        return n;
    }

    /// The best drafts by estimated path acceptance, as many as commit the most tokens a millisecond, up to `limit`.
    pub fn build(p: *const Plan, pending: u32, drafted: ?Drafted, chain: []const u32, match_length: u32, limit: usize) Tree {
        var nodes: [max_pool_nodes]Node = undefined;
        const n = p.collectNodes(pending, drafted, chain, match_length, &nodes);
        var order: [max_pool_nodes]u8 = undefined;
        for (1..n) |i| order[i - 1] = @intCast(i);
        const Rank = struct {
            nodes: []const Node,
            fn less(x: @This(), a: u8, b: u8) bool {
                const na = x.nodes[a];
                const nb = x.nodes[b];
                if (na.path != nb.path) return na.path > nb.path;
                if (na.depth != nb.depth) return na.depth < nb.depth;
                return a < b;
            }
        };
        std.mem.sort(u8, order[0 .. n - 1], Rank{ .nodes = nodes[0..n] }, Rank.less);
        const most = @max(1, @min(@min(limit, p.most), n));
        const draft = if (drafted != null) p.draft_ms else 0;
        var rows: usize = 1;
        var expected: f64 = 1;
        var best = expected / (draft + p.verify_ms[1]);
        var sum: f64 = 1;
        for (2..most + 1) |r| {
            sum += nodes[order[r - 2]].path;
            const value = sum / (draft + p.verify_ms[r]);
            if (value > best) {
                best = value;
                rows = r;
                expected = sum;
            }
        }
        var t = Tree{ .rows = rows, .match = if (chain.len > 0) match_length else 0, .expected = expected, .drafted = drafted != null };
        var row_of: [max_pool_nodes]u8 = undefined;
        row_of[0] = 0;
        t.tokens[0] = pending;
        for (order[0 .. rows - 1], 1..) |i, r| {
            const node = nodes[i];
            row_of[i] = @intCast(r);
            t.tokens[r] = node.token;
            t.parents[r - 1] = row_of[node.parent];
            t.depth[r] = node.depth;
            t.q[r] = node.q;
            t.chain[r] = node.chain;
        }
        return t;
    }

    /// Jointly allocates up to `total_limit` rows across concurrent streams to maximize combined tokens per millisecond.
    pub fn buildBatch(inputs: []const BatchInput, total_limit: usize, out: []Tree) void {
        std.debug.assert(inputs.len > 0 and inputs.len <= 8 and inputs.len == out.len);
        const n = inputs.len;
        var all_nodes: [8][max_pool_nodes]Node = undefined;
        var counts: [8]usize = undefined;
        var any_drafted = false;
        const ref = inputs[0].plan;
        const Item = struct { stream: u8, node: u8, depth: u8, path: f64 };
        var pool: [8 * max_pool_nodes]Item = undefined;
        var pool_len: usize = 0;
        for (inputs, 0..) |inp, s| {
            if (inp.drafted != null) any_drafted = true;
            counts[s] = inp.plan.collectNodes(inp.pending, inp.drafted, inp.chain, inp.match_length, &all_nodes[s]);
            if (inp.limit > 1) {
                for (1..counts[s]) |i| {
                    const nd = all_nodes[s][i];
                    pool[pool_len] = .{ .stream = @intCast(s), .node = @intCast(i), .depth = nd.depth, .path = nd.path };
                    pool_len += 1;
                }
            }
        }
        const Rank = struct {
            fn less(_: void, a: Item, b: Item) bool {
                if (a.path != b.path) return a.path > b.path;
                if (a.depth != b.depth) return a.depth < b.depth;
                if (a.stream != b.stream) return a.stream < b.stream;
                return a.node < b.node;
            }
        };
        std.mem.sort(Item, pool[0..pool_len], {}, Rank.less);
        const cap = @min(total_limit, max_rows);
        const draft = if (any_drafted) ref.draft_ms else 0;
        var picked: [max_rows]Item = undefined;
        var picked_len: usize = 0;
        var stream_rows: [8]usize = @splat(1);
        var total_rows: usize = n;
        var sum: f64 = @floatFromInt(n);
        var best = sum / (draft + ref.verify_ms[@min(total_rows, max_rows)]);
        var best_picked: usize = 0;
        for (pool[0..pool_len]) |item| {
            if (total_rows >= cap) break;
            const s = item.stream;
            const stream_cap = @min(inputs[s].limit, inputs[s].plan.most);
            if (stream_rows[s] >= stream_cap) continue;
            picked[picked_len] = item;
            picked_len += 1;
            stream_rows[s] += 1;
            total_rows += 1;
            sum += item.path;
            const value = sum / (draft + ref.verify_ms[total_rows]);
            if (value > best) {
                best = value;
                best_picked = picked_len;
            }
        }
        var row_of: [8][max_pool_nodes]u8 = undefined;
        for (inputs, 0..) |inp, s| {
            out[s] = .{ .rows = 1, .match = if (inp.chain.len > 0) inp.match_length else 0, .expected = 1, .drafted = inp.drafted != null };
            out[s].tokens[0] = inp.pending;
            row_of[s][0] = 0;
        }
        for (picked[0..best_picked]) |item| {
            const s = item.stream;
            const node = all_nodes[s][item.node];
            const r = out[s].rows;
            row_of[s][item.node] = @intCast(r);
            out[s].tokens[r] = node.token;
            out[s].parents[r - 1] = row_of[s][node.parent];
            out[s].depth[r] = node.depth;
            out[s].q[r] = node.q;
            out[s].chain[r] = node.chain;
            out[s].expected += node.path;
            out[s].rows = r + 1;
        }
    }

    /// Learns from each row whose parent the target accepted: whether the target's pick there is the row's token.
    pub fn learn(p: *Plan, t: *const Tree, picks: []const u32) void {
        var on: [max_rows]bool = @splat(false);
        var taken: [max_rows]bool = @splat(false);
        on[0] = true;
        for (1..t.rows) |r| {
            const parent = t.parents[r - 1];
            if (!on[parent]) continue;
            const accepted = t.tokens[r] == picks[parent];
            if (accepted and !taken[parent]) {
                on[r] = true;
                taken[parent] = true;
            }
            const hit: f64 = @floatFromInt(@intFromBool(accepted));
            const q = t.q[r];
            const chained = t.chain[r] != unlooked;
            const c = if (q > 0 and chained) &p.agree[qBin(q)] else if (q > 0) &p.drafter[@min(@as(usize, t.depth[r]) - 1, drafter_depths - 1)][qBin(q)] else if (t.drafted) &p.lookup[t.chain[r]][lengthBin(t.match)] else &p.pooled[t.chain[r]][lengthBin(t.match)];
            c.hits += hit;
            c.trials += 1;
            if (chained and t.drafted) {
                const u = &p.pooled[t.chain[r]][lengthBin(t.match)];
                u.hits += hit;
                u.trials += 1;
            }
        }
    }
};

/// Acceptance after an accepted parent: the drafter's draft alone by depth and `Plan.q_edges` bin (run.py calibrate).
pub const drafter_prior = [Plan.drafter_depths][Plan.q_bins]f64{
    .{ 0.026, 0.026, 0.032, 0.125, 0.265, 0.4, 0.663, 0.759, 0.873, 0.97 },
    .{ 0.045, 0.045, 0.045, 0.137, 0.202, 0.435, 0.518, 0.747, 0.802, 0.959 },
    .{ 0.048, 0.048, 0.048, 0.085, 0.133, 0.395, 0.638, 0.707, 0.773, 0.966 },
    .{ 0.062, 0.062, 0.062, 0.114, 0.222, 0.521, 0.521, 0.6, 0.654, 0.937 },
    .{ 0.064, 0.064, 0.064, 0.128, 0.249, 0.488, 0.628, 0.628, 0.767, 0.941 },
    .{ 0.064, 0.064, 0.064, 0.128, 0.249, 0.424, 0.588, 0.618, 0.856, 0.947 },
    .{ 0.064, 0.064, 0.064, 0.128, 0.249, 0.424, 0.523, 0.68, 0.724, 0.981 },
};

/// The lookup chain's draft alone in a round the drafter ran, by place and `Plan.length_edges` bin.
pub const lookup_prior = [max_drafts][Plan.length_bins]f64{
    .{ 0.077, 0.205, 0.527, 0.733, 0.88 },
    .{ 0.284, 0.336, 0.615, 0.844, 0.845 },
    .{ 0.212, 0.358, 0.713, 0.797, 0.98 },
    .{ 0.192, 0.429, 0.836, 0.935, 0.935 },
    .{ 0.192, 0.303, 0.811, 0.932, 0.932 },
    .{ 0.192, 0.34, 0.726, 0.733, 0.94 },
    .{ 0.192, 0.303, 0.692, 0.873, 0.985 },
    .{ 0.192, 0.52, 0.721, 0.894, 0.951 },
    .{ 0.192, 0.372, 0.712, 0.894, 0.993 },
    .{ 0.192, 0.34, 0.726, 0.884, 0.963 },
    .{ 0.192, 0.34, 0.692, 0.773, 0.991 },
    .{ 0.192, 0.34, 0.692, 0.841, 0.99 },
    .{ 0.192, 0.34, 0.692, 0.841, 0.989 },
    .{ 0.192, 0.34, 0.692, 0.841, 0.945 },
    .{ 0.192, 0.34, 0.692, 0.841, 0.986 },
};

/// The lookup chain's draft whatever the drafter drew, by place and `Plan.length_edges` bin.
pub const pooled_prior = [max_drafts][Plan.length_bins]f64{
    .{ 0.642, 0.696, 0.858, 0.895, 0.988 },
    .{ 0.788, 0.788, 0.818, 0.91, 0.966 },
    .{ 0.673, 0.673, 0.882, 0.882, 0.995 },
    .{ 0.831, 0.831, 0.948, 0.979, 0.987 },
    .{ 0.829, 0.831, 0.966, 0.966, 0.966 },
    .{ 0.777, 0.854, 0.897, 0.897, 0.973 },
    .{ 0.749, 0.749, 0.945, 0.949, 0.984 },
    .{ 0.722, 0.814, 0.816, 0.941, 0.97 },
    .{ 0.707, 0.707, 0.839, 0.941, 0.995 },
    .{ 0.722, 0.745, 0.896, 0.936, 0.981 },
    .{ 0.722, 0.745, 0.853, 0.853, 0.995 },
    .{ 0.722, 0.745, 0.883, 0.912, 0.963 },
    .{ 0.722, 0.745, 0.883, 0.912, 0.995 },
    .{ 0.722, 0.745, 0.883, 0.912, 0.976 },
    .{ 0.722, 0.745, 0.883, 0.912, 0.995 },
};

/// Both drafting the same token, by `Plan.q_edges` bin.
pub const agree_prior = [Plan.q_bins]f64{ 0.347, 0.347, 0.347, 0.385, 0.385, 0.853, 0.853, 0.853, 0.989, 0.995 };

test "Plan.buildBatch allocates rows across concurrent streams by path probability and reduces to Plan.build for one stream" {
    const plan = Plan.init();
    const d0 = Drafted{
        .tokens = &.{ 10, 11, 12, 13, 14, 15, 16 },
        .parents = &.{ -1, 0, 1, 2, 3, 4, 5 },
        .scores = &.{ -0.02, -0.04, -0.06, -0.08, -0.10, -0.12, -0.14 },
    };
    const d1 = Drafted{
        .tokens = &.{ 20, 21, 22, 23 },
        .parents = &.{ -1, 0, 1, 2 },
        .scores = &.{ -0.05, -2.5, -5.0, -7.5 },
    };
    const solo = plan.build(1, d0, &.{}, 0, 16);
    var one: [1]Tree = undefined;
    Plan.buildBatch(&.{.{ .plan = &plan, .pending = 1, .drafted = d0, .chain = &.{}, .match_length = 0, .limit = 16 }}, 16, &one);
    try std.testing.expectEqual(solo.rows, one[0].rows);
    try std.testing.expectEqualSlices(u32, solo.tokens[0..solo.rows], one[0].tokens[0..one[0].rows]);
    try std.testing.expectEqualSlices(u8, solo.shape(), one[0].shape());

    var pair: [2]Tree = undefined;
    Plan.buildBatch(&.{
        .{ .plan = &plan, .pending = 1, .drafted = d0, .chain = &.{}, .match_length = 0, .limit = 16 },
        .{ .plan = &plan, .pending = 2, .drafted = d1, .chain = &.{}, .match_length = 0, .limit = 16 },
    }, 10, &pair);
    try std.testing.expect(pair[0].rows + pair[1].rows <= 10);
    try std.testing.expect(pair[0].rows > pair[1].rows);
    try std.testing.expect(pair[1].rows >= 2);
}
