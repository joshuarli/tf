const config = @import("config.zig");

pub const Capture = union(enum) {
    entering: struct { layer: u32, pass: u32 },
    final_normalized,
};

pub fn capture(loop: config.Loop, layers: u32, tap: u32) !Capture {
    const slots = loop.slotCount(layers);
    if (tap > slots) return error.InvalidTap;
    if (tap == slots) return .final_normalized;
    if (tap < loop.core_start) return .{ .entering = .{ .layer = tap, .pass = 0 } };
    const width = loop.core_end - loop.core_start;
    const offset = tap - loop.core_start;
    if (offset < width * loop.passes) return .{ .entering = .{
        .layer = loop.core_start + offset % width,
        .pass = offset / width,
    } };
    return .{ .entering = .{ .layer = loop.core_end + offset - width * loop.passes, .pass = 0 } };
}

pub fn proposalCount(mtp: config.Mtp) u32 {
    return mtp.block - 1;
}

/// The drafts the target's greedy picks confirm: `targets[j]` is the target's token after the stream's newest token
/// and its first `j` drafts, so drafts are accepted while each equals the pick before it.
pub fn accepted(drafts: []const u32, targets: []const u32) usize {
    var n: usize = 0;
    while (n < drafts.len and n < targets.len and drafts[n] == targets[n]) n += 1;
    return n;
}

/// A stream's chance that its next draft is accepted given the previous one was, from its recent rounds: each round
/// with drafts counts its accepted drafts as hits and a rejection as a miss, older rounds fading by `decay`.
pub const Estimate = struct {
    hits: f32 = 0,
    misses: f32 = 0,
    /// Rounds since the stream last verified a draft.
    idle: u32 = 0,

    pub const decay: f32 = 0.9;

    pub fn rate(e: Estimate) f32 {
        return (e.hits + 2) / (e.hits + e.misses + 3);
    }

    pub fn note(e: *Estimate, offered: usize, taken: usize) void {
        if (offered == 0) {
            e.idle +|= 1;
            return;
        }
        e.idle = 0;
        e.hits = e.hits * decay + @as(f32, @floatFromInt(taken));
        e.misses = e.misses * decay + @as(f32, @floatFromInt(@intFromBool(taken < offered)));
    }
};

/// Most rows a round verifies, across all its streams.
pub const max_rows: usize = 8;

/// What a round costs in seconds: a verification of `r` rows, and a proposal when any stream drafts. Each row count's
/// cost is measured as rounds run; a count not yet measured follows the line through the fewest and most rows measured,
/// or the prior `base + row * (r - 1)` shifted to the one count measured.
pub const Costs = struct {
    base: f32,
    row: f32,
    propose: f32,
    measured: [max_rows + 1]f32 = @splat(0),

    pub const keep: f32 = 0.8;

    pub fn verify(c: *const Costs, rows: usize) f32 {
        if (rows <= max_rows and c.measured[rows] > 0) return c.measured[rows];
        var lo: usize = 0;
        var hi: usize = 0;
        for (1..max_rows + 1) |r| if (c.measured[r] > 0) {
            if (lo == 0) lo = r;
            hi = r;
        };
        const x: f32 = @floatFromInt(rows);
        if (lo == 0) return c.base + c.row * (x - 1);
        const at_lo: f32 = @floatFromInt(lo);
        if (lo == hi) return c.measured[lo] + c.row * (x - at_lo);
        const at_hi: f32 = @floatFromInt(hi);
        const slope = @max(0, (c.measured[hi] - c.measured[lo]) / (at_hi - at_lo));
        return c.measured[lo] + slope * (x - at_lo);
    }

    pub fn note(c: *Costs, rows: usize, seconds: f32) void {
        if (rows == 0 or rows > max_rows or !std.math.isFinite(seconds) or seconds <= 0) return;
        const old = c.measured[rows];
        c.measured[rows] = if (old > 0) old * keep + seconds * (1 - keep) else seconds;
    }

    pub fn noteProposal(c: *Costs, seconds: f32) void {
        if (!std.math.isFinite(seconds) or seconds <= 0) return;
        c.propose = c.propose * keep + seconds * (1 - keep);
    }
};

/// Drafts to verify for each stream this round: of every split of the spare rows, the one with the most expected
/// emitted tokens per second, each stream's `j`-th draft accepted with chance `rates[i]^j`. A stream always emits one
/// token; drafting adds the proposal's cost. `limits[i]` bounds stream `i`'s drafts.
pub fn plan(rates: []const f32, limits: []const usize, costs: *const Costs, out: []usize) void {
    const n = rates.len;
    @memset(out[0..n], 0);
    if (n == 0 or n >= max_rows) return;
    var best: [max_rows]usize = @splat(0);
    var best_value: f32 = @as(f32, @floatFromInt(n)) / costs.verify(n);
    var k: [max_rows]usize = @splat(0);
    var next: [max_rows]f32 = undefined;
    for (0..n) |i| next[i] = rates[i];
    var tokens: f32 = @floatFromInt(n);
    for (1..max_rows - n + 1) |total| {
        var pick: ?usize = null;
        for (0..n) |i| {
            if (k[i] >= limits[i]) continue;
            if (pick == null or next[i] > next[pick.?]) pick = i;
        }
        const i = pick orelse break;
        tokens += next[i];
        k[i] += 1;
        next[i] *= rates[i];
        const value = tokens / (costs.verify(n + total) + costs.propose);
        if (value > best_value) {
            best_value = value;
            @memcpy(best[0..n], k[0..n]);
        }
    }
    @memcpy(out[0..n], best[0..n]);
}

const std = @import("std");
