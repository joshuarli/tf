//! Parent-conditioned DFlash tree search leaves all target verification and keyed draws authoritative.
const std = @import("std");
const weights = @import("weights.zig");
const op = @import("operators.zig");
pub const Noise = struct { ptr: *anyopaque, draw: *const fn (*anyopaque, u64, u32) anyerror!f64 };
pub const Options = struct { nodes: u32 = 15, children: u32 = 4, tau: f64 = 1.5, edge: f64 = 0.6, noise_weight: f64 = 0.7, temperature: f64 = 1, noise: ?Noise = null, costs: []const f64 = &.{}, overhead_ms: f64 = 0 };
pub const Tree = struct {
    gpa: std.mem.Allocator,
    tokens: []u32,
    parents: []i32,
    scores: []f64,
    pub fn deinit(t: *Tree) void {
        t.gpa.free(t.tokens);
        t.gpa.free(t.parents);
        t.gpa.free(t.scores);
        t.* = undefined;
    }
    pub fn verifyParents(t: Tree, gpa: std.mem.Allocator) ![]i32 {
        const parents = try gpa.alloc(i32, t.parents.len + 1);
        parents[0] = -1;
        for (t.parents, 0..) |p, i| {
            if (p < -1 or p >= i) {
                gpa.free(parents);
                return error.BadDraftParents;
            }
            parents[i + 1] = p + 1;
        }
        return parents;
    }
};
const Node = struct { score: f64, parent: i32, token: u32, depth: u32 };
fn before(a: Node, b: Node) bool {
    return a.score > b.score or (a.score == b.score and (a.parent < b.parent or (a.parent == b.parent and a.token < b.token)));
}
/// The most selector rank `expand` holds a weighted row for.
const max_rank = 1024;
/// A codebook row's bf16 values, every one finite.
fn codebookRow(t: weights.Tensor, row: u32, rank: usize) ![]const u8 {
    const at = (try std.math.mul(usize, row, t.shape[1])) * 2;
    if (rank > t.shape[1] or at + rank * 2 > t.bytes.len) return error.DraftCodebookShape;
    const bytes = t.bytes[at..][0 .. rank * 2];
    for (0..rank) |r| if (!std.math.isFinite(bf16(bytes, r))) return error.DraftCodebookFinite;
    return bytes;
}
fn bf16(bytes: []const u8, r: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bytes[r * 2 ..][0..2], .little)) << 16);
}
/// The parent's predecessor row times the depth's projection; `edge` multiplies each by a successor value.
fn weighted(g: *const weights.Graph, lattice: op.Lattice, parent: u32, depth: u32, out: []f64) !void {
    @setFloatMode(.strict);
    const row = try codebookRow(g.predecessor, parent, out.len);
    for (out, 0..) |*w, r| w.* = @as(f64, bf16(row, r)) * @as(f64, lattice.projected[@as(usize, depth) * lattice.rank + r]);
}
/// The edge score from the parent's weighted row: (p * x) * s summed in rank order.
fn edge(g: *const weights.Graph, parent: []const f64, token: u32) !f64 {
    @setFloatMode(.strict);
    const row = try codebookRow(g.successor, token, parent.len);
    var sum: f64 = 0;
    for (parent, 0..) |w, r| sum += w * @as(f64, bf16(row, r));
    return sum;
}
fn capped(score: f64, cap: f32) f64 {
    @setFloatMode(.strict);
    const c: f64 = cap;
    return c * std.math.tanh(score / c);
}
fn expand(g: *const weights.Graph, lattice: op.Lattice, options: Options, first: u64, parent_token: u32, parent: i32, depth: u32, cumulative: f64, frontier: *std.ArrayList(Node)) !void {
    @setFloatMode(.strict);
    var scores: [16]f64 = undefined;
    var order: [16]u32 = undefined;
    const temperature = if (options.noise != null) @max(options.temperature, 1e-6) else 1;
    const row = @as(usize, depth) * lattice.topk;
    var maximum: f64 = -std.math.inf(f64);
    if (g.config.rank > max_rank) return error.DraftCodebookShape;
    var parent_row: [max_rank]f64 = undefined;
    const parent_weights = parent_row[0..g.config.rank];
    try weighted(g, lattice, parent_token, depth, parent_weights);
    for (0..lattice.topk) |i| {
        const token = lattice.candidates[row + i];
        const e = try edge(g, parent_weights, token);
        var score = if (g.config.softcap > 0) capped(@as(f64, lattice.unary[row + i]) + options.edge * e, g.config.softcap) / temperature else @as(f64, lattice.unary[row + i]) / temperature + options.edge * e / temperature;
        if (options.noise) |n| score += options.noise_weight * try n.draw(n.ptr, try std.math.add(u64, first, depth), token);
        score /= options.tau;
        if (!std.math.isFinite(score)) return error.DraftScoreFinite;
        scores[i] = score;
        order[i] = @intCast(i);
        maximum = @max(maximum, score);
    }
    var denominator: f64 = 0;
    for (scores[0..lattice.topk]) |*score| {
        score.* -= maximum;
        denominator += @exp(score.*);
    }
    const normalizer = @log(denominator);
    for (scores[0..lattice.topk]) |*score| score.* -= normalizer;
    const Rank = struct {
        scores: []const f64,
        fn less(x: @This(), a: u32, b: u32) bool {
            return x.scores[a] > x.scores[b] or (x.scores[a] == x.scores[b] and a < b);
        }
    };
    std.mem.sort(u32, order[0..lattice.topk], Rank{ .scores = scores[0..lattice.topk] }, Rank.less);
    for (order[0..@min(options.children, lattice.topk)]) |i| {
        std.debug.assert(frontier.items.len < frontier.capacity);
        frontier.appendAssumeCapacity(.{ .score = cumulative + scores[i], .parent = parent, .token = lattice.candidates[row + i], .depth = depth });
    }
}
pub fn propose(gpa: std.mem.Allocator, g: *const weights.Graph, lattice: op.Lattice, anchor: u32, first_position: u64, options: Options) !Tree {
    try g.config.check();
    for ([_]weights.Tensor{ g.predecessor, g.successor }) |t| if (t.dtype != .bf16 or t.rank != 2 or t.shape[0] != g.config.vocab or t.shape[1] != g.config.rank) return error.DraftCodebookShape;
    try lattice.check(g.config);
    if (anchor >= g.config.vocab or options.nodes > 63 or options.children == 0 or options.children > 16 or lattice.topk > 16 or !std.math.isFinite(options.tau) or options.tau <= 0 or !std.math.isFinite(options.edge) or !std.math.isFinite(options.noise_weight) or !std.math.isFinite(options.temperature) or options.temperature <= 0) return error.BadDraftSelector;
    if (!std.math.isFinite(options.overhead_ms) or options.overhead_ms < 0) return error.BadDraftCosts;
    if (options.costs.len > 0) {
        if (options.costs.len <= options.nodes) return error.BadDraftCosts;
        for (options.costs) |ms| if (!std.math.isFinite(ms) or ms <= 0) return error.BadDraftCosts;
    }
    // At most sixteen roots and sixteen children for each of the sixty-three selected nodes.
    var storage: [16 * 64]Node = undefined;
    var frontier: std.ArrayList(Node) = .initBuffer(&storage);
    var tokens: [63]u32 = undefined;
    var parents: [63]i32 = undefined;
    var scores: [63]f64 = undefined;
    if (options.nodes > 0) try expand(g, lattice, options, first_position, anchor, -1, 0, 0, &frontier);
    var count: usize = 0;
    while (frontier.items.len > 0 and count < options.nodes) {
        var best: usize = 0;
        for (frontier.items[1..], 1..) |node, i| if (before(node, frontier.items[best])) {
            best = i;
        };
        const node = frontier.swapRemove(best);
        tokens[count] = node.token;
        parents[count] = node.parent;
        scores[count] = node.score;
        const me = count;
        count += 1;
        if (node.depth + 1 < lattice.depth) try expand(g, lattice, options, first_position, node.token, @intCast(me), node.depth + 1, node.score, &frontier);
    }
    // Each included path contributes its probability to expected accepted depth; every prefix is parent-closed.
    if (options.costs.len > 0) {
        var best: usize = 0;
        var expected: f64 = 1;
        var rate = expected / (options.costs[0] + options.overhead_ms);
        for (scores[0..count], 0..) |score, i| {
            expected += @exp(score);
            const next = expected / (options.costs[i + 1] + options.overhead_ms);
            if (next > rate) {
                rate = next;
                best = i + 1;
            }
        }
        count = best;
    }
    const out_tokens = try gpa.dupe(u32, tokens[0..count]);
    errdefer gpa.free(out_tokens);
    const out_parents = try gpa.dupe(i32, parents[0..count]);
    errdefer gpa.free(out_parents);
    return .{ .gpa = gpa, .tokens = out_tokens, .parents = out_parents, .scores = try gpa.dupe(f64, scores[0..count]) };
}

test "draft roots shift to pending row while descendants retain topology" {
    const a = std.testing.allocator;
    const t = Tree{ .gpa = a, .tokens = &.{}, .parents = @constCast(&[_]i32{ -1, -1, 0, 2 }), .scores = &.{} };
    const parents = try t.verifyParents(a);
    defer a.free(parents);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 1, 3 }, parents);
}
test "a softcapped draft caps each candidate's score after its edge and keeps a greedy chain's picks" {
    const a = std.testing.allocator;
    var store = try @import("fixtures.zig").Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    var candidates = [_]u32{ 3, 7, 3, 7 };
    var unary = [_]f32{ 100, 90, 60, 61 };
    var projected: [32]f32 = @splat(0);
    const lattice = op.Lattice{ .gpa = a, .depth = 2, .topk = 2, .rank = 16, .candidates = &candidates, .unary = &unary, .projected = &projected };
    var plain = try propose(a, &g, lattice, 1, 10, .{ .nodes = 2, .children = 1 });
    defer plain.deinit();
    g.config.softcap = 30;
    var capped_tree = try propose(a, &g, lattice, 1, 10, .{ .nodes = 2, .children = 1 });
    defer capped_tree.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 3, 7 }, plain.tokens);
    try std.testing.expectEqualSlices(u32, plain.tokens, capped_tree.tokens);
    const gap: f64 = (30 * std.math.tanh(@as(f64, 100) / 30) - 30 * std.math.tanh(@as(f64, 90) / 30)) / 1.5;
    try std.testing.expectApproxEqAbs(-@log(1 + @exp(-gap)), capped_tree.scores[0], 1e-12);
    try std.testing.expectApproxEqAbs(-@log(1 + @exp(@as(f64, -10) / 1.5)), plain.scores[0], 1e-12);
}
test "an edge from the parent's weighted row has the bits of the predecessor, projection and successor product summed by rank" {
    @setFloatMode(.strict);
    const a = std.testing.allocator;
    var store = try @import("fixtures.zig").Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x5c1);
    const random = prng.random();
    for ([_]weights.Tensor{ g.predecessor, g.successor }) |t| {
        const bytes: []u8 = @constCast(t.bytes);
        for (0..bytes.len / 2) |i| {
            const value: f32 = (random.float(f32) - 0.5) * @as(f32, @floatFromInt(1 + i % 7));
            std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @intCast(@as(u32, @bitCast(value)) >> 16), .little);
        }
    }
    const rank = g.config.rank;
    var projected: [3 * 16]f32 = undefined;
    for (&projected) |*x| x.* = random.float(f32) * 3 - 1.5;
    var candidates = [_]u32{ 0, 1, 2, 3, 4, 5 };
    var unary = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const lattice = op.Lattice{ .gpa = a, .depth = 3, .topk = 2, .rank = @intCast(rank), .candidates = &candidates, .unary = &unary, .projected = &projected };
    var row: [16]f64 = undefined;
    for (0..g.config.vocab) |parent| for (0..3) |depth| {
        try weighted(&g, lattice, @intCast(parent), @intCast(depth), row[0..rank]);
        for (0..g.config.vocab) |token| {
            var want: f64 = 0;
            for (0..rank) |r| want += @as(f64, bf16(g.predecessor.bytes[parent * rank * 2 ..], r)) * @as(f64, projected[depth * rank + r]) * @as(f64, bf16(g.successor.bytes[token * rank * 2 ..], r));
            try std.testing.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(try edge(&g, row[0..rank], @intCast(token)))));
        }
    };
    const bytes: []u8 = @constCast(g.successor.bytes);
    std.mem.writeInt(u16, bytes[5 * rank * 2 + 6 ..][0..2], 0x7fc0, .little);
    try std.testing.expectError(error.DraftCodebookFinite, edge(&g, row[0..rank], 5));
}
