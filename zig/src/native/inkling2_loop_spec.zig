const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const q = tf.qwen27;
const rt = q.dflash.runtime_model;
const cfg = q.dflash.config;
const ck = q.checkpoint;
const Allocator = std.mem.Allocator;

pub const depth: u32 = 7;
/// The most drafts a round's tree holds.
pub const max_drafts: u32 = inkling.verify.max_drafts;
const tap_ids = cfg.inkling_tap_ids[0..4];

pub const Rows = struct {
    bytes: []u8 = &.{},
    first: u64 = 0,

    pub fn deinit(r: *Rows, a: Allocator) void {
        a.free(r.bytes);
        r.* = .{};
    }
};

pub const Round = struct {
    /// The accepted path's drafts and then the target's pick after them.
    tokens: [inkling.group.max_rows]u32 = undefined,
    offered: u32 = 0,
    kept: u32 = 0,
    /// The tree's lookup rows, and those on the accepted path.
    looked: u32 = 0,
    looked_kept: u32 = 0,
    draft_ns: u64 = 0,
    verify_ns: u64 = 0,
    commit_ns: u64 = 0,

    pub fn token(r: *const Round, j: usize) u32 {
        return r.tokens[j];
    }
};

/// A request's tokens: its prompt and what it has generated, the newest last.
pub const History = struct { prompt: []const u32, emitted: []const u32 };

const Lane = struct { context: *rt.Context, first: u64 = 0, kept: Rows = .{}, stash: Rows = .{}, stash_rows: usize = 0, lookup: inkling.verify.Lookup, plan: inkling.verify.Plan };

pub const Draft = struct {
    a: Allocator,
    model: *rt.Model,
    glue: q.glue_gpu.Kernels,
    kernels: q.quant_gpu.Kernels,
    dims: mtl.Buffer,
    bound: *tf.checkpoint.Checkpoint,
    config: cfg.Config,
    verifier: inkling.group.Verifier,
    step: inkling.Taps,
    lanes: []Lane,
    gathered: []u8,
    name: []const u8,
    failed: bool = false,

    pub fn open(a: Allocator, io: std.Io, target: *inkling.Model, dir: []const u8, bits: u8, slots: usize) !*Draft {
        if (bits != 0 and bits != 4) return error.UnsupportedDraftBits;
        const c = try admit(a, io, target, dir);
        const text = &target.config.text;
        const embed = try affine8(target.weights.embed, text.vocab, text.hidden);
        const unembed = try affine8(target.weights.unembed, text.vocab, text.hidden);
        const d = try a.create(Draft);
        errdefer a.destroy(d);
        const name = try a.dupe(u8, dir);
        errdefer a.free(name);
        d.* = .{ .a = a, .model = undefined, .glue = try q.glue_gpu.Kernels.init(target.device), .kernels = .{ .allocator = a, .device = target.device, .raw_tiles = false }, .dims = undefined, .bound = undefined, .config = c, .verifier = undefined, .step = undefined, .lanes = &.{}, .gathered = &.{}, .name = name };
        errdefer d.glue.deinit();
        errdefer d.kernels.deinit();
        d.dims = try target.device.buffer(4096, mtl.ResourceOptions.shared);
        errdefer d.dims.deinit();
        d.bound = try a.create(tf.checkpoint.Checkpoint);
        errdefer a.destroy(d.bound);
        d.bound.* = tf.checkpoint.Checkpoint.init(a);
        errdefer d.bound.deinit();
        const head = tf.row_projection.Weights{ .w = unembed.weight.buffer, .w_off = unembed.weight.offset, .scales = unembed.scales.buffer, .s_off = unembed.scales.offset, .biases = unembed.biases.buffer, .b_off = unembed.biases.offset, .n = (@as(usize, text.unpadded_vocab) + 31) / 32 * 32, .k = text.hidden, .group = 64, .bits = 8 };
        try head.validate();
        const bindings: q.dflash.runtime_backend.Target = .{
            .device = target.device,
            .queue = target.queue,
            .checkpoint = d.bound,
            .config = .{ .hidden = text.hidden, .vocab = text.vocab, .layers = target.config.slots() },
            .weights = .{ .embed = embed, .head = .{ .words = .{ .buffer = head.w, .offset = head.w_off }, .pairs = .{ .buffer = head.scales, .offset = head.s_off }, .n = @intCast(head.n), .k = @intCast(text.hidden), .tile = 32, .slices = 1, .raw = head, .bits = 8, .group = 64 } },
            .kernels = &d.kernels,
            .glue = &d.glue,
            .dims = .{ .buffer = d.dims },
            .embed_binding = try ck.validate(hostTensor(embed.weight), hostTensor(embed.scales), hostTensor(embed.biases), .{ .bits = 8, .group_size = 64 }, text.vocab, text.hidden),
            .embed_norm = .{ .weight = .{ .buffer = target.weights.embed_norm.buffer, .offset = target.weights.embed_norm.offset }, .eps = text.eps },
        };
        const shards = try tf.checkpoint_host.shardFiles(a, io, dir);
        defer tf.checkpoint_host.freeShardFiles(a, shards);
        d.model = try rt.Model.openBound(a, bindings, c, shards, if (bits == 4) .prepared_q4_reference else .bf16_reference);
        errdefer d.model.deinit();
        try d.model.setTreeBlock(c.block);
        try d.warm();
        d.verifier = try inkling.group.Verifier.init(target.device, &target.config);
        errdefer d.verifier.deinit();
        d.step = try inkling.Taps.init(target.device, &target.config, tap_ids, inkling.group.max_rows);
        errdefer d.step.deinit();
        if (d.step.rowBytes() != d.stride()) return error.TapWidth;
        d.gathered = try a.alloc(u8, inkling.group.max_rows * d.stride());
        errdefer a.free(d.gathered);
        d.lanes = try a.alloc(Lane, slots);
        var made: usize = 0;
        errdefer {
            for (d.lanes[0..made]) |*l| {
                l.context.deinit(a);
                l.lookup.deinit();
            }
            a.free(d.lanes);
        }
        for (d.lanes) |*l| {
            var plan = inkling.verify.Plan.init();
            plan.most = d.verifier.limit;
            l.* = .{ .context = try rt.Context.init(a, d.model, true), .lookup = inkling.verify.Lookup.init(a), .plan = plan };
            made += 1;
        }
        return d;
    }

    pub fn deinit(d: *Draft) void {
        for (d.lanes) |*l| {
            l.context.deinit(d.a);
            l.kept.deinit(d.a);
            l.stash.deinit(d.a);
            l.lookup.deinit();
        }
        d.a.free(d.lanes);
        d.a.free(d.gathered);
        d.step.deinit();
        d.verifier.deinit();
        d.model.deinit();
        d.bound.deinit();
        d.a.destroy(d.bound);
        d.dims.deinit();
        d.kernels.deinit();
        d.glue.deinit();
        d.a.free(d.name);
        d.a.destroy(d);
    }

    pub fn bytes(d: *Draft) !u64 {
        var list: std.ArrayList(mtl.Buffer) = .empty;
        defer list.deinit(d.a);
        var it = d.model.checkpoint.tensors.valueIterator();
        while (it.next()) |t| try list.append(d.a, t.buffer);
        try list.appendSlice(d.a, &.{ d.dims, d.model.pending, d.verifier.kv, d.verifier.conv.buffer, d.verifier.targets, d.step.buffer });
        try q.residency.appendDraft(&d.model.backend, d.a, &list);
        for (d.lanes) |l| try l.context.buffers(&list, d.a);
        var total: u64 = 0;
        for (list.items, 0..) |b, i| {
            var seen = false;
            for (list.items[0..i]) |prior| if (prior.id == b.id) {
                seen = true;
            };
            if (!seen) total += b.length();
        }
        return total + d.lanes.len * (d.config.window - 1) * d.stride() + d.gathered.len;
    }

    fn stride(d: *const Draft) usize {
        return @as(usize, d.config.tapWidth()) * 2;
    }

    pub fn windowRows(d: *const Draft) usize {
        return d.config.window;
    }

    pub fn ringWrite(d: *const Draft, lane: usize, at: usize, take: usize, need_from: usize) ?inkling.Taps.RingWrite {
        if (d.failed) return null;
        const start = @max(at, need_from);
        const end = at + take;
        if (start >= end) return null;
        const w: usize = d.config.window;
        return .{
            .taps = &d.step,
            .ring = d.lanes[lane].context.pending,
            .skip = @intCast(start - at),
            .rows = @intCast(end - start),
            .base = @intCast(start % w),
            .window = d.config.window,
        };
    }

    pub fn stashOverwritten(d: *Draft, lane: usize, capture_at: usize, chunk_end: usize) !void {
        if (d.failed) return;
        const l = &d.lanes[lane];
        const w: usize = d.config.window;
        const from_u64 = @max(l.first, capture_at -| (w -| 1));
        const overwritten_end = @min(capture_at, chunk_end -| w);
        if (from_u64 >= overwritten_end) {
            l.stash_rows = 0;
            return;
        }
        const from: usize = @intCast(from_u64);
        const n = overwritten_end - from;
        const s = d.stride();
        if (l.stash.bytes.len < n * s) l.stash.bytes = try d.a.realloc(l.stash.bytes, n * s);
        l.stash.first = from;
        l.stash_rows = n;
        const ring = l.context.pending.contents();
        var i: usize = 0;
        while (i < n) {
            const slot: usize = (from + i) % w;
            const count = @min(n - i, w - slot);
            @memcpy(l.stash.bytes[i * s ..][0 .. count * s], ring[slot * s ..][0 .. count * s]);
            i += count;
        }
    }

    pub fn clearStash(d: *Draft, lane: usize) void {
        d.lanes[lane].stash_rows = 0;
    }

    pub fn advanceTo(d: *Draft, lane: usize, end: u64) void {
        if (d.failed) return;
        d.lanes[lane].context.committed_end = end;
    }

    pub fn stepTaps(d: *const Draft) ?*const inkling.Taps {
        return if (d.failed) null else &d.step;
    }

    pub fn ready(d: *const Draft, lane: usize, at: u64) bool {
        return !d.failed and d.lanes[lane].context.committed_end == at;
    }

    pub fn kept(d: *const Draft, lane: usize) Rows {
        return d.lanes[lane].kept;
    }

    fn fail(d: *Draft, err: anyerror) void {
        if (!d.failed) std.log.err("inkling2_loop: the DFlash2 drafter failed ({s}); every reply now decodes without it", .{@errorName(err)});
        d.failed = true;
    }

    pub fn begin(d: *Draft, lane: usize, at: u64) void {
        if (d.failed) return;
        const l = &d.lanes[lane];
        l.context.session = q.dflash.session.Session.init(d.config, 1, at) catch |e| return d.fail(e);
        l.context.context = l.context.session.context;
        l.context.committed_end = at;
        l.first = at;
        l.stash_rows = 0;
    }

    pub fn absorb(d: *Draft, lane: usize, rows: []const u8, n: usize) void {
        if (d.failed or n == 0) return;
        d.lanes[lane].context.ring(d.config).absorbBytes(rows, @intCast(n)) catch |e| d.fail(e);
    }

    pub fn follow(d: *Draft, lane: usize, row: usize, at: u64) void {
        if (d.failed or d.lanes[lane].context.committed_end + 1 != at) return;
        d.absorb(lane, d.step.bytes(row, 1), 1);
    }

    pub fn restore(d: *Draft, lane: usize, rows: Rows, end: u64) void {
        const s = d.stride();
        const have: u64 = rows.bytes.len / s;
        if (end < rows.first or end > rows.first + have) return d.begin(lane, end);
        const from = @max(rows.first, end -| (d.config.window - 1));
        d.begin(lane, from);
        d.absorb(lane, rows.bytes[@intCast((from - rows.first) * s)..@intCast((end - rows.first) * s)], @intCast(end - from));
    }

    pub fn keepBytes(d: *const Draft, lane: usize, end: u64) usize {
        const l = &d.lanes[lane];
        if (d.failed or l.context.committed_end != end) return 0;
        return @intCast((end - @max(l.first, end -| (d.config.window - 1))) * d.stride());
    }

    pub fn keep(d: *Draft, lane: usize, end: u64, a: Allocator, out: *Rows) !void {
        const l = &d.lanes[lane];
        if (d.failed or l.context.committed_end != end) {
            out.deinit(a);
            out.first = end;
            return;
        }
        const from = @max(l.first, end -| (d.config.window - 1));
        const n: usize = @intCast(end - from);
        const s = d.stride();
        out.bytes = try a.realloc(out.bytes, n * s);
        out.first = from;
        var i: usize = 0;
        if (l.stash_rows > 0 and from >= l.stash.first and from < l.stash.first + l.stash_rows) {
            const off: usize = @intCast(from - l.stash.first);
            const take = @min(n, l.stash_rows - off);
            @memcpy(out.bytes[0 .. take * s], l.stash.bytes[off * s ..][0 .. take * s]);
            i = take;
        }
        const ring = l.context.pending.contents();
        const w: usize = d.config.window;
        while (i < n) {
            const slot: usize = @intCast((from + i) % w);
            const count = @min(n - i, w - slot);
            @memcpy(out.bytes[i * s ..][0 .. count * s], ring[slot * s ..][0 .. count * s]);
            i += count;
        }
    }

    pub fn keepPrompt(d: *Draft, lane: usize, end: u64) !void {
        return d.keep(lane, end, d.a, &d.lanes[lane].kept);
    }

    /// A new request takes the lane: its lookup and estimates start over.
    pub fn forget(d: *Draft, lane: usize) void {
        const l = &d.lanes[lane];
        l.lookup.reset();
        l.plan.reset();
    }

    /// One tree round from the request's newest token over the drafter and its lookup chain: the accepted path.
    pub fn round(d: *Draft, lane: usize, m: *inkling.Model, batch: *inkling.BatchScratch, cache: *inkling.Cache, history: History, offered: u32) !Round {
        const l = &d.lanes[lane];
        const pending = history.emitted[history.emitted.len - 1];
        var r: Round = .{};
        const t0 = now();
        var chain: [inkling.verify.max_drafts]u32 = undefined;
        var chained: u32 = 0;
        var match_length: u32 = 0;
        if (offered > 0) {
            if (catchUp(&l.lookup, history)) |_| {
                if (l.lookup.match()) |found| {
                    chained = l.lookup.proposals(found, chain[0..offered]);
                    match_length = found.length;
                }
            } else |_| l.lookup.reset();
        }
        var proposal: ?tf.qwen27.dflash.selector.Tree = null;
        defer if (proposal) |*t| t.deinit();
        if (offered > 0 and !l.plan.lookupSuffices(chained, match_length)) proposal = d.propose(lane, pending, l.plan.nodes, l.plan.children) catch |e| {
            d.fail(e);
            return error.DrafterFailed;
        };
        const drafted: ?inkling.verify.Drafted = if (proposal) |t| .{ .tokens = t.tokens, .parents = t.parents, .scores = t.scores } else null;
        const tree = l.plan.build(pending, drafted, chain[0..chained], match_length, @as(usize, offered) + 1);
        const t1 = now();
        const g = try d.verifier.tree(&.{cache}, &.{tree.rows}, tree.shape());
        batch.rows.taps = &d.step;
        const decided = d.verifier.decide(m, batch, &g, tree.tokens[0..tree.rows], .commit);
        batch.rows.taps = null;
        const outcome = try decided;
        const t2 = now();
        var line: [inkling.group.max_rows]usize = undefined;
        const path = g.line(0, outcome.last[0], &line);
        for (path[1..], 0..) |row, j| {
            r.tokens[j] = tree.tokens[row];
            r.looked_kept += @intFromBool(tree.chain[row] != inkling.verify.Plan.unlooked);
        }
        r.tokens[path.len - 1] = outcome.picks[path[path.len - 1]];
        r.offered = @intCast(tree.rows - 1);
        r.kept = @intCast(path.len - 1);
        r.looked = @intCast(tree.looked());
        l.plan.learn(&tree, outcome.picks);
        const s = d.stride();
        for (path, 0..) |row, i| @memcpy(d.gathered[i * s ..][0..s], d.step.bytes(row, 1));
        d.absorb(lane, d.gathered[0 .. path.len * s], path.len);
        r.draft_ns = t1 - t0;
        r.verify_ns = t2 - t1;
        r.commit_ns = now() - t2;
        return r;
    }

    fn catchUp(lookup: *inkling.verify.Lookup, history: History) !void {
        var have = lookup.tokens.items.len;
        if (have < history.prompt.len) {
            try lookup.append(history.prompt[have..]);
            have = history.prompt.len;
        }
        if (have - history.prompt.len > history.emitted.len) return error.LookupAhead;
        try lookup.append(history.emitted[have - history.prompt.len ..]);
    }

    fn propose(d: *Draft, lane: usize, pending: u32, nodes: u32, children: u32) !tf.qwen27.dflash.selector.Tree {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const x = d.lanes[lane].context;
        try d.model.swap(x);
        var result = d.model.proposeOptions(pending, .{ .nodes = nodes, .children = children, .edge = 1 });
        d.model.swap(x) catch |e| {
            if (result) |*tree| tree.deinit() else |_| {}
            return e;
        };
        return result;
    }

    fn warm(d: *Draft) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        for (0..3) |_| {
            try d.model.reset();
            @memset(d.model.pending.contents()[0..d.stride()], 0);
            d.model.committed_end = 1;
            var tree = try d.model.proposeOptions(1000, .{ .nodes = depth, .children = 1, .edge = 1 });
            tree.deinit();
        }
        try d.model.reset();
    }
};

fn admit(a: Allocator, io: std.Io, target: *const inkling.Model, dir: []const u8) !cfg.Config {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    defer a.free(text);
    return bind(try cfg.parse(a, text), &target.config);
}

fn bind(parsed: cfg.Config, t: *const inkling.config.Config) !cfg.Config {
    if (t.checkpoint != .mlx) return error.InklingDFlash2NeedsAffineHead;
    var c = parsed;
    try c.bind(t.text.hidden, t.text.vocab, t.slots());
    if (!std.mem.eql(u32, c.tapIds(), tap_ids) or !c.final_norm) return error.InklingDFlash2Taps;
    if (c.softcap != t.text.logit_cap) return error.InklingDFlash2Softcap;
    if (c.block != depth + 1) return error.InklingDFlash2Block;
    c.candidates = t.text.unpadded_vocab;
    try c.requireStock();
    return c;
}

fn affine8(t: inkling.Tensor, rows: usize, cols: usize) !q.projection.Triple {
    const a = t.affine orelse return error.InklingDFlash2NeedsAffineHead;
    if (a.kind != .affine8 or t.dtype != .u32 or t.rank != 2 or t.shape[0] != rows or t.bytes != rows * cols) return error.InklingDFlash2NeedsAffineHead;
    const meta = rows * (cols / 64) * 2;
    return .{
        .weight = .{ .buffer = t.buffer, .offset = t.offset, .bytes = t.bytes, .dtype = .u32, .shape = .{ rows, cols / 4, 1, 1, 1 }, .rank = 2 },
        .scales = .{ .buffer = a.scales, .offset = a.scale_offset, .bytes = meta, .dtype = .bf16, .shape = .{ rows, cols / 64, 1, 1, 1 }, .rank = 2 },
        .biases = .{ .buffer = a.biases, .offset = a.bias_offset, .bytes = meta, .dtype = .bf16, .shape = .{ rows, cols / 64, 1, 1, 1 }, .rank = 2 },
        .bits = 8,
        .group = 64,
    };
}

fn hostTensor(t: tf.checkpoint.Tensor) ck.Tensor {
    var shape: @FieldType(ck.Tensor, "shape") = @splat(1);
    @memcpy(shape[0..t.rank], t.shape[0..t.rank]);
    return .{ .dtype = t.dtype, .rank = @intCast(t.rank), .shape = shape, .bytes = t.buffer.contents()[t.offset..][0..t.bytes] };
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
