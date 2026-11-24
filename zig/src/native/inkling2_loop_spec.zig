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
    ids: [inkling.group.max_rows]u32 = undefined,
    picks: [inkling.group.max_rows]u32 = undefined,
    offered: u32 = 0,
    kept: u32 = 0,
    draft_ns: u64 = 0,
    verify_ns: u64 = 0,
    commit_ns: u64 = 0,

    pub fn token(r: *const Round, j: usize) u32 {
        return if (j < r.kept) r.ids[j + 1] else r.picks[r.kept];
    }
};

const Lane = struct { context: *rt.Context, first: u64 = 0, kept: Rows = .{} };

pub const Draft = struct {
    a: Allocator,
    model: *rt.Model,
    glue: q.glue_gpu.Kernels,
    kernels: q.quant_gpu.Kernels,
    dims: mtl.Buffer,
    bound: *tf.checkpoint.Checkpoint,
    config: cfg.Config,
    verifier: inkling.group.Verifier,
    prompt: inkling.Taps,
    step: inkling.Taps,
    lanes: []Lane,
    name: []const u8,
    failed: bool = false,

    pub fn open(a: Allocator, io: std.Io, target: *inkling.Model, dir: []const u8, bits: u8, slots: usize, prompt_rows: usize) !*Draft {
        if (bits != 0 and bits != 4) return error.UnsupportedDraftBits;
        const c = try admit(a, io, target, dir);
        const text = &target.config.text;
        const embed = try affine8(target.weights.embed, text.vocab, text.hidden);
        const unembed = try affine8(target.weights.unembed, text.vocab, text.hidden);
        const d = try a.create(Draft);
        errdefer a.destroy(d);
        const name = try a.dupe(u8, dir);
        errdefer a.free(name);
        d.* = .{ .a = a, .model = undefined, .glue = try q.glue_gpu.Kernels.init(target.device), .kernels = .{ .allocator = a, .device = target.device, .raw_tiles = false }, .dims = undefined, .bound = undefined, .config = c, .verifier = undefined, .prompt = undefined, .step = undefined, .lanes = &.{}, .name = name };
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
        d.prompt = try inkling.Taps.init(target.device, &target.config, tap_ids, prompt_rows);
        errdefer d.prompt.deinit();
        d.step = try inkling.Taps.init(target.device, &target.config, tap_ids, inkling.group.max_rows);
        errdefer d.step.deinit();
        if (d.step.rowBytes() != d.stride()) return error.TapWidth;
        d.lanes = try a.alloc(Lane, slots);
        var made: usize = 0;
        errdefer {
            for (d.lanes[0..made]) |l| l.context.deinit(a);
            a.free(d.lanes);
        }
        for (d.lanes) |*l| {
            l.* = .{ .context = try rt.Context.init(a, d.model, true) };
            made += 1;
        }
        return d;
    }

    pub fn deinit(d: *Draft) void {
        for (d.lanes) |*l| {
            l.context.deinit(d.a);
            l.kept.deinit(d.a);
        }
        d.a.free(d.lanes);
        d.step.deinit();
        d.prompt.deinit();
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
        try list.appendSlice(d.a, &.{ d.dims, d.model.pending, d.verifier.kv, d.verifier.conv.buffer, d.verifier.targets, d.prompt.buffer, d.step.buffer });
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
        return total + d.lanes.len * (d.config.window - 1) * d.stride();
    }

    pub fn stride(d: *const Draft) usize {
        return @as(usize, d.config.tapWidth()) * 2;
    }

    pub fn promptTaps(d: *const Draft) ?*const inkling.Taps {
        return if (d.failed) null else &d.prompt;
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

    pub fn fail(d: *Draft, err: anyerror) void {
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
        const ring = l.context.pending.contents();
        const w: usize = d.config.window;
        var i: usize = 0;
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

    pub fn round(d: *Draft, lane: usize, m: *inkling.Model, batch: *inkling.BatchScratch, cache: *inkling.Cache, pending: u32, offered: u32) !Round {
        var r: Round = .{ .offered = offered };
        r.ids[0] = pending;
        const t0 = now();
        if (offered > 0) d.propose(lane, pending, r.ids[1..][0..offered]) catch |e| {
            d.fail(e);
            return error.DrafterFailed;
        };
        const t1 = now();
        const caches = [1]*inkling.Cache{cache};
        const sizes = [1]usize{@as(usize, offered) + 1};
        const g = try d.verifier.group(&caches, &sizes);
        batch.rows.taps = &d.step;
        const forward = d.verifier.forward(m, batch, &g, r.ids[0 .. offered + 1]);
        batch.rows.taps = null;
        try forward;
        const picks = try d.verifier.picks(m, batch.rows.logits, offered + 1);
        @memcpy(r.picks[0 .. offered + 1], picks);
        r.kept = @intCast(inkling.group.acceptedDrafts(r.ids[1..][0..offered], r.picks[0 .. offered + 1]));
        const t2 = now();
        const accepted = [1]usize{r.kept};
        try d.verifier.commit(m, &g, &accepted);
        d.absorb(lane, d.step.bytes(0, r.kept + 1), r.kept + 1);
        r.draft_ns = t1 - t0;
        r.verify_ns = t2 - t1;
        r.commit_ns = now() - t2;
        return r;
    }

    fn propose(d: *Draft, lane: usize, pending: u32, out: []u32) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const x = d.lanes[lane].context;
        try d.model.swap(x);
        var result = d.model.proposeOptions(pending, .{ .nodes = @intCast(out.len), .children = 1, .edge = 1 });
        d.model.swap(x) catch |e| {
            if (result) |*tree| tree.deinit() else |_| {}
            return e;
        };
        var tree = try result;
        defer tree.deinit();
        if (tree.tokens.len != out.len) return error.ShortDraftChain;
        for (tree.parents, 0..) |p, i| if (p != @as(i32, @intCast(i)) - 1) return error.NotDraftChain;
        @memcpy(out, tree.tokens);
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

pub fn admit(a: Allocator, io: std.Io, target: *const inkling.Model, dir: []const u8) !cfg.Config {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    defer a.free(text);
    return bind(try cfg.parse(a, text), &target.config);
}

pub fn bind(parsed: cfg.Config, t: *const inkling.config.Config) !cfg.Config {
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

pub fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
