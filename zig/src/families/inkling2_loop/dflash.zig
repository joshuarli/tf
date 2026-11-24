//! incoai/Inkling-2-Mini-DFlash2 on the shared Metal DFlash2 runtime (`qwen27/dflash`), bound to the oQ4e target's
//! affine 8-bit embedding through its embedding norm and to its head over the target's real tokens.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const q = tf.qwen27;
const rt = q.dflash.runtime_model;
const cfg = q.dflash.config;
const ck = q.checkpoint;

pub const tap_ids = cfg.inkling_tap_ids[0..4];
/// Proposals a round verifies after its anchor: the checkpoint's block of 8.
pub const depth: u32 = 7;
/// Rows the head computes: whole 32-row tiles over the 200,058 real tokens, inside the 201,024-row storage.
const head_rows: usize = 200064;

pub const Drafter = struct {
    a: std.mem.Allocator,
    model: *rt.Model,
    glue: q.glue_gpu.Kernels,
    kernels: q.quant_gpu.Kernels,
    dims: mtl.Buffer,
    bound: *tf.checkpoint.Checkpoint,
    config: cfg.Config,
    step_ms: f64 = 0,

    /// `bits` 0 reads the checkpoint's BF16 linears, 4 their prepared q4 copies; `embed_norm` false is a diagnostic
    /// that feeds the token table's rows without the target's norm.
    pub const Options = struct { bits: u8 = 0, embed_norm: bool = true };

    pub fn init(a: std.mem.Allocator, io: std.Io, target: *inkling.Model, dir: []const u8, options: Options) !*Drafter {
        const bits = options.bits;
        if (bits != 0 and bits != 4) return error.UnsupportedDraftBits;
        const c = try admit(a, io, target, dir);
        const embed = try affine8(target.weights.embed, target.config.text.vocab, target.config.text.hidden);
        const unembed = try affine8(target.weights.unembed, target.config.text.vocab, target.config.text.hidden);
        const d = try a.create(Drafter);
        errdefer a.destroy(d);
        d.* = .{ .a = a, .model = undefined, .glue = try q.glue_gpu.Kernels.init(target.device), .kernels = .{ .allocator = a, .device = target.device, .raw_tiles = false }, .dims = undefined, .bound = undefined, .config = c };
        errdefer d.glue.deinit();
        errdefer d.kernels.deinit();
        d.dims = try target.device.buffer(4096, mtl.ResourceOptions.shared);
        errdefer d.dims.deinit();
        d.bound = try a.create(tf.checkpoint.Checkpoint);
        errdefer a.destroy(d.bound);
        d.bound.* = tf.checkpoint.Checkpoint.init(a);
        errdefer d.bound.deinit();
        const hidden = target.config.text.hidden;
        const head = tf.row_projection.Weights{ .w = unembed.weight.buffer, .w_off = unembed.weight.offset, .scales = unembed.scales.buffer, .s_off = unembed.scales.offset, .biases = unembed.biases.buffer, .b_off = unembed.biases.offset, .n = head_rows, .k = hidden, .group = 64, .bits = 8 };
        try head.validate();
        const bindings: q.dflash.runtime_backend.Target = .{
            .device = target.device,
            .queue = target.queue,
            .checkpoint = d.bound,
            .config = .{ .hidden = hidden, .vocab = target.config.text.vocab, .layers = target.config.slots() },
            .weights = .{ .embed = embed, .head = .{ .words = .{ .buffer = head.w, .offset = head.w_off }, .pairs = .{ .buffer = head.scales, .offset = head.s_off }, .n = head_rows, .k = @intCast(hidden), .tile = 32, .slices = 1, .raw = head, .bits = 8, .group = 64 } },
            .kernels = &d.kernels,
            .glue = &d.glue,
            .dims = .{ .buffer = d.dims },
            .embed_binding = try ck.validate(host(embed.weight), host(embed.scales), host(embed.biases), .{ .bits = 8, .group_size = 64 }, target.config.text.vocab, hidden),
            .embed_norm = if (options.embed_norm) .{ .weight = .{ .buffer = target.weights.embed_norm.buffer, .offset = target.weights.embed_norm.offset }, .eps = target.config.text.eps } else null,
        };
        const shards = try tf.checkpoint_host.shardFiles(a, io, dir);
        defer tf.checkpoint_host.freeShardFiles(a, shards);
        d.model = try rt.Model.openBound(a, bindings, c, shards, if (bits == 4) .prepared_q4_reference else .bf16_reference);
        errdefer d.model.deinit();
        try d.model.setTreeBlock(c.block);
        try d.warm();
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.model.deinit();
        d.bound.deinit();
        d.a.destroy(d.bound);
        d.dims.deinit();
        d.kernels.deinit();
        d.glue.deinit();
        d.a.destroy(d);
    }

    /// An empty context: no committed taps.
    pub fn reset(d: *Drafter) !void {
        try d.model.reset();
    }

    /// Committed positions whose taps the context holds.
    pub fn committed(d: *const Drafter) u64 {
        return d.model.committed_end;
    }

    /// `rows` committed target positions' tap rows, `tapWidth` bf16 each, after the ones the context holds.
    pub fn absorb(d: *Drafter, rows_bytes: []const u8, rows: u32) !void {
        try d.model.ring().absorbBytes(rows_bytes, rows);
    }

    /// The window's keys and values from the committed taps, so the next proposal absorbs only its round's rows.
    pub fn flush(d: *Drafter) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try d.model.flush();
    }

    /// The greedy chain of `n` proposals after `pending`, the newest committed token's successor not yet verified.
    pub fn propose(d: *Drafter, pending: u32, n: u32, out: []u32) !void {
        if (n == 0) return;
        if (n > depth or out.len < n) return error.BadDraftDepth;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var tree = try d.model.proposeOptions(pending, .{ .nodes = n, .children = 1, .edge = 1 });
        defer tree.deinit();
        if (tree.tokens.len != n) return error.ShortDraftChain;
        for (tree.parents, 0..) |p, i| if (p != @as(i32, @intCast(i)) - 1) return error.NotDraftChain;
        @memcpy(out[0..n], tree.tokens);
    }

    pub fn buffers(d: *Drafter, list: *std.ArrayList(mtl.Buffer)) !void {
        var it = d.model.checkpoint.tensors.valueIterator();
        while (it.next()) |t| try list.append(d.a, t.buffer);
        try list.appendSlice(d.a, &.{ d.dims, d.model.pending });
        try q.residency.appendDraft(&d.model.backend, d.a, list);
    }

    /// Device bytes the drafter holds beyond its target's: its weights, prepared copies, context and workspace.
    pub fn fixedBytes(d: *Drafter) !u64 {
        var list: std.ArrayList(mtl.Buffer) = .empty;
        defer list.deinit(d.a);
        try d.buffers(&list);
        var total: u64 = 0;
        for (list.items, 0..) |b, i| {
            var seen = false;
            for (list.items[0..i]) |prior| if (prior.id == b.id) {
                seen = true;
            };
            if (!seen) total += b.length();
        }
        return total;
    }

    fn warm(d: *Drafter) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        for (0..3) |i| {
            try d.model.reset();
            @memset(d.model.pending.contents()[0 .. d.config.tapWidth() * 2], 0);
            d.model.committed_end = 1;
            const t0 = seconds();
            var tree = try d.model.proposeOptions(1000, .{ .nodes = depth, .children = 1, .edge = 1 });
            tree.deinit();
            if (i == 2) d.step_ms = (seconds() - t0) * 1000;
        }
        try d.model.reset();
    }
};

/// The drafter's config against its target, before any weights load: a refusal names what differs.
pub fn admit(a: std.mem.Allocator, io: std.Io, target: *const inkling.Model, dir: []const u8) !cfg.Config {
    const t = &target.config;
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    defer a.free(text);
    return bind(try cfg.parse(a, text), t);
}

/// A parsed draft config bound to the Inkling target `t`: its taps, final norm, softcap and real tokens.
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

fn host(t: tf.checkpoint.Tensor) ck.Tensor {
    var shape: @FieldType(ck.Tensor, "shape") = @splat(1);
    @memcpy(shape[0..t.rank], t.shape[0..t.rank]);
    return .{ .dtype = t.dtype, .rank = @intCast(t.rank), .shape = shape, .bytes = t.buffer.contents()[t.offset..][0..t.bytes] };
}

fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) / 1e9;
}
