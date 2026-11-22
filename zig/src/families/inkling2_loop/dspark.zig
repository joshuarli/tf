const std = @import("std");
const mtl = @import("metal");
const st = @import("core").safetensors;
const model = @import("model.zig");

const Tensor = model.Tensor;

pub const Weights = struct {
    arena: std.heap.ArenaAllocator,
    map: mtl.MappedFile,
    buffer: mtl.Buffer,
    data: usize,
    header: st.Header,

    pub fn init(m: *const model.Model, dir: []const u8) !Weights {
        var arena = std.heap.ArenaAllocator.init(m.gpa);
        errdefer arena.deinit();
        const path = try std.fs.path.joinZ(arena.allocator(), &.{ dir, "mtp.safetensors" });
        const map = try mtl.MappedFile.open(path.ptr);
        errdefer map.deinit();
        if (map.size < 8) return error.BadSafetensors;
        const n = try st.headerSize(std.mem.readInt(u64, map.bytes[0..8], .little), map.size);
        const data = 8 + n;
        const header = try st.parseHeader(arena.allocator(), map.bytes[8..data], map.size - data);
        const buffer = try m.device.bufferNoCopy(map.bytes.ptr, map.bytes.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        return .{ .arena = arena, .map = map, .buffer = buffer, .data = data, .header = header };
    }

    pub fn deinit(w: *Weights) void {
        w.buffer.deinit();
        w.map.deinit();
        w.arena.deinit();
    }

    pub fn get(w: *const Weights, name: []const u8, dims: []const usize) !Tensor {
        const e = w.header.get(name) orelse return error.MissingTensor;
        if (e.dtype != .bf16 or e.rank != dims.len) return error.BadTensor;
        for (dims, e.shape[0..e.rank]) |want, actual| if (want != actual) return error.BadTensor;
        var result = Tensor{ .buffer = w.buffer, .offset = w.data + e.begin, .bytes = e.end - e.begin, .dtype = e.dtype, .rank = e.rank };
        for (0..e.rank) |i| result.shape[i] = e.shape[i];
        return result;
    }
};

const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    context_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    context_k: Tensor,
    context_v: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    context_k_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
};

const Names = enum {
    matrix, norm, mix, add, copy, silu, rope, scores, attention, anchor, rnn_input, recur, bias, logits, argmax,
};

const names = [_][:0]const u8{
    "ds_matrix", "ds_norm", "ds_mix", "ds_add", "ds_copy", "ds_silu", "ds_rope", "ds_scores", "ds_attention",
    "ds_anchor", "ds_rnn_input", "ds_recur", "ds_bias", "ds_logits", "ds_argmax",
};

const Matrix = extern struct { rows: u32, inner: u32, count: u32 };
const Norm = extern struct { width: u32, rows: u32, epsilon: f32 };
const Mix = extern struct { rows: u32, width: u32 };
const Rope = extern struct { rows: u32, heads: u32, dim: u32, position: u32, theta: f32 };
const Attention = extern struct { context: u32, heads: u32, kv_heads: u32, dim: u32, window: u32 };
const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };

fn ref(buffer: mtl.Buffer) Ref { return .{ .buffer = buffer }; }
fn tensor(t: Tensor) Ref { return .{ .buffer = t.buffer, .offset = t.offset }; }
fn at(buffer: mtl.Buffer, offset: usize) Ref { return .{ .buffer = buffer, .offset = offset }; }

const Buffers = struct {
    device: mtl.Device,
    items: [20]mtl.Buffer = undefined,
    len: usize = 0,

    fn get(b: *Buffers, n: usize) !mtl.Buffer {
        const buffer = try b.device.buffer(n * 4, mtl.ResourceOptions.shared);
        b.items[b.len] = buffer;
        b.len += 1;
        return buffer;
    }
    fn deinit(b: *Buffers) void {
        for (b.items[0..b.len]) |buffer| buffer.deinit();
    }
};

pub const Drafter = struct {
    m: *const model.Model,
    sidecar: *const Weights,
    layers: [3]Layer,
    fuse: Tensor,
    mask: Tensor,
    norm: Tensor,
    rnn_norm: Tensor,
    lookup: Tensor,
    rnn: Tensor,
    rnn_bias: Tensor,
    hidden_out: Tensor,
    hidden_bias: Tensor,
    pipes: [names.len]mtl.Pipeline,
    x: mtl.Buffer,
    residual: mtl.Buffer,
    normed: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    attn: mtl.Buffer,
    projected: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    f_rnn: mtl.Buffer,
    input: mtl.Buffer,
    rnn_result: mtl.Buffer,
    recurrent: mtl.Buffer,
    latent: mtl.Buffer,
    correction: mtl.Buffer,
    base: mtl.Buffer,
    head: mtl.Buffer,
    logits: mtl.Buffer,
    tokens: mtl.Buffer,
    keys: [3]?mtl.Buffer = @splat(null),
    values: [3]?mtl.Buffer = @splat(null),
    length: usize = 0,
    capacity: usize = 0,

    pub fn init(m: *const model.Model, w: *Weights) !Drafter {
        const c = m.config.mtp orelse return error.MissingMtp;
        if (c.hidden != 2304 or c.layers != 3 or c.q_heads != 24 or c.kv_heads != 8 or
            c.head_dim != 128 or c.inter != 9216 or c.rank != 256 or c.block != 8 or c.taps_n != 4)
            return error.UnsupportedGeometry;
        var layers: [3]Layer = undefined;
        for (&layers, 0..) |*l, i| {
            const a = w.arena.allocator();
            const prefix = try std.fmt.allocPrint(a, "layers.{d}.", .{i});
            const get = struct {
                fn field(weights: *const Weights, alloc: std.mem.Allocator, p: []const u8, n: []const u8, dims: []const usize) !Tensor {
                    const name = try std.fmt.allocPrint(alloc, "{s}{s}", .{ p, n });
                    return weights.get(name, dims);
                }
            }.field;
            l.* = .{
                .input_norm = try get(w, a, prefix, "input_layernorm.weight", &.{2304}),
                .post_norm = try get(w, a, prefix, "post_attention_layernorm.weight", &.{2304}),
                .context_norm = try get(w, a, prefix, "dflare_norm.weight", &.{2304}),
                .q = try get(w, a, prefix, "self_attn.q_proj.weight", &.{ 3072, 2304 }),
                .k = try get(w, a, prefix, "self_attn.k_proj.weight", &.{ 1024, 2304 }),
                .v = try get(w, a, prefix, "self_attn.v_proj.weight", &.{ 1024, 2304 }),
                .context_k = try get(w, a, prefix, "self_attn.k_proj_ctx.weight", &.{ 1024, 2304 }),
                .context_v = try get(w, a, prefix, "self_attn.v_proj_ctx.weight", &.{ 1024, 2304 }),
                .o = try get(w, a, prefix, "self_attn.o_proj.weight", &.{ 2304, 3072 }),
                .q_norm = try get(w, a, prefix, "self_attn.q_norm.weight", &.{128}),
                .k_norm = try get(w, a, prefix, "self_attn.k_norm.weight", &.{128}),
                .context_k_norm = try get(w, a, prefix, "self_attn.k_norm_ctx.weight", &.{128}),
                .gate = try get(w, a, prefix, "mlp.gate_proj.weight", &.{ 9216, 2304 }),
                .up = try get(w, a, prefix, "mlp.up_proj.weight", &.{ 9216, 2304 }),
                .down = try get(w, a, prefix, "mlp.down_proj.weight", &.{ 2304, 9216 }),
            };
        }
        const lib = try mtl.Library.fromSource(m.device, @import("kernels.zig").dspark, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [names.len]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |p| p.deinit();
        for (names, 0..) |name, i| { pipes[i] = try mtl.Pipeline.init(m.device, lib, name, false); made += 1; }
        var allocation = Buffers{ .device = m.device };
        errdefer allocation.deinit();
        return .{
            .m = m, .sidecar = w, .layers = layers, .pipes = pipes,
            .fuse = try w.get("fuse_weight", &.{ 4, 4 }), .mask = try w.get("mask_embed", &.{2304}),
            .norm = try w.get("norm.weight", &.{2304}), .rnn_norm = try w.get("markov_head.dflare_norm.weight", &.{2304}),
            .lookup = try w.get("markov_head.markov_w1.weight", &.{ m.config.text.vocab, 256 }),
            .rnn = try w.get("markov_head.rnn_proj.weight", &.{ 768, 5120 }),
            .rnn_bias = try w.get("markov_head.rnn_proj.bias", &.{768}),
            .hidden_out = try w.get("markov_head.hidden_out.weight", &.{ 2304, 256 }),
            .hidden_bias = try w.get("markov_head.hidden_out.bias", &.{2304}),
            .x = try allocation.get(8 * 2304), .residual = try allocation.get(8 * 2304),
            .normed = try allocation.get(8 * 2304), .q = try allocation.get(8 * 3072),
            .k = try allocation.get(8 * 1024), .v = try allocation.get(8 * 1024),
            .attn = try allocation.get(8 * 3072), .projected = try allocation.get(8 * 2304),
            .gate = try allocation.get(8 * 9216), .up = try allocation.get(8 * 9216),
            .f_rnn = try allocation.get(2304), .input = try allocation.get(5120),
            .rnn_result = try allocation.get(768), .recurrent = try allocation.get(256),
            .latent = try allocation.get(256), .correction = try allocation.get(2304),
            .base = try allocation.get(7 * m.config.text.unpadded_vocab),
            .head = try allocation.get(m.config.text.unpadded_vocab),
            .logits = try allocation.get(7 * m.config.text.unpadded_vocab), .tokens = try allocation.get(8),
        };
    }

    pub fn deinit(d: *Drafter) void {
        for (d.pipes) |p| p.deinit();
        for (d.keys) |k| if (k) |v| v.deinit();
        for (d.values) |k| if (k) |v| v.deinit();
        inline for (.{ "x", "residual", "normed", "q", "k", "v", "attn", "projected", "gate", "up", "f_rnn", "input", "rnn_result", "recurrent", "latent", "correction", "base", "head", "logits", "tokens" }) |name|
            @field(d, name).deinit();
    }

    pub fn reset(d: *Drafter) void { d.length = 0; }

    fn reserve(d: *Drafter, size: usize) !void {
        if (size <= d.capacity) return;
        var keys: [3]?mtl.Buffer = @splat(null);
        var values: [3]?mtl.Buffer = @splat(null);
        errdefer {
            for (keys) |entry| if (entry) |v| v.deinit();
            for (values) |entry| if (entry) |v| v.deinit();
        }
        const capacity = std.math.ceilPowerOfTwo(usize, size) catch return error.ContextTooLong;
        const bytes = try std.math.mul(usize, capacity, 1024 * 4);
        for (0..3) |i| {
            keys[i] = try d.m.device.buffer(bytes, mtl.ResourceOptions.shared);
            values[i] = try d.m.device.buffer(bytes, mtl.ResourceOptions.shared);
            if (d.length > 0) {
                @memcpy(keys[i].?.contents()[0 .. d.length * 1024 * 4], d.keys[i].?.contents()[0 .. d.length * 1024 * 4]);
                @memcpy(values[i].?.contents()[0 .. d.length * 1024 * 4], d.values[i].?.contents()[0 .. d.length * 1024 * 4]);
            }
        }
        for (d.keys) |entry| if (entry) |v| v.deinit();
        for (d.values) |entry| if (entry) |v| v.deinit();
        d.keys = keys;
        d.values = values;
        d.capacity = capacity;
    }

    fn bind(d: *Drafter, e: mtl.ComputeEncoder, name: Names, refs: []const Ref) void {
        e.setPipeline(d.pipes[@intFromEnum(name)]);
        for (refs, 0..) |b, i| e.setBuffer(b.buffer, b.offset, i);
    }

    fn flat(d: *Drafter, e: mtl.ComputeEncoder, name: Names, refs: []const Ref, count: usize) void {
        d.bind(e, name, refs);
        e.setValue(@as(u32, @intCast(count)), refs.len);
        e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn matrix(d: *Drafter, e: mtl.ComputeEncoder, input: Ref, weight: Tensor, output: Ref, rows: u32) void {
        d.bind(e, .matrix, &.{ input, tensor(weight), output });
        const args = Matrix{ .rows = @intCast(weight.shape[0]), .inner = @intCast(weight.shape[1]), .count = rows };
        e.setValue(args, 3);
        e.dispatchGroups(mtl.Size.of((args.rows + 7) / 8, rows, 1), mtl.Size.of(256, 1, 1));
    }

    fn headMatrix(d: *Drafter, e: mtl.ComputeEncoder, input: Ref, output: Ref, rows: u32) void {
        var weight = d.m.weights.unembed;
        weight.shape[0] = d.m.config.text.unpadded_vocab;
        d.matrix(e, input, weight, output, rows);
    }

    fn normRows(d: *Drafter, e: mtl.ComputeEncoder, x: Ref, weight: Tensor, y: Ref, rows: u32, width: u32) void {
        d.bind(e, .norm, &.{ x, tensor(weight), y });
        e.setValue(Norm{ .width = width, .rows = rows, .epsilon = d.m.config.text.eps }, 3);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn mixRows(d: *Drafter, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: u32, stage: usize) void {
        d.bind(e, .mix, &.{ x, .{ .buffer = d.fuse.buffer, .offset = d.fuse.offset + stage * 8 }, y });
        e.setValue(Mix{ .rows = rows, .width = 2304 }, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, rows) * 2304, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn rope(d: *Drafter, e: mtl.ComputeEncoder, x: Ref, rows: u32, heads: u32, position: u32) void {
        d.bind(e, .rope, &.{x});
        e.setValue(Rope{ .rows = rows, .heads = heads, .dim = 128, .position = position, .theta = d.m.config.mtp.?.rope_theta }, 1);
        e.dispatchThreads(mtl.Size.of(@as(usize, rows) * heads * 64, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn propose(d: *Drafter, taps: []const f32, anchor: u32, width: u32) ![]const u32 {
        if (taps.len == 0 or taps.len % (4 * 2304) != 0 or width < 2 or width > 8 or anchor >= d.m.config.text.vocab)
            return error.InvalidInput;
        const rows = taps.len / (4 * 2304);
        const length = try std.math.add(usize, d.length, rows);
        if (length > std.math.maxInt(u32)) return error.ContextTooLong;
        try d.reserve(length);
        const tap_buffer = try d.m.device.buffer(taps.len * 4, mtl.ResourceOptions.shared);
        defer tap_buffer.deinit();
        @memcpy(tap_buffer.slice(f32, taps.len), taps);
        const mixed = try d.m.device.buffer(rows * 2304 * 4, mtl.ResourceOptions.shared);
        defer mixed.deinit();
        const normalized = try d.m.device.buffer(rows * 2304 * 4, mtl.ResourceOptions.shared);
        defer normalized.deinit();
        const score = try d.m.device.buffer(8 * 24 * (length + 8) * 4, mtl.ResourceOptions.shared);
        defer score.deinit();
        @memset(d.recurrent.slice(f32, 256), 0);
        d.tokens.slice(u32, 8)[0] = anchor;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.m.queue.commandBuffer();
        const e = cb.compute(.serial);
        d.bind(e, .anchor, &.{ tensor(d.m.weights.embed), tensor(d.mask), ref(d.tokens), ref(d.x) });
        e.setValue(@as(u32, 2304), 4);
        e.dispatchThreads(mtl.Size.of(8 * 2304, 1, 1), mtl.Size.of(256, 1, 1));
        d.normRows(e, ref(d.x), d.m.weights.embed_norm, ref(d.x), 1, 2304);
        d.mixRows(e, at(tap_buffer, (rows - 1) * 4 * 2304 * 4), ref(d.f_rnn), 1, 3);
        d.normRows(e, ref(d.f_rnn), d.rnn_norm, ref(d.f_rnn), 1, 2304);
        for (d.layers, 0..) |l, index| {
            d.mixRows(e, ref(tap_buffer), ref(mixed), @intCast(rows), index);
            d.normRows(e, ref(mixed), l.context_norm, ref(normalized), @intCast(rows), 2304);
            const key = at(d.keys[index].?, d.length * 1024 * 4);
            const value = at(d.values[index].?, d.length * 1024 * 4);
            d.matrix(e, ref(normalized), l.context_k, key, @intCast(rows));
            d.matrix(e, ref(normalized), l.context_v, value, @intCast(rows));
            d.normRows(e, key, l.context_k_norm, key, @intCast(rows * 8), 128);
            d.rope(e, key, @intCast(rows), 8, @intCast(d.length));
            if (index == 0) d.flat(e, .copy, &.{ ref(d.x), ref(d.residual) }, 8 * 2304)
            else d.flat(e, .add, &.{ ref(d.x), ref(d.residual), ref(d.residual) }, 8 * 2304);
            d.normRows(e, ref(d.residual), l.input_norm, ref(d.normed), 8, 2304);
            d.matrix(e, ref(d.normed), l.q, ref(d.q), 8);
            d.matrix(e, ref(d.normed), l.k, ref(d.k), 8);
            d.matrix(e, ref(d.normed), l.v, ref(d.v), 8);
            d.normRows(e, ref(d.q), l.q_norm, ref(d.q), 8 * 24, 128);
            d.normRows(e, ref(d.k), l.k_norm, ref(d.k), 8 * 8, 128);
            d.rope(e, ref(d.q), 8, 24, @intCast(length));
            d.rope(e, ref(d.k), 8, 8, @intCast(length));
            const args = Attention{ .context = @intCast(length), .heads = 24, .kv_heads = 8, .dim = 128,
                .window = if (d.m.config.mtp.?.local[index]) d.m.config.mtp.?.window else 0 };
            d.bind(e, .scores, &.{ ref(d.q), ref(d.keys[index].?), ref(d.k), ref(score) });
            e.setValue(args, 4);
            e.dispatchThreads(mtl.Size.of(8, 24, length + 8), mtl.Size.of(8, 1, 1));
            d.bind(e, .attention, &.{ ref(score), ref(d.values[index].?), ref(d.v), ref(d.attn) });
            e.setValue(args, 4);
            e.dispatchThreads(mtl.Size.of(8, 24, 128), mtl.Size.of(8, 1, 1));
            d.matrix(e, ref(d.attn), l.o, ref(d.projected), 8);
            d.flat(e, .add, &.{ ref(d.projected), ref(d.residual), ref(d.residual) }, 8 * 2304);
            d.normRows(e, ref(d.residual), l.post_norm, ref(d.normed), 8, 2304);
            d.matrix(e, ref(d.normed), l.gate, ref(d.gate), 8);
            d.matrix(e, ref(d.normed), l.up, ref(d.up), 8);
            d.flat(e, .silu, &.{ ref(d.gate), ref(d.up), ref(d.gate) }, 8 * 9216);
            d.matrix(e, ref(d.gate), l.down, ref(d.x), 8);
        }
        d.flat(e, .add, &.{ ref(d.x), ref(d.residual), ref(d.x) }, 8 * 2304);
        d.normRows(e, ref(d.x), d.norm, ref(d.x), 8, 2304);
        d.headMatrix(e, at(d.x, 2304 * 4), ref(d.base), width - 1);
        const vocab = d.m.config.text.unpadded_vocab;
        for (0..width - 1) |i| {
            d.bind(e, .rnn_input, &.{ ref(d.recurrent), at(d.tokens, i * 4), tensor(d.lookup), ref(d.x), ref(d.f_rnn), ref(d.input) });
            e.setValue([2]u32{ @intCast(i + 1), 0 }, 6);
            e.dispatchThreads(mtl.Size.of(5120, 1, 1), mtl.Size.of(256, 1, 1));
            d.matrix(e, ref(d.input), d.rnn, ref(d.rnn_result), 1);
            d.bind(e, .recur, &.{ ref(d.rnn_result), tensor(d.rnn_bias), ref(d.recurrent), ref(d.latent) });
            e.dispatchThreads(mtl.Size.of(256, 1, 1), mtl.Size.of(256, 1, 1));
            d.matrix(e, ref(d.latent), d.hidden_out, ref(d.correction), 1);
            d.flat(e, .bias, &.{ ref(d.correction), tensor(d.hidden_bias) }, 2304);
            d.headMatrix(e, ref(d.correction), ref(d.head), 1);
            const offset = i * @as(usize, vocab) * 4;
            d.flat(e, .logits, &.{ at(d.base, offset), ref(d.head), at(d.logits, offset) }, vocab);
            d.bind(e, .argmax, &.{ at(d.logits, offset), at(d.tokens, (i + 1) * 4) });
            e.setValue(vocab, 2);
            e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("dspark: {s}", .{message});
            return error.GpuFailed;
        }
        d.length = length;
        return d.tokens.slice(u32, 8)[1..width];
    }
};
