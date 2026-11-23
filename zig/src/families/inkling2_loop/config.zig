//! Inkling 2 Mini (`inkling2_loop`) config.json: the Inkling 2 Small package's text and vision shapes, plus the
//! core loop and the MTP head. The Small package is imported, not copied: this file normalises the two root names
//! its parser checks and removes the MTP block its parser refuses for taps past the physical layers.
const std = @import("std");
const small = @import("inkling2_config");

pub const Why = small.Why;
pub const Heads = small.Heads;
pub const max_layers = small.max_layers;
pub const conv_taps = small.conv_taps;
pub const isGlobal = small.isGlobal;

/// `loop_config`: the layers 6 to 18 run three times, each pass with its own KV and convolution state.
pub const Loop = struct {
    core_start: u32,
    core_end: u32,
    passes: u32,
    gate_rank: u32,

    pub fn slotCount(self: Loop, layers: u32) u32 {
        return layers + (self.passes - 1) * (self.core_end - self.core_start);
    }

    pub fn physicalLayer(self: Loop, layers: u32, slot: u32) u32 {
        if (slot < layers) return slot;
        return self.core_start + (slot - layers) % (self.core_end - self.core_start);
    }
};

/// `mtp_config`: the native draft head, which the text graph never reads. Taps index executed hidden states, so they
/// reach `Loop.slotCount` (48), not the physical layer count.
pub const Mtp = struct {
    hidden: u32,
    layers: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    inter: u32,
    rope_theta: f32,
    window: u32,
    block: u32,
    rank: u32,
    local: [8]bool,
    taps: [16]u32,
    taps_n: u32,
};

pub const Checkpoint = enum { modelopt, mlx };

pub const Config = struct {
    text: small.Text,
    vision: small.Vision,
    loop: Loop,
    mtp: ?Mtp,
    eos: u32,
    image_token: u32,
    checkpoint: Checkpoint = .modelopt,
    /// `layer_types`: which layers attend over the whole context. The checkpoint carries it, and a checkpoint whose
    /// every-other layer is global is not the published every-sixth one the shared package assumes.
    local: [small.max_layers]bool = @splat(false),

    pub fn slots(c: *const Config) u32 {
        return c.loop.slotCount(c.text.layers);
    }

    pub fn isFull(c: *const Config, layer: u32) bool {
        return layer < small.max_layers and !c.local[layer];
    }

    pub fn heads(c: *const Config, layer: u32) small.Heads {
        return if (c.isFull(layer)) c.text.global else c.text.local;
    }

    pub fn extent(c: *const Config, layer: u32) u32 {
        return if (c.isFull(layer)) c.text.rel_extent else c.text.window;
    }
};

/// `layer_types` (or its `local_layer_ids` form) as the mask the graph indexes; absent keeps the published layout.
fn parseLayout(tc: std.json.ObjectMap, layers: u32, out: *[small.max_layers]bool, why: *Why) !void {
    if (tc.get("layer_types")) |v| {
        if (v != .array or v.array.items.len != layers) {
            why.set("layer_types must list all {d} layers", .{layers});
            return error.Layout;
        }
        for (v.array.items, 0..) |e, i| {
            const t = if (e == .string) e.string else {
                why.set("layer_types' entries are not strings", .{});
                return error.Layout;
            };
            if (std.mem.eql(u8, t, "hybrid_sliding")) {
                out[i] = true;
            } else if (std.mem.eql(u8, t, "hybrid")) {
                out[i] = false;
            } else {
                why.set("layer_types names {s}: the engine reads hybrid and hybrid_sliding only", .{t});
                return error.Unsupported;
            }
        }
        return;
    }
    const v = try get(tc, "local_layer_ids", why);
    if (v != .array or v.array.items.len > layers) {
        why.set("local_layer_ids must list at most {d} layers", .{layers});
        return error.Layout;
    }
    for (0..layers) |i| out[i] = !small.isGlobal(@intCast(i));
    for (v.array.items) |e| {
        if (e != .integer or e.integer < 0 or e.integer >= layers) {
            why.set("local_layer_ids names a layer the backbone does not have", .{});
            return error.Layout;
        }
        out[@intCast(e.integer)] = true;
    }
}

/// A checkpoint's expert geometry per layer, read from its own tensor headers rather than from config.json.
pub const Expert = enum { nvfp4, mxfp8 };

/// The text shard's file name, from `model.safetensors.index.json`; the checkpoint splits the text stack from the
/// MTP head, so exactly one file holds it.
pub fn shardNames(a: std.mem.Allocator, body: []const u8) ![][]const u8 {
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, body, .{ .allocate = .alloc_always });
    if (doc != .object) return error.BadIndex;
    const map = doc.object.get("weight_map") orelse return error.BadIndex;
    if (map != .object) return error.BadIndex;
    var found: std.StringHashMapUnmanaged(void) = .empty;
    var it = map.object.iterator();
    while (it.next()) |kv| {
        if (kv.value_ptr.* != .string) return error.BadIndex;
        const f = kv.value_ptr.string;
        if (f.len == 0 or std.mem.indexOf(u8, f, "..") != null or std.fs.path.isAbsolute(f)) return error.BadIndex;
        try found.put(a, f, {});
    }
    if (found.count() == 0) return error.BadIndex;
    var names: std.ArrayList([]const u8) = .empty;
    var keys = found.keyIterator();
    while (keys.next()) |name| try names.append(a, name.*);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    return names.toOwnedSlice(a);
}

pub fn shards(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![][]const u8 {
    const idx = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    defer a.free(idx);
    const body = try std.Io.Dir.cwd().readFileAlloc(io, idx, a, .limited(1 << 26));
    defer a.free(body);
    return shardNames(a, body);
}

pub fn shard(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const files = try shards(arena.allocator(), io, dir);
    if (files.len != 1) return error.MultipleShards;
    return a.dupe(u8, files[0]);
}

/// Each layer's expert format from its `w13_weight` header: packed U8 is NVFP4, F8_E4M3 is MXFP8.
pub fn expertFormats(entries: anytype, layers: u32, out: []Expert) !void {
    if (out.len < layers) return error.FormatCount;
    for (0..layers) |i| {
        var buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.mlp.experts.w13_weight", .{i});
        const e = entries.get(name) orelse return error.NoExperts;
        out[i] = switch (e.dtype) {
            .u8 => .nvfp4,
            .f8_e4m3 => .mxfp8,
            else => return error.ExpertFormat,
        };
    }
}

fn get(o: std.json.ObjectMap, key: []const u8, why: *Why) !std.json.Value {
    return o.get(key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
}

fn object(o: std.json.ObjectMap, key: []const u8, why: *Why) !std.json.ObjectMap {
    const v = try get(o, key, why);
    if (v != .object) {
        why.set("config.json's {s} is not an object", .{key});
        return error.BadConfig;
    }
    return v.object;
}

fn int(o: std.json.ObjectMap, key: []const u8, why: *Why) !u32 {
    const v = try get(o, key, why);
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) {
        why.set("config.json's {s} is not a positive whole number", .{key});
        return error.BadConfig;
    }
    return @intCast(v.integer);
}

fn float(o: std.json.ObjectMap, key: []const u8, why: *Why) !f32 {
    const v = try get(o, key, why);
    const x: f64 = switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
    if (!std.math.isFinite(x) or x <= 0) {
        why.set("config.json's {s} is not a positive number", .{key});
        return error.BadConfig;
    }
    return @floatCast(x);
}

fn text(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn fixed(o: std.json.ObjectMap, key: []const u8, want: bool, why: *Why) !void {
    const v = o.get(key) orelse return;
    if (v == .bool and v.bool == want) return;
    why.set("{s} must be {s}: the engine runs no other Inkling 2 Mini layout", .{ key, if (want) "true" else "false" });
    return error.Unsupported;
}

fn parseLoop(tc: std.json.ObjectMap, layers: u32, hidden: u32, why: *Why) !Loop {
    const o = try object(tc, "loop_config", why);
    const core_start = try int(o, "core_start", why);
    const core_end = try int(o, "core_end", why);
    const passes = try int(o, "num_passes", why);
    const gate_rank = try int(o, "gate_rank", why);
    if (o.get("is_varloop")) |v| {
        if (v != .bool or v.bool) {
            why.set("loop_config.is_varloop must be false: the engine runs the fixed 24-layer loop", .{});
            return error.Unsupported;
        }
    }
    if (core_start >= core_end or core_end > layers) {
        why.set("loop_config cores {d} to {d} do not lie inside {d} layers", .{ core_start, core_end, layers });
        return error.Layout;
    }
    if (gate_rank > 2 * hidden) {
        why.set("loop_config.gate_rank {d} is past the controller tables' width", .{gate_rank});
        return error.Layout;
    }
    return .{ .core_start = core_start, .core_end = core_end, .passes = passes, .gate_rank = gate_rank };
}

fn parseMtp(o: std.json.ObjectMap, text_hidden: u32, slots: u32, why: *Why) !Mtp {
    var m = std.mem.zeroes(Mtp);
    m.hidden = try int(o, "hidden_size", why);
    m.layers = try int(o, "num_hidden_layers", why);
    m.q_heads = try int(o, "num_attention_heads", why);
    m.kv_heads = try int(o, "num_key_value_heads", why);
    m.head_dim = try int(o, "head_dim", why);
    m.inter = try int(o, "intermediate_size", why);
    m.rope_theta = try float(o, "rope_theta", why);
    m.window = try int(o, "sliding_window_size", why);
    m.block = try int(o, "block_size", why);
    m.rank = try int(o, "markov_rank", why);
    if (m.hidden != text_hidden or m.layers > m.local.len or m.q_heads % m.kv_heads != 0) {
        why.set("mtp_config must be the backbone's width, at most {d} layers, with query heads a multiple of key/value heads", .{m.local.len});
        return error.Layout;
    }
    const ids = try get(o, "local_layer_ids", why);
    const taps = try get(o, "target_layer_ids", why);
    if (ids != .array or taps != .array or taps.array.items.len == 0 or taps.array.items.len > m.taps.len) {
        why.set("mtp_config must list its local layers and its taps", .{});
        return error.Layout;
    }
    for (ids.array.items) |e| {
        if (e != .integer or e.integer < 0 or e.integer >= m.layers) {
            why.set("mtp_config's local_layer_ids names a layer the head does not have", .{});
            return error.Layout;
        }
        m.local[@intCast(e.integer)] = true;
    }
    for (taps.array.items) |e| {
        if (e != .integer or e.integer < 0 or e.integer > slots) {
            why.set("mtp_config's target_layer_ids names an executed hidden state outside the {d} slots", .{slots});
            return error.Layout;
        }
        m.taps[m.taps_n] = @intCast(e.integer);
        m.taps_n += 1;
    }
    if (!std.mem.eql(u8, text(o, "markov_head_type") orelse "", "rnn_hidden")) {
        why.set("mtp_config.markov_head_type must be \"rnn_hidden\"", .{});
        return error.Unsupported;
    }
    try fixed(o, "q_bias", false, why);
    try fixed(o, "o_bias", false, why);
    return m;
}

fn mlxWidth(o: std.json.ObjectMap, bits: i64, group: i64, mode: []const u8) bool {
    const b = o.get("bits") orelse return false;
    const g = o.get("group_size") orelse return false;
    return b == .integer and b.integer == bits and g == .integer and g.integer == group and
        std.mem.eql(u8, text(o, "mode") orelse "", mode);
}

fn mlxEntry(o: std.json.ObjectMap, key: []const u8, bits: i64, group: i64, mode: []const u8, why: *Why) !void {
    const entry = o.get(key) orelse {
        why.set("quantization has no {s} override", .{key});
        return error.Quant;
    };
    if (entry != .object or !mlxWidth(entry.object, bits, group, mode)) {
        why.set("quantization override {s} has an unsupported format", .{key});
        return error.Quant;
    }
}

fn quantization(o: std.json.ObjectMap, layers: u32, local: *const [small.max_layers]bool, why: *Why) !Checkpoint {
    if (o.get("quantization")) |v| {
        if (v != .object or !mlxWidth(v.object, 4, 64, "affine")) {
            why.set("quantization must default to affine 4-bit group 64", .{});
            return error.Quant;
        }
        const q = v.object;
        try mlxEntry(q, "model.embed", 8, 64, "affine", why);
        try mlxEntry(q, "lm_head", 8, 64, "affine", why);
        if (layers != 24) {
            why.set("quantization requires 24 layers", .{});
            return error.Quant;
        }
        var expected: usize = 5;
        for (0..layers) |i| {
            var key: [128]u8 = undefined;
            const projections: []const []const u8 = if (local[i]) &.{ "in_proj", "o_proj" } else &.{ "q_proj", "k_proj", "v_proj", "o_proj" };
            for (projections) |p| {
                const name = try std.fmt.bufPrint(&key, "model.layers.{d}.attn.{s}", .{ i, p });
                try mlxEntry(q, name, 8, 64, "affine", why);
                expected += 1;
            }
            if (i < 20) for ([_][]const u8{ "w13", "w2" }) |p| {
                const name = try std.fmt.bufPrint(&key, "model.layers.{d}.mlp.experts.{s}", .{ i, p });
                try mlxEntry(q, name, 4, 16, "nvfp4", why);
                expected += 1;
            };
        }
        if (q.count() != expected) {
            why.set("quantization has unknown overrides", .{});
            return error.Quant;
        }
        return .mlx;
    }
    const v = o.get("quantization_config") orelse {
        why.set("config.json has no ModelOpt quantization_config", .{});
        return error.Quant;
    };
    if (v != .object) {
        why.set("quantization_config is not an object", .{});
        return error.Quant;
    }
    const q = v.object;
    const ok = std.ascii.eqlIgnoreCase(text(q, "quant_method") orelse "", "modelopt") and
        std.ascii.eqlIgnoreCase(text(q, "quant_algo") orelse "", "NVFP4");
    if (!ok) {
        why.set("quantization_config: the loader reads ModelOpt NVFP4 experts (MXFP8 layers follow from their tensors)", .{});
        return error.Quant;
    }
    return .modelopt;
}

/// config.json as the reference and the checkpoint read it; a refusal is in `why`.
pub fn parse(gpa: std.mem.Allocator, json: []const u8, why: *Why) !Config {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var doc = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch {
        why.set("config.json is not JSON", .{});
        return error.BadConfig;
    };
    if (doc != .object) {
        why.set("config.json is not an object", .{});
        return error.BadConfig;
    }
    const o = doc.object;
    const arch = o.get("architectures");
    const arch_ok = arch != null and arch.? == .array and arch.?.array.items.len == 1 and
        arch.?.array.items[0] == .string and
        std.mem.eql(u8, arch.?.array.items[0].string, "Inkling2LoopForConditionalGeneration");
    if (!arch_ok or !std.mem.eql(u8, text(o, "model_type") orelse "", "inkling2_loop")) {
        why.set("not an Inkling 2 Mini checkpoint: architectures Inkling2LoopForConditionalGeneration, model_type inkling2_loop", .{});
        return error.NotInkling2Loop;
    }
    const tc = try object(o, "text_config", why);
    if (text(tc, "model_type")) |mt| {
        if (!std.mem.eql(u8, mt, "inkling2_loop_text")) {
            why.set("text_config.model_type must be \"inkling2_loop_text\"", .{});
            return error.NotInkling2Loop;
        }
    }
    const layers = try int(tc, "num_hidden_layers", why);
    const hidden = try int(tc, "hidden_size", why);
    const loop = try parseLoop(tc, layers, hidden, why);
    const slots = loop.slotCount(layers);
    const mtp = if (o.get("mtp_config") != null) try parseMtp(try object(o, "mtp_config", why), hidden, slots, why) else null;
    if (o.get("tie_word_embeddings")) |v| {
        if (v != .bool or v.bool) {
            why.set("tie_word_embeddings must be false: the checkpoint stores a separate unembed", .{});
            return error.Unsupported;
        }
    }
    var layout: [small.max_layers]bool = @splat(false);
    try parseLayout(tc, layers, &layout, why);
    const checkpoint = try quantization(o, layers, &layout, why);

    doc.object.getPtr("architectures").?.array.items[0] = .{ .string = "Inkling2ForConditionalGeneration" };
    doc.object.getPtr("model_type").?.* = .{ .string = "inkling2" };
    _ = doc.object.orderedRemove("mtp_config");
    if (checkpoint == .mlx) _ = doc.object.orderedRemove("quantization_config");
    // The shared package checks `local_layer_ids` against its own every-sixth layout. Only its shapes are used here:
    // `parseLayout` above holds the checkpoint's real mask, and every layout question goes through `Config`.
    const base_tc = &doc.object.getPtr("text_config").?.object;
    var published: std.json.Array = .init(a);
    for (0..layers) |i| if (!small.isGlobal(@intCast(i))) try published.append(.{ .integer = @intCast(i) });
    base_tc.put(a, "local_layer_ids", .{ .array = published }) catch |e| {
        why.set("config.json's text_config is not writable", .{});
        return e;
    };
    const normalized = try std.json.Stringify.valueAlloc(a, doc, .{});
    const base = try small.parse(gpa, normalized, why);
    return .{
        .text = base.text,
        .vision = base.vision,
        .loop = loop,
        .mtp = mtp,
        .eos = base.eos,
        .image_token = base.image_token,
        .checkpoint = checkpoint,
        .local = layout,
    };
}
