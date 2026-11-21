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

pub const Config = struct {
    text: small.Text,
    vision: small.Vision,
    loop: Loop,
    mtp: ?Mtp,
    eos: u32,
    image_token: u32,

    pub fn slots(c: *const Config) u32 {
        return c.loop.slotCount(c.text.layers);
    }

    pub fn isFull(_: *const Config, layer: u32) bool {
        return small.isGlobal(layer);
    }

    pub fn extent(c: *const Config, layer: u32) u32 {
        return c.text.extent(layer);
    }
};

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

fn quantization(o: std.json.ObjectMap, why: *Why) !void {
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
    try quantization(o, why);

    doc.object.getPtr("architectures").?.array.items[0] = .{ .string = "Inkling2ForConditionalGeneration" };
    doc.object.getPtr("model_type").?.* = .{ .string = "inkling2" };
    _ = doc.object.orderedRemove("mtp_config");
    const normalized = try std.json.Stringify.valueAlloc(a, doc, .{});
    const base = try small.parse(gpa, normalized, why);
    return .{
        .text = base.text,
        .vision = base.vision,
        .loop = loop,
        .mtp = mtp,
        .eos = base.eos,
        .image_token = base.image_token,
    };
}
