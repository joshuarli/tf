//! Inkling 2 Mini's `hf_quant_config.json`: ModelOpt's per-layer NVFP4 and MXFP8 expert groups, checked against the
//! published geometry (NVFP4 a 4-bit code an input in groups of 16, MXFP8 an e4m3 code an input in groups of 32).
const std = @import("std");
const small = @import("inkling2_config");

pub const Why = small.Why;

pub const Kind = enum { nvfp4, mxfp8, affine4, affine8 };

/// One quantized parameter group: its layers, tensor count, stored block shape and activation block.
pub const Group = struct {
    start: u32,
    end: u32, // exclusive
    tensors: u32,
    block: [2]u32,
    act_block: u32,
    format: []const u8,
    scale_dtype: []const u8,

    pub fn span(g: Group) u32 {
        return g.end - g.start;
    }
};

pub const Quant = struct {
    arena: std.heap.ArenaAllocator,
    nvfp4: Group,
    mxfp8: Group,
    cast_groups: u32,

    pub fn deinit(q: *Quant) void {
        q.arena.deinit();
    }

    pub fn kindOf(q: *const Quant, layer: u32) ?Kind {
        if (layer >= q.nvfp4.start and layer < q.nvfp4.end) return .nvfp4;
        if (layer >= q.mxfp8.start and layer < q.mxfp8.end) return .mxfp8;
        return null;
    }
};

fn get(o: std.json.ObjectMap, key: []const u8, why: *Why) !std.json.Value {
    return o.get(key) orelse {
        why.set("hf_quant_config.json has no {s}", .{key});
        return error.BadQuantConfig;
    };
}

fn object(o: std.json.ObjectMap, key: []const u8, why: *Why) !std.json.ObjectMap {
    const v = try get(o, key, why);
    if (v != .object) {
        why.set("hf_quant_config.json's {s} is not an object", .{key});
        return error.BadQuantConfig;
    }
    return v.object;
}

fn string(o: std.json.ObjectMap, key: []const u8, why: *Why) ![]const u8 {
    const v = try get(o, key, why);
    if (v != .string) {
        why.set("hf_quant_config.json's {s} is not a string", .{key});
        return error.BadQuantConfig;
    }
    return v.string;
}

fn whole(v: std.json.Value, what: []const u8, why: *Why) !u32 {
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) {
        why.set("{s} is not a whole number", .{what});
        return error.BadQuantConfig;
    }
    return @intCast(v.integer);
}

fn pair(v: std.json.Value, what: []const u8, why: *Why) ![2]u32 {
    if (v != .array or v.array.items.len != 2) {
        why.set("{s} is not a pair", .{what});
        return error.BadQuantConfig;
    }
    return .{ try whole(v.array.items[0], what, why), try whole(v.array.items[1], what, why) };
}

/// `model.llm.layers.{layer}.mlp.experts.{w13,w2}_weight`; `which` 0 for w13, 1 for w2.
const Tensor = struct { layer: u32, which: u2 };

fn parseTensor(name: []const u8) ?Tensor {
    const prefix = "model.llm.layers.";
    const mid = ".mlp.experts.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const rest = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    const tail = rest[dot..];
    if (!std.mem.startsWith(u8, tail, mid)) return null;
    const what = tail[mid.len..];
    if (std.mem.eql(u8, what, "w13_weight")) return .{ .layer = layer, .which = 0 };
    if (std.mem.eql(u8, what, "w2_weight")) return .{ .layer = layer, .which = 1 };
    return null;
}

fn parseGroup(go: std.json.ObjectMap, kind: Kind, layers: u32, arena: std.mem.Allocator, why: *Why) !Group {
    const tensors = try get(go, "tensors", why);
    if (tensors != .array) {
        why.set("{s} group tensors is not a list", .{@tagName(kind)});
        return error.BadQuantConfig;
    }
    const block = try pair(try get(go, "weight_block_shape", why), "weight_block_shape", why);
    const act = try object(go, "required_activation_quantization", why);
    const act_block = try whole(try get(act, "block_size", why), "required_activation_quantization.block_size", why);
    const format = try string(act, "format", why);
    const scale_dtype = try string(act, "scale_dtype", why);

    if (tensors.array.items.len == 0) {
        why.set("{s} group names no expert projections", .{@tagName(kind)});
        return error.BadQuantConfig;
    }
    var start: u32 = std.math.maxInt(u32);
    var end: u32 = 0;
    var seen = try arena.alloc(bool, @as(usize, layers) * 2);
    @memset(seen, false);
    for (tensors.array.items) |v| {
        if (v != .string) {
            why.set("{s} group names a tensor that is not a string", .{@tagName(kind)});
            return error.BadQuantConfig;
        }
        const t = parseTensor(v.string) orelse {
            why.set("{s} group names {s}, not an expert projection", .{ @tagName(kind), v.string });
            return error.BadQuantConfig;
        };
        if (t.layer >= layers) {
            why.set("{s} group names layer {d} of {d}", .{ @tagName(kind), t.layer, layers });
            return error.Layout;
        }
        const at = @as(usize, t.layer) * 2 + t.which;
        if (seen[at]) {
            why.set("{s} group names {s} twice", .{ @tagName(kind), v.string });
            return error.BadQuantConfig;
        }
        seen[at] = true;
        start = @min(start, t.layer);
        end = @max(end, t.layer + 1);
    }
    for (seen[@as(usize, start) * 2 .. @as(usize, end) * 2]) |found| {
        if (!found) {
            why.set("{s} group skips an expert projection between layers {d} and {d}", .{ @tagName(kind), start, end - 1 });
            return error.BadQuantConfig;
        }
    }
    const count: u32 = @intCast(tensors.array.items.len);
    if (count != 2 * (end - start)) {
        why.set("{s} group holds {d} tensors for {d} layers", .{ @tagName(kind), count, end - start });
        return error.BadQuantConfig;
    }
    return .{
        .start = start,
        .end = end,
        .tensors = count,
        .block = block,
        .act_block = act_block,
        .format = format,
        .scale_dtype = scale_dtype,
    };
}

fn checkNvfp4(g: Group, why: *Why) !void {
    if (g.block[0] != 1 or g.block[1] != 16 or g.act_block != 16 or
        !std.mem.eql(u8, g.format, "nvfp4_e2m1") or !std.mem.eql(u8, g.scale_dtype, "float8_e4m3fn"))
    {
        why.set("NVFP4 experts must be [1,16] blocks of nvfp4_e2m1 codes under float8_e4m3fn scales every 16 inputs", .{});
        return error.Unsupported;
    }
}

fn checkMxfp8(g: Group, why: *Why) !void {
    if (g.block[0] != 32 or g.block[1] != 32 or g.act_block != 32 or
        !std.mem.eql(u8, g.format, "mxfp8_e4m3") or !std.mem.eql(u8, g.scale_dtype, "ue8m0"))
    {
        why.set("MXFP8 experts must be [32,32] blocks of mxfp8_e4m3 codes under ue8m0 scales every 32 inputs", .{});
        return error.Unsupported;
    }
}

/// `hf_quant_config.json` for a `layers`-layer text stack; the groups must partition every expert layer.
pub fn parse(gpa: std.mem.Allocator, json: []const u8, layers: u32, why: *Why) !Quant {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch {
        why.set("hf_quant_config.json is not JSON", .{});
        return error.BadQuantConfig;
    };
    if (doc != .object) {
        why.set("hf_quant_config.json is not an object", .{});
        return error.BadQuantConfig;
    }
    const model = try object(doc.object, "model", why);
    const groups = try get(model, "parameter_groups", why);
    if (groups != .array) {
        why.set("hf_quant_config.json's parameter_groups is not a list", .{});
        return error.BadQuantConfig;
    }
    var nvfp4: ?Group = null;
    var mxfp8: ?Group = null;
    var cast_groups: u32 = 0;
    for (groups.array.items) |v| {
        if (v != .object) {
            why.set("hf_quant_config.json's parameter_groups holds a non-object", .{});
            return error.BadQuantConfig;
        }
        const go = v.object;
        const transform = try string(go, "weight_transform", why);
        if (std.mem.eql(u8, transform, "nvfp4_nearest")) {
            if (nvfp4 != null) {
                why.set("hf_quant_config.json declares NVFP4 experts twice", .{});
                return error.BadQuantConfig;
            }
            nvfp4 = try parseGroup(go, .nvfp4, layers, a, why);
            try checkNvfp4(nvfp4.?, why);
        } else if (std.mem.eql(u8, transform, "mxfp8_nearest")) {
            if (mxfp8 != null) {
                why.set("hf_quant_config.json declares MXFP8 experts twice", .{});
                return error.BadQuantConfig;
            }
            mxfp8 = try parseGroup(go, .mxfp8, layers, a, why);
            try checkMxfp8(mxfp8.?, why);
        } else if (std.mem.eql(u8, transform, "cast")) {
            cast_groups += 1;
        } else {
            why.set("hf_quant_config.json's weight_transform {s} is not one the loader reads", .{transform});
            return error.Unsupported;
        }
    }
    const nv = nvfp4 orelse {
        why.set("hf_quant_config.json declares no NVFP4 expert group", .{});
        return error.BadQuantConfig;
    };
    const mx = mxfp8 orelse {
        why.set("hf_quant_config.json declares no MXFP8 expert group", .{});
        return error.BadQuantConfig;
    };
    if (nv.start != 0 or nv.start >= nv.end or mx.start >= mx.end or nv.end != mx.start or mx.end != layers) {
        why.set("HF_QUANT expert groups do not split the {d} expert layers: NVFP4 {d}..{d}, MXFP8 {d}..{d}", .{ layers, nv.start, nv.end, mx.start, mx.end });
        return error.Layout;
    }
    return .{ .arena = arena, .nvfp4 = nv, .mxfp8 = mx, .cast_groups = cast_groups };
}
