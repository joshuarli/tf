//! The weight map against the pinned Inkling 2 Mini index and safetensors headers, and the geometry refusals.
const std = @import("std");
const core = @import("core");
const st = core.safetensors;
const config = @import("config.zig");
const names = @import("names.zig");
const quant = @import("quant.zig");

const t = std.testing;
const model_header = @embedFile("fixtures/model_header.json");
const mtp_header = @embedFile("fixtures/mtp_header.json");
const index_json = @embedFile("fixtures/index.json");

test "oQ4e index text tensors have family roles" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, @embedFile("fixtures/oq4e-config.json"), &why);
    const idx = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/oq4e-index.json"), .{});
    const map = idx.object.get("weight_map").?.object;
    const roles = try names.roles(a, &c, .model);
    var text_count: usize = 0;
    for (map.keys()) |name| {
        if (std.mem.startsWith(u8, name, "vision_tower.")) continue;
        try t.expect(roles.contains(name));
        text_count += 1;
    }
    try t.expectEqual(text_count, roles.count());
    try t.expectEqual(names.Role.attn_q, roles.get("model.layers.0.attn.in_proj.weight").?);
    try t.expectEqual(names.Role.expert_w13_scale, roles.get("model.layers.20.mlp.experts.w13.biases").?);
    const entries = try header(a, @embedFile("fixtures/oq4e-header.json"));
    const report = try names.check(a, &c, entries, .model);
    if (!report.clean()) {
        std.debug.print("missing {any} wrong {any} unknown {any}\n", .{ report.missing.items, report.wrong.items, report.unknown.items });
    }
    try t.expect(report.clean());
    try names.geometry(&c, entries, &why);
    var bad = try entries.clone(a);
    bad.getPtr("model.layers.20.mlp.experts.w13.biases").?.shape[2] += 1;
    try t.expectError(error.BadGeometry, names.geometry(&c, bad, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "model.layers.20.mlp.experts.w13.biases") != null);
}

fn load(a: std.mem.Allocator) !config.Config {
    var why: config.Why = .{};
    return config.parse(a, @embedFile("fixtures/config.json"), &why);
}

fn header(a: std.mem.Allocator, json: []const u8) !st.Header {
    return st.parseHeader(a, json, std.math.maxInt(usize));
}

test "every tensor in the real index maps to a typed role, and the header holds exactly the index's names" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    const h = try header(a, model_header);
    const idx = try std.json.parseFromSliceLeaky(std.json.Value, a, index_json, .{});
    const map = idx.object.get("weight_map").?.object;
    try t.expectEqual(@as(usize, 1244), map.count());
    try t.expectEqual(@as(usize, 1244), h.count());
    const roles = try names.roles(a, &c, .model);
    for (map.keys()) |name| {
        try t.expect(roles.get(name) != null);
        try t.expect(h.contains(name));
    }
    try t.expectEqual(names.Role.vision_projection, roles.get("model.visual.proj_out.weight").?);
    try t.expectEqual(names.Role.cross_pass_query, roles.get("model.llm.middle_cycle_cross_pass_queries").?);
    try t.expectEqual(names.Role.controller_down, roles.get("model.llm.layers.6.controllers.mlp.down_weight").?);
}

test "the real headers match the map: 1244 model tensors and 54 draft-head tensors, clean" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    const model = try header(a, model_header);
    const r = try names.check(a, &c, model, .model);
    try t.expectEqual(@as(usize, 1244), r.expected);
    try t.expect(r.clean());
    var prec: [config.max_layers]names.Precision = undefined;
    try names.precisions(model, c.text.layers, &prec);
    for (prec[0..24], 0..) |p, i| try t.expectEqual(if (i < 20) names.Precision.nvfp4 else names.Precision.mxfp8, p);
    var why: config.Why = .{};
    try names.geometry(&c, model, &why);
    const mtp = try names.check(a, &c, try header(a, mtp_header), .mtp);
    try t.expectEqual(@as(usize, 54), mtp.expected);
    try t.expect(mtp.clean());
}

test "a corrupted NVFP4 scale or MXFP8 scale fails geometry by tensor name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    const model = try header(a, model_header);

    var nv = try model.clone(a);
    nv.getPtr("model.llm.layers.0.mlp.experts.w2_weight_scale").?.shape[2] = 64;
    var why: config.Why = .{};
    try t.expectError(error.BadGeometry, names.geometry(&c, nv, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "model.llm.layers.0.mlp.experts.w2") != null);
    const nv_report = try names.check(a, &c, nv, .model);
    try t.expect(!nv_report.clean());

    var mx = try model.clone(a);
    mx.getPtr("model.llm.layers.20.mlp.experts.w13_weight_scale").?.shape[2] = 64;
    try t.expectError(error.BadGeometry, names.geometry(&c, mx, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "model.llm.layers.20.mlp.experts.w13") != null);
    const mx_report = try names.check(a, &c, mx, .model);
    try t.expect(!mx_report.clean());
}

test "a missing tensor, a wrong shape, a wrong dtype and an unknown tensor are each reported by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    var kept = try (try header(a, model_header)).clone(a);
    _ = kept.orderedRemove("model.llm.layers.3.attn.wq_du.weight");
    kept.getPtr("model.llm.layers.4.attn.wk_dv.weight").?.shape[0] += 1;
    kept.getPtr("model.llm.layers.2.mlp.gate.bias").?.dtype = .bf16;
    try kept.put(a, "model.llm.layers.0.attn.rope_freqs", kept.values()[0]);
    const r = try names.check(a, &c, kept, .model);
    try t.expectEqualStrings("model.llm.layers.3.attn.wq_du.weight", r.missing.items[0]);
    try t.expectEqual(@as(usize, 1), r.missing.items.len);
    try t.expectEqual(@as(usize, 2), r.wrong.items.len);
    try t.expect(std.mem.indexOf(u8, r.wrong.items[0], "model.llm.layers.2.mlp.gate.bias") != null or
        std.mem.indexOf(u8, r.wrong.items[1], "model.llm.layers.2.mlp.gate.bias") != null);
    try t.expect(std.mem.indexOf(u8, r.wrong.items[0], "model.llm.layers.4.attn.wk_dv.weight") != null or
        std.mem.indexOf(u8, r.wrong.items[1], "model.llm.layers.4.attn.wk_dv.weight") != null);
    try t.expectEqualStrings("model.llm.layers.0.attn.rope_freqs", r.unknown.items[0]);
    try t.expectEqual(@as(usize, 1), r.unknown.items.len);
    try t.expect(!r.clean());
}

test "expert precision follows each layer's w13 tensor and refuses another dtype" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const model = try header(a, model_header);
    var prec: [24]names.Precision = undefined;
    try names.precisions(model, 24, &prec);
    var bad = try model.clone(a);
    bad.getPtr("model.llm.layers.7.mlp.experts.w13_weight").?.dtype = .bf16;
    try t.expectError(error.ExpertFormat, names.precisions(bad, 24, &prec));
    try t.expectError(error.PrecisionCount, names.precisions(model, 24, prec[0..23]));
}

test "MXFP8 values decode through the shared e4m3 table and a ue8m0 exponent" {
    try t.expectEqual(@as(f64, 1), names.mxfp8Value(0x38, 127));
    try t.expectEqual(@as(f64, 2), names.mxfp8Value(0x38, 128));
    try t.expectEqual(@as(f64, -1), names.mxfp8Value(0xc0, 126));
    try t.expectEqual(@as(f64, 0.5), names.mxfp8Value(0x38, 126));
    try t.expectEqual(@as(f64, 0), names.mxfp8Value(0x00, 127));
    try t.expectEqual(@as(f64, 1.5), core.quant.nvfp4.e2m1(3));
    try t.expectEqual(@as(u32, 16), core.quant.nvfp4.group);
}

test "the quant config names the same quantized layers as the headers" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try load(a);
    var why: config.Why = .{};
    var q = try quant.parse(a, @embedFile("fixtures/hf_quant_config.json"), c.text.layers, &why);
    defer q.deinit();
    const model = try header(a, model_header);
    var prec: [config.max_layers]names.Precision = undefined;
    try names.precisions(model, c.text.layers, &prec);
    for (prec[0..c.text.layers], 0..) |p, i| try t.expectEqual(q.kindOf(@intCast(i)).?, p);
}
