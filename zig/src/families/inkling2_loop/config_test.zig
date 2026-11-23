//! Config parsing: the pinned Inkling 2 Mini config.json and hf_quant_config.json, and what the loaders refuse.
const std = @import("std");
const config = @import("config.zig");
const quant = @import("quant.zig");

const t = std.testing;
const real = @embedFile("fixtures/config.json");
const hf_quant = @embedFile("fixtures/hf_quant_config.json");
const oq4e = @embedFile("fixtures/oq4e-config.json");

test "oQ4e config admits affine weights and rejects a changed default" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, oq4e, &why);
    try t.expectEqual(@as(u32, 24), c.text.layers);
    try t.expectEqual(config.Checkpoint.mlx, c.checkpoint);
    const edited = try std.mem.replaceOwned(u8, t.allocator, oq4e, "\"group_size\": 64,\n    \"bits\": 4,\n    \"mode\": \"affine\",", "\"group_size\": 32,\n    \"bits\": 4,\n    \"mode\": \"affine\",");
    defer t.allocator.free(edited);
    try t.expectError(error.Quant, config.parse(t.allocator, edited, &why));
}

fn failed(json: []const u8, from: []const u8, to: []const u8, want: anyerror, says: []const u8) !void {
    if (std.mem.count(u8, json, from) != 1) return error.EditAnchor;
    const edited = try std.mem.replaceOwned(u8, t.allocator, json, from, to);
    defer t.allocator.free(edited);
    var why: config.Why = .{};
    try t.expectError(want, config.parse(t.allocator, edited, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), says) != null);
}

test "Inkling 2 Mini's config.json parses to the recipe's values" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, real, &why);
    try t.expectEqual(@as(u32, 2304), c.text.hidden);
    try t.expectEqual(@as(u32, 24), c.text.layers);
    try t.expectEqual(@as(u32, 201024), c.text.vocab);
    try t.expectEqual(@as(u32, 200058), c.text.unpadded_vocab);
    try t.expectEqual(@as(u32, 200006), c.eos);
    try t.expectEqual(@as(u32, 200054), c.image_token);
    try t.expectEqual(config.Heads{ .q = 24, .kv = 8, .dim = 128 }, c.text.global);
    try t.expectEqual(config.Heads{ .q = 24, .kv = 8, .dim = 128 }, c.text.local);
    try t.expectEqual(@as(u32, 1024), c.text.window);
    try t.expectEqual(@as(u32, 16), c.text.d_rel);
    try t.expectEqual(@as(u32, 1024), c.text.rel_extent);
    try t.expectEqual(@as(u32, 8192), c.text.log_floor);
    try t.expectApproxEqAbs(@as(f32, 0.1), c.text.log_alpha, 1e-7);
    try t.expectApproxEqAbs(@as(f32, 1e-6), c.text.eps, 1e-12);
    try t.expectApproxEqAbs(@as(f32, 22.627417), c.text.qk_clip, 1e-5);
    try t.expectEqual(@as(f32, 8), c.text.rel_clip);
    try t.expectEqual(@as(u32, 6), c.text.block);
    try t.expectEqual(@as(u32, 128), c.text.experts);
    try t.expectEqual(@as(u32, 8), c.text.topk);
    try t.expectEqual(@as(u32, 1152), c.text.expert_inter);
    try t.expectEqual(@as(f32, 8), c.text.route_scale);
    try t.expectEqual(@as(f32, 4), c.text.gate_cap);
    try t.expectEqual(@as(f32, 25), c.text.up_cap);
    try t.expectEqual(@as(f32, 30), c.text.logit_cap);
    try t.expectEqual(@as(u32, 1048576), c.text.max_context);
    try t.expectEqualDeep(config.Loop{ .core_start = 6, .core_end = 18, .passes = 3, .gate_rank = 64 }, c.loop);
    try t.expectEqual(@as(u32, 48), c.slots());
    for (0..c.text.layers) |i| {
        try t.expectEqual(i == 5 or i == 11 or i == 17 or i == 23, c.isFull(@intCast(i)));
        try t.expectEqual(@as(u32, 1024), c.extent(@intCast(i)));
    }
    // The mask follows the checkpoint's own `local_layer_ids`, so a layout that is not the published every-sixth one
    // reads its own layer types rather than the shared package's default.
    try t.expect(!c.isFull(0) and !c.isFull(4) and c.isFull(5));
    const m = c.mtp.?;
    try t.expectEqual(@as(u32, 2304), m.hidden);
    try t.expectEqual(@as(u32, 3), m.layers);
    try t.expectEqual(@as(u32, 24), m.q_heads);
    try t.expectEqual(@as(u32, 8), m.kv_heads);
    try t.expectEqual(@as(u32, 128), m.head_dim);
    try t.expectEqual(@as(u32, 9216), m.inter);
    try t.expectEqual(@as(u32, 8), m.block);
    try t.expectEqual(@as(u32, 256), m.rank);
    try t.expectEqualSlices(bool, &.{ true, false, true }, m.local[0..3]);
    try t.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, m.taps[0..m.taps_n]);
    try t.expectEqual(@as(u32, 23), c.vision.layers);
    try t.expectEqual(@as(u32, 1024), c.vision.hidden);
    try t.expectEqual(@as(u32, 16), c.vision.heads);
    try t.expectEqual(@as(u32, 20), c.vision.patch);
    try t.expectEqual(@as(u32, 2), c.vision.merge);
    try t.expectEqual(@as(u32, 32), c.vision.grid);
    try t.expectEqual(@as(u32, 4096), c.vision.proj_hidden);
    try t.expectEqual(@as(u32, 2304), c.vision.text_hidden);
    try t.expectEqual(@as(u32, 4), c.vision.mlp_ratio);
}

test "hf_quant_config.json splits the expert layers 0 to 19 NVFP4 and 20 to 23 MXFP8" {
    var why: config.Why = .{};
    var q = try quant.parse(t.allocator, hf_quant, 24, &why);
    defer q.deinit();
    try t.expectEqual(@as(u32, 0), q.nvfp4.start);
    try t.expectEqual(@as(u32, 20), q.nvfp4.end);
    try t.expectEqual(@as(u32, 40), q.nvfp4.tensors);
    try t.expectEqualSlices(u32, &.{ 1, 16 }, &q.nvfp4.block);
    try t.expectEqual(@as(u32, 16), q.nvfp4.act_block);
    try t.expectEqualStrings("nvfp4_e2m1", q.nvfp4.format);
    try t.expectEqualStrings("float8_e4m3fn", q.nvfp4.scale_dtype);
    try t.expectEqual(@as(u32, 20), q.mxfp8.start);
    try t.expectEqual(@as(u32, 24), q.mxfp8.end);
    try t.expectEqual(@as(u32, 8), q.mxfp8.tensors);
    try t.expectEqualSlices(u32, &.{ 32, 32 }, &q.mxfp8.block);
    try t.expectEqual(@as(u32, 32), q.mxfp8.act_block);
    try t.expectEqualStrings("mxfp8_e4m3", q.mxfp8.format);
    try t.expectEqualStrings("ue8m0", q.mxfp8.scale_dtype);
    try t.expectEqual(@as(u32, 3), q.cast_groups);
    for (0..24) |i| try t.expectEqual(if (i < 20) quant.Kind.nvfp4 else quant.Kind.mxfp8, q.kindOf(@intCast(i)));
}

test "a config that is not Inkling 2 Mini is refused by name" {
    try failed(real, "\"Inkling2LoopForConditionalGeneration\"", "\"Inkling2ForConditionalGeneration\"", error.NotInkling2Loop, "not an Inkling 2 Mini checkpoint");
    try failed(real, "\"model_type\": \"inkling2_loop\"", "\"model_type\": \"llama\"", error.NotInkling2Loop, "not an Inkling 2 Mini checkpoint");
    try failed(real, "\"model_type\": \"inkling2_loop_text\"", "\"model_type\": \"inkling2_text\"", error.NotInkling2Loop, "inkling2_loop_text");
}

test "the loop and the MTP head are refused by key when they are not the ones the engine runs" {
    try failed(real, "\"is_varloop\": false", "\"is_varloop\": true", error.Unsupported, "is_varloop must be false");
    try failed(real, "\"core_end\": 18", "\"core_end\": 30", error.Layout, "do not lie inside 24 layers");
    try failed(real, "\"core_start\": 6", "\"core_start\": 18", error.Layout, "do not lie inside 24 layers");
    try failed(real, "\"gate_rank\": 64", "\"gate_rank\": 100000", error.Layout, "gate_rank");
    try failed(real, "\"loop_config\"", "\"loop_cfg\"", error.BadConfig, "loop_config");
    try failed(real, "\"mtp_config\": {\n    \"hidden_size\": 2304", "\"mtp_config\": {\n    \"hidden_size\": 1024", error.Layout, "backbone's width");
    try failed(real, "\"target_layer_ids\": [\n      5,\n      23,\n      41,\n      48", "\"target_layer_ids\": [\n      5,\n      23,\n      41,\n      49", error.Layout, "outside the 48 slots");
    try failed(real, "\"markov_head_type\": \"rnn_hidden\"", "\"markov_head_type\": \"gru\"", error.Unsupported, "rnn_hidden");
}

test "layouts the engine does not run are refused with the key that differs" {
    try failed(real, "\"local_layer_ids\": [\n      0,\n      1,", "\"local_layer_ids\": [\n      0,\n      25,", error.Layout, "local_layer_ids names a layer the backbone does not have");
    try failed(real, "\"sconv_kernel_size\": 4", "\"sconv_kernel_size\": 3", error.Unsupported, "sconv_kernel_size must be 4");
    try failed(real, "\"hidden_size\": 2304,\n    \"num_hidden_layers\": 24", "\"hidden_size\": 2300,\n    \"num_hidden_layers\": 24", error.Layout, "backbone's width");
    try failed(real, "\"quant_algo\": \"NVFP4\"", "\"quant_algo\": \"FP8\"", error.Quant, "ModelOpt NVFP4");
    try failed(real, "\"eos_token_id\": 200006", "\"eos_token_id\": 300000", error.BadConfig, "eos_token_id");
    try failed(real, "\"use_gate_bias\": true,", "\"use_gate_bias\": true,\n    \"tie_word_embeddings\": true,", error.Unsupported, "tie_word_embeddings");
    try failed(real, "\"n_shared_experts\": 0", "\"n_shared_experts\": 1", error.Unsupported, "n_shared_experts must be 0");
    try failed(real, "\"use_value_norm_no_gamma\": true", "\"use_value_norm_no_gamma\": false", error.Unsupported, "use_value_norm_no_gamma must be true");
}

test "a root that is not JSON or not an object is refused, and a missing key is named" {
    var why: config.Why = .{};
    try t.expectError(error.BadConfig, config.parse(t.allocator, "[]", &why));
    try t.expectEqualStrings("config.json is not an object", why.text());
    try t.expectError(error.BadConfig, config.parse(t.allocator, "{", &why));
    try t.expectEqualStrings("config.json is not JSON", why.text());
    try failed(real, "\"num_hidden_layers\": 24", "\"num_hidden_layer\": 24", error.BadConfig, "num_hidden_layers");
}

test "a quant config that is not the published split is refused by what differs" {
    var why: config.Why = .{};
    if (std.mem.count(u8, hf_quant, "\"weight_transform\": \"mxfp8_nearest\"") != 1) return error.EditAnchor;
    const no_mx = try std.mem.replaceOwned(u8, t.allocator, hf_quant, "\"weight_transform\": \"mxfp8_nearest\"", "\"weight_transform\": \"cast\"");
    defer t.allocator.free(no_mx);
    try t.expectError(error.BadQuantConfig, quant.parse(t.allocator, no_mx, 24, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "no MXFP8") != null);

    if (std.mem.count(u8, hf_quant, "\"block_size\": 16") != 1) return error.EditAnchor;
    const bad_block = try std.mem.replaceOwned(u8, t.allocator, hf_quant, "\"block_size\": 16", "\"block_size\": 32");
    defer t.allocator.free(bad_block);
    try t.expectError(error.Unsupported, quant.parse(t.allocator, bad_block, 24, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "NVFP4 experts must be") != null);

    try t.expectError(error.Layout, quant.parse(t.allocator, hf_quant, 25, &why));
}
