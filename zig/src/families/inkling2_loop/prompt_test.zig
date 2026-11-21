//! The checkpoint's chat template and tokenizer against the reference histories' pinned token ids.
const std = @import("std");
const template = @import("template");
const tokenizer = @import("tokenizer");
const prompt = @import("prompt.zig");

const t = std.testing;
const jinja = @embedFile("fixtures/chat_template.jinja");
const histories = @embedFile("fixtures/histories.json");

/// `$TF_INKLING2_LOOP_CKPT`, else the campaign's pinned checkpoint under $HOME; `null` when the tokenizer is absent.
fn checkpoint(a: std.mem.Allocator) ![]const u8 {
    if (std.c.getenv("TF_INKLING2_LOOP_CKPT")) |p| return std.mem.span(p);
    const home = std.c.getenv("HOME") orelse return error.SkipZigTest;
    return try std.fmt.allocPrint(a, "{s}/models/Inkling-2-Mini@ed40e7da625f", .{std.mem.span(home)});
}

test "the pinned chat template and tokenizer reproduce campaign/ref/histories.json token ids" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try checkpoint(a);
    var tok = tokenizer.loadTokenizer(t.io, t.allocator, dir) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer tok.deinit();
    var compiled = try template.compile(t.allocator, jinja, null);
    defer compiled.deinit();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, histories, .{});
    try t.expectEqual(@as(usize, 4), doc.array.items.len);
    for (doc.array.items) |entry| {
        const o = entry.object;
        const text = o.get("text").?.string;
        const want = o.get("token_ids").?.array.items;
        const rendered = try prompt.render(a, &compiled, .{ .user = text }, null);
        const got = try prompt.encode(a, &tok, rendered);
        try t.expectEqual(want.len, got.len);
        for (want, got) |x, id| try t.expectEqual(@as(u32, @intCast(x.integer)), id);
    }
}

test "the Mini template writes the default thinking effort before the first user turn" {
    var compiled = try template.compile(t.allocator, jinja, null);
    defer compiled.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const rendered = try prompt.render(arena.allocator(), &compiled, .{ .user = "Hi" }, null);
    try t.expectEqualStrings(
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>" ++
            "<|message_user|><|content_text|>Hi<|end_message|>",
        rendered,
    );
}
