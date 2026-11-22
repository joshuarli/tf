const std = @import("std");
const inkling = @import("inkling_reply.zig");
const json = @import("json");
const tool_parse = @import("tool_parse.zig");

const reply = "<|message_model|><|content_thinking|>Think first.<|end_message|>" ++
    "<|message_model|><|content_text|>Reading.<|end_message|>" ++
    "<|message_model|>read<|content_invoke_tool_json|>{\"name\":\"read\",\"args\":{\"path\":\"a\"}}<|end_message|>" ++
    "<|message_model|>read<|content_invoke_tool_json|>{\"name\":\"read\",\"args\":{\"path\":\"b\"}}<|end_message|>";

test "Inkling reply splits completed messages and holds incomplete calls" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var reading: inkling.Reading = .{};
    const done = try reading.split(a, reply, .ended);
    try std.testing.expectEqualStrings("Think first.", done.reasoning);
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{\"type\":\"object\"}}}]")).ok.array;
    const parsed = try tool_parse.parse(a, done.answer, tools, null);
    try std.testing.expectEqualStrings("Reading.", parsed.content);
    try std.testing.expectEqual(@as(usize, 2), parsed.calls.?.len);
    try std.testing.expectEqualStrings("{\"path\":\"a\"}", parsed.calls.?[0].get("function").?.strField("arguments").?);
    for (1..reply.len + 1) |n| {
        const part = try reading.split(a, reply[0..n], .streaming);
        try std.testing.expect(std.mem.startsWith(u8, done.reasoning, part.reasoning));
        try std.testing.expect(std.mem.startsWith(u8, done.answer, part.answer));
        try std.testing.expect(std.mem.indexOf(u8, part.answer, "<|content_") == null);
    }
    const cut = std.mem.indexOf(u8, reply, "{\"path\":\"b\"").? + 5;
    const truncated = try reading.split(a, reply[0..cut], .cut);
    const calls = try tool_parse.parse(a, truncated.answer, tools, null);
    try std.testing.expectEqual(@as(usize, 1), calls.calls.?.len);
}

test "Inkling checkpoint template and tokenizer round trip assistant calls" {
    const template = @import("template");
    const tokenizer = @import("tokenizer");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const source = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/src/families/inkling2_loop/fixtures/chat_template.jinja", a, .limited(1 << 20));
    var compiled = try template.compile(std.testing.allocator, source, null);
    defer compiled.deinit();
    const messages = try std.json.parseFromSliceLeaky(std.json.Value, a, "[{\"role\":\"assistant\",\"reasoning_content\":\"Think first.\",\"content\":\"Reading.\",\"tool_calls\":[{\"function\":{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}},{\"function\":{\"name\":\"read\",\"arguments\":{\"path\":\"b\"}}}]}]", .{});
    const rendered = try template.render(a, &compiled, messages, .null, .null, false, null);
    var tok = try tokenizer.loadTokenizer(std.testing.io, std.testing.allocator, "/Users/josh/models/Inkling-2-Mini@ed40e7da625f");
    defer tok.deinit();
    const ids = try tok.encode(a, rendered);
    const decoded = try tok.decode(a, ids, false);
    try std.testing.expectEqualStrings(rendered, decoded);
    const Text = struct {
        tok: *const tokenizer.Tokenizer,
        source: []const u8,
        pub fn tokenId(t: @This(), marker: []const u8) ?u32 { return t.tok.specialTokenId(marker); }
        pub fn templateSource(t: @This()) []const u8 { return t.source; }
    };
    try std.testing.expect(inkling.detect(Text{ .tok = &tok, .source = source }));
    try std.testing.expect(!inkling.detect(Text{ .tok = &tok, .source = "not Inkling" }));
    try std.testing.expectEqual(@as(?u32, 200006), tok.specialTokenId("<|content_model_end_sampling|>"));
    var reader: inkling.Reading = .{};
    const split = try reader.split(a, decoded, .ended);
    try std.testing.expectEqualStrings("Think first.", split.reasoning);
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{\"type\":\"object\"}}}]")).ok.array;
    const parsed = try tool_parse.parse(a, split.answer, tools, null);
    try std.testing.expectEqualStrings("Reading.", parsed.content);
    try std.testing.expectEqual(@as(usize, 2), parsed.calls.?.len);
}
