const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const Allocator = std.mem.Allocator;
const start = "<|message_model|>";
const thinking = "<|content_thinking|>";
const content = "<|content_text|>";
const invoke = "<|content_invoke_tool_json|>";
const end_message = "<|end_message|>";
const end_turn = "<|content_model_end_sampling|>";

pub fn detect(text: anytype) bool {
    for ([_][]const u8{ start, thinking, content, invoke, end_message, end_turn }) |marker| if (text.tokenId(marker) == null) return false;
    return std.mem.indexOf(u8, text.templateSource(), invoke) != null;
}

/// Generated tokens inside thinking messages, each counted from its `<|message_model|>` through its
/// `<|end_message|>` (an unclosed one through the last token); the marker tokens are single ids in this tokenizer.
pub fn reasoningTokens(tokens: []const u32, text: anytype) usize {
    const ids = .{ text.tokenId(start), text.tokenId(thinking), text.tokenId(end_message) };
    const open = ids[0] orelse return 0;
    const think = ids[1] orelse return 0;
    const close = ids[2] orelse return 0;
    var count: usize = 0;
    var message_at: ?usize = null;
    var in_thinking = false;
    for (tokens, 0..) |t, i| {
        if (t == open) {
            message_at = i;
            in_thinking = false;
        } else if (t == think and message_at != null) {
            in_thinking = true;
        } else if (t == close) {
            if (in_thinking) count += i + 1 - message_at.?;
            message_at = null;
            in_thinking = false;
        }
    }
    if (in_thinking) count += tokens.len - message_at.?;
    return count;
}

pub const End = @import("recipient_reply.zig").End;
pub const Reading = struct {
    reasoning: bool = true,

    pub fn split(r: *Reading, a: Allocator, reply: []const u8, end: End) Allocator.Error!reply_text.Split {
        var thought: std.ArrayList(u8) = .empty;
        var answer: std.ArrayList(u8) = .empty;
        var pos: usize = 0;
        while (std.mem.indexOfPos(u8, reply, pos, start)) |s| {
            const header = s + start.len;
            const tag_at = std.mem.indexOfPos(u8, reply, header, "<|") orelse break;
            const tag = reply[tag_at..];
            const is_thinking = std.mem.startsWith(u8, tag, thinking);
            const is_content = std.mem.startsWith(u8, tag, content);
            const is_call = std.mem.startsWith(u8, tag, invoke);
            if (!is_thinking and !is_content and !is_call) break;
            const body_at = tag_at + (if (is_thinking) thinking.len else if (is_content) content.len else invoke.len);
            const close = std.mem.indexOfPos(u8, reply, body_at, end_message);
            var body_end = close orelse reply.len;
            if (close == null and end == .streaming) body_end -= reply_text.partialTag(reply[body_at..body_end], end_message);
            const body = reply[body_at..body_end];
            if (is_call) {
                if (close != null) {
                    const parsed = try json.parse(a, body);
                    if (parsed == .ok and parsed.ok == .object) {
                        const name = parsed.ok.strField("name");
                        const args = parsed.ok.get("args");
                        if (name != null and std.mem.eql(u8, name.?, reply[header..tag_at]) and args != null and args.? == .object) {
                            if (answer.items.len > 0) try answer.appendSlice(a, "\n\n");
                            try answer.appendSlice(a, "<tool_call>");
                            try answer.appendSlice(a, body);
                            try answer.appendSlice(a, "</tool_call>");
                        }
                    }
                }
            } else if (is_thinking and r.reasoning) {
                if (thought.items.len > 0) try thought.appendSlice(a, "\n\n");
                try thought.appendSlice(a, reply_text.pyStrip(body));
            } else if (is_content) {
                if (answer.items.len > 0) try answer.appendSlice(a, "\n\n");
                try answer.appendSlice(a, reply_text.pyStrip(body));
            }
            if (close == null) break;
            pos = close.? + end_message.len;
        }
        return .{ .reasoning = thought.items, .answer = answer.items };
    }
};
