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
