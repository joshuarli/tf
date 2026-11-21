//! Image parts and `--vision` are refused with messages that name Inkling 2 Mini.
const std = @import("std");
const refuse = @import("refuse.zig");

const t = std.testing;

test "image content parts are refused by name, text and tool references pass" {
    const image = refuse.contentPart(&.{ .text, .image }).?;
    try t.expect(std.mem.indexOf(u8, image, "Inkling 2 Mini") != null);
    try t.expect(std.mem.indexOf(u8, image, "image content parts") != null);
    try t.expect(std.mem.indexOf(u8, image, "not support vision") != null);
    try t.expectEqual(@as(?[]const u8, null), refuse.contentPart(&.{ .text, .tool_reference }));
    try t.expect(std.mem.indexOf(u8, refuse.contentPart(&.{.video}).?, "text only") != null);
}

test "--vision is refused by name and no other flag is" {
    const why = refuse.visionFlag(true).?;
    try t.expect(std.mem.indexOf(u8, why, "Inkling 2 Mini") != null);
    try t.expect(std.mem.indexOf(u8, why, "--vision") != null);
    try t.expect(std.mem.indexOf(u8, why, "not support vision") != null);
    try t.expectEqual(@as(?[]const u8, null), refuse.visionFlag(false));
}

test "the engine's error names route to the same messages" {
    try t.expectEqualStrings(refuse.image, refuse.explain(error.VisionUnsupported).?);
    try t.expectEqualStrings(refuse.vision, refuse.explain(error.VisionFlag).?);
    try t.expectEqual(@as(?[]const u8, null), refuse.explain(error.OutOfMemory));
}
