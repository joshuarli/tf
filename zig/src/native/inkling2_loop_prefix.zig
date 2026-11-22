const std = @import("std");

pub fn reuse(resident: []const u32, prompt: []const u32, enabled: bool) usize {
    if (!enabled or resident.len >= prompt.len or !std.mem.eql(u32, resident, prompt[0..resident.len])) return 0;
    return resident.len;
}

test "prefix reuse requires a strict exact prefix" {
    const t = std.testing;
    try t.expectEqual(2, reuse(&.{ 1, 2 }, &.{ 1, 2, 3 }, true));
    try t.expectEqual(0, reuse(&.{ 1, 2 }, &.{ 1, 2 }, true));
    try t.expectEqual(0, reuse(&.{ 1, 3 }, &.{ 1, 2, 3 }, true));
    try t.expectEqual(0, reuse(&.{ 1, 2, 3 }, &.{ 1, 2 }, true));
    try t.expectEqual(0, reuse(&.{}, &.{ 1, 2 }, true));
    try t.expectEqual(0, reuse(&.{ 1, 2 }, &.{ 1, 2, 3 }, false));
}
