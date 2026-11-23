const std = @import("std");

pub fn reuse(resident: []const u32, prompt: []const u32, enabled: bool) usize {
    if (!enabled or resident.len >= prompt.len or !std.mem.eql(u32, resident, prompt[0..resident.len])) return 0;
    return resident.len;
}

pub const Choice = struct {
    index: usize = 0,
    reused: usize = 0,
    last_used: u64 = std.math.maxInt(u64),

    pub fn consider(c: *Choice, index: usize, resident: []const u32, last_used: u64, prompt: []const u32, enabled: bool) void {
        const n = reuse(resident, prompt, enabled);
        if (n > c.reused or (n == c.reused and last_used < c.last_used)) {
            c.* = .{ .index = index, .reused = n, .last_used = last_used };
        }
    }
};

test "prefix reuse requires a strict exact prefix" {
    const t = std.testing;
    try t.expectEqual(2, reuse(&.{ 1, 2 }, &.{ 1, 2, 3 }, true));
    try t.expectEqual(0, reuse(&.{ 1, 2 }, &.{ 1, 2 }, true));
    try t.expectEqual(0, reuse(&.{ 1, 3 }, &.{ 1, 2, 3 }, true));
    try t.expectEqual(0, reuse(&.{ 1, 2, 3 }, &.{ 1, 2 }, true));
    try t.expectEqual(0, reuse(&.{}, &.{ 1, 2 }, true));
    try t.expectEqual(0, reuse(&.{ 1, 2 }, &.{ 1, 2, 3 }, false));
}

test "select the longest resident prompt or the least recently used slot" {
    const t = std.testing;
    var choice = Choice{};
    choice.consider(0, &.{ 1, 2 }, 3, &.{ 1, 2, 3, 4 }, true);
    choice.consider(1, &.{ 1, 2, 3 }, 9, &.{ 1, 2, 3, 4 }, true);
    try t.expectEqual(@as(usize, 1), choice.index);
    try t.expectEqual(@as(usize, 3), choice.reused);
    choice = .{};
    choice.consider(0, &.{ 8 }, 4, &.{ 1, 2 }, true);
    choice.consider(1, &.{ 9 }, 2, &.{ 1, 2 }, true);
    try t.expectEqual(@as(usize, 1), choice.index);
    try t.expectEqual(@as(usize, 0), choice.reused);
}

test "prompt end survives dropped generated text and disabled reuse chooses LRU" {
    const t = std.testing;
    const prompt = &[_]u32{ 1, 2, 3 };
    const generated = &[_]u32{ 1, 2, 3, 8, 9 };
    const next = &[_]u32{ 1, 2, 3, 7, 6 };
    try t.expectEqual(@as(usize, 0), reuse(generated, next, true));
    var choice = Choice{};
    choice.consider(0, prompt, 10, next, true);
    choice.consider(1, &.{ 1, 2 }, 1, next, true);
    try t.expectEqual(@as(usize, 0), choice.index);
    try t.expectEqual(@as(usize, 3), choice.reused);
    choice = .{};
    choice.consider(0, prompt, 10, next, false);
    choice.consider(1, &.{ 1, 2 }, 1, next, false);
    try t.expectEqual(@as(usize, 1), choice.index);
    try t.expectEqual(@as(usize, 0), choice.reused);
}
