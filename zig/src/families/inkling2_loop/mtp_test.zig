const std = @import("std");
const config = @import("config.zig");
const mtp = @import("mtp.zig");

test "checkpoint taps name the raw state entering each visit and the final normalized state" {
    const t = std.testing;
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, @embedFile("fixtures/config.json"), &why);
    const m = c.mtp.?;
    try t.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, m.taps[0..m.taps_n]);
    const expected = [_]mtp.Capture{
        .{ .entering = .{ .layer = 5, .pass = 0 } },
        .{ .entering = .{ .layer = 11, .pass = 1 } },
        .{ .entering = .{ .layer = 17, .pass = 2 } },
        .final_normalized,
    };
    for (m.taps[0..m.taps_n], expected) |tap, want| {
        try t.expectEqualDeep(want, try mtp.capture(c.loop, c.text.layers, tap));
    }
    const first = try mtp.capture(c.loop, c.text.layers, m.taps[0]);
    try t.expect(!std.meta.eql(first, mtp.Capture{ .entering = .{ .layer = 6, .pass = 0 } }));
    try t.expectEqualDeep(mtp.Capture{ .entering = .{ .layer = 6, .pass = 1 } }, try mtp.capture(c.loop, c.text.layers, 18));
    try t.expectEqualDeep(mtp.Capture{ .entering = .{ .layer = 6, .pass = 2 } }, try mtp.capture(c.loop, c.text.layers, 30));
    try t.expectEqualDeep(mtp.Capture{ .entering = .{ .layer = 18, .pass = 0 } }, try mtp.capture(c.loop, c.text.layers, 42));
    try t.expectError(error.InvalidTap, mtp.capture(c.loop, c.text.layers, 49));
    try t.expectEqual(@as(u32, 8), m.block);
    try t.expectEqual(@as(u32, 7), mtp.proposalCount(m));
    var other = m;
    other.block = 4;
    try t.expectEqual(@as(u32, 3), mtp.proposalCount(other));
}
