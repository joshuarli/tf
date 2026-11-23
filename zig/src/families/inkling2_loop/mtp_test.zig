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

test "acceptance stops at the first draft the target's pick contradicts" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 0), mtp.accepted(&.{ 5, 6 }, &.{ 4, 6, 7 }));
    try t.expectEqual(@as(usize, 1), mtp.accepted(&.{ 5, 6 }, &.{ 5, 7, 7 }));
    try t.expectEqual(@as(usize, 2), mtp.accepted(&.{ 5, 6 }, &.{ 5, 6, 7 }));
    try t.expectEqual(@as(usize, 0), mtp.accepted(&.{}, &.{9}));
}

test "the acceptance estimate rises with accepted drafts and falls with rejections" {
    const t = std.testing;
    var e: mtp.Estimate = .{};
    const prior = e.rate();
    try t.expectApproxEqAbs(@as(f32, 2.0 / 3.0), prior, 1e-6);
    e.note(4, 4);
    try t.expect(e.rate() > prior);
    var f: mtp.Estimate = .{};
    for (0..6) |_| f.note(3, 0);
    try t.expect(f.rate() < 0.3);
    f.note(0, 0);
    f.note(0, 0);
    try t.expectEqual(@as(u32, 2), f.idle);
    f.note(1, 1);
    try t.expectEqual(@as(u32, 0), f.idle);
}

test "the planner drafts when acceptance pays for the rows and falls back when it does not" {
    const t = std.testing;
    const costs = mtp.Costs{ .base = 0.019, .row = 0.007, .propose = 0.004 };
    var out: [8]usize = undefined;
    mtp.plan(&.{0.9}, &.{7}, &costs, out[0..1]);
    try t.expect(out[0] >= 3);
    mtp.plan(&.{0.1}, &.{7}, &costs, out[0..1]);
    try t.expectEqual(@as(usize, 0), out[0]);
    mtp.plan(&.{0.9}, &.{2}, &costs, out[0..1]);
    try t.expectEqual(@as(usize, 2), out[0]);
    mtp.plan(&.{ 0.95, 0.2 }, &.{ 7, 7 }, &costs, out[0..2]);
    try t.expect(out[0] > 0);
    try t.expectEqual(@as(usize, 0), out[1]);
    var total: usize = 2;
    for (out[0..2]) |k| total += k;
    try t.expect(total <= mtp.max_rows);
    mtp.plan(&.{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 }, &.{ 7, 7, 7, 7, 7, 7, 7, 7 }, &costs, out[0..8]);
    for (out[0..8]) |k| try t.expectEqual(@as(usize, 0), k);
    var c = costs;
    try t.expectApproxEqAbs(@as(f32, 0.019 + 2 * 0.007), c.verify(3), 1e-7);
    c.note(3, 0.040);
    try t.expectApproxEqAbs(@as(f32, 0.040), c.verify(3), 1e-7);
    try t.expectApproxEqAbs(@as(f32, 0.040 + 2 * 0.007), c.verify(5), 1e-7);
    c.note(7, 0.060);
    try t.expectApproxEqAbs(@as(f32, 0.050), c.verify(5), 1e-6);
    try t.expectApproxEqAbs(@as(f32, 0.030), c.verify(1), 1e-6);
    c.note(3, 0.050);
    try t.expectApproxEqAbs(@as(f32, 0.040 * 0.8 + 0.050 * 0.2), c.verify(3), 1e-7);
}

test "fallback probes back off while they leave the stream without drafts" {
    const t = std.testing;
    var e: mtp.Estimate = .{};
    for (0..mtp.Estimate.first_gap - 1) |_| {
        try t.expect(!e.probe(0));
        e.note(0, 0);
    }
    e.note(0, 0);
    try t.expect(e.probe(0));
    e.note(1, 0);
    var waited: u32 = 0;
    while (!e.probe(0)) : (waited += 1) e.note(0, 0);
    try t.expectEqual(2 * mtp.Estimate.first_gap, waited);
    e.note(1, 1);
    try t.expect(!e.probe(3));
    try t.expectEqual(mtp.Estimate.first_gap, e.gap);
}
