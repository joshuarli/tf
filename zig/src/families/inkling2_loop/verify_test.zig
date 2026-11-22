const std = @import("std");
const model = @import("model.zig");
const state = @import("state.zig");
const prefill = @import("prefill.zig");
const Verification = @import("verify.zig").Verification;

test "rollback preserves local window and convolution rings" {
    const path = std.c.getenv("TF_INKLING2_LOOP_DIR") orelse return error.SkipZigTest;
    const m = try model.Model.load(std.testing.allocator, std.testing.io, std.mem.span(path));
    defer m.deinit();
    const c = &m.config;
    const base: usize = c.text.window - 1;
    var a = try state.Cache.init(m.device, c, base + 24);
    defer a.deinit();
    var b = try state.Cache.init(m.device, c, base + 24);
    defer b.deinit();
    var sa = try prefill.PrefillScratch.init(m.device, c, 512);
    defer sa.deinit();
    var sb = try prefill.PrefillScratch.init(m.device, c, 512);
    defer sb.deinit();
    var tokens: [512]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(100 + i % 71);
    var at: usize = 0;
    while (at < base) {
        const take = @min(tokens.len, base - at);
        try prefill.prefillWant(m, &sa, &a, tokens[0..take], .none);
        try prefill.prefillWant(m, &sb, &b, tokens[0..take], .none);
        at += take;
    }
    const draft = [_]u32{ 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116 };
    var best: [draft.len]u32 = undefined;
    var first: [draft.len]u32 = undefined;
    var baseline = try Verification.begin(m, &sa, &a, &draft, &first);
    try baseline.reset();
    var v = try Verification.begin(m, &sa, &a, &draft, &best);
    try std.testing.expectEqualSlices(u32, &first, &best);
    try v.finish(&draft, 2);
    try prefill.prefillWant(m, &sb, &b, draft[0..2], .all);
    const vocab: usize = c.text.unpadded_vocab;
    const reference = sb.logits.slice(f32, 2 * vocab);
    for (0..2) |row| {
        const logits = reference[row * vocab ..][0..vocab];
        var top: usize = 0;
        for (logits[1..], 1..) |value, i| if (value > logits[top]) { top = i; };
        try std.testing.expectEqual(@as(u32, @intCast(top)), best[row]);
    }
    const continuation = [_]u32{ 109, 110, 111, 112, 113 };
    try prefill.prefillWant(m, &sa, &a, &continuation, .all);
    try prefill.prefillWant(m, &sb, &b, &continuation, .all);
    const n = continuation.len * c.text.unpadded_vocab;
    const x = sa.logits.slice(f32, n);
    const y = sb.logits.slice(f32, n);
    var max_value: f32 = 0;
    var max_diff: f32 = 0;
    for (x, y) |lhs, rhs| {
        max_value = @max(max_value, @abs(rhs));
        max_diff = @max(max_diff, @abs(lhs - rhs));
    }
    std.debug.print("verify rollback max diff {e}, max abs {e}\n", .{ max_diff, max_value });
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(y), std.mem.sliceAsBytes(x));
    try baseline.finish(&draft, 0);
}
