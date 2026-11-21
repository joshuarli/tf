//! A stream served without drafts verifies no copy window, pipelined or windowed, alone or sharing rounds.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const be = @import("backend.zig");
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const Proposer = @import("proposer.zig").Proposer;

const gpa = std.testing.allocator;

/// Always proposes the fake target's own continuation, so any copy the engine asks for is verified.
const Copier = struct {
    out: [8]u32 = undefined,
    asked: u32 = 0,

    fn propose(ptr: *anyopaque, context: []const u32, most: i64) anyerror![]const u32 {
        const c: *Copier = @ptrCast(@alignCast(ptr));
        c.asked += 1;
        if (most < 2) return &.{};
        var history: [256]u32 = undefined;
        @memcpy(history[0..context.len], context);
        const n: usize = @intCast(@min(c.out.len, most));
        for (c.out[0..n], 0..) |*t, k| {
            t.* = fake.next(history[0 .. context.len + k], null, 0);
            history[context.len + k] = t.*;
        }
        return c.out[0..n];
    }

    fn lastMatch(_: *anyopaque) i64 {
        return 100;
    }

    const vtable: Proposer.VTable = .{ .propose = propose, .last_match = lastMatch };
};

const Watch = struct {
    var inner: *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void = undefined;
    var copy_rows: u32 = 0;

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        for (windows) |w| if (w.held == 0) {
            copy_rows += @intCast(w.tokens.len);
        };
        return inner(ptr, windows, out);
    }
};

const Run = struct { tokens: [2][48]u32 = undefined, lens: [2]usize = @splat(0), copy_rows: u32, asked: u32 };

fn run(n: usize, drafts: bool, pipelined: bool) !Run {
    Watch.copy_rows = 0;
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var seen = target.backend();
    var table = seen.vtable.*;
    Watch.inner = table.verify;
    table.verify = Watch.verify;
    seen.vtable = &table;
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = pipelined, .hidden_rows = true, .batch_rows = 32, .max_streams = 4 }, 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, seen, clock.clock());
    defer engine.deinit();
    const prompts = [_][]const u32{ &.{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 }, &.{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8 } };
    var copiers: [2]Copier = @splat(.{});
    var streams: [2]sm.Stream = undefined;
    for (streams[0..n], copiers[0..n], prompts[0..n]) |*s, *c, p| s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = p, .max_new = 40, .eos = &.{}, .drafts = drafts, .proposer = .{ .ptr = c, .vtable = &Copier.vtable } });
    defer for (streams[0..n]) |*s| s.deinit(gpa);
    for (streams[0..n]) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Run = .{ .copy_rows = Watch.copy_rows, .asked = 0 };
    for (streams[0..n], copiers[0..n], 0..) |*s, c, i| {
        out.lens[i] = s.emitted().len;
        @memcpy(out.tokens[i][0..out.lens[i]], s.emitted());
        out.asked += c.asked;
    }
    return out;
}

test "no drafts: no copy is proposed or verified, and replies equal the drafted replies" {
    for ([_]usize{ 1, 2 }) |n| for ([_]bool{ true, false }) |pipelined| {
        const drafted = try run(n, true, pipelined);
        const plain = try run(n, false, pipelined);
        try std.testing.expect(drafted.copy_rows > 0);
        try std.testing.expectEqual(@as(u32, 0), plain.asked);
        try std.testing.expectEqual(@as(u32, 0), plain.copy_rows);
        for (0..n) |i| try std.testing.expectEqualSlices(u32, drafted.tokens[i][0..drafted.lens[i]], plain.tokens[i][0..plain.lens[i]]);
    };
}
