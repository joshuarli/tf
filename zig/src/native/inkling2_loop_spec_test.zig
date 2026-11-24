const std = @import("std");
const api = @import("engine_api");
const native = @import("inkling2_loop.zig");
const host_mod = @import("inkling2_loop_host.zig");

const Prompt = struct { name: []const u8, token_ids: []const u32 };

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

const Reply = struct {
    request: api.Request,
    tokens: std.ArrayList(u32) = .empty,
    cached: u32 = 0,
    reason: ?api.Reason = null,
    stats: api.Stats = .{},
    done: std.Io.Event = .unset,
};

const Board = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    engine: api.Engine,
    replies: std.ArrayList(*Reply) = .empty,
    lost: std.atomic.Value(bool) = .init(false),

    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const b: *Board = @ptrCast(@alignCast(ctx));
        const r = b.replies.items[@intCast(id)];
        switch (e.*) {
            .prefilled => |n| r.cached = n,
            .tokens => |ids| r.tokens.appendSlice(b.gpa, ids) catch b.lost.store(true, .release),
            .finished => |f| {
                r.reason = f.reason;
                r.stats = f.stats;
                r.stats.telemetry_json = "";
                r.done.set(b.io);
            },
            else => {},
        }
    }

    fn submit(b: *Board, prompt: []const u32, max: u32, eos: []const u32, drafts: bool) !*Reply {
        const r = try b.gpa.create(Reply);
        errdefer b.gpa.destroy(r);
        r.* = .{ .request = .{ .prompt = prompt, .max_tokens = max, .eos = eos, .drafts = drafts } };
        if (b.replies.items.len == b.replies.capacity) return error.TooManyReplies;
        b.replies.appendAssumeCapacity(r);
        try b.engine.submit(b.replies.items.len - 1, &r.request, .{ .ctx = b, .event = event });
        return r;
    }

    fn wait(b: *Board, r: *Reply) !void {
        r.done.waitUncancelable(b.io);
        if (b.lost.load(.acquire)) return error.OutOfMemory;
        if (r.reason == .failed or r.reason == .cancelled) return error.ReplyFailed;
    }

    fn ask(b: *Board, prompt: []const u32, max: u32, eos: []const u32, drafts: bool) !*Reply {
        const r = try b.submit(prompt, max, eos, drafts);
        try b.wait(r);
        return r;
    }

    fn deinit(b: *Board) void {
        for (b.replies.items) |r| {
            r.tokens.deinit(b.gpa);
            b.gpa.destroy(r);
        }
        b.replies.deinit(b.gpa);
    }
};

fn expectSame(on: *const Reply, off: *const Reply) !void {
    try std.testing.expectEqualSlices(u32, off.tokens.items, on.tokens.items);
    try std.testing.expectEqual(off.reason, on.reason);
    try std.testing.expectEqual(@as(u64, 0), off.stats.drafted);
}

fn expectDrafted(name: []const u8, r: *const Reply) !void {
    std.debug.print("inkling served speculation: {s}: {d} tokens in {d} rounds, {d} drafted, {d} accepted, {d} cached\n", .{ name, r.tokens.items.len, r.stats.rounds, r.stats.drafted, r.stats.accepted, r.cached });
    if (r.stats.drafted == 0 or r.stats.accepted == 0 or r.stats.rounds >= r.tokens.items.len) return error.NoSpeculation;
}

test "inkling served speculation: replies with the DFlash2 drafter equal draft-off replies and a host without the drafter, through reuse, the store, a fork and two lanes" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const target = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return error.SkipZigTest;
    const drafter = std.c.getenv("TF_INKLING2_DFLASH2_DIR") orelse return error.SkipZigTest;
    if (setenv("INKLING_DRAFTER", drafter, 1) != 0) return error.SetEnv;
    defer _ = unsetenv("INKLING_DRAFTER");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const o: api.Open = .{ .dir = std.mem.span(target), .model_type = "inkling2_loop", .lanes = 2, .lanes_fixed = true, .context = 8192, .prompt_cache_gib = 2 };
    const parsed = try std.json.parseFromSlice([]const Prompt, gpa, @import("inkling_prompts").text, .{});
    defer parsed.deinit();
    var named: std.StringHashMapUnmanaged([]const u32) = .empty;
    defer named.deinit(gpa);
    for (parsed.value) |p| try named.put(gpa, p.name, p.token_ids);
    const agent = named.get("agent").?;
    const code = named.get("code").?;
    const prose = named.get("prose").?;
    const list = named.get("list").?;
    var board: Board = .{ .gpa = gpa, .io = io, .engine = undefined };
    defer board.deinit();
    try board.replies.ensureTotalCapacity(gpa, 32);
    var on: struct { code: *Reply, agent: *Reply, long: *Reply, short: *Reply } = undefined;
    {
        const opened = try openHost(arena.allocator(), o);
        defer opened.close(opened.ctx);
        try std.testing.expect(opened.engine.info().draft_sources.drafter != null);
        var out: [api.refusal_bytes]u8 = undefined;
        const sampled: api.Request = .{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 4, .sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 0, .top_p = 1, .min_p = 0 } };
        try std.testing.expectEqualStrings(host_mod.sampling_refusal, opened.engine.refusal(&sampled, &out).?);
        board.engine = opened.engine;

        const code_on = try board.ask(code, 128, &.{}, true);
        const code_off = try board.ask(code, 128, &.{}, false);
        try expectSame(code_on, code_off);
        try expectDrafted("code", code_on);
        try std.testing.expectEqual(@as(usize, 128), code_on.tokens.items.len);

        var stop: usize = 9;
        while (stop < 100 and std.mem.indexOfScalar(u32, code_off.tokens.items[0..stop], code_off.tokens.items[stop]) != null) stop += 1;
        try std.testing.expect(stop < 100);
        const eos = [1]u32{code_off.tokens.items[stop]};
        const cut_on = try board.ask(code, 128, &eos, true);
        const cut_off = try board.ask(code, 128, &eos, false);
        try expectSame(cut_on, cut_off);
        try expectDrafted("eos", cut_on);
        try std.testing.expectEqual(stop + 1, cut_on.tokens.items.len);
        try std.testing.expectEqual(@as(?api.Reason, .stop), cut_on.reason);

        const agent_on = try board.ask(agent, 192, &.{}, true);
        const agent_off = try board.ask(agent, 192, &.{}, false);
        try expectSame(agent_on, agent_off);
        try expectDrafted("agent", agent_on);

        const grown = try std.mem.concat(gpa, u32, &.{ agent, prose[prose.len / 2 ..] });
        defer gpa.free(grown);
        const grown_on = try board.ask(grown, 160, &.{}, true);
        const grown_off = try board.ask(grown, 160, &.{}, false);
        try expectSame(grown_on, grown_off);
        try expectDrafted("grown", grown_on);
        try std.testing.expect(grown_on.cached >= agent.len);

        const fresh = try gpa.dupe(u32, agent);
        defer gpa.free(fresh);
        fresh[100] = if (fresh[100] == 1000) 1001 else 1000;
        const forked = try std.mem.concat(gpa, u32, &.{ fresh[0..2560], code[code.len / 2 ..] });
        defer gpa.free(forked);
        const source_on = try board.submit(fresh, 48, &.{}, true);
        const fork_on = try board.submit(forked, 160, &.{}, true);
        try board.wait(source_on);
        try board.wait(fork_on);
        try std.testing.expectEqual(@as(u32, 2560), fork_on.cached);
        const source_off = try board.ask(fresh, 48, &.{}, false);
        const fork_off = try board.ask(forked, 160, &.{}, false);
        try expectSame(source_on, source_off);
        try expectSame(fork_on, fork_off);
        try expectDrafted("fork", fork_on);

        const long_on = try board.submit(prose, 192, &.{}, true);
        const short_on = try board.submit(list, 48, &.{}, true);
        try board.wait(long_on);
        try board.wait(short_on);
        const long_off = try board.ask(prose, 192, &.{}, false);
        const short_off = try board.ask(list, 48, &.{}, false);
        try expectSame(long_on, long_off);
        try expectSame(short_on, short_off);
        try expectDrafted("two lanes", long_on);
        on = .{ .code = code_on, .agent = agent_on, .long = long_on, .short = short_on };
    }
    if (unsetenv("INKLING_DRAFTER") != 0) return error.SetEnv;
    const drafted = board.replies.items.len / 2;
    const plain = try openHost(arena.allocator(), o);
    defer plain.close(plain.ctx);
    try std.testing.expect(plain.engine.info().draft_sources.drafter == null);
    board.engine = plain.engine;
    try expectSame(on.code, try board.ask(code, 128, &.{}, true));
    try expectSame(on.agent, try board.ask(agent, 192, &.{}, true));
    const long_plain = try board.submit(prose, 192, &.{}, true);
    const short_plain = try board.submit(list, 48, &.{}, true);
    try board.wait(long_plain);
    try board.wait(short_plain);
    try expectSame(on.long, long_plain);
    try expectSame(on.short, short_plain);
    std.debug.print("inkling served speculation: {d} drafted replies equal their draft-off replies, and 4 equal a host without the drafter\n", .{drafted});
}

fn openHost(a: std.mem.Allocator, o: api.Open) !api.Opened {
    var problem: []const u8 = "";
    return try native.open(a, std.testing.allocator, std.testing.io, o, &problem) orelse {
        std.debug.print("inkling served speculation: {s}\n", .{problem});
        return error.TestUnexpectedResult;
    };
}
