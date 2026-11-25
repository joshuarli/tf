const std = @import("std");
const api = @import("engine_api");
const native = @import("inkling2_loop.zig");
const host_mod = @import("inkling2_loop_host.zig");
const prefix = @import("inkling2_loop_prefix.zig");
const mtl = @import("metal");

const Prompt = struct { name: []const u8, token_ids: []const u32 };

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

const Reply = struct {
    request: api.Request,
    tokens: std.ArrayList(u32) = .empty,
    cached: u32 = 0,
    reason: ?api.Reason = null,
    stats: api.Stats = .{},
    /// Lookup rows the reply's rounds verified and committed, from its telemetry.
    lookup_rows: u64 = 0,
    lookup_kept: u64 = 0,
    done: std.Io.Event = .unset,
};

fn telemetryField(json: []const u8, name: []const u8) u64 {
    var key: [64]u8 = undefined;
    const quoted = std.fmt.bufPrint(&key, "\"{s}\":", .{name}) catch return 0;
    const at = std.mem.indexOf(u8, json, quoted) orelse return 0;
    var end = at + quoted.len;
    while (end < json.len and std.ascii.isDigit(json[end])) end += 1;
    return std.fmt.parseInt(u64, json[at + quoted.len .. end], 10) catch 0;
}

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
                r.lookup_rows = telemetryField(f.stats.telemetry_json, "lookup_rows");
                r.lookup_kept = telemetryField(f.stats.telemetry_json, "lookup_kept");
                r.stats.telemetry_json = "";
                r.done.set(b.io);
            },
            else => {},
        }
    }

    fn submit(b: *Board, prompt: []const u32, max: u32, eos: []const u32, drafts: bool) !*Reply {
        return b.submitRequest(.{ .prompt = prompt, .max_tokens = max, .eos = eos, .drafts = drafts });
    }

    fn submitRequest(b: *Board, request: api.Request) !*Reply {
        const r = try b.gpa.create(Reply);
        errdefer b.gpa.destroy(r);
        r.* = .{ .request = request };
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
    std.debug.print("inkling served speculation: {s}: {d} tokens in {d} rounds, {d} drafted, {d} accepted, {d} cached, {d} lookup rows, {d} kept\n", .{ name, r.tokens.items.len, r.stats.rounds, r.stats.drafted, r.stats.accepted, r.cached, r.lookup_rows, r.lookup_kept });
    if (r.stats.drafted == 0 or r.stats.accepted == 0 or r.stats.rounds >= r.tokens.items.len) return error.NoSpeculation;
}

test "inkling served speculation: tree rounds with lookup equal draft-off replies and a host without the drafter, through reuse, the store, a fork, two lanes and an EOS in a lookup round" {
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

        const repeat = try std.mem.concat(gpa, u32, &.{ agent, agent[0..40] });
        defer gpa.free(repeat);
        const repeat_on = try board.ask(repeat, 96, &.{}, true);
        const repeat_off = try board.ask(repeat, 96, &.{}, false);
        try expectSame(repeat_on, repeat_off);
        try expectDrafted("lookup", repeat_on);
        try std.testing.expect(repeat_on.lookup_kept > 0);
        var lookup_stop: usize = 9;
        while (lookup_stop < 90 and std.mem.indexOfScalar(u32, repeat_off.tokens.items[0..lookup_stop], repeat_off.tokens.items[lookup_stop]) != null) lookup_stop += 1;
        try std.testing.expect(lookup_stop < 90);
        const lookup_eos = [1]u32{repeat_off.tokens.items[lookup_stop]};
        const repeat_cut_on = try board.ask(repeat, 96, &lookup_eos, true);
        const repeat_cut_off = try board.ask(repeat, 96, &lookup_eos, false);
        try expectSame(repeat_cut_on, repeat_cut_off);
        try expectDrafted("lookup eos", repeat_cut_on);
        try std.testing.expectEqual(lookup_stop + 1, repeat_cut_on.tokens.items.len);

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

test "inkling host: full-chunk prefill keeps a repeated prompt equal to its cold reply and a fork equal to a store hit at a point inside a chunk" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const target = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return error.SkipZigTest;
    if (unsetenv("INKLING_DRAFTER") != 0) return error.SetEnv;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const parsed = try std.json.parseFromSlice([]const Prompt, gpa, @import("inkling_prompts").text, .{});
    defer parsed.deinit();
    var named: std.StringHashMapUnmanaged([]const u32) = .empty;
    defer named.deinit(gpa);
    for (parsed.value) |p| try named.put(gpa, p.name, p.token_ids);
    const code = named.get("code").?;
    const source = try gpa.dupe(u32, named.get("agent").?);
    defer gpa.free(source);
    source[100] = if (source[100] == 1000) 1001 else 1000;
    const point: u32 = 2432;
    try std.testing.expect(source.len > point + 128 and source.len - 1 < 2048 + 1024);
    const forked = try std.mem.concat(gpa, u32, &.{ source[0 .. point + 50], code[code.len / 2 ..] });
    defer gpa.free(forked);
    const grown = try std.mem.concat(gpa, u32, &.{ source, code[code.len / 2 ..] });
    defer gpa.free(grown);
    var board: Board = .{ .gpa = gpa, .io = io, .engine = undefined };
    defer board.deinit();
    try board.replies.ensureTotalCapacity(gpa, 8);
    const o: api.Open = .{ .dir = std.mem.span(target), .model_type = "inkling2_loop", .lanes = 2, .lanes_fixed = true, .context = 8192, .prompt_cache_gib = 2 };
    var two: struct { cold: *Reply, again: *Reply, fork: *Reply, grown: *Reply } = undefined;
    {
        const opened = try openHost(arena.allocator(), o);
        defer opened.close(opened.ctx);
        board.engine = opened.engine;
        two.cold = try board.ask(source, 32, &.{}, false);
        two.again = try board.ask(source, 32, &.{}, false);
        two.fork = try board.ask(forked, 48, &.{}, false);
        two.grown = try board.ask(grown, 48, &.{}, false);
    }
    var one_lane = o;
    one_lane.lanes = 1;
    const opened = try openHost(arena.allocator(), one_lane);
    defer opened.close(opened.ctx);
    board.engine = opened.engine;
    const cold = try board.ask(source, 32, &.{}, false);
    const grown_one = try board.ask(grown, 48, &.{}, false);
    const hit = try board.ask(forked, 48, &.{}, false);
    try std.testing.expectEqual(@as(u32, 0), two.cold.cached);
    try std.testing.expectEqual(@as(u32, @intCast(source.len)), two.again.cached);
    try expectSame(two.again, two.cold);
    try expectSame(cold, two.cold);
    try std.testing.expectEqual(point, two.fork.cached);
    try std.testing.expectEqual(point, hit.cached);
    try expectSame(two.fork, hit);
    try std.testing.expectEqual(@as(u32, @intCast(source.len)), two.grown.cached);
    try std.testing.expectEqual(@as(u32, @intCast(source.len)), grown_one.cached);
    try expectSame(two.grown, grown_one);
    std.debug.print("inkling host: a repeated prompt, a fork at {d} and a store hit there reply as their references\n", .{point});
}

fn askSampled(b: *Board, prompt: []const u32) !*Reply {
    const r = try b.submitRequest(.{ .prompt = prompt, .max_tokens = 4, .sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 0, .top_p = 1, .min_p = 0 } });
    r.done.waitUncancelable(b.io);
    if (b.lost.load(.acquire)) return error.OutOfMemory;
    return r;
}

fn expectRefused(r: *const Reply) !void {
    try std.testing.expectEqual(@as(?api.Reason, .failed), r.reason);
    try std.testing.expectEqual(@as(usize, 0), r.tokens.items.len);
    try std.testing.expectEqual(@as(u32, 0), r.cached);
}

test "inkling host: a prompt in one chunk past 1,024 rows replies as its cold reply from its whole stored state and last-row logits, with an EOS, no tokens, a refused sampled request, a fork inside the chunk and reuse off" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const target = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return error.SkipZigTest;
    if (unsetenv("INKLING_DRAFTER") != 0) return error.SetEnv;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const parsed = try std.json.parseFromSlice([]const Prompt, gpa, @import("inkling_prompts").text, .{});
    defer parsed.deinit();
    var named: std.StringHashMapUnmanaged([]const u32) = .empty;
    defer named.deinit(gpa);
    for (parsed.value) |p| try named.put(gpa, p.name, p.token_ids);
    const code = named.get("code").?;
    const agent = named.get("agent").?;
    const len = host_mod.last_chunk_rows - 40;
    try std.testing.expect(len > 1024 and agent.len > len);
    const source = agent[0..len];
    const point: u32 = 896;
    const forked = try std.mem.concat(gpa, u32, &.{ source[0 .. point + 60], code[code.len / 2 ..] });
    defer gpa.free(forked);
    var board: Board = .{ .gpa = gpa, .io = io, .engine = undefined };
    defer board.deinit();
    try board.replies.ensureTotalCapacity(gpa, 12);
    const o: api.Open = .{ .dir = std.mem.span(target), .model_type = "inkling2_loop", .lanes = 2, .lanes_fixed = true, .context = 4096, .prompt_cache_gib = 2 };
    var two: struct { cold: *Reply, fork: *Reply, again: *Reply, stop: *Reply, none: *Reply, sampled: *Reply } = undefined;
    var eos: [1]u32 = undefined;
    var stored: ?[]f32 = null;
    defer if (stored) |row| gpa.free(row);
    {
        const opened = try openHost(arena.allocator(), o);
        defer opened.close(opened.ctx);
        board.engine = opened.engine;
        two.cold = try board.ask(source, 32, &.{}, false);
        try std.testing.expect(two.cold.tokens.items.len > 0);
        eos = .{two.cold.tokens.items[0]};
        const h: *host_mod.Host = @ptrCast(@alignCast(opened.ctx));
        const index = prefix.whole(h.entries.items, source, true) orelse return error.TestUnexpectedResult;
        stored = try gpa.dupe(f32, h.entries.items[index].logits.?);
        two.fork = try board.ask(forked, 48, &.{}, false);
        two.again = try board.ask(source, 32, &.{}, false);
        two.stop = try board.ask(source, 32, &eos, false);
        two.none = try board.ask(source, 0, &.{}, false);
        two.sampled = try askSampled(&board, source);
    }
    var one_lane = o;
    one_lane.lanes = 1;
    var one: struct { cold: *Reply, hit: *Reply } = undefined;
    {
        const opened = try openHost(arena.allocator(), one_lane);
        defer opened.close(opened.ctx);
        board.engine = opened.engine;
        one.cold = try board.ask(source, 32, &.{}, false);
        one.hit = try board.ask(forked, 48, &.{}, false);
    }
    if (setenv("INKLING_PREFIX_REUSE", "0", 1) != 0) return error.SetEnv;
    defer _ = unsetenv("INKLING_PREFIX_REUSE");
    const opened = try openHost(arena.allocator(), one_lane);
    defer opened.close(opened.ctx);
    board.engine = opened.engine;
    const off = try board.ask(source, 32, &.{}, false);
    const h: *host_mod.Host = @ptrCast(@alignCast(opened.ctx));
    const cold_row = h.prefill.logits.slice(f32, h.model.config.text.unpadded_vocab);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(cold_row), std.mem.sliceAsBytes(stored.?));
    const off_stop = try board.ask(source, 32, &eos, false);
    const off_sampled = try askSampled(&board, source);
    try expectRefused(two.sampled);
    try expectRefused(off_sampled);
    try std.testing.expectEqual(@as(u32, 0), two.cold.cached);
    try std.testing.expectEqual(point, two.fork.cached);
    try std.testing.expectEqual(point, one.hit.cached);
    try expectSame(two.fork, one.hit);
    try std.testing.expectEqual(@as(u32, @intCast(len)), two.again.cached);
    try expectSame(two.again, two.cold);
    try std.testing.expectEqual(@as(u32, @intCast(len)), two.stop.cached);
    try std.testing.expectEqualSlices(u32, &eos, two.stop.tokens.items);
    try std.testing.expectEqual(@as(?api.Reason, .stop), two.stop.reason);
    try std.testing.expectEqual(@as(u32, @intCast(len)), two.none.cached);
    try std.testing.expectEqual(@as(usize, 0), two.none.tokens.items.len);
    try std.testing.expectEqual(@as(?api.Reason, .length), two.none.reason);
    try std.testing.expectEqual(@as(u32, 0), one.cold.cached);
    try expectSame(one.cold, two.cold);
    try std.testing.expectEqual(@as(u32, 0), off.cached);
    try expectSame(off, two.cold);
    try expectSame(off_stop, two.stop);
    std.debug.print("inkling host: {d} tokens in one chunk, its whole-prompt hit and stored logits, EOS, empty and sampled requests, a fork at {d} and reuse off match their references\n", .{ len, point });
}

test "inkling host: the warm-up at load leaves the first slot's caches, rollback and snapshots zero and stores nothing" {
    const target = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return error.SkipZigTest;
    if (unsetenv("INKLING_DRAFTER") != 0) return error.SetEnv;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const opened = try openHost(arena.allocator(), .{ .dir = std.mem.span(target), .model_type = "inkling2_loop", .lanes = 1, .lanes_fixed = true, .context = 4096, .prompt_cache_gib = 2 });
    defer opened.close(opened.ctx);
    const h: *host_mod.Host = @ptrCast(@alignCast(opened.ctx));
    const slot = &h.slots[0];
    try std.testing.expectEqual(@as(usize, 0), slot.cache.len);
    try std.testing.expectEqual(@as(usize, 0), slot.resident.items.len);
    try std.testing.expectEqual(@as(usize, 0), h.entries.items.len);
    for (slot.points) |point| try std.testing.expectEqual(@as(usize, 0), point);
    var buffers: std.ArrayList(mtl.Buffer) = .empty;
    defer buffers.deinit(std.testing.allocator);
    try buffers.appendSlice(std.testing.allocator, &.{ slot.cache.keys, slot.cache.values, slot.cache.k_conv, slot.cache.v_conv, slot.cache.attn_conv, slot.cache.mlp_conv, slot.rollback.?.buffer });
    for (slot.snapshots) |snapshot| if (snapshot) |r| try buffers.append(std.testing.allocator, r.buffer);
    for (buffers.items) |buf| {
        const shared = try h.model.device.buffer(buf.length(), mtl.ResourceOptions.shared);
        defer shared.deinit();
        const cb = h.model.queue.commandBuffer();
        const blit = cb.blit();
        blit.copy(buf, 0, shared, 0, buf.length());
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        for (shared.slice(u64, buf.length() / 8)) |word| try std.testing.expectEqual(@as(u64, 0), word);
    }
}

fn openHost(a: std.mem.Allocator, o: api.Open) !api.Opened {
    var problem: []const u8 = "";
    return try native.open(a, std.testing.allocator, std.testing.io, o, &problem) orelse {
        std.debug.print("inkling served speculation: {s}\n", .{problem});
        return error.TestUnexpectedResult;
    };
}
