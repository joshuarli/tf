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

pub fn chooseFree(slots: anytype, prompt: []const u32, enabled: bool) ?Choice {
    var choice = Choice{};
    var found = false;
    for (slots, 0..) |*slot, i| {
        if (slot.job != null) continue;
        choice.consider(i, slot.resident.items, slot.last_used, prompt, enabled);
        found = true;
    }
    return if (found) choice else null;
}

pub const Fork = struct { source: usize, point: usize, common: usize };

pub fn chooseFork(slots: anytype, target: usize, prompt: []const u32, own: usize, enabled: bool) ?Fork {
    if (!enabled) return null;
    var best: ?Fork = null;
    for (slots, 0..) |*slot, i| {
        if (i == target) continue;
        var common: usize = 0;
        const resident = slot.resident.items;
        while (common < @min(resident.len, prompt.len) and resident[common] == prompt[common]) : (common += 1) {}
        for (slot.points) |point| {
            if (point <= common and point < prompt.len and point > own and (best == null or point > best.?.point)) {
                best = .{ .source = i, .point = point, .common = common };
            }
        }
    }
    return best;
}

pub const Admission = struct { choice: Choice, fork: ?Fork };

pub fn chooseAdmission(slots: anytype, prompt: []const u32, enabled: bool) ?Admission {
    var best: ?Admission = null;
    for (slots, 0..) |*slot, i| {
        if (slot.job != null) continue;
        var choice = Choice{};
        choice.consider(i, slot.resident.items, slot.last_used, prompt, enabled);
        const fork = chooseFork(slots, i, prompt, choice.reused, enabled);
        const tokens = if (fork) |f| f.point else choice.reused;
        const best_tokens = if (best) |b| (if (b.fork) |f| f.point else b.choice.reused) else 0;
        if (best == null or tokens > best_tokens or (tokens == best_tokens and choice.last_used < best.?.choice.last_used)) {
            best = .{ .choice = choice, .fork = fork };
        }
    }
    return best;
}

pub fn storeLookup(entries: anytype, prompt: []const u32, enabled: bool, minimum: usize) ?usize {
    if (!enabled) return null;
    var best: ?usize = null;
    var longest = minimum;
    for (entries, 0..) |entry, i| {
        const n = reuse(entry.tokens, prompt, true);
        if (n > longest) {
            best = i;
            longest = n;
        }
    }
    return best;
}

pub fn duplicate(entries: anytype, tokens: []const u32) ?usize {
    for (entries, 0..) |entry, i| if (std.mem.eql(u32, entry.tokens, tokens)) return i;
    return null;
}

pub fn oldest(entries: anytype) ?usize {
    var result: ?usize = null;
    for (entries, 0..) |entry, i| {
        if (result == null or entry.last_used < entries[result.?].last_used) result = i;
    }
    return result;
}

pub fn touchPrefixes(entries: anytype, prompt: []const u32, sequence: u64) u64 {
    const now = sequence + entries.len + 1;
    for (entries) |*entry| {
        if (entry.tokens.len > prompt.len or !std.mem.eql(u32, entry.tokens, prompt[0..entry.tokens.len])) continue;
        var shorter: u64 = 0;
        for (entries) |other| {
            if (other.tokens.len < entry.tokens.len and std.mem.eql(u32, other.tokens, prompt[0..other.tokens.len])) shorter += 1;
        }
        entry.last_used = now - shorter;
    }
    return now;
}

pub fn eviction(entries: anytype, used: u64, budget: u64, incoming: u64) ?usize {
    if (incoming > budget or used <= budget - incoming) return null;
    return oldest(entries);
}

pub fn cancelled(id: u64, background: bool, closing: bool, foreground_waiting: bool, cancels: []const u64) bool {
    return closing or (background and foreground_waiting) or std.mem.indexOfScalar(u64, cancels, id) != null;
}

pub fn canRestore(saved_len: usize, current_len: usize, saved_rows: usize, window: usize, cap: usize, emitted: usize) bool {
    return emitted <= cap and current_len >= saved_len and (current_len - saved_len <= saved_rows or saved_rows == window);
}

pub fn release(slot: anytype, restored: bool) void {
    if (!restored) {
        slot.cache.len = 0;
        slot.resident.clearRetainingCapacity();
    }
    slot.job = null;
    slot.live_generated = 0;
}

test "busy slots queue admissions and freed slots retain their own prefixes" {
    const T = struct { job: ?usize = null, resident: std.ArrayList(u32) = .empty, last_used: u64 = 0, cache: struct { len: usize = 0 } = .{}, live_generated: u64 = 0 };
    var slots = [_]T{ .{}, .{} };
    defer for (&slots) |*slot| slot.resident.deinit(std.testing.allocator);
    try slots[0].resident.appendSlice(std.testing.allocator, &.{ 1, 2 });
    try slots[1].resident.appendSlice(std.testing.allocator, &.{ 1, 2, 3 });
    const first = chooseFree(&slots, &.{ 1, 2, 3, 4 }, true).?;
    try std.testing.expectEqual(@as(usize, 1), first.index);
    slots[first.index].job = 1;
    const second = chooseFree(&slots, &.{ 1, 2, 3, 4 }, true).?;
    try std.testing.expectEqual(@as(usize, 0), second.index);
    slots[second.index].job = 2;
    try std.testing.expectEqual(null, chooseFree(&slots, &.{ 1, 2, 3, 4 }, true));
    slots[first.index].cache.len = 3;
    slots[second.index].cache.len = 2;
    slots[first.index].live_generated = 8;
    slots[second.index].live_generated = 8;
    release(&slots[first.index], true);
    try std.testing.expectEqual(@as(usize, 1), chooseFree(&slots, &.{ 1, 2, 3, 4 }, true).?.index);
    try std.testing.expectEqual(@as(?usize, 2), slots[second.index].job);
    slots[first.index].job = 3;
    try std.testing.expect(cancelled(2, false, false, false, &.{2}));
    try std.testing.expect(!cancelled(3, false, false, false, &.{2}));
    release(&slots[second.index], true);
    try std.testing.expectEqual(@as(usize, 0), chooseFree(&slots, &.{ 1, 2, 3, 4 }, true).?.index);
    try std.testing.expectEqual(@as(?usize, 3), slots[first.index].job);
}

test "rollback cap invalidates only the overflowing session" {
    const T = struct { job: ?usize = null, resident: std.ArrayList(u32) = .empty, cache: struct { len: usize = 0 } = .{}, live_generated: u64 = 0 };
    var slots = [_]T{ .{ .job = 1, .cache = .{ .len = 613 }, .live_generated = 513 }, .{ .job = 2, .cache = .{ .len = 300 }, .live_generated = 100 } };
    defer for (&slots) |*slot| slot.resident.deinit(std.testing.allocator);
    try slots[0].resident.appendSlice(std.testing.allocator, &.{ 1, 2 });
    try slots[1].resident.appendSlice(std.testing.allocator, &.{ 3, 4 });
    try std.testing.expect(canRestore(100, 612, 512, 1024, 512, 512));
    release(&slots[0], canRestore(100, 613, 512, 1024, 512, 513));
    try std.testing.expectEqual(@as(usize, 0), slots[0].resident.items.len);
    try std.testing.expectEqual(@as(usize, 0), slots[0].cache.len);
    try std.testing.expectEqual(@as(?usize, 2), slots[1].job);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4 }, slots[1].resident.items);
    try std.testing.expect(canRestore(200, 300, 512, 1024, 512, 100));
    try std.testing.expect(!canRestore(200, 199, 512, 1024, 512, 100));
    try std.testing.expect(canRestore(200, 1700, 1024, 1024, 2048, 1500));
    try std.testing.expect(!canRestore(200, 2250, 1024, 1024, 2048, 2050));
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

test "choose the longest available fork and retain longer own reuse" {
    const T = struct { job: ?usize = null, resident: std.ArrayList(u32) = .empty, last_used: u64 = 0, points: []const usize = &.{} };
    var slots = [_]T{ .{ .points = &.{2, 4} }, .{ .job = 1, .points = &.{2, 4, 6} }, .{ .points = &.{2} } };
    defer for (&slots) |*slot| slot.resident.deinit(std.testing.allocator);
    try slots[0].resident.appendSlice(std.testing.allocator, &.{ 1, 2, 9 });
    try slots[1].resident.appendSlice(std.testing.allocator, &.{ 1, 2, 3, 4, 5, 8, 9 });
    try slots[2].resident.appendSlice(std.testing.allocator, &.{ 1, 2, 3 });
    const fork = chooseFork(&slots, 0, &.{ 1, 2, 3, 4, 5, 6 }, 0, true).?;
    try std.testing.expectEqual(@as(usize, 1), fork.source);
    try std.testing.expectEqual(@as(usize, 4), fork.point);
    try std.testing.expectEqual(@as(usize, 5), fork.common);
    try std.testing.expectEqual(null, chooseFork(&slots, 0, &.{ 1, 2, 3, 4, 5, 6 }, 5, true));
    try std.testing.expectEqual(null, chooseFork(&slots, 0, &.{ 1, 2, 3, 4, 5, 6 }, 0, false));
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 8, 9 }, slots[1].resident.items);
    slots[1].job = null;
    slots[1].last_used = 0;
    slots[0].last_used = 1;
    slots[2].last_used = 2;
    const admission = chooseAdmission(&slots, &.{ 1, 2, 3, 4, 5, 6 }, true).?;
    try std.testing.expectEqual(@as(usize, 1), admission.fork.?.source);
    try std.testing.expect(admission.choice.index != 1);
}

test "store selection respects strict prefix and disabled reuse" {
    const entries = [_]struct { tokens: []const u32, last_used: u64 }{
        .{ .tokens = &.{ 1, 2 }, .last_used = 2 },
        .{ .tokens = &.{ 1, 2, 3, 4 }, .last_used = 1 },
        .{ .tokens = &.{ 1, 2, 3, 4, 5 }, .last_used = 3 },
    };
    try std.testing.expectEqual(@as(?usize, 1), storeLookup(&entries, &.{ 1, 2, 3, 4, 5 }, true, 3));
    try std.testing.expectEqual(@as(?usize, null), storeLookup(&entries, &.{ 1, 2, 3, 4, 5 }, false, 0));
    try std.testing.expectEqual(@as(?usize, 1), duplicate(&entries, &.{ 1, 2, 3, 4 }));
    try std.testing.expectEqual(@as(?usize, null), duplicate(&entries, &.{ 1, 2, 3 }));
    try std.testing.expectEqual(@as(?usize, 1), oldest(&entries));
}

test "store insertion deduplicates and evicts least recently used within byte budget" {
    const Entry = struct { tokens: []const u32, bytes: u64, last_used: u64 };
    var entries: std.ArrayList(Entry) = .empty;
    defer entries.deinit(std.testing.allocator);
    try entries.appendSlice(std.testing.allocator, &.{
        .{ .tokens = &.{ 1, 2 }, .bytes = 4, .last_used = 9 },
        .{ .tokens = &.{ 1, 3 }, .bytes = 6, .last_used = 3 },
    });
    try std.testing.expectEqual(@as(?usize, 0), duplicate(entries.items, &.{ 1, 2 }));
    try std.testing.expectEqual(@as(?usize, null), eviction(entries.items, 10, 14, 4));
    const victim = eviction(entries.items, 10, 14, 6).?;
    try std.testing.expectEqual(@as(usize, 1), victim);
    const removed = entries.swapRemove(victim);
    try std.testing.expectEqual(@as(u64, 6), removed.bytes);
    try entries.append(std.testing.allocator, .{ .tokens = &.{ 1, 4 }, .bytes = 6, .last_used = 10 });
    try std.testing.expectEqual(@as(?usize, null), eviction(entries.items, 10, 14, 4));
    try std.testing.expectEqual(@as(?usize, 0), duplicate(entries.items, &.{ 1, 2 }));
}

test "stored strict prefix beats own and fork reuse" {
    const T = struct { job: ?usize = null, resident: std.ArrayList(u32) = .empty, last_used: u64 = 0, points: []const usize = &.{} };
    var slots = [_]T{ .{}, .{ .job = 1, .points = &.{3} } };
    defer for (&slots) |*slot| slot.resident.deinit(std.testing.allocator);
    try slots[0].resident.appendSlice(std.testing.allocator, &.{ 1, 2 });
    try slots[1].resident.appendSlice(std.testing.allocator, &.{ 1, 2, 3, 8 });
    const prompt = &[_]u32{ 1, 2, 3, 4, 5, 6 };
    const admission = chooseAdmission(&slots, prompt, true).?;
    try std.testing.expectEqual(@as(usize, 3), admission.fork.?.point);
    const entries = [_]struct { tokens: []const u32 }{.{ .tokens = &.{ 1, 2, 3, 4, 5 } }};
    try std.testing.expectEqual(@as(?usize, 0), storeLookup(&entries, prompt, true, admission.fork.?.point));
    try std.testing.expectEqual(@as(?usize, null), storeLookup(&entries, prompt, false, 0));
}

test "shared prefixes remain recent as later prompts grow" {
    var entries = [_]struct { tokens: []const u32, last_used: u64 }{
        .{ .tokens = &.{ 1, 2 }, .last_used = 1 },
        .{ .tokens = &.{ 1, 2, 3 }, .last_used = 2 },
        .{ .tokens = &.{ 8, 9 }, .last_used = 3 },
    };
    const now = touchPrefixes(&entries, &.{ 1, 2, 3, 4 }, 3);
    try std.testing.expectEqual(@as(u64, 7), now);
    try std.testing.expectEqual(@as(?usize, 2), oldest(&entries));
    try std.testing.expect(entries[0].last_used > entries[1].last_used);
    try std.testing.expectEqual(@as(u64, 3), entries[2].last_used);
}


pub const Progress = struct {
    at: usize,
    next: usize,
    saved: usize = 0,

    pub const Step = struct { at: usize, take: usize, snapshot: ?usize, first_store: bool, last_store: bool, done: bool };

    pub fn init(length: usize, reused: usize, window: usize, interval: usize) Progress {
        const first = ((length -| window) + interval - 1) / interval * interval;
        return .{ .at = reused, .next = @max(interval, @max(first, ((reused + interval - 1) / interval) * interval)) };
    }

    pub fn step(p: *Progress, length: usize, chunk: usize, interval: usize, enabled: bool, reused: usize) Step {
        const at = p.at;
        var snapshot: ?usize = null;
        var first_store = false;
        if (enabled and p.next == at and p.next < length and p.next > 0) {
            snapshot = p.saved;
            first_store = reused == 0 and p.saved == 0;
            p.saved += 1;
            p.next += interval;
        }
        const last_store = enabled and at == length - 1;
        const remaining = if (enabled and at < length - 1) length - 1 - at else length - at;
        const take = @min(chunk, @min(remaining, if (enabled and p.next > at) p.next - at else remaining));
        p.at += take;
        return .{ .at = at, .take = take, .snapshot = snapshot, .first_store = first_store, .last_store = last_store, .done = p.at == length };
    }
};


pub fn admittingIndex(slots: anytype) ?usize {
    for (slots, 0..) |slot, i| if (slot.job != null and slot.admission != null) return i;
    return null;
}

pub fn decoding(slot: anytype) bool {
    return slot.job != null and slot.admission == null;
}

test "inkling scheduler: prefill chunks alternate with decode and preserve boundaries" {
    const Slot = struct { job: ?u8, admission: ?Progress };
    var slots = [_]Slot{
        .{ .job = 1, .admission = null },
        .{ .job = 2, .admission = Progress.init(13, 0, 8, 4) },
    };
    const expected_at = [_]usize{ 0, 3, 6, 8, 11, 12 };
    const expected_take = [_]usize{ 3, 3, 2, 3, 1, 1 };
    var decode_steps: usize = 0;
    for (expected_at, expected_take) |at, take| {
        const index = admittingIndex(&slots).?;
        const step = slots[index].admission.?.step(13, 3, 4, true, 0);
        try std.testing.expectEqual(at, step.at);
        try std.testing.expectEqual(take, step.take);
        try std.testing.expectEqual(if (at == 8) @as(?usize, 0) else if (at == 12) @as(?usize, 1) else null, step.snapshot);
        try std.testing.expectEqual(at == 8, step.first_store);
        try std.testing.expectEqual(at == 12, step.last_store);
        try std.testing.expectEqual(at + take == 13, step.done);
        try std.testing.expect(decoding(slots[0]));
        try std.testing.expect(!decoding(slots[1]));
        decode_steps += 1;
    }
    try std.testing.expectEqual(@as(usize, 6), decode_steps);
    try std.testing.expectEqual(@as(usize, 13), slots[1].admission.?.at);
    slots[1].admission = null;
    try std.testing.expect(decoding(slots[1]));
    try std.testing.expectEqual(@as(?usize, null), admittingIndex(&slots));
    var reused = Progress.init(13, 8, 8, 4);
    const fork = reused.step(13, 3, 4, true, 8);
    try std.testing.expectEqual(@as(?usize, 0), fork.snapshot);
    try std.testing.expect(!fork.first_store);
    try std.testing.expect(!fork.last_store);
    var coincident = Progress.init(5, 0, 4, 4);
    _ = coincident.step(5, 4, 4, true, 0);
    const last = coincident.step(5, 4, 4, true, 0);
    try std.testing.expect(last.first_store and last.last_store and last.done);
    try std.testing.expectEqual(@as(?usize, 0), last.snapshot);
    var disabled = Progress.init(13, 0, 8, 4);
    while (disabled.at < 13) {
        const step = disabled.step(13, 3, 4, false, 0);
        try std.testing.expectEqual(@as(?usize, null), step.snapshot);
        try std.testing.expect(!step.first_store);
        try std.testing.expect(!step.last_store);
    }
}
