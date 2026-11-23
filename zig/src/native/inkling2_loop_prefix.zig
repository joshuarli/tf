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
