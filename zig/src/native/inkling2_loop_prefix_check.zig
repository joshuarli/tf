const std = @import("std");
const mtl = @import("metal");
const inkling = @import("inkling");

fn prefill(m: *const inkling.Model, s: *inkling.PrefillScratch, c: *inkling.Cache, ids: []const u32) !void {
    var at: usize = 0;
    while (at < ids.len) {
        const take = @min(s.chunk, ids.len - at);
        try inkling.prefillWant(m, s, c, ids[at..][0..take], if (at + take == ids.len) .last else .none);
        at += take;
    }
}

fn scheduled(m: *const inkling.Model, p: *inkling.PrefillScratch, s: *inkling.Scratch, c: *inkling.Cache, ids: []const u32, first: usize, decoded: usize) !void {
    try prefill(m, p, c, ids[0..first]);
    for (ids[first .. first + decoded]) |id| try inkling.position(m, s, c, id);
    try prefill(m, p, c, ids[first + decoded ..]);
}

fn argmax(values: []const f32) usize {
    var best: usize = 0;
    for (values[1..], 1..) |v, i| if (v > values[best]) { best = i; };
    return best;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.ExpectedModelAndCapture;
    const a = init.gpa;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(16 << 20));
    defer a.free(bytes);
    var words = std.mem.tokenizeAny(u8, bytes, " \n\r\t");
    const first = try std.fmt.parseInt(usize, words.next() orelse return error.BadCapture, 10);
    const decoded = try std.fmt.parseInt(usize, words.next() orelse return error.BadCapture, 10);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    while (words.next()) |word| try ids.append(a, try std.fmt.parseInt(u32, word, 10));
    if (first == 0 or first + decoded >= ids.items.len) return error.BadCapture;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try inkling.Model.load(a, init.io, args[1]);
    defer m.deinit();
    var c = try inkling.Cache.init(m.device, &m.config, ids.items.len);
    defer c.deinit();
    var p = try inkling.PrefillScratch.init(m.device, &m.config, inkling.default_chunk);
    defer p.deinit();
    var s = try inkling.Scratch.init(m.device, &m.config);
    defer s.deinit();
    const vocab: usize = m.config.text.unpadded_vocab;
    try scheduled(m, &p, &s, &c, ids.items, first, decoded);
    const retained = try a.dupe(f32, p.logits.slice(f32, vocab));
    defer a.free(retained);
    c.deinit();
    c = try inkling.Cache.init(m.device, &m.config, ids.items.len);
    try scheduled(m, &p, &s, &c, ids.items, first, decoded);
    const replay = p.logits.slice(f32, vocab);
    var max_same: f32 = 0;
    var bit_mismatches: usize = 0;
    for (retained, replay) |x, y| {
        max_same = @max(max_same, @abs(x - y));
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) bit_mismatches += 1;
    }
    std.debug.print("same schedule: max_abs={d} bit_mismatches={d}\n", .{ max_same, bit_mismatches });
    if (bit_mismatches != 0) return error.ReplayMismatch;
    c.deinit();
    c = try inkling.Cache.init(m.device, &m.config, ids.items.len);
    try prefill(m, &p, &c, ids.items);
    const fresh = p.logits.slice(f32, vocab);
    const max_r: f64 = retained[argmax(retained)];
    const max_f: f64 = fresh[argmax(fresh)];
    var sum_r: f64 = 0;
    var sum_f: f64 = 0;
    var max_abs: f64 = 0;
    for (retained, fresh) |x, y| {
        sum_r += @exp(@as(f64, x) - max_r);
        sum_f += @exp(@as(f64, y) - max_f);
        max_abs = @max(max_abs, @abs(@as(f64, x) - @as(f64, y)));
    }
    var kl: f64 = 0;
    for (retained, fresh) |x, y| {
        const log_r = @as(f64, x) - max_r - @log(sum_r);
        const log_f = @as(f64, y) - max_f - @log(sum_f);
        kl += @exp(log_r) * (log_r - log_f);
    }
    std.debug.print("full prefill: max_abs={d} kl={d} greedy_retained={d} greedy_fresh={d} agreement={}\n", .{ max_abs, kl, argmax(retained), argmax(fresh), argmax(retained) == argmax(fresh) });
    c.deinit();
    c = try inkling.Cache.init(m.device, &m.config, ids.items.len);
    var rollback = try inkling.Cache.Rollback.init(m.device, &m.config);
    defer rollback.deinit();
    try prefill(m, &p, &c, ids.items[0..first]);
    try rollback.copy(m.queue, &m.config, &c, true);
    for (ids.items[first .. first + decoded]) |id| try inkling.position(m, &s, &c, id);
    try rollback.copy(m.queue, &m.config, &c, false);
    try prefill(m, &p, &c, ids.items[first + decoded ..]);
    const restored = try a.dupe(f32, p.logits.slice(f32, vocab));
    defer a.free(restored);
    c.deinit();
    c = try inkling.Cache.init(m.device, &m.config, ids.items.len);
    const full = try a.alloc(u32, ids.items.len - decoded);
    defer a.free(full);
    @memcpy(full[0..first], ids.items[0..first]);
    @memcpy(full[first..], ids.items[first + decoded ..]);
    try prefill(m, &p, &c, full);
    const fresh_rollback = p.logits.slice(f32, vocab);
    var rollback_max: f32 = 0;
    for (restored, fresh_rollback) |x, y| rollback_max = @max(rollback_max, @abs(x - y));
    std.debug.print("rollback vs full prefill: max_abs={d} greedy_restored={d} greedy_fresh={d} agreement={}\n", .{ rollback_max, argmax(restored), argmax(fresh_rollback), argmax(restored) == argmax(fresh_rollback) });
}
