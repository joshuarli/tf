const std = @import("std");
const mtl = @import("metal");
const inkling = @import("inkling");

pub const tf_metal4 = true;

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

fn readState(m: *const inkling.Model, c: *inkling.Cache) !mtl.Buffer {
    const config = &m.config;
    const row_bytes = inkling.Cache.row(c, config, 0, 1) - inkling.Cache.row(c, config, 0, 0);
    const kv_bytes = (inkling.Cache.localSlots(config) * config.text.window + inkling.Cache.globalSlots(config) * c.len) * row_bytes * 2;
    const conv_bytes = c.k_conv.length() + c.v_conv.length() + c.attn_conv.length() + c.mlp_conv.length();
    const result = try m.device.buffer(kv_bytes + conv_bytes, mtl.ResourceOptions.shared);
    errdefer result.deinit();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    var at: usize = 0;
    for ([_]mtl.Buffer{ c.k_conv, c.v_conv, c.attn_conv, c.mlp_conv }) |buf| {
        blit.copy(buf, 0, result, at, buf.length());
        at += buf.length();
    }
    for (0..config.slots()) |slot| {
        const bytes = (if (inkling.Cache.isLocalSlot(config, slot)) config.text.window else c.len) * row_bytes;
        for ([_]mtl.Buffer{ c.keys, c.values }) |buf| {
            blit.copy(buf, c.baseOffset(config, slot), result, at, bytes);
            at += bytes;
        }
    }
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    return result;
}

fn checkStore(m: *const inkling.Model, p: *inkling.PrefillScratch, ids: []const u32, point: usize, a: std.mem.Allocator) !void {
    var source = try inkling.Cache.init(m.device, &m.config, ids.len);
    defer source.deinit();
    var direct = try inkling.Cache.init(m.device, &m.config, ids.len);
    defer direct.deinit();
    var restored = try inkling.Cache.init(m.device, &m.config, ids.len);
    defer restored.deinit();
    try prefill(m, p, &source, ids[0..point]);
    var entry = try inkling.Cache.init(m.device, &m.config, point);
    defer entry.deinit();
    try entry.copyFrom(m.queue, &m.config, &source, point);
    try prefill(m, p, &direct, ids[0..point]);
    try restored.copyFrom(m.queue, &m.config, &entry, point);
    if (restored.len != direct.len) return error.StoreLengthMismatch;
    const expected = try readState(m, &direct);
    defer expected.deinit();
    const actual = try readState(m, &restored);
    defer actual.deinit();
    if (!std.mem.eql(u8, expected.slice(u8, expected.length()), actual.slice(u8, actual.length()))) return error.StoreStateMismatch;
    try prefill(m, p, &direct, ids[point..]);
    const logits = try a.dupe(f32, p.logits.slice(f32, m.config.text.unpadded_vocab));
    defer a.free(logits);
    try prefill(m, p, &restored, ids[point..]);
    if (!std.mem.eql(f32, logits, p.logits.slice(f32, logits.len))) return error.StoreLogitsMismatch;
    std.debug.print("store: len={d} state_bytes={d} state_equal=true continuation_logits_equal=true\n", .{ point, actual.length() });
}

const usage =
    \\tf-inkling2-prefix-check <model> <capture>  a store, a replay and a rollback against full prefill
    \\tf-inkling2-prefix-check <model> <ids.json> <chunks> <out>  every row's logits under a chunk schedule
    \\tf-inkling2-prefix-check <model> jobs <jobs.txt>  one <ids.json> <chunks> <out> a line, one load
    \\tf-inkling2-prefix-check <model> time <ids.json> <reps> <chunks>...  each chunk's wall and GPU ms, the last row's logits only
    \\tf-inkling2-prefix-check <model> profile <ids.json> <reps> <chunks>...  `time` on a Metal 4 queue (TF_METAL4_TIMELINE)
    \\
;

const max_jobs = 256;
const max_schedules = 16;

const Job = struct { ids: []u32, sizes: []usize, out: []const u8 };

fn readIds(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    defer a.free(bytes);
    const parsed = try std.json.parseFromSlice([]const u32, a, bytes, .{});
    defer parsed.deinit();
    if (parsed.value.len == 0) return error.EmptyPrompt;
    return a.dupe(u32, parsed.value);
}

fn readSizes(a: std.mem.Allocator, text: []const u8, len: usize) ![]usize {
    var sizes: std.ArrayList(usize) = .empty;
    errdefer sizes.deinit(a);
    var words = std.mem.tokenizeScalar(u8, text, ',');
    var total: usize = 0;
    while (words.next()) |word| {
        const n = try std.fmt.parseInt(usize, word, 10);
        if (n == 0) return error.EmptyChunk;
        try sizes.append(a, n);
        total += n;
    }
    if (total != len) return error.ScheduleLength;
    return sizes.toOwnedSlice(a);
}

fn scratchRows(sizes: []const usize) usize {
    var top: usize = inkling.default_chunk;
    for (sizes) |n| top = @max(top, n);
    return top;
}

fn allRows(m: *const inkling.Model, p: *inkling.PrefillScratch, job: Job, out: []f32) !void {
    const vocab: usize = m.config.text.unpadded_vocab;
    std.debug.assert(out.len == job.ids.len * vocab);
    var c = try inkling.Cache.init(m.device, &m.config, job.ids.len);
    defer c.deinit();
    var at: usize = 0;
    for (job.sizes) |n| {
        try inkling.prefillWant(m, p, &c, job.ids[at..][0..n], .all);
        @memcpy(out[at * vocab ..][0 .. n * vocab], p.logits.slice(f32, n * vocab));
        at += n;
    }
    std.debug.assert(at == job.ids.len);
}

fn capture(init: std.process.Init, model: []const u8, jobs: []const Job) !void {
    const a = init.gpa;
    var rows: usize = 1;
    var top: usize = inkling.default_chunk;
    for (jobs) |job| {
        top = @max(top, scratchRows(job.sizes));
        for (job.sizes) |n| rows = @max(rows, n);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try inkling.Model.load(a, init.io, model);
    defer m.deinit();
    var p = try inkling.PrefillScratch.initRows(m.device, &m.config, top, rows);
    defer p.deinit();
    for (jobs) |job| {
        const out = try a.alloc(f32, job.ids.len * m.config.text.unpadded_vocab);
        defer a.free(out);
        try allRows(m, &p, job, out);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = job.out, .data = std.mem.sliceAsBytes(out) });
        std.debug.print("schedule: {d} tokens in {d} chunks, every row's logits, to {s}\n", .{ job.ids.len, job.sizes.len, job.out });
    }
}

fn schedule(init: std.process.Init, args: []const [:0]const u8) !void {
    const a = init.gpa;
    const ids = try readIds(a, init.io, args[2]);
    defer a.free(ids);
    const sizes = try readSizes(a, args[3], ids.len);
    defer a.free(sizes);
    try capture(init, args[1], &.{.{ .ids = ids, .sizes = sizes, .out = args[4] }});
}

fn scheduleJobs(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len != 4) return error.ExpectedArgs;
    const a = init.gpa;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], arena.allocator(), .limited(1 << 20));
    var jobs: std.ArrayList(Job) = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, " \t\r");
        const path = words.next() orelse continue;
        const chunks = words.next() orelse return error.BadJob;
        const out = words.next() orelse return error.BadJob;
        if (words.next() != null) return error.BadJob;
        if (jobs.items.len == max_jobs) return error.TooManyJobs;
        const ids = try readIds(arena.allocator(), init.io, path);
        try jobs.append(arena.allocator(), .{ .ids = ids, .sizes = try readSizes(arena.allocator(), chunks, ids.len), .out = out });
    }
    if (jobs.items.len == 0) return error.NoJobs;
    try capture(init, args[1], jobs.items);
}

fn ms(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn timed(init: std.process.Init, args: []const [:0]const u8, metal4: bool) !void {
    const a = init.gpa;
    if (args.len < 6 or args.len - 5 > max_schedules) return error.ExpectedArgs;
    const ids = try readIds(a, init.io, args[3]);
    defer a.free(ids);
    const reps = try std.fmt.parseInt(usize, args[4], 10);
    if (reps == 0) return error.ExpectedArgs;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var schedules: [max_schedules][]usize = undefined;
    const count = args.len - 5;
    var top: usize = inkling.default_chunk;
    for (args[5..], 0..) |text, s| {
        schedules[s] = try readSizes(arena.allocator(), text, ids.len);
        top = @max(top, scratchRows(schedules[s]));
        std.debug.print("time schedule={d} chunks={s}\n", .{ s, text });
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try inkling.Model.load(a, init.io, args[1]);
    defer m.deinit();
    if (metal4) {
        const q = try m.device.queue4();
        m.queue.deinit();
        m.queue = q;
    }
    var p = try inkling.PrefillScratch.initRows(m.device, &m.config, top, 1);
    defer p.deinit();
    var c = try inkling.Cache.init(m.device, &m.config, ids.len);
    defer c.deinit();
    const vocab: usize = m.config.text.unpadded_vocab;
    for (0..reps) |rep| for (schedules[0..count], 0..) |sizes, s| {
        c.len = 0;
        var at: usize = 0;
        var wall: i96 = 0;
        var gpu: f64 = 0;
        for (sizes) |n| {
            const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
            try inkling.prefillWant(m, &p, &c, ids[at..][0..n], if (at + n == ids.len) .last else .none);
            const spent = std.Io.Clock.awake.now(init.io).toNanoseconds() - t0;
            wall += spent;
            gpu += p.gpu_seconds;
            std.debug.print("time schedule={d} rep={d} at={d} rows={d} wall_ms={d:.3} gpu_ms={d:.3}\n", .{ s, rep, at, n, ms(spent), p.gpu_seconds * 1e3 });
            at += n;
        }
        std.debug.print("time schedule={d} rep={d} tokens={d} chunks={d} wall_ms={d:.3} gpu_ms={d:.3} greedy={d}\n", .{ s, rep, ids.len, sizes.len, ms(wall), gpu * 1e3, argmax(p.logits.slice(f32, vocab)) });
    };
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 4 and std.mem.eql(u8, args[2], "jobs")) return scheduleJobs(init, args);
    if (args.len >= 4 and std.mem.eql(u8, args[2], "time")) return timed(init, args, false);
    if (args.len >= 4 and std.mem.eql(u8, args[2], "profile")) return timed(init, args, true);
    if (args.len == 5) return schedule(init, args);
    if (args.len != 3) {
        std.debug.print("{s}", .{usage});
        return error.ExpectedModelAndCapture;
    }
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
    try checkStore(m, &p, ids.items, first, a);
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
