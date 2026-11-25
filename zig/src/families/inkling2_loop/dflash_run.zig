//! `tf-inkling2-loop-run`'s DFlash2 modes: speculative against plain greedy decoding, and grouped verify widths.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const dflash = @import("dflash.zig");
const spec = @import("dflash_spec.zig");
const run = @import("run.zig");

pub const usage =
    \\tf-inkling2-loop-run <model> <prompts.json> dflash <drafter> <tokens> <out.jsonl> [bits] [entering|outputs|unnormed]
    \\  [incremental|rebuilt|f32_taps|first128|first512|window8192] [chain|lookup|tree|drafter_tree]  the drafter's
    \\  context: built row by row, rebuilt each round at positions from 0, prompt taps from eight-row (f32) prefill
    \\  chunks, the request's first 128 or 512 rows rebuilt each round ahead of its latest, or an 8,192-row window;
    \\  lookup drafts a round whose latest tokens occurred before the drafter's window from what followed them there;
    \\  tree verifies up to 16 rows of the drafter's tree merged with the lookup chain, drafter_tree without the chain;
    \\  either takes ,nodes=N ,children=N ,skip=P (the drafter's tree and the chain estimate that leaves it out)
    \\tf-inkling2-loop-run <model> <prompts.json> dflash-widths <reps> <out.jsonl> [load_s] [widths]  grouped verify at
    \\  widths 1, 2, 4, 8, 12 and 16 (or the comma-separated widths; 0: the one-row batch step), each prompt after load_s
    \\  seconds of eight-row verifies (sustained load; 0: none)
    \\tf-inkling2-loop-run <model> <prompts.json> dflash-kernels <reps> <out.jsonl> [picks.json]  one-row against
    \\  shared-row kernels on the weights at 1 to 16 rows, each shared output checked byte for byte; picks.json: a widths
    \\  run's 48 visits of routed experts
    \\tf-inkling2-loop-run <model> <prompts.json> dflash-widths <reps> <out.jsonl> <load_s> <widths> phases  the same
    \\  widths' GPU ms with one kernel class encoded at a time (prefill.Phase), the others left uncommitted
    \\  prompts.json: [{"name": ..., "ids": [...]}]; EOS is ignored; a run whose tokens differ from plain exits non-zero
    \\
;

const Prompt = @import("kbench.zig").Prompt;

pub fn main(init: std.process.Init, args: []const [:0]const u8) !void {
    const gpa = init.gpa;
    const mode = args[3];
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    const parsed = try std.json.parseFromSlice([]const Prompt, gpa, bytes, .{});
    defer parsed.deinit();
    if (parsed.value.len == 0) return error.EmptyPrompt;
    const timeline = std.mem.eql(u8, mode, "dflash-widths") and run.timelineWanted();
    const m = try inkling.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    if (timeline) run.timelineQueue(m);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    if (std.mem.eql(u8, mode, "dflash")) {
        if (args.len < 7 or args.len > 11) return error.ExpectedArgs;
        const limit = try std.fmt.parseInt(usize, args[5], 10);
        const bits = if (args.len >= 8) try std.fmt.parseInt(u8, args[7], 10) else 0;
        const reading = if (args.len >= 9) std.meta.stringToEnum(Reading, args[8]) orelse return error.ExpectedArgs else .entering;
        const context = if (args.len >= 10) std.meta.stringToEnum(Context, args[9]) orelse return error.ExpectedArgs else .incremental;
        var options = std.mem.splitScalar(u8, if (args.len == 11) args[10] else "chain", ',');
        const drafts = std.meta.stringToEnum(Drafts, options.first()) orelse return error.ExpectedArgs;
        var plan = inkling.verify.Plan.init();
        while (options.next()) |option| {
            const eq = std.mem.indexOfScalar(u8, option, '=') orelse return error.ExpectedArgs;
            const value = option[eq + 1 ..];
            if (std.mem.eql(u8, option[0..eq], "nodes")) plan.nodes = try std.fmt.parseInt(u32, value, 10) else if (std.mem.eql(u8, option[0..eq], "children")) plan.children = try std.fmt.parseInt(u32, value, 10) else if (std.mem.eql(u8, option[0..eq], "skip")) plan.skip_at = try std.fmt.parseFloat(f64, value) else return error.ExpectedArgs;
        }
        if (limit < 2) return error.ExpectedArgs;
        const same = try speculate(gpa, init.io, m, args[4], parsed.value, limit, bits, reading, context, drafts, plan, &out);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[6], .data = out.items });
        if (!same) return error.SpeculationDiffersFromPlain;
        return;
    }
    if (std.mem.eql(u8, mode, "dflash-kernels")) {
        if (args.len != 6 and args.len != 7) return error.ExpectedArgs;
        try @import("kbench.zig").run(gpa, m, try std.fmt.parseInt(usize, args[4], 10), if (args.len == 7) args[6] else null, init.io, &out);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = out.items });
        return;
    }
    if (std.mem.eql(u8, mode, "dflash-widths")) {
        if (args.len < 6 or args.len > 9) return error.ExpectedArgs;
        const by_phase = args.len == 9;
        if (by_phase and !std.mem.eql(u8, args[8], "phases")) return error.ExpectedArgs;
        const reps = try std.fmt.parseInt(usize, args[4], 10);
        const load_s = if (args.len >= 7) try std.fmt.parseFloat(f64, args[6]) else 0;
        if (reps == 0 or load_s < 0) return error.ExpectedArgs;
        var chosen: [16]usize = undefined;
        var count: usize = 0;
        for ([_]usize{ 0, 1, 2, 4, 8, 12, 16 }) |w| {
            chosen[count] = w;
            count += 1;
        }
        if (args.len >= 8) {
            count = 0;
            var it = std.mem.splitScalar(u8, args[7], ',');
            while (it.next()) |text| {
                if (count == chosen.len) return error.ExpectedArgs;
                const w = try std.fmt.parseInt(usize, text, 10);
                if (w > inkling.group.max_rows) return error.ExpectedArgs;
                chosen[count] = w;
                count += 1;
            }
            if (count == 0) return error.ExpectedArgs;
        }
        const result = if (by_phase) @import("kbench.zig").phases(gpa, m, parsed.value, reps, chosen[0..count], &out) else widths(gpa, m, parsed.value, reps, load_s, chosen[0..count], &out);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = out.items });
        return result;
    }
    return error.ExpectedArgs;
}

/// The capture reading a run drafts from: the checkpoint's, or a diagnostic of another (`outputs`, `unnormed`).
const Reading = enum { entering, outputs, unnormed };

const Context = enum { incremental, rebuilt, f32_taps, first128, first512, window8192 };

const Drafts = enum { chain, lookup, tree, drafter_tree };

const Rebuilt = struct {
    drafter: *dflash.Drafter,
    rows: []u8,
    first: []u8,
    first_rows: usize = 0,

    fn call(ptr: *anyopaque, pending: u32, _: u64, out: []u32) anyerror!void {
        const r: *Rebuilt = @ptrCast(@alignCast(ptr));
        const ring = r.drafter.model.ring();
        const end = ring.end.*;
        const keep: usize = @intCast(@min(end, ring.window - 1));
        const first = if (end > keep) r.first_rows else 0;
        @memcpy(r.rows[0 .. first * ring.stride], r.first[0 .. first * ring.stride]);
        var i: usize = first;
        while (i < keep) {
            const slot: usize = @intCast((end - keep + i) % ring.window);
            const count = @min(keep - i, ring.window - slot);
            @memcpy(r.rows[i * ring.stride ..][0 .. count * ring.stride], ring.buf.contents()[slot * ring.stride ..][0 .. count * ring.stride]);
            i += count;
        }
        try r.drafter.reset();
        try r.drafter.absorb(r.rows[0 .. keep * ring.stride], @intCast(keep));
        try r.drafter.flush();
        try r.drafter.propose(pending, @intCast(out.len), out);
    }
};

fn startF32(s: *spec.Speculator, cache: *inkling.Cache, twin: *inkling.Cache, small: *inkling.PrefillScratch, taps: *const inkling.Taps, ids: []const u32) !spec.Prefill {
    try s.drafter.reset();
    const t0 = spec.now();
    const first = try spec.prefillPlain(s.model, &s.prefill, cache, ids);
    small.taps = taps;
    defer small.taps = null;
    var at: usize = 0;
    while (at < ids.len) {
        const take = @min(small.chunk, ids.len - at);
        try inkling.prefillWant(s.model, small, twin, ids[at..][0..take], .none);
        try s.drafter.absorb(taps.bytes(0, take), @intCast(take));
        at += take;
    }
    const t1 = spec.now();
    try s.drafter.flush();
    return .{ .first = first, .prefill_ns = t1 - t0, .flush_ns = spec.now() - t1 };
}

fn digest(tokens: []const u32) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (tokens) |t| {
        var word: [4]u8 = undefined;
        std.mem.writeInt(u32, &word, t, .little);
        h.update(&word);
    }
    return h.finalResult();
}

fn tokensJson(gpa: std.mem.Allocator, out: *std.ArrayList(u8), tokens: []const u32) !void {
    try out.append(gpa, '[');
    for (tokens, 0..) |t, i| try out.print(gpa, "{s}{d}", .{ if (i == 0) "" else ",", t });
    try out.append(gpa, ']');
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn sorted(a: std.mem.Allocator, values: []const f64) ![]f64 {
    const copy = try a.dupe(f64, values);
    std.mem.sort(f64, copy, {}, std.sort.asc(f64));
    return copy;
}

fn quantile(values: []const f64, p: f64) f64 {
    const rank: usize = @intFromFloat(@ceil(p * @as(f64, @floatFromInt(values.len))));
    return values[@max(rank, 1) - 1];
}

fn mean(values: []const f64) f64 {
    var sum: f64 = 0;
    for (values) |v| sum += v;
    return sum / @as(f64, @floatFromInt(values.len));
}

fn speculate(gpa: std.mem.Allocator, io: std.Io, m: *inkling.Model, dir: []const u8, prompts: []const Prompt, limit: usize, bits: u8, reading: Reading, context: Context, drafts: Drafts, plan: inkling.verify.Plan, out: *std.ArrayList(u8)) !bool {
    const shifted = reading == .outputs;
    const unnormed = reading == .unnormed;
    const drafter = try dflash.Drafter.init(gpa, io, m, dir, .{ .bits = bits, .embed_norm = !unnormed, .window = if (context == .window8192) 8192 else null });
    defer drafter.deinit();
    const visits = if (shifted) &[_]u32{ dflash.tap_ids[0] + 1, dflash.tap_ids[1] + 1, dflash.tap_ids[2] + 1, dflash.tap_ids[3] } else dflash.tap_ids;
    var s = try spec.Speculator.init(m, drafter, visits);
    defer s.deinit();
    var rebuilt = Rebuilt{ .drafter = drafter, .rows = try gpa.alloc(u8, @as(usize, drafter.config.window - 1) * s.taps.rowBytes()), .first = try gpa.alloc(u8, 512 * s.taps.rowBytes()), .first_rows = switch (context) {
        .first128 => 128,
        .first512 => 512,
        else => 0,
    } };
    defer gpa.free(rebuilt.rows);
    defer gpa.free(rebuilt.first);
    if (context == .rebuilt or rebuilt.first_rows > 0) s.proposer = .{ .ptr = &rebuilt, .call = Rebuilt.call };
    var lookup = spec.Lookup.init(gpa);
    defer lookup.deinit();
    if (drafts == .lookup or drafts == .tree) s.lookup = &lookup;
    if (drafts == .tree or drafts == .drafter_tree) s.mode = .tree;
    s.plan.nodes = plan.nodes;
    s.plan.children = plan.children;
    s.plan.skip_at = plan.skip_at;
    var small: ?inkling.PrefillScratch = if (context == .f32_taps) try inkling.PrefillScratch.init(m.device, &m.config, 8) else null;
    defer if (small) |*p| p.deinit();
    var small_taps: ?inkling.Taps = if (context == .f32_taps) try inkling.Taps.init(m.device, &m.config, visits, 8) else null;
    defer if (small_taps) |*t| t.deinit();
    const residency = try drafter.fixedBytes();
    try out.print(gpa, "{{\"kind\":\"meta\",\"drafter_bits\":{d},\"draft_residency_bytes\":{d},\"warm_draft_ms\":{d:.3},\"limit\":{d},\"depth\":{d},\"taps\":[{d},{d},{d},{d}],\"embed_norm\":{},\"context\":\"{s}\",\"drafts\":\"{s}\",\"plan\":{{\"nodes\":{d},\"children\":{d},\"most\":{d},\"skip_at\":{d:.3},\"draft_ms\":{d:.2},\"verify_ms\":[", .{ bits, residency, drafter.step_ms, limit, s.depth, visits[0], visits[1], visits[2], visits[3], !unnormed, @tagName(context), @tagName(drafts), s.plan.nodes, s.plan.children, s.plan.most, s.plan.skip_at, s.plan.draft_ms });
    for (s.plan.verify_ms[1..], 0..) |v, i| try out.print(gpa, "{s}{d:.2}", .{ if (i == 0) "" else ",", v });
    try out.appendSlice(gpa, "]}}\n");
    var all_same = true;
    for (prompts, 0..) |prompt, index| {
        var spec_tokens: std.ArrayList(u32) = .empty;
        defer spec_tokens.deinit(gpa);
        var plain_tokens: std.ArrayList(u32) = .empty;
        defer plain_tokens.deinit(gpa);
        var rounds: std.ArrayList(spec.Round) = .empty;
        defer rounds.deinit(gpa);
        var steps: std.ArrayList(u64) = .empty;
        defer steps.deinit(gpa);
        var started: spec.Prefill = undefined;
        var spec_ns: u64 = 0;
        var plain_ns: u64 = 0;
        for (0..2) |turn| {
            const speculative = (turn == 0) == (index % 2 == 0);
            var cache = try inkling.Cache.init(m.device, &m.config, prompt.ids.len + limit + inkling.group.max_rows);
            defer cache.deinit();
            if (speculative) {
                var twin: ?inkling.Cache = if (context == .f32_taps) try inkling.Cache.init(m.device, &m.config, prompt.ids.len) else null;
                defer if (twin) |*t| t.deinit();
                if (rebuilt.first_rows > 0) {
                    var first_cache = try inkling.Cache.init(m.device, &m.config, rebuilt.first_rows);
                    defer first_cache.deinit();
                    s.prefill.taps = &s.taps;
                    defer s.prefill.taps = null;
                    try inkling.prefillWant(m, &s.prefill, &first_cache, prompt.ids[0..rebuilt.first_rows], .none);
                    @memcpy(rebuilt.first[0 .. rebuilt.first_rows * s.taps.rowBytes()], s.taps.bytes(0, rebuilt.first_rows));
                }
                started = if (twin) |*t| try startF32(&s, &cache, t, &small.?, &small_taps.?, prompt.ids) else try s.start(&cache, prompt.ids);
                try spec_tokens.append(gpa, started.first);
                const t0 = spec.now();
                try s.run(&cache, started.first, limit, &.{}, gpa, &spec_tokens, &rounds);
                spec_ns = spec.now() - t0;
            } else {
                const first = try spec.prefillPlain(m, &s.prefill, &cache, prompt.ids);
                try plain_tokens.append(gpa, first);
                const t0 = spec.now();
                try spec.plain(m, &s.batch, &cache, first, limit, &.{}, gpa, &plain_tokens, &steps);
                plain_ns = spec.now() - t0;
            }
        }
        const same = std.mem.eql(u32, spec_tokens.items, plain_tokens.items);
        var first_diff: usize = 0;
        while (first_diff < @min(spec_tokens.items.len, plain_tokens.items.len) and spec_tokens.items[first_diff] == plain_tokens.items[first_diff]) first_diff += 1;
        all_same = all_same and same;
        var totals: std.ArrayList(f64) = .empty;
        defer totals.deinit(gpa);
        var parts: [5]f64 = @splat(0);
        var emitted: usize = 0;
        var full: usize = 0;
        var zero: usize = 0;
        for (rounds.items, 0..) |r, i| {
            try out.print(gpa, "{{\"kind\":\"round\",\"prompt\":\"{s}\",\"round\":{d},\"position\":{d},\"offered\":{d},\"accepted\":{d},\"emitted\":{d},\"draft_ms\":{d:.4},\"verify_ms\":{d:.4},\"select_ms\":{d:.4},\"commit_ms\":{d:.4},\"host_ms\":{d:.4},\"total_ms\":{d:.4},\"source\":\"{s}\",\"drafts\":", .{ prompt.name, i, r.position, r.offered, r.accepted, r.emitted, ms(r.draft_ns), ms(r.verify_ns), ms(r.select_ns), ms(r.commit_ns), ms(r.host_ns), ms(r.total_ns), @tagName(r.source) });
            try tokensJson(gpa, out, r.drafts[0..r.offered]);
            if (r.source == .tree) {
                try out.print(gpa, ",\"looked\":{d},\"looked_accepted\":{d},\"expected\":{d:.4},\"match\":{d},\"hits\":{d},\"drafted\":{},\"parents\":[", .{ r.looked, r.looked_accepted, r.expected, r.match, r.hits, r.drafted });
                for (r.parents[0..r.offered], 0..) |p, k| try out.print(gpa, "{s}{d}", .{ if (k == 0) "" else ",", p });
                try out.appendSlice(gpa, "],\"q\":[");
                for (r.q[0..r.offered], 0..) |v, k| try out.print(gpa, "{s}{d:.5}", .{ if (k == 0) "" else ",", v });
                try out.appendSlice(gpa, "],\"chain\":[");
                for (r.chain[0..r.offered], 0..) |v, k| try out.print(gpa, "{s}{d}", .{ if (k == 0) "" else ",", if (v == inkling.verify.Plan.unlooked) -1 else @as(i32, v) });
                try out.appendSlice(gpa, "]");
            }
            try out.appendSlice(gpa, "}\n");
            try totals.append(gpa, ms(r.total_ns));
            for ([_]u64{ r.draft_ns, r.verify_ns, r.select_ns, r.commit_ns, r.host_ns }, 0..) |ns, k| parts[k] += ms(ns);
            emitted += r.emitted;
            full += @intFromBool(r.offered > 0 and r.accepted == r.offered);
            zero += @intFromBool(r.offered > 0 and r.accepted == 0);
        }
        const n: f64 = @floatFromInt(@max(rounds.items.len, 1));
        const ordered = try sorted(gpa, totals.items);
        defer gpa.free(ordered);
        var step_ms: std.ArrayList(f64) = .empty;
        defer step_ms.deinit(gpa);
        for (steps.items) |ns| try step_ms.append(gpa, ms(ns));
        const plain_sorted = try sorted(gpa, step_ms.items);
        defer gpa.free(plain_sorted);
        const spec_rate = @as(f64, @floatFromInt(emitted)) / (@as(f64, @floatFromInt(spec_ns)) / 1e9);
        const flush_rate = @as(f64, @floatFromInt(emitted)) / (@as(f64, @floatFromInt(spec_ns + started.flush_ns)) / 1e9);
        const plain_rate = @as(f64, @floatFromInt(steps.items.len)) / (@as(f64, @floatFromInt(plain_ns)) / 1e9);
        const committed = @as(f64, @floatFromInt(emitted)) / n;
        try out.print(gpa, "{{\"kind\":\"spec\",\"prompt\":\"{s}\",\"prompt_tokens\":{d},\"limit\":{d},\"equal_plain\":{},\"first_difference\":{d},\"rounds\":{d},\"emitted\":{d},\"full_rounds\":{d},\"zero_rounds\":{d},\"mean_committed\":{d:.4},\"round_ms_mean\":{d:.4},\"round_ms_p50\":{d:.4},\"round_ms_p90\":{d:.4},\"draft_ms_mean\":{d:.4},\"verify_ms_mean\":{d:.4},\"select_ms_mean\":{d:.4},\"commit_ms_mean\":{d:.4},\"host_ms_mean\":{d:.4},\"prefill_ms\":{d:.3},\"flush_ms\":{d:.3},\"decode_s\":{d:.4},\"tok_s\":{d:.3},\"tok_s_with_flush\":{d:.3},\"spec_first\":{},\"tokens_sha256\":\"{s}\"}}\n", .{ prompt.name, prompt.ids.len, limit, same, first_diff, rounds.items.len, emitted, full, zero, committed, mean(totals.items), quantile(ordered, 0.5), quantile(ordered, 0.9), parts[0] / n, parts[1] / n, parts[2] / n, parts[3] / n, parts[4] / n, ms(started.prefill_ns), ms(started.flush_ns), @as(f64, @floatFromInt(spec_ns)) / 1e9, spec_rate, flush_rate, index % 2 == 0, &std.fmt.bytesToHex(digest(spec_tokens.items), .lower) });
        try out.print(gpa, "{{\"kind\":\"plain\",\"prompt\":\"{s}\",\"steps\":{d},\"step_ms_mean\":{d:.4},\"step_ms_p50\":{d:.4},\"step_ms_p90\":{d:.4},\"decode_s\":{d:.4},\"tok_s\":{d:.3},\"tokens_sha256\":\"{s}\",\"tokens\":", .{ prompt.name, steps.items.len, mean(step_ms.items), quantile(plain_sorted, 0.5), quantile(plain_sorted, 0.9), @as(f64, @floatFromInt(plain_ns)) / 1e9, plain_rate, &std.fmt.bytesToHex(digest(plain_tokens.items), .lower) });
        try tokensJson(gpa, out, plain_tokens.items);
        try out.appendSlice(gpa, "}\n");
        std.debug.print("dflash {s}: prompt={d} equal={} rounds={d} committed/round={d:.3} round_ms mean={d:.2} p90={d:.2} (draft {d:.2} verify {d:.2} select {d:.2} commit {d:.2} host {d:.2}) spec_tok_s={d:.2} plain_tok_s={d:.2} plain_ms={d:.2} full={d} zero={d}\n", .{ prompt.name, prompt.ids.len, same, rounds.items.len, committed, mean(totals.items), quantile(ordered, 0.9), parts[0] / n, parts[1] / n, parts[2] / n, parts[3] / n, parts[4] / n, spec_rate, plain_rate, mean(step_ms.items), full, zero });
    }
    return all_same;
}

fn widths(gpa: std.mem.Allocator, m: *inkling.Model, prompts: []const Prompt, reps: usize, load_s: f64, cells: []const usize, out: *std.ArrayList(u8)) !void {
    var verifier = try inkling.group.Verifier.init(m.device, &m.config);
    defer verifier.deinit();
    var bs = try inkling.group.scratch(m.device, &m.config);
    defer bs.deinit();
    var ps = try inkling.PrefillScratch.init(m.device, &m.config, spec.prefill_chunk);
    defer ps.deinit();
    var rollback = try inkling.Cache.Rollback.initRows(m.device, &m.config, inkling.group.max_rows);
    defer rollback.deinit();
    const vocab: usize = m.config.text.unpadded_vocab;
    const logits = try gpa.alloc(f32, vocab);
    defer gpa.free(logits);
    const plain = try gpa.alloc(f32, inkling.group.max_rows * vocab);
    defer gpa.free(plain);
    const load_rows = 8;
    var all_equal = true;
    for (prompts) |prompt| {
        var cache = try inkling.Cache.init(m.device, &m.config, prompt.ids.len + 16);
        defer cache.deinit();
        const first = try spec.prefillPlain(m, &ps, &cache, prompt.ids);
        var rows: [inkling.group.max_rows]u32 = undefined;
        rows[0] = first;
        try rollback.copy(m.queue, &m.config, &cache, true);
        for (0..rows.len) |i| {
            try inkling.batchStep(m, &bs, &.{&cache}, &.{rows[i]}, plain[i * vocab ..][0..vocab]);
            if (i + 1 < rows.len) rows[i + 1] = spec.argmax(plain[i * vocab ..][0..vocab]);
        }
        try rollback.copy(m.queue, &m.config, &cache, false);
        for (cells) |width| {
            if (width == 0) continue;
            const g = try verifier.group(&.{&cache}, &.{width});
            try verifier.forward(m, &bs, &g, rows[0..width]);
            const got = bs.rows.logits.slice(f32, width * vocab);
            var equal: usize = 0;
            var worst: f32 = 0;
            for (0..width) |r| {
                const a = got[r * vocab ..][0..vocab];
                const b = plain[r * vocab ..][0..vocab];
                if (std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b))) equal += 1;
                for (a, b) |x, y| worst = @max(worst, @abs(x - y));
            }
            all_equal = all_equal and equal == width;
            try out.print(gpa, "{{\"kind\":\"exact\",\"prompt\":\"{s}\",\"context\":{d},\"width\":{d},\"rows_equal\":{d},\"max_abs_diff\":{e}}}\n", .{ prompt.name, cache.len, width, equal, worst });
            std.debug.print("exact {s}: width={d} rows bit-equal to one-row steps {d}/{d} max_abs_diff={e}\n", .{ prompt.name, width, equal, width, worst });
        }
        const load_start = spec.now();
        var loaded: usize = 0;
        while (@as(f64, @floatFromInt(spec.now() - load_start)) / 1e9 < load_s) : (loaded += 1) {
            const g = try verifier.group(&.{&cache}, &.{load_rows});
            try verifier.forward(m, &bs, &g, rows[0..load_rows]);
            _ = try verifier.picks(m, bs.rows.logits, load_rows);
        }
        var samples: [16]std.ArrayList(f64) = @splat(.empty);
        defer for (&samples) |*list| list.deinit(gpa);
        var forward_only: [16]std.ArrayList(f64) = @splat(.empty);
        defer for (&forward_only) |*list| list.deinit(gpa);
        var gpu_only: [16]std.ArrayList(f64) = @splat(.empty);
        defer for (&gpu_only) |*list| list.deinit(gpa);
        for (0..reps + 1) |rep| {
            for (0..cells.len) |k| {
                const cell = (k + rep) % cells.len;
                const width = cells[cell];
                const t0 = spec.now();
                var t1 = t0;
                if (width == 0) {
                    try rollback.copy(m.queue, &m.config, &cache, true);
                    const s0 = spec.now();
                    try inkling.batchStep(m, &bs, &.{&cache}, &.{first}, logits);
                    _ = spec.argmax(logits);
                    t1 = spec.now();
                    try rollback.copy(m.queue, &m.config, &cache, false);
                    if (rep > 0) {
                        try samples[cell].append(gpa, ms(t1 - s0));
                        try forward_only[cell].append(gpa, ms(t1 - s0));
                        try gpu_only[cell].append(gpa, bs.rows.gpu_seconds * 1e3);
                    }
                    continue;
                }
                const g = try verifier.group(&.{&cache}, &.{width});
                try verifier.forward(m, &bs, &g, rows[0..width]);
                t1 = spec.now();
                const gpu_ms = bs.rows.gpu_seconds * 1e3;
                _ = try verifier.picks(m, bs.rows.logits, width);
                const t2 = spec.now();
                if (rep > 0) {
                    try samples[cell].append(gpa, ms(t2 - t0));
                    try forward_only[cell].append(gpa, ms(t1 - t0));
                    try gpu_only[cell].append(gpa, gpu_ms);
                }
            }
        }
        for (cells, 0..) |width, cell| {
            const ordered = try sorted(gpa, samples[cell].items);
            defer gpa.free(ordered);
            const fw = try sorted(gpa, forward_only[cell].items);
            defer gpa.free(fw);
            const gw = try sorted(gpa, gpu_only[cell].items);
            defer gpa.free(gw);
            try out.print(gpa, "{{\"kind\":\"width\",\"prompt\":\"{s}\",\"load_s\":{d:.1},\"load_forwards\":{d},\"context\":{d},\"width\":{d},\"path\":\"{s}\",\"reps\":{d},\"ms_mean\":{d:.4},\"ms_p50\":{d:.4},\"ms_p90\":{d:.4},\"ms_min\":{d:.4},\"ms_max\":{d:.4},\"forward_ms_p50\":{d:.4},\"gpu_ms_p50\":{d:.4}}}\n", .{ prompt.name, load_s, loaded, cache.len, if (width == 0) 1 else width, if (width == 0) "batch_step" else "grouped", reps, mean(ordered), quantile(ordered, 0.5), quantile(ordered, 0.9), ordered[0], ordered[ordered.len - 1], quantile(fw, 0.5), quantile(gw, 0.5) });
            std.debug.print("widths {s}: context={d} {s} width={d} p50={d:.3} ms p90={d:.3} ms forward p50={d:.3} ms gpu p50={d:.3} ms\n", .{ prompt.name, cache.len, if (width == 0) "batch_step" else "grouped", if (width == 0) 1 else width, quantile(ordered, 0.5), quantile(ordered, 0.9), quantile(fw, 0.5), quantile(gw, 0.5) });
        }
        try routeOverlap(gpa, m, &verifier, &bs, &cache, &rows, prompt.name, out);
    }
    if (!all_equal) return error.GroupedDiffersFromPlain;
}

fn routeOverlap(gpa: std.mem.Allocator, m: *inkling.Model, verifier: *const inkling.group.Verifier, bs: *inkling.BatchScratch, cache: *inkling.Cache, rows: []const u32, name: []const u8, out: *std.ArrayList(u8)) !void {
    const c = &m.config;
    const topk: usize = c.text.topk;
    const experts: usize = c.text.experts;
    const per = rows.len * (topk + experts) + experts;
    const routes = try m.device.buffer(c.slots() * per * 4, mtl.ResourceOptions.shared);
    defer routes.deinit();
    const g = try verifier.group(&.{cache}, &.{rows.len});
    bs.rows.routes = routes;
    defer bs.rows.routes = null;
    try verifier.forward(m, bs, &g, rows);
    const words = routes.slice(u32, c.slots() * per);
    try out.print(gpa, "{{\"kind\":\"picks\",\"prompt\":\"{s}\",\"rows\":{d},\"picks\":[", .{ name, rows.len });
    for (0..c.slots()) |slot| for (words[slot * per ..][0 .. rows.len * topk], 0..) |e, i| {
        try out.print(gpa, "{s}{d}", .{ if (slot == 0 and i == 0) "" else ",", e });
    };
    try out.appendSlice(gpa, "]}\n");
    for ([_]usize{ 1, 2, 4, 8, 12, 16 }) |width| {
        if (width > rows.len) continue;
        var total: usize = 0;
        var most: usize = 0;
        var least: usize = experts;
        for (0..c.slots()) |slot| {
            var seen: std.StaticBitSet(256) = .empty;
            for (words[slot * per ..][0 .. width * topk]) |e| if (e < 256) seen.set(e);
            total += seen.count();
            most = @max(most, seen.count());
            least = @min(least, seen.count());
        }
        const visits: f64 = @floatFromInt(c.slots());
        try out.print(gpa, "{{\"kind\":\"routes\",\"prompt\":\"{s}\",\"context\":{d},\"width\":{d},\"visits\":{d},\"distinct_mean\":{d:.3},\"distinct_min\":{d},\"distinct_max\":{d}}}\n", .{ name, cache.len, width, c.slots(), @as(f64, @floatFromInt(total)) / visits, least, most });
        std.debug.print("routes {s}: width={d} distinct experts a visit mean={d:.2} min={d} max={d}\n", .{ name, width, @as(f64, @floatFromInt(total)) / visits, least, most });
    }
}
