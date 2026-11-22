//! `tf-inkling2-loop-run`: teacher-forced passes on Metal for the campaign's parity check and the family's own checks.
//! Every mode runs the same per-token forward; they differ in what they report and whether they feed the model back.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const tk = @import("tokenizer");
const inkling = tf.inkling2_loop;

const usage =
    \\tf-inkling2-loop-run <model> <tokens.json> logits <out>       f32 logits [tokens, 200058], teacher-forced
    \\tf-inkling2-loop-run <model> <tokens.json> prefill <chunk> <out>  chunked prefill logits [tokens, 200058] plus tok/s
    \\tf-inkling2-loop-run <model> <tokens.json> greedy <steps> <out>  then <steps> greedy tokens, one decode a step
    \\tf-inkling2-loop-run <model> <tokens.json> greedy-prefill|greedy-copy|greedy-hybrid|greedy-decode <steps> <out>  raw u32 output IDs
    \\tf-inkling2-loop-run <model> <tokens.json> verifybench <min-width> <rounds>  time interleaved prefill widths
    \\tf-inkling2-loop-run <model> <tokens.json> verifycheck <accepted>  compare continuation with replay
    \\tf-inkling2-loop-run <model> <tokens.json> roundbench <rounds> [width]  time full verification rounds
    \\tf-inkling2-loop-run <model> <tokens.json> trace <out> [at]  per-tensor summaries in the reference's order, the whole pass or only position <at>
    \\tf-inkling2-loop-run <model> <tokens.json> prefillrow <chunk> <out> <at>  capture a chunked prefill row and router sidecars
    \\tf-inkling2-loop-run <model> <tokens.json> prefillroutes <chunk> <out> <at>  capture per-chunk router picks, logits, and bias
    \\tf-inkling2-loop-run <model> <tokens.json> routes <out> <at>  capture decode router picks, logits, and bias
    \\tf-inkling2-loop-run <model> <tokens.json> tracerow <out> <at>  the raw per-tensor rows of position <at> as f32, [slots][9][hidden], and router sidecars
    \\tf-inkling2-loop-run <model> <tokens.json> reset <out>        the same logits twice from a fresh cache, byte-compared
    \\tf-inkling2-loop-run <model> <tokens.json> bench <steps> [warmup] [chunk]  the prompt, warmup, then steps: wall, encode, GPU ms
    \\tf-inkling2-loop-run <tokenizer.json> <text> tokenize <out>   the text's token ids, for the modes above
    \\tf-inkling2-loop-run <tokenizer.json> <ids.json> detokenize <out>  the ids' text
    \\
;

pub const std_options: std.Options = .{ .log_level = .info, .logFn = logFn };
pub const tf_metal4 = true;

var metrics_ms: f64 = 0;
var metrics_wait_ms: f64 = 0;
var metrics_gpu_ms: f64 = 0;
var metrics_commits: u64 = 0;
var metrics_seen: bool = false;

/// Keeps the forward's pass receipts (one `q27_pass` line) out of stderr and in this program's hands.
fn logFn(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    if (comptime std.mem.startsWith(u8, format, "q27_pass")) {
        var buf: [512]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, format, args) catch return;
        metrics_ms += field(line, "ms=") orelse 0;
        metrics_wait_ms += field(line, "wait_ms=") orelse 0;
        metrics_gpu_ms += field(line, "gpu_ms=") orelse 0;
        if (field(line, "commits=")) |v| metrics_commits += @intFromFloat(v);
        metrics_seen = true;
        return;
    }
    std.log.defaultLog(level, scope, format, args);
}

fn field(line: []const u8, key: []const u8) ?f64 {
    const at = std.mem.indexOf(u8, line, key) orelse return null;
    const rest = line[at + key.len ..];
    const end = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
    return std.fmt.parseFloat(f64, rest[0..end]) catch null;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) {
        std.debug.print("{s}", .{usage});
        return error.ExpectedArgs;
    }
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const mode = args[3];
    if (std.mem.eql(u8, mode, "tokenize")) {
        try tokenize(gpa, init.io, args[1], args[2], args[4]);
        return;
    }
    if (std.mem.eql(u8, mode, "detokenize")) {
        try detokenize(gpa, init.io, args[1], args[2], args[4]);
        return;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    const ids = try std.json.parseFromSlice([]u32, gpa, bytes, .{});
    defer ids.deinit();
    if (ids.value.len == 0) return error.EmptyPrompt;
    const timeline = std.c.getenv("TF_METAL4_TIMELINE");
    const want_timeline = if (timeline) |p| std.mem.span(p).len > 0 and !std.mem.eql(u8, std.mem.span(p), "0") else false;
    if (want_timeline) mtl.Pipeline.labels(true);
    const m = try inkling.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    if (want_timeline) {
        const q4 = m.device.queue4() catch null;
        if (q4) |q| {
            m.queue.deinit();
            m.queue = q;
        }
    }
    const c = &m.config;
    const vocab: usize = c.text.unpadded_vocab;
    const t0 = now();
    if (std.mem.eql(u8, mode, "greedy-prefill") or std.mem.eql(u8, mode, "greedy-copy") or std.mem.eql(u8, mode, "greedy-hybrid") or std.mem.eql(u8, mode, "greedy-decode")) {
        if (args.len != 6) return error.ExpectedArgs;
        const steps = try std.fmt.parseInt(usize, args[4], 10);
        var cache = try inkling.Cache.init(m.device, c, ids.value.len + steps);
        defer cache.deinit();
        var ps = try inkling.PrefillScratch.init(m.device, c, inkling.default_chunk);
        defer ps.deinit();
        const hybrid = std.mem.eql(u8, mode, "greedy-hybrid");
        var decode: ?inkling.Scratch = if (std.mem.eql(u8, mode, "greedy-decode") or hybrid) try inkling.Scratch.init(m.device, c) else null;
        defer if (decode) |*s| s.deinit();
        var at: usize = 0;
        while (at < ids.value.len) {
            const take = @min(ps.chunk, ids.value.len - at);
            try inkling.prefillWant(m, &ps, &cache, ids.value[at..][0..take], if (at + take == ids.value.len) .last else .none);
            at += take;
        }
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, ids.value);
        var proposer = try tf.lanes.SuffixLookup.init(gpa, .{});
        defer proposer.deinit();
        var token = argmax(ps.logits.slice(f32, vocab));
        var rounds: usize = 0;
        var drafted: usize = 0;
        var accepted_total: usize = 0;
        var draft_ns: u64 = 0;
        var save_ns: u64 = 0;
        var forward_ns: u64 = 0;
        var argmax_ns: u64 = 0;
        var restore_ns: u64 = 0;
        var replay_ns: u64 = 0;
        var decode_ns: u64 = 0;
        const start = now();
        while (history.items.len < ids.value.len + steps) {
            try history.append(gpa, token);
            if (history.items.len == ids.value.len + steps) break;
            if (!hybrid and decode != null) {
                const decode_start = now();
                const s = &decode.?;
                try inkling.position(m, s, &cache, token);
                token = argmax(s.logits.slice(f32, vocab));
                decode_ns += now() - decode_start;
                rounds += 1;
                continue;
            }
            if (std.mem.eql(u8, mode, "greedy-prefill")) {
                try inkling.prefillWant(m, &ps, &cache, (&token)[0..1], .all);
                token = argmax(ps.logits.slice(f32, vocab));
                rounds += 1;
                continue;
            }
            const most: usize = if (hybrid) (if (std.c.getenv("TF_INKLING_HYBRID_DRAFTS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch inkling.hybrid_drafts else inkling.hybrid_drafts) else 15;
            const draft_start = now();
            const draft = try proposer.propose(history.items, @intCast(most));
            draft_ns += now() - draft_start;
            const count: usize = @min(draft.len, @min(most, ids.value.len + steps - history.items.len - 1));
            const row_count = try std.math.add(usize, count, 1);
            if (count == 0 and hybrid) {
                const decode_start = now();
                try inkling.position(m, &decode.?, &cache, token);
                token = argmax(decode.?.logits.slice(f32, vocab));
                decode_ns += now() - decode_start;
                rounds += 1;
                continue;
            }
            if (count == 0) {
                try inkling.prefillWant(m, &ps, &cache, (&token)[0..1], .all);
                token = argmax(ps.logits.slice(f32, vocab));
                rounds += 1;
                continue;
            }
            var rows: [16]u32 = undefined;
            rows[0] = token;
            @memcpy(rows[1..][0..count], draft[0..count]);
            var predictions: [16]u32 = undefined;
            var round_times = inkling.verify.RoundTimes{};
            inkling.verify.round_times = &round_times;
            var verify = try inkling.Verification.begin(m, &ps, &cache, rows[0..row_count], predictions[0..row_count]);
            var accepted: usize = 0;
            while (accepted < count and predictions[accepted] == rows[accepted + 1]) : (accepted += 1) {}
            try verify.finish(rows[0..row_count], try std.math.add(usize, accepted, 1));
            inkling.verify.round_times = null;
            save_ns += round_times.save;
            forward_ns += round_times.forward;
            argmax_ns += round_times.argmax;
            restore_ns += round_times.restore;
            replay_ns += round_times.replay;
            try proposer.observe(@intCast(count), @intCast(accepted));
            try history.appendSlice(gpa, predictions[0..accepted]);
            token = predictions[accepted];
            rounds += 1;
            drafted += count;
            accepted_total += accepted;
        }
        const elapsed_ns = now() - start;
        const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = std.mem.sliceAsBytes(history.items[ids.value.len..]) });
        std.debug.print("{s}: tokens={d} rounds={d} drafted={d} accepted={d} seconds={d:.3}\n", .{ mode, steps, rounds, drafted, accepted_total, seconds });
        std.debug.print("roundcost draft_ns={d} save_ns={d} forward_ns={d} argmax_ns={d} restore_ns={d} replay_ns={d} decode_ns={d} elapsed_ns={d}\n", .{ draft_ns, save_ns, forward_ns, argmax_ns, restore_ns, replay_ns, decode_ns, elapsed_ns });
        return;
    }
    if (std.mem.eql(u8, mode, "verifycheck")) {
        if (args.len != 5 or ids.value.len < c.text.window + 21) return error.ExpectedArgs;
        const accepted = try std.fmt.parseInt(usize, args[4], 10);
        if (accepted == 0 or accepted >= 16) return error.ExpectedArgs;
        const base: usize = c.text.window - 1;
        var a = try inkling.Cache.init(m.device, c, base + 32);
        defer a.deinit();
        var bcache = try inkling.Cache.init(m.device, c, base + 32);
        defer bcache.deinit();
        var sa = try inkling.PrefillScratch.init(m.device, c, 512);
        defer sa.deinit();
        var sb = try inkling.PrefillScratch.init(m.device, c, 512);
        defer sb.deinit();
        var at: usize = 0;
        while (at < base) {
            const take = @min(sa.chunk, base - at);
            try inkling.prefillWant(m, &sa, &a, ids.value[at..][0..take], .none);
            try inkling.prefillWant(m, &sb, &bcache, ids.value[at..][0..take], .none);
            at += take;
        }
        const draft = ids.value[base..][0..16];
        var predictions: [16]u32 = undefined;
        var v = try inkling.Verification.begin(m, &sa, &a, draft, &predictions);
        const prefix = try gpa.dupe(f32, sa.logits.slice(f32, accepted * vocab));
        defer gpa.free(prefix);
        try v.finish(draft, accepted);
        try inkling.prefillWant(m, &sb, &bcache, draft[0..accepted], .all);
        var prefix_changed: usize = 0;
        var prefix_maximum: f32 = 0;
        for (prefix, sb.logits.slice(f32, accepted * vocab)) |lhs, rhs| {
            if (@as(u32, @bitCast(lhs)) != @as(u32, @bitCast(rhs))) prefix_changed += 1;
            prefix_maximum = @max(prefix_maximum, @abs(lhs - rhs));
        }
        const continuation = ids.value[base + 16 ..][0..5];
        try inkling.prefillWant(m, &sa, &a, continuation, .all);
        try inkling.prefillWant(m, &sb, &bcache, continuation, .all);
        const n = continuation.len * vocab;
        const x = sa.logits.slice(f32, n);
        const y = sb.logits.slice(f32, n);
        var changed: usize = 0;
        var maximum: f32 = 0;
        for (x, y) |lhs, rhs| {
            if (@as(u32, @bitCast(lhs)) != @as(u32, @bitCast(rhs))) changed += 1;
            maximum = @max(maximum, @abs(lhs - rhs));
        }
        std.debug.print("verifycheck accepted={d} prefix_changed={d} prefix_max_diff={e} continuation_changed={d} continuation_max_diff={e}\n", .{ accepted, prefix_changed, prefix_maximum, changed, maximum });
        return;
    }
    if (std.mem.eql(u8, mode, "roundbench")) {
        if (args.len != 5 and args.len != 6) return error.ExpectedArgs;
        const rounds = try std.fmt.parseInt(usize, args[4], 10);
        if (rounds == 0) return error.ExpectedArgs;
        const selected_width = if (args.len == 6) try std.fmt.parseInt(usize, args[5], 10) else 0;
        if (selected_width != 0 and selected_width != 1 and selected_width != 2 and selected_width != 4 and selected_width != 8 and selected_width != 16) return error.ExpectedArgs;
        var cache = try inkling.Cache.init(m.device, c, ids.value.len + 32);
        defer cache.deinit();
        var ps = try inkling.PrefillScratch.init(m.device, c, 512);
        defer ps.deinit();
        var at: usize = 0;
        while (at < ids.value.len) {
            const take = @min(ps.chunk, ids.value.len - at);
            try inkling.prefillWant(m, &ps, &cache, ids.value[at..][0..take], .none);
            at += take;
        }
        var tokens: [16]u32 = undefined;
        for (&tokens, 0..) |*token, i| token.* = ids.value[i % ids.value.len];
        var predictions: [16]u32 = undefined;
        var initial = try inkling.Verification.begin(m, &ps, &cache, &tokens, &predictions);
        const widths = [_]usize{ 1, 2, 4, 8, 16 };
        for (0..rounds) |round| {
            for (0..widths.len) |i| {
                const width = widths[if (round % 2 == 0) i else widths.len - 1 - i];
                if (selected_width != 0 and selected_width != width) continue;
                try initial.reset();
                var times = inkling.verify.RoundTimes{};
                inkling.verify.round_times = &times;
                const start = now();
                var v = try inkling.Verification.begin(m, &ps, &cache, tokens[0..width], predictions[0..width]);
                const accepted = if (want_timeline) 0 else width / 2;
                try v.finish(tokens[0..width], accepted);
                const total = now() - start;
                inkling.verify.round_times = null;
                std.debug.print("roundbench width={d} accepted={d} save_ns={d} forward_ns={d} argmax_ns={d} restore_ns={d} replay_ns={d} total_ns={d}\n", .{ width, accepted, times.save, times.forward, times.argmax, times.restore, times.replay, total });
            }
        }
        try initial.finish(&tokens, 0);
        return;
    }
    if (std.mem.eql(u8, mode, "verifybench")) {
        if (args.len != 6) return error.ExpectedArgs;
        const width = try std.fmt.parseInt(usize, args[4], 10);
        const rounds = try std.fmt.parseInt(usize, args[5], 10);
        if ((width != 2 and width != 4 and width != 8 and width != 16) or rounds == 0) return error.ExpectedArgs;
        var cache = try inkling.Cache.init(m.device, c, ids.value.len + 30 * rounds);
        defer cache.deinit();
        var ps = try inkling.PrefillScratch.init(m.device, c, @max(512, width));
        defer ps.deinit();
        var at: usize = 0;
        while (at < ids.value.len) {
            const take = @min(ps.chunk, ids.value.len - at);
            try inkling.prefillWant(m, &ps, &cache, ids.value[at..][0..take], .none);
            at += take;
        }
        const tokens = try gpa.alloc(u32, 16);
        defer gpa.free(tokens);
        for (tokens, 0..) |*token, i| token.* = ids.value[i % ids.value.len];
        const walls = try gpa.alloc(f64, 4 * rounds);
        defer gpa.free(walls);
        const widths = [_]usize{ 2, 4, 8, 16 };
        for (0..rounds) |round| {
            for (0..widths.len) |index| {
                const w = if (round % 2 == 0) index else widths.len - 1 - index;
                const rows = widths[w];
                if (rows < width) continue;
                const start = now();
                try inkling.prefillWant(m, &ps, &cache, tokens[0..rows], .all);
                walls[w * rounds + round] = @as(f64, @floatFromInt(now() - start)) / 1e6;
                std.mem.doNotOptimizeAway(argmax(ps.logits.slice(f32, vocab)));
            }
        }
        for (widths, 0..) |rows, w| {
            if (rows < width) continue;
            const samples = walls[w * rounds ..][0..rounds];
            std.mem.sort(f64, samples, {}, std.sort.asc(f64));
            std.debug.print("verifybench width={d} rounds={d} median_ms={d:.3} min_ms={d:.3} max_ms={d:.3}\n", .{ rows, rounds, samples[rounds / 2], samples[0], samples[rounds - 1] });
        }
        return;
    }
    if (std.mem.eql(u8, mode, "greedy")) {
        if (args.len != 6) {
            std.debug.print("{s}", .{usage});
            return error.ExpectedArgs;
        }
        const steps = try std.fmt.parseInt(usize, args[4], 10);
        var cache = try inkling.Cache.init(m.device, c, ids.value.len + steps);
        defer cache.deinit();
        var scratch = try inkling.Scratch.init(m.device, c);
        defer scratch.deinit();
        const out = try gpa.alloc(u32, steps);
        defer gpa.free(out);
        try run(m, &scratch, &cache, ids.value);
        for (0..steps) |i| {
            out[i] = argmax(scratch.logits.slice(f32, vocab));
            try inkling.position(m, &scratch, &cache, out[i]);
        }
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = std.mem.sliceAsBytes(out) });
        std.debug.print("inkling2_loop greedy: {d} prompt tokens, {d} steps, {d} ms\n", .{ ids.value.len, steps, (now() - t0) / std.time.ns_per_ms });
        return;
    }
    if (std.mem.eql(u8, mode, "prefillroutes")) {
        if (args.len != 7) return error.ExpectedArgs;
        const chunk = try std.fmt.parseInt(usize, args[4], 10);
        const end = try std.fmt.parseInt(usize, args[6], 10);
        if (chunk == 0 or end >= ids.value.len) return error.ExpectedArgs;
        var cache = try inkling.Cache.init(m.device, c, ids.value.len);
        defer cache.deinit();
        var ps = try inkling.PrefillScratch.init(m.device, c, chunk);
        defer ps.deinit();
        const count = c.slots() * (chunk * (c.text.topk + c.text.experts) + c.text.experts);
        const routes = try m.device.buffer(count * 4, mtl.ResourceOptions.shared);
        defer routes.deinit();
        var offset: usize = 0;
        while (offset <= end) {
            const take = @min(chunk, ids.value.len - offset);
            ps.routes = routes;
            try inkling.prefillWant(m, &ps, &cache, ids.value[offset..][0..take], .none);
            const actual = c.slots() * (take * (c.text.topk + c.text.experts) + c.text.experts);
            const path = try std.fmt.allocPrint(gpa, "{s}.{d}", .{ args[5], offset });
            defer gpa.free(path);
            try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(routes.slice(f32, actual)) });
            offset += take;
        }
        return;
    }
    if (std.mem.eql(u8, mode, "prefillrow")) {
        if (args.len != 7) return error.ExpectedArgs;
        const chunk = try std.fmt.parseInt(usize, args[4], 10);
        const at = try std.fmt.parseInt(usize, args[6], 10);
        if (chunk == 0 or at >= ids.value.len) return error.ExpectedArgs;
        var cache = try inkling.Cache.init(m.device, c, ids.value.len);
        defer cache.deinit();
        var ps = try inkling.PrefillScratch.init(m.device, c, chunk);
        defer ps.deinit();
        const count = c.slots() * (9 * c.text.hidden + c.text.topk + 2 * c.text.experts);
        const capture_buf = try m.device.buffer(count * 4, mtl.ResourceOptions.shared);
        defer capture_buf.deinit();
        var offset: usize = 0;
        while (offset <= at) {
            const take = @min(chunk, ids.value.len - offset);
            if (at < offset + take) {
                ps.capture = capture_buf;
                ps.capture_row = at - offset;
            }
            try inkling.prefillWant(m, &ps, &cache, ids.value[offset..][0..take], .none);
            offset += take;
        }
        const rows = c.slots() * 9 * c.text.hidden;
        const data = capture_buf.slice(f32, count);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = std.mem.sliceAsBytes(data[0..rows]) });
        const picks_path = try std.fmt.allocPrint(gpa, "{s}.picks", .{args[5]});
        defer gpa.free(picks_path);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = picks_path, .data = std.mem.sliceAsBytes(data[rows .. rows + c.slots() * c.text.topk]) });
        const router_path = try std.fmt.allocPrint(gpa, "{s}.router", .{args[5]});
        defer gpa.free(router_path);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = router_path, .data = std.mem.sliceAsBytes(data[rows + c.slots() * c.text.topk ..]) });
        std.debug.print("inkling2_loop prefillrow: position {d} chunk {d}\n", .{ at, chunk });
        return;
    }
    if (std.mem.eql(u8, mode, "prefill")) {
        if (args.len != 6) {
            std.debug.print("{s}", .{usage});
            return error.ExpectedArgs;
        }
        const chunk = try std.fmt.parseInt(usize, args[4], 10);
        if (chunk == 0) {
            std.debug.print("{s}", .{usage});
            return error.ExpectedArgs;
        }
        var cache = try inkling.Cache.init(m.device, c, ids.value.len);
        defer cache.deinit();
        var ps = try inkling.PrefillScratch.init(m.device, c, chunk);
        defer ps.deinit();
        const outp = try gpa.alloc(f32, ids.value.len * vocab);
        defer gpa.free(outp);
        const p0 = now();
        var at: usize = 0;
        while (at < ids.value.len) {
            const take = @min(chunk, ids.value.len - at);
            try inkling.prefill(m, &ps, &cache, ids.value[at..][0..take]);
            @memcpy(outp[at * vocab ..][0..take * vocab], ps.logits.slice(f32, take * vocab));
            at += take;
        }
        const p1 = now();
        const ms: f64 = @as(f64, @floatFromInt(p1 - p0)) / 1e6;
        const rate: f64 = @as(f64, @floatFromInt(ids.value.len)) / (ms / 1000.0);
        const output = try std.Io.Dir.cwd().createFile(init.io, args[5], .{});
        defer output.close(init.io);
        const output_bytes = std.mem.sliceAsBytes(outp);
        var written: usize = 0;
        while (written < output_bytes.len) {
            const end = @min(written + (1 << 30), output_bytes.len);
            try output.writeStreamingAll(init.io, output_bytes[written..end]);
            written = end;
        }
        std.debug.print("inkling2_loop prefill: {d} tokens in {d:.1} ms, {d:.1} tok/s chunk {d}\n", .{ ids.value.len, ms, rate, chunk });
        return;
    }
    if (std.mem.eql(u8, mode, "bench")) {
        if (args.len < 5 or args.len > 7) {
            std.debug.print("{s}", .{usage});
            return error.ExpectedArgs;
        }
        const steps = try std.fmt.parseInt(usize, args[4], 10);
        if (steps == 0) {
            std.debug.print("{s}", .{usage});
            return error.ExpectedArgs;
        }
        const warm = if (args.len >= 6) try std.fmt.parseInt(usize, args[5], 10) else 8;
        const chunk: usize = if (args.len == 7) try std.fmt.parseInt(usize, args[6], 10) else 0;
        var cache = try inkling.Cache.init(m.device, c, ids.value.len + warm + steps + 1);
        defer cache.deinit();
        var scratch = try inkling.Scratch.init(m.device, c);
        defer scratch.deinit();
        const walls = try gpa.alloc(f64, steps);
        defer gpa.free(walls);
        if (chunk == 0) {
            try run(m, &scratch, &cache, ids.value);
        } else {
            var ps = try inkling.PrefillScratch.init(m.device, c, chunk);
            defer ps.deinit();
            const p0 = now();
            var at: usize = 0;
            while (at < ids.value.len) {
                const take = @min(chunk, ids.value.len - at);
                const last = at + take >= ids.value.len;
                try inkling.prefillWant(m, &ps, &cache, ids.value[at..][0..take], if (last) .last else .none);
                at += take;
            }
            const p1 = now();
            const ms: f64 = @as(f64, @floatFromInt(p1 - p0)) / 1e6;
            const rate: f64 = @as(f64, @floatFromInt(ids.value.len)) / (ms / 1000.0);
            std.debug.print("inkling2_loop prefill: {d} tokens in {d:.1} ms, {d:.1} tok/s chunk {d}\n", .{ ids.value.len, ms, rate, chunk });
            @memcpy(scratch.logits.slice(f32, vocab), ps.logits.slice(f32, vocab));
        }
        var tok = argmax(scratch.logits.slice(f32, vocab));
        for (0..warm) |_| {
            try inkling.position(m, &scratch, &cache, tok);
            tok = argmax(scratch.logits.slice(f32, vocab));
        }
        var argmax_ns: u64 = 0;
        metrics_ms = 0;
        metrics_wait_ms = 0;
        metrics_gpu_ms = 0;
        metrics_commits = 0;
        metrics_seen = false;
        mtl.pass_metrics.begin(true, "inkling2-bench", @intCast(steps), mtl.clock.seconds);
        for (0..steps) |i| {
            const start = now();
            m.queue.roundBegin("inkling2-bench");
            try inkling.position(m, &scratch, &cache, tok);
            m.queue.roundEnd();
            const stop = now();
            walls[i] = @as(f64, @floatFromInt(stop - start)) / 1e6;
            tok = argmax(scratch.logits.slice(f32, vocab));
            argmax_ns += now() - stop;
        }
        mtl.pass_metrics.end();
        std.mem.sort(f64, walls, {}, std.sort.asc(f64));
        const median = if (steps % 2 == 1) walls[steps / 2] else (walls[steps / 2 - 1] + walls[steps / 2]) / 2;
        const n: f64 = @floatFromInt(steps);
        const host = if (metrics_seen) (metrics_ms - metrics_wait_ms - @as(f64, @floatFromInt(argmax_ns)) / 1e6) / n else 0;
        const gpu = if (metrics_seen) metrics_gpu_ms / n else 0;
        std.debug.print("inkling2_loop bench at {d}: {d:.3} ms a step wall (median of {d}), {d:.3} encode, {d:.3} GPU, {d} warmup steps, {d} command buffers\n", .{ ids.value.len, median, steps, host, gpu, warm, metrics_commits });
        std.debug.print("receipt: seen={} ms={d:.3} wait={d:.3} gpu={d:.3} commits={d}\n", .{ metrics_seen, metrics_ms, metrics_wait_ms, metrics_gpu_ms, metrics_commits });
        return;
    }
    var cache = try inkling.Cache.init(m.device, c, ids.value.len);
    defer cache.deinit();
    var scratch = try inkling.Scratch.init(m.device, c);
    defer scratch.deinit();
    if (std.mem.eql(u8, mode, "reset")) {
        const first = try gpa.alloc(f32, ids.value.len * vocab);
        defer gpa.free(first);
        try capture(m, &scratch, &cache, ids.value, first, vocab);
        cache.len = 0;
        const second = try gpa.alloc(f32, ids.value.len * vocab);
        defer gpa.free(second);
        try capture(m, &scratch, &cache, ids.value, second, vocab);
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(first), std.mem.sliceAsBytes(second))) {
            std.debug.print("inkling2_loop reset: the rerun differs from the first pass\n", .{});
            return error.NotDeterministic;
        }
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(first) });
        std.debug.print("inkling2_loop reset: {d} rows reproduced bit for bit in {d} ms\n", .{ ids.value.len, (now() - t0) / std.time.ns_per_ms });
        return;
    }
    if (std.mem.eql(u8, mode, "continue")) {
        if (args.len != 6) {
            std.debug.print("{s}", .{usage});
            return error.ExpectedArgs;
        }
        const split = try std.fmt.parseInt(usize, args[4], 10);
        if (split == 0 or split >= ids.value.len) {
            std.debug.print("continue: the split must land inside the {d}-token history\n", .{ids.value.len});
            return error.ExpectedArgs;
        }
        cache.len = 0;
        const whole = try gpa.alloc(f32, ids.value.len * vocab);
        defer gpa.free(whole);
        try capture(m, &scratch, &cache, ids.value, whole, vocab);
        cache.len = 0;
        const head = try gpa.alloc(f32, split * vocab);
        defer gpa.free(head);
        try capture(m, &scratch, &cache, ids.value[0..split], head, vocab);
        const rest = try gpa.alloc(f32, (ids.value.len - split) * vocab);
        defer gpa.free(rest);
        try capture(m, &scratch, &cache, ids.value[split..], rest, vocab);
        const tail = whole[split * vocab ..];
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(head), std.mem.sliceAsBytes(whole[0 .. split * vocab])) or
            !std.mem.eql(u8, std.mem.sliceAsBytes(rest), std.mem.sliceAsBytes(tail)))
        {
            std.debug.print("inkling2_loop continue: the split pass differs from one pass\n", .{});
            return error.NotIncremental;
        }
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = std.mem.sliceAsBytes(whole) });
        std.debug.print("inkling2_loop continue: {d} tokens from a {d}-token split reproduce one pass in {d} ms\n", .{ ids.value.len, split, (now() - t0) / std.time.ns_per_ms });
        return;
    }
    if (std.mem.eql(u8, mode, "routes")) {
        if (args.len != 6) return error.ExpectedArgs;
        const end = try std.fmt.parseInt(usize, args[5], 10);
        if (end >= ids.value.len) return error.ExpectedArgs;
        var t = try inkling.Trace.init(gpa, c);
        defer t.deinit();
        const output = try std.Io.Dir.cwd().createFile(init.io, args[4], .{});
        defer output.close(init.io);
        const row_count = c.slots() * 9 * c.text.hidden;
        const count = row_count + c.slots() * (c.text.topk + 2 * c.text.experts);
        for (ids.value[0 .. end + 1]) |id| {
            try inkling.positionTrace(m, &scratch, &cache, id, &t);
            try output.writeStreamingAll(init.io, std.mem.sliceAsBytes(scratch.capture.slice(f32, count)[row_count..]));
        }
        return;
    }
    const out = try gpa.alloc(f32, ids.value.len * vocab);
    defer gpa.free(out);
    if (std.mem.eql(u8, mode, "logits")) {
        try capture(m, &scratch, &cache, ids.value, out, vocab);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(out) });
        std.debug.print("inkling2_loop logits: {d} rows in {d} ms\n", .{ ids.value.len, (now() - t0) / std.time.ns_per_ms });
    } else if (std.mem.eql(u8, mode, "trace")) {
        var t = try inkling.Trace.init(gpa, c);
        defer t.deinit();
        const at: ?usize = if (args.len > 5) try std.fmt.parseInt(usize, args[5], 10) else null;
        if (at) |a| {
            if (a >= ids.value.len) return error.ExpectedArgs;
        }
        for (ids.value, 0..) |id, i| {
            const tr: ?*inkling.Trace = if (at) |a| (if (i == a) &t else null) else &t;
            try inkling.positionTrace(m, &scratch, &cache, id, tr);
            @memcpy(out[i * vocab ..][0..vocab], scratch.logits.slice(f32, vocab));
        }
        try t.write(init.io, args[4]);
        std.debug.print("inkling2_loop trace: {d} rows, {d} tensors in {d} ms\n", .{ ids.value.len, t.names.items.len, (now() - t0) / std.time.ns_per_ms });
    } else if (std.mem.eql(u8, mode, "tracerow")) {
        if (args.len != 6) {
            std.debug.print("{s}", .{usage});
            return error.ExpectedArgs;
        }
        const at = try std.fmt.parseInt(usize, args[5], 10);
        if (at >= ids.value.len) return error.ExpectedArgs;
        var t = try inkling.Trace.init(gpa, c);
        defer t.deinit();
        const per = t.names.items.len / c.slots();
        for (ids.value, 0..) |id, i| {
            if (i > at) break;
            const tr: ?*inkling.Trace = if (i == at) &t else null;
            try inkling.positionTrace(m, &scratch, &cache, id, tr);
        }
        const row_count = c.slots() * per * m.config.text.hidden;
        const rows = scratch.capture.slice(f32, row_count + c.slots() * (c.text.topk + 2 * c.text.experts));
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(rows[0..row_count]) });
        const picks_path = try std.fmt.allocPrint(gpa, "{s}.picks", .{args[4]});
        defer gpa.free(picks_path);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = picks_path, .data = std.mem.sliceAsBytes(rows[row_count .. row_count + c.slots() * c.text.topk]) });
        const router_path = try std.fmt.allocPrint(gpa, "{s}.router", .{args[4]});
        defer gpa.free(router_path);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = router_path, .data = std.mem.sliceAsBytes(rows[row_count + c.slots() * c.text.topk ..]) });
        std.debug.print("inkling2_loop tracerow: position {d} of {d}, {d} rows in {d} ms\n", .{ at, ids.value.len, c.slots() * per, (now() - t0) / std.time.ns_per_ms });
    } else {
        std.debug.print("{s}", .{usage});
        return error.UnknownMode;
    }
}

fn tokenize(gpa: std.mem.Allocator, io: std.Io, tok_path: []const u8, text_path: []const u8, out_path: []const u8) !void {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, text_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    var tok = try tk.loadTokenizer(io, gpa, tok_path);
    defer tok.deinit();
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    var w: std.Io.Writer.Allocating = .init(gpa);
    defer w.deinit();
    try std.json.Stringify.value(ids, .{}, &w.writer);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = w.written() });
    std.debug.print("inkling2_loop tokenize: {d} tokens\n", .{ids.len});
}

fn detokenize(gpa: std.mem.Allocator, io: std.Io, tok_path: []const u8, ids_path: []const u8, out_path: []const u8) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    const ids = try std.json.parseFromSlice([]u32, gpa, bytes, .{});
    defer ids.deinit();
    var tok = try tk.loadTokenizer(io, gpa, tok_path);
    defer tok.deinit();
    const text = try tok.decode(gpa, ids.value, false);
    defer gpa.free(text);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = text });
    std.debug.print("inkling2_loop detokenize: {d} tokens\n", .{ids.value.len});
}

fn run(m: *const inkling.Model, s: *inkling.Scratch, cache: *inkling.Cache, ids: []const u32) !void {
    for (ids) |id| try inkling.position(m, s, cache, id);
}

fn capture(m: *const inkling.Model, s: *inkling.Scratch, cache: *inkling.Cache, ids: []const u32, out: []f32, vocab: usize) !void {
    for (ids, 0..) |id, i| {
        try inkling.position(m, s, cache, id);
        @memcpy(out[i * vocab ..][0..vocab], s.logits.slice(f32, vocab));
    }
}

fn argmax(x: []const f32) u32 {
    var best: usize = 0;
    for (x, 0..) |v, i| {
        if (v > x[best]) best = i;
    }
    return @intCast(best);
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
