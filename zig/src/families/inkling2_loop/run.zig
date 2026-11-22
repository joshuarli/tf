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
    \\tf-inkling2-loop-run <model> <tokens.json> trace <out> [at]  per-tensor summaries in the reference's order, the whole pass or only position <at>
    \\tf-inkling2-loop-run <model> <tokens.json> prefillrow <chunk> <out> <at>  capture a chunked prefill row
    \\tf-inkling2-loop-run <model> <tokens.json> tracerow <out> <at>  the raw per-tensor rows of position <at> as f32, [slots][9][hidden]
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
    if (std.mem.eql(u8, mode, "prefillrow")) {
        if (args.len != 7) return error.ExpectedArgs;
        const chunk = try std.fmt.parseInt(usize, args[4], 10);
        const at = try std.fmt.parseInt(usize, args[6], 10);
        if (chunk == 0 or at >= ids.value.len) return error.ExpectedArgs;
        var cache = try inkling.Cache.init(m.device, c, ids.value.len);
        defer cache.deinit();
        var ps = try inkling.PrefillScratch.init(m.device, c, chunk);
        defer ps.deinit();
        const count = c.slots() * (9 * c.text.hidden + c.text.topk);
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
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = picks_path, .data = std.mem.sliceAsBytes(data[rows..]) });
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
        const rows = scratch.capture.slice(f32, row_count + c.slots() * c.text.topk);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(rows[0..row_count]) });
        const picks_path = try std.fmt.allocPrint(gpa, "{s}.picks", .{args[4]});
        defer gpa.free(picks_path);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = picks_path, .data = std.mem.sliceAsBytes(rows[row_count..]) });
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
