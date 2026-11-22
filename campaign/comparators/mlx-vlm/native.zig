const std = @import("std");
const inkling = @import("tensorfold").inkling2_loop;
pub const tf_metal4 = true;
const context_limit: usize = 32768;

const Timing = struct {
    prefill_seconds: f64,
    decode_seconds: f64,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4 and args.len != 5) {
        std.debug.print("usage: {s} MODEL PROMPT_IDS_JSON GENERATED_TOKENS [WARMUP_RUNS]\n", .{args[0]});
        return error.ExpectedModelPromptSteps;
    }
    const gpa = init.arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(64 << 20));
    const ids = try std.json.parseFromSlice([]u32, gpa, bytes, .{});
    defer ids.deinit();
    const steps = try std.fmt.parseInt(usize, args[3], 10);
    const warmups = if (args.len == 5) try std.fmt.parseInt(usize, args[4], 10) else 0;
    if (ids.value.len == 0 or steps < 2) return error.InvalidSize;
    // Validate before loading weights or allocating GPU state; subtraction avoids overflow.
    if (steps > context_limit or ids.value.len > context_limit - steps) {
        std.debug.print("prompt ({d}) + generated tokens ({d}) exceeds the {d}-token context cap\n", .{
            ids.value.len, steps, context_limit,
        });
        return error.ContextFull;
    }
    const m = try inkling.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    const vocab: usize = m.config.text.unpadded_vocab;
    for (ids.value) |id| {
        if (id >= vocab) return error.InvalidToken;
    }
    var cache = try inkling.Cache.init(m.device, &m.config, context_limit);
    defer cache.deinit();
    var scratch = try inkling.Scratch.init(m.device, &m.config);
    defer scratch.deinit();
    var ps = try inkling.PrefillScratch.init(m.device, &m.config, inkling.default_chunk);
    defer ps.deinit();
    const out = try gpa.alloc(u32, steps);
    defer gpa.free(out);
    const reference = try gpa.alloc(u32, if (warmups == 0) 0 else steps);
    defer gpa.free(reference);
    for (0..warmups) |run| {
        _ = try generate(m, &scratch, &ps, &cache, ids.value, out);
        if (run == 0) {
            @memcpy(reference, out);
        } else {
            try requireSame(reference, out);
        }
    }
    const timing = try generate(m, &scratch, &ps, &cache, ids.value, out);
    if (warmups > 0) try requireSame(reference, out);

    // One complete JSON line on stderr, alongside any model-loading diagnostics.
    std.debug.print(
        "{{\"type\":\"tf_inkling_bench\",\"model_type\":\"inkling2_loop\"," ++
            "\"prompt_tokens\":{d},\"decode_tokens\":{d},\"generated_tokens\":{d}," ++
            "\"warmup_runs\":{d},\"prefill_seconds\":{d:.9},\"decode_seconds\":{d:.9}," ++
            "\"prefill_chunk\":{d},\"context\":{d},\"cache_capacity_tokens\":{d}," ++
            "\"warmup_trial_match\":{s},\"decode_mode\":\"free-running greedy (fixed length)\"," ++
            "\"predictions\":[",
        .{
            ids.value.len,
            steps - 1,
            steps,
            warmups,
            timing.prefill_seconds,
            timing.decode_seconds,
            inkling.default_chunk,
            context_limit,
            cache.capacity,
            if (warmups > 0) "true" else "null",
        },
    );
    for (out, 0..) |token, i| std.debug.print("{s}{d}", .{ if (i == 0) "" else ",", token });
    std.debug.print("]}}\n", .{});
}

fn generate(
    m: *const inkling.Model,
    scratch: *inkling.Scratch,
    ps: *inkling.PrefillScratch,
    cache: *inkling.Cache,
    ids: []const u32,
    out: []u32,
) !Timing {
    // This is the family's reset contract (also used by its reset/continue checks).
    // At base zero, convolution taps before position zero are skipped; valid KV,
    // convolution history and per-forward scratch are overwritten before use.
    cache.len = 0;
    const vocab: usize = m.config.text.unpadded_vocab;
    const start = now();
    var at: usize = 0;
    while (at < ids.len) {
        const take = @min(inkling.default_chunk, ids.len - at);
        try inkling.prefillWant(m, ps, cache, ids[at..][0..take], if (at + take == ids.len) .last else .none);
        at += take;
    }
    out[0] = argmax(ps.logits.slice(f32, vocab));
    const first = now();
    // The prefill already predicts output zero. Ignore EOS and do exactly N-1
    // decode forwards, matching nano bench's fixed-length greedy accounting.
    for (1..out.len) |i| {
        try inkling.position(m, scratch, cache, out[i - 1]);
        out[i] = argmax(scratch.logits.slice(f32, vocab));
    }
    const end = now();
    return .{
        .prefill_seconds = @as(f64, @floatFromInt(first - start)) / 1e9,
        .decode_seconds = @as(f64, @floatFromInt(end - first)) / 1e9,
    };
}

fn requireSame(reference: []const u32, out: []const u32) !void {
    for (reference, out, 0..) |expected, actual, i| {
        if (expected != actual) {
            std.debug.print("warmup/trial mismatch at prediction {d}: expected {d}, got {d}\n", .{
                i, expected, actual,
            });
            return error.NotDeterministic;
        }
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
