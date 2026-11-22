const std = @import("std");
const inkling = @import("tensorfold").inkling2_loop;
pub const tf_metal4 = true;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) return error.ExpectedModelPromptSteps;
    const gpa = init.arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(64 << 20));
    const ids = try std.json.parseFromSlice([]u32, gpa, bytes, .{});
    const steps = try std.fmt.parseInt(usize, args[3], 10);
    if (ids.value.len == 0 or steps < 2) return error.InvalidSize;
    const m = try inkling.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    const vocab: usize = m.config.text.unpadded_vocab;
    var cache = try inkling.Cache.init(m.device, &m.config, ids.value.len + steps);
    defer cache.deinit();
    var scratch = try inkling.Scratch.init(m.device, &m.config);
    defer scratch.deinit();
    var ps = try inkling.PrefillScratch.init(m.device, &m.config, 128);
    defer ps.deinit();
    const out = try gpa.alloc(u32, steps);
    const start = now();
    var at: usize = 0;
    while (at < ids.value.len) {
        const take = @min(128, ids.value.len - at);
        try inkling.prefillWant(m, &ps, &cache, ids.value[at..][0..take], if (at + take == ids.value.len) .last else .none);
        at += take;
    }
    @memcpy(scratch.logits.slice(f32, vocab), ps.logits.slice(f32, vocab));
    out[0] = argmax(scratch.logits.slice(f32, vocab));
    const first = now();
    for (1..steps) |i| {
        try inkling.position(m, &scratch, &cache, out[i - 1]);
        out[i] = argmax(scratch.logits.slice(f32, vocab));
    }
    const end = now();
    std.debug.print("prompt_tokens={d} ttft_s={d:.6} prefill_tps={d:.6} decode_tps={d:.6} generated_ids=[", .{ ids.value.len, @as(f64, @floatFromInt(first - start)) / 1e9, @as(f64, @floatFromInt(ids.value.len)) * 1e9 / @as(f64, @floatFromInt(first - start)), @as(f64, @floatFromInt(steps - 1)) * 1e9 / @as(f64, @floatFromInt(end - first)) });
    for (out, 0..) |token, i| std.debug.print("{s}{d}", .{ if (i == 0) "" else ",", token });
    std.debug.print("]\n", .{});
}

fn argmax(x: []const f32) u32 {
    var best: usize = 0;
    for (x, 0..) |v, i| if (v > x[best]) { best = i; };
    return @intCast(best);
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
