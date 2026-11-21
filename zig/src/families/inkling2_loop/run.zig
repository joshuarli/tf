//! `tf-inkling2-loop-run <model> <tokens.json> logits <out>`: every position's float32 logits from a teacher-forced
//! pass on Metal, which the campaign's parity check compares with `campaign/ref/inkling2_loop_ref.py`.
const std = @import("std");
const mtl = @import("metal");
const inkling = @import("native.zig");

const usage =
    \\tf-inkling2-loop-run <model> <tokens.json> logits <out>   f32 logits [tokens, 200058], one pass
    \\tf-inkling2-loop-run <model> <tokens.json> layer <out>    one layer's output row at the token's position
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) {
        std.debug.print("{s}", .{usage});
        return error.ExpectedArgs;
    }
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    const ids = try std.json.parseFromSlice([]u32, gpa, bytes, .{});
    defer ids.deinit();
    if (ids.value.len == 0) return error.EmptyPrompt;
    const m = try inkling.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    const c = &m.config;
    var cache = try inkling.Cache.init(m.device, c, ids.value.len);
    defer cache.deinit();
    var scratch = try inkling.Scratch.init(m.device, c);
    defer scratch.deinit();
    const vocab: usize = c.text.unpadded_vocab;
    const out = try gpa.alloc(f32, ids.value.len * vocab);
    defer gpa.free(out);
    const mode = args[3];
    const t0 = now();
    for (ids.value, 0..) |id, i| {
        try inkling.position(m, &scratch, &cache, id);
        if (std.mem.eql(u8, mode, "logits")) @memcpy(out[i * vocab ..][0..vocab], scratch.logits.slice(f32, vocab));
    }
    if (!std.mem.eql(u8, mode, "logits")) {
        std.debug.print("{s}", .{usage});
        return error.UnknownMode;
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(out) });
    std.debug.print("inkling2_loop logits: {d} rows in {d} ms\n", .{ ids.value.len, (now() - t0) / std.time.ns_per_ms });
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
