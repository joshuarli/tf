//! `decode-logits`: teacher-forced logits through the served decode paths, one-row batch steps or grouped verifies.
const std = @import("std");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;

const History = struct { name: []const u8, token_ids: []const u32 };

/// Each history from an empty cache: a batch step a token, or grouped verifies of `width` rows that all commit.
pub fn main(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len != 6) return error.ExpectedArgs;
    const gpa = init.gpa;
    const width = try std.fmt.parseInt(usize, args[4], 10);
    if (width == 0 or width > inkling.group.max_rows) return error.ExpectedArgs;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    const parsed = try std.json.parseFromSlice([]const History, gpa, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = try inkling.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    const c = &m.config;
    const vocab: usize = c.text.unpadded_vocab;
    var bs = try inkling.BatchScratch.init(m.device, c, inkling.group.max_rows);
    defer bs.deinit();
    var verifier = try inkling.group.Verifier.init(m.device, c);
    defer verifier.deinit();
    try std.Io.Dir.cwd().createDirPath(init.io, args[5]);
    for (parsed.value) |history| {
        const ids = history.token_ids;
        const out = try gpa.alloc(f32, ids.len * vocab);
        defer gpa.free(out);
        var cache = try inkling.Cache.init(m.device, c, ids.len + inkling.group.max_rows);
        defer cache.deinit();
        var at: usize = 0;
        while (at < ids.len) {
            const n = @min(width, ids.len - at);
            if (width == 1) {
                try inkling.batchStep(m, &bs, &.{&cache}, ids[at..][0..1], out[at * vocab ..][0..vocab]);
            } else {
                const g = try verifier.group(&.{&cache}, &.{n});
                try verifier.forward(m, &bs, &g, ids[at..][0..n]);
                @memcpy(out[at * vocab ..][0 .. n * vocab], bs.rows.logits.slice(f32, n * vocab));
                try verifier.commit(m, &g, &.{n - 1});
            }
            at += n;
        }
        var name: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&name, "{s}/{s}.bin", .{ args[5], history.name });
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(out) });
        std.debug.print("decode-logits {s}: {d} rows, width {d}\n", .{ history.name, ids.len, width });
    }
}
