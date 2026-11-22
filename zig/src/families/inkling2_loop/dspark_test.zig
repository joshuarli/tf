const std = @import("std");
const model = @import("model.zig");
const ds = @import("dspark.zig");
const t = std.testing;

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(r: *Reader, n: usize) ![]const u8 {
        if (n > r.bytes.len - r.pos) return error.BadFixture;
        const out = r.bytes[r.pos..][0..n];
        r.pos += n;
        return out;
    }
    fn int(r: *Reader) !u32 { return std.mem.readInt(u32, (try r.take(4))[0..4], .little); }
};

fn timeRounds(draft: *ds.Drafter, features: []const f32, anchor: u32) !void {
    const length = draft.length;
    for ([_]u32{ 4, 8 }) |width| {
        var samples: [21]i96 = undefined;
        for (0..24) |i| {
            draft.length = length;
            const start = std.Io.Clock.awake.now(t.io).toNanoseconds();
            const ids = try draft.propose(features, anchor, width);
            const elapsed = std.Io.Clock.awake.now(t.io).toNanoseconds() - start;
            try t.expectEqual(@as(usize, width - 1), ids.len);
            if (i >= 3) samples[i - 3] = elapsed;
        }
        std.mem.sort(i96, &samples, {}, std.sort.asc(i96));
        std.debug.print("dspark timing width={d} samples=21 median_ms={d:.3} min_ms={d:.3} max_ms={d:.3} context={d}\n", .{
            width, @as(f64, @floatFromInt(samples[10])) / 1e6,
            @as(f64, @floatFromInt(samples[0])) / 1e6,
            @as(f64, @floatFromInt(samples[20])) / 1e6, length,
        });
    }
    draft.length = length;
}

test "dspark proposals and logits match captured MLX rounds" {
    const fixture_path = std.c.getenv("TF_INKLING2_DSPARK_FIXTURE") orelse return error.SkipZigTest;
    const model_path = std.c.getenv("TF_INKLING2_LOOP_DIR") orelse return error.SkipZigTest;
    const a = t.allocator;
    const file = try std.Io.Dir.cwd().readFileAlloc(t.io, std.mem.span(fixture_path), a, .limited(2_000_000_000));
    defer a.free(file);
    var r = Reader{ .bytes = file };
    try t.expectEqualStrings("DSPARK01", try r.take(8));
    const groups = try r.int();
    const m = try model.Model.load(a, t.io, std.mem.span(model_path));
    defer m.deinit();
    var weights = try ds.Weights.init(m, std.mem.span(model_path));
    defer weights.deinit();
    var draft = try ds.Drafter.init(m, &weights);
    defer draft.deinit();
    var blocks: usize = 0;
    var count4: usize = 0;
    var count8: usize = 0;
    var mismatches: usize = 0;
    var max_abs: f32 = 0;
    var max_relative: f32 = 0;
    var bad_logits: usize = 0;
    for (0..groups) |_| {
        draft.reset();
        const n = try r.int();
        for (0..n) |_| {
            const anchor = try r.int();
            const width = try r.int();
            const rows = try r.int();
            const vocab = try r.int();
            try t.expectEqual(m.config.text.unpadded_vocab, vocab);
            try t.expect(width >= 2 and width <= 8);
            const feature_count = @as(usize, rows) * 4 * 2304;
            const features = try a.alloc(f32, feature_count);
            defer a.free(features);
            @memcpy(std.mem.sliceAsBytes(features), try r.take(feature_count * 4));
            const expected = try r.take(@as(usize, width - 1) * vocab * 4);
            const ids = try r.take(@as(usize, width - 1) * 4);
            const actual = try draft.propose(features, anchor, width);
            for (actual, 0..) |id, i| {
                const wanted = std.mem.readInt(u32, ids[i * 4 ..][0..4], .little);
                if (id != wanted) {
                    mismatches += 1;
                    std.debug.print("block {d} step {d}: got {d}, expected {d}\n", .{ blocks, i, id, wanted });
                }
            }
            const logits = draft.logits.slice(f32, @as(usize, width - 1) * vocab);
            for (logits, 0..) |got, i| {
                const want: f32 = @bitCast(std.mem.readInt(u32, expected[i * 4 ..][0..4], .little));
                const diff = @abs(got - want);
                const relative = diff / @max(@abs(want), 1);
                max_abs = @max(max_abs, diff);
                max_relative = @max(max_relative, relative);
                if (!std.math.isFinite(got) or relative > 0.001) bad_logits += 1;
            }
            if (blocks == 0 and std.c.getenv("TF_INKLING2_DSPARK_TIMING") != null) {
                try timeRounds(&draft, features[features.len - 4 * 2304 ..], anchor);
                try t.expectEqual(@as(usize, 0), mismatches);
                try t.expectEqual(@as(usize, 0), bad_logits);
                return;
            }
            blocks += 1;
            if (width == 4) count4 += 1;
            if (width == 8) count8 += 1;
        }
    }
    std.debug.print("dspark blocks={d} width4={d} width8={d} argmax_mismatches={d} max_abs={e} max_relative={e} bad_logits={d}\n", .{ blocks, count4, count8, mismatches, max_abs, max_relative, bad_logits });
    try t.expectEqual(file.len, r.pos);
    try t.expect(count4 >= 16 and count8 >= 16);
    try t.expectEqual(@as(usize, 0), mismatches);
    try t.expectEqual(@as(usize, 0), bad_logits);
}
