//! A chosen row's hidden states read from the forward's own scratch after each layer cutoff.
const std = @import("std");
const tf = @import("tensorfold");
const fl = tf.flint;

/// Records per row: `layers + 1`, one per layer cutoff. Record `l` holds the residual after `l` layers, the
/// cutoff's own final norm, and, when `l > 0`, layer `l - 1`'s sublayer outputs: its attention output (routed
/// layers), its MLP output (dense layers), router logits and picks, per-pick expert outputs and shared output.
pub fn run(
    io: std.Io,
    gpa: std.mem.Allocator,
    m: *fl.Model,
    s: *fl.state.Scratch,
    cache: *fl.state.Cache,
    ids: []const u32,
    rows: []const u32,
    dir: []const u8,
) !void {
    const c = &m.config;
    const h: usize = c.hidden;
    const experts: usize = c.experts;
    const topk: usize = c.topk;
    const vocab: usize = c.vocab;
    const total: u32 = c.layers;
    const sorted = try gpa.dupe(u32, rows);
    defer gpa.free(sorted);
    std.mem.sort(u32, sorted, {}, std.sort.asc(u32));
    const width: usize = fl.state.window_rows;
    std.Io.Dir.cwd().createDir(io, dir, .default_dir) catch |e| switch (e) {
        error.PathAlreadyExists => {},
        else => return e,
    };
    var rec: std.ArrayList(u8) = .empty;
    defer rec.deinit(gpa);
    var path: [1024]u8 = undefined;
    for (sorted) |row| {
        if (row >= ids.len) return error.DumpRowOutOfRange;
        const start = row - row % width;
        while (cache.len < start) {
            const n = @min(width, start - cache.len);
            try fl.forward.run(m, s, cache, ids[cache.len..][0..n], .{});
            cache.len += n;
        }
        const end = @min(ids.len, start + width);
        const n = end - start;
        const r = row - start;
        rec.clearRetainingCapacity();
        for (0..total + 1) |l| {
            c.layers = @intCast(l);
            try fl.forward.run(m, s, cache, ids[start..end], .{ .all = true });
            try rec.appendSlice(gpa, std.mem.sliceAsBytes(s.resid.slice(f32, n * h)[r * h ..][0..h]));
            try rec.appendSlice(gpa, std.mem.sliceAsBytes(s.x.slice(u16, n * h)[r * h ..][0..h]));
            if (l > 0) {
                try rec.appendSlice(gpa, std.mem.sliceAsBytes(s.y.slice(u16, n * h)[r * h ..][0..h]));
                try rec.appendSlice(gpa, std.mem.sliceAsBytes(s.shared.slice(u16, n * h)[r * h ..][0..h]));
                try rec.appendSlice(gpa, std.mem.sliceAsBytes(s.logits32.slice(f32, n * experts)[r * experts ..][0..experts]));
                try rec.appendSlice(gpa, std.mem.sliceAsBytes(s.pick.slice(i32, n * topk)[r * topk ..][0..topk]));
                try rec.appendSlice(gpa, std.mem.sliceAsBytes(s.wts.slice(f32, n * topk)[r * topk ..][0..topk]));
                try rec.appendSlice(gpa, std.mem.sliceAsBytes(s.ey.slice(u16, n * topk * h)[r * topk * h ..][0 .. topk * h]));
            }
        }
        c.layers = total;
        const layers_path = try std.fmt.bufPrint(&path, "{s}/{d}.layers.bin", .{ dir, row });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = layers_path, .data = rec.items });
        const final_path = try std.fmt.bufPrint(&path, "{s}/{d}.final.bin", .{ dir, row });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = final_path, .data = std.mem.sliceAsBytes(s.logits_all.slice(u16, n * vocab)[r * vocab ..][0..vocab]) });
        std.debug.print("flint dump: row {d} at {d}, {d} records\n", .{ row, start, total + 1 });
    }
}
