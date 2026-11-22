const std = @import("std");
const mtl = @import("metal");
const shapes = @import("fp4_shapes.zig");
const probe = @import("split_probe.zig");

pub fn main() !void {
    const pool = mtl.objc.Pool.push(); defer pool.pop();
    const a = std.heap.c_allocator;
    for ([_]shapes.Format{ .nvfp4, .mxfp8 }) |fmt| for ([_]u32{ 1152, 2304 }) |k| for ([_]u32{ 8, 32, 64, 128 }) |rows| {
        var p = try probe.Fixture.init(k, rows, fmt, true); defer p.deinit();
        std.debug.print("WEIGHTS fmt={s} k={d} rows={d} half_inexact={d}\n", .{ fmt.name(), k, rows, p.half_inexact });
        p.fill(false);
        var samples: [probe.names.len][20]f64 = undefined;
        for (0..23) |iteration| for (0..probe.names.len) |i| {
            const arm = (i + iteration) % probe.names.len;
            if (!p.supported(arm)) continue;
            const us = try p.run(arm);
            if (iteration >= 3) samples[arm][iteration - 3] = us;
        };
        for (probe.names, 0..) |name, arm| {
            if (!p.supported(arm)) continue;
            std.mem.sort(f64, &samples[arm], {}, std.sort.asc(f64));
            std.debug.print("TIMING fmt={s} k={d} rows={d} arm={s} median_us={d:.3} min_us={d:.3} max_us={d:.3}\n", .{ fmt.name(), k, rows, name, (samples[arm][9] + samples[arm][10]) / 2, samples[arm][0], samples[arm][19] });
        }
        const count = @as(usize, p.n) * rows;
        const outputs = try a.alloc(f32, count * probe.names.len); defer a.free(outputs);
        for ([_]bool{ true, false }) |exact| {
            p.fill(exact);
            for (0..probe.names.len) |arm| {
                if (!p.supported(arm)) continue;
                _ = try p.run(arm);
                @memcpy(outputs[arm * count ..][0..count], p.y.slice(f32, count));
            }
            var stats: [probe.names.len]probe.ErrorStats = @splat(.{});
            try probe.evaluate(&p, outputs, &stats);
            for (probe.names, stats, 0..) |name, s, arm| {
                if (!p.supported(arm)) continue;
                std.debug.print("ERROR fmt={s} k={d} rows={d} inputs={s} arm={s} abs={e:.6} rel={e:.6} bound_ratio={e:.6} failed={d}\n", .{ fmt.name(), k, rows, if (exact) "bf16_exact" else "random_f32", name, s.abs, s.rel, s.ratio, s.failed });
            }
        }
    };
}
