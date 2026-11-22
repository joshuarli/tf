//! tf-fp4-bench: median GPU time per fp4 decode variant and Inkling 2 Mini shape, its weight
//! bandwidth, and the share of this host's measured copy bandwidth that represents.
const std = @import("std");
const mtl = @import("metal");
const shapes = @import("fp4_shapes.zig");
const synth = @import("fp4_synth.zig");
const gpu = @import("fp4_gpu.zig");

const warm = 5;
const runs = 50;

pub fn main() !void {
    if (std.c.getenv("TF_MXFP8_TENSOROPS") != null) return @import("mxfp8_bench.zig").main();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try gpu.Pipes.init(device);
    defer pipes.deinit();

    std.debug.print("tf-fp4-bench: {s}, max working set {d} B, warmup {d}, timed {d}\n", .{ device.name(), device.maxWorkingSet(), warm, runs });

    // Two stream sizes: 128 MiB can sit in the SLC, 512 MiB cannot, so the larger one is the ceiling.
    var copy_gbs: f64 = 0;
    var read_gbs: f64 = 0;
    for ([_]usize{ 128, 512 }) |mib| {
        const bytes = mib * 1024 * 1024;
        const src = try device.buffer(bytes, gpu.opts);
        defer src.deinit();
        const dst = try device.buffer(bytes, gpu.opts);
        defer dst.deinit();
        for (src.slice(u8, bytes), 0..) |*b, i| b.* = @truncate(i *% 2654435761);
        for (dst.slice(u8, bytes)) |*b| b.* = 0;
        const copy_us = try gpu.measureCopy(queue, &pipes, src, dst, bytes, 5, runs);
        const moved = 2.0 * @as(f64, @floatFromInt(bytes)) / (copy_us * 1e-6) / 1e9;
        const words = try device.buffer(bytes / 4, gpu.opts);
        defer words.deinit();
        const read_us = try gpu.measureRead(queue, &pipes, src, words, bytes, 5, runs);
        const read = @as(f64, @floatFromInt(bytes)) / (read_us * 1e-6) / 1e9;
        std.debug.print("streams {d:>4} MiB: copy {d:.2} us -> {d:.1} GB/s moved, read {d:.2} us -> {d:.1} GB/s read\n", .{ mib, copy_us, moved, read_us, read });
        copy_gbs = moved;
        read_gbs = @max(read_gbs, read);
    }
    // The ceiling is the best read stream either size reached. A matvec whose weights fit in the SLC
    // can beat it, so a fraction above 1 is cache residency, not an error.
    std.debug.print("ceiling: {d:.1} GB/s read (best stream), {d:.1} GB/s moved by the copy (fraction denominators)\n\n", .{ read_gbs, copy_gbs });

    std.debug.print("format  rows proj     variant      median_us   weight_GB/s  of_read_ceiling  of_copy_GBs\n", .{});
    for ([_]shapes.Format{ .nvfp4, .mxfp8 }) |fmt| {
        for (shapes.decode_rows) |token_rows| {
            var w = try gpu.Weights.alloc(device, fmt, token_rows);
            defer w.deinit();
            const gu = shapes.geometry(fmt, .gu);
            const dn = shapes.geometry(fmt, .down);
            const a = std.heap.c_allocator;
            const max_codes = @max(gu.code_bytes, dn.code_bytes);
            const max_scales = @max(gu.scale_bytes, dn.scale_bytes);
            const codes = try a.alloc(u8, max_codes);
            defer a.free(codes);
            const scales = try a.alloc(u8, max_scales);
            defer a.free(scales);
            for (0..w.pairs) |pair| {
                inline for ([_]shapes.Projection{ .gu, .down }) |proj| {
                    const g = shapes.geometry(fmt, proj);
                    var prng = std.Random.DefaultPrng.init(synth.seedFor(fmt, proj, pair));
                    const global = synth.fillPair(prng.random(), fmt, codes[0..g.code_bytes], scales[0..g.scale_bytes]);
                    w.upload(proj, pair, codes[0..g.code_bytes], scales[0..g.scale_bytes], global);
                }
            }
            inline for ([_]shapes.Projection{ .gu, .down }) |proj| {
                const g = shapes.geometry(fmt, proj);
                const bytes: f64 = @floatFromInt(w.pairs * (g.code_bytes + g.scale_bytes));
                var best_us = std.math.floatMax(f64);
                var best: gpu.Variant = .baseline;
                for (gpu.all_variants) |variant| {
                    const us = try gpu.measure(queue, &w, &pipes, variant, proj, warm, runs);
                    const gbs = bytes / (us * 1e-6) / 1e9;
                    std.debug.print("{s:>6} {d:>4} {s:>4} {s:>12} {d:>10.2} {d:>13.1} {d:>15.3} {d:>12.3}\n", .{
                        fmt.name(), token_rows, proj.name(), gpu.variant_names[@intFromEnum(variant)], us, gbs, gbs / read_gbs, gbs / copy_gbs,
                    });
                    if (us < best_us) {
                        best_us = us;
                        best = variant;
                    }
                }
                std.debug.print("winner {s} rows {d} {s}: {s} {d:.2} us over {d} pairs ({d:.1} MB of weights)\n\n", .{
                    fmt.name(), token_rows, proj.name(), gpu.variant_names[@intFromEnum(best)], best_us, w.pairs, bytes / 1e6,
                });
            }
        }
    }
}
