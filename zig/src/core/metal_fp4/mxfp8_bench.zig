const std = @import("std");
const mtl = @import("metal");
const gpu = @import("fp4_gpu.zig");
const ref = @import("fp4_ref.zig");
const synth = @import("fp4_synth.zig");
const probe = @import("mxfp8_probe.zig");

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const a = std.heap.c_allocator;
    const source = try std.mem.replaceOwned(u8, a, "#define FP4_MXFP8 1\n" ++ @embedFile("fp4_matvec.metal"), "ulong(e) * a.rows +", "0u * a.rows +");
    defer a.free(source);
    for ([_]u32{ 1, 8, 64 }) |rows| {
        for ([_]bool{ true, false }) |gu| {
            const k: u32 = if (gu) 2304 else 1152;
            const n: u32 = 2304;
            var p = try probe.Probe.init(k, n, rows);
            defer p.deinit();
            const lib = try mtl.Library.fromSource(p.device, source, mtl.CompileOptions.mlx());
            defer lib.deinit();
            const post = try mtl.Pipeline.init(p.device, p.lib, "mxfp8_situ", false);
            defer post.deinit();
            const xf = try p.device.buffer(@as(usize, k) * rows * 4, gpu.opts);
            defer xf.deinit();
            const out = try p.device.buffer(@as(usize, n) * rows * 4, gpu.opts);
            defer out.deinit();
            const amax = try p.device.buffer(rows * 4, gpu.opts);
            defer amax.deinit();
            const den = try p.device.buffer(4, gpu.opts);
            defer den.deinit();
            const t16 = try p.device.buffer(16 * 4, gpu.opts);
            defer t16.deinit();
            const t256 = try p.device.buffer(256 * 4, gpu.opts);
            defer t256.deinit();
            for (ref.nvfp4_codes, 0..) |v, i| t16.slice(f32, 16)[i] = @floatCast(v);
            for (0..256) |i| t256.slice(f32, 256)[i] = @floatCast(ref.e4m3(@intCast(i)));
            @memset(amax.slice(f32, rows), 1);
            den.slice(f32, 1)[0] = 1;
            var prng = std.Random.DefaultPrng.init(42 + rows + k);
            _ = synth.fillPair(prng.random(), .mxfp8, p.w.slice(u8, @as(usize, k) * n), p.s.slice(u8, @as(usize, k) * n / 32));
            for (p.x.slice(f16, @as(usize, k) * rows), xf.slice(f32, @as(usize, k) * rows)) |*h, *f| {
                h.* = @floatCast(prng.random().float(f32) * 2 - 1);
                f.* = @floatCast(h.*);
            }
            @memset(p.y.slice(f32, @as(usize, n) * rows), 0);
            const table = try mtl.Pipeline.init(p.device, lib, if (gu) "fp4_gu_table" else "fp4_down_table", false);
            defer table.deinit();
            const thread_pipe = try mtl.Pipeline.init(p.device, lib, if (gu) "fp4_gu_thread" else "fp4_down_thread", false);
            defer thread_pipe.deinit();
            var samples: [3][20]f64 = undefined;
            for (0..23) |iter| {
                for (0..3) |j| {
                    const arm = (j + iter) % 3;
                    const tensor = arm == 0;
                    const thread = arm == 2;
                    const cb = p.queue.commandBuffer();
                    const e = cb.compute(.serial);
                    e.setPipeline(if (tensor) p.pipe else if (thread) thread_pipe else table);
                    if (tensor) {
                        e.setBuffer(p.x, 0, 0); e.setBuffer(p.w, 0, 1); e.setBuffer(p.s, 0, 2); e.setBuffer(p.y, 0, 3);
                        e.setValue(k, 4); e.setValue(n, 5); e.setValue(rows, 6);
                        e.dispatchGroups(mtl.Size.of(n / 32, (rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
                    } else {
                        e.setBuffer(xf, 0, 0); e.setBuffer(p.w, 0, 1); e.setBuffer(p.s, 0, 2); e.setBuffer(amax, 0, 3); e.setBuffer(den, 0, 4);
                        e.setValue(gpu.Args{ .k = k, .n = if (gu) n / 2 else n, .rows = n, .topk = 1, .gate_cap = 4, .up_cap = 25 }, 5);
                        e.setBuffer(out, 0, 6); e.setBuffer(t16, 0, 7); e.setBuffer(t256, 0, 8);
                        const cols = if (gu) n / 2 else n;
                        e.dispatchGroups(mtl.Size.of(if (thread) (cols + 63) / 64 else cols, rows, 1), mtl.Size.of(if (thread) 64 else 32, 1, 1));
                    }
                    e.end();
                    if (tensor and gu) {
                        const ep = cb.compute(.serial);
                        ep.setPipeline(post); ep.setBuffer(p.y, 0, 0); ep.setBuffer(out, 0, 1);
                        ep.setValue(rows * n / 2, 2);
                        ep.dispatchThreads(mtl.Size.of(rows * n / 2, 1, 1), mtl.Size.of(256, 1, 1)); ep.end();
                    }
                    cb.commit(); cb.wait();
                    if (cb.failure()) |msg| { std.debug.print("GPU: {s}\n", .{msg}); return error.GpuFailure; }
                    if (iter >= 3) samples[arm][iter - 3] = cb.gpuSeconds() * 1e6;
                }
            }
            const expected = try a.alloc(f64, n);
            defer a.free(expected);
            const bound = try a.alloc(f64, n);
            defer a.free(bound);
            const net = ref.Net{ .fmt = .mxfp8, .k = k, .n = n, .rows = n, .codes = p.w.slice(u8, @as(usize, k) * n), .scales = p.s.slice(u8, @as(usize, k) * n / 32), .global = 1 };
            for ([_]usize{ 0, rows - 1 }) |row| {
                ref.down(net, xf.slice(f32, @as(usize, k) * rows)[row * k ..][0..k], expected, bound);
                for (0..n) |col| {
                    const got: f64 = p.y.slice(f32, @as(usize, n) * rows)[row * n + col];
                    if (!std.math.isFinite(got) or @abs(got - expected[col]) > bound[col]) return error.IncorrectTensorOutput;
                }
            }
            for ([_][]const u8{ "tensor", "table", "thread" }, 0..) |name, arm| {
                std.mem.sort(f64, &samples[arm], {}, std.sort.asc(f64));
                std.debug.print("rows={d} proj={s} variant={s} median_us={d:.3} range_us={d:.3}..{d:.3}\n", .{ rows, if (gu) "gu" else "down", name, (samples[arm][9] + samples[arm][10]) / 2, samples[arm][0], samples[arm][19] });
            }
        }
    }
}
