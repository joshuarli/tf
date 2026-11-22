//! Every fp4 decode variant against the fp64 dequant reference on Inkling 2 Mini's decode shapes,
//! under the fp32 accumulation bound, with a corrupted scale as the negative control.
const std = @import("std");

/// The zig build runner fails a test step that writes to stderr, so the per-shape evidence prints only on request.
fn verbose() bool {
    const v = std.c.getenv("TF_FP4_VERBOSE") orelse return false;
    return std.mem.span(v).len > 0;
}
const mtl = @import("metal");
const shapes = @import("fp4_shapes.zig");
const synth = @import("fp4_synth.zig");
const ref = @import("fp4_ref.zig");
const gpu = @import("fp4_gpu.zig");

const caps = ref.Caps{};

const Check = struct {
    values: usize = 0,
    worst_abs: f64 = 0,
    worst_ratio: f64 = 0,
    failed: usize = 0,
    nonfinite: usize = 0,
    first: usize = 0,
    first_got: f64 = 0,
    first_want: f64 = 0,

    fn add(self: *Check, got: f32, want: f64, bound: f64) void {
        self.values += 1;
        const v: f64 = @floatCast(got);
        if (!std.math.isFinite(v)) {
            self.nonfinite += 1;
            self.failed += 1;
            return;
        }
        const err = @abs(v - want);
        self.worst_abs = @max(self.worst_abs, err);
        const ratio = err / (bound + 1e-30);
        self.worst_ratio = @max(self.worst_ratio, ratio);
        if (err > bound) {
            if (self.failed == 0) {
                self.first = self.values - 1;
                self.first_got = v;
                self.first_want = want;
            }
            self.failed += 1;
        }
    }

    fn report(self: Check, fmt: shapes.Format, rows: usize, proj: shapes.Projection, variant: gpu.Variant) void {
        if (self.failed == 0 and !verbose()) return;
        std.debug.print("  {s} rows {d:2} {s:>4} {s:>12}: {d} values, worst |err| {e:.3}, worst err/bound {d:.4}, failed {d}", .{
            fmt.name(), rows, proj.name(), gpu.variant_names[@intFromEnum(variant)], self.values, self.worst_abs, self.worst_ratio, self.failed,
        });
        if (self.failed > 0) std.debug.print(" (first at {d}: got {d} want {d})", .{ self.first, self.first_got, self.first_want });
        std.debug.print("\n", .{});
    }
};

fn expectWithin(check: Check) !void {
    try std.testing.expectEqual(@as(usize, 0), check.failed);
    try std.testing.expect(check.worst_ratio < 1.0);
}

test "fp4 decode variants match the fp64 reference on Inkling decode shapes" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try gpu.Pipes.init(device);
    defer pipes.deinit();

    const a = std.testing.allocator;
    if (verbose()) std.debug.print("device {s}, {d} bytes working set\n", .{ device.name(), device.maxWorkingSet() });

    for ([_]shapes.Format{ .nvfp4, .mxfp8 }) |fmt| {
        for (shapes.decode_rows) |token_rows| {
            var w = try gpu.Weights.alloc(device, fmt, token_rows);
            defer w.deinit();
            const pairs = w.pairs;
            const gu = shapes.geometry(fmt, .gu);
            const dn = shapes.geometry(fmt, .down);

            var codes = try a.alloc(u8, @max(gu.code_bytes, dn.code_bytes));
            defer a.free(codes);
            var scales = try a.alloc(u8, @max(gu.scale_bytes, dn.scale_bytes));
            defer a.free(scales);

            const act_ref = try a.alloc(f64, pairs * gu.n);
            defer a.free(act_ref);
            const act_bound = try a.alloc(f64, pairs * gu.n);
            defer a.free(act_bound);
            const out_ref = try a.alloc(f64, pairs * dn.n);
            defer a.free(out_ref);
            const out_bound = try a.alloc(f64, pairs * dn.n);
            defer a.free(out_bound);
            const act_f32 = try a.alloc(f32, pairs * shapes.intermediate);
            defer a.free(act_f32);

            const x = w.x.slice(f32, token_rows * shapes.hidden);
            for (0..pairs) |pair| {
                var prng = std.Random.DefaultPrng.init(synth.seedFor(fmt, .gu, pair));
                const global = synth.fillPair(prng.random(), fmt, codes[0..gu.code_bytes], scales[0..gu.scale_bytes]);
                w.upload(.gu, pair, codes[0..gu.code_bytes], scales[0..gu.scale_bytes], global);
                const net = ref.Net{ .fmt = fmt, .k = gu.k, .n = gu.n, .rows = gu.rows, .codes = codes[0..gu.code_bytes], .scales = scales[0..gu.scale_bytes], .global = global };
                const t = pair / shapes.topk;
                const slice = pair * gu.n;
                ref.gu(net, x[t * gu.k ..][0..gu.k], .{
                    .act = act_ref[slice..][0..gu.n],
                    .bound = act_bound[slice..][0..gu.n],
                }, caps);
            }

            for (gpu.all_variants) |variant| {
                try gpu.run(queue, &w, &pipes, variant, .gu);
                var check = Check{};
                const got = w.actSlice();
                for (0..pairs * gu.n) |i| check.add(got[i], act_ref[i], act_bound[i]);
                check.report(fmt, token_rows, .gu, variant);
                try expectWithin(check);
            }

            for (0..pairs * gu.n) |i| act_f32[i] = @floatCast(act_ref[i]);
            w.uploadAct(act_f32);

            for (0..pairs) |pair| {
                var prng = std.Random.DefaultPrng.init(synth.seedFor(fmt, .down, pair));
                const global = synth.fillPair(prng.random(), fmt, codes[0..dn.code_bytes], scales[0..dn.scale_bytes]);
                w.upload(.down, pair, codes[0..dn.code_bytes], scales[0..dn.scale_bytes], global);
                const net = ref.Net{ .fmt = fmt, .k = dn.k, .n = dn.n, .rows = dn.rows, .codes = codes[0..dn.code_bytes], .scales = scales[0..dn.scale_bytes], .global = global };
                const slice = pair * dn.n;
                ref.down(net, act_f32[pair * dn.k ..][0..dn.k], out_ref[slice..][0..dn.n], out_bound[slice..][0..dn.n]);
            }

            for (gpu.all_variants) |variant| {
                try gpu.run(queue, &w, &pipes, variant, .down);
                var check = Check{};
                const got = w.outSlice();
                for (0..pairs * dn.n) |i| check.add(got[i], out_ref[i], out_bound[i]);
                check.report(fmt, token_rows, .down, variant);
                try expectWithin(check);
            }

            if (fmt == .nvfp4 and token_rows == 1) {
                // Every scale byte of the first pair moved one e4m3 exponent step: a kernel that
                // ignored the scales would still match, so this must fail for every variant.
                const s13 = w.s13.slice(u8, pairs * gu.scale_bytes);
                for (0..gu.scale_bytes) |i| s13[i] ^= 0x40;
                for (gpu.all_variants) |variant| {
                    try gpu.run(queue, &w, &pipes, variant, .gu);
                    var corrupted = Check{};
                    const got = w.actSlice();
                    for (0..gu.n) |i| corrupted.add(got[i], act_ref[i], act_bound[i]);
                    if (verbose()) std.debug.print("  negative control {s}: pair 0 scales x2^-8, {d} of {d} values outside the bound\n", .{
                        gpu.variant_names[@intFromEnum(variant)], corrupted.failed, corrupted.values,
                    });
                    try std.testing.expect(corrupted.failed > gu.n * 3 / 4);
                }
                for (0..gu.scale_bytes) |i| s13[i] ^= 0x40;
            }
        }
    }
}

test {
    _ = @import("mxfp8_probe.zig");
}
