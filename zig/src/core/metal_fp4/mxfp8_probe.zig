const std = @import("std");
const mtl = @import("metal");
const ref = @import("fp4_ref.zig");
const synth = @import("fp4_synth.zig");

pub const Probe = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    x: mtl.Buffer,
    w: mtl.Buffer,
    s: mtl.Buffer,
    y: mtl.Buffer,
    k: u32,
    n: u32,
    rows: u32,

    pub fn init(k: u32, n: u32, rows: u32) !Probe {
        const device = try mtl.Device.init();
        const queue = try device.queue();
        const lib = try mtl.Library.fromSource(device, @embedFile("mxfp8_tensorops.metal"), mtl.CompileOptions.mlx());
        const pipe = try mtl.Pipeline.init(device, lib, "mxfp8_tensorops", false);
        return .{ .device = device, .queue = queue, .lib = lib, .pipe = pipe,
            .x = try device.buffer(@as(usize, k) * rows * 2, mtl.ResourceOptions.shared),
            .w = try device.buffer(@as(usize, k) * n, mtl.ResourceOptions.shared),
            .s = try device.buffer(@as(usize, k) * n / 32, mtl.ResourceOptions.shared),
            .y = try device.buffer(@as(usize, n) * rows * 4, mtl.ResourceOptions.shared),
            .k = k, .n = n, .rows = rows };
    }
    pub fn deinit(p: *Probe) void {
        for ([_]mtl.Buffer{p.x, p.w, p.s, p.y}) |b| b.deinit();
        p.pipe.deinit(); p.lib.deinit(); p.queue.deinit(); p.device.deinit();
    }
    pub fn run(p: *Probe) !f64 {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = p.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(p.pipe);
        e.setBuffer(p.x, 0, 0); e.setBuffer(p.w, 0, 1); e.setBuffer(p.s, 0, 2); e.setBuffer(p.y, 0, 3);
        e.setValue(p.k, 4); e.setValue(p.n, 5); e.setValue(p.rows, 6);
        e.dispatchGroups(mtl.Size.of((p.n + 31) / 32, (p.rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        e.end(); cb.commit(); cb.wait();
        if (cb.failure()) |msg| { std.debug.print("GPU error: {s}\n", .{msg}); return error.GpuFailure; }
        return cb.gpuSeconds() * 1e6;
    }
};

test "scaled tensor operation matches the fp64 reference and detects corrupt scales" {
    const a = std.testing.allocator;
    for ([_]u32{ 1152, 2304 }) |k| {
        var p = try Probe.init(k, 2304, 8);
        defer p.deinit();
        var prng = std.Random.DefaultPrng.init(0x7431 + k);
        const codes = p.w.slice(u8, @as(usize, k) * p.n);
        const scales = p.s.slice(u8, @as(usize, k) * p.n / 32);
        _ = synth.fillPair(prng.random(), .mxfp8, codes, scales);
        for (scales, 0..) |*s, i| if (i % 17 == 0) { s.* = 128; };
        const x = p.x.slice(f16, @as(usize, k) * p.rows);
        for (x) |*v| v.* = @floatCast(prng.random().float(f32) * 2 - 1);
        const x32 = try a.alloc(f32, k);
        defer a.free(x32);
        const want = try a.alloc(f64, p.n);
        defer a.free(want);
        const bound = try a.alloc(f64, p.n);
        defer a.free(bound);
        const net = ref.Net{ .fmt = .mxfp8, .k = k, .n = p.n, .rows = p.n, .codes = codes, .scales = scales, .global = 1 };
        @memset(p.y.slice(f32, @as(usize, p.n) * p.rows), 0);
        _ = try p.run();
        var worst: f64 = 0;
        for (0..p.rows) |row| {
            for (x32, 0..) |*v, i| v.* = @floatCast(x[row * k + i]);
            ref.down(net, x32, want, bound);
            for (0..p.n) |col| {
                const err = @abs(@as(f64, p.y.slice(f32, @as(usize, p.n) * p.rows)[row * p.n + col]) - want[col]);
                worst = @max(worst, err / bound[col]);
                try std.testing.expect(err <= bound[col]);
            }
        }
        if (k == 2304) {
            const post = try mtl.Pipeline.init(p.device, p.lib, "mxfp8_situ", false);
            defer post.deinit();
            const act = try p.device.buffer(@as(usize, p.n / 2) * p.rows * 4, mtl.ResourceOptions.shared);
            defer act.deinit();
            const cb = p.queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(post); enc.setBuffer(p.y, 0, 0); enc.setBuffer(act, 0, 1);
            enc.setValue(p.n / 2 * p.rows, 2);
            enc.dispatchThreads(mtl.Size.of(p.n / 2 * p.rows, 1, 1), mtl.Size.of(256, 1, 1));
            enc.end(); cb.commit(); cb.wait();
            if (cb.failure() != null) return error.GpuFailure;
            var gu_net = net;
            gu_net.n = p.n / 2;
            for (0..p.rows) |row| {
                for (x32, 0..) |*v, i| v.* = @floatCast(x[row * k + i]);
                ref.gu(gu_net, x32, .{ .act = want[0 .. p.n / 2], .bound = bound[0 .. p.n / 2] }, .{});
                for (0..p.n / 2) |col| {
                    const got: f64 = act.slice(f32, @as(usize, p.n / 2) * p.rows)[row * (p.n / 2) + col];
                    try std.testing.expect(std.math.isFinite(got) and @abs(got - want[col]) <= bound[col]);
                }
            }
        }
        for (x32, 0..) |*v, i| v.* = @floatCast(x[i]);
        ref.down(net, x32, want, bound);
        for (scales) |*s| s.* -= 1;
        @memset(p.y.slice(f32, @as(usize, p.n) * p.rows), 0);
        _ = try p.run();
        var failed: usize = 0;
        for (0..p.n) |col| {
            if (@abs(@as(f64, p.y.slice(f32, p.n)[col]) - want[col]) > bound[col]) failed += 1;
        }
        std.debug.print("tensor k={d}: worst err/bound={d:.5}; corrupted scales {d}/{d} outside bound\n", .{ k, worst, failed, p.n });
        try std.testing.expect(failed > p.n * 3 / 4);
    }
}
