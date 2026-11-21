//! The fp4 decode kernels on one Metal device: pipeline set, synthetic expert slabs, dispatches for
//! every variant and projection, and GPU-timestamped measurement.
const std = @import("std");
const mtl = @import("metal");
const shapes = @import("fp4_shapes.zig");
const synth = @import("fp4_synth.zig");
const ref = @import("fp4_ref.zig");

pub const opts = mtl.ResourceOptions.shared;

pub const Variant = enum(u8) { baseline, rows, shared, table, thread };
pub const all_variants = [_]Variant{ .baseline, .rows, .shared, .table, .thread };
pub const variant_names = [_][]const u8{ "v0-baseline", "v1-rows", "v2-shared", "v3-table", "v4-thread" };

const baseline_source = @embedFile("baseline_experts.metal");
const baseline_mxfp8_source = "#define I2_MXFP8 1\n" ++ baseline_source;
const variant_source = @embedFile("fp4_matvec.metal");
const variant_mxfp8_source = "#define FP4_MXFP8 1\n" ++ variant_source;

pub const gate_cap: f32 = 4.0;
pub const up_cap: f32 = 25.0;
pub const den_value: f32 = 2688.0;

pub const Args = extern struct { k: u32, n: u32, rows: u32, topk: u32, gate_cap: f32, up_cap: f32 };
const Moe = extern struct { k: u32, n: u32, rows: u32, gate_cap: f32, up_cap: f32 };

const ProjPipes = struct { gu: mtl.Pipeline, down: mtl.Pipeline };

pub const Pipes = struct {
    libs: [4]mtl.Library,
    base: [2]ProjPipes,
    rows: [2]ProjPipes,
    shared: [2]ProjPipes,
    table: [2]ProjPipes,
    thread: [2]ProjPipes,
    copy: mtl.Pipeline,
    read: mtl.Pipeline,
    t16: mtl.Buffer,
    t256: mtl.Buffer,

    fn proj(device: mtl.Device, lib: mtl.Library) !ProjPipes {
        return .{
            .gu = try mtl.Pipeline.init(device, lib, "i2_gu", false),
            .down = try mtl.Pipeline.init(device, lib, "i2_down", false),
        };
    }

    fn named(device: mtl.Device, lib: mtl.Library, name: []const u8) !ProjPipes {
        var gu_buf: [64]u8 = undefined;
        var down_buf: [64]u8 = undefined;
        return .{
            .gu = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&gu_buf, "fp4_gu_{s}", .{name}), false),
            .down = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&down_buf, "fp4_down_{s}", .{name}), false),
        };
    }

    pub fn init(device: mtl.Device) !Pipes {
        var p: Pipes = undefined;
        p.libs[0] = try mtl.Library.fromSource(device, baseline_source, mtl.CompileOptions.mlx());
        p.libs[1] = try mtl.Library.fromSource(device, baseline_mxfp8_source, mtl.CompileOptions.mlx());
        p.libs[2] = try mtl.Library.fromSource(device, variant_source, mtl.CompileOptions.mlx());
        p.libs[3] = try mtl.Library.fromSource(device, variant_mxfp8_source, mtl.CompileOptions.mlx());
        p.base[0] = try proj(device, p.libs[0]);
        p.base[1] = try proj(device, p.libs[1]);
        p.rows[0] = try named(device, p.libs[2], "rows");
        p.rows[1] = try named(device, p.libs[3], "rows");
        p.shared[0] = try named(device, p.libs[2], "shared");
        p.shared[1] = try named(device, p.libs[3], "shared");
        p.table[0] = try named(device, p.libs[2], "table");
        p.table[1] = try named(device, p.libs[3], "table");
        p.thread[0] = try named(device, p.libs[2], "thread");
        p.thread[1] = try named(device, p.libs[3], "thread");
        p.copy = try mtl.Pipeline.init(device, p.libs[2], "fp4_copy", false);
        p.read = try mtl.Pipeline.init(device, p.libs[2], "fp4_read", false);
        p.t16 = try device.buffer(16 * 4, opts);
        p.t256 = try device.buffer(256 * 4, opts);
        for (ref.nvfp4_codes, 0..) |v, i| p.t16.slice(f32, 16)[i] = @floatCast(v);
        for (0..256) |i| p.t256.slice(f32, 256)[i] = @floatCast(ref.e4m3(@intCast(i)));
        return p;
    }

    pub fn deinit(self: *Pipes) void {
        inline for (0..2) |f| {
            self.base[f].gu.deinit();
            self.base[f].down.deinit();
            self.rows[f].gu.deinit();
            self.rows[f].down.deinit();
            self.shared[f].gu.deinit();
            self.shared[f].down.deinit();
            self.table[f].gu.deinit();
            self.table[f].down.deinit();
            self.thread[f].gu.deinit();
            self.thread[f].down.deinit();
        }
        self.copy.deinit();
        self.read.deinit();
        self.t16.deinit();
        self.t256.deinit();
        for (self.libs) |l| l.deinit();
    }

    fn forVariant(self: *const Pipes, variant: Variant, fmt: shapes.Format) *const ProjPipes {
        const f = @intFromEnum(fmt);
        return switch (variant) {
            .baseline => &self.base[f],
            .rows => &self.rows[f],
            .shared => &self.shared[f],
            .table => &self.table[f],
            .thread => &self.thread[f],
        };
    }
};

pub const Weights = struct {
    fmt: shapes.Format,
    token_rows: usize,
    pairs: usize,
    w13: mtl.Buffer,
    s13: mtl.Buffer,
    w2: mtl.Buffer,
    s2: mtl.Buffer,
    amax13: mtl.Buffer,
    amax2: mtl.Buffer,
    den: mtl.Buffer,
    off: mtl.Buffer,
    order: mtl.Buffer,
    x: mtl.Buffer,
    act: mtl.Buffer,
    out: mtl.Buffer,

    pub fn alloc(device: mtl.Device, fmt: shapes.Format, token_rows: usize) !Weights {
        const pairs = token_rows * shapes.topk;
        const gu = shapes.geometry(fmt, .gu);
        const dn = shapes.geometry(fmt, .down);
        var w = Weights{
            .fmt = fmt,
            .token_rows = token_rows,
            .pairs = pairs,
            .w13 = try device.buffer(pairs * gu.code_bytes, opts),
            .s13 = try device.buffer(pairs * gu.scale_bytes, opts),
            .w2 = try device.buffer(pairs * dn.code_bytes, opts),
            .s2 = try device.buffer(pairs * dn.scale_bytes, opts),
            .amax13 = try device.buffer(shapes.expert_count * 4, opts),
            .amax2 = try device.buffer(shapes.expert_count * 4, opts),
            .den = try device.buffer(4, opts),
            .off = try device.buffer((pairs + 1) * 4, opts),
            .order = try device.buffer(pairs * 4, opts),
            .x = try device.buffer(token_rows * shapes.hidden * 4, opts),
            .act = try device.buffer(pairs * shapes.intermediate * 4, opts),
            .out = try device.buffer(pairs * shapes.hidden * 4, opts),
        };
        for (w.off.slice(u32, pairs + 1), 0..) |*v, i| v.* = @intCast(i);
        for (w.order.slice(u32, pairs), 0..) |*v, i| v.* = @intCast(i);
        for (w.amax13.slice(f32, shapes.expert_count)) |*v| v.* = 0;
        for (w.amax2.slice(f32, shapes.expert_count)) |*v| v.* = 0;
        w.den.slice(f32, 1)[0] = den_value;
        var prng = std.Random.DefaultPrng.init(0x5359_4e54);
        synth.fillData(prng.random(), w.x.slice(f32, token_rows * shapes.hidden));
        synth.fillData(prng.random(), w.act.slice(f32, pairs * shapes.intermediate));
        return w;
    }

    pub fn deinit(self: *Weights) void {
        for ([_]mtl.Buffer{ self.w13, self.s13, self.w2, self.s2, self.amax13, self.amax2, self.den, self.off, self.order, self.x, self.act, self.out }) |b| b.deinit();
    }

    /// Copies one expert's bytes into its pair slot; a gate/up slab also carries the pair's global.
    pub fn upload(self: *Weights, proj: shapes.Projection, pair: usize, codes: []const u8, scales: []const u8, global: f64) void {
        const g = shapes.geometry(self.fmt, proj);
        if (g.code_bytes != codes.len or g.scale_bytes != scales.len) @panic("fp4 pair size");
        const at = pair * g.code_bytes;
        const sat = pair * g.scale_bytes;
        if (proj == .gu) {
            @memcpy(self.w13.slice(u8, self.pairs * g.code_bytes)[at..][0..codes.len], codes);
            @memcpy(self.s13.slice(u8, self.pairs * g.scale_bytes)[sat..][0..scales.len], scales);
            self.amax13.slice(f32, shapes.expert_count)[pair] = @floatCast(global * den_value);
        } else {
            @memcpy(self.w2.slice(u8, self.pairs * g.code_bytes)[at..][0..codes.len], codes);
            @memcpy(self.s2.slice(u8, self.pairs * g.scale_bytes)[sat..][0..scales.len], scales);
            self.amax2.slice(f32, shapes.expert_count)[pair] = @floatCast(global * den_value);
        }
    }

    pub fn uploadAct(self: *Weights, values: []const f32) void {
        @memcpy(self.act.slice(f32, values.len), values);
    }

    pub fn actSlice(self: *Weights) []f32 {
        return self.act.slice(f32, self.pairs * shapes.intermediate);
    }

    pub fn outSlice(self: *Weights) []f32 {
        return self.out.slice(f32, self.pairs * shapes.hidden);
    }
};

fn argsOf(g: shapes.Geometry) Args {
    return .{ .k = @intCast(g.k), .n = @intCast(g.n), .rows = @intCast(g.rows), .topk = shapes.topk, .gate_cap = gate_cap, .up_cap = up_cap };
}

fn moeOf(g: shapes.Geometry) Moe {
    return .{ .k = @intCast(g.k), .n = @intCast(g.n), .rows = @intCast(g.rows), .gate_cap = gate_cap, .up_cap = up_cap };
}

fn encodeBaseline(w: *const Weights, p: *const Pipes, proj: shapes.Projection, e: mtl.ComputeEncoder) void {
    const g = shapes.geometry(w.fmt, proj);
    const pipes = p.forVariant(.baseline, w.fmt);
    const weight = if (proj == .gu) w.w13 else w.w2;
    const scales = if (proj == .gu) w.s13 else w.s2;
    const amax = if (proj == .gu) w.amax13 else w.amax2;
    const input = if (proj == .gu) w.x else w.act;
    const output = if (proj == .gu) w.act else w.out;
    e.setPipeline(if (proj == .gu) pipes.gu else pipes.down);
    e.setValue(moeOf(g), 7);
    e.setBuffer(w.den, 0, 4);
    for (0..w.token_rows) |t| {
        // The kernel reads its expert as tg.y, so the weight, scale and global windows move by this
        // token's base; OFF shifted by the same base makes base+member an absolute pick index, which
        // is what ORDER names and how the output slot is placed. Only the input row also moves.
        e.setBuffer(input, if (proj == .gu) t * g.k * 4 else 0, 0);
        e.setBuffer(weight, t * shapes.topk * g.code_bytes, 1);
        e.setBuffer(scales, t * shapes.topk * g.scale_bytes, 2);
        e.setBuffer(amax, t * shapes.topk * 4, 3);
        e.setBuffer(w.off, t * shapes.topk * 4, 5);
        e.setBuffer(w.order, 0, 6);
        e.setBuffer(output, 0, 8);
        e.dispatchGroups(mtl.Size.of(g.n / 32, shapes.topk, 1), mtl.Size.of(256, 1, 1));
    }
}

pub fn encode(w: *const Weights, p: *const Pipes, variant: Variant, proj: shapes.Projection, cb: mtl.CommandBuffer) void {
    const e = cb.compute(.serial);
    defer e.end();
    if (variant == .baseline) return encodeBaseline(w, p, proj, e);
    const g = shapes.geometry(w.fmt, proj);
    const pipes = p.forVariant(variant, w.fmt);
    e.setPipeline(if (proj == .gu) pipes.gu else pipes.down);
    e.setBuffer(if (proj == .gu) w.x else w.act, 0, 0);
    e.setBuffer(if (proj == .gu) w.w13 else w.w2, 0, 1);
    e.setBuffer(if (proj == .gu) w.s13 else w.s2, 0, 2);
    e.setBuffer(if (proj == .gu) w.amax13 else w.amax2, 0, 3);
    e.setBuffer(w.den, 0, 4);
    e.setValue(argsOf(g), 5);
    e.setBuffer(if (proj == .gu) w.act else w.out, 0, 6);
    if (variant == .table) {
        e.setBuffer(p.t16, 0, 7);
        e.setBuffer(p.t256, 0, 8);
    }
    switch (variant) {
        .rows => e.dispatchGroups(mtl.Size.of(g.n, w.pairs, 1), mtl.Size.of(32, 1, 1)),
        .shared => e.dispatchGroups(mtl.Size.of((g.n + 3) / 4, w.pairs, 1), mtl.Size.of(128, 1, 1)),
        .table => e.dispatchGroups(mtl.Size.of(g.n, w.pairs, 1), mtl.Size.of(32, 1, 1)),
        .thread => e.dispatchGroups(mtl.Size.of((g.n + 63) / 64, w.pairs, 1), mtl.Size.of(64, 1, 1)),
        .baseline => unreachable,
    }
}

pub fn run(queue: mtl.Queue, w: *const Weights, p: *const Pipes, variant: Variant, proj: shapes.Projection) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    encode(w, p, variant, proj, cb);
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed ({s} {s}): {s}\n", .{ variant_names[@intFromEnum(variant)], proj.name(), msg });
        return error.GpuFailure;
    }
}

pub const max_runs = 64;

fn median(samples: []f64) f64 {
    std.mem.sort(f64, samples, {}, std.sort.asc(f64));
    const n = samples.len;
    return if (n % 2 == 1) samples[n / 2] else 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

/// Median GPU microseconds over `runs` command buffers after `warm` untimed ones.
pub fn measure(
    queue: mtl.Queue,
    w: *const Weights,
    p: *const Pipes,
    variant: Variant,
    proj: shapes.Projection,
    warm: usize,
    runs: usize,
) !f64 {
    if (runs > max_runs) return error.TooManyRuns;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var samples: [max_runs]f64 = undefined;
    var i: usize = 0;
    while (i < warm + runs) : (i += 1) {
        const cb = queue.commandBuffer();
        encode(w, p, variant, proj, cb);
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
        if (i >= warm) samples[i - warm] = cb.gpuSeconds() * 1e6;
    }
    return median(samples[0..runs]);
}

pub fn measureRead(queue: mtl.Queue, p: *const Pipes, src: mtl.Buffer, dst: mtl.Buffer, bytes: usize, warm: usize, runs: usize) !f64 {
    const count: u32 = @intCast(bytes / 16);
    const threads = (count + 3) / 4;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var samples: [max_runs]f64 = undefined;
    var i: usize = 0;
    while (i < warm + runs) : (i += 1) {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(p.read);
        e.setBuffer(src, 0, 0);
        e.setBuffer(dst, 0, 1);
        e.setValue(count, 2);
        e.dispatchThreads(mtl.Size.of(threads, 1, 1), mtl.Size.of(256, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
        if (i >= warm) samples[i - warm] = cb.gpuSeconds() * 1e6;
    }
    return median(samples[0..runs]);
}

pub fn measureCopy(queue: mtl.Queue, p: *const Pipes, src: mtl.Buffer, dst: mtl.Buffer, bytes: usize, warm: usize, runs: usize) !f64 {
    const count: u32 = @intCast(bytes / 16);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var samples: [max_runs]f64 = undefined;
    var i: usize = 0;
    while (i < warm + runs) : (i += 1) {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(p.copy);
        e.setBuffer(src, 0, 0);
        e.setBuffer(dst, 0, 1);
        e.setValue(count, 2);
        e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
        if (i >= warm) samples[i - warm] = cb.gpuSeconds() * 1e6;
    }
    return median(samples[0..runs]);
}
