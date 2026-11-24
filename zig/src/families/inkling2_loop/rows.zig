const std = @import("std");
const mtl = @import("metal");

pub const source = @embedFile("kernels/rows.metal") ++ "\n#define I2T_ROUTED 1\n" ++ @embedFile("kernels/experts.metal");

pub const max_rows: usize = 16;
/// The most rows one 8-row tile of batch rows holds; wider steps take two.
const tile_rows: usize = 8;
pub const wide_output: u32 = 65536;

pub const Dispatch = struct { pipe: mtl.Pipeline, groups: mtl.Size, threads: mtl.Size = mtl.Size.of(256, 1, 1) };

const names = [_][:0]const u8{ "i2t_dense_8_2", "i2t_projections_8_2", "i2e_gate_up_f3", "i2e_down_f3", "i2r_dense_ahead_2", "i2t_dense16_8_2", "i2t_projections16_8_2", "i2e_gate_up16_f3", "i2e_down16_f3" };

/// The batch step's few-row kernels: every row's output has its one-row kernel's bits at every width.
pub const Pipes = struct {
    dense_tiles: mtl.Pipeline,
    projection_tiles: mtl.Pipeline,
    gate_up_tiles: mtl.Pipeline,
    down_tiles: mtl.Pipeline,
    head_pair: mtl.Pipeline,
    dense_wide: mtl.Pipeline,
    projection_wide: mtl.Pipeline,
    gate_up_wide: mtl.Pipeline,
    down_wide: mtl.Pipeline,
    /// Whether this device's matrix products are the fma chains the tiles rely on; if not, one-row kernels serve.
    tiles: bool,

    pub fn init(device: mtl.Device) !Pipes {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [names.len]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for (names, 0..) |name, i| {
            pipes[i] = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        const probe = try mtl.Pipeline.init(device, lib, "i2t_probe", false);
        defer probe.deinit();
        const tiles = try chains(device, probe);
        return .{ .dense_tiles = pipes[0], .projection_tiles = pipes[1], .gate_up_tiles = pipes[2], .down_tiles = pipes[3], .head_pair = pipes[4], .dense_wide = pipes[5], .projection_wide = pipes[6], .gate_up_wide = pipes[7], .down_wide = pipes[8], .tiles = tiles };
    }

    pub fn deinit(p: *Pipes) void {
        for ([_]mtl.Pipeline{ p.dense_tiles, p.projection_tiles, p.gate_up_tiles, p.down_tiles, p.head_pair, p.dense_wide, p.projection_wide, p.gate_up_wide, p.down_wide }) |pipe| pipe.deinit();
    }

    /// Affine 8-bit `n`x`k` rows for 2 to 16 rows; one row keeps `i2a_dense`.
    pub fn dense(p: *const Pipes, rows: usize, n: u32, k: u32, bits: u32) ?Dispatch {
        if (bits != 8 or rows < 2 or rows > max_rows or k % 256 != 0) return null;
        if (rows == 2 and n >= wide_output) return .{ .pipe = p.head_pair, .groups = mtl.Size.of((n + 31) / 32, 1, 1) };
        if (!p.tiles) return null;
        return .{ .pipe = if (rows > tile_rows) p.dense_wide else p.dense_tiles, .groups = mtl.Size.of((n + 15) / 16, 1, 1) };
    }

    /// A layer's fused attention projections for 3 to 16 rows; fewer keep `i2a_batch_projections`.
    pub fn fused(p: *const Pipes, rows: usize, m: [8]u32, k: u32) ?Dispatch {
        if (!p.tiles or rows < 3 or rows > max_rows or k % 256 != 0) return null;
        var tiles: u32 = 0;
        for (m) |part| tiles += (part + 15) / 16;
        return .{ .pipe = if (rows > tile_rows) p.projection_wide else p.projection_tiles, .groups = mtl.Size.of(tiles, 1, 1) };
    }

    /// NVFP4 gate and up for 2 to 16 rows; one row keeps `i2p_gu`.
    pub fn gateUp(p: *const Pipes, rows: usize, inter: u32, k: u32, slots: u32) ?Dispatch {
        if (!p.tiles or k % 16 != 0 or rows < 2 or rows > max_rows or slots > 8 * (if (rows > tile_rows) max_rows else tile_rows)) return null;
        return .{ .pipe = if (rows > tile_rows) p.gate_up_wide else p.gate_up_tiles, .groups = mtl.Size.of((2 * inter + 15) / 16, slots, 1), .threads = mtl.Size.of(128, 1, 1) };
    }

    /// NVFP4 down for 1 to 16 rows.
    pub fn downs(p: *const Pipes, rows: usize, n: u32, k: u32, slots: u32) ?Dispatch {
        if (!p.tiles or k % 16 != 0 or rows == 0 or rows > max_rows or slots > 8 * (if (rows > tile_rows) max_rows else tile_rows)) return null;
        return .{ .pipe = if (rows > tile_rows) p.down_wide else p.down_tiles, .groups = mtl.Size.of((n + 31) / 32, slots, 1) };
    }
};

const probe_tiles = 8;

/// Two chained 8x8x8 products of each probe tile on the device, compared with the same fma chains on the host.
fn chains(device: mtl.Device, probe: mtl.Pipeline) !bool {
    const count = probe_tiles * 2 * 64;
    const a = try device.buffer(count * 4, mtl.ResourceOptions.shared);
    defer a.deinit();
    const b = try device.buffer(count * 4, mtl.ResourceOptions.shared);
    defer b.deinit();
    const d = try device.buffer(probe_tiles * 64 * 4, mtl.ResourceOptions.shared);
    defer d.deinit();
    var prng = std.Random.DefaultPrng.init(0x7469_6c65);
    const rng = prng.random();
    for ([_][]f32{ a.slice(f32, count), b.slice(f32, count) }) |values| for (values) |*v| {
        const scale = std.math.ldexp(@as(f32, 1), rng.intRangeAtMost(i32, -12, 12));
        v.* = (1 + rng.float(f32)) * scale * @as(f32, if (rng.boolean()) 1 else -1);
    };
    const queue = try device.queue();
    defer queue.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(probe);
    enc.setBuffer(a, 0, 0);
    enc.setBuffer(b, 0, 1);
    enc.setBuffer(d, 0, 2);
    enc.dispatchGroups(mtl.Size.of(probe_tiles, 1, 1), mtl.Size.of(32, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    return sameChains(a.slice(f32, count), b.slice(f32, count), d.slice(f32, probe_tiles * 64));
}

fn sameChains(a: []const f32, b: []const f32, d: []const f32) bool {
    for (0..probe_tiles) |tile| for (0..8) |i| for (0..8) |j| {
        var acc: f32 = 0;
        for (0..2) |s| for (0..8) |t| {
            const at = (tile * 2 + s) * 64;
            acc = @mulAdd(f32, a[at + i * 8 + t], b[at + t * 8 + j], acc);
        };
        if (@as(u32, @bitCast(acc)) != @as(u32, @bitCast(d[tile * 64 + i * 8 + j]))) return false;
    };
    return true;
}

test "the few-row kernels compile and this device's matrix products are fma chains" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try Pipes.init(device);
    defer p.deinit();
    try std.testing.expect(p.tiles);
}

test "the chain check refuses a product summed in another order" {
    var a: [probe_tiles * 2 * 64]f32 = @splat(0);
    var b: [probe_tiles * 2 * 64]f32 = @splat(0);
    var d: [probe_tiles * 64]f32 = @splat(0);
    a[0] = 1e8;
    a[1] = 1;
    a[2] = -1e8;
    for (0..3) |t| b[t * 8] = 1;
    try std.testing.expect(sameChains(&a, &b, &d));
    d[0] = 1;
    try std.testing.expect(!sameChains(&a, &b, &d));
}
