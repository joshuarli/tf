const std = @import("std");
const mtl = @import("metal");

pub const source = @embedFile("kernels/rows.metal");

pub const max_rows: usize = 8;
pub const wide_output: u32 = 65536;

pub const Dispatch = struct { pipe: mtl.Pipeline, groups: mtl.Size, threads: mtl.Size = mtl.Size.of(256, 1, 1) };

const names = [_][:0]const u8{ "i2t_dense_8_2", "i2t_projections_8_2", "i2t_gate_up", "i2t_down", "i2r_dense_ahead_2" };

/// The batch step's few-row kernels: every row's output has its one-row kernel's bits at every width.
pub const Pipes = struct {
    dense_tiles: mtl.Pipeline,
    projection_tiles: mtl.Pipeline,
    gate_up_tiles: mtl.Pipeline,
    down_tiles: mtl.Pipeline,
    head_pair: mtl.Pipeline,

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
        return .{ .dense_tiles = pipes[0], .projection_tiles = pipes[1], .gate_up_tiles = pipes[2], .down_tiles = pipes[3], .head_pair = pipes[4] };
    }

    pub fn deinit(p: *Pipes) void {
        for ([_]mtl.Pipeline{ p.dense_tiles, p.projection_tiles, p.gate_up_tiles, p.down_tiles, p.head_pair }) |pipe| pipe.deinit();
    }

    /// Affine 8-bit `n`x`k` rows for 2 to 8 rows; one row keeps `i2a_dense`.
    pub fn dense(p: *const Pipes, rows: usize, n: u32, k: u32, bits: u32) ?Dispatch {
        if (bits != 8 or rows < 2 or rows > max_rows or k % 256 != 0) return null;
        if (rows == 2 and n >= wide_output) return .{ .pipe = p.head_pair, .groups = mtl.Size.of((n + 31) / 32, 1, 1) };
        return .{ .pipe = p.dense_tiles, .groups = mtl.Size.of((n + 15) / 16, 1, 1) };
    }

    /// A layer's fused attention projections for 3 to 8 rows; fewer keep `i2a_batch_projections`.
    pub fn fused(p: *const Pipes, rows: usize, m: [8]u32, k: u32) ?Dispatch {
        if (rows < 3 or rows > max_rows or k % 256 != 0) return null;
        var tiles: u32 = 0;
        for (m) |part| tiles += (part + 15) / 16;
        return .{ .pipe = p.projection_tiles, .groups = mtl.Size.of(tiles, 1, 1) };
    }

    /// NVFP4 gate and up for 2 to 8 rows; one row keeps `i2p_gu`.
    pub fn gateUp(p: *const Pipes, rows: usize, inter: u32, k: u32, slots: u32) ?Dispatch {
        if (k % 16 != 0 or slots > 64 or rows < 2 or rows > max_rows) return null;
        return .{ .pipe = p.gate_up_tiles, .groups = mtl.Size.of((2 * inter + 15) / 16, slots, 1), .threads = mtl.Size.of(128, 1, 1) };
    }

    /// NVFP4 down for 1 to 8 rows.
    pub fn downs(p: *const Pipes, rows: usize, n: u32, k: u32, slots: u32) ?Dispatch {
        if (k % 16 != 0 or slots > 64 or rows == 0 or rows > max_rows) return null;
        return .{ .pipe = p.down_tiles, .groups = mtl.Size.of((n + 31) / 32, slots, 1) };
    }
};

test "the few-row kernels compile" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try Pipes.init(device);
    p.deinit();
}
