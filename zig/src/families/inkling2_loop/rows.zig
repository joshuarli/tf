const std = @import("std");
const mtl = @import("metal");

pub const source = @embedFile("kernels/rows.metal") ++ "\n#define I2T_ROUTED 1\n" ++ @embedFile("kernels/experts.metal");

pub const max_rows: usize = 16;
/// The most rows one 8-row tile of batch rows holds; wider steps take two.
const tile_rows: usize = 8;
pub const wide_output: u32 = 65536;

pub const Dispatch = struct { pipe: mtl.Pipeline, groups: mtl.Size, threads: mtl.Size = mtl.Size.of(256, 1, 1) };

/// A few-row kernel: its simdgroups a threadgroup and 8-row output tiles a simdgroup.
const Kernel = struct { name: [:0]const u8, sg: u32, t: u32 };

/// The served kernels, in `Pipes.pipes` order: up to 8 rows, then 9 to 16; last, down for a one-row step.
const kernels = [_]Kernel{
    .{ .name = "i2t_dense_8_2", .sg = 8, .t = 2 },
    .{ .name = "i2t_dense16_8_1", .sg = 8, .t = 1 },
    .{ .name = "i2t_projections_8_2", .sg = 8, .t = 2 },
    .{ .name = "i2t_projections16_8_1", .sg = 8, .t = 1 },
    .{ .name = "i2e_gate_up", .sg = 4, .t = 2 },
    .{ .name = "i2e_gate_up16", .sg = 4, .t = 2 },
    .{ .name = "i2e_down", .sg = 8, .t = 4 },
    .{ .name = "i2e_down16", .sg = 8, .t = 4 },
    .{ .name = "i2e_aff4_up", .sg = 4, .t = 2 },
    .{ .name = "i2e_aff4_up16", .sg = 4, .t = 2 },
    .{ .name = "i2e_aff4_down", .sg = 8, .t = 4 },
    .{ .name = "i2e_aff4_down16", .sg = 8, .t = 4 },
    .{ .name = "i2e_down1", .sg = 8, .t = 4 },
};
const Kind = enum(u8) { dense, projections, gate_up, down, aff4_up, aff4_down };

/// The batch step's few-row kernels: every row's output has its one-row kernel's bits at every width.
pub const Pipes = struct {
    pub const Nax = struct {
        sort: mtl.Pipeline,
        nvfp4_up: mtl.Pipeline,
        nvfp4_down: mtl.Pipeline,
        aff4_up: mtl.Pipeline,
        aff4_down: mtl.Pipeline,
        dense4: mtl.Pipeline,
        dense8: mtl.Pipeline,
        projections: mtl.Pipeline,
        controls_down_bf16: mtl.Pipeline,
        controls_apply_bf16: mtl.Pipeline,
        attn_split_nax: mtl.Pipeline,
        router_bf16: mtl.Pipeline,
        pick_ks9: mtl.Pipeline,
        meta: mtl.Buffer,
        indirect: mtl.Buffer,
        router_ks: mtl.Buffer,
        xh: mtl.Buffer,

        pub fn init(device: mtl.Device) !Nax {
            const lib = try mtl.Library.fromSource(device, @import("kernels.zig").verify_nax, mtl.CompileOptions.mlx());
            defer lib.deinit();
            const sort = try mtl.Pipeline.init(device, lib, "i2n_moe_sort", false);
            errdefer sort.deinit();
            const nvfp4_up = try mtl.Pipeline.init(device, lib, "i2n_nvfp4_up", false);
            errdefer nvfp4_up.deinit();
            const nvfp4_down = try mtl.Pipeline.init(device, lib, "i2n_nvfp4_down", false);
            errdefer nvfp4_down.deinit();
            const aff4_up = try mtl.Pipeline.init(device, lib, "i2n_aff4_up", false);
            errdefer aff4_up.deinit();
            const aff4_down = try mtl.Pipeline.init(device, lib, "i2n_aff4_down", false);
            errdefer aff4_down.deinit();
            const dense4_pipe = try mtl.Pipeline.init(device, lib, "i2n_dense4", false);
            errdefer dense4_pipe.deinit();
            const dense8_pipe = try mtl.Pipeline.init(device, lib, "i2n_dense8", false);
            errdefer dense8_pipe.deinit();
            const proj_pipe = try mtl.Pipeline.init(device, lib, "i2n_projections", false);
            errdefer proj_pipe.deinit();
            const ctl_down = try mtl.Pipeline.init(device, lib, "i2n_controls_down_bf16", false);
            errdefer ctl_down.deinit();
            const ctl_apply = try mtl.Pipeline.init(device, lib, "i2n_controls_apply_bf16", false);
            errdefer ctl_apply.deinit();
            const attn_split = try mtl.Pipeline.init(device, lib, "i2v_split_nax", false);
            errdefer attn_split.deinit();
            const router_bf16 = try mtl.Pipeline.init(device, lib, "i2n_router_bf16", false);
            errdefer router_bf16.deinit();
            const pick_ks9 = try mtl.Pipeline.init(device, lib, "i2n_pick_ks9", false);
            errdefer pick_ks9.deinit();
            const meta = try device.buffer(640 * 4, mtl.ResourceOptions.private);
            errdefer meta.deinit();
            const indirect = try device.buffer(8 * 4, mtl.ResourceOptions.private);
            errdefer indirect.deinit();
            const router_ks = try device.buffer(9 * 32 * 128 * 4, mtl.ResourceOptions.private);
            errdefer router_ks.deinit();
            const xh = try device.buffer(32 * 4096 * 2, mtl.ResourceOptions.private);
            return .{
                .sort = sort,
                .nvfp4_up = nvfp4_up,
                .nvfp4_down = nvfp4_down,
                .aff4_up = aff4_up,
                .aff4_down = aff4_down,
                .dense4 = dense4_pipe,
                .dense8 = dense8_pipe,
                .projections = proj_pipe,
                .controls_down_bf16 = ctl_down,
                .controls_apply_bf16 = ctl_apply,
                .attn_split_nax = attn_split,
                .router_bf16 = router_bf16,
                .pick_ks9 = pick_ks9,
                .meta = meta,
                .indirect = indirect,
                .router_ks = router_ks,
                .xh = xh,
            };
        }

        pub fn deinit(n: *Nax) void {
            n.xh.deinit();
            n.router_ks.deinit();
            n.indirect.deinit();
            n.meta.deinit();
            n.pick_ks9.deinit();
            n.router_bf16.deinit();
            n.attn_split_nax.deinit();
            n.controls_apply_bf16.deinit();
            n.controls_down_bf16.deinit();
            n.projections.deinit();
            n.dense8.deinit();
            n.dense4.deinit();
            n.aff4_down.deinit();
            n.aff4_up.deinit();
            n.nvfp4_down.deinit();
            n.nvfp4_up.deinit();
            n.sort.deinit();
        }
    };

    pipes: [kernels.len]mtl.Pipeline,
    gu1: mtl.Pipeline,
    dense1: mtl.Pipeline,
    projections1: mtl.Pipeline,
    projections2: mtl.Pipeline,
    router_bf16_1: mtl.Pipeline,
    router_bf16_2: mtl.Pipeline,
    router_bf16_4: mtl.Pipeline,
    router_bf16_6: mtl.Pipeline,
    controls_down_b4_bf16: mtl.Pipeline,
    controls_down_b8_bf16: mtl.Pipeline,
    head_pair: mtl.Pipeline,
    controls_down: mtl.Pipeline,
    controls_down_bf16: mtl.Pipeline,
    nax: ?Nax,
    /// Whether this device's matrix products are the fma chains the tiles rely on; if not, one-row kernels serve.
    tiles: bool,

    pub fn init(device: mtl.Device) !Pipes {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [kernels.len]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for (kernels, 0..) |k, i| {
            pipes[i] = try mtl.Pipeline.init(device, lib, k.name, false);
            made += 1;
        }
        const gu1 = try mtl.Pipeline.init(device, lib, "i2e_gu1", false);
        errdefer gu1.deinit();
        const dense1 = try mtl.Pipeline.init(device, lib, "i2r_dense1", false);
        errdefer dense1.deinit();
        const projections1 = try mtl.Pipeline.init(device, lib, "i2r_projections1", false);
        errdefer projections1.deinit();
        const projections2 = try mtl.Pipeline.init(device, lib, "i2r_projections2", false);
        errdefer projections2.deinit();
        const router_bf16_1 = try mtl.Pipeline.init(device, lib, "i2r_router_bf16_1", false);
        errdefer router_bf16_1.deinit();
        const router_bf16_2 = try mtl.Pipeline.init(device, lib, "i2r_router_bf16_2", false);
        errdefer router_bf16_2.deinit();
        const router_bf16_4 = try mtl.Pipeline.init(device, lib, "i2r_router_bf16_4", false);
        errdefer router_bf16_4.deinit();
        const router_bf16_6 = try mtl.Pipeline.init(device, lib, "i2r_router_bf16_6", false);
        errdefer router_bf16_6.deinit();
        const controls_down_b4_bf16 = try mtl.Pipeline.init(device, lib, "i2r_controls_down_b4_bf16", false);
        errdefer controls_down_b4_bf16.deinit();
        const controls_down_b8_bf16 = try mtl.Pipeline.init(device, lib, "i2r_controls_down_b8_bf16", false);
        errdefer controls_down_b8_bf16.deinit();
        const head_pair = try mtl.Pipeline.init(device, lib, "i2r_dense_ahead_2", false);
        errdefer head_pair.deinit();
        const controls_down = try mtl.Pipeline.init(device, lib, "i2r_controls_down", false);
        errdefer controls_down.deinit();
        const controls_down_bf16 = try mtl.Pipeline.init(device, lib, "i2r_controls_down_bf16", false);
        errdefer controls_down_bf16.deinit();
        var nax: ?Nax = if (device.tensorUnits()) try Nax.init(device) else null;
        errdefer if (nax) |*n| n.deinit();
        const probe = try mtl.Pipeline.init(device, lib, "i2t_probe", false);
        defer probe.deinit();
        return .{ .pipes = pipes, .gu1 = gu1, .dense1 = dense1, .projections1 = projections1, .projections2 = projections2, .router_bf16_1 = router_bf16_1, .router_bf16_2 = router_bf16_2, .router_bf16_4 = router_bf16_4, .router_bf16_6 = router_bf16_6, .controls_down_b4_bf16 = controls_down_b4_bf16, .controls_down_b8_bf16 = controls_down_b8_bf16, .head_pair = head_pair, .controls_down = controls_down, .controls_down_bf16 = controls_down_bf16, .nax = nax, .tiles = try chains(device, probe) };
    }

    pub fn deinit(p: *Pipes) void {
        if (p.nax) |*n| n.deinit();
        for (p.pipes) |pipe| pipe.deinit();
        p.gu1.deinit();
        p.dense1.deinit();
        p.projections1.deinit();
        p.projections2.deinit();
        p.router_bf16_1.deinit();
        p.router_bf16_2.deinit();
        p.router_bf16_4.deinit();
        p.router_bf16_6.deinit();
        p.controls_down_b4_bf16.deinit();
        p.controls_down_b8_bf16.deinit();
        p.head_pair.deinit();
        p.controls_down.deinit();
        p.controls_down_bf16.deinit();
    }

    pub fn routerBatchBf16(p: *const Pipes, rows: usize, n: u32, k: u32) ?Dispatch {
        if (rows == 0 or rows > max_rows or n > 256 or k % 128 != 0) return null;
        const g: u32 = (n + 7) / 8;
        if (rows == 1) return .{ .pipe = p.router_bf16_1, .groups = mtl.Size.of(g, 1, 1), .threads = mtl.Size.of(256, 1, 1) };
        if (rows == 2) return .{ .pipe = p.router_bf16_2, .groups = mtl.Size.of(g, 1, 1), .threads = mtl.Size.of(256, 1, 1) };
        if (rows <= 4) return .{ .pipe = p.router_bf16_4, .groups = mtl.Size.of(g, 1, 1), .threads = mtl.Size.of(256, 1, 1) };
        return .{ .pipe = p.router_bf16_6, .groups = mtl.Size.of(g, @intCast((rows + 5) / 6), 1), .threads = mtl.Size.of(256, 1, 1) };
    }

    pub fn controlsDownBatchBf16(p: *const Pipes, rows: usize, rank: u32, h: u32, count: u32) ?Dispatch {
        if (rows == 0 or rows > 8 or (2 * h) % 128 != 0) return null;
        const pipe = if (rows <= 4) p.controls_down_b4_bf16 else p.controls_down_b8_bf16;
        return .{ .pipe = pipe, .groups = mtl.Size.of(rank, count, 1), .threads = mtl.Size.of(128, 1, 1) };
    }

    pub fn fastDense1(p: *const Pipes, rows: usize, n: u32, k: u32, bits: u32) ?Dispatch {
        if (bits != 8 or rows != 1 or k % 256 != 0) return null;
        return .{ .pipe = p.dense1, .groups = mtl.Size.of((n + 15) / 16, 1, 1), .threads = mtl.Size.of(128, 1, 1) };
    }

    pub fn fastProjectionsFew(p: *const Pipes, rows: usize, m: [8]u32, k: u32) ?Dispatch {
        if ((rows != 1 and rows != 2) or k % 256 != 0) return null;
        const span_rows: u32 = if (rows == 1) 16 else 8;
        var groups: u32 = 0;
        for (m) |n| groups += (n + span_rows - 1) / span_rows;
        return .{ .pipe = if (rows == 1) p.projections1 else p.projections2, .groups = mtl.Size.of(groups, 1, 1), .threads = mtl.Size.of(128, 1, 1) };
    }

    pub fn fastGateUp1(p: *const Pipes, rows: usize, inter: u32, k: u32, slots: u32) ?Dispatch {
        if (rows != 1 or k % 16 != 0) return null;
        return .{ .pipe = p.gu1, .groups = mtl.Size.of((inter + 7) / 8, slots, 1), .threads = mtl.Size.of(128, 1, 1) };
    }

    pub fn naxDense(p: *const Pipes, rows: usize, n: u32, k: u32, bits: u32) ?Dispatch {
        const nax = p.nax orelse return null;
        if (bits != 8 or rows < 3 or rows > max_rows or k % 64 != 0) return null;
        if (n <= 1024) return .{ .pipe = nax.dense8, .groups = mtl.Size.of((n + 31) / 32, 1, 1), .threads = mtl.Size.of(256, 1, 1) };
        return .{ .pipe = nax.dense4, .groups = mtl.Size.of((n + 31) / 32, 1, 1), .threads = mtl.Size.of(128, 1, 1) };
    }

    pub fn naxProjections(p: *const Pipes, rows: usize, m: [8]u32, k: u32) ?Dispatch {
        const nax = p.nax orelse return null;
        if (rows < 3 or rows > max_rows or k % 64 != 0) return null;
        var groups: u32 = 0;
        for (m) |n| groups += (n + 31) / 32;
        return .{ .pipe = nax.projections, .groups = mtl.Size.of(groups, 1, 1), .threads = mtl.Size.of(128, 1, 1) };
    }

    fn pick(p: *const Pipes, kind: Kind, rows: usize, outputs: []const u32, slots: u32) Dispatch {
        const i = if (kind == .down and rows == 1) kernels.len - 1 else 2 * @as(usize, @intFromEnum(kind)) + @intFromBool(rows > tile_rows);
        const k = kernels[i];
        var groups: u32 = 0;
        for (outputs) |n| groups += (n + 8 * k.t - 1) / (8 * k.t);
        return .{ .pipe = p.pipes[i], .groups = mtl.Size.of(groups, slots, 1), .threads = mtl.Size.of(32 * k.sg, 1, 1) };
    }

    /// Affine 8-bit `n`x`k` rows for 2 to 16 rows; one row keeps `i2a_dense`.
    pub fn dense(p: *const Pipes, rows: usize, n: u32, k: u32, bits: u32) ?Dispatch {
        if (bits != 8 or rows < 2 or rows > max_rows or k % 256 != 0) return null;
        if (rows == 2 and n >= wide_output) return .{ .pipe = p.head_pair, .groups = mtl.Size.of((n + 31) / 32, 1, 1) };
        if (!p.tiles) return null;
        return p.pick(.dense, rows, &.{n}, 1);
    }

    /// A layer's fused attention projections for 3 to 16 rows; fewer keep `i2a_batch_projections`.
    pub fn fused(p: *const Pipes, rows: usize, m: [8]u32, k: u32) ?Dispatch {
        if (!p.tiles or rows < 3 or rows > max_rows or k % 256 != 0) return null;
        return p.pick(.projections, rows, &m, 1);
    }

    /// NVFP4 gate and up for 2 to 16 rows; one row keeps `i2p_gu`.
    pub fn gateUp(p: *const Pipes, rows: usize, inter: u32, k: u32, slots: u32) ?Dispatch {
        if (!p.tiles or k % 16 != 0 or rows < 2 or rows > max_rows or slots > 8 * (if (rows > tile_rows) max_rows else tile_rows)) return null;
        return p.pick(.gate_up, rows, &.{2 * inter}, slots);
    }

    /// NVFP4 down for 1 to 16 rows.
    pub fn downs(p: *const Pipes, rows: usize, n: u32, k: u32, slots: u32) ?Dispatch {
        if (!p.tiles or k % 16 != 0 or rows == 0 or rows > max_rows or slots > 8 * (if (rows > tile_rows) max_rows else tile_rows)) return null;
        return p.pick(.down, rows, &.{n}, slots);
    }

    /// Affine 4-bit routed gate and up for 2 to 16 rows (`k % 64 == 0`); one row keeps `i2a_expert_up`.
    pub fn aff4GateUp(p: *const Pipes, rows: usize, inter: u32, k: u32, slots: u32) ?Dispatch {
        if (!p.tiles or k % 64 != 0 or inter > std.math.maxInt(u32) / 2 or rows < 2 or rows > max_rows or slots > 8 * (if (rows > tile_rows) max_rows else tile_rows)) return null;
        return p.pick(.aff4_up, rows, &.{2 * inter}, slots);
    }

    /// Affine 4-bit routed down for 2 to 16 rows (`k % 64 == 0`); one row keeps `i2a_expert_down`.
    pub fn aff4Down(p: *const Pipes, rows: usize, n: u32, k: u32, slots: u32) ?Dispatch {
        if (!p.tiles or k % 64 != 0 or rows < 2 or rows > max_rows or slots > 8 * (if (rows > tile_rows) max_rows else tile_rows)) return null;
        return p.pick(.aff4_down, rows, &.{n}, slots);
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
