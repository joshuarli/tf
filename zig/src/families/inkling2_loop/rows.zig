const std = @import("std");
const mtl = @import("metal");

pub const source = @embedFile("kernels/rows.metal") ++ "\n" ++ @embedFile("kernels/gather.metal");

pub const widths = [_]u32{ 2, 4, 8 };
pub const wide_output: u32 = 65536;

pub const Dispatch = struct { pipe: mtl.Pipeline, groups: mtl.Size, threads: mtl.Size = mtl.Size.of(256, 1, 1) };

pub const Pipes = struct {
    wide: [widths.len]mtl.Pipeline,
    staged: [widths.len]mtl.Pipeline,
    projections: mtl.Pipeline,
    gate_up: [2]mtl.Pipeline,
    down: mtl.Pipeline,

    pub fn init(device: mtl.Device) !Pipes {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipes = undefined;
        var list: [2 * widths.len + 4]*mtl.Pipeline = undefined;
        var names: [list.len][40]u8 = undefined;
        var texts: [list.len][]const u8 = undefined;
        var n: usize = 0;
        for (widths, 0..) |rows, i| {
            for ([_]*mtl.Pipeline{ &p.wide[i], &p.staged[i] }, [_][]const u8{ "i2r_dense_ahead", "i2r_dense_staged" }) |pipe, base| {
                list[n] = pipe;
                texts[n] = try std.fmt.bufPrint(&names[n], "{s}_{d}", .{ base, rows });
                n += 1;
            }
        }
        for ([_]*mtl.Pipeline{ &p.projections, &p.gate_up[0], &p.gate_up[1], &p.down }, [_][]const u8{ "i2r_projections_8", "i2g_gate_up_1", "i2g_gate_up_2", "i2g_down_all_2" }) |pipe, name| {
            list[n] = pipe;
            texts[n] = name;
            n += 1;
        }
        var made: usize = 0;
        errdefer for (list[0..made]) |pipe| pipe.deinit();
        for (list[0..n], texts[0..n]) |pipe, text| {
            pipe.* = try mtl.Pipeline.init(device, lib, text, false);
            made += 1;
        }
        return p;
    }

    pub fn deinit(p: *Pipes) void {
        for (p.wide) |pipe| pipe.deinit();
        for (p.staged) |pipe| pipe.deinit();
        p.projections.deinit();
        for (p.gate_up) |pipe| pipe.deinit();
        p.down.deinit();
    }

    pub fn dense(p: *const Pipes, rows: usize, n: u32, k: u32, bits: u32) ?Dispatch {
        const i = width(rows) orelse return null;
        if (bits != 8 or k % 64 != 0) return null;
        if (n >= wide_output) return .{ .pipe = p.wide[i], .groups = mtl.Size.of((n + 31) / 32, 1, 1) };
        return .{ .pipe = p.staged[i], .groups = mtl.Size.of((n + 15) / 16, 1, 1) };
    }

    pub fn fused(p: *const Pipes, rows: usize, total: usize, k: u32) ?Dispatch {
        const i = width(rows) orelse return null;
        if (i < widths.len - 1 or k % 64 != 0) return null;
        return .{ .pipe = p.projections, .groups = mtl.Size.of((total + 31) / 32, (rows + 3) / 4, 1) };
    }

    pub fn gateUp(p: *const Pipes, rows: usize, inter: u32, k: u32, slots: u32) ?Dispatch {
        const i = width(rows) orelse return null;
        if (k % 16 != 0 or slots > 64) return null;
        const wide: usize = @intFromBool(i > 0);
        const cols: u32 = if (wide == 1) 16 else 8;
        return .{ .pipe = p.gate_up[wide], .groups = mtl.Size.of((inter + cols - 1) / cols, slots, 1) };
    }

    pub fn downs(p: *const Pipes, rows: usize, n: u32, k: u32, slots: u32) ?Dispatch {
        _ = width(rows) orelse return null;
        if (k % 16 != 0 or slots > 64) return null;
        return .{ .pipe = p.down, .groups = mtl.Size.of((n + 15) / 16, slots, 1) };
    }
};

fn width(rows: usize) ?usize {
    if (rows < 2) return null;
    for (widths, 0..) |w, i| if (rows <= w) return i;
    return null;
}

test "the few-row kernels compile" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try Pipes.init(device);
    p.deinit();
}

test "batches of two to eight rows take a few-row kernel and one row keeps its own" {
    try std.testing.expectEqual(@as(?usize, null), width(1));
    try std.testing.expectEqual(@as(?usize, 0), width(2));
    try std.testing.expectEqual(@as(?usize, 1), width(3));
    try std.testing.expectEqual(@as(?usize, 1), width(4));
    try std.testing.expectEqual(@as(?usize, 2), width(5));
    try std.testing.expectEqual(@as(?usize, 2), width(8));
    try std.testing.expectEqual(@as(?usize, null), width(9));
}
