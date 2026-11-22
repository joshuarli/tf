//! The same decodes on the host in fp64: the reference the kernels are checked against, with the
//! fp32 accumulation bound each output may differ by.
const std = @import("std");
const shapes = @import("fp4_shapes.zig");

pub const Caps = struct { gate: f64 = 4.0, up: f64 = 25.0 };

pub const nvfp4_codes: [16]f64 = .{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6 };

pub fn e2m1(n: u8) f64 {
    return nvfp4_codes[n & 15];
}

pub fn e4m3(b: u8) f64 {
    if (b & 0x7f == 0x7f) return std.math.nan(f64);
    const sign: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 0xf;
    const m: f64 = @floatFromInt(b & 7);
    return sign * if (e == 0) m * (1.0 / 512.0) else (1.0 + m / 8.0) * std.math.ldexp(@as(f64, 1), e - 7);
}

pub fn e8m0(b: u8) f64 {
    if (b == 0xff) return std.math.nan(f64);
    return std.math.ldexp(@as(f64, 1), @as(i32, b) - 127);
}

pub const Net = struct {
    fmt: shapes.Format,
    k: usize,
    n: usize,
    rows: usize,
    codes: []const u8,
    scales: []const u8,
    global: f64,

    pub fn codeStride(self: Net) usize {
        return if (self.fmt == .nvfp4) self.k / 2 else self.k;
    }

    pub fn scaleStride(self: Net) usize {
        return if (self.fmt == .nvfp4) self.k / 16 else self.k / 32;
    }
};

/// The output rounding plus the k-deep fp32 accumulation the GPU's order may introduce.
pub fn sumBound(y: f64, terms: f64, k: usize) f64 {
    const u = std.math.ldexp(@as(f64, 1), -24);
    const gamma = @as(f64, @floatFromInt(k + 64)) * u;
    return 4.0 * gamma * terms + 2.0 * u * @abs(y);
}

fn gateOf(g: f64, cap: f64) f64 {
    return cap * std.math.tanh(g / cap) * (1.0 / (1.0 + @exp(-g)));
}

fn upOf(u: f64, cap: f64) f64 {
    return if (cap > 0) cap * std.math.tanh(u / cap) else u;
}

/// The situ product and its act bound for n intermediates against one x row.
pub const GuOut = struct {
    act: []f64,
    bound: []f64,

    pub fn init(a: std.mem.Allocator, n: usize) !GuOut {
        return .{ .act = try a.alloc(f64, n), .bound = try a.alloc(f64, n) };
    }

    pub fn deinit(self: GuOut, a: std.mem.Allocator) void {
        a.free(self.act);
        a.free(self.bound);
    }
};

fn guFmt(comptime f: shapes.Format, net: Net, x: []const f32, out: GuOut, caps: Caps) void {
    const group: usize = f.scaleGroup();
    const cs = net.codeStride();
    const ss = net.scaleStride();
    for (0..net.n) |c| {
        const rg = 2 * c;
        const ru = rg + 1;
        var g: f64 = 0;
        var u: f64 = 0;
        var tg: f64 = 0;
        var tu: f64 = 0;
        for (0..net.k / group) |grp| {
            const sg = (if (f == .nvfp4) e4m3(net.scales[rg * ss + grp]) * net.global else e8m0(net.scales[rg * ss + grp]));
            const su = (if (f == .nvfp4) e4m3(net.scales[ru * ss + grp]) * net.global else e8m0(net.scales[ru * ss + grp]));
            for (0..group) |j| {
                const col = grp * group + j;
                const xv: f64 = @floatCast(x[col]);
                var wg: f64 = undefined;
                var wu: f64 = undefined;
                if (f == .nvfp4) {
                    const bg = net.codes[rg * cs + col / 2];
                    const bu = net.codes[ru * cs + col / 2];
                    wg = e2m1(if (col & 1 == 0) bg & 15 else bg >> 4) * sg;
                    wu = e2m1(if (col & 1 == 0) bu & 15 else bu >> 4) * su;
                } else {
                    wg = e4m3(net.codes[rg * cs + col]) * sg;
                    wu = e4m3(net.codes[ru * cs + col]) * su;
                }
                g += wg * xv;
                u += wu * xv;
                tg += @abs(wg * xv);
                tu += @abs(wu * xv);
            }
        }
        const gate = gateOf(g, caps.gate);
        const up = upOf(u, caps.up);
        out.act[c] = gate * up;
        const bg = sumBound(g, tg, net.k);
        const bu = sumBound(u, tu, net.k);
        out.bound[c] = 1.5 * @abs(up) * bg + @abs(gate) * bu + 2e-6 * @abs(gate * up) + 1e-12;
    }
}

pub fn gu(net: Net, x: []const f32, out: GuOut, caps: Caps) void {
    switch (net.fmt) {
        inline else => |f| guFmt(f, net, x, out, caps),
    }
}

/// out[row] = sum over k of W[row, k] * x[k], with each row's fp32 bound.
pub fn down(net: Net, x: []const f32, out: []f64, bound: []f64) void {
    switch (net.fmt) {
        inline else => |f| downFmt(f, net, x, out, bound),
    }
}

fn downFmt(comptime f: shapes.Format, net: Net, x: []const f32, out: []f64, bound: []f64) void {
    const group: usize = f.scaleGroup();
    const cs = net.codeStride();
    const ss = net.scaleStride();
    for (0..net.n) |r| {
        var y: f64 = 0;
        var terms: f64 = 0;
        for (0..net.k / group) |grp| {
            const s = if (f == .nvfp4) e4m3(net.scales[r * ss + grp]) * net.global else e8m0(net.scales[r * ss + grp]);
            for (0..group) |j| {
                const col = grp * group + j;
                const xv: f64 = @floatCast(x[col]);
                const w = if (f == .nvfp4) blk: {
                    const b = net.codes[r * cs + col / 2];
                    break :blk e2m1(if (col & 1 == 0) b & 15 else b >> 4) * s;
                } else e4m3(net.codes[r * cs + col]) * s;
                y += w * xv;
                terms += @abs(w * xv);
            }
        }
        out[r] = y;
        bound[r] = sumBound(y, terms, net.k);
    }
}

test "the decoders agree with the reference tables" {
    try std.testing.expectEqual(@as(f64, 6), e2m1(7));
    try std.testing.expectEqual(@as(f64, -6), e2m1(15));
    try std.testing.expectEqual(@as(f64, 1), e4m3(0x38));
    try std.testing.expectEqual(@as(f64, 0.5), e4m3(0x30));
    try std.testing.expect(std.math.isNan(e4m3(0x7f)));
    try std.testing.expectEqual(@as(f64, 1), e8m0(127));
    try std.testing.expectEqual(@as(f64, 0.5), e8m0(126));
    try std.testing.expectEqual(@as(f64, 2), e8m0(128));
    try std.testing.expect(std.math.isNan(e8m0(255)));
}
