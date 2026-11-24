const std = @import("std");
const mtl = @import("metal");
const wts = @import("model.zig");

const Tensor = wts.Tensor;

pub const source = @embedFile("kernels/prefill_staged.metal");

pub const max_parts = 6;
pub const max_listed = 8;

pub const Pipes = struct {
    gemm: mtl.Pipeline,
    gated: mtl.Pipeline,

    pub fn init(device: mtl.Device) !Pipes {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const gemm = try mtl.Pipeline.init(device, lib, "i2ps_gemm_split", false);
        errdefer gemm.deinit();
        return .{ .gemm = gemm, .gated = try mtl.Pipeline.init(device, lib, "i2ps_split_gated", false) };
    }

    pub fn deinit(p: Pipes) void {
        p.gemm.deinit();
        p.gated.deinit();
    }
};

const StageArgs = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, bits: u32, cap: f32 };
const PartArgs = extern struct { tile: u32, row: u32, n: u32, pad: u32 = 0 };
const GemmArgs = extern struct { k: u32, rows: u32, count: u32, pad: u32 = 0, part: [max_parts]PartArgs };
const GateArgs = extern struct { count: u32 };

pub const Input = struct {
    buf: mtl.Buffer,
    off: usize = 0,
    xs: u32 = 1,
    xb: u32 = 0,

    fn same(a: Input, b: Input) bool {
        return a.buf.id == b.buf.id and a.off == b.off and a.xs == b.xs and a.xb == b.xb;
    }
};

pub const Part = struct { view: Tensor, out: mtl.Buffer, input: usize };

pub const Stage = struct {
    enc: mtl.ComputeEncoder,
    prepare: mtl.Pipeline,
    split: mtl.Pipeline,
    gemm_split: mtl.Pipeline,
    pipes: Pipes,
    weights: mtl.Buffer,
    inputs: mtl.Buffer,
    overflow: mtl.Buffer,
    rows: u32,
};

fn rowBytes(view: Tensor) usize {
    return view.shape[1] * view.affine.?.bits() / 8;
}

fn scaleRow(view: Tensor) usize {
    return (view.shape[1] / 64) * 2;
}

pub fn joined(views: []const Tensor) ?Tensor {
    var out = views[0];
    const q = out.affine orelse return null;
    if (out.shape[1] % 64 != 0) return null;
    var rows: usize = out.shape[0];
    for (views[1..]) |v| {
        const p = v.affine orelse return null;
        if (v.buffer.id != out.buffer.id or p.scales.id != q.scales.id or p.biases.id != q.biases.id) return null;
        if (p.kind != q.kind or v.shape[1] != out.shape[1] or v.offset != out.offset + rows * rowBytes(out)) return null;
        if (p.scale_offset != q.scale_offset + rows * scaleRow(out) or p.bias_offset != q.bias_offset + rows * scaleRow(out)) return null;
        rows += v.shape[0];
    }
    out.shape[0] = rows;
    out.bytes = rows * rowBytes(out);
    return out;
}

fn within(view: Tensor, src: Tensor) ?usize {
    const p = view.affine orelse return null;
    const q = src.affine.?;
    if (view.buffer.id != src.buffer.id or view.shape[1] != src.shape[1] or view.offset < src.offset) return null;
    const at = view.offset - src.offset;
    if (at % rowBytes(src) != 0) return null;
    const row = at / rowBytes(src);
    if (row + view.shape[0] > src.shape[0]) return null;
    if (p.scale_offset != q.scale_offset + row * scaleRow(src) or p.bias_offset != q.bias_offset + row * scaleRow(src)) return null;
    return row;
}

pub fn place(sources: []const Tensor, parts: []const Part, capacity: usize, rows: *[max_listed]usize) bool {
    if (sources.len == 0 or sources.len > 4 or parts.len > max_listed) return false;
    const k = sources[0].shape[1];
    var bases: [4]usize = undefined;
    var total: usize = 0;
    for (sources, 0..) |src, i| {
        if (src.affine == null or src.shape[1] != k) return false;
        bases[i] = total;
        total += src.shape[0];
    }
    if (total * k * 2 > capacity) return false;
    for (parts, 0..) |part, j| {
        rows[j] = for (sources, 0..) |src, i| {
            if (within(part.view, src)) |row| break bases[i] + row;
        } else return false;
    }
    return true;
}

fn prepare(s: Stage, w: Tensor, row: usize) void {
    const q = w.affine.?;
    const n: u32 = @intCast(w.shape[0]);
    const k: u32 = @intCast(w.shape[1]);
    s.enc.setPipeline(s.prepare);
    s.enc.setBuffer(w.buffer, w.offset, 0);
    s.enc.setBuffer(q.scales, q.scale_offset, 1);
    s.enc.setBuffer(q.biases, q.bias_offset, 2);
    s.enc.setValue(StageArgs{ .n = n, .k = k, .rows = s.rows, .xs = 1, .xb = 0, .bits = q.bits(), .cap = 0 }, 3);
    s.enc.setBuffer(s.weights, row * k * 2, 4);
    s.enc.setBuffer(s.overflow, 0, 5);
    s.enc.dispatchThreads(mtl.Size.of(@as(usize, n) * k / 4, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn encode(s: Stage, sources: []const Tensor, parts: []const Part, inputs: [3]Input) bool {
    var rows: [max_listed]usize = undefined;
    if (!place(sources, parts, s.weights.length(), &rows)) return false;
    const k: u32 = @intCast(sources[0].shape[1]);
    if (k % 4 != 0 or @as(usize, s.rows) * k * 4 > s.inputs.length()) return false;
    for (parts) |lead| {
        var count: usize = 0;
        for (parts) |part| count += @intFromBool(inputs[part.input].same(inputs[lead.input]));
        if (count > max_parts) return false;
    }
    var at: usize = 0;
    for (sources) |src| {
        prepare(s, src, at);
        at += src.shape[0];
    }
    var done: [max_listed]bool = @splat(false);
    for (parts, 0..) |lead, j| {
        if (done[j]) continue;
        const in = inputs[lead.input];
        s.enc.setPipeline(s.split);
        s.enc.setBuffer(in.buf, in.off, 0);
        s.enc.setValue(StageArgs{ .n = 0, .k = k, .rows = s.rows, .xs = in.xs, .xb = in.xb, .bits = 0, .cap = 0 }, 1);
        s.enc.setBuffer(s.inputs, 0, 2);
        s.enc.setBuffer(s.overflow, 0, 3);
        s.enc.dispatchThreads(mtl.Size.of(@as(usize, s.rows) * k / 4, 1, 1), mtl.Size.of(256, 1, 1));
        var args = GemmArgs{ .k = k, .rows = s.rows, .count = 0, .part = undefined };
        var outs: [max_parts]mtl.Buffer = undefined;
        var tiles: u32 = 0;
        for (parts[j..], j..) |part, i| {
            if (done[i] or !inputs[part.input].same(in)) continue;
            done[i] = true;
            const n: u32 = @intCast(part.view.shape[0]);
            args.part[args.count] = .{ .tile = tiles, .row = @intCast(rows[i]), .n = n };
            outs[args.count] = part.out;
            args.count += 1;
            tiles += (n + 63) / 64;
        }
        for (args.count..max_parts) |i| {
            args.part[i] = .{ .tile = tiles, .row = 0, .n = 0 };
            outs[i] = outs[0];
        }
        s.enc.setPipeline(s.pipes.gemm);
        s.enc.setBuffer(s.inputs, 0, 0);
        s.enc.setBuffer(s.weights, 0, 1);
        s.enc.setValue(args, 2);
        for (outs, 0..) |out, i| s.enc.setBuffer(out, 0, 3 + i);
        s.enc.dispatchGroups(mtl.Size.of(tiles, (s.rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
    }
    return true;
}

pub fn gatedProjection(s: Stage, w: Tensor, out: mtl.Buffer, x: mtl.Buffer, gate: mtl.Buffer) bool {
    const q = w.affine orelse return false;
    const n: u32 = @intCast(w.shape[0]);
    const k: u32 = @intCast(w.shape[1]);
    if (k % 4 != 0 or @as(usize, n) * k * 2 > s.weights.length() or @as(usize, s.rows) * k * 4 > s.inputs.length()) return false;
    prepare(s, w, 0);
    const count: u32 = s.rows * k;
    s.enc.setPipeline(s.pipes.gated);
    s.enc.setBuffer(x, 0, 0);
    s.enc.setBuffer(gate, 0, 1);
    s.enc.setValue(GateArgs{ .count = count }, 2);
    s.enc.setBuffer(s.inputs, 0, 3);
    s.enc.setBuffer(s.overflow, 0, 4);
    s.enc.dispatchThreads(mtl.Size.of(count / 4, 1, 1), mtl.Size.of(256, 1, 1));
    s.enc.setPipeline(s.gemm_split);
    s.enc.setBuffer(s.inputs, 0, 0);
    s.enc.setBuffer(s.weights, 0, 1);
    s.enc.setValue(StageArgs{ .n = n, .k = k, .rows = s.rows, .xs = 1, .xb = 0, .bits = q.bits(), .cap = 0 }, 2);
    s.enc.setBuffer(out, 0, 3);
    s.enc.dispatchGroups(mtl.Size.of((n + 63) / 64, (s.rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
    return true;
}
