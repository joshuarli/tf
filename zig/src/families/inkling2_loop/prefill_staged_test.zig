const std = @import("std");
const mtl = @import("metal");
const model = @import("model.zig");
const state = @import("state.zig");
const prefill = @import("prefill.zig");
const staged = @import("prefill_staged.zig");

const t = std.testing;

const Snapshot = struct {
    logits: []f32,
    buffers: [6][]u8,

    fn take(m: *const model.Model, cache: *const state.Cache, logits: []const f32) !Snapshot {
        var out: Snapshot = .{ .logits = try t.allocator.dupe(f32, logits), .buffers = undefined };
        const sources = [_]mtl.Buffer{ cache.keys, cache.values, cache.k_conv, cache.v_conv, cache.attn_conv, cache.mlp_conv };
        var shared: [6]mtl.Buffer = undefined;
        for (sources, 0..) |src, i| shared[i] = try m.device.buffer(src.length(), mtl.ResourceOptions.shared);
        defer for (shared) |buf| buf.deinit();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const blit = cb.blit();
        for (sources, shared) |src, dst| blit.copy(src, 0, dst, 0, src.length());
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        for (shared, 0..) |buf, i| out.buffers[i] = try t.allocator.dupe(u8, buf.contents()[0..buf.length()]);
        return out;
    }

    fn deinit(s: Snapshot) void {
        t.allocator.free(s.logits);
        for (s.buffers) |buf| t.allocator.free(buf);
    }
};

fn fill(m: *const model.Model, ps: *prefill.PrefillScratch, ids: []const u32, chunks: []const usize) !Snapshot {
    var cache = try state.Cache.init(m.device, &m.config, ids.len);
    defer cache.deinit();
    var at: usize = 0;
    for (chunks) |n| {
        try prefill.prefillWant(m, ps, &cache, ids[at..][0..n], if (at + n == ids.len) .all else .none);
        at += n;
    }
    try t.expectEqual(ids.len, cache.len);
    return Snapshot.take(m, &cache, ps.logits.slice(f32, chunks[chunks.len - 1] * m.config.text.unpadded_vocab));
}

test "oQ4e exact fused prefill keeps the unfused logits and cache state byte for byte" {
    const path = std.c.getenv("TF_INKLING2_LOOP_OQ4E_DIR") orelse return error.SkipZigTest;
    const m = try model.Model.load(t.allocator, t.io, std.mem.span(path));
    defer m.deinit();
    if (!m.device.tensorUnits()) return error.SkipZigTest;
    var ids: [309]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1000 + (i * 7919) % 50000);
    for (m.weights.layers, 0..) |lay, i| {
        const a = &lay.attn;
        if (m.config.isFull(@intCast(i))) {
            try t.expect(staged.joined(&.{ a.q, a.r, a.gate, a.sink }) != null);
            try t.expect(staged.joined(&.{ a.k, a.k_dyn.? }) != null);
            try t.expect(staged.joined(&.{ a.q, a.gate }) == null);
        } else {
            const all = staged.joined(&.{ a.q, a.k, a.v, a.r, a.gate, a.sink }) orelse return error.NotJoined;
            try t.expectEqual(@as(usize, 8600), all.shape[0]);
            try t.expect(staged.joined(&.{ a.k, a.q }) == null);
            try t.expect(staged.joined(&.{ a.q, a.v }) == null);
        }
    }
    var ps = try prefill.PrefillScratch.initRounding(m.device, &m.config, 163, 163, .{});
    defer ps.deinit();
    try t.expect(ps.pipes.staged != null);
    const capacity = ps.affine_stage_w.?.length();
    var rows: [staged.max_listed]usize = undefined;
    for (m.weights.layers, 0..) |lay, i| {
        const a = &lay.attn;
        const out = ps.q;
        if (m.config.isFull(@intCast(i))) {
            const sources = [_]model.Tensor{ staged.joined(&.{ a.q, a.r, a.gate, a.sink }).?, staged.joined(&.{ a.k, a.k_dyn.? }).?, staged.joined(&.{ a.v, a.v_dyn.? }).? };
            const parts = [_]staged.Part{ .{ .view = a.q, .out = out, .input = 0 }, .{ .view = a.sink, .out = out, .input = 0 }, .{ .view = a.k_dyn.?, .out = out, .input = 1 }, .{ .view = a.v, .out = out, .input = 2 } };
            try t.expect(staged.place(&sources, &parts, capacity, &rows));
            try t.expectEqual(@as(usize, 3072 + 384 + 3072), rows[1]);
            try t.expectEqual(@as(usize, 6552 + 1024), rows[2]);
            try t.expectEqual(@as(usize, 6552 + 1056), rows[3]);
        } else {
            const sources = [_]model.Tensor{staged.joined(&.{ a.q, a.k, a.v, a.r, a.gate, a.sink }).?};
            const parts = [_]staged.Part{ .{ .view = a.v, .out = out, .input = 2 }, .{ .view = a.sink, .out = out, .input = 0 } };
            try t.expect(staged.place(&sources, &parts, capacity, &rows));
            try t.expectEqual(@as(usize, 4096), rows[0]);
            try t.expectEqual(@as(usize, 8576), rows[1]);
            try t.expect(!staged.place(&sources, &parts, 8600 * 2304 * 2 - 1, &rows));
        }
    }
    for ([_][]const usize{ &.{ 137, 163, 9 }, &.{ 9, 137, 163 }, &.{ 163, 137, 5, 1, 3 } }) |chunks| {
        ps.fuse = true;
        const fused = try fill(m, &ps, &ids, chunks);
        defer fused.deinit();
        ps.fuse = false;
        const plain = try fill(m, &ps, &ids, chunks);
        defer plain.deinit();
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(plain.logits), std.mem.sliceAsBytes(fused.logits));
        for (plain.buffers, fused.buffers) |a, b| try t.expectEqualSlices(u8, a, b);
    }
}
