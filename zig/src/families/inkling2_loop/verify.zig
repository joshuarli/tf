const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const Cache = @import("state.zig").Cache;
const prefill = @import("prefill.zig");

/// The most copy drafts a decode round verifies at once: a verify of 8 rows costs about four decode steps.
pub const hybrid_drafts: usize = 7;

pub const Verification = struct {
    model: *const Model,
    cache: *Cache,
    scratch: *prefill.PrefillScratch,
    snapshot: mtl.Buffer,
    start: usize,
    rows: usize,

    pub fn begin(model: *const Model, scratch: *prefill.PrefillScratch, cache: *Cache, ids: []const u32, argmax: []u32) !Verification {
        if (ids.len == 0 or ids.len > scratch.chunk or ids.len > model.config.text.window or argmax.len != ids.len) return error.BadChunk;
        const c = &model.config;
        const row_bytes: usize = c.text.local.kv * c.text.local.dim * 4;
        const conv_bytes = cache.k_conv.length() + cache.v_conv.length() + cache.attn_conv.length() + cache.mlp_conv.length();
        const bytes = conv_bytes + Cache.localSlots(c) * ids.len * row_bytes * 2;
        const snapshot = try model.device.buffer(@max(bytes, 16), mtl.ResourceOptions.private);
        errdefer snapshot.deinit();
        var v = Verification{ .model = model, .scratch = scratch, .cache = cache, .snapshot = snapshot, .start = cache.len, .rows = ids.len };
        try v.copy(true);
        try prefill.prefillWant(model, scratch, cache, ids, .all);
        const vocab: usize = c.text.unpadded_vocab;
        const logits = scratch.logits.slice(f32, ids.len * vocab);
        for (argmax, 0..) |*out, row| {
            const values = logits[row * vocab ..][0..vocab];
            var best: usize = 0;
            for (values[1..], 1..) |value, i| if (value > values[best]) { best = i; };
            out.* = @intCast(best);
        }
        return v;
    }

    pub fn finish(v: *Verification, ids: []const u32, accepted: usize) !void {
        defer v.snapshot.deinit();
        if (ids.len != v.rows or accepted > ids.len) return error.BadAcceptedPrefix;
        if (accepted == ids.len) return;
        try v.copy(false);
        v.cache.len = v.start;
        if (accepted != 0) try prefill.prefillWant(v.model, v.scratch, v.cache, ids[0..accepted], .all);
    }

    fn copy(v: *Verification, save: bool) !void {
        const c = &v.model.config;
        const row_bytes: usize = c.text.local.kv * c.text.local.dim * 4;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = v.model.queue.commandBuffer();
        const blit = cb.blit();
        var at: usize = 0;
        for ([_]mtl.Buffer{ v.cache.k_conv, v.cache.v_conv, v.cache.attn_conv, v.cache.mlp_conv }) |buf| {
            if (save) blit.copy(buf, 0, v.snapshot, at, buf.length()) else blit.copy(v.snapshot, at, buf, 0, buf.length());
            at += buf.length();
        }
        for (0..c.slots()) |slot| {
            if (!Cache.isLocalSlot(c, slot)) continue;
            for (0..v.rows) |row| {
                const off = v.cache.row(c, slot, v.start + row);
                for ([_]mtl.Buffer{ v.cache.keys, v.cache.values }) |buf| {
                    if (save) blit.copy(buf, off, v.snapshot, at, row_bytes) else blit.copy(v.snapshot, at, buf, off, row_bytes);
                    at += row_bytes;
                }
            }
        }
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("inkling verify snapshot: {s}", .{message});
            return error.GpuFailed;
        }
    }
};
