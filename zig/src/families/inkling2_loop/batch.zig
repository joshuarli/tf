const std = @import("std");
const mtl = @import("metal");
const model = @import("model.zig");
const state = @import("state.zig");
const prefill = @import("prefill.zig");

pub const BatchScratch = struct {
    rows: prefill.PrefillScratch,
    decode: state.Scratch,

    pub fn init(device: mtl.Device, config: *const @import("config.zig").Config, capacity: usize) !BatchScratch {
        if (capacity == 0 or capacity > 8) return error.BadBatch;
        var rows = try prefill.PrefillScratch.init(device, config, capacity);
        errdefer rows.deinit();
        return .{ .rows = rows, .decode = try state.Scratch.init(device, config) };
    }

    pub fn deinit(s: *BatchScratch) void {
        s.decode.deinit();
        s.rows.deinit();
    }
};

pub fn step(m: *const model.Model, s: *BatchScratch, caches: []const *state.Cache, ids: []const u32, logits_out: []f32) !void {
    if (ids.len == 0 or ids.len > 8 or ids.len != caches.len or ids.len > s.rows.chunk) return error.BadBatch;
    const vocab = m.config.text.unpadded_vocab;
    if (logits_out.len != ids.len * vocab) return error.BadLogits;
    for (caches, 0..) |cache, i| {
        if (cache.len >= cache.capacity) return error.ContextFull;
        for (caches[0..i]) |prior| if (prior == cache) return error.DuplicateCache;
    }
    try prefill.batchStep(m, s, caches, ids);
    @memcpy(logits_out, s.rows.logits.slice(f32, logits_out.len));
}
