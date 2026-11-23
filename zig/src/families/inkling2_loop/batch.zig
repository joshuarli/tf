const std = @import("std");
const mtl = @import("metal");
const model = @import("model.zig");
const state = @import("state.zig");
const prefill = @import("prefill.zig");

pub const BatchScratch = struct {
    rows: prefill.PrefillScratch,
    decode: state.Scratch,
    parts: mtl.Buffer,
    split: mtl.Pipeline,
    combine: mtl.Pipeline,

    pub fn init(device: mtl.Device, config: *const @import("config.zig").Config, capacity: usize) !BatchScratch {
        if (capacity == 0 or capacity > 8) return error.BadBatch;
        var rows = try prefill.PrefillScratch.init(device, config, capacity);
        errdefer rows.deinit();
        var decode = try state.Scratch.init(device, config);
        errdefer decode.deinit();
        const heads = @max(config.text.local.q, config.text.global.q);
        const dim = @max(config.text.local.dim, config.text.global.dim);
        const parts = try device.buffer(capacity * heads * state.attn_splits * (dim + 2) * 4, mtl.ResourceOptions.private);
        errdefer parts.deinit();
        const lib = try mtl.Library.fromSource(device, @import("kernels.zig").attention, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const split = try mtl.Pipeline.init(device, lib, "i2_batch_split", false);
        errdefer split.deinit();
        const combine = try mtl.Pipeline.init(device, lib, "i2_batch_combine", false);
        return .{ .rows = rows, .decode = decode, .parts = parts, .split = split, .combine = combine };
    }

    pub fn deinit(s: *BatchScratch) void {
        s.combine.deinit();
        s.split.deinit();
        s.parts.deinit();
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
