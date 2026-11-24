//! A draft frame owns reusable buffers until its serial command buffers complete; parts commit in order behind a fence.
const std = @import("std");
const mtl = @import("metal");
const op = @import("operators.zig");
pub const Ref = @import("core").draft_ops.Ref;
const Entry = struct { ref: Ref, view: op.View };
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
/// Whether a frame's command buffer failed on the GPU, its Metal error logged.
fn failed(cb: mtl.CommandBuffer) bool {
    const msg = cb.failure() orelse return false;
    std.log.err("qwen27: a draft frame failed on the GPU: {s}", .{msg});
    return true;
}

/// Frames' command buffers on `mtl.clock` (host) and the GPU's clock, summed over every frame finished while set.
pub const Times = struct {
    /// Host seconds: the first frame's first commit, and the last frame's return from its wait.
    committed: f64 = 0,
    returned: f64 = 0,
    /// GPU seconds: the first buffer's start, the last buffer's end, and the buffers' spans summed.
    start: f64 = 0,
    end: f64 = 0,
    busy: f64 = 0,
    buffers: u32 = 0,
    frames: u32 = 0,
};

pub const Frame = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    /// When set, `finish` adds each frame's buffers to it (null: no clock is read).
    times: ?*Times = null,
    committed: f64 = 0,
    pool: std.ArrayList(mtl.Buffer) = .empty,
    entries: std.ArrayList(Entry) = .empty,
    cursor: usize = 0,
    epoch: u64 = 0,
    active: bool = false,
    completed: bool = false,
    cb: mtl.CommandBuffer = undefined,
    encoder: mtl.ComputeEncoder = undefined,
    external: ?Entry = null,
    external_id: u64 = 0,
    external_epoch: u64 = 0,
    fence: ?mtl.objc.Id = null,
    parts: [8]mtl.CommandBuffer = undefined,
    part_count: usize = 0,
    ops: u32 = 0,
    residency: ?mtl.ResidencySet.Follow = null,
    profiler: ?*@import("core").gpu_profile.Trace = null, // TF_Q27_PROFILE: each pass's buffers time its `drafter` class
    hold: @import("core").gpu_profile.Hold = .{}, // while open, the trace's spans keep this frame's encoder
    borrowed: bool = false, // open on another's buffer: its owner commits and waits for it
    /// A whole frame commits no part while it is encoded: its owner fills its inputs, then `submit` commits it once.
    whole: bool = false,
    sealed: bool = false,
    submitted: bool = false,
    pub fn setResident(f: *Frame, hook: ?mtl.ResidencySet.Follow) void {
        for (f.pool.items) |buffer| {
            if (f.residency) |h| h.leave(h.ctx, buffer);
            if (hook) |h| h.join(h.ctx, buffer);
        }
        f.residency = hook;
    }
    pub fn deinit(f: *Frame) void {
        f.release();
        f.setResident(null);
        if (f.fence) |fence| mtl.objc.release(fence);
        for (f.pool.items) |buffer| buffer.deinit();
        f.pool.deinit(f.allocator);
        f.entries.deinit(f.allocator);
    }
    pub fn begin(f: *Frame) !u64 {
        if (f.active or f.epoch >= 0x7fffffff) return error.DraftFrameBusy;
        f.epoch += 1;
        if (f.external_epoch != f.epoch) f.external = null;
        f.cursor = 0;
        f.entries.clearRetainingCapacity();
        f.cb = f.queue.commandBuffer();
        if (f.whole) f.cb = f.cb.retain(); // it is committed after the pool that made it is gone
        f.encoder = f.cb.compute(.serial);
        f.active = true;
        f.completed = false;
        f.sealed = false;
        f.submitted = false;
        f.part_count = 0;
        f.ops = 0;
        f.hold.take(f.profiler); // a span's own buffer would overtake this frame's pending ops
        return f.epoch;
    }
    /// Opens on another's open buffer (a target window's): ops encode into its encoder, its owner commits and waits.
    pub fn beginOn(f: *Frame, cb: mtl.CommandBuffer, encoder: mtl.ComputeEncoder) !u64 {
        if (f.active or f.epoch >= 0x7fffffff) return error.DraftFrameBusy;
        f.epoch += 1;
        if (f.external_epoch != f.epoch) f.external = null;
        f.cursor = 0;
        f.entries.clearRetainingCapacity();
        f.cb = cb;
        f.encoder = encoder;
        f.active = true;
        f.completed = false;
        f.borrowed = true;
        f.part_count = 0;
        f.ops = 0;
        return f.epoch;
    }
    /// A borrowed frame's owner saw its buffer complete: its outputs are readable until release.
    pub fn landed(f: *Frame) !void {
        if (!f.active or !f.borrowed or f.completed) return error.DraftFrameNotActive;
        f.completed = true;
    }
    /// Commits the work so far and opens the next buffer behind a fence, so the GPU starts early; between ops only.
    pub fn part(f: *Frame) !void {
        if (!f.active or f.completed or f.borrowed or f.whole or f.part_count == f.parts.len) return;
        if (f.fence == null) f.fence = mtl.objc.msg(?mtl.objc.Id, f.device.id, "newFence", .{}) orelse return error.NoFence;
        mtl.objc.msg(void, f.encoder.id, "updateFence:", .{f.fence.?});
        f.encoder.end();
        if (f.times != null and f.part_count == 0) f.committed = mtl.clock.seconds();
        f.cb.commit();
        f.parts[f.part_count] = f.cb;
        f.part_count += 1;
        f.cb = f.queue.commandBuffer();
        f.encoder = f.cb.compute(.serial);
        mtl.objc.msg(void, f.encoder.id, "waitForFence:", .{f.fence.?});
    }
    pub fn finish(f: *Frame) !void {
        if (!f.active or f.completed or f.borrowed or f.whole) return error.DraftFrameNotActive;
        f.encoder.end();
        if (f.times != null and f.part_count == 0) f.committed = mtl.clock.seconds();
        f.cb.commit();
        f.cb.wait();
        f.completed = true;
        for (f.parts[0..f.part_count]) |cb| {
            cb.wait();
            if (failed(cb)) return error.DraftGpuFailure;
        }
        if (failed(f.cb)) return error.DraftGpuFailure;
        if (f.times) |t| f.time(t);
        if (f.hold.trace) |t| {
            var seconds = f.cb.gpuSeconds();
            for (f.parts[0..f.part_count]) |cb| seconds += cb.gpuSeconds();
            t.record(.drafter, seconds);
        }
    }
    /// Ends a whole frame's encoding; nothing of it runs until `submit`.
    pub fn seal(f: *Frame) !void {
        if (!f.active or f.completed or !f.whole or f.sealed) return error.DraftFrameNotActive;
        f.encoder.end();
        f.sealed = true;
    }
    /// Commits a sealed frame once its inputs are filled.
    pub fn submit(f: *Frame) !void {
        if (!f.active or !f.sealed or f.submitted) return error.DraftFrameNotActive;
        if (f.times != null) f.committed = mtl.clock.seconds();
        f.cb.commit();
        f.submitted = true;
    }
    /// Waits for a submitted frame: its outputs are readable until release.
    pub fn land(f: *Frame) !void {
        if (!f.active or !f.submitted or f.completed) return error.DraftFrameNotActive;
        f.cb.wait();
        f.completed = true;
        if (failed(f.cb)) return error.DraftGpuFailure;
        if (f.times) |t| f.time(t);
    }
    fn time(f: *const Frame, t: *Times) void {
        const returned = mtl.clock.seconds();
        var start = f.cb.gpuStart();
        var end = f.cb.gpuEnd();
        var busy = end - start;
        for (f.parts[0..f.part_count]) |cb| {
            start = @min(start, cb.gpuStart());
            end = @max(end, cb.gpuEnd());
            busy += cb.gpuSeconds();
        }
        if (t.frames == 0) {
            t.committed = f.committed;
            t.start = start;
        }
        t.end = end;
        t.returned = returned;
        t.busy += busy;
        t.buffers += @intCast(f.part_count + 1);
        t.frames += 1;
    }
    pub fn release(f: *Frame) void {
        if (f.active and f.whole) {
            if (!f.sealed) f.encoder.end();
            if (f.submitted and !f.completed) f.cb.wait(); // committed work still uses the pool
            f.cb.release();
            f.sealed = false;
            f.submitted = false;
        } else if (f.active and !f.completed and !f.borrowed) {
            f.encoder.end();
            for (f.parts[0..f.part_count]) |cb| cb.wait(); // committed parts still use the pool
        }
        f.hold.release();
        f.active = false;
        f.borrowed = false;
        f.external = null;
    }
    pub fn require(f: *const Frame, epoch: u64) !void {
        if (!f.active or f.completed or f.sealed or f.epoch != epoch) return error.DraftFrameNotActive;
    }
    pub fn add(f: *Frame, r: Ref, rows: u32, width: u32, stride: u32, dtype: @import("../checkpoint.zig").DType) !op.View {
        const view = op.View{ .handle = (f.epoch << 32) | (f.entries.items.len + 1), .rows = rows, .width = width, .stride = stride, .dtype = dtype };
        try view.check(rows, width, dtype);
        const bytes = try std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, rows - 1, stride), width), dtype.size());
        if (r.off > r.buf.length() or bytes > r.buf.length() - r.off) return error.DraftOperatorShape;
        try f.entries.append(f.allocator, .{ .ref = r, .view = view });
        return view;
    }
    pub fn alloc(f: *Frame, rows: u32, width: u32, dtype: @import("../checkpoint.zig").DType) !op.View {
        if (!f.active or f.completed or rows == 0 or width == 0) return error.DraftFrameNotActive;
        const bytes = try std.math.mul(usize, try std.math.mul(usize, rows, width), dtype.size());
        // Each slot keeps less than 64 KiB of spare scratch as context scores grow across rounds.
        const reserved = (try std.math.add(usize, bytes, 65535)) & ~@as(usize, 65535);
        if (f.cursor == f.pool.items.len) {
            const buffer = try f.device.buffer(reserved, options);
            f.pool.append(f.allocator, buffer) catch |err| {
                buffer.deinit();
                return err;
            };
            if (f.residency) |h| h.join(h.ctx, buffer);
        } else if (f.pool.items[f.cursor].length() < bytes) {
            const buffer = try f.device.buffer(reserved, options);
            if (f.residency) |h| h.leave(h.ctx, f.pool.items[f.cursor]);
            f.pool.items[f.cursor].deinit();
            f.pool.items[f.cursor] = buffer;
            if (f.residency) |h| h.join(h.ctx, buffer);
        }
        const buffer = f.pool.items[f.cursor];
        f.cursor += 1;
        return f.add(.{ .buf = buffer }, rows, width, width, dtype);
    }
    pub fn ref(f: *const Frame, v: op.View) !Ref {
        if (v.handle & (@as(u64, 1) << 63) != 0) {
            if (!f.active or f.external_epoch != f.epoch) return error.StaleDraftView;
            const entry = f.external orelse return error.StaleDraftView;
            if (!std.meta.eql(v, entry.view)) return error.StaleDraftView;
            return entry.ref;
        }
        const index: usize = @intCast(v.handle & 0xffffffff);
        if (!f.active or v.handle >> 32 != f.epoch or index == 0 or index > f.entries.items.len) return error.StaleDraftView;
        const entry = f.entries.items[index - 1];
        if (!std.meta.eql(v, entry.view)) return error.StaleDraftView;
        return entry.ref;
    }
    pub fn borrow(f: *Frame, r: Ref, rows: u32, width: u32) !op.View {
        if (f.active or rows == 0 or width == 0 or r.off > r.buf.length() or @as(usize, rows) * width * 2 > r.buf.length() - r.off) return error.DraftFrameBusy;
        f.external_id += 1;
        if (f.external_id >= (@as(u64, 1) << 63)) return error.DraftFrameBusy;
        const view = op.View{ .handle = (@as(u64, 1) << 63) | f.external_id, .rows = rows, .width = width, .stride = width, .dtype = .bf16 };
        f.external = .{ .ref = r, .view = view };
        f.external_epoch = f.epoch + 1;
        return view;
    }
};
