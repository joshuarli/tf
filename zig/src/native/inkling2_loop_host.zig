const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const cache_fit = @import("cache_fit.zig");
const prefix = @import("inkling2_loop_prefix.zig");
const Allocator = std.mem.Allocator;

const Job = struct {
    id: api.Id,
    request: *const api.Request,
    sink: api.Sink,
    emitted: std.ArrayList(u32) = .empty,
    queued: i96 = 0,
    began: i96 = 0,
    prefill_sent: bool = false,
    prefill_seconds: ?f64 = null,
    rounds: u64 = 0,
    reused: usize = 0,
    saved: bool = false,
};

pub const sampling_refusal = "Inkling 2 Mini decodes greedily only: send temperature 0 or leave it out";

fn visionPrepare(_: *anyopaque, _: Allocator, _: []const u8, _: u32) anyerror!api.PreparedImage {
    return error.VisionUnsupported;
}

fn visionExplain(err: anyerror) ?[]const u8 {
    return inkling.refuse.explain(err);
}

const Slot = struct {
    cache: inkling.Cache,
    resident: std.ArrayList(u32) = .empty,
    last_used: u64 = 0,
    rollback: ?inkling.Cache.Rollback = null,
    job: ?*Job = null,
    live_generated: u64 = 0,
};

pub const Host = struct {
    gpa: Allocator,
    io: std.Io,
    model: *inkling.Model,
    slots: []Slot,
    batch: inkling.BatchScratch,
    batch_logits: []f32,
    sequence: u64 = 0,
    prefill: inkling.PrefillScratch,
    capacity: usize,
    eos: u32,
    prefix_reuse: bool,
    invalidated: std.atomic.Value(bool) = .init(false),
    info_: api.Info,
    load_limit: u64 = 0,
    vision_markers: [1]u32 = undefined,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(api.Id) = .empty,
    closing: bool = false,
    thread: ?std.Thread = null,
    decoded: std.ArrayList(Mark) = .empty,
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    prefilling: bool = false,
    const Mark = struct { at: i96, tokens: u64 };
    const window_ns: i96 = 2 * std.time.ns_per_s;

    const vtable: api.Engine.VTable = .{
        .info = infoFn,
        .submit = submitFn,
        .cancel = cancelFn,
        .status = statusFn,
        .memory = memoryFn,
        .refusal = refusalFn,
        .drain = drainFn,
    };

    pub fn engine(h: *Host) api.Engine {
        return .{ .ctx = h, .vtable = &vtable };
    }

    pub fn start(h: *Host) !void {
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    pub fn stop(h: *Host) void {
        drainFn(h);
        h.queued.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        for (h.slots) |*slot| {
            slot.resident.deinit(h.gpa);
            slot.cache.deinit();
            if (slot.rollback) |*rollback| rollback.deinit();
        }
        h.gpa.free(h.slots);
        h.batch.deinit();
        h.gpa.free(h.batch_logits);
    }
    fn self(ctx: *anyopaque) *Host {
        return @ptrCast(@alignCast(ctx));
    }

    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }

    fn now(h: *Host) i96 {
        return std.Io.Clock.awake.now(h.io).toNanoseconds();
    }

    fn infoFn(ctx: *anyopaque) api.Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch {
            h.invalidated.store(true, .release);
            return error.Busy;
        };
        job.* = .{ .id = id, .request = request, .sink = sink, .queued = h.now() };
        h.lock();
        defer h.unlock();
        if (h.closing) {
            h.invalidated.store(true, .release);
            h.gpa.destroy(job);
            return error.Closed;
        }
        var at = h.queued.items.len;
        while (!request.background and at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        h.queued.insert(h.gpa, at, job) catch {
            h.invalidated.store(true, .release);
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: api.Id) void {
        const h = self(ctx);
        h.invalidated.store(true, .release);
        h.lock();
        defer h.unlock();
        for (h.queued.items, 0..) |job, i| if (job.id == id) {
            _ = h.queued.orderedRemove(i);
            h.unlock();
            std.debug.print("inkling prefix: reused 0 of {d} prompt tokens\n", .{job.request.prompt.len});
            h.finish(job, .cancelled, .{}, "");
            h.lock();
            return;
        };
        h.cancels.append(h.gpa, id) catch {};
    }

    fn statusFn(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        const t = h.now();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= t - window_ns) tokens += m.tokens;
        }
        var n: usize = 0;
        var generation_tokens: u64 = 0;
        var running: usize = 0;
        for (h.slots) |*slot| {
            if (slot.job) |job| {
                running += 1;
                generation_tokens += slot.live_generated;
                if (n < stream_tokens.len) {
                    stream_tokens[n] = @intCast(job.request.prompt.len + slot.live_generated);
                    n += 1;
                }
            }
        }
        out.* = .{
            .healthy = !h.closing,
            .running = @intCast(running),
            .waiting = @intCast(h.queued.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (t - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .streams = n,
            .generation_tokens = generation_tokens,
            .fill_unknown = h.prefilling,
        };
    }

    fn memoryFn(ctx: *anyopaque, reset_peak: bool) ?api.Memory {
        return cache_fit.hostMemory(self(ctx).load_limit, reset_peak);
    }

    fn refusalFn(ctx: *anyopaque, request: *const api.Request, out: []u8) ?[]const u8 {
        const h = self(ctx);
        if (request.images.len > 0) {
            h.invalidated.store(true, .release);
            std.debug.print("inkling prefix: reused 0 of {d} prompt tokens\n", .{request.prompt.len});
            const msg = inkling.refuse.image;
            const n = @min(msg.len, out.len);
            @memcpy(out[0..n], msg[0..n]);
            return out[0..n];
        }
        if (request.sampling != null) {
            h.invalidated.store(true, .release);
            std.debug.print("inkling prefix: reused 0 of {d} prompt tokens\n", .{request.prompt.len});
            const msg = sampling_refusal;
            const n = @min(msg.len, out.len);
            @memcpy(out[0..n], msg[0..n]);
            return out[0..n];
        }
        return null;
    }

    fn drainFn(ctx: *anyopaque) void {
        const h = self(ctx);
        h.lock();
        h.closing = true;
        h.wake.broadcast(h.io);
        h.unlock();
        if (h.thread) |t| t.join();
        h.thread = null;
    }

    fn emit(job: *Job, event: api.Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    fn finish(h: *Host, job: *Job, reason: api.Reason, stats: api.Stats, message: []const u8) void {
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        job.emitted.deinit(h.gpa);
        h.gpa.destroy(job);
    }

    fn noteDecoded(h: *Host, n: usize) void {
        const t = h.now();
        h.lock();
        defer h.unlock();
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < t - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = t, .tokens = n }) catch {};
    }

    fn cancelled(h: *Host, job: *Job) bool {
        h.lock();
        defer h.unlock();
        const foreground_waiting = h.queued.items.len > 0 and !h.queued.items[0].request.background;
        return prefix.cancelled(job.id, job.request.background, h.closing, foreground_waiting, h.cancels.items);
    }

    fn retire(h: *Host, slot: *Slot, reason0: api.Reason, message0: []const u8) void {
        const job = slot.job.?;
        var reason = reason0;
        var message = message0;
        var restored = false;
        if (job.saved and prefix.canRestore(slot.rollback.?.len, slot.cache.len, slot.rollback.?.saved_rows, h.model.config.text.window, slot.rollback.?.limit, job.emitted.items.len)) {
            if (slot.rollback.?.copy(h.model.queue, &h.model.config, &slot.cache, false)) |_| {
                restored = true;
            } else |e| {
                reason = .failed;
                message = @errorName(e);
            }
        }
        std.debug.print("inkling prefix: reused {d} of {d} prompt tokens\n", .{ job.reused, job.request.prompt.len });
        const stats: api.Stats = .{ .rounds = job.rounds, .drafted = 0, .accepted = 0, .min_rows = 1, .prefill_seconds = job.prefill_seconds, .queued_seconds = @as(f64, @floatFromInt(@max(0, job.began - job.queued))) / 1e9 };
        h.lock();
        prefix.release(slot, restored);
        if (std.mem.indexOfScalar(api.Id, h.cancels.items, job.id)) |i| _ = h.cancels.swapRemove(i);
        h.unlock();
        _ = h.invalidated.swap(false, .acq_rel);
        h.finish(job, reason, stats, message);
    }

    fn argmax(logits: []const f32) u32 {
        var best: u32 = 0;
        var value = logits[0];
        for (logits[1..], 1..) |v, i| if (v > value) {
            value = v;
            best = @intCast(i);
        };
        return best;
    }

    fn token(h: *Host, slot: *Slot, best: u32) void {
        const job = slot.job.?;
        const r = job.request;
        job.emitted.append(h.gpa, best) catch {
            h.retire(slot, .failed, @errorName(error.OutOfMemory));
            return;
        };
        var one: [1]u32 = .{best};
        emit(job, .{ .tokens = &one });
        h.lock();
        slot.live_generated = @intCast(job.emitted.items.len);
        h.unlock();
        h.noteDecoded(1);
        const stop_hit = if (r.stop) |stop_rule| stop_rule.check(stop_rule.ctx, job.emitted.items) else false;
        const is_eos = r.eos.len > 0 and (std.mem.indexOfScalar(u32, r.eos, best) != null or best == h.eos);
        if (stop_hit or is_eos) {
            h.retire(slot, .stop, "");
        } else if (job.emitted.items.len >= r.max_tokens) {
            h.retire(slot, .length, "");
        }
    }

    fn admit(h: *Host, slot: *Slot, choice: prefix.Choice) void {
        const job = slot.job.?;
        const r = job.request;
        job.began = h.now();
        if (r.images.len > 0 or r.prompt.len == 0 or r.prompt.len + r.max_tokens > h.capacity or r.sampling != null) {
            emit(job, .{ .prefilled = 0 });
            job.prefill_sent = true;
            const message: []const u8 = if (r.images.len > 0) inkling.refuse.image else if (r.sampling != null) sampling_refusal else if (r.prompt.len == 0) "rendered prompt is empty" else "the prompt and max_tokens do not fit this server's --context";
            h.retire(slot, .failed, message);
            return;
        }
        h.sequence += 1;
        slot.last_used = h.sequence;
        slot.resident.ensureTotalCapacity(h.gpa, r.prompt.len) catch {
            h.retire(slot, .failed, @errorName(error.OutOfMemory));
            return;
        };
        job.reused = choice.reused;
        if (job.reused == 0) {
            slot.cache.len = 0;
            slot.resident.clearRetainingCapacity();
        }
        std.debug.print("inkling session: slot {d}\n", .{choice.index});
        const prefill_start = h.now();
        var at = job.reused;
        while (at < r.prompt.len) {
            if (h.cancelled(job)) {
                emit(job, .{ .prefilled = 0 });
                h.retire(slot, .cancelled, "");
                return;
            }
            const take = @min(h.prefill.chunk, r.prompt.len - at);
            inkling.prefillWant(h.model, &h.prefill, &slot.cache, r.prompt[at..][0..take], if (at + take == r.prompt.len) .last else .none) catch |e| {
                emit(job, .{ .prefilled = 0 });
                h.retire(slot, .failed, @errorName(e));
                return;
            };
            slot.resident.appendSliceAssumeCapacity(r.prompt[at..][0..take]);
            at += take;
        }
        if (slot.rollback) |*rollback| {
            rollback.copyRows(h.model.queue, &h.model.config, &slot.cache, true, r.max_tokens -| 1) catch |e| {
                h.retire(slot, .failed, @errorName(e));
                return;
            };
            job.saved = true;
        }
        const end = h.now();
        job.prefill_seconds = if (end > prefill_start) @as(f64, @floatFromInt(end - prefill_start)) / 1e9 else null;
        h.lock();
        if (job.prefill_seconds) |seconds| h.prefill_rate = @as(f64, @floatFromInt(r.prompt.len - job.reused)) / seconds;
        h.prefill_at = end;
        h.prefilling = false;
        h.unlock();
        emit(job, .{ .prefilled = @intCast(job.reused) });
        job.prefill_sent = true;
        if (job.prefill_seconds) |seconds| std.debug.print("inkling prefill: {d:.6} seconds\n", .{seconds});
        if (r.max_tokens == 0) {
            h.retire(slot, .length, "");
            return;
        }
        if (h.cancelled(job)) {
            h.retire(slot, .cancelled, "");
            return;
        }
        const vocab = h.model.config.text.unpadded_vocab;
        h.token(slot, argmax(h.prefill.logits.slice(f32, vocab)));
    }

    fn run(h: *Host) void {
        while (true) {
            h.lock();
            var free = false;
            var active = false;
            for (h.slots) |*slot| {
                if (slot.job == null) free = true else active = true;
            }
            if (h.closing and !active and h.queued.items.len == 0) {
                h.unlock();
                return;
            }
            if (!active and h.queued.items.len == 0) {
                h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
                h.unlock();
                continue;
            }
            if (free and h.queued.items.len > 0 and !h.closing) {
                const job = h.queued.orderedRemove(0);
                const choice = prefix.chooseFree(h.slots, job.request.prompt, h.prefix_reuse).?;
                const slot = &h.slots[choice.index];
                slot.job = job;
                h.prefilling = true;
                h.unlock();
                h.admit(slot, choice);
                h.lock();
                h.prefilling = false;
                h.unlock();
                continue;
            }
            if (h.closing and h.queued.items.len > 0) {
                const job = h.queued.orderedRemove(0);
                h.unlock();
                std.debug.print("inkling prefix: reused 0 of {d} prompt tokens\n", .{job.request.prompt.len});
                h.finish(job, .cancelled, .{}, "");
                continue;
            }
            h.unlock();
            var caches: [8]*inkling.Cache = undefined;
            var ids: [8]u32 = undefined;
            var selected: [8]*Slot = undefined;
            var count: usize = 0;
            for (h.slots) |*slot| if (slot.job) |job| {
                if (h.cancelled(job)) {
                    h.retire(slot, .cancelled, "");
                } else {
                    caches[count] = &slot.cache;
                    ids[count] = job.emitted.items[job.emitted.items.len - 1];
                    selected[count] = slot;
                    count += 1;
                }
            };
            if (count == 0) continue;
            const vocab = h.model.config.text.unpadded_vocab;
            inkling.batchStep(h.model, &h.batch, caches[0..count], ids[0..count], h.batch_logits[0 .. count * vocab]) catch |e| {
                for (selected[0..count]) |slot| h.retire(slot, .failed, @errorName(e));
                continue;
            };
            for (selected[0..count], 0..) |slot, i| {
                slot.job.?.rounds += 1;
                if (h.cancelled(slot.job.?)) h.retire(slot, .cancelled, "") else h.token(slot, argmax(h.batch_logits[i * vocab ..][0..vocab]));
            }
        }
    }

};

pub fn openHost(gpa: Allocator, io: std.Io, model: *inkling.Model, capacity: usize, count: usize, eos: u32, load_limit: u64, copy_drafts: bool, prefix_reuse: bool, rollback_rows: usize) !*Host {
    _ = copy_drafts;
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    const slots = try gpa.alloc(Slot, count);
    errdefer gpa.free(slots);
    var made: usize = 0;
    errdefer for (slots[0..made]) |*slot| {
        slot.cache.deinit();
        if (slot.rollback) |*rollback| rollback.deinit();
    };
    for (slots) |*slot| {
        slot.* = .{ .cache = try inkling.Cache.init(model.device, &model.config, capacity) };
        made += 1;
        if (prefix_reuse) slot.rollback = try inkling.Cache.Rollback.initRows(model.device, &model.config, rollback_rows);
    }
    var batch = try inkling.BatchScratch.init(model.device, &model.config, count);
    errdefer batch.deinit();
    const batch_logits = try gpa.alloc(f32, count * model.config.text.unpadded_vocab);
    errdefer gpa.free(batch_logits);
    var prefill = try inkling.PrefillScratch.init(model.device, &model.config, inkling.default_chunk);
    errdefer prefill.deinit();
    const image_token: u32 = model.config.image_token;
    h.* = .{
        .gpa = gpa,
        .io = io,
        .model = model,
        .slots = slots,
        .batch = batch,
        .batch_logits = batch_logits,
        .prefill = prefill,
        .capacity = capacity,
        .eos = eos,
        .prefix_reuse = prefix_reuse,
        .info_ = .{
            .name = "inkling2-loop",
            .lanes = @intCast(count),
            .context_window = @intCast(capacity),
            .context_fitted = true,
            .greedy_only = true,
            .reasoning = true,
            .vision = .{
                .ctx = h,
                .prepare = visionPrepare,
                .image_token = image_token,
                .image_tokens = 4096,
                .max_images = 4,
                .explain = visionExplain,
            },
        },
        .load_limit = load_limit,
    };
    h.vision_markers[0] = image_token;
    h.info_.vision.?.markers = &h.vision_markers;
    h.info_.vision.?.ctx = h;
    try h.start();
    return h;
}

pub fn closeHost(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.stop();
    h.prefill.deinit();
    h.model.deinit();
    h.gpa.destroy(h);
}


test "inkling scheduler: queued jobs, finish and cancellation leave other slots running" {
    const Capture = struct {
        tokens: [4]usize = @splat(0),
        reasons: [4]?api.Reason = @splat(null),

        fn event(ctx: *anyopaque, id: api.Id, value: *const api.Event) void {
            const capture: *@This() = @ptrCast(@alignCast(ctx));
            switch (value.*) {
                .tokens => |ids| capture.tokens[@intCast(id)] += ids.len,
                .finished => |result| capture.reasons[@intCast(id)] = result.reason,
                else => {},
            }
        }
    };
    var capture = Capture{};
    const sink: api.Sink = .{ .ctx = &capture, .event = Capture.event };
    const requests = [_]api.Request{
        .{ .prompt = &.{1}, .max_tokens = 1 },
        .{ .prompt = &.{2}, .max_tokens = 3 },
        .{ .prompt = &.{3}, .max_tokens = 2 },
    };
    var slots = [_]Slot{ .{ .cache = undefined }, .{ .cache = undefined } };
    for (&slots) |*slot| slot.cache.len = 0;
    var h: Host = undefined;
    h.gpa = std.testing.allocator;
    h.io = std.testing.io;
    h.slots = &slots;
    h.eos = 42;
    h.invalidated = .init(false);
    h.mutex = .init;
    h.wake = .init;
    h.queued = .empty;
    h.cancels = .empty;
    h.decoded = .empty;
    h.closing = false;
    h.prefilling = false;
    h.prefill_at = 0;
    h.prefill_rate = 0;
    defer {
        h.queued.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        for (&slots) |*slot| slot.resident.deinit(h.gpa);
    }
    for (&requests, 1..) |*request, id| try Host.submitFn(&h, @intCast(id), request, sink);
    for (&slots) |*slot| {
        slot.job = h.queued.orderedRemove(0);
        slot.job.?.began = h.now();
    }
    var status: api.Status = undefined;
    var stream_tokens: [2]u32 = undefined;
    Host.statusFn(&h, &status, &stream_tokens);
    try std.testing.expectEqual(@as(u32, 2), status.running);
    try std.testing.expectEqual(@as(u32, 1), status.waiting);
    h.token(&slots[0], 10);
    try std.testing.expectEqual(@as(?api.Reason, .length), capture.reasons[1]);
    try std.testing.expectEqual(@as(?api.Id, 2), if (slots[1].job) |job| job.id else null);
    h.token(&slots[1], 11);
    try std.testing.expectEqual(@as(usize, 1), capture.tokens[2]);
    slots[0].job = h.queued.orderedRemove(0);
    slots[0].job.?.began = h.now();
    Host.cancelFn(&h, 2);
    try std.testing.expect(h.cancelled(slots[1].job.?));
    try std.testing.expect(!h.cancelled(slots[0].job.?));
    h.retire(&slots[1], .cancelled, "");
    try std.testing.expectEqual(@as(?api.Reason, .cancelled), capture.reasons[2]);
    h.token(&slots[0], 12);
    try std.testing.expectEqual(@as(usize, 1), capture.tokens[3]);
    try std.testing.expectEqual(@as(?api.Reason, null), capture.reasons[3]);
    h.token(&slots[0], 13);
    try std.testing.expectEqual(@as(?api.Reason, .length), capture.reasons[3]);
}
