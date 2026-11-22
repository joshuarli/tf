const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const cache_fit = @import("cache_fit.zig");
const Allocator = std.mem.Allocator;

const Job = struct {
    id: api.Id,
    request: *const api.Request,
    sink: api.Sink,
    emitted: std.ArrayList(u32) = .empty,
    queued: i96 = 0,
    began: i96 = 0,
    prefill_sent: bool = false,
};

pub const sampling_refusal = "Inkling 2 Mini decodes greedily only: send temperature 0 or leave it out";

fn visionPrepare(_: *anyopaque, _: Allocator, _: []const u8, _: u32) anyerror!api.PreparedImage {
    return error.VisionUnsupported;
}

fn visionExplain(err: anyerror) ?[]const u8 {
    return inkling.refuse.explain(err);
}

pub const Host = struct {
    gpa: Allocator,
    io: std.Io,
    model: *inkling.Model,
    cache: inkling.Cache,
    scratch: inkling.Scratch,
    prefill: inkling.PrefillScratch,
    capacity: usize,
    eos: u32,
    copy_drafts: bool,
    info_: api.Info,
    load_limit: u64 = 0,
    vision_markers: [1]u32 = undefined,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(api.Id) = .empty,
    running: ?*Job = null,
    closing: bool = false,
    thread: ?std.Thread = null,
    decoded: std.ArrayList(Mark) = .empty,
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_prompt: u64 = 0,
    live_generated: u64 = 0,
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
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .id = id, .request = request, .sink = sink, .queued = h.now() };
        h.lock();
        defer h.unlock();
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        var at = h.queued.items.len;
        while (!request.background and at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: api.Id) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        for (h.queued.items, 0..) |job, i| if (job.id == id) {
            _ = h.queued.orderedRemove(i);
            h.unlock();
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
        if (h.running != null and stream_tokens.len > 0) {
            stream_tokens[0] = @intCast(h.live_prompt + h.live_generated);
            generation_tokens = h.live_generated;
            n = 1;
        }
        out.* = .{
            .healthy = !h.closing,
            .running = @intFromBool(h.running != null),
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
        _ = ctx;
        if (request.images.len > 0) {
            const msg = inkling.refuse.image;
            const n = @min(msg.len, out.len);
            @memcpy(out[0..n], msg[0..n]);
            return out[0..n];
        }
        if (request.sampling != null) {
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
        h.lock();
        if (h.running == job) h.running = null;
        h.unlock();
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

    fn run(h: *Host) void {
        while (true) {
            h.lock();
            if (h.queued.items.len == 0 and !h.closing) {
                h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
                h.unlock();
                continue;
            }
            if (h.closing) {
                var left = h.queued;
                h.queued = .empty;
                h.unlock();
                for (left.items) |job| h.finish(job, .cancelled, .{}, "");
                left.deinit(h.gpa);
                return;
            }
            if (h.queued.items.len == 0) {
                h.unlock();
                continue;
            }
            const job = h.queued.orderedRemove(0);
            h.running = job;
            h.live_prompt = job.request.prompt.len;
            h.live_generated = 0;
            h.prefilling = true;
            h.cancels.clearRetainingCapacity();
            h.unlock();
            h.serve(job);
            h.lock();
            h.running = null;
            h.prefilling = false;
            h.live_generated = 0;
            h.unlock();
        }
    }

    fn cancelled(h: *Host, job: *Job) bool {
        h.lock();
        defer h.unlock();
        if (h.closing) return true;
        if (job.request.background and h.queued.items.len > 0 and !h.queued.items[0].request.background) return true;
        return std.mem.indexOfScalar(api.Id, h.cancels.items, job.id) != null;
    }

    const Fin = struct { reason: api.Reason = .failed, stats: api.Stats = .{}, message: []const u8 = "" };

    fn serve(h: *Host, job: *Job) void {
        var fin: Fin = .{};
        defer h.finish(job, fin.reason, fin.stats, fin.message);
        const r = job.request;
        job.began = h.now();
        if (r.images.len > 0) {
            emit(job, .{ .prefilled = 0 });
            job.prefill_sent = true;
            fin.message = inkling.refuse.image;
            return;
        }
        if (r.sampling) |s| {
            if (s.temperature > 0) {
                emit(job, .{ .prefilled = 0 });
                job.prefill_sent = true;
                fin.message = sampling_refusal;
                return;
            }
        }
        if (r.prompt.len == 0) {
            emit(job, .{ .prefilled = 0 });
            job.prefill_sent = true;
            fin.message = "rendered prompt is empty";
            return;
        }
        if (r.prompt.len + r.max_tokens > h.capacity) {
            emit(job, .{ .prefilled = 0 });
            job.prefill_sent = true;
            fin.message = "the prompt and max_tokens do not fit this server's --context";
            return;
        }
        h.cache.len = 0;
        const vocab: usize = h.model.config.text.unpadded_vocab;
        const chunk: usize = h.prefill.chunk;
        var at: usize = 0;
        while (at < r.prompt.len) {
            if (h.cancelled(job)) {
                if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
                fin.reason = .cancelled;
                h.cache.len = 0;
                return;
            }
            const take = @min(chunk, r.prompt.len - at);
            const last = at + take >= r.prompt.len;
            inkling.prefillWant(h.model, &h.prefill, &h.cache, r.prompt[at..][0..take], if (last) .last else .none) catch |e| {
                if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
                fin.message = @errorName(e);
                h.cache.len = 0;
                return;
            };
            at += take;
        }
        @memcpy(h.scratch.logits.slice(f32, vocab), h.prefill.logits.slice(f32, vocab));
        const done_prefill = h.now();
        h.lock();
        if (done_prefill > job.began) h.prefill_rate = @as(f64, @floatFromInt(r.prompt.len)) / (@as(f64, @floatFromInt(done_prefill - job.began)) / 1e9);
        h.prefill_at = done_prefill;
        h.prefilling = false;
        h.unlock();
        emit(job, .{ .prefilled = 0 });
        job.prefill_sent = true;
        const prefill_s: ?f64 = if (done_prefill > job.began) @as(f64, @floatFromInt(done_prefill - job.began)) / 1e9 else null;
        const queued_s: f64 = @as(f64, @floatFromInt(@max(0, job.began - job.queued))) / 1e9;
        if (r.max_tokens == 0) {
            fin.reason = .length;
            fin.stats = .{ .rounds = 0, .drafted = 0, .accepted = 0, .min_rows = 1, .prefill_seconds = prefill_s, .queued_seconds = queued_s };
            h.cache.len = 0;
            return;
        }
        var proposer = tf.lanes.SuffixLookup.init(h.gpa, .{}) catch {
            fin.message = @errorName(error.OutOfMemory);
            return;
        };
        defer proposer.deinit();
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(h.gpa);
        if (h.copy_drafts and r.drafts) {
            history.appendSlice(h.gpa, r.prompt) catch {
                fin.message = @errorName(error.OutOfMemory);
                return;
            };
        }
        var pending: [16]u32 = undefined;
        var pending_at: usize = 0;
        var pending_len: usize = 0;
        var rounds: u64 = 0;
        var drafted: u64 = 0;
        var accepted_total: u64 = 0;
        while (job.emitted.items.len < r.max_tokens) {
            if (h.cancelled(job)) {
                fin.reason = .cancelled;
                fin.stats = .{ .rounds = rounds, .drafted = drafted, .accepted = accepted_total, .min_rows = 1, .prefill_seconds = prefill_s, .queued_seconds = queued_s };
                h.cache.len = 0;
                return;
            }
            const logits = h.scratch.logits.slice(f32, vocab);
            var best: u32 = 0;
            var best_v = logits[0];
            for (logits[1..], 1..) |v, i| {
                if (v > best_v) {
                    best_v = v;
                    best = @intCast(i);
                }
            }
            if (pending_at < pending_len) {
                best = pending[pending_at];
                pending_at += 1;
            }
            job.emitted.append(h.gpa, best) catch {
                fin.message = @errorName(error.OutOfMemory);
                h.cache.len = 0;
                return;
            };
            if (h.copy_drafts and r.drafts) {
                history.append(h.gpa, best) catch {
                    fin.message = @errorName(error.OutOfMemory);
                    h.cache.len = 0;
                    return;
                };
            }
            var one: [1]u32 = .{best};
            emit(job, .{ .tokens = &one });
            h.lock();
            h.live_generated = @intCast(job.emitted.items.len);
            h.unlock();
            h.noteDecoded(1);
            var stop_hit = false;
            if (r.stop) |s| {
                if (s.check(s.ctx, job.emitted.items)) stop_hit = true;
            }
            const is_eos = r.eos.len > 0 and std.mem.indexOfScalar(u32, r.eos, best) != null;
            const hit_gen_eos = r.eos.len > 0 and best == h.eos;
            if (stop_hit or is_eos or hit_gen_eos) {
                fin.reason = .stop;
                fin.stats = .{ .rounds = rounds, .drafted = drafted, .accepted = accepted_total, .min_rows = 1, .prefill_seconds = prefill_s, .queued_seconds = queued_s };
                h.cache.len = 0;
                return;
            }
            if (job.emitted.items.len >= r.max_tokens) {
                fin.reason = .length;
                fin.stats = .{ .rounds = rounds, .drafted = drafted, .accepted = accepted_total, .min_rows = 1, .prefill_seconds = prefill_s, .queued_seconds = queued_s };
                h.cache.len = 0;
                return;
            }
            if (pending_at < pending_len) continue;
            if (h.copy_drafts and r.drafts) {
                const draft = proposer.propose(history.items, @intCast(inkling.hybrid_drafts)) catch |e| {
                    fin.message = @errorName(e);
                    return;
                };
                var rows: [16]u32 = undefined;
                rows[0] = best;
                const count: usize = @min(draft.len, @min(inkling.hybrid_drafts, r.max_tokens - job.emitted.items.len - 1));
                const row_count = std.math.add(usize, count, 1) catch {
                    fin.message = @errorName(error.Overflow);
                    return;
                };
                if (count > 0) {
                    @memcpy(rows[1..][0..count], draft[0..count]);
                    var predictions: [16]u32 = undefined;
                    var verification = inkling.Verification.begin(h.model, &h.prefill, &h.cache, rows[0..row_count], predictions[0..row_count]) catch |e| {
                        fin.message = @errorName(e);
                        return;
                    };
                    var accepted: usize = 0;
                    while (accepted < count and predictions[accepted] == rows[accepted + 1]) : (accepted += 1) {}
                    const kept = std.math.add(usize, accepted, 1) catch {
                        fin.message = @errorName(error.Overflow);
                        return;
                    };
                    verification.finish(rows[0..row_count], kept) catch |e| {
                        fin.message = @errorName(e);
                        return;
                    };
                    proposer.observe(@intCast(count), @intCast(accepted)) catch |e| {
                        fin.message = @errorName(e);
                        return;
                    };
                    rounds += 1;
                    drafted += count;
                    accepted_total += accepted;
                    pending_len = kept;
                    pending_at = 0;
                    @memcpy(pending[0..pending_len], predictions[0..pending_len]);
                    continue;
                }
            }
            inkling.position(h.model, &h.scratch, &h.cache, best) catch |e| {
                fin.message = @errorName(e);
                h.cache.len = 0;
                return;
            };
            rounds += 1;
        }
        fin.reason = .length;
        fin.stats = .{ .rounds = rounds, .drafted = drafted, .accepted = accepted_total, .min_rows = 1, .prefill_seconds = prefill_s, .queued_seconds = queued_s };
        h.cache.len = 0;
    }
};

pub fn cacheBytesPerToken(model: *const inkling.Model) usize {
    const c = &model.config;
    const kv: usize = c.text.local.kv * c.text.local.dim;
    const slots: usize = c.slots();
    return slots * kv * 4 * 2;
}

pub fn fixedCacheBytes(model: *const inkling.Model) usize {
    const c = &model.config;
    const kv: usize = c.text.local.kv * c.text.local.dim;
    const h: usize = c.text.hidden;
    const slots: usize = c.slots();
    const ring: usize = 4;
    return slots * ring * kv * 4 * 2 + slots * ring * h * 4 * 2;
}

pub fn longestFitting(model: *const inkling.Model, budget: u64, most: usize) usize {
    const per = cacheBytesPerToken(model);
    const fixed = fixedCacheBytes(model);
    if (per == 0 or budget <= fixed) return 0;
    const avail = budget - fixed;
    return @min(most, avail / per);
}

pub fn openHost(gpa: Allocator, io: std.Io, model: *inkling.Model, capacity: usize, eos: u32, load_limit: u64, copy_drafts: bool) !*Host {
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    var cache = try inkling.Cache.init(model.device, &model.config, capacity);
    errdefer cache.deinit();
    var scratch = try inkling.Scratch.init(model.device, &model.config);
    errdefer scratch.deinit();
    var prefill = try inkling.PrefillScratch.init(model.device, &model.config, inkling.default_chunk);
    errdefer prefill.deinit();
    const image_token: u32 = model.config.image_token;
    h.* = .{
        .gpa = gpa,
        .io = io,
        .model = model,
        .cache = cache,
        .scratch = scratch,
        .prefill = prefill,
        .capacity = capacity,
        .eos = eos,
        .copy_drafts = copy_drafts,
        .info_ = .{
            .name = "inkling2-loop",
            .lanes = 1,
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
    h.cache.deinit();
    h.scratch.deinit();
    h.prefill.deinit();
    h.model.deinit();
    h.gpa.destroy(h);
}

test "inkling host: image requests are refused by name without touching the GPU" {
    var out: [api.refusal_bytes]u8 = undefined;
    var host: Host = undefined;
    host.eos = 200006;
    const engine: api.Engine = .{ .ctx = &host, .vtable = &Host.vtable };
    const prompt = [_]u32{ 1, 2, 3 };
    const images = [_]api.Image{.{
        .pixels = &.{},
        .gh = 1,
        .gw = 1,
        .at = 0,
        .tokens = 1,
    }};
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4, .images = &images };
    const msg = engine.refusal(&request, &out).?;
    try std.testing.expectEqualStrings(inkling.refuse.image, msg);
}

test "inkling host: sampling beyond greedy is refused by name" {
    var out: [api.refusal_bytes]u8 = undefined;
    var host: Host = undefined;
    host.eos = 200006;
    const engine: api.Engine = .{ .ctx = &host, .vtable = &Host.vtable };
    const prompt = [_]u32{ 1, 2, 3 };
    const sampling: api.Sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 0, .top_p = 1, .min_p = 0 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4, .sampling = sampling };
    const msg = engine.refusal(&request, &out).?;
    try std.testing.expectEqualStrings(sampling_refusal, msg);
}

test "inkling host: greedy requests with no images are accepted" {
    var out: [api.refusal_bytes]u8 = undefined;
    var host: Host = undefined;
    host.eos = 200006;
    const engine: api.Engine = .{ .ctx = &host, .vtable = &Host.vtable };
    const prompt = [_]u32{ 1, 2, 3 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try std.testing.expectEqual(@as(?[]const u8, null), engine.refusal(&request, &out));
}
