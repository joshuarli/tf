const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const host_mod = @import("inkling2_loop_host.zig");
const spec = @import("inkling2_loop_spec.zig");

pub const model_type = inkling.model_type;
pub const formats = inkling.formats;

const load_margin: u64 = 2 << 30;
const scratch_reserve: u64 = 1 << 30;

fn copyDrafts(text: ?[]const u8) error{BadCopyDrafts}!bool {
    const value = text orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.BadCopyDrafts;
}

fn drafterBits(text: ?[]const u8) error{BadDrafterBits}!u8 {
    const value = text orelse return 4;
    if (std.mem.eql(u8, value, "4")) return 4;
    if (std.mem.eql(u8, value, "0")) return 0;
    return error.BadDrafterBits;
}

fn prefixReuse(text: ?[]const u8) error{BadPrefixReuse}!bool {
    const value = text orelse return true;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.BadPrefixReuse;
}

fn rollbackRows(text: ?[]const u8) error{BadRollbackRows}!usize {
    const value = text orelse return 512;
    const n = std.fmt.parseInt(usize, value, 10) catch return error.BadRollbackRows;
    if (n == 0) return error.BadRollbackRows;
    return n;
}

fn loadLimit(ram: u64, working_set: u64, text: ?[]const u8) error{BadLoadLimit}!u64 {
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len == 0) return @as(u64, @intFromFloat(@as(f64, @floatFromInt(ram)) * 0.75));
    const gb = std.fmt.parseFloat(f64, t) catch return error.BadLoadLimit;
    if (!std.math.isFinite(gb) or gb <= 0 or gb * 1e9 > @as(f64, @floatFromInt(ram))) return error.BadLoadLimit;
    return @min(@as(u64, @intFromFloat(gb * 1e9)), @min(ram, working_set));
}

fn readEos(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, fallback: u32) u32 {
    const path = std.fs.path.join(gpa, &.{ dir, "generation_config.json" }) catch return fallback;
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch return fallback;
    defer gpa.free(bytes);
    const doc = std.json.parseFromSliceLeaky(std.json.Value, gpa, bytes, .{}) catch return fallback;
    if (doc != .object) return fallback;
    const v = doc.object.get("eos_token_id") orelse return fallback;
    if (v == .integer and v.integer >= 0 and v.integer <= std.math.maxInt(u32)) return @intCast(v.integer);
    return fallback;
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.vision != null) {
        problem.* = inkling.refuse.vision;
        return null;
    }
    if (o.drafts and o.drafter != null) {
        problem.* = "Inkling 2 Mini serves its DFlash2 drafter from INKLING_DRAFTER, not --drafter";
        return null;
    }
    const drafter_env: ?[]const u8 = if (std.c.getenv("INKLING_DRAFTER")) |v| std.mem.span(v) else null;
    const drafter: ?[]const u8 = if (drafter_env) |dir| (if (o.drafts and dir.len > 0) dir else null) else null;
    const bits_env: ?[]const u8 = if (std.c.getenv("INKLING_DRAFTER_BITS")) |v| std.mem.span(v) else null;
    const bits = drafterBits(bits_env) catch {
        problem.* = "INKLING_DRAFTER_BITS must be 4 (prepared q4) or 0 (bf16)";
        return null;
    };
    const profile_env: ?[]const u8 = if (std.c.getenv("INKLING_PROFILE")) |v| std.mem.span(v) else null;
    _ = inkling.profile(profile_env) catch {
        problem.* = "INKLING_PROFILE must be f32 or fast";
        return null;
    };
    const env: ?[]const u8 = if (std.c.getenv("INKLING_COPY_DRAFTS")) |v| std.mem.span(v) else null;
    const copy_drafts = o.drafts and (copyDrafts(env) catch {
        problem.* = "INKLING_COPY_DRAFTS must be 0 or 1";
        return null;
    });
    const prefix_env: ?[]const u8 = if (std.c.getenv("INKLING_PREFIX_REUSE")) |v| std.mem.span(v) else null;
    const prefix_reuse = prefixReuse(prefix_env) catch {
        problem.* = "INKLING_PREFIX_REUSE must be 0 or 1";
        return null;
    };
    const rollback_env: ?[]const u8 = if (std.c.getenv("INKLING_ROLLBACK_ROWS")) |v| std.mem.span(v) else null;
    const rollback_rows = rollbackRows(rollback_env) catch {
        problem.* = "INKLING_ROLLBACK_ROWS must be a positive integer";
        return null;
    };
    const fork_window = rollbackRows(if (std.c.getenv("INKLING_FORK_WINDOW")) |v| std.mem.span(v) else "256") catch {
        problem.* = "INKLING_FORK_WINDOW must be a positive integer";
        return null;
    };
    const fork_interval = rollbackRows(if (std.c.getenv("INKLING_FORK_INTERVAL")) |v| std.mem.span(v) else "128") catch {
        problem.* = "INKLING_FORK_INTERVAL must be a positive integer";
        return null;
    };
    if (fork_window > 1024 or fork_interval > fork_window or (fork_window - 1) / fork_interval >= 8) {
        problem.* = "INKLING_FORK_WINDOW must be at most 1024 and contain one to eight fork intervals";
        return null;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = inkling.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer m.deinit();
    if (fork_window > m.config.text.window) {
        m.deinit();
        problem.* = "INKLING_FORK_WINDOW exceeds the model's local attention window";
        return null;
    }
    var ram: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) ram = 0;
    const text: ?[]const u8 = if (std.c.getenv("INKLING_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const limit = loadLimit(ram, m.device.maxWorkingSet(), text) catch {
        m.deinit();
        problem.* = "INKLING_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0 and at most this Mac's RAM, or unset it for 75% of RAM";
        return null;
    };
    const resident = blk: {
        const file = inkling.config.shard(gpa, io, o.dir) catch break :blk m.device.allocated() + scratch_reserve;
        defer gpa.free(file);
        const full = std.fs.path.join(gpa, &.{ o.dir, file }) catch break :blk m.device.allocated() + scratch_reserve;
        defer gpa.free(full);
        const st = std.Io.Dir.cwd().statFile(io, full, .{}) catch break :blk m.device.allocated() + scratch_reserve;
        break :blk st.size + scratch_reserve;
    };
    const count: usize = if (o.lanes_fixed) o.lanes else 1;
    if (count == 0 or count > 8) {
        m.deinit();
        problem.* = "--parallel must be between 1 and 8";
        return null;
    }
    const draft: ?*spec.Draft = if (drafter) |dir| spec.Draft.open(gpa, io, m, dir, bits, count, inkling.default_chunk) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "INKLING_DRAFTER: the DFlash2 drafter in {s} cannot serve this checkpoint ({s})", .{ dir, @errorName(err) });
        m.deinit();
        return null;
    } else null;
    errdefer if (draft) |d| d.deinit();
    const drafted: u64 = if (draft) |d| try d.bytes() else 0;
    const rollback: u64 = if (prefix_reuse) count *| inkling.Cache.rollbackBytes(&m.config, rollback_rows) else 0;
    const forks: u64 = if (prefix_reuse) count *| ((fork_window + fork_interval - 1) / fork_interval) *| inkling.Cache.rollbackBytes(&m.config, fork_window) else 0;
    const gib = o.prompt_cache_gib orelse 2;
    if (!std.math.isFinite(gib) or gib < 0 or gib > 1024) {
        if (draft) |d| d.deinit();
        m.deinit();
        problem.* = "--prompt-cache-gib must be between 0 and 1024";
        return null;
    }
    const store_budget: u64 = if (prefix_reuse) @intFromFloat(gib * (1 << 30)) else 0;
    const budget = if (limit > resident +| load_margin +| rollback +| forks +| store_budget +| drafted) limit - resident - load_margin - rollback - forks - store_budget - drafted else 0;
    const most: usize = m.config.text.max_context;
    const fits = inkling.Cache.longestFitting(&m.config, budget / count, most);
    const window: i64 = o.context orelse @intCast(fits);
    if (window < 1 or window > fits) {
        const requested: u64 = if (window > 0) @intCast(window) else 0;
        const total = resident +| load_margin +| rollback +| forks +| store_budget +| drafted +| (count *| (inkling.Cache.fixedBytes(&m.config) +| (requested *| inkling.Cache.bytesPerToken(&m.config))));
        problem.* = try std.fmt.allocPrint(a, "--parallel {d} --context {d}: Inkling needs {d} bytes including caches and reserves under a {d}-byte load limit; each session fits at most {d} tokens", .{ count, window, total, limit, fits });
        if (draft) |d| d.deinit();
        m.deinit();
        return null;
    }
    const per_slot = inkling.Cache.fixedBytes(&m.config) + @as(usize, @intCast(window)) * inkling.Cache.bytesPerToken(&m.config);
    std.log.info("inkling2_loop: {d} session caches, {d:.3} GiB per slot, {d:.3} GiB total rollback, context {d}", .{ count, @as(f64, @floatFromInt(per_slot)) / (1 << 30), @as(f64, @floatFromInt(rollback)) / (1 << 30), window });
    if (draft != null) std.log.info("inkling2_loop: DFlash2 drafter {s}, {s}, {d:.3} GiB with {d} lane contexts; greedy speculation while one lane decodes", .{ drafter.?, if (bits == 4) "prepared q4" else "bf16", @as(f64, @floatFromInt(drafted)) / (1 << 30), count });
    const eos = readEos(a, io, o.dir, m.config.eos);
    const h = host_mod.openHost(gpa, io, m, @intCast(window), count, eos, limit, copy_drafts, prefix_reuse, rollback_rows, fork_window, fork_interval, store_budget, draft) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot hold {d} sessions of {d} tokens ({s})", .{ count, window, @errorName(err) });
        if (draft) |d| d.deinit();
        m.deinit();
        return null;
    };
    return .{ .engine = h.engine(), .close = host_mod.closeHost, .ctx = h };
}

test "the registry's open refuses --vision by name before any weight is read" {
    var problem: []const u8 = "";
    const o = api.Open{ .dir = "no-weights", .model_type = "inkling2_loop", .vision = .{} };
    try std.testing.expectEqual(null, try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem));
    try std.testing.expectEqualStrings(tf.inkling2_loop_refuse.vision, problem);
}

test "the registry's open refuses --drafter by name before any weight is read" {
    var problem: []const u8 = "";
    const o = api.Open{ .dir = "no-weights", .model_type = "inkling2_loop", .drafter = "no-head" };
    try std.testing.expectEqual(null, try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "INKLING_DRAFTER") != null);
}

test "drafter bits default to prepared q4 and take only 4 or 0" {
    try std.testing.expectEqual(@as(u8, 4), try drafterBits(null));
    try std.testing.expectEqual(@as(u8, 4), try drafterBits("4"));
    try std.testing.expectEqual(@as(u8, 0), try drafterBits("0"));
    try std.testing.expectError(error.BadDrafterBits, drafterBits("8"));
}

test "a bad INKLING_LOAD_LIMIT_GB names the variable" {
    try std.testing.expectError(error.BadLoadLimit, loadLimit(64 << 30, 64 << 30, "lots"));
    try std.testing.expectError(error.BadLoadLimit, loadLimit(64 << 30, 64 << 30, "900"));
}

test "copy drafts require an explicit valid opt-in" {
    try std.testing.expect(!try copyDrafts(null));
    try std.testing.expect(!try copyDrafts("0"));
    try std.testing.expect(try copyDrafts("1"));
    try std.testing.expectError(error.BadCopyDrafts, copyDrafts("yes"));
}

test "prefix reuse defaults on and requires a valid value" {
    try std.testing.expect(try prefixReuse(null));
    try std.testing.expect(!try prefixReuse("0"));
    try std.testing.expect(try prefixReuse("1"));
    try std.testing.expectError(error.BadPrefixReuse, prefixReuse("yes"));
}

test "inkling host: image requests are refused by name without touching the GPU" {
    var out: [api.refusal_bytes]u8 = undefined;
    var host: host_mod.Host = undefined;
    host.eos = 200006;
    host.invalidated = .init(false);
    const engine = host.engine();
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
    var host: host_mod.Host = undefined;
    host.eos = 200006;
    host.invalidated = .init(false);
    const engine = host.engine();
    const prompt = [_]u32{ 1, 2, 3 };
    const sampling: api.Sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 0, .top_p = 1, .min_p = 0 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4, .sampling = sampling };
    const msg = engine.refusal(&request, &out).?;
    try std.testing.expectEqualStrings(host_mod.sampling_refusal, msg);
}

test "inkling host: greedy requests with no images are accepted" {
    var out: [api.refusal_bytes]u8 = undefined;
    var host: host_mod.Host = undefined;
    host.eos = 200006;
    host.invalidated = .init(false);
    const engine = host.engine();
    const prompt = [_]u32{ 1, 2, 3 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try std.testing.expectEqual(@as(?[]const u8, null), engine.refusal(&request, &out));
}
