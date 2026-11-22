const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;
const host_mod = @import("inkling2_loop_host.zig");

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
        problem.* = "Inkling 2 Mini has no served drafter in this campaign (its mtp.safetensors head is not wired); run mode ar";
        return null;
    }
    const env: ?[]const u8 = if (std.c.getenv("INKLING_COPY_DRAFTS")) |v| std.mem.span(v) else null;
    const copy_drafts = o.drafts and (copyDrafts(env) catch {
        problem.* = "INKLING_COPY_DRAFTS must be 0 or 1";
        return null;
    });
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = inkling.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer m.deinit();
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
    const budget = if (limit > resident + load_margin) limit - resident - load_margin else 0;
    const most: usize = m.config.text.max_context;
    const fits = inkling.Cache.longestFitting(&m.config, budget, most);
    const window: i64 = o.context orelse @intCast(fits);
    if (window < 1 or window > fits) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: this Mac's Inkling caches fit {d} tokens under its load limit (75% of RAM, or INKLING_LOAD_LIMIT_GB): {d} global slots a token plus {d} local rings of {d}", .{ window, fits, inkling.Cache.globalSlots(&m.config), inkling.Cache.localSlots(&m.config), m.config.text.window });
        m.deinit();
        return null;
    }
    std.log.info("inkling2_loop: {d:.1} GiB for caches ({d:.1} GiB in {d} local rings, {d} KiB a token over {d} global slots), context up to {d} tokens", .{ @as(f64, @floatFromInt(budget)) / (1 << 30), @as(f64, @floatFromInt(inkling.Cache.fixedBytes(&m.config))) / (1 << 30), inkling.Cache.localSlots(&m.config), inkling.Cache.bytesPerToken(&m.config) >> 10, inkling.Cache.globalSlots(&m.config), window });
    const eos = readEos(a, io, o.dir, m.config.eos);
    const h = host_mod.openHost(gpa, io, m, @intCast(window), eos, limit, copy_drafts) catch |err| {
        m.deinit();
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot hold {d} tokens of cache ({s})", .{ window, @errorName(err) });
        return null;
    };
    return .{ .engine = h.engine(), .close = host_mod.closeHost, .ctx = h };
}

test "the registry's open refuses --vision by name before any weight is read" {
    var problem: []const u8 = "";
    const o = api.Open{ .dir = "no-weights", .model_type = "inkling2_loop", .vision = .{ .dir = "no-weights" } };
    try std.testing.expectEqual(null, try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem));
    try std.testing.expectEqualStrings(tf.inkling2_loop_refuse.vision, problem);
}

test "the registry's open refuses a drafter by name before any weight is read" {
    var problem: []const u8 = "";
    const o = api.Open{ .dir = "no-weights", .model_type = "inkling2_loop", .drafter = "no-head" };
    try std.testing.expectEqual(null, try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "no served drafter") != null);
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
