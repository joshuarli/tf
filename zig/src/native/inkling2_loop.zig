//! Inkling 2 Mini on Metal through the native registry. The lane host, streaming and reset are a later item; this is
//! the open() the registry reaches, its refusals, and the teacher-forced forward the family exports.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;

pub const model_type = inkling.model_type;
pub const formats = inkling.formats;

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.vision != null) {
        problem.* = inkling.refuse.vision;
        return null;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = inkling.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    m.deinit();
    problem.* = "the native Inkling 2 Mini engine opens for teacher-forced logits only; serving it is a later item";
    return null;
}

test "the registry's open refuses --vision by name before any weight is read" {
    var problem: []const u8 = "";
    const o = api.Open{ .dir = "no-weights", .model_type = "inkling2_loop", .vision = .{ .dir = "no-weights" } };
    try std.testing.expectEqual(null, try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem));
    try std.testing.expectEqualStrings(tf.inkling2_loop_refuse.vision, problem);
}
