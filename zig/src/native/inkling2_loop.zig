//! Inkling 2 Mini on Metal served through the native registry. The lane host, streaming and reset are a later item;
//! this exposes the open() the registry reaches and the refusals that must fire before any weight is read.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const inkling = tf.inkling2_loop;

pub const model_type = inkling.model_type;
pub const formats = inkling.formats;

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    return inkling.open(a, gpa, io, o, problem);
}

test "the family's open is reached through the registry's refusal path" {
    var problem: []const u8 = "";
    const o = api.Open{ .dir = "no-weights", .model_type = "inkling2_loop", .vision = .auto };
    try std.testing.expectEqual(null, try open(std.testing.allocator, std.testing.allocator, std.testing.io, o, &problem));
    try std.testing.expectEqualStrings(tf.inkling2_loop_refuse.vision, problem);
}
