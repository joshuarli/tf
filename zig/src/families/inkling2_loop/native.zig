//! Inkling 2 Mini on Metal: the family's model, its per-token forward and the adapter the native engine opens.
//! Vision is out of scope, so image parts and `--vision` are refused before any weight is read.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const cfg = @import("config.zig");
const model = @import("model.zig");
const state = @import("state.zig");
const forward = @import("forward.zig");
const refuse = @import("refuse.zig");

pub const Model = model.Model;
pub const Cache = state.Cache;
pub const Scratch = state.Scratch;
pub const position = forward.position;

/// Which family's checkpoint this is: the native registry's entry.
pub const model_type = "inkling2_loop";

/// Every quantization the family reads, per tensor.
pub const formats: []const []const u8 = &.{ "nvfp4", "mxfp8" };

/// The server-side adapter: refused features first, then the model.
pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.vision != null) {
        problem.* = refuse.vision;
        return null;
    }
    if (try refuse.contentParts(a, o)) |why| {
        problem.* = why;
        return null;
    }
    _ = mtl.objc.Pool.push();
    defer mtl.objc.Pool.pop();
    const m = Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Inkling 2 Mini engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    m.deinit();
    problem.* = "the native Inkling 2 Mini engine opens for teacher-forced logits only; serving it is a later item";
    return null;
}

test {
    _ = @import("config.zig");
    _ = @import("model.zig");
    _ = @import("state.zig");
    _ = @import("forward.zig");
    _ = @import("kernels.zig");
    _ = refuse;
    _ = cfg;
}
