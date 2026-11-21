//! Inkling 2 Mini's Metal family without the engine's request surface: the model, its per-token forward, the
//! config it reads and the refusals. The native registry's adapter (`zig/src/native/inkling2_loop.zig`) adds the
//! parts that name `engine_api`, which the engine module graph does not carry.
pub const config = @import("config.zig");
pub const Model = @import("model.zig").Model;
pub const Cache = @import("state.zig").Cache;
pub const Scratch = @import("state.zig").Scratch;
pub const Trace = @import("state.zig").Trace;
pub const position = @import("forward.zig").position;
pub const positionTrace = @import("forward.zig").positionTrace;
pub const advance = @import("forward.zig").advance;
pub const Output = @import("forward.zig").Output;
pub const refuse = @import("refuse.zig");
pub const kernels = @import("kernels.zig");

pub const model_type = "inkling2_loop";
pub const formats: []const []const u8 = &.{ "nvfp4", "mxfp8" };

test {
    _ = @import("config.zig");
    _ = @import("model.zig");
    _ = @import("state.zig");
    _ = @import("forward.zig");
    _ = @import("kernels.zig");
    _ = @import("refuse.zig");
}
