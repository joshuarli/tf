//! Inkling 2 Mini's Metal family without the engine's request surface: the model, its per-token forward, the
//! config it reads and the refusals. The native registry's adapter (`zig/src/native/inkling2_loop.zig`) adds the
//! parts that name `engine_api`, which the engine module graph does not carry.
pub const config = @import("config.zig");
pub const Model = @import("model.zig").Model;
pub const Tensor = @import("model.zig").Tensor;
pub const Cache = @import("state.zig").Cache;
pub const Scratch = @import("state.zig").Scratch;
pub const BatchScratch = @import("batch.zig").BatchScratch;
pub const batchStep = @import("batch.zig").step;
pub const Trace = @import("state.zig").Trace;
pub const position = @import("forward.zig").position;
pub const positionTrace = @import("forward.zig").positionTrace;
pub const profile = @import("prefill.zig").profile;
pub const prefill = @import("prefill.zig").prefill;
pub const prefillWant = @import("prefill.zig").prefillWant;
pub const WantLogits = @import("prefill.zig").WantLogits;
pub const PrefillScratch = @import("prefill.zig").PrefillScratch;
pub const default_chunk = @import("prefill.zig").default_chunk;
pub const Verification = @import("verify.zig").Verification;
pub const verify = @import("verify.zig");
pub const hybrid_drafts = @import("verify.zig").hybrid_drafts;
pub const group = @import("group.zig");
pub const Taps = @import("taps.zig").Taps;
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
    _ = @import("prefill.zig");
    _ = @import("prefill_test.zig");
    _ = @import("units_test.zig");
    _ = @import("units_attention_test.zig");
    _ = @import("verify_test.zig");
    _ = @import("batch_test.zig");
    _ = @import("taps.zig");
    _ = @import("group.zig");
    _ = @import("group_test.zig");
}
