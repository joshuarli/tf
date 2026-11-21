//! Inkling 2 Mini's Metal kernel sources, compiled when a model opens.
//!
//! `glue` is the dense graph: one token at a time, f32 activations, the checkpoint's bf16 weights widened per read.
//! `experts` runs the routed projections in the two stored formats, decoding the checkpoint's own bytes. Both mirror
//! `campaign/ref/inkling2_loop_ref.py`, a transcription of llama.cpp `40b7a546`.
const std = @import("std");

pub const glue = @embedFile("kernels/glue.metal");

/// The routed experts' source for one stored format: MXFP8 defines its decoder, NVFP4 is the default.
pub fn expertSource(comptime mxfp8: bool) []const u8 {
    const text = @embedFile("kernels/experts.metal");
    return if (mxfp8) "#define I2_MXFP8 1\n" ++ text else text;
}

test "the sources name the kernels the model loads" {
    try std.testing.expect(std.mem.indexOf(u8, glue, "i2_attn") != null);
    try std.testing.expect(std.mem.indexOf(u8, glue, "i2_conv_store") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(false), "I2_MXFP8 1") == null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(true), "I2_MXFP8 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(false), "i2_gu") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(false), "i2_down") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(false), "i2_gu_table") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(false), "i2_down_table") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(true), "i2_gu_table") != null);
    try std.testing.expect(std.mem.indexOf(u8, expertSource(true), "i2_down_thread") != null);
}
