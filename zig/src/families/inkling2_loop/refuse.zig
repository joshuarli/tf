//! Inkling 2 Mini's refusals: the checkpoint carries a vision tower, but the text engine does not read it. Image
//! content parts and `--vision` name the model in their messages so a caller knows why the input is refused.
const std = @import("std");

pub const vision = "Inkling 2 Mini does not support vision input: --vision is refused for inkling2_loop; serve it without --vision";
pub const image = "Inkling 2 Mini does not support vision input: image content parts are refused for inkling2_loop; send text parts only";
pub const media = "Inkling 2 Mini serves text only: audio, video and file content parts are refused";

pub const Part = enum { text, image, audio, video, file, tool_reference };

pub fn visionFlag(requested: bool) ?[]const u8 {
    return if (requested) vision else null;
}

pub fn contentPart(parts: []const Part) ?[]const u8 {
    for (parts) |p| switch (p) {
        .image => return image,
        .audio, .video, .file => return media,
        .text, .tool_reference => {},
    };
    return null;
}

pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.VisionUnsupported => image,
        error.VisionFlag => vision,
        else => null,
    };
}
