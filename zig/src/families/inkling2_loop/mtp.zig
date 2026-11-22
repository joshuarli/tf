const config = @import("config.zig");

pub const Capture = union(enum) {
    entering: struct { layer: u32, pass: u32 },
    final_normalized,
};

pub fn capture(loop: config.Loop, layers: u32, tap: u32) !Capture {
    const slots = loop.slotCount(layers);
    if (tap > slots) return error.InvalidTap;
    if (tap == slots) return .final_normalized;
    if (tap < loop.core_start) return .{ .entering = .{ .layer = tap, .pass = 0 } };
    const width = loop.core_end - loop.core_start;
    const offset = tap - loop.core_start;
    if (offset < width * loop.passes) return .{ .entering = .{
        .layer = loop.core_start + offset % width,
        .pass = offset / width,
    } };
    return .{ .entering = .{ .layer = loop.core_end + offset - width * loop.passes, .pass = 0 } };
}

pub fn proposalCount(mtp: config.Mtp) u32 {
    return mtp.block - 1;
}
