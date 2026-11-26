const std = @import("std");

pub fn main(init: std.process.Init) !void {
    return @import("aot/aotc.zig").main(init);
}
