//! Host-only Inkling 2 Mini (`inkling2_loop`) package: config, quant geometry, weight map, prompt and refusals.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("mtp_test.zig");
    _ = @import("quant.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("prompt.zig");
    _ = @import("prompt_test.zig");
    _ = @import("refuse.zig");
    _ = @import("refuse_test.zig");
}
