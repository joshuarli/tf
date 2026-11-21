//! Inkling 2 Mini's prompt: the Inkling 2 Small package's chat-template request, rendering and tokenisation. The
//! checkpoint's template uses the same user/system/effort context and no generation prompt, so its code is imported.
const small = @import("inkling2_prompt");

pub const Request = small.Request;
pub const render = small.render;
pub const encode = small.encode;
pub const decode = small.decode;
