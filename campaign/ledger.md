# Ledger

Every experiment and finding, including failures. Ranked optimization hypotheses go in their own table once phase 4 profiling starts.

| Date | Item | Finding | Evidence |
| --- | --- | --- | --- |
| 2026-11-21 | C0 | The baseline's `_memcpy` link check is upstream red, not a Zig 0.17.1 effect: unmodified `fa13df5d` built with the verified Zig 0.17.0 tarball (sha256 `b607e9b9…c536a`) also links its own `_memcpy` | `nm -m tensorfold-native` shows `(__TEXT,__text) non-external (was a private external) _memcpy` under both compilers |
| 2026-11-21 | F1 | `--no-drafts` already disables copy drafts and the drafter: with drafts off the lane engine never asks a proposer or verifies a copy window, on pipelined and windowed backends, alone or in shared rounds | `zig/src/core/lanes/no_drafts_test.zig`, run by `zig build test-flint-draft-host` |
