# Ledger

Every experiment and finding, including failures. Ranked optimization hypotheses go in their own table once phase 4 profiling starts.

| Date | Item | Finding | Evidence |
| --- | --- | --- | --- |
| 2026-11-21 | C0 | The baseline's `_memcpy` link check fails under Zig 0.17.1; result with Zig 0.17.0 on unmodified `fa13df5d` pending | `nm -m` on the baseline binary shows a non-external `_memcpy` |
| 2026-11-21 | F1 | `--no-drafts` already disables copy drafts and the drafter: with drafts off the lane engine never asks a proposer or verifies a copy window, on pipelined and windowed backends, alone or in shared rounds | `zig/src/core/lanes/no_drafts_test.zig`, run by `zig build test-flint-draft-host` |
