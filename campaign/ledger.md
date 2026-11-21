# Ledger

Every experiment and finding, including failures. Ranked optimization hypotheses go in their own table once phase 4 profiling starts.

| Date | Item | Finding | Evidence |
| --- | --- | --- | --- |
| 2026-11-21 | C0 | The baseline's `_memcpy` link check is upstream red, not a Zig 0.17.1 effect: unmodified `fa13df5d` built with the verified Zig 0.17.0 tarball (sha256 `b607e9b9…c536a`) also links its own `_memcpy` | `nm -m tensorfold-native` shows `(__TEXT,__text) non-external (was a private external) _memcpy` under both compilers |
| 2026-11-21 | F1 | `--no-drafts` already disables copy drafts and the drafter: with drafts off the lane engine never asks a proposer or verifies a copy window, on pipelined and windowed backends, alone or in shared rounds | `zig/src/core/lanes/no_drafts_test.zig`, run by `zig build test-flint-draft-host` |
| 2026-11-21 | C1 | Cold requests run on a fresh server with a 2 GiB prompt cache that is still empty, so they are genuinely cold (0 cached tokens), but Decision 8 asks for `--prompt-cache-gib 0` on cold-prefill cells; F3 should run its cold-prefill cells at 0 | `campaign/runs/2026-11-21/flint-ar-1024-092827.jsonl` |
| 2026-11-21 | Q1 | The baseline scores 12/12 on the dev subset, so the dev set cannot show a gain, only a loss; promotion uses the full 48 | `campaign/quality/results/flint-dev.json` |
| 2026-11-21 | Q1 | Pending lock entry: `xgrammar/xgrammar-v0.2.8.tar.gz`, version `97787376faee5ed8466cfad57c99855e4ce2f6aa`, sha256 `16c06f7cef8f13ae597b007cce515991725603bded6ebce5f1828e6a9c9685be`, 1,269,197 bytes, `https://github.com/mlc-ai/xgrammar/archive/refs/tags/v0.2.8.tar.gz`, Apache-2.0, `store: src`; submodules pinned by commit in `campaign/quality/xgrammar.pins.json` | Q1 worker report |
| 2026-11-21 | H1 | pi 1.5.0 creates `pi-subagents-uid-<uid>/` under TMPDIR even with `--no-extensions`, and against an unreachable endpoint prints nothing and retries for a long time; a short run with the Decision 4 flags wrote nothing under `~/.pi` | `.work/tmp/pi-repro`, `find ~/.pi -newer stamp` empty |
