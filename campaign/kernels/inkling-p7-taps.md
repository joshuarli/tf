# P7: Drafter Prompt-Side Tap Capture

## Summary

Replaced the per-chunk flat tap staging buffer (`d.prompt`, `18.87 MB`) and CPU `@memcpy` absorb loop in `inkling2_loop_host.zig` with:

1. Direct-to-ring GPU tap capture (`i2t_tap_ring` in `zig/src/families/inkling2_loop/kernels/taps.metal`, wired via `Taps.RingWrite` in `taps.zig` and `PrefillScratch.taps_ring` in `prefill.zig`).
2. Selective prompt-window capture (`prefix.tapNeedFrom` in `inkling2_loop_prefix.zig`): chunks strictly before `keep_at -| (window - 1)` skip both `i2t_tap_ring` and the unused final `mixBlocks + rmsW` head pass.
3. Wraparound-safe fork-point stashing (`Draft.stashOverwritten` / `Draft.keep`) so storing the first fork point (`saved == 0 and job.reused == 0`) inside a chunk that wraps past `window = 2,048` preserves every overwritten ring row bit-identically.
4. Unlocked P6's `last_chunk_rows = 1,152` last-chunk merge whenever the drafter is loaded (`@min(last_chunk_rows, d.windowRows())` instead of `@min(last_chunk_rows, d.prompt.rows)`).

## Correctness

- `zig build test-inkling2-loop-host --summary all`: `25/25` passed (including `tapNeedFrom` unit tests).
- `zig build test-inkling2-loop-filtered -Dinkling-filter="ring tap capture" --summary all`: `5/5` passed (proving `i2t_tap_ring` matches `i2t_tap` bit-identically on skipped/wrapped rows without touching unwritten slots).
- `TF_INKLING2_LOOP_OQ4E_DIR=... TF_INKLING2_DFLASH2_DIR=... INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- zig build test-inkling2-loop-served --summary all`: `13/13` passed (`10` drafted replies equal draft-off, `4` equal a host without the drafter, across cold, prompt-cache hit, fork at `2,560`, grown prompt, two lanes, lookup, EOS, and `1,112` tokens in one chunk; peak charged `21.7 GB`).

## Served Prefill Check (`tensorfold-native` ReleaseFast, `INKLING_LOAD_LIMIT_GB=26`)

Paired cold-chassis gated runs (`pre=[] post=[]` on both `p7-spec-1` with drafter loaded and `p7-ar-1` with `--no-drafts`):

| Length | Tokens | P7 `ar` (`--no-drafts`) | P7 `spec` (drafter loaded) | `spec / ar` |
| --- | ---: | ---: | ---: | ---: |
| `1K` rep 0 | 1,027 | 2,579 tok/s (0.398 s) | 2,534 tok/s (0.405 s) | 0.983× |
| `1K` rep 1 | 1,032 | 2,569 tok/s (0.402 s) | 2,541 tok/s (0.406 s) | 0.989× |
| `1K` rep 2 | 1,033 | 2,526 tok/s (0.409 s) | 2,514 tok/s (0.411 s) | 0.995× |
| `1K` rep 3 | 1,019 | 2,427 tok/s (0.420 s) | 2,434 tok/s (0.419 s) | 1.003× |
| `4K` | 3,882 | 2,214 tok/s (1.754 s) | 2,163 tok/s (1.795 s) | 0.977× |
| `8K` | 8,594 | 2,026 tok/s (4.242 s) | 2,063 tok/s (4.165 s) | 1.018× |
| `16K` | 16,593 | 1,773 tok/s (9.361 s) | 1,931 tok/s (8.593 s) | — |
| `31.7K` | 31,627 | 1,585 tok/s (19.952 s) | 1,746 tok/s (18.115 s) | — |

`1K`, `4K`, and `8K` all exceed `2,000 tok/s` under the default served configuration with the drafter loaded, and the `d.prompt` buffer (`18.87 MB`) is eliminated. Back-to-back 31.7K runs in the same loop showed thermal droop on the third and fourth servers (`p7-ar-2`, `p7-spec-2`); final multi-session qualification runs after P8 and decode optimization.
