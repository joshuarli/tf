# Inkling 2 Mini chunked prefill

Chunked teacher-forced prefill for `inkling2_loop`: T prompt tokens through all 48
execution slots per pass, writing the same KV and convolution state `position`
would. One command buffer a chunk, decode stays on `position`.

## Layout

- New path: `zig/src/families/inkling2_loop/prefill.zig` (`prefill`, `prefillWant`,
  `PrefillScratch`, `default_chunk = 64`), exported from `family.zig`.
- New kernels: `zig/src/families/inkling2_loop/kernels/prefill.metal`, listed in
  `kernels.zig` (`prefill`, `prefillSource`). Dense uses the warp-per-row exact
  kernel (`i2p_router_exact`, `i2p_cap_exact`); the rest mirrors the decode
  arithmetic per row: `i2p_rms`, `i2p_gather`/`i2p_mix`/`i2p_mix4`, `i2p_ctl`,
  `i2p_clip_scaled`/`i2p_clip`, `i2p_conv`/`i2p_conv_store` (in order),
  `i2p_attn` (causal within the chunk plus cached prefix, relative bias
  `min(pos-j, extent)`, sink, tau, window), routed experts grouped per pick
  (`i2p_routepick`, `i2p_gu`/`i2p_down` table decode, `i2p_combine`).
- One KV-store helper (`slotBase`/`slotRing`) targets both layouts: linear
  (`Cache.row`) pre-R1 and the R1 window ring (`at % window` local, full global,
  detected via `keys.length()`). A chunk crossing a window boundary writes ring
  rows in order and never overwrites rows a later token in the same chunk still
  needs: KV stores run after attention; conv stores keep only the last writer
  per ring row (chunks larger than the 4-tap ring would otherwise race).
- Runner: `prefill <chunk> <out>` writes full-history logits plus tok/s;
  `bench <steps> [warmup] [chunk]` uses chunked prefill (last-row-only logits)
  then decodes. `detokenize` added for token-equality checks.
- Server (`zig/src/native/inkling2_loop_host.zig`): prompt loop uses chunked
  prefill (`default_chunk`) with last-row handoff to decode scratch; per-token
  `position` kept for decode.

## Correctness

`zig build test-inkling2-loop -Dcpu=apple_m1` (via `heavy.py`) passes, including:

- tiny-fixture (`tiny-Inkling2LoopForConditionalGeneration`, 8 layers, hidden 32)
  chunked-vs-per-token: N = 10, chunks 1, 4, 5 (5 divides N, 4 does not, 1 is the
  base case). Logits, KV, convolution state within `1.5e-5 * max(1, maxAbs)`:

| chunk | logits | keys | values | conv | mlp | bound |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.85e-08 | 3.40e-06 | 5.01e-06 | 2.03e-06 | 8.94e-07 | 1.5e-05 |
| 4 | 8.85e-08 | 3.40e-06 | 5.01e-06 | 2.03e-06 | 8.94e-07 | 1.5e-05 |
| 5 | 8.85e-08 | 3.40e-06 | 5.01e-06 | 2.03e-06 | 8.94e-07 | 1.5e-5 |

Router picks equal (`expectEqualSlices`); continuation with one more token
matches. Same-kernel self-consistency (n = 4 vs 4x1) matches exactly (0e0).

- Parity full-history chunked logits pass `logit_gate.py` f32-vs-f32 on all 14
  short histories (chunk 64, one chunk each, 25-45 tokens):

| history | max_abs | mean_abs | maxKL | meanKL | top1 |
| --- | --- | --- | --- | --- | --- |
| greeting | 7.75e-05 | 1.01e-05 | 1.24e-10 | 3.55e-11 | 100% |
| explain_sliding | 9.16e-05 | 9.60e-06 | 1.08e-10 | 4.15e-11 | 100% |
| unit_test | 1.12e-04 | 9.88e-06 | 1.53e-10 | 3.93e-11 | 100% |
| refactor | 1.27e-04 | 9.19e-06 | 9.56e-11 | 3.28e-11 | 100% |
| math_mod | 9.82e-05 | 9.19e-06 | 9.16e-11 | 2.58e-11 | 100% |
| math_frac | 1.20e-04 | 8.67e-06 | 9.40e-11 | 2.77e-11 | 100% |
| dialogue | 1.19e-04 | 8.22e-06 | 1.02e-10 | 2.85e-11 | 100% |
| list_steps | 1.08e-04 | 1.03e-05 | 1.18e-10 | 3.43e-11 | 100% |
| structured | 8.49e-05 | 8.70e-06 | 2.38e-10 | 4.46e-11 | 100% |
| summary | 8.77e-05 | 9.00e-06 | 2.25e-10 | 4.11e-11 | 100% |
| sql | 1.13e-04 | 1.06e-05 | 1.60e-10 | 3.98e-11 | 100% |
| prose_review | 1.57e-04 | 9.58e-06 | 1.53e-10 | 4.23e-11 | 100% |
| code_trace | 9.58e-05 | 8.35e-06 | 9.59e-11 | 2.37e-11 | 100% |
| token_budget | 9.13e-05 | 8.53e-06 | 9.93e-11 | 3.15e-11 | 100% |

Bounds: max_abs 1e-3, mean_abs 1e-4, maxKL 1e-6, meanKL 1e-8, confident_margin
0.1, confident_top1 100%. No problems.

## Speed (1,334-token code-1024 prompt, M5 Max, via heavy.py)

| path | tokens | time | tok/s | chunk |
| --- | --- | --- | --- | --- |
| per-token (`logits`) | 1334 | 53748 ms | 24.8 | - |
| chunked full-history (`prefill 64`) | 1334 | 17159 ms | 77.7 | 64 |
| chunked full-history (`prefill 128`) | 1334 | 15534 ms | 85.9 | 128 |
| chunked full-history (`prefill 256`) | 1334 | 14697 ms | 90.8 | 256 |
| chunked last-only (`bench 4 1 64`) | 1334 | 12422 ms | 107.4 | 64 |
| chunked last-only (`bench 4 1 128`) | 1334 | 11660 ms | 114.4 | 128 |
| chunked last-only (`bench 4 1 256`) | 1334 | 12917 ms | 103.3 | 256 |

Target >= 300 tok/s not yet met (best 114.4 tok/s, 4.6x over per-token).
Remaining work (all within writable paths): simdgroup_matrix GEMM for dense
bf16 projections, grouped-by-expert MOE (sort picks, share weights), split-K
chunk attention for large prefixes. Current warp-parallel dense + last-only
unembedding + conv-store race fix give 4.6x; the three batched kernels above
should give another ~2.6x.

## Server

- `make -C campaign smoke MODEL=inkling2-mini MODE=ar
  SMOKE_BIN=<tree>/zig-out/native/bin/tensorfold-native` passes (no outer
  heavy.py: smoke takes the lock itself):
  `completion_tokens 512, rounds 511, 1.0 per round`.
- Served first 16 tokens equal runner greedy on the same ids (completions with
  raw prompt "Explain, step by step, how a hash map handles collisions.",
  14 prompt tokens):
  runner `[68014, 364, 31917, 13, 1513, 198347, 68014, 3256, 149920, 198347,
  68014, 32, 156518, 150874, 196976, 4799]`, served detokenized then
  retokenized equal, texts equal:
  `"\n\nsobes.tech AI\n\nAnswer from AI\n\nA hash map (hash"`.

## Commands

```sh
zig build test-inkling2-loop -Dcpu=apple_m1
zig build tf-inkling2-loop-run -Dcpu=apple_m1
./zig-out/bin/tf-inkling2-loop-run ~/models/Inkling-2-Mini@ed40e7da625f/tokenizer.json campaign/prompts/dev/code-1024.txt tokenize /tmp/ids-1024.json
python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-loop-run ~/models/Inkling-2-Mini@ed40e7da625f /tmp/ids-1024.json logits /tmp/logits-per-token.bin
python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-loop-run ~/models/Inkling-2-Mini@ed40e7da625f /tmp/ids-1024.json prefill 64 /tmp/logits-prefill64.bin
python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-loop-run ~/models/Inkling-2-Mini@ed40e7da625f /tmp/ids-1024.json bench 4 1 128
make -C campaign smoke MODEL=inkling2-mini MODE=ar SMOKE_BIN=<tree>/zig-out/native/bin/tensorfold-native
```

## P3 timeline attribution

Metal 4 timestamp rounds were split at execution slots and every 16 rows of
cross-pass copies, with one command buffer per segment. One timed chunk after
8 warm chunks is sampled at base position 1024 (1,334-token code prompt) and
4992 (5,265-token code prompt), both with 128 rows and full-history logits.
The sums below exclude gaps between command buffers and include timestamp and
profiling overhead; they are not end-to-end timings. The timeline prints at
most 25 kernel names per round, so the smallest entries may be incomplete.
The initial long run captured every receipt but exited 1 writing the 4.2 GB
output with a single write; its rerun is pending.

| kernel | 1024 base ms | 4992 base ms |
| --- | ---: | ---: |
| i2p_gu_grouped | 325.993 | 367.317 |
| i2p_down_grouped | 248.569 | 276.051 |
| i2p_attn | 264.433 | 480.580 |
| i2p_gemm | 253.100 | 288.721 |
| i2_copy | 16.431 | 17.820 |
| i2p_relbias | 5.964 | 6.722 |
| i2p_routepick | 5.840 | 6.696 |
| i2p_router_exact | 3.845 | 4.098 |
| i2p_conv | 2.405 | 2.141 |
| i2p_rms | 2.324 | 2.395 |
| i2_rms | 1.992 | 1.561 |
| i2p_clip_scaled | 1.819 | 2.046 |
| i2p_mix | 1.704 | 1.868 |
| i2p_gather | 1.693 | 1.901 |
| i2p_combine | 1.291 | 1.001 |
| i2p_conv_store | 0.916 | 1.010 |
| i2p_ctl | 0.778 | 0.883 |
| i2p_mix4 | 0.733 | 0.814 |
| i2p_moe_offsets | 0.659 | 0.365 |
| i2_add | 0.413 | 0.433 |
| i2p_clip | 0.372 | 0.414 |
| i2p_kv_store | 0.304 | 0.318 |
| i2_sig_mul | 0.287 | 0.299 |
| i2p_moe_sort | 0.144 | 0.147 |
| i2p_moe_count | 0.122 | 0.144 |
| i2p_moe_zero | 0.058 | 0.054 |
| i2p_embed | 0.012 | 0.012 |
| total span, 4,374 dispatches | 1,142.250 | 1,465.868 |

Expert gate/up and down account for 574.562 ms (50.3%) and 643.368 ms
(43.9%), respectively; attention rises with context. A 56 TFLOPS measured
fp16/bf16 GEMM rate at 50% utilization divided by approximately 2 × 4.4B
active parameters per token gives about 3,182 tok/s. This is a compute-only
arithmetic ceiling, not an observed prefill rate; compared with the previously
reported 120 tok/s it is about 26.5 times higher.
