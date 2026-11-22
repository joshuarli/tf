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
The long run reports the embedding segment at base 3968 because that round
name had already passed its warmup during unsplit earlier chunks; its 0.012 ms
is included in the sum. The sums below exclude gaps between command buffers and include timestamp and
profiling overhead; they are not end-to-end timings. The timeline prints at
most 25 kernel names per round, so the smallest entries may be incomplete.
The initial long run captured every receipt but exited 1 writing the 4.2 GB
output with a single write. A bounded-write rerun exited 0 and measured
1,439.780 ms across 4,373 reported dispatches; its occupancy readback moves
one copy out of the reported segments. The table retains the complete first
trace; neither trace is a throughput measurement.

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

## Routed expert occupancy

One 128-row chunk from each real code prompt: base 1024 and base 4992,
respectively. Each of 48 execution slots routes 1,024 picks to 128 experts.
The histogram counts expert/slot cases; it does not weight by routed rows.

| rows per expert | base 1024 | base 4992 |
| --- | ---: | ---: |
| 0 | 1170 | 1889 |
| 1 | 628 | 727 |
| 2–3 | 959 | 791 |
| 4–7 | 1308 | 909 |
| 8–15 | 1240 | 846 |
| 16–31 | 551 | 659 |
| 32–63 | 246 | 224 |
| 64–128 | 42 | 99 |

All cases use `i2p_gu_grouped` and `i2p_down_grouped`, with the layer's
NVFP4 or MXFP8 decoding. The baseline does not pad expert rows: it loops
from each expert's start to end, gathers through `ROW_OF`, and scatters
by the original slot. In the timeline, sort and count together cost at least
0.266 ms and 0.291 ms, and combine costs 1.291 ms and 1.001 ms, at the short
and long positions respectively; indirect gather and scatter inside the
expert kernels cannot be timed separately by this instrumentation. The
2-row experiment would perform 2,680 and 2,366 padded row computations
respectively, or 5.45% and 4.81% of the 49,152 useful routed rows.

Per-slot zero count, median and maximum of the 128 expert occupancies:

| slot | base 1024: zero / median / max | base 4992: zero / median / max |
| ---: | ---: | ---: |
| 0 | 11 / 5 / 60 | 28 / 3 / 71 |
| 1 | 13 / 5 / 101 | 23 / 3 / 115 |
| 2 | 3 / 6 / 53 | 16 / 4.5 / 98 |
| 3 | 1 / 7 / 42 | 9 / 5 / 65 |
| 4 | 8 / 6 / 63 | 24 / 4 / 83 |
| 5 | 6 / 6 / 58 | 12 / 5 / 59 |
| 6 | 18 / 5 / 69 | 29 / 4 / 88 |
| 7 | 26 / 4.5 / 66 | 39 / 3 / 118 |
| 8 | 26 / 4 / 67 | 46 / 2 / 91 |
| 9 | 30 / 3.5 / 76 | 48 / 1.5 / 98 |
| 10 | 23 / 4 / 78 | 35 / 2.5 / 99 |
| 11 | 32 / 3 / 58 | 48 / 1.5 / 99 |
| 12 | 13 / 5 / 53 | 39 / 3 / 69 |
| 13 | 23 / 5 / 55 | 39 / 2.5 / 101 |
| 14 | 26 / 3.5 / 64 | 39 / 2 / 75 |
| 15 | 35 / 3 / 100 | 57 / 1 / 99 |
| 16 | 33 / 4 / 111 | 50 / 1.5 / 123 |
| 17 | 72 / 0 / 128 | 78 / 0 / 128 |
| 18 | 44 / 2 / 68 | 64 / 0.5 / 112 |
| 19 | 26 / 3 / 97 | 49 / 1.5 / 101 |
| 20 | 30 / 3 / 101 | 46 / 2 / 110 |
| 21 | 21 / 3.5 / 71 | 38 / 3 / 99 |
| 22 | 20 / 4 / 50 | 37 / 2.5 / 88 |
| 23 | 18 / 7 / 45 | 32 / 4 / 58 |
| 24 | 20 / 6 / 50 | 30 / 4 / 62 |
| 25 | 24 / 3 / 74 | 35 / 3.5 / 103 |
| 26 | 30 / 3.5 / 57 | 39 / 3 / 81 |
| 27 | 26 / 4 / 70 | 48 / 1.5 / 88 |
| 28 | 32 / 4 / 71 | 39 / 2 / 92 |
| 29 | 29 / 3 / 56 | 50 / 1.5 / 98 |
| 30 | 19 / 5 / 49 | 33 / 3.5 / 53 |
| 31 | 22 / 4 / 59 | 37 / 2.5 / 106 |
| 32 | 27 / 4 / 84 | 43 / 2 / 73 |
| 33 | 31 / 3 / 68 | 56 / 1.5 / 76 |
| 34 | 23 / 5 / 74 | 44 / 2 / 111 |
| 35 | 54 / 1 / 128 | 64 / 0.5 / 128 |
| 36 | 16 / 6 / 50 | 24 / 4 / 66 |
| 37 | 25 / 4 / 80 | 40 / 3 / 86 |
| 38 | 33 / 3 / 56 | 35 / 2 / 97 |
| 39 | 31 / 4 / 74 | 52 / 1.5 / 97 |
| 40 | 26 / 4 / 73 | 37 / 3 / 86 |
| 41 | 32 / 3 / 55 | 42 / 2 / 90 |
| 42 | 14 / 6 / 58 | 36 / 3 / 54 |
| 43 | 21 / 4 / 58 | 42 / 2 / 107 |
| 44 | 18 / 5 / 42 | 36 / 3 / 80 |
| 45 | 27 / 5 / 51 | 47 / 2 / 87 |
| 46 | 15 / 6 / 33 | 28 / 2 / 65 |
| 47 | 17 / 5.5 / 54 | 27 / 3 / 56 |

The timeline table can be reproduced with `python3 campaign/kernels/inkling_profile.py
timeline --split --log LOG`. Set `TF_METAL4_TIMELINE=1` and
`TF_INKLING_PREFILL_PROFILE_BASE` to the desired base minus 1024 (eight warm
chunks), then set `TF_INKLING_PREFILL_OCCUPANCY_BASE` to the sampled base for
the additional occupancy lines. The readback splits the corresponding layer
rounds and must not be enabled for throughput measurement.

Rows per expert by execution slot, base 1024; each entry counts experts in that bin:

| slot | 0 | 1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11 | 9 | 16 | 48 | 30 | 9 | 5 | 0 |
| 1 | 13 | 11 | 25 | 43 | 23 | 8 | 4 | 1 |
| 2 | 3 | 8 | 30 | 40 | 35 | 8 | 4 | 0 |
| 3 | 1 | 9 | 18 | 44 | 43 | 11 | 2 | 0 |
| 4 | 8 | 7 | 24 | 39 | 38 | 9 | 3 | 0 |
| 5 | 6 | 14 | 20 | 35 | 39 | 10 | 4 | 0 |
| 6 | 18 | 12 | 23 | 25 | 32 | 14 | 3 | 1 |
| 7 | 26 | 13 | 17 | 29 | 28 | 9 | 4 | 2 |
| 8 | 26 | 17 | 15 | 31 | 20 | 11 | 7 | 1 |
| 9 | 30 | 13 | 21 | 24 | 21 | 11 | 7 | 1 |
| 10 | 23 | 14 | 23 | 24 | 27 | 10 | 6 | 1 |
| 11 | 32 | 17 | 20 | 17 | 21 | 14 | 7 | 0 |
| 12 | 13 | 13 | 28 | 33 | 20 | 15 | 6 | 0 |
| 13 | 23 | 18 | 11 | 29 | 30 | 11 | 6 | 0 |
| 14 | 26 | 17 | 21 | 24 | 20 | 14 | 5 | 1 |
| 15 | 35 | 18 | 24 | 15 | 17 | 11 | 5 | 3 |
| 16 | 33 | 6 | 21 | 29 | 25 | 8 | 4 | 2 |
| 17 | 72 | 12 | 12 | 12 | 5 | 6 | 4 | 5 |
| 18 | 44 | 17 | 20 | 18 | 12 | 5 | 10 | 2 |
| 19 | 26 | 21 | 20 | 27 | 17 | 9 | 5 | 3 |
| 20 | 30 | 17 | 22 | 22 | 15 | 14 | 7 | 1 |
| 21 | 21 | 17 | 26 | 21 | 25 | 12 | 5 | 1 |
| 22 | 20 | 15 | 20 | 29 | 24 | 15 | 5 | 0 |
| 23 | 18 | 11 | 13 | 34 | 38 | 10 | 4 | 0 |
| 24 | 20 | 11 | 16 | 31 | 32 | 15 | 3 | 0 |
| 25 | 24 | 14 | 28 | 21 | 20 | 16 | 4 | 1 |
| 26 | 30 | 15 | 19 | 15 | 31 | 10 | 8 | 0 |
| 27 | 26 | 15 | 17 | 25 | 28 | 11 | 4 | 2 |
| 28 | 32 | 7 | 19 | 23 | 26 | 15 | 5 | 1 |
| 29 | 29 | 17 | 24 | 17 | 18 | 16 | 7 | 0 |
| 30 | 19 | 15 | 13 | 35 | 27 | 13 | 6 | 0 |
| 31 | 22 | 14 | 21 | 25 | 30 | 9 | 7 | 0 |
| 32 | 27 | 8 | 24 | 28 | 23 | 11 | 6 | 1 |
| 33 | 31 | 17 | 26 | 14 | 19 | 14 | 5 | 2 |
| 34 | 23 | 16 | 17 | 27 | 31 | 8 | 5 | 1 |
| 35 | 54 | 12 | 24 | 10 | 13 | 7 | 4 | 4 |
| 36 | 16 | 8 | 18 | 35 | 36 | 11 | 4 | 0 |
| 37 | 25 | 13 | 24 | 28 | 21 | 10 | 5 | 2 |
| 38 | 33 | 11 | 26 | 13 | 23 | 14 | 8 | 0 |
| 39 | 31 | 11 | 10 | 39 | 22 | 7 | 6 | 2 |
| 40 | 26 | 11 | 21 | 20 | 30 | 16 | 3 | 1 |
| 41 | 32 | 14 | 21 | 20 | 18 | 16 | 7 | 0 |
| 42 | 14 | 12 | 18 | 36 | 35 | 8 | 5 | 0 |
| 43 | 21 | 15 | 21 | 28 | 26 | 10 | 7 | 0 |
| 44 | 18 | 15 | 12 | 37 | 25 | 15 | 6 | 0 |
| 45 | 27 | 13 | 13 | 31 | 27 | 11 | 6 | 0 |
| 46 | 15 | 8 | 18 | 32 | 37 | 17 | 1 | 0 |
| 47 | 17 | 10 | 19 | 26 | 37 | 17 | 2 | 0 |

Rows per expert by execution slot, base 4992; each entry counts experts in that bin:

| slot | 0 | 1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 28 | 26 | 11 | 18 | 22 | 16 | 5 | 2 |
| 1 | 23 | 18 | 24 | 23 | 19 | 16 | 4 | 1 |
| 2 | 16 | 22 | 19 | 27 | 29 | 10 | 3 | 2 |
| 3 | 9 | 11 | 27 | 35 | 27 | 16 | 2 | 1 |
| 4 | 24 | 13 | 21 | 30 | 22 | 12 | 5 | 1 |
| 5 | 12 | 19 | 18 | 36 | 25 | 14 | 4 | 0 |
| 6 | 29 | 10 | 21 | 21 | 26 | 19 | 1 | 1 |
| 7 | 39 | 11 | 18 | 24 | 20 | 9 | 5 | 2 |
| 8 | 46 | 8 | 21 | 18 | 12 | 16 | 5 | 2 |
| 9 | 48 | 16 | 17 | 18 | 9 | 12 | 4 | 4 |
| 10 | 35 | 19 | 16 | 16 | 22 | 13 | 6 | 1 |
| 11 | 48 | 16 | 14 | 17 | 13 | 12 | 5 | 3 |
| 12 | 39 | 8 | 18 | 24 | 17 | 14 | 7 | 1 |
| 13 | 39 | 12 | 20 | 20 | 17 | 14 | 3 | 3 |
| 14 | 39 | 20 | 11 | 17 | 19 | 17 | 4 | 1 |
| 15 | 57 | 9 | 16 | 10 | 14 | 14 | 5 | 3 |
| 16 | 50 | 14 | 13 | 15 | 18 | 10 | 5 | 3 |
| 17 | 78 | 7 | 8 | 8 | 15 | 5 | 1 | 6 |
| 18 | 64 | 15 | 12 | 9 | 11 | 7 | 5 | 5 |
| 19 | 49 | 15 | 16 | 13 | 15 | 9 | 7 | 4 |
| 20 | 46 | 17 | 11 | 19 | 16 | 12 | 5 | 2 |
| 21 | 38 | 13 | 20 | 22 | 16 | 12 | 4 | 3 |
| 22 | 37 | 17 | 21 | 17 | 15 | 14 | 4 | 3 |
| 23 | 32 | 9 | 22 | 25 | 14 | 21 | 5 | 0 |
| 24 | 30 | 13 | 10 | 30 | 26 | 13 | 6 | 0 |
| 25 | 35 | 12 | 17 | 23 | 24 | 11 | 4 | 2 |
| 26 | 39 | 12 | 21 | 22 | 14 | 12 | 5 | 3 |
| 27 | 48 | 16 | 17 | 13 | 12 | 13 | 6 | 3 |
| 28 | 39 | 20 | 12 | 18 | 17 | 15 | 6 | 1 |
| 29 | 50 | 14 | 11 | 21 | 9 | 18 | 2 | 3 |
| 30 | 33 | 14 | 17 | 22 | 18 | 19 | 5 | 0 |
| 31 | 37 | 20 | 13 | 19 | 20 | 13 | 3 | 3 |
| 32 | 43 | 17 | 11 | 12 | 25 | 13 | 6 | 1 |
| 33 | 56 | 8 | 14 | 9 | 20 | 12 | 7 | 2 |
| 34 | 44 | 17 | 18 | 11 | 20 | 11 | 5 | 2 |
| 35 | 64 | 13 | 15 | 8 | 9 | 12 | 2 | 5 |
| 36 | 24 | 23 | 14 | 21 | 26 | 14 | 5 | 1 |
| 37 | 40 | 9 | 17 | 22 | 21 | 13 | 4 | 2 |
| 38 | 35 | 23 | 18 | 20 | 12 | 13 | 4 | 3 |
| 39 | 52 | 12 | 16 | 17 | 11 | 12 | 5 | 3 |
| 40 | 37 | 14 | 19 | 24 | 7 | 19 | 7 | 1 |
| 41 | 42 | 18 | 18 | 13 | 18 | 13 | 3 | 3 |
| 42 | 36 | 12 | 20 | 17 | 18 | 21 | 4 | 0 |
| 43 | 42 | 15 | 16 | 16 | 17 | 15 | 5 | 2 |
| 44 | 36 | 15 | 15 | 20 | 16 | 17 | 8 | 1 |
| 45 | 47 | 13 | 8 | 20 | 14 | 20 | 4 | 2 |
| 46 | 28 | 26 | 24 | 13 | 16 | 13 | 6 | 2 |
| 47 | 27 | 26 | 15 | 16 | 23 | 13 | 8 | 0 |

The profile commands used the primary checkout's locked runner, with
`PATH=/opt/homebrew/bin:$PATH` to select Python 3.15 and `TMPDIR` set to
`/Users/josh/repos/tensorfold-m5/.work/tmp`. After building
`zig build tf-inkling2-loop-run -Dcpu=apple_m1`, tokenize each prompt with
`python3 /Users/josh/repos/tensorfold-m5/campaign/heavy.py --
./zig-out/bin/tf-inkling2-loop-run MODEL/tokenizer.json PROMPT tokenize IDS`.
For each row set `TF_METAL4_TIMELINE=1`, `TF_INKLING_PREFILL_PROFILE_BASE`
and `TF_INKLING_PREFILL_OCCUPANCY_BASE` as above, then run
`python3 /Users/josh/repos/tensorfold-m5/campaign/heavy.py --
./zig-out/bin/tf-inkling2-loop-run MODEL IDS prefill 128 OUTPUT 2>LOG`.

The two-row gate/up candidate preserves float32 inputs and accumulation; it
shares decoded weight values across a pair of rows and computes an unused
second row when the occupancy is odd. The exact router is unchanged. With
`zig build test-inkling2-loop -Dcpu=apple_m1` under `heavy.py`, the command
exited 0 and the chunked state test's worst value difference was 1.1958182e-5
against its 1.5e-5 bound. The candidate's full-history chunk-128 logits passed
`load_contract("f32-vs-f32")` and `check` for all 14 short histories. The
candidate and control full-history outputs were bit-identical for both
`harbour` and `cells_code`; their documented reference divergences therefore
did not worsen. The gated throughput comparison is pending.
