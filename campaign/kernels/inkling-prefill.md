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

Open item: Inkling2 Loop kernels are not yet registered with the core kernel
contract runner. Family-local checks do not replace that registration or its
non-finite, repeatability, shape, accumulator and launch-limit gates.

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

### Contention stop

The 2-row candidate is not retained: the host remained contended for over an
hour and the required throughput comparison could not be qualified. This is
not a performance rejection and establishes no speedup. The corrected timing
harness acquired the campaign GPU lock before `gpu-gate-ready -1` and retained
it through the runner and the post-run check with its PID ignored. Every
attempt stopped at the pre-run gate. The controller exited 75 after one hour;
the final check reported 20.5% other GPU time and 19% device
utilization against limits of 5% and 15%, with Ghostty and WindowServer
listed. Earlier control and candidate samples were discarded because the lock
was not held during both checks and another job could run in the gap. The
candidate-specific state and logit checks above therefore do not verify the
current reverted kernel; no measured improvement is claimed.

## Dense projection tiling

The dense prefill GEMM stages 32 activation rows and 32 K elements per
threadgroup, with four float32 8-by-8 accumulators per simdgroup. Weights
remain bf16 in storage and are converted directly to float32; activation and
accumulation precision stay float32. The output tile is 32 by 64, including
bounds checks for partial rows and columns.

On the M5 Max, one split Metal 4 timeline sample of the 128-row chunk at base
1024 attributed 253.100 ms to the original `i2p_gemm` and 153.871 ms to the
tiled kernel. The respective sums of reported spans were 1,142.250 ms and
1,045.760 ms. These are single profiles with timestamp overhead and possible
host contention, not paired throughput measurements. Both traces have 49
reported GEMM entries. The candidate used the same 1,334 token input as the P3
profile and ran with `TF_METAL4_TIMELINE=1` and
`TF_INKLING_PREFILL_PROFILE_BASE=0`; splitting starts at zero so each named
round receives eight warmups before base 1024. With `WORK` set to the main
checkout’s `.work` directory and `MODEL` to the checkpoint, the candidate
command was:

```sh
TF_METAL4_TIMELINE=1 TF_INKLING_PREFILL_PROFILE_BASE=0 python3 campaign/heavy.py -- \
  "$WORK/tmp/p4-candidate" MODEL "$WORK/tmp/p3-1024.json" prefill 128 "$WORK/tmp/p4-profile.bin"
```

The direct GPU comparison against the original GEMM covered every listed
projection shape at 1, 7 and 128 rows, including strided inputs, and returned
zero maximum difference for each case. Full-history chunk-128 logits for
harbour and cells_code were byte-identical to the base runner. The candidate
still requires a noise-controlled end-to-end timing comparison.
The parity make target reported 14 histories within the f32-vs-f32 contract,
2 explained long-history divergences, and 0 unexplained. It exited 2 because
its overall qualification requires all 16 histories within contract. The
first comparison's harbour diagnostic was interrupted by critical host memory
pressure; a retry completed that diagnostic and reproduced the same counts.

## P4 end-to-end timing

Runner prefill (`tf-inkling2-loop-run <model> .work/tmp/ids-1k.json bench 2 1 128`, 1,139 tokens, chunk 128),
base `main` 6b2614352 against P4 `a8c227997`, six base-candidate-base rounds, each run inside `heavy.py`
between two `gpu-gate-ready -1` checks. Busy host: the desktop (ghostty, WindowServer) held about 20%
device utilization and 25% GPU time, so both limits were raised to 35%; memory pressure stayed normal.
Logits are byte-identical to the base runner.

| | median tok/s | samples |
| --- | ---: | --- |
| base | 128.7 | 127.8, 131.2, 124.2, 123.3, 123.5, 129.5, 129.9, 133.7, 136.5, 123.5, 137.0, 110.9 |
| P4 | 136.5 | 131.9, 132.1, 133.6, 147.5, 146.5, 139.4 |

Per-round ratio against the mean of its two base runs: 1.019, 1.067, 1.056, 1.119, 1.127, 1.125; geometric
mean 1.085 (95% t-interval 1.038-1.133). Every round favours P4. Verdict: keep. Script:
`.work/tmp/prefill_ab.py BASE CAND IDS PAIRS` (scratch; the protocol above reproduces it).

## Routed expert matrix experiment

NVFP4 experts with at least 16 routed rows and MXFP8 experts with at least 8
use a 32-row, 64-column float32 matrix product. Smaller groups retain the
grouped row kernel. Both NVFP4 and MXFP8
weights are decoded to float32; NVFP4's per-expert global scale is applied
before the gate activation or down output. Selection reads the GPU's expert
offsets and requires no host readback. A diagnostic interleaved comparison at
8, 16 and 32 rows informed the two thresholds; the host was busy, so this is
not a qualified performance gate and the crossover needs quiet-host validation.
For 16 experts at production projection widths, five interleaved samples per
path gave median grouped versus matrix GPU spans of 1.691 versus 1.368 ms for
NVFP4 gate/up at 16 rows and 0.892 versus 0.504 ms for NVFP4 down. At eight
rows, MXFP8 gate/up measured 1.168 versus 0.886 ms and down 0.617 versus
0.428 ms. A separate 64-expert, 128-output diagnostic favored the NVFP4
grouped path at eight rows. These are synthetic, busy-host observations, not
proof of the best threshold on real routing distributions.

A split timeline at base 1024, on the same 1,334-token prompt as the
P3 sample, attributed 127.651 ms to `i2p_gu_grouped`, 99.114 ms to
`i2p_down_grouped`, and 129.594 ms to `i2p_expert_matrix`, a total of
356.359 ms across the three. P3 attributed 325.993 ms and 248.569 ms to
its two grouped kernels, a total of 574.562 ms. These are separate single
profiles with timestamp overhead and possible contention; they do not establish
an end-to-end throughput result or a threshold crossover. The matrix dispatches
combine gate/up and down in the timeline. The candidate profile used:

```sh
TF_METAL4_TIMELINE=1 TF_INKLING_PREFILL_PROFILE_BASE=0 python3 campaign/heavy.py -- \
  "$WORK/tmp/e1-final-candidate" MODEL "$WORK/tmp/p3-1024.json" prefill 128 "$WORK/tmp/e1-final-profile.bin" 2>"$WORK/tmp/e1-final-profile.log"
python3 campaign/kernels/inkling_profile.py timeline --split --log "$WORK/tmp/e1-final-profile.log"
```

The family test compares both projections and formats against host FP64 at
occupancies 1, 3, 8, 15, 16, 32 and 128; the largest tested shapes have K 2304 for
gate/up and 1152 for down. On the tested M5 Max the command exited 0. Compared
with the P4 candidate's saved full-history chunk-128 logits, all 1,336
harbour rows and 1,178 cells_code rows retained the same argmax. Their
largest absolute differences against P4 were 0.004470 and 0.000928,
respectively; this comparison is distinct from the reference parity gate.
The parity target reported 14 histories within contract, two explained
divergences and zero unexplained; it exited 2 because its overall qualification
requires all 16 histories within contract.

## E1 end-to-end timing

Same protocol as P4 (runner `bench 2 1 128` on `.work/tmp/ids-1k.json`, six base-candidate-base rounds inside
`heavy.py` between `gpu-gate-ready -1` checks, busy-host limits of 35% GPU time and utilization; memory pressure
normal). Base `main` 206faa4f5 (P4), candidate E1 `36f1311b9`.

| | median tok/s | samples |
| --- | ---: | --- |
| base | 135.4 | 116.9, 134.9, 128.3, 113.2, 129.0, 135.9, 144.5, 146.8, 131.5, 149.7, 139.5, 136.2 |
| E1 | 191.3 | 201.0, 171.9, 186.5, 196.1, 183.1, 198.5 |

Per-round ratios 1.597, 1.424, 1.408, 1.346, 1.302, 1.440; geometric mean 1.417 (95% t-interval 1.317-1.524).
Logits are not byte-identical to P4 (max abs 4.5e-3 on harbour, 9.3e-4 on cells_code, argmax identical on
every row) and the parity gate is unchanged at 14 within contract, 2 explained, 0 unexplained. Verdict: keep.

## Tiled prefill attention

For 24 query heads, eight KV heads and 128 channels, each threadgroup handles
eight query rows and the three heads sharing one KV head. It stages eight K/V
rows at a time and preserves the original per-query dot-product reduction and
online softmax order, including the zero-logit sink. Other attention shapes
retain the original kernel.

The family test covers global and 1024-slot ring caches at bases 0, 900 and
5000, with 1, 7 and 128 query rows, plus 9, 17, 33, 35, 63 and 100 rows at
bases 0 and 900. It compares with the original attention and an FP64 reference.
The original tiled matrix candidate passed an absolute-error kernel bound but
failed `structured` prefill: maximum logit error was 0.552188 against the
oracle, versus 0.0000992 for the baseline. Preserving the original arithmetic
order produces identical logits to the baseline on that history.
The 35-row attention regression test rejects the first candidate's 5.96e-8
change from the original kernel and passes with the order-preserving kernel.

All 14 short-history `prefill 128` logits passed the f32-vs-f32 oracle gate;
the largest absolute difference was 0.000132. Harbour and cells_code retained
the base runner's argmax on all 1,336 and 1,178 rows respectively.
The ordinary `logits` parity report lists 14 within contract, two explained
divergences and zero unexplained; its command exits 2 for incomplete overall
qualification because the two explained divergences remain.

Single split-timeline profiles measured 300.099 ms in base attention and
60.936 ms in the tiled kernel at base 1024, and 469.065 ms and 92.515 ms at
base 4992. These are load-sensitive profiler samples, not a timing gate. The
host failed `perf-gate-ready`; a controlled comparison remains outstanding.

## P5 end-to-end timing

Same protocol as P4 and E1 (runner `bench 2 1 128` on `.work/tmp/ids-1k.json`, six base-candidate-base rounds
inside `heavy.py` between `gpu-gate-ready -1` checks, busy-host limits of 35%). Base `main` 73df69de6 (P4 + E1),
candidate the same plus P5 `f9b0bfb59`. The prefill-mode parity gate on the candidate reports 14 within contract,
2 explained, 0 unexplained.

| | median tok/s | samples |
| --- | ---: | --- |
| base | 185.8 | 203.9, 191.0, 187.3, 186.6, 187.3, 186.0, 185.5, 160.1, 167.6, 160.7, 156.8, 155.8 |
| P5 | 217.1 | 237.3, 217.1, 217.0, 218.3, 195.8, 186.7 |

Per-round ratios 1.202, 1.161, 1.163, 1.263, 1.193, 1.194; geometric mean 1.196 (95% t-interval 1.158-1.235).
Verdict: keep. The profile's larger saving at base 4992 (469 -> 93 ms) suggests a larger gain at longer prompts;
that is not yet measured end to end.

## Chunk size

With P4, E1 and P5 merged, larger chunks raise expert occupancy (more experts reach the matrix path) and
enlarge every dense GEMM. Runner prefill on `.work/tmp/ids-1k.json`, one run each on main b9336a1ea: chunk 128
184 tok/s, 256 225, 512 348, 1024 361 (peak charged ~21 GB throughout). Six base-candidate-base rounds of 128
against 512 on the same binary (protocol as above): medians 241.4 and 337.9 tok/s, per-round ratios 1.418, 1.426,
1.370, 1.429, 1.420, 1.445, geometric mean 1.418 (95% t-interval 1.391-1.445). The prefill-mode parity gate at
`PREFILL_CHUNK=512` reports 14 within contract, 2 explained, 0 unexplained. `default_chunk` is now 512.

Served (`campaign/runs/2026-11-22/inkling2-mini-ar-1024-080205.jsonl`, two interleaved pairs against main
f6d8f83f0): TTFT on code-1024 (1,349 tokens) 5.65/5.97 -> 4.04/4.14 s, prose-1024 5.23/5.12 -> 3.70/3.62 s,
structured-1024 4.69/4.64 -> 3.59/3.42 s; greedy outputs byte-identical; server peak device bytes 3.99 -> 4.54 GB
(the prefill scratch scales with the chunk, mostly its all-rows logits buffer). One baseline trial was marked
invalid by the harness; the rest are valid at the busy-host limits.

## Dense TensorOps experiment

On M5 and later, the dense prefill projections use MetalPerformancePrimitives
`matmul2d` with float32 activations, checkpoint bf16 weights and float32
accumulators. The tensor retains the input row stride and offset. A 16-by-32
output tile handles partial rows and columns; the full-history vocabulary head
applies its existing tanh cap before storing. Short chunks (fewer than 16 rows),
small projections (N below 384) and K below 2304 keep the existing GEMM. Other
GPUs keep it for every shape. These selection thresholds are provisional and
have not been separately timed by shape.

The family test covers N/K shapes 3072/2304, 384/2304, 24/2304, 1024/2304,
32/2304, 2304/3072, 128/2304, 64/4608, 2304/64 and 200058/2304 at 1, 7,
35, 128 and 512 rows, with a row stride of two and a one-row input offset. It
compares all outputs to the earlier GEMM and samples up to five rows and 64
columns per shape against an independently accumulated FP64 dot product; the
vocabulary shape also exercises the cap. The bound is
`4 * (K + 64) * 2^-24 * sum(abs(x*w)) + 8 * 2^-24 * max(1, abs(result))`.
The FP64 check is sampled, not exhaustive. The family command returned zero.

The prefill-512 parity make target returned 2 with 14 histories within the
f32-vs-f32 contract, two explained long-history divergences and zero
unexplained. A split timeline on the 5265-token `p3-4096.json` prompt, with
`TF_METAL4_TIMELINE=1` and `TF_INKLING_PREFILL_PROFILE_BASE=0`, reported
133.862 ms for 289 `i2p_gemm_mpp` stages and 49.308 ms for 352 retained
`i2p_gemm` stages (183.170 ms combined), versus 333.6 ms for 641 stages in the
supplied main baseline at base 4096. These are separate single instrumented
profiles, not a paired timing result; the candidate observed pressure level 2
and 1.6 GB swap growth during the guarded run. The parent must perform the
paired end-to-end timing and assess the per-shape dispatch choices.

```sh
python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1
make -C campaign -f parity/inkling2-mini/parity.mk parity-inkling2-mini RUNNER_MODE=prefill PREFILL_CHUNK=512 RUNNER="$WORK/tmp/p7-candidate" LANE=p7 REPORT="$WORK/tmp/p7-parity.md"
TF_METAL4_TIMELINE=1 TF_INKLING_PREFILL_PROFILE_BASE=0 python3 campaign/heavy.py -- "$WORK/tmp/p7-candidate" MODEL "$WORK/tmp/p3-4096.json" prefill 512 "$WORK/tmp/p7-profile.bin" >"$WORK/tmp/p7-profile.log" 2>&1
python3 campaign/kernels/inkling_profile.py timeline --split --log "$WORK/tmp/p7-profile.log"
```

## P7 and E3 end-to-end timing

Runner prefill (`bench 2 1 512`, `.work/tmp/ids-1k.json`), base-candidate-base rounds inside `heavy.py` between
`gpu-gate-ready -1` checks at the busy-host limits, base `main` 77d023f9d.

- P7 (dense GEMM through TensorOps, float x bfloat -> float): twelve rounds, base median ~350, candidate ~382 tok/s;
  ratios 0.911 (disturbed first round), 1.308, 1.198, then nine rounds at 1.085-1.095; geometric mean 1.100 (95%
  t-interval 1.044-1.158). Prefill parity at chunk 512: 14 within contract, 2 explained, 0 unexplained. Kept.
- E3 (routed-expert matrix through TensorOps, float activations; lane/e3-expert-tensorops `d28b261d6`): six rounds,
  base median 350.0, candidate 355.3; ratios 1.014-1.023, geometric mean 1.018 (1.014-1.022). Real but small for a
  second expert kernel and device-dependent dispatch; parked, not merged. Float activations appear to keep the
  expert product off the fast path; an exact-enough hi/lo half split of the activations is the untested next step.

## Guarded two-half expert experiment

The parent measured the float TensorOps candidate at 1.018× end to end over
six rounds, with a 95% interval of 1.014–1.022. A follow-up tested two half
activation operands with float accumulation. NVFP4 used the exact half
weight. MXFP8 used the half E4M3 value and applied its power-of-two scale
outside the half operand for each 32-value block. A threadgroup fell back to
the float operation when any activation was non-finite, exceeded 65,504, or
could not be reconstructed exactly from the two halves. MXFP8 also fell back
for incompatible cooperative layouts and extreme or non-finite scales.

The split is not exact for every float: 2^-30 converts to zero in both halves,
so the proposed relative error bound cannot hold without a fallback. For an
admitted exact split the representation error is zero. Two half products per
term add more float rounding steps than the original dot product. Under finite,
non-underflowing intermediate arithmetic, a conservative sum bound must
account for up to `2*k` products and, for MXFP8, one scale accumulation per
32-value block. The existing `gamma_k` bound does not follow from the split
alone. The temporary family test nevertheless passed that existing FP64 bound
for both formats and all requested occupancies, and did not exceed the legacy
kernel's worst fixture error. It exercised the float fallback with 65,520,
2^-30 and infinity. A separate half × half TensorOps fixture preserved
residuals 2^-23 and 2^-24. A host representation check confirmed fallback for
the smallest float32 normal and subnormal values; float32 subnormal behavior
on the GPU was not independently established.

Five rotated split timeline profiles used the same 5,265-token input,
chunk 512 and base 4096. The legacy runner selected `i2p_expert_matrix`; the
float runner was the prior TensorOps candidate. Each run used `heavy.py`, with
host-load checks immediately before and after. All returned zero, peak charged
memory was 21.3 GB, and the worst pressure level was 1. Times below are the
median and full range of summed GPU time in milliseconds:

| Variant | Expert kernel | Expert plus grouped kernels |
| --- | ---: | ---: |
| Legacy matrix | 733.817 (690.927–752.105) | 877.505 (827.152–899.998) |
| Float TensorOps | 691.272 (661.501–707.850) | 831.668 (797.320–853.039) |
| Guarded half split | 1,253.466 (1,244.227–1,287.202) | 1,388.713 (1,377.812–1,425.117) |

The guarded split is slower than both comparisons, with non-overlapping ranges.
The result includes guarding, fallback, and MXFP8 scale handling; it is not a
measurement of an unguarded split. Path usage was not counted. The split
kernel and its temporary tests were discarded, and no split parity run was
made. The earlier float TensorOps implementation remains on the lane for the
parent's decision.

## E4 staged expert decode screen

The initial busy-host decode screen used the candidate's `i2p_stage_weights` kernel on 42,467,328 synthetic weights, with three warmups and 20 timed GPU command buffers per format. It counted packed weights and scales read plus bf16 output written. NVFP4 took a median 0.484 ms (range 0.400–1.185 ms), or 225 GB/s; MXFP8 took 0.399 ms (0.374–0.500 ms), or 322 GB/s. Sampled output values matched the synthetic inputs. These are standalone, warm-buffer measurements, not end-to-end gains; the NVFP4 range is wide. The 60/60 GPU and CPU admission checks and immediate post-CPU check passed; `heavy.py` reported 0.4 GB peak charged memory and pressure level 1. No full model was loaded.

The new E4 kernels are unregistered in the kernel contract, as are the other Inkling 2 Mini family kernels. Registration is a family-wide promotion requirement tracked by the parent.

The candidate stages eight experts at a time. The serialized family test (`zig build test-inkling2-loop -Dcpu=apple_m1 -j1` under `heavy.py` with `TF_INKLING2_LOOP_DIR` set to the Mini checkpoint) exited 0. Its fp64 checkpoint comparison examined 24,461,180,928 staged weights: 0 inexact conversions, 0 staged-value mismatches and 768 detected corrupted-value controls. The `-j1` scheduling flag kept the two test executables from loading the checkpoint concurrently. The f32 global scale remains outside NVFP4 staging and is applied to the f32 product.

Prefill parity with chunks 512 and 4 each exited 2 with 14 within contract, 2 explained divergences and 0 unexplained. The first chunk-512 attempt generated the candidate outputs but failed before comparison because a space in the reference PATH was split by make; the corrected reference PATH reused the matching receipts and produced the result above. End-to-end keep/revert timing and memory evidence are pending.
