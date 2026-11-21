# Inkling 2 Mini split-K decode attention

`i2_attn` ran one simdgroup per head (24 heads, 8 KV heads, head_dim 128) and walked every cached position
serially with an online softmax. At 1,334 tokens it cost 2.6 ms a call x 48 executed slots = 125.5 ms of a
237 ms step (see [inkling-profile.md](inkling-profile.md)). Only 24 simdgroups (768 threads) were active per
slot, so the kernel was occupancy-bound, far from the KV-read bandwidth floor.

## What changed

`kernels/attention.metal` replaces the serial walk with flash-decoding:

- `i2_attn_split`: partitions the window's positions into contiguous chunks (at most 64 positions each, at most
  16 chunks; `state.zig:attnPartition`). One threadgroup per (KV group, chunk); the group's 3 query heads run
  as 3 simdgroups sharing each chunk's K/V rows through threadgroup memory (one K/V read per KV group instead
  of one per head). Each writes its head's (peak, mass, D-wide accumulator) partial.
- `i2_attn_combine`: one threadgroup per head folds the chunk partials, starting from the reference's extra
  zero logit (peak 0, mass 1).

Semantics are unchanged: q scaled by `scale*tau` (tau from pos/floor/alpha when the layer is full),
score = q.k + B[h, min(pos-j, extent)] - sink[h], sliding-window lo = pos+1-window for local layers.
`i2_relbias` is untouched. The old `i2_attn` stays in `glue.metal` for the unit test's comparison.

Dispatch per slot is now two kernels (split, then combine) instead of one. The partials live in
`Scratch.attn_part` ([heads][16][dim+2] floats); the two pipelines are built once in `Scratch.init` from the
standalone `attention.metal` source, so `model.zig`'s pipeline table is untouched.

## Correctness

`zig/src/families/inkling2_loop/attention_test.zig` (wired into `test-inkling2-loop`) compares the old and new
kernels against an f64 host reference on random Q/K/V/bias at positions 1, 31, 1023, 1024, 1500 and 4100 in
both sliding and full modes, plus 16384 in full mode (tau > 1 there):

```sh
python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1
```

Worst measured old-vs-new max abs: 1.7e-6, against a derived fp32 bound of 5.5e-3
(16 * eps32 * sqrt(n * D) * max|V| at the largest case).

## Before / after

Before is the correctness-first port ([inkling-profile.md](inkling-profile.md), `lane/i2-forward` at `d150e2963`,
M5 Max 40 GPU cores): one simdgroup per head walking the window serially. After is this change. Medians are the
runner's `bench 14 2` wall median; per-kernel times are the `TF_METAL4_TIMELINE=4` receipt (4 timed rounds).

| context | step median ms, before | step median ms, after | attn ms a step, before | attn ms a step, after |
| ---: | ---: | ---: | ---: | ---: |
| 83 | 105.1 | 95.8 | 11.4 (`i2_attn`) | 3.2 (split 2.7 + combine 0.5) |
| 1,334 | 222.4 | 107.4 | 125.5 (`i2_attn`, 2.6 ms a call) | 5.3 (split 4.4 at 92 us a call + combine 0.9) |

At 1,334 tokens the step is now bounded by the routed experts (`i2_gu` 43.8 + `i2_down` 23.3 ms) and the dense
matvecs (`i2_mv` 21.0 ms), not attention (5.3 ms, 4.4% of the step). `i2_relbias` stays at 0.5 ms a step.

Reproduce:

```sh
zig build tf-inkling2-loop-run -Dcpu=apple_m1
M=~/models/Inkling-2-Mini@ed40e7da625f
./zig-out/bin/tf-inkling2-loop-run $M/tokenizer.json campaign/prompts/dev/code-1024.txt tokenize ids-1k.json
TF_METAL4_TIMELINE=4 python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-loop-run $M ids-1k.json bench 14 2 2> timeline.log
python3 campaign/kernels/inkling_profile.py timeline --log timeline.log
```

The 83-token prompt is the first 400 bytes of `campaign/prompts/dev/code-1024.txt` (83 tokens). Exact runs:
`bench 14 2` at 1,334 tokens gave 107.9 ms wall without the timeline and 107.4 ms with `TF_METAL4_TIMELINE=4`;
at 83 tokens, 97.3 ms (`bench 4 2`) and 95.8 ms (`bench 14 2` with the timeline).
