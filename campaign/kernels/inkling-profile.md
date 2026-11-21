# Inkling 2 Mini native decode profile

Where the first native Metal forward (`lane/i2-forward` at `d150e2963`) spends a decode step, measured on
2026-11-21 on the M5 Max (40 GPU cores, 48 GiB). This is the correctness-first port: no kernel here has been
tuned yet.

## Step time

`tf-inkling2-loop-run <model> <ids> bench <steps> [warmup]` prefills the prompt one position at a time, runs the
warmup steps, then times single-token greedy decode steps.

| context | median ms a step | GPU ms | encode ms | steps | tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 83 | 105.1 | 104.2 | 0.6 | 4 | 9.5 |
| 1,334 | 222.4 | 221.5 | 0.7 | 32 | 4.5 |

Flint decodes in 6.5 ms a step at 1K on the same host. Peak footprint through `campaign/heavy.py` is 0.3–0.8 GB,
because the weights are file-backed and do not count against the process footprint.

## Weight-bandwidth floor

`inkling_profile.py weights` counts the bytes one decode token reads across the 48 executed slots (physical layers
0–5, then 6–17 three times, then 18–23):

| class | bytes a token | share |
| --- | ---: | ---: |
| attention projections | 2,583,564,288 | 46.5% |
| routed experts (top-8) | 1,839,411,552 | 33.1% |
| unembedding | 921,867,264 | 16.6% |
| controllers | 169,869,312 | 3.1% |
| everything else | 35,698,176 | 0.6% |
| total | 5,550,410,592 | |

At the host's measured 521 GB/s read ceiling (from `tf-fp4-bench`'s stream kernel) the floor is about 10.7 ms a
step, so the port runs 10× (83 tokens) to 21× (1,334 tokens) above it.

## Per-kernel GPU time

From the Metal 4 timeline receipt (`TF_METAL4_TIMELINE=4`, 4 timed rounds after 8 warm), 3,158 dispatches a step,
one encoder:

| kernel | calls a step | ms a step at 83 | ms a step at 1,334 | share at 1,334 |
| --- | ---: | ---: | ---: | ---: |
| `i2_attn` | 48 | 11.4 | 125.5 | 53.0% |
| `i2_gu` (expert gate/up) | 48 | 41.4 | 42.4 | 17.9% |
| `i2_down` (expert down) | 48 | 22.3 | 22.2 | 9.4% |
| `i2_mv` (dense bf16 matvec) | 688 | 20.3 | 21.6 | 9.1% |
| `i2_route` | 48 | 5.1 | 4.9 | 2.1% |
| all others | 2,278 | 20.4 | 20.3 | 8.6% |
| total | 3,158 | 120.9 | 236.9 | |

## The three largest attributable costs

1. **Attention, at 2.6 ms a call at 1,334 tokens.** It is the only kernel whose cost grows with context: from 0.24
   to 2.6 ms a call over 1,250 extra tokens, about 2 µs per cached token per slot. Reading the KV for 1,334 tokens
   over 48 slots is a few hundred MB, so this is far from bandwidth-bound. It is the first target, and it should
   reuse the Flint or GLM decode attention where their contracts match.
2. **Routed experts, 64 ms a step.** `i2_gu` and `i2_down` read at 26–85 GB/s. The K1 experiment
   ([fp4.md](fp4.md)) measured a table-decode kernel 11–16× faster on the same shapes, which would bring these to
   about 5 ms.
3. **Dense bf16 matvecs, 21.6 ms a step.** The 688 `i2_mv` calls read about 3.5 GB (attention projections and
   unembedding) at roughly 165 GB/s, a third of the read ceiling.

With those three at their measured or floor rates, the step would be bounded by the dispatch count (3,158) and the
remaining small kernels, about 20 ms. That is the next profile target.

## Reproduce

```sh
zig build tf-inkling2-loop-run -Dcpu=apple_m1
M=~/models/Inkling-2-Mini@ed40e7da625f
./zig-out/bin/tf-inkling2-loop-run $M/tokenizer.json campaign/prompts/dev/code-1024.txt tokenize ids-1k.json
python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-loop-run $M ids-1k.json bench 32
TF_METAL4_TIMELINE=4 python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-loop-run $M ids-1k.json bench 14 2 2> timeline.log
python3 campaign/kernels/inkling_profile.py timeline --log timeline.log
python3 campaign/kernels/inkling_profile.py weights --model $M
```

The 83-token prompt is the first 400 bytes of `campaign/prompts/dev/code-1024.txt`. Prefill runs one position at a
time, so the 1,334-token run spends about 4 minutes before the first timed step.
