# Inkling 2 Mini independent decode batches

Status: correctness checked for one step at widths 1, 2, and 4. Gated decode measurements cover widths 1, 2, 4, and 8.

The step uses multi-row prefill projections, routing, experts, and the vocabulary head. It dispatches convolution,
cache writes, relative bias, and split attention separately for each cache at its own position. It uses one command
buffer and leaves server integration for later.

The family test uses distinct prompt prefixes at positions 4, 5, 6, and 7. It copies every cache before the step,
then compares every cache buffer and every logit with the single-position decoder. Each comparison uses an absolute
`1.5e-5` bound. Argmax matches for every tested row. The largest observed logit difference is `1.335144e-5` and the
largest cache difference is `9.536743e-6`. The dense and expert paths use different reduction orders that can
produce f32 differences.
This is a one-step fixture, not a multi-step or long-context equivalence claim.

With `TF_INKLING2_LOOP_DIR` set to the pinned Inkling 2 Mini checkpoint and `INKLING_LOAD_LIMIT_GB=26`,
`python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1` exited 0. The family test reported
the row comparisons above. Peak charged memory was 22.8 GB under the 28 GB cap. The build used Zig 0.17.1 and
`ZIG_GLOBAL_CACHE_DIR` at the primary checkout's `.work/zig-global-cache`.

The runner's `batchbench` mode prefills one cache per row from prefixes of `campaign/prompts/dev/prose-1024.txt`.
The prefixes have different lengths, from 1,139 tokens down to 1,132 for width 8. Each cache has capacity 4,096.
The mode alternates batched and sequential steps on the same caches, with one warmup pair. It times the forward and
copy of logits, then chooses each row's next greedy token outside the timer. Positions advance between samples.
The mode prints every sample and the median and range for each path. Its component times exclude prefill, model load,
greedy selection, and request handling. It does not measure served throughput.

On an Apple M5 Max running macOS 27.2, five alternating pairs per width produced these medians and ranges. All
runner invocations exited 0. GPU admission, CPU admission, and the CPU post-check exited 0 for each accepted run.
The batch and serial columns use the same progressing caches, so their positions differ by one step within a pair.

| Width | Batch ms per step, median (range) | Serial ms per step, median (range) | Batch aggregate tok/s | Serial aggregate tok/s | Peak charged GB |
| ---: | :--- | :--- | ---: | ---: | ---: |
| 1 | 29.513 (29.345–29.781) | 25.595 (25.279–25.678) | 33.883 | 39.070 | 1.5 |
| 2 | 39.446 (39.317–39.829) | 50.441 (50.406–50.558) | 50.702 | 39.650 | 21.7 |
| 4 | 62.961 (62.314–64.182) | 104.331 (103.805–105.200) | 63.531 | 38.340 | 22.9 |
| 8 | 111.118 (110.688–112.078) | 203.512 (202.788–203.951) | 71.996 | 39.310 | 25.4 |

Peak charge is the `heavy.py` process-tree and host wired-growth receipt, not a total device-residency measurement.
The width-1 checkpoint pages were already wired at admission, so its 1.5 GB charge does not represent the checkpoint
footprint. The width-8 process allocated eight 4,096-token caches and stayed under the configured 28 GB charge cap.
No phase split was collected. The timing does not establish long-context, multi-step output equality or served rates.
The width-4 rate is below the reported oMLX served rate of 73–77 aggregate tok/s, but the workloads are not identical.

An initial width-1 process completed five pairs but failed the immediate CPU post-check with 2.06 busy cores against
the 2-core limit. Its timings are discarded. Eight subsequent width-1 attempts failed CPU admission while other
processes were active. The first width-8 attempt also failed CPU admission. No throughput figure is accepted from
those attempts. The initial width-1 process charged 20.2 GB.

For reproduction, build with `zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1`. Use the runner's `tokenize` mode
with the checkpoint tokenizer and `campaign/prompts/dev/prose-1024.txt` to create an IDs file. Under `heavy.py`,
run `gpu-gate-ready -1` with `GPU_GATE_MAX_GPU_PCT=60` and `GPU_GATE_MAX_UTIL_PCT=60`, then `perf-gate-ready`,
then `tf-inkling2-loop-run <model> <ids> batchbench <width> 5`, then `perf-gate-ready -1`. Discard a run if either
admission or the post-check fails. Run widths 1, 2, 4, and 8, and record the charged peak from `heavy.py`. The
accepted runner was built before timing, using the same Zig and cache settings as the family test.
