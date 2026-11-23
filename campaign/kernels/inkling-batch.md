# Inkling 2 Mini independent decode batches

Status: correctness checked for one step at widths 1, 2, and 4. Gated throughput measurements remain open.

The step uses multi-row prefill projections, routing, experts, and the vocabulary head. It dispatches convolution,
cache writes, relative bias, and split attention separately for each cache at its own position. It uses one command
buffer and leaves server integration for later.

The family test uses distinct prompt prefixes at positions 4, 5, 6, and 7. It copies every cache before the step,
then compares every cache buffer and every logit with the single-position decoder. Each comparison uses an absolute
`1.5e-5` bound. Argmax matches for every tested row. The largest observed logit difference is `1.335144e-5` and the
largest cache difference is `9.536743e-6`. The changed dense and expert reduction order allows f32 differences.
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

An initial width-1 process completed five pairs but failed the immediate CPU post-check with 2.06 busy cores against
the 2-core limit. Its timings are discarded. Eight subsequent width-1 attempts failed CPU admission while other
processes were active. No throughput figure is accepted from those attempts. The initial process charged 20.2 GB.

For reproduction, build with `zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1`. Use the runner's `tokenize` mode
with the checkpoint tokenizer and `campaign/prompts/dev/prose-1024.txt` to create an IDs file. Under `heavy.py`,
run `gpu-gate-ready -1` with `GPU_GATE_MAX_GPU_PCT=60` and `GPU_GATE_MAX_UTIL_PCT=60`, then `perf-gate-ready`,
then `tf-inkling2-loop-run <model> <ids> batchbench <width> 5`, then `perf-gate-ready -1`. Discard a run if either
admission or the post-check fails. Run widths 1, 2, 4, and 8, and record the charged peak from `heavy.py`.
