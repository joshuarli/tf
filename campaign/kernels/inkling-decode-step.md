# Inkling oQ4e decode step

The admission source differs from base `b9bdf4e55` only in the previous blocker receipt. The oQ4e checkpoint is `Inkling-2-Mini-oQ4e-mtp@3eea042751cd`; the workload is the first 2,500 IDs of `campaign/prompts/dev/code-4096.txt`. MTP is off.

## Weight floor

`python3 campaign/kernels/inkling_profile.py weights --model "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" --gbs 521` exited 0. Across 48 executed slots it counts 3,985,723,184 bytes: 1,719,929,600 expert, 1,374,664,704 attention projection, 489,741,984 unembedding, 339,738,624 controller and 61,648,272 other bytes. This includes quantized scales and biases and excludes padded output rows. At the previously measured 521 GB/s streaming ceiling the ideal weight-only floor is 7.650 ms; KV traffic, activation traffic and dispatches are additional. Routed experts assume eight selected experts and ideal reuse. At K>1 expert overlap and traffic depend on routing.

The native checkpoint command with `Inkling-2-Mini@ed40e7da625f` exited 0 and reproduced 5,550,410,592 bytes from `inkling-profile.md`.

## Baseline timeline

Each profile passed GPU admission, CPU admission and immediate CPU post-check under `campaign/heavy.py`, with `INKLING_LOAD_LIMIT_GB=26`. Each is one instrumented batch round after eight warmups. These spans attribute costs and are not throughput measurements. K=1 and K=4 each recorded 1,446 dispatches in one encoder; spans were 28.127 and 47.270 ms respectively.

| Kernel | Calls/step | K=1 ms | K=4 ms |
| --- | ---: | ---: | ---: |
| `i2p_gu` | 44 | 2.331 | 7.799 |
| `i2_batch_split` | 48 | 4.642 | 7.242 |
| `i2p_down` | 44 | 1.963 | 6.098 |
| `i2a_batch_projections` | 48 | 2.263 | 4.961 |
| `i2a_dense` | 49 | 2.022 | 4.155 |
| `i2p_batch_conv` | 192 | 1.987 | 2.077 |
| `i2a_float` | 144 | 1.944 | 2.030 |
| `i2p_mix` | 96 | 1.851 | 2.001 |
| `i2_batch_combine` | 48 | 1.458 | 1.550 |

K=4 experts are the largest combined cost, while attention splitting is the largest individual K=1 cost. The profile includes timestamp overhead and does not establish the attainable savings.

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" INKLING_LOAD_LIMIT_GB=26
export ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 TF_METAL4_TIMELINE=1
python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh "$PWD/.work/tmp/base-build/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" .work/tmp/code-2500.json batchbench K 9
```

Set `K` to 1 or 4. Generate the input and base runner as in `inkling-f16-kv.md`, using base `b9bdf4e55`. The base build command is recorded in the previous blocker receipt at `00c3711a7`.

## Rejected attention experiment

A candidate replaced shared KV staging and two threadgroup barriers per token in `i2_batch_split` with direct half-precision KV loads by each SIMD group. The unchanged family GPU test command exited 0: 82/82 tests passed and width 1, 2 and 4 batch/decode comparisons reported zero maximum logit and cache differences. It is rejected because it regressed K=1.

Three interleaved 2,500-token `batchbench 1 5` pairs passed the GPU and CPU gates and compiler monitor. Pair order was base/candidate, candidate/base, base/candidate. Base medians were 24.223, 21.633 and 21.749 ms (outer median 21.749); candidate medians were 26.110, 26.186 and 26.236 ms (outer median 26.186). Across all inner samples the base spanned 21.540–25.761 ms and candidate 25.977–26.310 ms. Candidate serial controls were 21.483, 21.669 and 21.592 ms. The host caused repeated admission deferrals; no deferred attempt was counted. The experimental matrix was stopped after the K=1 regression was established, after its current K=4 base invocation completed; the orchestrator exited 143. The candidate source was reverted.

## Rejected expert scheduling experiment

A candidate packed eight independent NVFP4 expert output columns into each 256-thread threadgroup while retaining the same per-column reductions. The family GPU tests exited 0 (82/82) with zero reported batch/decode logit and cache differences. The candidate was reverted because the measurements showed no benefit beyond noise.

Three interleaved 2,500-token `batchbench K 5` pairs passed the gates and compiler monitor, with base/candidate, candidate/base, base/candidate order. At K=1 the base medians were 21.678, 21.756 and 26.797 ms; candidate medians were 21.770, 21.637 and 21.546 ms. Outer medians were 21.756 and 21.637 ms; the slow base sample also had a slow serial control. At K=4 the base medians were 40.012, 40.046 and 39.902 ms; candidate medians were 39.934, 39.954 and 40.063 ms. Outer medians were 40.012 and 39.954 ms; all K=4 inner samples spanned 39.257–40.635 ms. The 0.058 ms K=4 difference is within this spread. Repeated gate deferrals were discarded.

## Rejected attention reduction layout

A candidate assigned adjacent dimensions to each SIMD lane instead of strided dimensions. `python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1 --summary all` exited 1: `batch_test.test.batched independent cache steps match decode` failed the unchanged finite-logit and `1.5e-5` difference check at `batch_test.zig:113`. The candidate was reverted without timing or tolerance changes.

## Barrier and key/value convolution investigation

`run` opens one serial compute encoder for each batch step and waits for its command buffer before returning. On Metal 4, `Slot.dispatched` inserts a device-visibility barrier after each serial dispatch. The next dependent step's CPU encoding therefore cannot overlap the preceding step's GPU work. The timeline stamps are emitted at those barriers, so their per-stage spans include kernel execution, barrier effects and timestamp instrumentation; they do not separately measure gaps. The instrumented K=1 span was 28.127 ms, versus an uninstrumented 21.569 ms base median below, and subtracting them does not yield a gap measurement.

A candidate fused the independent key and value convolutions into one dispatch per slot for affine batch projections, saving 48 dispatches. Native and oQ4e family test runs each exited 0 with 82/82 tests; the oQ4e batch/decode checks reported zero maximum logit and cache differences. Three gated 2,500-token `batchbench K 5` pairs, in base/candidate, candidate/base, base/candidate order, showed no measurable speedup. K=1 base medians were 21.626, 21.569 and 21.547 ms, candidate 21.782, 21.605 and 21.672 ms; outer medians were 21.569 and 21.672 ms. K=4 base medians were 41.086, 42.412 and 41.790 ms, candidate 42.022, 42.455 and 41.546 ms; outer medians were 41.790 and 42.022 ms. K=4 inner samples spanned 40.833–43.639 ms. Deferrals were retried; all retained samples passed the GPU/CPU gates and compiler monitor. The candidate was reverted.

A gated Metal System Trace on the base runner at K=1 and 2,500 cached IDs exited 0. The traced `batchbench 1 9` reported a 22.062 ms batch median and 21.945 ms serial median. The final six GPU command-buffer spans, containing both batch and serial operations, had a 20.869 ms median and 20.661–21.508 ms range. Their median is about 1.2 ms below the batch wall median, but the sample sets are not paired and tracing adds overhead. This is not an isolated inter-dispatch or CPU-encode measurement. Earlier instrumented stage spans include barriers and timestamps, so they cannot be subtracted from the uninstrumented wall time to produce such a measurement. The successful trace stayed under the 28 GB guard, peaking at 26.8 GB.

```sh
python3 campaign/heavy.py -- "$PWD/.work/tmp/timed-run.sh" "$PWD/.work/tmp/d1-trace3-compiler" \
  xctrace record --template 'Metal System Trace' --output "$PWD/.work/tmp/d1-base-trace3.trace" \
  --target-stdout "$PWD/.work/tmp/d1-trace3-target.log" --no-prompt --launch -- /bin/sh -c 'exec "$@" 2>&1' sh \
  "$PWD/.work/tmp/base-build/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" \
  "$PWD/.work/tmp/code-2500.json" batchbench 1 9
python3 campaign/kernels/inkling_profile.py trace --trace .work/tmp/d1-base-trace3.trace --steps 6 --process 'sh (9976)'
```

The trace process label includes the process ID; substitute the recorded process label when reproducing. The first launch attempt used `sh` instead of `/bin/sh` and exited 2 without running the model; the second exited 0. The `timed-run.sh` wrapper is the one in `inkling-f16-kv.md` and applies the GPU admission, CPU admission and post-checks, and compiler monitor.

## Rejected attention gate fusion

A candidate folded the sigmoid gate into the batch attention combine kernel for affine projections, removing 48 dispatches. The oQ4e family test command exited 0 with 82/82 tests and zero batch/decode logit and cache differences at widths 1, 2 and 4. The candidate was reverted because timings did not establish a reproducible speedup.

Three interleaved K=1 pairs at 2,500 cached IDs gave base 21.590, 21.450 and 21.513 ms and candidate 21.560, 21.436 and 21.713 ms. Five K=4 pairs gave base 45.174, 45.658, 45.236, 39.773 and 39.740 ms, and candidate 43.848, 43.855, 43.123, 44.832 and 39.863 ms. Pair order was base/candidate, candidate/base, base/candidate, candidate/base, base/candidate. The first three pairs favored the candidate but later pairs did not; serial controls varied from approximately 84 to 101 ms. All retained `batchbench K 5` invocations passed GPU and CPU gates and the compiler monitor; gate deferrals were discarded. The spread and serial-control drift make a causal speed claim inconclusive.

## Rejected depth-mix fusion

A candidate fused depth-state normalization and mixing into one batch kernel, avoiding 96 staging dispatches while preserving the reduction order. The oQ4e family suite exited 0 with 82/82 tests and zero batch/decode logit and cache differences at widths 1, 2 and 4. Three gated 2,500-token `batchbench K 5` pairs in base/candidate, candidate/base, base/candidate order gave K=1 base medians 21.538, 21.520 and 21.637 ms, candidate 21.812, 21.926 and 21.850 ms. K=4 base medians were 44.320, 47.921 and 48.095 ms, candidate 44.113, 47.011 and 48.736 ms; serial controls rose from about 90 to 103 ms across those pairs. The fusion was reverted: K=1 favored the base in every pair, and K=4 was inconclusive amid substantial drift. All retained attempts passed the GPU/CPU gates and compiler monitor.
