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
