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

## Grouped attention controllers

The candidate groups the three independent attention controllers in each repeated core slot into three dispatches: normalization, down projection, and gain/forget application. Previously each controller issued those three operations separately. On the oQ4e checkpoint the instrumented K=1 step used 1,302 dispatches versus 1,446 at base, a reduction of 144; its timestamped span was 25.756 ms versus 28.127 ms. These instrumented spans are attribution, not throughput.

The oQ4e family command below exited 0 with 82/82 tests, including unchanged batch/decode consistency at widths 1, 2 and 4 with zero reported maximum logit and cache differences. Its peak charge was 20.7 GB under the 28 GB cap.

```sh
TF_INKLING2_LOOP_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" \
TF_INKLING2_LOOP_OQ4E_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" INKLING_LOAD_LIMIT_GB=26 \
python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1 --summary all
```

Five gated interleaved 2,500-token `batchbench K 5` pairs used base/candidate, candidate/base, base/candidate, candidate/base, base/candidate order. All listed samples passed GPU and CPU gates and the compiler monitor; deferred attempts were discarded. Values are the internal median milliseconds per step.

| Width | Base samples | Candidate samples | Base median | Candidate median |
| --- | --- | --- | ---: | ---: |
| 1 | 21.448, 21.469, 21.396, 24.437, 24.252 | 20.760, 20.667, 20.670, 26.281, 22.954 | 21.469 | 20.760 |
| 4 | 44.288, 49.447, 49.359, 39.948, 39.887 | 43.556, 43.028, 49.549, 39.252, 42.768 | 44.288 | 43.028 |

The first three K=1 pairs had stable serial controls around 21.3–21.5 ms; their base inner samples ranged 21.335–21.669 ms, and candidate samples 20.581–20.882 ms. The latter two pairs and K=4 exhibited substantial serial-control drift. The first three pairs support a roughly 0.78 ms K=1 improvement under those conditions; the five-pair K=4 median difference is inconclusive in the observed spread. Neither 18 ms nor 36 ms target is established. The base and candidate runner binaries were 2,044,872 and 2,061,672 bytes respectively. K=8, final saved-capture fidelity, final suite, native build and served sweep remain pending.

## Rejected 32-token attention partitions

A 32-token split chunk and 64-split cap were applied to both batch and single-stream decode, with the batch partial-buffer stride carried in the per-row arguments. The oQ4e family command `python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1 --summary all`, with both `TF_INKLING2_LOOP_DIR` and `TF_INKLING2_LOOP_OQ4E_DIR` selecting the oQ4e checkpoint and `INKLING_LOAD_LIMIT_GB=26`, exited 1. The unchanged `oQ4e prefill without half pipelines matches per-token decode` test reported a maximum logit difference of `4.7588348e-4` at scale `2.1919092e1`, exceeding its `scale * 1.5e-5` bound. Batch/decode cases reported zero differences and the attention FP64 reference checks passed, but those results do not cure the prefill/decode failure. The candidate was reverted without changing the tolerance or timing it.

## Shared half4 attention loads

The batch and single-stream split kernels now load each shared f16 key/value tile in packed half4 units into threadgroup memory. All queries in the GQA group still share that tile; the existing split boundaries, per-head reduction order and f16 KV contract are unchanged. The oQ4e family command in the controller section exited 0 with 82/82 tests. The unchanged batch/decode checks at widths 1, 2 and 4 reported zero logit and cache differences, and the existing attention FP64 and ring checks passed.

Three interleaved 2,500-token `batchbench K 5` pairs compared controller-only `d2d7ddf40` to the half4 candidate in base/candidate, candidate/base, base/candidate order. Every retained invocation passed the GPU and CPU gates and compiler monitor; deferrals were retried.

| Width | Controller-only samples, ms | Half4 samples, ms | Controller median | Half4 median |
| --- | --- | --- | ---: | ---: |
| 1 | 20.199, 20.207, 20.154 | 19.209, 19.140, 19.047 | 20.199 | 19.140 |
| 4 | 39.169, 41.337, 43.703 | 39.588, 41.205, 43.213 | 41.337 | 41.205 |

K=1 favored half4 in every pair, by approximately 1.06 ms at the outer medians. One candidate inner sample reached 20.522 ms, so the full inner ranges overlap. K=4 drifted across the matrix, and its median difference is within the observed variation. This is an isolated comparison against controller-only, not another `b9bdf4e55` comparison. Both runner binaries were 2,061,672 bytes. Neither original target is established; K=8, final saved-capture fidelity, final suite, native build and served sweep remain pending.

## Decode attention microbenchmark

The split kernel was examined outside the model with a Swift harness that compiles a kernel source with the runner's options (safe math, fast functions, Metal 4.0) and times one command buffer of 40 local (1,024-row window) and 8 global split-and-merge pairs at position 2,500: the attention of one K=1 step at a 2,500-token cache. Random f16 keys and values sit in 48 separate buffers, as a cache's slots do. Each figure is the median, or the range of medians across repeated runs, of 15 to 30 command buffers under `campaign/heavy.py` after `gpu-gate-ready`. These numbers attribute kernel time; they are not step timings. The harness and the variants are lane scratch, not part of the repository.

| Variant | 48 split and merge pairs, us |
| --- | ---: |
| Base `4c2af5d2a`: one row a step through threadgroup memory | 3,990–4,034 |
| 32-row staged blocks, same per-row code | 3,159–3,203 |
| ... and the head width fixed at compile time | 1,675–1,695 |
| ... and each lane's four dimensions as one transposed `half4` | 1,480–1,538 |
| Committed kernel: block reduce-scatter, prefix maximum, loaded merge | 1,297–1,379 |
| Same, merge skipped (split cost alone) | 847–865 |
| Same, keys and values staged but unused (staging and merge) | 1,112–1,141 |
| Block softmax for multi-split windows (changes the order; not adopted) | 1,165–1,215 |
| Split kernels returning at once (dispatch and merge only) | 271 |

The largest single cost in the base kernel was not memory: its per-lane loops ran to a run-time `per = D / 32`, so `qv[]` and `acc[]` were dynamically indexed; fixing the width at 128 halved the time. Removing either exponential, the SIMD sum or the value update from the staged kernel saved 3–9% each. With only one key/value head dispatched (16 threadgroups instead of 128) the pairs still took 1,289 us against 1,535 us, so the staged kernel was bound by each threadgroup's serial chain, not by bandwidth: 24 heads × 16 splits give 384 serial chains of 64 (local) or 157 (global) rows at K=1. In the committed kernel arithmetic is a small share: staging plus the merge alone cost 1,112–1,141 us. The merge pass costs about 0.4–0.5 ms of the remainder; a lane-parallel merge (1,302–1,326 us) and a split that merges in its last threadgroup (1,174–1,573 us, noisy) did not reduce it reproducibly. Block softmax, the conventional flash-decoding order, gained under 0.1 ms here and would change bytes for windows over 64 positions, so it was not adopted.

`simd_sum` on this GPU equals an xor butterfly with partners 1, 2, 4, 8 and 16, in that order: 65,536 random sums with exponents spread over 2^±12 gave zero mismatches, while the 16-to-1 order mismatched in 40,588. A recursive-halving reduce-scatter in that partner order gives every row of a 32-row block the identical score in five levels of shuffles.

## Byte-preserving decode kernels

Every change in this section keeps the bytes of the kernels it replaces, so batch and single-stream decode, prefill and the saved captures see unchanged arithmetic. Each has a test that runs the earlier kernel, embedded as source where it was replaced, on random inputs at served shapes and requires equal words.

- **Staged decode attention.** For 128-wide heads with three query heads to a key/value head (the served geometry), `i2_attn_split` and `i2_batch_split` stage 32-row blocks of keys and values with every thread, lane `l` holding dimensions `l`, `l + 32`, `l + 64` and `l + 96` of a row as one `half4`. Each lane forms its four partial products for all 32 rows; the reduce-scatter above turns them into the rows' scores. The running maximum is a lane prefix maximum (maximum is exact), both exponentials are computed one row a lane, and only the mass and value recurrences run serially, with the serial kernel's expressions. Other geometries, such as the tiny fixture's 8-wide heads, keep the previous one-row-a-step kernel. The merge loads every split's partial before it starts. Test: `staged split attention keeps the serial kernel's bytes, single and batched`: the base split and merge, embedded as source, against the new pair at 40 single-stream cases from position 0 to 4,100 in sliding and full windows, and 18 batch rows at mixed positions against single streams.
- **Shared affine rows.** For 2 to 8 batch rows, `i2a_dense_shared` and `i2a_shared_projections` read each packed weight word and its scale and bias once and feed every row, each row accumulated in `i2a_dot`'s order; previously each row's threadgroup read the weights again. One row keeps the original kernels. Test: `shared-row affine kernels give each row the per-row kernel's bytes`, 42 cases over 4- and 8-bit weights, the logit cap, and 2 to 8 rows of the fused projections in both layouts.
- **Fused depth mix.** The batch path's `i2_batch_mix` and `i2_batch_mix4` accumulate every query and state dot of a token's depth mix in one pass, keeping each dot's per-thread order, SIMD reduction and part order, where `i2p_mix4` ran 20 reductions with two barriers each in one threadgroup. Test: `batched depth mixes keep the prefill mixes' bytes`, depths 1 to 5 and 1 to 4 queries.
- **Loads ahead of sums.** `i2a_dot` and `i2a_float` issue the loads of four (affine) or six (float) lane iterations before their fused multiply-adds, in the same order. The 48 f32 router projections and 24 f32 controller projections ran with 16 and 8 threadgroups, one load round trip an iteration. Test: `prefetched affine and float projections keep the serial kernels' bytes`, 15 cases including tails.
- **Batch route selection.** `i2_batch_pick` is `i2p_routepick` with its per-lane arrays indexed at compile time and its weights held in registers until the normalization. Test: `batched route selection keeps the prefill selection's picks and weights`, 12 cases including exact ties.
- **Shared NVFP4 experts.** For 2 to 8 batch rows, the lowest slot routed to an expert computes every slot routed to it (`i2_gu_rows`, `i2_down_rows`), decoding each weight group once, each slot's sums in `i2p_gu`'s and `i2p_down`'s order; the expert's other slots return at once. MXFP8 layers keep their kernels. Test: `batched expert projections keep the per-slot kernels' bytes`, 2 to 8 rows of random distinct picks over 12 experts.

The reduce-scatter's byte identity depends on `simd_sum`'s partner order measured on this M5 Max. On a GPU whose `simd_sum` adds in another order the staged kernel is still an fp32 sum, but its bytes would differ from prefill's attention; the staged-split test would then fail and name it. The new kernels are not registered in the Metal kernel-contract runner; no Inkling 2 Mini Metal kernel is registered there today.

The byte tests above passed on the final kernel source under the GPU lock; the invocation also ran the attention FP64 and ring tests:

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache"
python3 campaign/heavy.py -- sh -c 'zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/candG-build" &&
  zig build test-inkling2-loop -Dcpu=apple_m1 -j1 -- --test-filter "expert projections" --test-filter "route selection" \
  --test-filter "staged split" --test-filter "depth mix" --test-filter "prefetched" --test-filter "shared-row"'
```

It exited 0 and reported 58 staged-split, 42 shared-row, 25 depth-mix, 15 prefetched-projection, 12 route-selection and 14 expert cases equal, with the attention FP64 test's worst new-versus-reference error 7.30e-7 against a 5.52e-3 bound and the ring test's worst pair difference 1.79e-6, both unchanged from base. An earlier full oQ4e family run on the source before the expert sharing (`TF_INKLING2_LOOP_DIR` and `TF_INKLING2_LOOP_OQ4E_DIR` both oQ4e, `zig build test-inkling2-loop --summary all` under `heavy.py`) passed all 67 tests of the family module, with zero logit and cache differences in batch/decode at widths 1, 2 and 4 and zero chunk 1, 4 and 5 differences, but exited 1 because the affine test module did not compile (a syntax error in the new prefetch test, since fixed). The oQ4e prefill-versus-decode test in that module has therefore not run on the final source.

## Step timing status

The six kernel changes are committed as `wip:` because none has a gated step timing: `a363b7a3e` staged attention, `e11143f3a` shared affine rows, `d7c9be11c` fused depth mix, `4db85fb4d` loads ahead of sums, `2f4d126a8` batch route selection and `dc6323e59` shared NVFP4 experts, on `4697abcb8`, which adds `--round` to `inkling_profile.py timeline`. Each commit's tests are its byte-equality tests above.

No gated step timing of the final kernels completed. The one gated pair of an early staged kernel (32-row blocks with the run-time head width, before the changes above) gave K=1 19.331 ms against base 19.617 ms and K=4 38.893 ms against 38.831 ms, and its timeline cut `i2_batch_split` from 3,552 to 3,164 us at K=1; that kernel is not the committed one. A later screen of base against the committed attention collected one base K=1 sample, 19.750 ms (19.468–19.881 ms), before the host blocked it.

From about 14:10 the host's CPU gate deferred most timed runs: OrbStack Helper, outside the campaign, used 4 to 10 cores in bursts and `fseventsd` about one core with 5 GB resident, and the memory compressor ran at up to 100,000 pages a second. The GPU lock queue held 6 to 13 jobs across lanes. Gates were not relaxed; deferred samples were discarded.

Expected effect, not measured end to end: the microbenchmark puts the K=1 attention pairs at about 1.3 ms against 4.0 ms at base. The depth-mix, route-selection and f32 projection changes remove serial latency from 96, 24, 48 and 72 small dispatches a step; their size is unmeasured. At K>1 the shared affine and expert reads remove repeated weight reads that the base K=4 timeline showed as `i2a_batch_projections` 5.25 ms, `i2a_dense` 4.14 ms and `i2p_gu` plus `i2p_down` 13.56 ms against 2.63, 2.03 and 4.29 ms at K=1.

## Next step

1. Build the base runner at `4c2af5d2a` and the final commit's runner into separate prefixes, then run one gated `batchbench K 5` pair at K=1 and K=4 for base against the attention commit and for the attention commit against the final commit, with `.work/tmp/timed-run.sh` as in `inkling-f16-kv.md` and `PERF_GATE_MAX_BUSY_CORES=4`. Profile the final runner with `TF_METAL4_TIMELINE=1` and reduce the batch round with `inkling_profile.py timeline --round inkling2-batch` to attribute each change.
2. If they win, run the matrix: an identical-build control pair, then three interleaved pairs at K=1, K=4 and K=8.
3. On the final commit run `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu`, the oQ4e family run above, `zig build native -Dcpu=apple_m1 -j1`, and `INKLING_MODEL=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd sh campaign/fx/sweep-subagents.sh 4 4096` with the GPU-gate wrapper of `inkling-f16-kv.md`.
4. Remaining levers, prepared in lane scratch but unmeasured: one SIMD group computing four expert columns so each input group is loaded once for four columns (the per-slot expert kernels read the whole 9 KB input row for 2.3 KB of weights); staging attention rows as natural `half4` loads transposed with SIMD shuffles; and registering the new kernels in the Metal contract runner.
