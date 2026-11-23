# Inkling oQ4e prefill TensorOps

The baseline is revision `2376a289ddc24425ec7fefd23132b88269781385`. The checkpoint is Inkling 2 Mini oQ4e revision `3eea042751cd`. The host is an M5 Max with 48 GiB on macOS 27.2. The runner uses greedy decode and 512-row prefill chunks.

The split timeline reports GPU attribution, not throughput. Each table entry is the mean GPU time per 512-row chunk within one prompt. The runner first executes eight warm chunks, then resets the cache and captures all chunks of the requested prompt. The 1K prompt has 1,024 tokens and the 4K prompt has 4,096 tokens. The per-segment timeline retains at most 25 kernel entries, so small kernel totals can be incomplete.

The baseline runner differs from the base revision only by the `profile-prefill` runner mode. Both baseline profiles passed GPU and CPU admission and the immediate CPU post-check. The 1K run peaked at 19.7 GB charged and the 4K run peaked at 19.0 GB charged under the 28 GB `heavy.py` cap.

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp"
export ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26
export GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 TF_INKLING_PREFILL_PROFILE_BASE=0
TF_METAL4_TIMELINE=2 python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh \
  "$RUNNER" "$MODEL" campaign/prompts/inkling/code-1024.json profile-prefill 512 > "$LOG" 2>&1
python3 campaign/kernels/inkling_profile.py timeline --split --log "$LOG"
```

For 4K set `TF_METAL4_TIMELINE=8` and use the corrected Inkling-tokenized fixture described below. `profile-prefill` resets the cache length between warm prompts. Use the same prompt and runner mode for both binaries.

On a device with TensorOps, oQ4e chunks over eight rows stage affine dense weights as half and multiply them with f32 activations. Affine experts, NVFP4 expert matrices, and tiled attention use half operands. The matrix accumulators, residual stream, and KV storage remain f32. Smaller rows and devices without TensorOps retain the existing default paths. The native checkpoint keeps its existing default arithmetic.

Half conversion overflow on the default path returns `HalfActivationOverflow` and resets the cache length to zero. The caller must refill that cache before reuse. The opt-in fast profile retains its existing overflow reporting behavior.

The delivery candidate passed the host and GPU suite with 25 host tests and 81 GPU tests and no skips. The GPU suite's peak charged memory was 23.9 GB.

```sh
PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" \
ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26 \
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
```

## Threshold-14 per-kernel GPU time

Both threshold-14 delivery profiles exited 0 and passed the same admission and immediate post-checks as the baseline. A dash means the timeline did not report the kernel in that run. Values are GPU milliseconds per chunk.

| Kernel | Base 1K | Delivery 1K | Base 4K | Delivery 4K |
| --- | ---: | ---: | ---: | ---: |
| `i2p_expert_matrix` | 625.474 | - | 618.509 | - |
| `i2p_expert_matrix_wide` | - | 367.514 | - | 350.293 |
| `i2a_dense_rows` | 316.524 | - | 329.093 | - |
| `i2p_attn_tiled` | 93.981 | - | 193.206 | - |
| `i2p_attn_half` | - | 60.379 | - | 107.671 |
| `i2as_gemm` | - | 98.369 | - | 100.809 |
| `i2a_float` | 80.039 | - | 82.594 | - |
| `i2p_gu_grouped` | 78.828 | 66.895 | 59.064 | 51.392 |
| `i2p_down_grouped` | 60.303 | 50.424 | 44.169 | 38.480 |
| `i2a_group_up` | 47.792 | - | 46.502 | - |
| `i2a_half_group_up` | - | 41.405 | - | 40.350 |
| `i2a_group_down` | 22.474 | - | 21.860 | - |
| `i2p_relbias` | 21.134 | 21.328 | 21.851 | 22.093 |
| `i2a_half_group_down` | - | 18.898 | - | 18.310 |
| `i2af_gemm` | - | 11.927 | - | 11.882 |
| `i2as_prepare` | - | 8.233 | - | 7.910 |
| `i2p_clip_scaled` | 6.250 | 6.366 | 6.236 | 6.410 |
| `i2p_conv` | 6.141 | 6.191 | 5.900 | 6.133 |
| `i2p_gather_rms` | 5.147 | 5.618 | 5.358 | 5.829 |
| `i2p_mix` | 4.639 | 4.720 | 4.624 | 4.749 |
| `i2p_combine` | 4.211 | 4.085 | 4.413 | 4.289 |
| `i2p_kv_store` | 0.680 | 3.657 | 0.619 | 2.067 |
| `i2p_ctl` | 3.557 | 3.236 | 3.515 | 3.200 |
| `i2p_rms_pair` | 2.660 | 2.759 | 2.692 | 2.746 |
| `i2p_conv_store` | 2.330 | 2.384 | 2.223 | 2.370 |
| `i2p_mix4` | 1.564 | 1.577 | 1.610 | 1.612 |
| `i2_add` | 1.291 | 1.298 | 1.328 | 1.343 |
| `i2_sig_mul` | 1.147 | 1.264 | 1.050 | 1.167 |
| `i2p_clip` | 1.041 | 1.059 | 1.075 | 1.095 |
| `i2_rms` | 0.991 | 0.997 | 1.013 | 1.029 |
| `i2p_rms` | 0.960 | 1.012 | 0.921 | 0.976 |
| `i2p_rms3` | 0.931 | 0.935 | 0.930 | 0.956 |
| `i2p_routepick` | 0.737 | 0.681 | 0.698 | 0.648 |
| `i2p_moe_offsets` | 0.269 | 0.156 | 0.297 | 0.165 |
| `i2p_scatter` | 0.164 | 0.172 | 0.169 | 0.176 |
| `i2p_moe_sort` | 0.095 | 0.094 | 0.096 | 0.096 |
| `i2p_moe_zero` | 0.066 | 0.087 | 0.073 | 0.071 |
| `i2p_moe_count` | 0.078 | 0.054 | 0.072 | 0.057 |
| `i2p_pass_copy` | 0.054 | 0.054 | 0.057 | 0.058 |
| `i2a_embed` | 0.018 | 0.018 | 0.018 | 0.018 |
| `i2_copy` | 0.009 | 0.010 | 0.010 | 0.012 |

## Tile experiments

The first half-affine tile used strided scalar weight loads. Its 1K instrumented affine stage took 329.265 ms per chunk. Loading four adjacent weight codes per thread reduced that stage to 240.896 ms in the next profile. Increasing the affine K tile from 32 to 64 did not improve the profile. A 64-row by 32-column tile raised the affine stage to 334.184 ms and was discarded.

The default half expert kernel with a 32-column output tile took 382.858 ms per chunk in the 1K group profile. A 64-column tile took 343.168 ms. Three gated 1K pairs compared these two candidates in alternating order. The 32-column rates were 470.7, 470.0, and 470.4 tok/s. The 64-column rates were 486.8, 483.2, and 484.7 tok/s. Each run exited 0 and had no observed compiler overlap. A 128-column tile showed no paired improvement beyond the observed spread, so the 64-column tile was selected.

Reducing the NVFP4 matrix occupancy threshold from 16 to 8 reduced the instrumented total but missed the fidelity floor. Its saved-capture comparison found 2,756 of 3,015 top-1 matches (91.410%) and mean KL 0.0392384. The threshold remains 16. Expert K tiles of 32 and 128 did not improve on the 64-element tile in the profile.

An MPP `matmul2d` experiment dequantized affine tiles to half in threadgroup memory and accumulated in f32. The first fixture failed because the MPP descriptor defaults to multiply rather than multiply-accumulate. After selecting multiply-accumulate explicitly, the rounded-operand fixture passed. The gated 1K profile then measured 306.825 ms per chunk in `i2am_dense`, compared with 236.941 ms for the retained half-affine kernel. The MPP experiment was discarded.

## Wide expert candidate measurements

Each retained run used the pinned oQ4e checkpoint, greedy decoding, `heavy.py`, the 60/60 GPU gate and CPU admission and post-checks. The 1K and 4K runs used `bench 32 4 512`. Batch runs used `batchbench K 5` with 2,500 cached prompt tokens. Pair 2 reversed the side order. One K=4 baseline attempt overlapped a compiler and was discarded.

| Workload | Pair | Base | Selected | Base decode ms | Selected decode ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k | 1 | 351.8 | 478.8 | 19.946 | 19.958 |
| 1k | 2 | 355.9 | 485.2 | 20.001 | 19.975 |
| 1k | 3 | 355.2 | 484.0 | 19.980 | 19.980 |
| 4k | 1 | 355.2 | 499.7 | 21.810 | 21.861 |
| 4k | 2 | 332.3 | 495.6 | 21.976 | 21.885 |
| 4k | 3 | 328.5 | 492.7 | 21.888 | 21.880 |
| k1 | 1 | 20.981 | 24.074 | - | - |
| k1 | 2 | 24.384 | 20.948 | - | - |
| k1 | 3 | 24.773 | 20.893 | - | - |
| k4 | 1 | 43.220 | 43.250 | - | - |
| k4 | 2 | 41.931 | 43.267 | - | - |
| k4 | 3 | 42.722 | 43.219 | - | - |

Prefill values are tokens per second. Batch values are milliseconds per batch step. Decode values are the median of 32 steps in each run.

A per-projection staging experiment converts affine weights to half and uses MPP GEMM. The half-input version reduced the instrumented affine GPU stages to 65.656 ms per chunk but missed fidelity: 2,749 of 3,015 top-1 matches (91.177%) and mean KL 0.0389152. Using f32 inputs passed with 2,766 matches (91.741%) and mean KL 0.0409931. With that f32-input path, threshold 8 found 2,757 matches (91.443%), and threshold 12 found 2,758 (91.476%). Both missed the 91.5% floor. These experiments used saved oracle captures and did not rerun the reference model.

The 32-column candidate's saved-oracle command returned 2 under the stricter cross-checkpoint f32 contract. The separate streamed comparator returned 0 with 2,760 of 3,015 top-1 matches (91.542%), mean KL 0.0383510, and zero non-finite logits. The wide expert kernel produced the same comparator totals. The oracle verifier checked all 16 saved histories, including two longer than the sliding window.

`python3 tools/pr_check.py --base 2376a289ddc24425ec7fefd23132b88269781385` exited 1. The lean count was 246 at both base and head, with zero new findings. The checker also reported four touched sources over 600 lines. All four exceeded 600 lines at the pinned base: `prefill.metal` 1,329, `prefill.zig` 1,158, `prefill_test.zig` 1,374, and `run.zig` 749.

## Delivery fidelity

The delivery runner captured all 16 histories against the saved oracle. The oracle verifier exited 0. The cross-checkpoint f32 parity command exited 2 under its stricter contract. The independent streamed comparator exited 0: 2,761 of 3,015 top-1 matches (91.575%), mean KL 0.0390074, and zero non-finite logits. This is two matches above the top-1 floor.

```sh
PATH="$HOME/repos/tf/.work/venv-ref/bin:/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" \
ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26 \
make -C campaign -f parity/inkling2-mini/parity.mk parity-inkling2-mini \
  RUNNER_MODE=prefill PREFILL_CHUNK=512 LANE=p2-delivery \
  REPORT="$PWD/.work/tmp/p2-delivery-parity.md" METAL_LOGITS="$PWD/.work/tmp/p2-delivery-parity" \
  WORK="$PWD/.work" REF_LOGITS="$HOME/repos/tf/.work/parity/inkling2-mini/ref" \
  REF_UV=env "REF_VENV=''" RUNNER="$PWD/.work/tmp/p2-delivery-build/bin/tf-inkling2-loop-run" \
  INKLING_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
```

The comparator reads the saved oracle and each candidate capture in eight-row blocks. It calculates f64 log-softmax KL and averages the 16 history means. The procedure is in [Inkling oQ4e fast](inkling-oq4e-fast.md).

Threshold 13 for both expert projections also missed the saved-capture floor: 2,757 of 3,015 top-1 matches (91.443%), mean KL 0.0384418, and zero non-finite logits. The streamed comparator exited 1. The stricter cross-checkpoint parity command exited 2 after producing all 16 candidate captures.

## Delivery alternating comparison

The gated three-pair comparison below uses the threshold-14 delivery binary. Every retained run exited 0, passed GPU and CPU admission and the immediate CPU post-check, and had no observed compiler overlap. Gate deferrals and one compiler-overlap attempt were retried. Each pair alternated base/delivery, delivery/base, base/delivery. The model and runner arguments are those in the profile and suite instructions above; the timing invocations use `bench 32 4 512` on the 1,024- and 4,096-token prompts and `batchbench K 5` on a 2,500-token prompt. Each invocation is wrapped in `python3 campaign/heavy.py --` with `INKLING_LOAD_LIMIT_GB=26`, `GPU_GATE_MAX_GPU_PCT=60`, and `GPU_GATE_MAX_UTIL_PCT=60`.

| Workload | Pair | Base | Delivery | Base decode ms | Delivery decode ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | 353.2 | 601.1 | 20.001 | 19.963 |
| 1K | 2 | 333.0 | 587.6 | 19.954 | 20.960 |
| 1K | 3 | 354.1 | 596.7 | 19.996 | 20.008 |
| 4K | 1 | 364.5 | 646.4 | 21.776 | 21.799 |
| 4K | 2 | 344.3 | 637.6 | 21.853 | 21.828 |
| 4K | 3 | 330.0 | 611.1 | 22.207 | 21.874 |
| K=1 | 1 | 21.007 | 21.001 | - | - |
| K=1 | 2 | 21.084 | 21.113 | - | - |
| K=1 | 3 | 20.973 | 21.039 | - | - |
| K=4 | 1 | 40.541 | 39.986 | - | - |
| K=4 | 2 | 39.880 | 40.502 | - | - |
| K=4 | 3 | 40.267 | 39.780 | - | - |

Prefill is tok/s; batch is milliseconds per step. The median delivery 1K prefill is 596.7 tok/s and does not meet the 600 tok/s target. The median 4K prefill is 637.6 tok/s. The decode and batch differences are within the observed run-to-run spread.

The split with matrix threshold 13 for gate/up and 14 for down also failed fidelity: 2,758/3,015 top-1 matches (91.476%), mean KL 0.0385366, zero non-finite logits; the streamed comparator exited 1. The stricter cross-checkpoint parity command exited 2.

The threshold-14/13 candidate uses matrix threshold 14 for gate/up and 13 for down. Its saved-capture comparator exited 0 with 2,769/3,015 top-1 matches (91.841%), mean KL 0.0386953, and zero non-finite logits. The oracle verifier checked all 16 saved histories; the stricter cross-checkpoint parity command exited 2. The command above was run with `LANE=p2-inverse`, output paths `p2-inverse-parity`, and runner `.work/tmp/p2-inverse-build/bin/tf-inkling2-loop-run`.

A gated screening comparison of threshold 14 on both projections against the final split used the same three-pair order. The 1K prefill results were 597.1/616.8, 615.3/619.1, and 615.9/619.0 tok/s, respectively. All six runs exited 0, passed the gates, and had no observed compiler overlap. The split was retained for the corrected acceptance checks that follow.

## Threshold-14/13 per-kernel GPU time

All profiles exited 0 and passed the admission and post-check gates. The 4K profiles use the corrected Inkling-tokenized fixture. Values are GPU milliseconds per 512-row chunk. A dash means the timeline did not report the kernel; each segment retains at most its 25 largest entries.

| Kernel | Base 1K | Final 1K | Base 4K | Final 4K |
| --- | ---: | ---: | ---: | ---: |
| `i2p_expert_matrix` | 625.474 | - | 728.616 | - |
| `i2p_expert_matrix_wide` | - | 353.910 | - | 401.543 |
| `i2a_dense_rows` | 316.524 | - | 365.487 | - |
| `i2p_attn_tiled` | 93.981 | - | 215.128 | - |
| `i2p_attn_half` | - | 57.962 | - | 113.632 |
| `i2as_gemm` | - | 94.351 | - | 107.750 |
| `i2a_float` | 80.039 | - | 90.781 | - |
| `i2p_gu_grouped` | 78.828 | 63.485 | 86.204 | 68.124 |
| `i2p_down_grouped` | 60.303 | 42.458 | 63.687 | 46.309 |
| `i2a_group_up` | 47.792 | - | 54.664 | - |
| `i2a_half_group_up` | - | 39.980 | - | 44.593 |
| `i2a_group_down` | 22.474 | - | 25.812 | - |
| `i2p_relbias` | 21.134 | 20.459 | 24.089 | 23.404 |
| `i2a_half_group_down` | - | 18.235 | - | 20.467 |
| `i2af_gemm` | - | 10.930 | - | 11.774 |
| `i2as_prepare` | - | 7.298 | - | 7.067 |
| `i2p_clip_scaled` | 6.250 | 5.682 | 6.423 | 6.280 |
| `i2p_conv` | 6.141 | 5.161 | 5.571 | 5.431 |
| `i2p_gather_rms` | 5.147 | 5.460 | 5.725 | 5.960 |
| `i2p_mix` | 4.639 | 4.461 | 4.763 | 4.616 |
| `i2p_combine` | 4.211 | 3.749 | 4.447 | 4.027 |
| `i2p_kv_store` | 0.680 | 4.017 | 0.431 | 1.321 |
| `i2p_ctl` | 3.557 | 3.082 | 3.128 | 3.075 |
| `i2p_rms_pair` | 2.660 | 2.592 | 2.630 | 2.614 |
| `i2p_conv_store` | 2.330 | 1.947 | 2.053 | 2.003 |
| `i2p_mix4` | 1.564 | 1.534 | 1.777 | 1.732 |
| `i2_add` | 1.291 | 1.263 | 1.377 | 1.305 |
| `i2p_clip` | 1.041 | 1.014 | 1.167 | 1.149 |
| `i2_sig_mul` | 1.147 | 0.973 | 0.891 | 0.920 |
| `i2_rms` | 0.991 | 0.956 | 1.105 | 1.076 |
| `i2p_rms` | 0.960 | 0.814 | 0.840 | 0.829 |
| `i2p_rms3` | 0.931 | 0.881 | 0.937 | 0.932 |
| `i2p_routepick` | 0.737 | 0.519 | 0.523 | 0.505 |
| `i2p_moe_offsets` | 0.269 | 0.144 | 0.308 | 0.168 |
| `i2p_scatter` | 0.164 | 0.162 | 0.180 | 0.172 |
| `i2p_moe_sort` | 0.095 | 0.092 | 0.097 | 0.098 |
| `i2p_moe_count` | 0.078 | 0.063 | 0.072 | 0.069 |
| `i2p_moe_zero` | 0.066 | 0.052 | 0.047 | 0.034 |
| `i2p_pass_copy` | 0.054 | 0.054 | 0.058 | 0.058 |
| `i2a_embed` | 0.018 | 0.017 | 0.019 | 0.018 |
| `i2_copy` | 0.009 | 0.009 | 0.010 | 0.010 |

Summed reported GPU spans were 1,391.838/754.156 ms for base/final at 1K and 1,699.282/889.478 ms at 4K. The profile command above used `p2-final-build` for the final runner. All runs exited 0 on their first attempt.

The threshold-14/13 candidate passed the same suite command above with exit 0, 25/25 host tests and 81/81 GPU tests, no skips, and 23.9 GB peak charged memory. Its timing prefix and the worktree's native server were built first with `/opt/homebrew/bin/zig build native tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/p2-final-build"` and `/opt/homebrew/bin/zig build native -Dcpu=apple_m1 -j1`, respectively; both exited 0.

The inherited main-checkout `fixtures/code-4096.json` does not match the Inkling tokenizer. All measurements above using that fixture, including its 2,500-ID prefix, are exploratory only and do not establish 4K or batch acceptance. Recreate the corrected fixtures with the final runner:

```sh
python3 campaign/heavy.py -- "$PWD/.work/tmp/p2-final-build/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd/tokenizer.json" campaign/prompts/dev/code-4096.txt tokenize "$PWD/.work/tmp/p2-tokenized-4096.json"
jaq '.[0:4096]' .work/tmp/p2-tokenized-4096.json > .work/tmp/p2-correct-4096.json
jaq '.[0:2500]' .work/tmp/p2-tokenized-4096.json > .work/tmp/p2-correct-2500.json
```

A 16-row expert tile was tested against the retained 32-row tile using the corrected 4K fixture. Its gated profile exited 0, but expert matrix GPU time rose from 401.543 to 434.512 ms per chunk; it was discarded. The focused affine build exited 0 with 16 passed and 4 skipped; that focused result is not the required full suite.

## Corrected alternating comparison, threshold 14/13

These retained runs passed admission and post-check gates, exited 0, and had no observed compiler overlap. They use `campaign/prompts/inkling/code-1024.json` for 1K and the corrected fixtures above for 4K and batch. Pair order was base/candidate, candidate/base, base/candidate. Prefill is tok/s; batch and decode are milliseconds per step.

| Workload | Pair | Base | Candidate | Base decode ms | Candidate decode ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | 353.1 | 610.3 | 19.987 | 19.997 |
| 1K | 2 | 357.3 | 611.7 | 20.001 | 20.007 |
| 1K | 3 | 356.3 | 612.0 | 19.959 | 20.034 |
| 4K | 1 | 334.2 | 500.3 | 22.737 | 23.075 |
| 4K | 2 | 275.9 | 632.6 | 22.324 | 22.242 |
| 4K | 3 | 285.3 | 513.2 | 21.870 | 21.968 |
| K=1 | 1 | 24.856 | 24.698 | - | - |
| K=1 | 2 | 21.009 | 24.895 | - | - |
| K=1 | 3 | 20.980 | 21.016 | - | - |
| K=4 | 1 | 39.750 | 39.937 | - | - |
| K=4 | 2 | 39.915 | 40.228 | - | - |
| K=4 | 3 | 40.209 | 40.201 | - | - |

The 1K candidate median is 611.7 tok/s; the corrected 4K median is 513.2 tok/s, below target. The candidate K=1 median is 24.698 ms against 21.009 ms for base, within the large observed spread (base 20.980–24.856 ms; candidate 21.016–24.895 ms). K=4 and single-stream decode differences are within their observed spreads. The first matrix was stopped after its valid 1K pairs once the incorrect inherited fixture was identified; it exited 143. The corrected 4K/batch matrix exited 0.

The threshold-14/8 candidate (gate/up 14, down 8) passed the saved-capture comparator with 2,763/3,015 top-1 matches (91.642%), mean KL 0.0388930, and zero non-finite logits; the streamed comparator exited 0. The saved-oracle verifier checked 16 histories. The capture command exited 2 under the stricter cross-checkpoint f32 contract. The candidate capture used the fidelity command above with `LANE=p2-down8`, output paths `p2-down8-parity`, and runner `.work/tmp/p2-down8-build/bin/tf-inkling2-loop-run`.

A corrected 4K screening comparison of threshold 14/13 versus 14/8 gave 554.6/648.4, 636.4/651.5, and 641.6/655.0 tok/s in three alternating pairs. All six runs exited 0, passed the gates, and had no observed compiler overlap. This is a screening comparison between candidates, not the required base comparison.

The final threshold-14/8 candidate passed `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` with exit 0, 25/25 host tests and 81/81 GPU tests and no skips. Peak charged memory was 23.9 GB. Before the suite and sweeps, `/opt/homebrew/bin/zig build native tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/p2-final8-build"` and `/opt/homebrew/bin/zig build native -Dcpu=apple_m1 -j1` both exited 0 with `ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache"`.

## Final threshold-14/8 per-kernel GPU time

These gated profiles use the corrected 4K Inkling-tokenized fixture. Each exited 0 on its first attempt. Values are GPU milliseconds per 512-row chunk; dashes indicate stages not reported in the top-25 per-segment timeline.

| Kernel | Base 1K | Final 1K | Base 4K | Final 4K |
| --- | ---: | ---: | ---: | ---: |
| `i2p_expert_matrix` | 625.474 | - | 728.616 | - |
| `i2p_expert_matrix_wide` | - | 370.893 | - | 402.917 |
| `i2a_dense_rows` | 316.524 | - | 365.487 | - |
| `i2p_attn_tiled` | 93.981 | - | 215.128 | - |
| `i2p_attn_half` | - | 59.185 | - | 114.168 |
| `i2as_gemm` | - | 96.660 | - | 105.702 |
| `i2a_float` | 80.039 | - | 90.781 | - |
| `i2p_gu_grouped` | 78.828 | 64.230 | 86.204 | 66.290 |
| `i2p_down_grouped` | 60.303 | 19.166 | 63.687 | 21.616 |
| `i2a_group_up` | 47.792 | - | 54.664 | - |
| `i2a_half_group_up` | - | 40.603 | - | 43.782 |
| `i2a_group_down` | 22.474 | - | 25.812 | - |
| `i2p_relbias` | 21.134 | 21.049 | 24.089 | 22.869 |
| `i2a_half_group_down` | - | 18.619 | - | 20.083 |
| `i2af_gemm` | - | 11.648 | - | 11.649 |
| `i2as_prepare` | - | 8.133 | - | 7.032 |
| `i2p_clip_scaled` | 6.250 | 6.186 | 6.423 | 6.134 |
| `i2p_conv` | 6.141 | 5.888 | 5.571 | 5.331 |
| `i2p_kv_store` | 0.680 | 6.044 | 0.431 | 1.286 |
| `i2p_gather_rms` | 5.147 | 5.501 | 5.725 | 5.840 |
| `i2p_mix` | 4.639 | 4.652 | 4.763 | 4.593 |
| `i2p_combine` | 4.211 | 3.967 | 4.447 | 3.981 |
| `i2p_ctl` | 3.557 | 3.200 | 3.128 | 3.062 |
| `i2p_rms_pair` | 2.660 | 2.801 | 2.630 | 2.608 |
| `i2p_conv_store` | 2.330 | 2.277 | 2.053 | 1.963 |
| `i2p_mix4` | 1.564 | 1.557 | 1.777 | 1.705 |
| `i2_add` | 1.291 | 1.285 | 1.377 | 1.301 |
| `i2_sig_mul` | 1.147 | 1.196 | 0.891 | 0.942 |
| `i2p_clip` | 1.041 | 1.042 | 1.167 | 1.114 |
| `i2_rms` | 0.991 | 0.988 | 1.105 | 1.056 |
| `i2p_rms` | 0.960 | 0.965 | 0.840 | 0.825 |
| `i2p_rms3` | 0.931 | 0.930 | 0.937 | 0.899 |
| `i2p_routepick` | 0.737 | 0.603 | 0.523 | 0.495 |
| `i2p_moe_offsets` | 0.269 | 0.150 | 0.308 | 0.168 |
| `i2p_scatter` | 0.164 | 0.167 | 0.180 | 0.172 |
| `i2p_moe_sort` | 0.095 | 0.093 | 0.097 | 0.097 |
| `i2p_moe_count` | 0.078 | 0.060 | 0.072 | 0.066 |
| `i2p_moe_zero` | 0.066 | 0.063 | 0.047 | 0.037 |
| `i2p_pass_copy` | 0.054 | 0.054 | 0.058 | 0.055 |
| `i2a_embed` | 0.018 | 0.024 | 0.019 | 0.018 |
| `i2_copy` | 0.009 | 0.010 | 0.010 | 0.010 |

Summed GPU spans were 1,391.838/760.314 ms at 1K and 1,699.282/860.302 ms at 4K, base/final respectively. The command above used `p2-final8-build` as the final runner.

## Final alternating comparison

The final 24 retained cells exited 0, passed GPU and CPU admission and the immediate CPU post-check, and had no observed compiler overlap. They used the corrected fixtures above. The order was base/final, final/base, base/final. Prefill is tok/s; batch and decode are milliseconds per step. Decode is the median of 32 steps after four warmups, and batch is the median of five paired steps.

| Workload | Pair | Base | Final | Base decode ms | Final decode ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | 354.9 | 619.6 | 20.024 | 20.026 |
| 1K | 2 | 354.7 | 619.2 | 19.958 | 19.992 |
| 1K | 3 | 356.7 | 620.2 | 19.986 | 19.998 |
| 4K | 1 | 345.8 | 619.4 | 21.790 | 21.845 |
| 4K | 2 | 319.8 | 612.2 | 21.868 | 21.840 |
| 4K | 3 | 318.2 | 603.6 | 21.923 | 21.893 |
| K=1 | 1 | 21.046 | 21.085 | - | - |
| K=1 | 2 | 21.071 | 21.051 | - | - |
| K=1 | 3 | 21.064 | 21.055 | - | - |
| K=4 | 1 | 40.387 | 40.515 | - | - |
| K=4 | 2 | 39.914 | 40.513 | - | - |
| K=4 | 3 | 39.883 | 42.300 | - | - |

Final prefill medians are 619.6 tok/s at 1K and 612.2 tok/s at 4K; every final sample exceeds 600. K=1 medians are 21.064/21.055 ms base/final. K=4 medians are 39.914/40.515 ms; the 0.601 ms difference is smaller than the final range of 1.787 ms. Single-stream decode medians are 19.986/19.998 ms, within the observed spread.

The exact per-cell command was:

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp"
export ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26
export GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60
python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh \
  "$RUNNER" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" "$IDS" $ARGS
```

`RUNNER` was the base or final timing binary, `IDS` was the 1K, corrected 4K, or corrected 2,500-token fixture, and `ARGS` was `bench 32 4 512`, `batchbench 1 5`, or `batchbench 4 5` as appropriate. No compilation occurred inside the timing windows. The full matrix exited 0.

## Served-agent sweeps

With the final worktree native build, `INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" sh campaign/fx/sweep-subagents.sh 1 4096` exited 0 on its first attempt. Per repetition, server turn-1 TTFT was 4.640 s and 0.220 s; aggregate throughput was 27.567 and 42.799 tok/s. The second repetition can reuse the first repetition's cached prefix. `INKLING_LOAD_LIMIT_GB=26` was set, and a local `perf-gate-ready` wrapper invoked `GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1` before each CPU admission check; the script retained its immediate CPU post-checks.

`INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" sh campaign/fx/sweep-subagents.sh 4 4096` exited 0 on its first attempt. For repetition 1, turn-1 server TTFTs by agent were 6.155, 5.626, 5.100, and 4.569 s; aggregate throughput was 55.064 tok/s. For repetition 2, turn-1 TTFTs were 0.674, 1.647, 2.130, and 1.158 s; aggregate throughput was 67.625 tok/s. The concurrent-equals-solo gate found zero unequal or failed replies for code and chat at 1, 2, and 4 streams. The serial comparisons also found zero unequal or failed replies, but their speculative-draft comparisons were vacuous because no drafts were accepted. Peak charged memory was 23.7 GB.

To reproduce the additional GPU admission check for both sweep sizes, place this wrapper at `.work/tmp/p2-gates/perf-gate-ready` and make it executable, then prepend `.work/tmp/p2-gates` to `PATH` when running the two commands above:

```sh
#!/bin/sh
if [ "$#" -eq 0 ]; then
    GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 /Users/josh/.local/bin/gpu-gate-ready -1
    result=$?
    if [ "$result" -ne 0 ]; then exit "$result"; fi
fi
exec /Users/josh/.local/bin/perf-gate-ready "$@"
```
