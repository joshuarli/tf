# Inkling 2 Mini oQ4e

Status: checkpoint loading, family-local tests, fidelity comparisons, and gated measurements are complete.

The candidate is `mlx-community/Inkling-2-Mini-oQ4e-mtp` at `3eea042751cd`, compared with the native
`Inkling-2-Mini` checkpoint at `ed40e7da625f`. Both revisions are pinned in `campaign/sources.lock.json`.

The new affine kernels join the family-wide kernel-contract registration item required before promotion out of the
campaign. This lane uses family-local FP64 tests and parity under the campaign's explicit ruling.

The checkpoint's four `model-*.safetensors` headers contain 1,102 tensors in total. Layer 0's attention uses
`model.layers.0.attn.in_proj.weight` with shape `[8600, 576]` and dtype U32, plus BF16 scales and biases.
Its NVFP4 `w13.weight` has shape `[128, 2304, 288]` and dtype U32, with U8 scales and F32 global scales.
Layer 20's `w13.weight` has the same U32 shape but BF16 scales and biases for affine quantization.
Layer 5 uses separate `q_proj`, `k_proj`, and `v_proj` tensors. The loader maps the MLX names alongside the native ModelOpt names. Sliding `in_proj` rows are q, k, v, relative, gate, then sink, with widths
3072, 1024, 1024, 384, 3072, and 24. Global `q_proj` rows are q, relative, gate, then sink.
Global `k_proj` and `v_proj` each append 32 dynamic convolution rows to 1024 key or value rows.
Loop passes can supply distinct q, k, and v inputs, so those slices must be projected separately on those passes.

The 16 saved histories produced 3,015 rows with no non-finite logits. Overall top-1 agreement with the f32 oracle
was 2,763 of 3,015 positions (91.64%). At an f32 top-two margin of at least 0.1, agreement was 2,681 of 2,862
positions (93.68%). Mean KL, averaged across histories, was 0.03935111. Both long histories exceeded 1,024 tokens.
The standard f32-versus-f32 parity command exited 2 because the checkpoint weights differ. Its existing exception
labels do not establish the cause of differences between these checkpoints.

The oracle check verified 16 histories. The captures used `RUNNER_MODE=prefill`, `PREFILL_CHUNK=512`, f32
activations, the oQ4e checkpoint and the locally built runner. The initial comparator was stopped because it began
optional traces outside this lane. The rerun used `REF_VENV="''"` to disable those traces and reused the 16 captures.
The fidelity values above came from a separate comparison streamed eight rows at a time.

With the native checkpoint in `TF_INKLING2_LOOP_DIR`, `python3 campaign/heavy.py -- zig build test-inkling2-loop
-Dcpu=apple_m1 -j1` exited 0. With `TF_INKLING2_LOOP_OQ4E_DIR` set to the oQ4e checkpoint, the focused
`python3 campaign/heavy.py -- zig build test-inkling2-loop-affine -Dcpu=apple_m1 -j1` exited 0. The latter checks
8-bit dense and 4-bit expert projections against dequantized FP64 weights and checks finite oQ4e decode logits.
The aligned oQ4e one-token probe peaked at 20.6 GB charged under the 28 GB cap.

For `campaign/prompts/inkling/code-1024.json`, `greedy-decode 512` exited 0. Against the MTP-off oMLX trial in
`campaign/comparators/omlx/results/code-1024-1795386552036119000.json`, the first 45 IDs agree and 53 of 512 IDs
agree by position. At zero-based position 45, TensorFold returns 170289 and oMLX returns 68014. The first 32
TensorFold IDs detokenize to a coherent response to the prompt. The run peaked at 23.4 GB charged, with host memory
pressure level 2 and no increase in swap. This greedy comparison is not a gated throughput measurement.

The first 1K native timing passed the load checks but overlapped another lane's compilation and was discarded.
The first width-1 batch attempt also overlapped compilation and failed the CPU post-check; it was discarded.

## Gated single-stream measurements

The host was an Apple M5 Max with 48 GiB memory, macOS 27.2 and Zig 0.17.1. The runner was built before timing.
Each measurement ran under `heavy.py`, with `gpu-gate-ready -1` and `perf-gate-ready` before the runner,
`perf-gate-ready -1` immediately after, and a one-second process monitor rejecting compiler overlap. All 12
retained runs passed; the order was native/oQ4e, oQ4e/native, native/oQ4e at each input length. `bench 32 4 512`
uses greedy decoding and a prefill chunk of 512. The 1K input is `campaign/prompts/inkling/code-1024.json`; the 4K
input is the first 4,096 tokens obtained by tokenizing `campaign/prompts/dev/code-4096.txt` with the runner.

| Input | Pair | Native prefill tok/s | oQ4e prefill tok/s | Native decode tok/s | oQ4e decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | 222.1 | 79.1 | 41.31 | 25.37 |
| 1K | 2 | 153.4 | 79.5 | 41.14 | 25.33 |
| 1K | 3 | 422.4 | 79.2 | 41.23 | 25.32 |
| 4K | 1 | 327.7 | 69.0 | 38.34 | 23.67 |
| 4K | 2 | 307.2 | 72.8 | 36.89 | 23.51 |
| 4K | 3 | 368.6 | 73.1 | 37.14 | 23.76 |

The respective median native and oQ4e rates are 222.1 and 79.2 prefill and 41.23 and 25.33 decode at 1K;
at 4K they are 327.7 and 72.8 prefill and 37.14 and 23.67 decode. Native 1K prefill varied widely even in
admitted runs. These compare different checkpoints and implementations; no profile established the limiting cause.
The largest charged peak in these paired runs was 21.1 GB.

## Batch measurements

The same admission, post-check and compile monitor were used with `batchbench K 5` on the 1K input. Five
alternating batch/serial pairs followed warmup. Times are median milliseconds per step with observed ranges;
throughput is tokens per second.

| K | Batch ms (range) | Serial ms (range) | Batch tok/s | Serial tok/s | Peak charged GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.014 (40.751–41.437) | 40.282 (40.254–40.824) | 24.382 | 24.825 | 17.8 |
| 2 | 64.220 (63.773–64.292) | 81.169 (81.074–81.341) | 31.143 | 24.640 | 18.5 |
| 4 | 110.553 (110.466–110.768) | 163.555 (162.357–164.023) | 36.182 | 24.457 | 19.7 |
| 8 | 202.581 (202.003–202.990) | 325.737 (325.191–326.308) | 39.490 | 24.560 | 22.1 |

The largest observed charge across these measurements and the greedy comparison was 23.4 GB, below the 28 GB
cap. The batch benchmark measures steps; it does not separately prove multi-step concurrent output equivalence.

## Reproduction

Set `PATH=/opt/homebrew/bin:$PATH`, `ZIG_GLOBAL_CACHE_DIR` to the shared Zig cache, `TMPDIR` to this worktree's
`.work/tmp`, `INKLING_LOAD_LIMIT_GB=26`, `GPU_GATE_MAX_GPU_PCT=60`, and `GPU_GATE_MAX_UTIL_PCT=60`. Set
`NATIVE_MODEL` and `OQ_MODEL` to the pinned checkpoint directories and `ORACLE_DIR` to the saved f32 oracle.
All model-resident commands below use `heavy.py`; the parity makefile invokes it for the captures.

```sh
TF_INKLING2_LOOP_DIR="$NATIVE_MODEL" python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1
TF_INKLING2_LOOP_OQ4E_DIR="$OQ_MODEL" python3 campaign/heavy.py -- zig build test-inkling2-loop-affine -Dcpu=apple_m1 -j1
zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/oq-final-build"
python3 campaign/heavy.py -- .work/tmp/oq-final-build/bin/tf-inkling2-loop-run "$OQ_MODEL" campaign/prompts/inkling/code-1024.json greedy-decode 512 .work/tmp/oq-cert-greedy.bin
make -C campaign -f parity/inkling2-mini/parity.mk parity-inkling2-mini RUNNER_MODE=prefill PREFILL_CHUNK=512 LANE=h3-oq4e-cert REPORT="$PWD/.work/tmp/oq-cert-parity.md" METAL_LOGITS="$PWD/.work/tmp/oq-cert-parity" WORK="$PWD/.work" REF_LOGITS="$ORACLE_DIR" REF_UV=env REF_VENV="''" RUNNER="$PWD/.work/tmp/oq-final-build/bin/tf-inkling2-loop-run" INKLING_DIR="$OQ_MODEL"
```

The parity command needs the oracle environment's Python on `PATH`. Compare each generated `.bin` with the
corresponding oracle `.npy` by memory-mapping both, reading eight rows at a time, counting non-finite candidate
logits and top-1 matches, and computing f64 softmax KL from reference to candidate. Mean KL above is the mean of
the 16 per-history row means, not a row-weighted mean. Count the confident subset using the f32 reference's
top-two logit margin of at least 0.1. Compare the greedy binary IDs with the `generated_ids` in the cited oMLX receipt.

For timing, invoke `python3 campaign/heavy.py --` around the built runner with `"$MODEL" "$IDS" bench 32 4 512`
or `"$OQ_MODEL" campaign/prompts/inkling/code-1024.json batchbench "$K" 5`. Run the stated GPU and CPU admission
checks and immediate CPU post-check around each invocation, monitor compiler processes during the run, and reject
a failed check or compiler overlap. The runner must be compiled before the measurement window.
