# Inkling 2 Mini fast prefill

The opt-in `INKLING_PROFILE=fast` uses half fragments for routed experts and
tiled attention, and bf16 operands for Metal 4 dense TensorOps. Accumulation,
residuals, norms, routing and KV storage remain f32. Unset and `f32` use the
existing arithmetic. Any other value fails. Values outside the half range are
saturated, and the runner reports their conversion count per chunk. All
reported packed-runner parity chunks had zero saturations. Expert weights are
decoded within the matmul rather than staged to a bf16 buffer.

The new kernels are unregistered with the kernel contract. Registration is a
family-wide promotion requirement, deferred by the campaign. Quality and Pi
are left to the parent campaign.

## Correctness

Zig 0.17.1 on Apple M5 Max and macOS 27.2 used the Mini checkpoint with
`INKLING_LOAD_LIMIT_GB=26`. The serialized family test and host test exited 0.
The family test compares half-rounded expert operands against FP64 for both
NVFP4 and MXFP8, including varied values and partial tiles. It checks a
negative control and half-range saturation. Dense TensorOps and half
attention have rounded-operand reference tests, including causal and window
masking across tile boundaries.

The default prefill parity command exited 2, as expected for explained
exceptions, and reported **14 within contract, two explained, zero
unexplained**. The packed runner repeated 14/2/0 after a memory-pressure
retry. The saved oracle verified all 16 histories. The fast capture also
exited 2 and reported zero within contract, two explained and 14 unexplained;
the f32 agreement is reported rather than gated for fast.

A streamed comparison of the packed fast outputs found **zero non-finite
logits**, 2,765 agreements in 2,862 positions with f32 top-two margin at least
0.1 (**96.61%**), and mean KL **0.0167438**, averaged across histories. Overall
top-1 agreement was 94.79%. The fast profile has lower fidelity; task quality
is not established by this receipt.

From the lane worktree, with the primary checkout three levels above it:

```sh
export PATH=/opt/homebrew/bin:$PATH
export ZIG_GLOBAL_CACHE_DIR="$PWD/../../../.work/zig-global-cache"
export TMPDIR="$PWD/.work/tmp"
export INKLING_LOAD_LIMIT_GB=26
export TF_INKLING2_LOOP_DIR="$HOME/models/Inkling-2-Mini@ed40e7da625f"
python3 ../../../campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1
zig build test-inkling2-loop-host -j1
make -C campaign -f parity/inkling2-mini/parity.mk parity-inkling2-mini RUNNER_MODE=prefill PREFILL_CHUNK=512 LANE=h1-default REPORT="$PWD/.work/tmp/h1-default-512.md"
INKLING_PROFILE=fast make -C campaign -f parity/inkling2-mini/parity.mk parity-inkling2-mini RUNNER_MODE=prefill PREFILL_CHUNK=512 LANE=h1-packed-fast REPORT="$PWD/.work/tmp/h1-packed-fast-512.md" RUNNER="$PWD/.work/tmp/h1-packed-build/bin/tf-inkling2-loop-run"
```

The packed runner was built with `zig build tf-inkling2-loop-run -Dcpu=apple_m1
-j1 --prefix "$PWD/.work/tmp/h1-packed-build"`. The fast and default parity
captures use distinct lanes. The first packed capture attempts exceeded the
28 GB `heavy.py` charge cap during host wired-memory growth; the retries
completed. The reference environment can be kept in scratch with
`UV_PROJECT_ENVIRONMENT="$PWD/.work/tmp/ref-venv"`.

## Paired prefill

The busy-host measurements used chunk 512 and `bench 2 1 512`, with f32,
fast, then f32 in each bracket. Each bracket ran inside `heavy.py`, with
`GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1` and
`perf-gate-ready` before, and `perf-gate-ready -1` afterward. Attempts that
failed admission or postcheck were discarded. The paired ratio is mean
adjacent-control time divided by fast time. Times are milliseconds.

| Prompt | Control 1 | Fast | Control 2 |
| --- | ---: | ---: | ---: |
| 1,139 tokens | 2938.3 | 2240.5 | 3574.5 |
| 1,139 tokens | 2879.1 | 2180.2 | 2931.6 |
| 1,139 tokens | 2887.5 | 2226.2 | 2918.9 |
| 1,139 tokens | 2891.1 | 2187.9 | 3052.6 |
| 1,139 tokens | 2870.8 | 2164.9 | 2874.2 |
| 5,265 tokens | 13452.6 | 9958.4 | 14365.8 |
| 5,265 tokens | 14814.8 | 10416.8 | 14965.2 |
| 5,265 tokens | 13640.4 | 10383.2 | 15132.6 |
| 5,265 tokens | 15413.3 | 10609.9 | 15272.4 |

For 1K, the five paired ratios have median **1.333** and range **1.304–1.453**.
Control times had median 2,905.0 ms and range 2,870.8–3,574.5 ms; fast times
had median 2,187.9 ms and range 2,164.9–2,240.5 ms. For 4K, the four paired
ratios have median **1.413** and range **1.386–1.446**. Control times had median
14,890.0 ms and range 13,452.6–15,413.3 ms; fast times had median 10,400.0 ms
and range 9,958.4–10,609.9 ms.
Across these brackets the highest `heavy.py` charged peak was **25.5 GB**.
The charge includes host-wide wired growth, so it does not isolate the
individual profile's allocation. The host was shared and some first controls
incurred cold reads.

The runner and inputs were `./.work/tmp/h1-packed-build/bin/tf-inkling2-loop-run`,
`../../../.work/tmp/ids-1k.json`, and `../../../.work/tmp/p3-4096.json`.
Each bracket used this command structure inside the GPU lock:

```sh
python3 ../../../campaign/heavy.py -- sh -c '
GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 || exit 75
perf-gate-ready || exit 75
for profile in f32 fast f32; do
  INKLING_PROFILE="$profile" ./.work/tmp/h1-packed-build/bin/tf-inkling2-loop-run "$1" "$2" bench 2 1 512 || exit "$?"
done
perf-gate-ready -1 || exit 76
' h1 "$TF_INKLING2_LOOP_DIR" ../../../.work/tmp/ids-1k.json
```

Use `../../../.work/tmp/p3-4096.json` for the 4K bracket. The input files are
campaign scratch fixtures. These commands regenerate identical token IDs from
tracked prompts:

```sh
./.work/tmp/h1-packed-build/bin/tf-inkling2-loop-run "$TF_INKLING2_LOOP_DIR/tokenizer.json" campaign/prompts/dev/prose-1024.txt tokenize ../../../.work/tmp/ids-1k.json
./.work/tmp/h1-packed-build/bin/tf-inkling2-loop-run "$TF_INKLING2_LOOP_DIR/tokenizer.json" campaign/prompts/dev/code-4096.txt tokenize ../../../.work/tmp/p3-4096.json
```

The regenerated arrays were compared with both measured fixtures and were
identical: 1,139 prose tokens and 5,265 code tokens.

Five 1K brackets isolated dense bf16 TensorOps against the fragment-only
runner. Their median paired ratio was 1.047, range 1.040–1.264. Five isolated
attention brackets gave median 1.064, range 1.028–1.336. All candidates beat
both adjacent controls; cold first controls widen the ranges. The isolated
screens used the earlier fragment layout, so their sizes are indicative rather
than an attribution of the packed runner.

## Split timeline

The packed runner's instrumented 5,265-token run passed the same admission and
postcheck. The split timeline had 115 segments, 9,366 dispatches and 999.397 ms
of summed GPU span. Expert matrix kernels accounted for 457.567 ms, half
attention for 147.434 ms, bf16 TensorOps for 49.140 ms, and grouped expert
gate/up and down for 80.608 and 59.975 ms. The instrumented spans are
attribution, not paired end-to-end timings.

```sh
python3 ../../../campaign/heavy.py -- sh -c '
GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 || exit 75
perf-gate-ready || exit 75
TF_METAL4_TIMELINE=1 TF_INKLING_PREFILL_PROFILE_BASE=0 INKLING_PROFILE=fast ./.work/tmp/h1-packed-build/bin/tf-inkling2-loop-run "$1" "$2" bench 2 1 512 || exit "$?"
perf-gate-ready -1 || exit 76
' h1 "$TF_INKLING2_LOOP_DIR" ../../../.work/tmp/p3-4096.json
python3 campaign/kernels/inkling_profile.py timeline --split --log .work/tmp/h1-packed-timeline.log
```
