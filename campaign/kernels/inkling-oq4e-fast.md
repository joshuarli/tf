# Inkling 2 Mini oQ4e affine kernels

Status: the final candidate is `2ea63f495`. The requested 1K absolute thresholds and median checkpoint comparison
pass. The 4K prefill medians differ by 0.2 tok/s, inside the observed spread. Native 1K prefill varies widely.
The first correctness and timing sections describe the earlier `1d2539780` candidate.

The host is an Apple M5 Max with 48 GiB RAM. The native checkpoint is `Inkling-2-Mini@ed40e7da625f` and the
oQ4e checkpoint is `Inkling-2-Mini-oQ4e-mtp@3eea042751cd`. The runner uses f32 activations and greedy decode.

## Correctness

`python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0. The host suite passed 31/31
and the GPU suite passed 71/71 tests. The GPU suite peaked at 23.5 GB charged under the 28 GB cap.

The saved oracle had 16 histories and 3,015 rows. The streamed float64 comparison found 2,773 top-1 matches
(91.973%), mean KL 0.03929142 averaged across histories, and zero non-finite logits. The oracle verifier exited 0.
The parity make target exited 2 because its f32-versus-f32 contract reports 14 unexplained differences across the
unequal checkpoints. The independent top-1 and KL thresholds passed.

## Single-stream timings

Each retained run passed `GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1`, then
`perf-gate-ready` before the runner, and `perf-gate-ready -1` immediately afterward. Each ran under `heavy.py`
with `INKLING_LOAD_LIMIT_GB=26` and a 28 GB cap. The runner was built before timing. No compiler was observed
during the queued measurements. The first native 1K run had no compiler monitor. The order was native/oQ4e,
oQ4e/native, native/oQ4e. Decode is the median of 32 greedy steps after four warmups. Prefill uses chunk 512.

| Input | Pair | Native prefill tok/s | oQ4e prefill tok/s | Native decode ms/step | oQ4e decode ms/step | Peak native/oQ4e GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | 217.7 | 174.2 | 25.048 | 29.010 | 20.9 / 17.6 |
| 1K | 2 | 213.7 | 177.1 | 25.047 | 29.033 | 20.9 / 17.6 |
| 1K | 3 | 407.1 | 172.4 | 25.081 | 29.117 | 20.9 / 20.3 |
| 4K | 1 | 341.2 | 165.7 | 26.895 | 30.854 | 21.1 / 20.7 |
| 4K | 2 | 323.3 | 158.6 | 27.039 | 31.071 | 21.1 / 17.9 |
| 4K | 3 | 396.1 | 171.3 | 26.978 | 31.002 | 21.5 / 20.1 |

The oQ4e 1K medians are 174.2 prefill tok/s and 34.44 decode tok/s. The requested minima are 222 and 41.
One oQ4e 1K attempt and one width-8 batch attempt failed the CPU post-check. Two native 1K attempts failed
CPU admission. An earlier native attempt also failed CPU admission. Those attempts are excluded.

## Batch timings

`batchbench K 5` alternates five batch and serial steps after warmup on the 1K prompt. All retained runs exited 0.

| K | Batch median ms | Batch range ms | Serial median ms | Batch tok/s | Peak GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29.521 | 29.518–29.681 | 28.947 | 33.874 | 17.8 |
| 2 | 43.400 | 43.253–43.432 | 58.265 | 46.083 | 18.4 |
| 4 | 71.402 | 70.471–71.594 | 117.837 | 56.021 | 19.7 |
| 8 | 125.774 | 124.863–125.855 | 233.798 | 63.606 | 22.1 |

## Diagnostic profile

An instrumented `TF_METAL4_TIMELINE=4` decode run exited 0 and passed the gates. The timeline parser reported
43.451 ms per instrumented round, including 9.483 ms in 353 `i2a_dense` calls and 7.379 ms in 336 `i2a_float`
calls. `i2a_expert_up` and `i2a_expert_down` together accounted for 0.630 ms. Instrumentation changes timing;
these values attribute work and are not throughput measurements.

## Commands

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp"
export ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26
export GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60
/opt/homebrew/bin/zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/h5-build"
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh \
  "$PWD/.work/tmp/h5-build/bin/tf-inkling2-loop-run" "$MODEL" "$IDS" bench 32 4 512
python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh \
  "$PWD/.work/tmp/h5-build/bin/tf-inkling2-loop-run" "$OQ_MODEL" campaign/prompts/inkling/code-1024.json batchbench "$K" 5
TF_METAL4_TIMELINE=4 python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh \
  "$PWD/.work/tmp/h5-build/bin/tf-inkling2-loop-run" "$OQ_MODEL" campaign/prompts/inkling/code-1024.json bench 14 2 512
python3 campaign/kernels/inkling_profile.py timeline --log .work/tmp/h5-profile.log
```

`MODEL` selects either checkpoint. `IDS` is `campaign/prompts/inkling/code-1024.json` for 1K and the 4,096-token
fixture `.work/tmp/fixtures/code-4096.json` in the main checkout for 4K. The fixture is the first 4,096 tokens
of `campaign/prompts/dev/code-4096.txt`. `K` is 1, 2, 4, or 8. The runner build and timeline parser exited 0.

The fidelity capture used the saved oracle and the command in `campaign/kernels/inkling-oq4e.md`, with
`RUNNER_MODE=prefill`, `PREFILL_CHUNK=512`, the oQ4e checkpoint, and this runner. A separate comparison mapped
the f32 reference `.npy` and candidate `.bin`, read eight rows at a time, used f64 log-softmax for KL, and
averaged the 16 history means. It exited 0. No f32 oracle model was run.

The exact capture and streamed comparison commands were:

```sh
export PATH="/Users/josh/repos/tf/.work/venv-ref/bin:/opt/homebrew/bin:$PATH"
make -C campaign -f parity/inkling2-mini/parity.mk parity-inkling2-mini \
  RUNNER_MODE=prefill PREFILL_CHUNK=512 LANE=h5-affine-fast \
  REPORT="$PWD/.work/tmp/h5-parity.md" METAL_LOGITS="$PWD/.work/tmp/h5-parity" WORK="$PWD/.work" \
  REF_LOGITS="/Users/josh/repos/tf/.work/parity/inkling2-mini/ref" REF_UV=env "REF_VENV=''" \
  RUNNER="$PWD/.work/tmp/h5-build/bin/tf-inkling2-loop-run" \
  INKLING_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
/Users/josh/repos/tf/.work/venv-ref/bin/python .work/tmp/h5-compare.py | tee .work/tmp/h5-compare.log
```

The comparison script used for that run was:

```python
import json
from pathlib import Path
import numpy as np

histories = json.loads(Path('campaign/parity/inkling2-mini/histories.json').read_text())
refdir = Path('/Users/josh/repos/tf/.work/parity/inkling2-mini/ref')
outdir = Path('.work/tmp/h5-parity')
total = matches = nonfinite = 0
means = []
for history in histories:
    name = history['name']
    ref = np.load(refdir / f'{name}.npy', mmap_mode='r')
    out = np.memmap(outdir / f'{name}.bin', dtype=np.float32, mode='r', shape=ref.shape)
    count = hit = bad = 0
    kl_sum = 0.0
    for i in range(0, ref.shape[0], 8):
        a = np.asarray(ref[i:i+8], dtype=np.float64)
        b = np.asarray(out[i:i+8], dtype=np.float64)
        bad += np.count_nonzero(~np.isfinite(b))
        hit += np.count_nonzero(np.argmax(a, axis=1) == np.argmax(b, axis=1))
        count += len(a)
        a -= np.max(a, axis=1, keepdims=True)
        b -= np.max(b, axis=1, keepdims=True)
        a -= np.log(np.exp(a).sum(axis=1, keepdims=True))
        b -= np.log(np.exp(b).sum(axis=1, keepdims=True))
        kl_sum += np.sum(np.exp(a) * (a - b))
    total += count
    matches += hit
    nonfinite += bad
    means.append(kl_sum / count)
    print(name, 'rows', count, 'top1', hit, 'mean_kl', kl_sum / count, 'nonfinite', bad, flush=True)
print('TOTAL', total, 'TOP1', matches, 'AGREEMENT', matches / total, 'MEAN_KL', np.mean(means), 'NONFINITE', nonfinite)
```

## Float projection experiment

Commit `baa5d3648` changes the f32 projection to four-value loads. The focused affine check and the host/GPU
suite exited 0. Three gated pairs compared the same oQ4e checkpoint on the `1d2539780` runner and the vector
runner. Both runners were built before timing. Each run used the single-stream command above with the 1K prompt.

| Pair | Original decode ms | Vector decode ms | Original prefill tok/s | Vector prefill tok/s | Peak original/vector GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29.097 | 25.586 | 172.8 | 179.3 | 20.0 / 18.0 |
| 2 | 29.115 | 25.522 | 171.8 | 178.2 | 17.6 / 17.6 |
| 3 | 29.115 | 25.543 | 168.3 | 167.5 | 17.6 / 18.0 |

The vector runner used `zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix
"$PWD/.work/tmp/h5-vector-build"`, which exited 0. The median decode time fell from 29.115 ms to 25.543 ms;
1K prefill remains below the requested minimum. These comparisons do not replace the complete six-pair
native-versus-oQ4e acceptance matrix or fidelity check for the final revision.

## Tiled prefill experiment

Commit `d62d1c4b9` changes the many-row dense affine path to dequantize tiles in threadgroup memory and use
float simdgroup matrix accumulation. The focused affine check and host/GPU suite exited 0. Three gated pairs
compared the vector runner from `baa5d3648` with a separate matrix runner on oQ4e at 1K.

| Pair | Vector prefill tok/s | Matrix prefill tok/s | Vector decode ms | Matrix decode ms | Peak vector/matrix GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 171.8 | 299.4 | 25.559 | 25.517 | 19.8 / 17.6 |
| 2 | 179.3 | 299.7 | 25.502 | 25.505 | 17.6 / 17.6 |
| 3 | 175.0 | 294.9 | 25.485 | 25.511 | 17.6 / 17.6 |

The matrix runner used `zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix
"$PWD/.work/tmp/h5-matrix-build"`, which exited 0. Median prefill rose from 175.0 to 299.4 tok/s. Decode did
not materially change. The experiment still requires the final native comparison matrix and streamed fidelity run.

## Rejected packed-dot experiment

A local variant accumulated code and activation sums before applying each packed word's scale and bias. The
focused affine check and host/GPU suite exited 0. Three gated 1K oQ4e pairs compared the matrix runner with this
variant. All six runs exited 0 with no compiler overlap. The variant was rejected because decode regressed.

| Pair | Matrix decode ms | Variant decode ms | Matrix prefill tok/s | Variant prefill tok/s | Peak matrix/variant GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.500 | 26.047 | 285.4 | 298.4 | 20.3 / 17.6 |
| 2 | 25.529 | 25.983 | 300.5 | 299.7 | 17.6 / 17.6 |
| 3 | 25.484 | 26.069 | 299.6 | 299.7 | 17.6 / 17.6 |

The variant runner used `zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix
"$PWD/.work/tmp/h5-dot-build"`, which exited 0. The benchmark command and gate settings match the prior pairs.

## Vector unpack experiment

Commit `3ad57dee6` uses four-value input loads and vector code extraction in the affine dot product. The focused
affine check and host/GPU suite exited 0. Three gated 1K oQ4e pairs compared the matrix runner with the vector
unpack runner. All six runs exited 0 with no compiler overlap.

| Pair | Matrix decode ms | Vector unpack decode ms | Matrix prefill tok/s | Vector unpack prefill tok/s | Peak matrix/vector GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.554 | 23.934 | 285.2 | 298.9 | 19.9 / 17.6 |
| 2 | 25.465 | 23.897 | 301.0 | 298.3 | 19.8 / 18.2 |
| 3 | 25.514 | 23.904 | 299.5 | 300.8 | 20.1 / 19.6 |

The vector unpack runner used `zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix
"$PWD/.work/tmp/h5-pack-build"`, which exited 0. Median decode improved from 25.514 ms to 23.904 ms.

## Checkpoint comparisons for `3ad57dee6`

All retained cells used the gate and `heavy.py` command above with `h5-pack-build`. Each exited 0 and the
compiler monitor reported no overlap. The native 1K first attempt and oQ4e 1K third-pair first attempt failed
the CPU post-check. Two native first-pair retries failed admission. Those runs are excluded.

| Input | Pair | Native prefill tok/s | oQ4e prefill tok/s | Native decode ms | oQ4e decode ms | Peak native/oQ4e GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | 407.5 | 277.1 | 25.047 | 23.952 | 20.9 / 20.4 |
| 1K | 2 | 216.1 | 288.9 | 25.093 | 23.877 | 20.9 / 19.7 |
| 1K | 3 | 412.0 | 298.4 | 25.107 | 23.908 | 20.9 / 17.6 |
| 4K | 1 | 340.3 | 300.8 | 26.948 | 25.831 | 21.1 / 20.4 |
| 4K | 2 | 310.2 | 284.2 | 27.048 | 25.878 | 21.1 / 17.9 |
| 4K | 3 | 370.8 | 301.6 | 27.149 | 25.818 | 21.1 / 20.5 |

At 1K, the oQ4e median is 288.9 prefill tok/s and 41.83 decode tok/s. Native prefill has a wide
216.1–412.0 tok/s range and a 407.5 median. The absolute 222 and 41 thresholds pass; oQ4e prefill does not
match the measured native median. At 4K, oQ4e prefill is below native in all three pairs.

| K | Batch median ms | Batch range ms | Serial median ms | Batch tok/s | Peak GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24.776 | 24.542–24.822 | 23.889 | 40.362 | 20.3 |
| 2 | 34.671 | 34.472–34.729 | 47.434 | 57.685 | 18.4 |
| 4 | 58.194 | 57.881–58.339 | 97.069 | 68.736 | 19.7 |
| 8 | 104.822 | 104.264–106.609 | 192.885 | 76.320 | 22.1 |

The streamed fidelity rerun for `3ad57dee6` found 2,775 of 3,015 matching top tokens (92.0398%), mean KL
0.03927278, and zero non-finite logits. The oracle verifier checked all 16 saved histories. The capture command
above used `LANE=h5-final`, `REPORT="$PWD/.work/tmp/h5-final-parity.md"`,
`METAL_LOGITS="$PWD/.work/tmp/h5-final-parity"`, and the `h5-pack-build` runner. The parity make command exited 2
under its f32-versus-f32 contract. The streamed comparison exited 0 using the same script above with `outdir` set
to `.work/tmp/h5-final-parity`. Peak charged memory during captures was 20.5 GB under the 28 GB cap.

## Rejected f32 row-reuse experiment

A local f32 matrix kernel reused weights across prompt rows. Its focused affine check and host/GPU suite exited 0.
Three gated 1K oQ4e pairs compared `3ad57dee6` with the variant. All six runs exited 0 with no compiler overlap.

| Pair | Original prefill tok/s | Variant prefill tok/s | Original decode ms | Variant decode ms | Peak original/variant GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 286.8 | 306.3 | 23.960 | 23.917 | 19.8 / 17.6 |
| 2 | 300.9 | 305.3 | 23.924 | 23.930 | 17.6 / 17.6 |
| 3 | 300.8 | 305.1 | 23.920 | 23.964 | 17.6 / 17.6 |

The variant runner used `zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix
"$PWD/.work/tmp/h5-floatrows-build"`, which exited 0. Its median prefill gain was 1.5%, below the campaign's
3% keep criterion. The added kernel was rejected. The first original run had approximately 991,000 pageins;
the next two original runs had 144 and 104.

## Prefill profile for `3ad57dee6`

A separate instrumented prefill used the 4K fixture followed by its first 1,024 IDs, chunk 512, and the command
below. Both gates and the command exited 0. `heavy.py` reported 17.7 GB peak charged. The split parser reported
1,791.090 ms of summed GPU span across 115 segments. The largest entries were native NVFP4 expert matrix
605.590 ms, affine dense rows 323.475 ms, attention 272.709 ms, affine grouped up 176.573 ms, and affine grouped
down 108.688 ms. The profile attributes work; it is not an end-to-end throughput measurement.

```sh
TF_METAL4_TIMELINE=1 TF_INKLING_PREFILL_PROFILE_BASE=0 python3 campaign/heavy.py -- \
  sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh \
  "$PWD/.work/tmp/h5-pack-build/bin/tf-inkling2-loop-run" "$OQ_MODEL" \
  "$PWD/.work/tmp/h5-profile-5120.json" bench 2 1 512
python3 campaign/kernels/inkling_profile.py timeline --split --log .work/tmp/h5-prefill-profile.log
```

## Grouped expert matrix experiment

Commit `2ea63f495` tiles grouped affine expert up and down projections across routed rows. The focused check,
including a 68-column tile-tail fixture, and the host/GPU suite exited 0. Three gated 1K oQ4e pairs compared
`3ad57dee6` with the grouped matrix runner. All six runs exited 0 without compiler overlap.

| Pair | Original prefill tok/s | Grouped matrix prefill tok/s | Original decode ms | Grouped matrix decode ms | Peak original/matrix GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 277.8 | 341.7 | 23.919 | 23.926 | 20.6 / 17.6 |
| 2 | 298.7 | 342.2 | 23.935 | 23.864 | 17.6 / 17.6 |
| 3 | 301.0 | 340.5 | 23.934 | 23.880 | 17.6 / 17.6 |

The candidate runner used `zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix
"$PWD/.work/tmp/h5-expertmatrix-build"`, which exited 0. Median prefill rose from 298.7 to 341.7 tok/s.

For `2ea63f495`, the full suite command exited 0 with 24/24 host and 71/71 GPU tests. Peak charged memory was
23.5 GB. The expanded affine fixture covers 68 output columns as well as rows 1, 4, 16, and 256.

The saved-oracle capture for this revision used `LANE=h5-expert-final`,
`REPORT="$PWD/.work/tmp/h5-expert-final-parity.md"`, `METAL_LOGITS="$PWD/.work/tmp/h5-expert-final-parity"`,
and the `h5-expertmatrix-build` runner in the capture command above. The oracle verifier checked 16 histories.
The make command exited 2 on its stricter f32-versus-f32 contract. The streamed script above, with `outdir` set
to `.work/tmp/h5-expert-final-parity`, exited 0. It found 2,775/3,015 top-1 matches (92.0398%), mean KL
0.03927280, and zero non-finite logits. The largest capture peak was 20.2 GB.

## Final checkpoint comparison for `2ea63f495`

All retained cells exited 0 with no observed compiler overlap. Every model ran under `heavy.py`, the 28 GB cap,
and `INKLING_LOAD_LIMIT_GB=26`. The GPU admission, CPU admission, and immediate CPU post-check described above
passed for every cell. This final matrix required no rejected attempts. The order was native/oQ4e, oQ4e/native,
then native/oQ4e. Decode is the median of 32 steps after four warmups.

| Input | Pair | Native prefill tok/s | oQ4e prefill tok/s | Native decode ms | oQ4e decode ms | Peak native/oQ4e GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | 219.7 | 316.6 | 25.109 | 23.924 | 20.9 / 20.4 |
| 1K | 2 | 223.1 | 341.9 | 25.123 | 23.916 | 20.9 / 17.6 |
| 1K | 3 | 413.2 | 328.5 | 25.081 | 23.984 | 20.9 / 20.6 |
| 4K | 1 | 345.5 | 345.3 | 26.887 | 25.854 | 21.1 / 20.0 |
| 4K | 2 | 309.2 | 319.5 | 27.086 | 25.760 | 21.1 / 17.9 |
| 4K | 3 | 365.5 | 345.7 | 27.060 | 25.841 | 21.1 / 20.3 |

The 1K median prefill rates are 223.1 native and 328.5 oQ4e tok/s. Median decode rates are 39.83 native and
41.80 oQ4e tok/s. Native prefill ranges from 219.7 to 413.2 tok/s: the first two runs incurred over one million
pageins, and the third incurred 97. This spread limits any claim about warm-cache parity. The 4K median prefill
rates are 345.5 native and 345.3 oQ4e tok/s. The 0.2 tok/s difference is within the observed spread. The 4K
median decode rates are 36.95 native and 38.70 oQ4e tok/s.

| K | Batch median ms | Batch range ms | Serial median ms | Batch tok/s | Peak GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24.677 | 24.560–24.776 | 23.904 | 40.524 | 17.8 |
| 2 | 34.579 | 34.501–34.731 | 47.432 | 57.839 | 18.4 |
| 4 | 57.962 | 57.597–58.108 | 97.148 | 69.011 | 19.6 |
| 8 | 104.501 | 103.826–109.841 | 199.427 | 76.554 | 22.1 |

The final timing command is the single-stream or batch command above with the following values. All twelve
single-stream runs and four batch runs exited 0. The same runner was used for both checkpoints.

```sh
NATIVE_MODEL="$HOME/models/Inkling-2-Mini@ed40e7da625f"
OQ_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
IDS_1K="$PWD/campaign/prompts/inkling/code-1024.json"
IDS_4K="/Users/josh/repos/tf/.work/tmp/fixtures/code-4096.json"
RUNNER="$PWD/.work/tmp/h5-expertmatrix-build/bin/tf-inkling2-loop-run"
python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh \
  "$RUNNER" "$MODEL" "$IDS" bench 32 4 512
python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh \
  "$RUNNER" "$OQ_MODEL" "$IDS_1K" batchbench "$K" 5
```

Set `MODEL` to each checkpoint, `IDS` to the relevant prompt, and `K` to 1, 2, 4, and 8. The candidate and
initial runners each measured 1,862,456 bytes by `stat`; no new dependency was added. The final suite peak was
23.5 GB, the final fidelity capture peak was 20.2 GB, and the largest final timing peak was 22.1 GB.
