# Inkling oQ4e decode profile

The baseline is `9a0a513ad2fd9091bb1e9691a37733db32f7ee40` on an M5 Max. The checkpoint is
`Inkling-2-Mini-oQ4e-mtp@3eea042751cd`. The 1,024-token input is `campaign/prompts/inkling/code-1024.json`.
The runner uses greedy tokens and prefill chunks of 512. Metal 4 timestamps instrument every dispatch, so these
spans attribute work and do not establish uninstrumented throughput.

The 1K profile passed GPU admission, CPU admission, and the CPU post-check. The command exited 0. The timeline
collected four rounds after eight timeline warmups. Each round had 2,966 dispatches and a 37.789 ms GPU span.
The largest kernel groups were:

| Kernel | Calls per step | GPU ms per step |
| --- | ---: | ---: |
| `i2a_dense` | 353 | 8.046 |
| `i2a_float` | 336 | 3.905 |
| `i2_attn_split` | 48 | 3.888 |
| `i2_rms` | 513 | 3.831 |
| `i2_copy` | 499 | 3.339 |
| `i2_gu_table` | 44 | 2.481 |
| `i2_conv` | 192 | 2.057 |
| `i2_down_table` | 44 | 1.992 |
| `i2_mix` | 96 | 1.765 |
| `i2_attn_combine` | 48 | 1.397 |

The process-tree peak charged by `heavy.py` was 17.7 GB. This is not total device residency. The cache uses
65,536 global KV bytes per token and 8,192 bytes for each slot's key and value row together. Convolution rings
are additional. No cache change was made.

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp"
export ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26
export GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60
/opt/homebrew/bin/zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/base-build"
TF_METAL4_TIMELINE=4 python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh \
  "$PWD/.work/tmp/base-build/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" \
  campaign/prompts/inkling/code-1024.json bench 14 2 512
python3 campaign/kernels/inkling_profile.py timeline --log .work/tmp/base-profile-1k-fourth.log
```

The build and parser exited 0. Six initial 4K profile attempts exited 75 at CPU admission. A prior 1K model run
exited 75 at its CPU post-check and was discarded. The later results are recorded below. The baseline uses f32 KV
storage.


## Decode experiments

All profiles below use the same 1K input, four timeline rounds and eight timeline warmups. The GPU and CPU
admission checks and the CPU post-check passed. These instrumented spans attribute dispatches and do not establish
throughput. Each process ran under `campaign/heavy.py` with `INKLING_LOAD_LIMIT_GB=26`.

| Change | Dispatches per step | GPU span ms |
| --- | ---: | ---: |
| Baseline | 2,966 | 37.789 |
| Convolution and ring store together | 2,774 | 35.037 |
| Gather and normalize mix states | 2,394 | 32.917 |
| Fuse affine attention projections | 2,138 | 30.931 |
| Pair f32 controller projections | 2,042 | 30.194 |
| Remove FFN residual copies | 1,946 | 29.410 |
| Pair controller normalizations | 1,850 | 29.054 |
| Group weighted attention normalizations | 1,794 | 28.422 |
| Fuse value normalization and KV store | 1,746 | 28.511 |
| Group attention clipping | 1,650 | 27.523 |
| Fuse f32 controller projection and update | 1,554 | 27.131 |

The cumulative changes preserve f32 activations and KV storage. Four-row 8-bit matvec was rejected: its affected
profile group took 2.333 ms against 1.982 ms in the previous profile. One admitted uninstrumented pair measured
21.321 ms versus 20.966 ms per step. The pair alone does not establish typical performance.

The 64-token oQ4e greedy decode IDs matched the baseline through the controller fusion. The final host and GPU
suite passed: 25 host and 73 GPU-family tests, no skips.
The saved-oracle comparison for the final runner covered 3,015 positions and 16 histories. It found 2,775 top-1 matches (92.04%), mean KL 0.0392728 and zero non-finite logits.
The standard f32 parity command exited 2 because the two checkpoints differ.

## Per-kernel 1K profile

The baseline and final runner each used four timeline rounds after eight warmups. Times are GPU milliseconds per
decode step and calls are dispatches per step. Both runs passed the stated admission and post-checks. The final
profile returned exit 0 and the process-tree peak was measured by `heavy.py`.

| Kernel | Baseline calls | Baseline ms | Final calls | Final ms |
| --- | ---: | ---: | ---: | ---: |
| `i2a_dense` | 353 | 8.046 | 49 | 1.998 |
| `i2_attn_split` | 48 | 3.888 | 48 | 4.156 |
| `i2a_float` | 336 | 3.905 | 144 | 2.226 |
| `i2_rms` | 513 | 3.831 | 69 | 0.774 |
| `i2_copy` | 499 | 3.339 | 23 | 0.053 |
| `i2_gu_table` | 44 | 2.481 | 44 | 2.619 |
| `i2a_projections` | 0 | 0.000 | 48 | 2.221 |
| `i2_conv` | 192 | 2.057 | 192 | 2.116 |
| `i2_down_table` | 44 | 1.992 | 44 | 1.953 |
| `i2_rms_gather` | 0 | 0.000 | 216 | 1.863 |
| `i2_mix` | 96 | 1.765 | 96 | 1.454 |
| `i2_attn_combine` | 48 | 1.397 | 48 | 0.727 |
| `i2a_float_control` | 0 | 0.000 | 96 | 1.105 |
| `i2_conv_store` | 192 | 0.790 | 0 | 0.000 |
| `i2_relbias` | 48 | 0.684 | 48 | 0.370 |
| `i2_mix4` | 24 | 0.628 | 24 | 0.605 |
| `i2_clip_scaled` | 96 | 0.543 | 0 | 0.000 |
| `i2_add` | 88 | 0.202 | 88 | 0.528 |
| `i2a_expert_up` | 4 | 0.214 | 4 | 0.495 |
| `i2_combine` | 48 | 0.214 | 48 | 0.493 |
| `i2_sig_mul` | 48 | 0.484 | 48 | 0.126 |
| `i2a_expert_down` | 4 | 0.402 | 4 | 0.105 |
| `i2_route` | 48 | 0.390 | 48 | 0.386 |
| `i2_ctl` | 96 | 0.302 | 0 | 0.000 |
| `i2_rms_weighted3` | 0 | 0.000 | 28 | 0.199 |
| `i2_clip3` | 0 | 0.000 | 48 | 0.146 |
| `i2_rms_kv_store` | 0 | 0.000 | 48 | 0.137 |
| `i2_clip` | 48 | 0.112 | 0 | 0.000 |
| `i2_kv_store` | 48 | 0.112 | 0 | 0.000 |
| `i2a_embed` | 1 | 0.011 | 1 | 0.010 |

The instrumented span was 37.789 ms across 2,966 dispatches before and 26.865 ms across 1,554 dispatches after.

## Per-kernel 4K profile

Both runs passed GPU admission, CPU admission and the CPU post-check, with no observed compiler overlap.
Times are GPU milliseconds per decode step and calls are dispatches per step.

| Kernel | Baseline calls | Baseline ms | Final calls | Final ms |
| --- | ---: | ---: | ---: | ---: |
| `i2a_dense` | 353 | 7.671 | 49 | 1.954 |
| `i2_attn_split` | 48 | 5.589 | 48 | 5.947 |
| `i2_rms` | 513 | 3.713 | 69 | 0.771 |
| `i2a_float` | 336 | 3.686 | 144 | 2.217 |
| `i2_copy` | 499 | 3.361 | 23 | 0.054 |
| `i2_gu_table` | 44 | 2.315 | 44 | 2.637 |
| `i2a_projections` | 0 | 0.000 | 48 | 2.198 |
| `i2_conv` | 192 | 2.105 | 192 | 2.181 |
| `i2_down_table` | 44 | 1.926 | 44 | 1.949 |
| `i2_rms_gather` | 0 | 0.000 | 216 | 1.888 |
| `i2_mix` | 96 | 1.765 | 96 | 1.492 |
| `i2_attn_combine` | 48 | 1.347 | 48 | 0.730 |
| `i2a_float_control` | 0 | 0.000 | 96 | 1.111 |
| `i2_conv_store` | 192 | 0.804 | 0 | 0.000 |
| `i2_relbias` | 48 | 0.677 | 48 | 0.371 |
| `i2_mix4` | 24 | 0.606 | 24 | 0.607 |
| `i2_clip_scaled` | 96 | 0.540 | 0 | 0.000 |
| `i2_combine` | 48 | 0.212 | 48 | 0.539 |
| `i2_add` | 88 | 0.202 | 88 | 0.533 |
| `i2a_expert_up` | 4 | 0.199 | 4 | 0.521 |
| `i2_sig_mul` | 48 | 0.456 | 48 | 0.128 |
| `i2a_expert_down` | 4 | 0.395 | 4 | 0.103 |
| `i2_route` | 48 | 0.391 | 48 | 0.395 |
| `i2_ctl` | 96 | 0.306 | 0 | 0.000 |
| `i2_rms_weighted3` | 0 | 0.000 | 28 | 0.204 |
| `i2_clip3` | 0 | 0.000 | 48 | 0.146 |
| `i2_rms_kv_store` | 0 | 0.000 | 48 | 0.139 |
| `i2_clip` | 48 | 0.112 | 0 | 0.000 |
| `i2_kv_store` | 48 | 0.111 | 0 | 0.000 |
| `i2a_embed` | 1 | 0.011 | 1 | 0.012 |

The instrumented span was 38.495 ms across 2,966 dispatches before and 28.826 ms across 1,554 dispatches after.

## Alternating timings

The runner was built before timing. Each measurement used `bench 32 4 512` or `batchbench K 5`, greedy
decoding, the pinned oQ4e checkpoint, and the admission checks in the commands below. A one-second process monitor
discarded runs with compiler overlap. Each retained run exited 0 and passed the CPU post-check. Pair 2 reversed
the order. The 1K prompt was `campaign/prompts/inkling/code-1024.json`; 4K used the first 4,096 IDs from
`campaign/prompts/dev/code-4096.txt`.

| Input | Pair | Base prefill tok/s | Final prefill tok/s | Base decode ms | Final decode ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k | 1 | 194.9 | 312.8 | 23.230 | 19.990 |
| 1k | 2 | 318.9 | 315.8 | 23.100 | 19.969 |
| 1k | 3 | 325.9 | 330.6 | 23.044 | 19.886 |
| 4k | 1 | 346.4 | 330.9 | 24.936 | 21.741 |
| 4k | 2 | 320.0 | 323.3 | 24.998 | 21.773 |
| 4k | 3 | 315.2 | 320.0 | 24.982 | 21.824 |

Each decode figure is the median of 32 steps. The median of the three 1K run medians was 23.100 ms
(43.29 tok/s) before and 19.969 ms (50.08 tok/s) after. The 1K prefill medians were 318.9 and 315.8 tok/s,
within the observed ranges of 194.9–325.9 and 312.8–330.6 tok/s. At 4K the prefill medians were 320.0 and
323.3 tok/s.

| K | Pair | Base batch ms | Final batch ms |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 23.946 | 23.607 |
| 1 | 2 | 23.849 | 23.667 |
| 1 | 3 | 23.770 | 23.551 |
| 2 | 1 | 33.960 | 33.289 |
| 2 | 2 | 34.034 | 33.296 |
| 2 | 3 | 33.926 | 33.309 |
| 4 | 1 | 56.459 | 55.775 |
| 4 | 2 | 56.781 | 56.879 |
| 4 | 3 | 57.154 | 55.949 |
| 8 | 1 | 103.636 | 100.974 |
| 8 | 2 | 101.057 | 99.916 |
| 8 | 3 | 101.001 | 99.055 |

The largest process-tree charges in the matrix were 22.5 GB before and 22.1 GB after. The host/GPU suite's largest
charge was 23.9 GB. These are `heavy.py` charges, not total device residency. The global KV allocation remains
65,536 bytes per token. Each slot uses 8,192 bytes per token for its key and value rows together.

## Verification

`python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0 with 25/25 host and 73/73
GPU-family tests passed, with no skips. The final saved-oracle capture verified all 16 histories. Its standard
f32 parity command exited 2 on the expected cross-checkpoint differences. The streamed comparator exited 0:
2,775/3,015 top-1 matches (92.04%), mean KL 0.0392727959, and zero non-finite logits. It used the comparison
procedure in [Inkling oQ4e](inkling-oq4e.md).

## Reproduction

Set the environment shown in the baseline command above. Build the base revision and final revision into separate
prefixes before timing. The final runner was built with:

```sh
/opt/homebrew/bin/zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/h4-final-build"
PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26 python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
sh .work/tmp/h4-matrix.sh > .work/tmp/h4-matrix.log 2>&1
sh .work/tmp/h4-4k-matrix.sh > .work/tmp/h4-4k-matrix.log 2>&1
python3 .work/tmp/h4_measure.py profile-final-1k "$PWD/.work/tmp/h4-final-build/bin/tf-inkling2-loop-run" campaign/prompts/inkling/code-1024.json bench 14 2 512
python3 .work/tmp/h4_measure.py profile-corrected-base-4k "$PWD/.work/tmp/h4-base-build/bin/tf-inkling2-loop-run" .work/tmp/h4-code-4096.json bench 14 2 512
python3 .work/tmp/h4_measure.py profile-corrected-final-4k "$PWD/.work/tmp/h4-final-build/bin/tf-inkling2-loop-run" .work/tmp/h4-code-4096.json bench 14 2 512
```

The helper and matrix scripts used for these commands follow. Place them under `.work/tmp`. An initial 4K fixture from the campaign did not match the Inkling tokenizer and its results were discarded.
Recreate the measured fixture with:

```sh
python3 campaign/heavy.py -- "$PWD/.work/tmp/h4-final-build/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd/tokenizer.json" campaign/prompts/dev/code-4096.txt tokenize .work/tmp/h4-code-all.json
jaq '.[0:4096]' .work/tmp/h4-code-all.json > .work/tmp/h4-code-4096.json
```

`h4_measure.py` retries attempts rejected by the gate or compiler monitor. The profile and matrix scripts exited 0.

```python
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path.cwd()
label, binary, prompt, *args = sys.argv[1:]
env = os.environ.copy()
env.update(PATH='/opt/homebrew/bin:' + env['PATH'], TMPDIR=str(root / '.work/tmp'), ZIG_GLOBAL_CACHE_DIR=str(Path.home() / 'repos/tf/.work/zig-global-cache'), INKLING_LOAD_LIMIT_GB='26', GPU_GATE_MAX_GPU_PCT='60', GPU_GATE_MAX_UTIL_PCT='60')
if label.startswith('profile'):
    env['TF_METAL4_TIMELINE'] = '4'
model = str(Path.home() / 'models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd')
wrapper = 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; echo H4_WINDOW_START; "$@"; result=$?; perf-gate-ready -1 || exit 75; echo H4_WINDOW_END; exit "$result"'
command = ['python3', 'campaign/heavy.py', '--', 'sh', '-c', wrapper, 'sh', binary, model, prompt, *args]
for attempt in range(1, 10000):
    log = root / '.work/tmp' / f'h4-{label}-{attempt}.log'
    with log.open('w') as out:
        proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env)
        overlap = False
        while proc.poll() is None:
            time.sleep(1)
            data = log.read_text(errors='replace')
            if 'H4_WINDOW_START' in data and 'H4_WINDOW_END' not in data:
                listing = subprocess.run(['ps', '-Ao', 'comm'], text=True, capture_output=True, check=True).stdout
                if any(Path(line.strip()).name in ('zig', 'clang', 'rustc') for line in listing.splitlines()):
                    overlap = True
    code = proc.returncode
    print(f'{label} attempt={attempt} exit={code} compiler_overlap={overlap} log={log}', flush=True)
    if code == 0 and not overlap:
        sys.exit(0)
    if code != 75 and code != 0:
        sys.exit(code)
    time.sleep(5)
sys.exit(75)
```

```sh
set -eu
base="$PWD/.work/tmp/h4-base-build/bin/tf-inkling2-loop-run"
new="$PWD/.work/tmp/h4-final-build/bin/tf-inkling2-loop-run"
for cell in 1k 4k k1 k2 k4 k8; do
    case "$cell" in
        1k) prompt=campaign/prompts/inkling/code-1024.json; set -- bench 32 4 512 ;;
        4k) prompt=.work/tmp/h4-code-4096.json; set -- bench 32 4 512 ;;
        k1) prompt=campaign/prompts/inkling/code-1024.json; set -- batchbench 1 5 ;;
        k2) prompt=campaign/prompts/inkling/code-1024.json; set -- batchbench 2 5 ;;
        k4) prompt=campaign/prompts/inkling/code-1024.json; set -- batchbench 4 5 ;;
        k8) prompt=campaign/prompts/inkling/code-1024.json; set -- batchbench 8 5 ;;
    esac
    for pair in 1 2 3; do
        if test "$pair" -eq 2; then order='new base'; else order='base new'; fi
        for side in $order; do
            if test "$side" = base; then bin="$base"; else bin="$new"; fi
            python3 .work/tmp/h4_measure.py "matrix-$cell-$pair-$side" "$bin" "$prompt" "$@"
        done
    done
done
```

The corrected 4K matrix used this script after regenerating the fixture:

```sh
set -eu
base="$PWD/.work/tmp/h4-base-build/bin/tf-inkling2-loop-run"
new="$PWD/.work/tmp/h4-final-build/bin/tf-inkling2-loop-run"
for cell in 4k; do
    case "$cell" in
        1k) prompt=campaign/prompts/inkling/code-1024.json; set -- bench 32 4 512 ;;
        4k) prompt=.work/tmp/h4-code-4096.json; set -- bench 32 4 512 ;;
        k1) prompt=campaign/prompts/inkling/code-1024.json; set -- batchbench 1 5 ;;
        k2) prompt=campaign/prompts/inkling/code-1024.json; set -- batchbench 2 5 ;;
        k4) prompt=campaign/prompts/inkling/code-1024.json; set -- batchbench 4 5 ;;
        k8) prompt=campaign/prompts/inkling/code-1024.json; set -- batchbench 8 5 ;;
    esac
    for pair in 1 2 3; do
        if test "$pair" -eq 2; then order='new base'; else order='base new'; fi
        for side in $order; do
            if test "$side" = base; then bin="$base"; else bin="$new"; fi
            python3 .work/tmp/h4_measure.py "corrected-$cell-$pair-$side" "$bin" "$prompt" "$@"
        done
    done
done
```

The saved-oracle capture and streamed comparison were:

```sh
PATH="/Users/josh/repos/tf/.work/venv-ref/bin:/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26 make -C campaign -f parity/inkling2-mini/parity.mk parity-inkling2-mini RUNNER_MODE=prefill PREFILL_CHUNK=512 LANE=h4-final REPORT="$PWD/.work/tmp/h4-delivery-parity.md" METAL_LOGITS="$PWD/.work/tmp/h4-delivery-parity" WORK="$PWD/.work" REF_LOGITS="/Users/josh/repos/tf/.work/parity/inkling2-mini/ref" REF_UV=env "REF_VENV=''" RUNNER="$PWD/.work/tmp/h4-final-build/bin/tf-inkling2-loop-run" INKLING_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
/Users/josh/repos/tf/.work/venv-ref/bin/python .work/tmp/h4-delivery-compare.py > .work/tmp/h4-delivery-compare.log 2>&1
```

The streamed comparison script was the script in [Inkling oQ4e fast](inkling-oq4e-fast.md), with `outdir` set
to `.work/tmp/h4-delivery-parity`. It reads eight rows at a time and averages the 16 history means.
