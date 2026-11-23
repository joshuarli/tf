# Inkling 2 Mini f16 KV cache

Commit `e86ffc599` stores keys and values as f16. Attention accumulates in f32. Convolution rings remain f32.
Prefill stages rounded keys and values before attention, then copies them to the cache after attention.
This order preserves earlier local-window rows when a chunk wraps the ring.
The full-float oQ4e path uses decode-order row reductions for affine and expert projections so f16 rounding sees the same values. Its throughput was not separately measured.

## Cache accounting

The model has eight global slots and 40 local slots. Each KV row has 1,024 elements per key or value.
Global slots use 32,768 bytes per token, compared with 65,536 bytes for f32 KV.
Local KV occupies 167,772,160 bytes per session. The f32 convolution rings occupy 5,111,808 bytes.

| Capacity | Global KV bytes | Local KV bytes | Total KV bytes | KV and convolution bytes |
| ---: | ---: | ---: | ---: | ---: |
| 131,072 | 4,294,967,296 | 167,772,160 | 4,462,739,456 | 4,467,851,264 |
| 262,144 | 8,589,934,592 | 167,772,160 | 8,757,706,752 | 8,762,818,560 |

The values describe cache allocations, excluding model weights, rollback snapshots, scratch and allocator overhead.

## Saved-capture fidelity

The oQ4e checkpoint matched 2,763 of 3,015 top tokens (91.6418%). Mean KL was 0.0388930435, averaged over the 16 history means. The candidate had no non-finite logits. The oracle verifier checked all 16 saved histories. The streamed comparator exited 0 against the 91.0% and 0.045 limits.

The standard parity command exited 2 because its f32-versus-f32 contract compares different checkpoints. It produced all captures and reported the same top-1 and mean KL. No oracle model was run.

```sh
export PATH="/Users/josh/repos/tf/.work/venv-ref/bin:/opt/homebrew/bin:$PATH"
export TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26
make -C campaign -f parity/inkling2-mini/parity.mk parity-inkling2-mini \
  RUNNER_MODE=prefill PREFILL_CHUNK=512 LANE=h6-f16-kv \
  REPORT="$PWD/.work/tmp/f16-final-parity.md" METAL_LOGITS="$PWD/.work/tmp/f16-final-parity" \
  WORK="$PWD/.work" REF_LOGITS="$HOME/repos/tf/.work/parity/inkling2-mini/ref" REF_UV=env "REF_VENV=''" \
  RUNNER="$PWD/.work/tmp/candidate-build/bin/tf-inkling2-loop-run" \
  INKLING_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
python3 - <<'PYCODE'
from pathlib import Path
s = Path('campaign/kernels/inkling-oq4e-fast.md').read_text()
start = s.index('```python\nimport json\nfrom pathlib import Path\nimport numpy as np') + len('```python\n')
end = s.index('\n```', start)
s = s[start:end].replace("Path('.work/tmp/h5-parity')", "Path('.work/tmp/f16-final-parity')")
s += '\nif matches / total < 0.91 or np.mean(means) > 0.045 or nonfinite: raise SystemExit(1)\n'
Path('.work/tmp/f16-final-compare.py').write_text(s)
PYCODE
/Users/josh/repos/tf/.work/venv-ref/bin/python .work/tmp/f16-final-compare.py
```

The streamed comparator reads eight rows at a time from the saved captures and exits nonzero if agreement is below 0.91, mean KL exceeds 0.045, or any logit is non-finite.

## Correctness and builds

The following commands exited 0 on the final source. The suite reported 35/35 host tests and 82/82 GPU tests, with no skips. The chunk sizes 1, 4 and 5 had zero maximum logit, KV and convolution differences against decode on the tiny fixture. Batch widths 1, 2 and 4 had zero maximum logit and cache differences in the native checkpoint test. The oQ4e full-float prefill test had zero maximum logit difference against decode. Rollback, fork and verification tests passed.

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp"
export ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26
/opt/homebrew/bin/zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/candidate-build"
/opt/homebrew/bin/zig build native -Dcpu=apple_m1 -j1
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
```

The final suite's GPU process peaked at 22.0 GB charged under the 28 GB cap.

## Timing procedure

The timing runner uses the oQ4e checkpoint, greedy decoding, chunk 512, and separate base and candidate builds. The 1K input is `campaign/prompts/inkling/code-1024.json`. Generate the other inputs with:

```sh
python3 campaign/heavy.py -- "$PWD/.work/tmp/candidate-build/bin/tf-inkling2-loop-run" \
  "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd/tokenizer.json" \
  campaign/prompts/dev/code-4096.txt tokenize "$PWD/.work/tmp/tokenized.json"
jaq '.[0:4096]' .work/tmp/tokenized.json > .work/tmp/code-4096.json
jaq '.[0:2500]' .work/tmp/tokenized.json > .work/tmp/code-2500.json
```

The fixture commands exited 0 and produced 4,096 and 2,500 token IDs. The base runner was built at the admission revision `0e38e2b5e0dad68fdf4668bda60da30f719b5720` with `/opt/homebrew/bin/zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/base-build"`. The candidate runner used the corresponding `candidate-build` prefix. Both builds exited 0 before the retained timing runs.

Each measurement invoked `python3 campaign/heavy.py --`, then `GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1`, `perf-gate-ready`, the runner, and `perf-gate-ready -1`. A one-second process monitor rejected compiler overlap. Gate or monitor failures were discarded and retried. Pair order was base/candidate, candidate/base, base/candidate. Prefill used `bench 32 4 512`. Batch used `batchbench K 5`, with K equal to 1 or 4.

At 4,096 tokens, each session's global KV, local KV and convolution allocations total 307,101,696 bytes. Eight such sessions total 2,456,813,568 bytes, excluding weights, rollback, prefix retention and scratch.

The base and candidate timing executables were 2,044,728 and 2,044,856 bytes, respectively. The candidate native server was 21,796,832 bytes.

All retained timing commands exited 0. Prefill is tokens per second. Batch is median milliseconds per step over five internal pairs. Each table entry lists the three outer pairs in order.

| Workload | Base | Candidate | Base median | Candidate median |
| --- | --- | --- | ---: | ---: |
| Prefill 1K, tok/s | 569.6, 603.5, 537.6 | 330.8, 576.4, 608.6 | 569.6 | 576.4 |
| Prefill 4K, tok/s | 642.2, 600.0, 617.6 | 627.5, 641.9, 614.9 | 617.6 | 627.5 |
| Batch K=1, ms | 21.082, 21.823, 21.621 | 20.955, 21.658, 21.644 | 21.621 | 21.644 |
| Batch K=4, ms | 45.294, 44.489, 48.894 | 44.671, 42.772, 46.795 | 45.294 | 44.671 |

The K=1 median difference was 0.023 ms, inside the observed spread. The candidate's first 1K prefill sample was much slower than its other two. The median and overall ranges do not establish a regression.

Each timed invocation used the following command. Set `RUNNER`, `IDS` and `ARGS` for the case and set `MARKER` to a unique path under `.work/tmp`.

```sh
python3 campaign/heavy.py -- "$PWD/.work/tmp/timed-run.sh" "$MARKER" \
  "$RUNNER" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" "$IDS" $ARGS
```

Create `.work/tmp/timed-run.sh` with the following contents:

```sh
#!/bin/sh
set -u
marker=$1
shift
GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 || exit 75
perf-gate-ready || exit 75
rm -f "$marker"
(
    while :; do
        if ps -Ao comm | rg -q '(^|/)(zig|maker|clang|metal|metallib)$'; then printf 'compiler observed\n' >> "$marker"; fi
        sleep 1
    done
) &
monitor=$!
"$@"
result=$?
kill "$monitor" 2>/dev/null || true
wait "$monitor" 2>/dev/null || true
perf-gate-ready -1 || exit 75
if [ -s "$marker" ]; then cat "$marker"; exit 76; fi
exit "$result"
```

Run `chmod +x .work/tmp/timed-run.sh` before invoking the wrapper. The orchestrator exited 0 after three passing pairs per case. Earlier measurements before the matrix-dispatch correction were discarded.

## Served-agent sweeps

The four-agent command exited 0 with the GPU and CPU gates and both CPU post-checks passing. Repetitions 1 and 2 produced 57.419 and 70.294 aggregate tok/s, with sampled peak charges of 21,893,477,072 and 21,893,509,840 bytes. The guard reported a 21.9 GB peak for the process. Concurrent versus solo comparisons for code and chat at 1, 2 and 4 streams had zero unequal and zero failed replies. The serial draft comparisons were vacuous because no draft was accepted.

```sh
INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" sh campaign/fx/sweep-subagents.sh 4 4096
```

The first completed four-agent attempt passed the script's checks but failed an additional outer CPU post-check, so it was discarded. Admission failures were retried.

The eight-agent command exited 0 with its GPU and CPU admission and immediate CPU post-checks passing. Repetitions 1 and 2 produced 62.116 and 66.325 aggregate tok/s, with sampled peak charges of 23,873,434,368 and 23,873,925,912 bytes. The guard reported a 23.9 GB peak for the process.

```sh
INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" sh campaign/fx/sweep-subagents.sh 8 4096
```

The gate wrapper for the sweep prepended `.work/tmp/gates` to `PATH`. Its `perf-gate-ready` called `GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 /Users/josh/.local/bin/gpu-gate-ready -1` before every CPU admission and then executed `/Users/josh/.local/bin/perf-gate-ready "$@"`. The script's immediate CPU post-checks were retained. Gate failures were discarded and retried.

The standard eight-agent sweep does not run concurrent-versus-solo checks. A second gated run used the same server and harness with that comparison enabled for eight streams. It exited 0. Code and chat each had 16 equal, zero unequal and zero failed replies. Serial draft comparisons were vacuous because no draft was accepted. The guard reported a 23.9 GB peak.

```sh
python3 - <<'PY'
from pathlib import Path
s = Path('campaign/fx/sweep-subagents.sh').read_text()
s = s.replace('if [ "$INKLING_AGENTS" = 4 ]; then', 'if [ "$INKLING_AGENTS" = 8 ]; then')
s = s.replace('--levels 1,2,4 --tokens', '--levels 8 --tokens')
Path('.work/tmp/sweep-eight-check.sh').write_text(s)
PY
INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" sh .work/tmp/sweep-eight-check.sh 8 4096
```

Create the gate wrapper used for the sweeps before running those commands:

```sh
mkdir -p .work/tmp/gates
cat > .work/tmp/gates/perf-gate-ready <<'SH'
#!/bin/sh
if [ "$#" -eq 0 ]; then
    GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 /Users/josh/.local/bin/gpu-gate-ready -1 || exit "$?"
fi
exec /Users/josh/.local/bin/perf-gate-ready "$@"
SH
chmod +x .work/tmp/gates/perf-gate-ready
export PATH="$PWD/.work/tmp/gates:/opt/homebrew/bin:$PATH"
```
