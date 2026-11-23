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

The build and parser exited 0. Six 4K profile attempts exited 75 at CPU admission. A prior 1K model run exited
75 at its CPU post-check and was discarded. No 4K profile, candidate measurements, fidelity comparison, or suite
result was obtained. The proposed f16 KV change requires updating `zig/src/families/inkling2_loop/verify_test.zig`:
its reference reconstruction and cache comparison copy four bytes per KV element. That file is outside the
permitted write scope, so the candidate was not implemented.
