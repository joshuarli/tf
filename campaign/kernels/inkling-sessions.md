# Inkling 2 Mini resident sessions

The server keeps one prompt-end cache per session and restores its local and convolution rings from one shared rollback buffer after decoding. Global rows before the prompt end are append-only. The server still decodes one request at a time.

The checkpoint is `Inkling-2-Mini@ed40e7da625f` on the 48 GiB M5 Max. Requests use greedy decoding, `INKLING_COPY_DRAFTS=1`, four turns, a 256-token output cap, and the 115-function prompt used in [the S1 sweep](inkling-subagents.md). The server uses `INKLING_LOAD_LIMIT_GB=26` and the 28 GB `heavy.py` charge cap.

S1 used one resident state and `--parallel 1 --context 32768` for every client level. All 120 requests reused zero prompt tokens. Its two-repetition aggregate output rates and peak charges were:

| Agents | Aggregate output tokens/s | Peak charged GB |
|---:|---:|---:|
| 1 | 10.22, 9.92 | 23.00, 22.91 |
| 2 | 10.10, 9.84 | 23.55, 23.36 |
| 4 | 9.39, 9.82 | 23.63, 24.22 |

The host selection tests pass for longest prompt prefix, least-recently-used fallback, omitted generated text, and disabled reuse. The checkpoint-backed Metal suite reports 39 of 39 tests passed. Its rollback test compares both local rings, all four convolution rings, valid global KV rows, and length before generation and after restoration. Its guarded peak charge was 22.2 GB.

A separate synthetic continuation used a 1,024-token prompt, four generated positions, and a 61-token suffix. Token at index `i` was `100 + i % 71`. After rollback, the continuation logits matched fresh full prefill with maximum absolute difference 0 and greedy token 198 in both runs. Its guarded peak charge was 21.2 GB. These checks establish the tested state and continuation, not general token equality across all chunk schedules.

For the 4,096-token session sweep, each client level uses the same number of server slots as agents. The one-agent 32,768-token run is a separate comparison. TTFT is client time to first visible token and includes queueing.

Each row below is one repetition. The accepted runs passed `perf-gate-ready` admission and `perf-gate-ready -1` after measurement, both with exit 0. GPU admission used `GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60` and exited 0. The client completed every request.

| Agents | Context per session | Repetition | Wall seconds | Aggregate output tokens/s | Peak charged GB |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 1 | 16.52 | 22.7 | 21.43 |
| 1 | 4096 | 2 | 17.02 | 22.04 | 21.43 |
| 2 | 4096 | 1 | 34.21 | 21.83 | 21.63 |
| 2 | 4096 | 2 | 36.7 | 20.35 | 21.66 |
| 4 | 4096 | 1 | 65.85 | 22.52 | 22.44 |
| 4 | 4096 | 2 | 66.18 | 22.41 | 22.04 |
| 1 | 32768 | 1 | 16.77 | 22.36 | 23.36 |
| 1 | 32768 | 2 | 17.05 | 21.99 | 23.31 |

Each list is ordered by turn. The cached-token counts come from each streamed response. The script also checked that the existing server-log parser saw all prefix and completion records, and that every prompt plus 256 output tokens fit the context.

| Agents | Context | Repetition | Agent | Reused prompt tokens by turn | Client TTFT seconds by turn |
|---:|---:|---:|---:|---|---|
| 1 | 4096 | 1 | 1 | 0, 2572, 2624, 2673 | 7.76, 2.22, 2.06, 2.01 |
| 1 | 4096 | 2 | 1 | 0, 2572, 2624, 2673 | 8.07, 2.29, 2.11, 2.05 |
| 2 | 4096 | 1 | 1 | 0, 2572, 2624, 2673 | 16.51, 5.26, 4.61, 5.24 |
| 2 | 4096 | 1 | 2 | 0, 2572, 2624, 2673 | 7.83, 10.82, 5.19, 4.73 |
| 2 | 4096 | 2 | 1 | 0, 2572, 2624, 2673 | 19.68, 5.4, 4.63, 4.5 |
| 2 | 4096 | 2 | 2 | 0, 2572, 2624, 2673 | 10.29, 11.6, 5.29, 4.57 |
| 4 | 4096 | 1 | 1 | 0, 2572, 2624, 2673 | 7.52, 28.56, 11.09, 9.38 |
| 4 | 4096 | 1 | 2 | 0, 2572, 2624, 2673 | 24.8, 17.42, 9.97, 8.92 |
| 4 | 4096 | 1 | 3 | 0, 2572, 2624, 2673 | 33.91, 11.26, 9.27, 8.77 |
| 4 | 4096 | 1 | 4 | 0, 2572, 2624, 2679 | 15.71, 23.4, 10.29, 8.88 |
| 4 | 4096 | 2 | 1 | 0, 2572, 2624, 2673 | 7.48, 28.9, 11.08, 9.4 |
| 4 | 4096 | 2 | 2 | 0, 2572, 2624, 2673 | 34.23, 11.24, 9.44, 8.87 |
| 4 | 4096 | 2 | 3 | 0, 2572, 2624, 2673 | 16.75, 22.57, 10.39, 9.25 |
| 4 | 4096 | 2 | 4 | 0, 2572, 2624, 2679 | 25.6, 16.77, 9.74, 8.83 |
| 1 | 32768 | 1 | 1 | 0, 2572, 2624, 2673 | 7.85, 2.27, 2.05, 2.04 |
| 1 | 32768 | 2 | 1 | 0, 2572, 2624, 2673 | 8.24, 2.19, 2.06, 2.06 |

The observed GPU rollback copy times were 1.413–7.561 ms for saves and 3.373–6.701 ms for restores across the accepted 4,096-token runs. The 32,768-token comparison logged 1.467–87.702 ms for saves and 5.443–29.410 ms for restores. These are per-request wall times, not isolated kernel timings.

The first two four-agent attempts and the first 32,768-token attempt completed their workloads, but `perf-gate-ready -1` exited 75 after each. Their timings are discarded. Both four-agent attempts reported nonzero reuse on turns 2–4 for every agent. Two earlier two-agent starts also failed the admission gate before measurement. The accepted 32,768-token run passed admission after an initially busy sample.

To reproduce each configuration from the repository root, build the native server with Zig 0.17.1, then run the GPU admission check and each sweep:

```sh
PATH=/opt/homebrew/bin:$PATH ZIG_GLOBAL_CACHE_DIR="$(git rev-parse --git-common-dir)/../.work/zig-global-cache" TMPDIR="$PWD/.work/tmp" zig build native -Dcpu=apple_m1 -j1
GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 && sh campaign/fx/sweep-subagents.sh 1 4096
GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 && sh campaign/fx/sweep-subagents.sh 2 4096
GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 && sh campaign/fx/sweep-subagents.sh 4 4096
GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 && sh campaign/fx/sweep-subagents.sh 1 32768
```

The script holds the shared `heavy.py` lock for the model process, runs two repetitions, checks prompt capacity and per-agent reuse, and retains the established prefix-log parser check. The 32,768-token result uses one session. It does not qualify four sessions at that context.
