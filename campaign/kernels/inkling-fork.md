# Inkling 2 Mini prompt forks

The server can copy a saved prompt prefix from another session. It retains convolution rings and overwritten local rows at aligned points near each prompt end. It copies global rows through the chosen point and restores the saved local and convolution state in the destination. A decoding source's generation rollback is applied to the destination first. The source remains unchanged.

`INKLING_FORK_WINDOW` sets the retained window in tokens and defaults to 256. `INKLING_FORK_INTERVAL` sets the alignment and defaults to 128. The window must fit the model's local attention window. The interval must be positive and no greater than the window. At most eight snapshots are admitted per session. `INKLING_PREFIX_REUSE=0` disables both resident reuse and forks. `INKLING_ROLLBACK_ROWS` limits generation rollback and defaults to 512. The sweep used 256 rows, enough for its 256-token output cap. Forked tokens are included in the existing cached-token usage count. `inkling fork:` identifies the source and point, and the existing `inkling prefix:` line remains available to the log parser.

For four sessions at context 3072, rollback 256, fork window 256, and interval 128, the admission estimate is 25,983,150,176 bytes under the 26,000,000,000-byte limit. It consists of a 19,526,019,168-byte checkpoint shard, 3,221,225,472 bytes of reserves, four 541,982,720-byte caches, four 88,997,888-byte generation rollbacks, and eight 88,997,888-byte fork buffers. Each point reserves a full window, even when fewer rows need saving. The estimate is an admission bound, not a measured peak.

## Multi-turn sweep

The host was an Apple M5 Max with 48 GiB unified memory. TensorFold used f32 and `Inkling-2-Mini@ed40e7da625f`. oMLX O2 used oQ4e, so the precision differs. Each sweep used four turns, greedy decoding, a 256-token output cap, context 3072 per session, and one session per agent. The script checked that each prompt and output cap fit the context. Each repetition passed `perf-gate-ready` before measurement and `perf-gate-ready -1` immediately afterward. The GPU guard admitted each server and enforced a 28 GB charge cap. All requests completed.

The B1d and oMLX O2 figures are from [the batched serving record](inkling-batch-serve.md). These are two runs per configuration, not interleaved comparisons. The spread, different context and precision limit a speedup claim.

| Agents | B1d output tokens/s | Fork output tokens/s | oMLX O2 output tokens/s | Fork peak charged GB |
| ---: | --- | --- | --- | --- |
| 1 | 21.20, 21.22 | 19.10, 20.48 | 26–45 | 21.183, 21.506 |
| 2 | 25.83, 25.31 | 30.64, 31.76 | 55–59 | 21.899, 22.138 |
| 4 | 28.70, 26.88 | 35.98, 40.52 | 73–77 | 23.159, 23.159 |

The charge values in the table are sampled during each repetition. The GPU guard's full-process peaks were 21.5, 22.1, and 23.2 GB for one, two, and four sessions respectively. The four-session command also completed its existing concurrent benchmark with no failed requests and matching alone outputs.

Each row below lists turns in order. On turn 1, 2432 is a fork and 0 is a cold prefill. On turns 2–4, the counts are own-session reuse. Client TTFT is time to first visible content. Server TTFT comes from the matching completion record, paired by response ID. All 56 IDs matched exactly once.

| Agents | Rep | Agent | Cached tokens | Client TTFT seconds | Server TTFT seconds |
| ---: | ---: | ---: | --- | --- | --- |
| 1 | 1 | 1 | 0, 2572, 2624, 2673 | 9.631, 2.321, 2.122, 2.194 | 7.794, 0.28, 0.273, 0.397 |
| 1 | 2 | 1 | 0, 2572, 2624, 2673 | 8.285, 2.326, 2.126, 2.201 | 6.461, 0.28, 0.275, 0.401 |
| 2 | 1 | 1 | 2432, 2572, 2624, 2673 | 10.089, 3.352, 2.827, 2.871 | 7.614, 0.552, 0.294, 0.415 |
| 2 | 1 | 2 | 0, 2572, 2624, 2673 | 10.089, 3.233, 3.091, 3.254 | 6.934, 0.281, 0.303, 0.422 |
| 2 | 2 | 1 | 0, 2572, 2624, 2673 | 9.239, 3.35, 2.825, 2.869 | 6.145, 0.279, 0.294, 0.415 |
| 2 | 2 | 2 | 2432, 2572, 2624, 2673 | 9.239, 3.228, 3.088, 3.252 | 6.771, 0.551, 0.301, 0.422 |
| 4 | 1 | 1 | 0, 2572, 2624, 2673 | 17.933, 5.544, 4.605, 4.715 | 12.092, 0.283, 0.316, 0.438 |
| 4 | 1 | 2 | 2432, 2572, 2624, 2673 | 17.932, 5.348, 5.081, 5.089 | 13.999, 1.096, 0.539, 0.438 |
| 4 | 1 | 3 | 2432, 2572, 2624, 2673 | 17.932, 5.35, 4.697, 5.217 | 12.728, 0.553, 0.274, 0.445 |
| 4 | 1 | 4 | 2432, 2572, 2624, 2679 | 17.933, 5.605, 4.352, 3.827 | 13.362, 0.824, 0.33, 0.431 |
| 4 | 2 | 1 | 2432, 2572, 2624, 2673 | 12.593, 5.734, 4.758, 4.849 | 8.643, 0.94, 0.35, 0.47 |
| 4 | 2 | 2 | 2432, 2572, 2624, 2673 | 12.593, 5.544, 5.321, 5.264 | 7.955, 0.314, 0.307, 0.47 |
| 4 | 2 | 3 | 0, 2572, 2624, 2673 | 12.593, 5.542, 4.926, 5.425 | 6.592, 0.628, 0.619, 0.478 |
| 4 | 2 | 4 | 2432, 2572, 2624, 2679 | 12.593, 5.797, 4.467, 3.913 | 7.271, 1.254, 0.367, 0.459 |

The first four-agent run passed the script's original gates around both repetitions together. It is excluded because it lacked an immediate post-check for each repetition. The accepted runs used the updated per-repetition gates.

## Verification

The host target `zig build test-inkling2-loop-host -Dcpu=apple_m1 -j1` exited 0. It tests source selection across sessions, saved-point selection, longer own-session reuse, preservation of the source, and destination selection when the best source is free.

The checkpoint target `INKLING_LOAD_LIMIT_GB=26 GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 TF_INKLING2_LOOP_DIR="$HOME/models/Inkling-2-Mini@ed40e7da625f" python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1` exited 0. The fork test compares local rings, convolution rings, valid global rows, and length with direct prefill using the same chunk schedule. It also compares continuation logits and checks that the source is unchanged. The guard reported a 23.1 GB peak charge. `zig build native -Dcpu=apple_m1` exited 0.

The builds used Zig 0.17.1, `-Dcpu=apple_m1`, and the shared Zig cache. Each sweep command exited 0:

```sh
INKLING_ROLLBACK_ROWS=256 INKLING_FORK_WINDOW=256 INKLING_FORK_INTERVAL=128 sh campaign/fx/sweep-subagents.sh 1 3072
INKLING_ROLLBACK_ROWS=256 INKLING_FORK_WINDOW=256 INKLING_FORK_INTERVAL=128 sh campaign/fx/sweep-subagents.sh 2 3072
INKLING_ROLLBACK_ROWS=256 INKLING_FORK_WINDOW=256 INKLING_FORK_INTERVAL=128 sh campaign/fx/sweep-subagents.sh 4 3072
```
