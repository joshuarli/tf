# Inkling 2 Mini batched serving

Status: the host checks, checkpoint suite, native build, multi-turn sweeps, and concurrent benchmark passed.

The server fully prefills each newly admitted request before its next decode step. It then advances all decoding sessions together. `INKLING_ROLLBACK_ROWS` sets the maximum generated tokens retained per session, defaulting to 512. The server saves at most that many local ring rows and the convolution rings. When output exceeds the limit, the server clears that session's resident prefix. It admits one through eight sessions and uses no copy drafts.

For four sessions at context 4096 and 512 rollback rows, the admission formula uses a 19,526,019,168-byte checkpoint shard, 3,221,225,472 bytes of existing reserves, four 609,091,584-byte caches, and four 172,883,968-byte rollbacks. The total is 25,875,146,848 bytes, below the 26,000,000,000-byte load limit. This is an admission estimate. The guarded four-agent server and concurrent benchmark peaked at 24.4 GB charged against the 28 GB cap.

The host was an Apple M5 Max with 48 GiB of unified memory. TensorFold uses f32 arithmetic on the original checkpoint. oMLX uses oQ4e. Their precision is not matched. S1 and B1ab are from [the session record](inkling-sessions.md). The oMLX O2 figures are from [the concurrency record](inkling-omlx-concurrency.md). Each pair is two repetitions of the multi-turn workload, in aggregate output tokens per second.

| Agents | S1 | B1ab | Batched server | oMLX O2 |
| ---: | --- | --- | --- | --- |
| 1 | 10.22, 9.92 | 22.70, 22.04 | 21.20, 21.22 | 26–45 |
| 2 | 10.10, 9.84 | 21.83, 20.35 | 25.83, 25.31 | 55–59 |
| 4 | 9.39, 9.82 | 22.52, 22.41 | 28.70, 26.88 | 73–77 |

B1ab enabled copy drafts. The batched server did not use them. The multi-turn rates include prompt processing and request handling. The two-agent and four-agent batched rates are below the reported oMLX ranges for this workload, with the precision difference above.

For one agent, wall times were 17.69 and 17.67 seconds. Per-agent rates were 21.20 and 21.22 tokens per second. Each repetition produced 375 tokens. The guard's peak charge was 21.2 GB. Cached prompt tokens by turn were `0, 2572, 2624, 2673` in both repetitions. TTFT seconds by turn were `7.809, 2.333, 2.115, 2.057` and `7.780, 2.327, 2.122, 2.061`.

For two agents, wall times were 28.92 and 29.52 seconds. Per-agent rates were `12.97, 12.90` and `12.70, 12.64` tokens per second. Each repetition produced 747 tokens. The guard's peak charge was 22.4 GB. Both agents reused `0, 2572, 2624, 2673` prompt tokens by turn. TTFT seconds by turn were `14.832, 3.364, 2.832, 2.759` and `14.832, 3.241, 3.095, 3.018` for the first repetition, and `14.975, 3.420, 2.898, 2.962` and `14.975, 3.297, 3.178, 3.239` for the second.

For four agents, the two repetitions had wall times of 51.67 and 55.18 seconds. Their per-agent rates were `7.26, 7.22, 7.10, 7.24` and `6.80, 6.76, 6.64, 6.78` tokens per second. Client-observed peak charges were 23.862 and 23.598 GB. Each repetition produced 1,483 output tokens. GPU admission and CPU admission and post-checks passed for all three levels. The client completed every request.

Cached prompt tokens by turn were `0, 2572, 2624, 2673` for agents 1–3 and `0, 2572, 2624, 2679` for agent 4 in both repetitions. Client TTFT seconds by turn were:

| Repetition | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
| ---: | --- | --- | --- | --- |
| 1 | 28.470, 5.604, 4.661, 4.599 | 28.470, 5.416, 5.190, 4.886 | 28.470, 5.411, 4.803, 4.921 | 28.470, 5.672, 4.386, 3.785 |
| 2 | 31.697, 5.681, 4.682, 4.673 | 31.696, 5.482, 5.229, 4.985 | 31.697, 5.485, 4.846, 5.042 | 31.696, 5.742, 4.393, 3.820 |

The concurrent benchmark used the same four-session server and passed CPU admission and post-checks. Values below are aggregate and per-stream tokens per second, respectively, with two repetitions at each level.

| Prompt | 1 stream | 2 streams | 4 streams |
| --- | --- | --- | --- |
| Code | 37.3, 37.3 | 56.0, 28.2 | 72.0, 18.3 |
| Chat | 37.5, 37.5 | 57.2, 28.6 | 74.0, 18.5 |

All 28 replies matched their alone outputs. There were no failed requests. The serial draft comparison was vacuous because the server did not use drafts. The concurrent benchmark's short prompts and rate calculation differ from the multi-turn sweep, so its rates are not substituted for that workload's table.

Two one-agent attempts on the final server completed but failed their immediate CPU post-checks, with 2.95 and 2.12 busy cores against the 2-core limit. Their timing results are discarded. Two subsequent admission attempts failed on a busy host before loading a model. All discarded attempts were excluded from the table.

The host target `zig build test-inkling2-loop-host -Dcpu=apple_m1 -j1` exited 0. Its host-only checks cover free-slot selection, a queue beyond slot capacity, finish and cancel handling while another slot continues, and isolated rollback overflow. The checkpoint suite `INKLING_LOAD_LIMIT_GB=26 TF_INKLING2_LOOP_DIR="$INKLING_MODEL_DIR" python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1` exited 0 with a 23.1 GB peak charge. It checked both full-ring restoration and bounded restoration across a ring wrap. The direct host test populates active slots without a GPU prefill; it exercises the real queue, status, token, finish, and cancel handlers. `zig build native -Dcpu=apple_m1` exited 0. Builds used Zig 0.17.1, `-Dcpu=apple_m1`, and the shared Zig cache. The server binary was 21,543,680 bytes. A baseline binary size was not measured.

A rebuilt binary differed from the measured binary in its Mach-O UUID and link metadata. Its loaded executable and data segments were byte-identical. The sweeps use the saved measured binary so all levels share the same executable.

For each agent count, run `GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1`, then `sh campaign/fx/sweep-subagents.sh N 4096`. The script holds the GPU lock and runs the CPU admission and post-checks. At four sessions it also runs `tools/bench_concurrent.py` with levels 1, 2, and 4, 256 tokens, two repetitions, temperature zero, alone comparisons, and serial comparisons on the same server.

Shutdown printed `SafeAllocator` leak reports. Earlier baseline logs also contain these reports. They were not attributed to this change.
