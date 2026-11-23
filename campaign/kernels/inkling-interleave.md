# Inkling 2 Mini interleaved admission

Admission advances at most one prompt chunk before the next batched decode step. One prompt is admitted at a time. Its slot joins decoding after prefill and first-token sampling. Fork snapshots and store inserts use the same token boundaries as before. Cancellation releases a partially admitted slot.

## Correctness

The base revision is `a63d507cf3734e19e0d4e03158dcae9b897c6681`. Both builds use Zig 0.17.1 and `-Dcpu=apple_m1`. The host test exercises chunk boundaries, decoding eligibility, fork points, first and last store positions, and prompt completion.

```sh
ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" TMPDIR="$PWD/.work/tmp" /opt/homebrew/bin/zig build test-inkling2-loop-host -Dcpu=apple_m1
PATH="/opt/homebrew/bin:$PATH" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" TMPDIR="$PWD/.work/tmp" INKLING_LOAD_LIMIT_GB=26 python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
```

Both commands exited 0. The suite reported `ok` for `inkling-host` and `inkling-gpu`, with no skips. The GPU suite peaked at 19.8 GB charged against the 28 GB cap.

The N=1 baseline and candidate server logs each contained eight `done` lines. Their reply hashes matched in order: `8752dff0d022`, `c639d102c328`, `dcdd2b41dbf7`, `aa28a7416de2`, repeated twice.

## Workload

The host is an Apple M5 Max with 48 GiB unified memory. The workload uses oQ4e weights, a 4096-token context, greedy sampling, four turns per agent, and a 256-token output cap. Each command uses the GPU lock and a 28 GB charge cap through `campaign/heavy.py`. `INKLING_LOAD_LIMIT_GB` is 26. The script checks each repetition with `perf-gate-ready` and checks N=4 concurrent replies against solo replies.

The base binary was built in a detached scratch worktree at the base revision. The candidate was built in the task worktree before measurements:

```sh
git worktree add --detach .work/wt/base a63d507cf3734e19e0d4e03158dcae9b897c6681
ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" TMPDIR="$PWD/.work/tmp" /opt/homebrew/bin/zig build native -Dcpu=apple_m1 --prefix "$PWD/.work/base-clean"
ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" TMPDIR="$PWD/.work/tmp" /opt/homebrew/bin/zig build native -Dcpu=apple_m1
```

The first build command runs from the scratch worktree. Both builds exited 0. The sweep expects its binary at `zig-out/native/bin/tensorfold-native`. Before each invocation the selected prebuilt binary is copied there. The N=4 order is base, candidate, base, candidate. Each invocation is:

```sh
INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" INKLING_LOAD_LIMIT_GB=26 sh campaign/fx/sweep-subagents.sh N 4096
```

The measured commands use these prebuilt binaries. Run the following sequence from the task worktree, waiting for each command to exit before starting the next:

```sh
export INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" INKLING_LOAD_LIMIT_GB=26
cp .work/tmp/interleave-runs/base zig-out/native/bin/tensorfold-native
sh campaign/fx/sweep-subagents.sh 1 4096
cp .work/tmp/interleave-runs/candidate zig-out/native/bin/tensorfold-native
sh campaign/fx/sweep-subagents.sh 1 4096
sh campaign/fx/sweep-subagents.sh 2 4096
cp .work/tmp/interleave-runs/base zig-out/native/bin/tensorfold-native
sh campaign/fx/sweep-subagents.sh 4 4096
cp .work/tmp/interleave-runs/candidate zig-out/native/bin/tensorfold-native
sh campaign/fx/sweep-subagents.sh 4 4096
cp .work/tmp/interleave-runs/base zig-out/native/bin/tensorfold-native
sh campaign/fx/sweep-subagents.sh 4 4096
cp .work/tmp/interleave-runs/candidate zig-out/native/bin/tensorfold-native
sh campaign/fx/sweep-subagents.sh 4 4096
```

Each accepted sweep exited 0. The sequence had host-load deferrals: base N=1 attempts 1–2, base N=4 A attempts 1–8, candidate N=4 A attempts 1–6, and base N=4 B attempts 1–10. An inner Python wrapper returned 1 when its `perf-gate-ready` subprocess exited 75. Those attempts were discarded and retried.

N=4 concurrent and serial replies matched their solo replies at levels 1, 2, and 4. All cells reported zero failed and zero unequal replies, and overlapping cells were nonvacuous. The maximum guarded peak was 23.4 GB.

The two repetitions within each sweep differ in cache warmth. These are end-to-end rates including prefill and client time. The measurements do not establish a throughput improvement. Cold N=4 candidate rates were 37.47–39.42 tokens/s versus 39.66–40.46 for the base. Warm candidate rates were 47.83–48.09 versus 48.33–48.78 for the base. Two sweeps per binary are too few to establish a stable difference.

| Sweep | Rep | Aggregate tokens/s | Peak charged GB | Client TTFT seconds by agent and turn |
| --- | ---: | ---: | ---: | --- |
| base N=1 | 1 | 20.53 | 20.52 | 9.224 / 2.121 / 1.932 / 1.935 |
| base N=1 | 2 | 35.88 | 20.52 | 1.810 / 1.805 / 1.933 / 1.927 |
| candidate N=1 | 1 | 20.53 | 20.52 | 9.269 / 2.110 / 1.928 / 1.925 |
| candidate N=1 | 2 | 36.02 | 20.52 | 1.795 / 1.796 / 1.927 / 1.921 |
| candidate N=2 | 1 | 30.76 | 21.49 | 10.623 / 2.922 / 2.676 / 2.638; 10.407 / 3.160 / 3.007 / 3.198 |
| candidate N=2 | 2 | 42.72 | 21.49 | 3.884 / 3.214 / 3.026 / 2.932; 3.884 / 3.176 / 2.981 / 3.157 |
| base N=4 A | 1 | 40.46 | 23.43 | 13.511 / 4.868 / 4.151 / 3.985; 13.334 / 5.407 / 5.142 / 5.036; 13.511 / 4.687 / 4.132 / 5.322; 13.334 / 5.406 / 5.140 / 4.795 |
| base N=4 A | 2 | 48.78 | 23.43 | 6.917 / 4.885 / 4.159 / 4.183; 6.739 / 5.428 / 5.148 / 5.253; 6.917 / 4.703 / 4.140 / 5.526; 6.739 / 5.431 / 5.151 / 4.998 |
| candidate N=4 A | 1 | 39.42 | 23.43 | 13.214 / 5.648 / 5.421 / 5.234; 13.573 / 4.513 / 4.885 / 5.125; 13.393 / 5.069 / 4.968 / 5.661; 13.393 / 5.187 / 5.202 / 5.080 |
| candidate N=4 A | 2 | 48.09 | 23.43 | 6.662 / 5.859 / 5.103 / 4.864; 6.662 / 5.555 / 5.392 / 5.717; 7.200 / 4.550 / 3.970 / 5.130; 6.843 / 4.951 / 5.032 / 4.946 |
| base N=4 B | 1 | 39.66 | 23.43 | 13.549 / 4.861 / 4.260 / 4.235; 13.371 / 5.434 / 5.236 / 5.403; 13.549 / 4.684 / 4.208 / 5.692; 13.371 / 5.434 / 5.234 / 5.138 |
| base N=4 B | 2 | 48.33 | 23.43 | 6.914 / 4.926 / 4.256 / 4.152; 6.735 / 5.489 / 5.298 / 5.265; 6.914 / 4.744 / 4.242 / 5.562; 6.735 / 5.491 / 5.300 / 5.012 |
| candidate N=4 B | 1 | 37.47 | 23.42 | 14.630 / 5.927 / 5.561 / 5.299; 15.006 / 5.121 / 4.472 / 4.808; 15.006 / 4.993 / 4.453 / 5.556; 14.630 / 5.865 / 5.487 / 5.463 |
| candidate N=4 B | 2 | 47.83 | 23.42 | 6.922 / 4.987 / 4.739 / 4.291; 6.385 / 5.448 / 5.543 / 5.889; 7.101 / 4.476 / 4.485 / 4.694; 6.564 / 5.114 / 5.188 / 5.223 |

Each group in the TTFT column lists four turns for one agent. The peak column is the per-repetition sample. The `heavy.py` process peaks were 20.5 GB for N=1, 21.5 GB for N=2, and 23.4 GB for N=4.
