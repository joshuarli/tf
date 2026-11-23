# Inkling 2 Mini prefix store

The server retains prompt and shared-prefix states in the `--prompt-cache-gib` budget. It compares retained states with resident and fork states at admission. `INKLING_PREFIX_REUSE=0` disables all three reuse paths. This receipt uses the oQ4e checkpoint, greedy decoding, a 4096-token context, a 26 GB load limit, and the 28 GB `heavy.py` cap.

## Correctness

The host command exited 0:

```sh
ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" TMPDIR="$PWD/.work/tmp" /opt/homebrew/bin/zig build test-inkling2-loop-host -Dcpu=apple_m1
```

The suite command exited 0. The suite reported `ok` for `inkling-host` and `inkling-gpu` with no skips.

```sh
PATH=/opt/homebrew/bin:$PATH ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" TMPDIR="$PWD/.work/tmp" INKLING_LOAD_LIMIT_GB=26 python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
```

The prefix-check command exited 0 with the capture generated below. It compared the store-restored local rings, convolution rings, valid global rows, length, and continuation logits against direct prefill. It reported `state_bytes=419299328 state_equal=true continuation_logits_equal=true`. The guarded run peaked at 19.9 GB charged.

```sh
python3 -c 'from pathlib import Path; Path(".work/tmp/prefix-capture.txt").write_text("1200 8\n" + " ".join(str(100+i%71) for i in range(1240)) + "\n")'
PATH=/opt/homebrew/bin:$PATH INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-prefix-check "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" .work/tmp/prefix-capture.txt
```

## Workload

The sweeps ran on an Apple M5 Max with 48 GiB unified memory. Both binaries use Zig 0.17.1, `-Dcpu=apple_m1`, greedy decoding, the same checkpoint, two repetitions, four turns per agent, and a 256-token output cap. The base is `9a0a513ad2fd9091bb1e9691a37733db32f7ee40`. The candidate is `d1f244b09`.

Build the native server and prefix-check tool before timing:

```sh
ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" TMPDIR="$PWD/.work/tmp" /opt/homebrew/bin/zig build native tf-inkling2-prefix-check -Dcpu=apple_m1 -j1
```

Build the base server from the base revision in the same worktree. Save the candidate files and executable first, then restore them after the base build:

```sh
mkdir -p .work/tmp/candidate
for name in inkling2_loop.zig inkling2_loop_host.zig inkling2_loop_prefix.zig; do cp -p "zig/src/native/$name" ".work/tmp/candidate/$name"; done
cp -p zig-out/native/bin/tensorfold-native .work/tmp/candidate/tensorfold-native
for name in inkling2_loop.zig inkling2_loop_host.zig inkling2_loop_prefix.zig; do git show "9a0a513ad2fd9091bb1e9691a37733db32f7ee40:zig/src/native/$name" > "zig/src/native/$name"; done
ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" TMPDIR="$PWD/.work/tmp" /opt/homebrew/bin/zig build native -Dcpu=apple_m1 -j1
cp -p zig-out/native/bin/tensorfold-native .work/tmp/base-tensorfold-native
for name in inkling2_loop.zig inkling2_loop_host.zig inkling2_loop_prefix.zig; do cp -p ".work/tmp/candidate/$name" "zig/src/native/$name"; done
cp -p .work/tmp/candidate/tensorfold-native zig-out/native/bin/tensorfold-native
```

Both builds exited 0. Run each sweep against each executable, saving the results before the next run. The sweep invokes `heavy.py` and the pre- and post-measurement gates.

```sh
PATH=/opt/homebrew/bin:$PATH INKLING_MODEL=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd sh campaign/fx/sweep-subagents.sh 1 4096
PATH=/opt/homebrew/bin:$PATH INKLING_MODEL=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd sh campaign/fx/sweep-subagents.sh 2 4096
PATH=/opt/homebrew/bin:$PATH INKLING_MODEL=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd sh campaign/fx/sweep-subagents.sh 4 4096
```

All six accepted sweeps exited 0. The base and candidate measurements were not interleaved. Earlier attempts were discarded when a CPU gate exited 75, including attempts where Python reported an outer exit 1. Two repetitions are insufficient for a general performance claim.

The table reports each repetition's aggregate output tokens per second and client-observed turn-1 time to first token in seconds. For multiple agents, times follow the result's agent order.

| Agents | Revision | Rep 1 tok/s | Rep 1 turn-1 TTFT | Rep 2 tok/s | Rep 2 turn-1 TTFT | Peak charged GB |
| --- | --- | ---: | --- | ---: | --- | ---: |
| 1 | base | 20.84 | 9.132 | 20.89 | 9.123 | 18.5 |
| 1 | candidate | 19.18 | 10.404 | 34.01 | 1.928 | 20.5 |
| 2 | base | 31.48 | 10.460, 10.351 | 31.60 | 10.366, 10.258 | 19.4 |
| 2 | candidate | 30.49 | 10.806, 10.698 | 42.16 | 3.927, 3.819 | 21.5 |
| 4 | base | 40.09 | 13.264, 13.086, 13.264, 13.086 | 39.70 | 13.384, 13.207, 13.385, 13.206 | 21.4 |
| 4 | candidate | 35.77 | 15.551, 15.358, 15.551, 15.358 | 45.23 | 6.852, 6.660, 6.851, 6.660 | 23.4 |

The candidate's first repetition includes cold store insertion. At N=1 the second repetition's turn 1 reused 2,571 of 2,572 prompt tokens. At N=2 and N=4, each agent reused 2,432 tokens on that turn. The largest logged store totals were 2,047,255,352 bytes at N=1, 2,057,151,892 at N=2, and 2,064,012,800 at N=4, each below the 2 GiB budget. The N=4 concurrent-versus-solo check reported zero unequal or failed replies for both code and chat at every tested width.

Run the N=1 reuse-off control:

```sh
PATH=/opt/homebrew/bin:$PATH INKLING_PREFIX_REUSE=0 INKLING_MODEL=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd sh campaign/fx/sweep-subagents.sh 1 4096
```

Both repetitions and their timing gates completed. The script exited 1 because its final assertion requires nonzero cached tokens on later turns, while this control disables reuse. The eight reply token counts and SHA prefixes matched the candidate run in order: `94:8752dff0d022`, `99:c639d102c328`, `92:dcdd2b41dbf7`, `87:aa28a7416de2`, repeated twice.
