# M5 Max campaign tooling

Development-only tooling for the native inference campaign in [plan.md](plan.md). Nothing here ships in `tensorfold-native`.

## Requirements

- Zig at the version pinned in [bootstrap.lock.json](bootstrap.lock.json); `make check-zig` refuses any other.
- `uv`, `jaq`, Xcode with the Metal toolchain, and `swift` for the Metal device probe.
- Python dependencies come from [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock) (`laputa-buildlib`, pinned by wheel hash).

Every command runs from the repository root as `make -C campaign <target>`. Each workstream defines its targets in its own `campaign/<dir>/<name>.mk` or `campaign/<dir>/<subdir>/<name>.mk`, which the Makefile includes, so parallel lanes never edit the Makefile itself. `parity`, `quality` and `pi-harness` dispatch on `MODEL` to `<target>-<model>`, which each model's fragment defines. Scratch goes to the main checkout's `.work/`: `TMPDIR` is `.work/tmp` and the Zig global cache is `.work/zig-global-cache`.

## Fetch weights

```sh
make -C campaign fetch NAMES="flint-int4 inkling-2-mini"
```

Each file in [sources.lock.json](sources.lock.json) is fetched to `~/models/<repo-name>@<rev12>/` and checked against its pinned size and sha256. Files that already verify are skipped, so a second run downloads nothing. Omit `NAMES` to fetch every model, including `flint-dflash2`.

## Record the host

```sh
make -C campaign host
```

This writes [host.json](host.json) with the CPU, memory, GPU, macOS, Xcode, Metal toolchain and Zig versions, and the Metal device's family support and `recommendedMaxWorkingSetSize`.

## Build the frozen baseline

```sh
make -C campaign build-baseline
make -C campaign baseline-timings
```

`build-baseline` creates a detached worktree of `fa13df5d` at `.work/wt/baseline`, overrides only `.zig-version` and `minimum_zig_version` to the pinned Zig, builds `tensorfold-native` and prints its sha256. `baseline-timings` reruns the cold and warm builds and the baseline suites, writing logs to `.work/logs/baseline/`. The suite that touches Metal runs through `heavy.py`, which holds `.work/gpu.lock` (with `lockf -k`) and stops the command if its process tree passes 28 GB; every model-loading command in the campaign uses it, so the campaign stays under the owner's 32 GB cap. [baseline.md](baseline.md) records the results and the known red tests.

## Resume

Read the **Status** block of [plan.md](plan.md), run `make -C campaign host`, compare the result with the committed `host.json`, then take the next item.

## Checks and benchmarks

`make -C campaign bench-list` prints every workload with its tier, area, purpose and cost. Pick the smallest tier, or the named workloads, that cover a change.

| Tier | What it runs | Cost |
| --- | --- | --- |
| `edit` | CPU fixtures and host tests: the logit gate's controls, the evidence fixtures, quality cases, the prompt corpus, Flint and Inkling 2 Mini host tests | about 30 s warm |
| `change` | one short round on the GPU: `tf-flint-run` decode steps and one-row against 16-row self-consistency (no server, the fastest Flint loop), the served smoke, one 1,024-token bench process, the cached Inkling reference check | minutes |
| `scenario` | the six-pair primary matrix, the quality dev set, the Pi tasks, Flint parity | tens of minutes each |
| `full` | the 48-case quality set, the 104 GB Inkling oracle rerun, `zig build test test-golden` | manual only |

```sh
make -C campaign bench-check TIER=change            # preflight only: names each unmet requirement and its fix
make -C campaign bench-dry NAMES=flint-steps        # print the commands; writes nothing
make -C campaign bench-suite TIER=edit              # run; AREAS=flint, NAMES=a,b and SKIP=a,b narrow it
make -C campaign bench-full FULL=requested          # every tier; only when the owner asks for it
```

Preflight checks only the selected workloads: the pinned Zig, the frozen baseline's sha256, weight files against their pinned sizes, `tf-flint-run` against its sources, and that each workload's make target exists. It reports host readiness (free disk, memory pressure, thermal state, running servers, the GPU lock holder) without blocking on it. Any unmet requirement stops the run before anything starts. An unknown name or area, or a selection that matches nothing, is an error. A full-tier workload never runs outside `bench-full FULL=requested`. Each run writes per-workload logs and `summary.json` to `.work/suite/<timestamp>/` and exits non-zero if any workload fails. Completion is not a performance verdict: read each workload's own records, which mark invalid trials.
