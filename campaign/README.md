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

`build-baseline` creates a detached worktree of `fa13df5d` at `.work/wt/baseline`, overrides only `.zig-version` and `minimum_zig_version` to the pinned Zig, builds `tensorfold-native` and prints its sha256. `baseline-timings` reruns the cold and warm builds and the baseline suites, writing logs to `.work/logs/baseline/`. The suite that touches Metal runs under `lockf .work/gpu.lock`. [baseline.md](baseline.md) records the results and the known red tests.

## Resume

Read the **Status** block of [plan.md](plan.md), run `make -C campaign host`, compare the result with the committed `host.json`, then take the next item.
