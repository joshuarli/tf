# Frozen baseline

The baseline is TensorFold `fa13df5dd1fdd23ada775510a464c9aa03a39e09` built with Zig 0.17.1. Its worktree overrides only `.zig-version` and `minimum_zig_version` (see Decision 3 in [plan.md](plan.md)). `make -C campaign build-baseline` rebuilds it in `.work/wt/baseline`.

- `tensorfold-native` sha256: `8c2c8e6f4fd9f2c111d6fba7d89251a32a5afb309cc950ee3ff2d6a50816ac87`
- Built with `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`, Zig 0.17.1, Xcode 27.2 (27B52), Metal toolchain 27.2.47.0, macOS 27.2 (26B123). [host.json](host.json) has the full host record.
- The Metal device reports `recommendedMaxWorkingSetSize` W = 40,200,896,512 bytes, so benchmark servers set `FLINT_LOAD_LIMIT_GB=36` (floor(0.9·W/1e9)).

## Loop timings

Measured on 2026-11-21 by `make -C campaign baseline-timings`. The cold build ran with an empty project cache and an empty campaign Zig global cache. The pinned weights were downloading at the same time, so the cold figure may be slightly high.

| Command | Wall time |
| --- | --- |
| `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`, cold | 239.3 s |
| `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`, warm no-op | 4.0 s |
| `zig build test-flint-draft-host` | 17.7 s |
| `zig build test test-golden -Dcpu=apple_m1`, under the GPU lock | 547.6 s |
| `make -C campaign fetch NAMES="flint-int4 inkling-2-mini"`, all files already verified | 17.2 s |

An independent reproduction with Zig 0.17.0 on the pristine `fa13df5d` tree (2026-11-21, `/usr/bin/time -p`):
`zig build native -Dcpu=apple_m1 -Dversion=1.0.0` cold **209.28 s** (user 203.13, sys 7.74);
warm no-op **3.57 s** (user 3.06, sys 0.37); `zig build test-flint-draft-host -Dcpu=apple_m1` **15.18 s**
(user 19.25, sys 1.37, exit 0); `tensorfold-native` sha256
`9a8968ac165be5c4990faf57d0c719e9c4440a61e08fcbc229d38c80f78285a9` (Zig 0.17.0 build).

## Known red

`zig build test test-golden -Dcpu=apple_m1` on the baseline: 928 of 938 steps succeeded with 7 failed, and 16,372 of 16,519 tests passed with 141 skipped and 6 failed. The goldens report 324 of 337 equal and 0 unexpected differences. Every failure:

| Failing step or test | Modules | Failure |
| --- | --- | --- |
| `run sh`: the `nm -m` check that `tensorfold-native` takes `_memcpy` from libSystem | 1 step | The binary has a non-external `_memcpy` in `__TEXT,__text`: "tensorfold-native links its own memcpy". |
| `the release's vision_config parses to the defaults` | `vision`, `families.deepseek_v41.metal.vision` | expected `error.UnsupportedVision`, found `error.ZeroHeads` |
| `copied shard pages leave the file cache without changing the copy or stored bytes` | `checkpoint_file_cache`, `src.core.checkpoint_file_cache` | expected 0, found 2 |
| `PDF render worker: a worker killed by a signal is named as a crash; a limit or a missing reply keeps today's message` | `pdf_worker` (two test binaries) | the `ulimit -t 1` busy loop reports "PDF rendering exceeded resource limits or failed" instead of a crash |
| `GLM copies and transforms join before shard eviction; errors and duplicate opens retain ownership` | `families.glm.weights_fix_test` | fails at fa13df5d on Zig 0.17.0 (see below); not observed in the 0.17.1 run above |

A second reproduction of the same command on the pristine `fa13df5d` tree with Zig 0.17.0
(`lockf ... zig build test test-golden -Dcpu=apple_m1`, 2026-11-21) gave 931/938 steps succeeded
(5 failed), 5,238/5,300 tests passed (57 skipped, 5 failed), real 226.59 s (user 139.46, sys 51.95)
under the GPU lock. Its failed tests are: the `vision_config` test in `vision` and in
`families.deepseek_v41.metal.vision`, the `GLM copies and transforms join before shard eviction`
test, and the `PDF render worker` test in `pdf_worker` (two binaries). The `copied shard pages`
failure above did not reproduce on this run. Every failing test name from both runs is known red at fa13df5d.

The memcpy check may depend on the toolchain. Whether the unmodified `fa13df5d` built with Zig 0.17.0 also fails it is recorded in [ledger.md](ledger.md).

Logs are in `.work/logs/baseline/` while the scratch tree exists. Rerun `make -C campaign baseline-timings` to reproduce them.
