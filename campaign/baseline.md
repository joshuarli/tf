# Frozen baseline

The baseline is TensorFold `fa13df5dd1fdd23ada775510a464c9aa03a39e09` built with Zig 0.17.1. Its worktree overrides only `.zig-version` and `minimum_zig_version` (see Decision 3 in [docs/roadmap.md](../docs/roadmap.md)). `make -C campaign build-baseline` rebuilds it in `.work/wt/baseline`.

- `tensorfold-native` sha256: `8c2c8e6f4fd9f2c111d6fba7d89251a32a5afb309cc950ee3ff2d6a50816ac87`
- Built with `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`, Zig 0.17.1, Xcode 27.2 (27B52), Metal toolchain 27.2.47.0, macOS 27.2 (26B123). [host.json](host.json) has the full host record.
- The Metal device reports `recommendedMaxWorkingSetSize` W = 40,200,896,512 bytes, so benchmark servers set `FLINT_LOAD_LIMIT_GB=36` (floor(0.9·W/1e9)) until the owner's 32 GB campaign cap; since then they set 27 (Decision 8 in [docs/roadmap.md](../docs/roadmap.md)).

## Loop timings

Measured on 2026-11-21 by `make -C campaign baseline-timings`. The cold build ran with an empty project cache and an empty campaign Zig global cache. The pinned weights were downloading at the same time, so the cold figure may be slightly high.

| Command | Wall time |
| --- | --- |
| `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`, cold | 239.3 s |
| `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`, warm no-op | 4.0 s |
| `zig build test-flint-draft-host` | 17.7 s |
| `zig build test test-golden -Dcpu=apple_m1`, under the GPU lock | 547.6 s |
| `make -C campaign fetch NAMES="flint-int4 inkling-2-mini"`, all files already verified | 17.2 s |

## Known red

`zig build test test-golden -Dcpu=apple_m1` on the baseline: 928 of 938 steps succeeded with 7 failed, and 16,372 of 16,519 tests passed with 141 skipped and 6 failed. The goldens report 324 of 337 equal and 0 unexpected differences. Every failure, plus one reported but unverified:

| Failing step or test | Modules | Failure |
| --- | --- | --- |
| `run sh`: the `nm -m` check that `tensorfold-native` takes `_memcpy` from libSystem | 1 step | The binary has a non-external `_memcpy` in `__TEXT,__text`: "tensorfold-native links its own memcpy". |
| `the release's vision_config parses to the defaults` | `vision`, `families.deepseek_v41.metal.vision` | expected `error.UnsupportedVision`, found `error.ZeroHeads` |
| `copied shard pages leave the file cache without changing the copy or stored bytes` | `checkpoint_file_cache`, `src.core.checkpoint_file_cache` | expected 0, found 2 |
| `PDF render worker: a worker killed by a signal is named as a crash; a limit or a missing reply keeps today's message` | `pdf_worker` (two test binaries) | the `ulimit -t 1` busy loop reports "PDF rendering exceeded resource limits or failed" instead of a crash |
| `GLM copies and transforms join before shard eviction; errors and duplicate opens retain ownership` | `families.glm.weights_fix_test` | unverified: reported once by an earlier C0 worker's Zig 0.17.0 run, whose log was not kept; it did not fail in the run above |

The `copied shard pages` test passed in that same earlier run, so treat it and the GLM test as possibly flaky. The memcpy check also fails on the unmodified `fa13df5d` built with Zig 0.17.0, so it is upstream red rather than an effect of the Zig pin ([ledger.md](ledger.md)).

Logs are in `.work/logs/baseline/` while the scratch tree exists. Rerun `make -C campaign baseline-timings` to reproduce them.
