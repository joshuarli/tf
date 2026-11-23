# Zig conformance

This page records how TensorFold meets the shared Zig 0.17.1 project standard. TensorFold's own gates stay in
[CONTRIBUTING.md](../CONTRIBUTING.md) and the [native contributor skill](recipes/SKILL.md): the kernel contract,
the Metal, CUDA and HIP builds, and `-Dcpu` and `-Dfixed-cpu` selection.

## Toolchain

`.zig-version` and `minimum_zig_version` in `build.zig.zon` are both 0.17.1. CI's `mlugg/setup-zig` steps omit
`version:`, so the action reads `minimum_zig_version`. Zig cache keys use `hashFiles('build.zig.zon')`, so a
toolchain change starts a new cache.

## Entry points

| Command | macOS | Linux |
| --- | --- | --- |
| `make test` | `zig build test test-golden -Dcpu=apple_m1` | `zig build -Dfixed-cpu test-host --test-timeout 20m` |
| `make lint` | `zig build dev-tools test-kernel-contract -Doptimize=Debug -Dcpu=apple_m1` | `zig build -Dfixed-cpu test-kernel-contract` |

`make lint` compiles and links every developer tool on macOS and checks kernel-contract registration. It runs no
formatter.

## Known failures

Under Zig 0.17.1 on macOS, `make test` fails on the five tests below. Unmodified upstream TensorFold 1.3.4 fails
identically under the same compiler, and upstream itself requires 0.17.0. A known failure never excuses a new one:
any failure not listed here is a regression.

| Test | Symptom |
| --- | --- |
| `persistence_identity_test`: "persistence identity: tokenizer and template edits reuse only identical canonical token prefixes" | The canonical prefix hash differs from the recorded value. |
| `vision`: "the release's vision_config parses to the defaults" | Returns `error.ZeroHeads` instead of `error.UnsupportedVision`. |
| `families.deepseek_v41.metal.vision`: "the release's vision_config parses to the defaults" | The same test in its second module, with the same symptom. |
| `pdf_worker`: "PDF render worker: a worker killed by a signal is named as a crash; a limit or a missing reply keeps today's message" | Expected "PDF rendering exceeded resource limits or failed" and got a different message. |
| `tensorfold-native` libc memcpy check (`zig/build/metal_server.zig`) | The binary links its own `memcpy` rather than libSystem's. |

The `pdf_worker` test also runs, and fails, in a second test module.

## Published artifacts

`zig build dist` (`zig/build/dist.zig`, packaged by `tools/release/package.sh`) builds every published server
with a fixed optimize mode, independent of `-Doptimize`:

| Archive | Target | CPU baseline | Optimize |
| --- | --- | --- | --- |
| `macos-arm64` | aarch64-macos 13.0+ | `apple_m1` | ReleaseFast |
| `linux-x86_64` | x86_64-linux-gnu, glibc 2.28 | x86_64 baseline | ReleaseFast |
| `linux-aarch64` | aarch64-linux-gnu, glibc 2.28 | aarch64 baseline | ReleaseFast |

No published artifact targets the host's native CPU. The CUDA servers carry SASS for sm_80, sm_86, sm_120 and
sm_121 (`release_sms` in `zig/build/cuda.zig`).
