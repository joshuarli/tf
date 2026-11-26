# A1 Phase B: Offline AOT Compiler, Standalone Runner, and Static Schedule / ICB

Phase B builds the complete offline AOT toolchain (`tf-inkling2-aotc`) and standalone pure-AOT Metal runner (`tf-inkling2-mini-aot`) on `lane/inkling-metal-aot` (Apple M5 Max, `applegpu_g17s`, Metal 32023.922, Zig 0.17.1).

## Artifacts

- Compiler: `zig/src/families/inkling2_loop/aot/aotc.zig` (`tf-inkling2-aotc`, 992 KB)
- Standalone runner: `zig/src/families/inkling2_loop/aot/{main,runtime,format,binary_archive}.zig` (`tf-inkling2-mini-aot`, 670 KB)
- Module isolation: `tf-inkling2-mini-aot` imports only `metal`. `strings zig-out/bin/tf-inkling2-mini-aot | rg 'newLibraryWithSource|MTLCompileOptions|safetensors'` returns zero matches.
- Offline bundle `.work/tmp/a1/bundle-b`:
  - `weights.aot`: `15,575,220,224` B (15.58 GB — vision weights stripped, 144 controllers pre-transposed, `cross_pass` pre-split, `t16`/`t256`/`nvfp4_den` embedded)
  - `kernels.metallib`: 801,045 B (compiled offline via `xcrun -sdk macosx metal -c -std=metal4.1 -fmetal-math-mode=safe -O3` + `metallib`)
  - `pipelines.mtlarchive`: 724,352 B (31 GPU-binary PSOs serialized via `MTLBinaryArchive`)
  - `schedule.bin`: 25,816 B (immutable 48-slot + 24-layer descriptor)
  - `manifest.json`: 1.2 KB (target arch `applegpu_g17s`, min gen `17`, context cap `32768`, SHA-256 of every file)
  - Offline compile time: 16.8–33.4 s (peak charged 15.6–15.8 GB under `heavy.py`, host pressure worst 1, swap `+0 MB`).

## Gate Checks (G0, G1, G2, G4)

### G0 (Standalone & Fail-Closed)

All executed with `env -i PATH=/usr/bin:/bin HOME=/Users/josh` (no checkpoint dir, no Python, no `xcrun`, no runtime MSL compiler):

- `verify`: **PASS** (`arch=applegpu_g17s gen=17 weights=15575220224 verify_ms=19.9 pipelines_ms=4.8`)
- `--device-arch-override applegpu_g13g`: **PASS** (rejected with `error: UnsupportedDevice`)
- 1-byte bit-flip in `schedule.bin`: **PASS** (rejected with `error: DigestMismatch`)
- 1-byte bit-flip in `kernels.metallib`: **PASS** (rejected with `error: DigestMismatch`)

### G1 (Bit-Exact Differential Parity vs `main`)

Teacher-forced 32-token logits (`[32, 200058]` float32 = `25,607,424` B) against Phase A control `.work/tmp/a1/main-run/bin/tf-inkling2-loop-run`:

| Runner | Mode | Logits SHA-256 | Diff vs `main` | Peak Charged (`heavy.py`) |
| --- | --- | --- | --- | --- |
| `main` (`tf-inkling2-loop-run`) | `position` | `be8e393736532c7e36303c6f6d4c5b3a1431ff99e439ad5fff8601111266979b` | reference | 19.9 GB |
| `tf-inkling2-mini-aot` | `direct` | `be8e393736532c7e36303c6f6d4c5b3a1431ff99e439ad5fff8601111266979b` | **0 bits** | **15.2 GB** |
| `tf-inkling2-mini-aot` | `icb` | `be8e393736532c7e36303c6f6d4c5b3a1431ff99e439ad5fff8601111266979b` | **0 bits** | **16.3 GB** |

### Hardware Discovery: Metal ICB 4 GiB Offset Limit

Initial ICB execution produced `NaN` starting at slot 5 (`w13`, the first tensor crossing `4 GiB` in `weights.aot`) while `direct` was bit-exact. On Apple Silicon, `MTLIndirectComputeCommand.setKernelBuffer:offset:atIndex:` silently truncates buffer offsets `>= 4 GiB` (`0x1_0000_0000`). Partitioning `weights.aot` at load time into `< 3.5 GiB` chunk buffers along layer boundaries (zero disk format change, identical RAM) resolved the corruption to 0 differing bits.

### G2 & G4 (Startup, Memory, and Steady-State Decode)

Measured on quiet host (`memory_pressure` 85% free, worst pressure 1, `+0 MB` swap, all under `campaign/heavy.py` 28 GB cap) on `ctx-1024` and `ctx-4096` (32 measured decode steps; `mmap` 16 steps):

| Mode | Load total (`pipelines`) | `ctx-1024` wall / GPU p50 | `ctx-4096` wall / GPU p50 | Dispatches / CBs | Peak Charged |
| --- | --- | --- | --- | --- | --- |
| `main` control (Phase A `batch_step`, no argmax) | ~2,300 ms (~1,800 ms MSL) | 13.365–13.471 / 13.136–13.216 ms | 13.611–13.668 / 13.377–13.417 ms | 878 / 2 | 18.6–20.3 GB |
| AOT `direct` (CPU argmax, `pread`) | 7,660 ms (**2.1 ms**) | **13.464 / 13.250 ms** (74.27 tok/s) | 14.132 / 13.902 ms (70.76 tok/s) | 878 / 2 | **16.1 GB** |
| AOT `direct` (GPU argmax, `pread`) | 6,837 ms (**1.9 ms**) | 13.887 / 13.598 ms (72.01 tok/s) | 15.187 / 14.933 ms (65.85 tok/s) | 879 / 2 | **16.7 GB** |
| AOT `icb` (GPU argmax, `pread`) | 6,880 ms (**1.9 ms**) | 14.570 / 14.332 ms (68.63 tok/s) | 15.235 / 15.011 ms (65.64 tok/s) | 879 / 1 | **16.4 GB** |
| AOT `direct` (GPU argmax, `--mmap`) | **899.5 ms** (**1.7 ms**) | 15.329 / 15.028 ms (65.24 tok/s) | — | 879 / 2 | **16.0 GB** |

All runs matched bit-identically on first-token logits (`2bfe11565dcd` at 1K, `cd465ff196d3` at 4K) and generated token hashes (`8534f71c2459` at 1K, `1d1d67c9c3ab` at 4K).

## Phase B Takeaways

1. **Startup & memory wins are real:** binary archive pipeline load is **1.7–2.1 ms** (vs ~1,800 ms runtime MSL compile), `--mmap` reaches first step in **0.90 s**, and peak charged memory drops by **3.6–4.0 GB** (`16.0–16.7 GB` vs `19.9–20.3 GB`).
2. **ICB is NOT a decode speedup on M5 Max:** with V4's 2-CB early commit already shrinking host gap to `0.214 ms`, Metal 3 ICB adds `+0.68 ms` GPU overhead at 1K (concurrent dispatch barriers + `param_arena` loads instead of inline `setBytes`) and loses to `direct`.
3. **`--mmap` costs steady-state GPU time** (`15.03 ms` vs `13.25 ms` GPU) compared to `F_NOCACHE pread` into shared Metal buffers, and `i2v_argmax` (single 1024-thread TG) is slower than NEON CPU argmax unless hidden inside a GPU-resident multi-token loop (Phase E).
4. **Why `ctx-4096` drifted in Phase B:** Phase B fed 4,096 prompt tokens sequentially (~55 s of 100% GPU load) before timing, warming the GPU; Phase F chunked prefill eliminates that thermal pre-soak, and Phase D kernel fusions target the `>= 1.8 ms/token` GPU savings needed for G3 (`>= 1.15x`).
