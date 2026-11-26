# A1 Phase C: Static Memory, Bit-Exact Weight Repacking, and Parallel Verified Load

Phase C eliminates runtime layout overhead and weight-loading bottlenecks in `tf-inkling2-aotc` and `tf-inkling2-mini-aot` while maintaining 0-bit numerical parity with `main` (`98a7c1028`).

## Changes

1. **Affine 8-bit & 4-bit `(scale, bias)` interleaving (`--repack=phase_c`):**
   - `aotc.zig` interleaves `scales` (`bf16`) and `biases` (`bf16`) into contiguous `bfloat2` pairs (`layout_interleaved_sb`) across `embed`, `lm_head`, all 24 `attn` projections (`in_proj`/`q_proj`/`k_proj`/`v_proj`, `o_proj`), and affine 4-bit experts (`layers 20..23`).
   - Reduces per-group affine metadata loads from 2 disjoint cache lines to 1 coalesced `bfloat2` load in `i2_aot_embed_sb`, `i2_aot_dense8_sb`, `i2_aot_batch_proj8_sb`, `i2_aot_expert_up4_sb`, and `i2_aot_expert_down4_sb`.
2. **Pre-divided NVFP4 global scales & trimmed unembedding:**
   - Pre-divides `w13.global_scale` and `w2.global_scale` by `2688.0` offline (`layout_predivided_nvfp4`), eliminating the runtime scalar division and buffer binding.
   - Trims `lm_head` from `201,024` padded rows to `200,058` active vocabulary rows (`layout_trimmed_lm_head`, `-2.38 MB`).
3. **Non-indirect binary archive PSOs (`--indirect=false`):**
   - Since Phase B proved Metal 3 ICB is slower than direct 2-CB encoding on M5 Max (`+0.68 ms`), Phase C compiles `pipelines.mtlarchive` with `supportIndirectCommandBuffers = false` (`layout_non_indirect`), allowing tighter Metal register allocation and removing ICB prologue overhead.
4. **Parallel 5-thread `F_NOCACHE pread` with per-chunk SHA-256 verification:**
   - `format.zig` (`planWeightChunks`) deterministically partitions `weights.aot` along layer boundaries into 5 chunks (`<= 3.5 GB` each) and stores per-chunk SHA-256 digests in `schedule.bin` alongside the whole-file digest in `manifest.json`.
   - `runtime.zig` spawns 5 worker threads doing `16 MiB` `F_NOCACHE pread` directly into Metal shared buffers while hashing each chunk in parallel, preserving fail-closed integrity verification while cutting verified weight load time by **4.1x** (`6.6–7.7 s -> 1.58–1.71 s`).
5. **Phase C2 physical weight repacking (`--repack=phase_c2`, `layout_packed_w13_gf`):**
   - Interleaves NVFP4 `w13` gate and up weight rows per 16-element group into `ulong2(wg, wu)` and `uchar2(sg, su)` so `i2_aot_gu_nv_packed` reads both gate and up from a single sequential 16-byte stream instead of 2 strided pointers (`1,152 B` apart).
   - Interleaves controller `gain` and `forget` matrices into `float2(g, f)` for `i2_aot_ctl_apply3_gf2` / `i2_aot_ctl_apply1_gf2`.

## Verification and Measurements

All GPU runs ran on a quiet host (`memory_pressure` `80–83%` free, worst pressure `1`, `+0 MB` swap) under `campaign/heavy.py` (`28 GB` cap).

### Bit-Exact Parity (G1)

- `bundle-c` (`--repack=phase_c`, `--opt=0`) vs `main` (`logits-main.bin`, `[32, 200058]` f32 = `25,607,424 B`): `cmp` exited `0` (**0 differing bits**).
- `bundle-c2` (`--repack=phase_c2`, `--opt=7`) vs `main` (`logits-main.bin`): `cmp` exited `0` (**0 differing bits**).
- Host unit tests (`zig build test-inkling2-aot-host`): `2/2` passed.

### Load, Memory (G2 / G4), and 1K Decode (`ctx-1024`)

| Bundle / Config | Load Total | Weights (`verify`) | Pipelines | Residency | `ctx-1024` Wall p50 | `ctx-1024` GPU p50 | Plain tok/s | Peak Charged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `main` (`tf-inkling2-loop-run`, Phase A `batch_step`) | ~2,300 ms | ~500 ms | ~1,800 ms (MSL) | — | 13.449 ms | 13.181 ms | 74.35 | 18.6–19.9 GB |
| Phase B `bundle-b` (`repack=none`, `indirect=true`, 1-thr SHA) | 6,627 ms | 6,410 ms | 2.1 ms | 200 ms | 14.148 ms | 13.877 ms | 70.68 | 16.0 GB |
| Phase C `bundle-c` (`repack=phase_c`, `indirect=false`, `--opt=0`) | **1,683 ms** | 1,454 ms (24 ms) | **2.0 ms** | 203 ms | **13.458 ms** | **13.225 ms** | **74.31** | **16.0 GB** |
| Phase C2 `bundle-c2` (`repack=phase_c2`, `--opt=7`, `--burst=0`) | **1,581–1,642 ms** | 1,354 ms (22 ms) | **2.2 ms** | 203 ms | **13.150 ms** | **12.918 ms** | **76.05** | **16.0 GB** |

First-token logit SHA-256 (`2bfe11565dcd`) and 32-token output SHA-256 (`8534f71c2459`) are identical across all bundles.

## Takeaways

1. **Non-indirect PSOs + interleaved `bfloat2 sb` recovered `-0.65 ms` GPU** vs Phase B (`13.877 -> 13.225 ms`), matching `main`'s `batch_step` at 1K while including CPU argmax and cutting peak memory by **2.6–3.9 GB** (`16.0 GB` vs `18.6–19.9 GB`).
2. **5-thread `F_NOCACHE pread` + per-chunk SHA-256 hits `1.58–1.68 s` total verified load** (`~11.5 GB/s` effective read+hash), beating both single-thread Phase B (`6.63 s`) and `main`'s cold server load (`~23.6 s`), with zero steady-state GPU penalty (unlike `--mmap`).
3. **Physical `ulong2(wg, wu)` + `float2(g, f)` repacking (`bundle-c2`)** is bit-exact and saves an additional `~0.15 ms` when paired with Phase D's fused NVFP4 and controller kernels.
