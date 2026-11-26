# A1 (5i) Final Verdict and Independent Adversarial Audit

**Branch:** `lane/inkling-metal-aot` (rebased onto `main` `98a7c1028`, post-Q4)
**Target:** `Inkling-2-Mini-oQ4e-mtp@3eea042751cd` on Apple M5 Max (`applegpu_g17s`, 40 GPU cores, 48 GiB, macOS `27.2`, Zig `0.17.1`)
**Overall Verdict:** **DO NOT PROMOTE AS A FULL RUNTIME REPLACEMENT (G3 NEAR-MISS: `1.120x` vs `1.15x` GATE); SELECTIVE EXTRACTION RECOMMENDED** (G0, G1, G2, G4 all **PASS** cleanly; G3 plain decode reaches a statistically significant **`1.120x`** with 95% CI `1.052–1.193x` and zero long-context regression, falling `3.0` percentage points short of the `1.15x` bar under unchanged bit-exact arithmetic).

---

## 1. Gate Scorecard (G0–G5)

| Gate | Requirement (`docs/roadmap.md` 5i) | Measured Result | Verdict |
| --- | --- | --- | --- |
| **G0** | Standalone native artifact; no checkpoint dir, Python, or Metal compiler; fails closed on incompatible device or corruption | Zero forbidden symbols (`newLibraryWithSource`, `MTLCompileOptions`, `safetensors`, `xcrun`, `python3`) in `tf-inkling2-mini-aot`; runs `verify` and `logits` (0-bit exact) under `env -i PATH=/usr/bin:/bin`; rejects `applegpu_g13g` & `applegpu_g16s` (`UnsupportedDevice`); rejects 1-byte flips in `schedule.bin`, `kernels.metallib`, `pipelines.mtlarchive`, `manifest.json` (`DigestMismatch`/`SyntaxError`) | **PASS** |
| **G1** | Existing Inkling host tests, bit-exact differentials, and oMLX oracle passing | `zig build test-inkling2-loop-host test-inkling2-aot-host` green; `0` differing bits vs `main` (`cmp` on `[32, 200058]` f32 = `25,607,424 B` under `env -i`); `chain8`/`chain16`/`tree16`/`commit_rollback_exact` all `0`-bit exact; **oMLX oracle (`3,015` positions, `16` histories):** `2,865/3,015` top-1 (`95.02% >= 94.392%`), mean KL `0.02089 <= 0.02926`, row-weighted KL `0.07408 <= 0.09630`, `0` non-finite | **PASS** |
| **G2** | Work demonstrably moved offline | All `94` Metal 4.1 / MPP TensorOps PSOs compiled offline via `xcrun metal` + `MTLBinaryArchive` (`3.6–4.1 ms` runtime load vs `~1,800 ms` MSL compile); `636.5 MB` vision stripped, `144` controllers pre-transposed/interleaved, affine `(scale, bias)` interleaved `bfloat2`, NVFP4 `w13` interleaved `ulong2`/`uchar2` + global scales pre-divided; total verified 5-thread `F_NOCACHE pread` + SHA-256 load **`1.52–1.71 s`** (vs `~23.6 s` cold server load on `main`) | **PASS** |
| **G3** | `>= 1.15x` geomean plain decode over `main` with 95% CI excluding `1.0`, no unexplained long-context regression | **6 balanced interleaved pairs across 2 sessions (V4 6 prompts `1K/4K`):** wall geomean **`1.1202x`** (95% CI **`1.0521–1.1927x`**, 6/6 pairs faster), GPU geomean **`1.1073x`** (95% CI `1.0394–1.1796x`) vs `main` `batch_step` (note AOT includes GPU argmax while `main` `batch_step` omits argmax). **Long-context curve (`4` pairs, `2` sessions, `D9` `--opt=263`):** `1K: 1.075x`, `4K: 1.018x`, `16K: 1.012x`, `31.7K: 1.033x` — zero long-context regression | **FAIL** *(near-miss: `1.120x < 1.15x`)* |
| **G4** | `<= 28 GB` cap (`heavy.py`), `<= 32 GB` budget, `32,768` context, usability | V4 peak charged **`16.7–17.0 GB`** (vs `main` `17.9–20.4 GB`, `-1.2 to -3.4 GB`); 32K context peak charged **`17.6–18.8 GB`** (vs `main` `18.6–20.3 GB`); `1.52–1.71 s` verified load; bounded cancellable burst (`cancel-test` PASS) | **PASS** |
| **G5** | Stretch: `2,000 tok/s` unseen prefill across curve; `200 tok/s` single-stream decode | Prefill clears `2,000 tok/s` at `1K` (`2,475–2,825`), `4K` (`2,287–2,604`), and `8K` (`2,100`), misses at `16K` (`1,875`) and `31.7K` (`1,653`); first request `419–453 ms` (`~2x` faster than `main`). Plain decode `78–80 tok/s` rested at `1K` (`12.45–12.81 ms`, physically capped at `134.5 tok/s` by the `7.433 ms` weight floor). Speculative decode (separate) remains below `200 tok/s` on non-code prompts (`~105–135 tok/s`) | **FAIL** *(partial)* |

---

## 2. Formal G3 Paired Results (6 Pairs / 2 Sessions V4 + 4 Pairs / 2 Sessions Context Curve)

All runs executed via `.work/tmp/a1/run_formal_eval.py` under `campaign/heavy.py` (`28 GB` cap, `memory_pressure` checked `>= 50%` free before every process — observed `81–83%` free, swap `+0 MB`, 3 s page-unwire pause between processes, 25 s cooldown between sessions).

### 2.1 Primary Plain Decode (6 V4 Prompts, `main` `batch_step` vs AOT `bundle-c2 --opt=7 --burst=8`)

| Prompt | Context | Paired Wall Geomean (`main / aot`) | 95% CI | Pair Ratios (S1 p0..p2, S2 p3..p5) |
| --- | ---: | ---: | --- | --- |
| `code-1024` | 1,024 | **1.148x** | `1.070–1.231x` | `1.071, 1.104, 1.256, 1.236, 1.142, 1.092` |
| `prose-1024` | 1,024 | **1.139x** | `0.988–1.313x` | `1.079, 1.045, 1.390, 0.982, 1.300, 1.092` |
| `structured-1024` | 1,024 | **1.249x** | `0.971–1.605x` | `1.067, 1.840, 1.343, 0.947, 1.390, 1.092` |
| `code-4096` | 4,096 | **1.130x** | `1.033–1.236x` | `1.206, 1.223, 1.078, 1.004, 1.224, 1.064` |
| `prose-4096` | 4,096 | **1.055x** | `0.944–1.178x` | `1.294, 0.963, 1.023, 1.003, 1.070, 1.006` |
| `structured-4096` | 4,096 | **1.016x** | `0.943–1.094x` | `1.095, 0.893, 1.056, 1.019, 1.048, 0.996` |
| **Pooled 6-Prompt Wall** | **1K/4K** | **1.1202x** | **`1.0521–1.1927x`** | `1.132, 1.143, 1.182, 1.028, 1.189, 1.056` |
| **Pooled 6-Prompt GPU** | **1K/4K** | **1.1073x** | **`1.0394–1.1796x`** | `1.116, 1.137, 1.167, 1.015, 1.176, 1.043` |

### 2.2 Long-Context Curve (4 Balanced Pairs / 2 Sessions, `main` vs AOT `bundle-c2 --opt=263` `D9` `--burst=8`)

| Context | Paired Wall Geomean | 95% CI | Paired GPU Geomean | Peak Charged (`main` vs AOT) |
| --- | ---: | --- | ---: | --- |
| `1,024` | **1.075x** | `1.009–1.145x` | **1.095x** | `18.6–20.3 GB` vs `17.6–18.8 GB` |
| `4,096` | **1.018x** | `0.864–1.201x` | **1.008x** | (same process) |
| `16,384` | **1.012x** | `0.873–1.172x` | **1.000x** | (same process) |
| `31,744` | **1.033x** | `0.857–1.245x` | **1.023x** | (same process) |

---

## 3. Independent Adversarial Audit Findings

Executed directly on `gemini-4-neon-high` against `bundle-c2` (`423e13133` + Phase F):

1. **G0 Isolation & Fail-Closed:**
   - Binary symbol audit (`strings`): `0` matches for `newLibraryWithSource`, `MTLCompileOptions`, `safetensors`, `xcrun`, `python3`.
   - Clean environment execution (`env -i PATH=/usr/bin:/bin HOME=/Users/josh`, no checkpoint path referenced): `verify` passed (`verify_ms=22.3`, `pipelines_ms=3.6`), `logits` ran 32 rows in `394.9 ms` (`12.34 ms/row`, `14.0 GB` peak charged) and matched `logits-main.bin` with `0` differing bits (`cmp` exit 0).
   - Device mismatch rejection: both `--device-arch-override applegpu_g13g` and `applegpu_g16s` exited non-zero (`error: UnsupportedDevice`).
   - Tamper resistance: 1-byte XOR flip at offset 16 in each of `schedule.bin`, `kernels.metallib`, `pipelines.mtlarchive`, and `manifest.json` was rejected fail-closed (`DigestMismatch` / `SyntaxError`).
2. **G1 Correctness & Oracle:**
   - Existing host tests (`zig build test-inkling2-loop-host test-inkling2-aot-host`) passed cleanly.
   - Full `16`-history (`3,015` positions) oMLX oracle check passed all thresholds with zero non-finite logits (`95.02%` top-1, `0.02089` mean KL, `0.07408` row-weighted mean KL).
   - Phase F `verifyTree` passed 0-bit parity on 8-row chain, 16-row chain, 16-row depth-5 branching tree, and GPU commit/rollback.
3. **Key Technical Deficiencies / Honest Caveats Identified by Audit:**
   - **G3 short of `1.15x`:** Under strict bit-exact arithmetic, M5 Max plain decode reaches `1.120x` over `main` `batch_step` (`12.45–12.81 ms` rested at 1K vs `13.45 ms` on `main`). The remaining `~2.1 ms` above the `10.8 ms` isolated kernel sum sits in serial barrier drain across 782 dependent dispatches of Inkling 2 Mini's 48-slot recurrent graph, which cannot be fused into a persistent megakernel without unsafe global barriers or register-pressure occupancy drops (proven across 12 parked Phase D variants).
   - **No standalone AOT DFlash2 drafter engine:** To preserve G0 purity (no runtime MSL compilation from `qwen27/dflash`), Phase F implemented AOT tap capture (`i2t_tap`) and bit-exact `1..16`-row tree verify + commit, but did not port the 3,600-line `qwen27/dflash` runtime into `aot/`.
   - **Thermal sensitivity on sustained high-width verify:** Under back-to-back GPU soak, `ctx-1024 w=12/16` spiked in `eval2` (`80–113 ms` vs `40–47 ms` rested), whereas `ctx-4096` beat `main` across all widths (`1.074–1.220x`).
   - **Back-to-back `heavy.py` teardown artifact:** Launching two `16 GB` AOT processes with `0 ms` gap triggered a transient `28.1 GB` charge in `heavy.py` during macOS wired-page drain even though actual footprint stayed `17.0 GB`; a `3 s` settle pause eliminated it completely.

---

## 4. Actionable Extractions Back to `main` (Even Without Full AOT Promotion)

Even though A1's full standalone AOT runtime misses the `1.15x` G3 promotion gate (`1.120x`), six isolated, low-risk wins proven here transfer directly to `main`:
1. **`MTLBinaryArchive` + offline `.metallib` PSO caching** (`1,800 ms -> 3.6 ms` pipeline load; cuts cold startup dramatically).
2. **Parallel 5-thread `F_NOCACHE pread`** (`1.35–1.65 s` verified load of `15.57 GB`, `-3 GB` peak memory by stripping `636.5 MB` dead vision weights and pre-transposing controllers).
3. **Phase D bit-exact kernels** (`i2_aot_combine_sig_mul`, `i2_aot_gu_nv` + `i2_aot_down1_nv` TG-SRAM LUTs, `i2_aot_mix{2,}_inv` + 6-way unrolled `i2_aot_ctl_down{3,1}`): **`-0.75 ms` GPU** (`-96` dispatches/step) with `0`-bit diff.
4. **Phase E 2-stage parallel GPU `argmax`** (`i2_aot_argmax_p1`/`p2`): replaces slow single-TG `i2v_argmax` (`+0.35 ms`) and CPU argmax (`~0.4 ms`) with a `~0.02 ms` deterministic GPU reduction.
5. **`D9` head-major `half4`-permuted KV layout** (`i2_aot_qkv_perm`/`i2_aot_split_perm`): **`+7.8–8.1%`** faster long-context decode at `16K–31.7K` (`0`-bit diff).
6. **Hardware errata documented:** Metal 3 ICB `setKernelBuffer:offset:atIndex:` silently truncates offsets `>= 4 GiB` on Apple Silicon; ICB itself is `+0.68 ms` slower than direct 2-CB encoding on M5 Max.
