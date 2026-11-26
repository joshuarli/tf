# A1 Phase A: Current-Main Control and Verified Cost Model

**Lane:** `lane/inkling-metal-aot` (rebased onto `main` `98a7c1028`, post-Q4)
**Control Runner:** `tf-inkling2-loop-run` built from `98a7c1028` (`-Dcpu=apple_m1`, sha256 `ce4782cb…db69d`)
**Served Control Binary:** `tensorfold-native` (`b925b31e…15f6`, Q4 scorecard `campaign/kernels/inkling-q4-qualify.md`)
**Target Checkpoint:** `~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd` (plus DFlash2 `@4f9088ea66c2`)
**Host:** Apple M5 Max (40 GPU cores, 48 GiB, macOS `27.2` / `26B123`, Zig `0.17.1`, `85%` free memory, worst pressure `1`, `0 MB` swap growth)

## 1. Q4 Repaired Serving Qualification Control (`main` `98a7c1028`)

From `campaign/kernels/inkling-q4-qualify.md` (pinned `b925b31e…15f6`):
- **Plain served fixed-window decode (`ar`, quiet sessions `b+c`, 12 prompts):** `63.9–72.2 tok/s` (`13.85–15.65 ms/tok` at 1K–4K).
- **Loaded-drafter disabled control (`disabled`, `draft: false`):** neutral against `ar` on every prompt (`disabled/ar` `0.983–1.053x` in `b+c`, every 95% CI includes 1.0).
- **Speculative served decode (`spec`, `b+c`):** `90.2–171.1 tok/s` (`1.351–2.579x` over `ar`; best `171.3` across `a+b+c` on `code-1024`).
- **Completed tasks (`tasks-half`, 0/72 timing-invalid, 36/36 valid pairs):** `32/36` passed (`ar == spec` 100%), total time `715.2 -> 440.0 s` (`1.625x`).
- **Unseen half prefill (`ar`, quiet `b+c`):** `1,892 / 2,001 / 1,933 / 1,833 / 1,694 tok/s` (1K / 4K / 8K / 16K / 31.7K).
- **Served footprint:** `ar` ready `18.98 GB`, peak `21.15 GB`; `spec` ready `19.53 GB`, peak `21.95 GB`.
- **Cold server load (`server.engine_load_s`):** `~23.6 s` (safetensors shard copy of `16.21 GB` including `636.5 MB` dead vision weights + `144` CPU controller transposes + `25` runtime MSL library compiles / `127` PSO builds + `Host.warm` 1-token prefill).

## 2. Runner Plain 1-Row `batch_step` Control (`98a7c1028`, 9 reps, rested)

Every measurement ran under `campaign/heavy.py` (`INKLING_LOAD_LIMIT_GB=26`, worst pressure `1`, `+0 MB` swap):

| Prompt | Context | Dispatches | CBs | Syncs | Wall p50 (ms) | GPU p50 (ms) | Host gap (ms) | Plain tok/s | Bytes Floor (`521 GB/s`) | GPU Excess Above Bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `code-1024` | 1,024 | 878 | 2 | 1 | 13.471 | 13.216 | 0.255 | 74.2 | 7.819 | +5.397 ms (1.69x) |
| `prose-1024` | 1,024 | 878 | 2 | 1 | 13.365 | 13.136 | 0.229 | 74.8 | 7.819 | +5.317 ms (1.68x) |
| `structured-1024` | 1,024 | 878 | 2 | 1 | 13.447 | 13.207 | 0.240 | 74.4 | 7.819 | +5.388 ms (1.69x) |
| `code-4096` | 4,096 | 878 | 2 | 1 | 13.611 | 13.377 | 0.234 | 73.5 | 8.012 | +5.365 ms (1.67x) |
| `prose-4096` | 4,096 | 878 | 2 | 1 | 13.668 | 13.397 | 0.271 | 73.2 | 8.012 | +5.385 ms (1.67x) |
| `structured-4096` | 4,096 | 878 | 2 | 1 | 13.643 | 13.408 | 0.235 | 73.3 | 8.012 | +5.396 ms (1.67x) |
| `ctx-1024` | 1,024 | 878 | 2 | 1 | 13.449 | 13.181 | 0.268 | 74.4 | 7.819 | +5.362 ms (1.69x) |
| `ctx-4096` | 4,096 | 878 | 2 | 1 | 13.650 | 13.417 | 0.233 | 73.3 | 8.012 | +5.405 ms (1.67x) |
| `ctx-16384` | 16,384 | 878 | 2 | 1 | 14.816 | 14.588 | 0.228 | 67.5 | 8.785 | +5.803 ms (1.66x) |
| `ctx-31744` | 31,744 | 878 | 2 | 1 | 16.393 | 16.123 | 0.270 | 61.0 | 9.816 | +6.307 ms (1.64x) |

- **Peak charged memory (`heavy.py`):** `19.9 GB` on the 1K/4K V4 prompts (`17.7 GB` footprint), `18.6 GB` on the 1K–31.7K context curve (`18.6 GB` footprint, `15.7 GB` resident, `18.6 GB` wired growth).
- **Exposed host gap inside `batch_step` (`wall - gpu`):** `0.228–0.271 ms` (`1.7–2.0%`), because V4's early commit after slot 5 overlaps slots 6–47 CPU encoding with GPU execution. Full end-to-end generation also pays per-token CPU `argmax` over `200,058` `f32` logits (`800.2 KB` coherent read, `~0.35–0.50 ms` in served/runner decode).

## 3. Correctness Baselines (G1)

- **Bit-exact reference hashes (`dflash-widths 1 ... hashes`):**
  - `INKLING_PREFILL_HALF` unset (`main-hashes-unset.jsonl`): all 6 prompts × widths `1,2,4,8,16` verified `true` (grouped verify rows bit-equal to one-row steps); peak charged `20.3 GB`.
  - `INKLING_PREFILL_HALF=off` (`main-hashes-off.jsonl`): all 6 prompts × widths `1,2,4,8,16` verified `true`; peak charged `17.9 GB`.
- **oMLX oracle gate (`make -C campaign omlx-oracle-gate`):** **PASS**
  - Positions: `3,015` across `16` histories, `0` non-finite logits.
  - Top-1 agreement: `2,856 / 3,015` (`94.73%`; threshold `>= 94.392%`); confident top-1 (`margin >= 0.1`): `2,747 / 2,872` (`95.65%`).
  - Mean KL (mean of history means): `0.0199132` (threshold `<= 0.0292627`); row-weighted mean KL: `0.0670132` (threshold `<= 0.0962956`).

## 4. Verified Bandwidth and Fixed-Cost Model (Target for G3 >= 1.15x)

- **Active bytes per 1-row token:** `3,872,476,976 B` weights (`7.433 ms` at `521 GB/s`) + KV (`201.3 MB / 0.386 ms` at 1K, `302.0 MB / 0.580 ms` at 4K, `704.6 MB / 1.352 ms` at 16K, `1,241.5 MB / 2.383 ms` at 31.7K).
- **Slope vs context length (`1K -> 31.7K`):** Measured GPU delta is `+2.942 ms` (`13.181 -> 16.123 ms`) for `+1,040.2 MB` additional KV reads (`353 GB/s` effective incremental KV bandwidth vs `521 GB/s` peak -> `+0.945 ms` attention overhead at 31.7K).
- **G3 threshold (`>= 1.15x` plain decode over `main`):** Requires cutting `batch_step` from `13.45–13.65 ms` down to **`<= 11.70–11.87 ms`** (and full decode with argmax from `~14.1 ms` down to `<= 12.26 ms`), saving **`>= 1.75–1.85 ms/token`** without changing arithmetic or regressing 16K/31.7K.
- **Confirmed implication for Phases B–E:**
  - **Phase B** (static schedule, today's kernels) targets the `0.23–0.27 ms` host/driver gap -> near-zero decode gain (`~1–2%`) as expected, while establishing **G0** and **G2** (offline `.metallib` + `MTLBinaryArchive` + `schedule.bin` + `manifest.json`).
  - **Phase C** (bit-exact `weights.aot` repacking: strip `636.5 MB` vision, pre-transpose 144 controllers, interleave `(scale, bias)` `bfloat2`, pre-divide NVFP4 `global_scale`, `MTLResidencySet`) cuts load time and memory (**G2/G4**) and trims ~`0.2–0.4 ms` of scattered-read overhead.
  - **Phase D** (specialized offline Metal codegen & fusion) must harvest **`>= 1.4–2.0 ms`** from the `+5.38 ms` GPU excess: `combine + sig_mul` (`-48` dispatches), specialized multi-SG constant-LUT `i2p_gu` / `i2e_down1`, `mix -> inv` + 64-TG `ctl_down` + fused `ctl_apply + rms` (`-96` dispatches), `router + routepick` (`-48` dispatches), specialized `bits=8` interleaved `(s,b)` projections/dense, and attention split/combine tuning.
  - **Phase E** (GPU `argmax` + bounded cancellable multi-token loop) removes the `~0.35–0.50 ms` CPU argmax + per-token host sync gap, locking in **G3** cleanly.
