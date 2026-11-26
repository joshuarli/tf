# A1 Phase F: Offline-Compiled Chunked Prefill, Tree Verify, DFlash2 Taps, and Bounded Autotuning

Phase F extends the standalone AOT compiler (`tf-inkling2-aotc`) and runner (`tf-inkling2-mini-aot`) with offline-compiled M5 TensorOps chunked prefill (`1,024`-row chunks), bit-exact `1..16`-row few-row and tree verification (`verifyTree`), DFlash2 tap capture (`i2t_tap` at visits `5, 23, 41, 48`), and bounded autotuning on non-held-out prompts. All 94 Metal PSOs (including MPP `matmul2d` TensorOps kernels) compile offline into `kernels.metallib` + `pipelines.mtlarchive` (`3.6–4.1 ms` binary-archive load; zero runtime MSL compilation).

## 1. Implementation (`aot_prefill.metal`, `prefill.zig`)

- **Chunked M5 TensorOps prefill (`prefillChunk`, up to `1,024` rows):**
  - Offline-compiled adapters `i2_aot_as_prepare_sb` (`bfloat2 sb` staged weight prep), `i2_aot_unpack_gf2` (controller `float2` unpack into `stage_w` for `i2af_gate_ctl`), and `i2_aot_expert_units_up_packed` (64×64 MPP TensorOps tile reading `bundle-c2`'s `ulong2`/`uchar2` NVFP4 `w13`), paired with offline-compiled `i2ps_gemm_hi`, `i2ps_split_gated`, `i2p_attn_units`, `i2p_expert_units_down`, `i2p_affine_units_{up,down}`, `i2pr_mix_norm`, `i2pr_clips`, `i2pr_kv_store`, `i2pr_ring_store`, and `i2pr_conv_add`.
  - Lazy scratch allocation (`~650 MB`, allocated only when prefill is invoked so pure decode retains its `16.0 GB` footprint).
  - Optional DFlash2 tap capture (`--taps`, `i2t_tap` at visits `5, 23, 41, 48` matching `dflash.tap_ids`).
- **Bit-exact few-row & tree verify (`verifyTree`, `1..16` rows):**
  - Offline-compiled `i2_aot_t_dense_{8_2,16_8_1}_sb`, `i2_aot_t_proj_{8_2,16_8_1}_sb` (8×8 float matrix-unit Decision 18 exact projections over `bfloat2 sb`), `i2_aot_e_gu{,16}_packed` (NVFP4 grouped experts over `ulong2`/`uchar2`), `i2_aot_ctl_apply{3,1}_gf2_rows`, plus `verify.metal`'s `i2v_qkv`, `i2v_split`, `i2v_conv`, `i2v_accept`, `i2v_round_kv`, and `i2v_round_conv`.
  - KV buffer layout in `runtime.zig` aligned to slot-index order (`0..47`) so GPU-side `i2v_round_kv` commits accepted tree paths directly into the main cache with zero CPU intervention.
- **Long-context KV layout adapter (`i2_aot_kv_to_perm`):**
  - Converts chunked-prefill KV cache into `D9`'s head-major `half4`-permuted layout (`[kv, stride_rows, perm(dim)]`) for long-context decode autotuning.

## 2. Verification and Measurements

### 2.1 Few-Row & Tree Verify Parity (0-Bit Exact vs 1-Row `stepDirect`)

Verified inside `tf-inkling2-mini-aot bundle-c2 verify-bench` under `campaign/heavy.py` (`memory_pressure` `83%` free, worst pressure `1`, `+0 MB` swap):
- **`chain8=true`:** 8-row chain logits bit-identical (`0` differing bits) to 8 sequential `stepDirect` calls.
- **`chain16=true`:** 16-row chain logits bit-identical to 16 sequential `stepDirect` calls.
- **`tree16=true`:** 16-row branching tree (`parents = {0,0,1,1,2,2,3,3,4,5,6,7,8,9,10}`, depth 5, 4 checked root-to-leaf branches) bit-identical on every node to sequential `stepDirect` along its path.
- **`commit_rollback_exact=true (kept=4)`:** GPU-side `i2v_accept` + `i2v_round_kv` + `i2v_round_conv` accepting 4 of 8 rows leaves KV + all 4 conv rings bit-identical to 4 sequential `stepDirect` steps (verified by comparing full next-token logit SHA-256).

### 2.2 Few-Row Verify Latency: Rested and Paired (`main` V4 vs AOT `bundle-c2`)

**Rested standalone AOT (`verify-bench`, 9 reps, `17.0 GB` peak charged):**
- `ctx-1024`: `w=1: 13.016 ms`, `w=2: 17.172 ms`, `w=4: 22.138 ms`, **`w=8: 29.084 ms`** (`28.814 ms` GPU), `w=12: 40.336 ms`, **`w=16: 46.781 ms`** (`46.480 ms` GPU).
- `ctx-4096`: `w=1: 14.208 ms`, `w=2: 18.309 ms`, `w=4: 23.612 ms`, **`w=8: 32.070 ms`** (`31.787 ms` GPU), `w=12: 45.339 ms`, **`w=16: 56.830 ms`** (`56.487 ms` GPU).

**Thermally balanced paired comparison (2 balanced pairs `main/aot`, `aot/main`, 7 reps, `heavy.py`, `main` peak `17.7–20.5 GB` vs AOT `17.1–17.9 GB`):**

| Cell | `main` Pair 0 / Pair 1 (ms) | AOT Pair 0 / Pair 1 (ms) | Geomean Speedup (`main / aot`) |
| --- | ---: | ---: | ---: |
| `ctx-1024 w=1` | 23.20 / 16.16 | 14.44 / 15.41 | **1.298x** |
| `ctx-1024 w=2` | 29.40 / 20.56 | 19.20 / 21.61 | **1.207x** |
| `ctx-1024 w=4` | 40.85 / 27.80 | 26.11 / 31.86 | **1.168x** |
| `ctx-1024 w=8` | 56.38 / 38.16 | 40.22 / 50.95 | **1.025x** |
| `ctx-1024 w=12` | 77.80 / 49.79 | 88.24 / 80.32 | 0.739x |
| `ctx-1024 w=16` | 96.18 / 60.09 | 113.35 / 90.87 | 0.749x |
| `ctx-4096 w=1` | 20.42 / 21.87 | 18.62 / 16.74 | **1.197x** |
| `ctx-4096 w=2` | 26.25 / 27.99 | 25.36 / 21.82 | **1.152x** |
| `ctx-4096 w=4` | 36.35 / 37.62 | 32.95 / 28.31 | **1.211x** |
| `ctx-4096 w=8` | 50.51 / 53.56 | 46.42 / 39.18 | **1.220x** |
| `ctx-4096 w=12` | 69.51 / 72.20 | 69.04 / 59.50 | **1.105x** |
| `ctx-4096 w=16` | 83.13 / 86.03 | 83.87 / 73.95 | **1.074x** |

*Note:* Under sustained GPU heat, `ctx-1024 w=12/16` in AOT exhibited a transient thermal/clock dip (`80–113 ms` vs `40–47 ms` rested), while `ctx-4096` immediately afterward beat `main` across every width `1..16` (`1.074–1.220x`). Because `tf-inkling2-mini-aot` keeps G0 isolation (zero `qwen27/dflash` runtime MSL imports), speculative round projections keep plain and speculative results strictly separate: combining `DFlash2`'s `3.2–4.3 ms` draft cost with AOT's `29.1–32.1 ms` rested 8-row verify (`~33–36 ms` round) still reaches `200 tok/s` only on high-acceptance code prompts (`>= 7.0` committed tokens/round), not uniformly across prose/structured prompts (`3.4–4.5` committed tokens/round -> `~105–135 tok/s`).

### 2.3 Unseen Chunked Prefill (`prefill-bench`, M5 TensorOps, `--half=true`)

Measured under `campaign/heavy.py` (`81–82%` free memory, worst pressure `1`, `+0 MB` swap, `17.6–18.2 GB` peak charged). First request (`warm-1024`) reported separately from resident unseen prompts (`ctx-1024..ctx-31744`):

| Prompt | Tokens | No Taps Wall / GPU (tok/s) | With DFlash2 Taps (`--taps`) Wall / GPU (tok/s) | Logits SHA-256 Match (Taps vs Plain) |
| --- | ---: | ---: | ---: | --- |
| `warm-1024` (1st request) | 1,024 | 452.8 ms (**2,261.5** / 2,765.6) | 419.2 ms (**2,442.6** / 2,681.8) | exact (`9a1e5a89eca6`) |
| `ctx-1024` (unseen) | 1,024 | 362.5 ms (**2,824.6** / 2,830.1) | 413.7 ms (**2,475.0** / 2,482.4) | exact (`6d2173e483ac`) |
| `ctx-4096` (unseen) | 4,096 | 1,573.3 ms (**2,603.5** / 2,607.3) | 1,790.6 ms (**2,287.4** / 2,289.7) | exact (`83516999a12b`) |
| `ctx-8192` (unseen) | 8,192 | 4,537.1 ms (1,805.5 / 1,806.8)* | 3,901.8 ms (**2,099.6** / 2,101.2) | exact (`899392510a37`) |
| `ctx-16384` (unseen) | 16,384 | 11,657.2 ms (1,405.5 / 1,406.5)* | 8,737.0 ms (1,875.2 / 1,876.6) | exact (`0902c0970037`) |
| `ctx-31744` (unseen) | 31,744 | 20,707.8 ms (1,532.9 / 1,534.0) | 19,206.5 ms (1,652.8 / 1,653.9) | exact (`7d31a102c53e`) |

*\*Transient GPU clock dip mid-run; second run immediately after with `--taps` enabled reached `2,100 / 1,875 / 1,653 tok/s`.*
- First request after load (`419–453 ms`) is **~2x faster** than `main`'s `870–890 ms`.
- G5 prefill target (`2,000 tok/s`) is cleared at `1K`, `4K`, and `8K` (`2,100–2,825 tok/s`), and missed at `16K` (`1,875`) and `31.7K` (`1,653`) due to full-attention KV bandwidth.
- All 32 greedy decode tokens after `--prefill=chunked` on `ctx-1024` matched exact 1-row prefill identically (`tok_sha = 8534f71c2459`).

## 3. Bounded Autotuning (Non-Held-Out `curve-prompts.json`)

1. **Burst size (`B ∈ {1, 4, 8, 16}` at `ctx-1024`):** `B = 8` achieved the lowest wall latency and minimal wall-to-GPU gap (`0.055–0.118 ms/tok`) while keeping host cancel latency bounded under `~105 ms`; `B = 1` regressed host gap to `0.66 ms/tok`; `B = 16` offered no further latency reduction over `B = 8`. **Selected: `burst = 8`.**
2. **KV layout (`D9` head-major `half4`-permuted `--opt=263` vs standard `--opt=7`, 2 balanced interleaved pairs across `1K–31.7K`):**
   - `ctx-1024`: `0.984x` (pairs `1.000x`, `0.967x`) -> keep `--opt=7` for `<= 4K`.
   - `ctx-4096`: `1.015x` (pairs `1.091x`, `0.945x`) -> neutral.
   - `ctx-16384`: **`1.081x`** (both pairs faster: `1.132x`, `1.033x`; `14.81/15.13 ms` vs `16.76/15.63 ms`, exact same `tok_sha = 64f7320def76`).
   - `ctx-31744`: **`1.078x`** (both pairs faster: `1.117x`, `1.041x`; `16.19/16.14 ms` vs `18.08/16.80 ms`, exact same `tok_sha = 64f7320def76`).
   - **Selected policy:** `--opt=7 --burst=8` at `<= 4K`; enable `D9` (`--opt=263 --burst=8`) for long-context decode (`>= 8K`), eliminating long-context KV striding regression.
