# P8: Long-Context Prefill Optimization (`1K–31.7K >= 2,000 tok/s`)

## Summary

Eliminated the dominant bottlenecks identified by P6's Metal 4 timeline profile (`buckets.md`) across `1K–31.7K` prefill without changing numerical behavior:

1. **Routed expert tensor-unit tiles (`BM = 64 → 128`, `4 → 8` SGs, `prefill_expert_wide.metal`, `prefill.zig`):** Doubled rows per threadgroup (`BM = 128, BN = 64, 256` threads) with the same per-thread register count and `9.2 KB` weight tile, eliminating duplicate DRAM reads of `w13` and `w2` for experts receiving `65–128` rows in a `1,024`-row chunk, and replaced inner-loop `i2p_e4m3()` (`exp2`) with a 256-entry `threadgroup half e4m3_lut[256]`.
2. **Eliminate empty grouped-expert dispatches (`prefill.zig`):** Guarded `wide_gu_nv` and `wide_down_nv` with `if (!s.rounding.experts)` (under `s.rounding.experts == true`, `I2P_MATRIX_THRESHOLD == 1` caused `19.5M` empty threadgroups per chunk, costing `15.4 ms/chunk`).
3. **Long-context tensor-unit attention (`i2p_attn_units`, `prefill_expert_wide.metal`):** Added 128-bit `ulong2` coalesced KV loads, hoisted `far_bias[2] = B[..., extent]` once per thread before the KV loop, and added a fast `full_cache` / `far_block` path (`!a.window && block + KB - 1 <= first - int(a.extent)`) that eliminates ~6 billion `B[...]` loads per 31.7K chunk (`18.80 → 9.78 ms/layer`, `1.92×` faster, `0.00e+00` max diff).
4. **Staged half GEMM & split-write trim (`prefill_staged.metal`, `prefill_staged.zig`, `affine_stage.metal`):** Switched `i2ps_gemm_hi` to `matmul2d_descriptor(128, 64, dynamic_extent, ..., mode::multiply)` (`28%` faster) and skipped writing unused `lo` halves in `i2as_split_hi` and `i2ps_split_gated` when `half_inputs` is active.
5. **Skip doomed fork-point store (`inkling2_loop_host.zig`):** Skip allocating and copying the intermediate fork-point cache entry (`capture.at`) when `entryBytes(capture.at) + entryBytes(prompt.len) > store_limit`, where `store(prompt.len)` would immediately evict it inside the same `admit()` call (`-1.24 GB` transient allocation + blit at `31.7K`).

## Correctness

- `zig build test-inkling2-loop-host --summary all`: `25/25` passed.
- `zig build test-inkling2-loop-filtered -Dinkling-filter="tensor-unit" --summary all`: `7/7` passed (`expert units` and `affine units` error bounds identical to the last digit; `i2p_attn_units` passes all 3 cases including `base=160 rows=77`).
- `zig build test-inkling2-loop-filtered -Dinkling-filter="split-half" --summary all`: `5/5` passed.
- `python3 -I campaign/suite/suite.py run --names inkling-omlx-oracle`: **PASS** (`2,856/3,015 = 94.73%` top-1, mean KL `0.0199132`, row-weighted `0.0670132`, max KL `16.8877` at `cells_code:378` — identical to `main`).

## Performance (`oQ4e`, `INKLING_LOAD_LIMIT_GB=26`, half prefill default)

Standalone `tf-inkling2-prefix-check time` (`31,744` tokens, `31 × 1,024` chunks): `13,759.4 ms` GPU (`2,307 tok/s`), `14,123.8 ms` wall including cold first-chunk compile (`2,248 tok/s`), peak charged `18.7 GB`.

Gated served prefill curve (`tensorfold-native` ReleaseFast, drafter loaded, `pre=[] post=[]`):

| Length | Tokens | P6/main (tok/s) | P8 served with drafter (tok/s) | Time (s) | vs Decision 17 (`>= 2,000`) |
| --- | ---: | ---: | ---: | ---: | --- |
| `1K` dev (3 prompts) | 1,027–1,033 | ~2,002 (ar) / 2,654 (P6 ar) | **2,521–2,690** | 0.384–0.407 | **PASS** |
| `1K` held-out (3 prompts) | 1,003–1,019 | 2,415 (P6 ar) | **2,527–2,688** | 0.379–0.397 | **PASS** |
| `4K` | 3,882 | 2,249 (P6 ar) | **2,496** | 1.555 | **PASS** |
| `8K` | 8,594 | 2,092 (P6 ar) | **2,384** | 3.605 | **PASS** |
| `16K` | 16,593 | 1,879 (P6 ar) | **2,223** | 7.465 | **PASS** |
| `31.7K` | 31,627 | 1,749 (P6 ar) | **2,057** | 15.376 | **PASS** |
