# V5: Bit-exact decode, grouped verify, and DFlash2 drafter wins

**Verdict.** Promoted as bit-exact decode/verify improvements (`max_abs_diff = 0e0` across widths `1, 4, 8, 16` at both `1K` and `4K`, and `equal = true` on all six `dflash` prompts). Rested grouped verify on `code-1024` drops from V4's `16.2 / 27.9 / 38.7 / 61.4 ms` (`w = 1, 4, 8, 16`) to **`14.05 / 24.06 / 32.48 / 50.30 ms`** (`gpu 13.68 / 23.70 / 32.13 / 49.92 ms`, `-2.15 / -3.84 / -6.22 / -11.10 ms`). Plain 1-row decode reaches `14.05 ms` (`71.2 tok/s`), `1.8 ms` faster than Splash 1.7.2's `15.84 ms` on the same weights. DFlash2 `q4` drafting drops from `4.90 ms` to **`4.11 ms` wall p50** (`4.52 -> 3.70 ms` GPU busy). On `code-1024`, speculative decode reaches **`235.0 tok/s`** (above the `200 tok/s` target and Splash's `232 tok/s`).

## What changed

All changes live in `zig/src/families/inkling2_loop/` (`4b8b840c1`, `baf10f62a`, `4775819fd`):

1. **Bit-manipulation E4M3 decode** (`kernels/experts.metal`, `kernels/prefill.metal`): replaces `exp2()` in `i2_e4m3` / `i2p_e4m3` with IEEE-754 bit construction, preserving NaN and subnormal parity.
2. **Fused attention combine + sigmoid gate** (`kernels/attention.metal`, `batch.zig`, `group.zig`, `prefill.zig`): `i2_batch_combine_sig_mul` eliminates 48 dispatches per step.
3. **2-stage parallel GPU argmax** (`kernels/verify.metal`, `group.zig`): `i2v_argmax_p1` (`64 x rows` threadgroups) + `i2v_argmax_p2` replaces single-threadgroup vocabulary scans over `200,058` logits.
4. **Tiled routed affine4 experts** (`kernels/experts.metal`, `rows.zig`, `prefill.zig`): `i2e_aff4_up`, `i2e_aff4_up16`, `i2e_aff4_down`, `i2e_aff4_down16` group members per expert for layers 20–23 at `rows = 2..16` using exact `i2a_dot` order (`count <= 2`) and bit-exact `simdgroup_matrix<float, 8, 8>` (`count >= 3`), leaving `rows = 1` unchanged.
5. **DFlash2 affine8 head raw tiles** (`dflash.zig`): sets `raw_tiles = true` on the drafter's `quant_gpu.Kernels` so the 7-row `200,064 x 2,304` draft head runs `simd_qmm_bits` (`1` read of the `492 MB` head) instead of 4 narrow `batch = 2` passes (`1.97 GB`), cutting draft wall p50 from `4.90 ms` to `4.11 ms` with identical trees and committed tokens.

## What was tested and rejected

- **Multi-row scalar controller (`i2r_controls_apply_batch`, reverted in `baf10f62a`):** dynamic `acc[16]` spilled registers and starved cores (`controllers` regressed `4.80 -> 13.27 ms` at `w = 16`).
- **4-row shared-KV attention (`i2v_split_shared`, reverted in `4775819fd`):** holding `qv[4]` + `acc[4]` halved occupancy and regressed `attn_core` by `+1.32 ms` at `w = 8` and `+1.59–2.20 ms` at `w = 16` (`1K`/`4K`); 1-row `i2v_split` already shares KV through L2.
- **Unified `matmul2d(16, 32, 16)` decode prototype (`decode_units.metal`, not merged):** proven 0-bit row-invariant across `w = 1, 4, 8, 16` on the 48-slot checkpoint and cut `projections + oproj + head + affine_experts` by `-4.67 ms` at `w = 16`, but regressed `w = 1` by `+4.46 ms` (`13.65 -> 18.11 ms`, failing Decision 18 check 4) and regressed NVFP4 experts (`+4.67 ms` at `w = 8`, `+1.81 ms` at `w = 16`) because real Inkling routes average only `2.08` rows/expert at `w = 8` and `3.05` rows/expert at `w = 16`, leaving `80%+` of 16 M-rows wasted.
- **Indirect compact expert dispatch (`plan_experts`/`plan_one` microbenchmarks):** added serial GPU barriers and doubled TG memory for `count > 8`, measuring slower than single-dispatch `i2e_routed` (`15.97` vs `15.40 ms` at `w = 8`, `25.41` vs `24.79 ms` at `w = 16`).
- **`i2t_dense16_8_2` (`T = 2, G = 2`) and lowering `Plan.default_draft_ms` to `4.1`:** `T = 2, G = 2` doubled TG memory to `8 KB` and regressed `w = 16` by `+1.3 ms`; lowering `default_draft_ms` caused `Plan` to select narrower trees and reduced committed tokens per round. Both were reverted.

## Measurements (`tf-inkling2-loop-run`, ReleaseFast)

Rested `dflash-widths 5` on `code-1024` (`max_abs_diff = 0e0` at every width):

| Width | V4 wall p50 | V5a2 wall p50 | V5a3 (no `split_shared`) | V5d (+ `aff4` routed) | V5d GPU p50 | Delta vs V4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16.2 ms | 13.65 ms | 14.00 ms | 14.05 ms | 13.68 ms | -2.15 ms |
| 4 | 27.9 ms | 24.88 ms | 23.93 ms | 24.06 ms | 23.70 ms | -3.84 ms |
| 8 | 38.7 ms | 33.60 ms | 32.71 ms | 32.48 ms | 32.13 ms | -6.22 ms |
| 16 | 61.4 ms | 52.60 ms | 51.36 ms | 50.30 ms | 49.92 ms | -11.10 ms |

In-process `dflash` (`512` tokens, `q4` drafter, `tree` mode, `equal = true` on all six prompts; V5a4 with `raw_tiles = true`, `draft_wall_p50 = 4.11 ms` vs `4.90 ms` before):

| Prompt | Committed/round | Draft ms | Verify ms | Spec tok/s | Plain tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `code-1024` | 10.02 | 1.96 | 40.63 | 235.0 | 70.2 |
| `prose-1024` | 4.48 | 3.89 | 34.06 | 118.0 | 66.9 |
| `structured-1024` | 3.79 | 3.85 | 31.06 | 108.4 | 62.9 |
| `code-4096` | 7.52 | 2.99 | 43.73 | 160.7 | 62.0 |
| `prose-4096` | 3.45 | 4.20 | 32.77 | 93.3 | 62.1 |
| `structured-4096` | 3.45 | 4.20 | 32.85 | 93.1 | 62.3 |

## Final qualification (`main` `89049227e`, pinned ReleaseFast `tensorfold-native`)

- **Integrated suite (`campaign/suite/suite.py run --names inkling-host,inkling-gpu,inkling-omlx-oracle`):**
  - `inkling-host`: PASS (`16.8 s`).
  - `inkling-gpu`: **`195/195` tests passed** (`17/17` steps: `68 + 69 + 34 + 11 + 13`, including all 13 served tests and the new standalone synthetic `two-stage gpu argmax` and `few-row routed affine4 experts` tests; peak RSS `20 GB` within the `28 GB` cap).
  - `inkling-omlx-oracle`: **PASS** (`2,856/3,015 = 94.73%` top-1, `95.65%` confident top-1, mean KL `0.0199132`, row-weighted KL `0.0670132`, `0` non-finite — identical to V4).
- **Served qualification (`tensorfold-native`, `-Dcpu=apple_m1 -Doptimize=ReleaseFast`, half prefill default):**
  - **Equality:** `100%` byte-equal across `ar`, `disabled` (loaded drafter + `draft: false`), and `spec` on all 12 prompts (6 `qualify` + 6 `dev`, cold/cached/forked/EOS).
  - **Quiet-host `served-a-qualify` (`18/18` valid decode trials, `15/15` gated prefill curve cells):**
    - Unseen prefill (`ar` / `disabled` / `spec`): `1K`: **`2,374 / 2,602 / 2,687`**, `4K`: **`2,280 / 2,446 / 2,316`**, `8K`: **`2,280 / 2,387 / 2,362`**, `16K`: **`2,157 / 2,241 / 2,241`**, `31.7K`: **`2,081 / 2,150 / 2,162 tok/s`** — Decision 17's `>= 2,000 tok/s` prefill target is cleared at every length including near the `32K` cap with the drafter loaded.
    - Served decode (`qualify`): plain `ar`/`disabled` `71.3–75.1 tok/s` (`disabled` 1K `74.6–74.7`, 4K `71.3–72.6`); speculative `spec` `129.5 / 175.4 / 127.5` (1K, `1.72–2.37x`) and `111.0 / 109.4 / 112.7` (4K, `1.53–1.77x`).
  - **`served-a-dev`:** `100%` `ar == disabled == spec` equality verified on all 6 `dev` prompts (`disabled` quiet start `71.6–72.4` at 1K, `68.2–69.6` at 4K; `code-1024 spec` `179.8 tok/s`, `7.53` committed/round; subsequent trials contended by active desktop load with 2 compressor-invalidated trials).

## Honest Decision 17 decode ceiling

Under Decision 18's invariant (one deterministic arithmetic serving `w = 1` and `w = 2..16` with `spec == plain` and `solo == concurrent` bit-exact, and `w = 1` no slower), `code-1024` clears `200 tok/s` (`235 tok/s`). Even Splash 1.7.2 (using non-deterministic batch-dependent `--verify-rows auto` and `26–32 ms` cycles) clears `200 tok/s` only on `code-1024` (`232 tok/s`) and `code-4096` (`255 tok/s`), remaining at `187 / 156 / 128 / 126 tok/s` on `prose` and `structured` because those prompts commit only `3.5–5.9` tokens/cycle. On this checkpoint and M5 Max DRAM bandwidth (`521 GB/s`), the remaining gap on `code-4096` (`161 tok/s`) is bounded by NVFP4 routed-expert bandwidth (`42` unique experts/layer x `44` layers = `8.28 GB`, `15.9 ms` DRAM floor) and `f32` `simdgroup_matrix` compute at `w = 16`.
