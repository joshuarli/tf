# P6: the 1K dev tail, and where 16K and the cap spend their time

Item 5f of Decision 17, judged by Decision 11 and Decision 20. Lane `lane/p6-tail`, rebased onto `main` (`7e1771817`).
M5 Max, oQ4e `3eea042751cd`, greedy, context 32,768, `INKLING_LOAD_LIMIT_GB=26`, half prefill (the default since PH2).
Raw scripts are in [`inkling-p6-tail-runs/`](inkling-p6-tail-runs/).

**Verdict: accepted and promoted to `main`.** Passes Decision 11 (`1.095×` over 6 balanced pairs across 2 sessions, 95%
CI `1.048–1.143×`, `1K dev` `2,002 → 2,654 tok/s`, peak memory `+84 MB ≤ 128 MiB`) and Decision 20 (oMLX oracle
`94.73%` top-1 and `0.0199132` mean KL identical to `main`, `lane-vs-main-served` `100.00%` top-1, `lane` vs `main
reuse-off` 100% bit-equal, quality dev 1 case lost with 0 new runaways).

## What changed

- The host's last prefill chunk takes the rest of the prompt once it fits in 1,152 rows (`last_chunk_rows`, 1,024 +
  128): the prefill scratch holds 1,152 rows, and a host with a drafter keeps its 1,024 tap rows as the limit. Under
  prefix reuse the last token no longer runs alone.
- The store keeps a prompt's whole state together with its last row's logits (200,058 f32, charged to the store), so
  a repeated prompt takes its first token from them with no replay. A state-only entry of the same tokens is replaced
  by the whole entry. Today the server refuses sampled requests before any store lookup, so no sampled request reaches
  a stored entry; the stored row keeps a future sampler exact.
- `tf-inkling2-prefix-check` enables Metal 4 timelines (`pub const tf_metal4 = true`), takes chunks past 1,024 rows,
  runs many schedules in one load (`jobs`), and times/profiles a prompt's chunks the way the host runs them.

## Decision 20 & Correctness

- **Tests:** `zig build test-inkling2-loop-host --summary all` (37/37 tests), `test-inkling2-loop-served` (whole-prompt
  hit, stored logits bit-equal to cold prefill, EOS as first token, `max_tokens=0`, sampled refusal, fork inside chunk
  equal to store hit, reuse-off equal to reuse-on), and integrated `inkling-host,inkling-gpu` suite on `main`.
- **oMLX oracle (`3,015` positions, 16 histories):**
  - `lane-served` vs oMLX oracle: `2,856/3,015` (`94.73%`), confident top-1 `95.65%`, mean KL `0.0199132` (`1.000×`
    `main`'s `0.0199132`, bar `≤ 1.10×`, top-1 delta `0.00` pts, bar `≥ -0.5` pts), 0 non-finite.
  - `lane-served` vs `main-served`: `3,015/3,015` (`100.00%`) top-1, mean KL `2.096e-16`, max `|Δlogit|` `0.01103`.
  - `uniform-1152` vs oMLX oracle: `2,856/3,015` (`94.73%`), mean KL `0.0199132`.
- **Schedule spread (`probe-spread.txt`):**
  - At `1024`, `1033`, and `1100` held-out code tokens, `lane` vs `main reuse-off` is **100% bit-equal on every row**
    (`max |Δlogit| = 0.000`).
  - `lane` vs `main reuse-on` matches `main reuse-off` vs `main reuse-on` identically (`100.00%` top-1 at margin ≥ 0.1,
    mean KL `0.00000`, `max |Δlogit| = 0.007` at 1024/1100 and `1.966` at row 1032 of 1033).
- **Quality dev (`inkling-p6-tail-runs/quality/`):**
  - `main`: `7/11` (code `3/3`, reasoning `3/3`, retrieval `1/2`, structured `0/3` unsupported, 1 runaway `retrieval-t01`).
  - `lane`: `6/11` (code `3/3`, reasoning `3/3`, retrieval `0/2`, structured `0/3` unsupported, 1 runaway `retrieval-t01`).
  - **Lost case (1 of 1 allowed, 0 new runaways):** `retrieval-t02` (`4,633` prompt tokens: `main` ran `1024×4, 536, 1`,
    `lane` ran `1024×4, 537` matching `main reuse-off`). Moving the last token from a 1-row exact step into the half
    chunk shifted the 5,038-token reasoning chain into refusing the access code (`[[REDACTED]]` vs `[[VN-6403]]`),
    stopping normally without a runaway.

## Served Prefill Curve (Decision 11)

Six balanced pairs across two sessions (`s1`: `ABBAAB`, `s2`: `BAABBA`, `--no-drafts`, cold prompt cache, half default).
Note: 4 of 6 `main` and 3 of 6 `lane` trials carried a startup `conditions valid=False` compressor-burst marker during
model load while every trial passed its pre- and post-trial CPU/GPU/compiler gates (`trial.sh`).

| Prompt | main tok/s, median (trial medians) | lane tok/s, median (trial medians) | lane / main per request, median (range) | lane at 2,000 |
| --- | ---: | ---: | ---: | --- |
| 1K dev | 2,002 (1,719–2,210) | 2,654 (2,182–2,723) | 1.243 (1.124–1.589) | met, 6 of 6 trials |
| 1K held-out | 2,175 (1,989–2,399) | 2,415 (2,236–2,571) | 1.076 (0.984–1.285) | met, 6 of 6 trials |
| 4K | 2,130 (2,014–2,333) | 2,249 (2,166–2,353) | 1.071 (1.005–1.139) | met, 6 of 6 trials |
| 8K | 1,967 (1,917–2,130) | 2,092 (2,023–2,175) | 1.036 (0.961–1.215) | met, 6 of 6 trials |
| 16K | 1,870 (1,775–1,957) | 1,879 (1,792–1,960) | 1.000 (0.919–1.139) | missed, 0 of 6 trials |
| 31.7K | 1,742 (1,643–1,815) | 1,749 (1,651–1,809) | 1.011 (0.922–1.070) | missed, 0 of 6 trials |

- **Pairs' geometric means over curve requests:** `1.057, 1.099, 1.063, 1.185, 1.093, 1.076` → **`1.095×`** across 6
  pairs (**95% CI `1.048–1.143×`**).
- **Memory:** `main` `18.97 / 18.98 GB`, `lane` `19.06 / 19.06 GB` (`+84 MB`, within `max(2%, 128 MiB)`).
- **Tail schedule micro-timings (`tail.sh`, greedy identical on all schedules):**
  - `1027` tokens: `1024,2,1` `543.5 ms` (`1,890 tok/s`) → `1027` `471.1 ms` (`2,180 tok/s`).
  - `1033` tokens: `1024,8,1` `514.2 ms` (`2,009 tok/s`) → `1033` `405.1 ms` (`2,550 tok/s`).
  - `1019` tokens: `1018,1` `403.7 ms` (`2,524 tok/s`) → `1019` `379.0 ms` (`2,688 tok/s`).

## Metal 4 Long-Context Profile (`profile.sh`)

Full 31-chunk `31,744`-token prompt GPU timings (`chunkcurve.md`, warm medians):
- `base = 0`: `473.3 ms` (`0.462 ms/row`)
- `base = 1,024`: `495.3 ms` (`0.484 ms/row`)
- `base = 15,360`: `569.9 ms` (`0.557 ms/row`)
- `base = 30,720`: `657.2 ms` (`0.642 ms/row`)
- **Linear fit (30 chunks from base 1,024):** `492.7 ms + 5.51 ms` per `1,024` positions of base (`17,818 ms` total for
  `31,744` tokens).

Split Metal 4 timeline per `1,024`-row chunk (`buckets.md`, 192 stamps per base):

| ms a 1,024-row chunk | base 0 | base 7,168 | base 15,360 | base 30,720 |
| --- | ---: | ---: | ---: | ---: |
| attention | 42.9 | 115.4 | 167.5 | 278.3 |
| experts | 221.3 | 302.8 | 302.7 | 306.1 |
| routing | 10.4 | 10.5 | 10.6 | 11.9 |
| dense | 78.4 | 107.0 | 105.0 | 105.8 |
| glue | 36.4 | 45.8 | 44.3 | 45.5 |
| total | 389.5 | 581.4 | 630.1 | 747.7 |

Top kernels driving 16K and 31.7K miss:
1. **`i2p_attn_units`:** `32.14 → 103.67 → 155.88 → 266.80 ms` (35.7% of the 30,720 chunk; grows purely from full
   attention's `R = 32` KV traffic and redundant `B[extent]` loads across 8 full layers).
2. **`i2p_expert_units_up` + `i2p_expert_units_down`:** `173.38 + 91.51 = 264.89 ms` (steady from 7K to 31.7K).
3. **`i2p_down_grouped` (`10.82 ms`) + `i2p_gu_grouped` (`4.59 ms`) = `15.41 ms` of empty dispatches** under
   `s.rounding.experts == true` (`I2P_MATRIX_THRESHOLD = 1`), where every non-empty expert is already handled by
   `units_up`/`units_down`.
4. **`i2ps_gemm_hi` (`66.47 ms`), `i2p_relbias_rows` (`11.51 ms`), `i2pr_mix_norm` (`15.05 ms`).**
