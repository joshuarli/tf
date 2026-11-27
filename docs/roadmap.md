# Roadmap

This is the single plan for this repository: all current and future work, and the handoff for resuming it. Campaigns are sections below, and their items interleave in **Sequence**. "Resume the roadmap" means continue from **Status**.

## Status

**Active 2026-11-27 — M5 Max native inference hillclimb Round 2 complete on `main` (past Round 1 `5a07f8689` / V5 `2faa69cb8`).**
- Resume prompt: "resume the roadmap". Campaign **M5 Max native inference** (slice Decision 17) plus post-V5 general-performance hillclimb Rounds 1 and 2 are complete and pushed on `main`.
- **Post-V5 hillclimb Round 2 summary (oQ4e `3eea042751cd` + DFlash2 `4f9088ea66c2`, `INKLING_LOAD_LIMIT_GB=26`, `tensorfold-native` ReleaseFast, quiet-host gated via `perf-gate-ready` / `perf-gate-ready -1`):**
  - **What landed (`5a07f8689`..`HEAD`):**
    1. Shared-KV M5 NAX split verify attention (`i2v_split_nax`, `PAD=128` 16 KB TG occupancy, `packed_float4` relbias loads, 32-row arbitrary tree ancestor masking in one NAX pass): `attn_core` at `w=16` drops `5.26 → 2.11 ms` (`2.49×` faster).
    2. Lossless BF16 packing of MLX `mlp.router_weight` (`exactBf16`, `-14 MB` RAM, 0-bit exact) + barrier-free 8-SG `i2r_router_bf16_1/2/4/6` (`w <= 6`), Split-K `KS=9` NAX `i2n_router_bf16` + fused `i2n_pick_ks9` (`w >= 7`), exact `i2p_small` alignment for `half=false` (`0e0` diff vs per-token decode), and `k % 256 == 0` tail for `i2r_affine8_vec`.
    3. Hybrid BF16 controllers (`i2r_controls_down_b4/b8_bf16` at `w <= 8`, `KS=6` NAX `i2n_controls_down_bf16` + 64-col TG-cached `i2n_controls_apply_bf16` with `float4` stores, `float4` `i2_controls_norm`), `dense4` routing for `oproj` (`n=2304`).
    4. Half-precision expert pipeline: fused `half` input conversion (`nax.xh`) inside `i2n_moe_sort` (zero extra dispatches) + `half` activation handoff between `i2n_nvfp4/aff4_up` and `down`, vectorized `float4` `i2v_conv` (`1.30 → 0.785 ms`), register-cached `i2p_rms3` (`0.68 → 0.136 ms`), and TG-hoisted `tau` in `i2_qkv_rel`.
    5. Scaled NAX grouped verify width from `16` to **`32` rows** (`group.max_rows = 32`, `rows.nax_max_rows = 32`, 16-row M-tiling across all NAX kernels, tile-emitting `i2n_moe_sort` for parallel 16-row expert tiles, updated `Plan.default_verify_ms` curve): rested `code-1024` (`context=1334`) verify is `w=1: 14.05 ms`, `w=4: 23.23 ms`, `w=8: 28.94 ms`, `w=12: 32.82 ms`, **`w=16: 35.19 ms`** (down from `38.96 ms`, `-3.77 ms`), `w=24: 40.69 ms`, **`w=32: 45.14 ms`** (`1.41 ms/row`, `44.73 ms` GPU).
  - **Quiet-host served frontier (Round 2 vs Round 1 `5a07f8689` vs oMLX 0.8.4 `mtp_on`, 512 out, cold):**
    - **Concurrent `drafts_on` (`sum_decode` / `agg_wall` tok/s):** `n=2` (`8K` / `32K`): **`266.6 / 187.5`** (`263.1 / 192.4` @ 32K, vs Round 1 `229.3 / 158.6`, **`3.64×`** oMLX 0.8.4 `73.3`); `n=4` (`8K`): **`437.6 / 209.6`** (`69.5–209.1` per stream, vs Round 1 `284.6 / 175.3`, **`+53.8%`**, **`4.74×`** oMLX 0.8.4 `92.4`); `n=8` (`4K`): **`418.7 / 229.6`** (`40.4–68.8` per stream).
    - **Single-stream speculative decode (served HTTP, tok/s):** `qualify-1K` **`165.4 / 215.6 / 135.8`** (vs Round 1 `135.5 / 203.1 / 130.1`, **`2.03–2.60×`** oMLX 0.8.4), `qualify-4K` **`124.8 / 113.8 / 106.5`**, `dev-code-1K/4K` **`224.5 / 171.7`** (vs Round 1 `200.2 / 155.9`, **`2.84× / 2.36×`** oMLX 0.8.4).
  - **Quality & verification:** official `omlx-oracle-gate` (`prefill 512`, 3,015 pos) **PASS** (`2,868/3,015 = 95.12%` top-1, mean KL `0.0194447` < exact `f32` `0.0195085`, `0` non-finite); `decode-logits 16` vs oracle **PASS** (`2,858/3,015 = 94.79%`, mean KL `0.019587`, row-weighted `0.0628513`, `0` non-finite); `decode-logits 32` vs oracle **PASS** (`2,862/3,015 = 94.93%`, mean KL `0.0199353`, `0` non-finite); `test-inkling2-loop-host` `24/24`, `test-inkling2-loop-affine` `34/34` (`0e0` diff in exact mode), `test-inkling2-loop-dflash2` `11/11`, `test-inkling2-loop-served` `13/13`, `zig` coding-standards gate PASS.
  - **Round 2 rejected experiments (do not repeat):** 128-TG 4-SG split-K BF16 router, `i2r_controls_down_b16_bf16` (`+0.83 ms`), `constant float` e2m1 LUT + half-warp + contiguous `g` (`+1.17 ms`), 3-bucket serial indirect expert dispatch (`+3.13 ms`), `S=1` zero-barrier NAX experts (`+4.48 ms`), inline `R=1`/`aff4_few` register bloat (`+1.14 ms`), pure NAX without `cnt<=2` `few_lut` (`+1.10–1.59 ms`), inline FP8 scale replacing TG `scales[256]` in `few_lut` (`+0.42 ms`), 8-head tiled `i2_qkv_rel` with dynamic `pv[16]` spill (`+0.45 ms`).
- **Post-V5 hillclimb Round 1 summary (oQ4e `3eea042751cd` + DFlash2 `4f9088ea66c2`, `32,768` context, `INKLING_LOAD_LIMIT_GB=26`, `tensorfold-native` ReleaseFast, all 6 served blocks gated quiet by `perf-gate-ready` / `perf-gate-ready -1`):**
  - **What landed (`393a66896`..`3fbf612be`):**
    1. Lossless BF16 packing of all 24 `loop_down`/`loop_gain_forget` controllers (`452 → 226 MB/step`, `-113 MB` RAM, 0-bit exact) + `raw_tiles = true` in served DFlash2 (`393a66896`).
    2. Multi-stream concurrent DFlash2 speculation (`Plan.buildBatch`, `Draft.roundBatch`/`proposeBatch`, `speculateBatch`) verifying up to `16` shared rows across `2–8` concurrent streams (`c2bfab01b`).
    3. Coalesced `half4` K/V loads and 32-key block-max online softmax in split decode/verify attention (`0a145113e`, aligned with `i2p_attn` / `i2p_attn_tiled` in `990a9a1a9` / `3fbf612be` for `0e0` diff vs per-token decode and `4.77e-7` ref error).
    4. Hybrid M5 NAX verify kernels (`verify_nax.metal`, Split-K K-permuted zero-barrier `matmul2d(16,32,16)` at `w >= 3` with GPU counting sort + indirect dispatch, gated on `rounding.experts`, exact `f32` fallback with `INKLING_PREFILL_HALF=off`) + fast `w=1..2` GEMV/NVFP4 kernels (`a5e1161e6`): `code-1024` rested verify drops `w=1: 14.05 → 13.70 ms` (`gpu 13.50 ms`), `w=8: 32.48 → 29.64 ms`, `w=12: 43.12 → 35.32 ms`, `w=16: 50.30 → 39.34 ms` (`-10.96 ms`, `+27.9%`).
    5. 4-bit M5 NAX `reg_layout` DFlash2 drafter head (`affine4.prepareFromAffine8` + `affine4_lane.pack`, `544 → 288 MB`, zero target quality impact, `323c6b011`): `draft_ms` drops `3.85 → 2.97 ms` (`-22.9%`) with identical 28 rounds and `4.536` committed/round.
    6. Double-buffered `i2p_expert_units` (`1` barrier/`BK=64`), 16-bit `half4`/`half` e2m1/e4m3 decode, `fast::exp2` + rescale fast-skip in `i2p_attn_units`, batched `i2pr_mix_norm`, `32×64` BF16 `i2af_gate_ctl_bf16`, `2`-TG `i2p_relbias_rows`, and `float4` `i2p_combine` (`e05898eaa`): 1,024-tok prefill chunk at `8K` drops `387.87 → 368.31 ms` (`-19.55 ms/chunk`).
  - **Quiet-host served frontier vs oMLX 0.8.4 (same 8 prompts, 512 out, cold unseen `prefill_curve.py`, concurrent `n=2,4` fresh 512-tok sessions):**
    - **Concurrent decode (`sum_decode` / `agg_wall` tok/s):** `n=2`: TF `drafts_off` `104.9 / 92.2`, TF `drafts_on` **`229.3 / 158.6`** (`103.5, 125.9` per stream) vs oMLX 0.8.4 `mtp_off` `57.4 / 53.8`, `mtp_on` `73.3 / 67.3` (**`3.13×`** `sum_decode`, **`2.36×`** wall); `n=4`: TF `drafts_off` `141.9 / 120.7`, TF `drafts_on` **`284.6 / 175.3`** (`57.3–83.8` per stream) vs oMLX 0.8.4 `mtp_off` `72.4 / 60.9`, `mtp_on` `92.4 / 77.0` (**`3.08×`** `sum_decode`, **`2.28×`** wall).
    - **Single-stream speculative decode (served HTTP, tok/s):** `qualify-1K` `135.5 / 203.1 / 130.1` (vs oMLX 0.8.4 `mtp_on` `64.0 / 106.0 / 52.2`, **`1.92–2.49×`**), `qualify-4K` `115.9 / 117.0 / 112.7` (vs oMLX `56.9 / 61.0 / 57.3`, **`1.92–2.04×`**), `dev-code-1K/4K` **`200.2 / 155.9`** (vs oMLX `79.1 / 72.8`, **`2.14–2.53×`**). Both `qualify-prose-1024` (`203.1`) and `dev-code-1024` (`200.2`) clear `200 tok/s` served over HTTP.
    - **Unseen prefill curve (`1K / 4K / 8K / 16K / 31.7K`, tok/s):** TF `drafts_off` **`2,833 / 2,411 / 2,348 / 2,239 / 2,148`**, TF `drafts_on` **`2,756 / 2,364 / 2,323 / 2,225 / 2,153`** vs oMLX 0.8.4 `mtp_off` `1,660 / 1,762 / 1,598 / 1,157 / 1,081`, `mtp_on` `1,480 / 1,648 / 1,550 / 1,539 / 1,545` (**`1.39–1.99×`**).
  - **Quality & verification:** official `omlx-oracle-gate` (`prefill 512`, 3,015 pos) **PASS** (`2,857/3,015 = 94.76%` top-1, mean KL `0.019935`, row-weighted KL `0.062223` < V5 `0.067013` and < exact `f32` `0.064197`, `0` non-finite); `decode-logits 16` vs oracle **PASS** (`2,854/3,015 = 94.66%`, mean KL `0.019347` < exact `f32` `0.019508`, `0` non-finite); `campaign/quality` `dev` on speculative server `6/11` (`code 2/3, reasoning 3/3, retrieval 1/2`, matching V5 `main`); `test-inkling2-loop-host` `38/38`, `test-inkling2-loop-affine` `34/34` (`0e0` diff in exact mode), `test-inkling2-loop-dflash2` `11/11`, `test-inkling2-loop-served` `13/13`, `hybrid M5 NAX` `29/29` argmax match, `tiled attention` `0e0` across 42 cases, `affine4` `5/5`, coding standards gate (`lifecycle-concurrency`, `zig`) passed and retired.
  - **Rejected experiments (do not repeat):** register-only `fp4x4` replacing TG `lut[16]` in `f32` NVFP4 (`+0.5–2.56 ms`), `BM=64, BN=128` in `i2p_expert_units` (`+6.5 ms`), shared single `kv_tile` in `i2p_attn_units` (`+2.9 ms`), `i2p_attn_units64` `R=64, 384` threads (`+6.0 ms`), `i2ps_gemm_hi` `128×128` (`+6.6 ms`), `i2af_gemm_bf16` `64×64` (`+9.8 ms`), zero-TG K-permuted `i2p_expert_units` (`+16 ms`).
- **Decision 17 baseline (V5 `2faa69cb8`, oQ4e, `32,768` context, `INKLING_LOAD_LIMIT_GB=26`, half prefill default):**
  - **Prefill (`>= 2,000 tok/s`, ACHIEVED across `1K–31.7K` with drafter loaded):** quiet served `ar` `2,374 / 2,280 / 2,280 / 2,157 / 2,081 tok/s`, `disabled` `2,602 / 2,446 / 2,387 / 2,241 / 2,150 tok/s`, `spec` `2,687 / 2,316 / 2,362 / 2,241 / 2,162 tok/s` (`1K, 4K, 8K, 16K, 31.7K`).
  - **Decode (`>= 200 tok/s` single stream, PARTIAL — cleared on `code-1024`, hardware/acceptance-bounded elsewhere under Decision 18):** plain `w=1` reaches `14.05 ms` (`71.2–75.1 tok/s` served, `1.8 ms` faster than Splash 1.7.2's `15.84 ms`); rested grouped verify drops to `14.05 / 24.06 / 32.48 / 50.30 ms` (`w=1, 4, 8, 16`, 0-bit exact); DFlash2 `q4` draft drops to `4.11 ms` wall p50 (`raw_tiles = true`); in-process `code-1024` clears **`235.0 tok/s`** (`10.02` committed/round), while `code-4096` (`160.7`), `prose-1024` (`118.0`), `structured-1024` (`108.4`), `prose-4096` (`93.3`), and `structured-4096` (`93.1`) stay below `200 tok/s` under Decision 18's bit-exact single-arithmetic rule (even Splash 1.7.2 clears `200 tok/s` only on code with non-deterministic `--verify-rows auto`).
  - **Qualification (`main`):** `inkling-host` PASS (`16.8 s`), `inkling-gpu` **`195/195`** PASS (`20 GB` max RSS), `inkling-omlx-oracle` PASS (`2,856/3,015 = 94.73%` top-1, mean KL `0.0199132`, row-weighted `0.0670132`, `0` non-finite), served `ar == disabled == spec` 100% byte-equal across all 12 `qualify` + `dev` prompts.
- First command: `cd ~/repos/tf && python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu,inkling-omlx-oracle`.
- Acceptance flow (2026-11-24): lanes prove changes with filtered GPU tests; the parent runs the full suite once on an integration commit of main plus the lane and fast-forwards main on green, instead of a lane suite and a second main suite.
- **Last accepted code: SV2, fast-forwarded from `lane/sv2-serve`** (speculation served by default; integrated suite on `33b38ed43`: `inkling-host` ok, `inkling-gpu` 185/185 plus the restored served test 12/12, oracle pass). Before it, V4 `ec925cb41` (fused fixed-cost verify kernels; integrated main green: `inkling-host` ok, `inkling-gpu` 186/186, oracle pass at 94.73% top-1 and mean KL 0.0199). Before it, PH2 merge `291240211` (half prefill by default; `inkling-gpu` 181/181), P5c merge `7f7cb75a3` (181/181), SC1 merge `1668d5caa` (180/180), V3 merge `0a2551b36` (179/179), DF2b merge `53a3b8894`, P4 merge `3f0054737` and V2 merge `3881b3d8c`. The intermittent 28 GB cap breach is fixed: `zig build -j1` limits async jobs but not run-step child waits, so two GPU test binaries could each hold a checkpoint at once (sampled at 28.4 GB). Each GPU test run in `zig/build/inkling2_loop.zig` now claims 26 GB (`gpuRun`), and the suite passes `--maxrss 28000000000`; the sampled fixed run never overlaps two.
- **Open from P3c:** `code-c02` now falls into a repeatable reasoning loop, and oQ4e quality dev is 7/11 (base 6/11), below the fast profile's 8-case bar. Mean TTFT at four agents rose on the warm repetition; the cause is not traced.
- **Parked branches** (all on `origin`), with item, cause and retry:
  - `lane/p3c-prefill` at `f0ff849e6`: commits the prefill scratch at setup. Two 1K pairs are inconclusive and it has not run the suite. Retry: the suite, then three first-pass pairs at 1K.
  - `lane/m1-mtp`: greedy-exact MTP serving, suite green at `dc7a4dba3`, but negligible gain: proposals take 10–14 ms and one-row verification 20–23 ms. `2cd5f9561` is wip. D2 has landed, so its retry condition is met: re-time few-row verification, then run corrected sweeps for MTP on, off and paused. The earlier drafter, `lane/ds1-dspark-drafter` (16.7 / 29.7 ms per round), is superseded by it.
  - `lane/q2-reasoning-replay`: the reasoning-replay quality harness. It never ran because of the GPU queue. Retry: about 2 h on a quiet GPU.
  - `lane/d3-harbour`: a wip probe tool. No bug was found; the receipt is on main.
  - Rejected and kept for reference: `b1f-interleave` (retry only with staggered arrivals), `s1-skip-logits` (token-buffer race), `e2-prefill-experts2`, `e3-expert-tensorops`, `m1-bf16-prefill` and `p6-prefill-gemm2`. Their receipts are in `campaign/kernels/`.
  - Old wip records: `c0-ground`, `k2-inkling-experts`.
- **B2, first pass done** (`campaign/kernels/inkling-b2-baselines.md`): TensorFold leads oMLX at one and two agents, trails it at four (0.84× and 0.90×) and matches at eight. Its warm aggregate is flat from two to eight agents, and its four-agent TTFT is 998 ms against 623 ms for oMLX.

## Charter

- Execute until **Acceptance** is met or every remaining item is parked behind a hard stop. Never stop after writing a plan. Before ending a session, rewrite **Status**, including any lane paths and the first command to resume. Resume by reading **Status**, checking that `campaign/host.json` still matches the machine, and taking the next item.
- Decide and keep going. Record a decision in **Decisions** only when a later slice depends on it.
- The execution grant: everything listed under **Execution boundary**. Park any item that needs more than that.
- Hard stops, each of which parks the item:
  - an action that is irreversible or acts outside this repo, `~/models` and `.work/`;
  - an effect that **Execution boundary** forbids;
  - a loop that cannot check anything.
  
  Push only as Decision 2 allows.
- Failing items: find the cause before retrying (`root-cause`). After three attempts at the same item, or three failed variants of one optimization idea, park the item. Park it on its branch with the cause and a retry condition, then go to the next ready item.
- Fix egregious problems in their own commit when you meet them. Egregious means any of:
  - a crash or hang over 60 s;
  - resident memory growing more than 5% across 10 identical requests;
  - a warm no-op `zig build native` over 60 s;
  - a campaign script over 2× its recorded time with unchanged inputs.
  
  Record anything smaller in the ledger and leave it.
- Per-model dependencies, not phase barriers (owner steering, 2026-11-21). Each model advances on its own chain (see **Sequence**). Profiling is diagnostic and may run while parity investigations continue. Promoting changed arithmetic waits for that model's qualified correctness contract. Speculation is a separate branch after the model's target path is qualified, and comparators run alongside engineering under the GPU lock; neither blocks autoregressive optimization. Broad comparator coverage is needed only before a competitive claim.
- **Lanes:** parallel for the disjoint sets named in **Sequence**, serial otherwise. Each lane works in `.work/wt/<lane>` on branch `lane/<id>`, following `orchestrate`'s worktree lifecycle; never system temp. Set `TMPDIR=$PWD/.work/tmp`. The parent integrates into `main` and runs the parent steering checkpoint (`orchestrate` skill, `references/campaigns.md`, section "Parent steering checkpoint") after each accepted slice.
- **GPU lock:** run every timed run, every process that loads a full model, and the Metal device suite through `campaign/heavy.py`, which takes the lock (Decision 8). No compiles inside a timed window. Stop only processes the campaign started.
- **Commits:** Conventional Commits, scoped to the workstream (`campaign`, `flint`, `inkling2-loop`, `metal`). Put the item ID in the body. End every commit with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Code comments never cite this roadmap.
- **TensorFold rules:** follow `docs/recipes/SKILL.md`, `docs/kernel-contract.md` (for GPU work) and `docs/recipes/adding-a-zig-family.md` (for I-items). Skip their ATC, PR and publication steps; see **Execution boundary**.
- **GPU timing:** run one GPU-timing lane at a time: the shared lock and outside load (OrbStack at 4–10 cores) left parallel lanes unmeasurable. Code-only lanes may run beside a timing lane, but only reading and editing source: their compiles run outside the GPU lock and spoil timed runs ("compiler observed"), and their device tests take the lock between timed runs.

## Sequence

Current items, campaign **M5 Max native inference**, slice Decision 17. Item specifications for earlier phases are under that campaign's **Items**.

1. S1, done (`campaign/kernels/inkling-splash.md`). On oQ4e, Splash prefills unseen prompts at 2,200–3,250 tok/s
   from 1K to the cap; tf `93de31eca` manages 1,180–1,483. Splash cycles take 28–36 ms and commit 3.2–9.1
   tokens; it reaches 200 tok/s only on code. Its 8-row verify costs 26 ms against tf's 56–66 ms. Its
   controlled profile (bf16 KV) breaches the 28 GB cap at 22K tokens.
2. DF1, done and merged (`inkling-dflash2.md`). Greedy-exact. Rounds take 61–77 ms (93% of it the 8-row verify)
   and commit 5.81/4.52/4.33 at 1K and 2.7–3.4 at 4K: 95 tok/s at best, below plain decode at 4K.
3. V1, done and merged (`inkling-v1-verify.md`). Bit-exact grouped rows. The 8-row verify takes 45.6 ms
   rested at 1K (was 56.2) and 52.7 sustained. Speculative decode is up 23–29% on every prompt
   (code-1024 116.6 tok/s, 1.90× plain), and a round waits once. No width meets a Decision 17 budget: the 8-row
   verify is 2.7× its 17.0 ms floor under per-row summation order, and the one-row step is 2.1× its 7.65 ms floor.
3c. V2, done and merged `3881b3d8c` (`inkling-v2-arith.md`, Decision 18). On this M5 Max a float matrix-unit
   product is one fma chain in index order, so 8x8 tiles reproduce every one-row kernel bit for bit (a device
   probe falls back otherwise). The 8-row verify drops from 45.7 to 34.3 ms rested at 1K (40.2 sustained); 4 rows
   25.7, 2 rows 19.9, one-row step 15.7 (no slower). Speculative tok/s is 25–30% over V1 (code-1024 149.5,
   2.41× plain). Fidelity bit-equal; oQ4e quality dev 11 of 11 replies equal to main (7/11). Still no width meets a
   Decision 17 budget at today's acceptance: width 8 needs 7.7 committed tokens a round, the best prompt gives
   5.84. Next lever there: expert tiles for 4–8 live rows, which cost about twice their bytes.
   `inkling-quality-dev` now runs oQ4e with the binary's own grammar library (`50b4a681a`): the native
   checkpoint needs 26.32 GB at 32,768 context with a 2 GiB prompt cache, over the 26 GB load limit.
3a. DF2a, done and merged (`inkling-df2-accept.md`). The 4K drop is the drafter's own 2,047-row window, not a tf
   bug: 4K replies quote text from before the window, and widening the window does not help. Lookup rounds,
   which draft from an earlier match that lies before the window, raise 4K committed tokens per round to
   3.12/2.95/3.57 (from 2.76/2.69/3.43) and leave 1K unchanged. Served lookup still needs the host to pass the
   request's tokens in. First-rows retention (3.43/3.01/4.15) is parked: it departs from the checkpoint's window,
   only works as a full rebuild every round, and every test prompt puts its instruction first.
3b. DF2b, done and merged `53a3b8894` (`inkling-df2b-tree.md`). Best-first trees of up to 16 rows from the
   drafter plus lookup chains up to 15 deep, ancestor-masked verify with a per-node recurrent path, and lookup in the
   served round. Every reply equals plain greedy, and tree rows equal their paths' one-row steps bit for bit.
   Speculative decode is 1.193× main `50b4a681a` (six process pairs, 95% interval 1.140–1.249): code-1024
   132.3 → 150.8 tok/s (7.51 committed a round), prose/structured-1024 1.11×/1.09×, the 4K prompts 1.24–1.35×
   (78–93 tok/s). Verify costs 40.4 ms at 8 rows and 71.9 at 16 in rounds. Rounds that run the drafter peak at
   0.117 tokens/ms, so 200 tok/s needs 8-row rounds close to a one-row step's cost. The six pairs ran in two
   back-to-back sessions on one host state; accepted for an opt-in mode (delegated, Decision 11), to be re-paired
   across separate sessions before speculation becomes a default.
3d. V3, done and merged `0a2551b36` (`inkling-v3-experts.md`). Verifies of 9–16 rows are 6–12% cheaper (16 rows
   in rounds 69.1 → 60.8 ms; rested 1K 61.9 → 58.4, 4K 72.6 → 67.4); 1–8 rows and plain decode are unchanged, and
   every row stays bit-equal to its one-row step. The fixed cost table is refit from in-round costs (a 9th row
   4.6 ms instead of 10.5). Round pairs: geometric mean 1.022, CI 0.986–1.059, inconclusive; code-1024 1.09× (187
   tok/s median). Kept for cumulative benefit (Decision 11); the second session ran beside an outside soak, so SC2
   re-pairs it on a quiet host. megakernel-zoo's layout under our summation order is 1.23–1.51× slower at 8–16 rows.
   Open: experts with 4–8 live rows still cost about 1.5× their bytes, and about 13 ms a round at every width sits
   in kernels dispatched from `prefill.zig` (mixes, norms, convolutions, router, controller, affine 4-bit experts),
   which V4 takes after P5c.
3f. V4, done and merged `ec925cb41` (`inkling-v4-fixed.md`). The ~13 ms a round V3 attributed to fixed kernels was about
   5 ms measured unstamped on the served queue: depth mixes and norms 2.35 ms in place, controllers 1.01, the k/v
   prologue and store 0.77, the router 0.56 — small dependent dispatches at 25–170 times their bytes, not bytes. The
   lane fuses the same arithmetic in the same order into 424–472 fewer dispatches a step; every row and prefill stays
   bit-equal (228/228 cross-build hashes at both prefill settings), and the verify is 1.5–2.2 ms cheaper at every width.
   Six interleaved round pairs: speculative 1.038× main (CI 1.016–1.062), the one-row step 1.125× (1.108–1.141);
   code-1024 reaches 230 tok/s, no other prompt reaches 200. Decision 17's remaining cost past one row is the routed
   experts (19–33 ms of a 38–64 ms step at 8 and 16 rows), out of this lane's scope.
4. SV1, done and merged (`inkling-sv1-serve.md`). `INKLING_DRAFTER=<dir>` turns on served greedy DFlash2
   speculation, byte-identical to plain greedy across prompt-cache hits, forks and EOS (133/133 in `inkling-gpu`).
   It is faster only on code-1024 (1.47×) and prose-1024 (1.16×), so it stays opt-in until V1 and DF2 land. It
   speculates only while one lane decodes. `bench.py MODE=drafter` still passes Flint's `--drafter`.
5. P4, done and merged `3f0054737` (`inkling-p4-prefill.md`). Default prefill output stays byte-identical to main
   and the oracle gate gives D2's numbers exactly. GPU time per 512 rows is 0.81–0.94× main; served prefill is 1–12%
   faster from 4K to 31.7K (lane 1,454–1,591 at 4K, 1,540–1,625 at 8K, 1,395–1,503 near the cap); 1K is
   inconclusive (0.89× and 1.02×). `f0ff849e6` (P3c scratch at setup) is kept: first request 2.89 → 2.45 s median.
   Opt-in `INKLING_PREFILL_HALF=both` serves 2,048/2,059 tok/s at 4K/8K (1.18–1.36× default) but agrees with the
   exact path's top-1 on 97.49% of rows at margin ≥ 0.1, under the 99.5% bar, so it is not promoted; `inputs`
   97.32%, `experts` 97.25%. Unresolved: `experts` alone measured slower in its one trial.
5a. PH, done (`inkling-ph-half.md`): `INKLING_PREFILL_HALF=both` passes Decision 19. Oracle KL 1.02×, top-1
   −0.23 points; full quality 29/44 on both; Pi 7/8 on both (`read-and-answer` fails on both). It swaps one runaway
   for another, repeatably: exact loops on `code-c02`, which half passes; half loops on `retrieval-t01`, which exact
   answers. Both are tracked under Q3.
5e. PH2, done and merged `291240211` (`inkling-ph2-default.md`). Contract change: `INKLING_PREFILL_HALF` unset
   now serves `both`; `off` is the exact path. Requalified on P5c's schedule under Decision 19: oracle KL 1.021×,
   top-1 −0.23 points; full quality 29/44 against exact's 28/44 (half loses `retrieval-t01`, gains `code-c15` and
   `retrieval-t04`); runaways 2 against 3; Pi 7/8 on both. The dev set alone reads 7/11 against 8/11, so the
   long-reasoning runaways are judged on the full set. Served curve under the default (three interleaved pairs
   against `off`, one session): Decision 17's 2,000 tok/s is met at 1K held-out (median 2,063), 4K (2,130) and 8K
   (2,072); missed at 1K dev (1,839, the short last chunk), 16K (1,905) and 31.7K (1,777). Half is 1.17–1.22× exact
   at every length; first request 0.87–0.89 s. Next for prefill: the 1K dev tail (store the full prompt's state with
   its logits) and a profile of 16K and the cap.
5b. P5c, done and merged `7f7cb75a3` (`inkling-p5c-chunks.md`). Served prompts prefill in 1,024-row chunks that
   capture fork snapshots inside them. Six interleaved pairs: 1.175× (CI 1.109–1.245); 1K held-out 1.56×, 4K 1.12×,
   8K 1.06×, 16K and 31.7K 1.03×; best 1,859 tok/s at 1K, 1,826 at 4K. Decision 20 passes (KL 1.001×, top-1
   −0.13 points, held-out 99.90%, quality dev 8/11 vs 7/11). Contract change: a one-token warm-up through slot 0 at
   load, then zeroed, cuts the first request from 2.4–2.9 s to 1.03–1.06 s; it adds 0.75–1.81 s to load and 1.57 GB
   to ready footprint at one 32K slot (0.63 GB at eight 4K slots); peak stays 18.98 GB. Open: dev 1K prompts still
   end in a 2–8-row chunk plus the lone last token (4–30% short of reuse-off); storing the full prompt's state with
   its logits would remove the one-row chunk.
3e. SC1, done and merged `1668d5caa` (`inkling-sc1-chain.md`). A tree
   round is GPU-bound: 35–41 ms, of which verify GPU 30.6–35.2 and drafter GPU 3.2–4.3; the host idles the GPU
   1.0–1.3 ms (2.7–3.3%). Draft end → verify start is 0.73–0.99 ms (selector ~0.2, the verify's CPU encode in
   `prefill.zig` 0.56–0.72); verify end → next draft is 0.20–0.25 ms, so a chained draft recovers at most 0.5–0.8%.
   The chain (`4716d11f6..4e44a9870`) drew identical trees but no smaller gap and is reverted. Kept:
   `INKLING_ROUND_SPLIT=1`, a bit-identical selector, and a plain-greedy scratch determinism test (rerun it with
   chaining on before reviving the chain). Decision 17's levers are verify GPU time, tokens per round and drafter
   GPU time; hiding the verify's ~0.6 ms encode joins V4.
5f. P6, done and fast-forwarded `a6bc4fa26` (`inkling-p6-tail.md`): the 1K dev tail merged into a last chunk of up to
   1,152 rows, and the store keeps the whole prompt's state with its last-row logits so a hit needs no replay. Six
   balanced pairs across two sessions (plain prefill): `1.095×` geomean (95% CI `1.048–1.143×`), 1K dev `2,002 → 2,654`
   (`1.243×`), 1K held-out `2,415`, 4K `2,249`, 8K `2,092` (all ≥ 2,000 on 6/6 trials), 16K `1,879` (`1.000×`), 31.7K
   `1,749` (`1.011×`), memory `+84 MB`. Decision 20 passes (oracle `94.73%` top-1 and `0.0199132` mean KL identical to
   main; `lane` vs `main reuse-off` 100% bit-equal; quality dev 1 case lost `retrieval-t02`, 0 new runaways); integrated
   `inkling-host` + `inkling-gpu` green. Metal 4 `profile.sh` (`chunkcurve.md`: `492.7 ms + 5.51 ms / 1,024 base`;
   `buckets.md`) identifies `i2p_attn_units` (`32.1 → 266.8 ms` at base `30,720`) and `15.4 ms` of empty
   `i2p_gu_grouped`/`i2p_down_grouped` dispatches as the 16K/31.7K bottlenecks.
5c. SV2, done and merged (`inkling-sv2-served.md`). Speculation is now the served default. On six untouched
   qualify prompts (`campaign/prompts/qualify/`) over three rotated sessions, speculative decode is 1.43–2.43× plain
   (every lower 95% bound above 1; code/prose/structured-1024 148/172/127 tok/s, 4K 101–113). Every reply is
   byte-equal to plain across cold, cached, forked and natural-EOS requests. The drafter adds 0.42–0.56 GB at ready
   and 0.81 GB at peak; unseen prefill with the drafter loaded is 0.96–0.99× at 1K–8K and unchanged at 16K and the
   cap (SC2's two sessions: 0.965–0.968× at 1K–8K). Its `idle` mode measured nothing: `--no-drafts` drops the drafter
   at load, so `idle` is plain decode. Contract change: serve pairs Inkling 2 Mini with `incoai/Inkling-2-Mini-DFlash2` from the local
   Hugging Face cache, and `--drafter` names it, like `INKLING_DRAFTER` (giving both is refused). `--no-drafts` and
   `--drafter none` serve plain. The cost table is refitted to V4's in-round costs: neutral, 0.998× (CI
   0.961–1.036). Decision 17's 200 tok/s still holds only on code-1024; the routed experts past one row remain the
   cost. Sessions a and b ran back to back, so SC2 re-pairs on a quiet host.
5d. SC2, done (`inkling-sc2-scorecard.md`), pinned on main `de53331e3`.
   - Prefill: plain meets 2,000 tok/s at 1K/4K/8K (2,004/2,067/2,011). The default with the drafter is 3% short
     (1,939/1,994/1,947) because every prefill chunk captures and absorbs the drafter's taps. Both miss at 16K (1,885)
     and near the cap (1,766).
   - Decode: 200 tok/s on no served prompt (best 186 on code-1024; 93–172 elsewhere). The verify's routed experts
     dominate the round.
   - Round pairs, re-paired over six pairs in two sessions: DF2b 1.219× (1.203–1.235), V3 1.002× (0.969–1.037,
     inconclusive).
   - Quality: PH2's results stand, since no arithmetic changed since.
   - Correction: SV2's `idle` mode was plain decode, because `--no-drafts` drops the drafter at load.
5h. Q4, done (`campaign/kernels/inkling-q4-qualify.md`, owner steering 2026-11-25, from an independent audit of
   `46061a3aa`): serving qualification now enforces and describes the accepted policies before any further performance
   claim. P6 and P7 are unblocked.
   1. Equality: compare served streams by token and output hashes, token counts, finish reasons and tool output;
      missing evidence fails, with mutation tests (`campaign/tests/test_served_equality.py`). Done in `1f285be08`:
      the old check compared a status field; SC2's records still match under the repaired check.
   2. Completed tasks beside the fixed 512-token window (relabelled fixed-window generation throughput): frozen 36-task
      suite with EOS on and 8,192 tokens of room, scoring success and format, recording cap hits, total time, first
      answer latency, reasoning/answer/total tokens, and follow-up turns from completed answers. Quiet-host `tasks-half`
      (0/72 timing-invalid, 36/36 valid pairs): `ar == spec` on every task and follow-up, 32/36 passed (2 cap hits:
      `code-c13`, `retrieval-t01`), total time 715.2 → 440.0 s (1.625×; low-copy 1.802×, copy-heavy 1.424×,
      tool-continuation 1.588×; first-answer 1.652× / 1.383×). `tasks-off`: `ar == spec`, 31/36 passed (3 cap hits:
      `code-c13`, `code-c15`, `retrieval-t04`; `retrieval-t01` passes), 22/72 timing-invalid from a pressure burst.
   3. Corpus: exclude model outputs, results, receipts and run records from prompt sources. Existing prompts kept as
      regression cases; frozen external tasks labelled low-copy / copy-heavy / tool-continuation with target-tokenizer
      lengths.
   4. Configuration identity on every result: checkpoint, KV dtype, prefill rounding, decode arithmetic, drafter,
      lookup, sampler, context and output policy. Half and `off` prefill reported separately.
   5. Decision 19 as an executable gate (`campaign/quality/decision19.py`, `test_decision19.py`): net-runaway rule,
      every lost case listed (`retrieval-t01`), explicit unsupported/over-context/missing/invalid handling. Passes on
      PH2's records (29/32 vs 28/32, runaways 2 vs 3, KL 1.021×, top-1 −0.23 pts, Pi 7/8 both).
   6. Regenerated scorecard on `main` (`b925b31e…15f6`): three rotated half-prefill served sessions (`a`, `b`, `c`:
      95/108 valid trials; quiet `b+c` 69/72 valid, 24/24 curve gates) plus `off` (36/36 valid, 12/12 curve gates),
      100% equality across `ar`, `disabled` (loaded drafter + `draft: false`) and `spec`. Valid pairs counted from
      records for all 12 prompts: `spec/ar` 1.351–2.842× (1.351–2.579× in `b+c`, best 171.3 tok/s on `code-1024`;
      Decision 17's 200 tok/s unmet); `disabled/ar` neutral (every 95% CI includes 1.0). Paired gated prefill curves
      show every `disabled/ar` and `spec/ar` 95% CI includes 1.0 (`b+c` plain 1,892 / 2,001 / 1,933 / 1,833 / 1,694
      tok/s at 1K–31.7K; `off` 1,538–1,616), so SC2's ~3% drafter prefill cost remains an unconfirmed hypothesis for P7.
5i. A1, done on `lane/inkling-metal-aot` (`campaign/aot/verdict.md`, `phase_a..f.md`, `research.md`, `inventory.md`):
    complete offline AOT compiler (`tf-inkling2-aotc`) and standalone native runner (`tf-inkling2-mini-aot`) for Inkling 2
    Mini oQ4e on M5 Max (`applegpu_g17s`), compiling all 94 Metal 4.1 / MPP TensorOps PSOs into `kernels.metallib` +
    `pipelines.mtlarchive` (`3.6–4.1 ms` archive load, `1.52–1.71 s` verified 5-thread `F_NOCACHE pread` + SHA-256 load
    of `15.57 GB`). Gates:
    - **G0/G2/G4 PASS:** runs under `env -i PATH=/usr/bin:/bin` with zero runtime Metal compiler/Python/checkpoint
      dependencies; fails closed on incompatible device (`g13g`/`g16s`) and 1-byte bundle flips; peak memory `16.7–17.0 GB`
      at V4 / `17.6–18.8 GB` at 32K (2–3.4 GB below `main`, `636.5 MB` dead vision weights stripped).
    - **G1 PASS:** bit-exact 0-diff logits vs `main` (`stepDirect`, `verifyTree` `chain8/chain16/tree16/commit_rollback`,
      `generateBurst` 16-step parity + EOS + host cancel, `--taps`); full 16-history (`3,015`-position) oMLX oracle pass
      (`2,865/3,015 = 95.02%` top-1, mean KL `0.0208875 <= 0.02926`, row-weighted KL `0.0740774 <= 0.09630`, `0` non-finite).
    - **G3 honest FAIL (near-miss under unchanged arithmetic):** 6 balanced pairs over 2 sessions give V4 6-prompt wall
      geomean `1.1202×` (95% CI `1.0521–1.1927×`, 6/6 pairs faster, below `1.15×`), GPU geomean `1.1073×` (`1.0394–1.1796×`);
      no long-context regression (`1.012–1.075×` from `1K–31.7K` with `D9` head-major `half4`-permuted KV, `+7.8–8.1%` at
      `16K–31.7K`). Remaining `~2.1 ms` above isolated kernel sum is serial barrier drain across 782 dependent dispatches.
    - **G5 partial:** chunked prefill hits `2,475–2,825` (1K), `2,287–2,604` (4K), `2,100` (8K), `1,875` (16K), `1,653`
      (31.7K) tok/s (`419–453 ms` first request); rested verify drops to `29.1 ms` (w=8) and `46.8 ms` (w=16) bit-exact.
    - **Verdict:** keep `lane/inkling-metal-aot` as reference and extract its 6 proven wins into `main` (`MTLBinaryArchive`,
      5-thread `F_NOCACHE pread` + vision weight strip, Phase D bit-exact fusions, Phase C2 `bfloat2 sb` + `ulong2/uchar2 w13`
      + `float2(g,f)` repacking for verify, 2-stage GPU argmax, `D9` permuted KV).
5g. P7, done (`campaign/kernels/inkling-p7-taps.md`): replaced the `18.87 MB` `d.prompt` staging buffer and CPU absorb
   loop with direct-to-ring `i2t_tap_ring` GPU capture, selective skipping outside `tapNeedFrom` (also skipping unused
   final `mixBlocks + rmsW` on earlier chunks), wraparound-safe fork-point stashing, and restored P6's `1,152`-row last
   chunk with the drafter loaded. `test-inkling2-loop-host` (`25/25`), `ring tap capture` (`5/5`), and
   `test-inkling2-loop-served` (`13/13`, 10/10 spec==draft-off, 4/4 spec==no-drafter) pass. Served prefill with drafter
   loaded reaches `2,434–2,541` (1K), `2,163` (4K), and `2,063` (8K) tok/s (within `0.5–2.3%` of `--no-drafts`).
5j. P8, done (`campaign/kernels/inkling-p8-longprefill.md`): `BM=128` (8-SG) `i2p_expert_units` + `e4m3_lut`, skip empty
   `wide_gu_nv`/`wide_down_nv` dispatches under `rounding.experts`, `i2p_attn_units` `ulong2` KV loads + hoisted
   `far_bias` + fast `full_cache`/`far_block` path, `128×64 mode::multiply` `i2ps_gemm_hi`, `i2as_split_hi` / hi-only
   `i2ps_split_gated`, and skip doomed `store(capture.at)` when `fork + final > store_limit`. Oracle gate matches `main`
   identically (`2,856/3,015 = 94.73%`, mean KL `0.0199132`); served prefill with drafter clears Decision 17 at every
   length (`1K: 2,521–2,690`, `4K: 2,496`, `8K: 2,384`, `16K: 2,223`, `31.7K: 2,057 tok/s`; `2,307 tok/s` GPU).
6. Q3, done (`campaign/kernels/inkling-q3-runaways.md`): full untruncated traces under `half` and `off` show no
   TensorFold numerical defect. `retrieval-t01` (`half`) finds `QX-7419` in ~100 tokens then writes 26,125 chars of
   coherent non-repeating code review (max 12-gram = 1) until the 8,192 cap (`off` chooses a compact review, 4,066
   tokens); `code-c02` passes under default `half` (1,561 tokens) and loops on `'o'` under `off`; `code-c13` matches
   for 1,123 chars then enters greedy `<think>` repetition loops under both `half` (44×) and `off` (78×).
5k. V5, done and qualified on `main` (`campaign/kernels/inkling-v5-decode.md`, `4b8b840c1..56274f2e2`): bit-manipulation
   E4M3 decode, fused `i2_batch_combine_sig_mul` (`-48` dispatches/step), 2-stage parallel GPU argmax (`i2v_argmax_p1/p2`),
   bit-exact `simdgroup_matrix` routed affine4 experts (`i2e_aff4_up/down[16]`), and DFlash2 drafter `raw_tiles = true`
   (cutting `q4` draft wall p50 `4.90 → 4.11 ms`). Rested `code-1024` verify drops to `14.05 / 24.06 / 32.48 / 50.30 ms`
   (`w = 1, 4, 8, 16`; `max_abs_diff = 0e0`, `-2.15` to `-11.10 ms` vs V4); plain `w=1` (`71.2–75.1 tok/s` served) beats
   Splash's `15.84 ms`; in-process `code-1024` hits `235.0 tok/s`. Full integrated suite green (`inkling-host` ok,
   `inkling-gpu` `195/195`, `inkling-omlx-oracle` pass `94.73%` / `0.0199132`); quiet served `qualify` clears `>= 2,000 tok/s`
   prefill at every length `1K–31.7K` (`2,687–2,162 tok/s` with drafter) with 100% `ar == disabled == spec` equality across
   all 12 `qualify` + `dev` prompts.
7. Deferred off the critical path: see **Later**.

Items run on per-model chains, and items whose owned files are disjoint run in parallel:

- **Flint:** trustworthy measurement (C2, C3) and F2's divergence trace, then a profile at 1K and 4K, then one bounded O-item screened and, if promising, qualified on the promotion matrix and held-out prompts. Any near-cap fixture must reserve reply headroom. F5 speculation branches off after the target path is qualified.
- **Inkling 2 Mini:** I2's executable native forward path, then I3 minimal serving, then the frozen first-native baseline, then profiling and O-items. I5 branches off after I3.
- **Comparators:** one runnable strong comparator is the early anchor (F3). The full matrix is needed only before a competitive claim.

The phase headings below group items; they are not barriers. Every item ends on its acceptance command plus the narrowest affected suite, run on the integrated `main`. The full `zig build test test-golden` runs at milestones M1 (after F3), M2 (after I3) and the close.

**Disjoint ownership sets:**
- `campaign/` (C, Q and H items, plus the F2 and I2 harness files): one writer at a time.
- Flint: `zig/src/families/flint/`, `zig/src/native/flint.zig`, `zig/kernels/metal/flint/`, `zig/tests/flint/`, `zig/tests/contract/metal_flint*`.
- Inkling 2 Mini: `zig/src/families/inkling2_loop/` (including its kernels and tests), `zig/src/native/inkling2_loop.zig`, `zig/src/native/inkling2_loop_host.zig`, `zig/build/inkling2_loop.zig`, `campaign/parity/inkling2-mini/`.
- Shared: `build.zig`, `zig/build/metal_*.zig`, the native registry, `zig/src/core/`, `zig/src/server/` and `zig/kernels/metal/core/`. One registration edit is folded into the item that needs it, and the parent merges these. Any other shared-core change is its own serial item. While it is open, it blocks the other workstream from touching core.

GPU-timed and full-model steps serialize on the GPU lock even when their code lanes run in parallel.

## Campaign: M5 Max native inference

### Goal

State: active. Native Flint, then Inkling 2 Mini, inference on this M5 Max (40-core GPU, 48 GiB) inside the owner's 32 GB budget, measured against the strongest reproducible engines. The current slice is Decision 17: 2,000 tok/s unseen-prompt prefill across the curve and 200 tok/s single-stream decode. Tooling and records live in [campaign/](../campaign/README.md).

### Decisions

The owner made the first two decisions on 2026-11-21:

1. **Owner: layout.** One repo at `~/repos/tf` (`~/repos/tensorfold-m5` is a symlink to it), a TensorFold clone with `upstream` as a fetch-only remote (push URL `DISABLED`). Campaign tooling and records live in `campaign/`.
2. **Owner: push to `origin` (`joshuarli/tf`) only.** The earlier no-push rule is lifted. Never push to `upstream` (ashhart/TensorFold); its push URL is disabled.
3. **Owner: build inputs through `laputa-buildlib`.** Every downloaded input is pinned in `campaign/sources.lock.json` (schema 1, see the [build repository contract](https://github.com/laputa-systems/stuff/blob/main/docs/build-repos.md)) and fetched through `buildlib.locks.fetch_input`. Downloaded inputs include weights, comparator sources and the xgrammar sources. Pin buildlib in `campaign/pyproject.toml` as follows and commit `campaign/uv.lock`:
   - version `laputa-buildlib==0.1.0`;
   - index `https://pypi.17166969.xyz/current/simple/`;
   - wheel sha256 `edd102a1713c903f93e19b10036983d123dd76fa1b04a0e5f3c5070f00a95383`.
   
   `campaign/bootstrap.lock.json` pins the Zig 0.17.1 macOS aarch64 tarball. The executor uses the installed Zig after checking it matches that tarball's version. Python 3.15.1 is installed.
   - **Owner, 2026-11-21: Zig 0.17.1.** The owner upgraded the installed Zig and pinned 0.17.1. `build.zig` requires an exact `.zig-version` match, so the campaign tree sets `.zig-version` and `minimum_zig_version` to 0.17.1. The frozen baseline is `fa13df5d` with only those two lines overridden (`make -C campaign baseline-toolchain` applies the override and refuses any other diff), so the baseline and every candidate share one compiler.
4. **Owner: the harness is tested with Pi.** Coding-agent serving checks drive the local server with `pi` 1.5.0 (H1). Use a campaign-local `PI_CODING_AGENT_DIR=campaign/pi/agent` with `PI_OFFLINE=1`, `PI_TELEMETRY=0`, `--no-extensions --no-mcp --no-skills -nc --no-session --mode json`. Never read or write `~/.pi`.
5. **Owner: Inkling 2 Mini is the Inkling target.** That is `thinkingmachines/Inkling-2-Mini` at `ed40e7da625f594fc9c4990f5f66be24873c22d7`, `model_type: inkling2_loop`. Upstream's `zig/src/families/inkling2/` is Inkling 2 **Small**: CUDA/EXL3 only, about 394 GB, and not served. That makes Mini a new Metal family. Reuse the Small package's config, template and tokenizer code only where its tests show the contracts match.
6. **Prescribed: base and baseline.** TensorFold stays the base: it is Apache-2.0, it serves Flint natively on Metal (#1177, merged), and it has in-tree NVFP4/FP8 readers and Metal kernels (`zig/src/core/nvfp4.zig`, `zig/src/core/quant/fp8.zig`, `zig/kernels/metal/drama3/nvfp4.metal`). The frozen Flint baseline is a build of `fa13df5d`. The Inkling baseline is the first qualified port (I3), labelled "first native port", never "upstream". Reverse only if a licensed, buildable alternative is shown to save a phase.
7. **Prescribed: weights.** Weights go in `~/models/<repo-name>@<rev12>/`, pinned and verified by sha256:
   - Flint: `RecursiveSI/flint-int4@3b93f9053b530488405ae9fca51d4900d6ce21f0`, 22.6 GB.
   - Flint drafter: `incoai/Flint-DFlash2@39b0a7f7a12a14461b9eac8d90a9fcec24168889`, 0.8 GB.
   - Inkling 2 Mini: 20.2 GB, mixed NVFP4/MXFP8. It includes `mtp.safetensors`, its native draft head.
   - Inkling 2 Mini oQ4e, the T2 target: `Inkling-2-Mini-oQ4e-mtp@3eea042751cd`, 16.8 GB. Its experts are NVFP4 in layers 0–19 and affine 4-bit in layers 20–23. Attention, embeddings and head are affine 8-bit.
   - Comparator weights, fetched only when their item runs.

   The fetch total stays under 120 GB. Never delete files you did not create. Only one full model is resident at a time.
8. **Prescribed: memory.** The machine has 48 GiB of RAM. C0 records the Metal device's `recommendedMaxWorkingSetSize` (W).
   - Owner cap (2026-11-21): the campaign as a whole stays under 32 GB so the host stays usable for other work. Every command that loads a model or uses more than about 2 GB runs through `campaign/heavy.py`, which takes the GPU lock and stops the process tree past 28 GB; bare `lockf` is not used.
   - Every benchmark server sets its load limit (`FLINT_LOAD_LIMIT_GB=27` from `heavy.FLINT_LOAD_LIMIT_GB`, or
     `INKLING_LOAD_LIMIT_GB=26`), an explicit `--prompt-cache-gib`, and `--context` of at most 32768. The old memory-derived Flint limit was 56,192
     tokens; the owner's total-context cap now takes precedence. Runs recorded before the memory cap used
     36 and pair only with runs at the same memory and context limits.
   - Prompt cache is 0 for cold-prefill cells and 2 otherwise.
   - Speculation may add at most 4 GiB of draft residency, which is min(4 GiB, 10% of RAM).
   - A run that sees memory pressure or thermal throttling (`pmset -g therm`, `memory_pressure`) is recorded as an invalid trial and the workload is reduced. Never change system settings.
9. **Prescribed: references.**
   - **Flint:** the oracle is `rec_mlx` (`recursive-oss/rec` v0.1.0 at `e53663b2`), which is where TensorFold's kernels match rounding. It runs in a `.work/venv-rec` uv environment as a development tool only.
   - **Inkling 2 Mini:** the semantic reference is llama.cpp `src/models/inkling2-loop.cpp` plus `conversion/inkling2.py`, pinned at the commit I0 records (master was `40b7a546defe1f60edd1338c719552ffc58c89a0` on 2026-11-21). The vLLM recipe and the model card's linked code are cross-checks. The numerical oracle is oMLX 0.8.3 on the oQ4e checkpoint (owner, 2026-11-23): `make -C campaign omlx-oracle-gate` holds TensorFold to its saved teacher-forced captures (`campaign/kernels/inkling-omlx-oracle.md`). `trl-internal-testing/tiny-Inkling2LoopForConditionalGeneration@5e761216759d5d040c09f458d5f077b9aaa5ac78` is the tiny unit fixture.
10. **Prescribed: quality rules.** These are fixed now and never relaxed later.
    - Unchanged-arithmetic transformations must be bit-identical to the frozen baseline on logits: compare the frozen baseline's and the candidate's tensor and logit dumps on identical forced histories. `tf-flint-run exact` is only a self-consistency check inside one build (one-row steps against 16-row windows); keep it, but it does not compare builds.
    - **Inkling 2 Mini is exempt from the logit rules below (owner, 2026-11-23; Decision 16).** Its `omlx_gate.py` comparison against the pinned oMLX oracle is reported, not gating. Divergences are traced against oMLX (`campaign/kernels/inkling-long-context.md`). The owner accepted the oQ4e weights and f16 KV.
    - Reordering changes are judged on the F2/I2 teacher-forced corpus:
      - The candidate's mean KL to the oracle must stay within 1.10× the baseline's.
      - Top-1 agreement with the baseline must be at least 99.5% on positions where the baseline's top-2 margin is at least 0.1 logit.
      - Every disagreeing position must be traced to its first divergent tensor before acceptance.
    - Q1 quality suite: no more than one case lost against the baseline, and every lost case is examined. Lossy requantization and KV compression are separate experimental configurations, never promoted as speedups.
    - Deterministic operator, state and protocol fixtures allow zero new failures.
11. **Prescribed: performance gates.** These apply to promotion only; a rejected screen needs no six-pair run.
    - **Promotion:** a geometric-mean gain of at least 3%, with a paired 95% CI above 1.0.
    - **No regressions:** no primary-cell decode or prefill regression above 5%, and peak-memory growth no more than max(2%, 128 MiB).
    - **Memory route:** a change may instead qualify with at least 5% lower peak memory and no latency regression above 2%.
    - **Pairing:** at least 6 balanced baseline/candidate process pairs across two sessions, expanding to 12 only to settle a borderline result; otherwise the result is "inconclusive".
    - **Dispatch:** workload-specific dispatch is allowed, and the final dispatching build is validated on every primary cell. A gain below 3% is kept only with a stated simplification or cumulative benefit.
12. **Prescribed: targets.** These are goals, not gates. Unmet targets are reported, never hidden.

    | Milestone | Target |
    | --- | --- |
    | Flint runtime | ≥20% geometric-mean true-autoregressive decode gain over the frozen `fa13df5d` build |
    | Competitive | ≥10% geometric-mean gain over the strongest reproducible comparator on the same suite |
    | Stretch | ≥30% runtime gain, or a substantial measured memory reduction at equal speed and quality |
    | Inkling | Qualified native text inference, then the same optimization process against I3's baseline |

13. **Prescribed: speculation.** Advertise greedy-only speculation first, labelled as such. A sampled mode needs the Leviathan et al. acceptance/rejection correction, or the model's native method. Its proof is the item's own enumerable-distribution tests and argument (see **Sequence** F5/I5). Every mode keeps its no-drafter path and automatic fallback, and fallback cost is included in timings.
14. **Delegated: optimization order.** For each model's optimization, the executor ranks hypotheses by measured cost share and keeps the ranked table in `campaign/ledger.md`. Candidates (the frontier brief's §4 A–M extends them):
    - A: Exo #2533 fusions;
    - B: M5 TensorOps/MPP matrix paths and quantized layouts;
    - C: attention and state;
    - D: larger fusion or persistent execution, only where dispatch cost is significant;
    - E: model-specific speculation.
    
    Bound: one experiment per item, smallest decisive change first. Tie-break: lower correctness risk.
15. **Prescribed: Exo's M5 exclusions.** Exo PR #2533 (head `c4685725bf6a71fb424e668744744499d44a4fea`, open) gates its fused kernels to M1/M3/M4. The reason it gives is "until verified on that hardware". That is missing validation, not a known numerical difference. Enable a port on M5 only after a bit-exact check against the unfused path on this machine. Exo PR #2530 (open) targets Inkling-**Small**, not Mini; its ideas transfer only where the Mini graph has the same operator.

16. **Owner: frontier campaign T3 (2026-11-23).** The specification is **Owner specification: frontier brief** below.
    - **Target (T3, owner, 2026-11-23).** In every cell, at least 1.10× the strongest reproducible engine on this
      machine under the same memory budget. The cells cover cold-turn prefill, warm decode, and 1/2/4/8-agent
      aggregate throughput, each on both repetitions. The comparators are:
      - current llama.cpp upstream;
      - llama.cpp's best Inkling PR or fork stack;
      - oMLX;
      - mlx-serve;
      - MTPLX.

      The bar is measured, not assumed. Same-weights cells go in the controlled table, and other quantizations go in
      the best-practical table at a declared quality standard. If either side speculates, compare speculation on
      against speculation on, and keep the off comparison.
    - **Stretch.** 2,000 tok/s cold prefill and 80 tok/s single-stream decode.
    - **Floor (T2).** At least 1.10× oMLX in every same-weights 1/2/4/8-agent cell, on both repetitions,
      recalculated on the corrected workload.
    - **Workload.** Both engines use the reasoning-replay clients from `69651393c`. Earlier sweeps are not comparable.
    - **Correctness.** Consistency tests and greedy equality are the gates. Fidelity is reporting-only. At integration
      milestones and at final integration, report:
      - the oMLX oracle (`inkling-omlx-oracle`);
      - the f32 saved captures (`.work/parity/inkling2-mini/ref`, through `omlx_gate.py compare`).

      Neither blocks integration.
    - **Delegation.** Native Claude Code subagents on Opus at max effort only: no `pi-subagent-bridge`, Codex,
      Command Code or Pi workers. The parent owns worktrees, review and integration.
    - **Measurement.**
      - Run the CPU gate with `PERF_GATE_MAX_BUSY_CORES=4`.
      - Screen each candidate with one gated pair; give winners three interleaved pairs; give retained changes
        integrated workload and quality checks.
    - **Unchanged.** The 32 GB budget, the 28 GB `heavy.py` cap, the 26 GB load limit, and the 32,768-token context
      cap. 128K, 256K and 1M remain analytic projections only.

17. **Owner: 2K prefill and 200 tok/s single stream (2026-11-23).** These replace Decision 16's stretch (2,000 cold
    prefill, 80 decode). T3's cells stay as reported targets but leave the critical path.
    - **Prefill target.** 2,000 tok/s at 1K, 4K, 8K and 16K, and near the cap with output headroom reserved, in a
      resident process on prompts it has not seen. First request after readiness is reported beside it, never
      folded in. A single 1K cell does not stand for the curve.
    - **Decode target.** 200 committed output tok/s, one stream, sustained over 512-token greedy replies on the
      quick prompt classes (code, reasoning, prose, tool continuation). Speculation is allowed and labelled.
      Ordinary decode, speculative decode and concurrent aggregate are reported separately.
    - **Why speculation.** oQ4e reads about 3.87 GB of weights per ordinary token (`inkling_profile.py`, with
      controllers charged only on passes 2 and 3). At the measured ~500 GB/s streaming rate that bounds ordinary
      decode near 129 tok/s before KV and activations. 200 needs several committed tokens per target round.
    - **Round budgets.** Mean round time, including draft, verify, selection, state commit, rejection and host
      work, must stay under 20, 25, 30 and 40 ms at 4, 5, 6 and 8 mean committed tokens. Acceptance alone is not a
      result.
    - **Reference.** Splash 1.7.2 (`incoai/splash`, Apache-2.0) is the same-model reference. Its 213 tok/s on a
      32-core M5 Max is an unqualified external prerelease peak until reproduced here. Its 16-core M5 Pro notes
      give 25.6 ms one-row and 48.3 ms eight-row cycles after 2,048 tokens, 6.30 tokens a cycle.
    - **Drafter.** `incoai/Inkling-2-Mini-DFlash2@4f9088ea66c2` (Apache-2.0, 0.99 GB BF16): 5 layers of hidden 2304,
      block 8, window 2048, taps 5/23/41/48, a rank-256 top-16 candidate selector and the target's head with
      softcap 30 after correction. It has the shape of `incoai/Flint-DFlash2`, which TensorFold already serves on
      the shared Metal DFlash2 runtime. Its draft residency fits Decision 8's 4 GiB.
    - **Correctness.** Greedy speculative output equals the qualified target's greedy output (Decision 13); KV,
      convolution and per-pass state roll back exactly. Task quality, looping, truncation and tool behaviour stay
      reported. Fidelity stays reporting-only (Decision 16).
18. **Delegated: one arithmetic for decode and verify (2026-11-24).** V1 found that keeping every grouped verify
    row bit-equal to the existing one-row kernels caps the 8-row verify at 45.6 ms, 2.7× its 17.0 ms weight
    floor, because a weight can then feed only the rows one SIMD group holds in registers. Splash's 26.3 ms uses
    matrix-unit expert tiles with another summation order, and its greedy output varies run to run.
    - **The ruling.** The target's decode arithmetic may change, provided one arithmetic serves the one-row step,
      the batch step and the grouped verify. A row's result must not depend on how many other rows share a
      kernel, tile or expert. Greedy speculative output must then still equal greedy plain output (Decision 17).
    - **The qualification.** A new arithmetic replaces the old only after four checks:
      - the `inkling-omlx-oracle` report shows no regression in mean KL beyond its run-to-run noise (0.0195 at D2);
      - quality dev shows no regression;
      - concurrent replies still equal solo replies;
      - the one-row step does not get slower.
    - **What it does not allow.** Output that depends on batch composition, as Splash's `--verify-rows auto`
      allows.

19. **Owner: oracle-relative rule for activation precision (2026-11-24).** A change that only lowers activation
    precision (for example `INKLING_PREFILL_HALF`) is not held to Decision 10's 99.5% top-1 agreement with the exact
    path, which measures distance from our own float path rather than from the reference. Instead, against the
    independent oMLX oracle (`inkling-omlx-oracle`, 3,015 rows):
    - mean KL within 1.10× the exact path's;
    - top-1 no more than 0.5 points below the exact path's;
    - the full quality set (`SET=full`) loses at most one case against the exact path, every lost case examined, and
      no more runaways (replies reaching the 8,192-token cap) than the exact path has (owner, 2026-11-24: counted
      net, after PH swapped one runaway for another);
    - Pi coding tasks no worse than exact.

    A variant that passes becomes the default; the exact path stays selectable. Weight requantization and KV
    compression are still separate experimental configurations under Decision 10.

20. **Delegated: prefill chunk schedules (2026-11-24).** Prefill bytes depend on chunk boundaries: on main's
    runner, one 1,024-row chunk against 896+128 rows of the same held-out code prompt gives no bit-equal row,
    99.32% top-1, mean KL 0.0036 and max |Δlogit| 14.4 (P5c). Main's own reuse-on and reuse-off schedules already
    differ this way, so no schedule is privileged. A change of chunk schedule passes when:
    - against the oMLX oracle, mean KL is within 1.10× main's and top-1 is at most 0.5 points below main's (adding a
      comparison at the served chunk size if it differs from the gate's 512);
    - lane-vs-main top-1 at margin ≥ 0.1, mean KL and max |Δlogit| are reported beside main's reuse-on vs reuse-off
      spread;
    - quality dev loses at most one case, every lost case examined, with no new runaway to the 8,192 cap.

    Bound: exact replay still holds inside one schedule (a repeated prompt, a store hit and a fork restore the state
    the chunk computed byte for byte).

### Baseline

[campaign/baseline.md](../campaign/baseline.md) holds the frozen binary's sha256, the loop timings and every known red test. In brief, for `fa13df5d` built with Zig 0.17.1:

- `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`: 239.3 s cold and 4.0 s warm no-op;
- `zig build test-flint-draft-host`: 17.7 s;
- `zig build test test-golden -Dcpu=apple_m1`: 547.6 s under the GPU lock, with 7 known red steps (the memcpy link check plus three tests in two modules each).

The benchmark load limit was `FLINT_LOAD_LIMIT_GB=36`, from W = 40,200,896,512 bytes, until the owner's 32 GB cap; it is now 27 (Decision 8).

Host facts measured on 2026-11-21:
- Apple M5 Max: 6 performance and 12 efficiency CPU cores, a 40-core GPU with Metal 4, and 48 GiB of RAM.
- macOS 27.2 (26B123), Xcode 27.2 (27B52), Metal toolchain 27.2.47.0.
- Zig 0.17.1 (owner upgrade; upstream `.zig-version` was 0.17.0). Python 3.15.1, uv, and Pi 1.5.0.
- 621 GiB of free disk.

### Items

#### P0: ground

- **C0. Ground** (S, no prerequisites). Owns:
  - `campaign/pyproject.toml`, `campaign/uv.lock`, `campaign/sources.lock.json`, `campaign/bootstrap.lock.json`;
  - `campaign/fetch.py`, `campaign/host.py`, `campaign/host.json`, `campaign/Makefile`;
  - `campaign/baseline.md` and `campaign/README.md`.

  It delivers:
  - `make -C campaign fetch [NAMES=...]`, an idempotent fetch with a hash check. It skips verified files, and small Hugging Face files are pinned by the sha256 of their first fetch at the pinned revision.
  - `make -C campaign host`, which writes `host.json`: sysctl, `system_profiler`, `sw_vers`, Xcode, Metal toolchain, Zig, and the Metal device name, family support and `recommendedMaxWorkingSetSize` through a `swift` probe.
  - `make -C campaign build-baseline`, which builds `fa13df5d` in `.work/wt/baseline` and records the sha256 of `tensorfold-native`.
  - **Baseline** timings and known red.

  **Acceptance:** `make -C campaign fetch NAMES="flint-int4 inkling-2-mini" && make -C campaign host && make -C campaign build-baseline` exits 0, the second `fetch` run completes with no downloads, and `baseline.md` lists each timing and every red test.

#### P1: Flint parity and harness

- **F1. True autoregressive switch and first inference** (S, after C0). Owns the Flint set.
  - Prove whether `--no-drafts` disables Flint's copy drafts as well as `--drafter`. Make the per-request lane counters show 0 proposals and exactly 1.00 tokens per round.
  - If it does not, add a verified switch and a test that fails when copies run.
  - Serve one greedy request.
  
  **Acceptance:** `zig build test-flint-draft-host`, plus a new Flint test for the switch; under the lock, `make -C campaign smoke MODEL=flint MODE=ar` returns 512 tokens with zero proposals.
- **C1. Benchmark harness** (M, after C0; parallel with F1). Owns `campaign/bench/`, `campaign/prompts/` and `campaign/scorecard.py`.
  - Build on `tensorfold benchmark` and `tools/bench_*.py`; do not write a new framework.
  - **Corpus:** preserve the original fixed 1,024-, 4,096- and 32,768-token prompts and their hashes, covering
    code, prose/reasoning and structured output with separate held-out prompts. The active quick matrix is
    1K/4K. The original 32K prompts cannot fit a reply under the new 32K total cap; any future near-cap cell
    needs a separately named and hashed fixture with reply headroom, not an in-place truncation.
  - **Modes:** `ar` (`--no-drafts`), `copy`, `drafter` and `combined`.
  - **Fixed length:** `ignore_eos: true` with 512 output tokens on every engine. Natural terminations are reported separately.
  - **Records:**
    - startup and load time, cold prefill and warm requests, and prefix-cache hits, with cached and uncached token counts;
    - TTFT, prefill rate, decode rate and total latency;
    - peak and steady footprint;
    - acceptance and committed tokens per verification.
  - **Decode interval:** from the end of prefill to the last committed token, using synchronized wall time.
  - Raw trials go to `campaign/runs/<date>/<run>.jsonl`, tracked in git, including failures. Binary, weight and config hashes go in each record. `scorecard.py` regenerates `campaign/scorecard.md` from `runs/` alone.
  - `--publish` and `--upload-url` are never passed.
  
  **Acceptance:** `make -C campaign bench MODEL=flint MODE=ar CELLS=1024 PAIRS=1 && make -C campaign scorecard`. A second `scorecard` run must produce an identical file.
- **F2. Flint reference parity** (M, after F1). Owns `campaign/parity/flint/`.
  - Run teacher-forced logits from `tf-flint-run` against `rec_mlx` on 16 held-out histories of at most 4,096 tokens.
  - Report the full distribution of error: absolute and relative error, top-1 agreement, logit margins and KL.
  - Trace the first divergent tensor for any outlier. Check tokenizer, masks, quantization and config before calling a difference "floating-point noise".
  - Write the baseline numbers into **Baseline**.
  
  **Acceptance:** `make -C campaign parity MODEL=flint` exits 0 and writes `campaign/parity/flint/report.md`.
- **Q1. Quality suite** (M, after C1). Owns `campaign/quality/`.
  - 48 cases, drawn from public or local evaluations where practical:
    - 16 executable code tasks, scored by tests;
    - 12 structured-output or tool-call cases, which need `zig build grammar` with xgrammar v0.2.8 pinned in the lock;
    - 12 reasoning cases;
    - 8 context-retrieval cases at 4K and 32K.
  - A development subset of 12.
  
  **Acceptance:** `make -C campaign quality MODEL=flint SET=dev` scores the frozen baseline and records it.
- **H1. Pi harness** (S, after C1). Owns `campaign/pi/`.
  - `campaign/pi/agent/models.json` defines a local OpenAI-compatible provider at `http://127.0.0.1:8888/v1`.
  - 8 coding tasks run under `sandbox-exec -f campaign/pi/sandbox.sb`. Writes are allowed only in `.work/pi/<task>` and `.work/tmp`; network only to localhost.
  - Record task pass/fail, tool-call parse errors, turns and per-turn latency.
  
  **Acceptance:** `make -C campaign pi-harness MODEL=flint` gives 8/8 runs completing, and 0 tool-call parse errors on the baseline.
- **F3. Freeze the Flint baseline matrix** (M, after F1, C1 and F2). Owns `campaign/runs/`.
  - Run the primary matrix, 3 prompts × 3 lengths, on the frozen build in `ar` and `copy` modes with 6 pairs.
  - **Comparators:** `rec` v0.1.0, Exo `main` and the #2533 head, and mlx-lm where it serves Flint. Pin each in the lock. An engine that cannot run here is marked unavailable.
  - Rach's engine and codegraff's nanoattention get a 30-minute check for public, licensed, runnable source. Code under the "Other" licence is benchmarked only if its licence permits, and never copied.
  - Record omitted cells and the reason for each.
  
  **Acceptance:** `make -C campaign scorecard` shows both tables, matched and best-useful, for every runnable engine. **Milestone M1:** full suite.

#### P2: Inkling 2 Mini parity

I-items own the Inkling set. They may start code work parallel to F-items, and use the GPU only under the lock.

- **I0. Prior-art map and reference** (M, after C0). Write `docs/recipes/inkling-2-mini.md` from the pinned llama.cpp graph, the checkpoint index, `hf_quant_config.json` and the chat template. It must cover:
  - the 24 layers, with `local_layer_ids` and a 1024-token sliding window;
  - the core loop from layer 6 to 18 with 3 passes and `gate_rank` 64; each pass keeps its own KV and convolution state, and later passes take different Q/K/V inputs;
  - the attention-residual blocks;
  - `d_rel`/`rel_extent` relative positions and log scaling;
  - the qk and rel L2 clips;
  - 128 experts with 8 active per token, gate bias, `route_scale` and softcaps;
  - the final logit softcap of 30, and the 201,024 vocabulary padded from 200,058;
  - the per-tensor NVFP4/MXFP8 groups and the MTP head taps.
  
  Done. Its f32 Python reference and `ref-check` were retired on 2026-11-23 for the oMLX oracle (Decision 9).
- **I1. Family loader and refusals** (M, after I0). This covers config, tensor map, tokenizer and template, NVFP4/MXFP8 geometry checks, registration, and a clear refusal of image parts and `--vision`. Vision is out of scope.
  
  **Acceptance:** `zig build test-inkling2-loop-host` passes, including tests that the refusals fire.
- **I2. Executable native forward path** (L, after I1). Reuse the shared NVFP4/FP8 kernels and the Flint/GLM attention pieces where their contracts match; write new kernels only where they do not. State stays explicit for each pass. Increments: a traceable tiny-fixture path through all loop passes and both packed formats; a short teacher-forced path on the real checkpoint with experts kept packed; incremental continuation equal to recomputation, with distinct per-pass KV and convolution state; focused probes at convolution, window and position boundaries using targeted fixtures and selected rows. Name the arithmetic configuration implemented. The checkpoint's quantization metadata specifies activation quantization while the reference computes float32 activations, so a higher-precision-activation path is qualified as that configuration, and the official activation-quantized path is validated separately before claiming equivalence. The MTP tap interpretation stays marked inferred.
  
  **Acceptance:** the pinned checkpoint loads within the memory budget; `tf-inkling2-loop-run` executes a short prompt through all execution slots and at least two greedy decode steps with speculation off; captured logits and selected intermediates pass the logit gate under the declared initial-port contract (independent oracle and state checks, not a regression bound against an I3 baseline that does not yet exist); reset and rerun reproduce the result bit for bit.
- **I3. Serving and baseline** (M, after I2, C1, Q1 and H1). Cover chunked versus full prefill, cached continuation versus a fresh request, EOS, cancellation, context limits, reset after errors, reasoning template, special tokens and tool-call syntax. Then freeze the Inkling baseline matrix. Comparators: llama.cpp (Metal, ggml-org GGUF), MLX (`mlx-community/Inkling-2-Mini-4bit` through mlx-vlm), Exo and Ollama where they run.
  
  **Acceptance:** `zig build test-inkling2-loop`, `make -C campaign quality MODEL=inkling2-mini SET=full`, `make -C campaign pi-harness MODEL=inkling2-mini` and `make -C campaign scorecard`. **Milestone M2:** full suite.

#### P3: hardening and speculation

- **F5 and I5. Greedy speculation** (M each; F5 after F3, I5 after I3). F5 is Flint DFlash2 plus copies. I5 is the Inkling MTP head. Test:
  - zero, partial and full acceptance;
  - EOS and the output limit inside a block;
  - rollback of every KV, convolution and per-pass state;
  - resumption after fallback;
  - block versus stepwise verification masks.
  
  Speculative output must equal the qualified target's greedy output on the same histories. Then add sampled mode: enumerable distributions, zero-probability cases, top-k/top-p, temperature, repetition penalty, and grammar constraints. Tune draft length from total measured cost.
  
  **Acceptance:** each model's speculation tests plus `make -C campaign bench MODE=drafter` within the speculation memory envelope.
- **K1. Contract write-up:** record the contract changes listed under **Execution boundary**, and update `docs/supported-models.md` and the recipes.

#### P4: optimization

- **O-items** (each S–M). Each is created from the ranked hypothesis table after profiling: Metal System Trace and capture outside timed windows, short and long context. Profiling attributes time to experts and projections, attention, routing, norms and residuals, quantization, dispatch, synchronization, sampling and memory movement, together with bandwidth estimates.
  
  Each O-item follows this loop: bottleneck, hypothesis, smallest change with a fallback, focused correctness checks, a short paired screen, then the full matrix and held-out prompts for winners. Commit winners. Revert losers but keep their ledger entry and runs. Re-profile and pick the next.
  
  Flint O-items and Inkling O-items run in parallel lanes, with the GPU serialized.
  
  **Plateau:** stop a model's optimization after re-profiling and at least eight substantive failed candidates across more than one hypothesis family, or sooner if a measured hard bound (for example, decode at about 90% of measured weight-read bandwidth) establishes the limit.
- **S2. Secondary investigations:** 128K context is deferred by the owner's 2026-11-22 cap. Any future
  concurrency investigation uses at most 32,768 total tokens per stream and reports aggregate throughput
  separately from single-user speed; it is not part of the current quick matrix.

#### P5: close

- **Z1.** Fetch upstream once and record what changed, without rebasing; import upstream fixes only as separately recorded commits. Use `dead-code-archaeology` to delete campaign scaffolding and kernels made dead by a promoted replacement; prove deadness by search plus the build. Run `ultrareview` and fix its P1 and P2 findings. Write `campaign/report.md`:
  - the Mac, models and quantization, and lengths and modes;
  - raw rates with uncertainty, memory peaks and baseline commits;
  - task quality and its limits;
  - local reproduction kept separate from external claims;
  - regressions, unsupported features, targets met or missed, and the narrowest claim the data supports.

### Owner specification: frontier brief (2026-11-23)

Kept verbatim from the owner's T3 brief, *TensorFold: advance the Inkling 2 Mini inference frontier*. Decision 17 narrows its critical path.

Continue the optimization campaign in joshuarli/tf. Establish how current TensorFold compares with the strongest reproducible Inkling 2 Mini implementations, then improve the native Zig/Metal engine on the campaign machine. Optimize complete coding-agent work: cold and warm turn latency, sustained generation, concurrent throughput, memory, and useful task completion.

Treat this as a research-and-implementation campaign. Use the source map below as a starting inventory, discover newer work, and turn the best supported ideas into measured experiments. “SOTA” means an explicitly stated workload, hardware, memory budget, quality level, and date. It is not a universal ranking.

Research snapshot: 23 November 2026. Refresh every moving branch and PR before using it.

#### 1. Recover the current ground truth and preserve campaign scope

Read the current AGENTS instructions, docs/roadmap.md, campaign/shootout.md, campaign/kernels ledgers, current source, pending branches, and latest owner steering. Resolve historical statements by chronology and actual code. Preserve the current delegation and integration policy.

The target is Inkling 2 Mini, architecture inkling2_loop. Distinguish it from Inkling 2 Small, earlier Inkling models, research checkpoints, and modified/abliterated variants. Other models supply possible mechanisms, not interchangeable benchmark results.

The known campaign host is M5 Max, 40 GPU cores, 48 GiB. Reconfirm OS, Xcode/Metal, compiler, build options and power/thermal state. Preserve the current 32 GB campaign budget, 28 GB charged-run ceiling, 26 GB load limit, and heavy.py accounting unless later owner instructions supersede them. Keep benchmark context within 32,768 total tokens including reserved output. Research 128K/256K/1M scalability and report projections separately; do not run those cells without an authorized cap change.

The shipped implementation remains native Zig/Metal. Python, MLX and other engines may serve as development references and comparators. Preserve existing boundaries on system settings, global installations, external writes and paid compute.

Two mandatory starting facts:

- TF commit [69651393](https://github.com/joshuarli/tf/commit/69651393c736d3fa263b6959514fb897e9082207) corrects reasoning replay in both agent workload clients. Earlier sweeps dropped reasoning_content and used an ignored enable_thinking setting. Remeasure both sides on the corrected workload.
- Read [the existing DSpark receipt](https://github.com/joshuarli/tf/blob/main/campaign/kernels/inkling-dspark.md) and parked lane/ds1-dspark-drafter before proposing a new implementation. Its measured draft cost invalidated earlier optimistic estimates.

Audit which ideas are implemented, pending, rejected, superseded, or genuinely untested. Existing f16 KV, prefix reuse/store, batching, controller fusion and TensorOps work are not new research. Revisit a failed experiment only with a concrete changed hypothesis, workload, precision contract or implementation.

Maintain the distinction between state/kernel consistency, fidelity to the reference, and speculation against its chosen target. Under the latest owner steering, fidelity is reporting-only: run the saved-capture comparison at final integration, and do not revive older top-1/KL thresholds as lane or integration gates. Consistency, greedy-equality where required, and applicable task-quality requirements remain in force.

#### 2. Expand and maintain the source map

Search GitHub commits, open and merged PRs, review comments, issues, releases, forks, experimental branches, benchmark scripts, raw receipts and negative-result ledgers. Search X/Twitter for author announcements, replies, corrections, linked branches and independent reproductions. Follow every material social claim to primary code or measurements where possible.

Use model aliases, architecture names and mechanism terms: Inkling-2-Mini, inkling2_loop, DSpark, Markov, MTP, NVFP4, MXFP8, relative bias, sink attention, Metal, NAX, TensorOps, speculative verification, expert routing, KV, prefix and agent serving. Search both the last 24–72 hours and the release window. Trace dependencies beyond commits mentioning the model.

For each lead record date checked, exact revision, status, source/license, hardware, model/quantization, actual execution path, workload, metric definition, reproducibility, correctness evidence and applicability to TF. Deduplicate forks. Distinguish author claims, reproduced results, source inspection, hypotheses, and inaccessible evidence. Do not claim exhaustive X coverage if search or post retrieval is incomplete.

##### Required starting sources

| Area | Starting points and questions |
|---|---|
| Official model and conversions | [Official model](https://huggingface.co/thinkingmachines/Inkling-2-Mini), [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/), [Unsloth GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), [DSpark conversion notes](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/dspark/README.md), [Xylolabs](https://huggingface.co/Xylolabs/Inkling-2-Mini-GGUF). Verify tensor formats, scales, tokenizer, template, drafter schema and checkpoint identity. |
| llama.cpp upstream | [Current upstream](https://github.com/ggml-org/llama.cpp). Refresh #32825 native Metal NVFP4, #32740 compact-bias contract, #32808 Metal indexed-bias attention, #32752/#32883 chunking, #32855 depth-mix operand order, #32824 CONCAT, #32218 DSpark and #32773 speculation costs. Inspect final diffs and dependencies. |
| llama.cpp supporting work | #32240 separate global/SWA KV types, #32777 loop-aware placement, #30196 output synchronization, #32076 MoE cache, #30685 host-input copies. Audit tokenizer/template/scale/cache fixes #32799, #32763, #32704, #32807, #32828 and #32833. Establish backend and architecture eligibility before proposing flags. |
| llama.cpp forks | [Koosh0610](https://github.com/Koosh0610/llama.cpp-inkling), its [technical write-up](https://koosh0610.github.io/llama.cpp-inkling/), and [realugbun/inkling2-speed](https://github.com/realugbun/llama.cpp/tree/inkling2-speed). Compare their changes with current upstream; isolate what remains novel. |
| MLX ecosystem | [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm), [MLX-VLM](https://github.com/Blaizzy/mlx-vlm), [oMLX](https://github.com/jundot/omlx). Refresh model branches and kernel changes, including mlx-lm #2086 and MLX #4712/#4781/#4796. Inspect oMLX #5505, #5508, #5528, #5531, #5559 and #5560. |
| New Mac comparators | [mlx-serve](https://github.com/ddalcu/mlx-serve), especially [drafter global-context fix f501857](https://github.com/ddalcu/mlx-serve/commit/f50185721650f6d8045a550844c70a87f341f49b); [MTPLX](https://github.com/youssofal/MTPLX), [2.13.2 notes](https://mtplx.ai/releases/2.13.2/), and [Nov 23 comparison](https://www.mtplx.com/compare/mlx-serve-vs-omlx-vs-mtplx/). Audit sampler paths, exactness and raw benchmark methodology. |
| Native Metal research | [nanoattention](https://github.com/justrach/nanoattention), [DankAttn](https://github.com/Dankpaws/DankAttn), [erisinky](https://github.com/Eriskii/erisinky), [gall](https://github.com/ltrgoddard/gall), [tur](https://github.com/danylokravchenko/tur), [rustling](https://github.com/onatm/rustling), [rink](https://github.com/adapt/rink), [saetta](https://github.com/antonellof/saetta). Identify actual Mini support, serving capability and numerical tradeoffs. |
| Quantization and profiling | [lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant), [deeflect/inkling-m4-metal](https://github.com/deeflect/inkling-m4-metal), [mps-moe-perf-notes](https://github.com/ArthurZucker/mps-moe-perf-notes). Separate weight compression, KV compression, activation precision and changed reasoning policies. |
| CUDA/ROCm/Vulkan prior art | [TokenSpeed](https://github.com/lightseekorg/tokenspeed), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [kern-inkling-2mini](https://github.com/pegainfer-project/kern-inkling-2mini), [NInfer](https://github.com/iamwavecut/ninfer-all), [ninfer-3090](https://github.com/ashalliants/ninfer-3090), [Strata](https://github.com/Niko1221/Strata), [capy-flash](https://github.com/ValerioDolci/capy-flash). Extract mechanisms, including exact hardware guards; do not transfer their tok/s or percentage gains to Metal. |
| Scheduling, offload and broader discovery | [Resin](https://github.com/ekzhang/resin), [Exo](https://github.com/exo-explore/exo), [TideLM](https://github.com/raspy135/tideLM), [Pulsar](https://github.com/loopai-hq/pulsar), [awesome-inkling](https://github.com/deeflect/awesome-inkling). Follow references to additional projects. Larger-model, other-model and distributed work stays transfer research until Mini applicability is established. |

This is a discovery inventory, not a mandate to install every engine. Prioritize sources with relevant code and reproducible evidence. Verify licenses before porting and preserve required notices.

#### 3. Establish comparisons that answer different questions

Keep three result tables:

1. **Controlled engine comparison:** matched checkpoint/tensor values where formats permit, tokenizer, rendered input, reasoning policy, precision settings, KV policy, sampler, speculation mode and workload. Disclose every remaining arithmetic difference.
2. **Best practical configuration:** each engine tuned within the same machine/memory budget and a declared task-quality standard. Different quants or drafting strategies are allowed but identified.
3. **Experimental speed–quality frontier:** explicitly approximate changes, with their quality and memory costs. Never merge these results into the controlled table.

Same file names or “official weights” labels are insufficient. Inspect tensor hashes/values, scale layouts, conversion steps and included vision/drafter weights. Preserve distinct official-format and oQ4e results.

Benchmark current upstream llama.cpp separately from selected PR/fork stacks. Verify GPU placement and eliminate unintended CPU fallback. Refresh oMLX, mlx-serve, MTPLX and nanoattention before treating an older version as competitive. Record exact commands and SHAs.

Use identical token arrays or record and explain tokenizer differences. Fix chat templates, reasoning_effort, hidden reasoning, tool schemas, stop IDs, EOS handling and output budgets. Correctly replay reasoning and tool calls.

Measure cold load separately from first request, unseen prefill, warm repeated prompt, continuation, prefix fork and mixed multi-turn work. Warm up on unrelated inputs when measuring cold prefill. Let engines use appropriate chunk sizes and record those choices.

Start with 1K/4K cells; extend winners to 16K and applicable cells under the total-token cap. Exercise 1/2/4/8 concurrent agents, including staggered arrivals and unequal histories. State client count and resident slot count separately. Include long generations and copy-heavy edits alongside novel code, prose, reasoning and structured/tool outputs.

Report:

- Client and server TTFT; first engine token, first reasoning token and first visible answer token separately.
- Actual evaluated input tokens and reused tokens; prefill latency and throughput.
- Committed output tokens over the full decode interval, per-agent and aggregate rates; distinguish reasoning, visible answer and tool arguments.
- Complete turn/task time, queue delay, latency tails, errors, retries and completed work.
- Charged memory, resident/file-backed weights, scratch, target/draft KV, snapshots and prefix stores.
- Speculative proposals, acceptance, committed tokens per round and the full cost of each round.
- Quality, output/schema correctness, thermal/contention conditions and uncertainty.

Do not count proposed or replayed tokens as useful output. Do not call a configured context capacity a populated-context result. Do not compare a fastest 32-token stretch with a full-answer average. Use both fixed replay workloads for kernel isolation and live agent tasks for practical value.

#### 4. Research avenues to cover and triage

##### A. Model semantics and loop-aware execution

Audit 24 physical layers versus 48 logical visits, repeated expert traffic, per-pass KV and convolution states, local/global attention, relative-position bias and sinks, depth-residual banks, normalization and loop controllers. Preserve raw DSpark taps and frozen snapshot ownership.

Investigate reuse of immutable normalized states, graph order, depth-mix operand orientation, shared reads and direct writes into the next consumer. Check aliasing and buffer lifetime. Account for prior TF failures before retrying cached normalization. Do not share loop states merely because physical weights are shared.

##### B. Expert execution and packed weights

Profile real routed-row distributions and expert fan-out at single-token decode, few-row verification, concurrency and prefill. Investigate expert grouping/worklists, gather/scatter elimination, route sorting, occupancy-dependent dispatch, gate/up/down tiles, shared activation loads, fused activation/reduction and register-level dequantization.

Compare native NVFP4/MXFP8, affine and other permitted layouts. Evaluate scale traffic, vectorization, split-K, tensor operands and precision by shape. Check whether staging weights costs more bandwidth than it saves. Inspect the actual Metal scale-plane contract: native NVFP4 and MXFP8 scale formats are not interchangeable. Treat single-row and multirow results separately.

##### C. Attention and positional bias

Investigate compact/indexed relative bias inside attention, sink handling, FlashAttention/FlashDecoding, GQA sharing, split-KV, query tiling, local-window bounds and paged/ring layouts. Measure crossover points by populated context and row count.

Audit masking, absolute positions, clipping/softcaps, summation order and row-width consistency. Compare dense-mask, mask-chunk and query-chunk approaches with both peak memory and latency. Metal scheduling may execute nominally separate chunks concurrently, extending allocation lifetimes. Smaller intermediates and fewer launches are hypotheses, not speed guarantees.

##### D. Prefill and intermediate lifetime

Investigate larger/adaptive chunks, aggregate expert routes, tile occupancy, scratch packing by live interval, phase aliasing, direct controller outputs, and backward row-liveness that skips outputs never consumed. Preserve every state needed by future loops, convolution, DSpark, caching and later turns.

Benchmark at actual prefix depth, with cold caches and real token distributions. Distinguish terminal-row-only head work from skipping required intermediate rows. Recheck nanoattention’s corrected methodology and rejected variants before relying on early headline gains.

##### E. GPU scheduling and host overhead

Measure CPU encoding, command-buffer submission, synchronization, resource barriers, dependency depth, allocations, first-touch residency, page faults, queue gaps and CPU/GPU overlap. Use paired traces and wall timings; do not subtract unrelated profiling runs.

Evaluate reusable buffers/arguments, pipeline caches, shape buckets, indirect dispatch, concurrent independent work, reduced readbacks and safe command scheduling. Investigate persistent kernels only where dependencies and platform guarantees support them. Check register pressure, occupancy and pipeline-specific threadgroup limits. Removing dispatches alone is not a success criterion.

##### F. Few-row verification

Treat verification as a separate workload. Compare decode-style row-exact kernels with batched GEMM/attention for widths around all dispatch thresholds. Preserve the chosen target’s arithmetic where greedy equality is required.

Use fixed real token/route traces as well as live drafts. Price padding, expert fan-out, head computation, graph capture/rebuild, compilation and additional scratch. Evaluate fixed and bucketed widths before adaptive widths; include the cost and memory of keeping multiple graphs.

##### G. Native DSpark and alternative proposals

Audit the existing TF branch first. Confirm runtime/converter/drafter compatibility, ctx_mix normalization, Markov tensors, target taps, embedding/head sharing, vocabulary padding and local/global drafter attention. The trained eight-slot block includes an anchor; distinguish verification width from up to seven newly proposed tokens.

Investigate lower-cost drafter precision, efficient head/Markov computation, context updates, full-history attention, shared target features and batched drafts. Benchmark no drafter, loaded-but-paused drafter, and active drafting.

Compare fixed proposal counts 0–7 with adaptive policies based on measured accepted tokens per millisecond, context, concurrency and expert fan-out. Include warmup, draft, verification, rejection, rollback, state-copy and scheduler costs. Use hysteresis/backoff only if they win.

Explore prompt lookup, context copies, suffix/ngram proposals, hybrid copy-plus-DSpark, and carefully bounded trees/beam proposals. Longer copy windows need not obey the trained drafter’s block size, but must satisfy target verification and state semantics. External drafters, EAGLE/DFlash-style methods, self-speculation and distillation are conditional research: establish model support, training needs and feasible resources first.

##### H. GPU sampling and exact speculative semantics

Profile GPU-to-CPU logits transfer, vocabulary reduction, softmax, top-k/top-p/min-p, temperature, penalties, logit bias, RNG and grammar work. Investigate GPU-resident selection and certified filtering without unnecessary full sorts.

For sampled speculation, establish the actual proposal and target distributions, acceptance/rejection correction, bonus tokens and history-dependent controls per speculative row. Include earlier proposals when computing that row’s penalties or grammar state.

Separate greedy equality, preservation of a specified target distribution, and seed-identical outputs. These are different claims. Audit fallback behavior for unsupported controls, uncertain thresholds and numerical edges; do not restart already reached sampling decisions in a way that conditions them on later events. Keep a correct fallback and measure when real clients trigger it.

##### I. KV, convolution and speculative state

f16 KV is already implemented. Explore global/local policies independently; key/value asymmetry; FP8/int8 and lower-bit methods; scales and outliers; fused dequantization; paging; bounded sliding rings; and state reserved only for history that can be read again.

Treat generic KV-quantization support as unproven for Inkling-specific attention until the actual kernels accept it. Audit per-pass ownership, absolute positions, prefix boundaries, convolution history, rejected tokens and target/drafter rollback together.

Measure prefix sharing/copy-on-write, selective snapshots, immutable shared prefixes and private suffixes. Include draft state and graph buffers in admission. Research 128K/256K scalability analytically under the existing run cap; distinguish projected allocation from measured speed, stability and retrieval quality.

##### J. Serving, concurrency and cache policy

Investigate continuous batching, ragged histories, prefill/decode interleaving, decode latency protection, admission, fair scheduling, cancellation and session eviction. Compare synchronized synthetic agents with realistic staggered work.

Measure longest-prefix lookup, cross-session forks, prompt-state checkpoints, cache eviction, disk persistence and target-plus-draft restore atomicity. Optimize useful reuse under the corrected reasoning history. Price cache storage and copying against saved prefill, and ensure all mutable state is restored consistently.

Choose speculation per session or batch when warranted. A single-stream gain may lose under concurrency because verification increases expert fan-out or occupies the GPU longer.

##### K. Agent protocol and application costs

Profile rendering/tokenization, reasoning replay, streaming, tool argument parsing, structured decoding, stop handling, response assembly and network flushing. Include long source-file tool arguments and parallel calls.

Look for quadratic rescanning, whole-history copying, repeated tokenization, blocked event loops and unnecessary synchronization. Inspect vLLM’s recent Inkling parser/state work, including #69293/#69297/#51141, #69351/#69347 and #69444/#69181, as mechanism leads. Check final status and applicable model/backend.

Measure completed tool round-trips and task success. Preserve exact arguments, escaping, Unicode, newlines, call IDs and finish reasons. Keep first visible answer latency separate from hidden reasoning generation.

##### L. Precision and the quality–speed frontier

Explore selective activation precision, accumulator choices, per-layer/per-operation sensitivity, mixed-bit weights, importance-calibrated quantization, head/embedding compression, drafter quantization and compressed KV. Separate each intervention so its benefit and loss can be understood.

Report task quality, coding checks, tool/schema success, retrieval, refusals, looping/runaways, non-finite values and representative logit/greedy comparisons. Include calibration/holdout separation and relevant long histories within the cap.

Expert pruning/skipping, sparse attention, stale-state reuse, reduced loops, early exits and reduced reasoning budgets change model behavior. Keep them explicitly experimental and separately evaluated. A variable-loop research checkpoint is a different comparator. Do not silently apply fewer loops to the fixed Mini checkpoint or label modified models equivalent.

##### M. Hardware limits, offload and other architectures

Build a measured traffic/compute model for actual kernels: physical and repeated weights, KV, activations, scales, head, controllers and temporary buffers. Compare attainable bandwidth and tensor throughput by shape with measured stage time. Use Amdahl-style end-to-end estimates, not multiplication of isolated speedups.

Study CUDA/ROCm/Vulkan implementations for transferable scheduling, worklists, paging and sampling ideas. Inspect architecture guards before assuming a generic optimization executes for Inkling.

Expert caching/offload, CPU/GPU overlap, tensor/expert/context parallelism and multi-device execution are conditional future avenues. Mini fits the present machine, so justify their relevance before introducing complexity or new hardware. Vision preprocessing, encoder/projection caching and multimodal prefixes are similarly conditional on expanding the current text-agent scope.

#### 5. Execute in an evidence-driven order

1. Freeze the corrected workload and current source/weight pins. Produce fresh TF and primary comparator baselines, including speculation on/off where supported.
2. Measure the integrated TF build, including already landed post-shootout changes. Profile single-stream attention, multi-stream expert work, prefill experts and actual served latency. Choose priorities from current measurements.
3. Build a compact research ledger: mechanism, source/revision, applicability, existing TF status, expected benefit, implementation cost, correctness risk and next discriminating experiment. Mark each avenue tested, active, rejected with evidence, deferred with reason, or unsupported.
4. Select the highest-value bounded experiments. Parallelize independent source research and coding within the existing delegation policy; serialize GPU timing and keep one resident engine at a time.
5. Follow the current lightweight progression: one gated paired screen; three interleaved pairs for promising candidates; integrated workload and quality checks for retained changes. Use identical-build controls or additional pairs when a concrete noise concern requires them. Reuse existing harnesses and receipts.
6. Keep, revise or revert based on complete-request benefit, memory and applicable correctness/quality requirements. A local kernel win, higher acceptance, lower dispatch count or smaller scratch allocation does not establish an end-to-end win.
7. Refresh the frontier at integration milestones. Continue when new evidence changes the best next experiment. Avoid endless surveying before implementing a strong lead or repeatedly testing an unchanged failed idea.

Retain the owner’s target of at least **1.10× oMLX aggregate throughput in every same-weights 1/2/4/8-agent cell, on both repetitions**, recalculating thresholds from the corrected workload. If TF uses MTP, remeasure oMLX with MTP on and retain the off comparison. Treat previously suggested absolute rates such as 2,000 prefill tok/s or 80 decode tok/s as aspirations unless current hardware evidence supports them. Establish the broader competitive frontier beyond one old oMLX baseline.

#### 6. Required deliverables

Produce an updated source/coverage map, reproducible current scorecards, a ranked experiment ledger, integrated improvements with concise receipts, and a remaining-opportunities list.

The final comparison must answer:

- Where does TF lead or trail current llama.cpp upstream and the best relevant experimental stack?
- Where does it stand against current oMLX, mlx-serve, MTPLX and other credible native Metal implementations?
- Which results isolate engine performance, and which reflect different precision, weights, speculation or output behavior?
- What wins on cold turns, warm turns, long generations and 1/2/4/8 agents under the same memory limit?
- Which proposed optimizations failed, why, and what evidence would justify retrying?
- What is the highest-value next experiment, and which claims remain unverified?

Support performance claims with exact revisions, settings and receipts. State inaccessible X posts, missing raw data and untested transfer assumptions explicitly. Do not call TF SOTA or the research exhaustive on the strength of marketing peaks, a short greedy match, theoretical bandwidth, or an incomplete search.

**Begin with the current-state audit and corrected baseline, then run the first high-value experiment. Do not stop at a literature review.**

### Acceptance

1. **Native engine.** `zig build native -Dcpu=apple_m1` builds. Flint and Inkling 2 Mini serve text. Qualified defaults are on, with opt-in experimental modes and fallbacks. `zig build test test-golden` adds no red beyond what **Baseline** lists.
2. **Correctness.** `make -C campaign parity MODEL=flint` and `make -C campaign quality MODEL=… SET=full` meet Decision 10. Inkling's `make -C campaign omlx-oracle-gate` is reported (Decision 16).
   - Speculation tests pass, including state rollback.
   - **Failure condition:** any advertised mode whose output differs from its target-distribution contract, or a free-run text comparison standing in for teacher-forced evidence.
3. **Harness.** `make -C campaign bench …` and `make -C campaign scorecard` regenerate both tables from `campaign/runs/`. `make -C campaign pi-harness` passes for both models.
4. **Provenance.** `make -C campaign fetch` verifies every pin. `campaign/sources.lock.json` and `host.json` record source SHAs, PR heads, compilers, model revisions, tokenizer and config hashes, quantization metadata and licences. `NOTICE` and `THIRD_PARTY_NOTICES.md` cover any imported code.
5. **Ledger and report.** `campaign/ledger.md` lists every experiment, including failures. `campaign/report.md` states each target in Decision 12 as met or not met, with its numbers.
6. **Docs.** `campaign/README.md` says how to build, fetch weights, run, benchmark, verify and resume.

### Needs environment

- M1–M4 compatibility of new kernels: an offline compile only, run with `zig build dev-tools`. Running them needs older Macs; the commands are the same `make -C campaign parity MODEL=flint` and `make -C campaign omlx-oracle-gate` on such a machine.
- Linux and CUDA host CI (`zig build -Dfixed-cpu test-host`) on a Linux host, if the owner wants upstream-style evidence.
- Comparators that need CUDA, such as vLLM and SGLang: unavailable here, not defeated.

### Execution boundary and non-goals

**Authorized:**
- edits and local commits on `main` and `lane/*` branches in this repo;
- `git fetch upstream`;
- downloads pinned through the lock: public weights, comparator sources and Python development oracles in `.work/` uv environments;
- writes under `~/models/`;
- local servers on 127.0.0.1;
- GPU runs and profiling under the lock;
- adding development-only Python tools in `campaign/`.

The shipping path stays native Zig/Metal with its existing C/Objective-C glue. A new Zig or native runtime dependency must:
- keep the stripped `tensorfold-native` size growth at or below 2%;
- add no decode-path cost;
- be recorded in the lock with its licence.

**Forbidden:**
- pushing, PRs, issues or comments upstream;
- ATC `claim`, `start`, `pulse`, `land` or `--public`;
- `tensorfold benchmark --publish` and `--upload-url`;
- renting compute or buying services;
- changing system-wide security, memory or power settings;
- writing to `~/.pi`, or installing tools globally (the owner allows global caches);
- deleting files the campaign did not create;
- Python, MLX or another framework in the shipping path;
- a language rewrite.

**Contract changes for the owner** (K1 keeps this list current):
- the new `inkling2_loop` family and its CLI and capability surface;
- any new switch from F1;
- any default changed by a promoted O-item.

**Non-goals, with reasons:**
- distributed inference, CUDA optimization, training and new UI: all outside the single-Mac scope;
- Inkling vision input: text and coding use come first, and images are refused with a clear message;
- the variable-loop research checkpoint (`TML-d24-varloop-research`): a different configuration;
- compatibility work for older Macs: see **Needs environment**;
- repository-wide mutation, coverage or formal-proof work: only the changed inference path is tested.

### Reference

These references were verified on 2026-11-21:

- **TensorFold.** Apache-2.0. Flint recipe at `docs/recipes/flint.md`. Flint native entry at `zig/src/native/flint.zig`. `--no-drafts` defined at `zig/src/server/cli.zig:88`. `ignore_eos` honoured at `zig/src/server/chat_generation.zig:82`. `tf-flint-run` registered at `zig/build/metal_targets.zig:283`. The Inkling 2 Small host package is wired at `build.zig:123`.
- **Upstream receipts** in `docs/recipes/flint.md` come from an M5 Ultra with 256 GiB. They are not a baseline for this machine.
- **`buildlib.locks`:** `load_sources` and `fetch_input` (URL order, digest and size checks, atomic publish).
- **Hugging Face pins:**
  - `Inkling-2-Mini` `model-00001-of-00001.safetensors`: sha256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`, 19,526,019,168 bytes.
  - `mtp.safetensors`: sha256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`, 635,801,416 bytes.
  - `tokenizer.json`: sha256 `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6`.

## Later

- B1: price Metal ordering in one bounded dependency region (the serial path's barrier after every dispatch): its
  dispatch sequence against fusion or dependency-aware barriers, by uninstrumented wall time, separating GPU
  barriers, CPU encoding, submission and completion waits. Only a measured, recoverable remainder justifies a
  persistent-kernel experiment.
- A1: 4K acceptance as its own problem, scored in committed tokens per millisecond. Any retained-context or
  alternative-drafter experiment uses unfamiliar prompts with instructions in varied positions and counts the full
  cost of maintaining the context.
- M2: native MTP (parked `lane/m1-mtp`) against DFlash2 trees on the cheaper verifier, with the same target, prompts,
  output accounting and memory budget, counting drafter startup, steady drafting, lookup, verification, rejected
  work and state maintenance.
- megakernel-zoo's Mac engine as a comparator: its reported 32K configuration (27.7 GB active plus about 6 GB of
  MLX buffer cache) must be reconciled with the 28 GB process-tree cap before it runs here.
- J1: multi-agent scheduling (four-agent TTFT 998 ms against oMLX's 623 ms, `campaign/kernels/inkling-b2-baselines.md`).
- B2 remainder: comparator rows not yet run on the corrected workload.
- R1: the campaign report's reproduction pass.
- Splash `--verify-rows all` profile, for a deterministic Splash speculative comparison.
- ZL1: a bounded `baskingspot/ziglint` pilot on owned Zig source, with the exact 0.17.1 standard library and an
  explicit reviewed rule set. Prove that missing inputs, malformed configuration and a known violation fail,
  and that a clean fixture passes, before making the gate blocking. Preserve compile/link, kernel-contract and
  lean checks; triage findings without a house-style rewrite. Source pins and observed failure modes are in the
  [reuse assessment](../campaign/baskingspot.md#ziglint-validate-the-gate-before-enforcing-it).
- PS1: resolve the swallowed permission-setting failure in `zig/src/sessions/disk_namespace.zig:prepare`.
  The directory promises owner-only prompt-state storage, yet `setPermissions` failure is ignored. Add a failing
  permission-operation fixture, follow the existing initialization error path when privacy cannot be established,
  and verify descriptor/lock cleanup. This contract review is independent of adopting the linter; see the
  [inspected finding](../campaign/baskingspot.md#ziglint-validate-the-gate-before-enforcing-it).
- TK1: profile added-token matching on real coding prompts and common-prefix stress cases. If material, compare
  the existing trie with pinned `baskingspot/aho-corasick` runtime `Automaton`, preserving token IDs,
  leftmost-longest selection, normalized duplicates, strip flags and request masks. Reuse tokenizer parity;
  include construction cost, actual retained memory and served TTFT. Use existing benchmark machinery first,
  match both arms' module optimize modes, and retain the current trie unless request benefit and dependency
  limits justify adoption. The [contract comparison](../campaign/baskingspot.md#added-token-matching-the-strongest-runtime-lead)
  names the fallback cases. This stays behind V4/P6's measured GPU work.
