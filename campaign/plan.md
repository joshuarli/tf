# M5 Max native inference campaign: Flint, then Inkling 2 Mini

## Status

- **Owner steering (2026-11-22):** cap campaign context at **32,768 total tokens**, including reserved output;
  quick benchmark defaults are 1K/4K. Refuse oversized benchmark inputs before timing; preserve long fixtures
  and report quality exclusions as incomplete coverage. Defer 128K investigations. This changes campaign
  admission and server settings, not the model's runtime capability outside the campaign.
- **Nanoattention setup (2026-11-22):** source `6f782a8d` is already pinned and includes experimental Inkling
  on `main`. `nanoattention-setup`, `nanoattention-smoke` and `nanoattention-bench` compare its `nano bench`
  CLI with current TF on identical raw prompt IDs and the original checkpoint. Its BF16 arithmetic and
  failed upstream logit parity make it a performance investigation, not a qualified numerical oracle.
  Source preparation and protocol tests run on Linux; Mac build/smoke/M5 receipts are pending. See
  `comparators/nanoattention/README.md`. Use any measured advantage to select one bounded TF experiment.

- **Goal:** a faster, leaner native Zig/Metal TensorFold on this M5 Max. Flint came first; the focus is now a qualified Inkling 2 Mini port (see **Acceptance**).
- **Upstream frozen at:** `fa13df5dd1fdd23ada775510a464c9aa03a39e09` (TensorFold `main`, 2026-11-21). Work integrates on `main` (owner, 2026-11-21: `campaign/m5` was fast-forwarded into `main` and retired).
- **Last accepted revision:** `a1862ad55` on `main`.
- **Owner focus (2026-11-21):** Inkling 2 Mini. Flint work is limited to the shared-core guards (`bench-suite NAMES=flint-steps,flint-exact`, about 10 s) and the existing accepted results; the Flint optimization (O1) is deferred, not rejected.
- **Accepted:** C0 (fetch, host and build-baseline exit 0; second fetch downloads nothing); F1 (`--no-drafts` already disables copies and the drafter: `no_drafts_test.zig` in `test-flint-draft-host`; `make -C campaign smoke MODEL=flint MODE=ar` gives 512 tokens, 0 drafted, 1.00 tokens per decode round, about 169 tok/s); C1 (`make -C campaign bench MODEL=flint MODE=ar CELLS=1024 PAIRS=1` plus a byte-identical second `scorecard`; baseline ar at 1,024 tokens decodes at 153 tok/s with a 22.5 GiB peak); Q1 (`make -C campaign quality MODEL=flint SET=dev`: 12/12 on the baseline; 48 cases; libtfgrammar built from xgrammar v0.2.8 `97787376`); H1 (`make -C campaign pi-harness MODEL=flint`: 8/8 completed, 8/8 checkers passed, 0 tool-call parse errors); I0 (`make -C campaign ref-check MODEL=inkling2-mini`: the float32 reference matches llama.cpp `40b7a546` top-1 on every confident position of 4 histories, max logit error 1.4e-4, mean KL 3e-11; recipe at `docs/recipes/inkling-2-mini.md`); I1 (`zig build test-inkling2-loop-host`: 20/20 including config, tensor-map geometry, tokenizer/template against `campaign/ref/histories.json`, and the image and `--vision` refusals; native still builds). The registry edit and the image-part message are deferred to I2 and I3. C3: `campaign/ref/logit_gate.py` enforces shape, position count, nonempty evidence, finiteness and `gate_policy.json` bounds (`make -C campaign ref-gate-test`: 7 negative controls). Oracle outputs are reused from hashed receipts (ref-check takes 1 s instead of an hour), and the converter environment is pinned. C4: `bench-list`, `bench-check`, `bench-dry`, `bench-suite`, `bench-full` in `campaign/suite/`. F3: baseline ar and copy at 1K/4K/32K, six pairs over two sessions; anchor comparator rec `e53663b2` decodes 222.7/190.6/128.8 tok/s against TensorFold's 162.8/150.7/98.1 on the same fixed-length greedy protocol (comparator rates are client-timed and only three pairs; not a promotion claim). C2: pairing by declared identity, per-pair log-ratio intervals, holdout selection, refused `combined` mode, qualification apart from completion, owned Pi server (`make -C campaign evidence-tests`: 68 fixtures; A/A on one binary 1.001, 95% CI 0.975–1.028). F2: all 142 outlier rows are accumulation order with a named first divergent tensor, 0 defects; baseline mean KL 0.019, top-1 95.5%. Memory cap: `campaign/heavy.py` (Decision 8). **Red:** none beyond **Baseline**. **Parked:** O1 (no commits; a 1K Metal System Trace was taken but is scratch).
- **Inkling tranche accepted (2026-11-21):** I2 short-prompt acceptance (checkpoint loads with a 0.3 GB footprint since weights are file-backed; all 48 execution slots and greedy decode run; reset reproduces 25 rows bit for bit; the 14 short histories pass the f32-vs-f32 gate at max abs 1.55e-4). Open from I2: harbour fails the gate from position 557 (max abs 6.0, 6 confident disagreements) and cells_code from 304 (max 0.68, top-1 100%). Q2: bench, quality, Pi and smoke accept `MODEL=inkling2-mini` (`INKLING_LOAD_LIMIT_GB=26`, ar only). K1 (`campaign/kernels/fp4.md`): a table-decode expert matvec is 11-16x faster than the family's kernel at 0.62-0.99 of the 521 GB/s read ceiling. P1 (`campaign/kernels/inkling-profile.md`): decode is 105 ms a step at 83 tokens and 222 ms at 1,334 against a 10.7 ms weight floor; attention is 53% at 1,334 tokens, experts 64 ms, dense matvecs 22 ms.
- **Inkling native, merged on `main` (2026-11-21):** I3 minimal greedy serving (`make -C campaign smoke MODEL=inkling2-mini`: 512 tokens, 0 drafted, one token a round; served 16 greedy tokens equal the runner's); K2 table-decode expert kernels; A1 split-K decode attention; D1 long-history diagnosis (`campaign/parity/inkling2-mini/long-divergence.md`: harbour is router near-ties at 556/557, reproduced by forced-route reference runs; cells_code 304 is drift amplification; cells_code 656-694 is not yet diagnosed); R1 ring KV for the 40 local slots (served context 8,259 -> 44,435 tokens at `INKLING_LOAD_LIMIT_GB=26`); the parity fragment's variable clashes fixed.
- **Measurement scope:** the K2 timing (105 -> 47 ms at 83 tokens) and the A1 timing (222 -> 107 ms at 1,334 tokens) were each measured on its own branch, which still carried the other's old bottleneck; they are not additive. Integrated-main measurements so far are single short runs, repeatability unmeasured: 38.3 ms a step at 83 tokens (A1+K2) and 39.7/40.9 ms at 1,334 tokens (A1+K2+R1, R1's worker). The integrated parity run on A1+K2 reported 100% top-1 over 16 histories, but compare.py did not enforce the numerical contract (repaired by G1, below), so that run is diagnostic, not qualification.
- **Memory accounting:** the 0.3-0.8 GB process footprints quoted for Inkling omit the file-backed 19 GB checkpoint. `heavy.py` now charges each process the larger of footprint and resident size and stops on critical host pressure; the server's cache budget already reserves the checkpoint bytes.
- **Owner steering (2026-11-21, against `e2549d4a3`):** make the acceptance path trustworthy (gate wired into native parity, candidate provenance, model-keyed scorecard), finish the cells_code 656-694 witness, take a small integrated served baseline, then improve prefill (skip unused prompt logits, then bounded batched prefill) and re-profile combined decode for one bounded experiment. Scorecard model keys done in `3cd2ac36d`.
- **Owner prior-art steering (2026-11-21):** MLX-VLM `f1618d6d` (native-checkpoint Mini prefill, per-visit caches, loop banks, DSpark), oMLX `ade9f20a` (Metal compact-relative-bias prefill attention, SIMD router, grouped experts), MLX PRs #4781/#4572/#4712 (packed-format NAX GEMM, expert-boundary scheduling, small-row tactics), TokenSpeed `947b25bc` (DSpark taps capture the raw partial state entering a visit), macOS 27 FP8 TensorOps. Order: matched TF vs MLX-VLM AR at 512/1K (X1); skipped prompt logits and bounded multi-token prefill (B1; the first skip-logits variant was rejected for a CPU-shared-buffer race, parked on `lane/s1-skip-logits`); compact-bias multi-query attention and occupancy-selected expert kernels; in parallel an MXFP8 TensorOps probe (N1) and the DSpark tap contract (T1); short verification then native DSpark; down-plus-reduction fusion, SIMD router and prefix reuse only if the profile shows them. Every candidate: baseline, identity, correctness, paired end-to-end latency, peak memory, keep/revert.
- **Served baseline `inkling-native-v0`** (binary `819b83096a28`, main `3b02b563f`; one pair, unmeasured repeatability): code-1024 serves as 1,349 Inkling tokens; cold TTFT 53.4 s (prefill 25.3 tok/s), decode 24.9 tok/s, 512 tokens in 73.9 s; footprint 3.8 GB excluding the 19 GB mapped checkpoint; no EOS within 512 tokens; warm request reused 0 cached tokens.
- **Decode ideas from realugbun/llama.cpp `2eb2530` (inkling2-speed; owner, 2026-11-21), against main `318cdda7f`:** (1) normalize each completed depth block once when its snapshot is published (commit `7ec7714`) and keep normalized previous-pass bank entries for that pass, re-normalizing only the mutable current partial: a static count of about 380 -> 106 normalized rows a token, and most of the 380 row-copy dispatches that pack mix inputs; the cache must belong to a frozen snapshot, not a scratch address, and raw states stay for the DSpark taps. (2) Fuse the controller tail (two tanh plus the gated residual) into one kernel at the 96 controller sites, keeping M1's distributed projections and the existing `x - f*x` arithmetic order: 192 fewer dispatches a token. (3) Fuse convolution with its history store (192 sites), decode-only first, with read/write overlap rules for multi-token chunks; inspect the copies around the FFN convolution. (4) Its relative-bias fusion still materializes the full mask; compact bias stays the multi-query design. Test each fused operator against its unfused form on decode and prefill shapes, then a complete-request measurement.
- **Integrated Inkling, 2026-11-22 (gated with `gpu-gate-ready`, short screens, repeatability unmeasured):** M1 fused dense matvecs (decode 37.2 -> 27.4 ms at 83 tokens, 39.3 -> 29.4 at 1,334), R2 SIMD router (27.7 -> 23.1 ms at 83; logits byte-identical), C3 controller tail in i2_ctl (no measurable change, byte-identical, kept as a simplification), B1 chunked prefill (chunk 128; prompt 25 -> 120 tok/s in the runner). Served on code-1024 (1,349 tokens, 512 out, two pairs) against v0: TTFT 53.5 -> 13.4 s, prefill 25 -> 101 tok/s, decode 25.2 -> 35.7 tok/s, request 73.8 -> 27.7 s, output byte-identical to v0. Rejected: skipping prompt logits with unwaited passes (CPU-shared token buffer race; parked on `lane/s1-skip-logits`). Accounting: an Inkling run charges ~20.6 GB of the 28 GB cap once GPU-wired weights are counted.
- **First Inkling Pi task (`read-and-answer`):** protocol fixes for served-model attribution and greedy default temperature landed; the model then emits its native tool call (`<|message_model|>read<|content_invoke_tool_json|>{...}`) as text because the server has no Inkling reply parser (I4). No Inkling Pi or quality result exists yet; Flint's do not transfer.
- **Ceilings (inference-optimisation skill method, 2026-11-22):** measured 521 GB/s read and 56 TFLOPS fp16/bf16 GEMM (MLX 8192^3, contended host). Decode floor 10.7 ms a token (5.55 GB read) vs ~23-25 ms on main. The ~3,200 tok/s prefill figure assumes 50% of that half-precision GEMM rate and is not an attainable ceiling for the f32-activation configuration: the f32 simdgroup expert kernels run at ~4.5 TFLOPS and float-operand TensorOps experts gained only x1.018. An f32-arithmetic ceiling has not been measured.
- **Landed 2026-11-22:** G1 (native parity runs `logit_gate` per history with runner/model/token/policy receipts; on main `6b2614352`: 14 within contract, harbour and cells_code explained routing divergences, 0 unexplained, not qualified; `campaign/parity/inkling2-mini/report.md`). N1 (`campaign/kernels/mxfp8-tensorops.md`: TensorOps f16 x E4M3 is slower than the table kernel at 1 routed row, 3-20x faster at 8 and 64: a prefill/verify tool, not decode; it also fixed a bit-7 E8M0 mask in the standalone fp4 probe, not in the engine). T1 (`campaign/ref/dspark_taps.md`, `mtp.zig`: taps 5/23/41/48 are raw partials entering layer 5, layer 11 pass 1, layer 17 pass 2, and the final normalized state). I4 (Inkling reply parser: Pi `read-and-answer` made 3 tool calls with 0 parse errors; its checker failed on `backfill, ingest` vs `backfill,ingest`). `heavy.py` charges wired Metal buffers once (an MLX load was double-charged 29 GB). A Pi run now fails when its server exits.
- **Prefill, integrated on main (2026-11-22):** P3 split-timeline profile, then each candidate gated by fp64 family tests, the prefill-mode parity gate and base-candidate-base runner timing (busy-host limits 35%):
  - P4 tiled dense GEMM: x1.085 (95% 1.038-1.133).
  - E1 matrix-unit routed experts chosen by occupancy (MLX gather_qmm / expert-boundary scheduling as prior art): x1.417 (1.317-1.524).
  - P5 shared K/V tiles in prefill attention (oMLX ade9f20a tiled prefill attention as prior art): x1.196 (1.158-1.235).
  - Chunk 512 instead of 128 (more experts reach the matrix path, bigger GEMMs): x1.418 (1.391-1.445); +0.55 GB server scratch.
  - P7 dense GEMM through Metal 4 TensorOps, float x bfloat -> float (macOS 27 MPP matmul2d; N1's probe as the in-repo prior art): x1.100 (1.044-1.158) over 12 rounds.
  - Runner prefill at 1,139 tokens: ~120 -> ~382 tok/s. Rejected or parked: A2 decode attention, E2 low-occupancy experts (slower), P6 64-wide K staging (x1.035, inconclusive; `lane/p6-prefill-gemm2`), E3 TensorOps experts with float activations (x1.018; `lane/e3-expert-tensorops`), E3 split-half activations (1.7x slower). `lane/e2-prefill-experts2` keeps E2.
  - The parity target only exercised one-position decode; `RUNNER_MODE=prefill` now gates the prefill kernels (`report-prefill.md`: 14 within contract, 2 explained, 0 unexplained; also at `PREFILL_CHUNK=512`). It caught a tile-edge defect in P5's first candidate.
  - Remaining profile, one 512-row chunk at base 4096 before P7: experts 837 ms (matrix 699), dense GEMM 334, attention 270, of 1,577 ms. The experts run f32 simdgroup math at ~4.5 TFLOPS; the accelerator's fast paths need half activations, which the f32-activation parity policy excludes.
- **Comparator (X1b, `campaign/kernels/inkling-mlxvlm.md`, before chunk 512 and P7):** same prompt ids, 16 greedy tokens, three interleaved pairs, busy host. 512 tokens: native TTFT 2.28 s, prefill 225 tok/s, decode 40.5 tok/s, 20.4 GB; MLX-VLM 0.86 s, 592 tok/s, 35.1 tok/s, 25.5 GB. 1,024 tokens: native 4.10 s, 250, 39.6, 20.4 GB; MLX-VLM 3.78 s, 271, 32.7, 24.9 GB. Native decodes 15-20% faster in 5 GB less; MLX-VLM prefill was faster.
- **Decode drift (D3, `long-divergence.md` 'Decode drift'):** on a 1,172-token history decode departs from the reference far more than prefill because the two paths take opposite sides of a router near-tie (row 581, slot 38, 8th/9th gap ~1e-6; the reference sides with prefill), then the recurrent state carries the other expert's output. No kernel defect found; the runner now captures prefill rows and router picks.
- **Served, v0 against main `f6d8f83f0`** (`campaign/runs/2026-11-21/inkling2-mini-ar-1024-060122.jsonl`, two interleaved pairs, all trials valid at busy-host limits): code-1024 (1,349 tokens) TTFT 53.4 -> 5.66 s, 512-token request 73.9 -> 18.6 s, decode 24.8 -> 39.6 tok/s; prose and structured alike. Greedy ids match v0 on code and prose; structured diverges at generated token 48, where main prefers token 153966 over v0's 150161 by 0.09 logits (decode and prefill paths agree to 3e-4); the float32 reference on that 1,172-token context agrees with main (153966 at 26.9149 against 150161 at 26.8227), so v0's token was the error; main's prefill path matches the reference's last row to 1.5e-4, its one-position decode path to 1.5e-2.
- **Decode:** A2 (parallel attention scores) won 4% at 83 tokens and lost 5% at ~1.1K; reverted (`campaign/kernels/inkling-attention.md`). Decode is 23.4 ms a token at 83 tokens and ~25 ms at ~1.1K.
- **Quality and Pi on main:** Pi 7/8 checkers (read-and-answer fails on comma spacing), 0 parse errors. Quality dev 7/12: code 3/3, reasoning 3/3, retrieval 1/3 (one 8,192-token runaway, one prompt beyond the 44,435-token context at the 26 GB limit), structured 0/3 (no token grammar on the native Inkling path).
- **Measurement:** the desktop (ghostty, WindowServer) holds ~20% device utilization and ~25% accumulated GPU time, so the default gate never passes; timings use GPU_GATE_MAX_GPU_PCT=35 GPU_GATE_MAX_UTIL_PCT=35 with interleaving and are labelled busy-host. From 2026-11-22 the campaign gate is 60/60 (`campaign/bench/harness.py` CAMPAIGN_GPU_GATE): the owner uses this desktop during runs, so evidence rests on interleaved paired ratios with per-trial contention recorded, and needs more pairs where the ranges widen (USR-134 filed for the gate). The bench's in-trial check now reads the same setting. Only one GPU timing runs at a time: two lanes timing at once invalidated each other.
- **Host:** 48 GB; other work keeps swap at ~15 of 16 GB and has pushed pressure to critical several times; heavy runs wait for pressure level 1.
- **Workers:** Command Code and OpenCode Go are exhausted and Muse is intermittent; lanes run on Codex. Fallback happens only on `worker_status`.
- **Served, v0 against main `23793959f`** (`campaign/runs/2026-11-22/inkling2-mini-ar-1024-092408.jsonl`): code-1024 TTFT 53.3 -> 3.53 s (15x), request 73.8 -> 16.4 s, decode 24.8 -> 39.6 tok/s, server peak device bytes 3.8 -> 4.5 GB; prose 45.5 -> 3.1 s TTFT, structured 44.4 -> 2.96 s.
- **Verification (V1, V2, hybrid; `campaign/kernels/inkling-verify.md`):** merged. Copy-draft verification on the prefill path with exact rollback, opt-in via INKLING_COPY_DRAFTS=1 (`--no-drafts` overrides). Few-row prefill kernels cut the verify forward to 46/70/106/153 ms at 2/4/8/16 rows (from 123/131/153/153). Undrafted rounds now run ordinary decode and drafts are capped at 7: 512 greedy tokens code 18.0 s, prose 12.0 s, structured 8.1 s against decode 12.8/13.1/13.0 s, ids equal to decode on all three (single gated runs). Code still loses: each round carries snapshot, readback and replay overhead.
- **Quality, read by case (main `c5aa3aa7b`, 8/12):** code 3/3 and reasoning 3/3. structured 0/3 are refusals (no token grammar or constrained tool choice on the native Inkling path), not wrong answers. retrieval 2/3; retrieval-t05 needs more context than the cap (43,076 prompt tokens; campaign runs now serve 32,768). An earlier 7/12 run lost retrieval-t02 to an 8,192-token runaway. Pi (main `f6d8f83f0`-era build): 7/8 checkers, 0 parse errors; read-and-answer fails only on comma spacing. Both predate chunk 512, P7 and V1-V2 and need a rerun on the current build.
- **Campaign context cap:** campaign runs serve Inkling with `--context 32768` (campaign/bench/harness.py CONTEXT_CAP).
- **Owner steering 3 (2026-11-22):** progress accepted; next phase is cheap short verification and faster repeated agent turns, with a clearer numerical contract.
  - Primary lane V3: full-round verification cost at widths 1/2/4/8/16 from identical states (draft, save, target forward incl. head/argmax, restore/replay, host), then one measured bottleneck at a time (named candidates: `.none` replay, snapshot reuse/coalescing, GPU argmax, small-row head tiling, expert reuse at few rows). Metric: committed tokens / whole-round time against chunked prefill + ordinary decode (~39 tok/s), never one-row prefill. An 8-row DSpark round must take ~67 ms to reach 120 tok/s; at 153 ms for the verify forward alone no drafter can.
  - Second lane C1: measure Pi token-prefix overlap; if useful, one resident retained state for exact append-only continuations (consumed-token bookkeeping, speculative rows excluded, fresh fallback), validated against replay of the same schedule and characterized separately against fresh batched prefill.
  - Three separate questions, each with its own evidence: kernel/state correctness (operator and state tests); fidelity to the independent reference (teacher-forced comparisons, task quality); speculation against its chosen native target (verification, rollback, continuation). The current f32 configuration is not qualified (14 within contract, 2 explained, 0 unexplained) and stays visible as such. At most one separately named opt-in mixed-precision configuration, only if the verification profile shows it addresses a dominant cost, with quality criteria declared before its results.
  - DSpark integration only once a whole speculative round can beat ordinary decode at realistic acceptance.
- **Speculative target, named:** verified rows run the chunked prefill kernels and undrafted rows the decode kernels; the control is chunked prefill + ordinary decode. Matching ids on three prompts is not a greedy-equivalence claim; differences are classified against near-tie witnesses (`long-divergence.md`).
- **Accepted (steering 3, second lane):** prefix reuse (C1 measurement + C2 implementation, `campaign/kernels/inkling-prefix.md`). All 21 Pi continuation requests are exact appends of prompt + emitted tokens. The native host keeps one resident state, reuses it only for a strict exact prefix, and otherwise falls back to fresh prefill. It is on by default; `INKLING_PREFIX_REUSE=0` turns it off. Correctness: retained-state logits are bit-identical (max_abs 0) to a replay of the same schedule. Fidelity: against a fresh full prefill, max_abs 0.72 and KL 2.6e-13, with the same greedy token. Paired Pi runs on a busy host (off/on/off) took 329.6/188.9/324.7 s, 8/8 completed, 7/8 checkers each time, with continuation-turn prefill down from 5–10 s to 0.1–1.2 s and the same memory. Harness repairs: Pi's output budget now fits the served context, and a task whose last turn errored is not counted completed (13f49d8a2). The parity report now regenerates after each capture. Prefill parity at chunk 512: 14 within contract, 2 explained, 0 unexplained.
- **Accepted (comparator):** nanoattention `6f782a8d` (N2, N3; `campaign/kernels/inkling-nanoattention.md`). With its optimized knobs it prefills at 629–650 tok/s against TensorFold's 392–414, and decode is a tie (42.2 against 41.1 tok/s at 1K). Its reference defaults run at 34 tok/s prefill. nanoattention rounds activations to bf16 unconditionally. A leave-one-out ablation puts its prefill lead in bf16/fp16 matrix operands: the expert block fragments, the bf16 prefill matrix and the fp16 tiled prefill attention. None of these ports under the f32 contract. The f32-compatible ideas measured small: concurrent dispatch about 3%, plus expert-tile striping with tail tiles and four-output reuse. Those are queued as ordinary candidates after V3.
- **Declared before results: the one mixed-precision experiment, M1 "bf16-operand prefill".** It is opt-in behind `INKLING_PREFILL_BF16=1`, off by default, and the f32 gate and its policy string stay unchanged. Prefill activations are cast to bf16 only as matrix operands into the dense GEMM and expert matrix kernels, with f32 accumulation and f32 residual stream, norms, router and KV. Attention stays f32. Keep criteria, all required:
  1. Paired prefill at 1K, control-candidate-control, ≥ ×1.20.
  2. On the 16 parity histories, under its own policy string `bf16-operand-prefill`: confident-position top-1 agreement with the f32 reference ≥ 99.5%, and no new unexplained divergence class.
  3. Greedy ids identical to the f32 path on the three 512-token bench prompts for the first 64 tokens, or the first divergence explained by a near-tie (reference top-2 margin < 0.1).
  4. Quality dev set no worse than 8/12, and Pi checkers no worse than 7/8.
  5. Peak memory within +0.5 GB.
  Otherwise revert and record. It runs after V3 merges, because it edits V3's files.
- **M1 result: reverted** (`campaign/kernels/inkling-bf16-prefill.md`). Criterion 2 fails: confident top-1 is 96.37%, against the required 99.5%. Criterion 3 fails: structured greedy ids diverge at token 56, at margin 0.21. The code is parked on `lane/m1-bf16-prefill`. No mixed-precision configuration remains in this campaign.
- **Owner steering 4 (2026-11-22): the prefill target is at least 1,000 tok/s on this host.** The owner approved a second named precision experiment, M2 "split-bf16 prefill", gated first by a microbenchmark. The criteria below were declared before any result.
  - **Gate G0 (microbenchmark, `campaign/kernels/split-bf16-tensorops.md`).** Run at Inkling expert shapes: K 2304 and 1152, N 2304, 8/32/64/128 routed rows to one expert, NVFP4 and MXFP8 weights converted exactly to bf16. Arms: the current f32 simdgroup expert kernel, float×bf16 TensorOps, bf16×bf16→f32 TensorOps with one term, and the same with 2 and 3 split-activation terms (x = hi + mid + lo, each bf16). Record median GPU time and error against the fp64 oracle. Continue to M2 only if the 3-term split, or the 2-term split if its error fits the f32-vs-f32 bounds, beats the float×bf16 TensorOps arm by at least ×1.5 at 32 and 64 rows. The threshold must hold for every combination of K (2304, 1152) and weight format (NVFP4, MXFP8) at both row counts; one qualifying combination is not enough. This clarification was made before any result.
  - **M2 keep criteria, all required.** The split stays within the f32 contract, so it can become the default rather than an opt-in:
    1. Prefill parity at chunk 512 under the f32-vs-f32 contract, with policy string `f32-activations+split-bf16-prefill-512`, gives the same classification: 14 within contract, the same 2 explained divergences, 0 unexplained.
    2. Greedy ids are identical to the f32 path for the first 64 tokens on the three 512-token bench prompts.
    3. Paired prefill at 1K and 4K, control-candidate-control, is at least ×1.20.
    4. Peak memory is within +0.5 GB.
    5. Pi checkers are no worse than 7/8.
  - If G0 fails, record the measured ceiling. Then 1,000 tok/s needs a looser named contract, which is the owner's decision, and the f32 path continues with ordinary candidates only.
- **Owner steering 5 (2026-11-22): an explicitly experimental fast-precision lane.** The f32 path remains the default and the correctness oracle. No contract is weakened, and the checkpoint is never requantized.
  - **Tiers.** Exploratory candidates use 512–1,024-token inputs and 32–64-token generations, plus confident top-1 and greedy-id comparison against the f32 path. Only a promising candidate gets promotion evidence: the f32 parity oracle under its own policy string, the quality dev set, Pi, and 4K/32K contexts.
  - **F1, selective fp16-operand expert prefill.** Use half matrix operands in the routed-expert products only, with f32 accumulation. Routing, normalization, residuals, attention and KV stay f32. Measure sensitivity per physical layer and per operation (gate/up against down), and convert only what tolerates it. fp16 carries 10 mantissa bits to bf16's 7, which is the reason to retry after M1. Values beyond half range need an explicit per-block handling, and the count of those values must be reported.
  - **F2, W8A8 TensorOps experts.** MXFP8 experts use their own E4M3 bytes as E4M3×E4M3, or half×E4M3, with E8M0 scales. NVFP4 experts use half×fp4_e2m1 with their block scales. The measurement includes repacking, activation quantization, scale application and dispatch, and it compares full prefill time, not GEMM time alone.
  - **Order.** G0, extended with the fp16 and fp8 arms, gives the exploratory GEMM ceilings. F1 and F2 start after V4 because they share prefill.zig and prefill.metal, and they run serially.
  - **GPU contention.** The campaign gate is 60/60 (`campaign/bench/harness.py` CAMPAIGN_GPU_GATE).
- **Owner ruling (2026-11-22): global tool caches are allowed.** Writes to `~/.cache` and similar caches (uv, pip, Hugging Face, XDG) are permitted, and the earlier "never write ~/.cache or global caches" rule is lifted. `~/.pi`, global tool installs and system settings stay out of bounds.
- **E4 result: reverted** (`campaign/kernels/inkling-prefill.md`, staged expert section). Staging is exact: 24.46B weights checked, zero mismatches, parity 14/2/0. It is slower end to end, with paired median ratios of 0.879 at 1K and 0.893 at 4K. Expert decode runs at only 225 GB/s for NVFP4 and 322 GB/s for MXFP8. Per chunk, staging takes 321 ms, TensorOps 356 ms and gather/scatter 150 ms, which together exceed the current kernel. With f32 activations, staged TensorOps cannot beat the fused kernel. H1 has to decode weights inside the half-precision matmul, as nanoattention does, rather than stage them.
- **Owner steering 6 (2026-11-22): trade accuracy for large speed and throughput, and hold a three-way shootout.** The owner accepts lower fidelity to the f32 reference in exchange for much higher performance. The f32 path stays as the oracle. A new TensorFold **fast profile** is judged on task quality and Pi, and its agreement with the f32 oracle is reported but does not gate. This supersedes steering 5's "f32 remains the default" for the fast profile. Whether the fast profile becomes the shipped default is decided after the shootout.
  - **Shootout protocol, declared before any result.**
    - **Engines:** TensorFold; oMLX 0.8.3 (`jundot/omlx` `e72868da`, MLX 0.32.4); nanoattention `6f782a8d` (still `main` on 2026-11-22).
    - **Host and order:** one host, the M5 Max. One engine resident at a time under `heavy.py` and the 60/60 gate, with interleaved alternating pairs (at least 3 per cell), labelled busy-host.
    - **Prompts:** identical token ids for the code, prose and structured prompts at 1K, 4K and 16K. 16K is the largest that fits the 32K cap.
    - **Decoding:** greedy at temperature 0, 512 generated tokens.
    - **Configurations:**
      - (a) each engine as shipped at its best: oMLX `Inkling-2-Mini-oQ4e-mtp` with Lightning MTP on (default depth 8) and off; nanoattention with its optimized knobs; TensorFold default and fast profile, each with and without speculation;
      - (b) matched weights: the original checkpoint on every engine that loads it, and oQ4e on oMLX and on TensorFold once TensorFold loads it.
    - **Metrics:** TTFT and prefill tok/s; decode tok/s as committed tokens over decode wall time; peak charged memory.
    - **Fidelity:** teacher-forced agreement with the f32 oracle on the 16 parity histories (confident top-1 and mean KL), and each engine's first greedy divergence from the f32 path.
    - **Served engines** (TensorFold and oMLX) also run the quality dev set and Pi.
    - **Claims:** no "fastest" claim from isolated kernels or from a single cell.
  - **What TensorFold must add to be comparable, as serial lanes after E4:**
    - **H1, fast-profile prefill and decode:** half-precision matrix operands with f32 accumulation in dense, experts and attention, following nanoattention's ablated recipe and building on E4's staged weights.
    - **H2, cheap speculation:** a bf16 DSpark drafter round, plus copy drafts up to 15 tokens.
    - **H3, an oQ4e loader:** affine 4-bit experts for layers 20–23, and affine 8-bit attention, embeddings and head.
  - **Fast-profile gate, declared before results:**
    - quality dev at least 8 of the scorable cases (the f32 path scores 8/12 with retrieval-t05 beyond the cap), with no new refusal class;
    - Pi checkers at least 7/8;
    - no non-finite logits on the 16 histories;
    - a paired speed win over the f32 default at 1K and 4K.
  - **Report owed after O1 and N5:** how TensorFold compares with both engines per cell, how much accuracy each configuration loses against the f32 oracle, and the recommended next steps. The owner expects to accept accuracy loss for much higher speed and lower memory overhead, and calls 32K context too small for coding agents, where 256K is practical. The standing 32,768-token campaign cap still holds until the owner changes it. Every engine therefore also reports KV and state bytes per context token and its measured peak memory at 4K and 16K, so the report can project 128K and 256K against the 32 GB machine cap.
  - **Concurrency (owner, 2026-11-22).** oMLX serves concurrent requests. TensorFold's Inkling 2 Mini host serves one job at a time from a FIFO queue with one resident prefix state (`zig/src/native/inkling2_loop_host.zig`), so concurrent agents queue and evict each other's prefix. Other TensorFold families already have batched hosts. The shootout therefore adds concurrency cells: aggregate and per-agent output tok/s, TTFT and queueing delay at 1, 2, 4 and 8 concurrent multi-turn agents, for TensorFold (lane S1) and for oMLX. **B1, "concurrent Inkling sessions"**, is the next engine lane after H1, because both edit the Inkling family files. It adds several resident session states, each with its own exact-prefix reuse, plus batched decode across sessions.
  - **S1 result (2026-11-22, `campaign/kernels/inkling-subagents.md`).** Multi-turn agents (115-function system prompt, about 2.6K prompt tokens, four turns, 256 output) give about 10 aggregate output tok/s at 1, 2, 4 and 8 agents; per-agent rate falls as 1/N and median queueing reaches 68 s at 8. One agent is optimal; concurrency buys nothing on the current host. `--parallel 2` and `4` start but still serialize. **All 120 sweep requests reused zero prompt tokens, even at one agent**: reuse needs the whole resident sequence to be a strict prefix of the next prompt, and the template drops generated thinking from earlier assistant turns, so the next render diverges. The fx run (no hidden thinking in history) reused 64.5K of 102K prompt tokens. B1 therefore also needs prefix reuse that survives dropped thinking: keep a snapshot of the state at the end of each prompt, before generation, so the next turn extends that snapshot instead of the generated sequence. Truncating the resident KV to a shorter common prefix does not work as is, because the 1,024-token sliding-window layers have already overwritten the older positions. Without it, every turn of a thinking agent repays full prefill.
  - **O2 result (2026-11-22, `campaign/kernels/inkling-omlx-concurrency.md`): oMLX on the same multi-turn agent workload** (oQ4e checkpoint, MTP off, `--memory-guard-gb 26`, four server slots, two reps, post-run gate passed). Aggregate output tok/s at 1, 2, 4 and 8 agents: 26–45, 55–59, 73–77 and 70, against TensorFold's flat 10. Median TTFT is 0.2–1.2 s up to four agents, against TensorFold's 6–9 s per turn plus queueing. 119 of 120 requests hit oMLX's prefix cache, against TensorFold's zero. With the default eight slots, oMLX's own 26 GB guard did not hold the 28 GB cap at eight agents (28.9 GB charged), so the eight-agent cell is eight clients on four slots. Peak charge with four slots was 25.7 GB. The standard `bench_concurrent` run failed its post-run gate (host busy), so its timings are discarded. Batched outputs differ from solo outputs in text, and the server gives no token hashes. **The gap to the goal is about 7× at four agents**, and it splits into three TensorFold deficits: no prefix reuse across turns (about 6.5 s of prefill per turn), no batched decode, and f32 activations (about 41 against 55 tok/s single-stream).
  - **P1 result (2026-11-23, `campaign/kernels/inkling-prefix-store.md`): durable prefix store.** The server keeps standalone prompt states within `--prompt-cache-gib`: at a fresh prompt's first fork point, at each prompt's last token, and at every fork point used. Admission takes the longest of own-slot reuse, a live fork and a store entry. A restored state equals direct prefill byte for byte, replies equal the reuse-off run, and concurrent replies equal solo. After dropping a redundant full-prompt insert, oQ4e agent sweep reps 1 and 2: 1 agent 20.8 / 36.0 tok/s (was 20.3 / 20.2; oMLX 26.0 / 45.2), 4 agents 39.0 / 47.6 (oMLX 72.7 / 77.4). Warm turn 1 reuses 2,571 of 2,572 tokens at 1 agent and 2,432 per agent at 4. Peak charged 23.4 GB.
  - **Same-weights agent sweep (2026-11-23): TF on oQ4e after H5, against oMLX 0.8.3 on oQ4e (O2).** Same multi-turn workload, 4,096 context per session, gated. Aggregate tok/s, reps 1 and 2: 1 agent TF 20.3 / 20.2 vs oMLX 26.0 / 45.2; 2 agents TF 29.4 / 28.5 vs oMLX 54.5 / 59.4. The 4-agent TF run (39.0 tok/s, oMLX 72.7 / 77.4) failed its post-run gate on another session's CPU load and is unqualified. Peak charged 18.5–21.4 GB. At 1 agent, TF's 18.3 s splits into a 7.7 s turn-1 prefill of 2,572 tokens (333 tok/s) and about 10 s of decode at about 37 tok/s; reused turns have a 0.32 s server TTFT. oMLX's second repetition skips the turn-1 prefill through its prefix cache, while TF re-prefills it because no live slot holds that prefix. Lanes opened: H4 (decode speed: profile, 16-bit KV, dispatch fusion; target 50 tok/s) and P1 (durable prefix store in the server), on disjoint files.
  - **H5 result (2026-11-23, `campaign/kernels/inkling-oq4e-fast.md`): fast affine kernels; oQ4e now matches or beats the native checkpoint.** Packed-word vector dot products for decode, 64-column by 32-row weight-reusing tiles for prefill and the batch step, and expert-grouped tiles for the affine 4-bit layers. Three gated pairs, native vs oQ4e: 1K prefill 223 vs 329 tok/s, 1K decode 39.8 vs 41.8, 4K prefill 346 vs 345, 4K decode 37.0 vs 38.7. oQ4e batchbench at K = 1, 2, 4, 8: 40.5, 57.8, 69.0, 76.6 tok/s, up from 24, 31, 36, 39 (native: 34, 51, 64, 72). Fidelity against the saved f32 oracle captures: top-1 92.04%, mean KL 0.039. All 71 GPU tests pass at a 23.1 GB peak. Remaining raw gap to oMLX on the same weights: single-stream decode about 42 vs 55 tok/s and prefill about 330 vs 934 tok/s.
  - **H3 result (2026-11-23, `campaign/kernels/inkling-oq4e.md`): TF runs oMLX's oQ4e weights.** The family loads the four-shard oQ4e checkpoint: affine 8-bit attention, embed and head, NVFP4 experts in layers 0–19 and affine 4-bit experts in layers 20–23. The affine kernels match float64 dequantized references, and all 70 Inkling GPU tests pass on both checkpoints. Against the f32 oracle, oQ4e top-1 agreement is 91.64% (93.68% where the f32 margin is at least 0.1), with mean KL 0.039; this is the accuracy cost of oMLX's weights. The first 45 greedy IDs equal oMLX's. **oQ4e is slower than the native checkpoint**: prefill 79 vs 222 tok/s at 1K and 73 vs 328 at 4K, decode 25 vs 41 tok/s, batch 24/31/36/39 tok/s at K = 1, 2, 4, 8. The affine kernels are first cuts: they unpack each weight element with integer division, and prefill and the batch step reread every weight matrix once per row. Next is H5, fast affine kernels (a vectorized matrix-vector kernel for decode, a weight-reusing matrix kernel for prefill and batch), then H4 (16-bit KV and decode).
  - **B1e result (2026-11-23, `campaign/kernels/inkling-fork.md`): cross-session prefix forks.** Each session keeps conv-ring snapshots and the overwritten local-ring rows at aligned points near its prompt end (`INKLING_FORK_WINDOW` 256, `INKLING_FORK_INTERVAL` 128). A new request copies the best other session's state at the largest saved point within its common prefix. The copy is byte-identical to direct prefill, continuation logits match, and the source is unchanged. On turn 1, agents fork 2,432 shared tokens. Multi-turn aggregate at 2 and 4 agents rises to 30.6–31.8 and 36.0–40.5 tok/s, from 25.6 and 27.8. Four-agent turn-1 client TTFT falls from 28–31 s to 12.6–17.9 s. This runs at 3072 context per session with 256 rollback rows, which fits the budget at 25.98 of 26 GB. Remaining gap to oMLX, at 1 agent: prefill 430 vs 934 tok/s and decode 34 vs about 55 tok/s, so raw speed, which H3 and 16-bit arithmetic address.
  - **B1d result (2026-11-23, `campaign/kernels/inkling-batch-serve.md`): continuous batching in the server.** All decoding sessions advance in one batch step, with up to eight sessions. Rollback is bounded by `INKLING_ROLLBACK_ROWS` (default 512); a longer output clears that session's prefix. On the standard short-prompt bench, aggregate tok/s at 1, 2 and 4 streams are 37, 56–57 and 72–74, and all 28 replies equal their solo outputs. Multi-turn agent workload at 1, 2 and 4 agents: 21, 25.6 and 27.8 tok/s, against oMLX's 26–45, 55–59 and 73–77. The server log locates the gap. Reused turns have a server TTFT of 0.27 s, and decode runs at 34 tok/s single-stream. **The 4-agent turn-1 TTFT is 28–31 s**: four separate 2.6K-token prefills at about 430 tok/s. oMLX gets about 1 s because agents share the system-prompt prefix and its prefill runs at about 930 tok/s. Remaining gaps, largest first: cross-session prefix sharing (B1e), prefill speed (H3 weights plus 16-bit), single-stream decode (H3 plus 16-bit decode), and full-prefill admission blocking the batch.
  - **B1ab result (2026-11-23, `campaign/kernels/inkling-sessions.md`): resident sessions with prompt-end rollback.** `--parallel N` keeps N session caches, and `--context` is per session. One shared rollback buffer restores each session's local and conv rings to its prompt-end state after decoding. Requests go to the session with the longest prompt prefix, falling back to the least recently used. On S1's workload, turns 2–4 reuse 2,572–2,673 tokens for every agent at 1, 2 and 4 sessions; S1 reused 0. Aggregate output rises from about 10 to about 22 tok/s. One-agent TTFT on later turns falls from about 8 s to about 2 s, mostly hidden thinking. Rollback is byte-exact; a synthetic continuation after rollback equals fresh prefill with max diff 0. Decode is still serial; B1d wires in B1c's batch step.
  - **B1c result (2026-11-23, `campaign/kernels/inkling-batch.md`): batched decode across independent session caches.** The first version runs the weight-reading work (projections, routing, experts, head) once over K rows and dispatches conv, KV store and split-K attention per row against each row's cache, in one command buffer, at f32. One step per row matches `position` within 1.5e-5, with the same argmax on every row at K = 1, 2 and 4. Five gated pairs per width give aggregate decode-step rates of 34, 51, 64 and 72 tok/s at K = 1, 2, 4 and 8, against 38–40 tok/s serial. oMLX served 26–45, 55–59, 73–77 and 70 (O2). Eight 4K caches fit at 25.4 GB charged. **Decision signal: GO.** TF's architecture reaches oMLX's concurrent decode rates at full precision with an unoptimized batch step. Still to do: server integration (continuous batching over B1ab's session slots), then the served rerun of O2's workload. Headroom: per-row dispatch overhead (29.5 ms batched against 25.6 ms decode at K = 1), half-precision decode, and fp16 KV for more sessions.
  - **Owner goal (2026-11-22), after the shootout benchmarks land: TensorFold matches or beats oMLX on concurrent agent workloads.** The measure is aggregate and per-agent output tok/s, TTFT and prefix reuse at 1, 2, 4 and 8 concurrent agents, taken with the project's `tools/bench_concurrent.py` against both servers. B1 builds on the existing multi-stream core (`zig/src/core/lanes/`, already used by the Flash Next, GLM and Qwen 27B hosts) rather than a new scheduler.
  - **Upstream activity (fetched 2026-11-22; 698 commits since our base `fa13df5d`, release 1.3.4).**
    - **A CUDA-only `inkling2` family is in progress** (`zig/src/families/inkling2`, last commit 2026-11-20). It has config, tokenizer and template, a weight map and layer plan, and CUDA attention, glue and ops kernels registered under the kernel contract (`zig/src/core/contract/inkling2_*_records.zig`). It has no Metal path and no served forward yet. It does not overlap this campaign's Metal `inkling2_loop`, but its contract records are a template for our registration gap.
    - **DeepSeek V4.1 Metal DSpark work, the most transferable.** The drafter's chain and its Markov bias run on the GPU behind the head, the head runs on its own FP8 copy, proposals go through one concurrent encoder, the next pass is queued right behind verify, and decode MoE runs in four waves. Each item targets the drafter round cost that made our DSpark a NO-GO (16.7 ms at width 4).
    - **Lanes.** Lone-stream copy pricing, MTP drafting across streams, and batched DFlash2 absorb across a round's streams (CUDA).
  - **Rebased onto upstream 1.3.4 (`bdcb6e608`) on 2026-11-22**, preserving merges. The 263 campaign commits plus one merge fix (`39a1958cc`, in `zig/src/server/chat.zig`) are on top, and the old main is kept as `backup/main-pre-upstream-rebase`. Checks:
    - The native build, the Inkling runner, the Inkling host tests and the 128 campaign tests pass.
    - `zig build test` passes except for five failures that unmodified upstream shows identically under the campaign's Zig 0.17.1 pin (upstream requires 0.17.0): `persistence_identity` (tokenizer and template prefix hash), both `vision_config parses to the defaults`, `pdf_worker` signal naming, and the "links its own memcpy" check.
    - Lanes started on the old main are integrated by cherry-pick.
  - **Done (2026-11-22, at the owner's request, with H1, N5 and O2 running): the checkout is `~/repos/tf`**, matching origin `joshuarli/tf`. `~/repos/tensorfold-m5` is a symlink to it, so old absolute paths and the running lanes still resolve. Worktrees were repaired and `~/.local/bin/fx-local-inkling` re-pointed. The rename broke the bridge's session lookup: it compared the recorded cwd with the real path. That is fixed in `~/usr` `404e7d97`. Pending: replace the hardcoded `/Users/josh/repos/tensorfold-m5` paths in `campaign/comparators/{mlx-vlm,nanoattention,omlx}/` and `campaign/kernels/inkling-depth-norm.md` with repository-relative ones after N5 and O2 land, because both lanes edit those directories.
  - **Comparator lanes:** O1 sets up oMLX 0.8.3 (source under `.work/src/`, the model under `~/models/` through `campaign/sources.lock.json`). N5 reruns nanoattention under this protocol.
- **Known gap: kernel-contract registration.** `docs/kernel-contract.md` requires every GPU kernel to register before it lands, and none of the Inkling 2 Mini `i2p_*` kernels are registered, including the accepted P7, V3 and V4 kernels. Each candidate is proved by its own family tests and the parity gate. Registration is required before the family is promoted out of this campaign and is tracked as one family-wide item, not per lane.
- **Accepted V4: parallel route selection** (`campaign/kernels/inkling-verify.md`, at main `6cd318428`). Route selection had a fixed cost of about 5.1 ms per forward on a single lane; it now takes 0.47 ms. Five alternating pairs give a width-4 verify round of 59.6 ms against 55.0 ms, busy-host. The logits are byte-identical on the 1,139-token prose fixture, prefill parity is unchanged at 14/2/0 at chunks 4 and 512, and the family tests pass. The round is still above the 45.4 ms that would revive DSpark. The same kernel runs in every prefill chunk.
- **Colab: built and parked** (`campaign/colab/README.md`, "Verdict"). The budgeted harness works: one T4 smoke used under 0.01 CU, with driver 580.82 and CUDA 13.0. The Inkling campaign does not need it. Its bottlenecks are Metal-specific, its numerical studies run cheaper on the Mac's CPU, and full-model checks need either the 104 GB f32 oracle or the native Metal engine. Revisit it for large CUDA-portable sweeps or for TensorFold CUDA work.
- **G0 result: fail, and a different lead found** (`campaign/kernels/split-bf16-tensorops.md`). The best qualifying split, two terms, fits sumBound everywhere but reaches only ×1.25 over float×bf16 TensorOps, against the required ×1.5, so M2 does not start. In the same microbenchmark, float×bf16 TensorOps reading **preconverted** bf16 weights beats the engine's current expert kernel by ×3.5 to ×5.7 at 32 to 128 rows. For example, NVFP4 at K 2304 with 64 rows takes 58 µs against 203 µs. Decoding the packed weights inside the matmul, the cooperative form that requires one SIMD group, is slower than the engine kernel. That explains E3's ×1.018: the cost is weight decode, not the matmul. P7 already accepted float×bf16 TensorOps for the dense GEMMs within the f32 contract, so this is not a precision change.
  - **Next prefill candidate E4, "staged bf16 experts".** For each prefill chunk, decode each active expert's NVFP4 or MXFP8 weights into a bounded bf16 scratch, then run the float×bf16 TensorOps expert product. Every weight converts exactly. The measurement must include the decode, scratch traffic and scheduling, judged on full prefill time. The scratch stays bounded and small rather than a 3.5× copy of the checkpoint. Criteria are those of an ordinary candidate: parity unchanged at 14/2/0 under the f32 contract, paired prefill at 1K and 4K, and memory. E4 starts after V4, because they share prefill.zig and prefill.metal.
  - **Exploratory arms, which do not gate.** half×half is ×1.1 to ×2 faster than float×bf16, with about 100× larger random-input error that is still inside sumBound. The untuned half×E4M3, half×fp4 and E4M3×E4M3 arms are slower than preconverted float×bf16. E4M3 activation quantization fails sumBound badly. F1 (fp16) stays a candidate after E4; F2 (W8A8) is deprioritized until a tuned kernel shows a gain.
- **DS1 result: native drafter exact, DSpark NO-GO** (`campaign/kernels/inkling-dspark.md`, "Verdict"). Against the TF32-off MLX fixture, all greedy proposals match on 65 blocks, with logit error 1.1e-4. The drafter takes 16.7 ms at width 4 and 29.7 ms at width 8. Width 4 then projects 40.5, 35.7 and 39.4 tok/s, which beats neither ar nor served copy drafts. DS2 is not started. The code is parked on `lane/ds1-dspark-drafter`. Revisit when the four-row verify round falls to 45 ms or less.
- **Parked:** `lane/ds1-dspark-drafter` (NO-GO), `lane/m1-bf16-prefill` (reverted), `lane/s1-skip-logits` (race), `lane/p6-prefill-gemm2`, `lane/e2-prefill-experts2`, `lane/e3-expert-tensorops` (see Prefill), `lane/d2-depth-norm` (slower), `lane/c0-ground` (superseded), `.work/wt/k2-inkling-experts` (untracked scratch).

## Charter

- Execute until **Acceptance** is met or every remaining item is parked behind a hard stop. Never stop after writing a plan. Before ending a session, rewrite **Status**, including any lane paths and the first command to resume. Resume by reading **Status**, checking that `campaign/host.json` still matches the machine, and taking the next item.
- Decide and keep going. Record a decision in **Decisions** only when a later slice depends on it.
- The execution grant: everything listed under **Execution boundary**. Park any item that needs more than that.
- Hard stops, each of which parks the item:
  - an action that is irreversible or acts outside this repo, `~/models` and `.work/`;
  - an effect that **Execution boundary** forbids;
  - a loop that cannot check anything.
  
  Pushing is always forbidden.
- Failing items: find the cause before retrying (`root-cause`). After three attempts at the same item, or three failed variants of one optimization idea, park the item. Park it on its branch with the cause and a retry condition, then go to the next ready item.
- Fix egregious problems in their own commit when you meet them. Egregious means any of:
  - a crash or hang over 60 s;
  - resident memory growing more than 5% across 10 identical requests;
  - a warm no-op `zig build native` over 60 s;
  - a campaign script over 2× its recorded time with unchanged inputs.
  
  Record anything smaller in the ledger and leave it.
- Per-model dependencies, not phase barriers (owner steering, 2026-11-21). Each model advances on its own chain (see **Sequence**). Profiling is diagnostic and may run while parity investigations continue. Promoting changed arithmetic waits for that model's qualified correctness contract. Speculation is a separate branch after the model's target path is qualified, and comparators run alongside engineering under the GPU lock; neither blocks autoregressive optimization. Broad comparator coverage is needed only before a competitive claim.
- **Lanes:** parallel for the disjoint sets named in **Sequence**, serial otherwise. Each lane works in `.work/wt/<lane>` on branch `lane/<id>`, following `orchestrate`'s worktree lifecycle; never `/tmp`. Set `TMPDIR=$PWD/.work/tmp`. The parent integrates into `main` and runs the parent steering checkpoint (`orchestrate` skill, `references/campaigns.md`, section "Parent steering checkpoint") after each accepted slice.
- **GPU lock:** take it for any timed run, any process that loads a full model, and the Metal device suite. Use `lockf ~/repos/tensorfold-m5/.work/gpu.lock <command>`. No compiles inside a timed window. Stop only processes the campaign started.
- **Commits:** Conventional Commits, scoped to the workstream (`campaign`, `flint`, `inkling2-loop`, `metal`). Put the item ID in the body. End every commit with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Code comments never cite this doc.
- **TensorFold rules:** follow `docs/recipes/SKILL.md`, `docs/kernel-contract.md` (for GPU work) and `docs/recipes/adding-a-zig-family.md` (for I-items). Skip their ATC, PR and publication steps; see **Execution boundary**.

## Decisions

The owner made the first two decisions on 2026-11-21:

1. **Owner: layout.** One repo at `~/repos/tensorfold-m5`, a TensorFold clone with `upstream` as a fetch-only remote (push URL `DISABLED`). Campaign tooling and records live in `campaign/`. The original draft lived at `~/d/infer/plan.md`; this file supersedes it.
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
   - Comparator weights, fetched only when their item runs.

   The fetch total stays under 120 GB; 621 GiB is free. Never delete files you did not create. Only one full model is resident at a time.
8. **Prescribed: memory.** The machine has 48 GiB of RAM. C0 records the Metal device's `recommendedMaxWorkingSetSize` (W).
   - Owner cap (2026-11-21): the campaign as a whole stays under 32 GB so the host stays usable for other work. Every command that loads a model or uses more than about 2 GB runs through `campaign/heavy.py`, which takes the GPU lock and stops the process tree past 28 GB; bare `lockf` is not used. The f32 GGUF oracle cannot run under this cap and is reached only through its receipts.
   - Every benchmark server sets `FLINT_LOAD_LIMIT_GB=27` (`heavy.FLINT_LOAD_LIMIT_GB`; or the Inkling equivalent),
     an explicit `--prompt-cache-gib` and `--context 32768`. The old memory-derived Flint limit was 56,192
     tokens; the owner's total-context cap now takes precedence. Runs recorded before the memory cap used
     36 and pair only with runs at the same memory and context limits.
   - Prompt cache is 0 for cold-prefill cells and 2 otherwise.
   - Speculation may add at most 4 GiB of draft residency, which is min(4 GiB, 10% of RAM).
   - A run that sees memory pressure or thermal throttling (`pmset -g therm`, `memory_pressure`) is recorded as an invalid trial and the workload is reduced. Never change system settings.
9. **Prescribed: references.**
   - **Flint:** the oracle is `rec_mlx` (`recursive-oss/rec` v0.1.0 at `e53663b2`), which is where TensorFold's kernels match rounding. It runs in a `.work/venv-rec` uv environment as a development tool only.
   - **Inkling 2 Mini:** the semantic reference is llama.cpp `src/models/inkling2-loop.cpp` plus `conversion/inkling2.py`, pinned at the commit I0 records (master was `40b7a546defe1f60edd1338c719552ffc58c89a0` on 2026-11-21). The vLLM recipe and the model card's linked code are cross-checks. The numerical oracle is `campaign/ref/inkling2_loop_ref.py`: a CPU float32 reference that streams one layer at a time and decodes the checkpoint's exact NVFP4/MXFP8 values. It needs no 105 GB float32 load. `trl-internal-testing/tiny-Inkling2LoopForConditionalGeneration@5e761216759d5d040c09f458d5f077b9aaa5ac78` is the tiny unit fixture.
   - **Delegated (I0):** if llama.cpp converts the official checkpoint to GGUF without requantizing, use it as a second teacher-forced oracle and as the matched comparator. Otherwise it goes only in the "best useful" table. Tie-break: the float32 reference wins.
10. **Prescribed: quality rules.** These are fixed now and never relaxed later.
    - Unchanged-arithmetic transformations must be bit-identical to the frozen baseline on logits: compare the frozen baseline's and the candidate's tensor and logit dumps on identical forced histories. `tf-flint-run exact` is only a self-consistency check inside one build (one-row steps against 16-row windows); keep it, but it does not compare builds.
    - Every logit comparison passes `campaign/ref/logit_gate.py` under a declared arithmetic contract in `campaign/ref/gate_policy.json`: expected shape and position count, nonempty evidence, finite values and numerical bounds derived from the datatype contract, not from a candidate's results.
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

13. **Prescribed: speculation.** Advertise greedy-only speculation first, labelled as such. A sampled mode needs the Leviathan et al. acceptance/rejection correction, or the model's native method. Its proof is the item's own enumerable-distribution tests and argument, as §7 of the source draft required (see **Sequence** F5/I5). Every mode keeps its no-drafter path and automatic fallback, and fallback cost is included in timings.
14. **Delegated: optimization order.** For each model's optimization, the executor ranks hypotheses by measured cost share and keeps the ranked table in `campaign/ledger.md`. Candidates come from §8 A–E of the source draft:
    - A: Exo #2533 fusions;
    - B: M5 TensorOps/MPP matrix paths and quantized layouts;
    - C: attention and state;
    - D: larger fusion or persistent execution, only where dispatch cost is significant;
    - E: model-specific speculation.
    
    Bound: one experiment per item, smallest decisive change first. Tie-break: lower correctness risk.
15. **Prescribed: Exo's M5 exclusions.** Exo PR #2533 (head `c4685725bf6a71fb424e668744744499d44a4fea`, open) gates its fused kernels to M1/M3/M4. The reason it gives is "until verified on that hardware". That is missing validation, not a known numerical difference. Enable a port on M5 only after a bit-exact check against the unfused path on this machine. Exo PR #2530 (open) targets Inkling-**Small**, not Mini; its ideas transfer only where the Mini graph has the same operator.

## Baseline

[baseline.md](baseline.md) holds the frozen binary's sha256, the loop timings and every known red test. In brief, for `fa13df5d` built with Zig 0.17.1:

- `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`: 239.3 s cold and 4.0 s warm no-op;
- `zig build test-flint-draft-host`: 17.7 s;
- `zig build test test-golden -Dcpu=apple_m1`: 547.6 s under the GPU lock, with 7 known red steps (the memcpy link check plus three tests in two modules each).

The benchmark load limit was `FLINT_LOAD_LIMIT_GB=36`, from W = 40,200,896,512 bytes, until the owner's 32 GB cap; it is now 27 (Decision 8).

Host facts measured on 2026-11-21:
- Apple M5 Max: 6 performance and 12 efficiency CPU cores, a 40-core GPU with Metal 4, and 48 GiB of RAM.
- macOS 27.2 (26B123), Xcode 27.2 (27B52), Metal toolchain 27.2.47.0.
- Zig 0.17.1 (owner upgrade; upstream `.zig-version` was 0.17.0). Python 3.15.1, uv, and Pi 1.5.0.
- 621 GiB of free disk.

## Sequence

Items run on per-model chains, and items whose owned files are disjoint run in parallel:

- **Flint:** trustworthy measurement (C2, C3) and F2's divergence trace, then a profile at 1K and 4K, then one bounded O-item screened and, if promising, qualified on the promotion matrix and held-out prompts. Any near-cap fixture must reserve reply headroom. F5 speculation branches off after the target path is qualified.
- **Inkling 2 Mini:** I2's executable native forward path, then I3 minimal serving, then the frozen first-native baseline, then profiling and O-items. I5 branches off after I3.
- **Comparators:** one runnable strong comparator is the early anchor (F3). The full matrix is needed only before a competitive claim.

The phase headings below group items; they are not barriers. Every item ends on its acceptance command plus the narrowest affected suite, run on the integrated `main`. The full `zig build test test-golden` runs at milestones M1 (after F3), M2 (after I3) and the close.

**Disjoint ownership sets:**
- `campaign/` (C, Q and H items, plus the F2 and I2 harness files): one writer at a time.
- Flint: `zig/src/families/flint/`, `zig/src/native/flint.zig`, `zig/kernels/metal/flint/`, `zig/tests/flint/`, `zig/tests/contract/metal_flint*`.
- Inkling 2 Mini: `zig/src/families/inkling2_loop/`, `zig/src/native/inkling2_loop.zig`, `zig/kernels/metal/inkling2_loop/`, `zig/build/inkling2_loop.zig`, `zig/tests/inkling2_loop/`, `campaign/ref/`.
- Shared: `build.zig`, `zig/build/metal_*.zig`, the native registry, `zig/src/core/`, `zig/src/server/` and `zig/kernels/metal/core/`. One registration edit is folded into the item that needs it, and the parent merges these. Any other shared-core change is its own serial item. While it is open, it blocks the other workstream from touching core.

GPU-timed and full-model steps serialize on the GPU lock even when their code lanes run in parallel.

### P0: ground

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

### P1: Flint parity and harness

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

### P2: Inkling 2 Mini parity

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
  
  Then build `campaign/ref/inkling2_loop_ref.py` and check it against the tiny fixture's transformers output, if that output exists, or against llama.cpp.
  
  **Acceptance:** `make -C campaign ref-check MODEL=inkling2-mini` reproduces llama.cpp's teacher-forced top-1 on 4 short histories and states the error it finds.
- **I1. Family loader and refusals** (M, after I0). This covers config, tensor map, tokenizer and template, NVFP4/MXFP8 geometry checks, registration, and a clear refusal of image parts and `--vision`. Vision is out of scope.
  
  **Acceptance:** `zig build test-inkling2-loop-host` passes, including tests that the refusals fire.
- **I2. Executable native forward path** (L, after I1). Reuse the shared NVFP4/FP8 kernels and the Flint/GLM attention pieces where their contracts match; write new kernels only where they do not. State stays explicit for each pass. Increments: a traceable tiny-fixture path through all loop passes and both packed formats; a short teacher-forced path on the real checkpoint with experts kept packed; incremental continuation equal to recomputation, with distinct per-pass KV and convolution state; focused probes at convolution, window and position boundaries using targeted fixtures and selected rows. Name the arithmetic configuration implemented. The checkpoint's quantization metadata specifies activation quantization while the reference computes float32 activations, so a higher-precision-activation path is qualified as that configuration, and the official activation-quantized path is validated separately before claiming equivalence. The MTP tap interpretation stays marked inferred.
  
  **Acceptance:** the pinned checkpoint loads within the memory budget; `tf-inkling2-loop-run` executes a short prompt through all execution slots and at least two greedy decode steps with speculation off; captured logits and selected intermediates pass the logit gate under the declared initial-port contract (independent oracle and state checks, not a regression bound against an I3 baseline that does not yet exist); reset and rerun reproduce the result bit for bit.
- **I3. Serving and baseline** (M, after I2, C1, Q1 and H1). Cover chunked versus full prefill, cached continuation versus a fresh request, EOS, cancellation, context limits, reset after errors, reasoning template, special tokens and tool-call syntax. Then freeze the Inkling baseline matrix. Comparators: llama.cpp (Metal, ggml-org GGUF), MLX (`mlx-community/Inkling-2-Mini-4bit` through mlx-vlm), Exo and Ollama where they run.
  
  **Acceptance:** `zig build test-inkling2-loop`, `make -C campaign quality MODEL=inkling2-mini SET=full`, `make -C campaign pi-harness MODEL=inkling2-mini` and `make -C campaign scorecard`. **Milestone M2:** full suite.

### P3: hardening and speculation

- **F5 and I5. Greedy speculation** (M each; F5 after F3, I5 after I3). F5 is Flint DFlash2 plus copies. I5 is the Inkling MTP head. Test:
  - zero, partial and full acceptance;
  - EOS and the output limit inside a block;
  - rollback of every KV, convolution and per-pass state;
  - resumption after fallback;
  - block versus stepwise verification masks.
  
  Speculative output must equal the qualified target's greedy output on the same histories. Then add sampled mode: enumerable distributions, zero-probability cases, top-k/top-p, temperature, repetition penalty, and grammar constraints. Tune draft length from total measured cost.
  
  **Acceptance:** each model's speculation tests plus `make -C campaign bench MODE=drafter` within the speculation memory envelope.
- **K1. Contract write-up:** record the contract changes listed under **Execution boundary**, and update `docs/supported-models.md` and the recipes.

### P4: optimization

- **O-items** (each S–M). Each is created from the ranked hypothesis table after profiling: Metal System Trace and capture outside timed windows, short and long context. Profiling attributes time to experts and projections, attention, routing, norms and residuals, quantization, dispatch, synchronization, sampling and memory movement, together with bandwidth estimates.
  
  Each O-item follows the source draft's §10 loop: bottleneck, hypothesis, smallest change with a fallback, focused correctness checks, a short paired screen, then the full matrix and held-out prompts for winners. Commit winners. Revert losers but keep their ledger entry and runs. Re-profile and pick the next.
  
  Flint O-items and Inkling O-items run in parallel lanes, with the GPU serialized.
  
  **Plateau:** stop a model's optimization after re-profiling and at least eight substantive failed candidates across more than one hypothesis family, or sooner if a measured hard bound (for example, decode at about 90% of measured weight-read bandwidth) establishes the limit.
- **S2. Secondary investigations:** 128K context is deferred by the owner's 2026-11-22 cap. Any future
  concurrency investigation uses at most 32,768 total tokens per stream and reports aggregate throughput
  separately from single-user speed; it is not part of the current quick matrix.

### P5: close

- **Z1.** Fetch upstream once and record what changed, without rebasing; import upstream fixes only as separately recorded commits. Use `dead-code-archaeology` to delete campaign scaffolding and kernels made dead by a promoted replacement; prove deadness by search plus the build. Run `ultrareview` and fix its P1 and P2 findings. Write `campaign/report.md`:
  - the Mac, models and quantization, and lengths and modes;
  - raw rates with uncertainty, memory peaks and baseline commits;
  - task quality and its limits;
  - local reproduction kept separate from external claims;
  - regressions, unsupported features, targets met or missed, and the narrowest claim the data supports.

## Acceptance

1. **Native engine.** `zig build native -Dcpu=apple_m1` builds. Flint and Inkling 2 Mini serve text. Qualified defaults are on, with opt-in experimental modes and fallbacks. `zig build test test-golden` adds no red beyond what **Baseline** lists.
2. **Correctness.** Run `make -C campaign parity MODEL={flint,inkling2-mini}` and `make -C campaign quality MODEL=… SET=full`; both must meet Decision 10.
   - Speculation tests pass, including state rollback.
   - **Failure condition:** any advertised mode whose output differs from its target-distribution contract, or a free-run text comparison standing in for teacher-forced evidence.
3. **Harness.** `make -C campaign bench …` and `make -C campaign scorecard` regenerate both tables from `campaign/runs/`. `make -C campaign pi-harness` passes for both models.
4. **Provenance.** `make -C campaign fetch` verifies every pin. `campaign/sources.lock.json` and `host.json` record source SHAs, PR heads, compilers, model revisions, tokenizer and config hashes, quantization metadata and licences. `NOTICE` and `THIRD_PARTY_NOTICES.md` cover any imported code.
5. **Ledger and report.** `campaign/ledger.md` lists every experiment, including failures. `campaign/report.md` states each target in Decision 12 as met or not met, with its numbers.
6. **Docs.** `campaign/README.md` says how to build, fetch weights, run, benchmark, verify and resume.

## Needs environment

- M1–M4 compatibility of new kernels: an offline compile only, run with `zig build dev-tools`. Running them needs older Macs; the command is the same `make -C campaign parity` on such a machine.
- Linux and CUDA host CI (`zig build -Dfixed-cpu test-host`) on a Linux host, if the owner wants upstream-style evidence.
- Comparators that need CUDA, such as vLLM and SGLang: unavailable here, not defeated.

## Execution boundary and non-goals

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
- writing to `~/.pi`, `~/.cache` or other global caches, beyond reading the Hugging Face cache;
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

## Reference

These references were verified on 2026-11-21:

- **TensorFold.** Apache-2.0. Flint recipe at `docs/recipes/flint.md`. Flint native entry at `zig/src/native/flint.zig`. `--no-drafts` defined at `zig/src/server/cli.zig:88`. `ignore_eos` honoured at `zig/src/server/chat_generation.zig:82`. `tf-flint-run` registered at `zig/build/metal_targets.zig:283`. The Inkling 2 Small host package is wired at `build.zig:123`.
- **Upstream receipts** in `docs/recipes/flint.md` come from an M5 Ultra with 256 GiB. They are not a baseline for this machine.
- **`buildlib.locks`:** `load_sources` and `fetch_input` (URL order, digest and size checks, atomic publish).
- **Hugging Face pins:**
  - `Inkling-2-Mini` `model-00001-of-00001.safetensors`: sha256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`, 19,526,019,168 bytes.
  - `mtp.safetensors`: sha256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`, 635,801,416 bytes.
  - `tokenizer.json`: sha256 `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6`.
