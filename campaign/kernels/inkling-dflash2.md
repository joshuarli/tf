# Inkling 2 Mini DFlash2 drafter (DF1)

**Verdict.** Greedy speculation with `incoai/Inkling-2-Mini-DFlash2@4f9088ea66c2` on the shared Metal DFlash2
runtime is exact against plain greedy decoding of the oQ4e target, but it is far from Decision 17's budget. On
the quick prompt classes it commits 4.3–5.8 tokens a round at 1K and 2.7–3.4 at 4K. A round takes 61–77 ms, of
which the 8-row grouped verify forward is 93%. That is 10.5–28 ms a committed token against the 5 ms that
200 tok/s needs. Effective decode is 1.04–1.56× plain greedy at 1K and 0.65–0.82× at 4K, in the same binary and
process.

## What was built

All on `lane/df1-dflash2`.

- **Shared runtime, by configuration** (`zig/src/families/qwen27/dflash/`). New config fields are `final_norm`,
  `softcap` and `candidates`. The first admits taps that end at the id past the target's layers. The second is
  `final_logit_softcapping` with `selector_softcap_mode` `after_correction`: the selector caps each candidate's
  score after its edge. The third limits candidates to the target's real tokens. The target binding gains an
  `embed_binding`, for a table not named `model.embed_tokens`, and an `embed_norm` for the draft's own rows.
  `requireStock` admits the published Inkling shape and refuses those fields on the Qwen, Ornith and Flint shapes.
  Each changed expression keeps its old form when the fields are off.
- **Tap capture** (`zig/src/families/inkling2_loop/taps.zig`, `kernels/taps.metal`). A prefill chunk or batch step
  can copy the block residual entering listed visits, and the final norm the head reads, as bf16 rows
  `[rows][4 × 2304]`, rounded to nearest even.
- **Grouped verification** (`group.zig`, `kernels/verify.metal`). This is `lane/m1-mtp`'s design (`dc7a4dba3`)
  ported onto main's D2 kernels. One batch step takes the newest token and its drafts as consecutive rows, each
  computing what an ordinary batch step at its position computes. The convolution, key/value-store and
  split-attention twins read the group's earlier rows from captures. The split twin keeps D2's staged 32-row
  blocks; m1-mtp's twin was the pre-D2 serial kernel. The forward writes no cache. The commit writes only the
  accepted rows' keys, values and convolution inputs, so a rejected row needs no save or restore. Picks are made
  on the GPU, as the host's scan of the logits would make them.
- **Drafter** (`dflash.zig`). It loads the checkpoint through `rt.Model.openBound`, as `families/flint/draft.zig`
  does. It binds the oQ4e target's affine 8-bit embedding (`model.embed`) through the target's
  `model.embed_norm`. It also binds the affine 8-bit `lm_head` over 200,064 rows: whole tiles covering the
  200,058 real tokens, with the candidate view cut at 200,058. The selector draws a greedy chain of seven with edge
  weight 1, as the references sum unary score and edge (mlx-vlm `dflash2.py`, tokenspeed `dflash2.py`, Splash's
  `draft_select_tree`). The shared selector's default of 0.6 stays Flint's. Linears run as BF16 (`bits 0`) or as
  prepared q4 copies (`bits 4`). The drafter refuses by name: `InklingDFlash2NeedsAffineHead` for the native
  NVFP4 checkpoint or any target without that head, `InklingDFlash2Softcap` for another softcap,
  `InklingDFlash2Taps`, `InklingDFlash2Block`, and, from `requireStock`, `NotStockDFlash2` or
  `UnsupportedSelectorSoftcap` for any other shape, taps or softcap placement. Its draft residency is 522 MB at
  q4 and 1,084 MB at BF16, inside Decision 8's 4 GiB.
- **Speculation and runner** (`dflash_spec.zig`, `dflash_run.zig`). A round has five phases. It drafts a chain after
  the newest token, verifies the chain in one grouped step, picks on the GPU, commits the accepted prefix, then
  absorbs those rows' taps into the drafter. The output limit shortens a round's chain; an EOS cuts its accepted
  tokens. `tf-inkling2-loop-run <model> <prompts.json> dflash <drafter> <tokens> <out.jsonl> [bits]` runs
  speculative and plain greedy decoding of each prompt in one process, in alternating order. It writes every
  round's timing and both sides' token arrays and hashes, and it exits non-zero if the tokens differ.
  `dflash-widths <reps> <out.jsonl> [load_s]` times the grouped verifier at widths 1, 2, 4 and 8, and one batch
  step, from identical states.

Serving integration (`inkling2_loop_host.zig`) is SV1's and was not touched.

## Capture interpretation

**Implemented.** Draft id k reads the block residual's partial sum entering execution visit k, as the visit
before wrote it. Visits count every core pass: layers 0–5 are 0–5, the core is 6–41 and layers 18–23 are 42–47.
Id 48, the id past the 48 visits, is the post-final-norm row the head reads. So ids 5, 23 and 41 are the inputs of
applications 5 (layer 5), 23 (layer 11, pass 2) and 41 (layer 17, pass 3), that is the outputs of applications 4,
22 and 40. None of them is a pass's first visit, so a pass's restart from the prelude does not enter. The draft's
own rows, the anchor and the mask token 200056, are the target's table rows through its embedding norm.

**Why.**

1. The checkpoint's `sglang_inkling_dflash2.py` makes `set_dflash_layers_to_capture` call
   `set_dspark_layers_to_capture`, under the comment "Both drafts use native post-residual visit features and
   final norm". It requires sorted, unique ids ending at `num_layer_visits`, "the final-norm layer visit".
   DFlash2's ids are therefore Inkling's DSpark ids, not the HF "after decoder block k" convention that Flint's
   `5, 15, 26, 36, 46` follow.
2. The DSpark capture in the pinned mlx-vlm (`mlx_vlm/models/inkling2/language.py:700-712`) is adapted from vLLM's
   Inkling 2 DSpark. It stores `captures[visit] = x` before `self.layers[index](x, ...)`, under the comment
   "DSpark taps the partial block entering the layer, not its output". It stores the normalized output at
   `len(self.layer_indices)`. Its embedding is `embed_norm(embed_tokens(x))`.
3. Splash 1.7.2 says so in `docs/models.md`: Inkling's drafts read "the rows passed into those applications, the
   outputs of applications 4, 22 and 40 … with no shift by one … and 48 its final norm's; their own rows take
   its table's norm". `Inkling2LoopTarget.cpp` captures `input` before each application's attention and
   `output` after the head's mix-norm, under `requireDraftRows(true, true)`. The release note's phrase "the rows
   applications 5, 23 and 41 read" means the same rows.
4. TensorFold's MTP head already maps the same ids this way (`mtp.zig`, test "checkpoint taps name the raw state
   entering each visit and the final normalized state").

**Checked by acceptance** (1K prompts, q4 drafter, 512 tokens). The token streams are deterministic. Every run's
tokens equalled plain greedy.

| Reading | code-1024 | prose-1024 | structured-1024 |
| --- | ---: | ---: | ---: |
| Entering visits 5, 23, 41 + final norm, normed token rows (implemented) | 5.81 | 4.52 | 4.33 |
| Outputs of visits 5, 23, 41 (entering 6, 24, 42) + final norm | 3.76 | 2.73 | 3.19 |
| Entering visits, token rows without the embedding norm | 1.00 | 1.00 | 1.00 |

Committed tokens per round. Without the norm, all 1,530 rounds that offered drafts accepted none.

## Correctness

- `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0 on `fff7591e7`
  (`.work/suite/20261123T231107`, 9.6 min). `inkling-host` passed 35/35 on its first run. `inkling-gpu` passed
  14/14 steps and 123/123 tests, with none skipped: 115 already on main and the 8 below.
- Family module, both checkpoints:
  - Grouped logits, GPU picks and committed caches equal sequential batch decoding bit for bit. Three streams of
    4, 3 and 1 rows share a step, the first crossing the 1,024-row local ring; their commits keep partial
    prefixes, single rows and whole groups.
  - A grouped row's taps equal a one-row batch step's at its position, across the ring's wrap.
  - Host tests: tap rounding, and the acceptance rule.
- `test-inkling2-loop-dflash2`, oQ4e, real drafter:
  - Speculation equals plain greedy on four chat prompts (2,572, 37, 32 and 30 tokens) for 256 tokens each. That
    is 1,024 tokens over 273 rounds: 72 zero, 140 partial and 58 full.
  - Forced drafts give full, zero and prefix acceptance; a limit shortens a block, and an EOS cuts one. The output
    equals plain greedy, and after full, zero and partial runs the target caches (keys, values, all four
    convolution rings, every slot) equal plain decoding's bit for bit.
  - Config and target refusals by name.
- Flint/qwen27 DFlash2 host tests, before and after:
  - `zig build test-flint-draft-host`: 47/47 steps and 86/86 tests both times.
  - `zig build test-qwen27`: 48/48 steps. Before, 197/202 passed with 5 skipped; after, 200/205 with 5
    skipped. The 3 new tests are the Inkling config, Flint's refusal of a final-norm tap, softcap or narrowed
    vocabulary, and the selector's softcap.
- `zig build native -Dcpu=apple_m1` exited 0.

## Round table

oQ4e target with the q4 drafter: draft depth 7, an 8-row verify and 512 greedy output tokens with EOS ignored.
Three processes ran 22:42–22:59 on 2026-11-23: `q4-a`, `q4-b` and `q4-c`. Prompts are the dev corpus rendered with
the checkpoint's chat template at its default thinking effort (0.9), cut to 1,024 or 4,096 tokens (`prompts.json`).
Rounds and committed tokens were identical in all three. Times are the median of the three per-process means,
range in brackets. Spec tok/s counts the 511 tokens after prefill's first over the rounds. Plain is one batch step
per token plus the host's argmax, from the same prefill state in the same process.

| Prompt | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Select | Commit | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 88 | 5.81 | 61.1 (60.5–61.3) | 65.3 (63.6–65.6) | 4.00 | 56.7 | 0.24 | 0.19 | 0.01 | 95.0 (94.8–96.0) | 16.39 (16.38–16.43) | 61.0 (60.9–61.0) | 1.56 (1.55–1.58) |
| prose-1024 | 113 | 4.52 | 73.7 (72.8–74.4) | 76.1 (75.5–77.0) | 4.80 | 68.5 | 0.26 | 0.21 | 0.01 | 61.3 (60.8–62.1) | 17.43 (17.33–17.48) | 57.4 (57.2–57.7) | 1.07 (1.06–1.08) |
| structured-1024 | 118 | 4.33 | 74.9 (73.5–75.3) | 76.5 (75.0–77.1) | 4.84 | 69.6 | 0.26 | 0.21 | 0.00 | 57.8 (57.5–58.9) | 17.86 (17.67–17.98) | 56.0 (55.6–56.6) | 1.04 (1.03–1.04) |
| code-4096 | 185 | 2.76 | 76.6 (74.4–77.6) | 78.4 (76.4–79.3) | 4.99 | 71.1 | 0.26 | 0.21 | 0.00 | 36.1 (35.6–37.1) | 17.93 (17.70–18.31) | 55.8 (54.6–56.5) | 0.66 (0.64–0.66) |
| prose-4096 | 190 | 2.69 | 75.7 (74.2–75.9) | 77.5 (76.0–77.8) | 4.93 | 70.3 | 0.26 | 0.21 | 0.01 | 35.5 (35.4–36.2) | 18.39 (17.95–18.40) | 54.4 (54.3–55.7) | 0.65 (0.65–0.65) |
| structured-4096 | 149 | 3.43 | 75.1 (74.1–75.3) | 77.2 (76.7–77.5) | 4.90 | 69.8 | 0.26 | 0.21 | 0.00 | 45.6 (45.5–46.3) | 17.86 (17.69–18.06) | 56.0 (55.4–56.5) | 0.82 (0.82–0.82) |

Phases are mean ms a round.

- **Draft**: absorbing the previous round's taps into the drafter, its block forward, head, top-k and readback,
  and the CPU selector.
- **Verify**: the grouped forward over the anchor and seven drafts, with tap capture and logits for all rows.
- **Select**: GPU argmax, readback and acceptance.
- **Commit**: writing the accepted rows' keys, values and convolution state. No save or restore phase exists.
- **Host**: absorbing taps into the drafter's ring and bookkeeping.

Over all 843 rounds of each process, verify was 92.8–92.9% of round time, draft 6.5%, and select, commit and host
together 0.7%.

**Other costs.**

- Prefill with tap capture took 0.83–0.91 s at 1K and 3.2–3.7 s at 4K. Building the drafter's 2,047-row window
  then took 11–30 ms.
- The BF16 drafter (`bf16-a`) proposes almost the same chains: committed 5.81, 4.60, 4.44, 2.84, 2.81 and 3.50.
  Its draft costs 12.1–14.0 ms a round against 4.0–5.0 ms, so its rounds are 7–9 ms longer (69.7–83.2 ms). Its
  linears run through the shared runtime's `tf_bf16_linear`, one row a threadgroup grid row, so each of the eight
  block rows reads the 0.74 GB of BF16 weights again.

## Verifier cost at widths 1, 2, 4 and 8

Each prompt was prefilled. The rows are the greedy continuation; width w verifies its first w. Grouped forwards
leave the caches unchanged, and the batch step is rolled back, so every sample starts from the same state. Nine
interleaved repetitions followed one warm-up; p50 of forward plus GPU picks is in ms, averaged over the three
prompts of a context, range over them in brackets. "Rested" is `widths-a`, run right after a model load. "Sustained" is
`widths-b`, run after 20 s of continuous 8-row verifies per prompt (see Conditions).

| Context | Path | Width | Rested p50 | Sustained p50 |
| ---: | --- | ---: | ---: | ---: |
| 1,024 | batch step | 1 | 16.3 (16.0–16.9) | 19.1 (19.1–19.1) |
| 1,024 | grouped | 1 | 16.8 (16.3–17.8) | 19.4 (19.2–19.6) |
| 1,024 | grouped | 2 | 21.7 (21.1–22.6) | 25.0 (24.6–25.6) |
| 1,024 | grouped | 4 | 33.2 (32.7–34.0) | 38.7 (38.6–38.7) |
| 1,024 | grouped | 8 | 56.3 (55.4–57.6) | 65.2 (64.9–65.5) |
| 4,096 | batch step | 1 | 18.8 (18.2–19.5) | 19.6 (19.5–19.7) |
| 4,096 | grouped | 1 | 18.9 (18.4–19.3) | 19.4 (18.9–19.8) |
| 4,096 | grouped | 2 | 24.3 (23.8–24.8) | 25.5 (25.0–25.8) |
| 4,096 | grouped | 4 | 37.9 (37.4–38.5) | 39.6 (39.5–39.8) |
| 4,096 | grouped | 8 | 63.1 (61.7–64.0) | 66.1 (65.6–66.3) |

A grouped row of one costs what a batch step costs. Each further row adds 5.7 ms rested and 6.5 ms sustained, so
eight rows cost 3.4× one. `run.py project q4-c widths-b` cuts each prompt's full-depth chains to width w and prices
the round at sustained cost. Width 4 beats width 8 on five of the six prompts: 52–72 tok/s against 38–64. On
code-4096 and prose-4096, width 2 is marginally better still, at 55 tok/s. Code-1024 is best at width 8, at
85 tok/s. No width reaches 100 tok/s.

## Conditions

- **Lock and gates.** Every run went through `campaign/heavy.py`: the GPU lock and the 28 GB cap, with peaks
  17.6–20.3 GB. Each passed `gpu-gate-ready -1` (60% limits) and `perf-gate-ready` (`PERF_GATE_MAX_BUSY_CORES=4`)
  before and after on its first attempt. Readings are in each `NAME.host.json`.
- **Host.**
  - Memory: 84–85% free, pressure level 1, no swap growth.
  - Thermal: `pmset -g therm` recorded no thermal or performance warning before or after any run.
  - Other GPU load: `gpu_other` read 23–37% at the gate readings. Sampled idle, the terminal and WindowServer took
    about 25% of GPU time; neither is a campaign process.
- **Sustained-load slowdown.** The host is a MacBook Pro (Mac17,7).
  - The 8-row verify ran 55–57 ms over the first 50–70 rounds after a model load, then rose to 66–72 ms and stayed
    there. Plain steps after it ran 1–3 ms slower.
  - About 15 s idle, the next model load, restored it.
  - It follows elapsed load, not the prompt. With prose-1024 run first (`swap-q4`), prose-1024 was the fast one and
    code-1024 the slow one. Autorelease pools around the drafter's frames did not change it (`swap-pool-q4`).
  - `ProcessInfo.thermalState`, sampled each second in `q4-c` and `widths-b`, stayed `nominal` through the onset
    and read `fair` only for the last 15–19 s of each 150 s run.
  - The cause is not traced; the laptop's GPU power and clock management fits.
  - In the round table, code-1024 ran first in each process and is partly in the fast regime; the other prompts are
    in the sustained one.
- **Invalid trial.** `shift-q4`'s timings are invalid: a runner compile overlapped its last minute. Its acceptance
  is deterministic and stands.
- **Build identities.**
  - `dca724f2…` is commit `f7a2a4329`. It ran `q4-a`, `q4-b`, `bf16-a`, `widths-a`, `shift-q4`, `nonorm-q4` and
    `swap-q4`.
  - `2dd830ae…` is `f7a2a4329` with the autorelease pools and token records of `345142815`. It ran `q4-c` and
    `swap-pool-q4`; its dflash mode is `345142815`'s.
  - `19ee2880…` is `345142815`. It ran `widths-b`.
- **Benchmark checklist.**
  - Limiter: the verify forward, 93% of a round.
  - Tuning: plain is the served batch step in a ReleaseFast build. The drafter ran in both modes. The grouped
    verifier is an untuned port (V1).
  - Errors: none. All 34 prompt runs in the 8 recorded dflash processes equalled plain.
  - Reproduction: three processes for the table, with round-mean spread at most 4.3% and speedup spread at most
    0.03.
  - Work done: every prompt emitted 511 checked tokens.

## Where the time goes

**The single largest cost is the 8-row grouped verify forward:** 56.7–71.1 ms of a 61–77 ms round, 93%. It is
3.4× a one-row step, and its growth with rows is mostly routed experts.

- One token reads about 3.99 GB of oQ4e weights, 1.72 GB of it experts (`inkling-decode-step.md`).
- Eight rows routed independently would touch 128 × (1 − (120/128)^8) ≈ 52 distinct experts a visit instead of 8.
  That is about 11.1 GB of experts plus 2.3 GB of everything else: a 25.7 ms floor at the measured 521 GB/s.
- The measured 56–66 ms is 2.2–2.6× that floor, about the one-row step's own 2.1× (16.3 ms against 7.65 ms).

**Even at that floor, 200 tok/s needs acceptance this checkpoint does not reach here.**

- A 26 ms verify plus a 4–5 ms draft is about 31 ms a round: 5 ms a committed token only at 6.2 committed. These
  prompts commit 2.7–5.8.
- At 4K, 2.7 committed allows 13.5 ms a round, less than one ordinary decode step.
- The 4K prompts accept less in every quarter of their 512 tokens (2.2–3.8 committed) than the 1K prompts
  (3.7–7.2). The drafter sees only its 2,047-row window, while these replies review a 4K prompt. Its window
  handling is the shared runtime's, as Flint's is. A capture or context bug would show as lower acceptance, which
  greedy equality cannot catch, and none was found. Lower acceptance at 4K is not explained beyond that.

**Next experiment (V1).** Take a Metal System Trace of one 8-row grouped verify against one 1-row step at the same
1K state. Attribute the 40 ms difference to the expert gate/up/down kernels, the attention twins, the projections
and dispatch. Then take the largest share toward the 26 ms weight floor. That cost, not the drafter, the commit or
the host, decides whether DFlash2 can approach 200 tok/s.

The width projection gives a cheaper interim gain. A chain cut to four rows, or a selector-confidence cut,
already beats eight rows on five of the six prompts at today's costs.

## Reproduction

From the worktree, with the runner built by
`zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix .work/tmp/df1/bench-build`:

```sh
R=/Users/josh/repos/tf/.work/tmp/df1/bench-build/bin/tf-inkling2-loop-run
D=campaign/kernels/inkling-dflash2-runs
python3 -I $D/run.py prompts $R $D/prompts.json --cells 1024,4096 --effort 0.9
python3 -I $D/run.py bench $R $D/prompts.json q4-a --bits 4 --tokens 512
python3 -I $D/run.py bench $R $D/prompts.json bf16-a --bits 0 --tokens 512
python3 -I $D/run.py widths $R $D/prompts.json widths-a --reps 9
python3 -I $D/run.py widths $R $D/prompts.json widths-b --reps 9 --load 20
python3 -I $D/run.py bench $R $D/prompts-1k.json shift-q4 --bits 4 --env TF_INKLING_DFLASH2_TAP_SHIFT=1
python3 -I $D/run.py bench $R $D/prompts-1k.json nonorm-q4 --bits 4 --env TF_INKLING_DFLASH2_NO_EMBED_NORM=1
python3 -I $D/run.py table q4-a bf16-a widths-a
python3 -I $D/run.py project q4-c widths-b
```

The two `TF_INKLING_DFLASH2_*` switches are runner-only diagnostics of the capture reading. The prompt builder
reproduces `campaign/prompts/inkling/code-1024.json` at effort 0, except its last content token: that fixture was
cut as text, these are cut as tokens.

## Open problems

- The 4K acceptance (2.7–3.4) is not explained beyond the drafter's window.
- The sustained-load slowdown (about +20% on the 8-row verify) is not traced.
- Prefill with taps was not compared with plain prefill.
- Prefill-chunk taps come from the half-precision prefill path, and decoded rows' taps from the batch step.
  Their effect on acceptance is unmeasured.
- `prefill.zig` (1,491 lines) and `run.zig` (784) were over the 600-line module limit before this lane.
