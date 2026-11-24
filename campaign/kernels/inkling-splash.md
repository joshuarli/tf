# S1: Splash 1.7.2 as the Inkling 2 Mini reference on this M5 Max

Item S1 of Decision 17. This is a single bounded measurement pass. Its purpose is to show what this Mac
reaches on the same model, so it is not a qualified comparison. tf numbers come from main `93de31eca`
(DF1 merged), built with `zig build native -Dcpu=apple_m1`.

**Verdict.** On this Mac, Splash clears the 2,000 tok/s prefill target at every length from 1K to the cap,
on the same oQ4e weights. It reaches 200 committed tok/s only on code prompts. tf trails on all three
measures:

| | Splash (oQ4e, bf16 KV and activations) | tf `93de31eca` |
|---|---:|---:|
| Unseen-prompt prefill, 1K / 4K / 16K / ~31.7K tokens | 2,912 / 2,820 / 2,294 / (cap breach) | 1,180 / 1,456 / 1,373 / 1,357 |
| Unseen-prompt prefill on Splash defaults (int8 KV) | 3,192 / 2,723 / 2,427 / 2,213 | |
| One speculative cycle (code / prose / structured, 1K) | 32.7 / 31.6 / 30.2 ms | 61.1 / 73.7 / 74.9 ms |
| Committed tokens per cycle, 1K | 7.6 / 5.9 / 4.7 | 5.81 / 4.52 / 4.33 |
| Committed tokens per cycle, 4K | 9.1 / 3.8 / 3.9 | 2.76 / 2.69 / 3.43 |
| Speculative decode, 512 tokens, 1K | 232 / 187 / 156 tok/s | 95.0 / 61.3 / 57.8 |

The tf speculative numbers are DF1's round table (`inkling-dflash2.md`). They come from the same prompt files,
but that table was timed in-process, not through the server.

**Correction (DF2a, `inkling-df2-accept.md`).** The 4K rows do not compare like for like. DF1's harness cut the 4K
prompt files to 4,096 tokens, so code-4096 lost its last 1,184 tokens, while Splash served all 5,280. Splash's
9.1 tokens a cycle also comes from its 16-row tree with prompt lookup, not from its drafter: its drafter has the
same 2,048-row window as tf's. tf's 4K drop is that window. The 4K replies quote text from before it, and the
checkpoint does not use rows past its trained distances. It is not a tf bug.

## Setup

- **Splash.** The 1.7.2 release build (`2afc7337aacc`, tarball sha in `campaign/sources.lock.json`) was run
  from `.work/src` with `HOME=.work/splash-home`. Nothing was written under `~/Library`.
  - Flags: `--language-only --offline --max-memory 26G --max-context 32K --max-cache-disk 0
    --idle-release off --max-concurrent-requests 1 --verify-rows auto`.
  - The plugin is `campaign/comparators/splash/server.py`. It runs under `heavy.py`, which holds the GPU lock
    and enforces the 28 GB cap.
- **Draft.** `incoai/Inkling-2-Mini-DFlash2@4f9088ea66c2` for every Splash run. Splash has no
  autoregressive mode.
- **Profiles.**
  - `controlled`: oQ4e with `--kv-format bf16 --activations bf16`, the closest match to tf's half-precision
    path.
  - `default`: oQ4e on Splash's defaults (int8 KV and int8 prefill activations on M5).
  - `gguf`: `unsloth/Inkling-2-Mini-GGUF` UD-Q4_K_M (`8078d149286e`) on Splash's defaults. This is the
    best-practical configuration.
- **Host.** M5 Max, 40-core GPU, 48 GiB. The owner's desktop was in use, with GPU idle at a 23% baseline and
  load average about 3. Every curve and every decode trial except two passed the gate's before and after
  checks.

## Prefill across the curve

- **Commands.** Each curve came from one fresh server:
  - Splash: `make -C campaign splash-curve SPLASH_PROFILE=... [SPLASH_MODEL=...]`.
  - tf: `comparators/serve_curve.py tensorfold --model-dir <oQ4e>`, with `--parallel 1 --prompt-cache-gib 0
    --no-drafts`.
- **Client.** `comparators/prefill_curve.py` sends `max_tokens` 1, non-streamed.
- **Prompts.** Each prompt is an unused span of one of the six 32K corpus files (dev and holdout; code, prose
  and structured), with a distinct leading line.
  - The first request after ready is reported separately as "first".
  - At 1K, one prompt comes from each file; these calibrate characters per token.
  - Longer lengths get two prompts each.
  - No request reused a prefix: every request reported 0 cached tokens.
- **Measure.** Client wall time, including one decode step, per uncached prompt token. Medians over the n
  shown; ranges in brackets.

| Target tokens | n | Splash controlled | Splash default (oQ4e) | Splash default (GGUF) | tf `93de31eca` |
|---:|---:|---:|---:|---:|---:|
| first (1,258) | 1 | 3,190 | 3,247 | 3,186 | 981 |
| 1,024 | 6 | 2,912 (2,823–3,181) | 3,192 (2,791–3,251) | 2,671 (2,583–2,865) | 1,180 (1,008–1,422) |
| 4,096 | 2 | 2,820 (2,809–2,830) | 2,723 (2,710–2,736) | 2,671 (2,653–2,690) | 1,456 (1,361–1,550) |
| 8,192 | 2 | 2,331 (2,284–2,379) | 2,567 (2,488–2,646) | 2,482 (2,370–2,593) | 1,483 (1,387–1,579) |
| 16,384 | 2 | 2,294 (2,266–2,322) | 2,427 (2,426–2,427) | 2,386 (2,377–2,395) | 1,373 (1,368–1,378) |
| 31,744 | 2 | not run (see below) | 2,213 (2,201–2,225) | 2,237 (2,231–2,242) | 1,357 (1,354–1,360) |

Time to first token at about 16.5K tokens is 6.8–7.1 s for Splash and 12.0 s for tf. Near the cap it is
14.3 s for Splash and 23.5 s for tf.

- **The controlled profile cannot reach the cap under the campaign limits.** In the first curve pass, a
  22,058-token prompt pushed the process tree to 28.2 GB, and heavy.py stopped the server
  (`heavy: process tree charged 28.2 GB exceeded the 28 GB cap`). The default int8-KV profile served
  32,125 tokens within the cap. The controlled curve was therefore rerun only up to 16K.
- **tf's 1K cell is weak and noisy.** The three dev prompts ran 1,311–1,422 tok/s; the three holdout prompts
  ran 1,008–1,049. Splash shows a much smaller split. The cause is not traced.
- **The earlier figure of "about 1,995 tok/s once warm" does not hold on unseen prompts.** On this curve,
  tf's best cell is 1,579 tok/s.

## Speculative decode

- **Command.** `make -C campaign splash-bench SPLASH_PROFILE=... SPLASH_PAIRS=2 [SPLASH_MODEL=...]`. This
  runs a fresh server per cell and the six dev prompts, each sent as cold, warm (cached) and natural-EOS
  requests of 512 tokens.
- **Records.** `campaign/runs/2026-11-23/splash-spec-*.jsonl`.
- **Cycle data.** Splash's server log reports committed tokens per cycle for every request. Cycle time is
  tokens per cycle divided by tok/s.
- **Runs.** Pair 1 and pair 2 are separate server processes. The pair-1 records are marked invalid, but see
  "Gate artifact" below: they are good data.

Cold requests (Splash log `tok/s`, tok/cycle, derived ms/cycle):

| Prompt | Controlled p1 | Controlled p2 | Default p1 | Default p2 | GGUF p2 |
|---|---|---|---|---|---|
| code-1024 | 232 · 7.6 · 32.7 | 233 · 7.6 · 32.7 | 216 · 7.3 · 33.8 | 163 · 5.4 · 33.1 | 247 · 8.3 · 33.6 |
| prose-1024 | 186 · 5.9 · 31.8 | 187 · 5.9 · 31.6 | 144 · 4.3 · 30.0 | 167 · 5.1 · 30.5 | 150 · 4.5 · 30.1 |
| structured-1024 | 158 · 4.7 · 29.7 | 156 · 4.7 · 30.2 | 148 · 4.5 · 30.4 | 142 · 4.4 · 30.9 | 157 · 5.0 · 31.9 |
| code-4096 | 255 · 9.1 · 35.7 | 255 · 9.1 · 35.6 | 179 · 5.8 · 32.5 | 170 · 5.7 · 33.5 | 260 · 9.1 · 35.0 |
| prose-4096 | 116 · 3.5 · 30.3 | 128 · 3.8 · 29.8 | 124 · 3.5 · 28.2 | 127 · 3.7 · 29.1 | 107 · 3.2 · 30.0 |
| structured-4096 | 126 · 4.0 · 31.8 | 120 · 3.9 · 32.6 | 129 · 3.9 · 30.2 | 123 · 3.6 · 29.4 | 140 · 4.2 · 30.0 |

- **Cycle time is nearly flat.** Every cell runs 28–36 ms a cycle, whatever the prompt, context or
  acceptance. Splash's start-up calibration of verify cost on this Mac, in ms per cycle:

  | Verified rows | 1 | 2 | 3 | 8 | 9 | 15 | 16 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | ms per cycle | 15.84 | 18.62 | 21.48 | 26.32 | 27.78 | 32.74 | 32.34 |

  DF1 measured tf's grouped verify on the same weights at 16.8 ms for one row, 21.7 for 2, 33.2 for 4 and
  56.3 for 8 (1K, rested). The one-row costs match. Splash's cost per extra row is about a fifth of tf's.
- **Splash commits more per cycle than an 8-token chain allows.** Code-4096 commits 9.1 tokens a cycle, which
  is more than an anchor plus seven drafts could give. Splash must verify a tree or a longer block within
  its rows (up to 16). Its controlled runs commit 9.1 a cycle on code-4096 where tf's chain commits 2.76 on
  the same prompt file. So tf's 4K acceptance drop is not a property of the drafter.
- **Reaching 200 tok/s.** Splash clears 200 only on code. Prose and structured commit 3.2–5.9 tokens a cycle,
  giving 107–192 tok/s. At Splash's ~31 ms cycle, 200 tok/s needs 6.2 committed tokens a cycle.
- **Output is not reproducible.** With `--verify-rows auto`, greedy output varies between runs: cold and warm
  requests for the same prompt return different output hashes. Splash says this in its log ("greedy and
  seeded outputs can differ from run to run"). The `--verify-rows all` profile was not run in this pass.
- **Controlled reproduces across processes; default does not.** In controlled, p1 and p2 agree to within
  1–2 tok/s and identical tokens per cycle on every prompt except prose-4096. The int8 default varies more
  (code-1024: 7.3 against 5.4 tokens a cycle).

## Gate artifact (fixed)

- **Symptom.** The first two passes marked nearly every Splash trial invalid. The gate reported device
  utilization of 98–99%.
- **Cause.** `comparators/run.py` called the GPU gate without the server's process IDs. The gate therefore
  charged Splash's own trailing utilization, from the request it had just finished, as outside load. That
  showed up in the after check, and in the before check of the next trial on the same server.
- **Check.** With Splash idle, before and after a request, the GPU sat at the desktop's 23% baseline.
- **Fix.** The runner now passes the server's process tree, as the bench harness does. After the fix, the
  only invalid trials were two flagged for real compressor activity (structured-4096 in controlled p2 and in
  default p2).

## Benchmark checklist

1. **Limiter.**
   - For Splash, not profiled; it is the reference. Its calibration shows that verify cost per row is the
     whole difference from tf.
   - For tf prefill, the P3c profile still applies.
   - For tf decode, the 8-row verify is 93% of a round (DF1).
2. **Tuning.**
   - Splash ran its release build with its own defaults, apart from the cache and memory flags set to the
     campaign limits.
   - tf ran its served ReleaseFast binary. The prompt cache was off, which is irrelevant on unseen prompts.
3. **Limits.** 2,200–3,250 tok/s is within this GPU's compute. There is no cache: 0 cached tokens on every
   curve request.
4. **Errors.**
   - Every curve request returned 200 with its prompt token count, except the controlled-profile cap breach
     noted above.
   - Every decode trial returned 512 tokens.
   - Prefill output quality was not checked: each curve request returns one token.
5. **Reproducibility.** This is one pass, not interleaved.
   - n = 6 at 1K and n = 2 above, with ranges as shown.
   - The engines ran one after another: Splash default oQ4e, Splash GGUF, Splash controlled, then tf.
   - The tf-to-Splash gap (1.6–2.5×) is far outside the run-to-run spread.
   - Profile differences within Splash (controlled against default against GGUF) are mostly inside it and
     are not claimed.
6. **Relevance.** Time to first token at 16K is 12.0 s for tf against 6.8–7.1 s for Splash, which a user
   waits through on every new long prompt.
7. **Work done.** The prompt token counts match the targets. Splash's log shows its own prefill time per
   request, which agrees with the client: for example, 12,931 tokens took 5,289 ms, or 2,445 tok/s.

## What this changes for the next slice

- **V1 has a measured target.** On this Mac and these weights, the 8-row verify costs 26 ms in Splash and
  16-row costs 32 ms, against 56–66 ms for 8 rows in tf.
- **Acceptance is a gap of its own.**
  - Splash's draft path commits 1.3× (code-1024) to 3.3× (code-4096) what tf's chain commits on the same
    prompts.
  - A tree, or a deeper verify within the same row budget, is part of the gap.
  - tf's 4K acceptance collapse needs its own trace: candidates are the prompt taps from the half-precision
    prefill path, and window handling.
- **P4 has a measured target.** Splash runs 2,200–2,900 tok/s in the profile closest to tf's arithmetic.

## Reproduce

```
make -C campaign fetch NAMES="splash inkling-2-mini-dflash2 inkling-2-mini-ud-q4km-gguf"
make -C campaign splash-setup
make -C campaign splash-bench SPLASH_PROFILE=controlled SPLASH_PAIRS=2
make -C campaign splash-bench SPLASH_PROFILE=default SPLASH_PAIRS=2 [SPLASH_MODEL=~/models/Inkling-2-Mini-GGUF@8078d149286e]
make -C campaign splash-curve SPLASH_PROFILE=default [SPLASH_MODEL=...]
make -C campaign splash-curve SPLASH_PROFILE=controlled SPLASH_CURVE=1024,4096,8192,16384
cd campaign && uv run --project . python comparators/serve_curve.py tensorfold \
  --model-dir ~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd --out runs/<date>/tensorfold-prefill-<sha>-oQ4e.jsonl
```

Records:
- `campaign/runs/2026-11-23/splash-spec-*.jsonl` (decode trials).
- `campaign/runs/2026-11-24/*-prefill-*.jsonl` (curves, with a final `conditions` record each).
- `campaign/kernels/inkling-splash-runs/` (the Splash server logs that carry tokens per cycle).

## How Splash gets there (source study, 1.7.2 `2afc7337aacc`, Apache-2.0)

These findings come from reading the pinned source. Paths are under `runtime/`, and anything inferred is marked.

- **The cycle is a tree, not a chain.** A cycle runs the drafter once: 8 rows, the anchor plus 7 mask rows
  (`model/DFlashDraft.cpp:268`). A lane then verifies up to 16 rows (`metal/abi/ExecutionGeometry.h:8-18`).
  - Drafted rows stop at depth 7.
  - A prompt-lookup chain is merged into the same tree: the latest earlier match of the last 3 or more
    tokens, proposing up to 15 tokens (`model/LookupProposals.hpp:32-39`). It is the only source of rows
    deeper than 7 (`metal/kernels/decode/sampling.metal:1777-1787`). Code-4096's 9.1 tokens a cycle comes
    from it (inferred).
- **How the tree grows.** It grows best-first (`draft_select_tree`, `sampling.metal:1632-1804`):
  - each node takes up to 4 children;
  - a step's estimate is sigmoid(slope·logit(q) + intercept[position]), refit online every 64 cycles
    (`model/VerifyBudget.hpp:31-53`).
- **`--verify-rows auto`** picks the row count with the most expected tokens per millisecond
  (`model/StepCosts.cpp:274-302`). The cost side is a per-kernel-class table measured at startup, plus a
  history term.
- **Tree verify.** Each row carries an ancestor bitmask and a depth (`metal/abi/VerifyTree.h:20-37`).
  - Relative attention masks block keys that are not ancestors.
  - Accepted rows' KV move from slot committed+row to committed+depth (`kernels/common/kv_path_commit.h`).
  - The short convolutions run along each row's path and rebuild state from a tape of the retained path
    (`shared/short_convolution.metal:171-204`).
  - Rejected rows are simply overwritten in the next cycle.
- **Routed experts, grouped by expert** (`ops/MoE.cpp:294-357`).
  - `moe_group_routes_verified_s0` counts routes per expert and writes tiles {expert, rows, first_row} with
    indirect grids.
  - Rows are gathered into dense M8 tiles, or M32 above 128 routes.
  - Each threadgroup dequantizes its expert's weights once and runs a small matmul over the tile's rows. An
    expert with 1–2 rows gets a register GEMV.
  - The combine sums weight × output in fp32 in route order (`shared/moe.metal:230-520`).
  - Its cost grows about 0.75 ms a row from 8 to 16 rows. tf's grows 5.7 ms.
- **The drafter's context window.**
  - Ring slot = position % 2048, with K roped at the absolute position (`kernels/common/draft_context_kv.h`).
  - Prefill captures only the last min(prompt, 2048) rows (`model/DraftContextPlan.cpp:44-46`).
  - A query at position p sees keys from p−2047 onward, plus its own 8 block rows in both directions
    (`kernels/decode/draft.metal:197-389`).
- **Selector.**
  - Top 16 candidates by raw logit.
  - Rank-256 edge from the predecessor's candidate.
  - Score 30·tanh((logit + edge)/30).
  - q is a temperature-1 softmax.
  - The greedy path takes the argmax given its parent's own candidate at each step (`sampling.metal:1099-1696`).

For tf, this splits the gap into three pieces:
- V1: verify cost per row, by grouping experts.
- DF2a: acceptance at 4K, by checking the drafter context against the window semantics above.
- DF2b: a tree with prompt lookup, after V1 lands, since it needs a tree-masked verify.
