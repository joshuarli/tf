# Inkling 2 Mini DFlash2 acceptance past the drafter's window (DF2a)

**Verdict.** tf's DFlash2 chain does not collapse at 4K through a fault of its own. Its drafter context keeps
Splash's window semantics, and it drafts what a context rebuilt from scratch drafts. The drop is the drafter's own
2,047-row window. The 4K replies cite text from the start of their prompts, which the drafter cannot see, and letting
it attend to the whole request gains nothing. Cutting a 4K prompt to fit the window brings code back from 2.76 to
5.44 committed tokens a round.

Within this lane's paths, a lookup round now drafts from an earlier occurrence of the latest tokens when that
occurrence lies before the window. It raises the 4K prompts from 2.76 / 2.69 / 3.43 to 3.12 / 2.95 / 3.57 committed
tokens a round, leaves the 1K prompts unchanged, and stays exact against plain greedy. The 1K level (4.3–5.8) is not
reached. Keeping the request's first 512 rows in the drafter's window reaches 3.43 / 3.01 / 4.15, but only as a
rebuild each round, which is an experiment and not a serving path.

Every run measured acceptance only: committed tokens a round and draft agreement, which are deterministic. No round
time in these runs is a measurement: they took no timing gate and ran beside V1.

## What the 4K replies draft against

Each draft the target judged is sorted by where its target token's 4-gram (the token and the three before it)
occurred earlier in the request: inside the drafter's 2,047 rows, only before them, or nowhere
(`run.py copies`). Baseline chain, DF1's protocol:

| Prompt | In the window | Only before it | Nowhere |
| --- | ---: | ---: | ---: |
| code-1024 | 0.97 of 332 (71%) | none | 0.76 of 135 (29%) |
| prose-1024 | 0.96 of 160 (33%) | none | 0.76 of 320 (67%) |
| structured-1024 | 0.96 of 146 (30%) | none | 0.76 of 334 (70%) |
| code-4096 | 0.91 of 65 (13%) | 0.57 of 102 (20%) | 0.62 of 337 (67%) |
| prose-4096 | 0.91 of 64 (13%) | 0.58 of 67 (13%) | 0.60 of 373 (74%) |
| structured-4096 | 0.96 of 50 (10%) | 0.58 of 53 (11%) | 0.72 of 393 (79%) |

Each cell is the share accepted, of the drafts in that class. Two things fall at 4K:

- **Copies from before the window.** 11–20% of the 4K drafts copy text the drafter cannot see, and only 57–58% of
  them are accepted. Inside the window, copies are accepted 91–96% of the time at both lengths.
- **Text that occurred nowhere earlier.** At 1K it is accepted 76% of the time on all three prompts; at 4K,
  60–72%. The instruction and the first excerpts sit at the start of each prompt, outside the window. Three
  sentences in, the code reply turns to "1. `zig/src/server/slide_correction_source.zig` — `independent` function",
  the prompt's first excerpt, 4,000 tokens back.

## Experiments that settled the cause

Committed tokens a round on the 4K prompts, q4 drafter, depth 7, 512 greedy tokens with EOS ignored. Every run's
tokens equalled plain greedy decoding in the same process.

| Run | What changes | code-4096 | prose-4096 | structured-4096 |
| --- | --- | ---: | ---: | ---: |
| `base-incremental` | nothing: the chain as on main | 2.76 | 2.69 | 3.43 |
| `rebuilt-4k` | context rebuilt every round from its 2,047 rows at positions 0–2,046 | 2.76 | 2.68 | 3.43 |
| `f32taps-4k` | prompt taps from eight-row prefill chunks, which take the decode step's f32 kernels | 2.75 | 2.70 | 3.48 |
| `wide-4k` | an 8,192-row window: the drafter sees the whole request at untrained distances | 2.86 | 2.62 | 3.26 |
| `tail1500` | each prompt cut to its chat header, instruction and last content tokens, 1,500 in all | 5.44 | 3.73 | 3.93 |
| `head128-4k` | the request's first 128 rows kept ahead of its latest 1,919, rebuilt every round | 2.94 | 3.08 | 3.76 |
| `head512-4k` | the same with the first 512 rows ahead of the latest 1,535 | 3.43 | 3.01 | 4.15 |

- **Window handling (wrap, slide, absolute positions).** `rebuilt-4k` drafts from a fresh context every round: the
  last 2,047 tap rows, in position order, absorbed at positions 0 to 2,046. Its keys sit in slots 0 to 2,046, it
  attends in place, and its positions stay below 2,055. It commits what the incremental context commits: 174 of 185,
  171 of 190 and 134 of 148 chains are identical at the same anchors, and 99% share their first draft. The others
  differ, by inference, through the bf16 rounding of keys rotated at other angles. tf's ring slots, catch-up absorbs
  and gather path are not the cause.
- **Prompt taps from the half-precision prefill.** `f32taps-4k` prefills a twin cache in eight-row chunks, which
  take the f32 expert, attention and projection kernels a decode step takes, and feeds the drafter those taps. The
  target's own prefill is unchanged. Acceptance moves by under 0.05 a round. Prompt-tap precision is not the cause.
- **RoPE at large positions.** `rope.swift` runs `tf_full_rope` from `zig/kernels/metal/core/draft_ops.metal` (fast
  math, as compiled) at positions 0 to 33,000 against float64 rotations (`rope.txt`). The angle error is at most
  0.0028 rad below 4,096 and 0.0052 rad at 16K–33K, with a mean of 0.0005–0.0006 rad at every length. Precise math
  gives the same errors, so they are the bf16 rounding of the output, not the angle.
- **More context at untrained distances.** `wide-4k` raises the window to 8,192 rows, so the drafter attends to
  every row of the request. It gains nothing (−5% to +3%): the checkpoint (`sliding_window` 2048) does not use rows
  beyond the distances it was trained on.
- **The content the window drops.** `tail1500` keeps each 4K prompt's chat header and instruction paragraph and its
  last content tokens, 1,500 in all, so the prompt and the 512-token reply fit the window. Code returns to 5.44,
  near code-1024's 5.81. Its reply now copies in-window text for 76% of its drafts, at 97%. Prose and structured
  rise to 3.73 and 3.93.
- **The same rows at trained distances.** `head128-4k` and `head512-4k` (modes `first128` and `first512`) rebuild
  the context every round as the request's first 128 or 512 rows followed by its latest rows, at positions 0 to
  2,046, so every row sits within the trained distance. With 512 rows kept, copies from before the window are
  accepted 84–94% of the time instead of 57–58%, and text that occurred nowhere earlier 62–76% instead of
  60–72%.

## Semantics against Splash

Splash 1.7.2 (`2afc7337aacc`, Apache-2.0), read only, against tf on main `933078894`:

| Item | Splash | tf |
| --- | --- | --- |
| Taps | rows entering applications 5, 23, 41 and the final norm (48), bf16 | `taps.zig`, ids 5/23/41/48, f32 rows rounded to nearest even |
| Context row | RMSNorm(fc(taps)); each layer's K/V from it | `execution.zig:64-68` |
| Keys | k-norm, then RoPE at the absolute position | `execution.zig:68` |
| Ring slot | position % 2048 | `tap_ring.zig:23`, `runtime_context.zig:46` |
| Prompt rows | the last min(prompt, 2048) | the ring keeps 2,048; `flush` builds the last 2,047 (`runtime_model.zig:104`), all a query at the anchor sees |
| Attention | a query at p sees keys at p − 2,047 or later, and all 8 block rows | `fillMask` (`runtime_context.zig:147`) |
| Anchor and mask rows | token table through the embedding norm; mask 200,056 | `runtime_callbacks.zig:53-63`, `execution.zig:117` |
| Candidates | top 16 by raw logit, tokens past 200,058 excluded | `dflash.zig:165` sets `candidates` to 200,058 |
| Logits | f32 | bf16 from the borrowed head kernel |

The last row is the only difference found. It can change a pick only where two candidates' scores, logit plus
edge, lie within one bf16 rounding step of each other, at 1K as at 4K. It was not measured: an f32 head and top-k
would change shared kernels outside this lane.

**Selector.** `Drafter.propose` asks `proposeOptions` for `.children = 1, .edge = 1`. With one child, `expand`
(`selector.zig:54`) keeps only the best candidate at each depth. Its score is `30·tanh((logit + edge·E)/30)` with
the parent's own token as predecessor (the anchor at depth 0), then divided by the temperature (1 without noise) and
by `tau` (1.5). Neither division and no softmax changes the order, so the chain is Splash's greedy path through the
lattice. Flint's default edge of 0.6 is not used here.

## Lookup rounds

`Lookup` (`dflash_spec.zig`) indexes a request's prompt and reply by trigram. For the latest token it finds the
longest earlier match of its suffix, 3 to 64 tokens, the most recent of that length, among the 4,096 latest earlier
occurrences of its last three tokens. Repetitive prompts need that reach: structured-4096 repeats one trigram 758
times, and the GPU test's agent prompt 116 times.

When the match's occurrence ends before the drafter's window (`beyondWindow`), the round drafts up to 7 tokens that
followed it there, through the same 8-row grouped verify, and the drafter does not run. Every other round drafts the
drafter's chain. The rule draws nothing at 1K, where every match is inside the window.

Drafting from every match, inside the window too, would lose at 1K. At the baseline's own anchors on prose-1024 and
structured-1024, those lookups accept 3.27 and 2.93 drafts a round where the drafter's chains accept 3.52 and 3.31:
the drafter already sees those matches.

Committed tokens a round, before (`base-incremental`, equal to DF1's `q4-a`, `q4-b` and `q4-c`) and after
(`after-lookup`):

| Prompt | Before | After | Lookup rounds | Drafts accepted a lookup round | Drafts accepted a drafter round |
| --- | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 5.81 | 5.81 | 0 of 88 | | 4.81 |
| prose-1024 | 4.52 | 4.52 | 0 of 113 | | 3.52 |
| structured-1024 | 4.33 | 4.33 | 0 of 118 | | 3.33 |
| code-4096 | 2.76 | 3.12 | 27 of 164 | 3.30 | 1.88 |
| prose-4096 | 2.69 | 2.95 | 15 of 173 | 4.80 | 1.68 |
| structured-4096 | 3.43 | 3.57 | 13 of 143 | 3.31 | 2.50 |

The 1K rows are the chain's own rounds, token for token.

Copies from before the window are now accepted 92%, 95% and 90% of the time on the three 4K prompts, from 57–58%.

Per quarter of each 4K reply (committed tokens a round over the rounds that emitted that quarter's tokens):

| Prompt | Before | After |
| --- | --- | --- |
| code-4096 | 2.52 / 2.78 / 2.76 / 3.05 | 2.85 / 3.05 / 3.21 / 3.43 |
| prose-4096 | 2.41 / 2.65 / 3.37 / 2.52 | 3.51 / 2.70 / 3.37 / 2.47 |
| structured-4096 | 2.91 / 3.55 / 3.38 / 4.10 | 3.88 / 3.21 / 3.29 / 4.10 |

With the first 512 rows kept as well (`head512-lookup-4k`), the 4K prompts commit 3.43, 2.99 and 3.96: no more than
the kept rows alone. Those rows already show the copied text, and the lookup's window test does not know it.

## Correctness

- `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0 on `491045e3a`
  (`.work/suite/20261124T025802`). `inkling-host` passed 10/10 steps and 35/35 tests. `inkling-gpu` passed 17/17
  steps and 137/137 tests with none skipped: SV1's 133 and the 4 below.
- `zig build test-flint-draft-host` (47/47 steps, 86/86 tests) and `zig build test-qwen27` (48/48 steps, 200/205 with
  5 skipped) exited 0 on `491045e3a`, the counts DF1 recorded. The shared runtime (`zig/src/families/qwen27/dflash/`)
  is unchanged, so Flint's DFlash2 output is unchanged.
- `zig build native -Dcpu=apple_m1` exited 0 on `491045e3a`. `tools/zig/lean_new.py` against main `933078894` finds
  0 new problems (257 on both).
- Every DF2 run's tokens equal plain greedy decoding in the same process (`equal_plain`, 0 differences).
- New tests in `dflash_test.zig`:
  - host: a lookup takes the longest earlier suffix of three or more tokens, the most recent of that length; it finds
    the longest match behind 300 shorter recent ones; it drafts only from an occurrence that ends before a drafter's
    window;
  - GPU, oQ4e with the real drafter: the agent fixture (2,572 tokens) followed by its own first 40 tokens. The
    opening round drafts the 7 tokens that followed them 2,572 positions back, and speculation over 96 tokens equals
    plain greedy decoding, its caches bit for bit, and the drafter's committed rows. 8 of its 21 rounds drafted from
    before the window and accepted 34 drafts.
- The served binding (`zig/src/native/inkling2_loop_spec.zig`) and `dflash.zig`'s target binding are unchanged.
  `Drafter.Options.window` serves only the runner's `window8192` experiment.

## Two misleading comparisons

- **The 4K prompts were not Splash's.** DF1's prompt builder cuts each prompt to 4,096 tokens, so code-4096 drops
  the last 1,184 tokens of its file (prose-4096 543, structured-4096 107). Splash served the whole rendered file:
  5,280 tokens for code-4096. SV1's server rendered the same 5,280 tokens and committed 3.26–3.52 a round on it.
- **Splash's cycles are not chains.** 9.1 tokens a cycle on code-4096 is more than an anchor and seven drafts can
  commit. Splash verifies up to 16 rows a lane (`SPLASH_TARGET_VERIFY_ROWS`, `runtime/metal/abi/ExecutionGeometry.h`)
  of a tree that holds the drafter's 7 proposals and their alternatives beside a prompt-lookup chain
  (`runtime/model/LookupProposals.hpp`), the only rows deeper than 7. Its drafter window is tf's. The 1.3× to 3.3×
  gap S1 reported mixes that window, which both engines share, with Splash's tree and lookup, which DF2b owns.

## Conditions

- Every model process ran through `campaign/heavy.py` (the GPU lock and the 28 GB cap) with
  `INKLING_LOAD_LIMIT_GB=26`. Charged peaks were 18.2–22.7 GB. The highest, `f32taps-4k`, holds a twin cache with a
  second 4,096-token target state.
- The runs took no GPU or CPU gate. Acceptance does not depend on timing, and V1 timed its verify between them under
  the same lock.

## Builds

Each record's first line names its runner and sha256. The lane's code changed only by adding modes, so the chain's
arithmetic is the same in all five builds. `after-lookup`, on the last, repeats `base-incremental` on the 1K prompts
round for round, and an earlier lookup run on `ad3088d3d94e` on the 4K prompts (all 480 rounds, drafts included).

| Runner sha256 | Code | Records |
| --- | --- | --- |
| `a4f063f3e0a4` | drafts recorded, `rebuilt` and `f32_taps` | `base-incremental`, `rebuilt-4k` |
| `92dd4fc0b436` | and lookup rounds | `f32taps-4k`, `tail1500` |
| `ad3088d3d94e` | and the first-rows modes, then named `head128` and `head512` | `head128-4k`, `head512-4k` |
| `bd02c5026611` | and the 8,192-row window, then named `wide` | `wide-4k` |
| `292be660830d` | this lane's code: modes named `first128`, `first512` and `window8192`, lookup scans 4,096 earlier ends | `after-lookup`, `head512-lookup-4k` |

## Reproduction

From the worktree, with the runner built by
`zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix .work/tmp/df2/final`:

```sh
R=.work/tmp/df2/final/bin/tf-inkling2-loop-run
D=campaign/kernels/inkling-df2-accept-runs
P=campaign/kernels/inkling-dflash2-runs/prompts.json
python3 -I $D/run.py bench $R $P base-incremental
python3 -I $D/run.py bench $R $P after-lookup --drafts lookup
python3 -I $D/run.py bench $R $D/prompts-4k.json rebuilt-4k --context rebuilt
python3 -I $D/run.py bench $R $D/prompts-4k.json f32taps-4k --context f32_taps
python3 -I $D/run.py bench $R $D/prompts-4k.json wide-4k --context window8192
python3 -I $D/run.py tails $R $D/prompts-tail1500.json --total 1500
python3 -I $D/run.py bench $R $D/prompts-tail1500.json tail1500
python3 -I $D/run.py bench $R $D/prompts-4k.json head128-4k --context first128
python3 -I $D/run.py bench $R $D/prompts-4k.json head512-4k --context first512
python3 -I $D/run.py bench $R $D/prompts-4k.json head512-lookup-4k --context first512 --drafts lookup
python3 -I $D/run.py rope
python3 -I $D/run.py accept base-incremental after-lookup
python3 -I $D/run.py copies base-incremental $P
```

`prompts-4k.json` holds DF1's three 4K prompts. The GPU test alone:

```sh
TF_INKLING2_LOOP_OQ4E_DIR=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd \
TF_INKLING2_DFLASH2_DIR=$HOME/models/Inkling-2-Mini-DFlash2@4f9088ea66c2 INKLING_LOAD_LIMIT_GB=26 \
python3 campaign/heavy.py -- zig build test-inkling2-loop-dflash2 -Dcpu=apple_m1 -j1 --summary all
```

## Open problems

- **Served lookup.** The server's rounds (`Draft.round` in `inkling2_loop_spec.zig`) never see the request's tokens.
  Serving lookup rounds needs `inkling2_loop_host.zig` to pass them, outside this lane.
- **Keeping the first rows.** It is the larger lever on these prompts (+12% to +24% at 4K) but costs a 2,047-row
  rebuild a round here. A serving form would keep the first rows' normed keys unrotated, rotate them each round to
  the positions just before the window's latest rows, and attend to [first rows, latest rows, block] in the shared
  runtime. It departs from the checkpoint's sliding window and from Splash, and these three prompts all put their
  instruction first, so it needs prompts that do not, such as agent sessions, before it is worth building.
- **The 1K level at 4K.** No change measured here reaches it within an 8-row chain. DF2b's tree with lookup chains
  of up to 15 tokens is the path Splash takes.
- **bf16 drafter logits.** Splash ranks candidates on f32 logits; tf on bf16. Not measured.
