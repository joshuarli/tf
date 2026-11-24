# Inkling 2 Mini tree verify with lookup chains (DF2b)

**Verdict.** A round now verifies a tree of up to 16 rows instead of DF1's 8-row chain. Every verified row keeps its
path's one-row step bit for bit, so greedy speculation still emits plain greedy decoding's tokens: in all twelve timed
processes the replies on DF1's six dev prompts equal plain greedy, DF2b's equal main's, and DF2b's trees are the same in
every process. Each round merges the drafter's best-first tree with the request's lookup chain (up to 15 tokens) and
picks its rows from a fixed table of measured verify costs. In six balanced process pairs over two back-to-back
sessions, DF2b's speculative decode beats main's on every prompt: 1.11× on code-1024 (pairs 0.88–1.20), 1.11× on
prose-1024, 1.09× on structured-1024, and 1.29×, 1.35× and 1.24× on the three 4K prompts. The geometric mean is 1.193×
with a 95% interval over process pairs of 1.140–1.249, which meets Decision 11's promotion rule. DF2b reaches 150.8 /
111.4 / 105.3 tok/s at 1K and 78.0 / 79.0 / 93.4 at 4K, against main's 132.3 / 101.8 / 96.6 and 60.1 / 58.4 / 75.2.
Served speculation runs the same tree rounds over the request's prompt and generated tokens. Its replies equal draft-off
replies through prompt reuse, the store, a fork, two lanes and an EOS inside a lookup round. No prompt meets a Decision
17 budget: the best, code-1024, takes 6.6 ms a committed token against 5. Acceptance is now the bound. The 96% of rounds
that run the drafter verify at most eight rows and commit at best 5.28 tokens a round, in 45.0 ms at eight rows: 117
tok/s. Only the 16-row rounds behind long lookup chains reach 195 tok/s.

**Base.** Main is `50b4a681a`, whose Zig equals V2's merge `3881b3d8c`. P4's later merge `3f0054737` changes only
prefill files and `zig/build/inkling2_loop.zig`, so decode is timed against `50b4a681a`. No file changed on both
sides, and `git merge-tree` finds no conflict with main `0d4d2f07e`.

## Round pairs

Each process runs DF1's six prompts (`prompts.json`, DF1's file unchanged) with the oQ4e target and the q4 drafter:
512 greedy tokens with EOS ignored, plain and speculative from the same prefill state, alternating which goes first.
Main runs DF1's chain (seven drafts, an eight-row verify); DF2b runs `tree`. The twelve processes ran in two
back-to-back sessions, main first in pairs a, c and e and DF2b first in b, d and f. Spec tok/s counts the 511 tokens
after prefill's first. Before is main, after is DF2b; tok/s is the median over the six processes, range in brackets
(`run.py pairs`).

| Prompt | Committed/round before | after | Spec tok/s before | after | After/before per pair | Geomean |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| code-1024 | 5.81 | 7.51 | 132.3 (129.2–153.2) | 150.8 (134.2–183.8) | 0.88, 1.12, 1.17, 1.20, 1.15, 1.16 | 1.11 |
| prose-1024 | 4.52 | 4.52 | 101.8 (98.5–103.1) | 111.4 (101.6–131.0) | 1.03, 1.05, 1.08, 1.27, 1.12, 1.13 | 1.11 |
| structured-1024 | 4.33 | 4.26 | 96.6 (91.4–98.4) | 105.3 (96.2–106.7) | 1.04, 1.06, 1.08, 1.17, 1.09, 1.09 | 1.09 |
| code-4096 | 2.76 | 2.95 | 60.1 (57.9–61.0) | 78.0 (71.8–79.3) | 1.24, 1.26, 1.31, 1.33, 1.32, 1.28 | 1.29 |
| prose-4096 | 2.69 | 2.92 | 58.4 (57.5–59.6) | 79.0 (74.0–81.4) | 1.29, 1.32, 1.38, 1.37, 1.39, 1.33 | 1.35 |
| structured-4096 | 3.43 | 3.73 | 75.2 (69.8–75.8) | 93.4 (87.7–94.4) | 1.26, 1.23, 1.25, 1.26, 1.26, 1.21 | 1.24 |

Each process pair's geometric mean over the six prompts is 1.111, 1.168, 1.206, 1.265, 1.218 and 1.200. Their geometric
mean is 1.193, with a 95% interval of 1.140–1.249 (t, five degrees of freedom). Over all 36 prompt-pairs the interval is
1.153–1.235. The one pair below 1.0, code-1024's first, reflects its two processes' states more than their builds.
code-1024 is every process's first prompt, and its 8-row verify took 34 ms in three processes (`main-a`, `main-d`,
`df2b-d`) and 39–46 ms in the other nine. Pair a put main's fast process against DF2b's slowest (8 rows at 46.0 ms, 16
rows at 84.3), while pair d, both fast, gives 1.20.

### Where the time goes

Median over the six processes of each process's mean, range in brackets (`run.py roundtable`). Phases are mean ms a
round. **Verify** holds the grouped forward, the GPU argmax, the acceptance walk and the commit in one command
buffer; **Host** is the taps' absorption and bookkeeping.

DF2b:

| Prompt | Equal | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | yes | 68 | 7.51 | 49.9 (40.9–56.0) | 74.6 (58.7–86.2) | 3.39 | 46.4 | 0.02 | 150.8 (134.2–183.8) | 16.46 (15.73–18.02) | 60.8 (55.5–63.6) | 2.51 (2.42–2.89) |
| prose-1024 | yes | 113 | 4.52 | 40.6 (34.5–44.5) | 45.7 (38.6–50.8) | 4.64 | 35.9 | 0.02 | 111.4 (101.6–131.0) | 16.47 (15.84–18.41) | 60.7 (54.3–63.1) | 1.86 (1.82–2.08) |
| structured-1024 | yes | 120 | 4.26 | 40.4 (39.9–44.3) | 45.7 (44.8–51.5) | 4.50 | 35.9 | 0.01 | 105.3 (96.2–106.7) | 17.07 (16.43–18.18) | 58.6 (55.0–60.9) | 1.77 (1.75–1.81) |
| code-4096 | yes | 173 | 2.95 | 37.9 (37.2–41.1) | 44.9 (44.2–49.3) | 4.69 | 33.2 | 0.01 | 78.0 (71.8–79.3) | 16.91 (16.48–18.41) | 59.1 (54.3–60.7) | 1.32 (1.31–1.34) |
| prose-4096 | yes | 175 | 2.92 | 37.0 (35.9–39.5) | 44.6 (43.7–47.6) | 4.61 | 32.3 | 0.01 | 79.0 (74.0–81.4) | 17.23 (16.74–18.41) | 58.1 (54.3–59.7) | 1.36 (1.34–1.39) |
| structured-4096 | yes | 137 | 3.73 | 39.9 (39.5–42.5) | 47.2 (46.1–50.3) | 4.70 | 35.2 | 0.01 | 93.4 (87.7–94.4) | 17.25 (16.63–18.42) | 58.0 (54.3–60.1) | 1.60 (1.57–1.64) |

Main:

| Prompt | Equal | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | yes | 88 | 5.81 | 43.9 (37.9–45.0) | 46.1 (38.9–47.4) | 4.43 | 39.5 | 0.01 | 132.3 (129.2–153.2) | 16.22 (15.77–16.54) | 61.7 (60.5–63.4) | 2.16 (2.11–2.42) |
| prose-1024 | yes | 113 | 4.52 | 44.4 (43.9–45.9) | 46.9 (46.0–48.8) | 4.56 | 39.9 | 0.01 | 101.8 (98.5–103.1) | 16.47 (15.76–16.82) | 60.7 (59.4–63.4) | 1.67 (1.57–1.71) |
| structured-1024 | yes | 118 | 4.33 | 44.8 (44.0–47.4) | 46.8 (45.8–49.3) | 4.57 | 40.3 | 0.00 | 96.6 (91.4–98.4) | 16.60 (16.32–16.94) | 60.3 (59.0–61.3) | 1.60 (1.54–1.62) |
| code-4096 | yes | 185 | 2.76 | 46.0 (45.3–47.7) | 48.4 (47.3–49.6) | 4.68 | 41.3 | 0.00 | 60.1 (57.9–61.0) | 16.76 (16.37–17.29) | 59.7 (57.8–61.1) | 1.00 (0.97–1.04) |
| prose-4096 | yes | 190 | 2.69 | 46.1 (45.1–46.8) | 48.6 (47.1–49.4) | 4.71 | 41.4 | 0.00 | 58.4 (57.5–59.6) | 17.02 (16.61–18.34) | 58.8 (54.5–60.2) | 1.00 (0.97–1.05) |
| structured-4096 | yes | 149 | 3.43 | 45.6 (45.2–49.1) | 48.4 (48.0–52.2) | 4.64 | 40.9 | 0.00 | 75.2 (69.8–75.8) | 16.82 (16.52–18.16) | 59.5 (55.1–60.5) | 1.26 (1.25–1.29) |

The gain has two sources: fewer rounds (more tokens a round) and cheaper rounds (fewer rows). At 4K, cheaper rounds
give 61–74% of it. When acceptance is low the plan verifies 4.95–5.85 rows on average instead of eight, and the
lookup chain adds 0.19–0.30 tokens a round; the drafter's tree alone commits what the chain does (Ablations). On
prose-1024 and structured-1024 all of the gain is cheaper rounds. On code-1024 all of it is more tokens a round (7.51
against 5.81, from lookup chains), and its rounds cost more (49.9 against 43.9 ms) because 18 of its 68 rounds verify
16 rows. `run.py same` finds every DF2b round's tree, the round
counts and the output hashes identical in all six DF2b processes, and the six output hashes (`e694b52ab532`,
`03832d4821d9`, `cffafa0bf13f`, `bcec689a428e`, `46e6c4217d25`, `a850b54dc998`) equal main's.

## Decision 17 budgets

Decision 17 allows at most 20, 25, 30 and 40 ms a round at 4, 5, 6 and 8 mean committed tokens: 5 ms a committed
token, which is also 200 tok/s (`run.py budgets`, six processes each):

| Prompt | DF2b committed/round | Round ms | Budget at that commitment ms | ms a committed token | Main's ms a committed token | Meets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| code-1024 | 7.51 | 49.9 | 37.6 | 6.63 | 7.56 | no |
| prose-1024 | 4.52 | 40.6 | 22.6 | 8.98 | 9.82 | no |
| structured-1024 | 4.26 | 40.4 | 21.3 | 9.49 | 10.35 | no |
| code-4096 | 2.95 | 37.9 | 14.8 | 12.83 | 16.64 | no |
| prose-4096 | 2.92 | 37.0 | 14.6 | 12.66 | 17.13 | no |
| structured-4096 | 3.73 | 39.9 | 18.6 | 10.70 | 13.30 | no |

No prompt meets its budget.

## What bounds it next

All six DF2b processes' rounds by verified rows (`run.py costs`): sustained verify, tokens committed and the whole
round.

| Rows | Rounds | Drafter ran | Verify ms mean | Verify p50 | Committed | Round ms | Committed a round-ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36 | 24 | 19.11 | 19.08 | 1.00 | 22.3 | 0.045 |
| 2 | 336 | 336 | 23.24 | 23.33 | 1.79 | 28.0 | 0.064 |
| 3 | 624 | 624 | 27.43 | 27.32 | 2.16 | 32.2 | 0.067 |
| 4 | 282 | 282 | 30.24 | 30.16 | 2.55 | 35.0 | 0.073 |
| 5 | 768 | 768 | 32.97 | 32.90 | 2.98 | 37.7 | 0.079 |
| 6 | 630 | 630 | 35.38 | 35.43 | 3.40 | 40.1 | 0.085 |
| 7 | 696 | 696 | 38.00 | 37.84 | 3.96 | 42.7 | 0.093 |
| 8 | 1,182 | 1,176 | 40.36 | 40.17 | 5.28 | 45.0 | 0.117 |
| 16 | 162 | 0 | 71.89 | 71.25 | 14.00 | 71.9 | 0.195 |

The drafter ran in 4,536 rounds at 4.72 ms mean. No round chose 9 to 15 rows. The cost table steps 10.5 ms at nine
rows, where the wider kernels begin, and then adds 2.3 ms a row, so rows past eight pay only in bulk.

- **Acceptance in drafter rounds.** 96% of rounds run the drafter. They verify one to eight rows and commit at most
  0.117 tokens a millisecond, at eight rows; 200 tok/s needs 0.2. At 45 ms an 8-row round would have to commit 9
  tokens, more than its eight rows can hold. At 5.28 tokens it would have to take 26 ms, about a one-row step plus
  the draft (19.1 + 4.7 ms in rounds), so a faster verify alone would have to make eight rows nearly as cheap as one.
  At 4K, DF2a traced the low acceptance (2.9–3.7 tokens) to the drafter's 2,047-row window: the replies cite their
  prompts' openings, which lie outside it.
- **Rows past eight cost more each than rows two to eight.** The 16-row rounds verify only lookup chains with the
  drafter skipped. They commit 14.0 tokens in 71.9 ms, 195 tok/s, but are 3% of rounds. Rested, rows 9 to 16 add
  26.3 ms (63.0 against 36.7), 3.3 ms a row against 2.7 for rows 2 to 8. The dense kernels' time follows the rows,
  not the weights (below): two 8-row tiles sharing one decode of each weight word cost 2.0–2.25× one 8-row tile.
- **The 8-row verify itself**: 40.4 ms in rounds and 36.7 ms rested, against Splash's 26.3 ms (Decision 18). V2
  traced the largest part of its growth over one row, 12.5 ms, to the routed experts, most of it the bytes of the
  30.7 distinct experts that eight rows route. Decision 18 permits a new arithmetic if one serves the one-row step,
  the batch step and the grouped verify.

## What changed

All in `zig/src/families/inkling2_loop/` and the served Inkling host files on `lane/df2b-tree`; no shared core, other
family, prefill file or build file.

- **Tree verify** (`group.zig`, `kernels/verify.metal`). A grouped verify's stream is a tree under its newest token:
  `Verifier.tree(caches, sizes, parents)` takes each draft's parent row, `group` keeps chains (each row's parent the
  row before it), and every row sits at its depth's position. Each row carries its path, and the verify's twins read
  only that path from the captures:
  - `i2v_split` reads the key at position `j` past the cache's length `gpos` from row `path[j - gpos]`, so a row's
    keys, values and split boundaries are its path's, decoded token by token;
  - `i2v_conv` reads the position `k` before a row from row `back[k - 1]`, its ancestor `k` levels up, and from the
    cache's ring for positions before the stream's first row;
  - a commit (`commit`, or `decide(.commit)`'s fused round) takes the deepest accepted row and writes only its path's
    keys, values and last four convolution inputs, at the positions decoding the path writes them;
  - `i2v_accept` walks the tree on the GPU: from the first row, the lowest-numbered child whose token is its parent's
    pick (`group.walk` is the same rule on the host).
- **Sixteen rows** (`rows.zig`, `kernels/rows.metal`, `group.scratch`). Rows 9 to 16 run two 8-row tiles against one
  decode of each weight word (`i2t_dense16_8_2`, `i2t_projections16_8_2`) and expert kernels whose members span 128
  slots (`i2t_gate_up16`, `i2t_down16`); each row's chains are the one-row kernels', as V2's tiles. Rows 1 to 8 keep
  V2's kernels. `group.scratch` builds the 16-row batch scratch with the prefill scratch's half path off, because
  chunks of more than eight rows otherwise take the prefill matrix kernels. The native NVFP4/MXFP8 checkpoint's bf16
  and MXFP8 projections leave the small-row kernels past eight rows, so its verifier limit is 8; oQ4e's is 16. To keep
  `rows.metal` under the lean rule's 600 lines, the two-row head kernel lost its never-instantiated 4-bit path and the
  tile kernels share each parameter list between definition and instantiation.
- **The plan** (`verify.zig`: `Lookup`, `Tree`, `Plan`; DF2a's lookup index moved here so the served host reaches it).
  Each round:
  1. the lookup index proposes up to 15 tokens that followed the longest earlier match of the newest tokens;
  2. unless that chain alone likely commits eight drafts (estimate at least 0.7, `lookupSuffices`), the drafter's
     selector draws a best-first tree of 32 nodes with at most 4 children (`proposeTree`, edge weight 1);
  3. the chain merges into the drafter's tree where tokens agree, and each node gets an estimate of its acceptance
     once its parent is accepted (next section), siblings capped at 0.98 in all;
  4. nodes rank by path estimate (the product along the path), and the round verifies the best `rows - 1` for the
     `rows` that maximise expected committed tokens a millisecond under a fixed cost table (1 to 16 rows) plus 4.5 ms
     of draft when the drafter ran.

  The cost table is the rested verify measured here (below): 17.9, 21.8, 28.0, 36.7, 54.0 and 63.0 ms at 1, 2, 4, 8,
  12 and 16 rows, with the step at nine rows where the wider kernels begin. No time measured in a round feeds the plan,
  so the same request makes the same trees in any process.
- **Served lookup** (`zig/src/native/inkling2_loop_spec.zig`, `inkling2_loop_host.zig`). The host passes the request's
  prompt and generated tokens to each round; a lane's lookup index and plan start over at each admission and catch up
  from those tokens (at most about 1.1 MiB of host memory a lane at 32K tokens). A round offers up to 15 drafts and
  runs one fused verify that commits the accepted path, and the drafter absorbs the path's taps. A lookup failure
  leaves the round without a chain; the reply's telemetry counts lookup rows and those kept.
- **Measurement** (`dflash_run.zig`, `kbench.zig`). `dflash ... tree` (and `drafter_tree`, without the chain, and
  `tree,nodes=N,children=N,skip=P`) runs the tree rounds; `dflash-widths` reaches 12 and 16 rows and takes a list of
  widths; `dflash-kernels` times 1 to 16 rows.

## The plan's estimates

The priors come from untimed tree runs on the holdout prompts (`prompts-holdout.json`: DF1's rule over the holdout
corpus files; the builder reproduces DF1's dev file byte for byte). The dev prompts measured below were not used. Each
request then learns from its own rounds.

Each draft is one of four kinds, with its own table (`run.py calibrate calib-holdout-a,calib-holdout-b`, drafts under
an accepted parent). A prior is a bin's rate shrunk toward its column's pooled rate by 8 pseudo-trials, made
non-decreasing in q or match length, and capped at 0.995:

| Kind | Bins | Holdout acceptance |
| --- | --- | --- |
| the drafter's alone | depth 0–6 × the drafter's probability q (10 bins) | q ≥ 0.9: 1,796 of 1,873, 0.94–0.98 at each depth |
| the chain's alone, in a round the drafter ran | place 0–14 × match length (3, 4–5, 6–9, 10–17, 18+) | place 0: 14 of 50, 0 of 12 at match 3 |
| the chain's, any round | the same | the prior for rounds that skip the drafter |
| both agreeing | q | q ≥ 0.9: 661 of 661; 739 of 776 at any q |

A chain token the drafter's tree also holds was right 95% of the time; one it lacks was right 14 of 50 times at the
root, while past the drafter's seven positions chain tokens of matches of 18 or more were right 201 of 205 times.

Three holdout passes, committed tokens a round, every reply equal to plain greedy:

| Pass | Plan | code-1024 | prose-1024 | structured-1024 | code-4096 | prose-4096 | structured-4096 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `calib-holdout-a` | one table a source, rough priors, every row learned | 8.24 | 3.93 | 5.21 | 3.30 | 3.34 | 3.96 |
| `calib-holdout-b` | four tables fitted on a, every row learned, drafter skip | 7.00 | 3.70 | 4.22 | 3.10 | 3.10 | 3.41 |
| `calib-holdout-c` | refitted on a and b from drafts under accepted parents only | 8.38 | 3.93 | 5.16 | 3.32 | 3.38 | 3.99 |

Pass b fell back because a draft whose parent the target rejected is judged against the target's continuation of a
path it would not take; counting those biased every estimate past the first draft low, and the plan verified too few
rows. Learning only from drafts under accepted parents (`83f9f2633`) fixed it. Priced on the final cost table
(`run.py modeled calib-holdout-a,calib-holdout-b,calib-holdout-c --table df2b-a`), pass c takes less modeled time a
token than pass a on five of the six prompts and ties on structured-1024 (8.02 against 8.01 ms).

## Correctness

The contract is V1's and V2's, extended to trees: every row of a grouped verify has its path's one-row batch steps'
logits, picks and taps bit for bit, and a commit leaves the caches that decoding the accepted path leaves, so greedy
speculation equals greedy decoding whatever tree a round verifies. Every check is bit equality.

- **Trees on the checkpoints** (`group_test.zig`, both checkpoints): five shapes from 5 to 16 rows (16 rows with three
  drafts under the newest token and a 12-deep branch, a 16-row chain, a 9-row two-level tree, an 8-row tree and a
  5-row fan), starting five positions before the local window's wrap. Every row's logits, pick and taps equal its
  path's one-row steps (54 rows on oQ4e; 37 on the native checkpoint at its 8-row limit). Committing the accepted path,
  the last row or another leaf (15 commits) leaves the caches of decoding that path, and the next step equals; the fused
  round picks, accepts and commits what forward, picks, `walk` and commit do.
- **Tiles at 9 to 16 rows** (`rows_test.zig`, `attention_test.zig`): dense tiles against `i2a_dense` at 2–16 rows in
  four shapes, with and without the cap and with offset inputs (120 cases); projection tiles against
  `i2a_batch_projections` at 3–16 rows in both layouts (28); NVFP4 tiles against `i2p_gu` and `i2p_down` at 1–16 rows
  of random picks over 12 experts and of every row on the same 8 (63). Every output's bytes are equal.
- **Speculation** (`dflash_test.zig`, oQ4e with the real drafter): tree rounds with lookup on the fixture prompts equal
  plain greedy decoding and its caches (1,024 tokens in 260 rounds; 4 verify past eight rows and 10 take a later
  branch); forced trees accept a 15-row path, none, or a later branch, and stop at the limit or an EOS inside a tree,
  keeping plain's caches; the agent prompt followed by its own first 40 tokens, whose matches lie before the drafter's
  window, merges the chain and equals plain greedy. DF2a's chain and lookup tests pass unchanged.
- **Served** (`inkling2_loop_spec_test.zig`): every drafted reply equals its draft-off reply through a fresh prompt, an
  EOS inside a round after a store hit, a 2,572-token prompt, own-slot reuse, a fork beside its decoding source, two
  lanes, a prompt that repeats its own opening (82 lookup rows, 49 kept) and an EOS inside one of its lookup rounds; a
  host without the drafter replays the cold replies equally. Every request but the 10-token EOS reply committed lookup
  rows.
- **The plan** (`verify_test.zig`): merging, row choice, learning and the drafter skip on fixed inputs.
- **Every timed run** verified exactness on the way: all 120 grouped widths in the widths runs equal their one-row
  steps, and every round-table reply equals plain greedy, with DF2b's tokens equal to main's on all six prompts.

On the final code (`d95cc3fe0`'s, at `2ca7c7e39`), `python3 -I campaign/suite/suite.py run --names
inkling-host,inkling-gpu` exited 0 (`.work/suite/20261124T111846`, 11:18–11:41): inkling-host 10/10 steps and 35/35
tests, inkling-gpu 17/17 steps and 169/169 tests with none skipped (`suite-summary.json`, `suite-inkling-host.txt`,
`suite-inkling-gpu.txt`). It had also passed on `83f9f2633`, before the cost table's refit. On the same code `zig
build test-flint-draft-host` (47/47 steps, 86/86 tests), `zig build test-qwen27` (48/48 steps, 200/205 with 5
skipped) and `zig build native -Dcpu=apple_m1` exited 0; the flint and qwen27 counts are DF2a's. The shared DFlash2
runtime is unchanged, so Flint's DFlash2 output is unchanged. `tools/zig/lean_new.py` against `50b4a681a` finds 0 new
problems (256 on both).

## Verify cost by width

Rested: `dflash-widths`, nine repetitions of every width on every prompt, each process after 60 s idle inside the GPU
lock. A cell takes the prompts' p50 at one context in the processes named, the range over prompts and processes in
brackets: their mean in the comparison (`run.py widthcompare`), their median in DF2b's table (`run.py widthtable`).
Every grouped width's rows were bit-equal to their one-row steps: 120 of 120 cells in DF2b's four processes and 96 of 96
in main's.

At main's widths (`widths-main-c`, `-d` against `widths-df2b-c`, `-d`, DF2b restricted to main's widths with
`--cells 0,1,2,4,8`; the processes ran DF2b, main, main, DF2b):

| Context | Path | Width | Main p50 | DF2b p50 | Change | DF2b GPU p50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | batch step | 1 | 15.4 (15.3–15.8) | 15.6 (15.3–16.4) | 1.01× | 15.0 |
| 1,024 | grouped | 1 | 15.7 (15.6–16.2) | 15.9 (15.6–16.7) | 1.01× | 15.1 |
| 1,024 | grouped | 2 | 19.6 (19.4–20.0) | 19.9 (19.6–20.7) | 1.01× | 19.0 |
| 1,024 | grouped | 4 | 25.2 (25.0–25.6) | 25.6 (25.1–26.4) | 1.02× | 24.7 |
| 1,024 | grouped | 8 | 33.6 (33.0–33.8) | 34.0 (33.0–35.0) | 1.01× | 33.1 |
| 4,096 | batch step | 1 | 16.6 (15.6–17.8) | 16.7 (15.7–17.4) | 1.01× | 16.1 |
| 4,096 | grouped | 1 | 16.9 (16.0–18.1) | 16.8 (16.0–17.7) | 0.99× | 16.0 |
| 4,096 | grouped | 2 | 20.9 (19.8–22.4) | 20.8 (19.9–21.5) | 1.00× | 19.9 |
| 4,096 | grouped | 4 | 27.0 (25.6–29.4) | 26.7 (25.8–27.8) | 0.99× | 25.9 |
| 4,096 | grouped | 8 | 36.5 (34.4–40.4) | 36.0 (34.4–37.2) | 0.99× | 35.1 |

Trees keep main's cost at main's widths: every cell is within 2%, inside both sides' ranges. Against all four main
processes (`widths-main-a` to `-d`) DF2b is 1.01–1.02× at every cell.

At every width (`widths-df2b-a`, `-b`, between `widths-main-a` and `-b`):

| Context | Path | Width | p50 ms | Forward p50 | GPU p50 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,024 | batch step | 1 | 16.2 (15.3–18.6) | 16.2 | 15.6 |
| 1,024 | grouped | 1 | 16.2 (15.6–18.7) | 16.0 | 15.4 |
| 1,024 | grouped | 2 | 19.8 (19.5–22.3) | 19.5 | 19.0 |
| 1,024 | grouped | 4 | 25.4 (25.2–28.8) | 25.1 | 24.6 |
| 1,024 | grouped | 8 | 33.8 (33.6–36.6) | 33.5 | 33.0 |
| 1,024 | grouped | 12 | 49.7 (49.3–54.0) | 49.4 | 48.8 |
| 1,024 | grouped | 16 | 58.1 (56.6–62.7) | 57.8 | 57.2 |
| 4,096 | batch step | 1 | 18.8 (17.7–19.6) | 18.8 | 18.2 |
| 4,096 | grouped | 1 | 19.0 (18.6–19.3) | 18.8 | 18.2 |
| 4,096 | grouped | 2 | 23.4 (22.6–24.0) | 23.1 | 22.5 |
| 4,096 | grouped | 4 | 30.0 (28.6–30.3) | 29.8 | 29.2 |
| 4,096 | grouped | 8 | 39.0 (38.3–39.3) | 38.7 | 38.1 |
| 4,096 | grouped | 12 | 57.5 (55.8–58.4) | 57.3 | 56.7 |
| 4,096 | grouped | 16 | 66.7 (65.1–69.1) | 66.4 | 65.8 |

In these processes the 4K prompts, which follow the 1K prompts' 12- and 16-row widths, run 2.1–3.3 ms slower at
every width than in the restricted processes, the one-row step included; the 1K cells match. The plan's cost table is
these two processes' mean over all six prompts: 17.9, 21.8, 28.0, 36.7, 54.0 and 63.0 ms at 1, 2, 4, 8, 12 and 16
rows. It interpolates linearly between them, and 9 to 11 rows lie on the line through 12 and 16. The draft is priced
at 4.5 ms.

In rounds the verify phase (forward, picks, tree walk and commit) takes 7–14% longer than the table: 19.1 ms at one
row, 30.2 at four, 40.4 at eight and 71.9 at sixteen (What bounds it next). The table thus prices 16 rows 4% low
against eight.

## What a 16-row verify spends

Metal 4 timelines of 12 verifies after 8 warm on code-1024, ms a round by category (`run.py attribution`).
`profile-df2b-8` ran the batch step and widths 1 and 8, `profile-df2b-16` width 16 alone: `prefill.zig` names every
grouped round past eight rows `inkling2-group-8`, so 16 rows need a process of their own.

| Category | batch | group-1 | group-8 | group-16 |
| --- | ---: | ---: | ---: | ---: |
| routed experts, NVFP4 | 3.83 | 3.81 | 16.07 | 28.06 |
| routed experts, affine 4-bit | 0.32 | 0.31 | 1.57 | 3.14 |
| router, pick, combine | 2.13 | 2.19 | 2.31 | 2.70 |
| attention projections | 2.42 | 2.42 | 3.31 | 6.10 |
| o-projection and head | 1.84 | 1.84 | 2.30 | 4.56 |
| attention | 3.53 | 3.53 | 5.40 | 7.99 |
| convolutions | 2.06 | 2.23 | 2.14 | 2.20 |
| controllers | 1.56 | 1.57 | 2.72 | 4.13 |
| depth mixes and norms | 3.64 | 3.58 | 3.74 | 3.60 |
| other kernels and stamps | 0.14 | 0.13 | 0.15 | 0.15 |
| round span | 21.05 | 21.20 | 39.30 | 62.21 |

Rows 9 to 16 add 22.9 ms. The routed NVFP4 experts take 12.0 ms of it, as the rows route 41.8 distinct experts a
visit against 30.7 at eight rows (`widths-df2b-a`, code-1024). The two-tile projections and head take 5.1 ms, and
attention, controllers and the affine experts most of the rest. The tile kernels alone, on code-1024's 16-row routes
(`kernels-df2b`, `run.py kerneltable`), µs a call:

| Rows | Head | O-projection | Projections | Gate/up | Down |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1,152.2 | 20.9 | 55.1 | 214.0 | 121.5 |
| 12 | 2,345.3 | 43.1 | 113.2 | 315.7 | 183.3 |
| 16 | 2,593.8 | 43.2 | 114.2 | 398.8 | 231.0 |

The two-tile dense kernels cost 2.04–2.25× their 8-row kernel at 12 and at 16 rows: two 8-row tiles sharing one
decode of each weight word cost at least what two separate 8-row kernels would. At eight rows the tiles are no longer
bound by the weight stream; their time follows the rows, and sharing the decode saves nothing measurable. The expert
kernels grow 1.86× (gate/up) and 1.90× (down) from eight rows to sixteen, against 1.36× more distinct experts.

## Tree shapes

The trees are the same in every process; `df2b-a`'s (`run.py shapes`):

| Prompt | Rounds | Rows mean | Lookup rows mean | Accepted mean | From lookup | Rounds accepting 8 or more | Widths chosen (rows: rounds) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| code-1024 | 68 | 8.99 | 4.79 | 6.51 | 4.60 | 18 | 1: 1, 2: 1, 3: 2, 4: 2, 5: 8, 6: 6, 7: 10, 8: 20, 16: 18 |
| prose-1024 | 113 | 6.47 | 1.10 | 3.52 | 0.84 | 1 | 2: 4, 3: 11, 4: 2, 5: 21, 6: 12, 7: 15, 8: 46, 16: 2 |
| structured-1024 | 120 | 6.52 | 0.95 | 3.26 | 0.66 | 1 | 1: 1, 2: 8, 3: 9, 4: 4, 5: 10, 6: 16, 7: 24, 8: 45, 16: 3 |
| code-4096 | 173 | 5.29 | 0.93 | 1.95 | 0.62 | 1 | 2: 15, 3: 31, 4: 15, 5: 34, 6: 24, 7: 22, 8: 31, 16: 1 |
| prose-4096 | 175 | 4.95 | 0.69 | 1.92 | 0.50 | 2 | 1: 3, 2: 25, 3: 37, 4: 9, 5: 29, 6: 23, 7: 24, 8: 23, 16: 2 |
| structured-4096 | 137 | 5.85 | 0.67 | 2.73 | 0.47 | 0 | 1: 1, 2: 3, 3: 14, 4: 15, 5: 26, 6: 24, 7: 21, 8: 32, 16: 1 |

Accepted counts drafts; a round commits one token more. On code-1024 the lookup chain supplies 4.60 of the 6.51
accepted drafts a round, and 18 of its 68 rounds verify 16 rows of chain with the drafter skipped. On the other
prompts the chain supplies 0.47–0.84.

## Ablations

Untimed runs of the round pairs' build on the same prompts (`ablate-drafter-tree`: the drafter's tree without the
lookup chain; `ablate-no-skip`: the drafter runs in every round). Tokens, rounds and trees are deterministic, so
committed tokens are exact. "Modeled" prices every round from the plan's own cost table (verify by rows, 4.5 ms when
the drafter ran; `run.py modeled main-a,df2b-a,ablate-drafter-tree,ablate-no-skip`) and is not a measurement. Every
reply equals plain greedy.

| Prompt | Committed/round: main's chain | DF2b | No lookup | No skip | Modeled ms/token: main's chain | DF2b | No lookup | No skip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 5.81 | 7.51 | 5.74 | 7.00 | 7.07 | 5.89 | 6.76 | 6.22 |
| prose-1024 | 4.52 | 4.52 | 4.48 | 4.56 | 9.10 | 8.36 | 8.34 | 8.27 |
| structured-1024 | 4.33 | 4.26 | 4.26 | 4.26 | 9.51 | 8.89 | 8.75 | 8.82 |
| code-4096 | 2.76 | 2.95 | 2.73 | 2.94 | 14.88 | 11.85 | 12.52 | 11.94 |
| prose-4096 | 2.69 | 2.92 | 2.69 | 2.90 | 15.26 | 11.67 | 12.37 | 11.71 |
| structured-4096 | 3.43 | 3.73 | 3.43 | 3.70 | 11.98 | 9.76 | 10.19 | 9.78 |

- **Branching buys the chain's tokens with fewer rows, not more tokens.** Without the chain, the drafter's tree
  commits what main's 8-row chain commits, from 6.94 / 6.31 / 6.29 / 4.96 / 4.62 / 5.29 rows a round on average
  instead of eight (`run.py shapes ablate-drafter-tree`).
- **The lookup chain pays on code and at 4K.** It cuts the modeled time a token 13% on code-1024 and 4–6% at 4K. On
  prose-1024 and structured-1024 it adds 0.2% and 1.6%.
- **Skipping the drafter pays only behind long chains.** On code-1024 it lets 18 rounds verify 16 rows of chain
  instead of 13 (7.51 against 7.00 tokens a round, 5% less modeled time). Elsewhere it changes the modeled time by
  at most 1.1% either way.

Main's modeled time over DF2b's predicts five of the six measured gains within 0.04 (1.26× modeled against 1.29×
measured on code-4096, 1.07× against 1.09× on structured-1024). It misses code-1024's (1.20× modeled, 1.11×
measured), where the 16-row rounds cost 14% more in rounds than the table.

## Plain decode and memory

- **Plain decode.** DF2b's one-row step uses the 16-row scratch. Rested it is 1.01–1.02× main's at both contexts
  (above). In the round processes, DF2b's plain ms a token over main's, per process pair, is 1.081, 1.044, 1.002,
  0.999, 0.994 and 1.013: a geometric mean of 1.022 with a 95% interval of 0.987–1.057. The first two pairs' DF2b
  processes ran slow throughout, their speculative rounds included; the four later pairs are within 1.3%. The
  estimate is under Decision 11's 5% limit.
- **Memory.** Every round process peaked at an 18.3 GB footprint on both builds (heavy.py's charged peak 18.3–19.8 GB
  on main, 18.3–19.0 GB on DF2b). The 16-row scratch and the verify's captures do not move the footprint at that
  resolution. The drafter's residency is 0.52 GB on both. A served lane's lookup index takes at most about 1.1 MiB
  of host memory at 32K tokens: two u32 arrays and a hash map of at most one entry a token.

## Conditions

- **Machine.** Apple M5 Max (18 CPU cores, 40 GPU cores, 48 GB), macOS 27.2 (26B123).
- **Lock and gates.** Every timed run, model load and Metal test went through `campaign/heavy.py` (the GPU lock and
  the 28 GB cap). Each timed `run.py` process passed `gpu-gate-ready -1` (60% limits) and `perf-gate-ready`
  (`PERF_GATE_MAX_BUSY_CORES=4`) before and after it; the one deferral, `profile-df2b-8`'s first attempt, was
  retried after 30 s. Readings are in each `NAME.host.json`. Other processes held 0.2–23.1% of GPU time around the
  runs (median 2.6%): WindowServer, a browser helper and once the terminal. `INKLING_LOAD_LIMIT_GB=26`; caches held
  at most 4,096 + 512 + 16 positions in the timed runs. The rested widths runs idled 60 s inside the lock before
  loading.
- **Quiet host.** From the timing gate on, no other lane compiled or used the GPU: the widths, kernels and profiles
  ran 10:28–10:44, the first session of round pairs 10:46–10:59 and the second 11:00–11:13, back to back. The
  ablations followed untimed (11:13–11:18).
- **Thermal and memory.** `pmset -g therm` recorded no thermal or performance warning before or after any run;
  heavy.py saw memory pressure 1 and no swap growth in every timed process.
- **Builds.** Main: `tf-inkling2-loop-run` built from `git archive 50b4a681a` (runner `1e9e1e18e4e0…`). DF2b:
  `d32a4425da82…` built at `83f9f2633` for the widths, kernels and profiles; `65a97be019f7…` built at `d95cc3fe0`
  for the round pairs and ablations. `d95cc3fe0` changes only the plan's cost table and one planner test, so both
  DF2b builds run the same kernels and verify. The other commits after `83f9f2633` change only records and `run.py`.
  `calib-holdout-a` and `-b` ran the plan's two earlier forms (runners `203555e855cf…`, `bff82646c0a0…`) and `-c`
  the `83f9f2633` build; every reply equals plain greedy, and the plans are described above.

## Reproduction

From the worktree. Build both runners (`--prefix` directories outside the tree):

```sh
git archive 50b4a681a | tar -x -C MAIN_SRC
(cd MAIN_SRC && zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix RUN_MAIN)
zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix RUN_FINAL
```

Then, with `M=RUN_MAIN/bin/tf-inkling2-loop-run`, `F=RUN_FINAL/bin/tf-inkling2-loop-run` and
`D=campaign/kernels/inkling-df2b-tree-runs`:

```sh
# holdout prompts and the plan's priors (untimed)
python3 -I $D/run.py prompts $F $D/prompts-holdout.json --set holdout
python3 -I $D/run.py bench $F $D/prompts-holdout.json calib-holdout-c --drafts tree --untimed
python3 -I $D/run.py calibrate calib-holdout-a,calib-holdout-b
# rested verify widths: main and DF2b at every width, then DF2b at main's widths
for run in "$M widths-main-a" "$F widths-df2b-a" "$F widths-df2b-b" "$M widths-main-b"; do
  set -- $run; python3 -I $D/run.py widths $1 $D/prompts.json $2 --reps 9 --rest 60
done
python3 -I $D/run.py widths $F $D/prompts.json widths-df2b-c --reps 9 --rest 60 --cells 0,1,2,4,8
python3 -I $D/run.py widths $M $D/prompts.json widths-main-c --reps 9 --rest 60
python3 -I $D/run.py widths $M $D/prompts.json widths-main-d --reps 9 --rest 60
python3 -I $D/run.py widths $F $D/prompts.json widths-df2b-d --reps 9 --rest 60 --cells 0,1,2,4,8
python3 -I $D/run.py widthtable widths-df2b-a,widths-df2b-b
python3 -I $D/run.py widthcompare widths-main-c,widths-main-d widths-df2b-c,widths-df2b-d
# kernels alone and the 8- and 16-row timelines
python3 -I $D/run.py picks widths-df2b-a code-1024 $D/picks16-code-1024.json
python3 -I $D/run.py kernels $F $D/prompts.json kernels-df2b --picks $D/picks16-code-1024.json
python3 -I $D/run.py kerneltable kernels-df2b
python3 -I $D/run.py widths $F $D/code-1024.json profile-df2b-8 --reps 20 --timeline 12 --cells 0,1,8
python3 -I $D/run.py widths $F $D/code-1024.json profile-df2b-16 --reps 20 --timeline 12 --cells 16
cp $D/profile-df2b-8.log $D/profile-df2b-8.txt; cp $D/profile-df2b-16.log $D/profile-df2b-16.txt
python3 -I $D/run.py attribution profile-df2b-8.txt inkling2-batch=batch inkling2-group-1=group-1 \
  inkling2-group-8=group-8 profile-df2b-16.txt:inkling2-group-8=group-16
# six round pairs in two sessions, the order balanced
for run in "$M main-a chain" "$F df2b-a tree" "$F df2b-b tree" "$M main-b chain" "$M main-c chain" "$F df2b-c tree" \
           "$F df2b-d tree" "$M main-d chain" "$M main-e chain" "$F df2b-e tree" "$F df2b-f tree" "$M main-f chain"; do
  set -- $run; python3 -I $D/run.py bench $1 $D/prompts.json $2 --drafts $3 --tokens 512
done
python3 -I $D/run.py pairs main-a,main-b,main-c,main-d,main-e,main-f df2b-a,df2b-b,df2b-c,df2b-d,df2b-e,df2b-f
python3 -I $D/run.py roundtable df2b-a,df2b-b,df2b-c,df2b-d,df2b-e,df2b-f
python3 -I $D/run.py roundtable main-a,main-b,main-c,main-d,main-e,main-f
python3 -I $D/run.py same df2b-a,df2b-b,df2b-c,df2b-d,df2b-e,df2b-f
python3 -I $D/run.py costs df2b-a,df2b-b,df2b-c,df2b-d,df2b-e,df2b-f
python3 -I $D/run.py shapes df2b-a
python3 -I $D/run.py budgets df2b-a,df2b-b,df2b-c,df2b-d,df2b-e,df2b-f
# acceptance-only ablations (untimed)
python3 -I $D/run.py bench $F $D/prompts.json ablate-drafter-tree --drafts drafter_tree --untimed
python3 -I $D/run.py bench $F $D/prompts.json ablate-no-skip --drafts tree,skip=0 --untimed
python3 -I $D/run.py modeled main-a,df2b-a,ablate-drafter-tree,ablate-no-skip
python3 -I $D/run.py shapes ablate-drafter-tree,ablate-no-skip
python3 -I $D/run.py modeled calib-holdout-a,calib-holdout-b,calib-holdout-c --table df2b-a
# the acceptance suite
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
```

The `.log` files are not tracked; the two `.txt` copies of the timeline logs are, so `attribution` runs on them in
a fresh checkout. The tests alone:

```sh
TF_INKLING2_LOOP_DIR=$HOME/models/Inkling-2-Mini@ed40e7da625f \
TF_INKLING2_LOOP_OQ4E_DIR=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd \
TF_INKLING2_DFLASH2_DIR=$HOME/models/Inkling-2-Mini-DFlash2@4f9088ea66c2 INKLING_LOAD_LIMIT_GB=26 \
python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1 --summary all
```

## Open problems

- **A duplicated batch scratch.** `group.scratch` repeats `BatchScratch.init` for 16 rows with the prefill scratch's
  half path off, because `batch.zig` caps a batch scratch at eight rows and `prefill.zig` sends chunks of more than
  eight rows to the prefill matrix kernels unless `half` is off. Both files are outside this lane; an option there
  (few-row kernels at every width) would remove the copy and the reliance on `half`.
- **Timeline names stop at eight rows.** `prefill.zig` names every grouped round past eight rows `inkling2-group-8`,
  so a 16-row attribution needs a process that runs only that width.
- **The native checkpoint stays at eight rows.** Its bf16 and MXFP8 projections take the batch step's small-row
  kernels only to eight rows; past them they take the prefill GEMM, another arithmetic. Speculation does not run there
  (the drafter needs the affine head), so only the verifier's limit records it.
- **Not in the kernel-contract runner.** As V2 recorded, none of the family's kernels is registered there; the new
  tile instantiations are checked by the byte tests in `rows_test.zig` and `attention_test.zig` only.
- **One lane at a time.** As SV1: with two or more lanes decoding, every lane takes the plain batch step.
- **Server-level equality** (SV1's HTTP checks) was not rerun; the served Zig test carries the equality claims.
- **The cost table is rested.** In rounds the verify phase runs 7–14% longer than the table, most at 16 rows; a table
  priced from sustained rounds, still fixed so trees stay deterministic, is untested.
- **P4's scratch fields.** P4's merge adds `units`, `fuse` and `rounding` to `PrefillScratch`, which `group.scratch`
  builds before turning `half` off. Its half-precision paths need `half` and its staged and fused kernels a prefill
  chunk, but its fused ring and KV stores follow `fuse` in batch chunks too. That the merged 9- to 16-row verify still
  equals its one-row steps is untested here; inkling-gpu's tree tests settle it after the merge.
- **Sessions.** The two sessions of pairs ran back to back on one quiet host, so they share its state.
