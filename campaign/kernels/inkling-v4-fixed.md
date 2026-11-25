# Inkling 2 Mini grouped verify: the fixed cost (V4)

**Verdict.** Promoted under Decision 11. Over six interleaved round pairs in two sessions on V3's six prompts,
speculative decode is 1.038 times main's tok/s, 95% interval 1.016 to 1.062 over process pairs; plain greedy decode,
the one-row step, is 1.125 times main's (1.108 to 1.141). All twelve processes made the same rounds and trees and
committed plain greedy's tokens. The fixed cost a round was not ~13 ms: unstamped on the served queue, main's one-row
verify spends about 5 ms in glue kernels (depth mixes and norms 2.35 ms in place, controllers 1.01, the key/value
prologue and store 0.77, the router 0.56, the output glue 0.36), chains of small dependent dispatches at 25 to 170
times their bytes. The lane encodes 424 to 472 fewer of a step's 1,302 dispatches with the same arithmetic in the same
order, so every row keeps main's bits and its one-row step's; in rounds the verify is 1.5 to 2.2 ms cheaper at every
width (0.88× at one row, 0.96× at 8, 0.975× at 16). Prefill is untouched, byte-equal with `INKLING_PREFILL_HALF`
unset and `=off`. Decision 17's 200 tok/s holds on code-1024 only (230 against main's 218); the other five prompts
stay at 94 to 160. What remains past one row is the routed experts (stage 3, cut from this lane).

**Base.** Main is `d129b60f8` (P5c plus PH2's half-precision prefill default). Main's runner is `git archive 570188a7c`
(main plus this lane's measurement commits, no kernel changes) with the lane's final `kbench.zig`, built
`-Dcpu=apple_m1` (`run-base3`, sha256 `a644a260…`). The lane's runner is `0394b8734` (`run-lane5`, `788db2f1…`).

## Where the fixed cost goes (main)

`dflash-widths REPS OUT 0 WIDTHS phases` (`kbench.phases`, new here) runs a prompt's grouped verify of its greedy
continuation, or its one-row batch step, with one set of `prefill.Phase` classes encoded into the committed command
buffer and every other dispatch into an uncommitted one. It times the served encoder's own dispatches, unstamped, on
the Metal 3 queue the host serves on. Every set (each class with the always-on edges, each class left out of the
whole step, the router with each expert class, the edges alone, the whole step) runs in a rotated order for eight
repetitions, the first dropped, and a whole step refreshes every scratch buffer before each filtered run. A class
**alone** is its set less the edges alone; **in place** is the whole step less the step without it. Leaving a class
out changes the hidden states and so the routes, which moves the routed experts by milliseconds once a step has more
than one row, so in place is read at one row only. Bytes floors are each class's weights once a round at 521 GB/s
(experts: the distinct experts of DF2b's recorded 16-row routes), the attention's cached keys and values once for the
shared prefix, and other activations once a row (`run.py phase_bytes`).

Main, process `phases-main-a`, ms a round, p50 of seven:

| Class (kernels) | 1 row alone | 1 row in place | Floor | In place / floor | 8 rows alone | 16 rows alone |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| depth mixes and norms before attention (`i2p_gather_rms`, `i2_batch_mix`, `i2_batch_mix4`, `i2p_rms3`, `i2p_rms`) | 1.11 | 1.37 | 0.013 | 105 | 1.48 | 1.50 |
| MLP depth mix and norm | 0.78 | 0.98 | 0.006 | 165 | 1.01 | 1.03 |
| loop controllers (`i2_controls_*`, `i2p_rms_pair`, `i2a_float`, `i2a_batch_float_control`) | 0.91 | 1.01 | 0.435 | 2.3 | 2.53 | 4.80 |
| k/v convolutions and clips (`i2v_conv`, `i2p_clip3`) | 0.36 | 0.48 | 0.007 | 67 | 0.51 | 0.70 |
| k/v row store and relative bias (`i2v_store_rel`) | 0.23 | 0.29 | 0.010 | 29 | 0.66 | 1.01 |
| gate, attention convolution, residual (`i2_sig_mul`, `i2v_conv`, `i2_add`) | 0.22 | 0.25 | 0.010 | 25 | 0.24 | 0.29 |
| MLP convolution | 0.12 | 0.11 | 0.004 | 27 | 0.13 | 0.16 |
| router and pick (`i2a_float`, `i2p_routepick`) | 0.37 | 0.56 | 0.109 | 5.1 | 0.67 | 0.95 |
| affine 4-bit experts and combine (layers 20 to 23) | 0.30 | 0.25 | 0.275 | 0.9 | 1.91 | 4.12 |
| split attention and combine | 1.16 | 1.65 | 0.386 | 4.3 | 3.51 | 6.48 |
| q/k/v/r/gate/sink projections | 1.95 | 2.18 | 1.940 | 1.1 | 2.99 | 5.24 |
| o-projection | 0.84 | 1.02 | 0.693 | 1.5 | 1.32 | 2.01 |
| NVFP4 experts and combine | 3.68 | 4.21 | 3.027 | 1.4 | 19.18 | 31.64 |
| output mix, norm and head | 0.87 | 0.86 | 0.940 | 0.9 | 1.33 | 2.91 |
| whole step | – | 14.90 | – | – | 37.90 | 64.35 |

Context 1,024 (code-1024), grouped rows; the one-row batch step reads within 0.1 ms of the grouped one-row verify in
every class. At 4,096 the glue classes are the same within 0.15 ms and the attention core grows (1.53, 5.14 and 9.39
ms at 1, 8 and 16 rows).

What the table names:

- **The fixed glue is about 5 ms a round, not 13.** At one row the depth mixes and norms (2.35 ms in place), the
  controllers (1.01), the router and pick (0.56), the k/v convolutions, clips and store (0.77) and the output
  convolutions and residuals (0.36) are 5.05 ms in place, 4.10 alone. V3's ~13 ms came from the Metal 4 timeline,
  which writes a GPU timestamp between every two of a step's 1,302 dispatches on a Metal 4 queue the host does not
  serve on; it attributes cost, it does not time it.
- **The glue is latency, not bytes.** The mixes and norms run at 100 to 170 times their bytes, the k/v prologue and
  store at 30 to 70, the output glue at 25 to 35: each is a chain of small dependent dispatches. The lever is fewer
  dispatches doing the same arithmetic, not fewer bytes.
- **Two classes grow with rows**: the controllers (0.91, 2.53, 4.80 ms at 1, 8, 16 rows; 11 times their weights at 16)
  and the affine 4-bit experts (0.30, 1.91, 4.12; 1.8 and 2.9 times their bytes at 8 and 16). The router grows
  0.37 → 0.95.
- **The weights are near their bytes at one row** (projections 1.1, head 0.9, NVFP4 experts 1.4), so the one-row step's
  excess over its floor is the glue and the attention core, whose 1.16 to 1.65 ms is 3 to 4 times its keys and
  values.

## What changed

All in `zig/src/families/inkling2_loop/`. A batch step and a grouped verify (`c.batch != null`) take these while
`PrefillScratch.fuse` is set, the default; with it off they encode main's kernels, which the fused-against-unfused test
compares. Prefill (`c.batch == null`) encodes main's kernels.

1. **Depth mixes with their norms** (`i2pr_batch_mix`, `kernels/prefill_rows.metal`). One threadgroup a row computes
   the states' norms, every query's dots, the softmax, the weighted sums and the output norms that `i2p_gather_rms`,
   `i2_batch_mix` (or `i2_batch_mix4`) and `i2p_rms` or `i2p_rms3` computed, each reduction with their thread mapping,
   SIMD sums and eight-part order, one barrier round a pass (P4's `i2pr_mix_norm`, prefill's, takes two a reduction).
   A bank slot's attention mix and cross-pass message mix share one dispatch (`i2pr_batch_mix2`, the grid's y the
   mix). Every attention mix, MLP mix, message and the head's output mix: 192 fewer dispatches a round.
2. **The joined key/value prologue** (`i2_batch_qkv`, `i2v_qkv`). The key and value convolutions, `i2p_clip3`'s q, r
   and k clips, and the key/value row store with the relative bias (`i2p_batch_store_rel`, `i2v_store_rel`) in one
   dispatch a layer: 144 fewer.
3. **The attention residual in its convolution**: `x += conv(o)` as the convolution's own `+=`, as the MLP site already
   did, instead of a second `i2_add`: 40 fewer.
4. **The experts' combine in the MLP convolution**: the convolution forms each input row from `down` and `wts` with
   `i2p_combine`'s eight-term chain, its tree ancestors' rows too: 48 fewer.
5. **Controllers** at two rows or more (`i2r_controls_down`, `i2_controls_apply_rows`). A controller's down kernel
   takes its row's two norms (`i2_controls_norm`, `i2p_rms_pair`) in one pass and normalizes on the fly; the MLP
   controller runs the attention controllers' kernels (its `i2a_float` down is the same chain over 4,608 inputs); the
   apply computes as many adjacent columns a simdgroup as the step has rows (up to eight), so its grid stays at one
   row's 864 threadgroups instead of 864 a row. A one-row step keeps main's three kernels, which measured 0.16 ms a
   step faster at one row. 48 fewer dispatches a step past one row.
6. **An early commit**: a batch step or verify commits its first six slots as a command buffer of their own, so the
   GPU starts while the rest encodes (prefill already did this; batch steps did not).

A step past one row encodes 472 fewer of the 1,302 dispatches an oQ4e grouped verify takes; a one-row step 424 fewer.

Tried and removed: affine 4-bit experts in runs of up to eight members of an expert (V3's `i2e_run`, each member
keeping `i2a_dot`'s chain). Byte-equal at 1 to 16 rows and every member count, but slower: real forwards without it
are 0.8 ms faster at 8 rows and 1.0 at 16 (below); with eight members' accumulators and decoded words a simdgroup it
likely spills registers. An apply of a fixed eight columns a simdgroup was also slower at one row (a few dozen
threadgroups each walking eight columns), which the row-sized column count replaces.

Measurement only: `prefill.Phase` and `dflash-widths ... phases` (each kernel class alone and the step without it,
unstamped, on the served Metal 3 queue), `dflash-widths ... hashes`, and a timeline round name for every width to 16.

## Round pairs

Each process runs V3's six prompts (DF1's `prompts.json`, unchanged) with the oQ4e target and the q4 drafter: 512
greedy tokens with EOS ignored, plain and speculative from the same prefill state, tree rounds (`--drafts tree`). Both
builds use V3's plan table (the lane leaves `verify.Plan` alone), so both verify the same trees. Twelve processes in
two sessions (pairs a to c, then d to f), main first in pairs a, c and e and the lane first in b, d and f. Spec tok/s
counts the 511 tokens after prefill's first. Before is main, after is the lane; tok/s is the median over the six
processes, range in brackets (`run.py pairs`):

| Prompt | Committed/round before | after | Spec tok/s before | after | After/before per pair | Geomean |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| code-1024 | 10.02 | 10.02 | 218.4 (218.0–218.8) | 229.9 (229.2–230.1) | 1.05, 1.05, 1.05, 1.05, 1.05, 1.05 | 1.05 |
| prose-1024 | 4.56 | 4.56 | 112.5 (109.2–116.4) | 115.2 (111.4–118.5) | 1.07, 0.99, 1.04, 0.99, 1.07, 0.97 | 1.02 |
| structured-1024 | 4.06 | 4.06 | 100.9 (100.2–101.4) | 105.4 (98.6–106.0) | 1.04, 1.05, 1.06, 1.03, 1.05, 0.98 | 1.03 |
| code-4096 | 7.51 | 7.51 | 154.2 (151.2–157.2) | 159.6 (153.8–163.4) | 1.02, 1.04, 1.06, 1.03, 1.05, 0.98 | 1.03 |
| prose-4096 | 3.57 | 3.57 | 88.9 (86.7–91.3) | 94.0 (89.4–96.0) | 1.02, 1.08, 1.06, 1.08, 1.05, 0.98 | 1.04 |
| structured-4096 | 3.52 | 3.52 | 91.4 (85.4–92.8) | 94.4 (92.5–96.3) | 1.07, 1.03, 1.11, 1.02, 1.03, 1.04 | 1.05 |

```text
all prompt-pairs: geometric mean 1.038, 95% interval 1.028–1.049 (n = 36)
1.047, 1.039, 1.064, 1.032, 1.051, 1.000: each process pair's geometric mean over prompts
process pairs: geometric mean 1.038, 95% interval 1.016–1.062 (n = 6)
```

**Promoted (Decision 11).** The geometric-mean gain is 3.8%, at least 3%, and its 95% interval over process pairs,
1.016 to 1.062, is above 1.0. Session 1 (pairs a to c) gives 1.050 (1.018 to 1.082), session 2 (d to f) 1.027 (0.965
to 1.093): session 2's interval alone includes 1.0, because of pair f.

**Pair f** is 1.000. `lane-f` ran slower than `lane-e` on five of the six prompts, its drafter too (4.76 to 5.05 ms a
round against 4.57 to 4.75; the drafter is the same code in both builds), while the gates read a quiet host before
and after it (load 2.9 to 3.8). It is counted as run; without it the five pairs give 1.046 (1.031 to 1.062).

**Same rounds.** `run.py same` over all twelve processes: each prompt's round count, every round's tree and the output
tokens' hash are identical in all twelve and equal to plain greedy (51, 112, 126, 68, 143 and 145 rounds; tokens
`c819a96ceaf7`, `125f75459e24`, `712e645eccf2`, `2c8b298d35bb`, `fbed92d634aa`, `3955a0d64585`). The lane changes the
time of a round, not what it verifies or commits.

Attempts (`NAME.host.json`, `NAME.log`): a deferral is the opening gate refusing, a failed closing check a run that
completed and was discarded and run again after the five-minute backoff.

| Run | Attempts | Last attempt |
| --- | --- | --- |
| main-a | 9: deferred ×7, closing check failed, ran | 00:37:13–00:39:31 |
| lane-a | 2: closing check failed, ran | 00:46:48–00:49:01 |
| lane-b | 1: ran | 00:49:04–00:51:10 |
| main-b | 1: ran | 00:51:14–00:53:28 |
| main-c | 3: closing check failed ×2, ran | 01:08:13–01:10:33 |
| lane-c | 1: ran | 01:10:35–01:12:38 |
| lane-d | 3: closing check failed ×2, ran | 01:27:08–01:29:27 |
| main-d | 1: ran | 01:29:30–01:31:45 |
| main-e | 1: ran | 01:31:48–01:34:01 |
| lane-e | 1: ran | 01:34:04–01:36:12 |
| lane-f | 1: ran | 01:36:15–01:38:23 |
| main-f | 1: ran | 01:38:27–01:40:40 |

### Plain decode

The same processes' plain greedy decode, 511 one-row batch steps a prompt (`run.py plaincompare`):

| Prompt | Plain ms/token before | after | After/before tok/s per pair | Geomean |
| --- | ---: | ---: | --- | ---: |
| code-1024 | 16.21 (16.18–16.23) | 14.16 (14.13–14.21) | 1.146, 1.145, 1.144, 1.147, 1.141, 1.141 | 1.144 |
| prose-1024 | 16.28 (16.23–16.69) | 14.66 (14.43–14.97) | 1.157, 1.112, 1.114, 1.106, 1.101, 1.104 | 1.116 |
| structured-1024 | 17.16 (16.99–17.40) | 15.24 (15.11–15.93) | 1.118, 1.124, 1.133, 1.130, 1.125, 1.092 | 1.120 |
| code-4096 | 17.28 (17.08–17.46) | 15.48 (15.26–16.33) | 1.064, 1.116, 1.137, 1.114, 1.134, 1.083 | 1.108 |
| prose-4096 | 17.43 (17.25–17.80) | 15.52 (15.43–15.73) | 1.104, 1.147, 1.132, 1.114, 1.117, 1.119 | 1.122 |
| structured-4096 | 17.40 (17.15–19.74) | 15.51 (15.39–15.77) | 1.107, 1.127, 1.271, 1.104, 1.113, 1.119 | 1.139 |

```text
all prompt-pairs: geometric mean 1.125, 95% interval 1.114–1.135 (n = 36)
process pairs: geometric mean 1.125, 95% interval 1.108–1.141 (n = 6)
```

The one-row step is 1.125 times main's tok/s (1.108 to 1.141), 1.6 to 2.1 ms a token cheaper: it takes the fused
mixes, the joined key/value prologue, both residual folds and the early commit, and keeps main's controllers.

### Verify by rows in rounds

All twelve processes' tree rounds by verified rows (`run.py costs` on each build's six): the verify phase's mean ms
(the grouped forward, the GPU argmax, the acceptance walk and the commit in one command buffer). Both builds verified
the same 3,870 rounds.

| Rows | Rounds (main / lane) | Main verify ms | Lane verify ms | Lane − main | Lane/main | Committed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 66 / 66 | 18.38 | 16.19 | -2.20 | 0.880 | 1.00 |
| 2 | 258 / 258 | 22.50 | 20.70 | -1.80 | 0.920 | 1.79 |
| 3 | 324 / 324 | 26.74 | 24.99 | -1.75 | 0.934 | 2.30 |
| 4 | 126 / 126 | 29.49 | 27.87 | -1.62 | 0.945 | 2.90 |
| 5 | 948 / 948 | 32.56 | 31.09 | -1.47 | 0.955 | 2.99 |
| 6 | 486 / 486 | 35.07 | 33.54 | -1.54 | 0.956 | 3.81 |
| 7 | 42 / 42 | 38.41 | 36.70 | -1.71 | 0.956 | 5.57 |
| 8 | 1212 / 1212 | 40.26 | 38.67 | -1.59 | 0.960 | 5.09 |
| 11 | 12 / 12 | 53.92 | 52.59 | -1.33 | 0.975 | 11.00 |
| 13 | 6 / 6 | 49.00 | 46.73 | -2.27 | 0.954 | 13.00 |
| 14 | 12 / 12 | 62.13 | 62.41 | +0.29 | 1.005 | 8.00 |
| 15 | 6 / 6 | 68.82 | 67.29 | -1.53 | 0.978 | 8.00 |
| 16 | 372 / 372 | 62.95 | 61.36 | -1.59 | 0.975 | 14.27 |

Over all tree rounds main's verify is 37.01 ms and the lane's 35.42 (0.957×). The lane is 1.5 to 2.2 ms cheaper at
every width rounds verify more than a dozen times, at one row and at 16 alike: the fixed cost a round, which is what
the lane cut. The drafter ran in 3,498 rounds of each at 4.78 ms (main) and 4.85 ms (lane); the 16-row rounds are
lookup chains, the drafter skipped in all but 6 of 372.

### Where the time goes

Median over the six processes of each process's mean, range in brackets (`run.py roundtable`).

The lane:

| Prompt | Equal | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | yes | 51 | 10.02 | 43.6 (43.5–43.7) | 54.2 (54.0–54.3) | 2.35 | 41.2 | 0.03 | 229.9 (229.2–230.1) | 14.16 (14.13–14.21) | 70.6 (70.4–70.8) | 3.26 (3.24–3.27) |
| prose-1024 | yes | 112 | 4.56 | 39.6 (38.5–40.9) | 44.9 (43.8–49.7) | 4.60 | 35.0 | 0.02 | 115.2 (111.4–118.5) | 14.66 (14.43–14.97) | 68.2 (66.8–69.3) | 1.68 (1.64–1.75) |
| structured-1024 | yes | 126 | 4.06 | 38.5 (38.2–41.1) | 44.0 (43.8–50.0) | 4.62 | 33.9 | 0.01 | 105.4 (98.6–106.0) | 15.24 (15.11–15.93) | 65.6 (62.8–66.2) | 1.60 (1.57–1.61) |
| code-4096 | yes | 68 | 7.51 | 47.1 (46.0–48.9) | 70.0 (68.8–77.1) | 3.13 | 43.9 | 0.01 | 159.6 (153.8–163.4) | 15.48 (15.26–16.33) | 64.6 (61.2–65.5) | 2.48 (2.44–2.51) |
| prose-4096 | yes | 143 | 3.57 | 38.0 (37.2–40.0) | 46.5 (45.6–51.6) | 4.82 | 33.2 | 0.01 | 94.0 (89.4–96.0) | 15.52 (15.43–15.73) | 64.4 (63.6–64.8) | 1.47 (1.38–1.49) |
| structured-4096 | yes | 145 | 3.52 | 37.3 (36.6–38.1) | 45.7 (44.3–46.9) | 4.80 | 32.5 | 0.01 | 94.4 (92.5–96.3) | 15.51 (15.39–15.77) | 64.5 (63.4–65.0) | 1.47 (1.46–1.48) |

Main:

| Prompt | Equal | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | yes | 51 | 10.02 | 45.9 (45.8–46.0) | 56.4 (56.3–56.6) | 2.39 | 43.5 | 0.03 | 218.4 (218.0–218.8) | 16.21 (16.18–16.23) | 61.7 (61.6–61.8) | 3.54 (3.53–3.55) |
| prose-1024 | yes | 112 | 4.56 | 40.6 (39.2–41.8) | 46.6 (46.1–46.9) | 4.46 | 36.1 | 0.02 | 112.5 (109.2–116.4) | 16.28 (16.23–16.69) | 61.4 (59.9–61.6) | 1.85 (1.80–1.89) |
| structured-1024 | yes | 126 | 4.06 | 40.2 (40.0–40.5) | 46.1 (45.7–46.6) | 4.58 | 35.6 | 0.01 | 100.9 (100.2–101.4) | 17.16 (16.99–17.40) | 58.3 (57.5–58.9) | 1.73 (1.72–1.75) |
| code-4096 | yes | 68 | 7.51 | 48.7 (47.8–49.7) | 71.5 (69.6–74.0) | 3.10 | 45.6 | 0.01 | 154.2 (151.2–157.2) | 17.28 (17.08–17.46) | 57.9 (57.3–58.6) | 2.67 (2.63–2.69) |
| prose-4096 | yes | 143 | 3.57 | 40.2 (39.1–41.2) | 48.9 (46.8–54.4) | 4.81 | 35.4 | 0.01 | 88.9 (86.7–91.3) | 17.43 (17.25–17.80) | 57.4 (56.2–58.0) | 1.57 (1.52–1.58) |
| structured-4096 | yes | 145 | 3.52 | 38.6 (38.0–41.3) | 46.0 (45.6–49.6) | 4.70 | 33.9 | 0.01 | 91.4 (85.4–92.8) | 17.40 (17.15–19.74) | 57.5 (50.7–58.3) | 1.59 (1.53–1.68) |

## Kernel classes: main against the lane

One process each, `phases-main-a` (main) against `phases-lane-a` (the lane at `11966b253`, the tool already
refreshing; `0394b8734` changed only the controllers, below), ms a round, grouped rows, alone unless marked:

| Context | Class | 1 row in place, main → lane | 1 row | 8 rows | 16 rows |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,024 | depth mixes and norms | 2.36 → 1.26 | 1.89 → 1.04 | 2.49 → 1.33 | 2.53 → 1.35 |
| 1,024 | controllers | 1.01 → 1.21 | 0.91 → 1.09 | 2.53 → 2.54 | 4.80 → 4.39 |
| 1,024 | k/v convolutions, clips and store | 0.77 → 0.42 | 0.59 → 0.26 | 1.16 → 0.72 | 1.70 → 1.28 |
| 1,024 | gate, convolutions and residuals | 0.36 → 0.54 | 0.35 → 0.40 | 0.38 → 0.64 | 0.44 → 0.70 |
| 1,024 | router and pick | 0.56 → 0.56 | 0.37 → 0.40 | 0.67 → 0.69 | 0.95 → 1.02 |
| 1,024 | affine 4-bit experts | 0.25 → 0.34 | 0.30 → 0.30 | 1.91 → 2.05 | 4.12 → 4.29 |
| 1,024 | split attention and combine | 1.65 → 1.73 | 1.16 → 1.22 | 3.51 → 3.79 | 6.48 → 6.85 |
| 1,024 | projections, o-projection, NVFP4 experts and head | 8.28 → 8.06 | 7.34 → 7.31 | 24.82 → 26.43 | 41.80 → 43.00 |
| 1,024 | whole step | 14.90 → 13.70 (0.92×) | – | 37.90 → 38.67 (1.02×) | 64.35 → 64.04 (1.00×) |
| 1,024 | one-row batch step | 14.81 → 13.33 (0.90×) | – | – | – |
| 4,096 | depth mixes and norms | 2.52 → 1.26 | 2.16 → 1.15 | 2.59 → 1.34 | 2.56 → 1.35 |
| 4,096 | controllers | 0.94 → 1.28 | 1.00 → 1.15 | 2.64 → 2.54 | 4.85 → 4.39 |
| 4,096 | k/v convolutions, clips and store | 0.79 → 0.50 | 0.65 → 0.28 | 1.22 → 0.76 | 1.76 → 1.28 |
| 4,096 | gate, convolutions and residuals | 0.28 → 0.41 | 0.38 → 0.41 | 0.42 → 0.63 | 0.45 → 0.69 |
| 4,096 | router and pick | 0.66 → 0.60 | 0.42 → 0.42 | 0.74 → 0.74 | 1.02 → 1.02 |
| 4,096 | affine 4-bit experts | 0.23 → 0.24 | 0.31 → 0.31 | 2.15 → 2.12 | 4.21 → 4.21 |
| 4,096 | split attention and combine | 2.12 → 1.98 | 1.53 → 1.58 | 5.14 → 5.18 | 9.39 → 9.54 |
| 4,096 | projections, o-projection, NVFP4 experts and head | 8.49 → 8.29 | 7.77 → 7.75 | 27.54 → 27.04 | 43.25 → 43.20 |
| 4,096 | whole step | 16.20 → 14.60 (0.90×) | – | 41.91 → 39.78 (0.95×) | 68.36 → 66.73 (0.98×) |
| 4,096 | one-row batch step | 15.72 → 14.40 (0.92×) | – | – | – |

The lane's whole step is 0.90 to 0.92 times main's at one row on both contexts, 0.95 and 1.02 at 8 rows and 0.98 and
1.00 at 16. The depth mixes and norms lose 0.85 to 1.25 ms at every width and context, the key/value prologue and store
0.33 to 0.48; the gate, convolutions and residuals gain about 0.25 ms past one row, which is the experts' combine moved
into the MLP convolution (it leaves the experts' class). At 8 and 16 rows a single pair cannot place the lane within
2%: main's untouched NVFP4 experts alone read 19.2 against the lane's 20.6 ms at 8 rows here, and the experts vary 5%
between processes of one build.

The controllers in that lane build took the joined kernels at every width and read 0.18 ms slower than main's alone
at one row (1.09 against 0.91), break even at 8 rows and 0.41 ms faster at 16. `0394b8734` keeps main's three kernels
for a one-row step and takes the two norms in one pass.

## Real forwards

`dflash-widths` (nine repetitions, code-1024, p50 ms, wall and GPU), main against an earlier lane build and that
build without one change at a time, run back to back before the outside load peaked:

| Build | Batch step | 1 row | 8 rows | 16 rows | Attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 15.43 / 14.87 | 15.73 / 14.95 | 33.64 / 32.80 | 52.50 / 51.63 | 1 |
| lane, with the affine runs and an eight-column apply | 13.74 / 13.51 | 14.06 / 13.62 | 32.69 / 32.20 | 51.17 / 50.66 | 2 |
| that lane without the early commit | 14.12 / 13.61 | 14.43 / 13.71 | 33.07 / 32.29 | 51.57 / 50.72 | 7 |
| that lane without the affine runs | 13.81 / 13.55 | 14.09 / 13.66 | 31.89 / 31.43 | 50.13 / 49.67 | 7 |

Every grouped width in these processes was bit-equal to its one-row steps. The lane is faster at every width; the
early commit is worth 0.4 ms a step; the affine runs cost 0.8 ms at 8 rows and 1.0 at 16, so the final lane
removes them. At one row the fixed eight-column apply gave back part of the gain (the controllers alone read 0.16
to 0.6 ms slower than main's at one row), which the final lane's as-many-columns-as-rows apply avoids.

## Correctness

The contract is V1's to V3's, unchanged (Decision 18): a row of a batch step or grouped verify has its path's one-row
steps' logits, picks, taps and committed caches bit for bit. The lane goes further and keeps main's bits: every fused
kernel performs the per-element operations, fma chains, SIMD sums and part orders of the kernels it replaces, so its
outputs are main's, and prefill is untouched.

- **Fused against unfused** (`batch_test.zig`, new): `fuse` on and off on the same scratch, batch steps of 1, 2, 3, 5
  and 8 streams three steps deep and grouped verifies of every width up to the limit (16 on oQ4e, 8 on the mixed
  checkpoint) plus a 16-row and a 5-row tree, committing: logits, picks, accepted rows and all six cache buffers byte
  for byte, on both checkpoints.
- **The joined prologue** (`attention_test.zig`, new): `i2_batch_qkv` and `i2v_qkv` against `i2p_batch_conv` or
  `i2v_conv`, `i2p_clip3` and `i2p_batch_store_rel` or `i2v_store_rel` on random inputs: 1, 3 and 6 rows, local and
  full layers, positions 0, 1, 2, 5 and 1,500 (the convolution's skipped taps), batch steps and verify chains; q, the
  relative bias, the clipped keys, values, rings or captures and the stored rows byte for byte (60 cases).
- **Across builds** (`dflash-widths ... hashes`, `run.py hashcompare`): main's runner and the lane's (`run-lane5`) on
  the six prompts, with `INKLING_PREFILL_HALF` unset and `=off`: the 512-row chunk's logits, the last row's logits and
  all six cache buffers after the whole prompt, 16 one-row greedy steps' logits, tokens and caches, grouped verifies
  at 1 to 9, 12 and 16 rows, and each width's count of rows equal to their one-row steps: 228 of 228 records equal at
  each setting (`hash-main-*`, `hash-lane-*`). Main's own unset and off records differ in 157 of 228, so the setting
  was exercised. Main's hashes come from `run-base2`, `570188a7c` built before the phase tool's refresh fix, which the
  hashes do not use.
- **Filtered GPU tests** through `heavy.py` (`zig build test-inkling2-loop-filtered -Dcpu=apple_m1 -j1 --maxrss
  28000000000`, both checkpoints): `keep` 15/15 (the two new tests; grouped rows alone and shared; tree verification,
  54 rows of trees to 16 rows equal to their paths' one-row steps and 15 commits keeping sequential decoding's
  caches; V3's expert, mix and split byte tests), `verif` 14/14, `bit-equal` 5/5 (35 batch-step rows at widths 2 to 8
  equal to their one-row steps), `concurrent` 5/5 (8 greedy streams joining and leaving equal to their solo replies),
  `match decode` 5/5, `taps` 6/6. `zig build test-inkling2-loop-dflash2` 11/11: speculation and tree speculation equal
  plain greedy and its caches on real prompts (259 and 260 rounds), forced drafts and trees, lookup rounds and the
  plain-scratch determinism test.
- **Round pairs**: all twelve processes' speculative replies equal plain greedy's, with the same rounds, trees and
  tokens in both builds (`run.py same`, above).
- **Flint**: `zig build test-flint-draft-host --summary all` exits 0, 47/47 steps and 87/87 tests
  (`flint-draft-host.txt`).
- **Lint and whitespace**: `tools/zig/lean_check.py` finds 256 problems on `d129b60f8` and on the lane, `lean_new.py`
  0 new; `git diff --check d129b60f8 HEAD` is clean. No source file the lane touched passes 600 lines that did not
  already.

## Conditions

- **Machine.** Apple M5 Max (18 CPU cores, 40 GPU cores, 48 GB), macOS 27.2. `INKLING_LOAD_LIMIT_GB=26`, every model
  load and GPU test through `campaign/heavy.py` (the lock and the 28 GB cap); the highest charged peak was 25.2 GB
  (a filtered test run), the timed processes 17.6 to 19.8 GB.
- **Gates.** Each timed run passed `gpu-gate-ready -1` (60% limits) and `perf-gate-ready`
  (`PERF_GATE_MAX_BUSY_CORES=4`) before and after it; a run whose closing check failed is invalid and was run again.
  Attempts are in each `NAME.host.json` and `NAME.log`.
- **Outside load**, not the campaign's and not stopped (owner's other sessions): from about 22:33, OrbStack
  (`dev.kdrag0n.MacVirt`), then `rustc` builds from other terminals (load 12 to 34 on 18 cores), `fseventsd` at a core,
  a `laputa-systems/stuff` perf run, shellcheck, dsymutil and a `zig build test` in `.work/wt/portable-shell`, with host
  swap at 87 to 91% and memory pressure up to 2. It deferred runs up to 6 times each; `phases-main-a` ran on its
  second attempt (its first was invalidated by its closing check), `phases-lane-a` on its first.
- **Invalid trials kept**: `invalid-phases-*` (the first pass, without the refresh), `invalid-main-a`,
  `invalid-main-a-2`, `invalid-diag-cols1`, `invalid-diag-main-b` and `invalid-phases-lane-b-2` (stopped or deferred
  runs, attempts only).
- **Builds** (`zig build tf-inkling2-loop-run -Dcpu=apple_m1`): main `run-base3` above; the lane `run-lane5` at
  `0394b8734`; the phase lane `run-lane4` at `11966b253`; the diagnostic lane `run-lane3` (the two-mix dispatch,
  affine runs and an eight-column apply, uncommitted, its patch kept outside the tree) and its variants without the
  early commit and without the affine runs.

## Reproduction

From the worktree, with `D=campaign/kernels/inkling-v4-fixed-runs`, `M` and `L` the main and lane runners'
`bin/tf-inkling2-loop-run`:

```sh
git archive 570188a7c | tar -x -C MAIN_SRC
cp zig/src/families/inkling2_loop/kbench.zig MAIN_SRC/zig/src/families/inkling2_loop/
(cd MAIN_SRC && zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix RUN_MAIN)
zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix RUN_LANE
for run in "$M phases-main-a" "$L phases-lane-a"; do set -- $run
  python3 -I $D/run.py phases $1 $D/code-1k4k.json $2 --reps 7 --cells 0,1,8,16; done
python3 -I $D/run.py phasetable phases-main-a
python3 -I $D/run.py phasecompare phases-main-a phases-lane-a
for run in "$M diag-main-a" "$L diag-lane"; do set -- $run
  python3 -I $D/run.py widths $1 $D/code-1024.json $2 --reps 9 --cells 0,1,8,16; done
for half in unset off; do  # INKLING_PREFILL_HALF unset, then =off
  $M ~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd $D/prompts.json dflash-widths 1 hash-main-$half.jsonl 0 \
    1,2,3,4,5,6,7,8,9,12,16 hashes
  $L ~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd $D/prompts.json dflash-widths 1 hash-lane-$half.jsonl 0 \
    1,2,3,4,5,6,7,8,9,12,16 hashes
done
for f in keep verif bit-equal concurrent "match decode" taps; do
  python3 campaign/heavy.py -- zig build test-inkling2-loop-filtered -Dcpu=apple_m1 "-Dinkling-filter=$f" -j1 \
    --maxrss 28000000000 --summary all; done
python3 campaign/heavy.py -- zig build test-inkling2-loop-dflash2 -Dcpu=apple_m1 -j1 --maxrss 28000000000
```

The GPU tests need `TF_INKLING2_LOOP_DIR`, `TF_INKLING2_LOOP_OQ4E_DIR` and `TF_INKLING2_DFLASH2_DIR`; every runner
command goes through `campaign/heavy.py` with `INKLING_LOAD_LIMIT_GB=26` (`run.py` does this behind the gates).

## Open problems

- **The routed experts dominate past one row** (19 to 33 ms of a 38 to 64 ms step at 8 and 16 rows, 1.6 to 2.0 times
  their bytes). Out of this lane's scope after the owner's cut (stage 3).
- **Affine 4-bit experts past one row** still run one-slot kernels at 1.8 to 2.9 times their bytes at 8 and 16 rows;
  the runs kernel tried here lost (above). A version with fewer members a simdgroup, or V3's tile layout for affine
  codes, is the next try.
- **The attention core** (split and combine) is the largest fixed class left at one row (1.2 to 1.7 ms, 3 to 4 times
  its keys and values) and grows to 6.5 to 9.4 ms at 16 rows; its summation order fixes its split, and V3's shared
  key staging lost.
- **The plan's cost table** (`verify.Plan.default_verify_ms`) is V3's; the lane makes every width cheaper, so SV2's
  refit from in-round costs should follow.
- **Not in the kernel-contract runner**: as V2, DF2b and V3 recorded, the family's kernels are checked by its own byte
  tests only.
