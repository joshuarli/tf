# Inkling 2 Mini grouped verify (V1)

**Verdict.** The grouped verify now reads each weight once for two to eight rows while every row keeps the one-row
step's bits, and the speculation round waits on one command buffer instead of three. Rested at 1K the eight-row verify
falls from 56.2 to 45.6 ms (0.81×), four rows from 33.2 to 28.6 ms and two rows from 21.4 to 20.1 ms; sustained, eight
rows fall from 65.5 to 52.7 ms at 1K and from 68.2 to 52.5 ms at 4K. Width 1 does not regress. On DF1's six prompts the
round table keeps DF1's tokens and rounds exactly and gains 23–29% in speculative tok/s over DF1 (25–26% over a
same-session control): `code-1024` 95.0 → 116.6 (1.90× plain), the other 1K prompts 1.31–1.32× plain, 4K 0.81–1.02×. The
target of about 30 ms for eight rows rested at 1K is **not met**, and no width meets a Decision 17 round budget. Round
time bounds 200 tok/s on every prompt (`code-1024` takes 49.8 ms where its 5.81 committed tokens allow 29.0), and at 4K
acceptance does too: 2.7–3.4 committed tokens allow 13–17 ms rounds, less than the eight-row verify's weight floor plus
the draft.

## Where the eight-row verify spent its time

Baseline: commit `3e23dbc99` (main `93de31eca`'s kernels plus the measurement below). One `dflash-widths` process on
`code-1024` (1,024-token state) under `TF_METAL4_TIMELINE=12`: every width and the one-row batch step ran 21 times
from identical states, and the timeline stamped 12 rounds of each after 8 warm ones. On a Metal 4 queue each serial
dispatch ends in a barrier with a timestamp, so these spans carry about 5 µs a dispatch of instrumentation (22.2 ms
for one grouped row against 16.2 ms uninstrumented); they attribute cost and are not timings. ms a round:

| Category | batch | group-1 | group-2 | group-4 | group-8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| routed experts, NVFP4 | 4.33 | 4.24 | 7.45 | 13.97 | 25.41 |
| routed experts, affine 4-bit | 0.20 | 0.23 | 0.48 | 0.89 | 1.59 |
| router, pick, combine | 1.90 | 2.11 | 2.15 | 2.28 | 2.39 |
| attention projections | 2.42 | 2.53 | 2.87 | 4.19 | 7.61 |
| o-projection and head | 1.05 + 0.86 | 1.03 + 0.93 | 1.37 + 1.54 | 2.40 + 3.01 | 4.59 + 6.04 |
| attention | 3.38 | 3.62 | 3.74 | 4.29 | 5.47 |
| convolutions | 2.10 | 2.21 | 2.19 | 2.24 | 2.24 |
| controllers | 1.54 | 1.62 | 1.74 | 1.97 | 2.76 |
| depth mixes and norms | 4.38 | 3.96 | 3.71 | 3.76 | 3.71 |
| other kernels and stamps | 0.16 | 0.16 | 0.14 | 0.16 | 0.17 |
| round span | 22.03 | 22.20 | 26.97 | 38.74 | 61.54 |

The head is the round's last stage; the o-projection is the rest of `i2a_dense`. From one row to eight the round grows
39.3 ms:

- **Routed experts, +22.5 ms (57%).** At eight rows the shared NVFP4 kernels (`i2_gu_rows`, `i2_down_rows`) moved
  6.05 GB in 25.4 ms, 238 GB/s.
- **Head and o-projection, +8.7 ms; fused attention projections, +5.1 ms (35%).** Every row read the weights again:
  the head 461 MB eight times.
- **Attention, +1.9 ms; controllers, +1.1 ms.** Mixes, norms and convolutions do not grow.

**Measured routing.** The same run captured every visit's routed experts (`routes` and `picks` records). Consecutive
tokens share experts: per visit, a mean of 13.2 distinct experts at width 2, 20.3 at 4 and 30.7 at 8 (17–39), against
52 for eight independently routed rows. 55% of an eight-row visit's experts have one routed row, and most visits have
one routed by seven or eight. The weight floor at eight rows is therefore 6.60 GB of experts plus 2.27 GB of everything
else, 17.0 ms at 521 GB/s, not DF1's 25.7 ms; at two and four rows it is 9.8 and 12.7 ms.

## What changed

All on `lane/v1-verify`, in `zig/src/families/inkling2_loop/`. One row keeps every kernel it ran before.

- **Each weight read once for two to eight batch rows** (`rows.zig`, `kernels/rows.metal`, `kernels/gather.metal`).
  Every output keeps its one-row kernel's arithmetic: the same lane walks the same words or groups in the same order,
  each product is the same `fma` into the same running sum, and the row's sum is the same `simd_sum`. Only the loads
  are shared, so every row's bits are the one-row step's. Head and o-projection kernels exist for two, four and
  eight rows; three and five to seven rows run the next larger one with the missing rows skipped.
  - Head: four columns a SIMD group for every row, weights loaded two words ahead (`i2r_dense_ahead`).
  - O-projection: two columns a SIMD group, the rows' inputs staged in threadgroup memory 64 words at a time
    (`i2r_dense_staged`).
  - Fused attention projections at five to eight rows: four columns a SIMD group, four rows a threadgroup
    (`i2r_projections`). At two to four rows the one-row kernel was as fast and runs unchanged.
  - NVFP4 gate/up and down, gathered by expert (`i2g_gate_up`, `i2g_down_all`): the threadgroup of the lowest slot
    routed to an expert computes every slot routed to it, decoding each code group once for each pair (gate/up) or for
    all routed rows (down); the others exit before any work. Scales are decoded through a 256-entry table that the
    kernel builds with the per-slot kernels' own e4m3 decoder.
  - The eight-row shared NVFP4 kernels these replace (`i2_gu_rows`, `i2_down_rows`) are removed.
- **One command buffer a round** (`group.zig` `decide`, `kernels/verify.metal`). The speculation loop's verify forward
  now ends with the GPU argmax, the acceptance count (`i2v_accept`, the host rule `acceptedDrafts`) and the commit of
  the accepted rows' keys, values and convolution inputs (`i2v_round_kv`, `i2v_round_conv`, which read the count the
  GPU wrote). The host waits once a round instead of three times. `forward`, `picks` and `commit` remain for the tests
  and the widths mode.
- **Measurement** (`dflash_run.zig`, `run.zig`, `prefill.zig`). Grouped rounds are named by width on the Metal 4
  timeline. `dflash-widths` records each forward's GPU span, every visit's routed experts, and whether each grouped
  width's logits equal the one-row steps that produced its rows (it exits non-zero if not). `dflash-kernels`
  (`kbench.zig`) times each shared kernel against its one-row kernel on the checkpoint's weights, layers rotating per
  call so each call reads from DRAM.

Tried and dropped, on the same kernel bench (`kb*.jsonl` in the runs directory record every variant):

- one column a SIMD group with the lane's whole weight row held in registers (slower: 330–379 µs against 283–293 µs
  for eight-row gate/up);
- per-element `float2` inputs with per-element decode, to cut registers (8–20% slower);
- the one-row kernels with the grid transposed, or the slots sorted by expert, so threadgroups sharing an expert run
  together (no gain: the caches do not merge the duplicate reads);
- staged inputs for the head (6% slower than a column-tiled head, 45% slower than the kept one), and a staged
  projection kernel for the single-input local layers (5% faster at eight rows, 1% slower at four, left out for its
  size);
- a refactor of the head kernel with the same arithmetic that compiled 36% slower at eight rows; the kept source is
  the exploratory one.

## Correctness

The contract is unchanged: every grouped row's logits, picks, taps and committed caches equal the plain one-row
batch step's, bit for bit, so greedy speculation equals greedy decoding token for token.

- `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0 on the trees of `c4584a653`
  (`.work/suite/20261124T012955`, 11.4 min) and `eb7ebfa23` (`.work/suite/20261124T021136`, 11.2 min), with the suite's
  no-skip check on for both workloads. `inkling-gpu` passed 14/14 steps and 135/135 tests both times; `inkling-host`
  passed 10/10 steps, 35/35 tests the first time and 25/25 the second, when Zig reused one unchanged test binary's
  earlier pass of 10 tests. `6a8396c23` changes only `kbench.zig`, which only the runner compiles, so the suite's
  binaries are unchanged by it. `zig build native -Dcpu=apple_m1` exited 0 on `6a8396c23`.
- On the real checkpoints (native NVFP4/MXFP8 and oQ4e):
  - `test-inkling2-loop-dflash2`: speculation equals plain greedy on four chat prompts, 1,024 tokens over 273 rounds
    (72 zero, 140 partial, 58 full: DF1's counts exactly, as bit equality requires); forced drafts give full, zero and
    prefix acceptance, a limit and an EOS inside a block, with outputs equal to plain greedy and target caches equal
    to plain decoding's.
  - Grouped verification (three streams of 4, 3 and 1 rows, the first crossing the 1,024-row ring) equals sequential
    batch decoding bit for bit in logits, picks and caches, and a grouped row's taps equal a one-row step's.
  - New: a fused round (`decide`) gives the same picks, acceptance and committed caches as forward, picks and commit at
    zero, partial and full acceptance.
  - `batched independent cache steps match decode` at widths 1, 2, 4 and 8 reports zero logit and cache difference
    from the single-stream forward; the 2-to-8-row batch step now runs the new kernels.
- Kernel byte tests on random inputs: the head and o-projection kernels at two to eight rows with and without the logit
  cap, tails included (28 cases); the fused projections at five to eight rows in both layouts (8 cases); the gathered
  gate/up and down against the per-slot kernels at two to eight rows of shared picks over 12 experts (14 cases). Every
  output's bytes equal the one-row kernel's.
- Every `dflash-widths` run on the before and after runners compares each grouped width's logits with the one-row
  steps that produced its rows: all 24 widths of each six-prompt run (widths 1, 2, 4 and 8) and the 4 of
  `profile-after-1k` are bit-equal (`profile-base-1k` predates the check). Every `dflash` run's speculative tokens
  equal plain greedy (`equal_plain`).
- No logit difference, so no bounded-difference argument is needed.
- `python3 -I tools/zig/lean_new.py` on `6a8396c23` reports 0 new problems against `93de31eca` and against main
  `20364662d` (256 problems, 257 on each base).

## Kernels alone

`dflash-kernels` on the after runner: each projection kind timed alone, 24 calls a command buffer with the layer
rotating per call (so each call reads its weights from DRAM), five interleaved rounds, median µs a call. Routed calls
use the 48 visits' real picks from `widths-before-a`'s `code-1024` (`picks-code-1024.json`). "One-row kernel" is the
kernel a batch of that many rows ran before (per row or per routed slot); at two and four rows the fused projections
still run it.

| Rows | Kind | One-row kernel µs | Shared µs | Change | Bytes floor µs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | head | 1505.8 | 1005.1 | 0.67× | 885 |
| 2 | o-projection | 25.9 | 22.2 | 0.85× | 14 |
| 2 | fused projections | 53.5 | (unchanged) | — | 40 |
| 2 | NVFP4 gate/up | 91.7 | 86.5 | 0.94× | 76 |
| 2 | NVFP4 down | 66.7 | 55.0 | 0.82× | 38 |
| 4 | head | 2986.7 | 1402.9 | 0.47× | 885 |
| 4 | o-projection | 47.3 | 28.9 | 0.61× | 14 |
| 4 | fused projections | 74.0 | (unchanged) | — | 40 |
| 4 | NVFP4 gate/up | 184.2 | 159.4 | 0.87× | 116 |
| 4 | NVFP4 down | 129.1 | 96.2 | 0.75× | 58 |
| 8 | head | 5951.5 | 2461.9 | 0.41× | 885 |
| 8 | o-projection | 93.3 | 64.8 | 0.69× | 14 |
| 8 | fused projections | 146.8 | 125.5 | 0.86× | 40 |
| 8 | NVFP4 gate/up | 353.8 | 281.5 | 0.80× | 176 |
| 8 | NVFP4 down | 263.7 | 176.6 | 0.67× | 88 |

`kernels-final.jsonl` also holds `proj` records at two and four rows, where production keeps the one-row kernel
(`shared_kernel` false). Their `shared_us` is not a measurement: the bench timed the one-row kernel again with the
shared kernel's argument layout. `6a8396c23` gives that repeat the one-row kernel's own arguments. In `kernels-check`,
run after the fix, both sides of those records time the same kernel: 48.9 against 52.5 µs at two rows and 74.0 against
74.0 µs at four, so a call near 50 µs carries about 7% of side-to-side noise. `kernels-check` is a check of the fix, not
a timing record: another lane's compile started during its eight-row calls.

The bytes floor is each call's weights once at 521 GB/s: the head 461 MB; the o-projection 7.5 MB; the fused local
projections 21.1 MB; gate/up 2.99 MB and down 1.49 MB for each distinct routed expert (13.2, 20.3 and 30.7 at two, four
and eight rows). At eight rows the shared kernels reach 36% of that floor for the head, 63% for gate/up and 50% for
down; at two rows the head reaches 88%. What holds them back:

- **The fixed per-lane order.** Each output must be one SIMD group summing lane `l`'s words or groups `l`, `l + 32`,
  … in order, so a weight can feed only the rows and columns that one SIMD group holds in registers. A matrix unit
  (the 8- to 32-row expert tiles Splash runs) sums in another order and would change the bits.
- **Decode and multiply–add issue.** By an instruction count, the gathered gate/up kernel at eight rows issues about
  167 µs of decode, scale and `fma` work per call against its 176 µs DRAM floor, and the two do not overlap fully.
- **Skewed routing.** 55% of an eight-row visit's experts have one row, while one expert is usually routed by seven or
  eight; synthetic picks of two rows per expert ran gate/up at 219 µs, real picks at 283 µs.

## Verifier cost at widths 1, 2, 4 and 8

Each prompt is prefilled; the rows are the greedy continuation and width w verifies its first w. Grouped forwards
leave the caches unchanged and the batch step is rolled back, so every sample starts from the same state. Nine
interleaved repetitions follow one warm-up; p50 of forward plus GPU picks in ms (DF1's measure), averaged over the
three prompts of a context and the processes on each side, range over them in brackets. "After GPU p50" is the
forward's own command-buffer span. Every run's grouped widths are bit-equal to the one-row steps.

**Rested**: four processes in before, after, after, before order, each started after 60 s idle (`widths-before-c`,
`widths-after-e`, `widths-after-f`, `widths-before-d`):

| Context | Path | Width | Before p50 | After p50 | Change | After GPU p50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | batch step | 1 | 16.4 (16.0–17.6) | 15.9 (15.9–16.0) | 0.97× | 15.4 |
| 1,024 | grouped | 1 | 16.6 (16.2–17.6) | 16.2 (16.1–16.3) | 0.98× | 15.5 |
| 1,024 | grouped | 2 | 21.4 (21.0–22.1) | 20.1 (19.9–20.2) | 0.94× | 19.4 |
| 1,024 | grouped | 4 | 33.2 (32.7–34.3) | 28.6 (28.4–28.9) | 0.86× | 27.8 |
| 1,024 | grouped | 8 | 56.2 (55.4–57.9) | 45.6 (45.3–45.9) | 0.81× | 44.8 |
| 4,096 | batch step | 1 | 18.8 (17.7–19.5) | 18.2 (17.3–18.8) | 0.97× | 17.6 |
| 4,096 | grouped | 1 | 18.9 (17.9–19.7) | 18.5 (17.6–19.0) | 0.98× | 17.7 |
| 4,096 | grouped | 2 | 24.5 (23.6–25.2) | 22.4 (21.3–23.2) | 0.91× | 21.6 |
| 4,096 | grouped | 4 | 37.9 (35.9–39.0) | 31.9 (30.6–33.1) | 0.84× | 31.1 |
| 4,096 | grouped | 8 | 64.2 (61.5–67.2) | 50.5 (48.3–52.4) | 0.79× | 49.7 |

**Sustained**: each prompt after 20 s of continuous eight-row verifies (`widths-before-s`, then `widths-after-s`):

| Context | Path | Width | Before p50 | After p50 | Change | After GPU p50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | batch step | 1 | 19.5 (19.4–19.5) | 19.0 (18.8–19.1) | 0.97× | 18.4 |
| 1,024 | grouped | 1 | 19.5 (19.3–19.6) | 19.2 (18.9–19.4) | 0.98× | 18.3 |
| 1,024 | grouped | 2 | 25.3 (25.2–25.4) | 23.6 (23.4–23.7) | 0.93× | 22.8 |
| 1,024 | grouped | 4 | 38.9 (38.9–38.9) | 33.5 (33.2–33.9) | 0.86× | 32.7 |
| 1,024 | grouped | 8 | 65.5 (65.5–65.6) | 52.7 (52.3–53.2) | 0.80× | 51.9 |
| 4,096 | batch step | 1 | 19.9 (19.6–20.3) | 18.9 (18.8–19.1) | 0.95× | 18.4 |
| 4,096 | grouped | 1 | 20.3 (20.2–20.4) | 19.2 (19.1–19.4) | 0.95× | 18.5 |
| 4,096 | grouped | 2 | 25.8 (25.6–25.9) | 23.5 (23.4–23.6) | 0.91× | 22.7 |
| 4,096 | grouped | 4 | 40.6 (40.3–40.9) | 33.4 (33.2–33.5) | 0.82× | 32.6 |
| 4,096 | grouped | 8 | 68.2 (67.4–69.2) | 52.5 (52.3–52.7) | 0.77× | 51.8 |

The one-row batch step's kernels did not change, so its 0.95–0.98× is the pairing's noise and drift. The before side
reproduces DF1 (rested 56.3 ms, sustained 65.2 ms at eight rows, 1K). The earlier rested pairing, `widths-before-a`
and `-b` against `widths-after-c` and `-d`, is recorded but not used: `after-c` and `after-d` started straight after
the sustained runs, and at 1K their unchanged one-row step ran 1.12× slower than the before side's. Even that pairing
gives eight rows 0.90× at 1K and 0.79× at 4K.

The same timeline as the baseline profile, on the after runner (`profile-after-1k`), ms a round:

| Category | batch | group-1 | group-2 | group-4 | group-8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| routed experts, NVFP4 | 4.14 | 4.23 | 7.22 | 12.50 | 21.07 |
| routed experts, affine 4-bit | 0.22 | 0.21 | 0.52 | 0.91 | 1.65 |
| router, pick, combine | 2.17 | 2.06 | 2.10 | 2.18 | 2.39 |
| attention projections | 2.54 | 2.45 | 2.89 | 4.15 | 6.81 |
| o-projection and head | 1.09 + 0.89 | 1.03 + 0.87 | 1.19 + 1.01 | 1.50 + 1.43 | 3.19 + 2.49 |
| attention | 3.46 | 3.65 | 3.79 | 4.28 | 5.61 |
| convolutions | 2.12 | 2.25 | 2.22 | 2.20 | 2.40 |
| controllers | 1.59 | 1.66 | 1.49 | 1.77 | 2.79 |
| depth mixes and norms | 3.87 | 4.01 | 3.72 | 3.66 | 3.42 |
| other kernels and stamps | 0.15 | 0.16 | 0.32 | 0.36 | 0.40 |
| round span | 21.87 | 22.21 | 26.07 | 34.51 | 51.79 |

Eight rows now cost 51.8 ms on the timeline against 61.5 ms: experts −4.3 ms, head and o-projection −4.9 ms,
projections −0.8 ms. The eight-row step is still 2.3 one-row steps on the timeline (2.8 uninstrumented), and its
growth from one row (29.6 ms there) is still mostly routed experts (+18.3 ms), then projections (+4.4), head and
o-projection (+3.8) and attention (+2.0).

## Round table

oQ4e target with the q4 drafter: draft depth 7, an eight-row verify and 512 greedy output tokens with EOS ignored, on
DF1's six prompts (`prompts.json`, copied unchanged) and protocol. The after processes `v1-q4-a`, `v1-q4-b` and
`v1-q4-c` ran 01:50–02:00, with the before runner's `base-q4-a` between the first two as a same-session control. Rounds
and committed tokens equal DF1's in every process, as bit equality requires, and every prompt's tokens equal plain
greedy. Times are the median of the three processes' means, range in brackets; "Spec tok/s before" is DF1's `q4-a`,
`q4-b` and `q4-c`. Spec tok/s counts the 511 tokens after prefill's first; plain is one batch step a token from the
same prefill state in the same process.

| Prompt | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain | Spec tok/s before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 88 | 5.81 | 49.8 (49.6–49.8) | 50.7 (50.5–51.1) | 3.92 | 45.9 | 0.02 | 116.6 (116.5–117.0) | 16.31 (16.26–16.31) | 61.3 (61.3–61.5) | 1.90 (1.89–1.91) | 95.0 (94.8–96.0) |
| prose-1024 | 113 | 4.52 | 58.2 (58.0–58.6) | 60.2 (60.0–60.7) | 4.53 | 53.7 | 0.01 | 77.7 (77.2–77.9) | 16.93 (16.65–17.05) | 59.1 (58.7–60.1) | 1.32 (1.29–1.32) | 61.3 (60.8–62.1) |
| structured-1024 | 118 | 4.33 | 58.3 (57.9–58.5) | 59.7 (59.3–59.8) | 4.52 | 53.8 | 0.00 | 74.3 (74.0–74.8) | 17.54 (17.46–17.61) | 57.0 (56.8–57.3) | 1.31 (1.30–1.31) | 57.8 (57.5–58.9) |
| code-4096 | 185 | 2.76 | 59.5 (59.2–61.3) | 61.1 (60.8–62.8) | 4.69 | 54.8 | 0.00 | 46.5 (45.1–46.7) | 17.64 (17.49–17.69) | 56.7 (56.5–57.2) | 0.82 (0.79–0.82) | 36.1 (35.6–37.1) |
| prose-4096 | 190 | 2.69 | 59.4 (59.1–60.2) | 60.9 (60.7–62.7) | 4.70 | 54.8 | 0.00 | 45.2 (44.7–45.5) | 17.81 (17.81–17.97) | 56.1 (55.6–56.2) | 0.81 (0.80–0.82) | 35.5 (35.4–36.2) |
| structured-4096 | 149 | 3.43 | 59.2 (59.1–59.6) | 61.3 (60.9–61.7) | 4.67 | 54.5 | 0.00 | 58.0 (57.6–58.0) | 17.65 (17.58–17.79) | 56.7 (56.2–56.9) | 1.02 (1.02–1.02) | 45.6 (45.5–46.3) |

Phases are mean ms a round. **Verify** now holds the grouped forward, the GPU argmax, the acceptance and the commit,
one command buffer and one wait; **Host** is the taps' absorption and bookkeeping. In DF1 the separate select and commit
waits cost 0.45–0.50 ms a round; they are 0.00–0.02 ms here. **Draft** is unchanged.

The control reproduces DF1 within 3% on every prompt:

| Prompt | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 88 | 5.81 | 62.2 (62.2–62.2) | 67.1 (67.1–67.1) | 4.11 | 57.6 | 0.45 | 93.4 (93.4–93.4) | 16.63 (16.63–16.63) | 60.1 (60.1–60.1) | 1.55 (1.55–1.55) |
| prose-1024 | 113 | 4.52 | 72.7 (72.7–72.7) | 75.5 (75.5–75.5) | 4.70 | 67.5 | 0.50 | 62.2 (62.2–62.2) | 17.32 (17.32–17.32) | 57.7 (57.7–57.7) | 1.08 (1.08–1.08) |
| structured-1024 | 118 | 4.33 | 73.5 (73.5–73.5) | 75.1 (75.1–75.1) | 4.75 | 68.3 | 0.46 | 58.9 (58.9–58.9) | 17.54 (17.54–17.54) | 57.0 (57.0–57.0) | 1.03 (1.03–1.03) |
| code-4096 | 185 | 2.76 | 74.3 (74.3–74.3) | 76.3 (76.3–76.3) | 4.84 | 69.0 | 0.46 | 37.2 (37.2–37.2) | 17.65 (17.65–17.65) | 56.6 (56.6–56.6) | 0.66 (0.66–0.66) |
| prose-4096 | 190 | 2.69 | 74.5 (74.5–74.5) | 76.2 (76.2–76.2) | 4.86 | 69.1 | 0.47 | 36.1 (36.1–36.1) | 17.94 (17.94–17.94) | 55.7 (55.7–55.7) | 0.65 (0.65–0.65) |
| structured-4096 | 149 | 3.43 | 74.2 (74.2–74.2) | 76.9 (76.9–76.9) | 4.84 | 68.9 | 0.47 | 46.2 (46.2–46.2) | 17.69 (17.69–17.69) | 56.5 (56.5–56.5) | 0.82 (0.82–0.82) |

## Decision 17 budgets

Decision 17 allows at most 20, 25, 30 and 40 ms a round at 4, 5, 6 and 8 mean committed tokens, 5 ms a committed
token, which is also 200 tok/s. Measured rounds against that rule (`run.py budgets`):

| Prompt | Committed/round | Round ms | Budget at that commitment ms | ms a committed token | Meets |
| --- | ---: | ---: | ---: | ---: | --- |
| code-1024 | 5.81 | 49.8 | 29.0 | 8.58 | no |
| prose-1024 | 4.52 | 58.2 | 22.6 | 12.87 | no |
| structured-1024 | 4.33 | 58.3 | 21.7 | 13.46 | no |
| code-4096 | 2.76 | 59.5 | 13.8 | 21.52 | no |
| prose-4096 | 2.69 | 59.4 | 13.4 | 22.10 | no |
| structured-4096 | 3.43 | 59.2 | 17.1 | 17.25 | no |

No prompt meets its budget. No width can meet one at any acceptance: width `w` commits at most `w` tokens, so its
best budget is `5w` ms, 10, 20 and 40 ms for widths 2, 4 and 8. Their verifies alone take 20.1, 28.6 and 45.6 ms
rested at 1K, before any draft. Cutting each prompt's full-depth chains to a narrower width and pricing the round at the
after runner's sustained cost (`run.py project v1-q4-c widths-after-s`):

| Prompt | Width | Committed/round | Draft | Verify+picks p50 | Commit+host | Round ms | Budget ms | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 1 | 1.00 | 0.00 | 19.20 | 0.02 | 19.21 | 5.0 | 52.1 |
| code-1024 | 2 | 1.92 | 3.89 | 23.68 | 0.02 | 27.59 | 9.6 | 69.6 |
| code-1024 | 4 | 3.45 | 3.89 | 33.86 | 0.02 | 37.77 | 17.2 | 91.3 |
| code-1024 | 8 | 5.84 | 3.89 | 53.21 | 0.02 | 57.11 | 29.2 | 102.2 |
| prose-1024 | 1 | 1.00 | 0.00 | 18.93 | 0.01 | 18.94 | 5.0 | 52.8 |
| prose-1024 | 2 | 1.83 | 4.52 | 23.44 | 0.01 | 27.98 | 9.2 | 65.4 |
| prose-1024 | 4 | 3.13 | 4.52 | 33.24 | 0.01 | 37.78 | 15.7 | 83.0 |
| prose-1024 | 8 | 4.52 | 4.52 | 52.50 | 0.01 | 57.03 | 22.6 | 79.2 |
| structured-1024 | 1 | 1.00 | 0.00 | 19.36 | 0.00 | 19.37 | 5.0 | 51.6 |
| structured-1024 | 2 | 1.84 | 4.52 | 23.53 | 0.00 | 28.05 | 9.2 | 65.5 |
| structured-1024 | 4 | 2.93 | 4.52 | 33.39 | 0.00 | 37.91 | 14.7 | 77.3 |
| structured-1024 | 8 | 4.31 | 4.52 | 52.33 | 0.00 | 56.84 | 21.5 | 75.8 |
| code-4096 | 1 | 1.00 | 0.00 | 19.07 | 0.00 | 19.07 | 5.0 | 52.4 |
| code-4096 | 2 | 1.69 | 4.66 | 23.54 | 0.00 | 28.20 | 8.4 | 59.8 |
| code-4096 | 4 | 2.42 | 4.66 | 33.21 | 0.00 | 37.87 | 12.1 | 63.8 |
| code-4096 | 8 | 2.77 | 4.66 | 52.65 | 0.00 | 57.31 | 13.8 | 48.3 |
| prose-4096 | 1 | 1.00 | 0.00 | 19.29 | 0.00 | 19.29 | 5.0 | 51.8 |
| prose-4096 | 2 | 1.66 | 4.68 | 23.38 | 0.00 | 28.06 | 8.3 | 59.3 |
| prose-4096 | 4 | 2.34 | 4.68 | 33.53 | 0.00 | 38.22 | 11.7 | 61.3 |
| prose-4096 | 8 | 2.70 | 4.68 | 52.32 | 0.00 | 57.00 | 13.5 | 47.4 |
| structured-4096 | 1 | 1.00 | 0.00 | 19.36 | 0.00 | 19.36 | 5.0 | 51.6 |
| structured-4096 | 2 | 1.76 | 4.67 | 23.59 | 0.00 | 28.26 | 8.8 | 62.1 |
| structured-4096 | 4 | 2.75 | 4.67 | 33.46 | 0.00 | 38.14 | 13.7 | 72.1 |
| structured-4096 | 8 | 3.44 | 4.67 | 52.66 | 0.00 | 57.34 | 17.2 | 60.0 |

At sustained cost width 8 is the best width for `code-1024` (102 tok/s) and width 4 for the other five prompts
(61–83 tok/s).

## What bounds 200 tok/s

Round time on every prompt, and acceptance as well at 4K, by the measured numbers:

- **Round time.** `code-1024` commits 5.81 tokens a round, so 200 tok/s allows 29.0 ms; the round takes 49.8 ms
  (3.9 ms draft, 45.9 ms verify). `prose-1024` and `structured-1024` commit 4.3–4.5 and allow 21.7–22.6 ms against
  58 ms. Even at full acceptance, eight committed tokens allow 40 ms, and the eight-row verify alone takes 46–53 ms.
  It is 2.7–3.1× its weight floor of 17.0 ms (6.60 GB of distinct experts and 2.27 GB of everything else at
  521 GB/s); the one-row step is 2.1× its own 7.65 ms floor.
- **Acceptance.** After the draft, 200 tok/s leaves the eight-row verify 25.1 ms at `code-1024`'s acceptance (1.5×
  its floor), 17.2–18.1 ms at the other 1K prompts' (the floor itself) and 8.7–12.4 ms at the 4K prompts' 2.7–3.4
  committed tokens. That is below the floor (17.0 ms at 1K, more with 4K keys and values) and below one plain step
  (17.6–17.8 ms), so at 4K no verify kernel reaches 200 tok/s at this acceptance. S1 measured Splash committing 9.1
  tokens a cycle on `code-4096` with up to 16 verified rows, where this chain commits 2.76.
- **The verify kernels.** Within the exactness contract, the remaining distance to the floor is per-kernel efficiency
  (table above), not bytes: each eight-row visit already reads only its 30.7 distinct experts once. An extra verified
  row now costs 4.2 ms rested at 1K (5.7 ms before). Splash 1.7.2's start-up calibration on this Mac, same weights,
  costs 1.5 ms a row, 26.3 ms at eight (S1, `campaign/kernels/inkling-splash.md` on main); its expert tiles sum in
  another order than its one-row step, and S1 records its greedy output varying between runs under
  `--verify-rows auto`.

## Conditions

- **Lock and gates.** Every timed run, model load and the Metal suite went through `campaign/heavy.py`: the GPU lock
  and the 28 GB cap, with peaks of 16.9–20.6 GB. Each `run.py` run passed `gpu-gate-ready -1` (60% limits) and
  `perf-gate-ready` (`PERF_GATE_MAX_BUSY_CORES=4`) before and after, retrying a deferral; readings are in each
  `NAME.host.json`. They show 22–37% of GPU time in other processes around the runs, mostly the terminal and
  WindowServer; the reading before `widths-before-a` caught another lane's server, which that run then waited
  11 minutes behind for the lock. `INKLING_LOAD_LIMIT_GB=26`, contexts at most 4,096 + 16 tokens.
- **Shared host.** Other lanes ran their own suites, servers and timing runs on the same lock throughout, so runs
  here queued behind them and the host's state between runs was not under this lane's control. Before and after were
  therefore interleaved within one session (order under **Reproduction**): rested `before-c`, `after-e`, `after-f`,
  `before-d`, each after 60 s idle; sustained `before-s`, `after-s`; the round table `v1-q4-a`, `base-q4-a`,
  `v1-q4-b`, `v1-q4-c`.
- **Invalid and unused trials.** `widths-after-a` and `widths-after-b` overlapped a test compile of this lane
  (01:42:35–01:43:10), so their timings are invalid. `widths-after-c` and `-d` replaced them but started straight
  after the sustained runs, so the rested pairing with `widths-before-a` and `-b` is not used (see above). All six
  stay in the runs directory, and their exactness records stand.
- **Thermal.** `pmset -g therm` recorded no thermal or performance warning before or after any run. Absolute times
  moved by up to 20% between sessions on this host (a 16.0 ms one-row step measured 19.7 ms an hour later, before any
  change of its kernels), which is why every comparison here is within interleaved processes.
- **Build identities.**
  - Before: `5a088508…`, runner of `3e23dbc99` (main `93de31eca`'s kernels and this lane's measurement code), built
    from a `git archive` export.
  - After: `13fa5b60…`, runner of `c4584a653`. `eb7ebfa23` changes only a test, so its runner's sources are the
    same. `6a8396c23` changes only the kernel bench, which no widths or round-table run calls.
  - The baseline profile ran on `a3b909e4…`, `3e23dbc99`'s tree before its exactness check was added; its kernels
    are the baseline's.

## Reproduction

Runners: the before runner is `zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix DIR` in a `git archive 3e23dbc99`
export; the after runner is the same command in this worktree at `c4584a653` or later. `run.py` runs each process
through `campaign/heavy.py` and the gates above, and writes `NAME.jsonl`, `NAME.log` and `NAME.host.json`. The session
ran in this order (01:30–02:08), with `kernels-final` through the same gates:

```sh
R0=BEFORE_DIR/bin/tf-inkling2-loop-run
R1=AFTER_DIR/bin/tf-inkling2-loop-run
D=campaign/kernels/inkling-v1-verify-runs
python3 -I $D/run.py widths $R0 $D/prompts.json widths-before-a --reps 9
python3 -I $D/run.py widths $R1 $D/prompts.json widths-after-a --reps 9
python3 -I $D/run.py widths $R1 $D/prompts.json widths-after-b --reps 9
python3 -I $D/run.py widths $R0 $D/prompts.json widths-before-b --reps 9
python3 -I $D/run.py widths $R0 $D/prompts.json widths-before-s --reps 9 --load 20
python3 -I $D/run.py widths $R1 $D/prompts.json widths-after-s --reps 9 --load 20
python3 -I $D/run.py widths $R1 $D/prompts.json widths-after-c --reps 9
python3 -I $D/run.py widths $R1 $D/prompts.json widths-after-d --reps 9
python3 -I $D/run.py bench $R1 $D/prompts.json v1-q4-a --bits 4 --tokens 512
python3 -I $D/run.py bench $R0 $D/prompts.json base-q4-a --bits 4 --tokens 512
python3 -I $D/run.py bench $R1 $D/prompts.json v1-q4-b --bits 4 --tokens 512
python3 -I $D/run.py bench $R1 $D/prompts.json v1-q4-c --bits 4 --tokens 512
python3 -I $D/run.py widths $R1 $D/code-1024.json profile-after-1k --reps 20 --timeline 12
INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- $R1 ~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd \
  $D/code-1024.json dflash-kernels 24 $D/kernels-final.jsonl $D/picks-code-1024.json
for run in "$R0 widths-before-c" "$R1 widths-after-e" "$R1 widths-after-f" "$R0 widths-before-d"; do
  set -- $run
  sleep 60
  python3 -I $D/run.py widths $1 $D/prompts.json $2 --reps 9
done
```

The tables above come from the recorded runs:

```sh
python3 -I $D/run.py attribution profile-base-1k.log
python3 -I $D/run.py attribution profile-after-1k.log
python3 -I $D/run.py compare widths-before-c,widths-before-d widths-after-e,widths-after-f
python3 -I $D/run.py compare widths-before-s widths-after-s
python3 -I $D/run.py roundtable v1-q4-a,v1-q4-b,v1-q4-c \
  --against ../inkling-dflash2-runs/q4-a,../inkling-dflash2-runs/q4-b,../inkling-dflash2-runs/q4-c
python3 -I $D/run.py roundtable base-q4-a
python3 -I $D/run.py budgets v1-q4-a,v1-q4-b,v1-q4-c
python3 -I $D/run.py project v1-q4-c widths-after-s
```

`profile-base-1k` ran earlier, on the `a3b909e4…` build, as `TF_METAL4_TIMELINE=12 $R0 TARGET $D/code-1024.json
dflash-widths 20 OUT 0` through the same gates rather than through `run.py`. `picks-code-1024.json` is the `picks`
record of `widths-before-a`'s `code-1024`. `kernels-check` is the `kernels-final` command on the runner of
`6a8396c23`, run at 02:35. The `kb*.jsonl` files and `kernels-live-list` are the development bench's records of the
variants above, from runners that no longer exist in the tree.

## Open problems

- The eight-row target (about 30 ms rested at 1K) is not met: 46–51 ms rested, 52.7 ms sustained.
- No variant tried here reaches it within the contract. The structure that reaches it in Splash, matrix-unit expert
  tiles, sums in another order than the one-row kernels, so adopting it is a contract change, unless the one-row step
  moves onto the same tiles and both paths keep one arithmetic.
- Live-tile grouping, as Splash's `moe_group_routes_verified_s0` does: one pass writes each expert's routed rows and
  an indirect grid, so no threadgroup launches for a slot whose expert another slot owns. Here about half of each
  gathered grid exits at once. On the kernel bench with each visit's live slots listed by the host
  (`kernels-live-list`): eight-row gate/up 282 → 279 µs and down 175 → 165 µs, four-row 167 → 156 and 96 → 93 µs. Over
  44 NVFP4 visits that is about 0.6 ms an eight-row verify before the grouping pass's own dispatch a visit, so it was
  not adopted.
- The affine 4-bit experts of layers 20–23 (1.6 ms at eight rows) and the attention splits (5.6 ms at eight rows,
  every row reading the same keys and values) still run per row.
- The served host's batch step for two to eight streams now runs the shared kernels too, with each row's bits
  unchanged; its served throughput was not measured (SV1).
- The new kernels are not registered in the Metal kernel-contract runner; none of the family's kernels is.
- The 4K acceptance (2.7–3.4) and the sustained-load slowdown are DF1's and unchanged.
- A refactor of the head kernel with the same arithmetic compiled 36% slower at eight rows; the cause is not traced,
  and the kept source is the exploratory one.
