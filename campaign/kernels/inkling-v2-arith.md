# Inkling 2 Mini decode tiles (V2)

**Verdict.** The grouped verify's routed NVFP4 experts, head, o-projection and fused attention projections now run on
8x8 simdgroup matrix tiles, and every row of every width keeps the one-row step's bits. Decision 18 permitted a new
arithmetic; none was needed. On this M5 Max a float matrix product is a sequential fma chain, so the tiles compute the
existing one-row kernels' chains in their own order. Rested at 1K the eight-row verify falls from 45.7 to 34.3 ms
(0.75×), four rows from 28.7 to 25.7 ms and two rows from 20.2 to 19.9 ms; at 4K eight rows fall from 50.3 to 37.2 ms.
Sustained, eight rows fall from 54.6 to 40.2 ms at 1K and from 54.9 to 40.8 ms at 4K. The one-row step is no slower
rested (15.9 → 15.7 ms at 1K, ahead on every prompt) and as plain greedy decoding (0.98–1.01×); its routed down
projection now runs the tile kernel's one-row path, 28.6 µs a visit against 35.6. Sustained at 1K it measured 1.03×,
inside both sides' spread. Decode-path logits through one-row batch steps and 8-row grouped verifies are
byte-identical to main's on all 3,015 parity positions, and all 11 scored quality dev replies on oQ4e equal main's (7
of 11 pass on both); the native-checkpoint quality workload cannot start on main or V2 under the 26 GB load limit. On
DF1's six prompts the round table keeps V1's tokens and rounds exactly and gains 25–30% in speculative tok/s over V1
(26–33% over a same-session control on main's kernels): `code-1024` 116.6 → 149.5 (2.41× plain), the other 1K prompts
1.64–1.66× plain, 4K 1.02–1.30×. The target of about 26–30 ms for eight rows rested at 1K is **not met**, and no width
meets a Decision 17 round budget at the measured acceptance.

## The finding that sets the design

Decision 18 allowed the decode arithmetic to change so that matrix units could serve every width. It turned out not
to need changing. On this M5 Max a float `simdgroup_multiply_accumulate` is one fma chain over the inner index, in
index order, from the accumulator (`mma-probe.swift`): of 1,280,000 output elements with operands spread over 2^±12,
all 1,280,000 equal `d = c; d = fma(a[t], b[t], d)` for t = 0..7, and no other order tried matches more than 71%. A
nine-step chain of products, and operands that include ±0, subnormals, infinities and values that overflow, give the
same bytes as a GPU scalar fma chain in every one of 256,000 elements each (`mma-edges.swift`). The kernels' map
from lane to matrix element holds on this GPU (`mma-lanes.swift`). Float products run at 16.2 TFLOP/s, half at 10.5
and scalar fma at 13.5 (`mma-edges.log`).

So a tile can reproduce any one-row kernel whose per-lane work is a sequential fma chain followed by `simd_sum`: each
lane's chain becomes a chain of 8x8 products over the same products in the same order (the lane's k values run along
the tile's inner index), and the 32 lanes' partial sums are added in `simd_sum`'s butterfly order, which
`inkling-decode-step.md` measured as partners 1, 2, 4, 8 and 16. Every row of every width then has the one-row
kernel's bits. The "one arithmetic" of Decision 18 is the existing one: no fidelity, quality or greedy-equality
question opens, and the new tests check bit equality, not tolerances.

## What changed

All in `zig/src/families/inkling2_loop/` on `lane/v2-arith`; no shared core, other family or served-host file.

- **Tile kernels** (`kernels/rows.metal`). They replace V1's `i2r_dense_ahead_N` for N above 2, `i2r_dense_staged_N`,
  `i2r_projections_8` and `kernels/gather.metal` (`i2g_gate_up_1`, `i2g_gate_up_2`, `i2g_down_all_2`):
  - `i2t_dense_8_2`, affine 8-bit rows (the head with its logit cap, the o-projection). Eight simdgroups a
    threadgroup, each owning 4 of `i2a_dot`'s 32 lane chains (lane `l`: words `l`, `l + 32`, … in order) for two
    8-row output tiles. One chain step is an 8x8x8 product of two of the lane's words against up to 8 batch rows; a
    word's values are the one-row kernel's `fma(code, scale, bias)`. The chains add in `simd_sum`'s tree, four in
    registers and eight across simdgroups.
  - `i2t_projections_8_2`, the same over a layer's fused attention projections, each threadgroup inside one
    projection so it reads one input.
  - `i2t_gate_up` and `i2t_down`, routed NVFP4 experts. A threadgroup is one block of output rows for the lowest
    slot routed to an expert. An expert routed one to three rows (gate/up) or one row (down) runs scalar code that
    keeps `i2p_gu`'s and `i2p_down`'s lane chains and decodes each group once for all its rows. More rows run tiles:
    a group's 16 products are two chained 8x8x8 products from zero (the one-row kernel's group partial), then the
    group's scale is fma'd into the lane class's chain, and the classes add in `simd_sum`'s tree.
  - V1's two-row head kernel `i2r_dense_ahead_2` stays.
- **Dispatch** (`rows.zig`, `prefill.zig`), by rows in the batch step or grouped verify:

  | Projection | 1 row | 2 rows | 3 to 8 rows |
  | --- | --- | --- | --- |
  | head, affine 8-bit | `i2a_dense` | `i2r_dense_ahead_2` (V1's) | `i2t_dense_8_2` |
  | o-projection, affine 8-bit | `i2a_dense` | `i2t_dense_8_2` | `i2t_dense_8_2` |
  | fused attention projections, affine 8-bit | `i2a_batch_projections` | `i2a_batch_projections` | `i2t_projections_8_2` |
  | routed gate/up, NVFP4 | `i2p_gu` | `i2t_gate_up` | `i2t_gate_up` |
  | routed down, NVFP4 | `i2t_down` | `i2t_down` | `i2t_down` |

  The affine 8-bit rows are oQ4e's; on the native NVFP4/MXFP8 checkpoint only the expert rows change. The one-row
  batch step, which serves plain decode, changes only in its routed down projection.
- **Device check** (`rows.zig`, `i2t_probe`). Opening the pipelines runs two chained 8x8x8 products on 8 tiles of
  operands spread over 2^±12 and compares every element's bytes with the same fma chains on the host. On a mismatch
  no tile kernel is dispatched and every width runs the one-row kernels and V1's two-row head kernel, which have the
  one-row arithmetic.
- **Measurement** (`steps_run.zig`, `run.zig`, `kbench.zig`). The runner's `decode-logits <width> <dir>` mode for the
  decode-path fidelity check below; `dflash-kernels` times rows 1, 2, 3, 4 and 8, each against the one-row kernel.
- **Tests** (`batch_test.zig`, `rows_test.zig`, `attention_test.zig`, `rows.zig`), under Correctness.

## Where the one-row kernels stay

A tile kernel serves a width only where it is both bit-equal to the one-row kernel and faster:

- **One row.** The head and the fused projections already run at their bytes floor (875 against 885 µs, 40.6 against
  40 µs), and a tile would do eight rows' products for one. The tile kernel's one-row gate/up path took 52.9 µs
  against `i2p_gu`'s 49.3 (`harness-experts.log`), so `i2p_gu` stays; its one-row down path takes 28.6 µs against
  `i2p_down`'s 35.6 with the same bits, so one-row steps use it.
- **Two rows.** The head keeps V1's `i2r_dense_ahead_2`: 1,002 µs against the tiles' 1,086 (`kernels-v1-tiles.jsonl`).
  The fused projections keep `i2a_batch_projections` (48.3 µs): a tile's products are fixed whatever its live rows,
  and the projection tiles take 55 µs at three, four and eight rows alike.
- **Experts with few rows.** Inside `i2t_gate_up` an expert routed one to three rows runs scalar code with the one-row
  chains. With tiles for every expert, two-row gate/up took 93.2 µs against V1's 86.7 (`kernels-v1-tiles.jsonl`);
  with the scalar path it takes 85.4.
- **Attention** keeps its per-row split kernels. The rows of one stream sit at consecutive positions, so their split
  boundaries differ, and staging one stream's keys and values once for all its rows would change each row's per-split
  summation order; that is not row-independent.
- **The convolutions, controllers, router and affine 4-bit experts (layers 20–23)** keep their kernels; tiles for them
  were not attempted. Together they grow 2.9 ms from one row to eight, against the NVFP4 experts' 12.5 ms.

## Correctness

The contract is V1's, unchanged: every row of a batch step or grouped verify has the one-row batch step's logits,
picks, taps and committed caches bit for bit, so greedy speculation equals greedy decoding token for token. What is
new is how the tiles keep it; the checks that show it are bit equality, not tolerances.

- **Kernel byte tests** on random inputs at served shapes (`rows_test.zig`, `attention_test.zig`): the dense tiles
  and the two-row head kernel against `i2a_dense` at 2–8 rows in four shapes, with and without the logit cap,
  including output tails of 2 and 10 rows past a 16-row tile (56 cases); the fused projection tiles against
  `i2a_batch_projections` at 3–8 rows in both layouts (12 cases); the NVFP4 tiles against `i2p_gu` and `i2p_down`
  at 1–8 rows of random picks over 12 experts and of every row routed the same 8 experts, so experts of 1 to 8 rows
  take each path (31 cases). Every output's bytes equal the one-row kernel's.
- **Row independence on the checkpoints** (`batch_test.zig`, both the native NVFP4/MXFP8 checkpoint and oQ4e):
  - eight streams with different prompts, so different routes: each stream's logits and caches after a batch step
    at widths 2–8, in seven compositions with the stream at different places, equal its one-row step (35 rows);
  - grouped verifies of two to five streams of one to three rows, in seven groupings: every row equals the one-row
    batch steps of its stream (46 rows);
  - eight greedy streams of six tokens joining and leaving, so batch steps run 2, 4, 6 and 8 rows: every step's
    logits and token equal the stream's solo decode.
- **The device check** (`rows.zig`): a test checks that this M5 passes it, and another that a product summed in
  another order fails the comparison.
- **The existing contract tests** pass unchanged, among them DFlash2 greedy speculation emitting plain greedy's
  tokens on real prompts and forced drafts that accept all, none or a prefix keeping plain's caches
  (`dflash_test.zig`), and grouped verification equal to batch decoding bit for bit (`group_test.zig`).

The acceptance suite at `48fbf2955`, `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu`,
exits 0 with no test skipped: `inkling-host` passes 35 of 35 tests in 8 s and `inkling-gpu` 151 of 151 in 1,127 s
against both checkpoints and the DFlash2 drafter, every test above among them (`suite-summary.json`,
`suite-inkling-host.log`, `suite-inkling-gpu.log`).

## Decode-path fidelity

The oMLX oracle gate checks teacher-forced prefill logits, which V2 does not touch, and the parity runner's `logits`
mode runs the single-stream forward, which V2 does not touch either. The new runner mode `decode-logits <width>`
teacher-forces each of the 16 parity histories (3,015 positions, two of them past the 1,024-token window) through
the served decode paths from an empty cache: one-row batch steps at width 1, grouped verifies of eight rows that all
commit at width 8. Main's runner (`build-base`, main's kernels plus the mode) and V2's ran both widths;
`fidelity.py` compares each capture with the oMLX oracle (`.work/parity/inkling2-mini/omlx-ref`) and the captures
with each other (`fidelity.json`):

| Capture | Top-1 agreement | Mean KL | Row-mean KL | Max KL | Max \|Δlogit\| p50 / p90 / p99 / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| main, one-row batch steps | 2,865 / 3,015 (95.02%) | 0.020888 | 0.074077 | 14.747 | 1.273 / 5.431 / 16.513 / 26.493 |
| V2, one-row batch steps | 2,865 / 3,015 (95.02%) | 0.020888 | 0.074077 | 14.747 | 1.273 / 5.431 / 16.513 / 26.493 |
| main, 8-row grouped verifies | 2,865 / 3,015 (95.02%) | 0.020888 | 0.074077 | 14.747 | 1.273 / 5.431 / 16.513 / 26.493 |
| V2, 8-row grouped verifies | 2,865 / 3,015 (95.02%) | 0.020888 | 0.074077 | 14.747 | 1.273 / 5.431 / 16.513 / 26.493 |

All four captures are the same bytes: V2 against main at width 1, 3,015 of 3,015 rows; V2's width 8 against its
width 1, 3,015 of 3,015; main's width 8 against its width 1, 3,015 of 3,015 (and every `.bin` file compares equal
under `cmp`). There is no regression to bound. The oracle distances are the existing arithmetic's against oMLX's
bf16 path; the one-row decode path's mean KL (0.0209) sits beside the prefill gate's 0.0195 at chunk 512. Prefill
kernels are unchanged, so the oMLX oracle gate was not rerun.

## Quality

`inkling-quality-dev`, the suite's quality workload, cannot run on this host for main or for V2. Its server, the
native checkpoint at `--parallel 1 --context 32768` with the harness's 2 GiB prompt cache, refuses to start under the
26 GB load limit on both binaries with the same estimate: "Inkling needs 26324461664 bytes including caches and
reserves under a 26000000000-byte load limit; each session fits at most 22866 tokens" (`quality-native-main.log`,
`quality-native-v2.log`, `quality-native-v2-summary.json`). That is main's state, not this change's, and the limit and
context are campaign rules, so it is reported rather than worked around. The workload's `quality-grammar` step also
builds the structured-output library in the frozen baseline worktree `.work/wt/baseline`, which is absent here; the
runs below set `BASELINE_WT` to each binary's own tree, so `libtfgrammar` is built from the pinned xgrammar beside
the binary that loads it.

The same dev set ran on the oQ4e checkpoint instead, as the D2 and P3c lanes recorded it, through
`make -C campaign quality MODEL=inkling2-mini-oq4e SET=dev`, on main's served binary (`zig build native` in the
`efc986f27` export) and on V2's at `48fbf2955`. The server runs `--parallel 1` at temperature 0 with no drafter, so
every generated token is a one-row batch step, where V2 changes only the routed down projection's one-row path; the
grouped and few-row paths are covered by the fidelity captures and the tests above.

| Binary | Passed | Code | Reasoning | Retrieval | Structured |
| --- | ---: | ---: | ---: | ---: | ---: |
| main (`efc986f27`) | 7 of 11 | 2 of 3 | 3 of 3 | 2 of 2 | 0 of 3 |
| V2 (`48fbf2955`) | 7 of 11 | 2 of 3 | 3 of 3 | 2 of 2 | 0 of 3 |

All 11 scored replies are equal in content, reasoning, tool calls and finish reason (`quality_same.py quality-main
quality-v2`; each directory holds the run's `result.json`, `cases.jsonl` and log). On both sides the harness excludes
`retrieval-t05` for its 43,076-token prompt, and the structured cases fail because the engine refuses structured
output and constrained tool choice.

## Where the eight-row verify spends its time

One `dflash-widths` process each on `code-1024` (1,024-token state) under `TF_METAL4_TIMELINE=12`, in the same
session, on main's kernels (`profile-base-1k`) and on V2 (`profile-after-1k`): every width and the one-row batch
step ran 20 times from identical states and the timeline stamped 12 rounds of each after 8 warm ones. A stamp a
dispatch adds about 5 µs, so these spans attribute cost and are not timings. ms a round, main → V2:

| Category | batch | group-1 | group-2 | group-4 | group-8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| routed experts, NVFP4 | 4.20 → 3.95 | 4.11 → 3.97 | 7.24 → 6.96 | 12.49 → 10.88 | 21.09 → 16.49 |
| routed experts, affine 4-bit | 0.21 → 0.21 | 0.31 → 0.20 | 0.50 → 0.49 | 0.90 → 0.93 | 1.60 → 1.62 |
| router, pick, combine | 2.27 → 2.21 | 2.12 → 2.13 | 2.17 → 2.24 | 2.19 → 2.24 | 2.30 → 2.41 |
| attention projections | 2.52 → 2.52 | 2.48 → 2.51 | 2.87 → 2.93 | 4.07 → 3.27 | 6.81 → 3.36 |
| o-projection and head | 1.94 → 1.88 | 1.99 → 1.91 | 2.23 → 2.14 | 2.94 → 2.27 | 5.69 → 2.38 |
| attention | 3.67 → 3.62 | 3.71 → 3.69 | 3.80 → 3.86 | 4.31 → 4.35 | 5.53 → 5.53 |
| convolutions | 2.25 → 2.25 | 2.32 → 2.32 | 2.27 → 2.25 | 2.24 → 2.27 | 2.33 → 2.32 |
| controllers | 1.62 → 1.65 | 1.63 → 1.62 | 1.54 → 1.59 | 1.78 → 2.06 | 2.84 → 2.80 |
| depth mixes and norms | 3.81 → 3.86 | 3.67 → 3.74 | 3.60 → 3.67 | 3.72 → 3.62 | 3.50 → 3.68 |
| other kernels and stamps | 0.14 → 0.15 | 0.14 → 0.14 | 0.31 → 0.30 | 0.37 → 0.19 | 0.38 → 0.19 |
| round span | 22.23 → 21.86 | 22.06 → 21.82 | 26.10 → 25.98 | 34.56 → 31.65 | 51.66 → 40.32 |

At eight rows the round loses 11.3 ms: routed NVFP4 experts −4.6 ms, the fused attention projections −3.5 ms,
head and o-projection −3.3 ms. Its growth from one row is now 18.5 ms (was 29.6): experts +12.5, attention +1.8,
affine 4-bit experts +1.4, controllers +1.2, the projections +0.9, head plus o-projection +0.5 and the router +0.3.
The ranking V1's attribution gave (experts, then head and o-projection, then the fused projections) is the order the
tiles took them in.

## Kernels alone

`dflash-kernels` on the final runner (`kernels-final.jsonl`) and on the runner before the gate/up threshold moved to
three rows (`kernels-v1-tiles.jsonl`, which also times V1's kernels): each projection kind timed alone, 24 calls a
command buffer with the layer rotating per call so each call reads from DRAM, five interleaved rounds, median µs a
call. Routed calls use the 48 visits' real picks from V1's `code-1024` (`picks-code-1024.json`). The one-row and V2
columns are `kernels-final.jsonl`, V1's is `kernels-v1-tiles.jsonl`; where V1 ran the one-row kernel (every kind at
one row, the fused projections at two to four) its column repeats the one-row kernel. The bytes floor is each call's
weights once at 521 GB/s: the head 461 MB, the o-projection 7.5 MB, the fused projections 21.1 MB, and 2.99 MB of
gate/up and 1.49 MB of down for each distinct routed expert, of which there are 8.0, 13.2, 16.9, 20.3 and 30.7 at
one, two, three, four and eight rows.

| Rows | Kind | One-row kernel | V1 | V2 | V2 against V1 | Bytes floor |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | head | 875.4 | 875.4 | 875.4 (one-row kernel) | 1.00× | 885 |
| 1 | o-projection | 19.2 | 19.2 | 19.2 (one-row kernel) | 1.00× | 14 |
| 1 | fused projections | 40.6 | 40.6 | 40.6 (one-row kernel) | 1.00× | 40 |
| 1 | NVFP4 gate/up | 50.9 | 50.9 | 50.9 (one-row kernel) | 1.00× | 46 |
| 1 | NVFP4 down | 35.6 | 35.6 | 28.6 | 0.80× | 23 |
| 2 | head | 1506.9 | 1002.1 | 1001.8 (V1 kernel) | 1.00× | 885 |
| 2 | o-projection | 25.9 | 22.3 | 20.8 | 0.93× | 14 |
| 2 | fused projections | 48.3 | 48.3 | 48.3 (one-row kernel) | 1.00× | 40 |
| 2 | NVFP4 gate/up | 91.5 | 86.7 | 85.4 | 0.99× | 76 |
| 2 | NVFP4 down | 66.6 | 55.0 | 53.9 | 0.98× | 38 |
| 3 | head | 2246.8 | 1165.6 | 1095.4 | 0.94× | 885 |
| 3 | o-projection | 36.5 | 25.4 | 20.9 | 0.82× | 14 |
| 3 | fused projections | 56.9 | 56.9 | 55.1 | 0.97× | 40 |
| 3 | NVFP4 gate/up | 134.0 | 137.1 | 114.7 | 0.84× | 97 |
| 3 | NVFP4 down | 97.6 | 76.1 | 76.7 | 1.01× | 48 |
| 4 | head | 2994.8 | 1407.2 | 1105.9 | 0.79× | 885 |
| 4 | o-projection | 47.3 | 28.9 | 20.9 | 0.72× | 14 |
| 4 | fused projections | 74.0 | 74.0 | 54.9 | 0.74× | 40 |
| 4 | NVFP4 gate/up | 183.8 | 165.6 | 139.4 | 0.84× | 116 |
| 4 | NVFP4 down | 128.5 | 105.2 | 81.8 | 0.78× | 58 |
| 8 | head | 5953.6 | 2611.5 | 1191.0 | 0.46× | 885 |
| 8 | o-projection | 91.5 | 67.2 | 21.0 | 0.31× | 14 |
| 8 | fused projections | 143.7 | 128.2 | 55.3 | 0.43× | 40 |
| 8 | NVFP4 gate/up | 353.1 | 289.6 | 221.0 | 0.76× | 176 |
| 8 | NVFP4 down | 261.1 | 185.4 | 124.4 | 0.67× | 88 |

At eight rows the tiles take the head to 1.35× its floor (V1: 2.95×), the o-projection to 1.46× (4.67×), the fused
projections to 1.37× (3.17×), gate/up to 1.25× (1.64×) and down to 1.42× (2.11×). The V1 column comes from another
process: there the unchanged down tiles measured 68.7 µs at three rows and 133.7 µs at eight against 76.7 and 124.4
here, so differences under about 10% between the V1 and V2 columns are within this bench's spread.

## Verifier cost at widths 1, 2, 4 and 8

V1's protocol: each prompt is prefilled, the rows are the greedy continuation and width w verifies its first w;
grouped forwards leave the caches unchanged and the batch step is rolled back, so every sample starts from the same
state. Nine interleaved repetitions follow one warm-up; p50 of forward plus GPU picks in ms, averaged over the three
prompts of a context and the processes on each side, range over them in brackets. "After GPU p50" is the forward's
own command-buffer span. Every run's grouped widths are bit-equal to the one-row steps that produced their rows
(`dflash-widths` exits non-zero otherwise): 24 of 24 in each process.

**Rested**: four processes in before, after, after, before order (`widths-before-a`, `widths-after-a`,
`widths-after-b`, `widths-before-b`), each after 60 s idle inside the GPU lock (`run.py widths --rest 60`):

| Context | Path | Width | Before p50 | After p50 | Change | After GPU p50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | batch step | 1 | 15.9 (15.8–16.1) | 15.7 (15.6–15.9) | 0.99× | 15.2 |
| 1,024 | grouped | 1 | 16.3 (16.2–16.4) | 16.1 (16.0–16.2) | 0.99× | 15.3 |
| 1,024 | grouped | 2 | 20.2 (20.1–20.4) | 19.9 (19.8–20.0) | 0.99× | 19.2 |
| 1,024 | grouped | 4 | 28.7 (28.5–28.8) | 25.7 (25.4–25.8) | 0.90× | 24.9 |
| 1,024 | grouped | 8 | 45.7 (45.3–46.2) | 34.3 (33.8–34.7) | 0.75× | 33.5 |
| 4,096 | batch step | 1 | 18.3 (17.5–18.9) | 17.5 (16.0–18.6) | 0.96× | 16.9 |
| 4,096 | grouped | 1 | 18.5 (17.3–19.1) | 17.9 (16.3–19.0) | 0.96× | 17.1 |
| 4,096 | grouped | 2 | 22.5 (21.2–23.2) | 21.8 (20.2–23.1) | 0.97× | 21.0 |
| 4,096 | grouped | 4 | 31.7 (30.4–32.4) | 27.9 (26.1–29.4) | 0.88× | 27.1 |
| 4,096 | grouped | 8 | 50.3 (48.0–52.0) | 37.2 (35.0–38.6) | 0.74× | 36.4 |

**Sustained**: each prompt after 20 s of continuous eight-row verifies (`run.py widths --load 20`), four processes
in before, after, after, before order (`widths-before-s`, `widths-after-s`, `widths-after-s2`, `widths-before-s2`):

| Context | Path | Width | Before p50 | After p50 | Change | After GPU p50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | batch step | 1 | 18.4 (17.8–19.4) | 19.0 (18.3–19.8) | 1.03× | 18.4 |
| 1,024 | grouped | 1 | 19.0 (18.2–19.7) | 19.1 (18.5–19.7) | 1.01× | 18.3 |
| 1,024 | grouped | 2 | 23.3 (22.2–24.0) | 23.4 (22.8–24.5) | 1.01× | 22.7 |
| 1,024 | grouped | 4 | 33.8 (32.1–35.3) | 30.0 (29.3–31.3) | 0.89× | 29.2 |
| 1,024 | grouped | 8 | 54.6 (51.3–58.2) | 40.2 (39.0–41.8) | 0.74× | 39.3 |
| 4,096 | batch step | 1 | 19.3 (18.0–20.4) | 19.1 (18.3–19.5) | 0.99× | 18.5 |
| 4,096 | grouped | 1 | 19.8 (17.7–20.9) | 19.5 (18.8–19.9) | 0.99× | 18.7 |
| 4,096 | grouped | 2 | 24.2 (23.2–25.2) | 23.8 (23.2–24.4) | 0.99× | 23.0 |
| 4,096 | grouped | 4 | 34.5 (32.6–36.6) | 30.3 (29.2–31.1) | 0.88× | 29.5 |
| 4,096 | grouped | 8 | 54.9 (52.4–57.1) | 40.8 (38.6–41.9) | 0.74× | 40.0 |

Sustained, the eight-row verify is 0.74× at both contexts and four rows 0.88–0.89×. The one-row batch step measured
1.03× at 1K and 0.99× at 4K, and grouped width 1, the same one-row kernels in the same processes, 1.01× and 0.99×. At
1K the batch step ranges 17.8–19.4 ms over the before processes and prompts and 18.3–19.8 ms over the after ones, and
the sustained state follows each side's own 20 s of eight-row verifies, which V2 runs in 0.74× the time each; rested,
V2's batch step is ahead on all six prompts in both pairs. Main's kernels measured 54.6 ms here at 1K for eight rows
where V1 measured them at 52.7 ms, so only same-session pairs are compared.

## Plain one-row decode

The served plain decode is the one-row batch step, whose kernels are main's except the routed NVFP4 down projection:
each simdgroup of `i2t_down`'s one-row path computes four output rows from one decoded input group, the one-row
kernel's chains with more loads in flight, 28.6 µs a visit against `i2p_down`'s 35.6 (`kernels-final.jsonl`), about
0.3 ms a step over the 44 NVFP4 visits. As the widths runs' batch step above, it is 0.99× main rested at 1K
(15.9 → 15.7 ms) and 0.96× at 4K (18.3 → 17.5 ms); sustained it is 1.03× at 1K and 0.99× at 4K, discussed above.
As the round table's plain greedy, one batch step a token for 511 tokens from each prompt's prefill state (ms a
token, the before runner's `base-q4-a` against the median of `v2-q4-a`, `v2-q4-b` and `v2-q4-c`, range in
brackets):

| Prompt | Before | After | Change |
| --- | ---: | ---: | ---: |
| code-1024 | 16.31 | 16.13 (16.10–16.24) | 0.99× |
| prose-1024 | 16.52 | 16.21 (16.12–16.63) | 0.98× |
| structured-1024 | 17.91 | 17.76 (17.69–17.92) | 0.99× |
| code-4096 | 18.01 | 17.97 (17.77–18.00) | 1.00× |
| prose-4096 | 18.16 | 18.03 (17.97–18.14) | 0.99× |
| structured-4096 | 17.86 | 18.02 (17.97–18.11) | 1.01× |

## Round table

oQ4e target with the q4 drafter: draft depth 7, an eight-row verify and 512 greedy output tokens with EOS ignored, on
DF1's six prompts (`prompts.json`, copied unchanged from V1) and protocol. The after processes `v2-q4-a`, `v2-q4-b`
and `v2-q4-c` ran after the sustained widths, with the before runner's `base-q4-a` between the first two as a
same-session control. Every prompt's tokens equal plain greedy in every process, and rounds, committed tokens and the
output token hashes equal V1's (`v1-q4-a`), as bit equality requires. Times are the median of the three processes'
means, range in brackets; "Spec tok/s before" is V1's `v1-q4-a`, `v1-q4-b` and `v1-q4-c` (main's kernels, an
earlier session). Spec tok/s counts the 511 tokens after prefill's first; plain is one batch step a token from the
same prefill state in the same process.

| Prompt | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain | Spec tok/s before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 88 | 5.81 | 38.8 (38.8–39.8) | 40.2 (40.0–41.7) | 3.94 | 34.9 | 0.01 | 149.5 (145.8–149.7) | 16.13 (16.10–16.24) | 62.0 (61.6–62.1) | 2.41 (2.37–2.42) | 116.6 (116.5–117.0) |
| prose-1024 | 113 | 4.52 | 44.8 (44.3–45.3) | 47.0 (46.7–47.3) | 4.57 | 40.2 | 0.01 | 101.0 (99.9–102.1) | 16.21 (16.12–16.63) | 61.7 (60.1–62.1) | 1.64 (1.64–1.66) | 77.7 (77.2–77.9) |
| structured-1024 | 118 | 4.33 | 46.5 (46.0–46.9) | 48.4 (48.3–48.6) | 4.70 | 41.8 | 0.00 | 93.1 (92.4–94.1) | 17.76 (17.69–17.92) | 56.3 (55.8–56.5) | 1.66 (1.65–1.66) | 74.3 (74.0–74.8) |
| code-4096 | 185 | 2.76 | 47.8 (46.6–47.8) | 50.0 (48.5–50.0) | 4.84 | 42.9 | 0.00 | 57.8 (57.8–59.3) | 17.97 (17.77–18.00) | 55.7 (55.5–56.3) | 1.04 (1.04–1.05) | 46.5 (45.1–46.7) |
| prose-4096 | 190 | 2.69 | 47.3 (47.1–47.7) | 49.6 (49.1–49.6) | 4.83 | 42.4 | 0.00 | 56.9 (56.4–57.1) | 18.03 (17.97–18.14) | 55.5 (55.1–55.6) | 1.02 (1.02–1.03) | 45.2 (44.7–45.5) |
| structured-4096 | 149 | 3.43 | 47.5 (47.4–48.9) | 51.8 (50.1–52.2) | 4.83 | 42.7 | 0.00 | 72.2 (70.2–72.4) | 18.02 (17.97–18.11) | 55.5 (55.2–55.7) | 1.30 (1.27–1.30) | 58.0 (57.6–58.0) |

Phases are mean ms a round. **Verify** holds the grouped forward, the GPU argmax, the acceptance and the commit in one
command buffer; **Host** is the taps' absorption and bookkeeping. **Draft** is unchanged from V1. The verify phase
falls 11.0–13.5 ms a round on every prompt against V1 (45.9 → 34.9 ms at `code-1024`). It is 34.9 ms at `code-1024`,
as the rested eight-row cost, and 40.2–42.9 ms at the other five prompts, where the widths runs' greedy-continuation
rows cost 33.9–38.1 ms.

The same-session control on main's kernels:

| Prompt | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 88 | 5.81 | 49.7 (49.7–49.7) | 50.5 (50.5–50.5) | 3.90 | 45.7 | 0.01 | 116.9 (116.9–116.9) | 16.31 (16.31–16.31) | 61.3 (61.3–61.3) | 1.91 (1.91–1.91) |
| prose-1024 | 113 | 4.52 | 59.6 (59.6–59.6) | 62.0 (62.0–62.0) | 4.64 | 55.0 | 0.01 | 75.8 (75.8–75.8) | 16.52 (16.52–16.52) | 60.5 (60.5–60.5) | 1.25 (1.25–1.25) |
| structured-1024 | 118 | 4.33 | 59.1 (59.1–59.1) | 60.7 (60.7–60.7) | 4.58 | 54.5 | 0.00 | 73.3 (73.3–73.3) | 17.91 (17.91–17.91) | 55.8 (55.8–55.8) | 1.31 (1.31–1.31) |
| code-4096 | 185 | 2.76 | 60.2 (60.2–60.2) | 62.1 (62.1–62.1) | 4.72 | 55.5 | 0.00 | 45.9 (45.9–45.9) | 18.01 (18.01–18.01) | 55.5 (55.5–55.5) | 0.83 (0.83–0.83) |
| prose-4096 | 190 | 2.69 | 60.1 (60.1–60.1) | 61.8 (61.8–61.8) | 4.73 | 55.3 | 0.00 | 44.8 (44.8–44.8) | 18.16 (18.16–18.16) | 55.1 (55.1–55.1) | 0.81 (0.81–0.81) |
| structured-4096 | 149 | 3.43 | 59.9 (59.9–59.9) | 62.0 (62.0–62.0) | 4.71 | 55.1 | 0.00 | 57.3 (57.3–57.3) | 17.86 (17.86–17.86) | 56.0 (56.0–56.0) | 1.02 (1.02–1.02) |

## Decision 17 budgets

Decision 17 allows at most 20, 25, 30 and 40 ms a round at 4, 5, 6 and 8 mean committed tokens, 5 ms a committed
token, which is also 200 tok/s. Measured rounds against that rule (`run.py budgets v2-q4-a,v2-q4-b,v2-q4-c`):

| Prompt | Committed/round | Round ms | Budget at that commitment ms | ms a committed token | Meets |
| --- | ---: | ---: | ---: | ---: | --- |
| code-1024 | 5.81 | 38.8 | 29.0 | 6.69 | no |
| prose-1024 | 4.52 | 44.8 | 22.6 | 9.90 | no |
| structured-1024 | 4.33 | 46.5 | 21.7 | 10.74 | no |
| code-4096 | 2.76 | 47.8 | 13.8 | 17.29 | no |
| prose-4096 | 2.69 | 47.3 | 13.4 | 17.58 | no |
| structured-4096 | 3.43 | 47.5 | 17.1 | 13.84 | no |

No prompt meets its budget. By width, each prompt's full-depth chains cut to the width and the round priced at V2's
measured verify cost (`run.py project`), first rested (`widths-after-a`, `widths-after-b`), the cost the round
table's own verify phase matches (34.9 ms at `code-1024` against 34.3 ms rested):

| Prompt | Width | Committed/round | Draft | Verify+picks p50 | Commit+host | Round ms | Budget ms | tok/s | Meets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| code-1024 | 1 | 1.00 | 0.00 | 16.08 | 0.01 | 16.09 | 5.0 | 62.1 | no |
| code-1024 | 2 | 1.92 | 4.05 | 19.95 | 0.01 | 24.01 | 9.6 | 79.9 | no |
| code-1024 | 4 | 3.45 | 4.05 | 25.80 | 0.01 | 29.86 | 17.2 | 115.5 | no |
| code-1024 | 8 | 5.84 | 4.05 | 34.65 | 0.01 | 38.71 | 29.2 | 150.9 | no |
| prose-1024 | 1 | 1.00 | 0.00 | 16.07 | 0.00 | 16.07 | 5.0 | 62.2 | no |
| prose-1024 | 2 | 1.83 | 4.66 | 19.97 | 0.00 | 24.64 | 9.2 | 74.3 | no |
| prose-1024 | 4 | 3.13 | 4.66 | 25.67 | 0.00 | 30.34 | 15.7 | 103.3 | no |
| prose-1024 | 8 | 4.52 | 4.66 | 34.28 | 0.00 | 38.95 | 22.6 | 116.0 | no |
| structured-1024 | 1 | 1.00 | 0.00 | 16.10 | 0.00 | 16.10 | 5.0 | 62.1 | no |
| structured-1024 | 2 | 1.84 | 4.68 | 19.88 | 0.00 | 24.57 | 9.2 | 74.8 | no |
| structured-1024 | 4 | 2.93 | 4.68 | 25.54 | 0.00 | 30.22 | 14.7 | 97.0 | no |
| structured-1024 | 8 | 4.31 | 4.68 | 33.90 | 0.00 | 38.58 | 21.5 | 111.6 | no |
| code-4096 | 1 | 1.00 | 0.00 | 16.68 | 0.00 | 16.68 | 5.0 | 59.9 | no |
| code-4096 | 2 | 1.69 | 4.71 | 20.40 | 0.00 | 25.11 | 8.4 | 67.2 | no |
| code-4096 | 4 | 2.42 | 4.71 | 26.34 | 0.00 | 31.05 | 12.1 | 77.9 | no |
| code-4096 | 8 | 2.77 | 4.71 | 35.70 | 0.00 | 40.41 | 13.8 | 68.5 | no |
| prose-4096 | 1 | 1.00 | 0.00 | 18.11 | 0.00 | 18.11 | 5.0 | 55.2 | no |
| prose-4096 | 2 | 1.66 | 4.78 | 22.21 | 0.00 | 26.99 | 8.3 | 61.6 | no |
| prose-4096 | 4 | 2.34 | 4.78 | 28.15 | 0.00 | 32.93 | 11.7 | 71.1 | no |
| prose-4096 | 8 | 2.70 | 4.78 | 37.71 | 0.00 | 42.49 | 13.5 | 63.6 | no |
| structured-4096 | 1 | 1.00 | 0.00 | 18.77 | 0.00 | 18.78 | 5.0 | 53.3 | no |
| structured-4096 | 2 | 1.76 | 4.92 | 22.75 | 0.00 | 27.67 | 8.8 | 63.4 | no |
| structured-4096 | 4 | 2.75 | 4.92 | 29.25 | 0.00 | 34.17 | 13.7 | 80.4 | no |
| structured-4096 | 8 | 3.44 | 4.92 | 38.13 | 0.00 | 43.05 | 17.2 | 80.0 | no |

And sustained (`widths-after-s`, `widths-after-s2`):

| Prompt | Width | Committed/round | Draft | Verify+picks p50 | Commit+host | Round ms | Budget ms | tok/s | Meets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| code-1024 | 1 | 1.00 | 0.00 | 18.87 | 0.01 | 18.89 | 5.0 | 52.9 | no |
| code-1024 | 2 | 1.92 | 4.05 | 23.65 | 0.01 | 27.71 | 9.6 | 69.3 | no |
| code-1024 | 4 | 3.45 | 4.05 | 30.47 | 0.01 | 34.53 | 17.2 | 99.9 | no |
| code-1024 | 8 | 5.84 | 4.05 | 40.63 | 0.01 | 44.69 | 29.2 | 130.7 | no |
| prose-1024 | 1 | 1.00 | 0.00 | 19.18 | 0.00 | 19.18 | 5.0 | 52.1 | no |
| prose-1024 | 2 | 1.83 | 4.66 | 23.35 | 0.00 | 28.01 | 9.2 | 65.3 | no |
| prose-1024 | 4 | 3.13 | 4.66 | 29.98 | 0.00 | 34.64 | 15.7 | 90.5 | no |
| prose-1024 | 8 | 4.52 | 4.66 | 40.09 | 0.00 | 44.75 | 22.6 | 101.0 | no |
| structured-1024 | 1 | 1.00 | 0.00 | 19.36 | 0.00 | 19.37 | 5.0 | 51.6 | no |
| structured-1024 | 2 | 1.84 | 4.68 | 23.31 | 0.00 | 27.99 | 9.2 | 65.7 | no |
| structured-1024 | 4 | 2.93 | 4.68 | 29.61 | 0.00 | 34.29 | 14.7 | 85.5 | no |
| structured-1024 | 8 | 4.31 | 4.68 | 39.91 | 0.00 | 44.59 | 21.5 | 96.6 | no |
| code-4096 | 1 | 1.00 | 0.00 | 19.77 | 0.00 | 19.77 | 5.0 | 50.6 | no |
| code-4096 | 2 | 1.69 | 4.71 | 24.27 | 0.00 | 28.99 | 8.4 | 58.2 | no |
| code-4096 | 4 | 2.42 | 4.71 | 30.71 | 0.00 | 35.42 | 12.1 | 68.3 | no |
| code-4096 | 8 | 2.77 | 4.71 | 41.41 | 0.00 | 46.12 | 13.8 | 60.0 | no |
| prose-4096 | 1 | 1.00 | 0.00 | 19.67 | 0.00 | 19.68 | 5.0 | 50.8 | no |
| prose-4096 | 2 | 1.66 | 4.78 | 23.89 | 0.00 | 28.66 | 8.3 | 58.0 | no |
| prose-4096 | 4 | 2.34 | 4.78 | 30.64 | 0.00 | 35.42 | 11.7 | 66.1 | no |
| prose-4096 | 8 | 2.70 | 4.78 | 41.41 | 0.00 | 46.19 | 13.5 | 58.5 | no |
| structured-4096 | 1 | 1.00 | 0.00 | 19.17 | 0.00 | 19.17 | 5.0 | 52.2 | no |
| structured-4096 | 2 | 1.76 | 4.92 | 23.31 | 0.00 | 28.23 | 8.8 | 62.2 | no |
| structured-4096 | 4 | 2.75 | 4.92 | 29.58 | 0.00 | 34.50 | 13.7 | 79.7 | no |
| structured-4096 | 8 | 3.44 | 4.92 | 39.72 | 0.00 | 44.64 | 17.2 | 77.1 | no |

Per width, against the budget its most committed tokens allow (5 ms a token), from the two projections:

| Width | Most committed a round | Budget at that | Round ms rested, 1K | Rested, 4K | Sustained | Budgets it can meet |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 5 ms | 16.1 | 16.7–18.8 | 18.9–19.8 | none |
| 2 | 2 | 10 ms | 24.0–24.6 | 25.1–27.7 | 27.7–29.0 | none |
| 4 | 4 | 20 ms | 29.9–30.3 | 31.1–34.2 | 34.3–35.4 | none |
| 8 | 8 | 40 ms | 38.6–39.0 | 40.4–43.1 | 44.6–46.2 | 40 ms at 8 committed, rested at 1K only |

Widths 1, 2 and 4 cannot meet a Decision 17 budget at any acceptance: their verifies alone take 16.1, 19.9 and
25.7 ms rested at 1K, more than the 5, 10 and 20 ms their most committed tokens allow. Width 8 now can in principle,
rested at 1K only: its 38.6–39.0 ms round fits the 40 ms budget for eight committed tokens, which needs 7.7–7.8
committed tokens a round, nearly every draft accepted, where the best prompt commits 5.84; it can meet none of the
20, 25 and 30 ms budgets at four to six. Rested at 4K and sustained its round exceeds 40 ms. At the measured
acceptance the fastest width is 8 on the 1K prompts and 4 at 4K in both projections (`structured-4096` ties rested).
The projection prices every width at the widths runs' greedy-continuation rows, which the round table's own verify
exceeds by 4.6–7.9 ms on the five prompts other than `code-1024`, so the projected rounds there are optimistic.

## What still bounds the eight-row verify

Rested at 1K the eight-row verify takes 34.3 ms against its 17.0 ms weight floor (6.60 GB of distinct experts and
2.27 GB of everything else at 521 GB/s). By the timeline, ranked by what an eighth row costs over one:

- **Routed NVFP4 experts, +12.5 ms**, 16.5 ms at eight rows against an 11.6 ms floor for their 6.05 GB. Most of the
  growth is the floor's own: eight rows route 30.7 distinct experts a visit where one row routes 8.0, which takes
  the floor from 3.0 to 11.6 ms. The rest, 3.9 ms, is the kernels' excess over their bytes. At eight rows of real
  picks gate/up runs at 1.25× its floor and down at 1.42×, and an expert with eight live rows costs about twice its
  bytes: eight experts each routed all eight rows take 97 µs of gate/up against a 46 µs floor and 55 µs of down
  against 23 (`harness-experts.log`). The products themselves are about half of that: the gate/up's 663,552 8x8x8
  products take 42 µs at the measured 16.2 TFLOP/s.
- **Attention, +1.8 ms.** Each row's split kernel stages its stream's keys and values again; staging them once per
  stream would change the rows' per-split order (above).
- **Affine 4-bit experts (layers 20–23), +1.4 ms; controllers, +1.2 ms; the router, +0.3 ms.** These run the
  one-row kernels per row or per slot.
- **The fused projections, +0.9 ms; head and o-projection, +0.5 ms**, now 1.35–1.46× their bytes at eight rows.

## Conditions

- **Lock and gates.** Every timed run, model load and Metal test went through `campaign/heavy.py` (the GPU lock and
  the 28 GB cap). Each `run.py` run passed `gpu-gate-ready -1` (60% limits) and `perf-gate-ready`
  (`PERF_GATE_MAX_BUSY_CORES=4`) before and after; readings are in each `NAME.host.json`. They show 20–34% of
  GPU time in other processes around the runs, the terminal and WindowServer, as in V1. `INKLING_LOAD_LIMIT_GB=26`;
  contexts at most 4,096 + 16 tokens in the timed runs and 32,768 for the quality server.
- **Shared host.** DF2a and a P4 prefill lane ran their own suites, oracle gates and benches on the same lock
  throughout. A first session took the kernel bench and both profiles and was stopped before its rested runs, which
  would have started straight after the P4 lane's oracle run; the rested runs were taken again with their idle
  minute inside the lock.
- **Thermal.** `pmset -g therm` recorded no thermal or performance warning before or after any run.
- **Build identities.** Before: runner `915f8cae…`, built from `git archive efc986f27` plus this lane's
  `steps_run.zig` mode. After: runner `d602ae6e…`, built at `820662dde`. `773b4fcd9` changes a doc comment and
  `48fbf2955` only how the pipelines are opened, so later commits time the same kernels.

## Reproduction

`build-base` is `zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix DIR` in a `git archive efc986f27` export
with this lane's `zig/src/families/inkling2_loop/steps_run.zig` copied in and the two `decode-logits` lines this lane
adds to `run.zig` applied (main's kernels, V2's measurement mode); `build-c` is the same command in this worktree at
`820662dde`. `BASE_EXPORT` is that export after `zig build native -Dcpu=apple_m1`, main's served binary.

```sh
R0=BASE_DIR/bin/tf-inkling2-loop-run
R1=AFTER_DIR/bin/tf-inkling2-loop-run
D=campaign/kernels/inkling-v2-arith-runs
M=~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd
F=OUT_DIR
INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- $R1 $M $D/code-1024.json dflash-kernels 24 \
  $D/kernels-final.jsonl $D/picks-code-1024.json
python3 -I $D/run.py widths $R0 $D/code-1024.json profile-base-1k --reps 20 --timeline 12
python3 -I $D/run.py widths $R1 $D/code-1024.json profile-after-1k --reps 20 --timeline 12
python3 -I $D/run.py attribution profile-base-1k.log
python3 -I $D/run.py attribution profile-after-1k.log
for run in "$R0 widths-before-a" "$R1 widths-after-a" "$R1 widths-after-b" "$R0 widths-before-b"; do
  set -- $run
  python3 -I $D/run.py widths $1 $D/prompts.json $2 --reps 9 --rest 60
done
for run in "$R0 widths-before-s" "$R1 widths-after-s" "$R1 widths-after-s2" "$R0 widths-before-s2"; do
  set -- $run
  python3 -I $D/run.py widths $1 $D/prompts.json $2 --reps 9 --load 20
done
python3 -I $D/run.py compare widths-before-a,widths-before-b widths-after-a,widths-after-b
python3 -I $D/run.py compare widths-before-s,widths-before-s2 widths-after-s,widths-after-s2
python3 -I $D/run.py bench $R1 $D/prompts.json v2-q4-a --bits 4 --tokens 512
python3 -I $D/run.py bench $R0 $D/prompts.json base-q4-a --bits 4 --tokens 512
python3 -I $D/run.py bench $R1 $D/prompts.json v2-q4-b --bits 4 --tokens 512
python3 -I $D/run.py bench $R1 $D/prompts.json v2-q4-c --bits 4 --tokens 512
python3 -I $D/run.py roundtable v2-q4-a,v2-q4-b,v2-q4-c \
  --against ../inkling-v1-verify-runs/v1-q4-a,../inkling-v1-verify-runs/v1-q4-b,../inkling-v1-verify-runs/v1-q4-c
python3 -I $D/run.py roundtable base-q4-a
python3 -I $D/run.py budgets v2-q4-a,v2-q4-b,v2-q4-c
python3 -I $D/run.py project v2-q4-c widths-after-a,widths-after-b
python3 -I $D/run.py project v2-q4-c widths-after-s,widths-after-s2
for side in "$R0 before" "$R1 after"; do set -- $side; for w in 1 8; do
  INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- $1 $M campaign/parity/inkling2-mini/histories.json \
    decode-logits $w $F/$2-w$w
done; done
uv run --frozen --project campaign/ref python -I -B $D/fidelity.py \
  --histories campaign/parity/inkling2-mini/histories.json --ref .work/parity/inkling2-mini/omlx-ref \
  --cand before-w1=$F/before-w1 --cand after-w1=$F/after-w1 --cand before-w8=$F/before-w8 --cand after-w8=$F/after-w8 \
  --same after-w1=before-w1 --same after-w8=after-w1 --same before-w8=before-w1 --json $D/fidelity.json
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
MAKEFLAGS="BASELINE_WT=$PWD" python3 -I campaign/suite/suite.py run --names inkling-quality-dev
make -C campaign quality MODEL=inkling2-mini-oq4e SET=dev BIN=BASE_EXPORT/zig-out/native/bin/tensorfold-native \
  BASELINE_WT=BASE_EXPORT
make -C campaign quality MODEL=inkling2-mini-oq4e SET=dev BIN=$PWD/zig-out/native/bin/tensorfold-native \
  BASELINE_WT=$PWD
python3 -I $D/quality_same.py $D/quality-main $D/quality-v2
```

`run.py widths --rest 60` idles 60 s inside the GPU lock before its admission gates, so a rested run cannot start
straight after another lane's model load. The MMA probes are Swift programs: `swiftc -O $D/mma-probe.swift -o P &&
python3 campaign/heavy.py -- ./P 12` (12 is the operand spread), and `mma-edges.swift` and `mma-lanes.swift` the same
way without an argument. `harness-experts.log` comes from a development harness that timed expert-kernel variants on
the same picks; its source is not part of this record. Each quality run writes `campaign/quality/results/RUN/`; the
two oQ4e runs' directories were moved to `quality-main/` and `quality-v2/` here, and the tracked `latest-*` pointer
restored.

## Open problems

- The 26–30 ms target for eight rows rested at 1K is not met: 34.3 ms (45.7 on main). The ranked remainder is under
  "What still bounds the eight-row verify".
- An expert routed four or more rows (gate/up) or two or more (down) pays for 8-row tiles whatever its live rows, at
  up to twice its bytes. A cheaper tile path for experts with many rows was not found: wider tiles per simdgroup,
  transposed `simdgroup_load` of the input, a byte-indexed code-pair table and inputs staged in threadgroup memory
  were all slower in the development harness.
- Attention, the controllers, the router and the affine 4-bit experts still run once per row at 2–8 rows.
- The tiles' bit equality rests on a property measured on this M5 Max. Other Apple GPUs keep the one-row kernels if
  they fail the device check; whether M1–M4 pass it is untested here (**Needs environment**).
- Sustained at 1K the one-row batch step measured 1.03× main in two interleaved pairs, inside both sides' spread,
  while rested it is ahead on every prompt. The sustained protocol loads each side with its own eight-row verifies; a
  load that is the same on both sides would settle whether the difference is real.
- **Shared blocker:** the suite's `inkling-quality-dev` cannot start on main or V2 here. The native checkpoint at
  32,768 tokens with the harness's 2 GiB prompt cache needs 26.32 GB under the 26 GB load limit, and every quality
  target's `quality-grammar` step needs the frozen baseline worktree `.work/wt/baseline`, which this host lacks.
- The new kernels are not registered in the Metal kernel-contract runner; none of the family's kernels is.
- The 4K acceptance (2.7–3.4 committed tokens a round) is DF1's and unchanged; DF2a owns it.
