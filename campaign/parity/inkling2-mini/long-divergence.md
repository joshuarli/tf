# Inkling 2 Mini long-history divergence: explained verdict

Diagnosis only. The row and router captures do not change inference arithmetic.

## Verdict

Both long-history divergences are **accumulation order**, not a defect. No kernel fix is
needed, and no kernel arithmetic was changed.

- `harbour` position 556: the first tensor where Metal departs from the float32 reference is
  **l_out-38** (execution slot 38: second extra pass over the core, core position 8, physical
  layer 8, a sliding-window layer). The reference's own router 8th/9th selection gap there is
  **2.98e-08**, the unique near-tie across rows 0-556, while the two implementations agree to
  ~5e-05 on every earlier tensor. Forcing the reference's 8th pick from expert 34 to expert 51
  reproduces Metal's logits to **9.1e-05** (unforced: 1.14). Metal took the other side of a tie
  that rounding order decides.
- `harbour` position 557: the first tensor is **l_out-24** (first extra pass, core position 6,
  physical layer 6, sliding-window). Gap **1.34e-07**. Forcing both picks reproduces Metal to
  **4.6e-05** (unforced: 1.61); forcing only the 556 pick leaves exactly the oracle's 0.0616
  residual. The corrupted KV/convolution state then persists: every one of the 779 positions
  557-1335 stays over 1e-3, including confident top-1 flips.
- `cells_code` position 304: the first tensor is **l_out-47** (the last executed slot: second
  extra pass, core position 17, physical layer 17, full attention). The gap there is 2.0e-03,
  forcing the 9th expert in reproduces nothing (0.29, not 0.00137), and all eight picked experts
  plus the 9th were used at that slot in rows 0-303 with zero error, so this is neither a router
  flip nor a weight-decode defect. Elementwise, all 2,304 inputs agree to <=5.9e-04 while 40
  output elements reach 1.5e-03: ordinary MoE amplification of accumulated drift (~1.6e-04 at the
  303 control), whose state echo damps back to 8.8e-05 at 305. The 656-694 regime (max 0.68
  at 690, top-1 still 100%) is drift plus two reference-side router ties, diagnosed in
  the cells_code 656-694 section below: a tie at slot 21, row 473 (gap ~3e-8) whose echo
  damps into the drift floor, and a tie at slot 21, row 690 (gap ~1.5e-7) behind the peak.
  In prefix-matched comparisons Metal never departs from the reference beyond their
  mutual spread; the discrete jumps exist only between the reference's own
  full-history and prefix batch shapes.

Hypotheses closed out: the relative-bias/attention index idea is excluded arithmetically (onset
positions 304/556/557 sit below the 1,024 relative-bias extent and sliding-window thresholds, and
each onset's attention output agrees while its slot MoE output diverges); NVFP4/MXFP8 decode is
excluded by clean prior usage of the same experts at the same slots (harbour 34: 139x, 51: 44x;
cells_code slot-47 experts: all used in rows 0-303 with the 303 control fully agreeing to
1.6e-04); a convolution/state-carry origin is excluded by clean agreement through 555/303 with a
discrete jump at the onset, while state carry is confirmed as the propagation mechanism.

A side finding, resolved during the work: the saved oracle's `harbour` rows >= 556 do not match
fresh reference prefix runs (1.14 at 556). A full-history reference rerun reproduces the oracle
bit-for-bit (max diff 0.0), so the oracle is sound; the prefix/full difference is the same
coin-flip mechanism (a 3e-8 tie decided by batch-shape rounding), not a reference bug. The
forensics below compare identical inputs on both sides throughout.

## First divergent tensors, with numbers

Execution order per position is slots 0-17 (prelude plus first core pass), 24-35 (second pass),
36-47 (third pass), 18-23 (tail); physical layer is the slot number below 24 and
6 + (slot - 24) % 12 above it. The per-position traces fold only the named position
(runner `trace <out> <at>`, reference `--trace-row`), so each line below is one position's
2,304 hidden values, reference against Metal.

### harbour 556: l_out-38

171 overlapping tensors agree within 1e-3 first (all of execution slots 0-17, then slot 38's
three mixed inputs, three normed inputs, attention output and FFN input), then:

- l_out-38: reference (mean 0.02863518, std 1.849855, min -25.76209, max 54.65753) against
  Metal (mean 0.02853821, std 1.852335, min -25.93238, max 54.71870), largest difference
  1.703e-01 (in the minimum).
- The slot-38 MoE inputs agree to ~1e-07 (mixin-38-0: 0.02881603 vs 0.02881602;
  ffn_inp-38: 0.03085151 vs 0.03085143), so the origin is inside the slot-38 MoE.
- Reference router, slot 38 (layer 8): picks [42, 124, 80, 52, 97, 82, 36, 34] against 9th
  expert 51, 8th/9th selection gap 2.9802322e-08, the only row in 0-556 under 1e-6. Both
  experts decoded cleanly in 139 (34) and 44 (51) prior uses at this slot.
- Forced reproduction (`--force-pick 38,51,556` on the 557-token prefix): reference-vs-Metal
  row-556 logit error 9.0599e-05, down from 1.1440034 unforced. Row 555 stays 4.6e-05 either
  way; row 557 goes 1.6113992 -> 0.0616464, exactly the oracle comparison's 557 number.

### harbour 557: l_out-24

On the 558-token prefix, 170 overlapping tensors agree within 1e-3 first, then:

- l_out-24: reference (mean 0.01062705, std 0.6998027, min -12.48173, max 19.33292) against
  Metal (mean 0.01030361, std 0.7039530, min -12.48278, max 19.76026), largest difference
  4.273e-01 (in the maximum). ffn_inp-24 agrees to ~1e-08, so again the origin is the slot
  MoE.
- Reference router, slot 24 (layer 6): picks [67, 33, 66, 50, 78, 64, 23, 125] against 9th
  expert 22, gap 1.3411045e-07 (prob gap 0.00997: the bias closes a 1%-scale prob margin to
  1e-7). Both experts used cleanly before (125: 35x, 22: 27x, last at rows 541/537). Row 450
  holds a 4.3e-07 tie at this slot that did not manifest in the oracle comparison.
- Forced reproduction (`--force-pick 38,51,556 --force-pick 24,22,557`): reference-vs-Metal
  row-557 logit error 4.5776e-05, down from 1.6113992 unforced (4.6e-05 at row 556). Both
  flips jointly explain the full divergence; the single-forced residual (0.0616464) equals the
  oracle number, confirming the decomposition into state carry plus the second flip.

### cells_code 304: l_out-47

377 overlapping tensors agree within 1e-3 first (all execution slots through 46), then:

- l_out-47: reference (mean 0.1393519, std 3.949756, min -30.64513, max 43.11888) against
  Metal (mean 0.1393489, std 3.949716, min -30.64463, max 43.11749), largest difference
  1.391e-03 (in the maximum). The 303 control agrees on all 432 tensors (max 1.62e-04).
- Reference router, slot 47 (layer 17): picks [79, 86, 89, 108, 47, 120, 13, 9] against 9th
  expert 43, gap 2.0275e-03, ~40x above the implementations' mutual noise floor.
- `--force-pick 47,43,304` gives a 0.2946 logit error, not the observed 0.00137, so Metal did
  not take the 9th expert. All eight picked experts and expert 43 appear at slot 47 in rows
  0-303 with no error, so no decode defect is hiding in them either.
- Elementwise (raw rows on both sides): all eight slot-47 input tensors agree everywhere
  (max 5.9e-04 on mixin-47-2, zero elements over 1e-3), while l_out-47 differs on 40 elements
  (largest: index 2246, ref -0.9838784 vs Metal -0.9853932; index 188: -4.718834 vs -4.720299;
  the row maximum 43.11888 vs 43.11749). Same picks, same decoded weights, small dense input
  difference in, sparse 1.5e-03 out: MoE forward amplification of drift, not a discrete event.
  The state echo is visible and damped: row-305 error 8.8e-05.

### cells_code 656-694: reference-side router ties at rows 473 and 690

Fresh full-history numbers below use the integrated main runner
(`zig-out/bin/tf-inkling2-loop-run` sha256
`eaea2a5d727307b1aeaf7ca17e900af47d287961bf60295584a2519d325faf21`)
wrote to a lane-private directory, against the saved reference logits
(`.work/parity/inkling2-mini/ref/cells_code.npy`, sha256
`8fb4a4f2d156906f9ff1bf95188dff32bf02b154b182d1ada3d89a23fe9e940d`;
D1 recorded no digest for this file). Overall max abs logit error 0.6775174 at
position 690, top-1 agreement 100% over all 1,178 positions. Post-304 drift peaks
at 9.24e-04 (row 327); rows 328-655 never exceed 6.99e-04 (row 337) and rows
600-655 never exceed 1.74e-04. Row 656 (9.034e-04) is the first position since
the 327 spike to approach 1e-3, ~5x the 600-655 envelope; rows 688/689 sit at
4.84e-04/3.35e-04, then 690 explodes to 0.6775174 and the echo decays (691:
0.074331, 692: 0.016438, 693: 0.005901, 694: 0.001617).

Onset (656): ordinary drift on the same discrete path, no Metal departure. The
prefix-matched comparison (both sides fold row 656 of the 657-token prefix from
empty state) agrees on all 432 overlapping tensors within 1e-3, and raw,
elementwise, on all 432 x 2,304 values: max 8.81e-04 (ffn_inp-23), median
4.18e-05, p90 2.21e-04, zero values over 1e-3, against the 655 control's max
1.26e-04 (median 1.14e-05). There is no first tensor where Metal departs beyond
the implementations' mutual spread; the 656 blip is a diffuse drift peak, the
same kind as position 304. A reference-internal tie contributes to the
reference side of the comparison: slot 21 (tail layer 21, sliding-window) at row
473 has selection gap 2.98e-08 in the 657-prefix batch vs 4.47e-08 in the
full-history batch, the unique near-tie at that row (next-tightest anywhere is
slot 5 at 3.0e-04), with the 657-prefix picking 8th expert 88 and the full run
picking 89 (9th expert swapped accordingly). Forcing the prefix run onto the
full side (`--force-pick 21,89,473`) reproduces the full-history row 473 to
4.10e-05 (unforced: 0.6320047) and tracks it to ~4-6e-05 through row 600, while
by row 656 the forced run rejoins the same ~1.3e-03 batch-shape drift floor as
the unforced prefix run: the 473 echo damps out and does not explain row 656.
Metal demonstrably takes the full-history side of that tie: the fresh Metal
full-history run tracks the saved oracle to ~4-10e-05 across rows 473-500,
where the 88-side prefix run sits 0.63 away at row 473. So both compared sides
share one discrete path through 656; the 9.03e-04 is inter-implementation
drift, not a flip.

Peak (690): a demonstrated routing discontinuity, and it sits inside the
reference's own batch shapes. The prefix-matched comparison agrees everywhere:
logits to 4.36e-05, all 432 overlapping summary tensors within 1e-3, raw rows
max 1.52e-04 (median 1.05e-05) — while both sides disagree with the saved
full-history oracle row by 0.6775. The reference's slot-21 router at row 690 is
a 1.639e-07 tie (prob gap 0.0164, the bias closing a 1.6%-scale prob margin to
1e-7, as at harbour 557): the 691-prefix run picks
[97, 24, 53, 72, 43, 17, 85, 125] against 9th expert 120, while the
full-history run picks 8th expert 120 against 9th 125 (gap 1.341e-07); nearby
slots are ~1000x wider (slot 2: 1.43e-04, slot 20: 3.91e-04, slot 0: 7.32e-04,
slot 22: 8.27e-04). Forcing the prefix run onto the full side
(`--force-pick 21,120,690`) reproduces the oracle row 690 to 9.1553e-05, down
from 0.67751 unforced, top-1 37860 on all three. Forced-vs-unforced prefix
traces agree on the first 413 overlapping tensors and differ first at
**l_out-21** (3.714e-02 in the summaries), so the MoE at slot 21, row 690 is
the origin. Elementwise: all eight slot-21 input tensors are identical between
forced and unforced (0.0) and agree with Metal to <=2.22e-05, while l_out-21
differs by max 0.5408 between the two sides (worst element 1549: 125-side
7.5828648 vs 120-side 7.0420713; Metal 7.58287 matches the 125 side to
1.36e-05). As at row 473, slot 22's row-690 pick follows the flip (29 vs 53 at
a non-tie gap) as downstream state carry, not a second tie. Metal behaviourally
takes the 125 side (it matches the 125-side prefix run to 4.4e-05 logits); the
oracle's 120 side is the reference's own batch-shape coin flip.

A full-history reference rerun (1,178 tokens, peak 18.9 GB) reproduces the saved
oracle bit-for-bit (max diff 0.0), so the oracle is sound, and a fresh Metal
prefix-690 logits run reproduces the Metal full-history row 690 bit-for-bit
(0.0): neither side has a state-handling defect. Between the two discrete
ties, the 656-694 span is continued drift amplification of the position-304
kind, with the 690 tie's state echo decaying through 694 and a raised floor
(whole-history top-1 still 100%) carried downstream.

## How to reproduce

Every model load below ran through `python3 /Users/josh/repos/tensorfold-m5/campaign/heavy.py --`,
with `TMPDIR=/Users/josh/repos/tensorfold-m5/.work/tmp` and
`ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tensorfold-m5/.work/zig-global-cache`.
`$M` is `/Users/josh/models/Inkling-2-Mini@ed40e7da625f`,
`$D` is `/Users/josh/repos/tensorfold-m5/.work/parity/inkling2-mini/metal/trace`,
`$R` is `/Users/josh/repos/tensorfold-m5/.work/parity/inkling2-mini/trace-ref`,
`$V` is `/Users/josh/repos/tensorfold-m5/.work/venv-ref/bin/python`.
Build: `zig build tf-inkling2-loop-run -Dcpu=apple_m1` (exit 0).

- Metal onset/control traces (exits 0; peaks 0.9/0.9/0.6/0.6 GB):
  `python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-loop-run $M $D/harbour.p557.tokens.json trace $D/harbour.p557.trace 557`
  and the same for `harbour.p556 … 556`, `cells_code.p304 … 304`, `cells_code.p303 … 303`
  (prefix token files hold ids 0..pos).
- Reference onset/control traces with row stats, router gaps and raw rows (exits 0; peaks
  16.2/16.2/15.3/15.3 GB):
  `python3 campaign/heavy.py -- $V campaign/ref/inkling2_loop_ref.py --model $M --tokens "<ids 0..557>" --out $R/harbour.p557B.npy --trace $R/harbour.p557B.trace --trace-row 557 --router-report $R/harbour.p557B.routers.json --row-dump $R/harbour.p557B.rows.npz --expert-cache-gib 10`
  (likewise p556/cells; `--tokens` takes the space-separated prefix).
- Forced-pick reproductions (exits 0; peaks 16.1/16.1/15.3 GB):
  `… --out $R/harbour.p557F38b.npy --trace-row 556 --row-dump $R/harbour.p557F38b.rows.npz --force-pick 38,51,556 …`
  `… --out $R/harbour.p557F2.npy --trace-row 557 --force-pick 38,51,556 --force-pick 24,22,557 …`
  `… --out $R/cells_code.p304F47.npy --trace-row 304 --row-dump $R/cells_code.p304F47.rows.npz --force-pick 47,43,304 …`
  then compare each forced `.npy` row against the Metal prefix logits
  (`tracerow`-less `logits` runs: harbour 558 rows 73 s / 0.9 GB, cells_code 305 rows 36 s /
  0.6 GB) and against the unforced reference prefix.
- Reference full-harbour rerun: same reference command with all 1,336 ids, no flags
  (exit 0, peak 19.2 GB); `harbour.full-rerun.npy` matches the saved oracle bit-for-bit
  (max diff 0.0), proving the oracle sound and the prefix/full difference a tie flip.
- cells_code 656-694 forensics (all exits 0; `$W` is the lane-private
  `/Users/josh/repos/tensorfold-m5/.work/tmp/w1/` holding every file this run wrote):
  fresh full-history Metal logits `cells_code.metal.bin` (1,178 rows in 46.5 s, peak
  1.6 GB; sha256 `77e41236fb6355a145d286e567ee34be6d26fa8cfa7736919ceafb56b7540ab9`)
  and Metal prefix traces/rows (`trace`/`tracerow` at 655, 656, 689, 690; peaks 1.2/0.6 GB);
  reference prefix runs with `--trace-row`, `--router-report`, `--row-dump` for 655, 656,
  689, 690 (peaks 18.5-19.7 GB); a full-history reference run with `--router-report`
  (peak 18.9 GB, reproduces the saved cells_code oracle bit-for-bit, max diff 0.0);
  forced runs `--force-pick 21,120,690` on the 691-prefix (peak 18.6 GB, row-690
  reproduction 9.1553e-05) and `--force-pick 21,89,473` on the 657-prefix (peak
  19.6 GB, row-473 reproduction 4.101e-05); a Metal prefix-690 `logits` run (peak
  1.2 GB) reproducing the Metal full-history row 690 bit-for-bit (0.0).
- Harness proof: `uv run --frozen --project campaign/ref python
  campaign/parity/inkling2-mini/compare.py --histories <harbour+cells_code subset> --ref-dir
  <ref bins> --metal-dir <metal bins> --report <report> --runner … --model $M --heavy … --ref-venv …
  --reference campaign/ref/inkling2_loop_ref.py` (exit 0; the "first divergent tensor" and
  "clean control" lines in the Divergences section now name tensors with both sides' numbers
  instead of "absent from the Metal trace").

## Changes (diagnosis scaffolding only)

- `zig/src/families/inkling2_loop/run.zig`: `trace` accepts an optional position to fold only
  that position; new `tracerow` mode writes the raw per-tensor rows of one position.
- `campaign/parity/inkling2-mini/compare.py`: onset prefixes are traced on both sides (reference
  `--trace-row`, runner `trace <at>`) with an onset-1 clean control, compared in execution order
  over the overlapping slot-indexed tensors, with router-gap context; per-disagreement lines stay.
- `campaign/ref/inkling2_loop_ref.py`: `--router-report` (per-slot picks plus 8th/9th gaps for
  every row), `--force-pick SLOT,EXPERT[,ROW]`, `--row-dump` (full last-row vectors).

## Residual risks and follow-ups

- Row 450's 4.3e-07 tie at slot 24 left no mark on the oracle comparison; ties flip
  stochastically and only sometimes propagate visibly. A future tie could flip the other way on
  a rebuild with different kernel order; the parity gate tolerates this (confident agreement
  stays >= 99.5%) but exact-logit reproducibility across implementations is not achievable at
  ties. The cells_code regime adds two instances: the 656- and 657-token prefix
  batches resolve the row-473 tie on opposite sides (88 vs 89) while the
  full-history batch sides with the 656-token prefix, so a different prefix length
  alone moves the echo.
- The cells_code 656-694 regime is now diagnosed (routing discontinuities at slot 21,
  rows 473 and 690, plus position-304-kind drift amplification between and after
  them). Fresh Metal builds still drift from each other inside the regime
  (fresh-vs-stale Metal max 9.24e-04 at row 908) while agreeing at the discrete events.
- During this run the shared `.work/parity/inkling2-mini/metal/*.bin` files were deleted and
  regenerated by concurrent harness activity; all numbers above come from files this run wrote
  itself or verified by hash (saved harbour oracle md5 6d39491b57d480d00ef4560f47e3c475,
  unchanged; full rerun matches it at 0.0).

## structured-1024 served divergence (v0 against main f6d8f83f0)

Served greedy ids for `campaign/prompts/dev/structured-1024.txt` (1,124 prompt tokens) match v0 for 48 generated
tokens and then differ: v0 emits 150161, main 153966. The float32 reference
(`campaign/ref/inkling2_loop_ref.py --tokens <prompt + first 48 generated ids>`) on the 1,172-token context
ranks 153966 first at 26.91493 and 150161 second at 26.82265, a 0.092 margin, so main's choice is the
reference's and v0's was the error. On the last row, main's chunked prefill path (`prefill 128`) is within
1.5e-4 of the reference and its one-position decode path (`logits`) within 1.5e-2. Peak charged memory for the
reference run was 24.7 GB.


## Decode drift

For the 1,172-token structured history above, the decode and 128-token prefill paths take
different sides of a router near-tie at row 581, slot 38. This is their first difference in
the set of eight selected experts in rows 0–723, across all 48 slots in execution order.
The prefill sets match the full-history float32 router report throughout those rows.
The first ordered-pick difference is earlier, at row 288, slot 28, where both select
the same experts but swap experts 54 and 84 in the second and third positions.

| Path | Eighth expert and selection score | Ninth expert and selection score | Gap |
| --- | --- | --- | --- |
| Decode | 28, 0.15397964 | 98, 0.15397863 | 1.01e-6 |
| Prefill | 98, 0.15398206 | 28, 0.15397675 | 5.30e-6 |
| Float32 reference | 98 | 28 | 5.47e-6 |

Selection scores are sigmoid router logits plus the router bias. The scores in the first
two rows come from the native router inputs and bias captured at the slot. The reference
report records the gap and picks, rather than individual selection scores. Both native
choices are consistent with rounding a near-tie. No kernel arithmetic was changed.

At row 581, the decode and prefill captures differ by at most 1.82e-4 at slot 38
`ffn_inp`, then by 0.43284 at `l_out` after the different expert selection. The first
difference above 1e-4 in execution order is slot 28 `mixin-2`, at 1.41e-4. The first
above 1e-2 is slot 38 `l_out`. The full-history reference supports prefill at this
routing decision: its row-581 maximum logit error is 6.98e-4 for prefill and 0.48634
for decode. The native logit difference stays above 0.00333 on every row 581–723,
consistent with the recurrent state carrying the divergent expert output.

At row 723, the first captured decode/prefill tensor difference above 1e-4 in execution
order is slot 26 `mixin-2`, at 1.75e-4. The first above 1e-2 is slot 39 `attn_out`,
at 0.01674. Slot 44 then changes its eighth expert from prefill's 98 to decode's 95.
The full-history reference selects 98 over 95 at that slot, with a 0.004518 gap. The
slot-44 `l_out` values differ by 1.8566, and later captured rows differ by up to 18.167.
The full-history logits at row 723 differ from reference by 6.21e-4 for prefill and
3.33951 for decode. The reference router and logits support prefill at the amplified
slot and final output. They do not establish which path is closer elementwise at the
first small slot-26 tensor difference.

At row 719, slot 41, decode chooses 98 over 93 with scores 0.34984463 and 0.34945092.
Prefill chooses 93 over 98 with scores 0.35049203 and 0.34970510. The full-history
reference also chooses 93, with a 7.88e-4 gap. This later flip amplifies already
divergent inputs; its slot output differs by 3.499. At row 78 all expert sets agree.
There, slot 17 `l_out` is the first captured tensor above 1e-4 between the native paths
(1.37e-4). Against the saved 79-token float32 row, decode is closer at that tensor
(4.58e-5 versus 9.16e-5); prefill is closer at slot 27 `mixin-1` and slot 47 `l_out`.

The table lists every slot with a different expert set at rows 718–724. Each cell
shows the eighth and ninth experts and their selection scores. Rows 718 and 721
have no expert-set differences.

| Row | Slot | Decode eighth, ninth | Prefill eighth, ninth |
| --- | --- | --- | --- |
| 719 | 41 | 98: 0.349845, 93: 0.349451 | 93: 0.350492, 98: 0.349705 |
| 720 | 19 | 51: 0.125924, 101: 0.124600 | 101: 0.124714, 51: 0.124107 |
| 722 | 19 | 19: 0.137427, 127: 0.134696 | 75: 0.135809, 19: 0.135451 |
| 722 | 20 | 110: 0.099619, 44: 0.097536 | 110: 0.100187, 57: 0.100036 |
| 723 | 44 | 95: 0.228757, 98: 0.227050 | 98: 0.231004, 95: 0.226462 |
| 723 | 45 | 72: 0.210731, 26: 0.206078 | 58: 0.286283, 60: 0.279431 |
| 723 | 46 | 21: 0.247345, 89: 0.238643 | 89: 0.135493, 51: 0.115357 |
| 723 | 47 | 75: 0.220142, 120: 0.218454 | 20: 0.265750, 33: 0.211796 |
| 723 | 21 | 126: 0.066777, 77: 0.064408 | 41: 0.073492, 39: 0.069943 |
| 723 | 23 | 9: 0.020294, 50: 0.018988 | 50: 0.018968, 68: 0.017356 |
| 724 | 43 | 46: 0.339391, 42: 0.339296 | 42: 0.339312, 46: 0.339062 |
| 724 | 44 | 75: 0.152590, 54: 0.146208 | 121: 0.143021, 89: 0.141110 |
| 724 | 45 | 29: 0.214998, 20: 0.209988 | 20: 0.221270, 29: 0.206304 |

A separate 724-token reference prefix cannot substitute for the full-history report:
its first expert-set difference from prefill is row 504, slot 36, where the prefix
reference's eighth/ninth gap is 2.68e-7. Its row-723 logits differ from the saved
full-history reference by 0.983. The prefix run used `--expert-cache-gib 4`,
`--trace-row 723`, and `--router-report`, without a row dump. It exited 0 and charged
27.5 GB at peak. An earlier prefix attempt with cache 10 and a row dump exceeded the
28 GB cap and was stopped; it supplied no result.

Reproduction uses the model and `structured-48.json` history named above, with
`campaign/heavy.py` around every runner invocation. `tracerow <out> <row>` captures
decode tensors. `prefillrow 128 <out> <row>` captures the same `[slots][9][hidden]`
layout for prefill. Both write `.picks` and `.router` sidecars. `routes <out> 723`
and `prefillroutes 128 <out> 723` capture router inputs and choices for the full
comparison. Compare expert sets in execution order, then compute the scores from the
captured logits and bias. The saved full-history router report and logits supply the
reference picks, gaps and final-output errors. Each `.picks` sidecar holds
`[slots][topk]` 32-bit integers. Each `.router` sidecar holds float32
`[slots][2][experts]`, with logits followed by bias. The decode `routes` file stores
one picks block and one router block per position. Each `prefillroutes` file is suffixed
with its chunk's first position and stores, per slot, `[rows][topk]` picks,
`[rows][experts]` logits and `[experts]` bias.

For example, with `M` set to the checkpoint, `H` to the history, `R` to the runner,
and `T` to the campaign scratch directory, the native captures use:

```sh
python3 campaign/heavy.py -- "$R" "$M" "$H" tracerow "$T/decode723.bin" 723
python3 campaign/heavy.py -- "$R" "$M" "$H" prefillrow 128 "$T/prefill723.bin" 723
python3 campaign/heavy.py -- "$R" "$M" "$H" routes "$T/decoderoutes.bin" 723
python3 campaign/heavy.py -- "$R" "$M" "$H" prefillroutes 128 "$T/prefillroutes" 723
```

For single-row comparisons, repeat the first two commands with rows 78 and 581–724.
The saved history consists of the tokenized `campaign/prompts/dev/structured-1024.txt`
prompt followed by its first 48 greedy tokens from main `ca2713cf1`.
