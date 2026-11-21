# Inkling 2 Mini long-history divergence: explained verdict

Diagnosis only. No forward or kernel changes were made.

## Verdict

Both long-history divergences are **accumulation order**, not a defect. No kernel fix is
needed, and no forward code was changed.

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
  303 control), whose state echo damps back to 8.8e-05 at 305. The 656-694 regime (max 0.68,
  top-1 still 100%) is further drift growth and was not separately diagnosed.

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
  ties.
- The cells_code 656-694 growth (max 0.68, top-1 still 100%) was not separately diagnosed; the
  working model is continued drift amplification of the same kind as position 304.
- During this run the shared `.work/parity/inkling2-mini/metal/*.bin` files were deleted and
  regenerated by concurrent harness activity; all numbers above come from files this run wrote
  itself or verified by hash (saved harbour oracle md5 6d39491b57d480d00ef4560f47e3c475,
  unchanged; full rerun matches it at 0.0).
