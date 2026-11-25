# Inkling 2 Mini grouped verify: the fixed cost (V4)

**Status.** In progress. This first section is the decision input: where main's kernels spend a batch step's and a
grouped verify's time, by kernel class, measured before the lane changed any kernel.

**Base.** Main is `d129b60f8` (P5c plus PH2's half-precision prefill default). Main's runner is `git archive 570188a7c`
(main plus this lane's measurement commits, no kernel changes) with the lane's final `kbench.zig`, built
`-Dcpu=apple_m1` (`run-base3`).

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
