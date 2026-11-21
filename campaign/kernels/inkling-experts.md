# Inkling 2 Mini routed-expert kernels: K1 winners in the family

The family's decode experts now run the K1 experiment's measured winners
([fp4.md](fp4.md)) instead of the correctness-first kernels. Decode is one row,
so the family dispatches exactly the batch-1 winners; the choice is explicit in
`expertKernels` (`zig/src/families/inkling2_loop/model.zig`).

## Kernel choice at decode (batch 1)

| format | gate/up | down |
| --- | --- | --- |
| NVFP4 (layers 0-19) | `i2_gu_table` (v3-table) | `i2_down_table` (v3-table) |
| MXFP8 (layers 20-23) | `i2_gu_table` (v3-table) | `i2_down_thread` (v4-thread) |

`v3-table` is one simdgroup a gate/up pair or down row with the 32 lanes split
over the scale groups and the e2m1/e4m3 codes decoded through an f32 table;
`v4-thread` is one thread a down row with no reduction, which wins only MXFP8
down at batch 1, where the 1152-input reduction leaves 36 groups a lane. The
table bytes live in two shared buffers (`t16`, `t256`) filled from the same
`core.quant` decoders the reference uses; the scale path, the per-expert
NVFP4 `amax/denominator` global, the situ caps (4.0/25.0) and the checkpoint
layout indexed by `PICKS[slot]` match the old kernels exactly. The old
expert-major `i2_gu`/`i2_down` stay in the source as the layout reference; the
per-slot grid drops the `i2_plan` dispatch (3,158 dispatches a step before,
3,110 after).

## Before and after at the 83-token prompt

Before is the profiled baseline ([inkling-profile.md](inkling-profile.md));
after is `lane/k2-inkling-experts` on the same host and prompt, `bench 14 2`
with `TF_METAL4_TIMELINE=4` (4 timed rounds after 8 warm, GPU timestamps).

| kernel | calls a step | ms a step before | ms a step after |
| --- | ---: | ---: | ---: |
| `i2_gu` / `i2_gu_table` | 48 | 41.4 | 3.0 |
| `i2_down` / `i2_down_table` + `i2_down_thread` | 48 | 22.3 | 2.0 + 0.3 |
| experts together | 96 | 63.7 | 5.3 |
| whole step (GPU) | | 104.2 | 51.1 |
| whole step (wall median) | | 105.1 | 47.2 |

The expert slowdown is gone: gate/up runs 13.9x faster, down 9.8x faster, and
the step is now bounded by the dense `i2_mv` projections (19.9 ms), `i2_attn`
(12.2 ms) and the single-thread `i2_route` (5.2 ms). Prefill still runs one
position at a time, so decode-shaped kernels cover every forward.

## Parity

`make -C campaign parity MODEL=inkling2-mini` against the saved
receipt-verified reference logits: all 14 short histories pass the
`campaign/ref/logit_gate.py` f32-vs-f32 contract, worst per-history max abs
1.583e-04 (`prose_review`) against the 1.6e-4 bound. `harbour` and
`cells_code` still fail as before: `harbour` first exceeds the bound at
position 557 with the first top-1 disagreement at 588 (10 disagreements,
max abs 6.04), byte-identical with and without this change; `cells_code`
keeps its 0.678 spike at position 690 with no top-1 disagreement in either
run, and only the noise-floor threshold crossing moves (95 before, 151
after). The long-history signature is an attention-trace mismatch
(`Qcur-0` absent from the Metal trace), outside the expert kernels.

## Reproduce

```sh
zig build tf-inkling2-loop-run -Dcpu=apple_m1
M=~/models/Inkling-2-Mini@ed40e7da625f
head -c 400 campaign/prompts/dev/code-1024.txt > /tmp/code-400.txt
./zig-out/bin/tf-inkling2-loop-run $M/tokenizer.json /tmp/code-400.txt tokenize ids-83.json
python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-loop-run $M ids-83.json bench 14 2
TF_METAL4_TIMELINE=4 python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-loop-run $M ids-83.json bench 14 2 2> timeline.log
python3 campaign/kernels/inkling_profile.py timeline --log timeline.log
zig build test-inkling2-loop -Dcpu=apple_m1
make -C campaign parity MODEL=inkling2-mini HISTORIES=parity/inkling2-mini/histories.json HEAVY=$PWD/campaign/heavy.py
```

Every model-loading or GPU command runs through `campaign/heavy.py`. The
parity overrides pin the fragment's own defaults: the top-level Makefile's
`HEAVY` (prefixed form) and `campaign/ref/ref.mk`'s `HISTORIES` otherwise
clobber what the fragment's recipes and `compare.py` expect.
