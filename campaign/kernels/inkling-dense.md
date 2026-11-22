# Inkling 2 Mini dense bf16 matvecs

The dense projections' decode cost, before and after replacing `i2_mv` / `i2_mv_cap`
(`zig/src/families/inkling2_loop/kernels/glue.metal`) with fused kernels in
`zig/src/families/inkling2_loop/kernels/dense.metal`, dispatched by `lin` / `headModel`
in `forward.zig`. `glue.metal` is untouched (another lane owns it).

## Step time (83-token prompt, greedy decode)

|  | median ms a step | GPU ms | dispatches a step |
| --- | --- | --- | --- |
| before (brief) | ~45 | — | 3,158 |
| after | 36.9 | 39.9 | 2,758 |

Runner: `tf-inkling2-loop-run <model> <tokens> bench 14 2`, prompt is the first
400 bytes of `campaign/prompts/dev/code-1024.txt` (83 tokens).

## Per-kernel GPU time (`TF_METAL4_TIMELINE=4`, 4 timed rounds)

Before at 83 tokens (from `campaign/kernels/inkling-profile.md`):

| kernel | calls a step | ms a step |
| --- | --- | --- |
| `i2_mv` (dense bf16 matvec) | 688 | 20.3 |
| `i2_mv_cap` (unembedding) | 1 | ~2.1 |
| total dense | 689 | ~22 |

After (measured `/tmp/timeline83_fix2.log`):

| kernel | calls a step | ms a step | us a call |
| --- | --- | --- | --- |
| `i2_dmv_fusedx` (q/r/sink/gate/k/kdyn/v/vdyn per slot) | 48 | 4.17 | 87.0 |
| `i2_dmv` (o-proj, router, controller down) | 192 | 3.86 | 20.1 |
| `i2_dmv_cap` (unembedding) | 1 | 1.68 | 1,678 |
| `i2_dmv_fused` (controller gain+forget) | 96 | 0.83 | 8.6 |
| total dense | 337 | 10.54 | — |

## Shapes per step (distinct `(m, k)`)

| projection | m | k | sites a step |
| --- | --- | --- | --- |
| q, gate | 3,072 | 2,304 | 48 each |
| r | 384 | 2,304 | 48 |
| sink | 24 | 2,304 | 48 |
| k, v | 1,024 | 2,304 | 48 each |
| k_dyn, v_dyn (full layers only) | 32 | 2,304 | 4 each |
| o-proj | 2,304 | 3,072 | 48 |
| router | 128 | 2,304 | 48 |
| controller down | 64 | 4,608 | 96 |
| controller gain, forget | 2,304 | 64 | 96 each |
| unembedding | 200,058 | 2,304 | 1 |

## What was measured, not assumed

- One simdgroup per output row (8 rows per 256-thread group, 32 rows per
  1024-thread group for the unembedding), bf16x8 vectorized loads, activation
  held in registers: big shapes run at 460–550 GB/s of weight bytes against the
  521 GB/s read ceiling (`i2_dmv_fusedx` ~40 MB a call in 87 us;
  `i2_dmv_cap` 926 MB in 1.68 ms).
- Staging the activation in threadgroup memory wins only for tiny `k`
  (controller gain/forget at `k = 64`: staged `i2_dmv_fused` 8.6 us a call
  against ~20 us direct); everywhere else direct reads are faster.
- Fusing projections that share one input into one dispatch cut 689 dense
  dispatches to 337: q/r/sink/gate/k/kdyn/v/vdyn per slot share three inputs
  through `i2_dmv_fusedx`; controller gain+forget share one input through
  `i2_dmv_fused`.
- Small matvecs are launch-bound, not bandwidth-bound: controller down plus
  router move 84 MB a step yet cost ~1 ms (~84 GB/s effective). Fusing the full
  controller chain (down+gain+forget) into one single-threadgroup dispatch was
  built and measured 4x slower than the staged pair (71 us a call: one
  threadgroup runs on one CU), so it was dropped; the staged path stays.
- Bit-reproducibility: `reset` reproduces 83 rows bit for bit; the new kernels
  match `i2_mv`/`i2_mv_cap` bit for bit on reruns and sit inside the fp32
  accumulation bound on every real projection shape
  (`zig build test-inkling2-loop`).

## Commands

```sh
zig build tf-inkling2-loop-run -Dcpu=apple_m1
head -c 400 campaign/prompts/dev/code-1024.txt > /tmp/prompt83.txt
./zig-out/bin/tf-inkling2-loop-run $M/tokenizer.json /tmp/prompt83.txt tokenize /tmp/ids83.json
TF_METAL4_TIMELINE=4 python3 campaign/heavy.py -- ./zig-out/bin/tf-inkling2-loop-run $M /tmp/ids83.json bench 14 2
python3 campaign/kernels/inkling_profile.py timeline --log <stderr log>
zig build test-inkling2-loop -Dcpu=apple_m1
make -C campaign parity MODEL=inkling2-mini HISTORIES=parity/inkling2-mini/histories.json HEAVY=$PWD/campaign/heavy.py
```

Every model-loading or GPU command above ran through `campaign/heavy.py`.

## Residual risks

- Dense matvecs sit at ~10.5 ms a step, not the 8 ms target. The remainder is
  launch overhead on small shapes (down/router/gain/forget) plus in-situ cold
  reads on o-proj; program order fixes the dispatch count at 337. Cutting
  further needs cross-slot batched GEMV with shared scratch, which lives in
  `state.zig` outside this lane's write boundary.
- The Metal 4 timeline overcounts wall time (stages include stamp bubbles);
  the 10.5 ms figure is conservative.
