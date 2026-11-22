# Inkling 2 Mini bf16-operand prefill (M1)

Status: **reverted** by the rule declared before results in
[campaign/plan.md](../plan.md#status). Criteria 2 and 3 fail. The code is parked
on `lane/m1-bf16-prefill` at `8df2790ec` and is not on main.

The candidate selected bf16 matrix operands with `INKLING_PREFILL_BF16=1`. The
bf16 matrix kernels accumulated in f32. Router, attention, residual,
normalization and KV storage stayed f32. The default path was unchanged.

## Why bf16 was tried

A split Metal 4 profile ran one 512-row chunk at base 4,608 on a 5,265-token code
prompt, under `heavy.py` with peak charged memory 20.5 GB. The instrumented GPU
span summed to 1,358.7 ms. Expert matrix kernels took 682.4 ms, attention
273.4 ms, and the two dense GEMM kernels 142.7 ms. These spans include profiling
overhead and are not end-to-end timings. The matrix kernels are the dominant
prefill cost, and the nanoattention ablation (`inkling-nanoattention.md`) put
its lead in bf16 matrix operands.

## Criteria

| # | Criterion | Result |
| --- | --- | --- |
| 1 | Paired 1K prefill ≥ ×1.20 | not measured: brackets incomplete, busy host |
| 2 | Confident top-1 ≥ 99.5% on the 16 parity histories, no new unexplained class | **FAIL**: 2,758 of 2,862, 96.37% |
| 3 | Greedy ids match f32 for 64 tokens on three 512-token prompts, or diverge at margin < 0.1 | **FAIL**: structured diverges at index 56, margin 0.2116 |
| 4 | Quality ≥ 8/12, Pi checkers ≥ 7/8 | Pi 7/8; quality 7 of 11 scored (one case excluded) |
| 5 | Peak memory within +0.5 GB | not measured |

Criterion 2 used a streamed comparison of the 16 saved candidate outputs against
the f32 oracle. Agreement counts positions where the reference top-two margin is
at least 0.1.

Criterion 3 ran `greedy-decode 64` after 512 prompt tokens. Code used
`campaign/prompts/inkling/code-512.json`. Prose and structured used the first 512
Inkling token ids of `campaign/prompts/dev/*-1024.txt`. The margin comes from
the f32 control's logits at the first differing token.

| Prompt | First differing index | Control id | Candidate id | Control top-two margin |
| --- | ---: | ---: | ---: | ---: |
| code | 15 | 151089 | 150311 | 0.0342 |
| prose | none in 64 | | | |
| structured | 56 | 4580 | 84483 | 0.2116 |

For criterion 4:
- **Pi:** the flag-on run completed 8/8 tasks. `read-and-answer` failed on
  dead-letter ordering. The result is
  `campaign/pi/results/m1-bf16-20261122T185056Z.json`.
- **Quality:** the run scored 11 cases, with 7 passes. It is recorded in
  `campaign/quality/results/inkling2-mini-dev-20261122T185441Z/result.json`.
  The excluded case, retrieval-t05, needs 43,076 prompt tokens plus an
  8,192-token reply, which is more than the 32,768-token campaign context. That
  makes a 12-case score impossible under the cap for any configuration.

The default f32 path was checked on the lane:
- `zig build test-inkling2-loop -Dcpu=apple_m1` exited 0 under `heavy.py`.
- Prefill parity at chunk 512 stayed at 14 within contract, 2 explained and 0
  unexplained.

## Reproduce

Check out `lane/m1-bf16-prefill` and build `tf-inkling2-loop-run`. Run the
parity capture with `INKLING_PREFILL_BF16=1` and compare under
`--arithmetic-policy bf16-operand-prefill`. Then run the runner's
`greedy-decode 64` on the three prompts with and without the flag.
`TF_INKLING_GREEDY_MARGINS=1` prints the control margins.
