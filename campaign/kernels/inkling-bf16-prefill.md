# Inkling 2 Mini bf16-operand prefill

Status: measurements in progress. The five criteria are declared in
[campaign/plan.md](../plan.md#status).

The candidate selects bf16 matrix operands with `INKLING_PREFILL_BF16=1`.
The default selects the original f32 operand pipelines. The bf16 matrix kernels
accumulate in f32. Router, attention, residual, normalization and KV storage
remain f32.

## Split profile

A split Metal 4 profile used chunk 512, 5,265 code-prompt tokens and the
sampled chunk at base 4,608. It ran on Apple M5 Max under `heavy.py` with the
28 GB cap. The summed instrumented GPU span was 1,358.650 ms. Expert matrix
kernels accounted for 682.371 ms, attention for 273.385 ms, and the two dense
GEMM kernels for 142.730 ms. These spans include profiling overhead and are
not end-to-end timings. The profile exited 0 and peak charged memory was 20.5 GB.

## Greedy IDs

The control and candidate used `greedy-decode 64` after 512-token prompt
prefill. Code uses `campaign/prompts/inkling/code-512.json`. Prose and structured
use the first 512 IDs obtained by tokenizing their `campaign/prompts/dev/*-1024.txt`
files with the Inkling tokenizer. Each run used `heavy.py` and one model process.
Indices below are zero-based. The margin is from the control's actual logits at
the first differing token. The diagnostic rerun reproduced the control IDs.

| Prompt | First differing index | Control ID | Candidate ID | Control top-two margin |
| --- | ---: | ---: | ---: | ---: |
| code | 15 | 151089 | 150311 | 0.03423119 |
| prose | none in 64 | | | |
| structured | 56 | 4580 | 84483 | 0.21164513 |

Criterion 3: **FAIL**. The structured divergence exceeds the 0.1 margin limit.

## Confident-position agreement

A streamed comparison of all 16 saved candidate outputs against the f32 oracle
found 2,758 agreements among 2,862 positions with a reference top-two margin
of at least 0.1: 96.366%. Criterion 2 is **FAIL** because this is below 99.5%.
The original comparator's per-history classifications are pending.

## Quality and Pi

The flag-on Pi run completed all eight tasks and passed seven checkers. The
`read-and-answer` checker failed on a dead-letter ordering mismatch. The run
returned 0, with no tool parse errors. The result is
`campaign/pi/results/m1-bf16-20261122T185056Z.json` in the primary checkout.
The quality command exited 2 after scoring 11 of 12 cases, with seven passes.
The twelfth case requires 43,076 prompt tokens and an 8,192-token reply budget,
above the campaign's 32,768-token limit. Three structured requests also failed
with HTTP 400 capability errors. Its result is
`campaign/quality/results/inkling2-mini-dev-20261122T185441Z/result.json`
in the primary checkout. Criterion 4 is unverified because the required
12-case result is incomplete. The existing campaign harness enforces the
context limit outside this lane's writable paths.

## Verification in progress

The family test command exited 0 under `heavy.py`. The full oracle verified
all 16 histories. The unmodified parity comparator exceeded the 28 GB cap when
the full target retained large arrays while generating traces. Three batches,
using the verified oracle and separately generated traces, produced 14 within
contract, two explained divergences and zero unexplained. The batches retained
the original comparator and per-history policy. The candidate parity remains
pending.

## Acceptance status

1. **NOT VERIFIED.** Five valid 1,024-token pairs were not completed. One
   initial bracket was discarded because its first control had substantially
   more page-ins than its neighbors. An earlier five-round comparison used a
   Flint-tokenized fixture and is not evidence for this Inkling workload.
2. **FAIL.** The candidate agrees at 2,758 of 2,862 confident positions,
   or 96.366%. The unchanged comparator classified 14 short histories as
   unexplained and applied two existing exception labels to the long histories.
   Those labels do not explain the new position-zero bf16 errors: its trace
   runs the per-token f32 path and cannot identify the prefill difference.
3. **FAIL.** The structured prompt's first differing token has reference
   margin 0.21164513. The three prompt results are above.
4. **NOT VERIFIED.** Pi passed 7 of 8 checkers. Quality passed 7 of 11 scored
   cases, with one excluded by the context limit. This is not a complete 12-case
   score and cannot establish the declared 8-of-12 threshold.
5. **NOT VERIFIED.** The incomplete timing brackets cannot establish the
   required paired peak-memory comparison.

The full 16-history f32 oracle passed. Three unchanged-comparator batches at
chunk 512 produced 14 within-contract histories, two explained divergences and
zero unexplained. The two long batches returned 2 because strict f32 parity
reports the explained divergences as failures. The short batch returned 0.
`zig build test-inkling2-loop -Dcpu=apple_m1` returned 0 under `heavy.py` with
`TF_INKLING2_LOOP_DIR` set to the checkpoint.

Decision: **REVERT** by the declared rule because criteria 2 and 3 fail. The
experimental code remains on this branch as requested. Acceptance is incomplete:
scoring the excluded quality case requires changing the fixed context limit in
the campaign harness or fixture, both outside this lane's writable paths.

ACCEPTANCE: not met
