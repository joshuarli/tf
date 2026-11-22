# Inkling 2 Mini: TensorFold and nanoattention

The table reports medians and ranges of three interleaved pairs per cell. Peak
charged memory is the largest observed by `heavy.py` in each cell. Token indices
are zero-based. Each engine produced the same sequence in all three runs.

| Prompt | Configuration | Engine | Context capacity | Prefill tok/s | Decode tok/s | Peak charged GB | First differing index |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 512 | reference | TensorFold | 577 | 360.30 (357.80–402.40) | 41.90 (41.77–42.02) | 21.1 | 15 |
| 512 | reference | nano | 576 | 34.49 (32.81–35.45) | 5.47 (5.47–5.48) | 22.2 | 15 |
| 512 | optimized | TensorFold | 577 | 392.20 (380.60–392.30) | 41.75 (41.60–41.84) | 20.9 | 9 |
| 512 | optimized | nano | 576 | 650.30 (649.27–650.54) | 44.21 (44.04–44.28) | 24.6 | 9 |
| 1024 | reference | TensorFold | 1089 | 407.00 (402.10–414.40) | 41.07 (40.89–41.09) | 21.2 | 45 |
| 1024 | reference | nano | 1088 | 34.27 (33.74–35.00) | 4.16 (4.16–4.16) | 22.2 | 45 |
| 1024 | optimized | TensorFold | 1089 | 414.40 (252.60–417.20) | 41.14 (41.11–41.15) | 21.0 | none in 65 |
| 1024 | optimized | nano | 1088 | 629.24 (628.81–629.46) | 42.23 (42.20–42.29) | 24.3 | none in 65 |

The reference path was slower in these runs. The optimized path had higher
prefill and decode rates in these runs, but its arithmetic and output can differ
from TensorFold. The 512-token outputs differ for both nanoattention settings.
The optimized 1,024-token outputs match for these 65 tokens. These results do not
establish parity on other prompts. Timer boundaries and warmup differ, and no
limiter was profiled, so a general engine speed verdict is inconclusive.

The workload uses the same checkpoint revision `ed40e7da625f` and the public
512-token and 1,024-token code prompts. The host is an Apple M5 Max with 48 GiB
of unified memory, running macOS 27.2. The comparison is labeled busy-host.

TensorFold is built from revision `a1862ad552db19779c280729f1d3c7f127001115`
with Zig 0.17.1 and `-Dcpu=apple_m1`. Its runner binary SHA-256 is
`3526b596feccfa3b5493994bcc68665bd26941bee2bdde5f52ca080449b789fa`.
Nanoattention is built from the source pin in the campaign lock, revision
`6f782a8dace846e4e64a5a12cbdf5c0252166199`, with Zig 0.17.1, Lean 4.35.0
and `-Doptimize=ReleaseFast`. Its binary SHA-256 is
`8340fc9c1fa944ad2d50ebcbd593597ae80c15a056c44801f5d652c9022d5152`.

TensorFold's `bench` mode uses a chunk of 512, zero warmup steps, and 64
sequential decode steps. Its prefill timer surrounds the chunked forwards and
ends before copying logits and selecting the first token. Its per-step wall
clock excludes the host argmax, and the reported decode rate is the reciprocal
of its median step latency. A separate `greedy-decode` process captures 65
output IDs, including the prefill prediction.

Nanoattention's `bench` mode uses the same prompt IDs, 64 decode forwards,
one untimed warmup, and one timed repeat per process. Its prefill timer ends
after the first argmax, and its decode timer includes each argmax. These timer
boundaries differ from TensorFold's, so the rates are not identical latency
metrics. Model loading is outside both engines' timers. Nanoattention uses its
reference defaults with batch 64, or the documented optimized knobs with batch
128. The optimized knobs include BF16 tiles and can change arithmetic.

The TensorFold cache capacity is prompt length plus 65. Nanoattention's context
is prompt length plus 64, covering the prefill and all 64 decode forwards. The
final predicted token is not forwarded. Global KV allocation capacity does not
change the number of tokens attended to at these prompt lengths. The campaign's
32,768-token serving cap is not a requirement for this runner comparison.

Each measured command runs under `heavy.py` and its 28 GB cap. Admission checks
use GPU and utilization limits of 35 percent, with the memory limits unchanged.
The procedure checks memory pressure before each attempt, alternates engine
order across pairs, and discards failed admission checks. The build and run
scripts are in `campaign/comparators/nanoattention/`.

[Run records](../comparators/nanoattention/runs/pairs.jsonl) contain all 24
accepted runs and 154 rejected attempts. Each record includes its command, exit
status, gate output, and memory receipt. Gate diagnostics that included unrelated
process command lines were redacted. The [summary script](../comparators/nanoattention/summarize.py)
checks sample counts and output stability before producing the table.

## Last-row logits

The existing TensorFold runner wrote all prefill logits with chunk 512. The
comparison reads only the last row. Nanoattention `eval` wrote that row with
feed 64 for reference defaults and feed 128 for the optimized knobs. Every row
contains 200,058 finite float32 values. The maximum absolute differences from
TensorFold were:

| Prompt IDs | Nanoattention reference | Nanoattention optimized |
| ---: | ---: | ---: |
| 512 | 0.7571 | 0.5919 |
| 1,024 | 1.8921 | 1.0913 |

These compare engine outputs; they do not establish an independent float32
oracle for these two prompts. [Logits run records](../comparators/nanoattention/runs/logits.jsonl)
contain the commands and gate results. The [difference values](../comparators/nanoattention/runs/logits-summary.json)
are retained separately. The raw float files are temporary and can be rebuilt
with `python3 campaign/comparators/nanoattention/logits.py`.
