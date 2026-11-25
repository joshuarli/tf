# Q4: completed-task qualification (preliminary)

Q4 repairs Inkling 2 Mini oQ4e serving qualification before any further performance claim (roadmap 5h). This receipt
covers the completed-task step only. The fixed-window served sessions, their `off`-prefill pair and the regenerated
scorecard (5h.6) have not been run.

## What ran

`campaign/kernels/inkling-q4-qualify-runs/run.sh tasks`, binary `b925b31e…15f6` (built from the Q4 lane), checkpoint
`Inkling-2-Mini-oQ4e-mtp@3eea042751cd`, DFlash2 drafter `@4f9088ea66c2`. The full frozen set (36 scored tasks, cases hash
`65ae002d…7e46`) ran with EOS on and 8,192 tokens of room: plain decode (`ar`) and the speculative default (`spec`),
once with half prefill (`tasks-half.jsonl`) and once with `INKLING_PREFILL_HALF=off` (`tasks-off.jsonl`). Each
completed answer got one follow-up turn. Every record carries its configuration identity.

Reproduce: `Q4_BIN=<binary> campaign/kernels/inkling-q4-qualify-runs/run.sh tasks`, then
`python3 -I tasks.py table tasks-half` (or `tasks-off`) and `python3 -I tasks.py equal tasks-half tasks-off`.

## Results

Speculation changes no output: every task is equal between `ar` and `spec` under both prefill modes (`tasks.py
equal`, comparing token and output hashes, token counts, finish reasons and tool output).

| Prefill | Copy class | Scored | Passed (ar = spec) |
| --- | --- | ---: | ---: |
| half | low-copy | 28 | 26 |
| half | copy-heavy | 4 | 3 |
| half | tool-continuation | 4 | 3 |
| off | low-copy | 28 | 25 |
| off | copy-heavy | 4 | 3 |
| off | tool-continuation | 4 | 3 |

Under half prefill the failures are `code-c04`, `code-c13`, `retrieval-t01` and `structured-s12`, the same in both
modes. `off` loses one more low-copy task than half. That is a prefill-precision difference, not a speculation one.
`structured-s01`..`s08` are reported as unsupported (tool continuations without a recorded tool result) and are not
scored.

## What the timings do not show

The timings are not evidence. Parallel agent lanes held the host at memory-pressure level 2 for much of the run, and
35 trials started on an unsettled host (compressor and swap gates over their limits). Medians in `tasks.py table`
are recorded but must not be quoted as speedups. Repeat the step on a quiet host before making any completion-time
or first-answer-latency claim.
