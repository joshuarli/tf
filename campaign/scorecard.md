# Benchmark scorecard

Generated from `campaign/runs/` by `make -C campaign scorecard`. Rates are medians over measured trials; brackets are a 95% confidence interval when more than one trial entered a cell.

- Run files: 4
- Trials recorded: 6 (6 valid, 0 invalid)
- Unavailable cell records: 2
- Malformed lines skipped: 0

## Matched

Same engine, mode, cell and prompt-cache size, so every row compares like with like.

| Engine | Mode | Cell | Cache GiB | Trials | Pairs | Cold decode tok/s | Warm decode tok/s | Prefill tok/s | TTFT ms | Total s | Peak GiB | Accept | Tokens/round |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | 3 | 1 | 153.4 [138.6–168.2] | 151.2 [140.1–162.2] | 3787.8 [2695.7–4880.0] | 272.7 [192.4–353.1] | 3.60 [3.21–4.00] | 22.54 [22.41–22.68] | — | 1.00 [1.00–1.00] |
| tensorfold-native@8c2c8e6f4fd9 | copy | 1024 | 2 | 3 | 1 | 156.8 [133.3–180.4] | 161.4 [132.0–190.7] | 3820.4 [3165.8–4475.0] | 270.6 [227.9–313.3] | 3.54 [3.07–4.00] | 22.57 [22.41–22.73] | 0.592 [0.190–0.600] | 1.21 [1.08–1.24] |

### Paired baseline/candidate

Per-pair ratios of baseline cold decode rate to candidate; the interval is on the log ratios.

No run pairs a baseline and a candidate process for the same mode and cell.

## Best useful

The fastest measured configuration per cell, among engines and modes that actually ran.

| Cell | Cache GiB | Best engine | Mode | Cold decode tok/s | Trials | Other measured |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 1024 | 2 | tensorfold-native@8c2c8e6f4fd9 | copy | 156.8 | 3 | tensorfold-native@8c2c8e6f4fd9 ar 153.4 |

## Natural termination

Requests sent without `ignore_eos` (they do not enter the fixed-length rates above).

| Engine | Mode | Cell | Cache GiB | Completions | Finish reasons | Mean tokens |
| --- | --- | ---: | ---: | --- | --- | ---: |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | truncated 3 | length 3 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | copy | 1024 | 2 | truncated 3 | length 3 | 512.0 |

## Unavailable

| Engine | Mode | Cell | Reason |
| --- | --- | ---: | --- |
| tensorfold-native@8c2c8e6f4fd9 | combined | 1024 | drafter checkpoint unavailable at ~/models/Flint-DFlash2@39b0a7f7a12a; not fetched |
| tensorfold-native@8c2c8e6f4fd9 | drafter | 1024 | drafter checkpoint unavailable at ~/models/Flint-DFlash2@39b0a7f7a12a; not fetched |
