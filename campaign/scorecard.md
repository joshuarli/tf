# Benchmark scorecard

Generated from `campaign/runs/` by `make -C campaign scorecard`. Rates are medians over measured trials; brackets are a 95% confidence interval when more than one trial entered a cell.

- Run files: 15
- Trials recorded: 182 (164 valid, 18 invalid)
- Unavailable cell records: 5
- Malformed lines skipped: 0

## Matched

Same engine, mode, cell and prompt-cache size, so every row compares like with like.

| Engine | Mode | Cell | Cache GiB | Trials | Pairs | Cold decode tok/s | Warm decode tok/s | Prefill tok/s | TTFT ms | Total s | Peak GiB | Accept | Tokens/round |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlx-lm | greedy | 1024 | 0 | 9 | 3 | 123.6 [113.1–134.1] | 123.9 [112.9–134.9] | — | 239.9 [-447.6–927.5] | 5.41 [4.72–6.10] | — | — | — |
| mlx-lm | greedy | 4096 | 0 | 9 | 3 | 117.5 [108.2–126.8] | 119.8 [108.6–130.9] | — | 1113.8 [-1664.2–3891.8] | 5.81 [3.11–8.52] | — | — | — |
| mlx-lm | greedy | 32768 | 0 | 8 | 3 | 88.3 [74.2–102.4] | 79.4 [67.3–91.6] | — | 14739.5 [10981.2–18497.7] | 20.43 [15.32–25.54] | — | — | — |
| rec | greedy | 1024 | 0 | 9 | 3 | 222.7 [214.4–231.0] | 231.7 [220.5–242.9] | — | 482.6 [379.2–585.9] | 2.83 [2.67–3.00] | — | — | — |
| rec | greedy | 4096 | 0 | 7 | 3 | 190.6 [179.6–201.7] | 200.0 [183.2–216.8] | — | 2651.6 [-1340.4–6643.5] | 5.33 [1.26–9.40] | — | — | — |
| rec | greedy | 32768 | 0 | 9 | 3 | 128.8 [125.8–131.7] | 130.7 [127.2–134.1] | — | 18283.8 [15798.7–20768.9] | 22.37 [19.91–24.84] | — | — | — |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 0 | 18 | 6 | 162.8 [160.8–164.8] | 161.5 [159.5–163.5] | 4061.8 [3967.2–4156.4] | 253.5 [247.5–259.6] | 3.40 [3.36–3.44] | 22.41 [22.41–22.42] | — | 1.00 [1.00–1.00] |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | 3 | 1 | 153.4 [138.6–168.2] | 151.2 [140.1–162.2] | 3787.8 [2695.7–4880.0] | 272.7 [192.4–353.1] | 3.60 [3.21–4.00] | 22.54 [22.41–22.68] | — | 1.00 [1.00–1.00] |
| tensorfold-native@8c2c8e6f4fd9 | ar | 4096 | 0 | 17 | 6 | 150.7 [148.1–153.3] | 151.4 [148.2–154.6] | 4557.3 [4260.1–4854.4] | 901.6 [824.7–978.5] | 4.28 [4.15–4.41] | 22.50 [22.49–22.50] | — | 1.00 [1.00–1.00] |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 0 | 17 | 6 | 98.1 [97.0–99.2] | 98.2 [96.9–99.5] | 2820.4 [2717.5–2923.3] | 11630.2 [11231.8–12028.6] | 16.83 [16.41–17.25] | 23.26 [23.25–23.28] | — | 1.00 [1.00–1.00] |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 2 | 1 | 1 | 93.2 | 94.6 | 3325.6 | 9876.7 | 15.36 | 24.98 | — | 1.00 |
| tensorfold-native@8c2c8e6f4fd9 | copy | 1024 | 0 | 18 | 6 | 170.5 [163.5–177.5] | 169.1 [161.4–176.9] | 4254.9 [4171.0–4338.9] | 242.1 [236.2–248.0] | 3.25 [3.11–3.39] | 22.44 [22.43–22.44] | 0.592 [0.190–0.600] | 1.21 [1.08–1.24] |
| tensorfold-native@8c2c8e6f4fd9 | copy | 1024 | 2 | 3 | 1 | 156.8 [133.3–180.4] | 161.4 [132.0–190.7] | 3820.4 [3165.8–4475.0] | 270.6 [227.9–313.3] | 3.54 [3.07–4.00] | 22.57 [22.41–22.73] | 0.592 [0.190–0.600] | 1.21 [1.08–1.24] |
| tensorfold-native@8c2c8e6f4fd9 | copy | 4096 | 0 | 18 | 6 | 149.5 [144.2–154.9] | 146.1 [141.1–151.0] | 3894.4 [3611.9–4176.9] | 1054.6 [989.5–1119.8] | 4.33 [4.18–4.47] | 22.52 [22.52–22.52] | 0.343 [0.286–0.653] | 1.24 [1.06–1.25] |
| tensorfold-native@8c2c8e6f4fd9 | copy | 32768 | 0 | 18 | 6 | 93.3 [90.1–96.6] | 93.8 [90.9–96.7] | 2886.2 [2793.5–2978.9] | 11371.4 [11026.1–11716.6] | 16.74 [16.32–17.16] | 23.29 [23.28–23.30] | 0.387 [0.303–0.660] | 1.17 [1.12–1.24] |

### Paired baseline/candidate

Per-pair ratios of baseline cold decode rate to candidate; the interval is on the log ratios.

| Mode | Cell | Cache GiB | Baseline | Candidate | Pairs | Baseline/candidate | 95% CI |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| greedy | 1024 | 0 | mlx-lm | rec | 3 | 0.567 | 0.523–0.610 |
| greedy | 4096 | 0 | mlx-lm | rec | 3 | 0.620 | 0.501–0.841 |
| greedy | 32768 | 0 | mlx-lm | rec | 3 | 0.646 | 0.498–0.974 |

## Best useful

The fastest measured configuration per cell, among engines and modes that actually ran.

| Cell | Cache GiB | Best engine | Mode | Cold decode tok/s | Trials | Other measured |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 1024 | 0 | rec | greedy | 222.7 | 9 | tensorfold-native@8c2c8e6f4fd9 copy 170.5, tensorfold-native@8c2c8e6f4fd9 ar 162.8, mlx-lm greedy 123.6 |
| 1024 | 2 | tensorfold-native@8c2c8e6f4fd9 | copy | 156.8 | 3 | tensorfold-native@8c2c8e6f4fd9 ar 153.4 |
| 4096 | 0 | rec | greedy | 190.6 | 7 | tensorfold-native@8c2c8e6f4fd9 ar 150.7, tensorfold-native@8c2c8e6f4fd9 copy 149.5, mlx-lm greedy 117.5 |
| 32768 | 0 | rec | greedy | 128.8 | 9 | tensorfold-native@8c2c8e6f4fd9 ar 98.1, tensorfold-native@8c2c8e6f4fd9 copy 93.3, mlx-lm greedy 88.3 |
| 32768 | 2 | tensorfold-native@8c2c8e6f4fd9 | ar | 93.2 | 1 | — |

## Natural termination

Requests sent without `ignore_eos` (they do not enter the fixed-length rates above).

| Engine | Mode | Cell | Cache GiB | Completions | Finish reasons | Mean tokens |
| --- | --- | ---: | ---: | --- | --- | ---: |
| mlx-lm | greedy | 1024 | 0 | truncated 9 | length 9 | 512.0 |
| mlx-lm | greedy | 4096 | 0 | truncated 9 | length 9 | 512.0 |
| mlx-lm | greedy | 32768 | 0 | truncated 13 | length 13 | 512.0 |
| rec | greedy | 1024 | 0 | truncated 9 | length 9 | 512.0 |
| rec | greedy | 4096 | 0 | truncated 15 | length 15 | 512.0 |
| rec | greedy | 32768 | 0 | truncated 9 | length 9 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 0 | truncated 18 | length 18 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | truncated 3 | length 3 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 4096 | 0 | truncated 17 | length 17 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 0 | truncated 18 | length 18 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 2 | truncated 1 | length 1 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | copy | 1024 | 0 | truncated 18 | length 18 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | copy | 1024 | 2 | truncated 3 | length 3 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | copy | 4096 | 0 | truncated 18 | length 18 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | copy | 32768 | 0 | truncated 18 | length 18 | 512.0 |

## Unavailable

| Engine | Mode | Cell | Reason |
| --- | --- | ---: | --- |
| exo-2533 | greedy | None | deferred: not needed until a competitive claim; the PR #2533 head is pinned but not built this pass |
| exo-main | greedy | None | deferred: not needed until a competitive claim; Exo needs a single-node cluster and an npm dashboard build, then its API serves the same checkpoint but with no ignore_eos or prompt-cache size |
| nanoattention | greedy | None | deferred: not needed until a competitive claim; it is public MIT Zig/Metal and built here (zig-out/bin/flint session), but this pass measured no full 512-token cell |
| tensorfold-native@8c2c8e6f4fd9 | combined | 1024 | drafter checkpoint unavailable at ~/models/Flint-DFlash2@39b0a7f7a12a; not fetched |
| tensorfold-native@8c2c8e6f4fd9 | drafter | 1024 | drafter checkpoint unavailable at ~/models/Flint-DFlash2@39b0a7f7a12a; not fetched |

## Invalid trials

Host pressure, throttling or a short reply marked these out of the rates.

| Engine | Mode | Cell | Pair | Reasons |
| --- | --- | ---: | ---: | --- |
| mlx-lm | greedy | 32768 | 1 | memory pressure level 2; memory pressure level 2 |
| mlx-lm | greedy | 32768 | 1 | memory pressure level 2 |
| mlx-lm | greedy | 32768 | 1 | memory pressure level 2 |
| mlx-lm | greedy | 32768 | 2 | memory pressure level 2 |
| mlx-lm | greedy | 32768 | 2 | memory pressure level 2 |
| rec | greedy | 4096 | 1 | memory pressure level 2 |
| rec | greedy | 4096 | 1 | memory pressure level 2 |
| rec | greedy | 4096 | 1 | memory pressure level 2 |
| rec | greedy | 4096 | 1 | memory pressure level 2 |
| rec | greedy | 4096 | 2 | memory pressure level 2 |
| rec | greedy | 4096 | 2 | memory pressure level 2 |
| rec | greedy | 4096 | 2 | memory pressure level 2 |
| rec | greedy | 4096 | 2 | memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 4096 | 6 | HTTPError: HTTP Error 503: Service Unavailable |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 4 | HTTPError: HTTP Error 503: Service Unavailable |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 4 | HTTPError: HTTP Error 503: Service Unavailable |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 4 | HTTPError: HTTP Error 503: Service Unavailable |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 5 | memory pressure level 2 |
