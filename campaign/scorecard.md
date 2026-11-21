# Benchmark scorecard

Generated from `campaign/runs/` by `make -C campaign scorecard`. Rates are medians; brackets are descriptive ranges across process-pair medians (or across prompts when only one pair ran). Repeatability is measured only by the six-pair, two-session protocol; otherwise it is marked "unmeasured" even when a range is shown.

- Run files: 18
- Trials recorded: 218 (187 valid, 31 invalid)
- Unavailable cell records: 5
- Malformed lines skipped: 0

## Matched

Same engine, mode, cell and prompt-cache size. Sampled request peak is the `/memory` high-water mark after readiness; lifetime peak is the kernel's whole-process high-water mark and is not gated on the 250 ms samples.

| Engine | Mode | Cell | Cache GiB | Trials | Pairs | Cold decode tok/s | Warm decode tok/s | Prefill tok/s | TTFT ms | Total s | Sampled request peak GiB | Lifetime peak GiB | Accept | Tokens/round | Repeatability |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mlx-lm | greedy | 1024 | 0 | 3 | 3 | 123.6 [121.7–125.4] | 123.3 [122.1–124.8] | — | 239.9 [233.7–244.4] | 5.41 [4.41–5.64] | — | — | — | — | unmeasured |
| mlx-lm | greedy | 4096 | 0 | 3 | 3 | 117.5 [109.4–121.4] | 119.8 [91.2–120.5] | — | 1113.8 [1071.4–1462.2] | 5.81 [5.46–7.10] | — | — | — | — | unmeasured |
| mlx-lm | greedy | 32768 | 0 | 5 | 5 | 86.2 [53.3–92.2] | 80.1 [56.2–90.7] | — | 17306.1 [12276.0–23215.3] | 23.25 [18.28–32.68] | — | — | — | — | unmeasured |
| rec | greedy | 1024 | 0 | 3 | 3 | 215.3 [211.0–227.8] | 230.6 [203.4–236.9] | — | 499.4 [352.7–538.6] | 2.86 [2.60–2.95] | — | — | — | — | unmeasured |
| rec | greedy | 4096 | 0 | 3 | 3 | 182.1 [181.0–198.9] | 194.1 [180.4–215.8] | — | 3602.9 [1379.6–7797.0] | 6.42 [3.85–10.62] | — | — | — | — | unmeasured |
| rec | greedy | 32768 | 0 | 3 | 3 | 129.0 [126.7–131.7] | 130.7 [126.5–132.5] | — | 19112.4 [17046.1–20830.2] | 23.07 [21.01–24.66] | — | — | — | — | unmeasured |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 0 | 6 | 6 | 162.6 [158.7–167.4] | 162.4 [158.2–166.6] | 4089.9 [3732.3–4252.5] | 251.8 [242.4–276.3] | 3.40 [3.30–3.49] | 22.41 [22.41–22.41] | 22.50 [22.50–22.52] | — | 1.00 [1.00–1.00] | measured |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | 7 | 7 | 160.0 [153.4–165.8] | 159.0 [151.2–165.8] | 4176.2 [3787.8–4236.1] | 246.9 [243.3–272.7] | 3.44 [3.33–3.60] | 22.54 [22.50–22.54] | 22.56 [22.53–22.60] | — | 1.00 [1.00–1.00] | measured |
| tensorfold-native@8c2c8e6f4fd9 | ar | 4096 | 0 | 6 | 6 | 148.9 [144.6–157.4] | 148.6 [141.1–157.4] | 4276.2 [3332.9–4683.8] | 978.3 [876.7–1231.4] | 4.41 [4.16–4.76] | 22.50 [22.50–22.51] | 22.60 [22.59–22.61] | — | 1.00 [1.00–1.00] | measured |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 0 | 6 | 6 | 98.0 [95.3–98.7] | 97.1 [95.6–101.4] | 2787.6 [2646.0–3107.4] | 11768.9 [10559.6–12402.8] | 16.98 [15.76–17.77] | 23.26 [23.26–23.26] | 23.55 [23.52–23.67] | — | 1.00 [1.00–1.00] | measured |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 2 | 1 | 1 | 93.2 | 94.6 | 3325.6 | 9876.7 | 15.36 | 24.98 | 25.01 | — | 1.00 | unmeasured |
| tensorfold-native@8c2c8e6f4fd9 | copy | 1024 | 0 | 6 | 6 | 174.6 [143.8–179.9] | 172.6 [147.5–179.9] | 4257.1 [3936.9–4329.6] | 242.1 [237.8–262.8] | 3.17 [3.08–3.81] | 22.44 [22.43–22.44] | 22.51 [22.47–22.52] | 0.592 [0.592–0.592] | 1.21 [1.21–1.21] | measured |
| tensorfold-native@8c2c8e6f4fd9 | copy | 1024 | 2 | 3 | 1 | 156.8 [146.7–165.7] | 161.4 [141.4–162.4] | 3820.4 [3708.6–4210.5] | 270.6 [245.2–277.9] | 3.54 [3.35–3.73] | 22.57 [22.50–22.63] | 22.60 [22.55–22.65] | 0.592 [0.190–0.600] | 1.21 [1.08–1.24] | unmeasured |
| tensorfold-native@8c2c8e6f4fd9 | copy | 4096 | 0 | 6 | 6 | 148.9 [145.0–152.4] | 146.1 [137.8–152.3] | 3834.0 [3532.8–4744.2] | 1071.3 [866.6–1162.6] | 4.34 [4.18–4.69] | 22.52 [22.52–22.52] | 22.60 [22.57–22.62] | 0.343 [0.343–0.343] | 1.24 [1.24–1.24] | measured |
| tensorfold-native@8c2c8e6f4fd9 | copy | 32768 | 0 | 6 | 6 | 93.9 [91.9–94.7] | 93.8 [92.3–95.1] | 2878.3 [2799.9–3145.7] | 11405.8 [10429.2–11769.9] | 16.84 [15.68–17.26] | 23.29 [23.29–23.29] | 23.48 [23.41–23.56] | 0.387 [0.387–0.387] | 1.17 [1.17–1.17] | measured |

### Paired baseline/candidate

Speedup is candidate cold decode throughput over baseline; a value above 1.0 is faster. A unit is one process pair whose baseline and candidate records share prompt hashes, model and quantization, host, cell and output policy, prompt-cache policy and acceleration mode. Pairs combine only inside one declared experiment, and the interval is a t interval on per-pair log ratios; with fewer than six pairs across two sessions, it is descriptive and repeatability is unmeasured. Weights are predeclared: each prompt within a cell carries equal weight, and each independent process pair carries equal weight in the interval.

| Experiment | Mode | Cell | Cache GiB | Baseline | Candidate | Pairs | Sessions | Speedup (cand/base) | 95% CI | Repeatability |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| flint-ar-1024-151350 | ar | 1024 | 2 | tensorfold-native@8c2c8e6f4fd9 | tensorfold-native@8c2c8e6f4fd9 | 2 | 1 | 1.001 | 0.975–1.028 | unmeasured |

#### Rejected pairs

These process pairs were not combined into a matched result.

| Experiment | Process pair | Reason |
| --- | --- | --- |
| flint-ar-1024-092827 | flint-ar-1024-092827:1024:ar:1 | missing candidate counterpart |
| flint-ar-1024-150624 | flint-ar-1024-150624:ar:1024:1 | missing candidate counterpart |
| flint-ar-1024-150624 | flint-ar-1024-150624:ar:1024:2 | prompt counterparts differ: [('structured-1024', '12d86f119aa7aaa6c18892daaad2c39a7e187330f8c6ef24eb9066d4e945e5a6')] |
| flint-ar-1024-150956 | flint-ar-1024-150956:ar:1024:1 | missing candidate counterpart |
| flint-ar-1024-150956 | flint-ar-1024-150956:ar:1024:2 | prompt counterparts differ: [('code-1024', 'c5a63fccf2c9866ac60e5b13ec268e2a56546fbed3b1b4d2cbfa5b808b0de4aa'), ('prose-1024', '1d63bf9f3da787c95031b66edf6cbd95dc130389e7d47a83207c138b91fbfe56'), ('structured-1024', '12d86f119aa7aaa6c18892daaad2c39a7e187330f8c6ef24eb9066d4e945e5a6')] |
| flint-ar-1024-4096-32768-124223 | flint-ar-1024-4096-32768-124223:1024:ar:1 | missing candidate counterpart |
| flint-ar-1024-4096-32768-124223 | flint-ar-1024-4096-32768-124223:1024:ar:2 | missing candidate counterpart |
| flint-ar-1024-4096-32768-124223 | flint-ar-1024-4096-32768-124223:1024:ar:3 | missing candidate counterpart |
| flint-ar-1024-4096-32768-124223 | flint-ar-1024-4096-32768-124223:32768:ar:1 | missing candidate counterpart |
| flint-ar-1024-4096-32768-124223 | flint-ar-1024-4096-32768-124223:32768:ar:2 | missing candidate counterpart |
| flint-ar-1024-4096-32768-124223 | flint-ar-1024-4096-32768-124223:32768:ar:3 | missing candidate counterpart |
| flint-ar-1024-4096-32768-124223 | flint-ar-1024-4096-32768-124223:4096:ar:1 | missing candidate counterpart |
| flint-ar-1024-4096-32768-124223 | flint-ar-1024-4096-32768-124223:4096:ar:2 | missing candidate counterpart |
| flint-ar-1024-4096-32768-124223 | flint-ar-1024-4096-32768-124223:4096:ar:3 | missing candidate counterpart |
| flint-ar-1024-4096-32768-sB-134546 | flint-ar-1024-4096-32768-sB-134546:1024:ar:4 | missing candidate counterpart |
| flint-ar-1024-4096-32768-sB-134546 | flint-ar-1024-4096-32768-sB-134546:1024:ar:5 | missing candidate counterpart |
| flint-ar-1024-4096-32768-sB-134546 | flint-ar-1024-4096-32768-sB-134546:1024:ar:6 | missing candidate counterpart |
| flint-ar-1024-4096-32768-sB-134546 | flint-ar-1024-4096-32768-sB-134546:32768:ar:5 | missing candidate counterpart |
| flint-ar-1024-4096-32768-sB-134546 | flint-ar-1024-4096-32768-sB-134546:32768:ar:6 | missing candidate counterpart |
| flint-ar-1024-4096-32768-sB-134546 | flint-ar-1024-4096-32768-sB-134546:4096:ar:4 | missing candidate counterpart |
| flint-ar-1024-4096-32768-sB-134546 | flint-ar-1024-4096-32768-sB-134546:4096:ar:5 | missing candidate counterpart |
| flint-ar-1024-4096-32768-sB-134546 | flint-ar-1024-4096-32768-sB-134546:4096:ar:6 | missing candidate counterpart |
| flint-ar-32768-sB-140905 | flint-ar-32768-sB-140905:32768:ar:5 | missing candidate counterpart |
| flint-ar-32768-sB-145954 | flint-ar-32768-sB-145954:32768:ar:4 | missing candidate counterpart |
| flint-copy-1024-092747 | flint-copy-1024-092747:1024:copy:1 | missing candidate counterpart |
| flint-copy-1024-4096-32768-125417 | flint-copy-1024-4096-32768-125417:1024:copy:1 | missing candidate counterpart |
| flint-copy-1024-4096-32768-125417 | flint-copy-1024-4096-32768-125417:1024:copy:2 | missing candidate counterpart |
| flint-copy-1024-4096-32768-125417 | flint-copy-1024-4096-32768-125417:1024:copy:3 | missing candidate counterpart |
| flint-copy-1024-4096-32768-125417 | flint-copy-1024-4096-32768-125417:32768:copy:1 | missing candidate counterpart |
| flint-copy-1024-4096-32768-125417 | flint-copy-1024-4096-32768-125417:32768:copy:2 | missing candidate counterpart |
| flint-copy-1024-4096-32768-125417 | flint-copy-1024-4096-32768-125417:32768:copy:3 | missing candidate counterpart |
| flint-copy-1024-4096-32768-125417 | flint-copy-1024-4096-32768-125417:4096:copy:1 | missing candidate counterpart |
| flint-copy-1024-4096-32768-125417 | flint-copy-1024-4096-32768-125417:4096:copy:2 | missing candidate counterpart |
| flint-copy-1024-4096-32768-125417 | flint-copy-1024-4096-32768-125417:4096:copy:3 | missing candidate counterpart |
| flint-copy-1024-4096-32768-sB-135530 | flint-copy-1024-4096-32768-sB-135530:1024:copy:4 | missing candidate counterpart |
| flint-copy-1024-4096-32768-sB-135530 | flint-copy-1024-4096-32768-sB-135530:1024:copy:5 | missing candidate counterpart |
| flint-copy-1024-4096-32768-sB-135530 | flint-copy-1024-4096-32768-sB-135530:1024:copy:6 | missing candidate counterpart |
| flint-copy-1024-4096-32768-sB-135530 | flint-copy-1024-4096-32768-sB-135530:32768:copy:4 | missing candidate counterpart |
| flint-copy-1024-4096-32768-sB-135530 | flint-copy-1024-4096-32768-sB-135530:32768:copy:5 | missing candidate counterpart |
| flint-copy-1024-4096-32768-sB-135530 | flint-copy-1024-4096-32768-sB-135530:32768:copy:6 | missing candidate counterpart |
| flint-copy-1024-4096-32768-sB-135530 | flint-copy-1024-4096-32768-sB-135530:4096:copy:4 | missing candidate counterpart |
| flint-copy-1024-4096-32768-sB-135530 | flint-copy-1024-4096-32768-sB-135530:4096:copy:5 | missing candidate counterpart |
| flint-copy-1024-4096-32768-sB-135530 | flint-copy-1024-4096-32768-sB-135530:4096:copy:6 | missing candidate counterpart |

## Best useful

The fastest measured configuration per cell, among engines, modes and configurations that actually ran. Heterogeneous configurations stay here rather than entering a matched result.

| Cell | Cache GiB | Best engine | Mode | Cold decode tok/s | Trials | Other measured |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 1024 | 0 | rec | greedy | 215.3 | 3 | tensorfold-native@8c2c8e6f4fd9 copy 174.6, tensorfold-native@8c2c8e6f4fd9 ar 162.6, mlx-lm greedy 123.6 |
| 1024 | 2 | tensorfold-native@8c2c8e6f4fd9 | ar | 160.0 | 7 | tensorfold-native@8c2c8e6f4fd9 copy 156.8 |
| 4096 | 0 | rec | greedy | 182.1 | 3 | tensorfold-native@8c2c8e6f4fd9 copy 148.9, tensorfold-native@8c2c8e6f4fd9 ar 148.9, mlx-lm greedy 117.5 |
| 32768 | 0 | rec | greedy | 129.0 | 3 | tensorfold-native@8c2c8e6f4fd9 ar 98.0, tensorfold-native@8c2c8e6f4fd9 copy 93.9, mlx-lm greedy 86.2 |
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
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | truncated 39 | length 39 | 512.0 |
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
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 1 | memory pressure level 2; memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 1 | memory pressure level 2; memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 1 | memory pressure level 2; memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 1 | memory pressure level 2; memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 1 | memory pressure level 2; memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 1 | memory pressure level 2; memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 1 | memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 1 | memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 1 | memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | memory pressure level 2; memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | memory pressure level 2 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | memory pressure level 4 |
| tensorfold-native@8c2c8e6f4fd9 | ar | 4096 | 6 | HTTPError: HTTP Error 503: Service Unavailable |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 4 | HTTPError: HTTP Error 503: Service Unavailable |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 4 | HTTPError: HTTP Error 503: Service Unavailable |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 4 | HTTPError: HTTP Error 503: Service Unavailable |
| tensorfold-native@8c2c8e6f4fd9 | ar | 32768 | 5 | memory pressure level 2 |
