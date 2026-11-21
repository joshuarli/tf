# Benchmark scorecard

Generated from `campaign/runs/` by `make -C campaign scorecard`. Rates are medians; brackets are descriptive ranges across process-pair medians (or across prompts when only one pair ran). Repeatability is measured only by the six-pair, two-session protocol; otherwise it is marked "unmeasured" even when a range is shown.

- Run files: 7
- Trials recorded: 42 (29 valid, 13 invalid)
- Unavailable cell records: 2
- Malformed lines skipped: 0

## Matched

Same engine, mode, cell and prompt-cache size. Sampled request peak is the `/memory` high-water mark after readiness; lifetime peak is the kernel's whole-process high-water mark and is not gated on the 250 ms samples.

| Engine | Mode | Cell | Cache GiB | Trials | Pairs | Cold decode tok/s | Warm decode tok/s | Prefill tok/s | TTFT ms | Total s | Sampled request peak GiB | Lifetime peak GiB | Accept | Tokens/round | Repeatability |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | 7 | 7 | 160.0 [153.4–165.8] | 159.0 [151.2–165.8] | 4176.2 [3787.8–4236.1] | 246.9 [243.3–272.7] | 3.44 [3.33–3.60] | 22.54 [22.50–22.54] | 22.56 [22.53–22.60] | — | 1.00 [1.00–1.00] | measured |
| tensorfold-native@8c2c8e6f4fd9 | copy | 1024 | 2 | 3 | 1 | 156.8 [146.7–165.7] | 161.4 [141.4–162.4] | 3820.4 [3708.6–4210.5] | 270.6 [245.2–277.9] | 3.54 [3.35–3.73] | 22.57 [22.50–22.63] | 22.60 [22.55–22.65] | 0.592 [0.190–0.600] | 1.21 [1.08–1.24] | unmeasured |

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
| flint-copy-1024-092747 | flint-copy-1024-092747:1024:copy:1 | missing candidate counterpart |

## Best useful

The fastest measured configuration per cell, among engines, modes and configurations that actually ran. Heterogeneous configurations stay here rather than entering a matched result.

| Cell | Cache GiB | Best engine | Mode | Cold decode tok/s | Trials | Other measured |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 1024 | 2 | tensorfold-native@8c2c8e6f4fd9 | ar | 160.0 | 7 | tensorfold-native@8c2c8e6f4fd9 copy 156.8 |

## Natural termination

Requests sent without `ignore_eos` (they do not enter the fixed-length rates above).

| Engine | Mode | Cell | Cache GiB | Completions | Finish reasons | Mean tokens |
| --- | --- | ---: | ---: | --- | --- | ---: |
| tensorfold-native@8c2c8e6f4fd9 | ar | 1024 | 2 | truncated 39 | length 39 | 512.0 |
| tensorfold-native@8c2c8e6f4fd9 | copy | 1024 | 2 | truncated 3 | length 3 | 512.0 |

## Unavailable

| Engine | Mode | Cell | Reason |
| --- | --- | ---: | --- |
| tensorfold-native@8c2c8e6f4fd9 | combined | 1024 | drafter checkpoint unavailable at ~/models/Flint-DFlash2@39b0a7f7a12a; not fetched |
| tensorfold-native@8c2c8e6f4fd9 | drafter | 1024 | drafter checkpoint unavailable at ~/models/Flint-DFlash2@39b0a7f7a12a; not fetched |

## Invalid trials

Host pressure, throttling or a short reply marked these out of the rates.

| Engine | Mode | Cell | Pair | Reasons |
| --- | --- | ---: | ---: | --- |
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
