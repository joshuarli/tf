# Q4: Inkling 2 Mini serving qualification and regenerated scorecard

Q4 repairs Inkling 2 Mini oQ4e serving qualification and regenerates the integrated scorecard (roadmap 5h, superseding
`inkling-sc2-scorecard.md`). All measurements run on one pinned `main` binary with full configuration identity, repaired
stream-evidence equality, completed tasks with EOS on alongside fixed-window 512-token generation throughput, a
genuinely loaded drafter with per-request `draft: false` as the disabled control, gated and session-paired unseen
prefill curves, and half (`INKLING_PREFILL_HALF` unset) reported separately from exact (`INKLING_PREFILL_HALF=off`).

## Summary

- **Equality (5h.1):** Speculation changes no output. Under the repaired stream-evidence check (`token_sha256`,
  `output_sha256`, `completion_tokens`, `finish_reasons`, `tool_calls`/`tool_sha256`), `ar`, `disabled` and `spec`
  match on every reply across completed tasks (`tasks-half`, `tasks-off`, including follow-up turns) and all served
  fixed-window sessions (`served-a`, `served-b`, `served-c`, `served-off`: cold, warm cache hits, natural requests, and
  forked turns).
- **Completed tasks (5h.2–5h.3):** On a quiet host under default half prefill (`tasks-half`, 0/72 timing-invalid, 36/36
  valid pairs), both `ar` and `spec` pass 32/36 scored tasks (2 cap hits: `code-c13`, `retrieval-t01`; 8 unsupported
  `structured-s01..s08`; 4 over-context excluded `retrieval-t05..t08`). Speculation cuts total task time from 715.2 s
  to 440.0 s (1.625×): low-copy 1.802× geomean (28/28 pairs, median 5.7 → 3.0 s; first-answer latency 1.652× over 27
  pairs, median 4.11 → 2.34 s), copy-heavy 1.424× (4/4 pairs, median 78.9 → 56.8 s; first-answer 1.383× over 3 pairs),
  and tool-continuation 1.588× (4/4 pairs, median 3.3 → 2.1 s). Under `INKLING_PREFILL_HALF=off` (`tasks-off`), both
  modes pass 31/36 (3 cap hits: `code-c13`, `code-c15`, `retrieval-t04`; `retrieval-t01` passes); 22/72 trials saw
  outside memory pressure, so `off` task timings are partial (11/28 valid low-copy pairs at 1.424× geomean, 4/4
  tool-continuation at 1.276×, 0/4 valid copy-heavy pairs).
- **Fixed-window decode and disabled control (5h.6):** Across three rotated half-prefill sessions (`a`, `b`, `c`; 95/108
  valid trials, 69/72 in quiet sessions `b+c`), every one of the 12 prompts has valid `spec/ar` (2–3 pairs) and
  `disabled/ar` (1–3 pairs) comparisons. `spec/ar` ranges from 1.351× to 2.842× (1.351× to 2.579× in quiet `b+c`),
  reaching 171.3 tok/s on `code-1024` and 170.2 tok/s on `qualify-prose-1024` — Decision 17's 200 tok/s single-stream
  decode target is met on no served prompt. The loaded-drafter `disabled` control (`draft: false`) is neutral against
  `ar` on every prompt (`disabled/ar` 0.926–1.077× across `a+b+c`, 0.983–1.053× in quiet `b+c`; every 95% interval
  includes 1.0). Under `off` (36/36 valid), `spec/ar` is 1.273–2.344× and `disabled/ar` is 0.973–1.178×.
- **Unseen prefill and the ~3% drafter hypothesis (5h.6):** Paired within gated sessions, neither `disabled/ar` nor
  `spec/ar` shows a statistically significant prefill slowdown at any length: every 95% interval includes 1.0 (`a+b+c`:
  `spec/ar` 1.012, 1.011, 1.013, 0.958, 0.953 across 1K–31.7K; quiet `b+c`: `disabled/ar` 0.990, 0.991, 1.014, 0.981,
  0.955 and `spec/ar` 1.013, 0.997, 1.026, 1.003, 1.000; `off`: `disabled/ar` 1.000–1.046 and `spec/ar` 1.009–1.091).
  SC2's ~3% drafter prefill penalty remains an unconfirmed hypothesis pending P7's per-kernel profile. Plain half
  prefill in quiet `b+c` sits at 1,892 / 2,001 / 1,933 / 1,833 / 1,694 tok/s (1K / 4K / 8K / 16K / 31.7K), meeting
  Decision 17's 2,000 tok/s only at 4K; `off` prefill runs at 1,538 / 1,616 / 1,551 / 1,386 / 1,375 tok/s.

## Configuration identity (5h.4)

Every record in `campaign/kernels/inkling-q4-qualify-runs/` carries this configuration block:

| Field | Value |
| --- | --- |
| Binary | `tensorfold-native` (`b925b31e…15f6`), built from `main` (`4334e9b34`, unchanged native code at `eb18fe252`) |
| Target checkpoint | `Inkling-2-Mini-oQ4e-mtp@3eea042751cd` (aggregate sha256 `4496e94e…d067`) |
| KV cache | `f16` keys and values (engine fixed: `zig/src/families/inkling2_loop/state.zig`) |
| Prefill rounding | Default: half experts and inputs (`INKLING_PREFILL_HALF` unset). Separate control: exact (`INKLING_PREFILL_HALF=off`). |
| Decode arithmetic | `f32` one-row kernels; grouped verify rows bit-equal to them (Decision 18) |
| Drafter | `Inkling-2-Mini-DFlash2@4f9088ea66c2` (prepared q4). `ar`: `--no-drafts` (not loaded); `disabled`: loaded + per-request `draft: false`; `spec`: loaded + server default. |
| Lookup | Prompt lookup chains in tree rounds (`spec` only); none for `ar` and `disabled` |
| Sampler | Greedy (`temperature: 0`, `seed: 1234`) |
| Context & cache | `32,768` total tokens, `INKLING_LOAD_LIMIT_GB=26`, `--prompt-cache-gib 2`, `heavy.py` 28 GB cap |
| Output policies | Completed tasks: EOS on, `max_tokens=8192` (+ follow-up `1024`). Fixed-window served: `max_tokens=512`, `ignore_eos=True`. |
| Host | Apple M5 Max (40 GPU cores, 48 GiB, macOS `27.2` / `26B123`, Zig `0.17.1`, `campaign/host.json`) |

## Completed tasks (5h.2, 5h.3)

Frozen quality suite (`cases_sha256` `65ae002d…7e46`): 36 scored tasks across `low-copy` (28), `copy-heavy` (4) and
`tool-continuation` (4), plus 8 unsupported (`structured-s01..s08`: grammar-constrained JSON schema / tool choice
refused by Inkling) and 4 excluded (`retrieval-t05..t08`: over 32,768 context with 8,192 reply headroom). Each
completed non-capped answer gets one follow-up turn (`FOLLOW_UP`, 1,024-token cap). `tasks.py equal tasks-half
tasks-off` exits 0: `ar` and `spec` are equal on every task and every follow-up turn under both prefill modes.

### Default half prefill (`tasks-half.jsonl`: 0/72 timing-invalid, 36/36 valid pairs)

| Copy class | Mode | Scored | Passed | Cap hits | Valid pairs | Median s | Median first-answer s | Total s | Geomean `ar/spec` (total / first-answer) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `low-copy` | `ar` | 28 | 26 | 1 | 28/28 | 5.7 | 4.11 | 381.9 | 1.802× / 1.652× (27 pairs) |
| `low-copy` | `spec` | 28 | 26 | 1 | 28/28 | 3.0 | 2.34 | 211.7 | — |
| `copy-heavy` | `ar` | 4 | 3 | 1 | 4/4 | 78.9 | 50.88 | 316.8 | 1.424× / 1.383× (3 pairs) |
| `copy-heavy` | `spec` | 4 | 3 | 1 | 4/4 | 56.8 | 37.31 | 217.9 | — |
| `tool-continuation` | `ar` | 4 | 3 | 0 | 4/4 | 3.3 | — | 16.5 | 1.588× / — |
| `tool-continuation` | `spec` | 4 | 3 | 0 | 4/4 | 2.1 | — | 10.4 | — |

Failures under half prefill (identical in `ar` and `spec`): `code-c04`, `code-c13` (8,192-token cap hit),
`retrieval-t01` (8,192-token cap hit, tracked under Q3), and `structured-s12` (no tool call). 60 follow-up turns were
recorded (54 with non-empty answer text), all equal between `ar` and `spec`.

### Exact prefill (`INKLING_PREFILL_HALF=off`, `tasks-off.jsonl`: 22/72 timing-invalid)

| Copy class | Mode | Scored | Passed | Cap hits | Valid pairs | Median s (all) | Total s (all) | Valid-pair geomean `ar/spec` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `low-copy` | `ar` | 28 | 25 | 2 | 11/28 | 5.8 | 567 | 1.424× (first-answer 1.287×, 10 pairs) |
| `low-copy` | `spec` | 28 | 25 | 2 | 11/28 | 3.9 | 312 | — |
| `copy-heavy` | `ar` | 4 | 3 | 1 | 0/4 | 81.7 | 365 | 0 valid pairs (pressure burst; do not quote) |
| `copy-heavy` | `spec` | 4 | 3 | 1 | 0/4 | 71.3 | 288 | — |
| `tool-continuation` | `ar` | 4 | 3 | 0 | 4/4 | 4.0 | 19 | 1.276× |
| `tool-continuation` | `spec` | 4 | 3 | 0 | 4/4 | 3.3 | 14 | — |

Failures under `off` (identical in `ar` and `spec`): `code-c04`, `code-c13` (cap hit), `code-c15` (cap hit),
`retrieval-t04` (cap hit), and `structured-s12`. `retrieval-t01` passes under `off` (5,402 tokens). 58 follow-up turns
were recorded (54 non-empty), all equal between `ar` and `spec`.

## Decision 19 gate (5h.5)

`campaign/quality/decision19.py` enforces Decision 19 as an executable gate (tested in `campaign/tests/test_decision19.py`).
On PH2's recorded runs (`main` arithmetic unchanged since PH2), it passes: 29/32 against exact's 28/32, lost case
`retrieval-t01` (also a new runaway), net runaways 2 against 3, oracle KL 1.021×, top-1 −0.23 points, Pi 7/8 on both,
with unsupported and over-context cases listed explicitly as unscored. Q4's completed-task set above reproduces the
same directional outcome: half passes 32/36 with 2 cap hits (`code-c13`, `retrieval-t01`) against exact's 31/36 with 3
cap hits (`code-c13`, `code-c15`, `retrieval-t04`).

## Fixed-window generation throughput and unseen prefill (5h.6)

512-token cold replies (`ignore_eos=True`) measure fixed-window generation throughput, not completed answers. Each
session rotates mode order across `ar`, `disabled` (drafter verified loaded in server log, per-request `draft: false`,
0 tokens drafted) and `spec`. Every reply matches `ar` (`run.py equal` exits 0 for `a`, `b`, `c` and `off`).

Session validity:
- **Half `a` (`served-a-qualify`, `served-a-dev`):** 26/36 trials valid (10 invalid from outside builds/GPU bursts up
  to 65.1% GPU and compressor spikes); `served-a-qualify-disabled` prefill curve failed its host-condition gates and is
  excluded from paired curve ratios.
- **Half `b` (`served-b-qualify`, `served-b-dev`):** 33/36 trials valid (3 minor compressor spikes); 12/12 curve gates
  passed.
- **Half `c` (`served-c-qualify`, `served-c-dev`):** 36/36 trials valid; 12/12 curve gates passed.
- **Exact `off` (`served-off-qualify`, `served-off-dev`):** 36/36 trials valid; 12/12 curve gates passed.

### Half prefill — all three rotated sessions (`a`, `b`, `c`: 95/108 valid trials)

| Prompt | `ar` tok/s (n) | `disabled` tok/s (n) | `spec` tok/s (n) | `spec/ar`: pairs, ratio (95%) | `disabled/ar`: pairs, ratio (95%) | Committed/round (`spec`) |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `code-1024` | 64.1 (3) | 64.7 (3) | 171.3 (3) | 3, 2.842 (1.872–4.313) | 3, 1.043 (0.815–1.336) | 7.76 |
| `prose-1024` | 70.1 (3) | 70.7 (3) | 134.2 (2) | 2, 1.867 (1.638–2.129) | 3, 1.038 (0.927–1.162) | 4.23 |
| `qualify-code-1024` | 66.3 (3) | 64.1 (3) | 134.6 (2) | 2, 2.031 (0.527–7.824) | 3, 0.934 (0.592–1.475) | 4.74 |
| `qualify-prose-1024` | 70.6 (3) | 69.7 (3) | 170.2 (3) | 3, 2.089 (1.211–3.606) | 3, 0.926 (0.715–1.199) | 6.40 |
| `qualify-structured-1024` | 65.0 (3) | 69.1 (1) | 127.9 (2) | 2, 1.915 (1.566–2.342) | 1, 1.006 | 3.88 |
| `structured-1024` | 69.8 (3) | 70.2 (3) | 107.1 (3) | 3, 1.417 (0.886–2.264) | 3, 1.027 (0.949–1.110) | 3.35 |
| `code-4096` | 63.2 (3) | 65.3 (3) | 140.1 (2) | 2, 2.169 (0.860–5.473) | 3, 1.027 (0.854–1.235) | 5.33 |
| `prose-4096` | 65.9 (3) | 65.1 (3) | 90.2 (2) | 2, 1.351 (1.030–1.771) | 3, 0.979 (0.885–1.083) | 2.91 |
| `qualify-code-4096` | 65.1 (2) | 65.0 (3) | 96.6 (2) | 2, 1.483 (0.906–2.426) | 2, 1.035 (0.747–1.434) | 3.28 |
| `qualify-prose-4096` | 64.3 (3) | 65.2 (3) | 100.8 (2) | 2, 1.552 (1.004–2.401) | 3, 1.077 (0.930–1.247) | 3.37 |
| `qualify-structured-4096` | 63.4 (3) | 65.5 (2) | 94.2 (2) | 2, 1.630 (0.777–3.420) | 2, 1.025 (0.897–1.172) | 3.76 |
| `structured-4096` | 66.0 (3) | 61.0 (2) | 101.4 (3) | 3, 1.541 (1.283–1.852) | 2, 1.025 (0.669–1.569) | 3.76 |

| Unseen prefill tokens | `ar` tok/s (sessions) | `disabled/ar`: pairs, ratio (95%) | `spec/ar`: pairs, ratio (95%) | Gated |
| ---: | ---: | --- | --- | --- |
| 1,024 | 1,845 (6) | 5, 1.035 (0.914–1.173) | 6, 1.012 (0.961–1.066) | yes |
| 4,096 | 1,983 (6) | 5, 1.024 (0.934–1.122) | 6, 1.011 (0.956–1.069) | yes |
| 8,192 | 1,877 (6) | 5, 1.025 (0.985–1.068) | 6, 1.013 (0.953–1.076) | yes |
| 16,384 | 1,801 (6) | 5, 0.954 (0.875–1.039) | 6, 0.958 (0.853–1.075) | yes |
| 31,744 | 1,674 (6) | 5, 0.908 (0.776–1.064) | 6, 0.953 (0.830–1.094) | yes |

### Half prefill — quiet sessions only (`b`, `c`: 69/72 valid trials, 24/24 curve gates passed)

Because session `a` experienced outside background builds during its run, `b+c` alone is shown alongside `a+b+c` for
absolute-rate reference (not to discard `a`):

| Prompt | `ar` tok/s (n) | `disabled` tok/s (n) | `spec` tok/s (n) | `spec/ar`: pairs, ratio (95%) | `disabled/ar`: pairs, ratio (95%) | Committed/round (`spec`) |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `code-1024` | 66.1 (2) | 65.1 (2) | 170.4 (2) | 2, 2.579 (2.545–2.614) | 2, 0.986 (0.742–1.312) | 7.76 |
| `prose-1024` | 71.9 (2) | 72.7 (2) | 134.2 (2) | 2, 1.867 (1.638–2.129) | 2, 1.011 (0.970–1.054) | 4.23 |
| `qualify-code-1024` | 65.9 (2) | 68.3 (2) | 134.6 (2) | 2, 2.031 (0.527–7.824) | 2, 1.033 (0.502–2.126) | 4.74 |
| `qualify-prose-1024` | 72.2 (2) | 71.0 (2) | 171.1 (2) | 2, 2.369 (1.649–3.404) | 2, 0.983 (0.940–1.029) | 6.40 |
| `qualify-structured-1024` | 66.8 (2) | 69.1 (1) | 127.9 (2) | 2, 1.915 (1.566–2.342) | 1, 1.006 | 3.88 |
| `structured-1024` | 70.0 (2) | 72.5 (2) | 110.6 (2) | 2, 1.578 (1.103–2.258) | 2, 1.034 (0.718–1.491) | 3.35 |
| `code-4096` | 64.5 (2) | 68.0 (2) | 140.1 (2) | 2, 2.169 (0.860–5.473) | 2, 1.053 (0.493–2.250) | 5.33 |
| `prose-4096` | 66.7 (2) | 66.8 (2) | 90.2 (2) | 2, 1.351 (1.030–1.771) | 2, 1.001 (0.840–1.191) | 2.91 |
| `qualify-code-4096` | 65.1 (2) | 67.4 (2) | 96.6 (2) | 2, 1.483 (0.906–2.426) | 2, 1.035 (0.747–1.434) | 3.28 |
| `qualify-prose-4096` | 64.9 (2) | 68.0 (2) | 100.8 (2) | 2, 1.552 (1.004–2.401) | 2, 1.046 (0.693–1.579) | 3.37 |
| `qualify-structured-4096` | 63.9 (2) | 65.5 (2) | 109.6 (1) | 1, 1.728 | 2, 1.025 (0.897–1.172) | 3.76 |
| `structured-4096` | 66.5 (2) | 65.4 (1) | 106.4 (2) | 2, 1.598 (0.969–2.635) | 1, 0.991 | 3.76 |

| Unseen prefill tokens | `ar` tok/s (sessions) | `disabled/ar`: pairs, ratio (95%) | `spec/ar`: pairs, ratio (95%) | Gated |
| ---: | ---: | --- | --- | --- |
| 1,024 | 1,892 (4) | 4, 0.990 (0.966–1.016) | 4, 1.013 (0.943–1.089) | yes |
| 4,096 | 2,001 (4) | 4, 0.991 (0.970–1.013) | 4, 0.997 (0.971–1.023) | yes |
| 8,192 | 1,933 (4) | 4, 1.014 (0.974–1.056) | 4, 1.026 (0.980–1.075) | yes |
| 16,384 | 1,833 (4) | 4, 0.981 (0.932–1.032) | 4, 1.003 (0.947–1.063) | yes |
| 31,744 | 1,694 (4) | 4, 0.955 (0.858–1.064) | 4, 1.000 (0.947–1.057) | yes |

### Exact prefill (`INKLING_PREFILL_HALF=off`: 36/36 valid trials, 12/12 curve gates passed)

Reported separately because exact prefill produces different greedy continuations (note committed tokens per round
differ from half on most prompts):

| Prompt | `ar` tok/s (n) | `disabled` tok/s (n) | `spec` tok/s (n) | `spec/ar`: pairs, ratio | `disabled/ar`: pairs, ratio | Committed/round (`spec`) |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `code-1024` | 63.7 (1) | 64.2 (1) | 149.3 (1) | 1, 2.344 | 1, 1.008 | 7.11 |
| `prose-1024` | 69.0 (1) | 69.9 (1) | 146.7 (1) | 1, 2.125 | 1, 1.012 | 5.28 |
| `qualify-code-1024` | 66.1 (1) | 64.3 (1) | 99.7 (1) | 1, 1.508 | 1, 0.973 | 3.91 |
| `qualify-prose-1024` | 69.9 (1) | 70.9 (1) | 154.6 (1) | 1, 2.212 | 1, 1.013 | 6.24 |
| `qualify-structured-1024` | 58.3 (1) | 68.6 (1) | 107.0 (1) | 1, 1.837 | 1, 1.178 | 3.58 |
| `structured-1024` | 69.1 (1) | 70.4 (1) | 106.5 (1) | 1, 1.540 | 1, 1.018 | 3.44 |
| `code-4096` | 65.0 (1) | 65.7 (1) | 123.9 (1) | 1, 1.905 | 1, 1.011 | 5.22 |
| `prose-4096` | 64.7 (1) | 65.5 (1) | 82.4 (1) | 1, 1.273 | 1, 1.012 | 2.68 |
| `qualify-code-4096` | 63.2 (1) | 65.0 (1) | 97.2 (1) | 1, 1.538 | 1, 1.028 | 3.71 |
| `qualify-prose-4096` | 63.9 (1) | 65.2 (1) | 84.0 (1) | 1, 1.314 | 1, 1.020 | 2.89 |
| `qualify-structured-4096` | 61.0 (1) | 63.6 (1) | 91.3 (1) | 1, 1.496 | 1, 1.043 | 3.46 |
| `structured-4096` | 65.2 (1) | 65.5 (1) | 95.8 (1) | 1, 1.468 | 1, 1.003 | 3.46 |

| Unseen prefill tokens | `ar` tok/s (sessions) | `disabled/ar`: pairs, ratio (95%) | `spec/ar`: pairs, ratio (95%) | Gated |
| ---: | ---: | --- | --- | --- |
| 1,024 | 1,538 (2) | 2, 1.000 (0.723–1.384) | 2, 1.009 (0.769–1.323) | yes |
| 4,096 | 1,616 (2) | 2, 1.023 (0.924–1.133) | 2, 1.021 (0.907–1.150) | yes |
| 8,192 | 1,551 (2) | 2, 1.046 (0.638–1.715) | 2, 1.043 (0.612–1.780) | yes |
| 16,384 | 1,386 (2) | 2, 1.030 (0.692–1.533) | 2, 1.091 (0.365–3.265) | yes |
| 31,744 | 1,375 (2) | 2, 0.975 (0.701–1.354) | 2, 1.025 (0.753–1.395) | yes |

## Footprint and first-request latency

From `GET /memory` (`proc_pid_rusage` physical footprint) across all served trials (identical bounds under half and
`off`), all well inside `INKLING_LOAD_LIMIT_GB=26` and `heavy.py`'s 28 GB cap:

| Mode | Ready footprint | Peak footprint | First-request wall time (`b+c` quiet half / `off`) |
| --- | ---: | ---: | ---: |
| `ar` (`--no-drafts`) | 18.98–21.09 GB | 21.15 GB | 0.84–0.89 s / 1.01–1.08 s |
| `disabled` (drafter loaded, `draft: false`) | 19.50–21.76 GB | 21.80 GB | 0.84–1.05 s / 1.09–1.12 s |
| `spec` (drafter loaded, drafting) | 19.53–21.92 GB | 21.95 GB | 0.85–0.98 s / 1.01–1.09 s |

## Verification and reproduction

- Unit tests: `python3 -I -m unittest discover -s campaign/tests -p 'test_*.py'` — 153/153 OK (including
  `test_served_equality.py` mutation tests and `test_decision19.py`).
- Reproduction from repo root with `Q4_BIN=.work/tmp/q4/native/bin/tensorfold-native` (`b925b31e…15f6`):

```sh
for step in "served a" "served b" "served c" "off" "tasks"; do
  Q4_BIN=.work/tmp/q4/native/bin/tensorfold-native campaign/kernels/inkling-q4-qualify-runs/run.sh $step
done
python3 -I campaign/kernels/inkling-q4-qualify-runs/tasks.py equal tasks-half tasks-off
python3 -I campaign/kernels/inkling-q4-qualify-runs/tasks.py table tasks-half
python3 -I campaign/kernels/inkling-q4-qualify-runs/tasks.py table tasks-off
S=campaign/kernels/inkling-sv2-served-runs/run.py
R=../inkling-q4-qualify-runs
python3 -I $S equal $R/served-a-qualify $R/served-a-dev $R/served-b-qualify $R/served-b-dev $R/served-c-qualify $R/served-c-dev $R/served-off-qualify $R/served-off-dev
python3 -I $S table $R/served-a-qualify $R/served-a-dev $R/served-b-qualify $R/served-b-dev $R/served-c-qualify $R/served-c-dev
python3 -I $S table $R/served-b-qualify $R/served-b-dev $R/served-c-qualify $R/served-c-dev
python3 -I $S table $R/served-off-qualify $R/served-off-dev
```
