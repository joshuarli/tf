# Inkling 2 Mini: the oMLX correctness oracle

The correctness oracle for the oQ4e checkpoint is oMLX 0.8.3 running the same weights. Its teacher-forced
captures of the 16 parity histories replace the deleted f32 GGUF oracle, which ran the native checkpoint's
weights. `make -C campaign omlx-oracle-gate` holds TensorFold to the oracle within a margin of the baseline below.
The suite runs it as `inkling-omlx-oracle`, in the scenario tier only.

## Oracle

| Item | Value |
| --- | --- |
| Engine | oMLX 0.8.3, commit `e72868dac12df4be4ced2dc8355f031686283974`, in `.work/venv-omlx` |
| Packages | MLX 0.32.4, mlx-metal 0.32.4, mlx-lm 0.32.3.dev8+g0d4d3d04f, mlx-vlm 0.7.11, transformers 5.22.2, numpy 2.3.5, Python 3.13.18 |
| Checkpoint | `Inkling-2-Mini-oQ4e-mtp@3eea042751cd`, revision `3eea042751cd296b7266628a8342c1749242e1fe`; all 15 files matched `campaign/sources.lock.json` by size and sha256 |
| Model code | `omlx.patches.inkling2.vlm.Model`, the class `VLMBatchedEngine` loads for `inkling2_loop` |
| Settings | `ModelSettings(mtp_enabled=False)` with no drafter attached; TurboQuant KV off; `SchedulerConfig()` defaults |
| Forward | `engine._adapter(token_ids[1, n], cache=engine._adapter.make_cache())` on oMLX's MLX executor thread, at most 2,048 tokens a forward (the scheduler's prefill step), so one forward a history |
| Captures | `.work/parity/inkling2-mini/omlx-ref/<name>.npy`, float32 [tokens, 200058], 2.2 GB; row i holds the logits after token i |
| Digest | `b01512f0396eeb235f756913c6f9688e27e2d1fe117e48e49db9f0dd39b45816`, the sha256 of the sorted name-to-sha256 map |

`omlx_capture.py` starts `VLMBatchedEngine` in process, as `campaign/comparators/omlx/run.py` does, so every
patch oMLX installs at load is in place. No request goes through the scheduler. The `.npy` headers are byte for
byte those of the f32 captures. Each capture has a receipt beside it, `<name>.npy.receipt.json`.
`manifest.json` records the versions, the checkpoint file checks, the device, the settings, the command, the
campaign revision and every output sha256. A capture is kept on a later run only while its receipt names the
same engine, checkpoint, chunk and tokens and its sha256 still holds.

The oracle computes in bfloat16. Every logit before the f32 softcap is a bfloat16 value: a sampled row holds a
median of 2,583 distinct values among its 200,058, and 50 of the 3,015 rows tie exactly for the maximum. The
comparator therefore counts two rows as agreeing on top-1 when their sets of maximal tokens intersect. Breaking
ties by token index, as argmax does, would have counted 26 more TensorFold disagreements.

The capture is deterministic. `make -C campaign omlx-oracle-capture OMLX_ORACLE_CAPTURE_ARGS=--force` recaptured
all 16 histories in a new process from the committed script, and every file came back byte for byte: the digest
did not change. The manifest on disk is from that run and names the clean campaign revision `5dc8d29ab`. An earlier
smoke run, a third process, had already produced identical harbour and greeting files.

Once it holds the GPU lock, `make -C campaign omlx-oracle-capture` takes about 20 s. That includes hashing
16.8 GB of weights against `sources.lock.json` and a 2.2 s engine load. Harbour, the longest history, takes 1.8 to
3.7 s. The runs charged peaks of 20.6 and 22.9 GB against the 28 GB cap; MLX's own peak was 20.5 GB, on harbour.
The first run on this host compiled kernels: harbour took 10.4 s and the run charged 24.9 GB.

## Baseline: TensorFold c053f52b2 against the oracle

The runner `tf-inkling2-loop-run` was built at `c053f52b2` (sha256
`edc7b33f2604dae73d393827dac2f46613f6132278f5ceca4ebda70529d49210`). It captured each history with
`prefill 512`, the served prompt path, under the arithmetic policy `f32-activations+prefill-512`. The 16
captures took 1 min 37 s, one model load each, and charged at most 20.5 GB.

| Statistic | Value |
| --- | --- |
| Positions | 3,015 over 16 histories, two past the 1,024-token sliding window |
| Top-1 agreement | 2,861 (94.89%) |
| Top-1 where the oracle's top-2 margin is at least 0.1 | 2,753 of 2,872 (95.86%) |
| Mean KL, mean of the 16 history means | 0.0195085 |
| Mean KL, row-weighted | 0.0641971 |
| Max KL | 8.012 at harbour position 1300 |
| Max absolute logit difference | 27.8 |
| Non-finite logits | 0 in either capture |

| History | Rows | Top-1 | Top-1 % | Confident top-1 % | Mean KL | Max KL | At |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1241 | 92.89 | 94.78 | 0.04367 | 8.012 | 1300 |
| cells_code | 1178 | 1130 | 95.93 | 96.04 | 0.1101 | 7.567 | 1074 |
| greeting | 25 | 25 | 100.00 | 100.00 | 0.003636 | 0.02618 | 23 |
| explain_sliding | 34 | 32 | 94.12 | 93.75 | 0.005613 | 0.05578 | 7 |
| unit_test | 32 | 30 | 93.75 | 96.77 | 0.03933 | 0.9269 | 28 |
| refactor | 36 | 34 | 94.44 | 94.29 | 0.01148 | 0.07741 | 34 |
| math_mod | 36 | 35 | 97.22 | 97.06 | 0.003685 | 0.05586 | 7 |
| math_frac | 44 | 42 | 95.45 | 97.62 | 0.008951 | 0.08063 | 31 |
| dialogue | 44 | 43 | 97.73 | 97.67 | 0.004977 | 0.05549 | 7 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 0.005432 | 0.02582 | 26 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.005756 | 0.05579 | 7 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.007218 | 0.05583 | 7 |
| sql | 33 | 32 | 96.97 | 96.77 | 0.04381 | 1.009 | 31 |
| prose_review | 35 | 35 | 100.00 | 100.00 | 0.006686 | 0.05548 | 7 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0.005412 | 0.05588 | 7 |
| token_budget | 37 | 37 | 100.00 | 100.00 | 0.006351 | 0.05559 | 7 |

The worst histories are harbour, unit_test and explain_sliding by top-1, and cells_code, sql and harbour by mean
KL.

## The f32 captures in context

The same streamed comparison, with the f32 captures as the reference:

| Reference | Candidate | Top-1 | Confident top-1 | Mean KL | Row-weighted KL | Max KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| f32 | oMLX | 2,758 (91.48%) | 93.57% | 0.0415838 | 0.0986031 | 14.58, cells_code 950 |
| f32 | TensorFold | 2,763 (91.64%) | 93.71% | 0.0388930 | 0.0939557 | 13.92, cells_code 395 |
| oMLX | TensorFold | 2,861 (94.89%) | 95.86% | 0.0195085 | 0.0641971 | 8.01, harbour 1300 |

TensorFold against f32 reproduces the f16 KV receipt exactly: 2,763 matches and mean KL 0.0388930. oMLX does
slightly worse against f32 than TensorFold does.

Each of the 3,015 positions falls into one case of the three engines' top-1 agreement:

| Case | Positions | Points |
| --- | ---: | ---: |
| All three agree | 2,683 | 88.99 |
| oMLX and TensorFold agree, f32 differs | 160 | 5.31 |
| f32 and TensorFold agree, oMLX differs | 80 | 2.65 |
| f32 and oMLX agree, TensorFold differs | 57 | 1.89 |
| No majority | 35 | 1.16 |

TensorFold misses f32's top-1 at 252 positions, 8.36 points. In 160 of them, 5.31 points, oMLX makes the same
choice as TensorFold. The two engines share the oQ4e weights but not their activation precision, so that share
belongs to the checkpoint, not to TensorFold. TensorFold alone departs from two agreeing references at 57
positions, 1.89 points, fewer than oMLX's 80. The other 35 positions have no majority.

## Noise floor

Neither engine reproduces itself across forward shapes. A second oMLX capture at one token a forward, the
single-sequence decode path, is in `.work/parity/inkling2-mini/omlx-ref-chunk1` (digest `d741ce2a8d2753df...`).

| Reference | Candidate | Top-1 | Mean KL | Row-weighted KL | Max KL |
| --- | --- | ---: | ---: | ---: | ---: |
| oMLX, 2,048-token forwards | oMLX, one-token forwards | 2,870 (95.19%) | 0.0217629 | 0.0613418 | 10.82, harbour 1268 |
| oMLX, one-token forwards | TensorFold `prefill 512` | 2,855 (94.69%) | 0.0213212 | 0.0782923 | 16.80, cells_code 660 |

TensorFold's `prefill 1` captures exercise its served decode path, rows of at most 8. That capture was stopped
after 8 histories (harbour, cells_code, greeting, explain_sliding, unit_test, refactor, math_mod and math_frac,
2,721 positions) to free the shared GPU lock. On those 8:

| Reference | Candidate | Top-1 | Mean KL | Row-weighted KL |
| --- | --- | ---: | ---: | ---: |
| oMLX | TensorFold `prefill 512`, the baseline | 2,569 (94.41%) | 0.02831 | 0.07003 |
| oMLX | TensorFold `prefill 1` | 2,577 (94.71%) | 0.02679 | 0.06448 |
| oMLX | oMLX, one-token forwards | 2,585 (95.00%) | 0.02988 | 0.06661 |
| TensorFold `prefill 512` | TensorFold `prefill 1` | 2,605 (95.74%) | 0.01565 | 0.05311 |

TensorFold's two paths differ from each other almost as much as each differs from the oracle. Against the oracle
the decode path gains 8 positions on the prefill path; against f32 it loses 7 (2,494 against 2,501). TensorFold is
about as far from the oracle as the oracle's own decode path. The gate therefore detects changes against the
baseline. It cannot certify agreement finer than this floor.

## Gate

`make -C campaign omlx-oracle-gate` builds `tf-inkling2-loop-run`, captures the 16 histories with parity.mk's
`metal-logits` under `heavy.py`, and runs `omlx_gate.py gate`. Captures with receipts that still name the current
runner are reused. The report goes to `.work/parity/inkling2-mini/omlx-gate/<lane>-prefill-512.md`, with the
same results in `.json`.

`omlx_gate.py` derives the bounds from the baseline and margins in `campaign/parity/inkling2-mini/omlx_gate.json`:

| Statistic | Baseline | Bound |
| --- | ---: | ---: |
| Top-1 agreement | 94.892% | at least 94.392% (2,846 positions), the baseline less 0.5 points |
| Mean KL, mean of history means | 0.0195085 | at most 0.0292627, 1.5 times the baseline |
| Mean KL, row-weighted | 0.0641971 | at most 0.0962956, 1.5 times the baseline |
| Non-finite logits | 0 | 0 |

The mean of history means weights the 14 short histories, and the row-weighted mean the two long ones, so both are
bounded. Max KL is reported, not gated: across the pairings above it ranges from 8 to 17 nats, at a different
position each time.

Two measurements show how far legitimate changes have moved the prefill arithmetic. `0e38e2b5e` predates the f16
KV commit `c19cb4dd5` and the two attention commits after it. Its runner produced byte-identical `prefill 512`
captures for the 8 histories compared, both long ones included, so those commits moved no gate statistic. In
`campaign/kernels/inkling-prefill-tensorops.md`, kernel variants moved top-1 against f32 between 2,749 and 2,769
positions around the landed 2,763, and the old 91.5% floor sat 4 positions below the landed value. The 0.5-point
margin is 15 positions, one more than the drop of the worst of those variants against f32. Shifts against f32 need
not equal shifts against the oracle, so this sizes the margin; it does not prove those variants would pass.

Whatever the statistics, the gate fails when the oracle is not the pinned capture: an incomplete manifest, other
histories, a digest other than `oracle.digest`, or a file whose sha256 differs from its manifest entry. It also
fails when a candidate's receipt names another runner, checkpoint, token history or arithmetic policy.

`python3 -I campaign/suite/suite.py run --names inkling-omlx-oracle` exited 0 in 17.0 s at `5dc8d29ab`. That run
rebuilt nothing, because the runner it built was byte-identical to the baseline's (`edc7b33f...`), and it reused
all 16 captures by receipt. The comparison, including hashing the oracle files, takes about 11 s. A fresh capture
adds about 1 min 37 s, the time the same `metal-logits` recipe took for the baseline, one model load a history.
Each load waits for the GPU lock: a fresh run started while other lanes held it made no progress in 8 minutes and
was stopped. In the suite, preflight requires a complete oracle manifest and the oQ4e checkpoint.

To re-baseline after a deliberate arithmetic change, run the gate and read its per-history table and the f32
comparison for the cause. Then copy `top1`, `mean_kl`, `row_mean_kl` and `max_kl` from the gate's JSON summary,
with the revision and runner sha256, into `baseline` in `omlx_gate.json`, and record the reason in a receipt.
A recaptured oracle with a new digest needs the same re-measurement.

## Finding: harbour's last 56 positions

At harbour positions 1,280 to 1,335, the references agree with one another and TensorFold departs from all of
them. Mean KL over those 56 rows:

| Reference | Candidate | Mean KL |
| --- | --- | ---: |
| f32 | oMLX | 0.0384 |
| oMLX, 2,048-token forwards | oMLX, one-token forwards | 0.0215 |
| oMLX | TensorFold `prefill 512` | 0.2056 |
| oMLX | TensorFold `prefill 1` | 0.1391 |
| f32 | TensorFold `prefill 512` | 0.2282 |
| f32 | TensorFold `prefill 1` | 0.1535 |

At position 1300, f32 and both oMLX captures choose token 157679, and both TensorFold paths choose token 220.
Over positions 0 to 1,279 the order is reversed: TensorFold's mean KL is 0.0366 to the oracle, while oMLX's is
0.0635 to f32. The last 56 rows, 4.2% of harbour, hold 19.7% of harbour's KL to the oracle. The pre-f16 runner's
harbour capture is byte-identical to the baseline's, so f16 KV storage did not cause it. The cause was not
investigated.

## Reproduction

Run these from the repository root with `/opt/homebrew/bin` first on `PATH`, `TMPDIR` set to `.work/tmp` and
`ZIG_GLOBAL_CACHE_DIR` set to the shared Zig cache. `make -C campaign omlx-setup` is needed once, for the pinned
oMLX environment. The oracle and the gate:

```sh
make -C campaign omlx-oracle-capture
make -C campaign omlx-oracle-gate
python3 -I campaign/suite/suite.py run --names inkling-omlx-oracle
```

The baseline and calibration captures, with `W` and `R` naming the shared work directories:

```sh
W="$HOME/repos/tf/.work"
R="$W/parity/inkling2-mini"
OQ="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
BASE="$PWD/.work/tmp/base-build/bin/tf-inkling2-loop-run"
zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/base-build"
INKLING_LOAD_LIMIT_GB=26 make -C campaign -f parity/inkling2-mini/parity.mk metal-logits \
  RUNNER_MODE=prefill PREFILL_CHUNK=512 RUNNER="$BASE" INKLING_DIR="$OQ" \
  METAL_LOGITS="$R/omlx-gate/baseline-c053f52b2-prefill-512"
INKLING_LOAD_LIMIT_GB=26 make -C campaign -f parity/inkling2-mini/parity.mk metal-logits \
  RUNNER_MODE=prefill PREFILL_CHUNK=1 RUNNER="$BASE" INKLING_DIR="$OQ" \
  METAL_LOGITS="$R/omlx-gate/lane_o3-omlx-oracle-prefill-1"
make -C campaign omlx-oracle-capture OMLX_ORACLE_REF="$R/omlx-ref-chunk1" \
  OMLX_ORACLE_CAPTURE_ARGS="--chunk 1 --skip-weight-hashes"
mkdir -p .work/tmp/rev-0e38e2b5e
git archive 0e38e2b5e | tar -x -C .work/tmp/rev-0e38e2b5e
(cd .work/tmp/rev-0e38e2b5e && \
  zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$OLDPWD/.work/tmp/pre-f16-build")
INKLING_LOAD_LIMIT_GB=26 make -C campaign -f parity/inkling2-mini/parity.mk metal-logits \
  RUNNER_MODE=prefill PREFILL_CHUNK=512 RUNNER="$PWD/.work/tmp/pre-f16-build/bin/tf-inkling2-loop-run" \
  INKLING_DIR="$OQ" METAL_LOGITS="$R/omlx-gate/rev-0e38e2b5e-prefill-512"
```

The `prefill 1` and pre-f16 captures were stopped after 8 histories. Each comparison row above is one `compare`
run with that row's reference and candidate directories, for example oMLX against f32. The 8-history rows add
`--histories` with a file that holds only those 8 entries of `histories.json`.

```sh
uv run --frozen --project campaign/ref python -I -B campaign/parity/inkling2-mini/omlx_gate.py compare \
  --ref "$R/ref" --cand "$R/omlx-ref" --json .work/tmp/omlx-vs-f32.json --report .work/tmp/omlx-vs-f32.md
```

The three-engine breakdown reads the f32, oMLX and TensorFold directories from its arguments:

```python
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'campaign/parity/inkling2-mini')
from omlx_gate import open_capture

dirs = [Path(p) for p in sys.argv[1:4]]
keys = ('all agree', 'f32 differs', 'oMLX differs', 'TensorFold differs', 'no majority')
total = dict.fromkeys(keys, 0)
for h in json.loads(Path('campaign/parity/inkling2-mini/histories.json').read_text()):
    a, b, c = (open_capture(d, h['name'], 200058)[0] for d in dirs)
    for s in range(0, a.shape[0], 64):
        top = [x == x.max(axis=1, keepdims=True) for x in (np.asarray(y[s:s + 64]) for y in (a, b, c))]
        ab, ac, bc = (np.any(top[i] & top[j], axis=1) for i, j in ((0, 1), (0, 2), (1, 2)))
        cases = [ab & ac & bc, bc & ~ab & ~ac, ac & ~ab & ~bc, ab & ~ac & ~bc]
        cases.append(~(cases[0] | cases[1] | cases[2] | cases[3]))
        for k, m in zip(keys, cases):
            total[k] += int(m.sum())
print(total)
```

Run it with `.work/venv-ref/bin/python -I -B three_way.py "$R/ref" "$R/omlx-ref"
"$R/omlx-gate/baseline-c053f52b2-prefill-512"`.
