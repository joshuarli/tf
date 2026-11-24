# PH: qualifying half-precision prefill (`INKLING_PREFILL_HALF=both`) under Decision 19

Main `6f4f7ae19` (code of the DF2b merge `53a3b8894`), oQ4e `Inkling-2-Mini-oQ4e-mtp@3eea042751cd`, one binary
(`tensorfold-native` sha256 in `inkling-ph-half-runs/bin.sha256`), exact path against `INKLING_PREFILL_HALF=both`, greedy,
`INKLING_LOAD_LIMIT_GB=26`, `--context 32768`, run on 2026-11-24 through `campaign/heavy.py`.

## Verdict

Numerical checks pass, Pi ties, and full quality ties at 29/44. The two configurations swap one case each, and each
swap is a runaway to the 8,192-token cap. Read literally, Decision 19's "no new runaway" rule fails on `retrieval-t01`.

| Decision 19 check | Exact | Half (`both`) | Result |
| --- | ---: | ---: | --- |
| oMLX oracle mean KL (P4) | 0.0194994 | 0.0199132 (1.02×) | pass (≤ 1.10×) |
| oMLX oracle top-1 (P4) | 94.96% | 94.73% (−0.23) | pass (≥ −0.5 points) |
| Quality, full set (44 text cases) | 29/44 | 29/44 | one lost, one gained |
| Runaways to the 8,192 cap | `code-c02` | `retrieval-t01` | one new (`retrieval-t01`), one removed (`code-c02`) |
| Pi coding tasks | 7/8 | 7/8 | same; both fail `read-and-answer` |

## The swapped cases

- `code-c02`: exact reasons until the cap with no fenced code block (the loop open since P3c); half passes its tests.
- `retrieval-t01`: exact answers `QX-7419` (after about 5,345 tokens, per the dev run); half reasons until the cap
  with no `[[...]]` answer.

The same swap appears in P4's dev run and in this full run, so it is repeatable for these prompts: greedy decoding is
deterministic, and both long reasoning traces are sensitive to small numeric differences. All other cases agree
on pass or fail.

## Reproduce

```sh
python3 campaign/heavy.py -- make -C campaign quality-inkling2-mini-oq4e SET=full BASELINE_WT=$PWD BIN=$PWD/zig-out/native/bin/tensorfold-native
INKLING_PREFILL_HALF=both python3 campaign/heavy.py -- make -C campaign quality-inkling2-mini-oq4e SET=full BASELINE_WT=$PWD BIN=$PWD/zig-out/native/bin/tensorfold-native
python3 campaign/heavy.py -- make -C campaign pi-harness-inkling2-mini-oq4e BIN=$PWD/zig-out/native/bin/tensorfold-native
INKLING_PREFILL_HALF=both python3 campaign/heavy.py -- make -C campaign pi-harness-inkling2-mini-oq4e BIN=$PWD/zig-out/native/bin/tensorfold-native
```

Results: `inkling-ph-half-runs/quality/{exact,both}/`, `inkling-ph-half-runs/pi/{exact,both}.json`, and the run logs
beside them.
