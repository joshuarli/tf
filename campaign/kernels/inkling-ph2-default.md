# PH2: half-precision prefill by default

Roadmap item 5e. `INKLING_PREFILL_HALF` unset now means `both` (half routed-expert operands and half inputs);
`off` selects the exact path. PH (`inkling-ph-half.md`) qualified `both` under Decision 19 on P4's chunk schedule;
P5c then changed the schedule (Decision 20), so this receipt requalifies `both` against exact on P5c's schedule.

Verdict: `both` passes every Decision 19 condition on P5c's schedule and is the default from `d129b60f8`.

## Configurations

Both sides are one binary built from `d129b60f8` (main `7f7cb75a3` plus the flip), oQ4e weights, context 32,768,
`INKLING_LOAD_LIMIT_GB=26`. Exact is `INKLING_PREFILL_HALF=off`; half is the variable unset.

## Decision 19

| Condition | Exact (`off`) | Half (default) | Bar | Result |
| --- | --- | --- | --- | --- |
| oMLX oracle mean KL | 0.0194994 | 0.0199132 (1.021×) | ≤ 1.10× exact | pass |
| oMLX oracle top-1 | 94.96% (2,863/3,015) | 94.73% (2,856/3,015) | ≥ exact − 0.5 points | pass (−0.23) |
| Full quality | 28/44 | 29/44 | lose at most one case | pass (loses 1, gains 2) |
| Runaways (8,192-token cap, net) | 3 | 2 | no more than exact | pass |
| Pi tasks | 7/8 | 7/8 | no worse | pass |

Max KL is reported, not gated: 16.89 at `cells_code` position 378 under half against 8.01 under exact.

Full-set case changes, exact → half:

| Case | Exact | Half |
| --- | --- | --- |
| `code-c15` | runaway | pass |
| `retrieval-t04` | runaway | pass |
| `retrieval-t01` | pass | runaway |

`code-c13` runs away under both. Structured cases fail under both because the served host refuses grammar
constraints for Inkling. Four retrieval fixtures exceed the 32,768 context and are excluded on both. Pi fails
`read-and-answer` on both, as in PH.

The dev set alone points the other way: exact 8/11 and half 7/11, half's one loss being `retrieval-t01`. P5c's
schedule change alone turned exact's `code-c02` loop into a pass. Long-reasoning runaways flip with small numeric
changes in either direction, so Decision 19 judges them on the full set, net.

## Tests

`python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu,inkling-omlx-oracle` on the flip found one
test that asserted an exact-path invariant through the environment default: fused and unfused prefill byte-equal.
Half rounding falls at different points in the two, and serving never unfuses, so the test now builds its scratch
with exact rounding through `PrefillScratch.initRounding`. Then `inkling-gpu` 181/181 (peak 23.9 GB) and the
oracle gate pass with the half default.

## Reproduce

From the worktree at `d129b60f8`, after `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`:

```
INKLING_LOAD_LIMIT_GB=26 INKLING_PREFILL_HALF=off python3 campaign/heavy.py -- make -C campaign quality-inkling2-mini-oq4e SET=full BASELINE_WT=$PWD BIN=$PWD/zig-out/native/bin/tensorfold-native
INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- make -C campaign quality-inkling2-mini-oq4e SET=full BASELINE_WT=$PWD BIN=$PWD/zig-out/native/bin/tensorfold-native
INKLING_LOAD_LIMIT_GB=26 INKLING_PREFILL_HALF=off python3 campaign/heavy.py -- make -C campaign pi-harness-inkling2-mini-oq4e BIN=$PWD/zig-out/native/bin/tensorfold-native
INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- make -C campaign pi-harness-inkling2-mini-oq4e BIN=$PWD/zig-out/native/bin/tensorfold-native
python3 -I campaign/suite/suite.py run --names inkling-omlx-oracle
```

Results: `inkling-ph2-default-runs/quality/{off,half,half-dev}/`, `inkling-ph2-default-runs/pi/{off,half}.json`,
the oracle summary in `oracle-half.txt` and the run logs in `logs/`. The served prefill curve under the new
default is measured separately (roadmap Status).
