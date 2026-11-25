# P6: the 1K dev tail, and where 16K and the cap spend their time

Item 5f of Decision 17, judged by Decision 20. Lane `lane/p6-tail`, base main `f65124c46`.
M5 Max, oQ4e `3eea042751cd`, greedy, context 32,768, `INKLING_LOAD_LIMIT_GB=26`, half prefill (the default since PH2).
Raw scripts are in [`inkling-p6-tail-runs/`](inkling-p6-tail-runs/).

**Status: parked, not promoted.** The code, its tests and its scripts were complete when the coordinating session
stopped on 2026-11-25. On wind-down the host build was checked: `zig build test-inkling2-loop-host --summary all`
exits 0 with 10/10 steps and 37/37 tests. No GPU test, timing, served-curve, Decision 20 or quality result exists, so
this receipt makes no acceptance claim. The lane keeps the work as a starting point for a retry. Since the lane was
started, V4 has landed on main (`ec925cb41`) and changed `prefill.zig`, a file this lane does not own; a retry must
rebase onto current main and re-run the correctness checks before measuring anything.

## What changed

- The host's last prefill chunk takes the rest of the prompt once it fits in 1,152 rows (`last_chunk_rows`, 1,024 +
  128): the prefill scratch holds 1,152 rows, and a host with a drafter keeps its 1,024 tap rows as the limit. Under
  prefix reuse the last token no longer runs alone.
- The store keeps a prompt's whole state together with its last row's logits (200,058 f32, charged to the store), so
  a repeated prompt takes its first token from them with no replay. A state-only entry of the same tokens is replaced
  by the whole entry. Today the server refuses sampled requests before any store lookup, so no sampled request reaches
  a stored entry; the stored row keeps a future sampler exact.
- `tf-inkling2-prefix-check` takes chunks past 1,024 rows, runs many schedules in one load (`jobs`), and times a
  prompt's chunks the way the host runs them (`time`, `profile` on a Metal 4 queue for the timeline).

The lane's new schedule equals main's reuse-off schedule on every served-curve length except 1K dev and the 8,262-token
prompt; the Decision 20 comparisons are scripted but unrun.

## Checks written, not executed

- Planner and store-selection host tests in `inkling2_loop_prefix.zig`: a whole prompt hits only an entry holding its
  last row's logits, a whole entry replaces a state-only entry of the same tokens, and a merged last chunk of up to
  1,152 rows splits as planned at every length from 2 to 8,262.
- A served GPU test in `inkling2_loop_spec_test.zig`: the stored row bit-equal to a cold prefill's last row, a whole hit
  equal to the cold reply, EOS as the first token, `max_tokens` 0, a refused sampled request ending as on a cold host,
  a fork inside the chunk equal to a store hit, and reuse off equal to reuse on.
- `zig build test-inkling2-loop-host --summary all`: 10/10 steps, 37/37 tests (run on wind-down, no model load).
- `lean_new.py` on the saved base and head outputs: 256 problems on both, 0 new. `git diff --check` is clean, and every
  script in `inkling-p6-tail-runs/` passes `shellcheck` and `dash -n`.

## Retry condition

1. Rebase `lane/p6-tail` onto current main (past V4) and re-run the lean and whitespace checks.
2. Build the frozen main and lane binaries with `builds.sh`, then run `tests.sh` (host, served and family GPU tests),
   `d20.sh` (oracle and lane-vs-main schedule comparisons) and `quality.sh`.
3. Only after those pass and the GPU is quiet, run the gated timing: `tail.sh`, `session.sh 1`, `session.sh 2` and
   `profile.sh`. Promotion needs Decision 11 pairs and Decision 20 to pass; a schedule change must also keep exact
   replay inside one schedule (a repeated prompt, a store hit and a fork restore the state the chunk computed).

The lane owns only its four host files and `state.zig`; V4 owns `prefill.zig` and the kernels. Any fix that needs a
kernel or prefill change ends this lane and needs a new one.
