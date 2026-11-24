# Inkling 2 Mini served DFlash2 speculation (SV1)

**Verdict.** `tensorfold-native serve` decodes Inkling 2 Mini oQ4e with DF1's DFlash2 drafter when `INKLING_DRAFTER`
names it. Every served drafted reply checked is byte-identical to plain greedy decoding: the served Zig test, two suite
runs, 32 server-level checks, a server without the drafter, and main's own build. Speculation runs while one lane
decodes. Beside V1, single-stream cold decode rises from 62.5 to 91.8 tok/s on code-1024 (1.47×) and from 60.9 to 70.6
on prose-1024 (1.16×). It falls to 0.60–0.84× on structured-1024 and the three 4K prompts. A round takes 68–89 ms, 93%
of it the 8-row verify, and those replies commit 2.9–3.6 tokens a round where 4.2–4.9 break even. The switch therefore
stays opt-in until V1's verify cost, or a per-round depth cut, moves the break-even.

## Switch

- `INKLING_DRAFTER=~/models/Inkling-2-Mini-DFlash2@4f9088ea66c2` serves the drafter. `INKLING_DRAFTER_BITS` is 4
  (prepared q4, the default) or 0 (bf16).
- Off is `INKLING_DRAFTER` unset or empty, or `--no-drafts`. Off loads no drafter, sizes the batch scratch by
  `--parallel` and captures no taps: every path is main's.
- `--drafter DIR` stays refused. The shared registry (`zig/src/native/metal.zig`, `refusal`) refuses `--drafter` for
  families outside its list before `inkling2_loop.zig` opens, and the family's own check now names `INKLING_DRAFTER`.
- On a drafter server, a request with `"draft": false` decodes plain.
- The serving line reads `drafts: DFlash2 (<dir>)`, and the startup log gives the drafter's bytes and lane contexts.
- `campaign/bench/harness.py` gives the environment as `drafter_env(key)`. `bench.py`'s `MODE=drafter` still refuses
  Inkling, because it passes Flint's `--drafter`. `inkling-sv1-serve-runs/run.py` runs the equality checks
  below, and the timing pass on `bench.trial`.

## Design

`zig/src/native/inkling2_loop_spec.zig` (`Draft`) holds the drafter, bound to the target as `dflash.zig` binds it, the
grouped verifier (`group.Verifier`), prompt taps of 1,024 rows (the host's prefill chunk), step taps of 8 rows, and one
drafter context (`rt.Context`) a slot. A slot's context is swapped into the drafter to propose and swapped back, as
Flint's served drafter does.

**A round.** It runs while exactly one lane decodes, its request allows drafts, and its drafter context ends at its
cache's length.

1. Offer `min(7, max_tokens − emitted − 1)` drafts.
2. The drafter proposes a greedy chain from the newest token (edge weight 1, as DF1).
3. One grouped verify of the anchor and drafts captures every row's taps and picks on the GPU.
4. The accepted prefix and the bonus pick are committed, and those rows' taps are absorbed into the lane's context.
5. Each token goes through the host's own per-token path (`token`), which applies EOS, stop strings and
   `max_tokens`, so a reply can end inside a round.

Cancellation is checked after the round, as after a plain step.

**Two or more lanes.** Every lane takes main's batch step. The step captures taps, and each row is absorbed into its
lane's context, so a lane that is later alone speculates at once. Speculation never spans lanes: an 8-row verify costs
3.4× a one-row step (DF1), and Decision 17's target is one stream.

**Prompt taps.** The host's own prefill captures them, with its chunks and its reuse cut points. Capture copies rows and
changes no arithmetic, so the prompt state and the first token are main's.

**Prefix reuse.** The drafter reads its last 2,047 rows and their positions, and its session drops older rows.
Restoring those rows therefore gives it the state a fresh prefill gives. In `equal-a`, each prompt's cache-hit reply
drafted and accepted exactly what its cold reply did.

- At the end of each prefill a slot keeps its prompt's last 2,047 tap rows (`keepPrompt`, host memory).
- A store entry keeps the last 2,047 rows ending at its length, counted in its bytes. A 2,571-token entry grows from
  257 MB to 295 MB, so the 2 GiB store holds fewer entries with the drafter on.
- An admission restores the slot's context before the suffix prefills: a store hit from the entry's rows, a fork from
  the source slot's kept rows cut at the fork point, an own-slot reuse from the slot's kept rows, and no reuse as an
  empty context at 0.
- A fork reaches back only to the source prompt's oldest kept row, so it can lose up to the fork window (256 rows by
  default) of drafter context. The output does not change.

**Rollback and forks.** The verify writes no cache, and its commit writes only accepted rows. Rollback restores the
target cache as on main. A drafter context is rebuilt from kept rows at the next admission, never rolled back in place.
A reply that stops inside a round leaves the rest of that round's accepted rows committed. `retire` sees the true
length, so the prompt state is restored when the rollback rows cover them and the slot is cleared otherwise.

**Sampling.** The server refuses a temperature above 0 before admission (`greedy_only`), and the host refuses a sampled
request in `admit`, both as on main. No sampled request reaches the drafter, so there is no sampled decode to fall back
to. The per-request plain fallback is `"draft": false`.

**Failure.** A drafter error is logged once and turns the drafter off for the process. The round that failed becomes a
plain step. A target error fails the request, as a failed batch step does.

**Memory.** `Draft.bytes` counts the drafter's weights and scratch, the verifier, the taps, the lane contexts and the
kept rows. `inkling2_loop.zig` takes it off the load limit before it fits the context. At `--parallel 1` it is
0.625 GiB, and `--context 32768` still fits under `INKLING_LOAD_LIMIT_GB=26`. A lane adds about 118 MB: its 2,048-row
tap ring and drafter keys and values (80 MB, GPU) and its kept rows (38 MB, host). With the drafter on, the batch
scratch holds 8 rows whatever `--parallel` is, because the verify shares it.

**One duplication.** The served module repeats `dflash.zig`'s target binding (`admit`, `bind`, `affine8` and the
binding block). The engines module, rooted at `zig/src/native/metal.zig`, cannot import files under `families/`. One
registration line in `zig/build/metal_targets.zig` would let the host import `dflash.zig` instead.

## Equality

Every run went through `campaign/heavy.py` with `INKLING_LOAD_LIMIT_GB=26`. No served drafted reply differed from its
reference in any run.

**Served Zig test.** `test-inkling2-loop-served` is part of `test-inkling2-loop`, so the suite runs it
(`zig/src/native/inkling2_loop_spec_test.zig`). It opens the host through the registry's `open` with `INKLING_DRAFTER`
set, at `--parallel 2`, context 8,192 and a 2 GiB store. Each drafted reply is compared with the same request sent with
drafts off:

| Reply | Path | Tokens | Rounds | Drafted | Accepted | Cached |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| code | fresh 37-token prompt | 128 | 29 | 201 | 98 | 0 |
| eos | EOS inside a round, store hit | 10 | 2 | 14 | 8 | 36 |
| agent | fresh 2,572-token prompt | 192 | 54 | 357 | 137 | 0 |
| grown | own-slot reuse | 160 | 35 | 237 | 124 | 2,572 |
| fork | fork at 2,560 while its source decodes | 160 | 73 | 177 | 86 | 2,560 |
| two lanes | plain beside a 48-token reply, then alone | 192 | 90 | 295 | 101 | 0 |

The test then closes that host, opens one without `INKLING_DRAFTER` (main's path), and replays the cold code, agent and
two-lane requests: all four equal the drafted replies. Rounds and acceptance were identical in three processes. Peak
21.4 GB of the 28 GB cap.

**Suite.** `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0 on `b008f749a`
(`.work/suite/20261124T004745`) and on `3da9a564f`, the final code (`.work/suite/20261124T011310`). Both times
`inkling-gpu` passed 17/17 steps and 133/133 tests with none skipped: DF1's 123 and the served step's 10.

**Through the server.** `run.py equal --name equal-a` served binary `7eac62d43c16` with `--parallel 1`,
`--prompt-cache-gib 2`, `--context 32768` and `--temperature 0`. All 32 checks were byte-identical in the token ids, the
message (content and reasoning) and the finish reason:

- For each dev prompt: draft on with a cold prompt, draft off, draft on again after a prompt-cache hit, and draft on and
  off with EOS kept. All are 512 greedy tokens.
- A second turn of code-1024 reusing 1,349 cached tokens, draft on against off.
- A short prompt ("Is 7 a prime number? Answer with one word.") that stops at a natural EOS after 66 tokens, draft on
  against off.
- A second server without `INKLING_DRAFTER`: the same six replies.

`run.py against --name main-a` then served main's own build (`8cd81baf2`, binary `31654307e746`, built from
`git archive` in the worktree's `.work/tmp`): the same six replies. Off is main.

| Prompt | Prompt tokens | Draft-on rounds | Committed/round | Accepted/drafted | Token sha256 |
| --- | ---: | ---: | ---: | ---: | --- |
| code-1024 | 1,349 | 84 | 6.10 | 427/584 | `b824626b5f87` |
| prose-1024 | 1,154 | 95 | 5.39 | 416/664 | `3aff0891f35e` |
| structured-1024 | 1,124 | 144 | 3.56 | 367/1,001 | `67e4038a4ed3` |
| code-4096 | 5,280 | 157 | 3.26 | 354/1,079 | `5122c9d33180` |
| prose-4096 | 4,639 | 194 | 2.64 | 317/1,355 | `f1c8ad523000` |
| structured-4096 | 4,203 | 154 | 3.32 | 357/1,060 | `62aeee3bbb06` |

Committed per round is the server's `tokens_per_round`, which counts the prefill's first token.

- None of the six dev replies reaches EOS within 512 tokens: they are still in the reasoning block, so their EOS-kept
  requests stop at the limit as well. The natural-EOS evidence is the short prompt and the Zig test's EOS inside a
  round.
- On every server alike, the 4K prompts' first requests reused the 1,348, 1,153 and 1,123 tokens their 1K prompts had
  stored: the renderings share that prefix.
- The drafter adds 0.625 GiB at `--parallel 1` (prepared q4 weights, scratch, one lane context and its kept rows),
  within Decision 8's 4 GiB.

## Timing beside V1

`run.py timing --name timing-a` ran one server per cell and mode on binary `7eac62d43c16`, from 01:23 to 01:30 on
2026-11-24. V1's lane was running its own GPU timing in the same session, serialized with this pass by the GPU lock.
Each trial is `bench.trial`: a cold request, a warm repeat that hits the prompt cache, and an EOS-kept request, each 512
greedy tokens. Off is `--no-drafts`. On is `INKLING_DRAFTER`. The 1K cell ran off first and the 4K cell on first.

| Prompt | Prompt tokens | Off tok/s | Off ms/token | On tok/s | On/off | Committed/round | Round ms | Draft ms | Verify ms | Commit ms | Accepted/drafted | Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| code-1024 | 1349 | 62.5 | 16.0 | 91.8 | 1.47 | 6.08 | 67.7 | 5.1 | 62.5 | 0.21 | 427/584 | True |
| prose-1024 | 1154 | 60.9 | 16.4 | 70.6 | 1.16 | 5.38 | 77.0 | 5.1 | 71.7 | 0.23 | 416/664 | True |
| structured-1024 | 1124 | 55.6 | 18.0 | 47.0 | 0.84 | 3.55 | 76.0 | 5.0 | 70.8 | 0.22 | 367/1001 | True |
| code-4096 | 5280 | 55.3 | 18.1 | 44.9 | 0.81 | 3.52 | 79.1 | 5.3 | 73.6 | 0.22 | 366/1007 | False |
| prose-4096 | 4639 | 55.6 | 18.0 | 33.4 | 0.60 | 2.94 | 88.6 | 5.8 | 82.6 | 0.22 | 337/1200 | True |
| structured-4096 | 4203 | 57.5 | 17.4 | 41.3 | 0.72 | 3.48 | 84.7 | 5.5 | 79.0 | 0.20 | 364/1025 | True |

- **Columns.** Tok/s is the cold request's decode rate: the 511 tokens after the first, over the client's interval from
  the first token to the last. Committed/round is those 511 tokens over the reply's rounds. Round ms and its phases
  are the server's per-round means over drafted rounds, from the reply's `speculative.proposer` counters. Draft
  includes the first round's absorption of the prompt's tap rows, up to 2,047.
- **Where the time goes.** The verify is 92–93% of a round, the draft 6.5–7.5% and the commit 0.2–0.3%, as in DF1.
- **Break-even.** A drafted reply beats plain only when a round commits more than its round time over plain's time per
  token: 4.2–4.9 here. Only code-1024 (6.08) and prose-1024 (5.38) do. Decision 17 budgets 30 ms a round at 6 committed
  and 25 ms at 5: code-1024's 67.7 ms round is 2.3× its budget and prose-1024's 77.0 ms about 3×. No prompt reaches
  200 tok/s.
- **Against DF1's runner.** DF1 measured 61–77 ms rounds, 1.04–1.56× at 1K and 0.65–0.82× at 4K, on prompts cut to
  1,024 and 4,096 tokens. These served prompts render to 1,124–1,349 and 4,203–5,280 tokens through the chat template.
- **Equality in this pass.** For every prompt the six replies (off and on, each cold, warm and EOS-kept) are identical.
  The 4K replies differ from `equal-a`'s, where the 4K prompts reused their 1K prompts' stored prefixes. A retained
  prefix and a fresh prefill run different kernel schedules (`inkling-prefix.md`), and in each run the drafter and the
  plain servers saw the same history.

## Conditions

- **Lock and cap.** Every model process ran under `campaign/heavy.py`: the GPU lock and the 28 GB cap, with
  `INKLING_LOAD_LIMIT_GB=26` and `--context 32768`. The served test peaked at 21.4 GB charged. The timing servers'
  footprints peaked at 20.1–21.5 GB with the drafter and 19.4–20.9 GB without it.
- **Gate.** `gpu-gate-ready -1` read quiet before and after 11 of the 12 trials. Before the code-4096 drafter trial it
  read 63% device utilization against the 60% limit, so `bench.trial` marked that trial invalid. Its numbers sit with
  the other 4K trials.
- **Host.** Memory pressure stayed at level 1, with 48–88% free, and `pmset -g therm` recorded no warning. Other GPU
  users were WindowServer (0.5–1.1 s a trial) and the terminal (1.2–3.8 s a trial).
- **Prompts.** The 1K lengths are the `inkling-2-mini` pins, which apply because the oQ4e checkpoint's tokenizer and
  chat template hash the same. No 4K pin exists, so the first 4K server's `/v1/tokenize` measured those lengths.
- **Sustained load.** DF1 saw the 8-row verify slow by about 20% after a minute of continuous load. Each server ran its
  three prompts in the same order, so code ran first in both modes.
- **Benchmark checklist.**
  - Limiter: the verify forward, 93% of a round.
  - Pairing: one process per cell and mode, so this is a single bounded pass, not a promotion result.
  - Errors: none. Every reply in both modes matched.
  - Work done: every request emitted 512 tokens.
- **Builds.** `7eac62d43c16` is the native build of `8fd5e8906`'s Zig, which `3da9a564f` keeps. Main's reference
  build, `31654307e746`, is `8cd81baf2`.

## Reproduction

From the worktree:

```sh
zig build native -Dcpu=apple_m1
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
R=campaign/kernels/inkling-sv1-serve-runs/run.py
python3 -I $R equal --name equal-a
mkdir -p .work/tmp/main-src && git archive 8cd81baf2 | tar -x -C .work/tmp/main-src
(cd .work/tmp/main-src && zig build native -Dcpu=apple_m1)
python3 -I $R against --bin .work/tmp/main-src/zig-out/native/bin/tensorfold-native --reference equal-a --name main-a
python3 -I $R timing --name timing-a --cells 1024,4096
python3 -I $R table timing-a
```

The served test alone:

```sh
TF_INKLING2_LOOP_OQ4E_DIR=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd \
TF_INKLING2_DFLASH2_DIR=$HOME/models/Inkling-2-Mini-DFlash2@4f9088ea66c2 INKLING_LOAD_LIMIT_GB=26 \
python3 campaign/heavy.py -- zig build test-inkling2-loop-served -Dcpu=apple_m1 -j1 --summary all
```

Serving by hand:

```sh
INKLING_DRAFTER=$HOME/models/Inkling-2-Mini-DFlash2@4f9088ea66c2 INKLING_LOAD_LIMIT_GB=26 \
python3 campaign/heavy.py -- zig-out/native/bin/tensorfold-native serve \
  ~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd --parallel 1 --prompt-cache-gib 2 --context 32768 --temperature 0
```

## Open problems

- **Slower than plain at low acceptance.** At today's verify cost a reply needs 4.2–4.9 committed tokens a round to
  beat plain decoding, and four of the six dev prompts commit 2.9–3.6. A second turn of code-1024 committed 2.94 and
  took 13.8 s against 9.0 s plain (`equal-a`). DF1's width projection already beats 8 rows on five of its six prompts
  at these costs. A per-round depth or confidence cut is the next change, and V1's verify cost decides how far it
  goes.
- **One lane at a time.** With two or more lanes decoding, every lane decodes plain. `docs/recipes/SKILL.md` counts a
  drafts-off fallback at some stream count as a bug. This lane took the single-stream scope Decision 17 sets. Per-lane
  speculation needs `rt.Model.proposeBatch` over the lane contexts, a row budget across lanes, and V1's verify cost.
- **Forks lose drafter rows.** A fork's drafter context starts at the source prompt's oldest kept row, up to 256 rows
  short of a fresh prefill's.
- **Duplicate binding.** `inkling2_loop_spec.zig` repeats `dflash.zig`'s target binding until a registration line in
  `zig/build/metal_targets.zig` lets the engines module import `dflash.zig`.
- **`min_rows`.** It reports 1 for nearly every drafted reply, because the last round offers no draft.
- **Bench mode.** `make -C campaign bench MODE=drafter` still cannot serve Inkling: `bench.py` passes Flint's
  `--drafter`. It needs a model-aware drafter in `bench.py`, outside this lane's paths.
- **DF1's problems stand.** Acceptance at 4K (DF2), prompt taps from the half-precision prefill path, and the
  sustained-load slowdown of the 8-row verify.
