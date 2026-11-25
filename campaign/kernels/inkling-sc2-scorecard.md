# Inkling 2 Mini integrated scorecard (SC2)

> **Superseded by Q4** (`inkling-q4-qualify.md`). An independent audit of `46061a3aa` found the following, and Q4
> rechecked each against the records:
> - The served equality check compared a termination-status field ("truncated"), so it would have passed different
>   capped outputs. Under the repaired check every recorded stream here still matches; the check was broken, not
>   the engine.
> - The 512-token trials are fixed-window generation throughput. All 72 natural-termination requests hit the
>   512-token cap, and all 144 saved first and forked replies have empty answer content (reasoning only). The
>   forked turn therefore follows an answer that was never given. No completed answer is measured here.
> - Trials: 72, of which 61 are valid and 11 invalid. code-1024 has **zero** valid plain/speculative pairs: its 72.4
>   and 185.8 tok/s are from different sessions. The table below's "one valid pair" is wrong.
> - The runner's code-1024 (233 tok/s) and the served code-1024 are different workloads: the runner prompt is 1,024
>   target tokens and the served one 1,349, and they generate different tokens (10.02 against 7.74 committed a
>   round). The gap is not server overhead. In the runner, 405 of 511 code-1024 tokens (79.3%) and 388 of 511
>   code-4096 tokens (75.9%) were accepted on prompt-lookup paths, against 79 to 109 on the other four prompts:
>   the code results are copy-friendly.
> - The ~3% drafter prefill cost is a difference of pooled medians. Paired within sessions, its intervals include 1
>   at every length, and the curves were not gated per request. It is a profiling hypothesis, not a measured cause.
> - Chronological gains mix three things. The code-4096 runner rose from 77.8 tok/s (V3, exact prefill) to 154.2
>   (V4's base, after PH2's half prefill, before V4's kernels) on a *different* greedy continuation that commits 7.51
>   tokens a round instead of 2.94. V4's own same-output gain is 1.038 times speculative and 1.125 times plain.
> - SV2's `idle` mode never loaded the drafter.

The integrated configuration is main `de53331e3`: half prefill by default (PH2), V4's verify kernels, SV2's cost table,
and served speculation by default. It is scored against Decision 17 on one pinned binary. Local measurements,
external reports and projections are kept in separate sections.

**Summary.**
- Prefill reaches 2,000 tok/s without a drafter at 1K, 4K and 8K. With the default drafter it falls 3% short
  (1,939 to 1,994). It misses at 16K (1,895) and near the cap (1,763) with or without the drafter.
- Speculative decode reaches 200 tok/s on no served prompt. The best is code-1024 at 186 served and 233 in the
  round runner.
- DF2b's round gain re-pairs at 1.219 over six pairs in two sessions. V3's re-pairs at 1.002, inconclusive as in its
  own receipt.

## Pins

- **Served binary.** `zig build native -Dcpu=apple_m1 -Dversion=1.0.0` at `de53331e3`, sha256 `c02fbd649e29…`.
  Target `Inkling-2-Mini-oQ4e-mtp@3eea042751cd`, drafter `Inkling-2-Mini-DFlash2@4f9088ea66c2` (prepared q4, named
  through `INKLING_DRAFTER`). Context 32,768, `INKLING_LOAD_LIMIT_GB=26`, prompt cache 2 GiB.
- **Round runner.** `tf-inkling2-loop-run` at `de53331e3` (sha256 `3f6310d9dfe6…`). Its plan and kernels equal the
  `17b9cdd33` runner whose six processes SV2 recorded (`inkling-sv2-served-runs/lane-a`…`f`), so the round tables
  below are those processes.
- **Host.** M5 Max. Every trial passes `bench.trial`'s gates; trials the gates rejected are left out and named.

## Prefill

Unseen prompts through `prefill_curve.py` on each server, with one-token replies, two prompts a length a session, and
sessions `dev-a` and `dev-b`, `qualify-a` and `qualify-b`. Client tok/s, median:

| Unseen prompt tokens | Plain (no drafter) | Default (drafter) | Default/plain | Decision 17 (2,000) |
| ---: | ---: | ---: | ---: | --- |
| 1,024 | 2,004 | 1,939 | 0.968 | plain meets, default misses |
| 4,096 | 2,067 | 1,994 | 0.965 | plain meets, default misses |
| 8,192 | 2,011 | 1,947 | 0.968 | plain meets, default misses |
| 16,384 | 1,885 | 1,895 | 1.006 | misses |
| 31,744 | 1,766 | 1,763 | 0.998 | misses |

The first request after readiness takes 0.98 to 1.05 s in every mode (SV2).

**Dominant costs of the misses.**
- **1K to 8K under the default.** The drafter's prompt-side work costs about 3%. While a drafter is loaded, every
  prefill chunk captures the drafter's tap rows (`h.prefill.taps`) and absorbs them into the drafter's context ring
  (`inkling2_loop_host.zig`, prefill loop). This happens whether or not the request then drafts, so a `draft: false`
  request on a drafter server pays it too. Measured as default against plain: 17 ms at 1K, 72 ms at 4K and 134 ms at
  8K. It is not profiled by kernel.
- **16K and the cap.** Prefill itself: PH2's curve fell off the same way (1,905 at 16K and 1,777 near the cap). P6
  owns that profile, which is parked unrun.

## Decode

Cold 512-token replies, one stream. The plain and default columns come from separate servers in one session.
Median over the sessions; the ratio interval covers sessions, and a single valid session gives no interval.

| Prompt | Plain tok/s | Default (speculative) tok/s | Default/plain | Committed/round |
| --- | ---: | ---: | --- | ---: |
| code-1024 | 72.4 | 185.8 | — (one valid pair) | 7.76 |
| prose-1024 | 72.5 | 137.0 | 1.889 | 4.23 |
| structured-1024 | 72.5 | 113.2 | 1.573 | 3.35 |
| code-4096 | — (both invalid) | 143.4 | — | 5.33 |
| prose-4096 | 69.5 | 93.7 | 1.339 | 2.91 |
| structured-4096 | 71.0 | 113.5 | 1.577 | 3.76 |
| qualify-code-1024 | 72.5 | 148.2 | 2.042 | 4.74 |
| qualify-prose-1024 | 72.5 | 172.2 | 2.374 (2.367–2.382) | 6.40 |
| qualify-structured-1024 | 72.3 | 127.5 | 1.763 (1.348–2.305) | 3.88 |
| qualify-code-4096 | 70.5 | 99.6 | 1.413 (1.173–1.702) | 3.28 |
| qualify-prose-4096 | 70.9 | 103.9 | 1.466 (1.369–1.569) | 3.37 |
| qualify-structured-4096 | 69.3 | 110.8 | 1.599 (1.572–1.627) | 3.76 |

Every valid reply equals plain decode's: cold, prompt-cache hit, forked turn and natural termination
(`run.py equal`). Session `dev-b` lost five plain trials to the compressor and memory-pressure gates, and both
sessions lost one default trial each (code-1024 in `dev-a`, qualify-code-1024 in `qualify-b`). The dev prompts'
ratios therefore rest on one session. Decision 17's 200 tok/s is met on no served prompt.

### Round time and accepted work

The round runner's tree rounds on the six dev prompts (SV2's six processes), median of process means, and Decision
17's round budgets (20, 25, 30 and 40 ms at 4, 5, 6 and 8 committed tokens, 5 ms a committed token):

| Prompt | Committed/round | Round ms | Draft ms | Verify ms | Spec tok/s | Budget ms | ms a committed token | Meets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| code-1024 | 10.02 | 43.0 | 2.32 | 40.7 | 232.9 | 50.1 | 4.29 | yes |
| prose-1024 | 4.48 | 38.4 | 4.60 | 33.8 | 116.8 | 22.4 | 8.56 | no |
| structured-1024 | 3.79 | 35.7 | 4.59 | 31.1 | 105.9 | 18.9 | 9.44 | no |
| code-4096 | 7.51 | 46.0 | 3.47 | 42.6 | 163.2 | 37.6 | 6.13 | no |
| prose-4096 | 3.45 | 37.2 | 4.94 | 32.3 | 92.8 | 17.3 | 10.78 | no |
| structured-4096 | 3.45 | 35.5 | 4.70 | 30.7 | 97.4 | 17.3 | 10.27 | no |

The runner's code-1024 (233) is higher than served (186) because its prompt is tokenized as the runner's own
1,024-token prompt and its rounds run without the server's per-request work. The served figure is the scored one.

**Dominant cost of the decode misses.** The verify is 31 to 43 ms of a 36 to 46 ms round. V4 measured that its
routed experts take 19 to 33 ms of a 38 to 64 ms step at 8 and 16 rows, at 1.6 to 2.0 times their bytes. At 3.5 to
4.5 committed tokens a round, prose and structured would need rounds of 17 to 22 ms, about half today's verify alone.

## Quality

No arithmetic changed after PH2's qualification. V4 keeps every row and prefill bit-equal (228/228 cross-build
hashes), and SV2's speculation is greedy-exact (every reply equal above). PH2's results therefore stand for this
configuration: oracle KL 1.021×, top-1 −0.23 points, full quality 29/44, runaways 2, Pi 7/8. The integrated suite on
SV2's merge passed the oMLX oracle gate (top-1 at least 94.392%, mean KL at most 0.0293). The long-reasoning runaways
(`retrieval-t01` under half prefill, `code-c02` under exact) remain Q3's.

## Footprint

The largest peak over the scorecard's decode trials is 21.94 GB with the drafter and 21.13 GB without it. The ready
footprint is 18.98 GB without the drafter and 19.39 to 19.53 GB with it. Both stay under the 26 GB load limit and the
28 GB cap.

## Re-paired round gains

DF2b's and V3's round pairs were re-run from their own runners, rebuilt from `git archive` of their recorded
commits, in their receipts' balanced order. The six pairs ran in two sessions (06:19 to 06:51, then from 07:31),
with the scorecard's served sessions in between. `pairs.sh` waited out host deferrals; the session-1 attempt that
failed on a gate deferral left no record. The rebuilt runners' hashes differ from the receipts' recorded ones
(DF2b main `56f9beaa841f…` against `1e9e1e18e4e0…`), so these are rebuilds from source, not the recorded binaries.

| Item | Builds | Six process pairs | 95% interval | Receipt's figure |
| --- | --- | ---: | --- | --- |
| DF2b | `50b4a681a` chain against `d95cc3fe0` tree | 1.219 | 1.203–1.235 | 1.193 (1.140–1.249) |
| V3 | `53a3b8894` against `e3b30971c`, both tree | 1.002 | 0.969–1.037 | 1.022 (0.986–1.059) |

- **DF2b** reproduces and tightens. Every pair is 1.197 to 1.237, and code-1024 rises from 149.8 to 180.4 tok/s.
- **V3** stays inconclusive. Pairs a to e give 1.008 to 1.024. Pair f gives 0.940: its lane process ran slow on
  several prompts (code-4096 66.5 against 77.8), and its gate logs show no deferral or contention, so it is kept.
  V3 remains kept for cumulative benefit (Decision 11). Its 9-to-16-row cost cut is in the cost table, and SV2's
  refit was neutral.

## External and projected results

- **External.** Splash 1.7.2's 213 tok/s on a 32-core M5 Max is an unqualified prerelease report (Decision 17).
  S1's local reproduction on this host prefilled unseen prompts at 2,200 to 3,250 tok/s, with 28 to 36 ms cycles
  committing 3.2 to 9.1 tokens; Splash reached 200 tok/s only on code (S1's receipt).
- **Projected.** None. This scorecard makes no projection.

## Reproduction

From the worktree, with `T` the build directory and `S=campaign/kernels/inkling-sv2-served-runs/run.py`:

```sh
for c in 50b4a681a d95cc3fe0 53a3b8894 e3b30971c; do
  mkdir -p $T/src-$c && git archive $c | tar -x -C $T/src-$c
  (cd $T/src-$c && zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix $T/run-$c); done
zig build native -Dcpu=apple_m1 -Dversion=1.0.0 && cp zig-out/native/bin/tensorfold-native $T/native/
SC2_RUNNERS=$T campaign/kernels/inkling-sc2-scorecard-runs/pairs.sh 1
for s in "dev-a dev 0" "qualify-a qualify 1"; do set -- $s
  python3 -I $S session --bin $T/native/tensorfold-native --name ../inkling-sc2-scorecard-runs/served/$1 --set $2 \
    --rotate $3 --curve 1024,4096,8192,16384,31744; done
SC2_RUNNERS=$T campaign/kernels/inkling-sc2-scorecard-runs/pairs.sh 2
# then qualify-b (rotate 2) and dev-b (rotate 1) as above, after a 15 min rest
R=../inkling-sc2-scorecard-runs/served
python3 -I $S equal $R/dev-a $R/dev-b $R/qualify-a $R/qualify-b
python3 -I $S table $R/dev-a $R/dev-b $R/qualify-a $R/qualify-b
cd campaign/kernels
python3 -I inkling-df2b-tree-runs/run.py pairs ../inkling-sc2-scorecard-runs/df2b/main-{a..f} ../inkling-sc2-scorecard-runs/df2b/df2b-{a..f}
python3 -I inkling-v3-experts-runs/run.py pairs ../inkling-sc2-scorecard-runs/v3/main-{a..f} ../inkling-sc2-scorecard-runs/v3/lane-{a..f}
python3 -I inkling-sv2-served-runs/rounds.py roundtable lane-a,lane-b,lane-c,lane-d,lane-e,lane-f
python3 -I inkling-sv2-served-runs/rounds.py budgets lane-a,lane-b,lane-c,lane-d,lane-e,lane-f
```

(The `pairs` commands take comma-joined names; the brace lists above stand for them.)

## Open

- **The drafter's prompt-side cost** (about 3% at 1K to 8K) is paid by every request on a drafter server. Capturing
  and absorbing only the rows the drafter's 2,047-row window and its restore points need may remove most of it at 4K
  and 8K. This needs a profile by kernel first.
- **Prefill at 16K and the cap** is P6's profile, which is parked unrun.
- **Decode** is bound by the verify's routed experts past one row (V4's open item).
- **The `idle` mode** of SV2's harness never loaded the drafter (`--no-drafts` drops it at load). Measuring a loaded
  but disabled drafter needs per-request `draft: false` in `bench.trial`.
