# P5c: Inkling 2 Mini served prefill in full chunks

Item P5c of Decision 17, judged by Decision 20. Lane `lane/p5c-chunks`; the code is at `335ea7f43`, with the timed
schedule at `22b8fec3b`. Base: main `53a3b8894`. The served curves time main `a1984516d` (V3 merged) against
`a1984516d` with this item's code. M5 Max, oQ4e `3eea042751cd`, greedy, context 32,768, `INKLING_LOAD_LIMIT_GB=26`.
Raw records are in [`inkling-p5c-chunks-runs/`](inkling-p5c-chunks-runs/).

**Verdict.** Served prompts now prefill in full 1,024-row chunks. Each chunk captures the fork snapshots and the
first-point store that fall inside it, so the host no longer stops chunks at fork points. The last token still runs
alone, so a repeated prompt replays its cold reply byte for byte through the len-1 store. Every reuse path restores the
state its own chunk computed:
- a capture equals `copyRows` at the chunk's start and end, and inside the chunk it equals the state assembled from
  the chunk's own data;
- forks, store hits and own-slot reuse give identical replies;
- the acceptance suite passes, with D2's oracle numbers unchanged.

Prefill bytes depend on chunk boundaries, so the lane cannot be byte-equal to main. Main's own reuse-off schedule is not
either. Under Decision 20 the new schedule passes all three tests:
- (a) against the oracle, mean KL is 1.001× main's and top-1 is 0.13 points lower;
- (b) on the probe, the spread from main equals main's own spread between reuse on and reuse off;
- (c) quality dev scores 8/11 against 7/11, with no case lost.

In six interleaved pairs over two sessions, served prefill is 1.175× main's (95% interval 1.109–1.245). By length:
held-out 1K 1.56×, dev 1K 1.17×, 4K 1.12×, 8K 1.06×, 16K and 31.7K 1.03×. From 4K up and on held-out 1K, the lane with
reuse on serves as fast as with reuse off. No length reaches 2,000 tok/s: the best are 1,859 at 1K and 1,826 at 4K.

The first request's unexplained seconds sat in its first chunk, outside GPU time. It is Metal's first command buffer on
the weights and caches, and its size tracks the host's memory state after load. A one-token warm-up at load
(`335ea7f43`) cuts the first request to 1.03–1.06 s, from 2.4–2.9 s on the lane in the same session, and adds 0.75–1.81 s
to load.

| Prompt | main tok/s | lane tok/s | lane / main (six pairs) | 2,000 met |
| --- | ---: | ---: | ---: | --- |
| first request (1,258 tokens), server seconds | 1.24-3.82 | 1.03-1.06 with the warm-up (1.14-4.90 without) | - | - |
| 1K dev | 1,390-1,451 | 1,353-1,857 | 1.17 | no |
| 1K held-out | 1,048-1,103 | 1,510-1,859 | 1.56 | no |
| 4K | 1,453-1,598 | 1,522-1,826 | 1.12 | no |
| 8K | 1,382-1,627 | 1,586-1,720 | 1.06 | no |
| 16K | 1,411-1,529 | 1,428-1,591 | 1.03 | no |
| 31.7K | 1,335-1,461 | 1,428-1,498 | 1.03 | no |

Client tok/s, each trial's median at a length. Ratios are geometric means of per-request lane/main over the six pairs.

## What changed

| Commit | Change |
| --- | --- |
| `2e7b62b65` | Prefill captures. `PrefillScratch.captures` lists points inside the chunk. At each one the chunk saves what `Cache.Rollback.copyRows` would save once the cache holds that many positions: the four convolution rings and the local key and value rows the rest of the prompt overwrites. `i2p_ring_capture` runs before each ring store and `i2p_kv_capture` before each local slot's key and value store. Each takes every value from the chunk's own rows or from the ring before the chunk overwrites it. Both only copy 32- and 16-bit words. `Rollback.savedRing` and `savedRows` define the saved layout for `copyRows` and the captures alike. |
| `c9b045b14` | The host's schedule. `prefix.firstPoint` and `prefix.chunk` give the points. Chunks run 1,024 rows up to the last prompt token, and that token runs alone as on main. Each chunk captures the fork points inside it or at its start. The first point's store copies the cache after the chunk, then restores that point's snapshot into the copy. The drafter's taps are absorbed in pieces ending at each point, so that store keeps the drafter's rows up to the point. |
| `22b8fec3b` | `tf-inkling2-prefix-check MODEL IDS.json CHUNKS OUT` writes every row's logits for a given chunk schedule. Every schedule comparison below uses it. |
| `335ea7f43` | At load the host prefills one token through slot 0, then zero-fills slot 0's cache, rollback and snapshot buffers. That moves the first command buffer's cost off the first request (see First request). |

Nothing changes with prefix reuse off: main already prefilled that path in 1,024-row chunks with the last token inside the
last chunk. With reuse on, a prompt's schedule changes as below (`schedules.py`); the snapshot points do not:

| Prompt tokens | main | lane |
| ---: | --- | --- |
| 1,258 (first request) | 1,024, 128, 105, 1 | 1,024, 233, 1 |
| 1,027 / 1,032 / 1,033 (dev 1K) | 896, 128, 2-8, 1 | 1,024, 2-8, 1 |
| 1,003 / 1,014 / 1,019 (held-out 1K) | 768, 128, 106-122, 1 | 1,002-1,018, 1 |
| 4,003 / 4,058 | 3 x 1,024, 768, 128, 34 / 89, 1 | 3 x 1,024, 930 / 985, 1 |
| 31,627 / 32,125 | 30 x 1,024 + 768, 128, 10, 1 / 31 x 1,024 + 128, 128, 124, 1 | 30 x 1,024 + 906, 1 / 31 x 1,024 + 380, 1 |

## Exactness of reuse, forks and the store

A changed schedule changes every row's logits (next section), so the lane cannot be byte-equal to main. What must hold
exactly is narrower: a snapshot or store entry holds the state its own chunk computed, and every reuse path restores
that state byte for byte. New tests:

- `oQ4e prefill captures inside a chunk save what copyRows saves at that row and leave the prefill unchanged`
  (`prefill_test.zig`). Prompt length 1,500; one 300-row chunk, then the 1,024-row chunk under test, which crosses
  the 1,024-row ring. The test captures at eight points: the chunk's start; one and three rows in, where some
  convolution taps still sit in the ring before the chunk; row 1,024, where the ring wraps; rows 1,100 and 1,200,
  where some saved key rows come from the chunk itself; and the chunk's last row and its end. Four checks:
  - the logits and every cache byte equal a run of the same chunk without captures;
  - at the start and the end, each snapshot equals `copyRows` on the cache before and after the chunk;
  - inside the chunk, each snapshot equals the state assembled from `copyRows` before and after the chunk and from
    the chunk's convolution inputs, which `ConvCapture` records;
  - each snapshot carries its length and row count.
- `inkling host: full-chunk prefill keeps a repeated prompt equal to its cold reply and a fork equal to a store hit
  at a point inside a chunk` (`inkling2_loop_spec_test.zig`, served hosts on oQ4e, prompt cache 2 GiB). A 2,572-token
  prompt (the agent fixture, one token changed) saves fork points 2,432 and 2,560 inside its third chunk. Four
  replies must be byte-equal:
  - the prompt sent again, through the len-1 store (cached 2,571), equals its cold reply;
  - a prompt sharing 2,482 tokens, forked from the source slot at 2,432 on a two-lane host, equals the same prompt on
    a one-lane host, where it hits the first-point store at 2,432;
  - the prompt extended by own-slot reuse (cached 2,572) is equal on both hosts;
  - the cold replies are equal on both hosts.
- `a fresh prompt prefills in full chunks with fork points inside them and its last token alone`
  (`inkling2_loop_prefix.zig`, host only): the schedule and points for nine lengths and starts.
- `inkling host: the warm-up at load leaves the first slot's caches, rollback and snapshots zero and stores nothing`
  (`inkling2_loop_spec_test.zig`): the load-time warm-up leaves no state (see First request).

The existing tests stay in place. P5c moved the rollback and fork tests from `state.zig` to `prefill_test.zig`, and the
served speculation test still checks drafted replies against draft-off replies and a drafter-less host. That test covers
reuse, the store, a fork at 2,560 (now inside a chunk), two lanes and EOS inside a lookup round.

Checks:
- `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu,inkling-omlx-oracle` exit 0, no skips, on
  the final head `335ea7f43` (`suite-335ea7f43/`): `inkling-gpu` 180/180 tests in 17/17 steps, with all three new GPU
  tests, peak charged 24.0 GB; `inkling-host` ok (24/24 run, its other steps cached); the oMLX gate passes at D2's and
  P4's numbers exactly (next section). On `22b8fec3b` (`suite/`) the same command gave `inkling-host` 36/36 and
  `inkling-gpu` 179/179. One run in between on `335ea7f43` stopped without a failed test: heavy.py stopped
  `inkling-gpu` on critical host memory pressure, with outside swap at 10.3 of 11.3 GB (`suite-335ea7f43/stopped/`).
- `zig build native tf-inkling2-loop-run tf-inkling2-prefix-check -Dcpu=apple_m1` exit 0.
- `tools/zig/lean_new.py` against the base: 255 problems, 256 on the base, 0 new. Moving the tests retires one base
  problem: `state.zig`'s direct `getenv` outside a test block.

## What the schedule change does to the logits (Decision 20)

Prefill bytes depend on the chunk boundaries. Each expert's arithmetic depends on how many of the chunk's rows route
to it: grouped f32 kernels below 14 rows for gate and up and below 8 for down, tensor-unit matrices above. Each dense
projection's kernel depends on the chunk's row count. On main's own runner, the same 1,024 held-out code tokens as one
chunk and as 896 + 128 agree bit for bit on no row, row 0 included. Top-1 agrees on 99.32% of rows, mean KL is 0.0036
and max |Δlogit| is 14.4 (`schedule/chunk-1024-vs-896.txt`). So no schedule that moves a boundary can be byte-equal
to main. Main itself is not one schedule either: `INKLING_PREFIX_REUSE=0` already gives other bytes. Decision 20 therefore
judges the schedule against the independent oracle.

**The max |Δlogit| is a routing flip at a near tie, not a bug** (`schedule/flip.txt`, routes from main's runner
`prefillroutes`). Row 834 (token 166502: −10.10 against 4.31) keeps its top-1 token. Its router logits agree exactly at
layer 0. Layer 0's experts then run under different dispatch, so layer 1's logits differ by 5.5e-6, and the drift
grows to 0.024 by layer 10. There, in pass 0, the top-8 swaps expert 75 for expert 101. Their router scores are
0.36285 and 0.36124 in one schedule and 0.36282 and 0.36376 in the other, with 8th/9th margins of 0.0016 and 0.0009. Through the loop's three passes the row's
top-8 then differs at 32 of 48 slots. Over rows 0-895, 18.9% of row-slots change their top-8 between the two schedules.
The median 8th/9th margin at those changes is 0.0047, against 0.0096 over all row-slots.

**(a) Against the oMLX oracle** (3,015 rows, 16 histories; `schedule/oracle/`). The gate's runner prefills in uniform
512-row chunks and the host in 1,024-row ones. So `tf-inkling2-prefix-check` also captured every history under each
host's served schedule and under uniform 1,024-row chunks (`schedule/oracle/jobs.txt`). Each capture was then compared
with `omlx_gate.py compare`. Only `harbour` (1,336 tokens) and `cells_code` (1,178) change schedule. The other 14
histories run as L-1 rows plus the last token alone on both sides.

| Capture | top-1 | top-1 at margin ≥ 0.1 | mean KL | row-weighted KL | max KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate, uniform 512 (`inkling-omlx-oracle`; lane = main = D2) | 2,863 (94.96%) | 2,750/2,872 (95.75%) | 0.0194994 | 0.0639952 | 14.25 |
| main's served schedule | 2,865 (95.02%) | 2,753/2,872 (95.86%) | 0.0186238 | 0.0605144 | 8.42 |
| lane's served schedule | 2,861 (94.89%) | 2,753/2,872 (95.86%) | 0.0186412 | 0.0606687 | 8.42 |
| uniform 1,024 (the served chunk) | 2,858 (94.79%) | 2,750/2,872 (95.75%) | 0.0191658 | 0.0622099 | 9.70 |

The lane's served schedule is 1.001× main's mean KL and 0.13 points under its top-1, inside Decision 20's 1.10× and 0.5
points. Lane against main directly: top-1 on 3,001 of 3,015 rows (99.54%), 2,884/2,891 (99.76%) at margin ≥ 0.1, mean KL
0.000255.

**(b) Lane against main on the held-out probe** (1,024 held-out code tokens, every row's logits,
`schedule/probe-spread.txt`). The probe uses main's reuse-on schedule (768, 128, 127, 1) as the reference, and main's
reuse-off schedule (1,024) shows the spread main already has. Reference margin ≥ 0.1:

| Against main reuse-on | top-1 | top-1 at margin ≥ 0.1 | mean KL | max KL | max abs Δlogit |
| --- | ---: | ---: | ---: | ---: | ---: |
| lane (1,023, 1) | 99.71% | 1,008/1,009 (99.90%) | 0.00235 | 0.198 | 14.88 |
| main reuse-off (1,024), the reference spread | 99.71% | 1,008/1,009 (99.90%) | 0.00235 | 0.198 | 14.88 |

Lane against main reuse-off: top-1 100.00%, 1,006/1,006 at margin ≥ 0.1, mean KL 1.3e-6, max |Δlogit| 1.19. Moving the
last token into its own chunk is the lane's only change against reuse-off, and it is the smallest step here.

**(c) Quality dev** (oQ4e, `quality/`; main is 53a3b8894's build). The lane scores 8/11 and main 7/11: code 3/3
against 2/3, reasoning 3/3 on both, retrieval 2/2 on both, structured 0/3 on both. Structured output is unsupported,
so those requests get HTTP 400 on both sides. Main's replies equal its P4 record (`main-3881b3d8c`) on all 11 cases.
Against main, the lane's replies are byte-equal on 6 of 11 cases. Those are the three reasoning cases, whose prompts
of 71-75 tokens keep main's schedule, and the three structured refusals. The five that differ are the code cases
(136-175 tokens; main ran 128 + rest + 1 there, the lane runs rest + 1) and the two retrieval cases (4,633 and 5,270
tokens). The lane loses no case and runs into no cap. It gains `code-c02`: main thinks to the 8,192-token cap with no
code block, while the lane finishes in 1,735 tokens and its tests pass. Both retrieval answers match main's
(`QX-7419`, `VN-6403`) after 5,402 and 4,344 tokens, against main's 5,345 and 4,790. `code-c02` is Q3's
reasoning loop. Escaping it here comes from the changed logits, not from a fix.

## Served curves

Six interleaved process pairs over two sessions, main first in session 1 (ABBAAB) and lane first in session 2
(BAABBA). Main is `a1984516d` (V3 merged). The lane is `a1984516d` plus this item's code (`git diff 53a3b8894
22b8fec3b -- zig/`, which applies cleanly), so each pair isolates P5c (`curves/binaries.txt`). Each trial passed
the gates before and after. serve_curve's own record, taken right after load, flags compressor or swap activity in
six of the fourteen trials (`valid` false). P4 saw the same post-load flag. The ratios below therefore come from
interleaved pairs. Client tok/s, each trial's median at a length; ratios per request within a pair
(`curvetable.py`, `curvepair.py`):

| Prompt | main | lane | lane, reuse off | lane/main median (range) | geomean | 2,000 met |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1K dev (1,027-1,033) | 1,390-1,451 | 1,353-1,857 | 1,930 | 1.26 (0.92-1.34) | 1.17 | no |
| 1K held-out (1,003-1,019) | 1,048-1,103 | 1,510-1,859 | 1,795-1,803 | 1.60 (1.13-1.72) | 1.56 | no |
| 4K | 1,453-1,598 | 1,522-1,826 | 1,786-1,806 | 1.14 (0.97-1.23) | 1.12 | no |
| 8K | 1,382-1,627 | 1,586-1,720 | 1,705-1,719 | 1.06 (0.97-1.18) | 1.06 | no |
| 16K | 1,411-1,529 | 1,428-1,591 | 1,589-1,592 | 1.04 (0.98-1.06) | 1.03 | no |
| 31.7K | 1,335-1,461 | 1,428-1,498 | 1,500-1,501 | 1.02 (0.98-1.13) | 1.03 | no |

Over every curve request, the six pairs' geometric means are 1.086, 1.151, 1.228, 1.233, 1.140 and 1.218: 1.175×
across pairs, 95% interval 1.109-1.245. That meets Decision 11's promotion rule (at least 3%, interval above 1.0, six
pairs over two sessions). Both sources gain. Each long length has one dev and one held-out prompt; on them the
per-pair medians run 0.99-1.06 for dev and 0.98-1.14 for held-out.

Held-out 1K gains most because main gave it two small chunks before the last token, against one for dev (schedule table
above). With reuse on, the lane runs as fast as with reuse off from 4K up and on held-out 1K. At dev 1K it gives up
4-30% to reuse off: those prompts end in a 2-8-row chunk and the lone last token, where reuse off ends in one 1,027-1,033-row
chunk. Steady server-side prefill on the instrumented lane is 0.52-0.60 s for every 1K prompt (`curves/first-*`), 1,740-1,980
tok/s, against main's 0.65-0.97 s. Client tok/s adds the HTTP round trip. The lane's two lowest dev 1K medians come from
its first two trials. There the requests right after a slow first request ran 10-60% slower than the same requests in
the trace trials; see the next section. No length reaches 2,000 tok/s. The best are 1,859 at held-out 1K, 1,857 at dev
1K and 1,826 at 4K.

## First request and the warm-up at load

The first request is reported apart from the curve (1,258 tokens; server-side prefill seconds from the trace lines
of instrumented builds, `patch_trace.py`). Its extra cost sits in its first 1,024-row chunk, outside the GPU's
execution time. GPU time is 0.46 s on every build; the wall time runs higher. Every later chunk of the same request,
and every chunk of later requests, runs at steady speed. The small chunks were not the cost.

| Build | First request, s | First chunk wall / GPU, s | Load (spawn to ready), s | Footprint ready / peak, GB |
| --- | --- | --- | --- | --- |
| main | 3.82, 1.24, 2.62, 1.42 (traces); 2.44-2.71 (six curves) | 3.38/0.46, 0.79/0.47, 2.17/0.47, 0.97/0.46 | 5.8-26.6 | 17.41 / 18.98 |
| lane | 4.90, 2.41, 2.51, 2.93 (traces); 1.14-2.54 (six curves) | 4.60/0.46, 2.12/0.46, 2.22/0.47, 2.63/0.46 | 5.7-28.9 | 17.41-17.52 / 18.98 |
| lane, session buffers zero-filled at load | 2.60, 2.56, 2.04 | 2.31/0.46, 2.28/0.46, 1.74/0.46 | 7.3-25.4 | 18.95 / 18.98 |
| lane, warm-up at load (committed) | 1.06, 1.03 | 0.77/0.46, 0.74/0.46 | 9.2, 7.4 | 18.97 / 18.98 |

Wall minus GPU in the first chunk ranges from 0.3 to 4.1 s on main and the lane. It tracks the host's memory state after
load rather than the build or the schedule: serve_curve's post-load record shows the compressor at 11,000-196,000 pages/s
in the slow trials. It behaves like Metal readying a buffer on the first command buffer that uses it: the first
chunk pays for the 16.8 GB of weights and the slot's caches, more when memory pressure has moved their pages. Zeroing
the session buffers at load moves part of it. The warm-up moves most of it: one token prefilled through slot 0 at load,
then slot 0's caches, rollback and snapshots zero-filled. In two trials it cut the first request to 1.03-1.06 s, with
0.28-0.31 s left in the first chunk. The warm-up's own trace line gives 0.75 and 1.81 s inside load, which takes
6-29 s here anyway. So the cost moves into load rather than vanishing.

The probes ran a scratch variant that zero-filled every slot. At `--parallel 1` that is the same as `335ea7f43`.
The committed warm-up touches slot 0 only, so what it commits does not grow with the slot count. The footprints below
and the reply check come from a build of `335ea7f43`'s host:

| Server | Plain lane, ready footprint | Warm lane, ready footprint |
| --- | ---: | ---: |
| `--parallel 1 --context 32768` (served curve) | 17.41 GB | 18.97 GB |
| `--parallel 8 --context 4096` (B2's sessions) | 17.42 GB | 18.05 GB |

Peak footprint is 18.98 GB on every build at `--parallel 1`. The admission math already counts every slot's caches
under `INKLING_LOAD_LIMIT_GB`, and the warm-up commits one slot's share early, so load stays inside the limit. It leaves
no state behind. A new served test (`inkling host: the warm-up at load leaves the first slot's caches, rollback and
snapshots zero and stores nothing`) reads back every buffer of slot 0 and finds it zero, with empty length, prompt and
fork points and an empty store. With prompt cache 0, four fixed-prompt 48-token greedy replies are byte-equal between
builds with and without the warm-up (`first/replycheck.json`, `replycheck.py`).

## Measurement conditions and method

- Builds, each `git archive` of a tree and `zig build native -Dcpu=apple_m1 --prefix`:
  - quality dev and the schedule analyses use main `53a3b8894` and the lane worktree;
  - the served curves use main `a1984516d` and `a1984516d` with `git diff 53a3b8894 22b8fec3b -- zig/` applied;
  - the first-request trials use those two trees patched by `patch_trace.py`, plus two scratch variants of the lane:
    session buffers zero-filled at load, and the warm-up that `335ea7f43` commits.
  Quality runs put the pinned `libtfgrammar.dylib` beside each binary. No build ran inside a timed window.
  `trial.sh` keeps any trial that sees a `zig`, `clang` or `metal` process as invalid, and `probe.sh` any that sees
  `zig` or `clang`.
- Every timed trial ran under `campaign/heavy.py` (28 GB cap, `INKLING_LOAD_LIMIT_GB=26`, `--context 32768`) and the
  timing gates: `gpu-gate-ready -1` with 60% limits and `perf-gate-ready` at 4 busy cores before the trial, and
  `perf-gate-ready -1` after it. Two trials (`s2-lane-f`, `probe-main-b`) waited out a busy pre-trial gate;
  none tripped after starting.
- Served curve: `comparators/serve_curve.py tensorfold`, one fresh server a trial, prompt cache 0. The first request
  (1,258 tokens) is reported alone. Then come six unseen 1K prompts, one from each dev and held-out corpus file, and
  two each at 4K, 8K, 16K and 31,744; each long length has one dev and one held-out prompt. Requests pair across trials
  by kind, length and rep, which fix the corpus span. Ratios are per request, lane over main, within interleaved pairs
  (`curvepair.py`).

## Reproduction

Environment for every command: `PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp"
ZIG_GLOBAL_CACHE_DIR="$PWD/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26` and
`MODEL=~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd`. The scripts named here are in `inkling-p5c-chunks-runs/`, which keeps
each server and suite `.log` as `.txt` because the repository ignores `*.log`.

```sh
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu,inkling-omlx-oracle
python3 tools/zig/lean_check.py > head.txt; python3 tools/zig/lean_new.py base.txt head.txt   # 255 problems, 256 on the base, 0 new (335ea7f43)
# builds, one prefix a side; no compile while a trial runs
git archive 53a3b8894 | tar -x -C .work/tmp/p5c/main-src
(cd .work/tmp/p5c/main-src && zig build native tf-inkling2-loop-run -Dcpu=apple_m1 --prefix "$PWD/../main-build")
zig build native tf-inkling2-loop-run tf-inkling2-prefix-check -Dcpu=apple_m1 --prefix .work/tmp/p5c/lane-build
# schedules and the oracle: every row's logits under a chunk schedule, then compared with the oracle
python3 campaign/kernels/inkling-p5c-chunks-runs/schedules.py 1258 1019 4058
python3 campaign/heavy.py -- tf-inkling2-prefix-check "$MODEL" ids.json 768,128,127,1 main-on.bin
python3 spread.py main-on.bin lane.bin; sh oracle-compare.sh
# the routing flip, on main's runner
python3 campaign/heavy.py -- tf-inkling2-loop-run "$MODEL" ids.json prefillroutes 1024 routes/c1024 1023
python3 flip.py logits-c1024.bin logits-c896.bin routes/c1024 routes/c896 896
# quality dev, each side, then the per-case comparison
python3 campaign/heavy.py -- make -C campaign quality-inkling2-mini-oq4e SET=dev BASELINE_WT="$PWD" BIN=<build>/native/bin/tensorfold-native
python3 qualcmp.py <main result.json> <lane result.json>
# served curves: one gated trial a fresh server, interleaved; first-request traces from builds patched by patch_trace.py
sh session.sh                                  # trial.sh <name> <build>/native/bin/tensorfold-native [--lengths 1024]
python3 curvetable.py; python3 curvepair.py <main.jsonl> <lane.jsonl> ...; python3 tracesum.py <name>.server.txt
# first request with load time and footprint, then the warm-up's state and reply checks
sh probes.sh                                   # probe.sh <name> <build>/native/bin/tensorfold-native, via firstprobe.py
python3 replycheck.py replycheck.json <plain lane>/tensorfold-native <warm lane>/tensorfold-native
```
