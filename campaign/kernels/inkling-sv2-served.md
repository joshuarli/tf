# Inkling 2 Mini integrated served path (SV2)

**Verdict.** Speculation is promoted to the served default. On six prompts that no fitted prior has seen, three
served sessions give speculative decode 1.43 to 2.43 times plain decode. Every prompt's lower 95% bound is above 1,
and every reply is byte-equal to plain decode: cold, prompt-cache hit, forked second turn and natural EOS. The
drafter costs 0.42 to 0.56 GB at ready and 0.81 GB at peak, inside Decision 8's 4 GiB. Unseen prefill with the drafter
is 0.978 to 1.006 times plain from 1K to the cap here (SC2's two later sessions put it at 0.965 to 0.968 times at 1K to 8K).
Correction (SC2): the `idle` mode measured nothing. `--no-drafts` drops the drafter at load, so `idle` served plain
decode, and its 1.00 to 1.04 times `ar` is two plain servers compared.
Serve now pairs Inkling 2 Mini with `incoai/Inkling-2-Mini-DFlash2` from the local Hugging Face cache, as it does
Qwen3.8-27B's drafter. The planner's cost table is refitted to V4's in-round costs, which is neutral (0.998 times,
95% interval 0.961 to 1.036).

Decision 17's 200 tok/s is still met only on code-1024 of the dev prompts (V4's 230). On the qualify prompts the best
is 172 tok/s (prose-1024) and the 4K prompts reach 101 to 113 tok/s.

## Base and builds

Lane `lane/sv2-serve` from main `ec925cb41` (V4). Every served session ran one binary built at `17b9cdd33` (the
refitted table, `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`, sha256 `57023dcf149c…`), with the drafter
named through `INKLING_DRAFTER` as before the default change. The round pairs ran `tf-inkling2-loop-run` from
`ec925cb41` (main, sha256 `3f371160c64b…`) and `17b9cdd33` (lane, `138e7e1e949d…`). Target
`Inkling-2-Mini-oQ4e-mtp@3eea042751cd`; drafter `Inkling-2-Mini-DFlash2@4f9088ea66c2`, prepared q4.

## The planner's cost table

`verify.Plan.default_verify_ms` was V3's. V4 made every width 1.5 to 2.2 ms cheaper, so V3's table priced one row at
18.3 ms against 16.2 ms measured in rounds, and 16 rows at 58.1 ms against 61.4 ms. The refit (`rounds.py costtable`)
takes:

- widths 1 to 8 and 16 from V4's six lane processes' tree rounds (3,870 rounds; widths seen 20 times or more);
- widths 9, 10 and 12 from two sustained widths passes on main (`widths-s`, `widths-s2`: 20 s of 8-row load, 9 reps,
  1K and 4K context), scaled by the in-round to sustained ratio of their measured neighbours;
- widths 11 and 13 to 15 from the line between, made non-decreasing.

| Rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 | 18.3 | 22.3 | 25.9 | 29.2 | 31.1 | 33.7 | 36.5 | 38.1 | 42.7 | 45.4 | 47.9 | 50.4 | 52.4 | 54.3 | 56.2 | 58.1 |
| SV2 | 16.2 | 20.7 | 25.0 | 27.9 | 31.1 | 33.5 | 36.7 | 38.7 | 43.9 | 47.6 | 50.4 | 53.2 | 55.2 | 57.3 | 59.3 | 61.4 |

The draft is priced at 4.9 ms, the mean over 3,498 drafted rounds (V3: 4.5).

Six interleaved process pairs (main, lane, lane, main and so on), tree rounds, 512 tokens on V4's six dev prompts:

| Prompt | Committed/round V3 table | SV2 table | Spec tok/s V3 table | SV2 table | Geomean |
| --- | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 10.02 | 10.02 | 230.0 | 232.9 | 1.01 |
| prose-1024 | 4.56 | 4.48 | 116.0 | 116.8 | 0.99 |
| structured-1024 | 4.06 | 3.79 | 105.9 | 105.9 | 0.98 |
| code-4096 | 7.51 | 7.51 | 162.8 | 163.2 | 0.99 |
| prose-4096 | 3.57 | 3.45 | 94.7 | 92.8 | 0.99 |
| structured-4096 | 3.52 | 3.45 | 95.9 | 97.4 | 1.02 |

Process pairs: geometric mean 0.998, 95% interval 0.961 to 1.036 (n = 6). Each build's six processes made the same
rounds and tokens, and every reply equals plain greedy. The cheaper small widths make the planner choose slightly
smaller trees (0.07 to 0.27 fewer committed tokens a round on prose and structured) at no speed change. The refit is
kept because the planner now prices widths at what they cost. The policy is frozen here: no qualify prompt fitted
anything.

## The qualify prompts

DF2b's acceptance priors were fitted on the holdout prompts, which makes them calibration data. `build.py --qualify`
appends a third set to `campaign/prompts/manifest.json` without rebuilding dev or holdout. It has one 1K and one 4K
prompt for each of code, prose and structured, built from source files that no dev or holdout prompt cites. Their
served Flint lengths are confirmed (`build.py --check`: 24 prompts verified), and a corpus test checks that their
sources are disjoint from the other sets.

## Served sessions

`run.py session` starts three servers in turn, in an order rotated each session:

- `ar`: no drafter, `--no-drafts`;
- `idle`: `INKLING_DRAFTER` set, `--no-drafts` (which drops the drafter at load, so this is plain decode too);
- `spec`: `INKLING_DRAFTER` set, drafting.

Each server runs every qualify prompt through `bench.trial`: a cold 512-token request, a warm repeat (a prompt-cache
hit of the whole prompt) and a natural-termination request. Then a second turn forks the first reply's cached state,
and `prefill_curve.py` prefills unseen prompts at 1K, 4K, 16K and 31,744 tokens (two each, one-token replies). Each
trial waits until the host passes its gate, because a fresh load leaves the memory compressor busy.

| Session | Mode order | Trials (UTC) |
| --- | --- | --- |
| a | ar, idle, spec | 11:47 to 12:01 |
| b | idle, spec, ar | 12:03 to 12:19 |
| c | spec, ar, idle | 12:42 to 12:58 (after a 20 min rest) |

Session a's 32,768-token decode trials failed on the context cap (a full cell leaves no room for a reply) and are
replaced by the prefill curve from session b on. Two decode trials failed the compressor gate and are left out
(a's structured-1024 spec, c's prose-4096 idle). Sessions a and b ran back to back on one host state.

### Decode

Cold 512-token requests, median over the sessions. Ratios are per session (`spec` against `ar` in the same session),
geometric mean and 95% t interval.

| Prompt | ar tok/s | idle tok/s | spec tok/s | spec/ar (95%) | idle/ar (95%) | Committed/round |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| qualify-code-1024 | 72.5 | 72.6 | 148.3 | 2.000 (1.787–2.237) | 1.002 (1.002–1.002) | 4.74 |
| qualify-prose-1024 | 70.8 | 72.5 | 172.4 | 2.429 (2.331–2.531) | 1.019 (0.970–1.071) | 6.40 |
| qualify-structured-1024 | 71.9 | 72.2 | 127.3 | 1.765 (1.328–2.348) | 1.007 (0.917–1.107) | 3.88 |
| qualify-code-4096 | 69.6 | 69.7 | 101.4 | 1.431 (1.210–1.692) | 0.998 (0.938–1.063) | 3.28 |
| qualify-prose-4096 | 69.2 | 70.2 | 104.8 | 1.530 (1.400–1.673) | 1.035 (0.719–1.489) | 3.37 |
| qualify-structured-4096 | 66.9 | 69.3 | 112.9 | 1.690 (1.540–1.853) | 1.042 (0.956–1.137) | 3.76 |

Warm repeats (whole-prompt cache hits) keep the ordering: spec 90 to 169 tok/s against ar 63 to 71.

### Unseen prefill

Client tok/s, median over sessions b and c (four prompts a length a mode):

| Unseen prompt tokens | ar | idle | spec | spec/ar |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1,959 | 1,886 | 1,915 | 0.978 |
| 4,096 | 2,052 | 2,048 | 2,020 | 0.985 |
| 16,384 | 1,883 | 1,884 | 1,879 | 0.998 |
| 31,744 | 1,757 | 1,752 | 1,768 | 1.006 |

The first request after readiness took 0.98 to 1.05 s in every mode.

### Footprint

The largest peak over the decode trials was 21.13 GB plain and 21.94 GB speculative (0.81 GB). Ready footprint was
18.98 GB plain and 19.39 to 19.53 GB with the drafter loaded (0.42 to 0.56 GB). The drafter reports 0.636 GiB
prepared q4 at one lane. Both fit Decision 8's 4 GiB draft residency and the 28 GB cap.

### Correctness

`run.py equal session-a session-b session-c` compares every reply against `ar` in the same session. For every valid
trial, the cold, warm and natural replies, the forked turn and its cached tokens are equal, and only `spec` drafts.
The prompts never reach EOS inside 512 tokens, so `run.py eos` checks natural EOS separately with three short prompts
on each mode's server. They stop after 66, 64 and 128 tokens, every mode's reply equals `ar`'s, and only `spec`
drafts (73, 55 and 155 drafted tokens).

## The default

The rule was that speculation becomes the default only if it beats plain decode on every class and stays within
4 GiB of draft residency. It does both, so serve's known-drafter table (`zig/src/server/drafter.zig`) now names
`incoai/Inkling-2-Mini-DFlash2` for `inkling2_loop` with text hidden 2,304 and 24 layers. The Metal registry passes
`--drafter` to Inkling, and the engine takes its drafter from `INKLING_DRAFTER` or `--drafter`, refusing both by name.
Contract change: an Inkling 2 Mini checkpoint served without `--no-drafts` or `--drafter none` speculates when the
drafter is in the local Hugging Face cache. Serve never downloads it. Checked on the built binary, with the drafter
behind a stand-in cache directory:

- with no flags, serve finds the drafter and drafts;
- with `INKLING_DRAFTER` also set, the environment's drafter serves and the found one is dropped with a log line;
- `--no-drafts` drafts nothing;
- an explicit `--drafter` drafts;
- `--drafter` with `INKLING_DRAFTER` is refused.

Every reply equals plain decode's. The campaign's servers are unchanged: they name the pinned drafter through
`INKLING_DRAFTER` and pass `--no-drafts` for plain decode, and this host's Hugging Face cache holds no Inkling drafter.

## Verification

- `test-drafter-host` (new step): 32/32. Moving the Inkling entry to 25 layers fails it.
- `test-native`: 196/200. The one failure, `draft sources: Metal Qwen27 copies and DFlash2 use actual slot and head
  facts`, fails identically on main `748c36835`. Removing the both-sources refusal fails the new Inkling test.
- Integrated suite on `33b38ed43` (main `748c36835` merged into the lane): `inkling-host` ok, `inkling-gpu` 185/185,
  oMLX oracle pass. The served step's name filter dropped the renamed drafter test (12 to 11); after adding it back,
  `test-inkling2-loop-served` passes 12/12, which gives 186 tests.
- `test-inkling2-loop-host`: 24/24. `test-inkling2-loop-filtered` for `plan` 7/7 and `tree` 7/7;
  `test-inkling2-loop-dflash2` 11/11 (all with the model directories set).
- `test_evidence.py`: the new qualify corpus test passes. `ExistingRecords` (62 malformed records) fails identically
  on main.

## Reproduction

From the worktree, with `D=campaign/kernels/inkling-sv2-served-runs`, `V=campaign/kernels/inkling-v4-fixed-runs`,
`M` and `L` the main and lane `tf-inkling2-loop-run`, and `B` the served binary:

```sh
for n in widths-s widths-s2; do
  python3 -I $D/rounds.py widths $M $V/prompts.json $n --reps 9 --load 20 --cells 0,1,2,4,8,9,10,12,16; done
python3 -I $D/rounds.py costtable ../inkling-v4-fixed-runs/lane-a,../inkling-v4-fixed-runs/lane-b,../inkling-v4-fixed-runs/lane-c,../inkling-v4-fixed-runs/lane-d,../inkling-v4-fixed-runs/lane-e,../inkling-v4-fixed-runs/lane-f \
  --widths widths-s,widths-s2
for run in "$M main-a" "$L lane-a" "$L lane-b" "$M main-b" "$M main-c" "$L lane-c" \
           "$L lane-d" "$M main-d" "$M main-e" "$L lane-e" "$L lane-f" "$M main-f"; do
  set -- $run; python3 -I $D/rounds.py bench $1 $V/prompts.json $2 --drafts tree --tokens 512; done
python3 -I $D/rounds.py pairs main-a,main-b,main-c,main-d,main-e,main-f lane-a,lane-b,lane-c,lane-d,lane-e,lane-f
python3 campaign/heavy.py -- python3 -I campaign/prompts/build.py --qualify --measure --binary $B
for s in "session-a 0" "session-b 1" "session-c 2"; do set -- $s
  python3 -I $D/run.py session --bin $B --name $1 --rotate $2; done
python3 -I $D/run.py eos --bin $B --name eos-a
python3 -I $D/run.py equal session-a session-b session-c
python3 -I $D/run.py table session-a session-b session-c
```

## Open

- **Decision 17's decode target** is met only on code-1024 (V4's dev prompt, 230 tok/s). The qualify prompts give
  101 to 172 tok/s. The verify past one row is the routed experts (V4's open item), and the 4K prompts commit 3.3 to
  3.8 tokens a round.
- **Separate sessions on separate host states.** Sessions a and b ran back to back; SC2 re-pairs on a quiet host.
- **Multi-lane serving** still speculates only while one lane decodes, so concurrent aggregate is unchanged.
