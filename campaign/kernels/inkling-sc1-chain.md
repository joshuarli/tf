# Inkling 2 Mini: the next draft behind the verify (SC1)

**Verdict.** A speculative round is GPU-bound. A round of 35 to 41 ms spends 30.6 to 35.2 ms in the verify's command
buffer and 3.2 to 4.3 ms in the drafter's. The host holds the GPU idle for 1.0 to 1.3 ms a round, 2.7 to 3.3% of it.
The boundary this item targets, from the verify's GPU end to the next draft's GPU start, is 0.20 to 0.25 ms of that.
Most of it is the fixed cost of one host round trip: about 0.06 ms for the host to wake from the verify's wait and
0.11 to 0.14 ms from the drafter's first commit to its GPU start. The host's own work there is 0.04 ms, and the
drafter frame never waits for its encoder (0.00 to 0.08 ms idle inside it). A chained draft can therefore recover at
most 0.18 to 0.29 ms a round, 0.5 to 0.8%. The chained draft built here drafted the same trees as unchained drafting
on all 302 rounds compared, but its own gap from the verify's end to its GPU start was 0.22 to 0.38 ms, so it was
reverted. The lane keeps the round-split instrumentation and a bit-identical selector speedup. No prompt meets a
Decision 17 budget, and removing every host gap would not change that: the verify's and the drafter's GPU time are
the levers.

**Base.** Main is `b77ed8fd0`, whose Zig equals the DF2b merge `53a3b8894`. The lane is `lane/sc1-chain`, its code
final at `6c4e53305`. The chained draft is `4716d11f6`, `2c49d4489` and `4e44a9870`, reverted in `89f479658`.

## Where a round's time goes

`INKLING_ROUND_SPLIT=1` makes the runner's speculator record each tree round on two clocks without adding a wait or
splitting a command buffer: host times around the draft, the plan and the verify, and each command buffer's GPU start
and end, read once it has completed (`dflash_spec.Split`, `runtime_frame.Times`). The records are printed after the
run, outside its rounds. The verify's GPU end is its host return less the drafter's own wake latency in that round
(its last buffer's end to the host's return). The verify's commit kernels (`i2v_round_kv`, `i2v_round_conv`) run in
its command buffer, so their time is inside the verify's GPU time. DF2b's Metal 4 timeline put every kernel outside
the forward pass, the commit's included, at 0.13 to 0.15 ms a round.

Unchained rounds of main's design (`attr-plain-a`: the six dev prompts, 128 greedy tokens, drafter q4, lookup tree
rounds). Mean ms a round over every round; select and encode are over rounds that ran the drafter. Gap A runs from
the drafter's GPU end to the verify's GPU start, gap B from the verify's GPU end to the next draft's GPU start. Host
is every GPU-idle ms of the round. `python3 -I run.py split attr-plain-a`:

| Prompt | Rounds | Round | Drafter GPU | Drafter idle | Verify GPU | Gap A | Selector | Verify encode and submit | Gap B | Other host | Host | Host share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 24 | 36.65 | 3.25 | 0.00 | 32.43 | 0.73 | 0.16 | 0.58 | 0.18 | 0.07 | 0.98 | 2.7% |
| prose-1024 | 33 | 35.51 | 3.91 | 0.02 | 30.57 | 0.81 | 0.18 | 0.56 | 0.20 | 0.01 | 1.04 | 2.9% |
| structured-1024 | 25 | 40.29 | 3.97 | 0.00 | 35.12 | 0.92 | 0.21 | 0.68 | 0.24 | 0.05 | 1.21 | 3.0% |
| code-4096 | 51 | 37.35 | 4.25 | 0.08 | 31.87 | 0.92 | 0.23 | 0.64 | 0.21 | 0.02 | 1.23 | 3.3% |
| prose-4096 | 37 | 40.62 | 4.16 | 0.04 | 35.17 | 0.99 | 0.26 | 0.72 | 0.22 | 0.05 | 1.30 | 3.2% |
| structured-4096 | 35 | 39.17 | 4.21 | 0.03 | 33.75 | 0.95 | 0.24 | 0.68 | 0.20 | 0.03 | 1.21 | 3.1% |

- **The verify's own CPU encode is the largest host gap.** `prefill.zig`'s `run` encodes the whole grouped verify,
  then commits it and waits, so the GPU stays idle for 0.56 to 0.72 ms a round while it is encoded. That includes
  the 0.11 to 0.14 ms from commit to GPU start.
- **The selector takes 0.16 to 0.26 ms**, from the drafter's return to the tree; the plan's `build` takes 0.004 ms.
- **The drafter frame keeps its GPU busy.** It commits after 6 ops and then every 40, and the host encodes faster than
  the GPU runs them: 0.00 to 0.08 ms of idle inside the frame.

Gap B's parts, in rounds whose next round drafts:

| Prompt | Wake from the verify | Host work | Draft commit to GPU start | Gap B |
| --- | ---: | ---: | ---: | ---: |
| code-1024 | 0.056 | 0.038 | 0.110 | 0.204 |
| prose-1024 | 0.056 | 0.037 | 0.112 | 0.205 |
| structured-1024 | 0.074 | 0.041 | 0.139 | 0.254 |
| code-4096 | 0.064 | 0.034 | 0.123 | 0.221 |
| prose-4096 | 0.067 | 0.040 | 0.134 | 0.240 |
| structured-4096 | 0.063 | 0.034 | 0.113 | 0.210 |

The host's work between the verify and the next draft is the path, the plan's learning, the tap rows into the
drafter's ring, the emitted tokens, the lookup and six encoded drafter ops: 0.03 to 0.04 ms. The rest is a round
trip's fixed latency, which only a draft committed before the verify completes avoids.

## What a chained draft can recover

At most gap B and the drafter's idle span: 0.18 to 0.29 ms a round, 0.5 to 0.8%. The drafter's 3.2 to 4.3 ms is GPU
work that a chain still runs after the verify on the same queue. The megakernel-zoo reference (`mk_draft.py`,
`_cyc_pure`) chains by running the verify and the next draft as one compiled MLX graph with one host sync. tf's
verify already runs its argmax, tree acceptance and commit on the GPU behind one wait (DF2b), so a tf round's host
boundaries are its two waits.

## The chained draft, reverted

Design (`4716d11f6`, `2c49d4489`, `4e44a9870`, all in the shared DFlash2 runtime and the runner's speculator):

- A whole frame (`runtime_frame`: `seal`, `submit`, `land`) encodes the next round's draft onto the drafter's context
  while the GPU runs the current one, committing nothing.
- `chain.zig` and `chain.metal` absorb up to 16 path rows ahead. The path's tap rows are gathered on the GPU from the
  target's tap buffer, and keys and values reach the ring only below the path's length.
- The block's anchor, rope positions, mask and context offsets sit in buffers the host fills once the verify returns.
  The embed, rope and attention callbacks read them while a follow plan is set.
- A round whose lookup chain leaves the drafter out, or a drafter behind on taps, drafted as before.

It was exact:

- Trees were identical on every one of 302 rounds: `attr-chain-a` against `attr-plain-a` (205 rounds) and
  `fixture-chain-a` against `fixture-plain-a` (97 rounds on the test fixtures at 101 tokens). The comparison covers
  drafts, parents, probabilities, chain places, accepted counts and positions.
- Masked keys past the context add nothing in either attention kernel. The lane projection keeps every row's sums at
  every width, so a 16-row absorb gives each kept row the bits of an L-row one.
- With chaining on by default, the DFlash2 tests passed: real-prompt tree speculation, forced trees, and lookup rounds
  before the drafter's window. The tree test reproduced DF2b's counts exactly: 260 rounds, 4 past eight rows and 10
  later branches.

It recovered nothing (`python3 -I run.py split attr-chain-a`):

| Prompt | Gap B unchained | Gap B chained | Commit to GPU start, chained | Drafter GPU unchained | chained |
| --- | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 0.204 | 0.290 | 0.179 | 3.25 | 3.38 |
| prose-1024 | 0.205 | 0.265 | 0.187 | 3.91 | 4.21 |
| structured-1024 | 0.254 | 0.224 | 0.146 | 3.97 | 4.12 |
| code-4096 | 0.221 | 0.293 | 0.207 | 4.25 | 4.30 |
| prose-4096 | 0.240 | 0.296 | 0.212 | 4.16 | 4.24 |
| structured-4096 | 0.210 | 0.382 | 0.284 | 4.21 | 4.07 |

A whole frame committed at once took 0.15 to 0.28 ms to start on the GPU, against 0.11 to 0.14 ms for the unchained
frame's six-op first part. Its frame also ran 0.1 to 0.3 ms longer at 1K, where it absorbs 16 rows instead of the
path's own. The pair is one ungated process each, so the frame difference is not a measurement, but neither
direction leaves room for a gain. The chain added about 400 lines to the shared runtime and was reverted in `89f479658`.

A draft committed before the verify completes would remove the wake and the commit-to-start latency (about 0.2 ms a
round). That needs a hook in `prefill.zig`'s `run` between the verify's commit and its wait, which this lane may not
add, and a follow kernel computing the block's inputs from `i2v_accept`'s result on the GPU. At 0.5% it was not
pursued.

## What the lane keeps

- **The round split** (`c66de7b4a`, renamed `INKLING_ROUND_SPLIT` in `89f479658` after the family's other switches).
  A frame reads its buffers' GPU times only while a `Times` record is set, so Flint and unrecorded rounds read no
  clock.
- **The selector's weighted parent row** (`0a9fc969e`). An expansion multiplies the parent's predecessor row by the
  depth's projection once. Each candidate sums that row times its successor row, `(p * x) * s` in rank order, which
  is the old expression's rounding. A CPU test in `selector.zig` holds every edge of a random codebook bit-equal to
  the old product, so trees, and Flint's and Inkling's drafts, are unchanged. Its gain is at most 0.1 ms a round:
  the selector's span, lattice read included, was 0.20 to 0.30 ms before (0.51 on prose-1024 under contention) and
  0.19 to 0.21 ms after (`attr-base-a`, `attr-sel-a`, both contended processes).

## What remains to the Decision 17 budgets

Decision 17 allows 5 ms a committed token. On these rounds (`attr-plain-a`):

| Prompt | Committed a round | Budget | Round | Over budget | Host | GPU (drafter and verify) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 5.29 | 26.5 | 36.65 | 10.2 | 0.98 | 35.67 |
| prose-1024 | 3.85 | 19.2 | 35.51 | 16.3 | 1.04 | 34.48 |
| structured-1024 | 5.08 | 25.4 | 40.29 | 14.9 | 1.21 | 39.09 |
| code-4096 | 2.49 | 12.5 | 37.35 | 24.9 | 1.23 | 36.12 |
| prose-4096 | 3.43 | 17.2 | 40.62 | 23.5 | 1.30 | 39.32 |
| structured-4096 | 3.63 | 18.1 | 39.17 | 21.1 | 1.21 | 37.96 |

Every round is 10 to 25 ms over its budget and spends at most 1.3 ms on the host. DF2b's 512-token rounds tell the
same story: 37.0 to 49.9 ms against budgets of 14.6 to 37.6 ms. In order of size, the levers are:

1. The verify's GPU time (V3).
2. Tokens committed a round, which set the budget.
3. The drafter's GPU time: its head reads the 8-bit unembedding over seven rows every round.
4. The verify's 0.6 ms CPU encode, which could overlap the drafter only if its dispatches stopped depending on the
   tree's rows.

## Correctness

The lane's code changes nothing a round computes: the split only reads clocks, and the selector test holds every
edge bit-equal. On the lane's final commit `6c4e53305`:

- `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exits 0 (`.work/suite/20261124T163752`).
  `inkling-host` passes 10/10 steps and 35/35 tests. `inkling-gpu` passes 17/17 steps and 180/180 tests with none
  skipped: main's 179 and the plain-scratch test. Every speculative reply equals plain greedy, with its caches,
  through forced zero, partial and full acceptance, a limit and an EOS inside a block or a tree, lookup rounds, and
  the served host's prompt reuse, fork, two lanes and EOS inside a lookup round. Peak charged memory was 21.6 GB.
- `zig build test-flint-draft-host` passes 47/47 steps and 87/87 tests. `zig build test-qwen27` passes 48/48 steps and
  201/206 tests with 5 skipped; main has 200/205, and the new one is the selector's bit-identity test. The selector's
  trees are unchanged, so Flint's DFlash2 output is too.
- One earlier `test-qwen27` run failed in `safetensors_copied_test` ("close drops cache after unmapping": expected 0,
  found 2). It passed on rerun and passes on main; it is host page-cache state, not this lane.
- `tools/zig/lean_new.py` against `b77ed8fd0`: 256 problems on both, 0 new.

## Conditions

- **Machine.** Apple M5 Max (40 GPU cores, 48 GB), macOS 27.2.
- **Lock and gates.** Every model load and device test went through `campaign/heavy.py` with
  `INKLING_LOAD_LIMIT_GB=26`; peak charged memory was 19.9 to 20.6 GB. The attribution processes are untimed: they ran
  without the admission gates while other lanes were active, so their absolute times are not measurements. Rounds and
  committed tokens are deterministic and equal in every process.
- **Pairs waived.** The parent waived Decision 11's timed pairs for this lane: its kept change moves host time inside
  the noise, and the split above is attribution, not a speedup claim.
- **Builds.** `tf-inkling2-loop-run`, `-Dcpu=apple_m1`:
  - `4b59c0c2…` at `c66de7b4a` (main with the split) ran `attr-base-a`.
  - `190f356b…` at `0a9fc969e` (with the selector change) ran `attr-sel-a`.
  - `d3aeacdb…` at `2c49d4489` (the chain) ran `attr-chain-a` and `attr-plain-a` (chaining off) and the fixture runs.
  - `00caaf08…` at `89f479658` is the lane's.

## Reproduction

From the worktree, with `D=campaign/kernels/inkling-sc1-chain-runs`, `T=~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd`
and `R` a runner built at the commit named above:

```sh
INKLING_ROUND_SPLIT=1 INKLING_DRAFT_CHAIN=0 python3 campaign/heavy.py -- $R $T $D/prompts.json dflash \
  ~/models/Inkling-2-Mini-DFlash2@4f9088ea66c2 128 $D/attr-plain-a.jsonl 4 entering incremental tree \
  > $D/attr-plain-a.log 2>&1
python3 -I $D/run.py split attr-plain-a
```

The `.log` files are not tracked. Their `inkling2 split` and `dflash` lines are, as `NAME.txt`, which `split` reads
when the log is absent. `INKLING_DRAFT_CHAIN` exists only in the chain's builds (`2c49d4489`), where chaining is on
unless it is `0`. The builds at `c66de7b4a` and `0a9fc969e` read the split switch as `TF_INKLING_ROUND_SPLIT`.
`run.py bench … --split` runs the same processes behind the admission gates.

## Plain greedy across scratches

A chain-era test chose an EOS token from plain greedy on the speculator's 16-row verify scratch after speculation.
That run disagreed by token 45 with plain greedy on a fresh one-row scratch on the 37-token "code" fixture. A state
leak in the verify scratch would break the determinism rules, so `dflash_test.zig` now holds it ("plain greedy steps
keep their logits …"). The test hashes each of 95 greedy steps' logits in four runs and compares each with the first:

1. a fresh one-row `BatchScratch`;
2. the speculator's verify scratch, never used;
3. the same verify scratch after 97 tree rounds of speculation on all four fixtures;
4. a fresh one-row scratch again.

All four agree on every step's logits bit for bit, so neither the scratch nor speculation changes plain greedy, and
repeated runs agree. The run ran unchained speculation; the chain-era disagreement is not explained. The code read
for it found no carrier: on one row, every kernel is chosen by the row count and reads only rows written in the same
step, and the convolutions skip history before position 0.

## Open problems

- **The verify's encode.** 0.6 ms a round of GPU idle is in `prefill.zig`'s `run` (P5c's and V3's files).
- **The chained draft, if revived.** Its draft-side machinery is in `4716d11f6`..`4e44a9870`. A GPU-side version needs
  a hook between the verify's commit and its wait. Rerun the plain-scratch test with chaining on before trusting it.
