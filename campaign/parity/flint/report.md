# Flint reference parity

Teacher-forced logits from TensorFold's Metal decode kernels against `rec_mlx` 0.1.0 at
`e53663b262af4cb1ca498e596ca8b1a2d79e0800` (`recursive-oss/rec`, the oracle whose rounding
TensorFold's Flint kernels are documented to match; Apache-2.0). Both engines run the same token ids
on the same checkpoint and every position is compared.

## Setup

- Checkpoint: `RecursiveSI/flint-int4@3b93f9053b530488405ae9fca51d4900d6ce21f0`.
- Tokenizer sha256: `a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68`; chat template sha256: `7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466`.
- Corpus: 16 held-out histories, 6069 tokens, 911223936 logit positions per side. Corpus sha256: `1ba105265c727efde78943b642e0f93471d994c1f349c518ef6107cbf4760fd4`.
- Render: checkpoint chat_template.jinja, add_generation_prompt=True, assistant reasoning_content rendered as <think>.
- Oracle: rec_mlx 0.1.0 at e53663b262af4cb1ca498e596ca8b1a2d79e0800 (recursive-oss/rec, Apache-2.0), resident decode one token at a time
- TensorFold: `tf-flint-run <model> <ids> logits`, bf16 logits in 16-row windows from position 0, `zig build tf-flint-run -Dcpu=apple_m1`.
- Oracle licence: Apache-2.0 (`recursive-oss/rec`). TensorFold is Apache-2.0.
- Tokenization: TensorFold's Zig tokenizer reproduces the checkpoint tokenizer's ids on every rendered history before either engine runs (0 disagreements).
- The oracle's own residency claim is checked on the same machine: a resident decode step equals the native one-token forward bit for bit (row 10 of linked-list-reversal, the equivalence asserted in tests/validate_resident.py, bit-identical).

## Error distribution

Absolute error is over every logit position and vocabulary entry; relative error divides by the larger
of the two magnitudes (floored at 1e-3). KL is `KL(oracle || TensorFold)` in float64 nats, one value
per position. Margins are each side's own top-1 minus top-2 logit.

| Quantity | p0 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 | p100 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Absolute logit error | +0 | +0 | +0 | +0.125 | +0.25 | +0.5 | +1 | +2.25 | +6.75 | +31 | +0.35515 |
| Relative logit error | +0 | +0 | +0 | +0.004329 | +0.0076923 | +0.014778 | +0.035874 | +0.077922 | +0.22791 | +429.45 | +0.012171 |
| KL (nats) | +3.569e-11 | +7.7395e-09 | +1.0126e-07 | +7.5171e-05 | +0.0025106 | +0.0096122 | +0.053219 | +0.31119 | +1.7017 | +7.2926 | +0.019052 |

| Metric | Value |
| --- | ---: |
| Top-1 agreement | 95.518% (5797/6069) |
| Positions with both top-2 margins >= 0.1 | 5667 (93.4%) |
| Top-1 agreement there | 98.218% |
| TensorFold top-2 margin p1 / p50 / p99 | 0.02768 / 1.973 / 16.09 |
| Oracle top-2 margin p1 / p50 / p99 | 0.02219 / 1.972 / 16.11 |
| Positions below the 0.1 margin (TensorFold / oracle) | 255 / 241 |

## Per history

| History | Kind | Tokens | KL mean | KL p99 | Absolute mean | Absolute p99 | Top-1 | Decisive top-1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linked-list-reversal` | code | 175 | 0.0099 | 0.1412 | 0.3578 | 2.5000 | 93.14% | 96.95% |
| `json-invoice-records` | structured | 423 | 0.0136 | 0.2526 | 0.4522 | 2.3750 | 97.87% | 99.03% |
| `tool-call-test-selection` | structured | 281 | 0.1334 | 1.8694 | 0.7829 | 6.6250 | 90.39% | 93.21% |
| `zig-slice-bounds` | code | 212 | 0.0172 | 0.3723 | 0.4116 | 2.2500 | 94.81% | 98.45% |
| `rust-borrow-tangle` | prose | 236 | 0.0133 | 0.1231 | 0.3269 | 2.5000 | 96.61% | 98.65% |
| `retry-backoff-analysis` | prose | 236 | 0.0201 | 0.2119 | 0.3475 | 1.8750 | 91.95% | 97.12% |
| `observability-plan` | prose | 240 | 0.0157 | 0.1229 | 0.3200 | 1.5000 | 89.17% | 96.70% |
| `token-bucket-ts` | code | 241 | 0.0060 | 0.0516 | 0.3001 | 1.7500 | 97.10% | 100.00% |
| `binary-search-edges` | code | 253 | 0.0248 | 0.1838 | 0.2885 | 2.0000 | 96.44% | 98.33% |
| `long-module-scheduler` | code | 401 | 0.0101 | 0.1089 | 0.2932 | 1.7500 | 96.51% | 99.20% |
| `transformer-attention-math` | prose | 243 | 0.0132 | 0.1600 | 0.3587 | 2.0000 | 94.65% | 98.60% |
| `dijkstra-walkthrough` | code | 701 | 0.0090 | 0.0803 | 0.2720 | 1.7500 | 96.86% | 99.09% |
| `rollout-postmortem` | prose | 592 | 0.0142 | 0.1440 | 0.3455 | 1.8750 | 94.76% | 97.98% |
| `arrow-etl-pipeline` | code | 507 | 0.0133 | 0.1343 | 0.3503 | 2.3750 | 96.65% | 98.35% |
| `debug-session-nan-loss` | code | 1060 | 0.0141 | 0.1524 | 0.3020 | 2.0000 | 96.32% | 98.38% |
| `changelog-structured` | structured | 268 | 0.0149 | 0.2419 | 0.4302 | 2.2500 | 97.01% | 98.45% |

## The oracle's own spread on the same tokens

`oracle chunked native prefill` is rec_mlx's other teacher-forced path. rec_mlx groups a prefill's sums by chunk, and its own `tests/validate_prefix_rollback.py` records that a different grouping changes BF16 rounding, so this pair measures the spread that belongs to accumulation order alone -- on exactly the tokens TensorFold was compared on.

| Pair | Mean KL (nats) | Mean abs error | Abs p99 | Top-1 | Decisive top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorFold vs oracle | 0.019052 | 0.355153 | 2.2500 | 95.518% | 98.218% |
| oracle chunked native prefill vs oracle | 0.023038 | 0.352461 | 2.3750 | 95.222% | 98.180% |

TensorFold's distance from the oracle's decode path is smaller than the oracle's own chunked prefill's distance from it, on the same token ids. That bounds how much of the residual can be attributed to TensorFold: a difference this size is what two accumulation orders already produce inside rec_mlx.

The largest per-history mean KL is `tool-call-test-selection` at 0.1334 nats over its 281 tokens.

## Where the residual is

Absolute error by absolute position, maximum over the vocabulary: the worst band is the history's first
row, which has no context at all, so the residual is not a mask, window or ring artefact. If it were, it
would grow with position and peak at the 513-token boundaries rather than at position 0.

| Position band | Rows | Mean per-row max abs |
| --- | ---: | ---: |
| [0, 1) | 16 | 4.047 |
| [1, 4) | 48 | 2.057 |
| [4, 16) | 192 | 2.325 |
| [16, 64) | 768 | 2.090 |
| [64, 128) | 1024 | 1.814 |
| [128, 256) | 1836 | 1.686 |
| [256, 512) | 1368 | 1.538 |
| [512, 513) | 3 | 3.250 |
| [513, 1000000000) | 814 | 1.306 |

Row 0 mean per-row max abs error: 4.047; every row: 1.705.

The classification below compares TensorFold's per-row error with the oracle's own per-row error
between its two paths. A row is `stable` when any path's per-row maximum is at most 0.5 logit.

- Rows stable for the oracle against itself: 197 of 6069; their mean TensorFold per-row maximum is 0.6777.
- Rows unstable for the oracle against itself: their mean TensorFold per-row maximum is 1.7392, 2.57x larger.
- Rows whose stability classification disagrees between the two pairs: 234 of 6069.

So TensorFold's error concentrates on the rows where rec_mlx itself disagrees between its two
accumulation orders. It is not zero on the rows rec_mlx reproduces exactly, but it is well below the
oracle's own spread on the rows where the oracle is unstable, which is the shape of an accumulation-order
difference rather than a structural one.

Per-row maximum absolute error, the whole corpus: TensorFold against the oracle, and the
oracle against its own other path. The two distributions are the same size.

| Pair | p50 | p75 | p95 | p99 | p100 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TensorFold vs oracle | 1.250 | 1.750 | 4.000 | 9.290 | 31.000 | 1.705 |
| Oracle vs itself | 1.250 | 1.875 | 4.000 | 9.540 | 52.000 | 1.733 |

TensorFold's per-row error is at most the oracle's own per-row error on 3696 of 6069 rows (60.9%), and the two per-row error series correlate at 0.702.

## Disagreeing positions

272 of 6069 positions have a different top-1 (4.48%). 171 of those have a top-2 margin below 0.1 on at least one side, which is a near-tie rather than a different distribution.

The split matters more than the total. rec_mlx's two paths choose a different top-1 on 4.78% of positions by themselves, so a single oracle row is not a well-defined target on those positions. Comparing TensorFold with the oracle's other path as well:

- Top-1 agreement against the oracle's chunked prefill: 95.683%.
- Of the 272 disagreements with the decode path, 137 pick exactly the token the oracle's other path picks.
- On 163 of them the oracle's own two paths disagree, so no single oracle answer exists to agree with.
- Only 109 positions are disagreements where both oracle paths agree and TensorFold is alone.

The 20 disagreements with the largest margin on both sides, where the margin makes a miss least
excusable:

| TensorFold top-1 | Oracle top-1 | TensorFold margin | Oracle margin | min margin |
| ---: | ---: | ---: | ---: | ---: |
| 29800 | 25 | 4.6875 | 1.5000 | 1.5000 |
| 332 | 6722 | 1.0312 | 1.5781 | 1.0312 |
| 3817 | 34548 | 1.0938 | 1.0000 | 1.0000 |
| 25 | 284 | 1.0625 | 0.8750 | 0.8750 |
| 85859 | 16 | 0.8125 | 1.0000 | 0.8125 |
| 24 | 4920 | 1.0000 | 0.7812 | 0.7812 |
| 43200 | 269 | 0.7812 | 0.5938 | 0.5938 |
| 489 | 1387 | 0.5000 | 0.8750 | 0.5000 |
| 92 | 284 | 2.0312 | 0.5000 | 0.5000 |
| 2579 | 461 | 0.5000 | 0.8750 | 0.5000 |
| 13207 | 12766 | 2.5156 | 0.5000 | 0.5000 |
| 1166 | 1129 | 0.4844 | 0.5781 | 0.4844 |
| 794 | 7700 | 0.8125 | 0.4375 | 0.4375 |
| 838 | 5317 | 0.4375 | 0.5000 | 0.4375 |
| 1464 | 1647 | 0.4062 | 0.8125 | 0.4062 |
| 43087 | 45806 | 0.3906 | 0.4219 | 0.3906 |
| 5406 | 3957 | 0.6875 | 0.3750 | 0.3750 |
| 11981 | 602 | 0.3750 | 0.4375 | 0.3750 |
| 23842 | 630 | 0.4922 | 0.3594 | 0.3594 |
| 11 | 311 | 0.3516 | 0.4570 | 0.3516 |

## First divergent tensors

Every history with a top-1 disagreement appears below. The row is that history's largest-KL row.
`tf.bin`, `oracle.bin` and `oracle-other.bin` hold that row's bf16 bits in `.work/parity/flint/divergent/`
for a re-check. The `oracle self` columns are the same row under rec_mlx's other path: when the oracle
disagrees with itself at that row's top-1, the miss is not TensorFold's alone.

| History | Row | Global | Max abs | Oracle self max abs | Oracle top-1 | Self top-1 | TF top-1 | TF agrees with self |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `linked-list-reversal` | 114 | 114 | 1.7500 | 1.2500 | 311 | 311 | 11 | no |
| `json-invoice-records` | 32 | 207 | 16.0000 | 3.2500 | 110874 | 1943 | 1648 | no |
| `tool-call-test-selection` | 82 | 680 | 10.1875 | 10.2500 | 25 | 29800 | 29800 | yes |
| `zig-slice-bounds` | 200 | 1079 | 4.5000 | 2.5000 | 423 | 423 | 27105 | no |
| `rust-borrow-tangle` | 223 | 1314 | 2.5000 | 1.8750 | 15994 | 15468 | 15468 | yes |
| `retry-backoff-analysis` | 43 | 1370 | 9.8750 | 7.6250 | 1647 | 1464 | 1464 | yes |
| `observability-plan` | 215 | 1778 | 2.0000 | 1.6250 | 7044 | 7044 | 4106 | no |
| `token-bucket-ts` | 181 | 1984 | 1.2500 | 0.7500 | 597 | 439 | 439 | yes |
| `binary-search-edges` | 3 | 2047 | 18.6250 | 1.1875 | 6722 | 6722 | 332 | no |
| `long-module-scheduler` | 15 | 2312 | 3.7500 | 5.2500 | 368 | 43087 | 43087 | yes |
| `transformer-attention-math` | 19 | 2717 | 3.3750 | 2.1250 | 19489 | 19489 | 113617 | no |
| `dijkstra-walkthrough` | 688 | 3629 | 4.2500 | 2.3750 | 269 | 269 | 260 | no |
| `rollout-postmortem` | 33 | 3675 | 7.0000 | 10.7500 | 45806 | 45806 | 43087 | no |
| `arrow-etl-pipeline` | 2 | 4236 | 25.2500 | 9.1250 | 764 | 764 | 8916 | no |
| `debug-session-nan-loss` | 297 | 5038 | 14.0000 | 12.8750 | 12766 | 13207 | 13207 | yes |
| `changelog-structured` | 32 | 5833 | 3.5000 | 3.0000 | 617 | 617 | 3068 | no |

In 7 of 16 outliers the oracle itself chooses a
different top-1 under its other path, so those are differences between two accumulation orders on the
same checkpoint and the same ids, not a TensorFold defect. The rows that remain are traced to their first
divergent tensor in the next section.

## First divergent tensors, traced stage by stage

Every corpus row whose per-row max abs logit error is more than 2x the oracle's own
two-path max abs error on that same row -- 141 rows -- is traced below to its embedding, together with
the row the brief names that measures just under that threshold here (`zig-slice-bounds` row 200). This is the check
the plan asks for before a difference may be called floating-point noise: the final logits alone do not say
which tensor first departs.

Instrumentation, on the exact path each side was measured on:

- TensorFold: `tf-flint-run <model> <ids> dump <rows>,<out>` re-runs the row's own 16-row window with the
  layer loop cut at 0, 1, ..., 49 layers and reads the forward's existing scratch after each cutoff. No
  product code under `zig/src/` changed. It records the residual after each layer, the layer's attention
  output (routed layers) or dense MLP output (dense layers 0 and 1, whose down projection reuses the same
  scratch buffer that held the attention output), the router logits and top-k, the per-pick expert outputs, the shared
  output, the final norm and the logits. Each dumped row's logits equal the compared `.bin` bits on 142 of 142 rows.
- Oracle: both teacher-forced paths of the checkpoint's own modules (`oracle_trace.py`): one token at a
  time from position 0, and one native prefill chunk, both reading and writing the same BF16 KV storage.
  Each path's kept row is checked against the logits that path produced in the parity run: 142 of 142 decode rows and 142 of 142 chunked rows are bit-identical.

A stage's oracle spread is `max |chunked - decode|`; TensorFold's difference is `max |tf - decode|`. The
first stage where TensorFold's difference exceeds the oracle's own is the first divergent tensor. A router
selection change is a near-tie when the k-th and (k+1)-th logits at that row are closer than twice the
router logits' own path difference -- both endpoints can move by that difference -- and a clear-margin
change otherwise. `material` is the first stage where TensorFold's difference is more than four times the
spread or 0.01 logit, which separates the bf16-ulp departures from the change that actually grows the error.

### One-row steps against 16-row windows

`tf-flint-run <model> <prompt> exact <continuation>` feeds the prompt in chunks, then the continuation as
one-row steps and as 16-row windows, and compares every row's logits bit for bit:

- `debug-session-nan-loss`, prompt 996 tokens, continuation 64 tokens: flint exact at 996: 0 of 64 rows' logits differ between one-row steps and windows
- `json-invoice-records`, prompt 359 tokens, continuation 64 tokens: flint exact at 359: 0 of 64 rows' logits differ between one-row steps and windows

Verdicts: 142 rows accumulation order, 0 defect.

### The outlier rows named in the brief

| History | Row | TF max abs | Oracle self max abs | First departure | First selection change | Material departure | Verdict |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `binary-search-edges` | 3 | 18.6250 | 1.1875 | residual, layer 1: 0.0009766 vs 0.0006104 | L24: gap 0.01562 vs router delta 0.125; oracle [148] instead of tf [125] | residual, L48: 13.38 vs 0.75 | accumulation order |
| `arrow-etl-pipeline` | 2 | 25.2500 | 9.1250 | dense mlp output, layer 0: 0.007812 vs 0.003906 | L9: gap 0.01562 vs router delta 0.06641; oracle [237] instead of tf [239] | shared expert output, L42: 0.03674 vs 0.006104 | accumulation order |
| `json-invoice-records` | 32 | 16.0000 | 3.2500 | attention output, layer 2: 0.001953 vs 0.001831 | L22: gap 0.02734 vs router delta 0.125; oracle [185] instead of tf [121] | router logits, L23: 0.375 vs 0.07812 | accumulation order |
| `zig-slice-bounds` | 200 | 4.5000 | 2.5000 | dense mlp output, layer 0: 0.003906 vs 0.00293 | L13: gap 0.03125 vs router delta 0.09375; oracle [82] instead of tf [198] | residual, L24: 0.0625 vs 0.006836 | accumulation order |

- `binary-search-edges` row 3, 253 tokens: the first departure is residual at layer 1 (0.0009766 vs the oracle's own 0.0006104); the earliest of 10 expert selection changes is at layer 24 across a boundary logit gap of 0.01562, not clear of the router logits' own path difference of 0.125; the state becomes materially different at layer 48 (13.38 vs 0.75).
  Largest residual departure: layer 48 at 13.38 vs the oracle's own 0.75.
- `arrow-etl-pipeline` row 2, 507 tokens: the first departure is mlp output at layer 0 (0.007812 vs the oracle's own 0.003906); the earliest of 12 expert selection changes is at layer 9 across a boundary logit gap of 0.01562, not clear of the router logits' own path difference of 0.06641; the state becomes materially different at layer 42 (0.03674 vs 0.006104).
  Largest residual departure: layer 48 at 161.5 vs the oracle's own 11.
- `json-invoice-records` row 32, 423 tokens: the first departure is attention output at layer 2 (0.001953 vs the oracle's own 0.001831); the earliest of 14 expert selection changes is at layer 22 across a boundary logit gap of 0.02734, not clear of the router logits' own path difference of 0.125; the state becomes materially different at layer 23 (0.375 vs 0.07812).
  Largest residual departure: layer 47 at 27 vs the oracle's own 4.25.
- `zig-slice-bounds` row 200, 212 tokens: the first departure is mlp output at layer 0 (0.003906 vs the oracle's own 0.00293); the earliest of 10 expert selection changes is at layer 13 across a boundary logit gap of 0.03125, not clear of the router logits' own path difference of 0.09375; the state becomes materially different at layer 24 (0.0625 vs 0.006836).
  Largest residual departure: layer 24 at 0.0625 vs the oracle's own 0.006836.

### Every traced row

| History | Row | TF max abs | Oracle self max abs | Ratio | First divergent tensor | Verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `dijkstra-walkthrough` | 173 | 11.6250 | 0.5000 | 23.25 | dense mlp output, layer 1: 0.003052 vs 0.002808 | accumulation order |
| `binary-search-edges` | 3 | 18.6250 | 1.1875 | 15.68 | residual, layer 1: 0.0009766 vs 0.0006104 | accumulation order |
| `debug-session-nan-loss` | 306 | 31.0000 | 2.0000 | 15.50 | dense mlp output, layer 1: 0.006409 vs 0.005737 | accumulation order |
| `dijkstra-walkthrough` | 377 | 14.1250 | 1.2500 | 11.30 | residual, layer 0: 0.001953 vs 0.001862 | accumulation order |
| `json-invoice-records` | 368 | 8.5000 | 1.0000 | 8.50 | router logits, layer 2: 0.0625 vs 0.04688 | accumulation order |
| `zig-slice-bounds` | 9 | 10.0000 | 1.3125 | 7.62 | residual, layer 1: 0.0009766 vs 0.0007324 | accumulation order |
| `retry-backoff-analysis` | 137 | 9.0000 | 1.2500 | 7.20 | shared expert output, layer 2: 0.0003967 vs 0.0002594 | accumulation order |
| `zig-slice-bounds` | 27 | 12.5000 | 2.0000 | 6.25 | shared expert output, layer 2: 0.0003548 vs 0.0003395 | accumulation order |
| `binary-search-edges` | 237 | 12.3750 | 2.0000 | 6.19 | dense mlp output, layer 1: 0.008301 vs 0.007935 | accumulation order |
| `long-module-scheduler` | 332 | 5.8750 | 1.0000 | 5.88 | residual, layer 0: 0.001953 vs 0.001465 | accumulation order |
| `debug-session-nan-loss` | 746 | 7.2500 | 1.3750 | 5.27 | dense mlp output, layer 1: 0.004639 vs 0.004395 | accumulation order |
| `arrow-etl-pipeline` | 43 | 5.2500 | 1.0000 | 5.25 | expert output, layer 2: 0.0002098 vs 0.0001984 | accumulation order |
| `transformer-attention-math` | 46 | 6.3750 | 1.2500 | 5.10 | residual, layer 0: 0.001465 vs 0.001221 | accumulation order |
| `retry-backoff-analysis` | 8 | 14.0000 | 2.7500 | 5.09 | residual, layer 1: 0.00293 vs 0.0009766 | accumulation order |
| `token-bucket-ts` | 8 | 3.7500 | 0.7500 | 5.00 | expert output, layer 2: 0.0001907 vs 0.0001755 | accumulation order |
| `json-invoice-records` | 32 | 16.0000 | 3.2500 | 4.92 | attention output, layer 2: 0.001953 vs 0.001831 | accumulation order |
| `rollout-postmortem` | 512 | 4.7500 | 1.0000 | 4.75 | dense mlp output, layer 1: 0.002319 vs 0.002014 | accumulation order |
| `token-bucket-ts` | 31 | 7.0000 | 1.5000 | 4.67 | dense mlp output, layer 1: 0.001312 vs 0.001213 | accumulation order |
| `debug-session-nan-loss` | 739 | 4.0000 | 0.8750 | 4.57 | expert output, layer 2: 0.0002899 vs 0.0002518 | accumulation order |
| `dijkstra-walkthrough` | 411 | 2.7500 | 0.6250 | 4.40 | dense mlp output, layer 0: 0.003174 vs 0.00293 | accumulation order |
| `transformer-attention-math` | 25 | 5.3750 | 1.2500 | 4.30 | dense mlp output, layer 1: 0.0007706 vs 0.0007172 | accumulation order |
| `rollout-postmortem` | 7 | 12.0000 | 2.8750 | 4.17 | expert output, layer 2: 0.000267 vs 0.0002365 | accumulation order |
| `tool-call-test-selection` | 39 | 6.2500 | 1.5000 | 4.17 | attention output, layer 3: 0.0002441 vs 0.0001831 | accumulation order |
| `transformer-attention-math` | 186 | 7.0000 | 1.7500 | 4.00 | shared expert output, layer 2: 0.0003967 vs 0.0003891 | accumulation order |
| `dijkstra-walkthrough` | 677 | 5.5000 | 1.3750 | 4.00 | residual, layer 1: 0.0007324 vs 0.0006104 | accumulation order |
| `linked-list-reversal` | 35 | 10.3750 | 2.6250 | 3.95 | residual, layer 0: 0.007812 vs 0.003906 | accumulation order |
| `changelog-structured` | 200 | 4.7500 | 1.2500 | 3.80 | expert output, layer 5: 0.0002136 vs 0.0001526 | accumulation order |
| `linked-list-reversal` | 160 | 6.6250 | 1.7500 | 3.79 | dense mlp output, layer 1: 0.001709 vs 0.001648 | accumulation order |
| `debug-session-nan-loss` | 575 | 2.2500 | 0.6250 | 3.60 | dense mlp output, layer 0: 0.002533 vs 0.002441 | accumulation order |
| `json-invoice-records` | 261 | 5.2500 | 1.5000 | 3.50 | dense mlp output, layer 1: 0.004013 vs 0.003891 | accumulation order |
| `rollout-postmortem` | 337 | 5.2500 | 1.5000 | 3.50 | residual, layer 1: 0.0004883 vs 0.0004272 | accumulation order |
| `debug-session-nan-loss` | 90 | 7.7500 | 2.2500 | 3.44 | dense mlp output, layer 0: 0.003418 vs 0.003174 | accumulation order |
| `json-invoice-records` | 323 | 6.0000 | 1.7500 | 3.43 | router logits, layer 2: 0.125 vs 0.09375 | accumulation order |
| `json-invoice-records` | 57 | 3.3750 | 1.0000 | 3.38 | residual, layer 1: 0.0009766 vs 0.0006714 | accumulation order |
| `dijkstra-walkthrough` | 511 | 1.2500 | 0.3750 | 3.33 | dense mlp output, layer 0: 0.001709 vs 0.001587 | accumulation order |
| `rollout-postmortem` | 586 | 2.5000 | 0.7500 | 3.33 | residual, layer 1: 0.001953 vs 0.0009766 | accumulation order |
| `debug-session-nan-loss` | 764 | 2.5000 | 0.7500 | 3.33 | shared expert output, layer 2: 0.0006104 vs 0.0004272 | accumulation order |
| `rust-borrow-tangle` | 226 | 5.7500 | 1.7500 | 3.29 | attention output, layer 2: 0.001221 vs 0.0009766 | accumulation order |
| `dijkstra-walkthrough` | 646 | 3.6250 | 1.1250 | 3.22 | shared expert output, layer 3: 0.0001841 vs 0.0001678 | accumulation order |
| `arrow-etl-pipeline` | 504 | 7.2500 | 2.2500 | 3.22 | residual, layer 1: 0.003906 vs 0.001953 | accumulation order |
| `rust-borrow-tangle` | 227 | 2.3750 | 0.7500 | 3.17 | residual, layer 1: 0.00293 vs 0.0004883 | accumulation order |
| `arrow-etl-pipeline` | 364 | 2.7500 | 0.8750 | 3.14 | residual, layer 1: 0.00293 vs 0.0009766 | accumulation order |
| `linked-list-reversal` | 159 | 11.7500 | 3.7500 | 3.13 | dense mlp output, layer 0: 0.001709 vs 0.001587 | accumulation order |
| `debug-session-nan-loss` | 152 | 3.1250 | 1.0000 | 3.12 | residual, layer 2: 0.0003662 vs 0.0003204 | accumulation order |
| `rust-borrow-tangle` | 203 | 6.1250 | 2.0000 | 3.06 | attention output, layer 2: 0.0009499 vs 0.0009155 | accumulation order |
| `json-invoice-records` | 285 | 6.0000 | 2.0000 | 3.00 | attention output, layer 2: 0.0004044 vs 0.0003662 | accumulation order |
| `tool-call-test-selection` | 32 | 3.0000 | 1.0000 | 3.00 | router logits, layer 5: 0.0625 vs 0.04688 | accumulation order |
| `transformer-attention-math` | 237 | 1.8750 | 0.6250 | 3.00 | dense mlp output, layer 1: 0.004639 vs 0.004395 | accumulation order |
| `debug-session-nan-loss` | 86 | 9.0000 | 3.0000 | 3.00 | attention output, layer 2: 0.0008545 vs 0.0007629 | accumulation order |
| `tool-call-test-selection` | 73 | 5.8750 | 2.0000 | 2.94 | dense mlp output, layer 0: 0.01562 vs 0.01172 | accumulation order |
| `linked-list-reversal` | 29 | 9.8750 | 3.3750 | 2.93 | residual, layer 3: 0.0004883 vs 0.0002441 | accumulation order |
| `observability-plan` | 76 | 4.3750 | 1.5000 | 2.92 | dense mlp output, layer 1: 0.001022 vs 0.0009766 | accumulation order |
| `arrow-etl-pipeline` | 449 | 1.4375 | 0.5000 | 2.88 | residual, layer 2: 0.0003662 vs 0.0003052 | accumulation order |
| `token-bucket-ts` | 237 | 5.0000 | 1.7500 | 2.86 | dense mlp output, layer 1: 0.001213 vs 0.001175 | accumulation order |
| `transformer-attention-math` | 13 | 5.0000 | 1.7500 | 2.86 | dense mlp output, layer 1: 0.001846 vs 0.001221 | accumulation order |
| `json-invoice-records` | 189 | 2.1250 | 0.7500 | 2.83 | dense mlp output, layer 1: 0.002441 vs 0.001709 | accumulation order |
| `dijkstra-walkthrough` | 23 | 9.3750 | 3.3125 | 2.83 | expert output, layer 2: 0.0001373 vs 0.0001183 | accumulation order |
| `debug-session-nan-loss` | 747 | 1.7500 | 0.6250 | 2.80 | dense mlp output, layer 0: 0.007812 vs 0.002075 | accumulation order |
| `arrow-etl-pipeline` | 2 | 25.2500 | 9.1250 | 2.77 | dense mlp output, layer 0: 0.007812 vs 0.003906 | accumulation order |
| `retry-backoff-analysis` | 47 | 2.7500 | 1.0000 | 2.75 | dense mlp output, layer 1: 0.001659 vs 0.001282 | accumulation order |
| `dijkstra-walkthrough` | 141 | 1.3750 | 0.5000 | 2.75 | attention output, layer 2: 0.001404 vs 0.00116 | accumulation order |
| `rollout-postmortem` | 399 | 2.3750 | 0.8750 | 2.71 | dense mlp output, layer 1: 0.004028 vs 0.003235 | accumulation order |
| `debug-session-nan-loss` | 567 | 2.3750 | 0.8750 | 2.71 | dense mlp output, layer 1: 0.001465 vs 0.0009766 | accumulation order |
| `zig-slice-bounds` | 131 | 4.0000 | 1.5000 | 2.67 | dense mlp output, layer 1: 0.0007935 vs 0.0007401 | accumulation order |
| `rust-borrow-tangle` | 213 | 2.0000 | 0.7500 | 2.67 | residual, layer 1: 0.001953 vs 0.001465 | accumulation order |
| `rollout-postmortem` | 170 | 2.0000 | 0.7500 | 2.67 | dense mlp output, layer 1: 0.002014 vs 0.001709 | accumulation order |
| `rollout-postmortem` | 378 | 2.0000 | 0.7500 | 2.67 | dense mlp output, layer 0: 0.01562 vs 0.007812 | accumulation order |
| `dijkstra-walkthrough` | 452 | 4.6250 | 1.7500 | 2.64 | residual, layer 2: 0.0004883 vs 0.0003662 | accumulation order |
| `linked-list-reversal` | 1 | 1.3125 | 0.5000 | 2.62 | dense mlp output, layer 1: 0.002808 vs 0.002563 | accumulation order |
| `observability-plan` | 1 | 1.3125 | 0.5000 | 2.62 | dense mlp output, layer 1: 0.002808 vs 0.002563 | accumulation order |
| `zig-slice-bounds` | 2 | 4.2500 | 1.6250 | 2.62 | residual, layer 0: 0.003906 vs 0.0009766 | accumulation order |
| `debug-session-nan-loss` | 742 | 3.2500 | 1.2500 | 2.60 | dense mlp output, layer 1: 0.001953 vs 0.001556 | accumulation order |
| `zig-slice-bounds` | 55 | 2.2500 | 0.8750 | 2.57 | dense mlp output, layer 1: 0.001404 vs 0.0008392 | accumulation order |
| `dijkstra-walkthrough` | 293 | 2.2500 | 0.8750 | 2.57 | dense mlp output, layer 1: 0.0009766 vs 0.0006104 | accumulation order |
| `debug-session-nan-loss` | 81 | 9.0000 | 3.5000 | 2.57 | attention output, layer 2: 0.0003662 vs 0.0002289 | accumulation order |
| `json-invoice-records` | 273 | 5.7500 | 2.2500 | 2.56 | dense mlp output, layer 1: 0.005615 vs 0.004883 | accumulation order |
| `json-invoice-records` | 182 | 3.7500 | 1.5000 | 2.50 | router logits, layer 2: 0.1094 vs 0.09375 | accumulation order |
| `zig-slice-bounds` | 26 | 3.7500 | 1.5000 | 2.50 | residual, layer 0: 0.001953 vs 0.001465 | accumulation order |
| `retry-backoff-analysis` | 45 | 5.6250 | 2.2500 | 2.50 | expert output, layer 2: 0.0002899 vs 0.0002594 | accumulation order |
| `long-module-scheduler` | 194 | 2.5000 | 1.0000 | 2.50 | attention output, layer 4: 0.0001144 vs 0.0001106 | accumulation order |
| `long-module-scheduler` | 211 | 2.5000 | 1.0000 | 2.50 | dense mlp output, layer 1: 0.001648 vs 0.001038 | accumulation order |
| `transformer-attention-math` | 66 | 2.5000 | 1.0000 | 2.50 | dense mlp output, layer 1: 0.001162 vs 0.0009766 | accumulation order |
| `rollout-postmortem` | 250 | 1.8750 | 0.7500 | 2.50 | attention output, layer 2: 0.001953 vs 0.001831 | accumulation order |
| `rollout-postmortem` | 383 | 1.2500 | 0.5000 | 2.50 | expert output, layer 2: 0.0004272 vs 0.0003662 | accumulation order |
| `arrow-etl-pipeline` | 52 | 2.5000 | 1.0000 | 2.50 | shared expert output, layer 4: 0.0001831 vs 0.0001259 | accumulation order |
| `arrow-etl-pipeline` | 231 | 1.8750 | 0.7500 | 2.50 | dense mlp output, layer 1: 0.00293 vs 0.001709 | accumulation order |
| `debug-session-nan-loss` | 693 | 2.5000 | 1.0000 | 2.50 | dense mlp output, layer 1: 0.001831 vs 0.001099 | accumulation order |
| `changelog-structured` | 179 | 3.7500 | 1.5000 | 2.50 | residual, layer 1: 0.0009766 vs 0.0006104 | accumulation order |
| `long-module-scheduler` | 4 | 2.7500 | 1.1250 | 2.44 | expert output, layer 2: 0.0002022 vs 0.0001879 | accumulation order |
| `arrow-etl-pipeline` | 12 | 2.1250 | 0.8750 | 2.43 | residual, layer 2: 0.0002441 vs 0.0002136 | accumulation order |
| `dijkstra-walkthrough` | 403 | 1.5000 | 0.6250 | 2.40 | dense mlp output, layer 0: 0.002335 vs 0.002075 | accumulation order |
| `dijkstra-walkthrough` | 698 | 3.0000 | 1.2500 | 2.40 | attention output, layer 3: 0.0001526 vs 0.0001392 | accumulation order |
| `rollout-postmortem` | 301 | 1.5000 | 0.6250 | 2.40 | dense mlp output, layer 0: 0.007812 vs 0.001953 | accumulation order |
| `json-invoice-records` | 393 | 4.7500 | 2.0000 | 2.38 | attention output, layer 2: 0.0008545 vs 0.0006104 | accumulation order |
| `dijkstra-walkthrough` | 32 | 2.3750 | 1.0000 | 2.38 | router logits, layer 2: 0.08838 vs 0.07812 | accumulation order |
| `json-invoice-records` | 112 | 6.5000 | 2.7500 | 2.36 | attention output, layer 2: 0.0004883 vs 0.0004272 | accumulation order |
| `linked-list-reversal` | 27 | 1.7500 | 0.7500 | 2.33 | attention output, layer 2: 0.0004883 vs 0.0002441 | accumulation order |
| `linked-list-reversal` | 81 | 1.7500 | 0.7500 | 2.33 | expert output, layer 2: 0.0003052 vs 0.0002365 | accumulation order |
| `dijkstra-walkthrough` | 1 | 1.3125 | 0.5625 | 2.33 | dense mlp output, layer 1: 0.002808 vs 0.002319 | accumulation order |
| `dijkstra-walkthrough` | 386 | 0.8750 | 0.3750 | 2.33 | shared expert output, layer 3: 0.0001907 vs 0.0001698 | accumulation order |
| `rollout-postmortem` | 1 | 1.3125 | 0.5625 | 2.33 | dense mlp output, layer 1: 0.002808 vs 0.002319 | accumulation order |
| `rollout-postmortem` | 461 | 1.7500 | 0.7500 | 2.33 | router logits, layer 2: 0.07812 vs 0.0625 | accumulation order |
| `rollout-postmortem` | 467 | 1.7500 | 0.7500 | 2.33 | dense mlp output, layer 0: 0.001953 vs 0.001709 | accumulation order |
| `arrow-etl-pipeline` | 225 | 1.7500 | 0.7500 | 2.33 | attention output, layer 2: 0.0006104 vs 0.0005493 | accumulation order |
| `debug-session-nan-loss` | 1 | 1.3125 | 0.5625 | 2.33 | dense mlp output, layer 1: 0.002808 vs 0.002319 | accumulation order |
| `debug-session-nan-loss` | 461 | 1.7500 | 0.7500 | 2.33 | residual, layer 1: 0.0009155 vs 0.0008545 | accumulation order |
| `tool-call-test-selection` | 91 | 5.7500 | 2.5000 | 2.30 | attention output, layer 2: 0.0003185 vs 0.0003033 | accumulation order |
| `debug-session-nan-loss` | 429 | 2.8750 | 1.2500 | 2.30 | residual, layer 0: 0.0009766 vs 0.0008545 | accumulation order |
| `rust-borrow-tangle` | 206 | 2.0000 | 0.8750 | 2.29 | dense mlp output, layer 0: 0.007812 vs 0.002441 | accumulation order |
| `token-bucket-ts` | 3 | 2.0000 | 0.8750 | 2.29 | router logits, layer 2: 0.09375 vs 0.0625 | accumulation order |
| `long-module-scheduler` | 17 | 2.0000 | 0.8750 | 2.29 | dense mlp output, layer 0: 0.001953 vs 0.001709 | accumulation order |
| `transformer-attention-math` | 9 | 5.1250 | 2.2500 | 2.28 | dense mlp output, layer 0: 0.001953 vs 0.001923 | accumulation order |
| `dijkstra-walkthrough` | 659 | 5.3750 | 2.3750 | 2.26 | router logits, layer 2: 0.125 vs 0.09375 | accumulation order |
| `tool-call-test-selection` | 6 | 6.7500 | 3.0000 | 2.25 | residual, layer 2: 0.0004883 vs 0.0002441 | accumulation order |
| `binary-search-edges` | 102 | 1.1250 | 0.5000 | 2.25 | router logits, layer 2: 0.09375 vs 0.08594 | accumulation order |
| `binary-search-edges` | 227 | 1.1250 | 0.5000 | 2.25 | residual, layer 1: 0.001465 vs 0.001221 | accumulation order |
| `long-module-scheduler` | 93 | 2.2500 | 1.0000 | 2.25 | router logits, layer 3: 0.09375 vs 0.07812 | accumulation order |
| `transformer-attention-math` | 91 | 2.2500 | 1.0000 | 2.25 | dense mlp output, layer 1: 0.003662 vs 0.003418 | accumulation order |
| `rollout-postmortem` | 162 | 2.2500 | 1.0000 | 2.25 | dense mlp output, layer 1: 0.001953 vs 0.001465 | accumulation order |
| `arrow-etl-pipeline` | 453 | 2.2500 | 1.0000 | 2.25 | expert output, layer 2: 0.0003033 vs 0.000268 | accumulation order |
| `debug-session-nan-loss` | 389 | 1.1250 | 0.5000 | 2.25 | dense mlp output, layer 1: 0.004822 vs 0.004272 | accumulation order |
| `arrow-etl-pipeline` | 8 | 3.6250 | 1.6250 | 2.23 | dense mlp output, layer 1: 0.001035 vs 0.0009766 | accumulation order |
| `debug-session-nan-loss` | 122 | 2.5000 | 1.1250 | 2.22 | residual, layer 0: 0.001465 vs 0.0009766 | accumulation order |
| `zig-slice-bounds` | 162 | 2.7500 | 1.2500 | 2.20 | residual, layer 1: 0.0009766 vs 0.0008545 | accumulation order |
| `long-module-scheduler` | 189 | 2.7500 | 1.2500 | 2.20 | dense mlp output, layer 0: 0.01562 vs 0.007812 | accumulation order |
| `long-module-scheduler` | 343 | 1.3750 | 0.6250 | 2.20 | attention output, layer 3: 0.0002594 vs 0.0002441 | accumulation order |
| `rollout-postmortem` | 227 | 2.7500 | 1.2500 | 2.20 | shared expert output, layer 2: 0.0003967 vs 0.0002823 | accumulation order |
| `transformer-attention-math` | 188 | 3.0000 | 1.3750 | 2.18 | dense mlp output, layer 0: 0.002197 vs 0.001953 | accumulation order |
| `zig-slice-bounds` | 192 | 3.2500 | 1.5000 | 2.17 | router logits, layer 4: 0.07812 vs 0.0625 | accumulation order |
| `dijkstra-walkthrough` | 541 | 1.6250 | 0.7500 | 2.17 | expert output, layer 6: 0.0002747 vs 0.000267 | accumulation order |
| `debug-session-nan-loss` | 655 | 1.6250 | 0.7500 | 2.17 | dense mlp output, layer 1: 0.003174 vs 0.00293 | accumulation order |
| `changelog-structured` | 28 | 1.6250 | 0.7500 | 2.17 | residual, layer 0: 0.001953 vs 0.001465 | accumulation order |
| `debug-session-nan-loss` | 65 | 2.1250 | 1.0000 | 2.12 | dense mlp output, layer 1: 0.001709 vs 0.001099 | accumulation order |
| `debug-session-nan-loss` | 293 | 4.2500 | 2.0000 | 2.12 | expert output, layer 2: 0.0001554 vs 0.0001411 | accumulation order |
| `debug-session-nan-loss` | 866 | 2.1250 | 1.0000 | 2.12 | residual, layer 1: 0.001465 vs 0.0009766 | accumulation order |
| `json-invoice-records` | 18 | 6.8750 | 3.2500 | 2.12 | shared expert output, layer 2: 0.0006714 vs 0.0006104 | accumulation order |
| `tool-call-test-selection` | 1 | 1.3125 | 0.6250 | 2.10 | dense mlp output, layer 1: 0.002808 vs 0.002319 | accumulation order |
| `retry-backoff-analysis` | 39 | 5.2500 | 2.5000 | 2.10 | attention output, layer 2: 0.002197 vs 0.002075 | accumulation order |
| `observability-plan` | 88 | 5.2500 | 2.5000 | 2.10 | attention output, layer 5: 8.202e-05 vs 8.011e-05 | accumulation order |
| `long-module-scheduler` | 14 | 2.8750 | 1.3750 | 2.09 | dense mlp output, layer 0: 0.00293 vs 0.002686 | accumulation order |
| `debug-session-nan-loss` | 28 | 3.6250 | 1.7500 | 2.07 | residual, layer 2: 0.0004883 vs 0.0003662 | accumulation order |
| `zig-slice-bounds` | 200 | 4.5000 | 2.5000 | 1.80 | dense mlp output, layer 0: 0.003906 vs 0.00293 | accumulation order |

The first departure is rounding-sized on every row: median 0.001709, at most 0.01562 at a hidden-state tensor and 0.125 at router
logits. The rows that grow into large logit errors do so where a near-tied expert selection changes and
the changed expert's output is large. The router logits' own path difference is many times the boundary
gaps at those layers: no selection change in this corpus is clear of it (0 clear-margin changes in 142 rows). The
embedding is bit-identical on 142 of 142 traced rows; the final norm differs beyond its own
spread on 134 rows, and on 0 of those the residual entering it is
bit-identical to the oracle's.

## What was checked before calling a difference floating-point noise

1. **Tokenizer.** TensorFold's Zig tokenizer and the checkpoint tokenizer return identical ids on all 16 rendered histories; the corpus stores those ids and both engines consume them unchanged, so no position is compared across different tokens.
2. **Masks.** Both engines run one causal pass over the same ids from position 0 with no padding and no reordering. The 513-token sliding layers see the same window on both sides; the corpus includes 3 histories longer than the 513-token window so the ring is exercised.
3. **Quantization.** Both sides decode the same affine INT4 groups of 64 with INT8 embeddings from the same checkpoint files; the oracle is built with rec's `QATModel` and TensorFold reads the same shards.
4. **Config.** Both read the checkpoint's `config.json`: 49 layers, 14 NoPE full-attention layers, 513-token sliding window, 32 query and 4 key heads, head dim 128, 256 experts with 6 routed, `rms_norm_eps` 1e-6, `rope_theta` 10000, and BF16 KV.
5. **Accumulation grouping.** This is the difference that remains, and the two sections above size it on the same tokens: TensorFold's distance from the oracle's decode path is within the oracle's own chunked-prefill distance from that path, the residual peaks on the very first row (no context, so no mask or window can be involved), and it tracks the rows where rec_mlx already disagrees with itself.

## Quality gate

Decision 10 gates reordering changes against a baseline, and this item measures the baseline itself. The 
figures a later candidate must be compared against are the ones above; the numbers a candidate is allowed 
to reach are 1.10x the mean KL and at least 99.5% top-1 agreement on rows whose baseline top-2 margin is 
at least 0.1 logit.

## Baseline numbers

The parent copies this block into the plan's **Baseline** section.

```
flint parity: 16 histories, 6069 tokens, 911223936 positions
mean KL to oracle (rec_mlx e53663b2): 0.019052 nats
top-1 agreement: 95.518%
top-1 agreement, both margins >= 0.1: 98.218% (5667 positions)
```
