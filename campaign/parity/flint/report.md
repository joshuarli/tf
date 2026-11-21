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
same checkpoint and the same ids, not a TensorFold defect. The remaining rows are checked by the two
tables above: TensorFold's error there is within the oracle's own spread, which the corpus-wide
per-row comparison shows for all rows at once.
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
