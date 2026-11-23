# Inkling 2 Mini: harbour past the sliding window

`campaign/kernels/inkling-omlx-oracle.md` found that over harbour positions 1,280 to 1,335, the last 56 scored
positions, TensorFold's mean KL to the oMLX oracle was 0.2056 (`prefill 512`) and 0.1391 (`prefill 1`), while the
references agreed with one another to within 0.02 to 0.04. Harbour runs past the 1,024-row sliding window, so the
local ring, the relative bias past its extent, the loop's slot mapping, the convolution rings and the log scaling
were suspects. This receipt records what was measured. The investigation was stopped at the campaign's wind-down,
before the last two planned steps; they are listed under "Next step".

## Verdict

No TensorFold defect was found, and the evidence points to none. No source was changed, and the gate baseline
stands.

- There is no onset. The tail statistic comes from two positions, 1281 and 1300. Without them the tail's mean KL
  is 0.0189 (`prefill 512`) and 0.0162 (`prefill 1`). The tail's median KL, 0.008 on both paths, is close to the
  median of positions 0 to 1,279, 0.006.
- At both positions the output is chaotic in the arithmetic. Every engine agrees with every other at its rounding
  floor until a router near-tie in a loop pass, with an 8th/9th selection gap of 1e-4 to 5e-3. Rounding decides the
  tie, and the expert sets then cascade apart through the remaining passes.
- At 1300, TensorFold's per-token decode forward (`logits` mode) agrees with the oracle. It gives `' minute'`
  0.79 against the oracle's 0.77, a KL of 0.15. TensorFold's decode and `prefill 512` paths differ from each other
  by 8.28 nats at 1300, more than `prefill 512` differs from the oracle (8.01).
- oMLX departs from its own oracle at 1300 when only its activation precision changes. With every bf16 parameter
  widened to f32, f32 activations and no bf16 rounding in its loop controllers, its KL to the bf16 oracle at 1300
  is 11.86, and it gives `' minute'` less
  than 0.0001, as TensorFold `prefill 512` does.
- Past the window as a whole, TensorFold is as close to the oracle as oMLX's own decode path. Over positions 1,024
  to 1,335 the mean KL to the oracle is 0.0681 for TensorFold's decode forward and 0.0719 for oMLX's one-token
  forwards.

The first divergence, layer by layer, is at visit 18 (slot 24: layer 6 in the second pass) at 1300, and at
visit 19 (slot 25: layer 7 in the second pass) at 1281, for TensorFold's decode forward against the oracle. Both
are expert-choice near-ties, with the oracle's gap at 0.0040 and 0.00067. Before them, TensorFold and the oracle
agree at every visit to the same degree that the oracle agrees with itself run in f32.

## The tail is two positions

KL(oracle ‖ candidate) per harbour position, in nats. The oracle is oMLX with 2,048-token forwards. "oMLX one-token"
is the oracle's one-token-a-forward capture, `.work/parity/inkling2-mini/omlx-ref-chunk1`. The f32 rows use the
native checkpoint's saved f32 captures. TensorFold's decode forward is the runner's `logits` mode, one
`inkling.position` a token, which the oracle receipt did not capture.

| Pairing | Mean 1,280–1,335 | Without 1281 and 1300 | Share of those two | Median 1,280–1,335 | Median 0–1,279 |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle, TensorFold `prefill 512` | 0.2056 | 0.0189 | 91% | 0.0083 | 0.0064 |
| oracle, TensorFold `prefill 1` | 0.1391 | 0.0162 | 89% | 0.0077 | 0.0059 |
| oracle, TensorFold decode | 0.0793 | 0.0342 | 58% | 0.0103 | 0.0057 |
| oracle, oMLX one-token | 0.0215 | 0.0136 | 39% | 0.0095 | 0.0075 |
| f32, oracle | 0.0384 | 0.0299 | 28% | 0.0213 | 0.0288 |

The 56-row window holds TensorFold's two spikes and none of oMLX's. Starting it 44 rows earlier, at 1,236, takes
in oMLX's own spikes at 1237 and 1268, and oMLX one-token's mean rises from 0.0215 to 0.1628:

| Window | `prefill 512` | `prefill 1` | TensorFold decode | oMLX one-token | f32 against the oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,024–1,335 | 0.0862 | 0.0669 | 0.0681 | 0.0719 | 0.0692 |
| 1,200–1,335 | 0.1535 | 0.1097 | 0.0995 | 0.1238 | 0.0723 |
| 1,236–1,335 | 0.2003 | 0.1332 | 0.1216 | 0.1628 | 0.0788 |
| 1,280–1,335 | 0.2056 | 0.1391 | 0.0793 | 0.0215 | 0.0384 |

## Past the window

Positions with KL above 1 nat against the oracle occur on both sides of the window, for every engine:

| Candidate | Below 1,024 | 1,024 and past |
| --- | --- | --- |
| TensorFold decode | 152, 200, 229, 881, 961 | 1237 (6.07), 1281 (2.44) |
| TensorFold `prefill 1` | 152, 200, 229, 961 | 1237, 1268, 1281, 1300 (4.48) |
| TensorFold `prefill 512` | 152, 200, 427, 841, 961 | 1170, 1237, 1268, 1281, 1300 (8.01) |
| oMLX one-token | 354, 771, 881, 961 | 1237 (3.30), 1268 (10.82) |

TensorFold's own paths part at the same kind of position. Decode against `prefill 1` exceeds 1 nat only at 1237
(4.42) and 1300 (4.64). Decode against `prefill 512` exceeds it at 229, 427, 841 and 1300 (8.28).

An oracle-confident miss is a position where the oracle gives its top token at least 0.5 and a candidate gives
that token less than a tenth of the oracle's probability. In harbour, the f32 reference misses at 1237 and 1268,
oMLX one-token at 1237 and 1268 (0.0 against the oracle's 0.69 at 1268), both TensorFold prefill paths at 1237 and
1300, and TensorFold decode at 1237 only. In cells_code each engine has 9 to 12.

The top-1 breakdown of `inkling-omlx-oracle.md`, split at the window:

| History and range | All agree | f32 differs | oMLX differs | TensorFold differs | No majority |
| --- | ---: | ---: | ---: | ---: | ---: |
| harbour 0–1,023 | 878 | 69 | 34 | 25 | 18 |
| harbour 1,024–1,335 | 262 | 18 | 14 | 13 | 5 |
| cells_code 0–1,023 | 955 | 26 | 22 | 14 | 7 |
| cells_code 1,024–1,177 | 145 | 4 | 3 | 2 | 0 |

TensorFold here is the `prefill 512` baseline. Past the window it is the odd one out as often as oMLX is.

## Position 1300, visit by visit

At 1300 the history reads `... a minute that is not a`, and the next token is `' minute'`:

| Engine | p(`' minute'`) | Top tokens |
| --- | ---: | --- |
| oracle (oMLX bf16, 2,048-token forward) | 0.771 | `' minute'` 0.771, `' '` 0.120 |
| oMLX bf16, one-token forwards | 0.964 | `' minute'` 0.964, `' '` 0.013 |
| f32 reference, native checkpoint | 0.951 | `' minute'` 0.951, `' '` 0.016 |
| TensorFold decode | 0.794 | `' minute'` 0.794, `' '` 0.070 |
| TensorFold `prefill 1` | 0.002 | `' '` 0.610, `' n'` 0.164 |
| TensorFold `prefill 512` | <0.0001 | `' '` 0.104, `' n'` 0.084, `' particular'` 0.052 |
| oMLX f32 | <0.0001 | `''` 0.048, `' '` 0.047, `' new'` 0.046 |
| oMLX f32, f16 keys and values | <0.0001 | `' '` 0.107 |

"oMLX f32" is the oracle's model with three changes: every bf16 parameter widened to f32 (no float parameter is
left narrower), f32 embeddings fed in, and the bf16 rounding in the loop controllers' down projection removed. "f16
keys and values" also rounds keys and values to f16 before attention, as TensorFold stores them. Both ran as one
1,336-token forward. On the M5 Max, MLX's f32 matmuls over many rows measured 7.7e-4 relative error against float64
(plain matmuls of 64 and 1,336 rows, 8-bit quantized ones of 1,336 rows), against 2e-7 to 9e-7 for one row, so
these runs are f32 only to about 1e-3. Over rows 1,200 to 1,335,
KL(oracle ‖ oMLX f32) averages 0.1725, and KL(oMLX f32 ‖ TensorFold `prefill 512`) 0.0690.

The oMLX probe recorded, for each of the 48 visits at the chosen positions, the attention sub-layer output after its
short convolution, the FFN input, the layer output and the router's 128 selection scores. The runner's `tracerow`
records the same rows for TensorFold's decode forward. Relative L2 differences over the prelude and first-pass
visits (0 to 17) that come before the first visit whose eight experts differ, and the largest layer-output
difference over all 48 visits:

| Pairing | Position | First visit with other experts (8th/9th gaps) | Local attention | Global attention | Layer output | Largest layer output |
| --- | ---: | --- | --- | --- | --- | ---: |
| TensorFold decode, oracle | 1299 | 20 (0.0032, 0.0020) | 0.35–3.8% | 1.2–5.9% | 0.57–2.9% | 8.1% |
| TensorFold decode, oracle | 1300 | 18 (0.0018, 0.0040) | 0.34–3.6% | 2.0–5.8% | 0.39–2.6% | 85% |
| oracle, oMLX f32 with f16 KV | 1299 | 20 (0.0020, 0.0028) | 0.34–3.9% | 1.3–5.9% | 0.59–2.9% | 7.6% |
| oracle, oMLX f32 with f16 KV | 1300 | 18 (0.0040, 0.0028) | 0.38–3.7% | 2.0–5.9% | 0.39–2.6% | 128% |
| TensorFold decode, oMLX f32 with f16 KV | 1299 | 21 (0.0020, 0.0011) | 0.09–1.5% | 0.22–0.92% | 0.13–0.47% | 5.5% |
| TensorFold decode, oMLX f32 with f16 KV | 1300 | 23 (0.0036, 0.0047) | 0.09–1.6% | 0.21–0.85% | 0.11–0.53% | 79% |

The oracle against oMLX at f32 is the same engine at two precisions. It agrees with itself at the bf16 floor
through the first pass, parts at visit 18 over expert 28 against 72, and from visit 29 on selects a different set
at every visit. By the end of the third pass the layer output differs by 128%. TensorFold against the oracle has
the same shape and size. TensorFold against oMLX at f32 agrees two to ten times more closely and parts later. At
1299 no pairing cascades: the isolated 8th-expert swaps stay local, and the layer outputs stay within 8.1%.

TensorFold's decode forward predicts `' minute'` at 1300 although its experts differ from the oracle's at every
visit of the third pass. Different expert sets do not by themselves change the prediction. Where the cascade ends
depends on every choice along it, and it ends differently for each of TensorFold's three paths.

## Position 1281

At 1281 every engine's top token is `' '` and every engine gives the actual next token, `' bell'`, less than
0.0001. The KL comes from the tail of the distribution. All three TensorFold paths give 2.44 to 2.48.

| Pairing | Position | First visit with other experts (8th/9th gaps) | Local attention | Global attention | Layer output | Largest layer output |
| --- | ---: | --- | --- | --- | --- | ---: |
| TensorFold decode, oracle | 1280 | 10 (0.0023, 0.0058) | 0.41–4.6% | 1.1% | 0.49–2.5% | 12% |
| TensorFold decode, oracle | 1281 | 19 (0.0047, 0.00067) | 0.38–4.7% | 1.1–5.0% | 0.42–3.0% | 88% |
| TensorFold decode, oMLX f32 with f16 KV | 1280 | 18 (0.00088, 0.00031) | 0.09–1.5% | 0.23–1.3% | 0.11–0.61% | 7.3% |
| TensorFold decode, oMLX f32 with f16 KV | 1281 | 11 (0.00015, 0.00061) | 0.09–0.40% | 0.25% | 0.12–0.20% | 73% |

At visit 11 (slot 11: layer 11, a full-attention layer in the first pass) experts 14 and 124 sit 8th and 9th.
TensorFold's selection scores are 0.321496 and 0.321345, a gap of 1.5e-4; the oracle's are 0.320786 and 0.318161.
Both pick 14. oMLX at f32 picks 124 by 9.6e-4, and with f16 keys and values by 6.1e-4. TensorFold and the oracle
first part at visit 19, where the oracle's gap is 6.7e-4 and the two sides' selection scores differ by up to 0.019.
oMLX at f32 with and without f16 keys and values, which differ only in that rounding, are 0.78 nats apart at 1281.
TensorFold is 0.64 to 0.67 nats from oMLX at f32 with f16 keys and values, against 2.44 to 2.48 from the oracle.

## The suspects

A defect in any of these would change the attention outputs of the visits that use it, from the first such visit
on, at every position past the window. It would not wait for a router near-tie. The tables above bound any such effect at 1280, 1281, 1299 and 1300 to
TensorFold's agreement with oMLX at f32 with f16 keys and values, about 1% and no more than 1.6%. Removing MLX's
own multi-row matmul error from that bound is the first step under "Next step".

- **Local ring wrap and read index.** At these positions every local visit reads wrapped ring rows: position p's
  window is p − 1,023 to p, stored at row p mod 1,024. Their attention outputs agree with oMLX to 0.09% to 1.6%.
  oMLX keeps a window buffer of whole 16-key tiles, not a ring.
- **Relative bias past the extent.** Global visits 5, 11 and 17 attend over 1,281 to 1,301 keys, 257 to 277 of them
  at distances past the 1,024-distance extent, where both engines add no bias. Their attention outputs agree with
  oMLX to 0.21% to 1.3%.
- **Global and local slot mapping in the loop.** The second and third passes run slots 24 to 47. At 1280 and 1299,
  where routing does not cascade, every one of the 48 visits stays within 7.3% of oMLX at f32, and expert choices
  differ at only five and three of them.
- **Convolution rings.** Every visit reads four rings of four positions. They wrap every four positions, below and
  past the window alike.
- **Log-floor and log-alpha scaling.** The checkpoint sets `log_scaling_n_floor` to 8,192, so the scale,
  `1 + 0.1 · log(max(1, (p + 1) / 8192))`, is exactly 1 at every position below 8,191. Neither history reaches it.

The oracle receipt's two TensorFold paths agree with each other at 1300 only in missing `' minute'`: KL(`prefill
1` ‖ `prefill 512`) is 1.22 there, and 2.55 at 1237.

## The f32 traces

No conclusion here depends on the f32 per-layer traces in `.work/parity/inkling2-mini/trace-ref`. They come from
the native checkpoint, whose weights differ from the oQ4e weights by the quantization, so they cannot localize an
oQ4e difference at this precision. Their harbour files are whole-history summaries and prefixes ending at
positions 556, 557, 1008 and 1009; none reaches 1281 or 1300.

## Gate

No arithmetic changed, so the gate baseline in `campaign/parity/inkling2-mini/omlx_gate.json` stands: top-1 2,861
of 3,015 (94.89%), mean KL 0.0195085 over the 16 history means, row-weighted mean KL 0.0641971, maximum 8.012 at
harbour 1300. The gate was not re-run. The runner built from this branch has sha256
`78bb660db4e10c5409ba149ec68e7a84a23c7b6feac57d2c3903b036365c55c3`, not the baseline's `edc7b33f...`, from the same
Zig sources. No threshold was edited. The decode forward's harbour capture, which the gate does not use, has top-1
1,250 of 1,336 and mean KL 0.0394 against the oracle, against `prefill 512`'s 1,241 and 0.0437. Its tail mean KL is
0.0793, against 0.2056.

## Next step

Two prepared steps were not run before the wind-down.

1. Run oMLX at f32 one token a forward, eagerly, so that every matmul is a single-row product, with f16 keys and
   values. Record visits at 998–1001, 1022–1025 and 1278–1301, and compare them with `tracerow` at 1000, 1023, 1024,
   1280, 1281, 1299 and 1300. Agreement near 1e-5 before the first near-tie would bound any long-context defect to
   that level, below and past the window alike.
2. Impose TensorFold's expert choices on the oracle at one row: `prefillrow 512` at 1281 and 1300, then route
   those rows' 48 visits in oMLX to the sidecar's picks, with the oracle's own selection scores renormalized as its
   router does. If the oracle then reproduces TensorFold `prefill 512`'s distribution at that row, the departure is
   routing alone.

## Reproduction

From the repository root, with `TMPDIR=$HOME/repos/tf/.work/tmp` and
`ZIG_GLOBAL_CACHE_DIR=$HOME/repos/tf/.work/zig-global-cache`, the runner:

```sh
zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$TMPDIR/d3/base-build"
RUN="$TMPDIR/d3/base-build/bin/tf-inkling2-loop-run"
OQ="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
TOK="$HOME/repos/tf/.work/parity/inkling2-mini/omlx-gate/baseline-c053f52b2-prefill-512/harbour.tokens.json"
python3 campaign/heavy.py -- "$RUN" "$OQ" "$TOK" logits "$TMPDIR/d3/tf/harbour-logits.bin"
for at in 1280 1281 1299 1300; do
  python3 campaign/heavy.py -- "$RUN" "$OQ" "$TOK" tracerow "$TMPDIR/d3/tf/tracerow-$at.bin" "$at"
done
```

Each `tracerow` run took 42 to 47 s and the `logits` run 45 s. The harbour token file is the history's 1,336 ids,
identical to `histories.json`.

The oMLX captures load the model as the oracle does, with `omlx.patches.inkling2.vlm.load`, and call
`model.forward(ids, cache)` on one `model.make_cache()`, in the pinned `.work/venv-omlx` under `heavy.py` with
`HF_HUB_OFFLINE=1`. Loaded that way, oMLX bf16 reproduced the oracle's harbour rows 1,200 to 1,335 bit for bit
(maximum absolute difference 0). For the f32 runs, `model.apply` casts every bf16 or f16 parameter to f32, the
embeddings `model.model.embed_norm(model.model.embed(ids))` are passed as `input_embeddings`, and
`omlx.patches.inkling2.model._mix_messages` is replaced by the same function without its two bf16 casts. For f16
keys and values, `Inkling2Attention.attend` rounds `k` and `v` to f16 and back before calling the original.
Per-visit rows come from replacing `Inkling2Layer._post` by a copy that also returns, for the chosen rows, the
short convolution's output (`residual - h`, or `residual` at a block boundary), `residual`, the layer output and
`sigmoid(x @ router_weight.T) + router_bias` of the MoE input. A visit's experts are the eight highest selection
scores, with ties to the lower index. The three runs each took 2 to 5 s after a 2 s load and charged at most
22.9 GB under `heavy.py`.

`campaign/kernels/inkling_long_context.py` implements these captures (`capture`), the per-visit comparison
(`visits`) and the per-position KL (`kl`). It is committed separately as work in progress: `visits` and `kl`
reproduce the numbers above from the saved files, and `capture` has not been run in its committed form.

```sh
PY="$HOME/repos/tf/.work/venv-ref/bin/python -I -B"
$PY campaign/kernels/inkling_long_context.py kl "$HOME/repos/tf/.work/parity/inkling2-mini/omlx-ref" \
  "$TMPDIR/d3/tf/harbour-logits.bin" --at 1281,1300
$PY campaign/kernels/inkling_long_context.py visits "tf:$TMPDIR/d3/tf" "$TMPDIR/d3/out/omlx-bf16.npz" 1299,1300
```

The first prints the decode forward's rows of the tables above, among them 1281 at 2.4415 and 1300 at 0.1545. The
second prints the per-visit rows behind the "TensorFold decode, oracle" lines. The oMLX per-visit captures were one
1,336-token forward each, recording rows 1,278 to 1,301.
