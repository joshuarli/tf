# Inkling 2 Mini: TensorFold against the oMLX oracle

- oracle: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-ref (digest b01512f0396eeb23, oMLX 0.8.3 e72868dac12d on MLX 0.32.4, 2,048-token forwards, MTP off)
- candidate: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_d2-integrate-prefill-512
- runner sha256: 5d09ed6385c0444d1113bcdbea14c1d374bf4a5522c120fd6d01d02d5d56a6b5
- model identity: 7b1841c868807cd5df91a76646f550f6b0bd2ff803f4078b1009d540fd34195a
- arithmetic policy: f32-activations+prefill-512
- positions: 3015 over 16 histories
- top-1 agreement: 2863/3015 (94.96%); 50 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2750/2872 (95.75%)
- mean KL, mean of history means: 0.0194994; row-weighted: 0.0639952
- max KL: 14.2476 at cells_code position 378
- max absolute logit difference: 26.23
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1244 | 93.11 | 94.37 | 0.03916 | 4.284 | 1300 | 15.3 |
| cells_code | 1178 | 1130 | 95.93 | 96.22 | 0.1148 | 14.25 | 378 | 26.2 |
| greeting | 25 | 25 | 100.00 | 100.00 | 0.003636 | 0.02615 | 23 | 1.87 |
| explain_sliding | 34 | 32 | 94.12 | 93.75 | 0.005628 | 0.05557 | 7 | 4.54 |
| unit_test | 32 | 30 | 93.75 | 96.77 | 0.03928 | 1.021 | 28 | 8.26 |
| refactor | 36 | 34 | 94.44 | 94.29 | 0.008512 | 0.05565 | 7 | 6.53 |
| math_mod | 36 | 35 | 97.22 | 97.06 | 0.006921 | 0.1183 | 34 | 4.54 |
| math_frac | 44 | 42 | 95.45 | 97.62 | 0.008641 | 0.08055 | 31 | 5.24 |
| dialogue | 44 | 43 | 97.73 | 97.67 | 0.004588 | 0.0558 | 7 | 4.54 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 0.005437 | 0.02582 | 26 | 2.53 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.007188 | 0.05603 | 7 | 4.54 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.008553 | 0.05564 | 7 | 4.54 |
| sql | 33 | 32 | 96.97 | 96.77 | 0.04265 | 1.023 | 31 | 15.9 |
| prose_review | 35 | 34 | 97.14 | 100.00 | 0.006068 | 0.05532 | 7 | 4.54 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0.004597 | 0.05572 | 7 | 4.54 |
| token_budget | 37 | 37 | 100.00 | 100.00 | 0.006382 | 0.05588 | 7 | 4.64 |

Lowest top-1: harbour (93.11%), unit_test (93.75%), explain_sliding (94.12%).
Highest mean KL: cells_code (0.1148), sql (0.04265), unit_test (0.03928).

Baseline c053f52b2: top-1 2861/3015, mean KL 0.0195085, row-weighted 0.0641971, max KL 8.01205 (reported, not gated).
Thresholds: top-1 at least 94.392%, mean KL at most 0.0292627, row-weighted mean KL at most 0.0962956, no non-finite logits.

omlx-oracle gate: pass
