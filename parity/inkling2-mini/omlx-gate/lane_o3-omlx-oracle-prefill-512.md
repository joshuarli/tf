# Inkling 2 Mini: TensorFold against the oMLX oracle

- oracle: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-ref (digest b01512f0396eeb23, oMLX 0.8.3 e72868dac12d on MLX 0.32.4, 2,048-token forwards, MTP off)
- candidate: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_o3-omlx-oracle-prefill-512
- runner sha256: edc7b33f2604dae73d393827dac2f46613f6132278f5ceca4ebda70529d49210
- model identity: 7b1841c868807cd5df91a76646f550f6b0bd2ff803f4078b1009d540fd34195a
- arithmetic policy: f32-activations+prefill-512
- positions: 3015 over 16 histories
- top-1 agreement: 2861/3015 (94.89%); 50 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2753/2872 (95.86%)
- mean KL, mean of history means: 0.0195085; row-weighted: 0.0641971
- max KL: 8.01205 at harbour position 1300
- max absolute logit difference: 27.8
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1241 | 92.89 | 94.78 | 0.04367 | 8.012 | 1300 | 19.1 |
| cells_code | 1178 | 1130 | 95.93 | 96.04 | 0.1101 | 7.567 | 1074 | 27.8 |
| greeting | 25 | 25 | 100.00 | 100.00 | 0.003636 | 0.02618 | 23 | 1.86 |
| explain_sliding | 34 | 32 | 94.12 | 93.75 | 0.005613 | 0.05578 | 7 | 4.54 |
| unit_test | 32 | 30 | 93.75 | 96.77 | 0.03933 | 0.9269 | 28 | 8.01 |
| refactor | 36 | 34 | 94.44 | 94.29 | 0.01148 | 0.07741 | 34 | 6.51 |
| math_mod | 36 | 35 | 97.22 | 97.06 | 0.003685 | 0.05586 | 7 | 4.54 |
| math_frac | 44 | 42 | 95.45 | 97.62 | 0.008951 | 0.08063 | 31 | 5.24 |
| dialogue | 44 | 43 | 97.73 | 97.67 | 0.004977 | 0.05549 | 7 | 4.54 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 0.005432 | 0.02582 | 26 | 2.53 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.005756 | 0.05579 | 7 | 4.54 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.007218 | 0.05583 | 7 | 4.55 |
| sql | 33 | 32 | 96.97 | 96.77 | 0.04381 | 1.009 | 31 | 15.8 |
| prose_review | 35 | 35 | 100.00 | 100.00 | 0.006686 | 0.05548 | 7 | 4.54 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0.005412 | 0.05588 | 7 | 4.54 |
| token_budget | 37 | 37 | 100.00 | 100.00 | 0.006351 | 0.05559 | 7 | 4.64 |

Lowest top-1: harbour (92.89%), unit_test (93.75%), explain_sliding (94.12%).
Highest mean KL: cells_code (0.1101), sql (0.04381), harbour (0.04367).

Baseline c053f52b2: top-1 2861/3015, mean KL 0.0195085, row-weighted 0.0641971, max KL 8.01205 (reported, not gated).
Thresholds: top-1 at least 94.392%, mean KL at most 0.0292627, row-weighted mean KL at most 0.0962956, no non-finite logits.

omlx-oracle gate: pass
