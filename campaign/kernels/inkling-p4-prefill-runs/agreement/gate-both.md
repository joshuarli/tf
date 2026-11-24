# Inkling 2 Mini: TensorFold against the oMLX oracle

- oracle: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-ref (digest b01512f0396eeb23, oMLX 0.8.3 e72868dac12d on MLX 0.32.4, 2,048-token forwards, MTP off)
- candidate: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_p4-prefill-wip-both-prefill-512
- runner sha256: dad7a356aac51214b6b383fdf125c562e094936afd96395921efbcc7a6de8fb5
- model identity: 7b1841c868807cd5df91a76646f550f6b0bd2ff803f4078b1009d540fd34195a
- arithmetic policy: f32-activations+prefill-512
- positions: 3015 over 16 histories
- top-1 agreement: 2856/3015 (94.73%); 50 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2747/2872 (95.65%)
- mean KL, mean of history means: 0.0199132; row-weighted: 0.0670132
- max KL: 16.8877 at cells_code position 378
- max absolute logit difference: 23.85
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1232 | 92.22 | 93.88 | 0.04521 | 7.595 | 1300 | 20.9 |
| cells_code | 1178 | 1135 | 96.35 | 96.56 | 0.1156 | 16.89 | 378 | 23.8 |
| greeting | 25 | 25 | 100.00 | 100.00 | 0.003617 | 0.02613 | 23 | 1.86 |
| explain_sliding | 34 | 31 | 91.18 | 90.62 | 0.006234 | 0.0555 | 7 | 4.54 |
| unit_test | 32 | 30 | 93.75 | 96.77 | 0.03936 | 0.9268 | 28 | 8.01 |
| refactor | 36 | 34 | 94.44 | 94.29 | 0.00951 | 0.05798 | 34 | 6.53 |
| math_mod | 36 | 35 | 97.22 | 97.06 | 0.006954 | 0.1189 | 34 | 4.54 |
| math_frac | 44 | 42 | 95.45 | 97.62 | 0.00866 | 0.08056 | 31 | 5.24 |
| dialogue | 44 | 43 | 97.73 | 97.67 | 0.004722 | 0.0555 | 7 | 4.54 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 0.005439 | 0.02576 | 26 | 2.53 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.005781 | 0.0555 | 7 | 4.54 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.00625 | 0.0555 | 7 | 4.54 |
| sql | 33 | 32 | 96.97 | 96.77 | 0.04265 | 1.022 | 31 | 15.9 |
| prose_review | 35 | 35 | 100.00 | 100.00 | 0.005031 | 0.0555 | 7 | 4.54 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0.005424 | 0.0555 | 7 | 4.54 |
| token_budget | 37 | 37 | 100.00 | 100.00 | 0.008158 | 0.112 | 30 | 4.64 |

Lowest top-1: explain_sliding (91.18%), harbour (92.22%), unit_test (93.75%).
Highest mean KL: cells_code (0.1156), harbour (0.04521), sql (0.04265).

Baseline c053f52b2: top-1 2861/3015, mean KL 0.0195085, row-weighted 0.0641971, max KL 8.01205 (reported, not gated).
Thresholds: top-1 at least 94.392%, mean KL at most 0.0292627, row-weighted mean KL at most 0.0962956, no non-finite logits.

omlx-oracle gate: pass
