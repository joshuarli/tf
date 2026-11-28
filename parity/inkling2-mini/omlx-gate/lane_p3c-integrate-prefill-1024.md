# P3c at the served chunk 1024

- reference: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-ref
- candidate: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_p3c-integrate-prefill-1024
- positions: 3015 over 16 histories
- top-1 agreement: 2858/3015 (94.79%); 50 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2750/2872 (95.75%)
- mean KL, mean of history means: 0.0191658; row-weighted: 0.0622099
- max KL: 9.70252 at harbour position 1300
- max absolute logit difference: 23.97
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1240 | 92.81 | 94.45 | 0.04488 | 9.703 | 1300 | 18.8 |
| cells_code | 1178 | 1129 | 95.84 | 96.13 | 0.1037 | 8.423 | 280 | 24 |
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

Lowest top-1: harbour (92.81%), unit_test (93.75%), explain_sliding (94.12%).
Highest mean KL: cells_code (0.1037), harbour (0.04488), sql (0.04265).

