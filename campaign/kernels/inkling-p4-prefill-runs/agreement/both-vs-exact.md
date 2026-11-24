# INKLING_PREFILL_HALF=both against the exact path

- reference: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_p4-prefill-wip-exact-prefill-512
- candidate: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_p4-prefill-wip-both-prefill-512
- positions: 3015 over 16 histories
- top-1 agreement: 2899/3015 (96.15%); 0 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2802/2874 (97.49%)
- mean KL, mean of history means: 0.00605888; row-weighted: 0.0328124
- max KL: 12.9484 at cells_code position 307
- max absolute logit difference: 21.2
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1255 | 93.94 | 96.34 | 0.02792 | 3.302 | 1281 | 20.6 |
| cells_code | 1178 | 1149 | 97.54 | 98.11 | 0.05179 | 12.95 | 307 | 21.2 |
| greeting | 25 | 25 | 100.00 | 100.00 | 1.042e-06 | 5.916e-06 | 3 | 0.0225 |
| explain_sliding | 34 | 33 | 97.06 | 96.77 | 0.0005207 | 0.0136 | 22 | 1.6 |
| unit_test | 32 | 32 | 100.00 | 100.00 | 0.003672 | 0.03283 | 18 | 2.66 |
| refactor | 36 | 36 | 100.00 | 100.00 | 0.001501 | 0.04574 | 34 | 2.56 |
| math_mod | 36 | 36 | 100.00 | 100.00 | 1.005e-05 | 0.0002387 | 9 | 0.15 |
| math_frac | 44 | 44 | 100.00 | 100.00 | 6.501e-06 | 0.0002332 | 9 | 0.145 |
| dialogue | 44 | 41 | 93.18 | 95.35 | 0.001684 | 0.02088 | 30 | 2.52 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 1.029e-06 | 6.181e-06 | 3 | 0.0285 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.001568 | 0.04328 | 9 | 2.88 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.002552 | 0.04316 | 9 | 2.89 |
| sql | 33 | 33 | 100.00 | 100.00 | 1.187e-06 | 1.004e-05 | 5 | 0.0243 |
| prose_review | 35 | 34 | 97.14 | 97.14 | 0.001862 | 0.02523 | 29 | 2.78 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0.001234 | 0.01314 | 41 | 2.37 |
| token_budget | 37 | 36 | 97.30 | 97.22 | 0.002625 | 0.0582 | 30 | 2.65 |

Lowest top-1: dialogue (93.18%), harbour (93.94%), explain_sliding (97.06%).
Highest mean KL: cells_code (0.05179), harbour (0.02792), unit_test (0.003672).

