# INKLING_PREFILL_HALF=inputs against the exact path

- reference: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_p4-prefill-wip-exact-prefill-512
- candidate: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_p4-prefill-wip-inputs-prefill-512
- positions: 3015 over 16 histories
- top-1 agreement: 2900/3015 (96.19%); 0 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2797/2874 (97.32%)
- mean KL, mean of history means: 0.00758986; row-weighted: 0.0384035
- max KL: 8.30221 at cells_code position 307
- max absolute logit difference: 24.05
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1262 | 94.46 | 96.50 | 0.02399 | 2.499 | 1283 | 15.1 |
| cells_code | 1178 | 1142 | 96.94 | 97.42 | 0.07023 | 8.302 | 307 | 24 |
| greeting | 25 | 25 | 100.00 | 100.00 | 9.496e-07 | 3.444e-06 | 12 | 0.511 |
| explain_sliding | 34 | 34 | 100.00 | 100.00 | 0.0005671 | 0.01598 | 30 | 2.12 |
| unit_test | 32 | 32 | 100.00 | 100.00 | 0.003662 | 0.03273 | 18 | 2.66 |
| refactor | 36 | 36 | 100.00 | 100.00 | 0.001117 | 0.03696 | 34 | 1.8 |
| math_mod | 36 | 36 | 100.00 | 100.00 | 0.003476 | 0.1248 | 34 | 2.49 |
| math_frac | 44 | 44 | 100.00 | 100.00 | 0.001701 | 0.04257 | 9 | 2.9 |
| dialogue | 44 | 41 | 93.18 | 95.35 | 0.001426 | 0.02078 | 30 | 1.72 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 1.025e-05 | 0.0002586 | 9 | 0.145 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.001282 | 0.02418 | 32 | 1.53 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.002808 | 0.04195 | 9 | 2.9 |
| sql | 33 | 33 | 100.00 | 100.00 | 0.002346 | 0.02722 | 16 | 1.51 |
| prose_review | 35 | 34 | 97.14 | 97.14 | 0.003119 | 0.04269 | 9 | 2.96 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0.002819 | 0.04457 | 9 | 2.88 |
| token_budget | 37 | 36 | 97.30 | 97.22 | 0.002882 | 0.04458 | 9 | 3.34 |

Lowest top-1: dialogue (93.18%), harbour (94.46%), cells_code (96.94%).
Highest mean KL: cells_code (0.07023), harbour (0.02399), unit_test (0.003662).

