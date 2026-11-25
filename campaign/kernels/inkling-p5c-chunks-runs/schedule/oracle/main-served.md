# oMLX oracle against the main-served captures

- reference: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-ref
- candidate: /Users/josh/repos/tf/.work/tmp/p5c/oracle/main-served
- positions: 3015 over 16 histories
- top-1 agreement: 2865/3015 (95.02%); 50 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2753/2872 (95.86%)
- mean KL, mean of history means: 0.0186238; row-weighted: 0.0605144
- max KL: 8.42268 at cells_code position 280
- max absolute logit difference: 23.97
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1246 | 93.26 | 94.70 | 0.0412 | 4.978 | 1300 | 19 |
| cells_code | 1178 | 1128 | 95.76 | 96.04 | 0.1037 | 8.423 | 280 | 24 |
| greeting | 25 | 25 | 100.00 | 100.00 | 0.00363 | 0.02614 | 23 | 1.87 |
| explain_sliding | 34 | 32 | 94.12 | 93.75 | 0.005635 | 0.05585 | 7 | 4.54 |
| unit_test | 32 | 30 | 93.75 | 96.77 | 0.03942 | 1.024 | 28 | 8.27 |
| refactor | 36 | 34 | 94.44 | 94.29 | 0.008481 | 0.05559 | 7 | 6.53 |
| math_mod | 36 | 35 | 97.22 | 97.06 | 0.003358 | 0.05537 | 7 | 4.54 |
| math_frac | 44 | 42 | 95.45 | 97.62 | 0.0105 | 0.07757 | 39 | 5.21 |
| dialogue | 44 | 44 | 100.00 | 100.00 | 0.003819 | 0.05575 | 7 | 4.54 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 0.005421 | 0.02578 | 26 | 2.53 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.005279 | 0.05557 | 7 | 4.54 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.006974 | 0.0556 | 7 | 4.54 |
| sql | 33 | 32 | 96.97 | 96.77 | 0.04378 | 1.008 | 31 | 15.8 |
| prose_review | 35 | 35 | 100.00 | 100.00 | 0.005027 | 0.05575 | 7 | 4.55 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0.005422 | 0.0558 | 7 | 4.54 |
| token_budget | 37 | 37 | 100.00 | 100.00 | 0.006375 | 0.05574 | 7 | 4.64 |

Lowest top-1: harbour (93.26%), unit_test (93.75%), explain_sliding (94.12%).
Highest mean KL: cells_code (0.1037), sql (0.04378), harbour (0.0412).

