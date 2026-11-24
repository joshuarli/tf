# main's served schedule against the lane's

- reference: /Users/josh/repos/tf/.work/tmp/p5c/oracle/main-served
- candidate: /Users/josh/repos/tf/.work/tmp/p5c/oracle/lane-served
- positions: 3015 over 16 histories
- top-1 agreement: 3001/3015 (99.54%); 0 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2884/2891 (99.76%)
- mean KL, mean of history means: 0.000254906; row-weighted: 0.0017786
- max KL: 0.570673 at harbour position 1110
- max absolute logit difference: 11.3
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1325 | 99.18 | 99.60 | 0.003532 | 0.5707 | 1110 | 10.7 |
| cells_code | 1178 | 1175 | 99.75 | 99.83 | 0.0005468 | 0.2315 | 1074 | 11.3 |
| greeting | 25 | 25 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| explain_sliding | 34 | 34 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| unit_test | 32 | 32 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| refactor | 36 | 36 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| math_mod | 36 | 36 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| math_frac | 44 | 44 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| dialogue | 44 | 44 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| structured | 35 | 35 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| summary | 35 | 35 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| sql | 33 | 33 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| prose_review | 35 | 35 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |
| token_budget | 37 | 37 | 100.00 | 100.00 | 0 | 0 | 0 | 0 |

Lowest top-1: harbour (99.18%), cells_code (99.75%), greeting (100.00%).
Highest mean KL: harbour (0.003532), cells_code (0.0005468), greeting (0).

