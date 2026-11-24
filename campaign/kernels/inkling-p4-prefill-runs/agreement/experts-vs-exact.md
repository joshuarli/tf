# INKLING_PREFILL_HALF=experts against the exact path

- reference: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_p4-prefill-wip-exact-prefill-512
- candidate: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_p4-prefill-wip-experts-prefill-512
- positions: 3015 over 16 histories
- top-1 agreement: 2899/3015 (96.15%); 0 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2795/2874 (97.25%)
- mean KL, mean of history means: 0.00736306; row-weighted: 0.0410334
- max KL: 8.88897 at cells_code position 307
- max absolute logit difference: 24.51
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1263 | 94.54 | 96.50 | 0.03048 | 3.471 | 1300 | 20.8 |
| cells_code | 1178 | 1140 | 96.77 | 97.25 | 0.06992 | 8.889 | 307 | 24.5 |
| greeting | 25 | 25 | 100.00 | 100.00 | 1.168e-05 | 0.0002629 | 9 | 0.514 |
| explain_sliding | 34 | 33 | 97.06 | 96.77 | 0.0005299 | 0.01359 | 22 | 1.61 |
| unit_test | 32 | 32 | 100.00 | 100.00 | 8.808e-06 | 0.0002489 | 9 | 0.162 |
| refactor | 36 | 36 | 100.00 | 100.00 | 7.494e-06 | 0.0002463 | 9 | 0.154 |
| math_mod | 36 | 36 | 100.00 | 100.00 | 0.004165 | 0.1218 | 34 | 2.5 |
| math_frac | 44 | 44 | 100.00 | 100.00 | 0.0001493 | 0.004579 | 31 | 0.933 |
| dialogue | 44 | 42 | 95.45 | 97.67 | 0.00126 | 0.01534 | 42 | 3.06 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 1.01e-05 | 0.0002677 | 9 | 0.147 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.001406 | 0.04186 | 9 | 2.89 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.002729 | 0.04175 | 9 | 2.9 |
| sql | 33 | 33 | 100.00 | 100.00 | 0.002342 | 0.02718 | 16 | 1.49 |
| prose_review | 35 | 34 | 97.14 | 97.14 | 0.001908 | 0.0253 | 29 | 2.78 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 6.145e-06 | 0.000252 | 9 | 0.153 |
| token_budget | 37 | 36 | 97.30 | 97.22 | 0.002873 | 0.05799 | 30 | 3.41 |

Lowest top-1: harbour (94.54%), dialogue (95.45%), cells_code (96.77%).
Highest mean KL: cells_code (0.06992), harbour (0.03048), math_mod (0.004165).

