# Inkling 2 Mini: TensorFold against the oMLX oracle

- oracle: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-ref (digest b01512f0396eeb23, oMLX 0.8.3 e72868dac12d on MLX 0.32.4, 2,048-token forwards, MTP off)
- candidate: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_p4-prefill-wip-inputs-prefill-512
- runner sha256: dad7a356aac51214b6b383fdf125c562e094936afd96395921efbcc7a6de8fb5
- model identity: 7b1841c868807cd5df91a76646f550f6b0bd2ff803f4078b1009d540fd34195a
- arithmetic policy: f32-activations+prefill-512
- positions: 3015 over 16 histories
- top-1 agreement: 2857/3015 (94.76%); 50 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2750/2872 (95.75%)
- mean KL, mean of history means: 0.0191293; row-weighted: 0.0600793
- max KL: 7.15443 at cells_code position 280
- max absolute logit difference: 29.63
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1239 | 92.74 | 94.37 | 0.03847 | 3.052 | 1237 | 15.3 |
| cells_code | 1178 | 1129 | 95.84 | 96.30 | 0.1054 | 7.154 | 280 | 29.6 |
| greeting | 25 | 25 | 100.00 | 100.00 | 0.003623 | 0.02623 | 23 | 1.87 |
| explain_sliding | 34 | 32 | 94.12 | 93.75 | 0.006069 | 0.05565 | 7 | 4.55 |
| unit_test | 32 | 30 | 93.75 | 96.77 | 0.03931 | 0.9266 | 28 | 8.01 |
| refactor | 36 | 34 | 94.44 | 94.29 | 0.01013 | 0.07788 | 34 | 6.54 |
| math_mod | 36 | 35 | 97.22 | 97.06 | 0.003656 | 0.05542 | 7 | 4.54 |
| math_frac | 44 | 42 | 95.45 | 97.62 | 0.01019 | 0.07646 | 31 | 5.21 |
| dialogue | 44 | 43 | 97.73 | 97.67 | 0.004397 | 0.05607 | 7 | 4.55 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 0.005372 | 0.02581 | 26 | 2.53 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.007784 | 0.05574 | 7 | 4.55 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.005972 | 0.05572 | 7 | 4.54 |
| sql | 33 | 32 | 96.97 | 96.77 | 0.04379 | 1.009 | 31 | 15.8 |
| prose_review | 35 | 35 | 100.00 | 100.00 | 0.006061 | 0.0556 | 7 | 4.55 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0.006959 | 0.05605 | 7 | 4.54 |
| token_budget | 37 | 36 | 97.30 | 97.22 | 0.008931 | 0.05595 | 7 | 4.68 |

Lowest top-1: harbour (92.74%), unit_test (93.75%), explain_sliding (94.12%).
Highest mean KL: cells_code (0.1054), sql (0.04379), unit_test (0.03931).

Baseline c053f52b2: top-1 2861/3015, mean KL 0.0195085, row-weighted 0.0641971, max KL 8.01205 (reported, not gated).
Thresholds: top-1 at least 94.392%, mean KL at most 0.0292627, row-weighted mean KL at most 0.0962956, no non-finite logits.

omlx-oracle gate: pass
