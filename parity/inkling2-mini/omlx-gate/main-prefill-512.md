# Inkling 2 Mini: TensorFold against the oMLX oracle

- oracle: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-ref (digest b01512f0396eeb23, oMLX 0.8.3 e72868dac12d on MLX 0.32.4, 2,048-token forwards, MTP off)
- candidate: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/main-prefill-512
- runner sha256: 2b015b96f75fbdb99488bf52e07253ba898cc8a9bd81f00f306941865f867287
- model identity: 7b1841c868807cd5df91a76646f550f6b0bd2ff803f4078b1009d540fd34195a
- arithmetic policy: f32-activations+prefill-512
- positions: 3015 over 16 histories
- top-1 agreement: 2868/3015 (95.12%); 50 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2757/2872 (96.00%)
- mean KL, mean of history means: 0.0194447; row-weighted: 0.0648103
- max KL: 10.6673 at cells_code position 448
- max absolute logit difference: 29.97
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1249 | 93.49 | 95.02 | 0.03822 | 2.89 | 961 | 19.3 |
| cells_code | 1178 | 1131 | 96.01 | 96.22 | 0.118 | 10.67 | 448 | 30 |
| greeting | 25 | 25 | 100.00 | 100.00 | 0.003545 | 0.02597 | 23 | 1.87 |
| explain_sliding | 34 | 31 | 91.18 | 90.62 | 0.006239 | 0.05559 | 7 | 4.54 |
| unit_test | 32 | 30 | 93.75 | 96.77 | 0.03921 | 0.9236 | 28 | 8 |
| refactor | 36 | 34 | 94.44 | 94.29 | 0.01185 | 0.1414 | 34 | 6.54 |
| math_mod | 36 | 35 | 97.22 | 97.06 | 0.003868 | 0.05559 | 7 | 4.54 |
| math_frac | 44 | 42 | 95.45 | 97.62 | 0.008845 | 0.0887 | 31 | 5.39 |
| dialogue | 44 | 43 | 97.73 | 97.67 | 0.004405 | 0.05559 | 7 | 4.54 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 0.005382 | 0.0258 | 26 | 2.53 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.005759 | 0.05559 | 7 | 4.54 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.00597 | 0.05559 | 7 | 4.54 |
| sql | 33 | 32 | 96.97 | 96.77 | 0.04157 | 1.012 | 31 | 15.8 |
| prose_review | 35 | 34 | 97.14 | 100.00 | 0.005421 | 0.05559 | 7 | 4.54 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0.004587 | 0.05559 | 7 | 4.54 |
| token_budget | 37 | 37 | 100.00 | 100.00 | 0.008258 | 0.1123 | 30 | 4.65 |

Lowest top-1: explain_sliding (91.18%), harbour (93.49%), unit_test (93.75%).
Highest mean KL: cells_code (0.118), sql (0.04157), unit_test (0.03921).

Baseline c053f52b2: top-1 2861/3015, mean KL 0.0195085, row-weighted 0.0641971, max KL 8.01205 (reported, not gated).
Thresholds: top-1 at least 94.392%, mean KL at most 0.0292627, row-weighted mean KL at most 0.0962956, no non-finite logits.

omlx-oracle gate: pass
