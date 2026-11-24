# Inkling 2 Mini: TensorFold against the oMLX oracle

- oracle: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-ref (digest b01512f0396eeb23, oMLX 0.8.3 e72868dac12d on MLX 0.32.4, 2,048-token forwards, MTP off)
- candidate: /Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-gate/lane_p4-prefill-wip-experts-prefill-512
- runner sha256: dad7a356aac51214b6b383fdf125c562e094936afd96395921efbcc7a6de8fb5
- model identity: 7b1841c868807cd5df91a76646f550f6b0bd2ff803f4078b1009d540fd34195a
- arithmetic policy: f32-activations+prefill-512
- positions: 3015 over 16 histories
- top-1 agreement: 2872/3015 (95.26%); 50 reference rows tie for the maximum
- top-1 agreement where the reference's top-2 margin is at least 0.1: 2760/2872 (96.10%)
- mean KL, mean of history means: 0.0185822; row-weighted: 0.0596688
- max KL: 8.0127 at cells_code position 280
- max absolute logit difference: 26.6
- non-finite logits: reference 0, candidate 0

| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| harbour | 1336 | 1251 | 93.64 | 95.11 | 0.03937 | 5.887 | 1237 | 19.4 |
| cells_code | 1178 | 1132 | 96.10 | 96.39 | 0.1035 | 8.013 | 280 | 26.6 |
| greeting | 25 | 25 | 100.00 | 100.00 | 0.003546 | 0.02623 | 23 | 1.86 |
| explain_sliding | 34 | 31 | 91.18 | 90.62 | 0.00623 | 0.05572 | 7 | 4.54 |
| unit_test | 32 | 30 | 93.75 | 96.77 | 0.03921 | 1.021 | 28 | 8.25 |
| refactor | 36 | 34 | 94.44 | 94.29 | 0.008503 | 0.05572 | 7 | 6.53 |
| math_mod | 36 | 35 | 97.22 | 97.06 | 0.003522 | 0.05572 | 7 | 4.54 |
| math_frac | 44 | 42 | 95.45 | 97.62 | 0.008822 | 0.08851 | 31 | 5.39 |
| dialogue | 44 | 44 | 100.00 | 100.00 | 0.004223 | 0.05572 | 7 | 4.54 |
| list_steps | 30 | 30 | 100.00 | 100.00 | 0.005365 | 0.0258 | 26 | 2.53 |
| structured | 35 | 35 | 100.00 | 100.00 | 0.005782 | 0.05572 | 7 | 4.54 |
| summary | 35 | 35 | 100.00 | 100.00 | 0.005973 | 0.05572 | 7 | 4.54 |
| sql | 33 | 32 | 96.97 | 96.77 | 0.04378 | 1.009 | 31 | 15.8 |
| prose_review | 35 | 35 | 100.00 | 100.00 | 0.005026 | 0.05572 | 7 | 4.54 |
| code_trace | 45 | 45 | 100.00 | 100.00 | 0.004586 | 0.05572 | 7 | 4.54 |
| token_budget | 37 | 36 | 97.30 | 97.22 | 0.009833 | 0.1107 | 30 | 4.64 |

Lowest top-1: explain_sliding (91.18%), harbour (93.64%), unit_test (93.75%).
Highest mean KL: cells_code (0.1035), sql (0.04378), harbour (0.03937).

Baseline c053f52b2: top-1 2861/3015, mean KL 0.0195085, row-weighted 0.0641971, max KL 8.01205 (reported, not gated).
Thresholds: top-1 at least 94.392%, mean KL at most 0.0292627, row-weighted mean KL at most 0.0962956, no non-finite logits.

omlx-oracle gate: pass
