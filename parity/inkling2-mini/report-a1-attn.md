# Inkling 2 Mini parity: Metal against the float32 reference

Arithmetic: NVFP4 and MXFP8 weights decoded from the checkpoint's own bytes, float32 activations. This is
not the official activation-quantized path and no equivalence with it is claimed.

## Summary

- positions compared: 3015 over 16 histories (2 past the 1,024-token sliding window)
- mean KL: 4.21473e-08
- top-1 agreement overall: 100.00%
- top-1 agreement on positions with reference top-2 margin >= 0.1: 100.00% over 2862 positions (required >= 99.5%)
- absolute logit error: p50 5.245e-05, p90 1.149e-03, p99 8.554e-03, max 6.775e-01
- relative logit error: p50 2.353e-06, p90 4.794e-05, p99 3.755e-04, max 2.482e-02

## Per history

| history | tokens | max abs error | mean KL | top-1 | confident top-1 | confident disagreements |
| --- | --- | --- | --- | --- | --- | --- |
| harbour | 1336 | 2.787e-01 | 5.915e-07 | 100.00% | 100.00% | 0 |
| cells_code | 1178 | 6.775e-01 | 8.228e-08 | 100.00% | 100.00% | 0 |
| greeting | 25 | 8.583e-05 | 3.856e-11 | 100.00% | 100.00% | 0 |
| explain_sliding | 34 | 1.040e-04 | 4.760e-11 | 100.00% | 100.00% | 0 |
| unit_test | 32 | 1.092e-04 | 4.302e-11 | 100.00% | 100.00% | 0 |
| refactor | 36 | 1.235e-04 | 4.475e-11 | 100.00% | 100.00% | 0 |
| math_mod | 36 | 1.049e-04 | 2.721e-11 | 100.00% | 100.00% | 0 |
| math_frac | 44 | 1.173e-04 | 3.471e-11 | 100.00% | 100.00% | 0 |
| dialogue | 44 | 1.154e-04 | 3.063e-11 | 100.00% | 100.00% | 0 |
| list_steps | 30 | 1.135e-04 | 3.418e-11 | 100.00% | 100.00% | 0 |
| structured | 35 | 9.346e-05 | 5.391e-11 | 100.00% | 100.00% | 0 |
| summary | 35 | 9.632e-05 | 5.001e-11 | 100.00% | 100.00% | 0 |
| sql | 33 | 1.206e-04 | 3.866e-11 | 100.00% | 100.00% | 0 |
| prose_review | 35 | 1.540e-04 | 4.392e-11 | 100.00% | 100.00% | 0 |
| code_trace | 45 | 1.073e-04 | 3.316e-11 | 100.00% | 100.00% | 0 |
| token_budget | 37 | 9.394e-05 | 3.582e-11 | 100.00% | 100.00% | 0 |

## Divergences

Every position agrees on top-1.
