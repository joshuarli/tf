# Inkling 2 Mini parity: Metal against the float32 reference

Arithmetic: NVFP4 and MXFP8 weights decoded from the checkpoint's own bytes, float32 activations. This is
not the official activation-quantized path and no equivalence with it is claimed.

## Summary

- positions compared: 3015 over 16 histories (2 past the 1,024-token sliding window)
- mean KL: 5.18253e-09
- top-1 agreement overall: 100.00%
- top-1 agreement on positions with reference top-2 margin >= 0.1: 100.00% over 2862 positions (required >= 99.5%)
- absolute logit error: p50 5.150e-05, p90 3.594e-04, p99 2.930e-03, max 6.775e-01
- relative logit error: p50 2.310e-06, p90 1.387e-05, p99 1.187e-04, max 2.482e-02

## Per history

| history | tokens | max abs error | mean KL | top-1 | confident top-1 | confident disagreements |
| --- | --- | --- | --- | --- | --- | --- |
| harbour | 1336 | 2.761e-03 | 1.469e-10 | 100.00% | 100.00% | 0 |
| cells_code | 1178 | 6.775e-01 | 8.229e-08 | 100.00% | 100.00% | 0 |
| greeting | 25 | 8.392e-05 | 3.251e-11 | 100.00% | 100.00% | 0 |
| explain_sliding | 34 | 9.918e-05 | 4.121e-11 | 100.00% | 100.00% | 0 |
| unit_test | 32 | 1.125e-04 | 3.603e-11 | 100.00% | 100.00% | 0 |
| refactor | 36 | 1.268e-04 | 3.113e-11 | 100.00% | 100.00% | 0 |
| math_mod | 36 | 1.059e-04 | 2.469e-11 | 100.00% | 100.00% | 0 |
| math_frac | 44 | 1.206e-04 | 2.698e-11 | 100.00% | 100.00% | 0 |
| dialogue | 44 | 1.187e-04 | 2.744e-11 | 100.00% | 100.00% | 0 |
| list_steps | 30 | 1.173e-04 | 3.640e-11 | 100.00% | 100.00% | 0 |
| structured | 35 | 9.251e-05 | 4.598e-11 | 100.00% | 100.00% | 0 |
| summary | 35 | 9.727e-05 | 4.163e-11 | 100.00% | 100.00% | 0 |
| sql | 33 | 1.144e-04 | 4.104e-11 | 100.00% | 100.00% | 0 |
| prose_review | 35 | 1.574e-04 | 4.112e-11 | 100.00% | 100.00% | 0 |
| code_trace | 45 | 1.009e-04 | 2.332e-11 | 100.00% | 100.00% | 0 |
| token_budget | 37 | 9.155e-05 | 3.350e-11 | 100.00% | 100.00% | 0 |

## Divergences

### harbour onset position 1009: largest logit error 0.002761, KL 8.877e-08
- first divergent tensor: mixin-33-1: reference (mean 0.0745012, std 1.40933, min -15.6863, max 33.5561) against Metal (mean 0.0745012, std 1.40937, min -15.687, max 33.5583), largest difference 2.273e-03 over 245 overlapping tensors
- clean control at 1008: l_out-47: reference (mean 0.0165318, std 4.87425, min -25.1202, max 57.767) against Metal (mean 0.0165344, std 4.87423, min -25.1198, max 57.7688), largest difference 1.712e-03 over 378 overlapping tensors
- tightest reference router 8th/9th gaps at 1009: slot 10 (layer 10): picks [85, 61, 121, 1, 71, 60, 125, 54] against 9th 35, 8th/9th selection gap 0.000142 (prob gap -0.106); slot 28 (layer 10): picks [61, 127, 35, 60, 88, 71, 85, 90] against 9th 56, 8th/9th selection gap 0.000264 (prob gap 0.104); slot 36 (layer 6): picks [20, 98, 28, 115, 23, 62, 106, 64] against 9th 76, 8th/9th selection gap 0.00062 (prob gap -0.000431)

### cells_code onset position 690: largest logit error 0.6775, KL 9.007e-05
- first divergent tensor: every one of 432 overlapping tensors agrees within 0.001
- clean control at 689: every one of 432 overlapping tensors agrees within 0.001
- tightest reference router 8th/9th gaps at 690: slot 21 (layer 21): picks [97, 24, 53, 72, 43, 17, 85, 125] against 9th 120, 8th/9th selection gap 1.64e-07 (prob gap 0.0164); slot 2 (layer 2): picks [104, 45, 9, 32, 66, 5, 33, 51] against 9th 31, 8th/9th selection gap 0.000143 (prob gap 0.00202); slot 20 (layer 20): picks [127, 31, 81, 46, 26, 23, 117, 57] against 9th 62, 8th/9th selection gap 0.000391 (prob gap 0.00881)
