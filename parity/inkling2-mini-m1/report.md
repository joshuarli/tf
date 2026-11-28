# Inkling 2 Mini parity: Metal against the float32 reference

Arithmetic: NVFP4 and MXFP8 weights decoded from the checkpoint's own bytes, float32 activations. This is
not the official activation-quantized path and no equivalence with it is claimed.

## Summary

- positions compared: 3015 over 16 histories (2 past the 1,024-token sliding window)
- mean KL: 8.02251e-05
- top-1 agreement overall: 99.67%
- top-1 agreement on positions with reference top-2 margin >= 0.1: 99.79% over 2862 positions (required >= 99.5%)
- absolute logit error: p50 7.105e-05, p90 3.891e-02, p99 8.394e-01, max 6.038e+00
- relative logit error: p50 2.900e-06, p90 1.881e-03, p99 3.957e-02, max 2.949e-01

## Per history

| history | tokens | max abs error | mean KL | top-1 | confident top-1 | confident disagreements |
| --- | --- | --- | --- | --- | --- | --- |
| harbour | 1336 | 6.038e+00 | 1.284e-03 | 99.25% | 99.51% | 6 |
| cells_code | 1178 | 6.775e-01 | 8.231e-08 | 100.00% | 100.00% | 0 |
| greeting | 25 | 8.583e-05 | 3.727e-11 | 100.00% | 100.00% | 0 |
| explain_sliding | 34 | 9.918e-05 | 4.246e-11 | 100.00% | 100.00% | 0 |
| unit_test | 32 | 1.068e-04 | 3.860e-11 | 100.00% | 100.00% | 0 |
| refactor | 36 | 1.154e-04 | 3.158e-11 | 100.00% | 100.00% | 0 |
| math_mod | 36 | 1.049e-04 | 2.745e-11 | 100.00% | 100.00% | 0 |
| math_frac | 44 | 1.001e-04 | 2.713e-11 | 100.00% | 100.00% | 0 |
| dialogue | 44 | 9.823e-05 | 2.779e-11 | 100.00% | 100.00% | 0 |
| list_steps | 30 | 1.154e-04 | 3.490e-11 | 100.00% | 100.00% | 0 |
| structured | 35 | 8.965e-05 | 4.534e-11 | 100.00% | 100.00% | 0 |
| summary | 35 | 9.155e-05 | 4.173e-11 | 100.00% | 100.00% | 0 |
| sql | 33 | 1.063e-04 | 4.105e-11 | 100.00% | 100.00% | 0 |
| prose_review | 35 | 1.369e-04 | 4.489e-11 | 100.00% | 100.00% | 0 |
| code_trace | 45 | 9.894e-05 | 2.301e-11 | 100.00% | 100.00% | 0 |
| token_budget | 37 | 9.274e-05 | 3.309e-11 | 100.00% | 100.00% | 0 |

## Divergences

### harbour position 588: reference top-1 149859, Metal top-1 196993
- reference top-2 margin 0.0392, largest logit error 0.1222, KL 0.0002644
- first divergent tensor: Qcur-0: absent from the Metal trace

### harbour position 654: reference top-1 149859, Metal top-1 196993
- reference top-2 margin 0.2136, largest logit error 1.662, KL 0.01686
- first divergent tensor: Qcur-0: absent from the Metal trace

### harbour position 657: reference top-1 196993, Metal top-1 220
- reference top-2 margin 0.3334, largest logit error 5.04, KL 0.1721
- first divergent tensor: Qcur-0: absent from the Metal trace

### harbour position 782: reference top-1 68014, Metal top-1 197001
- reference top-2 margin 1.383, largest logit error 2.908, KL 0.2032
- first divergent tensor: Qcur-0: absent from the Metal trace

### harbour position 924: reference top-1 220, Metal top-1 197017
- reference top-2 margin 0.0003796, largest logit error 0.01442, KL 1.433e-06
- first divergent tensor: Qcur-0: absent from the Metal trace

### harbour position 1058: reference top-1 196993, Metal top-1 150382
- reference top-2 margin 0.05386, largest logit error 0.4063, KL 0.002157
- first divergent tensor: Qcur-0: absent from the Metal trace

### harbour position 1080: reference top-1 25, Metal top-1 196993
- reference top-2 margin 0.341, largest logit error 6.038, KL 0.1984
- first divergent tensor: Qcur-0: absent from the Metal trace

### harbour position 1171: reference top-1 197017, Metal top-1 196993
- reference top-2 margin 0.102, largest logit error 0.3827, KL 0.001882
- first divergent tensor: Qcur-0: absent from the Metal trace

### harbour position 1187: reference top-1 196993, Metal top-1 68014
- reference top-2 margin 0.2184, largest logit error 2.688, KL 0.04717
- first divergent tensor: Qcur-0: absent from the Metal trace

### harbour position 1268: reference top-1 220, Metal top-1 154275
- reference top-2 margin 0.07067, largest logit error 0.305, KL 0.001604
- first divergent tensor: Qcur-0: absent from the Metal trace
