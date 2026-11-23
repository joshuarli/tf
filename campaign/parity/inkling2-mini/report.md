# Inkling 2 Mini parity: Metal against the float32 reference

**Superseded record.** The f32-oracle pipeline that produced this was retired on 2026-11-23 (plan Decision 9), and its tools are gone. The oMLX oracle ([inkling-omlx-oracle.md](../../kernels/inkling-omlx-oracle.md)) replaces it.

Arithmetic: NVFP4 and MXFP8 weights decoded from the checkpoint's own bytes, float32 activations. This is
not the official activation-quantized path and no equivalence with it is claimed.

Contract: f32-vs-f32 over a 200058-token vocabulary: per-logit max abs 0.001, mean abs 0.0001, per-position max KL 1e-06, mean KL 1e-08, teacher top-2 margin 0.1 with confident top-1 100%.

## Summary

- positions compared: 3015 over 16 histories (2 past the 1,024-token sliding window)
- mean KL: 8.02252e-05
- top-1 agreement overall: 99.67%
- top-1 agreement on positions with reference top-2 margin >= 0.1: 99.79% over 2862 positions
- absolute logit error: p50 7.093e-05, p90 3.891e-02, p99 8.393e-01, max 6.038e+00
- relative logit error: p50 2.911e-06, p90 1.882e-03, p99 3.957e-02, max 2.949e-01
- qualification: 14 within contract, 2 explained divergences, 0 unexplained

## Per history

| history | tokens | max abs error | mean KL | top-1 | confident top-1 | confident | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| harbour | 1336 | 6.038e+00 | 1.284e-03 | 99.25% | 99.51% | 1214 | explained divergence |
| cells_code | 1178 | 6.775e-01 | 8.227e-08 | 100.00% | 100.00% | 1164 | explained divergence |
| greeting | 25 | 7.725e-05 | 3.577e-11 | 100.00% | 100.00% | 24 | within contract |
| explain_sliding | 34 | 9.060e-05 | 4.157e-11 | 100.00% | 100.00% | 30 | within contract |
| unit_test | 32 | 1.030e-04 | 3.762e-11 | 100.00% | 100.00% | 31 | within contract |
| refactor | 36 | 1.183e-04 | 3.207e-11 | 100.00% | 100.00% | 34 | within contract |
| math_mod | 36 | 1.001e-04 | 2.551e-11 | 100.00% | 100.00% | 36 | within contract |
| math_frac | 44 | 1.073e-04 | 2.898e-11 | 100.00% | 100.00% | 42 | within contract |
| dialogue | 44 | 1.054e-04 | 2.820e-11 | 100.00% | 100.00% | 43 | within contract |
| list_steps | 30 | 1.097e-04 | 3.444e-11 | 100.00% | 100.00% | 29 | within contract |
| structured | 35 | 8.249e-05 | 4.617e-11 | 100.00% | 100.00% | 35 | within contract |
| summary | 35 | 8.869e-05 | 4.101e-11 | 100.00% | 100.00% | 34 | within contract |
| sql | 33 | 1.027e-04 | 4.000e-11 | 100.00% | 100.00% | 31 | within contract |
| prose_review | 35 | 1.440e-04 | 4.292e-11 | 100.00% | 100.00% | 35 | within contract |
| code_trace | 45 | 9.418e-05 | 2.474e-11 | 100.00% | 100.00% | 45 | within contract |
| token_budget | 37 | 9.227e-05 | 3.243e-11 | 100.00% | 100.00% | 35 | within contract |

## Provenance

- runner sha256: 4bc68d5975e5913b453f83149ace11468ac25d9038ac8b7009ff1a12f9deae78
- model identity: 27dd9e1ad71c134fbe909b9a74c1102c8589ccdf08e12f4c9eeef185f99b6c1c
- arithmetic policy: f32-activations
- Open provenance gap: campaign/ref/oracle_receipts.json records output digests only for the ref-check outputs, not for these parity histories' reference .npy files; the reference files are verified by oracle.py shape/finiteness/token-count checks, and their observed digests are recorded below. The current build identity is not attached to those older files.

| history | token-history sha256 | reference sha256 | candidate receipt |
| --- | --- | --- | --- |
| harbour | d1d16573335e0247c7cd5589b2db692318e02b079b1616ff3dd961293d0a8541 | 097d3cea55506781f0d66fa9668bfaf78277a7ec4f676a638acb798dff9a325b | matches |
| cells_code | 372e9686994e48a8d41202401e263c9f16953d92ecf89d569849edcaf5612fc3 | 8fb4a4f2d156906f9ff1bf95188dff32bf02b154b182d1ada3d89a23fe9e940d | matches |
| greeting | 997a527596cabdf4858d18d6e7b5a953e3ba1b06a9636de94d5b6dbc9167c469 | f031d19e5b250ae0a6a497a6e0d4f3bac0500d2ddc84515de78c01e7c7cc5b63 | matches |
| explain_sliding | 0cde54b009b7661b29a7219fecdad568527a92dfa199c220ac068ea6db71a3f5 | 00355ec8fc5d5709f3a4db2a345209c02f8b05691db0f1cea4157da0681634f9 | matches |
| unit_test | 394ca375bf7d28f880093fb8d995b04cbde94ba9ba1bb8bda8513071dadeec75 | 624f5c498887c1ad46137a2b4b895ac62cccec6cf5921c856133f08c7116bef3 | matches |
| refactor | 06c704ea3e3d963ff0c98bd2af4005fb5a638cfd16fd3e584c5825e37f6ea12a | 266961b4da7d8c9f2224fc8cc1b28cd92add6857e79f3f52981323d1bd9fa50f | matches |
| math_mod | 4cc90e7f706d995a60953767dabd53d40222e7667b47556d3b108da8bce3bc69 | d7a1034ac146d23eadaec74e206abf2ff763e2d8a2e2605971735db7b905142f | matches |
| math_frac | 6a08f516e6213f53c2509c00e8c3831cc763408332c973dc36416510a44a92e2 | 3765746b2b8a4a93eaaa2a48dba0e8a930d5528f9b25072b90ea37f612a8edb8 | matches |
| dialogue | 3bfdc4a3e888da23a1ce69893e0d4633324b4411c6090967b25cd4d9aa5cdf54 | 040f999800937eb8be904f473ee67971692a11ca2cf7665571d9f482393f1dbe | matches |
| list_steps | a87a3aeba14c085076adef58547e93cdbc815cd0eee2beb4a7cca31475c98e8b | c9ffa156179974cc9b607a924c4637640a5002f02f6f6fd64a77f1f0c78e2b05 | matches |
| structured | 41e9f0c258a7f04ff55a8424b218e8a828ada7cbce51f0bd6b5f864f1d64dc98 | 4656c1fc6a92eedb9898cf78dcd2b42ac88e2647923c401ba541c129dc5e70c0 | matches |
| summary | 42cefb3f758fc46f8f1c726fbd8d0a6fd1e8de270a208aecd6d707fc89267a1e | d64f0b6d8f5cc23a217b968e8eb03cf155cc9ebbe18714451f0783af38b8d167 | matches |
| sql | 7604828fe042c51cf68fdb5aa05906cfacfeb0559b220f648b4403a5b2e206f4 | 57f9b0c4a4a1d51f4bdc223f322c97bacbc0ad952dd1605a2c1aed030ebece9f | matches |
| prose_review | 2b1a6566ed8a94ba6345593a621df4603d50171f8402054a1c77f5e653da7130 | d7f3e332f01323535af19c2bc3902d692b15739fa397a86eddbbd8b5cd5fe837 | matches |
| code_trace | e6e737faca38b83146c7b91fe65dcc8c0523e20850eda396f889800633871af1 | 55e8c6aeea7fc6845af13e9c540d82bb1fccd00d99ffec2f30092570a0733ef3 | matches |
| token_budget | 7db090b6e3ac721eff94f31301ee11f8d1bda4aaab7f552e0998eeb990d257b0 | 27f737a73c94babe19f2aee35440f5c02163a4e5277b6fcf2422b83f58221a9b | matches |

## Divergences

### harbour gate problems: harbour: confident top-1 99.5058% < 100.0000% at positions [654, 657, 782, 1080, 1171, 1187]; harbour: max_abs 6.038 exceeds the f32-vs-f32 bound 0.001; harbour: mean_abs 0.01669 exceeds the f32-vs-f32 bound 0.0001; harbour: max_position_kl 0.3936 exceeds the f32-vs-f32 bound 1e-06; harbour: mean_kl 0.001284 exceeds the f32-vs-f32 bound 1e-08

### harbour outliers: position 1080 (max abs 6.038e+00, KL 1.984e-01), position 991 (max abs 5.878e+00, KL 3.155e-05), position 657 (max abs 5.040e+00, KL 1.721e-01), position 943 (max abs 4.132e+00, KL 1.329e-01), position 623 (max abs 3.247e+00, KL 3.936e-01)

### harbour position 588: reference top-1 149859, Metal top-1 196993
- reference top-2 margin 0.0392, largest logit error 0.1222

### harbour position 654: reference top-1 149859, Metal top-1 196993
- reference top-2 margin 0.2136, largest logit error 1.662

### harbour position 657: reference top-1 196993, Metal top-1 220
- reference top-2 margin 0.3334, largest logit error 5.04

### harbour position 782: reference top-1 68014, Metal top-1 197001
- reference top-2 margin 1.383, largest logit error 2.908

### harbour position 924: reference top-1 220, Metal top-1 197017
- reference top-2 margin 0.0003796, largest logit error 0.01442

### harbour position 1058: reference top-1 196993, Metal top-1 150382
- reference top-2 margin 0.05386, largest logit error 0.4063

### harbour position 1080: reference top-1 25, Metal top-1 196993
- reference top-2 margin 0.341, largest logit error 6.038

### harbour position 1171: reference top-1 197017, Metal top-1 196993
- reference top-2 margin 0.102, largest logit error 0.3827

### harbour: 2 further top-1 disagreements

### harbour onset position 557: largest logit error 0.06164, KL 1.964e-05
- first divergent tensor: l_out-24: reference (mean 0.0106271, std 0.699803, min -12.4817, max 19.3329) against Metal (mean 0.0103036, std 0.703953, min -12.4828, max 19.7603), largest difference 4.273e-01 over 171 overlapping tensors
- clean control at 556: l_out-38: reference (mean 0.0286352, std 1.84985, min -25.7621, max 54.6575) against Metal (mean 0.0285382, std 1.85234, min -25.9324, max 54.7187), largest difference 1.703e-01 over 297 overlapping tensors
- tightest reference router 8th/9th gaps at 557: slot 24 (layer 6): picks [67, 33, 66, 50, 78, 64, 23, 125] against 9th 22, 8th/9th selection gap 1.34e-07 (prob gap 0.00997); slot 17 (layer 17): picks [95, 7, 124, 110, 111, 72, 104, 43] against 9th 4, 8th/9th selection gap 3.15e-05 (prob gap -0.0219); slot 28 (layer 10): picks [61, 92, 59, 102, 118, 11, 19, 97] against 9th 98, 8th/9th selection gap 0.000372 (prob gap -0.219)

### cells_code gate problems: cells_code: max_abs 0.6775 exceeds the f32-vs-f32 bound 0.001; cells_code: mean_abs 0.0001648 exceeds the f32-vs-f32 bound 0.0001; cells_code: max_position_kl 9.008e-05 exceeds the f32-vs-f32 bound 1e-06; cells_code: mean_kl 8.227e-08 exceeds the f32-vs-f32 bound 1e-08

### cells_code outliers: position 690 (max abs 6.775e-01, KL 9.008e-05), position 691 (max abs 7.433e-02, KL 2.964e-06), position 783 (max abs 1.944e-02, KL 4.304e-07), position 692 (max abs 1.644e-02, KL 1.168e-07), position 699 (max abs 9.602e-03, KL 2.160e-09)

### cells_code onset position 656: largest logit error 0.001057, KL 1.499e-09
- first divergent tensor: every one of 432 overlapping tensors agrees within 0.001
- clean control at 655: every one of 432 overlapping tensors agrees within 0.001
- tightest reference router 8th/9th gaps at 656: slot 19 (layer 19): picks [67, 103, 81, 52, 45, 37, 61, 40] against 9th 51, 8th/9th selection gap 3.25e-05 (prob gap 0.0179); slot 39 (layer 9): picks [21, 39, 93, 65, 49, 45, 102, 78] against 9th 62, 8th/9th selection gap 0.00053 (prob gap -0.0116); slot 22 (layer 22): picks [89, 68, 90, 123, 118, 124, 13, 75] against 9th 62, 8th/9th selection gap 0.000586 (prob gap 0.00993)

### greeting outliers: position 7 (max abs 7.725e-05, KL 1.111e-10), position 6 (max abs 7.629e-05, KL 1.181e-10), position 8 (max abs 7.629e-05, KL 1.157e-12), position 5 (max abs 7.057e-05, KL 7.579e-11), position 10 (max abs 6.866e-05, KL 5.857e-14)

### explain_sliding outliers: position 8 (max abs 9.060e-05, KL 1.363e-11), position 10 (max abs 8.869e-05, KL 4.328e-14), position 5 (max abs 7.582e-05, KL 8.353e-11), position 7 (max abs 7.319e-05, KL 6.632e-11), position 15 (max abs 6.711e-05, KL 4.006e-11)

### unit_test outliers: position 8 (max abs 1.030e-04, KL 1.479e-11), position 5 (max abs 9.918e-05, KL 4.271e-11), position 15 (max abs 8.750e-05, KL 9.011e-11), position 4 (max abs 7.820e-05, KL 2.468e-12), position 7 (max abs 7.153e-05, KL 7.097e-11)

### refactor outliers: position 24 (max abs 1.183e-04, KL 1.378e-11), position 5 (max abs 1.135e-04, KL 5.274e-11), position 7 (max abs 1.044e-04, KL 6.342e-11), position 8 (max abs 8.488e-05, KL 8.264e-12), position 10 (max abs 8.011e-05, KL 5.483e-14)

### math_mod outliers: position 8 (max abs 1.001e-04, KL 1.740e-11), position 10 (max abs 8.297e-05, KL 2.691e-14), position 7 (max abs 8.202e-05, KL 8.062e-11), position 5 (max abs 6.962e-05, KL 4.201e-11), position 9 (max abs 6.676e-05, KL 5.150e-11)

### math_frac outliers: position 5 (max abs 1.073e-04, KL 1.023e-10), position 7 (max abs 8.488e-05, KL 6.146e-11), position 8 (max abs 8.297e-05, KL 1.457e-11), position 9 (max abs 7.010e-05, KL 4.952e-11), position 19 (max abs 6.866e-05, KL 1.480e-11)

### dialogue outliers: position 5 (max abs 1.054e-04, KL 6.888e-11), position 6 (max abs 7.057e-05, KL 1.766e-11), position 9 (max abs 6.962e-05, KL 5.228e-11), position 8 (max abs 6.962e-05, KL 5.493e-13), position 7 (max abs 6.771e-05, KL 3.616e-11)

### list_steps outliers: position 8 (max abs 1.097e-04, KL 2.114e-11), position 7 (max abs 8.106e-05, KL 4.936e-11), position 10 (max abs 7.343e-05, KL 2.162e-14), position 9 (max abs 7.248e-05, KL 4.575e-11), position 5 (max abs 7.153e-05, KL 3.996e-11)

### structured outliers: position 8 (max abs 8.249e-05, KL 1.164e-11), position 7 (max abs 8.130e-05, KL 2.551e-10), position 9 (max abs 7.772e-05, KL 3.691e-11), position 10 (max abs 7.343e-05, KL 4.879e-15), position 5 (max abs 6.962e-05, KL 7.534e-11)

### summary outliers: position 8 (max abs 8.869e-05, KL 2.234e-12), position 6 (max abs 8.059e-05, KL 4.416e-11), position 10 (max abs 7.820e-05, KL 7.215e-14), position 4 (max abs 7.629e-05, KL 5.548e-12), position 7 (max abs 7.439e-05, KL 1.161e-10)

### sql outliers: position 15 (max abs 1.027e-04, KL 8.940e-11), position 10 (max abs 1.011e-04, KL 2.163e-14), position 8 (max abs 9.918e-05, KL 1.239e-11), position 5 (max abs 9.608e-05, KL 7.424e-11), position 16 (max abs 8.488e-05, KL 1.579e-10)

### prose_review outliers: position 5 (max abs 1.440e-04, KL 5.501e-11), position 8 (max abs 8.488e-05, KL 1.149e-12), position 7 (max abs 7.248e-05, KL 9.233e-11), position 9 (max abs 6.723e-05, KL 4.916e-11), position 6 (max abs 6.676e-05, KL 1.601e-10)

### code_trace outliers: position 10 (max abs 9.418e-05, KL 5.389e-14), position 8 (max abs 9.274e-05, KL 1.079e-11), position 7 (max abs 8.869e-05, KL 1.061e-10), position 5 (max abs 8.273e-05, KL 9.774e-11), position 9 (max abs 6.476e-05, KL 3.469e-11)

### token_budget outliers: position 7 (max abs 9.227e-05, KL 4.649e-11), position 8 (max abs 8.583e-05, KL 2.989e-12), position 6 (max abs 7.725e-05, KL 4.314e-11), position 10 (max abs 7.677e-05, KL 2.342e-14), position 9 (max abs 7.343e-05, KL 4.116e-11)

## Verdict

parity: not qualified: 14 within contract, 2 explained divergences, 0 unexplained

Qualification is incomplete: every failing history above must return to within contract.

Recorded explanations (evidence only; none of these turns a failure into a pass):
- harbour: recorded onset 556, observed onset in this run 557; this run's observed onset differs from the recorded one; demonstrated routing discontinuity: the Metal pass takes the other side of reference router ties (slot 38 at 556 with an 8th/9th selection gap of 2.98e-08, slot 24 at 557 with a gap of 1.34e-07); forcing the reference's 8th picks reproduces Metal's logits to 9.1e-05 and 4.6e-05, and the corrupted state then persists (evidence: campaign/parity/inkling2-mini/long-divergence.md)
- cells_code: recorded onset 304, observed onset in this run 656; this run's observed onset differs from the recorded one; demonstrated routing discontinuity plus drift amplification: onset at 304 is MoE forward amplification of drift (all eight slot-47 inputs agree to <=5.9e-04 while l_out-47 differs on 40 elements up to 1.5e-03); row 656 is diffuse drift under 1e-3 on the same discrete path (prefix-matched sides agree on all 432x2304 values); reference-side router ties at slot 21, row 473 (gap ~3e-08, --force-pick 21,89,473 reproduces the full-history row to 4.1e-05) and row 690 (gap ~1.6e-07, --force-pick 21,120,690 reproduces the oracle row to 9.2e-05), with the 690 tie's state echo decaying through 694 (evidence: campaign/parity/inkling2-mini/long-divergence.md)

