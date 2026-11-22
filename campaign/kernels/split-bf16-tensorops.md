# Split-bf16 TensorOps G0

## Verdict

**G0 fails.** This is a busy-host standalone microbenchmark on an Apple M5 Max with macOS 27.2 and Xcode 27.2. It measures one synthetic expert, not end-to-end model throughput. The declared gate requires a qualifying split arm to beat float × bf16 by at least 1.5× for every K and weight format at both 32 and 64 rows. The 8- and 128-row cases are reported but do not gate. The qualifying arm is three-term, or two-term when every measured result fits `sumBound`. Both arms are reported. The preconverted two-term result fits the bound in every measured case, but neither split reaches 1.5× across the gate. The in-kernel packed variants also miss the gate. The largest observed gated preconverted two-term median ratio is 1.244×; this is the measured ceiling of this configuration, not an end-to-end projection. Some ranges are broad, and the kernels were not tuned or profiled to identify a limiter.

The owner ruled that the eight-row NVFP4 baseline must use the kernel the engine actually dispatches. Those cells use `i2p_down_grouped`; every other baseline cell uses `i2p_expert_matrix`. The latter is selected at eight rows for MXFP8 and at 32 or more rows for NVFP4. Both baseline kernels are copies of the engine kernels, with one expert and raw down-projection output.

## Method and limits

Each case has N=2304 and one expert, with the K and routed-row counts shown below. The packed codes and block scales come from the existing deterministic synthetic generator. NVFP4 uses global scale 1, so the exactly representable E2M1 × E4M3 product is converted to bf16; arbitrary checkpoint per-expert globals are not modeled. The host checks every converted weight against the fp64 decoded value. MXFP8 similarly checks E4M3 × E8M0 on the generated finite scales. Extreme scale codes, non-finite values, real checkpoint distributions, and f32 subnormal behavior are not qualified.

The float arm uses the engine's 16 × 32 float × bf16 form. The `pre` arms read preconverted bf16 weights; the `packed` arms decode the original bytes inside the matmul kernel into cooperative bf16 inputs. The latter uses one SIMD group because the SDK's `MPPTensorOpsMatMul2dImpl.h:5363` requires one for cooperative inputs. All bf16 arms include activation conversion in the timed command buffer. Two and three terms use bf16 residuals and accumulate into one f32 destination. The packed and preconverted variants use different tiling and SIMD-group counts, so their comparison includes those implementation choices.

Each arm has three warmups and 20 timed command buffers. Arm order rotates on every iteration. The table reports median and full observed range in GPU microseconds. `exact` activations are deterministic random f32 values rounded to bf16 before all arms. `random` activations use the same deterministic uniform f32 generator without that rounding. Errors are maxima over every raw output against `fp4_ref.zig`'s fp64 reference. Relative error is `abs(error) / max(abs(reference), 1e-30)` and can be large near zero. `fail` counts results outside the unchanged `sumBound`; one-term bf16 is intentionally approximate and may fail. Two-term and three-term had zero bound failures across both input sets, formats, K values, and row counts, for both weight paths. This does not establish the end-to-end per-logit contract.

The focused test checks every arm against `sumBound` on bf16-exact inputs at eight and 32 rows for each K and format. It checks the raw results on random f32 inputs as well, permitting the one-term approximation to exceed the bound. The negative control doubles the preconverted weights and requires over three quarters of one output row to exceed the unchanged bound.

## Measurements

| format | K | rows | arm | median µs (range) | exact max abs | exact max rel | random max abs | random max rel | random bound failures |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nvfp4 | 1152 | 8 | baseline | 38.292 (37.750–93.500) | 3.576279e-5 | 1.367989e-3 | 6.546825e-5 | 3.993300e-4 | 0 |
| nvfp4 | 1152 | 8 | float | 22.000 (21.750–49.708) | 1.015663e-4 | 6.839945e-4 | 5.376786e-4 | 4.745712e-3 | 0 |
| nvfp4 | 1152 | 8 | one_pre | 14.187 (14.042–31.750) | 1.015663e-4 | 6.839945e-4 | 7.031391e-1 | 4.239658e1 | 4 |
| nvfp4 | 1152 | 8 | two_pre | 21.375 (21.125–50.208) | 1.015663e-4 | 6.839945e-4 | 1.083571e-3 | 2.331745e-2 | 0 |
| nvfp4 | 1152 | 8 | three_pre | 28.667 (28.333–68.833) | 1.015663e-4 | 6.839945e-4 | 4.950203e-4 | 4.529084e-4 | 0 |
| nvfp4 | 1152 | 8 | one_packed | 165.917 (165.667–430.917) | 1.015663e-4 | 6.839945e-4 | 7.031391e-1 | 4.239658e1 | 4 |
| nvfp4 | 1152 | 8 | two_packed | 175.125 (174.708–489.458) | 1.015663e-4 | 6.839945e-4 | 1.007278e-3 | 2.251029e-2 | 0 |
| nvfp4 | 1152 | 8 | three_packed | 183.979 (183.625–463.375) | 1.015663e-4 | 6.839945e-4 | 3.809445e-4 | 1.316843e-3 | 0 |
| nvfp4 | 1152 | 32 | baseline | 107.521 (107.250–207.792) | 1.068115e-4 | 2.816324e-3 | 5.089268e-4 | 4.321524e-2 | 0 |
| nvfp4 | 1152 | 32 | float | 19.792 (19.125–39.375) | 1.068115e-4 | 2.816324e-3 | 5.089268e-4 | 4.321524e-2 | 0 |
| nvfp4 | 1152 | 32 | one_pre | 14.979 (14.833–29.750) | 1.068115e-4 | 4.608295e-3 | 7.055279e-1 | 2.096808e2 | 8 |
| nvfp4 | 1152 | 32 | two_pre | 23.313 (23.125–43.750) | 1.068115e-4 | 4.608295e-3 | 1.123523e-3 | 5.328442e-1 | 0 |
| nvfp4 | 1152 | 32 | three_pre | 31.708 (31.458–61.125) | 1.068115e-4 | 4.608295e-3 | 7.212330e-4 | 5.962129e-3 | 0 |
| nvfp4 | 1152 | 32 | one_packed | 166.604 (166.417–338.417) | 1.068115e-4 | 4.608295e-3 | 7.055279e-1 | 2.096808e2 | 8 |
| nvfp4 | 1152 | 32 | two_packed | 176.479 (176.250–355.750) | 1.068115e-4 | 4.608295e-3 | 1.154041e-3 | 5.346509e-1 | 0 |
| nvfp4 | 1152 | 32 | three_packed | 186.375 (186.125–375.792) | 1.068115e-4 | 4.608295e-3 | 3.969166e-4 | 2.182617e-2 | 0 |
| nvfp4 | 1152 | 64 | baseline | 206.000 (153.750–484.875) | 1.277924e-4 | 1.614907e-2 | 6.444007e-4 | 9.704625e-2 | 0 |
| nvfp4 | 1152 | 64 | float | 61.479 (45.333–144.208) | 1.277924e-4 | 1.614907e-2 | 6.444007e-4 | 9.704625e-2 | 0 |
| nvfp4 | 1152 | 64 | one_pre | 29.500 (22.375–67.250) | 1.029670e-4 | 1.614907e-2 | 7.749838e-1 | 1.054685e2 | 16 |
| nvfp4 | 1152 | 64 | two_pre | 49.417 (37.000–114.125) | 1.029670e-4 | 1.614907e-2 | 1.221327e-3 | 1.558132e-1 | 0 |
| nvfp4 | 1152 | 64 | three_pre | 68.875 (51.333–160.500) | 1.029670e-4 | 1.614907e-2 | 6.426573e-4 | 2.703160e-2 | 0 |
| nvfp4 | 1152 | 64 | one_packed | 343.583 (253.000–807.042) | 1.029670e-4 | 1.614907e-2 | 7.749838e-1 | 1.054685e2 | 16 |
| nvfp4 | 1152 | 64 | two_packed | 363.062 (362.500–852.250) | 1.029670e-4 | 1.614907e-2 | 1.209883e-3 | 1.953778e-1 | 0 |
| nvfp4 | 1152 | 64 | three_packed | 382.375 (368.792–898.125) | 1.029670e-4 | 1.614907e-2 | 4.360341e-4 | 1.548154e-1 | 0 |
| nvfp4 | 1152 | 128 | baseline | 163.938 (163.375–165.625) | 1.204014e-4 | 2.941176e-2 | 5.636811e-4 | 5.891977e-2 | 0 |
| nvfp4 | 1152 | 128 | float | 52.125 (51.833–52.542) | 1.204014e-4 | 2.941176e-2 | 5.636811e-4 | 5.891977e-2 | 0 |
| nvfp4 | 1152 | 128 | one_pre | 26.250 (25.500–27.833) | 1.103878e-4 | 2.941176e-2 | 8.533753e-1 | 8.490725e1 | 29 |
| nvfp4 | 1152 | 128 | two_pre | 46.104 (45.000–89.917) | 1.103878e-4 | 2.941176e-2 | 1.212690e-3 | 1.876421e-1 | 0 |
| nvfp4 | 1152 | 128 | three_pre | 66.271 (65.125–131.792) | 1.103878e-4 | 2.941176e-2 | 7.230751e-4 | 3.937219e-3 | 0 |
| nvfp4 | 1152 | 128 | one_packed | 202.771 (202.083–413.583) | 1.103878e-4 | 2.941176e-2 | 8.533753e-1 | 8.490725e1 | 29 |
| nvfp4 | 1152 | 128 | two_packed | 212.750 (211.125–217.667) | 1.103878e-4 | 2.941176e-2 | 1.212690e-3 | 1.872673e-1 | 0 |
| nvfp4 | 1152 | 128 | three_packed | 222.917 (220.875–229.000) | 1.103878e-4 | 2.941176e-2 | 4.770085e-4 | 2.782123e-2 | 0 |
| nvfp4 | 2304 | 8 | baseline | 64.854 (64.250–127.208) | 5.340576e-5 | 5.924951e-3 | 8.086674e-5 | 1.999850e-3 | 0 |
| nvfp4 | 2304 | 8 | float | 40.021 (39.500–78.750) | 1.697540e-4 | 3.291639e-3 | 1.341313e-3 | 1.333762e-2 | 0 |
| nvfp4 | 2304 | 8 | one_pre | 23.896 (23.583–43.708) | 1.611710e-4 | 3.291639e-3 | 9.280852e-1 | 5.481908e1 | 0 |
| nvfp4 | 2304 | 8 | two_pre | 42.292 (41.625–78.583) | 1.611710e-4 | 3.291639e-3 | 1.455095e-3 | 1.818073e-2 | 0 |
| nvfp4 | 2304 | 8 | three_pre | 60.521 (59.625–112.708) | 1.611710e-4 | 3.291639e-3 | 1.437292e-3 | 1.723473e-3 | 0 |
| nvfp4 | 2304 | 8 | one_packed | 326.229 (325.750–659.417) | 1.611710e-4 | 3.291639e-3 | 9.280852e-1 | 5.481908e1 | 0 |
| nvfp4 | 2304 | 8 | two_packed | 344.000 (343.625–695.125) | 1.611710e-4 | 3.291639e-3 | 1.485612e-3 | 2.140899e-2 | 0 |
| nvfp4 | 2304 | 8 | three_packed | 362.104 (361.667–731.750) | 1.611710e-4 | 3.291639e-3 | 7.420480e-4 | 4.623732e-3 | 0 |
| nvfp4 | 2304 | 32 | baseline | 210.312 (208.208–358.208) | 2.536774e-4 | 1.620300e-2 | 1.029946e-3 | 4.324477e-3 | 0 |
| nvfp4 | 2304 | 32 | float | 36.813 (35.375–64.375) | 2.536774e-4 | 1.620300e-2 | 1.029946e-3 | 4.324477e-3 | 0 |
| nvfp4 | 2304 | 32 | one_pre | 24.875 (24.250–50.000) | 4.196167e-4 | 1.620300e-2 | 1.124586e0 | 3.994360e1 | 0 |
| nvfp4 | 2304 | 32 | two_pre | 44.521 (43.792–75.250) | 4.196167e-4 | 1.620300e-2 | 1.580527e-3 | 2.209091e-2 | 0 |
| nvfp4 | 2304 | 32 | three_pre | 64.125 (63.375–108.875) | 4.196167e-4 | 1.620300e-2 | 1.145447e-3 | 2.140584e-3 | 0 |
| nvfp4 | 2304 | 32 | one_packed | 327.604 (327.000–604.625) | 4.196167e-4 | 1.620300e-2 | 1.124586e0 | 3.994360e1 | 0 |
| nvfp4 | 2304 | 32 | two_packed | 347.188 (346.542–675.792) | 4.196167e-4 | 1.620300e-2 | 1.540042e-3 | 1.843291e-2 | 0 |
| nvfp4 | 2304 | 32 | three_packed | 366.937 (366.292–665.458) | 4.196167e-4 | 1.620300e-2 | 1.034351e-3 | 3.449467e-3 | 0 |
| nvfp4 | 2304 | 64 | baseline | 203.333 (201.125–293.167) | 3.170967e-4 | 2.175665e-2 | 1.160624e-3 | 3.721653e-2 | 0 |
| nvfp4 | 2304 | 64 | float | 58.000 (57.542–86.000) | 3.170967e-4 | 2.175665e-2 | 1.160624e-3 | 3.721653e-2 | 0 |
| nvfp4 | 2304 | 64 | one_pre | 27.000 (26.833–38.542) | 2.597570e-4 | 3.464948e-2 | 1.165150e0 | 6.765216e1 | 0 |
| nvfp4 | 2304 | 64 | two_pre | 48.500 (48.000–69.875) | 2.597570e-4 | 3.464948e-2 | 1.813453e-3 | 7.875375e-2 | 0 |
| nvfp4 | 2304 | 64 | three_pre | 69.604 (69.000–100.250) | 2.597570e-4 | 3.464948e-2 | 1.390515e-3 | 1.270379e-2 | 0 |
| nvfp4 | 2304 | 64 | one_packed | 332.875 (332.000–496.375) | 2.597570e-4 | 3.464948e-2 | 1.165150e0 | 6.765216e1 | 0 |
| nvfp4 | 2304 | 64 | two_packed | 351.083 (350.292–564.167) | 2.597570e-4 | 3.464948e-2 | 1.602413e-3 | 1.131508e-1 | 0 |
| nvfp4 | 2304 | 64 | three_packed | 370.125 (369.542–554.333) | 2.597570e-4 | 3.464948e-2 | 8.044839e-4 | 1.933159e-1 | 0 |
| nvfp4 | 2304 | 128 | baseline | 321.062 (320.417–326.625) | 2.880096e-4 | 3.153962e-2 | 1.199793e-3 | 2.507350e-1 | 0 |
| nvfp4 | 2304 | 128 | float | 101.208 (100.333–103.583) | 2.880096e-4 | 3.153962e-2 | 1.199793e-3 | 2.507350e-1 | 0 |
| nvfp4 | 2304 | 128 | one_pre | 46.896 (45.667–49.500) | 3.490448e-4 | 3.153962e-2 | 1.139539e0 | 1.056695e2 | 0 |
| nvfp4 | 2304 | 128 | two_pre | 87.708 (86.292–89.875) | 3.490448e-4 | 3.153962e-2 | 1.707405e-3 | 8.490946e-2 | 0 |
| nvfp4 | 2304 | 128 | three_pre | 128.458 (126.333–133.000) | 3.490448e-4 | 3.153962e-2 | 1.524299e-3 | 1.155297e-2 | 0 |
| nvfp4 | 2304 | 128 | one_packed | 403.375 (402.083–405.250) | 3.490448e-4 | 3.153962e-2 | 1.139539e0 | 1.056695e2 | 0 |
| nvfp4 | 2304 | 128 | two_packed | 423.771 (420.375–448.500) | 3.490448e-4 | 3.153962e-2 | 1.689212e-3 | 1.311934e-1 | 0 |
| nvfp4 | 2304 | 128 | three_packed | 441.000 (438.417–482.000) | 3.490448e-4 | 3.153962e-2 | 9.415708e-4 | 7.094163e-2 | 0 |
| mxfp8 | 1152 | 8 | baseline | 157.854 (154.667–373.875) | 6.332994e-6 | 3.997294e-3 | 1.432691e-5 | 4.300075e-3 | 0 |
| mxfp8 | 1152 | 8 | float | 40.604 (39.875–91.750) | 6.332994e-6 | 3.997294e-3 | 1.432691e-5 | 4.300075e-3 | 0 |
| mxfp8 | 1152 | 8 | one_pre | 26.042 (25.750–57.875) | 5.736947e-6 | 4.349928e-3 | 2.145571e-2 | 8.386415e1 | 105 |
| mxfp8 | 1152 | 8 | two_pre | 40.750 (40.500–106.625) | 5.736947e-6 | 4.349928e-3 | 3.325157e-5 | 7.996583e-2 | 0 |
| mxfp8 | 1152 | 8 | three_pre | 55.562 (55.125–126.500) | 5.736947e-6 | 4.349928e-3 | 1.630888e-5 | 6.088576e-4 | 0 |
| mxfp8 | 1152 | 8 | one_packed | 339.229 (335.875–856.667) | 5.736947e-6 | 4.349928e-3 | 2.145571e-2 | 8.386415e1 | 105 |
| mxfp8 | 1152 | 8 | two_packed | 354.583 (354.125–884.750) | 5.736947e-6 | 4.349928e-3 | 3.441002e-5 | 8.649353e-2 | 0 |
| mxfp8 | 1152 | 8 | three_packed | 372.479 (372.125–875.500) | 5.736947e-6 | 4.349928e-3 | 1.157128e-5 | 5.941848e-3 | 0 |
| mxfp8 | 1152 | 32 | baseline | 200.312 (197.375–548.625) | 1.266971e-5 | 1.868161e-3 | 2.012759e-5 | 1.108361e0 | 0 |
| mxfp8 | 1152 | 32 | float | 38.979 (37.750–89.500) | 1.266971e-5 | 1.868161e-3 | 2.012759e-5 | 1.108361e0 | 0 |
| mxfp8 | 1152 | 32 | one_pre | 27.125 (26.458–125.500) | 7.565133e-6 | 1.734721e-3 | 2.442267e-2 | 6.125312e3 | 442 |
| mxfp8 | 1152 | 32 | two_pre | 44.938 (43.333–214.542) | 7.565133e-6 | 1.734721e-3 | 5.361692e-5 | 8.821453e0 | 0 |
| mxfp8 | 1152 | 32 | three_pre | 60.875 (60.375–277.792) | 7.565133e-6 | 1.734721e-3 | 3.263608e-5 | 2.064321e-1 | 0 |
| mxfp8 | 1152 | 32 | one_packed | 338.354 (337.667–800.875) | 7.565133e-6 | 1.734721e-3 | 2.442267e-2 | 6.125312e3 | 442 |
| mxfp8 | 1152 | 32 | two_packed | 361.813 (357.833–889.458) | 7.565133e-6 | 1.734721e-3 | 3.656810e-5 | 9.366836e0 | 0 |
| mxfp8 | 1152 | 32 | three_packed | 378.792 (377.125–1025.333) | 7.565133e-6 | 1.734721e-3 | 1.798094e-5 | 4.486529e-1 | 0 |
| mxfp8 | 1152 | 64 | baseline | 101.083 (100.333–107.375) | 9.547919e-6 | 4.318937e-2 | 2.109316e-5 | 3.204799e-2 | 0 |
| mxfp8 | 1152 | 64 | float | 31.083 (30.500–32.125) | 9.547919e-6 | 4.318937e-2 | 2.109316e-5 | 3.204799e-2 | 0 |
| mxfp8 | 1152 | 64 | one_pre | 16.250 (15.917–69.583) | 8.963048e-6 | 7.356431e-2 | 2.905446e-2 | 2.286216e2 | 796 |
| mxfp8 | 1152 | 64 | two_pre | 25.875 (25.667–56.792) | 8.963048e-6 | 7.356431e-2 | 4.103734e-5 | 4.981208e-1 | 0 |
| mxfp8 | 1152 | 64 | three_pre | 35.625 (35.250–65.333) | 8.963048e-6 | 7.356431e-2 | 2.330456e-5 | 8.321607e-3 | 0 |
| mxfp8 | 1152 | 64 | one_packed | 175.062 (170.417–240.125) | 8.963048e-6 | 7.356431e-2 | 2.905446e-2 | 2.286216e2 | 796 |
| mxfp8 | 1152 | 64 | two_packed | 180.562 (180.083–203.375) | 8.963048e-6 | 7.356431e-2 | 3.831700e-5 | 5.041151e-1 | 0 |
| mxfp8 | 1152 | 64 | three_packed | 190.104 (189.667–210.042) | 8.963048e-6 | 7.356431e-2 | 1.650784e-5 | 9.090153e-3 | 0 |
| mxfp8 | 1152 | 128 | baseline | 147.688 (146.917–149.542) | 1.315773e-5 | 1.652893e-2 | 2.674726e-5 | 1.047046e-1 | 0 |
| mxfp8 | 1152 | 128 | float | 52.125 (51.875–53.000) | 1.315773e-5 | 1.652893e-2 | 2.674726e-5 | 1.047046e-1 | 0 |
| mxfp8 | 1152 | 128 | one_pre | 26.562 (26.000–29.625) | 1.006527e-5 | 1.652893e-2 | 2.709401e-2 | 1.219461e3 | 1585 |
| mxfp8 | 1152 | 128 | two_pre | 45.917 (45.125–207.625) | 1.006527e-5 | 1.652893e-2 | 4.228809e-5 | 1.422650e0 | 0 |
| mxfp8 | 1152 | 128 | three_pre | 66.187 (64.875–159.750) | 1.006527e-5 | 1.652893e-2 | 2.200412e-5 | 7.864350e-2 | 0 |
| mxfp8 | 1152 | 128 | one_packed | 187.500 (186.625–196.083) | 1.006527e-5 | 1.652893e-2 | 2.709401e-2 | 1.219461e3 | 1585 |
| mxfp8 | 1152 | 128 | two_packed | 196.562 (195.500–206.125) | 1.006527e-5 | 1.652893e-2 | 4.276493e-5 | 1.274641e0 | 0 |
| mxfp8 | 1152 | 128 | three_packed | 207.083 (205.708–214.667) | 1.006527e-5 | 1.652893e-2 | 1.844771e-5 | 7.231321e-2 | 0 |
| mxfp8 | 2304 | 8 | baseline | 155.812 (153.958–314.167) | 1.816638e-5 | 2.130059e-3 | 3.162124e-5 | 5.746586e-2 | 0 |
| mxfp8 | 2304 | 8 | float | 40.125 (39.875–74.375) | 1.816638e-5 | 2.130059e-3 | 3.162124e-5 | 5.746586e-2 | 0 |
| mxfp8 | 2304 | 8 | one_pre | 23.937 (23.750–43.750) | 9.967946e-6 | 9.942377e-4 | 2.842505e-2 | 6.453750e1 | 0 |
| mxfp8 | 2304 | 8 | two_pre | 42.167 (41.875–78.125) | 9.967946e-6 | 9.942377e-4 | 5.942031e-5 | 5.024955e-2 | 0 |
| mxfp8 | 2304 | 8 | three_pre | 60.479 (60.250–113.250) | 9.967946e-6 | 9.942377e-4 | 5.937414e-5 | 3.088180e-3 | 0 |
| mxfp8 | 2304 | 8 | one_packed | 331.917 (331.625–665.250) | 9.967946e-6 | 9.942377e-4 | 2.842505e-2 | 6.453750e1 | 0 |
| mxfp8 | 2304 | 8 | two_packed | 349.375 (349.125–701.792) | 9.967946e-6 | 9.942377e-4 | 4.508395e-5 | 5.086230e-2 | 0 |
| mxfp8 | 2304 | 8 | three_packed | 366.687 (366.458–737.333) | 9.967946e-6 | 9.942377e-4 | 2.051889e-5 | 1.305369e-2 | 0 |
| mxfp8 | 2304 | 32 | baseline | 201.562 (198.625–399.000) | 1.970679e-5 | 2.144235e-2 | 3.657467e-5 | 2.928965e-2 | 0 |
| mxfp8 | 2304 | 32 | float | 35.604 (35.250–45.500) | 1.970679e-5 | 2.144235e-2 | 3.657467e-5 | 2.928965e-2 | 0 |
| mxfp8 | 2304 | 32 | one_pre | 24.833 (24.625–86.125) | 1.387578e-5 | 2.574236e-2 | 3.622150e-2 | 5.029003e1 | 0 |
| mxfp8 | 2304 | 32 | two_pre | 44.625 (44.375–82.958) | 1.387578e-5 | 2.574236e-2 | 5.355748e-5 | 6.509191e-1 | 0 |
| mxfp8 | 2304 | 32 | three_pre | 64.500 (64.125–120.375) | 1.387578e-5 | 2.574236e-2 | 5.620044e-5 | 3.362860e-2 | 0 |
| mxfp8 | 2304 | 32 | one_packed | 333.792 (333.625–668.792) | 1.387578e-5 | 2.574236e-2 | 3.622150e-2 | 5.029003e1 | 0 |
| mxfp8 | 2304 | 32 | two_packed | 352.292 (351.958–708.750) | 1.387578e-5 | 2.574236e-2 | 5.169832e-5 | 7.255677e-1 | 0 |
| mxfp8 | 2304 | 32 | three_packed | 371.750 (371.625–749.125) | 1.387578e-5 | 2.574236e-2 | 2.690017e-5 | 4.651838e-2 | 0 |
| mxfp8 | 2304 | 64 | baseline | 198.125 (197.250–296.750) | 2.416037e-5 | 9.829272e-2 | 3.903507e-5 | 5.851253e0 | 0 |
| mxfp8 | 2304 | 64 | float | 58.229 (57.500–86.458) | 2.416037e-5 | 9.829272e-2 | 3.903507e-5 | 5.851253e0 | 0 |
| mxfp8 | 2304 | 64 | one_pre | 27.146 (26.750–38.500) | 1.518056e-5 | 1.342616e-2 | 4.186553e-2 | 5.000880e3 | 0 |
| mxfp8 | 2304 | 64 | two_pre | 48.417 (48.208–69.708) | 1.518056e-5 | 1.342616e-2 | 5.462680e-5 | 2.322265e1 | 0 |
| mxfp8 | 2304 | 64 | three_pre | 69.854 (69.333–243.250) | 1.518056e-5 | 1.342616e-2 | 4.286827e-5 | 1.088244e0 | 0 |
| mxfp8 | 2304 | 64 | one_packed | 344.479 (341.125–512.000) | 1.518056e-5 | 1.342616e-2 | 4.186553e-2 | 5.000880e3 | 0 |
| mxfp8 | 2304 | 64 | two_packed | 355.479 (354.792–533.333) | 1.518056e-5 | 1.342616e-2 | 5.558047e-5 | 2.441459e1 | 0 |
| mxfp8 | 2304 | 64 | three_packed | 374.625 (374.208–555.500) | 1.518056e-5 | 1.342616e-2 | 3.678389e-5 | 2.742078e0 | 0 |
| mxfp8 | 2304 | 128 | baseline | 291.354 (290.750–292.125) | 2.000481e-5 | 1.221282e-1 | 4.341230e-5 | 1.196656e-1 | 0 |
| mxfp8 | 2304 | 128 | float | 101.167 (100.375–103.833) | 2.000481e-5 | 1.221282e-1 | 4.341230e-5 | 1.196656e-1 | 0 |
| mxfp8 | 2304 | 128 | one_pre | 46.938 (46.708–50.458) | 2.235547e-5 | 7.053607e-2 | 3.635468e-2 | 3.631725e2 | 0 |
| mxfp8 | 2304 | 128 | two_pre | 87.771 (86.542–130.375) | 2.235547e-5 | 7.053607e-2 | 6.524136e-5 | 6.603757e-1 | 0 |
| mxfp8 | 2304 | 128 | three_pre | 130.104 (128.417–169.625) | 2.235547e-5 | 7.053607e-2 | 4.205498e-5 | 5.130046e-2 | 0 |
| mxfp8 | 2304 | 128 | one_packed | 375.104 (374.000–379.750) | 2.235547e-5 | 7.053607e-2 | 3.635468e-2 | 3.631725e2 | 0 |
| mxfp8 | 2304 | 128 | two_packed | 394.458 (392.375–398.250) | 2.235547e-5 | 7.053607e-2 | 6.142666e-5 | 6.183585e-1 | 0 |
| mxfp8 | 2304 | 128 | three_packed | 413.563 (411.875–419.250) | 2.235547e-5 | 7.053607e-2 | 3.292101e-5 | 2.198188e-2 | 0 |

## Commands and admission

Zig 0.17.1 was selected with `PATH=/opt/homebrew/bin:$PATH`. `TMPDIR` was the worktree's `.work/tmp`; `ZIG_GLOBAL_CACHE_DIR` was the campaign's `.work/zig-global-cache`. The test command exited 0:

```sh
python3 "$CAMPAIGN/heavy.py" -- zig build test-fp4-kernels -Dcpu=apple_m1
```

The test runner labels stderr as `failed command`, but the build and wrapper exited 0. Its existing MXFP8 negative controls reported 2297/2304 and 2280/2304 outputs outside the bound. The new split test passed its own corrupted-weight control.

The shaders and executable were compiled before admission:

```sh
xcrun -sdk macosx metal -std=metal4.1 -c zig/src/core/metal_fp4/split_tensorops.metal -o .work/tmp/split_tensorops.air
xcrun -sdk macosx metal -std=metal4.1 -c zig/src/core/metal_fp4/split_baseline.metal -o .work/tmp/split_nv.air
xcrun -sdk macosx metal -std=metal4.1 -DI2P_MXFP8=1 -c zig/src/core/metal_fp4/split_baseline.metal -o .work/tmp/split_mx.air
xcrun -sdk macosx metallib .work/tmp/split_tensorops.air -o .work/tmp/split_tensorops.metallib
xcrun -sdk macosx metallib .work/tmp/split_nv.air -o .work/tmp/split_nv.metallib
xcrun -sdk macosx metallib .work/tmp/split_mx.air -o .work/tmp/split_mx.metallib
zig build tf-split-bf16-bench -Dcpu=apple_m1
```

All build commands exited 0. The admitted run used the stricter 35/35 thresholds that applied before the later campaign-wide 60/60 update:

```sh
GPU_GATE_MAX_GPU_PCT=35 GPU_GATE_MAX_UTIL_PCT=35 python3 "$CAMPAIGN/heavy.py" -- sh -c '
  gpu-gate-ready -1 || exit 75
  perf-gate-ready || exit 75
  ./zig-out/bin/tf-split-bf16-bench
  run=$?
  perf-gate-ready -1
  post=$?
  printf "bench_exit=%s post_cpu_exit=%s\n" "$run" "$post"
  if [ "$run" -ne 0 ]; then exit "$run"; fi
  if [ "$post" -ne 0 ]; then exit 76; fi
'
```

The wrapper exited 0 and printed `bench_exit=0 post_cpu_exit=0`. The prechecks therefore both exited 0. Peak charged memory was 1.1 GB; worst host pressure was 1. Shader libraries were precompiled, and no full model was loaded.
