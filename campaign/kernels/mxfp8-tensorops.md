# Inkling 2 Mini MXFP8 tensor operations

This is a standalone experiment, not an engine path. Host: Apple M5 Max, macOS 27.2 (26B123), Xcode 27.2 (27B52), Metal toolchain 27.2.47.0. The source is `zig/src/core/metal_fp4/mxfp8_tensorops.metal`.

## Representation and arithmetic

The inline API can describe one checkpoint expert without relayout: a rank-2 E4M3 tensor with extents `(K, 2304)`, K contiguous and row stride K bytes, and a `tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>` pointing to the unmodified scale bytes. The test uses 1152 and 2304 for K and random, distinct scale bytes in each row, so its comparison exercises packed scale row strides 36 and 72. These are the expert slices of `[128,2304,1152]` / `[128,2304,2304]`, and their corresponding `[128,2304,36]` / `[128,2304,72]` scale arrays. The shader slices an expert before dispatch; it does not create a rank-3 tensor. The E4M3 data row strides and both expert-plane offsets are multiples of 128 bytes. The scale row is not independently padded for this inline form.

The SDK's `metal_tensor` lines 333–355 restrict the scale element to UE8M0 and block factors to `(32,1,...)`; lines 5507–5559 construct an inline tensor from the two pointers, with explicit or implied data strides. Format tensor static extent 0 must be a multiple of 32, and the packed implied stride must be 128-byte aligned (`metal_tensor` lines 305–308, 505–531, 5402–5403 and 5532–5539). The host `MTLTensor` descriptor has its own rules: dimension 0 multiple of 32, non-first format strides multiples of 128 bytes, and plane attachment offsets aligned to 128 bytes (`MTLTensor.h` lines 224–255 and 316–324). These host descriptor rules are not proof of independent inline scale strides; the correctness experiment establishes the packed inline case.

`MPPTensorOpsMatMul2d.h` lines 72–83 list half × E4M3 with half or float destination and E4M3 × E4M3 with half or float destination; bfloat or float × E4M3 is not listed. This probe converts activation values to f16 before the operation and uses f32 destination/accumulation, with `relaxed_precision=false`; the table and thread paths consume f32 activations and accumulate in f32. The benchmark gives both paths the same exactly f16-representable activation values. It excludes the activation conversion from the timed path, so a measured tensor advantage is an optimistic bound for f32-input use.

The existing fp64 oracle and table shader masked bit 7 of the scale, incorrectly interpreting valid E8M0 codes 128–254. This experiment corrects that mask, and the reference explicitly maps 255 to NaN. E4M3 codes 0x7f and 0xff are NaN in the reference. The random tests cover finite E4M3 and E8M0 codes 118–128, including 128, but do not establish TensorOps behavior for NaNs, infinity, extreme scales, subnormal values, or complete real checkpoint contents. TensorOps handling of those cases remains unverified.

NVFP4's E4M3 block scales every 16 values cannot be attached through this scale-plane API, which accepts only UE8M0 and block 32. Its extra per-expert global would also require separate multiplication. Cooperative tensor inputs could instead implement custom dequantization, but that is different arithmetic and was not tested here.

## Correctness

`zig build test-fp4-kernels -Dcpu=apple_m1` includes the existing variants and the TensorOps test. For each K, the test generates 2304 weight rows and eight activation rows, rounds the latter to f16 and feeds the exact resulting values to the fp64 reference. It checks all 18,432 raw matrix results against `fp4_ref.zig`'s `sumBound`; those values include every 17th scale set to 128. The existing bound is `4 * (K+64) * 2^-24 * sum(abs(w*x)) + 2 * 2^-24 * abs(y)`. A second run decrements every scale by one and requires more than 75% of the first row's results to exceed that unchanged bound. On this host:

| K | worst error / bound | corrupted-scale results outside bound |
| --- | ---: | ---: |
| 1152 | 0.00059 | 2297 / 2304 |
| 2304 | 0.00024 | 2280 / 2304 |

For K=2304 the test also runs the benchmark's separate pointwise situ operation and checks all 9,216 activations against `ref.gu` and its derived bound. A sample of the first and last raw output rows is also checked against the fp64 reference for each timed size. Existing test output calls its stderr-producing test a `failed command`, but the test step and the heavy wrapper both returned exit 0.

## Timing and decision

**Keep as a prefill/verify investigation; stop for single-row decode.** This is an optimistic microbenchmark, not a production speed claim. It assumes one active expert receives exactly 1, 8 or 64 routed rows; it does not model the eight simultaneously selected experts or routing distribution. Each arm reads the same synthetic expert bytes and activation values. The table and thread arms are the current kernels with only the expert weight index fixed to zero so rows reuse the same expert. Gate/up includes its situ calculation on both paths; TensorOps uses a second dispatch for it. The TensorOps activation conversion is excluded. The 3 arms are rotated each iteration, with 3 warmups and 20 timed command buffers per arm; the table shows median GPU microseconds and the full observed range. The check covers finite values only and uses warm buffers. No relayout, conversion, routing, or whole-model cost is included.

| routed rows per expert | projection | TensorOps f16 × E4M3 → f32 | v3 table f32 | v4 thread f32 |
| ---: | --- | ---: | ---: | ---: |
| 1 | gate/up | 101.042 (100.208–102.125) | 56.583 (56.250–57.292) | 424.396 (420.875–428.625) |
| 1 | down | 52.708 (52.167–53.625) | 39.437 (39.083–40.000) | 120.479 (120.125–120.958) |
| 8 | gate/up | 106.396 (106.000–107.375) | 314.000 (313.000–314.708) | 428.875 (427.042–431.000) |
| 8 | down | 55.875 (55.583–56.500) | 206.313 (206.000–206.750) | 190.125 (187.250–191.500) |
| 64 | gate/up | 28.771 (28.375–54.875) | 487.500 (486.917–858.375) | 457.917 (456.417–684.417) |
| 64 | down | 15.125 (14.875–15.458) | 316.500 (316.250–317.625) | 224.354 (221.250–225.042) |

The large shift in absolute latency between the 8- and 64-row measurements, including the broad 64-row gate/up ranges, indicates changing device conditions; the interleaved within-shape comparisons support only the bounded verdict above. This single measurement is not tuned, and does not establish a limiter or an end-to-end speedup. Compare against the eight-expert decode in `fp4.md` only with those occupancy differences in mind.

## Commands and guard results

The `PATH` selects Homebrew Python because the system Python 3.9 cannot run `heavy.py`'s type annotations. Scratch and Zig global cache were set to `/Users/josh/repos/tensorfold-m5/.work/tmp` and `/Users/josh/repos/tensorfold-m5/.work/zig-global-cache` respectively.

```sh
TMPDIR=/Users/josh/repos/tensorfold-m5/.work/tmp xcrun -sdk macosx metal -std=metal4.1 -c zig/src/core/metal_fp4/mxfp8_tensorops.metal -o /Users/josh/repos/tensorfold-m5/.work/tmp/mxfp8_tensorops.air
PATH=/opt/homebrew/bin:$PATH TMPDIR=/Users/josh/repos/tensorfold-m5/.work/tmp ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tensorfold-m5/.work/zig-global-cache python3 /Users/josh/repos/tensorfold-m5/campaign/heavy.py --cap-gb 4 -- zig build test-fp4-kernels -Dcpu=apple_m1
PATH=/opt/homebrew/bin:$PATH TMPDIR=/Users/josh/repos/tensorfold-m5/.work/tmp ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tensorfold-m5/.work/zig-global-cache zig build tf-fp4-bench -Dcpu=apple_m1
PATH=/opt/homebrew/bin:$PATH TMPDIR=/Users/josh/repos/tensorfold-m5/.work/tmp ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tensorfold-m5/.work/zig-global-cache TF_MXFP8_TENSOROPS=1 python3 /Users/josh/repos/tensorfold-m5/campaign/heavy.py --cap-gb 4 -- sh -c 'perf-gate-ready -1 && gpu-gate-ready -1 || exit $?; ./zig-out/bin/tf-fp4-bench; run=$?; gpu-gate-ready -1; gpu=$?; perf-gate-ready -1; cpu=$?; printf "run_exit=%s post_gpu_exit=%s post_cpu_exit=%s\n" "$run" "$gpu" "$cpu"; test "$run" -eq 0 && test "$gpu" -eq 0 && test "$cpu" -eq 0'
```

All four commands exited 0. The timing wrapper printed `run_exit=0 post_gpu_exit=0 post_cpu_exit=0`. An earlier timing was discarded because its post-run CPU gate exited 75. While queued behind other work, three test attempts were stopped by the 4 GB wrapper on host wired-memory growth, and another timing attempt was stopped by the same guard; none is counted as a pass. The final test's peak charged memory was 1.6 GB.
