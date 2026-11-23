# Inkling 2 Mini fast prefill

Status: in progress. The initial expert-only prototype did not show a clear
win. A replacement fragment layout lost its first 1K screen. The packed-load
revision passes the family tests and its five 1K brackets favor fast. The 4K,
dense and attention comparisons remain in progress. The added half expert variants are not registered with the
kernel contract. Registration remains a family-wide promotion requirement.

The fast profile selects half expert and attention fragments, bf16 dense
TensorOps, and float accumulation with `INKLING_PROFILE=fast`. Unset and `f32` select the existing f32 arithmetic.
Other values fail. It saturates activations outside the half range and prints
an overflow count per prefill chunk. The f32 residual stream, router, norms and
KV storage remain in place.

## Correctness

With the Mini checkpoint in `TF_INKLING2_LOOP_DIR`, the following serialized
family command exited 0 before the later profile parser test was added:

```sh
INKLING_LOAD_LIMIT_GB=26 python3 /Users/josh/repos/tensorfold-m5/campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1
```

The half expert test for the initial prototype covers both NVFP4 and MXFP8. It compares the output to
an FP64 reference of half-rounded operands and distinguishes the unrounded
reference. It also checks activation saturation and the overflow count. The
original f32 expert kernel failed the half-rounded reference negative control:
32.01921081542969 versus 32.03125. The host command
`zig build test-inkling2-loop-host -j1` exited 0. Both commands need repeating
after the final changes.

Default prefill parity at chunk 512 exited 2 as expected and reported 14 within
contract, two explained divergences and zero unexplained. The packed runner
repeated the same 14/2/0 result after a memory-pressure retry. The saved oracle
verified 16 histories. The packed fast capture exited 2 with 14 unexplained
under the f32 contract, as expected for its lower precision. The initial fast prototype produced zero non-finite logits across all 16 histories.
A streamed comparison found 2,807 agreements among 2,862 positions with a
reference top-two margin of at least 0.1. Mean KL averaged across histories was
0.0080107. These numbers do not apply to the replacement fragment layout.

```sh
make -C campaign -f parity/inkling2-mini/parity.mk parity-inkling2-mini RUNNER_MODE=prefill PREFILL_CHUNK=512 LANE=h1-default REPORT="$PWD/.work/tmp/h1-default-512.md"
```

## Expert-only timing screen

The following five 1,139-token brackets passed 60/60 GPU admission and the CPU
precheck and postcheck. Each bracket ran the same runner sequentially with
`INKLING_PROFILE=f32`, `fast`, then `f32`, under `heavy.py`. Prefill used chunk
512 and `bench 2 1 512`. The host was busy. Times are milliseconds. The first
control in three brackets suffered large page-in costs, so these observations
do not establish a speedup. Failed admission or postcheck attempts were
excluded.

| Bracket | Control 1 | Fast | Control 2 |
| --- | ---: | ---: | ---: |
| 1 | 4852.6 | 2852.0 | 2972.3 |
| 2 | 5159.9 | 2842.7 | 3071.8 |
| 3 | 2928.6 | 3111.8 | 3332.4 |
| 4 | 2915.1 | 2950.0 | 3101.2 |
| 5 | 5004.0 | 2921.6 | 3068.7 |

The five 5,265-token brackets measured control 1, fast and control 2 in
milliseconds as follows: (14,490.0, 15,166.6, 14,871.4), (15,040.0, 16,426.8,
16,272.7), (13,994.4, 14,968.8, 15,153.4), (15,973.3, 15,120.3, 16,127.6),
and (19,996.6, 17,800.5, 18,544.3). The median ratio of mean control time to
fast time was 0.974, with a range of 0.953 to 1.083. The initial prototype did
not demonstrate a win at this size and has been replaced. The replacement's
measurements and split Metal timeline are pending. One earlier diagnostic run
omitted admission checks and its timing is discarded.

## Combined prototype attribution

A separate instrumented run of the first combined fragment, dense TensorOps and
half attention implementation used 5,265 prompt tokens and chunk 512. The GPU
and CPU admission checks and the CPU postcheck passed. The split timeline had
115 segments, 9,366 dispatches and 1,296.362 ms of summed GPU span. Expert
matrix kernels accounted for 789.927 ms, half attention for 137.950 ms, bf16
TensorOps for 47.116 ms and activation conversion for 3.901 ms. The summed
instrumented span is not an end-to-end timing. The expert decode and activation
loads have since changed, so this attribution does not measure the current
source.

```sh
TF_METAL4_TIMELINE=1 TF_INKLING_PREFILL_PROFILE_BASE=0 INKLING_PROFILE=fast python3 /Users/josh/repos/tensorfold-m5/campaign/heavy.py -- RUNNER MODEL IDS bench 2 1 512
python3 campaign/kernels/inkling_profile.py timeline --split --log LOG
```

The first combined runner produced zero non-finite logits over the 16 parity
histories. It agreed on 2,765 of 2,862 confident positions. Its mean KL,
averaged across histories, was 0.0167438. The packed expert decode revision
was added afterward. Its 16 captured histories also contain zero non-finite
logits and have the same confident agreement and mean KL.

The packed-decode runner's split timeline used the same 5,265-token workload
and admission checks. It exited 0 with 115 segments, 9,366 dispatches and
999.397 ms of summed GPU span. Expert matrix kernels accounted for 457.567 ms,
half attention for 147.434 ms, bf16 TensorOps for 49.140 ms, and grouped
expert gate/up and down for 80.608 and 59.975 ms. These instrumented spans
are attribution, not paired end-to-end timings.

Five admitted packed-profile brackets on the 1,139-token prompt used the
control, fast, control sequence. Times in milliseconds were (2938.3, 2240.5,
3574.5), (2879.1, 2180.2, 2931.6), (2887.5, 2226.2, 2918.9), (2891.1,
2187.9, 3052.6), and (2870.8, 2164.9, 2874.2). The median ratio of mean
control time to fast time was 1.333, with a range of 1.304 to 1.453. This is
busy-host evidence. The 5,265-token packed brackets remain in progress.
