# Inkling Chunk fusion

Baseline revision `d898777a893e83b614831e4acbb8c831b0a8e5e4`; oQ4e checkpoint `3eea042751cd`. The input is the first 2,500 IDs from tokenizing `campaign/prompts/dev/code-4096.txt`. Each profile is one instrumented batch step after eight warmups. GPU stage spans are attribution, not timing results.

The baseline runner was built with `/opt/homebrew/bin/zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/b1h-base-build"` (exit 0). Tokenization was performed under `campaign/heavy.py` and exited 0. GPU and CPU admission and the immediate CPU post-check passed for each profile.

```sh
PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26 GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 TF_METAL4_TIMELINE=1 python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh "$PWD/.work/tmp/b1h-base-build/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" .work/tmp/b1h-code-2500.json batchbench K 9
```

## Baseline K=1

Command exit 0; profile reducer exit 0.

```text
1 timed rounds after 8 warm: 1 encoders, 2514 stages, 2514 dispatches a round
35.352 ms a round (35.352 in stages, 0.000 between encoders)
kernel                     calls/step    us/step   us/call   share
i2a_dense                       353.0     7441.0      21.1   21.0%
i2_batch_split                   48.0     4683.0      97.6   13.2%
i2p_rms                         345.0     3711.0      10.8   10.5%
i2p_gu                           44.0     2333.0      53.0    6.6%
i2_rms                          168.0     2226.0      13.2    6.3%
i2a_float                       336.0     2221.0       6.6    6.3%
i2p_down                         44.0     1888.0      42.9    5.3%
i2p_mix                          96.0     1781.0      18.6    5.0%
i2p_gather                      120.0     1685.0      14.0    4.8%
i2_conv                         192.0     1488.0       7.8    4.2%
i2_copy                         119.0      964.0       8.1    2.7%
i2_batch_combine                 48.0      734.0      15.3    2.1%
i2p_ctl                          96.0      635.0       6.6    1.8%
i2p_mix4                         24.0      627.0      26.1    1.8%
i2p_clip_scaled                  96.0      506.0       5.3    1.4%
i2_add                           88.0      491.0       5.6    1.4%
i2_kv_store                      48.0      395.0       8.2    1.1%
i2p_routepick                    48.0      378.0       7.9    1.1%
i2_relbias                       48.0      368.0       7.7    1.0%
i2p_combine                      48.0      217.0       4.5    0.6%
i2a_expert_up                     4.0      197.0      49.3    0.6%
i2p_clip                         48.0      146.0       3.0    0.4%
i2_sig_mul                       48.0      124.0       2.6    0.4%
i2a_expert_down                   4.0      103.0      25.8    0.3%
i2a_embed                         1.0       10.0       9.8    0.0%
sum of kernels                           35352.0            100.0%
```

## Baseline K=4

Command exit 0; profile reducer exit 0.

```text
1 timed rounds after 8 warm: 1 encoders, 3414 stages, 3414 dispatches a round
61.840 ms a round (61.840 in stages, 0.000 between encoders)
kernel                     calls/step    us/step   us/call   share
i2a_dense                       353.0    12570.0      35.6   20.3%
i2p_gu                           44.0     8172.0     185.7   13.2%
i2_batch_split                   48.0     7862.0     163.8   12.7%
i2p_down                         44.0     5818.0     132.2    9.4%
i2_conv                         768.0     5695.0       7.4    9.2%
i2a_float                       336.0     3647.0      10.9    5.9%
i2p_rms                         345.0     3320.0       9.6    5.4%
i2p_gather                      120.0     1669.0      13.9    2.7%
i2_rms                          168.0     1604.0       9.5    2.6%
i2p_mix                          96.0     1583.0      16.5    2.6%
i2_batch_combine                 48.0     1186.0      24.7    1.9%
i2_relbias                      192.0     1077.0       5.6    1.7%
i2_kv_store                     192.0     1061.0       5.5    1.7%
i2p_ctl                          96.0     1020.0      10.6    1.6%
i2_copy                         155.0      982.0       6.3    1.6%
i2p_mix4                         24.0      931.0      38.8    1.5%
i2_add                           88.0      889.0      10.1    1.4%
i2p_routepick                    48.0      679.0      14.1    1.1%
i2a_expert_up                     4.0      580.0     145.0    0.9%
i2p_combine                      48.0      495.0      10.3    0.8%
i2a_expert_down                   4.0      326.0      81.5    0.5%
i2p_clip_scaled                  96.0      324.0       3.4    0.5%
i2_sig_mul                       48.0      182.0       3.8    0.3%
i2p_clip                         48.0      154.0       3.2    0.2%
i2a_embed                         1.0       13.0      12.6    0.0%
sum of kernels                           61839.0            100.0%
```

## Intermediate correctness at 1cb799836

The native build `/opt/homebrew/bin/zig build native -Dcpu=apple_m1 -j1` exited 0. The timing runner was built separately with `/opt/homebrew/bin/zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/b1h-candidate-build"` and exited 0. Both used `ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache"` and `TMPDIR="$PWD/.work/tmp"`.

`INKLING_LOAD_LIMIT_GB=26 python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0. The GPU summary reported 73 of 73 tests passed, with no skips. The unchanged batch test checked logits and cache differences against `1.5e-5` and checked argmax. The suite also includes rollback and fork tests.

The oQ4e checkpoint ran the same family tests with `TF_INKLING2_LOOP_DIR` and `TF_INKLING2_LOOP_OQ4E_DIR` both set to `$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd`:

```sh
INKLING_LOAD_LIMIT_GB=26 TF_INKLING2_LOOP_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" TF_INKLING2_LOOP_OQ4E_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" python3 campaign/heavy.py -- /opt/homebrew/bin/zig build test-inkling2-loop -Dcpu=apple_m1 -j1 --summary all
```

The command exited 0 with 73 of 73 tests passed. The batch comparison reported zero logit and cache difference for all tested rows at widths 1, 2 and 4, including the cache-boundary case. Peak charged memory was 23.9 GB.

## Intermediate streamed fidelity at 1cb799836

The saved oracle verifier exited 0 and verified 16 histories, including two beyond the sliding window. The candidate capture command exited 0. The streamed comparison exited 0 with 2,775 top-1 matches in 3,015 rows (92.0398%), mean KL 0.039272796 and zero non-finite values. Mean KL is the mean of the 16 per-history means. The f32 oracle was not rerun.

```sh
export PATH="/Users/josh/repos/tf/.work/venv-ref/bin:/opt/homebrew/bin:$PATH"
export TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26
make -C campaign -f parity/inkling2-mini/parity.mk verify-oracle REF_UV=env REF_LOGITS="/Users/josh/repos/tf/.work/parity/inkling2-mini/ref"
make -C campaign -f parity/inkling2-mini/parity.mk metal-logits RUNNER_MODE=prefill PREFILL_CHUNK=512 LANE=b1h-chunk-fuse METAL_LOGITS="$PWD/.work/tmp/b1h-parity" WORK="$PWD/.work" REF_LOGITS="/Users/josh/repos/tf/.work/parity/inkling2-mini/ref" RUNNER="$PWD/.work/tmp/b1h-candidate-build/bin/tf-inkling2-loop-run" INKLING_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
/Users/josh/repos/tf/.work/venv-ref/bin/python .work/tmp/b1h-compare.py
```

The streamed comparator reads eight rows at a time:

```python
import json
from pathlib import Path
import numpy as np
histories = json.loads(Path('campaign/parity/inkling2-mini/histories.json').read_text())
refdir = Path('/Users/josh/repos/tf/.work/parity/inkling2-mini/ref')
outdir = Path('.work/tmp/b1h-parity')
total = matches = nonfinite = 0
means = []
for history in histories:
    name = history['name']
    ref = np.load(refdir / f'{name}.npy', mmap_mode='r')
    out = np.memmap(outdir / f'{name}.bin', dtype=np.float32, mode='r', shape=ref.shape)
    count = hit = bad = 0
    kl_sum = 0.0
    for i in range(0, ref.shape[0], 8):
        a = np.asarray(ref[i:i+8], dtype=np.float64)
        b = np.asarray(out[i:i+8], dtype=np.float64)
        bad += np.count_nonzero(~np.isfinite(b))
        hit += np.count_nonzero(np.argmax(a, axis=1) == np.argmax(b, axis=1))
        count += len(a)
        a -= np.max(a, axis=1, keepdims=True)
        b -= np.max(b, axis=1, keepdims=True)
        a -= np.log(np.exp(a).sum(axis=1, keepdims=True))
        b -= np.log(np.exp(b).sum(axis=1, keepdims=True))
        kl_sum += np.sum(np.exp(a) * (a - b))
    total += count
    matches += hit
    nonfinite += bad
    means.append(kl_sum / count)
    print(name, 'rows', count, 'top1', hit, 'mean_kl', kl_sum / count, 'nonfinite', bad, flush=True)
print('TOTAL', total, 'TOP1', matches, 'AGREEMENT', matches / total, 'MEAN_KL', np.mean(means), 'NONFINITE', nonfinite)
if matches / total < 0.915 or np.mean(means) > 0.045 or nonfinite:
    raise SystemExit(1)
```

## Intermediate K=1 profile at 1cb799836

Attempt 4 exited 0 after three CPU-gate deferrals. One instrumented batch step at 2,500 cached tokens had 1,494 dispatches and a 27.778 ms GPU span. The instrumented GPU span is attribution and does not establish throughput.

| Kernel | Calls per step | GPU ms per step |
| --- | ---: | ---: |
| `i2_batch_split` | 48.0 | 5.101 |
| `i2p_gu` | 44.0 | 2.345 |
| `i2a_batch_projections` | 48.0 | 2.194 |
| `i2a_float` | 144.0 | 2.192 |
| `i2a_dense` | 49.0 | 1.954 |
| `i2p_mix` | 96.0 | 1.792 |
| `i2p_down` | 44.0 | 1.664 |
| `i2p_batch_conv` | 192.0 | 1.280 |
| `i2a_batch_float_control` | 96.0 | 1.179 |
| `i2p_gather_rms` | 120.0 | 1.178 |
| `i2_batch_combine` | 48.0 | 1.056 |
| `i2p_combine` | 48.0 | 0.922 |
| `i2p_rms_pair` | 96.0 | 0.728 |
| `i2p_batch_store_rel` | 48.0 | 0.630 |
| `i2p_mix4` | 24.0 | 0.623 |
| `i2p_rms3` | 28.0 | 0.568 |
| `i2p_rms` | 69.0 | 0.483 |
| `i2p_routepick` | 48.0 | 0.405 |
| `i2_add` | 40.0 | 0.395 |
| `i2p_scatter` | 8.0 | 0.309 |
| `i2a_expert_up` | 4.0 | 0.197 |
| `i2p_clip3` | 48.0 | 0.188 |
| `i2_rms` | 48.0 | 0.142 |
| `i2_sig_mul` | 48.0 | 0.131 |
| `i2a_expert_down` | 4.0 | 0.104 |

## Baseline K=1 GPU times

| Kernel | Calls per step | GPU ms per step |
| --- | ---: | ---: |
| `i2a_dense` | 353.0 | 7.441 |
| `i2_batch_split` | 48.0 | 4.683 |
| `i2p_rms` | 345.0 | 3.711 |
| `i2p_gu` | 44.0 | 2.333 |
| `i2_rms` | 168.0 | 2.226 |
| `i2a_float` | 336.0 | 2.221 |
| `i2p_down` | 44.0 | 1.888 |
| `i2p_mix` | 96.0 | 1.781 |
| `i2p_gather` | 120.0 | 1.685 |
| `i2_conv` | 192.0 | 1.488 |
| `i2_copy` | 119.0 | 0.964 |
| `i2_batch_combine` | 48.0 | 0.734 |
| `i2p_ctl` | 96.0 | 0.635 |
| `i2p_mix4` | 24.0 | 0.627 |
| `i2p_clip_scaled` | 96.0 | 0.506 |
| `i2_add` | 88.0 | 0.491 |
| `i2_kv_store` | 48.0 | 0.395 |
| `i2p_routepick` | 48.0 | 0.378 |
| `i2_relbias` | 48.0 | 0.368 |
| `i2p_combine` | 48.0 | 0.217 |
| `i2a_expert_up` | 4.0 | 0.197 |
| `i2p_clip` | 48.0 | 0.146 |
| `i2_sig_mul` | 48.0 | 0.124 |
| `i2a_expert_down` | 4.0 | 0.103 |
| `i2a_embed` | 1.0 | 0.010 |

## Baseline K=4 GPU times

| Kernel | Calls per step | GPU ms per step |
| --- | ---: | ---: |
| `i2a_dense` | 353.0 | 12.570 |
| `i2p_gu` | 44.0 | 8.172 |
| `i2_batch_split` | 48.0 | 7.862 |
| `i2p_down` | 44.0 | 5.818 |
| `i2_conv` | 768.0 | 5.695 |
| `i2a_float` | 336.0 | 3.647 |
| `i2p_rms` | 345.0 | 3.320 |
| `i2p_gather` | 120.0 | 1.669 |
| `i2_rms` | 168.0 | 1.604 |
| `i2p_mix` | 96.0 | 1.583 |
| `i2_batch_combine` | 48.0 | 1.186 |
| `i2_relbias` | 192.0 | 1.077 |
| `i2_kv_store` | 192.0 | 1.061 |
| `i2p_ctl` | 96.0 | 1.020 |
| `i2_copy` | 155.0 | 0.982 |
| `i2p_mix4` | 24.0 | 0.931 |
| `i2_add` | 88.0 | 0.889 |
| `i2p_routepick` | 48.0 | 0.679 |
| `i2a_expert_up` | 4.0 | 0.580 |
| `i2p_combine` | 48.0 | 0.495 |
| `i2a_expert_down` | 4.0 | 0.326 |
| `i2p_clip_scaled` | 96.0 | 0.324 |
| `i2_sig_mul` | 48.0 | 0.182 |
| `i2p_clip` | 48.0 | 0.154 |
| `i2a_embed` | 1.0 | 0.013 |

## Intermediate K=4 profile at 1cb799836

Attempt 25 exited 0 after 24 admission or post-check deferrals. The retained run passed both CPU checks and GPU admission.

1 timed rounds after 8 warm: 1 encoders, 1494 stages, 1494 dispatches a round. 46.171 ms a round (46.170 in stages, 0.000 between encoders).

| Kernel | Calls per step | GPU ms per step |
| --- | ---: | ---: |
| `i2p_expert_matrix` | 88.0 | 665.775 |
| `i2a_dense_rows` | 352.0 | 338.532 |
| `i2p_attn_tiled` | 48.0 | 204.208 |
| `i2a_float` | 336.0 | 87.022 |
| `i2p_gu_grouped` | 44.0 | 81.408 |
| `i2p_down_grouped` | 44.0 | 61.414 |
| `i2a_group_up` | 4.0 | 46.503 |
| `i2p_relbias` | 48.0 | 22.529 |
| `i2a_group_down` | 4.0 | 21.956 |
| `i2p_gu` | 44.0 | 7.854 |
| `i2_batch_split` | 48.0 | 7.522 |
| `i2p_clip_scaled` | 96.0 | 6.001 |
| `i2p_down` | 44.0 | 5.777 |
| `i2p_gather_rms` | 120.0 | 5.716 |
| `i2p_conv` | 192.0 | 5.499 |
| `i2p_mix` | 96.0 | 5.024 |
| `i2a_batch_projections` | 48.0 | 5.003 |
| `i2a_dense` | 49.0 | 4.231 |
| `i2p_combine` | 48.0 | 4.134 |
| `i2p_conv_store` | 192.0 | 3.330 |
| `i2p_ctl` | 96.0 | 3.324 |
| `i2p_rms_pair` | 96.0 | 3.040 |
| `i2a_float` | 144.0 | 2.312 |
| `i2_add` | 88.0 | 2.007 |
| `i2p_mix` | 96.0 | 1.774 |
| `i2p_mix4` | 24.0 | 1.690 |
| `i2p_rms` | 69.0 | 1.567 |
| `i2a_batch_float_control` | 96.0 | 1.489 |
| `i2_rms` | 48.0 | 1.404 |
| `i2p_batch_conv` | 192.0 | 1.313 |
| `i2p_gather_rms` | 120.0 | 1.267 |
| `i2_batch_combine` | 48.0 | 1.171 |
| `i2p_clip` | 48.0 | 1.102 |
| `i2p_rms3` | 28.0 | 0.940 |
| `i2_sig_mul` | 48.0 | 0.917 |
| `i2p_combine` | 48.0 | 0.842 |
| `i2p_batch_store_rel` | 48.0 | 0.734 |
| `i2p_rms_pair` | 96.0 | 0.710 |
| `i2p_moe_zero` | 48.0 | 0.707 |
| `i2p_mix4` | 24.0 | 0.630 |
| `i2a_expert_up` | 4.0 | 0.578 |
| `i2p_rms3` | 28.0 | 0.563 |
| `i2p_rms` | 69.0 | 0.471 |
| `i2_add` | 40.0 | 0.424 |
| `i2p_routepick` | 48.0 | 0.391 |
| `i2a_expert_down` | 4.0 | 0.327 |
| `i2p_scatter` | 8.0 | 0.301 |
| `i2p_clip3` | 48.0 | 0.195 |
| `i2_rms` | 48.0 | 0.140 |
| `i2_sig_mul` | 48.0 | 0.132 |

The intermediate profile command used the baseline command above with `b1h-candidate-build` in place of `b1h-base-build`. The K=1 retained attempt was 4 and the K=4 retained attempt was 25. The profiler used `TF_METAL4_TIMELINE=1`, GPU admission, CPU admission, and the immediate CPU post-check.

The token fixture was created by:

```sh
INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- "$PWD/.work/tmp/b1h-base-build/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd/tokenizer.json" campaign/prompts/dev/code-4096.txt tokenize "$PWD/.work/tmp/b1h-code-all.json"
jaq '.[0:2500]' .work/tmp/b1h-code-all.json > .work/tmp/b1h-code-2500.json
```

Tokenization exited 0 and reported 5,265 tokens. The slice holds 2,500 IDs.

## Intermediate K=1 timing

The `1cb799836` runner used three gated alternating pairs at 2,500 cached tokens. The run then stopped the first K=8 baseline step because the K=1 target was not met and a further fusion was selected. The interrupted K=8 attempt exited 130 and was discarded. The final timing matrix must use the final runner.

| Pair | Baseline batch ms | Candidate batch ms |
| ---: | ---: | ---: |
| 1 | 24.632 | 21.198 |
| 2 | 24.735 | 21.066 |
| 3 | 24.425 | 21.134 |

## Final K=1 profile

The candidate command exited 0 on attempt 1 with GPU and CPU admission and an immediate CPU post-check. 1 timed rounds after 8 warm: 1 encoders, 1446 stages, 1446 dispatches a round. 27.888 ms a round (27.888 in stages, 0.000 between encoders).

| Kernel | Calls per step | GPU ms per step |
| --- | ---: | ---: |
| `i2_batch_split` | 48.0 | 4.709 |
| `i2p_gu` | 44.0 | 2.330 |
| `i2a_batch_projections` | 48.0 | 2.284 |
| `i2p_down` | 44.0 | 1.983 |
| `i2a_dense` | 49.0 | 1.978 |
| `i2p_batch_conv` | 192.0 | 1.930 |
| `i2a_float` | 144.0 | 1.901 |
| `i2p_mix` | 96.0 | 1.867 |
| `i2_batch_combine` | 48.0 | 1.446 |
| `i2p_mix4` | 24.0 | 0.944 |
| `i2p_combine` | 48.0 | 0.858 |
| `i2a_batch_float_control` | 96.0 | 0.813 |
| `i2_sig_mul` | 48.0 | 0.766 |
| `i2p_rms_pair` | 96.0 | 0.737 |
| `i2p_routepick` | 48.0 | 0.689 |
| `i2p_gather_rms` | 120.0 | 0.636 |
| `i2p_rms3` | 28.0 | 0.563 |
| `i2p_rms` | 69.0 | 0.484 |
| `i2p_batch_store_rel` | 48.0 | 0.353 |
| `i2a_expert_up` | 4.0 | 0.202 |
| `i2p_clip3` | 48.0 | 0.175 |
| `i2a_expert_down` | 4.0 | 0.104 |
| `i2_add` | 40.0 | 0.097 |
| `i2p_scatter` | 8.0 | 0.020 |
| `i2a_embed` | 1.0 | 0.011 |

## Final K=4 profile

The candidate command exited 0 on attempt 1 with GPU and CPU admission and an immediate CPU post-check. 1 timed rounds after 8 warm: 1 encoders, 1446 stages, 1446 dispatches a round. 48.485 ms a round (48.485 in stages, 0.000 between encoders).

| Kernel | Calls per step | GPU ms per step |
| --- | ---: | ---: |
| `i2p_expert_matrix` | 88.0 | 695.234 |
| `i2a_dense_rows` | 352.0 | 355.550 |
| `i2p_attn_tiled` | 48.0 | 210.981 |
| `i2a_float` | 336.0 | 88.966 |
| `i2p_gu_grouped` | 44.0 | 84.239 |
| `i2p_down_grouped` | 44.0 | 62.784 |
| `i2a_group_up` | 4.0 | 52.432 |
| `i2a_group_down` | 4.0 | 24.681 |
| `i2p_relbias` | 48.0 | 23.372 |
| `i2p_gu` | 44.0 | 8.497 |
| `i2_batch_split` | 48.0 | 7.665 |
| `i2p_down` | 44.0 | 6.567 |
| `i2p_clip_scaled` | 96.0 | 6.295 |
| `i2p_gather_rms` | 120.0 | 5.957 |
| `i2a_batch_projections` | 48.0 | 5.373 |
| `i2p_conv` | 192.0 | 5.346 |
| `i2p_mix` | 96.0 | 4.831 |
| `i2a_dense` | 49.0 | 4.542 |
| `i2p_combine` | 48.0 | 4.310 |
| `i2p_conv_store` | 192.0 | 3.403 |
| `i2p_ctl` | 96.0 | 3.144 |
| `i2p_rms_pair` | 96.0 | 2.935 |
| `i2a_float` | 144.0 | 1.941 |
| `i2_add` | 88.0 | 1.916 |
| `i2p_mix` | 96.0 | 1.899 |
| `i2p_batch_conv` | 192.0 | 1.857 |
| `i2p_mix4` | 24.0 | 1.687 |
| `i2_batch_combine` | 48.0 | 1.476 |
| `i2_rms` | 48.0 | 1.452 |
| `i2p_rms` | 69.0 | 1.364 |
| `i2a_batch_float_control` | 96.0 | 1.170 |
| `i2p_clip` | 48.0 | 1.150 |
| `i2p_mix4` | 24.0 | 0.945 |
| `i2p_combine` | 48.0 | 0.942 |
| `i2p_rms3` | 28.0 | 0.916 |
| `i2p_moe_zero` | 48.0 | 0.848 |
| `i2_sig_mul` | 48.0 | 0.845 |
| `i2_sig_mul` | 48.0 | 0.743 |
| `i2p_rms_pair` | 96.0 | 0.715 |
| `i2p_routepick` | 48.0 | 0.703 |
| `i2p_gather_rms` | 120.0 | 0.684 |
| `i2a_expert_up` | 4.0 | 0.593 |
| `i2p_rms3` | 28.0 | 0.565 |
| `i2p_rms` | 69.0 | 0.499 |
| `i2p_batch_store_rel` | 48.0 | 0.421 |
| `i2a_expert_down` | 4.0 | 0.339 |
| `i2p_clip3` | 48.0 | 0.200 |
| `i2_add` | 40.0 | 0.105 |
| `i2p_scatter` | 8.0 | 0.022 |
| `i2a_embed` | 1.0 | 0.011 |

The final profile used the baseline command above with `b1h-candidate2-build` replacing `b1h-base-build`.

## Final correctness and fidelity

The final native build and separate `b1h-candidate2-build` runner build both exited 0 using the commands above with the corresponding prefix. The required `INKLING_LOAD_LIMIT_GB=26 python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0. The GPU build reported 73 of 73 tests passed and no skips. The oQ4e family test command above exited 0 and reported 73 of 73 tests passed, with zero batch logit and cache differences in its tested rows. The original tolerance and argmax checks remained unchanged.

The 16-history oracle verifier exited 0. The capture and streamed comparison commands above were repeated with `b1h-candidate2-build` in place of `b1h-candidate-build`. Both exited 0. The final result was 2,775 of 3,015 top-1 matches (92.0398%), mean KL 0.039272796 and zero non-finite logits.

## Final K=1 long-cache pairs

Each side used `batchbench 1 5` with the 2,500-token fixture, GPU admission and CPU admission and post-checks. The order was baseline, candidate; candidate, baseline; baseline, candidate. All six retained commands exited 0. The candidate median of the three run medians is 21.007 ms, above the strict 21 ms target. The measured candidate samples span 20.918–21.079 ms.

| Pair | Baseline batch ms | Candidate batch ms | Candidate sample range ms |
| ---: | ---: | ---: | ---: |
| 1 | 24.473 | 21.016 | 20.985-21.078 |
| 2 | 24.509 | 21.007 | 20.927-21.079 |
| 3 | 24.551 | 20.993 | 20.918-21.021 |

## Final alternating timing matrix

The matrix command exited 0. All 60 runs passed GPU admission, CPU admission, and the immediate CPU post-check. The monitor rejected compiler overlap. Each cell uses three alternating pairs, ordered baseline/candidate, candidate/baseline, baseline/candidate. Batch entries are run medians of five timed steps in milliseconds. Prefill entries are milliseconds for 1,024 or 4,096 prompt tokens with 512-token chunks. The short-cache batch fixture holds 1,024 IDs. The long-cache fixture starts at 2,500 IDs, less the row number. KV remains f32.

| Cell | Baseline pair results, ms | Candidate pair results, ms | Baseline median, ms | Candidate median, ms |
| --- | --- | --- | ---: | ---: |
| k1long | 24.473, 24.509, 24.551 | 21.016, 21.007, 20.993 | 24.509 | 21.007 |
| k2long | 30.170, 30.814, 30.196 | 25.843, 25.920, 25.855 | 30.196 | 25.855 |
| k4long | 45.375, 45.290, 45.134 | 39.554, 39.719, 39.654 | 45.290 | 39.654 |
| k8long | 81.096, 79.775, 79.817 | 71.805, 71.813, 71.862 | 79.817 | 71.813 |
| k1short | 23.723, 23.657, 23.447 | 20.117, 20.096, 20.018 | 23.657 | 20.096 |
| k2short | 29.185, 29.158, 29.112 | 24.907, 24.910, 25.027 | 29.158 | 24.910 |
| k4short | 43.956, 45.183, 45.718 | 38.658, 39.135, 39.957 | 45.183 | 39.135 |
| k8short | 81.357, 78.843, 78.071 | 73.946, 71.753, 69.387 | 78.843 | 71.753 |
| prefill1k | 2937.800, 2892.700, 2893.400 | 2862.300, 2856.000, 2853.100 | 2893.400 | 2856.000 |
| prefill4k | 11758.100, 11794.100, 11916.500 | 11682.100, 11668.800, 11728.800 | 11794.100 | 11682.100 |

The K=1 long-cache candidate median is 21.007 ms. It is 0.007 ms above the strict 21 ms target. The 15 candidate samples span 20.918–21.079 ms, so this difference is smaller than the measured spread. The three K=4 long-cache medians were 45.290 ms baseline and 39.654 ms candidate. Both prefill candidate medians are lower than their baseline medians; the prefill differences are small relative to the observed run-to-run range.

Set the environment and create `.work/tmp/b1h-measure.py` and `.work/tmp/b1h-matrix.sh` from the following blocks before running `sh .work/tmp/b1h-matrix.sh`. The first program wraps each model run in `campaign/heavy.py`, checks GPU and CPU admission, checks CPU immediately after the model, and rejects compiler overlap. The shell script retries exit 75 and stops on other failures.

```python
import os
from pathlib import Path
import subprocess
import sys
import time

log = Path(sys.argv[1])
command = sys.argv[2:]
wrapper = '''gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; printf 'B1H_WINDOW_START\\n'; "$@"; result=$?; perf-gate-ready -1 || exit 75; printf 'B1H_WINDOW_END\\n'; exit "$result"'''
with log.open('w') as out:
    proc = subprocess.Popen(['python3', 'campaign/heavy.py', '--', 'sh', '-c', wrapper, 'sh', *command], stdout=out, stderr=subprocess.STDOUT, env=os.environ.copy())
    overlap = False
    while proc.poll() is None:
        time.sleep(1)
        data = log.read_text(errors='replace')
        if 'B1H_WINDOW_START' in data and 'B1H_WINDOW_END' not in data:
            listing = subprocess.run(['ps', '-Ao', 'comm'], capture_output=True, text=True, check=True).stdout
            if any(Path(line.strip()).name in ('zig', 'clang', 'rustc') for line in listing.splitlines()):
                overlap = True
if proc.returncode not in (0, 75):
    raise SystemExit(proc.returncode)
if overlap:
    with log.open('a') as out:
        out.write('B1H_REJECT compiler overlap\n')
    raise SystemExit(75)
raise SystemExit(proc.returncode)
```

```sh
#!/bin/sh
set -u
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26 GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60
for cell in k1long k8long k2long k4long k1short k2short k4short k8short prefill1k prefill4k; do
    case "$cell" in
        k1short) ids=campaign/prompts/inkling/code-1024.json; set -- batchbench 1 5 ;;
        k2short) ids=campaign/prompts/inkling/code-1024.json; set -- batchbench 2 5 ;;
        k4short) ids=campaign/prompts/inkling/code-1024.json; set -- batchbench 4 5 ;;
        k8short) ids=campaign/prompts/inkling/code-1024.json; set -- batchbench 8 5 ;;
        k1long) ids=.work/tmp/b1h-code-2500.json; set -- batchbench 1 5 ;;
        k2long) ids=.work/tmp/b1h-code-2500.json; set -- batchbench 2 5 ;;
        k4long) ids=.work/tmp/b1h-code-2500.json; set -- batchbench 4 5 ;;
        k8long) ids=.work/tmp/b1h-code-2500.json; set -- batchbench 8 5 ;;
        prefill1k) ids=campaign/prompts/inkling/code-1024.json; set -- bench 32 4 512 ;;
        prefill4k) ids=.work/tmp/b1h-code-4096.json; set -- bench 32 4 512 ;;
    esac
    for pair in 1 2 3; do
        if [ "$pair" -eq 2 ]; then order='candidate base'; else order='base candidate'; fi
        for side in $order; do
            if [ "$side" = candidate ]; then build=candidate2; else build=base; fi
            binary="$PWD/.work/tmp/b1h-$build-build/bin/tf-inkling2-loop-run"
            attempt=0
            while :; do
                attempt=$((attempt + 1))
                log=".work/tmp/b1h-final2-$cell-$pair-$side-$attempt.log"
                python3 .work/tmp/b1h-measure.py "$log" "$binary" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" "$ids" "$@"
                code=$?
                printf '%s pair=%s side=%s attempt=%s exit=%s log=%s\n' "$cell" "$pair" "$side" "$attempt" "$code" "$log"
                if [ "$code" -eq 0 ]; then break; fi
                if [ "$code" -ne 75 ]; then exit "$code"; fi
                sleep 5
            done
        done
    done
done
```

The 4K fixture was `jaq '.[0:4096]' .work/tmp/b1h-code-all.json > .work/tmp/b1h-code-4096.json` and contains 4,096 IDs.

## Served-agent sweeps

The server was rebuilt with `/opt/homebrew/bin/zig build native -Dcpu=apple_m1 -j1`; the build exited 0. The following commands each exited 0 on their first attempt. GPU admission and CPU admission and post-checks passed. The sweep script also performs CPU checks around its internal measurements.

```sh
INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" sh campaign/fx/sweep-subagents.sh 1 4096
INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" sh campaign/fx/sweep-subagents.sh 4 4096
```

The actual command was `sh .work/tmp/b1h-sweep.sh`. It exited 0. Create that file from the following wrapper to reproduce its admission, post-check, retry, and recording behavior:

```sh
#!/bin/sh
set -u
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26 GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60
for n in 1 4; do
    attempt=0
    while :; do
        attempt=$((attempt + 1))
        log=".work/tmp/b1h-sweep-$n-$attempt.log"
        (
            gpu-gate-ready -1 || exit 75
            perf-gate-ready || exit 75
            INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" sh campaign/fx/sweep-subagents.sh "$n" 4096
            result=$?
            perf-gate-ready -1 || exit 75
            exit "$result"
        ) > "$log" 2>&1
        code=$?
        base="$PWD/.work/tmp/sessions-Inkling-2-Mini-oQ4e-mtp@3eea042751cd-$n-4096"
        if [ -f "$base/server.log" ]; then
            cp "$base/server.log" ".work/tmp/b1h-sweep-$n-$attempt-server.log"
            if rg -q 'stopped over cap|process tree charged .* exceeded' "$base/server.log"; then exit 3; fi
        fi
        if [ -f "$base/results.jsonl" ]; then cp "$base/results.jsonl" ".work/tmp/b1h-sweep-$n-$attempt-results.jsonl"; fi
        if [ -f "$base/concurrent.json" ]; then cp "$base/concurrent.json" ".work/tmp/b1h-sweep-$n-$attempt-concurrent.json"; fi
        printf 'n=%s attempt=%s exit=%s log=%s\n' "$n" "$attempt" "$code" "$log"
        if [ "$code" -eq 0 ]; then break; fi
        if [ "$code" -eq 75 ]; then sleep 5; continue; fi
        if [ "$code" -eq 1 ] && rg -q 'host busy:|perf-gate-ready:' "$log" && ! rg -q 'stopped over cap|process tree charged .* exceeded|concurrent reply differs|ValueError:' "$log"; then sleep 5; continue; fi
        exit "$code"
    done
done
```

The per-request numbers are the server's `tok/s` values, ordered by the sweep's request IDs. Aggregate rates cover each multi-turn repetition and include its prefill and other client time.

| Agents | Repetition | Aggregate tok/s | Server decode tok/s per request |
| ---: | ---: | ---: | --- |
| 1 | 1 | 22.841 | 47.3, 47.2, 47.2, 47.1 |
| 1 | 2 | 41.690 | 47.4, 47.3, 47.2, 47.1 |
| 4 | 1 | 47.869 | 14.0, 16.6, 16.1, 22.5, 25.0, 19.1, 16.8, 19.7, 17.9, 18.9, 19.9, 15.8, 17.8, 17.8, 15.7, 17.4 |
| 4 | 2 | 60.313 | 21.2, 17.7, 14.9, 22.0, 18.1, 19.1, 14.7, 17.2, 18.2, 20.3, 18.4, 15.2, 15.8, 20.6, 15.7, 18.7 |

The N=4 concurrent-versus-solo check covered streams 1, 2, and 4 in two repetitions. Every `alone` comparison had zero unequal and zero failed replies. The per-cell equal counts were 2, 4, and 8 in each repetition. Serial equality also had zero unequal and failed replies, but its drafted comparison was vacuous because no draft was accepted. The N=1 and N=4 guards reported peak charges of 20.5 GB and 23.4 GB under the 28 GB cap.

## Acceptance

The tests, both profiles, the complete timing matrix, both sweeps, and streamed fidelity passed their commands. The K=4 speed and prefill criteria passed. The strict K=1 long-cache threshold was not met: the median was 21.007 ms against a 21 ms maximum. The 0.007 ms excess is smaller than the observed sample spread, so these runs do not establish that 21 ms is unattainable. They also do not establish a passing result.
