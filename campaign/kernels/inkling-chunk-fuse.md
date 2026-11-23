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
