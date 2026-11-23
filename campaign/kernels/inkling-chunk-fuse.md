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

## Correctness

The native build `/opt/homebrew/bin/zig build native -Dcpu=apple_m1 -j1` exited 0. The timing runner was built separately with `/opt/homebrew/bin/zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/b1h-candidate-build"` and exited 0. Both used `ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache"` and `TMPDIR="$PWD/.work/tmp"`.

`INKLING_LOAD_LIMIT_GB=26 python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0. The GPU summary reported 73 of 73 tests passed, with no skips. The unchanged batch test checked logits and cache differences against `1.5e-5` and checked argmax. The suite also includes rollback and fork tests.

The oQ4e checkpoint ran the same family tests with `TF_INKLING2_LOOP_DIR` and `TF_INKLING2_LOOP_OQ4E_DIR` both set to `$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd`:

```sh
INKLING_LOAD_LIMIT_GB=26 TF_INKLING2_LOOP_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" TF_INKLING2_LOOP_OQ4E_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" python3 campaign/heavy.py -- /opt/homebrew/bin/zig build test-inkling2-loop -Dcpu=apple_m1 -j1 --summary all
```

The command exited 0 with 73 of 73 tests passed. The batch comparison reported zero logit and cache difference for all tested rows at widths 1, 2 and 4, including the cache-boundary case. Peak charged memory was 23.9 GB.

## Streamed fidelity

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
