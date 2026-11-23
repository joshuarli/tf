# Inkling oQ4e decode step

No optimization was attempted. The required weight-floor profiler does not support the target checkpoint, and its source is outside this lane's writable paths.

At base revision `b9bdf4e55f3b64855ce52a6359efb15463e5ae14`, the following command exited 1 twice:

```sh
PATH="/opt/homebrew/bin:$PATH" INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- python3 campaign/kernels/inkling_profile.py weights --model "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" --gbs 521
```

It reported `the text stack must be one shard` and listed four safetensors shards. `inkling_profile.py` also expects `model.llm.layers.*` tensor names and assumes a BF16 unembedding, whereas this checkpoint's index contains `model.layers.*`, `lm_head.weight`, `lm_head.scales`, and `lm_head.biases`. Supporting this checkpoint requires changes to the profiler's accounting, not just its shard reader.

The base runner build exited 0:

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache"
zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/base-build"
```

The K=1 and K=4 timeline profiles, interleaved timings, saved-capture fidelity, suite, native build, and four-agent sweep remain unverified. No decode improvement or throughput claim is supported.
