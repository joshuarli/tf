# Inkling batched attention

The baseline is `4875e3f98b077bf1230dd0644b21d43a0ccd3943`, with the batch timeline named separately from prefill.
On the oQ4e checkpoint, four caches were prefilled with the first 2,500, 2,499, 2,498 and 2,497 tokens of
`campaign/prompts/dev/code-4096.txt`. One instrumented batch round after eight warmups had 3,702 dispatches and
a 76.191 ms GPU span. Split attention accounted for 19.739 ms across 192 dispatches; combine took 4.258 ms across
192. The GPU and CPU admission and CPU post-check passed. The run exited 0 and `heavy.py` reported a peak charge
of 20.7 GB under its 28 GB cap. Instrumented spans are attribution, not throughput measurements.

The batched split and combine candidate passed the same gates and exited 0 at 20.7 GB peak charge. Its one
instrumented round had 3,414 dispatches and a 62.014 ms span. Split took 7.987 ms across 48 dispatches and combine
took 1.142 ms across 48. The host and GPU suite exited 0, with 73/73 GPU tests and no skips; its peak charge was
23.1 GB. The cache remained f32 for these measurements.

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp"
export ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26
export GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60
/opt/homebrew/bin/zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/profile-base-build"
python3 campaign/heavy.py -- "$PWD/.work/tmp/base-build/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd/tokenizer.json" campaign/prompts/dev/code-4096.txt tokenize "$PWD/.work/tmp/code-all.json"
jaq '.[0:2500]' .work/tmp/code-all.json > .work/tmp/code-2500.json
TF_METAL4_TIMELINE=1 python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh "$PWD/.work/tmp/profile-base-build/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" .work/tmp/code-2500.json batchbench 4 9 > .work/tmp/profile-base.log 2>&1
python3 campaign/kernels/inkling_profile.py timeline --log .work/tmp/profile-base.log
/opt/homebrew/bin/zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/candidate-build"
TF_METAL4_TIMELINE=1 python3 campaign/heavy.py -- sh -c 'gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh "$PWD/.work/tmp/candidate-build/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" .work/tmp/code-2500.json batchbench 4 9 > .work/tmp/profile-candidate.log 2>&1
python3 campaign/kernels/inkling_profile.py timeline --log .work/tmp/profile-candidate.log
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
```

## f16 KV attempt

An uncommitted candidate changed global and local KV storage to f16, retained f32 accumulators and convolution
rings, rounded current prefill KV to f16 before attention, and changed rollback, verification and copy offsets.
It was removed after failing qualification. The committed implementation therefore still stores KV in f32.

The saved-oracle capture verified all 16 histories. Its standard f32 cross-checkpoint parity command exited 2,
as that contract differs from the oQ4e fidelity gate. The streamed comparison used the eight-row script in
[Inkling oQ4e fast](inkling-oq4e-fast.md) with `outdir` set to `.work/tmp/half-parity`. It exited 0 and reported
2,756/3,015 top-1 matches (91.4096%), mean KL 0.0389854086 and zero non-finite logits. The top-1 floor is 91.5%.
Sixteen guarded model processes completed, with a maximum `heavy.py` charge of 20.2 GB.

The f16 suite command below exited 1: the GPU build reported 69/73 tests passed and four failures across two
test executables. The prefill-versus-decode fixture reported maximum logit difference `3.077276e-5`, key and value
differences `0.00390625`, against the existing `1.5e-5` bound; the batch-versus-decode fixture also failed its
existing logit bound. Rollback and verification checks passed. These measurements establish a failed candidate,
not that an f16 implementation cannot meet the gate. No tolerance was relaxed. The f16 candidate was not retained,
so the commands below document the experiment and do not reproduce its binary from the final tree.

```sh
PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26 python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu > .work/tmp/suite-half.log 2>&1
PATH="/Users/josh/repos/tf/.work/venv-ref/bin:/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26 make -C campaign -f parity/inkling2-mini/parity.mk parity-inkling2-mini RUNNER_MODE=prefill PREFILL_CHUNK=512 LANE=b1g-half REPORT="$PWD/.work/tmp/half-parity.md" METAL_LOGITS="$PWD/.work/tmp/half-parity" WORK="$PWD/.work" REF_LOGITS="/Users/josh/repos/tf/.work/parity/inkling2-mini/ref" REF_UV=env "REF_VENV=''" RUNNER="$PWD/.work/tmp/half-build/bin/tf-inkling2-loop-run" INKLING_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" > .work/tmp/half-parity.log 2>&1
/Users/josh/repos/tf/.work/venv-ref/bin/python .work/tmp/half-compare.py > .work/tmp/half-compare.log 2>&1
```

The alternating timing matrix, final f16 profile and four-agent sweep were not run. The f32 batch candidate's
profile and tests do not establish the requested end-to-end performance target.
