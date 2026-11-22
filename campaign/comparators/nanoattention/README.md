# Inkling 2 Mini: nanoattention comparator

Pinned to [justrach/nanoattention `6f782a8d`](https://github.com/justrach/nanoattention/tree/6f782a8dace846e4e64a5a12cbdf5c0252166199).
That revision of `main` includes the experimental Inkling engine. This adapter uses its **`nano bench`** CLI;
the older `server.py` and `run_nano.py` adapter in this directory is for Flint.

## On the M5

From the repository root, with the campaign's pinned Zig and usual Mac tools:

```sh
make -C campaign nanoattention-setup
make -C campaign nanoattention-smoke
make -C campaign nanoattention-bench
```

Setup verifies the pinned source archive and original Inkling checkpoint, builds only nanoattention's `nano`
executable, and builds the current TensorFold raw-token comparator. No model conversion is needed. Artifacts
and the build receipt stay under the main checkout's `.work/build/`. A named build target is added only in
the extracted comparator tree; it selects `nano` without building unrelated Flint targets or their Lean
proof-binding gates. This is not a claim that nanoattention's entire upstream build or proof suite passed.
The original source and license files are preserved. The setup refuses a changed extraction rather than
overwriting it. Actual Zig version, build patch, source archive and binary hashes are recorded.

The smoke runs one process pair at 512 prompt tokens and 16 generated tokens. The bench runs three
alternating process pairs at 512 and 1,024 prompt tokens, with 512 generated tokens each. Each process does
one complete untimed warmup followed by one timed run. Start with the smoke; build or runtime failures
are useful results and must be fixed or recorded before interpreting speed.

```sh
# Inspect the exact workload without loading a model; also works on Linux.
make -C campaign nanoattention-plan
# Fetch, verify and prepare the source without attempting a Mac build.
make -C campaign nanoattention-prepare
# Compare upstream's reference configuration separately from its optimized BF16/tiled bundle.
make -C campaign nanoattention-smoke NANO_VARIANT=reference
# A larger sample, only after the smoke is sound.
make -C campaign nanoattention-bench NANO_PAIRS=6
```

`NANO_PROMPTS`, `NANO_STEPS`, `NANO_PAIRS` and `NANO_VARIANT` select the workload. Do not change host gate
thresholds to obtain a clean result. If the owner already uses a relaxed busy-host threshold, the exact
environment settings are recorded and those results remain busy-host measurements.

## What the comparison measures

- Both engines read the same original packed checkpoint, and the helper verifies its actual hashes before
  timing. Both receive exactly the checked-in `campaign/prompts/inkling/*.json` token IDs, without templates,
  retokenization, prompt repetition, speculation or EOS termination.
- Context is at most **32,768 total tokens**, including all generated tokens. Oversized inputs are refused
  before loading a model. A 512-token completion leaves at most 32,256 prompt tokens.
- Nanoattention's first generated token is the prefill argmax. Thus 512 generated tokens means
  `--decode-tokens 511`; decode throughput uses 511 divided by decode seconds in both engines.
- Timings cover engine prefill plus first argmax, then remaining decode. They exclude loading and warmup;
  these are **not HTTP TTFT, cold-start or prefix-cache measurements**. TF uses its current default prefill
  chunk; nano uses explicit `batch=128`. Both are recorded rather than presented as identical kernels.
- Each model process runs under `heavy.py`'s GPU lock and 28 GB charged-memory watchdog. Pre/post GPU,
  memory and thermal gates are required. Other-process GPU time is measured across the full child process,
  including load and warmup, and labelled with that scope. Failed GPU samples or changed other-client sets
  invalidate the interval. No concurrent GPU timing is launched.
- Receipts keep source/build/toolchain/checkpoint and prompt identities, all generated IDs, first output
  difference, timing scope, host conditions, raw output, failures and the charged-memory receipt. Failed
  attempts remain in the file; they are not silently retried until a fast run appears.

Results go to `campaign/comparators/nanoattention/results/inkling-<UTC timestamp>.jsonl`; scratch logs go to
`.work/tmp/nano-inkling-<timestamp>/`. This engine-only protocol stays separate from the served scorecard.
`valid_timing` means the run completed its declared timing protocol under the recorded conditions. It does
not certify numerical equivalence. The command can succeed while recording different greedy token IDs.

## Numerical status and next decision

Upstream explicitly labels Inkling support experimental and reports **failed MLX logit parity** for its
reference and optimized configurations. The original checkpoint is shared, but nanoattention uses BF16
rounding, with additional expert BF16 and relative-attention FP16 tiles in the optimized bundle. TF's
campaign target is the existing float32 activation configuration. Equal greedy output on a short prompt
would not establish equal logits or long-history behavior. See upstream's
[Inkling notes](https://github.com/justrach/nanoattention/blob/6f782a8dace846e4e64a5a12cbdf5c0252166199/docs/MEGAKERNEL.md).

If nano is faster on this M5, use the results to pick a bounded kernel or scheduling experiment, then
qualify that TF change under its declared arithmetic contract. Do not promote borrowed arithmetic,
weaken the logit gate, or claim a competitive win from these short diagnostic pairs alone.

The integration was prepared and host-tested on Linux. Mac compilation, runtime smoke, numerical output
comparison and M5 performance remain unmeasured until these commands run on the Mac.
