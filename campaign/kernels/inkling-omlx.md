# Inkling 2 Mini: oMLX 0.8.3

The comparator uses oMLX commit `e72868dac12df4be4ced2dc8355f031686283974`,
MLX 0.32.4, and `mlx-community/Inkling-2-Mini-oQ4e-mtp` at
`3eea042751cd296b7266628a8342c1749242e1fe`. The source archive and every
checkpoint file have pinned sizes and SHA-256 values in `campaign/sources.lock.json`.
The environment is locked in `campaign/comparators/omlx/uv.lock`.
The Python 3.13 environment contains 105 packages and occupies 1.1 GiB on this
host. Build dependencies have separate hash constraints.

The model keeps NVFP4 experts in layers 0–19, uses affine 4-bit experts in
layers 20–23, and uses affine 8-bit attention, embeddings, and output head.
The checkpoint is therefore not numerically identical to the original.

The adapter uses `VLMBatchedEngine` in process. It passes token IDs directly to
the scheduler and takes token IDs and producer timestamps from its output.
It starts TTFT immediately before submitting a request. Decode rate counts
committed tokens after the first token over the producer's decode interval.
The smoke times the first request after loading, including any first-request
kernel compilation. It is a single sample for each mode, not a speed ranking.
The receipt records `heavy.py` peak charged memory and MLX peak used memory.
The adapter resets the MLX peak after model and drafter loading.
`ignore_eos` keeps each shootout completion at 512 tokens. Prefix caching is
disabled. `ModelSettings.turboquant_kv_enabled` defaults to `False`, so KV
compression is off. The `turboquant_kv_enabled` setting controls it.

Inkling's model code invokes MLX kernels directly. The native extension is not
built for this adapter: Inkling uses its NAX detection helper, which falls back
to MLX device information when the extension is absent. The pinned source uses
MLX's Metal kernels for the Inkling operators. The source extraction is checked
against the archive before setup accepts it. The installed oMLX Python files
are checked against that extraction.

## Original checkpoint probe

The original `Inkling-2-Mini@ed40e7da625f` loaded successfully on the M5 Max
under `heavy.py` with a 26 GB cap. The in-process engine produced IDs
`[200001, 200008]` after the checked-in 1,024-token prompt. Peak charged memory
was 23.4 GB. This two-token probe establishes loading, not output fidelity.
The probe disabled Lightning MTP and KV compression. GPU admission and the
pre- and post-run performance checks passed. Earlier admission attempts
were rejected for host load.

Repeat the probe from the primary checkout with:

```sh
env PATH="$PWD/.work/venv-omlx/bin:$PATH" \
  HF_HUB_OFFLINE=1 INKLING_LOAD_LIMIT_GB=26 \
  GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 \
  python3 campaign/heavy.py --cap-gb 26 -- sh -c '
    gpu-gate-ready -1 || exit 75
    perf-gate-ready || exit 75
    .work/venv-omlx/bin/python campaign/comparators/omlx/run.py --worker \
      --model "$HOME/models/Inkling-2-Mini@ed40e7da625f" \
      --prompt campaign/prompts/inkling/code-1024.json --steps 2
    result=$?
    perf-gate-ready -1 || exit 75
    exit "$result"
  '
```

## Smoke

`make -C campaign omlx-setup` verified the source archive, all 15 checkpoint
files, and 624 installed oMLX Python and JSON files. The CPU adapter tests pass.
`make -C campaign omlx-smoke` passed with one busy-host sample per mode. Each
process received the same 1,024 IDs and generated 512 tokens at temperature 0.

| Lightning MTP | TTFT seconds | Prefill tokens/s | Decode tokens/s | Peak charged GB |
| --- | ---: | ---: | ---: | ---: |
| off | 0.846 | 1,210.2 | 56.17 | 17.9 |
| on | 1.830 | 559.6 | 59.91 | 25.6 |

All 512 generated IDs matched between the two runs. The on run reported 182
draft-and-verify cycles. The drafter reported an eight-position block and an
adaptive maximum of seven draft tokens. Peak charged memory was 0.4 GB below
the watchdog cap. This single pair cannot establish a speed difference or
general greedy equivalence. The [smoke receipt](../comparators/omlx/results/code-1024-1795386552036119000.json)
contains the IDs, commands, gate attempts, and raw output.

For a peak-delta estimate, the MTP-off 1,024-token run reached 18,834,789,592
bytes of MLX peak used memory. The MTP-off `code-4096` corpus run used 5,280
tokens and reached 21,250,879,084 bytes. Both generated 512 tokens. The delta
is 2,416,089,492 bytes over 4,256 extra prompt tokens, or approximately
**567,690 bytes per context token**. This is a peak-memory proxy, not a direct
KV allocation: it also includes temporary prefill workspace and other peak
allocations. It is not suitable as a pure KV projection. The longer run peaked
at 23.9 GB charged. Its [receipt](../comparators/omlx/results/code-4096-1795387434631940000.json)
contains the command, output IDs, and gate attempts. The 16K memory cell has
not been measured.

## Shootout inputs

The `code`, `prose`, and `structured` corpus prompts labelled 1K contain
1,349, 1,154, and 1,124 Inkling tokens. Those labelled 4K contain 5,280,
4,639, and 4,203 tokens. Each 16K prompt contains 16,384 tokens. The 16K
arrays keep the first 16,380 IDs and the four final IDs from the corresponding
32K corpus prompt. All nine arrays matched the nanoattention lane byte for byte.
The smoke instead uses the checked-in 1,024-ID `code-1024.json` fixture.

Create the nine arrays from the primary checkout with:

```sh
make -C campaign omlx-setup
mkdir -p .work/tmp/omlx-prompts
for size in 1024 4096 16384; do
  source_size=$size
  if test "$size" = 16384; then source_size=32768; fi
  for topic in code prose structured; do
    .work/venv-omlx/bin/python campaign/comparators/omlx/prompt_ids.py \
      "$HOME/models/Inkling-2-Mini@ed40e7da625f" \
      "campaign/prompts/dev/$topic-$source_size.txt" "$size" \
      ".work/tmp/omlx-prompts/$topic-$size.json"
  done
done
```

For each cell, run the following command once for each oMLX mode in each of
three interleaved engine pairs. Set `PROMPT` to an absolute generated path and
`MODE` to `on` or `off`. The campaign coordinator supplies the alternating engine
order. Each command requests 512 greedy tokens and records a separate receipt.

```sh
PROMPT="$PWD/.work/tmp/omlx-prompts/code-1024.json" MODE=off
make -C campaign omlx-cell OMLX_PROMPT="$PROMPT" OMLX_MODE="$MODE"
```

For the original-checkpoint cell, add
`OMLX_MODEL="$HOME/models/Inkling-2-Mini@ed40e7da625f"`.

Run `make -C campaign omlx-tests` for the CPU adapter tests and
`make -C campaign omlx-smoke` for the 1K on/off smoke. These single-engine
commands do not themselves run the other engines or the fidelity, quality,
and Pi cells.
