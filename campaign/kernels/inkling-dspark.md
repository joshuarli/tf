# Inkling 2 Mini DSpark acceptance

**NO-GO** for native DSpark on this evidence. The best projected prose rate is
45.27 tok/s, 10.02 tok/s below the measured 55.288 tok/s copy hybrid. The
zero-drafter-cost ceiling for this model is 45.91 tok/s, still 9.38 tok/s short.
All nine MLX speculative outputs differ from the corresponding MLX AR output.
These results do not establish that a native implementation would be exact.

## Measurement

On Apple M5 Max, MLX-VLM `f1618d6d52be` loaded Inkling 2 Mini checkpoint
`ed40e7da625f`, including its bundled BF16 `mtp.safetensors`. Each run used
one user prompt from `campaign/prompts/dev`, the checkpoint chat template,
`temperature=0`, prefill chunks of 128, and 256 generated tokens with no EOS
stopping. The rendered code, prose, and structured prompts contain 1349, 1154,
and 1124 tokens respectively. The same prompt IDs and target model were used
for AR and each width. The runner uses MLX-VLM's `generate_step` and
`load_drafter`, with explicit `draft_block_size` values. There was one sample
per setting. These are busy-host observations, not a statistically qualified
speed comparison. Each process passed the GPU and performance admission checks,
and the performance check passed immediately afterward. The GPU was shared with
other lanes; no claim of an isolated measurement follows from admission.

A block contains one anchor and width minus one proposals. The table counts
accepted proposals in full-width verification rounds, using MLX-VLM's recorded
`accept_lens` and `draft_lens`. A shortened final round is excluded where
present. Committed per round is one plus the average accepted proposals. A
final full-width round can have its emitted output truncated by the 256-token
budget, while the acceptance count still describes its complete verification.
AR equality compares the SHA-256 of the full 256-token ID sequence, serialized
identically in the runner. All nine comparisons failed.

| Prompt | Width | Full rounds | Accepted proposal counts, 0 through width−1 | Committed per round | MLX DSpark tok/s | Equal AR |
| :--- | ---: | ---: | :--- | ---: | ---: | :--- |
| code | 8 | 62 | 9, 11, 12, 5, 7, 7, 2, 9 | 4.032 | 42.43 | no |
| code | 4 | 84 | 14, 16, 7, 47 | 3.036 | 54.77 | no |
| code | 2 | 136 | 17, 119 | 1.875 | 44.17 | no |
| prose | 8 | 68 | 16, 16, 5, 7, 6, 4, 5, 9 | 3.706 | 42.89 | no |
| prose | 4 | 95 | 23, 22, 13, 37 | 2.674 | 47.41 | no |
| prose | 2 | 139 | 23, 116 | 1.835 | 42.78 | no |
| structured | 8 | 66 | 15, 13, 4, 11, 4, 5, 3, 11 | 3.879 | 44.29 | no |
| structured | 4 | 86 | 12, 21, 12, 41 | 2.953 | 49.19 | no |
| structured | 2 | 143 | 30, 113 | 1.790 | 36.25 | no |

MLX AR decode rates were 33.87, 33.76, and 33.83 tok/s for code, prose,
and structured. Rates exclude prefill and the first generated token, and count
255 subsequent IDs over the time to consume them. The runner checks the 256-ID
length. Lazy work is consumed by iteration. One process per prompt held one
target and its drafter, under `heavy.py` and its 28 GB cap. Peak charged memory
was 25.3, 25.5, and 25.0 GB respectively. Raw summary records are in
[dspark-accept.jsonl](../comparators/mlx-vlm/runs/dspark-accept.jsonl).

The pinned MLX implementation registers `inkling2_loop` for its Inkling DSpark
splitter (`mlx_vlm/speculative/drafters/mtp_split.py:352-355`). The splitter
loads `mtp.safetensors` (`inkling2_dspark/split.py:16-22`). Its model retains
requested widths (`inkling2_dspark/dspark.py:193-194`), while the round loop
records accepted proposals (`speculative/dflash.py:642-661`). The Inkling README
says the default is one anchor and seven proposals and warns that verification
shape can change greedy outputs (`models/inkling2/README.md:99-109`). The separate
DeepSeek-V4 DSpark default is two tokens because larger blocks can cost more
than they save (`docs/speculative-decoding.md:149-163`).

## TensorFold projection

The replay-free V3 measurements in [inkling-verify.md](inkling-verify.md) give
45.599, 58.241, and 91.239 ms for widths 2, 4, and 8 at synthetic half
acceptance. Ordinary decode is approximately 25 ms per token, or 40 tok/s.
The measured copy hybrid achieves 42.869, 55.288, and 79.844 tok/s on code,
prose, and structured. These are different execution paths, so the projection
is a screening estimate, not a benchmark of native DSpark.

Assume an optimistic constant 0.814 ms per drafter round: 25 ms multiplied by
635,795,744 bytes of BF16 drafter tensors divided by the full 19,526,019,168-byte
target shard. The sidecar header has 54 tensors and 317,897,872 parameters,
including three layers with 2304 by 9216 MLP matrices and a 201024 by 256
Markov matrix. This byte ratio assumes the entire target shard is read per
decode token and the drafter has the same effective bandwidth. The target is
MoE and reads only active experts, so the assumption likely understates drafter
cost. It also omits the shared vocabulary head, recurrent proposal sampling,
context processing, and width-dependent work. No native drafter was timed.

`projected tok/s = 1000 × committed per round / (V3 round ms + 0.814 ms)`.
The result assumes MLX acceptance transfers to TensorFold and substitutes the
half-acceptance V3 round cost for each observed distribution. Both assumptions
need native validation.

| Prompt | Width 8 | Width 4 | Width 2 | Copy hybrid |
| :--- | ---: | ---: | ---: | ---: |
| code | 43.80 | 51.40 | 40.40 | 42.869 |
| prose | 40.26 | 45.27 | 39.53 | 55.288 |
| structured | 42.14 | 50.01 | 38.57 | 79.844 |

At zero drafter cost, the prose projections are 40.62, 45.91, and 40.23 tok/s
at widths 8, 4, and 2. Thus none clears prose's copy-hybrid rate even under
that optimistic bound. The code width-4 projection clears its copy baseline,
but the required code-and-prose criterion is not met. The nine exactness
failures independently prevent treating MLX's path as a lossless reference.

## Reproduction

From the checkout, set `PYTHONPATH` to the pinned MLX-VLM source,
`PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`, and `HF_HOME` to the campaign's
existing `.work/hf-home`. Set both GPU gate limits to 35. Before each process,
run `gpu-gate-ready -1` and `perf-gate-ready`. Then run
`python3 campaign/heavy.py -- <mlx-vlm-venv>/bin/python campaign/comparators/mlx-vlm/dspark_accept.py <code|prose|structured>`.
Run `perf-gate-ready -1` immediately after each process. The runner appends
summary rows to the JSONL file. Use an empty output file for a fresh sweep.
