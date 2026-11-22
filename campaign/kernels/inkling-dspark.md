# Inkling 2 Mini DSpark acceptance

**Corrected: GO for a scoped native DSpark, decided on the served chat comparator (see "Correction" below).**
The original verdict follows. It compared chat-prompt acceptance against runner copy rates measured on raw
text continuation.

**Original: NO-GO** for native DSpark on this evidence. The best projected prose rate is
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

## Correction: the like-for-like comparator

The acceptance above was measured on chat-templated prompts (1,349, 1,154 and 1,124 tokens). The copy-hybrid
rates it was compared with (42.869, 55.288 and 79.844 tok/s) came from the runner on raw prompt text without
the assistant turn (1,334, 1,139 and 1,109 IDs). There the model continues the prompt, which copy drafts predict
well. Served copy drafts on the same chat prompts measure 44.3, 43.1 and 39.5 tok/s, against 40.7, 40.8 and
40.9 ar (`inkling-verify.md`, "Served copy drafts after the replay-free commit").

Against that comparator, the width-4 projection (51.40, 45.27 and 50.01 tok/s with the 0.814 ms drafter
assumption) beats both ar and served copy drafts on all three prompts. With a more conservative 3 ms drafter
round (about 0.64 GB of BF16 drafter weights read once per round, plus three sequential Markov-head steps),
width 4 projects 3.036/61.2, 2.674/61.2 and 2.953/61.2 rounds per ms, i.e. 49.6, 43.7 and 48.2 tok/s. That is
clear on code and structured and marginal on prose. Both gains grow with any cut in verification round cost.

The decision rule is unchanged: beat both ar and copy drafts on at least code and prose. On the correct
comparator it gives GO for a scoped native implementation at width 4. Native acceptance and drafter cost must
be measured before any keep decision.

## Native drafter

The native drafter, its fixture test and `campaign/comparators/mlx-vlm/dspark_ref.py` are parked on
`lane/ds1-dspark-drafter` at `c33e27796`; they are not on main. The reproduction below runs from that branch.

The Metal drafter consumes captured tap rows and constructs the normalized anchor embedding from its token ID. It
retains per-layer context keys and values across blocks and resets the Markov recurrence for each block. The f32 computation reads BF16
weights from `mtp.safetensors` and the target embedding, embedding norm, and unembedding tensors. The full trained
block has eight rows even when the requested width is four.

MLX 0.32.4 enables TF32 by default (`mlx/utils.h:225-227`), including float32 matrix products on the NAX path
(`mlx/backend/metal/matmul.cpp:964-966`). The reference script sets `MLX_ENABLE_TF32=0` before importing MLX. On the
first captured query projection, the default MLX output was 0.33485273, while native f32 and an independent f64 dot
product gave 0.33504700 and 0.33504684. MLX without TF32 gave 0.33504700.

The f32 fixture comparison covered 65 blocks from code, prose, and structured chat prompts, including 35 width-four
and 25 width-eight blocks. Every greedy proposal matched. The maximum absolute logit error was 0.000114441, and the
maximum error divided by `max(abs(reference), 1)` was 0.000087679. The test bounds that scaled error by 0.001 for
every logit and requires exact proposal IDs. The floor of one gives near-zero logits an absolute tolerance, while
the relative component scales for larger logits. The observed maximum remains more than ten times below the bound.

The binary fixture starts with `DSPARK01` and a little-endian u32 group count. Each group begins with a u32 block
count. Each block contains u32 anchor, width, context-row count, and vocabulary size, followed by f32 packed context
rows `[rows, 4, 2304]`, f32 proposal logits `[width-1, vocab]`, and i32 proposal IDs `[width-1]`. Groups reset the
context cache. The first block holds the prompt context; later blocks hold the target rows accepted since the prior
round. The fixture is disposable and is not committed.

To reproduce, set `MODEL` to the checkpoint directory, `MLX_VLM` to the pinned source checkout, and `MLX_PYTHON`
to the pinned virtual environment's Python. Set `TMPDIR=$PWD/.work/tmp`, `HF_HOME=$PWD/.work/tmp/hf`,
`XDG_CACHE_HOME=$PWD/.work/tmp/cache`, `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`, `INKLING_LOAD_LIMIT_GB=26`,
and `PYTHONPATH=$MLX_VLM`.
Then run:

```sh
set -o pipefail
python3 campaign/heavy.py -- "$MLX_PYTHON" campaign/comparators/mlx-vlm/dspark_ref.py "$MODEL" .work/tmp/dspark.bin
TF_INKLING2_LOOP_DIR="$MODEL" TF_INKLING2_DSPARK_FIXTURE="$PWD/.work/tmp/dspark.bin" \
  python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 2>&1 | tee .work/tmp/dspark-test.log
```

Use Zig 0.17.1 with `PATH=/opt/homebrew/bin:$PATH` and the campaign's shared `ZIG_GLOBAL_CACHE_DIR`. The test skips
the fixture comparison if `TF_INKLING2_DSPARK_FIXTURE` is absent.

For timing, compile and run the checkpoint test first. Set `DSPARK_TEST` to the last `.zig-cache/o/.../test`
executable named in `.work/tmp/dspark-test.log`. Run the following commands after both admission checks pass:

```sh
GPU_GATE_MAX_GPU_PCT=35 GPU_GATE_MAX_UTIL_PCT=35 gpu-gate-ready -1
perf-gate-ready
TF_INKLING2_LOOP_DIR="$MODEL" TF_INKLING2_DSPARK_FIXTURE="$PWD/.work/tmp/dspark.bin" \
  TF_INKLING2_DSPARK_TIMING=1 python3 campaign/heavy.py -- "$DSPARK_TEST"
perf-gate-ready -1
```

The timing path warms each width for three rounds, then reports the median and range of 21 rounds with a fixed
context and one new target row. It includes dispatch, synchronization, and per-round buffer allocation. The same
model process measures both widths.

On Apple M5 Max with macOS 27.2, the admitted run used 1,349 cached context rows, one new context row, the same
checkpoint and MLX-VLM revision as above, and a `.fast` Zig build. The GPU and host admission checks passed before
the run, and the host check passed afterward. The heavy guard reported a peak charged 20.4 GB against the 28 GB cap.

| Width | Median of 21 | Range | Compared with 0.814 ms | Compared with 3 ms |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 16.675 ms | 16.476–16.841 ms | 20.49× | 5.56× |
| 8 | 29.747 ms | 29.541–30.049 ms | 36.54× | 9.92× |

These measured round costs exceed both projection assumptions. They do not establish a served decoding rate or a
keep decision for native speculation. An earlier admitted run using the identical worktree copy of `heavy.py` gave
16.769 ms and 29.820 ms medians for widths four and eight respectively.

## Verdict: NO-GO for integration

The measured drafter takes 16.675 ms at width 4. Each round costs that plus the 58.241 ms V3 verification round,
74.916 ms in all. At the MLX acceptance above (3.036, 2.674 and 2.953 committed per round), that projects:

| Prompt | Width-4 DSpark | ar | Served copy drafts |
| :--- | ---: | ---: | ---: |
| code | 40.5 | 40.7 | 44.3 |
| prose | 35.7 | 40.8 | 43.1 |
| structured | 39.4 | 40.9 | 39.5 |

It beats neither comparator on any prompt, so the decision rule gives NO-GO. Host integration (DS2) is not started.

At unchanged verification cost, the drafter round must fit in the following budgets:

- Code: at most 10.3 ms, since 3.036 / 44.3 is 68.5 ms per round.
- Prose: at most 3.8 ms, since 2.674 / 43.1 is 62.0 ms per round.

The prose budget is not reachable with the trained width-8 block. Its three sequential proposal steps each read the
201,024 × 2,304 shared unembedding, about 2.8 GB in BF16 in all, before the drafter's own 0.64 GB.

Revisit integration only if verification at four rows falls far enough that 2.674 committed tokens beat 43.1 tok/s
with the measured drafter. That needs a round of at most 45.4 ms, which is a cut of 12.9 ms or more from 58.2 ms.
