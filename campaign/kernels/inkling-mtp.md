# Inkling 2 Mini MTP speculation

Status: greedy MTP speculation serves and is exact by construction; the corrected-workload served sweep, the
drafter's ds1-method round cost and the few-row proposal head (`wip` commit `2cd5f9561`) remain unmeasured.

| Item | State |
| --- | --- |
| MTP head forward on Metal, tap capture in the forward | committed `3a334fab3`, parity checked |
| Exact grouped verification through the batch step | committed `5884b2b1d`, bit-exact tests on both checkpoints |
| Served speculation, planner, fallback, acceptance stats | committed `0e33a4718`, `dc7a4dba3` |
| Greedy equality, widths 1 and 4, real prompts | passes on both checkpoints (family tests and suite) |
| Suite `inkling-host,inkling-gpu` | exit 0, no skips (95/95 family GPU tests), on the code of `dc7a4dba3` |
| `zig build native -Dcpu=apple_m1 -j1` | exit 0 on the code of `2cd5f9561` |
| Served sweep, corrected client (`69651393c`) | not run |
| Drafter round cost by the ds1 method | not run (timing mode in `wip` `2cd5f9561`) |
| oMLX with MTP on | handed to the baseline lane (`campaign/fx/omlx-mtp-sweep.sh`) |

## What serves

Greedy-only speculation with the checkpoint's own MTP head (`mtp.safetensors`), labelled as greedy. With
`INKLING_COPY_DRAFTS=1` (the served sweep's setting) and a checkpoint that has `mtp_config`, the Inkling host loads the
head and decodes through speculative rounds; without it, or with `INKLING_COPY_DRAFTS=0` or `--no-drafts`, every step
is the existing no-drafter batch step. The adapter's variable name predates the MTP head; it now enables MTP drafts.

Each round, for every decoding slot:

1. A planner chooses how many drafts the slot verifies (zero is the fallback), from the slot's recent acceptance and the
   measured round costs, within eight verification rows for the whole batch.
2. The drafter proposes them: an 8-row block (the slot's newest token, then the trained mask embedding) attends the
   slot's drafter context and itself, and the recurrent Markov head picks one token after another on the GPU.
3. One grouped batch step verifies every slot's newest token and its drafts.
4. Each slot keeps the drafts that equal the target's greedy picks, then the target's next pick; the accepted rows are
   committed into its cache and its drafter context.

## Head architecture

The reference is mlx-vlm's `mlx_vlm.speculative.drafters.inkling2_dspark` (adapted from vLLM's Inkling 2 DSpark,
`WoosukKwon/vllm@d664cd1e`), cross-checked with oMLX 0.8.3's `omlx/patches/inkling2/dspark.py` and its
`omlx/speculative/dflash/drafter.py`:

- Context: the target's partial block sum entering visits 5, 23 and 41 (execution order, every core pass counting) and
  its final normalized state. Each draft layer mixes the four taps with `softmax(fuse_weight[layer])` (stored at bf16),
  RMS-normalizes the mix with its `dflare_norm`, and projects it with `k_proj_ctx`/`v_proj_ctx`; keys are
  RMS-normalized per head with `k_norm_ctx` and rotated (non-traditional RoPE, theta 1e6) at their stream position.
- Block: the anchor's normalized target embedding, then `mask_embed` for the 7 remaining rows, at positions after the
  context. Three pre-norm layers: unmasked attention (24 query heads, 8 key/value heads, dim 128) over the context and
  the block's own keys, then a SiLU-gated MLP (9216). Local layers 0 and 2 see the newest 1,024 context positions; layer
  1 sees all of them (mlx-vlm's semantics; oMLX bounds every layer at 1,023).
- Proposals: the final norm of block rows 1..7 through the target's unembedding, plus the Markov bias
  `fold @ latent + fold_bias`, where `fold = unembed @ hidden_out` is built once (bf16). The RNN state starts at zero;
  each step mixes `[state, markov_w1[previous token]]` and the step's `[draft row, mixed final-context tap]` through
  `rnn_proj` into gate, candidate and output; logits are soft-capped at 30 and the argmax is the next previous token.

## Exactness

Every emitted token equals plain greedy decoding by construction, not by tolerance. The served plain path is the batch
step (`batchStep`), whose few-row kernels compute each row independently of how many rows share the step (the existing
batch/decode tests at widths 1, 2 and 4 report zero differences). Verification is that batch step over grouped rows,
each stream's newest token followed by its drafts, at most eight rows in all. Only three batch kernels let rows
interact, and each has a grouped twin in `kernels/verify.metal` with the same arithmetic in the same order:

- `i2v_conv` (`i2p_batch_conv`): a row's earlier positions in its own group come from the step's rows, not the ring;
  nothing is written to the ring, and each row's input is captured.
- `i2v_store_rel` (`i2p_batch_store_rel`): the normalized keys and values go to a per-slot capture, not the cache.
- `i2v_split` (`i2_batch_split`, as of base `4c2af5d2a`): positions from the group's start are read from the capture.
  The batch combine (`i2_batch_combine`) is reused unchanged.

So the forward never writes a cache, which also removes the local ring's hazard (a later row's key overwriting a key an
earlier row of the same stream still attends). `i2v_commit_kv` and `i2v_commit_conv` then write the accepted rows'
keys, values and newest four convolution inputs, and `i2v_argmax` picks each row's token exactly as the host's scan of
the logits picks it (first maximum; a NaN never wins; a NaN at index 0 keeps index 0).

Tests, run with both checkpoints (`TF_INKLING2_LOOP_DIR`, `TF_INKLING2_LOOP_OQ4E_DIR`):

- `grouped verification equals batch decoding bit for bit and commits the same caches`: three streams of 4, 3 and 1
  rows (8 in all), the first at positions 1,022 to 1,025 so it crosses the 1,024-row local ring. Every row's logits
  equal the sequential batch steps' bytes; after committing 2, 1 and 0 accepted drafts every cache buffer equals the
  cache of sequential decoding of the accepted tokens; the next step's logits and a single-row group's logits and
  caches also match.
- `speculative decoding emits plain greedy decode's tokens on real prompts at widths one and four`: the agent
  workload's first prompt (2,572 tokens) for 129 tokens at width 1, and four rendered chat prompts for 65 tokens each
  at width 4, through `Speculator.round` against plain batch steps.
- `the drafter's proposals follow its own logits, repeat bit for bit and batch by stream`.

## Draft parity

`campaign/kernels/inkling_mtp_parity.py` drafts with mlx-vlm's head from TensorFold's own tapped states and anchor,
its Markov chain teacher-forced on TensorFold's tokens, and compares every step's logits over the unpadded vocabulary.
In float32 the reference keeps TensorFold's two bf16 rounding points (mixture weights, fold); in bfloat16 it runs as
shipped. On the agent workload's first prompt (2,572 context positions, anchor 200001), oQ4e checkpoint:

| Step | TensorFold draft | Reference draft (f32) | Max abs diff, f32 | Mean abs diff, f32 | Max abs diff, bf16 | Mean abs diff, bf16 | Reference top-2 margin | Top-5 overlap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 200008 | 200008 | 0.0336 | 0.0059 | 0.2228 | 0.0505 | 14.30 | 5 |
| 2 | 371 | 371 | 0.0284 | 0.0042 | 0.1772 | 0.0384 | 4.73 | 5 |
| 3 | 150409 | 150409 | 0.0270 | 0.0037 | 0.1704 | 0.0387 | 9.11 | 5 |
| 4 | 171960 | 171960 | 0.0267 | 0.0040 | 0.1909 | 0.0406 | 2.68 | 5 |
| 5 | 25 | 25 | 0.0345 | 0.0047 | 0.1785 | 0.0380 | 1.67 | 5 |
| 6 | 151297 | 151297 | 0.0286 | 0.0045 | 0.1732 | 0.0382 | 7.67 | 5 |
| 7 | 62 | 62 | 0.0310 | 0.0041 | 0.1755 | 0.0389 | 1.84 | 5 |

Logit magnitudes reach 25; the soft cap is 30. The reference's free-running drafts equal TensorFold's seven drafts in
both precisions. The float32 residual is TensorFold's f16 context keys and values and f32 activations against the
reference's f32 ones; the bf16 column is the reference's own bf16 rounding.

Across all four fixture prompts (each prompt's first block after prefill; drafts equal is the reference's free-running
draft prefix that equals TensorFold's; differences are the worst step's):

| Prompt | Context positions | Drafts equal, f32 | Drafts equal, bf16 | Max abs, f32 | Mean abs, f32 | Max abs, bf16 | Mean abs, bf16 | Least top-5 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| agent | 2572 | 7/7 | 7/7 | 0.0345 | 0.0059 | 0.223 | 0.050 | 5 |
| code | 37 | 7/7 | 7/7 | 0.0330 | 0.0049 | 0.194 | 0.041 | 5 |
| prose | 32 | 7/7 | 7/7 | 0.0323 | 0.0053 | 0.209 | 0.045 | 4 |
| list | 30 | 7/7 | 7/7 | 0.0300 | 0.0048 | 0.198 | 0.042 | 5 |

The float32 reference ran with MLX's default TF32 enabled. The parked native DSpark drafter (`lane/ds1-dspark-drafter`,
`inkling-dspark.md` on main) found that TF32 changes MLX's float32 products and matched to 1.1e-4 with
`MLX_ENABLE_TF32=0`; this comparison was not repeated without TF32, so part of the 0.03 float32 residual may be the
reference's own TF32 rounding.

## Served measurements (pre-correction client)

These used the agent client before main's `69651393c`, which did not replay `reasoning_content` and sent an ignored
`enable_thinking: false`. Per the campaign owner they are not comparable with later sweeps and are kept only as
evidence of behaviour. Binary built at `bfd0c56b7` (code of `dc7a4dba3`), `sweep-subagents.sh N 4096`, MTP on, oQ4e,
busy host (an OrbStack VM held about 4.7 cores). The N=4 attempt was deferred by its post-cell CPU gate (4.27 busy
cores) and the sweep was stopped when the workload changed; N=4 and N=8 have no MTP-on result. No same-binary no-drafter
control was run.

| Agents | Rep | Aggregate tok/s | Tokens | Rounds | Drafts accepted | Tokens a round |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 26.14 | 369 | 197 | 168 / 263 (63.9%) | 1.87 |
| 1 | 2 | 45.55 | 369 | 364 | 1 / 1 | 1.01 |
| 2 | 1 | 38.37 | 744 | 692 | 46 / 63 (73.0%) | 1.08 |
| 2 | 2 | 63.37 | 744 | 721 | 17 / 21 (81.0%) | 1.03 |

Per request at one agent, first repetition (decode tok/s as the server logs it; the earlier no-drafter run of the same
workload logged about 46 tok/s a turn): 50.0 (62/95 accepted), 54.9 (70/94), 40.9 (35/72) and 46.1 (1/2). In the
second repetition the planner stayed in fallback, at 48.7 to 49.1 tok/s a turn: with the measured costs below, drafting
did not pay. Two slots' drafter contexts restarted mid-conversation because the old client's prompts did not contain
the reasoning the slot had fed; the corrected client's prompts do.

## Costs

The speculator's measured costs (exponential averages of whole rounds, including dispatch and synchronization; one
agent, about 2.6K to 2.8K context positions):

| Verification rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms | 20.4–23.4 | 26.9–29.4 | 35.3–40.9 | 46.3–51.5 | 55.6–59.3 | 63.1–69.4 | 75.6–81.6 |

A proposal (one block of 8 rows and up to 7 Markov steps, after one context row) measured 10.4 to 13.6 ms in serving.
For comparison, the parked ds1 drafter's round (fixed 1,349-position context, one new row, median of 21 rounds,
including per-round allocation) was 16.675 ms at width 4 and 29.747 ms at width 8. The two are not the same measurement:
the served figure is an average over rounds of varying depth at a longer context, without allocation. It shows the
fold of the unembedding through the Markov head paying (ds1 read the 0.46 GB unembedding for every Markov step), and
it is still too large: drafting plus verifying `k` rows must cost much less than `k` target steps (about 20 ms each).

The remaining proposal cost is dominated, by estimate, by the base logits through the 8-bit unembedding in the 32-row
tiled kernel, whose per-element dequantized loads stride across vocabulary rows. The `wip` commit replaces it for up to
16 rows with `i2m_head`, which reads each vocabulary row once and coalesced; it is untimed.

## Limits

- Verification costs what the batch step costs per row, because exactness is the batch step's own arithmetic. Its
  few-row kernels read each projection's weights once per row: the 8-bit unembedding alone is 0.49 GB a row (about
  1.1 ms at 450 GB/s), and each row's routed experts are read for that row. A few-row variant that shares weight reads
  across rows while keeping each row's accumulation order would cut verification and plain multi-stream decode alike;
  those kernels belong to the decode-step work, not this change.
- At most eight verification rows a round: the batch step's few-row regime, where per-row arithmetic does not depend
  on the row count. Eight decoding streams therefore never draft, and four share at most four drafts.
- The new Metal operators (`kernels/mtp.metal`, `kernels/verify.metal`) are tested in the family's GPU tests, like the
  family's existing kernels, and are not registered in the kernel-contract runner or its launch registry.
- The native adapter's memory plan does not count the head: 0.64 GB of weights, the 0.10 GB fold, about 12 KB a
  position for each slot's drafter context (50 MB a slot at 4,096 positions), and the drafter's scratch.
- `INKLING_COPY_DRAFTS` (read by the adapter) now selects MTP drafts; the adapter's refusal text for a separately named
  drafter still says the head is not wired.
- A slot whose prefix came from the prompt store or a fork takes its drafter context from a slot that fed those
  positions; when none did, its context starts at the reused position, and acceptance is lower until it grows.

## What remains, and the next step

1. Verify and time the `wip` commit in one GPU-lock acquisition: the drafter, grouped and speculative tests on both
   checkpoints, and the drafter's round cost by the ds1 method (width 4 and 8, both head kernels):

   ```sh
   TF_INKLING2_LOOP_DIR=$HOME/models/Inkling-2-Mini@ed40e7da625f \
   TF_INKLING2_LOOP_OQ4E_DIR=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd TF_INKLING2_MTP_TIMING=1 \
   INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1 --summary all
   ```

   gated by `gpu-gate-ready -1` and `perf-gate-ready` before and `perf-gate-ready -1` after; it prints
   `drafter round width=W head=few-row|tiled context=1349: median ... over 21`.
2. If the few-row head brings a width-2 to width-4 round well under the 6 to 9 ms a verification row costs, run the
   corrected-client sweep (with `69651393c`, now on this branch) at 1, 2, 4 and 8 agents in three configurations: MTP
   on (`INKLING_COPY_DRAFTS=1`), the no-drafter control (`INKLING_COPY_DRAFTS=0`), and the drafter loaded but paused
   (`INKLING_MTP_PAUSE=1`). The script used for this was `.work/tmp/m1/job3.sh`: one acquisition, each cell retried on a
   gate deferral. Report `python3 campaign/fx/mtp_acceptance.py <session dir>` per cell.
3. The larger lever is verification: each extra row costs 6 to 10 ms because the batch step's few-row kernels read the
   weights once per row. A few-row path that shares weight reads while keeping each row's accumulation order would cut
   both verification and multi-stream decode; it belongs to the decode-step lanes.

## Reproduction

From the worktree root, with `PATH=/opt/homebrew/bin:$PATH`, `TMPDIR=$PWD/.work/tmp` and
`ZIG_GLOBAL_CACHE_DIR=$HOME/repos/tf/.work/zig-global-cache`:

```sh
# family tests, including the exactness, equality and drafter tests, on both checkpoints
TF_INKLING2_LOOP_DIR=$HOME/models/Inkling-2-Mini@ed40e7da625f \
TF_INKLING2_LOOP_OQ4E_DIR=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd INKLING_LOAD_LIMIT_GB=26 \
python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1 --summary all

# draft parity: the drafter test dumps taps, anchor, drafts and logits for each fixture prompt, then the reference
TF_INKLING2_LOOP_OQ4E_DIR=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd TF_INKLING2_MTP_DUMP=$PWD/.work/tmp/dump \
python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1 -j1
python3 campaign/heavy.py -- $HOME/repos/tf/.work/venv-mlx-vlm/bin/python -I campaign/kernels/inkling_mtp_parity.py \
  $HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd .work/tmp/dump --dtype float32   # or bfloat16; also dump-code etc.

# served sweep with MTP on (INKLING_COPY_DRAFTS=0 for the no-drafter control), then acceptance per cell
INKLING_MODEL=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd sh campaign/fx/sweep-subagents.sh N 4096
python3 campaign/fx/mtp_acceptance.py .work/tmp/sessions-Inkling-2-Mini-oQ4e-mtp@3eea042751cd-N-4096

# oMLX 0.8.3 with Lightning MTP on, same weights, guard, slots and client as the MTP-off run
sh campaign/fx/omlx-mtp-sweep.sh
```

The fixture prompts (`zig/src/families/inkling2_loop/fixtures/mtp_prompts.json`) were rendered with the checkpoint's
`chat_template.jinja` and `tokenizer.json` (jinja2 and tokenizers from `.work/venv-mlx-vlm`); the agent prompt's 2,572
tokens match the served sweep's logged prompt length.
