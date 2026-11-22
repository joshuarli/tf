# Inkling 2 Mini: TensorFold and nanoattention

## Shootout (2026-11-22)

The shootout is in progress. The following state-size calculation is independent
of the remaining timing and fidelity trials.

The completed cells are code at 1,349 prompt tokens, prose at 1,154 and
structured at 1,124. Values are medians and ranges across three pairs; peak
memory is the largest charged value. Both engines' outputs were stable across
the three runs. The
first different generated indices were 204 for code, 323 for prose and 29
for structured in every pair. Three pairs and differing engine configurations
do not establish a general performance verdict.

| Engine | TTFT seconds | Prefill tok/s | Decode tok/s | Peak charged GB |
| --- | ---: | ---: | ---: | ---: |
| TensorFold, code | 4.039 (3.356–4.127) | 333.99 (326.90–402.01) | 38.77 (38.47–39.08) | 23.0 |
| nanoattention, code | 2.671 (2.298–3.276) | 505.11 (411.81–587.10) | 40.00 (38.83–40.16) | 24.5 |
| TensorFold, prose | 2.778 (2.771–2.782) | 415.37 (414.77–416.46) | 39.08 (39.00–39.11) | 23.1 |
| nanoattention, prose | 2.028 (1.991–2.048) | 569.02 (563.61–579.63) | 40.38 (40.25–40.84) | 24.6 |
| TensorFold, structured | 2.635 (2.627–2.658) | 426.64 (422.82–427.90) | 39.11 (39.05–39.16) | 23.0 |
| nanoattention, structured | 1.823 (1.819–2.106) | 616.42 (533.82–617.84) | 40.39 (39.83–40.63) | 23.9 |

The [gated records](../comparators/nanoattention/runs/shootout-20261122-gated.jsonl)
retain successful and rejected attempts. The remaining cells and oracle
comparison are not yet complete.

The initial second prose TensorFold trial completed but the driver redacted
its entire receipt and exited 1, so that trial was discarded. The driver
now redacts individual diagnostic lines, and its replacement trial passed.

The TensorFold raw-token runner was built from commit `86ca89451d882fd99bbfad7c7f9a86461e7c59e3`
with Zig 0.17.1 and `-Dcpu=apple_m1`. Its SHA-256 is
`540277e48f31197e1daa2be1d2d0b431fbab44c628556c27008576a1534a0f42`.
Nanoattention is revision `6f782a8dace846e4e64a5a12cbdf5c0252166199`; its
binary SHA-256 is `8340fc9c1fa944ad2d50ebcbd593597ae80c15a056c44801f5d652c9022d5152`.
Both use the original `ed40e7da625f` checkpoint, greedy decoding and 512
generated tokens. TensorFold allocates 32,768 tokens. Nanoattention allocates
the prompt length plus 512 tokens. Each process performs one full warmup and
one timed run. TTFT here starts at engine prefill and ends after its first
argmax. It excludes loading and is not served TTFT. Decode is 511 committed
tokens divided by decode wall time.

The 1K and 4K prompts use the corresponding `campaign/prompts/dev` text,
rendered with the checkpoint chat template and tokenizer. Inkling's actual
token counts differ from the nominal sizes. Each 16K input takes the first
16,380 and last four token IDs of the corresponding rendered 32K dev prompt,
preserving its template suffix. Both engines receive the same resulting IDs.
The trial sequence alternates engines and runs 16K cells last. Each command
runs through `heavy.py` with `INKLING_LOAD_LIMIT_GB=26`, a 60/60 GPU admission
gate, and the performance gate before and after. Results are busy-host. All
initial attempts lacked in-trial GPU contention accounting and were discarded.
The 32K-capacity nanoattention attempts also reached memory-pressure level 2,
so the nanoattention allocation was reduced. The
[discarded attempts](../comparators/nanoattention/runs/shootout-20261122.jsonl)
remain available for inspection.

The run uses the `shootout.py` adapter in `campaign/comparators/nanoattention`.
Its records contain the complete engine commands, gates, accepted timings,
predictions and charged-memory receipts. The setup commands from the worktree
root were:

```sh
mkdir -p .work/tmp/native-build .work/tmp/zig-cache
PATH=/opt/homebrew/bin:$PATH ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tensorfold-m5/.work/zig-global-cache TMPDIR="$PWD/.work/tmp" zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix "$PWD/.work/tmp/native-build" --cache-dir "$PWD/.work/tmp/zig-cache" --verbose > .work/tmp/native-build/build.log 2>&1
PATH=/opt/homebrew/bin:$PATH ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tensorfold-m5/.work/zig-global-cache TMPDIR="$PWD/.work/tmp" python3 -c 'import importlib.util, pathlib, subprocess; p=pathlib.Path("campaign/comparators/mlx-vlm/build_native.py").resolve(); s=importlib.util.spec_from_file_location("builder",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); cmd=m.native_command(pathlib.Path(".work/tmp/native-build/build.log").read_text(), pathlib.Path(".work/tmp/native-build/native-compare").resolve()); print("building raw-token comparator", flush=True); subprocess.run(cmd,check=True)'
PYTHONDONTWRITEBYTECODE=1 /Users/josh/repos/tensorfold-m5/.work/venv-mlx-lm/bin/python campaign/comparators/nanoattention/shootout.py prepare
PATH=/opt/homebrew/bin:$PATH PYTHONDONTWRITEBYTECODE=1 /Users/josh/repos/tensorfold-m5/.work/venv-mlx-lm/bin/python -u campaign/comparators/nanoattention/shootout.py timings > .work/tmp/n5-shootout/gated-timings.log 2>&1
```

The two build commands and prompt preparation exited 0. The oracle verifier
also exited 0 and reported 16 saved histories, including two longer than the
sliding window:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/josh/repos/tensorfold-m5/.work/venv-ref/bin/python campaign/parity/inkling2-mini/oracle.py --histories campaign/parity/inkling2-mini/histories.json --dir /Users/josh/repos/tensorfold-m5/.work/parity/inkling2-mini/ref
```

The comparator tests exited 0 with 17 tests and four dependency-related
skips. The skipped template and numerical regressions passed separately in
the environment with Jinja, tokenizers and NumPy:

```sh
PATH=/opt/homebrew/bin:$PATH PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT="$PWD/.work/tmp/n5-uv-venv" make -C campaign nanoattention-tests
PYTHONDONTWRITEBYTECODE=1 /Users/josh/repos/tensorfold-m5/.work/venv-mlx-lm/bin/python -m unittest discover -s campaign/comparators/nanoattention -p test_prompt_ids.py
PYTHONDONTWRITEBYTECODE=1 /Users/josh/repos/tensorfold-m5/.work/venv-mlx-lm/bin/python -m unittest discover -s campaign/comparators/nanoattention -p test_shootout_fidelity.py
```

Both engines allocate **65,536 bytes per additional context token** for global
key and value state. The checkpoint has eight global attention visits after
expanding the loop. Each visit stores keys and values for eight 128-element
heads as float32: `8 * 2 * 8 * 128 * 4 = 65,536`. TensorFold's
`zig/src/families/inkling2_loop/state.zig:38-71` allocates full-capacity rows
for global slots and fixed-window rows for local slots. Nanoattention's pinned
`src/mega/inkling2_loop.zig:59-74,372-373,421-423`, `src/mega/ir.zig:39-45`,
`src/mega/exec.zig:137-147`, and `src/metal.zig:154-156` allocate the same
float32 global sequence state. Convolution and local-window state are fixed
once the capacity exceeds the local window. Scratch and allocation overhead
are outside this per-token figure.

At 128K and 256K tokens, global KV alone would occupy 8 GiB and 16 GiB,
respectively. Relative to TensorFold's 32K-capacity allocation, those add
6 GiB and 14 GiB. These are allocation projections, not measured long-context
peak memory; they assume other allocations stay fixed and do not lift the
campaign's 32,768-token cap.

Nanoattention rounds normalized activations and expert inputs to BF16 in both
its reference and optimized configurations. Its pinned
`src/mega/inkling2_loop.zig:343` and `:482` establish these operations.
TensorFold retains f32 activations, so neither comparison is f32 against f32.

The table reports medians and ranges of three interleaved pairs per cell. Peak
charged memory is the largest observed by `heavy.py` in each cell. Token indices
are zero-based. Each engine produced the same sequence in all three runs.

| Prompt | Configuration | Engine | Context capacity | Prefill tok/s | Decode tok/s | Peak charged GB | First differing index |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 512 | reference | TensorFold | 577 | 360.30 (357.80–402.40) | 41.90 (41.77–42.02) | 21.1 | 15 |
| 512 | reference | nano | 576 | 34.49 (32.81–35.45) | 5.47 (5.47–5.48) | 22.2 | 15 |
| 512 | optimized | TensorFold | 577 | 392.20 (380.60–392.30) | 41.75 (41.60–41.84) | 20.9 | 9 |
| 512 | optimized | nano | 576 | 650.30 (649.27–650.54) | 44.21 (44.04–44.28) | 24.6 | 9 |
| 1024 | reference | TensorFold | 1089 | 407.00 (402.10–414.40) | 41.07 (40.89–41.09) | 21.2 | 45 |
| 1024 | reference | nano | 1088 | 34.27 (33.74–35.00) | 4.16 (4.16–4.16) | 22.2 | 45 |
| 1024 | optimized | TensorFold | 1089 | 414.40 (252.60–417.20) | 41.14 (41.11–41.15) | 21.0 | none in 65 |
| 1024 | optimized | nano | 1088 | 629.24 (628.81–629.46) | 42.23 (42.20–42.29) | 24.3 | none in 65 |

The reference path was slower in these runs. The optimized path had higher
prefill and decode rates in these runs, but its arithmetic and output can differ
from TensorFold. The 512-token outputs differ for both nanoattention settings.
The optimized 1,024-token outputs match for these 65 tokens. These results do not
establish parity on other prompts. Timer boundaries and warmup differ, and no
limiter was profiled, so a general engine speed verdict is inconclusive.

The workload uses the same checkpoint revision `ed40e7da625f` and the public
512-token and 1,024-token code prompts. The host is an Apple M5 Max with 48 GiB
of unified memory, running macOS 27.2. The comparison is labeled busy-host.

TensorFold is built from revision `a1862ad552db19779c280729f1d3c7f127001115`
with Zig 0.17.1 and `-Dcpu=apple_m1`. Its runner binary SHA-256 is
`3526b596feccfa3b5493994bcc68665bd26941bee2bdde5f52ca080449b789fa`.
Nanoattention is built from the source pin in the campaign lock, revision
`6f782a8dace846e4e64a5a12cbdf5c0252166199`, with Zig 0.17.1, Lean 4.35.0
and `-Doptimize=ReleaseFast`. Its binary SHA-256 is
`8340fc9c1fa944ad2d50ebcbd593597ae80c15a056c44801f5d652c9022d5152`.

TensorFold's `bench` mode uses a chunk of 512, zero warmup steps, and 64
sequential decode steps. Its prefill timer surrounds the chunked forwards and
ends before copying logits and selecting the first token. Its per-step wall
clock excludes the host argmax, and the reported decode rate is the reciprocal
of its median step latency. A separate `greedy-decode` process captures 65
output IDs, including the prefill prediction.

Nanoattention's `bench` mode uses the same prompt IDs, 64 decode forwards,
one untimed warmup, and one timed repeat per process. Its prefill timer ends
after the first argmax, and its decode timer includes each argmax. These timer
boundaries differ from TensorFold's, so the rates are not identical latency
metrics. Model loading is outside both engines' timers. Nanoattention uses its
reference defaults with batch 64, or the documented optimized knobs with batch
128. The optimized knobs include BF16 tiles and can change arithmetic.

The TensorFold cache capacity is prompt length plus 65. Nanoattention's context
is prompt length plus 64, covering the prefill and all 64 decode forwards. The
final predicted token is not forwarded. Global KV allocation capacity does not
change the number of tokens attended to at these prompt lengths. The campaign's
32,768-token serving cap is not a requirement for this runner comparison.

Each measured command runs under `heavy.py` and its 28 GB cap. Admission checks
use GPU and utilization limits of 35 percent, with the memory limits unchanged.
The procedure checks memory pressure before each attempt, alternates engine
order across pairs, and discards failed admission checks. The build and run
scripts are in `campaign/comparators/nanoattention/`.

[Run records](../comparators/nanoattention/runs/pairs.jsonl) contain all 24
accepted runs and 154 rejected attempts. Each record includes its command, exit
status, gate output, and memory receipt. Gate diagnostics that included unrelated
process command lines were redacted. The [summary script](../comparators/nanoattention/summarize.py)
checks sample counts and output stability before producing the table.

## Last-row logits

The existing TensorFold runner wrote all prefill logits with chunk 512. The
comparison reads only the last row. Nanoattention `eval` wrote that row with
feed 64 for reference defaults and feed 128 for the optimized knobs. Every row
contains 200,058 finite float32 values. The maximum absolute differences from
TensorFold were:

| Prompt IDs | Nanoattention reference | Nanoattention optimized |
| ---: | ---: | ---: |
| 512 | 0.7571 | 0.5919 |
| 1,024 | 1.8921 | 1.0913 |

These compare engine outputs; they do not establish an independent float32
oracle for these two prompts. [Logits run records](../comparators/nanoattention/runs/logits.jsonl)
contain the commands and gate results. The [difference values](../comparators/nanoattention/runs/logits-summary.json)
are retained separately. The raw float files are temporary and can be rebuilt
with `python3 campaign/comparators/nanoattention/logits.py`.

## Ablation

The table reports a busy-host leave-one-out experiment on the same 1,024-token
prompt, with 64 decode tokens and nano batch 128. Removing `batch` selects its
reference default of 64. Each row has three candidate runs, each bracketed by
full-optimized controls. Rates are medians in tok/s; each ratio is the median of
three candidate rates divided by the arithmetic mean of that candidate's two
neighboring controls. Control columns are medians of the six controls. A ratio
below one indicates a loss on removal. These marginal effects are not additive.
The control prefill rates ranged from 382.78 to 630.27 tok/s across the experiment,
so small differences can be host noise. No limiter was profiled.

| Removed | Control prefill | Candidate prefill | Prefill ratio | Control decode | Candidate decode | Decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| batch | 629.09 | 454.68 | 0.725 | 42.14 | 41.90 | 0.999 |
| expert_simd | 605.48 | 597.51 | 1.046 | 42.09 | 9.80 | 0.239 |
| expert_wide_dot | 629.37 | 628.29 | 0.998 | 42.23 | 38.05 | 0.901 |
| expert_gate_pair | 629.40 | 629.45 | 1.000 | 42.10 | 42.21 | 1.005 |
| expert_prefill | 629.34 | 173.65 | 0.276 | 42.10 | 41.60 | 0.987 |
| expert_wide_k | 629.24 | 629.03 | 1.001 | 42.04 | 42.15 | 1.001 |
| expert_bf16_tiles | 629.39 | 629.07 | 1.001 | 42.17 | 42.21 | 1.001 |
| expert_mlx_scale | 629.45 | 359.39 | 0.571 | 42.27 | 42.13 | 0.998 |
| bf16_linear_tile | 629.59 | 469.79 | 0.746 | 42.24 | 29.14 | 0.690 |
| bf16_prefill_tile | 629.58 | 554.54 | 0.881 | 42.27 | 42.12 | 0.997 |
| bf16_prefill_matrix | 625.60 | 422.66 | 0.676 | 41.93 | 40.67 | 0.990 |
| bf16_decode_vec | 603.82 | 604.48 | 1.001 | 40.69 | 30.12 | 0.737 |
| fuse_bf16_cast | 599.48 | 569.53 | 0.930 | 40.93 | 37.02 | 0.907 |
| concurrent_scratch | 628.83 | 613.34 | 0.974 | 41.88 | 39.53 | 0.946 |
| relative_split | 629.33 | 629.38 | 1.001 | 42.22 | 6.99 | 0.166 |
| relative_split_4 | 594.21 | 593.67 | 1.003 | 42.16 | 38.51 | 0.914 |
| relative_tile_prefill | 629.59 | 418.02 | 0.664 | 42.24 | 42.22 | 1.003 |

The `expert_prefill` row removes `expert_wide_k`, `expert_bf16_tiles` and
`expert_mlx_scale` as dependent options. The `relative_split` row also removes
`relative_split_4`. The tile-8 dense vector variant needs both
`bf16_prefill_tile` and `bf16_prefill_matrix`, but the tile also selects the
independent grouped projection in `src/mega/exec.zig:728` and `:803`.
Consequently, these rows do not isolate a single kernel. The zero-effect rows
can be masked by higher-priority dispatch paths. In particular, the block path
precedes both expert BF16-storage and wide-K fallbacks at `exec.zig:589-613`.

[Records](../comparators/nanoattention/runs/ablation.jsonl) contain 153 accepted
runs, 71 refused admissions and one failed initial attempt from selecting Python
3.9. The rerun used Homebrew Python. The largest recorded heavy charge was
24.3 GB under the 28 GB cap. All accepted prediction checksums were 5885143;
checksums alone do not prove logit parity. Run
`PYTHONDONTWRITEBYTECODE=1 python3 campaign/comparators/nanoattention/ablate_summary.py`
to reproduce the table. The script uses three rounds, so effects near the observed
spread are inconclusive.

### Contributors and portability

Source references below are to nanoattention revision
`6f782a8dace846e4e64a5a12cbdf5c0252166199`. A structural idea can be tested
with f32 activations; it is not a parity result. Nano's unconditional BF16
rounding still applies to every run. Any changed reduction order requires the
TensorFold f32 logit and independent arithmetic checks.

- `batch`: **structural scheduling**. `src/mega/knobs.zig:7` sets the maximum
  prefill rows per forward. The default 64 requires 16 forwards for 1,024 rows,
  versus eight at 128. TensorFold's `zig/src/families/inkling2_loop/prefill.zig:14`
  uses chunk 512 and its float matrix kernel at `kernels/prefill.metal:76` uses
  32-row tiles. This comparison does not show that changing TensorFold's chunk
  would improve it.
- `expert_simd` and `expert_wide_dot`: **structural SIMD and reuse**.
  `src/mega/exec.zig:618` selects SIMD dot kernels; `src/mega/inkling_ops.metal:620`
  reuses float inputs across four outputs and reduces in float. Gate and down
  outputs may still be rounded by nano at `inkling_ops.metal:667` and `:722`.
  TensorFold's `kernels/experts.metal:227` and `:263` use a SIMD group per
  output column with float accumulation. Four-output reuse is the concrete
  alternative; it needs parity validation.
- `expert_prefill`: **mixed structural and reduced-precision group**. The
  expert-major route lists, reuse and split workgroups are structural:
  `src/mega/exec.zig:587` and `:615`; `src/mega/inkling_ops.metal:1177`.
  The selected block also casts inputs to BF16 at `inkling_ops.metal:1122`.
  TensorFold already sorts routes at `prefill.zig:630` and dispatches an expert
  matrix at `:644`; `kernels/prefill.metal:861` uses 32x64x32 float tiles.
  Nano stripes 32-row tiles across up to four workgroups and selects 8- or
  16-row tails at `inkling_ops.metal:1185`. The group ratio cannot attribute the
  gain between those structural choices and BF16 arithmetic.
- `expert_mlx_scale`: **mixed, including BF16-dependent matrix operands**.
  `src/mega/exec.zig:589` selects the block kernel and the row-list and workgroup
  scheme above. The block stages BF16 inputs and decoded weights and uses BF16
  fragments with float accumulators at `src/mega/inkling_ops.metal:1086`,
  `:1122` and `:1148`. TensorFold's `kernels/prefill.metal:861` decodes weights
  into float threadgroup storage and multiplies float matrices. The stripe and
  tail selection can be investigated independently of the BF16 operands.
- `bf16_linear_tile`: **structural float tile**. `src/mega/inkling_ops.metal:736`
  reuses float input across four output rows and BF16 weights across tokens,
  accumulating in float. Nano may round the result at `:758`. TensorFold's
  `kernels/prefill_smallrow.metal:6` uses one output column per SIMD group and
  reuses each weight across rows; its large `i2p_gemm` at
  `kernels/prefill.metal:76` uses 32x64x32 float matrix tiles.
- `bf16_prefill_tile`: **structural tiling**, with two dispatch effects.
  `src/mega/inkling_ops.metal:803` uses a four-output, eight-token vector tile
  with float inputs and float accumulation; `src/mega/exec.zig:803` also selects
  a four-row grouped projection. TensorFold's matching dense and relative-bias
  kernels are `kernels/prefill.metal:76` and `:380`, respectively. Nano's dense
  path uses register tiles and TensorFold's large dense path uses threadgroup
  float matrices; the relative-bias paths should be compared separately.
- `bf16_prefill_matrix`: **BF16 operand dependent**. `src/mega/exec.zig:720`
  selects 16- or 32-row matrix tiles; `src/mega/inkling_ops.metal:885` casts
  float activations into BF16 threadgroup storage and `:890` multiplies BF16
  fragments into float accumulators. Its removal also changes the vector-tile
  selection at `exec.zig:736`, so the measured ratio includes both effects.
- `bf16_decode_vec`: **structural vector loads and reuse**.
  `src/mega/inkling_ops.metal:841` loads float4 activations and BF16 weights,
  accumulates four outputs in float and may round output. TensorFold's
  `kernels/dense.metal:8` uses float activations and BF16 weights; compare its
  SIMD work distribution and weight loads with nano's four-output reuse.
- `fuse_bf16_cast`: **BF16-cast dependent dispatch reduction**.
  `src/mega/plan.zig:446` matches an in-place BF16 round after a producer and
  `:469` folds it into that producer. TensorFold does not need that cast under
  its f32 contract, so the measured saving is not an f32 kernel port.
- `concurrent_scratch`: **structural scheduling**. `src/mega/plan.zig:227`
  detects independent accesses and `src/mega/exec.zig:450` marks dispatches
  concurrent. TensorFold's prefill encoder dispatches the operations at
  `prefill.zig:630` and `:642`; overlapping independent operations would need
  a separate dependency and buffer-lifetime proof. Busy-host prefill ratios
  varied substantially, while the median decode ratio was 0.946.
- `relative_split` and `relative_split_4`: **structural partitioning**.
  `src/mega/exec.zig:1005` dispatches partitions and a merge; the four-way path
  at `src/mega/inkling_ops.metal:1278` processes keys with four SIMD groups
  and float data. TensorFold already partitions and combines float attention at
  `kernels/attention.metal:6` and `:61`. Nano assigns four SIMD groups to keys
  within a partition, while TensorFold groups heads for a KV head and stages
  keys and values in threadgroup memory at `attention.metal:14` and `:30`.
  The `relative_split` group result includes the four-way option.
- `relative_tile_prefill`: **reduced-precision operand dependent, FP16**.
  `src/mega/inkling_ops.metal:103` tiles 32 queries by 16 keys and converts
  queries, keys and values to half at `:118` and `:135`. This is FP16, not BF16,
  and cannot be ported unchanged under the f32 contract. TensorFold's
  `kernels/prefill.metal:440` keeps its attention inputs in float.
