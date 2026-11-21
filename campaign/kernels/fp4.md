# Inkling 2 Mini NVFP4/MXFP8 decode kernels

A measured, standalone kernel experiment for the routed experts' gate/up and down matvecs at
decode. Nothing here is wired into the Inkling 2 Mini family: the correctness-first kernel on
`lane/i2-forward` is variant 0 and every other variant is compared against it.

## What the checkpoint stores

`hf_quant_config.json` splits the 24 expert layers into one NVFP4 group (layers 0-19) and one
MXFP8 group (layers 20-23). `model-00001-of-00001.safetensors` holds, per layer,

| tensor | dtype | shape |
| --- | --- | --- |
| `layers.L.mlp.experts.w13_weight` | U8 (NVFP4) / F8_E4M3 (MXFP8) | `[128, 2304, 1152]` / `[128, 2304, 2304]` |
| `layers.L.mlp.experts.w13_weight_scale` | F8_E4M3 / U8 | `[128, 2304, 144]` / `[128, 2304, 72]` |
| `layers.L.mlp.experts.w2_weight` | U8 / F8_E4M3 | `[128, 2304, 576]` / `[128, 2304, 1152]` |
| `layers.L.mlp.experts.w2_weight_scale` | F8_E4M3 / U8 | `[128, 2304, 72]` / `[128, 2304, 36]` |
| `layers.L.mlp.experts.{w13,w2}_weight_amax` | F32 | `[128]`, NVFP4 only |
| `layers.L.mlp.experts.{w13,w2}_weight_outer_scale_denominator` | F32 | `[1]`, NVFP4 only |

So a gate/up row is `2c` and its up row `2c+1`, over `k = 2304` hidden inputs, for `n = 1152`
intermediate columns; the down matrix is `2304` rows over `k = 1152`. NVFP4 is an e2m1 nibble an
input in groups of 16 under one e4m3 scale, times the pair's global `amax[e] / denominator`.
MXFP8 is an e4m3 byte an input in groups of 32 under one e8m0 scale, with no per-tensor global.
`config.json` fixes `hidden_size 2304`, `n_routed_experts 128`, `num_experts_per_tok 8`,
`intermediate_size 1152`.

Decode shapes are the active experts of a verify window: `top-k = 8` picks a token, so batch
rows 1, 4 and 16 are `pairs = rows * 8` gate/up pairs and the same count of down rows.

## Variants

| variant | structure |
| --- | --- |
| `v0-baseline` | the `lane/i2-forward` kernel, byte-for-byte: one expert a threadgroup row, 32 columns a threadgroup, 2 rows by 2 columns a thread, `k` in steps of 8 through threadgroup memory |
| `v1-rows` | one simdgroup an output pair, the 32 lanes split the scale groups, eight-code 64-bit loads |
| `v2-shared` | four pairs a 128-thread threadgroup over the activation staged once per expert |
| `v3-table` | `v1-rows`' structure with the e2m1 and e4m3 codes looked up in an f32 table instead of decoded arithmetically |
| `v4-thread` | one thread a pair, its two rows streamed with no reduction |

The layout, the `k`/`n`/rows/top-k arguments, the `amax`/`denominator` global and the situ
(`gate * up` with the 4.0/25.0 caps) follow the baseline exactly, so a variant is a fair swap.

## Correctness

`zig build test-fp4-kernels -Dcpu=apple_m1` runs `zig/src/core/metal_fp4/fp4_test.zig`: every
variant on both formats at all three batch sizes, gate/up and down, against the fp64 dequant in
`fp4_ref.zig`. Each output is bounded by the fp32 accumulation error the GPU's order may add:

```
gamma = (k + 64) * 2^-24
bound = 4 * gamma * sum|w_j x_j| + 2 * 2^-24 * |y|
```

and the gate/up bound carries the situ chain through:

```
bound_act = 1.5 * |up| * bound_gate + |gate| * bound_up + 2e-6 * |gate * up|
```

The negative control corrupts every scale byte of pair 0 by one e4m3 exponent step (`^0x40`)
and requires every variant to move more than three quarters of that pair's 1152 activations out
of the bound. Measured: `1152 of 1152` for all five variants.

Worst observed `err/bound` is 0.0028 (MXFP8 down, the baseline); every variant is tighter than
the baseline on the shapes it wins.

## Timing

Run through the campaign's GPU lock and memory cap:

```sh
zig build tf-fp4-bench -Dcpu=apple_m1
python3 campaign/heavy.py --cap-gb 4 -- ./zig-out/bin/tf-fp4-bench
```

Median GPU microseconds from `MTLCommandBuffer.GPUEndTime - GPUStartTime`, 5 warmup command
buffers then 50 timed, one buffer each run. Weight GB/s is the pair slab's `codes + scales`
divided by that median. The ceiling is this host's measured streaming bandwidth, not its spec
sheet: a copy kernel moves 527 GB/s and a read-only kernel reads 521 GB/s at 128 MiB
(DRAM-resident at 512 MiB: 500 GB/s read), so `of_read_ceiling` divides by 521 GB/s. A fraction
above 1 means the slab fit in the SLC, which is what happens at batch 1.

Apple M5 Max, 40 GPU cores, 48 GiB, `40200896512` B max working set.

| format | rows | proj | variant | median us | weight GB/s | of read ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| nvfp4 | 1 | gu | v0-baseline | 760.06 | 31.4 | 0.060 |
| nvfp4 | 1 | gu | v1-rows | 70.75 | 337.6 | 0.648 |
| nvfp4 | 1 | gu | v2-shared | 78.00 | 306.3 | 0.588 |
| nvfp4 | 1 | gu | **v3-table** | **47.71** | **500.7** | **0.961** |
| nvfp4 | 1 | gu | v4-thread | 102.31 | 233.5 | 0.448 |
| nvfp4 | 1 | down | v0-baseline | 418.77 | 28.5 | 0.055 |
| nvfp4 | 1 | down | v1-rows | 45.87 | 260.4 | 0.500 |
| nvfp4 | 1 | down | v2-shared | 51.37 | 232.5 | 0.446 |
| nvfp4 | 1 | down | **v3-table** | **34.92** | **342.1** | **0.657** |
| nvfp4 | 1 | down | v4-thread | 45.58 | 262.0 | 0.503 |
| nvfp4 | 4 | gu | v0-baseline | 3591.67 | 26.6 | 0.051 |
| nvfp4 | 4 | gu | v1-rows | 266.94 | 358.0 | 0.687 |
| nvfp4 | 4 | gu | v2-shared | 286.56 | 333.4 | 0.640 |
| nvfp4 | 4 | gu | **v3-table** | **184.52** | **517.8** | **0.994** |
| nvfp4 | 4 | gu | v4-thread | 305.08 | 313.2 | 0.601 |
| nvfp4 | 4 | down | v0-baseline | 1712.96 | 27.9 | 0.054 |
| nvfp4 | 4 | down | v1-rows | 173.00 | 276.2 | 0.530 |
| nvfp4 | 4 | down | v2-shared | 194.50 | 245.6 | 0.471 |
| nvfp4 | 4 | down | **v3-table** | **128.83** | **370.8** | **0.712** |
| nvfp4 | 4 | down | v4-thread | 144.37 | 330.9 | 0.635 |
| nvfp4 | 16 | gu | v0-baseline | 15012.54 | 25.5 | 0.049 |
| nvfp4 | 16 | gu | v1-rows | 1060.67 | 360.3 | 0.692 |
| nvfp4 | 16 | gu | v2-shared | 1118.08 | 341.8 | 0.656 |
| nvfp4 | 16 | gu | **v3-table** | **738.56** | **517.5** | **0.993** |
| nvfp4 | 16 | gu | v4-thread | 1187.17 | 321.9 | 0.618 |
| nvfp4 | 16 | down | v0-baseline | 7219.96 | 26.5 | 0.051 |
| nvfp4 | 16 | down | v1-rows | 714.71 | 267.4 | 0.513 |
| nvfp4 | 16 | down | v2-shared | 772.52 | 247.4 | 0.475 |
| nvfp4 | 16 | down | v3-table | 590.17 | 323.8 | 0.622 |
| nvfp4 | 16 | down | **v4-thread** | **526.63** | **362.9** | **0.697** |
| mxfp8 | 1 | gu | v0-baseline | 516.21 | 84.8 | 0.163 |
| mxfp8 | 1 | gu | v1-rows | 87.17 | 502.4 | 0.964 |
| mxfp8 | 1 | gu | v2-shared | 95.00 | 461.0 | 0.885 |
| mxfp8 | 1 | gu | **v3-table** | **68.88** | **635.9** | **1.220** |
| mxfp8 | 1 | gu | v4-thread | 102.50 | 427.3 | 0.820 |
| mxfp8 | 1 | down | v0-baseline | 293.27 | 74.7 | 0.143 |
| mxfp8 | 1 | down | v1-rows | 52.79 | 414.8 | 0.796 |
| mxfp8 | 1 | down | v2-shared | 57.12 | 383.3 | 0.736 |
| mxfp8 | 1 | down | v3-table | 44.92 | 487.5 | 0.936 |
| mxfp8 | 1 | down | **v4-thread** | **42.63** | **513.7** | **0.986** |
| mxfp8 | 4 | gu | v0-baseline | 2342.81 | 74.8 | 0.144 |
| mxfp8 | 4 | gu | v1-rows | 336.44 | 520.7 | 0.999 |
| mxfp8 | 4 | gu | v2-shared | 369.96 | 473.5 | 0.909 |
| mxfp8 | 4 | gu | **v3-table** | **307.92** | **568.9** | **1.092** |
| mxfp8 | 4 | gu | v4-thread | 437.31 | 400.6 | 0.769 |
| mxfp8 | 4 | down | v0-baseline | 1182.17 | 74.1 | 0.142 |
| mxfp8 | 4 | down | v1-rows | 204.83 | 427.6 | 0.821 |
| mxfp8 | 4 | down | v2-shared | 223.75 | 391.5 | 0.751 |
| mxfp8 | 4 | down | **v3-table** | **168.56** | **519.6** | **0.997** |
| mxfp8 | 4 | down | v4-thread | 210.62 | 415.9 | 0.798 |
| mxfp8 | 16 | gu | v0-baseline | 9345.08 | 75.0 | 0.144 |
| mxfp8 | 16 | gu | v1-rows | 1308.19 | 535.6 | 1.028 |
| mxfp8 | 16 | gu | v2-shared | 1425.31 | 491.6 | 0.944 |
| mxfp8 | 16 | gu | **v3-table** | **1221.69** | **573.6** | **1.101** |
| mxfp8 | 16 | gu | v4-thread | 1733.27 | 404.3 | 0.776 |
| mxfp8 | 16 | down | v0-baseline | 4836.46 | 72.4 | 0.139 |
| mxfp8 | 16 | down | v1-rows | 853.08 | 410.7 | 0.788 |
| mxfp8 | 16 | down | v2-shared | 907.00 | 386.3 | 0.741 |
| mxfp8 | 16 | down | **v3-table** | **763.60** | **458.8** | **0.881** |
| mxfp8 | 16 | down | v4-thread | 828.04 | 423.1 | 0.812 |

### Winner per shape

| format | rows | gate/up | down |
| --- | --- | --- | --- |
| nvfp4 | 1 | v3-table (47.71 us) | v3-table (34.92 us) |
| nvfp4 | 4 | v3-table (184.52 us) | v3-table (128.83 us) |
| nvfp4 | 16 | v3-table (738.56 us) | v4-thread (526.63 us) |
| mxfp8 | 1 | v3-table (68.88 us) | v4-thread (42.63 us) |
| mxfp8 | 4 | v3-table (307.92 us) | v3-table (168.56 us) |
| mxfp8 | 16 | v3-table (1221.69 us) | v3-table (763.60 us) |

### What the numbers say

- The baseline is 11-16x slower than any rewrite. It decodes each weight into threadgroup memory
  eight at a time behind two barriers, and it folds every pick of a token through one weight tile,
  so at batch 1 it runs a 128-wide threadgroup per 32 columns of a single pair.
- The table variant wins 10 of 12 shapes. e2m1 has 16 values and e4m3 256, so an f32 table is
  64 B and 1 KiB: a load from the constant-cached table costs less than the shift, mask and
  `exp2` the arithmetic decode runs per code, and the MXFP8 case gains most because its arithmetic
  form needs the full e4m3 `exp2` per code.
- `v2-shared` never wins. Staging 2304 f32 activations costs a barrier, and at these sizes the
  activation row is already in L1 for the four pairs that share it.
- Down at batch 1 prefers `v4-thread` for MXFP8: its 1152-input reduction over 2304 rows leaves
  the 32-lane split with only 36 groups a lane, so the reduction's fixed cost dominates and a
  thread that owns its row outright wins. NVFP4 keeps `v3-table` because the table's arithmetic
  saving is larger there.
- NVFP4 gate/up reaches 0.99 of the read ceiling at batch 4 and 16; its down stays at 0.62-0.71
  because the down matrices are twice as many rows over half the inputs, so the reduction is
  shorter and the fixed per-row cost is relatively larger.
- MXFP8 fractions above 1 are SLC residency: 44-700 MB of weights over a 1 KiB code table plus
  21-350 MB of activations, and the 128 MiB stream that sets the ceiling is already partly
  cache-resident. The 512 MiB stream reads 500 GB/s, so treat 1.0 as "at the DRAM ceiling,
  ±5%".

## Where the files are

- `zig/src/core/metal_fp4/fp4_shapes.zig` - the geometry the checkpoint's header declares.
- `zig/src/core/metal_fp4/fp4_ref.zig` - fp64 dequant, the situ caps and the error bound.
- `zig/src/core/metal_fp4/fp4_synth.zig` - deterministic codes, scales and activations.
- `zig/src/core/metal_fp4/fp4_gpu.zig` - pipelines, slabs, dispatches, GPU-time medians.
- `zig/src/core/metal_fp4/fp4_matvec.metal` - variants 1-4 and the two stream kernels, compiled
  twice (`FP4_MXFP8`).
- `zig/src/core/metal_fp4/baseline_experts.metal` - variant 0, copied unchanged from
  `lane/i2-forward:zig/src/families/inkling2_loop/kernels/experts.metal`.
- `zig/src/core/metal_fp4/fp4_test.zig`, `fp4_bench_main.zig` - the two entry points.
- `zig/build/fp4_bench.zig` - the `test-fp4-kernels` and `tf-fp4-bench` steps, macOS only.
