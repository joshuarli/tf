# Inkling 2 Mini (`inkling2_loop`)

The concurrent Inkling host serves no drafts. It accepts `INKLING_COPY_DRAFTS` and ignores it
(`zig/src/native/inkling2_loop_host.zig:581`). The [copy verification receipt](../../campaign/kernels/inkling-verify.md)
records the earlier single-stream copy drafts.

This page records the graph of `thinkingmachines/Inkling-2-Mini` as read from the pinned llama.cpp reference
(commit `40b7a546defe1f60edd1338c719552ffc58c89a0`) and the checkpoint's own config, index, quantization config and
chat template. It is the prior-art map for the native `inkling2_loop` family (I1–I3).

Citations are `file:line` in the llama.cpp tree at that commit unless the file is named explicitly. Checkpoint
paths are relative to `~/models/Inkling-2-Mini@ed40e7da625f` (revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`,
20.2 GB, Apache-2.0).

## Weights

| fact | value | source |
| --- | --- | --- |
| architecture | `Inkling2LoopForConditionalGeneration`, `model_type: inkling2_loop` | `config.json:3-6` |
| text hidden size | 2304 | `config.json:12` |
| physical layers | 24 | `config.json:13` |
| attention | 24 query heads, 8 KV heads, head 128 | `config.json:15-17` |
| vocabulary | 201,024 padded, 200,058 unpadded, EOS 200006 | `config.json:14,51,7` |
| context | 1,048,576 | `config.json:10` |
| experts | 128 routed, 8 per token, no shared expert, FFN 1152 | `config.json:59-62` |
| quantization | ModelOpt `NVFP4` at the top level, mixed per tensor below | `config.json:118-121` |

`model.safetensors.index.json` is one 19.5 GB file plus a 0.64 GB `mtp.safetensors`. `mtp.safetensors` is the native
draft head; llama.cpp's converter skips it (`conversion/inkling2.py:188`) and the text graph never reads it. Vision
weights live in the same checkpoint and are out of scope here.

### Per-tensor NVFP4 and MXFP8

`hf_quant_config.json` has five `model.parameter_groups`:

| `weight_transform` | tensors | layers | storage |
| --- | --- | --- | --- |
| `nvfp4_nearest` | 40 (`w13_weight`, `w2_weight`) | 0–19 | `U8` nibble pairs + `F8_E4M3` block scales + `F32` per-expert `amax` and one `outer_scale_denominator` |
| `mxfp8_nearest` | 8 | 20–23 | `F8_E4M3` weights + `U8` UE8M0 block scales |
| `cast` | the rest | all | BF16 (a few `F32`: clip scales, `gate.bias`, attention-residual queries, conv weights, rel bias) |

Shapes per layer: `w13_weight` is `[128, 2304, 1152]` for NVFP4 (two interleaved 1152-row halves, two 4-bit values a
byte; 2304 logical rows = 2 × FFN 1152) and `[128, 2304, 2304]` for MXFP8 (one byte a value). `w2_weight` is
`[128, 2304, 576]` and `[128, 2304, 1152]`. The scale tensors are `[128, 2304, cols/16]` (NVFP4) and
`[128, 2304, cols/32]` (MXFP8); `w13_weight_amax` is `[128]`, `w13_weight_outer_scale_denominator` is `[1]`.

Decoding is the converter's (`conversion/inkling2.py:33-65`):

- NVFP4: `v = E2M1[nibble] × E4M3[block_scale] × (amax[expert] / denominator)`, block 16 along the input dimension;
  the low nibble is the even column.
- MXFP8: `v = E4M3[byte] × 2^(scale_byte − 127)`, block 32; no `amax` or denominator is allowed
  (`conversion/inkling2.py:134-139`).

The converter rejects NaN scale bytes, a block shape other than `(1,16)`/`(1,32)`, and any floating-point expert
weight that still has quantization companions (`conversion/inkling2.py:143-160`).

## Architecture

### 24 layers, two attention types

`local_layer_ids` lists 20 of the 24 layers: `0-4, 6-10, 12-16, 18-22` (`config.json:28-49`). Those are
`hybrid_sliding` with a **1024-token sliding window** (`config.json:56`); the other four, **layers 5, 11, 17 and 23**,
are `hybrid` (full attention). The converter derives `layer_types` from that list when the config omits it
(`conversion/inkling2.py:108-114`).

The two types differ in more than the mask:

- SWA layers normalise one attention input with `attn_norm` and reuse it for Q, K and V (`src/models/inkling2.cpp:316-318`);
  full layers have separate `q_input_norm`, `k_input_norm` and `v_input_norm` (`src/models/inkling2.cpp:319-322`).
- SWA K/V convolutions are static; full K/V convolutions add per-token taps projected from the same input
  (`src/models/inkling2.cpp:328-331`).
- Full layers multiply Q and the relative states by the log-scaled position factor `tau`
  (`src/models/inkling2.cpp:339-342`); SWA layers do not.

There is no RoPE. Position enters only through the relative bias and `tau` (`src/models/inkling2.cpp:59-60`).

### The core loop: layers 6–18, three passes, rank 64

`loop_config` is `core_start 6`, `core_end 18`, `num_passes 3`, `gate_rank 64`, `is_varloop false`
(`config.json:67-73`).

llama.cpp assigns one KV/conv *cache slot* per executed layer instance: 24 physical layers plus
`(3−1) × (18−6) = 24` extra slots, **48 in total** (`src/models/inkling2-loop.cpp:24-30`). Each pass keeps its own
KV and convolution state; the physical weights are shared pointer-for-pointer
(`src/models/inkling2-loop.cpp:64-67`), and a slot maps back to its physical layer with
`layer_phys(il) = il < 24 ? il : 6 + (il − 24) mod 12` (`src/models/inkling2-loop.cpp:3-5`).

Execution order (`src/models/inkling2-loop.cpp:74-112`):

1. layers 0–5 once; the result is the *prelude*.
2. pass 0 over physical layers 6–17 (slots 6–17).
3. passes 1 and 2 restart from the prelude each time, over slots 24–35 and 36–47; at each core layer the graph
   mixes a message from the previous pass into the Q, K, V and FFN inputs.
4. layers 18–23 once, on the final pass's output.

Later passes do not see the previous pass through the KV cache: each pass recomputes its own K/V from its own inputs,
and its cache slot is separate. The only cross-pass path is the message.

**Messages.** Per core layer `i` the checkpoint has four rank-64 controllers; the converter stacks
`middle_cycle_cross_pass_queries[2i]`, `..._kv_queries[2i]`, `..._kv_queries[2i+1]`, `..._queries[2i+1]` into the
four query rows `[Q, K, V, FFN]` (`conversion/inkling2.py:315-329`). Each pass's *bank* is the previous pass's
block outputs after the first core block plus the previous pass's final output
(`src/models/inkling2-loop.cpp:88-97`); the message is a depth attention over that bank, and a controller applies it:
`out = (1 − tanh(f(c)))·x + tanh(g(c))·message`, where `c = [rms(x), rms(message)]`, `f`/`g` are the controller
`forget`/`gain` matrices, and the `down` matrix is rank 64 (`src/models/inkling2.cpp:240-248`).

The 12 core layers carry `controllers.{attn,k,v,mlp}.{down,gain,forget}_weight`
(shapes `[64,4608]`, `[2304,64]`, `[2304,64]`); no other layer does.

### Attention-residual blocks

Every layer except layer 0 has a learned attention-input query: `[n_embd, 3]` for full layers, `[n_embd, 1]` for
SWA (`src/models/inkling2.cpp:145-147`). The query mixes the depth of earlier block outputs and the current state:

```
states = rms_norm([blocks in reverse order, cur])          # no weight, eps 1e-6
w      = softmax(statesᵀ · query / sqrt(n_embd))            # over the depth axis
mixed  = Σ_i w_i · states_i
```

(`src/models/inkling2.cpp:178-191`.) The full-layer three queries become the Q, K and V inputs; the SWA single query
is shared (`src/models/inkling2.cpp:296-303`). The FFN input is mixed the same way with one per-layer query
(`src/models/inkling2.cpp:390`), and the final hidden state with `output_query` (`src/models/inkling2.cpp:423-424`).

Residual streams restart every `n_layers_per_attn_res_block = 6` executed positions (`config.json:57`). At those
positions the pre-attention state is pushed as a new block state instead of being added to the attention output; at
all other positions it is added (`src/models/inkling2.cpp:379-384`). The boundaries are at executed positions 0 (prelude), 6 and 12 of every pass, and 18 (the first post-loop layer);
the output mix therefore averages four block states plus the final state.

### Relative positions and log scaling

`d_rel = 16`, `rel_extent = 1024` (`config.json:18-19`). Each token projects `R = x·W_r` to
`[n_head, d_rel]`, clips it, and builds one bias row per backward distance:

```
table[t, h, d]     = Σ_r attn_rel_bias[r, d] · R[t, h, r] − sinks[t, h]     # d ∈ [0, 1024)
kq_b[t, h, j]      = table[t, h, distance] with distance = t − j, or the zero row at index 1024 if outside
```

(`src/models/inkling2.cpp:356-366`.) The bias table has shape `[16, 1024]` and is padded with a **zero** row before
the projection (`src/models/inkling2.cpp:357`); that zero row is what out-of-window distances get. The learned sink
is subtracted from every row, then a zero-valued sink logit is added to the softmax denominator: for SWA layers the
out-of-window row becomes `−sink`, for full layers distances past 1024 read the zero row.

**The bias is applied after the softmax scale.** This llama.cpp routes `kq_b` through the attention mask
(`src/llama-graph.cpp:4508`: `kq_mask = ggml_add(kq_b, kq_mask)`), so the score is
`q·k / sqrt(head_dim) + kq_b`, not `(q·k + kq_b) / sqrt(head_dim)`. The sink algebra is equivalent to Transformers'
extra zero-value logit (`transformers/models/inkling2/modeling_inkling2.py`, `eager_attention_forward`).

Full layers scale Q and R by `tau = 1 + 0.1·log(max(1, (pos+1)/8192))` (`src/models/inkling2.cpp:258-261`,
`config.json:22-23`). For the first 8192 positions the log term is exactly 0, so `tau = 1`; these histories never reach it.

### Clipping and normalization

| tensor | transform | source |
| --- | --- | --- |
| Q | per `(head, dim)` learned scale, then L2 clip to 22.627416997969522 per head | `src/models/inkling2.cpp:334`, `config.json:26` |
| K | L2 clip to 22.627416997969522 per head, no scale | `src/models/inkling2.cpp:335` |
| V | unweighted RMS norm per head, eps 1e-6 | `src/models/inkling2.cpp:336`, `config.json:25` |
| R | per `(head, rel)` learned scale, then L2 clip to 8 | `src/models/inkling2.cpp:337`, `config.json:27` |

The clip is `x·min(1, threshold/(‖x‖ + 1e-6))` (`src/models/inkling2.cpp:164-175`). The learned scales are stored
`[n_head, head_dim]` and `[n_head, d_rel]`.

### Attention output and short convolutions

The attention output is multiplied elementwise by `sigmoid(x·W_gate)` (`src/models/inkling2.cpp:374`), projected by
`W_o`, and passed through a causal depthwise convolution with kernel 4 (`config.json:50`). The same convolution
shape appears four times in a layer: K, V, the attention output and the FFN output, writing disjoint channel ranges
of one recurrent state of `2·n_embd_k_gqa + 2·n_embd = 6656` channels (`src/models/inkling2.cpp:62-80`,
`src/models/inkling2.cpp:290-292`):

| site | offset |
| --- | --- |
| K | 0 |
| V | `n_embd_k_gqa` = 1024 |
| attention output | `2·n_embd_k_gqa` = 2048 |
| FFN output | `2·n_embd_k_gqa + n_embd` = 4352 |

Each site returns `(x + conv(x)) / sqrt(k·4 + 1)`, with `k = 1` for a static convolution and `k = 2` when dynamic
taps are added (`src/models/inkling2.cpp:236`), i.e. `sqrt(5)` for SWA layers and `3` for full layers.

Static taps: SWA layers store `k_sconv.weight`/`v_sconv.weight` as `[channels, 1, 4]`; full layers store
`static_w` as `[4, channels]` and the converter flips the kernel axis (`conversion/inkling2.py:260-266`), so
`static_w[0]` weights the current token. Dynamic taps: `k_sconv.weight_proj`/`v_sconv.weight_proj` are
`[4·n_kv, n_embd]`; each row group `[lag, head]` scales the input `lag` steps back, per head, and adds to the static
tap before the conv (`src/models/inkling2.cpp:218-234`).

### Mixture of experts

128 routed experts, 8 active, no shared expert, FFN width 1152 (`config.json:59-62`). Per token:

1. `logits = x·W_routerᵀ` computed in F32 (`src/models/inkling2.cpp:56`).
2. `probs = sigmoid(logits)`. The learned `gate.bias` (the checkpoint's `mlp.gate.bias`, mapped to
   `exp_probs_b`, `conversion/inkling2.py:194`) is added **only** for top-8 selection
   (`src/llama-graph.cpp:2446-2452`). The selected weights are the unbiased `probs`.
3. Weights are renormalised with `+1e-20` (`expert_weights_norm_eps`, `src/models/inkling2.cpp:57`) and multiplied
   by `route_scale = 8` (`config.json:65`).
4. Each expert computes `gate_up = x·W13ᵀ` (gate rows first, up rows second), then
   `situ(g,u) = 4·tanh(g/4)·sigmoid(g) · 25·tanh(u/25)` (`src/llama-graph.cpp:2666-2683`, softcaps from
   `config.json:62-63`), then `down = situ·W2ᵀ`; outputs are summed with the selected weights.
5. The sum is convolved with the FFN tap set and added to the residual (`src/models/inkling2.cpp:414-418`).

The checkpoint stores `w13` interleaved: even rows are gate, odd rows are up. The converter reshapes
`(E, 2·FFN, H) → (E, FFN, 2, H) → transpose(1,2)` so the GGUF tensor holds gate rows first
(`conversion/inkling2.py:255-258`).

### Output

The final state is depth-mixed with `output_query`, RMS-normalised with `model.llm.norm`, projected by the
`unembed` matrix over the padded vocabulary, and soft-capped at 30: `30·tanh(logit/30)`. The padded rows are dropped
afterwards, so the returned logits have 200,058 columns (`src/models/inkling2.cpp:423-449`, `config.json:52`).
`embed` and `unembed` are `[201024, 2304]`; the tokenizer must cover exactly the unpadded 200,058
(`src/models/inkling2.cpp:87-100`).

### MTP head (not in the text graph)

`mtp_config` (`config.json:90-117`) and `mtp.safetensors` define a 3-layer draft head: the same 24/8 head geometry,
dense FFN 9216, local (sliding) layers 0 and 2, `block_size 8`, a 256-rank `rnn_hidden` Markov head, and
`target_layer_ids [5, 23, 41, 48]`. The indices count execution visits. Intermediate taps capture the raw partial
state entering the visit, after any loop rewind, before the layer runs; they are neither completed-layer outputs nor
sums of residual-history banks. Tap 5 enters physical layer 5, tap 23 enters layer 11 in pass 1, and tap 41 enters
layer 17 in pass 2. Tap 48 is the final depth-mixed, normalized output. TokenSpeed captures before the layer and
appends its normalized output (`tokenspeed@947b25bc92d7/python/tokenspeed/runtime/models/inkling2.py:640-671`);
its CPU capture test distinguishes the raw partial from the residual sum
(`tokenspeed@947b25bc92d7/test/runtime/models/test_inkling2_capture.py:81-127`). MLX-VLM independently captures
`x` before the layer and the normalized final state (`mlx-vlm@f1618d6d52be/mlx_vlm/models/inkling2/language.py:688-712`);
its visit ordering is in `mlx_vlm/models/inkling2/config.py:74-83`.

The head mixes the four taps with `fuse_weight [4,4]` for its layers and recurrent head
(`tokenspeed@947b25bc92d7/python/tokenspeed/runtime/models/inkling2_mtp.py:557-559,618-629`). Its block has one
anchor and seven proposal rows (`tokenspeed@947b25bc92d7/python/tokenspeed/runtime/models/inkling2_mtp.py:715-733`),
and the next round uses the last accepted target features
(`tokenspeed@947b25bc92d7/python/tokenspeed/runtime/execution/drafter/inkling2.py:197-229`). See
[campaign/ref/dspark_taps.md](../../campaign/ref/dspark_taps.md) for the later drafter's contract. llama.cpp does
not load or execute the head.

## Checks

```bash
make -C campaign fetch NAMES="inkling-2-mini inkling-2-mini-tiny"
zig build test-inkling2-loop-host
```

The host tests cover the config, the tensor map, the image and `--vision` refusals and the chat template. They also
check that the checkpoint's tokenizer reproduces the token ids of the four histories in `campaign/ref/histories.json`,
which the pinned llama.cpp tokenizer produced. The tiny fixture
`trl-internal-testing/tiny-Inkling2LoopForConditionalGeneration@5e761216759d` (8 layers, hidden 32, core [2,6),
3 passes, rank 8, 8 experts, NVFP4 layers 0–5 and MXFP8 6–7) is pinned in the same lock for the family's prefill
tests in `zig build test-inkling2-loop`. `make -C campaign omlx-oracle-gate` holds the native teacher-forced logits
on the oQ4e checkpoint to the oMLX oracle; the [campaign README](../../campaign/README.md#omlx-correctness-oracle)
gives its setup and bounds.

## Unverified

- **Long context.** Positions ≥ 8192 exercise `tau` (`src/models/inkling2.cpp:258-261`). The native kernels apply
  it (`zig/src/families/inkling2_loop/kernels/prefill.metal:566`), but no history reaches it.
- **Vision.** The encoder and projector are not covered; the plan refuses image input.
- **MTP numerics.** Capture semantics are source-verified. Native drafters on the parked branches `lane/ds1-dspark-drafter`
  and `lane/m1-mtp` match MLX's proposals; neither is on main.
- **Model quality.** The logits agreement is a numerical check, not a capability check.
