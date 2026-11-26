# A1 (5i) TensorFold Specialization Inventory: Inkling 2 Mini oQ4e

**Lane:** `lane/inkling-metal-aot` (`eb18fe252`)
**Primary Sources:** `zig/src/families/inkling2_loop/*.zig`, `zig/src/families/inkling2_loop/kernels/*.metal`, `zig/src/families/inkling2/config.zig`, `zig/src/native/inkling2_loop*.zig`, `zig/build/inkling2_loop.zig`, `~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd/{config.json,model.safetensors.index.json}`, `~/models/Inkling-2-Mini-DFlash2@4f9088ea66c2/config.json`

---

## 1. Exact Geometry, Looped-Recurrence Topology & Quantization

*(Note: In `~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd/`, quantization overrides live directly inside `config.json` under `quantization` / `quantization_config`; `hf_quant_config.json` belongs to the unquantized/ModelOpt `Inkling-2-Mini@ed40e7da625f` and `fixtures/hf_quant_config.json`.)*

### 1.1 Model & Looped-Recurrence Parameters (`config.json` + `inkling2_loop/config.zig`)

| Parameter | Target (`Inkling-2-Mini-oQ4e-mtp`) | Drafter (`Inkling-2-Mini-DFlash2`) |
| --- | --- | --- |
| `model_type` / `architectures` | `inkling2_loop` / `Inkling2LoopForConditionalGeneration` | `qwen3` / `DFlash2DraftModel` |
| `hidden_size` (`h`) | `2,304` | `2,304` |
| `vocab_size` / `unpadded_vocab_size` | `201,024` / `200,058` (`eos_token_id = 200006`) | `201,024` (reuses target `embed` + trimmed `200,058` `lm_head`; `mask_token_id = 200056`) |
| `num_hidden_layers` -> executed slots | `24` physical layers -> **`48` executed slots** (`slotCount`) | `5` sliding layers (`is_causal = false`) |
| `loop_config` | `core_start = 6`, `core_end = 18`, `num_passes = 3`, `gate_rank = 64`, `is_varloop = false` | N/A |
| Attention heads (`Q` / `KV` / `dim`) | Local & Global: `24` / `8` / `128` (`qw = 3,072`, `kvw = 1,024`) | `32` / `8` / `128` |
| Position / Bias / Window | Relative bias `d_rel = 16`, `rel_extent = 1,024`, `sliding_window_size = 1,024`, `log_scaling_n_floor = 8,192`, `log_scaling_alpha = 0.1` | RoPE `rope_theta = 10,000.0`, `sliding_window = 2,048` (`max_window_layers = 5`) |
| Short convolutions & clips | `sconv_kernel_size = 4` (`ring = 4`), `qk_l2_clip = 22.627416997969522`, `rel_l2_clip = 8.0`, `rms_norm_eps = 1e-6`, `use_value_norm_no_gamma = true` | `conv_group_size = 16`, `conv_kernel_size = 2` |
| Attention Residuals (`attn_res`) | `n_layers_per_attn_res_block = 6`, `mix_states = 5`, `attn_res_normalize_values = true` | N/A |
| MoE FFN (target) / Dense FFN (drafter) | `n_routed_experts = 128`, `num_experts_per_tok = 8`, `n_shared_experts = 0`, `intermediate_size = 1,152`, `route_scale = 8.0`, `use_gate_bias = true`, `mlp_gate_softcap = 4.0`, `mlp_up_softcap = 25.0`, `final_logit_softcapping = 30.0` | Dense SwiGLU `intermediate_size = 6,912`, `block_size = 8`, `selector_rank = 256`, `selector_top_k = 16`, `target_layer_ids = [5, 23, 41, 48]`, `final_logit_softcapping = 30.0` |

### 1.2 Executed 48-Slot Topology (`prefill.zig` L1855–1894, `config.zig` L20–27)

Every physical layer is MoE (`128` routed experts, `0` shared/dense FFN layers). The `48` executed slots map to `24` physical layers as follows:

| Execution Stage | Visit IDs (`taps`) | Slot IDs (`Cache`) | Physical Layers (`ip`) | Attention Type (`isFull`) | Expert Format | `bank` (Cross-Pass Msg + 4 Controllers) | Block Pushes (`ip % 6 == 0`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Prelude** | `0..5` | `0..5` | `0..5` | `5` local (`0..4`), `1` global (`5`) | NVFP4 (`0..5`) | None (`prelude` saved after slot `5`, `count = 1`) | Visit/Slot `0` (`c.count` -> `1`) |
| **Core Pass 0** | `6..17` | `6..17` | `6..17` | `10` local, `2` global (`11, 17`) | NVFP4 (`6..17`) | None (`bank` captured after visit `17`: `bn = 2` states + `passout`) | Visits/Slots `6, 12` (`c.count` -> `3` -> reset to `1`) |
| **Core Pass 1** | `18..29` | `24..35` | `6..17` | `10` local, `2` global (`11, 17`) | NVFP4 (`6..17`) | **Active** (`bn = 2`, 4 controllers/layer) | Visits `18, 24` / Slots `24, 30` (`c.count` -> `3` -> reset to `1`) |
| **Core Pass 2** | `30..41` | `36..47` | `6..17` | `10` local, `2` global (`11, 17`) | NVFP4 (`6..17`) | **Active** (`bn = 2`, 4 controllers/layer) | Visits `30, 36` / Slots `36, 42` (`c.count` -> `3`) |
| **Exit** | `42..47` | `18..23` | `18..23` | `5` local, `1` global (`23`) | NVFP4 (`18..19`), **Affine-4 (`20..23`)** | None | Visit/Slot `18` / Visit `42` (`c.count` -> `4`) |
| **Head / Final** | `48` | — | `output_query` + `norm` + `lm_head` | — | Affine-8 | Mixes `4` block states + `x` (`depth = 5 <= mix_states`) | — |

- **Totals across 48 slots:** `40` local sliding slots (`window = 1,024`), `8` global full-context slots (`5, 11, 17, 23` in Prelude/Pass 0/Exit + `11, 17` x2 in Passes 1–2), `44` NVFP4 expert slots (`0..19` + `2*12`), `4` affine-4 expert slots (`20..23`), `24` bank/controller slots (`24..47`), and `8` `attn_res` block pushes.
- **Taps vs Slots:** `DFlash2` taps index **visit order** `[5, 23, 41, 48]` (`taps.encode` at `prefill.zig` L1857, L1883, L1891, L1901): visit `5` = slot `5` (prelude end), visit `23` = slot `29` (Pass 1 layer `11`), visit `41` = slot `47` (Pass 2 layer `17`), visit `48` = final normed hidden state `s.n0`.

### 1.3 Quantization Format & Safetensors Inventory (`model-0000{1..4}-of-00004.safetensors`)

Verified directly from checkpoint `config.json` (`98` overrides over default `affine 4-bit gs=64`) and all four safetensors JSON headers (`1,098` entries in `model.safetensors.index.json`, `16,211,691,939` B raw shard sum, `16,211,566,592` B summed tensor payloads):

| Tensor Group | Count | Precision / Pack | Exact Shapes | Checkpoint Bytes | 1-Row Active Bytes (48 slots) |
| --- | ---: | --- | --- | ---: | ---: |
| `model.embed.{weight,scales,biases}` + `embed_norm` | `4` | Affine-8 (`gs=64`) + `BF16` | `[201024,576]` U32, `[201024,36]` BF16, `[2304]` BF16 | `492,111,360` | `~7,056` (1 token row) |
| `lm_head.{weight,scales,biases}` + `norm` + `final_input.query` | `5` | Affine-8 (`gs=64`) + `BF16`/`F32` | `[201024,576]` U32, `[201024,36]` BF16 (`200,058` active rows) | `492,120,576` | `489,755,808` |
| Local attn `in_proj` (`20` layers) & `o_proj` (`24` layers) + Global `q_proj,k_proj,v_proj` (`4` layers) | `216` (`72 * 3`) | Affine-8 (`gs=64`) | Local `in_proj [8600,576]`; Global `q_proj [6552,576]`, `k/v_proj [1056,576]`; all `o_proj [2304,768]` U32 + BF16 `scales`/`biases` | `681,878,016` | `1,372,760,064` (`attn 1,011,787,776` + `oproj 360,972,288`) |
| Attention `q_scale`, `r_scale`, `rel_proj`, `k/v_conv`/`static`, `attn_conv`, `mlp_conv`, `norms`, `attn_res` queries | `223` | `BF16` (weights/scales/convs) + `F32` (queries) | `rel_proj [16,1024]`, convs `[1024,4]`/`[4,1024]`/`[2304,4]`, queries `[1|3,2304]` | `7,497,216` | `6,013,440` (`convs,scales,rel,norms`) |
| Controllers `loop_down`, `loop_gain_forget`, `loop_message.query` (layers `6..17`) | `36` | `F32` | `loop_down [4,2,2304,64]`, `loop_gain_forget [4,64,4608]`, `loop_message.query [4,2304]` | `113,688,576` | `227,377,152` (Passes 1 & 2 = `24` slots) |
| Routers `mlp.router_weight` + `router_bias` (`24` layers) | `48` | `F32` | `[128,2304]` F32 + `[128]` F32 | `28,323,840` | `56,647,680` |
| **NVFP4 Experts** layers `0..19` (`w13`, `w2` + `scales` + `global_scale`) | `120` | NVFP4 (`gs=16`, `U32` nibble codes + `U8` FP8 e4m3 scales + `F32` global; `den=2688` runtime) | `w13 [128,2304,288]` U32 + `[128,2304,144]` U8; `w2 [128,2304,144]` U32 + `[128,2304,72]` U8 (`4,478,976` B/expert) | `11,466,199,040` | `1,576,602,368` (`44` slots x `8` experts) |
| **Affine-4 Experts** layers `20..23` (`w13`, `w2` + `scales` + `biases`) | `24` | Affine-4 (`gs=64`, `U32` + `BF16` scales/biases) | `w13 [128,2304,288]` U32 + `[128,2304,36]` BF16 x2; `w2 [128,2304,144]` U32 + `[128,2304,18]` BF16 x2 (`4,478,976` B/expert) | `2,308,835,136` | `143,327,232` (`4` slots x `8` experts) |
| **Subtotal: Text Model Weights (Active Engine)** | **`676`** | — | — | **`15,575,054,336`** (`15.58 GB`) | **`3,872,476,976` (`3.872 GB` = `7.433 ms` @ `521 GB/s`)** |
| **Dead Vision Tower** (`vision_tower.*` — refused at open/request) | `422` | `BF16` | Unused by `inkling2_loop` (`refuse.vision`) | `636,512,256` (`0.64 GB`) | `0` |
| **Total Safetensors Model Payload** | **`1,098`** | — | `model-0000{1..4}` sum | **`16,211,566,592`** (`16.21 GB`) | **`3,872,476,976`** |
| **DFlash2 Drafter** (`Inkling-2-Mini-DFlash2/model.safetensors`) | `81` | `BF16` on disk -> prepared `affine4` (`q4`) in runtime (`~0.41–0.55 GiB` total drafter footprint in `SC2`) | 5 sliding layers (`Q=32, KV=8, inter=6912`) | `989,966,848` (`0.99 GB`) | Separate drafter step (`~0.4 GB` q4 prepared) |

### 1.4 KV Cache, Recurrent Conv State & Scratch Shapes up to `32,768` Context (`state.zig`, `prefill.zig`, `batch.zig`, `group.zig`)

- **Per-slot KV cache layout (`bf16` K and V buffers, `kvw = 8 * 128 = 1,024` dims = `2,048` B/pos/K or V = `4,096` B/pos K+V):**
  - `40` local slots hold a circular ring of `window = 1,024` rows each: **`40 * 1,024 * 4,096 = 167,772,160` B (`160.00 MiB`) fixed**.
  - `8` global slots grow linearly with `capacity`: **`8 * 4,096 = 32,768` B (`32.0 KiB`) per token (`Cache.bytesPerToken`)**.
- **Per-slot short-conv state (`f32`, `ring = 4`):**
  - `k_conv` (`48 * 4 * 1024 * 4` = `786,432` B) + `v_conv` (`786,432` B) + `attn_conv` (`48 * 4 * 2304 * 4` = `1,769,472` B) + `mlp_conv` (`1,769,472` B) = **`5,111,808` B (`4.88 MiB`) fixed**.
- **`Cache.fixedBytes` = `172,883,968` B (`164.88 MiB`).**

| Buffer / Context Configuration | 1,024 ctx | 4,096 ctx | 16,384 ctx | 32,768 ctx |
| --- | ---: | ---: | ---: | ---: |
| **1 Session `Cache`** (`fixedBytes + ctx * 32 KiB`) | `0.192 GiB` (`206.4 MB`) | `0.286 GiB` (`307.1 MB`) | `0.661 GiB` (`709.8 MB`) | **`1.161 GiB` (`1.247 GB`)** |
| **Rollback (`512` rows)** + **Fork Snapshots (`2 x 256` rows)** | `84.88 + 89.75 = 174.63 MiB` | `174.63 MiB` | `174.63 MiB` | `174.63 MiB` |
| **KV read per 1-token forward step** | `201.3 MB` (`0.386 ms`) | `302.0 MB` (`0.580 ms`) | `704.6 MB` (`1.352 ms`) | `1,241.5 MB` (`2.383 ms`) |
| **Runtime Scratch** | Plain `BatchScratch(cap=1)`: `47.51 MiB` (+ `PrefillScratch(1024,1)`: `686.55 MiB`) \| Spec `group.scratch(16)`: `72.07 MiB` + `Verifier`: `22.50 MiB` | — | — | — |

*(With `14.40 GiB` [`15.46 GB`] repacked text weights + `1.16 GiB` 32K KV + `0.17 GiB` rollback/fork + `0.67 GiB` prefill + `0.05 GiB` batch scratch = **`~16.45 GiB` (`17.66 GB`) plain ready** (`~16.9–17.0 GiB` / `18.2 GB` with drafter), giving **~8 GB headroom** below the `26 GB` load limit and `28 GB` `heavy.py` cap — vs `19.0–19.5 GB` ready / `21.9 GB` peak in SC2.)*

---

## 2. Complete Kernel & Dispatch Inventory

### 2.1 Reconciliation of Dispatch Counts per Step (`prefill.zig`, `group.zig`)

Unfused 1-row `batchStep` encodes **`1,302`** dispatches (unfused grouped verify with `.commit` encodes **`1,306`** = `1,302` forward + `4` finish). V4's `PrefillScratch.fuse = true` removes **`424` dispatches at 1 row** and **`472` dispatches at `>= 2` rows**:
- **Mixes + norms (`-192`):** `23` non-bank `attn_input` (`3 -> 1`), `24` bank `attn_input + messages` (`4 -> 1` via `i2pr_batch_mix2`), `24` non-bank `mlp_input` (`3 -> 1`), `24` bank `mlp_input` (`2 -> 1`), `1` head (`3 -> 1`) = `46 + 72 + 48 + 24 + 2 = 192`.
- **QKV prologue (`-144`):** `48` x (`conv_k`, `clip3`, `conv_v`, `store_rel`) -> `48` x `i2_batch_qkv` / `i2v_qkv` (`48 * 3 = 144`).
- **Attention residual (`-40`):** `40` non-push `i2_add` folded into `i2p_batch_conv` / `i2v_conv`.
- **Expert combine (`-48`):** `48` `i2p_combine` folded into `mlp_post` `conv` (`moeInConv()`).
- **Controllers at `n >= 2` (`-48`):** `24` attn3 (`3 -> 2`) + `24` mlp1 (`3 -> 2`) = `48` (reverted to main's 3-kernel at `n == 1`, `prefill.zig` L1209).
- **Totals:**
  - **Plain 1-row `batchStep` (V4):** **`878` dispatches** (`1,302 - 424`; `+4` `i2t_tap` = `882` when DFlash2 active).
  - **Grouped verify `8 / 16` rows (`groupFinish`):** **`834` dispatches** (`1,302 - 472 = 830` forward + `4` finish; `+4` `i2t_tap` = `838` when DFlash2 active).
  - **Prefill `1,024`-row chunk (`PH2` default `both`, `want = .none`/`.last`, `0` fork captures):** **`2,031` / `2,032` dispatches** (`+232` per fork capture point inside chunk; `+4` `i2t_tap` when drafter active).

### 2.2 Complete Active Kernel Table

*(All kernels satisfy `max_buffer_idx <= 20 <= 31`. Inline `setValue`/`setBytes` counts below must be packed into constant-buffer offsets for ICB / `MTL4ArgumentTable` execution.)*

| Phase / Kernel | Source `.metal` | Dispatch Count (`batch1` / `grp8` / `grp16` / `prefill1024`) | Dtypes (In / W / Out) | Grid x TG Size | TG Mem (B) | Dynamic Runtime Args vs Compile-Time Static Constants |
| --- | --- | --- | --- | --- | --- | --- |
| `i2a_embed` | `affine.metal` L82 | `1` / `1` / `1` / `1` | `u32`, aff8 -> `f32` | `(n,1,1) x 256` | `128` | **Dyn:** `ids`, `rows=n` \| **Stat:** `width=2304`, `eps=1e-6`, `embed` ptrs |
| `i2_copy` (prelude), `i2p_pass_copy` | `glue.metal` L423, `prefill.metal` L284 | `3` / `3` / `3` / `3` | `f32` -> `f32` | `((n*h\|h,n)) x 256` | `0` | **Dyn:** `rows=n` \| **Stat:** `width=2304`, `stride=5`, `start=2`, `count=2` |
| `i2pr_batch_mix`, `i2pr_batch_mix2` | `prefill_rows.metal` L249, L262 | `72`+`24` / `72`+`24` / `72`+`24` / `0` | `f32` -> `f32` | `(n, 1\|2, 1) x 256` | `~660` (`4*5*8`+`4*5`+`5` f32) | **Dyn:** `rows=n` (via grid.x) \| **Stat:** `width=2304`, `depth`, `k`, `norm`, `scale`, `eps`, query & norm ptrs |
| `i2pr_mix_norm` | `prefill_rows.metal` L20 | `0` / `0` / `0` / `119` | `f32` -> `f32` | `(n,1,1) x 256` | `~112` | **Dyn:** `rows=n` \| **Stat:** `width=2304`, `depth`, `k`, `norm`, `scale`, `eps`, query & norm ptrs |
| `i2p_gather_rms`, `i2p_mix` | `prefill.metal` L262, L305 | `0` / `0` / `0` / `2` (`head`) | `f32` -> `f32` | `(depth*n\|n,1,1) x 256` | `128` / `208` | **Dyn:** `rows=n` \| **Stat:** `width=2304`, `depth=5`, `k=1`, `output_query` |
| `i2_controls_norm`, `i2_controls_down`, `i2_controls_apply` | `attention.metal` L373, L396, L428 | `72` / `0` / `0` / `0` | `f32`, `f32` -> `f32` | `(6n\|8,n,3\|288,n,3) x 256` | `128` / `0` / `0` | **Dyn:** `rows=n` \| **Stat:** `h=2304`, `rank=64`, `xs`/`xb`, `cols=1`, controller weight ptrs |
| `i2p_rms_pair`, `i2a_float`, `i2a_batch_float_control` | `prefill.metal` L208, `affine.metal` L104, L188 | `72` (`24*3` mlp) / `0` / `0` / `0` | `f32`, `f32` -> `f32` | `(2n\|8\|288,n) x 256` | `128` / `0` / `0` | **Dyn:** `rows=n` \| **Stat:** `h=2304`, `k=64`, `cxs/cxb`, `mxs/mxb=4/3`, mlp ctl ptrs |
| `i2r_controls_down`, `i2_controls_apply_rows` | `rows.metal` L334, `attention.metal` L428 | `0` / `96` / `96` / `0` | `f32`, `f32` -> `f32` | `(8,n,1\|3) x 256`, `((h+8c-1)/8c,n,1\|3) x 256` | `72` / `0` | **Dyn:** `rows=n`, `cols=min(n,8)` \| **Stat:** `h=2304`, `rank=64`, `xs`/`xb`, controller weight ptrs |
| `i2p_rms_pair`, `i2af_gemm`, `i2af_gate_ctl` | `prefill.metal` L208, `float_mpp.metal` L7, L25 | `0` / `0` / `0` / `288` | `f32`, `f32` -> `f32` | `(2n\|2,ceil\|72,ceil) x 256/128` | `128`/`0`/`4096` | **Dyn:** `rows=n` \| **Stat:** `h=2304`, `rank=64`, `k=4608/64`, controller weight ptrs |
| `i2p_rms`, `i2p_rms3` | `prefill.metal` L185, L229 | `49` / `49` / `49` / `50` | `f32`, `bf16` -> `f32` | `(n\|3n,1,1) x 256` | `128` | **Dyn:** `rows=n` \| **Stat:** `width=2304`, `eps`, norm weight ptrs |
| `i2a_batch_projections` | `affine.metal` L260 | `48` / `0` / `0` / `0` | `f32`, aff8 -> `f32` | `((total+7)/8,n,1) x 256` | `0` | **Dyn:** `rows=n` \| **Stat:** `k=2304`, `full`, `m[8]` (`[3072,1024,1024,384,3072,24,0\|32,0\|32]`), proj ptrs |
| `i2t_projections_8_2`, `i2t_projections16_8_1` | `rows.metal` L218, L219 | `0` / `48` / `48` / `0` | `f32`, aff8 -> `f32` | `(sum(ceil),1,1) x 256` | `16,384` | **Dyn:** `rows=n` \| **Stat:** `k=2304`, `full`, `m[8]`, tile sizes `8x2` / `8x1`, proj ptrs |
| `i2as_prepare`, `i2as_split`, `i2ps_gemm_hi`, `i2ps_split_gated` | `affine_stage.metal`, `prefill_staged.metal` | `0` / `0` / `0` / `416` | aff8->`f16`, `f32`->`2xf16` -> `f32` | `64x64` MPP (`128` threads) | `~18K` / MPP | **Dyn:** `rows=n` \| **Stat:** `n`, `k`, `part[6]` tile offsets/row offsets, projection ptrs |
| `i2_batch_qkv` | `attention.metal` L527 | `48` / `0` / `0` / `0` | `f32`,`bf16` -> `f32`,`bf16` | `(8+24+(24*(ext+1)+255)/256,n) x 256` | `160` | **Dyn:** `pos=cache.len`, `rows[8]` GPU ptrs \| **Stat:** `kv=8`, `heads=24`, `dim=128`, `rel=16`, `ext=1024`, `layout/dyn/tau` |
| `i2v_qkv` | `verify.metal` L400 | `0` / `48` / `48` / `0` | `f32`,`bf16` -> `f32`,`bf16` | `(32+97,g.rows,1) x 256` | `160` | **Dyn:** `rows[16]` (tree `pos/back`), `g.rows` \| **Stat:** `kv,heads,dim,rel,ext`, weight ptrs, temp `v.kv/conv` |
| `i2p_conv`, `i2pr_ring_store`, `i2pr_clips`, `i2p_relbias_rows`, `i2pr_kv_store` | `prefill.metal`, `prefill_rows.metal` | `0` / `0` / `0` / `336` (`prefill`) | `f32`,`bf16` -> `f32`,`bf16` | `((ext+256)/256\|kvw/4, n) x 256` | `0` / `288` | **Dyn:** `base=cache.len`, `rows=n`, `tail` \| **Stat:** `channels`, `ring=4`, `ext=1024`, `window=1024`, `slotRing` |
| `i2_batch_split`, `i2_batch_combine` | `attention.metal` L266, L283 | `96` / `48` (`combine`) / `48` (`combine`) / `0` | `f32`,`bf16` -> `f32` | `(8*splits,n)x96`; `(24,n)x32` | `16,384` / `0` | **Dyn:** `pos`, `splits/chunk` (`attnPartition`), `rows[]` ptrs \| **Stat:** `heads=24,kv=8,dim=128,win,scale,floor,alpha` |
| `i2v_split` | `verify.metal` L246 | `0` / `48` / `48` / `0` | `f32`,`bf16` -> `f32` | `(8*max_splits,g.rows) x 96` | `16,384` | **Dyn:** `rows[16]` (`pos,splits,chunk,path,gpos`) \| **Stat:** same + `v.kv` scratch offsets per `slot` |
| `i2p_attn_units` | `prefill_expert_wide.metal` L302 | `0` / `0` / `0` / `48` | `f32`,`bf16` -> `f16`/`f32` | `(8,(n+31)/32,1) x 192` | `16,384`+ | **Dyn:** `base=cache.len`, `rows=n` \| **Stat:** `heads=24,kv=8,dim=128,ext,win,scale`, `cache.keys/vals` base |
| `i2_sig_mul` | `glue.metal` L385 | `48` / `48` / `48` / `0` | `f32` -> `f32` | `(3072*n,1,1) x 256` | `0` | **Dyn:** `count=3072*n` \| **Stat:** `s.attn`, `s.gate` scratch buffers |
| `i2a_dense` (`o_proj`, `head_1r`) | `affine.metal` L144 | `49` / `0` / `0` / `1` (`head_last`) | `f32`, aff8 -> `f32` | `((n+7)/8,rows) x 256` | `0` | **Dyn:** `rows=1\|n` \| **Stat:** `n=2304\|200058`, `k=3072\|2304`, `cap=0\|30`, weight/scale/bias ptrs |
| `i2t_dense_8_2`, `i2t_dense16_8_1` (`o_proj`, `head`) | `rows.metal` L212, L213 | `0` / `49` / `49` / `0` | `f32`, aff8 -> `f32` | `(ceil(n/16\|8),1,1) x 256` | `8K` / `16K` | **Dyn:** `rows=n` \| **Stat:** `n`, `k`, `cap`, weight ptrs (`rows=2` head uses `i2r_dense_ahead_2`) |
| `i2p_batch_conv`, `i2v_conv` (`attn_post`+`mlp_post`) | `prefill.metal` L453, `verify.metal` L15 | `96` / `96` / `96` / `0` | `f32`,`bf16` -> `f32` | `(2304,n,1) x 256` | `0` | **Dyn:** `rows[]` (`pos,back,ring`) \| **Stat:** `channels=2304,ring=4,residual,topk=0\|8`, conv weight ptrs |
| `i2p_conv`, `i2pr_conv_add`, `i2pr_ring_store` (`post`) | `prefill.metal`, `prefill_rows.metal` | `0` / `0` / `0` / `192` | `f32`,`bf16` -> `f32` | `(2304,n\|4) x 256` | `0` | **Dyn:** `base=cache.len`, `rows=n`, `tail` \| **Stat:** `channels=2304,ring=4,norm` |
| `i2p_scatter` (`ip % 6 == 0`) | `prefill.metal` L297 | `8` / `8` / `8` / `8` | `f32` -> `f32` | `(2304,n,1) x 256` | `0` | **Dyn:** `rows=n` \| **Stat:** `width=2304,stride=5,slot` (`0..3`) |
| `i2a_float` / `i2p_router_exact` + `i2p_routepick` | `affine.metal` L104, `prefill.metal` L60, L125 | `96` / `96` / `96` / `96` | `f32`,`f32` -> `i32`,`f32` | `(16,n)x256` + `(n,1)x32` | `0` / `0` | **Dyn:** `rows=n` \| **Stat:** `E=128,K=8,h=2304,route_scale=8.0`, router ptrs |
| `i2p_moe_{zero,count,offsets,sort}` + `i2p_half_rows` + `i2p_combine` | `prefill.metal`, `prefill_expert_wide.metal` | `0` / `0` / `0` / `288` | `i32`/`f32` -> `i32`/`f16`/`f32` | various (`32..256`) | `0`/`512` (`counts[128]`) | **Dyn:** `rows=n,slots=8n` \| **Stat:** `E=128,K=8,h=2304` |
| `i2p_gu` (`1r`), `i2e_down1` (`1r`) | `prefill.metal` L957, `experts.metal` L597 | `88` / `0` / `0` / `0` | `f32`, NVFP4 -> `f32` | `(1152,8)x32`; `(72,8)x256` | `0` / `9,216` (`16+256+SG*T*32`) | **Dyn:** `PICKS[8]` \| **Stat:** `k=2304/1152,n=1152/2304,slots=8,caps`, layer expert ptrs |
| `i2e_gate_up{,16}`, `i2e_down{,16}` (`grp8/16`) | `experts.metal` L595–599 | `0` / `88` / `88` / `0` | `f32`, NVFP4 -> `f32` | `(72\|90,8n)x128\|256` | `9,216`–`33,792` | **Dyn:** `slots=8n,PICKS` \| **Stat:** `k,n,topk=8,caps`, layer expert ptrs |
| `i2p_gu_grouped` + `i2p_expert_units_up`, `i2p_down_grouped` + `i2p_expert_units_down` | `prefill.metal`, `prefill_expert_wide.metal` L460 | `0` / `0` / `0` / `176` | `f16`/`f32`, NVFP4 -> `f16`/`f32` | `(36,tiles)x128` (`units`), `(inter,128)x32` | `9,216` (`64*72` `f16`) | **Dyn:** `rows=n,slots=8n,moe_off,moe_sorted` \| **Stat:** `k,n,E=128,caps`, expert ptrs |
| `i2a_expert_up`, `i2a_expert_down` (layers `20..23`) | `affine.metal` L403, L423 | `8` / `8` / `8` / `0` | `f32`, aff4 -> `f32` | `((n+7)/8,8n) x 256` | `0` | **Dyn:** `slots=8n,PICKS` \| **Stat:** `k,n,topk=8,caps`, expert ptrs |
| `i2p_affine_units_up`, `i2p_affine_units_down` (layers `20..23`) | `prefill_expert_wide.metal` L462–463 | `0` / `0` / `0` / `8` | `f16`, aff4 -> `f16`/`f32` | `(36,tiles) x 128` | `9,216` (`64*72` `f16`) | **Dyn:** `rows=n,slots=8n,moe_off,moe_sorted` \| **Stat:** `k,n,E=128,caps`, expert ptrs |
| `i2v_{argmax,accept,round_kv,round_conv}` | `verify.metal` L300, L343, L371, L385 | `0` / `4` / `4` / `0` | `f32`/`u32` -> `u32`/`bf16` | `(g.rows)x1024`, `(1)x32`, `...` | `256` / `0` | **Dyn:** tree `streams/parents/rows`, `pos` \| **Stat:** `vocab=200058`, `slots=48`, `kvw,h,win,ring` |
| `i2t_tap` (when DFlash2 taps enabled) | `taps.metal` L7 | `+4` if drafter loaded | `f32` -> `bf16` | `(2304,n,1) x 256` | `0` | **Dyn:** `rows=n` \| **Stat:** `visits=[5,23,41,48]`, `width=2304,stride=9216` |

- **DFlash2 Drafter Dispatches per Round (`qwen27/dflash/execution.zig`):**
  - `encodeAppendTo` (absorbing fresh `L` tap rows): `fusion` proj + `hidden_norm` + `5` layers x (`k` proj + `k_norm` + `rope` + `v` proj + `context_append`) = **`27` op callbacks**.
  - `encodeStreams` (`1` stream, `block = 8` rows): `embed` + `embed_norm` + `5` layers x (`21` ops: `input_norm`, `attn_conv_dyn`, `conv`, `q/k/v` projs, `q/k` norms, `q/k` ropes, `attention`, `o`, `conv`, `post_norm`, `mlp_conv_dyn`, `conv`, `gate/up`, `swiglu`, `down`, `conv`) + `final norm` + `head` + `hidden_projection` + `bf16_topk` = **`111` op callbacks** (each `project` uses `core.affine4_lane.Adapter` + `core.draft_ops` + `core.shared_attention`).

---

## 3. Startup, Pipeline & Per-Step Submission Inventory

### 3.1 Cold Load & Initialization Today (`inkling2_loop.zig`, `model.zig`, `inkling2_loop_host.zig`)

| Startup Phase | Current Implementation | Exact Cost / Overhead | Offline AOT Target (**G2**) |
| --- | --- | --- | --- |
| 1. Checkpoint parsing & shard copy | `Model.load` -> `readConfig` + `4` `mapShard` calls (`model.zig` L235–255): opens mmap, parses JSON header (`1,098` entries), allocates `16.21 GB` `MTLBuffer`s and `@memcpy`s **all 4 shards (including `636.5 MB` unused vision weights)**, then closes mmaps. `Model.mapMlxWeights` allocates **144** separate `ownedTensor` GPU buffers (`mapMlxControllers`, L453–477) and CPU-transposes `4 * 3 * 12` `f32` matrices, plus `2` `cross_pass` buffers (`L372–383`) and **20** `den = 2688.0` buffers (`names.zig` L223). | Copies `16.21 GB` on CPU + wastes `636.5 MB` vision + `~113.7 MB` raw controller copies + CPU transpose loops + `170+` tiny `MTLBuffer` objects. | Offline `repack` tool emits single text-only `weights.aot` (`~15.46 GB`) with pre-transposed controllers, pre-split `cross_pass`, stripped vision; zero JSON/safetensors parse, zero CPU transpose. |
| 2. Target MSL JIT & PSO build | **`25` `mtl.Library.fromSource` + `127` `mtl.Pipeline.init(..., false)` static sites** in `inkling2_loop`: `Model.load` compiles `glue` (29 PSOs), `experts` NVFP4+MXFP8 (4 PSOs), `affine` (12 PSOs). `openHost` then constructs **TWO** `PrefillScratch` objects (`BatchScratch.init` + `PrefillScratch.initRows`, `inkling2_loop_host.zig` L701, L705) — **compiling the exact same 10 prefill MSL libraries twice (~44 PSOs x2) and `attention.metal` twice (`BatchScratch` + `state.Scratch`), plus `rows.Pipes.init` twice (`i2t_probe` GPU test run twice!)**; if drafter loaded, adds `Verifier.init` (`10` PSOs) + `2x` `Taps.init` (`taps.metal`). | Heavy runtime `newLibraryWithSource` + AIR->`applegpu_g17s` PSO JIT compilation, duplicated across scratch objects, plus runtime `i2t_probe` command buffer. | Compile all Inkling `.metal` shaders offline (`metal` -> `kernels.metallib` for `applegpu_g17s`) + pre-bake `MTLBinaryArchive` / `MTL4Archive` + specialize `MTLFunctionConstantValues` + emit `i2t_probe` as a startup invariant check or build-time device attestation. Dedup shared pipeline cache! |
| 3. Scratch & KV alloc + zeroing + warm-up | `PrefillScratch.init` runs `commit()` (`prefill.zig` L715–730) blitting zeros over `45+` buffers; `Host.openHost` allocates `Slot` KV/conv/rollback/fork buffers; **`Host.warm()`** (`inkling2_loop_host.zig` L638–653) runs a full 1-token `prefillWant(..., .none)` + blit zero-fill (measured at `0.75–1.81 s` added to load in `inkling-p5c-chunks.md` L29–30 to absorb first-touch GPU page wiring off the first request!). | `0.75–1.81 s` load warm-up + `PrefillScratch` blit; `BatchScratch(cap=1)` also wastes `40.5 MiB` on unused `affine_stage_w` + `5.38 MiB` dead `state.Scratch`. | Eliminate duplicate/dead scratch allocs; replace first-touch JIT + driver lazy compile via `MTLBinaryArchive` / `MTLResidencySet.requestResidency()` + single lightweight residency warm touch. |
| 4. DFlash2 Drafter Load (`Draft.open`) | Runtime MSL compiles across 7 modules (`glue_gpu` x2, `draft_ops`, `shared_attention`, `bf16_topk_gpu`, `lane_projection` (`tf_lane_order` + per-shape `tf_lane`), `quant_gpu`, `row_projection`), CPU-quantizes `81` BF16 tensors to `q4` + runs GPU `RegOrder` (`tf_lane_order`) packing (`runtime_backend.zig` L115–148), then runs `3` full proposal warm-ups (`Draft.warm`, `inkling2_loop_spec.zig` L338–349). | Several hundred ms–1 s of JIT compile + q4 quantization + GPU reorder + 3 warm proposals. | Phase F: precompile drafter `.metallib` / archive + save pre-packed `affine4_lane` drafter `weights.aot` sidecar! |

### 3.2 Per-Step Host Submission Today (`prefill.zig`, `inkling2_loop_host.zig`)

1. **Queue & Command Buffers:** `Model.load` creates one Metal 3 serial queue (`device.queue()`, `model.zig` L179). `prefill.zig`'s `run()` opens serial encoders (`cb.compute(.serial)`) inside `mtl.objc.Pool.push()/pop()`.
   - **`batchStep` / `groupStep`:** flushes `1` early command buffer after slot `5` (`ProfileRounds.flush` L1733–1742), then commits the remaining `42` slots + head (+ verifier finish) in a second command buffer = **`2` command buffers/step**, ending with synchronous `rounds.cb.wait()`.
   - **`prefillWant` (1,024-row chunk):** flushes `7` early CBs (after slots `5, 11, 17, 29, 35, 41, 47`) + `1` final CB = **`8` command buffers/chunk**.
2. **Per-Dispatch Host Cost (`878` times in `batch1`):** `setP()` executes `setPipeline` + `setValue`/`setBytes` inline memcpy (`~878+` Objective-C `objc_msgSend` calls) + dynamic `useResource:` in `i2_batch_qkv` / `batchAttention` / `group.zig` (since target buffers have no `MTLResidencySet`) + per-row struct construction on CPU (`BatchRow`, `BatchStoreRow`, `BatchQkvRow`, `GroupRow`, `ConvRow`, `CommitRow`).
3. **Plain Decode Host Tail (`Host.run`, `inkling2_loop_host.zig` L621–634):** Waits for `rounds.cb.wait()`, reads `200,058` `f32` logits (`800.2 KB`) via `logits.slice(f32, vocab)`, runs scalar CPU `argmax()`, appends token, checks stop/EOS/cancel on CPU, and re-enters `batchStep`. Only `Verifier.encodeFinish` (`group.zig` L162–167) runs GPU `i2v_argmax`.

---

## 4. Phase B–F AOT Specialization & Artifact Blueprint

### 4.1 G0 Native Artifact Layout & Fail-Closed Contract

The offline compiler (`tf-inkling2-aotc`) consumes a validated checkpoint directory (`config.json`, `model-0000*.safetensors`, optionally DFlash2) and emits a self-contained artifact bundle `inkling2-mini-oq4e.aot/` requiring **zero checkpoint dir, zero Python, and zero Metal runtime compiler at serve time**:

```text
inkling2-mini-oq4e.aot/
  tf-inkling2-mini-aot      # Specialized Mach-O arm64 binary (linked without Metal runtime compiler calls)
  manifest.json             # Immutable schema: model_sha256, schedule_sha256, metallib_sha256,
                            #   archive_sha256, target_arch="applegpu_g17s", os_sdk="27.2",
                            #   context_cap=32768, quant_contract, file sizes
  kernels.metallib          # Offline `metal` + `metallib` AIR/bitcode compiled for air64-apple-darwin27.2.0
                            #   with [[function_constant]] specialized shapes
  pipelines.mtl4archive     # (or MTLBinaryArchive) Pre-compiled `applegpu_g17s` ISA PSOs (`indirect=true`)
  schedule.bin              # Immutable flat binary plan: 48-slot table (slot->layer, full, push, bank,
                            #   precision, static buffer base offsets, dispatch templates, const arena image)
  weights.aot               # Contiguous page-aligned repacked text-only sidecar (~15.46 GB, zero vision):
                            #   single weight buffer(s) <= maxBufferLength (28.08 GiB)
  [drafter.aot]             # Optional Phase F DFlash2 pre-packed q4 weights + metadata
```

**Fail-Closed Startup Validation & `heavy.py` Memory Rule:**
1. Verify `device.name` / `device.architecture` (`generation() == 17` / `applegpu_g17s`), `hasMetal4()`, `recommendedMaxWorkingSetSize`, and `supportsFamily.apple10` BEFORE loading weights; exit non-zero with explicit diagnostic on mismatch.
2. Verify SHA-256 of `schedule.bin`, `kernels.metallib`, `pipelines.mtl4archive`, and `weights.aot` header/chunk checksums against `manifest.json` before execution; fail closed on any corruption or truncation.
3. **Critical `heavy.py` / `28 GB` Cap Loading Rule (`campaign/heavy.py` L65–96, `campaign/ledger.md` B2):** Never bind clean `MAP_SHARED` weight pages directly with `newBufferWithBytesNoCopy` if macOS/Metal wires them *and* leaves them resident in `rusage.ri_resident_size` (causing double charge). Instead, stream `weights.aot` with `fcntl(F_NOCACHE)` (`pread` in chunks) directly into a unified `MTLResourceStorageModeShared` buffer (or `madvise(MADV_DONTNEED)` file-mapped pages immediately after copy into a single shared `MTLBuffer` added to an `MTLResidencySet`), measuring exact `heavy.py` peak charged RSS at `<= 19 GB` plain / `<= 20.5 GB` with drafter (`<< 26 GB` load limit and `28 GB` cap).

### 4.2 Ranked Compile-Time Specialization & Bit-Exact Phase C Repacking Targets

| Rank | Specialization / Repack Target | Exact Savings / Mechanical Effect | Bit-Exactness & Memory Constraints |
| --- | --- | --- | --- |
| **C1** | **Strip `636.5 MB` vision weights + eliminate `113.7 MB` duplicate controller tensors + `144` tiny `ownedTensor` buffers** | Drops resident weight footprint from `16.21 GB -> ~15.46 GB` (`-750 MB`), eliminates `4` shard copies + `144` CPU matrix transposes + `20` `den` buffers (`2688.0f`). Enables `1` static weight `MTLBuffer` (`15.46 GB < 28.08 GB maxBufferLength`)! | 100% bit-exact (`loop_down`/`gain`/`forget` & `cross_pass` values identical). **Zero risk.** |
| **C2** | **Contiguous weight arena + `MTLResidencySet` + eliminate dead `BatchScratch` buffers (`40.5 MiB affine_stage_w`, `5.38 MiB` `state.Scratch`)** | Removes all `useResource:` calls (`prefill.zig`, `group.zig`), removes `~46 MiB` dead scratch, locks all weight/cache/arena pages resident via `MTLResidencySet.requestResidency()`. | 100% bit-exact. Must verify `ResidencySet.withinUnder` respects `26 GB` load limit. |
| **C3** | **Interleave Affine-8 & Affine-4 `scales` + `biases` (`bf16x2` packed pairs, padded/aligned per row)** | Halves scale/bias buffer pointers (`6 -> 3` in `i2a_batch_projections`!), turns `s[group]` + `b[group]` in `i2a_dot` (`affine.metal` L35–36) into 1 coalesced `uint` load (`32`-bit aligned), cutting TLB/cacheline misses across `489.7 MB lm_head` + `1.37 GB attn/oproj` + `143 MB affine4`. | **Bit-exact:** `float(s) * code + float(b)` unchanged; only storage order of `(s, b)` pairs changes. |
| **C4** | **Compile-time `[[function_constant]]` geometry baking + pre-divided NVFP4 `global_scale = amax[e] / 2688.0f` (`10 KB -> 10 KB`)** | Replaces `a.k=2304`, `a.n`, `a.bits=8`, `heads=24`, `kv=8`, `dim=128`, `d_rel=16`, `extent=1024`, `E=128`, `K=8`, `eps`, `caps` struct reads with compile-time constants in `.metallib` (`k` loops completely unrollable!); eliminates `den` pointer indirection. | **Verify Decision 18:** confirm pre-dividing `amax[e] / 2688.0f` at offline repack time produces identical `f32` IEEE-754 bits to `amax[e] / den[0]` on GPU for all `20 * 2 * 128` entries (it is plain f32 division on identical inputs!). |
| **C5** | **Contiguous `lm_head` trimmed to `200,058` rows + per-expert cacheline-aligned NVFP4 `(W, S)` pairing** | Saves `~2.3 MB` unused padding on `lm_head` + co-locates each expert's `w13` (`2.99 MB`) and `w2` (`1.49 MB`) weights + FP8 scales in adjacent pages so `i2p_gu` / `i2e_down1` prefetch without cross-tensor TLB jumps. | 100% bit-exact. Measure at 1r and 8r/16r (`i2e_gate_up`/`i2e_down`). |

### 4.3 Ranked Phase D Generated Metal Fusion Targets (No Global Barriers, Decision 18 Exact)

| Rank | Fusion Target | Dispatches Saved (1r) | Est. Savings | TG Memory & Register / Occupancy Notes | Decision 18 Bit-Exact Verification Plan |
| --- | --- | ---: | ---: | --- | --- |
| **D1** | **`i2_batch_combine + i2_sig_mul`** (or `sig_mul` into `i2_batch_combine` writeout) | **`-48`** (`48 -> 0` `sig_mul`) | `~0.20–0.30 ms` | Zero extra TG mem (`24` TGs x `32` threads already write `O[h*128]` in `i2_batch_combine`); reads `gate[h*128]` once in registers, eliminates `3072*4B` DRAM write+read & 48 barriers! Safe for 1..16 rows too (`group.zig`!). | Compare fused `s.attn` against `i2_batch_combine` + `i2_sig_mul` bit-for-bit across `1..16` rows and `1..16` attention splits. |
| **D2** | **`i2a_float (router) + i2p_routepick`** (per-row single-kernel or 2-warp cooperative top-8) | **`-48`** (`96 -> 48`) | `~0.25–0.35 ms` | Watch **occupancy**: `i2a_float` uses `16` TGs x `256` threads (`128` cols = 1 simdgroup/expert) for high `h=2304` BW; `i2p_routepick` needs reduction over `128` logits (`1` TG of `128` warps/4 SGs = `128*4=512 B` TG mem). **Risk:** 1 TG computing all `128 * 2304 = 1.18 MB` router matvec serially might lose BW vs `16` TGs (`i2a_float`), so benchmark `1` TG of `1024` threads (`4` SGs x `4` experts/warp? Or `32` SGs = 1024 threads, 4 cols/warp) against `96 -> 48` first! | Bit-exact `picks[8]` and `wts[8]` on all `48` slots across random & real router logits + tie cases. |
| **D3** | **1-row Controllers + Post-Norms (`144` ctl + `48` bank `rms`/`rms3` -> `~48–96`)** | **`-96..144`** | `~0.40–0.65 ms` | **Biggest single dispatch pool (`192` of `878` dispatches in V4: `24` `i2_controls_norm` + `24` `i2_controls_down` + `24` `i2_controls_apply` + `24` `i2p_rms_pair` + `24` `i2a_float` + `24` `i2a_batch_float_control` + `24` `i2p_rms3` + `24` `i2p_rms`)!** Note `i2_controls_down`/`apply` already launch `grid.z=3` for the 3 attn controllers; the waste is `48` standalone norm dispatches (`i2_controls_norm` + `i2p_rms_pair`) + `48` standalone post-RMS dispatches (`i2p_rms3` + `i2p_rms`) + separate `down`/`apply` passes. Two safe sub-fusions: (a) emit `cur`/`msg` `inv` RMS scalars right out of the preceding `mix` kernel (`-48` norm dispatches, ~64B TG, zero extra DRAM scan); (b) fuse `ctl_down` (`rank=64` fits in 1–2 TG barriers? Check `down` `8`-TG parallelism vs 1-TG!) or fold post-`rms` (`inv`) into `apply` prelude (`-48` dispatches). | Compare `n0, n1, n2, m0` bit-for-bit against unfused 3-kernel controller + `rms3`/`rms` (`0394b8734` bits) across all `24` bank slots. (Abort if `down` loses to separate `i2a_float`.) |
| **D4** | **End-of-layer `mlp_post` conv + next-slot `mix_norm` / `scatter` fold** | **`-40..80`** | `~0.25–0.45 ms` | After `mlp_post` residual `x += conv(moe)` (writes `2304` floats across `9` TGs), next slot runs `i2pr_batch_mix` (`1` TG of `256` threads reading `blocks[0..3]` + `x`). Since `mlp_post` only needs `conv_ring` (`4 * 2304 = 36 KB`), `1` TG (`256` or `512` threads) computing `conv_add` + `mix_norm` directly in registers for `1`-row non-bank slots avoids `x` roundtrip AND eliminates `1` TG launch barrier (`48 -> 24`? Wait: `1` TG computing `2304 * 4` conv + `mix` is tiny compute, ~`36 KB` read, **check 1-TG vs 9-TG latency carefully!**). Separately, folding `i2p_scatter` (`8` dispatches) into `conv` on `ip % 6 == 0` is trivial (`0` extra TG mem, saves `8` dispatches). | Bit-exact `s.x`, `s.blocks`, `s.n0..2`, `s.m0` against V4 unfused/fused across all `48` slots. |
| **D5** | **`k`-specialized `i2a_dense` (`o_proj`, `lm_head`), `i2p_gu`, `i2e_down1`** | `0` (tail trim) | `~0.35–0.55 ms` | Bake `k=3072` (`o_proj`), `k=2304` (`gu`, `lm_head`), `k=1152` (`down1`) via `[[function_constant]]`, pre-interleave `(s,b)` (`C3`), eliminate `if (col >= a.n)` / remainder loops & `den` indirection, and tune TG simdgroup count (`256` vs `128` threads) to cut the `~0.22–0.25 ms` per-class overhead above DRAM bytes on `o_proj`/`gu`/`down1`. | Bit-exact per-kernel byte tests (`rows_test`, `dense_test`, `batch_test`). |

### 4.4 Recommended Implementation Order & Go/No-Go Checkpoints (Phases A–F)

1. **Phase A (Baseline Control — Prerequisite: Q4 quiet-host scorecard):**
   - Pin `main` (`eb18fe252`) baseline metrics: dispatch counts (`878` 1r, `834` grp8/16, `2,032` prefill1024), CB counts (`2` / `8`), `phases`, `roundtable`, cold load time, pipeline compile time, and `heavy.py` footprint. Verify Q4 tasks/serve parity cleanly on quiet host.
   - **Go:** Reproducible baseline + automated bit-exact differential harness (`batch_test` + `dflash-widths hashes` 228/228 + `oMLX` oracle).
2. **Phase B (G0 Scaffold + Static Schedule with Today's Kernels):**
   - Implement `tf-inkling2-aotc` offline shader compiler (`kernels.metallib` targeting `applegpu_g17s` + `MTLBinaryArchive` / `MTL4Archive`) + `manifest.json` fail-closed verifier + immutable `schedule.bin` (flat `878`-dispatch list with parameter arena replacing `setBytes`/`setValue`, indirect `MTLIndirectCommandBuffer` / `MTL4ArgumentTable` + `MTLResidencySet`).
   - **Go/No-Go:** **G0 & G2 (zero runtime MSL JIT)** + **G1 (228/228 hashes bit-exact)**; record honest Phase B plain decode delta (expected `~1–4%`). Do **not** stall waiting for `1.15x` here.
3. **Phase C (Static Memory Layout + Bit-Exact Repacking):**
   - Implement `weights.aot` packer (strip `636.5 MB` vision, pre-transpose 144 controllers, `den` pre-divide if bitwise identical, interleave `(scale,bias)` `bf16x2`, co-locate per-expert NVFP4 `W+S`, contiguous `lm_head` `200,058` rows), drop duplicate `PrefillScratch` & unused `state.Scratch`/`affine_stage_w` in `BatchScratch`, single-charge load under `heavy.py`.
   - **Go/No-Go:** **G2 load time cut `>=50%`**, **G4 memory (`<=19 GB` plain, `<<26 GB` limit, zero mmap double-charge)**, **G1 bit-exact**. Kill any single repack that regresses `lm_head`/`attn_proj`/`NVFP4` microbenchmarks.
4. **Phase D (Generated Metal Fusion — Critical Path to G3):**
   - Implement fusions strictly in order **`D1` (`combine+sig_mul`) -> `D5` (`k`-specialized + `(s,b)` kernels) -> `D3` (1r `mix+rms_pair` scalar save + 3-in-1 `ctl_down`/`apply`) -> `D2` (`router+pick`) -> `D4` (`scatter`/`conv+mix`)**. Measure each independently at 1K, 4K, and 32K contexts (`phases` + `roundtable`).
   - **Go/No-Go:** Keep ONLY fusions that strictly beat baseline at **both 1K and 4K/32K** and pass `228/228` bit-exact hashes; park any fusion failing 3 variants. Gate target entering Phase E: **`>=1.11x` plain decode over `main`**.
5. **Phase E (GPU-Resident Bounded/Cancellable Token Loop):**
   - Replace per-token host `logits` read + CPU `argmax` with GPU `i2v_argmax` writing directly into `ids` ring, execute bounded multi-token bursts (`N=8..16` tokens) via `MTL4CommandBuffer` / ICB parameter-update + indirect dispatch (`max_splits`) with `MTLSharedEvent` per-token streaming and host cancellation / EOS / `max_tokens` stop signal.
   - **Go/No-Go (Clears G3):** Paired plain decode geomean **`>=1.15x`** over `main` (95% CI excluding `1.0`), no 16K/32K regression, cancellation/EOS/stop semantic tests passing, `G1` & `G4` green.
6. **Phase F (Prefill + Few-Row/Tree Verify + DFlash2 Reuse — G5 Stretch):**
   - Port prefill `2,032`-dispatch chunk schedule + `P7` selective drafter tap capture/absorb (`2,047`-row window + restore points only) into AOT binary; extend Phase C/D (`D1`, `D5`, `group.zig` `i2v_qkv` + `(s,b)` packed projections) to 8/16-row verify; precompile DFlash2 `.metallib` + pack `q4` drafter sidecar.
   - **Go/No-Go:** Verify `2,000 tok/s` prefill at 1K–8K with drafter on (**G5 prefill**), check `Q4` full task parity (`ar == spec`), and measure honest speculative tok/s ceiling (**G5 decode**). Run final adversarial audit (`A1` step 9) against all G0–G5 gates.
